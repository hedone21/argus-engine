//! The rank-`r` truncation of the output projection, computed without a linear-algebra library.
//!
//! ## Why not a randomized or Krylov method
//!
//! Because the spectrum of `W_o` is *flat* where the metric cuts it. Measured across six models at
//! `r = d/128`, `σ_{r+1}/σ_r` is 0.90 to 0.998 (median 0.995) and the top `r` directions hold only
//! 4–17 % of the Frobenius energy. Every method whose convergence rate is governed by the `r`/`r+1`
//! gap therefore does not converge: a randomized range finder with three-fold oversampling and four
//! power iterations lands 18–88 % away from the true truncation, per cell.
//!
//! What *does* work is raising the **block**, not the iteration count. Under subspace iteration
//! direction `i` converges as `(λ_{k+1}/λ_i)^t`, so a block of `k = 8r` buys a real gap
//! (`σ_{8r}/σ_r ≈ 0.70`) even though the `r`/`r+1` gap is nil. At `k = 8r` and ~20 iterations the
//! truncation operator is reproduced to `1e-7`, which is a hundred times inside the tolerance the
//! reference's own float32 arithmetic already imposes.
//!
//! ## What is actually being computed
//!
//! Not the singular vectors as such. The measurement depends on `W_o` only through
//! `M_r = V_r Σ_r² V_rᵀ`, the rank-`r` truncation of the Gram — and that is far better conditioned
//! than the vectors are: near-degenerate directions may mix freely, and `M_r` barely moves, because
//! the mixing is weighted by the gap it crosses. So the target is `M_r`, the convergence criterion
//! is stated on `M_r`'s trace, and the gates below check `M_r`'s defining properties rather than
//! any individual vector.
//!
//! The Gram is never formed. `G Q = W_oᵀ (W_o Q)` costs the same as one dense multiply by an
//! explicit `G` and skips a `d × d` f64 intermediate — 134 MB at the largest model in play — and
//! the Rayleigh–Ritz projection `QᵀGQ = (W_oQ)ᵀ(W_oQ)` then comes out of a product already
//! computed.
//!
//! ## f64, not f32
//!
//! In f32 the iteration plateaus at `1e-6` and then wanders non-monotonically, which leaves the
//! convergence criterion reading a negative increment and unable to certify anything at all. The
//! decomposition runs in f64; only the stored basis is narrowed to f32, which costs `5e-8` per cell.

use rayon::prelude::*;

/// How a decomposition went, and the evidence for it.
#[derive(Clone, Debug, PartialEq)]
pub struct FactorReport {
    /// Iterations actually run.
    pub iters: usize,
    /// The last relative gain in `tr(M_r)`. The stopping signal.
    pub energy_gain: f64,
    /// `max_i ‖G v_i − σ_i² v_i‖ / σ_1²` — the residual of the eigenproblem.
    ///
    /// The workhorse gate: measured to track the error in `M_r` within a factor of 2.5 for every
    /// method and model tried. It certifies that the returned subspace is *invariant*, which is not
    /// the same as certifying that it is the *top* one — see [`Self::deflation_ratio`].
    pub residual: f64,
    /// `‖V_rᵀ V_r − I‖_max`.
    pub orthogonality: f64,
    /// `max_i |v_iᵀ G v_i − σ_i²| / σ_1²` — consistency of the returned values with the vectors.
    pub ritz: f64,
    /// The largest singular value found in the orthogonal complement of `V_r`, over `σ_r`.
    ///
    /// The only check here that can catch a subspace that converged to the *wrong* invariant
    /// subspace — a residual gate cannot, because such a subspace has zero residual. It is a lower
    /// bound, so it can disprove and never prove, and its margin is only `(σ_r − σ_{r+1})/σ_r`,
    /// measured at 0.2 %–10 %. It detects a gross substitution, not a small error.
    pub deflation_ratio: f64,
}

/// Why a decomposition cannot be trusted.
#[derive(Debug, Clone, PartialEq)]
pub enum FactorError {
    /// The iteration hit its cap without the energy settling.
    NotConverged { iters: usize, energy_gain: f64 },
    /// A gate failed. The report says which quantity and what it read.
    Gate {
        gate: &'static str,
        got: f64,
        limit: f64,
    },
    /// The geometry is impossible.
    Shape(&'static str),
    /// The weight handed in was not finite. Reported separately so a bad input is not mistaken for
    /// a bad iteration.
    NonFiniteWeight { index: usize, value: f64 },
    /// The Ritz values came out non-finite even though the weight was. That is an arithmetic
    /// failure inside the iteration, not a convergence question.
    NonFiniteRitz { iter: usize },
}

impl std::fmt::Display for FactorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConverged { iters, energy_gain } => write!(
                f,
                "aperturb: the output-projection subspace did not settle in {iters} iterations \
                 (last relative energy gain {energy_gain:e}) — raise the block or the cap rather \
                 than accepting the factors"
            ),
            Self::Gate { gate, got, limit } => write!(
                f,
                "aperturb: output-projection gate {gate} read {got:e}, limit {limit:e}"
            ),
            Self::Shape(m) => write!(f, "aperturb: {m}"),
            Self::NonFiniteWeight { index, value } => write!(
                f,
                "aperturb: the output projection is not finite (element {index} is {value}) — the \
                 factorization has nothing to converge to"
            ),
            Self::NonFiniteRitz { iter } => write!(
                f,
                "aperturb: the output-projection Ritz values went non-finite at iteration {iter}"
            ),
        }
    }
}

impl std::error::Error for FactorError {}

/// Tuning for one decomposition. The defaults are the measured operating point.
#[derive(Clone, Copy, Debug)]
pub struct FactorConfig {
    /// Block size as a multiple of `r`. Eight is where the measured gap `σ_{8r}/σ_r ≈ 0.70` makes
    /// the iteration converge in ~20 steps; smaller blocks do not converge at this rank.
    pub block_mult: usize,
    /// Stop once the relative gain in `tr(M_r)` stays at or below this twice running.
    /// `1e-12` corresponds to a measured `M_r` error near `6e-7`.
    pub energy_tol: f64,
    /// Hard cap. Reaching it is a failure, not a result.
    pub max_iters: usize,
    /// Seed for the starting block. Fixed by default: the factors are a model constant and must not
    /// change run to run.
    pub seed: u64,
    /// `residual` limit. `1e-6` leaves roughly four times the margin the parity bar needs.
    pub residual_limit: f64,
}

impl Default for FactorConfig {
    fn default() -> Self {
        Self {
            block_mult: 8,
            energy_tol: 1e-12,
            max_iters: 48,
            seed: 0x5157_4346_5744_0001, // "QCFWD" || 1
            residual_limit: 1e-6,
        }
    }
}

/// SplitMix64 — a counter-based generator, so the starting block depends on `(layer, row, column)`
/// and on nothing else: not on the iteration order, not on the block size, not on the platform.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn unit(x: u64) -> f64 {
    // 53 bits into [2^-53, 1) — never exactly zero, so the log below is finite.
    (((x >> 11) as f64) + 0.5) * (1.0 / 9007199254740992.0)
}

/// A standard normal from two counter draws (Box–Muller). Reproducible everywhere.
fn gaussian(seed: u64, i: u64) -> f64 {
    let a = splitmix64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let b = splitmix64(seed ^ i.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0x1234_5678_9ABC_DEF0);
    let (u1, u2) = (unit(a), unit(b));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Orthonormalize the rows of `qt` (`[k][d]`) in place — modified Gram–Schmidt with one
/// reorthogonalization pass, which is what makes it as accurate as Householder at this shape.
///
/// A row that collapses is **refilled from the sketch and re-orthogonalized immediately**, not
/// deferred. `G Q` is rank-deficient whenever the weight is, and a block that quietly kept a
/// dependent row would hand Rayleigh–Ritz a non-orthonormal basis and produce eigenvalues that look
/// plausible and are wrong. `k <= d` is a precondition, so a replacement always succeeds.
///
/// Returns the number of orthonormal rows produced — `k` unless the precondition was violated.
fn orthonormalize_rows(qt: &mut [f64], k: usize, d: usize, seed: u64) -> usize {
    let mut kept = 0usize;
    for j in 0..k {
        let mut ok = false;
        for attempt in 0u64..4 {
            for _pass in 0..2 {
                for i in 0..kept {
                    let (head, tail) = qt.split_at_mut(j * d);
                    let qi = &head[i * d..(i + 1) * d];
                    let qj = &mut tail[..d];
                    let dot: f64 = qi.iter().zip(qj.iter()).map(|(a, b)| a * b).sum();
                    for (a, b) in qj.iter_mut().zip(qi.iter()) {
                        *a -= dot * b;
                    }
                }
            }
            let n: f64 = qt[j * d..(j + 1) * d]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            if n > 1e-12 {
                let inv = 1.0 / n;
                for x in &mut qt[j * d..(j + 1) * d] {
                    *x *= inv;
                }
                ok = true;
                break;
            }
            for (a, x) in qt[j * d..(j + 1) * d].iter_mut().enumerate() {
                *x = gaussian(
                    seed ^ 0xDEAD_BEEF_u64.wrapping_add(attempt),
                    (j * d + a) as u64,
                );
            }
        }
        if ok {
            if kept != j {
                qt.copy_within(j * d..(j + 1) * d, kept * d);
            }
            kept += 1;
        }
    }
    kept
}

/// Symmetric eigendecomposition of a small dense `[k][k]` matrix by cyclic Jacobi.
///
/// Returns `(eigenvalues, eigenvectors)` with the values descending and eigenvector `m` in column
/// `m` of the `[k][k]` row-major result. `k` is the block size — at most a few hundred — so the
/// cubic cost here is a fraction of a percent of the iteration it serves, and Jacobi's accuracy on
/// a positive-definite matrix is worth far more than the speed a tridiagonal method would buy.
pub fn jacobi_eigh(a_in: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0f64; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }
    for _sweep in 0..60 {
        let off: f64 = (0..k)
            .flat_map(|p| ((p + 1)..k).map(move |q| (p, q)))
            .map(|(p, q)| a[p * k + q] * a[p * k + q])
            .sum();
        if off <= 1e-30 {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[p * k + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[q * k + q] - a[p * k + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for i in 0..k {
                    let (aip, aiq) = (a[i * k + p], a[i * k + q]);
                    a[i * k + p] = c * aip - s * aiq;
                    a[i * k + q] = s * aip + c * aiq;
                }
                for i in 0..k {
                    let (api, aqi) = (a[p * k + i], a[q * k + i]);
                    a[p * k + i] = c * api - s * aqi;
                    a[q * k + i] = s * api + c * aqi;
                }
                for i in 0..k {
                    let (vip, viq) = (v[i * k + p], v[i * k + q]);
                    v[i * k + p] = c * vip - s * viq;
                    v[i * k + q] = s * vip + c * viq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&x, &y| {
        a[y * k + y]
            .partial_cmp(&a[x * k + x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let vals: Vec<f64> = order.iter().map(|&i| a[i * k + i]).collect();
    let mut vecs = vec![0.0f64; k * k];
    for (m, &i) in order.iter().enumerate() {
        // Sign convention: the largest-magnitude component is positive, so the output is stable
        // across runs and platforms. The metric is invariant to it; a reproducible file is not.
        let col: Vec<f64> = (0..k).map(|row| v[row * k + i]).collect();
        let piv = col
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.abs()
                    .partial_cmp(&b.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let sign = if col[piv] < 0.0 { -1.0 } else { 1.0 };
        for (row, x) in col.into_iter().enumerate() {
            vecs[row * k + m] = sign * x;
        }
    }
    (vals, vecs)
}

/// `wqt[j] = W · qt[j]`, i.e. `wqt[j][i] = ⟨W[i], qt[j]⟩`.
fn apply_w(w: &[f32], qt: &[f64], wqt: &mut [f64], d_out: usize, d_in: usize, k: usize) {
    wqt[..k * d_out]
        .par_chunks_mut(d_out)
        .enumerate()
        .for_each(|(j, out)| {
            let q = &qt[j * d_in..(j + 1) * d_in];
            for (i, o) in out.iter_mut().enumerate() {
                let wr = &w[i * d_in..(i + 1) * d_in];
                let mut acc = 0.0f64;
                for a in 0..d_in {
                    acc += wr[a] as f64 * q[a];
                }
                *o = acc;
            }
        });
}

/// `yt[j] = Wᵀ · wqt[j]`, i.e. `yt[j][a] = Σ_i W[i][a] · wqt[j][i]`.
fn apply_w_transpose(w: &[f32], wqt: &[f64], yt: &mut [f64], d_out: usize, d_in: usize, k: usize) {
    yt[..k * d_in]
        .par_chunks_mut(d_in)
        .enumerate()
        .for_each(|(j, out)| {
            out.fill(0.0);
            let c = &wqt[j * d_out..(j + 1) * d_out];
            for i in 0..d_out {
                let s = c[i];
                if s == 0.0 {
                    continue;
                }
                let wr = &w[i * d_in..(i + 1) * d_in];
                for a in 0..d_in {
                    out[a] += s * wr[a] as f64;
                }
            }
        });
}

/// The top-`r` right singular subspace of `w`, as `(V_r, Σ_r)`.
///
/// `w` is `[d_out][d_in]` row-major, the layout the output projection is stored in.
/// `v_r` comes back `[d_in][r]` row-major and `sigma_r` descending.
pub fn top_right_singular(
    w: &[f32],
    d_out: usize,
    d_in: usize,
    r: usize,
    cfg: &FactorConfig,
    layer: usize,
) -> Result<(Vec<f64>, Vec<f64>, FactorReport), FactorError> {
    if r == 0 || r >= d_in.min(d_out) {
        return Err(FactorError::Shape(
            "rank must satisfy 1 <= r < min(d_in, d_out)",
        ));
    }
    if w.len() != d_out * d_in {
        return Err(FactorError::Shape("the weight is not [d_out][d_in]"));
    }
    // A non-finite weight would poison the Ritz values and surface as "did not converge", which
    // sends the reader looking at the iteration instead of at the weight it was handed.
    if let Some((i, x)) = w.iter().enumerate().find(|(_, x)| !x.is_finite()) {
        return Err(FactorError::NonFiniteWeight {
            index: i,
            value: *x as f64,
        });
    }
    let k = (cfg.block_mult * r).min(d_in.min(d_out));
    let seed = cfg.seed ^ (layer as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

    let mut qt = vec![0.0f64; k * d_in];
    for j in 0..k {
        for a in 0..d_in {
            qt[j * d_in + a] = gaussian(seed, (j * d_in + a) as u64);
        }
    }
    orthonormalize_rows(&mut qt, k, d_in, seed);

    let mut wqt = vec![0.0f64; k * d_out];
    let mut yt = vec![0.0f64; k * d_in];
    let mut b = vec![0.0f64; k * k];

    let mut prev_energy = f64::NEG_INFINITY;
    let mut settled = 0usize;
    let mut iters = 0usize;
    let mut gain = f64::INFINITY;
    let (mut vals, mut vecs) = (Vec::new(), Vec::new());

    for t in 1..=cfg.max_iters {
        iters = t;
        apply_w(w, &qt, &mut wqt, d_out, d_in, k);

        // B = Qᵀ G Q = (WQ)ᵀ (WQ), symmetric by construction.
        b.par_chunks_mut(k).enumerate().for_each(|(j, row)| {
            let cj = &wqt[j * d_out..(j + 1) * d_out];
            for (jp, o) in row.iter_mut().enumerate() {
                let cp = &wqt[jp * d_out..(jp + 1) * d_out];
                *o = cj.iter().zip(cp.iter()).map(|(a, c)| a * c).sum();
            }
        });

        let (l, c) = jacobi_eigh(&b, k);
        if l[..r].iter().any(|x| !x.is_finite()) {
            return Err(FactorError::NonFiniteRitz { iter: t });
        }
        let energy: f64 = l[..r].iter().sum();
        gain = if prev_energy.is_finite() && energy > 0.0 {
            ((energy - prev_energy) / energy).abs()
        } else {
            f64::INFINITY
        };
        vals = l;
        vecs = c;
        if gain <= cfg.energy_tol {
            settled += 1;
            if settled >= 2 {
                break;
            }
        } else {
            settled = 0;
        }
        prev_energy = energy;

        apply_w_transpose(w, &wqt, &mut yt, d_out, d_in, k);
        qt.copy_from_slice(&yt);
        orthonormalize_rows(&mut qt, k, d_in, seed);
    }

    if settled < 2 {
        return Err(FactorError::NotConverged {
            iters,
            energy_gain: gain,
        });
    }

    // V_r = Qᵀ · C[:, ..r], written [d_in][r].
    let mut v_r = vec![0.0f64; d_in * r];
    for m in 0..r {
        for j in 0..k {
            let cjm = vecs[j * k + m];
            if cjm == 0.0 {
                continue;
            }
            let q = &qt[j * d_in..(j + 1) * d_in];
            for a in 0..d_in {
                v_r[a * r + m] += cjm * q[a];
            }
        }
    }
    let sigma_r: Vec<f64> = vals[..r].iter().map(|&x| x.max(0.0).sqrt()).collect();

    let report = check(w, &v_r, &sigma_r, d_out, d_in, r, iters, gain, seed)?;
    if report.residual > cfg.residual_limit {
        return Err(FactorError::Gate {
            gate: "residual",
            got: report.residual,
            limit: cfg.residual_limit,
        });
    }
    Ok((v_r, sigma_r, report))
}

/// The oracle-free checks, run after every build and after every external load.
#[allow(clippy::too_many_arguments)]
pub fn check(
    w: &[f32],
    v_r: &[f64],
    sigma_r: &[f64],
    d_out: usize,
    d_in: usize,
    r: usize,
    iters: usize,
    energy_gain: f64,
    seed: u64,
) -> Result<FactorReport, FactorError> {
    if sigma_r.iter().any(|s| !s.is_finite()) || sigma_r.last().is_none_or(|&s| s <= 0.0) {
        return Err(FactorError::Gate {
            gate: "order",
            got: *sigma_r.last().unwrap_or(&f64::NAN),
            limit: 0.0,
        });
    }
    if sigma_r.windows(2).any(|s| s[0] < s[1]) {
        return Err(FactorError::Gate {
            gate: "order",
            got: 0.0,
            limit: 0.0,
        });
    }
    let s1sq = sigma_r[0] * sigma_r[0];

    // W V_r, [d_out][r] — every remaining quantity comes out of it.
    let mut wv = vec![0.0f64; d_out * r];
    wv.par_chunks_mut(r).enumerate().for_each(|(i, out)| {
        let wr = &w[i * d_in..(i + 1) * d_in];
        out.fill(0.0);
        for a in 0..d_in {
            let x = wr[a] as f64;
            if x == 0.0 {
                continue;
            }
            let vrow = &v_r[a * r..(a + 1) * r];
            for m in 0..r {
                out[m] += x * vrow[m];
            }
        }
    });

    // G v_m = Wᵀ (W v_m), [d_in][r]
    let mut gv = vec![0.0f64; d_in * r];
    for i in 0..d_out {
        let wr = &w[i * d_in..(i + 1) * d_in];
        let c = &wv[i * r..(i + 1) * r];
        for a in 0..d_in {
            let x = wr[a] as f64;
            if x == 0.0 {
                continue;
            }
            let g = &mut gv[a * r..(a + 1) * r];
            for m in 0..r {
                g[m] += x * c[m];
            }
        }
    }

    let mut orthogonality: f64 = 0.0;
    for m in 0..r {
        for n in 0..r {
            let dot: f64 = (0..d_in).map(|a| v_r[a * r + m] * v_r[a * r + n]).sum();
            let want = if m == n { 1.0 } else { 0.0 };
            orthogonality = orthogonality.max((dot - want).abs());
        }
    }

    let mut residual: f64 = 0.0;
    let mut ritz: f64 = 0.0;
    for m in 0..r {
        let lam = sigma_r[m] * sigma_r[m];
        let mut res2 = 0.0f64;
        let mut quad = 0.0f64;
        for a in 0..d_in {
            let (g, v) = (gv[a * r + m], v_r[a * r + m]);
            let e = g - lam * v;
            res2 += e * e;
            quad += v * g;
        }
        residual = residual.max(res2.sqrt() / s1sq);
        ritz = ritz.max((quad - lam).abs() / s1sq);
    }

    // The energy identity: tr(M_r) must equal ‖W V_r‖_F². A consistency check between the returned
    // vectors and values — NOT a proof that the subspace is the top one. It holds exactly for a
    // subspace built from the *smallest* r directions, which is why the deflation check below
    // exists.
    let wv_energy: f64 = wv.iter().map(|x| x * x).sum();
    let claimed: f64 = sigma_r.iter().map(|s| s * s).sum();
    let w_energy: f64 = w.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    let energy_rel = (claimed - wv_energy).abs() / w_energy.max(1e-300);
    if energy_rel > 1e-10 {
        return Err(FactorError::Gate {
            gate: "energy",
            got: energy_rel,
            limit: 1e-10,
        });
    }

    // Deflation: the largest singular value left outside V_r. Power iteration on
    // W (I − V_r V_rᵀ), which needs no extra storage.
    let mut x: Vec<f64> = (0..d_in)
        .map(|a| gaussian(seed ^ 0xD3F1_A710, a as u64))
        .collect();
    let mut beta = 0.0f64;
    let mut proj = vec![0.0f64; r];
    let mut wx = vec![0.0f64; d_out];
    for _ in 0..60 {
        for m in 0..r {
            proj[m] = (0..d_in).map(|a| v_r[a * r + m] * x[a]).sum();
        }
        for a in 0..d_in {
            let vrow = &v_r[a * r..(a + 1) * r];
            let mut s = 0.0;
            for m in 0..r {
                s += vrow[m] * proj[m];
            }
            x[a] -= s;
        }
        wx.par_iter_mut().enumerate().for_each(|(i, o)| {
            let wr = &w[i * d_in..(i + 1) * d_in];
            *o = (0..d_in).map(|a| wr[a] as f64 * x[a]).sum();
        });
        let mut y = vec![0.0f64; d_in];
        for i in 0..d_out {
            let s = wx[i];
            if s == 0.0 {
                continue;
            }
            let wr = &w[i * d_in..(i + 1) * d_in];
            for a in 0..d_in {
                y[a] += s * wr[a] as f64;
            }
        }
        let n: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if n <= 0.0 {
            beta = 0.0;
            break;
        }
        beta = n.sqrt();
        let inv = 1.0 / n;
        for (a, v) in y.into_iter().enumerate() {
            x[a] = v * inv;
        }
    }
    let deflation_ratio = beta / sigma_r[r - 1];
    if deflation_ratio > 1.0 + 1e-3 {
        return Err(FactorError::Gate {
            gate: "deflation",
            got: deflation_ratio,
            limit: 1.0 + 1e-3,
        });
    }

    Ok(FactorReport {
        iters,
        energy_gain,
        residual,
        orthogonality,
        ritz,
        deflation_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matrix with a known spectrum: `W = U diag(s) Vᵀ` built from two orthonormal blocks.
    fn synth(d: usize, spectrum: &[f64], seed: u64) -> (Vec<f32>, Vec<f64>) {
        let n = spectrum.len();
        let mut u = vec![0.0f64; n * d];
        let mut v = vec![0.0f64; n * d];
        for j in 0..n {
            for a in 0..d {
                u[j * d + a] = gaussian(seed, (j * d + a) as u64);
                v[j * d + a] = gaussian(seed ^ 0xABCD, (j * d + a) as u64);
            }
        }
        orthonormalize_rows(&mut u, n, d, seed);
        orthonormalize_rows(&mut v, n, d, seed);
        let mut w = vec![0.0f32; d * d];
        for j in 0..n {
            let s = spectrum[j];
            for i in 0..d {
                let ui = u[j * d + i] * s;
                for a in 0..d {
                    w[i * d + a] += (ui * v[j * d + a]) as f32;
                }
            }
        }
        (w, v)
    }

    #[test]
    fn recovers_a_known_spectrum() {
        let d = 40;
        let spectrum: Vec<f64> = (0..24).map(|i| 10.0 * 0.82f64.powi(i)).collect();
        let (w, _) = synth(d, &spectrum, 7);
        let cfg = FactorConfig {
            block_mult: 8,
            ..Default::default()
        };
        let (_, sigma, rep) = top_right_singular(&w, d, d, 4, &cfg, 0).expect("converged");
        for (got, want) in sigma.iter().zip(&spectrum) {
            assert!(
                (got - want).abs() <= 1e-5 * want,
                "sigma {got} vs {want} (report {rep:?})"
            );
        }
        assert!(rep.residual < 1e-6, "{rep:?}");
        assert!(rep.orthogonality < 1e-10, "{rep:?}");
        assert!(rep.deflation_ratio <= 1.0 + 1e-3, "{rep:?}");
    }

    /// The property the metric actually depends on: `M_r = V_r Σ_r² V_rᵀ`, which is well determined
    /// even where the individual directions are not.
    #[test]
    fn the_truncation_operator_is_reproduced_even_with_a_flat_spectrum() {
        let d = 48;
        // A deliberately flat cut: sigma_{r+1}/sigma_r ~ 0.99, the regime the real weights are in.
        let spectrum: Vec<f64> = (0..48).map(|i| 3.0 * 0.99f64.powi(i)).collect();
        let (w, v_true) = synth(d, &spectrum, 11);
        let r = 4;
        let cfg = FactorConfig::default();
        let (v_r, sigma, rep) = top_right_singular(&w, d, d, r, &cfg, 0).expect("converged");

        // M_r from the recovered factors against M_r from the construction.
        let mut got = vec![0.0f64; d * d];
        let mut want = vec![0.0f64; d * d];
        for m in 0..r {
            let l = sigma[m] * sigma[m];
            let lw = spectrum[m] * spectrum[m];
            for a in 0..d {
                for b in 0..d {
                    got[a * d + b] += l * v_r[a * r + m] * v_r[b * r + m];
                    want[a * d + b] += lw * v_true[m * d + a] * v_true[m * d + b];
                }
            }
        }
        let num: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w) * (g - w))
            .sum::<f64>()
            .sqrt();
        let den: f64 = want.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            num / den < 1e-5,
            "truncation operator off by {:e} (report {rep:?})",
            num / den
        );
    }

    #[test]
    fn the_deflation_gate_catches_a_bottom_r_subspace() {
        let d = 32;
        let spectrum: Vec<f64> = (0..32).map(|i| 5.0 * 0.9f64.powi(i)).collect();
        let (w, v_true) = synth(d, &spectrum, 3);
        let r = 3;
        // Hand it the SMALLEST r directions: orthonormal, exactly invariant, zero residual — every
        // gate but deflation is satisfied.
        let mut v_r = vec![0.0f64; d * r];
        let mut sigma = vec![0.0f64; r];
        for m in 0..r {
            // descending within the bottom block, so the ordering gate has nothing to say
            let src = spectrum.len() - r + m;
            sigma[m] = spectrum[src];
            for a in 0..d {
                v_r[a * r + m] = v_true[src * d + a];
            }
        }
        let e = check(&w, &v_r, &sigma, d, d, r, 1, 0.0, 5).expect_err("must be rejected");
        assert!(
            matches!(
                e,
                FactorError::Gate {
                    gate: "deflation",
                    ..
                }
            ),
            "expected the deflation gate to fire, got {e:?}"
        );
    }

    #[test]
    fn the_decomposition_is_reproducible() {
        let d = 32;
        let spectrum: Vec<f64> = (0..32).map(|i| 4.0 * 0.85f64.powi(i)).collect();
        let (w, _) = synth(d, &spectrum, 5);
        let cfg = FactorConfig::default();
        let a = top_right_singular(&w, d, d, 3, &cfg, 2).unwrap();
        let b = top_right_singular(&w, d, d, 3, &cfg, 2).unwrap();
        assert_eq!(
            a.0, b.0,
            "V_r must not depend on anything but (w, r, seed, layer)"
        );
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn jacobi_diagonalizes_a_symmetric_block() {
        let k = 6;
        let mut a = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let x = gaussian(99, (i * k + j) as u64);
                a[i * k + j] += x;
                a[j * k + i] += x;
            }
        }
        let (vals, vecs) = jacobi_eigh(&a, k);
        assert!(vals.windows(2).all(|w| w[0] >= w[1]), "descending");
        for m in 0..k {
            for i in 0..k {
                let av: f64 = (0..k).map(|j| a[i * k + j] * vecs[j * k + m]).sum();
                assert!((av - vals[m] * vecs[i * k + m]).abs() < 1e-9);
            }
        }
    }
}
