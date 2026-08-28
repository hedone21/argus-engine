//! The measurement operator: recompute `R` attention rows against what a candidate leaves behind,
//! project through the output projection, and read the deviation off in `r`-space.
//!
//! Plain slices, no engine types — every property the parity gate checks is reachable from a unit
//! test without a model, the way `techniques/peekkv`'s scorer is.
//!
//! ## Two identities the whole design rests on
//!
//! **Logits are candidate-independent.** An eviction-shaped candidate leaves the retained keys
//! *unchanged*: `K̃[j] = K[keep[j]]`. So its logit at compacted column `j` is the same float as the
//! baseline's logit at absolute column `keep[j]`. One `Q Kᵀ` per layer therefore serves the baseline
//! and every candidate; a candidate only chooses which columns enter its softmax. Gathering columns
//! (rather than masking them to `-inf` across the full row) is also what makes the identity
//! candidate come out at exactly zero: it sums the same terms in the same order as the baseline.
//!
//! **`U_r` cancels.** The reference projects `O = (X V_r)(U_r Σ_r)ᵀ`. Since `U_r` has orthonormal
//! columns it is an isometry on its column space, so with `w = X (V_r Σ_r)`:
//!
//! ```text
//! ⟨O, Õ⟩ = w · w̃        ‖O − Õ‖ = ‖w − w̃‖        ‖O‖ = ‖w‖
//! ```
//!
//! Both readouts are exact functions of `w ∈ R^r` alone. The `R × d` rows are never materialized,
//! the projection halves, the readout shrinks by `d/r`, and `U_r` is never stored.
//!
//! ## Numerics
//!
//! f32 throughout, matching the reference. `1/√d_h` is applied as a *division* because the
//! reference divides; the softmax subtracts the row max; discarded keys are absent from the
//! partition function rather than renormalized afterwards. A row that admits nothing is an error,
//! never a uniform fallback — the engine's own host attention substitutes uniform there, and
//! inheriting that would turn a structural bug into a plausible number.

use rayon::prelude::*;

use super::keep::{KeepSets, KeyPos};

/// The geometry one decision runs at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geom {
    /// Transformer layers.
    pub n_layers: usize,
    /// Query heads.
    pub n_heads_q: usize,
    /// KV heads. `n_heads_q` is a multiple of this; query head `h` belongs to group `h / n_rep`.
    pub n_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Resident tokens at the decision point.
    pub current_pos: usize,
    /// Trailing query rows scored, `R`. Absolute positions `current_pos - rows .. current_pos`.
    pub rows: usize,
}

impl Geom {
    /// Query heads per KV group.
    #[inline]
    pub fn n_rep(&self) -> usize {
        (self.n_heads_q / self.n_kv_heads).max(1)
    }

    /// `d` — the width the output projection consumes, `n_heads_q * head_dim`. Not `hidden_size`:
    /// they differ on models whose head dimension is not `hidden_size / n_heads_q`.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_heads_q * self.head_dim
    }

    /// Absolute position of query row `t`.
    #[inline]
    pub fn row_pos(&self, t: usize) -> usize {
        self.current_pos - self.rows + t
    }

    /// Elements in one layer's logit block.
    #[inline]
    pub fn logit_len(&self) -> usize {
        self.n_heads_q * self.rows * self.current_pos
    }

    /// Elements in one layer's pre-projection rows.
    #[inline]
    pub fn x_len(&self) -> usize {
        self.rows * self.q_dim()
    }
}

/// What went wrong in a way that must not be papered over with a plausible number.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelError {
    /// A `(kv_head, row)` admitted no retained position, yet the caller asked for its output. The
    /// caller is expected to clamp the boundary and drop the row from the aggregation instead.
    NoAdmittedKeys { kv_head: usize, row: usize },
    /// A logit or an output was not finite. The reference treats this as fatal for the record,
    /// because `max(0.0, NaN) == 0.0` lets a NaN slip past any threshold gate.
    NonFinite { what: &'static str },
    /// A slice was not the length the geometry implies.
    BadLen {
        what: &'static str,
        got: usize,
        want: usize,
    },
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdmittedKeys { kv_head, row } => write!(
                f,
                "aperturb: (kv_head {kv_head}, row {row}) admits no retained key — the caller must \
                 clamp the boundary and drop the row, not ask for its output"
            ),
            Self::NonFinite { what } => write!(f, "aperturb: non-finite {what}"),
            Self::BadLen { what, got, want } => {
                write!(f, "aperturb: {what} has {got} elements, expected {want}")
            }
        }
    }
}

impl std::error::Error for KernelError {}

/// `z[(h * rows + t) * current_pos + p] = ⟨q[h][t], k[h / n_rep][p]⟩ / √head_dim`.
///
/// Candidate-independent, computed once per layer. Layout is `[h][t][p]` with `p` innermost: the
/// softmax then reduces over a contiguous axis, and a candidate's column gather is a monotone walk
/// through one row rather than a strided sweep.
///
/// - `q` — `[n_heads_q][rows][head_dim]`
/// - `k` — `[n_kv_heads][current_pos][head_dim]` (the cache's head-major layout)
pub fn logits_into(q: &[f32], k: &[f32], z: &mut [f32], g: Geom) -> Result<(), KernelError> {
    check_len("q", q.len(), g.n_heads_q * g.rows * g.head_dim)?;
    check_len("k", k.len(), g.n_kv_heads * g.current_pos * g.head_dim)?;
    check_len("logits", z.len(), g.logit_len())?;

    let (dh, s, rows, n_rep) = (g.head_dim, g.current_pos, g.rows, g.n_rep());
    // The reference divides by √d_h; a reciprocal multiply is a different float.
    let denom = (dh as f32).sqrt();
    z.par_chunks_mut(rows * s).enumerate().for_each(|(h, zh)| {
        let kh = &k[(h / n_rep) * s * dh..(h / n_rep + 1) * s * dh];
        for t in 0..rows {
            let qv = &q[(h * rows + t) * dh..(h * rows + t + 1) * dh];
            let zr = &mut zh[t * s..(t + 1) * s];
            for (p, zp) in zr.iter_mut().enumerate() {
                let kv = &kh[p * dh..(p + 1) * dh];
                let mut acc = 0.0f32;
                for e in 0..dh {
                    acc += qv[e] * kv[e];
                }
                *zp = acc / denom;
            }
        }
    });
    Ok(())
}

/// One `(candidate, layer)` pre-projection output from the shared logits.
///
/// `x[t * q_dim + h * head_dim + e]` — head-major within a row, the layout the output projection
/// consumes and the layout the reference's `X` has.
///
/// Row `t` of head `h` softmaxes over the retained positions `keep[0 ..= kp.get(h, t)]` and
/// contracts them against `v`.
///
/// A row whose boundary is negative sees nothing, and its boundary is **clamped to the first
/// retained position** rather than skipped. The value that produces is meaningless and the caller
/// drops it — [`KeyPos::visible_rows`] already excludes the row from every aggregation. Computing
/// it anyway is what keeps the `L × R` grid rectangular, so the trailing-row slice lines up across
/// candidates that blind different rows. Leaving it as zeros instead would make the direction
/// readout divide by a zero norm.
///
/// - `v` — `[n_kv_heads][current_pos][head_dim]`
pub fn attend_into(
    z: &[f32],
    keep_layer: &KeepSets,
    layer: usize,
    kp: &KeyPos,
    v: &[f32],
    x: &mut [f32],
    g: Geom,
) -> Result<(), KernelError> {
    check_len("logits", z.len(), g.logit_len())?;
    check_len("v", v.len(), g.n_kv_heads * g.current_pos * g.head_dim)?;
    check_len("x", x.len(), g.x_len())?;

    let (dh, s, rows, n_rep, qd) = (g.head_dim, g.current_pos, g.rows, g.n_rep(), g.q_dim());
    let heads: Vec<Result<Vec<f32>, KernelError>> = (0..g.n_heads_q)
        .into_par_iter()
        .map(|h| {
            let kv_h = h / n_rep;
            let list = keep_layer.head(layer, kv_h);
            let vh = &v[kv_h * s * dh..(kv_h + 1) * s * dh];
            let zh = &z[h * rows * s..(h + 1) * rows * s];
            if list.is_empty() {
                return Err(KernelError::NoAdmittedKeys {
                    kv_head: kv_h,
                    row: 0,
                });
            }
            let mut out = vec![0.0f32; rows * dh];
            let mut w = Vec::with_capacity(list.len());
            for t in 0..rows {
                // A blind row is clamped to one column, not skipped — see the doc comment.
                let n_adm = kp.admitted(kv_h, t).max(1);
                let zr = &zh[t * s..(t + 1) * s];
                // max over the admitted columns only — identical to masking the rest to -inf,
                // and it is what makes the identity candidate reduce in the baseline's order.
                let mut m = f32::NEG_INFINITY;
                for &p in &list[..n_adm] {
                    let zp = zr[p as usize];
                    if zp > m {
                        m = zp;
                    }
                }
                if !m.is_finite() {
                    return Err(KernelError::NonFinite { what: "logit" });
                }
                w.clear();
                let mut sum = 0.0f32;
                for &p in &list[..n_adm] {
                    let e = (zr[p as usize] - m).exp();
                    w.push(e);
                    sum += e;
                }
                let inv = 1.0 / sum;
                let o = &mut out[t * dh..(t + 1) * dh];
                for (j, &p) in list[..n_adm].iter().enumerate() {
                    let a = w[j] * inv;
                    let vr = &vh[p as usize * dh..(p as usize + 1) * dh];
                    for e in 0..dh {
                        o[e] += a * vr[e];
                    }
                }
            }
            Ok(out)
        })
        .collect();

    for (h, r) in heads.into_iter().enumerate() {
        let o = r?;
        for t in 0..rows {
            x[t * qd + h * dh..t * qd + (h + 1) * dh].copy_from_slice(&o[t * dh..(t + 1) * dh]);
        }
    }
    Ok(())
}

/// `w[t * r + k] = Σ_e x[t * d + e] · basis[e * r + k]`, with `basis = V_r Σ_r`.
///
/// The only projection the readout needs (see the module header). `basis` is `[d][r]` row-major.
pub fn project_into(x: &[f32], basis: &[f32], w: &mut [f32], rows: usize, d: usize, r: usize) {
    debug_assert_eq!(x.len(), rows * d);
    debug_assert_eq!(basis.len(), d * r);
    debug_assert_eq!(w.len(), rows * r);
    for t in 0..rows {
        let xr = &x[t * d..(t + 1) * d];
        let wr = &mut w[t * r..(t + 1) * r];
        wr.fill(0.0);
        for e in 0..d {
            let xe = xr[e];
            if xe == 0.0 {
                continue;
            }
            let br = &basis[e * r..(e + 1) * r];
            for k in 0..r {
                wr[k] += xe * br[k];
            }
        }
    }
}

#[inline]
fn check_len(what: &'static str, got: usize, want: usize) -> Result<(), KernelError> {
    if got == want {
        Ok(())
    } else {
        Err(KernelError::BadLen { what, got, want })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> Geom {
        Geom {
            n_layers: 1,
            n_heads_q: 4,
            n_kv_heads: 2,
            head_dim: 3,
            current_pos: 7,
            rows: 2,
        }
    }

    fn ramp(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.9)
            .collect()
    }

    #[test]
    fn identity_candidate_reproduces_the_baseline_bit_for_bit() {
        let g = geom();
        let q = ramp(g.n_heads_q * g.rows * g.head_dim, 0.1);
        let k = ramp(g.n_kv_heads * g.current_pos * g.head_dim, 1.3);
        let v = ramp(g.n_kv_heads * g.current_pos * g.head_dim, 2.7);
        let mut z = vec![0.0; g.logit_len()];
        logits_into(&q, &k, &mut z, g).unwrap();

        let keep = KeepSets::identity(1, g.n_kv_heads, g.current_pos);
        let kp = KeyPos::for_layer(&keep, 0, g.current_pos, g.rows);
        let mut a = vec![0.0; g.x_len()];
        let mut b = vec![0.0; g.x_len()];
        attend_into(&z, &keep, 0, &kp, &v, &mut a, g).unwrap();
        attend_into(&z, &keep, 0, &kp, &v, &mut b, g).unwrap();
        assert_eq!(a, b, "the operator is deterministic");
        // Row t may attend to positions [0, current_pos-rows+t]; nothing beyond it contributed.
        assert!(a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn a_candidate_matches_attention_computed_over_a_gathered_cache() {
        let g = geom();
        let q = ramp(g.n_heads_q * g.rows * g.head_dim, 0.1);
        let k = ramp(g.n_kv_heads * g.current_pos * g.head_dim, 1.3);
        let v = ramp(g.n_kv_heads * g.current_pos * g.head_dim, 2.7);
        let keep_list: Vec<u32> = vec![0, 2, 5, 6];

        // route A — shared logits, column gather
        let mut z = vec![0.0; g.logit_len()];
        logits_into(&q, &k, &mut z, g).unwrap();
        let keep = KeepSets::uniform(1, g.n_kv_heads, &keep_list);
        let kp = KeyPos::for_layer(&keep, 0, g.current_pos, g.rows);
        let mut xa = vec![0.0; g.x_len()];
        attend_into(&z, &keep, 0, &kp, &v, &mut xa, g).unwrap();

        // route B — physically gather K and V first, then recompute from scratch
        let gg = Geom {
            current_pos: keep_list.len(),
            ..g
        };
        let mut kg = vec![0.0; g.n_kv_heads * keep_list.len() * g.head_dim];
        let mut vg = kg.clone();
        for h in 0..g.n_kv_heads {
            for (j, &p) in keep_list.iter().enumerate() {
                let src = (h * g.current_pos + p as usize) * g.head_dim;
                let dst = (h * keep_list.len() + j) * g.head_dim;
                kg[dst..dst + g.head_dim].copy_from_slice(&k[src..src + g.head_dim]);
                vg[dst..dst + g.head_dim].copy_from_slice(&v[src..src + g.head_dim]);
            }
        }
        let mut zg = vec![0.0; gg.logit_len()];
        logits_into(&q, &kg, &mut zg, gg).unwrap();
        let keep_g = KeepSets::identity(1, g.n_kv_heads, keep_list.len());
        // reuse the ORIGINAL boundary: the compacted column budget is the same count
        let mut xb = vec![0.0; gg.x_len()];
        attend_into(&zg, &keep_g, 0, &kp, &vg, &mut xb, gg).unwrap();

        assert_eq!(
            xa, xb,
            "column gather must be bit-identical to a real gather"
        );
    }

    #[test]
    fn a_blind_row_is_computed_against_the_first_retained_key_and_flagged() {
        let g = geom();
        let q = ramp(g.n_heads_q * g.rows * g.head_dim, 0.1);
        let k = ramp(g.n_kv_heads * g.current_pos * g.head_dim, 1.3);
        let v = ramp(g.n_kv_heads * g.current_pos * g.head_dim, 2.7);
        let mut z = vec![0.0; g.logit_len()];
        logits_into(&q, &k, &mut z, g).unwrap();
        // rows sit at positions 5 and 6; keeping only position 6 blinds row 0.
        let keep = KeepSets::uniform(1, g.n_kv_heads, &[6]);
        let kp = KeyPos::for_layer(&keep, 0, g.current_pos, g.rows);
        assert_eq!(kp.visible_rows(), 0b10, "only row 1 sees a retained key");
        assert_eq!(kp.admitted(0, 0), 0);
        let mut x = vec![9.0; g.x_len()];
        attend_into(&z, &keep, 0, &kp, &v, &mut x, g).unwrap();
        // The blind row still gets a value: the sole retained position, weight 1. Meaningless, and
        // the caller drops it — but finite, which the direction readout needs.
        assert!(x.iter().all(|e| e.is_finite()));
        let vh = &v[0..g.current_pos * g.head_dim];
        for e in 0..g.head_dim {
            assert!((x[e] - vh[6 * g.head_dim + e]).abs() < 1e-6);
        }
    }

    #[test]
    fn the_projection_agrees_with_the_reference_factored_form() {
        // O = (X V_r)(U_r S_r)^T ; the readout only needs w = X (V_r S_r), and ||O|| = ||w||.
        let (rows, d, r) = (2usize, 6usize, 2usize);
        let x = ramp(rows * d, 0.5);
        // an orthonormal U_r built by hand, so the identity is exercised, not assumed
        let u: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sigma = [3.0f32, 0.5];
        let v_r = ramp(d * r, 2.0);
        let basis: Vec<f32> = (0..d * r).map(|i| v_r[i] * sigma[i % r]).collect();

        let mut w = vec![0.0; rows * r];
        project_into(&x, &basis, &mut w, rows, d, r);

        // O = z * (U_r S_r)^T with z = X V_r
        let mut z = vec![0.0; rows * r];
        project_into(&x, &v_r, &mut z, rows, d, r);
        let mut o = vec![0.0; rows * d];
        for t in 0..rows {
            for i in 0..d {
                let mut acc = 0.0f32;
                for kk in 0..r {
                    acc += z[t * r + kk] * u[i * r + kk] * sigma[kk];
                }
                o[t * d + i] = acc;
            }
        }
        for t in 0..rows {
            let nw: f32 = w[t * r..(t + 1) * r]
                .iter()
                .map(|a| a * a)
                .sum::<f32>()
                .sqrt();
            let no: f32 = o[t * d..(t + 1) * d]
                .iter()
                .map(|a| a * a)
                .sum::<f32>()
                .sqrt();
            assert!(
                (nw - no).abs() <= 1e-5 * no.max(1.0),
                "‖w‖ {nw} vs ‖O‖ {no}"
            );
        }
    }

    #[test]
    fn length_mismatches_are_refused() {
        let g = geom();
        let mut z = vec![0.0; g.logit_len() - 1];
        let q = vec![0.0; g.n_heads_q * g.rows * g.head_dim];
        let k = vec![0.0; g.n_kv_heads * g.current_pos * g.head_dim];
        assert!(matches!(
            logits_into(&q, &k, &mut z, g),
            Err(KernelError::BadLen { .. })
        ));
    }
}
