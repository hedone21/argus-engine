//! The paper's Quality Cost Function — ranking KV-cache compression candidates by how far each one
//! moves the model's own attention output, without running a forward pass.
//!
//! ## Not [`crate::qcf`]
//!
//! That module is an older metric with the same name: a per-KV-head relative L2 on `Σ α V`, with no
//! keys, no output projection, and no re-softmax. This one recomputes attention against the cache a
//! candidate *would* leave behind. The two answer different questions and neither replaces the
//! other, so they live apart. The name here is `aperturb` — the reference harness's own identifier
//! for these columns (`aperturb_prev16_wo0p78125_l2_rms`), which keeps engine module, metric key and
//! paper-side join key spelled the same way.
//!
//! ## The measurement
//!
//! At a decision point with `N` tokens resident, for each layer `ℓ`:
//!
//! 1. Take the post-RoPE query rows of the last `R` positions, `Q_ℓ ∈ R^{n_q × R × d_h}`. They are
//!    what the forward already computed; nothing is re-derived through a perturbed earlier layer
//!    (**frozen propagation**), and no FFN or LM head runs.
//! 2. Recompute those `R` rows' attention over the **uncompressed** cache → the reference `X_ℓ`.
//! 3. For each candidate, recompute the same `R` rows over what it retains → `X̃_ℓ`.
//! 4. Project both through a rank-`r` truncation of the output projection and read the deviation
//!    per `(layer, row)`.
//! 5. Aggregate the `L × R` grid into one score; the smallest wins.
//!
//! **Step 2 recomputes rather than reusing the forward's own pre-projection output.** That looks
//! wasteful — it costs one extra baseline pass, `1/|C|` of the total — and it is the difference
//! between a working identity gate and a broken one: reference and candidate then travel the same
//! code with the same rounding, so a candidate that retains everything scores exactly zero instead
//! of the recompute's own error. Everything downstream is calibrated against that zero.
//!
//! ## Cost
//!
//! Per decision, `L · (2 n_q R N d_h + 2 n_q R N d_h + Σ_c 2 n_q R N_c d_h) + (|C|+1) · L · 2 R d r`.
//! Two structural savings against the literal form (see [`kernel`]): the `Q Kᵀ` product is shared by
//! the baseline and every eviction-shaped candidate, so each extra candidate costs half of what it
//! otherwise would; and the projection's `U_r` factor cancels in both readouts, halving the
//! projection and shrinking the readout by `d/r`.
//!
//! ## Agreement with the reference implementation
//!
//! Both sides reduce in different orders, so bit equality is not on offer; what is measured is
//! whether the disagreement is bounded by the arithmetic rather than by the algorithm.
//!
//! | check | worst relative disagreement |
//! |---|---|
//! | synthetic fixture, reference's own factors, `l2` | `1.9e-6` |
//! | synthetic fixture, no truncation at all, `l2` | `8.3e-7` |
//! | synthetic fixture, factors computed here, `l2` | `2.0e-7` |
//! | Qwen2.5-0.5B, real tensors, 24 layers × 16 rows × 5 candidates, `l2` | `5.7e-7` |
//! | a candidate that retains everything, `l2` | exactly `0` |
//!
//! `dcos` agrees to the same `~6e-7` on every candidate that perturbs anything, and to `1e-7`
//! *absolute* on one that does not — where its relative accuracy is meaningless on both sides
//! because `1 − cos` cancels. That is a property of the quantity, and it is why `l2` is the default.

pub mod keep;
pub mod kernel;
pub mod readout;
pub mod subspace;

#[cfg(test)]
mod tests;

use std::fmt;

pub use keep::{KeepError, KeepSets, KeyPos};
pub use kernel::{Geom, KernelError};
pub use readout::{CellAgg, CellGrid, CellValue, Readout, ReadoutError, ReadoutSet, metric_key};

/// One layer's low-rank output-projection basis, `B_r = V_r Σ_r`, stored `[d][r]` row-major.
///
/// The rank-`r` truncation of `W_o` is `U_r Σ_r V_rᵀ`; both readouts are invariant to `U_r`, so
/// this is the whole of what the measurement needs. Owning it as one flat table per model, built
/// once, keeps the decision path free of any weight access.
pub struct OutputBasis {
    /// `layers[ℓ]` is `[d * rank]`.
    layers: Vec<Vec<f32>>,
    d: usize,
    rank: usize,
    /// The fraction the rank came from, or `None` for the untruncated arm. It names the metric key,
    /// so it travels with the table.
    frac: Option<f64>,
}

impl OutputBasis {
    /// Adopt per-layer `[d * rank]` bases that were produced elsewhere.
    ///
    /// This is how externally computed factors enter, which is what lets a parity run separate
    /// "the measurement disagrees" from "the decomposition disagrees".
    pub fn from_layers(
        layers: Vec<Vec<f32>>,
        d: usize,
        rank: usize,
        frac: Option<f64>,
    ) -> Result<Self, AperturbError> {
        for (l, b) in layers.iter().enumerate() {
            if b.len() != d * rank {
                return Err(AperturbError::BasisShape {
                    layer: l,
                    got: b.len(),
                    want: d * rank,
                });
            }
        }
        Ok(Self {
            layers,
            d,
            rank,
            frac,
        })
    }

    /// Build `V_r Σ_r` from the reference's factor pair — right singular vectors `[d][rank]` and the
    /// matching singular values.
    pub fn from_factors(
        v_r: &[Vec<f32>],
        sigma: &[Vec<f32>],
        d: usize,
        rank: usize,
        frac: f64,
    ) -> Result<Self, AperturbError> {
        let mut layers = Vec::with_capacity(v_r.len());
        for (l, (v, s)) in v_r.iter().zip(sigma).enumerate() {
            if v.len() != d * rank || s.len() != rank {
                return Err(AperturbError::BasisShape {
                    layer: l,
                    got: v.len(),
                    want: d * rank,
                });
            }
            let mut b = vec![0.0f32; d * rank];
            for e in 0..d {
                for k in 0..rank {
                    b[e * rank + k] = v[e * rank + k] * s[k];
                }
            }
            layers.push(b);
        }
        Self::from_layers(layers, d, rank, Some(frac))
    }

    /// Factor the output projection in-engine.
    ///
    /// `wo[ℓ]` is `[d_out][d_in]` row-major. `frac` names the rank through the reference's integer
    /// rule; the width it is a fraction of is `d_in`, the projection's input, which is what the
    /// measurement's rows live in.
    ///
    /// One decomposition per layer, once per model. Every layer's gates run and a failure is
    /// returned rather than logged: a basis that did not converge produces a metric that is wrong by
    /// an unknown amount, and there is nothing downstream that could notice.
    pub fn from_weights(
        wo: &[Vec<f32>],
        d_in: usize,
        d_out: usize,
        frac: f64,
        cfg: &subspace::FactorConfig,
    ) -> Result<(Self, Vec<subspace::FactorReport>), AperturbError> {
        let rank = Self::rank_for(frac, d_in);
        let mut layers = Vec::with_capacity(wo.len());
        let mut reports = Vec::with_capacity(wo.len());
        for (l, w) in wo.iter().enumerate() {
            if w.len() != d_out * d_in {
                return Err(AperturbError::BasisShape {
                    layer: l,
                    got: w.len(),
                    want: d_out * d_in,
                });
            }
            let (v_r, sigma, rep) = subspace::top_right_singular(w, d_out, d_in, rank, cfg, l)
                .map_err(|e| AperturbError::Factor {
                    layer: l,
                    source: e,
                })?;
            let mut b = vec![0.0f32; d_in * rank];
            for a in 0..d_in {
                for k in 0..rank {
                    b[a * rank + k] = (v_r[a * rank + k] * sigma[k]) as f32;
                }
            }
            layers.push(b);
            reports.push(rep);
        }
        Ok((Self::from_layers(layers, d_in, rank, Some(frac))?, reports))
    }

    /// The untruncated arm: the output projection itself, transposed.
    ///
    /// `O = X W_oᵀ` is what a "basis" of `W_oᵀ` at full rank computes, so the exact projection and
    /// the low-rank one are the same code with a different table — no second projection routine,
    /// and no way for the two to drift apart. This arm carries no decomposition at all, which makes
    /// it the control that separates an error in the measurement from an error in the factors.
    ///
    /// `wo[ℓ]` is `[d_out][d_in]` row-major, the layout the weight is stored in.
    pub fn untruncated(wo: &[Vec<f32>], d_in: usize, d_out: usize) -> Result<Self, AperturbError> {
        let mut layers = Vec::with_capacity(wo.len());
        for (l, w) in wo.iter().enumerate() {
            if w.len() != d_out * d_in {
                return Err(AperturbError::BasisShape {
                    layer: l,
                    got: w.len(),
                    want: d_out * d_in,
                });
            }
            let mut b = vec![0.0f32; d_in * d_out];
            for i in 0..d_out {
                for e in 0..d_in {
                    b[e * d_out + i] = w[i * d_in + e];
                }
            }
            layers.push(b);
        }
        Self::from_layers(layers, d_in, d_out, None)
    }

    /// The rank the reference's integer rule produces for a fraction: `max(1, round(frac · width))`,
    /// rounding halves to even.
    pub fn rank_for(frac: f64, width: usize) -> usize {
        let x = frac * width as f64;
        let r = if (x - x.floor() - 0.5).abs() < f64::EPSILON {
            // ties-to-even, matching the reference's `round`
            let f = x.floor();
            if (f as i64) % 2 == 0 { f } else { f + 1.0 }
        } else {
            x.round()
        };
        (r as usize).max(1)
    }

    #[inline]
    pub fn layer(&self, layer: usize) -> &[f32] {
        &self.layers[layer]
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// The rank fraction, or `None` on the untruncated arm.
    #[inline]
    pub fn frac(&self) -> Option<f64> {
        self.frac
    }

    /// The metric key this basis's scores are published under.
    pub fn metric_key(&self, rows: usize, r: Readout) -> String {
        metric_key(rows, self.frac, r)
    }
}

/// Where a layer's key/value blocks and query rows come from.
///
/// A trait rather than owned buffers so the decision path can stream one layer at a time: the whole
/// model's dequantized cache does not have to be resident, which is the difference between tens of
/// megabytes and gigabytes at a long context.
pub trait LayerSource {
    /// Post-RoPE query rows of the trailing `R` positions, `[n_heads_q][rows][head_dim]`.
    fn query_rows(&self, layer: usize) -> &[f32];
    /// Keys, `[n_kv_heads][current_pos][head_dim]`, dequantized to f32.
    fn keys(&self, layer: usize) -> &[f32];
    /// Values, same shape.
    fn values(&self, layer: usize) -> &[f32];
}

/// A candidate and the score it earned.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub name: String,
    pub scores: ReadoutSet,
    /// Query rows that saw a retained key in every layer and head, as a bitmask.
    pub visible_rows: u32,
    /// `true` when the candidate's per-head lengths differ within a layer. Such a candidate can be
    /// *measured* but not applied: the cache gives every KV head of a layer one length.
    pub ragged: bool,
    /// The per-cell grid, kept when [`Config::keep_cells`] asks for it — the tightest thing a parity
    /// comparison can look at.
    pub cells: Option<CellGrid>,
}

/// What one decision produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub scored: Vec<Scored>,
    /// Index into `scored` of the smallest score under [`Config::readout`].
    pub winner: usize,
}

impl Decision {
    pub fn winner(&self) -> &Scored {
        &self.scored[self.winner]
    }
}

/// Knobs for one decision.
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// Which of the four scores `winner` is chosen by. All four are always computed.
    pub readout: Readout,
    /// Retain the per-cell grid on every candidate.
    pub keep_cells: bool,
}

/// Anything that stops a decision from producing a number it can stand behind.
#[derive(Debug, Clone, PartialEq)]
pub enum AperturbError {
    Keep(KeepError),
    Kernel(KernelError),
    Readout(ReadoutError),
    /// A basis layer is not `[d * rank]`.
    BasisShape {
        layer: usize,
        got: usize,
        want: usize,
    },
    /// The output projection could not be factored for this layer.
    Factor {
        layer: usize,
        source: subspace::FactorError,
    },
    /// The candidate pool was empty, or the geometry cannot hold `rows` positions.
    Degenerate(&'static str),
}

impl fmt::Display for AperturbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Keep(e) => write!(f, "{e}"),
            Self::Kernel(e) => write!(f, "{e}"),
            Self::Readout(e) => write!(f, "{e}"),
            Self::BasisShape { layer, got, want } => write!(
                f,
                "aperturb: output basis for layer {layer} has {got} elements, expected {want}"
            ),
            Self::Factor { layer, source } => {
                write!(f, "aperturb: layer {layer}: {source}")
            }
            Self::Degenerate(m) => write!(f, "aperturb: {m}"),
        }
    }
}

impl std::error::Error for AperturbError {}

impl From<KeepError> for AperturbError {
    fn from(e: KeepError) -> Self {
        Self::Keep(e)
    }
}
impl From<KernelError> for AperturbError {
    fn from(e: KernelError) -> Self {
        Self::Kernel(e)
    }
}
impl From<ReadoutError> for AperturbError {
    fn from(e: ReadoutError) -> Self {
        Self::Readout(e)
    }
}

/// Score every candidate and return the smallest.
///
/// One layer at a time: dequantized keys and values, the shared logit block and every candidate's
/// pre-projection rows are live for that layer only. What survives across layers is `(|C|+1) × R × r`
/// floats of projected rows — a few hundred kilobytes at any model size in play — plus the cell
/// grids.
pub fn decide(
    src: &dyn LayerSource,
    basis: &OutputBasis,
    pool: &[(String, KeepSets)],
    g: Geom,
    cfg: &Config,
) -> Result<Decision, AperturbError> {
    if pool.is_empty() {
        return Err(AperturbError::Degenerate("the candidate pool is empty"));
    }
    if g.rows == 0 || g.rows > g.current_pos || g.rows > 32 {
        return Err(AperturbError::Degenerate(
            "rows must be in 1..=32 and no larger than the resident token count",
        ));
    }
    let d = g.q_dim();
    if basis.d() != d {
        return Err(AperturbError::BasisShape {
            layer: 0,
            got: basis.d(),
            want: d,
        });
    }
    let r = basis.rank();
    let n_c = pool.len();

    for (_, keep) in pool {
        keep.validate(g.current_pos)?;
    }
    let identity = KeepSets::identity(g.n_layers, g.n_kv_heads, g.current_pos);

    let mut grids: Vec<CellGrid> = (0..n_c)
        .map(|_| CellGrid::new(g.n_layers, g.rows))
        .collect();
    let mut visible = vec![u32::MAX; n_c];

    let mut z = vec![0.0f32; g.logit_len()];
    let mut x = vec![0.0f32; g.x_len()];
    let mut w_base = vec![0.0f32; g.rows * r];
    let mut w_cand = vec![0.0f32; g.rows * r];

    for l in 0..g.n_layers {
        let (q, k, v) = (src.query_rows(l), src.keys(l), src.values(l));
        kernel::logits_into(q, k, &mut z, g)?;

        // The reference: the same operator over the untouched cache, so common-mode rounding
        // cancels and an identity candidate lands on exactly zero.
        let kp_base = KeyPos::for_layer(&identity, l, g.current_pos, g.rows);
        kernel::attend_into(&z, &identity, l, &kp_base, v, &mut x, g)?;
        kernel::project_into(&x, basis.layer(l), &mut w_base, g.rows, d, r);

        for (c, (_, keep)) in pool.iter().enumerate() {
            let kp = KeyPos::for_layer(keep, l, g.current_pos, g.rows);
            visible[c] &= kp.visible_rows();
            kernel::attend_into(&z, keep, l, &kp, v, &mut x, g)?;
            kernel::project_into(&x, basis.layer(l), &mut w_cand, g.rows, d, r);
            grids[c].fill_layer(l, &w_base, &w_cand, r)?;
        }
    }

    let mut scored = Vec::with_capacity(n_c);
    for (c, (name, keep)) in pool.iter().enumerate() {
        let s = grids[c].aggregate(visible[c])?;
        scored.push(Scored {
            name: name.clone(),
            scores: s,
            visible_rows: visible[c],
            ragged: keep.is_ragged(),
            cells: cfg.keep_cells.then(|| grids[c].clone()),
        });
    }

    // Ties go to the earlier candidate, which makes the pool order the tie-break rather than
    // whatever order a sort happened to leave behind.
    let winner = scored.iter().enumerate().fold(0usize, |best, (i, s)| {
        if s.scores.get(cfg.readout) < scored[best].scores.get(cfg.readout) {
            i
        } else {
            best
        }
    });
    Ok(Decision { scored, winner })
}
