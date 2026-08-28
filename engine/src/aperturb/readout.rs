//! How a projected row pair becomes one number.
//!
//! Two axes, deliberately kept apart because they answer different questions and the reference
//! sweeps both:
//!
//! - the **value** of a cell — how far the candidate moved that row's contribution to the residual
//!   stream — as either the relative displacement `‖Õ − O‖ / ‖O‖` or the direction change
//!   `1 − cos(O, Õ)`;
//! - the **aggregation** of the `L × R` cell grid into a score — arithmetic mean, or RMS, which
//!   weights the few cells a compression actually damages instead of diluting them.
//!
//! Both are computed from `w = X (V_r Σ_r)` in `r`-space; see [`super::kernel`] for why that is
//! exact rather than an approximation.
//!
//! ⚠ `1 − cos` is a subtraction of two nearly equal quantities: for a near-lossless candidate the
//! cosine is `1 − ε` with `ε` far below f32's resolution, so the *relative* accuracy of `dcos`
//! collapses while `l2` stays well conditioned across the same range. That is a property of the
//! quantity, not of this implementation — the reference has it too — so a parity comparison on
//! `dcos` needs an absolute floor, and `l2` is the better default.

use std::fmt;

/// What one cell measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellValue {
    /// `‖Õ − O‖ / ‖O‖` — direction and magnitude together.
    L2,
    /// `1 − cos(O, Õ)` — direction only.
    Dcos,
}

/// How the cell grid collapses to a score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellAgg {
    /// Arithmetic mean over the grid.
    Mean,
    /// `√(Σ x² / n)`. Weights damaged cells more heavily than the mean, with no free parameter,
    /// and — unlike a plain Euclidean norm or a max — it does not change when the number of valid
    /// cells does.
    Rms,
}

/// A `(value, aggregation)` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readout {
    pub value: CellValue,
    pub agg: CellAgg,
}

impl Readout {
    /// The four combinations, in the order [`ReadoutSet`] stores them.
    pub const ALL: [Readout; 4] = [
        Readout {
            value: CellValue::L2,
            agg: CellAgg::Mean,
        },
        Readout {
            value: CellValue::L2,
            agg: CellAgg::Rms,
        },
        Readout {
            value: CellValue::Dcos,
            agg: CellAgg::Mean,
        },
        Readout {
            value: CellValue::Dcos,
            agg: CellAgg::Rms,
        },
    ];

    /// The suffix the reference's metric keys carry.
    ///
    /// Note the asymmetry: the mean is the **bare** value name, not `*_mean`. Spelling it
    /// `"l2_mean"` would compare against a key that does not exist, and a JSON lookup answers a
    /// missing key with `None`, not with an error.
    pub fn suffix(self) -> &'static str {
        match (self.value, self.agg) {
            (CellValue::L2, CellAgg::Mean) => "l2",
            (CellValue::L2, CellAgg::Rms) => "l2_rms",
            (CellValue::Dcos, CellAgg::Mean) => "dcos",
            (CellValue::Dcos, CellAgg::Rms) => "dcos_rms",
        }
    }

    /// Inverse of [`Self::suffix`].
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.suffix() == s)
    }

    /// Position in a [`ReadoutSet`].
    #[inline]
    fn slot(self) -> usize {
        match (self.value, self.agg) {
            (CellValue::L2, CellAgg::Mean) => 0,
            (CellValue::L2, CellAgg::Rms) => 1,
            (CellValue::Dcos, CellAgg::Mean) => 2,
            (CellValue::Dcos, CellAgg::Rms) => 3,
        }
    }
}

impl Default for Readout {
    /// `l2` with RMS aggregation.
    fn default() -> Self {
        Self {
            value: CellValue::L2,
            agg: CellAgg::Rms,
        }
    }
}

impl fmt::Display for Readout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.suffix())
    }
}

/// All four scores for one candidate, plus the grid they came from.
///
/// Emitting the whole set costs three extra reductions over a grid of a few hundred cells, and it
/// buys two things: a parity comparison that a bug happening to cancel in one readout cannot pass,
/// and `mean ≤ rms` as a free in-run assertion that the two aggregations read the *same* cells.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadoutSet {
    scores: [f32; 4],
    /// Cells that entered the aggregation — `n_layers × visible rows`. Two scores are comparable
    /// only when this matches, so it travels with them.
    pub n_cells: usize,
}

impl ReadoutSet {
    /// The score for one readout.
    #[inline]
    pub fn get(&self, r: Readout) -> f32 {
        self.scores[r.slot()]
    }
}

/// The per-cell values of one candidate, `[layer][row]`, before any row is dropped.
///
/// Blind rows are kept in the grid so the trailing-row slice stays aligned across candidates; they
/// are excluded at the reduction instead.
#[derive(Clone, Debug, PartialEq)]
pub struct CellGrid {
    pub n_layers: usize,
    pub rows: usize,
    /// `‖Õ − O‖ / ‖O‖` per `(layer, row)`.
    pub l2: Vec<f32>,
    /// `1 − cos(O, Õ)` per `(layer, row)`.
    pub dcos: Vec<f32>,
}

impl CellGrid {
    pub fn new(n_layers: usize, rows: usize) -> Self {
        Self {
            n_layers,
            rows,
            l2: vec![0.0; n_layers * rows],
            dcos: vec![0.0; n_layers * rows],
        }
    }

    /// Fill one layer's row from the baseline and candidate `r`-space rows.
    ///
    /// `w_base` / `w_cand` are `[rows][r]`. A zero-norm baseline row means the reference quantity
    /// is undefined; the reference divides by it and raises on the resulting non-finite, so this
    /// reports rather than clamping to a plausible zero.
    pub fn fill_layer(
        &mut self,
        layer: usize,
        w_base: &[f32],
        w_cand: &[f32],
        r: usize,
    ) -> Result<(), ReadoutError> {
        for t in 0..self.rows {
            let a = &w_base[t * r..(t + 1) * r];
            let b = &w_cand[t * r..(t + 1) * r];
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            let mut dot = 0.0f32;
            let mut diff = 0.0f32;
            for k in 0..r {
                na += a[k] * a[k];
                nb += b[k] * b[k];
                dot += a[k] * b[k];
                let d = b[k] - a[k];
                diff += d * d;
            }
            let (na, nb, diff) = (na.sqrt(), nb.sqrt(), diff.sqrt());
            // `<= 0.0` is false for NaN, so a NaN norm falls through to the finiteness check below
            // and is reported as such rather than misfiled as a zero baseline.
            if na <= 0.0 {
                return Err(ReadoutError::ZeroBaseline { layer, row: t });
            }
            let i = layer * self.rows + t;
            self.l2[i] = diff / na;
            self.dcos[i] = 1.0 - dot / (na * nb);
            if !self.l2[i].is_finite() || !self.dcos[i].is_finite() {
                return Err(ReadoutError::NonFinite { layer, row: t });
            }
        }
        Ok(())
    }

    /// Collapse the grid to all four scores, over the rows in `visible` (a bitmask over `row`).
    pub fn aggregate(&self, visible: u32) -> Result<ReadoutSet, ReadoutError> {
        let n_vis = (0..self.rows.min(32))
            .filter(|&t| visible & (1 << t) != 0)
            .count();
        if n_vis == 0 {
            return Err(ReadoutError::NoVisibleRows);
        }
        let n = self.n_layers * n_vis;
        let inv = 1.0 / n as f32;
        let mut scores = [0.0f32; 4];
        for (slot, src) in [(0usize, &self.l2), (2, &self.dcos)] {
            let mut sum = 0.0f32;
            let mut sq = 0.0f32;
            for l in 0..self.n_layers {
                for t in 0..self.rows {
                    if visible & (1 << t) == 0 {
                        continue;
                    }
                    let x = src[l * self.rows + t];
                    sum += x;
                    sq += x * x;
                }
            }
            scores[slot] = sum * inv;
            scores[slot + 1] = (sq * inv).sqrt();
        }
        Ok(ReadoutSet { scores, n_cells: n })
    }
}

/// Why a readout could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadoutError {
    /// The baseline row has zero norm, so every relative quantity is undefined.
    ZeroBaseline { layer: usize, row: usize },
    /// A cell came out non-finite. Fatal, because `max(0.0, NaN) == 0.0` would let it pass a
    /// threshold gate unnoticed.
    NonFinite { layer: usize, row: usize },
    /// Every row was blind in some layer, so there is nothing to average.
    NoVisibleRows,
}

impl fmt::Display for ReadoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBaseline { layer, row } => write!(
                f,
                "aperturb: baseline row (layer {layer}, row {row}) has zero norm — the relative \
                 readout is undefined there"
            ),
            Self::NonFinite { layer, row } => {
                write!(f, "aperturb: non-finite cell at (layer {layer}, row {row})")
            }
            Self::NoVisibleRows => f.write_str(
                "aperturb: no query row sees a retained key in every layer and head — the \
                 candidate leaves nothing to measure against",
            ),
        }
    }
}

impl std::error::Error for ReadoutError {}

/// The reference's rank-fraction token: the fraction as a percentage, with `.` written `p`.
///
/// `1/128` → `0.78125%` → `"0p78125"`, which is how the metric key spells the canonical rank.
pub fn frac_token(frac: f64) -> String {
    let pct = frac * 100.0;
    let s = format!("{pct}");
    s.replace('.', "p")
}

/// The reference's metric key: `aperturb_prev{rows}[_wo{tok}]_{suffix}`.
///
/// `wo_frac` of `None` is the untruncated arm, which carries no rank token.
pub fn metric_key(rows: usize, wo_frac: Option<f64>, r: Readout) -> String {
    match wo_frac {
        Some(f) => format!("aperturb_prev{rows}_wo{}_{}", frac_token(f), r.suffix()),
        None => format!("aperturb_prev{rows}_{}", r.suffix()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffixes_round_trip_and_the_mean_is_the_bare_name() {
        for r in Readout::ALL {
            assert_eq!(Readout::parse(r.suffix()), Some(r));
        }
        assert_eq!(
            Readout {
                value: CellValue::L2,
                agg: CellAgg::Mean
            }
            .suffix(),
            "l2"
        );
        assert_eq!(Readout::default().suffix(), "l2_rms");
    }

    #[test]
    fn rank_tokens_match_the_five_swept_fractions() {
        assert_eq!(frac_token(0.0625), "6p25");
        assert_eq!(frac_token(0.03125), "3p125");
        assert_eq!(frac_token(0.015625), "1p5625");
        assert_eq!(frac_token(0.0078125), "0p78125");
        assert_eq!(frac_token(0.00390625), "0p390625");
    }

    #[test]
    fn metric_keys_match_the_reference_spelling() {
        assert_eq!(
            metric_key(16, Some(0.0078125), Readout::default()),
            "aperturb_prev16_wo0p78125_l2_rms"
        );
        assert_eq!(
            metric_key(
                16,
                Some(0.0078125),
                Readout {
                    value: CellValue::Dcos,
                    agg: CellAgg::Rms
                }
            ),
            "aperturb_prev16_wo0p78125_dcos_rms"
        );
        assert_eq!(
            metric_key(
                16,
                None,
                Readout {
                    value: CellValue::L2,
                    agg: CellAgg::Mean
                }
            ),
            "aperturb_prev16_l2"
        );
    }

    #[test]
    fn mean_never_exceeds_rms_and_both_read_the_same_cells() {
        let mut g = CellGrid::new(2, 3);
        g.l2 = vec![0.1, 0.4, 0.0, 0.2, 0.9, 0.0];
        g.dcos = vec![0.01, 0.2, 0.0, 0.05, 0.5, 0.0];
        let all = g.aggregate(0b111).unwrap();
        assert_eq!(all.n_cells, 6);
        for v in [CellValue::L2, CellValue::Dcos] {
            let mean = all.get(Readout {
                value: v,
                agg: CellAgg::Mean,
            });
            let rms = all.get(Readout {
                value: v,
                agg: CellAgg::Rms,
            });
            assert!(mean <= rms + 1e-7, "{mean} > {rms}");
        }
    }

    #[test]
    fn dropping_a_row_changes_the_denominator_not_just_the_sum() {
        let mut g = CellGrid::new(1, 2);
        g.l2 = vec![0.4, 100.0];
        g.dcos = vec![0.0, 0.0];
        let kept = g.aggregate(0b01).unwrap();
        assert_eq!(kept.n_cells, 1);
        assert!(
            (kept.get(Readout {
                value: CellValue::L2,
                agg: CellAgg::Mean
            }) - 0.4)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn an_all_blind_candidate_is_an_error_not_a_zero() {
        let g = CellGrid::new(1, 2);
        assert_eq!(g.aggregate(0), Err(ReadoutError::NoVisibleRows));
    }

    #[test]
    fn a_zero_baseline_row_is_reported() {
        let mut g = CellGrid::new(1, 1);
        assert_eq!(
            g.fill_layer(0, &[0.0, 0.0], &[1.0, 1.0], 2),
            Err(ReadoutError::ZeroBaseline { layer: 0, row: 0 })
        );
    }

    #[test]
    fn an_identical_row_reads_as_no_perturbation() {
        let mut g = CellGrid::new(1, 1);
        let w = [0.3f32, -1.2, 4.0];
        g.fill_layer(0, &w, &w, 3).unwrap();
        assert_eq!(g.l2[0], 0.0);
        assert!(
            g.dcos[0].abs() < 1e-6,
            "dcos {} should be f32 noise",
            g.dcos[0]
        );
    }
}
