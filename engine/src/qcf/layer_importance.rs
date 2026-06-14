//! Layer Skip QCF: cosine-similarity-based layer importance.
//!
//! Measures each layer's contribution to hidden state transformation.
//! `importance(layer_i) = 1 - cosine_similarity(input_i, output_i)`
//!
//! QCF = Σ importance(skipped) / Σ importance(all)
//!
//! Built during prefill (1 pass), reused for all decode steps.

/// Output of `ImportanceCollector::build_with_raws()`:
/// `(table, x_means_per_layer, before_snapshots(snapshot, dim, count))`.
pub type ImportanceWithRaws = (
    ImportanceTable,
    Vec<Vec<f32>>,
    Vec<(Vec<f32>, usize, usize)>,
);

pub use crate::qcf_types::{ImportanceEntry, SubLayer};

/// Pre-computed importance table for all layers.
///
/// Built once during prefill via `ImportanceCollector`, then reused
/// to compute QCF for any skip pattern.
#[derive(Debug, Clone)]
pub struct ImportanceTable {
    entries: Vec<ImportanceEntry>,
    total_importance: f32,
}

impl ImportanceTable {
    /// Build table from collected entries.
    pub fn from_entries(entries: Vec<ImportanceEntry>) -> Self {
        let total_importance = entries.iter().map(|e| e.importance).sum();
        Self {
            entries,
            total_importance,
        }
    }

    /// Compute total OPR for skipped layers (sum of per-layer OPR).
    ///
    /// `skip_set`: list of `(layer_id, SubLayer)` pairs being skipped.
    /// Returns the sum of `opr` values for matched entries.
    pub fn compute_opr_skip(&self, skip_set: &[(usize, SubLayer)]) -> f32 {
        self.entries
            .iter()
            .filter(|e| skip_set.contains(&(e.layer_id, e.sublayer)))
            .map(|e| e.opr)
            .sum()
    }

    /// Compute QCF for a given skip set.
    ///
    /// `skip_set`: list of `(layer_id, SubLayer)` pairs being skipped.
    /// Returns `Σ importance(skipped) / Σ importance(all)`, in [0, 1].
    pub fn compute_qcf_weight(&self, skip_set: &[(usize, SubLayer)]) -> f32 {
        let _t = crate::qcf_timer!(QCF_LAYER_SKIP);
        if self.total_importance < 1e-8 || skip_set.is_empty() {
            return 0.0;
        }

        let skipped_importance: f32 = self
            .entries
            .iter()
            .filter(|e| skip_set.contains(&(e.layer_id, e.sublayer)))
            .map(|e| e.importance)
            .sum();

        (skipped_importance / self.total_importance).clamp(0.0, 1.0)
    }

    /// Estimate QCF for a target skip count by selecting the least
    /// important layers first. Layer 0 and the last layer are protected.
    ///
    /// Returns `(qcf_value, selected_skip_set)`.
    pub fn estimate_qcf_for_count(
        &self,
        skip_count: usize,
        num_layers: usize,
    ) -> (f32, Vec<(usize, SubLayer)>) {
        let _t = crate::qcf_timer!(QCF_LAYER_SKIP);
        if skip_count == 0 || self.entries.is_empty() {
            return (0.0, Vec::new());
        }

        let last_layer = if num_layers > 0 { num_layers - 1 } else { 0 };

        // Sort eligible entries by importance (ascending = cheapest to skip first)
        let mut eligible: Vec<&ImportanceEntry> = self
            .entries
            .iter()
            .filter(|e| e.layer_id != 0 && e.layer_id != last_layer)
            .collect();
        eligible.sort_by(|a, b| a.importance.partial_cmp(&b.importance).unwrap());

        let take = skip_count.min(eligible.len());
        let skip_set: Vec<(usize, SubLayer)> = eligible[..take]
            .iter()
            .map(|e| (e.layer_id, e.sublayer))
            .collect();

        let qcf = self.compute_qcf_weight(&skip_set);
        (qcf, skip_set)
    }

    /// Number of entries in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total importance across all entries.
    pub fn total_importance(&self) -> f32 {
        self.total_importance
    }

    /// Get all entries.
    pub fn entries(&self) -> &[ImportanceEntry] {
        &self.entries
    }
}

/// Collects per-layer hidden state snapshots during prefill to build
/// an `ImportanceTable`.
///
/// Usage in the layer loop:
/// ```ignore
/// let mut collector = ImportanceCollector::new();
/// for (i, layer) in layers.iter().enumerate() {
///     collector.snapshot_before(&x);
///     layer.forward(...)?;
///     collector.record_after(&x, i);
/// }
/// let table = collector.build();
/// ```
pub struct ImportanceCollector {
    entries: Vec<ImportanceEntry>,
    /// Mean-pooled hidden state before the current layer (`[dim]`).
    /// Always populated; used by the MeanPool formula.
    before_snapshot: Vec<f32>,
    /// Raw `[seq_len × dim]` hidden state before the current layer (batch=1).
    /// Populated only when `three_way` is true; used by ShortGPT BI.
    before_snapshot_raw: Vec<f32>,
    before_seq_len: usize,
    before_dim: usize,
    /// 3-way comparison mode: side-by-side measurement of MeanPool + ShortGptBi.
    three_way: bool,
    /// Primary formula whose value fills `ImportanceEntry::importance`.
    primary_formula: super::ImportanceFormula,
    /// Per-layer mean-pooled input cache for DpllmProxy (post-warmup
    /// `noise_table::compute_input_aware_epsilon`). Populated only when
    /// `three_way` is true; one [dim] vector per `snapshot_before` call.
    x_means: Vec<Vec<f32>>,
    /// Per-layer raw `[T × d]` snapshots for cascade attention forms (F4, F5).
    /// Populated only when `cache_raw_per_layer()` returns true (i.e. three_way
    /// compare mode or `DirectAttn` primary). One entry per `snapshot_before`
    /// call, chronological order. Stored as `(raw_data, seq_len, dim)`.
    before_snapshots_raw: Vec<(Vec<f32>, usize, usize)>,
}

impl ImportanceCollector {
    pub fn new() -> Self {
        Self::new_with_formula(super::ImportanceFormula::MeanPool, false)
    }

    /// Construct with a chosen primary formula and optional 3-way mode.
    /// In 3-way mode the collector additionally computes the ShortGptBi
    /// importance per layer and caches `x_means` for DpllmProxy.
    pub fn new_with_formula(formula: super::ImportanceFormula, three_way: bool) -> Self {
        Self {
            entries: Vec::new(),
            before_snapshot: Vec::new(),
            before_snapshot_raw: Vec::new(),
            before_seq_len: 0,
            before_dim: 0,
            three_way,
            primary_formula: formula,
            x_means: Vec::new(),
            before_snapshots_raw: Vec::new(),
        }
    }

    /// Whether per-layer raw `[T × d]` snapshots should be cached.
    /// True for 3-way compare mode or `DirectAttn` primary; false otherwise.
    fn cache_raw_per_layer(&self) -> bool {
        self.three_way || matches!(self.primary_formula, super::ImportanceFormula::DirectAttn)
    }

    /// Snapshot the hidden state before a layer processes it.
    ///
    /// For prefill, `x_data` is `[seq_len × dim]` row-major (batch=1).
    /// Always retains the mean-pooled vector (`[dim]`) for MeanPool.
    /// In 3-way mode also retains the raw `[seq_len × dim]` slice for
    /// ShortGptBi and pushes the mean to `x_means` for DpllmProxy.
    pub fn snapshot_before(&mut self, x_data: &[f32], seq_len: usize, dim: usize) {
        // Mean-pool path (always)
        self.before_snapshot.clear();
        self.before_snapshot.resize(dim, 0.0);
        if seq_len > 0 {
            for pos in 0..seq_len {
                let offset = pos * dim;
                for d in 0..dim {
                    if offset + d < x_data.len() {
                        self.before_snapshot[d] += x_data[offset + d];
                    }
                }
            }
            let scale = 1.0 / seq_len as f32;
            for v in &mut self.before_snapshot {
                *v *= scale;
            }
        }

        // Raw path (3-way only) — needed for ShortGPT BI per-token cosine
        if self.three_way {
            let total = seq_len.saturating_mul(dim).min(x_data.len());
            self.before_snapshot_raw.clear();
            self.before_snapshot_raw.extend_from_slice(&x_data[..total]);
            self.before_seq_len = seq_len;
            self.before_dim = dim;
            // x_means cache for DpllmProxy stage (`noise_table` post-warmup)
            self.x_means.push(self.before_snapshot.clone());
        }

        // Per-layer raw cache for cascade attention (F4, F5).
        // Independent of 3-way mode so DirectAttn primary works standalone.
        if self.cache_raw_per_layer() {
            let total = seq_len.saturating_mul(dim).min(x_data.len());
            self.before_snapshots_raw
                .push((x_data[..total].to_vec(), seq_len, dim));
        }
    }

    /// Record importance after a layer has processed the hidden state.
    ///
    /// Always populates `importance` from the primary formula and `opr`
    /// (mean-pool residual ratio). In 3-way mode also fills
    /// `importance_mean_pool` and `importance_shortgpt_bi`.
    pub fn record_after(
        &mut self,
        x_data: &[f32],
        seq_len: usize,
        dim: usize,
        layer_id: usize,
        sublayer: SubLayer,
    ) {
        // (1) Mean-pool after state — used by MeanPool formula + OPR
        let mut after = vec![0.0f32; dim];
        if seq_len > 0 {
            for pos in 0..seq_len {
                let offset = pos * dim;
                for d in 0..dim {
                    if offset + d < x_data.len() {
                        after[d] += x_data[offset + d];
                    }
                }
            }
            let scale = 1.0 / seq_len as f32;
            for v in &mut after {
                *v *= scale;
            }
        }
        let imp_mean_pool = (1.0 - cosine_similarity(&self.before_snapshot, &after)).max(0.0);
        let opr = residual_norm_ratio(&self.before_snapshot, &after);

        // (2) ShortGPT BI per-token cosine mean (3-way only)
        let imp_shortgpt_bi: Option<f32> = if self.three_way {
            let t = self.before_seq_len.min(seq_len);
            let d = self.before_dim.min(dim);
            if t == 0 || d == 0 {
                Some(0.0)
            } else {
                let mut sum_cos = 0.0f32;
                let mut valid_t: u32 = 0;
                for pos in 0..t {
                    let off = pos * d;
                    let be = off + d;
                    if be > self.before_snapshot_raw.len() || be > x_data.len() {
                        break;
                    }
                    let before_tok = &self.before_snapshot_raw[off..be];
                    let after_tok = &x_data[off..be];
                    let mut bm = 0.0f32;
                    for &b in before_tok.iter().take(d) {
                        bm += b * b;
                    }
                    if bm > 1e-24 {
                        sum_cos += cosine_similarity(before_tok, after_tok);
                        valid_t += 1;
                    }
                }
                let v = if valid_t > 0 {
                    (1.0 - sum_cos / valid_t as f32).max(0.0)
                } else {
                    0.0
                };
                Some(v)
            }
        } else {
            None
        };

        // (3) Primary importance selection
        let importance = match self.primary_formula {
            super::ImportanceFormula::MeanPool => imp_mean_pool,
            super::ImportanceFormula::ShortGptBi => imp_shortgpt_bi.unwrap_or(imp_mean_pool),
            // DP-LLM variants + DirectAttn: real signal is computed post-warmup
            // in `noise_table`. Importance stays a constant placeholder here so
            // the swap_key composition is well-defined for all variants.
            super::ImportanceFormula::DpllmProxy
            | super::ImportanceFormula::DpllmMulti
            | super::ImportanceFormula::DpllmAbs
            | super::ImportanceFormula::DpllmQcf
            | super::ImportanceFormula::DirectAttn => 1.0,
        };

        let (mp_field, sb_field) = if self.three_way {
            (Some(imp_mean_pool), imp_shortgpt_bi)
        } else {
            (None, None)
        };

        self.entries.push(ImportanceEntry {
            layer_id,
            sublayer,
            importance,
            opr,
            importance_mean_pool: mp_field,
            importance_shortgpt_bi: sb_field,
        });
    }

    /// Build the final `ImportanceTable`.
    pub fn build(self) -> ImportanceTable {
        ImportanceTable::from_entries(self.entries)
    }

    /// Build the final `ImportanceTable` and return the cached x_means
    /// (one `[dim]` vector per snapshot_before call, in chronological order).
    /// Use this in 3-way mode to feed `noise_table::compute_input_aware_epsilon`.
    pub fn build_with_xmeans(self) -> (ImportanceTable, Vec<Vec<f32>>) {
        (ImportanceTable::from_entries(self.entries), self.x_means)
    }

    /// Whether 3-way comparison mode is active.
    pub fn is_three_way(&self) -> bool {
        self.three_way
    }

    /// Read-only access to cached x_means (3-way mode only).
    pub fn x_means(&self) -> &[Vec<f32>] {
        &self.x_means
    }

    /// Read-only access to per-layer raw `[T × d]` snapshots. One entry per
    /// `snapshot_before` call, chronological order. Empty unless
    /// `cache_raw_per_layer()` returned true at construction. Each tuple is
    /// `(raw_data_row_major, seq_len, dim)`.
    pub fn raws_per_layer(&self) -> &[(Vec<f32>, usize, usize)] {
        &self.before_snapshots_raw
    }

    /// Consume the collector and return `(ImportanceTable, x_means, raws_per_layer)`.
    /// Use this in DirectAttn mode to feed
    /// `noise_table::compute_cascade_attn_perturbation`.
    pub fn build_with_raws(self) -> ImportanceWithRaws {
        (
            ImportanceTable::from_entries(self.entries),
            self.x_means,
            self.before_snapshots_raw,
        )
    }
}

impl Default for ImportanceCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Trait impls for L2 abstraction (S-3b-4 γ-2 trait inversion) ─────

impl crate::qcf_collector::ImportanceCollect for ImportanceCollector {
    fn snapshot_before(&mut self, x: &[f32], seq_len: usize, dim: usize) {
        ImportanceCollector::snapshot_before(self, x, seq_len, dim);
    }

    fn record_after(
        &mut self,
        x: &[f32],
        seq_len: usize,
        hidden_size: usize,
        layer: usize,
        sublayer: SubLayer,
    ) {
        ImportanceCollector::record_after(self, x, seq_len, hidden_size, layer, sublayer);
    }
}

impl crate::qcf_collector::ImportanceLookup for ImportanceTable {
    fn is_empty(&self) -> bool {
        ImportanceTable::is_empty(self)
    }

    fn entries(&self) -> &[ImportanceEntry] {
        ImportanceTable::entries(self)
    }
}

/// OPR for layer skip: `||output - input|| / ||input||`.
///
/// Returns 0.0 if `input` has zero magnitude.
fn residual_norm_ratio(before: &[f32], after: &[f32]) -> f32 {
    let len = before.len().min(after.len());
    if len == 0 {
        return 0.0;
    }
    let mut diff_sq = 0.0f32;
    let mut before_sq = 0.0f32;
    for i in 0..len {
        let d = after[i] - before[i];
        diff_sq += d * d;
        before_sq += before[i] * before[i];
    }
    let denom = before_sq.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        diff_sq.sqrt() / denom
    }
}

/// Cosine similarity between two float slices.
///
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let cos = cosine_similarity(&a, &a);
        assert!((cos - 1.0).abs() < 1e-6, "identical vectors: cos={cos}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let cos = cosine_similarity(&a, &b);
        assert!(cos.abs() < 1e-6, "orthogonal: cos={cos}");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let cos = cosine_similarity(&a, &b);
        assert!((cos - (-1.0)).abs() < 1e-6, "opposite: cos={cos}");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        let cos = cosine_similarity(&a, &b);
        assert_eq!(cos, 0.0);
    }

    #[test]
    fn test_importance_table_empty_skip() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.5,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.3,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
        ];
        let table = ImportanceTable::from_entries(entries);
        assert_eq!(table.compute_qcf_weight(&[]), 0.0);
    }

    #[test]
    fn test_importance_table_full_skip() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.5,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.3,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
        ];
        let table = ImportanceTable::from_entries(entries);
        let qcf = table.compute_qcf_weight(&[(0, SubLayer::Full), (1, SubLayer::Full)]);
        assert!((qcf - 1.0).abs() < 1e-6, "full skip: qcf={qcf}");
    }

    #[test]
    fn test_importance_table_partial_skip() {
        // Guide §3-3 simplified example
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.42,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.08,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 2,
                sublayer: SubLayer::Full,
                importance: 0.31,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 3,
                sublayer: SubLayer::Full,
                importance: 0.05,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
        ];
        let table = ImportanceTable::from_entries(entries);
        // total = 0.86
        // skip layers 1 and 3 (low importance): 0.08 + 0.05 = 0.13
        let qcf = table.compute_qcf_weight(&[(1, SubLayer::Full), (3, SubLayer::Full)]);
        let expected = 0.13 / 0.86;
        assert!(
            (qcf - expected).abs() < 0.01,
            "expected ~{expected:.3}, got {qcf:.3}"
        );
    }

    #[test]
    fn test_estimate_qcf_protects_first_last() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.9,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.1,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 2,
                sublayer: SubLayer::Full,
                importance: 0.2,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 3,
                sublayer: SubLayer::Full,
                importance: 0.8,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
        ];
        let table = ImportanceTable::from_entries(entries);

        // Skip 2 layers from 4-layer model: layer 0 and 3 protected
        let (qcf, skip_set) = table.estimate_qcf_for_count(2, 4);
        // Only layers 1 and 2 eligible → both skipped
        assert_eq!(skip_set.len(), 2);
        assert!(skip_set.contains(&(1, SubLayer::Full)));
        assert!(skip_set.contains(&(2, SubLayer::Full)));
        // QCF = (0.1 + 0.2) / 2.0 = 0.15
        assert!((qcf - 0.15).abs() < 0.01, "expected ~0.15, got {qcf}");
    }

    #[test]
    fn test_estimate_qcf_selects_least_important() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.9,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.5,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 2,
                sublayer: SubLayer::Full,
                importance: 0.1,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 3,
                sublayer: SubLayer::Full,
                importance: 0.3,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 4,
                sublayer: SubLayer::Full,
                importance: 0.7,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
        ];
        let table = ImportanceTable::from_entries(entries);

        // Skip 1 from 5-layer model: should pick layer 2 (importance=0.1)
        let (_, skip_set) = table.estimate_qcf_for_count(1, 5);
        assert_eq!(skip_set.len(), 1);
        assert_eq!(skip_set[0], (2, SubLayer::Full));
    }

    #[test]
    fn test_collector_basic() {
        let dim = 4;
        let seq_len = 2;

        // Before: [[1,0,0,0], [1,0,0,0]] → mean = [1,0,0,0]
        let before_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        // After: [[0,1,0,0], [0,1,0,0]] → mean = [0,1,0,0]
        let after_data = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let mut collector = ImportanceCollector::new();
        collector.snapshot_before(&before_data, seq_len, dim);
        collector.record_after(&after_data, seq_len, dim, 0, SubLayer::Full);

        let table = collector.build();
        assert_eq!(table.len(), 1);
        // cos_sim([1,0,0,0], [0,1,0,0]) = 0 → importance = 1.0
        let entry = &table.entries()[0];
        assert!(
            (entry.importance - 1.0).abs() < 0.01,
            "imp={}",
            entry.importance
        );
    }

    #[test]
    fn test_collector_identity_layer() {
        let dim = 4;
        let seq_len = 1;
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let mut collector = ImportanceCollector::new();
        collector.snapshot_before(&data, seq_len, dim);
        // Same data → identity layer → importance ≈ 0
        collector.record_after(&data, seq_len, dim, 5, SubLayer::Full);

        let table = collector.build();
        let entry = &table.entries()[0];
        assert!(
            entry.importance < 0.01,
            "identity layer imp={}",
            entry.importance
        );
    }

    #[test]
    fn test_sublayer_separation() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Attention,
                importance: 0.4,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Mlp,
                importance: 0.1,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Attention,
                importance: 0.3,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Mlp,
                importance: 0.2,
                opr: 0.0,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
        ];
        let table = ImportanceTable::from_entries(entries);
        // total = 1.0
        // Skip only MLP of layer 0 and attention of layer 1
        let qcf = table.compute_qcf_weight(&[(0, SubLayer::Mlp), (1, SubLayer::Attention)]);
        // (0.1 + 0.3) / 1.0 = 0.4
        assert!((qcf - 0.4).abs() < 1e-6, "sublayer qcf={qcf}");
    }

    // --- OPR tests ---

    #[test]
    fn test_residual_norm_ratio_identity() {
        // output == input → residual = 0 → OPR = 0
        let v = vec![1.0, 2.0, 3.0];
        let opr = residual_norm_ratio(&v, &v);
        assert!(opr.abs() < 1e-6, "identity opr={opr}");
    }

    #[test]
    fn test_residual_norm_ratio_orthogonal() {
        // before=[1,0], after=[0,1]: diff=[−1,1], ||diff||=√2, ||before||=1 → OPR=√2
        let before = vec![1.0, 0.0];
        let after = vec![0.0, 1.0];
        let opr = residual_norm_ratio(&before, &after);
        let expected = 2.0_f32.sqrt();
        assert!(
            (opr - expected).abs() < 1e-5,
            "orthogonal opr={opr}, expected {expected}"
        );
    }

    #[test]
    fn test_residual_norm_ratio_zero_input() {
        // zero input → OPR = 0 (no division by zero)
        let before = vec![0.0, 0.0, 0.0];
        let after = vec![1.0, 2.0, 3.0];
        let opr = residual_norm_ratio(&before, &after);
        assert_eq!(opr, 0.0, "zero input opr={opr}");
    }

    #[test]
    fn test_compute_opr_skip() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.5,
                opr: 0.2,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.3,
                opr: 0.4,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
            ImportanceEntry {
                layer_id: 2,
                sublayer: SubLayer::Full,
                importance: 0.2,
                opr: 0.1,
                importance_mean_pool: None,
                importance_shortgpt_bi: None,
            },
        ];
        let table = ImportanceTable::from_entries(entries);

        // Skip layers 0 and 2 → OPR sum = 0.2 + 0.1 = 0.3
        let opr_sum = table.compute_opr_skip(&[(0, SubLayer::Full), (2, SubLayer::Full)]);
        assert!((opr_sum - 0.3).abs() < 1e-6, "opr_sum={opr_sum}");

        // Empty skip set → 0.0
        assert_eq!(table.compute_opr_skip(&[]), 0.0);
    }

    #[test]
    fn test_collector_records_opr() {
        let dim = 4;
        let seq_len = 1;

        // before=[1,0,0,0], after=[0,1,0,0]
        // residual=[-1,1,0,0], ||residual||=√2, ||before||=1 → OPR=√2
        let before_data = vec![1.0, 0.0, 0.0, 0.0];
        let after_data = vec![0.0, 1.0, 0.0, 0.0];

        let mut collector = ImportanceCollector::new();
        collector.snapshot_before(&before_data, seq_len, dim);
        collector.record_after(&after_data, seq_len, dim, 0, SubLayer::Full);

        let table = collector.build();
        let entry = &table.entries()[0];
        let expected_opr = 2.0_f32.sqrt();
        assert!(
            (entry.opr - expected_opr).abs() < 1e-5,
            "collector opr={}, expected {expected_opr}",
            entry.opr
        );
        // importance should still be 1.0 (orthogonal vectors)
        assert!(
            (entry.importance - 1.0).abs() < 0.01,
            "importance should be 1.0, got {}",
            entry.importance
        );
    }

    #[test]
    fn test_importance_formula_as_str() {
        use super::super::ImportanceFormula;
        assert_eq!(ImportanceFormula::MeanPool.as_str(), "mean_pool");
        assert_eq!(ImportanceFormula::ShortGptBi.as_str(), "shortgpt_bi");
        assert_eq!(ImportanceFormula::DpllmProxy.as_str(), "dpllm_proxy");
    }

    #[test]
    fn test_collector_single_mode_no_extras() {
        // Default `new()` is mean_pool / single mode → optional 3-way fields stay None.
        let dim = 4;
        let seq_len = 2;
        let before = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let after = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let mut collector = ImportanceCollector::new();
        collector.snapshot_before(&before, seq_len, dim);
        collector.record_after(&after, seq_len, dim, 0, SubLayer::Full);

        let table = collector.build();
        let e = &table.entries()[0];
        assert!(
            e.importance_mean_pool.is_none(),
            "single mode must leave mean_pool=None"
        );
        assert!(
            e.importance_shortgpt_bi.is_none(),
            "single mode must leave shortgpt_bi=None"
        );
    }

    #[test]
    fn test_collector_three_way_both_formulas_populated() {
        use super::super::ImportanceFormula;
        let dim = 2;
        let seq_len = 2;
        // before = [[1,0], [1,0]]   after = [[0,1], [1,0]]
        //  → per-token cos = (0, 1)  → shortgpt_bi mean = 0.5  → importance_shortgpt_bi = 0.5
        //  → mean_pool(before)=[1,0], mean_pool(after)=[0.5,0.5]
        //    cos = 1/√2 ≈ 0.7071     → importance_mean_pool ≈ 0.2929
        let before = vec![1.0, 0.0, 1.0, 0.0];
        let after = vec![0.0, 1.0, 1.0, 0.0];

        let mut collector =
            ImportanceCollector::new_with_formula(ImportanceFormula::MeanPool, true);
        collector.snapshot_before(&before, seq_len, dim);
        collector.record_after(&after, seq_len, dim, 0, SubLayer::Full);

        let table = collector.build();
        let e = &table.entries()[0];
        let mp = e.importance_mean_pool.expect("mean_pool must be Some");
        let sb = e.importance_shortgpt_bi.expect("shortgpt_bi must be Some");
        assert!((mp - 0.2929).abs() < 1e-3, "mean_pool ≈ 0.2929, got {mp}");
        assert!((sb - 0.5).abs() < 1e-3, "shortgpt_bi ≈ 0.5, got {sb}");
        // Primary is MeanPool → entry.importance == importance_mean_pool
        assert!(
            (e.importance - mp).abs() < 1e-6,
            "primary importance must match selected formula"
        );
    }

    #[test]
    fn test_collector_three_way_caches_xmeans() {
        use super::super::ImportanceFormula;
        let dim = 3;
        let seq_len = 2;
        // before tokens = [[2,4,6], [4,6,8]]  → mean = [3,5,7]
        let before = vec![2.0, 4.0, 6.0, 4.0, 6.0, 8.0];
        let after = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut collector =
            ImportanceCollector::new_with_formula(ImportanceFormula::MeanPool, true);
        collector.snapshot_before(&before, seq_len, dim);
        collector.record_after(&after, seq_len, dim, 0, SubLayer::Full);

        let (_table, x_means) = collector.build_with_xmeans();
        assert_eq!(x_means.len(), 1);
        let mean = &x_means[0];
        assert!((mean[0] - 3.0).abs() < 1e-5);
        assert!((mean[1] - 5.0).abs() < 1e-5);
        assert!((mean[2] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_collector_primary_formula_shortgpt_selects_importance() {
        use super::super::ImportanceFormula;
        let dim = 2;
        let seq_len = 2;
        let before = vec![1.0, 0.0, 1.0, 0.0];
        let after = vec![0.0, 1.0, 1.0, 0.0];

        let mut collector =
            ImportanceCollector::new_with_formula(ImportanceFormula::ShortGptBi, true);
        collector.snapshot_before(&before, seq_len, dim);
        collector.record_after(&after, seq_len, dim, 0, SubLayer::Full);

        let table = collector.build();
        let e = &table.entries()[0];
        let sb = e.importance_shortgpt_bi.expect("shortgpt_bi must be Some");
        // Primary is ShortGptBi → entry.importance == importance_shortgpt_bi
        assert!(
            (e.importance - sb).abs() < 1e-6,
            "primary importance must match ShortGptBi when selected"
        );
    }
}
