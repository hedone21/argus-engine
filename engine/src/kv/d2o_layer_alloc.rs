//! D2O layer-level dynamic allocation.
//!
//! Implements the per-layer KV cache budget allocation algorithm from the D2O paper.
//! During prefill, each layer's attention column-sums are collected to compute the
//! variance of attention patterns. Layers with low variance (dense, uniform attention)
//! receive larger budgets; layers with high variance (sparse, focused attention) receive
//! smaller budgets.
//!
//! Reference: <https://github.com/AIoT-MLSys-Lab/d2o>
//!
//! # Usage
//!
//! 1. Create before prefill: `D2OVarianceCollector::new(...)`
//! 2. For each layer during prefill: `collector.collect_layer(layer_idx, q, k, ...)`
//! 3. After prefill: `collector.compute_budgets(hh_ratio, recent_ratio)` → per-layer ratios

/// Collects per-layer attention column-sums during prefill and computes
/// per-layer budgets for D2O layer-level dynamic allocation.
///
/// Usage:
/// 1. Create before prefill: `D2OVarianceCollector::new(...)`
/// 2. For each layer during prefill: `collector.collect_layer(layer_idx, q, k, ...)`
/// 3. After prefill: `collector.compute_budgets(hh_ratio, recent_ratio)` → per-layer ratios
pub struct D2OVarianceCollector {
    /// Per-layer column sums: column_sums[layer][h * seq_len + j]
    /// where h = kv_head index, j = key position.
    column_sums: Vec<Vec<f32>>,
    n_layers: usize,
    n_kv_heads: usize,
    n_heads_q: usize,
    head_dim: usize,
    seq_len: usize,
}

impl D2OVarianceCollector {
    /// Create a new collector.
    ///
    /// `seq_len` is the prefill sequence length; it determines the column-sum buffer size.
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        n_heads_q: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Self {
        let column_sums = vec![vec![0.0f32; n_kv_heads * seq_len]; n_layers];
        Self {
            column_sums,
            n_layers,
            n_kv_heads,
            n_heads_q,
            head_dim,
            seq_len,
        }
    }

    /// Compute per-head attention column-sums for a single layer.
    ///
    /// For each KV head h and query position i (causal: j <= i):
    ///   column_sum[h, j] += softmax(Q[i, q_h] · K[j, h]^T / sqrt(d))[j] / n_rep
    ///
    /// where n_rep = n_heads_q / n_kv_heads (GQA grouping).
    ///
    /// Q layout: position-major, i.e. Q[i * q_stride + q_h * head_dim + d]
    /// K layout: position-major, i.e. K[j * k_stride + kv_h * kv_head_stride + d]
    ///   (for SeqMajor: k_stride = n_kv_heads * head_dim, kv_head_stride = head_dim)
    ///   (for HeadMajor: k_stride depends on layout, kv_head_stride = capacity * head_dim)
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::too_many_arguments)]
    pub fn collect_layer(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        cache_seq_len: usize,
        q_stride: usize,
        k_stride: usize,
        kv_head_stride: usize,
        start_pos: usize,
    ) {
        assert!(layer_idx < self.n_layers, "layer_idx out of bounds");

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let n_rep = self.n_heads_q.checked_div(self.n_kv_heads).unwrap_or(1);
        let col_sums = &mut self.column_sums[layer_idx];

        for kv_h in 0..self.n_kv_heads {
            for rep in 0..n_rep {
                let q_h = kv_h * n_rep + rep;
                for i in 0..seq_len {
                    let global_i = i + start_pos;
                    // Causal masking: only attend to positions <= global_i,
                    // but also bounded by cache_seq_len.
                    let valid_len = (global_i + 1).min(cache_seq_len);

                    let mut max_s = f32::NEG_INFINITY;
                    let mut scores = vec![0.0f32; valid_len];

                    for j in 0..valid_len {
                        let mut dot = 0.0f32;
                        let q_off = i * q_stride + q_h * self.head_dim;
                        let k_off = j * k_stride + kv_h * kv_head_stride;
                        for d in 0..self.head_dim {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        scores[j] = dot * scale;
                        if scores[j] > max_s {
                            max_s = scores[j];
                        }
                    }

                    // Softmax (numerically stable)
                    let mut sum_exp = 0.0f32;
                    for j in 0..valid_len {
                        scores[j] = (scores[j] - max_s).exp();
                        sum_exp += scores[j];
                    }

                    // Accumulate column sums (normalized by n_rep for GQA)
                    if sum_exp > 0.0 {
                        let inv_sum = 1.0 / (sum_exp * n_rep as f32);
                        for j in 0..valid_len {
                            col_sums[kv_h * self.seq_len + j] += scores[j] * inv_sum;
                        }
                    }
                }
            }
        }
    }

    /// Compute per-layer normalized variance from collected column-sums.
    ///
    /// Steps (matching official D2O code):
    /// 1. Per-head variance of column sums across key positions
    /// 2. Min-max normalize across heads → [0, 1]
    /// 3. Mean across heads → scalar per layer
    ///
    /// Returns: `Vec<f32>` of length `n_layers`, each value in [0, 1].
    #[allow(clippy::needless_range_loop)]
    pub fn compute_variances(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.n_layers);

        for layer in 0..self.n_layers {
            let mut head_variances = Vec::with_capacity(self.n_kv_heads);

            for h in 0..self.n_kv_heads {
                let start = h * self.seq_len;
                let end = start + self.seq_len;
                let slice = &self.column_sums[layer][start..end];

                let mean = slice.iter().sum::<f32>() / self.seq_len as f32;
                let var = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>()
                    / self.seq_len as f32;
                head_variances.push(var);
            }

            // Min-max normalize across heads
            let v_min = head_variances.iter().cloned().fold(f32::INFINITY, f32::min);
            let v_max = head_variances
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let range = v_max - v_min + 1e-6;

            let normalized: Vec<f32> = head_variances.iter().map(|v| (v - v_min) / range).collect();

            // Mean across heads
            let layer_variance = normalized.iter().sum::<f32>() / self.n_kv_heads as f32;
            result.push(layer_variance);
        }

        result
    }

    /// Compute per-layer (hh_ratio, recent_ratio) budgets from variances.
    ///
    /// Official D2O: softmax(-variance) × L × ρ → clamp [0.01, 1.0]
    /// Then split into hh/recent proportionally.
    ///
    /// Returns: `Vec<(f32, f32)>` of length `n_layers`.
    pub fn compute_budgets(&self, hh_ratio: f32, recent_ratio: f32) -> Vec<(f32, f32)> {
        let variances = self.compute_variances();
        let n = variances.len();
        let rho = hh_ratio + recent_ratio;

        // softmax(-variance) with numerical stability
        let neg_vars: Vec<f32> = variances.iter().map(|v| -v).collect();
        let max_neg = neg_vars.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = neg_vars.iter().map(|v| (v - max_neg).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();

        // Per-layer dynamic ratios
        let weights: Vec<f32> = exp_vals.iter().map(|e| e / sum_exp).collect();
        let dynamic_ratios: Vec<f32> = weights
            .iter()
            .map(|w| (w * n as f32 * rho).clamp(0.01, 1.0))
            .collect();

        // Split into hh/recent proportionally
        let hh_split = if rho > 0.0 { hh_ratio / rho } else { 0.5 };
        let recent_split = if rho > 0.0 { recent_ratio / rho } else { 0.5 };
        dynamic_ratios
            .iter()
            .map(|r| (r * hh_split, r * recent_split))
            .collect()
    }
}

// L2 trait impl — L3-inference (transformer forward) 가 concrete 의존 없이
// dyn dispatch 로 collect_layer 호출 (S-C3 trait inversion).
impl crate::qcf_collector::VarianceObserver for D2OVarianceCollector {
    fn collect_layer(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        cache_seq_len: usize,
        q_stride: usize,
        k_stride: usize,
        kv_head_stride: usize,
        start_pos: usize,
    ) {
        D2OVarianceCollector::collect_layer(
            self,
            layer_idx,
            q,
            k,
            seq_len,
            cache_seq_len,
            q_stride,
            k_stride,
            kv_head_stride,
            start_pos,
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::erasing_op)]
mod tests {
    use super::*;

    /// Helper: create a collector with zero-initialized column sums.
    fn make_collector(
        n_layers: usize,
        n_kv_heads: usize,
        n_heads_q: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> D2OVarianceCollector {
        D2OVarianceCollector::new(n_layers, n_kv_heads, n_heads_q, head_dim, seq_len)
    }

    // ── Variance tests ──

    #[test]
    fn test_variance_uniform_attention() {
        // Layer 0: uniform column sums → low variance.
        // Layer 1: two heads with different spike magnitudes → non-zero variance
        //   after min-max normalization across the two heads.
        //
        // Note: if both heads have identical patterns the min-max range is 0 and
        // the normalized variance collapses to 0. We therefore use different
        // spike magnitudes for head 0 (strong) and head 1 (weak).
        let mut c = make_collector(2, 2, 2, 4, 4);

        // Layer 0: all equal values across both heads and all positions.
        for i in 0..c.column_sums[0].len() {
            c.column_sums[0][i] = 1.0;
        }

        let seq = c.seq_len;

        // Layer 1, head 0: strong spike at position 0.
        c.column_sums[1][0] = 100.0;
        for j in 1..seq {
            c.column_sums[1][j] = 0.01;
        }
        // Layer 1, head 1: moderate spike (different magnitude so head variances differ).
        c.column_sums[1][seq] = 10.0;
        for j in 1..seq {
            c.column_sums[1][seq + j] = 0.01;
        }

        let vars = c.compute_variances();
        assert_eq!(vars.len(), 2);

        // Layer 0 is uniform → variance = 0.
        assert_eq!(vars[0], 0.0, "uniform layer should have variance 0");

        // Layer 1 has two heads with different variance magnitudes; after
        // min-max normalization the mean across heads is in (0, 1].
        assert!(
            vars[0] < vars[1],
            "uniform layer ({}) should have lower variance than sparse layer ({})",
            vars[0],
            vars[1]
        );
    }

    #[test]
    fn test_variance_min_max_normalization() {
        // Single layer, 4 heads: construct head variances that differ by orders of magnitude.
        // After min-max normalization, the result must lie in [0, 1].
        let seq_len = 8;
        let mut c = make_collector(1, 4, 4, 4, seq_len);

        // head 0: all 1.0 → var ≈ 0
        for j in 0..seq_len {
            c.column_sums[0][0 * seq_len + j] = 1.0;
        }
        // head 1: slight spread
        for j in 0..seq_len {
            c.column_sums[0][seq_len + j] = j as f32;
        }
        // head 2: bigger spread
        for j in 0..seq_len {
            c.column_sums[0][2 * seq_len + j] = (j as f32) * 10.0;
        }
        // head 3: large spike
        c.column_sums[0][3 * seq_len] = 1000.0;
        for j in 1..seq_len {
            c.column_sums[0][3 * seq_len + j] = 0.0;
        }

        let vars = c.compute_variances();
        assert_eq!(vars.len(), 1);
        let v = vars[0];
        assert!(v >= 0.0, "variance should be non-negative, got {}", v);
        assert!(v <= 1.0, "normalized variance should be <= 1.0, got {}", v);
    }

    // ── Budget tests ──

    #[test]
    fn test_budget_softmax_clamp() {
        // 4 layers with known variance ordering: 0.0 < 0.33 < 0.67 < 1.0
        // Manually set column_sums to produce these approximate variances.
        // Easiest: override column_sums directly via the helper struct.
        let seq_len = 4;
        let mut c = make_collector(4, 1, 1, 4, seq_len);

        // Layer 0: uniform → variance ≈ 0
        for j in 0..seq_len {
            c.column_sums[0][j] = 1.0;
        }
        // Layer 1: mild spread
        for j in 0..seq_len {
            c.column_sums[1][j] = j as f32;
        }
        // Layer 2: moderate spike
        c.column_sums[2][0] = 5.0;
        for j in 1..seq_len {
            c.column_sums[2][j] = 0.5;
        }
        // Layer 3: strong spike
        c.column_sums[3][0] = 100.0;
        for j in 1..seq_len {
            c.column_sums[3][j] = 0.0;
        }

        let hh_ratio = 0.15_f32;
        let recent_ratio = 0.05_f32;
        let budgets = c.compute_budgets(hh_ratio, recent_ratio);

        assert_eq!(budgets.len(), 4);

        // Total budget for each layer
        let totals: Vec<f32> = budgets.iter().map(|(h, r)| h + r).collect();

        // Layer 0 (lowest variance) should get highest budget
        assert!(
            totals[0] >= totals[3],
            "layer 0 should have >= budget than layer 3: {} vs {}",
            totals[0],
            totals[3]
        );

        // All ratios must be in [0.01, 1.0]
        for (i, (hh, rec)) in budgets.iter().enumerate() {
            assert!(*hh >= 0.0 && *rec >= 0.0, "layer {} has negative budget", i);
            let total = hh + rec;
            assert!(
                (0.01..=1.0).contains(&total),
                "layer {} total budget {} out of [0.01, 1.0]",
                i,
                total
            );
        }

        // hh/recent split ratio must be maintained (hh : recent = 0.15 : 0.05 = 3:1)
        let rho = hh_ratio + recent_ratio;
        let expected_hh_frac = hh_ratio / rho;
        let expected_rec_frac = recent_ratio / rho;
        for (i, (hh, rec)) in budgets.iter().enumerate() {
            let total = hh + rec;
            if total > 1e-6 {
                let actual_hh_frac = hh / total;
                let actual_rec_frac = rec / total;
                assert!(
                    (actual_hh_frac - expected_hh_frac).abs() < 1e-5,
                    "layer {} hh fraction mismatch: {} vs {}",
                    i,
                    actual_hh_frac,
                    expected_hh_frac
                );
                assert!(
                    (actual_rec_frac - expected_rec_frac).abs() < 1e-5,
                    "layer {} recent fraction mismatch: {} vs {}",
                    i,
                    actual_rec_frac,
                    expected_rec_frac
                );
            }
        }
    }

    #[test]
    fn test_budget_single_layer() {
        // Edge case: 1 layer. softmax of a single element = 1.0.
        // dynamic_ratio = 1.0 * 1 * rho = rho, then clamp(rho, 0.01, 1.0).
        let seq_len = 2;
        let mut c = make_collector(1, 1, 1, 2, seq_len);

        // Any column sums work; variance doesn't matter when n=1.
        c.column_sums[0][0] = 0.6;
        c.column_sums[0][1] = 0.4;

        let hh_ratio = 0.15_f32;
        let recent_ratio = 0.05_f32;
        let rho = hh_ratio + recent_ratio; // 0.20

        let budgets = c.compute_budgets(hh_ratio, recent_ratio);
        assert_eq!(budgets.len(), 1);

        let (hh, rec) = budgets[0];
        let total = hh + rec;
        let expected = rho.clamp(0.01, 1.0);
        assert!(
            (total - expected).abs() < 1e-5,
            "single-layer total budget should be rho={}, got {}",
            expected,
            total
        );
    }

    // ── collect_layer tests ──

    #[test]
    fn test_collect_layer_causal_masking() {
        // 1 layer, 1 kv_head, 1 q_head, head_dim=2, seq_len=3
        // Query positions: 0, 1, 2
        // Position 0 is attended by queries 0, 1, 2 → highest column sum
        // Position 2 is attended only by query 2 → lowest column sum
        let n_layers = 1;
        let n_kv_heads = 1;
        let n_heads_q = 1;
        let head_dim = 2;
        let seq_len = 3;

        let mut c = make_collector(n_layers, n_kv_heads, n_heads_q, head_dim, seq_len);

        // Q: [seq_len, n_heads_q, head_dim] packed flat
        // All query vectors identical: [1.0, 0.0]
        // q_stride = n_heads_q * head_dim = 2
        let q_stride = n_heads_q * head_dim;
        let q: Vec<f32> = (0..seq_len).flat_map(|_| vec![1.0f32, 0.0f32]).collect();

        // K: [seq_len, n_kv_heads, head_dim] packed flat
        // All key vectors identical: [1.0, 0.0]
        // k_stride = n_kv_heads * head_dim = 2, kv_head_stride = head_dim = 2
        let k_stride = n_kv_heads * head_dim;
        let kv_head_stride = head_dim;
        let k: Vec<f32> = (0..seq_len).flat_map(|_| vec![1.0f32, 0.0f32]).collect();

        c.collect_layer(
            0,
            &q,
            &k,
            seq_len,
            seq_len, // cache_seq_len = seq_len for prefill
            q_stride,
            k_stride,
            kv_head_stride,
            0, // start_pos = 0
        );

        let col_sums = &c.column_sums[0];
        // head 0: positions 0, 1, 2
        let s0 = col_sums[0]; // position 0
        let s1 = col_sums[1]; // position 1
        let s2 = col_sums[2]; // position 2

        // Causal: pos 0 is attended by rows 0,1,2 (3 queries)
        //         pos 1 is attended by rows 1,2 (2 queries)
        //         pos 2 is attended only by row 2 (1 query)
        // With identical Q and K vectors, each query's softmax distributes
        // uniformly over valid positions. So position 0 accumulates more.
        assert!(
            s0 > s2,
            "position 0 should have higher column sum than position 2: {} vs {}",
            s0,
            s2
        );
        assert!(
            s0 > 0.0 && s1 > 0.0 && s2 > 0.0,
            "all positions should have positive column sums: {}, {}, {}",
            s0,
            s1,
            s2
        );
    }
}
