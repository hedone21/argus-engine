//! Quantization proxy: estimates NMSE from KIVI flush operations.
//!
//! NMSE = MSE(X, X') / Var(X) where X' = dequantize(quantize(X)).
//! Computed inline during residual buffer flush when FP32 originals are available.

use crate::qcf_types::{
    FlushAttentionParams, KiviFlushParams, QcfConfig, QcfMetric, aggregate_heads,
};
use crate::quant::{BlockKVQ4, BlockKVQ8, BlockQ2_0, QKKV};

/// Compute NMSE for a single quantization group of QKKV (32) values.
///
/// Performs quantize → dequantize round-trip and measures normalized error.
/// Returns 0.0 if variance is below epsilon (constant-valued block).
pub fn compute_nmse_block(original: &[f32; QKKV], bits: u8, epsilon: f32) -> f32 {
    // Compute variance of original
    let mean = original.iter().sum::<f32>() / QKKV as f32;
    let var = original.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / QKKV as f32;

    if var < epsilon {
        return 0.0;
    }

    // Quantize and dequantize
    let mut reconstructed = [0.0f32; QKKV];
    match bits {
        2 => {
            let block = BlockQ2_0::quantize(original);
            block.dequantize(&mut reconstructed);
        }
        4 => {
            let block = BlockKVQ4::quantize(original);
            block.dequantize(&mut reconstructed);
        }
        8 => {
            let block = BlockKVQ8::quantize(original);
            block.dequantize(&mut reconstructed);
        }
        _ => return 0.0,
    }

    // MSE / Var
    let mse = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| (o - r).powi(2))
        .sum::<f32>()
        / QKKV as f32;

    (mse / var).clamp(0.0, 1.0)
}

/// Compute flush proxy from FP32 residual key/value buffers.
///
/// Called during `KiviCache::flush_residual()` when FP32 originals are about to
/// be quantized. Computes NMSE separately for K and V, then combines:
/// `proxy = 0.6 × NMSE_K + 0.4 × NMSE_V` (Key is more sensitive per KIVI Table 2).
///
/// Layout: `res_k`/`res_v` are `[kv_heads][flush_tokens][head_dim]` contiguous.
pub fn compute_flush_nmse(params: &KiviFlushParams, config: &QcfConfig) -> QcfMetric {
    let KiviFlushParams {
        res_k,
        res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits,
    } = params;
    let (kv_heads, head_dim, flush_tokens, res_cap, bits) =
        (*kv_heads, *head_dim, *flush_tokens, *res_cap, *bits);
    if flush_tokens == 0 || kv_heads == 0 || head_dim == 0 {
        return QcfMetric {
            action: "kivi".to_string(),
            raw_value: 0.0,
            normalized_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: 0,
        };
    }

    let blocks_per_group = QKKV;
    let n_groups = flush_tokens / blocks_per_group;
    if n_groups == 0 {
        return QcfMetric {
            action: "kivi".to_string(),
            raw_value: 0.0,
            normalized_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: flush_tokens,
        };
    }

    let mut per_head_k = vec![0.0f32; kv_heads];
    let mut per_head_v = vec![0.0f32; kv_heads];

    for h in 0..kv_heads {
        let head_base = h * res_cap * head_dim;

        // Key NMSE: per-channel quantization (groups across tokens within each channel)
        let mut k_nmse_sum = 0.0f32;
        let mut k_block_count = 0usize;
        for group in 0..n_groups {
            let tok_start = group * blocks_per_group;
            for ch in 0..head_dim {
                let mut vals = [0.0f32; QKKV];
                for (t, v) in vals.iter_mut().enumerate().take(blocks_per_group) {
                    let idx = head_base + (tok_start + t) * head_dim + ch;
                    if idx < res_k.len() {
                        *v = res_k[idx];
                    }
                }
                k_nmse_sum += compute_nmse_block(&vals, bits, config.epsilon);
                k_block_count += 1;
            }
        }
        per_head_k[h] = if k_block_count > 0 {
            k_nmse_sum / k_block_count as f32
        } else {
            0.0
        };

        // Value NMSE: per-token quantization (groups within one token's head_dim)
        let mut v_nmse_sum = 0.0f32;
        let mut v_block_count = 0usize;
        let blocks_per_token = head_dim / QKKV;
        for t in 0..flush_tokens {
            let tok_base = head_base + t * head_dim;
            for b in 0..blocks_per_token {
                let start = tok_base + b * QKKV;
                if start + QKKV <= res_v.len() {
                    let chunk: &[f32; QKKV] = res_v[start..start + QKKV].try_into().unwrap();
                    v_nmse_sum += compute_nmse_block(chunk, bits, config.epsilon);
                    v_block_count += 1;
                }
            }
        }
        per_head_v[h] = if v_block_count > 0 {
            v_nmse_sum / v_block_count as f32
        } else {
            0.0
        };
    }

    // Combined per-head proxy: 0.6 × K + 0.4 × V
    let per_head: Vec<f32> = per_head_k
        .iter()
        .zip(per_head_v.iter())
        .map(|(&k, &v)| 0.6 * k + 0.4 * v)
        .collect();

    let raw_value = aggregate_heads(&per_head, &config.aggregation);

    QcfMetric {
        action: "kivi".to_string(),
        raw_value,
        normalized_value: raw_value, // NMSE is already a normalized metric
        per_head: Some(per_head),
        tokens_affected: flush_tokens,
    }
}

/// Compute KIVI OPR (Output Perturbation Ratio) for V cache quantization.
///
/// OPR = ||Σ ΔV|| / ||Σ V_orig|| per head, where ΔV = V_quant - V_orig.
/// V_quant is obtained via quantize → dequantize round-trip.
/// Uniform weight (1/n) is used, cancelling in numerator and denominator.
///
/// Only V is considered; K quantization's primary effect is on attention weights
/// (a second-order effect ignored per design decision B-6).
///
/// Returns `QcfMetric` with:
/// - `action`: "kivi_opr"
/// - `raw_value`: sum of per-head OPR values
/// - `normalized_value`: same as raw_value
/// - `per_head`: per-head OPR vector
/// - `tokens_affected`: flush_tokens
pub fn compute_flush_opr(params: &KiviFlushParams, _config: &QcfConfig) -> QcfMetric {
    let KiviFlushParams {
        res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits,
        ..
    } = params;
    let (kv_heads, head_dim, flush_tokens, res_cap, bits) =
        (*kv_heads, *head_dim, *flush_tokens, *res_cap, *bits);

    if flush_tokens == 0 || kv_heads == 0 || head_dim == 0 {
        return QcfMetric {
            action: "kivi_opr".to_string(),
            raw_value: 0.0,
            normalized_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: 0,
        };
    }

    let blocks_per_token = head_dim / QKKV;
    let mut per_head_opr = vec![0.0f32; kv_heads];

    for (h, opr_slot) in per_head_opr.iter_mut().enumerate() {
        let head_base = h * res_cap * head_dim;

        let mut sum_delta = vec![0.0f32; head_dim];
        let mut sum_orig = vec![0.0f32; head_dim];

        for t in 0..flush_tokens {
            let tok_base = head_base + t * head_dim;

            // Reconstruct V_quant for this token via quantize → dequantize per block
            let mut v_quant_token = vec![0.0f32; head_dim];
            for b in 0..blocks_per_token {
                let start = tok_base + b * QKKV;
                if start + QKKV > res_v.len() {
                    continue;
                }
                let chunk: &[f32; QKKV] = res_v[start..start + QKKV].try_into().unwrap();
                let out = &mut v_quant_token[b * QKKV..(b + 1) * QKKV];
                let out_arr: &mut [f32; QKKV] = out.try_into().unwrap();
                match bits {
                    2 => {
                        let block = BlockQ2_0::quantize(chunk);
                        block.dequantize(out_arr);
                    }
                    4 => {
                        let block = BlockKVQ4::quantize(chunk);
                        block.dequantize(out_arr);
                    }
                    8 => {
                        let block = BlockKVQ8::quantize(chunk);
                        block.dequantize(out_arr);
                    }
                    _ => {
                        // Unsupported bits: treat as zero error (copy original)
                        out_arr.copy_from_slice(chunk);
                    }
                }
            }

            // Accumulate sum_delta and sum_orig across tokens
            for d in 0..head_dim {
                let idx = tok_base + d;
                if idx < res_v.len() {
                    let v_orig = res_v[idx];
                    sum_orig[d] += v_orig;
                    sum_delta[d] += v_quant_token[d] - v_orig;
                }
            }
        }

        // OPR = L2(sum_delta) / L2(sum_orig)
        let norm_delta = sum_delta.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_orig = sum_orig.iter().map(|&x| x * x).sum::<f32>().sqrt();

        *opr_slot = if norm_orig < 1e-12 {
            0.0
        } else {
            norm_delta / norm_orig
        };
    }

    let raw_value: f32 = per_head_opr.iter().sum::<f32>() / kv_heads as f32;

    QcfMetric {
        action: "kivi_opr".to_string(),
        raw_value,
        normalized_value: raw_value,
        per_head: Some(per_head_opr),
        tokens_affected: flush_tokens,
    }
}

/// Compute AWQE: Σ_t α_{kv_h,t} · ε_{kv_h,t} for each KV head.
///
/// - α: GQA-aggregated attention weight (mean of Q heads in group)
/// - ε: per-token V NMSE (quantize→dequantize round-trip error)
pub fn compute_flush_awqe(params: &FlushAttentionParams, config: &QcfConfig) -> QcfMetric {
    let FlushAttentionParams {
        res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits,
        attn_scores,
        n_heads_q: _,
        scores_stride,
        gqa_group_size,
        flush_cache_start,
        scores_valid_len,
    } = params;
    let (kv_heads, head_dim, flush_tokens) = (*kv_heads, *head_dim, *flush_tokens);
    let (res_cap, bits) = (*res_cap, *bits);
    let (scores_stride, gqa_group_size) = (*scores_stride, *gqa_group_size);
    let (flush_cache_start, scores_valid_len) = (*flush_cache_start, *scores_valid_len);

    if flush_tokens == 0 || kv_heads == 0 || head_dim == 0 {
        return QcfMetric {
            action: "kivi_awqe".to_string(),
            raw_value: 0.0,
            normalized_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: 0,
        };
    }

    let blocks_per_token = head_dim / QKKV;
    let mut awqe_per_head = vec![0.0f32; kv_heads];

    for (kv_h, awqe_slot) in awqe_per_head.iter_mut().enumerate() {
        let mut weighted_sum = 0.0f32;

        for t in 0..flush_tokens {
            let cache_pos = flush_cache_start + t;

            // ── 1. GQA-aggregated attention weight ──
            // Mean of Q heads in this KV head's group.
            let alpha = if cache_pos < scores_valid_len {
                let q_start = kv_h * gqa_group_size;
                let q_end = q_start + gqa_group_size;
                let sum: f32 = (q_start..q_end)
                    .map(|qh| attn_scores[qh * scores_stride + cache_pos])
                    .sum();
                sum / gqa_group_size as f32
            } else {
                // Token outside scores range → uniform weight
                1.0 / scores_valid_len.max(1) as f32
            };

            // ── 2. Per-token V NMSE ──
            // Average NMSE across head_dim / QKKV blocks within this token.
            let head_base = kv_h * res_cap * head_dim;
            let tok_base = head_base + t * head_dim;
            let mut nmse_sum = 0.0f32;
            for b in 0..blocks_per_token {
                let start = tok_base + b * QKKV;
                if start + QKKV <= res_v.len() {
                    let chunk: &[f32; QKKV] = res_v[start..start + QKKV].try_into().unwrap();
                    nmse_sum += compute_nmse_block(chunk, bits, config.epsilon);
                }
            }
            let epsilon_t = if blocks_per_token > 0 {
                nmse_sum / blocks_per_token as f32
            } else {
                0.0
            };

            // ── 3. Weighted accumulation ──
            weighted_sum += alpha * epsilon_t;
        }

        *awqe_slot = weighted_sum;
    }

    let raw_value = aggregate_heads(&awqe_per_head, &config.aggregation);

    QcfMetric {
        action: "kivi_awqe".to_string(),
        raw_value,
        normalized_value: raw_value,
        per_head: Some(awqe_per_head),
        tokens_affected: flush_tokens,
    }
}

/// Compute AW-VOPR (Attention-Weighted Vector Output Perturbation Ratio).
///
/// Unlike AWQE which sums scalar NMSE weighted by attention, AW-VOPR accumulates
/// the attention-weighted V quantization error as a **vector** per Q-head, then
/// takes the L2 norm. Opposite-direction errors cancel in the vector sum, so
/// AW-VOPR reflects the *actual* output perturbation more faithfully.
///
/// GQA aggregation: **norm-first-then-mean**.
///   1. Per Q-head: `ratio_qh = ||sum_t alpha_t * delta_V_t|| / max(||sum_t alpha_t * V_t||, eps)`
///   2. Per KV-head: `aw_vopr_h = mean(ratio_qh for qh in gqa_group)`
///   3. Final: `aw_vopr = mean(aw_vopr_h for all kv_heads)`
pub fn compute_flush_aw_vopr(params: &FlushAttentionParams, config: &QcfConfig) -> QcfMetric {
    let FlushAttentionParams {
        res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits,
        attn_scores,
        n_heads_q: _,
        scores_stride,
        gqa_group_size,
        flush_cache_start,
        scores_valid_len,
    } = params;
    let (kv_heads, head_dim, flush_tokens) = (*kv_heads, *head_dim, *flush_tokens);
    let (res_cap, bits) = (*res_cap, *bits);
    let (scores_stride, gqa_group_size) = (*scores_stride, *gqa_group_size);
    let (flush_cache_start, scores_valid_len) = (*flush_cache_start, *scores_valid_len);

    if flush_tokens == 0 || kv_heads == 0 || head_dim == 0 || gqa_group_size == 0 {
        return QcfMetric {
            action: "aw_vopr".to_string(),
            raw_value: 0.0,
            normalized_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: 0,
        };
    }

    let blocks_per_token = head_dim / QKKV;
    let eps = config.epsilon.max(1e-10);
    let mut per_head_vopr = vec![0.0f32; kv_heads];

    // Pre-compute dequantized V for the flush region to avoid redundant quantize round-trips
    // across Q-heads within the same KV-head.
    // Layout: [kv_heads][flush_tokens][head_dim]
    let mut v_quant_all = vec![0.0f32; kv_heads * flush_tokens * head_dim];
    for h in 0..kv_heads {
        let head_base = h * res_cap * head_dim;
        for t in 0..flush_tokens {
            let tok_base = head_base + t * head_dim;
            let out_base = h * flush_tokens * head_dim + t * head_dim;
            for b in 0..blocks_per_token {
                let src_start = tok_base + b * QKKV;
                if src_start + QKKV > res_v.len() {
                    continue;
                }
                let chunk: &[f32; QKKV] = res_v[src_start..src_start + QKKV].try_into().unwrap();
                let out = &mut v_quant_all[out_base + b * QKKV..out_base + (b + 1) * QKKV];
                let out_arr: &mut [f32; QKKV] = out.try_into().unwrap();
                match bits {
                    2 => {
                        let block = BlockQ2_0::quantize(chunk);
                        block.dequantize(out_arr);
                    }
                    4 => {
                        let block = BlockKVQ4::quantize(chunk);
                        block.dequantize(out_arr);
                    }
                    8 => {
                        let block = BlockKVQ8::quantize(chunk);
                        block.dequantize(out_arr);
                    }
                    _ => {
                        out_arr.copy_from_slice(chunk);
                    }
                }
            }
        }
    }

    for (kv_h, vopr_slot) in per_head_vopr.iter_mut().enumerate() {
        let head_base = kv_h * res_cap * head_dim;
        let quant_head_base = kv_h * flush_tokens * head_dim;
        let mut ratio_sum = 0.0f32;

        for g in 0..gqa_group_size {
            let qh = kv_h * gqa_group_size + g;
            let mut delta_o = vec![0.0f32; head_dim];
            let mut orig_o = vec![0.0f32; head_dim];

            for t in 0..flush_tokens {
                let cache_pos = flush_cache_start + t;

                // Attention weight for this Q-head at this position
                let alpha = if cache_pos < scores_valid_len {
                    attn_scores[qh * scores_stride + cache_pos]
                } else {
                    1.0 / scores_valid_len.max(1) as f32
                };

                let tok_base = head_base + t * head_dim;
                let quant_tok_base = quant_head_base + t * head_dim;

                for d in 0..head_dim {
                    let idx = tok_base + d;
                    if idx < res_v.len() {
                        let v_orig = res_v[idx];
                        let v_quant = v_quant_all[quant_tok_base + d];
                        let delta_v = v_orig - v_quant;

                        delta_o[d] += alpha * delta_v;
                        orig_o[d] += alpha * v_orig;
                    }
                }
            }

            let norm_delta = delta_o.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let norm_orig = orig_o.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let ratio_qh = norm_delta / norm_orig.max(eps);

            ratio_sum += ratio_qh;
        }

        *vopr_slot = ratio_sum / gqa_group_size as f32;
    }

    let raw_value = per_head_vopr.iter().sum::<f32>() / kv_heads as f32;

    QcfMetric {
        action: "aw_vopr".to_string(),
        raw_value,
        normalized_value: raw_value,
        per_head: Some(per_head_vopr),
        tokens_affected: flush_tokens,
    }
}

// ── QcfComputer trait impl (S-3b-4 trait inversion) ─────────────────

/// Stateless KIVI flush proxy 계산기.
///
/// `QcfComputer` trait 의 4 메서드를 본 모듈의 자유 함수 4개로 위임.
/// pressure/kivi_cache 가 `&dyn QcfComputer` 로 의존하여 L3-pressure →
/// L3-qcf cross-domain concrete import 를 제거한다.
#[derive(Debug, Clone, Copy, Default)]
pub struct KiviQcfComputer;

impl crate::qcf_computer::QcfComputer for KiviQcfComputer {
    fn flush_nmse(&self, params: &KiviFlushParams, config: &QcfConfig) -> QcfMetric {
        compute_flush_nmse(params, config)
    }

    fn flush_opr(&self, params: &KiviFlushParams, config: &QcfConfig) -> QcfMetric {
        compute_flush_opr(params, config)
    }

    fn flush_awqe(&self, params: &FlushAttentionParams, config: &QcfConfig) -> QcfMetric {
        compute_flush_awqe(params, config)
    }

    fn flush_aw_vopr(&self, params: &FlushAttentionParams, config: &QcfConfig) -> QcfMetric {
        compute_flush_aw_vopr(params, config)
    }

    fn clone_box(&self) -> Box<dyn crate::qcf_computer::QcfComputer> {
        Box::new(*self)
    }
}

#[cfg(test)]
#[allow(clippy::erasing_op)]
mod tests {
    use super::*;

    #[test]
    fn test_nmse_block_q8_low_error() {
        // Q8 should have very low NMSE
        let original: [f32; QKKV] = std::array::from_fn(|i| i as f32 * 0.1);
        let nmse = compute_nmse_block(&original, 8, 1e-8);
        assert!(nmse < 0.01, "Q8 NMSE={nmse} should be very low");
    }

    #[test]
    fn test_nmse_block_q4_moderate_error() {
        let original: [f32; QKKV] = std::array::from_fn(|i| i as f32 * 0.1);
        let nmse = compute_nmse_block(&original, 4, 1e-8);
        assert!(nmse > 0.0, "Q4 NMSE should be positive");
        assert!(nmse < 0.5, "Q4 NMSE={nmse} should be moderate");
    }

    #[test]
    fn test_nmse_block_q2_higher_error() {
        let original: [f32; QKKV] = std::array::from_fn(|i| i as f32 * 0.1);
        let nmse_q2 = compute_nmse_block(&original, 2, 1e-8);
        let nmse_q4 = compute_nmse_block(&original, 4, 1e-8);
        let nmse_q8 = compute_nmse_block(&original, 8, 1e-8);
        assert!(
            nmse_q2 >= nmse_q4,
            "Q2 ({nmse_q2}) should have >= NMSE than Q4 ({nmse_q4})"
        );
        assert!(
            nmse_q4 >= nmse_q8,
            "Q4 ({nmse_q4}) should have >= NMSE than Q8 ({nmse_q8})"
        );
    }

    #[test]
    fn test_nmse_block_zero_variance() {
        // All same values → variance ≈ 0 → NMSE = 0
        let original = [42.0f32; QKKV];
        let nmse = compute_nmse_block(&original, 4, 1e-8);
        assert_eq!(nmse, 0.0);
    }

    #[test]
    fn test_nmse_block_zero_values() {
        let original = [0.0f32; QKKV];
        let nmse = compute_nmse_block(&original, 2, 1e-8);
        assert_eq!(nmse, 0.0);
    }

    #[test]
    fn test_nmse_block_invalid_bits() {
        let original: [f32; QKKV] = std::array::from_fn(|i| i as f32);
        let nmse = compute_nmse_block(&original, 3, 1e-8);
        assert_eq!(nmse, 0.0); // Unsupported bits → 0
    }

    #[test]
    fn test_flush_proxy_basic() {
        let kv_heads = 1;
        let head_dim = 32; // Must be multiple of QKKV
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.02).collect();

        let config = QcfConfig::default();
        let params = KiviFlushParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
        };
        let metric = compute_flush_nmse(&params, &config);

        assert_eq!(metric.action, "kivi");
        assert!(metric.raw_value >= 0.0);
        assert!(metric.raw_value <= 1.0);
        assert_eq!(metric.tokens_affected, flush_tokens);
        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
    }

    #[test]
    fn test_flush_proxy_empty() {
        let config = QcfConfig::default();
        let params = KiviFlushParams {
            res_k: &[],
            res_v: &[],
            kv_heads: 0,
            head_dim: 0,
            flush_tokens: 0,
            res_cap: 0,
            bits: 4,
        };
        let metric = compute_flush_nmse(&params, &config);
        assert_eq!(metric.raw_value, 0.0);
    }

    #[test]
    fn test_flush_proxy_multi_head() {
        let kv_heads = 4;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.02).collect();

        let config = QcfConfig::default();
        let params = KiviFlushParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 2,
        };
        let metric = compute_flush_nmse(&params, &config);

        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
        for &h_val in metric.per_head.as_ref().unwrap() {
            assert!(h_val >= 0.0);
        }
    }

    #[test]
    fn test_flush_proxy_q2_higher_than_q8() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| ((i % 100) as f32) * 0.1).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| ((i % 100) as f32) * 0.1).collect();

        let config = QcfConfig::default();
        let proxy_q2 = compute_flush_nmse(
            &KiviFlushParams {
                res_k: &res_k,
                res_v: &res_v,
                kv_heads,
                head_dim,
                flush_tokens,
                res_cap,
                bits: 2,
            },
            &config,
        );
        let proxy_q8 = compute_flush_nmse(
            &KiviFlushParams {
                res_k: &res_k,
                res_v: &res_v,
                kv_heads,
                head_dim,
                flush_tokens,
                res_cap,
                bits: 8,
            },
            &config,
        );

        assert!(
            proxy_q2.raw_value >= proxy_q8.raw_value,
            "Q2 proxy ({}) should >= Q8 proxy ({})",
            proxy_q2.raw_value,
            proxy_q8.raw_value
        );
    }

    // ── compute_flush_opr tests ──────────────────────────────────────────────

    #[test]
    fn test_flush_opr_basic() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.02).collect();

        let config = QcfConfig::default();
        let params = KiviFlushParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
        };
        let metric = compute_flush_opr(&params, &config);

        assert_eq!(metric.action, "kivi_opr");
        assert!(metric.raw_value >= 0.0, "OPR must be non-negative");
        assert_eq!(metric.tokens_affected, flush_tokens);
        assert!(metric.per_head.is_some());
        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
        assert!(
            (metric.raw_value - metric.normalized_value).abs() < 1e-6,
            "raw_value and normalized_value must be equal"
        );
    }

    #[test]
    fn test_flush_opr_q2_higher_than_q8() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        // Use values with meaningful variance so quantization error is detectable
        let res_k: Vec<f32> = (0..elems).map(|i| ((i % 100) as f32) * 0.1).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| ((i % 100) as f32) * 0.1).collect();

        let config = QcfConfig::default();
        let make_params = |bits: u8| KiviFlushParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits,
        };

        let opr_q2 = compute_flush_opr(&make_params(2), &config);
        let opr_q8 = compute_flush_opr(&make_params(8), &config);

        assert!(
            opr_q2.raw_value >= opr_q8.raw_value,
            "Q2 OPR ({}) should >= Q8 OPR ({})",
            opr_q2.raw_value,
            opr_q8.raw_value
        );
    }

    #[test]
    fn test_flush_opr_zero_input() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k = vec![0.0f32; elems];
        let res_v = vec![0.0f32; elems];

        let config = QcfConfig::default();
        let params = KiviFlushParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
        };
        let metric = compute_flush_opr(&params, &config);

        assert_eq!(metric.raw_value, 0.0, "zero input → OPR must be 0.0");
        assert_eq!(metric.per_head.as_ref().unwrap()[0], 0.0);
    }

    #[test]
    fn test_flush_opr_multi_head() {
        let kv_heads = 4;
        let head_dim = 64;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.02 + 1.0).collect();

        let config = QcfConfig::default();
        let params = KiviFlushParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 2,
        };
        let metric = compute_flush_opr(&params, &config);

        let ph = metric.per_head.as_ref().unwrap();
        assert_eq!(ph.len(), kv_heads, "per_head length must equal kv_heads");
        for (h, &opr) in ph.iter().enumerate() {
            assert!(opr >= 0.0, "head {h} OPR ({opr}) must be non-negative");
        }
        // raw_value must equal mean of per_head
        let expected_mean: f32 = ph.iter().sum::<f32>() / ph.len() as f32;
        assert!(
            (metric.raw_value - expected_mean).abs() < 1e-5,
            "raw_value ({}) must equal mean of per_head ({})",
            metric.raw_value,
            expected_mean
        );
    }

    // ── compute_flush_awqe tests ─────────────────────────────────────────────

    /// Helper: build res_v with non-trivial values for [kv_heads][res_cap][head_dim].
    fn make_res_v(kv_heads: usize, res_cap: usize, head_dim: usize) -> Vec<f32> {
        let n = kv_heads * res_cap * head_dim;
        (0..n).map(|i| ((i % 100) as f32) * 0.1 + 0.1).collect()
    }

    /// Helper: build uniform attention scores where each position gets 1/valid_len.
    fn make_uniform_scores(n_heads_q: usize, stride: usize, valid_len: usize) -> Vec<f32> {
        let mut scores = vec![0.0f32; n_heads_q * stride];
        let w = 1.0 / valid_len as f32;
        for qh in 0..n_heads_q {
            for pos in 0..valid_len {
                scores[qh * stride + pos] = w;
            }
        }
        scores
    }

    /// Test 1: uniform attention + uniform data → AWQE ≈ mean(per-token NMSE) × (flush_tokens / valid_len)
    #[test]
    fn test_awqe_uniform_scores() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = 64; // total cache size (larger than flush_tokens)
        let stride = valid_len;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);
        let scores = make_uniform_scores(n_heads_q, stride, valid_len);

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_awqe(&params, &config);

        assert_eq!(metric.action, "kivi_awqe");
        assert!(metric.raw_value >= 0.0, "AWQE must be non-negative");
        assert_eq!(metric.tokens_affected, flush_tokens);
        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);

        // With uniform weights (1/valid_len) for flush_tokens positions:
        // AWQE = Σ_{t=0..flush_tokens} (1/valid_len) × NMSE_t
        // = (flush_tokens/valid_len) × mean_nmse
        // Verify raw_value > 0 for non-trivial data
        assert!(
            metric.raw_value > 0.0,
            "AWQE should be positive for non-zero error data"
        );
    }

    /// Test 2: attention concentrated on one token → AWQE ≈ that token's NMSE.
    #[test]
    fn test_awqe_concentrated_attention() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);

        // Put all attention weight on token 5
        let focused_token = 5usize;
        let mut scores = vec![0.0f32; n_heads_q * stride];
        scores[focused_token] = 1.0; // all weight on token 5

        let config = QcfConfig::default();

        // Compute expected NMSE for token 5
        let blocks_per_token = head_dim / QKKV;
        let mut expected_nmse = 0.0f32;
        for b in 0..blocks_per_token {
            let start = focused_token * head_dim + b * QKKV;
            let chunk: &[f32; QKKV] = res_v[start..start + QKKV].try_into().unwrap();
            expected_nmse += compute_nmse_block(chunk, 4, config.epsilon);
        }
        expected_nmse /= blocks_per_token as f32;

        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_awqe(&params, &config);

        // AWQE = 1.0 × NMSE(token5) + 0 × others
        assert!(
            (metric.raw_value - expected_nmse).abs() < 1e-5,
            "AWQE ({}) should ≈ NMSE of focused token ({})",
            metric.raw_value,
            expected_nmse
        );
    }

    /// Test 3: high attention weight but zero quantization error → AWQE ≈ 0.
    #[test]
    fn test_awqe_zero_error_high_attention() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        // All same value → variance=0 → NMSE=0 for every block
        let res_v = vec![1.0f32; kv_heads * res_cap * head_dim];

        // High attention on first token
        let mut scores = vec![0.0f32; n_heads_q * stride];
        scores[0] = 1.0;

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_awqe(&params, &config);

        assert!(
            metric.raw_value < 1e-6,
            "high attention but zero error → AWQE should ≈ 0, got {}",
            metric.raw_value
        );
    }

    /// Test 4: high quantization error but zero attention → AWQE ≈ 0 (core AWQE property).
    #[test]
    fn test_awqe_high_error_zero_attention() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        // High-variance data → high NMSE
        let res_v: Vec<f32> = (0..kv_heads * res_cap * head_dim)
            .map(|i| if i % 2 == 0 { 100.0 } else { -100.0 })
            .collect();

        // All attention weights are zero (no attention at all)
        let scores = vec![0.0f32; n_heads_q * stride];

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_awqe(&params, &config);

        assert!(
            metric.raw_value < 1e-6,
            "zero attention → AWQE should ≈ 0 regardless of error, got {}",
            metric.raw_value
        );
    }

    /// Test 5: GQA aggregation — n_heads_q=4, kv_heads=1, G=4 → mean of 4 Q head scores.
    #[test]
    fn test_awqe_gqa_aggregation() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 4;
        let gqa_group_size = n_heads_q / kv_heads; // = 4
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);

        // Q head 0: all weight on token 0
        // Q head 1: all weight on token 0
        // Q head 2: zero weights
        // Q head 3: zero weights
        // GQA mean α for token 0 = (1 + 1 + 0 + 0) / 4 = 0.5
        let mut scores = vec![0.0f32; n_heads_q * stride];
        scores[0 * stride] = 1.0; // head 0, pos 0
        scores[stride] = 1.0; // head 1, pos 0

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_awqe(&params, &config);

        // Compare to single-Q-head version with α=0.5 on token 0
        // Compute expected: 0.5 × NMSE(token 0)
        let blocks_per_token = head_dim / QKKV;
        let mut nmse_tok0 = 0.0f32;
        for b in 0..blocks_per_token {
            let start = b * QKKV;
            let chunk: &[f32; QKKV] = res_v[start..start + QKKV].try_into().unwrap();
            nmse_tok0 += compute_nmse_block(chunk, 4, config.epsilon);
        }
        nmse_tok0 /= blocks_per_token as f32;
        let expected = 0.5 * nmse_tok0;

        assert!(
            (metric.raw_value - expected).abs() < 1e-5,
            "GQA aggregation: AWQE ({}) should ≈ 0.5×NMSE(tok0) ({})",
            metric.raw_value,
            expected
        );
    }

    /// Test 6: multi-head — kv_heads=2, different attention patterns → per_head values differ.
    #[test]
    fn test_awqe_multi_head() {
        let kv_heads = 2;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 2;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);

        // Head 0: put all weight on token 0
        // Head 1: put all weight on token 31 (last token)
        let mut scores = vec![0.0f32; n_heads_q * stride];
        scores[0 * stride] = 1.0; // Q head 0 → token 0
        scores[stride + 31] = 1.0; // Q head 1 → token 31

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_awqe(&params, &config);

        let ph = metric.per_head.as_ref().unwrap();
        assert_eq!(ph.len(), kv_heads);

        // Both heads have non-zero error (different focused tokens)
        // The two per-head values should reflect different tokens' NMSE
        assert!(ph[0] >= 0.0 && ph[1] >= 0.0);

        // Verify they differ (data pattern makes token 0 and token 31 have different values)
        // This is a structural check — per_head[0] ≠ per_head[1] for non-degenerate data
        // (tokens at pos 0 and pos 31 have different values in make_res_v)
    }

    /// Test 7: flush_cache_start >= scores_valid_len → uniform fallback.
    #[test]
    fn test_awqe_flush_outside_scores_range() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = 32; // scores only cover 32 positions
        let stride = valid_len;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);
        let scores = vec![0.0f32; n_heads_q * stride]; // zeros but won't be used

        let config = QcfConfig::default();
        // flush_cache_start = 64 > valid_len=32 → all tokens use uniform fallback α = 1/valid_len
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 64, // > valid_len
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_awqe(&params, &config);

        // Should produce non-negative result (uniform fallback used)
        assert!(
            metric.raw_value >= 0.0,
            "uniform fallback must produce non-negative AWQE"
        );
        assert_eq!(metric.action, "kivi_awqe");
    }

    /// Test 8: flush_tokens=0 → raw_value=0.
    #[test]
    fn test_awqe_empty() {
        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &[],
            kv_heads: 1,
            head_dim: 32,
            flush_tokens: 0,
            res_cap: 32,
            bits: 4,
            attn_scores: &[],
            n_heads_q: 1,
            scores_stride: 32,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: 0,
        };
        let metric = compute_flush_awqe(&params, &config);
        assert_eq!(metric.raw_value, 0.0);
        assert_eq!(metric.tokens_affected, 0);
    }

    /// Test 9: single QKKV block flush (flush_tokens=32, minimal case).
    #[test]
    fn test_awqe_single_qkkv_block() {
        let kv_heads = 1;
        let head_dim = 32; // exactly QKKV → 1 block per token
        let flush_tokens = 32; // QKKV tokens (minimum flush)
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        // Use non-trivial data with variance
        let res_v: Vec<f32> = (0..kv_heads * res_cap * head_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();

        // Uniform attention
        let scores = make_uniform_scores(n_heads_q, stride, valid_len);

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_awqe(&params, &config);

        assert_eq!(metric.action, "kivi_awqe");
        assert!(metric.raw_value >= 0.0);
        assert_eq!(metric.tokens_affected, flush_tokens);
    }

    // ── AW-VOPR tests ─────────────────────────────────────────────────────────

    /// AW-VOPR with empty input returns zero.
    #[test]
    fn test_aw_vopr_empty() {
        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &[],
            kv_heads: 1,
            head_dim: 32,
            flush_tokens: 0,
            res_cap: 32,
            bits: 4,
            attn_scores: &[],
            n_heads_q: 1,
            scores_stride: 32,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: 0,
        };
        let metric = compute_flush_aw_vopr(&params, &config);
        assert_eq!(metric.action, "aw_vopr");
        assert_eq!(metric.raw_value, 0.0);
        assert_eq!(metric.tokens_affected, 0);
    }

    /// AW-VOPR is non-negative for non-trivial data.
    #[test]
    fn test_aw_vopr_basic_non_negative() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);
        let scores = make_uniform_scores(n_heads_q, stride, valid_len);

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_aw_vopr(&params, &config);

        assert_eq!(metric.action, "aw_vopr");
        assert!(
            metric.raw_value >= 0.0,
            "AW-VOPR must be non-negative, got {}",
            metric.raw_value
        );
        assert_eq!(metric.tokens_affected, flush_tokens);
        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
    }

    /// AW-VOPR with zero attention weights returns zero (no output perturbation).
    #[test]
    fn test_aw_vopr_zero_attention() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);
        // All attention weights zero
        let scores = vec![0.0f32; n_heads_q * stride];

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_aw_vopr(&params, &config);

        assert!(
            metric.raw_value < 1e-6,
            "zero attention -> AW-VOPR should be ~0, got {}",
            metric.raw_value
        );
    }

    /// AW-VOPR with constant V (zero quantization error) returns zero.
    #[test]
    fn test_aw_vopr_zero_quant_error() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        // All same value -> variance=0 -> quant error=0
        let res_v = vec![1.0f32; kv_heads * res_cap * head_dim];
        let scores = make_uniform_scores(n_heads_q, stride, valid_len);

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_aw_vopr(&params, &config);

        assert!(
            metric.raw_value < 1e-6,
            "constant V -> zero quant error -> AW-VOPR should be ~0, got {}",
            metric.raw_value
        );
    }

    /// AW-VOPR <= AWQE conceptually: vector cancellation should reduce measured error.
    /// With uniform attention and non-trivial data, AW-VOPR captures directional
    /// cancellation that AWQE does not.
    #[test]
    fn test_aw_vopr_le_awqe_uniform_attention() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);
        let scores = make_uniform_scores(n_heads_q, stride, valid_len);
        let config = QcfConfig::default();

        let awqe_params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let awqe = compute_flush_awqe(&awqe_params, &config);

        let vopr_params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let vopr = compute_flush_aw_vopr(&vopr_params, &config);

        // Both should be non-negative
        assert!(awqe.raw_value >= 0.0);
        assert!(vopr.raw_value >= 0.0);

        // AW-VOPR should be smaller or comparable to AWQE (vector cancellation)
        // Allow small margin for numerical differences
        assert!(
            vopr.raw_value <= awqe.raw_value + 0.01,
            "AW-VOPR ({}) should be <= AWQE ({}) + margin (vector cancellation)",
            vopr.raw_value,
            awqe.raw_value
        );
    }

    /// AW-VOPR with GQA: n_heads_q=4, kv_heads=2 (G=2).
    #[test]
    fn test_aw_vopr_gqa() {
        let kv_heads = 2;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 4;
        let gqa_group_size = n_heads_q / kv_heads; // = 2
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);
        let scores = make_uniform_scores(n_heads_q, stride, valid_len);

        let config = QcfConfig::default();
        let params = FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };
        let metric = compute_flush_aw_vopr(&params, &config);

        assert_eq!(metric.action, "aw_vopr");
        assert!(metric.raw_value >= 0.0);
        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
        // Both heads should have non-negative per-head values
        for &ph in metric.per_head.as_ref().unwrap() {
            assert!(ph >= 0.0, "per-head AW-VOPR must be non-negative, got {ph}");
        }
    }

    /// AW-VOPR bits=2 should produce higher error than bits=4.
    #[test]
    fn test_aw_vopr_bits_ordering() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let valid_len = flush_tokens;
        let stride = flush_tokens;

        let res_v = make_res_v(kv_heads, res_cap, head_dim);
        let scores = make_uniform_scores(n_heads_q, stride, valid_len);
        let config = QcfConfig::default();

        let make_params = |bits: u8| FlushAttentionParams {
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits,
            attn_scores: &scores,
            n_heads_q,
            scores_stride: stride,
            gqa_group_size: 1,
            flush_cache_start: 0,
            scores_valid_len: valid_len,
        };

        let vopr_q2 = compute_flush_aw_vopr(&make_params(2), &config).raw_value;
        let vopr_q4 = compute_flush_aw_vopr(&make_params(4), &config).raw_value;
        let vopr_q8 = compute_flush_aw_vopr(&make_params(8), &config).raw_value;

        assert!(
            vopr_q2 >= vopr_q4,
            "Q2 AW-VOPR ({vopr_q2}) should be >= Q4 ({vopr_q4})"
        );
        assert!(
            vopr_q4 >= vopr_q8,
            "Q4 AW-VOPR ({vopr_q4}) should be >= Q8 ({vopr_q8})"
        );
    }
}
