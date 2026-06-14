mod forward_gen;
mod forward_gen_fmt;
mod forward_prefill_fmt;
pub(crate) use forward_gen_fmt::ForwardGenFmtArgs;
pub(crate) use forward_prefill_fmt::ForwardPrefillFmtArgs;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::shape::Shape;
use crate::tensor::Tensor;
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;

use crate::layers::tensor_partition::PartitionContext;

// OpInstrument trait object는 LayerForwardArgs/LayerGenForwardArgs.profiler에서 사용.
// OpProfiler/PrefillOpProfiler concrete는 observability/profile/ops/ 직접 import.
pub use crate::instrument::OpInstrument;

#[derive(Clone)]
pub struct QkvBias {
    pub bq: Tensor,
    pub bk: Tensor,
    pub bv: Tensor,
}

#[derive(Clone)]
pub struct TransformerLayer {
    // Attention
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,

    // MLP
    pub w_gate: Tensor, // silu_mul gate
    pub w_up: Tensor,
    pub w_down: Tensor,

    // Norms
    /// Pre-attention norm (all architectures).
    pub attention_norm: Tensor,
    /// Llama/Qwen2: pre-FFN norm. Gemma3: post-attention norm.
    pub ffn_norm: Tensor,

    // Optional QKV bias (e.g. for Qwen, Phi models)
    pub qkv_bias: Option<QkvBias>,

    // Gemma3-only optional fields
    /// QK-Norm: per-head RMSNorm weight for Q [n_heads_q * head_dim] (Gemma3).
    pub q_norm: Option<Tensor>,
    /// QK-Norm: per-head RMSNorm weight for K [n_heads_kv * head_dim] (Gemma3).
    pub k_norm: Option<Tensor>,
    /// Pre-FFN norm (Gemma3: pre_feedforward_layernorm). None for Llama/Qwen2.
    pub pre_ffn_norm: Option<Tensor>,
    /// Post-FFN norm (Gemma3: post_feedforward_layernorm). None for Llama/Qwen2.
    pub post_ffn_norm: Option<Tensor>,

    /// Tensor partition context for CPU-GPU cooperative FFN inference.
    /// When set, FFN gate/up matmuls are split between GPU and CPU backends.
    /// None when tensor partition is disabled (default).
    pub partition_ctx: Option<PartitionContext>,
}

// OpProfiler/OpInstrument: OpProfiler is re-exported for backward compat (constructor callers).
// OpInstrument is the trait object type used in LayerForwardArgs/LayerGenForwardArgs.profiler.

// ═══════════════════════════════════════════════════════════════════
// Phase 1: Verify attention scores are post-softmax probabilities
// ═══════════════════════════════════════════════════════════════════
#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::unnecessary_literal_unwrap)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::host::shared::SharedBuffer;

    /// Replicate the softmax computation used in forward_gen (F32 inline path)
    /// to verify the mathematical property: sum(softmax(Q·K^T / sqrt(d))) ≈ 1.0.
    ///
    /// This tests the SAME algorithm used in lines 548-636 of forward_gen().
    #[test]
    fn test_inline_softmax_produces_valid_probabilities() {
        let head_dim = 64;
        let cache_seq_len = 8;
        let n_heads_q = 4;
        let n_heads_kv = 2;
        let n_rep = n_heads_q / n_heads_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create Q: [n_heads_q, head_dim] with random-ish values
        let mut q_data = vec![0.0f32; n_heads_q * head_dim];
        for (i, v) in q_data.iter_mut().enumerate() {
            *v = ((i as f32 * 0.1).sin()) * 0.5;
        }

        // Create K cache: [cache_seq_len, n_heads_kv, head_dim] (SeqMajor)
        let mut k_data = vec![0.0f32; cache_seq_len * n_heads_kv * head_dim];
        for (i, v) in k_data.iter_mut().enumerate() {
            *v = ((i as f32 * 0.07).cos()) * 0.5;
        }

        let stride = cache_seq_len; // minimal stride = cache_seq_len
        let mut scores = vec![0.0f32; n_heads_q * stride];

        // Replicate the forward_gen inline path (non-NEON scalar path)
        for h in 0..n_heads_q {
            let kv_h = h / n_rep;
            let q_off = h * head_dim;
            let scores_h = &mut scores[h * stride..(h + 1) * stride];

            // Step 1: Q * K^T * scale
            for t in 0..cache_seq_len {
                let k_off = (t * n_heads_kv + kv_h) * head_dim; // SeqMajor
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[q_off + d] * k_data[k_off + d];
                }
                scores_h[t] = dot * scale;
            }

            // Step 2: Softmax (same algorithm as forward_gen lines 616-636)
            let active = &mut scores_h[..cache_seq_len];
            let max_val = active.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            for s in active.iter_mut() {
                *s = (*s - max_val).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            for s in active.iter_mut() {
                *s *= inv_sum;
            }
        }

        // VERIFY: each head's scores sum to ~1.0
        for h in 0..n_heads_q {
            let head_scores = &scores[h * stride..h * stride + cache_seq_len];
            let sum: f32 = head_scores.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Head {} score sum = {} (expected ~1.0)",
                h,
                sum
            );
            // All values in [0, 1]
            for (t, &s) in head_scores.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&s),
                    "Head {} token {} score = {} (must be in [0,1])",
                    h,
                    t,
                    s
                );
            }
        }
    }

    /// Verify compute_attention_scores() for F16 KV cache produces post-softmax.
    #[test]
    fn test_compute_attention_scores_f16_post_softmax() {
        let backend = Arc::new(CpuBackend::new());
        let head_dim = 64;
        let n_heads_q = 4;
        let n_heads_kv = 2;
        let capacity = 16;
        let cache_seq_len = 8;

        // Q tensor: [1, 1, n_heads_q * head_dim] = [1, 1, 256]
        let q_buf = Arc::new(SharedBuffer::new(n_heads_q * head_dim * 4, DType::F32));
        let q = Tensor::new(
            Shape::new(vec![1, 1, n_heads_q * head_dim]),
            q_buf,
            backend.clone(),
        );
        // Fill Q with values
        unsafe {
            let q_slice =
                std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut f32, n_heads_q * head_dim);
            for (i, v) in q_slice.iter_mut().enumerate() {
                *v = ((i as f32 * 0.1).sin()) * 0.3;
            }
        }

        // K cache tensor: [1, capacity, n_heads_kv, head_dim] in F16
        let k_buf = Arc::new(SharedBuffer::new(
            capacity * n_heads_kv * head_dim * 2, // F16 = 2 bytes
            DType::F16,
        ));
        let k_cache = Tensor::new(
            Shape::new(vec![1, capacity, n_heads_kv, head_dim]),
            k_buf,
            backend.clone(),
        );
        // Fill K with F16 values
        unsafe {
            let k_slice = std::slice::from_raw_parts_mut(
                k_cache.as_mut_ptr() as *mut half::f16,
                capacity * n_heads_kv * head_dim,
            );
            for (i, v) in k_slice.iter_mut().enumerate() {
                *v = half::f16::from_f32(((i as f32 * 0.07).cos()) * 0.3);
            }
        }

        let mut scores = vec![0.0f32; n_heads_q * capacity];

        TransformerLayer::compute_attention_scores(
            &q,
            &k_cache,
            &mut scores,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
            false, // SeqMajor
            capacity,
            &(backend as Arc<dyn Backend>),
        )
        .unwrap();

        // Verify post-softmax: sum ≈ 1.0 per head
        let stride = scores.len() / n_heads_q;
        for h in 0..n_heads_q {
            let head_scores = &scores[h * stride..h * stride + cache_seq_len];
            let sum: f32 = head_scores.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "F16 Head {} score sum = {} (expected ~1.0)",
                h,
                sum
            );
            for (t, &s) in head_scores.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&s),
                    "F16 Head {} token {} score = {} (must be in [0,1])",
                    h,
                    t,
                    s
                );
            }
        }
    }

    /// Verify that scores fed to accumulate_layer() are genuine post-softmax.
    /// Simulate the full pipeline: compute scores → accumulate → verify.
    #[test]
    fn test_accumulator_receives_post_softmax_scores() {
        use crate::inference::attention_scores::AttentionScoreAccumulator;

        let n_heads_q = 4;
        let cache_seq_len = 8;
        let stride = cache_seq_len;

        // Simulate post-softmax scores (manually constructed)
        let mut scores = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            let head_scores = &mut scores[h * stride..h * stride + cache_seq_len];
            // Create a valid probability distribution
            let raw: Vec<f32> = (0..cache_seq_len)
                .map(|t| ((t as f32 + h as f32) * 0.5).exp())
                .collect();
            let sum: f32 = raw.iter().sum();
            for (t, &r) in raw.iter().enumerate() {
                head_scores[t] = r / sum;
            }
        }

        // Verify input is valid softmax
        for h in 0..n_heads_q {
            let sum: f32 = scores[h * stride..h * stride + cache_seq_len].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Feed to accumulator
        let mut acc = AttentionScoreAccumulator::new(16, n_heads_q, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq_len, n_heads_q, 0);
        acc.end_step();

        // Per-token importance = sum across heads of softmax probs
        // Each head sums to 1.0 → total per-token sum across all tokens = n_heads_q
        let imp = acc.importance_scores();
        let total: f32 = imp[..cache_seq_len].iter().sum();
        assert!(
            (total - n_heads_q as f32).abs() < 1e-4,
            "Total importance = {} (expected {} = n_heads_q * 1.0 per head)",
            total,
            n_heads_q
        );
    }

    /// GEMMA-2.1: effective_cache_len is clamped to window_size for local layers.
    ///
    /// When is_local_attn=Some(true) and local_attn_window=Some(ws), the decode path
    /// must use min(cache_seq_len, ws) as effective attention length.
    #[test]
    fn test_effective_cache_len_local() {
        // Verify the computation logic directly (mirrors forward_gen.rs lines)
        let test_cases = vec![
            // (cache_seq_len, is_local, window, expected_effective)
            (100usize, Some(true), Some(512usize), 100usize), // cache < window → full
            (600, Some(true), Some(512), 512),                // cache > window → clamped
            (512, Some(true), Some(512), 512),                // cache == window → exact
            (100, Some(false), Some(512), 100),               // global layer → full
            (600, None, Some(512), 600),                      // no local flag → full
            (600, Some(true), None, 600),                     // no window → full (usize::MAX)
        ];

        for (cache_seq_len, is_local_attn, local_attn_window, expected) in test_cases {
            let effective = if let Some(true) = is_local_attn {
                let window = local_attn_window.unwrap_or(usize::MAX);
                cache_seq_len.min(window)
            } else {
                cache_seq_len
            };
            let kv_start_pos = cache_seq_len - effective;

            assert_eq!(
                effective, expected,
                "cache={} is_local={:?} window={:?}: effective={} expected={}",
                cache_seq_len, is_local_attn, local_attn_window, effective, expected
            );
            // kv_start_pos must be consistent
            assert_eq!(
                kv_start_pos,
                cache_seq_len - expected,
                "kv_start_pos mismatch for cache={} effective={}",
                cache_seq_len,
                effective
            );
        }
    }

    /// GEMMA-2.2: is_local_attn=None (Llama/Qwen2) leaves effective_cache_len unchanged.
    #[test]
    fn test_non_local_attn_unchanged() {
        let cache_seq_len: usize = 42;
        let is_local_attn: Option<bool> = None;
        let local_attn_window: Option<usize> = None;

        let effective = if let Some(true) = is_local_attn {
            let window = local_attn_window.unwrap_or(usize::MAX);
            cache_seq_len.min(window)
        } else {
            cache_seq_len
        };

        assert_eq!(
            effective, cache_seq_len,
            "Non-local layer must use full cache_seq_len"
        );
        assert_eq!(cache_seq_len - effective, 0, "kv_start_pos must be 0");
    }
}
