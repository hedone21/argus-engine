//! Unified QCF (Quality Cost Function) metric for all KV cache actions.
//!
//! Core formula:
//!   QCF = ||O_before - O_after|| / ||O_before||
//!   O = sum_t alpha_t * V_t   (attention-weighted value sum)
//!
//! This module owns the **generic** half of the metric: the per-head O_before, the redistribution
//! α-weights (per-head attention with a flat-score fallback), the L2 / aggregation math, and the
//! dtype/layout V/K reader ([`read_v_f32`]). The **technique-specific** O_after — which tokens a
//! policy would retain and how their value vectors recombine — lives in each eviction technique
//! crate behind the [`QcfEstimator`] trait (observer/score axis, EPIC 2). The engine no longer
//! duplicates any concrete technique's arithmetic here: [`compute_qcf_kv`] builds an
//! [`EstimatorCtx`] over the cache and calls `estimator.o_after()`.

use super::{AggregationMode, aggregate_heads};
use crate::buffer::DType;
use crate::kv_cache_ops::KVLayout;
use crate::quant::{BlockQ4_0, QK4_0};
use crate::tensor::Tensor;
use argus_extension_api::{EstimatorCtx, QcfEstimator};

// ── V/K data source abstraction ─────────────────────────────────

/// Abstraction over KV buffer data types for read-only access.
///
/// Despite the historical `V` prefix the same enum is reused for both K and V
/// cache slices: the variants only encode the underlying dtype.
pub enum VDataSource<'a> {
    /// F32 cache data.
    F32(&'a [f32]),
    /// F16 cache data stored as raw u16 (half::f16 bit representation).
    F16(&'a [u16]),
    /// Q4_0 cache data stored as BlockQ4_0 blocks.
    Q4_0(&'a [BlockQ4_0]),
}

impl<'a> VDataSource<'a> {
    /// Build a `VDataSource` view from a Tensor buffer (V or K).
    ///
    /// - `cpu_bytes = Some(...)`: GPU backend with explicit host readback;
    ///   the byte slice is reinterpreted by `dtype`.
    /// - `cpu_bytes = None`: read directly from `buffer.as_slice()`.
    ///   Returns `None` if the host pointer is null (device-only buffer).
    ///
    /// Returns `None` for unsupported dtypes.
    ///
    /// 본 함수는 S-3b-3에서 KVCache struct 의존을 제거하기 위해 `&Tensor`
    /// 매개변수를 받도록 변경되었다 (β 옵션). caller는 `cache.v_buffer` /
    /// `cache.k_buffer` 를 직접 전달한다.
    pub fn from_buffer(buffer: &'a Tensor, cpu_bytes: Option<&'a [u8]>) -> Option<Self> {
        let dtype = buffer.dtype();
        if let Some(bytes) = cpu_bytes {
            return Some(match dtype {
                DType::F32 => {
                    let elems = bytes.len() / std::mem::size_of::<f32>();
                    let ptr = bytes.as_ptr() as *const f32;
                    VDataSource::F32(unsafe { std::slice::from_raw_parts(ptr, elems) })
                }
                DType::F16 => {
                    let elems = bytes.len() / std::mem::size_of::<u16>();
                    let ptr = bytes.as_ptr() as *const u16;
                    VDataSource::F16(unsafe { std::slice::from_raw_parts(ptr, elems) })
                }
                DType::Q4_0 => {
                    let n_blocks = bytes.len() / std::mem::size_of::<BlockQ4_0>();
                    let ptr = bytes.as_ptr() as *const BlockQ4_0;
                    VDataSource::Q4_0(unsafe { std::slice::from_raw_parts(ptr, n_blocks) })
                }
                _ => return None,
            });
        }
        if buffer.buffer().as_ptr().is_null() {
            return None;
        }
        Some(match dtype {
            DType::F32 => VDataSource::F32(buffer.as_slice::<f32>()),
            DType::F16 => VDataSource::F16(buffer.as_slice::<u16>()),
            DType::Q4_0 => VDataSource::Q4_0(buffer.as_slice::<BlockQ4_0>()),
            _ => return None,
        })
    }
}

// ── Parameters ──────────────────────────────────────────────────

/// All inputs needed to compute the unified QCF metric for one registered technique.
pub struct QcfKvParams<'a> {
    /// The technique's degradation estimator (the producer of O_after). Resolved by the caller via
    /// `argus_extension_api::find_qcf_estimator(name)` / `reg.make(...)`.
    pub estimator: &'a dyn QcfEstimator,
    /// Post-eviction token budget the estimate simulates (e.g. `current_pos / 2`). The estimator
    /// reads it via [`EstimatorCtx::target_len`].
    pub target_len: usize,
    /// V buffer data (F32, F16, or Q4_0).
    pub v_source: VDataSource<'a>,
    /// Optional K buffer data, consumed by techniques that need per-head nearest-token matching
    /// (d2o). When `None`, [`EstimatorCtx::read_k`] returns `false` and the estimator falls back to
    /// V. Other techniques ignore it.
    pub k_source: Option<VDataSource<'a>>,
    /// Flat importance scores, layout `[max_seq_len]`.
    pub attention_scores: &'a [f32],
    /// Optional per-KV-head attention, layout `[n_kv_heads * max_seq_len]`.
    pub head_attn: Option<&'a [f32]>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub current_pos: usize,
    pub capacity: usize,
    pub layout: KVLayout,
    pub aggregation: AggregationMode,
    /// β exponent for redistributed-attention amplification (ARGUS QCF #6).
    /// Baseline = β=1.0; β > 1 emphasises high-attention tokens in sparse
    /// distributions. Default: 1.0 (no amplification, bit-identical to legacy).
    pub beta: f32,
}

// ── EstimatorCtx adapter over QcfKvParams ───────────────────────

/// Engine-side [`EstimatorCtx`] over a [`QcfKvParams`]: it lends the estimator the per-head α-weights
/// and the dtype/layout-aware V/K reads it needs to build O_after, without exposing any engine type.
struct ParamsCtx<'a, 'p> {
    p: &'p QcfKvParams<'a>,
}

impl EstimatorCtx for ParamsCtx<'_, '_> {
    fn current_pos(&self) -> usize {
        self.p.current_pos
    }
    fn target_len(&self) -> usize {
        self.p.target_len
    }
    fn n_kv_heads(&self) -> usize {
        self.p.n_kv_heads
    }
    fn head_dim(&self) -> usize {
        self.p.head_dim
    }
    fn beta(&self) -> f32 {
        self.p.beta
    }
    fn alpha_h(&self, kv_head: usize, out: &mut [f32]) {
        build_alpha_h(
            self.p.head_attn,
            self.p.attention_scores,
            self.p.n_kv_heads,
            self.p.current_pos,
            kv_head,
            out,
        );
    }
    fn read_v(&self, kv_head: usize, pos: usize, out: &mut [f32]) {
        let v = read_v_f32(
            &self.p.v_source,
            kv_head,
            pos,
            self.p.head_dim,
            self.p.capacity,
            self.p.n_kv_heads,
            self.p.layout,
        );
        // `read_v_f32` returns ≤ head_dim elements (a partial slice when the buffer is short); zero-fill
        // the tail so out-of-range reads degrade to zeros per the EstimatorCtx contract (vs panicking).
        out.fill(0.0);
        out[..v.len()].copy_from_slice(&v);
    }
    fn read_k(&self, kv_head: usize, pos: usize, out: &mut [f32]) -> bool {
        match &self.p.k_source {
            Some(k) => {
                let v = read_v_f32(
                    k,
                    kv_head,
                    pos,
                    self.p.head_dim,
                    self.p.capacity,
                    self.p.n_kv_heads,
                    self.p.layout,
                );
                out.fill(0.0);
                out[..v.len()].copy_from_slice(&v);
                true
            }
            None => false,
        }
    }
}

/// Build the per-head redistribution weights α_h[t] (`out.len() == current_pos`).
///
/// Tries per-head attention first; falls back to the flat scores when the per-head slice is absent or
/// all-zero (the latter can happen when softmax produces NaN in attention weights, which the NaN
/// guard converts to 0). This is the engine-side α builder shared by O_before and the estimator's
/// O_after, so both share the same softmax-weight space.
fn build_alpha_h(
    head_attn: Option<&[f32]>,
    attention_scores: &[f32],
    n_kv_heads: usize,
    current_pos: usize,
    h: usize,
    out: &mut [f32],
) {
    let max_seq_len = attention_scores.len();
    let mut have = false;
    if let Some(ha) = head_attn {
        let head_offset = h * (ha.len() / n_kv_heads.max(1));
        let mut sum = 0.0f32;
        for (t, slot) in out.iter_mut().enumerate().take(current_pos) {
            let idx = head_offset + t;
            let v = if idx < ha.len() { ha[idx] } else { 0.0 };
            *slot = v;
            sum += v;
        }
        // Fall back to flat scores when the per-head α-sum is non-positive (matches legacy).
        have = sum > 0.0;
    }
    if !have {
        for (t, slot) in out.iter_mut().enumerate().take(current_pos) {
            *slot = if t < max_seq_len {
                attention_scores[t]
            } else {
                0.0
            };
        }
    }
}

// ── Main entry point ────────────────────────────────────────────

/// Compute unified QCF for the technique carried by `params.estimator`.
///
/// Returns `(aggregated_qcf, per_head_qcf)`. The harness owns O_before, the L2 / aggregation math and
/// β; the estimator owns only the per-head O_after (returning `false` for a within-budget no-op, in
/// which case O_after == O_before and the head contributes QCF 0).
pub fn compute_qcf_kv(params: &QcfKvParams) -> (f32, Vec<f32>) {
    let _t = crate::qcf_timer!(QCF_KV_UNIFIED);
    let n_kv_heads = params.n_kv_heads;
    let head_dim = params.head_dim;
    let current_pos = params.current_pos;

    if n_kv_heads == 0 || head_dim == 0 || current_pos == 0 {
        return (0.0, vec![0.0; n_kv_heads]);
    }

    let ctx = ParamsCtx { p: params };
    let mut per_head = vec![0.0f32; n_kv_heads];
    let mut alpha = vec![0.0f32; current_pos];
    let mut o_before = vec![0.0f32; head_dim];
    let mut o_after = vec![0.0f32; head_dim];
    let mut v_t = vec![0.0f32; head_dim];

    for (h, ph) in per_head.iter_mut().enumerate() {
        // 1. Per-head α (per-head attention with flat-score fallback).
        ctx.alpha_h(h, &mut alpha);

        // 2. O_before = Σ_t (α_t / Σ_s α_s) · V[h][t]   (ENG-ALG-051)
        //    O_before is normalised by the **full** token-set α-sum so it shares the same
        //    softmax-weight space as O_after's retained-set re-normalisation.
        let alpha_all_sum: f32 = alpha.iter().sum();
        for x in o_before.iter_mut() {
            *x = 0.0;
        }
        if alpha_all_sum > 0.0 {
            for (t, &alpha_t) in alpha.iter().enumerate() {
                ctx.read_v(h, t, &mut v_t);
                let w = alpha_t / alpha_all_sum;
                for d in 0..head_dim {
                    o_before[d] += w * v_t[d];
                }
            }
        }
        // alpha_all_sum <= 0 → o_before stays the zero vector, which routes to the QCF=0 branch
        // below (o_norm <= ε), consistent with the existing zero-guard.

        // 3. O_after via the technique's estimator. `false` = within-budget no-op → O_after = O_before.
        let produced = params.estimator.o_after(&ctx, h, &mut o_after);
        let o_after_ref: &[f32] = if produced { &o_after } else { &o_before };

        // 4. QCF = ||O_before - O_after|| / ||O_before||
        let diff_norm = l2_norm_diff(&o_before, o_after_ref);
        let o_norm = l2_norm(&o_before);
        *ph = if o_norm > 1e-10 {
            diff_norm / o_norm
        } else {
            0.0
        };
    }

    let qcf = aggregate_heads(&per_head, &params.aggregation);
    (qcf, per_head)
}

// ── Helper: read V vector as f32 ────────────────────────────────

fn read_v_f32(
    src: &VDataSource,
    head: usize,
    pos: usize,
    head_dim: usize,
    capacity: usize,
    n_kv_heads: usize,
    layout: KVLayout,
) -> Vec<f32> {
    let offset = compute_v_offset(layout, head, pos, head_dim, capacity, n_kv_heads);
    match src {
        VDataSource::F32(data) => {
            let end = (offset + head_dim).min(data.len());
            if offset >= data.len() {
                return vec![0.0; head_dim];
            }
            data[offset..end].to_vec()
        }
        VDataSource::F16(data) => {
            let end = (offset + head_dim).min(data.len());
            if offset >= data.len() {
                return vec![0.0; head_dim];
            }
            data[offset..end]
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect()
        }
        VDataSource::Q4_0(data) => {
            // Q4_0: blocks_per_pos = head_dim / QK4_0 (e.g. 64/32=2, 256/32=8)
            let blocks_per_pos = head_dim / QK4_0;
            let block_idx = match layout {
                KVLayout::HeadMajor => (head * capacity + pos) * blocks_per_pos,
                KVLayout::SeqMajor => (pos * n_kv_heads + head) * blocks_per_pos,
            };
            if block_idx >= data.len() {
                return vec![0.0; head_dim];
            }
            let mut out = vec![0.0f32; head_dim];
            let mut buf = [0.0f32; QK4_0];
            for b in 0..blocks_per_pos {
                let bi = block_idx + b;
                if bi >= data.len() {
                    break;
                }
                data[bi].dequantize(&mut buf);
                let dst_start = b * QK4_0;
                let dst_end = (dst_start + QK4_0).min(head_dim);
                out[dst_start..dst_end].copy_from_slice(&buf[..dst_end - dst_start]);
            }
            out
        }
    }
}

fn compute_v_offset(
    layout: KVLayout,
    head: usize,
    pos: usize,
    head_dim: usize,
    capacity: usize,
    n_kv_heads: usize,
) -> usize {
    match layout {
        KVLayout::HeadMajor => head * capacity * head_dim + pos * head_dim,
        KVLayout::SeqMajor => pos * n_kv_heads * head_dim + head * head_dim,
    }
}

// ── Math helpers ────────────────────────────────────────────────

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn l2_norm_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::erasing_op)]
mod tests {
    use super::*;
    use argus_extension_api::{StageParams, find_qcf_estimator};

    /// Which registered technique to drive in a test, plus its estimate parameters.
    enum TestAction {
        Sliding {
            target_len: usize,
        },
        H2o {
            hh_size: usize,
            recent_size: usize,
            protected_prefix: usize,
        },
        Streaming {
            sink_size: usize,
            window_size: usize,
        },
        D2o {
            target_len: usize,
            keep_ratio: f32,
            protected_prefix: usize,
        },
    }

    /// Resolve the registered estimator for `action`, build a [`QcfKvParams`], and run the unified
    /// metric. This exercises the production estimator + harness path (the former action-enum tests
    /// drove the same arithmetic in-engine).
    #[allow(clippy::too_many_arguments)]
    fn run(
        action: TestAction,
        v_source: VDataSource,
        k_source: Option<VDataSource>,
        attention_scores: &[f32],
        head_attn: Option<&[f32]>,
        n_kv_heads: usize,
        head_dim: usize,
        current_pos: usize,
        capacity: usize,
        layout: KVLayout,
        beta: f32,
    ) -> (f32, Vec<f32>) {
        // Faithful h2o takes ABSOLUTE budgets via the `--set` blob; other techniques use StageParams.
        let (name, sp, target_len, args): (
            &str,
            StageParams,
            usize,
            Vec<argus_extension_api::PluginArg<'static>>,
        ) = match action {
            TestAction::Sliding { target_len } => {
                ("sliding", StageParams::default(), target_len, vec![])
            }
            TestAction::H2o {
                hh_size,
                recent_size,
                protected_prefix,
            } => (
                "h2o",
                StageParams {
                    protected_prefix,
                    ..Default::default()
                },
                hh_size + recent_size + protected_prefix,
                vec![
                    argus_extension_api::PluginArg {
                        key: "hh_size",
                        val: Box::leak(hh_size.to_string().into_boxed_str()),
                    },
                    argus_extension_api::PluginArg {
                        key: "recent_size",
                        val: Box::leak(recent_size.to_string().into_boxed_str()),
                    },
                ],
            ),
            TestAction::Streaming {
                sink_size,
                window_size,
            } => (
                "streaming",
                StageParams {
                    sink_size,
                    streaming_window: window_size,
                    ..Default::default()
                },
                // streaming reads sink+window from its config, not target_len.
                0,
                vec![],
            ),
            TestAction::D2o {
                target_len,
                keep_ratio,
                protected_prefix,
            } => (
                "d2o",
                StageParams {
                    keep_ratio,
                    protected_prefix,
                    ..Default::default()
                },
                target_len,
                vec![],
            ),
        };
        let est = (find_qcf_estimator(name)
            .unwrap_or_else(|| panic!("estimator '{name}' registered"))
            .make)(sp, &args);
        let params = QcfKvParams {
            estimator: &*est,
            target_len,
            v_source,
            k_source,
            attention_scores,
            head_attn,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout,
            aggregation: AggregationMode::Mean,
            beta,
        };
        compute_qcf_kv(&params)
    }

    /// Build a simple HeadMajor V buffer for testing.
    /// V[h][t][d] = (h+1) * (t+1) * (d+1) as f32, giving predictable values.
    fn make_v_data(n_kv_heads: usize, capacity: usize, head_dim: usize) -> Vec<f32> {
        let total = n_kv_heads * capacity * head_dim;
        let mut data = vec![0.0f32; total];
        for h in 0..n_kv_heads {
            for t in 0..capacity {
                for d in 0..head_dim {
                    let offset = h * capacity * head_dim + t * head_dim + d;
                    data[offset] = (h as f32 + 1.0) * (t as f32 + 1.0) * (d as f32 + 1.0);
                }
            }
        }
        data
    }

    /// Uniform attention scores for testing.
    fn uniform_scores(n: usize) -> Vec<f32> {
        vec![1.0 / n as f32; n]
    }

    /// Build a HeadMajor V/K buffer with **directional diversity** for weighted-merge.
    ///
    /// `make_v_data` gives every token the same direction (V(h,t,d) ∝ (d+1)),
    /// so cosine similarity is uniformly 1.0 and weighted-merge's nearest-token merge can
    /// not discriminate. For weighted-merge to beat heavy-hitter the data must satisfy two
    /// conditions: directional diversity (distinct token directions) and
    /// nearest alignment (an evicted token has a same-direction retained twin).
    /// Tokens 1 and 2 form class B (the heavy hitter + its mergeable twin); all
    /// others class A. Unit magnitude keeps the merge norm-preserving.
    fn make_diverse_v_data(n_kv_heads: usize, capacity: usize, head_dim: usize) -> Vec<f32> {
        assert!(head_dim >= 32, "diverse V needs head_dim >= 32");
        let freq = 0.37f32;
        let total = n_kv_heads * capacity * head_dim;
        let mut data = vec![0.0f32; total];
        for h in 0..n_kv_heads {
            for t in 0..capacity {
                let class_phase = if t == 1 || t == 2 {
                    std::f32::consts::FRAC_PI_2
                } else {
                    0.0
                };
                let phase = class_phase + (h as f32) * 0.05;
                for d in 0..head_dim {
                    let offset = h * capacity * head_dim + t * head_dim + d;
                    data[offset] = (phase + d as f32 * freq).sin();
                }
            }
        }
        data
    }

    #[test]
    fn test_zero_change_sliding() {
        // target_len == current_pos -> nothing evicted -> QCF = 0
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let (qcf, per_head) = run(
            TestAction::Sliding {
                target_len: current_pos,
            },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        assert!(
            qcf.abs() < 1e-6,
            "expected QCF=0 when nothing evicted, got {qcf}"
        );
        for (h, &v) in per_head.iter().enumerate() {
            assert!(v.abs() < 1e-6, "head {h}: expected 0, got {v}");
        }
    }

    #[test]
    fn test_full_eviction() {
        // target_len = 0 -> everything evicted -> QCF should be high
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let (qcf, _) = run(
            TestAction::Sliding { target_len: 0 },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        // With 0 retained tokens, O_after = 0, so QCF = ||O_before|| / ||O_before|| = 1.0
        assert!(
            (qcf - 1.0).abs() < 1e-5,
            "expected QCF near 1.0 for full eviction, got {qcf}"
        );
    }

    #[test]
    fn test_eviction_monotonicity() {
        // More tokens evicted -> higher QCF
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 16;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let run_keep = |target_len: usize| {
            run(
                TestAction::Sliding { target_len },
                VDataSource::F32(&v_data),
                None,
                &scores,
                None,
                n_kv_heads,
                head_dim,
                current_pos,
                capacity,
                KVLayout::HeadMajor,
                1.0,
            )
            .0
        };

        let qcf_keep_12 = run_keep(12);
        let qcf_keep_8 = run_keep(8);
        let qcf_keep_4 = run_keep(4);

        assert!(
            qcf_keep_4 > qcf_keep_8,
            "keeping 4 ({qcf_keep_4}) should give higher QCF than keeping 8 ({qcf_keep_8})"
        );
        assert!(
            qcf_keep_8 > qcf_keep_12,
            "keeping 8 ({qcf_keep_8}) should give higher QCF than keeping 12 ({qcf_keep_12})"
        );
    }

    #[test]
    fn test_streaming_sink_window_no_eviction() {
        // sink + window >= current_pos -> nothing evicted -> QCF = 0
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let (qcf, _) = run(
            TestAction::Streaming {
                sink_size: 4,
                window_size: 4,
            },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        assert!(
            qcf.abs() < 1e-6,
            "expected QCF=0 when sink+window covers all tokens, got {qcf}"
        );
    }

    #[test]
    fn test_h2o_vs_sliding() {
        // heavy-hitter should have QCF <= Sliding at same target_len when importance and V norms correlate.
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 16;

        // V data where early (important) tokens have large V norms.
        let mut v_data = vec![0.0f32; n_kv_heads * capacity * head_dim];
        for t in 0..current_pos {
            for d in 0..head_dim {
                let offset = t * head_dim + d;
                v_data[offset] = (current_pos - t) as f32 * (d as f32 + 1.0);
            }
        }

        // Non-uniform: early tokens have very high importance.
        let mut scores = vec![0.1f32; current_pos];
        scores[0] = 10.0;
        scores[1] = 8.0;
        scores[2] = 6.0;
        scores[3] = 5.0;

        let target_len = 8;

        let (qcf_sliding, _) = run(
            TestAction::Sliding { target_len },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        let (qcf_h2o, _) = run(
            TestAction::H2o {
                // faithful absolute budget reproducing the old keep_ratio=0.5 of target_len=8.
                hh_size: 4,
                recent_size: 4,
                protected_prefix: 0,
            },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );

        assert!(
            qcf_h2o <= qcf_sliding + 1e-6,
            "heavy-hitter ({qcf_h2o}) should have QCF <= Sliding ({qcf_sliding}) \
             when important tokens have large V norms"
        );
    }

    #[test]
    fn test_f16_data_source() {
        // Verify F16 VDataSource works correctly.
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 8;
        let current_pos = 4;

        let v_f32 = make_v_data(n_kv_heads, capacity, head_dim);
        let v_f16: Vec<u16> = v_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let scores = uniform_scores(current_pos);

        let (qcf_f32, _) = run(
            TestAction::Sliding { target_len: 2 },
            VDataSource::F32(&v_f32),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        let (qcf_f16, _) = run(
            TestAction::Sliding { target_len: 2 },
            VDataSource::F16(&v_f16),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );

        assert!(
            (qcf_f32 - qcf_f16).abs() < 0.05,
            "F32 ({qcf_f32}) and F16 ({qcf_f16}) QCF should be close"
        );
    }

    #[test]
    fn test_streaming_evicts_middle() {
        // StreamingLLM: retains sink + recent, evicts the middle.
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 16;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let (qcf, _) = run(
            TestAction::Streaming {
                sink_size: 2,
                window_size: 4,
            },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        assert!(
            qcf > 0.0,
            "StreamingLLM eviction should produce non-zero QCF"
        );
        assert!(qcf < 1.0, "QCF should be bounded below 1.0, got {qcf}");
    }

    #[test]
    fn test_per_head_attn_different_from_flat() {
        // When per-head attention differs, results should diverge from flat scores.
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);

        let flat_scores = uniform_scores(current_pos);

        // Per-head: head 0 focuses on early tokens, head 1 on late tokens.
        let mut head_attn = vec![0.0f32; n_kv_heads * current_pos];
        for t in 0..current_pos {
            head_attn[0 * current_pos + t] = (current_pos - t) as f32;
            head_attn[current_pos + t] = (t + 1) as f32;
        }

        let (qcf_flat, _ph_flat) = run(
            TestAction::Sliding { target_len: 4 },
            VDataSource::F32(&v_data),
            None,
            &flat_scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        let (qcf_head, ph_head) = run(
            TestAction::Sliding { target_len: 4 },
            VDataSource::F32(&v_data),
            None,
            &flat_scores,
            Some(&head_attn),
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );

        assert!(
            (ph_head[0] - ph_head[1]).abs() > 1e-3,
            "per-head attn should produce different per-head QCFs: {ph_head:?}"
        );
        assert!(
            (qcf_flat - qcf_head).abs() > 1e-6,
            "flat ({qcf_flat}) vs per-head ({qcf_head}) should differ"
        );
    }

    #[test]
    fn test_d2o_less_than_h2o() {
        // weighted-merge additively merges each evicted token into its cosine-nearest retained token
        // (magnitude-preserving merge), partially restoring the evicted direction, so with directional diversity +
        // nearest alignment weighted-merge's QCF is ≤ heavy-hitter's. (d2o retained set re-baselined to plan() per
        // EPIC 2 Stage A decision #2; the relational property is unaffected.)
        let n_kv_heads = 2;
        let head_dim = 32;
        let capacity = 8;
        let current_pos = 4;
        let v_data = make_diverse_v_data(n_kv_heads, capacity, head_dim);
        let k_data = make_diverse_v_data(n_kv_heads, capacity, head_dim);

        let scores = vec![1.0f32, 9.0, 8.0, 1.0];
        let target_len = 2;
        let keep_ratio = 0.5;
        let protected_prefix = 1;

        let (qcf_h2o, ph_h2o) = run(
            TestAction::H2o {
                // faithful absolute budget reproducing the old keep_ratio=0.5 of target_len=2.
                hh_size: 1,
                recent_size: 1,
                protected_prefix,
            },
            VDataSource::F32(&v_data),
            Some(VDataSource::F32(&k_data)),
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        let (qcf_d2o, ph_d2o) = run(
            TestAction::D2o {
                target_len,
                keep_ratio,
                protected_prefix,
            },
            VDataSource::F32(&v_data),
            Some(VDataSource::F32(&k_data)),
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );

        assert!(
            qcf_d2o < qcf_h2o,
            "weighted-merge ({qcf_d2o}) should have lower QCF than heavy-hitter ({qcf_h2o})"
        );
        assert!(
            qcf_h2o > 0.0,
            "heavy-hitter QCF should be positive, got {qcf_h2o}"
        );
        for h in 0..n_kv_heads {
            assert!(
                ph_d2o[h] <= ph_h2o[h] + 1e-6,
                "head {h}: weighted-merge ({}) should have QCF <= heavy-hitter ({})",
                ph_d2o[h],
                ph_h2o[h]
            );
        }
    }

    #[test]
    fn test_d2o_no_eviction_equals_zero() {
        // When current_pos <= keep, no eviction happens, QCF = 0.
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let (qcf, _) = run(
            TestAction::D2o {
                target_len: current_pos,
                keep_ratio: 0.5,
                protected_prefix: 2,
            },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        assert!(
            qcf.abs() < 1e-6,
            "weighted-merge with no eviction should give QCF=0, got {qcf}"
        );
    }

    #[test]
    fn test_d2o_uses_k_for_nearest() {
        // When K and V give different nearest matches, supplying K changes the result vs V fallback.
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 8;

        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        // K deliberately inverted vs V so K-based nearest != V-based nearest.
        let mut k_data = vec![0.0f32; n_kv_heads * capacity * head_dim];
        for t in 0..current_pos {
            for d in 0..head_dim {
                let off = t * head_dim + d;
                k_data[off] = (current_pos as f32 - t as f32) * (d as f32 + 1.0);
            }
        }

        let mut scores = vec![0.1f32; current_pos];
        scores[0] = 5.0;
        scores[1] = 3.0;

        let target_len = 4;
        let keep_ratio = 0.5;
        let protected_prefix = 1;

        let (qcf_v, _) = run(
            TestAction::D2o {
                target_len,
                keep_ratio,
                protected_prefix,
            },
            VDataSource::F32(&v_data),
            None, // V fallback
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        let (qcf_k, _) = run(
            TestAction::D2o {
                target_len,
                keep_ratio,
                protected_prefix,
            },
            VDataSource::F32(&v_data),
            Some(VDataSource::F32(&k_data)),
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );

        assert!(
            (qcf_v - qcf_k).abs() > 1e-6,
            "K-based nearest ({qcf_k}) should differ from V-fallback ({qcf_v}) \
             when K and V geometries disagree"
        );
    }

    #[test]
    fn test_d2o_weight_grouping_bounded() {
        // the merge weights sum to 1 by construction (magnitude preserving), so the merged-V
        // redistribution keeps QCF in [0, 1]. (Post-rebaseline the retained set is the plan()
        // 3-partition clamp, so the former "single retained token" reconstruction no longer
        // applies; we assert the convex-hull bound instead.)
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let (qcf, _) = run(
            TestAction::D2o {
                target_len: 1,
                keep_ratio: 1.0,
                protected_prefix: 0,
            },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        assert!(
            (0.0..=1.0 + 1e-4).contains(&qcf),
            "weighted-merge merge QCF should be bounded in [0, 1], got {qcf}"
        );
    }

    // ── β-amplification tests (ARGUS QCF Step 3) ────────────────

    #[test]
    fn test_compute_qcf_kv_beta_one_matches_hand_computed() {
        // n_kv_heads=1, head_dim=2, current_pos=3, target_len=2, V[t][d]=(t+1)*(d+1), uniform α.
        // Retained={1,2}, O_after=[2.5,5.0], O_before=[2.0,4.0] → QCF ≈ 0.25.
        let n_kv_heads = 1;
        let head_dim = 2;
        let capacity = 8;
        let current_pos = 3;

        let mut v_data = vec![0.0f32; n_kv_heads * capacity * head_dim];
        for t in 0..current_pos {
            for d in 0..head_dim {
                let offset = t * head_dim + d;
                v_data[offset] = (t as f32 + 1.0) * (d as f32 + 1.0);
            }
        }
        let scores = uniform_scores(current_pos);

        let (qcf, _) = run(
            TestAction::Sliding { target_len: 2 },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        assert!(
            (qcf - 0.25).abs() < 1e-5,
            "β=1 QCF expected ~0.25, got {qcf}"
        );
    }

    #[test]
    fn test_compute_qcf_kv_beta_amplifies_non_uniform() {
        // For non-uniform retained scores β=2 should differ from β=1.
        let n_kv_heads = 1;
        let head_dim = 2;
        let capacity = 8;
        let current_pos = 3;

        let mut v_data = vec![0.0f32; n_kv_heads * capacity * head_dim];
        for t in 0..current_pos {
            for d in 0..head_dim {
                let offset = t * head_dim + d;
                v_data[offset] = (t as f32 + 1.0) * (d as f32 + 1.0);
            }
        }
        let scores = vec![0.1f32, 0.3, 0.6];

        let (qcf_b1, _) = run(
            TestAction::Sliding { target_len: 2 },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            1.0,
        );
        let (qcf_b2, _) = run(
            TestAction::Sliding { target_len: 2 },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            2.0,
        );

        assert!(
            (qcf_b1 - qcf_b2).abs() > 1e-5,
            "β=1 ({qcf_b1}) and β=2 ({qcf_b2}) should differ for non-uniform scores"
        );
    }

    #[test]
    fn test_compute_qcf_kv_beta_zero_uniform() {
        // β=0: α_t^0 = 1 for all t → equal weights over the retained set.
        let n_kv_heads = 1;
        let head_dim = 2;
        let capacity = 8;
        let current_pos = 4;

        let v_data: Vec<f32> = (0..n_kv_heads * capacity * head_dim)
            .map(|i| i as f32 + 1.0)
            .collect();
        let scores = vec![0.1f32, 5.0, 2.0, 0.5];

        let (qcf_b0, _) = run(
            TestAction::Sliding { target_len: 3 },
            VDataSource::F32(&v_data),
            None,
            &scores,
            None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            KVLayout::HeadMajor,
            0.0,
        );

        // Production usage note: β=0 is NOT a supported production value (default β=1).
        assert!(
            qcf_b0 >= 0.0,
            "β=0 QCF should be non-negative, got {qcf_b0}"
        );
        assert!(
            qcf_b0 <= 1.5,
            "β=0 QCF should be bounded (≤1.5), got {qcf_b0}"
        );
    }

    // ── Regression tests for ISSUE-6 (read_v_f32 tolerates undersized slices) ──

    #[test]
    fn test_read_v_f32_empty_f16_returns_zeros() {
        let empty: Vec<u16> = Vec::new();
        let out = read_v_f32(
            &VDataSource::F16(&empty),
            0,
            0,
            64,
            128,
            8,
            KVLayout::HeadMajor,
        );
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&x| x == 0.0));

        let out2 = read_v_f32(
            &VDataSource::F16(&empty),
            3,
            17,
            64,
            128,
            8,
            KVLayout::HeadMajor,
        );
        assert_eq!(out2.len(), 64);
        assert!(out2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_read_v_f32_empty_f32_returns_zeros() {
        let empty: Vec<f32> = Vec::new();
        let out = read_v_f32(
            &VDataSource::F32(&empty),
            0,
            0,
            64,
            128,
            8,
            KVLayout::HeadMajor,
        );
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&x| x == 0.0));

        let short = vec![1.0f32; 10];
        let out2 = read_v_f32(
            &VDataSource::F32(&short),
            1,
            0,
            64,
            128,
            8,
            KVLayout::HeadMajor,
        );
        assert_eq!(out2.len(), 64);
        assert!(out2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_read_v_f32_empty_q4_returns_zeros() {
        let empty: Vec<BlockQ4_0> = Vec::new();
        let out = read_v_f32(
            &VDataSource::Q4_0(&empty),
            0,
            0,
            64,
            128,
            8,
            KVLayout::HeadMajor,
        );
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&x| x == 0.0));

        let out2 = read_v_f32(
            &VDataSource::Q4_0(&empty),
            2,
            5,
            64,
            128,
            8,
            KVLayout::HeadMajor,
        );
        assert_eq!(out2.len(), 64);
        assert!(out2.iter().all(|&x| x == 0.0));
    }
}
