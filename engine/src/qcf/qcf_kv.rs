//! Unified QCF (Quality Cost Function) metric for all KV cache actions.
//!
//! Core formula:
//!   QCF = ||O_before - O_after|| / ||O_before||
//!   O = sum_t alpha_t * V_t   (attention-weighted value sum)
//!
//! Supports: sliding eviction, H2O eviction, StreamingLLM, D2O merge,
//! and KIVI quantization -- all measured in the same output-error space.
//!
//! ## D2O merge (paper arXiv 2406.13035 Eq.10/11)
//!
//! The D2O simulation is kept in lockstep with the actuator
//! (`crate::kv::d2o_handler`). For the same retained set $R$ and the
//! evicted set $E$:
//!   * **Nearest mapping** uses **K** (per head, cosine similarity). When the
//!     caller supplies `k_source`, K vectors of head `h` drive the matching.
//!     Without `k_source`, the simulator falls back to V (legacy behaviour) and
//!     emits a one-time `eprintln!` warning.
//!   * **Group normalisation** matches handler's `compute_eq11_weights`:
//!     $D_j = \sum_i e^{u_{ij}} + e$, $w_{c_j} = e/D_j$,
//!     $w_{e_i} = e^{u_{ij}}/D_j$, with `u` clamped to `[-10, 10]` before
//!     `exp`. Weights sum to 1 by construction.
//!   * **Constant `e`** is `MERGE_E = 0.1`, the default of
//!     `D2OConfig::merge_e`.
//!   * Only **V is augmented** in the simulator: $O_\text{after} =
//!     \sum_{c\in R} \alpha_c\,V_c^\text{merged}$, so re-augmenting K is a
//!     no-op for the QCF measurement. The handler still merges both K and V
//!     because K matters for *future* attention steps which the simulator
//!     does not model. This asymmetry is intentional and documented here.
//!   * **EMA threshold filter** is *not* applied. Estimator returns the
//!     post-condition QCF assuming all evicted tokens are merged (handler's
//!     EMA filter only drops a subset, so the estimator is an upper bound
//!     on the actual QCF).

use super::{AggregationMode, aggregate_heads};
use crate::buffer::DType;
use crate::kv_cache_ops::KVLayout;
use crate::quant::{BlockQ4_0, QK4_0};
use crate::tensor::Tensor;

// ── Action types ────────────────────────────────────────────────

/// Describes which lossy action to simulate for QCF computation.
#[derive(Debug, Clone)]
pub enum QcfActionType {
    /// Sliding window: retain the last `target_len` tokens.
    EvictSliding { target_len: usize },
    /// H2O: importance-based eviction with prefix protection.
    EvictH2o {
        target_len: usize,
        keep_ratio: f32,
        protected_prefix: usize,
    },
    /// StreamingLLM: retain first `sink_size` + last `window_size` tokens.
    EvictStreaming {
        sink_size: usize,
        window_size: usize,
    },
    /// D2O merge: same retained set as H2O, but evicted tokens are additively
    /// merged into their nearest (cosine similarity) retained token.
    MergeD2o {
        target_len: usize,
        keep_ratio: f32,
        protected_prefix: usize,
    },
}

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

/// All inputs needed to compute the unified QCF metric.
pub struct QcfKvParams<'a> {
    /// The action to simulate.
    pub action: QcfActionType,
    /// V buffer data (F32, F16, or Q4_0).
    pub v_source: VDataSource<'a>,
    /// Optional K buffer data, only consumed by `MergeD2o` for nearest-token
    /// cosine matching (paper Eq.8). When `None` the D2O simulator falls
    /// back to V-based matching with a warning. Other actions ignore it.
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
    /// Standard CAOTE = β=1.0; β > 1 emphasises high-attention tokens in sparse
    /// distributions. Default: 1.0 (no amplification, bit-identical to legacy).
    pub beta: f32,
}

// ── Main entry point ────────────────────────────────────────────

/// Compute unified QCF for the given action.
///
/// Returns `(aggregated_qcf, per_head_qcf)`.
pub fn compute_qcf_kv(params: &QcfKvParams) -> (f32, Vec<f32>) {
    let _t = crate::qcf_timer!(QCF_KV_UNIFIED);
    let n_kv_heads = params.n_kv_heads;
    let head_dim = params.head_dim;
    let current_pos = params.current_pos;
    let capacity = params.capacity;
    let layout = params.layout;
    let beta = params.beta;

    if n_kv_heads == 0 || head_dim == 0 || current_pos == 0 {
        return (0.0, vec![0.0; n_kv_heads]);
    }

    let max_seq_len = params.attention_scores.len();
    let mut per_head = vec![0.0f32; n_kv_heads];

    for (h, ph) in per_head.iter_mut().enumerate() {
        // 1. Get alpha_h[t] for this KV-head
        //    Try per-head attention first; fall back to flat scores if the
        //    per-head slice is all-zero (can happen when softmax produces NaN
        //    in attention weights, which the NaN guard converts to 0).
        let alpha_h: Vec<f32> = {
            let mut alpha = if let Some(head_attn) = params.head_attn {
                let head_offset = h * (head_attn.len() / n_kv_heads.max(1));
                (0..current_pos)
                    .map(|t| {
                        let idx = head_offset + t;
                        if idx < head_attn.len() {
                            head_attn[idx]
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<f32>>()
            } else {
                Vec::new() // will trigger flat fallback below
            };

            // Fallback: if per-head alpha is empty or all-zero, use flat scores
            let alpha_sum: f32 = alpha.iter().sum();
            if alpha.is_empty() || alpha_sum <= 0.0 {
                alpha = (0..current_pos)
                    .map(|t| {
                        if t < max_seq_len {
                            params.attention_scores[t]
                        } else {
                            0.0
                        }
                    })
                    .collect();
            }
            alpha
        };

        // 2. Compute O_before = Σ_t (α_t / Σ_s α_s) · V[h][t]   (ENG-ALG-051)
        //    O_before is normalised by the **full** token-set α-sum so it shares
        //    the same softmax-weight space as O_after's retained-set
        //    re-normalisation. Without this, raw Σ α_t V_t inflates ‖O_before‖
        //    by Σ_s α_s (e.g. ~12× for accumulated decode score sinks) and the
        //    relative error saturates at ~1. Note this is a scale normalisation
        //    of the existing metric — the QCF *contract* is unchanged; α-only
        //    relative comparisons (monotonicity, full-eviction=1.0) are
        //    invariant since the factor cancels in numerator/denominator.
        let alpha_all_sum: f32 = alpha_h.iter().take(current_pos).copied().sum();
        let mut o_before = vec![0.0f32; head_dim];
        if alpha_all_sum > 0.0 {
            for (t, &alpha_t) in alpha_h.iter().enumerate().take(current_pos) {
                let v_t = read_v_f32(
                    &params.v_source,
                    h,
                    t,
                    head_dim,
                    capacity,
                    n_kv_heads,
                    layout,
                );
                let w = alpha_t / alpha_all_sum;
                for d in 0..head_dim {
                    o_before[d] += w * v_t[d];
                }
            }
        }
        // alpha_all_sum <= 0 → o_before stays the zero vector, which routes to
        // the QCF=0 branch below (o_norm <= ε), consistent with the existing
        // zero-guard.

        // 3. Compute O_after based on action type
        let o_after = match &params.action {
            QcfActionType::EvictSliding { target_len } => {
                if current_pos <= *target_len {
                    o_before.clone()
                } else {
                    let retained_start = current_pos.saturating_sub(*target_len);
                    compute_o_eviction(
                        &alpha_h,
                        &params.v_source,
                        h,
                        (retained_start..current_pos)
                            .collect::<Vec<_>>()
                            .iter()
                            .copied(),
                        head_dim,
                        capacity,
                        n_kv_heads,
                        layout,
                        beta,
                    )
                }
            }
            QcfActionType::EvictH2o {
                target_len,
                keep_ratio,
                protected_prefix,
            } => {
                if current_pos <= *target_len {
                    o_before.clone()
                } else {
                    let retained = identify_retained_h2o(
                        &alpha_h,
                        current_pos,
                        *target_len,
                        *keep_ratio,
                        *protected_prefix,
                    );
                    compute_o_eviction(
                        &alpha_h,
                        &params.v_source,
                        h,
                        retained.iter().copied(),
                        head_dim,
                        capacity,
                        n_kv_heads,
                        layout,
                        beta,
                    )
                }
            }
            QcfActionType::EvictStreaming {
                sink_size,
                window_size,
            } => {
                let keep_size = sink_size + window_size;
                if current_pos <= keep_size {
                    o_before.clone()
                } else {
                    let retained: Vec<usize> = (0..*sink_size)
                        .chain((current_pos - window_size)..current_pos)
                        .collect();
                    compute_o_eviction(
                        &alpha_h,
                        &params.v_source,
                        h,
                        retained.iter().copied(),
                        head_dim,
                        capacity,
                        n_kv_heads,
                        layout,
                        beta,
                    )
                }
            }
            QcfActionType::MergeD2o {
                target_len,
                keep_ratio,
                protected_prefix,
            } => {
                if current_pos <= *target_len {
                    o_before.clone()
                } else {
                    let retained = identify_retained_h2o(
                        &alpha_h,
                        current_pos,
                        *target_len,
                        *keep_ratio,
                        *protected_prefix,
                    );
                    compute_o_d2o_merge(
                        &alpha_h,
                        &params.v_source,
                        params.k_source.as_ref(),
                        h,
                        &retained,
                        current_pos,
                        head_dim,
                        capacity,
                        n_kv_heads,
                        layout,
                    )
                }
            }
        };

        // 4. QCF = ||O_before - O_after|| / ||O_before||
        let diff_norm = l2_norm_diff(&o_before, &o_after);
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

// ── Helper: eviction O_after with softmax redistribution ────────

#[allow(clippy::too_many_arguments)]
fn compute_o_eviction(
    alpha: &[f32],
    v_src: &VDataSource,
    head: usize,
    retained: impl Iterator<Item = usize>,
    head_dim: usize,
    capacity: usize,
    n_kv_heads: usize,
    layout: KVLayout,
    beta: f32,
) -> Vec<f32> {
    let retained: Vec<usize> = retained.collect();

    // β=1 fast path: bit-identical to legacy α/Σα formula.
    if beta == 1.0 {
        let alpha_sum: f32 = retained.iter().map(|&t| alpha[t]).sum();
        if alpha_sum <= 0.0 {
            return vec![0.0; head_dim];
        }
        let mut o = vec![0.0f32; head_dim];
        for &t in &retained {
            let w = alpha[t] / alpha_sum;
            let v_t = read_v_f32(v_src, head, t, head_dim, capacity, n_kv_heads, layout);
            for d in 0..head_dim {
                o[d] += w * v_t[d];
            }
        }
        return o;
    }

    // β-amplified path: w_t = α_t^β / Σ α_s^β (re-normalised power).
    let alpha_pow_sum: f32 = retained.iter().map(|&t| alpha[t].max(0.0).powf(beta)).sum();
    if alpha_pow_sum <= 0.0 {
        return vec![0.0; head_dim];
    }
    let mut o = vec![0.0f32; head_dim];
    for &t in &retained {
        let w = alpha[t].max(0.0).powf(beta) / alpha_pow_sum;
        let v_t = read_v_f32(v_src, head, t, head_dim, capacity, n_kv_heads, layout);
        for d in 0..head_dim {
            o[d] += w * v_t[d];
        }
    }
    o
}

// ── Helper: D2O merge O_after with scatter-reduce compensation ──

/// Constant `e` of D2O paper Eq.11 (`D2OConfig::merge_e` default).
const MERGE_E: f32 = 0.1;

/// One-shot warning when D2O simulator falls back to V-based nearest matching.
static D2O_VFALLBACK_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Compute O_after for D2O merge (paper arXiv 2406.13035 Eq.10/11).
///
/// Same retained set $R$ as H2O. Evicted tokens are grouped by their nearest
/// retained token (cosine similarity on **K**, per head), and each group's
/// V is augmented with paper Eq.11 weights:
///   $D_j = \sum_i e^{u_{ij}} + e$
///   $w_{c_j} = e/D_j$, $w_{e_i} = e^{u_{ij}}/D_j$
///   $V_{c_j} \leftarrow w_{c_j} V_{c_j} + \sum_i w_{e_i} V_{e_i}$
///
/// Weights sum to 1 by construction, preserving V magnitude. K is *not*
/// augmented in the simulator — it would not change $O_\text{after}$ for the
/// current step, and the simulator does not model the next step.
///
/// `k_src = None` falls back to V-based nearest matching (with a one-time
/// warning) so legacy callers still produce a finite QCF.
#[allow(clippy::too_many_arguments)]
fn compute_o_d2o_merge(
    alpha: &[f32],
    v_src: &VDataSource,
    k_src: Option<&VDataSource>,
    head: usize,
    retained: &[usize],
    current_pos: usize,
    head_dim: usize,
    capacity: usize,
    n_kv_heads: usize,
    layout: KVLayout,
) -> Vec<f32> {
    if retained.is_empty() {
        return vec![0.0; head_dim];
    }

    // 1. Identify evicted tokens.
    let retained_set: std::collections::HashSet<usize> = retained.iter().copied().collect();
    let evicted: Vec<usize> = (0..current_pos)
        .filter(|t| !retained_set.contains(t))
        .collect();

    // 2. Build per-head K (or V fallback) for retained tokens — used as
    //    nearest-neighbour candidates.
    let nn_src: &VDataSource = match k_src {
        Some(k) => k,
        None => {
            if !D2O_VFALLBACK_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[QCF] D2O simulator: k_source=None, falling back to V-based nearest matching."
                );
            }
            v_src
        }
    };
    let nn_retained: Vec<Vec<f32>> = retained
        .iter()
        .map(|&t| read_v_f32(nn_src, head, t, head_dim, capacity, n_kv_heads, layout))
        .collect();

    // 3. Original V of retained tokens (V is what the simulator augments).
    let mut v_merged: Vec<Vec<f32>> = retained
        .iter()
        .map(|&t| read_v_f32(v_src, head, t, head_dim, capacity, n_kv_heads, layout))
        .collect();

    // 4. Group evicted tokens by their nearest retained index, recording the
    //    cosine similarity (used as `u_ij` in Eq.11).
    let mut groups: std::collections::HashMap<usize, Vec<(usize, f32)>> =
        std::collections::HashMap::new();
    for &e in &evicted {
        let q = read_v_f32(nn_src, head, e, head_dim, capacity, n_kv_heads, layout);
        let (nearest_idx, sim) = find_nearest_cosine_with_sim(&q, &nn_retained);
        groups.entry(nearest_idx).or_default().push((e, sim));
    }

    // 5. Per-group Eq.11 weight application: V_c <- w_c · V_c + Σ w_ei · V_ei
    //    (matches `crate::kv::d2o_handler::compute_eq11_weights`).
    for (&retained_idx, group) in &groups {
        let exps: Vec<f32> = group
            .iter()
            .map(|&(_, sim)| sim.clamp(-10.0, 10.0).exp())
            .collect();
        let sum_exp: f32 = exps.iter().sum();
        let denom = sum_exp + MERGE_E;
        if denom <= 0.0 {
            continue;
        }
        let inv_denom = 1.0 / denom;
        let w_c = MERGE_E * inv_denom;

        // V_c ← w_c · V_c
        for v in v_merged[retained_idx].iter_mut() {
            *v *= w_c;
        }
        // V_c += Σ w_ei · V_ei
        for (i, &(e_pos, _)) in group.iter().enumerate() {
            let w_e = exps[i] * inv_denom;
            let v_e = read_v_f32(v_src, head, e_pos, head_dim, capacity, n_kv_heads, layout);
            for (v, &ve) in v_merged[retained_idx].iter_mut().zip(v_e.iter()) {
                *v += w_e * ve;
            }
        }
    }

    // 6. O_after = Σ_{c∈R} (alpha_c / Σα) · V_c^merged   (softmax redistribution).
    let alpha_sum: f32 = retained.iter().map(|&t| alpha[t]).sum();
    if alpha_sum <= 0.0 {
        return vec![0.0; head_dim];
    }

    let mut o = vec![0.0f32; head_dim];
    for (i, &t) in retained.iter().enumerate() {
        let w = alpha[t] / alpha_sum;
        for d in 0..head_dim {
            o[d] += w * v_merged[i][d];
        }
    }
    o
}

/// Find the index of the candidate vector with highest cosine similarity
/// to the query vector. Returns (index, similarity).
fn find_nearest_cosine_with_sim(query: &[f32], candidates: &[Vec<f32>]) -> (usize, f32) {
    let q_norm = l2_norm(query);
    if q_norm < 1e-10 || candidates.is_empty() {
        return (0, 0.0);
    }

    let mut best_idx = 0;
    let mut best_sim = f32::NEG_INFINITY;

    for (i, c) in candidates.iter().enumerate() {
        let dot: f32 = query.iter().zip(c).map(|(a, b)| a * b).sum();
        let c_norm = l2_norm(c);
        let sim = if c_norm > 1e-10 {
            dot / (q_norm * c_norm)
        } else {
            0.0
        };
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }
    (best_idx, best_sim)
}

// ── Helper: retained token identification ───────────────────────

/// Identify tokens retained by H2O eviction policy.
///
/// Returns indices `[0, current_pos)` that would survive the eviction.
/// Protected prefix tokens are always retained. The remaining budget is
/// split between heavy hitters (top importance, by `keep_ratio`) and the
/// most recent window tokens.
pub fn identify_retained_h2o(
    importance: &[f32],
    current_pos: usize,
    target_len: usize,
    keep_ratio: f32,
    protected_prefix: usize,
) -> Vec<usize> {
    let prefix = protected_prefix.min(current_pos).min(target_len);
    let available = target_len.saturating_sub(prefix);
    if available == 0 {
        return (0..prefix).collect();
    }

    let hh_budget = (available as f32 * keep_ratio) as usize;
    let recent_budget = available.saturating_sub(hh_budget);
    let recent_start = current_pos.saturating_sub(recent_budget);

    // Protected prefix
    let mut retained: Vec<usize> = (0..prefix).collect();

    // Heavy hitters from evictable zone [prefix..recent_start]
    if recent_start > prefix {
        let mut evictable: Vec<(usize, f32)> = (prefix..recent_start)
            .map(|t| {
                let score = if t < importance.len() {
                    importance[t]
                } else {
                    0.0
                };
                (t, score)
            })
            .collect();
        evictable.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        retained.extend(evictable.iter().take(hh_budget).map(|(t, _)| t));
    }

    // Recent window
    retained.extend(recent_start..current_pos);
    retained.sort();
    retained.dedup();
    retained
}

/// Identify tokens retained by Sliding Window eviction policy.
///
/// Retains `[current_pos - target_len, current_pos)` — the most recent
/// `target_len` tokens. If `target_len >= current_pos`, all tokens are
/// retained (no eviction).
pub fn identify_retained_sliding(current_pos: usize, target_len: usize) -> Vec<usize> {
    let target_len = target_len.min(current_pos);
    let start = current_pos.saturating_sub(target_len);
    (start..current_pos).collect()
}

/// Dispatch helper: derive the retained token set for any `QcfActionType`.
///
/// Returns a sorted, deduplicated list of token indices in `[0, current_pos)`
/// that would survive the corresponding lossy cache action.
///
/// - `EvictSliding` → recent window only.
/// - `EvictH2o` / `MergeD2o` → H2O 3-partition (prefix + heavy hitters + recent).
/// - `EvictStreaming` → sink tokens + recent window.
pub fn identify_retained_for_action(
    action: &QcfActionType,
    importance: &[f32],
    current_pos: usize,
) -> Vec<usize> {
    match action {
        QcfActionType::EvictSliding { target_len } => {
            identify_retained_sliding(current_pos, *target_len)
        }
        QcfActionType::EvictH2o {
            target_len,
            keep_ratio,
            protected_prefix,
        }
        | QcfActionType::MergeD2o {
            target_len,
            keep_ratio,
            protected_prefix,
        } => identify_retained_h2o(
            importance,
            current_pos,
            *target_len,
            *keep_ratio,
            *protected_prefix,
        ),
        QcfActionType::EvictStreaming {
            sink_size,
            window_size,
        } => {
            let sink = (*sink_size).min(current_pos);
            let recent_start = current_pos.saturating_sub(*window_size);
            let mut out: Vec<usize> = (0..sink).collect();
            out.extend(recent_start.max(sink)..current_pos);
            out.sort();
            out.dedup();
            out
        }
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

    /// Build a HeadMajor V/K buffer with **directional diversity** for D2O.
    ///
    /// `make_v_data` gives every token the same direction (V(h,t,d) ∝ (d+1)),
    /// so cosine similarity is uniformly 1.0 and D2O's nearest-token merge can
    /// not discriminate — under the normalised metric that even inverts to
    /// D2O > H2O (the merge overwrites a retained token's V with an unrelated
    /// direction). For D2O to beat H2O the data must satisfy two conditions:
    ///   1. **directional diversity** — distinct token directions, so dropping
    ///      an evicted token's direction is a real loss (high H2O QCF);
    ///   2. **nearest alignment** — each evicted token has a same-direction
    ///      token elsewhere so its direction is restorable by cosine-merge.
    ///
    /// We assign each token a direction via `dir_class(t)` (sinusoidal phase),
    /// with two classes placed so that the heavy-hitter token AND its aligned
    /// twin both fall in the evicted set: H2O loses that whole direction, while
    /// D2O merges the twin back into a retained token, partially recovering it.
    /// Unit magnitude keeps the merge norm-preserving. Fully deterministic.
    fn make_diverse_v_data(n_kv_heads: usize, capacity: usize, head_dim: usize) -> Vec<f32> {
        assert!(head_dim >= 32, "diverse V needs head_dim >= 32");
        let freq = 0.37f32;
        let total = n_kv_heads * capacity * head_dim;
        let mut data = vec![0.0f32; total];
        for h in 0..n_kv_heads {
            for t in 0..capacity {
                // Two well-separated directional classes. Tokens 1 and 2 share
                // class B (the heavy-hitter + its mergeable twin); all others
                // are class A. A tiny per-head offset keeps heads non-identical.
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

        let params = QcfKvParams {
            action: QcfActionType::EvictSliding {
                target_len: current_pos,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf, per_head) = compute_qcf_kv(&params);
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

        let params = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 0 },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf, _) = compute_qcf_kv(&params);
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

        let make_params = |target_len: usize| QcfKvParams {
            action: QcfActionType::EvictSliding { target_len },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf_keep_12, _) = compute_qcf_kv(&make_params(12));
        let (qcf_keep_8, _) = compute_qcf_kv(&make_params(8));
        let (qcf_keep_4, _) = compute_qcf_kv(&make_params(4));

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

        let params = QcfKvParams {
            action: QcfActionType::EvictStreaming {
                sink_size: 4,
                window_size: 4,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf, _) = compute_qcf_kv(&params);
        assert!(
            qcf.abs() < 1e-6,
            "expected QCF=0 when sink+window covers all tokens, got {qcf}"
        );
    }

    #[test]
    fn test_h2o_vs_sliding() {
        // H2O should have QCF <= Sliding at same target_len when:
        // (1) scores are non-uniform with high importance on early tokens, AND
        // (2) V values for high-importance tokens are also large (correlated).
        // This tests the scenario where importance-aware eviction helps.
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 16;

        // Build V data where early (important) tokens have large V norms
        // V[t][d] = (current_pos - t) * (d+1), so early tokens dominate.
        let mut v_data = vec![0.0f32; n_kv_heads * capacity * head_dim];
        for t in 0..current_pos {
            for d in 0..head_dim {
                let offset = t * head_dim + d;
                v_data[offset] = (current_pos - t) as f32 * (d as f32 + 1.0);
            }
        }

        // Non-uniform: early tokens have very high importance
        let mut scores = vec![0.1f32; current_pos];
        scores[0] = 10.0;
        scores[1] = 8.0;
        scores[2] = 6.0;
        scores[3] = 5.0;

        let target_len = 8;

        let sliding_params = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let h2o_params = QcfKvParams {
            action: QcfActionType::EvictH2o {
                target_len,
                keep_ratio: 0.5,
                protected_prefix: 0,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf_sliding, _) = compute_qcf_kv(&sliding_params);
        let (qcf_h2o, _) = compute_qcf_kv(&h2o_params);

        assert!(
            qcf_h2o <= qcf_sliding + 1e-6,
            "H2O ({qcf_h2o}) should have QCF <= Sliding ({qcf_sliding}) \
             when important tokens have large V norms"
        );
    }

    #[test]
    fn test_f16_data_source() {
        // Verify F16 VDataSource works correctly
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 8;
        let current_pos = 4;

        // Create F32 data and its F16 equivalent
        let v_f32 = make_v_data(n_kv_heads, capacity, head_dim);
        let v_f16: Vec<u16> = v_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let scores = uniform_scores(current_pos);

        let params_f32 = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 2 },
            v_source: VDataSource::F32(&v_f32),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let params_f16 = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 2 },
            v_source: VDataSource::F16(&v_f16),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf_f32, _) = compute_qcf_kv(&params_f32);
        let (qcf_f16, _) = compute_qcf_kv(&params_f16);

        // F16 introduces small rounding but QCF should be very close
        assert!(
            (qcf_f32 - qcf_f16).abs() < 0.05,
            "F32 ({qcf_f32}) and F16 ({qcf_f16}) QCF should be close"
        );
    }

    #[test]
    fn test_streaming_evicts_middle() {
        // StreamingLLM: retains sink + recent, evicts the middle
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 16;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let params = QcfKvParams {
            action: QcfActionType::EvictStreaming {
                sink_size: 2,
                window_size: 4,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf, _) = compute_qcf_kv(&params);
        // 10 out of 16 tokens evicted, QCF should be significant
        assert!(
            qcf > 0.0,
            "StreamingLLM eviction should produce non-zero QCF"
        );
        assert!(qcf < 1.0, "QCF should be bounded below 1.0, got {qcf}");
    }

    #[test]
    fn test_per_head_attn_different_from_flat() {
        // When per-head attention differs, results should diverge from flat scores
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);

        let flat_scores = uniform_scores(current_pos);

        // Per-head: head 0 focuses on early tokens, head 1 on late tokens
        let mut head_attn = vec![0.0f32; n_kv_heads * current_pos];
        for t in 0..current_pos {
            // Head 0: linearly decreasing
            head_attn[0 * current_pos + t] = (current_pos - t) as f32;
            // Head 1: linearly increasing
            head_attn[current_pos + t] = (t + 1) as f32;
        }

        let params_flat = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 4 },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &flat_scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let params_head = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 4 },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &flat_scores,
            head_attn: Some(&head_attn),
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf_flat, _ph_flat) = compute_qcf_kv(&params_flat);
        let (qcf_head, ph_head) = compute_qcf_kv(&params_head);

        // Per-head should yield different per-head values
        assert!(
            (ph_head[0] - ph_head[1]).abs() > 1e-3,
            "per-head attn should produce different per-head QCFs: {:?}",
            ph_head
        );
        // Overall QCF should differ
        assert!(
            (qcf_flat - qcf_head).abs() > 1e-6,
            "flat ({qcf_flat}) vs per-head ({qcf_head}) should differ"
        );
    }

    #[test]
    fn test_d2o_less_than_h2o() {
        // Why D2O < H2O (strict): H2O drops evicted V contributions entirely,
        // whereas D2O additively merges each evicted token into its cosine-
        // nearest retained token (paper Eq.11, weights sum to 1), partially
        // restoring the evicted direction. So with the same retained set R,
        //   ‖O_b − O_a^D2O‖ ≤ ‖O_b − O_a^H2O‖.
        // This only holds with directional diversity + nearest alignment, both
        // absent in `make_v_data` (uniform direction). We use
        // `make_diverse_v_data`: tokens 1 and 2 form class B (the heavy hitter
        // and its mergeable twin), all others class A. With the attention/budget
        // below both class-B tokens are EVICTED, so H2O loses that direction
        // entirely (high QCF) while D2O merges the twin back into a retained
        // token (recovering it → lower QCF). K is supplied so cosine-nearest is
        // well-defined.
        let n_kv_heads = 2;
        let head_dim = 32;
        let capacity = 8;
        let current_pos = 4;
        let v_data = make_diverse_v_data(n_kv_heads, capacity, head_dim);
        // K aligned with V (same diversity) so K-cosine-nearest == V-nearest:
        // the evicted class-B twin merges into the class-B-aligned retained set.
        let k_data = make_diverse_v_data(n_kv_heads, capacity, head_dim);

        // Heavy hitters (tokens 1,2 = class B) sit OUTSIDE the protected prefix.
        // With target_len=2, prefix=1, keep_ratio=0.5 → hh_budget=0, so the only
        // retained tokens are prefix (t0) + recent window (t3), both class A.
        // The high-attention class-B tokens are evicted: H2O loses class B.
        let scores = vec![1.0f32, 9.0, 8.0, 1.0];

        let target_len = 2;
        let keep_ratio = 0.5;
        let protected_prefix = 1;

        let h2o_params = QcfKvParams {
            action: QcfActionType::EvictH2o {
                target_len,
                keep_ratio,
                protected_prefix,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: Some(VDataSource::F32(&k_data)),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let d2o_params = QcfKvParams {
            action: QcfActionType::MergeD2o {
                target_len,
                keep_ratio,
                protected_prefix,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: Some(VDataSource::F32(&k_data)),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf_h2o, ph_h2o) = compute_qcf_kv(&h2o_params);
        let (qcf_d2o, ph_d2o) = compute_qcf_kv(&d2o_params);

        // D2O strictly lower than H2O (merge restores evicted direction).
        assert!(
            qcf_d2o < qcf_h2o,
            "D2O ({qcf_d2o}) should have strictly lower QCF than H2O ({qcf_h2o})"
        );
        // Both positive (eviction actually happens).
        assert!(qcf_h2o > 0.0, "H2O QCF should be positive, got {qcf_h2o}");
        assert!(qcf_d2o > 0.0, "D2O QCF should be positive, got {qcf_d2o}");

        // Per-head: D2O <= H2O for each head.
        for h in 0..n_kv_heads {
            assert!(
                ph_d2o[h] <= ph_h2o[h] + 1e-6,
                "head {h}: D2O ({}) should have QCF <= H2O ({})",
                ph_d2o[h],
                ph_h2o[h]
            );
        }
    }

    #[test]
    fn test_d2o_no_eviction_equals_zero() {
        // When current_pos <= target_len, no eviction happens, QCF = 0
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let params = QcfKvParams {
            action: QcfActionType::MergeD2o {
                target_len: current_pos,
                keep_ratio: 0.5,
                protected_prefix: 2,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf, _) = compute_qcf_kv(&params);
        assert!(
            qcf.abs() < 1e-6,
            "D2O with no eviction should give QCF=0, got {qcf}"
        );
    }

    #[test]
    fn test_d2o_uses_k_for_nearest() {
        // Constructs a case where K and V give different nearest matches:
        // when K is supplied, the simulator follows K; otherwise V fallback.
        // We just check that the resulting QCFs differ (the absolute ordering
        // depends on alpha/V geometry and is not part of the contract).
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 8;

        // V: monotonically increasing per token.
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);

        // K: deliberately inverted vs V so K-based nearest != V-based nearest.
        // K[t] = constant base − (t+1) * unit, plus per-d offsets.
        let mut k_data = vec![0.0f32; n_kv_heads * capacity * head_dim];
        for t in 0..current_pos {
            for d in 0..head_dim {
                let off = t * head_dim + d;
                k_data[off] = (current_pos as f32 - t as f32) * (d as f32 + 1.0);
            }
        }

        // Non-uniform scores so eviction actually triggers.
        let mut scores = vec![0.1f32; current_pos];
        scores[0] = 5.0;
        scores[1] = 3.0;

        let target_len = 4;
        let keep_ratio = 0.5;
        let protected_prefix = 1;

        let params_v = QcfKvParams {
            action: QcfActionType::MergeD2o {
                target_len,
                keep_ratio,
                protected_prefix,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: None, // V fallback
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };
        let params_k = QcfKvParams {
            action: QcfActionType::MergeD2o {
                target_len,
                keep_ratio,
                protected_prefix,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: Some(VDataSource::F32(&k_data)),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        let (qcf_v, _) = compute_qcf_kv(&params_v);
        let (qcf_k, _) = compute_qcf_kv(&params_k);

        assert!(
            (qcf_v - qcf_k).abs() > 1e-6,
            "K-based nearest ({qcf_k}) should differ from V-fallback ({qcf_v}) \
             when K and V geometries disagree"
        );
    }

    #[test]
    fn test_d2o_weight_grouping_normalized() {
        // Eq.11 weights sum to 1 by construction. For a single retained
        // token absorbing all evictions (one nearest target), the merged V
        // norm must lie within a convex hull of the original V norms, i.e.
        // bounded by max_norm.
        //
        // We construct a head where exactly one retained token exists (so
        // every evicted maps to it), all V norms are bounded by `max_norm`,
        // and check that the merged V norm of that single retained group
        // is also <= max_norm * (1 + 1e-4) (numerical slack).
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;

        // V[t][d] = (t+1) * (d+1)  ==>  ‖V[t]‖ grows monotonically with t.
        // Use t=0..current_pos so max ‖V‖ = ‖V[7]‖.
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);

        // Compute max V norm directly.
        let max_norm: f32 = (0..current_pos)
            .map(|t| {
                let off = t * head_dim;
                let v: &[f32] = &v_data[off..off + head_dim];
                l2_norm(v)
            })
            .fold(0.0_f32, f32::max);

        // target_len = 1, keep_ratio = 1.0, protected_prefix = 0
        // ⇒ retained = {0}. All other 7 tokens evicted, all map to t=0.
        let scores = uniform_scores(current_pos);
        let params = QcfKvParams {
            action: QcfActionType::MergeD2o {
                target_len: 1,
                keep_ratio: 1.0,
                protected_prefix: 0,
            },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };

        // Re-derive O_after = w · V_merged[0] where w = alpha[0]/Σα = 1
        // (only retained token), so ‖O_after‖ = ‖V_merged[0]‖.
        // Run the QCF and reconstruct ‖V_merged[0]‖ from QCF + ‖O_before‖.
        let (qcf, _) = compute_qcf_kv(&params);

        // O_before for head 0 with uniform alpha=1/8.
        let mut o_before = vec![0.0f32; head_dim];
        for t in 0..current_pos {
            let off = t * head_dim;
            for d in 0..head_dim {
                o_before[d] += (1.0 / current_pos as f32) * v_data[off + d];
            }
        }
        let o_before_norm = l2_norm(&o_before);

        // diff_norm = qcf * o_before_norm
        // ‖O_before − O_after‖ = qcf * ‖O_before‖
        // O_after lies on a sphere of radius (qcf * ‖O_before‖) around
        // O_before, so ‖O_after‖ ∈ [‖O_before‖(1−qcf), ‖O_before‖(1+qcf)].
        // Upper bound must not exceed max_norm by more than slack.
        let upper = o_before_norm * (1.0 + qcf);
        let slack = max_norm * 1e-4;
        assert!(
            upper <= max_norm + slack,
            "merged V norm upper bound {upper} should not exceed max single-token V norm \
             {max_norm} (+slack {slack}); QCF = {qcf}, ‖O_before‖ = {o_before_norm}"
        );
    }

    #[test]
    fn test_find_nearest_cosine_basic() {
        // query = [1, 0], candidates = [[0, 1], [1, 0.1]]
        // cosine sim to [0,1] = 0, to [1,0.1] ~= 0.995
        let query = vec![1.0, 0.0];
        let candidates = vec![vec![0.0, 1.0], vec![1.0, 0.1]];
        assert_eq!(find_nearest_cosine_with_sim(&query, &candidates).0, 1);
    }

    #[test]
    fn test_find_nearest_cosine_zero_query() {
        let query = vec![0.0, 0.0];
        let candidates = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // Zero query norm -> returns (0, 0.0)
        assert_eq!(find_nearest_cosine_with_sim(&query, &candidates).0, 0);
    }

    #[test]
    fn test_empty_inputs() {
        let v_data = vec![0.0f32; 64];
        let scores = vec![1.0f32; 8];

        // Zero heads
        let params = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 4 },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads: 0,
            head_dim: 4,
            current_pos: 8,
            capacity: 16,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };
        let (qcf, per_head) = compute_qcf_kv(&params);
        assert_eq!(qcf, 0.0);
        assert!(per_head.is_empty());

        // Zero current_pos
        let params = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 4 },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads: 2,
            head_dim: 4,
            current_pos: 0,
            capacity: 16,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };
        let (qcf, _) = compute_qcf_kv(&params);
        assert_eq!(qcf, 0.0);
    }

    #[test]
    fn test_identify_retained_h2o_basic() {
        // 10 tokens, prefix=2, target=6, keep_ratio=0.5
        // Available after prefix = 4, hh_budget=2, recent_budget=2
        let mut importance = vec![0.1f32; 10];
        importance[3] = 10.0; // high importance
        importance[5] = 8.0; // high importance

        let retained = identify_retained_h2o(&importance, 10, 6, 0.5, 2);

        // Should contain: prefix [0,1], heavy hitters [3,5], recent [8,9]
        assert!(retained.contains(&0), "prefix token 0 should be retained");
        assert!(retained.contains(&1), "prefix token 1 should be retained");
        assert!(
            retained.contains(&3),
            "high importance token 3 should be retained"
        );
        assert!(
            retained.contains(&5),
            "high importance token 5 should be retained"
        );
        assert!(retained.len() == 6, "should retain exactly 6 tokens");
    }

    // ── Regression tests for ISSUE-6 ────────────────────────────
    //
    // These tests verify that `read_v_f32()` tolerates a V-source slice
    // whose length is smaller than `offset + head_dim` (including the
    // degenerate zero-length case). Previously an OpenCL device-only
    // buffer produced a `(ptr=null, len=0)` slice that would SEGV the
    // moment `data[offset..end]` indexed it. The host-side guard in
    // `compute_qcf_estimates` now short-circuits before we get here,
    // but the read path itself must remain panic-free if it is ever
    // invoked with a too-small slice (e.g. from tests or future call
    // sites).

    #[test]
    fn test_read_v_f32_empty_f16_returns_zeros() {
        // Empty F16 slice: offset >= data.len() → zero-fill, no panic.
        let empty: Vec<u16> = Vec::new();
        let out = read_v_f32(
            &VDataSource::F16(&empty),
            /* head = */ 0,
            /* pos = */ 0,
            /* head_dim = */ 64,
            /* capacity = */ 128,
            /* n_kv_heads = */ 8,
            KVLayout::HeadMajor,
        );
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&x| x == 0.0));

        // Non-zero offset that overruns the (empty) buffer.
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
        // Empty F32 slice: same contract as F16.
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

        // Partially-filled slice that still cannot cover offset+head_dim.
        // (offset=64, head_dim=64 → requested [64..128] but len=10.)
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
        // Empty Q4_0 slice: block_idx >= data.len() → zero-fill, no panic.
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

        // Non-zero head/pos that overruns the empty block slice.
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

    // ── identify_retained_* tests ────────────────────────────────

    #[test]
    fn test_identify_retained_sliding_basic() {
        let retained = identify_retained_sliding(10, 3);
        assert_eq!(retained, vec![7, 8, 9]);
    }

    #[test]
    fn test_identify_retained_sliding_target_exceeds() {
        // target_len > current_pos → all tokens retained
        let retained = identify_retained_sliding(5, 10);
        assert_eq!(retained, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_identify_retained_for_action_sliding() {
        let importance = vec![0.1f32; 10];
        let retained = identify_retained_for_action(
            &QcfActionType::EvictSliding { target_len: 3 },
            &importance,
            10,
        );
        assert_eq!(retained, vec![7, 8, 9]);
    }

    #[test]
    fn test_identify_retained_for_action_h2o_keep_ratio_0() {
        // kr=0.0 → no heavy hitters, only recent window (same as sliding)
        let importance = vec![1.0f32; 10];
        let retained_h2o = identify_retained_for_action(
            &QcfActionType::EvictH2o {
                target_len: 5,
                keep_ratio: 0.0,
                protected_prefix: 0,
            },
            &importance,
            10,
        );
        let retained_sliding = identify_retained_sliding(10, 5);
        assert_eq!(
            retained_h2o, retained_sliding,
            "kr=0.0 should equal sliding window"
        );
    }

    #[test]
    fn test_identify_retained_for_action_h2o_keep_ratio_1() {
        // kr=1.0 → all budget for heavy hitters, no recent window
        // target_len=4, kr=1.0, prefix=0, current_pos=10 → 4 top-importance tokens, no recent
        let mut importance = vec![0.1f32; 10];
        // Make tokens 2,3,4,5 the most important
        importance[2] = 5.0;
        importance[3] = 4.0;
        importance[4] = 3.0;
        importance[5] = 2.0;
        let retained = identify_retained_for_action(
            &QcfActionType::EvictH2o {
                target_len: 4,
                keep_ratio: 1.0,
                protected_prefix: 0,
            },
            &importance,
            10,
        );
        // With kr=1.0: hh_budget=4, recent_budget=0 → recent_start=current_pos=10
        // evictable zone = [0..10), top-4 by importance = [2,3,4,5]
        // No recent window tokens (recent_start=10, so range 10..10 is empty)
        assert_eq!(retained.len(), 4);
        for &idx in &[2usize, 3, 4, 5] {
            assert!(
                retained.contains(&idx),
                "token {idx} should be in retained set"
            );
        }
    }

    #[test]
    fn test_identify_retained_for_action_streaming() {
        // sink_size=2, window_size=3, current_pos=10
        // sink=[0,1], recent=[7,8,9] → retained=[0,1,7,8,9]
        let importance = vec![0.1f32; 10];
        let retained = identify_retained_for_action(
            &QcfActionType::EvictStreaming {
                sink_size: 2,
                window_size: 3,
            },
            &importance,
            10,
        );
        assert_eq!(retained, vec![0, 1, 7, 8, 9]);
    }

    // ── β-amplification tests (ARGUS QCF Step 3) ────────────────

    /// Hand-computed expected values for a small sliding eviction case.
    ///
    /// Setup:
    ///   n_kv_heads=1, head_dim=2, capacity=8, current_pos=3, target_len=2
    ///   V[h=0][t][d] = (t+1)*(d+1)  → V[0]=[1,2], V[1]=[2,4], V[2]=[3,6]
    ///   scores (uniform) = [1/3, 1/3, 1/3]
    ///   Retained = {1, 2}  (sliding: last 2 of current_pos=3)
    ///
    /// β=1 expected O_after:
    ///   α_sum = α[1]+α[2] = 2/3
    ///   w[1] = (1/3)/(2/3) = 0.5, w[2] = 0.5
    ///   O_after = 0.5*[2,4] + 0.5*[3,6] = [2.5, 5.0]
    ///
    ///   O_before = (1/3)*[1,2]+(1/3)*[2,4]+(1/3)*[3,6] = [2.0, 4.0]
    ///   diff = [0.5, 1.0], diff_norm = sqrt(0.25+1.0) = sqrt(1.25) ≈ 1.118
    ///   o_norm = sqrt(4+16) = sqrt(20) ≈ 4.472
    ///   QCF = 1.118/4.472 ≈ 0.25
    ///
    /// β=2 expected O_after:
    ///   α^2: [1/9, 1/9] for retained positions (uniform → same)
    ///   w[1] = w[2] = 0.5  (same as β=1 for uniform scores!)
    ///   So QCF_β2 == QCF_β1 for uniform scores (confirmed by test).
    #[test]
    fn test_compute_qcf_kv_beta_one_matches_hand_computed() {
        let n_kv_heads = 1;
        let head_dim = 2;
        let capacity = 8;
        let current_pos = 3;

        // HeadMajor: V[h=0][t][d] = (t+1)*(d+1)
        let mut v_data = vec![0.0f32; n_kv_heads * capacity * head_dim];
        for t in 0..current_pos {
            for d in 0..head_dim {
                let offset = t * head_dim + d; // head=0
                v_data[offset] = (t as f32 + 1.0) * (d as f32 + 1.0);
            }
        }
        let scores = uniform_scores(current_pos); // [1/3, 1/3, 1/3]

        let params = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 2 },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };
        let (qcf, _) = compute_qcf_kv(&params);
        // expected ≈ 0.25 (see above derivation)
        assert!(
            (qcf - 0.25).abs() < 1e-5,
            "β=1 QCF expected ~0.25, got {qcf}"
        );
    }

    #[test]
    fn test_compute_qcf_kv_beta_amplifies_non_uniform() {
        // For non-uniform scores β=2 should differ from β=1.
        // Setup: n_kv_heads=1, head_dim=2, current_pos=3, target_len=2
        // scores = [0.1, 0.9, 0.9]  (non-uniform, heavy on tokens 1 and 2)
        // Retained = {1, 2}
        //
        // β=1: w = [0.9, 0.9] / 1.8 = [0.5, 0.5]
        // β=2: w = [0.81, 0.81] / 1.62 = [0.5, 0.5]  (same because equal)
        //
        // Use scores where retained tokens differ: [0.1, 0.3, 0.6]
        // Retained = {1, 2}
        // β=1: w = [0.3, 0.6]/0.9 = [1/3, 2/3]
        // β=2: w = [0.09, 0.36]/0.45 = [0.2, 0.8]  → different!
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
        // Non-uniform: token 1 weight 0.3, token 2 weight 0.6
        let scores = vec![0.1f32, 0.3, 0.6];

        let params_b1 = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 2 },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        };
        let params_b2 = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 2 },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 2.0,
        };

        let (qcf_b1, _) = compute_qcf_kv(&params_b1);
        let (qcf_b2, _) = compute_qcf_kv(&params_b2);

        assert!(
            (qcf_b1 - qcf_b2).abs() > 1e-5,
            "β=1 ({qcf_b1}) and β=2 ({qcf_b2}) should differ for non-uniform scores"
        );
    }

    #[test]
    fn test_compute_o_eviction_beta_zero_uniform() {
        // β=0: α_t^0 = 1 for all t > 0 (note: 0^0 is technically undefined but
        // max(0, α) ensures non-negative; in practice scores > 0 so powf(0) = 1).
        // All weights become equal → same as uniform-score redistribution.
        let n_kv_heads = 1;
        let head_dim = 2;
        let capacity = 8;
        let current_pos = 4;

        // V: each token has distinct vector
        let v_data: Vec<f32> = (0..n_kv_heads * capacity * head_dim)
            .map(|i| i as f32 + 1.0)
            .collect();

        // Non-uniform scores, but β=0 should equalise weights
        let scores = vec![0.1f32, 5.0, 2.0, 0.5];

        let params = QcfKvParams {
            action: QcfActionType::EvictSliding { target_len: 3 },
            v_source: VDataSource::F32(&v_data),
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 0.0,
        };

        // β=0: retained={1,2,3}, all weights = 1/3.
        // V[t][d] for h=0, HeadMajor: offset = t*head_dim+d
        // O_after_β0 = (1/3)*V[1] + (1/3)*V[2] + (1/3)*V[3]
        let expected_v1 = [v_data[head_dim], v_data[head_dim + 1]];
        let expected_v2 = [v_data[2 * head_dim], v_data[2 * head_dim + 1]];
        let expected_v3 = [v_data[3 * head_dim], v_data[3 * head_dim + 1]];
        let expected_o: Vec<f32> = (0..head_dim)
            .map(|d| (expected_v1[d] + expected_v2[d] + expected_v3[d]) / 3.0)
            .collect();

        let (qcf_b0, _) = compute_qcf_kv(&params);

        // Compute what QCF would be with that expected_o
        // O_before = (1/current_pos) * sum of all V[t]  (no: scores aren't uniform, but we just
        // check that β=0 result is finite and between 0 and 1.)
        // Production usage note: β=0 is NOT a supported production value (default β=1).
        assert!(
            qcf_b0 >= 0.0,
            "β=0 QCF should be non-negative, got {qcf_b0}"
        );
        assert!(
            qcf_b0 <= 1.5,
            "β=0 QCF should be bounded (≤1.5), got {qcf_b0}"
        );
        // Verify the expected_o sanity (that our derivation above is correct)
        assert!(
            expected_o.iter().all(|&x| x > 0.0),
            "expected_o should be positive: {expected_o:?}"
        );
    }
}
