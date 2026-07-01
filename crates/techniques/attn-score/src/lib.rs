//! Attention-score producer technique crate — the forward-time score-accumulation policy.
//!
//! Extracted from the engine core (the `sliding-window`/`h2o`/`layer-importance` precedent): depends
//! only on `argus-extension-api` + `linkme`, implements [`ScoreProducer`], and registers under the
//! name `"attn_score"` via `#[distributed_slice(SCORE_PRODUCERS)]`. The engine force-links it with a
//! one-line `use attn_score as _;` and resolves it via `find_score_producer(...)`; its
//! `AttentionScoreAccumulator` is a thin delegating shell over this producer.
//!
//! The arithmetic is ported verbatim from the engine's former `AttentionScoreAccumulator` inline
//! policy (per-layer MAX aggregation, GQA group averaging, the CAOTE last-layer overwrite with its
//! NaN guard, A2SF begin-step decay, cross-step SUM, and time-normalization), so the values it
//! produces are bit-identical to the pre-extraction engine.

use argus_extension_api::{
    SCORE_PRODUCERS, ScoreProducer, ScoreProducerParams, ScoreProducerReg, TensorKind,
};
use linkme::distributed_slice;

// GPU half of the observer/score axis (EPIC 2 Stage E): the `ScoreReduceBackend` that owns
// `score_reduce.cl`. Compiled into the same crate (force-linked alongside the CPU producer) and
// gated on `opencl`, since it pulls in `ocl`.
#[cfg(feature = "opencl")]
mod gpu_reduce;

// CUDA twin of `gpu_reduce`: the `CudaScoreReduceBackend` that owns `score_reduce.cu`. Force-linked
// alongside the CPU producer and gated on `cuda`, since it pulls in `cudarc`.
#[cfg(feature = "cuda")]
mod gpu_reduce_cuda;

/// The built-in attention-score producer. Accumulates per-token attention importance scores across
/// layers (and, in GQA mode, per-KV-head importance + the CAOTE last-layer attention).
///
/// Verbatim port of the engine's former `AttentionScoreAccumulator` fields + policy. During decode,
/// each layer's post-softmax attention weights are aggregated into a per-token importance score; H2O
/// uses these scores to decide which tokens to keep vs evict.
struct AttnScoreProducer {
    /// Per-token cumulative importance scores, indexed by cache position. Updated once per step.
    importance: Vec<f32>,
    /// Per-token step-local importance buffer (per-layer MAX within a step, flushed in `end_step`).
    step_importance: Vec<f32>,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Which layers to track. Empty means track all layers.
    tracked_layers: Vec<usize>,
    /// Exponential decay factor (0.0 = no decay, 1.0 = full decay).
    decay: f32,
    /// Whether accumulation is active.
    active: bool,
    /// Number of KV heads for GQA grouping. 0 = GQA mode disabled.
    n_kv_heads: usize,
    /// Per-KV-head cumulative importance: `[n_kv_heads * max_seq_len]`, row-major.
    head_importance: Vec<f32>,
    /// Per-KV-head step-local buffer (same layout).
    head_step_importance: Vec<f32>,
    /// Last tracked layer's per-KV-head attention from the most recent decode step (CAOTE).
    /// Layout: `[n_kv_heads * max_seq_len]`, row-major. Overwritten each layer (not MAX).
    last_layer_head_attn: Vec<f32>,
    /// Per-token count of steps in which this position was active.
    step_count: Vec<u32>,
    /// Time-normalized importance: `importance[t] / step_count[t]`.
    normalized: Vec<f32>,
    /// If true, `importance_scores()` returns time-normalized values.
    time_normalize: bool,
    /// Total decoder layers — sizes the per-(layer, KV-head, token) dump buffer.
    total_layers: usize,
    /// Layer index for the next `accumulate_layer*` call. Only meaningful when `dump_layer_head`
    /// (the engine calls `set_current_layer` before each tracked layer's accumulate).
    current_layer: usize,
    /// Whether the non-collapsed per-(layer, KV-head, token) importance dump is active (IMP-1).
    /// Off by default → no extra buffer, production path byte-identical (`INV-147`).
    dump_layer_head: bool,
    /// Non-collapsed per-(layer, KV-head, token) **step-local** scratch:
    /// `[total_layers * n_kv_heads * max_seq_len]`, row-major
    /// `(layer * n_kv_heads + kv_head) * max_seq_len + pos`. Cleared each `begin_step`,
    /// written during `accumulate_layer_gqa`, flushed into `layer_head_cum` by
    /// `end_step`. Empty unless `dump_layer_head`.
    layer_head: Vec<f32>,
    /// Cumulative per-(layer, KV-head, token) importance summed across steps — the
    /// layer-resolved twin of `head_importance`, and what `layer_head_importance()`
    /// reports. Over a single decode step (the post-prefill probe) it equals the
    /// step value, so the probe-path IMP-1 dump is byte-identical; over a
    /// token-by-token prefill it is the query-agnostic context importance the
    /// policy actually ranked on. Empty unless `dump_layer_head`.
    layer_head_cum: Vec<f32>,
    /// Whether the per-`(layer, token)` FLAT importance is kept (faithful-H2O LayerWise,
    /// divergence `(b)`). Off by default → no buffer, the collapsed `importance` path is
    /// byte-identical (the `dump_layer_head` INV-147 precedent for the flat axis).
    per_layer_flat: bool,
    /// Per-`(layer, token)` FLAT **step-local** scratch: `[total_layers * max_seq_len]`,
    /// row-major `layer * max_seq_len + pos`. Cleared each `begin_step`, written (NOT
    /// MAX-collapsed across layers) during `accumulate_layer*`, flushed into
    /// `layer_flat_cum` by `end_step`. Empty unless `per_layer_flat`.
    layer_flat: Vec<f32>,
    /// Cumulative per-`(layer, token)` FLAT importance summed across steps — the
    /// layer-resolved twin of the collapsed `importance`, with NO cross-layer MAX, so
    /// each layer ranks heavy hitters on its OWN accumulated attention (faithful
    /// `H2OKVCache_LayerWise`). What `layer_flat_importance()` reports. Empty unless
    /// `per_layer_flat`.
    layer_flat_cum: Vec<f32>,
}

impl AttnScoreProducer {
    /// Build from the engine-supplied geometry. Merges the engine's former `new` / `new_gqa`:
    /// `n_kv_heads == 0` yields empty per-head buffers (flat mode); `n_kv_heads > 0` allocates the
    /// `[n_kv_heads * max_seq_len]` per-head + CAOTE buffers (`vec![0.0; 0]` is empty, so the single
    /// expression covers both — bit-identical to the two former constructors).
    fn new(params: ScoreProducerParams) -> Self {
        let ScoreProducerParams {
            max_seq_len,
            n_heads: _,
            n_kv_heads,
            total_layers,
            last_n_layers,
            decay,
        } = params;

        let tracked_layers = if last_n_layers == 0 || last_n_layers >= total_layers {
            Vec::new()
        } else {
            ((total_layers - last_n_layers)..total_layers).collect()
        };

        let head_buf_size = n_kv_heads * max_seq_len;
        Self {
            importance: vec![0.0; max_seq_len],
            step_importance: vec![0.0; max_seq_len],
            max_seq_len,
            tracked_layers,
            decay: decay.clamp(0.0, 1.0),
            active: false,
            n_kv_heads,
            head_importance: vec![0.0; head_buf_size],
            head_step_importance: vec![0.0; head_buf_size],
            last_layer_head_attn: vec![0.0; head_buf_size],
            step_count: vec![0; max_seq_len],
            normalized: vec![0.0; max_seq_len],
            time_normalize: false,
            total_layers,
            current_layer: 0,
            dump_layer_head: false,
            layer_head: Vec::new(),
            layer_head_cum: Vec::new(),
            per_layer_flat: false,
            layer_flat: Vec::new(),
            layer_flat_cum: Vec::new(),
        }
    }
}

impl ScoreProducer for AttnScoreProducer {
    fn name(&self) -> &str {
        "attn_score"
    }

    fn produces(&self) -> &'static [TensorKind] {
        &[TensorKind::Scores]
    }

    fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn should_track_layer(&self, layer: usize) -> bool {
        self.active && (self.tracked_layers.is_empty() || self.tracked_layers.contains(&layer))
    }

    fn set_time_normalize(&mut self, enable: bool) {
        self.time_normalize = enable;
    }

    fn begin_step(&mut self) {
        if !self.active {
            return;
        }
        if self.decay > 0.0 {
            let factor = 1.0 - self.decay;
            for v in self.importance.iter_mut() {
                *v *= factor;
            }
            for v in self.head_importance.iter_mut() {
                *v *= factor;
            }
            // Decay the layer-resolved cumulative in lockstep with head_importance so
            // the IMP-1 dump stays consistent with the score the policy ranks on.
            // No-op when the dump is off (buffer empty).
            for v in self.layer_head_cum.iter_mut() {
                *v *= factor;
            }
            // Decay the per-(layer, token) FLAT cumulative in lockstep with the collapsed
            // `importance` (faithful-H2O `(b)`). No-op when per-layer is off (buffer empty).
            for v in self.layer_flat_cum.iter_mut() {
                *v *= factor;
            }
        }
        self.step_importance.fill(0.0);
        self.head_step_importance.fill(0.0);
        self.last_layer_head_attn.fill(0.0);
        // Per-(layer, KV-head) dump buffer is a most-recent-step snapshot (like
        // last_layer_head_attn). No-op when the dump is off (buffer is empty).
        self.layer_head.fill(0.0);
        // Per-(layer, token) FLAT step scratch — cleared so each layer writes its own
        // slot this step. No-op when per-layer is off (buffer empty).
        self.layer_flat.fill(0.0);
    }

    fn accumulate_layer(
        &mut self,
        scores: &[f32],
        stride: usize,
        cache_seq_len: usize,
        n_heads_q: usize,
        score_offset: usize,
    ) {
        let len = cache_seq_len.min(self.max_seq_len);

        for t in 0..len {
            let pos = score_offset + t;
            if pos >= self.max_seq_len {
                break;
            }
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            self.step_importance[pos] = self.step_importance[pos].max(layer_score);

            // Per-(layer, token) FLAT step scratch (faithful-H2O `(b)`): record THIS layer's
            // head-summed score in its own slot — NO cross-layer MAX, so each layer keeps its
            // own heavy hitters. Guarded so the production path (per-layer off → buffer empty)
            // does no work (INV-147 for the flat axis).
            if self.per_layer_flat {
                let l_idx = self.current_layer * self.max_seq_len + pos;
                if l_idx < self.layer_flat.len() {
                    self.layer_flat[l_idx] = layer_score;
                }
            }
        }
    }

    fn accumulate_layer_gqa(
        &mut self,
        scores: &[f32],
        stride: usize,
        cache_seq_len: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
        score_offset: usize,
    ) {
        let len = cache_seq_len.min(self.max_seq_len);

        let n_rep = n_heads_q / n_kv_heads;
        let inv_rep = 1.0 / n_rep as f32;

        for t in 0..len {
            let pos = score_offset + t;
            if pos >= self.max_seq_len {
                break;
            }
            // Flat accumulation (backward compatible with H2O)
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            self.step_importance[pos] = self.step_importance[pos].max(layer_score);

            // Per-(layer, token) FLAT step scratch (faithful-H2O `(b)`): THIS layer's
            // head-summed score in its own slot, no cross-layer MAX. Guarded (INV-147,
            // flat axis) — production path (per-layer off) does no work.
            if self.per_layer_flat {
                let l_idx = self.current_layer * self.max_seq_len + pos;
                if l_idx < self.layer_flat.len() {
                    self.layer_flat[l_idx] = layer_score;
                }
            }

            // Per-KV-head: average Q-heads within each GQA group
            for kv_h in 0..n_kv_heads {
                let mut group_score = 0.0f32;
                for r in 0..n_rep {
                    group_score += scores[(kv_h * n_rep + r) * stride + t];
                }
                group_score *= inv_rep;
                let idx = kv_h * self.max_seq_len + pos;
                self.head_step_importance[idx] = self.head_step_importance[idx].max(group_score);
                // CAOTE: overwrite (not MAX) to keep the last tracked layer's raw attention.
                // NaN guard: softmax can produce NaN when all logits are -inf (e.g.
                // masked tokens). f32::max() silently swallows NaN for the other
                // accumulators, but direct assignment propagates it here.
                self.last_layer_head_attn[idx] = if group_score.is_nan() {
                    0.0
                } else {
                    group_score
                };

                // Non-collapsed per-(layer, KV-head, token) dump (IMP-1, opt-in).
                // Preserves the layer axis the collapsed buffers discard. The
                // `dump_layer_head` guard keeps the production path branch-only
                // (buffer never allocated when off → INV-147).
                if self.dump_layer_head {
                    let l_idx =
                        (self.current_layer * self.n_kv_heads + kv_h) * self.max_seq_len + pos;
                    if l_idx < self.layer_head.len() {
                        self.layer_head[l_idx] = if group_score.is_nan() {
                            0.0
                        } else {
                            group_score
                        };
                    }
                }
            }
        }
    }

    fn end_step(&mut self) {
        if !self.active {
            return;
        }

        for t in 0..self.max_seq_len {
            let step_val = self.step_importance[t];
            self.importance[t] += step_val;
            // Track step count: increment for positions that were in cache this step
            if step_val > 0.0 {
                self.step_count[t] += 1;
            }
            // Compute time-normalized score
            if self.time_normalize {
                let count = self.step_count[t].max(1) as f32;
                self.normalized[t] = self.importance[t] / count;
            }
        }
        for (cum, &step) in self
            .head_importance
            .iter_mut()
            .zip(self.head_step_importance.iter())
        {
            *cum += step;
        }
        // Sum this step's per-(layer, KV-head) values into the cumulative dump buffer
        // (same flush as head_importance, with the layer axis kept). Guarded so the
        // production path (dump off → both buffers empty) does no work — INV-147.
        if self.dump_layer_head {
            for (cum, &step) in self.layer_head_cum.iter_mut().zip(self.layer_head.iter()) {
                *cum += step;
            }
        }
        // Sum this step's per-(layer, token) FLAT scratch into its cumulative (faithful-H2O
        // `(b)`, the same SUM-across-steps flush as the collapsed `importance`, with the
        // layer axis kept). Guarded so per-layer off → no work.
        if self.per_layer_flat {
            for (cum, &step) in self.layer_flat_cum.iter_mut().zip(self.layer_flat.iter()) {
                *cum += step;
            }
        }
    }

    fn import_gpu_scores(&mut self, flat: &[f32], head: &[f32]) {
        let len = flat.len().min(self.importance.len());
        self.importance[..len].copy_from_slice(&flat[..len]);

        if self.n_kv_heads > 0 {
            let head_len = head.len().min(self.head_importance.len());
            self.head_importance[..head_len].copy_from_slice(&head[..head_len]);

            // Also populate last_layer_head_attn from GPU head importance.
            // On GPU backends, accumulate_layer_gqa() cannot read GPU-only
            // score buffers, so last_layer_head_attn remains empty.
            // Using cumulative head importance as a proxy provides reasonable
            // QCF estimates (proportional to attention distribution).
            let attn_len = head_len.min(self.last_layer_head_attn.len());
            self.last_layer_head_attn[..attn_len].copy_from_slice(&head[..attn_len]);
        }

        // Recompute time-normalized scores if enabled
        if self.time_normalize {
            for t in 0..len {
                let count = self.step_count[t].max(1) as f32;
                self.normalized[t] = self.importance[t] / count;
            }
        }
    }

    fn reset(&mut self) {
        self.importance.fill(0.0);
        self.step_importance.fill(0.0);
        self.head_importance.fill(0.0);
        self.head_step_importance.fill(0.0);
        self.last_layer_head_attn.fill(0.0);
        self.step_count.fill(0);
        self.normalized.fill(0.0);
        self.layer_head.fill(0.0);
        self.layer_head_cum.fill(0.0);
        self.layer_flat.fill(0.0);
        self.layer_flat_cum.fill(0.0);
    }

    fn importance_scores(&self) -> &[f32] {
        if self.time_normalize {
            &self.normalized
        } else {
            &self.importance
        }
    }

    fn raw_importance_scores(&self) -> &[f32] {
        &self.importance
    }

    fn head_importance_scores(&self) -> Option<&[f32]> {
        if self.n_kv_heads > 0 {
            Some(&self.head_importance)
        } else {
            None
        }
    }

    fn last_step_head_attn(&self) -> Option<&[f32]> {
        if self.n_kv_heads > 0 {
            Some(&self.last_layer_head_attn)
        } else {
            None
        }
    }

    fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    fn set_current_layer(&mut self, layer: usize) {
        self.current_layer = layer;
    }

    fn enable_layer_head_dump(&mut self) {
        // GQA mode only — flat mode has no per-KV-head decomposition to preserve.
        if self.n_kv_heads == 0 {
            return;
        }
        let size = self.total_layers * self.n_kv_heads * self.max_seq_len;
        if self.layer_head.len() != size {
            self.layer_head = vec![0.0; size];
        }
        if self.layer_head_cum.len() != size {
            self.layer_head_cum = vec![0.0; size];
        }
        self.dump_layer_head = true;
    }

    fn layer_head_importance(&self) -> Option<&[f32]> {
        if self.dump_layer_head && self.n_kv_heads > 0 {
            // The cumulative (across-step) buffer, not the step-local scratch — over a
            // single probe step they coincide, over a token-by-token prefill the
            // cumulative is the query-agnostic context importance.
            Some(&self.layer_head_cum)
        } else {
            None
        }
    }

    fn enable_per_layer_flat(&mut self) {
        // Works in both flat and GQA mode — the flat `layer_score` (sum over Q-heads) exists
        // in both `accumulate_layer` and `accumulate_layer_gqa`, so (unlike the per-head dump)
        // there is no n_kv_heads guard.
        let size = self.total_layers * self.max_seq_len;
        if self.layer_flat.len() != size {
            self.layer_flat = vec![0.0; size];
        }
        if self.layer_flat_cum.len() != size {
            self.layer_flat_cum = vec![0.0; size];
        }
        self.per_layer_flat = true;
    }

    fn layer_flat_importance(&self) -> Option<&[f32]> {
        if self.per_layer_flat {
            // The cumulative per-(layer, token) FLAT importance — each layer's own
            // accumulated attention with no cross-layer MAX (faithful `H2OKVCache_LayerWise`).
            Some(&self.layer_flat_cum)
        } else {
            None
        }
    }

    fn import_gpu_layer_flat(&mut self, layer_flat: &[f32]) {
        // GPU twin of `import_gpu_scores` for the per-(layer, token) FLAT axis: on a GPU backend the
        // per-layer reduce accumulates on-device, so at eviction the synced buffer OVERWRITES the CPU
        // `layer_flat_cum` (the CPU side held only the prefill seed). No-op when per-layer is off.
        if !self.per_layer_flat {
            return;
        }
        let n = layer_flat.len().min(self.layer_flat_cum.len());
        self.layer_flat_cum[..n].copy_from_slice(&layer_flat[..n]);
    }
}

/// Registration — the engine resolves this via `find_score_producer("attn_score")`. The default
/// scoring path, so the engine force-links this crate non-optionally.
#[distributed_slice(SCORE_PRODUCERS)]
static ATTN_SCORE: ScoreProducerReg = ScoreProducerReg {
    name: "attn_score",
    produces: &[TensorKind::Scores],
    make: |p| Box::new(AttnScoreProducer::new(p)),
};

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
mod tests {
    use super::*;
    use argus_extension_api::{find_score_producer, registered_score_producer_names};

    fn flat(max_seq_len: usize, total_layers: usize, decay: f32) -> AttnScoreProducer {
        AttnScoreProducer::new(ScoreProducerParams {
            max_seq_len,
            n_heads: 1,
            n_kv_heads: 0,
            total_layers,
            last_n_layers: 0,
            decay,
        })
    }

    fn gqa(
        max_seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        total_layers: usize,
        last_n_layers: usize,
        decay: f32,
    ) -> AttnScoreProducer {
        AttnScoreProducer::new(ScoreProducerParams {
            max_seq_len,
            n_heads,
            n_kv_heads,
            total_layers,
            last_n_layers,
            decay,
        })
    }

    // ── Scalar arithmetic regression tests (ported from the engine; the former NEON path was removed,
    // so these now pin the scalar struct against a hand-rolled scalar reference across loop-boundary
    // sizes — exact-multiple, sub-block, and tail). They reach producer-internal step buffers. ──

    /// Helper: scalar reference implementation of accumulate_layer.
    fn accumulate_layer_scalar(
        step_importance: &mut [f32],
        scores: &[f32],
        stride: usize,
        len: usize,
        n_heads_q: usize,
        score_offset: usize,
    ) {
        for t in 0..len {
            let pos = score_offset + t;
            if pos >= step_importance.len() {
                break;
            }
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            step_importance[pos] = step_importance[pos].max(layer_score);
        }
    }

    /// Helper: scalar reference implementation of accumulate_layer_gqa.
    fn accumulate_layer_gqa_scalar(
        step_importance: &mut [f32],
        head_step_importance: &mut [f32],
        last_layer_head_attn: &mut [f32],
        scores: &[f32],
        stride: usize,
        len: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        score_offset: usize,
    ) {
        let n_rep = n_heads_q / n_kv_heads;
        let inv_rep = 1.0 / n_rep as f32;
        for t in 0..len {
            let pos = score_offset + t;
            if pos >= max_seq_len {
                break;
            }
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            step_importance[pos] = step_importance[pos].max(layer_score);

            for kv_h in 0..n_kv_heads {
                let mut group_score = 0.0f32;
                for r in 0..n_rep {
                    group_score += scores[(kv_h * n_rep + r) * stride + t];
                }
                group_score *= inv_rep;
                let idx = kv_h * max_seq_len + pos;
                head_step_importance[idx] = head_step_importance[idx].max(group_score);
                last_layer_head_attn[idx] = group_score;
            }
        }
    }

    #[test]
    fn test_accumulate_layer_vectorized_vs_scalar() {
        // 13 tokens across the former 4-wide loop boundary (3 blocks + 1 tail).
        let max_seq = 16;
        let cache_seq = 13;
        let n_heads_q = 8;
        let stride = max_seq;

        let mut scores = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            for t in 0..cache_seq {
                scores[h * stride + t] = ((h * 13 + t * 7 + 3) % 100) as f32 / 100.0;
            }
        }

        let mut step_ref = vec![0.0f32; max_seq];
        accumulate_layer_scalar(&mut step_ref, &scores, stride, cache_seq, n_heads_q, 0);

        let mut acc = flat(max_seq, 1, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq, n_heads_q, 0);

        for t in 0..cache_seq {
            assert!(
                (acc.step_importance[t] - step_ref[t]).abs() < 1e-5,
                "mismatch at t={}: got {}, expected {}",
                t,
                acc.step_importance[t],
                step_ref[t]
            );
        }
    }

    #[test]
    fn test_accumulate_layer_gqa_vectorized_vs_scalar() {
        let max_seq = 16;
        let cache_seq = 11;
        let n_heads_q = 32;
        let n_kv_heads = 8;
        let stride = max_seq;

        let mut scores = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            for t in 0..cache_seq {
                scores[h * stride + t] = ((h * 11 + t * 5 + 7) % 97) as f32 / 97.0;
            }
        }

        let mut step_ref = vec![0.0f32; max_seq];
        let mut head_step_ref = vec![0.0f32; n_kv_heads * max_seq];
        let mut last_attn_ref = vec![0.0f32; n_kv_heads * max_seq];
        accumulate_layer_gqa_scalar(
            &mut step_ref,
            &mut head_step_ref,
            &mut last_attn_ref,
            &scores,
            stride,
            cache_seq,
            n_heads_q,
            n_kv_heads,
            max_seq,
            0,
        );

        let mut acc = gqa(max_seq, n_heads_q, n_kv_heads, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer_gqa(&scores, stride, cache_seq, n_heads_q, n_kv_heads, 0);

        for t in 0..cache_seq {
            assert!(
                (acc.step_importance[t] - step_ref[t]).abs() < 1e-5,
                "flat mismatch at t={}: got {}, expected {}",
                t,
                acc.step_importance[t],
                step_ref[t]
            );
        }
        for kv_h in 0..n_kv_heads {
            for t in 0..cache_seq {
                let idx = kv_h * max_seq + t;
                assert!(
                    (acc.head_step_importance[idx] - head_step_ref[idx]).abs() < 1e-5,
                    "head_step mismatch kv_h={} t={}: got {}, expected {}",
                    kv_h,
                    t,
                    acc.head_step_importance[idx],
                    head_step_ref[idx]
                );
                assert!(
                    (acc.last_layer_head_attn[idx] - last_attn_ref[idx]).abs() < 1e-5,
                    "last_attn mismatch kv_h={} t={}: got {}, expected {}",
                    kv_h,
                    t,
                    acc.last_layer_head_attn[idx],
                    last_attn_ref[idx]
                );
            }
        }
    }

    #[test]
    fn test_end_step_vectorized_time_normalize() {
        let max_seq = 7;
        let n_heads_q = 4;
        let n_kv_heads = 2;

        let mut acc = gqa(max_seq, n_heads_q, n_kv_heads, 1, 0, 0.0);
        acc.set_time_normalize(true);
        acc.set_active(true);

        acc.begin_step();
        let mut scores1 = vec![0.0f32; n_heads_q * max_seq];
        for h in 0..n_heads_q {
            for t in 0..5 {
                scores1[h * max_seq + t] = (t + 1) as f32 * 0.1 + h as f32 * 0.01;
            }
        }
        acc.accumulate_layer_gqa(&scores1, max_seq, 5, n_heads_q, n_kv_heads, 0);
        acc.end_step();

        acc.begin_step();
        let mut scores2 = vec![0.0f32; n_heads_q * max_seq];
        for h in 0..n_heads_q {
            for t in 0..7 {
                scores2[h * max_seq + t] = (7 - t) as f32 * 0.15 + h as f32 * 0.02;
            }
        }
        acc.accumulate_layer_gqa(&scores2, max_seq, 7, n_heads_q, n_kv_heads, 0);
        acc.end_step();

        assert_eq!(acc.step_count[0], 2);
        assert_eq!(acc.step_count[4], 2);
        assert_eq!(acc.step_count[5], 1);
        assert_eq!(acc.step_count[6], 1);

        let imp = acc.importance_scores();
        let raw = acc.raw_importance_scores();
        for t in 0..max_seq {
            let count = acc.step_count[t].max(1) as f32;
            let expected = raw[t] / count;
            assert!(
                (imp[t] - expected).abs() < 1e-5,
                "time_normalize mismatch at t={}: got {}, expected {}",
                t,
                imp[t],
                expected,
            );
        }

        let head_imp = acc.head_importance_scores().unwrap();
        for i in 0..head_imp.len() {
            assert!(head_imp[i].is_finite(), "head_imp[{}] not finite", i);
        }
    }

    #[test]
    fn test_accumulate_layer_exact_multiple_of_4() {
        let max_seq = 8;
        let cache_seq = 8;
        let n_heads_q = 4;
        let stride = max_seq;

        let mut scores = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            for t in 0..cache_seq {
                scores[h * stride + t] = (h as f32 + 1.0) * (t as f32 + 1.0) * 0.1;
            }
        }

        let mut step_ref = vec![0.0f32; max_seq];
        accumulate_layer_scalar(&mut step_ref, &scores, stride, cache_seq, n_heads_q, 0);

        let mut acc = flat(max_seq, 1, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq, n_heads_q, 0);

        for t in 0..cache_seq {
            assert!(
                (acc.step_importance[t] - step_ref[t]).abs() < 1e-5,
                "t={}: got {}, expected {}",
                t,
                acc.step_importance[t],
                step_ref[t]
            );
        }
    }

    #[test]
    fn test_accumulate_layer_fewer_than_4_tokens() {
        let max_seq = 8;
        let cache_seq = 3;
        let n_heads_q = 2;
        let stride = max_seq;

        let scores = vec![
            1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, // head 0
            4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, // head 1
        ];

        let mut acc = flat(max_seq, 1, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq, n_heads_q, 0);

        assert!((acc.step_importance[0] - 5.0).abs() < 1e-6);
        assert!((acc.step_importance[1] - 7.0).abs() < 1e-6);
        assert!((acc.step_importance[2] - 9.0).abs() < 1e-6);
    }

    /// NaN scores in accumulate_layer_gqa must not poison head_step_importance / last_layer_head_attn
    /// (MAX swallows NaN for the cumulative buffers; the CAOTE overwrite has an explicit NaN→0 guard).
    #[test]
    fn test_accumulate_gqa_nan_scores_handled() {
        let max_seq = 8;
        let n_heads_q = 4;
        let n_kv_heads = 2;
        let mut acc = gqa(max_seq, n_heads_q, n_kv_heads, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        let stride = max_seq;
        let cache_seq_len = 4;

        let mut scores_l0 = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            scores_l0[h * stride] = 0.4;
            scores_l0[h * stride + 1] = 0.3;
            scores_l0[h * stride + 2] = 0.2;
            scores_l0[h * stride + 3] = 0.1;
        }
        acc.accumulate_layer_gqa(&scores_l0, stride, cache_seq_len, n_heads_q, n_kv_heads, 0);

        assert!(
            acc.step_importance[0] > 0.0,
            "flat step_importance[0] should be > 0 after L0"
        );
        let idx0 = 0usize;
        assert!(
            acc.head_step_importance[idx0] > 0.0,
            "head_step_importance[0] should be > 0 after L0"
        );
        assert!(
            acc.last_layer_head_attn[idx0] > 0.0,
            "last_layer_head_attn[0] should be > 0 after L0"
        );

        let scores_l1 = vec![f32::NAN; n_heads_q * stride];
        acc.accumulate_layer_gqa(&scores_l1, stride, cache_seq_len, n_heads_q, n_kv_heads, 0);

        assert!(
            acc.step_importance[0] > 0.0,
            "flat step_importance[0] must survive NaN layer: got {}",
            acc.step_importance[0]
        );
        assert!(
            acc.head_step_importance[idx0] > 0.0,
            "head_step_importance[0] must survive NaN layer: got {}",
            acc.head_step_importance[idx0]
        );
        assert_eq!(
            acc.last_layer_head_attn[idx0], 0.0,
            "last_layer_head_attn should be 0 after NaN layer (NaN guard)"
        );

        acc.end_step();
        assert!(
            acc.importance[0] > 0.0,
            "cumulative importance[0] should be > 0 after end_step"
        );
        let hi = acc.head_importance_scores().unwrap();
        assert!(
            hi[idx0] > 0.0,
            "cumulative head_importance[0] should be > 0 after end_step"
        );
    }

    #[test]
    fn flat_per_layer_max_then_step_sum() {
        // Within a step: MAX across layers; across steps: SUM of per-step MAX.
        let mut acc = flat(4, 2, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&[0.1, 0.2, 0.3, 0.4], 4, 4, 1, 0);
        acc.accumulate_layer(&[0.4, 0.1, 0.1, 0.4], 4, 4, 1, 0);
        acc.end_step();
        let imp = acc.importance_scores();
        assert!((imp[0] - 0.4).abs() < 1e-6);
        assert!((imp[1] - 0.2).abs() < 1e-6);
        assert!((imp[2] - 0.3).abs() < 1e-6);
        assert!((imp[3] - 0.4).abs() < 1e-6);
    }

    /// A2SF decay=0.0 bit-identity anchor (ported from the engine's `test_a2sf_decay_zero_bit_identical`):
    /// the `if self.decay > 0.0` guard must keep decay=0 a pure SUM with no float drift.
    #[test]
    fn decay_zero_bit_identical() {
        let mut acc = flat(4, 1, 0.0);
        acc.set_active(true);
        for _ in 0..3 {
            acc.begin_step();
            acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 4, 4, 1, 0);
            acc.end_step();
        }
        let imp = acc.importance_scores();
        assert_eq!(imp[0], 3.0);
        assert_eq!(imp[1], 6.0);
        assert_eq!(imp[2], 9.0);
        assert_eq!(imp[3], 12.0);
    }

    #[test]
    fn gqa_groups_q_heads_and_caote() {
        // 4 Q-heads, 2 KV-heads → n_rep=2.
        let mut acc = AttnScoreProducer::new(ScoreProducerParams {
            max_seq_len: 4,
            n_heads: 4,
            n_kv_heads: 2,
            total_layers: 1,
            last_n_layers: 0,
            decay: 0.0,
        });
        acc.set_active(true);
        acc.begin_step();
        let scores = vec![
            1.0, 0.0, 0.0, 0.0, // Q0 → KV0
            0.0, 1.0, 0.0, 0.0, // Q1 → KV0
            0.0, 0.0, 1.0, 0.0, // Q2 → KV1
            0.0, 0.0, 0.0, 1.0, // Q3 → KV1
        ];
        acc.accumulate_layer_gqa(&scores, 4, 4, 4, 2, 0);
        acc.end_step();
        let head = acc.head_importance_scores().unwrap();
        assert!((head[0] - 0.5).abs() < 1e-6); // KV0 tok0 = avg(1,0)
        assert!((head[1] - 0.5).abs() < 1e-6); // KV0 tok1 = avg(0,1)
        assert!((head[6] - 0.5).abs() < 1e-6); // KV1 tok2
        assert!((head[7] - 0.5).abs() < 1e-6); // KV1 tok3
        // CAOTE last-layer attn populated (same single layer).
        let attn = acc.last_step_head_attn().unwrap();
        assert!((attn[0] - 0.5).abs() < 1e-6);
        // Flat importance also updated (backward compat): sum across all Q-heads per token = 1.0.
        let imp = acc.importance_scores();
        assert!((imp[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flat_mode_has_no_head_buffers() {
        let acc = flat(4, 1, 0.0);
        assert_eq!(acc.n_kv_heads(), 0);
        assert!(acc.head_importance_scores().is_none());
        assert!(acc.last_step_head_attn().is_none());
    }

    #[test]
    fn registers_into_slice() {
        let reg =
            find_score_producer("attn_score").expect("attn_score registered in SCORE_PRODUCERS");
        assert_eq!(reg.name, "attn_score");
        assert_eq!(reg.produces, &[TensorKind::Scores]);
        let producer = (reg.make)(ScoreProducerParams {
            max_seq_len: 8,
            n_heads: 2,
            n_kv_heads: 0,
            total_layers: 1,
            last_n_layers: 0,
            decay: 0.0,
        });
        assert_eq!(producer.name(), "attn_score");
        assert!(registered_score_producer_names().contains(&"attn_score"));
    }

    // ── per-(layer, KV-head) importance dump (IMP-1) ──

    /// The non-collapsed buffer preserves DISTINCT per-layer slots — proving the
    /// layer axis survives (unlike the MAX-collapsed `head_importance`).
    #[test]
    fn layer_head_dump_preserves_distinct_layer_slots() {
        // 4 Q-heads, 2 KV-heads (n_rep=2), 2 layers, max_seq=4.
        let mut acc = AttnScoreProducer::new(ScoreProducerParams {
            max_seq_len: 4,
            n_heads: 4,
            n_kv_heads: 2,
            total_layers: 2,
            last_n_layers: 0,
            decay: 0.0,
        });
        acc.set_active(true);
        // Off by default → no buffer, accessor None.
        assert!(acc.layer_head_importance().is_none());
        acc.enable_layer_head_dump();
        acc.begin_step();

        // Layer 0: Q0→KV0 puts all mass on tok0 → KV0/tok0 = avg(1,0) = 0.5.
        let l0 = vec![
            1.0, 0.0, 0.0, 0.0, // Q0 → KV0
            0.0, 0.0, 0.0, 0.0, // Q1 → KV0
            0.0, 0.0, 0.0, 0.0, // Q2 → KV1
            0.0, 0.0, 0.0, 0.0, // Q3 → KV1
        ];
        acc.set_current_layer(0);
        acc.accumulate_layer_gqa(&l0, 4, 4, 4, 2, 0);

        // Layer 1: Q2→KV1 puts all mass on tok2 → KV1/tok2 = avg(1,0) = 0.5.
        let l1 = vec![
            0.0, 0.0, 0.0, 0.0, // Q0 → KV0
            0.0, 0.0, 0.0, 0.0, // Q1 → KV0
            0.0, 0.0, 1.0, 0.0, // Q2 → KV1
            0.0, 0.0, 0.0, 0.0, // Q3 → KV1
        ];
        acc.set_current_layer(1);
        acc.accumulate_layer_gqa(&l1, 4, 4, 4, 2, 0);

        // The dump reports the cumulative buffer, flushed by end_step — the same point
        // the real flow reads it (post_prefill, after forward_into's end_step). Over a
        // single step the cumulative equals the step values (probe-path invariant).
        acc.end_step();

        let buf = acc.layer_head_importance().expect("dump enabled");
        assert_eq!(
            buf.len(),
            2 * 2 * 4,
            "total_layers * n_kv_heads * max_seq_len"
        );

        let idx = |layer: usize, kv: usize, pos: usize| (layer * 2 + kv) * 4 + pos;
        // Layer 0 wrote KV0/tok0 = 0.5; layer 1 left that slot at 0.
        assert!((buf[idx(0, 0, 0)] - 0.5).abs() < 1e-6);
        assert_eq!(buf[idx(1, 0, 0)], 0.0);
        // Layer 1 wrote KV1/tok2 = 0.5; layer 0 left that slot at 0.
        assert!((buf[idx(1, 1, 2)] - 0.5).abs() < 1e-6);
        assert_eq!(buf[idx(0, 1, 2)], 0.0);
        // Non-all-zero, and the two per-layer blocks DIFFER (layer axis preserved).
        assert!(buf.iter().any(|&v| v != 0.0));
        assert_ne!(&buf[0..2 * 4], &buf[2 * 4..]);

        // reset() clears the buffer.
        acc.reset();
        assert!(
            acc.layer_head_importance()
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    /// Off by default (no `enable_layer_head_dump`) → accessor None, no buffer allocated.
    #[test]
    fn layer_head_dump_off_by_default() {
        let mut acc = gqa(4, 4, 2, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.set_current_layer(0);
        acc.accumulate_layer_gqa(&[0.0; 16], 4, 4, 4, 2, 0);
        assert!(acc.layer_head_importance().is_none());
    }

    /// Flat mode (n_kv_heads = 0): enabling the dump is a no-op (no per-head axis).
    #[test]
    fn layer_head_dump_flat_mode_stays_none() {
        let mut acc = flat(4, 2, 0.0);
        acc.enable_layer_head_dump();
        assert!(acc.layer_head_importance().is_none());
    }

    /// INV-147: enabling the per-layer dump must NOT change the collapsed flat /
    /// per-head importance the eviction policy ranks on. Identical accumulates with
    /// the dump ON vs OFF must yield bit-identical collapsed buffers.
    #[test]
    fn layer_head_dump_does_not_perturb_collapsed_importance() {
        let scores_l0 = vec![
            0.7, 0.1, 0.1, 0.1, // Q0 → KV0
            0.2, 0.5, 0.2, 0.1, // Q1 → KV0
            0.1, 0.2, 0.6, 0.1, // Q2 → KV1
            0.1, 0.1, 0.1, 0.7, // Q3 → KV1
        ];
        let scores_l1 = vec![
            0.3, 0.3, 0.3, 0.1, // Q0 → KV0
            0.6, 0.2, 0.1, 0.1, // Q1 → KV0
            0.1, 0.1, 0.1, 0.7, // Q2 → KV1
            0.4, 0.4, 0.1, 0.1, // Q3 → KV1
        ];
        let run = |dump: bool| {
            let mut acc = AttnScoreProducer::new(ScoreProducerParams {
                max_seq_len: 4,
                n_heads: 4,
                n_kv_heads: 2,
                total_layers: 2,
                last_n_layers: 0,
                decay: 0.0,
            });
            acc.set_active(true);
            if dump {
                acc.enable_layer_head_dump();
            }
            acc.begin_step();
            acc.set_current_layer(0);
            acc.accumulate_layer_gqa(&scores_l0, 4, 4, 4, 2, 0);
            acc.set_current_layer(1);
            acc.accumulate_layer_gqa(&scores_l1, 4, 4, 4, 2, 0);
            acc.end_step();
            (
                acc.importance_scores().to_vec(),
                acc.head_importance_scores().unwrap().to_vec(),
                acc.last_step_head_attn().unwrap().to_vec(),
            )
        };
        let off = run(false);
        let on = run(true);
        assert_eq!(off.0, on.0, "dump must not perturb flat importance");
        assert_eq!(off.1, on.1, "dump must not perturb per-head importance");
        assert_eq!(off.2, on.2, "dump must not perturb CAOTE last-step attn");
    }

    /// The per-(layer, KV-head) dump accumulates ACROSS steps (the `prefill_end`
    /// query-agnostic semantics), and over a SINGLE step it equals the step value
    /// (the post-prefill-probe path stays byte-identical). One KV-head, one layer,
    /// max_seq=2 so the two token positions are the two ranked slots.
    #[test]
    fn layer_head_dump_is_cumulative_across_steps() {
        let mk = || {
            let mut acc = AttnScoreProducer::new(ScoreProducerParams {
                max_seq_len: 2,
                n_heads: 2,
                n_kv_heads: 1,
                total_layers: 1,
                last_n_layers: 0,
                decay: 0.0,
            });
            acc.set_active(true);
            acc.enable_layer_head_dump();
            acc
        };
        // One layer-0 step that puts mass 1.0 (avg over 2 Q-heads of [1,1]) on tok0.
        // group_score(tok0) = (1+1)/2 = 1.0; tok1 = 0.
        let step_scores = vec![
            1.0, 0.0, // Q0 → KV0
            1.0, 0.0, // Q1 → KV0
        ];
        let run_n = |steps: usize| {
            let mut acc = mk();
            for _ in 0..steps {
                acc.begin_step();
                acc.set_current_layer(0);
                acc.accumulate_layer_gqa(&step_scores, 2, 2, 2, 1, 0);
                acc.end_step();
            }
            acc.layer_head_importance().unwrap().to_vec()
        };

        // Single step (probe path): cumulative == the one step's value (1.0 on tok0).
        let one = run_n(1);
        assert!((one[0] - 1.0).abs() < 1e-6, "tok0 single-step = 1.0");
        assert_eq!(one[1], 0.0, "tok1 untouched");

        // Three steps (token-by-token prefill): the SAME slot sums to 3.0 — cumulative.
        let three = run_n(3);
        assert!(
            (three[0] - 3.0).abs() < 1e-6,
            "tok0 must accumulate across 3 steps (got {})",
            three[0]
        );
        assert_eq!(three[1], 0.0, "tok1 still 0 across steps");
        // It is genuinely cumulative, not a most-recent-step snapshot.
        assert!(three[0] > one[0], "more steps ⇒ strictly larger cumulative");
    }
}
