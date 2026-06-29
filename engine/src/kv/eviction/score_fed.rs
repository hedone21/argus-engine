//! Shared "score-fed eviction" body — the one place the `gpu_sync → extract → route`
//! quartet lives, so a score-fed feature (faithful-H2O `(b)` per-layer, value-aware `a_i`,
//! collapsed importance) is wired ONCE instead of being replicated across the three binaries
//! that drive eviction (`eval` `EvictionHook`, `bench` `EvictionStage`, `chat` `try_evict`).
//!
//! Each binary keeps its own *trigger* (cli=none, bench=pressure edge, eval=post_prefill/
//! streaming, chat=turn-boundary) and its own ownership/locking/cache-wrapping; it calls into
//! these free functions for the *body*. The reset half (`acc.reset()` + [`reset_gpu_layer_flat`])
//! is sequenced by the caller after a confirmed evict because it interleaves with each site's
//! own bookkeeping (dump capture, counters).
//!
//! The DRIVE layer below this (`CacheManager::run_policy_eviction` + `drive_mutation_layer`) is
//! already unified — see `cache_manager.rs`. This module unifies the score-fed TRIGGER layer
//! above it.

use crate::backend::Backend;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::kv::cache_manager::{CacheManager, EvictionResult};
use crate::kv::kv_cache::KVCache;
use anyhow::Result;

/// Borrowed `(collapsed scores, value-aware a_i, per-layer FLAT)` triple — the argument shape the
/// `CacheManager` eviction entry points take, produced by [`ExtractedScores::as_args`] and consumed
/// by [`route_evict`].
pub type EvictScoreArgs<'a> = (
    Option<&'a [f32]>,
    Option<&'a [f32]>,
    Option<(&'a [f32], usize)>,
);

/// Scores pulled out of an active accumulator for one score-fed eviction. Owns its buffers so the
/// caller can release any lock guard (bench) before routing into the `CacheManager`.
pub struct ExtractedScores {
    /// Collapsed per-token importance (cross-layer MAX → SUM-over-steps).
    pub scores: Vec<f32>,
    /// Value-aware `a_i`: last-layer last-step per-`(kv_head, pos)` attention, when an
    /// `AttnWeights` producer is active. `None` → the stage falls back to flat importance.
    pub last_attn: Option<Vec<f32>>,
    /// Faithful-H2O `(b)`: per-`(layer, token)` FLAT importance `(layer_flat, max_seq)` — each
    /// layer's own heavy hitters with no cross-layer MAX. `Some` only on the opt-in faithful path.
    pub per_layer: Option<(Vec<f32>, usize)>,
}

impl ExtractedScores {
    /// Borrow as the `(scores, last_attn, per_layer)` argument triple [`route_evict`] takes.
    pub fn as_args(&self) -> EvictScoreArgs<'_> {
        (
            Some(self.scores.as_slice()),
            self.last_attn.as_deref(),
            self.per_layer.as_ref().map(|(lf, ms)| (lf.as_slice(), *ms)),
        )
    }
}

/// Extract the `(importance, value-aware a_i, per-layer FLAT)` triple from an accumulator.
/// Returns `None` when the accumulator is inactive — the caller then routes score-free
/// (degrade to recency). Identical at all three sites (only the way each obtains `acc` differs).
pub fn extract_scores(acc: &AttentionScoreAccumulator) -> Option<ExtractedScores> {
    if !acc.is_active() {
        return None;
    }
    Some(ExtractedScores {
        scores: acc.importance_scores().to_vec(),
        last_attn: acc.last_step_head_attn().map(|s| s.to_vec()),
        // `Some` only when per-layer FLAT is armed (faithful-H2O); otherwise the collapsed path.
        per_layer: acc
            .layer_flat_importance()
            .map(|lf| (lf.to_vec(), acc.importance_scores().len())),
    })
}

/// Route extracted scores to the matching `CacheManager` entry point — the single copy of the
/// `(per_layer, scores) × (force, maybe)` dispatch that used to be inlined at every site.
///
/// `force == true` → `force_evict_with_*` (ratio-driven, eval/bench + chat at-pressure); `false`
/// → `maybe_evict_with_*` (pressure-checked, chat below-pressure). Per-layer FLAT takes priority
/// over collapsed `scores`; all-`None` is the score-free recency fallback.
#[allow(clippy::too_many_arguments)]
pub fn route_evict(
    cm: &CacheManager,
    caches: &mut [KVCache],
    scores: Option<&[f32]>,
    last_attn: Option<&[f32]>,
    per_layer: Option<(&[f32], usize)>,
    force: bool,
    target_ratio: f32,
) -> Result<EvictionResult> {
    match (per_layer, scores, force) {
        (Some((lf, ms)), _, true) => {
            cm.force_evict_with_per_layer_scores(caches, target_ratio, lf, ms, last_attn)
        }
        (Some((lf, ms)), _, false) => {
            cm.maybe_evict_with_per_layer_scores(caches, lf, ms, last_attn)
        }
        (None, Some(sc), true) => cm.force_evict_with_scores(caches, target_ratio, sc, last_attn),
        (None, Some(sc), false) => cm.maybe_evict_with_scores(caches, sc, last_attn),
        (None, None, true) => cm.force_evict(caches, target_ratio),
        (None, None, false) => cm.maybe_evict(caches),
    }
}

/// Sync GPU-accumulated attention scores (collapsed + faithful-H2O `(b)` per-layer) into the CPU
/// accumulator before a score-based eviction reads them. No-op on CPU backends / when the GPU
/// accumulator is inactive, so the host path is byte-identical. Used by `eval` (always) and `chat`
/// (closing the chat-on-GPU sync gap); `bench` uses the CPU accumulate path and never calls this.
#[cfg(feature = "opencl")]
pub fn sync_gpu_scores_to_cpu(acc: &mut AttentionScoreAccumulator, backend: &dyn Backend) {
    if acc.is_active()
        && let Some(ocl_be) = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
        && let Some(gpu_acc) = ocl_be.gpu_score_acc()
        && gpu_acc.is_active()
        && let Ok((flat, head)) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())
    {
        acc.import_gpu_scores(&flat, &head);
        // Faithful-H2O (b): when per-layer FLAT is armed, also sync the GPU per-layer cumulative
        // (accumulated on-device by the per-layer reduce) into the CPU `layer_flat_cum`.
        if acc.layer_flat_importance().is_some()
            && let Ok(layer_flat) = gpu_acc.sync_layer_flat_to_cpu(ocl_be.queue.as_core())
        {
            acc.import_gpu_layer_flat(&layer_flat);
        }
    }
}

/// CPU-only build: no GPU accumulator exists, so the sync is a no-op.
#[cfg(not(feature = "opencl"))]
pub fn sync_gpu_scores_to_cpu(_acc: &mut AttentionScoreAccumulator, _backend: &dyn Backend) {}

/// Reset the GPU per-`(layer, token)` FLAT cumulative buffer at an eviction boundary, in lockstep
/// with the CPU `acc.reset()` (faithful-H2O `(b)`). Without this a 2nd eviction would rank GPU
/// per-layer importance monotonically accumulated since prefill (misaligned with the compacted
/// cache) while the CPU twin starts fresh — they would diverge. Touches ONLY the per-layer buffer
/// (the collapsed GPU buffers keep their existing behavior — INV-147). No-op on CPU / unarmed.
#[cfg(feature = "opencl")]
pub fn reset_gpu_layer_flat(acc: &AttentionScoreAccumulator, backend: &dyn Backend) {
    if acc.is_active()
        && acc.layer_flat_importance().is_some()
        && let Some(ocl_be) = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
        && let Some(gpu_acc) = ocl_be.gpu_score_acc()
        && gpu_acc.is_active()
    {
        let _ = gpu_acc.reset_layer_flat(ocl_be.queue.as_core());
    }
}

/// CPU-only build: no GPU per-layer buffer exists, so the reset is a no-op.
#[cfg(not(feature = "opencl"))]
pub fn reset_gpu_layer_flat(_acc: &AttentionScoreAccumulator, _backend: &dyn Backend) {}
