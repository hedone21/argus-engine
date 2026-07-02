//! StepHook trait: abstracts per-step cache management (eviction vs quant-window flush).
//!
//! The generic eval loop calls these hooks without knowing the cache management
//! policy. Each implementation encapsulates its own eviction/flush logic and
//! QCF metric collection.

use crate::inference::attention_scores::AttentionScoreAccumulator;

/// Result of a post-decode-step hook invocation.
#[derive(Debug, Default)]
pub struct PostStepResult {
    /// Whether any eviction/flush occurred this step.
    pub evicted: bool,
    /// Number of tokens removed (eviction) or quantized (quant-window flush).
    pub tokens_affected: usize,
    /// New start_pos after eviction (if evicted, caller should update).
    pub new_start_pos: Option<usize>,
}

/// Snapshot of KV cache state for choice-level restore.
///
/// Phase α-K ①-c: `C: KVCacheOps` 바운드 제거 — `C` 는 concrete `KVCache`/`QuantizedRecentWindowCache` 둘뿐이고
/// impl 이 이미 concrete 타입 인자라 바운드 불요. KVCacheOps 폐기(Step 5)의 eval 차단 해소.
pub trait CacheSnapshot<C>: Send {
    /// Restore caches to the snapshotted state.
    fn restore_to(&self, caches: &mut [C]);
}

/// Per-step cache management hook for the generic eval loop.
///
/// Implementations:
/// - `EvictionHook` (KVCache): budget-based eviction + value-aware/attn QCF
/// - `QuantWindowFlushHook` (QuantizedRecentWindowCache): flush proxy collection (NMSE + OPR)
pub trait StepHook<C> {
    /// Called after prefill completes. Handles chunked-prefill eviction
    /// residuals or flush proxy collection.
    fn post_prefill(&mut self, caches: &mut [C]);

    /// The PFA observation window this hook wants armed during prefill, or `None` (the default —
    /// no prefill-attention producer). Direction-B unification: when a per-head SnapKV/PyramidKV
    /// keep-set is configured, `EvictionHook` returns `Some(window_size)` so the generic prefill
    /// arms a `TensorKind::PrefillAttention` producer at exactly that width (the SAME caps-driven
    /// window `build_standard_loop`/`build_bench_loop` arm at). `None` keeps prefill byte-identical.
    fn prefill_attn_window(&self) -> Option<usize> {
        None
    }

    /// Hand the per-layer PFA buffer (`[n_heads_q * prefix_len]` per layer) the armed prefill produced
    /// to this hook, so its `post_prefill` can apply the keep-set. Default no-op (only `EvictionHook`
    /// with a prefill keep-set configured consumes it). Called once, right after the prefill forward.
    fn stage_prefill_attn(&mut self, _pfa: Vec<Vec<f32>>) {}

    /// Reset caches for a new question evaluation.
    fn reset_caches(&mut self, caches: &mut [C]);

    /// Create a snapshot of the current cache state (after prefill).
    fn snapshot(&self, caches: &[C]) -> Box<dyn CacheSnapshot<C>>;

    /// Provide mutable access to the score accumulator (if any).
    /// EvictionHook returns Some; QuantWindowFlushHook returns None.
    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator>;

    /// Update the effective budget (used by ratio-mode per-question budget).
    /// Default is no-op (e.g., QuantWindowFlushHook ignores budget).
    fn set_effective_budget(&mut self, _budget: usize) {}

    /// Returns true if this hook needs a score probe step after prefill.
    /// True when score-based eviction will be needed (cache exceeds budget).
    /// The probe re-feeds the last prompt token as a decode step to populate
    /// the score accumulator before post_prefill eviction.
    fn needs_score_probe(&self, _caches: &[C]) -> bool {
        false
    }

    /// Whether this hook's eviction ranks tokens on accumulated attention scores
    /// (vs. position). The loop pairs this with `--evict-timing prefill_end` to
    /// decide whether prefill must run token-by-token to accumulate query-agnostic
    /// context importance — a score-free (positional) policy needs no such pass.
    /// Default `false`.
    fn ranks_on_scores(&self) -> bool {
        false
    }

    /// Called once per token during token-by-token prefill, **after** that token's
    /// forward (so the cache reflects the just-ingested token and per-step importance
    /// is accumulated). `orig_token_idx` is the token's original prompt index.
    ///
    /// `--evict-timing prefill_streaming` (variant b) uses this to cap the resident
    /// cache at the budget: on overflow it evicts down to a low-water mark, keeping
    /// occupancy bounded by `budget` (+ one step's slack). Default no-op — the other
    /// timings and the quant-window hook evict only at `post_prefill`, so their
    /// token-by-token prefill stays byte-identical (`INV-147`).
    fn on_prefill_step(&mut self, _caches: &mut [C], _orig_token_idx: usize) {}

    /// Drain the `prefill_streaming` per-event `evict_importance` (IMP-1) snapshots
    /// captured during prefill (one per eviction event, `schema_version: 2`). The eval
    /// loop writes them after prefill, adding the per-question metadata. Default empty
    /// (only `EvictionHook` in streaming mode produces any).
    fn take_streaming_evict_dumps(&mut self) -> Vec<super::dump::EvictImportanceSnapshot> {
        Vec::new()
    }

    /// Cache-specific per-question JSON fields (e.g., quant_q2_tokens).
    fn extra_question_fields(&self, caches: &[C]) -> serde_json::Value;

    /// Cache-specific top-level config JSON fields.
    fn extra_config_fields(&self) -> serde_json::Value;

    /// Take the most recent `evict_importance` (IMP-1) dump snapshot captured during
    /// `post_prefill`, if any. Default `None` (only `EvictionHook` with the dump
    /// enabled produces one). The eval loop drains it after `post_prefill`, adds the
    /// per-question metadata, and writes the record.
    fn take_evict_importance_dump(&mut self) -> Option<super::dump::EvictImportanceSnapshot> {
        None
    }
}
