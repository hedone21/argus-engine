//! StepHook trait: abstracts per-step cache management (eviction vs KIVI flush).
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
    /// Number of tokens removed (eviction) or quantized (KIVI flush).
    pub tokens_affected: usize,
    /// New start_pos after eviction (if evicted, caller should update).
    pub new_start_pos: Option<usize>,
}

/// Snapshot of KV cache state for choice-level restore.
///
/// Phase α-K ①-c: `C: KVCacheOps` 바운드 제거 — `C` 는 concrete `KVCache`/`KiviCache` 둘뿐이고
/// impl 이 이미 concrete 타입 인자라 바운드 불요. KVCacheOps 폐기(Step 5)의 eval 차단 해소.
pub trait CacheSnapshot<C>: Send {
    /// Restore caches to the snapshotted state.
    fn restore_to(&self, caches: &mut [C]);
}

/// Per-step cache management hook for the generic eval loop.
///
/// Implementations:
/// - `EvictionHook` (KVCache): budget-based eviction + CAOTE/attn QCF
/// - `KiviHook` (KiviCache): flush proxy collection (NMSE + OPR)
pub trait StepHook<C> {
    /// Called after prefill completes. Handles chunked-prefill eviction
    /// residuals or flush proxy collection.
    fn post_prefill(&mut self, caches: &mut [C]);

    /// Reset caches for a new question evaluation.
    fn reset_caches(&mut self, caches: &mut [C]);

    /// Create a snapshot of the current cache state (after prefill).
    fn snapshot(&self, caches: &[C]) -> Box<dyn CacheSnapshot<C>>;

    /// Provide mutable access to the score accumulator (if any).
    /// EvictionHook returns Some; KiviHook returns None.
    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator>;

    /// Update the effective budget (used by ratio-mode per-question budget).
    /// Default is no-op (e.g., KiviHook ignores budget).
    fn set_effective_budget(&mut self, _budget: usize) {}

    /// Returns true if this hook needs a score probe step after prefill.
    /// True when score-based eviction will be needed (cache exceeds budget).
    /// The probe re-feeds the last prompt token as a decode step to populate
    /// the score accumulator before post_prefill eviction.
    fn needs_score_probe(&self, _caches: &[C]) -> bool {
        false
    }

    /// Cache-specific per-question JSON fields (e.g., kivi_q2_tokens).
    fn extra_question_fields(&self, caches: &[C]) -> serde_json::Value;

    /// Cache-specific top-level config JSON fields.
    fn extra_config_fields(&self) -> serde_json::Value;
}
