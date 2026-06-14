use crate::format::Merge;
use crate::kv::kv_cache::KVCache;
use anyhow::Result;

/// Trait for KV cache eviction strategies.
///
/// Implementations decide WHEN and HOW to evict tokens from the cache.
/// This follows the Strategy pattern and SOLID principles:
/// - Single Responsibility: each policy handles one eviction strategy
/// - Open/Closed: add new policies without modifying existing code
/// - Liskov Substitution: all policies are interchangeable via this trait
/// - Dependency Inversion: consumers depend on this trait, not concrete types
pub trait EvictionPolicy: Send + Sync {
    /// Determines whether eviction should be triggered based on cache state
    /// and available system memory.
    fn should_evict(&self, cache: &KVCache, mem_available: usize) -> bool;

    /// Performs the actual eviction, reducing cache to `target_len` tokens.
    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()>;

    /// Returns the name of this policy (for logging/debugging).
    fn name(&self) -> &str;

    /// Performs eviction using per-token importance scores.
    /// Default implementation ignores scores and delegates to `evict()`.
    /// Override in score-aware policies like H2O.
    fn evict_with_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        _importance: &[f32],
    ) -> Result<()> {
        self.evict(cache, target_len)
    }

    /// Per-KV-head eviction with GQA-aware importance scores.
    ///
    /// `head_importance` is `[n_kv_heads * max_seq_len]` (row-major): each KV head
    /// has its own importance ranking, enabling per-head token selection.
    ///
    /// Default: ignores head scores, delegates to `evict_with_scores()`.
    /// Override in GQA-aware policies like H2O+.
    fn evict_with_head_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        flat_importance: &[f32],
        _head_importance: &[f32],
        _n_kv_heads: usize,
    ) -> Result<()> {
        self.evict_with_scores(cache, target_len, flat_importance)
    }

    /// (Phase α-K substep 3c-evict) keep-list 산출 — in-place `evict*` 와 **동일 의미**의
    /// 보존 토큰 목록(+merges)을 산출하되 버퍼는 건드리지 않는다. 호출자가
    /// [`crate::format::KVCacheFormat::compact`]`(keep, merges)` 로 적용한다 (
    /// interior-mutability eviction — eviction 을 `&mut KVCache` 가 아니라 `&self` compact 로 옮겨
    /// `Vec<Arc<StandardFormat>>` by-value 소유 ⊥ 연속 `&mut [KVCache]` 충돌을 해소).
    ///
    /// `keep` 은 **prefix 포함 ascending** 이어야 한다 — `compact(keep, write_start=0)` 의 재배치가
    /// in-place `evict` 의 버퍼/`current_pos` 와 bit-identical 이도록 (등가식: §9.1-EVICT (b)).
    /// `importance` Some → score-based(H2O `evict_with_scores`), None → score-free(`evict`).
    ///
    /// **Default = `None`** — 단일 layer-wide keep-list 로 표현 불가한 정책은 미지원을 나타낸다
    /// (per-head H2O+ = head 별 다른 keep, 가중 merge D2O = `EvictionPolicy` 아님; §9.1-EVICT-DEFER).
    /// branch-by-abstraction: in-place `evict*` 와 **공존**하며 그 경로를 refactor 하지 않는다
    /// (production World B 무회귀). 현 단계는 unwired — 정식 게이트는 host unit test (compact_parity).
    fn plan_keep(
        &self,
        current_pos: usize,
        target_len: usize,
        importance: Option<&[f32]>,
    ) -> Option<(Vec<usize>, Vec<Merge>)> {
        let _ = (current_pos, target_len, importance);
        None
    }
}

pub mod h2o;
pub mod h2o_plus;
pub mod method;
pub mod no_eviction;
pub mod sliding_window;
pub mod stage_registry;
pub mod streaming_llm;

#[cfg(test)]
mod compact_parity;

pub use h2o::H2OPolicy;
pub use h2o_plus::H2OPlusPolicy;
pub use method::EvictMethod;
pub use no_eviction::NoEvictionPolicy;
pub use sliding_window::SlidingWindowPolicy;
pub use streaming_llm::StreamingLLMPolicy;
