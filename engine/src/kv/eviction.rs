use crate::kv::kv_cache::KVCache;
use anyhow::Result;
use argus_extension_api::{KVMutationStage, StageCaps};

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
    /// Override in score-aware policies like heavy-hitter.
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
    /// Override in GQA-aware policies like heavy-hitter+.
    #[allow(clippy::too_many_arguments)]
    fn evict_with_head_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        flat_importance: &[f32],
        _head_importance: &[f32],
        _n_kv_heads: usize,
        _layer_idx: usize,
        _n_layers: usize,
    ) -> Result<()> {
        self.evict_with_scores(cache, target_len, flat_importance)
    }

    /// Per-layer eviction entry. The engine's per-cache loop calls this with the real
    /// `(layer_idx, n_layers)` so per-layer techniques (d2o `protected_layers` / last-layer
    /// protection) know which layer they handle. `importance` is the flat per-token score (or
    /// `None`, score-free); `last_attn` is the optional last-layer last-step per-(kv_head,pos)
    /// attention slice for value-aware techniques (the `a_i` slice), `None` when no AttnWeights
    /// producer is active. Default ignores the layer info + attn slice and dispatches by score
    /// presence — only layer-aware adapters (`StageBackedPolicy`) override it to thread both into
    /// the ctx.
    fn evict_layer(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: Option<&[f32]>,
        last_attn: Option<&[f32]>,
        layer_idx: usize,
        n_layers: usize,
    ) -> Result<()> {
        let _ = (layer_idx, n_layers, last_attn);
        match importance {
            Some(imp) => self.evict_with_scores(cache, target_len, imp),
            None => self.evict(cache, target_len),
        }
    }

    /// If this policy is backed by a WHOLE-MODEL [`KVMutationStage`] (one whose [`StageCaps::whole_model`]
    /// is set), expose the raw stage + its caps so the engine drives it through the cross-layer keepset
    /// path (`run_cross_layer_keepset_eviction`) — one decision over ALL layers — instead of the
    /// per-layer `evict*` loop. `None` (default) ⇒ a per-layer policy → the engine uses the per-layer
    /// loop, byte-identical to before. Only `StageBackedPolicy` overrides it (returning `Some` exactly
    /// when its caps' `whole_model` is set); `StageCaps` is `Copy`, so it is returned by value.
    fn as_whole_model_stage(&self) -> Option<(&dyn KVMutationStage, StageCaps)> {
        None
    }
}

// The score-free / LayerWide eviction policies were extracted to out-of-tree technique crates
// (registers via linkme, force-linked in stage_registry.rs): `streaming` → `streaming-llm`,
// `h2o` → `h2o`, `d2o` → `d2o`, `sliding` → `sliding-window`, `none` → `no-eviction`,
// `rkv` → `rkv` (feature-gated). The engine retains only the generic plumbing here.
pub(crate) mod keepset_dump;
pub mod score_fed;
pub mod stage_registry;
