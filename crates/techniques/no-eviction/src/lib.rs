//! No-eviction technique crate — the default `"none"` policy: never evicts.
//!
//! Extracted from the engine core into a self-registering technique crate (the
//! `streaming-llm`/`h2o`/`d2o`/`sliding-window` precedent): depends only on `argus-extension-api` +
//! `linkme`, implements [`KVCacheStage`], and registers under the name `"none"` via
//! `#[distributed_slice(KV_CACHE_STAGES)]`. The engine force-links it with a one-line
//! `use no_eviction as _;` and finds it via `find_stage("none")`.
//!
//! `plan()` returns `None` (no-op): the cache keeps every token and simply grows up to its
//! capacity. This is the exact behavior of the old engine `NoEvictionPolicy` (whose `evict` was a
//! no-op and whose `plan_keep` retained the whole `[0..current)` range — a no-op compaction).

use argus_extension_api::{
    CacheHandle, CacheOpError, KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg,
    KVMutationStage, MutationPhase, StageCaps, StageCtx, StageParams, register_kv_mutation_stage,
};
use linkme::distributed_slice;

/// The no-eviction stage. Stateless — never mutates the cache.
struct NoEviction;

// ── v3 native (imperative) surface — the production path ──

impl KVMutationStage for NoEviction {
    fn name(&self) -> &str {
        "none"
    }

    /// Never evicts — stages nothing, so the transaction commits as a no-op and the cache is
    /// untouched (the exact behavior of the old `NoEvictionPolicy`).
    fn on_phase(
        &self,
        _ctx: &dyn StageCtx,
        _cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        Ok(())
    }
}

// v3 registration: the engine resolves this via `find_mutation_stage("none")`. Score-free, KvMutate.
register_kv_mutation_stage!("none", |_p| Box::new(NoEviction), MutationPhase::KvMutate);

// ── v2 plan-returning surface (kept for the migration window; removed in Phase 2) ──

impl KVCacheStage for NoEviction {
    fn name(&self) -> &str {
        "none"
    }

    /// Never evicts — returns `None` (no plan applied), so the engine leaves the cache untouched.
    fn plan(&self, _ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        None
    }
}

/// v2 registration — the engine finds this entry via `find_stage("none")`. It takes no parameters.
#[distributed_slice(KV_CACHE_STAGES)]
static NONE: KVCacheStageReg = KVCacheStageReg {
    name: "none",
    make: |_p: StageParams| Box::new(NoEviction),
    make_with_args: |_p: StageParams, _args| Box::new(NoEviction),
    // No eviction → no scores, no protected-prefix default.
    caps: StageCaps::SCORE_FREE,
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorHandle, TensorKind, find_stage};

    struct Ctx;
    impl StageCtx for Ctx {
        fn current_pos(&self) -> usize {
            100
        }
        fn target_len(&self) -> usize {
            10
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn importance(&self) -> Option<&[f32]> {
            None
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            4
        }
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
    }

    #[test]
    fn registers_into_slice_and_is_noop() {
        let reg = find_stage("none").expect("none registered in KV_CACHE_STAGES");
        assert_eq!(reg.name, "none");
        assert!(reg.caps.reads.is_empty());
        let stage = (reg.make)(StageParams::default());
        assert_eq!(stage.name(), "none");
        // Even far over target, the plan is a no-op (None) — the cache is never pruned.
        assert!(stage.plan(&Ctx).is_none());
    }

    /// v3 native registration: "none" resolves in KV_MUTATION_STAGES (score-free, KvMutate) and the
    /// made stage is a no-op (the byte-identical no-op-vs-naive-reference gate runs engine-side, where
    /// the driver + oracle live).
    #[test]
    fn registers_into_mutation_slice() {
        use argus_extension_api::{find_mutation_stage, mutation_stage_caps};
        let reg = find_mutation_stage("none").expect("none registered in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "none");
        assert_eq!(reg.phase, MutationPhase::KvMutate);
        assert!(reg.caps.reads.is_empty());
        assert!(mutation_stage_caps("none").is_some());
        assert_eq!((reg.make)(StageParams::default(), &[]).name(), "none");
    }
}
