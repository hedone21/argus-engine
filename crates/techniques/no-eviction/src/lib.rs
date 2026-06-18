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
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, StageCaps, StageCtx, StageParams,
};
use linkme::distributed_slice;

/// The no-eviction stage. Stateless — `plan()` is always a no-op.
struct NoEviction;

impl KVCacheStage for NoEviction {
    fn name(&self) -> &str {
        "none"
    }

    /// Never evicts — returns `None` (no plan applied), so the engine leaves the cache untouched.
    fn plan(&self, _ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        None
    }
}

/// Registration — the engine finds this entry via `find_stage("none")`. It takes no parameters.
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
}
