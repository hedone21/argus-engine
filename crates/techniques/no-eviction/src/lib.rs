//! No-eviction technique crate — the default `"none"` policy: never evicts.
//!
//! Extracted from the engine core into a self-registering technique crate (the
//! `streaming-llm`/`h2o`/`d2o`/`sliding-window` precedent): depends only on `argus-extension-api` +
//! `linkme`, implements [`KVMutationStage`], and registers under the name `"none"` via
//! `register_kv_mutation_stage!`. The engine force-links it with a one-line
//! `use no_eviction as _;` and finds it via `find_mutation_stage("none")`.
//!
//! `on_phase()` is a no-op: the cache keeps every token and simply grows up to its
//! capacity. This is the exact behavior of the old engine `NoEvictionPolicy` (whose `evict` was a
//! no-op and whose `plan_keep` retained the whole `[0..current)` range — a no-op compaction).

use argus_extension_api::{
    CacheHandle, CacheOpError, KVMutationStage, MutationPhase, StageCtx, register_kv_mutation_stage,
};

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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            (reg.make)(argus_extension_api::StageParams::default(), &[]).name(),
            "none"
        );
    }
}
