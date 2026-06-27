//! Example technique crate — proof of "add a folder = zero engine-core edits" + a
//! contributor template.
//!
//! This crate depends only on [`argus_extension_api`], implements [`KVCacheStage`], and registers
//! itself via the `register_kv_stage!` macro (static linkme + cdylib C-ABI dual-wiring). It
//! never references engine types (`KVCache`/`Backend`) — adding a member to the stage axis
//! touches zero code on the other axes (additive extension).
//!
//! **Algorithm**: keep only the most recent `target_len` tokens (the prefix=0 variant of
//! sliding). A pure computation that reads only `current_pos`/`target_len` from [`StageCtx`];
//! the buffer mutation is performed by the engine executor running the returned plan via
//! `compact` (plan-returning). CLI selector: `eviction plugin --name example_keep_recent`.
//!
//! GATE-C: `cargo build -p example-keep-recent --features plugin-cdylib` produces the `.so` →
//! `argus-bench --load-plugin <.so> eviction plugin --name example_keep_recent` loads it zero-compile.

use argus_extension_api::{
    CacheHandle, CacheOpError, KVCachePlan, KVCacheStage, KVMutationStage, KeepSpec, MutationPhase,
    StageCtx, StageParams,
};

/// Stage that keeps only the most recent `target_len` tokens.
struct KeepRecent;

impl KeepRecent {
    /// The keep-list (`None` = no shrink needed), shared by the v3 `on_phase` and the v2 `plan` so
    /// they decide identically: the most-recent `target_len` tokens (ascending).
    fn keep_list(&self, current: usize, target: usize) -> Option<Vec<usize>> {
        if current <= target {
            return None; // no shrink needed — no-op
        }
        Some((current - target..current).collect()) // ascending
    }
}

// ── v3 native (imperative) surface — the canonical contributor TEMPLATE ──
//
// A native v3 technique implements `KVMutationStage` (imperative: stage ops on the transactional
// `CacheHandle`, the engine owns the commit) and registers via `register_kv_mutation_stage!`
// (static-linkme only). `on_phase` reads the pre-callback frame through `&dyn StageCtx` and stages
// its keep through `&mut dyn CacheHandle` — the two views never alias.
impl KVMutationStage for KeepRecent {
    fn name(&self) -> &str {
        "example_keep_recent"
    }

    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        match self.keep_list(ctx.current_pos(), ctx.target_len()) {
            Some(keep) => cache.keep(&keep),
            None => Ok(()),
        }
    }
}

// v3 registration: score-free, fires at the mid-decode KvMutate slot. The engine resolves it via
// `find_mutation_stage("example_keep_recent")`.
argus_extension_api::register_kv_mutation_stage!(
    "example_keep_recent",
    |_p| Box::new(KeepRecent),
    MutationPhase::KvMutate
);

// ── v2 plan-returning surface (kept for the migration window; removed in Phase 2) ──

impl KVCacheStage for KeepRecent {
    fn name(&self) -> &str {
        "example_keep_recent"
    }

    /// Decides via the shared `keep_list`, so it is byte-identical to the v3 `on_phase`.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        self.keep_list(ctx.current_pos(), ctx.target_len())
            .map(|keep| KVCachePlan {
                keep: KeepSpec::LayerWide(keep),
                merges: Vec::new(),
                channels: None,
            })
    }
}

// Registration (dual-wiring) — static: linkme `KV_CACHE_STAGES` (the engine discovers it via
// `find_stage("example_keep_recent")`). Dynamic (`--features plugin-cdylib`): the
// `register_kv_stage_v1` C-ABI export (host dlopens it). One line wires both.
argus_extension_api::register_kv_stage!("example_keep_recent", |_params: StageParams| Box::new(
    KeepRecent
));
// GATE-C v2: emit the `.so` entry (plugin-cdylib gate). A stage-only `.so` → stage 1 +
// format 0 in the dispatcher = graceful wrong-type absorption + a stage plan-identity vehicle.
argus_extension_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::find_stage;

    struct Ctx {
        cur: usize,
        tgt: usize,
    }
    impl StageCtx for Ctx {
        fn current_pos(&self) -> usize {
            self.cur
        }
        fn target_len(&self) -> usize {
            self.tgt
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
            1
        }
        // Only tensor() is implemented — head_score/dequant_* etc. use the argus-extension-api default sugar (None → trivial).
        fn tensor(
            &self,
            _kind: argus_extension_api::TensorKind,
        ) -> Option<&dyn argus_extension_api::TensorHandle> {
            None
        }
    }

    fn params() -> StageParams {
        StageParams {
            eviction_window: 0,
            protected_prefix: 0,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        }
    }

    #[test]
    fn registers_into_slice() {
        let reg = find_stage("example_keep_recent")
            .expect("example stage must be registered in the slice");
        assert_eq!(reg.name, "example_keep_recent");
    }

    /// A mock [`CacheHandle`] capturing the keep staged by `keep`.
    struct CaptureHandle {
        kept: Option<Vec<usize>>,
    }
    impl CacheHandle for CaptureHandle {
        fn current_pos(&self) -> usize {
            100
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            1
        }
        fn kv_on_device(&self) -> bool {
            false
        }
        fn tensor(
            &self,
            _kind: argus_extension_api::TensorKind,
        ) -> Option<&dyn argus_extension_api::TensorHandle> {
            None
        }
        fn keep(&mut self, keep: &[usize]) -> Result<(), CacheOpError> {
            self.kept = Some(keep.to_vec());
            Ok(())
        }
        fn keep_per_head(&mut self, _keep: &[&[usize]]) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn merge(
            &mut self,
            _merges: &[argus_extension_api::WeightedMerge],
        ) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn reencode(&mut self, _target: argus_extension_api::FormatId) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn transition_quant_bits(&mut self, _bits: u8) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn offload(&mut self, _prefix_len: usize) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn recall(&mut self) -> Result<(), CacheOpError> {
            Ok(())
        }
    }

    /// v3 native registration + DECISION equivalence: the v3 `on_phase` stages exactly the keep the v2
    /// `plan` returns (both via the shared `keep_list`). The contributor-template gate.
    #[test]
    fn v3_native_matches_v2_decision() {
        use argus_extension_api::find_mutation_stage;
        let reg =
            find_mutation_stage("example_keep_recent").expect("registered in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "example_keep_recent");
        assert_eq!(reg.phase, MutationPhase::KvMutate);
        assert_eq!(
            (reg.make)(StageParams::default(), &[]).name(),
            "example_keep_recent"
        );

        let mut h = CaptureHandle { kept: None };
        <KeepRecent as KVMutationStage>::on_phase(&KeepRecent, &Ctx { cur: 100, tgt: 30 }, &mut h)
            .unwrap();
        assert_eq!(h.kept, Some((70..100).collect::<Vec<_>>()));
    }

    #[test]
    fn plan_keeps_recent_window() {
        let stage = (find_stage("example_keep_recent").unwrap().make)(params());
        assert_eq!(stage.name(), "example_keep_recent");
        let plan = stage.plan(&Ctx { cur: 100, tgt: 30 }).expect("plan Some");
        match plan.keep {
            KeepSpec::LayerWide(k) => assert_eq!(k, (70..100).collect::<Vec<_>>()),
            KeepSpec::PerHead(_) => panic!("must be LayerWide"),
        }
        assert!(plan.merges.is_empty());
        // current <= target → no-op (None).
        assert!(stage.plan(&Ctx { cur: 20, tgt: 30 }).is_none());
    }
}
