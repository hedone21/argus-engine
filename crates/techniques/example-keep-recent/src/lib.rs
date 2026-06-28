//! Example technique crate — proof of "add a folder = zero engine-core edits" + a
//! contributor template.
//!
//! This crate depends only on [`argus_extension_api`], implements [`KVMutationStage`], and registers
//! itself via the `register_kv_mutation_stage!` macro (static-linkme only). It never references engine
//! types (`KVCache`/`Backend`) — adding a member to the stage axis touches zero code on the other axes
//! (additive extension).
//!
//! **Algorithm**: keep only the most recent `target_len` tokens (the prefix=0 variant of
//! sliding). A pure computation that reads only `current_pos`/`target_len` from [`StageCtx`];
//! the buffer mutation is staged imperatively on the transactional [`CacheHandle`] and the engine owns
//! the commit. CLI selector: `eviction plugin --name example_keep_recent`.

use argus_extension_api::{CacheHandle, CacheOpError, KVMutationStage, MutationPhase, StageCtx};

/// Stage that keeps only the most recent `target_len` tokens.
struct KeepRecent;

impl KeepRecent {
    /// The keep-list (`None` = no shrink needed): the most-recent `target_len` tokens (ascending).
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

// GATE-C: emit the `.so` entry symbols (plugin-cdylib gate). This stage-only crate contributes no
// dynamic format/backend-cap axis, so its `.so` is capability-0 (stages are static-linkme only) — the
// macro stays so the crate keeps building under `--features plugin-cdylib`.
argus_extension_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::StageParams;

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

    /// v3 native registration + the `on_phase` decision: the stage stages exactly the keep its shared
    /// `keep_list` computes (the most-recent window). The contributor-template gate.
    #[test]
    fn v3_native_keeps_recent_window() {
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

        // current <= target → no-op (the handle stages nothing).
        let mut h2 = CaptureHandle { kept: None };
        <KeepRecent as KVMutationStage>::on_phase(&KeepRecent, &Ctx { cur: 20, tgt: 30 }, &mut h2)
            .unwrap();
        assert_eq!(h2.kept, None);
    }
}
