//! Example technique crate — proof of "add a folder = zero engine-core edits" + a
//! contributor template.
//!
//! This crate depends only on [`technique_api`], implements [`KVCacheStage`], and registers
//! itself via the `register_kv_stage!` macro (static linkme + cdylib C-ABI dual-wiring). It
//! never references engine types (`KVCache`/`Backend`) — adding a member to the stage axis
//! touches zero code on the other axes (additive extension).
//!
//! **Algorithm**: keep only the most recent `target_len` tokens (the prefix=0 variant of
//! sliding). A pure computation that reads only `current_pos`/`target_len` from [`StageCtx`];
//! the buffer mutation is performed by the engine executor running the returned plan via
//! `compact` (plan-returning). CLI selector: `--eviction-policy example_keep_recent`.
//!
//! GATE-C: `cargo build -p example-keep-recent --features plugin-cdylib` produces the `.so` →
//! `argus_bench --load-plugin <.so> --eviction-policy example_keep_recent` loads it zero-compile.

use technique_api::{KVCachePlan, KVCacheStage, KeepSpec, StageCtx, StageParams};

/// Stage that keeps only the most recent `target_len` tokens.
struct KeepRecent;

impl KVCacheStage for KeepRecent {
    fn name(&self) -> &str {
        "example_keep_recent"
    }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let current = ctx.current_pos();
        let target = ctx.target_len();
        if current <= target {
            return None; // no shrink needed — no-op
        }
        let keep: Vec<usize> = (current - target..current).collect(); // ascending
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
            merges: Vec::new(),
        })
    }
}

// Registration (dual-wiring) — static: linkme `KV_CACHE_STAGES` (the engine discovers it via
// `find_stage("example_keep_recent")`). Dynamic (`--features plugin-cdylib`): the
// `register_kv_stage_v1` C-ABI export (host dlopens it). One line wires both.
technique_api::register_kv_stage!("example_keep_recent", |_params: StageParams| Box::new(
    KeepRecent
));
// GATE-C v2: emit the `.so` entry (plugin-cdylib gate). A stage-only `.so` → stage 1 +
// format 0 in the dispatcher = graceful wrong-type absorption + a stage plan-identity vehicle.
technique_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;
    use technique_api::find_stage;

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
        // Only tensor() is implemented — head_score/dequant_* etc. use the technique-api default sugar (None → trivial).
        fn tensor(
            &self,
            _kind: technique_api::TensorKind,
        ) -> Option<&dyn technique_api::TensorHandle> {
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
