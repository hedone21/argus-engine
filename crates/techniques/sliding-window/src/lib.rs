//! Sliding-window technique crate — recent-window KV eviction with a protected prefix.
//!
//! Extracted from the engine core into a self-registering technique crate (the
//! `streaming-llm`/`h2o`/`d2o` precedent): depends only on `argus-extension-api` + `linkme`,
//! implements [`KVMutationStage`], and registers under the name `"sliding"` via
//! `register_kv_mutation_stage!`. The engine force-links it with a one-line
//! `use sliding_window as _;` and finds it via `find_mutation_stage("sliding")`.
//!
//! Keeps the most recent `window_size` tokens plus a protected prefix of `protected_prefix` tokens
//! (e.g. system prompt / attention sinks):
//! ```text
//! [Protected Prefix] [ ... evicted ... ] [Recent Window]
//! ```
//! The stage is pure position arithmetic — it reads only `current_pos`/`target_len` from
//! [`StageCtx`] and stages a layer-wide keep-list on the [`CacheHandle`], which executes the
//! compaction. It never references engine types (`KVCache`/`Backend`).
//!
//! Ported verbatim from the original engine `SlidingWindowPolicy::plan_keep` (the keep-list that
//! `compact_parity` proved bit-identical to its in-place `evict`), so the World-B application is
//! unchanged.

use argus_extension_api::{
    CacheHandle, CacheOpError, EstimatorCtx, KVMutationStage, KeepTopK, MutationPhase,
    QCF_ESTIMATORS, QcfEstimator, QcfEstimatorReg, StageArgs, StageCtx, StageParams,
    compile_keep_top_k, redistribute_value, register_kv_mutation_stage,
};
use linkme::distributed_slice;

/// Sliding-window eviction stage. `protected_prefix` is clamped to a 4-token minimum (attention
/// sink), matching the original engine policy constructor.
struct SlidingWindow {
    window_size: usize,
    protected_prefix: usize,
}

impl SlidingWindow {
    fn new(window_size: usize, protected_prefix: usize) -> Self {
        Self {
            window_size,
            // Enforce a minimum protected prefix of 4 to act as an attention sink (matches the
            // original `SlidingWindowPolicy::new`).
            protected_prefix: protected_prefix.max(4),
        }
    }

    /// The keep-list (ascending, prefix-inclusive): `[0..protected_prefix) ∪ [recent_start..current)`,
    /// or the full range `[0..current)` when within budget. The retained count is `target_len` clamped
    /// to `[protected_prefix + 16, window_size + protected_prefix]`. Drives the v3 `on_phase` — the
    /// verbatim port of `SlidingWindowPolicy::plan_keep`.
    fn keep_list(&self, current: usize, target: usize) -> Vec<usize> {
        let max_keep = self.window_size + self.protected_prefix;
        let min_keep = (self.protected_prefix + 16).min(max_keep);
        let keep = target.clamp(min_keep, max_keep);
        if current <= keep {
            return (0..current).collect();
        }
        let removable_count = current - self.protected_prefix;
        let tokens_to_keep_after_prefix = keep.saturating_sub(self.protected_prefix);
        if tokens_to_keep_after_prefix >= removable_count {
            return (0..current).collect();
        }
        // prefix + recent window, score-free (heavy 0) — routed through the T1 compiler.
        compile_keep_top_k(
            KeepTopK {
                current,
                prefix: self.protected_prefix,
                recent: tokens_to_keep_after_prefix,
                heavy: 0,
            },
            |_| 0.0,
        )
    }
}

// ── v3 native (imperative) surface — the production path ──

impl KVMutationStage for SlidingWindow {
    fn name(&self) -> &str {
        "sliding"
    }

    /// Stage the keep-list (always — a full-keep no-op compaction still records the keep-set dump,
    /// matching the v2 path, which always returns a plan). Byte-identical to the v2 plan via the
    /// shared `keep_list`.
    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        cache.keep(&self.keep_list(ctx.current_pos(), ctx.target_len()))
    }
}

register_kv_mutation_stage!(
    "sliding",
    |p| Box::new(SlidingWindow::new(p.eviction_window, p.protected_prefix)),
    MutationPhase::KvMutate
);

// ── QCF estimator (observer/score axis) ──────────────────────────

/// Sliding-window QCF estimator: simulates dropping all but the most-recent `target_len` tokens and
/// rebuilds the per-head attention output O_after over that retained window. Ported verbatim from the
/// engine's former `compute_qcf_kv` `EvictSliding` arm (bit-identical), now owned by this crate.
struct SlidingEstimator;

impl QcfEstimator for SlidingEstimator {
    fn name(&self) -> &str {
        "sliding"
    }
    fn curve_key(&self) -> &'static str {
        "kv.evict_sliding"
    }
    fn o_after(&self, ctx: &dyn EstimatorCtx, kv_head: usize, out: &mut [f32]) -> bool {
        let current = ctx.current_pos();
        let target = ctx.target_len();
        if current <= target {
            return false; // within budget — no eviction
        }
        let retained_start = current - target;
        let retained: Vec<usize> = (retained_start..current).collect();
        let mut alpha = vec![0.0f32; current];
        ctx.alpha_h(kv_head, &mut alpha);
        redistribute_value(ctx, kv_head, &alpha, &retained, ctx.beta(), out);
        true
    }
}

/// Registration — the engine's QCF runtime finds this via `find_qcf_estimator("sliding")`. The
/// estimate uses the engine-derived `target_len` (current_pos based), not the actuator's
/// `eviction_window`, so the estimator carries no config. Score-free; needs no streaming config.
#[distributed_slice(QCF_ESTIMATORS)]
static SLIDING_QCF: QcfEstimatorReg = QcfEstimatorReg {
    name: "sliding",
    curve_key: "kv.evict_sliding",
    make: |_p: StageParams, _args: StageArgs<'_>| Box::new(SlidingEstimator),
    requires_scores: false,
    requires_streaming_config: false,
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorHandle, TensorKind};

    /// Minimal StageCtx — SlidingWindow only reads `current_pos`/`target_len`.
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
            4
        }
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
    }

    fn keep_of(stage: &SlidingWindow, cur: usize, tgt: usize) -> Vec<usize> {
        stage.keep_list(cur, tgt)
    }

    /// A mock [`CacheHandle`] capturing the keep staged by `keep`.
    struct CaptureHandle {
        cur: usize,
        kept: Option<Vec<usize>>,
    }
    impl CacheHandle for CaptureHandle {
        fn current_pos(&self) -> usize {
            self.cur
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            4
        }
        fn kv_on_device(&self) -> bool {
            false
        }
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
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

    /// v3 native registration + DECISION equivalence: the v3 `on_phase` stages exactly the keep-list
    /// the shared `keep_list` produces, including the full-keep no-op case (sliding always stages a
    /// keep, so the keep-set dump is recorded even when nothing is evicted).
    #[test]
    fn v3_native_matches_v2_decision() {
        use argus_extension_api::find_mutation_stage;
        let reg = find_mutation_stage("sliding").expect("sliding in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "sliding");
        assert_eq!(reg.phase, MutationPhase::KvMutate);
        assert_eq!((reg.make)(StageParams::default(), &[]).name(), "sliding");

        let s = SlidingWindow::new(10, 4);
        for (cur, tgt) in [(8usize, 0usize), (30, 0), (50, 20), (14, 14), (100, 5)] {
            let mut h = CaptureHandle { cur, kept: None };
            s.on_phase(&Ctx { cur, tgt }, &mut h).unwrap();
            assert_eq!(h.kept, Some(keep_of(&s, cur, tgt)), "cur={cur} tgt={tgt}");
        }
    }

    #[test]
    fn min_prefix_clamped_to_four() {
        // prefix 0 → clamped to 4. window=10 → max_keep=14, min_keep=min(20,14)=14, keep=14.
        // current=40, target=20(clamped to 14). prune_count = (40-4) - (14-4) = 26.
        // keep = [0..4) ∪ [30..40).
        let stage = SlidingWindow::new(10, 0);
        let mut expected: Vec<usize> = (0..4).collect();
        expected.extend(30..40);
        assert_eq!(keep_of(&stage, 40, 20), expected);
    }

    #[test]
    fn within_budget_keeps_all() {
        // current <= keep → full range retained (no-op compaction).
        let stage = SlidingWindow::new(64, 4);
        assert_eq!(keep_of(&stage, 20, 60), (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn protected_prefix_preserved() {
        // window=4, prefix=4 → max_keep=8, min_keep=min(20,8)=8, keep=8.
        // current=12 → removable=8, keep_after_prefix=4, prune_count=4 → keep [0..4) ∪ [8..12).
        let stage = SlidingWindow::new(4, 4);
        assert_eq!(keep_of(&stage, 12, 6), vec![0, 1, 2, 3, 8, 9, 10, 11]);
    }
}
