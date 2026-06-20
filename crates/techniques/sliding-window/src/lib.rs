//! Sliding-window technique crate — recent-window KV eviction with a protected prefix.
//!
//! Extracted from the engine core into a self-registering technique crate (the
//! `streaming-llm`/`h2o`/`d2o` precedent): depends only on `argus-extension-api` + `linkme`,
//! implements [`KVCacheStage`], and registers under the name `"sliding"` via
//! `#[distributed_slice(KV_CACHE_STAGES)]`. The engine force-links it with a one-line
//! `use sliding_window as _;` and finds it via `find_stage("sliding")`.
//!
//! Keeps the most recent `window_size` tokens plus a protected prefix of `protected_prefix` tokens
//! (e.g. system prompt / attention sinks):
//! ```text
//! [Protected Prefix] [ ... evicted ... ] [Recent Window]
//! ```
//! The stage is pure position arithmetic — it reads only `current_pos`/`target_len` from
//! [`StageCtx`] and returns a layer-wide keep-list; the engine executes the compaction
//! (plan-returning, D1). It never references engine types (`KVCache`/`Backend`).
//!
//! Ported verbatim from the original engine `SlidingWindowPolicy::plan_keep` (the keep-list that
//! `compact_parity` proved bit-identical to its in-place `evict`), so the World-B application is
//! unchanged.

use argus_extension_api::{
    EstimatorCtx, KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec,
    QCF_ESTIMATORS, QcfEstimator, QcfEstimatorReg, StageArgs, StageCaps, StageCtx, StageParams,
    redistribute_value,
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
}

impl KVCacheStage for SlidingWindow {
    fn name(&self) -> &str {
        "sliding"
    }

    /// Keep `[0..protected_prefix) ∪ [protected_prefix + prune_count..current)`, ascending and
    /// prefix-inclusive. The retained count is `target_len` clamped to `[protected_prefix + 16,
    /// window_size + protected_prefix]`. When already within budget the full range is kept (a
    /// no-op compaction). This is the verbatim port of the old `SlidingWindowPolicy::plan_keep`.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let current = ctx.current_pos();
        let max_keep = self.window_size + self.protected_prefix;
        let min_keep = (self.protected_prefix + 16).min(max_keep);
        let keep = ctx.target_len().clamp(min_keep, max_keep);

        let keep_list: Vec<usize> = if current <= keep {
            (0..current).collect()
        } else {
            let removable_count = current - self.protected_prefix;
            let tokens_to_keep_after_prefix = keep.saturating_sub(self.protected_prefix);
            if tokens_to_keep_after_prefix >= removable_count {
                (0..current).collect()
            } else {
                let prune_count = removable_count - tokens_to_keep_after_prefix;
                let mut k: Vec<usize> = (0..self.protected_prefix).collect();
                k.extend((self.protected_prefix + prune_count)..current);
                k
            }
        };
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep_list),
            merges: Vec::new(),
        })
    }
}

/// Registration — the engine finds this entry at construction via `find_stage("sliding")`.
/// `eviction_window`/`protected_prefix` flow in from [`StageParams`] (CLI `eviction plugin --name
/// sliding --set window=<N>` + `--protected-prefix`).
#[distributed_slice(KV_CACHE_STAGES)]
static SLIDING: KVCacheStageReg = KVCacheStageReg {
    name: "sliding",
    make: |p: StageParams| Box::new(SlidingWindow::new(p.eviction_window, p.protected_prefix)),
    // sliding takes no technique-private args — drop the blob, build from StageParams.
    make_with_args: |p: StageParams, _args| {
        Box::new(SlidingWindow::new(p.eviction_window, p.protected_prefix))
    },
    // Sliding is score-free (recency only); the constructor clamps the prefix to a 4-sink minimum,
    // so declare no stage default and let the engine pick the fallback.
    caps: StageCaps::SCORE_FREE,
};

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
    use argus_extension_api::{TensorHandle, TensorKind, find_stage};

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

    fn keep_of(stage: &dyn KVCacheStage, cur: usize, tgt: usize) -> Vec<usize> {
        match stage
            .plan(&Ctx { cur, tgt })
            .expect("sliding always returns a plan")
            .keep
        {
            KeepSpec::LayerWide(k) => k,
            KeepSpec::PerHead(_) => panic!("sliding is layer-wide"),
        }
    }

    #[test]
    fn registers_into_slice() {
        let reg = find_stage("sliding").expect("sliding registered in KV_CACHE_STAGES");
        assert_eq!(reg.name, "sliding");
        assert!(reg.caps.reads.is_empty(), "sliding is score-free");
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

    #[test]
    fn make_from_params() {
        let p = StageParams {
            eviction_window: 10,
            protected_prefix: 4,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        };
        let stage = (find_stage("sliding").unwrap().make)(p);
        assert_eq!(stage.name(), "sliding");
    }
}
