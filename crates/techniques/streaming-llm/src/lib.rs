//! StreamingLLM technique crate — attention-sink + recent-window KV eviction (Xiao et al., ICLR 2024).
//!
//! Extracted from the engine core into a self-registering technique crate (the `caote`/`quest`
//! precedent): depends only on `argus-extension-api` + `linkme`, implements [`KVCacheStage`], and
//! registers under the name `"streaming"` via `#[distributed_slice(KV_CACHE_STAGES)]`. The engine
//! force-links it with a one-line `use streaming_llm as _;` and finds it via `find_stage("streaming")`.
//!
//! Maintains a fixed two-region structure:
//! ```text
//! [Sink Tokens (S)] [Recent Window (W)]
//! ```
//! When `target_len` is 0 or ≥ `S + W`, keeps exactly `S + W` tokens. When `target_len` is
//! specified and < `S + W`, the recent window shrinks to fit the budget (minimum 1 token).
//!
//! The stage is pure position arithmetic — it reads only `current_pos`/`target_len` from
//! [`StageCtx`] and returns a layer-wide keep-list; the engine executes the compaction
//! (plan-returning, D1). It never references engine types (`KVCache`/`Backend`).

use argus_extension_api::{
    CacheHandle, CacheOpError, EstimatorCtx, KV_CACHE_STAGES, KVCachePlan, KVCacheStage,
    KVCacheStageReg, KVMutationStage, KeepSpec, KeepTopK, MutationPhase, QCF_ESTIMATORS,
    QcfEstimator, QcfEstimatorReg, StageArgs, StageCaps, StageCtx, StageParams, compile_keep_top_k,
    redistribute_value, register_kv_mutation_stage,
};
use linkme::distributed_slice;

/// StreamingLLM eviction stage. `sink_size`/`window_size` are clamped to ≥1, matching the
/// original engine policy constructor.
struct StreamingLlm {
    sink_size: usize,
    window_size: usize,
}

impl StreamingLlm {
    fn new(sink_size: usize, window_size: usize) -> Self {
        Self {
            sink_size: sink_size.max(1),
            window_size: window_size.max(1),
        }
    }

    /// The 3-partition shape (sink prefix + recent window, score-free) to keep, or `None` when already
    /// within budget (a no-op). Shared by the v3 `on_phase` and the v2 `plan` so they decide
    /// identically — when `target_len` is specified and smaller than `sink + window`, the recent
    /// window shrinks to fit (minimum 1).
    fn keep_spec(&self, current: usize, target: usize) -> Option<KeepTopK> {
        let keep = self.sink_size + self.window_size;
        let effective_window = if target > 0 && target < keep {
            target.saturating_sub(self.sink_size).max(1)
        } else {
            self.window_size
        };
        if current <= self.sink_size + effective_window {
            return None; // within budget — no-op
        }
        Some(KeepTopK {
            current,
            prefix: self.sink_size,
            recent: effective_window,
            heavy: 0,
        })
    }
}

// ── v3 native (imperative) surface — the production path ──

impl KVMutationStage for StreamingLlm {
    fn name(&self) -> &str {
        "streaming"
    }

    /// Keep `[0..sink) ∪ [recent_start..current)` via the T1 compiler (score-free), or no-op when
    /// within budget. Byte-identical to the v2 plan (both route through the same `keep_spec`).
    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        match self.keep_spec(ctx.current_pos(), ctx.target_len()) {
            Some(spec) => cache.keep_top_k(spec, &|_| 0.0),
            None => Ok(()),
        }
    }
}

register_kv_mutation_stage!(
    "streaming",
    |p| Box::new(StreamingLlm::new(p.sink_size, p.streaming_window)),
    MutationPhase::KvMutate
);

// ── v2 plan-returning surface (kept for the migration window; removed in Phase 2) ──

impl KVCacheStage for StreamingLlm {
    fn name(&self) -> &str {
        "streaming"
    }

    /// Keep `[0..sink_size) ∪ [recent_start..current)`. Returns `None` when already within budget.
    /// Decides via the shared `keep_spec`, so it is byte-identical to the v3 `on_phase`.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let spec = self.keep_spec(ctx.current_pos(), ctx.target_len())?;
        // prefix(sink) + recent window, score-free (heavy 0) — routed through the T1 compiler.
        let keep_list = compile_keep_top_k(spec, |_| 0.0);
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep_list),
            merges: Vec::new(),
            channels: None,
        })
    }
}

/// Registration — the engine finds this entry at construction via `find_stage("streaming")`.
/// `sink_size`/`streaming_window` flow in from [`StageParams`] (CLI `eviction plugin --name
/// streaming --set sink=<S> --set recent_window=<W>`); `streaming_window == 0` is auto-derived
/// upstream before make.
#[distributed_slice(KV_CACHE_STAGES)]
static STREAMING: KVCacheStageReg = KVCacheStageReg {
    name: "streaming",
    make: |p: StageParams| Box::new(StreamingLlm::new(p.sink_size, p.streaming_window)),
    // streaming takes no technique-private args — drop the blob, build from StageParams.
    make_with_args: |p: StageParams, _args| {
        Box::new(StreamingLlm::new(p.sink_size, p.streaming_window))
    },
    // StreamingLLM is score-free (sink + recent window); the engine picks the protected-prefix
    // fallback (it derives the sink itself), so declare no stage default.
    caps: StageCaps::SCORE_FREE,
};

// ── QCF estimator (observer/score axis) ──────────────────────────

/// StreamingLLM QCF estimator: retains sink + recent window, evicts the middle, then rebuilds O_after
/// over the retained set. Ported verbatim from the engine's former `compute_qcf_kv` `EvictStreaming`
/// arm (bit-identical). `sink_size`/`window_size` come from the engine-supplied estimate config.
struct StreamingEstimator {
    sink_size: usize,
    window_size: usize,
}

impl QcfEstimator for StreamingEstimator {
    fn name(&self) -> &str {
        "streaming"
    }
    fn curve_key(&self) -> &'static str {
        "kv.evict_streaming"
    }
    fn o_after(&self, ctx: &dyn EstimatorCtx, kv_head: usize, out: &mut [f32]) -> bool {
        let current = ctx.current_pos();
        let keep_size = self.sink_size + self.window_size;
        if current <= keep_size {
            return false;
        }
        // Guard (current > sink+window) keeps the two ranges disjoint — no double-counting.
        let retained: Vec<usize> = (0..self.sink_size)
            .chain((current - self.window_size)..current)
            .collect();
        let mut alpha = vec![0.0f32; current];
        ctx.alpha_h(kv_head, &mut alpha);
        redistribute_value(ctx, kv_head, &alpha, &retained, ctx.beta(), out);
        true
    }
}

/// Registration — found via `find_qcf_estimator("streaming")`. `sink_size`/`streaming_window` flow
/// from the engine-supplied estimate `StageParams`. Score-free, but needs an engine-supplied
/// `(sink, window)` config, so the QCF driver skips it when none is present
/// (`requires_streaming_config`) — matching the engine's former `streaming_config.is_some()` gate.
#[distributed_slice(QCF_ESTIMATORS)]
static STREAMING_QCF: QcfEstimatorReg = QcfEstimatorReg {
    name: "streaming",
    curve_key: "kv.evict_streaming",
    make: |p: StageParams, _args: StageArgs<'_>| {
        Box::new(StreamingEstimator {
            sink_size: p.sink_size,
            window_size: p.streaming_window,
        })
    },
    requires_scores: false,
    requires_streaming_config: true,
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorHandle, TensorKind, find_stage};

    /// Minimal StageCtx — StreamingLLM only reads `current_pos`/`target_len`.
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

    fn keep_of(stage: &dyn KVCacheStage, cur: usize, tgt: usize) -> Option<Vec<usize>> {
        stage.plan(&Ctx { cur, tgt }).map(|p| match p.keep {
            KeepSpec::LayerWide(k) => k,
            KeepSpec::PerHead(_) => panic!("streaming is layer-wide"),
        })
    }

    #[test]
    fn registers_into_slice() {
        let reg = find_stage("streaming").expect("streaming registered in KV_CACHE_STAGES");
        assert_eq!(reg.name, "streaming");
    }

    /// A mock [`CacheHandle`] that records the keep-set staged by `keep` (the default `keep_top_k`
    /// routes through it). Only the read scalars the stage uses + `keep` are meaningful.
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

    /// v3 native registration + DECISION equivalence: the v3 `on_phase` stages exactly the same
    /// keep-set the v2 `plan` returns (both route through `keep_spec` → `compile_keep_top_k`), and a
    /// within-budget call stages nothing. (The engine-side byte-identical compaction is covered by the
    /// driver tests + the naive-reference oracle.)
    #[test]
    fn v3_native_matches_v2_decision() {
        use argus_extension_api::find_mutation_stage;
        let reg = find_mutation_stage("streaming").expect("streaming in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "streaming");
        assert_eq!(reg.phase, MutationPhase::KvMutate);
        let stage = (reg.make)(StageParams::default(), &[]);
        assert_eq!(stage.name(), "streaming");

        let s = StreamingLlm::new(4, 6);
        for (cur, tgt) in [
            (15usize, 0usize),
            (11, 0),
            (20, 7),
            (20, 2),
            (8, 0),
            (10, 0),
        ] {
            let mut h = CaptureHandle { cur, kept: None };
            s.on_phase(&Ctx { cur, tgt }, &mut h).unwrap();
            // v3-staged keep == v2 plan keep (None ⇒ nothing staged).
            assert_eq!(h.kept, keep_of(&s, cur, tgt), "cur={cur} tgt={tgt}");
        }
    }

    // ── keep-list spec, ported verbatim from the original engine streaming_llm.rs unit tests ──

    #[test]
    fn keep_list_basic() {
        // sink=4, window=6, current=15, target=0 → keep [0,1,2,3] ∪ [9..15)
        let stage = StreamingLlm::new(4, 6);
        assert_eq!(
            keep_of(&stage, 15, 0),
            Some(vec![0, 1, 2, 3, 9, 10, 11, 12, 13, 14])
        );
    }

    #[test]
    fn within_budget_is_noop() {
        let stage = StreamingLlm::new(4, 6); // keep = 10
        assert_eq!(keep_of(&stage, 8, 0), None);
        assert_eq!(keep_of(&stage, 10, 0), None); // exactly at budget
    }

    #[test]
    fn one_over_budget_drops_one() {
        // current=11 → keep [0..4) ∪ [5..11)
        let stage = StreamingLlm::new(4, 6);
        assert_eq!(
            keep_of(&stage, 11, 0),
            Some(vec![0, 1, 2, 3, 5, 6, 7, 8, 9, 10])
        );
    }

    #[test]
    fn target_shrinks_window() {
        // current=20, target=7, sink=4 → eff_window=3 → keep [0..4) ∪ [17..20)
        let stage = StreamingLlm::new(4, 6);
        assert_eq!(keep_of(&stage, 20, 7), Some(vec![0, 1, 2, 3, 17, 18, 19]));
    }

    #[test]
    fn target_below_sink_floors_window_to_one() {
        // current=20, target=2 < sink=4 → eff_window=max(2-4,1)=1 → keep [0..4) ∪ [19..20)
        let stage = StreamingLlm::new(4, 6);
        assert_eq!(keep_of(&stage, 20, 2), Some(vec![0, 1, 2, 3, 19]));
    }

    #[test]
    fn make_from_params_uses_sink_and_window() {
        let p = StageParams {
            eviction_window: 0,
            protected_prefix: 0,
            keep_ratio: 0.0,
            sink_size: 4,
            streaming_window: 6,
        };
        let stage = (find_stage("streaming").unwrap().make)(p);
        assert_eq!(stage.name(), "streaming");
        let keep = stage
            .plan(&Ctx { cur: 15, tgt: 0 })
            .map(|pl| match pl.keep {
                KeepSpec::LayerWide(k) => k,
                KeepSpec::PerHead(_) => panic!("layer-wide"),
            });
        assert_eq!(keep, Some(vec![0, 1, 2, 3, 9, 10, 11, 12, 13, 14]));
    }
}
