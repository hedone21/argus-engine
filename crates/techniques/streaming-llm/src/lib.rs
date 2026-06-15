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
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageCtx, StageParams,
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
}

impl KVCacheStage for StreamingLlm {
    fn name(&self) -> &str {
        "streaming"
    }

    /// Keep `[0..sink_size) ∪ [recent_start..current)`. When `target_len` is specified and
    /// smaller than `sink + window`, the recent window shrinks to fit (minimum 1). Returns
    /// `None` when already within budget — a no-op, equivalent to the engine's full-keep plan.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let current = ctx.current_pos();
        let target = ctx.target_len();
        let keep = self.sink_size + self.window_size;

        let effective_window = if target > 0 && target < keep {
            target.saturating_sub(self.sink_size).max(1)
        } else {
            self.window_size
        };
        let effective_keep = self.sink_size + effective_window;

        if current <= effective_keep {
            return None; // within budget — no-op
        }

        let recent_start = current - effective_window;
        let mut keep_list: Vec<usize> = (0..self.sink_size).collect();
        keep_list.extend(recent_start..current);
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep_list),
            merges: Vec::new(),
        })
    }
}

/// Registration — the engine finds this entry at construction via `find_stage("streaming")`.
/// `sink_size`/`streaming_window` flow in from [`StageParams`] (CLI `eviction streaming --sink
/// <S> --recent-window <W>`); `streaming_window == 0` is auto-derived upstream before make.
#[distributed_slice(KV_CACHE_STAGES)]
static STREAMING: KVCacheStageReg = KVCacheStageReg {
    name: "streaming",
    make: |p: StageParams| Box::new(StreamingLlm::new(p.sink_size, p.streaming_window)),
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
