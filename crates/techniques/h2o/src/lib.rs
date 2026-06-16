//! H2O (Heavy-Hitter Oracle) technique crate — attention-score-based KV eviction (Zhang et al., 2023).
//!
//! Extracted from the engine core into a self-registering technique crate (the `caote`/`quest`
//! precedent): depends only on `argus-extension-api` + `linkme`, implements [`KVCacheStage`], and
//! registers under the name `"h2o"` via `#[distributed_slice(KV_CACHE_STAGES)]`. The engine
//! force-links it (`use h2o as _;`) so `eviction h2o` resolves the out-of-tree plugin.
//!
//! 3-partition model: `[Protected Prefix] [Heavy Hitters (score-ranked)] [Recent Window]`. After
//! reserving the prefix, the remaining budget splits between HH and recent by `keep_ratio`
//! (default 0.5 = the paper's 50:50). Heavy hitters are the highest-cumulative-attention tokens,
//! read from [`StageCtx::importance`]. When no scores are supplied the stage degrades to recency
//! (prefix + most-recent), matching the engine's score-free fallback.
//!
//! The stage only reads `current_pos`/`target_len`/`importance` and returns a layer-wide
//! keep-list; the engine executes the compaction (plan-returning, D1).

use argus_extension_api::{
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageCaps, StageCtx,
    StageParams,
};
use linkme::distributed_slice;

/// H2O eviction stage. `keep_ratio` is clamped to `[0,1]` and `protected_prefix` to ≥4, matching
/// the original engine policy constructor.
struct H2o {
    keep_ratio: f32,
    protected_prefix: usize,
}

impl H2o {
    fn new(keep_ratio: f32, protected_prefix: usize) -> Self {
        Self {
            keep_ratio: keep_ratio.clamp(0.0, 1.0),
            protected_prefix: protected_prefix.max(4),
        }
    }
}

impl KVCacheStage for H2o {
    fn name(&self) -> &str {
        "h2o"
    }

    /// Keep-list, ported verbatim from the engine `H2OPolicy::plan_keep`. `None` = no-op (within
    /// budget, or score-free with nothing to prune) — equivalent to the engine's full-keep plan.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let current = ctx.current_pos();
        let target = ctx.target_len();
        let keep = target.max(self.protected_prefix + 2);
        let prefix = self.protected_prefix;

        if current <= keep {
            return None;
        }

        let keep_list: Vec<usize> = match ctx.importance() {
            // score-free fallback: prefix + most-recent.
            None => {
                let available = keep.saturating_sub(prefix);
                let recent_budget = available;
                let actual_recent = recent_budget.min(current - prefix);
                let prune_count = current - prefix - actual_recent;
                if prune_count == 0 {
                    return None;
                }
                let mut k: Vec<usize> = (0..prefix).collect();
                k.extend((prefix + prune_count)..current);
                k
            }
            // score-based: prefix + heavy hitters (top score) + recent window.
            Some(imp) => {
                let available = keep.saturating_sub(prefix);
                let hh_budget = (available as f32 * self.keep_ratio) as usize;
                let recent_budget = available - hh_budget;
                let actual_recent = recent_budget.min(current - prefix);
                let recent_start = current.saturating_sub(actual_recent).max(prefix);
                let evictable_start = prefix;

                // (pos, score) over evictable range, stable sort desc, take top-hh_budget, re-sort
                // by position — identical token set + order to the engine's evict_with_scores.
                let mut token_scores: Vec<(usize, f32)> = (evictable_start..recent_start)
                    .map(|pos| (pos, imp.get(pos).copied().unwrap_or(0.0)))
                    .collect();
                token_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut hh_positions: Vec<usize> = token_scores
                    .iter()
                    .take(hh_budget)
                    .map(|(pos, _)| *pos)
                    .collect();
                hh_positions.sort();

                let mut k: Vec<usize> = (0..prefix).collect();
                k.extend_from_slice(&hh_positions);
                k.extend(recent_start..current);
                k
            }
        };
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep_list),
            merges: Vec::new(),
        })
    }
}

/// Registration — the engine finds this via `find_stage("h2o")`. `keep_ratio`/`protected_prefix`
/// flow in from [`StageParams`] (CLI `eviction h2o --keep-ratio <R>` + `--protected-prefix <N>`).
#[distributed_slice(KV_CACHE_STAGES)]
static H2O: KVCacheStageReg = KVCacheStageReg {
    name: "h2o",
    make: |p: StageParams| Box::new(H2o::new(p.keep_ratio, p.protected_prefix)),
    // h2o takes no technique-private args — drop the blob, build from StageParams.
    make_with_args: |p: StageParams, _args| Box::new(H2o::new(p.keep_ratio, p.protected_prefix)),
    // H2O selects heavy hitters by accumulated importance (score-based); protect 4 sinks by default.
    caps: StageCaps {
        is_score_based: true,
        default_protected_prefix: 4,
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorHandle, TensorKind, find_stage};

    struct Ctx {
        cur: usize,
        tgt: usize,
        imp: Option<Vec<f32>>,
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
            self.imp.as_deref()
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

    fn keep_of(stage: &dyn KVCacheStage, ctx: &Ctx) -> Option<Vec<usize>> {
        stage.plan(ctx).map(|p| match p.keep {
            KeepSpec::LayerWide(k) => k,
            KeepSpec::PerHead(_) => panic!("h2o is layer-wide"),
        })
    }

    #[test]
    fn registers_into_slice() {
        assert_eq!(find_stage("h2o").expect("h2o registered").name, "h2o");
    }

    #[test]
    fn prefix_clamped_to_four() {
        // protected_prefix < 4 is clamped to 4.
        let stage = H2o::new(0.5, 0);
        // current=20, target=10, prefix=4 → keep=10. score-free: available=6, actual_recent=min(6,16)=6,
        // prune_count = 16-6 = 10 → keep [0..4) ∪ [14..20)
        let keep = keep_of(
            &stage,
            &Ctx {
                cur: 20,
                tgt: 10,
                imp: None,
            },
        );
        assert_eq!(keep, Some(vec![0, 1, 2, 3, 14, 15, 16, 17, 18, 19]));
    }

    #[test]
    fn within_budget_is_noop() {
        let stage = H2o::new(0.5, 4);
        assert_eq!(
            keep_of(
                &stage,
                &Ctx {
                    cur: 5,
                    tgt: 10,
                    imp: None
                }
            ),
            None
        );
    }

    #[test]
    fn score_based_keeps_prefix_heavy_hitters_and_recent() {
        // current=20, target=12, prefix=4, keep_ratio=0.5.
        // keep=12, available=8, hh_budget=4, recent_budget=4, actual_recent=min(4,16)=4,
        // recent_start=16, evictable=[4..16). Heavy hitters by score:
        let stage = H2o::new(0.5, 4);
        let mut imp = vec![0.0f32; 20];
        // make positions 6,9,12,14 the highest scorers in [4..16)
        imp[6] = 10.0;
        imp[9] = 9.0;
        imp[12] = 8.0;
        imp[14] = 7.0;
        let keep = keep_of(
            &stage,
            &Ctx {
                cur: 20,
                tgt: 12,
                imp: Some(imp),
            },
        );
        // keep = [0..4) ∪ {6,9,12,14} ∪ [16..20)
        assert_eq!(keep, Some(vec![0, 1, 2, 3, 6, 9, 12, 14, 16, 17, 18, 19]));
    }

    #[test]
    fn make_from_params() {
        let p = StageParams {
            eviction_window: 0,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 0,
            streaming_window: 0,
        };
        let stage = (find_stage("h2o").unwrap().make)(p);
        assert_eq!(stage.name(), "h2o");
    }
}
