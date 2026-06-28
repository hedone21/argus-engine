//! H2O (Heavy-Hitter Oracle) technique crate — attention-score-based KV eviction (Zhang et al., 2023).
//!
//! Extracted from the engine core into a self-registering technique crate (the `caote`/`quest`
//! precedent): depends only on `argus-extension-api` + `linkme`, implements [`KVMutationStage`], and
//! registers under the name `"h2o"` via `register_kv_mutation_stage!`. The engine
//! force-links it (`use h2o as _;`) so `eviction plugin --name h2o` resolves the out-of-tree plugin.
//!
//! 3-partition model: `[Protected Prefix] [Heavy Hitters (score-ranked)] [Recent Window]`. After
//! reserving the prefix, the remaining budget splits between HH and recent by `keep_ratio`
//! (default 0.5 = the paper's 50:50). Heavy hitters are the highest-cumulative-attention tokens,
//! read from [`StageCtx::importance`]. When no scores are supplied the stage degrades to recency
//! (prefix + most-recent), matching the engine's score-free fallback.
//!
//! The stage only reads `current_pos`/`target_len`/`importance` and stages a layer-wide
//! keep-set on the cache handle (imperative, v3).

use argus_extension_api::{
    CacheHandle, CacheOpError, EstimatorCtx, KVMutationStage, KeepTopK, MutationPhase,
    QCF_ESTIMATORS, QcfEstimator, QcfEstimatorReg, StageArgs, StageCaps, StageCtx, StageParams,
    TensorKind, compile_keep_top_k, redistribute_value, register_kv_mutation_stage,
};
use linkme::distributed_slice;

/// The score-based caps for the v3 registration: H2O reads
/// accumulated importance (Scores) and protects 4 sinks by default.
const H2O_CAPS: StageCaps = StageCaps {
    reads: &[TensorKind::Scores],
    default_protected_prefix: 4,
    produces_merge_plan: false,
};

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

impl H2o {
    /// The keep-list (`None` = no-op within budget / nothing to prune), ported verbatim from the
    /// engine `H2OPolicy::plan_keep`. Drives the v3 `on_phase`. The 3-partition T1 shape is routed
    /// through `compile_keep_top_k` (the policy supplies budgets + a score fn; the compiler owns the
    /// recency window + STABLE top-k + ascending re-sort).
    fn keep_list(
        &self,
        current: usize,
        target: usize,
        importance: Option<&[f32]>,
    ) -> Option<Vec<usize>> {
        let keep = target.max(self.protected_prefix + 2);
        let prefix = self.protected_prefix;
        if current <= keep {
            return None;
        }
        match importance {
            // score-free fallback: prefix + most-recent (heavy 0).
            None => {
                let available = keep.saturating_sub(prefix);
                let actual_recent = available.min(current - prefix);
                if current - prefix - actual_recent == 0 {
                    return None; // nothing to prune — equivalent to the engine's full-keep plan.
                }
                Some(compile_keep_top_k(
                    KeepTopK {
                        current,
                        prefix,
                        recent: actual_recent,
                        heavy: 0,
                    },
                    |_| 0.0,
                ))
            }
            // score-based: prefix + heavy hitters (top score) + recent window.
            Some(imp) => {
                let available = keep.saturating_sub(prefix);
                let hh_budget = (available as f32 * self.keep_ratio) as usize;
                let recent_budget = available - hh_budget;
                let actual_recent = recent_budget.min(current - prefix);
                Some(compile_keep_top_k(
                    KeepTopK {
                        current,
                        prefix,
                        recent: actual_recent,
                        heavy: hh_budget,
                    },
                    |pos| imp.get(pos).copied().unwrap_or(0.0),
                ))
            }
        }
    }
}

// ── v3 native (imperative) surface — the production path ──

impl KVMutationStage for H2o {
    fn name(&self) -> &str {
        "h2o"
    }

    /// Stage the heavy-hitter keep-set (or no-op within budget). The driver supplies accumulated
    /// importance through `ctx.importance()`; the keep-set is decided by the shared `keep_list`.
    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        match self.keep_list(ctx.current_pos(), ctx.target_len(), ctx.importance()) {
            Some(list) => cache.keep(&list),
            None => Ok(()),
        }
    }
}

register_kv_mutation_stage!(
    "h2o",
    |p, _args| Box::new(H2o::new(p.keep_ratio, p.protected_prefix)),
    H2O_CAPS,
    MutationPhase::KvMutate
);

// ── QCF estimator (observer/score axis) ──────────────────────────

/// Identify the H2O-retained token set for the QCF simulation: protected prefix + top-importance
/// heavy hitters (by `keep_ratio`) + most-recent window. Ported verbatim from the engine's former
/// `qcf_kv::identify_retained_h2o` so the estimate is bit-identical to the old engine path.
fn identify_retained_h2o(
    importance: &[f32],
    current_pos: usize,
    target_len: usize,
    keep_ratio: f32,
    protected_prefix: usize,
) -> Vec<usize> {
    let prefix = protected_prefix.min(current_pos).min(target_len);
    let available = target_len.saturating_sub(prefix);
    if available == 0 {
        return (0..prefix).collect();
    }
    let hh_budget = (available as f32 * keep_ratio) as usize;
    let recent_budget = available.saturating_sub(hh_budget);
    let recent_start = current_pos.saturating_sub(recent_budget);
    let mut retained: Vec<usize> = (0..prefix).collect();
    if recent_start > prefix {
        let mut evictable: Vec<(usize, f32)> = (prefix..recent_start)
            .map(|t| {
                (
                    t,
                    if t < importance.len() {
                        importance[t]
                    } else {
                        0.0
                    },
                )
            })
            .collect();
        evictable.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        retained.extend(evictable.iter().take(hh_budget).map(|(t, _)| t));
    }
    retained.extend(recent_start..current_pos);
    retained.sort();
    retained.dedup();
    retained
}

/// H2O QCF estimator: prefix + heavy-hitter + recent retained set, then O_after redistribution over
/// it. Ported verbatim from the engine's former `compute_qcf_kv` `EvictH2o` arm (bit-identical).
struct H2oEstimator {
    keep_ratio: f32,
    protected_prefix: usize,
}

impl QcfEstimator for H2oEstimator {
    fn name(&self) -> &str {
        "h2o"
    }
    fn curve_key(&self) -> &'static str {
        "kv.evict_h2o"
    }
    fn o_after(&self, ctx: &dyn EstimatorCtx, kv_head: usize, out: &mut [f32]) -> bool {
        let current = ctx.current_pos();
        let target = ctx.target_len();
        if current <= target {
            return false;
        }
        let mut alpha = vec![0.0f32; current];
        ctx.alpha_h(kv_head, &mut alpha);
        let retained = identify_retained_h2o(
            &alpha,
            current,
            target,
            self.keep_ratio,
            self.protected_prefix,
        );
        redistribute_value(ctx, kv_head, &alpha, &retained, ctx.beta(), out);
        true
    }
}

/// Registration — found via `find_qcf_estimator("h2o")`. `keep_ratio`/`protected_prefix` flow from
/// the engine-supplied estimate `StageParams` with no actuator-style clamp, to stay bit-identical
/// with the former engine estimate (which used the raw values). Score-based.
#[distributed_slice(QCF_ESTIMATORS)]
static H2O_QCF: QcfEstimatorReg = QcfEstimatorReg {
    name: "h2o",
    curve_key: "kv.evict_h2o",
    make: |p: StageParams, _args: StageArgs<'_>| {
        Box::new(H2oEstimator {
            keep_ratio: p.keep_ratio,
            protected_prefix: p.protected_prefix,
        })
    },
    requires_scores: true,
    requires_streaming_config: false,
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorHandle, TensorKind};

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

    fn keep_of(stage: &H2o, ctx: &Ctx) -> Option<Vec<usize>> {
        stage.keep_list(ctx.current_pos(), ctx.target_len(), ctx.importance())
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

    /// v3 native registration + DECISION equivalence: the v3 `on_phase` stages the keep the shared
    /// `keep_list` returns, for the score-free AND score-based (importance) cases, and within-budget
    /// stages nothing.
    #[test]
    fn v3_native_matches_keep_list_decision() {
        use argus_extension_api::find_mutation_stage;
        let reg = find_mutation_stage("h2o").expect("h2o in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "h2o");
        assert_eq!(reg.phase, MutationPhase::KvMutate);
        assert_eq!(reg.caps, H2O_CAPS);
        assert_eq!((reg.make)(StageParams::default(), &[]).name(), "h2o");

        let s = H2o::new(0.5, 4);
        let imp: Vec<f32> = (0..20).map(|i| (i % 7) as f32).collect();
        let cases = [
            Ctx {
                cur: 8,
                tgt: 0,
                imp: None,
            }, // within budget -> no-op
            Ctx {
                cur: 20,
                tgt: 10,
                imp: None,
            }, // score-free prune
            Ctx {
                cur: 20,
                tgt: 10,
                imp: Some(imp.clone()),
            }, // score-based heavy-hitter
        ];
        for ctx in &cases {
            let mut h = CaptureHandle {
                cur: ctx.cur,
                kept: None,
            };
            <H2o as KVMutationStage>::on_phase(&s, ctx, &mut h).unwrap();
            assert_eq!(h.kept, keep_of(&s, ctx), "cur={} tgt={}", ctx.cur, ctx.tgt);
        }
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
}
