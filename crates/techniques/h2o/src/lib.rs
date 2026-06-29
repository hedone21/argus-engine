//! H2O (Heavy-Hitter Oracle) technique crate — attention-score-based KV eviction (Zhang et al., 2023).
//!
//! Faithful to the reference `H2OKVCache_LayerWise` (FMInference/H2O): the cache keeps a FIXED budget
//! of `hh_size` heavy hitters (highest cumulative attention) + `recent_size` most-recent tokens, with
//! NO attention-sink prefix by default (`default_protected_prefix = 0`). The budgets are ABSOLUTE
//! (`--set hh_size=… --set recent_size=…`), not a ratio of an engine-supplied `target_len` — the
//! budget IS the policy, so the CLI requires both explicitly (`Args::require_h2o_budgets`).
//!
//! Per-head when the engine supplies per-(kv_head, pos) scores via `ctx.tensor(Scores)` (each KV head
//! ranks its own heavy hitters, all heads keeping the same count so the single `current_pos`
//! invariant holds — the original H2O granularity); otherwise the flat layer-wide importance, and
//! score-free to recency.
//!
//! The faithful prefill-attention seed for the importance score (divergence `(c)`) is wired in the
//! engine (`seed_prefill_importance`); this crate only consumes `ctx.importance()` / `tensor(Scores)`.

use argus_extension_api::{
    CacheHandle, CacheOpError, EstimatorCtx, KVMutationStage, KeepSpec, KeepTopK, MutationPhase,
    QCF_ESTIMATORS, QcfEstimator, QcfEstimatorReg, StageArgs, StageCaps, StageCtx, StageParams,
    TensorKind, compile_keep_top_k, redistribute_value, register_kv_mutation_stage,
};
use linkme::distributed_slice;

/// The score-based caps for the v3 registration: H2O reads accumulated importance (Scores) and — to
/// stay faithful to the reference, which has no attention sink — protects NO prefix by default.
const H2O_CAPS: StageCaps = StageCaps {
    reads: &[TensorKind::Scores],
    default_protected_prefix: 0,
    produces_merge_plan: false,
};

/// Parse the absolute heavy-hitter / recent budgets from the technique-private `--set` blob.
///
/// EXPLICIT-REQUIRED: faithful H2O has no meaningful default budget (the budget IS the policy), so a
/// run that omits `hh_size`/`recent_size` is rejected at the CLI layer ([`Args::require_h2o_budgets`])
/// with a clean error — not here, since the `make` ABI is infallible. When absent here (e.g. the
/// registration self-test's empty args) both default to 0: a degenerate keep-prefix-only stage that
/// production never reaches.
fn parse_h2o_budgets(args: StageArgs<'_>) -> (usize, usize) {
    let mut hh_size = 0usize;
    let mut recent_size = 0usize;
    for a in args {
        match a.key {
            "hh_size" => {
                if let Ok(v) = a.val.parse() {
                    hh_size = v;
                }
            }
            "recent_size" => {
                if let Ok(v) = a.val.parse() {
                    recent_size = v;
                }
            }
            _ => {}
        }
    }
    (hh_size, recent_size)
}

/// H2O eviction stage. Absolute `hh_size`/`recent_size` budgets (+ optional `protected_prefix`), with
/// no clamps — faithful to `H2OKVCache_LayerWise(hh_size, recent_size)`.
struct H2o {
    hh_size: usize,
    recent_size: usize,
    protected_prefix: usize,
}

/// The shared 3-partition budget (prefix / top-`hh_size` heavy / `recent_size` recent), computed once
/// per plan. `None` from [`H2o::partition`] means already within budget (no-op).
struct Partition {
    prefix: usize,
    hh_budget: usize,
    recent: usize,
    current: usize,
}

impl Partition {
    /// One head's (or the layer-wide) ascending keep-list: prefix ∪ top-`hh_budget` scorers over the
    /// evictable middle ∪ the `recent`-token recency window. Routed through the engine T1 compiler.
    fn keep_list(&self, score: impl Fn(usize) -> f32) -> Vec<usize> {
        compile_keep_top_k(
            KeepTopK {
                current: self.current,
                prefix: self.prefix,
                recent: self.recent,
                heavy: self.hh_budget,
            },
            score,
        )
    }

    /// Score-free fallback: give the full evictable budget (`hh_budget + recent`) to recency.
    fn keep_list_recency(&self) -> Vec<usize> {
        compile_keep_top_k(
            KeepTopK {
                current: self.current,
                prefix: self.prefix,
                recent: self.hh_budget + self.recent,
                heavy: 0,
            },
            |_| 0.0,
        )
    }
}

impl H2o {
    fn from_args(p: StageParams, args: StageArgs<'_>) -> Self {
        let (hh_size, recent_size) = parse_h2o_budgets(args);
        Self {
            hh_size,
            recent_size,
            protected_prefix: p.protected_prefix,
        }
    }

    /// Partition by ABSOLUTE budget: keep `protected_prefix + hh_size + recent_size` tokens,
    /// independent of the engine's `target_len`. `None` when the cache is already within that budget
    /// (faithful to `H2OKVCache_LayerWise` evicting only past `hh_size + recent_size`).
    fn partition(&self, current: usize) -> Option<Partition> {
        let prefix = self.protected_prefix.min(current);
        let keep_total = prefix + self.hh_size + self.recent_size;
        if current <= keep_total {
            return None;
        }
        Some(Partition {
            prefix,
            hh_budget: self.hh_size,
            recent: self.recent_size,
            current,
        })
    }

    /// The keep-set shape (`None` = no-op within budget). Per-head when per-(kv_head, pos) scores are
    /// present (the reference granularity); otherwise layer-wide from the flat importance, and
    /// score-free to recency.
    fn keep_spec(&self, ctx: &dyn StageCtx) -> Option<KeepSpec> {
        let p = self.partition(ctx.current_pos())?;

        // (1) Per-head: each KV head ranks its own heavy hitters; all heads keep the same count.
        if ctx.has_head_scores() {
            let n_kv_heads = ctx.n_kv_heads().max(1);
            let heads: Vec<Vec<usize>> = (0..n_kv_heads)
                .map(|kv_h| p.keep_list(|pos| ctx.head_score(kv_h, pos)))
                .collect();
            return Some(KeepSpec::PerHead(heads));
        }

        // (2) Flat fallback: heavy hitters from the layer-wide importance. (3) Score-free: recency.
        let keep = match ctx.importance() {
            Some(imp) => p.keep_list(|pos| imp.get(pos).copied().unwrap_or(0.0)),
            None => p.keep_list_recency(),
        };
        Some(KeepSpec::LayerWide(keep))
    }
}

// ── v3 native (imperative) surface — the production path ──

impl KVMutationStage for H2o {
    fn name(&self) -> &str {
        "h2o"
    }

    /// Stage the per-head (or layer-wide fallback) heavy-hitter keep-set, or no-op within budget.
    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        match self.keep_spec(ctx) {
            None => Ok(()),
            Some(KeepSpec::LayerWide(keep)) => cache.keep(&keep),
            Some(KeepSpec::PerHead(heads)) => {
                let refs: Vec<&[usize]> = heads.iter().map(|h| h.as_slice()).collect();
                cache.keep_per_head(&refs)
            }
        }
    }
}

register_kv_mutation_stage!(
    "h2o",
    |p, args| Box::new(H2o::from_args(p, args)),
    H2O_CAPS,
    MutationPhase::KvMutate
);

// ── QCF estimator (observer/score axis) ──────────────────────────

/// Identify the H2O-retained token set for the QCF simulation: protected prefix + top-`hh_size` heavy
/// hitters (by importance) + the `recent_size` recency window. Absolute budgets, faithful to the
/// actuator's [`H2o::keep_spec`].
fn identify_retained_h2o(
    importance: &[f32],
    current_pos: usize,
    hh_size: usize,
    recent_size: usize,
    protected_prefix: usize,
) -> Vec<usize> {
    let prefix = protected_prefix.min(current_pos);
    let recent_start = current_pos.saturating_sub(recent_size);
    let mut retained: Vec<usize> = (0..prefix).collect();
    if recent_start > prefix {
        let mut evictable: Vec<(usize, f32)> = (prefix..recent_start)
            .map(|t| (t, importance.get(t).copied().unwrap_or(0.0)))
            .collect();
        evictable.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        retained.extend(evictable.iter().take(hh_size).map(|(t, _)| t));
    }
    retained.extend(recent_start..current_pos);
    retained.sort();
    retained.dedup();
    retained
}

/// H2O QCF estimator: prefix + heavy-hitter + recent retained set, then O_after redistribution over
/// it. Kept in lockstep with the faithful actuator (absolute `hh_size`/`recent_size` + prefix).
struct H2oEstimator {
    hh_size: usize,
    recent_size: usize,
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
        if current <= self.protected_prefix + self.hh_size + self.recent_size {
            return false;
        }
        let mut alpha = vec![0.0f32; current];
        ctx.alpha_h(kv_head, &mut alpha);
        let retained = identify_retained_h2o(
            &alpha,
            current,
            self.hh_size,
            self.recent_size,
            self.protected_prefix,
        );
        redistribute_value(ctx, kv_head, &alpha, &retained, ctx.beta(), out);
        true
    }
}

/// Registration — found via `find_qcf_estimator("h2o")`. Parses the same absolute budgets as the
/// actuator so the estimate ranks on the identical retained set. Score-based.
#[distributed_slice(QCF_ESTIMATORS)]
static H2O_QCF: QcfEstimatorReg = QcfEstimatorReg {
    name: "h2o",
    curve_key: "kv.evict_h2o",
    make: |p: StageParams, args: StageArgs<'_>| {
        let (hh_size, recent_size) = parse_h2o_budgets(args);
        Box::new(H2oEstimator {
            hh_size,
            recent_size,
            protected_prefix: p.protected_prefix,
        })
    },
    requires_scores: true,
    requires_streaming_config: false,
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{
        PluginArg, TensorDtype, TensorHandle, TensorShape, find_mutation_stage,
    };

    // ── per-head mock (carried over from the deleted h2o-plus crate) ──

    /// Minimal ctx supplying optional per-(kv_head, pos) scores via `tensor(Scores)` and optional flat
    /// importance. Faithful H2O ignores `target_len` (absolute budget), so it is a fixed 0 here.
    struct Ctx {
        current: usize,
        n_kv_heads: usize,
        stride: usize,
        head_scores: Option<Vec<f32>>, // [n_kv_heads * stride]
        importance: Option<Vec<f32>>,
    }
    struct ScoresHandle<'a> {
        data: &'a [f32],
        rows: usize,
        stride: usize,
    }
    impl TensorHandle for ScoresHandle<'_> {
        fn shape(&self) -> TensorShape {
            TensorShape {
                rows: self.rows,
                cols: 1,
                per_head: true,
            }
        }
        fn dtype(&self) -> TensorDtype {
            TensorDtype::F32
        }
        fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
            out[0] = self
                .data
                .get(kv_head * self.stride + row)
                .copied()
                .unwrap_or(0.0);
        }
    }
    impl StageCtx for Ctx {
        fn current_pos(&self) -> usize {
            self.current
        }
        fn target_len(&self) -> usize {
            0
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn importance(&self) -> Option<&[f32]> {
            self.importance.as_deref()
        }
        fn n_kv_heads(&self) -> usize {
            self.n_kv_heads
        }
        fn head_dim(&self) -> usize {
            4
        }
        fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
            match kind {
                TensorKind::Scores => self.head_scores.as_ref().map(|d| {
                    Box::leak(Box::new(ScoresHandle {
                        data: d,
                        rows: self.current,
                        stride: self.stride,
                    })) as &dyn TensorHandle
                }),
                _ => None,
            }
        }
    }

    /// A mock [`CacheHandle`] capturing `keep` / `keep_per_head`.
    #[derive(Default)]
    struct CaptureHandle {
        cur: usize,
        n_kv: usize,
        kept: Option<Vec<usize>>,
        kept_per_head: Option<Vec<Vec<usize>>>,
    }
    impl CacheHandle for CaptureHandle {
        fn current_pos(&self) -> usize {
            self.cur
        }
        fn n_kv_heads(&self) -> usize {
            self.n_kv
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
        fn keep_per_head(&mut self, keep: &[&[usize]]) -> Result<(), CacheOpError> {
            self.kept_per_head = Some(keep.iter().map(|h| h.to_vec()).collect());
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

    fn budgets(hh: &'static str, recent: &'static str) -> [PluginArg<'static>; 2] {
        [
            PluginArg {
                key: "hh_size",
                val: hh,
            },
            PluginArg {
                key: "recent_size",
                val: recent,
            },
        ]
    }

    /// Faithful H2O has NO attention-sink prefix by default, and still reads accumulated Scores.
    #[test]
    fn caps_have_no_forced_prefix() {
        assert_eq!(H2O_CAPS.default_protected_prefix, 0);
        assert_eq!(H2O_CAPS.reads, &[TensorKind::Scores]);
        // protected_prefix is passed through unclamped (no `.max(4)`).
        let s = H2o::from_args(
            StageParams {
                protected_prefix: 0,
                ..Default::default()
            },
            &budgets("4", "6"),
        );
        assert_eq!(s.protected_prefix, 0);
        assert_eq!(s.hh_size, 4);
        assert_eq!(s.recent_size, 6);
    }

    /// Within the absolute budget (`prefix + hh_size + recent_size`) → no-op.
    #[test]
    fn within_absolute_budget_is_noop() {
        let s = H2o::from_args(StageParams::default(), &budgets("4", "6"));
        // current=10 == 0+4+6 → within budget.
        let ctx = Ctx {
            current: 10,
            n_kv_heads: 1,
            stride: 0,
            head_scores: None,
            importance: None,
        };
        assert!(s.keep_spec(&ctx).is_none());
    }

    /// Score-free fallback gives the full evictable budget to recency (keep the last `hh+recent`).
    #[test]
    fn score_free_keeps_recency() {
        let s = H2o::from_args(StageParams::default(), &budgets("4", "6"));
        let ctx = Ctx {
            current: 20,
            n_kv_heads: 1,
            stride: 0,
            head_scores: None,
            importance: None,
        };
        match s.keep_spec(&ctx) {
            Some(KeepSpec::LayerWide(keep)) => assert_eq!(keep, (10..20).collect::<Vec<_>>()),
            other => panic!("expected LayerWide recency, got {other:?}"),
        }
    }

    /// Score-based: keep the top-`hh_size` heavy hitters (absolute) + the `recent_size` recency window.
    #[test]
    fn score_based_absolute_heavy_and_recent() {
        // hh_size=2, recent_size=4, current=20 → recent [16,20); heavy top-2 over [0,16).
        let s = H2o::from_args(StageParams::default(), &budgets("2", "4"));
        let mut imp = vec![0.0f32; 20];
        imp[5] = 10.0;
        imp[9] = 9.0;
        imp[2] = 1.0; // lower — must NOT be kept
        let ctx = Ctx {
            current: 20,
            n_kv_heads: 1,
            stride: 0,
            head_scores: None,
            importance: Some(imp),
        };
        match s.keep_spec(&ctx) {
            Some(KeepSpec::LayerWide(keep)) => {
                assert_eq!(keep, vec![5, 9, 16, 17, 18, 19]);
            }
            other => panic!("expected LayerWide, got {other:?}"),
        }
    }

    /// Per-head: each KV head ranks its own heavy hitters (the reference granularity); `on_phase`
    /// stages `keep_per_head`, not layer-wide.
    #[test]
    fn per_head_ranks_independently() {
        let s = H2o::from_args(StageParams::default(), &budgets("3", "3"));
        // head 0 prefers 5,6,7; head 1 prefers 10,11,12 (over the evictable [0, current-recent)).
        let (n_kv_heads, stride) = (2usize, 100usize);
        let mut hs = vec![0.0f32; n_kv_heads * stride];
        for (i, &pos) in [5usize, 6, 7].iter().enumerate() {
            hs[pos] = 10.0 - i as f32;
        }
        for (i, &pos) in [10usize, 11, 12].iter().enumerate() {
            hs[stride + pos] = 10.0 - i as f32;
        }
        let ctx = Ctx {
            current: 20,
            n_kv_heads,
            stride,
            head_scores: Some(hs),
            importance: None,
        };
        let expected = match s.keep_spec(&ctx).unwrap() {
            KeepSpec::PerHead(h) => h,
            KeepSpec::LayerWide(_) => panic!("expected PerHead"),
        };
        let mut h = CaptureHandle {
            cur: 20,
            n_kv: n_kv_heads,
            ..Default::default()
        };
        <H2o as KVMutationStage>::on_phase(&s, &ctx, &mut h).unwrap();
        assert_eq!(h.kept_per_head, Some(expected));
        assert_eq!(h.kept, None, "per-head path must NOT use layer-wide keep");
    }

    /// v3 native registration + DECISION equivalence: `on_phase` stages exactly what `keep_spec`
    /// decides, for the score-free, score-based, and per-head cases.
    #[test]
    fn v3_native_matches_keep_spec_decision() {
        let reg = find_mutation_stage("h2o").expect("h2o in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "h2o");
        assert_eq!(reg.phase, MutationPhase::KvMutate);
        assert_eq!(reg.caps, H2O_CAPS);
        assert_eq!(
            (reg.make)(StageParams::default(), &budgets("2", "4")).name(),
            "h2o"
        );

        let s = H2o::from_args(StageParams::default(), &budgets("2", "4"));
        let imp: Vec<f32> = (0..20).map(|i| (i % 7) as f32).collect();
        let cases = [
            Ctx {
                current: 8,
                n_kv_heads: 1,
                stride: 0,
                head_scores: None,
                importance: None,
            }, // within budget -> no-op
            Ctx {
                current: 20,
                n_kv_heads: 1,
                stride: 0,
                head_scores: None,
                importance: Some(imp),
            }, // score-based layer-wide
        ];
        for ctx in &cases {
            let mut h = CaptureHandle {
                cur: ctx.current,
                n_kv: 1,
                ..Default::default()
            };
            <H2o as KVMutationStage>::on_phase(&s, ctx, &mut h).unwrap();
            let expected = match s.keep_spec(ctx) {
                None => None,
                Some(KeepSpec::LayerWide(k)) => Some(k),
                Some(KeepSpec::PerHead(_)) => unreachable!(),
            };
            assert_eq!(h.kept, expected, "current={}", ctx.current);
        }
    }

    /// The QCF estimator's retained set matches the actuator's keep-set for a shared case (no silent
    /// estimate/actuator skew).
    #[test]
    fn qcf_estimator_matches_actuator_retained() {
        let hh_size = 2;
        let recent_size = 4;
        let prefix = 0;
        let current = 20;
        let imp: Vec<f32> = (0..current)
            .map(|i| if i == 5 { 10.0 } else { (i % 3) as f32 })
            .collect();

        let retained = identify_retained_h2o(&imp, current, hh_size, recent_size, prefix);

        let s = H2o {
            hh_size,
            recent_size,
            protected_prefix: prefix,
        };
        let ctx = Ctx {
            current,
            n_kv_heads: 1,
            stride: 0,
            head_scores: None,
            importance: Some(imp),
        };
        let actuator_keep = match s.keep_spec(&ctx).unwrap() {
            KeepSpec::LayerWide(k) => k,
            KeepSpec::PerHead(_) => unreachable!(),
        };
        assert_eq!(retained, actuator_keep);
    }
}
