//! H2O+ (GQA-aware per-head heavy-hitter eviction) technique crate — extends H2O with per-KV-head
//! token selection (each KV head keeps its own heavy hitters, all heads keeping the same *number*
//! of tokens so the single `current_pos` invariant holds).
//!
//! Extracted from the engine core into a self-registering technique crate (the
//! `streaming-llm`/`h2o`/`d2o` precedent): depends only on `argus-extension-api` + `linkme`,
//! implements [`KVCacheStage`], and registers under the name `"h2o_plus"` via
//! `#[distributed_slice(KV_CACHE_STAGES)]`. The engine force-links it (`use h2o_plus as _;`).
//!
//! This is the first plugin to emit a **per-head** plan ([`KeepSpec::PerHead`]): when the engine
//! supplies per-(kv_head, pos) accumulated importance via `ctx.tensor(Scores)` (the F5 score source,
//! routed by `StageBackedPolicy::evict_with_head_scores`), each head independently ranks its heavy
//! hitters and the engine's per-head plan executor compacts each head separately. When per-head
//! scores are absent the stage degrades to the flat H2O plan (score-based `LayerWide`), and with no
//! scores at all to recency (`LayerWide`) — identical to the engine's former `H2OPlusPolicy`
//! fallbacks, which is the only path production currently exercises.
//!
//! 3-partition model (per head): `[Protected Prefix] [Heavy Hitters] [Recent Window]`.

use argus_extension_api::{
    CacheHandle, CacheOpError, KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg,
    KVMutationStage, KeepSpec, KeepTopK, MutationPhase, StageCaps, StageCtx, StageParams,
    TensorKind, compile_keep_top_k, register_kv_mutation_stage,
};
use linkme::distributed_slice;

/// The score-based caps shared by the v2 [`KVCacheStageReg`] and the v3 registration: H2O+ ranks
/// per-head heavy hitters by accumulated importance (Scores), protecting 4 sinks by default.
const H2OPLUS_CAPS: StageCaps = StageCaps {
    reads: &[TensorKind::Scores],
    default_protected_prefix: 4,
    produces_merge_plan: false,
};

/// H2O+ eviction stage. `keep_ratio` is clamped to `[0,1]` and `protected_prefix` to ≥4 (attention
/// sink), matching the original engine `H2OPlusPolicy::new`.
struct H2OPlus {
    keep_ratio: f32,
    protected_prefix: usize,
}

impl H2OPlus {
    fn new(keep_ratio: f32, protected_prefix: usize) -> Self {
        Self {
            keep_ratio: keep_ratio.clamp(0.0, 1.0),
            protected_prefix: protected_prefix.max(4),
        }
    }
}

/// The shared 3-partition budget split (prefix / heavy-hitters / recent), computed once per plan.
struct Partition {
    prefix: usize,
    /// Total evictable budget `keep - prefix` (= hh_budget + recent_budget). The recent-window count
    /// passed to the T1 compiler is `available - hh_budget`.
    available: usize,
    hh_budget: usize,
    current: usize,
}

impl H2OPlus {
    /// Returns the partition, or `None` when already within budget (no-op). Mirrors the budget math
    /// in the original `H2OPlusPolicy::evict*`.
    fn partition(&self, current: usize, target_len: usize) -> Option<Partition> {
        let prefix = self.protected_prefix;
        let keep = target_len.max(prefix + 2);
        if current <= keep {
            return None;
        }
        let available = keep.saturating_sub(prefix);
        let hh_budget = (available as f32 * self.keep_ratio) as usize;
        Some(Partition {
            prefix,
            available,
            hh_budget,
            current,
        })
    }
}

/// Build one head's (or the layer-wide) prefix-inclusive ascending keep-list from a per-position
/// score reader over the evictable range `[prefix, recent_start)`: keep the prefix, the top
/// `hh_budget` scorers (re-sorted by position), and the recent window `[recent_start, current)`.
fn keep_list_from_scores(p: &Partition, score: impl Fn(usize) -> f32) -> Vec<usize> {
    // The recent window count is `available - hh_budget` (so `recent_start` matches `p.recent_start`).
    // Routed through the engine's T1 compiler — byte-identical to the verbatim selection it replaced.
    compile_keep_top_k(
        KeepTopK {
            current: p.current,
            prefix: p.prefix,
            recent: p.available - p.hh_budget,
            heavy: p.hh_budget,
        },
        score,
    )
}

impl H2OPlus {
    /// The keep-set shape (`None` = no-op within budget), shared by the v3 `on_phase` and the v2
    /// `plan` so they decide byte-identically. Per-head when head scores are present (each KV head
    /// ranks its own heavy hitters, all heads keep the same count); otherwise a layer-wide keep from
    /// the flat importance (score-based) or recency (score-free).
    fn keep_spec(&self, ctx: &dyn StageCtx) -> Option<KeepSpec> {
        let current = ctx.current_pos();
        let p = self.partition(current, ctx.target_len())?;

        // (1) Per-head: each KV head ranks its own heavy hitters from the per-(kv_head, pos) score
        //     source. All heads keep the same count (prefix + hh_budget + recent), so the engine's
        //     single current_pos invariant holds.
        if ctx.has_head_scores() {
            let n_kv_heads = ctx.n_kv_heads().max(1);
            let heads: Vec<Vec<usize>> = (0..n_kv_heads)
                .map(|kv_h| keep_list_from_scores(&p, |pos| ctx.head_score(kv_h, pos)))
                .collect();
            return Some(KeepSpec::PerHead(heads));
        }

        // (2) Flat fallback: heavy hitters from the layer-wide importance score (score-based H2O).
        // (3) Score-free fallback: no heavy hitters — give the FULL budget to recency (keep prefix +
        //     the last `available` tokens), matching the original `H2OPlusPolicy::evict` which
        //     retained `keep` tokens (NOT `keep - hh_budget`).
        let keep = match ctx.importance() {
            Some(imp) => keep_list_from_scores(&p, |pos| imp.get(pos).copied().unwrap_or(0.0)),
            None => compile_keep_top_k(
                KeepTopK {
                    current: p.current,
                    prefix: p.prefix,
                    recent: p.available,
                    heavy: 0,
                },
                |_| 0.0,
            ),
        };
        Some(KeepSpec::LayerWide(keep))
    }
}

// ── v3 native (imperative) surface — the production path ──

impl KVMutationStage for H2OPlus {
    fn name(&self) -> &str {
        "h2o_plus"
    }

    /// Stage the per-head (or layer-wide fallback) heavy-hitter keep-set, or no-op within budget.
    /// Byte-identical to the v2 plan via the shared `keep_spec`. The per-head path needs HeadMajor
    /// layout (the engine supplies head scores only there); `keep_per_head` enforces it.
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
    "h2o_plus",
    |p, _args| Box::new(H2OPlus::new(p.keep_ratio, p.protected_prefix)),
    H2OPLUS_CAPS,
    MutationPhase::KvMutate
);

// ── v2 plan-returning surface (kept for the migration window; removed in Phase 2) ──

impl KVCacheStage for H2OPlus {
    fn name(&self) -> &str {
        "h2o_plus"
    }

    /// Decides via the shared `keep_spec`, so it is byte-identical to the v3 `on_phase`.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        self.keep_spec(ctx).map(|keep| KVCachePlan {
            keep,
            merges: Vec::new(),
            channels: None,
        })
    }
}

/// Registration — the engine finds this via `find_stage("h2o_plus")`. `keep_ratio`/`protected_prefix`
/// flow in from [`StageParams`] (CLI `eviction plugin --name h2o_plus --set keep_ratio=<R>` +
/// `--protected-prefix`).
#[distributed_slice(KV_CACHE_STAGES)]
static H2O_PLUS: KVCacheStageReg = KVCacheStageReg {
    name: "h2o_plus",
    make: |p: StageParams| Box::new(H2OPlus::new(p.keep_ratio, p.protected_prefix)),
    make_with_args: |p: StageParams, _args| {
        Box::new(H2OPlus::new(p.keep_ratio, p.protected_prefix))
    },
    // H2O+ ranks per-head heavy hitters by accumulated importance (score-based); protect 4 sinks.
    caps: H2OPLUS_CAPS,
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorDtype, TensorHandle, TensorKind, TensorShape, find_stage};

    /// Minimal ctx supplying optional per-(kv_head, pos) scores via `tensor(Scores)` (stride = `cols`
    /// is 1; the handle indexes `data[kv_head * stride + pos]`) and optional flat importance.
    struct Ctx {
        current: usize,
        target: usize,
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
            self.target
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

    #[test]
    fn registers_with_score_based_caps() {
        let reg = find_stage("h2o_plus").expect("h2o_plus registered in KV_CACHE_STAGES");
        assert_eq!(reg.name, "h2o_plus");
        assert!(!reg.caps.reads.is_empty());
        assert_eq!(reg.caps.default_protected_prefix, 4);
    }

    /// A mock [`CacheHandle`] capturing keep / keep_per_head.
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

    /// v3 native registration + DECISION equivalence: the v3 `on_phase` stages the same shape the v2
    /// `plan` returns — `keep_per_head` for the per-head path, `keep` for the layer-wide fallback.
    #[test]
    fn v3_native_matches_v2_decision() {
        use argus_extension_api::find_mutation_stage;
        let reg = find_mutation_stage("h2o_plus").expect("h2o_plus in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "h2o_plus");
        assert_eq!(reg.caps, H2OPLUS_CAPS);
        assert_eq!((reg.make)(StageParams::default(), &[]).name(), "h2o_plus");

        let s = H2OPlus::new(0.5, 4);

        // per-head path: head 0 prefers 5,6,7; head 1 prefers 10,11,12.
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
            target: 10,
            n_kv_heads,
            stride,
            head_scores: Some(hs),
            importance: None,
        };
        let expected_per_head = match s.plan(&ctx).unwrap().keep {
            KeepSpec::PerHead(h) => h,
            KeepSpec::LayerWide(_) => panic!("expected PerHead"),
        };
        let mut h = CaptureHandle {
            cur: 20,
            n_kv: n_kv_heads,
            ..Default::default()
        };
        <H2OPlus as KVMutationStage>::on_phase(&s, &ctx, &mut h).unwrap();
        assert_eq!(h.kept_per_head, Some(expected_per_head));
        assert_eq!(h.kept, None, "per-head path must NOT use layer-wide keep");

        // layer-wide fallback (no head scores, flat importance).
        let imp: Vec<f32> = (0..20).map(|i| (i % 5) as f32).collect();
        let ctx2 = Ctx {
            current: 20,
            target: 10,
            n_kv_heads: 1,
            stride: 0,
            head_scores: None,
            importance: Some(imp),
        };
        let expected_lw = match s.plan(&ctx2).unwrap().keep {
            KeepSpec::LayerWide(k) => k,
            KeepSpec::PerHead(_) => panic!("expected LayerWide"),
        };
        let mut h2 = CaptureHandle {
            cur: 20,
            n_kv: 1,
            ..Default::default()
        };
        <H2OPlus as KVMutationStage>::on_phase(&s, &ctx2, &mut h2).unwrap();
        assert_eq!(h2.kept, Some(expected_lw));
        assert_eq!(h2.kept_per_head, None);
    }

    #[test]
    fn per_head_selects_different_heavy_hitters() {
        // current=20, target=10, prefix=4, keep_ratio=0.5 → keep=10, available=6, hh_budget=3,
        // recent_budget=3, recent_start=max(4,17)=17, evictable [4,17). Each head keeps prefix(0..4)
        // + its own 3 HH + recent (17..20) = 4+3+3 = 10 tokens.
        let n_kv_heads = 2;
        let stride = 100;
        let mut hs = vec![0.0f32; n_kv_heads * stride];
        // head 0 prefers 5,6,7; head 1 prefers 10,11,12.
        for (i, &pos) in [5usize, 6, 7].iter().enumerate() {
            hs[pos] = 10.0 - i as f32;
        }
        for (i, &pos) in [10usize, 11, 12].iter().enumerate() {
            hs[stride + pos] = 10.0 - i as f32;
        }
        let ctx = Ctx {
            current: 20,
            target: 10,
            n_kv_heads,
            stride,
            head_scores: Some(hs),
            importance: Some(vec![1.0; 100]),
        };
        let plan = H2OPlus::new(0.5, 4).plan(&ctx).expect("plan Some");
        match plan.keep {
            KeepSpec::PerHead(heads) => {
                assert_eq!(heads.len(), 2);
                assert_eq!(heads[0], vec![0, 1, 2, 3, 5, 6, 7, 17, 18, 19]);
                assert_eq!(heads[1], vec![0, 1, 2, 3, 10, 11, 12, 17, 18, 19]);
                // engine invariant: all heads keep the same count.
                assert_eq!(heads[0].len(), heads[1].len());
            }
            KeepSpec::LayerWide(_) => panic!("expected PerHead when head scores are supplied"),
        }
        assert!(plan.merges.is_empty());
    }

    #[test]
    fn flat_fallback_without_head_scores_is_layerwide() {
        // No head scores → flat H2O LayerWide using importance.
        let mut imp = vec![0.0f32; 100];
        imp[5] = 10.0;
        imp[6] = 9.0;
        imp[7] = 8.0;
        let ctx = Ctx {
            current: 20,
            target: 10,
            n_kv_heads: 2,
            stride: 100,
            head_scores: None,
            importance: Some(imp),
        };
        let plan = H2OPlus::new(0.5, 4).plan(&ctx).expect("plan Some");
        match plan.keep {
            KeepSpec::LayerWide(k) => assert_eq!(k, vec![0, 1, 2, 3, 5, 6, 7, 17, 18, 19]),
            KeepSpec::PerHead(_) => panic!("expected LayerWide flat fallback"),
        }
    }

    #[test]
    fn score_free_fallback_keeps_prefix_and_recent() {
        let ctx = Ctx {
            current: 20,
            target: 10,
            n_kv_heads: 2,
            stride: 100,
            head_scores: None,
            importance: None,
        };
        let plan = H2OPlus::new(0.5, 4).plan(&ctx).expect("plan Some");
        match plan.keep {
            // score-free gives the FULL budget to recency: available=6 → keep prefix + last 6 tokens
            // ([14,20)) = 10 tokens total (= target), matching the old H2OPlusPolicy::evict.
            KeepSpec::LayerWide(k) => assert_eq!(k, vec![0, 1, 2, 3, 14, 15, 16, 17, 18, 19]),
            KeepSpec::PerHead(_) => panic!("expected LayerWide"),
        }
    }

    #[test]
    fn within_budget_is_noop() {
        let ctx = Ctx {
            current: 8,
            target: 10,
            n_kv_heads: 2,
            stride: 100,
            head_scores: Some(vec![0.0; 200]),
            importance: None,
        };
        assert!(H2OPlus::new(0.5, 4).plan(&ctx).is_none());
    }
}
