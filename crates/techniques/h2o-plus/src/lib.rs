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
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageCaps, StageCtx,
    StageParams,
};
use linkme::distributed_slice;

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
    /// Total evictable budget `keep - prefix` (= hh_budget + recent_budget).
    available: usize,
    hh_budget: usize,
    /// Start of the recent window in the **score-based** split (`current - recent_budget`).
    recent_start: usize,
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
        let recent_budget = available - hh_budget;
        let recent_start = current.saturating_sub(recent_budget).max(prefix);
        Some(Partition {
            prefix,
            available,
            hh_budget,
            recent_start,
            current,
        })
    }
}

/// Build one head's (or the layer-wide) prefix-inclusive ascending keep-list from a per-position
/// score reader over the evictable range `[prefix, recent_start)`: keep the prefix, the top
/// `hh_budget` scorers (re-sorted by position), and the recent window `[recent_start, current)`.
fn keep_list_from_scores(p: &Partition, score: impl Fn(usize) -> f32) -> Vec<usize> {
    let mut token_scores: Vec<(usize, f32)> = (p.prefix..p.recent_start)
        .map(|pos| (pos, score(pos)))
        .collect();
    token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut hh: Vec<usize> = token_scores
        .iter()
        .take(p.hh_budget)
        .map(|(pos, _)| *pos)
        .collect();
    hh.sort_unstable();

    let mut keep: Vec<usize> = (0..p.prefix).collect();
    keep.extend_from_slice(&hh);
    keep.extend(p.recent_start..p.current);
    keep
}

impl KVCacheStage for H2OPlus {
    fn name(&self) -> &str {
        "h2o_plus"
    }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
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
            return Some(KVCachePlan {
                keep: KeepSpec::PerHead(heads),
                merges: Vec::new(),
                channels: None,
            });
        }

        // (2) Flat fallback: heavy hitters from the layer-wide importance score (score-based H2O).
        // (3) Score-free fallback: no heavy hitters — give the FULL budget to recency (keep prefix +
        //     the last `available` tokens), matching the original `H2OPlusPolicy::evict` which
        //     retained `keep` tokens (NOT `keep - hh_budget`).
        let keep = match ctx.importance() {
            Some(imp) => keep_list_from_scores(&p, |pos| imp.get(pos).copied().unwrap_or(0.0)),
            None => {
                let recent_start = p.current.saturating_sub(p.available).max(p.prefix);
                let mut keep: Vec<usize> = (0..p.prefix).collect();
                keep.extend(recent_start..p.current);
                keep
            }
        };
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
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
    caps: StageCaps {
        reads: &[argus_extension_api::TensorKind::Scores],
        default_protected_prefix: 4,
        produces_merge_plan: false,
    },
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
