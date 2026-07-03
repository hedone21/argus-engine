//! TOVA (Token Omission Via Attention) technique crate — last-step-attention KV eviction
//! (Oren et al. 2023, "Transformers are Multi-State RNNs", §4.1).
//!
//! **The P3 litmus of the signal-axis inversion** (`docs/design/signal-axis-inversion.md` §6–7): a NEW
//! signal-consuming eviction technique added as a producer/stage crate ONLY. The engine core is
//! untouched but for the one-line `use tova as _;` force-link (the litmus gate excludes it). TOVA
//! subscribes to a DIFFERENT signal than the other score-based techniques — `attn.last_step` (the most
//! recent decode step's per-head attention, the CAOTE `last_layer_head_attn` overwrite the
//! `attn_score` producer already tracks), NOT h2o's `attn.cum_importance` — proving the inversion lets
//! a technique observe a distinct signal with zero engine-core cost.
//!
//! **Policy.** A FIXED cache size `cache_size` (TOVA's multi-state budget). At an eviction boundary,
//! keep the `cache_size` tokens with the highest last-step attention (per KV head when per-head
//! attention is tracked, else layer-wide), plus the protected prefix. The budget is ABSOLUTE — the
//! budget IS the policy, so it ignores the engine's `target_len` (the h2o precedent). Select with
//! `eviction plugin --name tova --set cache_size=N`; unset (`0`) → no-op (nothing to omit).
//!
//! **Signal read.** The last-step attention is read via `ctx.signal(SignalId("attn.last_step"))` — the
//! P2 signal accessor — which `attn_score` declares it `produces`, so the boot
//! `descriptor::validate_signal_occupancy` handshake covers this read (the h2o `attn.cum_importance`
//! precedent). Falls back to the flat cumulative importance (`Scores`) when per-head last-step
//! attention is absent (flat models / the GPU cumulative proxy), and to recency when score-free.

use argus_extension_api::{
    CacheHandle, CacheOpError, KVMutationStage, KeepSpec, KeepTopK, MutationPhase, SignalId,
    StageArgs, StageCaps, StageCtx, StageParams, TensorKind, compile_keep_top_k,
    register_kv_mutation_stage,
};

/// The signal that gates TOVA's per-head ranking (the P2 signal accessor + boot handshake).
const ATTN_LAST_STEP: SignalId = SignalId("attn.last_step");

/// The caps for the v3 registration: TOVA reads the last-step attention (`AttnWeights`) it ranks on,
/// plus `Scores` (the flat-importance fallback + the non-empty `reads` that arms the score producer via
/// `stage_is_score_based`). No forced prefix — faithful to TOVA, which has no attention-sink concept;
/// the engine injects `--protected-prefix` through `StageParams`.
const TOVA_CAPS: StageCaps = StageCaps {
    reads: &[TensorKind::AttnWeights, TensorKind::Scores],
    // Open-signal-name handshake (L1): `attn_score` produces `"attn.last_step"`, so the boot
    // `descriptor::validate_signal_occupancy` self-test verifies this read is not orphaned — the
    // signal-axis sibling of the `reads`↔`produces` TensorKind occupancy invariant (h2o precedent).
    reads_signals: &[ATTN_LAST_STEP],
    default_protected_prefix: 0,
    produces_merge_plan: false,
    whole_model: false,
    prefill_attn_window: None,
};

/// Parse the absolute `cache_size` budget from the technique-private `--set` blob. Unset → 0 (no-op).
fn parse_cache_size(args: StageArgs<'_>) -> usize {
    let mut cache_size = 0usize;
    for a in args {
        if a.key == "cache_size"
            && let Ok(v) = a.val.parse()
        {
            cache_size = v;
        }
    }
    cache_size
}

/// TOVA eviction stage — fixed `cache_size` budget ranked by last-step attention (+ protected prefix).
struct Tova {
    cache_size: usize,
    protected_prefix: usize,
}

impl Tova {
    fn from_args(p: StageParams, args: StageArgs<'_>) -> Self {
        Self {
            cache_size: parse_cache_size(args),
            protected_prefix: p.protected_prefix,
        }
    }

    /// The keep shape (`None` = no-op: budget unset or already within it). Per-head when the last-step
    /// attention is tracked per KV head (each head keeps its own top-`cache_size`, equal counts so the
    /// single `current_pos` invariant holds — the h2o granularity); else layer-wide from the flat
    /// importance; else recency.
    fn keep_spec(&self, ctx: &dyn StageCtx) -> Option<KeepSpec> {
        let current = ctx.current_pos();
        let prefix = self.protected_prefix.min(current);
        // cache_size == 0 (unset) → nothing to omit; within the absolute budget → no-op.
        if self.cache_size == 0 || current <= prefix + self.cache_size {
            return None;
        }

        // (1) Per-head on the last-step attention. `ctx.signal(attn.last_step)` (the P2 accessor)
        // bridges to `AttnWeights`; its presence gates the per-head path.
        if ctx.signal(ATTN_LAST_STEP).is_some() {
            let n_kv_heads = ctx.n_kv_heads().max(1);
            let heads: Vec<Vec<usize>> = (0..n_kv_heads)
                .map(|kv_h| {
                    compile_keep_top_k(
                        KeepTopK {
                            current,
                            prefix,
                            recent: 0,
                            heavy: self.cache_size,
                        },
                        |pos| ctx.attn_weight(kv_h, pos),
                    )
                })
                .collect();
            return Some(KeepSpec::PerHead(heads));
        }

        // (2) Flat fallback: heavy hitters by the layer-wide cumulative importance.
        // (3) Score-free: give the whole budget to recency (keep the last `cache_size`).
        let keep = match ctx.importance() {
            Some(imp) => compile_keep_top_k(
                KeepTopK {
                    current,
                    prefix,
                    recent: 0,
                    heavy: self.cache_size,
                },
                |pos| imp.get(pos).copied().unwrap_or(0.0),
            ),
            None => compile_keep_top_k(
                KeepTopK {
                    current,
                    prefix,
                    recent: self.cache_size,
                    heavy: 0,
                },
                |_| 0.0,
            ),
        };
        Some(KeepSpec::LayerWide(keep))
    }
}

// ── v3 native (imperative) surface — the production path ──

impl KVMutationStage for Tova {
    fn name(&self) -> &str {
        "tova"
    }

    /// Stage the per-head (or layer-wide fallback) last-step-attention keep-set, or no-op within budget.
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
    "tova",
    |p, args| Box::new(Tova::from_args(p, args)),
    TOVA_CAPS,
    MutationPhase::KvMutate
);

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{
        PluginArg, TensorDtype, TensorHandle, TensorShape, find_mutation_stage,
    };

    /// Minimal ctx supplying optional per-(kv_head, pos) last-step attention via `tensor(AttnWeights)`
    /// and optional flat importance via `importance()`. TOVA ignores `target_len` (absolute budget), so
    /// it is a fixed 0.
    struct Ctx {
        current: usize,
        n_kv_heads: usize,
        stride: usize,
        last_attn: Option<Vec<f32>>, // [n_kv_heads * stride], the AttnWeights signal
        importance: Option<Vec<f32>>,
    }
    struct AttnHandle<'a> {
        data: &'a [f32],
        rows: usize,
        stride: usize,
    }
    impl TensorHandle for AttnHandle<'_> {
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
                TensorKind::AttnWeights => self.last_attn.as_ref().map(|d| {
                    Box::leak(Box::new(AttnHandle {
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

    fn cache_size(n: &'static str) -> [PluginArg<'static>; 1] {
        [PluginArg {
            key: "cache_size",
            val: n,
        }]
    }

    /// TOVA reads the last-step attention signal (a DIFFERENT signal than h2o's cum_importance), and
    /// declares it by name so the boot occupancy handshake covers the read.
    #[test]
    fn caps_declare_last_step_signal() {
        assert_eq!(TOVA_CAPS.reads_signals, &[ATTN_LAST_STEP]);
        assert!(TOVA_CAPS.reads.contains(&TensorKind::AttnWeights));
        assert_eq!(TOVA_CAPS.default_protected_prefix, 0);
    }

    /// Unset budget (`cache_size == 0`) and within-budget both no-op.
    #[test]
    fn unset_or_within_budget_is_noop() {
        // Unset → no-op regardless of occupancy.
        let unset = Tova::from_args(StageParams::default(), &[]);
        let ctx = Ctx {
            current: 100,
            n_kv_heads: 1,
            stride: 0,
            last_attn: None,
            importance: None,
        };
        assert!(unset.keep_spec(&ctx).is_none());

        // current == prefix(0) + cache_size(10) → within budget.
        let s = Tova::from_args(StageParams::default(), &cache_size("10"));
        let ctx = Ctx {
            current: 10,
            n_kv_heads: 1,
            stride: 0,
            last_attn: None,
            importance: None,
        };
        assert!(s.keep_spec(&ctx).is_none());
    }

    /// Per-head: each KV head keeps its own top-`cache_size` by last-step attention (the reference
    /// granularity); `on_phase` stages `keep_per_head`, not layer-wide.
    #[test]
    fn per_head_ranks_by_last_step_attention() {
        let s = Tova::from_args(StageParams::default(), &cache_size("3"));
        // head 0's last query attends most to 5,6,7; head 1 to 10,11,12.
        let (n_kv_heads, stride) = (2usize, 100usize);
        let mut aw = vec![0.0f32; n_kv_heads * stride];
        for (i, &pos) in [5usize, 6, 7].iter().enumerate() {
            aw[pos] = 10.0 - i as f32;
        }
        for (i, &pos) in [10usize, 11, 12].iter().enumerate() {
            aw[stride + pos] = 10.0 - i as f32;
        }
        let ctx = Ctx {
            current: 20,
            n_kv_heads,
            stride,
            last_attn: Some(aw),
            importance: None,
        };
        let expected = match s.keep_spec(&ctx).unwrap() {
            KeepSpec::PerHead(h) => h,
            KeepSpec::LayerWide(_) => panic!("expected PerHead"),
        };
        // cache_size=3, no recent window → each head keeps exactly its top-3.
        assert_eq!(expected[0], vec![5, 6, 7]);
        assert_eq!(expected[1], vec![10, 11, 12]);

        let mut h = CaptureHandle {
            cur: 20,
            n_kv: n_kv_heads,
            ..Default::default()
        };
        <Tova as KVMutationStage>::on_phase(&s, &ctx, &mut h).unwrap();
        assert_eq!(h.kept_per_head, Some(expected));
        assert_eq!(h.kept, None, "per-head path must NOT use layer-wide keep");
    }

    /// Flat fallback (no per-head last-step attention): rank heavy hitters by cumulative importance.
    #[test]
    fn flat_fallback_uses_importance() {
        let s = Tova::from_args(StageParams::default(), &cache_size("2"));
        let mut imp = vec![0.0f32; 20];
        imp[5] = 10.0;
        imp[9] = 9.0;
        imp[2] = 1.0; // lower — must NOT be kept
        let ctx = Ctx {
            current: 20,
            n_kv_heads: 1,
            stride: 0,
            last_attn: None, // no AttnWeights → flat path
            importance: Some(imp),
        };
        match s.keep_spec(&ctx) {
            // cache_size=2, recent=0 → keep the top-2 heavy hitters only.
            Some(KeepSpec::LayerWide(keep)) => assert_eq!(keep, vec![5, 9]),
            other => panic!("expected LayerWide, got {other:?}"),
        }
    }

    /// Score-free (no attention, no importance): the whole budget goes to recency.
    #[test]
    fn score_free_keeps_recency() {
        let s = Tova::from_args(StageParams::default(), &cache_size("4"));
        let ctx = Ctx {
            current: 20,
            n_kv_heads: 1,
            stride: 0,
            last_attn: None,
            importance: None,
        };
        match s.keep_spec(&ctx) {
            Some(KeepSpec::LayerWide(keep)) => assert_eq!(keep, vec![16, 17, 18, 19]),
            other => panic!("expected LayerWide recency, got {other:?}"),
        }
    }

    /// v3 native registration + DECISION equivalence: `on_phase` stages exactly what `keep_spec`
    /// decides (the registry make + caps are the ones the engine resolves by name).
    #[test]
    fn v3_native_registration_and_decision() {
        let reg = find_mutation_stage("tova").expect("tova in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "tova");
        assert_eq!(reg.phase, MutationPhase::KvMutate);
        assert_eq!(reg.caps, TOVA_CAPS);
        assert_eq!(
            (reg.make)(StageParams::default(), &cache_size("4")).name(),
            "tova"
        );

        let s = Tova::from_args(StageParams::default(), &cache_size("4"));
        let imp: Vec<f32> = (0..20).map(|i| (i % 7) as f32).collect();
        let ctx = Ctx {
            current: 20,
            n_kv_heads: 1,
            stride: 0,
            last_attn: None,
            importance: Some(imp),
        };
        let mut h = CaptureHandle {
            cur: 20,
            n_kv: 1,
            ..Default::default()
        };
        <Tova as KVMutationStage>::on_phase(&s, &ctx, &mut h).unwrap();
        let expected = match s.keep_spec(&ctx) {
            Some(KeepSpec::LayerWide(k)) => Some(k),
            _ => unreachable!(),
        };
        assert_eq!(h.kept, expected);
    }
}
