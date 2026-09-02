//! Parity tests against the NVIDIA kvpress `SnapKVPress` reference, plus the stage's decision
//! shapes. Two oracles, both regenerable from `reference/*.py`:
//!
//! * `budget_grid.csv` — `int(k_len·(1−cr))` over a grid; asserted byte-identically (the `f64`
//!   product is computed in the same order and `as usize` truncates like Python `int`).
//! * `select_fixture.txt` — the STAGE's per-kv-head keep sets (this budget × the shared SnapKV
//!   selection) on integer (LCG) attention; the Rust test regenerates the attention via the same LCG
//!   and drives the real stage through `keep_spec`. The selection pipeline itself is pinned by
//!   pyramidkv's fixture (the same `snapkv_per_head_keep`); this fixture pins its composition with
//!   THIS budget. Every case uses `n_kept ≥ window_size`; the sub-window path is covered by
//!   `sub_window_budget_keeps_raw_count`.

use super::*;
use argus_extension_api::{TensorDtype, TensorHandle, TensorKind, TensorShape};

// ── shared deterministic attention generator (mirrors pyramidkv/reference/pyramidkv_select_ref.py) ──

const LCG_A: i64 = 1_103_515_245;
const LCG_C: i64 = 12_345;
const LCG_MASK: i64 = 0x7FFF_FFFF;

/// Integer attention matrix `[n_q_heads * k_len]`, LCG row-major (head outer, pos inner), state
/// continuous across heads — byte-identical to the Python reference's `synth_attn`.
fn synth_attn(n_q_heads: usize, k_len: usize, seed: i64) -> Vec<f32> {
    let mut data = Vec::with_capacity(n_q_heads * k_len);
    let mut s = seed;
    for _ in 0..n_q_heads {
        for _ in 0..k_len {
            s = (LCG_A * s + LCG_C) & LCG_MASK;
            data.push((s % 1000) as f32);
        }
    }
    data
}

// ── mock StageCtx supplying PrefillAttention (+ importance) ──

struct PfaHandle {
    data: Vec<f32>,
    rows: usize, // n_heads_q (pre-GQA)
    cols: usize, // prefix_len
}
impl TensorHandle for PfaHandle {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.rows,
            cols: self.cols,
            per_head: false,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, _kv_head: usize, out: &mut [f32]) {
        let base = row * self.cols;
        let n = self.cols.min(out.len());
        out[..n].copy_from_slice(&self.data[base..base + n]);
    }
}

struct Ctx {
    current: usize,
    target: usize,
    n_kv_heads: usize,
    pfa: Option<PfaHandle>,
    importance: Option<Vec<f32>>,
    protected: usize,
}
impl Default for Ctx {
    fn default() -> Self {
        Ctx {
            current: 0,
            target: 0,
            n_kv_heads: 1,
            pfa: None,
            importance: None,
            protected: 0,
        }
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
    fn n_layers(&self) -> usize {
        1
    }
    fn protected_prefix(&self) -> usize {
        self.protected
    }
    fn importance(&self) -> Option<&[f32]> {
        self.importance.as_deref()
    }
    fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }
    fn head_dim(&self) -> usize {
        64
    }
    fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
        match kind {
            TensorKind::PrefillAttention => self.pfa.as_ref().map(|h| h as &dyn TensorHandle),
            _ => None,
        }
    }
}

fn pfa(n_q: usize, k_len: usize, seed: i64) -> Option<PfaHandle> {
    Some(PfaHandle {
        data: synth_attn(n_q, k_len, seed),
        rows: n_q,
        cols: k_len,
    })
}

fn cr_args(cr: &str, window: usize, kernel: usize) -> Vec<(String, String)> {
    vec![
        ("compression_ratio".to_string(), cr.to_string()),
        ("window_size".to_string(), window.to_string()),
        ("kernel_size".to_string(), kernel.to_string()),
    ]
}

/// Build a concrete `SnapKv` from CLI-style args (the `--set` blob).
fn stage_for(args: &[(String, String)]) -> SnapKv {
    let blob: Vec<argus_extension_api::PluginArg> = args
        .iter()
        .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
        .collect();
    SnapKv::new(SnapKvConfig::from_args(StageParams::default(), &blob))
}

fn keep_spec_for(args: &[(String, String)], ctx: &Ctx) -> Option<KeepSpec> {
    stage_for(args).keep_spec(ctx)
}

// ── 1. budget == kvpress ScorerPress.compress (CSV oracle) ───────────────────

#[test]
fn budget_matches_kvpress_grid() {
    let csv = include_str!("../tests/fixtures/budget_grid.csv");
    let mut checked = 0usize;
    for line in csv.lines().skip(1).filter(|l| !l.is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 3, "bad row: {line}");
        let k_len: usize = f[0].parse().unwrap();
        let cr: f64 = f[1].parse().unwrap(); // "0.1" parses to the same f64 Python used
        let want: usize = f[2].parse().unwrap();
        assert_eq!(
            snapkv_budget(k_len, cr),
            want,
            "snapkv_budget(k_len={k_len}, cr={cr})"
        );
        checked += 1;
    }
    assert!(checked > 90, "expected the full grid, checked {checked}");
}

/// The budget is Python `int()` — truncation — NOT `round`. PyramidKV's SnapKV fallback rounds
/// (half-to-even), so this is where the two presses' "uniform budget" part ways. Mutation-proof:
/// `round_ties_even()` gives 642 / 2 / 4 / 67 here.
#[test]
fn budget_truncates_like_python_int() {
    assert_eq!(snapkv_budget(1283, 0.5), 641); // 641.5 → 641 (round: 642)
    assert_eq!(snapkv_budget(3, 0.5), 1); // 1.5 → 1 (round-half-even: 2)
    assert_eq!(snapkv_budget(7, 0.5), 3); // 3.5 → 3 (round-half-even: 4)
    assert_eq!(snapkv_budget(100, 0.333), 66); // 66.7 → 66 (round: 67)
    assert_eq!(snapkv_budget(1281, 0.5), 640); // exact: 640.5 → 640 (round-half-even: 640, agrees)
    assert_eq!(snapkv_budget(64, 0.999_999), 0); // 6.4e-5 → 0 (the stage floors this to 1)
    // Truncation also bites on the f64 subtraction: `1 − 0.9` is 0.09999999999999998, so
    // `80 × (1 − 0.9)` = 7.999… keeps 7, not 8 — Python `int(80 * (1 - 0.9))` agrees.
    assert_eq!(snapkv_budget(80, 0.9), 7);
}

// ── 2. the STAGE's decision == kvpress SnapKVPress.compress (fixture oracle) ──

#[test]
fn stage_matches_kvpress_fixture() {
    let fixture = include_str!("../tests/fixtures/select_fixture.txt");
    let mut lines = fixture.lines().filter(|l| !l.is_empty()).peekable();
    let mut cases = 0usize;
    while let Some(header) = lines.next() {
        let h: Vec<&str> = header.split_whitespace().collect();
        assert_eq!(h[0], "CASE", "expected CASE header, got {header}");
        let n_kv: usize = h[1].parse().unwrap();
        let n_q: usize = h[2].parse().unwrap();
        let k_len: usize = h[3].parse().unwrap();
        let window: usize = h[4].parse().unwrap();
        let kernel: usize = h[5].parse().unwrap();
        let cr: &str = h[6];
        let seed: i64 = h[7].parse().unwrap();

        let mut expected: Vec<Vec<usize>> = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let kl = lines.next().expect("KEEP line");
            let mut it = kl.split_whitespace();
            assert_eq!(it.next(), Some("KEEP"));
            expected.push(it.map(|x| x.parse().unwrap()).collect());
        }

        // The real stage, explicit `compression_ratio`, faithful (protected prefix 0).
        let ctx = Ctx {
            current: k_len,
            n_kv_heads: n_kv,
            pfa: pfa(n_q, k_len, seed),
            ..Default::default()
        };
        let n_kept = snapkv_budget(k_len, cr.parse().unwrap());
        assert!(n_kept > window, "fixture cases keep more than the window");
        match keep_spec_for(&cr_args(cr, window, kernel), &ctx) {
            Some(KeepSpec::PerHead(heads)) => {
                assert_eq!(
                    heads, expected,
                    "case kv={n_kv} q={n_q} L={k_len} cr={cr} seed={seed}"
                );
                for head in &heads {
                    assert_eq!(head.len(), n_kept, "every head keeps exactly int(L·(1−cr))");
                }
            }
            other => panic!("expected PerHead for case seed={seed}, got {other:?}"),
        }
        cases += 1;
    }
    assert_eq!(cases, 8);
}

// ── 3. registration + the imperative surface ─────────────────────────────────

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

/// Registered as "snapkv": a PrefillEnd stage that reads the prefill attention at the SAME 64-query
/// window pyramidkv declares (a `--aperturb-select` pool refuses two PFA candidates with different
/// windows), protects no prefix (kvpress-faithful), drop-only.
#[test]
fn registration_is_a_prefill_end_pfa_reader_at_window_64() {
    let reg =
        argus_extension_api::find_mutation_stage("snapkv").expect("snapkv in KV_MUTATION_STAGES");
    assert_eq!(reg.name, "snapkv");
    assert_eq!(reg.phase, MutationPhase::PrefillEnd);
    assert!(reg.caps.reads.contains(&TensorKind::PrefillAttention));
    assert_eq!(reg.caps.prefill_attn_window, Some(64));
    assert_eq!(
        reg.caps.prefill_attn_window,
        Some(SnapKvConfig::default().window_size)
    );
    assert_eq!(reg.caps.default_protected_prefix, 0);
    assert!(!reg.caps.produces_merge_plan);
    assert!(!reg.caps.whole_model);
    let made = (reg.make)(StageParams::default(), &[]);
    assert_eq!(made.name(), "snapkv");
}

/// `on_phase` stages the per-head keep-set via `keep_per_head` (not `keep`), equal length per head.
#[test]
fn on_phase_stages_the_per_head_keep() {
    let (current, n_kv, n_q, window, kernel) = (512usize, 4usize, 8usize, 8usize, 5usize);
    let args = cr_args("0.5", window, kernel);
    let ctx = Ctx {
        current,
        n_kv_heads: n_kv,
        pfa: pfa(n_q, current, 123),
        ..Default::default()
    };
    let expected = match keep_spec_for(&args, &ctx) {
        Some(KeepSpec::PerHead(h)) => h,
        other => panic!("expected PerHead, got {other:?}"),
    };
    let stage = stage_for(&args);
    let mut h = CaptureHandle {
        cur: current,
        n_kv,
        ..Default::default()
    };
    <SnapKv as KVMutationStage>::on_phase(&stage, &ctx, &mut h).unwrap();
    assert_eq!(h.kept_per_head, Some(expected));
    for head in h.kept_per_head.as_ref().unwrap() {
        assert_eq!(head.len(), 256); // int(512·0.5), equal-length PerHead invariant
    }
    assert_eq!(h.kept, None, "per-head path uses keep_per_head, not keep");
}

// ── 4. the engine's target_len path ──────────────────────────────────────────

/// No explicit `compression_ratio`: the engine's `target_len` is the budget ITSELF. Deriving
/// `cr = 1 − target/current` and truncating `current·(1−cr)` would under-keep by one for most
/// pairs — 65 → 5 comes out as 4 — which is the mutation this test catches.
#[test]
fn engine_target_len_is_the_budget_itself() {
    // The round trip that must NOT be taken (f64, same as Python).
    let round_trip = (65.0f64 * (1.0 - (1.0 - 5.0f64 / 65.0))) as usize;
    assert_eq!(round_trip, 4, "the truncating cr round trip loses a token");

    // Sub-window regime: layer-wide, exactly the 5 most recent.
    let ctx = Ctx {
        current: 65,
        target: 5,
        n_kv_heads: 2,
        pfa: pfa(2, 65, 3),
        ..Default::default()
    };
    match keep_spec_for(&[], &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => assert_eq!(k, (60..65).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("target 5 ≤ window is layer-wide"),
    }

    // Per-head regime: current=200, target=120, window=8 → every head keeps exactly 120.
    let ctx = Ctx {
        current: 200,
        target: 120,
        n_kv_heads: 2,
        pfa: pfa(2, 200, 3),
        ..Default::default()
    };
    let args = [("window_size".to_string(), "8".to_string())];
    match keep_spec_for(&args, &ctx).expect("keep Some") {
        KeepSpec::PerHead(heads) => {
            for head in &heads {
                assert_eq!(head.len(), 120);
            }
        }
        KeepSpec::LayerWide(_) => panic!("expected PerHead"),
    }
}

// ── 5. edge cases / fallbacks (the pyramidkv branch structure, uniform budget) ──

#[test]
fn sub_window_budget_keeps_raw_count() {
    // High-cr degenerate case: the budget lands BELOW the observation window. kvpress keeps EXACTLY
    // n_kept positions (int(128·0.1) = 12); flooring to the window would keep 64 instead.
    let current = 128usize;
    let window = 64usize;
    let raw = snapkv_budget(current, 0.9);
    assert_eq!(raw, 12);
    assert!(raw < window, "must exercise the sub-window branch");
    let ctx = Ctx {
        current,
        ..Ctx::default()
    };
    match keep_spec_for(&cr_args("0.9", window, 5), &ctx) {
        Some(KeepSpec::LayerWide(keep)) => {
            assert_eq!(keep, (current - raw..current).collect::<Vec<usize>>());
        }
        _ => panic!("expected LayerWide of the 12 most-recent positions"),
    }
}

#[test]
fn window_only_keep_when_budget_equals_window() {
    // int(64·0.25) = 16 == window (an exact f64 product) → exactly the recent window, layer-wide
    // (kvpress's window-forced set).
    let ctx = Ctx {
        current: 64,
        n_kv_heads: 2,
        pfa: pfa(2, 64, 5),
        ..Default::default()
    };
    match keep_spec_for(&cr_args("0.75", 16, 5), &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => assert_eq!(k, (48..64).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("budget==window is the layer-wide window-only keep"),
    }
}

#[test]
fn zero_compression_is_noop() {
    // No explicit compression_ratio and no engine target ⇒ no-op (kvpress: cr==0 no-op).
    let ctx = Ctx {
        current: 128,
        n_kv_heads: 4,
        pfa: pfa(4, 128, 1),
        ..Default::default()
    };
    assert!(keep_spec_for(&[], &ctx).is_none());
    // A target that covers the cache (≥ current) is a no-op too.
    let ctx = Ctx {
        current: 128,
        target: 128,
        ..ctx
    };
    assert!(keep_spec_for(&[], &ctx).is_none());
    // And `on_phase` stages nothing.
    let mut h = CaptureHandle {
        cur: 128,
        n_kv: 4,
        ..Default::default()
    };
    <SnapKv as KVMutationStage>::on_phase(&stage_for(&[]), &ctx, &mut h).unwrap();
    assert_eq!(h.kept, None);
    assert_eq!(h.kept_per_head, None);
}

#[test]
fn empty_cache_is_noop_no_panic() {
    // `raw_budget.clamp(1, current)` panics when current==0 (min>max): an empty cache with cr>0
    // must be a SAFE no-op (None), not a panic.
    let ctx = Ctx {
        current: 0,
        ..Ctx::default()
    };
    assert!(keep_spec_for(&cr_args("0.9", 64, 5), &ctx).is_none());
}

#[test]
fn degenerate_cr_floors_to_one_never_empty() {
    // cr≈1 ⇒ int(64·1e-6) = 0. kvpress's `topk(0)` would EMPTY the cache; we floor to 1 and keep
    // the single most-recent token (not the window — that was the D3 over-compression divergence).
    let ctx = Ctx {
        current: 64,
        n_kv_heads: 2,
        pfa: pfa(2, 64, 7),
        ..Default::default()
    };
    match keep_spec_for(&cr_args("0.999999", 8, 5), &ctx).expect("must evict, not empty") {
        KeepSpec::LayerWide(k) => assert_eq!(k, vec![63]),
        KeepSpec::PerHead(_) => panic!("sub-window keep is LayerWide"),
    }
}

#[test]
fn degraded_layerwide_fallback_without_pfa() {
    // No PFA but flat importance present ⇒ layer-wide budget, H2O-style: window + heavy hitters.
    let mut imp = vec![0.0f32; 128];
    for (i, &p) in [10usize, 20, 30, 40].iter().enumerate() {
        imp[p] = 100.0 - i as f32;
    }
    let ctx = Ctx {
        current: 128,
        n_kv_heads: 4,
        importance: Some(imp),
        ..Default::default()
    };
    match keep_spec_for(&cr_args("0.5", 8, 5), &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => {
            assert_eq!(k.len(), 64);
            for p in 120..128 {
                assert!(k.contains(&p), "window position {p} kept");
            }
            for p in [10usize, 20, 30, 40] {
                assert!(k.contains(&p), "heavy hitter {p} kept");
            }
        }
        KeepSpec::PerHead(_) => panic!("expected LayerWide degraded fallback"),
    }
}

#[test]
fn recency_fallback_without_any_scores() {
    // No PFA, no importance ⇒ recency: the most-recent n_kept, layer-wide.
    let ctx = Ctx {
        current: 128,
        n_kv_heads: 4,
        ..Default::default()
    };
    match keep_spec_for(&cr_args("0.5", 8, 5), &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => assert_eq!(k, (64..128).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("expected LayerWide recency"),
    }
}

#[test]
fn protected_prefix_keeps_sink_per_head_equal_length() {
    // `--protected-prefix` (the attention-sink guard, 4 in a `--aperturb-select` pool): force-kept
    // in EVERY head, ADDITIVE to the budget, and every head still keeps an equal count.
    let ctx = Ctx {
        current: 200,
        target: 120,
        n_kv_heads: 2,
        pfa: pfa(2, 200, 3),
        protected: 4,
        ..Default::default()
    };
    let args = [("window_size".to_string(), "8".to_string())];
    match keep_spec_for(&args, &ctx).expect("keep Some") {
        KeepSpec::PerHead(heads) => {
            for (h, keep) in heads.iter().enumerate() {
                assert_eq!(keep.len(), 124, "head {h}: 120 + 4 protected");
                for p in 0..4 {
                    assert!(
                        keep.contains(&p),
                        "head {h} protected sink pos {p} must survive"
                    );
                }
            }
        }
        KeepSpec::LayerWide(_) => panic!("expected PerHead"),
    }
    // Sub-window regime: the prefix is unioned in front of the recency keep.
    let ctx = Ctx {
        current: 40,
        target: 20,
        n_kv_heads: 2,
        protected: 4,
        ..Default::default()
    };
    match keep_spec_for(&[], &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => {
            assert_eq!(k.len(), 24);
            assert_eq!(&k[..4], &[0, 1, 2, 3]);
            assert_eq!(&k[4..], &(20..40).collect::<Vec<_>>()[..]);
        }
        KeepSpec::PerHead(_) => panic!("sub-window recency is layer-wide"),
    }
}

#[test]
fn pfa_invalid_gqa_falls_back_to_layerwide() {
    // PFA present but n_q (3) is not a multiple of n_kv (2): per-head SnapKV is undefined, so the
    // stage DEGRADES to the layer-wide path (recency here) — never a malformed PerHead keep.
    let ctx = Ctx {
        current: 128,
        n_kv_heads: 2,
        pfa: pfa(3, 128, 9),
        ..Default::default()
    };
    match keep_spec_for(&cr_args("0.5", 8, 5), &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => assert_eq!(k, (64..128).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("invalid GQA geometry must not produce PerHead"),
    }
}

#[test]
fn even_kernel_is_forced_odd() {
    // Even kernels pool asymmetrically; the config forces odd (2→3), so kernel_size=2 must decide
    // exactly as kernel_size=3, and 3 must differ from 5 (pooling actually matters).
    let (current, n_kv, n_q) = (256usize, 2usize, 4usize);
    let attn = synth_attn(n_q, current, 314);
    let mk = |k: usize| {
        let ctx = Ctx {
            current,
            n_kv_heads: n_kv,
            pfa: Some(PfaHandle {
                data: attn.clone(),
                rows: n_q,
                cols: current,
            }),
            ..Default::default()
        };
        keep_spec_for(&cr_args("0.5", 8, k), &ctx).expect("keep Some")
    };
    assert_eq!(mk(2), mk(3), "kernel_size=2 must be forced to 3");
    assert_ne!(mk(3), mk(5));
}
