//! Algorithm-parity tests against the NVIDIA kvpress `PyramidKVPress` reference.
//!
//! Two oracles, both regenerable from `reference/*.py` (verbatim ports of kvpress):
//!
//! * `budget_grid.csv` — `get_layer_budget` over a grid; asserted byte-identically (the budgets
//!   are integers and the `f64` arithmetic + round-half-to-even match Python exactly).
//! * `select_fixture.txt` — per-kv-head keep sets from the SnapKV score→topk pipeline on integer
//!   (LCG) attention; the Rust test regenerates the identical attention via the same LCG and
//!   asserts the keep sets. Integer pooled-sums make the ranking f32/f64-agnostic, and BOTH sides
//!   break ties lower-index-first — a deterministic parity check of the *score pipeline*, NOT of
//!   torch.topk's exact tie order (which is implementation-defined, not lower-index-first; see the
//!   residuals in `lib.rs`). Every case uses `n_kept ≥ window_size` (window fully resident); the
//!   sub-window budget path (`lib.rs` D3) is covered by `sub_window_budget_keeps_raw_count`.

use super::*;
use argus_extension_api::{TensorDtype, TensorHandle, TensorKind, TensorShape};

// ── shared deterministic attention generator (mirrors reference/pyramidkv_select_ref.py) ──

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

// ── mock StageCtx supplying PrefillAttention (+ layer info, importance) ──

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
    layer_idx: usize,
    n_layers: usize,
    n_kv_heads: usize,
    pfa: Option<PfaHandle>,
    importance: Option<Vec<f32>>,
}
impl Default for Ctx {
    fn default() -> Self {
        Ctx {
            current: 0,
            target: 0,
            layer_idx: 0,
            n_layers: 1,
            n_kv_heads: 1,
            pfa: None,
            importance: None,
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
        self.layer_idx
    }
    fn n_layers(&self) -> usize {
        self.n_layers
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

fn cr_args(cr: &str, window: usize, kernel: usize, beta: u32) -> Vec<(String, String)> {
    vec![
        ("compression_ratio".to_string(), cr.to_string()),
        ("window_size".to_string(), window.to_string()),
        ("kernel_size".to_string(), kernel.to_string()),
        ("beta".to_string(), beta.to_string()),
    ]
}

/// Build a concrete `PyramidKv` from CLI-style args and return its v3 keep-set decision — the
/// shared `keep_spec` both `on_phase` arms route through (replaces the removed v2 `plan()` in the
/// shape tests below).
fn keep_spec_for(args: &[(String, String)], ctx: &Ctx) -> Option<KeepSpec> {
    let blob: Vec<argus_extension_api::PluginArg> = args
        .iter()
        .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
        .collect();
    let stage = PyramidKv::new(PyramidKvConfig::from_args(StageParams::default(), &blob));
    stage.keep_spec(ctx)
}

// ── 1. per-layer budget == kvpress get_layer_budget (CSV oracle) ──────────────

#[test]
fn budget_matches_kvpress_grid() {
    let csv = include_str!("../tests/fixtures/budget_grid.csv");
    let mut checked = 0usize;
    for line in csv.lines().skip(1).filter(|l| !l.is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 7, "bad row: {line}");
        let q_len: usize = f[0].parse().unwrap();
        let cr: f64 = f[1].parse().unwrap(); // "0.1" parses to the same f64 Python used
        let window: usize = f[2].parse().unwrap();
        let beta: u32 = f[3].parse().unwrap();
        let num_layers: usize = f[4].parse().unwrap();
        let layer_idx: usize = f[5].parse().unwrap();
        let want: usize = f[6].parse().unwrap();
        let got = get_layer_budget(q_len, cr, window, beta, num_layers, layer_idx);
        assert_eq!(
            got, want,
            "get_layer_budget(q={q_len}, cr={cr}, w={window}, beta={beta}, nl={num_layers}, li={layer_idx})"
        );
        checked += 1;
    }
    assert!(checked > 3000, "expected the full grid, checked {checked}");
}

#[test]
fn budget_pyramid_endpoints_average_to_uniform() {
    // The layer budgets form an arithmetic sequence whose mean is the uniform SnapKV budget
    // q_len*(1-cr) (the comment-derived invariant in kvpress). q=1024, cr=0.5 → mean 512.
    let (q, cr, w, beta, nl) = (1024usize, 0.5, 64usize, 20u32, 32usize);
    let first = get_layer_budget(q, cr, w, beta, nl, 0);
    let last = get_layer_budget(q, cr, w, beta, nl, nl - 1);
    assert_eq!(first, 960);
    assert_eq!(last, 64); // == window_size (min_num floor)
    assert_eq!((first + last) / 2, (q as f64 * (1.0 - cr)) as usize); // 512
}

// ── 2b. D3: budget below / at the observation window keeps the RAW count ──────

#[test]
fn sub_window_budget_keeps_raw_count() {
    // High-CR degenerate case (SnapKV-uniform fallback) — get_layer_budget rounds BELOW the window.
    // kvpress keeps EXACTLY n_kept positions; the former floor-to-window kept `window` instead (the
    // D3 over-compression divergence). q=128, cr=0.9, window=64, beta=20, 32 layers → raw budget 13.
    let current = 128usize;
    let window = 64usize;
    let raw = get_layer_budget(current, 0.9, window, 20, 32, 0);
    assert_eq!(raw, 13, "ground-truth kvpress budget for this config");
    assert!(raw < window, "must exercise the sub-window branch");

    let ctx = Ctx {
        current,
        n_layers: 32,
        ..Ctx::default()
    };
    match keep_spec_for(&cr_args("0.9", window, 5, 20), &ctx) {
        Some(KeepSpec::LayerWide(keep)) => {
            assert_eq!(
                keep.len(),
                raw,
                "keeps the raw budget (13), NOT the window floor (64)"
            );
            assert_eq!(
                keep,
                (current - raw..current).collect::<Vec<usize>>(),
                "the n_kept most-recent positions"
            );
        }
        _ => panic!("expected LayerWide of the 13 most-recent positions"),
    }
}

// ── 3. per-head SnapKV selection == kvpress (fixture oracle) ──────────────────

#[test]
fn per_head_selection_matches_kvpress_fixture() {
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
        let n_kept: usize = h[6].parse().unwrap();
        let seed: i64 = h[7].parse().unwrap();

        let mut expected: Vec<Vec<usize>> = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let kl = lines.next().expect("KEEP line");
            let mut it = kl.split_whitespace();
            assert_eq!(it.next(), Some("KEEP"));
            expected.push(it.map(|x| x.parse().unwrap()).collect());
        }

        let attn = synth_attn(n_q, k_len, seed);
        let got = per_head_keep(
            |qh, out| {
                let base = qh * k_len;
                out.copy_from_slice(&attn[base..base + k_len]);
            },
            n_q,
            n_kv,
            k_len,
            k_len,
            window,
            kernel,
            n_kept - window,
        );
        assert_eq!(
            got, expected,
            "case kv={n_kv} q={n_q} L={k_len} seed={seed}"
        );
        // every head keeps exactly the budget (the engine's equal-length PerHead invariant).
        for head in &got {
            assert_eq!(head.len(), n_kept);
        }
        cases += 1;
    }
    assert_eq!(cases, 7);
}

#[test]
fn per_head_keep_hand_traced() {
    // n_q=n_kv=1 (groups=1), kernel=1 (no pooling), window=2, heavy=2, current=6.
    // heavy region [0,4) scores ∝ attn (÷window is monotone): attn=[10,50,30,40].
    // ranking 1(50)>3(40)>2(30)>0(10) → top-2 = {1,3}; window=[4,5]. keep={1,3,4,5}.
    let attn = [10.0f32, 50.0, 30.0, 40.0, 7.0, 9.0];
    let got = per_head_keep(|_qh, out| out.copy_from_slice(&attn), 1, 1, 6, 6, 2, 1, 2);
    assert_eq!(got, vec![vec![1usize, 3, 4, 5]]);
}

// ── v3 native (imperative) decision ──────────────────────────────────────────

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

/// v3 native registration (PrefillEnd phase) + per-head decision: the v3 `on_phase` stages the
/// per-head SnapKV keep-set via `keep_per_head`. The expected keep is an independent
/// `get_layer_budget` × `per_head_keep` composition.
#[test]
fn v3_native_per_head_decision() {
    let reg = argus_extension_api::find_mutation_stage("pyramidkv")
        .expect("pyramidkv in KV_MUTATION_STAGES");
    assert_eq!(reg.name, "pyramidkv");
    assert_eq!(reg.phase, MutationPhase::PrefillEnd);
    assert!(reg.caps.reads.contains(&TensorKind::PrefillAttention));
    // kvpress PyramidKV protects no prefix; drop-only.
    assert_eq!(reg.caps.default_protected_prefix, 0);
    assert!(!reg.caps.produces_merge_plan);

    let (current, n_kv, n_q) = (512usize, 4usize, 4usize);
    let (window, kernel, beta, n_layers, layer_idx) = (8usize, 5usize, 20u32, 16usize, 3usize);
    let attn = synth_attn(n_q, current, 123);
    let args = cr_args("0.5", window, kernel, beta);
    let ctx = Ctx {
        current,
        target: 0,
        layer_idx,
        n_layers,
        n_kv_heads: n_kv,
        pfa: Some(PfaHandle {
            data: attn.clone(),
            rows: n_q,
            cols: current,
        }),
        ..Default::default()
    };

    // Independent oracle: budget (get_layer_budget) × per-head selection (per_head_keep).
    let n_kept = get_layer_budget(current, 0.5, window, beta, n_layers, layer_idx);
    assert!(n_kept > window && n_kept < current, "n_kept={n_kept}");
    let expected = per_head_keep(
        |qh, out| {
            let base = qh * current;
            out.copy_from_slice(&attn[base..base + current]);
        },
        n_q,
        n_kv,
        current,
        current,
        window,
        kernel,
        n_kept - window,
    );

    // v3 decision via a concrete PyramidKv built from the same config.
    let blob: Vec<argus_extension_api::PluginArg> = args
        .iter()
        .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
        .collect();
    let v3 = PyramidKv::new(PyramidKvConfig::from_args(StageParams::default(), &blob));
    let mut h = CaptureHandle {
        cur: current,
        n_kv,
        ..Default::default()
    };
    <PyramidKv as KVMutationStage>::on_phase(&v3, &ctx, &mut h).unwrap();
    assert_eq!(h.kept_per_head, Some(expected));
    for head in h.kept_per_head.as_ref().unwrap() {
        assert_eq!(head.len(), n_kept); // equal-length PerHead invariant
    }
    assert_eq!(h.kept, None, "per-head path uses keep_per_head, not keep");
}

/// v3 DECISION on the OTHER on_phase arms: the layer-wide degraded fallback (no PFA, flat
/// importance) stages `cache.keep`; and the no-op (cr==0) stages nothing.
#[test]
fn v3_native_layerwide_and_noop_arms() {
    let blob_args = cr_args("0.5", 8, 5, 20);
    let blob: Vec<argus_extension_api::PluginArg> = blob_args
        .iter()
        .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
        .collect();
    let v3 = PyramidKv::new(PyramidKvConfig::from_args(StageParams::default(), &blob));

    // (1) layer-wide fallback: no PFA but flat importance present → LayerWide keep.
    let lw_ctx = Ctx {
        current: 64,
        target: 0,
        layer_idx: 3,
        n_layers: 16,
        n_kv_heads: 4,
        pfa: None,
        importance: Some((0..64).map(|i| (i % 11) as f32).collect()),
    };
    let expected_lw = match keep_spec_for(&blob_args, &lw_ctx).unwrap() {
        KeepSpec::LayerWide(k) => k,
        KeepSpec::PerHead(_) => panic!("no-PFA fallback is layer-wide"),
    };
    let mut h = CaptureHandle {
        cur: 64,
        n_kv: 4,
        ..Default::default()
    };
    <PyramidKv as KVMutationStage>::on_phase(&v3, &lw_ctx, &mut h).unwrap();
    assert_eq!(h.kept, Some(expected_lw));
    assert_eq!(h.kept_per_head, None, "fallback uses layer-wide keep");

    // (2) no-op: compression_ratio == 0 (no explicit cr, no target) → plan None, on_phase stages nothing.
    let noop_v3 = PyramidKv::new(PyramidKvConfig::from_args(StageParams::default(), &[]));
    let noop_ctx = Ctx {
        current: 64,
        target: 0,
        layer_idx: 0,
        n_layers: 16,
        n_kv_heads: 4,
        pfa: None,
        importance: None,
    };
    assert!(keep_spec_for(&[], &noop_ctx).is_none(), "cr==0 → no-op");
    let mut h2 = CaptureHandle {
        cur: 64,
        n_kv: 4,
        ..Default::default()
    };
    <PyramidKv as KVMutationStage>::on_phase(&noop_v3, &noop_ctx, &mut h2).unwrap();
    assert_eq!(h2.kept, None);
    assert_eq!(h2.kept_per_head, None);
}

// ── 5. edge cases / fallbacks ────────────────────────────────────────────────

#[test]
fn zero_compression_is_noop() {
    // No explicit compression_ratio and no engine target_len ⇒ cr=0 ⇒ None (kvpress: cr==0 no-op).
    let ctx = Ctx {
        current: 128,
        target: 0,
        n_kv_heads: 4,
        pfa: Some(PfaHandle {
            data: synth_attn(4, 128, 1),
            rows: 4,
            cols: 128,
        }),
        ..Default::default()
    };
    assert!(keep_spec_for(&[], &ctx).is_none());
}

#[test]
fn budget_covers_everything_is_noop() {
    // Tiny compression ⇒ n_kept >= current ⇒ nothing to evict.
    let ctx = Ctx {
        current: 50, // n_kept = round(50*0.99)=50 >= current
        n_layers: 1,
        n_kv_heads: 2,
        pfa: Some(PfaHandle {
            data: synth_attn(2, 50, 2),
            rows: 2,
            cols: 50,
        }),
        ..Default::default()
    };
    assert!(keep_spec_for(&cr_args("0.01", 8, 5, 20), &ctx).is_none());
}

#[test]
fn compression_ratio_derived_from_target_len() {
    // No explicit compression_ratio: derive cr = 1 - target_len/current. current=200, target=120
    // ⇒ cr=0.4. With n_layers=1 the budget falls back to uniform round(200*0.6)=120 = target.
    let args = [("window_size".to_string(), "8".to_string())];
    let ctx = Ctx {
        current: 200,
        target: 120,
        n_layers: 1,
        n_kv_heads: 2,
        pfa: Some(PfaHandle {
            data: synth_attn(2, 200, 3),
            rows: 2,
            cols: 200,
        }),
        ..Default::default()
    };
    match keep_spec_for(&args, &ctx).expect("keep Some") {
        KeepSpec::PerHead(heads) => assert_eq!(heads[0].len(), 120),
        KeepSpec::LayerWide(_) => panic!("expected PerHead"),
    }
}

#[test]
fn degraded_layerwide_fallback_without_pfa() {
    // No PFA handle but flat importance present ⇒ layer-wide pyramid-budgeted H2O-style keep.
    let mut imp = vec![0.0f32; 128];
    for (i, &p) in [10usize, 20, 30, 40].iter().enumerate() {
        imp[p] = 100.0 - i as f32;
    }
    let ctx = Ctx {
        current: 128,
        n_layers: 1, // uniform budget round(128*0.5)=64
        n_kv_heads: 4,
        pfa: None,
        importance: Some(imp),
        ..Default::default()
    };
    match keep_spec_for(&cr_args("0.5", 8, 5, 20), &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => {
            assert_eq!(k.len(), 64);
            // window (last 8) always kept; the high-importance positions kept as heavy hitters.
            for p in 120..128 {
                assert!(k.contains(&p));
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
    // No PFA, no importance ⇒ recency: keep the most-recent n_kept, layer-wide.
    let ctx = Ctx {
        current: 128,
        n_layers: 1, // n_kept=64
        n_kv_heads: 4,
        pfa: None,
        importance: None,
        ..Default::default()
    };
    match keep_spec_for(&cr_args("0.5", 8, 5, 20), &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => assert_eq!(k, (64..128).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("expected LayerWide recency"),
    }
}

// ── 6. adversarial-review regression tests ───────────────────────────────────

#[test]
fn degenerate_cr_floors_to_one_never_empty() {
    // Adversarial finding: a degenerate compression_ratio makes the uniform fallback budget round to
    // 0. kvpress's `topk(0)` would EMPTY the cache; we floor to 1 — the minimal safe non-empty keep
    // — and keep the single most-recent token, NOT the window (the old floor-to-window was the D3
    // over-compression divergence). current=64, window=8, cr≈1 ⇒ raw budget round(64·1e-6)=0 →
    // floored to 1 → keep [63] (non-empty).
    let ctx = Ctx {
        current: 64,
        n_layers: 1,
        n_kv_heads: 2,
        pfa: Some(PfaHandle {
            data: synth_attn(2, 64, 7),
            rows: 2,
            cols: 64,
        }),
        ..Default::default()
    };
    let keep = keep_spec_for(&cr_args("0.999999", 8, 5, 20), &ctx)
        .expect("keep Some (must evict, not empty)");
    match keep {
        KeepSpec::LayerWide(k) => {
            assert_eq!(k, vec![63], "floored to 1 — the single most-recent token");
            assert!(!k.is_empty(), "must never produce an empty keep-list");
        }
        KeepSpec::PerHead(_) => panic!("sub-window keep is LayerWide"),
    }
}

#[test]
fn empty_cache_is_noop_no_panic() {
    // Adversarial finding (D3-A): `raw_budget.clamp(1, current)` panics when current==0 (min>max).
    // An empty cache (current==0) with cr>0 must be a SAFE no-op (None), not a panic.
    let ctx = Ctx {
        current: 0,
        n_layers: 32,
        ..Ctx::default()
    };
    assert_eq!(
        keep_spec_for(&cr_args("0.9", 64, 5, 20), &ctx),
        None,
        "empty cache must be a no-op, never a panic"
    );
}

#[test]
fn even_kernel_is_forced_odd() {
    // Adversarial finding: even kernels give asymmetric/length-mismatched pooling. The config forces
    // odd (2→3). So kernel_size=2 must produce the SAME plan as kernel_size=3 (both → 3).
    let current = 256usize;
    let n_kv = 2usize;
    let n_q = 4usize;
    let attn = synth_attn(n_q, current, 314);
    let mk = |k: usize| {
        let ctx = Ctx {
            current,
            n_layers: 8,
            layer_idx: 2,
            n_kv_heads: n_kv,
            pfa: Some(PfaHandle {
                data: attn.clone(),
                rows: n_q,
                cols: current,
            }),
            ..Default::default()
        };
        keep_spec_for(&cr_args("0.5", 8, k, 20), &ctx).expect("keep Some")
    };
    assert_eq!(mk(2), mk(3), "kernel_size=2 must be forced to 3");
    // sanity: 3 and 5 differ (pooling actually matters), so the equality above isn't trivial.
    assert_ne!(mk(3), mk(5));
}

#[test]
fn pfa_invalid_gqa_falls_back_to_layerwide() {
    // Adversarial coverage gap: PFA present but n_q (3) not a multiple of n_kv (2) ⇒ per-head SnapKV
    // is undefined, so the stage must DEGRADE to the layer-wide path (here, recency — no importance),
    // never emit a malformed PerHead keep.
    let ctx = Ctx {
        current: 128,
        n_layers: 1, // n_kept = 64 > window
        n_kv_heads: 2,
        pfa: Some(PfaHandle {
            data: synth_attn(3, 128, 9), // n_q = 3, not a multiple of n_kv = 2
            rows: 3,
            cols: 128,
        }),
        ..Default::default()
    };
    match keep_spec_for(&cr_args("0.5", 8, 5, 20), &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => assert_eq!(k, (64..128).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("invalid GQA geometry must not produce PerHead"),
    }
}

#[test]
fn window_only_keep_when_budget_equals_window() {
    // The n_kept == window boundary (after the floor) keeps exactly the recent window, layer-wide —
    // matching kvpress's window-forced set when the budget equals the window. Pick params where the
    // uniform fallback lands at the window: current=80, cr=0.9, window=8 ⇒ round(80*0.1)=8.
    let ctx = Ctx {
        current: 80,
        n_layers: 1,
        n_kv_heads: 2,
        pfa: Some(PfaHandle {
            data: synth_attn(2, 80, 5),
            rows: 2,
            cols: 80,
        }),
        ..Default::default()
    };
    match keep_spec_for(&cr_args("0.9", 8, 5, 20), &ctx).expect("keep Some") {
        KeepSpec::LayerWide(k) => assert_eq!(k, (72..80).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("budget==window is the layer-wide window-only keep"),
    }
}
