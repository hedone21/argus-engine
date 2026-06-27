//! Algorithm-parity tests against the NVIDIA kvpress `PyramidKVPress` reference.
//!
//! Two oracles, both regenerable from `reference/*.py` (verbatim ports of kvpress):
//!
//! * `budget_grid.csv` — `get_layer_budget` over a grid; asserted byte-identically (the budgets
//!   are integers and the `f64` arithmetic + round-half-to-even match Python exactly).
//! * `select_fixture.txt` — per-kv-head keep sets from the SnapKV score→topk pipeline on integer
//!   (LCG) attention; the Rust test regenerates the identical attention via the same LCG and
//!   asserts the keep sets. Integer pooled-sums make the ranking f32/f64-agnostic and ties are
//!   broken lower-index-first on both sides.

use super::*;
use argus_extension_api::{
    StageArgs, TensorDtype, TensorHandle, TensorKind, TensorShape, find_stage,
};

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

fn make_pyramidkv(args: &[(String, String)]) -> Box<dyn KVCacheStage> {
    let blob: Vec<argus_extension_api::PluginArg> = args
        .iter()
        .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
        .collect();
    let reg = find_stage("pyramidkv").expect("pyramidkv registered");
    (reg.make_with_args)(StageParams::default(), &blob as StageArgs)
}

// ── 1. registration ──────────────────────────────────────────────────────────

#[test]
fn registers_with_prefill_attention_caps() {
    let reg = find_stage("pyramidkv").expect("pyramidkv registered in KV_CACHE_STAGES");
    assert_eq!(reg.name, "pyramidkv");
    assert!(reg.caps.reads.contains(&TensorKind::PrefillAttention));
    // kvpress PyramidKV protects no prefix.
    assert_eq!(reg.caps.default_protected_prefix, 0);
    assert!(!reg.caps.produces_merge_plan);
}

// ── 2. per-layer budget == kvpress get_layer_budget (CSV oracle) ──────────────

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

// ── 4. full plan() wiring: budget + per-head selection → PerHead plan ──────────

#[test]
fn plan_wires_budget_to_per_head_selection() {
    // Pyramid branch (n_layers>1). plan() must use get_layer_budget for n_kept and per_head_keep
    // for the positions → assert it equals the composition computed here.
    let (current, n_kv, n_q) = (512usize, 4usize, 4usize);
    let (cr, window, kernel, beta, n_layers, layer_idx) =
        ("0.5", 8usize, 5usize, 20u32, 16usize, 3usize);
    let attn = synth_attn(n_q, current, 123);
    let stage = make_pyramidkv(&cr_args(cr, window, kernel, beta));

    let ctx = Ctx {
        current,
        target: 0, // explicit compression_ratio drives the budget; target_len unused
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
    let plan = stage.plan(&ctx).expect("plan Some");

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
    match plan.keep {
        KeepSpec::PerHead(heads) => {
            assert_eq!(heads, expected);
            for h in &heads {
                assert_eq!(h.len(), n_kept); // equal-length invariant
            }
        }
        KeepSpec::LayerWide(_) => panic!("expected PerHead when PFA is supplied"),
    }
    assert!(plan.merges.is_empty());
}

// ── v3 native (imperative) decision equivalence ──────────────────────────────

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

/// v3 native registration (PrefillEnd phase) + DECISION equivalence: the v3 `on_phase` stages the
/// per-head SnapKV keep the v2 `plan` returns (both via the shared `keep_spec`).
#[test]
fn v3_native_matches_v2_decision() {
    let reg = argus_extension_api::find_mutation_stage("pyramidkv")
        .expect("pyramidkv in KV_MUTATION_STAGES");
    assert_eq!(reg.name, "pyramidkv");
    assert_eq!(reg.phase, MutationPhase::PrefillEnd);
    assert!(reg.caps.reads.contains(&TensorKind::PrefillAttention));

    let (current, n_kv, n_q) = (512usize, 4usize, 4usize);
    let attn = synth_attn(n_q, current, 123);
    let args = cr_args("0.5", 8, 5, 20);
    let ctx = Ctx {
        current,
        target: 0,
        layer_idx: 3,
        n_layers: 16,
        n_kv_heads: n_kv,
        pfa: Some(PfaHandle {
            data: attn.clone(),
            rows: n_q,
            cols: current,
        }),
        ..Default::default()
    };
    // v2 decision via the registered (boxed) stage.
    let v2_heads = match make_pyramidkv(&args).plan(&ctx).unwrap().keep {
        KeepSpec::PerHead(h) => h,
        KeepSpec::LayerWide(_) => panic!("expected PerHead"),
    };
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
    assert_eq!(h.kept_per_head, Some(v2_heads));
    assert_eq!(h.kept, None, "per-head path uses keep_per_head, not keep");
}

// ── 5. edge cases / fallbacks ────────────────────────────────────────────────

#[test]
fn zero_compression_is_noop() {
    // No explicit compression_ratio and no engine target_len ⇒ cr=0 ⇒ None (kvpress: cr==0 no-op).
    let stage = make_pyramidkv(&[]);
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
    assert!(stage.plan(&ctx).is_none());
}

#[test]
fn budget_covers_everything_is_noop() {
    // Tiny compression ⇒ n_kept >= current ⇒ nothing to evict.
    let stage = make_pyramidkv(&cr_args("0.01", 8, 5, 20));
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
    assert!(stage.plan(&ctx).is_none());
}

#[test]
fn compression_ratio_derived_from_target_len() {
    // No explicit compression_ratio: derive cr = 1 - target_len/current. current=200, target=120
    // ⇒ cr=0.4. With n_layers=1 the budget falls back to uniform round(200*0.6)=120 = target.
    let stage = make_pyramidkv(&[("window_size".to_string(), "8".to_string())]);
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
    let plan = stage.plan(&ctx).expect("plan Some");
    match plan.keep {
        KeepSpec::PerHead(heads) => assert_eq!(heads[0].len(), 120),
        KeepSpec::LayerWide(_) => panic!("expected PerHead"),
    }
}

#[test]
fn degraded_layerwide_fallback_without_pfa() {
    // No PFA handle but flat importance present ⇒ layer-wide pyramid-budgeted H2O-style plan.
    let stage = make_pyramidkv(&cr_args("0.5", 8, 5, 20));
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
    let plan = stage.plan(&ctx).expect("plan Some");
    match plan.keep {
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
    let stage = make_pyramidkv(&cr_args("0.5", 8, 5, 20));
    let ctx = Ctx {
        current: 128,
        n_layers: 1, // n_kept=64
        n_kv_heads: 4,
        pfa: None,
        importance: None,
        ..Default::default()
    };
    let plan = stage.plan(&ctx).expect("plan Some");
    match plan.keep {
        KeepSpec::LayerWide(k) => assert_eq!(k, (64..128).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("expected LayerWide recency"),
    }
}

// ── 6. adversarial-review regression tests ───────────────────────────────────

#[test]
fn degenerate_cr_floors_to_window_never_empty() {
    // Adversarial finding: a degenerate compression_ratio could make the uniform fallback budget
    // round to 0, yielding an EMPTY keep-list that empties the cache. The window floor must keep at
    // least the recent window. current=64, window=8, cr≈1 ⇒ raw budget round(64*1e-6)=0 → floored
    // to window=8 → keep the last 8 (non-empty), NOT [].
    let stage = make_pyramidkv(&cr_args("0.999999", 8, 5, 20));
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
    let plan = stage.plan(&ctx).expect("plan Some (must evict, not empty)");
    match plan.keep {
        KeepSpec::LayerWide(k) => {
            assert_eq!(
                k,
                (56..64).collect::<Vec<_>>(),
                "floored to the recent window"
            );
            assert!(!k.is_empty(), "must never produce an empty keep-list");
        }
        KeepSpec::PerHead(_) => panic!("window-only keep is LayerWide"),
    }
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
        let stage = make_pyramidkv(&cr_args("0.5", 8, k, 20));
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
        stage.plan(&ctx).expect("plan Some")
    };
    assert_eq!(mk(2), mk(3), "kernel_size=2 must be forced to 3");
    // sanity: 3 and 5 differ (pooling actually matters), so the equality above isn't trivial.
    assert_ne!(mk(3), mk(5));
}

#[test]
fn pfa_invalid_gqa_falls_back_to_layerwide() {
    // Adversarial coverage gap: PFA present but n_q (3) not a multiple of n_kv (2) ⇒ per-head SnapKV
    // is undefined, so the stage must DEGRADE to the layer-wide path (here, recency — no importance),
    // never emit a malformed PerHead plan.
    let stage = make_pyramidkv(&cr_args("0.5", 8, 5, 20));
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
    let plan = stage.plan(&ctx).expect("plan Some");
    match plan.keep {
        KeepSpec::LayerWide(k) => assert_eq!(k, (64..128).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("invalid GQA geometry must not produce PerHead"),
    }
}

#[test]
fn window_only_keep_when_budget_equals_window() {
    // The n_kept == window boundary (after the floor) keeps exactly the recent window, layer-wide —
    // matching kvpress's window-forced set when the budget equals the window. Pick params where the
    // uniform fallback lands at the window: current=80, cr=0.9, window=8 ⇒ round(80*0.1)=8.
    let stage = make_pyramidkv(&cr_args("0.9", 8, 5, 20));
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
    let plan = stage.plan(&ctx).expect("plan Some");
    match plan.keep {
        KeepSpec::LayerWide(k) => assert_eq!(k, (72..80).collect::<Vec<_>>()),
        KeepSpec::PerHead(_) => panic!("budget==window is the layer-wide window-only keep"),
    }
}
