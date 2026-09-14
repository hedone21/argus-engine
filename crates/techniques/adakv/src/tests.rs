//! Parity tests against the FFY0/AdaKV oracle (`reference/adakv_select_ref.py`, the paper's arm as
//! argus-labs runs it), plus the stage's decision shapes — including the ragged ones this technique
//! exists for. The fixture stores case params, the per-head caps and the per-head keep sets; the
//! Rust test regenerates the attention through the same LCG and drives the real stage via
//! `keep_spec`, so what is pinned is the STAGE's decision, not a helper's.

use super::*;
use argus_extension_api::{TensorDtype, TensorHandle, TensorKind, TensorShape};

// ── deterministic attention generator (mirrors reference/adakv_select_ref.py `synth_attn`) ──

const LCG_A: i64 = 1_103_515_245;
const LCG_C: i64 = 12_345;
const LCG_MASK: i64 = 0x7FFF_FFFF;
const WIDE_MOD: i64 = 1_000_003;

fn synth_attn(n_q_heads: usize, k_len: usize, seed: i64) -> Vec<f32> {
    let mut data = Vec::with_capacity(n_q_heads * k_len);
    let mut s = seed;
    for _ in 0..n_q_heads {
        for _ in 0..k_len {
            s = (LCG_A * s + LCG_C) & LCG_MASK;
            data.push((s % WIDE_MOD) as f32);
        }
    }
    data
}

// ── mock StageCtx supplying PrefillAttention (+ importance, + head starts) ──

struct PfaHandle {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
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
    head_start: Vec<usize>,
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
            head_start: Vec::new(),
        }
    }
}
impl StageCtx for Ctx {
    fn current_pos(&self) -> usize {
        self.current
    }
    fn head_start(&self, kv_head: usize) -> usize {
        self.head_start.get(kv_head).copied().unwrap_or(0)
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

fn cfg(window: usize, kernel: usize, alpha: f64, pooling: Pooling) -> AdaKvConfig {
    AdaKvConfig {
        compression_ratio: 0.0,
        window_size: window,
        kernel_size: kernel,
        pooling,
        floor_alpha: alpha,
    }
}

// ── fixture ──

struct Case {
    n_kv: usize,
    n_q: usize,
    k_len: usize,
    window: usize,
    kernel: usize,
    alpha: f64,
    base: usize,
    protected: usize,
    seed: i64,
    caps: Vec<usize>,
    keeps: Vec<Vec<usize>>,
}

fn load_fixture() -> Vec<Case> {
    let text = include_str!("../tests/fixtures/select_fixture.txt");
    let mut cases: Vec<Case> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("CASE") => {
                let v: Vec<&str> = it.collect();
                cases.push(Case {
                    n_kv: v[0].parse().unwrap(),
                    n_q: v[1].parse().unwrap(),
                    k_len: v[2].parse().unwrap(),
                    window: v[3].parse().unwrap(),
                    kernel: v[4].parse().unwrap(),
                    alpha: v[5].parse().unwrap(),
                    base: v[6].parse().unwrap(),
                    protected: v[7].parse().unwrap(),
                    seed: v[8].parse().unwrap(),
                    caps: Vec::new(),
                    keeps: Vec::new(),
                });
            }
            Some("CAPS") => {
                cases.last_mut().unwrap().caps = it.map(|x| x.parse().unwrap()).collect();
            }
            Some("KEEP") => {
                cases
                    .last_mut()
                    .unwrap()
                    .keeps
                    .push(it.map(|x| x.parse().unwrap()).collect());
            }
            _ => {}
        }
    }
    assert!(!cases.is_empty(), "empty fixture");
    cases
}

/// The STAGE reproduces every fixture case: the same caps and the same per-head keep sets as the
/// FFY0 oracle, including the alpha=0 / alpha=1 corners, kernel 5, a protected prefix and a
/// non-power-of-two head count. Mutation-proof: pooling before the group mean (kvpress's order)
/// changes the max-pooled scores and fails the GQA cases; dropping `int(base·α)` fails every
/// alpha=0.2 case; a left-index-last tie rule fails the plateau cuts.
#[test]
fn stage_matches_the_ffy0_fixture() {
    for c in load_fixture() {
        let stage = AdaKv::new(cfg(c.window, c.kernel, c.alpha, Pooling::Max));
        let ctx = Ctx {
            current: c.k_len,
            target: c.base + c.window,
            n_kv_heads: c.n_kv,
            pfa: pfa(c.n_q, c.k_len, c.seed),
            protected: c.protected,
            ..Default::default()
        };
        // The caps, through the public selection.
        let (caps, keeps) = adakv_per_head_keep(
            AdaKvSelect {
                n_q_heads: c.n_q,
                n_kv_heads: c.n_kv,
                cols: c.k_len,
                current: c.k_len,
                window: c.window,
                kernel: c.kernel,
                pooling: Pooling::Max,
                base: c.base,
                floor_alpha: c.alpha,
                protected: c.protected,
            },
            |qh, out| ctx.pfa.as_ref().unwrap().read_row(qh, 0, out),
            &vec![0; c.n_kv],
        );
        assert_eq!(caps, c.caps, "caps, seed {}", c.seed);
        assert_eq!(keeps, c.keeps, "selection, seed {}", c.seed);
        // And the stage's own decision.
        match stage.keep_spec(&ctx) {
            Some(KeepSpec::PerHead(heads)) => assert_eq!(heads, c.keeps, "stage, seed {}", c.seed),
            other => panic!("seed {}: expected a per-head keep, got {other:?}", c.seed),
        }
        // Each head keeps protected + cap + window — different counts per head.
        for (h, k) in c.keeps.iter().enumerate() {
            assert_eq!(k.len(), c.protected + c.caps[h] + c.window);
        }
        assert!(
            c.caps.windows(2).any(|w| w[0] != w[1]) || c.alpha == 1.0,
            "seed {}: an adaptive case should not be uniform",
            c.seed
        );
    }
}

/// `cap_h = round_f32(count·(1−α) + int(base·α))`, half-to-even, floored at 0. Hand-traced
/// against the FFY0 formula.
#[test]
fn head_caps_are_the_ffy0_blend() {
    // base 224, α 0.2: int(44.8) = 44; counts 196 → 156.8+44 = 200.8 → 201; 248 → 198.4+44 → 242.
    assert_eq!(adakv_head_caps(&[196, 248], 224, 0.2), vec![201, 242]);
    // α = 1: every head gets exactly base (uniform, SnapKV-shaped).
    assert_eq!(adakv_head_caps(&[0, 448], 224, 1.0), vec![224, 224]);
    // α = 0: the raw competition.
    assert_eq!(adakv_head_caps(&[100, 348], 224, 0.0), vec![100, 348]);
    // Half-to-even: 5·0.5 + int(10·0.5) = 2.5 + 5 = 7.5 → 8; 7·0.5 + 5 = 8.5 → 8.
    assert_eq!(adakv_head_caps(&[5, 7], 10, 0.5), vec![8, 8]);
}

/// Registration: `"adakv"` is a prefill-end reader of the prefill attention at the pool's shared
/// window (64), protecting no prefix by itself.
#[test]
fn registration_is_a_prefill_end_pfa_reader_at_window_64() {
    let reg = argus_extension_api::KV_MUTATION_STAGES
        .iter()
        .find(|r| r.name == "adakv")
        .expect("adakv registered");
    assert_eq!(reg.phase, MutationPhase::PrefillEnd);
    assert!(reg.caps.reads.contains(&TensorKind::PrefillAttention));
    assert_eq!(reg.caps.prefill_attn_window, Some(64));
    assert_eq!(reg.caps.default_protected_prefix, 0);
    let stage = (reg.make)(
        StageParams {
            eviction_window: 0,
            protected_prefix: 0,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        },
        &[],
    );
    assert_eq!(stage.name(), "adakv");
}

/// `on_phase` stages the ragged per-head keep through `keep_per_head` — the handle receives lists
/// of different lengths, which is the whole point.
#[test]
fn on_phase_stages_a_ragged_per_head_keep() {
    struct Capture {
        per_head: Option<Vec<Vec<usize>>>,
        layer_wide: Option<Vec<usize>>,
    }
    impl CacheHandle for Capture {
        fn current_pos(&self) -> usize {
            128
        }
        fn n_kv_heads(&self) -> usize {
            2
        }
        fn head_dim(&self) -> usize {
            64
        }
        fn kv_on_device(&self) -> bool {
            false
        }
        fn tensor(&self, _: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
        fn keep(&mut self, keep: &[usize]) -> Result<(), CacheOpError> {
            self.layer_wide = Some(keep.to_vec());
            Ok(())
        }
        fn keep_per_head(&mut self, keep: &[&[usize]]) -> Result<(), CacheOpError> {
            self.per_head = Some(keep.iter().map(|k| k.to_vec()).collect());
            Ok(())
        }
        fn merge(&mut self, _: &[argus_extension_api::WeightedMerge]) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn reencode(&mut self, _: argus_extension_api::FormatId) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn transition_quant_bits(&mut self, _: u8) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn offload(&mut self, _: usize) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn recall(&mut self) -> Result<(), CacheOpError> {
            Ok(())
        }
    }
    let c = &load_fixture()[1]; // GQA groups=2, adaptive caps
    let stage = AdaKv::new(cfg(c.window, c.kernel, c.alpha, Pooling::Max));
    let ctx = Ctx {
        current: c.k_len,
        target: c.base + c.window,
        n_kv_heads: c.n_kv,
        pfa: pfa(c.n_q, c.k_len, c.seed),
        ..Default::default()
    };
    let mut cap = Capture {
        per_head: None,
        layer_wide: None,
    };
    stage.on_phase(&ctx, &mut cap).expect("staged");
    let heads = cap.per_head.expect("per-head keep");
    assert!(cap.layer_wide.is_none());
    assert_eq!(heads, c.keeps);
    assert_ne!(heads[0].len(), heads[1].len(), "ragged by construction");
}

/// On a ragged cache the stage ranks only what each head holds: holes never compete, never get
/// kept, and the protected prefix is the head's FIRST RESIDENT positions. Mutation-proof: a stage
/// that ignored `head_start` would keep `[0, 4)` for head 1 (its holes) and rank hole columns.
#[test]
fn a_ragged_cache_is_ranked_from_each_heads_own_start() {
    let c = &load_fixture()[6]; // protected prefix 4
    let stage = AdaKv::new(cfg(c.window, c.kernel, c.alpha, Pooling::Max));
    let starts = vec![0usize, 10];
    let ctx = Ctx {
        current: c.k_len,
        target: c.base + c.window,
        n_kv_heads: c.n_kv,
        pfa: pfa(c.n_q, c.k_len, c.seed),
        protected: c.protected,
        head_start: starts.clone(),
        ..Default::default()
    };
    let Some(KeepSpec::PerHead(heads)) = stage.keep_spec(&ctx) else {
        panic!("expected a per-head keep")
    };
    for (h, k) in heads.iter().enumerate() {
        assert!(
            k.iter().all(|&p| p >= starts[h]),
            "head {h} names a hole: {k:?}"
        );
        assert_eq!(
            &k[..c.protected],
            &(starts[h]..starts[h] + c.protected).collect::<Vec<_>>()
        );
        assert!(k.windows(2).all(|w| w[0] < w[1]));
    }
    // Head 0 (uniform) decides exactly as on a uniform cache; head 1's competition shrank.
    let uniform = stage.keep_spec(&Ctx {
        head_start: Vec::new(),
        pfa: pfa(c.n_q, c.k_len, c.seed),
        current: c.k_len,
        target: c.base + c.window,
        n_kv_heads: c.n_kv,
        protected: c.protected,
        ..Default::default()
    });
    let Some(KeepSpec::PerHead(uheads)) = uniform else {
        panic!()
    };
    assert_ne!(heads[1], uheads[1]);
}

/// Below the window the budget is a recency keep — layer-wide on a uniform cache, per head from
/// each head's own start on a ragged one (both are what the container accepts).
#[test]
fn sub_window_budgets_keep_the_most_recent_per_head() {
    let stage = AdaKv::new(cfg(16, 7, 0.2, Pooling::Max));
    let uniform = stage.keep_spec(&Ctx {
        current: 100,
        target: 12,
        n_kv_heads: 2,
        protected: 2,
        ..Default::default()
    });
    let mut want: Vec<usize> = vec![0, 1];
    want.extend(88..100);
    assert_eq!(uniform, Some(KeepSpec::LayerWide(want)));
    let ragged = stage.keep_spec(&Ctx {
        current: 100,
        target: 12,
        n_kv_heads: 2,
        protected: 2,
        head_start: vec![0, 50],
        ..Default::default()
    });
    let Some(KeepSpec::PerHead(heads)) = ragged else {
        panic!()
    };
    assert_eq!(heads[0], (0..2).chain(88..100).collect::<Vec<_>>());
    assert_eq!(heads[1], (50..52).chain(88..100).collect::<Vec<_>>());
}

/// No-ops and fallbacks mirror `snapkv`: empty cache, budget covering the cache, no PFA →
/// importance-ranked (then recency) layer-wide keep of the same budget.
#[test]
fn noops_and_degraded_fallbacks() {
    let stage = AdaKv::new(cfg(8, 7, 0.2, Pooling::Max));
    assert_eq!(stage.keep_spec(&Ctx::default()), None);
    assert_eq!(
        stage.keep_spec(&Ctx {
            current: 50,
            target: 50,
            ..Default::default()
        }),
        None
    );
    let imp: Vec<f32> = (0..64)
        .map(|p| if p % 5 == 0 { 9.0 } else { 1.0 })
        .collect();
    let spec = stage.keep_spec(&Ctx {
        current: 64,
        target: 20,
        n_kv_heads: 2,
        importance: Some(imp),
        ..Default::default()
    });
    let Some(KeepSpec::LayerWide(k)) = spec else {
        panic!()
    };
    assert_eq!(k.len(), 20);
    assert!(k[..12].iter().all(|p| p % 5 == 0));
    assert_eq!(&k[12..], &(56..64).collect::<Vec<_>>());
    let spec = stage.keep_spec(&Ctx {
        current: 64,
        target: 20,
        n_kv_heads: 2,
        ..Default::default()
    });
    assert_eq!(spec, Some(KeepSpec::LayerWide((44..64).collect())));
}

/// `--set` knobs: kernel forced odd, alpha clamped, pooling by name, cr clamped.
#[test]
fn config_from_args() {
    use argus_extension_api::PluginArg;
    let args = [
        PluginArg {
            key: "kernel_size",
            val: "6",
        },
        PluginArg {
            key: "floor_alpha",
            val: "1.7",
        },
        PluginArg {
            key: "pooling",
            val: "avg",
        },
        PluginArg {
            key: "compression_ratio",
            val: "0.5",
        },
        PluginArg {
            key: "window_size",
            val: "0",
        },
    ];
    let c = AdaKvConfig::from_args(
        StageParams {
            eviction_window: 0,
            protected_prefix: 0,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        },
        &args,
    );
    assert_eq!(c.kernel_size, 7);
    assert_eq!(c.floor_alpha, 1.0);
    assert_eq!(c.pooling, Pooling::Avg);
    assert_eq!(c.compression_ratio, 0.5);
    assert_eq!(c.window_size, 1);
    assert_eq!(c.budget(100, 0), Some(50));
}
