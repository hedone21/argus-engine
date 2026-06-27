//! PyramidKV (Cai et al., 2024 — <https://arxiv.org/abs/2406.02069>) technique crate — **pyramidal
//! per-layer KV budget allocation** on top of **SnapKV per-head attention scoring**.
//!
//! Self-registering stage-axis extension (the `h2o`/`h2o-plus`/`d2o` precedent): depends only on
//! `argus-extension-api` + `linkme`, implements [`KVCacheStage`], registers under the name
//! `"pyramidkv"` via `#[distributed_slice(KV_CACHE_STAGES)]`, and is force-linked by the engine
//! (`use pyramidkv as _;`). Private knobs ride the [`StageArgs`] blob (`eviction plugin --name
//! pyramidkv --set compression_ratio=<R> --set window_size=<W> ...`).
//!
//! ## Matched against NVIDIA kvpress `PyramidKVPress`
//!
//! Two pieces, ported to be byte-identical to
//! <https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/pyramidkv_press.py> (which in turn
//! ports the official authors' arithmetic, KVCache-Factory `pyramidkv_utils.py#L197`):
//!
//! 1. **Per-layer budget** ([`get_layer_budget`]) — a verbatim port of
//!    `PyramidKVPress.get_layer_budget`. `max_capacity = window + q_len·(1−cr)`; the layer budgets
//!    form an arithmetic sequence from `max_num` (layer 0) down to `min_num` (last layer), averaging
//!    `q_len·(1−cr)`. All arithmetic is `f64` and the final `round` is **round-half-to-even**
//!    (Python `round`) via [`f64::round_ties_even`] — `f64::round` (half-away-from-zero) would
//!    diverge at exact `.5` boundaries.
//!
//! 2. **Per-head selection** (the SnapKV `score`) — from the engine's
//!    [`TensorKind::PrefillAttention`] (per ATTENTION head, pre-GQA, the trailing q-window's
//!    attention summed over the window to every prefix key): mean over the window (÷window),
//!    `avg_pool1d(kernel, pad=kernel/2, stride=1, count_include_pad=True)`, GQA group-mean over the
//!    q-heads of each kv-head, then keep the budget's worth of highest-scored positions **plus the
//!    always-kept recent window** — i.e. `topk` with the window forced in. Routed through the
//!    engine's [`compile_keep_top_k`] (prefix `0`, recent = `window`, heavy = `budget − window`),
//!    whose STABLE-desc/ascending-resort tie-break matches `torch.topk`'s lower-index-first order.
//!
//! Each kv-head keeps the SAME NUMBER of tokens (the per-layer budget) at DIFFERENT positions, so a
//! [`KeepSpec::PerHead`] plan satisfies the engine's single-`current_pos` invariant (equal-length
//! per-head keep-lists). Per-head execution requires a HeadMajor cache (the engine bails cleanly
//! otherwise); when `PrefillAttention` is not supplied (e.g. the producer is unarmed) the stage
//! degrades to a layer-wide pyramid-budgeted plan using flat `importance()`, and with no scores at
//! all to recency — so it is always safe to run.

use argus_extension_api::{
    CacheHandle, CacheOpError, KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg,
    KVMutationStage, KeepSpec, KeepTopK, MutationPhase, StageArgs, StageCaps, StageCtx,
    StageParams, TensorKind, compile_keep_top_k, register_kv_mutation_stage,
};
use linkme::distributed_slice;

/// The caps shared by the v2 [`KVCacheStageReg`] and the v3 registration: PyramidKV reads the
/// prefill attention (SnapKV score source); protects no prefix; drop-only.
const PYRAMIDKV_CAPS: StageCaps = StageCaps {
    reads: &[TensorKind::PrefillAttention],
    default_protected_prefix: 0,
    produces_merge_plan: false,
};

// ── KVPress-parity arithmetic ────────────────────────────────────────────────

/// Per-layer KV budget — a verbatim port of NVIDIA kvpress `PyramidKVPress.get_layer_budget`.
///
/// `compression_ratio` is the fraction of KV pairs REMOVED (kvpress semantics). All arithmetic is
/// `f64` in the same operation order as the Python source, and `round` is round-half-to-even
/// ([`f64::round_ties_even`]) to match Python's `round`. Returns the token budget (kept count) for
/// `layer_idx` of `num_layers`. `num_layers <= 1` (Python would `ZeroDivisionError`) and the
/// out-of-range branch both fall back to the uniform SnapKV budget `round(q_len·(1−cr))`.
pub fn get_layer_budget(
    q_len: usize,
    compression_ratio: f64,
    window_size: usize,
    beta: u32,
    num_layers: usize,
    layer_idx: usize,
) -> usize {
    // assert beta >= 1 — enforced by the config (beta: u32 clamped to >= 1).
    let q = q_len as f64;
    let w = window_size as f64;

    let max_capacity_prompt = w + q * (1.0 - compression_ratio);

    let mut min_num = (max_capacity_prompt - w) / beta as f64;
    let mut max_num = (max_capacity_prompt - w) * 2.0 - min_num;

    if max_num >= q - w {
        max_num = q - w;
        min_num = (max_capacity_prompt - w) * 2.0 - max_num;
    }

    let uniform = || (q * (1.0 - compression_ratio)).round_ties_even().max(0.0) as usize;

    // if not (q_len >= max_num >= min_num >= window_size): fall back to SnapKV (uniform budget).
    if !(q >= max_num && max_num >= min_num && min_num >= w) {
        return uniform();
    }
    if num_layers <= 1 {
        return uniform();
    }

    let steps = (max_num - min_num) / (num_layers as f64 - 1.0);
    (max_num - layer_idx as f64 * steps)
        .round_ties_even()
        .max(0.0) as usize
}

/// `F.avg_pool1d(input, kernel_size, padding=kernel_size/2, stride=1, count_include_pad=True)`.
///
/// Zero-pads `kernel/2` on each side and divides every window by `kernel_size` (padded zeros are
/// counted in the denominator — `count_include_pad=True`, PyTorch's default). For an ODD kernel the
/// output length equals the input length (kvpress always uses `kernel_size=5`); `out.len()` must
/// equal `input.len()`.
fn avg_pool1d(input: &[f32], kernel: usize, out: &mut [f32]) {
    let n = input.len();
    let pad = kernel / 2;
    for (i, o) in out.iter_mut().enumerate().take(n) {
        let mut s = 0.0f32;
        for j in 0..kernel {
            let idx = i as isize - pad as isize + j as isize;
            if idx >= 0 && (idx as usize) < n {
                s += input[idx as usize];
            }
        }
        *o = s / kernel as f32;
    }
}

/// Per-head SnapKV keep-set selection from a per-q-head attention reader.
///
/// `read_qhead(qh, out)` fills `out[0..cols]` with attention head `qh`'s window-summed attention to
/// every prefix key. Produces one ascending keep-list per kv-head, each of length
/// `heavy + window.min(current)` (= the per-layer budget), so all heads keep an equal count.
#[allow(clippy::too_many_arguments)]
fn per_head_keep(
    read_qhead: impl Fn(usize, &mut [f32]),
    n_q_heads: usize,
    n_kv_heads: usize,
    cols: usize,
    current: usize,
    window: usize,
    kernel: usize,
    heavy: usize,
) -> Vec<Vec<usize>> {
    let groups = (n_q_heads / n_kv_heads).max(1);
    let heavy_len = current - window; // scoring region [0, current-window); window is force-kept

    // pooled[qh][pos] = avg_pool( attn[qh][0..heavy_len] / window )
    let mut pooled = vec![vec![0.0f32; heavy_len]; n_q_heads];
    let mut row = vec![0.0f32; cols];
    let inv_window = 1.0f32 / window as f32;
    for (qh, p) in pooled.iter_mut().enumerate() {
        read_qhead(qh, &mut row);
        // ÷window (KVPress's mean over the window queries; SUM→MEAN). Order-of-ops mirrors the
        // reference so the f32 values — hence the topk SET — match.
        let scaled: Vec<f32> = row[..heavy_len].iter().map(|&v| v * inv_window).collect();
        avg_pool1d(&scaled, kernel, p);
    }

    (0..n_kv_heads)
        .map(|kvh| {
            let base = kvh * groups;
            // GQA group-mean over the q-heads of this kv-head.
            let inv_groups = 1.0f32 / groups as f32;
            let scores: Vec<f32> = (0..heavy_len)
                .map(|pos| {
                    let mut s = 0.0f32;
                    for g in 0..groups {
                        s += pooled[base + g][pos];
                    }
                    s * inv_groups
                })
                .collect();
            // window force-kept (recent), top `heavy` from the scored region; ascending keep-list.
            compile_keep_top_k(
                KeepTopK {
                    current,
                    prefix: 0,
                    recent: window,
                    heavy,
                },
                |pos| scores.get(pos).copied().unwrap_or(0.0),
            )
        })
        .collect()
}

// ── config ───────────────────────────────────────────────────────────────────

/// PyramidKV knobs. Defaults mirror kvpress `PyramidKVPress` (`window_size=64`, `kernel_size=5`,
/// `beta=20`). `compression_ratio` is the fraction REMOVED; `0.0` means "derive from the engine's
/// resolved `target_len`" so the `--kv-budget-ratio` path also works.
#[derive(Clone, Copy, Debug)]
struct PyramidKvConfig {
    /// Fraction of KV pairs removed (kvpress semantics). `0.0` ⇒ derive from `target_len`.
    compression_ratio: f64,
    window_size: usize,
    kernel_size: usize,
    beta: u32,
}

impl Default for PyramidKvConfig {
    fn default() -> Self {
        Self {
            compression_ratio: 0.0,
            window_size: 64,
            kernel_size: 5,
            beta: 20,
        }
    }
}

impl PyramidKvConfig {
    fn from_args(_base: StageParams, args: StageArgs<'_>) -> Self {
        let mut c = PyramidKvConfig::default();
        for a in args {
            match a.key {
                "compression_ratio" => {
                    if let Ok(v) = a.val.parse::<f64>() {
                        // kvpress asserts 0 <= cr < 1.
                        c.compression_ratio = v.clamp(0.0, 0.999_999);
                    }
                }
                "window_size" => {
                    if let Ok(v) = a.val.parse::<usize>() {
                        c.window_size = v;
                    }
                }
                "kernel_size" => {
                    if let Ok(v) = a.val.parse::<usize>() {
                        // Force ODD: kvpress always uses an odd kernel (default 5), and
                        // `F.avg_pool1d(padding=k//2, stride=1)` only preserves length for odd k —
                        // an even kernel yields asymmetric, length-mismatched pooling. `| 1` rounds
                        // up to the nearest odd (2→3, 4→5); 5→5 unchanged.
                        c.kernel_size = v.max(1) | 1;
                    }
                }
                "beta" => {
                    if let Ok(v) = a.val.parse::<u32>() {
                        c.beta = v.max(1);
                    }
                }
                _ => {}
            }
        }
        c
    }

    /// The effective compression ratio: the explicit knob if set, else derived from the engine's
    /// `target_len` (keep `target_len` of `current` ⇒ remove `1 − target_len/current`).
    fn effective_cr(&self, current: usize, target_len: usize) -> f64 {
        if self.compression_ratio > 0.0 {
            self.compression_ratio
        } else if target_len > 0 && target_len < current {
            1.0 - (target_len as f64 / current as f64)
        } else {
            0.0
        }
    }
}

// ── stage ──────────────────────────────────────────────────────────────────

struct PyramidKv {
    cfg: PyramidKvConfig,
}

impl PyramidKv {
    fn new(cfg: PyramidKvConfig) -> Self {
        Self { cfg }
    }
}

impl PyramidKv {
    /// The keep-set shape (`None` = no-op), shared by the v3 `on_phase` and the v2 `plan` so they
    /// decide byte-identically. Faithful per-head SnapKV ([`KeepSpec::PerHead`]) when PrefillAttention
    /// is usable; otherwise a layer-wide pyramid budget (window-only or score-ranked fallback).
    fn keep_spec(&self, ctx: &dyn StageCtx) -> Option<KeepSpec> {
        let current = ctx.current_pos();
        let cr = self.cfg.effective_cr(current, ctx.target_len());
        if cr <= 0.0 {
            return None; // kvpress: compression_ratio == 0 ⇒ no compression (no-op).
        }

        let raw_budget = get_layer_budget(
            current,
            cr,
            self.cfg.window_size,
            self.cfg.beta,
            ctx.n_layers(),
            ctx.layer_idx(),
        );

        // Floor the budget to the observation window. kvpress's pyramid branch already guarantees
        // every layer budget ≥ window_size (the `min_num ≥ window_size` admission check), so this is
        // a NO-OP for the faithful path. It only bites the degenerate SnapKV-uniform fallback (very
        // high compression / tiny prompt), where `round(q·(1−cr))` can fall below the window or to 0
        // — which would evict the always-kept recent window and could empty the cache entirely
        // (`(current-0..current)` is an empty range). PyramidKV/SnapKV always retain the recent
        // window, so flooring to it is the correct, safe boundary.
        let window = self.cfg.window_size.min(current);
        let n_kept = raw_budget.clamp(window, current);
        if n_kept >= current {
            return None; // budget covers everything (incl. window == current) — nothing to evict.
        }
        if n_kept == window {
            // Exactly the window: keep the recent window only (== kvpress's window-forced set when
            // the budget equals the window). Layer-wide (identical across heads) — valid on any
            // cache layout, no HeadMajor requirement.
            let keep: Vec<usize> = (current - window..current).collect();
            return Some(KeepSpec::LayerWide(keep));
        }
        let heavy = n_kept - window;

        // (1) Faithful per-head SnapKV path: requires PrefillAttention (per attention head, pre-GQA).
        if let Some(pfa) = ctx.tensor(TensorKind::PrefillAttention) {
            let shape = pfa.shape();
            let n_q = shape.rows;
            let cols = shape.cols;
            let n_kv = ctx.n_kv_heads().max(1);
            let heavy_len = current - window;
            if n_q >= n_kv && n_q % n_kv == 0 && cols >= heavy_len {
                let heads = per_head_keep(
                    |qh, out| pfa.read_row(qh, 0, out), // PFA is per_head:false → kv_head ignored
                    n_q,
                    n_kv,
                    cols,
                    current,
                    window,
                    self.cfg.kernel_size,
                    heavy,
                );
                return Some(KeepSpec::PerHead(heads));
            }
        }

        // (2) Degraded fallback — taken when PFA is unavailable (producer unarmed) OR its geometry
        //     is unusable for per-head SnapKV (n_q < n_kv, n_q not a multiple of n_kv, or cols too
        //     small). Apply the SAME pyramid budget layer-wide, ranking heavy hitters by flat
        //     `importance()` (H2O-style), else recency. Not byte-identical to kvpress (which is
        //     per-head SnapKV) but keeps the pyramid allocation and is always safe on any layout.
        let keep = match ctx.importance() {
            Some(imp) => compile_keep_top_k(
                KeepTopK {
                    current,
                    prefix: 0,
                    recent: window,
                    heavy,
                },
                |pos| imp.get(pos).copied().unwrap_or(0.0),
            ),
            None => compile_keep_top_k(
                KeepTopK {
                    current,
                    prefix: 0,
                    recent: n_kept, // recency: keep the most-recent n_kept
                    heavy: 0,
                },
                |_| 0.0,
            ),
        };
        Some(KeepSpec::LayerWide(keep))
    }
}

// ── v3 native (imperative) surface — the production path (PrefillEnd phase) ──

impl KVMutationStage for PyramidKv {
    fn name(&self) -> &str {
        "pyramidkv"
    }

    /// Stage the SnapKV per-head (or layer-wide fallback) keep-set at prefill end, or no-op. The
    /// driver supplies PrefillAttention via `ctx.tensor(PrefillAttention)`. Byte-identical to the v2
    /// plan via the shared `keep_spec`.
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
    "pyramidkv",
    |p, args| Box::new(PyramidKv::new(PyramidKvConfig::from_args(p, args))),
    PYRAMIDKV_CAPS,
    MutationPhase::PrefillEnd
);

// ── v2 plan-returning surface (kept for the migration window; removed in Phase 2) ──

impl KVCacheStage for PyramidKv {
    fn name(&self) -> &str {
        "pyramidkv"
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

/// Registration — the engine resolves this via `make_stage_with_args("pyramidkv", ...)`.
///
/// `caps.reads = [PrefillAttention]` declares the SnapKV score source; the engine's caps-driven
/// `find_prefill_attn_stage_name` arms the prefill-attention producer for exactly this stage.
/// `default_protected_prefix = 0` — kvpress PyramidKV protects no prefix (the recent window + heavy
/// hitters are the whole keep-set). Drop-only (no weighted merges).
#[distributed_slice(KV_CACHE_STAGES)]
static PYRAMIDKV: KVCacheStageReg = KVCacheStageReg {
    name: "pyramidkv",
    make: |p: StageParams| Box::new(PyramidKv::new(PyramidKvConfig::from_args(p, &[]))),
    make_with_args: |p: StageParams, args| {
        Box::new(PyramidKv::new(PyramidKvConfig::from_args(p, args)))
    },
    caps: PYRAMIDKV_CAPS,
};

#[cfg(test)]
mod tests;
