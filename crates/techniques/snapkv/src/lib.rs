//! SnapKV (Li et al., 2024 — <https://arxiv.org/abs/2404.14469>) technique crate — a **uniform
//! per-layer KV budget** over the **SnapKV per-head attention selection**, at prefill end.
//!
//! Self-registering stage-axis extension (the `pyramidkv` precedent): depends only on
//! `argus-extension-api` + `linkme`, implements [`KVMutationStage`], registers under the name
//! `"snapkv"` via `register_kv_mutation_stage!`, and is force-linked by the engine (`use snapkv as
//! _;` under `--features snapkv`). Private knobs ride the [`StageArgs`] blob (`--set
//! compression_ratio=<R> --set window_size=<W> --set kernel_size=<K>`).
//!
//! ## SnapKV vs PyramidKV — one selection, two budgets
//!
//! kvpress `PyramidKVPress` subclasses `SnapKVPress` and overrides ONLY the budget
//! (`get_layer_budget`, a per-layer pyramid); the selection (`score` + `topk`) is inherited
//! unchanged. That selection is [`snapkv_per_head_keep`] in the extension API, shared by both
//! crates. This crate is the uniform budget over it — what PyramidKV falls back to when its pyramid
//! does not fit, except for rounding (below). The `n_kept ≤ window` handling, the protected prefix
//! and the degraded fallbacks mirror `pyramidkv` line for line, so the two arms of a
//! `--aperturb-select` pool differ in exactly one thing: the layer budget.
//!
//! ## Matched against NVIDIA kvpress `SnapKVPress`
//!
//! * **Budget** ([`snapkv_budget`]) — `ScorerPress.compress`, which `SnapKVPress` inherits:
//!   `n_kept = int(k_len * (1 − compression_ratio))`. Python `int` TRUNCATES toward zero; this is
//!   NOT `round`. (`PyramidKVPress`'s SnapKV *fallback* is `round(q_len·(1−cr))`, half-to-even, so
//!   the two presses' "uniform SnapKV budget" disagree at the fractional boundary: `q=1283, cr=0.5`
//!   keeps 641 here and 642 there.) Same `f64` operation order as the Python source, so the bits
//!   agree; asserted against `tests/fixtures/budget_grid.csv` and, in `reference/`, against the real
//!   library.
//! * **Selection** — [`snapkv_per_head_keep`]. The three residuals `pyramidkv` documents
//!   (sub-window budgets keep the COUNT only; exact score ties break lower-index-first, not in
//!   `torch.topk`'s order; f32 PFA vs kvpress's f16 cast) are inherited verbatim — the selection is
//!   the same code. See `crates/techniques/pyramidkv/src/lib.rs` "Residuals" and its
//!   `reference/README.md`; they are not re-derived here.
//!
//! ## Budget from the engine
//!
//! With no explicit `compression_ratio` the budget is the engine's resolved `target_len` ITSELF
//! (`--kv-budget-ratio`, `--eviction-target-ratio`, a `--aperturb-select` ask). kvpress takes no
//! target length, so this path is engine convention, and it is exact by construction: deriving
//! `cr = 1 − target/current` and truncating `current·(1−cr)` loses a token to `f64` rounding for
//! most pairs (65 → 5 truncates to 4). `pyramidkv` survives that round trip only because it rounds.
//!
//! ## Where the faithful path runs
//!
//! The faithful per-head selection needs the engine's [`TensorKind::PrefillAttention`] producer,
//! armed by the caps-driven `resolve_prefill_keepset_arming` on every loop. The PFA observation
//! window is declared on [`StageCaps::prefill_attn_window`] (64, the kvpress default) and MUST equal
//! pyramidkv's: `--aperturb-select` refuses a pool whose prefill-attention candidates want different
//! windows, because the forward SUMs over one window and the loser would rank a different set of
//! heavy hitters than it declares. Each kv-head keeps the SAME NUMBER of tokens at DIFFERENT
//! positions ([`KeepSpec::PerHead`], the single-`current_pos` invariant); per-head execution needs a
//! HeadMajor cache. When the PFA is unavailable (producer unarmed, unusable geometry) the stage
//! degrades to a layer-wide keep ranked by flat `importance()`, else recency — always safe to run.

use argus_extension_api::{
    CacheHandle, CacheOpError, KVMutationStage, KeepSpec, KeepTopK, MutationPhase, SignalId,
    SnapKvSelect, StageArgs, StageCaps, StageCtx, StageParams, TensorKind, compile_keep_top_k,
    register_kv_mutation_stage, snapkv_per_head_keep,
};

/// The caps for the registration: SnapKV reads the prefill attention; protects no prefix
/// (kvpress-faithful); drop-only.
const SNAPKV_CAPS: StageCaps = StageCaps {
    reads: &[TensorKind::PrefillAttention],
    reads_signals: &[],
    default_protected_prefix: 0,
    produces_merge_plan: false,
    whole_model: false,
    // The PFA producer must observe EXACTLY `window_size` trailing queries (kvpress scores the mean of
    // the last `window_size` queries' attention). MIRRORS `SnapKvConfig::default().window_size` (64),
    // and MUST equal pyramidkv's declaration: two prefill-attention candidates of a `--aperturb-select`
    // pool are refused when their windows differ (`build_bench_loop`), and the engine tests pin the
    // agreement. `--set window_size` is not threaded into the standard loop, so the const default is
    // what producer and consumer both use (the documented pyramidkv limitation).
    prefill_attn_window: Some(64),
};

// ── kvpress-parity arithmetic ────────────────────────────────────────────────

/// The kvpress `ScorerPress.compress` budget (which `SnapKVPress` inherits unchanged):
/// `n_kept = int(k_len * (1 - compression_ratio))`.
///
/// `compression_ratio` is the fraction of KV pairs REMOVED (kvpress semantics, `0 ≤ cr < 1`).
/// Python `int()` truncates toward zero, and so does `as usize` on a finite non-negative `f64` —
/// this is NOT `round`, which is what `PyramidKVPress`'s SnapKV fallback uses. Same `f64`
/// operation order as the Python source (`1 − cr` first, then the product).
pub fn snapkv_budget(k_len: usize, compression_ratio: f64) -> usize {
    (k_len as f64 * (1.0 - compression_ratio)) as usize
}

// ── config ───────────────────────────────────────────────────────────────────

/// SnapKV knobs. Defaults mirror kvpress `SnapKVPress` (`window_size=64`, `kernel_size=5`).
/// `compression_ratio` is the fraction REMOVED; `0.0` means "the engine's resolved `target_len` is
/// the budget" so the `--kv-budget-ratio` / `--aperturb-select` paths also work.
#[derive(Clone, Copy, Debug)]
struct SnapKvConfig {
    /// Fraction of KV pairs removed (kvpress semantics). `0.0` ⇒ the budget is `target_len`.
    compression_ratio: f64,
    window_size: usize,
    kernel_size: usize,
}

impl Default for SnapKvConfig {
    fn default() -> Self {
        Self {
            compression_ratio: 0.0,
            window_size: 64,
            kernel_size: 5,
        }
    }
}

impl SnapKvConfig {
    fn from_args(_base: StageParams, args: StageArgs<'_>) -> Self {
        let mut c = SnapKvConfig::default();
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
                        // Clamp ≥ 1: window_size == 0 would make the selection's `1/window` infinite;
                        // kvpress requires a positive window.
                        c.window_size = v.max(1);
                    }
                }
                "kernel_size" => {
                    if let Ok(v) = a.val.parse::<usize>() {
                        // Force ODD: `F.avg_pool1d(padding=k//2, stride=1)` only preserves length for
                        // odd k. `| 1` rounds up to the nearest odd (2→3, 4→5); 5→5 unchanged.
                        c.kernel_size = v.max(1) | 1;
                    }
                }
                _ => {}
            }
        }
        c
    }

    /// The kept count for a `current`-long cache: the kvpress budget when `compression_ratio` is
    /// explicit, else the engine's `target_len` itself (exact — no `cr` round trip, see the crate doc).
    /// `None` = no compression (kvpress: `cr == 0` ⇒ no-op; no target, or one that covers the cache).
    fn budget(&self, current: usize, target_len: usize) -> Option<usize> {
        if self.compression_ratio > 0.0 {
            Some(snapkv_budget(current, self.compression_ratio))
        } else if target_len > 0 && target_len < current {
            Some(target_len)
        } else {
            None
        }
    }
}

// ── stage ──────────────────────────────────────────────────────────────────

struct SnapKv {
    cfg: SnapKvConfig,
}

impl SnapKv {
    fn new(cfg: SnapKvConfig) -> Self {
        Self { cfg }
    }

    /// The keep-set shape (`None` = no-op). Faithful per-head SnapKV ([`KeepSpec::PerHead`]) when
    /// the prefill attention is usable; otherwise a layer-wide keep of the same budget (window-only,
    /// importance-ranked, or recency). The branch structure is pyramidkv's, with the uniform budget.
    fn keep_spec(&self, ctx: &dyn StageCtx) -> Option<KeepSpec> {
        let current = ctx.current_pos();
        if current == 0 {
            return None; // empty cache — nothing to evict (also: `clamp(1, 0)` below would panic).
        }
        let raw_budget = self.cfg.budget(current, ctx.target_len())?;

        // kvpress keeps EXACTLY `n_kept` positions, even below the observation window: its `score`
        // max-fills the window so `topk(n_kept)` keeps the whole window plus `n_kept − window` heavy
        // hitters when `n_kept ≥ window`, and `n_kept` of the (tied) window positions otherwise — we
        // keep the `n_kept` most recent there (only the COUNT is faithful; the tie order is torch's).
        // Floor to 1, never to the window: `int(k_len·(1−cr))` can hit 0 and a 0-length keep empties
        // the cache.
        let window = self.cfg.window_size.min(current);
        let n_kept = raw_budget.clamp(1, current);
        if n_kept >= current {
            return None; // budget covers everything — nothing to evict.
        }
        // `--protected-prefix`: force-KEEP the leading positions (the attention-sink guard). `0` =
        // kvpress-faithful (the default); a non-zero value is ADDITIVE to the budget, so it only ever
        // keeps MORE.
        let protected = ctx.protected_prefix().min(current);
        if n_kept <= window {
            // At or below the observation window: the `n_kept` most recent, layer-wide (identical
            // across heads — valid on any cache layout), with the protected prefix unioned in front.
            let mut keep: Vec<usize> = (0..protected).collect();
            keep.extend((current - n_kept..current).filter(|&p| p >= protected));
            return Some(KeepSpec::LayerWide(keep));
        }
        let heavy = n_kept - window;

        // (1) Faithful per-head SnapKV path: needs the prefill attention (per attention head,
        //     pre-GQA). Read via the open signal name — `signal()` bridges `"attn.prefill_window"`
        //     back to `tensor(PrefillAttention)`, so this is byte-identical.
        if let Some(pfa) = ctx.signal(SignalId("attn.prefill_window")) {
            let shape = pfa.shape();
            let n_q = shape.rows;
            let cols = shape.cols;
            let n_kv = ctx.n_kv_heads().max(1);
            let heavy_len = current - window;
            if n_q >= n_kv && n_q % n_kv == 0 && cols >= heavy_len {
                let heads = snapkv_per_head_keep(
                    SnapKvSelect {
                        n_q_heads: n_q,
                        n_kv_heads: n_kv,
                        cols,
                        current,
                        window,
                        kernel: self.cfg.kernel_size,
                        heavy,
                        protected,
                    },
                    |qh, out| pfa.read_row(qh, 0, out), // PFA is per_head:false → kv_head ignored
                );
                return Some(KeepSpec::PerHead(heads));
            }
        }

        // (2) Degraded fallback — PFA unavailable (producer unarmed) or its geometry unusable for
        //     per-head SnapKV. The SAME budget layer-wide, heavy hitters ranked by flat
        //     `importance()` (H2O-style), else recency. Not kvpress's per-head selection, but always
        //     safe on any layout.
        let keep = match ctx.importance() {
            Some(imp) => compile_keep_top_k(
                KeepTopK {
                    current,
                    prefix: protected,
                    recent: window,
                    heavy,
                },
                |pos| imp.get(pos).copied().unwrap_or(0.0),
            ),
            None => compile_keep_top_k(
                KeepTopK {
                    current,
                    prefix: protected,
                    recent: n_kept, // recency: keep the most-recent n_kept
                    heavy: 0,
                },
                |_| 0.0,
            ),
        };
        Some(KeepSpec::LayerWide(keep))
    }
}

impl KVMutationStage for SnapKv {
    fn name(&self) -> &str {
        "snapkv"
    }

    // NOTE: the PFA observation window (the SnapKV `window_size`) is declared statically on
    // `SNAPKV_CAPS.prefill_attn_window = Some(64)` (the engine reads it pre-`make`).

    /// Stage the SnapKV per-head (or layer-wide fallback) keep-set at prefill end, or no-op.
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
    "snapkv",
    |p, args| Box::new(SnapKv::new(SnapKvConfig::from_args(p, args))),
    SNAPKV_CAPS,
    MutationPhase::PrefillEnd
);

#[cfg(test)]
mod tests;
