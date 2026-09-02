//! AdaKV / Ada-SnapKV (Feng et al., 2024 — <https://arxiv.org/abs/2407.11550>) technique crate —
//! the SnapKV observation-window ranking with the layer's KV budget **redistributed across KV
//! heads**, at prefill end.
//!
//! Self-registering stage-axis extension (the `pyramidkv` / `snapkv` precedent): depends only on
//! `argus-extension-api` + `linkme`, implements [`KVMutationStage`], registers under the name
//! `"adakv"` via `register_kv_mutation_stage!`, and is force-linked by the engine (`use adakv as
//! _;` under `--features adakv`). Private knobs ride the [`StageArgs`] blob (`--set
//! compression_ratio=<R> --set window_size=<W> --set kernel_size=<K> --set floor_alpha=<A> --set
//! pooling=max|avg`).
//!
//! ## What AdaKV is, and which AdaKV this is
//!
//! SnapKV gives every KV head of a layer the same budget. AdaKV keeps the total and lets the heads
//! compete for it: the pooled observation-window scores of every `(head, position)` are ranked
//! together, the top `n_kv · base` win, and a head's budget is how many winners it holds — blended
//! with a floor so no head starves (`cap_h = round(count_h · (1 − α) + int(base · α))`,
//! `α = floor_alpha`). Each head then keeps its own top-`cap_h`. Heads keep DIFFERENT counts —
//! the ragged per-head keep the engine's container right-aligns ([`CacheHandle::head_start`]);
//! the CPU, OpenCL and CUDA attention paths read each head from its own first resident slot.
//!
//! Two reference implementations exist and they differ. NVIDIA kvpress `AdaKVPress` guards the
//! top `int(n_kept · α)` of every head and takes a global bottom-k — a pure floor, no blend — and
//! then only MASKS the pruned keys ("does not reduce peak memory", its own words). The original
//! authors' `FFY0/AdaKV` (`update_kv_gqa`, `--gqa_support`) does the blend above at the KV-head
//! unit with `gqa_func=mean`, and physically compacts. **This crate is the FFY0 form**, because
//! that is the arm the paper's quality numbers come from (argus-labs
//! `build_selection_adakv`, audited to Jaccard 1.0 against `FFY0/AdaKV` at
//! `analysis/audits/adakv_impl/`). Pipeline, in FFY0's order:
//!
//! 1. window-MEAN attention per query head over the prefix `[0, L)`, `L = current − window`
//!    (the engine's [`TensorKind::PrefillAttention`] is the SUM over the window; `÷ window`);
//! 2. GQA reduction to the KV head by the group **mean** (before pooling — FFY0's order, which
//!    is not kvpress SnapKV's pool-then-mean; the two commute for `avg`, not for `max`);
//! 3. `max_pool1d(kernel, padding = kernel / 2)` (FFY0's default pooling; `avg` selectable);
//! 4. flattened top-`n_kv · base` over every head's region `[start_h + protected, L)` → `count_h`;
//! 5. `cap_h = round(count_h · (1 − α) + int(base · α))`, in `f32` like the torch tensor;
//! 6. per head: protected prefix ∪ own top-`cap_h` over the region ∪ the window `[L, current)`.
//!
//! `base = n_kept − window` per head, where `n_kept` is the engine's `target_len` (or the
//! kvpress-style `int(current · (1 − cr))` when `compression_ratio` is set), exactly as `snapkv`
//! derives its heavy count. FFY0 keeps `base + window` per head on average; the protected prefix
//! is the engine's attention-sink guard and is ADDITIVE, like every other prefill-end arm's.
//!
//! ## Residuals
//!
//! * **Ties.** Max-pooling makes plateaus, so a cut inside a run of equal scores is the common
//!   case. `torch.topk`'s order among equals is implementation-defined; this crate breaks ties
//!   deterministically — flattened: (score desc, head asc, pos asc); per head: STABLE top-k
//!   (lower position first). The fixture oracle uses the same rule and records the ties it saw.
//! * **Window.** FFY0 observes 32 trailing queries; this crate declares 64 on
//!   [`StageCaps::prefill_attn_window`] because a `--aperturb-select` pool refuses candidates
//!   whose windows differ and pyramidkv/snapkv declare 64. `--set window_size` changes the
//!   arithmetic below, not the producer's window (the documented pyramidkv limitation).
//! * **Rounding.** `Σ cap_h` can differ from `n_kv · base` by the blend's rounding (FFY0 too);
//!   the pool's calibration re-asks when a candidate overshoots its budget.
//! * **f32 vs f16** prefill attention: inherited from the SnapKV family, not re-derived here.
//!
//! ## On a ragged cache
//!
//! The stage reads [`StageCtx::head_start`] and ranks only the slots each head still holds: a hole
//! never competes and is never kept, and the protected prefix is the head's first resident
//! positions ([`KeepTopK::start`]). A stage that ignored the starts would name a hole and be
//! refused by the handle ([`CacheOpError::NotResident`]).

use argus_extension_api::{
    CacheHandle, CacheOpError, KVMutationStage, KeepSpec, KeepTopK, MutationPhase, SignalId,
    StageArgs, StageCaps, StageCtx, StageParams, TensorKind, avg_pool1d, compile_keep_top_k,
    max_pool1d, register_kv_mutation_stage,
};

/// The caps for the registration: AdaKV reads the prefill attention; protects no prefix
/// (reference-faithful); drop-only.
const ADAKV_CAPS: StageCaps = StageCaps {
    reads: &[TensorKind::PrefillAttention],
    reads_signals: &[],
    default_protected_prefix: 0,
    produces_merge_plan: false,
    whole_model: false,
    // MUST equal pyramidkv's and snapkv's declaration (64): a `--aperturb-select` pool refuses
    // prefill-attention candidates that want different observation windows, and the engine tests
    // pin the agreement. FFY0's own default is 32 (a documented divergence, see the crate doc).
    prefill_attn_window: Some(64),
};

// ── FFY0-parity arithmetic ───────────────────────────────────────────────────

/// The kvpress `ScorerPress.compress` budget: `n_kept = int(k_len * (1 - compression_ratio))`
/// (Python `int` truncates). Same convention as the `snapkv` crate, so the two arms of a pool
/// derive their per-head count identically.
pub fn adakv_budget(k_len: usize, compression_ratio: f64) -> usize {
    (k_len as f64 * (1.0 - compression_ratio)) as usize
}

/// FFY0 `head_adaptive_capacity`: `round(count · (1 − α) + int(base · α))` per head, computed in
/// `f32` like the torch tensor it comes from (`bincount(..).float()`), rounded half-to-even like
/// `torch.round`. `α = 1` is a uniform `base` per head; `α = 0` is the raw competition.
pub fn adakv_head_caps(counts: &[usize], base: usize, floor_alpha: f64) -> Vec<usize> {
    let floor_cap = (base as f64 * floor_alpha).trunc() as f32;
    let one_minus = (1.0 - floor_alpha) as f32;
    counts
        .iter()
        .map(|&c| {
            (c as f32 * one_minus + floor_cap)
                .round_ties_even()
                .max(0.0) as usize
        })
        .collect()
}

/// Which 1-D pooling the pipeline's step 3 uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// FFY0's default (`pooling='maxpool'`).
    Max,
    /// kvpress SnapKV's choice (`F.avg_pool1d`, count_include_pad).
    Avg,
}

/// The inputs of one layer's AdaKV decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaKvSelect {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    /// Columns of the prefill attention (its prefix width).
    pub cols: usize,
    /// Resident tokens (the frame the keep-set is numbered in).
    pub current: usize,
    /// Observation window — the trailing positions always kept.
    pub window: usize,
    pub kernel: usize,
    pub pooling: Pooling,
    /// Per-head heavy-hitter budget (`n_kept − window`); the heads compete for `n_kv · base`.
    pub base: usize,
    pub floor_alpha: f64,
    /// Protected prefix, force-kept at the front of every head (additive).
    pub protected: usize,
}

/// The AdaKV per-head keep-set selection (FFY0 `update_kv_gqa`) from a per-query-head attention
/// reader. `read_qhead(qh, out)` fills `out[..spec.cols]` with attention head `qh`'s window-summed
/// attention to every prefix key; `head_start[h]` is KV head `h`'s first resident slot (`0` on a
/// uniform cache). Returns `(cap_h, keep_h)` — one ascending keep-list per KV head, of DIFFERENT
/// lengths in general.
pub fn adakv_per_head_keep(
    spec: AdaKvSelect,
    read_qhead: impl Fn(usize, &mut [f32]),
    head_start: &[usize],
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let n_kv = spec.n_kv_heads.max(1);
    let groups = (spec.n_q_heads / n_kv).max(1);
    let heavy_len = spec.current.saturating_sub(spec.window);

    // (1)+(2): window-mean per query head, group-mean to the KV head.
    let inv_window = 1.0f32 / spec.window.max(1) as f32;
    let inv_groups = 1.0f32 / groups as f32;
    let mut row = vec![0.0f32; spec.cols];
    let mut pooled = vec![vec![0.0f32; heavy_len]; n_kv];
    let mut m_kv = vec![0.0f32; heavy_len];
    for (kvh, p) in pooled.iter_mut().enumerate() {
        m_kv.fill(0.0);
        for g in 0..groups {
            read_qhead(kvh * groups + g, &mut row);
            for (acc, &x) in m_kv.iter_mut().zip(row[..heavy_len].iter()) {
                *acc += x * inv_window;
            }
        }
        for x in m_kv.iter_mut() {
            *x *= inv_groups;
        }
        // (3): pooling.
        match spec.pooling {
            Pooling::Max => max_pool1d(&m_kv, spec.kernel, p),
            Pooling::Avg => avg_pool1d(&m_kv, spec.kernel, p),
        }
    }

    // (4): flattened competition over every head's region [start_h + protected, heavy_len).
    let region_start =
        |h: usize| (head_start.get(h).copied().unwrap_or(0) + spec.protected).min(heavy_len);
    let mut flat: Vec<(usize, usize)> = (0..n_kv)
        .flat_map(|h| (region_start(h)..heavy_len).map(move |pos| (h, pos)))
        .collect();
    // STABLE sort by score desc: equal scores keep (head asc, pos asc) — the oracle's rule.
    flat.sort_by(|a, b| {
        pooled[b.0][b.1]
            .partial_cmp(&pooled[a.0][a.1])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let n_win = (n_kv * spec.base).min(flat.len());
    let mut counts = vec![0usize; n_kv];
    for &(h, _) in &flat[..n_win] {
        counts[h] += 1;
    }

    // (5): the floor blend.
    let caps = adakv_head_caps(&counts, spec.base, spec.floor_alpha);

    // (6): per head, its own top-`cap_h` over the region, with the prefix and the window forced.
    let keeps = (0..n_kv)
        .map(|h| {
            let start = head_start.get(h).copied().unwrap_or(0).min(spec.current);
            let scores = &pooled[h];
            compile_keep_top_k(
                KeepTopK {
                    start,
                    current: spec.current,
                    prefix: spec.protected,
                    recent: spec.window,
                    heavy: caps[h],
                },
                |pos| scores.get(pos).copied().unwrap_or(0.0),
            )
        })
        .collect();
    (caps, keeps)
}

// ── config ───────────────────────────────────────────────────────────────────

/// AdaKV knobs. Defaults mirror FFY0 (`kernel_size=7`, `pooling=maxpool`, `floor_alpha=0.2`) with
/// the pool's shared observation window (64). `compression_ratio` is the fraction REMOVED; `0.0`
/// means "the engine's resolved `target_len` is the budget".
#[derive(Clone, Copy, Debug)]
struct AdaKvConfig {
    compression_ratio: f64,
    window_size: usize,
    kernel_size: usize,
    pooling: Pooling,
    floor_alpha: f64,
}

impl Default for AdaKvConfig {
    fn default() -> Self {
        Self {
            compression_ratio: 0.0,
            window_size: 64,
            kernel_size: 7,
            pooling: Pooling::Max,
            floor_alpha: 0.2,
        }
    }
}

impl AdaKvConfig {
    fn from_args(_base: StageParams, args: StageArgs<'_>) -> Self {
        let mut c = AdaKvConfig::default();
        for a in args {
            match a.key {
                "compression_ratio" => {
                    if let Ok(v) = a.val.parse::<f64>() {
                        c.compression_ratio = v.clamp(0.0, 0.999_999);
                    }
                }
                "window_size" => {
                    if let Ok(v) = a.val.parse::<usize>() {
                        c.window_size = v.max(1);
                    }
                }
                "kernel_size" => {
                    if let Ok(v) = a.val.parse::<usize>() {
                        // Odd kernels preserve the length under `padding = k / 2`.
                        c.kernel_size = v.max(1) | 1;
                    }
                }
                "floor_alpha" => {
                    if let Ok(v) = a.val.parse::<f64>() {
                        // FFY0 asserts 0 <= floor_alpha <= 1.
                        c.floor_alpha = v.clamp(0.0, 1.0);
                    }
                }
                "pooling" => {
                    c.pooling = match a.val {
                        "avg" | "avgpool" => Pooling::Avg,
                        _ => Pooling::Max,
                    };
                }
                _ => {}
            }
        }
        c
    }

    /// The kept count per head for a `current`-long cache (see `snapkv`: the engine's `target_len`
    /// itself unless `compression_ratio` is explicit). `None` = no compression.
    fn budget(&self, current: usize, target_len: usize) -> Option<usize> {
        if self.compression_ratio > 0.0 {
            Some(adakv_budget(current, self.compression_ratio))
        } else if target_len > 0 && target_len < current {
            Some(target_len)
        } else {
            None
        }
    }
}

// ── stage ──────────────────────────────────────────────────────────────────

struct AdaKv {
    cfg: AdaKvConfig,
}

impl AdaKv {
    fn new(cfg: AdaKvConfig) -> Self {
        Self { cfg }
    }

    /// The keep-set shape (`None` = no-op). Faithful per-head AdaKV ([`KeepSpec::PerHead`], ragged)
    /// when the prefill attention is usable; otherwise the `snapkv` fallbacks (window-only,
    /// importance-ranked, or recency) layer-wide, which are budget-faithful but not AdaKV.
    fn keep_spec(&self, ctx: &dyn StageCtx) -> Option<KeepSpec> {
        let current = ctx.current_pos();
        if current == 0 {
            return None;
        }
        let raw_budget = self.cfg.budget(current, ctx.target_len())?;
        let window = self.cfg.window_size.min(current);
        let n_kept = raw_budget.clamp(1, current);
        if n_kept >= current {
            return None;
        }
        let protected = ctx.protected_prefix().min(current);
        let n_kv = ctx.n_kv_heads().max(1);
        let head_start: Vec<usize> = (0..n_kv).map(|h| ctx.head_start(h)).collect();
        if n_kept <= window {
            // At or below the window: the `n_kept` most recent, which every head holds (the
            // shared cursor), plus each head's own protected prefix — per head when ragged.
            let floor = head_start.iter().copied().max().unwrap_or(0);
            if floor == 0 {
                let mut keep: Vec<usize> = (0..protected).collect();
                keep.extend((current - n_kept..current).filter(|&p| p >= protected));
                return Some(KeepSpec::LayerWide(keep));
            }
            let heads = head_start
                .iter()
                .map(|&s| {
                    compile_keep_top_k(
                        KeepTopK {
                            start: s,
                            current,
                            prefix: protected,
                            recent: n_kept,
                            heavy: 0,
                        },
                        |_| 0.0,
                    )
                })
                .collect();
            return Some(KeepSpec::PerHead(heads));
        }
        let base = n_kept - window;

        // (1) Faithful per-head AdaKV: needs the prefill attention (per attention head, pre-GQA).
        if let Some(pfa) = ctx.signal(SignalId("attn.prefill_window")) {
            let shape = pfa.shape();
            let n_q = shape.rows;
            let cols = shape.cols;
            let heavy_len = current - window;
            if n_q >= n_kv && n_q % n_kv == 0 && cols >= heavy_len {
                let (_, heads) = adakv_per_head_keep(
                    AdaKvSelect {
                        n_q_heads: n_q,
                        n_kv_heads: n_kv,
                        cols,
                        current,
                        window,
                        kernel: self.cfg.kernel_size,
                        pooling: self.cfg.pooling,
                        base,
                        floor_alpha: self.cfg.floor_alpha,
                        protected,
                    },
                    |qh, out| pfa.read_row(qh, 0, out),
                    &head_start,
                );
                return Some(KeepSpec::PerHead(heads));
            }
        }

        // (2) Degraded fallback — no usable PFA: the same budget layer-wide (importance, else
        //     recency), exactly the `snapkv` fallback. Only meaningful on a uniform cache; on a
        //     ragged one it is issued per head from each head's own start.
        let uniform = head_start.iter().all(|&s| s == 0);
        let spec_for = |start: usize, heavy: usize, recent: usize| KeepTopK {
            start,
            current,
            prefix: protected,
            recent,
            heavy,
        };
        let list = |start: usize| match ctx.importance() {
            Some(imp) => compile_keep_top_k(spec_for(start, base, window), |pos| {
                imp.get(pos).copied().unwrap_or(0.0)
            }),
            None => compile_keep_top_k(spec_for(start, 0, n_kept), |_| 0.0),
        };
        if uniform {
            Some(KeepSpec::LayerWide(list(0)))
        } else {
            Some(KeepSpec::PerHead(
                head_start.iter().map(|&s| list(s)).collect(),
            ))
        }
    }
}

impl KVMutationStage for AdaKv {
    fn name(&self) -> &str {
        "adakv"
    }

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
    "adakv",
    |p, args| Box::new(AdaKv::new(AdaKvConfig::from_args(p, args))),
    ADAKV_CAPS,
    MutationPhase::PrefillEnd
);

#[cfg(test)]
mod tests;
