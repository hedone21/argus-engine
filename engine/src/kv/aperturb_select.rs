//! Choosing *which* compression to apply, in the engine, by measuring what each one would cost.
//!
//! The Manager's contract says only how much KV may remain (`KvCompress { budget }`). Which
//! technique reaches that budget is the engine's decision, and this module is where it is made:
//! every configured technique is asked what it would retain, [`crate::aperturb`] scores those
//! retained sets against the model's own attention output, and the smallest perturbation wins.
//!
//! ## Why the candidates are asked rather than reimplemented
//!
//! A candidate's keep-set comes from running the technique's own
//! [`KVMutationStage`](argus_extension_api::KVMutationStage) callback through the transactional
//! handle and reading the staged intent back **without committing**
//! ([`plan_mutation_layer`](crate::stages::kv::mutation::plan_mutation_layer)). So the set that is
//! scored is the set the plugin would really have applied — not a restatement of it that could
//! drift — and the winner is applied from that same recorded set rather than by re-running the
//! stage. What was measured is what lands.
//!
//! ## What the pool is
//!
//! The registered techniques named at assembly, and nothing else. The identity candidate is
//! deliberately absent: [`aperturb::decide`] already computes it as the *reference* every candidate
//! is measured against, and offering it as a candidate would let "retain everything" win every
//! compression — scoring exactly zero while answering nothing. A candidate that comes back over
//! budget is dropped for the same reason: it did not answer the request, so its lower score is not
//! a comparison, it is a different question.
//!
//! ## Cost
//!
//! One decision costs `|C|+1` attention passes over the resident cache at the `R` trailing query
//! rows, plus one host mirror of K and V. It is paid when a compression is requested, not per
//! token. `Choice` carries the split so a run can report it rather than assume it.

use std::sync::Arc;

use anyhow::{Context, Result};
use argus_extension_api::{KVMutationStage, StageCaps, TensorKind};

use crate::aperturb::{self, Config, Geom, KeepSets, LayerSource, OutputBasis, Readout};
use crate::inference::prefill_attn::PrefillAttn;
use crate::inference::q_rows::QRowCapture;
use crate::kv::cache_handle::EngineCacheHandle;
use crate::kv::kv_cache::KVCache;
use crate::stages::kv::mutation::{
    PlannedKeep, dequant_snapshot, plan_mutation_layer, plan_prefill_keepset_layer,
};

/// One technique the engine may choose, resolved once at assembly.
pub struct Candidate {
    /// The registry name, which is what the engine reports as its choice.
    pub name: String,
    stage: Box<dyn KVMutationStage>,
    caps: StageCaps,
    /// The attention-sink guard this candidate would have been configured with. Only the
    /// prefill-attention seam surfaces it (`ctx.protected_prefix()`); the mid-decode ctx reports `0`
    /// because the score-fed path applies it upstream.
    protected_prefix: usize,
}

impl Candidate {
    pub fn new(name: impl Into<String>, stage: Box<dyn KVMutationStage>, caps: StageCaps) -> Self {
        Self {
            name: name.into(),
            stage,
            caps,
            protected_prefix: caps.default_protected_prefix,
        }
    }

    /// Override the declared default with the resolved `--protected-prefix`.
    pub fn with_protected_prefix(mut self, n: usize) -> Self {
        self.protected_prefix = n;
        self
    }

    /// Whether this candidate decides off the prefill attention (SnapKV/PyramidKV).
    fn reads_prefill_attn(&self) -> bool {
        self.caps.reads.contains(&TensorKind::PrefillAttention)
    }
}

/// The score signals a candidate's callback reads, borrowed for one decision.
///
/// The same triple the score-fed eviction path routes
/// ([`ExtractedScores::as_args`](crate::kv::eviction::score_fed::ExtractedScores::as_args)) — a
/// score-based technique must see the same importance here that it would have seen had it been
/// applied directly, or the set that gets scored is not the set it would produce.
#[derive(Clone, Copy, Default)]
pub struct Signals<'a> {
    pub importance: Option<&'a [f32]>,
    pub head_scores: Option<&'a [f32]>,
    pub last_attn: Option<&'a [f32]>,
    /// The prompt attention the forward captured, when the producer was armed for a prefill-end
    /// candidate. It is a prompt-era measurement that never grows, so mid-decode it is narrower
    /// than the cache — see [`Selector::plan_one`] for what the engine does about that.
    pub prefill_attn: Option<&'a PrefillAttn>,
}

/// What one candidate came back with.
#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    pub name: String,
    /// The chosen readout's score. Smaller is a smaller deviation from the uncompressed output.
    pub score: f32,
    /// Positions retained over every layer and KV head.
    pub kept_total: usize,
    /// The per-layer budget this candidate was finally asked for. Equal to the Manager's
    /// `target_len` unless [`Selector::plan_calibrated`] had to ask for less.
    pub asked: usize,
}

/// A decision, and what it cost.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    /// The winning technique's registry name.
    pub winner: String,
    /// Every eligible candidate's score, in pool order.
    pub arms: Vec<Arm>,
    /// Candidates that were asked but could not be compared, with why.
    pub excluded: Vec<(String, String)>,
    /// Resident tokens at the decision point.
    pub tokens_before: usize,
    /// Resident tokens after the winner was applied.
    pub tokens_after: usize,
    /// `[layer][kv_head]` retained positions the budget allows in total.
    pub budget_total: usize,
    /// The per-layer budget the Manager's fraction resolved to — what every arm was asked for
    /// first, and what an [`Arm::asked`] below it means was calibrated.
    pub target_len: usize,
    /// Seconds inside [`aperturb::decide`].
    pub decide_s: f64,
    /// Seconds spent putting the cache where the metric can reach it (device mirror + dequantize).
    pub read_s: f64,
    /// What the prompt-attention capture must become now that the winner has been applied.
    ///
    /// The compaction renumbered the cache under it, so the capture the caller holds is about to
    /// describe the wrong keys. `Some` is that capture carried into the new numbering
    /// ([`PrefillAttn::gather`]); `None` means it could not be carried and the caller must drop it.
    /// Always `None` when nothing was compressed — there is then nothing to carry it through.
    pub(crate) prefill_attn: Option<PrefillAttn>,
}

/// Anything that stops the engine from making a choice it can stand behind.
///
/// Distinguished from an error because a caller answering a Manager wants to say *which* — a
/// selector that could not run is a different report from one whose candidates all failed.
#[derive(Debug, Clone, PartialEq)]
pub enum NoChoice {
    /// The cache holds fewer tokens than the metric scores rows.
    TooShort { resident: usize, rows: usize },
    /// The captured query rows no longer describe the resident cache — something renumbered it
    /// after they were captured.
    StaleRows { resident: usize },
    /// Every candidate was excluded, with the reasons in pool order.
    AllExcluded(Vec<(String, String)>),
}

impl std::fmt::Display for NoChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { resident, rows } => write!(
                f,
                "only {resident} token(s) resident but the metric scores {rows} query rows"
            ),
            Self::StaleRows { resident } => write!(
                f,
                "the captured query rows do not cover the {resident} resident token(s) — the cache \
                 was renumbered after they were captured, so there is nothing to measure the \
                 candidates on"
            ),
            Self::AllExcluded(v) => {
                write!(f, "no candidate was comparable: ")?;
                for (i, (name, why)) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{name}: {why}")?;
                }
                Ok(())
            }
        }
    }
}

/// The engine's compression chooser: a fixed candidate pool plus the model constant the metric
/// projects through.
pub struct Selector {
    candidates: Vec<Candidate>,
    /// `V_r Σ_r` per layer — a model constant, built or loaded once.
    basis: Arc<OutputBasis>,
    /// Query heads, which the cache does not know (it holds KV heads).
    n_heads_q: usize,
    readout: Readout,
}

/// An accepted plan: the sets the metric scores, the per-layer plans the winner is applied from,
/// and the per-layer budget the candidate was finally asked for.
type CalibratedPlan = (KeepSets, Vec<PlannedKeep>, usize);

impl Selector {
    /// `candidates` must be non-empty; a selector with nothing to choose between is a
    /// configuration error, not a runtime one.
    pub fn new(
        candidates: Vec<Candidate>,
        basis: Arc<OutputBasis>,
        n_heads_q: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            !candidates.is_empty(),
            "the candidate pool is empty — there is nothing to choose between"
        );
        Ok(Self {
            candidates,
            basis,
            n_heads_q,
            readout: Readout::default(),
        })
    }

    /// How many budgets one candidate may be asked for in a single decision
    /// ([`Self::plan_calibrated`]). Small on purpose: the linear case needs two, and anything
    /// needing more is not tracking the ask.
    const MAX_BUDGET_PROBES: usize = 4;

    /// The pool, in the order ties are broken.
    pub fn names(&self) -> Vec<&str> {
        self.candidates.iter().map(|c| c.name.as_str()).collect()
    }

    /// Ask every candidate what it would retain at `target_ratio`, score them, and apply the winner.
    ///
    /// `caches` is left byte-identical when this returns `Ok(Err(NoChoice))` or `Err`: the planning
    /// pass never commits, and the single commit is the last thing that runs.
    pub fn choose_and_apply(
        &self,
        caches: &mut [KVCache],
        target_ratio: f32,
        q_rows: &QRowCapture,
        signals: Signals<'_>,
    ) -> Result<std::result::Result<Choice, NoChoice>> {
        let n_layers = caches.len();
        anyhow::ensure!(n_layers > 0, "no KV cache layers to compress");
        anyhow::ensure!(
            self.basis.n_layers() == n_layers,
            "the output basis covers {} layers but the cache has {n_layers}",
            self.basis.n_layers()
        );
        let c0 = &caches[0];
        let current_pos = c0.current_pos();
        let n_kv_heads = c0.kv_heads();
        let head_dim = c0.head_dim();
        let q_dim = self.n_heads_q * head_dim;
        anyhow::ensure!(
            self.basis.d() == q_dim,
            "the output basis projects {} inputs but the query rows are {q_dim} wide",
            self.basis.d()
        );

        let rows = q_rows.rows().min(current_pos);
        if rows == 0 || current_pos <= rows {
            return Ok(Err(NoChoice::TooShort {
                resident: current_pos,
                rows: q_rows.rows(),
            }));
        }
        // Decline rather than measure against rows the ring does not actually hold. A compaction
        // renumbers the cache while the capture keeps stamping RoPE positions; the decode loop
        // reports each prune (`QRowCapture::set_drift`) so the two clocks stay reconcilable and a
        // later budget CAN be answered. Without that report this guard was permanent — one
        // compression per session, every later budget silently declined (measured on an S25,
        // 2026-09-02). What still lands here is a genuine gap: a capture that was not armed when
        // those tokens went past.
        if !q_rows.covers(current_pos) {
            return Ok(Err(NoChoice::StaleRows {
                resident: current_pos,
            }));
        }

        // The budget the Manager asked for, in the engine's own per-layer terms — the same
        // `(pos * ratio).max(1)` floor every other keep-set path uses, so a candidate here is
        // asked for exactly the budget it would have been asked for had it been applied directly.
        let target_len = (((current_pos as f32) * target_ratio) as usize).max(1);
        let budget_total = target_len * n_layers * n_kv_heads;

        // Plan first, while nothing has been read back: a candidate that cannot answer costs no
        // device round trip.
        let mut pool: Vec<(String, KeepSets)> = Vec::with_capacity(self.candidates.len());
        let mut plans: Vec<Vec<PlannedKeep>> = Vec::with_capacity(self.candidates.len());
        let mut asked: Vec<usize> = Vec::with_capacity(self.candidates.len());
        let mut excluded: Vec<(String, String)> = Vec::new();
        for cand in &self.candidates {
            match self.plan_calibrated(cand, caches, target_len, budget_total, n_kv_heads, signals)
            {
                Ok(Ok((keep, layers, ask))) => {
                    if let Err(e) = keep.validate(current_pos) {
                        excluded.push((cand.name.clone(), e.to_string()));
                        continue;
                    }
                    pool.push((cand.name.clone(), keep));
                    plans.push(layers);
                    asked.push(ask);
                }
                Ok(Err(why)) => excluded.push((cand.name.clone(), why)),
                // A stage that errors is excluded, not fatal: one broken plugin must not take the
                // whole decision down when the others answered.
                Err(e) => excluded.push((cand.name.clone(), format!("{e:#}"))),
            }
        }
        if pool.is_empty() {
            return Ok(Err(NoChoice::AllExcluded(excluded)));
        }

        let t_read = std::time::Instant::now();
        let src = HostLayers::read(caches, current_pos, n_kv_heads, head_dim, q_rows)?;
        let read_s = t_read.elapsed().as_secs_f64();

        let g = Geom {
            n_layers,
            n_heads_q: self.n_heads_q,
            n_kv_heads,
            head_dim,
            current_pos,
            rows: src.rows,
        };
        let cfg = Config {
            readout: self.readout,
            keep_cells: false,
        };
        let dec = aperturb::decide(&src, &self.basis, &pool, g, &cfg)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        drop(src);

        let arms = dec
            .scored
            .iter()
            .zip(&pool)
            .zip(&asked)
            .map(|((s, (_, keep)), ask)| Arm {
                name: s.name.clone(),
                score: s.scores.get(self.readout),
                kept_total: keep.total(),
                asked: *ask,
            })
            .collect();

        // Apply the winner from the set that was scored, not by re-running the stage.
        let winner = dec.winner;
        for (l, cache) in caches.iter_mut().enumerate() {
            apply_planned(cache, l, n_layers, &plans[winner][l])
                .with_context(|| format!("applying '{}' to layer {l}", pool[winner].0))?;
        }
        let tokens_after = caches[0].current_pos();

        // Carry the prompt attention into the numbering the compaction just imposed. It belongs to
        // the session and not to whoever won, so it is carried whenever the cache actually moved —
        // a technique that reads it can then answer the NEXT budget too, instead of being excluded
        // until decode regrows past the prompt and then readmitted against a capture that no longer
        // describes the cache. The plans are in the pre-compaction numbering, which is exactly what
        // `gather` maps from.
        let prefill_attn = signals
            .prefill_attn
            .filter(|_| tokens_after < current_pos)
            .and_then(|pfa| {
                let plan = &plans[winner];
                pfa.gather(tokens_after, self.n_heads_q, n_kv_heads, |l, h| {
                    plan.get(l).and_then(|p| p.head(h))
                })
            });

        Ok(Ok(Choice {
            winner: pool[winner].0.clone(),
            arms,
            excluded,
            tokens_before: current_pos,
            tokens_after,
            budget_total,
            target_len,
            decide_s: dec.times.total_s(),
            read_s,
            prefill_attn,
        }))
    }

    /// Ask a candidate for the Manager's budget, and if it answers with more than that, ask again
    /// for less — at most [`Self::MAX_BUDGET_PROBES`] times.
    ///
    /// The contract names a budget as a fraction of the resident cache. A technique's own budget
    /// knob need not mean the same thing: kvpress-family arithmetic adds its observation window on
    /// top of the ratio, so asked for `b` it retains `b + window`. Excluding such a candidate for
    /// overshooting would exclude it every time, on a mismatch of vocabulary rather than of quality.
    ///
    /// So the engine calibrates, naming no technique: it subtracts the per-(layer, head) overshoot
    /// from the ask and asks again. A candidate whose retention is a linear function of the ask —
    /// which every budget-driven technique's is — lands inside on the second try. A candidate that
    /// ignores the ask (an absolute-budget technique like faithful H2O) does not shrink at all and
    /// is excluded on the spot, which is what it was before this existed.
    ///
    /// Every probe is a dry run, so a rejected ask costs planning time and not a single byte.
    ///
    /// Returns the accepted plan together with the budget it was finally asked for, so the decision
    /// can report a calibrated arm as calibrated.
    fn plan_calibrated(
        &self,
        cand: &Candidate,
        caches: &mut [KVCache],
        target_len: usize,
        budget_total: usize,
        n_kv_heads: usize,
        signals: Signals<'_>,
    ) -> Result<std::result::Result<CalibratedPlan, String>> {
        let cells = caches.len().saturating_mul(n_kv_heads).max(1);
        let mut ask = target_len;
        let mut prev_total = usize::MAX;
        for _ in 0..Self::MAX_BUDGET_PROBES {
            let (keep, layers) = match self.plan_one(cand, caches, ask, n_kv_heads, signals)? {
                Ok(v) => v,
                Err(why) => return Ok(Err(why)),
            };
            let total = keep.total();
            if total <= budget_total {
                return Ok(Ok((keep, layers, ask)));
            }
            if total >= prev_total {
                return Ok(Err(format!(
                    "retains {total} of {budget_total} budgeted positions, and asking for less does \
                     not shrink it — its budget is not the one the contract names"
                )));
            }
            prev_total = total;
            let step = (total - budget_total).div_ceil(cells).max(1);
            let Some(next) = ask.checked_sub(step).filter(|b| *b > 0) else {
                return Ok(Err(format!(
                    "retains {total} of {budget_total} budgeted positions, and there is no smaller \
                     budget left to ask it for"
                )));
            };
            ask = next;
        }
        Ok(Err(format!(
            "still over the {budget_total}-position budget after {} asks — over budget, so its \
             score is not comparable",
            Self::MAX_BUDGET_PROBES
        )))
    }

    /// Run one candidate's callback over every layer without committing, and assemble the
    /// `KeepSets` the metric scores plus the per-layer plans the winner is applied from.
    ///
    /// `Ok(Err(why))` is a candidate that declined to answer — no keep staged for some layer, or a
    /// per-head plan that does not cover every head.
    #[allow(clippy::type_complexity)]
    fn plan_one(
        &self,
        cand: &Candidate,
        caches: &mut [KVCache],
        target_len: usize,
        n_kv_heads: usize,
        signals: Signals<'_>,
    ) -> Result<std::result::Result<(KeepSets, Vec<PlannedKeep>), String>> {
        let n_layers = caches.len();
        // A prefill-end candidate needs the prompt attention it was registered to read. It is
        // resolved once, before any layer is planned, so a pool member that cannot be asked at all
        // is excluded with that reason rather than reported as having staged nothing.
        let pfa = if cand.reads_prefill_attn() {
            match signals.prefill_attn.map(PrefillAttn::rows) {
                Some(p) if p.len() >= n_layers => Some(p),
                Some(p) => {
                    return Ok(Err(format!(
                        "reads prefill attention, but only {} of {n_layers} layers were captured",
                        p.len()
                    )));
                }
                None => {
                    return Ok(Err(
                        "reads prefill attention, which this run never captured".to_string(),
                    ));
                }
            }
        } else {
            None
        };
        let mut keep =
            KeepSets::with_capacity(n_layers, n_kv_heads, n_layers * n_kv_heads * target_len);
        let mut layers = Vec::with_capacity(n_layers);
        let mut asc: Vec<u32> = Vec::with_capacity(target_len);
        for (l, cache) in caches.iter_mut().enumerate() {
            let planned = match pfa {
                None => Ok(plan_mutation_layer(
                    cand.stage.as_ref(),
                    &cand.caps,
                    cache,
                    l,
                    n_layers,
                    target_len,
                    signals.importance,
                    signals.head_scores,
                    signals.last_attn,
                )?),
                Some(pfa) => self.plan_prefill_layer(cand, cache, l, n_layers, target_len, pfa)?,
            };
            let planned = match planned {
                Ok(p) => p,
                Err(why) => return Ok(Err(why)),
            };
            let Some(p) = planned else {
                return Ok(Err(format!("staged no keep-set for layer {l}")));
            };
            for h in 0..n_kv_heads {
                let Some(list) = p.head(h) else {
                    return Ok(Err(format!(
                        "layer {l}: a per-head plan that does not cover KV head {h}"
                    )));
                };
                asc.clear();
                asc.extend(list.iter().map(|&x| x as u32));
                keep.push(l, h, &asc).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            layers.push(p);
        }
        Ok(Ok((keep, layers)))
    }

    /// Plan one layer for a candidate that decides off the prefill attention.
    ///
    /// The budget arrives mid-decode, but this technique's evidence is a prompt-era measurement:
    /// `pfa[layer]` describes the first `prefix_len` positions and nothing after them. So the engine
    /// splits the layer in two. The stage is shown a cache exactly `prefix_len` long and asked for
    /// `target_len - tail` of it; the `tail` positions decode appended since are kept by the engine
    /// and charged to the same budget ([`PlannedKeep::keep_tail`]).
    ///
    /// This is an engine-side adaptation and is not what the technique does in its own paper, where
    /// it fires once at prefill end with no tail to account for. What it buys is that the technique
    /// answers from its real ranking rather than from the score-free fallback it takes when the
    /// prefill attention is absent, which is the thing that would not be worth ranking.
    ///
    /// A capture wider than the cache is the backstop for a compaction the capture was not carried
    /// through ([`PrefillAttn::gather`] carries it through the ones this selector applies, and the
    /// decode loop drops it on the ones nothing carried it through). It stays because the loop
    /// watches layer 0's occupancy alone, so a keep-set that shrinks only some layers is still
    /// possible, and reading a capture past its width zero-fills in silence.
    ///
    /// The outer `Err(String)` is an exclusion reason, not a failure.
    fn plan_prefill_layer(
        &self,
        cand: &Candidate,
        cache: &mut KVCache,
        layer_idx: usize,
        n_layers: usize,
        target_len: usize,
        pfa: &[Vec<f32>],
    ) -> Result<std::result::Result<Option<PlannedKeep>, String>> {
        let current_pos = cache.current_pos();
        if self.n_heads_q == 0 {
            return Ok(Err("the model reports no query heads".to_string()));
        }
        // The PFA's own width, never the cache's: reading past the data zero-fills silently.
        let prefix_len = pfa[layer_idx].len() / self.n_heads_q;
        if prefix_len == 0 {
            return Ok(Err(format!(
                "layer {layer_idx}: the prefill attention capture is empty"
            )));
        }
        if prefix_len > current_pos {
            return Ok(Err(format!(
                "layer {layer_idx}: the prefill attention covers {prefix_len} positions but only \
                 {current_pos} are resident — the cache was compressed after it was captured"
            )));
        }
        let tail = current_pos - prefix_len;
        let Some(stage_budget) = target_len.checked_sub(tail).filter(|b| *b > 0) else {
            return Ok(Err(format!(
                "the {tail} decode positions its prefill attention never saw already fill the \
                 {target_len}-position budget, so it has nothing left to rank"
            )));
        };
        let mut planned = plan_prefill_keepset_layer(
            cand.stage.as_ref(),
            cache,
            layer_idx,
            n_layers,
            stage_budget,
            &pfa[layer_idx],
            self.n_heads_q,
            cand.protected_prefix,
            prefix_len,
        )?;
        if let Some(p) = planned.as_mut() {
            p.keep_tail(prefix_len, current_pos);
        }
        Ok(Ok(planned))
    }
}

/// Apply one layer's recorded plan through the transactional handle — the same executor a
/// committed mutation uses, so an applied choice is byte-identical to the stage having run.
fn apply_planned(
    cache: &mut KVCache,
    layer_idx: usize,
    n_layers: usize,
    plan: &PlannedKeep,
) -> Result<()> {
    use argus_extension_api::CacheHandle;
    let mut handle = EngineCacheHandle::new(cache, layer_idx, n_layers);
    match plan {
        PlannedKeep::LayerWide(k) => handle.keep(k).map_err(|e| anyhow::anyhow!("{e:?}"))?,
        PlannedKeep::PerHead(h) => {
            let borrowed: Vec<&[usize]> = h.iter().map(|v| v.as_slice()).collect();
            handle
                .keep_per_head(&borrowed)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?
        }
    }
    handle.commit()?;
    Ok(())
}

/// Host-resident `(Q rows, K, V)` for one decision — the metric's [`LayerSource`].
struct HostLayers {
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    rows: usize,
}

impl HostLayers {
    fn read(
        caches: &[KVCache],
        current_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        q_rows: &QRowCapture,
    ) -> Result<Self> {
        let q_snap = q_rows.snapshot(current_pos)?;
        let n_layers = caches.len();
        let mut out = Self {
            q: Vec::with_capacity(n_layers),
            k: Vec::with_capacity(n_layers),
            v: Vec::with_capacity(n_layers),
            rows: q_snap.rows,
        };
        for (l, cache) in caches.iter().enumerate() {
            out.q.push(q_snap.layer_head_major(l, head_dim));
            let (k, v) = read_layer_kv(cache, current_pos, n_kv_heads, head_dim)
                .with_context(|| format!("reading layer {l}'s K/V for the decision"))?;
            out.k.push(k);
            out.v.push(v);
        }
        Ok(out)
    }
}

impl LayerSource for HostLayers {
    fn query_rows(&self, layer: usize) -> &[f32] {
        &self.q[layer]
    }
    fn keys(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }
    fn values(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }
}

/// Dequantize one layer's resident K and V to host f32, `[n_kv_heads][rows][head_dim]`.
///
/// A device-resident cache is mirrored once and both sides read from that mirror; doing it per side
/// would move the same bytes twice.
pub(crate) fn read_layer_kv(
    cache: &KVCache,
    rows: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if cache.k_buffer.buffer().is_gpu_buffer() {
        cache.k_buffer.backend().synchronize()?;
        let host = cache.host_snapshot()?;
        Ok((
            dequant_snapshot(&host, rows, n_kv_heads, head_dim, true),
            dequant_snapshot(&host, rows, n_kv_heads, head_dim, false),
        ))
    } else {
        Ok((
            dequant_snapshot(cache, rows, n_kv_heads, head_dim, true),
            dequant_snapshot(cache, rows, n_kv_heads, head_dim, false),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{CacheHandle, CacheOpError, StageCtx, TensorKind};

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    const LAYERS: usize = 2;
    const HEADS: usize = 1; // query heads == KV heads: one group, no GQA fan-out to reason about.
    const HD: usize = 4;
    const MAX_SEQ: usize = 16;
    const RESIDENT: usize = 8;
    const ROWS: usize = 2;

    /// A cache whose keys are identical at every position — so every logit is equal and the
    /// attention over an admitted set is its plain mean — and whose values carry the position
    /// itself. The reference output of a query row is then the mean of the positions it admits,
    /// which makes each candidate's deviation something the test can compute by hand.
    fn make_cache() -> KVCache {
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, MAX_SEQ, HEADS, HD]);
        let n = MAX_SEQ * HEADS * HD;
        let mut c = KVCache::new(
            Tensor::new(
                sh.clone(),
                Arc::new(SharedBuffer::new(n * 4, DType::F32)),
                be.clone(),
            ),
            Tensor::new(sh, Arc::new(SharedBuffer::new(n * 4, DType::F32)), be),
            MAX_SEQ,
        );
        c.set_current_pos(RESIDENT);
        for pos in 0..RESIDENT {
            let off = c.offset(pos, 0);
            c.k_buffer.as_mut_slice::<f32>()[off..off + HD].fill(0.25);
            c.v_buffer.as_mut_slice::<f32>()[off..off + HD].fill(pos as f32);
        }
        c
    }

    fn caches() -> Vec<KVCache> {
        (0..LAYERS).map(|_| make_cache()).collect()
    }

    /// The untruncated identity projection: the readout then reads the attention output itself, so
    /// a deviation the test computes in value space is the deviation the metric reports.
    fn identity_basis() -> Arc<OutputBasis> {
        let d = HEADS * HD;
        let mut b = vec![0.0f32; d * d];
        for i in 0..d {
            b[i * d + i] = 1.0;
        }
        Arc::new(OutputBasis::from_layers(vec![b; LAYERS], d, d, None).expect("identity basis"))
    }

    /// A ring armed over the trailing `ROWS` positions of every layer, with a constant query — with
    /// constant keys the query's value cannot change the (uniform) attention, only its presence can.
    fn armed_q_rows() -> QRowCapture {
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = crate::memory::galloc::Galloc::new();
        let q_dim = HEADS * HD;
        let mut cap = QRowCapture::new(be.clone(), &mem, LAYERS, ROWS, q_dim)
            .expect("arm the query-row ring");
        let buf = SharedBuffer::new(RESIDENT * q_dim * 4, DType::F32);
        let mut q = Tensor::new(
            Shape::new(vec![1, RESIDENT, HEADS, HD]),
            Arc::new(buf),
            be.clone(),
        );
        q.as_mut_slice::<f32>().fill(1.0);
        for l in 0..LAYERS {
            cap.capture(l, &q, be.as_ref(), 0, RESIDENT, q_dim)
                .expect("capture");
        }
        cap
    }

    /// A ring armed only over positions `[0, n)` — what is left of the capture after something
    /// renumbered the cache under it, since the ring is indexed by absolute position.
    fn q_rows_over(n: usize) -> QRowCapture {
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = crate::memory::galloc::Galloc::new();
        let q_dim = HEADS * HD;
        let mut cap =
            QRowCapture::new(be.clone(), &mem, LAYERS, ROWS, q_dim).expect("arm the ring");
        let buf = SharedBuffer::new(n * q_dim * 4, DType::F32);
        let mut q = Tensor::new(Shape::new(vec![1, n, HEADS, HD]), Arc::new(buf), be.clone());
        q.as_mut_slice::<f32>().fill(1.0);
        for l in 0..LAYERS {
            cap.capture(l, &q, be.as_ref(), 0, n, q_dim)
                .expect("capture");
        }
        cap
    }

    /// A stage that retains a fixed set of positions, whatever the budget — so the test names the
    /// retained set directly instead of inferring it from a policy.
    struct FixedKeep {
        name: &'static str,
        keep: Vec<usize>,
    }

    impl KVMutationStage for FixedKeep {
        fn name(&self) -> &str {
            self.name
        }
        fn on_phase(
            &self,
            _ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            cache.keep(&self.keep)
        }
    }

    /// A stage that stages nothing at all — the "declined to answer" arm.
    struct Silent;

    impl KVMutationStage for Silent {
        fn name(&self) -> &str {
            "silent"
        }
        fn on_phase(
            &self,
            _ctx: &dyn StageCtx,
            _cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            Ok(())
        }
    }

    /// A stage that fails. One broken plugin must not take the decision down with it.
    struct Broken;

    impl KVMutationStage for Broken {
        fn name(&self) -> &str {
            "broken"
        }
        fn on_phase(
            &self,
            _ctx: &dyn StageCtx,
            _cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            Err(CacheOpError::InvalidKeep)
        }
    }

    /// A stage that keeps the `target_len + OVERSHOOT` most recent positions — the shape of a
    /// kvpress-family budget, which adds its observation window on top of the ratio it was asked
    /// for. Retention tracks the ask linearly, so calibration lands it inside the budget.
    struct Windowed;
    const OVERSHOOT: usize = 2;

    impl KVMutationStage for Windowed {
        fn name(&self) -> &str {
            "windowed"
        }
        fn on_phase(
            &self,
            ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            let n = (ctx.target_len() + OVERSHOOT).min(ctx.current_pos());
            cache.keep(&(ctx.current_pos() - n..ctx.current_pos()).collect::<Vec<_>>())
        }
    }

    /// The prefill prefix the PFA covers, of the [`RESIDENT`] positions. The remaining
    /// `RESIDENT - PREFIX` stand for tokens decode appended after the capture.
    const PREFIX: usize = 6;

    /// A stage that ranks the prefix by its prefill attention and keeps the `target_len` best — the
    /// shape of SnapKV/PyramidKV, reduced to the part this seam is about. Records what the ctx told
    /// it, so a test can assert the stage was shown the PFA's window and not the whole cache.
    struct PrefixRanker {
        seen_pos: std::sync::atomic::AtomicUsize,
        seen_cols: std::sync::atomic::AtomicUsize,
    }

    impl KVMutationStage for PrefixRanker {
        fn name(&self) -> &str {
            "prefix_ranker"
        }
        fn on_phase(
            &self,
            ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            use std::sync::atomic::Ordering;
            self.seen_pos.store(ctx.current_pos(), Ordering::Relaxed);
            let Some(pfa) = ctx.tensor(TensorKind::PrefillAttention) else {
                return Ok(());
            };
            let cols = pfa.shape().cols;
            self.seen_cols.store(cols, Ordering::Relaxed);
            let mut row = vec![0.0f32; cols];
            pfa.read_row(0, 0, &mut row);
            let mut order: Vec<usize> = (0..cols).collect();
            order.sort_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            order.truncate(ctx.target_len().min(cols));
            order.sort_unstable();
            cache.keep(&order)
        }
    }

    fn pfa_caps() -> StageCaps {
        StageCaps {
            reads: &[TensorKind::PrefillAttention],
            ..caps()
        }
    }

    /// Prefill attention that ranks positions 3 and 4 above the rest, one row per query head.
    fn pfa_favouring_3_and_4() -> PrefillAttn {
        let mut row = vec![0.1f32; PREFIX];
        row[3] = 0.9;
        row[4] = 0.8;
        PrefillAttn::captured(vec![row; LAYERS])
    }

    fn caps() -> StageCaps {
        StageCaps {
            reads: &[TensorKind::Scores],
            reads_signals: &[],
            default_protected_prefix: 0,
            produces_merge_plan: false,
            whole_model: false,
            prefill_attn_window: None,
        }
    }

    fn fixed(name: &'static str, keep: &[usize]) -> Candidate {
        Candidate::new(
            name,
            Box::new(FixedKeep {
                name,
                keep: keep.to_vec(),
            }),
            caps(),
        )
    }

    fn selector(candidates: Vec<Candidate>) -> Selector {
        Selector::new(candidates, identity_basis(), HEADS).expect("selector")
    }

    /// The V of the surviving slots, layer 0 — what the winner's keep-set actually left behind.
    fn survivors(c: &KVCache) -> Vec<f32> {
        (0..c.current_pos())
            .map(|p| c.v_buffer.as_slice::<f32>()[c.offset(p, 0)])
            .collect()
    }

    /// The pool is ranked by measured deviation, not by pool order. `{3,4}` averages 3.5 against a
    /// reference that averages 3.0 and 3.5 at the two scored rows; `{0,1}` averages 0.5 against the
    /// same reference and is five times further off. Mutation-proof: picking `scored[0]` instead of
    /// the argmin makes `edge_pair` win, since it is first in the pool.
    #[test]
    fn the_smallest_perturbation_wins_and_is_the_set_that_lands() {
        let s = selector(vec![
            fixed("edge_pair", &[0, 1]),
            fixed("mid_pair", &[3, 4]),
        ]);
        let mut cs = caches();
        let q = armed_q_rows();
        let choice = s
            .choose_and_apply(&mut cs, 0.25, &q, Signals::default())
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.winner, "mid_pair");
        assert_eq!(choice.arms.len(), 2);
        let edge = choice.arms.iter().find(|a| a.name == "edge_pair").unwrap();
        let mid = choice.arms.iter().find(|a| a.name == "mid_pair").unwrap();
        assert!(
            mid.score < edge.score,
            "mid_pair {} should score below edge_pair {}",
            mid.score,
            edge.score
        );
        assert_eq!(choice.tokens_before, RESIDENT);
        assert_eq!(choice.tokens_after, 2);
        // What was scored is what landed — every layer, not just the one the winner was picked on.
        for c in &cs {
            assert_eq!(survivors(c), vec![3.0, 4.0]);
        }
    }

    /// A candidate that retains more than the budget is not a cheaper answer to the request, it is
    /// an answer to a different one — so it is excluded rather than allowed to win on score.
    /// `keep_all` ignores the ask, so calibration has no smaller budget to fall back to and stops.
    /// Mutation-proof: dropping the budget gate lets `keep_all` (which perturbs nothing at all) win
    /// every compression, leaving the cache at 8 tokens.
    #[test]
    fn an_over_budget_candidate_is_excluded_rather_than_winning_on_score() {
        let all: Vec<usize> = (0..RESIDENT).collect();
        let s = selector(vec![fixed("keep_all", &all), fixed("mid_pair", &[3, 4])]);
        let mut cs = caches();
        let q = armed_q_rows();
        let choice = s
            .choose_and_apply(&mut cs, 0.25, &q, Signals::default())
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.winner, "mid_pair");
        assert_eq!(
            choice.arms.len(),
            1,
            "only the in-budget candidate is ranked"
        );
        assert_eq!(choice.excluded.len(), 1);
        assert_eq!(choice.excluded[0].0, "keep_all");
        assert!(
            choice.excluded[0].1.contains("budgeted positions"),
            "an ask-ignoring candidate is excluded on the budget: {}",
            choice.excluded[0].1
        );
        assert_eq!(choice.tokens_after, 2);
    }

    /// A candidate whose own budget arithmetic overshoots is re-asked for less rather than
    /// excluded, and the arm reports the budget it was finally asked for. Mutation-proof: removing
    /// the re-ask (returning the exclusion on the first overshoot) drops `windowed` from the pool
    /// entirely, so `choice.arms.len()` falls to 1.
    #[test]
    fn a_candidate_that_overshoots_its_budget_is_re_asked_rather_than_excluded() {
        let s = selector(vec![
            Candidate::new("windowed", Box::new(Windowed), caps()),
            fixed("mid_pair", &[3, 4]),
        ]);
        let mut cs = caches();
        let q = armed_q_rows();
        let choice = s
            .choose_and_apply(&mut cs, 0.5, &q, Signals::default())
            .expect("decide")
            .expect("a choice");
        assert!(
            choice.excluded.is_empty(),
            "nothing should be excluded: {:?}",
            choice.excluded
        );
        assert_eq!(choice.arms.len(), 2);
        let w = choice
            .arms
            .iter()
            .find(|a| a.name == "windowed")
            .expect("the calibrated arm is ranked");
        // Asked for 4, it answers with 6 per layer; the engine subtracts the per-cell overshoot and
        // asks for 2, which it answers with 4 — exactly the budget.
        assert_eq!(choice.target_len, 4);
        assert_eq!(w.asked, 4 - OVERSHOOT);
        assert_eq!(w.kept_total, choice.budget_total);
    }

    /// A prefill-end candidate is asked about the window its prefill attention covers, and the
    /// positions decode appended since are kept by the engine on top of its answer.
    /// Mutation-proof: dropping the `keep_tail` call loses the two decode positions, and showing
    /// the stage `cache.current_pos()` instead of the PFA's width makes `seen_*` report 8.
    #[test]
    fn a_prefill_end_candidate_ranks_its_prefix_and_the_decode_tail_is_kept() {
        use std::sync::atomic::Ordering;
        let ranker = Arc::new(PrefixRanker {
            seen_pos: std::sync::atomic::AtomicUsize::new(0),
            seen_cols: std::sync::atomic::AtomicUsize::new(0),
        });
        struct Shared(Arc<PrefixRanker>);
        impl KVMutationStage for Shared {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn on_phase(
                &self,
                ctx: &dyn StageCtx,
                cache: &mut dyn CacheHandle,
            ) -> Result<(), CacheOpError> {
                self.0.on_phase(ctx, cache)
            }
        }
        let s = selector(vec![Candidate::new(
            "prefix_ranker",
            Box::new(Shared(Arc::clone(&ranker))),
            pfa_caps(),
        )]);
        let mut cs = caches();
        let q = armed_q_rows();
        let pfa = pfa_favouring_3_and_4();
        let choice = s
            .choose_and_apply(
                &mut cs,
                0.5,
                &q,
                Signals {
                    prefill_attn: Some(&pfa),
                    ..Signals::default()
                },
            )
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.winner, "prefix_ranker");
        // The stage saw the PFA's window, not the resident cache.
        assert_eq!(ranker.seen_pos.load(Ordering::Relaxed), PREFIX);
        assert_eq!(ranker.seen_cols.load(Ordering::Relaxed), PREFIX);
        // Budget 4 = the two it ranked out of the prefix, plus the two decode positions it never
        // measured, which the engine kept rather than let fall to a score that does not exist.
        for c in &cs {
            assert_eq!(survivors(c), vec![3.0, 4.0, 6.0, 7.0]);
        }
    }

    /// A prefill-end candidate whose prefill attention was never captured is excluded with that
    /// reason — it is not quietly asked anyway, which is what would send it down its score-free
    /// fallback and rank something the technique does not do.
    #[test]
    fn a_prefill_end_candidate_without_its_attention_is_excluded() {
        let s = selector(vec![
            Candidate::new(
                "prefix_ranker",
                Box::new(PrefixRanker {
                    seen_pos: std::sync::atomic::AtomicUsize::new(0),
                    seen_cols: std::sync::atomic::AtomicUsize::new(0),
                }),
                pfa_caps(),
            ),
            fixed("mid_pair", &[3, 4]),
        ]);
        let mut cs = caches();
        let q = armed_q_rows();
        let choice = s
            .choose_and_apply(&mut cs, 0.5, &q, Signals::default())
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.winner, "mid_pair");
        assert_eq!(choice.excluded.len(), 1);
        assert_eq!(choice.excluded[0].0, "prefix_ranker");
        assert!(
            choice.excluded[0].1.contains("never captured"),
            "{}",
            choice.excluded[0].1
        );
    }

    /// When the decode positions outside the prefill attention already fill the budget, the
    /// candidate is excluded with that reason rather than asked for a budget of nothing.
    #[test]
    fn a_prefill_end_candidate_is_excluded_when_the_decode_tail_fills_the_budget() {
        let s = selector(vec![
            Candidate::new(
                "prefix_ranker",
                Box::new(PrefixRanker {
                    seen_pos: std::sync::atomic::AtomicUsize::new(0),
                    seen_cols: std::sync::atomic::AtomicUsize::new(0),
                }),
                pfa_caps(),
            ),
            fixed("mid_pair", &[3, 4]),
        ]);
        let mut cs = caches();
        let q = armed_q_rows();
        let pfa = pfa_favouring_3_and_4();
        // ratio 0.25 of 8 resident = a 2-position budget, which the 2 decode positions consume.
        let choice = s
            .choose_and_apply(
                &mut cs,
                0.25,
                &q,
                Signals {
                    prefill_attn: Some(&pfa),
                    ..Signals::default()
                },
            )
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.winner, "mid_pair");
        assert_eq!(choice.excluded.len(), 1);
        assert!(
            choice.excluded[0].1.contains("already fill the"),
            "{}",
            choice.excluded[0].1
        );
    }

    /// A prefill-end candidate answers a SECOND budget, because the capture is carried through the
    /// compaction the first one applied rather than left behind in the old numbering.
    ///
    /// The compression keeps prompt positions 3 and 4 (the two the prompt attention favours) plus
    /// the 2-position decode tail, so `V` becomes `[3, 4, 6, 7]` and the carried capture is 2 wide.
    /// The second budget then finds the candidate still comparable, ranking a 2-column capture
    /// against a 4-long cache, and keeps the column that WAS position 3 — `V[0] == 3.0` is the
    /// prompt's own favourite surviving a renumbering.
    ///
    /// Mutation-proof three ways. Not carrying it at all (`prefill_attn: None` on `Choice`) makes
    /// the second decision exclude the candidate — "covers 6 positions but only 4 are resident" —
    /// and `mid_pair` wins with `V == [3, 6, 7]`... which is the same first element, so the
    /// `seen_cols` and exclusion assertions are the ones that catch it. Carrying the ROWS
    /// unchanged (skipping the gather) makes `seen_cols` 6. Gathering by `j` instead of `keep[j]`
    /// puts position 0's 0.1 where 3's 0.9 was, so the retained survivor becomes 4.0.
    #[test]
    fn a_carried_capture_lets_a_prefill_end_candidate_answer_the_next_budget() {
        let ranker = Arc::new(PrefixRanker {
            seen_pos: std::sync::atomic::AtomicUsize::new(0),
            seen_cols: std::sync::atomic::AtomicUsize::new(0),
        });
        struct Shared(Arc<PrefixRanker>);
        impl KVMutationStage for Shared {
            fn name(&self) -> &str {
                "prefix_ranker"
            }
            fn on_phase(
                &self,
                ctx: &dyn StageCtx,
                cache: &mut dyn CacheHandle,
            ) -> Result<(), CacheOpError> {
                self.0.on_phase(ctx, cache)
            }
        }
        let s = selector(vec![Candidate::new(
            "prefix_ranker",
            Box::new(Shared(Arc::clone(&ranker))),
            pfa_caps(),
        )]);
        let mut cs = caches();
        let mut q = armed_q_rows();
        let pfa = pfa_favouring_3_and_4();

        let first = s
            .choose_and_apply(
                &mut cs,
                0.5,
                &q,
                Signals {
                    prefill_attn: Some(&pfa),
                    ..Signals::default()
                },
            )
            .expect("decide")
            .expect("a choice");
        assert_eq!(first.tokens_after, 4);
        assert_eq!(survivors(&cs[0]), vec![3.0, 4.0, 6.0, 7.0]);
        let carried = first
            .prefill_attn
            .expect("the capture is carried, not dropped");
        assert_eq!(carried.rows()[0], vec![0.9, 0.8], "the favoured columns");

        // The ring keeps stamping RoPE positions while the cache is renumbered down; the decode
        // loop reports the gap. Without it the second decision declines on the rows, not the PFA.
        q.set_drift(RESIDENT - first.tokens_after);
        let second = s
            .choose_and_apply(
                &mut cs,
                0.75,
                &q,
                Signals {
                    prefill_attn: Some(&carried),
                    ..Signals::default()
                },
            )
            .expect("decide")
            .expect("a choice");
        assert!(
            second.excluded.is_empty(),
            "the carried capture keeps the candidate comparable: {:?}",
            second.excluded
        );
        assert_eq!(second.winner, "prefix_ranker");
        assert_eq!(
            ranker.seen_cols.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "the stage is shown the carried width, not the prompt's"
        );
        assert_eq!(survivors(&cs[0]), vec![3.0, 6.0, 7.0]);
    }

    /// A capture is carried only when the cache actually moved. A decision that retains everything
    /// leaves the numbering alone, so re-stamping the capture would hand the decode loop's shrink
    /// detector an excuse for a compaction that never happened.
    #[test]
    fn nothing_is_carried_when_nothing_was_compressed() {
        let s = selector(vec![fixed("keep_all", &(0..RESIDENT).collect::<Vec<_>>())]);
        let mut cs = caches();
        let q = armed_q_rows();
        let pfa = pfa_favouring_3_and_4();
        let choice = s
            .choose_and_apply(
                &mut cs,
                1.0,
                &q,
                Signals {
                    prefill_attn: Some(&pfa),
                    ..Signals::default()
                },
            )
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.tokens_after, choice.tokens_before);
        assert!(choice.prefill_attn.is_none());
    }

    /// Query rows that no longer describe the resident cache are a decline, not a measurement and
    /// not a failure. The ring is indexed by absolute position, so a compaction between the capture
    /// and the decision leaves it holding positions that no longer exist.
    ///
    /// Mutation-proof: dropping the `covers` check makes the snapshot's own position assert fire,
    /// which returns `Err` — and a stage that returns `Err` on `KvMutate` panics the pipeline
    /// registry, which is exactly what this configuration did before the check existed.
    #[test]
    fn rows_that_no_longer_describe_the_cache_are_declined_not_measured() {
        let s = selector(vec![fixed("mid_pair", &[3, 4])]);
        let mut cs = caches();
        // The ring holds positions 2..4; the cache is 8 long, so the window it would read is 6..8.
        let q = q_rows_over(4);
        let out = s
            .choose_and_apply(&mut cs, 0.5, &q, Signals::default())
            .expect("a stale ring is a decline, never an error");
        assert!(
            matches!(out, Err(NoChoice::StaleRows { resident: RESIDENT })),
            "{out:?}"
        );
        for c in &cs {
            assert_eq!(c.current_pos(), RESIDENT, "the cache is left alone");
        }
    }

    /// A stage that stages nothing and one that errors are both excluded with a reason, and the
    /// remaining candidate still decides. Mutation-proof: propagating the stage's `Err` instead of
    /// recording it makes this return `Err` and the surviving candidate never runs.
    #[test]
    fn a_silent_or_failing_candidate_is_excluded_and_the_rest_still_decide() {
        let s = selector(vec![
            Candidate::new("silent", Box::new(Silent), caps()),
            Candidate::new("broken", Box::new(Broken), caps()),
            fixed("mid_pair", &[3, 4]),
        ]);
        let mut cs = caches();
        let q = armed_q_rows();
        let choice = s
            .choose_and_apply(&mut cs, 0.25, &q, Signals::default())
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.winner, "mid_pair");
        assert_eq!(choice.excluded.len(), 2);
        assert_eq!(choice.excluded[0].0, "silent");
        assert_eq!(choice.excluded[1].0, "broken");
    }

    /// Every candidate excluded is a report, not a silent no-op — and the cache is untouched, since
    /// the planning pass never commits.
    #[test]
    fn all_excluded_leaves_the_cache_untouched() {
        let all: Vec<usize> = (0..RESIDENT).collect();
        let s = selector(vec![fixed("keep_all", &all)]);
        let mut cs = caches();
        let q = armed_q_rows();
        let r = s
            .choose_and_apply(&mut cs, 0.25, &q, Signals::default())
            .expect("decide");
        match r {
            Err(NoChoice::AllExcluded(v)) => assert_eq!(v.len(), 1),
            other => panic!("expected AllExcluded, got {other:?}"),
        }
        assert_eq!(cs[0].current_pos(), RESIDENT);
        assert_eq!(
            survivors(&cs[0]),
            (0..RESIDENT).map(|p| p as f32).collect::<Vec<_>>()
        );
    }

    /// A cache no longer than the scored window has no uncompressed reference to measure against —
    /// the metric would be comparing the rows to themselves.
    #[test]
    fn a_cache_no_longer_than_the_scored_window_yields_no_choice() {
        let s = selector(vec![fixed("mid_pair", &[0, 1])]);
        let mut cs = caches();
        for c in &mut cs {
            c.set_current_pos(ROWS);
        }
        let q = armed_q_rows();
        let r = s
            .choose_and_apply(&mut cs, 0.25, &q, Signals::default())
            .expect("decide");
        assert!(matches!(r, Err(NoChoice::TooShort { resident: 2, .. })));
        assert_eq!(cs[0].current_pos(), ROWS);
    }

    /// An empty pool is a configuration error, caught where it is configured.
    #[test]
    fn an_empty_pool_is_refused_at_construction() {
        assert!(Selector::new(Vec::new(), identity_basis(), HEADS).is_err());
    }
}
