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
//! rows, plus one host mirror of K and V, plus — with a prefill-end candidate in the pool — one
//! window-attention pass at the ring's `W` rows (below). It is paid when a compression is
//! requested, not per token. `Choice` carries the split so a run can report it rather than assume
//! it.
//!
//! ## Prefill-end candidates
//!
//! SnapKV, PyramidKV and AdaKV rank keys by how much a trailing observation window of queries
//! attended to them. In their papers that window is the prompt's last queries and the ranking
//! happens once, at prefill end. Here a budget arrives mid-decode, so the engine recomputes the
//! same quantity at the decision point: the query-row ring holds the trailing window (sized at
//! assembly to the window the candidates declare), the decision mirrors every layer's K anyway,
//! and the window's softmax over the resident positions — SUM-pooled over the rows, the prefill
//! capture's own format — is what the candidate is shown ([`window_attention`]). It is then
//! ranking the whole resident cache, decode positions included, so it answers any budget; nothing
//! is force-kept, and no prompt-era capture has to be carried through a compaction for it. The
//! ranking rule is the technique's; the placement is the engine's, and is stated as such.

use std::sync::Arc;

use anyhow::{Context, Result};
use argus_extension_api::{KVMutationStage, StageCaps, TensorKind};
use rayon::prelude::*;

use crate::aperturb::kernel::logits_into;
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
    /// Wall time of [`window_attention`] over every layer — the prefill-end candidates' input
    /// (`0.0` when none is in the pool).
    pub window_s: f64,
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
    /// How many of the ring's trailing rows the metric scores at. The ring may hold more — it is
    /// sized to the observation window a prefill-end candidate declares — and the metric's cost
    /// is linear in its rows, so the two are decoupled here.
    metric_rows: usize,
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
            metric_rows: usize::MAX,
        })
    }

    /// Score at the ring's trailing `rows` rather than at every row it holds.
    pub fn with_metric_rows(mut self, rows: usize) -> Self {
        self.metric_rows = rows.max(1);
        self
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
        q_rows: &mut QRowCapture,
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
        let tokens_before_resident = c0.resident_tokens();
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

        // Read back before planning: a prefill-end candidate is shown the window attention over
        // the cache as it stands, and that comes from this mirror. (A pool without one could plan
        // first and save the round trip when every candidate is excluded; the paper's pool is
        // not that pool.)
        let t_read = std::time::Instant::now();
        let src = HostLayers::read(
            caches,
            current_pos,
            n_kv_heads,
            head_dim,
            q_rows,
            self.metric_rows,
        )?;
        let read_s = t_read.elapsed().as_secs_f64();
        let t_window = std::time::Instant::now();
        let window_attn = if self.candidates.iter().any(Candidate::reads_prefill_attn) {
            let gw = Geom {
                n_layers,
                n_heads_q: self.n_heads_q,
                n_kv_heads,
                head_dim,
                current_pos,
                rows: src.window_rows,
            };
            Some(PrefillAttn::captured(window_attention_layers(
                caches, &src, gw,
            )?))
        } else {
            None
        };
        let window_s = t_window.elapsed().as_secs_f64();
        // What the candidates plan from. The prompt capture in `signals` is not it (module
        // header), but it is still what the compaction below carries forward for whoever else
        // reads it.
        let plan_signals = Signals {
            prefill_attn: window_attn.as_ref(),
            ..signals
        };
        let mut pool: Vec<(String, KeepSets)> = Vec::with_capacity(self.candidates.len());
        let mut plans: Vec<Vec<PlannedKeep>> = Vec::with_capacity(self.candidates.len());
        let mut asked: Vec<usize> = Vec::with_capacity(self.candidates.len());
        let mut excluded: Vec<(String, String)> = Vec::new();
        for cand in &self.candidates {
            match self.plan_calibrated(
                cand,
                caches,
                target_len,
                budget_total,
                n_kv_heads,
                plan_signals,
            ) {
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

        // Apply the winner from the set that was scored, not by re-running the stage. A per-head
        // winner is right-aligned on ONE cursor for every layer — the longest head of any layer —
        // so the layers keep a shared `current_pos` (the frame this metric, the next decision's
        // validation and the decode loop's occupancy watch are written against).
        let winner = dec.winner;
        let shared_cursor = plans[winner]
            .iter()
            .map(|p| match p {
                PlannedKeep::PerHead(h) => h.iter().map(|k| k.len()).max(),
                PlannedKeep::LayerWide(_) => None,
            })
            .collect::<Option<Vec<usize>>>()
            .and_then(|v| v.into_iter().max())
            .filter(|&c| caches.iter().all(|cache| c <= cache.current_pos()));
        for (l, cache) in caches.iter_mut().enumerate() {
            apply_planned(cache, l, n_layers, &plans[winner][l], shared_cursor)
                .with_context(|| format!("applying '{}' to layer {l}", pool[winner].0))?;
        }
        // Quoted as a per-head mean: a ragged winner leaves `current_pos` at the longest head.
        let tokens_after = caches[0].resident_tokens();
        let cursor_after = caches[0].current_pos();
        // Tell the ring now, not when the decode loop next notices: the next budget may arrive
        // before that step.
        if cursor_after < current_pos {
            q_rows.renumbered_to(cursor_after);
        }

        // Carry the prompt attention into the numbering the compaction just imposed. It belongs to
        // the session and not to whoever won, so it is carried whenever the cache actually moved —
        // a technique that reads it can then answer the NEXT budget too, instead of being excluded
        // until decode regrows past the prompt and then readmitted against a capture that no longer
        // describes the cache. The plans are in the pre-compaction numbering, which is exactly what
        // `gather` maps from.
        let prefill_attn = signals
            .prefill_attn
            .filter(|_| cursor_after < current_pos)
            .and_then(|pfa| {
                let plan = &plans[winner];
                pfa.gather(cursor_after, self.n_heads_q, n_kv_heads, |l, h| {
                    plan.get(l).and_then(|p| p.head(h))
                })
            });

        Ok(Ok(Choice {
            winner: pool[winner].0.clone(),
            arms,
            excluded,
            tokens_before: tokens_before_resident,
            tokens_after,
            budget_total,
            target_len,
            decide_s: dec.times.total_s(),
            window_s,
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

    /// Plan one layer for a candidate that decides off the observation-window attention.
    ///
    /// `pfa[layer_idx]` is [`window_attention`] over the resident cache for this decision, so its
    /// width is the cache's own and the stage is shown every resident position — the decode tail
    /// included, which its paper never has to rank because it fires before there is one. The
    /// width check stays as the guard it always was: a capture narrower or wider than the cache
    /// ranks keys by another key's score, silently.
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
        // The capture's own width, never the cache's: reading past the data zero-fills silently.
        let width = pfa[layer_idx].len() / self.n_heads_q;
        if width != current_pos {
            return Ok(Err(format!(
                "layer {layer_idx}: the window attention covers {width} positions but \
                 {current_pos} are resident"
            )));
        }
        plan_prefill_keepset_layer(
            cand.stage.as_ref(),
            cache,
            layer_idx,
            n_layers,
            target_len,
            &pfa[layer_idx],
            self.n_heads_q,
            cand.protected_prefix,
            width,
        )
        .map(Ok)
    }
}

/// Apply one layer's recorded plan through the transactional handle — the same executor a
/// committed mutation uses, so an applied choice is byte-identical to the stage having run.
fn apply_planned(
    cache: &mut KVCache,
    layer_idx: usize,
    n_layers: usize,
    plan: &PlannedKeep,
    shared_cursor: Option<usize>,
) -> Result<()> {
    use argus_extension_api::CacheHandle;
    if let (PlannedKeep::PerHead(h), Some(cursor)) = (plan, shared_cursor) {
        return crate::kv::cache_handle::apply_per_head_keep_at(
            cache, layer_idx, n_layers, h, cursor,
        );
    }
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
    /// The metric's rows: the trailing `rows` of the ring, `[n_heads_q][rows][head_dim]`.
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    rows: usize,
    /// Every row the ring holds, `[n_heads_q][window_rows][head_dim]` — the observation window a
    /// prefill-end candidate is shown ([`window_attention`]). The metric's rows are its tail.
    window_q: Vec<Vec<f32>>,
    window_rows: usize,
    /// Per-layer, per-KV-head first resident position (`KVCache::head_starts`).
    starts: Vec<Vec<usize>>,
}

impl HostLayers {
    fn read(
        caches: &[KVCache],
        current_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        q_rows: &QRowCapture,
        metric_rows: usize,
    ) -> Result<Self> {
        let q_snap = q_rows.snapshot(current_pos)?;
        let n_layers = caches.len();
        let rows = q_snap.rows.min(metric_rows).max(1);
        let mut out = Self {
            q: Vec::with_capacity(n_layers),
            k: Vec::with_capacity(n_layers),
            v: Vec::with_capacity(n_layers),
            rows,
            window_q: Vec::with_capacity(n_layers),
            window_rows: q_snap.rows,
            starts: Vec::with_capacity(n_layers),
        };
        for (l, cache) in caches.iter().enumerate() {
            let all = q_snap.layer_head_major(l, head_dim);
            out.q.push(trailing_rows(&all, q_snap.rows, rows, head_dim));
            out.window_q.push(all);
            let (k, v) = read_layer_kv(cache, current_pos, n_kv_heads, head_dim)
                .with_context(|| format!("reading layer {l}'s K/V for the decision"))?;
            out.k.push(k);
            out.v.push(v);
            out.starts.push(cache.head_starts());
        }
        Ok(out)
    }

    fn window_q(&self, layer: usize) -> &[f32] {
        &self.window_q[layer]
    }
}

/// The last `rows` of each head's `all_rows`: `[n_heads_q][all_rows][head_dim]` in,
/// `[n_heads_q][rows][head_dim]` out.
fn trailing_rows(all: &[f32], all_rows: usize, rows: usize, head_dim: usize) -> Vec<f32> {
    if rows >= all_rows {
        return all.to_vec();
    }
    let n_q = all.len() / (all_rows * head_dim).max(1);
    let mut out = Vec::with_capacity(n_q * rows * head_dim);
    for h in 0..n_q {
        let base = (h * all_rows + (all_rows - rows)) * head_dim;
        out.extend_from_slice(&all[base..base + rows * head_dim]);
    }
    out
}

/// Every layer's observation-window attention, on the device when the cache is there.
///
/// The GPU path reads the live K cache in place, so it also skips the host mirror the CPU path
/// re-walks — but it does not make that mirror unnecessary: [`aperturb::decide`] consumes host K
/// *and* V for the metric on the same decision, so `read_layer_kv` stays.
///
/// Falls back to [`window_attention`] whenever the device path cannot express the cache exactly
/// (non-F16 K, a SeqMajor layout, a non-OpenCL backend, a kernel that failed to compile, mixed
/// capacities across layers). `ARGUS_WINDOW_GPU_OFF` forces the fallback;
/// `ARGUS_WINDOW_GPU_VERIFY` runs both and reports the divergence.
fn window_attention_layers(caches: &[KVCache], src: &HostLayers, g: Geom) -> Result<Vec<Vec<f32>>> {
    let cpu = || -> Result<Vec<Vec<f32>>> {
        (0..g.n_layers)
            .map(|l| window_attention(src, l, g))
            .collect()
    };
    #[cfg(feature = "opencl")]
    if window_gpu_enabled() {
        // A dispatch that the device refuses (a work-group shape it cannot schedule, a register
        // spill) must cost the run a slower decision, not the decision itself. Latch it off after
        // the first refusal so a broken device does not pay the probe 18 times.
        match window_attention_layers_opencl(caches, src, g) {
            Ok(Some(gpu)) => {
                if std::env::var("ARGUS_WINDOW_GPU_VERIFY").is_ok() {
                    let host = cpu()?;
                    report_window_divergence(&host, &gpu, g);
                }
                return Ok(gpu);
            }
            Ok(None) => {}
            Err(e) => {
                if !WINDOW_GPU_FAILED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[aperturb-select] the GPU window attention failed, so this run computes \
                         it on the CPU: {e:#}"
                    );
                }
            }
        }
    }
    let _ = caches;
    cpu()
}

/// Set once the device path has refused a dispatch — see [`window_attention_layers`].
#[cfg(feature = "opencl")]
static WINDOW_GPU_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "opencl")]
fn window_gpu_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ARGUS_WINDOW_GPU_OFF").is_err())
        && !WINDOW_GPU_FAILED.load(std::sync::atomic::Ordering::Relaxed)
}

/// `Ok(None)` = this cache is not one the kernel can read; the caller runs the CPU path.
#[cfg(feature = "opencl")]
fn window_attention_layers_opencl(
    caches: &[KVCache],
    src: &HostLayers,
    g: Geom,
) -> Result<Option<Vec<Vec<f32>>>> {
    use crate::backend::opencl::{OpenCLBackend, WindowAttnGeom, get_cl_mem};
    use crate::kv_cache_ops::KVLayout;

    if caches.is_empty() || caches.len() != g.n_layers {
        return Ok(None);
    }
    let backend = caches[0].k_buffer.backend().clone();
    let Some(ocl_be) = backend.as_any().downcast_ref::<OpenCLBackend>() else {
        return Ok(None);
    };
    let capacity = caches[0].capacity();
    let ok = caches.iter().all(|c| {
        c.layout() == KVLayout::HeadMajor
            && c.k_buffer.dtype() == crate::buffer::DType::F16
            && c.k_buffer.buffer().is_gpu_buffer()
            && c.capacity() == capacity
            && c.kv_heads() == g.n_kv_heads
            && c.head_dim() == g.head_dim
    });
    if !ok || capacity < g.current_pos {
        return Ok(None);
    }
    let mems: Vec<&ocl::core::Mem> = match caches
        .iter()
        .map(|c| get_cl_mem(c.k_buffer.buffer().as_ref()))
        .collect::<Result<Vec<_>>>()
    {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    // The decode loop wrote K through the same queue, but say so rather than assume it.
    backend.synchronize()?;
    let starts: Vec<Vec<usize>> = (0..g.n_layers)
        .map(|l| (0..g.n_kv_heads).map(|h| src.head_start(l, h)).collect())
        .collect();
    let qwin: Vec<&[f32]> = (0..g.n_layers).map(|l| src.window_q(l)).collect();
    let geom = WindowAttnGeom {
        n_heads_q: g.n_heads_q,
        n_kv_heads: g.n_kv_heads,
        head_dim: g.head_dim,
        current_pos: g.current_pos,
        rows: g.rows,
        capacity,
    };
    let Some(flat) = ocl_be.window_attention_sum(&mems, &starts, &qwin, geom)? else {
        return Ok(None);
    };
    // Fail loud the way the CPU path does: a NaN column sorts as "equal" in the candidates'
    // top-k and would silently pick a different keep-set.
    anyhow::ensure!(
        flat.iter().all(|v| v.is_finite()),
        "the GPU window attention produced a non-finite score"
    );
    let per_layer = g.n_heads_q * g.current_pos;
    Ok(Some(flat.chunks(per_layer).map(<[f32]>::to_vec).collect()))
}

/// What `ARGUS_WINDOW_GPU_VERIFY` prints: the worst absolute and relative gap, and — the property
/// that actually matters — whether the two agree on the ranking each candidate reads.
#[cfg(feature = "opencl")]
fn report_window_divergence(host: &[Vec<f32>], gpu: &[Vec<f32>], g: Geom) {
    let (mut max_abs, mut max_rel, mut rank_mismatch) = (0.0f32, 0.0f32, 0usize);
    for (a, b) in host.iter().zip(gpu) {
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / x.abs().max(1e-6));
        }
        for h in 0..g.n_heads_q {
            let (ha, hb) = (
                &a[h * g.current_pos..(h + 1) * g.current_pos],
                &b[h * g.current_pos..(h + 1) * g.current_pos],
            );
            let arg = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|p, q| p.1.partial_cmp(q.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
            };
            if arg(ha) != arg(hb) {
                rank_mismatch += 1;
            }
        }
    }
    eprintln!(
        "[window-gpu-verify] layers={} heads={} cols={} max_abs={max_abs:.3e} \
         max_rel={max_rel:.3e} argmax_mismatch={rank_mismatch}",
        host.len(),
        g.n_heads_q,
        g.current_pos,
    );
}

/// The observation-window attention over the resident cache, in the prefill capture's format.
///
/// `[n_heads_q][current_pos]`, SUM-pooled over the ring's `g.rows` query rows: row `t`, at
/// absolute position `g.row_pos(t)`, softmaxes over the keys at or before it that its KV head
/// holds. A ragged head's holes (below its `head_start`) are absent from the softmax and read
/// `0.0` — where a prompt capture carried through a ragged keep puts them too. The logits are the
/// metric's own ([`logits_into`]), so a candidate ranks from the arithmetic it is scored by.
fn window_attention(src: &HostLayers, layer: usize, g: Geom) -> Result<Vec<f32>> {
    let (s, rows, n_rep) = (g.current_pos, g.rows, g.n_rep());
    let mut z = vec![0.0f32; g.logit_len()];
    logits_into(src.window_q(layer), src.keys(layer), &mut z, g)
        .map_err(|e| anyhow::anyhow!("layer {layer}: {e}"))?;
    let heads = (0..g.n_heads_q)
        .into_par_iter()
        .map(|h| -> Result<Vec<f32>> {
            let start = src.head_start(layer, h / n_rep).min(s);
            let zh = &z[h * rows * s..(h + 1) * rows * s];
            let mut acc = vec![0.0f32; s];
            let mut w = Vec::with_capacity(s);
            for t in 0..rows {
                // Causal: row `t` sees the keys at or before its own position.
                let end = (g.row_pos(t) + 1).min(s);
                if end <= start {
                    continue;
                }
                let zr = &zh[t * s + start..t * s + end];
                let m = zr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                anyhow::ensure!(
                    m.is_finite(),
                    "layer {layer}, head {h}: non-finite logit in the window attention"
                );
                w.clear();
                let mut sum = 0.0f32;
                for &zp in zr {
                    let e = (zp - m).exp();
                    w.push(e);
                    sum += e;
                }
                let inv = 1.0 / sum;
                for (j, &e) in w.iter().enumerate() {
                    acc[start + j] += e * inv;
                }
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(heads.concat())
}

impl LayerSource for HostLayers {
    fn query_rows(&self, layer: usize) -> &[f32] {
        &self.q[layer]
    }
    fn head_start(&self, layer: usize, kv_head: usize) -> usize {
        self.starts[layer].get(kv_head).copied().unwrap_or(0)
    }
    fn keys(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }
    fn values(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }
}

/// What one [`window_attention_selfcheck`] run found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSelfcheck {
    /// `false` when the device path declined (no OpenCL, kernel absent, cache not F16 HeadMajor)
    /// and both sides ran the same CPU code — a PASS that proves nothing.
    pub gpu_ran: bool,
    pub max_abs: f32,
    pub max_rel: f32,
    /// Per (layer, query head) disagreements on which column scores highest. This — not the float
    /// gap — is what a candidate's top-k actually reads.
    pub argmax_mismatch: usize,
    /// Columns whose relative gap exceeds 1e-3, over every layer and head.
    pub loose_cols: usize,
    /// Wall time of one whole-model CPU pass — the number `window=` reports today.
    pub cpu_s: f64,
    /// Wall time of one whole-model device pass, upload and readback included. `0.0` when the
    /// device path declined.
    pub gpu_s: f64,
}

/// Run the decision-time window pass on the device and on the CPU over the same synthetic cache,
/// and report how far apart they land.
///
/// The inputs are regenerated from a fixed LCG on both sides rather than stored, the way
/// [`crate::aperturb::tests`] does it, so a drift in the generator fails loudly instead of
/// silently comparing different data. `ragged` gives head `h` a first resident slot of
/// `h * current_pos / (2 * n_kv_heads)`, which is what a per-head keep leaves behind.
///
/// Device-gated by nature: with no OpenCL backend it returns `gpu_ran: false`.
#[allow(clippy::too_many_arguments)]
pub fn window_attention_selfcheck(
    backend: &Arc<dyn crate::backend::Backend>,
    memory: &dyn crate::memory::Memory,
    n_layers: usize,
    n_heads_q: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    current_pos: usize,
    rows: usize,
    ragged: bool,
) -> Result<WindowSelfcheck> {
    use crate::buffer::DType;
    use crate::kv_cache_ops::KVLayout;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    let mut lcg: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };

    let starts: Vec<Vec<usize>> = (0..n_layers)
        .map(|_| {
            (0..n_kv_heads)
                .map(|h| {
                    if ragged {
                        h * current_pos / (2 * n_kv_heads)
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect();

    let mut caches = Vec::with_capacity(n_layers);
    let mut host_k = Vec::with_capacity(n_layers);
    let mut window_q = Vec::with_capacity(n_layers);
    let mut q = Vec::with_capacity(n_layers);
    for layer_starts in &starts {
        // Device K: HeadMajor `[1, kv_heads, capacity, head_dim]` F16, holes included — the host
        // mirror dequantizes them too, and only `head_start` keeps them out of the softmax.
        let mut bits = vec![0u16; n_kv_heads * capacity * head_dim];
        let mut k32 = vec![0.0f32; n_kv_heads * current_pos * head_dim];
        for h in 0..n_kv_heads {
            for p in 0..capacity {
                for d in 0..head_dim {
                    let v = half::f16::from_f32(next());
                    bits[(h * capacity + p) * head_dim + d] = v.to_bits();
                    if p < current_pos {
                        k32[(h * current_pos + p) * head_dim + d] = v.to_f32();
                    }
                }
            }
        }
        let mk_f16 = |data: &[u16]| -> Result<Tensor> {
            let buf = memory.alloc(data.len() * 2, DType::F16)?;
            let mut t = Tensor::new(
                Shape::new(vec![1, n_kv_heads, capacity, head_dim]),
                buf,
                backend.clone(),
            );
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
            backend.write_buffer(&mut t, bytes)?;
            Ok(t)
        };
        let k = mk_f16(&bits)?;
        let v = mk_f16(&bits)?;
        let mut cache =
            KVCache::new_with_geometry(k, v, capacity, n_kv_heads, head_dim, KVLayout::HeadMajor);
        cache.set_current_pos(current_pos);
        cache.set_head_starts(layer_starts);
        caches.push(cache);
        host_k.push(k32);
        let qw: Vec<f32> = (0..n_heads_q * rows * head_dim).map(|_| next()).collect();
        q.push(trailing_rows(&qw, rows, rows, head_dim));
        window_q.push(qw);
    }
    let src = HostLayers {
        q,
        k: host_k,
        v: vec![Vec::new(); n_layers],
        rows,
        window_q,
        window_rows: rows,
        starts,
    };
    let g = Geom {
        n_layers,
        n_heads_q,
        n_kv_heads,
        head_dim,
        current_pos,
        rows,
    };

    let t_cpu = std::time::Instant::now();
    let host = (0..n_layers)
        .map(|l| window_attention(&src, l, g))
        .collect::<Result<Vec<_>>>()?;
    let cpu_s = t_cpu.elapsed().as_secs_f64();
    // Go through the device entry point directly rather than through `window_attention_layers`:
    // a silent decline there would compare the CPU against itself and report a perfect score.
    let t_gpu = std::time::Instant::now();
    #[cfg(feature = "opencl")]
    let device = window_attention_layers_opencl(&caches, &src, g)?;
    #[cfg(not(feature = "opencl"))]
    let device: Option<Vec<Vec<f32>>> = None;
    let gpu_ran = device.is_some();
    let gpu_s = if gpu_ran {
        t_gpu.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let got = match device {
        Some(v) => v,
        None => host.clone(),
    };
    let mut out = WindowSelfcheck {
        gpu_ran,
        max_abs: 0.0,
        max_rel: 0.0,
        argmax_mismatch: 0,
        loose_cols: 0,
        cpu_s,
        gpu_s,
    };
    for (a, b) in host.iter().zip(&got) {
        anyhow::ensure!(
            a.len() == b.len(),
            "window selfcheck: layer length mismatch"
        );
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs();
            let rel = d / x.abs().max(1e-6);
            out.max_abs = out.max_abs.max(d);
            out.max_rel = out.max_rel.max(rel);
            if rel > 1e-3 {
                out.loose_cols += 1;
            }
        }
        for h in 0..n_heads_q {
            let arg = |v: &[f32]| {
                v[h * current_pos..(h + 1) * current_pos]
                    .iter()
                    .enumerate()
                    .max_by(|p, q| p.1.partial_cmp(q.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
            };
            if arg(a) != arg(b) {
                out.argmax_mismatch += 1;
            }
        }
    }
    Ok(out)
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
        // `rows`-bounded mirror: both dequants below read only `[0, rows)` per head, so the
        // capacity-sized tail never needs to cross the bus (`host_snapshot_rows`).
        let host = cache.host_snapshot_rows(rows)?;
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

    /// [`make_cache`] with keys that make the window attend to positions 3 and 4 above the rest
    /// (`⟨q, k⟩` of 2.0 and 1.5 against 0.5 elsewhere, for the ring's all-ones query).
    fn make_cache_favouring_3_and_4() -> KVCache {
        let mut c = make_cache();
        for (pos, k) in [(3usize, 1.0f32), (4, 0.75)] {
            let off = c.offset(pos, 0);
            c.k_buffer.as_mut_slice::<f32>()[off..off + HD].fill(k);
        }
        c
    }

    fn caches_favouring_3_and_4() -> Vec<KVCache> {
        (0..LAYERS)
            .map(|_| make_cache_favouring_3_and_4())
            .collect()
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

    /// A prompt capture over the [`PREFIX`] that ranks `a` above `b` above the rest, one row per
    /// query head.
    fn pfa_favouring(a: usize, b: usize) -> PrefillAttn {
        let mut row = vec![0.1f32; PREFIX];
        row[a] = 0.9;
        row[b] = 0.8;
        PrefillAttn::captured(vec![row; LAYERS])
    }

    fn pfa_favouring_3_and_4() -> PrefillAttn {
        pfa_favouring(3, 4)
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
        let mut q = armed_q_rows();
        let choice = s
            .choose_and_apply(&mut cs, 0.25, &mut q, Signals::default())
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
        let mut q = armed_q_rows();
        let choice = s
            .choose_and_apply(&mut cs, 0.25, &mut q, Signals::default())
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
        let mut q = armed_q_rows();
        let choice = s
            .choose_and_apply(&mut cs, 0.5, &mut q, Signals::default())
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
    /// A prefill-end candidate is shown the window attention over the WHOLE resident cache — the
    /// decode tail included — recomputed from the ring, and ranks all of it.
    ///
    /// Mutation-proof: showing the stage a prompt-width capture makes `seen_*` report `PREFIX`;
    /// force-keeping the tail puts 6 and 7 in the survivors; a window that is not causal gives
    /// column 7 both rows' attention and it ties the rest instead of falling below them.
    #[test]
    fn a_prefill_end_candidate_ranks_the_resident_cache_off_the_ring() {
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
        let mut cs = caches_favouring_3_and_4();
        let mut q = armed_q_rows();
        // No prompt capture at all: the ring is the source.
        let choice = s
            .choose_and_apply(&mut cs, 0.5, &mut q, Signals::default())
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.winner, "prefix_ranker");
        assert!(choice.excluded.is_empty(), "{:?}", choice.excluded);
        assert_eq!(ranker.seen_pos.load(Ordering::Relaxed), RESIDENT);
        assert_eq!(ranker.seen_cols.load(Ordering::Relaxed), RESIDENT);
        // Budget 4: the two keys the window attends to most, then the earliest of the equal rest.
        // The decode positions 6 and 7 are ranked like any other and fall (7 lowest of all: only
        // the last row sees it).
        for c in &cs {
            assert_eq!(survivors(c), vec![0.0, 1.0, 3.0, 4.0]);
        }
    }

    /// The prompt capture in `Signals` is not what a prefill-end candidate ranks from: it is
    /// narrower than the cache and favours other keys, and neither shows in the survivors. It IS
    /// still carried through the compaction, for whoever else reads it.
    #[test]
    fn the_prompt_capture_is_carried_but_not_ranked_from() {
        let s = selector(vec![Candidate::new(
            "prefix_ranker",
            Box::new(PrefixRanker {
                seen_pos: std::sync::atomic::AtomicUsize::new(0),
                seen_cols: std::sync::atomic::AtomicUsize::new(0),
            }),
            pfa_caps(),
        )]);
        let mut cs = caches_favouring_3_and_4();
        let mut q = armed_q_rows();
        let pfa = pfa_favouring(1, 5);
        let choice = s
            .choose_and_apply(
                &mut cs,
                0.5,
                &mut q,
                Signals {
                    prefill_attn: Some(&pfa),
                    ..Signals::default()
                },
            )
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.winner, "prefix_ranker");
        assert!(choice.excluded.is_empty(), "{:?}", choice.excluded);
        for c in &cs {
            assert_eq!(survivors(c), vec![0.0, 1.0, 3.0, 4.0]);
        }
        let carried = choice.prefill_attn.expect("carried through the keep");
        // Prompt positions 0, 1, 3, 4 survived: the capture is 4 wide, column 1 is still the
        // favoured old column 1, and old column 5 is gone with its key.
        assert_eq!(carried.rows()[0], vec![0.1, 0.9, 0.1, 0.1]);
    }

    /// A budget the decode tail alone would once have filled is answered: the tail is ranked like
    /// any other position, so a 2-position budget keeps the two keys the window favours.
    #[test]
    fn the_decode_tail_is_ranked_like_any_other_position() {
        let s = selector(vec![Candidate::new(
            "prefix_ranker",
            Box::new(PrefixRanker {
                seen_pos: std::sync::atomic::AtomicUsize::new(0),
                seen_cols: std::sync::atomic::AtomicUsize::new(0),
            }),
            pfa_caps(),
        )]);
        let mut cs = caches_favouring_3_and_4();
        let mut q = armed_q_rows();
        // ratio 0.25 of 8 resident = a 2-position budget.
        let choice = s
            .choose_and_apply(&mut cs, 0.25, &mut q, Signals::default())
            .expect("decide")
            .expect("a choice");
        assert_eq!(choice.winner, "prefix_ranker");
        assert!(choice.excluded.is_empty(), "{:?}", choice.excluded);
        for c in &cs {
            assert_eq!(survivors(c), vec![3.0, 4.0]);
        }
    }

    /// A prefill-end candidate answers a SECOND budget from the ring alone: after the first
    /// compaction the window is recomputed over the renumbered cache, so the candidate is shown
    /// the 4 survivors and ranks them by where the favoured keys now sit.
    ///
    /// Mutation-proof: recomputing over the pre-compaction width excludes the candidate ("covers
    /// 8 positions but 4 are resident"); ranking by the old numbering keeps old columns 3 and 4,
    /// which are now the keys that WERE 0 and 1.
    #[test]
    fn a_prefill_end_candidate_answers_the_next_budget_from_the_ring() {
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
        let mut cs = caches_favouring_3_and_4();
        let mut q = armed_q_rows();

        let first = s
            .choose_and_apply(&mut cs, 0.5, &mut q, Signals::default())
            .expect("decide")
            .expect("a choice");
        assert_eq!(first.tokens_after, 4);
        assert_eq!(survivors(&cs[0]), vec![0.0, 1.0, 3.0, 4.0]);

        // The ring keeps stamping RoPE positions while the cache was renumbered down. The chooser
        // reported the gap itself, so the second budget is answered before any decode step has
        // run (mutation-proof: drop `renumbered_to` and this declines on stale rows).
        let second = s
            .choose_and_apply(&mut cs, 0.75, &mut q, Signals::default())
            .expect("decide")
            .expect("a choice");
        assert!(second.excluded.is_empty(), "{:?}", second.excluded);
        assert_eq!(second.winner, "prefix_ranker");
        assert_eq!(
            ranker.seen_cols.load(std::sync::atomic::Ordering::Relaxed),
            4,
            "the stage is shown the cache as it stands"
        );
        // Budget 3 of [0, 1, 3, 4]: the favoured keys (now at 2 and 3), then the earliest tie.
        assert_eq!(survivors(&cs[0]), vec![0.0, 3.0, 4.0]);
    }

    /// The metric scores at the ring's TAIL while the window a candidate is shown is the whole
    /// ring: `HostLayers::read` keeps both, and the metric's rows are the last of the window's.
    #[test]
    fn the_metric_rows_are_the_tail_of_the_window() {
        let cs = caches();
        let q = armed_q_rows();
        let src = HostLayers::read(&cs, RESIDENT, HEADS, HD, &q, 1).expect("read");
        assert_eq!((src.rows, src.window_rows), (1, ROWS));
        for l in 0..LAYERS {
            let win = src.window_q(l);
            assert_eq!(win.len(), HEADS * ROWS * HD);
            assert_eq!(src.query_rows(l), &win[(ROWS - 1) * HD..ROWS * HD]);
        }
        let all = HostLayers::read(&cs, RESIDENT, HEADS, HD, &q, usize::MAX).expect("read");
        assert_eq!(all.rows, ROWS);
        assert_eq!(all.query_rows(0), all.window_q(0));
    }

    /// A capture is carried only when the cache actually moved. A decision that retains everything
    /// leaves the numbering alone, so re-stamping the capture would hand the decode loop's shrink
    /// detector an excuse for a compaction that never happened.
    #[test]
    fn nothing_is_carried_when_nothing_was_compressed() {
        let s = selector(vec![fixed("keep_all", &(0..RESIDENT).collect::<Vec<_>>())]);
        let mut cs = caches();
        let mut q = armed_q_rows();
        let pfa = pfa_favouring_3_and_4();
        let choice = s
            .choose_and_apply(
                &mut cs,
                1.0,
                &mut q,
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
        let mut q = q_rows_over(4);
        let out = s
            .choose_and_apply(&mut cs, 0.5, &mut q, Signals::default())
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
        let mut q = armed_q_rows();
        let choice = s
            .choose_and_apply(&mut cs, 0.25, &mut q, Signals::default())
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
        let mut q = armed_q_rows();
        let r = s
            .choose_and_apply(&mut cs, 0.25, &mut q, Signals::default())
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
        let mut q = armed_q_rows();
        let r = s
            .choose_and_apply(&mut cs, 0.25, &mut q, Signals::default())
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
