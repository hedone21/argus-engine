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
use argus_extension_api::{KVMutationStage, StageCaps};

use crate::aperturb::{self, Config, Geom, KeepSets, LayerSource, OutputBasis, Readout};
use crate::inference::q_rows::QRowCapture;
use crate::kv::cache_handle::EngineCacheHandle;
use crate::kv::kv_cache::KVCache;
use crate::stages::kv::mutation::{PlannedKeep, dequant_snapshot, plan_mutation_layer};

/// One technique the engine may choose, resolved once at assembly.
pub struct Candidate {
    /// The registry name, which is what the engine reports as its choice.
    pub name: String,
    stage: Box<dyn KVMutationStage>,
    caps: StageCaps,
}

impl Candidate {
    pub fn new(name: impl Into<String>, stage: Box<dyn KVMutationStage>, caps: StageCaps) -> Self {
        Self {
            name: name.into(),
            stage,
            caps,
        }
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
}

/// What one candidate came back with.
#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    pub name: String,
    /// The chosen readout's score. Smaller is a smaller deviation from the uncompressed output.
    pub score: f32,
    /// Positions retained over every layer and KV head.
    pub kept_total: usize,
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
    /// Seconds inside [`aperturb::decide`].
    pub decide_s: f64,
    /// Seconds spent putting the cache where the metric can reach it (device mirror + dequantize).
    pub read_s: f64,
}

/// Anything that stops the engine from making a choice it can stand behind.
///
/// Distinguished from an error because a caller answering a Manager wants to say *which* — a
/// selector that could not run is a different report from one whose candidates all failed.
#[derive(Debug, Clone, PartialEq)]
pub enum NoChoice {
    /// The cache holds fewer tokens than the metric scores rows.
    TooShort { resident: usize, rows: usize },
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

        // The budget the Manager asked for, in the engine's own per-layer terms — the same
        // `(pos * ratio).max(1)` floor every other keep-set path uses, so a candidate here is
        // asked for exactly the budget it would have been asked for had it been applied directly.
        let target_len = (((current_pos as f32) * target_ratio) as usize).max(1);
        let budget_total = target_len * n_layers * n_kv_heads;

        // Plan first, while nothing has been read back: a candidate that cannot answer costs no
        // device round trip.
        let mut pool: Vec<(String, KeepSets)> = Vec::with_capacity(self.candidates.len());
        let mut plans: Vec<Vec<PlannedKeep>> = Vec::with_capacity(self.candidates.len());
        let mut excluded: Vec<(String, String)> = Vec::new();
        for cand in &self.candidates {
            match self.plan_one(cand, caches, target_len, n_kv_heads, signals) {
                Ok(Ok((keep, layers))) => {
                    if keep.total() > budget_total {
                        excluded.push((
                            cand.name.clone(),
                            format!(
                                "retains {} of {budget_total} budgeted positions — over budget, so \
                                 its score is not comparable",
                                keep.total()
                            ),
                        ));
                        continue;
                    }
                    if let Err(e) = keep.validate(current_pos) {
                        excluded.push((cand.name.clone(), e.to_string()));
                        continue;
                    }
                    pool.push((cand.name.clone(), keep));
                    plans.push(layers);
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
            .map(|(s, (_, keep))| Arm {
                name: s.name.clone(),
                score: s.scores.get(self.readout),
                kept_total: keep.total(),
            })
            .collect();

        // Apply the winner from the set that was scored, not by re-running the stage.
        let winner = dec.winner;
        for (l, cache) in caches.iter_mut().enumerate() {
            apply_planned(cache, l, n_layers, &plans[winner][l])
                .with_context(|| format!("applying '{}' to layer {l}", pool[winner].0))?;
        }

        Ok(Ok(Choice {
            winner: pool[winner].0.clone(),
            arms,
            excluded,
            tokens_before: current_pos,
            tokens_after: caches[0].current_pos(),
            budget_total,
            decide_s: dec.times.total_s(),
            read_s,
        }))
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
        let mut keep =
            KeepSets::with_capacity(n_layers, n_kv_heads, n_layers * n_kv_heads * target_len);
        let mut layers = Vec::with_capacity(n_layers);
        let mut asc: Vec<u32> = Vec::with_capacity(target_len);
        for (l, cache) in caches.iter_mut().enumerate() {
            let planned = plan_mutation_layer(
                cand.stage.as_ref(),
                &cand.caps,
                cache,
                l,
                n_layers,
                target_len,
                signals.importance,
                signals.head_scores,
                signals.last_attn,
            )?;
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
    /// Mutation-proof: dropping the `keep.total() > budget_total` gate lets `keep_all` (which
    /// perturbs nothing at all) win every compression, leaving the cache at 8 tokens.
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
        assert!(choice.excluded[0].1.contains("over budget"));
        assert_eq!(choice.tokens_after, 2);
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
