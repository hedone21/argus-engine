//! The `KvMutate`-phase stage that makes the engine's own compression choice.
//!
//! The command-driven twin of [`EvictionStage::one_shot_scored`](super::eviction::EvictionStage):
//! same lifecycle, same UER (unwrap–act–rewrap) over the layer handles, same score extraction. What
//! differs is who decides. `EvictionStage` applies the one technique the CLI configured;
//! this one hands the Manager's budget to a [`Selector`], which asks every configured technique what
//! it would retain and keeps the answer that moves the model's output least.
//!
//! Submitted once per accepted `KvCompress`, and `Consumed` after it fires — a directive is one
//! decision, not a standing policy.

use std::sync::{Arc, Mutex};

use crate::inference::q_rows::QRowCapture;
use crate::inference::signal_runtime::SignalRuntime;
use crate::kv::aperturb_select::{Selector, Signals};
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};

pub struct AperturbSelectStage {
    /// Register-time layer handles; enumeration order is layer index (INV-STAGE-LAYER-HANDLE).
    handles: Vec<Arc<StandardFormat>>,
    selector: Arc<Selector>,
    /// The trailing query rows the forward captured, shared with `ModelForward`.
    q_rows: Arc<Mutex<Option<QRowCapture>>>,
    /// The attention-score accumulator a score-based candidate reads through its ctx — the same
    /// cell `EvictionStage` extracts from, so a candidate sees here what it would have seen had it
    /// been applied directly.
    score_cell: Arc<Mutex<Option<SignalRuntime>>>,
    /// The per-layer prefill attention the forward captured, shared with `ModelForward`. Armed only
    /// when a pooled candidate declares it reads `PrefillAttention`; `None` inside otherwise.
    prefill_attn: Arc<Mutex<Option<Vec<Vec<f32>>>>>,
    /// Present when scores may live on-device; the sync before the read is then live.
    backend: Option<Arc<dyn crate::backend::Backend>>,
    /// The fraction of the resident cache the Manager asked to keep.
    target_ratio: f32,
}

impl AperturbSelectStage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        handles: Vec<Arc<StandardFormat>>,
        selector: Arc<Selector>,
        q_rows: Arc<Mutex<Option<QRowCapture>>>,
        target_ratio: f32,
        score_cell: Arc<Mutex<Option<SignalRuntime>>>,
        prefill_attn: Arc<Mutex<Option<Vec<Vec<f32>>>>>,
        backend: Option<Arc<dyn crate::backend::Backend>>,
    ) -> Self {
        Self {
            handles,
            selector,
            q_rows,
            score_cell,
            prefill_attn,
            backend,
            target_ratio,
        }
    }

    /// Decide and apply, reporting what was chosen.
    ///
    /// A decision that cannot be made is reported and left alone rather than degraded into some
    /// other compression: the Manager is told what the cache actually did through the dispatcher's
    /// read-back, and substituting an unscored technique here would make that answer a fiction.
    fn run_selection(&self) -> anyhow::Result<()> {
        let guard = self.q_rows.lock().unwrap_or_else(|e| e.into_inner());
        let Some(q_rows) = guard.as_ref() else {
            eprintln!(
                "[aperturb-select] declined: the query-row capture is not armed, so there are no \
                 rows to measure the candidates on"
            );
            return Ok(());
        };

        // Scores may have accumulated on-device; the same sync `EvictionStage` does before reading.
        if let Some(be) = self.backend.as_ref() {
            let mut cell = self
                .score_cell
                .lock()
                .expect("aperturb-select score_cell Mutex poisoned");
            if let Some(rt) = cell.as_mut() {
                rt.ensure_coherent(be.as_ref());
            }
        }
        // Owned snapshots, so the per-layer planning callbacks read them without holding the cell
        // lock. The same triple `KVMutationDriverStage` hands a stage that is being applied — a
        // candidate must see here exactly what it would have seen there.
        let (importance, head_scores, last_attn) = {
            let cell = self
                .score_cell
                .lock()
                .expect("aperturb-select score_cell Mutex poisoned");
            match cell
                .as_ref()
                .and_then(|rt| rt.view())
                .filter(|acc| acc.is_active())
            {
                Some(acc) => (
                    Some(acc.importance_scores().to_vec()),
                    acc.head_importance_scores().map(|s| s.to_vec()),
                    acc.last_step_head_attn().map(|s| s.to_vec()),
                ),
                None => (None, None, None),
            }
        };
        // The prefill attention is held across the decision, not copied: it is
        // `n_layers * n_heads_q * prompt_len` floats, and the producer never writes it again after
        // prefill (`ModelForward` publishes it once, on the final chunk).
        let pfa_guard = self.prefill_attn.lock().unwrap_or_else(|e| e.into_inner());
        let signals = Signals {
            importance: importance.as_deref(),
            head_scores: head_scores.as_deref(),
            last_attn: last_attn.as_deref(),
            prefill_attn: pfa_guard.as_deref(),
        };

        let mut temp: Vec<KVCache> = self.handles.iter().map(|f| f.take_inner()).collect();
        let outcome = self
            .selector
            .choose_and_apply(&mut temp, self.target_ratio, q_rows, signals);
        for (f, c) in self.handles.iter().zip(temp) {
            f.put_inner(c);
        }
        drop(pfa_guard);

        match outcome? {
            Ok(choice) => {
                let arms = choice
                    .arms
                    .iter()
                    .map(|a| {
                        // `@n` marks an arm whose own budget arithmetic overshot, so the engine
                        // asked it for `n` per layer to land on the budget the Manager set.
                        if a.asked == choice.target_len {
                            format!("{}={:.4e}", a.name, a.score)
                        } else {
                            format!("{}={:.4e}@{}", a.name, a.score, a.asked)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!(
                    "[aperturb-select] budget={:.3} {} → {} tokens, chose '{}' [{arms}] \
                     decide={:.3}s read={:.3}s",
                    self.target_ratio,
                    choice.tokens_before,
                    choice.tokens_after,
                    choice.winner,
                    choice.decide_s,
                    choice.read_s,
                );
                for (name, why) in &choice.excluded {
                    eprintln!("[aperturb-select]   excluded '{name}': {why}");
                }
                // The KV geometry moved, so every accumulated score is now indexed by positions
                // that no longer exist — the same reset `EvictionStage` does after a real eviction.
                if choice.tokens_after < choice.tokens_before {
                    let mut cell = self
                        .score_cell
                        .lock()
                        .expect("aperturb-select score_cell Mutex poisoned");
                    if let Some(rt) = cell.as_mut() {
                        match self.backend.as_ref() {
                            Some(be) => rt.reset(be.as_ref()),
                            None => rt.reset_host_only(),
                        }
                    }
                }
            }
            // One line per accepted directive, not per decode step: this stage is `OneShot`, so it
            // runs once and is collected. A budget that produced no compression and no message
            // reads exactly like one that was never delivered.
            Err(e) => {
                eprintln!("[aperturb-select] declined: {e}");
            }
        }
        Ok(())
    }
}

impl PipelineStage for AperturbSelectStage {
    fn name(&self) -> &str {
        "kv.aperturb_select"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::OneShot
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        if *phase != LifecyclePhase::KvMutate {
            return Ok(StageOutcome::Continue);
        }
        self.run_selection()?;
        Ok(StageOutcome::Consumed)
    }
}
