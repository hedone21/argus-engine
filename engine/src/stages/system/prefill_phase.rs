//! `PrefillPhaseStage` — makes prefill observable in the heartbeat.
//!
//! The command poll is the only emitter inside the decode loop, so before this stage a
//! Manager saw `(Decode, Running)` from the first sampled token onward and never learned
//! that prefill had happened at all. A long prompt therefore looked exactly like a hung
//! engine: no heartbeat, and a last-known phase that was a constant.
//!
//! This stage subscribes to the boundaries the driver already dispatches
//! (`DecodeLoop::prefill` fires `PrefillStart` before `Forward::prefill` and `PrefillEnd`
//! after) and stamps the phase there, forcing a heartbeat at each. Mid-prefill reporting
//! is a separate seam — a chunked forward calls
//! [`PrefillProgress`](crate::session::forward::PrefillProgress) on the same adapter —
//! because the chunk boundaries live inside the forward, not on the driver.

use std::sync::{Arc, Mutex};

use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};
use crate::session::resilience_adapter::ResilienceAdapter;

/// Stamps `Phase::Prefill` and emits a heartbeat on the driver's prefill boundaries.
///
/// Shares the single `Arc<Mutex<ResilienceAdapter>>` that `with_resilience` created, the
/// same one `TickStage` and the command-source wrapper hold. Two extra locks per prefill
/// (once per generation, or once per chat turn) — not a hot path.
pub struct PrefillPhaseStage {
    adapter: Arc<Mutex<ResilienceAdapter>>,
}

impl PrefillPhaseStage {
    pub fn new(adapter: Arc<Mutex<ResilienceAdapter>>) -> Self {
        Self { adapter }
    }
}

impl PipelineStage for PrefillPhaseStage {
    fn name(&self) -> &str {
        "system.prefill_phase"
    }

    fn lifecycle(&self) -> StageLifecycle {
        // Persistent: chat reuses one registry across turns, and every turn prefills.
        StageLifecycle::Persistent
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.3): every other phase is somebody else's.
        match phase {
            LifecyclePhase::PrefillStart => self
                .adapter
                .lock()
                .expect("PrefillPhaseStage ResilienceAdapter Mutex poisoned")
                .enter_prefill(),
            LifecyclePhase::PrefillEnd => self
                .adapter
                .lock()
                .expect("PrefillPhaseStage ResilienceAdapter Mutex poisoned")
                .leave_prefill(),
            _ => {}
        }
        Ok(StageOutcome::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::resilience::CommandExecutor;
    use argus_shared::{EngineMessage, EngineState, Phase};
    use std::sync::mpsc;
    use std::time::Duration;

    /// A long heartbeat interval, so anything that arrives got there by being forced.
    fn make_adapter() -> (Arc<Mutex<ResilienceAdapter>>, mpsc::Receiver<EngineMessage>) {
        let (_cmd_tx, cmd_rx) = mpsc::channel();
        let (status_tx, status_rx) = mpsc::channel();
        let executor = CommandExecutor::new(cmd_rx, status_tx, Duration::from_secs(3600));
        (
            Arc::new(Mutex::new(ResilienceAdapter::new(executor))),
            status_rx,
        )
    }

    fn make_ctx(profiler: &mut OpProfiler) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(0),
                prev_token: 0,
            },
            profiler,
        }
    }

    fn fire(stage: &PrefillPhaseStage, phase: LifecyclePhase) {
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        stage.on_phase(&phase, &mut ctx).unwrap();
    }

    fn heartbeats(rx: &mpsc::Receiver<EngineMessage>) -> Vec<(Phase, EngineState)> {
        let mut out = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            if let EngineMessage::Heartbeat(s) = msg {
                out.push((s.phase, s.state));
            }
        }
        out
    }

    /// Entering prefill reports `Prefill` right away, without waiting for the interval —
    /// the transition is what the Manager needs, and it needs it before the silence.
    #[test]
    fn prefill_start_forces_a_prefill_heartbeat() {
        let (adapter, rx) = make_adapter();
        let stage = PrefillPhaseStage::new(Arc::clone(&adapter));

        fire(&stage, LifecyclePhase::PrefillStart);

        assert_eq!(
            heartbeats(&rx),
            vec![(Phase::Prefill, EngineState::Running)],
            "one forced heartbeat stamped Prefill/Running"
        );
    }

    /// Leaving prefill reports again, still as `Prefill`: the payload describes the cache
    /// at the instant prefill ended, and the first decode poll is what stamps `Decode`.
    #[test]
    fn prefill_end_forces_a_second_heartbeat_still_in_prefill() {
        let (adapter, rx) = make_adapter();
        let stage = PrefillPhaseStage::new(Arc::clone(&adapter));

        fire(&stage, LifecyclePhase::PrefillStart);
        fire(&stage, LifecyclePhase::PrefillEnd);

        assert_eq!(
            heartbeats(&rx),
            vec![
                (Phase::Prefill, EngineState::Running),
                (Phase::Prefill, EngineState::Running),
            ],
        );
    }

    /// Every other phase belongs to somebody else — no heartbeat, no phase change.
    #[test]
    fn other_phases_are_noops() {
        let (adapter, rx) = make_adapter();
        let stage = PrefillPhaseStage::new(Arc::clone(&adapter));

        for phase in [
            LifecyclePhase::DecodeStart,
            LifecyclePhase::PostSample,
            LifecyclePhase::KvMutate,
            LifecyclePhase::Finalize,
        ] {
            let mut profiler = OpProfiler::new();
            let mut ctx = make_ctx(&mut profiler);
            let outcome = stage.on_phase(&phase, &mut ctx).unwrap();
            assert!(matches!(outcome, StageOutcome::Continue));
        }

        assert!(heartbeats(&rx).is_empty());
    }

    /// A chunk boundary reports on the clock, not on every chunk: a prompt that finishes
    /// inside one interval must not turn into a burst.
    #[test]
    fn chunk_progress_is_interval_gated() {
        let (adapter, rx) = make_adapter();
        let stage = PrefillPhaseStage::new(Arc::clone(&adapter));

        fire(&stage, LifecyclePhase::PrefillStart);
        let _ = heartbeats(&rx);

        let sink = Arc::clone(&adapter) as Arc<dyn crate::session::forward::PrefillProgress>;
        for _ in 0..5 {
            sink.on_prefill_chunk();
        }

        assert!(
            heartbeats(&rx).is_empty(),
            "the 1h interval has not elapsed, so no chunk heartbeat is due"
        );
    }

    /// ...but once it has elapsed, a chunk boundary is what carries the report — this is
    /// the only thing standing between a multi-second prefill and a dead-looking link.
    #[test]
    fn chunk_progress_reports_once_the_interval_elapsed() {
        let (_cmd_tx, cmd_rx) = mpsc::channel();
        let (status_tx, rx) = mpsc::channel();
        let executor = CommandExecutor::new(cmd_rx, status_tx, Duration::from_millis(0));
        let adapter = Arc::new(Mutex::new(ResilienceAdapter::new(executor)));
        let stage = PrefillPhaseStage::new(Arc::clone(&adapter));

        fire(&stage, LifecyclePhase::PrefillStart);
        let _ = heartbeats(&rx);

        let sink = Arc::clone(&adapter) as Arc<dyn crate::session::forward::PrefillProgress>;
        sink.on_prefill_chunk();

        assert_eq!(
            heartbeats(&rx),
            vec![(Phase::Prefill, EngineState::Running)],
            "the chunk heartbeat keeps the phase the boundary stamped"
        );
    }
}
