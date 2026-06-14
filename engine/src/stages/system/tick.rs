//! `TickStage` — v1 `TokenTickSink` 의 `PipelineStage` 화 (Phase β-6 commit C).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5 + roadmap β-6 commit C.
//!
//! `PostSample` phase 에서 register 시점 보유한 `Arc<Mutex<ResilienceAdapter>>` 의 per-token
//! tick 을 발화한다 — v1 `TickWrapper.on_token_generated`(decode_loop (f2))와 동일 효과
//! (executor throughput EMA 적재 + heartbeat token count 채널).
//!
//! **PostSample 구독 근거**: v1 은 stop 토큰에도 (f2) tick 을 발화했다(stop 체크가 tick 후).
//! `PostSample` 은 `DecodeEnd`-Stop break 보다 먼저 발화하므로(run_steps 순서: sample →
//! PostSample → ... → DecodeEnd), stop 토큰의 tick 이 누락되지 않아 v1 "stop 토큰에도 tick"
//! 시맨틱이 보존된다. tick 의 효과(executor token count)는 (f2) 위치(PostSample 직후)와 순서
//! 무관이라 PostSample 이동이 등가다.

use std::sync::{Arc, Mutex};

use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};
use crate::session::resilience_adapter::ResilienceAdapter;

/// `PostSample` phase 에서 `ResilienceAdapter` per-token tick 을 발화하는 Stage.
///
/// `with_resilience` 가 만든 단일 `Arc<Mutex<ResilienceAdapter>>` 를 CmdSrc/Report wrapper 와
/// 공유한다(per-token lock 1회 = 무시 가능 contention, v1 `TickWrapper` 와 동일).
pub struct TickStage {
    adapter: Arc<Mutex<ResilienceAdapter>>,
}

impl TickStage {
    pub fn new(adapter: Arc<Mutex<ResilienceAdapter>>) -> Self {
        Self { adapter }
    }
}

impl PipelineStage for TickStage {
    fn name(&self) -> &str {
        "system.tick"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::Persistent
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.3): PostSample 외 phase 는 무시.
        if *phase == LifecyclePhase::PostSample {
            self.adapter
                .lock()
                .expect("TickStage ResilienceAdapter Mutex poisoned")
                .tick();
        }
        Ok(StageOutcome::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::resilience::{CommandExecutor, KVSnapshot};
    use argus_shared::EngineMessage;
    use std::sync::mpsc;
    use std::time::Duration;

    /// (adapter, status_rx) — status_rx 로 heartbeat payload 를 받아 tick 효과를 검증한다.
    fn make_adapter() -> (Arc<Mutex<ResilienceAdapter>>, mpsc::Receiver<EngineMessage>) {
        let (_cmd_tx, cmd_rx) = mpsc::channel();
        let (status_tx, status_rx) = mpsc::channel();
        // heartbeat interval 0 → tick 후 poll 1회에 즉시 송출(actual_throughput 검증용).
        let executor = CommandExecutor::new(
            cmd_rx,
            status_tx,
            "cpu".to_string(),
            Duration::from_millis(0),
        );
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

    /// PostSample 에서 tick 발화 → executor throughput EMA 적재(heartbeat actual_throughput > 0).
    #[test]
    fn post_sample_fires_tick() {
        let (adapter, status_rx) = make_adapter();
        let stage = TickStage::new(Arc::clone(&adapter));
        let mut profiler = OpProfiler::new();

        // PostSample 2회 발화(첫 tick = prev_time 설정, 두 번째에서 instant_tps 산출 → EMA 적재).
        std::thread::sleep(Duration::from_millis(2));
        {
            let mut ctx = make_ctx(&mut profiler);
            stage
                .on_phase(&LifecyclePhase::PostSample, &mut ctx)
                .unwrap();
        }
        std::thread::sleep(Duration::from_millis(2));
        {
            let mut ctx = make_ctx(&mut profiler);
            stage
                .on_phase(&LifecyclePhase::PostSample, &mut ctx)
                .unwrap();
        }

        // heartbeat 송출 → status_rx 로 actual_throughput 확인(tick 적재 증명).
        adapter
            .lock()
            .unwrap()
            .executor_mut()
            .send_heartbeat_if_due(&KVSnapshot::default());
        let mut throughput = 0.0;
        while let Ok(EngineMessage::Heartbeat(status)) = status_rx.try_recv() {
            throughput = status.actual_throughput;
        }
        assert!(throughput > 0.0, "tick 2회 발화로 throughput EMA 적재");
    }

    /// PostSample 외 phase 는 tick 미발화(Continue).
    #[test]
    fn non_post_sample_is_noop() {
        let (adapter, _rx) = make_adapter();
        let stage = TickStage::new(Arc::clone(&adapter));
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);

        let outcome = stage
            .on_phase(&LifecyclePhase::DecodeEnd, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Continue));
    }
}
