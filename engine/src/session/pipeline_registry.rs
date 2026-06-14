//! `PipelineRegistry` — L4 stage 저장소 + `PipelineDispatcher` impl.
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.2 (PipelineRegistry) + §5.2.1
//! (driver↔Stage 계약 4건, Phase β-1 확정). 배치: §2.1 규칙 C (no-`mod.rs` 스타일),
//! L4 `engine/src/session/`.
//!
//! # 계약 요약 (INV-DECODE-STAGE-004/005)
//!
//! - submit 순서 = 순회 순서 (통합자 책임, §5.3).
//! - `Continue` → 다음 stage 진행.
//! - `Consumed` → OneShot GC (dispatch 완료 후 1회 제거, `len` 감소).
//!   Persistent stage 가 `Consumed` 를 반환하면 debug_assert 발화.
//! - `Stop(r)` → 즉시 break, `Some(r)` 반환, 후속 stage 미실행.
//! - `Err(e)` → panic (fail-fast; stage.name() + phase 포함 메시지).
//! - `len==0` fast-path: dispatch 진입부 `len.load(Relaxed)==0` 이면
//!   Mutex lock 없이 즉시 `None` 반환.
//!
//! # Phase β 배선 상태
//!
//! β-2 에서 `DecodeLoop::prefill`/`run` 에 배선됨 (`run_until_stop` 은 β-6).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::pipeline::{
    LifecyclePhase, PipelineDispatcher, PipelineStage, StageContext, StageLifecycle, StageOutcome,
    StopReason,
};

/// L4 stage 저장소.
///
/// `Arc<PipelineRegistry>` + 내부 `Mutex` interior mutability — DecodeLoop 이
/// Manager IPC handler 에서 `registry.submit(stage)` 가능(단일 스레드 추론 가정,
/// `INV-018`). `len` AtomicUsize 로 len==0 fast-path(lock-free).
pub struct PipelineRegistry {
    stages: Mutex<Vec<Arc<dyn PipelineStage + Send + Sync>>>,
    /// submit 시 +1, OneShot GC 시 -1. dispatch 진입부 0-체크에 사용.
    len: AtomicUsize,
}

impl PipelineRegistry {
    /// 빈 registry 생성.
    pub fn new() -> Self {
        Self {
            stages: Mutex::new(Vec::new()),
            len: AtomicUsize::new(0),
        }
    }

    /// stage 를 끝에 push. submit 순서 = 실행 순서.
    pub fn submit(&self, stage: Arc<dyn PipelineStage + Send + Sync>) {
        let mut guard = self
            .stages
            .lock()
            .expect("PipelineRegistry stages Mutex poisoned");
        guard.push(stage);
        self.len.fetch_add(1, Ordering::Release);
    }

    /// 현재 등록된 stage 수.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// registry 가 비어 있는지.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for PipelineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineDispatcher for PipelineRegistry {
    /// 등록된 stage 를 submit 순서로 순회하며 각 phase 를 dispatch.
    ///
    /// **len==0 fast-path**: `len.load(Relaxed)==0` 이면 Mutex lock 없이 즉시 `None` 반환.
    fn dispatch(&self, phase: LifecyclePhase, ctx: &mut StageContext<'_>) -> Option<StopReason> {
        // fast-path: 빈 registry 는 lock 없이 즉시 반환.
        if self.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        let mut guard = self
            .stages
            .lock()
            .expect("PipelineRegistry stages Mutex poisoned");

        let mut gc_indices: Vec<usize> = Vec::new();
        let mut stop_reason: Option<StopReason> = None;

        for (i, stage) in guard.iter().enumerate() {
            let outcome = stage.on_phase(&phase, ctx).unwrap_or_else(|e| {
                panic!(
                    "PipelineRegistry: stage '{}' returned Err on phase {:?}: {e}",
                    stage.name(),
                    phase,
                )
            });

            match outcome {
                StageOutcome::Continue => {
                    // 계속 진행.
                }
                StageOutcome::Consumed => {
                    // Persistent stage 가 Consumed 를 반환하면 설계 위반.
                    debug_assert!(
                        stage.lifecycle() == StageLifecycle::OneShot,
                        "PipelineRegistry: Persistent stage '{}' returned Consumed — \
                         Consumed 는 OneShot stage 전용 (INV-DECODE-STAGE-007)",
                        stage.name(),
                    );
                    gc_indices.push(i);
                    // Consumed 는 GC 후 계속(후속 stage 실행).
                }
                StageOutcome::Stop(r) => {
                    stop_reason = Some(r);
                    break;
                }
            }
        }

        // OneShot GC: 뒤에서부터 제거해야 인덱스가 유효하다.
        for &idx in gc_indices.iter().rev() {
            guard.remove(idx);
            self.len.fetch_sub(1, Ordering::Release);
        }

        stop_reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{
        LifecyclePhase, PipelineDispatcher, PipelineStage, Pressure, StageContext, StageLifecycle,
        StageOutcome, StepInfo, StopReason,
    };
    use std::sync::atomic::{AtomicUsize, Ordering as AOrdering};
    use std::sync::{Arc, Mutex};

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

    /// 항상 Continue 를 반환하는 Persistent stage. 호출 횟수를 기록한다.
    struct CountStage {
        name: &'static str,
        count: Arc<AtomicUsize>,
    }

    impl PipelineStage for CountStage {
        fn name(&self) -> &str {
            self.name
        }

        fn on_phase(
            &self,
            _phase: &LifecyclePhase,
            _ctx: &mut StageContext<'_>,
        ) -> anyhow::Result<StageOutcome> {
            self.count.fetch_add(1, AOrdering::SeqCst);
            Ok(StageOutcome::Continue)
        }
    }

    /// 지정 phase 에서 Stop 을 반환하는 stage.
    struct StopStage {
        target_phase: LifecyclePhase,
        reason: StopReason,
        count: Arc<AtomicUsize>,
    }

    impl PipelineStage for StopStage {
        fn name(&self) -> &str {
            "StopStage"
        }

        fn on_phase(
            &self,
            phase: &LifecyclePhase,
            _ctx: &mut StageContext<'_>,
        ) -> anyhow::Result<StageOutcome> {
            self.count.fetch_add(1, AOrdering::SeqCst);
            if phase == &self.target_phase {
                Ok(StageOutcome::Stop(self.reason))
            } else {
                Ok(StageOutcome::Continue)
            }
        }
    }

    /// 지정 phase 에서 Consumed 를 반환하는 OneShot stage.
    struct OneShotStage {
        target_phase: LifecyclePhase,
        fired: Arc<AtomicUsize>,
    }

    impl PipelineStage for OneShotStage {
        fn name(&self) -> &str {
            "OneShotStage"
        }

        fn lifecycle(&self) -> StageLifecycle {
            StageLifecycle::OneShot
        }

        fn on_phase(
            &self,
            phase: &LifecyclePhase,
            _ctx: &mut StageContext<'_>,
        ) -> anyhow::Result<StageOutcome> {
            if phase == &self.target_phase {
                self.fired.fetch_add(1, AOrdering::SeqCst);
                Ok(StageOutcome::Consumed)
            } else {
                Ok(StageOutcome::Continue)
            }
        }
    }

    /// Err 를 반환하는 stage.
    struct ErrStage;

    impl PipelineStage for ErrStage {
        fn name(&self) -> &str {
            "ErrStage"
        }

        fn on_phase(
            &self,
            _phase: &LifecyclePhase,
            _ctx: &mut StageContext<'_>,
        ) -> anyhow::Result<StageOutcome> {
            Err(anyhow::anyhow!("deliberate error"))
        }
    }

    // ── submit 순서 = 순회 순서 ──

    #[test]
    fn submit_order_is_dispatch_order() {
        let registry = PipelineRegistry::new();
        let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

        struct OrderStage {
            name: &'static str,
            order: Arc<Mutex<Vec<&'static str>>>,
        }
        impl PipelineStage for OrderStage {
            fn name(&self) -> &str {
                self.name
            }
            fn on_phase(
                &self,
                _phase: &LifecyclePhase,
                _ctx: &mut StageContext<'_>,
            ) -> anyhow::Result<StageOutcome> {
                self.order.lock().unwrap().push(self.name);
                Ok(StageOutcome::Continue)
            }
        }

        registry.submit(Arc::new(OrderStage {
            name: "A",
            order: order.clone(),
        }));
        registry.submit(Arc::new(OrderStage {
            name: "B",
            order: order.clone(),
        }));
        registry.submit(Arc::new(OrderStage {
            name: "C",
            order: order.clone(),
        }));

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        registry.dispatch(LifecyclePhase::DecodeStart, &mut ctx);

        let observed = order.lock().unwrap().clone();
        assert_eq!(observed, vec!["A", "B", "C"]);
    }

    // ── Consumed GC 1회성 ──

    #[test]
    fn consumed_oneshot_gc_once() {
        let registry = PipelineRegistry::new();
        let fired = Arc::new(AtomicUsize::new(0));

        registry.submit(Arc::new(OneShotStage {
            target_phase: LifecyclePhase::KvMutate,
            fired: fired.clone(),
        }));

        assert_eq!(registry.len(), 1);

        let mut profiler = OpProfiler::new();
        // 1회차: KvMutate 에서 Consumed → GC
        {
            let mut ctx = make_ctx(&mut profiler);
            registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
        }
        assert_eq!(
            fired.load(AOrdering::SeqCst),
            1,
            "첫 dispatch 에서 발화 1회"
        );
        assert_eq!(registry.len(), 0, "GC 후 len=0");

        // 2회차: 이미 GC 됐으므로 발화 없음
        {
            let mut ctx = make_ctx(&mut profiler);
            registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
        }
        assert_eq!(fired.load(AOrdering::SeqCst), 1, "GC 후 재발화 없음");
    }

    // ── Stop break + Some(r) 반환 + 후속 미실행 ──

    #[test]
    fn stop_breaks_and_returns_reason() {
        let registry = PipelineRegistry::new();
        let after_count = Arc::new(AtomicUsize::new(0));

        registry.submit(Arc::new(StopStage {
            target_phase: LifecyclePhase::PostSample,
            reason: StopReason::EosToken,
            count: Arc::new(AtomicUsize::new(0)),
        }));
        // Stop 이후 이 stage 는 실행되면 안 됨
        registry.submit(Arc::new(CountStage {
            name: "after",
            count: after_count.clone(),
        }));

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let result = registry.dispatch(LifecyclePhase::PostSample, &mut ctx);

        assert_eq!(result, Some(StopReason::EosToken));
        assert_eq!(
            after_count.load(AOrdering::SeqCst),
            0,
            "Stop 후 후속 stage 미실행"
        );
    }

    // ── Err panic (fail-fast) ──

    #[test]
    #[should_panic(expected = "returned Err")]
    fn err_panics_fail_fast() {
        let registry = PipelineRegistry::new();
        registry.submit(Arc::new(ErrStage));

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        registry.dispatch(LifecyclePhase::DecodeStart, &mut ctx);
    }

    // ── len==0 fast-path: 빈 registry dispatch 는 lock 진입 전 early return ──

    #[test]
    fn len_zero_fast_path() {
        let registry = PipelineRegistry::new();
        assert_eq!(registry.len(), 0);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        // 빈 registry: lock 진입 없이 None 반환 (코드 경로 검증)
        let result = registry.dispatch(LifecyclePhase::DecodeStart, &mut ctx);
        assert!(result.is_none());
    }

    // ── OpProfiler 재대여: dispatch 2회 연속 호출 시 borrow checker 통과 ──

    #[test]
    fn op_profiler_reborrow_two_dispatches() {
        let registry = PipelineRegistry::new();
        let count = Arc::new(AtomicUsize::new(0));
        registry.submit(Arc::new(CountStage {
            name: "x",
            count: count.clone(),
        }));

        // OpProfiler 1개를 소유하고 dispatch 2회 연속 호출 — borrow checker 통과가 목표.
        let mut profiler = OpProfiler::new();
        {
            let mut ctx = make_ctx(&mut profiler);
            registry.dispatch(LifecyclePhase::PreForward, &mut ctx);
        }
        {
            let mut ctx = make_ctx(&mut profiler);
            registry.dispatch(LifecyclePhase::PostForward, &mut ctx);
        }
        assert_eq!(count.load(AOrdering::SeqCst), 2);
    }
}
