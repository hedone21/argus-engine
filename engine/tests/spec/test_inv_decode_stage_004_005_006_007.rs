//! INV-DECODE-STAGE-004~007: driver↔Stage 계약 + PipelineRegistry 의미론 검증.
//!
//! Architect 명세 B-2 (Phase β-1). SSOT: `arch/pipeline_stage_design_v2.md`
//! §5.2.1 (driver↔Stage 계약 4건) + §5.2 (PipelineRegistry).
//!
//! - **INV-DECODE-STAGE-004**: `on_phase` Result ↔ dispatcher Err→panic fail-fast.
//!   Continue/Consumed/Stop(r) 의미론.
//! - **INV-DECODE-STAGE-005**: submit 순서 = 순회 순서. EvictionStage 트리거 phase.
//! - **INV-DECODE-STAGE-006**: StageContext 2-field 슬림 (`step` + `profiler`).
//! - **INV-DECODE-STAGE-007**: Consumed = OneShot 전용 GC.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use argus_engine::observability::profile::OpProfiler;
use argus_engine::pipeline::{
    LifecyclePhase, PipelineDispatcher, PipelineStage, Pressure, StageContext, StageLifecycle,
    StageOutcome, StepInfo, StopReason,
};
use argus_engine::session::pipeline_registry::PipelineRegistry;

// ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────

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

/// 항상 Continue 를 반환하는 Persistent stage.
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
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(StageOutcome::Continue)
    }
}

/// 지정 phase 에서만 Stop 을 반환하는 Persistent stage.
struct StopOnPhaseStage {
    target: LifecyclePhase,
    reason: StopReason,
    count: Arc<AtomicUsize>,
}

impl PipelineStage for StopOnPhaseStage {
    fn name(&self) -> &str {
        "StopOnPhaseStage"
    }
    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        self.count.fetch_add(1, Ordering::SeqCst);
        if phase == &self.target {
            Ok(StageOutcome::Stop(self.reason))
        } else {
            Ok(StageOutcome::Continue)
        }
    }
}

/// 지정 phase 에서 Consumed 를 반환하는 OneShot stage.
struct OneShotOnPhase {
    target: LifecyclePhase,
    fired: Arc<AtomicUsize>,
}

impl PipelineStage for OneShotOnPhase {
    fn name(&self) -> &str {
        "OneShotOnPhase"
    }
    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::OneShot
    }
    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        if phase == &self.target {
            self.fired.fetch_add(1, Ordering::SeqCst);
            Ok(StageOutcome::Consumed)
        } else {
            Ok(StageOutcome::Continue)
        }
    }
}

/// 항상 Err 를 반환하는 stage.
struct AlwaysErr;

impl PipelineStage for AlwaysErr {
    fn name(&self) -> &str {
        "AlwaysErr"
    }
    fn on_phase(
        &self,
        _phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        Err(anyhow::anyhow!("deliberate error for fail-fast test"))
    }
}

/// Persistent 인데 Consumed 를 반환하는 위반 stage (debug_assert 트리거).
struct PersistentReturnsConsumed;

impl PipelineStage for PersistentReturnsConsumed {
    fn name(&self) -> &str {
        "PersistentReturnsConsumed"
    }
    // lifecycle() 기본 = Persistent
    fn on_phase(
        &self,
        _phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        Ok(StageOutcome::Consumed)
    }
}

// ── mock held-handle: pos 감소를 interior-mutate 로 표현 ──────────────────────

/// eviction pos-환류 계약(§5.2.1 (가)) 을 시뮬레이션하는 mock held handle.
///
/// 실제 EvictionStage 는 `Arc<StandardFormat>` 를 interior-mutate 해 `current_pos`
/// 를 감소시키고, driver 가 dispatch 후 held handle 의 `current_pos()` 를 비교해
/// pos 를 갱신 + `forward.on_kv_prune()` 를 1회 호출한다.
struct MockHeldHandle {
    current_pos: std::sync::atomic::AtomicUsize,
}

impl MockHeldHandle {
    fn new(pos: usize) -> Self {
        Self {
            current_pos: std::sync::atomic::AtomicUsize::new(pos),
        }
    }
    fn current_pos(&self) -> usize {
        self.current_pos.load(Ordering::SeqCst)
    }
    fn set_pos(&self, pos: usize) {
        self.current_pos.store(pos, Ordering::SeqCst);
    }
}

/// mock held handle 의 pos 를 감소시키는 stage.
struct EvictPosStage {
    handle: Arc<MockHeldHandle>,
    new_pos: usize,
}

impl PipelineStage for EvictPosStage {
    fn name(&self) -> &str {
        "EvictPosStage"
    }
    fn on_phase(
        &self,
        _phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        self.handle.set_pos(self.new_pos);
        Ok(StageOutcome::Continue)
    }
}

// ── INV-DECODE-STAGE-004: outcome 의미론 ──────────────────────────────────────

/// Continue 는 다음 stage 를 실행시킨다.
#[test]
fn test_inv_decode_stage_004_outcome_continue_advances() {
    let registry = PipelineRegistry::new();
    let c1 = Arc::new(AtomicUsize::new(0));
    let c2 = Arc::new(AtomicUsize::new(0));

    registry.submit(Arc::new(CountStage {
        name: "s1",
        count: c1.clone(),
    }));
    registry.submit(Arc::new(CountStage {
        name: "s2",
        count: c2.clone(),
    }));

    let mut profiler = OpProfiler::new();
    let mut ctx = make_ctx(&mut profiler);
    let result = registry.dispatch(LifecyclePhase::DecodeStart, &mut ctx);

    assert!(result.is_none());
    assert_eq!(c1.load(Ordering::SeqCst), 1, "첫 stage 실행됨");
    assert_eq!(
        c2.load(Ordering::SeqCst),
        1,
        "Continue 로 두 번째 stage 도 실행됨"
    );
}

/// Stop(r) 는 즉시 break + Some(r) 반환 + 후속 미실행.
#[test]
fn test_inv_decode_stage_004_outcome_stop_breaks() {
    let registry = PipelineRegistry::new();
    let after = Arc::new(AtomicUsize::new(0));

    registry.submit(Arc::new(StopOnPhaseStage {
        target: LifecyclePhase::PostSample,
        reason: StopReason::EosToken,
        count: Arc::new(AtomicUsize::new(0)),
    }));
    registry.submit(Arc::new(CountStage {
        name: "after",
        count: after.clone(),
    }));

    let mut profiler = OpProfiler::new();
    let mut ctx = make_ctx(&mut profiler);
    let result = registry.dispatch(LifecyclePhase::PostSample, &mut ctx);

    assert_eq!(result, Some(StopReason::EosToken), "Stop reason 전달됨");
    assert_eq!(after.load(Ordering::SeqCst), 0, "Stop 이후 stage 미실행");
}

/// Consumed 는 OneShot stage GC 후 후속 계속.
#[test]
fn test_inv_decode_stage_004_outcome_consumed_oneshot_only() {
    let registry = PipelineRegistry::new();
    let fired = Arc::new(AtomicUsize::new(0));
    let after = Arc::new(AtomicUsize::new(0));

    registry.submit(Arc::new(OneShotOnPhase {
        target: LifecyclePhase::KvMutate,
        fired: fired.clone(),
    }));
    registry.submit(Arc::new(CountStage {
        name: "after",
        count: after.clone(),
    }));

    let mut profiler = OpProfiler::new();

    // 1회: Consumed → GC, 후속 stage 는 계속 실행
    {
        let mut ctx = make_ctx(&mut profiler);
        registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
    }
    assert_eq!(fired.load(Ordering::SeqCst), 1, "OneShot 발화");
    assert_eq!(
        after.load(Ordering::SeqCst),
        1,
        "Consumed 후 후속 stage 실행"
    );
    assert_eq!(registry.len(), 1, "OneShot GC 후 Persistent 1개 잔존");

    // 2회: OneShot GC 됐으므로 재발화 없음
    {
        let mut ctx = make_ctx(&mut profiler);
        registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
    }
    assert_eq!(fired.load(Ordering::SeqCst), 1, "GC 후 재발화 없음");
    assert_eq!(after.load(Ordering::SeqCst), 2, "Persistent 는 계속 실행");
}

/// Persistent stage 가 Consumed 를 반환하면 debug_assert panic (debug 빌드 한정).
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Persistent stage")]
fn test_inv_decode_stage_004_persistent_consumed_panics_in_debug() {
    let registry = PipelineRegistry::new();
    registry.submit(Arc::new(PersistentReturnsConsumed));

    let mut profiler = OpProfiler::new();
    let mut ctx = make_ctx(&mut profiler);
    registry.dispatch(LifecyclePhase::DecodeStart, &mut ctx);
}

/// Err 는 panic(fail-fast). stage.name() + phase 포함.
#[test]
#[should_panic(expected = "returned Err")]
fn test_inv_decode_stage_004_err_panics_fail_fast() {
    let registry = PipelineRegistry::new();
    registry.submit(Arc::new(AlwaysErr));

    let mut profiler = OpProfiler::new();
    let mut ctx = make_ctx(&mut profiler);
    registry.dispatch(LifecyclePhase::DecodeStart, &mut ctx);
}

/// StopReason 4-variant exhaustive match 컴파일 강제.
///
/// 새 variant 추가 시 match 아래 `StopFlag` 없는 구조를 깨뜨려 컴파일 에러로 조기 발견.
#[test]
fn test_inv_decode_stage_004_stopreason_no_stopflag_variant() {
    // 4-variant exhaustive match (StopFlag 없음 — §5.2.1 (다))
    let check = |r: StopReason| match r {
        StopReason::EosToken => "EosToken",
        StopReason::BudgetExhausted => "BudgetExhausted",
        StopReason::StopConditionMet => "StopConditionMet",
        StopReason::CommandRequested => "CommandRequested",
    };
    assert_eq!(check(StopReason::EosToken), "EosToken");
    assert_eq!(check(StopReason::BudgetExhausted), "BudgetExhausted");
}

/// eviction pos-환류 계약 (§5.2.1 (가)): stage 가 held-handle pos 를 감소시키고,
/// driver 가 dispatch 후 비교 + on_kv_prune-equivalent 1회 호출.
#[test]
fn test_inv_decode_stage_004_pos_reflux_via_held_handle() {
    let registry = PipelineRegistry::new();
    let handle = Arc::new(MockHeldHandle::new(100));

    registry.submit(Arc::new(EvictPosStage {
        handle: handle.clone(),
        new_pos: 70, // eviction 으로 30 감소
    }));

    let mut profiler = OpProfiler::new();

    let pos_before = handle.current_pos();
    assert_eq!(pos_before, 100);

    {
        let mut ctx = make_ctx(&mut profiler);
        registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
    }

    let pos_after = handle.current_pos();
    assert_eq!(pos_after, 70, "stage 가 held-handle pos 를 감소시킴");

    // driver 역할: pos 감소분 감지 → on_kv_prune equivalent 1회 호출
    let prune_count = Arc::new(AtomicUsize::new(0));
    if pos_after < pos_before {
        prune_count.fetch_add(1, Ordering::SeqCst);
        // 실제 코드: self.forward.on_kv_prune(pos_after)
    }
    assert_eq!(
        prune_count.load(Ordering::SeqCst),
        1,
        "on_kv_prune 1회 호출"
    );
}

// ── INV-DECODE-STAGE-005: submit 순서 + phase 공유 ───────────────────────────

/// submit 순서가 실행 순서와 일치한다.
#[test]
fn test_inv_decode_stage_005_submit_order_is_dispatch_order() {
    use std::sync::Mutex;

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
        name: "X",
        order: order.clone(),
    }));
    registry.submit(Arc::new(OrderStage {
        name: "Y",
        order: order.clone(),
    }));
    registry.submit(Arc::new(OrderStage {
        name: "Z",
        order: order.clone(),
    }));

    let mut profiler = OpProfiler::new();
    let mut ctx = make_ctx(&mut profiler);
    registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);

    let observed = order.lock().unwrap().clone();
    assert_eq!(observed, vec!["X", "Y", "Z"]);
}

/// Persistent 와 OneShot 둘 다 동일 phase(KvMutate)에서 발화한다 — phase-filter 등가.
///
/// 이는 §5.2.1 (나)의 "command-driven OneShot 도 KvMutate phase 에서 소비된다"
/// 계약의 registry-harness 수준 등가다.
#[test]
fn test_inv_decode_stage_005_eviction_phase_shared() {
    let registry = PipelineRegistry::new();
    let persistent_count = Arc::new(AtomicUsize::new(0));
    let oneshot_fired = Arc::new(AtomicUsize::new(0));

    // Persistent stage: KvMutate 마다 실행
    registry.submit(Arc::new(StopOnPhaseStage {
        target: LifecyclePhase::WeightMutate, // 다른 phase 에서만 Stop → KvMutate 에서 Continue
        reason: StopReason::EosToken,
        count: persistent_count.clone(),
    }));
    // OneShot stage: KvMutate 에서 Consumed
    registry.submit(Arc::new(OneShotOnPhase {
        target: LifecyclePhase::KvMutate,
        fired: oneshot_fired.clone(),
    }));

    let mut profiler = OpProfiler::new();
    {
        let mut ctx = make_ctx(&mut profiler);
        registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
    }

    assert_eq!(
        persistent_count.load(Ordering::SeqCst),
        1,
        "Persistent 발화"
    );
    assert_eq!(oneshot_fired.load(Ordering::SeqCst), 1, "OneShot 발화");
    assert_eq!(registry.len(), 1, "OneShot GC 후 Persistent 잔존");
}

// ── INV-DECODE-STAGE-006: StageContext 2-field ────────────────────────────────

/// StageContext 는 정확히 2 field (`step: StepInfo`, `profiler: &mut OpProfiler`) 다.
///
/// 구성 컴파일 강제: 두 필드가 제거되거나 추가되면 아래 코드가 컴파일 에러를 낸다.
#[test]
fn test_inv_decode_stage_006_context_two_fields_only() {
    let mut profiler = OpProfiler::new();
    // StageContext 를 직접 구성 — 필드가 바뀌면 여기서 컴파일 에러.
    let ctx = StageContext {
        step: StepInfo {
            pos: 42,
            decode_step: 7,
            pressure: Pressure::new(50),
            prev_token: 99,
        },
        profiler: &mut profiler,
    };
    // step 필드 읽기 검증
    assert_eq!(ctx.step.pos, 42);
    assert_eq!(ctx.step.decode_step, 7);
    assert_eq!(ctx.step.pressure.raw(), 50);
    assert_eq!(ctx.step.prev_token, 99);
}

// ── INV-DECODE-STAGE-007: OneShot GC 정확히 1회 ──────────────────────────────

/// 자기 phase 도달 전 Continue 면 GC 안 됨 + 도달 후 Consumed → 이후 부재.
#[test]
fn test_inv_decode_stage_007_oneshot_gc_once() {
    let registry = PipelineRegistry::new();
    let fired = Arc::new(AtomicUsize::new(0));

    registry.submit(Arc::new(OneShotOnPhase {
        target: LifecyclePhase::KvMutate,
        fired: fired.clone(),
    }));

    assert_eq!(registry.len(), 1);
    let mut profiler = OpProfiler::new();

    // 1회: 다른 phase → Continue → GC 안 됨
    {
        let mut ctx = make_ctx(&mut profiler);
        registry.dispatch(LifecyclePhase::DecodeStart, &mut ctx);
    }
    assert_eq!(
        fired.load(Ordering::SeqCst),
        0,
        "target phase 미도달 → 미발화"
    );
    assert_eq!(registry.len(), 1, "GC 안 됨");

    // 2회: target phase → Consumed → GC
    {
        let mut ctx = make_ctx(&mut profiler);
        registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
    }
    assert_eq!(fired.load(Ordering::SeqCst), 1, "target phase 도달 → 발화");
    assert_eq!(registry.len(), 0, "GC 완료");

    // 3회: GC 됐으므로 재발화 없음
    {
        let mut ctx = make_ctx(&mut profiler);
        registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
    }
    assert_eq!(fired.load(Ordering::SeqCst), 1, "GC 후 재발화 없음");
}
