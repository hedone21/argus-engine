//! INV-120 — Plan-Partition Stale Detection
//!
//! 대응 spec: spec/41-invariants.md §3.12 INV-120
//! 대응 arch: arch/plan_partition_integration.md A.6.2, A.11 R-PP2
//!
//! ## 불변식
//!
//! FullKernelPlan이 PartitionStep을 포함할 때, 각 PartitionStep::run 진입 시
//! PartitionPlanContext.ratio_generation_at_build와 PartitionContext.ratio_generation을
//! 비교한다. mismatch면 PlanInvalidated를 반환하며 caller는 plan을 재빌드하거나
//! forward_gen으로 fallback해야 한다.
//!
//! ## 검증 항목
//!
//! - [x] PartitionContext::ratio_generation 초기값 = 0, fetch_add 후 단조 증가
//! - [x] check_partition_generation — counter == at_build → Ok, counter > at_build → Err
//! - [x] caller가 PlanInvalidated 받아도 panic / UB 없이 match로 처리 가능
//! - [x] Multi-step ratio change (0 → 1 → 2 → 3)에서 generation 단조 증가, 매 단계 mismatch 감지
//!
//! ## 구현 메모
//!
//! OpenCL backend 실행 없이 Arc<AtomicU64> + check_partition_generation()만으로
//! 호스트에서 모든 경우를 검증 가능. R-PP2 시나리오: SetGpuBudget signal로 ratio가
//! 변경된 직후 plan execute가 호출되는 race를 시뮬레이션.

#[cfg(feature = "opencl")]
mod inv_120 {
    use argus_engine::backend::opencl::plan::{PlanInvalidated, check_partition_generation};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    // ── INV-120-A: ratio_generation 초기값 및 단조 증가 ──────────────────────────

    /// PartitionContext.ratio_generation은 생성 직후 0이어야 하고,
    /// set_ratio 시뮬레이션(fetch_add) 후 단조 증가해야 한다.
    #[test]
    fn inv_120_ratio_generation_monotonic() {
        // 초기값: PartitionContext는 ratio_generation: Arc::new(AtomicU64::new(0))로 초기화
        let counter = Arc::new(AtomicU64::new(0));
        assert_eq!(
            counter.load(Ordering::Acquire),
            0,
            "ratio_generation 초기값은 0이어야 한다"
        );

        // set_ratio 시뮬레이션 1 (prepare_tensor_partition 재호출)
        counter.fetch_add(1, Ordering::Release);
        assert_eq!(
            counter.load(Ordering::Acquire),
            1,
            "첫 번째 ratio 변경 후 generation = 1"
        );

        // set_ratio 시뮬레이션 2
        counter.fetch_add(1, Ordering::Release);
        assert_eq!(
            counter.load(Ordering::Acquire),
            2,
            "두 번째 ratio 변경 후 generation = 2"
        );

        // 단조 증가 검증 — 역방향 변화 없음
        let mut prev = 0u64;
        for _ in 0..5 {
            let cur = counter.load(Ordering::Acquire);
            assert!(
                cur >= prev,
                "ratio_generation은 단조 증가해야 한다: prev={prev} cur={cur}"
            );
            prev = cur;
            counter.fetch_add(1, Ordering::Release);
        }
    }

    // ── INV-120-B: mismatch → PlanInvalidated ────────────────────────────────────

    /// check_partition_generation은 counter == at_build 이면 Ok를, 불일치하면
    /// Err(PlanInvalidated)를 반환해야 한다.
    #[test]
    fn inv_120_partition_step_returns_plan_invalidated_on_mismatch() {
        let counter = Arc::new(AtomicU64::new(0));
        let at_build: u64 = 0;

        // 일치 → Ok
        assert_eq!(
            check_partition_generation(at_build, &counter),
            Ok(()),
            "generation 일치 시 Ok() 반환"
        );

        // counter 증가 (ratio 변경 시뮬레이션)
        counter.fetch_add(1, Ordering::Release);

        // 불일치 → PlanInvalidated
        assert_eq!(
            check_partition_generation(at_build, &counter),
            Err(PlanInvalidated),
            "generation 불일치 시 Err(PlanInvalidated) 반환"
        );
    }

    // ── INV-120-C: caller 안전 처리 ───────────────────────────────────────────────

    /// PlanInvalidated는 Debug + Clone + PartialEq 파생이므로 match/assert_eq에서
    /// panic 없이 처리 가능해야 한다. 재시도 루프 시뮬레이션도 포함.
    #[test]
    fn inv_120_caller_handles_plan_invalidated_safely() {
        // PlanInvalidated가 Debug + Clone + PartialEq 파생 검증
        let err = PlanInvalidated;
        let cloned = err.clone();
        assert_eq!(err, cloned, "PlanInvalidated는 Clone + PartialEq 지원");
        let _ = format!("{err:?}"); // Debug 파생 — panic 없음

        // 재시도 루프 시뮬레이션
        let counter = Arc::new(AtomicU64::new(1)); // 이미 bump된 상태
        let at_build: u64 = 0;
        let handled = match check_partition_generation(at_build, &counter) {
            Ok(()) => {
                // 일치 — 정상 실행 경로
                false
            }
            Err(PlanInvalidated) => {
                // PlanInvalidated 수신 → 재빌드 또는 forward_gen fallback
                true
            }
        };
        assert!(
            handled,
            "PlanInvalidated를 받아 fallback 처리가 실행되어야 한다"
        );

        // Multi-step ratio change: 0 → 1 → 2 → 3, 모든 단계에서 mismatch 감지
        let counter2 = Arc::new(AtomicU64::new(0));
        let at_build2: u64 = 0;
        for bump in 1u64..=3 {
            counter2.fetch_add(1, Ordering::Release);
            assert_eq!(
                counter2.load(Ordering::Acquire),
                bump,
                "generation 단조 증가"
            );
            // at_build=0 기준으로 항상 mismatch
            assert_eq!(
                check_partition_generation(at_build2, &counter2),
                Err(PlanInvalidated),
                "bump={bump}: at_build=0 vs live={bump} → PlanInvalidated"
            );
        }
    }
}
