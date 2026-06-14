//! ENG-ALG-219 — Global ratio_generation plan invalidation (Phase 3.5)
//!
//! 대응 spec: spec/32-engine-algorithms.md §3.12.13 (ENG-ALG-219)
//! 대응 inv: spec/41-invariants.md §3.14 (INV-129)
//! 대응 arch: arch/weight_swap.md v5 §2.2.2, arch/plan_partition_integration.md A.11
//!
//! ## 불변식
//!
//! `FullKernelPlan`은 빌드 시점의 `TransformerModel::ratio_generation` 값을
//! `ratio_generation_at_build`에 캡처한다. `execute()` 진입 시 1회 Acquire load
//! 비교 후 mismatch이면 `PlanInvalidated`를 반환한다.
//!
//! INV-129는 INV-120(PartitionStep 내부 체크)과 독립적으로 동작한다.
//! partition_ctx=None인 경우에도 global gen 체크는 수행된다.
//!
//! ## 검증 항목
//!
//! - [x] plan_uses_initial_generation: 일치 시 Ok (PlanInvalidated 미반환)
//! - [x] plan_invalidated_after_global_bump: fetch_add 후 execute → PlanInvalidated
//! - [x] plan_invalidated_independent_of_partition: partition_ctx=None에서도 global gen mismatch → PlanInvalidated
//! - [x] plan_or_combined_with_partition: global gen만 bump해도 stale 탐지
//! - [x] plan_rebuilt_clears_invalidation: 새 at_build 캡처 후 즉시 valid
//!
//! ## 구현 메모
//!
//! `FullKernelPlan::execute()`는 OpenCL 백엔드 실행이 필요하므로 호스트에서
//! 단위 테스트하려면 진입부 비교 로직을 `check_global_generation` 자유 함수로
//! 분리해야 한다. 이 파일의 테스트는 그 자유 함수를 직접 호출하여 GPU 없이
//! 모든 경우를 검증한다.

#[cfg(feature = "opencl")]
mod eng_alg_219 {
    use argus_engine::backend::opencl::plan::{PlanInvalidated, check_global_generation};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    // ── ENG-ALG-219-A: 일치 시 Ok ──────────────────────────────────────────────

    /// 빌드 시점과 execute 시점의 global generation이 일치하면 PlanInvalidated를
    /// 반환하지 않아야 한다 (정상 실행 경로).
    #[test]
    fn plan_uses_initial_generation() {
        let counter = Arc::new(AtomicU64::new(0));
        let at_build: u64 = counter.load(Ordering::Acquire);

        assert_eq!(
            check_global_generation(at_build, &counter),
            Ok(()),
            "빌드 시점 generation 일치 시 Ok 반환 — PlanInvalidated 미발생"
        );
    }

    // ── ENG-ALG-219-B: bump 후 PlanInvalidated ─────────────────────────────────

    /// 빌드 후 `model.ratio_generation.fetch_add(1, Release)` → execute →
    /// `PlanInvalidated` 반환 (weight swap 발생 탐지).
    #[test]
    fn plan_invalidated_after_global_bump() {
        let counter = Arc::new(AtomicU64::new(0));
        let at_build: u64 = counter.load(Ordering::Acquire); // = 0

        // Weight swap 시뮬레이션: SwapExecutor가 배치 완료 후 fetch_add(1)
        counter.fetch_add(1, Ordering::Release);

        assert_eq!(
            check_global_generation(at_build, &counter),
            Err(PlanInvalidated),
            "global gen bump 후 execute → PlanInvalidated 반환"
        );
    }

    // ── ENG-ALG-219-C: partition=None에서도 독립 동작 ─────────────────────────

    /// INV-129: partition_ctx=None인 경우에도 global gen mismatch 시
    /// PlanInvalidated가 반환된다. INV-120(partition 내부 체크)과 독립 동작.
    ///
    /// 이 테스트는 `check_global_generation`이 partition context 유무와
    /// 무관하게 순수히 counter 비교만 수행함을 검증한다.
    #[test]
    fn plan_invalidated_independent_of_partition() {
        let global_counter = Arc::new(AtomicU64::new(5));
        let at_build: u64 = 5;

        // 일치 → Ok (partition=None 상황 시뮬레이션)
        assert_eq!(
            check_global_generation(at_build, &global_counter),
            Ok(()),
            "generation 일치 시 Ok — partition_ctx=None 무관"
        );

        // global gen만 bump
        global_counter.fetch_add(1, Ordering::Release); // now 6

        // partition_ctx=None임에도 global gen mismatch → PlanInvalidated
        assert_eq!(
            check_global_generation(at_build, &global_counter),
            Err(PlanInvalidated),
            "INV-129: partition=None에서도 global gen mismatch → PlanInvalidated"
        );
    }

    // ── ENG-ALG-219-D: global gen bump만으로 stale 탐지 ──────────────────────

    /// global gen만 bump해도 stale 탐지 (OR 결합):
    /// - partition gen은 그대로, global gen만 변경 → PlanInvalidated
    /// - global gen이 at_build와 일치하면 → Ok
    ///
    /// 이는 ENG-ALG-219의 핵심: global gen check가 partition gen check와
    /// 별도의 독립 gate임을 확인한다.
    #[test]
    fn plan_or_combined_with_partition() {
        // Case 1: global gen만 bump해도 stale 탐지
        let global_counter = Arc::new(AtomicU64::new(0));
        let at_build_global: u64 = 0;

        // partition gen 변경 없음, global gen만 bump
        global_counter.fetch_add(1, Ordering::Release);
        assert_eq!(
            check_global_generation(at_build_global, &global_counter),
            Err(PlanInvalidated),
            "global gen만 bump해도 stale 탐지"
        );

        // Case 2: global gen이 at_build와 일치하면 Ok
        let global_counter2 = Arc::new(AtomicU64::new(3));
        let at_build2: u64 = 3;
        assert_eq!(
            check_global_generation(at_build2, &global_counter2),
            Ok(()),
            "global gen 일치 → Ok"
        );

        // Case 3: 여러 번 bump 후에도 일관성 있게 stale 감지
        let global_counter3 = Arc::new(AtomicU64::new(0));
        let at_build3: u64 = 0;
        for bump in 1u64..=5 {
            global_counter3.fetch_add(1, Ordering::Release);
            assert_eq!(
                global_counter3.load(Ordering::Acquire),
                bump,
                "bump={bump}: generation 단조 증가"
            );
            assert_eq!(
                check_global_generation(at_build3, &global_counter3),
                Err(PlanInvalidated),
                "bump={bump}: at_build=0 vs live={bump} → PlanInvalidated"
            );
        }
    }

    // ── ENG-ALG-219-E: rebuild 후 invalidation 해소 ─────────────────────────

    /// rebuild 후 새 generation 캡처 → 즉시 valid.
    /// lazy rebuild 패턴: PlanInvalidated 수신 → build_plan 재호출 →
    /// 새 at_build로 다음 execute는 Ok.
    #[test]
    fn plan_rebuilt_clears_invalidation() {
        let counter = Arc::new(AtomicU64::new(0));

        // 초기 빌드: at_build=0
        let at_build_old: u64 = counter.load(Ordering::Acquire); // 0

        // Weight swap 발생
        counter.fetch_add(1, Ordering::Release); // counter=1

        // 구 plan은 stale
        assert_eq!(
            check_global_generation(at_build_old, &counter),
            Err(PlanInvalidated),
            "구 plan stale 확인"
        );

        // rebuild: 새 at_build 캡처 (= 현재 counter = 1)
        let at_build_new: u64 = counter.load(Ordering::Acquire); // 1

        // 새 plan은 즉시 valid
        assert_eq!(
            check_global_generation(at_build_new, &counter),
            Ok(()),
            "rebuild 후 새 generation 캡처 → 즉시 valid"
        );

        // 추가 swap 없이 valid 상태 유지
        assert_eq!(
            check_global_generation(at_build_new, &counter),
            Ok(()),
            "추가 swap 없이 valid 상태 유지"
        );

        // 다시 swap 발생 → 다시 stale
        counter.fetch_add(1, Ordering::Release); // counter=2
        assert_eq!(
            check_global_generation(at_build_new, &counter),
            Err(PlanInvalidated),
            "2번째 swap 후 다시 stale"
        );
    }
}
