//! ENG-ALG-200 — GPU Plan × Tensor Partition 협업 통합 테스트
//!
//! 대응 spec: spec/32-engine-algorithms.md §3.11.1 [ENG-ALG-200]
//!
//! 대응 arch: arch/plan_partition_integration.md (전체)
//!
//! ## 시나리오
//!
//! Galaxy S25 / Qwen 2.5-1.5B Q4_0 / partition r=0.7에서:
//!
//! 1. plan 활성 (`LLMRS_PARTITION_PLAN=1`, default)
//! 2. forward_gen fallback (`LLMRS_PARTITION_PLAN=0`)
//!
//! 두 경로의 decode 결과가 토큰 단위로 동일해야 한다 (greedy, 100 tokens).
//!
//! ## 검증 항목 (호스트 실행 가능)
//!
//! - [x] FfnVariant enum에 GpuOnly + Partitioned variant가 존재 (컴파일 검증)
//! - [x] PartitionMerge enum에 Inline + Deferred variant가 존재 (컴파일 검증)
//! - [x] partition_plan_enabled() 기본값 = true (env var 미설정 시)
//! - [x] partition_fused_merge_enabled() 기본값 = false (env var 미설정 시)
//!
//! ## 검증 항목 (device test 필요)
//!
//! - [ ] build_partitioned_layer_plan이 FfnVariant::Partitioned를 반환
//! - [ ] PartitionStep이 expected sub-step 시퀀스 보유 (gpu_gate/up/act_mul/down)
//! - [ ] PartitionMerge::Inline (default) — plan path 정상 동작
//! - [ ] r=0.0 비-partition path 회귀 < 1 ms/tok
//! - [ ] plan path와 forward_gen path bit-exact (greedy, 100 tokens)

// ── 호스트 실행 가능: 타입 구조 + 기본값 검증 ──────────────────────────────

#[cfg(feature = "opencl")]
mod eng_alg_200_host {
    use argus_engine::backend::opencl::plan::{FfnVariant, PartitionMerge};
    use argus_engine::layers::tensor_partition::{
        partition_fused_merge_enabled, partition_plan_enabled,
    };

    /// FfnVariant와 PartitionMerge의 variant 구조를 컴파일 레벨에서 검증한다.
    ///
    /// 실제 variant construct는 OpenCL 타입(CoreKernel 등)이 필요하므로 여기서는
    /// `matches!` + 타입 존재성 검증으로 대체한다.
    #[test]
    fn eng_alg_200_ffn_variant_structure_compiles() {
        // FfnVariant::GpuOnly와 FfnVariant::Partitioned variant가 타입 시스템에 존재하는지
        // 컴파일 시간 검증 — 각 arm에서 해당 타입명을 매칭해 컴파일러가 exhaustive하게 확인.
        //
        // `_ffn`은 실제 construct 불가이므로 타입 레벨 검증만 수행한다.
        // 아래 closure는 인자를 받아 variant를 매칭하는 코드가 컴파일 가능함을 확인한다.
        fn _assert_ffn_variant_exhaustive(ffn: &FfnVariant) {
            match ffn {
                FfnVariant::GpuOnly { .. } => {}
                FfnVariant::Partitioned(_) => {}
            }
        }

        fn _assert_merge_variant_exhaustive(merge: &PartitionMerge) {
            match merge {
                PartitionMerge::Fused { .. } => {}
                PartitionMerge::Inline { .. } => {}
                PartitionMerge::Deferred => {}
            }
        }

        // 컴파일이 성공하면 테스트 통과.
    }

    /// partition_plan_enabled()의 기본값 검증.
    ///
    /// 환경 변수 LLMRS_PARTITION_PLAN이 설정되지 않은 경우 true가 기본값
    /// (2026-04-21 late: lm_head dtype 불일치로 인한 plan garbage 버그가
    /// F32 tied-embedding lm_head에 대한 matmul_transposed fallback으로
    /// 수정되어 partition plan 경로를 기본으로 복원). OnceLock으로
    /// 캐시되므로 이 테스트는 first-reader가 되어야 신뢰성 있음. CI
    /// 환경에서는 env var를 설정하지 않고 실행하는 것이 전제.
    #[test]
    fn eng_alg_200_partition_plan_default_enabled() {
        // env var가 설정된 경우 캐시는 해당 값에 따라 결정됨 — 기본값
        // 단언을 건너뛴다.
        if std::env::var("LLMRS_PARTITION_PLAN").is_ok() {
            return;
        }
        // OnceLock이 아직 초기화되지 않았다면 기본값은 true.
        assert!(
            partition_plan_enabled(),
            "LLMRS_PARTITION_PLAN 미설정 시 partition_plan_enabled() = true (기본값, \
             2026-04-21 lm_head dtype fix 이후)"
        );
    }

    /// partition_fused_merge_enabled()의 기본값 검증.
    ///
    /// LLMRS_PARTITION_FUSED_MERGE가 설정되지 않은 경우 false가 기본값.
    #[test]
    fn eng_alg_200_fused_merge_default_disabled() {
        if std::env::var("LLMRS_PARTITION_FUSED_MERGE").is_ok() {
            return;
        }
        assert!(
            !partition_fused_merge_enabled(),
            "LLMRS_PARTITION_FUSED_MERGE 미설정 시 partition_fused_merge_enabled() = false (기본값)"
        );
    }
}

// ── Device test 필요 (호스트에서 실행 불가) ─────────────────────────────────

/// build_partitioned_layer_plan이 FfnVariant::Partitioned를 반환하고
/// PartitionStep의 sub-step 시퀀스(gpu_gate, gpu_up, gpu_act_mul, gpu_down)가
/// 올바른지 검증한다.
///
/// 이 테스트는 OpenCL + 실제 모델 로딩이 필요하므로 Galaxy S25 디바이스에서만 의미 있다.
/// 호스트 실행 시에는 NVIDIA fallback이 garbage 결과를 낼 수 있어 #[ignore]로 유지.
/// device test: `deploy-test` 스킬 참조.
#[test]
#[ignore = "device test required: Galaxy S25 + Qwen-2.5-1.5B Q4_0 (see deploy-test skill)"]
fn eng_alg_200_partition_plan_bit_exact_vs_forward_gen() {
    // 1. plan path로 decode 100 tokens (LLMRS_PARTITION_PLAN=1)
    // 2. LLMRS_PARTITION_PLAN=0으로 forward_gen path decode 100 tokens
    // 3. token id diff = 0 검증
    // 호스트에서는 OpenCL NVIDIA fallback garbage 결과 문제로 실행 불가.
}

/// build_partitioned_layer_plan의 FfnVariant::Partitioned sub-step 시퀀스 구조 검증.
///
/// LayerPlanConfig + PartitionContext 구성에 OpenCL backend (cl_mem, CoreKernel)가
/// 필요하므로 호스트에서는 실행 불가. device test 환경에서만 의미 있음.
#[test]
#[ignore = "device test required: OpenCL backend + real weights needed to build PartitionStep"]
fn eng_alg_200_partitioned_ffn_step_sequence() {
    // 1. mock LayerPlanConfig + PartitionContext (r=0.7)
    // 2. build_partitioned_layer_plan 호출
    // 3. FfnVariant::Partitioned 분기 + PartitionStep의 gpu_gate, gpu_up,
    //    gpu_act_mul, gpu_down 4개 + merge sub-step 검사
    // OpenCL init이 호스트에서 비현실적임 (NVIDIA fallback garbage).
}

/// PartitionMerge::Inline vs Deferred 두 경로의 bit-exact 동치성 검증.
///
/// LLMRS_PARTITION_FUSED_MERGE env를 set/unset 후 build_partitioned_layer_plan의
/// merge variant 변화를 관찰한다. env var 조작은 OnceLock 캐싱으로 인해 테스트
/// 병렬 실행 안전성을 보장할 수 없어 #[ignore]로 유지.
#[test]
#[ignore = "device test required: env serialization + real model needed for bit-exact comparison"]
fn eng_alg_200_partition_merge_inline_vs_deferred() {
    // LLMRS_PARTITION_FUSED_MERGE=0 → PartitionMerge::Inline
    // LLMRS_PARTITION_FUSED_MERGE=1 → PartitionMerge::Deferred
    // 두 경로 bit-exact 비교 (device test 환경 필요, OnceLock으로 인한 직렬화 필요)
}
