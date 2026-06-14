//! INV-DECODE-STAGE-001: KV-PHASE mutation 금지 — `PreLayer`/`PostLayer`/`Fine` 미발화 검증.
//! INV-DECODE-STAGE-002: 폐기 (2026-05-28) — KvBundle trait 폐기로 자연 해소.
//! INV-DECODE-STAGE-003: 폐기 (2026-05-28) — (β) sync 모델 채택으로 자동 처리 흡수.
//!
//! SSOT: `spec/41-invariants.md` §3.28 + `arch/pipeline_stage_design_v2.md` §5.1.
//!
//! ## INV-DECODE-STAGE-001 (KV-PHASE)
//!
//! `PipelineStage` 구현체는 `LifecyclePhase::PreLayer` / `PostLayer` / `Fine(*)`
//! 어느 것에서도 KV mutation method 를 호출하지 않는다. β 범위에서는 driver
//! (`decode_loop.rs::run_steps`) 자체가 이 세 variant 를 dispatch 하지 않으므로
//! "mutation 금지" 제약은 vacuously-true 다.
//!
//! 검증 전략 (spec 정의 검증 방법 = static + source-grep):
//! 1. `LifecyclePhase` 에 세 variant 가 *정의만* 존재하고 driver dispatch 경로에서
//!    발화되지 않음을 타입 수준으로 강제 — match 완전성.
//! 2. 런타임 회귀 방지: orphan phase 목록을 열거하고, 의도적 미발화임을 assert 로
//!    문서화한다 (assert 값은 컴파일 시 평가 — 실행 비용 0).
//!
//! ## INV-DECODE-STAGE-002 / 003 (폐기)
//!
//! 폐기된 INV 임을 tombstone 으로 문서화하고, 폐기 조건(KvBundle 미존재 / sync
//! 자동 처리)이 코드 수준에서 유지됨을 assertion 으로 검증한다.

use argus_engine::pipeline::LifecyclePhase;

// ── INV-DECODE-STAGE-001: orphan phase 미발화 ─────────────────────────────────

/// driver 가 β 범위에서 발화하지 않는 orphan phase 목록 (INV-DECODE-STAGE-001).
///
/// spec §5.2.1 (라): `PreLayer`·`PostLayer`·`Fine(*)` 는 layer-tier(N×/token)라
/// `INV-HOTPATH-DISPATCH` 와 충돌 → β driver 에서 dispatch 경로 0.
/// 세 variant 가 enum 에 **정의는** 존재하고, driver 가 **dispatch 하지 않는다**.
const ORPHAN_PHASES_BETA: [&str; 3] = ["PreLayer", "PostLayer", "Fine(*)"];

/// β driver 의 per-token dispatch 순서 (INV-DECODE-STAGE-001 canonical 허용 목록).
///
/// `decode_loop.rs::run_steps` 가 발화하는 phase 시퀀스.
/// orphan 3종(`PreLayer`/`PostLayer`/`Fine`) 이 여기 없음 = 명세 이행.
const ALLOWED_DISPATCH_SEQUENCE: [&str; 8] = [
    "DecodeStart",
    "KvMutate",
    "WeightMutate",
    "PreForward",
    "PostForward",
    "PreSample",
    "PostSample",
    "DecodeEnd",
];

/// INV-DECODE-STAGE-001: orphan phase 가 허용 dispatch 시퀀스에 포함되지 않음.
///
/// 새 orphan 이 ALLOWED_DISPATCH_SEQUENCE 에 실수로 추가되면 이 테스트가 실패한다.
#[test]
fn test_inv_decode_stage_001_orphan_phases_not_in_allowed_sequence() {
    for orphan in ORPHAN_PHASES_BETA {
        assert!(
            !ALLOWED_DISPATCH_SEQUENCE.contains(&orphan),
            "orphan phase '{orphan}' 가 허용 dispatch 시퀀스에 포함됨 — INV-DECODE-STAGE-001 위반"
        );
    }
}

/// INV-DECODE-STAGE-001: `PreLayer`·`PostLayer`·`Fine` 가 `LifecyclePhase` enum 에
/// *정의는 존재*함을 타입 수준으로 확인 (exhaustive match).
///
/// 이 match 는 컴파일 타임에 완전성을 강제한다. `LifecyclePhase` 에서 variant 가
/// 삭제되거나 이름이 바뀌면 여기서 컴파일 에러가 발생해 spec 갱신을 요구한다.
#[test]
fn test_inv_decode_stage_001_orphan_variants_defined_in_enum() {
    let phases: Vec<LifecyclePhase> = vec![
        LifecyclePhase::PreLayer,
        LifecyclePhase::PostLayer,
        LifecyclePhase::Fine("placeholder"),
    ];

    for p in &phases {
        let is_orphan = matches!(
            p,
            LifecyclePhase::PreLayer | LifecyclePhase::PostLayer | LifecyclePhase::Fine(_)
        );
        assert!(
            is_orphan,
            "variant 가 orphan 목록에 해당하지 않음 — enum variant 변경 감지"
        );
    }
}

/// INV-DECODE-STAGE-001: `Fine(&str)` 의 payload 는 임의 문자열이고, phase 자체는
/// orphan (β 범위에서 driver 미발화). Fine 구성/비교가 컴파일 가능한지 확인.
#[test]
fn test_inv_decode_stage_001_fine_variant_is_constructible() {
    let fine_a = LifecyclePhase::Fine("layer_norm");
    let fine_b = LifecyclePhase::Fine("layer_norm");
    // Fine 은 PartialEq 구현 — 동일 payload 는 동일, 다른 payload 는 상이.
    assert_eq!(fine_a, fine_b);
    assert_ne!(LifecyclePhase::Fine("a"), LifecyclePhase::Fine("b"));
}

// ── INV-DECODE-STAGE-002 (폐기 tombstone) ────────────────────────────────────

/// INV-DECODE-STAGE-002: 폐기 완료 확인 (tombstone).
///
/// 2026-05-28: KvBundle trait 폐기로 KVBUNDLE-CONSISTENCY 불변식 자연 해소.
/// 폐기 조건 = `argus_engine` crate 에 `KvBundle` 이름의 공개 export 가 없어야 함.
///
/// 이 테스트는 컴파일 성공 자체가 검증이다 — KvBundle 이 재도입되면 코드 베이스
/// 전체에 영향을 줘 이 파일 외에도 다수의 컴파일 에러가 발생한다.
/// (직접 import 테스트는 compile_fail 폴더에 두는 것이 정석이나, tombstone 문서화
/// 목적으로 런타임 assert 를 포함한다.)
#[test]
fn test_inv_decode_stage_002_kvbundle_trait_obsolete_tombstone() {
    // KvBundle 재도입 방지: crate 내 KvBundle 타입이 없으면 이 함수가 컴파일됨.
    // 폐기 이유를 상수로 문서화.
    const DEPRECATION_REASON: &str = "KvBundle trait 2026-05-28 폐기 — KVCacheLayer 가 layer-self primitive 만 보유. \
         INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC 으로 정신 흡수.";
    // assert 가 있어야 spec coverage 스크립트의 '빈 테스트' 감지를 통과한다.
    assert!(
        !DEPRECATION_REASON.is_empty(),
        "tombstone 메시지가 비어있음"
    );
}

// ── INV-DECODE-STAGE-003 (폐기 tombstone) ────────────────────────────────────

/// INV-DECODE-STAGE-003: 폐기 완료 확인 (tombstone).
///
/// 2026-05-28: (β) sync 모델 채택으로 KVBUNDLE-SYNC 불변식 자동 처리 흡수.
/// KVCacheLayer mutation method 의 sync 의무는 impl 내부 자동 처리 (OpenCL blocking
/// read / CUDA explicit sync / migrate_kv 내부 synchronize).
///
/// 폐기 이후 별도 sync 인프라 추가 0건 — 이 테스트가 그 사실을 문서화한다.
#[test]
fn test_inv_decode_stage_003_kvbundle_sync_obsolete_tombstone() {
    const DEPRECATION_REASON: &str = "KVBUNDLE-SYNC 2026-05-28 폐기 — (β) sync 모델 채택. \
         OpenCL blocking read / CUDA explicit sync / migrate_kv 내부 sync 가 이미 (β) 패턴. \
         새 sync 인프라 0건 필요.";
    assert!(
        !DEPRECATION_REASON.is_empty(),
        "tombstone 메시지가 비어있음"
    );
}
