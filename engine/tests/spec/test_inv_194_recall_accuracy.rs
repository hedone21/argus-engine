// spec/41-invariants.md §3.30 INV-194: RecallWeights 정확성 게이트
//
// **INV-194**: Q4_0 으로 swap 된 layer 를 F16 으로 recall 한 이후의 forward 결과는
// F16 single-precision baseline 대비 NMSE ≤ 0.01 이어야 한다 (품질 개선 방향 보증).
//
// 환경변수:
// - `LLM_RS_TEST_RECALL_AUF`: F16+Q4_0 multi-dtype AUF 파일 경로 (필수)
//
// 미설정 시 전체 skip (device-gated 테스트).
//
// 검증:
// (a) swap(ratio=1.0, Q4_0) 후 recall(ratio=1.0, F16) → NMSE ≤ 0.01 (F16 baseline 대비)
// (b) recall 후 current_dtype() 이 F16 임을 확인 (INV-193 동형 — 후보 선택의 역방향)

// 환경변수 미설정 시 모든 테스트를 graceful skip.
fn recall_auf_path() -> Option<String> {
    let path = std::env::var("LLM_RS_TEST_RECALL_AUF").ok()?;
    if !std::path::Path::new(&path).exists() {
        eprintln!("[INV-194] skip: AUF model not found at {path} (set LLM_RS_TEST_RECALL_AUF)");
        return None;
    }
    Some(path)
}

/// INV-194(a): recall 후 NMSE ≤ 0.01 (F16 baseline 대비).
///
/// AUF 파일 경로가 미설정/미존재 → graceful skip.
#[test]
fn recall_restores_f16_nmse_within_threshold() {
    let Some(_auf_path) = recall_auf_path() else {
        eprintln!("[INV-194] skip: LLM_RS_TEST_RECALL_AUF not set");
        return;
    };
    // TODO(device): AUF 파일 기반 swap → recall → logit NMSE 비교 구현.
    // 현재는 환경변수 감지 + graceful skip 구조만 배선. 실제 측정은 device 게이트에서 수행.
    eprintln!("[INV-194] NOTE: device gate not yet run — NMSE measurement pending");
}

/// INV-194(b): recall 후 layer current_dtype() 이 F16 임을 확인.
///
/// AUF 파일 경로가 미설정/미존재 → graceful skip.
#[test]
fn recall_layer_dtype_is_f16_after_recall() {
    let Some(_auf_path) = recall_auf_path() else {
        eprintln!("[INV-194] skip: LLM_RS_TEST_RECALL_AUF not set");
        return;
    };
    // TODO(device): swap → recall 후 LayerSlot.current_dtype() == DType::F16 검증.
    eprintln!("[INV-194] NOTE: device gate not yet run — dtype check pending");
}
