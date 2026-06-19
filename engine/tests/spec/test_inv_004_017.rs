//! INV-004 / INV-017: QCF 수집 활성 상태에서 lossy action 실행 시 QcfMetric 생성 필수.
//!
//! 원본: 00-overview SYS-040 (INV-004), 01-architecture SYS-098 (INV-017, 재확인)
//! 검증 전략:
//!   - QCF 모듈의 순수 함수들이 QcfMetric을 생성하는지 확인.
//!   - QcfConfig.enabled = true 일 때 eviction/skip QCF 함수들이 유효한 QcfMetric 반환.
//!   - QcfConfig.enabled = false 일 때 생성하지 않는지 확인은
//!     호출자(generate.rs) 책임이므로 여기서는 config 필드 존재만 확인.
//!
//! 한계: 실제 lossy action (eviction/skip/quant) 실행 -> QcfMetric 생성 연결은
//!       full inference pipeline이 필요하여 단위 테스트로 완전 검증 불가.
//!       여기서는 QCF 함수가 올바른 QcfMetric을 반환하는지 확인하여
//!       "생성 가능성"을 보장한다.

use argus_engine::qcf::{QcfConfig, QcfMetric, QcfMode, SkipQcfTracker};

// ═══════════════════════════════════════════════════════════════
// INV-004/017: QcfConfig has an `enabled` field to control collection
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_004_qcf_config_enabled_field_exists() {
    let config = QcfConfig::default();
    assert!(
        config.enabled,
        "INV-004: QcfConfig default must have enabled=true"
    );
}

#[test]
fn test_inv_004_qcf_config_disabled() {
    let config = QcfConfig {
        enabled: false,
        ..QcfConfig::default()
    };
    assert!(
        !config.enabled,
        "QcfConfig.enabled can be set to false to disable collection"
    );
}

// ═══════════════════════════════════════════════════════════════
// INV-004/017: QcfMetric structure carries required fields
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_004_qcf_metric_structure() {
    let metric = QcfMetric {
        action: "h2o".to_string(),
        raw_value: 0.42,
        normalized_value: 0.35,
        per_head: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        tokens_affected: 100,
    };
    assert_eq!(metric.action, "h2o");
    assert!(metric.raw_value > 0.0);
    assert!(metric.normalized_value > 0.0);
    assert_eq!(metric.per_head.as_ref().unwrap().len(), 8);
    assert_eq!(metric.tokens_affected, 100);
}

// ═══════════════════════════════════════════════════════════════
// INV-004/017: SkipQcfTracker produces QcfMetric for layer skip action
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_004_skip_qcf_produces_metric_on_rejection() {
    let mut tracker = SkipQcfTracker::new(16);

    // Simulate lossy action: speculative decoding with partial rejection
    tracker.record(2, 5); // 3 out of 5 rejected -> 0.6 rejection rate

    let metric = tracker.current_proxy();
    assert_eq!(
        metric.action, "layer_skip",
        "INV-004: skip tracker must produce 'layer_skip' action metric"
    );
    assert!(
        metric.raw_value > 0.0,
        "INV-004: rejected tokens must produce non-zero raw_value, got {}",
        metric.raw_value
    );
    assert!(
        metric.tokens_affected > 0,
        "INV-004: lossy action must report affected tokens"
    );
}

#[test]
fn test_inv_004_skip_qcf_no_action_zero_metric() {
    let tracker = SkipQcfTracker::new(16);
    let metric = tracker.current_proxy();
    assert_eq!(metric.action, "layer_skip");
    assert!(
        metric.raw_value.abs() < 1e-6,
        "INV-004: no action should produce zero raw_value"
    );
}

#[test]
fn test_inv_017_qcf_metric_produced_after_lossy_record() {
    // INV-017 is a restatement of INV-004: verify the same behavior
    let mut tracker = SkipQcfTracker::new(10);
    tracker.record(0, 5); // 100% rejection = maximum lossy

    let metric = tracker.current_proxy();
    assert!(
        (metric.raw_value - 1.0).abs() < 1e-6,
        "INV-017: total rejection must produce raw_value=1.0"
    );
    assert_eq!(metric.tokens_affected, 5);
}

// ═══════════════════════════════════════════════════════════════
// INV-004/017: QcfMode variants for eviction
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_004_qcf_mode_attn() {
    let mode = QcfMode::Attn;
    assert!(mode.has_attn());
}

#[test]
fn test_inv_004_qcf_mode_both() {
    let mode = QcfMode::Both;
    assert!(mode.has_attn());
}
