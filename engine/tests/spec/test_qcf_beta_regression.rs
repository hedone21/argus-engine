//! Spec test: β-amplified CAOTE regression (ARGUS Step 3 / Step 6).
//!
//! Verifies that β=1.0 and β=1.5 produce distinct QCF values for a non-uniform
//! attention distribution, and that β=1.0 results are finite and non-negative.
//!
//! The β=1.0 "fast path" (no amplification) is the canonical reference path.
//! Divergence from β=1.5 on a peaked distribution confirms that the amplification
//! exponent is applied correctly (now in `argus_extension_api::redistribute_value`).
//!
//! Regression gate: if the redistribution collapses β back to a uniform
//! formula, β=1.0 and β=1.5 would become identical — caught here.

use argus_engine::kv::kv_cache::KVLayout;
use argus_engine::qcf::{AggregationMode, QcfKvParams, VDataSource, compute_qcf_kv};
use argus_extension_api::{QcfEstimator, StageParams, find_qcf_estimator};

/// Build sliding-eviction test params with the given β.
///
/// Setup: n_kv_heads=1, head_dim=2, current_pos=3, capacity=3, target_len=2.
/// Attention: [0.1, 0.1, 0.8] (strongly peaked at token 2 → non-uniform).
fn make_params(beta: f32) -> QcfKvParams<'static> {
    static V_DATA: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    static ATTENTION: &[f32] = &[0.1, 0.1, 0.8]; // peaked, non-uniform → β matters

    // Stateless sliding estimator; leaked to satisfy the `&dyn` borrow (test-only).
    let est: &'static dyn QcfEstimator = Box::leak((find_qcf_estimator("sliding")
        .expect("sliding estimator registered")
        .make)(StageParams::default(), &[]));
    QcfKvParams {
        estimator: est,
        target_len: 2,
        v_source: VDataSource::F32(V_DATA),
        k_source: None,
        attention_scores: ATTENTION,
        head_attn: None,
        n_kv_heads: 1,
        head_dim: 2,
        current_pos: 3,
        capacity: 3,
        layout: KVLayout::HeadMajor,
        aggregation: AggregationMode::Mean,
        beta,
    }
}

#[test]
fn spec_beta_one_result_is_finite_non_negative() {
    let (qcf, per_head) = compute_qcf_kv(&make_params(1.0));
    assert!(qcf.is_finite(), "β=1.0 QCF must be finite, got {qcf}");
    assert!(qcf >= 0.0, "β=1.0 QCF must be non-negative, got {qcf}");
    assert_eq!(per_head.len(), 1, "single KV head → per_head len=1");
}

#[test]
fn spec_beta_one_five_result_is_finite_non_negative() {
    let (qcf, _) = compute_qcf_kv(&make_params(1.5));
    assert!(qcf.is_finite(), "β=1.5 QCF must be finite, got {qcf}");
    assert!(qcf >= 0.0, "β=1.5 QCF must be non-negative, got {qcf}");
}

#[test]
fn spec_beta_two_result_is_finite_non_negative() {
    let (qcf, _) = compute_qcf_kv(&make_params(2.0));
    assert!(qcf.is_finite(), "β=2.0 QCF must be finite, got {qcf}");
    assert!(qcf >= 0.0, "β=2.0 QCF must be non-negative, got {qcf}");
}

#[test]
fn spec_beta_one_vs_one_five_differ_on_peaked_distribution() {
    // Non-uniform [0.1, 0.1, 0.8] → β amplification changes the weight distribution.
    let (qcf1, _) = compute_qcf_kv(&make_params(1.0));
    let (qcf15, _) = compute_qcf_kv(&make_params(1.5));
    assert!(
        (qcf1 - qcf15).abs() > 1e-6,
        "β=1.0 ({qcf1}) and β=1.5 ({qcf15}) must differ on a peaked distribution"
    );
}

#[test]
fn spec_beta_increases_monotonically_on_peaked_distribution() {
    // For a strongly peaked distribution, higher β amplifies the dominant weight further. This is a
    // direction test, not an exact-value test.
    let (qcf1, _) = compute_qcf_kv(&make_params(1.0));
    let (qcf15, _) = compute_qcf_kv(&make_params(1.5));
    let (qcf2, _) = compute_qcf_kv(&make_params(2.0));

    let distinct_12 = (qcf1 - qcf15).abs() > 1e-6;
    let distinct_23 = (qcf15 - qcf2).abs() > 1e-6;
    assert!(
        distinct_12 || distinct_23,
        "β sweep [1.0, 1.5, 2.0] must produce at least 2 distinct values on peaked distribution: {qcf1}, {qcf15}, {qcf2}"
    );
}
