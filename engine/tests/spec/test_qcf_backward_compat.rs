//! Spec test: QCF experimental backward compatibility (ARGUS Step 6).
//!
//! Verifies that the β=1.0 fast path produces sane (finite, non-negative) results
//! for all supported KV cache techniques. Acts as a non-regression gate: if a
//! technique's estimator silently corrupts the result for the default β=1.0 case,
//! these tests will catch it.
//!
//! Contract:
//! - `compute_qcf_kv` with β=1.0 must return (finite, ≥0) for every estimator.
//! - `per_head.len()` must equal `n_kv_heads`.
//! - When attention is uniform, QCF must be finite and well-defined (no NaN/inf).

use argus_engine::kv::kv_cache::KVLayout;
use argus_engine::qcf::{AggregationMode, QcfKvParams, VDataSource, compute_qcf_kv};
use argus_extension_api::{QcfEstimator, StageParams, find_qcf_estimator};

// ── Shared fixture ──────────────────────────────────────────────────────────

/// Leak a registered estimator built from `sp` (test-only — satisfies the `&dyn` borrow).
fn estimator(name: &str, sp: StageParams) -> &'static dyn QcfEstimator {
    Box::leak((find_qcf_estimator(name)
        .unwrap_or_else(|| panic!("estimator '{name}' registered"))
        .make)(sp, &[]))
}

/// Simple 2-token, 1-head, head_dim=2 setup with uniform attention.
fn make_uniform_params(name: &str, sp: StageParams, target_len: usize) -> QcfKvParams<'static> {
    static V: &[f32] = &[1.0, 2.0, 3.0, 4.0];
    static ATTN: &[f32] = &[0.5, 0.5]; // uniform → β=1.0 safe default
    QcfKvParams {
        estimator: estimator(name, sp),
        target_len,
        v_source: VDataSource::F32(V),
        k_source: None,
        attention_scores: ATTN,
        head_attn: None,
        n_kv_heads: 1,
        head_dim: 2,
        current_pos: 2,
        capacity: 2,
        layout: KVLayout::HeadMajor,
        aggregation: AggregationMode::Mean,
        beta: 1.0,
    }
}

// ── Sliding eviction ────────────────────────────────────────────────────────

#[test]
fn spec_beta_one_sliding_fast_path() {
    let (q, ph) = compute_qcf_kv(&make_uniform_params("sliding", StageParams::default(), 1));
    assert!(q.is_finite(), "sliding β=1.0 QCF must be finite: {q}");
    assert!(q >= 0.0, "sliding β=1.0 QCF must be non-negative: {q}");
    assert_eq!(ph.len(), 1, "1 KV head → per_head len=1");
}

// ── H2O eviction ────────────────────────────────────────────────────────────

#[test]
fn spec_beta_one_h2o_fast_path() {
    let sp = StageParams {
        keep_ratio: 0.5,
        protected_prefix: 0,
        ..Default::default()
    };
    let (q, ph) = compute_qcf_kv(&make_uniform_params("h2o", sp, 1));
    assert!(q.is_finite(), "h2o β=1.0 QCF must be finite: {q}");
    assert!(q >= 0.0);
    assert_eq!(ph.len(), 1);
}

// ── Streaming eviction ───────────────────────────────────────────────────────

#[test]
fn spec_beta_one_streaming_fast_path() {
    let sp = StageParams {
        sink_size: 1,
        streaming_window: 1,
        ..Default::default()
    };
    let (q, ph) = compute_qcf_kv(&make_uniform_params("streaming", sp, 0));
    assert!(q.is_finite(), "streaming β=1.0 QCF must be finite: {q}");
    assert!(q >= 0.0);
    assert_eq!(ph.len(), 1);
}

// ── D2O merge ───────────────────────────────────────────────────────────────

#[test]
fn spec_beta_one_d2o_fast_path() {
    let sp = StageParams {
        keep_ratio: 0.5,
        protected_prefix: 0,
        ..Default::default()
    };
    let (q, ph) = compute_qcf_kv(&make_uniform_params("d2o", sp, 1));
    assert!(q.is_finite(), "d2o β=1.0 QCF must be finite: {q}");
    assert!(q >= 0.0);
    assert_eq!(ph.len(), 1);
}

// ── Multi-head invariant ─────────────────────────────────────────────────────

#[test]
fn spec_per_head_len_matches_n_kv_heads() {
    // 2 KV heads, head_dim=4, 3 tokens.
    let v: Vec<f32> = (0..24).map(|x| x as f32).collect(); // 2 heads × 3 tokens × head_dim=4
    let attn: Vec<f32> = vec![0.2, 0.3, 0.5];
    let p = QcfKvParams {
        estimator: estimator("sliding", StageParams::default()),
        target_len: 2,
        v_source: VDataSource::F32(&v),
        k_source: None,
        attention_scores: &attn,
        head_attn: None,
        n_kv_heads: 2,
        head_dim: 4,
        current_pos: 3,
        capacity: 3,
        layout: KVLayout::HeadMajor,
        aggregation: AggregationMode::Mean,
        beta: 1.0,
    };
    let (q, ph) = compute_qcf_kv(&p);
    assert!(q.is_finite(), "2-head β=1.0 QCF must be finite: {q}");
    assert_eq!(
        ph.len(),
        2,
        "per_head len must equal n_kv_heads=2, got {}",
        ph.len()
    );
}

// ── Uniform vs peaked sanity ─────────────────────────────────────────────────

#[test]
fn spec_uniform_attention_gives_finite_qcf() {
    // Uniform attention → evicted token has average importance → non-zero QCF.
    let (q, _) = compute_qcf_kv(&make_uniform_params("sliding", StageParams::default(), 1));
    assert!(q.is_finite() && q >= 0.0, "uniform attention: q={q}");
}

#[test]
fn spec_single_token_cache_ok() {
    // Edge case: target_len=1, current_pos=1 → evict 0 tokens → QCF should be 0.
    static V: &[f32] = &[1.0, 2.0];
    static ATTN: &[f32] = &[1.0];
    let p = QcfKvParams {
        estimator: estimator("sliding", StageParams::default()),
        target_len: 1,
        v_source: VDataSource::F32(V),
        k_source: None,
        attention_scores: ATTN,
        head_attn: None,
        n_kv_heads: 1,
        head_dim: 2,
        current_pos: 1,
        capacity: 1,
        layout: KVLayout::HeadMajor,
        aggregation: AggregationMode::Mean,
        beta: 1.0,
    };
    let (q, ph) = compute_qcf_kv(&p);
    assert!(q.is_finite(), "single-token edge case: q={q}");
    assert_eq!(ph.len(), 1);
}
