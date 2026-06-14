//! Spec test: ENG-ALG-051 Unified QCF — O_before normalisation symmetry.
//!
//! ENG-ALG-051 was revised (2026-06-12, spec commit `eab908ab`) so that
//! `O_before` is normalised by the **full** token-set α-sum `Σ_s α_s`, sharing
//! the same softmax-weight space as `O_after`'s retained-set normalisation.
//! These tests pin the two consequences of that revision:
//!
//!   1. **Identity action → QCF = 0** (invariant, spec L647-652): when an action
//!      retains every token, `retained == full set`, so `O_after == O_before` and
//!      `QCF == 0`. This holds *only* because `O_before` is now normalised — with
//!      raw `Σ_t α_t V_t` the identity retain would give
//!      `O_after = O_before / Σ_s α_s ≠ O_before`.
//!   2. **Saturation regression guard**: with an accumulated (unnormalised)
//!      attention score whose sum is ≫ 1 (e.g. Σα ≈ 12, the decode score-sink
//!      regime that previously saturated QCF at ~0.985), the metric is now
//!      scale-invariant and yields a small QCF (< 0.5) for a realistic
//!      (correlated-V, sink-preserving) eviction.

use argus_engine::kv::kv_cache::KVLayout;
use argus_engine::qcf::{AggregationMode, QcfActionType, QcfKvParams, VDataSource, compute_qcf_kv};

const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const CAPACITY: usize = 16;
const CURRENT_POS: usize = 12;

/// Correlated value vectors mimicking real LLM V cache: a dominant low-frequency
/// direction shared across tokens, with a small per-token phase drift. (Highly
/// orthogonal synthetic V would inflate QCF beyond [0,1] — the spec's [0,1]
/// invariant assumes near-convex combinations, which correlated V provides.)
fn correlated_v() -> Vec<f32> {
    let freq = 0.05f32;
    let mut v = vec![0.0f32; N_KV_HEADS * CAPACITY * HEAD_DIM];
    for h in 0..N_KV_HEADS {
        for t in 0..CURRENT_POS {
            let phase = (h as f32) * 0.2 + (t as f32) * 0.08;
            for d in 0..HEAD_DIM {
                v[h * CAPACITY * HEAD_DIM + t * HEAD_DIM + d] = (phase + d as f32 * freq).sin();
            }
        }
    }
    v
}

/// Moderately non-uniform base attention, scaled so its sum equals `target`.
fn scores_summing_to(target: f32) -> Vec<f32> {
    let base: Vec<f32> = (0..CURRENT_POS)
        .map(|t| 1.0 + 0.3 * ((t as f32) * 0.8).sin())
        .collect();
    let s: f32 = base.iter().sum();
    base.iter().map(|x| x * target / s).collect()
}

fn make_params<'a>(action: QcfActionType, v: &'a [f32], scores: &'a [f32]) -> QcfKvParams<'a> {
    QcfKvParams {
        action,
        v_source: VDataSource::F32(v),
        k_source: None,
        attention_scores: scores,
        head_attn: None,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        current_pos: CURRENT_POS,
        capacity: CAPACITY,
        layout: KVLayout::HeadMajor,
        aggregation: AggregationMode::Mean,
        beta: 1.0,
    }
}

#[test]
fn test_eng_alg_051_identity_action_zero_qcf() {
    // Retain every token (target_len >= current_pos) → QCF must be exactly 0.
    // Uses the saturation-regime score (Σα ≈ 12) to prove the identity holds
    // independent of the α-sum scale (the pre-revision raw O_before would give
    // a non-zero QCF here).
    let v = correlated_v();
    let scores = scores_summing_to(12.0);
    let params = make_params(
        QcfActionType::EvictSliding {
            target_len: CURRENT_POS,
        },
        &v,
        &scores,
    );
    let (qcf, per_head) = compute_qcf_kv(&params);
    assert_eq!(
        qcf, 0.0,
        "ENG-ALG-051 identity action (full retain) must give QCF=0, got {qcf}"
    );
    for (h, &ph) in per_head.iter().enumerate() {
        assert_eq!(
            ph, 0.0,
            "head {h}: identity action must give per-head QCF=0, got {ph}"
        );
    }
}

#[test]
fn test_eng_alg_051_no_saturation_under_accumulated_scores() {
    // Regression guard against the 0.985 saturation: an unnormalised
    // accumulated score (Σα ≈ 12) with a realistic correlated-V eviction must
    // yield a small relative perturbation, NOT a near-1 saturated value.
    let v = correlated_v();
    let scores = scores_summing_to(12.0);

    // Sliding eviction retaining 10 of 12 tokens (sink + recent preserved).
    let (qcf, _) = compute_qcf_kv(&make_params(
        QcfActionType::EvictSliding { target_len: 10 },
        &v,
        &scores,
    ));
    assert!(
        qcf < 0.5,
        "ENG-ALG-051: accumulated score (Σα≈12) must not saturate; \
         expected QCF < 0.5 (S25 measured 0.08~0.22), got {qcf}"
    );
    assert!(
        qcf > 0.0,
        "eviction of 2 tokens should give a positive QCF, got {qcf}"
    );
}

#[test]
fn test_eng_alg_051_scale_invariant_after_normalisation() {
    // The O_before normalisation makes QCF invariant to the α-sum scale: the
    // same attention *distribution* at Σα=1 and Σα=12 must give the same QCF.
    // (Pre-revision, the raw O_before scaled with Σα and the metric saturated.)
    let v = correlated_v();
    let scores_unit = scores_summing_to(1.0);
    let scores_accum = scores_summing_to(12.0);

    let (qcf_unit, _) = compute_qcf_kv(&make_params(
        QcfActionType::EvictSliding { target_len: 10 },
        &v,
        &scores_unit,
    ));
    let (qcf_accum, _) = compute_qcf_kv(&make_params(
        QcfActionType::EvictSliding { target_len: 10 },
        &v,
        &scores_accum,
    ));
    assert!(
        (qcf_unit - qcf_accum).abs() < 1e-4,
        "ENG-ALG-051: QCF must be α-sum scale invariant; \
         Σα=1 gave {qcf_unit}, Σα=12 gave {qcf_accum}"
    );
}
