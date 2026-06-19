//! Weight-swap engine-side residue (EPIC 3 B3-3).
//!
//! The `WeightSwapDecider` ranking core (importance × ε bottom-k layer selection,
//! ENG-ALG-215) and its `WeightStage` adapter were extracted into the
//! `weight-swap` technique crate. This module retains only the two engine-coupled
//! pieces that cannot leave the engine, plus a re-export of the moved decider
//! types so the submodule path `crate::weight::decider::{...}` keeps resolving:
//!
//! - [`compute_qcf_weight_swap`] — the timed QCF_swap wrapper. It wraps the
//!   engine-only `crate::qcf_timer!` profile macro and delegates the arithmetic
//!   to the plugin's `weight_swap::compute_qcf_swap_internal` (single copy).
//! - [`flatten_importance`] — projects an `ImportanceLookup` (engine type) into
//!   the flat `Vec<f32>` the decider/ctx consume.
//!
//! Spec: ENG-ALG-215, ENG-ALG-217, INV-127.

use crate::qcf_collector::ImportanceLookup;
use crate::qcf_types::SubLayer;

/// Re-export the moved decider types from the `weight-swap` plugin so that both
/// `crate::weight::{...}` (via `weight.rs`) and the deeper `crate::weight::decider::{...}`
/// consumer paths resolve byte-identically (mirror of `engine/src/qcf.rs`
/// re-exporting the `layer-importance` plugin types).
pub use weight_swap::{SwapAlgorithm, SwapDecision, WeightSwapDecider};

/// `ImportanceLookup` 를 layer 인덱스 기준의 평탄 `Vec<f32>` 로 투영한다 (MW-C).
///
/// 길이 `n` 의 벡터를 반환하며, 각 원소는 해당 layer 의 `SubLayer::Full`
/// importance 값이다. lookup 에 해당 entry 가 없으면 `0.0` 으로 채운다 (decider
/// ranking 의 기존 디폴트 `0.0` 보존). decider 의 flat `importance: Option<&[f32]>`
/// 필드에 넣기 위한 호출자 측 어댑터.
pub fn flatten_importance(lookup: &dyn ImportanceLookup, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for e in lookup.entries() {
        if e.sublayer == SubLayer::Full && e.layer_id < n {
            out[e.layer_id] = e.importance;
        }
    }
    out
}

// ── QCF_swap computation (ENG-ALG-217) ───────────────────────────────────────

/// Compute QCF_swap for a given set of swapped layers (ENG-ALG-217).
///
/// ```text
/// QCF_swap(S) = Σ_{i ∈ S} importance_i × ε_i
///               ───────────────────────────────
///               Σ_{j ∈ all_valid} importance_j × ε_j
/// ```
///
/// - Layers with NaN ε are excluded from both numerator and denominator.
/// - Missing importance entries (table absent) default to `1.0`.
/// - Returns `0.0` when `swap_set` is empty or denominator ≈ 0.
///
/// `importance`/`noise` 는 layer 인덱스 기준의 평탄 슬라이스다 (MW-C). `noise` 는
/// `&[f32]` (decide() 의 계약대로 ε 가 실제 계산된 경우의 슬라이스); NaN 원소는
/// numerator/denominator 양쪽에서 제외된다.
///
/// Engine-side timed wrapper: keeps the engine-only `qcf_timer!` and delegates
/// the arithmetic to the `weight-swap` plugin (B3-3, single copy).
pub fn compute_qcf_weight_swap(
    swap_set: &[usize],
    noise: &[f32],
    importance: Option<&[f32]>,
    n_decoder_layers: usize,
) -> f32 {
    let _t = crate::qcf_timer!(QCF_WEIGHT_SWAP);
    weight_swap::compute_qcf_swap_internal(swap_set, n_decoder_layers, importance, Some(noise))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// layer 인덱스 기준 평탄 ε. index=layer_id, value=ε (NaN 허용).
    fn make_noise(vals: Vec<f32>) -> Vec<f32> {
        vals
    }

    // ── compute_qcf_weight_swap tests (ENG-ALG-217) ─────────────────────────────────

    /// Empty swap set → QCF_swap = 0.0.
    #[test]
    fn qcf_swap_empty_set_is_zero() {
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        let result = compute_qcf_weight_swap(&[], &noise, None, 4);
        assert_eq!(result, 0.0);
    }

    /// All-layers swap set (excluding NaN) → QCF_swap ≈ 1.0.
    #[test]
    fn qcf_swap_full_set_approx_one() {
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        // All layers in the "valid" set (no NaN)
        let result = compute_qcf_weight_swap(&[0, 1, 2, 3], &noise, None, 4);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "full set should give QCF_swap ≈ 1.0, got {result}"
        );
    }

    /// Monotonic property: adding a layer to the set must not decrease QCF_swap.
    #[test]
    fn qcf_swap_monotonic() {
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        let q1 = compute_qcf_weight_swap(&[1], &noise, None, 4);
        let q2 = compute_qcf_weight_swap(&[1, 2], &noise, None, 4);
        let q3 = compute_qcf_weight_swap(&[0, 1, 2], &noise, None, 4);
        assert!(q1 <= q2, "monotonic: q({{1}})={q1} <= q({{1,2}})={q2}");
        assert!(q2 <= q3, "monotonic: q({{1,2}})={q2} <= q({{0,1,2}})={q3}");
    }

    /// NaN ε layer contributes 0 to both numerator and denominator.
    #[test]
    fn qcf_swap_nan_layer_excluded_from_both() {
        let noise = make_noise(vec![0.2, f32::NAN, 0.3, 0.05]);
        // Layer 1 has NaN ε — should contribute 0 to numerator and denominator.
        // Including it in swap_set should not change result vs. excluding it.
        let without_nan = compute_qcf_weight_swap(&[0, 2, 3], &noise, None, 4);
        let with_nan = compute_qcf_weight_swap(&[0, 1, 2, 3], &noise, None, 4);
        assert!(
            (without_nan - with_nan).abs() < 1e-6,
            "NaN layer should not affect QCF_swap: without={without_nan}, with={with_nan}"
        );
    }
}
