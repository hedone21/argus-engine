//! INV-127 — NaN ε layers excluded from `WeightSwapDecider` candidates.
//!
//! A `QuantNoiseTable` entry of `f32::NAN` signals a computation failure for
//! that layer.  `WeightSwapDecider::decide()` must never include a NaN-ε layer
//! in `selected_layers`, and the resulting `qcf_swap_estimate` must be a finite
//! value (no NaN propagation).
//!
//! Spec: INV-127, ENG-DAT-095, ENG-ALG-215.

use argus_engine::weight::{SwapAlgorithm, WeightSwapDecider};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// MW-C: flat per-layer importance (index = layer_id, value = `SubLayer::Full`
/// importance). decider 의 `importance: Option<&[f32]>` 와 동형. 구
/// `ImportanceTable::from_entries` 입력을 동일한 평탄 형태로 변환.
fn make_importance(entries: Vec<(usize, f32)>) -> Vec<f32> {
    let n = entries.iter().map(|(id, _)| id + 1).max().unwrap_or(0);
    let mut out = vec![0.0f32; n];
    for (id, imp) in entries {
        out[id] = imp;
    }
    out
}

// ── INV-127 tests ─────────────────────────────────────────────────────────────

/// Spec fixture: 4-layer model, layer 1 has NaN ε.
/// importance = [0.1, 0.5, 0.3, 0.7], ε = [0.2, NaN, 0.3, 0.05]
/// Layers 0 and 3 are protected; candidate set before NaN filter = [1, 2].
/// After NaN filter: [2] only.
/// ratio=0.5 → k=floor(0.5 × 4)=2, need=2 from [2] → only [2] selected.
#[test]
fn inv_127_nan_layer_excluded_from_candidates() {
    let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
    let noise = vec![0.2f32, f32::NAN, 0.3, 0.05];

    let decider = WeightSwapDecider {
        importance: Some(&importance),
        noise: Some(&noise),
        n_decoder_layers: 4,
        currently_swapped: &[],
        allow_boundary_layers: false,
        algorithm: SwapAlgorithm::ImportanceAware,
    };

    // budget = floor(0.5 * 4) - 0 = 2
    let decision = decider.decide(2);

    assert!(
        !decision.selected_layers.contains(&1),
        "layer 1 with NaN ε must NOT be selected (INV-127)"
    );
    // Layer 2 is the only valid candidate after filtering
    assert!(
        decision.selected_layers.contains(&2),
        "layer 2 (valid ε) must be selected"
    );
}

/// QCF_swap estimate must be finite even when some layers have NaN ε.
#[test]
fn inv_127_qcf_estimate_is_finite_with_nan_layers() {
    let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
    let noise = vec![0.2f32, f32::NAN, 0.3, 0.05];

    let decider = WeightSwapDecider {
        importance: Some(&importance),
        noise: Some(&noise),
        n_decoder_layers: 4,
        currently_swapped: &[],
        allow_boundary_layers: false,
        algorithm: SwapAlgorithm::ImportanceAware,
    };

    // budget = floor(0.5 * 4) - 0 = 2
    let decision = decider.decide(2);

    assert!(
        decision.qcf_swap_estimate.is_finite(),
        "qcf_swap_estimate must be finite (got {:.6}) even with NaN ε layers (INV-127)",
        decision.qcf_swap_estimate
    );
}

/// When ALL non-protected layers have NaN ε, selected_layers must be empty
/// (nothing to swap), and qcf_swap = 0.0.
#[test]
fn inv_127_all_candidates_nan_gives_empty_selection() {
    let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
    // All non-protected (layers 1, 2) are NaN; protected (0, 3) NaN too
    let noise = vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN];

    let decider = WeightSwapDecider {
        importance: Some(&importance),
        noise: Some(&noise),
        n_decoder_layers: 4,
        currently_swapped: &[],
        allow_boundary_layers: false,
        algorithm: SwapAlgorithm::ImportanceAware,
    };

    // budget = floor(0.5 * 4) - 0 = 2
    let decision = decider.decide(2);

    assert!(
        decision.selected_layers.is_empty(),
        "all-NaN noise must give empty selection, got {:?}",
        decision.selected_layers
    );
    assert_eq!(
        decision.qcf_swap_estimate, 0.0,
        "empty selection → qcf_swap must be 0.0"
    );
}

/// NaN ε in a layer that is already swapped: must be excluded from re-selection.
#[test]
fn inv_127_nan_layer_not_re_selected_even_if_currently_swapped() {
    let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
    let noise = vec![0.2f32, f32::NAN, 0.3, 0.05];

    // Layer 2 is already swapped — the only remaining valid candidate (layer 1) has NaN ε.
    let decider = WeightSwapDecider {
        importance: Some(&importance),
        noise: Some(&noise),
        n_decoder_layers: 4,
        currently_swapped: &[2],
        allow_boundary_layers: false,
        algorithm: SwapAlgorithm::ImportanceAware,
    };

    // budget = floor(0.5 * 4) - |{2}| = 2 - 1 = 1
    let decision = decider.decide(1);

    assert!(
        !decision.selected_layers.contains(&1),
        "NaN-ε layer 1 must not be re-selected even when it is the only remaining candidate"
    );
}

/// NaN propagation check: compute_qcf_weight_swap with a NaN layer in the swap_set
/// must still produce a finite value (NaN layer contributes 0 to both parts).
#[test]
fn inv_127_compute_qcf_weight_swap_nan_layer_contributes_zero() {
    use argus_engine::weight::compute_qcf_weight_swap;

    let noise = vec![0.2f32, f32::NAN, 0.3, 0.05];
    let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);

    // Include layer 1 (NaN ε) in the swap set explicitly
    let qcf_with_nan = compute_qcf_weight_swap(&[0, 1, 2], &noise, Some(&importance), 4);
    let qcf_without_nan = compute_qcf_weight_swap(&[0, 2], &noise, Some(&importance), 4);

    assert!(
        qcf_with_nan.is_finite(),
        "qcf_swap including NaN layer must be finite (INV-127), got {qcf_with_nan}"
    );
    assert!(
        (qcf_with_nan - qcf_without_nan).abs() < 1e-6,
        "NaN layer must contribute 0 to qcf_swap: with={qcf_with_nan}, without={qcf_without_nan}"
    );
}
