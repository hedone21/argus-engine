//! Spec test for ARGUS QCF experimental dump (Step 6).
//!
//! Verifies invariants of aggregation modes, top-K retention, entropy,
//! layer aggregation, and auto sample-layer selection.
//!
//! These invariants are the regression gate for the ARGUS QCF experimental
//! infrastructure introduced in Steps 1–6. The tests do not require a model
//! file and run on every `cargo test` invocation.

use argus_engine::qcf::{
    AggregationMode, LayerAggregationMode, aggregate_heads, aggregate_layers,
    compute_auto_sample_layers, compute_normalized_entropy, compute_topk_retention,
};

// ── Aggregation invariants ──────────────────────────────────────────────────

#[test]
fn spec_aggregation_max_geq_mean() {
    let v = vec![0.1, 0.5, 0.2];
    let m = aggregate_heads(&v, &AggregationMode::Mean);
    let mx = aggregate_heads(&v, &AggregationMode::Max);
    assert!(
        mx >= m - 1e-6,
        "Max ({mx}) must be >= Mean ({m}) for non-negative values"
    );
}

#[test]
fn spec_aggregation_topk_k1_equals_max() {
    let v = vec![0.1, 0.5, 0.2, 0.4];
    let mx = aggregate_heads(&v, &AggregationMode::Max);
    let t1 = aggregate_heads(&v, &AggregationMode::TopK { k: 1 });
    assert!(
        (mx - t1).abs() < 1e-6,
        "TopK(k=1) ({t1}) must equal Max ({mx})"
    );
}

// ── Top-K retention invariants ──────────────────────────────────────────────

#[test]
fn spec_topk_retention_in_unit_interval() {
    let importance: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let evicted: std::collections::HashSet<usize> = (0..10).collect();
    let r = compute_topk_retention(&importance, &evicted, 5);
    assert!(
        0.0 <= r.retention_binary && r.retention_binary <= 1.0,
        "retention_binary={} must be in [0,1]",
        r.retention_binary
    );
    assert!(
        0.0 <= r.retention_weighted && r.retention_weighted <= 1.0,
        "retention_weighted={} must be in [0,1]",
        r.retention_weighted
    );
}

#[test]
fn spec_topk_retention_all_evicted_is_zero() {
    let importance: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let evicted: std::collections::HashSet<usize> = (0..5).collect();
    let r = compute_topk_retention(&importance, &evicted, 5);
    assert!(
        r.retention_binary < 1e-6,
        "all evicted → retention_binary must be 0, got {}",
        r.retention_binary
    );
    assert!(
        r.retention_weighted < 1e-6,
        "all evicted → retention_weighted must be 0, got {}",
        r.retention_weighted
    );
}

#[test]
fn spec_topk_retention_none_evicted_is_one() {
    let importance: Vec<f32> = vec![1.0, 2.0, 3.0];
    let evicted: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let r = compute_topk_retention(&importance, &evicted, 3);
    assert!(
        (r.retention_binary - 1.0).abs() < 1e-6,
        "no eviction → retention_binary must be 1.0, got {}",
        r.retention_binary
    );
}

// ── Entropy invariants ──────────────────────────────────────────────────────

#[test]
fn spec_entropy_normalized_in_unit_interval() {
    let r = compute_normalized_entropy(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(
        0.0 <= r.entropy_normalized && r.entropy_normalized <= 1.0,
        "entropy_normalized={} must be in [0,1]",
        r.entropy_normalized
    );
}

#[test]
fn spec_entropy_uniform_is_one() {
    let r = compute_normalized_entropy(&[1.0; 8]);
    assert!(
        (r.entropy_normalized - 1.0).abs() < 1e-5,
        "uniform distribution → entropy_normalized must be 1.0, got {}",
        r.entropy_normalized
    );
}

#[test]
fn spec_entropy_degenerate_is_zero() {
    // One token holds all attention mass → minimal entropy.
    let mut scores = vec![0.0f32; 16];
    scores[0] = 1.0;
    let r = compute_normalized_entropy(&scores);
    assert!(
        r.entropy_normalized < 1e-5,
        "degenerate distribution → entropy_normalized must be ~0, got {}",
        r.entropy_normalized
    );
}

// ── Layer aggregation invariants ────────────────────────────────────────────

#[test]
fn spec_layer_aggregation_max_dominates() {
    let v = vec![0.1, 0.3, 0.2];
    let mx = aggregate_layers(&v, &LayerAggregationMode::Max);
    assert!(
        (mx - 0.3).abs() < 1e-6,
        "LayerAggregationMode::Max must return 0.3, got {mx}"
    );
}

#[test]
fn spec_layer_aggregation_mean() {
    let v = vec![0.1, 0.3, 0.2];
    let mean = aggregate_layers(&v, &LayerAggregationMode::Mean);
    let expected = (0.1 + 0.3 + 0.2) / 3.0;
    assert!(
        (mean - expected).abs() < 1e-6,
        "LayerAggregationMode::Mean got {mean}, expected {expected}"
    );
}

// ── Auto sample-layer invariants ────────────────────────────────────────────

#[test]
fn spec_auto_sample_layers_count() {
    for n in [10usize, 16, 28, 32, 64] {
        let v = compute_auto_sample_layers(n);
        assert_eq!(v.len(), 5, "auto sample 항상 5개 (n={n})");
        assert_eq!(v[0], 0, "첫 번째 index는 항상 0 (n={n})");
        assert_eq!(*v.last().unwrap(), n - 1, "마지막 index는 항상 n-1 (n={n})");
    }
}

#[test]
fn spec_auto_sample_layers_small_model() {
    // n < 5 → all layers returned.
    let v = compute_auto_sample_layers(3);
    assert_eq!(v, vec![0, 1, 2]);
}

#[test]
fn spec_auto_sample_layers_empty() {
    let v = compute_auto_sample_layers(0);
    assert!(v.is_empty());
}

#[test]
fn spec_auto_sample_layers_sorted_unique() {
    for n in [6usize, 10, 16, 32] {
        let v = compute_auto_sample_layers(n);
        let sorted = {
            let mut s = v.clone();
            s.sort();
            s.dedup();
            s
        };
        assert_eq!(
            v, sorted,
            "auto sample layers must be sorted and unique (n={n})"
        );
    }
}
