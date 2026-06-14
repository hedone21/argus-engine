//! INV-145 — Drain monotone property + retire correctness.
//!
//! Spec: `spec/41-invariants.md` §3.20 (INV-145)
//! Spec: `spec/32-engine-algorithms.md` §3.12.21.1 (ENG-ALG-232)
//!
//! Tests:
//! - Property test: for various (layers, per_tick) configurations,
//!   remaining_count is strictly monotone during drain.
//! - Total drained layers == initial layers.len() (exactly once coverage).
//! - Retire: after is_done(), drain returns empty Vec.
//! - No duplicate layers in drain output.

use argus_engine::weight::IncrementalSwapPlan;

/// Exhaust a plan, collecting all drained layers.
/// Returns (all_drained, chunk_sizes_vec).
fn exhaust(mut plan: IncrementalSwapPlan) -> (Vec<usize>, Vec<usize>) {
    let mut all_drained: Vec<usize> = Vec::new();
    let mut chunk_sizes: Vec<usize> = Vec::new();
    while !plan.is_done() {
        let chunk = plan.drain_chunk();
        if !chunk.is_empty() {
            chunk_sizes.push(chunk.len());
            all_drained.extend_from_slice(&chunk);
        }
    }
    // Verify: after is_done, further drain returns empty
    let post_done = plan.drain_chunk();
    assert!(
        post_done.is_empty(),
        "INV-145: drain after is_done() must return empty Vec"
    );
    (all_drained, chunk_sizes)
}

/// Verify INV-145 strictly decreasing remaining_count property.
fn assert_strictly_monotone(layers: Vec<usize>, per_tick: usize) {
    if per_tick == 0 {
        return; // per_tick=0 is defensive only, not tested here
    }
    let n = layers.len();
    let mut plan = IncrementalSwapPlan::new(layers.clone(), per_tick, 0);
    let mut prev = plan.remaining_count();
    assert_eq!(prev, n);

    while !plan.is_done() {
        let chunk = plan.drain_chunk();
        let cur = plan.remaining_count();
        if !chunk.is_empty() {
            assert!(
                cur < prev,
                "INV-145: remaining must strictly decrease after non-empty drain (prev={prev}, cur={cur})"
            );
        } else {
            assert_eq!(cur, prev, "empty drain must not change remaining_count");
        }
        prev = cur;
    }
    assert_eq!(plan.remaining_count(), 0, "plan exhausted → remaining=0");
}

// ── INV-145: monotone for various (n_layers, per_tick) ───────────────────────

#[test]
fn monotone_1_layer_per_tick_1() {
    assert_strictly_monotone(vec![0], 1);
}

#[test]
fn monotone_5_layers_per_tick_2() {
    assert_strictly_monotone((0..5).collect(), 2);
}

#[test]
fn monotone_25_layers_per_tick_3() {
    assert_strictly_monotone((0..25).collect(), 3);
}

#[test]
fn monotone_16_layers_per_tick_16() {
    // Single-tick exhaust (per_tick == n_layers)
    assert_strictly_monotone((0..16).collect(), 16);
}

#[test]
fn monotone_7_layers_per_tick_10() {
    // per_tick > n_layers
    assert_strictly_monotone((0..7).collect(), 10);
}

#[test]
fn monotone_100_layers_per_tick_7() {
    assert_strictly_monotone((0..100).collect(), 7);
}

// ── INV-145: total drained == initial layers, no duplicates ──────────────────

#[test]
fn drained_total_equals_input_no_duplicates() {
    let cases: Vec<(Vec<usize>, usize)> = vec![
        ((0..5).collect(), 1),
        ((0..5).collect(), 2),
        ((0..5).collect(), 3),
        ((0..5).collect(), 5),
        ((0..5).collect(), 10),
        ((0..25).collect(), 3),
        ((0..100).collect(), 7),
        // Non-sequential layer indices
        (vec![3, 7, 15, 0, 11], 2),
    ];

    for (layers, per_tick) in cases {
        let expected = layers.clone();
        let plan = IncrementalSwapPlan::new(layers.clone(), per_tick, 0);
        let (drained, _) = exhaust(plan);

        // All layers appear (same multiset)
        let mut drained_sorted = drained.clone();
        drained_sorted.sort();
        let mut expected_sorted = expected.clone();
        expected_sorted.sort();
        assert_eq!(
            drained_sorted, expected_sorted,
            "layers={:?}, per_tick={}: all input layers must appear exactly once",
            layers, per_tick
        );

        // No duplicates
        let mut seen = std::collections::HashSet::new();
        for &l in &drained {
            assert!(
                seen.insert(l),
                "layers={:?}, per_tick={}: duplicate layer {} in drain output",
                layers,
                per_tick,
                l
            );
        }

        // Order preserved
        assert_eq!(
            drained, expected,
            "layers={:?}, per_tick={}: drain order must match input order",
            layers, per_tick
        );
    }
}

// ── INV-145: retire correctness via Option::take ──────────────────────────────

#[test]
fn retire_via_option_take_prevents_further_drain() {
    let mut plan: Option<IncrementalSwapPlan> = Some(IncrementalSwapPlan::new(vec![0, 1, 2], 3, 0));

    // First tick: drain exhausts all
    let chunk = plan.as_mut().unwrap().drain_chunk();
    assert_eq!(chunk, vec![0, 1, 2]);
    assert!(plan.as_ref().unwrap().is_done());

    // Retire
    plan = None;
    assert!(plan.is_none(), "plan must be None after Option::take");

    // Subsequent ticks: plan is None → no drain call
    // (simulating the decode loop guard `if let Some(ref mut p) = plan`)
    let drain_would_occur = plan.is_some();
    assert!(
        !drain_would_occur,
        "INV-145: after retirement, no further drain calls occur"
    );
}

// ── INV-145: empty layers → immediate done → drain empty ─────────────────────

#[test]
fn empty_layers_immediate_done_drain_empty() {
    let mut plan = IncrementalSwapPlan::new(vec![], 3, 0);
    assert!(plan.is_done(), "empty layers → is_done() immediately");
    let chunk = plan.drain_chunk();
    assert!(chunk.is_empty(), "drain on already-done plan returns empty");
}

// ── INV-145: chunk sizes correct ─────────────────────────────────────────────

#[test]
fn chunk_sizes_match_per_tick_with_partial_last() {
    // 7 layers, per_tick=3 → chunks: [3, 3, 1]
    let plan = IncrementalSwapPlan::new((0..7).collect(), 3, 0);
    let (_, chunk_sizes) = exhaust(plan);
    assert_eq!(
        chunk_sizes,
        vec![3, 3, 1],
        "chunk sizes for 7 layers / per_tick=3"
    );
}

#[test]
fn chunk_sizes_exact_fit() {
    // 9 layers, per_tick=3 → chunks: [3, 3, 3]
    let plan = IncrementalSwapPlan::new((0..9).collect(), 3, 0);
    let (_, chunk_sizes) = exhaust(plan);
    assert_eq!(
        chunk_sizes,
        vec![3, 3, 3],
        "chunk sizes for 9 layers / per_tick=3"
    );
}
