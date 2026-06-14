//! ENG-ALG-232 + INV-145 — `IncrementalSwapPlan` data structure and drain semantics.
//!
//! Spec: `spec/32-engine-algorithms.md` §3.12.21.1 (ENG-ALG-232)
//! Spec: `spec/41-invariants.md` §3.20 (INV-145)
//!
//! Tests:
//! - drain progression: per_tick=1/2/5 sequence exhaustion
//! - empty layers → is_done() immediately
//! - per_tick=0 → drain always empty, is_done() always false (defensive)
//! - order preservation: drain returns layers in commit order
//! - is_done() → further drain returns empty Vec
//! - remaining_count() monotone (INV-145)
//! - started_at_token() accessor

use argus_engine::weight::IncrementalSwapPlan;

// ── ENG-ALG-232: basic drain progression ─────────────────────────────────────

#[test]
fn drain_per_tick_2_five_layers() {
    // new(layers=[0,1,2,3,4], per_tick=2) →
    //   drain(): [0,1] → drain(): [2,3] → drain(): [4] → is_done()=true → drain(): []
    let mut plan = IncrementalSwapPlan::new(vec![0, 1, 2, 3, 4], 2, 0);
    assert_eq!(plan.remaining_count(), 5);
    assert!(!plan.is_done());

    let c0 = plan.drain_chunk();
    assert_eq!(c0, vec![0, 1], "first chunk");
    assert_eq!(plan.remaining_count(), 3);
    assert!(!plan.is_done());

    let c1 = plan.drain_chunk();
    assert_eq!(c1, vec![2, 3], "second chunk");
    assert_eq!(plan.remaining_count(), 1);
    assert!(!plan.is_done());

    let c2 = plan.drain_chunk();
    assert_eq!(c2, vec![4], "last (partial) chunk");
    assert_eq!(plan.remaining_count(), 0);
    assert!(plan.is_done());

    // After done: further drain returns empty (INV-145)
    let c3 = plan.drain_chunk();
    assert!(c3.is_empty(), "retired plan returns empty");
    assert!(plan.is_done());
}

#[test]
fn drain_per_tick_1() {
    let layers: Vec<usize> = (0..5).collect();
    let mut plan = IncrementalSwapPlan::new(layers, 1, 0);

    for expected in 0..5usize {
        assert!(!plan.is_done());
        let chunk = plan.drain_chunk();
        assert_eq!(chunk, vec![expected], "per_tick=1: one layer per drain");
    }
    assert!(plan.is_done());
    assert!(plan.drain_chunk().is_empty());
}

#[test]
fn drain_per_tick_5_exact_fit() {
    // per_tick >= len: one drain exhausts all
    let mut plan = IncrementalSwapPlan::new(vec![10, 11, 12, 13, 14], 5, 7);
    let chunk = plan.drain_chunk();
    assert_eq!(chunk, vec![10, 11, 12, 13, 14]);
    assert!(plan.is_done());
}

#[test]
fn drain_per_tick_larger_than_len() {
    // per_tick > len: drains all in one call (min behaviour)
    let mut plan = IncrementalSwapPlan::new(vec![0, 1, 2], 10, 0);
    let chunk = plan.drain_chunk();
    assert_eq!(chunk, vec![0, 1, 2]);
    assert!(plan.is_done());
}

// ── ENG-ALG-232: empty layers → immediate done ───────────────────────────────

#[test]
fn empty_layers_is_done_immediately() {
    let plan = IncrementalSwapPlan::new(vec![], 2, 0);
    assert!(
        plan.is_done(),
        "empty layers → is_done() = true immediately"
    );
    assert_eq!(plan.remaining_count(), 0);
}

// ── INV-145: per_tick=0 defensive behaviour ───────────────────────────────────

#[test]
fn per_tick_zero_always_empty_never_done() {
    // per_tick=0 should NOT be constructed in production (caller uses single-shot).
    // If constructed, drain returns empty and plan never self-retires (INV-145).
    let mut plan = IncrementalSwapPlan::new(vec![0, 1], 0, 0);
    assert!(!plan.is_done(), "per_tick=0 with layers: is_done() = false");

    let c0 = plan.drain_chunk();
    assert!(c0.is_empty(), "per_tick=0: drain returns empty");
    assert!(!plan.is_done(), "per_tick=0: plan never self-retires");

    // Multiple calls still empty
    let c1 = plan.drain_chunk();
    assert!(c1.is_empty());
    assert!(!plan.is_done());
}

// ── INV-145: order preservation ──────────────────────────────────────────────

#[test]
fn drain_order_preserves_commit_order() {
    // Layers in non-sequential order: drain must preserve input order
    let layers = vec![15, 3, 7, 0, 11];
    let mut plan = IncrementalSwapPlan::new(layers.clone(), 2, 0);

    let mut collected: Vec<usize> = Vec::new();
    while !plan.is_done() {
        collected.extend(plan.drain_chunk());
    }
    assert_eq!(collected, layers, "drain must preserve commit order");
}

// ── INV-145: remaining_count strictly monotone while non-empty ────────────────

#[test]
fn remaining_count_strictly_decreases_each_drain() {
    let mut plan = IncrementalSwapPlan::new((0..10).collect(), 3, 0);
    let mut prev = plan.remaining_count();

    while !plan.is_done() {
        let _chunk = plan.drain_chunk();
        let cur = plan.remaining_count();
        assert!(
            cur < prev,
            "remaining_count must strictly decrease: {} -> {}",
            prev,
            cur
        );
        prev = cur;
    }
    assert_eq!(plan.remaining_count(), 0);
}

// ── ENG-ALG-232: accessors ───────────────────────────────────────────────────

#[test]
fn started_at_token_preserved() {
    let plan = IncrementalSwapPlan::new(vec![0], 1, 42);
    assert_eq!(plan.started_at_token(), 42);
}

#[test]
fn remaining_count_initial_equals_layers_len() {
    let layers = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let plan = IncrementalSwapPlan::new(layers.clone(), 3, 0);
    assert_eq!(plan.remaining_count(), layers.len());
}

// ── INV-145: drain sum equals initial layers ──────────────────────────────────

#[test]
fn all_drained_layers_equal_input_set() {
    let layers: Vec<usize> = (0..25).collect();
    let mut plan = IncrementalSwapPlan::new(layers.clone(), 3, 0);
    let mut collected: Vec<usize> = Vec::new();

    while !plan.is_done() {
        collected.extend(plan.drain_chunk());
    }
    assert_eq!(
        collected, layers,
        "all input layers must appear exactly once"
    );
}
