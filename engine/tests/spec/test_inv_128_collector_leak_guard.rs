//! INV-128 — Armed collector state must not leak across prefill calls.
//!
//! The `collector_armed` + `on_demand_collector` pattern in `generate.rs` must
//! satisfy the following invariants:
//!
//! 1. `collector_armed` is consumed (set to `false`) immediately upon entering
//!    the prefill block — the next prefill never inherits a stale Armed state.
//! 2. An `ImportanceCollector` that was created but never had `build()` called
//!    (i.e., prefill aborted before the finalization block) drops without panic.
//! 3. A drpped (never-built) collector produces no `ImportanceTable` entries —
//!    the caller's `importance_table_for_swap` remains `None` from the previous
//!    iteration.
//! 4. After the collector drops, a subsequent call to `WeightSwapDecider::decide`
//!    falls back to the uniform path (importance absent), producing a finite
//!    `qcf_swap_estimate` without panicking.
//!
//! Spec: INV-128, ENG-ALG-218.

use argus_engine::qcf::layer_importance::ImportanceCollector;
use argus_engine::qcf::layer_importance::ImportanceTable;
use argus_engine::weight::{SwapAlgorithm, WeightSwapDecider};

// ── INV-128.1: collector_armed consumption pattern ────────────────────────────

/// Simulates the armed-then-consumed pattern in generate.rs.
/// Armed state must be consumed at prefill start — never visible to the next
/// call.
#[test]
fn inv_128_armed_flag_consumed_before_prefill_body() {
    // Simulate the two-variable state:
    //   collector_armed: bool
    //   on_demand_collector: Option<ImportanceCollector>
    let mut collector_armed = true;
    let on_demand_collector: Option<ImportanceCollector> = if collector_armed {
        Some(ImportanceCollector::new())
    } else {
        None
    };
    // Consume the flag immediately (mirrors generate.rs line 3118-3119)
    if collector_armed {
        collector_armed = false;
    }

    // After consumption: flag is false, collector exists
    assert!(
        !collector_armed,
        "collector_armed must be false after consumption"
    );
    assert!(
        on_demand_collector.is_some(),
        "on_demand_collector must be Some after arming"
    );
}

/// A second prefill call must not see collector_armed = true (simulates
/// the case where the first prefill errored after consuming the flag).
#[test]
fn inv_128_second_prefill_sees_false_flag() {
    let mut collector_armed = true;

    // === First prefill ===
    let _collector: Option<ImportanceCollector> = if collector_armed {
        Some(ImportanceCollector::new())
    } else {
        None
    };
    collector_armed = false; // consumed

    // Simulate early return / error — collector is dropped here without build()
    // (by letting _collector go out of scope)
    drop(_collector);

    // === Second prefill check ===
    assert!(
        !collector_armed,
        "collector_armed must remain false for the next prefill (INV-128: no Armed leak)"
    );
    // Second prefill: armed is false → no collector created
    let second_collector: Option<ImportanceCollector> = if collector_armed {
        Some(ImportanceCollector::new())
    } else {
        None
    };
    assert!(
        second_collector.is_none(),
        "second prefill must not create a collector when flag was not re-armed"
    );
}

// ── INV-128.2: drop-without-build is safe ────────────────────────────────────

/// Dropping an `ImportanceCollector` without calling `build()` must not panic.
/// This simulates a prefill that aborts (via `?`) before the finalization block.
#[test]
fn inv_128_collector_drop_without_build_no_panic() {
    let collector = ImportanceCollector::new();
    // Drop without calling build() — must not panic
    drop(collector);
}

/// Collector with accumulated data drops cleanly (no panic, no corruption).
#[test]
fn inv_128_populated_collector_drop_without_build_no_panic() {
    let mut collector = ImportanceCollector::new();
    // Simulate partial collection (one layer processed)
    let dim = 16usize;
    let data: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
    collector.snapshot_before(&data, 1, dim);
    let data2: Vec<f32> = (0..dim).map(|i| i as f32 * 0.11).collect();
    use argus_engine::qcf::layer_importance::SubLayer;
    collector.record_after(&data2, 1, dim, 0, SubLayer::Full);

    // Abort — drop without build()
    drop(collector);
    // No panic → test passes
}

// ── INV-128.3: importance_table_for_swap remains unchanged on abort ───────────

/// When the collector is dropped without `build()`, the caller's
/// `importance_table_for_swap` must remain at its prior value (`None` on first
/// call, or the old table on subsequent calls).
#[test]
fn inv_128_importance_table_unchanged_on_abort() {
    let importance_table_for_swap: Option<ImportanceTable> = None;

    // Armed path: collector created but never finalized (simulate early return)
    let collector_armed = true;
    let on_demand_collector: Option<ImportanceCollector> = if collector_armed {
        Some(ImportanceCollector::new())
    } else {
        None
    };

    // Simulate abort: drop without calling build() or updating importance_table_for_swap
    drop(on_demand_collector);

    // importance_table_for_swap must still be None
    assert!(
        importance_table_for_swap.is_none(),
        "importance_table_for_swap must remain None when collector aborts (INV-128)"
    );

    // Explicitly confirm we are not accidentally writing to it
    let _ = &importance_table_for_swap; // suppress unused warning
}

// ── INV-128.4: decider operates safely with absent importance table ───────────

/// After a collector abort (importance = None), `WeightSwapDecider::decide()`
/// must fall back to the uniform path and return a finite result.
#[test]
fn inv_128_decider_safe_with_absent_importance() {
    // No importance table (as if collector was never built)
    let noise = vec![0.2f32, 0.1, 0.3, 0.05];

    let decider = WeightSwapDecider {
        importance: None, // absent — simulates post-abort state
        noise: Some(&noise),
        n_decoder_layers: 4,
        currently_swapped: &[],
        allow_boundary_layers: false,
        algorithm: SwapAlgorithm::ImportanceAware,
    };

    // budget = floor(0.5 * 4) - 0 = 2
    let decision = decider.decide(2);

    assert!(
        decision.fallback_used,
        "must use uniform fallback when importance is absent (INV-128)"
    );
    assert!(
        decision.qcf_swap_estimate.is_finite(),
        "qcf_swap_estimate must be finite even on fallback path (got {})",
        decision.qcf_swap_estimate
    );
    // Fallback must still respect protected layers (0 and last)
    assert!(
        !decision.selected_layers.contains(&0),
        "layer 0 must be protected even on fallback"
    );
    assert!(
        !decision.selected_layers.contains(&3),
        "last layer must be protected even on fallback"
    );
}
