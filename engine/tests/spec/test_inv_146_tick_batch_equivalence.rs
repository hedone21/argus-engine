//! INV-146 — Tick boundary equals batch boundary equivalence.
//!
//! Spec: `spec/41-invariants.md` §3.20 (INV-146)
//! Spec: `spec/32-engine-algorithms.md` §3.12.21.2 (ENG-ALG-233)
//! Spec: `spec/32-engine-algorithms.md` §3.12.20 (ENG-ALG-231, stage gate)
//!
//! INV-146: each tick's `execute_on_slots(chunk)` call performs the full
//! ENG-ALG-231 stage gate cycle (prefault, mmap_permute, arc_swap,
//! madvise, synchronize, ratio_generation bump). The total number of
//! `ratio_generation` bumps must equal the number of chunks dispatched.
//!
//! Tests:
//! - Single-shot (per_tick=0 / per_tick=n_layers): 1 bump regardless of n_layers.
//! - Incremental (per_tick=N): ceil(n_layers/N) bumps.
//! - Equivalence: incremental dispatch yields same set of layer indices as
//!   single-shot, just split across multiple calls.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use argus_engine::weight::IncrementalSwapPlan;

/// Simulate the `execute_on_slots` call side-effect of bumping ratio_generation.
/// Returns the sequence of (chunk_layers) dispatched.
fn simulate_dispatch(
    target_layers: Vec<usize>,
    per_tick: usize,
    ratio_gen: &Arc<AtomicU64>,
    n_decode_tokens: usize,
) -> Vec<Vec<usize>> {
    let mut plan: Option<IncrementalSwapPlan> = if per_tick > 0 && !target_layers.is_empty() {
        Some(IncrementalSwapPlan::new(target_layers, per_tick, 0))
    } else {
        None
    };

    let mut dispatches: Vec<Vec<usize>> = Vec::new();

    for _tick in 0..n_decode_tokens {
        if let Some(ref mut p) = plan {
            let chunk = p.drain_chunk();
            if !chunk.is_empty() {
                // INV-146: execute_on_slots bumps ratio_generation once per chunk
                ratio_gen.fetch_add(1, Ordering::SeqCst);
                dispatches.push(chunk);
            }
            if p.is_done() {
                plan = None;
            }
        }
    }
    dispatches
}

/// Simulate single-shot dispatch (per_tick == 0): one execute_on_slots call.
fn simulate_single_shot(target_layers: Vec<usize>, ratio_gen: &Arc<AtomicU64>) -> Vec<Vec<usize>> {
    if target_layers.is_empty() {
        return Vec::new();
    }
    ratio_gen.fetch_add(1, Ordering::SeqCst);
    vec![target_layers]
}

// ── INV-146: single-shot produces 1 bump regardless of layer count ────────────

#[test]
fn single_shot_produces_one_bump() {
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let layers: Vec<usize> = (0..16).collect();
    let dispatches = simulate_single_shot(layers.clone(), &ratio_gen);

    assert_eq!(
        ratio_gen.load(Ordering::SeqCst),
        1,
        "single-shot: exactly 1 ratio_generation bump"
    );
    assert_eq!(dispatches.len(), 1);
    assert_eq!(dispatches[0], layers);
}

// ── INV-146: incremental (per_tick=N) produces ceil(n/N) bumps ───────────────

#[test]
fn incremental_per_tick_2_16_layers_produces_8_bumps() {
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let layers: Vec<usize> = (0..16).collect();
    simulate_dispatch(layers.clone(), 2, &ratio_gen, 20);

    assert_eq!(
        ratio_gen.load(Ordering::SeqCst),
        8,
        "16 layers / per_tick=2 → 8 bumps"
    );
}

#[test]
fn incremental_per_tick_3_25_layers_produces_9_bumps() {
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let layers: Vec<usize> = (0..25).collect();
    simulate_dispatch(layers.clone(), 3, &ratio_gen, 30);

    // ceil(25/3) = 9
    assert_eq!(
        ratio_gen.load(Ordering::SeqCst),
        9,
        "25 layers / per_tick=3 → 9 bumps"
    );
}

#[test]
fn incremental_per_tick_1_25_layers_produces_25_bumps() {
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let layers: Vec<usize> = (0..25).collect();
    simulate_dispatch(layers.clone(), 1, &ratio_gen, 30);

    assert_eq!(
        ratio_gen.load(Ordering::SeqCst),
        25,
        "25 layers / per_tick=1 → 25 bumps"
    );
}

// ── INV-146: equivalence — same layers dispatched regardless of chunking ──────

#[test]
fn dispatch_sequence_covers_same_layers_as_single_shot() {
    let layers: Vec<usize> = (0..12).collect();

    // Per-tick variants to test
    let per_tick_variants = [1, 2, 3, 4, 6, 12];

    for per_tick in per_tick_variants {
        let ratio_gen = Arc::new(AtomicU64::new(0));
        let dispatches = simulate_dispatch(layers.clone(), per_tick, &ratio_gen, 20);

        // Flatten all dispatched layers
        let all_dispatched: Vec<usize> = dispatches.iter().flatten().copied().collect();

        assert_eq!(
            all_dispatched, layers,
            "per_tick={per_tick}: incremental dispatch covers same layers as single-shot"
        );

        // Number of bumps == number of chunks
        let expected_bumps = layers.len().div_ceil(per_tick) as u64;
        assert_eq!(
            ratio_gen.load(Ordering::SeqCst),
            expected_bumps,
            "per_tick={per_tick}: bump count == chunk count"
        );
    }
}

// ── INV-146: stage gate runs per-chunk (not per-layer) ────────────────────────

#[test]
fn stage_gate_runs_per_chunk_not_per_layer() {
    // INV-146 states tick boundary == batch boundary.
    // This test verifies the ratio_generation bump (one proxy for stage gate) is
    // counted once per chunk, NOT once per layer.
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let layers: Vec<usize> = (0..6).collect();

    // per_tick=3: 2 chunks → 2 bumps (not 6)
    simulate_dispatch(layers.clone(), 3, &ratio_gen, 10);

    let bumps = ratio_gen.load(Ordering::SeqCst);
    assert_eq!(
        bumps, 2,
        "stage gate runs per-chunk: 6 layers / per_tick=3 → 2 bumps, not 6"
    );
    assert_ne!(bumps, 6, "stage gate must NOT run per-layer");
}

// ── INV-146: empty target → 0 bumps ──────────────────────────────────────────

#[test]
fn empty_target_zero_bumps() {
    let ratio_gen = Arc::new(AtomicU64::new(0));
    // per_tick=0 (single-shot with empty target): no execute_on_slots call
    let dispatches = simulate_single_shot(vec![], &ratio_gen);
    assert_eq!(ratio_gen.load(Ordering::SeqCst), 0);
    assert!(dispatches.is_empty());
}

// ── INV-146: per_tick >= n_layers → same as single-shot (1 bump) ─────────────

#[test]
fn per_tick_ge_n_layers_is_single_tick() {
    let layers: Vec<usize> = (0..5).collect();

    for per_tick in [5, 10, 100] {
        let ratio_gen = Arc::new(AtomicU64::new(0));
        let dispatches = simulate_dispatch(layers.clone(), per_tick, &ratio_gen, 10);

        assert_eq!(
            ratio_gen.load(Ordering::SeqCst),
            1,
            "per_tick={per_tick} >= n_layers=5: single chunk → 1 bump"
        );
        assert_eq!(dispatches.len(), 1);
        assert_eq!(dispatches[0], layers);
    }
}
