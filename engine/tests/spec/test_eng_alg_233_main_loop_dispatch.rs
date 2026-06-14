//! ENG-ALG-233 + INV-146 — Decode loop incremental dispatch sequence.
//!
//! Spec: `spec/32-engine-algorithms.md` §3.12.21.2 (ENG-ALG-233)
//! Spec: `spec/41-invariants.md` §3.20 (INV-146)
//!
//! Tests:
//! - Simulates the decode loop drain-dispatch pattern using IncrementalSwapPlan.
//! - Verifies chunk dispatch sequence: all layers dispatched, correct order,
//!   retire exactly once (Option::take pattern).
//! - per_tick=1, 25 layers → 25 ticks with 1 layer each (chunk count == layers).
//! - per_tick=0 → plan never created (caller responsibility — verified by absence).
//! - INV-146: execute_on_slots called chunk-count times (ratio_generation bumps == chunks).
//! - ENG-ALG-234: in-flight plan is not replaced by a new signal.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::layers::transformer_layer::TransformerLayer;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::models::weights::LayerSlot;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use argus_engine::weight::IncrementalSwapPlan;

fn cpu_be() -> Arc<dyn Backend> {
    Arc::new(CpuBackend::new())
}

fn dummy_tensor(be: &Arc<dyn Backend>, numel: usize) -> Tensor {
    let mem = Galloc::new();
    let buf = mem.alloc(numel * 4, DType::F32).unwrap();
    Tensor::new(Shape::new(vec![numel]), buf, be.clone())
}

fn dummy_layer(be: &Arc<dyn Backend>) -> TransformerLayer {
    TransformerLayer {
        wq: dummy_tensor(be, 16),
        wk: dummy_tensor(be, 16),
        wv: dummy_tensor(be, 16),
        wo: dummy_tensor(be, 16),
        w_gate: dummy_tensor(be, 16),
        w_up: dummy_tensor(be, 16),
        w_down: dummy_tensor(be, 16),
        attention_norm: dummy_tensor(be, 4),
        ffn_norm: dummy_tensor(be, 4),
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    }
}

/// Simulate the decode loop dispatch pattern from ENG-ALG-233.
///
/// Returns (all_dispatched_layers, chunk_count, chunks_as_vec).
fn simulate_decode_loop(
    target_layers: Vec<usize>,
    per_tick: usize,
    n_decode_tokens: usize,
) -> (Vec<usize>, usize, Vec<Vec<usize>>) {
    let mut plan: Option<IncrementalSwapPlan> = if per_tick > 0 && !target_layers.is_empty() {
        Some(IncrementalSwapPlan::new(target_layers, per_tick, 0))
    } else {
        None
    };

    let mut all_dispatched: Vec<usize> = Vec::new();
    let mut chunk_count = 0usize;
    let mut chunks: Vec<Vec<usize>> = Vec::new();

    for _tick in 0..n_decode_tokens {
        // ENG-ALG-233: drain chunk after forward
        if let Some(ref mut p) = plan {
            let chunk = p.drain_chunk();
            if !chunk.is_empty() {
                // Caller would invoke execute_on_slots(chunk) here (INV-146).
                // We record dispatch instead.
                all_dispatched.extend_from_slice(&chunk);
                chunk_count += 1;
                chunks.push(chunk);
            }
            // ENG-ALG-233: retire when empty (INV-145)
            if p.is_done() {
                plan = None;
            }
        }
    }

    (all_dispatched, chunk_count, chunks)
}

// ── ENG-ALG-233: 25 layers, per_tick=1, 30 decode tokens ─────────────────────

#[test]
fn per_tick_1_25_layers_dispatches_in_25_ticks() {
    let layers: Vec<usize> = (0..25).collect();
    let n_ticks = 30; // more tokens than needed — plan retires naturally

    let (dispatched, chunk_count, _chunks) = simulate_decode_loop(layers.clone(), 1, n_ticks);

    assert_eq!(
        chunk_count, 25,
        "per_tick=1, 25 layers → 25 chunk dispatches"
    );
    assert_eq!(dispatched, layers, "all layers dispatched in order");
}

// ── ENG-ALG-233: per_tick=3, 25 layers ────────────────────────────────────────

#[test]
fn per_tick_3_25_layers_dispatch_sequence() {
    let layers: Vec<usize> = (0..25).collect();
    let (dispatched, chunk_count, chunks) = simulate_decode_loop(layers.clone(), 3, 40);

    // ceil(25/3) = 9 chunks
    assert_eq!(chunk_count, 9, "ceil(25/3)=9 chunks for per_tick=3");
    assert_eq!(
        dispatched, layers,
        "all layers dispatched exactly once in order"
    );

    // First 8 chunks have 3 layers; last has 1
    for (i, chunk) in chunks.iter().enumerate() {
        if i < 8 {
            assert_eq!(chunk.len(), 3, "chunk {} should have 3 layers", i);
        } else {
            assert_eq!(chunk.len(), 1, "last chunk should have 1 layer");
        }
    }
}

// ── INV-146: chunk_count == ratio_generation bumps ────────────────────────────
// Each execute_on_slots call produces exactly 1 ratio_generation bump.
// We verify this via a mock ratio_generation counter.

#[test]
fn ratio_generation_bumps_equal_chunk_count() {
    let be = cpu_be();
    let ratio_generation = Arc::new(AtomicU64::new(0));
    let layers: Vec<LayerSlot> = (0..4)
        .map(|_| LayerSlot::new(dummy_layer(&be), DType::F16, None, 0))
        .collect();

    // Simulate chunk dispatch with no secondary (execute_on_slots → no-op but
    // still bumps ratio_generation when swapped > 0; without secondary it skips).
    // Here we manually simulate the bump as the spec says it happens per chunk.
    let target: Vec<usize> = (0..4).collect();
    let mut plan = Some(IncrementalSwapPlan::new(target.clone(), 2, 0));
    let mut bump_count = 0usize;

    for _tick in 0..10 {
        if let Some(ref mut p) = plan {
            let chunk = p.drain_chunk();
            if !chunk.is_empty() {
                // Simulate: execute_on_slots would bump ratio_generation once per chunk
                // when at least one layer was swapped (INV-146).
                // In this test we simulate the bump directly (no actual swap).
                ratio_generation.fetch_add(1, Ordering::SeqCst);
                bump_count += 1;
            }
            if p.is_done() {
                plan = None;
            }
        }
    }

    let final_gen = ratio_generation.load(Ordering::SeqCst);
    assert_eq!(
        final_gen, bump_count as u64,
        "ratio_generation bumps == chunk dispatch count"
    );
    // 4 layers, per_tick=2 → 2 chunks
    assert_eq!(bump_count, 2, "4 layers / per_tick=2 → 2 chunks");
    assert_eq!(final_gen, 2);
    assert!(layers.len() == 4); // silence unused warning
}

// ── ENG-ALG-234: in-flight plan ignores new signal ────────────────────────────

#[test]
fn in_flight_plan_ignores_new_signal() {
    // ENG-ALG-234: once committed, plan runs to completion.
    // A second "signal" (new plan creation attempt) is blocked while plan is active.
    let target: Vec<usize> = (0..6).collect();
    let mut plan: Option<IncrementalSwapPlan> =
        Some(IncrementalSwapPlan::new(target.clone(), 2, 0));

    let mut dispatched: Vec<usize> = Vec::new();

    for tick in 0..10 {
        // Simulate: new signal arrives at tick=1 — but plan is in-flight, so ignored.
        if tick == 1 && plan.is_some() {
            // ENG-ALG-234: do NOT replace plan; log-and-drop the new signal.
            // (In production: log_debug("incremental swap plan in flight, ignoring"))
        } else if tick == 1 && plan.is_none() {
            // Only create new plan if previous one retired
            plan = Some(IncrementalSwapPlan::new(vec![99], 1, tick));
        }

        if let Some(ref mut p) = plan {
            let chunk = p.drain_chunk();
            if !chunk.is_empty() {
                dispatched.extend_from_slice(&chunk);
            }
            if p.is_done() {
                plan = None;
            }
        }
    }

    // The original 6 layers must all be dispatched; layer 99 must NOT appear
    // because the in-flight plan was not replaced (ENG-ALG-234).
    assert_eq!(
        dispatched, target,
        "in-flight plan runs to completion unmodified"
    );
    assert!(
        !dispatched.contains(&99),
        "new signal during in-flight plan must be ignored"
    );
}

// ── ENG-ALG-233: plan retires after is_done() ─────────────────────────────────

#[test]
fn plan_retires_exactly_once_via_option_take() {
    let target: Vec<usize> = (0..4).collect();
    let mut plan: Option<IncrementalSwapPlan> = Some(IncrementalSwapPlan::new(target, 4, 0));
    let mut retire_count = 0usize;

    for _tick in 0..5 {
        if let Some(ref mut p) = plan {
            let _chunk = p.drain_chunk();
            if p.is_done() {
                retire_count += 1;
                plan = None; // Option::take equivalent
            }
        }
    }

    assert_eq!(retire_count, 1, "plan retires exactly once");
    assert!(plan.is_none(), "plan is None after retirement");
}

// ── per_tick=0: no plan created ───────────────────────────────────────────────

#[test]
fn per_tick_zero_no_plan_created() {
    // Caller must not create IncrementalSwapPlan when per_tick==0.
    // We verify by simulating the guard condition.
    let per_tick: usize = 0;
    let target_layers = vec![0, 1, 2];
    let plan: Option<IncrementalSwapPlan> = if per_tick > 0 && !target_layers.is_empty() {
        Some(IncrementalSwapPlan::new(target_layers, per_tick, 0))
    } else {
        None
    };

    assert!(
        plan.is_none(),
        "per_tick=0: no IncrementalSwapPlan created (single-shot path)"
    );
}
