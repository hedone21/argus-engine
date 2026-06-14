//! INV-144 — Forward snapshot consistency during incremental swap.
//!
//! Spec: `spec/41-invariants.md` §3.20 (INV-144)
//! Spec: `spec/32-engine-algorithms.md` §3.12.21.2 (ENG-ALG-233)
//!
//! INV-144: During incremental swap, a single forward pass uses the ArcSwap
//! snapshot set captured at token boundary. Even if a chunk swap occurs between
//! tokens T and T+1, token T's forward reads the weights at time T (not time T+1).
//!
//! This test verifies INV-121 preservation under incremental swap:
//! - Thread A reads a LayerSlot snapshot (load_weights) → holds Arc
//! - Thread B (simulating incremental swap) stores a new LayerWeights via swap_weights
//! - Thread A still sees the original snapshot for the duration of its "forward"
//!
//! This is the same guarantee as INV-121/123 but phrased for the incremental case.
//! See test_inv_123_dynamic.rs for the concurrent reader/writer test.

use std::sync::Arc;

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::layers::transformer_layer::TransformerLayer;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::models::weights::{LayerSlot, LayerWeights};
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use argus_engine::weight::IncrementalSwapPlan;

fn cpu_be() -> Arc<dyn Backend> {
    Arc::new(CpuBackend::new())
}

/// Build a tensor with a known generation tag embedded as the first f32 element.
fn tagged_tensor(be: &Arc<dyn Backend>, g_id: u64) -> Tensor {
    let mem = Galloc::new();
    let buf = mem.alloc(16, DType::F32).unwrap();
    let mut t = Tensor::new(Shape::new(vec![4]), buf, be.clone());
    t.as_mut_slice::<f32>()[0] = g_id as f32;
    t
}

fn make_layer(be: &Arc<dyn Backend>, g_id: u64) -> TransformerLayer {
    TransformerLayer {
        wq: tagged_tensor(be, g_id),
        wk: tagged_tensor(be, g_id),
        wv: tagged_tensor(be, g_id),
        wo: tagged_tensor(be, g_id),
        w_gate: tagged_tensor(be, g_id),
        w_up: tagged_tensor(be, g_id),
        w_down: tagged_tensor(be, g_id),
        attention_norm: tagged_tensor(be, g_id),
        ffn_norm: tagged_tensor(be, g_id),
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    }
}

fn read_tag(t: &Tensor) -> u64 {
    t.as_slice::<f32>()[0] as u64
}

fn snapshot_generation(arc: &Arc<LayerWeights>) -> u64 {
    read_tag(&arc.wq)
}

/// INV-144: snapshot captured before a chunk swap remains valid after the swap.
#[test]
fn snapshot_captured_before_swap_survives_incremental_chunk() {
    let be = cpu_be();

    // Build LayerSlot with generation=1 weights
    let gen1_layer = make_layer(&be, 1);
    let slot = LayerSlot::new(gen1_layer, DType::F16, None, 0);

    // Simulate token T: capture snapshot (INV-121 / INV-144)
    let snapshot_before_swap = slot.load_weights();
    assert_eq!(
        snapshot_generation(&snapshot_before_swap),
        1,
        "snapshot before swap should be gen=1"
    );

    // Simulate incremental swap: SwapExecutor installs gen=2 weights
    // (This simulates one chunk of an IncrementalSwapPlan being executed)
    let gen2_layer = make_layer(&be, 2);
    let _old = slot.swap_weights(Arc::new(gen2_layer), DType::F16);

    // After swap, slot now holds gen=2
    let snapshot_after_swap = slot.load_weights();
    assert_eq!(
        snapshot_generation(&snapshot_after_swap),
        2,
        "new snapshot should be gen=2"
    );

    // INV-144: the snapshot captured at token T still reads gen=1
    // (the Arc prevents the old LayerWeights from being dropped)
    assert_eq!(
        snapshot_generation(&snapshot_before_swap),
        1,
        "INV-144: token T snapshot must remain gen=1 even after incremental swap"
    );
}

/// INV-144: multiple chunk swaps between tokens preserve each token's snapshot.
#[test]
fn multiple_chunks_each_token_snapshot_is_isolated() {
    let be = cpu_be();

    // Build 4 slots
    let n = 4usize;
    let slots: Vec<LayerSlot> = (0..n)
        .map(|i| LayerSlot::new(make_layer(&be, i as u64), DType::F16, None, 0))
        .collect();

    // Token T: capture all snapshots
    let snapshots_t: Vec<Arc<LayerWeights>> = slots.iter().map(|s| s.load_weights()).collect();

    // Verify initial state
    for (i, snap) in snapshots_t.iter().enumerate() {
        assert_eq!(snapshot_generation(snap), i as u64);
    }

    // Simulate incremental plan: swap layers [0, 1] (chunk 1 of 2)
    let mut plan = IncrementalSwapPlan::new(vec![0, 1, 2, 3], 2, 0);
    let chunk1 = plan.drain_chunk(); // [0, 1]
    assert_eq!(chunk1, vec![0, 1]);

    for &layer_idx in &chunk1 {
        let new_layer = make_layer(&be, (layer_idx + 100) as u64); // gen = 100+i
        let _old = slots[layer_idx].swap_weights(Arc::new(new_layer), DType::F16);
    }

    // Token T's snapshots still see original gen values (INV-144)
    for (i, snap) in snapshots_t.iter().enumerate() {
        assert_eq!(
            snapshot_generation(snap),
            i as u64,
            "token T snapshot for slot {} must be unaffected by chunk1 swap",
            i
        );
    }

    // Token T+1: new snapshots see the updated slots
    let snapshots_t1: Vec<Arc<LayerWeights>> = slots.iter().map(|s| s.load_weights()).collect();
    assert_eq!(snapshot_generation(&snapshots_t1[0]), 100); // swapped
    assert_eq!(snapshot_generation(&snapshots_t1[1]), 101); // swapped
    assert_eq!(snapshot_generation(&snapshots_t1[2]), 2); // not yet swapped
    assert_eq!(snapshot_generation(&snapshots_t1[3]), 3); // not yet swapped
}

/// INV-144: plan completion doesn't affect already-captured snapshots.
#[test]
fn plan_completion_does_not_invalidate_prior_snapshots() {
    let be = cpu_be();
    let slot = LayerSlot::new(make_layer(&be, 1), DType::F16, None, 0);

    // Capture snapshot at token T
    let snap_t = slot.load_weights();
    assert_eq!(snapshot_generation(&snap_t), 1);

    // Run through a complete IncrementalSwapPlan (all chunks)
    let mut plan = IncrementalSwapPlan::new(vec![0], 1, 0);
    let chunk = plan.drain_chunk();
    assert!(!chunk.is_empty());

    // Simulate the swap for this layer (using slot directly)
    let new_lw = Arc::new(make_layer(&be, 2));
    let _old = slot.swap_weights(new_lw, DType::F16);
    assert!(plan.is_done());

    // snapshot_t still valid
    assert_eq!(
        snapshot_generation(&snap_t),
        1,
        "INV-144: snapshot captured before plan still valid after plan completion"
    );

    // New snapshot reflects completed swap
    let snap_after = slot.load_weights();
    assert_eq!(snapshot_generation(&snap_after), 2);
}
