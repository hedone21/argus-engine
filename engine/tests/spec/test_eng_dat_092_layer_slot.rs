//! ENG-DAT-092 — `LayerSlot` atomic swap primitive.
//!
//! Phase 1 scope: static construction + lock-free snapshot reads. Phase 2
//! will add concurrent reader/writer tests once `SwapExecutor` lands. For now
//! we exercise:
//!   - snapshot identity preservation across multiple `load_weights`
//!   - `current_dtype` / `generation` initial values
//!   - manual `swap_weights` bumps generation and switches dtype (INV-124)
//!   - `store_weights_same_dtype` does NOT bump generation (rewrap semantics)

use std::sync::Arc;

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::layers::transformer_layer::TransformerLayer;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::models::weights::LayerSlot;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;

fn make_tensor(shape: &[usize], dtype: DType, be: &Arc<dyn Backend>) -> Tensor {
    let mem = Galloc::new();
    let numel: usize = shape.iter().product();
    let byte_size = match dtype {
        DType::F32 => numel * 4,
        DType::F16 | DType::BF16 => numel * 2,
        DType::U8 => numel,
        _ => numel * 4,
    };
    let buf = mem.alloc(byte_size, dtype).unwrap();
    Tensor::new(Shape::new(shape.to_vec()), buf, be.clone())
}

fn make_dummy_layer(be: &Arc<dyn Backend>) -> TransformerLayer {
    TransformerLayer {
        wq: make_tensor(&[4, 4], DType::F32, be),
        wk: make_tensor(&[4, 4], DType::F32, be),
        wv: make_tensor(&[4, 4], DType::F32, be),
        wo: make_tensor(&[4, 4], DType::F32, be),
        w_gate: make_tensor(&[4, 4], DType::F32, be),
        w_up: make_tensor(&[4, 4], DType::F32, be),
        w_down: make_tensor(&[4, 4], DType::F32, be),
        attention_norm: make_tensor(&[4], DType::F32, be),
        ffn_norm: make_tensor(&[4], DType::F32, be),
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    }
}

#[test]
fn layer_slot_initial_state_matches_default_dtype() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let layer = make_dummy_layer(&be);
    let slot = LayerSlot::new(layer, DType::F16, None, 0);
    assert_eq!(slot.current_dtype(), DType::F16);
    assert_eq!(slot.generation(), 0);
    assert!(slot.secondary_mmap_handle().is_none());
}

#[test]
fn layer_slot_load_weights_yields_consistent_snapshot() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let layer = make_dummy_layer(&be);
    let slot = LayerSlot::new(layer, DType::F16, None, 0);

    let snap_a = slot.load_weights();
    let snap_b = slot.load_weights();
    // Without a swap, two snapshot loads must point at the same Arc (INV-123
    // atomic semantics, Phase 1 static case).
    assert!(Arc::ptr_eq(&snap_a, &snap_b));
}

#[test]
fn swap_weights_bumps_generation_and_current_dtype() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let layer = make_dummy_layer(&be);
    let slot = LayerSlot::new(layer, DType::F16, None, 0);

    let new_layer = make_dummy_layer(&be);
    let new_arc = Arc::new(new_layer);
    slot.swap_weights(new_arc.clone(), DType::Q4_0);

    // INV-124 postcondition: current_dtype reflects the installed snapshot.
    assert_eq!(slot.current_dtype(), DType::Q4_0);
    assert_eq!(slot.generation(), 1);

    // Subsequent load returns the new Arc.
    let loaded = slot.load_weights();
    assert!(Arc::ptr_eq(&loaded, &new_arc));
}

#[test]
fn store_weights_same_dtype_keeps_generation() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let layer = make_dummy_layer(&be);
    let slot = LayerSlot::new(layer, DType::F16, None, 0);

    let other = Arc::new(make_dummy_layer(&be));
    slot.store_weights_same_dtype(other.clone());

    // Rewrap path (backend migration, partition install) does not bump the
    // generation — it represents an internal buffer re-layout, not a dtype
    // transition. Downstream plan invalidation uses the global
    // `ratio_generation` counter instead.
    assert_eq!(slot.generation(), 0);
    assert_eq!(slot.current_dtype(), DType::F16);
    assert!(Arc::ptr_eq(&slot.load_weights(), &other));
}

#[test]
fn rcu_weights_clone_and_install() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let layer = make_dummy_layer(&be);
    let slot = LayerSlot::new(layer, DType::F16, None, 0);

    // Observe wq tensor size before the update.
    let pre = slot.load_weights();
    let pre_size = pre.wq.size();

    // RCU: rebuild layer with a new attention_norm shape (conceptually a
    // no-op on numerical semantics for this unit test — we just verify the
    // closure path works and the slot takes on the mutated state).
    slot.rcu_weights(|old| {
        let mut clone = old.clone();
        clone.attention_norm = make_tensor(&[8], DType::F32, &be);
        clone
    });

    let post = slot.load_weights();
    assert_eq!(post.attention_norm.shape().dims(), &[8]);
    // Untouched fields preserved.
    assert_eq!(post.wq.size(), pre_size);
    // RCU does not change the generation counter in Phase 1 (a partition
    // re-install has its own `ratio_generation` counter, leaving the slot
    // generation reserved for Phase 2 SwapExecutor events).
    assert_eq!(slot.generation(), 0);
}
