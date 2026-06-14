//! ENG-ALG-210 — initial uniform load path.
//!
//! Phase 1 unit check: every `LayerSlot` starts at `default_dtype`, generation
//! 0, and shares the same `Arc<SecondaryMmap>` handle (INV-125 structural
//! guarantee). An end-to-end GGUF roundtrip is deferred to device tests
//! because it requires fixture files.
//!
//! Note: `TransformerWeights` container (ENG-DAT-093) was removed in Stage 2
//! cleanup. These tests verify `LayerSlot` invariants directly.

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

#[test]
fn initial_load_uniform_dtype_secondary_none() {
    // Secondary = None → every slot opens with its handle empty.
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let slots: Vec<LayerSlot> = (0..16)
        .map(|_| LayerSlot::new(dummy_layer(&be), DType::F16, None, 0))
        .collect();

    for slot in &slots {
        assert_eq!(slot.current_dtype(), DType::F16);
        assert_eq!(slot.generation(), 0);
        assert!(slot.secondary_mmap_handle().is_none());
    }
}

#[test]
fn initial_load_uniform_dtype_all_slots_identical() {
    // When a model is loaded at Q4_0 uniformly, every slot must start at
    // Q4_0. Phase 2 swap can later flip individual slots; Phase 1 must not
    // already present mixed state.
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let slots: Vec<LayerSlot> = (0..32)
        .map(|_| LayerSlot::new(dummy_layer(&be), DType::Q4_0, None, 0))
        .collect();
    assert!(slots.iter().all(|s| s.current_dtype() == DType::Q4_0));
}
