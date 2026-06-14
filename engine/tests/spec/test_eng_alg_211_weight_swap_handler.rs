//! ENG-ALG-211 Stage 3 — `WeightSwapHandler` direct-dispatch interface.
//!
//! Phase 3 refactoring (ENG-ALG-214-ROUTE "1안"): `CachePressureHandler` trait
//! impl was removed.  `WeightSwapHandler` is now a thin orchestrator that
//! exposes `execute_swap(&[usize]) -> Result<ActionResult>` for use by
//! `dispatch_swap_weights` in `generate.rs`.
//!
//! Tests cover:
//! - No-op when target_layers is empty.
//! - No-op when secondary_mmap is absent (ENG-DAT-C09).
//! - No-op when zero-layer model + non-empty target_layers.
//! - `ActionResult::WeightSwapped` is recognised as an action.
//! - `SwapExecutor::uniform_target_layers` contract (ENG-ALG-212/213).
//!
//! Full execute() path (layer slot + secondary mmap fixture) requires
//! real GGUF fixture files → deferred to device integration tests.
//!
//! Spec: ENG-ALG-211, ENG-ALG-212, ENG-ALG-214-ROUTE, INV-121/123, WSWAP-2-HANDLER.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::kv::ActionResult;
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::layers::transformer_layer::TransformerLayer;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::model_config::{ModelArch, ModelConfig};
use argus_engine::models::weights::LayerSlot;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use argus_engine::weight::weight_swap_handler::{WeightSwapHandler, WeightSwapModelRef};

// ── Helpers ───────────────────────────────────────────────────────────────────

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

fn minimal_model_config() -> ModelConfig {
    ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 16,
        intermediate_size: 128,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_id: 1,
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
        weight_prefix: String::new(),
    }
}

/// Build a `WeightSwapModelRef` with no secondary mmap (swap disabled).
fn make_model_ref_no_secondary(n_layers: usize) -> Arc<WeightSwapModelRef> {
    let be = cpu_be();
    let layers: Vec<Arc<LayerSlot>> = (0..n_layers)
        .map(|_| Arc::new(LayerSlot::new(dummy_layer(&be), DType::F16, None, 0)))
        .collect();
    Arc::new(WeightSwapModelRef {
        layers: Arc::new(layers),
        secondary_mmap: None,
        ratio_generation: Arc::new(AtomicU64::new(0)),
        config: minimal_model_config(),
        backend: be,
    })
}

fn make_kv_caches(n: usize) -> Vec<KVCache> {
    let be = cpu_be();
    (0..n)
        .map(|_| {
            let buf_size = 32 * 4 * 4; // 32 positions, 1 head, dim=4, f32
            let k = Tensor::new(
                Shape::new(vec![1, 32, 1, 4]),
                Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                be.clone(),
            );
            let v = Tensor::new(
                Shape::new(vec![1, 32, 1, 4]),
                Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                be.clone(),
            );
            KVCache::new(k, v, 32)
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// execute_swap with empty target_layers → NoOp (ENG-ALG-214-ROUTE: no-op when needed=0).
#[test]
fn execute_swap_empty_target_is_noop() {
    let model_ref = make_model_ref_no_secondary(4);
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);
    let result = handler.execute_swap(&[]).unwrap();
    assert!(
        matches!(result, ActionResult::NoOp),
        "Expected NoOp for empty target_layers, got {result:?}"
    );
}

/// No secondary mmap → NoOp regardless of target_layers (ENG-DAT-C09).
#[test]
fn no_secondary_mmap_is_noop() {
    let model_ref = make_model_ref_no_secondary(4);
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);
    let result = handler.execute_swap(&[0, 1, 2]).unwrap();
    assert!(
        matches!(result, ActionResult::NoOp),
        "Expected NoOp (no secondary mmap), got {result:?}"
    );
}

/// Zero-layer model + non-empty target_layers → NoOp (secondary also absent).
#[test]
fn zero_layer_model_is_noop() {
    let be = cpu_be();
    let model_ref = Arc::new(WeightSwapModelRef {
        layers: Arc::new(Vec::new()),
        secondary_mmap: None,
        ratio_generation: Arc::new(AtomicU64::new(0)),
        config: minimal_model_config(),
        backend: be,
    });
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);
    let result = handler.execute_swap(&[0]).unwrap();
    assert!(
        matches!(result, ActionResult::NoOp),
        "Zero-layer model must return NoOp"
    );
}

/// `WeightSwapped` `ActionResult` variant is recognised as an action.
#[test]
fn weight_swapped_is_action() {
    let result = ActionResult::WeightSwapped {
        layers_changed: 2,
        freed_bytes: 1024,
        duration_ms: 15.0,
    };
    assert!(
        result.is_action(),
        "WeightSwapped must be is_action() == true"
    );
}

/// `WeightSwapped` variant carries correct fields.
#[test]
fn weight_swapped_fields_roundtrip() {
    let result = ActionResult::WeightSwapped {
        layers_changed: 3,
        freed_bytes: 2048,
        duration_ms: 7.5,
    };
    match result {
        ActionResult::WeightSwapped {
            layers_changed,
            freed_bytes,
            duration_ms,
        } => {
            assert_eq!(layers_changed, 3);
            assert_eq!(freed_bytes, 2048);
            assert!((duration_ms - 7.5).abs() < 1e-6);
        }
        other => panic!("Unexpected variant: {other:?}"),
    }
}

/// `uniform_target_layers` contract (ENG-ALG-212): idempotent, deduplicated,
/// bounded by ratio.
#[test]
fn uniform_target_layers_contract() {
    use argus_engine::weight::SwapExecutor;

    assert!(SwapExecutor::uniform_target_layers(0.0, 16).is_empty());
    assert_eq!(SwapExecutor::uniform_target_layers(1.0, 16).len(), 16);
    let half = SwapExecutor::uniform_target_layers(0.5, 16);
    assert_eq!(half.len(), 8);
    // No duplicates.
    let mut sorted = half.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        half.len(),
        "uniform_target_layers must be deduped"
    );
    // All indices in range.
    assert!(half.iter().all(|&i| i < 16));
}

/// Handler does not require kv_caches (pipeline path removed in Phase 3).
/// Verifies execute_swap() signature accepts only target_layers, not HandlerContext.
#[test]
fn handler_api_is_direct_dispatch() {
    let model_ref = make_model_ref_no_secondary(4);
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);

    // Direct dispatch: no HandlerContext, no PressureLevel needed.
    // This verifies the Phase 3 "1안" contract that the handler is called
    // via dispatch_swap_weights() in generate.rs, NOT via pipeline.
    let _caches = make_kv_caches(1); // not used — proves API doesn't need them
    let result = handler.execute_swap(&[]).unwrap();
    assert!(matches!(result, ActionResult::NoOp));
}
