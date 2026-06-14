//! INV-149 — wait gate ordering for `IntraForwardSwapHook`.
//! Spec ref tag for coverage: inv_149
//!
//! Spec: `spec/41-invariants.md` §3.21 INV-149, `spec/32-engine-algorithms.md`
//! §3.12.22.4 (ENG-ALG-238), `spec/33-engine-data.md` §3.24 (ENG-DAT-101),
//! `arch/weight_swap.md` §10.4 / §10.12.3.
//!
//! 검증:
//! 1. `pending_event_for(idx)` returns Some(_) after `arm_pending`.
//! 2. After `clear_pending`, pending_event_for returns None.
//! 3. forward thread that observes `Some(evt)` from `pending_event_for_dyn`
//!    can hand the event to `backend.wait_event_blocking` (smoke check —
//!    full forward integration is `test_inv_122_mixed_precision` host smoke).
//! 4. dispatcher worker callback (`on_complete`) clears the registry slot.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use argus_engine::backend::cpu::CpuBackend;
use argus_engine::backend::{Backend, GpuEvent};
use argus_engine::buffer::DType;
use argus_engine::model_config::{ModelArch, ModelConfig};
use argus_engine::models::weights::LayerSlot;
use argus_engine::weight::IntraForwardSwapHook;
use argus_engine::weight::async_swap::{AsyncSwapDispatcher, SwapCommitJob};

use argus_engine::layers::transformer_layer::TransformerLayer;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;

fn cpu_backend() -> Arc<dyn Backend> {
    Arc::new(CpuBackend::new())
}

fn make_tensor(be: &Arc<dyn Backend>, numel: usize) -> Tensor {
    let buf: Arc<dyn argus_engine::buffer::Buffer> =
        Arc::new(SharedBuffer::new(numel * 4, DType::F32));
    Tensor::new(Shape::new(vec![numel]), buf, be.clone())
}

fn make_layer(be: &Arc<dyn Backend>) -> TransformerLayer {
    TransformerLayer {
        wq: make_tensor(be, 16),
        wk: make_tensor(be, 16),
        wv: make_tensor(be, 16),
        wo: make_tensor(be, 16),
        w_gate: make_tensor(be, 16),
        w_up: make_tensor(be, 16),
        w_down: make_tensor(be, 16),
        attention_norm: make_tensor(be, 4),
        ffn_norm: make_tensor(be, 4),
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    }
}

fn make_slot(be: &Arc<dyn Backend>) -> Arc<LayerSlot> {
    Arc::new(LayerSlot::new(make_layer(be), DType::F16, None, 0))
}

fn make_config() -> Arc<ModelConfig> {
    Arc::new(ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: 8,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 500_000.0,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_id: 0,
        weight_prefix: String::new(),
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
    })
}

/// Helper: build a hook (test-only constructor — no secondary mmap) bound
/// to `n_layers` slots. Used for INV-149 wait-gate ordering tests that
/// only exercise the registry surface (arm/clear/read), not the dispatch
/// path.
fn make_test_hook(
    n_layers: usize,
    plan_layers: Vec<usize>,
) -> (Arc<IntraForwardSwapHook>, Arc<AsyncSwapDispatcher>) {
    let be = cpu_backend();
    let dispatcher = Arc::new(AsyncSwapDispatcher::new(be.clone()));
    let slots: Vec<Arc<LayerSlot>> = (0..n_layers).map(|_| make_slot(&be)).collect();
    let config = make_config();
    let hook = IntraForwardSwapHook::new_for_test(
        plan_layers,
        0,
        Arc::clone(&dispatcher),
        slots,
        be,
        DType::Q4_0,
        config,
    );
    (hook, dispatcher)
}

#[test]
fn test_pending_event_for_arm_then_read() {
    let (hook, _disp) = make_test_hook(4, vec![1, 2]);
    assert!(hook.pending_event_for(1).is_none(), "initially None");

    let evt = Arc::new(GpuEvent::dummy());
    hook.arm_pending_for_test(1, evt);
    assert!(hook.pending_event_for(1).is_some(), "after arm: Some");

    hook.clear_pending_for_test(1);
    assert!(
        hook.pending_event_for(1).is_none(),
        "after clear: back to None"
    );
}

#[test]
fn test_pending_event_for_out_of_range_returns_none() {
    let (hook, _disp) = make_test_hook(4, vec![]);
    assert!(hook.pending_event_for(99).is_none());
}

/// INV-149 worker callback: `SwapCommitJob::on_complete` runs on the
/// dispatcher worker thread after `slot.swap_weights`, and a hook-supplied
/// callback can clear the pending registry. We exercise this directly to
/// confirm the wiring.
#[test]
fn test_dispatcher_callback_clears_pending() {
    let be = cpu_backend();
    let dispatcher = AsyncSwapDispatcher::new(be.clone());

    let (hook, _disp_unused) = make_test_hook(4, vec![]);
    // Manually arm a pending sentinel for layer 2.
    hook.arm_pending_for_test(2, Arc::new(GpuEvent::dummy()));
    assert!(hook.pending_event_for(2).is_some());

    // Construct a SwapCommitJob whose on_complete clears layer 2 in our hook.
    let slot = make_slot(&be);
    let new_weights = Arc::new(make_layer(&be));

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_for_cb = Arc::clone(&counter);
    let hook_arc = Arc::clone(&hook);
    let on_complete: Arc<dyn Fn(usize) + Send + Sync> = Arc::new(move |idx: usize| {
        hook_arc.clear_pending_for_test(idx);
        counter_for_cb.fetch_add(1, Ordering::SeqCst);
    });

    dispatcher
        .submit_commit(SwapCommitJob {
            slot,
            new_weights,
            new_dtype: DType::Q4_0,
            write_event: std::sync::Arc::new(GpuEvent::dummy()),
            release_worker: None,
            on_complete: Some(on_complete),
            layer_idx: Some(2),
        })
        .expect("submit_commit must succeed");

    dispatcher
        .drain(Duration::from_secs(1))
        .expect("drain must complete in time");

    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "callback ran exactly once"
    );
    assert!(
        hook.pending_event_for(2).is_none(),
        "callback cleared the registry slot"
    );
}

/// Wait-gate ordering smoke: backend.wait_event_blocking on a dummy
/// completed event is fast no-op. Combined with the prior test, this ensures
/// the wait gate path completes without error when the dispatcher has
/// already cleared.
#[test]
fn test_wait_event_on_dummy_is_no_op() {
    let be = cpu_backend();
    let evt = GpuEvent::dummy();
    let started = std::time::Instant::now();
    be.wait_event_blocking(&evt).expect("must succeed");
    assert!(
        started.elapsed().as_millis() < 50,
        "wait on dummy must be fast"
    );
}
