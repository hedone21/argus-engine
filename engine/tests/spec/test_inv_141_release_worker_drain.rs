//! INV-141 — `PrimaryReleaseWorker` drain contract (ENG-ALG-228 / ENG-DAT-100).
//!
//! Tests:
//! 1. Normal spawn → enqueue → drain → pending==0.
//! 2. Drain timeout simulation — artificial pending bump, drain fails.
//! 3. Worker thread join on drop (no hang).
//! 4. Multiple enqueues all drained within deadline.
//! 5. `SwapExecutor` rejects batch when INV-141 would be violated
//!    (`SwapError::ReleaseDrainTimeout`).
//!
//! Spec: ENG-ALG-228, ENG-DAT-100, INV-141.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::layers::transformer_layer::TransformerLayer;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use argus_engine::weight::PrimaryReleaseWorker;

// ── Helpers ────────────────────────────────────────────────────────────────

fn cpu_be() -> Arc<dyn Backend> {
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

// ── Tests ──────────────────────────────────────────────────────────────────

/// INV-141 basic contract: enqueue → drain → pending == 0.
#[test]
fn inv_141_enqueue_drain_pending_zero() {
    let be = cpu_be();
    let worker = PrimaryReleaseWorker::spawn(be.clone());

    assert_eq!(worker.pending_count(), 0, "initial pending must be 0");

    worker.enqueue_release(make_layer(&be));
    // pending may be 0 or 1 here (race); drain resolves it.
    worker
        .drain(Duration::from_millis(500))
        .expect("drain should complete within 500ms for a CPU drop");

    assert_eq!(
        worker.pending_count(),
        0,
        "INV-141: pending must be 0 after drain"
    );
}

/// Multiple layers: all enqueues drain before deadline.
#[test]
fn inv_141_multiple_layers_drained() {
    let be = cpu_be();
    let worker = PrimaryReleaseWorker::spawn(be.clone());

    for _ in 0..8 {
        worker.enqueue_release(make_layer(&be));
    }

    worker
        .drain(Duration::from_millis(2000))
        .expect("drain should succeed for 8 layers");

    assert_eq!(
        worker.pending_count(),
        0,
        "all 8 layers must be released after drain"
    );
}

/// Drain on an already-empty queue must be Ok immediately.
#[test]
fn inv_141_drain_empty_is_ok() {
    let be = cpu_be();
    let worker = PrimaryReleaseWorker::spawn(be.clone());

    worker
        .drain(Duration::from_millis(10))
        .expect("drain on empty queue must return Ok");
}

/// Drain timeout simulation — artificially inflate pending without sending a
/// real job so the worker can never decrement it. Verifies that drain returns
/// Err after the deadline expires (INV-141 enforcement).
#[test]
fn inv_141_drain_timeout_returns_error() {
    let be = cpu_be();
    let worker = PrimaryReleaseWorker::spawn(be.clone());

    // Artificially bump pending without sending a real ReleaseJob.
    // The worker will never decrement this, so drain must time out.
    worker.pending.fetch_add(1, Ordering::Release);

    let result = worker.drain(Duration::from_millis(20));
    assert!(
        result.is_err(),
        "INV-141: drain must fail when pending > 0 after deadline"
    );

    let err = result.unwrap_err();
    assert!(
        err.pending > 0,
        "DrainError.pending must be non-zero on timeout"
    );
    assert_eq!(
        err.timeout_ms, 20,
        "DrainError.timeout_ms must match the requested deadline"
    );

    // Restore balance so drop/join does not block.
    worker.pending.fetch_sub(1, Ordering::Release);
}

/// Drop impl joins the worker thread (no hang).
///
/// If the join blocks indefinitely, the test harness will eventually kill the
/// process; the test succeeds if control returns from `drop(worker)`.
#[test]
fn inv_141_drop_joins_worker() {
    let be = cpu_be();
    let worker = PrimaryReleaseWorker::spawn(be.clone());
    worker.enqueue_release(make_layer(&be));
    // Drop must send Shutdown and join. Reaching the end of this function
    // implies the join completed within the test timeout.
    drop(worker);
}

/// Token-boundary race simulation: enqueue while a caller pretends to hold
/// an Arc snapshot. Since `PrimaryReleaseWorker::enqueue_release` takes an
/// owned `LayerWeights` (not an Arc), this scenario maps to: caller succeeds
/// `Arc::try_unwrap`, sends the owned value, worker drops it. Verifies that
/// pending reaches 0 even when there is concurrent "reader" activity.
#[test]
fn inv_141_concurrent_reader_simulation() {
    use std::sync::{Arc as StdArc, Barrier};
    use std::thread;

    let be = cpu_be();
    let worker = StdArc::new(PrimaryReleaseWorker::spawn(be.clone()));
    let barrier = StdArc::new(Barrier::new(2));

    // Spawn a reader thread that waits at the barrier (simulating a forward
    // pass that has already captured a snapshot but hasn't finished yet).
    let barrier2 = StdArc::clone(&barrier);
    let reader = thread::spawn(move || {
        // "Reader arrives" — simulates forward pass entry.
        barrier2.wait();
        // Simulate some compute time.
        std::hint::black_box(42u64.wrapping_mul(0xdeadbeef));
    });

    // Enqueue a layer while the reader is active.
    worker.enqueue_release(make_layer(&be));
    barrier.wait(); // let the reader proceed

    // Wait for reader to finish (ensuring the simulated forward pass ends).
    reader.join().expect("reader thread should not panic");

    // Now drain — the worker should complete independently of the reader.
    worker
        .drain(Duration::from_millis(500))
        .expect("drain should succeed even with concurrent reader simulation");

    assert_eq!(worker.pending_count(), 0);
}

/// INV-141 `SwapExecutor` enforcement: when pending > 0 at batch start,
/// `execute_on_slots` must return `SwapError::ReleaseDrainTimeout` after the
/// internal drain deadline expires.
///
/// This test injects an artificial pending count by bypassing `enqueue_release`
/// (so no real job is sent to the worker), verifying that the executor rejects
/// the batch rather than proceeding with leaking memory.
#[test]
fn inv_141_swap_executor_rejects_on_drain_timeout() {
    use std::sync::atomic::AtomicU64;

    use argus_engine::memory::galloc::Galloc;
    use argus_engine::model_config::{ModelArch, ModelConfig};
    use argus_engine::models::weights::LayerSlot;
    use argus_engine::weight::{SwapError, SwapExecutor};

    fn minimal_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 64,
            num_hidden_layers: 2,
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

    let be = cpu_be();
    let worker_concrete = Arc::new(PrimaryReleaseWorker::spawn(be.clone()));
    // Keep a concrete handle for the `pending` field injection below, and
    // hand a trait-object clone to the executor (which expects the cross-cutting
    // `ReleaseWorkerAccess` trait, §13.8-O 본질 해소 sprint).
    let worker: Arc<dyn argus_engine::runtime_resources_access::ReleaseWorkerAccess> =
        worker_concrete.clone();

    // Inject an artificial pending to force drain timeout.
    worker_concrete.pending.fetch_add(1, Ordering::Release);

    let layers: Vec<Arc<LayerSlot>> = (0..2)
        .map(|_| Arc::new(LayerSlot::new(make_layer(&be), DType::F16, None, 0)))
        .collect();
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let config = minimal_config();
    let memory = Galloc::new();

    let executor = SwapExecutor::new_with_worker(
        DType::Q4_0,
        &config,
        be.clone(),
        &memory,
        Arc::clone(&worker),
    );

    // Execute with no secondary mmap — normally would be a no-op, but
    // INV-141 check happens before the secondary-absent guard.
    let result = executor.execute_on_slots(&layers, None, &ratio_gen, &[0, 1], None);

    assert!(
        matches!(result, Err(SwapError::ReleaseDrainTimeout { .. })),
        "executor must return ReleaseDrainTimeout when INV-141 is violated, got: {:?}",
        result.map(|_| ())
    );

    // Restore balance so drop/join does not block.
    worker_concrete.pending.fetch_sub(1, Ordering::Release);
}
