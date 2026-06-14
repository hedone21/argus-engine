//! `PrimaryReleaseWorker` — asynchronous primary cl_mem drop worker
//! (ENG-ALG-228 / ENG-DAT-100).
//!
//! Moves `Arc::try_unwrap` + `release_primary_weights` out of the critical
//! path so `clReleaseMemObject` (≈ 1 ms × 7 tensors × 25 layers on Adreno)
//! does not block the swap dispatcher.
//!
//! # Contract
//!
//! - [`PrimaryReleaseWorker::enqueue_release`] is called from the swap
//!   dispatcher with a uniquely-owned `LayerWeights` value (caller has already
//!   succeeded `Arc::try_unwrap`).
//! - The background thread drops the value, decrementing `pending` on
//!   completion.
//! - Before the *next* swap batch starts, [`SwapExecutor::execute_on_slots`]
//!   calls [`PrimaryReleaseWorker::drain`] (INV-141) and rejects the batch on
//!   timeout to prevent memory leaks.
//!
//! # Invariants
//!
//! - INV-141: `pending_count() == 0` before a new swap batch.
//! - Drop impl: sends `Shutdown` and joins the worker thread so all
//!   destructors run before the process exits.
//!
//! Spec: ENG-ALG-228, ENG-DAT-100, INV-141.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::backend::Backend;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure worker → inference weight resource (LayerWeights)
use crate::models::weights::LayerWeights;
use crate::runtime_resources_access::{DrainError, ReleaseWorkerAccess};
use crate::weight::swap_executor::record_swap_release_pub;

/// Message sent to the release worker.
pub enum ReleaseJob {
    /// Drop the layer weights on the worker thread.
    ///
    /// Boxed to avoid a large-size-difference-between-variants clippy warning:
    /// `LayerWeights` (= `TransformerLayer`) is a large struct and `Shutdown`
    /// carries nothing, so the enum variant is heap-allocated. The allocation
    /// cost is negligible relative to the GPU buffer release we are deferring.
    Layer(Box<LayerWeights>),
    /// Terminate the worker loop.
    Shutdown,
}

/// Asynchronous primary cl_mem drop worker (ENG-DAT-100).
///
/// Owns a background thread that receives displaced `LayerWeights` via an
/// `mpsc` channel and drops them, triggering `clReleaseMemObject` outside
/// the swap critical path.
pub struct PrimaryReleaseWorker {
    sender: std::sync::mpsc::Sender<ReleaseJob>,
    /// Number of drop jobs in flight. Decremented after each drop completes.
    /// Exposed as `pub` so spec tests can inject artificial counts for
    /// drain-timeout scenarios (INV-141 verification).
    pub pending: Arc<AtomicUsize>,
    /// Join handle. `Some` until `drop` consumes it.
    handle: Option<JoinHandle<()>>,
}

impl PrimaryReleaseWorker {
    /// Spawn the background worker thread.
    ///
    /// The worker retains a clone of `backend` for diagnostic calls inside
    /// [`record_swap_release_pub`] so the caller does not need to keep the
    /// backend alive separately.
    pub fn spawn(backend: Arc<dyn Backend>) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<ReleaseJob>();
        let pending = Arc::new(AtomicUsize::new(0));
        let pending_worker = Arc::clone(&pending);

        let handle = thread::Builder::new()
            .name("llmrs-release".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    match job {
                        ReleaseJob::Layer(layer) => {
                            // Tally before drop so we can record bytes.
                            let (count, bytes) = tally_layer_bytes(&layer);
                            // Drop fires clReleaseMemObject destructors.
                            drop(*layer);
                            // Diagnostic hook (no-op on CPU/CUDA/non-diag builds).
                            record_swap_release_pub(&backend, count, bytes);
                            pending_worker.fetch_sub(1, Ordering::Release);
                        }
                        ReleaseJob::Shutdown => break,
                    }
                }
            })
            .expect("failed to spawn PrimaryReleaseWorker thread");

        Self {
            sender,
            pending,
            handle: Some(handle),
        }
    }

    /// Enqueue a `LayerWeights` for asynchronous drop.
    ///
    /// `pending` is incremented before the send so the caller can rely on
    /// `pending_count() > 0` being observable immediately after this call.
    ///
    /// # Panics
    ///
    /// Panics if the worker thread has already shut down (send error), which
    /// should not happen in normal operation because the worker outlives all
    /// callers.
    pub fn enqueue_release(&self, layer: LayerWeights) {
        self.pending.fetch_add(1, Ordering::Release);
        // If send fails (worker already dead), decrement and log — do not panic
        // in production to preserve correctness of the swap path.
        if self
            .sender
            .send(ReleaseJob::Layer(Box::new(layer)))
            .is_err()
        {
            self.pending.fetch_sub(1, Ordering::Release);
            eprintln!("[PrimaryReleaseWorker] WARNING: worker is dead, layer dropped inline");
        }
    }

    /// Number of drop jobs still in flight.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    /// Block until all pending jobs complete or `deadline` elapses.
    ///
    /// Polls at 1 ms intervals. Returns `Ok(())` when `pending == 0`.
    /// Returns `Err` when `deadline` expires with remaining jobs.
    ///
    /// Used by [`SwapExecutor::execute_on_slots`] to enforce INV-141 before
    /// starting a new swap batch.
    pub fn drain(&self, deadline: Duration) -> Result<(), DrainError> {
        let start = Instant::now();
        loop {
            let p = self.pending.load(Ordering::Acquire);
            if p == 0 {
                return Ok(());
            }
            if start.elapsed() >= deadline {
                let remaining = self.pending.load(Ordering::Acquire);
                return Err(DrainError {
                    pending: remaining,
                    timeout_ms: deadline.as_millis() as u64,
                });
            }
            thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for PrimaryReleaseWorker {
    fn drop(&mut self) {
        // Best-effort shutdown: ignore send error if the channel is already
        // closed (worker panicked or was never started).
        let _ = self.sender.send(ReleaseJob::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Cross-cutting trait impl: lets `TransformerModel` hold an
/// `Arc<dyn ReleaseWorkerAccess>` instead of the concrete worker type
/// (§13.8-O 본질 해소). Forwards every method to the inherent impl above.
impl ReleaseWorkerAccess for PrimaryReleaseWorker {
    fn enqueue_release(&self, layer: LayerWeights) {
        PrimaryReleaseWorker::enqueue_release(self, layer);
    }

    fn pending_count(&self) -> usize {
        PrimaryReleaseWorker::pending_count(self)
    }

    fn drain(&self, deadline: Duration) -> Result<(), DrainError> {
        PrimaryReleaseWorker::drain(self, deadline)
    }
}

/// Tally the byte sizes of all tensors in a `LayerWeights` before dropping.
///
/// Returns `(count, total_bytes)` so the diagnostic hook can record a
/// meaningful footprint even after the layer has been dropped.
fn tally_layer_bytes(layer: &LayerWeights) -> (usize, usize) {
    let mut bytes = 0usize;
    let mut count = 0usize;
    let mut tally = |t: &crate::tensor::Tensor| {
        bytes += t.size();
        count += 1;
    };
    tally(&layer.wq);
    tally(&layer.wk);
    tally(&layer.wv);
    tally(&layer.wo);
    tally(&layer.w_gate);
    tally(&layer.w_up);
    tally(&layer.w_down);
    tally(&layer.attention_norm);
    tally(&layer.ffn_norm);
    if let Some(bias) = &layer.qkv_bias {
        tally(&bias.bq);
        tally(&bias.bk);
        tally(&bias.bv);
    }
    if let Some(t) = &layer.q_norm {
        tally(t);
    }
    if let Some(t) = &layer.k_norm {
        tally(t);
    }
    if let Some(t) = &layer.pre_ffn_norm {
        tally(t);
    }
    if let Some(t) = &layer.post_ffn_norm {
        tally(t);
    }
    (count, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::layers::transformer_layer::TransformerLayer;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn cpu_be() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    fn make_tensor(be: &Arc<dyn Backend>, numel: usize) -> Tensor {
        let buf: Arc<dyn crate::buffer::Buffer> =
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

    #[test]
    fn spawn_enqueue_drain_drop_cycle() {
        let be = cpu_be();
        let worker = PrimaryReleaseWorker::spawn(be.clone());

        assert_eq!(worker.pending_count(), 0);

        let layer = make_layer(&be);
        worker.enqueue_release(layer);

        // drain with generous deadline — should complete immediately
        worker
            .drain(Duration::from_millis(500))
            .expect("drain should succeed");
        assert_eq!(
            worker.pending_count(),
            0,
            "all jobs must complete after drain"
        );
    }

    #[test]
    fn multiple_enqueues_all_drained() {
        let be = cpu_be();
        let worker = PrimaryReleaseWorker::spawn(be.clone());

        for _ in 0..5 {
            worker.enqueue_release(make_layer(&be));
        }

        worker
            .drain(Duration::from_millis(1000))
            .expect("drain should succeed for 5 layers");
        assert_eq!(worker.pending_count(), 0);
    }

    #[test]
    fn drop_joins_worker_thread() {
        let be = cpu_be();
        // Spawn, enqueue one job, then drop — the drop impl must join cleanly.
        let worker = PrimaryReleaseWorker::spawn(be.clone());
        worker.enqueue_release(make_layer(&be));
        // Drop here. If the join hangs, this test will timeout (not panic).
        drop(worker);
        // Reaching here means the join completed.
    }

    #[test]
    fn drain_returns_ok_when_already_empty() {
        let be = cpu_be();
        let worker = PrimaryReleaseWorker::spawn(be.clone());
        // No enqueue — drain should be a no-op Ok(()).
        worker
            .drain(Duration::from_millis(10))
            .expect("drain on empty must be Ok");
    }

    #[test]
    fn drain_timeout_returns_error() {
        // Simulate a stuck job by not enqueuing a proper layer but instead
        // manually bumping pending without sending a real drop job.
        let be = cpu_be();
        let worker = PrimaryReleaseWorker::spawn(be.clone());

        // Artificially inflate pending without sending a real job.
        worker.pending.fetch_add(1, Ordering::Release);

        let result = worker.drain(Duration::from_millis(10));
        assert!(result.is_err(), "drain should fail when pending stays > 0");

        // Clean up the artificial pending so the drop join doesn't hang.
        worker.pending.fetch_sub(1, Ordering::Release);
    }
}
