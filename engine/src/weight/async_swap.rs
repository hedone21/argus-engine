//! `AsyncSwapDispatcher` — off-critical-path layer weight commit worker.
//!
//! LISWAP-2 prototype (plan: `chasing-hopper`). Receives `SwapJob::Commit`
//! jobs from the main decode thread, waits on each job's GPU H2D write-
//! completion event off the critical path, then commits the new
//! `LayerWeights` to the slot via ArcSwap. Released primaries are chained
//! into `PrimaryReleaseWorker` (ENG-ALG-228) when the caller supplies one.
//!
//! # Design
//!
//! Mirrors `PrimaryReleaseWorker` (`release_worker.rs`) exactly:
//! - `mpsc` channel carries typed job enum
//! - single background thread (`llmrs-async-swap`)
//! - `Arc<AtomicUsize>` pending counter visible to caller
//! - `drain(deadline)` polls at 1 ms intervals (same API as `PrimaryReleaseWorker`)
//! - `Drop` sends `Shutdown` + joins thread
//!
//! # Thread safety
//!
//! `AsyncSwapDispatcher` is `Send + Sync` because:
//! - `mpsc::Sender<SwapJob>` is `Send` (not `Sync`, but we only ever call
//!   `send` from a reference, which is fine through `Arc`).
//! - `Arc<AtomicUsize>` is `Send + Sync`.
//! - `Option<JoinHandle<()>>` is `Send`.
//!
//! `GpuEvent` must be `Send + Sync` for the job to cross thread boundaries;
//! this is guaranteed by `core::backend::GpuEvent`'s `Send + Sync` bounds.
//!
//! # Error handling
//!
//! Prototype-grade: `wait_event_blocking` failures are logged to stderr and
//! the commit is skipped (the slot retains the old weights). No sophisticated
//! retry.

use crate::backend::{Backend, GpuEvent};
use crate::buffer::DType;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference weight resource (LayerSlot)
use crate::models::weights::slot::{LayerSlot, LayerWeights};
use crate::runtime_resources_access::ReleaseWorkerAccess;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// ── Public types ────────────────────────────────────────────────────────────

/// A single commit request for the async dispatcher worker.
pub struct SwapCommitJob {
    /// The layer slot whose weights will be replaced.
    pub slot: Arc<LayerSlot>,
    /// New weights to install.
    pub new_weights: Arc<LayerWeights>,
    /// DType of the new weights (written to `LayerSlot::current_dtype`).
    pub new_dtype: DType,
    /// GPU event that signals H2D write completion. Worker blocks on this
    /// before committing. Wrapped in `Arc` so the same event can be shared
    /// with the forward-thread wait gate (LISWAP-4 v2 / INV-149) — both the
    /// dispatcher worker and the forward thread call
    /// `Backend::wait_event_blocking` on the same underlying cl_event.
    /// Use `Arc::new(GpuEvent::dummy())` for a synchronous-fallback backend.
    pub write_event: Arc<GpuEvent>,
    /// Optional release worker for chaining displaced primary weights.
    /// When `Some`, successful `Arc::try_unwrap` on the old weights triggers
    /// `release_worker.enqueue_release(old_layer)`. When `None`, old weights
    /// are dropped inline.
    pub release_worker: Option<Arc<dyn ReleaseWorkerAccess>>,
    /// Optional commit-complete callback (LISWAP-4 / ENG-ALG-237).
    ///
    /// Invoked on the dispatcher worker thread immediately after
    /// `slot.swap_weights` succeeds, with `layer_idx` (when set) identifying
    /// which slot was committed. Used by `IntraForwardSwapHook` to clear its
    /// per-slot pending-event registry. When `layer_idx` is `None`, the
    /// callback receives `usize::MAX` (sentinel — caller-supplied closures
    /// that need a real index must always set `layer_idx`).
    pub on_complete: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    /// Layer index for the `on_complete` callback (LISWAP-4). `None` for
    /// production paths that do not use `on_complete`.
    pub layer_idx: Option<usize>,
}

/// LISWAP Phase 4: Worker-side chunk dispatch job.
///
/// Carries a `Weak<PhaseAwareSwapDispatcher>` so the worker can call back into
/// `try_dispatch_chunk_worker` without creating an Arc ownership cycle. When
/// the dispatcher is dropped, the upgrade-fail makes the job a noop.
///
/// The trait object is hidden behind a generic `Fn` so async_swap.rs does not
/// have to depend on phase_aware_swap.rs (avoiding a module cycle). The
/// callback closes over the `Weak<PhaseAwareSwapDispatcher>` and performs the
/// upgrade + dispatch internally.
pub struct ChunkDispatchJob {
    /// Boxed callback: invoked once on the worker thread. Implemented as a
    /// `FnOnce` so the closure can move the `Weak` into itself.
    pub run: Box<dyn FnOnce() + Send + 'static>,
}

/// Jobs sent to the background worker thread.
pub enum SwapJob {
    /// Commit a set of new layer weights after waiting for the write event.
    Commit(SwapCommitJob),
    /// LISWAP Phase 4: Dispatch the next phase-aware chunk on the worker
    /// thread instead of the forward thread.
    DispatchChunk(ChunkDispatchJob),
    /// Terminate the worker loop gracefully.
    Shutdown,
}

// ── AsyncSwapDispatcher ─────────────────────────────────────────────────────

/// Async layer swap dispatcher (LISWAP-2 prototype).
///
/// Submit `SwapCommitJob`s with [`submit_commit`][Self::submit_commit]. Each
/// job is processed by the background worker:
/// 1. Wait for the GPU H2D write event (`wait_event_blocking`).
/// 2. Commit the new weights into the slot via ArcSwap.
/// 3. Chain the displaced primary weights into `PrimaryReleaseWorker` (if
///    the job supplies one and `Arc::try_unwrap` succeeds).
///
/// Call [`drain`][Self::drain] to wait for all submitted jobs to complete
/// before reading results (e.g. at plan-completion time in the decode loop).
///
/// `Drop` sends a `Shutdown` job and joins the worker thread, ensuring all
/// destructors run before the process exits.
pub struct AsyncSwapDispatcher {
    sender: Sender<SwapJob>,
    /// Number of submitted jobs not yet acknowledged by the worker.
    pending: Arc<AtomicUsize>,
    /// Worker thread handle. `Some` until `drop` consumes it.
    handle: Option<JoinHandle<()>>,
}

impl AsyncSwapDispatcher {
    /// Spawn the background worker thread, retaining `backend` for
    /// `wait_event_blocking` calls inside the worker loop. Failures inside
    /// `process_commit` are logged directly via the action diag helpers.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        let (sender, receiver) = mpsc::channel::<SwapJob>();
        let pending = Arc::new(AtomicUsize::new(0));
        let pending_worker = Arc::clone(&pending);

        let handle = thread::Builder::new()
            .name("llmrs-async-swap".into())
            .spawn(move || {
                worker_loop(receiver, backend, pending_worker);
            })
            .expect("failed to spawn AsyncSwapDispatcher thread");

        Self {
            sender,
            pending,
            handle: Some(handle),
        }
    }

    /// Submit a commit job to the background worker.
    ///
    /// `pending` is incremented before the send so observers see
    /// `pending_count() > 0` immediately. If the channel is closed (worker
    /// already dead), `pending` is rolled back and an error is returned.
    pub fn submit_commit(&self, job: SwapCommitJob) -> Result<()> {
        self.pending.fetch_add(1, Ordering::Release);
        if let Err(e) = self.sender.send(SwapJob::Commit(job)) {
            self.pending.fetch_sub(1, Ordering::Release);
            return Err(anyhow!(
                "[AsyncSwapDispatcher] channel closed, cannot submit job: {e}"
            ));
        }
        Ok(())
    }

    /// LISWAP Phase 4: Submit a chunk-dispatch job to the worker.
    ///
    /// Mirrors `submit_commit` exactly — the job is enqueued (channel push,
    /// ~µs scale) and `pending` is incremented so the same `drain` loop covers
    /// both job kinds. The actual chunk build/install runs on the worker.
    pub fn submit_dispatch_chunk(&self, job: ChunkDispatchJob) -> Result<()> {
        self.pending.fetch_add(1, Ordering::Release);
        if let Err(e) = self.sender.send(SwapJob::DispatchChunk(job)) {
            self.pending.fetch_sub(1, Ordering::Release);
            return Err(anyhow!(
                "[AsyncSwapDispatcher] channel closed, cannot submit dispatch job: {e}"
            ));
        }
        Ok(())
    }

    /// Block until all pending jobs complete or `deadline` elapses.
    ///
    /// Polls at 1 ms intervals (same pattern as `PrimaryReleaseWorker::drain`).
    /// Returns `Ok(())` when `pending == 0`, or `Err` on timeout.
    pub fn drain(&self, deadline: Duration) -> Result<()> {
        let start = Instant::now();
        loop {
            if self.pending.load(Ordering::Acquire) == 0 {
                return Ok(());
            }
            if start.elapsed() >= deadline {
                let remaining = self.pending.load(Ordering::Acquire);
                return Err(anyhow!(
                    "[AsyncSwapDispatcher] drain timeout: {remaining} job(s) pending after {}ms",
                    deadline.as_millis()
                ));
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Number of submitted jobs not yet completed by the worker.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }
}

impl Drop for AsyncSwapDispatcher {
    fn drop(&mut self) {
        // Best-effort: ignore send error if worker already terminated.
        let _ = self.sender.send(SwapJob::Shutdown);
        if let Some(h) = self.handle.take() {
            // join() can only fail if the worker panicked; ignore the error
            // to avoid a double-panic on drop.
            let _ = h.join();
        }
    }
}

// ── Worker loop ─────────────────────────────────────────────────────────────

fn worker_loop(rx: mpsc::Receiver<SwapJob>, backend: Arc<dyn Backend>, pending: Arc<AtomicUsize>) {
    while let Ok(job) = rx.recv() {
        match job {
            SwapJob::Commit(commit) => {
                process_commit(commit, &backend);
                pending.fetch_sub(1, Ordering::Release);
            }
            SwapJob::DispatchChunk(dispatch) => {
                // The closure captures Weak<PhaseAwareSwapDispatcher> + invokes
                // try_dispatch_chunk_worker on it. Upgrade-fail = noop.
                (dispatch.run)();
                pending.fetch_sub(1, Ordering::Release);
            }
            SwapJob::Shutdown => break,
        }
    }
}

/// Execute one commit job: wait → swap → chain release → on_complete.
fn process_commit(job: SwapCommitJob, backend: &Arc<dyn Backend>) {
    // LISWAP-6 Phase 5: alias path는 cl_mem이 이미 GPU-visible 이므로
    // wait할 GPU 작업 없음. dummy event는 wait_event_blocking 의 fall-through
    // self.synchronize() 가 forward GPU op까지 block 시키는 부작용을 일으키므로
    // skip한다. 비-alias path (memcpy)는 정상대로 wait.
    if !job.write_event.is_dummy()
        && let Err(e) = backend.wait_event_blocking(job.write_event.as_ref())
    {
        crate::action_diag_helper::log_swap_failed(
            "Subsystem",
            &format!("wait_event_blocking failed: {e}; commit skipped"),
            job.layer_idx,
            None,
        );
        // slot retains old weights — safe to continue
        return;
    }

    // Atomically install new weights and retrieve the displaced old Arc.
    let old = job.slot.swap_weights(job.new_weights, job.new_dtype);

    // LISWAP-4 (ENG-ALG-237): notify the hook that the commit landed so it
    // can clear its `pending_events[idx]` registry. Run before the release
    // chain so forward-thread wait gates see the cleared state as soon as
    // possible.
    if let Some(cb) = &job.on_complete {
        let idx = job.layer_idx.unwrap_or(usize::MAX);
        cb(idx);
    }

    // Chain old weights into the release worker when we hold exclusive
    // ownership (the common case in steady-state swap).
    if let Some(rw) = job.release_worker {
        match Arc::try_unwrap(old) {
            Ok(layer) => rw.enqueue_release(layer),
            Err(_arc) => {
                // Another Arc clone exists; drop inline. This should not
                // occur in normal operation (forward pass snapshots are
                // short-lived), but is safe.
            }
        }
    }
    // When release_worker is None or try_unwrap fails, old is dropped here.
}

// ── Unit tests ───────────────────────────────────────────────────────────────

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

    // ── helpers ──────────────────────────────────────────────────────────

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

    fn make_slot(be: &Arc<dyn Backend>, dtype: DType) -> Arc<LayerSlot> {
        Arc::new(LayerSlot::new(make_layer(be), dtype, None, 0))
    }

    fn make_weights(be: &Arc<dyn Backend>) -> Arc<LayerWeights> {
        Arc::new(make_layer(be))
    }

    // ── test_submit_one_commit_drain ─────────────────────────────────────

    /// Submit one commit job and drain — verify the slot dtype changed.
    #[test]
    fn test_submit_one_commit_drain() {
        let be = cpu_be();
        let dispatcher = AsyncSwapDispatcher::new(be.clone());

        let slot = make_slot(&be, DType::F16);
        assert_eq!(slot.current_dtype(), DType::F16);

        let new_weights = make_weights(&be);
        dispatcher
            .submit_commit(SwapCommitJob {
                slot: slot.clone(),
                new_weights,
                new_dtype: DType::Q4_0,
                write_event: Arc::new(GpuEvent::dummy()),
                release_worker: None,
                on_complete: None,
                layer_idx: None,
            })
            .expect("submit_commit must succeed");

        dispatcher
            .drain(Duration::from_secs(1))
            .expect("drain must complete within 1 s");

        assert_eq!(slot.current_dtype(), DType::Q4_0, "dtype must be updated");
        assert_eq!(dispatcher.pending_count(), 0);
    }

    // ── test_submit_multiple_concurrent ──────────────────────────────────

    /// Submit 10 jobs from separate threads concurrently; drain; verify all
    /// slots reflect the new dtype and pending == 0.
    #[test]
    fn test_submit_multiple_concurrent() {
        let be = cpu_be();
        let dispatcher = Arc::new(AsyncSwapDispatcher::new(be.clone()));

        let slots: Vec<Arc<LayerSlot>> = (0..10).map(|_| make_slot(&be, DType::F16)).collect();

        // Spawn one thread per slot to submit jobs concurrently.
        let handles: Vec<_> = slots
            .iter()
            .map(|slot| {
                let d = dispatcher.clone();
                let s = slot.clone();
                let be2 = be.clone();
                thread::spawn(move || {
                    d.submit_commit(SwapCommitJob {
                        slot: s,
                        new_weights: make_weights(&be2),
                        new_dtype: DType::Q4_0,
                        write_event: Arc::new(GpuEvent::dummy()),
                        release_worker: None,
                        on_complete: None,
                        layer_idx: None,
                    })
                    .expect("concurrent submit_commit must succeed");
                })
            })
            .collect();

        for h in handles {
            h.join().expect("spawned thread must not panic");
        }

        dispatcher
            .drain(Duration::from_secs(2))
            .expect("drain must complete within 2 s");

        assert_eq!(dispatcher.pending_count(), 0, "all jobs must complete");
        for (i, slot) in slots.iter().enumerate() {
            assert_eq!(
                slot.current_dtype(),
                DType::Q4_0,
                "slot {i} dtype must be Q4_0"
            );
        }
    }

    // ── test_drop_without_drain_terminates_thread ────────────────────────

    /// Drop the dispatcher without calling drain — the Drop impl must join
    /// the worker thread without hanging.
    #[test]
    fn test_drop_without_drain_terminates_thread() {
        let be = cpu_be();
        let dispatcher = AsyncSwapDispatcher::new(be.clone());

        let slot = make_slot(&be, DType::F16);
        dispatcher
            .submit_commit(SwapCommitJob {
                slot,
                new_weights: make_weights(&be),
                new_dtype: DType::Q4_0,
                write_event: Arc::new(GpuEvent::dummy()),
                release_worker: None,
                on_complete: None,
                layer_idx: None,
            })
            .expect("submit must succeed");

        // Drop without drain. If join() in Drop hangs, the test runner will
        // time out — which counts as a failure.
        drop(dispatcher);
        // Reaching here means the join completed cleanly.
    }

    // ── test_drain_deadline_exceeded ─────────────────────────────────────

    /// Empty drain must return Ok immediately; then inflate pending and verify
    /// that a very short drain deadline returns Err.
    #[test]
    fn test_drain_deadline_exceeded() {
        let be = cpu_be();
        let dispatcher = AsyncSwapDispatcher::new(be.clone());

        // No jobs — drain on empty must be instant Ok.
        dispatcher
            .drain(Duration::from_millis(1))
            .expect("drain on empty queue must return Ok");

        // Artificially inflate pending without sending a real job so the
        // worker never decrements it.
        dispatcher.pending.fetch_add(1, Ordering::Release);

        // A 1 ms deadline against a permanently-stuck pending should time out.
        let result = dispatcher.drain(Duration::from_millis(10));
        assert!(
            result.is_err(),
            "drain must return Err when pending stays > 0 past deadline"
        );

        // Clean up artificial inflation so Drop join doesn't hang.
        dispatcher.pending.fetch_sub(1, Ordering::Release);
    }
}
