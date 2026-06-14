//! LISWAP-8 Phase B — `LayerObjectPool`.
//!
//! Pre-allocates a pool of `TransformerLayer` objects (with their device
//! buffers) and a background allocator thread that keeps the pool topped
//! up. The swap dispatch path takes a pool entry instead of calling
//! `cuMemAlloc` for each tensor — overwriting the existing device buffers
//! via `Backend::enqueue_write_into_async`.
//!
//! Hypothesis: CUDA driver context lock contention between worker
//! `cuMemAlloc` and main-thread `cuLaunchKernel` is the dominant cause of
//! the Phase A active-window forward regression (+15..+30 ms for K < 32,
//! measured 2026-05-15 K sweep). Moving alloc to a separate thread
//! (background re-supply) lets us isolate the contribution.
//!
//! Scope (PoC):
//! - CUDA embedded only (uses `CudaDeviceBuffer` directly).
//! - Weight dtype is fixed at construction (target_dtype, e.g. Q4_0).
//! - Norms use the sample layer's norm dtype (usually F32).
//! - Optional norm tensors (qkv_bias, q_norm, ...) are left `None` —
//!   swap_executor's caller path will clone them from the displaced
//!   primary just like `build_layer_from_mmap_async` does today.

#![cfg(feature = "cuda-embedded")]

use std::collections::VecDeque;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use anyhow::{Result, anyhow};

use crate::backend::Backend;
use crate::buffer::{Buffer, DType};
use crate::layers::transformer_layer::TransformerLayer;
use crate::memory::cuda::buffer::{CudaDeviceBuffer, CudaHostBuffer};
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Per-tensor allocation spec — captured once from a sample
/// `TransformerLayer` so the background allocator thread can rebuild
/// pool entries without touching the live model.
#[derive(Clone)]
pub struct LayerSpec {
    pub wq_shape: Shape,
    pub wk_shape: Shape,
    pub wv_shape: Shape,
    pub wo_shape: Shape,
    pub w_gate_shape: Shape,
    pub w_up_shape: Shape,
    pub w_down_shape: Shape,
    pub attention_norm_shape: Shape,
    pub ffn_norm_shape: Shape,
    /// Dtype for weight tensors (e.g. Q4_0).
    pub weight_dtype: DType,
    /// Dtype for norms (usually F32).
    pub norm_dtype: DType,
    /// LISWAP-8 Phase B-2: when `true`, allocate pool entries as
    /// `CudaHostBuffer` (pinned + DEVICEMAP) instead of `CudaDeviceBuffer`.
    /// On Jetson UMA this enables a pure CPU memcpy swap path with no
    /// `cuMemcpyHtoDAsync` driver lock contention against forward kernels.
    pub zero_copy: bool,
}

impl LayerSpec {
    pub fn from_sample(sample: &TransformerLayer, weight_dtype: DType) -> Self {
        Self {
            wq_shape: sample.wq.shape().clone(),
            wk_shape: sample.wk.shape().clone(),
            wv_shape: sample.wv.shape().clone(),
            wo_shape: sample.wo.shape().clone(),
            w_gate_shape: sample.w_gate.shape().clone(),
            w_up_shape: sample.w_up.shape().clone(),
            w_down_shape: sample.w_down.shape().clone(),
            attention_norm_shape: sample.attention_norm.shape().clone(),
            ffn_norm_shape: sample.ffn_norm.shape().clone(),
            weight_dtype,
            norm_dtype: sample.attention_norm.dtype(),
            zero_copy: false,
        }
    }

    /// Builder: opt into the UMA zero-copy pool path (Phase B-2).
    pub fn with_zero_copy(mut self, zero_copy: bool) -> Self {
        self.zero_copy = zero_copy;
        self
    }
}

/// Pool of pre-allocated `TransformerLayer` objects, refilled by a single
/// background allocator thread on demand.
pub struct LayerObjectPool {
    ready: Arc<Mutex<VecDeque<TransformerLayer>>>,
    /// Signal to the background thread to attempt a top-up.
    /// `Option` so `Drop` can close the channel (drops sender) before
    /// joining the background thread — otherwise the receiver blocks on
    /// `recv()` forever and join deadlocks.
    alloc_tx: Option<Sender<()>>,
    target_depth: usize,
    /// `Some` until `Drop` joins the thread.
    alloc_thread: Option<JoinHandle<()>>,
}

impl LayerObjectPool {
    /// Construct a pool with `target_depth` pre-allocated entries.
    ///
    /// The constructor performs the initial allocation on the calling
    /// thread, then spawns the background allocator that responds to
    /// `take()` signals by topping the pool back to `target_depth`.
    pub fn new(
        backend: Arc<dyn Backend>,
        spec: LayerSpec,
        target_depth: usize,
    ) -> Result<Arc<Self>> {
        if target_depth == 0 {
            anyhow::bail!("LayerObjectPool: target_depth must be > 0");
        }

        let mut initial = VecDeque::with_capacity(target_depth);
        for i in 0..target_depth {
            initial.push_back(
                create_empty_layer(&backend, &spec)
                    .map_err(|e| anyhow!("LayerObjectPool: initial alloc {i} failed: {e}"))?,
            );
        }
        let ready = Arc::new(Mutex::new(initial));

        let (alloc_tx, alloc_rx) = mpsc::channel::<()>();
        let ready_for_thread = Arc::clone(&ready);
        let backend_for_thread = Arc::clone(&backend);
        let spec_for_thread = spec.clone();

        let alloc_thread = thread::Builder::new()
            .name("llmrs-layer-pool-alloc".into())
            .spawn(move || {
                // Bind the backend's hardware context to this thread so
                // background alloc calls resolve to the same context as the
                // main thread (LISWAP-8 Phase B observed: CUDA_ERROR_INVALID_CONTEXT
                // without this). The `Backend` trait method abstracts the
                // CUDA-specific `bind_to_thread()` call — non-CUDA backends
                // default to a no-op (Migration Step 3-B, V-27 resolution).
                if let Err(e) = backend_for_thread.bind_current_thread() {
                    eprintln!("[LayerPool] bind_current_thread failed: {e}");
                    return;
                }
                while alloc_rx.recv().is_ok() {
                    loop {
                        let cur = ready_for_thread.lock().unwrap().len();
                        if cur >= target_depth {
                            break;
                        }
                        match create_empty_layer(&backend_for_thread, &spec_for_thread) {
                            Ok(entry) => {
                                ready_for_thread.lock().unwrap().push_back(entry);
                            }
                            Err(e) => {
                                eprintln!("[LayerPool] background alloc failed: {e}");
                                break;
                            }
                        }
                    }
                }
            })
            .map_err(|e| anyhow!("LayerObjectPool: spawn alloc thread failed: {e}"))?;

        Ok(Arc::new(Self {
            ready,
            alloc_tx: Some(alloc_tx),
            target_depth,
            alloc_thread: Some(alloc_thread),
        }))
    }

    /// Pop one entry from the pool, signalling the background thread to
    /// refill. Returns `None` if the pool is empty (caller should fall
    /// back to inline allocation in that case).
    pub fn take(&self) -> Option<TransformerLayer> {
        let entry = self.ready.lock().unwrap().pop_front();
        // Signal background to top up (non-blocking).
        if let Some(tx) = &self.alloc_tx {
            let _ = tx.send(());
        }
        entry
    }

    /// Current ready depth — for diagnostics.
    pub fn depth(&self) -> usize {
        self.ready.lock().unwrap().len()
    }

    pub fn target_depth(&self) -> usize {
        self.target_depth
    }
}

impl Drop for LayerObjectPool {
    fn drop(&mut self) {
        // Explicitly close the channel by dropping the sender. This makes
        // the background thread's `recv()` return `Err`, breaking the
        // loop. Without this, `join()` blocks forever (observed in PoC
        // first run, 2026-05-15).
        let _ = self.alloc_tx.take();
        if let Some(h) = self.alloc_thread.take() {
            let _ = h.join();
        }
    }
}

impl crate::layers::staging_pool::WeightStagingPool for LayerObjectPool {
    fn take(&self) -> Option<TransformerLayer> {
        LayerObjectPool::take(self)
    }

    fn depth(&self) -> usize {
        LayerObjectPool::depth(self)
    }

    fn target_depth(&self) -> usize {
        LayerObjectPool::target_depth(self)
    }
}

fn create_empty_layer(backend: &Arc<dyn Backend>, spec: &LayerSpec) -> Result<TransformerLayer> {
    let zero_copy = spec.zero_copy;
    let alloc = |shape: &Shape, dtype: DType| -> Result<Tensor> {
        let size = shape.numel() * dtype.size();
        let buf: Arc<dyn Buffer> = if zero_copy {
            // Phase B-2 zero-copy: pinned + DEVICEMAP. On Jetson UMA, the
            // host pointer and device pointer alias the same physical
            // DRAM page — CPU writes are immediately GPU-readable, so the
            // swap path can skip cuMemcpyHtoDAsync entirely.
            let b = CudaHostBuffer::new(size, dtype)
                .map_err(|e| anyhow!("CudaHostBuffer::new({size} bytes, {dtype:?}) failed: {e}"))?;
            Arc::new(b)
        } else {
            let b = CudaDeviceBuffer::new(size, dtype).map_err(|e| {
                anyhow!("CudaDeviceBuffer::new({size} bytes, {dtype:?}) failed: {e}")
            })?;
            Arc::new(b)
        };
        Ok(Tensor::new(shape.clone(), buf, Arc::clone(backend)))
    };

    Ok(TransformerLayer {
        wq: alloc(&spec.wq_shape, spec.weight_dtype)?,
        wk: alloc(&spec.wk_shape, spec.weight_dtype)?,
        wv: alloc(&spec.wv_shape, spec.weight_dtype)?,
        wo: alloc(&spec.wo_shape, spec.weight_dtype)?,
        w_gate: alloc(&spec.w_gate_shape, spec.weight_dtype)?,
        w_up: alloc(&spec.w_up_shape, spec.weight_dtype)?,
        w_down: alloc(&spec.w_down_shape, spec.weight_dtype)?,
        attention_norm: alloc(&spec.attention_norm_shape, spec.norm_dtype)?,
        ffn_norm: alloc(&spec.ffn_norm_shape, spec.norm_dtype)?,
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    })
}
