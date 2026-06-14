//! Per-op CUDA event based profiler for the `cuda_embedded` backend.
//!
//! Mirrors the design of `OpenCLBackend::enqueue_kernel_labeled` +
//! `flush_and_aggregate_profile` so that `--cuda-profile` produces a label
//! matrix compatible with the OpenCL `--profile-events` aggregate (same
//! `matmul_qkv / matmul_wo / matmul_ffn / rms_norm / rope / attention /
//! kv_update / silu_mul / lm_head` strings). This lets a single downstream
//! analyser compare Adreno and Jetson decode breakdowns apples-to-apples.
//!
//! # Timing model
//! Each launch site wraps the kernel call in a pair of `cuEventRecord`
//! calls (start → launch → end) taken from a pre-allocated pool. The
//! elapsed time between the two events — which `cuEventElapsedTime`
//! reports with ~0.5 µs resolution — approximates pure GPU execution time
//! excluding host-side launch overhead.
//!
//! `cuEventElapsedTime` is **only valid once both events are complete**,
//! so we do not compute elapsed at `record_end` time. Instead `flush()`
//! synchronises the stream once, then sweeps the whole pending record
//! list. This keeps the hot path free of any CUDA API call beyond the
//! two `cuEventRecord`s.
//!
//! # Pool reuse
//! Events are created once at construction and destroyed only on drop.
//! After a `flush()` both `records` and `next_idx` are reset, so the
//! same event objects are re-recorded next token. This avoids the
//! per-token `cuEventCreate` / `cuEventDestroy` churn that would
//! otherwise dominate at DK=64 decode (< 1 ms/tok GPU ops).

use anyhow::{Context, Result, anyhow};
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys as cuda_sys;
use std::collections::HashMap;

/// Fixed labels for CUDA decode ops. The string emitted by
/// `profile_label()` must stay in sync with `OpenCLBackend`'s
/// `OpTag::profile_label` (see `backend/opencl/plan.rs`) so that reports
/// across backends share the same keys.
///
/// `Matmul` is a catch-all for `matmul_transposed` dispatches; the actual
/// label (`matmul_qkv` / `matmul_wo` / `matmul_ffn` / `lm_head`) is
/// supplied via `CudaBackend::set_op_label` at the call site, matching
/// the OpenCL hint mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaOpTag {
    RmsNorm,
    Matmul,
    Rope,
    KvScatter,
    Attention,
    AddAssign,
    SiluMul,
    GeluTanhMul,
    Softmax,
    Cast,
    AddRowBias,
    Gather,
    FlashAttentionPrefill,
    Scale,
}

impl CudaOpTag {
    /// Default label — overridden by `op_label_hint` when present.
    pub fn profile_label(&self) -> &'static str {
        match self {
            CudaOpTag::RmsNorm => "rms_norm",
            CudaOpTag::Matmul => "matmul",
            CudaOpTag::Rope => "rope",
            CudaOpTag::KvScatter => "kv_update",
            CudaOpTag::Attention => "attention",
            CudaOpTag::AddAssign => "add_assign",
            CudaOpTag::SiluMul => "silu_mul",
            CudaOpTag::GeluTanhMul => "gelu_tanh_mul",
            CudaOpTag::Softmax => "softmax",
            CudaOpTag::Cast => "cast",
            CudaOpTag::AddRowBias => "add_row_bias",
            CudaOpTag::Gather => "gather",
            CudaOpTag::FlashAttentionPrefill => "flash_attn_prefill",
            CudaOpTag::Scale => "scale",
        }
    }
}

/// Per-op CUDA event profiler.
///
/// Thread-safety: `CUevent` is a raw driver handle (`*mut c_void` under
/// the hood) and must be used from the same CUDA context that created
/// it. The backend holds this behind a `Mutex<Option<_>>` which gives us
/// Send+Sync on the handle vector safely.
pub struct CudaOpProfiler {
    /// Pre-allocated (start, end) event pairs, reused across tokens.
    events: Vec<(cuda_sys::CUevent, cuda_sys::CUevent)>,
    /// (label, events-index) of records created since the last `flush`.
    /// `label` is the actual string to aggregate under (already
    /// resolved from `op_label_hint` at `record_start` time).
    records: Vec<(&'static str, usize)>,
    /// Next free pool slot.
    next_idx: usize,
    /// Marker for record_end pairing — set by record_start, consumed by
    /// record_end. Avoids having two public arg lists diverge.
    pending_idx: Option<usize>,
    /// (count, total_ms) per label, accumulated across flushes.
    aggregate: HashMap<&'static str, (u64, f64)>,
    /// Number of pool exhaustion drops (records_start attempts after
    /// `next_idx == pool_size` without an intervening flush). Reported
    /// in final summary for diagnostic purposes.
    dropped: u64,
}

// SAFETY: `CUevent` handles created on a given context are safe to use
// from any thread as long as only one thread touches each handle at a
// time; the outer `Mutex<Option<CudaOpProfiler>>` on the backend
// enforces that. No aliasing, no concurrent access.
unsafe impl Send for CudaOpProfiler {}
unsafe impl Sync for CudaOpProfiler {}

impl CudaOpProfiler {
    /// Allocate a pool of `pool_size` (start, end) event pairs.
    pub fn new(pool_size: usize) -> Result<Self> {
        let mut events = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let start = cuda_result::event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)
                .map_err(|e| anyhow!("cuEventCreate failed (start idx {i}): {e}"))?;
            let end = cuda_result::event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)
                .map_err(|e| anyhow!("cuEventCreate failed (end idx {i}): {e}"))?;
            events.push((start, end));
        }
        Ok(Self {
            events,
            records: Vec::with_capacity(pool_size),
            next_idx: 0,
            pending_idx: None,
            aggregate: HashMap::new(),
            dropped: 0,
        })
    }

    /// Record the start event for the next op on `stream`. `label` must
    /// be the already-resolved static label (caller handles the
    /// op_label_hint override).
    ///
    /// Returns `Ok(true)` if recorded, `Ok(false)` if the pool is full
    /// (the caller must still launch the kernel; we just skip timing).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn record_start(
        &mut self,
        label: &'static str,
        stream: cuda_sys::CUstream,
    ) -> Result<bool> {
        if self.next_idx >= self.events.len() {
            self.dropped += 1;
            return Ok(false);
        }
        let idx = self.next_idx;
        let (start_ev, _) = self.events[idx];
        // SAFETY: start_ev was created in this profiler's context; stream
        // is a valid CUstream (default stream from the same context).
        unsafe {
            cuda_result::event::record(start_ev, stream)
                .map_err(|e| anyhow!("cuEventRecord(start) failed: {e}"))?;
        }
        self.pending_idx = Some(idx);
        self.records.push((label, idx));
        Ok(true)
    }

    /// Record the end event paired with the most recent `record_start`.
    /// No-op if `record_start` returned `false` (pool full) or was not
    /// called.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn record_end(&mut self, stream: cuda_sys::CUstream) -> Result<()> {
        let Some(idx) = self.pending_idx.take() else {
            return Ok(());
        };
        let (_, end_ev) = self.events[idx];
        // SAFETY: end_ev was created in this profiler's context; stream
        // matches the one passed to record_start.
        unsafe {
            cuda_result::event::record(end_ev, stream)
                .map_err(|e| anyhow!("cuEventRecord(end) failed: {e}"))?;
        }
        self.next_idx = idx + 1;
        Ok(())
    }

    /// Synchronise the last recorded event and roll up every pending
    /// record into `aggregate`, then reset the pool for the next batch.
    ///
    /// Safe to call when no records are pending (returns immediately).
    pub fn flush(&mut self) -> Result<()> {
        if self.records.is_empty() {
            self.next_idx = 0;
            self.pending_idx = None;
            return Ok(());
        }

        // Ensure every end event is complete before reading elapsed.
        // Synchronising the last end event is sufficient because events
        // on the default stream complete in submission order.
        let last_idx = self.next_idx.saturating_sub(1);
        if last_idx < self.events.len() {
            let (_, last_end) = self.events[last_idx];
            // SAFETY: event was recorded above.
            unsafe {
                cuda_result::event::synchronize(last_end)
                    .map_err(|e| anyhow!("cuEventSynchronize failed: {e}"))?;
            }
        }

        for (label, idx) in self.records.drain(..) {
            let (start_ev, end_ev) = self.events[idx];
            // SAFETY: both events were recorded and are complete (see
            // synchronize above).
            let ms = unsafe {
                cuda_result::event::elapsed(start_ev, end_ev)
                    .with_context(|| format!("cuEventElapsedTime failed for '{label}'"))?
            };
            let entry = self.aggregate.entry(label).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += ms as f64;
        }

        self.next_idx = 0;
        self.pending_idx = None;
        Ok(())
    }

    /// Accumulated per-label (count, total_ms).
    pub fn report(&self) -> &HashMap<&'static str, (u64, f64)> {
        &self.aggregate
    }

    /// Number of start attempts dropped because the pool was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Clear aggregate + in-flight state without releasing the pool.
    pub fn reset(&mut self) {
        self.records.clear();
        self.next_idx = 0;
        self.pending_idx = None;
        self.aggregate.clear();
        self.dropped = 0;
    }
}

impl Drop for CudaOpProfiler {
    fn drop(&mut self) {
        for (start, end) in self.events.drain(..) {
            // SAFETY: events were created by `cuda_result::event::create`
            // and are destroyed exactly once here.
            unsafe {
                let _ = cuda_result::event::destroy(start);
                let _ = cuda_result::event::destroy(end);
            }
        }
    }
}
