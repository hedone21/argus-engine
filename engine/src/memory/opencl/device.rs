use crate::buffer::{Buffer, DType};
use anyhow::Result;
use ocl::core::Mem;
use ocl::{Buffer as OclBuffer, Queue};
use std::any::Any;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Live OpenCLBuffer counters for NVIDIA resource-accumulation diagnosis.
/// Incremented in `OpenCLBuffer::new`, decremented in `Drop`.
pub static LIVE_BUFFERS: AtomicUsize = AtomicUsize::new(0);
pub static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
pub static TOTAL_ALLOCS: AtomicUsize = AtomicUsize::new(0);
pub static TOTAL_RELEASES: AtomicUsize = AtomicUsize::new(0);

pub fn reset_alloc_counters() {
    LIVE_BUFFERS.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
    TOTAL_ALLOCS.store(0, Ordering::Relaxed);
    TOTAL_RELEASES.store(0, Ordering::Relaxed);
}

pub fn snapshot_alloc_counters() -> (usize, usize, usize, usize) {
    (
        LIVE_BUFFERS.load(Ordering::Relaxed),
        LIVE_BYTES.load(Ordering::Relaxed),
        TOTAL_ALLOCS.load(Ordering::Relaxed),
        TOTAL_RELEASES.load(Ordering::Relaxed),
    )
}

// unsafe impl Send for OpenCLBuffer {} // OclBuffer is Send/Sync usually?
// Check ocl docs. OclBuffer is Send/Sync.
// We don't need unsafe impl if fields are Send/Sync.

pub struct OpenCLBuffer {
    pub buffer: OclBuffer<u8>,
    queue: Queue,
    dtype: DType,
    size: usize,
    /// Per-`OpenCLMemory` accounting handle (telemetry only). When `Some`,
    /// `size` was added to the counter at attach time and is subtracted in
    /// `Drop`, so `OpenCLMemory::used_memory()` reflects live bytes rather than
    /// inflating monotonically. `None` for buffers built outside `OpenCLMemory`
    /// (e.g. `transformer.rs`), matching the previous "uncounted" behaviour.
    mem_accounting: Option<Arc<AtomicUsize>>,
}

impl OpenCLBuffer {
    pub fn new(queue: Queue, buffer: OclBuffer<u8>, size: usize, dtype: DType) -> Result<Self> {
        LIVE_BUFFERS.fetch_add(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_add(size, Ordering::Relaxed);
        TOTAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            buffer,
            queue,
            dtype,
            size,
            mem_accounting: None,
        })
    }

    /// Attach a per-`OpenCLMemory` accounting counter: adds `size` now and
    /// subtracts it once on `Drop`. Idempotent-safe to call at most once
    /// (`OpenCLMemory::alloc` calls it exactly once, right after construction).
    pub(crate) fn attach_mem_accounting(&mut self, counter: Arc<AtomicUsize>) {
        counter.fetch_add(self.size, Ordering::Relaxed);
        self.mem_accounting = Some(counter);
    }
}

impl Drop for OpenCLBuffer {
    fn drop(&mut self) {
        LIVE_BUFFERS.fetch_sub(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_sub(self.size, Ordering::Relaxed);
        TOTAL_RELEASES.fetch_add(1, Ordering::Relaxed);
        if let Some(counter) = &self.mem_accounting {
            counter.fetch_sub(self.size, Ordering::Relaxed);
        }
    }
}

impl Buffer for OpenCLBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    fn as_ptr(&self) -> *const u8 {
        // Warning: Direct access not supported without mapping
        // Logic should use copy_from or map explicitly.
        // For now, returning null to satisfy trait.
        ptr::null()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        ptr::null_mut()
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        Some(self.buffer.as_core())
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        self.queue.finish()?;
        Ok(())
    }

    fn is_host_managed(&self) -> bool {
        false // Device-only GPU memory
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}
