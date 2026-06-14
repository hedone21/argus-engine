//! `HostPtrPoolBuffer` — `Arc<dyn Buffer>` adapter around a
//! `HostPtrPoolGuard` so a `Tensor` can wrap a pool slot.
//!
//! LISWAP-3 prototype (plan: `compiled-chasing-hopper`, Direction A track,
//! Stage 3). The buffer holds:
//! - the `HostPtrPoolGuard` — releases the slot back to the pool on drop;
//! - an optional `Arc<dyn MmapKeepAlive>` — keeps the source mmap alive for
//!   the duration of the slot's use, mirroring `MmapBuffer::borrow`'s
//!   INV-143 contract. Migration Step 3-E (V-09) erased the previous
//!   concrete `Arc<SecondaryMmap>` to this marker-trait Arc.
//!
//! The buffer reports the *requested* tensor size (not the slot capacity)
//! so kernels see the correct logical size. `cl_mem()` returns the slot's
//! `cl_mem` handle directly. `as_ptr` / `as_mut_ptr` return null because
//! the GPU view has already been written via the backend's
//! `fill_host_ptr_buffer` (map / memcpy / unmap) before the buffer is
//! constructed, and any subsequent host access would require a fresh map
//! cycle that the swap path does not exercise.

use std::any::Any;
use std::sync::Arc;

use anyhow::Result;
use ocl::core::Mem;

use crate::backend::opencl::host_ptr_pool::HostPtrPoolGuard;
use crate::buffer::{Buffer, DType};
use crate::memory::host::mmap::MmapKeepAlive;

/// Buffer that wraps a `HostPtrPoolGuard`. See module docs.
pub struct HostPtrPoolBuffer {
    /// RAII slot guard. When this buffer drops, the slot returns to the pool.
    guard: HostPtrPoolGuard,
    /// Logical tensor size (= bytes copied into the slot).
    size: usize,
    /// Tensor dtype.
    dtype: DType,
    /// Optional Arc keeping the mmap region alive while we still reference
    /// it — same INV-143 lifetime contract as `MmapBuffer::borrow`. The
    /// fill path memcpy's into the slot synchronously, so once the slot is
    /// filled the mmap region is no longer required for correctness; the
    /// Arc is retained as a defence-in-depth guard until the buffer drops.
    _mmap_guard: Option<Arc<dyn MmapKeepAlive>>,
}

// SAFETY: `ocl::core::Mem` is `Send + Sync`; `HostPtrPoolGuard` carries an
// `Arc<HostPtrPool>` which is `Send + Sync`; the optional `Arc<dyn MmapKeepAlive>`
// is `Send + Sync` by the `MmapKeepAlive` super-trait bounds.
unsafe impl Send for HostPtrPoolBuffer {}
unsafe impl Sync for HostPtrPoolBuffer {}

impl HostPtrPoolBuffer {
    /// Wrap a freshly-filled pool slot. `size` is the logical tensor size
    /// in bytes (must be `<= guard.capacity()`).
    pub fn new(
        guard: HostPtrPoolGuard,
        size: usize,
        dtype: DType,
        mmap_guard: Option<Arc<dyn MmapKeepAlive>>,
    ) -> Self {
        debug_assert!(
            size <= guard.capacity(),
            "HostPtrPoolBuffer: size {size} > capacity {}",
            guard.capacity()
        );
        Self {
            guard,
            size,
            dtype,
            _mmap_guard: mmap_guard,
        }
    }
}

impl Buffer for HostPtrPoolBuffer {
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
        // Pool slots are GPU-visible after the unmap step in the fill
        // helper; host-side reads would need a fresh map cycle that the
        // swap path does not perform. Returning null mirrors
        // `OpenCLBuffer::as_ptr`'s contract.
        std::ptr::null()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }

    fn cl_mem(&self) -> Option<&Mem> {
        Some(self.guard.mem())
    }

    fn sync_device(&self) -> Result<()> {
        // The slot was unmapped by the fill helper; nothing more to do.
        Ok(())
    }

    fn is_host_managed(&self) -> bool {
        // Pool slots are driver-pinned (`CL_MEM_ALLOC_HOST_PTR`). madvise
        // would not be effective on them — same contract as `UnifiedBuffer`.
        false
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}
