//! `AufViewBuffer` — zero-copy read-only borrow into an AUF primary mmap region.
//!
//! Used by `AufSource::load_tensor` to expose tensor bytes from an `AufView`
//! without copying. The buffer holds an `Arc<AufView>` so the underlying mmap
//! stays alive for the buffer's lifetime.
//!
//! Safety contract: The pointer is derived from `AufView::raw_bytes()`. Caller
//! must guarantee `abs_offset + size <= raw_bytes().len()`. The buffer is
//! read-only; `as_mut_ptr()` returns `null_mut()`.

use crate::auf::AufView;
use crate::buffer::{Buffer, DType};
use anyhow::Result;
#[cfg(feature = "opencl")]
use ocl::core::Mem;
use std::any::Any;
use std::sync::Arc;

/// Zero-copy read-only buffer pointing into an `AufView`'s mmap.
///
/// Holds an `Arc<AufView>` clone so the mmap region is never unmapped while
/// this buffer is live.
pub struct AufViewBuffer {
    /// Pointer into the AUF mmap region.
    /// Valid for as long as `_view` is alive.
    ptr: *const u8,
    /// Length in bytes of the borrowed slice.
    size: usize,
    /// Data type of the tensor stored in this slice.
    dtype: DType,
    /// Keeps the AUF mmap alive.
    _view: Arc<AufView>,
}

// SAFETY: The underlying mmap is read-only. Pointer is derived from a slice
// owned by `_view` and remains valid for the Arc's lifetime.
unsafe impl Send for AufViewBuffer {}
unsafe impl Sync for AufViewBuffer {}

impl AufViewBuffer {
    /// Create a new `AufViewBuffer` referencing bytes within `view`'s mmap.
    ///
    /// # Safety
    /// `abs_offset + size` must not exceed `view.raw_bytes().len()`. The
    /// caller is responsible for ensuring this — typically by deriving
    /// `abs_offset` from a `TensorEntry::variant_offsets` value plus the
    /// WEIGHTS section base offset.
    pub unsafe fn new(view: Arc<AufView>, abs_offset: usize, size: usize, dtype: DType) -> Self {
        let raw = view.raw_bytes();
        debug_assert!(abs_offset + size <= raw.len());
        // Safety: caller guarantees abs_offset + size <= raw.len()
        let ptr = unsafe { raw.as_ptr().add(abs_offset) };
        Self {
            ptr,
            size,
            dtype,
            _view: view,
        }
    }
}

impl Buffer for AufViewBuffer {
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
        self.ptr
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        None
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        Ok(())
    }
}
