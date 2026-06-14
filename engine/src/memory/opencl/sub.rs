//! ClSubBuffer: Zero-copy sub-region of an existing OpenCL buffer.
//!
//! Wraps a parent `Arc<dyn Buffer>` and creates a `clCreateSubBuffer` handle
//! pointing to a byte sub-range `[offset, offset+length)` of the parent's
//! `cl_mem`. No data is copied -- the GPU accesses the same physical memory.
//!
//! Key properties:
//! - `as_ptr()` / `as_mut_ptr()` -> parent pointer + offset (always valid for CPU)
//! - `cl_mem()` -> the sub-buffer handle (valid for GPU kernels)
//! - **Zero additional memory allocation** -- only an OpenCL handle is created
//! - Parent buffer's lifetime is tied via Arc (memory stays valid)

use crate::buffer::{Buffer, DType};
use anyhow::{Result, anyhow};
use ocl::core::{self, BufferRegion, Mem};
use std::any::Any;
use std::sync::Arc;

pub struct ClSubBuffer {
    /// Original buffer that owns the memory (kept alive via Arc).
    parent: Arc<dyn Buffer>,
    /// `clCreateSubBuffer` result -- a view into the parent's cl_mem.
    sub_cl_mem: Mem,
    /// Byte offset from the start of the parent buffer.
    offset: usize,
    /// Byte length of this sub-buffer.
    length: usize,
    /// Data type label carried through from the weight tensor.
    dtype: DType,
}

// SAFETY: The parent buffer (Arc<dyn Buffer>) is Send+Sync by trait bound.
// The Mem handle is an opaque wrapper around cl_mem (a pointer-sized handle)
// that is safe to share across threads -- OpenCL runtime handles concurrency.
unsafe impl Send for ClSubBuffer {}
unsafe impl Sync for ClSubBuffer {}

impl ClSubBuffer {
    /// Create a zero-copy sub-buffer referencing `parent[offset..offset+length]`.
    ///
    /// The parent must have a valid `cl_mem()` handle. Returns `Err` if:
    /// - parent has no cl_mem (CPU-only buffer)
    /// - offset + length exceeds parent size
    /// - `clCreateSubBuffer` fails (e.g., alignment violation)
    pub fn new(
        parent: Arc<dyn Buffer>,
        offset: usize,
        length: usize,
        dtype: DType,
    ) -> Result<Self> {
        if offset + length > parent.size() {
            return Err(anyhow!(
                "ClSubBuffer out of bounds: offset({}) + length({}) = {} > parent.size({})",
                offset,
                length,
                offset + length,
                parent.size()
            ));
        }

        let parent_cl_mem = parent.cl_mem().ok_or_else(|| {
            anyhow!("ClSubBuffer requires a parent with cl_mem, but parent has None")
        })?;

        // BufferRegion<u8>::new(origin, len) -- origin and len are in units of
        // sizeof::<u8>() == 1, so they are byte values directly.
        let region = BufferRegion::<u8>::new(offset, length);
        let sub_cl_mem = core::create_sub_buffer::<u8>(parent_cl_mem, core::MEM_READ_ONLY, &region)
            .map_err(|e| anyhow!("clCreateSubBuffer failed: {}", e))?;

        Ok(Self {
            parent,
            sub_cl_mem,
            offset,
            length,
            dtype,
        })
    }

    /// Get a reference to the sub-buffer's cl_mem for use in `get_cl_mem()`.
    pub fn cl_mem_ref(&self) -> &Mem {
        &self.sub_cl_mem
    }
}

impl Buffer for ClSubBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.length
    }

    fn as_ptr(&self) -> *const u8 {
        let base = self.parent.as_ptr();
        if base.is_null() {
            return std::ptr::null();
        }
        // SAFETY: offset + length <= parent.size() is guaranteed by constructor.
        unsafe { base.add(self.offset) }
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        let base = self.parent.as_mut_ptr();
        if base.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: offset + length <= parent.size() is guaranteed by constructor.
        unsafe { base.add(self.offset) }
    }

    fn cl_mem(&self) -> Option<&Mem> {
        Some(&self.sub_cl_mem)
    }

    fn sync_device(&self) -> Result<()> {
        self.parent.sync_device()
    }

    fn map_for_cpu(&self) -> Result<()> {
        self.parent.map_for_cpu()
    }

    fn unmap_for_gpu(&self) -> Result<()> {
        self.parent.unmap_for_gpu()
    }

    fn is_mapped(&self) -> bool {
        self.parent.is_mapped()
    }

    fn is_host_managed(&self) -> bool {
        self.parent.is_host_managed()
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}
