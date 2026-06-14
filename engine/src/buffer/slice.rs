use crate::buffer::{Buffer, DType};
use anyhow::{Result, ensure};
#[cfg(feature = "opencl")]
use ocl::core::Mem;
use std::any::Any;
use std::sync::Arc;

/// Zero-copy sub-view of an existing Buffer.
///
/// Borrows a byte range `[offset, offset+length)` from a parent buffer
/// without copying data. The parent is kept alive via `Arc`.
///
/// # Safety
/// Pointer arithmetic in `as_ptr` / `as_mut_ptr` is safe as long as
/// `offset + length <= inner.size()`, which is enforced at construction time.
pub struct SliceBuffer {
    inner: Arc<dyn Buffer>,
    offset: usize,
    length: usize,
    dtype: DType,
}

impl SliceBuffer {
    /// Create a new SliceBuffer referencing `inner[offset..offset+length]`.
    ///
    /// Returns `Err` if the requested range exceeds the parent buffer size.
    pub fn new(inner: Arc<dyn Buffer>, offset: usize, length: usize, dtype: DType) -> Result<Self> {
        ensure!(
            offset + length <= inner.size(),
            "SliceBuffer out of bounds: offset({}) + length({}) = {} > inner.size({})",
            offset,
            length,
            offset + length,
            inner.size()
        );
        Ok(Self {
            inner,
            offset,
            length,
            dtype,
        })
    }
}

impl Buffer for SliceBuffer {
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
        let base = self.inner.as_ptr();
        if base.is_null() {
            return std::ptr::null();
        }
        // SAFETY: offset + length <= inner.size() is guaranteed by constructor.
        unsafe { base.add(self.offset) }
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        let base = self.inner.as_mut_ptr();
        if base.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: offset + length <= inner.size() is guaranteed by constructor.
        unsafe { base.add(self.offset) }
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        // SliceBuffer cannot apply an offset to a raw cl_mem handle.
        // GPU kernels that need sub-buffer access must use clCreateSubBuffer
        // or pass the byte offset as a kernel argument. For now, return None
        // so that callers fall back to CPU pointer access.
        None
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        self.inner.sync_device()
    }

    fn map_for_cpu(&self) -> Result<()> {
        self.inner.map_for_cpu()
    }

    fn unmap_for_gpu(&self) -> Result<()> {
        self.inner.unmap_for_gpu()
    }

    fn is_mapped(&self) -> bool {
        self.inner.is_mapped()
    }

    fn is_host_managed(&self) -> bool {
        self.inner.is_host_managed()
    }

    fn is_gpu_buffer(&self) -> bool {
        self.inner.is_gpu_buffer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::host::shared::SharedBuffer;

    fn make_buffer(size: usize, dtype: DType) -> Arc<dyn Buffer> {
        Arc::new(SharedBuffer::new(size, dtype))
    }

    /// PA-T1-01: offset=0, full range slice has same pointer as inner.
    #[test]
    fn test_full_range() {
        let inner = make_buffer(100, DType::F32);
        let slice = SliceBuffer::new(inner.clone(), 0, 100, DType::F32).unwrap();
        assert_eq!(slice.as_ptr(), inner.as_ptr());
        assert_eq!(slice.size(), 100);
    }

    /// PA-T1-02: offset=N yields pointer shifted by N bytes.
    #[test]
    fn test_offset_ptr() {
        let inner = make_buffer(100, DType::F32);
        let base = inner.as_ptr();
        let slice = SliceBuffer::new(inner.clone(), 32, 68, DType::F32).unwrap();
        // SAFETY: just comparing pointer values, not dereferencing.
        assert_eq!(slice.as_ptr(), unsafe { base.add(32) });
        assert_eq!(slice.as_mut_ptr(), unsafe { base.add(32) } as *mut u8);
    }

    /// PA-T1-03: out of bounds returns Err.
    #[test]
    fn test_out_of_bounds() {
        let inner = make_buffer(100, DType::F32);
        let result = SliceBuffer::new(inner, 50, 60, DType::F32);
        assert!(result.is_err());
    }

    /// PA-T1-04: dtype propagation.
    #[test]
    fn test_dtype_propagation() {
        let inner = make_buffer(100, DType::U8);
        // SliceBuffer can carry a different dtype label from the inner buffer
        // (e.g. when slicing a Q4_0 weight region that was allocated as raw bytes).
        let slice = SliceBuffer::new(inner, 0, 100, DType::F32).unwrap();
        assert_eq!(slice.dtype(), DType::F32);
    }

    /// PA-T1-05: size equals the requested length.
    #[test]
    fn test_size() {
        let inner = make_buffer(100, DType::F32);
        let slice = SliceBuffer::new(inner, 10, 50, DType::F32).unwrap();
        assert_eq!(slice.size(), 50);
    }

    /// Boundary: offset == inner.size(), length == 0 is valid (empty slice).
    #[test]
    fn test_zero_length_at_end() {
        let inner = make_buffer(64, DType::F32);
        let slice = SliceBuffer::new(inner, 64, 0, DType::F32).unwrap();
        assert_eq!(slice.size(), 0);
    }

    /// cl_mem returns None for a CPU-backed slice.
    #[test]
    fn test_cl_mem_none() {
        let inner = make_buffer(32, DType::F32);
        let slice = SliceBuffer::new(inner, 0, 32, DType::F32).unwrap();
        assert!(slice.cl_mem().is_none());
    }

    /// sync_device / map_for_cpu / unmap_for_gpu delegate to inner.
    #[test]
    fn test_delegation_methods() {
        let inner = make_buffer(32, DType::F32);
        let slice = SliceBuffer::new(inner, 0, 32, DType::F32).unwrap();
        assert!(slice.sync_device().is_ok());
        assert!(slice.map_for_cpu().is_ok());
        assert!(slice.unmap_for_gpu().is_ok());
        assert!(slice.is_mapped());
        assert!(slice.is_host_managed());
        assert!(!slice.is_gpu_buffer());
    }

    /// Data written through the parent is visible through the slice.
    #[test]
    fn test_data_visibility() {
        let inner = make_buffer(16, DType::U8);
        // Write known bytes at offset 4..8
        unsafe {
            let p = inner.as_mut_ptr();
            *p.add(4) = 0xAA;
            *p.add(5) = 0xBB;
            *p.add(6) = 0xCC;
            *p.add(7) = 0xDD;
        }
        let slice = SliceBuffer::new(inner, 4, 4, DType::U8).unwrap();
        unsafe {
            assert_eq!(*slice.as_ptr().add(0), 0xAA);
            assert_eq!(*slice.as_ptr().add(1), 0xBB);
            assert_eq!(*slice.as_ptr().add(2), 0xCC);
            assert_eq!(*slice.as_ptr().add(3), 0xDD);
        }
    }
}
