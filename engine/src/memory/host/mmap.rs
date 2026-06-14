use crate::buffer::{Buffer, DType};
use anyhow::Result;
#[cfg(feature = "opencl")]
use ocl::core::Mem;
use std::any::Any;
use std::sync::Arc;

/// INV-143 lifetime guard for `MmapBuffer`.
///
/// `MmapBuffer` holds an `Arc<dyn MmapKeepAlive>` to keep its backing mmap (or
/// any owner that contains an mmap region) alive for its full lifetime. Direct
/// `Arc<dyn Any>` would also work but loses type-restriction: with this marker
/// trait only types that explicitly opt in (mmap producers + their wrappers)
/// can serve as the keep-alive.
pub trait MmapKeepAlive: Send + Sync + 'static {}

impl MmapKeepAlive for memmap2::Mmap {}

/// Zero-copy buffer that directly references mmap'd memory.
///
/// Used both for primary weight load (safetensors/GGUF — keep-alive is the raw
/// `memmap2::Mmap`) and for secondary swap borrow (AOS/AUF path — keep-alive is
/// the wrapper `SecondaryMmap`). The two cases differ only in which `Arc<dyn
/// MmapKeepAlive>` is supplied.
///
/// Benefits over SharedBuffer (heap Vec):
/// - Kernel readahead for sequential access (mmap page cache)
/// - Better TLB behavior (kernel can use transparent huge pages)
/// - No memory copy at load time
///
/// Immutable: `as_mut_ptr()` returns the same pointer as `as_ptr()` (matching
/// the read-only-mmap contract). Callers must not mutate through it. Spec:
/// ENG-ALG-227, INV-143.
pub struct MmapBuffer {
    /// Pointer to the start of this tensor's data within the mmap'd region
    ptr: *const u8,
    /// Size in bytes
    size: usize,
    /// Data type
    dtype: DType,
    /// INV-143 lifetime guard. Holds the mmap (directly or via a wrapper) so
    /// the backing pages stay mapped for as long as this buffer is alive.
    _keep_alive: Arc<dyn MmapKeepAlive>,
}

// Safety: MmapBuffer is read-only and the underlying mmap is immutable.
// The Arc<dyn MmapKeepAlive> ensures the mapping lives as long as any
// MmapBuffer references it.
unsafe impl Send for MmapBuffer {}
unsafe impl Sync for MmapBuffer {}

impl MmapBuffer {
    /// Create a new MmapBuffer pointing to a region within an mmap'd file.
    ///
    /// # Safety
    /// `offset + size` must not exceed the mmap length.
    pub unsafe fn new(mmap: Arc<memmap2::Mmap>, offset: usize, size: usize, dtype: DType) -> Self {
        debug_assert!(offset + size <= mmap.len());
        // Safety: caller guarantees offset + size <= mmap.len()
        let ptr = unsafe { mmap.as_ptr().add(offset) };
        Self {
            ptr,
            size,
            dtype,
            _keep_alive: mmap,
        }
    }

    /// Borrow a typed view over an mmap region whose lifetime is anchored by
    /// `keep_alive`. The caller guarantees that `data` is a sub-slice of memory
    /// owned (directly or transitively) by `keep_alive`.
    ///
    /// Replaces the former `BorrowedMmapBuffer::new`. Used by the secondary
    /// swap path where the keep-alive is `Arc<SecondaryMmap>`.
    pub fn borrow<K: MmapKeepAlive>(data: &[u8], dtype: DType, keep_alive: Arc<K>) -> Self {
        Self {
            ptr: data.as_ptr(),
            size: data.len(),
            dtype,
            _keep_alive: keep_alive,
        }
    }
}

impl Buffer for MmapBuffer {
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
        // Weights are read-only — mutation not supported
        self.ptr as *mut u8
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        None // CPU-only buffer
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        Ok(())
    }
}
