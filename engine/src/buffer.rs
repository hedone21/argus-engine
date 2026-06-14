// ============================================================================
// Buffer sub-modules. Moved from `buffer/mod.rs` to the top-level
// `engine/src/buffer.rs` per Step 4-A (Rust 2018+ pattern).
//
// Generic L2 buffer views:
// - [`slice`]: `SliceBuffer` — typed offset view over any parent `Arc<dyn Buffer>`.
//
// Memory-resource-specific allocators and view adapters live in
// `crate::memory::<resource>/`. See ARCHITECTURE.md §13.8-D.
// ============================================================================

pub mod auf_view_buffer;
pub mod opaque;
pub mod slice;

// ============================================================================
// Buffer trait + DType (originally `core/buffer.rs`).
// ============================================================================

use anyhow::Result;
#[cfg(feature = "opencl")]
use ocl::core::Mem;

/// Data Type Enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Q4_0,
    Q4_1,
    Q8_0,
    F16,
    BF16,
    F32,
    U8,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::Q4_0 | DType::Q4_1 | DType::Q8_0 => 1, // Actually block quantized, handled separately usually
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            DType::U8 => 1,
        }
    }
}

use std::any::Any;

/// Buffer Trait: Physical Memory Layer
pub trait Buffer: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    /// Returns the data type
    fn dtype(&self) -> DType;

    /// Returns the total size in bytes
    fn size(&self) -> usize;

    /// Read-only pointer for CPU access
    fn as_ptr(&self) -> *const u8;

    /// Mutable pointer for CPU access
    fn as_mut_ptr(&self) -> *mut u8;

    /// OpenCL memory handle (None for CPU-only buffers)
    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem>;

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()>;

    /// Ensure data is synced to the desired device
    fn sync_device(&self) -> Result<()>;

    // --- Zero-Copy Shared Memory Support ---

    /// Map buffer for CPU access (makes GPU access invalid until unmapped).
    /// Default implementation is no-op for CPU-only buffers.
    fn map_for_cpu(&self) -> Result<()> {
        Ok(()) // No-op for regular buffers
    }

    /// Unmap buffer for GPU access (makes CPU pointer invalid).
    /// Default implementation is no-op for CPU-only buffers.
    fn unmap_for_gpu(&self) -> Result<()> {
        Ok(()) // No-op for regular buffers
    }

    /// Check if buffer is currently mapped for CPU access.
    /// Default: true (CPU-only buffers are always "mapped").
    fn is_mapped(&self) -> bool {
        true
    }

    /// Whether the host (app) manages the backing memory so that madvise is effective.
    /// true: SharedBuffer, MmapBuffer (app-managed mmap/heap)
    /// false: UnifiedBuffer (driver-pinned), OpenCLBuffer (device-only)
    fn is_host_managed(&self) -> bool {
        true // default: CPU buffers are host-managed
    }

    /// 이 버퍼가 GPU에 접근 가능한지 여부.
    /// OpenCL cl_mem 또는 CUDA device pointer를 가진 버퍼는 true.
    /// CPU-only 버퍼(SharedBuffer, MmapBuffer)는 false.
    fn is_gpu_buffer(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyBuffer;
    impl Buffer for DummyBuffer {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn size(&self) -> usize {
            1024
        }
        fn as_ptr(&self) -> *const u8 {
            std::ptr::null()
        }
        fn as_mut_ptr(&self) -> *mut u8 {
            std::ptr::null_mut()
        }
        #[cfg(feature = "opencl")]
        fn cl_mem(&self) -> Option<&ocl::core::Mem> {
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

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::BF16.size(), 2);
        assert_eq!(DType::U8.size(), 1);
        assert_eq!(DType::Q4_0.size(), 1); // 1 byte reported per struct/type enum representation although mapped as blocks
        assert_eq!(DType::Q4_1.size(), 1);
    }

    #[test]
    fn test_buffer_default_impls() {
        let buffer = DummyBuffer;
        assert!(buffer.map_for_cpu().is_ok());
        assert!(buffer.unmap_for_gpu().is_ok());
        assert!(buffer.is_mapped()); // Defaults to true
    }

    #[test]
    fn test_dtype_all_variant_sizes() {
        // Exhaustive match ensures we cover every DType variant
        let variants = [
            DType::Q4_0,
            DType::Q4_1,
            DType::Q8_0,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::U8,
        ];
        let expected = [1, 1, 1, 2, 2, 4, 1];
        for (dt, exp) in variants.iter().zip(expected.iter()) {
            assert_eq!(dt.size(), *exp, "DType::{:?} size mismatch", dt);
        }
    }

    #[test]
    fn test_dtype_equality_and_copy() {
        let a = DType::F32;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_eq!(a, DType::F32);
        assert_ne!(a, DType::F16);
        assert_ne!(DType::Q4_0, DType::Q4_1);

        // Copy produces identical value
        let c = a;
        assert_eq!(a, c);
    }

    #[test]
    fn test_buffer_metadata_accessors() {
        let buffer = DummyBuffer;
        assert_eq!(buffer.dtype(), DType::F32);
        assert_eq!(buffer.size(), 1024);
        assert!(buffer.as_ptr().is_null());
        assert!(buffer.as_mut_ptr().is_null());
        assert!(buffer.cl_mem().is_none());
        assert!(buffer.sync_device().is_ok());
    }
}
