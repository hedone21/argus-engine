//! CUDA Unified Memory buffer for Jetson UMA devices.
//!
//! Uses `cuMemAllocManaged` (via cudarc's `result` API) with `CU_MEM_ATTACH_GLOBAL`
//! so the returned pointer is accessible from both CPU and GPU without explicit copies.
//! This is the CUDA equivalent of OpenCL's `CL_MEM_ALLOC_HOST_PTR` zero-copy on Adreno.

use crate::buffer::{Buffer, DType};
use anyhow::{Result, anyhow};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};
use std::any::Any;

/// A CUDA Unified Memory buffer.
///
/// Manages a raw `CUdeviceptr` allocated with `cuMemAllocManaged`.
/// On Jetson (UMA), this pointer is directly dereferenceable from CPU code.
/// Drop calls `cuMemFree` to release the allocation.
pub struct CudaBuffer {
    /// Raw CUDA device pointer (also valid as host pointer on UMA).
    dev_ptr: cuda_sys::CUdeviceptr,
    /// Total allocation size in bytes.
    size: usize,
    /// Logical data type for this buffer.
    dtype: DType,
}

// SAFETY: The underlying CUDA unified memory is thread-safe.
// cuMemAllocManaged with CU_MEM_ATTACH_GLOBAL makes the pointer accessible
// from any thread/stream. Access synchronization is the caller's responsibility
// (same as OpenCL zero-copy buffers).
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocate `size` bytes of CUDA Unified Memory.
    ///
    /// Uses `CU_MEM_ATTACH_GLOBAL` so any stream/device can access the memory.
    /// On Jetson (UMA), the pointer is CPU-accessible without mapping.
    ///
    /// # Requirements
    /// A CUDA context must be current on the calling thread (typically set up
    /// by `CudaContext::new()` in `CudaBackend::new()`).
    pub fn new(size: usize, dtype: DType) -> Result<Self> {
        if size == 0 {
            return Err(anyhow!("CudaBuffer: cannot allocate 0 bytes"));
        }
        // SAFETY: cuMemAllocManaged requires a valid CUDA context on the current thread.
        // CudaBackend::new() ensures this via CudaContext::new() which calls cuInit+cuCtxCreate.
        // The returned pointer is uninitialized -- callers must write before reading.
        let dev_ptr = unsafe {
            cuda_result::malloc_managed(size, cuda_sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL)
                .map_err(|e| anyhow!("cuMemAllocManaged({size} bytes) failed: {e}"))?
        };
        // Zero-initialize: CUDA managed memory is uninitialized by default.
        // Many code paths assume buffers start at zero (KV cache, workspace, etc).
        unsafe {
            std::ptr::write_bytes(dev_ptr as *mut u8, 0, size);
        }
        Ok(Self {
            dev_ptr,
            size,
            dtype,
        })
    }

    /// Return the raw `CUdeviceptr` value (for passing to CUDA kernels).
    pub fn device_ptr(&self) -> cuda_sys::CUdeviceptr {
        self.dev_ptr
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // SAFETY: We own this allocation (created in new()) and only free it once (here in Drop).
        // Any async work must have been synchronized before dropping.
        unsafe {
            let _ = cuda_result::memory_free(self.dev_ptr);
        }
    }
}

impl Buffer for CudaBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    /// CPU-accessible pointer on UMA (Jetson).
    ///
    /// SAFETY: On Jetson (UMA), `CUdeviceptr` from `cuMemAllocManaged` is a valid
    /// host pointer. On discrete GPUs this would segfault -- but Phase 1 targets UMA only.
    fn as_ptr(&self) -> *const u8 {
        self.dev_ptr as *const u8
    }

    /// Mutable CPU-accessible pointer on UMA (Jetson).
    fn as_mut_ptr(&self) -> *mut u8 {
        self.dev_ptr as *mut u8
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&ocl::core::Mem> {
        None
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    /// On UMA, host/device coherence is automatic; no explicit sync needed
    /// for data visibility. Kernel completion is handled by `Backend::synchronize()`.
    fn sync_device(&self) -> Result<()> {
        Ok(())
    }

    /// UMA: madvise is NOT effective on driver-managed unified memory.
    fn is_host_managed(&self) -> bool {
        false
    }

    /// This buffer is GPU-accessible.
    fn is_gpu_buffer(&self) -> bool {
        true
    }
}

/// A CUDA pinned host memory buffer with GPU-mapped device pointer.
///
/// Uses `cuMemHostAlloc` with `CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE`
/// to allocate page-locked host memory that is also GPU-accessible via a device pointer.
/// On Jetson (UMA), the host and device pointers refer to the same physical DRAM.
///
/// Key properties:
/// - `as_ptr()` returns the host pointer for zero-cost CPU access
/// - `device_ptr()` returns the `CUdeviceptr` for cuBLAS/CUDA kernel calls
/// - Drop calls `cuMemFreeHost` to release the allocation
pub struct CudaHostBuffer {
    /// CPU-accessible host pointer (page-locked).
    host_ptr: *mut u8,
    /// GPU-accessible device pointer (mapped from host_ptr).
    dev_ptr: cuda_sys::CUdeviceptr,
    /// Total allocation size in bytes.
    size: usize,
    /// Logical data type for this buffer.
    dtype: DType,
}

// SAFETY: The underlying pinned memory with CU_MEMHOSTALLOC_PORTABLE is accessible
// from any CUDA context and any CPU thread. Access synchronization is the caller's
// responsibility (same as CudaBuffer and OpenCL zero-copy buffers).
unsafe impl Send for CudaHostBuffer {}
unsafe impl Sync for CudaHostBuffer {}

impl CudaHostBuffer {
    /// Allocate `size` bytes of pinned host memory with GPU mapping.
    ///
    /// Flags: `CU_MEMHOSTALLOC_DEVICEMAP` (GPU can access via device pointer)
    ///      + `CU_MEMHOSTALLOC_PORTABLE` (accessible from any CUDA context).
    ///
    /// # Requirements
    /// A CUDA context must be current on the calling thread.
    pub fn new(size: usize, dtype: DType) -> Result<Self> {
        if size == 0 {
            return Err(anyhow!("CudaHostBuffer: cannot allocate 0 bytes"));
        }

        let flags = cuda_sys::CU_MEMHOSTALLOC_DEVICEMAP | cuda_sys::CU_MEMHOSTALLOC_PORTABLE;

        // SAFETY: cuMemHostAlloc requires a valid CUDA context on the current thread.
        // CudaBackend::new() ensures this. The returned pointer is uninitialized.
        let host_ptr = unsafe {
            cuda_result::malloc_host(size, flags)
                .map_err(|e| anyhow!("cuMemHostAlloc({size} bytes) failed: {e}"))?
        } as *mut u8;

        // Get the device pointer mapped to this host allocation.
        // SAFETY: host_ptr was just allocated with CU_MEMHOSTALLOC_DEVICEMAP,
        // so cuMemHostGetDevicePointer_v2 is guaranteed to succeed.
        let dev_ptr = {
            let mut dptr: cuda_sys::CUdeviceptr = 0;
            let result = unsafe {
                cuda_sys::cuMemHostGetDevicePointer_v2(
                    &mut dptr,
                    host_ptr as *mut std::ffi::c_void,
                    0,
                )
            };
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                // Clean up host allocation on failure
                unsafe {
                    let _ = cuda_result::free_host(host_ptr as *mut std::ffi::c_void);
                }
                return Err(anyhow!("cuMemHostGetDevicePointer_v2 failed: {:?}", result));
            }
            dptr
        };

        // Zero-initialize: pinned memory is uninitialized by default.
        // Many code paths assume buffers start at zero (KV cache, workspace, etc).
        unsafe {
            std::ptr::write_bytes(host_ptr, 0, size);
        }

        Ok(Self {
            host_ptr,
            dev_ptr,
            size,
            dtype,
        })
    }

    /// Return the raw `CUdeviceptr` for passing to cuBLAS/CUDA kernels.
    pub fn device_ptr(&self) -> cuda_sys::CUdeviceptr {
        self.dev_ptr
    }
}

impl Drop for CudaHostBuffer {
    fn drop(&mut self) {
        // SAFETY: We own this allocation (created in new()) and only free it once (here in Drop).
        // Any async work must have been synchronized before dropping.
        unsafe {
            let _ = cuda_result::free_host(self.host_ptr as *mut std::ffi::c_void);
        }
    }
}

impl Buffer for CudaHostBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    /// CPU-accessible host pointer (page-locked, zero-cost access).
    fn as_ptr(&self) -> *const u8 {
        self.host_ptr
    }

    /// Mutable CPU-accessible host pointer.
    fn as_mut_ptr(&self) -> *mut u8 {
        self.host_ptr
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&ocl::core::Mem> {
        None
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    /// On UMA (Jetson), host/device coherence is automatic for pinned memory.
    fn sync_device(&self) -> Result<()> {
        Ok(())
    }

    /// Pinned memory is driver-managed, not eligible for madvise.
    fn is_host_managed(&self) -> bool {
        false
    }

    /// This buffer is GPU-accessible via its device pointer.
    fn is_gpu_buffer(&self) -> bool {
        true
    }
}

/// Pure device memory buffer (VRAM on discrete GPU).
///
/// Uses `cuMemAlloc` to allocate memory in GPU VRAM. Not host-accessible —
/// `as_ptr()` returns null. Use `device_ptr()` for GPU kernel / cuBLAS calls.
///
/// This is optimal for read-only weight tensors on discrete GPUs where
/// pinned host memory (CudaHostBuffer) would force every read over PCIe.
pub struct CudaDeviceBuffer {
    dev_ptr: cuda_sys::CUdeviceptr,
    size: usize,
    dtype: DType,
}

unsafe impl Send for CudaDeviceBuffer {}
unsafe impl Sync for CudaDeviceBuffer {}

impl CudaDeviceBuffer {
    /// Allocate `size` bytes of device memory (VRAM).
    pub fn new(size: usize, dtype: DType) -> Result<Self> {
        if size == 0 {
            return Err(anyhow!("CudaDeviceBuffer: cannot allocate 0 bytes"));
        }
        let dev_ptr = unsafe {
            let mut ptr: cuda_sys::CUdeviceptr = 0;
            let res = cuda_sys::cuMemAlloc_v2(&mut ptr, size);
            if res != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuMemAlloc({size} bytes) failed: {:?}", res));
            }
            ptr
        };
        Ok(Self {
            dev_ptr,
            size,
            dtype,
        })
    }

    /// Copy data from host to this device buffer.
    pub fn copy_from_host(&self, src: *const u8, len: usize) -> Result<()> {
        if len > self.size {
            return Err(anyhow!(
                "CudaDeviceBuffer::copy_from_host: len {len} > size {}",
                self.size
            ));
        }
        unsafe {
            let res = cuda_sys::cuMemcpyHtoD_v2(self.dev_ptr, src as *const std::ffi::c_void, len);
            if res != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuMemcpyHtoD failed: {:?}", res));
            }
        }
        Ok(())
    }

    /// Async H2D copy on `stream`. Returns immediately after enqueue;
    /// caller must synchronise the stream (or an event recorded after
    /// the copy) before the device buffer is safe to read from compute.
    ///
    /// LISWAP-2 prototype (plan: chasing-hopper). Used by the CUDA
    /// backend's `enqueue_write_async` to overlap weight uploads with
    /// the compute stream's forward kernels.
    ///
    /// # Safety
    ///
    /// Caller must ensure the bytes at `src..src+len` remain valid until
    /// the stream-recorded write actually fires (typically by holding
    /// the source weight tensor alive across the dispatcher commit).
    pub fn copy_from_host_async(
        &self,
        src: *const u8,
        len: usize,
        stream: cuda_sys::CUstream,
    ) -> Result<()> {
        if len > self.size {
            return Err(anyhow!(
                "CudaDeviceBuffer::copy_from_host_async: len {len} > size {}",
                self.size
            ));
        }
        unsafe {
            let res = cuda_sys::cuMemcpyHtoDAsync_v2(
                self.dev_ptr,
                src as *const std::ffi::c_void,
                len,
                stream,
            );
            if res != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuMemcpyHtoDAsync failed: {:?}", res));
            }
        }
        Ok(())
    }

    /// Copy data from this device buffer to host.
    pub fn copy_to_host(&self, dst: *mut u8, len: usize) -> Result<()> {
        if len > self.size {
            return Err(anyhow!(
                "CudaDeviceBuffer::copy_to_host: len {len} > size {}",
                self.size
            ));
        }
        unsafe {
            let res = cuda_sys::cuMemcpyDtoH_v2(dst as *mut std::ffi::c_void, self.dev_ptr, len);
            if res != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuMemcpyDtoH failed: {:?}", res));
            }
        }
        Ok(())
    }

    pub fn device_ptr(&self) -> cuda_sys::CUdeviceptr {
        self.dev_ptr
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_sys::cuMemFree_v2(self.dev_ptr);
        }
    }
}

impl Buffer for CudaDeviceBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    /// Not host-accessible. Returns null.
    fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    /// Not host-accessible. Returns null.
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

    fn is_host_managed(&self) -> bool {
        false
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Allocation tests require a real CUDA device.
    // Run with: cargo test --no-default-features --features cuda -- cuda_buffer

    #[test]
    fn test_cuda_buffer_dtype_accessors() {
        // Unit test that doesn't need a real device -- just checks compile-time logic.
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F16.size(), 2);
    }
}
