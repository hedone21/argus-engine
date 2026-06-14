//! LISWAP-8 Hammer D — CUDA mmap registration + alias buffer.
//!
//! `CudaMmapRegistration` pins the secondary GGUF mmap into a CUDA-
//! addressable region exactly once at swap setup time via
//! `cuMemHostRegister(..., CU_MEMHOSTREGISTER_DEVICEMAP)` and stores the
//! resulting device-pointer base. `Drop` calls `cuMemHostUnregister`.
//!
//! `CudaMmapAliasBuffer` is a read-only `Buffer` view into
//! `[offset..offset+size)` of the registered region. Its `device_ptr()`
//! is suitable for `cuLaunchKernel` / cuBLAS arguments. No `cuMemAlloc`,
//! no `cuMemcpyHtoDAsync`, no CPU memcpy — the swap path can install
//! these as `LayerWeights` tensors via `ArcSwap::store` alone.

#![cfg(feature = "cuda-embedded")]

use crate::buffer::{Buffer, DType};
use crate::memory::secondary::SecondaryMmapBytes;
use anyhow::{Result, anyhow};
use cudarc::driver::sys as cuda_sys;
use std::any::Any;
use std::sync::Arc;

/// Owning handle for a CUDA-registered mmap range. Drop unregisters.
pub struct CudaMmapRegistration {
    /// Kept alive so the underlying mmap pages stay valid for the
    /// duration of every `CudaMmapAliasBuffer` we hand out.
    _secondary: Arc<dyn SecondaryMmapBytes>,
    host_base: *const u8,
    dev_base: cuda_sys::CUdeviceptr,
    size: usize,
}

// SAFETY: All fields are either Arc (Send+Sync), raw pointers shared
// read-only, or numeric types. Access to the mmap pages is read-only
// from this side.
unsafe impl Send for CudaMmapRegistration {}
unsafe impl Sync for CudaMmapRegistration {}

impl CudaMmapRegistration {
    /// Register the full mmap range of `secondary` (GGUF only for the
    /// PoC) with CUDA. Returns an Arc so each alias buffer can keep the
    /// registration alive.
    pub fn register(secondary: Arc<dyn SecondaryMmapBytes>) -> Result<Arc<Self>> {
        let bytes = secondary.raw_bytes().map_err(|e| {
            anyhow!("CudaMmapRegistration: secondary does not expose raw mmap bytes: {e}")
        })?;
        if bytes.is_empty() {
            return Err(anyhow!("CudaMmapRegistration: empty mmap"));
        }
        let host_base = bytes.as_ptr();
        // cuMemHostRegister requires the size to be page-aligned (4 KB on
        // Jetson AArch64). Round down so we never extend past the mmap.
        // Weights are densely packed at the start of GGUF/AUF files so
        // the trailing dropped bytes are typically metadata padding.
        const PAGE: usize = 4096;
        let raw_size = bytes.len();
        let size = raw_size - (raw_size % PAGE);
        if size == 0 {
            return Err(anyhow!(
                "CudaMmapRegistration: mmap shorter than one page ({raw_size} bytes)"
            ));
        }

        // SAFETY: cuMemHostRegister requires a current CUDA context.
        // Caller (generate.rs swap setup) runs this on the main thread
        // which already has the context bound (CudaContext::new).
        //
        // Flag combinations (try in fallback order):
        //   1) DEVICEMAP | READ_ONLY  -- mmap is MAP_PRIVATE read-only,
        //      READ_ONLY tells CUDA we won't write through this mapping.
        //   2) DEVICEMAP | PORTABLE   -- if READ_ONLY isn't supported.
        //   3) DEVICEMAP only         -- last resort.
        let flag_attempts = [
            (
                cuda_sys::CU_MEMHOSTREGISTER_DEVICEMAP | cuda_sys::CU_MEMHOSTREGISTER_READ_ONLY,
                "DEVICEMAP|READ_ONLY",
            ),
            (
                cuda_sys::CU_MEMHOSTREGISTER_DEVICEMAP | cuda_sys::CU_MEMHOSTREGISTER_PORTABLE,
                "DEVICEMAP|PORTABLE",
            ),
            (cuda_sys::CU_MEMHOSTREGISTER_DEVICEMAP, "DEVICEMAP"),
        ];
        let mut last_err = String::new();
        let mut registered = false;
        for (flags, name) in flag_attempts.iter() {
            unsafe {
                let res = cuda_sys::cuMemHostRegister_v2(host_base as *mut _, size, *flags);
                if res == cuda_sys::CUresult::CUDA_SUCCESS {
                    eprintln!(
                        "[CudaMmapRegistration] registered {} MB with flags={}",
                        size / (1024 * 1024),
                        name
                    );
                    registered = true;
                    break;
                } else {
                    last_err = format!("flags={name} -> {:?}", res);
                }
            }
        }
        if !registered {
            return Err(anyhow!(
                "cuMemHostRegister({size} bytes) failed for all flag combinations: {last_err}"
            ));
        }

        let dev_base = unsafe {
            let mut dptr: cuda_sys::CUdeviceptr = 0;
            let res = cuda_sys::cuMemHostGetDevicePointer_v2(&mut dptr, host_base as *mut _, 0);
            if res != cuda_sys::CUresult::CUDA_SUCCESS {
                // Best-effort: unregister before returning the error.
                let _ = cuda_sys::cuMemHostUnregister(host_base as *mut _);
                return Err(anyhow!("cuMemHostGetDevicePointer_v2 failed: {:?}", res));
            }
            dptr
        };

        Ok(Arc::new(Self {
            _secondary: secondary,
            host_base,
            dev_base,
            size,
        }))
    }

    pub fn host_base(&self) -> *const u8 {
        self.host_base
    }

    pub fn dev_base(&self) -> cuda_sys::CUdeviceptr {
        self.dev_base
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for CudaMmapRegistration {
    fn drop(&mut self) {
        // SAFETY: Same pointer we registered. Any aliasing buffers must
        // already be dropped — they hold an `Arc<Self>` so this Drop
        // only runs after the last alias goes away.
        unsafe {
            let _ = cuda_sys::cuMemHostUnregister(self.host_base as *mut _);
        }
    }
}

/// Read-only buffer view into a CUDA-registered mmap range.
pub struct CudaMmapAliasBuffer {
    host_ptr: *const u8,
    dev_ptr: cuda_sys::CUdeviceptr,
    size: usize,
    dtype: DType,
    _registration: Arc<CudaMmapRegistration>,
}

unsafe impl Send for CudaMmapAliasBuffer {}
unsafe impl Sync for CudaMmapAliasBuffer {}

impl CudaMmapAliasBuffer {
    /// Construct an alias for `[offset..offset+size)` within the
    /// registered region. Returns an error if the slice is out of range.
    pub fn new(
        registration: Arc<CudaMmapRegistration>,
        offset: usize,
        size: usize,
        dtype: DType,
    ) -> Result<Self> {
        let total = registration.size();
        let end = offset
            .checked_add(size)
            .ok_or_else(|| anyhow!("CudaMmapAliasBuffer: offset+size overflow"))?;
        if end > total {
            return Err(anyhow!(
                "CudaMmapAliasBuffer: offset {offset} + size {size} > registered size {total}"
            ));
        }
        // SAFETY: offset/size were range-checked against the registered
        // region's length.
        let host_ptr = unsafe { registration.host_base().add(offset) };
        let dev_ptr = registration.dev_base() + offset as cuda_sys::CUdeviceptr;
        Ok(Self {
            host_ptr,
            dev_ptr,
            size,
            dtype,
            _registration: registration,
        })
    }

    pub fn device_ptr(&self) -> cuda_sys::CUdeviceptr {
        self.dev_ptr
    }
}

impl Buffer for CudaMmapAliasBuffer {
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
        self.host_ptr
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

    fn is_host_managed(&self) -> bool {
        false
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}
