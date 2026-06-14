//! `RpcmemKvBuffer` — rpcmem + OpenCL `CL_MEM_USE_HOST_PTR` dual-backed KV
//! cache buffer (backend-agnostic version of the older
//! `backend/qnn_oppkg/kv_buffer.rs::QnnOppkgKvBuffer`).
//!
//! Spec: `spec/30-engine.md` 부록 E.3 (ENG-RPCMEM-022).
//! Arch: `arch/rpcmem_allocator.md` §3.2, `arch/opencl_backend.md` §3.
//!
//! ## 동작 요약
//! - `host_ptr`: `RpcmemAllocator::alloc` 가 반환한 rpcmem heap 가상 주소.
//!   DMA-BUF 라서 CPU 와 GPU/DSP 가 동일 물리 페이지를 공유 (Adreno UMA).
//! - `cl_mem_obj`: 동일 host_ptr 를 가리키는 OpenCL `CL_MEM_USE_HOST_PTR`
//!   alias. zero-copy.
//! - Drop 시 alias 가 먼저 release 된 후 host_ptr 가 free 되어야 한다
//!   (alias 가 가리키는 페이지가 살아있어야 driver 가 안전하게 해제 가능).
//!
//! ## Drop 순서 (중요)
//!
//! Rust 의 field drop 순서는 **선언 순서** 다. 본 struct 는 다음 두 가지 안전성
//! 약속을 만족해야 한다:
//!
//! 1. `cl_mem_obj` (USE_HOST_PTR alias) 가 `host_ptr` 보다 먼저 drop 되어야 한다.
//!    → 따라서 `cl_mem_obj` 를 `_allocator` (host_ptr 의 lifecycle 을 결정) 보다
//!    먼저 선언한다.
//! 2. `_allocator` (Arc<RpcmemAllocator>) 가 fn-pointer 의 lifetime 을 type
//!    system 으로 강제. allocator strong_count 가 0 이 되어야 dlclose 가 호출
//!    되므로 본 buffer 가 살아있는 동안 allocator 도 살아있다 (INV-RPCMEM-005).

use crate::buffer::{Buffer, DType};
use crate::memory::rpcmem::allocator::RpcmemAllocator;
use anyhow::Result;
use std::any::Any;
use std::sync::Arc;

/// rpcmem + OpenCL `CL_MEM_USE_HOST_PTR` dual-backed KV cache buffer.
///
/// `cl_mem()` 이 `Some(...)` 를 반환하므로 OpenCL backend 의 `read_buffer` /
/// `write_buffer` 가 자동으로 호환된다. `as_ptr()` 는 host_ptr 을 반환하여
/// eviction / KV migrate 정책도 가능.
pub struct RpcmemKvBuffer {
    // DROP ORDER (선언 순서 == drop 순서):
    //   1. cl_mem_obj  — OpenCL alias 가 가리키는 host_ptr 페이지보다 먼저 release.
    //   2. _allocator  — Arc<RpcmemAllocator> strong count 감소. count == 0 시점에
    //                     도 free_fn 은 host_ptr 의 owner (본 struct) 가 별도로
    //                     수행 (Drop impl 의 host_ptr free 가 이미 수행됨).
    //
    // `host_ptr` 자체는 raw 포인터라 Drop 순서에 영향 없음 (manual free).
    /// CL_MEM_USE_HOST_PTR alias — 동일 host_ptr 페이지 참조.
    /// 반드시 host_ptr free 보다 먼저 drop 되어야 한다.
    cl_mem_obj: ocl::core::Mem,

    /// rpcmem allocator — Arc 보유로 lifetime 을 type system 강제
    /// (INV-RPCMEM-005). Drop impl 에서 `_allocator.free(host_ptr)` 호출.
    _allocator: Arc<RpcmemAllocator>,

    /// `RpcmemAllocator::alloc` 가 반환한 host-mapped 가상 주소.
    host_ptr: *mut u8,

    /// `rpcmem_to_fd` 결과 — Sprint 2c 의 future graph external buffer 주입을
    /// 위해 보관 (현 production path 는 cl_mem alias 만 사용).
    #[allow(dead_code)]
    fd: std::os::unix::io::RawFd,

    size: usize,
    dtype: DType,
}

// SAFETY: production OpenCL queue 는 단일 스레드 접근 보장. rpcmem heap 은
// CPU/GPU/DSP 가 cache coherent 하게 공유 (Adreno UMA). Arc<RpcmemAllocator>
// 는 Send + Sync.
unsafe impl Send for RpcmemKvBuffer {}
unsafe impl Sync for RpcmemKvBuffer {}

impl RpcmemKvBuffer {
    /// Construct from an existing rpcmem allocation + matching cl_mem alias.
    ///
    /// Caller (`OpenCLMemory::alloc_kv` rpcmem path) is responsible for:
    /// 1. `allocator.alloc(size)` → `(host_ptr, fd)`.
    /// 2. `clCreateBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size, host_ptr)`.
    /// 3. Construct via this method, passing the cloned `Arc<RpcmemAllocator>`.
    pub fn new(
        host_ptr: *mut u8,
        fd: std::os::unix::io::RawFd,
        cl_mem_obj: ocl::core::Mem,
        size: usize,
        dtype: DType,
        allocator: Arc<RpcmemAllocator>,
    ) -> Self {
        Self {
            cl_mem_obj,
            _allocator: allocator,
            host_ptr,
            fd,
            size,
            dtype,
        }
    }
}

impl Drop for RpcmemKvBuffer {
    fn drop(&mut self) {
        // SAFETY: host_ptr is the pointer returned by self._allocator.alloc.
        // Drop runs once. cl_mem_obj has just been dropped (field drop order)
        // so no live OpenCL alias remains on the rpcmem region.
        if !self.host_ptr.is_null() {
            unsafe { self._allocator.free(self.host_ptr) };
        }
    }
}

impl Buffer for RpcmemKvBuffer {
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
        self.host_ptr as *const u8
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.host_ptr
    }

    fn cl_mem(&self) -> Option<&ocl::core::Mem> {
        Some(&self.cl_mem_obj)
    }

    fn sync_device(&self) -> Result<()> {
        // rpcmem DMA-BUF + USE_HOST_PTR alias — Adreno UMA 에서 같은 물리 페이지
        // 공유. 추가 cache flush 불필요 (matches QnnOppkgKvBuffer semantics).
        Ok(())
    }

    fn is_host_managed(&self) -> bool {
        true
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}
