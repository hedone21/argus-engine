//! `RpcmemAllocator` — backend-agnostic rpcmem (DMA-BUF heap) allocator.
//!
//! `libcdsprpc.so` 의 3 심볼 (`rpcmem_alloc` / `rpcmem_free` / `rpcmem_to_fd`)
//! 을 단일 책임으로 caching 하는 L2 모듈. OpenCL backend 의
//! `--opencl-rpcmem` 활성 시 KV cache zero-copy + RpcmemSecondaryStore
//! precision swap 두 consumer 에게 동일 `Arc<RpcmemAllocator>` 인스턴스를
//! 주입한다 (INV-RPCMEM-002).
//!
//! Spec: `spec/30-engine.md` 부록 E.2 (ENG-RPCMEM-010 ~ ENG-RPCMEM-013) +
//! `spec/41-invariants.md` §3.27 (INV-RPCMEM-001/004/005).
//! Arch: `arch/rpcmem_allocator.md`.
//!
//! ## Drop 순서
//!
//! `RpcmemAllocator` 의 Drop 은 `libloading::Library` 를 release (dlclose) 만
//! 수행한다. outstanding host_ptr 의 `rpcmem_free` 는 호출하지 않는다.
//! `Arc<RpcmemAllocator>` 가 모든 buffer struct 의 field 로 보유되므로 buffer
//! 가 먼저 drop 된 후에야 allocator strong count 가 0 으로 떨어진다
//! (INV-RPCMEM-005).
//!
//! ## INV-RPCMEM-C01 / -004
//!
//! 본 모듈은 QNN GPU / QNN OpPackage 관련 shared library 를 dlopen 하지 않는다.
//! libcdsprpc.so 단독 의존이며, source-grep test (INV-RPCMEM-004) 가 본 사실을
//! 강제한다 (ENG-RPCMEM-C01).

use std::os::unix::io::RawFd;

/// rpcmem 친화적인 alloc 시 사용하는 fn-pointer (libcdsprpc.so `rpcmem_alloc`).
///
/// (heap_id, flags, size) → host_ptr (NULL on failure).
pub type RpcmemAllocFn = unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void;

/// `rpcmem_free` fn-pointer.
pub type RpcmemFreeFn = unsafe extern "C" fn(*mut std::ffi::c_void);

/// `rpcmem_to_fd` fn-pointer — DMA-BUF fd 추출.
pub type RpcmemToFdFn = unsafe extern "C" fn(*const std::ffi::c_void) -> i32;

/// Default heap id (`RPCMEM_HEAP_ID_SYSTEM`) + default flags. callsite 가
/// override 가능하지만 본 sprint 의 KV/secondary 두 consumer 모두 동일 값을
/// 사용하므로 module-level const 로 노출.
pub const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
/// `RPCMEM_DEFAULT_FLAGS` — Qualcomm Adreno driver 문서 권장값.
pub const RPCMEM_DEFAULT_FLAGS: u32 = 1;

/// rpcmem allocator.
///
/// 두 variant:
/// - `OwnedDlopen`: 본 모듈이 직접 `libcdsprpc.so` 를 dlopen 한 경우.
///   `_lib` field 가 Drop 시 dlclose 를 수행하므로 fn-pointer 보다 마지막에
///   drop 되어야 한다 (struct field 선언 순서가 강제). Android only.
/// - `ExternalFns`: Sprint 2a 공존 path — 기존 `QnnOppkgRuntime` 이 보유한
///   fn-pointer 를 그대로 빌려 쓰는 wrapper. 이 variant 는 `Library` 핸들을
///   소유하지 않으며, dlclose 는 `QnnOppkgRuntime::Drop` 시점에 수행된다.
///   Sprint 2c 에서 backend 가 삭제되면서 자연 소멸.
pub enum RpcmemAllocator {
    /// 본 모듈이 dlopen 한 case. Android only — host build 에선 본 variant
    /// 를 생성할 수 없다 (`new()` 가 Err).
    #[cfg(target_os = "android")]
    OwnedDlopen {
        // Drop 순서가 fn-pointer lifetime 을 결정 — `_lib` 가 마지막 field 라
        // 가장 늦게 drop 된다 (RFC 1857 / Rust drop order: 선언 역순).
        rpcmem_alloc: RpcmemAllocFn,
        rpcmem_free: RpcmemFreeFn,
        rpcmem_to_fd: RpcmemToFdFn,
        _lib: libloading::Library,
    },

    /// External fn-pointer wrapper. 본 variant 는 dlclose 책임이 없으며 단순
    /// fn-pointer container. 호스트/디바이스 모두 컴파일 가능 (테스트/공존용).
    ExternalFns {
        rpcmem_alloc: RpcmemAllocFn,
        rpcmem_free: RpcmemFreeFn,
        rpcmem_to_fd: RpcmemToFdFn,
    },
}

// SAFETY: 모든 variant 의 field 는 `Send + Sync` — fn-pointer 는 Copy +
// stateless, libloading::Library 는 internally Sync (Drop 만 mutates).
unsafe impl Send for RpcmemAllocator {}
unsafe impl Sync for RpcmemAllocator {}

impl RpcmemAllocator {
    /// `libcdsprpc.so` dlopen + 3 심볼 lookup (Android only).
    ///
    /// 환경 변수 `LLMRS_RPCMEM_LIB` 로 path override 가능
    /// (default `/vendor/lib64/libcdsprpc.so` — qnn_oppkg::runtime 의
    /// `QNN_RPCMEM_LIB` 와 동일).
    ///
    /// # Errors
    /// - host (non-Android) target: 즉시 Err (INV-RPCMEM-001).
    /// - dlopen 실패 (so 부재): Err with context.
    /// - 심볼 lookup 실패: Err with context.
    #[cfg(target_os = "android")]
    pub fn new() -> anyhow::Result<Self> {
        use anyhow::anyhow;
        use libloading::Symbol;

        let lib_path = std::env::var("LLMRS_RPCMEM_LIB")
            .or_else(|_| std::env::var("QNN_RPCMEM_LIB"))
            .unwrap_or_else(|_| "/vendor/lib64/libcdsprpc.so".to_string());

        // SAFETY: dlopen is always unsafe (executes .so init code). caller
        // accepts the risk by calling this entry point.
        let lib = unsafe {
            libloading::Library::new(&lib_path)
                .map_err(|e| anyhow!("RpcmemAllocator: dlopen {} failed: {}", lib_path, e))?
        };

        let alloc_raw: RpcmemAllocFn = unsafe {
            let sym: Symbol<RpcmemAllocFn> = lib
                .get(b"rpcmem_alloc\0")
                .map_err(|e| anyhow!("get rpcmem_alloc failed: {}", e))?;
            *sym.into_raw()
        };
        let free_raw: RpcmemFreeFn = unsafe {
            let sym: Symbol<RpcmemFreeFn> = lib
                .get(b"rpcmem_free\0")
                .map_err(|e| anyhow!("get rpcmem_free failed: {}", e))?;
            *sym.into_raw()
        };
        let to_fd_raw: RpcmemToFdFn = unsafe {
            let sym: Symbol<RpcmemToFdFn> = lib
                .get(b"rpcmem_to_fd\0")
                .map_err(|e| anyhow!("get rpcmem_to_fd failed: {}", e))?;
            *sym.into_raw()
        };

        Ok(Self::OwnedDlopen {
            rpcmem_alloc: alloc_raw,
            rpcmem_free: free_raw,
            rpcmem_to_fd: to_fd_raw,
            _lib: lib,
        })
    }

    /// Host (non-Android) build: `RpcmemAllocator::new()` 는 즉시 Err
    /// (INV-RPCMEM-001). `OpenCLBackend::new_with_options(_, true)` 는 본 Err
    /// 를 받아 `opencl_rpcmem = false` 로 강등하고 stderr 에 warning 1회.
    #[cfg(not(target_os = "android"))]
    pub fn new() -> anyhow::Result<Self> {
        Err(anyhow::anyhow!(
            "RpcmemAllocator is Android-only (libcdsprpc.so unavailable on this target)"
        ))
    }

    /// Sprint 2a 공존용 — `QnnOppkgRuntime::rpcmem_fns()` 가 반환한 fn-pointer
    /// 3 개를 그대로 wrap 한다. 본 path 는 dlopen 책임이 없으며 fn-pointer 의
    /// lifetime 은 `QnnOppkgRuntime` 의 Drop 까지 보장된다 (`AndroidRuntime` 이
    /// `libcdsprpc.so` Library 핸들을 소유). Sprint 2b 에서 backend 삭제 시
    /// 본 메서드도 자연 소멸.
    pub fn from_external_fns(
        rpcmem_alloc: RpcmemAllocFn,
        rpcmem_free: RpcmemFreeFn,
        rpcmem_to_fd: RpcmemToFdFn,
    ) -> Self {
        Self::ExternalFns {
            rpcmem_alloc,
            rpcmem_free,
            rpcmem_to_fd,
        }
    }

    /// rpcmem heap 에서 `size` byte alloc + DMA-BUF fd 추출.
    ///
    /// # Errors
    /// - `size == 0`: invalid input.
    /// - `rpcmem_alloc` 가 NULL 반환: heap exhaustion. 호출자는 per-buffer
    ///   fallback (INV-RPCMEM-003).
    ///
    /// # Safety
    /// 반환된 host_ptr 은 `self.free` 로 명시적으로 free 해야 한다 — Drop 은
    /// 자동 호출되지 않는다. 호출자가 (RpcmemKvBuffer / RpcmemLayerRegion)
    /// 의 Drop 에서 free 호출 책임.
    pub unsafe fn alloc(&self, size: usize) -> anyhow::Result<(*mut u8, RawFd)> {
        use anyhow::anyhow;
        if size == 0 {
            return Err(anyhow!("RpcmemAllocator::alloc: size == 0"));
        }
        if size > i32::MAX as usize {
            return Err(anyhow!(
                "RpcmemAllocator::alloc: size {} exceeds i32::MAX (libcdsprpc.so ABI limit)",
                size
            ));
        }

        let (alloc_fn, to_fd_fn) = match self {
            #[cfg(target_os = "android")]
            Self::OwnedDlopen {
                rpcmem_alloc,
                rpcmem_to_fd,
                ..
            } => (*rpcmem_alloc, *rpcmem_to_fd),
            Self::ExternalFns {
                rpcmem_alloc,
                rpcmem_to_fd,
                ..
            } => (*rpcmem_alloc, *rpcmem_to_fd),
        };

        // SAFETY: alloc_fn is a valid fn-pointer (validated at construction).
        let host_ptr =
            unsafe { alloc_fn(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size as i32) };
        if host_ptr.is_null() {
            return Err(anyhow!(
                "RpcmemAllocator::alloc: rpcmem_alloc(size={size}) returned NULL"
            ));
        }
        // SAFETY: host_ptr is a valid pointer from rpcmem_alloc.
        let fd = unsafe { to_fd_fn(host_ptr as *const _) };
        Ok((host_ptr as *mut u8, fd as RawFd))
    }

    /// `alloc` 으로 받은 host_ptr 해제.
    ///
    /// # Safety
    /// double-free / use-after-free 책임은 호출자.
    pub unsafe fn free(&self, host_ptr: *mut u8) {
        if host_ptr.is_null() {
            return;
        }
        let free_fn = match self {
            #[cfg(target_os = "android")]
            Self::OwnedDlopen { rpcmem_free, .. } => *rpcmem_free,
            Self::ExternalFns { rpcmem_free, .. } => *rpcmem_free,
        };
        // SAFETY: caller guarantees host_ptr was returned by self.alloc and
        // has not been freed yet.
        unsafe { free_fn(host_ptr as *mut std::ffi::c_void) };
    }

    /// Raw fn-pointer triple — Sprint 2a 마이그레이션 첫 단계에서 fn-pointer
    /// interface 호환을 위해 노출. Sprint 2c 에서 deprecate 대상.
    ///
    /// 신규 코드는 `alloc` / `free` 메서드를 사용한다.
    #[doc(hidden)]
    pub fn raw_fns(&self) -> (RpcmemAllocFn, RpcmemFreeFn, RpcmemToFdFn) {
        match self {
            #[cfg(target_os = "android")]
            Self::OwnedDlopen {
                rpcmem_alloc,
                rpcmem_free,
                rpcmem_to_fd,
                ..
            } => (*rpcmem_alloc, *rpcmem_free, *rpcmem_to_fd),
            Self::ExternalFns {
                rpcmem_alloc,
                rpcmem_free,
                rpcmem_to_fd,
            } => (*rpcmem_alloc, *rpcmem_free, *rpcmem_to_fd),
        }
    }
}

impl std::fmt::Debug for RpcmemAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(target_os = "android")]
            Self::OwnedDlopen { .. } => f.debug_struct("RpcmemAllocator::OwnedDlopen").finish(),
            Self::ExternalFns { .. } => f.debug_struct("RpcmemAllocator::ExternalFns").finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// host 빌드: `new()` 는 즉시 Err 반환 (INV-RPCMEM-001).
    #[cfg(not(target_os = "android"))]
    #[test]
    fn new_returns_err_on_host() {
        let r = RpcmemAllocator::new();
        assert!(
            r.is_err(),
            "host build 에서 RpcmemAllocator::new() 는 Err 반환해야 함"
        );
    }

    /// `from_external_fns` 는 host/Android 모두 컴파일 + 호출 가능.
    #[test]
    fn from_external_fns_constructs() {
        // dummy fn-pointers — 실제 호출하지 않으므로 무의미한 body 도 OK.
        unsafe extern "C" fn dummy_alloc(_: i32, _: u32, _: i32) -> *mut std::ffi::c_void {
            std::ptr::null_mut()
        }
        unsafe extern "C" fn dummy_free(_: *mut std::ffi::c_void) {}
        unsafe extern "C" fn dummy_to_fd(_: *const std::ffi::c_void) -> i32 {
            -1
        }
        let a = RpcmemAllocator::from_external_fns(dummy_alloc, dummy_free, dummy_to_fd);
        let (af, ff, tf) = a.raw_fns();
        // pointer equality
        assert_eq!(af as usize, dummy_alloc as *const () as usize);
        assert_eq!(ff as usize, dummy_free as *const () as usize);
        assert_eq!(tf as usize, dummy_to_fd as *const () as usize);
    }

    /// `alloc(0)` 은 Err.
    #[test]
    fn alloc_zero_size_errs() {
        unsafe extern "C" fn dummy_alloc(_: i32, _: u32, _: i32) -> *mut std::ffi::c_void {
            std::ptr::null_mut()
        }
        unsafe extern "C" fn dummy_free(_: *mut std::ffi::c_void) {}
        unsafe extern "C" fn dummy_to_fd(_: *const std::ffi::c_void) -> i32 {
            -1
        }
        let a = RpcmemAllocator::from_external_fns(dummy_alloc, dummy_free, dummy_to_fd);
        let r = unsafe { a.alloc(0) };
        assert!(r.is_err());
    }
}
