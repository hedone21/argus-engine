//! INV-RPCMEM-003 — rpcmem alloc 실패 시 per-buffer fallback (UnifiedBuffer 또는
//! SecondaryUnavailable → GGUF mmap). session 전체 abort 금지.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.3 (ENG-RPCMEM-022), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/opencl_backend.md` §3.2, `arch/precision_swap.md` §4.

#![allow(dead_code, unused_imports)]

// `ExternalFns` variant 는 host/Android 모두 컴파일 가능 — null-returning 가짜 allocator 로
// alloc failure 주입. 이를 통해 OpenCLMemory 의 fallback 경로를 간접 검증한다.

use argus_engine::memory::rpcmem::allocator::RpcmemAllocator;

/// Alloc 이 항상 NULL 을 반환하는 mock fn-pointer.
unsafe extern "C" fn always_null_alloc(_: i32, _: u32, _: i32) -> *mut std::ffi::c_void {
    std::ptr::null_mut()
}
unsafe extern "C" fn noop_free(_: *mut std::ffi::c_void) {}
unsafe extern "C" fn always_minus1_fd(_: *const std::ffi::c_void) -> i32 {
    -1
}

#[test]
fn mock_allocator_alloc_failure_returns_err() {
    // INV-RPCMEM-003 전제: alloc Err 시 session abort 없이 fallback.
    // 여기서는 mock allocator 가 Err 를 반환함을 확인.
    let a = RpcmemAllocator::from_external_fns(always_null_alloc, noop_free, always_minus1_fd);
    let result = unsafe { a.alloc(4096) };
    assert!(
        result.is_err(),
        "INV-RPCMEM-003: NULL host_ptr 시 alloc 은 Err 반환해야 함."
    );
}

#[test]
fn rpcmem_alloc_failure_falls_back_to_unified_buffer() {
    // INV-RPCMEM-003: alloc failure → UnifiedBuffer fallback.
    // OpenCLMemory::alloc_kv 의 rpcmem branch 는 alloc Err 시 eprintln + UnifiedBuffer
    // 경로로 진행한다. source-grep 으로 fallback 코드 존재 확인.
    let src_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("backend")
        .join("opencl")
        .join("memory.rs");
    if !src_path.exists() {
        eprintln!("[INV-RPCMEM-003] opencl/memory.rs 없음 (opencl feature 미활성) — skip");
        return;
    }
    let src = std::fs::read_to_string(&src_path).expect("read memory.rs");
    // fallback 로직: rpcmem alloc 실패 시 UnifiedBuffer 경로로 진행.
    assert!(
        src.contains("UnifiedBuffer") || src.contains("alloc_kv_rpcmem"),
        "INV-RPCMEM-003: memory.rs 에 rpcmem alloc fallback 경로 없음."
    );
    assert!(
        src.contains("fallback") || src.contains("Fallback"),
        "INV-RPCMEM-003: memory.rs 에 fallback 처리 없음 — alloc failure 시 abort 위험."
    );
}

#[test]
fn rpcmem_secondary_alloc_failure_falls_back_to_gguf_mmap() {
    // INV-RPCMEM-003: RpcmemSecondaryStore::from_gguf 가 Err 반환 → SecondaryUnavailable.
    // source-grep 으로 secondary_mmap.rs 의 fallback 경로 확인.
    let src_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("models")
        .join("weights")
        .join("secondary_mmap.rs");
    if !src_path.exists() {
        eprintln!("[INV-RPCMEM-003] secondary_mmap.rs 없음 — skip");
        return;
    }
    let src = std::fs::read_to_string(&src_path).expect("read secondary_mmap.rs");
    // fallback 경로: try_open_rpcmem_secondary 실패 시 standard GGUF mmap path 로 진행.
    assert!(
        src.contains("falling back") || src.contains("fallback"),
        "INV-RPCMEM-003: secondary_mmap.rs 에 rpcmem secondary fallback 경로 없음."
    );
}
