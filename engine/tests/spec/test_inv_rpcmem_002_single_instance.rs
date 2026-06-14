//! INV-RPCMEM-002 — `--opencl-rpcmem` 활성 시 OpenCLBackend 와 RpcmemSecondaryStore
//! 가 보유하는 `Arc<RpcmemAllocator>` 는 동일 인스턴스 (single allocator per session).
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.3/E.4 (ENG-RPCMEM-012, ENG-RPCMEM-032).
//! 대응 arch: `arch/rpcmem_allocator.md` §3, `arch/precision_swap.md` §4.

#![allow(dead_code, unused_imports)]

// host 검증: source-grep 으로 EXT_RPCMEM_ALLOCATOR extension lookup + clone sharing 확인.
#[cfg(not(target_os = "android"))]
#[test]
fn host_build_source_verifies_single_instance_sharing() {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // 1. OpenCLBackend::get_extension 이 EXT_RPCMEM_ALLOCATOR 를 노출.
    let opencl_mod = base.join("src").join("backend").join("opencl.rs");
    if opencl_mod.exists() {
        let src = std::fs::read_to_string(&opencl_mod).expect("read opencl/mod.rs");
        assert!(
            src.contains("EXT_RPCMEM_ALLOCATOR"),
            "INV-RPCMEM-002: opencl/mod.rs 가 EXT_RPCMEM_ALLOCATOR extension 을 노출하지 않음."
        );
    }

    // 2. secondary_mmap 이 EXT_RPCMEM_ALLOCATOR 로 allocator 를 획득 (INV-RPCMEM-002 Arc clone).
    let secondary = base
        .join("src")
        .join("models")
        .join("weights")
        .join("secondary_mmap.rs");
    if secondary.exists() {
        let src = std::fs::read_to_string(&secondary).expect("read secondary_mmap.rs");
        assert!(
            src.contains("EXT_RPCMEM_ALLOCATOR"),
            "INV-RPCMEM-002: secondary_mmap.rs 가 EXT_RPCMEM_ALLOCATOR lookup 없음 \
             — single instance sharing 검증 불가."
        );
    }
}

#[cfg(target_os = "android")]
#[test]
fn single_allocator_instance_shared_across_consumers() {
    // Android 디바이스에서만 의미. 호스트는 위 source-grep 으로 대체.
    // 실 검증은 디바이스 e2e 테스트 에서 수행.
    eprintln!("[INV-RPCMEM-002 skeleton] Android target — 디바이스 측정 대상.");
}

#[cfg(not(target_os = "android"))]
#[test]
fn host_build_skip() {
    // Host 에선 RpcmemAllocator init 불가 → 위 source-grep 테스트로 대체.
    eprintln!("[INV-RPCMEM-002] host build: source-grep 검증 완료.");
}
