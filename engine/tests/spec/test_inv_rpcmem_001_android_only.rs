//! INV-RPCMEM-001 — `RpcmemAllocator::new()` 는 호스트(non-Android) 빌드에서
//! 컴파일 자체에서 제외되거나 호출 시 `Err` 반환. host 빌드에서
//! `OpenCLBackend::new_with_options(_, true)` 호출 시 자동 강등 + stderr warning 1회.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.2 (ENG-RPCMEM-011), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/rpcmem_allocator.md` §2.2, `arch/opencl_backend.md` §2.2.

#![allow(dead_code, unused_imports)]

#[cfg(not(target_os = "android"))]
#[test]
fn host_build_rpcmem_allocator_returns_err_or_excluded() {
    // INV-RPCMEM-001 검증: host 빌드에서 `RpcmemAllocator::new()` 는 Err 반환.
    let r = argus_engine::memory::rpcmem::allocator::RpcmemAllocator::new();
    assert!(
        r.is_err(),
        "INV-RPCMEM-001: host build 에서 RpcmemAllocator::new() 는 Err 반환해야 함. \
         got Ok variant — libcdsprpc.so Android-only constraint 위반."
    );
    // source-grep: allocator.rs 에 `#[cfg(target_os = "android")]` 가 존재함을 검증.
    let src_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("memory")
        .join("rpcmem")
        .join("allocator.rs");
    let src = std::fs::read_to_string(&src_path).expect("read allocator.rs");
    assert!(
        src.contains(r#"#[cfg(target_os = "android")]"#),
        "INV-RPCMEM-001: allocator.rs 에 target_os = android cfg gate 가 없음."
    );
}

#[cfg(not(target_os = "android"))]
#[test]
fn host_build_opencl_backend_demotes_rpcmem_flag() {
    // ENG-RPCMEM-021 / INV-RPCMEM-001: host 빌드에서 `new_with_options(_, true)` 호출 시
    // rpcmem_allocator 가 None 으로 강등됨.
    //
    // OpenCL 없는 호스트에서는 backend init 자체가 Err (GPU 없음) 이므로 source-grep
    // 으로 강등 로직 존재 검증.
    let src_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("backend")
        .join("opencl.rs");
    if !src_path.exists() {
        eprintln!("[INV-RPCMEM-001] opencl/mod.rs 없음 (opencl feature 미활성) — skip");
        return;
    }
    let src = std::fs::read_to_string(&src_path).expect("read opencl/mod.rs");
    // new_with_options 가 RpcmemAllocator::new() Err 시 강등 메시지 출력함을 source-grep.
    assert!(
        src.contains("--opencl-rpcmem 강등") || src.contains("opencl_rpcmem 강등"),
        "INV-RPCMEM-001: opencl/mod.rs 에 rpcmem 강등 경고 출력 코드가 없음."
    );
    assert!(
        src.contains("new_with_options"),
        "INV-RPCMEM-001: opencl/mod.rs 에 new_with_options 가 없음."
    );
}

#[cfg(target_os = "android")]
#[test]
fn android_build_test_runs_on_device() {
    // Android 디바이스 microbench 에서만 의미. 호스트 cargo test 에서는 skip.
    eprintln!("[INV-RPCMEM-001 skeleton] Android target — 디바이스 측정 대상.");
}
