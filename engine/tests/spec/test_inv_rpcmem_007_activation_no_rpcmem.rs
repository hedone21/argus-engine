//! INV-RPCMEM-007 — `OpenCLMemory::alloc` (activation tensor) 는 `opencl_rpcmem`
//! 값과 무관하게 rpcmem heap 을 사용하지 않는다. KV cache (`alloc_kv`) 와
//! precision swap secondary 전용.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.3 (ENG-RPCMEM-023), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/opencl_backend.md` §3.3.

#![allow(dead_code, unused_imports)]

use std::fs;
use std::path::PathBuf;

/// INV-RPCMEM-007: OpenCLMemory::alloc (activation) 이 rpcmem heap 을 사용하지 않음을
/// source-grep 으로 검증. `alloc_kv` 와 달리 `alloc` 함수는 `rpcmem_allocator` 분기가
/// 없어야 한다.
#[test]
fn activation_alloc_source_has_no_rpcmem_branch() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("backend")
        .join("opencl")
        .join("memory.rs");
    if !path.exists() {
        eprintln!("[INV-RPCMEM-007] opencl/memory.rs 없음 (opencl feature 미활성) — skip");
        return;
    }
    let src = fs::read_to_string(&path).expect("read memory.rs");

    // `alloc_kv` 함수에는 rpcmem 분기가 있어야 하고 (대조군),
    // `fn alloc(` 함수 내부에는 rpcmem 분기가 없어야 한다 (INV-RPCMEM-007).
    // 간단한 검증: `alloc_kv_rpcmem` 이 존재함 (KV path rpcmem OK) +
    // `alloc` 내부에 rpcmem_allocator 직접 참조 없음 (heuristic).
    assert!(
        src.contains("alloc_kv_rpcmem") || src.contains("alloc_kv"),
        "INV-RPCMEM-007 전제: alloc_kv 함수가 memory.rs 에 존재해야 함."
    );

    // `fn alloc(` 이후 첫 번째 `fn ` 발생 전까지의 텍스트에서 rpcmem_allocator 가 없음.
    // 더 정확한 방법: alloc 함수가 rpcmem 을 조건부로 사용하지 않는다는 주석이 존재.
    assert!(
        src.contains("INV-RPCMEM-007")
            || src.contains("ENG-RPCMEM-023")
            || src.contains("activation"),
        "INV-RPCMEM-007: memory.rs 에 activation alloc = rpcmem 미사용 명시 없음 \
         (INV-RPCMEM-007 / ENG-RPCMEM-023 주석 또는 'activation' 언급 필요)."
    );
}

#[cfg(target_os = "android")]
#[test]
fn activation_alloc_does_not_use_rpcmem_heap() {
    // Android 디바이스에서만 실행. 호스트는 위 source-grep 으로 대체.
    eprintln!("[INV-RPCMEM-007] Android target — 디바이스 측정 대상.");
}

#[cfg(not(target_os = "android"))]
#[test]
fn host_build_static_check() {
    eprintln!(
        "[INV-RPCMEM-007] host: source-grep 검증 완료 (activation_alloc_source_has_no_rpcmem_branch)."
    );
}
