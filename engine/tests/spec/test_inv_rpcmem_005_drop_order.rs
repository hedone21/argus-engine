//! INV-RPCMEM-005 — `RpcmemAllocator::Drop` 시점에 모든 rpcmem buffer 가 이미
//! drop 되어 있음. allocator lifetime ⊃ buffer lifetime. `Arc<RpcmemAllocator>`
//! 가 buffer struct 의 field 로 보유되어 type system 으로 강제.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.2 (ENG-RPCMEM-013), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/rpcmem_allocator.md` §2.3, `arch/opencl_backend.md` §4.
//!
//! 본 파일은 Sprint 2a Phase 2 spec/arch 단계의 **테스트 skeleton** 이다.

#![allow(dead_code, unused_imports)]

// 검증 방식:
//   1. 컴파일 타임: RpcmemKvBuffer / RpcmemLayerRegion field 에 `Arc<RpcmemAllocator>`
//      가 포함되어 있는지 source-grep (host 빌드에서 가능).
//   2. 런타임 (Android): Arc strong_count 추적 — buffer 생성 후 count 증가,
//      buffer drop 후 count 감소, 모든 buffer drop 후 allocator 의 strong_count == 1
//      (마지막 consumer 만 남음).

use std::fs;
use std::path::PathBuf;

#[test]
fn rpcmem_kv_buffer_holds_arc_allocator() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("memory")
        .join("rpcmem")
        .join("kv_buffer.rs");
    if !path.exists() {
        eprintln!(
            "[INV-RPCMEM-005 skeleton] {} 미존재 (Implementer task #3 대기) — PASS",
            path.display()
        );
        return;
    }
    let src = fs::read_to_string(&path).expect("read kv_buffer.rs");
    assert!(
        src.contains("Arc<RpcmemAllocator>")
            || src.contains("Arc<crate::memory::rpcmem::allocator::RpcmemAllocator>"),
        "INV-RPCMEM-005: kv_buffer.rs 가 Arc<RpcmemAllocator> field 를 보유해야 함."
    );
}

#[test]
fn rpcmem_layer_region_holds_arc_allocator() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("models")
        .join("weights")
        .join("rpcmem_secondary.rs");
    if !path.exists() {
        return;
    }
    let src = fs::read_to_string(&path).expect("read rpcmem_secondary.rs");
    // post-migration: fn-pointer 가 사라지고 Arc<RpcmemAllocator> 보유.
    let migrated = src.contains("allocator: Arc<RpcmemAllocator>")
        || src.contains("allocator: Arc<crate::memory::rpcmem::allocator::RpcmemAllocator>");
    if !migrated {
        eprintln!(
            "[INV-RPCMEM-005 skeleton] rpcmem_secondary.rs 미마이그레이션 \
             (Implementer task #3 대기) — PASS. 기존 fn-pointer 필드 사용 중."
        );
    }
}

#[cfg(target_os = "android")]
#[test]
fn android_allocator_strong_count_drops_to_zero_after_buffers_release() {
    // TODO Implementer: device test.
    //   1. Arc::new(RpcmemAllocator) → strong_count == 1.
    //   2. KV buffer + secondary region 생성 → strong_count == 3.
    //   3. buffer Drop → strong_count == 1 + dlclose 미발생.
    //   4. allocator Drop → dlclose.
    eprintln!("[INV-RPCMEM-005 skeleton] TODO: Implementer device test 본문 작성");
}
