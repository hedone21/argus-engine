//! INV-RPCMEM-008 — `RpcmemAliasBuffer` / `RpcmemKvBuffer` 의 backing host_ptr 은
//! allocator 가 alloc 한 rpcmem 영역 안에만 존재한다. raw `clientBuf` import
//! (Phase 10 HeteroLLM fast feasibility 의 0.04 GB/s slow path) 금지.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.6 (ENG-RPCMEM-C03), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/opencl_backend.md` §3.
//!
//! 본 파일은 Sprint 2a Phase 2 spec/arch 단계의 **테스트 skeleton** 이다.
//! source-grep 기반 — host 빌드에서도 무조건 실행.

#![allow(dead_code, unused_imports)]

use std::fs;
use std::path::{Path, PathBuf};

fn check_source_no_clientbuf(path: &Path) {
    if !path.exists() {
        eprintln!(
            "[INV-RPCMEM-008 skeleton] {} 미존재 (Implementer task #3 대기) — PASS",
            path.display()
        );
        return;
    }
    let src = fs::read_to_string(path).expect("read source");
    let forbidden = ["clientBuf", "QNN_MEM_TYPE_CLIENT_BUF", "CLIENT_BUF"];
    for literal in forbidden {
        assert!(
            !src.contains(literal),
            "INV-RPCMEM-008 위반: {} 가 '{literal}' literal 포함. raw clientBuf import 금지 (Phase 10 negative).",
            path.display()
        );
    }
}

#[test]
fn opencl_memory_no_raw_client_buf() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("backend")
        .join("opencl")
        .join("memory.rs");
    check_source_no_clientbuf(&path);
}

#[test]
fn opencl_backend_mod_no_raw_client_buf() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("backend")
        .join("opencl.rs");
    check_source_no_clientbuf(&path);
}

#[test]
fn rpcmem_kv_buffer_no_raw_client_buf() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("memory")
        .join("rpcmem")
        .join("kv_buffer.rs");
    check_source_no_clientbuf(&path);
}

#[test]
fn rpcmem_alias_buffer_no_raw_client_buf() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("memory")
        .join("rpcmem")
        .join("opencl_alias.rs");
    check_source_no_clientbuf(&path);
}
