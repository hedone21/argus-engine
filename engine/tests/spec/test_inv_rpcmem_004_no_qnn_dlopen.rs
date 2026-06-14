//! INV-RPCMEM-004 — `RpcmemAllocator` 모듈은 `libQnnGpu.so` / `libqnn_oppkg.so` 를
//! dlopen 하지 않는다. libcdsprpc.so 단독 의존.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.6 (ENG-RPCMEM-C01), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/rpcmem_allocator.md` §2.
//!
//! Sprint 2b: production engine/src/ 전체에서 `crate::backend::qnn_oppkg` 사용 부재 검증 추가.
//! source-grep 기반 — host 빌드에서도 무조건 실행.

#![allow(dead_code, unused_imports)]

use std::fs;
use std::path::PathBuf;

fn allocator_source_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("memory")
        .join("rpcmem")
        .join("allocator.rs")
}

fn engine_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

#[test]
fn allocator_source_does_not_reference_libqnngpu() {
    let path = allocator_source_path();
    if !path.exists() {
        // Implementer 가 아직 allocator.rs 를 만들지 않은 단계 — placeholder PASS.
        eprintln!("[INV-RPCMEM-004 skeleton] {} 미존재 — PASS", path.display());
        return;
    }
    let src = fs::read_to_string(&path).expect("read allocator.rs");
    let forbidden = [
        "libQnnGpu.so",
        "libqnn_oppkg.so",
        "QNN_BACKEND_LIB",
        "QNN_OPPKG_LIB",
    ];
    for literal in forbidden {
        assert!(
            !src.contains(literal),
            "INV-RPCMEM-004 위반: allocator.rs 가 '{literal}' literal 포함. libcdsprpc.so 만 dlopen 해야 함."
        );
    }
}

/// Sprint 2b 강화 — production `engine/src/` 전체에서 `crate::backend::qnn_oppkg`
/// 사용 부재를 source-grep 으로 검증한다.
///
/// `backend/qnn_oppkg/` 디렉토리 자체는 삭제되었으므로, 남은 참조는 모두
/// production code 의 잔류 import / downcast 이다.
#[test]
fn production_engine_src_does_not_use_qnn_oppkg_backend_module() {
    let src_dir = engine_src_dir();
    // backend/qnn_oppkg 디렉토리 자체는 제거됨 — 잔류 use 경로만 검사.
    let forbidden_patterns = [
        "crate::backend::qnn_oppkg",
        "backend::qnn_oppkg::",
        "QnnOppkgBackend",
        "QnnOppkgMemory",
        "QnnOppkgHybridMemory",
    ];

    let mut violations: Vec<String> = Vec::new();
    collect_violations(&src_dir, &forbidden_patterns, &mut violations);

    assert!(
        violations.is_empty(),
        "INV-RPCMEM-004 (Sprint 2b) 위반: production engine/src/ 에 qnn_oppkg backend 참조 잔류:\n{}",
        violations.join("\n")
    );
}

fn collect_violations(dir: &PathBuf, patterns: &[&str], violations: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_violations(&path, patterns, violations);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            for (lineno, line) in src.lines().enumerate() {
                for pat in patterns {
                    if line.contains(pat) {
                        violations.push(format!(
                            "  {}:{}: {}",
                            path.display(),
                            lineno + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }
    }
}
