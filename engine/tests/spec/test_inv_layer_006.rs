//! INV-LAYER-006: `session::DecodeLoop` struct fields must only hold trait
//! objects (`Box<dyn ...>`, `Vec<Box<dyn ...>>`, `Arc<AtomicBool>`) or
//! primitives — never a concrete L1 backend or L3 struct.
//!
//! Verification: source-grep on `engine/src/session/decode_loop.rs`. Extract
//! the `pub struct DecodeLoop { ... }` block and assert no banned identifier
//! appears inside.
//!
//! Banned list (concrete L1 / L3 types whose presence would couple the
//! decode loop to a specific backend / domain implementation):
//! - L1: OpenCLBackend, CudaBackend, CpuBackend, QnnBackend
//! - L3 pressure: CacheManager, KiviCache, OffloadStore
//! - L3 inference: LlamaModel, TransformerModel
//! - cross-cutting concrete: ManagerClient
//!
//! Note: `OpProfiler` (from observability) is NOT banned — v2 §5.1 StageContext
//! 계약에서 DecodeLoop 이 OpProfiler 1개를 소유하고 dispatch 호출부마다 `&mut`
//! 재대여하는 것이 명시적으로 허용된다 (INV-DECODE-STAGE-006, roadmap β-1).
//!
//! **β-4 (G7 마감, 2026-06-10)**: `cache_manager: Option<CacheManager>` 과도기 필드가 제거되고
//! [`CommandDispatcher`](L4, CacheManager 를 `Arc<Mutex>` 로 보유) 로 cutover 되었으므로,
//! `ALLOWLISTED_TRANSITIONAL` allowlist 를 제거하여 `CacheManager` 금지를 자동 복원한다.
//!
//! Scope: this check applies *only* to the `DecodeLoop` struct field block —
//! impl blocks, helper functions, and trait implementor structs (e.g.
//! `ModelForward`) are free to hold concrete types since the builder
//! abstracts them via `Box<dyn Forward>` etc.

use std::path::PathBuf;

const BANNED: &[&str] = &[
    "OpenCLBackend",
    "CudaBackend",
    "CpuBackend",
    "QnnBackend",
    "CacheManager",
    "KiviCache",
    "OffloadStore",
    "LlamaModel",
    "TransformerModel",
    "ManagerClient",
];

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine 부모 경로")
        .to_path_buf()
}

fn extract_decode_loop_fields(src: &str) -> String {
    let marker = "pub struct DecodeLoop {";
    let start = src
        .find(marker)
        .expect("INV-LAYER-006: `pub struct DecodeLoop {` not found in decode_loop.rs");
    let after_marker = start + marker.len();
    let end_rel = src[after_marker..]
        .find('}')
        .expect("INV-LAYER-006: closing brace of DecodeLoop struct not found");
    src[after_marker..after_marker + end_rel].to_string()
}

#[test]
fn test_inv_layer_006_decode_loop_fields_are_abstractions_only() {
    let root = project_root();
    let path = root
        .join("engine")
        .join("src")
        .join("session")
        .join("decode_loop.rs");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("INV-LAYER-006: cannot read {path:?}: {e}"));

    let fields = extract_decode_loop_fields(&src);

    // β-4 (G7 마감): cache_manager 과도기 필드 제거 + CommandDispatcher cutover 완료로 allowlist
    // 가 사라졌다 — DecodeLoop 필드 전부가 trait object/primitive/추상화여야 한다.
    let mut violations = Vec::new();
    for ident in BANNED {
        if fields.contains(ident) {
            violations.push(*ident);
        }
    }

    assert!(
        violations.is_empty(),
        "INV-LAYER-006: DecodeLoop struct must hold only trait objects + primitives, \
         but found concrete type(s) in its field block: {violations:?}\n\
         Fields:\n{fields}"
    );
}
