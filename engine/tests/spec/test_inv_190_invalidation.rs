// spec/41-invariants.md §3.29 INV-190: 복원 무효화 4-tuple 검증
//
// 검증:
// (a) 정상 save→restore → Some(token_count) (happy path)
// (b) model_hash 변조 → None (panic 없음)
// (c) format_id 변조 → None
// (d) tokenizer_hash 변조 → None
// (e) token_ids 1개 divergence → None
// (f) magic 변조 → None
// (g) version 변조 → None
// 모두 panic 없이 fresh prefill 강하 (Ok(None))

use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::format::{KVCacheFormat, SnapshotRestore};
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::kv::standard_format::StandardFormat;
use argus_engine::kv_cache_ops::KVLayout;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::session::prefix_cache::{RestoredPrefix, save_prefix, try_restore_prefix};
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use std::sync::Arc;

fn make_headmajor_f32(capacity: usize, kv_heads: usize, head_dim: usize) -> KVCache {
    let total = kv_heads * capacity * head_dim;
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let backend = Arc::new(CpuBackend::new());
    let buf_k = mem.alloc_kv(total * 4, DType::F32).unwrap();
    let buf_v = mem.alloc_kv(total * 4, DType::F32).unwrap();
    let k = Tensor::new(
        Shape::new(vec![1, kv_heads, capacity, head_dim]),
        buf_k,
        backend.clone(),
    );
    let v = Tensor::new(
        Shape::new(vec![1, kv_heads, capacity, head_dim]),
        buf_v,
        backend.clone(),
    );
    KVCache::new_dynamic(k, v, capacity, capacity * 4, kv_heads, head_dim, mem)
        .with_layout(KVLayout::HeadMajor)
}

fn write_tokens(
    fmt: &StandardFormat,
    n: usize,
    kv_heads: usize,
    head_dim: usize,
    backend: &CpuBackend,
) {
    for pos in 0..n {
        let total = kv_heads * head_dim;
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        let be = Arc::new(CpuBackend::new());
        let buf_k = mem.alloc_kv(total * 4, DType::F32).unwrap();
        let buf_v = mem.alloc_kv(total * 4, DType::F32).unwrap();
        let mut tk = Tensor::new(
            Shape::new(vec![1, 1, kv_heads, head_dim]),
            buf_k,
            be.clone(),
        );
        let mut tv = Tensor::new(
            Shape::new(vec![1, 1, kv_heads, head_dim]),
            buf_v,
            be.clone(),
        );
        let k_data: Vec<f32> = (0..total).map(|i| (pos * 7 + i) as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..total).map(|i| (pos * 7 + i) as f32 * 0.2).collect();
        tk.as_mut_slice::<f32>().copy_from_slice(&k_data);
        tv.as_mut_slice::<f32>().copy_from_slice(&v_data);
        fmt.write_kv(&tk, &tv, backend).unwrap();
    }
}

const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const N_TOKENS: usize = 3;
const MODEL_HASH: [u8; 32] = [1u8; 32];
const TOK_HASH: [u8; 32] = [2u8; 32];

fn setup_saved_cache() -> (
    tempfile::TempDir,
    std::path::PathBuf,
    Vec<u32>,
    u32,
    CpuBackend,
) {
    let cache = make_headmajor_f32(16, KV_HEADS, HEAD_DIM);
    let fmt = StandardFormat::new(0, cache);
    let backend = CpuBackend::new();
    write_tokens(&fmt, N_TOKENS, KV_HEADS, HEAD_DIM, &backend);

    let token_ids: Vec<u32> = (0..N_TOKENS as u32).collect();
    let format_id = fmt.snapshot_format_id();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.cache");
    save_prefix(
        &path,
        &MODEL_HASH,
        &TOK_HASH,
        &token_ids,
        &[],
        format_id,
        &[&fmt as &dyn SnapshotRestore],
        KV_HEADS as u32,
        HEAD_DIM as u32,
        &backend,
    )
    .unwrap();
    (dir, path, token_ids, format_id, backend)
}

fn make_restore_fmt() -> StandardFormat {
    StandardFormat::new(0, make_headmajor_f32(16, KV_HEADS, HEAD_DIM))
}

/// INV-190(a): 정상 save → restore = Some(RestoredPrefix { token_count: N_TOKENS, .. })
#[test]
fn happy_path_returns_some() {
    let (dir, path, token_ids, format_id, backend) = setup_saved_cache();
    let _ = dir;
    let fmt_b = make_restore_fmt();
    let result = try_restore_prefix(
        &path,
        &MODEL_HASH,
        &TOK_HASH,
        format_id,
        &token_ids,
        &[&fmt_b as &dyn SnapshotRestore],
        KV_HEADS as u32,
        HEAD_DIM as u32,
        &backend,
    )
    .unwrap();
    let restored = result.expect("happy path must return Some");
    assert_eq!(restored.token_count, N_TOKENS);
}

/// INV-190(b): model_hash 변조 → None (panic 없음)
#[test]
fn model_hash_mismatch_returns_none() {
    let (dir, path, token_ids, format_id, backend) = setup_saved_cache();
    let _ = dir;
    let fmt_b = make_restore_fmt();
    let bad = [99u8; 32];
    let result = try_restore_prefix(
        &path,
        &bad,
        &TOK_HASH,
        format_id,
        &token_ids,
        &[&fmt_b as &dyn SnapshotRestore],
        KV_HEADS as u32,
        HEAD_DIM as u32,
        &backend,
    )
    .unwrap();
    assert_eq!(result, None, "model_hash mismatch → None (INV-190)");
}

/// INV-190(c): format_id 변조 → None
#[test]
fn format_id_mismatch_returns_none() {
    let (dir, path, token_ids, format_id, backend) = setup_saved_cache();
    let _ = dir;
    let fmt_b = make_restore_fmt();
    let result = try_restore_prefix(
        &path,
        &MODEL_HASH,
        &TOK_HASH,
        format_id + 100,
        &token_ids,
        &[&fmt_b as &dyn SnapshotRestore],
        KV_HEADS as u32,
        HEAD_DIM as u32,
        &backend,
    )
    .unwrap();
    assert_eq!(result, None, "format_id mismatch → None (INV-190)");
}

/// INV-190(d): tokenizer_hash 변조 → None
#[test]
fn tokenizer_hash_mismatch_returns_none() {
    let (dir, path, token_ids, format_id, backend) = setup_saved_cache();
    let _ = dir;
    let fmt_b = make_restore_fmt();
    let bad = [77u8; 32];
    let result = try_restore_prefix(
        &path,
        &MODEL_HASH,
        &bad,
        format_id,
        &token_ids,
        &[&fmt_b as &dyn SnapshotRestore],
        KV_HEADS as u32,
        HEAD_DIM as u32,
        &backend,
    )
    .unwrap();
    assert_eq!(result, None, "tokenizer_hash mismatch → None (INV-190)");
}

/// INV-190(e): token_ids 1개 divergence → None
#[test]
fn token_ids_divergence_returns_none() {
    let (dir, path, token_ids, format_id, backend) = setup_saved_cache();
    let _ = dir;
    let fmt_b = make_restore_fmt();
    let mut diverged = token_ids.clone();
    *diverged.last_mut().unwrap() = 9999u32;
    let result = try_restore_prefix(
        &path,
        &MODEL_HASH,
        &TOK_HASH,
        format_id,
        &diverged,
        &[&fmt_b as &dyn SnapshotRestore],
        KV_HEADS as u32,
        HEAD_DIM as u32,
        &backend,
    )
    .unwrap();
    assert_eq!(result, None, "token_ids divergence → None (INV-190)");
}

/// INV-190(f): magic 변조 → None (panic 없음)
#[test]
fn magic_corruption_returns_none() {
    let (dir, path, token_ids, format_id, backend) = setup_saved_cache();
    let _ = dir;
    let mut raw = std::fs::read(&path).unwrap();
    for b in &mut raw[..8] {
        *b = 0xFF;
    }
    std::fs::write(&path, &raw).unwrap();
    let fmt_b = make_restore_fmt();
    let result = try_restore_prefix(
        &path,
        &MODEL_HASH,
        &TOK_HASH,
        format_id,
        &token_ids,
        &[&fmt_b as &dyn SnapshotRestore],
        KV_HEADS as u32,
        HEAD_DIM as u32,
        &backend,
    )
    .unwrap();
    assert_eq!(result, None, "magic corruption → None (INV-190)");
}

/// INV-190(g): version 변조 → None
#[test]
fn version_mismatch_returns_none() {
    let (dir, path, token_ids, format_id, backend) = setup_saved_cache();
    let _ = dir;
    let mut raw = std::fs::read(&path).unwrap();
    raw[8..12].copy_from_slice(&9999u32.to_le_bytes());
    std::fs::write(&path, &raw).unwrap();
    let fmt_b = make_restore_fmt();
    let result = try_restore_prefix(
        &path,
        &MODEL_HASH,
        &TOK_HASH,
        format_id,
        &token_ids,
        &[&fmt_b as &dyn SnapshotRestore],
        KV_HEADS as u32,
        HEAD_DIM as u32,
        &backend,
    )
    .unwrap();
    assert_eq!(result, None, "version mismatch → None (INV-190)");
}
