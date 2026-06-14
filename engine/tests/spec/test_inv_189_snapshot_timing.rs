// spec/41-invariants.md §3.29 INV-189: snapshot 저장 시점 — prefill 직후·eviction 전
//
// 검증:
// (a) prefill 후 current_pos==prompt.len() 시 snapshot 정상 수행
// (b) eviction(current_pos 감소) 후 save 시도 시 snapshot current_pos != token_count → 실패
// (c) save_prefix에서 정상 저장 후 파일 존재 확인

use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::format::{KVCacheFormat, SnapshotRestore};
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::kv::standard_format::StandardFormat;
use argus_engine::kv_cache_ops::KVLayout;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::session::prefix_cache::save_prefix;
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

fn write_token_via_write_kv(
    fmt: &StandardFormat,
    pos: usize,
    kv_heads: usize,
    head_dim: usize,
    backend: &CpuBackend,
) {
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

/// INV-189(a): prefill 후 current_pos == n_tokens 에서 snapshot 가능
#[test]
fn snapshot_at_prefill_end_succeeds() {
    let kv_heads = 2usize;
    let head_dim = 4usize;
    let n_tokens = 3usize;

    let cache = make_headmajor_f32(16, kv_heads, head_dim);
    let fmt = StandardFormat::new(0, cache);
    let backend = CpuBackend::new();

    for i in 0..n_tokens {
        write_token_via_write_kv(&fmt, i, kv_heads, head_dim, &backend);
    }

    assert_eq!(
        fmt.current_pos(),
        n_tokens,
        "current_pos must equal n_tokens after prefill"
    );
    let bytes = fmt.snapshot_prefix(n_tokens, &backend).unwrap();
    assert!(!bytes.is_empty(), "snapshot bytes must not be empty");
}

/// INV-189(b): current_pos < token_count (eviction 후 상태) 에서 snapshot 실패
/// StandardFormat.snapshot_prefix에 current_pos != token_count 체크가 있어야 한다
#[test]
fn snapshot_with_wrong_token_count_fails() {
    let kv_heads = 2usize;
    let head_dim = 4usize;
    let n_tokens = 3usize;

    let cache = make_headmajor_f32(16, kv_heads, head_dim);
    let fmt = StandardFormat::new(0, cache);
    let backend = CpuBackend::new();

    for i in 0..n_tokens {
        write_token_via_write_kv(&fmt, i, kv_heads, head_dim, &backend);
    }
    // current_pos = 3, token_count 인자는 5 (불일치)
    let result = fmt.snapshot_prefix(5, &backend);
    assert!(
        result.is_err(),
        "snapshot_prefix must fail when token_count != current_pos (INV-189)"
    );
}

/// INV-189(c): prefill 직후 save_prefix 정상 수행 — 파일 생성 확인
#[test]
fn save_prefix_creates_file() {
    let kv_heads = 2usize;
    let head_dim = 4usize;
    let n_tokens = 3usize;

    let cache = make_headmajor_f32(16, kv_heads, head_dim);
    let fmt = StandardFormat::new(0, cache);
    let backend = CpuBackend::new();

    for i in 0..n_tokens {
        write_token_via_write_kv(&fmt, i, kv_heads, head_dim, &backend);
    }

    let token_ids: Vec<u32> = (0..n_tokens as u32).collect();
    let model_hash = [1u8; 32];
    let tok_hash = [2u8; 32];
    let format_id = fmt.snapshot_format_id();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.cache");
    save_prefix(
        &path,
        &model_hash,
        &tok_hash,
        &token_ids,
        &[],
        format_id,
        &[&fmt as &dyn SnapshotRestore],
        kv_heads as u32,
        head_dim as u32,
        &backend,
    )
    .unwrap();

    assert!(
        path.exists(),
        "snapshot file must exist after save_prefix (INV-189)"
    );
    // 파일 크기가 0보다 커야 한다
    let meta = std::fs::metadata(&path).unwrap();
    assert!(meta.len() > 0, "snapshot file must not be empty");
}
