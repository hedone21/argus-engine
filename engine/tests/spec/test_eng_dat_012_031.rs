//! ENG-DAT-012 ~ ENG-DAT-031: KVCache, Buffer, DType, Tensor
//!
//! KVCache 생성/메모리/overflow/get_view/prune_prefix/dynamic_growth,
//! Buffer trait default impls, DType size, Tensor creation/reshape/clone/slice.

use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::{Buffer, DType};
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use std::sync::Arc;

// ── 헬퍼 ──

fn make_cache(max_seq: usize, heads: usize, dim: usize) -> KVCache {
    let backend = Arc::new(CpuBackend::new());
    let buf_size = max_seq * heads * dim * 4;
    let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
    let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
    let k = Tensor::new(
        Shape::new(vec![1, max_seq, heads, dim]),
        k_buf,
        backend.clone(),
    );
    let v = Tensor::new(Shape::new(vec![1, max_seq, heads, dim]), v_buf, backend);
    KVCache::new(k, v, max_seq)
}

// ══════════════════════════════════════════════════════════════
// ENG-DAT-012: KVCache — 생성 + capacity + memory_usage
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_dat_012_cache_creation() {
    let cache = make_cache(256, 8, 64);
    assert_eq!(cache.current_pos(), 0);
    assert_eq!(cache.capacity(), 256);
    assert_eq!(cache.kv_heads(), 8);
    assert_eq!(cache.head_dim(), 64);
}

#[test]
fn test_eng_dat_012_memory_usage_bytes() {
    let cache = make_cache(256, 8, 64);
    // current_pos=0이면 메모리 사용량 0
    assert_eq!(cache.memory_usage_bytes(), 0);
}

#[test]
fn test_eng_dat_012_get_view() {
    let cache = make_cache(100, 1, 4);
    // get_view(seq_len)은 전체 버퍼 텐서를 반환
    let (k_view, v_view) = cache.get_view(0);
    // shape은 [1, max_seq, heads, dim]이지만 유효 데이터는 current_pos까지
    assert!(k_view.shape().numel() > 0);
    assert!(v_view.shape().numel() > 0);
}

#[test]
fn test_eng_dat_012_prune_prefix_basic() {
    let mut cache = make_cache(100, 1, 4);
    cache.current_pos = 20;

    // 앞에서 5개 토큰 제거
    cache.prune_prefix(5).unwrap();
    assert_eq!(cache.current_pos, 15);
}

#[test]
fn test_eng_dat_012_prune_prefix_zero() {
    let mut cache = make_cache(100, 1, 4);
    cache.current_pos = 20;

    cache.prune_prefix(0).unwrap();
    assert_eq!(cache.current_pos, 20);
}

#[test]
fn test_eng_dat_012_prune_prefix_all() {
    let mut cache = make_cache(100, 1, 4);
    cache.current_pos = 10;

    cache.prune_prefix(10).unwrap();
    assert_eq!(cache.current_pos, 0);
}

// ══════════════════════════════════════════════════════════════
// ENG-DAT-020: Buffer trait — 기본 구현 + metadata accessors
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_dat_020_buffer_metadata_accessors() {
    let buf = SharedBuffer::new(1024, DType::F32);
    assert_eq!(buf.dtype(), DType::F32);
    assert_eq!(buf.size(), 1024);
    assert!(!buf.as_ptr().is_null());
    assert!(!buf.as_mut_ptr().is_null());
}

#[test]
fn test_eng_dat_020_buffer_default_impls() {
    let buf = SharedBuffer::new(256, DType::F32);
    // map/unmap default impls는 no-op
    assert!(buf.map_for_cpu().is_ok());
    assert!(buf.unmap_for_gpu().is_ok());
    assert!(buf.is_mapped());
    assert!(buf.is_host_managed());
}

// ══════════════════════════════════════════════════════════════
// ENG-DAT-021: DType — size(), variant 커버리지
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_dat_021_dtype_size() {
    assert_eq!(DType::F32.size(), 4);
    assert_eq!(DType::F16.size(), 2);
    assert_eq!(DType::BF16.size(), 2);
    assert_eq!(DType::Q4_0.size(), 1);
    assert_eq!(DType::Q4_1.size(), 1);
    assert_eq!(DType::U8.size(), 1);
}

#[test]
fn test_eng_dat_021_dtype_all_variant_sizes() {
    let variants = [
        DType::Q4_0,
        DType::Q4_1,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::U8,
    ];
    for dtype in &variants {
        assert!(dtype.size() > 0, "{:?} size should be > 0", dtype);
    }
}

#[test]
fn test_eng_dat_021_dtype_equality_and_copy() {
    let a = DType::F32;
    let b = a; // Copy
    assert_eq!(a, b);
    assert_ne!(a, DType::F16);
}

// ══════════════════════════════════════════════════════════════
// ENG-DAT-031: Tensor — creation, reshape, clone, as_slice, to_device
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_dat_031_tensor_creation_and_metadata() {
    let backend = Arc::new(CpuBackend::new());
    let buf = Arc::new(SharedBuffer::new(256, DType::F32));
    let shape = Shape::new(vec![2, 8, 4]); // 64 elements * 4 bytes = 256

    let tensor = Tensor::new(shape.clone(), buf, backend);
    assert_eq!(tensor.shape().dims(), &[2, 8, 4]);
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.size(), 256);
    assert_eq!(tensor.numel(), 64);
}

#[test]
fn test_eng_dat_031_tensor_reshape() {
    let backend = Arc::new(CpuBackend::new());
    let buf = Arc::new(SharedBuffer::new(256, DType::F32));
    let shape = Shape::new(vec![4, 16]); // 64 elements

    let mut tensor = Tensor::new(shape, buf, backend);
    // reshape는 in-place 변경 (반환값 없음)
    tensor.reshape(Shape::new(vec![8, 8]));
    assert_eq!(tensor.shape().dims(), &[8, 8]);
    assert_eq!(tensor.numel(), 64);
}

#[test]
fn test_eng_dat_031_tensor_clone_shares_buffer() {
    let backend = Arc::new(CpuBackend::new());
    let buf = Arc::new(SharedBuffer::new(64, DType::F32));
    let tensor = Tensor::new(Shape::new(vec![4, 4]), buf, backend);

    let cloned = tensor.clone();
    // Clone은 Arc<dyn Buffer>를 공유하므로 같은 포인터를 가짐
    assert_eq!(tensor.as_ptr(), cloned.as_ptr());
}

#[test]
fn test_eng_dat_031_tensor_as_slice_bounds() {
    let backend = Arc::new(CpuBackend::new());
    let buf = Arc::new(SharedBuffer::new(16, DType::F32)); // 4 floats
    let tensor = Tensor::new(Shape::new(vec![4]), buf, backend);

    let slice: &[f32] = tensor.as_slice();
    assert_eq!(slice.len(), 4);
}
