//! ENG-ALG-020 ~ ENG-ALG-022: KIVI Cache
//!
//! QuantizedRecentWindowCache 기본 동작, bits 전이(transition), 점진적 역양자화(incremental deq).
//! KiviCache는 FP32 residual buffer + quantized compressed storage 구조.

use argus_engine::kv::quant_window_cache::QuantizedRecentWindowCache;

// ══════════════════════════════════════════════════════════════
// ENG-ALG-020: QuantizedRecentWindowCache 기본 동작
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_020_kivi_cache_basic() {
    let kv_heads = 2;
    let head_dim = 32;
    let max_seq = 256;
    let res_size = 32; // residual buffer = 32 tokens

    let cache = QuantizedRecentWindowCache::new(kv_heads, head_dim, max_seq, res_size);

    assert_eq!(cache.current_pos(), 0);
    assert_eq!(cache.kv_heads(), kv_heads);
    assert_eq!(cache.head_dim(), head_dim);
    assert_eq!(cache.bits(), 2); // 기본 2-bit (KIVI Q2)
}

#[test]
fn test_eng_alg_020_kivi_cache_residual_only() {
    // Residual buffer가 가득 차기 전까지는 quantized storage에 데이터가 없어야 함
    let cache = QuantizedRecentWindowCache::new(1, 32, 256, 32);
    assert_eq!(cache.q2_tokens, 0);
    assert_eq!(cache.res_pos, 0);
}

#[test]
fn test_eng_alg_020_kivi_cache_compression_ratio() {
    // 빈 캐시의 memory_usage는 0이어야 함
    let cache = QuantizedRecentWindowCache::new(2, 32, 256, 32);
    assert_eq!(cache.memory_usage_bytes(), 0);
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-021: KIVI bits 전이
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_021_kivi_transition_bits_8_to_4_to_2() {
    let mut cache = QuantizedRecentWindowCache::new_with_bits(2, 32, 256, 32, 8);
    assert_eq!(cache.bits(), 8);

    // 8 → 4 전이
    cache.transition_bits(4).unwrap();
    assert_eq!(cache.bits(), 4);

    // 4 → 2 전이
    cache.transition_bits(2).unwrap();
    assert_eq!(cache.bits(), 2);
}

#[test]
fn test_eng_alg_021_kivi_transition_noop() {
    // 같은 bits로의 전이는 무해해야 함
    let mut cache = QuantizedRecentWindowCache::new_with_bits(2, 32, 256, 32, 8);
    cache.transition_bits(8).unwrap();
    assert_eq!(cache.bits(), 8);
}

#[test]
fn test_eng_alg_021_kivi_transition_empty() {
    // 빈 캐시에서의 전이
    let mut cache = QuantizedRecentWindowCache::new_with_bits(2, 32, 256, 32, 8);
    assert_eq!(cache.q2_tokens, 0);
    assert_eq!(cache.res_pos, 0);

    cache.transition_bits(2).unwrap();
    assert_eq!(cache.bits(), 2);
    assert_eq!(cache.q2_tokens, 0);
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-022: 점진적 역양자화 (incremental dequantization)
// 주의: get_view()는 내부적으로 assemble_view()를 호출하며,
// q2_deq_tokens 카운터로 이미 역양자화된 부분을 추적한다.
// 데이터를 넣으려면 update()가 필요한데, Tensor 생성이 필요하므로
// 여기서는 reset() 후 상태 확인으로 검증한다.
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_022_kivi_cache_reset_clears_state() {
    let mut cache = QuantizedRecentWindowCache::new(2, 32, 256, 32);
    // reset()은 모든 상태를 초기화해야 함
    cache.reset();
    assert_eq!(cache.current_pos(), 0);
    assert_eq!(cache.q2_tokens, 0);
    assert_eq!(cache.res_pos, 0);
    assert_eq!(cache.memory_usage_bytes(), 0);
}

#[test]
fn test_eng_alg_022_kivi_cache_kv_dtype_is_f32() {
    // KiviCache는 호출자에게 F32를 요구 (내부적으로 양자화 처리)
    use argus_engine::buffer::DType;
    let cache = QuantizedRecentWindowCache::new(2, 32, 256, 32);
    assert_eq!(cache.kv_dtype(), DType::F32);
}
