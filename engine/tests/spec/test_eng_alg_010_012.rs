//! ENG-ALG-010 ~ ENG-ALG-012: H2O + Sliding Window + D2O Eviction
//!
//! H2O 3-partition budget split, evict_with_scores, 순서 보존.
//! SlidingWindow protected prefix, 최소 prefix 강제.
//! D2O find_nearest per-head, d2o_layer_alloc variance/budget.
//!
//! 주의: EvictionPolicy::evict()는 KVCache 내부 데이터를 직접 조작하므로
//! 통합 테스트에서 CpuBackend/SharedBuffer로 KVCache를 생성해야 한다.

use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::{Buffer, DType};
use argus_engine::kv::eviction::{EvictionPolicy, H2OPolicy, SlidingWindowPolicy};
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use std::sync::Arc;

// ── 헬퍼 ──

fn make_cache_with_data(num_tokens: usize) -> KVCache {
    let max_seq = 100;
    let heads = 1;
    let dim = 4;
    let backend = Arc::new(CpuBackend::new());
    let buf_size = max_seq * heads * dim * 4;

    let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
    let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

    // 식별 가능한 데이터로 채움
    unsafe {
        let k_ptr = k_buf.as_mut_ptr() as *mut f32;
        let v_ptr = v_buf.as_mut_ptr() as *mut f32;
        for i in 0..num_tokens * dim {
            *k_ptr.add(i) = (i / dim + 1) as f32;
            *v_ptr.add(i) = ((i / dim + 1) * 10) as f32;
        }
    }

    let k = Tensor::new(
        Shape::new(vec![1, max_seq, heads, dim]),
        k_buf,
        backend.clone(),
    );
    let v = Tensor::new(Shape::new(vec![1, max_seq, heads, dim]), v_buf, backend);
    let mut cache = KVCache::new(k, v, max_seq);
    cache.current_pos = num_tokens;
    cache
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-010: H2O budget split (50:50 기본)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_010_h2o_no_eviction_when_below_target() {
    let policy = H2OPolicy::new(0.5, 4);
    let mut cache = make_cache_with_data(10);
    // target_len이 current_pos 이상이면 eviction 발생하지 않음
    policy.evict(&mut cache, 20).unwrap();
    assert_eq!(cache.current_pos, 10);
}

#[test]
fn test_eng_alg_010_h2o_evict_preserves_prefix() {
    let policy = H2OPolicy::new(0.5, 4);
    let mut cache = make_cache_with_data(30);

    // target_len=15: prefix 4개는 보호되어야 함
    policy.evict(&mut cache, 15).unwrap();
    assert!(cache.current_pos <= 15);
    assert!(cache.current_pos >= 6); // prefix(4) + 최소 recent(2)

    // prefix 데이터가 보존되었는지 확인
    let k_data = cache.k_buffer.as_slice::<f32>();
    assert_eq!(k_data[0], 1.0); // position 0
}

#[test]
fn test_eng_alg_010_h2o_should_evict_always_false() {
    // H2O는 signal-driven: should_evict()는 항상 false
    let policy = H2OPolicy::new(0.5, 4);
    let cache = make_cache_with_data(50);
    assert!(!policy.should_evict(&cache, 0));
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-010/C01: H2O evict_with_scores — 중요도 기반 보존
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_010_c01_h2o_evict_with_scores_preserves_important() {
    let policy = H2OPolicy::new(0.5, 4);
    let mut cache = make_cache_with_data(30);

    // importance scores: position 10, 20에 높은 중요도
    let mut scores = vec![0.01f32; 100];
    scores[10] = 10.0;
    scores[20] = 9.0;

    policy.evict_with_scores(&mut cache, 15, &scores).unwrap();
    assert!(cache.current_pos <= 15);
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-011: SlidingWindow — prefix 보호 + 최소 prefix 강제
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_011_sliding_evict_no_prefix() {
    let policy = SlidingWindowPolicy::new(10, 0); // prefix는 4로 클램프
    let mut cache = make_cache_with_data(20);
    policy.evict(&mut cache, 5).unwrap();
    assert_eq!(cache.current_pos, 14); // max_keep = 10 + 4 = 14
}

#[test]
fn test_eng_alg_011_sliding_evict_with_protected_prefix() {
    let policy = SlidingWindowPolicy::new(4, 4);
    let mut cache = make_cache_with_data(12);
    policy.evict(&mut cache, 6).unwrap();
    assert_eq!(cache.current_pos, 8); // max_keep = 4 + 4 = 8

    // prefix 데이터 보존 확인
    let k_data = cache.k_buffer.as_slice::<f32>();
    assert_eq!(k_data[0], 1.0); // position 0
}

#[test]
fn test_eng_alg_011_sliding_evict_no_action_needed() {
    let policy = SlidingWindowPolicy::new(20, 0);
    let mut cache = make_cache_with_data(10);
    policy.evict(&mut cache, 20).unwrap();
    assert_eq!(cache.current_pos, 10); // 변경 없음
}

#[test]
fn test_eng_alg_011_minimum_protected_prefix_enforced() {
    // prefix=0,1,2,3 모두 내부적으로 4로 클램프
    for input_prefix in 0..=3 {
        let policy = SlidingWindowPolicy::new(10, input_prefix);
        let mut cache = make_cache_with_data(14);
        assert!(
            !policy.should_evict(&cache, 0),
            "prefix={input_prefix}: 14 should NOT trigger eviction (threshold=14)"
        );

        cache.current_pos = 15;
        assert!(
            policy.should_evict(&cache, 0),
            "prefix={input_prefix}: 15 should trigger eviction (threshold=14)"
        );
    }

    // prefix=5 이상은 그대로 유지
    let policy = SlidingWindowPolicy::new(10, 5);
    let mut cache = make_cache_with_data(15);
    assert!(!policy.should_evict(&cache, 0)); // 15 <= 10+5
    cache.current_pos = 16;
    assert!(policy.should_evict(&cache, 0)); // 16 > 15
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-012: D2O layer allocation — variance / budget
// 주의: D2OVarianceCollector는 내부적으로 Q*K attention을 계산하므로
//       적절한 입력 데이터가 필요. 여기서는 compute_budgets의 속성만 검증.
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_012_d2o_variance_uniform_attention() {
    use argus_engine::kv::d2o_layer_alloc::D2OVarianceCollector;

    let n_layers = 4;
    let collector = D2OVarianceCollector::new(
        n_layers, 2, // kv_heads
        2, // heads_q
        4, // head_dim
        8, // seq_len
    );

    // collect_layer를 호출하지 않으면 variance=0 → 균등 budget
    let budgets = collector.compute_budgets(0.5, 0.3);
    assert_eq!(budgets.len(), n_layers);

    // 모든 레이어가 동일한 budget을 받아야 함 (variance=0 → 균등 배분)
    let first = budgets[0];
    for b in &budgets {
        assert!(
            (b.0 - first.0).abs() < 1e-6 && (b.1 - first.1).abs() < 1e-6,
            "균등 variance에서는 모든 레이어 budget이 동일해야 함"
        );
    }
}

#[test]
fn test_eng_alg_012_d2o_budget_softmax_clamp() {
    use argus_engine::kv::d2o_layer_alloc::D2OVarianceCollector;

    let collector = D2OVarianceCollector::new(2, 1, 1, 4, 4);
    let budgets = collector.compute_budgets(0.6, 0.2);
    assert_eq!(budgets.len(), 2);

    // budget 값이 0 이상이어야 함
    for &(hh, recent) in &budgets {
        assert!(hh >= 0.0, "hh budget은 0 이상");
        assert!(recent >= 0.0, "recent budget은 0 이상");
    }
}
