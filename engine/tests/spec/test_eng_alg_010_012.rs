//! ENG-ALG-010 / ENG-ALG-011: H2O + Sliding Window Eviction
//!
//! H2O 3-partition budget split, evict_with_scores, 순서 보존.
//! SlidingWindow protected prefix, 최소 prefix 강제.
//!
//! (ENG-ALG-012 D2O layer-alloc variance/budget 테스트는 해당 dead 기계
//! `d2o_layer_alloc`(D2OVarianceCollector/compute_budgets) 제거와 함께 삭제됨.)
//!
//! 주의: EvictionPolicy::evict()는 KVCache 내부 데이터를 직접 조작하므로
//! 통합 테스트에서 CpuBackend/SharedBuffer로 KVCache를 생성해야 한다.

use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::{Buffer, DType};
use argus_engine::kv::eviction::EvictionPolicy;
use argus_engine::kv::eviction::stage_registry::{
    StageBackedPolicy, make_stage, sliding_backed_policy,
};
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use argus_extension_api::StageParams;
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

/// H2O was extracted to the `h2o` plugin crate; build the stage by registry name and wrap it as
/// a legacy EvictionPolicy (StageBackedPolicy) — the same seam production uses to resolve "h2o".
fn h2o_policy(keep_ratio: f32, protected_prefix: usize) -> StageBackedPolicy {
    StageBackedPolicy::new(
        make_stage(
            "h2o",
            &StageParams {
                keep_ratio,
                protected_prefix,
                ..Default::default()
            },
        )
        .expect("h2o stage registered"),
    )
}

#[test]
fn test_eng_alg_010_h2o_no_eviction_when_below_target() {
    let policy = h2o_policy(0.5, 4);
    let mut cache = make_cache_with_data(10);
    // target_len이 current_pos 이상이면 eviction 발생하지 않음
    policy.evict(&mut cache, 20).unwrap();
    assert_eq!(cache.current_pos, 10);
}

#[test]
fn test_eng_alg_010_h2o_evict_preserves_prefix() {
    let policy = h2o_policy(0.5, 4);
    let mut cache = make_cache_with_data(30);

    // target_len=15: prefix 4개는 보호되어야 함
    policy.evict(&mut cache, 15).unwrap();
    assert!(cache.current_pos <= 15);
    assert!(cache.current_pos >= 6); // prefix(4) + 최소 recent(2)

    // prefix 데이터가 보존되었는지 확인
    let k_data = cache.k_buffer.as_slice::<f32>();
    assert_eq!(k_data[0], 1.0); // position 0
}

// `test_eng_alg_010_h2o_should_evict_always_false` was removed when H2O was extracted to the
// `h2o` plugin: the stage no longer owns the WHEN decision (StageBackedPolicy delegates the
// trigger to the engine's MIN_EVICT/target-ratio guard), so should_evict() is no longer the
// policy's concern. The signal-driven trigger is exercised by the cache_manager guard tests.

// ══════════════════════════════════════════════════════════════
// ENG-ALG-010/C01: H2O evict_with_scores — 중요도 기반 보존
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_010_c01_h2o_evict_with_scores_preserves_important() {
    let policy = h2o_policy(0.5, 4);
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
    let policy = sliding_backed_policy(10, 0); // prefix는 4로 클램프
    let mut cache = make_cache_with_data(20);
    policy.evict(&mut cache, 5).unwrap();
    assert_eq!(cache.current_pos, 14); // max_keep = 10 + 4 = 14
}

#[test]
fn test_eng_alg_011_sliding_evict_with_protected_prefix() {
    let policy = sliding_backed_policy(4, 4);
    let mut cache = make_cache_with_data(12);
    policy.evict(&mut cache, 6).unwrap();
    assert_eq!(cache.current_pos, 8); // max_keep = 4 + 4 = 8

    // prefix 데이터 보존 확인
    let k_data = cache.k_buffer.as_slice::<f32>();
    assert_eq!(k_data[0], 1.0); // position 0
}

#[test]
fn test_eng_alg_011_sliding_evict_no_action_needed() {
    let policy = sliding_backed_policy(20, 0);
    let mut cache = make_cache_with_data(10);
    policy.evict(&mut cache, 20).unwrap();
    assert_eq!(cache.current_pos, 10); // 변경 없음
}

// `test_eng_alg_011_minimum_protected_prefix_enforced` was removed when SlidingWindow was extracted
// to the `sliding-window` plugin: it asserted `should_evict()` thresholds, but the stage no longer
// owns the WHEN decision (StageBackedPolicy delegates the trigger to the engine MIN_EVICT/target
// guard, same as the h2o case above). The 4-token minimum-prefix clamp is now pinned by the
// `sliding-window` crate's own unit tests (`min_prefix_clamped_to_four`).
