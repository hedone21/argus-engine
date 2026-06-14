//! Phase β-3 commit B — `EvictionStage` 발화 ↔ v1 `try_evict` 산출 **등가** integration test.
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.2.1 (가)(나) + roadmap β-3 commit B 게이트.
//!
//! **자기 비교 금지 (roadmap:77)** — anchor 는 stage 기계 없이 v1 live 경로의 inner op
//! (`CacheManager::force_evict(&mut [KVCache])`, model_forward.rs:524-531 의 `try_evict` 본문)를
//! 직접 호출한 산출이다. stage 는 동일 캐시·정책으로 `EvictionStage::one_shot` →
//! `PipelineRegistry::dispatch(KvMutate)` 를 거친 산출이다. 둘의 per-layer `current_pos` +
//! K/V valid-region byte 가 bit-identical 임을 증명한다.
//!
//! 범위: {sliding, h2o, streaming} × {F32, F16, Q4_0} = 9종 + min-floor 발화 경계 1종.
//! `compact_parity.rs`(src private)의 make_cache(deterministic byte)/valid_region 패턴을 여기
//! 재구현한다(pub 표면만 사용 — `KVCache` pub field + `take_inner` pub).

use std::sync::Arc;

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::{Buffer, DType};
use argus_engine::kv::cache_manager::CacheManager;
use argus_engine::kv::eviction::stage_registry::{
    StageBackedPolicy, ensure_builtin_stages_registered, make_stage,
};
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::kv::standard_format::StandardFormat;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::observability::profile::OpProfiler;
use argus_engine::pipeline::{
    LifecyclePhase, PipelineDispatcher, Pressure, StageContext, StepInfo,
};
use argus_engine::resilience::sys_monitor::NoOpMonitor;
use argus_engine::session::pipeline_registry::PipelineRegistry;
use argus_engine::shape::Shape;
use argus_engine::stages::kv::eviction::EvictionStage;
use argus_engine::tensor::Tensor;
use technique_api::StageParams;

const KV_HEADS: usize = 1;
/// head_dim=32 = QK4_0 → 위치당 정확히 1 Q4_0 block (compact_parity 상수 차용).
const HEAD_DIM: usize = 32;
const MAX_SEQ: usize = 128;

fn bytes_per_pos(dtype: DType) -> usize {
    let elems = KV_HEADS * HEAD_DIM;
    match dtype {
        DType::F32 => elems * 4,
        DType::F16 => elems * 2,
        DType::Q4_0 => {
            (elems / argus_engine::quant::QK4_0)
                * std::mem::size_of::<argus_engine::quant::BlockQ4_0>()
        }
        other => panic!("unsupported dtype: {other:?}"),
    }
}

/// 위치당 distinct byte 패턴(`(p+1) as u8`, K/V 구분 +128) SeqMajor KVCache.
/// compact_parity.rs::make_cache 패턴 재구현 (pub 표면만).
fn make_cache(dtype: DType, n_tokens: usize) -> KVCache {
    let bpp = bytes_per_pos(dtype);
    let total_bytes = MAX_SEQ * bpp;
    let k_buf = Arc::new(SharedBuffer::new(total_bytes, dtype));
    let v_buf = Arc::new(SharedBuffer::new(total_bytes, dtype));
    // SAFETY: k_buf/v_buf 는 방금 할당한 total_bytes 크기 버퍼. n_tokens*bpp ≤ total_bytes.
    unsafe {
        let kp = k_buf.as_mut_ptr();
        let vp = v_buf.as_mut_ptr();
        for p in 0..n_tokens {
            let byte = (p + 1) as u8;
            for b in 0..bpp {
                *kp.add(p * bpp + b) = byte;
                *vp.add(p * bpp + b) = byte.wrapping_add(128);
            }
        }
    }
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
    let k = Tensor::new(shape.clone(), k_buf, backend.clone());
    let v = Tensor::new(shape, v_buf, backend);
    let mut cache = KVCache::new(k, v, MAX_SEQ);
    cache.current_pos = n_tokens;
    cache
}

/// 유효 영역 `[0..current_pos)` 의 raw byte 를 K/V 각각 추출 (compact_parity::valid_region 재구현).
fn valid_region(cache: &KVCache) -> (Vec<u8>, Vec<u8>) {
    let nb = cache.current_pos * bytes_per_pos(cache.k_buffer.buffer().dtype());
    // SAFETY: nb ≤ 물리 버퍼 크기 (current_pos ≤ MAX_SEQ). read-only 복사.
    unsafe {
        let k = std::slice::from_raw_parts(cache.k_buffer.buffer().as_mut_ptr() as *const u8, nb)
            .to_vec();
        let v = std::slice::from_raw_parts(cache.v_buffer.buffer().as_mut_ptr() as *const u8, nb)
            .to_vec();
        (k, v)
    }
}

/// build_bench_loop.rs:90-128 미러 — 정책 이름으로 [`CacheManager`] 구성.
/// CacheManager 2벌은 각각 독립 생성(상태 공유 차단).
///
/// β-4: `EvictionStage::one_shot` 이 `Arc<Mutex<CacheManager>>` 를 받으므로 stage 용 CM 은
/// 래핑한다. anchor 용(`force_evict` 직접 호출)은 plain CM 그대로 사용 — 검증 로직·anchor 무변.
fn make_cache_manager(policy_name: &str, target_ratio: f32) -> CacheManager {
    ensure_builtin_stages_registered().expect("builtin stages register");
    let params = StageParams {
        eviction_window: 2048,
        protected_prefix: 4,
        keep_ratio: 0.5,
        sink_size: 4,
        streaming_window: 6,
    };
    let stage = make_stage(policy_name, &params)
        .unwrap_or_else(|| panic!("make_stage({policy_name}) returned None"));
    let policy = Box::new(StageBackedPolicy::new(stage));
    CacheManager::new(policy, Box::new(NoOpMonitor), usize::MAX, target_ratio)
}

/// 핵심 등가 검증. anchor(v1 force_evict 직접) vs stage(EvictionStage dispatch) 의 per-layer
/// current_pos + K/V valid-region byte 가 bit-identical 임을 단언한다.
///
/// `n_layers ≥ 2` → W1(enumerate 순서 == layer idx) 커버.
///
/// `expect_prune`: 비-vacuous 가드 — true 면 anchor 가 실제로 prune(current_pos < n_tokens)
/// 했는지 단언한다(둘 다 no-op 이라 byte 동일로 trivially-pass 하는 것 방지). min-floor 케이스는
/// false(불변 등가).
fn assert_equivalence(
    policy_name: &str,
    dtype: DType,
    n_tokens: usize,
    target_ratio: f32,
    expect_prune: bool,
) {
    let n_layers = 2usize;
    let label = format!("{policy_name}/{dtype:?}");

    // ── anchor: v1 try_evict inner op (CacheManager::force_evict, stage 기계 없음) ──
    let mut caches_a: Vec<KVCache> = (0..n_layers).map(|_| make_cache(dtype, n_tokens)).collect();
    let cm_a = make_cache_manager(policy_name, target_ratio);
    cm_a.force_evict(&mut caches_a, target_ratio)
        .unwrap_or_else(|e| panic!("{label}: anchor force_evict failed: {e}"));

    // 비-vacuous: 9종은 prune 발생, min-floor 는 불변.
    if expect_prune {
        assert!(
            caches_a[0].current_pos < n_tokens,
            "{label}: 비-vacuous 위반 — prune 미발생 (current_pos={} == n_tokens={n_tokens})",
            caches_a[0].current_pos
        );
    } else {
        assert_eq!(
            caches_a[0].current_pos, n_tokens,
            "{label}: min-floor no-op 기대 (current_pos 불변)"
        );
    }

    // ── stage: wrap → EvictionStage::one_shot → dispatch(KvMutate) ──
    let handles: Vec<Arc<StandardFormat>> = (0..n_layers)
        .map(|i| Arc::new(StandardFormat::new(i, make_cache(dtype, n_tokens))))
        .collect();
    let cm_b = std::sync::Mutex::new(make_cache_manager(policy_name, target_ratio));
    let stage = EvictionStage::one_shot(handles.clone(), Arc::new(cm_b), target_ratio);
    let registry = PipelineRegistry::new();
    registry.submit(Arc::new(stage));

    let mut profiler = OpProfiler::new();
    let mut ctx = StageContext {
        step: StepInfo {
            pos: n_tokens,
            decode_step: 0,
            pressure: Pressure::new(0),
            prev_token: 0,
        },
        profiler: &mut profiler,
    };
    registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
    // OneShot Consumed → GC.
    assert_eq!(
        registry.len(),
        0,
        "{label}: OneShot Consumed 후 registry GC"
    );

    // ── 비교: per-layer current_pos + valid-region byte ──
    for (i, anchor) in caches_a.iter().enumerate() {
        // stage 쪽 inner 를 꺼내 비교 (take_inner 는 β-3 에서 pub).
        let staged = handles[i].take_inner();
        assert_eq!(
            anchor.current_pos, staged.current_pos,
            "{label} layer{i}: current_pos mismatch (anchor={}, stage={})",
            anchor.current_pos, staged.current_pos
        );
        let (ka, va) = valid_region(anchor);
        let (kb, vb) = valid_region(&staged);
        assert_eq!(ka, kb, "{label} layer{i}: K valid-region byte mismatch");
        assert_eq!(va, vb, "{label} layer{i}: V valid-region byte mismatch");
        handles[i].put_inner(staged);
    }
}

// ── 9종: {sliding, h2o, streaming} × {F32, F16, Q4_0} ──
// n_tokens=120, ratio=0.3 → tokens_to_remove=84 ≥ MIN_EVICT_TOKENS(64) → 비-vacuous prune.

const N_TOKENS: usize = 120;
const RATIO: f32 = 0.3;

#[test]
fn sliding_equiv_f32() {
    assert_equivalence("sliding", DType::F32, N_TOKENS, RATIO, true);
}
#[test]
fn sliding_equiv_f16() {
    assert_equivalence("sliding", DType::F16, N_TOKENS, RATIO, true);
}
#[test]
fn sliding_equiv_q4_0() {
    assert_equivalence("sliding", DType::Q4_0, N_TOKENS, RATIO, true);
}

#[test]
fn h2o_equiv_f32() {
    // score-free → recency degrade 경로 (v1 AB-1 동일 조건).
    assert_equivalence("h2o", DType::F32, N_TOKENS, RATIO, true);
}
#[test]
fn h2o_equiv_f16() {
    assert_equivalence("h2o", DType::F16, N_TOKENS, RATIO, true);
}
#[test]
fn h2o_equiv_q4_0() {
    assert_equivalence("h2o", DType::Q4_0, N_TOKENS, RATIO, true);
}

#[test]
fn streaming_equiv_f32() {
    assert_equivalence("streaming", DType::F32, N_TOKENS, RATIO, true);
}
#[test]
fn streaming_equiv_f16() {
    assert_equivalence("streaming", DType::F16, N_TOKENS, RATIO, true);
}
#[test]
fn streaming_equiv_q4_0() {
    assert_equivalence("streaming", DType::Q4_0, N_TOKENS, RATIO, true);
}

// ── +1 min-floor: n_tokens=16 (floor 미달) → 양쪽 모두 no-op, current_pos·byte 불변 ──

#[test]
fn min_floor_noop_both_sides() {
    // n_tokens=16, ratio=0.3 → target_len=4, tokens_to_remove=12 < MIN_EVICT_TOKENS(64) → no-op.
    // anchor·stage 양쪽 모두 current_pos 불변 + valid-region byte 불변임을 등가로 확인.
    assert_equivalence("sliding", DType::F32, 16, 0.3, false);
}
