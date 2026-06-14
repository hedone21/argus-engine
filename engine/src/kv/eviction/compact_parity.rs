//! (Phase α-K substep 3c-evict) **정식 게이트** — `EvictionPolicy::plan_keep` 가 산출한 keep-list 를
//! `KVCacheFormat::compact` 로 적용한 버퍼가, 같은 정책의 in-place `evict*(&mut KVCache)` 버퍼와
//! **bit-identical** 임을 증명한다 (interior-mutability eviction 의 등가성).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §9.1-EVICT (b) 등가식 + §9.1-EVICT-DECISION
//! (γ 확정 — host unit test 가 1차 증명; device 는 sanity 한정). 범위 = Sliding · H2O ·
//! StreamingLLM · NoEviction × {F32, F16, Q4_0} (per-head H2O+ · 가중 merge D2O 는 §9.1-EVICT-DEFER).
//!
//! 왜 host 가 정식인가 (F4): `compact` → `compact_keep_positions` → `shift_positions` →
//! `backend.buffer_shift` 는 in-place sliding `evict` 가 쓰는 바로 그 연산이라 GPU 가 새로 증명할
//! 것이 없고, `CpuBackend` 가 동일 `buffer_shift` 경로를 결정적·taint-free 로 검증한다.

use std::sync::Arc;

use technique_api::{KVCachePlan, KeepSpec};

use crate::backend::cpu::CpuBackend;
use crate::buffer::{Buffer, DType};
use crate::kv::eviction::stage_registry::{execute_kv_plan, uniform_to_weighted};
use crate::kv::eviction::{
    EvictionPolicy, H2OPolicy, NoEvictionPolicy, SlidingWindowPolicy, StreamingLLMPolicy,
};
use crate::kv::kv_cache::KVCache;
use crate::memory::host::shared::SharedBuffer;
use crate::shape::Shape;
use crate::tensor::Tensor;

const KV_HEADS: usize = 1;
/// head_dim=32 = QK4_0 → 위치당 정확히 1 Q4_0 block (kv_heads*head_dim % QK4_0 == 0 보장).
const HEAD_DIM: usize = 32;
const MAX_SEQ: usize = 128;

/// 위치당 distinct 한 byte 패턴을 가진 SeqMajor KVCache 생성 (`KVCache::new`, memory=None).
///
/// 위치 `p` 의 모든 byte = `(p+1) as u8` (256 wrap). distinct 라 잘못된 keep-list 는 byte 비교로
/// 즉시 잡힌다. dtype 별 위치당 byte 크기만 다르고 shift 의미는 동일.
fn make_cache(dtype: DType, n_tokens: usize) -> KVCache {
    let bytes_per_pos = bytes_per_pos(dtype);
    let total_bytes = MAX_SEQ * bytes_per_pos;
    let k_buf = Arc::new(SharedBuffer::new(total_bytes, dtype));
    let v_buf = Arc::new(SharedBuffer::new(total_bytes, dtype));
    unsafe {
        let kp = k_buf.as_mut_ptr();
        let vp = v_buf.as_mut_ptr();
        for p in 0..n_tokens {
            let byte = (p + 1) as u8;
            for b in 0..bytes_per_pos {
                *kp.add(p * bytes_per_pos + b) = byte;
                *vp.add(p * bytes_per_pos + b) = byte.wrapping_add(128); // K/V 구분
            }
        }
    }
    let backend = Arc::new(CpuBackend::new());
    let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
    let k = Tensor::new(shape.clone(), k_buf, backend.clone());
    let v = Tensor::new(shape, v_buf, backend);
    let mut cache = KVCache::new(k, v, MAX_SEQ);
    cache.current_pos = n_tokens;
    cache
}

fn bytes_per_pos(dtype: DType) -> usize {
    let elems = KV_HEADS * HEAD_DIM;
    match dtype {
        DType::F32 => elems * 4,
        DType::F16 => elems * 2,
        DType::Q4_0 => {
            (elems / crate::quant::QK4_0) * std::mem::size_of::<crate::quant::BlockQ4_0>()
        }
        other => panic!("unsupported dtype in compact_parity test: {other:?}"),
    }
}

/// 유효 영역 `[0..current_pos)` 의 raw byte 를 K/V 각각 추출 (tail 무관 — attention 미독).
fn valid_region(cache: &KVCache) -> (Vec<u8>, Vec<u8>) {
    let nb = cache.current_pos * bytes_per_pos(cache.k_buffer.dtype());
    unsafe {
        let k = std::slice::from_raw_parts(cache.k_buffer.as_mut_ptr() as *const u8, nb).to_vec();
        let v = std::slice::from_raw_parts(cache.v_buffer.as_mut_ptr() as *const u8, nb).to_vec();
        (k, v)
    }
}

/// 핵심 등가성 검증. `evict_inplace` 가 정책의 in-place 경로를 실행하고, `plan_keep` → `compact` 가
/// fmt 경로를 실행한 뒤, 유효 영역 byte + current_pos 가 일치하는지 확인한다.
fn assert_parity<P, F>(
    policy: &P,
    dtype: DType,
    n_tokens: usize,
    target_len: usize,
    importance: Option<&[f32]>,
    evict_inplace: F,
    label: &str,
) where
    P: EvictionPolicy,
    F: Fn(&P, &mut KVCache),
{
    // Path A: in-place evict.
    let mut cache_a = make_cache(dtype, n_tokens);
    evict_inplace(policy, &mut cache_a);

    // Path B: plan_keep → execute_kv_plan (retarget — compact 폐기).
    let mut cache_b = make_cache(dtype, n_tokens);
    let (keep, merges) = policy
        .plan_keep(n_tokens, target_len, importance)
        .unwrap_or_else(|| panic!("{label}: plan_keep returned None (policy unsupported)"));
    // 빌트인 3정책은 빈 merge 를 내므로 변환 결과도 빈 vec (compact_keep_positions 만 실행).
    let plan = KVCachePlan {
        keep: KeepSpec::LayerWide(keep),
        merges: merges.into_iter().map(uniform_to_weighted).collect(),
    };
    execute_kv_plan(&mut cache_b, &plan)
        .unwrap_or_else(|e| panic!("{label}: execute_kv_plan failed: {e}"));

    // current_pos 일치.
    assert_eq!(
        cache_a.current_pos, cache_b.current_pos,
        "{label} [{dtype:?}]: current_pos mismatch (in-place={}, plan={})",
        cache_a.current_pos, cache_b.current_pos
    );

    // 유효 영역 byte bit-identical.
    let (ka, va) = valid_region(&cache_a);
    let (kb, vb) = valid_region(&cache_b);
    assert_eq!(ka, kb, "{label} [{dtype:?}]: K valid-region byte mismatch");
    assert_eq!(va, vb, "{label} [{dtype:?}]: V valid-region byte mismatch");
}

const DTYPES: [DType; 3] = [DType::F32, DType::F16, DType::Q4_0];

#[test]
fn sliding_parity_all_dtypes() {
    // window=10, prefix=4(clamped). current=40, target=20 → prefix>0 shift 경로.
    let policy = SlidingWindowPolicy::new(10, 4);
    for dt in DTYPES {
        assert_parity(
            &policy,
            dt,
            40,
            20,
            None,
            |p, c| p.evict(c, 20).unwrap(),
            "sliding",
        );
    }
}

#[test]
fn sliding_parity_noop_below_threshold() {
    // current <= keep → evict no-op → 전체 보존 keep-list.
    let policy = SlidingWindowPolicy::new(64, 4);
    for dt in DTYPES {
        assert_parity(
            &policy,
            dt,
            20,
            60,
            None,
            |p, c| p.evict(c, 60).unwrap(),
            "sliding-noop",
        );
    }
}

#[test]
fn h2o_fallback_parity_all_dtypes() {
    // importance None → evict() fallback (prefix + recent).
    let policy = H2OPolicy::new(0.5, 4);
    for dt in DTYPES {
        assert_parity(
            &policy,
            dt,
            40,
            20,
            None,
            |p, c| p.evict(c, 20).unwrap(),
            "h2o-fallback",
        );
    }
}

#[test]
fn h2o_scores_parity_all_dtypes() {
    // importance Some → evict_with_scores (prefix + heavy hitters + recent).
    let policy = H2OPolicy::new(0.5, 4);
    let n = 40;
    // distinct descending scores over [0..n): pos 0 highest. evictable=[4..recent_start).
    let importance: Vec<f32> = (0..MAX_SEQ).map(|i| (MAX_SEQ - i) as f32).collect();
    for dt in DTYPES {
        assert_parity(
            &policy,
            dt,
            n,
            20,
            Some(&importance),
            |p, c| p.evict_with_scores(c, 20, &importance).unwrap(),
            "h2o-scores",
        );
    }
}

#[test]
fn h2o_scores_parity_noncontiguous_hh() {
    // 비연속 HH 위치 (여러 batch shift) — compact batching 경로 검증.
    let policy = H2OPolicy::new(0.5, 4);
    let n = 30;
    let mut importance = vec![0.01f32; MAX_SEQ];
    importance[6] = 10.0;
    importance[12] = 9.0;
    importance[19] = 8.0;
    for dt in DTYPES {
        assert_parity(
            &policy,
            dt,
            n,
            16,
            Some(&importance),
            |p, c| p.evict_with_scores(c, 16, &importance).unwrap(),
            "h2o-noncontig",
        );
    }
}

#[test]
fn streaming_parity_all_dtypes() {
    // sink=4, window=6, current=40 → keep=10.
    let policy = StreamingLLMPolicy::new(4, 6);
    for dt in DTYPES {
        assert_parity(
            &policy,
            dt,
            40,
            0,
            None,
            |p, c| p.evict(c, 0).unwrap(),
            "streaming",
        );
    }
}

#[test]
fn streaming_parity_shrunk_window() {
    // target_len < keep_size → window 축소 경로.
    let policy = StreamingLLMPolicy::new(4, 6);
    for dt in DTYPES {
        assert_parity(
            &policy,
            dt,
            40,
            7,
            None,
            |p, c| p.evict(c, 7).unwrap(),
            "streaming-shrunk",
        );
    }
}

#[test]
fn no_eviction_parity_all_dtypes() {
    // evict no-op → 전체 보존, compact 도 무변.
    let policy = NoEvictionPolicy::new();
    for dt in DTYPES {
        assert_parity(
            &policy,
            dt,
            30,
            10,
            None,
            |p, c| p.evict(c, 10).unwrap(),
            "no-eviction",
        );
    }
}

#[test]
fn h2o_plus_plan_keep_unsupported() {
    // H2O+ 는 per-head divergence (§9.1-EVICT-DEFER) → plan_keep default None.
    use crate::kv::eviction::H2OPlusPolicy;
    let policy = H2OPlusPolicy::new(0.5, 4);
    assert!(
        policy.plan_keep(40, 20, None).is_none(),
        "H2O+ 는 단일 layer-wide keep-list 표현 불가 → plan_keep None (deferred)"
    );
}
