//! Phase β-5 게이트 2 — pressure-driven Persistent `EvictionStage` 발화 ↔ v1 `force_evict` 등가.
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.1/§5.4 G4 + roadmap β-5 게이트 2.
//!
//! mock `SystemMonitor`(주입 가능한 `MemAvailable`)로 [`LocalPressureSource`] 를 구성하고,
//! Warning 이상 압력에서 Persistent [`EvictionStage`] 가 `KvMutate` 에서 발화함을 보인다.
//! 그 prune 산출이 v1 `CacheManager::force_evict` 직접 호출(anchor)과 **bit-identical** 임을
//! 증명한다(자기 비교 금지 — beta3 패턴 차용). Normal 압력에서는 무발화(cache 불변).
//!
//! 추가로 [`LocalPressureSource`] 가 mock monitor 의 `available` → `Pressure.band()` 매핑을
//! 정확히 수행함을 end-to-end 로 확인한다(t/t÷2/t÷4 경계 = canonical cutoff 일원화 증명).

use std::sync::{Arc, Mutex};

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
    LifecyclePhase, PipelineDispatcher, Pressure, PressureSource, StageContext, StepInfo,
};
use argus_engine::resilience::sys_monitor::{MemoryStats, NoOpMonitor, SystemMonitor};
use argus_engine::session::local_pressure::LocalPressureSource;
use argus_engine::session::pipeline_registry::PipelineRegistry;
use argus_engine::shape::Shape;
use argus_engine::stages::kv::eviction::EvictionStage;
use argus_engine::tensor::Tensor;
use argus_shared::Level;
use technique_api::StageParams;

const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 32;
const MAX_SEQ: usize = 128;
const N_TOKENS: usize = 120;
const RATIO: f32 = 0.3;
const THRESHOLD: usize = 1024 * 1024 * 1024; // 1 GiB

// ── mock monitor: 주입 가능한 MemAvailable ──

struct FixedMonitor {
    available: usize,
}

impl SystemMonitor for FixedMonitor {
    fn mem_stats(&self) -> anyhow::Result<MemoryStats> {
        Ok(MemoryStats {
            total: usize::MAX,
            available: self.available,
            free: self.available,
        })
    }
}

// ── cache helpers (beta3_eviction_stage_equivalence.rs 패턴 재사용) ──

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

fn make_cache(dtype: DType, n_tokens: usize) -> KVCache {
    let bpp = bytes_per_pos(dtype);
    let total_bytes = MAX_SEQ * bpp;
    let k_buf = Arc::new(SharedBuffer::new(total_bytes, dtype));
    let v_buf = Arc::new(SharedBuffer::new(total_bytes, dtype));
    // SAFETY: 방금 할당한 total_bytes 버퍼, n_tokens*bpp ≤ total_bytes.
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

/// LocalPressureSource(mock monitor) 가 band 임계 이상의 압력을 내고, Persistent EvictionStage 가
/// KvMutate 에서 발화한 산출 == v1 force_evict 직접(anchor) 산출 (bit-identical).
fn assert_pressure_driven_equivalence(policy_name: &str, dtype: DType) {
    let n_layers = 2usize;
    let label = format!("{policy_name}/{dtype:?}");

    // ── anchor: v1 force_evict 직접 (stage 기계 없음) ──
    let mut caches_a: Vec<KVCache> = (0..n_layers).map(|_| make_cache(dtype, N_TOKENS)).collect();
    let cm_a = make_cache_manager(policy_name, RATIO);
    cm_a.force_evict(&mut caches_a, RATIO)
        .unwrap_or_else(|e| panic!("{label}: anchor force_evict failed: {e}"));
    assert!(
        caches_a[0].current_pos < N_TOKENS,
        "{label}: 비-vacuous 위반 — anchor prune 미발생"
    );

    // ── source: Warning 압력 (available = threshold/2 → band()=Warning) ──
    let source = LocalPressureSource::new(
        Arc::new(FixedMonitor {
            available: THRESHOLD / 2,
        }),
        THRESHOLD,
    );
    let pressure = source.pressure();
    assert_eq!(
        pressure.band(),
        Level::Warning,
        "{label}: mock monitor(available=t/2) → band()=Warning 기대"
    );

    // ── stage: Persistent EvictionStage(min_band=Warning) → dispatch(KvMutate) ──
    let handles: Vec<Arc<StandardFormat>> = (0..n_layers)
        .map(|i| Arc::new(StandardFormat::new(i, make_cache(dtype, N_TOKENS))))
        .collect();
    let cm_b = Mutex::new(make_cache_manager(policy_name, RATIO));
    let stage = EvictionStage::persistent(handles.clone(), Arc::new(cm_b), RATIO, Level::Warning);
    let registry = PipelineRegistry::new();
    registry.submit(Arc::new(stage));

    let mut profiler = OpProfiler::new();
    let mut ctx = StageContext {
        step: StepInfo {
            pos: N_TOKENS,
            decode_step: 0,
            // source 가 산출한 graded 압력 (Warning).
            pressure,
            prev_token: 0,
        },
        profiler: &mut profiler,
    };
    registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
    // Persistent → Consumed 안 함 → 상주 (GC 없음).
    assert_eq!(
        registry.len(),
        1,
        "{label}: Persistent 은 발화 후에도 registry 상주(GC 안 함)"
    );

    // ── 비교: per-layer current_pos + valid-region byte ──
    for (i, anchor) in caches_a.iter().enumerate() {
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

// ── 게이트 2: {sliding, h2o, streaming} × {F32, F16, Q4_0} pressure-driven 등가 ──

#[test]
fn pressure_driven_sliding_f32() {
    assert_pressure_driven_equivalence("sliding", DType::F32);
}
#[test]
fn pressure_driven_sliding_q4_0() {
    assert_pressure_driven_equivalence("sliding", DType::Q4_0);
}
#[test]
fn pressure_driven_h2o_f16() {
    assert_pressure_driven_equivalence("h2o", DType::F16);
}
#[test]
fn pressure_driven_streaming_f32() {
    assert_pressure_driven_equivalence("streaming", DType::F32);
}

/// Normal 압력(available >= threshold) → Persistent EvictionStage 무발화 (cache 불변).
#[test]
fn normal_pressure_does_not_fire() {
    let handle = Arc::new(StandardFormat::new(0, make_cache(DType::F32, N_TOKENS)));
    let source = LocalPressureSource::new(
        Arc::new(FixedMonitor {
            available: THRESHOLD,
        }),
        THRESHOLD,
    );
    let pressure = source.pressure();
    assert_eq!(pressure.band(), Level::Normal, "available=t → Normal");

    let cm = Mutex::new(make_cache_manager("sliding", RATIO));
    let stage =
        EvictionStage::persistent(vec![handle.clone()], Arc::new(cm), RATIO, Level::Warning);
    let registry = PipelineRegistry::new();
    registry.submit(Arc::new(stage));

    let mut profiler = OpProfiler::new();
    let mut ctx = StageContext {
        step: StepInfo {
            pos: N_TOKENS,
            decode_step: 0,
            pressure,
            prev_token: 0,
        },
        profiler: &mut profiler,
    };
    registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);

    let inner = handle.take_inner();
    assert_eq!(
        inner.current_pos, N_TOKENS,
        "Normal 압력에서 prune 없음 (cache 불변)"
    );
}

/// LocalPressureSource band 매핑 end-to-end: t/t÷2/t÷4 경계 = canonical cutoff 일원화.
#[test]
fn local_source_band_mapping_e2e() {
    let cases = [
        (THRESHOLD, Level::Normal),
        (THRESHOLD / 2, Level::Warning),
        (THRESHOLD / 4, Level::Critical),
        (THRESHOLD / 4 - 1, Level::Emergency),
    ];
    for (available, expected) in cases {
        let source = LocalPressureSource::new(Arc::new(FixedMonitor { available }), THRESHOLD);
        assert_eq!(
            source.pressure().band(),
            expected,
            "available={available} → band mismatch",
        );
        // CacheManager determine_pressure_level 과 동일 산식(Pressure::from_mem_available) 확인.
        assert_eq!(
            source.pressure(),
            Pressure::from_mem_available(available, THRESHOLD),
        );
    }
}
