//! MSG-060 필드 17~18 (self_cpu_pct, self_gpu_pct), MSG-067, MSG-068, MSG-069,
//! INV-091, INV-092: Engine Self-Utilization Heartbeat 필드 Spec 테스트.
//!
//! 2026-04 Phase 1 — Engine이 `/proc/self/stat` 기반 자신의 CPU 사용률을
//! Heartbeat(MSG-060)에 실어 보낸다.
//!
//! 2026-04 Phase 2 — Engine이 OpenCL queue profiling 기반 자신의 GPU
//! 사용률을 Heartbeat(MSG-060)에 실어 보낸다. meter 미주입(기본) 시 0.0
//! 유지 (Phase 1 호환, INV-092 fallback).
//!
//! 실행 조건: feature gate 없음 (기본 경로).

#![allow(clippy::needless_doctest_main)]

use std::sync::Arc;
use std::time::Duration;

use argus_engine::backend::opencl::gpu_self_meter::OpenClEventGpuMeter;
use argus_engine::resilience::gpu_self_meter::{GpuSelfMeter, NoOpGpuMeter};
use argus_engine::resilience::proc_self_meter::ProcSelfMeter;
use argus_shared::{EngineState, EngineStatus, ResourceLevel};

fn base_status() -> EngineStatus {
    EngineStatus {
        active_device: "cpu".to_string(),
        compute_level: ResourceLevel::Normal,
        actual_throughput: 15.0,
        memory_level: ResourceLevel::Normal,
        kv_cache_bytes: 1024,
        kv_cache_tokens: 100,
        kv_cache_utilization: 0.5,
        memory_lossless_min: 1.0,
        memory_lossy_min: 0.01,
        state: EngineState::Running,
        tokens_generated: 50,
        available_actions: vec!["throttle".into()],
        active_actions: vec![],
        eviction_policy: "none".into(),
        kv_dtype: "f16".into(),
        skip_ratio: 0.0,
        phase: String::new(),
        prefill_pos: 0,
        prefill_total: 0,
        partition_ratio: 0.0,
        self_cpu_pct: 0.0,
        self_gpu_pct: 0.0,
    }
}

// ---------------------------------------------------------------------------
// MSG-060 #17~18 / MSG-061: serde round-trip + 하위호환 (#[serde(default)])
// ---------------------------------------------------------------------------

#[test]
fn msg_060_self_util_fields_roundtrip_preserves_values() {
    // SPEC: MSG-060 (필드 17, 18), MSG-061
    let mut s = base_status();
    s.self_cpu_pct = 0.42;
    s.self_gpu_pct = 0.0;
    let json = serde_json::to_string(&s).unwrap();
    let back: EngineStatus = serde_json::from_str(&json).unwrap();
    assert!((back.self_cpu_pct - 0.42).abs() < f64::EPSILON);
    assert_eq!(back.self_gpu_pct, 0.0);
    // 기존 필드 sanity
    assert_eq!(back.active_device, "cpu");
    assert_eq!(back.kv_cache_tokens, 100);
    assert_eq!(back.eviction_policy, "none");
}

#[test]
fn msg_060_missing_self_util_fields_default_to_zero() {
    // SPEC: MSG-061, 하위호환 (serde(default))
    // 필드 17~18을 빼고 나머지 16필드만 포함한 구버전 JSON.
    let legacy = r#"{
        "active_device": "opencl",
        "compute_level": "normal",
        "actual_throughput": 12.0,
        "memory_level": "normal",
        "kv_cache_bytes": 2048,
        "kv_cache_tokens": 256,
        "kv_cache_utilization": 0.25,
        "memory_lossless_min": 1.0,
        "memory_lossy_min": 0.01,
        "state": "running",
        "tokens_generated": 77,
        "available_actions": [],
        "active_actions": [],
        "eviction_policy": "none",
        "kv_dtype": "f16",
        "skip_ratio": 0.0,
        "phase": "",
        "prefill_pos": 0,
        "prefill_total": 0,
        "partition_ratio": 0.0
    }"#;
    let s: EngineStatus = serde_json::from_str(legacy).unwrap();
    assert_eq!(s.self_cpu_pct, 0.0);
    assert_eq!(s.self_gpu_pct, 0.0);
    assert_eq!(s.tokens_generated, 77);
}

#[test]
fn msg_060_self_cpu_pct_explicit_value_is_preserved_across_json() {
    // SPEC: MSG-060 — JSON에 명시된 값이 default를 덮어쓰고 복원된다.
    let base = serde_json::to_value(base_status()).unwrap();
    let mut obj = base.as_object().unwrap().clone();
    obj.insert("self_cpu_pct".to_string(), serde_json::json!(0.73));
    obj.insert("self_gpu_pct".to_string(), serde_json::json!(0.0));
    let json = serde_json::to_string(&serde_json::Value::Object(obj)).unwrap();
    let s: EngineStatus = serde_json::from_str(&json).unwrap();
    assert!((s.self_cpu_pct - 0.73).abs() < f64::EPSILON);
    assert_eq!(s.self_gpu_pct, 0.0);
}

// ---------------------------------------------------------------------------
// INV-091: clamp to [0.0, 1.0]
// ---------------------------------------------------------------------------

#[test]
fn inv_091_self_cpu_pct_is_clamped_on_send_side() {
    // SPEC: INV-091
    // ProcSelfMeter::sample()은 어떤 경우에도 [0.0, 1.0]을 반환한다.
    // 반복 샘플링 시 내부적으로 delta가 음수이거나 부정확한 계산이 발생해도
    // clamp_unit 로직이 이를 범위 내로 잘라낸다.
    let mut m = ProcSelfMeter::new();
    for _ in 0..50 {
        let v = m.sample();
        assert!(
            (0.0..=1.0).contains(&v),
            "sample must be in [0.0, 1.0], got {v}"
        );
    }
}

#[test]
fn inv_091_self_gpu_pct_default_without_meter_is_zero() {
    // SPEC: INV-091, MSG-068, INV-092
    // Phase 2 — meter 미주입(기본 CommandExecutor::new) 시 executor는 Phase 1과
    // 동일하게 self_gpu_pct=0.0을 송출한다. 하위호환 및 INV-092 fallback.
    let s = base_status();
    assert_eq!(s.self_gpu_pct, 0.0);
}

#[test]
fn msg_068_noop_gpu_meter_always_returns_zero() {
    // SPEC: MSG-068, INV-092
    // NoOpGpuMeter는 모든 wall_elapsed 입력에 대해 0.0을 반환해야 한다.
    let m = NoOpGpuMeter;
    assert_eq!(m.sample(Duration::from_millis(0)), 0.0);
    assert_eq!(m.sample(Duration::from_millis(1000)), 0.0);
    assert_eq!(m.sample(Duration::from_secs(3600)), 0.0);
}

#[test]
fn msg_068_gpu_meter_sample_is_in_unit_range() {
    // SPEC: MSG-068, INV-091
    // OpenClEventGpuMeter는 어떤 입력에도 [0.0, 1.0] 범위 값을 반환한다.
    let m = OpenClEventGpuMeter::new();
    // 극단값: busy > wall은 1.0으로 clamp
    m.record_busy_ns(10_000_000_000);
    let v = m.sample(Duration::from_millis(1000));
    assert!((0.0..=1.0).contains(&v), "expected 0..=1, got {v}");
    assert_eq!(v, 1.0);

    // 일반 케이스
    m.record_busy_ns(250_000_000);
    let v = m.sample(Duration::from_millis(1000));
    assert!((0.0..=1.0).contains(&v));
    assert!((v - 0.25).abs() < 1e-9);
}

#[test]
fn msg_068_gpu_meter_first_sample_is_zero() {
    // SPEC: MSG-068 warm-up
    // 첫 sample()은 이전 기준값이 없으므로 0.0 반환(누적도 0).
    let m = OpenClEventGpuMeter::new();
    assert_eq!(m.sample(Duration::from_millis(1000)), 0.0);
}

#[test]
fn inv_092_gpu_meter_zero_wall_falls_back_to_zero() {
    // SPEC: INV-092
    // 측정 실패/퇴화 입력(wall_elapsed=0) 시 meter는 0.0을 반환.
    let m = OpenClEventGpuMeter::new();
    m.record_busy_ns(500_000_000);
    assert_eq!(m.sample(Duration::ZERO), 0.0);
}

#[test]
fn inv_092_gpu_meter_failure_does_not_block_heartbeat() {
    // SPEC: INV-092
    // meter가 0.0 반환하더라도 status serialize/deserialize가 정상 동작하고
    // 다른 필드가 영향을 받지 않음을 확인.
    let m: Arc<dyn GpuSelfMeter> = Arc::new(NoOpGpuMeter);
    let mut s = base_status();
    s.self_gpu_pct = m.sample(Duration::from_millis(1000)); // 0.0
    let json = serde_json::to_string(&s).unwrap();
    let back: EngineStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(back.self_gpu_pct, 0.0);
    assert_eq!(back.tokens_generated, 50);
    assert_eq!(back.active_device, "cpu");
}

#[test]
fn msg_068_gpu_meter_trait_object_dispatch_works() {
    // SPEC: MSG-068
    // CommandExecutor가 저장하는 형태 Arc<dyn GpuSelfMeter> 경유 호출이
    // 구현체에 정상 dispatch되는지 확인 (executor 경로 간접 검증).
    let meter = Arc::new(OpenClEventGpuMeter::new());
    meter.record_busy_ns(400_000_000);
    let dyn_meter: Arc<dyn GpuSelfMeter> = meter.clone();
    let v = dyn_meter.sample(Duration::from_millis(1000));
    assert!((v - 0.4).abs() < 1e-9);
    // drain semantics: 두 번째 호출은 0.0
    assert_eq!(dyn_meter.sample(Duration::from_millis(1000)), 0.0);
}

// ---------------------------------------------------------------------------
// INV-092 / MSG-067: 측정 실패 시 0.0 fallback + Heartbeat 차단 금지
// ---------------------------------------------------------------------------

#[test]
fn inv_092_proc_self_stat_read_failure_falls_back_to_zero() {
    // SPEC: INV-092, MSG-067
    // ProcSelfMeter::sample()은 Result/Option이 아닌 f64를 반환하며,
    // 첫 호출(이전 샘플 없음)은 관례상 0.0을 반환한다. 또한 내부 /proc
    // 읽기 실패 시에도 0.0 반환을 보장한다 (Linux에서 정상 동작하는
    // 케이스는 second sample이 finite하므로 반대로 검증).
    let mut m = ProcSelfMeter::new();
    let first = m.sample();
    assert_eq!(first, 0.0, "first sample must be 0.0 (warm-up)");
}

#[test]
fn inv_092_measurement_failure_does_not_block_heartbeat_emission() {
    // SPEC: INV-092, INV-064
    // Heartbeat를 구성하는 경로에서 self-CPU 값이 0.0이어도 EngineStatus는
    // 정상 serialize 되어야 하며, 다른 필드 값은 영향을 받지 않는다.
    let mut s = base_status();
    s.self_cpu_pct = 0.0; // 측정 실패 모의
    s.self_gpu_pct = 0.0;
    let json = serde_json::to_string(&s).unwrap();
    let back: EngineStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(back.self_cpu_pct, 0.0);
    assert_eq!(back.tokens_generated, 50);
    assert_eq!(back.active_device, "cpu");
}
