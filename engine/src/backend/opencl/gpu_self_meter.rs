//! OpenCL queue profiling 기반 GPU 사용률 측정기.
//!
//! `GpuSelfMeter` trait 구현 — OpenCL event profiling (`CL_QUEUE_PROFILING_ENABLE`)
//! 기반 실측. trait 정의와 no-op 구현은 `resilience::gpu_self_meter`에 유지.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

// L1 backend impl이 resilience 도메인의 `GpuSelfMeter` trait 정의만 import — trait
// inversion 자체는 이미 적용된 형태(impl-only)로 lint path-only 판정 false positive.
// LAYER-EXEMPT: cross_cutting_trait_usage
use crate::resilience::gpu_self_meter::GpuSelfMeter;

/// OpenCL queue profiling 기반 GPU 사용률 누적기.
///
/// 백엔드 측 `flush_and_aggregate_profile()`이 매 decode step 후
/// `record_busy_ns()`로 누적한 값을 `sample()`이 drain하여 wall-clock으로
/// 정규화한다. 샘플링 시 누적값은 0으로 reset된다.
#[derive(Debug, Default)]
pub struct OpenClEventGpuMeter {
    /// 누적 GPU busy nanoseconds. drain-on-sample 방식(snapshot + reset).
    busy_ns: AtomicU64,
}

impl OpenClEventGpuMeter {
    /// 새 meter를 만든다. 누적값은 0에서 시작한다.
    pub fn new() -> Self {
        Self {
            busy_ns: AtomicU64::new(0),
        }
    }

    /// 백엔드가 flush 시점에 호출하는 누적 엔트리. `delta_ns`는 단일 kernel
    /// event의 `(end - start)` 값이며 여러 호출에 걸쳐 saturating add 된다.
    pub fn record_busy_ns(&self, delta_ns: u64) {
        // fetch_add는 wrap-around 하지만 실제 GPU 누적이 u64를 overflow할
        // 가능성은 없으므로(584년 @ 1 GHz) 단순 fetch_add로 충분.
        self.busy_ns.fetch_add(delta_ns, Ordering::Relaxed);
    }

    /// 테스트용 현재 누적값을 (reset 없이) 읽는다.
    #[cfg(test)]
    pub(crate) fn peek_busy_ns(&self) -> u64 {
        self.busy_ns.load(Ordering::Relaxed)
    }
}

impl GpuSelfMeter for OpenClEventGpuMeter {
    fn sample(&self, wall_elapsed: Duration) -> f64 {
        // snapshot + reset (drain semantics)
        let busy_ns = self.busy_ns.swap(0, Ordering::Relaxed);
        let wall_ns = wall_elapsed.as_nanos();
        if wall_ns == 0 {
            return 0.0;
        }
        let ratio = busy_ns as f64 / wall_ns as f64;
        clamp_unit(ratio)
    }
}

impl GpuSelfMeter for Arc<OpenClEventGpuMeter> {
    fn sample(&self, wall_elapsed: Duration) -> f64 {
        (**self).sample(wall_elapsed)
    }
}

fn clamp_unit(x: f64) -> f64 {
    if !x.is_finite() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opencl_meter_first_sample_is_zero() {
        // SPEC: MSG-068 warm-up, INV-092
        let m = OpenClEventGpuMeter::new();
        assert_eq!(m.sample(Duration::from_millis(1000)), 0.0);
    }

    #[test]
    fn opencl_meter_records_and_drains() {
        // SPEC: MSG-068
        let m = OpenClEventGpuMeter::new();
        // 500ms busy out of 1000ms wall = 0.5
        m.record_busy_ns(500_000_000);
        let v = m.sample(Duration::from_millis(1000));
        assert!((v - 0.5).abs() < 1e-9, "expected 0.5, got {v}");
        // drain: 두 번째 샘플은 0.0
        assert_eq!(m.sample(Duration::from_millis(1000)), 0.0);
    }

    #[test]
    fn opencl_meter_clamps_over_one() {
        // SPEC: INV-091
        let m = OpenClEventGpuMeter::new();
        // busy > wall_elapsed (theoretically impossible, but must clamp)
        m.record_busy_ns(2_000_000_000);
        let v = m.sample(Duration::from_millis(1000));
        assert_eq!(v, 1.0);
    }

    #[test]
    fn opencl_meter_zero_wall_returns_zero() {
        // SPEC: INV-092 (degenerate input → 0.0 fallback)
        let m = OpenClEventGpuMeter::new();
        m.record_busy_ns(500_000_000);
        assert_eq!(m.sample(Duration::from_nanos(0)), 0.0);
    }

    #[test]
    fn opencl_meter_accumulates_multiple_records() {
        // SPEC: MSG-068
        let m = OpenClEventGpuMeter::new();
        m.record_busy_ns(100_000_000);
        m.record_busy_ns(150_000_000);
        m.record_busy_ns(50_000_000);
        assert_eq!(m.peek_busy_ns(), 300_000_000);
        let v = m.sample(Duration::from_millis(1000));
        assert!((v - 0.3).abs() < 1e-9, "expected 0.3, got {v}");
    }

    #[test]
    fn arc_wrapper_forwards_to_inner() {
        // 테스트: Arc<OpenClEventGpuMeter>도 GpuSelfMeter 구현.
        let m = Arc::new(OpenClEventGpuMeter::new());
        m.record_busy_ns(250_000_000);
        let v = GpuSelfMeter::sample(&m, Duration::from_millis(1000));
        assert!((v - 0.25).abs() < 1e-9);
    }

    #[test]
    fn arc_dyn_trait_object_works() {
        // executor가 저장하는 형태(Arc<dyn GpuSelfMeter>)가 동작.
        let inner = Arc::new(OpenClEventGpuMeter::new());
        inner.record_busy_ns(750_000_000);
        let dyn_meter: Arc<dyn GpuSelfMeter> = inner.clone();
        let v = dyn_meter.sample(Duration::from_millis(1000));
        assert!((v - 0.75).abs() < 1e-9);
    }

    #[test]
    fn nan_inf_safety() {
        // clamp_unit 안전망: NaN/Inf는 0.0으로.
        assert_eq!(clamp_unit(f64::NAN), 0.0);
        assert_eq!(clamp_unit(f64::INFINITY), 0.0);
        assert_eq!(clamp_unit(f64::NEG_INFINITY), 0.0);
        assert_eq!(clamp_unit(-0.5), 0.0);
        assert_eq!(clamp_unit(1.5), 1.0);
        assert_eq!(clamp_unit(0.42), 0.42);
    }
}
