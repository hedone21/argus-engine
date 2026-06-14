//! MSG-068, MGR-DAT-076 (Phase 2): Engine process 자신의 GPU 사용률 측정.
//!
//! Phase 1에서는 placeholder(항상 0.0)였으나 Phase 2에서 OpenCL queue
//! profiling(`CL_QUEUE_PROFILING_ENABLE`) 기반 실측으로 확장되었다.
//!
//! # 개요
//!
//! OpenCL 백엔드가 `CL_QUEUE_PROFILING_ENABLE`로 command queue를 구성하면
//! 각 커널 dispatch에 대해 `CL_PROFILING_COMMAND_{START,END}` 이벤트 정보가
//! 기록된다. Engine은 decode loop에서 주기적으로 이 이벤트를 drain하여
//! `(end - start)` ns를 누적한다. `GpuSelfMeter::sample(wall_elapsed)`는
//! 이 누적값을 wall-clock elapsed로 나누어 `[0.0, 1.0]` 사용률을 반환한다.
//!
//! # 정책
//!
//! - **Opt-in only**: Adreno에서 profiling 활성화 시 ~54 ms/token 오버헤드.
//!   CLI 플래그(`--heartbeat-gpu-profile`)로 명시적으로 켜야 한다.
//! - **Clamp**: 반환값은 `[0.0, 1.0]`로 clamp된다 (INV-091).
//! - **Fallback**: 측정 실패/meter 미주입 시 0.0 (INV-092, Heartbeat 송출을
//!   절대 차단하지 않음).
//! - **Warm-up**: 첫 샘플은 이전 기준점이 없으므로 0.0 반환.
//! - **비침투**: LuaPolicy `ctx.engine.gpu_pct`로만 노출. Pressure6D /
//!   EwmaReliefTable 계산에는 여전히 섞이지 않는다.

use std::time::Duration;

/// Engine process 자가 GPU 사용률 측정기 추상화.
///
/// `sample(wall_elapsed)`는 가장 최근 샘플과 현재 사이의 GPU busy 시간을
/// wall-clock 경과로 나눈 `[0.0, 1.0]` 범위 값을 반환한다. 구현은 측정 실패
/// 시 0.0 fallback을 보장해야 한다 (INV-092).
pub trait GpuSelfMeter: Send + Sync {
    /// 현재 사용률 샘플을 반환한다.
    ///
    /// * `wall_elapsed` — 직전 샘플 이후 실제 경과 시간. 호출자(executor)가
    ///   측정 주기를 관리한다. 0 또는 음수 등 비정상 값이 들어오면 0.0 반환.
    ///
    /// 반환값은 `[0.0, 1.0]`로 clamp된다 (INV-091). 측정 실패 시 0.0 (INV-092).
    fn sample(&self, wall_elapsed: Duration) -> f64;
}

/// 항상 0.0을 반환하는 no-op meter. 테스트 및 CPU-only 구성에서 사용한다.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpGpuMeter;

impl GpuSelfMeter for NoOpGpuMeter {
    fn sample(&self, _wall_elapsed: Duration) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_meter_always_returns_zero() {
        // SPEC: MSG-068 (Phase 2 default), INV-092
        let m = NoOpGpuMeter;
        assert_eq!(m.sample(Duration::from_millis(0)), 0.0);
        assert_eq!(m.sample(Duration::from_millis(1000)), 0.0);
        assert_eq!(m.sample(Duration::from_secs(3600)), 0.0);
    }
}
