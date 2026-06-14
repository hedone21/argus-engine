//! `LocalPressureSource` — memory-only graded `PressureSource` (Phase β-5).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.1/§5.4 G4 + roadmap β-5 item 2.
//!
//! `/proc/meminfo` 의 `MemAvailable` 을 [`Pressure::from_mem_available`] 로 graded scalar(0–100)
//! 로 융합한다. threshold(bytes)는 생성자 인자 — v1 `CacheManager::threshold_bytes` 와 동일 의미
//! (그 미만에서 압력 증가).
//!
//! **canonical cutoff 의 거처**: mem→Pressure 계단 산식(t/t÷2/t÷4)은 [`Pressure::from_mem_available`]
//! 단일 함수가 소유하고, `CacheManager::determine_pressure_level` 도 동일 함수를 경유한다(β-5 ripple).
//! 본 source 는 그 함수를 그대로 위임 호출하므로 cutoff 가 일원화된다.
//!
//! **β 범위 [G4]**: manager-less memory graded 만 β 1급이다. `ManagerPressureSource`(엔진측
//! manager 신호 융합)는 β 밖 후속 — [`PressureSource`](crate::pipeline::PressureSource) trait 이
//! 그 seam 이며, 신규 코드는 본 substep 에서 추가하지 않는다.

use std::sync::Arc;

use crate::pipeline::{Pressure, PressureSource};
use crate::resilience::sys_monitor::SystemMonitor;

/// memory-only graded `PressureSource`.
///
/// [`SystemMonitor`] 를 통해 `MemAvailable` 을 읽어 [`Pressure`] 로 변환한다. monitor 읽기 실패
/// 시(예: `/proc/meminfo` 부재) `Pressure::default()`(=0, Normal)로 강등한다 — 압력 없음으로
/// 간주(보수적: 미발화 쪽).
pub struct LocalPressureSource {
    monitor: Arc<dyn SystemMonitor>,
    /// 이 값 미만에서 압력이 증가하기 시작 (v1 `CacheManager::threshold_bytes` 동일 의미).
    threshold_bytes: usize,
}

impl LocalPressureSource {
    /// `monitor` 로 `MemAvailable` 을 읽고 `threshold_bytes` 기준으로 graded 압력을 산출한다.
    pub fn new(monitor: Arc<dyn SystemMonitor>, threshold_bytes: usize) -> Self {
        Self {
            monitor,
            threshold_bytes,
        }
    }
}

impl PressureSource for LocalPressureSource {
    fn pressure(&self) -> Pressure {
        match self.monitor.mem_stats() {
            Ok(stats) => Pressure::from_mem_available(stats.available, self.threshold_bytes),
            // monitor 실패 → 압력 없음(보수적 미발화). v1 determine_pressure_level 의 mem_stats Err
            // 경로(eviction skip)와 동일 시맨틱.
            Err(_) => Pressure::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::sys_monitor::MemoryStats;
    use argus_shared::Level;
    use std::sync::Mutex;

    /// available 값을 주입 가능한 mock monitor.
    struct MockMonitor {
        available: Mutex<usize>,
        fail: bool,
    }

    impl MockMonitor {
        fn with_available(available: usize) -> Self {
            Self {
                available: Mutex::new(available),
                fail: false,
            }
        }
        fn failing() -> Self {
            Self {
                available: Mutex::new(0),
                fail: true,
            }
        }
    }

    impl SystemMonitor for MockMonitor {
        fn mem_stats(&self) -> anyhow::Result<MemoryStats> {
            if self.fail {
                anyhow::bail!("mock monitor failure");
            }
            let available = *self.available.lock().unwrap();
            Ok(MemoryStats {
                total: usize::MAX,
                available,
                free: available,
            })
        }
    }

    const T: usize = 1024;

    #[test]
    fn maps_mem_available_to_band() {
        // mem >= t → Normal
        let src = LocalPressureSource::new(Arc::new(MockMonitor::with_available(T)), T);
        assert_eq!(src.pressure().band(), Level::Normal);
        // t/2 <= mem < t → Warning
        let src = LocalPressureSource::new(Arc::new(MockMonitor::with_available(T / 2)), T);
        assert_eq!(src.pressure().band(), Level::Warning);
        // t/4 <= mem < t/2 → Critical
        let src = LocalPressureSource::new(Arc::new(MockMonitor::with_available(T / 4)), T);
        assert_eq!(src.pressure().band(), Level::Critical);
        // mem < t/4 → Emergency
        let src = LocalPressureSource::new(Arc::new(MockMonitor::with_available(T / 4 - 1)), T);
        assert_eq!(src.pressure().band(), Level::Emergency);
    }

    #[test]
    fn monitor_failure_yields_zero_pressure() {
        let src = LocalPressureSource::new(Arc::new(MockMonitor::failing()), T);
        assert_eq!(src.pressure(), Pressure::default());
        assert_eq!(src.pressure().band(), Level::Normal);
    }

    /// `from_mem_available` 위임 — `CacheManager::determine_pressure_level` 과 동일 산식 확인.
    #[test]
    fn delegates_to_canonical_cutoff() {
        let src = LocalPressureSource::new(Arc::new(MockMonitor::with_available(T / 3)), T);
        // T/3 = 341, t/4=256 <= 341 < t/2=512 → Critical.
        assert_eq!(
            src.pressure(),
            Pressure::from_mem_available(T / 3, T),
            "source 는 canonical cutoff 함수를 그대로 위임해야 함"
        );
    }
}
