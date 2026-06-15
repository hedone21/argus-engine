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

use crate::format::KVCacheFormat;
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

/// `Pressure::band()` 의 Warning 하한 scalar (pipeline.rs cutoff `50..=74 => Warning`).
const KV_FILL_WARNING_SCALAR: u8 = 50;

/// `pos/max_seq_len >= high_water_pct/100` 정수 비교 (float 회피). `max_seq_len==0` 이면 false.
#[inline]
fn at_high_water(pos: usize, max_seq_len: usize, high_water_pct: u32) -> bool {
    max_seq_len > 0 && (pos as u64) * 100 >= (high_water_pct as u64) * (max_seq_len as u64)
}

/// KV-fill graded `PressureSource` — turn 경계가 없는 단일 프롬프트 디코드용 eviction 트리거.
///
/// argus-cli 처럼 manager 도 메모리 압력도 없는 happy-path 단일 프롬프트는 [`LocalPressureSource`]
/// 가 항상 Normal 을 보고하므로 pressure-driven
/// [`EvictionStage`](crate::stages::kv::eviction::EvictionStage)(`min_band=Warning`)가 발동하지
/// 않는다(chat 은 turn 경계 `on_turn_end` 로 evict — 단일 프롬프트엔 그 훅이 없다). 본 source 는
/// **메모리 대신 KV 점유율**을 압력으로 환산해, 점유율이 high-water 를 넘으면 `Warning` 밴드를
/// 보고한다. Persistent EvictionStage(episode edge-trigger)가 이를 받아 1회 prune → pos 가
/// high-water 밑으로 떨어지면 Normal 로 복귀 → re-arm 하는 sawtooth 로 KV 를 묶는다.
///
/// **천장은 동적 `capacity()` 가 아니라 고정 `max_seq_len`**: KVCache 의 `capacity()` 는 버퍼
/// 관리 상태라 디코드 중 출렁인다 — grow 시 doubling 으로 pos 를 앞질러 키우고, prune 후엔
/// `KVCache::shrink_to_fit` 가 ~1.5×current_pos(64-aligned)로 줄인다. `pos/capacity()` 는 이
/// 출렁임에 묶여 진짜 overflow 천장(= 할당 cap = `max_seq_len`, decode loop `kv_capacity` 와 동일
/// 출처)에 대한 안정적 high-water 를 주지 못한다(같은 fill ratio 도 capacity 변동에 따라 다른
/// 절대 pos 에서 발생). 따라서 고정 `max_seq_len` 을 기준으로 삼아 트리거를 결정한다.
///
/// **8-step 캐시 여유**: decode loop 은 `PRESSURE_QUERY_INTERVAL`(=8) step 마다 pressure 를
/// 샘플하므로 트리거 평가가 최대 8 토큰 지연될 수 있다. 기본 high-water 85% 는 `max_seq_len` 이
/// 약 50 이상이면 발화~prune 사이 headroom(>15%·max)이 그 지연을 충분히 덮는다.
pub struct KvFillPressureSource {
    /// layer-0 KV format handle — `current_pos()` 로 라이브 점유 토큰 수를 읽는다(syscall 0).
    handle: Arc<dyn KVCacheFormat>,
    /// overflow 천장(= `--max-seq-len`).
    max_seq_len: usize,
    /// 발화 high-water (점유율 percent, 0–100).
    high_water_pct: u32,
}

impl KvFillPressureSource {
    pub fn new(handle: Arc<dyn KVCacheFormat>, max_seq_len: usize, high_water_pct: u32) -> Self {
        Self {
            handle,
            max_seq_len,
            high_water_pct: high_water_pct.min(100),
        }
    }
}

impl PressureSource for KvFillPressureSource {
    fn pressure(&self) -> Pressure {
        if at_high_water(
            self.handle.current_pos(),
            self.max_seq_len,
            self.high_water_pct,
        ) {
            Pressure::new(KV_FILL_WARNING_SCALAR)
        } else {
            Pressure::default()
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

    // ─── KvFillPressureSource ─────────────────────────────────────────────

    /// `pos*100 >= pct*max` 경계 검증. max=384, pct=85 → pct*max=32640 → pos>=327 발화.
    #[test]
    fn kv_fill_high_water_boundary() {
        assert!(!at_high_water(0, 384, 85));
        assert!(!at_high_water(100, 384, 85));
        assert!(!at_high_water(326, 384, 85), "326*100=32600 < 32640");
        assert!(at_high_water(327, 384, 85), "327*100=32700 >= 32640");
        assert!(at_high_water(384, 384, 85), "가득 차면 발화");
        // max_seq_len==0 → 항상 false (0 나눗셈/스퓨리어스 발화 방지).
        assert!(!at_high_water(0, 0, 85));
        assert!(!at_high_water(100, 0, 85));
    }

    /// KvFill 발화 scalar 가 Persistent EvictionStage 가 기대하는 Warning 밴드로 사상되는지.
    #[test]
    fn kv_fill_warning_scalar_is_warning_band() {
        assert_eq!(Pressure::new(KV_FILL_WARNING_SCALAR).band(), Level::Warning);
    }
}
