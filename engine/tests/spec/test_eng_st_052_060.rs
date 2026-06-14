//! ENG-ST-052 ~ ENG-ST-060: Strategy + Conflict Resolution
//!
//! 3개 ResilienceStrategy (Thermal, Energy, Compute) 반응 검증 및
//! resolve_conflicts() 충돌 해소 규칙 검증.
//!
//! MemoryStrategy는 α-W-3에서 폐기됨 — memory 압력은 Pressure scalar로 처리.

use argus_engine::resilience::ResilienceStrategy;
use argus_engine::resilience::strategy::{ComputeStrategy, EnergyStrategy, ThermalStrategy};
use argus_shared::{
    ComputeReason, EnergyReason, EngineCommand, Level, RecommendedBackend, SystemSignal,
};

// ── 시그널 헬퍼 ──

fn thermal_signal(level: Level, throttle_ratio: f64) -> SystemSignal {
    SystemSignal::ThermalAlert {
        level,
        temperature_mc: 75000,
        throttling_active: throttle_ratio < 1.0,
        throttle_ratio,
    }
}

fn energy_signal(level: Level, reason: EnergyReason) -> SystemSignal {
    SystemSignal::EnergyConstraint {
        level,
        reason,
        power_budget_mw: 5000,
    }
}

fn compute_signal(level: Level, backend: RecommendedBackend) -> SystemSignal {
    SystemSignal::ComputeGuidance {
        level,
        recommended_backend: backend,
        reason: ComputeReason::Balanced,
        cpu_usage_pct: 50.0,
        gpu_usage_pct: 50.0,
    }
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-052: ThermalStrategy — 레벨별 반응
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_052_thermal_emergency_suspends() {
    let mut strategy = ThermalStrategy::new();
    let commands = strategy.react(&thermal_signal(Level::Emergency, 0.3));
    assert_eq!(commands.len(), 1);
    assert!(matches!(commands[0], EngineCommand::Suspend));
}

#[test]
fn test_eng_st_052_thermal_critical_throttles_proportionally() {
    let mut strategy = ThermalStrategy::new();
    // throttle_ratio=0.5 → delay = (1.0-0.5)*100 = 50ms
    let commands = strategy.react(&thermal_signal(Level::Critical, 0.5));
    assert_eq!(commands.len(), 2);
    match &commands[1] {
        EngineCommand::Throttle { delay_ms } => assert_eq!(*delay_ms, 50),
        _ => panic!("Expected Throttle"),
    }
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-052: EnergyStrategy — 레벨별 반응
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_052_energy_normal_restores() {
    let mut strategy = EnergyStrategy::new();
    let commands = strategy.react(&energy_signal(Level::Normal, EnergyReason::Charging));
    assert_eq!(commands.len(), 1);
    assert!(matches!(commands[0], EngineCommand::RestoreDefaults));
}

#[test]
fn test_eng_st_052_energy_emergency_suspends() {
    let mut strategy = EnergyStrategy::new();
    let commands = strategy.react(&energy_signal(
        Level::Emergency,
        EnergyReason::BatteryCritical,
    ));
    assert_eq!(commands.len(), 1);
    assert!(matches!(commands[0], EngineCommand::Suspend));
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-052: ComputeStrategy — 레벨별 반응
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_052_compute_critical_switches_backend() {
    let mut strategy = ComputeStrategy::new();
    let commands = strategy.react(&compute_signal(Level::Critical, RecommendedBackend::Cpu));
    assert_eq!(commands.len(), 1);
    match &commands[0] {
        EngineCommand::SwitchHw { device } => assert_eq!(device, "cpu"),
        _ => panic!("Expected SwitchHw"),
    }
}

#[test]
fn test_eng_st_052_compute_warning_does_not_switch() {
    let mut strategy = ComputeStrategy::new();
    let commands = strategy.react(&compute_signal(Level::Warning, RecommendedBackend::Cpu));
    assert!(commands.is_empty());
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-060: resolve_conflicts — 충돌 해소 규칙
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_060_suspend_overrides_all() {
    let commands = vec![
        EngineCommand::KvEvictH2o { keep_ratio: 0.50 },
        EngineCommand::SwitchHw {
            device: "cpu".to_string(),
        },
        EngineCommand::Suspend,
        EngineCommand::Throttle { delay_ms: 100 },
    ];
    let result = argus_engine::resilience::strategy::resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], EngineCommand::Suspend));
}

#[test]
fn test_eng_st_060_restore_only_when_no_other_constraints() {
    let commands = vec![
        EngineCommand::RestoreDefaults,
        EngineCommand::KvEvictH2o { keep_ratio: 0.85 },
    ];
    let result = argus_engine::resilience::strategy::resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], EngineCommand::KvEvictH2o { .. }));
}

#[test]
fn test_eng_st_060_restore_alone_passes_through() {
    let commands = vec![EngineCommand::RestoreDefaults];
    let result = argus_engine::resilience::strategy::resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], EngineCommand::RestoreDefaults));
}

#[test]
fn test_eng_st_060_cpu_always_wins_over_gpu() {
    let commands = vec![
        EngineCommand::SwitchHw {
            device: "gpu".to_string(),
        },
        EngineCommand::SwitchHw {
            device: "cpu".to_string(),
        },
    ];
    let result = argus_engine::resilience::strategy::resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    match &result[0] {
        EngineCommand::SwitchHw { device } => assert_eq!(device, "cpu"),
        _ => panic!("Expected SwitchHw"),
    }
}

#[test]
fn test_eng_st_060_largest_delay_wins() {
    let commands = vec![
        EngineCommand::Throttle { delay_ms: 30 },
        EngineCommand::Throttle { delay_ms: 100 },
        EngineCommand::Throttle { delay_ms: 50 },
    ];
    let result = argus_engine::resilience::strategy::resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    match &result[0] {
        EngineCommand::Throttle { delay_ms } => assert_eq!(*delay_ms, 100),
        _ => panic!("Expected Throttle"),
    }
}
