use super::signal::{EngineCommand, RecommendedBackend, SystemSignal};

pub mod compute;
pub mod energy;
pub mod thermal;

pub use compute::ComputeStrategy;
pub use energy::EnergyStrategy;
pub use thermal::ThermalStrategy;

/// 추상 backend 역할 → `SwitchHw` device 문자열. `Any` → `None`(switch 생략).
///
/// 구체 GPU backend(opencl/cuda) 해석은 dispatcher/`Hardware` 책임이다(`DeviceTarget::Gpu`
/// resolve, §3.5) — 여기선 추상 역할 문자열("cpu"/"gpu")만 낸다 (ENG-ST-052).
pub(crate) fn switch_device(backend: RecommendedBackend) -> Option<&'static str> {
    match backend {
        RecommendedBackend::Cpu => Some("cpu"),
        RecommendedBackend::Gpu => Some("gpu"),
        RecommendedBackend::Any => None,
    }
}

/// Signal reaction strategy interface (front-door ①, §0.4 / ENG-ST-053).
///
/// 출력은 이산 명령 `EngineCommand` 다 (§5.4 — 구 `ResilienceAction` 폐기, EngineCommand 단일
/// 어휘). manager-less 자율 정책(`LocalPolicy`)의 정책 단위. graded magnitude(eviction 강도)는
/// 이 채널이 아니라 `Pressure` scalar(§5.1)가 담당하므로, react 는 *mode* 출력
/// (switch/suspend/throttle/restore)만 낸다. 구 `mode: OperatingMode` 인자는 dead 라 제거됨.
pub trait ResilienceStrategy: Send + Sync {
    /// Receive signal and return discrete commands to execute. Empty Vec = no action.
    fn react(&mut self, signal: &SystemSignal) -> Vec<EngineCommand>;

    /// Strategy name (for logging).
    fn name(&self) -> &str;
}

/// Merge discrete commands from multiple strategies, resolving conflicts (ENG-ST-060, 4규칙).
///
/// - R1. `Suspend` overrides everything → `[Suspend]`
/// - R2. `RestoreDefaults` only when no other constraints
/// - R3. `SwitchHw`: `device == "cpu"` precedence (안전 우선), 아니면 마지막 값
/// - R4. `Throttle`: largest `delay_ms` wins
/// - 그 외 명령(`SetTargetTbt`/`Resume`/`Kv*`/...)은 원순서 보존 pass-through (vacuous-agnostic).
pub fn resolve_conflicts(commands: Vec<EngineCommand>) -> Vec<EngineCommand> {
    if commands.is_empty() {
        return vec![];
    }

    let mut max_delay = 0u64;
    let mut target_device: Option<String> = None;
    let mut has_suspend = false;
    let mut has_restore = false;
    let mut passthrough: Vec<EngineCommand> = Vec::new();

    for cmd in commands {
        match cmd {
            EngineCommand::Suspend => has_suspend = true,
            EngineCommand::RestoreDefaults => has_restore = true,
            EngineCommand::Throttle { delay_ms } => max_delay = max_delay.max(delay_ms),
            EngineCommand::SwitchHw { device } => {
                // R3: "cpu" always wins, else last value.
                target_device = Some(
                    if target_device.as_deref() == Some("cpu") || device == "cpu" {
                        "cpu".to_string()
                    } else {
                        device
                    },
                );
            }
            other => passthrough.push(other),
        }
    }

    // R1: Suspend overrides all.
    if has_suspend {
        return vec![EngineCommand::Suspend];
    }

    let mut result = Vec::new();
    if let Some(device) = target_device {
        result.push(EngineCommand::SwitchHw { device });
    }
    if max_delay > 0 {
        result.push(EngineCommand::Throttle {
            delay_ms: max_delay,
        });
    }
    result.extend(passthrough);

    // R2: RestoreDefaults only when no other constraint.
    if has_restore && result.is_empty() {
        return vec![EngineCommand::RestoreDefaults];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input_returns_empty() {
        assert!(resolve_conflicts(vec![]).is_empty());
    }

    #[test]
    fn test_cpu_always_wins_over_gpu() {
        let result = resolve_conflicts(vec![
            EngineCommand::SwitchHw {
                device: "gpu".to_string(),
            },
            EngineCommand::SwitchHw {
                device: "cpu".to_string(),
            },
        ]);
        assert_eq!(result.len(), 1);
        match &result[0] {
            EngineCommand::SwitchHw { device } => assert_eq!(device, "cpu"),
            _ => panic!("Expected SwitchHw"),
        }
    }

    #[test]
    fn test_largest_delay_wins() {
        let result = resolve_conflicts(vec![
            EngineCommand::Throttle { delay_ms: 30 },
            EngineCommand::Throttle { delay_ms: 100 },
            EngineCommand::Throttle { delay_ms: 50 },
        ]);
        assert_eq!(result.len(), 1);
        match &result[0] {
            EngineCommand::Throttle { delay_ms } => assert_eq!(*delay_ms, 100),
            _ => panic!("Expected Throttle"),
        }
    }

    #[test]
    fn test_suspend_overrides_all() {
        let result = resolve_conflicts(vec![
            EngineCommand::SwitchHw {
                device: "cpu".to_string(),
            },
            EngineCommand::Suspend,
            EngineCommand::Throttle { delay_ms: 100 },
        ]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], EngineCommand::Suspend));
    }

    #[test]
    fn test_restore_only_when_no_other_constraints() {
        let result = resolve_conflicts(vec![
            EngineCommand::RestoreDefaults,
            EngineCommand::SwitchHw {
                device: "cpu".to_string(),
            },
        ]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], EngineCommand::SwitchHw { .. }));
    }

    #[test]
    fn test_restore_alone_passes_through() {
        let result = resolve_conflicts(vec![EngineCommand::RestoreDefaults]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], EngineCommand::RestoreDefaults));
    }
}
