use super::{ResilienceStrategy, switch_device};
use crate::resilience::signal::{EngineCommand, Level, RecommendedBackend, SystemSignal};

/// Compute guidance response strategy.
/// Handles backend switching recommendations from Manager (ENG-ST-052).
pub struct ComputeStrategy {
    current_backend: RecommendedBackend,
}

impl Default for ComputeStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeStrategy {
    pub fn new() -> Self {
        Self {
            current_backend: RecommendedBackend::Any,
        }
    }
}

impl ResilienceStrategy for ComputeStrategy {
    fn react(&mut self, signal: &SystemSignal) -> Vec<EngineCommand> {
        let SystemSignal::ComputeGuidance {
            level,
            recommended_backend,
            ..
        } = signal
        else {
            return vec![];
        };

        match level {
            Level::Normal => {
                self.current_backend = RecommendedBackend::Any;
                vec![EngineCommand::RestoreDefaults]
            }
            Level::Warning => {
                // Prepare only. Don't switch yet at Warning level.
                self.current_backend = *recommended_backend;
                vec![]
            }
            Level::Critical => {
                if *recommended_backend != self.current_backend {
                    self.current_backend = *recommended_backend;
                    match switch_device(*recommended_backend) {
                        Some(dev) => vec![EngineCommand::SwitchHw {
                            device: dev.to_string(),
                        }],
                        // Any 는 구체 device 아님 → switch 생략 (Hardware 가 활성 device 유지).
                        None => vec![],
                    }
                } else {
                    // Already on recommended backend. Add throttle.
                    vec![EngineCommand::Throttle { delay_ms: 50 }]
                }
            }
            Level::Emergency => vec![
                EngineCommand::SwitchHw {
                    device: "cpu".to_string(),
                },
                EngineCommand::Throttle { delay_ms: 100 },
            ],
        }
    }

    fn name(&self) -> &str {
        "compute"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::signal::ComputeReason;

    fn compute_signal(level: Level, backend: RecommendedBackend) -> SystemSignal {
        SystemSignal::ComputeGuidance {
            level,
            recommended_backend: backend,
            reason: ComputeReason::Balanced,
            cpu_usage_pct: 50.0,
            gpu_usage_pct: 50.0,
        }
    }

    #[test]
    fn test_compute_critical_switches_backend() {
        let mut strategy = ComputeStrategy::new();
        let commands = strategy.react(&compute_signal(Level::Critical, RecommendedBackend::Cpu));
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            EngineCommand::SwitchHw { device } => assert_eq!(device, "cpu"),
            _ => panic!("Expected SwitchHw"),
        }
    }

    #[test]
    fn test_compute_warning_does_not_switch() {
        let mut strategy = ComputeStrategy::new();
        let commands = strategy.react(&compute_signal(Level::Warning, RecommendedBackend::Cpu));
        assert!(commands.is_empty());
    }
}
