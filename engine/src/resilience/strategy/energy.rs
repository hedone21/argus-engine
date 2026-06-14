use super::ResilienceStrategy;
use crate::resilience::signal::{EngineCommand, Level, SystemSignal};

/// Energy constraint response strategy.
/// Reduces power consumption proportional to severity (ENG-ST-052).
pub struct EnergyStrategy;

impl Default for EnergyStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl ResilienceStrategy for EnergyStrategy {
    fn react(&mut self, signal: &SystemSignal) -> Vec<EngineCommand> {
        let SystemSignal::EnergyConstraint { level, .. } = signal else {
            return vec![];
        };

        match level {
            Level::Normal => vec![EngineCommand::RestoreDefaults],
            Level::Warning => vec![EngineCommand::SwitchHw {
                device: "cpu".to_string(),
            }],
            // 구 LimitTokens{64} drop (ENG-ST-052 α-W-3): EngineCommand 등가 부재.
            Level::Critical => vec![
                EngineCommand::SwitchHw {
                    device: "cpu".to_string(),
                },
                EngineCommand::Throttle { delay_ms: 30 },
            ],
            // 구 RejectNew drop (ENG-ST-052 α-W-3) — Emergency stop-intent 는 Suspend 가 흡수.
            Level::Emergency => vec![EngineCommand::Suspend],
        }
    }

    fn name(&self) -> &str {
        "energy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::signal::EnergyReason;

    fn energy_signal(level: Level, reason: EnergyReason) -> SystemSignal {
        SystemSignal::EnergyConstraint {
            level,
            reason,
            power_budget_mw: 5000,
        }
    }

    #[test]
    fn test_energy_normal_restores() {
        let mut strategy = EnergyStrategy::new();
        let commands = strategy.react(&energy_signal(Level::Normal, EnergyReason::Charging));
        assert_eq!(commands.len(), 1);
        assert!(matches!(commands[0], EngineCommand::RestoreDefaults));
    }

    #[test]
    fn test_energy_emergency_suspends() {
        let mut strategy = EnergyStrategy::new();
        let commands = strategy.react(&energy_signal(
            Level::Emergency,
            EnergyReason::BatteryCritical,
        ));
        assert_eq!(commands.len(), 1);
        assert!(matches!(commands[0], EngineCommand::Suspend));
    }
}
