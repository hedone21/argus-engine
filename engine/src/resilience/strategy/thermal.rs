use super::ResilienceStrategy;
use crate::resilience::signal::{EngineCommand, Level, SystemSignal};

/// Thermal alert response strategy.
/// Reduces compute intensity proportional to thermal severity (ENG-ST-052).
pub struct ThermalStrategy;

impl Default for ThermalStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ThermalStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl ResilienceStrategy for ThermalStrategy {
    fn react(&mut self, signal: &SystemSignal) -> Vec<EngineCommand> {
        let SystemSignal::ThermalAlert {
            level,
            throttle_ratio,
            ..
        } = signal
        else {
            return vec![];
        };

        match level {
            Level::Normal => vec![EngineCommand::RestoreDefaults],
            Level::Warning => vec![EngineCommand::SwitchHw {
                device: "cpu".to_string(),
            }],
            Level::Critical => {
                let delay = ((1.0 - throttle_ratio) * 100.0) as u64;
                // 구 LimitTokens{64} drop (ENG-ST-052 α-W-3): EngineCommand 등가 부재.
                vec![
                    EngineCommand::SwitchHw {
                        device: "cpu".to_string(),
                    },
                    EngineCommand::Throttle { delay_ms: delay },
                ]
            }
            Level::Emergency => vec![EngineCommand::Suspend],
        }
    }

    fn name(&self) -> &str {
        "thermal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thermal_signal(level: Level, throttle_ratio: f64) -> SystemSignal {
        SystemSignal::ThermalAlert {
            level,
            temperature_mc: 75000,
            throttling_active: throttle_ratio < 1.0,
            throttle_ratio,
        }
    }

    #[test]
    fn test_thermal_emergency_suspends() {
        let mut strategy = ThermalStrategy::new();
        let commands = strategy.react(&thermal_signal(Level::Emergency, 0.3));
        assert_eq!(commands.len(), 1);
        assert!(matches!(commands[0], EngineCommand::Suspend));
    }

    #[test]
    fn test_thermal_critical_throttles_proportionally() {
        let mut strategy = ThermalStrategy::new();
        // throttle_ratio 0.5 → delay = (1.0 - 0.5) * 100 = 50ms
        let commands = strategy.react(&thermal_signal(Level::Critical, 0.5));
        assert_eq!(commands.len(), 2);
        match &commands[1] {
            EngineCommand::Throttle { delay_ms } => assert_eq!(*delay_ms, 50),
            _ => panic!("Expected Throttle"),
        }
    }
}
