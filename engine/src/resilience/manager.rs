use std::sync::mpsc;

use super::signal::{EngineCommand, Level, SystemSignal};
use super::state::OperatingMode;
use super::strategy::{
    ComputeStrategy, EnergyStrategy, ResilienceStrategy, ThermalStrategy, resolve_conflicts,
};

/// Cache of latest levels for each signal type.
struct SignalLevels {
    memory: Level,
    compute: Level,
    thermal: Level,
    energy: Level,
}

impl SignalLevels {
    fn new() -> Self {
        Self {
            memory: Level::Normal,
            compute: Level::Normal,
            thermal: Level::Normal,
            energy: Level::Normal,
        }
    }
}

/// Strategy implementations per signal type (discrete-command 산출, ENG-ST-052).
///
/// memory 압력은 strategy 가 아니라 graded `Pressure` scalar(§5.1/§5.4)로 흐르므로 여기 없다
/// (구 `MemoryStrategy` 폐기). thermal/energy/compute 만 *mode* 이산 명령을 낸다.
struct Strategies {
    compute: Box<dyn ResilienceStrategy>,
    thermal: Box<dyn ResilienceStrategy>,
    energy: Box<dyn ResilienceStrategy>,
}

impl Strategies {
    fn new() -> Self {
        Self {
            compute: Box::new(ComputeStrategy::new()),
            thermal: Box::new(ThermalStrategy::new()),
            energy: Box::new(EnergyStrategy::new()),
        }
    }
}

/// Central resilience orchestrator (manager-less 자율 정책의 원형 — `LocalPolicy` `CommandSource`
/// 로 발전 예정, ENG-ST-055).
///
/// Receives signals via mpsc channel, delegates to strategies, resolves conflicts,
/// and returns discrete `EngineCommand`s for the inference loop.
pub struct ResilienceManager {
    rx: mpsc::Receiver<SystemSignal>,
    mode: OperatingMode,
    current_levels: SignalLevels,
    strategies: Strategies,
}

impl ResilienceManager {
    /// Create a new ResilienceManager with default strategies.
    pub fn new(rx: mpsc::Receiver<SystemSignal>) -> Self {
        Self {
            rx,
            mode: OperatingMode::Normal,
            current_levels: SignalLevels::new(),
            strategies: Strategies::new(),
        }
    }

    /// Non-blocking poll: drain all pending signals, process through strategies, resolve
    /// conflicts, and return the final discrete commands. Called once per token (ENG-ST-051).
    pub fn poll(&mut self) -> Vec<EngineCommand> {
        let mut all_commands = Vec::new();

        while let Ok(signal) = self.rx.try_recv() {
            self.process_signal(&signal, &mut all_commands);
        }

        if all_commands.is_empty() {
            return vec![];
        }

        resolve_conflicts(all_commands)
    }

    /// Current operating mode.
    pub fn mode(&self) -> OperatingMode {
        self.mode
    }

    fn process_signal(&mut self, signal: &SystemSignal, commands: &mut Vec<EngineCommand>) {
        // 1. Update level cache (memory level 도 mode 계산에 기여하므로 추적 유지).
        match signal {
            SystemSignal::MemoryPressure { level, .. } => self.current_levels.memory = *level,
            SystemSignal::ComputeGuidance { level, .. } => self.current_levels.compute = *level,
            SystemSignal::ThermalAlert { level, .. } => self.current_levels.thermal = *level,
            SystemSignal::EnergyConstraint { level, .. } => self.current_levels.energy = *level,
        }

        // 2. Recalculate operating mode.
        self.mode = OperatingMode::from_levels(
            self.current_levels.memory,
            self.current_levels.compute,
            self.current_levels.thermal,
            self.current_levels.energy,
        );

        // 3. Delegate to strategy. memory 는 graded(§5.4) → discrete 명령 없음.
        let strategy_commands = match signal {
            SystemSignal::MemoryPressure { .. } => vec![],
            SystemSignal::ComputeGuidance { .. } => self.strategies.compute.react(signal),
            SystemSignal::ThermalAlert { .. } => self.strategies.thermal.react(signal),
            SystemSignal::EnergyConstraint { .. } => self.strategies.energy.react(signal),
        };

        commands.extend(strategy_commands);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::signal::EnergyReason;

    fn send_signal(tx: &mpsc::Sender<SystemSignal>, signal: SystemSignal) {
        tx.send(signal).unwrap();
    }

    #[test]
    fn test_manager_poll_returns_empty_when_no_signals() {
        let (_, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);
        assert!(mgr.poll().is_empty());
        assert_eq!(mgr.mode(), OperatingMode::Normal);
    }

    #[test]
    fn test_manager_memory_signal_is_graded_only() {
        // memory 압력은 graded(§5.4) → discrete 명령 없음(구 MemoryStrategy Evict 폐기). 단 mode 기여.
        let (tx, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);
        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Critical,
                available_bytes: 50 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 100 * 1024 * 1024,
            },
        );
        assert!(mgr.poll().is_empty());
        assert_eq!(mgr.mode(), OperatingMode::Minimal);
    }

    #[test]
    fn test_manager_handles_multiple_signals() {
        let (tx, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);
        // memory warning(graded, 명령 0) + thermal critical(discrete) 동시
        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Warning,
                available_bytes: 200 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 50 * 1024 * 1024,
            },
        );
        send_signal(
            &tx,
            SystemSignal::ThermalAlert {
                level: Level::Critical,
                temperature_mc: 80000,
                throttling_active: true,
                throttle_ratio: 0.5,
            },
        );
        let commands = mgr.poll();
        assert!(!commands.is_empty());
        assert_eq!(mgr.mode(), OperatingMode::Minimal);
    }

    #[test]
    fn test_manager_state_transitions() {
        let (tx, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);
        assert_eq!(mgr.mode(), OperatingMode::Normal);

        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Warning,
                available_bytes: 200 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 50 * 1024 * 1024,
            },
        );
        mgr.poll();
        assert_eq!(mgr.mode(), OperatingMode::Degraded);

        send_signal(
            &tx,
            SystemSignal::ThermalAlert {
                level: Level::Critical,
                temperature_mc: 80000,
                throttling_active: true,
                throttle_ratio: 0.6,
            },
        );
        mgr.poll();
        assert_eq!(mgr.mode(), OperatingMode::Minimal);

        send_signal(
            &tx,
            SystemSignal::EnergyConstraint {
                level: Level::Emergency,
                reason: EnergyReason::BatteryCritical,
                power_budget_mw: 0,
            },
        );
        mgr.poll();
        assert_eq!(mgr.mode(), OperatingMode::Suspended);

        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Normal,
                available_bytes: 1024 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 0,
            },
        );
        send_signal(
            &tx,
            SystemSignal::ThermalAlert {
                level: Level::Normal,
                temperature_mc: 40000,
                throttling_active: false,
                throttle_ratio: 1.0,
            },
        );
        send_signal(
            &tx,
            SystemSignal::EnergyConstraint {
                level: Level::Normal,
                reason: EnergyReason::Charging,
                power_budget_mw: 0,
            },
        );
        mgr.poll();
        assert_eq!(mgr.mode(), OperatingMode::Normal);
    }

    #[test]
    fn test_manager_survives_channel_disconnect() {
        let (tx, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);
        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Warning,
                available_bytes: 200 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 50 * 1024 * 1024,
            },
        );
        drop(tx);

        // memory → 명령 0, mode Degraded.
        assert!(mgr.poll().is_empty());
        assert_eq!(mgr.mode(), OperatingMode::Degraded);

        // channel disconnected: no panic, state preserved.
        assert!(mgr.poll().is_empty());
        assert_eq!(mgr.mode(), OperatingMode::Degraded);
    }
}
