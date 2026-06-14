use super::signal::Level;

/// LLM operating mode determined by the worst signal level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingMode {
    /// Normal operation. All features available.
    Normal,
    /// Degraded mode. Some resource constraints applied.
    Degraded,
    /// Minimal mode. Minimal resources, aggressive constraints.
    Minimal,
    /// Suspended mode. Inference paused.
    Suspended,
}

impl OperatingMode {
    /// Determine operating mode from the 4 signal levels.
    /// The most severe signal determines the mode.
    pub fn from_levels(memory: Level, compute: Level, thermal: Level, energy: Level) -> Self {
        let worst = memory.max(compute).max(thermal).max(energy);
        match worst {
            Level::Normal => OperatingMode::Normal,
            Level::Warning => OperatingMode::Degraded,
            Level::Critical => OperatingMode::Minimal,
            Level::Emergency => OperatingMode::Suspended,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_normal_yields_normal_mode() {
        let mode =
            OperatingMode::from_levels(Level::Normal, Level::Normal, Level::Normal, Level::Normal);
        assert_eq!(mode, OperatingMode::Normal);
    }

    #[test]
    fn test_single_warning_yields_degraded() {
        let mode =
            OperatingMode::from_levels(Level::Normal, Level::Warning, Level::Normal, Level::Normal);
        assert_eq!(mode, OperatingMode::Degraded);
    }

    #[test]
    fn test_single_critical_yields_minimal() {
        let mode = OperatingMode::from_levels(
            Level::Normal,
            Level::Normal,
            Level::Critical,
            Level::Normal,
        );
        assert_eq!(mode, OperatingMode::Minimal);
    }

    #[test]
    fn test_any_emergency_yields_suspended() {
        let mode = OperatingMode::from_levels(
            Level::Normal,
            Level::Normal,
            Level::Normal,
            Level::Emergency,
        );
        assert_eq!(mode, OperatingMode::Suspended);
    }

    #[test]
    fn test_mixed_levels_worst_wins() {
        // Warning + Critical → Minimal (Critical wins)
        let mode = OperatingMode::from_levels(
            Level::Warning,
            Level::Critical,
            Level::Normal,
            Level::Warning,
        );
        assert_eq!(mode, OperatingMode::Minimal);

        // Warning + Emergency → Suspended (Emergency wins)
        let mode = OperatingMode::from_levels(
            Level::Warning,
            Level::Normal,
            Level::Normal,
            Level::Emergency,
        );
        assert_eq!(mode, OperatingMode::Suspended);
    }
}
