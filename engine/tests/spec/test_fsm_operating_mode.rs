//! OperatingMode FSM full transition matrix verification.
//!
//! Validates:
//! - INV-070: from_levels() is a pure function (result determined solely by inputs)
//! - ENG-ST-011: worst-wins rule
//! - ENG-ST-012: Level-to-OperatingMode mapping
//! - ENG-ST-013: All 12 transitions (N^2 - N)

use argus_engine::resilience::OperatingMode;
use argus_shared::Level;

// ── INV-070: from_levels() is a pure function ───────────────

#[test]
fn test_inv_070_from_levels_is_pure_function() {
    // The same inputs must always produce the same output,
    // regardless of how many times we call it or in what order.
    let inputs = [
        (Level::Normal, Level::Normal, Level::Normal, Level::Normal),
        (Level::Warning, Level::Normal, Level::Normal, Level::Normal),
        (Level::Normal, Level::Critical, Level::Normal, Level::Normal),
        (
            Level::Normal,
            Level::Normal,
            Level::Normal,
            Level::Emergency,
        ),
        (
            Level::Warning,
            Level::Critical,
            Level::Warning,
            Level::Emergency,
        ),
    ];

    for &(m, c, t, e) in &inputs {
        let first = OperatingMode::from_levels(m, c, t, e);
        let second = OperatingMode::from_levels(m, c, t, e);
        let third = OperatingMode::from_levels(m, c, t, e);
        assert_eq!(
            first, second,
            "from_levels must be pure: same inputs must yield same output"
        );
        assert_eq!(
            second, third,
            "from_levels must be pure: same inputs must yield same output"
        );
    }
}

#[test]
fn test_inv_070_from_levels_no_hidden_state() {
    // Calling with different inputs in between should not affect the result
    // for a given set of inputs.
    let _ = OperatingMode::from_levels(
        Level::Emergency,
        Level::Emergency,
        Level::Emergency,
        Level::Emergency,
    );
    let result =
        OperatingMode::from_levels(Level::Normal, Level::Normal, Level::Normal, Level::Normal);
    assert_eq!(
        result,
        OperatingMode::Normal,
        "from_levels must not carry hidden state from previous calls"
    );
}

// ── ENG-ST-011/012: worst-wins mapping ──────────────────────

#[test]
fn test_eng_st_012_level_to_mode_mapping() {
    // Each Level maps to exactly one OperatingMode when it is the worst.
    assert_eq!(
        OperatingMode::from_levels(Level::Normal, Level::Normal, Level::Normal, Level::Normal),
        OperatingMode::Normal
    );
    assert_eq!(
        OperatingMode::from_levels(Level::Warning, Level::Normal, Level::Normal, Level::Normal),
        OperatingMode::Degraded
    );
    assert_eq!(
        OperatingMode::from_levels(Level::Normal, Level::Critical, Level::Normal, Level::Normal),
        OperatingMode::Minimal
    );
    assert_eq!(
        OperatingMode::from_levels(
            Level::Normal,
            Level::Normal,
            Level::Normal,
            Level::Emergency
        ),
        OperatingMode::Suspended
    );
}

#[test]
fn test_eng_st_011_worst_wins_any_signal_position() {
    // The worst level wins regardless of which signal domain it comes from.

    // Warning in each position
    for pos in 0..4 {
        let levels = [
            if pos == 0 {
                Level::Warning
            } else {
                Level::Normal
            },
            if pos == 1 {
                Level::Warning
            } else {
                Level::Normal
            },
            if pos == 2 {
                Level::Warning
            } else {
                Level::Normal
            },
            if pos == 3 {
                Level::Warning
            } else {
                Level::Normal
            },
        ];
        let mode = OperatingMode::from_levels(levels[0], levels[1], levels[2], levels[3]);
        assert_eq!(
            mode,
            OperatingMode::Degraded,
            "Warning in position {pos} should yield Degraded"
        );
    }

    // Critical in each position
    for pos in 0..4 {
        let levels = [
            if pos == 0 {
                Level::Critical
            } else {
                Level::Normal
            },
            if pos == 1 {
                Level::Critical
            } else {
                Level::Normal
            },
            if pos == 2 {
                Level::Critical
            } else {
                Level::Normal
            },
            if pos == 3 {
                Level::Critical
            } else {
                Level::Normal
            },
        ];
        let mode = OperatingMode::from_levels(levels[0], levels[1], levels[2], levels[3]);
        assert_eq!(
            mode,
            OperatingMode::Minimal,
            "Critical in position {pos} should yield Minimal"
        );
    }

    // Emergency in each position
    for pos in 0..4 {
        let levels = [
            if pos == 0 {
                Level::Emergency
            } else {
                Level::Normal
            },
            if pos == 1 {
                Level::Emergency
            } else {
                Level::Normal
            },
            if pos == 2 {
                Level::Emergency
            } else {
                Level::Normal
            },
            if pos == 3 {
                Level::Emergency
            } else {
                Level::Normal
            },
        ];
        let mode = OperatingMode::from_levels(levels[0], levels[1], levels[2], levels[3]);
        assert_eq!(
            mode,
            OperatingMode::Suspended,
            "Emergency in position {pos} should yield Suspended"
        );
    }
}

// ── ENG-ST-013: Complete transition table (12 transitions) ──

/// Helper: compute OperatingMode for the given worst level.
fn mode_for_worst(worst: Level) -> OperatingMode {
    OperatingMode::from_levels(worst, Level::Normal, Level::Normal, Level::Normal)
}

#[test]
fn test_fsm_operating_mode_all_transitions() {
    // Since from_levels() is stateless, we verify all 12 transitions
    // by checking that given appropriate input levels, the output mode
    // changes as specified in ENG-ST-013.

    // Normal -> Degraded: any Level >= Warning
    assert_eq!(mode_for_worst(Level::Warning), OperatingMode::Degraded);

    // Normal -> Minimal: any Level >= Critical
    assert_eq!(mode_for_worst(Level::Critical), OperatingMode::Minimal);

    // Normal -> Suspended: any Level = Emergency
    assert_eq!(mode_for_worst(Level::Emergency), OperatingMode::Suspended);

    // Degraded -> Normal: all Level = Normal
    // (from Degraded state, if all levels become Normal, result is Normal)
    assert_eq!(mode_for_worst(Level::Normal), OperatingMode::Normal);

    // Degraded -> Minimal: any Level >= Critical
    assert_eq!(
        OperatingMode::from_levels(
            Level::Warning,
            Level::Critical,
            Level::Normal,
            Level::Normal
        ),
        OperatingMode::Minimal
    );

    // Degraded -> Suspended: any Level = Emergency
    assert_eq!(
        OperatingMode::from_levels(
            Level::Warning,
            Level::Normal,
            Level::Normal,
            Level::Emergency
        ),
        OperatingMode::Suspended
    );

    // Minimal -> Normal: all Level = Normal
    assert_eq!(
        OperatingMode::from_levels(Level::Normal, Level::Normal, Level::Normal, Level::Normal),
        OperatingMode::Normal
    );

    // Minimal -> Degraded: worst Level = Warning
    assert_eq!(
        OperatingMode::from_levels(Level::Warning, Level::Normal, Level::Normal, Level::Normal),
        OperatingMode::Degraded
    );

    // Minimal -> Suspended: any Level = Emergency
    assert_eq!(
        OperatingMode::from_levels(
            Level::Critical,
            Level::Normal,
            Level::Emergency,
            Level::Normal
        ),
        OperatingMode::Suspended
    );

    // Suspended -> Normal: all Level = Normal
    assert_eq!(
        OperatingMode::from_levels(Level::Normal, Level::Normal, Level::Normal, Level::Normal),
        OperatingMode::Normal
    );

    // Suspended -> Degraded: worst Level = Warning
    assert_eq!(
        OperatingMode::from_levels(Level::Normal, Level::Warning, Level::Normal, Level::Normal),
        OperatingMode::Degraded
    );

    // Suspended -> Minimal: worst Level = Critical
    assert_eq!(
        OperatingMode::from_levels(Level::Normal, Level::Normal, Level::Critical, Level::Normal),
        OperatingMode::Minimal
    );
}

#[test]
fn test_fsm_operating_mode_mixed_levels() {
    // When multiple non-Normal levels are present, the worst wins.

    // Warning + Critical -> Minimal
    assert_eq!(
        OperatingMode::from_levels(
            Level::Warning,
            Level::Critical,
            Level::Normal,
            Level::Warning
        ),
        OperatingMode::Minimal
    );

    // Warning + Emergency -> Suspended
    assert_eq!(
        OperatingMode::from_levels(
            Level::Warning,
            Level::Normal,
            Level::Normal,
            Level::Emergency
        ),
        OperatingMode::Suspended
    );

    // Critical + Emergency -> Suspended
    assert_eq!(
        OperatingMode::from_levels(
            Level::Normal,
            Level::Critical,
            Level::Emergency,
            Level::Normal
        ),
        OperatingMode::Suspended
    );

    // All at Warning -> Degraded
    assert_eq!(
        OperatingMode::from_levels(
            Level::Warning,
            Level::Warning,
            Level::Warning,
            Level::Warning
        ),
        OperatingMode::Degraded
    );

    // All at Critical -> Minimal
    assert_eq!(
        OperatingMode::from_levels(
            Level::Critical,
            Level::Critical,
            Level::Critical,
            Level::Critical
        ),
        OperatingMode::Minimal
    );

    // All at Emergency -> Suspended
    assert_eq!(
        OperatingMode::from_levels(
            Level::Emergency,
            Level::Emergency,
            Level::Emergency,
            Level::Emergency
        ),
        OperatingMode::Suspended
    );
}

/// Exhaustively test all 256 (4^4) input combinations.
#[test]
fn test_fsm_operating_mode_exhaustive_256_combinations() {
    let all_levels = [
        Level::Normal,
        Level::Warning,
        Level::Critical,
        Level::Emergency,
    ];

    for &m in &all_levels {
        for &c in &all_levels {
            for &t in &all_levels {
                for &e in &all_levels {
                    let mode = OperatingMode::from_levels(m, c, t, e);
                    let worst = m.max(c).max(t).max(e);
                    let expected = match worst {
                        Level::Normal => OperatingMode::Normal,
                        Level::Warning => OperatingMode::Degraded,
                        Level::Critical => OperatingMode::Minimal,
                        Level::Emergency => OperatingMode::Suspended,
                    };
                    assert_eq!(
                        mode, expected,
                        "from_levels({m:?}, {c:?}, {t:?}, {e:?}) = {mode:?}, expected {expected:?}"
                    );
                }
            }
        }
    }
}
