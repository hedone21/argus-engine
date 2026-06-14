//! INV-072 ~ INV-076 individual verification tests.
//!
//! INV-072: resolve_conflicts() — Suspend present => return exactly [Suspend]
//! INV-073: resolve_conflicts() — RestoreDefaults only when no other constraints
//! INV-074: poll() suspended => evict, switch_device, prepare_device all None
//! INV-075: Resume => compute/memory_level Normal, throttle 0
//! INV-076: RestoreDefaults => active_actions cleared, throttle 0, levels Normal

use argus_engine::resilience::strategy::resolve_conflicts;
use argus_shared::{EngineCommand, EngineState, ResourceLevel};

use super::helpers::{empty_snap, make_executor, send_directive};

// ═══════════════════════════════════════════════════════════════
// INV-072: Suspend in resolve_conflicts() returns exactly [Suspend]
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_072_suspend_overrides_all() {
    // Suspend with multiple other actions should return only [Suspend].
    let commands = vec![
        EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
        EngineCommand::SwitchHw {
            device: "cpu".to_string(),
        },
        EngineCommand::Throttle { delay_ms: 100 },
        EngineCommand::RestoreDefaults,
        EngineCommand::Suspend,
    ];

    let result = resolve_conflicts(commands);
    assert_eq!(result.len(), 1, "Suspend must override all other actions");
    assert!(
        matches!(result[0], EngineCommand::Suspend),
        "Only Suspend should remain"
    );
}

#[test]
fn test_inv_072_suspend_alone() {
    let commands = vec![EngineCommand::Suspend];
    let result = resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], EngineCommand::Suspend));
}

#[test]
fn test_inv_072_suspend_with_evict() {
    let commands = vec![
        EngineCommand::KvEvictH2o { keep_ratio: 0.25 },
        EngineCommand::Suspend,
    ];
    let result = resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], EngineCommand::Suspend));
}

#[test]
fn test_inv_072_suspend_with_throttle() {
    let commands = vec![
        EngineCommand::Throttle { delay_ms: 50 },
        EngineCommand::Suspend,
    ];
    let result = resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], EngineCommand::Suspend));
}

// ═══════════════════════════════════════════════════════════════
// INV-073: RestoreDefaults returned only when no other constraints
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_073_restore_defaults_alone() {
    let commands = vec![EngineCommand::RestoreDefaults];
    let result = resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], EngineCommand::RestoreDefaults));
}

#[test]
fn test_inv_073_restore_defaults_multiple() {
    // Multiple RestoreDefaults (e.g., from all 3 strategies returning Normal)
    let commands = vec![
        EngineCommand::RestoreDefaults,
        EngineCommand::RestoreDefaults,
        EngineCommand::RestoreDefaults,
    ];
    let result = resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], EngineCommand::RestoreDefaults));
}

#[test]
fn test_inv_073_restore_suppressed_by_evict() {
    let commands = vec![
        EngineCommand::RestoreDefaults,
        EngineCommand::KvEvictH2o { keep_ratio: 0.85 },
    ];
    let result = resolve_conflicts(commands);
    // RestoreDefaults should be suppressed; KvEvictH2o passes through
    assert_eq!(result.len(), 1);
    assert!(
        matches!(result[0], EngineCommand::KvEvictH2o { .. }),
        "RestoreDefaults must be suppressed when KvEvictH2o is present"
    );
}

#[test]
fn test_inv_073_restore_suppressed_by_throttle() {
    let commands = vec![
        EngineCommand::RestoreDefaults,
        EngineCommand::Throttle { delay_ms: 30 },
    ];
    let result = resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(
        matches!(result[0], EngineCommand::Throttle { .. }),
        "RestoreDefaults must be suppressed when Throttle is present"
    );
}

#[test]
fn test_inv_073_restore_suppressed_by_switch_hw() {
    let commands = vec![
        EngineCommand::RestoreDefaults,
        EngineCommand::SwitchHw {
            device: "cpu".to_string(),
        },
    ];
    let result = resolve_conflicts(commands);
    assert_eq!(result.len(), 1);
    assert!(
        matches!(result[0], EngineCommand::SwitchHw { .. }),
        "RestoreDefaults must be suppressed when SwitchHw is present"
    );
}

// ═══════════════════════════════════════════════════════════════
// INV-074: suspended plan clears evict, switch_device, prepare_device
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_074_suspended_clears_evict_switch_prepare() {
    let (mut executor, tx, _rx) = make_executor();

    // Send evict + switch + prepare + suspend in one directive
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
            EngineCommand::SwitchHw {
                device: "opencl".to_string(),
            },
            EngineCommand::PrepareComputeUnit {
                device: "opencl".to_string(),
            },
            EngineCommand::Suspend,
        ],
    );

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended, "Plan must be suspended");
    assert!(
        plan.evict.is_none(),
        "INV-074: evict must be None when suspended"
    );
    assert!(
        plan.switch_device.is_none(),
        "INV-074: switch_device must be None when suspended"
    );
    assert!(
        plan.prepare_device.is_none(),
        "INV-074: prepare_device must be None when suspended"
    );
}

#[test]
fn test_inv_074_suspended_clears_throttle() {
    let (mut executor, tx, _rx) = make_executor();

    // Throttle + Suspend
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Throttle { delay_ms: 100 },
            EngineCommand::Suspend,
        ],
    );

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended);
    assert_eq!(
        plan.throttle_delay_ms, 0,
        "INV-074: throttle must be 0 when suspended"
    );
}

#[test]
fn test_inv_074_suspended_clears_resumed_flag() {
    let (mut executor, tx, _rx) = make_executor();

    // Resume + Suspend in same batch (Suspend wins by post-processing)
    send_directive(&tx, 1, vec![EngineCommand::Resume, EngineCommand::Suspend]);

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended);
    assert!(
        !plan.resumed,
        "INV-074: resumed must be false when suspended"
    );
}

#[test]
fn test_inv_074_suspend_across_multiple_directives() {
    let (mut executor, tx, _rx) = make_executor();

    // First directive: evict + switch
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::KvEvictSliding { keep_ratio: 0.7 },
            EngineCommand::SwitchHw {
                device: "gpu".to_string(),
            },
        ],
    );
    // Second directive: suspend
    send_directive(&tx, 2, vec![EngineCommand::Suspend]);

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended);
    assert!(
        plan.evict.is_none(),
        "INV-074: evict must be None when suspended (cross-directive)"
    );
    assert!(
        plan.switch_device.is_none(),
        "INV-074: switch_device must be None when suspended (cross-directive)"
    );
}

// ═══════════════════════════════════════════════════════════════
// INV-075: Resume resets compute/memory level to Normal, throttle to 0
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_075_resume_resets_levels_and_throttle() {
    let (mut executor, tx, _rx) = make_executor();

    // Set up throttle
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 100 }]);
    executor.poll(&empty_snap());
    assert_eq!(executor.throttle_delay_ms(), 100);

    // Suspend
    send_directive(&tx, 2, vec![EngineCommand::Suspend]);
    executor.poll(&empty_snap());
    assert_eq!(executor.state(), EngineState::Suspended);

    // Resume
    send_directive(&tx, 3, vec![EngineCommand::Resume]);
    let plan = executor.poll(&empty_snap());

    assert!(plan.resumed, "Plan must indicate resumed");
    assert_eq!(
        executor.state(),
        EngineState::Running,
        "INV-075: state must be Running after Resume"
    );
    assert_eq!(
        executor.compute_level(),
        ResourceLevel::Normal,
        "INV-075: compute_level must be Normal after Resume"
    );
    assert_eq!(
        executor.memory_level(),
        ResourceLevel::Normal,
        "INV-075: memory_level must be Normal after Resume"
    );
    assert_eq!(
        executor.throttle_delay_ms(),
        0,
        "INV-075: throttle must be 0 after Resume"
    );
    assert_eq!(
        plan.throttle_delay_ms, 0,
        "INV-075: plan throttle must be 0 after Resume"
    );
}

#[test]
fn test_inv_075_resume_from_idle() {
    // Resume from Idle (not strictly meaningful but should not panic)
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::Resume]);
    let plan = executor.poll(&empty_snap());

    assert!(plan.resumed);
    assert_eq!(executor.state(), EngineState::Running);
    assert_eq!(executor.compute_level(), ResourceLevel::Normal);
    assert_eq!(executor.memory_level(), ResourceLevel::Normal);
    assert_eq!(executor.throttle_delay_ms(), 0);
}

// ═══════════════════════════════════════════════════════════════
// INV-076: RestoreDefaults clears active_actions, throttle, levels
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_076_restore_defaults_full_reset() {
    let (mut executor, tx, _rx) = make_executor();

    // Activate multiple actions
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Throttle { delay_ms: 50 },
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
            EngineCommand::LayerSkip { skip_ratio: 0.25 },
            EngineCommand::KvQuantDynamic { target_bits: 4 },
        ],
    );
    executor.poll(&empty_snap());
    assert_eq!(
        executor.active_actions().len(),
        4,
        "All 4 actions must be active before RestoreDefaults"
    );
    assert_eq!(executor.throttle_delay_ms(), 50);

    // RestoreDefaults
    send_directive(&tx, 2, vec![EngineCommand::RestoreDefaults]);
    let plan = executor.poll(&empty_snap());

    assert!(plan.restore_defaults, "Plan must indicate restore_defaults");
    assert!(
        executor.active_actions().is_empty(),
        "INV-076: active_actions must be empty after RestoreDefaults"
    );
    assert_eq!(
        executor.throttle_delay_ms(),
        0,
        "INV-076: throttle must be 0 after RestoreDefaults"
    );
    assert_eq!(
        executor.compute_level(),
        ResourceLevel::Normal,
        "INV-076: compute_level must be Normal after RestoreDefaults"
    );
    assert_eq!(
        executor.memory_level(),
        ResourceLevel::Normal,
        "INV-076: memory_level must be Normal after RestoreDefaults"
    );
    assert_eq!(
        plan.throttle_delay_ms, 0,
        "INV-076: plan throttle_delay_ms must be 0"
    );
}

#[test]
fn test_inv_076_restore_defaults_idempotent() {
    let (mut executor, tx, _rx) = make_executor();

    // RestoreDefaults on clean state should be a no-op
    send_directive(&tx, 1, vec![EngineCommand::RestoreDefaults]);
    let plan = executor.poll(&empty_snap());

    assert!(plan.restore_defaults);
    assert!(executor.active_actions().is_empty());
    assert_eq!(executor.throttle_delay_ms(), 0);
    assert_eq!(executor.compute_level(), ResourceLevel::Normal);
    assert_eq!(executor.memory_level(), ResourceLevel::Normal);
}

#[test]
fn test_inv_076_restore_defaults_after_sliding_evict() {
    let (mut executor, tx, _rx) = make_executor();

    // Activate sliding eviction
    send_directive(
        &tx,
        1,
        vec![EngineCommand::KvEvictSliding { keep_ratio: 0.6 }],
    );
    executor.poll(&empty_snap());
    assert!(
        executor
            .active_actions()
            .contains(&"kv.evict_sliding".to_string())
    );

    // RestoreDefaults
    send_directive(&tx, 2, vec![EngineCommand::RestoreDefaults]);
    executor.poll(&empty_snap());
    assert!(
        executor.active_actions().is_empty(),
        "INV-076: all active actions must be cleared"
    );
}

// ═══════════════════════════════════════════════════════════════
// INV-071: EngineState transitions occur only inside CommandExecutor
// (structural verification — we verify that the API enforces this)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_071_engine_state_transitions_via_executor() {
    let (mut executor, tx, _rx) = make_executor();

    // Initial state is Idle
    assert_eq!(
        executor.state(),
        EngineState::Idle,
        "INV-071: initial state must be Idle"
    );

    // Idle -> Running via set_running()
    executor.set_running();
    assert_eq!(
        executor.state(),
        EngineState::Running,
        "INV-071: set_running() transitions Idle -> Running"
    );

    // Running -> Suspended via Suspend command
    send_directive(&tx, 1, vec![EngineCommand::Suspend]);
    executor.poll(&empty_snap());
    assert_eq!(
        executor.state(),
        EngineState::Suspended,
        "INV-071: Suspend command transitions Running -> Suspended"
    );

    // Suspended -> Running via Resume command
    send_directive(&tx, 2, vec![EngineCommand::Resume]);
    executor.poll(&empty_snap());
    assert_eq!(
        executor.state(),
        EngineState::Running,
        "INV-071: Resume command transitions Suspended -> Running"
    );
}

#[test]
fn test_inv_071_state_transition_full_lifecycle() {
    // Idle -> Running -> Suspended -> Running (full cycle)
    let (mut executor, tx, _rx) = make_executor();

    assert_eq!(executor.state(), EngineState::Idle);

    executor.set_running();
    assert_eq!(executor.state(), EngineState::Running);

    send_directive(&tx, 1, vec![EngineCommand::Suspend]);
    executor.poll(&empty_snap());
    assert_eq!(executor.state(), EngineState::Suspended);

    send_directive(&tx, 2, vec![EngineCommand::Resume]);
    executor.poll(&empty_snap());
    assert_eq!(executor.state(), EngineState::Running);

    // Second suspend-resume cycle
    send_directive(&tx, 3, vec![EngineCommand::Suspend]);
    executor.poll(&empty_snap());
    assert_eq!(executor.state(), EngineState::Suspended);

    send_directive(&tx, 4, vec![EngineCommand::Resume]);
    executor.poll(&empty_snap());
    assert_eq!(executor.state(), EngineState::Running);
}
