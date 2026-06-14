//! ENG-ST-010 ~ ENG-ST-035: OperatingMode + EngineState + Executor
//!
//! OperatingMode FSM, CommandExecutor 상태 전이, EngineCommand 13종 처리,
//! Suspend 오버라이드, 후행 Directive 덮어쓰기 검증.

use std::sync::mpsc;
use std::time::Duration;

use argus_engine::resilience::{CommandExecutor, EvictMethod, KVSnapshot, OperatingMode};
use argus_shared::{
    CommandResult, EngineCommand, EngineDirective, EngineMessage, EngineState, ManagerMessage,
    ResourceLevel,
};

// ── 헬퍼 ──

fn make_executor() -> (
    CommandExecutor,
    mpsc::Sender<ManagerMessage>,
    mpsc::Receiver<EngineMessage>,
) {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));
    (executor, cmd_tx, resp_rx)
}

fn empty_snap() -> KVSnapshot {
    KVSnapshot::default()
}

fn send_directive(tx: &mpsc::Sender<ManagerMessage>, seq_id: u64, commands: Vec<EngineCommand>) {
    tx.send(ManagerMessage::Directive(EngineDirective {
        seq_id,
        commands,
    }))
    .unwrap();
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-010: OperatingMode::from_levels — 4-signal worst-level 매핑
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_010_all_normal_yields_normal() {
    use argus_shared::Level;
    let mode =
        OperatingMode::from_levels(Level::Normal, Level::Normal, Level::Normal, Level::Normal);
    assert_eq!(mode, OperatingMode::Normal);
}

#[test]
fn test_eng_st_010_single_warning_yields_degraded() {
    use argus_shared::Level;
    let mode =
        OperatingMode::from_levels(Level::Normal, Level::Warning, Level::Normal, Level::Normal);
    assert_eq!(mode, OperatingMode::Degraded);
}

#[test]
fn test_eng_st_010_single_critical_yields_minimal() {
    use argus_shared::Level;
    let mode =
        OperatingMode::from_levels(Level::Normal, Level::Normal, Level::Critical, Level::Normal);
    assert_eq!(mode, OperatingMode::Minimal);
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-011: Emergency → Suspended 매핑
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_011_any_emergency_yields_suspended() {
    use argus_shared::Level;
    let mode = OperatingMode::from_levels(
        Level::Normal,
        Level::Normal,
        Level::Normal,
        Level::Emergency,
    );
    assert_eq!(mode, OperatingMode::Suspended);
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-012: 혼합 레벨 — worst-level 우선
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_012_mixed_levels_worst_wins() {
    use argus_shared::Level;
    // Warning + Critical → Minimal
    let mode = OperatingMode::from_levels(
        Level::Warning,
        Level::Critical,
        Level::Normal,
        Level::Warning,
    );
    assert_eq!(mode, OperatingMode::Minimal);

    // Warning + Emergency → Suspended
    let mode = OperatingMode::from_levels(
        Level::Warning,
        Level::Normal,
        Level::Normal,
        Level::Emergency,
    );
    assert_eq!(mode, OperatingMode::Suspended);
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-020: Executor — Suspend 상태 전이
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_020_executor_suspend_transitions_state() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::Suspend]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended);
    assert_eq!(executor.state(), EngineState::Suspended);
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-021: Executor — Resume 상태 전이
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_021_executor_resume_resets_state() {
    let (mut executor, tx, _rx) = make_executor();

    // Throttle → Suspend → Resume
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 100 }]);
    executor.poll(&empty_snap());

    send_directive(&tx, 2, vec![EngineCommand::Suspend]);
    executor.poll(&empty_snap());
    assert_eq!(executor.state(), EngineState::Suspended);

    send_directive(&tx, 3, vec![EngineCommand::Resume]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.resumed);
    assert_eq!(executor.state(), EngineState::Running);
    assert_eq!(executor.compute_level(), ResourceLevel::Normal);
    assert_eq!(executor.memory_level(), ResourceLevel::Normal);
    assert_eq!(executor.throttle_delay_ms(), 0);
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-031: active_actions 관리 — 각 커맨드가 올바르게 등록/제거
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_031_throttle_adds_active_action() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 50 }]);
    executor.poll(&empty_snap());
    assert!(executor.active_actions().contains(&"throttle".to_string()));
}

#[test]
fn test_eng_st_031_kv_evict_h2o_adds_active_action() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }]);
    executor.poll(&empty_snap());
    assert!(
        executor
            .active_actions()
            .contains(&"kv.evict_h2o".to_string())
    );
}

#[test]
fn test_eng_st_031_kv_evict_sliding_adds_active_action() {
    let (mut executor, tx, _rx) = make_executor();
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
}

#[test]
fn test_eng_st_031_layer_skip_adds_active_action() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::LayerSkip { skip_ratio: 0.25 }]);
    executor.poll(&empty_snap());
    assert!(
        executor
            .active_actions()
            .contains(&"weight.skip".to_string())
    );
}

#[test]
fn test_eng_st_031_restore_defaults_clears_all_actions() {
    let (mut executor, tx, _rx) = make_executor();

    // 여러 액션 활성화
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Throttle { delay_ms: 50 },
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
            EngineCommand::LayerSkip { skip_ratio: 0.2 },
        ],
    );
    executor.poll(&empty_snap());
    assert_eq!(executor.active_actions().len(), 3);

    // RestoreDefaults
    send_directive(&tx, 2, vec![EngineCommand::RestoreDefaults]);
    executor.poll(&empty_snap());
    assert!(executor.active_actions().is_empty());
    assert_eq!(executor.throttle_delay_ms(), 0);
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-033: EngineCommand 13종 처리 — 각 커맨드별 ExecutionPlan 검증
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_033_throttle_sets_delay() {
    let (mut executor, tx, rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 42 }]);
    let plan = executor.poll(&empty_snap());
    assert_eq!(plan.throttle_delay_ms, 42);

    // 응답이 Ok여야 함
    let resp = rx.recv().unwrap();
    match resp {
        EngineMessage::Response(r) => {
            assert_eq!(r.seq_id, 1);
            assert!(matches!(r.results[0], CommandResult::Ok));
        }
        _ => panic!("Expected Response"),
    }
}

#[test]
fn test_eng_st_033_kv_evict_h2o_creates_plan() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.48 }]);
    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("evict plan 있어야 함");
    assert!((evict.target_ratio - 0.48).abs() < f32::EPSILON);
    assert_eq!(evict.method, EvictMethod::H2o);
    assert_eq!(evict.level, ResourceLevel::Critical);
}

#[test]
fn test_eng_st_033_kv_evict_sliding_creates_plan() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(
        &tx,
        1,
        vec![EngineCommand::KvEvictSliding { keep_ratio: 0.6 }],
    );
    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("evict plan 있어야 함");
    assert!((evict.target_ratio - 0.6).abs() < f32::EPSILON);
    assert_eq!(evict.method, EvictMethod::Sliding);
}

#[test]
fn test_eng_st_033_kv_streaming_ok() {
    let (mut executor, tx, rx) = make_executor();
    send_directive(
        &tx,
        1,
        vec![EngineCommand::KvStreaming {
            sink_size: 4,
            window_size: 256,
        }],
    );
    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("KvStreaming은 evict plan을 생성해야 함");
    assert_eq!(evict.method, EvictMethod::Streaming);
    assert!((evict.target_ratio - 0.0).abs() < f32::EPSILON);
    let params = evict.streaming_params.expect("streaming_params 있어야 함");
    assert_eq!(params.sink_size, 4);
    assert_eq!(params.window_size, 256);
    assert!(
        executor
            .active_actions()
            .contains(&"kv.evict_streaming".to_string())
    );

    let resp = rx.recv().unwrap();
    match resp {
        EngineMessage::Response(r) => {
            assert!(matches!(r.results[0], CommandResult::Ok));
        }
        _ => panic!("Expected Response"),
    }
}

#[test]
fn test_eng_st_033_kv_quant_dynamic() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(
        &tx,
        1,
        vec![EngineCommand::KvQuantDynamic { target_bits: 4 }],
    );
    let plan = executor.poll(&empty_snap());
    assert_eq!(plan.kv_quant_bits, Some(4));
    assert!(
        executor
            .active_actions()
            .contains(&"kv.quant_dynamic".to_string())
    );
}

#[test]
fn test_eng_st_033_layer_skip() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::LayerSkip { skip_ratio: 0.3 }]);
    let plan = executor.poll(&empty_snap());
    assert_eq!(plan.layer_skip, Some(0.3));
}

#[test]
fn test_eng_st_033_restore_defaults() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 50 }]);
    executor.poll(&empty_snap());

    send_directive(&tx, 2, vec![EngineCommand::RestoreDefaults]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.restore_defaults);
    assert_eq!(plan.throttle_delay_ms, 0);
}

#[test]
fn test_eng_st_033_switch_hw() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(
        &tx,
        1,
        vec![EngineCommand::SwitchHw {
            device: "opencl".to_string(),
        }],
    );
    let plan = executor.poll(&empty_snap());
    assert_eq!(plan.switch_device.as_deref(), Some("opencl"));
}

#[test]
fn test_eng_st_033_prepare_compute_unit() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(
        &tx,
        1,
        vec![EngineCommand::PrepareComputeUnit {
            device: "gpu".to_string(),
        }],
    );
    let plan = executor.poll(&empty_snap());
    assert_eq!(plan.prepare_device.as_deref(), Some("gpu"));
}

#[test]
fn test_eng_st_033_suspend() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::Suspend]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended);
    assert_eq!(executor.state(), EngineState::Suspended);
}

#[test]
fn test_eng_st_033_resume() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::Suspend]);
    executor.poll(&empty_snap());

    send_directive(&tx, 2, vec![EngineCommand::Resume]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.resumed);
    assert_eq!(executor.state(), EngineState::Running);
}

#[test]
fn test_eng_st_033_switch_and_prepare_combined() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::PrepareComputeUnit {
                device: "gpu".to_string(),
            },
            EngineCommand::SwitchHw {
                device: "gpu".to_string(),
            },
        ],
    );
    let plan = executor.poll(&empty_snap());
    assert_eq!(plan.prepare_device.as_deref(), Some("gpu"));
    assert_eq!(plan.switch_device.as_deref(), Some("gpu"));
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-034: Suspend → plan.evict = None (Suspend가 evict를 무효화)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_034_suspend_nullifies_evict_plan() {
    let (mut executor, tx, _rx) = make_executor();
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
            EngineCommand::SwitchHw {
                device: "cpu".to_string(),
            },
            EngineCommand::Suspend,
        ],
    );
    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended);
    assert!(plan.evict.is_none(), "Suspend가 evict plan을 무효화해야 함");
    assert!(
        plan.switch_device.is_none(),
        "Suspend가 switch를 무효화해야 함"
    );
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-035: 후행 Directive가 선행을 덮어씀
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_035_superseding_evict_directive() {
    let (mut executor, tx, rx) = make_executor();

    // 두 Directive — 두 번째 evict가 첫 번째를 덮어씀
    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.8 }]);
    send_directive(
        &tx,
        2,
        vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }],
    );

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.unwrap();
    assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
    assert_eq!(evict.method, EvictMethod::Sliding);

    // 두 응답 모두 전송
    let r1 = rx.recv().unwrap();
    let r2 = rx.recv().unwrap();
    match (r1, r2) {
        (EngineMessage::Response(r1), EngineMessage::Response(r2)) => {
            assert_eq!(r1.seq_id, 1);
            assert_eq!(r2.seq_id, 2);
        }
        _ => panic!("Expected two Responses"),
    }
}
