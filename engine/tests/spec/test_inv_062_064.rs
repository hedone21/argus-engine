//! INV-062: Suspend 포함 ExecutionPlan에서 evict/switch_device/prepare_device = None.
//! INV-064: heartbeat_interval 내 최소 1회 Heartbeat 전송 (poll 호출 시).
//!
//! 원본: 30-engine ENG-032 (INV-062), 30-engine ENG-033 (INV-064)
//!
//! INV-062 주의: INV-074 (31-engine-state)와 동일 내용이며, test_inv_072_076.rs에서
//!              이미 검증됨. 여기서는 추가 엣지 케이스를 보충.
//!
//! INV-064 검증: 짧은 heartbeat_interval로 CommandExecutor를 생성하고,
//!              interval이 지난 후 poll()을 호출하면 Heartbeat 메시지가
//!              생성되는지 확인.

use std::sync::mpsc;
use std::time::Duration;

use argus_engine::resilience::{CommandExecutor, KVSnapshot};
use argus_shared::{EngineCommand, EngineDirective, EngineMessage, EngineState, ManagerMessage};

use super::helpers::{empty_snap, make_executor, send_directive};

// ═══════════════════════════════════════════════════════════════
// INV-062: Suspend plan clears evict, switch_device, prepare_device
// (Supplementary to INV-074 tests in test_inv_072_076.rs)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_062_suspend_only_no_other_fields() {
    let (mut executor, tx, _rx) = make_executor();
    executor.set_running();

    // Only Suspend, no other commands
    send_directive(&tx, 1, vec![EngineCommand::Suspend]);

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended, "Plan must be suspended");
    assert!(
        plan.evict.is_none(),
        "INV-062: evict must be None when suspended"
    );
    assert!(
        plan.switch_device.is_none(),
        "INV-062: switch_device must be None when suspended"
    );
    assert!(
        plan.prepare_device.is_none(),
        "INV-062: prepare_device must be None when suspended"
    );
    assert_eq!(
        plan.throttle_delay_ms, 0,
        "INV-062: throttle must be 0 when suspended"
    );
    assert!(
        !plan.resumed,
        "INV-062: resumed must be false when suspended"
    );
}

#[test]
fn test_inv_062_suspend_with_all_action_types() {
    let (mut executor, tx, _rx) = make_executor();
    executor.set_running();

    // All possible commands together with Suspend
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
            EngineCommand::KvEvictSliding { keep_ratio: 0.3 },
            EngineCommand::SwitchHw {
                device: "gpu".to_string(),
            },
            EngineCommand::PrepareComputeUnit {
                device: "gpu".to_string(),
            },
            EngineCommand::Throttle { delay_ms: 100 },
            EngineCommand::LayerSkip { skip_ratio: 0.5 },
            EngineCommand::KvQuantDynamic { target_bits: 4 },
            EngineCommand::Suspend,
        ],
    );

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended);
    assert!(
        plan.evict.is_none(),
        "INV-062: evict must be None even with multiple eviction commands"
    );
    assert!(
        plan.switch_device.is_none(),
        "INV-062: switch_device must be None"
    );
    assert!(
        plan.prepare_device.is_none(),
        "INV-062: prepare_device must be None"
    );
    assert_eq!(plan.throttle_delay_ms, 0, "INV-062: throttle must be 0");
}

#[test]
fn test_inv_062_suspend_at_beginning_of_commands() {
    let (mut executor, tx, _rx) = make_executor();
    executor.set_running();

    // Suspend appears before other commands
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Suspend,
            EngineCommand::KvEvictH2o { keep_ratio: 0.7 },
            EngineCommand::SwitchHw {
                device: "opencl".to_string(),
            },
        ],
    );

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended);
    assert!(
        plan.evict.is_none(),
        "INV-062: Suspend at start must still clear evict"
    );
    assert!(
        plan.switch_device.is_none(),
        "INV-062: Suspend at start must still clear switch_device"
    );
}

// ═══════════════════════════════════════════════════════════════
// INV-064: heartbeat_interval 내 최소 1회 Heartbeat 전송
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_064_heartbeat_sent_when_interval_elapsed() {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(10), // Very short interval to trigger heartbeat
    );
    executor.set_running();

    // Wait for interval to elapse
    std::thread::sleep(Duration::from_millis(20));

    let snap = KVSnapshot {
        total_bytes: 1024,
        total_tokens: 10,
        capacity: 2048,
        protected_prefix: 4,
        kv_dtype: "f16".to_string(),
        eviction_policy: "none".to_string(),
        skip_ratio: 0.0,
    };

    let _plan = executor.poll(&snap);

    // Check that a Heartbeat was sent
    let msg = resp_rx
        .recv_timeout(Duration::from_millis(100))
        .expect("INV-064: Heartbeat must be sent after interval elapses");

    match msg {
        EngineMessage::Heartbeat(status) => {
            assert_eq!(status.active_device, "cpu");
            assert_eq!(status.state, EngineState::Running);
            assert_eq!(status.kv_cache_tokens, 10);
        }
        other => panic!(
            "INV-064: Expected Heartbeat message, got {:?}",
            std::mem::discriminant(&other)
        ),
    }

    drop(cmd_tx);
}

#[test]
fn test_inv_064_no_heartbeat_before_interval() {
    let (_cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_secs(3600), // Very long interval — heartbeat should NOT trigger
    );
    executor.set_running();

    // Poll immediately (no time elapsed)
    let _plan = executor.poll(&empty_snap());

    // No heartbeat should be sent
    let result = resp_rx.try_recv();
    assert!(
        result.is_err(),
        "INV-064: No heartbeat should be sent before interval elapses"
    );
}

#[test]
fn test_inv_064_heartbeat_sent_on_each_interval() {
    let (_cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(10),
    );
    executor.set_running();

    // Wait for interval to elapse
    std::thread::sleep(Duration::from_millis(20));
    executor.poll(&empty_snap());

    // First heartbeat
    let msg1 = resp_rx.recv_timeout(Duration::from_millis(100));
    assert!(
        msg1.is_ok(),
        "INV-064: first heartbeat must be sent after interval"
    );
    assert!(matches!(msg1.unwrap(), EngineMessage::Heartbeat(_)));

    // Wait for another interval
    std::thread::sleep(Duration::from_millis(20));
    executor.poll(&empty_snap());

    // Second heartbeat
    let msg2 = resp_rx.recv_timeout(Duration::from_millis(100));
    assert!(
        msg2.is_ok(),
        "INV-064: second heartbeat must be sent after another interval"
    );
    assert!(matches!(msg2.unwrap(), EngineMessage::Heartbeat(_)));
}

#[test]
fn test_inv_064_heartbeat_reflects_current_state() {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(10),
    );
    executor.set_running();

    // Activate a throttle
    cmd_tx
        .send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Throttle { delay_ms: 50 }],
        }))
        .unwrap();
    executor.poll(&empty_snap());

    // Drain the response from the directive
    let _ = resp_rx.try_recv();

    // Wait for heartbeat
    std::thread::sleep(Duration::from_millis(20));
    let snap = KVSnapshot {
        total_bytes: 2048,
        total_tokens: 100,
        capacity: 1024,
        protected_prefix: 4,
        kv_dtype: "q4".to_string(),
        eviction_policy: "sliding".to_string(),
        skip_ratio: 0.25,
    };
    executor.poll(&snap);

    // Find the heartbeat message
    let mut found_heartbeat = false;
    for _ in 0..5 {
        if let Ok(EngineMessage::Heartbeat(status)) = resp_rx.try_recv() {
            assert!(
                status.active_actions.contains(&"throttle".to_string()),
                "INV-064: heartbeat must reflect active actions"
            );
            assert_eq!(status.kv_cache_tokens, 100);
            assert_eq!(status.eviction_policy, "sliding");
            assert_eq!(status.kv_dtype, "q4");
            assert!((status.skip_ratio - 0.25).abs() < 1e-6);
            found_heartbeat = true;
            break;
        }
    }
    assert!(
        found_heartbeat,
        "INV-064: heartbeat with current state must be sent"
    );
}
