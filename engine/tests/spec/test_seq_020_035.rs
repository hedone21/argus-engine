//! SEQ-013 / SEQ-034: heartbeat interval gate + graceful drain on disconnect.
//!
//! The legacy `CommandExecutor::poll()`/`ExecutionPlan` surface was removed; the
//! live path is `send_heartbeat_if_due()` (heartbeat emission) + `drain_commands()`
//! (command draining). These two tests pin the live behaviors that have no other
//! coverage: the negative interval gate (not-due → no heartbeat) and a graceful
//! empty drain after the manager channel is dropped.

use std::sync::mpsc;
use std::time::Duration;

use argus_engine::resilience::{CommandExecutor, KVSnapshot};
use argus_shared::{EngineMessage, ManagerMessage};

// ═══════════════════════════════════════════════════════════════
// SEQ-034: send_heartbeat_if_due interval gate
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_034_heartbeat_timing() {
    let (_cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(100),
    );

    // interval 미도달 -> heartbeat 없어야 함
    executor.send_heartbeat_if_due(&KVSnapshot::default());
    assert!(
        resp_rx.try_recv().is_err(),
        "interval 미도달 시 heartbeat가 발생하면 안 됨"
    );

    // 100ms 이상 대기 후 -> heartbeat 발생해야 함
    std::thread::sleep(Duration::from_millis(110));
    executor.send_heartbeat_if_due(&KVSnapshot::default());
    let msg = resp_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(
        matches!(msg, EngineMessage::Heartbeat(_)),
        "interval 도달 후 heartbeat가 발생해야 함"
    );
}

// ═══════════════════════════════════════════════════════════════
// SEQ-013: EOF disconnect → graceful empty drain
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_013_eof_disconnect_graceful() {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, _resp_rx) = mpsc::channel();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_secs(3600),
    );

    // sender drop으로 EOF 시뮬레이션
    drop(cmd_tx);

    // drain_commands() 호출 시 패닉 없음, 빈 Vec 반환 (반복 호출도 안전).
    assert!(executor.drain_commands().is_empty());
    assert!(executor.drain_commands().is_empty());
}
