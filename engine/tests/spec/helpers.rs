use std::sync::mpsc;
use std::time::Duration;

use argus_engine::resilience::{CommandExecutor, KVSnapshot};
use argus_shared::{EngineCommand, EngineDirective, EngineMessage, ManagerMessage};

/// Create a CommandExecutor with connected channels.
/// Heartbeat interval is set to 60s to avoid noise in tests.
pub fn make_executor() -> (
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

/// Create an empty KVSnapshot with all defaults.
pub fn empty_snap() -> KVSnapshot {
    KVSnapshot::default()
}

/// Send a directive with the given seq_id and commands.
pub fn send_directive(
    tx: &mpsc::Sender<ManagerMessage>,
    seq_id: u64,
    commands: Vec<EngineCommand>,
) {
    tx.send(ManagerMessage::Directive(EngineDirective {
        seq_id,
        commands,
    }))
    .unwrap();
}
