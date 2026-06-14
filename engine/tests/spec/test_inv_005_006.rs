//! INV-005: Manager 장애가 Engine 추론 루프를 중단시키지 않음.
//! INV-006: Engine 장애가 Manager 모니터링 루프를 중단시키지 않음.
//!
//! 원본: 00-overview SYS-050 (INV-005), 00-overview SYS-051 (INV-006)
//! 검증 전략:
//!   - INV-005: MockTransport 연결 끊김 후 CommandExecutor.poll()이
//!     panic하지 않고 정상 ExecutionPlan을 반환하는지 확인.
//!     (Manager가 죽어도 Engine 추론 루프는 계속 동작해야 한다)
//!   - INV-006: Manager 측 코드는 manager 크레이트에 있으므로 여기서는
//!     Engine 측에서 응답 전송 실패 시 panic하지 않는지 확인.
//!     (Engine이 죽어도 = resp_rx 끊김, Manager가 recv 시 graceful 에러)
//!
//! 한계: 실제 2-프로세스 격리는 통합 테스트로 검증해야 함.
//!       여기서는 채널 기반 장애 시뮬레이션으로 단위 수준 검증.

use std::sync::mpsc;
use std::time::Duration;

use argus_engine::resilience::{CommandExecutor, MessageLoop, MockTransport};
use argus_shared::{EngineCommand, EngineDirective, EngineMessage, ManagerMessage};

use super::helpers::empty_snap;

// ═══════════════════════════════════════════════════════════════
// INV-005: Manager 장애가 Engine 추론 루프를 중단시키지 않음
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_005_manager_disconnect_does_not_crash_executor() {
    // Manager 측 채널(cmd_tx)을 drop하여 Manager 장애를 시뮬레이션
    let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, _resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));
    executor.set_running();

    // Manager sends a directive then disconnects
    cmd_tx
        .send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Throttle { delay_ms: 10 }],
        }))
        .unwrap();
    drop(cmd_tx); // Manager crashes / disconnects

    // First poll should process the pending directive
    let plan = executor.poll(&empty_snap());
    assert_eq!(plan.throttle_delay_ms, 10);

    // Subsequent polls after Manager disconnect should NOT panic
    // They should return a default-like plan with existing throttle state
    let plan2 = executor.poll(&empty_snap());
    assert_eq!(
        plan2.throttle_delay_ms, 10,
        "INV-005: executor must maintain state after Manager disconnect"
    );
    assert!(
        !plan2.suspended,
        "INV-005: executor must not suspend on Manager disconnect"
    );

    // Multiple polls should all succeed
    for _ in 0..10 {
        let plan = executor.poll(&empty_snap());
        assert!(!plan.suspended);
    }
}

#[test]
fn test_inv_005_message_loop_stops_gracefully_on_transport_disconnect() {
    // MockTransport with pre-loaded messages (sender dropped = disconnect)
    let msgs = vec![ManagerMessage::Directive(EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::Throttle { delay_ms: 20 }],
    })];
    let transport = MockTransport::from_messages(msgs);

    let (cmd_rx, _resp_tx, handle) = MessageLoop::spawn(transport).unwrap();

    // Receive the one message
    let msg = cmd_rx.recv().unwrap();
    match msg {
        ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
    }

    // After sender is exhausted, MessageLoop thread should exit gracefully (not panic)
    let result = handle.join();
    assert!(
        result.is_ok(),
        "INV-005: MessageLoop thread must exit gracefully on transport disconnect, not panic"
    );
}

#[test]
fn test_inv_005_executor_poll_after_resp_channel_closed() {
    // Simulate the response receiver being dropped (Manager side gone)
    let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));

    // Drop the response receiver (Manager crashed)
    drop(resp_rx);

    // Send a directive
    cmd_tx
        .send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Throttle { delay_ms: 30 }],
        }))
        .unwrap();

    // poll should NOT panic even though response channel is broken
    // The executor uses `let _ = self.resp_tx.send(...)` which ignores send errors
    let plan = executor.poll(&empty_snap());
    assert_eq!(
        plan.throttle_delay_ms, 30,
        "INV-005: executor must still process commands even when response channel is broken"
    );
}

// ═══════════════════════════════════════════════════════════════
// INV-006: Engine 장애가 Manager 모니터링 루프를 중단시키지 않음
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_006_engine_channel_drop_graceful() {
    // Engine 측 채널이 drop되면 MessageLoop 스레드가 다음 메시지를
    // cmd_tx.send() 할 때 에러를 감지하고 종료해야 한다.
    // 이를 위해 Manager 측에서 한 메시지를 보내고, Engine 측 cmd_rx를 drop한 후,
    // 다시 보내면 MessageLoop가 cmd_tx.send 실패로 종료한다.
    let (transport, mgr) = MockTransport::bidirectional();

    let (cmd_rx, resp_tx, handle) = MessageLoop::spawn(transport).unwrap();

    // Engine receives first directive
    mgr.send(ManagerMessage::Directive(EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::Throttle { delay_ms: 10 }],
    }))
    .unwrap();

    let msg = cmd_rx.recv().unwrap();
    match msg {
        ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
    }

    // Engine crashes: drop the engine-side channels
    drop(cmd_rx);
    drop(resp_tx);

    // Send another message so MessageLoop attempts cmd_tx.send() on dropped receiver
    let _ = mgr.send(ManagerMessage::Directive(EngineDirective {
        seq_id: 2,
        commands: vec![EngineCommand::Suspend],
    }));

    // MessageLoop should detect Engine channel drop and exit gracefully
    let result = handle.join();
    assert!(
        result.is_ok(),
        "INV-006: MessageLoop must exit gracefully when Engine channels are dropped"
    );
}

#[test]
fn test_inv_006_manager_recv_after_engine_disconnect() {
    // Pre-loaded transport (sender dropped after construction = will disconnect)
    let msgs = vec![ManagerMessage::Directive(EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::Throttle { delay_ms: 0 }],
    })];
    let transport = MockTransport::from_messages(msgs);
    let (cmd_rx, _resp_tx, handle) = MessageLoop::spawn(transport).unwrap();

    // Engine receives the message
    let _ = cmd_rx.recv().unwrap();

    // Drop engine-side channels to simulate Engine crash
    drop(cmd_rx);
    drop(_resp_tx);

    // MessageLoop thread should exit gracefully (transport will return Disconnected)
    let result = handle.join();
    assert!(
        result.is_ok(),
        "INV-006: MessageLoop must exit gracefully after Engine disconnect"
    );
}
