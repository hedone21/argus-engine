//! PROTO-042 / PROTO-073 행위 명세 테스트
//!
//! - PROTO-042: Connection 3-state FSM (Engine 측 Transport)
//!   Engine Transport도 Connected/Disconnected 상태를 갖는다.
//!   MockTransport로 연결/해제 시나리오를 검증한다.
//!
//! - PROTO-073: try_recv 드레인 — CommandExecutor.poll()이
//!   while let Ok(msg) = cmd_rx.try_recv() 로 모든 대기 directive를 배치 처리

use argus_engine::resilience::{CommandExecutor, KVSnapshot};
use argus_shared::{CommandResult, EngineCommand, EngineDirective, EngineMessage, ManagerMessage};
use std::sync::mpsc;
use std::time::Duration;

/// PROTO-042: MockTransport 기반 연결 상태 전이 시뮬레이션
///
/// CommandExecutor는 cmd_rx 채널을 통해 메시지를 수신한다.
/// 채널이 열려있으면 Connected 상태에 해당하고,
/// Sender가 drop되면 Disconnected에 해당한다.
#[test]
fn test_proto_042_connection_state_transitions() {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));

    let snap = KVSnapshot::default();

    // State 1: Connected — 메시지 수신 가능
    cmd_tx
        .send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Throttle { delay_ms: 10 }],
        }))
        .unwrap();
    let plan = executor.poll(&snap);
    assert_eq!(plan.throttle_delay_ms, 10);

    // 응답 확인
    let resp = resp_rx.recv().unwrap();
    assert!(matches!(resp, EngineMessage::Response(_)));

    // State 2: Sender drop → 채널 Disconnected
    drop(cmd_tx);

    // poll은 빈 계획을 반환 (에러 없이 graceful)
    let plan = executor.poll(&snap);
    assert!(!plan.suspended);
    assert_eq!(plan.throttle_delay_ms, 10); // 기존 스로틀 유지
}

/// PROTO-073: poll()이 여러 대기 directive를 한 번에 드레인
#[test]
fn test_proto_073_drain_multiple_directives() {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));

    let snap = KVSnapshot::default();

    // 3개 directive를 채널에 미리 넣어둔다
    for i in 1..=3 {
        cmd_tx
            .send(ManagerMessage::Directive(EngineDirective {
                seq_id: i,
                commands: vec![EngineCommand::Throttle { delay_ms: i * 10 }],
            }))
            .unwrap();
    }

    // 단일 poll() 호출로 3개 모두 처리
    let plan = executor.poll(&snap);
    // 마지막 directive의 throttle이 적용됨 (seq_id=3, delay=30)
    assert_eq!(plan.throttle_delay_ms, 30);

    // 3개 Response가 모두 전송되어야 함
    let mut responses = Vec::new();
    while let Ok(msg) = resp_rx.try_recv() {
        if let EngineMessage::Response(r) = msg {
            responses.push(r);
        }
    }
    assert_eq!(
        responses.len(),
        3,
        "All 3 directives should produce responses"
    );

    // seq_id 순서 확인
    let seq_ids: Vec<u64> = responses.iter().map(|r| r.seq_id).collect();
    assert_eq!(seq_ids, vec![1, 2, 3]);

    // 각 응답의 results 길이 = 1 (명령 1개씩)
    for r in &responses {
        assert_eq!(r.results.len(), 1);
        assert!(matches!(r.results[0], CommandResult::Ok));
    }
}

/// PROTO-073: 대기 메시지 없을 때 poll() = 빈 계획
#[test]
fn test_proto_073_drain_empty_channel() {
    let (_cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, _resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));

    let snap = KVSnapshot::default();

    // 빈 채널에서 poll
    let plan = executor.poll(&snap);
    assert!(plan.evict.is_none());
    assert!(plan.switch_device.is_none());
    assert!(!plan.suspended);
    assert!(!plan.resumed);
    assert_eq!(plan.throttle_delay_ms, 0);
}
