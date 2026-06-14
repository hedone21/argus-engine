//! SEQ-070 ~ SEQ-093: 재연결, 에러 처리, 배압 시퀀스 (Engine-side)
//!
//! MessageLoop disconnect 처리, executor graceful degradation,
//! Unix socket ParseError/oversized/EOF 에러 내성,
//! 고속 Directive 배압 처리를 검증한다.

use std::time::Duration;

use argus_engine::resilience::{MessageLoop, MockTransport};
use argus_shared::{EngineCommand, EngineMessage, ManagerMessage};

use super::helpers::{empty_snap, make_executor, send_directive};

// ═══════════════════════════════════════════════════════════════
// SEQ-071: MessageLoop exits on disconnect
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_071_message_loop_exits_on_disconnect() {
    // 빈 메시지 목록 -> sender 즉시 drop -> Disconnected
    let transport = MockTransport::from_messages(vec![]);
    let (_cmd_rx, _resp_tx, handle) = MessageLoop::spawn(transport).unwrap();

    // 스레드가 정상 종료되어야 함 (Disconnected로 loop 탈출)
    let result = handle.join();
    assert!(
        result.is_ok(),
        "MessageLoop 스레드가 패닉 없이 정상 종료되어야 함"
    );
}

// ═══════════════════════════════════════════════════════════════
// SEQ-071: executor graceful after channel drop
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_071_executor_graceful_after_channel_drop() {
    let (mut executor, tx, _rx) = make_executor();

    // sender drop -> 채널 종료
    drop(tx);

    // poll() 호출 시 패닉 없음
    let plan = executor.poll(&empty_snap());
    assert!(plan.evict.is_none());
    assert!(!plan.suspended);

    // 반복 호출도 안전
    let plan2 = executor.poll(&empty_snap());
    assert!(!plan2.suspended);
}

// ═══════════════════════════════════════════════════════════════
// SEQ-080: ParseError 후 연결 유지 (Unix socket)
// ═══════════════════════════════════════════════════════════════

#[cfg(unix)]
#[test]
fn test_seq_080_parse_error_then_normal_recv() {
    use argus_engine::resilience::{Transport, UnixSocketTransport};
    use std::io::Write;
    use std::os::unix::net::UnixListener;

    let path = std::env::temp_dir().join(format!(
        "argus_seq080_{}.sock",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let listener = UnixListener::bind(&path).unwrap();

    let path2 = path.clone();
    let handle = std::thread::spawn(move || {
        let mut transport = UnixSocketTransport::new(path2);
        transport.connect().unwrap();

        // 첫 번째 프레임: ParseError
        let result1 = transport.recv();
        let is_parse_error = matches!(
            result1,
            Err(argus_engine::resilience::TransportError::ParseError(_))
        );

        // 두 번째 프레임: 정상 수신
        let result2 = transport.recv();
        let is_ok = result2.is_ok();

        (is_parse_error, is_ok)
    });

    let (mut server_stream, _) = listener.accept().unwrap();

    // 잘못된 JSON 전송
    let bad_json = b"not valid json at all!";
    let len = (bad_json.len() as u32).to_be_bytes();
    server_stream.write_all(&len).unwrap();
    server_stream.write_all(bad_json).unwrap();
    server_stream.flush().unwrap();

    // 정상 JSON 전송
    let msg = ManagerMessage::Directive(argus_shared::EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::Throttle { delay_ms: 10 }],
    });
    let json = serde_json::to_vec(&msg).unwrap();
    let len = (json.len() as u32).to_be_bytes();
    server_stream.write_all(&len).unwrap();
    server_stream.write_all(&json).unwrap();
    server_stream.flush().unwrap();

    drop(server_stream);

    let (is_parse_error, is_ok) = handle.join().unwrap();
    assert!(is_parse_error, "첫 프레임은 ParseError여야 함");
    assert!(is_ok, "두 번째 프레임은 정상 수신되어야 함");

    std::fs::remove_file(&path).ok();
}

// ═══════════════════════════════════════════════════════════════
// SEQ-081: 65537B 페이로드 거부 후 연결 유지
// ═══════════════════════════════════════════════════════════════

#[cfg(unix)]
#[test]
fn test_seq_081_oversized_payload_rejected_then_normal() {
    use argus_engine::resilience::{Transport, UnixSocketTransport};
    use std::io::Write;
    use std::os::unix::net::UnixListener;

    let path = std::env::temp_dir().join(format!(
        "argus_seq081_{}.sock",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let listener = UnixListener::bind(&path).unwrap();

    let path2 = path.clone();
    let handle = std::thread::spawn(move || {
        let mut transport = UnixSocketTransport::new(path2);
        transport.connect().unwrap();

        // 첫 번째 프레임: oversized -> ParseError
        let result1 = transport.recv();
        let is_rejected = matches!(
            result1,
            Err(argus_engine::resilience::TransportError::ParseError(_))
        );

        // 두 번째 프레임: 정상 수신
        let result2 = transport.recv();
        let is_ok = result2.is_ok();

        (is_rejected, is_ok)
    });

    let (mut server_stream, _) = listener.accept().unwrap();

    // 65537B 페이로드 길이 전송 (MAX_PAYLOAD_SIZE=65536 초과)
    let oversized_len: u32 = 65537;
    server_stream
        .write_all(&oversized_len.to_be_bytes())
        .unwrap();
    // oversized 길이 이후의 데이터는 보내지 않음 (read_length_prefixed가 길이 체크에서 먼저 거부)
    server_stream.flush().unwrap();

    // 정상 JSON 전송
    let msg = ManagerMessage::Directive(argus_shared::EngineDirective {
        seq_id: 2,
        commands: vec![EngineCommand::Throttle { delay_ms: 20 }],
    });
    let json = serde_json::to_vec(&msg).unwrap();
    let len = (json.len() as u32).to_be_bytes();
    server_stream.write_all(&len).unwrap();
    server_stream.write_all(&json).unwrap();
    server_stream.flush().unwrap();

    drop(server_stream);

    let (is_rejected, is_ok) = handle.join().unwrap();
    assert!(
        is_rejected,
        "oversized 페이로드는 거부(ParseError)되어야 함"
    );
    assert!(is_ok, "이후 정상 페이로드는 수신되어야 함");

    std::fs::remove_file(&path).ok();
}

// ═══════════════════════════════════════════════════════════════
// SEQ-082: 서버측 drop -> Engine Disconnected
// ═══════════════════════════════════════════════════════════════

#[cfg(unix)]
#[test]
fn test_seq_082_server_drop_disconnected() {
    use argus_engine::resilience::{Transport, TransportError, UnixSocketTransport};
    use std::os::unix::net::UnixListener;

    let path = std::env::temp_dir().join(format!(
        "argus_seq082_{}.sock",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let listener = UnixListener::bind(&path).unwrap();

    let path2 = path.clone();
    let handle = std::thread::spawn(move || {
        let mut transport = UnixSocketTransport::new(path2);
        transport.connect().unwrap();
        transport.recv()
    });

    // 서버 소켓 즉시 drop
    let (server_stream, _) = listener.accept().unwrap();
    drop(server_stream);

    let result = handle.join().unwrap();
    assert!(
        matches!(result, Err(TransportError::Disconnected)),
        "서버 drop 후 Disconnected 에러여야 함"
    );

    std::fs::remove_file(&path).ok();
}

// ═══════════════════════════════════════════════════════════════
// SEQ-083: EOF -> Disconnected
// ═══════════════════════════════════════════════════════════════

#[cfg(unix)]
#[test]
fn test_seq_083_eof_disconnected() {
    use argus_engine::resilience::{Transport, TransportError, UnixSocketTransport};
    use std::os::unix::net::UnixListener;

    let path = std::env::temp_dir().join(format!(
        "argus_seq083_{}.sock",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let listener = UnixListener::bind(&path).unwrap();

    let path2 = path.clone();
    let handle = std::thread::spawn(move || {
        let mut transport = UnixSocketTransport::new(path2);
        transport.connect().unwrap();
        transport.recv()
    });

    // accept 후 아무 데이터도 보내지 않고 종료
    let (server_stream, _) = listener.accept().unwrap();
    drop(server_stream);

    let result = handle.join().unwrap();
    assert!(
        matches!(result, Err(TransportError::Disconnected)),
        "EOF 후 Disconnected 에러여야 함"
    );

    std::fs::remove_file(&path).ok();
}

// ═══════════════════════════════════════════════════════════════
// SEQ-086: unknown type -> ParseError
// ═══════════════════════════════════════════════════════════════

#[cfg(unix)]
#[test]
fn test_seq_086_unknown_type_parse_error() {
    use argus_engine::resilience::{Transport, TransportError, UnixSocketTransport};
    use std::io::Write;
    use std::os::unix::net::UnixListener;

    let path = std::env::temp_dir().join(format!(
        "argus_seq086_{}.sock",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let listener = UnixListener::bind(&path).unwrap();

    let path2 = path.clone();
    let handle = std::thread::spawn(move || {
        let mut transport = UnixSocketTransport::new(path2);
        transport.connect().unwrap();
        transport.recv()
    });

    let (mut server_stream, _) = listener.accept().unwrap();

    // unknown type JSON 전송
    let unknown_json = br#"{"type":"unknown_type","data":"test"}"#;
    let len = (unknown_json.len() as u32).to_be_bytes();
    server_stream.write_all(&len).unwrap();
    server_stream.write_all(unknown_json).unwrap();
    server_stream.flush().unwrap();
    drop(server_stream);

    let result = handle.join().unwrap();
    assert!(
        matches!(result, Err(TransportError::ParseError(_))),
        "unknown type은 ParseError여야 함"
    );

    std::fs::remove_file(&path).ok();
}

// ═══════════════════════════════════════════════════════════════
// SEQ-092: 100개 Directive 고속 전송 -> 모두 처리
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_092_100_directives_all_processed() {
    let (mut executor, tx, resp_rx) = make_executor();

    // 100개 Directive 전송
    for i in 1..=100u64 {
        send_directive(&tx, i, vec![EngineCommand::Throttle { delay_ms: i }]);
    }

    // poll() 1회로 모두 처리
    let _plan = executor.poll(&empty_snap());

    // resp_rx에서 100개 Response 수신 확인
    let mut received_seq_ids = Vec::new();
    for _ in 0..100 {
        let msg = resp_rx
            .recv_timeout(Duration::from_millis(500))
            .expect("100개 Response 중 하나를 수신하지 못함");
        match msg {
            EngineMessage::Response(r) => {
                received_seq_ids.push(r.seq_id);
            }
            other => panic!("Response를 기대했으나 {:?} 수신", other),
        }
    }

    assert_eq!(received_seq_ids.len(), 100);
    // seq_id 순서대로 1..=100
    for (i, id) in received_seq_ids.iter().enumerate() {
        assert_eq!(*id, (i + 1) as u64, "seq_id가 순서대로 증가해야 함");
    }
}
