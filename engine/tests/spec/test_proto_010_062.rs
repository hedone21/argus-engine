//! PROTO-010 ~ PROTO-062: Transport Protocol
//!
//! Length-prefix frame round-trip, MAX_PAYLOAD 제한,
//! UnixSocket/TCP connect/round-trip, parse error, connection closed 검증.
//!
//! 주의: transport.rs의 write_manager_message/read_engine_message는 #[cfg(test)]
//! 전용이므로 integration test에서는 직접 접근 불가.
//! 대신 MockTransport/TcpTransport/UnixSocketTransport의 public API를 사용.

use argus_engine::resilience::{MockTransport, TcpTransport, Transport, TransportError};
use argus_shared::{
    EngineCapability, EngineCommand, EngineDirective, EngineMessage, ManagerMessage,
};

fn sample_directive() -> ManagerMessage {
    ManagerMessage::Directive(EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::KvEvictH2o { keep_ratio: 0.85 }],
    })
}

fn sample_capability() -> EngineMessage {
    EngineMessage::Capability(EngineCapability {
        available_devices: vec!["cpu".into()],
        active_device: "cpu".into(),
        max_kv_tokens: 2048,
        bytes_per_kv_token: 256,
        num_layers: 16,
        ..Default::default()
    })
}

// ══════════════════════════════════════════════════════════════
// PROTO-010/014: MockTransport round-trip (length-prefix frame)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_proto_010_mock_from_messages_delivers_all() {
    let msgs = vec![
        sample_directive(),
        ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::Suspend],
        }),
    ];
    let mut transport = MockTransport::from_messages(msgs);
    assert!(transport.connect().is_ok());

    let m1 = transport.recv().unwrap();
    match m1 {
        ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
    }

    let m2 = transport.recv().unwrap();
    match m2 {
        ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 2),
    }

    // 모든 메시지 소진 후 Disconnected
    assert!(matches!(
        transport.recv(),
        Err(TransportError::Disconnected)
    ));
}

#[test]
fn test_proto_014_mock_bidirectional_round_trip() {
    let (mut transport, mgr) = MockTransport::bidirectional();
    transport.connect().unwrap();

    // Manager → Engine
    mgr.send(sample_directive()).unwrap();
    let msg = transport.recv().unwrap();
    match msg {
        ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
    }

    // Engine → Manager
    transport.send(&sample_capability()).unwrap();
    let resp = mgr.recv().unwrap();
    assert!(matches!(resp, EngineMessage::Capability(_)));
}

// ══════════════════════════════════════════════════════════════
// PROTO-030/031: UnixSocket + TCP round-trip
// ══════════════════════════════════════════════════════════════

#[cfg(unix)]
mod unix_proto {
    use super::*;
    use argus_engine::resilience::UnixSocketTransport;
    use std::os::unix::net::UnixListener;
    use std::path::PathBuf;

    fn tmp_socket_path() -> PathBuf {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("llm_spec_proto_{}.sock", id))
    }

    /// 서버 측에서 length-prefix JSON ManagerMessage를 전송하는 헬퍼
    fn write_manager_message_to_stream(
        stream: &mut std::os::unix::net::UnixStream,
        msg: &ManagerMessage,
    ) {
        use std::io::Write;
        let json = serde_json::to_vec(msg).unwrap();
        let len = (json.len() as u32).to_be_bytes();
        stream.write_all(&len).unwrap();
        stream.write_all(&json).unwrap();
        stream.flush().unwrap();
    }

    #[test]
    fn test_proto_030_unix_socket_round_trip() {
        let path = tmp_socket_path();
        let listener = UnixListener::bind(&path).unwrap();

        let path2 = path.clone();
        let handle = std::thread::spawn(move || {
            let mut transport = UnixSocketTransport::new(path2);
            transport.connect().unwrap();
            transport.recv().unwrap()
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        write_manager_message_to_stream(&mut server_stream, &sample_directive());
        drop(server_stream);

        let received = handle.join().unwrap();
        match received {
            ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_proto_043_unix_socket_connect_fail() {
        let mut transport =
            UnixSocketTransport::new(PathBuf::from("/tmp/nonexistent_llm_spec_test.sock"));
        assert!(matches!(
            transport.connect(),
            Err(TransportError::ConnectionFailed(_))
        ));
    }

    #[test]
    fn test_proto_061_unix_socket_parse_error() {
        use std::io::Write;
        let path = tmp_socket_path();
        let listener = UnixListener::bind(&path).unwrap();

        let path2 = path.clone();
        let handle = std::thread::spawn(move || {
            let mut transport = UnixSocketTransport::new(path2);
            transport.connect().unwrap();
            transport.recv()
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        let bad_json = b"not json!";
        let len = (bad_json.len() as u32).to_be_bytes();
        server_stream.write_all(&len).unwrap();
        server_stream.write_all(bad_json).unwrap();
        server_stream.flush().unwrap();
        drop(server_stream);

        let result = handle.join().unwrap();
        assert!(matches!(result, Err(TransportError::ParseError(_))));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_proto_012_unix_socket_oversized_rejected() {
        use std::io::Write;
        let path = tmp_socket_path();
        let listener = UnixListener::bind(&path).unwrap();

        let path2 = path.clone();
        let handle = std::thread::spawn(move || {
            let mut transport = UnixSocketTransport::new(path2);
            transport.connect().unwrap();
            transport.recv()
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        // MAX_PAYLOAD_SIZE = 64 * 1024 = 65536
        let len = (65537u32).to_be_bytes();
        server_stream.write_all(&len).unwrap();
        server_stream.flush().unwrap();
        drop(server_stream);

        let result = handle.join().unwrap();
        assert!(matches!(result, Err(TransportError::ParseError(_))));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_proto_062_unix_socket_connection_closed() {
        let path = tmp_socket_path();
        let listener = UnixListener::bind(&path).unwrap();

        let path2 = path.clone();
        let handle = std::thread::spawn(move || {
            let mut transport = UnixSocketTransport::new(path2);
            transport.connect().unwrap();
            transport.recv()
        });

        let (server_stream, _) = listener.accept().unwrap();
        drop(server_stream);

        let result = handle.join().unwrap();
        assert!(matches!(result, Err(TransportError::Disconnected)));
        std::fs::remove_file(&path).ok();
    }
}

// ══════════════════════════════════════════════════════════════
// PROTO-031: TCP round-trip
// ══════════════════════════════════════════════════════════════

mod tcp_proto {
    use super::*;
    use std::net::TcpListener;

    fn free_tcp_addr() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        format!("127.0.0.1:{}", addr.port())
    }

    fn write_manager_message_to_tcp(stream: &mut std::net::TcpStream, msg: &ManagerMessage) {
        use std::io::Write;
        let json = serde_json::to_vec(msg).unwrap();
        let len = (json.len() as u32).to_be_bytes();
        stream.write_all(&len).unwrap();
        stream.write_all(&json).unwrap();
        stream.flush().unwrap();
    }

    #[test]
    fn test_proto_031_tcp_round_trip() {
        let addr = free_tcp_addr();
        let listener = TcpListener::bind(&addr).unwrap();

        let addr2 = addr.clone();
        let handle = std::thread::spawn(move || {
            let mut transport = TcpTransport::new(addr2);
            transport.connect().unwrap();
            transport.recv().unwrap()
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        write_manager_message_to_tcp(&mut server_stream, &sample_directive());
        drop(server_stream);

        let received = handle.join().unwrap();
        match received {
            ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
        }
    }

    #[test]
    fn test_proto_043_tcp_connect_fail() {
        let mut transport = TcpTransport::new("127.0.0.1:1".into());
        assert!(matches!(
            transport.connect(),
            Err(TransportError::ConnectionFailed(_))
        ));
    }

    #[test]
    fn test_proto_061_tcp_parse_error() {
        use std::io::Write;
        let addr = free_tcp_addr();
        let listener = TcpListener::bind(&addr).unwrap();

        let addr2 = addr.clone();
        let handle = std::thread::spawn(move || {
            let mut transport = TcpTransport::new(addr2);
            transport.connect().unwrap();
            transport.recv()
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        let bad_json = b"not json!";
        let len = (bad_json.len() as u32).to_be_bytes();
        server_stream.write_all(&len).unwrap();
        server_stream.write_all(bad_json).unwrap();
        server_stream.flush().unwrap();
        drop(server_stream);

        let result = handle.join().unwrap();
        assert!(matches!(result, Err(TransportError::ParseError(_))));
    }

    #[test]
    fn test_proto_012_tcp_oversized_rejected() {
        use std::io::Write;
        let addr = free_tcp_addr();
        let listener = TcpListener::bind(&addr).unwrap();

        let addr2 = addr.clone();
        let handle = std::thread::spawn(move || {
            let mut transport = TcpTransport::new(addr2);
            transport.connect().unwrap();
            transport.recv()
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        let len = (65537u32).to_be_bytes();
        server_stream.write_all(&len).unwrap();
        server_stream.flush().unwrap();
        drop(server_stream);

        let result = handle.join().unwrap();
        assert!(matches!(result, Err(TransportError::ParseError(_))));
    }

    #[test]
    fn test_proto_062_tcp_connection_closed() {
        let addr = free_tcp_addr();
        let listener = TcpListener::bind(&addr).unwrap();

        let addr2 = addr.clone();
        let handle = std::thread::spawn(move || {
            let mut transport = TcpTransport::new(addr2);
            transport.connect().unwrap();
            transport.recv()
        });

        let (server_stream, _) = listener.accept().unwrap();
        drop(server_stream);

        let result = handle.join().unwrap();
        assert!(matches!(result, Err(TransportError::Disconnected)));
    }
}
