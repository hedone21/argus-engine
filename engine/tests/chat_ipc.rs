//! Integration tests for the chat IPC input multiplexer. Exercises the
//! Unix socket and TCP paths end-to-end: bind → connect → send line →
//! verify the mpsc receiver yields the message → stream bytes back →
//! verify the client reads them up to the 0x04 end-of-turn delimiter.

#![cfg(unix)]

use argus_engine::session::chat_ipc::{
    ChatInput, finish_reply_stream, spawn_socket_listener, spawn_tcp_listener, write_reply_bytes,
};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::time::Duration;

fn read_until_eot<R: Read>(r: &mut R) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while r.read_exact(&mut byte).is_ok() {
        if byte[0] == 0x04 {
            break;
        }
        buf.push(byte[0]);
    }
    buf
}

fn unique_sock_path(tag: &str) -> String {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("/tmp/argus_engine_chat_ipc_{tag}_{pid}_{nanos}.sock")
}

#[test]
fn socket_accepts_connection_and_delivers_line() {
    let path = unique_sock_path("single");
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_socket_listener(tx, &path).expect("spawn");

    // Give the listener thread a moment to bind.
    for _ in 0..50 {
        if std::path::Path::new(&path).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(std::path::Path::new(&path).exists(), "socket not bound");

    let mut client = UnixStream::connect(&path).expect("connect");
    client.write_all(b"hello world\n").unwrap();
    client.flush().unwrap();

    let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
    match msg {
        ChatInput::Line(line, writer) => {
            assert_eq!(line, "hello world");
            let writer = writer.expect("socket input must carry a reply writer");
            write_reply_bytes(Some(&writer), b"reply-payload");
            finish_reply_stream(Some(&writer));
        }
        _ => panic!("expected Line"),
    }

    let mut buf = Vec::new();
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    // Read until EOT (0x04). finish_reply_stream does NOT shut down the
    // write side, so we stop on the delimiter rather than EOF.
    let mut byte = [0u8; 1];
    while client.read_exact(&mut byte).is_ok() {
        if byte[0] == 0x04 {
            break;
        }
        buf.push(byte[0]);
    }
    assert_eq!(&buf, b"reply-payload");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn socket_supports_multiple_turns_on_same_connection() {
    let path = unique_sock_path("multi");
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_socket_listener(tx, &path).expect("spawn");

    for _ in 0..50 {
        if std::path::Path::new(&path).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let mut client = UnixStream::connect(&path).expect("connect");
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    for i in 0..3 {
        let user_line = format!("turn-{i}\n");
        client.write_all(user_line.as_bytes()).unwrap();
        client.flush().unwrap();

        let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
        let reply = match msg {
            ChatInput::Line(line, writer) => {
                assert_eq!(line, format!("turn-{i}"));
                let writer = writer.expect("writer");
                let payload = format!("ack-{i}");
                write_reply_bytes(Some(&writer), payload.as_bytes());
                finish_reply_stream(Some(&writer));
                payload
            }
            _ => panic!("expected Line"),
        };

        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        while client.read_exact(&mut byte).is_ok() {
            if byte[0] == 0x04 {
                break;
            }
            buf.push(byte[0]);
        }
        assert_eq!(String::from_utf8(buf).unwrap(), reply);
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn socket_handles_concurrent_connections() {
    let path = unique_sock_path("concurrent");
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_socket_listener(tx, &path).expect("spawn");

    for _ in 0..50 {
        if std::path::Path::new(&path).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let mut clients: Vec<UnixStream> = (0..3)
        .map(|_| UnixStream::connect(&path).expect("connect"))
        .collect();
    for (i, c) in clients.iter_mut().enumerate() {
        c.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        writeln!(c, "client-{i}").unwrap();
        c.flush().unwrap();
    }

    // Collect 3 messages regardless of ordering and ACK each one through
    // its own writer.
    let mut lines = Vec::new();
    for _ in 0..3 {
        let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
        if let ChatInput::Line(line, Some(writer)) = msg {
            let reply = format!("ack:{line}");
            write_reply_bytes(Some(&writer), reply.as_bytes());
            finish_reply_stream(Some(&writer));
            lines.push(line);
        } else {
            panic!("expected Line with writer");
        }
    }
    lines.sort();
    assert_eq!(lines, vec!["client-0", "client-1", "client-2"]);

    // Each client should see exactly its own ack payload.
    for (i, c) in clients.iter_mut().enumerate() {
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        while c.read_exact(&mut byte).is_ok() {
            if byte[0] == 0x04 {
                break;
            }
            buf.push(byte[0]);
        }
        assert_eq!(String::from_utf8(buf).unwrap(), format!("ack:client-{i}"));
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn rebind_removes_stale_socket_file() {
    let path = unique_sock_path("stale");
    // Create a stale regular file at the socket path.
    std::fs::write(&path, b"stale").unwrap();
    assert!(std::path::Path::new(&path).exists());

    let (tx, _rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_socket_listener(tx, &path).expect("spawn must remove stale file");
    for _ in 0..50 {
        if std::path::Path::new(&path).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    // After spawn, the file exists as a socket (bind created it).
    let meta = std::fs::metadata(&path).unwrap();
    assert!(
        std::os::unix::fs::FileTypeExt::is_socket(&meta.file_type()),
        "expected unix socket, got {:?}",
        meta.file_type()
    );

    let _ = std::fs::remove_file(&path);
}

// ─────────────────────── TCP listener tests ───────────────────────

#[test]
fn tcp_accepts_connection_and_delivers_line() {
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    let local = spawn_tcp_listener(tx, "127.0.0.1:0").expect("bind tcp");

    let mut client = TcpStream::connect(local).expect("connect");
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    client.write_all(b"hello tcp\n").unwrap();
    client.flush().unwrap();

    let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
    match msg {
        ChatInput::Line(line, writer) => {
            assert_eq!(line, "hello tcp");
            let writer = writer.expect("tcp input must carry a reply writer");
            write_reply_bytes(Some(&writer), b"tcp-reply");
            finish_reply_stream(Some(&writer));
        }
        _ => panic!("expected Line"),
    }

    let buf = read_until_eot(&mut client);
    assert_eq!(&buf, b"tcp-reply");
}

#[test]
fn tcp_supports_multiple_turns_on_same_connection() {
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    let local = spawn_tcp_listener(tx, "127.0.0.1:0").expect("bind tcp");

    let mut client = TcpStream::connect(local).expect("connect");
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    for i in 0..3 {
        writeln!(client, "tcp-turn-{i}").unwrap();
        client.flush().unwrap();

        let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
        let reply = match msg {
            ChatInput::Line(line, writer) => {
                assert_eq!(line, format!("tcp-turn-{i}"));
                let writer = writer.expect("writer");
                let payload = format!("tcp-ack-{i}");
                write_reply_bytes(Some(&writer), payload.as_bytes());
                finish_reply_stream(Some(&writer));
                payload
            }
            _ => panic!("expected Line"),
        };

        let buf = read_until_eot(&mut client);
        assert_eq!(String::from_utf8(buf).unwrap(), reply);
    }
}

#[test]
fn tcp_handles_concurrent_connections() {
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    let local = spawn_tcp_listener(tx, "127.0.0.1:0").expect("bind tcp");

    let mut clients: Vec<TcpStream> = (0..3)
        .map(|_| TcpStream::connect(local).expect("connect"))
        .collect();
    for (i, c) in clients.iter_mut().enumerate() {
        c.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        writeln!(c, "tcp-client-{i}").unwrap();
        c.flush().unwrap();
    }

    let mut lines = Vec::new();
    for _ in 0..3 {
        let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
        if let ChatInput::Line(line, Some(writer)) = msg {
            let reply = format!("tcp-ack:{line}");
            write_reply_bytes(Some(&writer), reply.as_bytes());
            finish_reply_stream(Some(&writer));
            lines.push(line);
        } else {
            panic!("expected Line with writer");
        }
    }
    lines.sort();
    assert_eq!(lines, vec!["tcp-client-0", "tcp-client-1", "tcp-client-2"]);

    for (i, c) in clients.iter_mut().enumerate() {
        let buf = read_until_eot(c);
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            format!("tcp-ack:tcp-client-{i}")
        );
    }
}

#[test]
fn tcp_and_unix_listeners_share_one_channel() {
    // Both a TCP listener and a Unix socket listener feed the same mpsc
    // channel. Messages from either transport appear in the same stream
    // and each carries a writer routed back to its own client.
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();

    let tcp_addr = spawn_tcp_listener(tx.clone(), "127.0.0.1:0").expect("tcp");
    let unix_path = unique_sock_path("combined");
    spawn_socket_listener(tx, &unix_path).expect("unix");
    for _ in 0..50 {
        if std::path::Path::new(&unix_path).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let mut tcp_client = TcpStream::connect(tcp_addr).expect("tcp connect");
    tcp_client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut unix_client = UnixStream::connect(&unix_path).expect("unix connect");
    unix_client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    writeln!(tcp_client, "from-tcp").unwrap();
    tcp_client.flush().unwrap();
    writeln!(unix_client, "from-unix").unwrap();
    unix_client.flush().unwrap();

    let mut seen = std::collections::HashMap::new();
    for _ in 0..2 {
        let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
        if let ChatInput::Line(line, Some(writer)) = msg {
            let reply = format!("ack:{line}");
            write_reply_bytes(Some(&writer), reply.as_bytes());
            finish_reply_stream(Some(&writer));
            seen.insert(line, reply);
        } else {
            panic!("expected Line");
        }
    }
    assert!(seen.contains_key("from-tcp"));
    assert!(seen.contains_key("from-unix"));

    let tcp_buf = read_until_eot(&mut tcp_client);
    assert_eq!(String::from_utf8(tcp_buf).unwrap(), "ack:from-tcp");
    let unix_buf = read_until_eot(&mut unix_client);
    assert_eq!(String::from_utf8(unix_buf).unwrap(), "ack:from-unix");

    let _ = std::fs::remove_file(&unix_path);
}
