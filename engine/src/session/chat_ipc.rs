//! Phase 4-5-b 이관: core → session (INV-LAYER V-11 해소)
//! Chat mode input multiplexer (stdin + optional Unix/TCP socket).
//!
//! Provides a single `Receiver<ChatInput>` that merges newline-delimited
//! messages from stdin, from a Unix domain socket listener, and/or from
//! a TCP listener. Each socket-sourced message carries a writer handle
//! so reply bytes stream back to the same connection.

use std::io::Write as _;

/// Bidirectional stream wrapping either a Unix domain socket or TCP
/// connection. The reply path uses this behind an `Arc<Mutex<..>>` so
/// multiple reply bytes can be routed to the same client safely.
pub enum ChatStream {
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixStream),
    Tcp(std::net::TcpStream),
}

impl ChatStream {
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Unix(s) => s.write_all(bytes),
            Self::Tcp(s) => s.write_all(bytes),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Unix(s) => s.flush(),
            Self::Tcp(s) => s.flush(),
        }
    }
}

pub type ChatReplyWriter = std::sync::Arc<std::sync::Mutex<ChatStream>>;

pub enum ChatInput {
    Line(String, Option<ChatReplyWriter>),
    Eof,
}

/// Write bytes to the reply stream if present. Errors are swallowed — a
/// client that has closed its end should not abort the model loop.
pub fn write_reply_bytes(reply: Option<&ChatReplyWriter>, bytes: &[u8]) {
    if let Some(arc) = reply
        && let Ok(mut s) = arc.lock()
    {
        let _ = s.write_all(bytes);
        let _ = s.flush();
    }
}

/// Delimit end-of-turn on the reply stream with a 0x04 (EOT) byte. Does
/// not shut down the write side — the stream stays open so the next user
/// message on the same connection can receive its own reply.
pub fn finish_reply_stream(reply: Option<&ChatReplyWriter>) {
    if let Some(arc) = reply
        && let Ok(mut s) = arc.lock()
    {
        let _ = s.write_all(&[0x04]);
        let _ = s.flush();
    }
}

/// Spawn a stdin reader thread that forwards each newline-delimited line
/// into the given sender. On EOF or error, sends one `ChatInput::Eof` and
/// exits.
pub fn spawn_stdin_reader(tx: std::sync::mpsc::Sender<ChatInput>) {
    std::thread::spawn(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        loop {
            let mut buf = String::new();
            match reader.read_line(&mut buf) {
                Ok(0) => {
                    let _ = tx.send(ChatInput::Eof);
                    break;
                }
                Ok(_) => {
                    if tx.send(ChatInput::Line(buf, None)).is_err() {
                        break;
                    }
                }
                Err(_) => {
                    let _ = tx.send(ChatInput::Eof);
                    break;
                }
            }
        }
    });
}

/// Bind a Unix socket listener and spawn an accept loop. Each accepted
/// connection gets its own reader thread that forwards newline-delimited
/// lines into `tx`, tagged with a writer handle pointing at the same
/// connection so replies stream back.
#[cfg(unix)]
pub fn spawn_socket_listener(
    tx: std::sync::mpsc::Sender<ChatInput>,
    path: &str,
) -> anyhow::Result<()> {
    use std::os::unix::net::UnixListener;
    let path = path.to_string();
    // Remove any stale socket file before binding.
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path)
        .map_err(|e| anyhow::anyhow!("failed to bind Unix socket at {}: {}", path, e))?;
    eprintln!("[Chat] Listening for Unix socket input at {}", path);
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(_) => continue,
            };
            let writer_clone = match stream.try_clone() {
                Ok(c) => c,
                Err(_) => continue,
            };
            let stream_arc =
                std::sync::Arc::new(std::sync::Mutex::new(ChatStream::Unix(writer_clone)));
            let conn_tx = tx.clone();
            spawn_line_reader_unix(stream, stream_arc, conn_tx);
        }
    });
    Ok(())
}

#[cfg(unix)]
fn spawn_line_reader_unix(
    reader_stream: std::os::unix::net::UnixStream,
    writer: ChatReplyWriter,
    tx: std::sync::mpsc::Sender<ChatInput>,
) {
    std::thread::spawn(move || {
        use std::io::BufRead;
        let reader = std::io::BufReader::new(reader_stream);
        for line in reader.lines().map_while(Result::ok) {
            if tx
                .send(ChatInput::Line(line, Some(writer.clone())))
                .is_err()
            {
                break;
            }
        }
    });
}

/// Bind a TCP listener and spawn an accept loop. Behaves like
/// `spawn_socket_listener` but over TCP. `addr` is passed through to
/// `TcpListener::bind` (e.g. "127.0.0.1:7878", "[::1]:7878", "0.0.0.0:0").
pub fn spawn_tcp_listener(
    tx: std::sync::mpsc::Sender<ChatInput>,
    addr: &str,
) -> anyhow::Result<std::net::SocketAddr> {
    use std::net::TcpListener;
    let listener = TcpListener::bind(addr)
        .map_err(|e| anyhow::anyhow!("failed to bind TCP listener at {}: {}", addr, e))?;
    let local = listener
        .local_addr()
        .map_err(|e| anyhow::anyhow!("listener has no local addr: {}", e))?;
    eprintln!("[Chat] Listening for TCP input at {}", local);
    if !local.ip().is_loopback() {
        eprintln!(
            "[Chat] WARNING: TCP listener bound to non-loopback address {}. \
             Anyone with network access can inject chat input.",
            local
        );
    }
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(_) => continue,
            };
            // Nagle off → reply bytes reach the client promptly.
            let _ = stream.set_nodelay(true);
            let writer_clone = match stream.try_clone() {
                Ok(c) => c,
                Err(_) => continue,
            };
            let stream_arc =
                std::sync::Arc::new(std::sync::Mutex::new(ChatStream::Tcp(writer_clone)));
            let conn_tx = tx.clone();
            spawn_line_reader_tcp(stream, stream_arc, conn_tx);
        }
    });
    Ok(local)
}

fn spawn_line_reader_tcp(
    reader_stream: std::net::TcpStream,
    writer: ChatReplyWriter,
    tx: std::sync::mpsc::Sender<ChatInput>,
) {
    std::thread::spawn(move || {
        use std::io::BufRead;
        let reader = std::io::BufReader::new(reader_stream);
        for line in reader.lines().map_while(Result::ok) {
            if tx
                .send(ChatInput::Line(line, Some(writer.clone())))
                .is_err()
            {
                break;
            }
        }
    });
}

/// Spawn the stdin reader and (optionally) Unix and/or TCP listeners.
/// Both socket listeners can be active simultaneously and feed the same
/// channel. Use `spawn_stdin_reader` / `spawn_socket_listener` /
/// `spawn_tcp_listener` directly for tests or custom compositions.
pub fn spawn_chat_input_sources(
    unix_path: Option<&str>,
    tcp_addr: Option<&str>,
) -> anyhow::Result<std::sync::mpsc::Receiver<ChatInput>> {
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_stdin_reader(tx.clone());

    #[cfg(unix)]
    if let Some(path) = unix_path {
        spawn_socket_listener(tx.clone(), path)?;
    }
    #[cfg(not(unix))]
    if unix_path.is_some() {
        anyhow::bail!("--chat-socket is only supported on unix targets");
    }

    if let Some(addr) = tcp_addr {
        spawn_tcp_listener(tx.clone(), addr)?;
    }
    drop(tx);
    Ok(rx)
}
