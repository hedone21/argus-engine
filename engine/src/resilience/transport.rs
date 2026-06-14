use std::fmt;
use std::io::Read;
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::mpsc;

use argus_shared::{EngineMessage, ManagerMessage};

// ── TransportError ──────────────────────────────────────────

/// Error type for transport operations.
#[derive(Debug)]
pub enum TransportError {
    ConnectionFailed(String),
    Disconnected,
    Timeout,
    ParseError(String),
    Io(std::io::Error),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::ConnectionFailed(msg) => write!(f, "connection failed: {}", msg),
            TransportError::Disconnected => write!(f, "disconnected"),
            TransportError::Timeout => write!(f, "timeout"),
            TransportError::ParseError(msg) => write!(f, "parse error: {}", msg),
            TransportError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<std::io::Error> for TransportError {
    fn from(e: std::io::Error) -> Self {
        TransportError::Io(e)
    }
}

// ── Transport trait ─────────────────────────────────────────

/// Bidirectional transport for Manager↔Engine communication.
///
/// - `recv()` blocks until a `ManagerMessage` arrives (Manager→Engine).
/// - `send()` writes an `EngineMessage` to the Manager (Engine→Manager).
pub trait Transport: Send + 'static {
    /// Establish connection to the signal source.
    fn connect(&mut self) -> Result<(), TransportError>;

    /// Receive the next message from Manager (blocking).
    fn recv(&mut self) -> Result<ManagerMessage, TransportError>;

    /// Send a message to Manager (Engine→Manager).
    fn send(&mut self, msg: &EngineMessage) -> Result<(), TransportError>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

// ── MessageLoop ─────────────────────────────────────────────

/// Bidirectional message loop bridging Transport to mpsc channels.
///
/// The reader thread owns the transport and does both recv (blocking) and
/// send (via try_recv on write queue between blocking recv calls).
/// The resp_tx channel is used by the executor to queue outgoing messages.
pub struct MessageLoop;

impl MessageLoop {
    /// Spawn the message loop thread.
    ///
    /// Returns channels for the executor:
    /// - `cmd_rx`: receives ManagerMessages from transport
    /// - `resp_tx`: sends EngineMessages to transport
    #[allow(clippy::type_complexity)]
    pub fn spawn<T: Transport>(
        mut transport: T,
    ) -> Result<
        (
            mpsc::Receiver<ManagerMessage>,
            mpsc::Sender<EngineMessage>,
            std::thread::JoinHandle<()>,
        ),
        TransportError,
    > {
        transport.connect()?;

        let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
        let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();

        let transport_name = transport.name().to_string();

        let handle = std::thread::Builder::new()
            .name(format!("{}-loop", transport_name))
            .spawn(move || {
                Self::run_loop(transport, cmd_tx, resp_rx);
            })
            .expect("Failed to spawn message loop thread");

        Ok((cmd_rx, resp_tx, handle))
    }

    fn run_loop<T: Transport>(
        mut transport: T,
        cmd_tx: mpsc::Sender<ManagerMessage>,
        resp_rx: mpsc::Receiver<EngineMessage>,
    ) {
        loop {
            // Drain all pending outgoing messages first
            while let Ok(msg) = resp_rx.try_recv() {
                if let Err(e) = transport.send(&msg) {
                    log::warn!("{} send error: {}. Stopping.", transport.name(), e);
                    return;
                }
            }

            // Blocking recv for incoming messages
            match transport.recv() {
                Ok(msg) => {
                    if cmd_tx.send(msg).is_err() {
                        log::info!(
                            "Receiver dropped. Stopping {} message loop.",
                            transport.name()
                        );
                        break;
                    }
                }
                Err(TransportError::Timeout) => {
                    // Read timeout — normal for non-blocking poll cycle.
                    // Loop back to try_recv to drain outgoing messages.
                    continue;
                }
                Err(TransportError::ParseError(msg)) => {
                    log::warn!("{} parse error: {}. Skipping.", transport.name(), msg);
                    continue;
                }
                Err(TransportError::Disconnected) => {
                    log::info!("{} disconnected. Stopping message loop.", transport.name());
                    break;
                }
                Err(e) => {
                    log::warn!("{} error: {}. Stopping message loop.", transport.name(), e);
                    break;
                }
            }
        }
    }
}

// ── MockTransport ───────────────────────────────────────────

/// Mock transport for testing. Bidirectional via channels.
pub struct MockTransport {
    rx: mpsc::Receiver<ManagerMessage>,
    sent: mpsc::Sender<EngineMessage>,
}

/// Sender handle paired with MockTransport (simulates Manager side).
pub struct MockSender {
    tx: mpsc::Sender<ManagerMessage>,
}

impl MockSender {
    pub fn send(&self, msg: ManagerMessage) -> Result<(), mpsc::SendError<ManagerMessage>> {
        self.tx.send(msg)
    }
}

/// Manager-side handle for bidirectional mock testing.
pub struct MockManagerEnd {
    pub tx: mpsc::Sender<ManagerMessage>,
    pub rx: mpsc::Receiver<EngineMessage>,
}

impl MockManagerEnd {
    pub fn send(&self, msg: ManagerMessage) -> Result<(), mpsc::SendError<ManagerMessage>> {
        self.tx.send(msg)
    }

    pub fn recv(&self) -> Result<EngineMessage, mpsc::RecvError> {
        self.rx.recv()
    }

    pub fn try_recv(&self) -> Result<EngineMessage, mpsc::TryRecvError> {
        self.rx.try_recv()
    }
}

impl MockTransport {
    /// Create a channel-based mock transport (unidirectional sender).
    pub fn channel() -> (Self, MockSender) {
        let (tx, rx) = mpsc::channel();
        let (sent_tx, _sent_rx) = mpsc::channel();
        (Self { rx, sent: sent_tx }, MockSender { tx })
    }

    /// Create a bidirectional mock transport.
    /// Returns (transport, manager_end) where manager_end can send commands and receive responses.
    pub fn bidirectional() -> (Self, MockManagerEnd) {
        let (mgr_tx, eng_rx) = mpsc::channel(); // Manager → Engine
        let (eng_tx, mgr_rx) = mpsc::channel(); // Engine → Manager
        (
            Self {
                rx: eng_rx,
                sent: eng_tx,
            },
            MockManagerEnd {
                tx: mgr_tx,
                rx: mgr_rx,
            },
        )
    }

    /// Create a mock transport pre-loaded with ManagerMessages.
    pub fn from_messages(messages: Vec<ManagerMessage>) -> Self {
        let (tx, rx) = mpsc::channel();
        for msg in messages {
            tx.send(msg).unwrap();
        }
        drop(tx);
        let (sent_tx, _sent_rx) = mpsc::channel();
        Self { rx, sent: sent_tx }
    }
}

impl Transport for MockTransport {
    fn connect(&mut self) -> Result<(), TransportError> {
        Ok(())
    }

    fn recv(&mut self) -> Result<ManagerMessage, TransportError> {
        self.rx.recv().map_err(|_| TransportError::Disconnected)
    }

    fn send(&mut self, msg: &EngineMessage) -> Result<(), TransportError> {
        self.sent
            .send(msg.clone())
            .map_err(|_| TransportError::Disconnected)
    }

    fn name(&self) -> &str {
        "Mock"
    }
}

// ── Wire format helpers ──────────────────────────────────────

/// Maximum message payload size (64KB sanity check).
const MAX_PAYLOAD_SIZE: u32 = 64 * 1024;

/// Read exactly `buf.len()` bytes from `reader`, mapping EOF to `Disconnected`.
fn read_exact_from<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<(), TransportError> {
    match reader.read_exact(buf) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            Err(TransportError::Disconnected)
        }
        Err(e)
            if e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut =>
        {
            Err(TransportError::Timeout)
        }
        Err(e) => Err(TransportError::Io(e)),
    }
}

/// Read a length-prefixed JSON message from `reader`.
/// Wire format: `[4 bytes BE u32 length][UTF-8 JSON payload]`
fn read_length_prefixed<R: Read, T: serde::de::DeserializeOwned>(
    reader: &mut R,
) -> Result<T, TransportError> {
    let mut len_buf = [0u8; 4];
    read_exact_from(reader, &mut len_buf)?;
    let len = u32::from_be_bytes(len_buf);

    if len > MAX_PAYLOAD_SIZE {
        return Err(TransportError::ParseError(format!(
            "payload too large: {} bytes (max {})",
            len, MAX_PAYLOAD_SIZE
        )));
    }

    let mut payload = vec![0u8; len as usize];
    read_exact_from(reader, &mut payload)?;

    serde_json::from_slice(&payload)
        .map_err(|e| TransportError::ParseError(format!("invalid JSON: {}", e)))
}

/// Write a length-prefixed JSON message to `writer`.
fn write_length_prefixed<W: std::io::Write, T: serde::Serialize>(
    writer: &mut W,
    msg: &T,
) -> Result<(), TransportError> {
    let json = serde_json::to_vec(msg)
        .map_err(|e| TransportError::ParseError(format!("serialize error: {}", e)))?;
    let len = (json.len() as u32).to_be_bytes();
    writer.write_all(&len)?;
    writer.write_all(&json)?;
    writer.flush()?;
    Ok(())
}

// ── UnixSocketTransport ─────────────────────────────────────

/// Unix domain socket transport (bidirectional).
/// Wire format: `[4 bytes BE u32 length][UTF-8 JSON payload]`
#[cfg(unix)]
pub struct UnixSocketTransport {
    path: PathBuf,
    reader: Option<UnixStream>,
    writer: Option<UnixStream>,
}

#[cfg(unix)]
impl UnixSocketTransport {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            reader: None,
            writer: None,
        }
    }

    /// Create from an already-connected stream (for testing).
    #[cfg(test)]
    pub fn from_stream(stream: UnixStream) -> Self {
        let writer = stream.try_clone().expect("try_clone failed in test");
        Self {
            path: PathBuf::new(),
            reader: Some(stream),
            writer: Some(writer),
        }
    }
}

// ── TcpTransport ────────────────────────────────────────────

/// TCP loopback transport for Android SELinux environments where
/// Unix domain socket bind is blocked.
/// Wire format: `[4 bytes BE u32 length][UTF-8 JSON payload]` (same as Unix transport)
pub struct TcpTransport {
    addr: String,
    reader: Option<TcpStream>,
    writer: Option<TcpStream>,
}

impl TcpTransport {
    pub fn new(addr: String) -> Self {
        Self {
            addr,
            reader: None,
            writer: None,
        }
    }
}

impl Transport for TcpTransport {
    fn connect(&mut self) -> Result<(), TransportError> {
        match TcpStream::connect(&self.addr) {
            Ok(stream) => {
                stream
                    .set_read_timeout(Some(std::time::Duration::from_millis(50)))
                    .map_err(TransportError::Io)?;
                let writer = stream.try_clone().map_err(TransportError::Io)?;
                self.reader = Some(stream);
                self.writer = Some(writer);
                Ok(())
            }
            Err(e) => Err(TransportError::ConnectionFailed(format!(
                "{}: {}",
                self.addr, e
            ))),
        }
    }

    fn recv(&mut self) -> Result<ManagerMessage, TransportError> {
        let stream = self
            .reader
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionFailed("not connected".into()))?;
        read_length_prefixed(stream)
    }

    fn send(&mut self, msg: &EngineMessage) -> Result<(), TransportError> {
        let stream = self
            .writer
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionFailed("not connected".into()))?;
        write_length_prefixed(stream, msg)
    }

    fn name(&self) -> &str {
        "Tcp"
    }
}

#[cfg(unix)]
impl Transport for UnixSocketTransport {
    fn connect(&mut self) -> Result<(), TransportError> {
        match UnixStream::connect(&self.path) {
            Ok(stream) => {
                let writer = stream.try_clone().map_err(TransportError::Io)?;
                self.reader = Some(stream);
                self.writer = Some(writer);
                Ok(())
            }
            Err(e) => Err(TransportError::ConnectionFailed(format!(
                "{}: {}",
                self.path.display(),
                e
            ))),
        }
    }

    fn recv(&mut self) -> Result<ManagerMessage, TransportError> {
        let stream = self
            .reader
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionFailed("not connected".into()))?;
        read_length_prefixed(stream)
    }

    fn send(&mut self, msg: &EngineMessage) -> Result<(), TransportError> {
        let stream = self
            .writer
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionFailed("not connected".into()))?;
        write_length_prefixed(stream, msg)
    }

    fn name(&self) -> &str {
        "UnixSocket"
    }
}

/// Helper: write a length-prefixed JSON ManagerMessage to any `Write` stream.
#[cfg(test)]
pub fn write_manager_message<W: std::io::Write>(
    stream: &mut W,
    msg: &ManagerMessage,
) -> std::io::Result<()> {
    let json = serde_json::to_vec(msg).unwrap();
    let len = (json.len() as u32).to_be_bytes();
    stream.write_all(&len)?;
    stream.write_all(&json)?;
    stream.flush()
}

/// Helper: read a length-prefixed JSON EngineMessage from any `Read` stream.
#[cfg(test)]
pub fn read_engine_message<R: Read>(stream: &mut R) -> std::io::Result<EngineMessage> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    serde_json::from_slice(&payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use argus_shared::*;

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

    // ── MockTransport tests ────────────────────────────────

    #[test]
    fn test_mock_from_messages_delivers_all() {
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

        assert!(matches!(
            transport.recv(),
            Err(TransportError::Disconnected)
        ));
    }

    #[test]
    fn test_mock_channel_send_recv() {
        let (mut transport, sender) = MockTransport::channel();
        assert!(transport.connect().is_ok());

        sender.send(sample_directive()).unwrap();
        let msg = transport.recv().unwrap();
        match msg {
            ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
        }
    }

    #[test]
    fn test_mock_channel_disconnect() {
        let (mut transport, sender) = MockTransport::channel();
        assert!(transport.connect().is_ok());
        drop(sender);
        assert!(matches!(
            transport.recv(),
            Err(TransportError::Disconnected)
        ));
    }

    #[test]
    fn test_mock_connect_always_ok() {
        let transport = MockTransport::from_messages(vec![]);
        assert!(transport.rx.try_recv().is_err());
    }

    #[test]
    fn test_mock_bidirectional() {
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

    #[test]
    fn test_mock_bidirectional_multiple_roundtrips() {
        let (mut transport, mgr) = MockTransport::bidirectional();
        transport.connect().unwrap();

        for i in 1..=5 {
            let directive = ManagerMessage::Directive(EngineDirective {
                seq_id: i,
                commands: vec![EngineCommand::Throttle { delay_ms: 30 }],
            });
            mgr.send(directive).unwrap();
            let msg = transport.recv().unwrap();
            match msg {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, i),
            }

            let response = EngineMessage::Response(CommandResponse {
                seq_id: i,
                results: vec![CommandResult::Ok],
            });
            transport.send(&response).unwrap();
            let resp = mgr.recv().unwrap();
            match resp {
                EngineMessage::Response(r) => assert_eq!(r.seq_id, i),
                _ => panic!("Expected Response"),
            }
        }
    }

    // ── UnixSocketTransport tests ──────────────────────────

    #[cfg(unix)]
    mod unix_tests {
        use super::*;
        use std::os::unix::net::UnixListener;

        fn tmp_socket_path() -> PathBuf {
            let id = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            std::env::temp_dir().join(format!("argus_test_{}.sock", id))
        }

        #[test]
        fn test_unix_socket_round_trip() {
            let path = tmp_socket_path();
            let listener = UnixListener::bind(&path).unwrap();

            let path2 = path.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = UnixSocketTransport::new(path2);
                transport.connect().unwrap();
                transport.recv().unwrap()
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            let msg = sample_directive();
            write_manager_message(&mut server_stream, &msg).unwrap();
            drop(server_stream);

            let received = handle.join().unwrap();
            match received {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
            }
            std::fs::remove_file(&path).ok();
        }

        #[test]
        fn test_unix_socket_bidirectional() {
            let path = tmp_socket_path();
            let listener = UnixListener::bind(&path).unwrap();

            let path2 = path.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = UnixSocketTransport::new(path2);
                transport.connect().unwrap();

                // Receive directive
                let msg = transport.recv().unwrap();

                // Send capability back
                transport.send(&sample_capability()).unwrap();

                msg
            });

            let (mut server_stream, _) = listener.accept().unwrap();

            // Send directive
            write_manager_message(&mut server_stream, &sample_directive()).unwrap();

            // Read response
            let response = read_engine_message(&mut server_stream).unwrap();
            assert!(matches!(response, EngineMessage::Capability(_)));

            drop(server_stream);
            let received = handle.join().unwrap();
            match received {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
            }
            std::fs::remove_file(&path).ok();
        }

        #[test]
        fn test_unix_socket_multiple_messages() {
            let path = tmp_socket_path();
            let listener = UnixListener::bind(&path).unwrap();

            let path2 = path.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = UnixSocketTransport::new(path2);
                transport.connect().unwrap();
                let m1 = transport.recv().unwrap();
                let m2 = transport.recv().unwrap();
                let m3 = transport.recv().unwrap();
                (m1, m2, m3)
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            for seq_id in 1..=3 {
                let msg = ManagerMessage::Directive(EngineDirective {
                    seq_id,
                    commands: vec![EngineCommand::Throttle { delay_ms: 0 }],
                });
                write_manager_message(&mut server_stream, &msg).unwrap();
            }
            drop(server_stream);

            let (m1, m2, m3) = handle.join().unwrap();
            match m1 {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
            }
            match m2 {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 2),
            }
            match m3 {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 3),
            }
            std::fs::remove_file(&path).ok();
        }

        #[test]
        fn test_unix_socket_parse_error() {
            use std::io::Write as _;
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
        fn test_unix_socket_oversized_rejected() {
            use std::io::Write as _;
            let path = tmp_socket_path();
            let listener = UnixListener::bind(&path).unwrap();

            let path2 = path.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = UnixSocketTransport::new(path2);
                transport.connect().unwrap();
                transport.recv()
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            let len = (MAX_PAYLOAD_SIZE + 1).to_be_bytes();
            server_stream.write_all(&len).unwrap();
            server_stream.flush().unwrap();
            drop(server_stream);

            let result = handle.join().unwrap();
            assert!(matches!(result, Err(TransportError::ParseError(_))));
            std::fs::remove_file(&path).ok();
        }

        #[test]
        fn test_unix_socket_connection_closed() {
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

        #[test]
        fn test_unix_socket_connect_fail() {
            let mut transport =
                UnixSocketTransport::new(PathBuf::from("/tmp/nonexistent_llm_test.sock"));
            assert!(matches!(
                transport.connect(),
                Err(TransportError::ConnectionFailed(_))
            ));
        }
    }

    // ── TcpTransport tests ────────────────────────────────

    mod tcp_tests {
        use super::*;
        use std::net::TcpListener;

        fn free_tcp_addr() -> String {
            // Bind to port 0, let the OS assign a free port, then release.
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            format!("127.0.0.1:{}", addr.port())
        }

        #[test]
        fn test_tcp_round_trip() {
            let addr = free_tcp_addr();
            let listener = TcpListener::bind(&addr).unwrap();

            let addr2 = addr.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = TcpTransport::new(addr2);
                transport.connect().unwrap();
                transport.recv().unwrap()
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            let msg = sample_directive();
            write_manager_message(&mut server_stream, &msg).unwrap();
            drop(server_stream);

            let received = handle.join().unwrap();
            match received {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
            }
        }

        #[test]
        fn test_tcp_bidirectional() {
            let addr = free_tcp_addr();
            let listener = TcpListener::bind(&addr).unwrap();

            let addr2 = addr.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = TcpTransport::new(addr2);
                transport.connect().unwrap();
                let msg = transport.recv().unwrap();
                transport.send(&sample_capability()).unwrap();
                msg
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            write_manager_message(&mut server_stream, &sample_directive()).unwrap();
            let response = read_engine_message(&mut server_stream).unwrap();
            assert!(matches!(response, EngineMessage::Capability(_)));

            drop(server_stream);
            let received = handle.join().unwrap();
            match received {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
            }
        }

        #[test]
        fn test_tcp_multiple_messages() {
            let addr = free_tcp_addr();
            let listener = TcpListener::bind(&addr).unwrap();

            let addr2 = addr.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = TcpTransport::new(addr2);
                transport.connect().unwrap();
                let m1 = transport.recv().unwrap();
                let m2 = transport.recv().unwrap();
                let m3 = transport.recv().unwrap();
                (m1, m2, m3)
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            for seq_id in 1u64..=3 {
                let msg = ManagerMessage::Directive(EngineDirective {
                    seq_id,
                    commands: vec![EngineCommand::Throttle { delay_ms: 0 }],
                });
                write_manager_message(&mut server_stream, &msg).unwrap();
            }
            drop(server_stream);

            let (m1, m2, m3) = handle.join().unwrap();
            match m1 {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
            }
            match m2 {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 2),
            }
            match m3 {
                ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 3),
            }
        }

        #[test]
        fn test_tcp_parse_error() {
            use std::io::Write as _;
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
        fn test_tcp_oversized_rejected() {
            use std::io::Write as _;
            let addr = free_tcp_addr();
            let listener = TcpListener::bind(&addr).unwrap();

            let addr2 = addr.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = TcpTransport::new(addr2);
                transport.connect().unwrap();
                transport.recv()
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            let len = (MAX_PAYLOAD_SIZE + 1).to_be_bytes();
            server_stream.write_all(&len).unwrap();
            server_stream.flush().unwrap();
            drop(server_stream);

            let result = handle.join().unwrap();
            assert!(matches!(result, Err(TransportError::ParseError(_))));
        }

        #[test]
        fn test_tcp_connection_closed() {
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

        #[test]
        fn test_tcp_connect_fail() {
            // Port 1 is privileged and almost certainly not listening.
            let mut transport = TcpTransport::new("127.0.0.1:1".into());
            assert!(matches!(
                transport.connect(),
                Err(TransportError::ConnectionFailed(_))
            ));
        }

        #[test]
        fn test_tcp_name() {
            let t = TcpTransport::new("127.0.0.1:9100".into());
            assert_eq!(t.name(), "Tcp");
        }
    }

    // ── MessageLoop tests ──────────────────────────────────

    #[test]
    fn test_message_loop_receives_directives() {
        let msgs = vec![
            sample_directive(),
            ManagerMessage::Directive(EngineDirective {
                seq_id: 2,
                commands: vec![EngineCommand::Suspend],
            }),
        ];
        let transport = MockTransport::from_messages(msgs);

        let (cmd_rx, _resp_tx, _handle) = MessageLoop::spawn(transport).unwrap();

        let m1 = cmd_rx.recv().unwrap();
        match m1 {
            ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
        }
        let m2 = cmd_rx.recv().unwrap();
        match m2 {
            ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 2),
        }
    }

    #[test]
    fn test_message_loop_bidirectional() {
        let (transport, mgr) = MockTransport::bidirectional();

        let (cmd_rx, _resp_tx, _handle) = MessageLoop::spawn(transport).unwrap();

        // Manager sends directive
        mgr.send(sample_directive()).unwrap();

        // Engine receives it
        let msg = cmd_rx.recv().unwrap();
        match msg {
            ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
        }

        // Drop manager to disconnect, which allows thread to exit
        drop(mgr);
    }
}
