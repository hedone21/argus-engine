use super::transport::{Transport, TransportError};
use argus_shared::{EngineMessage, ManagerMessage};

/// D-Bus well-known name for the LLM resource manager.
const MANAGER_DEST: &str = "org.llm.Manager1";
/// D-Bus object path for the LLM resource manager.
const MANAGER_PATH: &str = "/org/llm/Manager1";
/// D-Bus interface for the LLM resource manager.
const MANAGER_IFACE: &str = "org.llm.Manager1";

/// D-Bus transport that connects to `org.llm.Manager1` on the System Bus.
///
/// Receives D-Bus signals and converts them to ManagerMessage directives.
/// Engine→Manager responses are sent as D-Bus method calls (best-effort).
pub struct DbusTransport {
    conn: Option<zbus::blocking::Connection>,
    proxy: Option<zbus::blocking::Proxy<'static>>,
    signals: Option<Box<dyn Iterator<Item = zbus::Message> + Send>>,
}

impl Default for DbusTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl DbusTransport {
    pub fn new() -> Self {
        Self {
            conn: None,
            proxy: None,
            signals: None,
        }
    }
}

impl Transport for DbusTransport {
    fn connect(&mut self) -> Result<(), TransportError> {
        let conn = zbus::blocking::Connection::system()
            .map_err(|e| TransportError::ConnectionFailed(format!("D-Bus system bus: {}", e)))?;

        let proxy = zbus::blocking::Proxy::new(&conn, MANAGER_DEST, MANAGER_PATH, MANAGER_IFACE)
            .map_err(|e| TransportError::ConnectionFailed(format!("D-Bus proxy: {}", e)))?;

        let signals = proxy.receive_all_signals().map_err(|e| {
            TransportError::ConnectionFailed(format!("D-Bus signal iterator: {}", e))
        })?;

        self.conn = Some(conn);
        self.proxy = Some(proxy);
        self.signals = Some(Box::new(signals));

        log::info!("D-Bus transport connected to {}", MANAGER_DEST);
        Ok(())
    }

    fn recv(&mut self) -> Result<ManagerMessage, TransportError> {
        let signals = self
            .signals
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionFailed("not connected".into()))?;

        loop {
            let msg = match signals.next() {
                Some(msg) => msg,
                None => return Err(TransportError::Disconnected),
            };

            let header = msg.header();
            let member = match header.member() {
                Some(m) => m.to_owned(),
                None => continue,
            };

            // The Directive signal is the whole inbound vocabulary. Legacy per-domain
            // signals (MemoryPressure, ComputeGuidance, ThermalAlert, EnergyConstraint)
            // used to be converted into commands here — the engine deciding what to do
            // about resource pressure, which is the Manager's job by construction. They
            // went with `SystemSignal`.
            if member.as_str() != "Directive" {
                log::debug!("ignoring non-Directive D-Bus signal: {}", member);
                continue;
            }
            let body = msg.body();
            let json_str: String = body
                .deserialize()
                .map_err(|e| TransportError::ParseError(format!("Directive body: {}", e)))?;
            let directive: ManagerMessage = serde_json::from_str(&json_str)
                .map_err(|e| TransportError::ParseError(format!("Directive JSON: {}", e)))?;
            return Ok(directive);
        }
    }

    fn send(&mut self, msg: &EngineMessage) -> Result<(), TransportError> {
        let conn = self
            .conn
            .as_ref()
            .ok_or_else(|| TransportError::ConnectionFailed("not connected".into()))?;

        let json = serde_json::to_string(msg)
            .map_err(|e| TransportError::ParseError(format!("serialize: {}", e)))?;

        // Emit as a D-Bus signal (best-effort, Engine→Manager)
        conn.emit_signal(
            Option::<&str>::None,
            MANAGER_PATH,
            MANAGER_IFACE,
            "EngineMessage",
            &(json,),
        )
        .map_err(|e| TransportError::Io(std::io::Error::other(format!("D-Bus emit: {}", e))))?;

        Ok(())
    }

    fn name(&self) -> &str {
        "D-Bus"
    }
}
