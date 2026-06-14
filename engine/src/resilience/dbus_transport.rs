use super::signal::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};
use super::transport::{Transport, TransportError};
use argus_shared::{EngineCommand, EngineDirective, EngineMessage, ManagerMessage};

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
    next_seq_id: u64,
}

impl DbusTransport {
    pub fn new() -> Self {
        Self {
            conn: None,
            proxy: None,
            signals: None,
            next_seq_id: 1,
        }
    }

    /// Convert a legacy SystemSignal to the new ManagerMessage format.
    fn signal_to_manager_message(&mut self, signal: SystemSignal) -> ManagerMessage {
        let commands = match signal {
            SystemSignal::MemoryPressure { level, .. } => match level {
                Level::Normal => vec![EngineCommand::RestoreDefaults],
                Level::Warning => vec![EngineCommand::KvEvictSliding { keep_ratio: 0.85 }],
                Level::Critical => vec![EngineCommand::KvEvictH2o { keep_ratio: 0.50 }],
                Level::Emergency => {
                    return ManagerMessage::Directive(EngineDirective {
                        seq_id: self.next_seq(),
                        commands: vec![EngineCommand::Suspend],
                    });
                }
            },
            SystemSignal::ComputeGuidance {
                level,
                recommended_backend,
                ..
            } => {
                let device = match recommended_backend {
                    RecommendedBackend::Cpu => "cpu",
                    RecommendedBackend::Gpu => "gpu",
                    RecommendedBackend::Any => "any",
                };
                match level {
                    Level::Normal => vec![EngineCommand::RestoreDefaults],
                    Level::Warning => vec![
                        EngineCommand::Throttle { delay_ms: 30 },
                        EngineCommand::SwitchHw {
                            device: device.to_string(),
                        },
                    ],
                    Level::Critical => vec![
                        EngineCommand::Throttle { delay_ms: 70 },
                        EngineCommand::SwitchHw {
                            device: device.to_string(),
                        },
                    ],
                    Level::Emergency => {
                        return ManagerMessage::Directive(EngineDirective {
                            seq_id: self.next_seq(),
                            commands: vec![EngineCommand::Suspend],
                        });
                    }
                }
            }
            SystemSignal::ThermalAlert { level, .. } => match level {
                Level::Normal => vec![EngineCommand::RestoreDefaults],
                Level::Warning => vec![
                    EngineCommand::Throttle { delay_ms: 30 },
                    EngineCommand::PrepareComputeUnit {
                        device: "cpu".to_string(),
                    },
                ],
                Level::Critical => vec![
                    EngineCommand::Throttle { delay_ms: 70 },
                    EngineCommand::SwitchHw {
                        device: "cpu".to_string(),
                    },
                ],
                Level::Emergency => vec![EngineCommand::Suspend],
            },
            SystemSignal::EnergyConstraint { level, .. } => match level {
                Level::Normal => vec![EngineCommand::RestoreDefaults],
                Level::Warning => vec![EngineCommand::SwitchHw {
                    device: "cpu".to_string(),
                }],
                Level::Critical => vec![
                    EngineCommand::SwitchHw {
                        device: "cpu".to_string(),
                    },
                    EngineCommand::Throttle { delay_ms: 70 },
                ],
                Level::Emergency => vec![EngineCommand::Suspend],
            },
        };

        ManagerMessage::Directive(EngineDirective {
            seq_id: self.next_seq(),
            commands,
        })
    }

    fn next_seq(&mut self) -> u64 {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        id
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

            // Try the new JSON-based Directive signal first
            if member.as_str() == "Directive" {
                let body = msg.body();
                let json_str: String = body
                    .deserialize()
                    .map_err(|e| TransportError::ParseError(format!("Directive body: {}", e)))?;
                let directive: ManagerMessage = serde_json::from_str(&json_str)
                    .map_err(|e| TransportError::ParseError(format!("Directive JSON: {}", e)))?;
                return Ok(directive);
            }

            // Legacy signal conversion
            let result = match member.as_str() {
                "MemoryPressure" => parse_memory_pressure(&msg),
                "ComputeGuidance" => parse_compute_guidance(&msg),
                "ThermalAlert" => parse_thermal_alert(&msg),
                "EnergyConstraint" => parse_energy_constraint(&msg),
                _ => {
                    log::debug!("Unknown D-Bus signal: {}", member);
                    continue;
                }
            };

            let signal =
                result.map_err(|e| TransportError::ParseError(format!("{}: {}", member, e)))?;

            return Ok(self.signal_to_manager_message(signal));
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

/// Parse MemoryPressure signal: (level: s, available_bytes: t, total_bytes: t, reclaim_target_bytes: t)
fn parse_memory_pressure(msg: &zbus::Message) -> anyhow::Result<SystemSignal> {
    let body = msg.body();
    let (level_str, available_bytes, total_bytes, reclaim_target_bytes): (String, u64, u64, u64) =
        body.deserialize()?;

    let level = Level::from_dbus_str(&level_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid level: {}", level_str))?;

    Ok(SystemSignal::MemoryPressure {
        level,
        available_bytes,
        total_bytes,
        reclaim_target_bytes,
    })
}

/// Parse ComputeGuidance signal: (level: s, recommended_backend: s, reason: s, cpu_usage_pct: d, gpu_usage_pct: d)
fn parse_compute_guidance(msg: &zbus::Message) -> anyhow::Result<SystemSignal> {
    let body = msg.body();
    let (level_str, backend_str, reason_str, cpu_usage_pct, gpu_usage_pct): (
        String,
        String,
        String,
        f64,
        f64,
    ) = body.deserialize()?;

    let level = Level::from_dbus_str(&level_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid level: {}", level_str))?;
    let recommended_backend = RecommendedBackend::from_dbus_str(&backend_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid backend: {}", backend_str))?;
    let reason = ComputeReason::from_dbus_str(&reason_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid reason: {}", reason_str))?;

    Ok(SystemSignal::ComputeGuidance {
        level,
        recommended_backend,
        reason,
        cpu_usage_pct,
        gpu_usage_pct,
    })
}

/// Parse ThermalAlert signal: (level: s, temperature_mc: i, throttling_active: b, throttle_ratio: d)
fn parse_thermal_alert(msg: &zbus::Message) -> anyhow::Result<SystemSignal> {
    let body = msg.body();
    let (level_str, temperature_mc, throttling_active, throttle_ratio): (String, i32, bool, f64) =
        body.deserialize()?;

    let level = Level::from_dbus_str(&level_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid level: {}", level_str))?;

    Ok(SystemSignal::ThermalAlert {
        level,
        temperature_mc,
        throttling_active,
        throttle_ratio,
    })
}

/// Parse EnergyConstraint signal: (level: s, reason: s, power_budget_mw: u)
fn parse_energy_constraint(msg: &zbus::Message) -> anyhow::Result<SystemSignal> {
    let body = msg.body();
    let (level_str, reason_str, power_budget_mw): (String, String, u32) = body.deserialize()?;

    let level = Level::from_dbus_str(&level_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid level: {}", level_str))?;
    let reason = EnergyReason::from_dbus_str(&reason_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid reason: {}", reason_str))?;

    Ok(SystemSignal::EnergyConstraint {
        level,
        reason,
        power_budget_mw,
    })
}
