//! Manager IPC: the transport, the command executor, and the self-metering the
//! heartbeat carries.
//!
//! The contract types themselves live in `argus-shared` and are re-exported here so
//! engine code has one import path for them.
//!
//! An autonomous `ResilienceManager` + per-domain `ResilienceStrategy` prototype used to
//! sit alongside this, turning `SystemSignal`s into commands inside the engine. It never
//! had a caller outside its own module, and deciding what to do about resource pressure
//! is the Manager's job by construction — the split is the point of the two-process
//! design — so it went with `SystemSignal`.

pub mod executor;
pub mod gpu_self_meter;
pub mod gpu_yield;
pub mod proc_self_meter;
pub mod sys_monitor;
pub mod transport;

#[cfg(feature = "resilience")]
pub mod dbus_transport;

pub use argus_shared::{
    CommandResponse, CommandResult, EngineCommand, EngineDirective, EngineMessage, EngineState,
    EngineStatus, ManagerMessage, Phase,
};
pub use executor::{CommandExecutor, KVSnapshot};
pub use gpu_self_meter::{GpuSelfMeter, NoOpGpuMeter};
#[cfg(unix)]
pub use transport::UnixSocketTransport;
pub use transport::{
    MessageLoop, MockManagerEnd, MockSender, MockTransport, TcpTransport, Transport, TransportError,
};

#[cfg(feature = "resilience")]
pub use dbus_transport::DbusTransport;
