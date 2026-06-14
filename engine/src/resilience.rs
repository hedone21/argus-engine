pub mod executor;
pub mod gpu_self_meter;
pub mod gpu_yield;
pub mod manager;
pub mod proc_self_meter;
pub mod signal;
pub mod state;
pub mod strategy;
pub mod sys_monitor;
pub mod transport;

#[cfg(feature = "resilience")]
pub mod dbus_transport;

// LAYER-EXEMPT: cross_cutting_trait_usage — §13.8-N §F enum-as-data identifier re-export (V-10)
pub use crate::kv::eviction::EvictMethod;
pub use executor::{CommandExecutor, EvictPlan, ExecutionPlan, KVSnapshot, StreamingParams};
pub use gpu_self_meter::{GpuSelfMeter, NoOpGpuMeter};
pub use manager::ResilienceManager;
pub use signal::{
    CommandResponse, CommandResult, EngineCapability, EngineCommand, EngineDirective,
    EngineMessage, EngineState, EngineStatus, ManagerMessage, ResourceLevel,
};
pub use signal::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};
pub use state::OperatingMode;
pub use strategy::ResilienceStrategy;
#[cfg(unix)]
pub use transport::UnixSocketTransport;
pub use transport::{
    MessageLoop, MockManagerEnd, MockSender, MockTransport, TcpTransport, Transport, TransportError,
};

#[cfg(feature = "resilience")]
pub use dbus_transport::DbusTransport;
