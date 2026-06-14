// Re-export signal types from shared crate.
// This keeps all internal `use crate::resilience::signal::*` paths working.
pub use argus_shared::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};

// Re-export new protocol types.
pub use argus_shared::{
    CommandResponse, CommandResult, EngineCapability, EngineCommand, EngineDirective,
    EngineMessage, EngineState, EngineStatus, ManagerMessage, ResourceLevel,
};
