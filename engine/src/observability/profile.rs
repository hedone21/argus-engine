//! Inference profiling framework.
//!
//! Houses the per-op timing profiler ([`OpProfiler`]) plus the op-trace and
//! quality-metrics probes wired through the `instrument!` macros.

pub mod op_trace;
pub mod ops;
pub mod quality_metrics;

pub use ops::OpProfiler;
