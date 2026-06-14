pub mod common;
pub use common::CpuBackendCommon;

#[cfg(target_arch = "x86_64")]
pub mod x86;
#[cfg(target_arch = "x86_64")]
pub use x86::CpuBackendAVX2;

#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "aarch64")]
pub use neon::CpuBackendNeon;

#[cfg(target_arch = "x86_64")]
pub type CpuBackend = CpuBackendAVX2;

#[cfg(target_arch = "aarch64")]
pub type CpuBackend = CpuBackendNeon;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub type CpuBackend = CpuBackendCommon;

/// Process-wide shared `Arc<dyn Backend>` to the CPU backend.
///
/// Used by GPU backends as the `cpu_companion` field at bootstrap so each
/// new GPU backend instance doesn't allocate its own `CpuBackend`. Feature
/// detection (NEON/AVX2 etc.) runs once at first call. S-2 sprint
/// 2026-05-24 — baseline neutral but removes per-bootstrap allocation +
/// makes the shared-companion intent explicit at the call site.
pub fn cpu_singleton() -> std::sync::Arc<dyn crate::backend::Backend> {
    use std::sync::{Arc, OnceLock};
    static SINGLETON: OnceLock<Arc<dyn crate::backend::Backend>> = OnceLock::new();
    SINGLETON
        .get_or_init(|| Arc::new(CpuBackend::new()))
        .clone()
}
