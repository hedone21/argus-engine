//! CUDA driver allocations (L2).
//!
//! - [`buffer`]: `CudaBuffer` (managed), `CudaDeviceBuffer` (device-only),
//!   `CudaHostBuffer` (pinned host) — single file due to shared driver/free
//!   pairing. 분리 검토는 backlog.
//! - [`mmap`]: `CudaMmapRegistration` (cuMemHostRegister owner) +
//!   `CudaMmapAliasBuffer` (typed offset view of the registration).
//!
//! cuda_embedded와 cuda_pc backend가 공유. (D8' 결정: 단일 위치)
pub mod buffer;
#[cfg(feature = "cuda-embedded")]
pub mod mmap;
