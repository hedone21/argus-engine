//! OpenCL cl_mem owners + view adapters (L2).
//!
//! - [`device`]: device-only `OpenCLBuffer` (`clCreateBuffer` with `MEM_READ_WRITE`).
//! - [`unified`]: host-mappable `UnifiedBuffer` (`CL_MEM_ALLOC_HOST_PTR`).
//! - [`sub`]: zero-copy sub-region (`ClSubBuffer`, `clCreateSubBuffer`).
//! - [`noshuffle`]: Q4_0 SOA replacement (`NoshuffleWeightBuffer`).
//!
//! Pool slot view `HostPtrPoolBuffer` lives in `backend/opencl/host_ptr_pool_buffer.rs`
//! (S-D1, 2026-05-24) because it imports `backend::opencl::host_ptr_pool::HostPtrPoolGuard`
//! — keeping it alongside the pool allocator removes the cross-layer import.
//!
//! See ARCHITECTURE.md §13.8-D / proud-strolling-whale plan (B안).
pub mod device;
pub mod noshuffle;
pub mod sub;
pub mod unified;
