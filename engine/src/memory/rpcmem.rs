//! rpcmem (DMA-BUF heap) view adapters (L2).
//!
//! - [`allocator`]: `RpcmemAllocator` — backend-agnostic dlopen wrapper for
//!   `libcdsprpc.so` (Sprint 2a Phase 2, ENG-RPCMEM-010 ~ C04).
//! - [`kv_buffer`]: `RpcmemKvBuffer` — rpcmem + OpenCL USE_HOST_PTR alias for
//!   KV cache (consumer of `RpcmemAllocator`).
//! - [`opencl_alias`]: `RpcmemAliasBuffer` — rpcmem fd → cl_mem alias
//!   (CL_MEM_USE_HOST_PTR) for the OpenCL backend (precision swap path).
//!
//! `RpcmemLayerRegion` (fd allocator/lifecycle owner) stays in
//! `models/weights/rpcmem_secondary.rs` because it is structurally bound to
//! `RpcmemSecondaryStore` (D5'). The QNN consumer (`QnnBuffer::Rpcmem`) stays
//! in `backend/qnn_oppkg/` as a backend-internal view (Sprint 2b cleanup).
pub mod allocator;
#[cfg(feature = "opencl")]
pub mod kv_buffer;
#[cfg(feature = "opencl")]
pub mod opencl_alias;
