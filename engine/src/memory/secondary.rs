//! Memory-layer abstractions for secondary weight stores.
//!
//! Bridges L2 (memory adapters: `CudaMmapRegistration`, `HostPtrPoolBuffer`,
//! `RpcmemAliasBuffer`) and the L3 secondary stores (`SecondaryMmap`,
//! `RpcmemLayerRegion`) without forcing L2 to import L3 concrete types.
//!
//! Resolves INV-LAYER-002 violations (Migration Step 3-E, V-09):
//! - `memory/cuda/mmap.rs`     — `Arc<SecondaryMmap>`        → `Arc<dyn SecondaryMmapBytes>`
//! - `memory/opencl/host_ptr_pool_buffer.rs` — `Option<Arc<SecondaryMmap>>` → `Option<Arc<dyn MmapKeepAlive>>`
//! - `memory/rpcmem/opencl_alias.rs` — `Weak<SecondaryMmap>` + `Arc<RpcmemLayerRegion>`
//!   → `Weak<dyn MmapKeepAlive>` + `Arc<dyn RpcmemRegionGuard>`
//!
//! Two responsibility tiers:
//! 1. **Bytes access** (`SecondaryMmapBytes`): one call site (`memory/cuda/mmap.rs`)
//!    needs the raw byte span for `cuMemHostRegister`. The trait method
//!    encapsulates the enum dispatch and Rpcmem-variant rejection inside the
//!    L3 impl so L2 stays agnostic.
//! 2. **Lifetime guard** (re-use of `MmapKeepAlive` + new `RpcmemRegionGuard`):
//!    the four remaining sites only hold an `Arc`/`Weak` for drop ordering
//!    (INV-143) and never invoke methods. Marker-trait erasure suffices.

use crate::memory::host::mmap::MmapKeepAlive;
use anyhow::Result;

/// Bytes-level access to a secondary weight store.
///
/// Used by `memory/cuda/mmap.rs::CudaMmapRegistration::register` to obtain a
/// contiguous host-memory byte span suitable for `cuMemHostRegister`. The
/// super-trait `MmapKeepAlive` lets implementors double as the INV-143 mmap
/// lifetime guard.
///
/// Implementors must return `Err` for variants that do not back a single
/// contiguous mmap region (e.g. `SecondaryMmap::Rpcmem`, which is a per-layer
/// rpcmem heap allocator rather than a single mmap).
pub trait SecondaryMmapBytes: MmapKeepAlive {
    fn raw_bytes(&self) -> Result<&[u8]>;
}

/// Marker trait for per-layer rpcmem region holders.
///
/// `RpcmemAliasBuffer` keeps an `Arc<dyn RpcmemRegionGuard>` solely so that
/// the rpcmem allocation (and its `rpcmem_free` Drop) outlives the cl_mem
/// alias. No methods are invoked — pure drop-ordering anchor.
///
/// Implemented by `models::weights::rpcmem_secondary::RpcmemLayerRegion`.
pub trait RpcmemRegionGuard: Send + Sync + 'static {}
