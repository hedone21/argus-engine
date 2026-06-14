//! `RpcmemAliasBuffer` — LISWAP-6 alias `cl_mem` for swapped weight tensors.
//!
//! Wraps an OpenCL `CL_MEM_USE_HOST_PTR` alias whose backing storage is a
//! per-layer rpcmem region owned by `RpcmemSecondaryStore`. The buffer itself
//! does not own the host memory — it pins the layer region
//! (`Arc<RpcmemLayerRegion>`) so the rpcmem allocation is freed only after
//! every alias has been dropped (Drop ordering: cl_mem → Arc<RpcmemLayerRegion>
//! → rpcmem_free).
//!
//! ## Cycle break — Phase 1 alias cache
//!
//! Phase 1 caches `Arc<RpcmemAliasBuffer>` inside `RpcmemSecondaryStore`
//! (eliminating per-swap `clCreateBuffer` overhead). Storing a strong
//! `Arc<SecondaryMmap>` here would close a self-cycle:
//!   `Arc<SecondaryMmap>` → store.alias_cache → `Arc<RpcmemAliasBuffer>` → ...
//! We therefore retain only a `Weak<SecondaryMmap>`. The cache itself lives
//! inside the store, so the entire graph drops together when the model
//! releases its `Arc<SecondaryMmap>`.
//!
//! Spec: ENG-DAT-094, INV-143 (alias lifetime via Arc retention).

use crate::buffer::{Buffer, DType};
use crate::memory::host::mmap::MmapKeepAlive;
use crate::memory::secondary::RpcmemRegionGuard;
use anyhow::Result;
use ocl::core::Mem;
use std::any::Any;
use std::sync::{Arc, Weak};

/// Alias `cl_mem` whose host pointer points into a per-layer rpcmem region.
///
/// **Read-only**: backed by the secondary GGUF mmap copy in rpcmem; weight
/// swap never writes through this alias. `as_mut_ptr()` returns the host
/// pointer (matching `as_ptr()`) only because some downstream paths
/// (`copy_weight_from`) consult `as_mut_ptr()` for the raw address; mutation
/// through it is contractually forbidden.
pub struct RpcmemAliasBuffer {
    /// CL alias handle (CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR).
    cl_buffer: Mem,
    /// Host pointer into the rpcmem region (`region_base + offset`).
    host_ptr: *mut u8,
    /// Tensor byte length.
    size: usize,
    /// Tensor dtype.
    dtype: DType,
    /// Lifetime guard 1 (weak): observe the secondary store without closing
    /// the alias-cache self-cycle (see module docs). The store owns the
    /// layer-region map; if the model has released the secondary the upgrade
    /// fails — which is fine, because the cache is dropping with the store.
    /// Migration Step 3-E (V-09): erased from concrete `Weak<SecondaryMmap>`
    /// to `Weak<dyn MmapKeepAlive>`.
    _secondary_weak: Weak<dyn MmapKeepAlive>,
    /// Lifetime guard 2 (strong): keep this layer's rpcmem region alive even
    /// if the secondary store evicts the entry from its HashMap (defensive —
    /// current implementation never evicts). This is the only ref that *must*
    /// outlive the cl_mem; the rpcmem free runs from this Arc's Drop.
    /// Migration Step 3-E (V-09): erased from concrete `Arc<RpcmemLayerRegion>`
    /// to `Arc<dyn RpcmemRegionGuard>`.
    _layer_region: Arc<dyn RpcmemRegionGuard>,
}

// SAFETY:
// - `cl_buffer` (`ocl::core::Mem`) is internally `Send + Sync` per `ocl` crate.
// - `host_ptr` is read-only; the rpcmem region is single-allocator and pinned
//   for the lifetime of `_layer_region`.
// - The strong `_layer_region` Arc prevents the rpcmem region from being
//   freed before the cl_mem is dropped. The `Weak<SecondaryMmap>` is a
//   non-owning back-reference (no aliasing concern).
unsafe impl Send for RpcmemAliasBuffer {}
unsafe impl Sync for RpcmemAliasBuffer {}

impl RpcmemAliasBuffer {
    /// Construct an alias from an existing `cl_mem` and the lifetime guards.
    ///
    /// Caller (typically the `OpenCLBackend::alloc_alias_weight_buffer`
    /// override) is responsible for creating the `cl_mem` with
    /// `CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY` against the same `host_ptr`.
    ///
    /// `secondary` is consumed as `Arc` and demoted to `Weak` to avoid the
    /// self-cycle introduced by Phase 1 alias caching (the store itself
    /// caches `Arc<RpcmemAliasBuffer>`).
    pub fn new(
        cl_buffer: Mem,
        host_ptr: *mut u8,
        size: usize,
        dtype: DType,
        secondary: Arc<dyn MmapKeepAlive>,
        layer_region: Arc<dyn RpcmemRegionGuard>,
    ) -> Self {
        Self {
            cl_buffer,
            host_ptr,
            size,
            dtype,
            _secondary_weak: Arc::downgrade(&secondary),
            _layer_region: layer_region,
        }
    }
}

impl Buffer for RpcmemAliasBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    fn as_ptr(&self) -> *const u8 {
        self.host_ptr as *const u8
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        // Returned only so size-of-pointer consumers (`copy_weight_from`
        // diagnostics) can identify the buffer; mutation through this
        // pointer is contractually forbidden (the underlying rpcmem region
        // mirrors a read-only mmap snapshot).
        self.host_ptr
    }

    fn cl_mem(&self) -> Option<&Mem> {
        Some(&self.cl_buffer)
    }

    fn sync_device(&self) -> Result<()> {
        // rpcmem DMA-BUF + USE_HOST_PTR alias share the same physical pages
        // on Adreno UMA; no explicit cache flush needed (matches
        // QnnOppkgKvBuffer semantics).
        Ok(())
    }

    fn is_host_managed(&self) -> bool {
        true
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}
