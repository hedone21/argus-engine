// ============================================================================
// Memory sub-modules. Moved from `memory/mod.rs` to the top-level
// `engine/src/memory.rs` per Step 4-A (Rust 2018+ pattern: trait definition
// next to physical-resource allocators, no `mod.rs` needed).
//
// Memory layer (L2) — physical memory resources grouped by source:
// - [`galloc`]: generic heap allocator wrapper.
// - [`host`]: host-managed buffers (heap Vec, mmap views).
// - [`opencl`]: OpenCL cl_mem owners (device-only, unified, sub-region, host_ptr_pool, noshuffle SOA).
// - [`cuda`]: CUDA driver allocations (managed/device/pinned + mmap alias).
// - [`rpcmem`]: rpcmem (DMA-BUF heap) view adapters for cross-backend sharing.
// - [`secondary`]: SecondaryMmapBytes + RpcmemRegionGuard traits (Step 3-E, V-09).
//
// See ARCHITECTURE.md §13.8-D and the proud-strolling-whale plan (B안).
// ============================================================================

#[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
pub mod cuda;
pub mod galloc;
pub mod host;
#[cfg(feature = "opencl")]
pub mod opencl;
pub mod rpcmem;
pub mod secondary;

// ============================================================================
// Memory trait (originally `core/memory.rs`).
// ============================================================================

use crate::buffer::{Buffer, DType};
use anyhow::Result;
use std::sync::Arc;

pub trait Memory: Send + Sync {
    /// Allocate a buffer of specific size and type
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>>;

    /// Allocate a KV cache buffer — madvise-capable on GPU (UMA) backends.
    /// Default: delegates to alloc() (CPU backends already return host-managed buffers).
    fn alloc_kv(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        self.alloc(size, dtype)
    }

    /// Total used memory in bytes
    fn used_memory(&self) -> usize;
}
