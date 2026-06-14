//! CUDA memory allocator implementing the `Memory` trait.
//!
//! Two modes based on GPU type:
//! - **UMA (Jetson)**: CudaHostBuffer (cuMemHostAlloc+DEVICEMAP) — zero-copy,
//!   CPU and GPU share physical DRAM.
//! - **Discrete GPU**: CudaBuffer (cuMemAllocManaged) — CUDA driver auto-migrates
//!   pages to VRAM on GPU access, giving device-local bandwidth for activations.
//!   CPU access (logit reads, sampling) triggers migration back, but these are
//!   infrequent compared to the per-token GPU compute.

use crate::buffer::{Buffer, DType};
use crate::memory::Memory;
use crate::memory::cuda::buffer::{CudaBuffer, CudaHostBuffer};
use anyhow::Result;
use std::sync::Arc;

/// Memory allocator for the CUDA backend.
pub struct CudaMemory {
    /// If true, use managed memory (discrete GPU). Otherwise pinned host (UMA).
    use_managed: bool,
}

impl Default for CudaMemory {
    fn default() -> Self {
        Self { use_managed: false }
    }
}

impl CudaMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create allocator for discrete GPU — uses managed memory for activations.
    pub fn managed() -> Self {
        Self { use_managed: true }
    }
}

impl Memory for CudaMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        if self.use_managed {
            let buf = CudaBuffer::new(size, dtype)?;
            Ok(Arc::new(buf))
        } else {
            let buf = CudaHostBuffer::new(size, dtype)?;
            Ok(Arc::new(buf))
        }
    }

    fn used_memory(&self) -> usize {
        0
    }
}
