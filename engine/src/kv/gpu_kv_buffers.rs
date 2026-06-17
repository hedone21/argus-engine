//! Engine-owned GPU buffer set for the quantized-recent-window KV cache.
//!
//! `GpuKvBuffers` groups the persistent device buffers that the engine
//! **allocates and owns** for GPU-mode KIVI: the F32 residual buffers, the
//! F16 assembled-attention buffers, and the U8 quantized-block buffers, plus
//! the `Backend`/`Memory` handles used to allocate and write them.
//!
//! Rationale (FORMAT Phase 2, Stage B): consolidating these buffers behind one
//! owned container makes the engine the unambiguous buffer **owner** that lends
//! `cl_mem` to the compute capability. In Stage C the `DynQuantCache` adapter
//! reuses this exact container to marshal named `cl_mem` pointers (mirroring
//! `QuantAttnArgs`) into the cache-construction C-ABI — no generic indexed pool
//! / role→token map is needed because the buffer roles are a fixed set.
//!
//! Field placement: only **buffer ownership** lives here. Compute capability
//! (`kivi: Arc<dyn QuantAttnBackend>`) and write-progress logic state
//! (`gpu_q2k_blocks`/`gpu_q2v_blocks`) stay on `KiviCache` — they are not
//! storage. The CPU-side residual `Vec<f32>` / `SharedBuffer` fields are
//! distinct and also stay on `KiviCache` (GPU mode reuses the CPU residual for
//! the quantize cold path).
//!
//! Borrow note: access buffer fields through **field paths** off a single
//! `slab.as_ref()/as_mut()` binding (e.g. `let s = self.slab.as_mut()?;` then
//! `s.res_k`/`s.res_v`). Rust's disjoint-field borrow then permits borrowing two
//! different buffers simultaneously, exactly as the previous flat fields did.

use crate::backend::Backend;
use crate::memory::Memory;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Engine-owned persistent GPU buffers for GPU-mode KIVI.
///
/// Presence of the enclosing `Option<GpuKvBuffers>` is the GPU-mode sentinel
/// (`is_gpu() == slab.is_some()`), because `backend` is a mandatory non-`Option`
/// field: a slab cannot exist without a backend, and GPU mode cannot exist
/// without a slab. The individual `Option<Tensor>` buffer fields toggle during
/// `transition_bits` (e.g. bits=16 defers attn/Q allocation; 16→Q re-allocates
/// residual) while the slab container itself stays `Some`.
#[derive(Clone)]
pub(crate) struct GpuKvBuffers {
    /// Backend handle for buffer alloc / zero-init / write / kernel dispatch.
    pub(crate) backend: Arc<dyn Backend>,
    /// Allocator used to create the GPU buffers (residual grow / deferred Q alloc).
    pub(crate) memory: Arc<dyn Memory>,
    /// GPU F32 residual K buffer: `[kv_heads, res_cap, head_dim]`.
    pub(crate) res_k: Option<Tensor>,
    /// GPU F32 residual V buffer: `[kv_heads, res_cap, head_dim]`.
    pub(crate) res_v: Option<Tensor>,
    /// GPU F16 attention K output: `[attn_cap, kv_heads, head_dim]`.
    pub(crate) attn_k: Option<Tensor>,
    /// GPU F16 attention V output: `[attn_cap, kv_heads, head_dim]`.
    pub(crate) attn_v: Option<Tensor>,
    /// Allocated capacity of `attn_k`/`attn_v` in tokens (0 if not allocated).
    pub(crate) attn_cap: usize,
    /// GPU U8 byte buffer for quantized K blocks.
    pub(crate) q2k: Option<Tensor>,
    /// GPU U8 byte buffer for quantized V blocks.
    pub(crate) q2v: Option<Tensor>,
}
