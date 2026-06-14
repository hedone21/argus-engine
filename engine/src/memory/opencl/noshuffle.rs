//! `NoshuffleWeightBuffer` — replaces the original Q4_0 `cl_mem` after SOA
//! conversion so that the AOS allocation can be released.
//!
//! Motivation
//! ----------
//! `convert_q4_0_to_noshuffle()` produces a separate `q_buf` + `d_buf` pair
//! (plus an optional `q_img`). On Adreno/UMA those SOA allocations are fully
//! sufficient for both decode (`matmul_q4_0_noshuffle`) and prefill
//! (`matmul_q4_0_gemm_adreno` via `mul_mat_Ab_Bi_8x4`). The original AOS
//! `cl_mem` is therefore dead weight (~523 MiB across 112 Q4_0 tensors for
//! Llama 3.2 1B) and should be dropped.
//!
//! Design
//! ------
//! After conversion we swap the weight tensor's backing buffer with a
//! `NoshuffleWeightBuffer` that owns the `NoshuffleSoaEntry` and returns the
//! SOA `d_buf` from `cl_mem()`. Two effects:
//!
//! 1. The old `OpenCLBuffer` / `UnifiedBuffer` `Arc` drops to 0 once the
//!    caller hands over the replacement tensor, which triggers
//!    `clReleaseMemObject` on the AOS allocation.
//! 2. `OpenCLBackend::lookup_noshuffle_soa()` keys on `b_buf.as_ptr() as
//!    usize`. Because the swapped buffer returns `d_buf`, the registry key
//!    becomes the SOA `d_buf` pointer — stable as long as the weight tensor
//!    (and therefore this `NoshuffleWeightBuffer`) is alive.
//!
//! The SOA registry still holds a reference to the entry for lookup, but the
//! real owner of the SOA `cl_mem`s is this buffer — see
//! `OpenCLBackend::register_noshuffle_soa_placeholder`.
//!
//! The buffer is tagged `Q4_0` so that `Tensor::dtype()` still reports Q4_0
//! for the weight, keeping `matmul_transposed()` dispatch unchanged.

#[cfg(feature = "opencl")]
use crate::buffer::{Buffer, DType};
#[cfg(feature = "opencl")]
use anyhow::Result;
#[cfg(feature = "opencl")]
use ocl::core::Mem;
#[cfg(feature = "opencl")]
use std::any::Any;

/// Weight buffer that backs the SOA noshuffle layout.
///
/// Keeps the three `cl_mem` handles alive for the lifetime of the weight
/// tensor. `cl_mem()` returns `d_buf`, which doubles as a stable lookup key
/// into the backend's noshuffle SOA registry.
#[cfg(feature = "opencl")]
pub struct NoshuffleWeightBuffer {
    /// SOA nibbles buffer (transposed, ushort-level column-major).
    q_buf: Mem,
    /// SOA scales buffer (transposed, half-level column-major). Also used as
    /// the registry lookup key — its address must outlive any dispatch that
    /// queries the noshuffle SOA registry.
    d_buf: Mem,
    /// image1d_buffer_t wrapping `q_buf` (R32UI). May be `None` when image
    /// creation fails (e.g. driver refuses for that texel count).
    q_img: Option<Mem>,
    /// K dimension (elements per row) of the original Q4_0 weight.
    ne00: usize,
    /// M dimension (number of rows) of the original Q4_0 weight.
    ne01: usize,
    /// Reported byte size — logical AOS size so consumers that inspect
    /// `Tensor::size()` still see the weight's byte footprint. `num_blocks *
    /// 18` (16 nibble bytes + 2 scale bytes per `QK4_0=32` block).
    logical_bytes: usize,
}

#[cfg(feature = "opencl")]
impl NoshuffleWeightBuffer {
    /// Construct a new placeholder. Takes ownership of all three SOA
    /// allocations (`q_buf`, `d_buf`, optional `q_img`).
    ///
    /// `logical_bytes` is the *original* AOS size in bytes (used only for the
    /// `Buffer::size()` report). `ne00` / `ne01` mirror the registry entry.
    pub fn new(
        q_buf: Mem,
        d_buf: Mem,
        q_img: Option<Mem>,
        ne00: usize,
        ne01: usize,
        logical_bytes: usize,
    ) -> Self {
        Self {
            q_buf,
            d_buf,
            q_img,
            ne00,
            ne01,
            logical_bytes,
        }
    }

    pub fn q_buf(&self) -> &Mem {
        &self.q_buf
    }

    pub fn d_buf(&self) -> &Mem {
        &self.d_buf
    }

    pub fn q_img(&self) -> Option<&Mem> {
        self.q_img.as_ref()
    }

    pub fn ne00(&self) -> usize {
        self.ne00
    }

    pub fn ne01(&self) -> usize {
        self.ne01
    }
}

#[cfg(feature = "opencl")]
impl Buffer for NoshuffleWeightBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        // Weight is still Q4_0 logically — the SOA transpose is just an
        // alternate physical layout.
        DType::Q4_0
    }

    fn size(&self) -> usize {
        self.logical_bytes
    }

    fn as_ptr(&self) -> *const u8 {
        // SOA placeholders are GPU-only — no host pointer. `map_weights_for_cpu`
        // must swap the buffer out *before* a placeholder is installed for a
        // weight that needs CPU access.
        std::ptr::null()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }

    fn cl_mem(&self) -> Option<&Mem> {
        // Returns `d_buf` so that `get_cl_mem(buf).as_ptr() as usize` is a
        // stable key into the noshuffle SOA registry. Callers that need
        // `q_buf` or `q_img` must downcast to `NoshuffleWeightBuffer`.
        Some(&self.d_buf)
    }

    fn sync_device(&self) -> Result<()> {
        // All data lives in device memory — nothing to sync.
        Ok(())
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}

// SAFETY: `ocl::core::Mem` is `Send + Sync` (refcounted `cl_mem` handles).
// The SOA buffers are only mutated at construction time; every downstream
// kernel dispatch treats them as read-only.
#[cfg(feature = "opencl")]
unsafe impl Send for NoshuffleWeightBuffer {}
#[cfg(feature = "opencl")]
unsafe impl Sync for NoshuffleWeightBuffer {}

// End-to-end coverage (construction against real SOA buffers, `get_cl_mem`
// returning `d_buf`, and registry-key stability after the original AOS cl_mem
// is dropped) lives in
// `engine/src/backend/opencl/mod.rs::tests::test_noshuffle_weight_buffer_key_stability`,
// which is gated on an available OpenCL device.
