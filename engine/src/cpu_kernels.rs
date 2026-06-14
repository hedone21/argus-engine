//! CPU kernel function pointer set.
//!
//! `Backend::cpu_kernels()` returns a `&'static CpuKernelSet` that exposes the
//! freestanding NEON fused-matmul entry points used by the OpenCL `plan.rs`
//! GPU↔CPU tensor partition fast path and the `forward_gen` CPU FFN fused
//! gate+up dispatch.
//!
//! Introduced by B-5b Phase 2 Stage 1 (infrastructure-only). Stage 2 will
//! replace `crate::backend::cpu::neon::fused_matmul_*` direct call sites with
//! `backend.cpu_kernels()?.fused_matmul_*` (~4 sites in `plan.rs` + 6 sites in
//! `forward_gen.rs`) so the GPU/CUDA backends can route the fused fast path
//! through their `cpu_companion` CPU backend without an L2→L3 downcast.
//!
//! On non-aarch64 targets the freestanding NEON kernels do not exist, so this
//! module is gated to aarch64. The default `Backend::cpu_kernels()` impl
//! returns `None`, and consumers fall back to per-matmul `matmul_transposed`
//! calls (the current pre-Stage-2 behavior).

#[cfg(target_arch = "aarch64")]
use crate::quant::BlockQ4_0;

/// NEON fused-matmul capability bundle.
///
/// Each field is the freestanding `unsafe fn` exported by
/// `engine/src/backend/cpu/neon.rs`. The signatures mirror the current call
/// sites exactly (Stage 1: no signature changes — pure indirection through a
/// function pointer table).
#[cfg(target_arch = "aarch64")]
// fn-pointer fields; the pointer/CPU-feature contract is documented per field above.
#[allow(clippy::missing_safety_doc, clippy::type_complexity)]
pub struct CpuKernelSet {
    /// Fused multi-matmul for F16 decode (M=1): single F32→F16 A conversion
    /// plus a single SpinPool dispatch. `matmuls`: up to 3
    /// `(weight_base: *const u16, out_ptr: *mut f32, n_rows: usize)`.
    pub fused_matmul_f16: unsafe fn(*const f32, usize, &[(*const u16, *mut f32, usize)]),

    /// Fused multi-matmul for Q4_0 decode (M=1): single Q8_0 quantization plus
    /// a single Rayon dispatch. `matmuls`: up to 3
    /// `(weight_base: *const BlockQ4_0, out_ptr: *mut f32, n_rows: usize)`.
    pub fused_matmul_q4_0: unsafe fn(*const f32, usize, &[(*const BlockQ4_0, *mut f32, usize)]),
}

/// Static `CpuKernelSet` returned by `CpuBackend::cpu_kernels()` on aarch64.
/// `'static` lifetime lets the trait method return a borrow without per-call
/// allocation.
#[cfg(target_arch = "aarch64")]
pub static CPU_KERNEL_SET: CpuKernelSet = CpuKernelSet {
    fused_matmul_f16: crate::backend::cpu::neon::fused_matmul_f16,
    fused_matmul_q4_0: crate::backend::cpu::neon::fused_matmul_q4_0,
};

/// Placeholder `CpuKernelSet` for non-aarch64 targets. Exposed so trait
/// signatures and consumer code can name the type unconditionally; the
/// default `Backend::cpu_kernels()` impl returns `None` on these targets.
#[cfg(not(target_arch = "aarch64"))]
pub struct CpuKernelSet {
    _private: (),
}
