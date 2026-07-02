//! kivi — KIVI (Q2/Q4/Q8 quantized KV + recent-window F32 residual) GPU attention/flush kernels as a
//! backend-capability (ATTENTION category) plugin.
//!
//! Two disjoint backends, one per GPU axis (mirrors `attn-score`):
//! - [`gpu_kivi`] (`opencl`): the canonical dlopen `.so` path — compiles `kivi_q2.cl`/`kivi_attn.cl`
//!   from the host's borrowed `cl_context`, registers the OpenCL [`QuantAttnBackend`] statically
//!   (`QUANT_ATTN_REGS`) and, under `plugin-cdylib`, as a C-ABI vtable. Selected with
//!   `--load-plugin <kivi>.so --backend-cap kivi_abi`.
//! - [`gpu_kivi_cuda`] (`cuda`): the CUDA twin — compiles `kivi_q2.cu`/`kivi_attn.cu` via system nvcc
//!   and registers a [`CudaQuantAttnBackend`](argus_extension_api::CudaQuantAttnBackend) in the
//!   static-linkme `CUDA_QUANT_ATTN_REGS` slice (no cdylib twin — the CUDA axis is static-linkme
//!   only), so the engine force-links this crate under its own `cuda` feature and selects it with the
//!   same `--backend-cap kivi_abi` key.
//!
//! `opencl` and `cuda` are mutually exclusive in practice (the engine gates one backend at a time),
//! but nothing here forbids both being compiled.

#[cfg(feature = "opencl")]
mod gpu_kivi;
#[cfg(feature = "cuda")]
mod gpu_kivi_cuda;
