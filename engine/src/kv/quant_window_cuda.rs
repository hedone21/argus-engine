//! CUDA marshalling for the KIVI quant-window native path (P4b engine dispatch).
//!
//! All the `CUdeviceptr`/`CUstream` extraction + `CudaQuantAttn*Args` packing lives here so the
//! `quant_window_cache` / `quant_window_format` call sites stay a one-line `#[cfg(cuda)]` branch —
//! the CUDA twin of the inline OpenCL `get_cl_mem` + `QuantAttn*Args` blocks. The engine owns every
//! buffer (backend-generic `Tensor`s); these helpers only lend the raw device pointers + the engine's
//! default stream to the plugin's `CudaQuantAttnBackend`, which runs the verified `kivi_*.cu` kernels.

use anyhow::{Result, anyhow};
use argus_extension_api::{
    CudaQuantAttnArgs, CudaQuantAttnBackend, CudaQuantDequantFlushArgs,
    CudaQuantScatterResidualArgs,
};

use crate::backend::Backend;
use crate::backend::cuda_pc::CudaBackend;
use crate::buffer::Buffer;
use crate::tensor::Tensor;

/// Extract the raw `CUdeviceptr` (u64) from a CUDA buffer (managed / pinned-host / pure-device).
/// Mirror of `CudaBackend::get_device_ptr`; the three engine CUDA buffer types all expose `device_ptr`.
fn cu_devptr(buf: &dyn Buffer) -> Result<u64> {
    use crate::memory::cuda::buffer::{CudaBuffer, CudaDeviceBuffer, CudaHostBuffer};
    let any = buf.as_any();
    if let Some(b) = any.downcast_ref::<CudaHostBuffer>() {
        Ok(b.device_ptr())
    } else if let Some(b) = any.downcast_ref::<CudaBuffer>() {
        Ok(b.device_ptr())
    } else if let Some(b) = any.downcast_ref::<CudaDeviceBuffer>() {
        Ok(b.device_ptr())
    } else {
        Err(anyhow!(
            "quant-window CUDA: buffer is not a CUDA device buffer"
        ))
    }
}

/// The engine's default `CUstream` (null == legacy default stream) as a `*mut c_void` for the plugin.
/// The KIVI kernels + any readback share this stream, preserving in-order semantics (mirror of the
/// OpenCL path lending its single in-order `cl_command_queue`).
fn cu_stream(backend: &dyn Backend) -> Result<*mut std::ffi::c_void> {
    let cuda_be = backend
        .as_any()
        .downcast_ref::<CudaBackend>()
        .ok_or_else(|| anyhow!("quant-window CUDA: backend is not CudaBackend"))?;
    Ok(cuda_be.context().default_stream().cu_stream() as *mut std::ffi::c_void)
}

/// Q2 dequant-flush of one K or V flush window into the F16 attention view (mirror of the OpenCL
/// `dequant_flush` dispatch in `flush_residual_gpu`). Only Q2 has a GPU flush kernel; Q4/Q8 use the
/// generic CPU-dequant path (unchanged, shared).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_flush_q2(
    cap: &dyn CudaQuantAttnBackend,
    backend: &dyn Backend,
    q_blocks: &Tensor,
    attn: &Tensor,
    kv_heads: usize,
    head_dim: usize,
    n_groups_or_tokens: usize,
    tok_base: usize,
    block_start: usize,
    is_key: bool,
) -> Result<()> {
    let args = CudaQuantDequantFlushArgs {
        cu_stream: cu_stream(backend)?,
        q_blocks_mem: cu_devptr(q_blocks.buffer().as_ref())?,
        attn_mem: cu_devptr(attn.buffer().as_ref())?,
        kv_heads,
        head_dim,
        n_groups_or_tokens,
        tok_base,
        block_start,
        bits: 2,
        is_key,
    };
    let rc = cap.dequant_flush(&args);
    if rc != 0 {
        anyhow::bail!(
            "quant-window CUDA dequant_flush ({}) failed (rc={rc})",
            if is_key { "K" } else { "V" }
        );
    }
    Ok(())
}

/// Scatter the F32 residual ring into the F16 attention view (mirror of the OpenCL `scatter_residual`
/// dispatch in `assemble_view_gpu`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_residual(
    cap: &dyn CudaQuantAttnBackend,
    backend: &dyn Backend,
    res: &Tensor,
    attn: &Tensor,
    kv_heads: usize,
    res_cap: usize,
    head_dim: usize,
    res_pos: usize,
    tok_base: usize,
    is_key: bool,
) -> Result<()> {
    let args = CudaQuantScatterResidualArgs {
        cu_stream: cu_stream(backend)?,
        res_mem: cu_devptr(res.buffer().as_ref())?,
        attn_mem: cu_devptr(attn.buffer().as_ref())?,
        kv_heads,
        res_cap,
        head_dim,
        res_pos,
        tok_base,
    };
    let rc = cap.scatter_residual(&args);
    if rc != 0 {
        anyhow::bail!(
            "quant-window CUDA scatter_residual ({}) failed (rc={rc})",
            if is_key { "K" } else { "V" }
        );
    }
    Ok(())
}

/// Fused native decode attention over quantized blocks + F32 residual (mirror of the OpenCL
/// `attention_gen_quant` dispatch in `attention_native`). `scores_out` is a host pointer (may be null).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    cap: &dyn CudaQuantAttnBackend,
    backend: &dyn Backend,
    q: &Tensor,
    qk: &Tensor,
    qv: &Tensor,
    res_k: &Tensor,
    res_v: &Tensor,
    out: &Tensor,
    scores_out: *mut f32,
    scores_len: usize,
    num_heads_q: usize,
    num_heads_kv: usize,
    head_dim: usize,
    q_tokens: usize,
    res_tokens: usize,
    res_cap: usize,
    scale: f32,
    bits: u8,
) -> Result<()> {
    let args = CudaQuantAttnArgs {
        cu_stream: cu_stream(backend)?,
        q_mem: cu_devptr(q.buffer().as_ref())?,
        qk_mem: cu_devptr(qk.buffer().as_ref())?,
        qv_mem: cu_devptr(qv.buffer().as_ref())?,
        res_k_mem: cu_devptr(res_k.buffer().as_ref())?,
        res_v_mem: cu_devptr(res_v.buffer().as_ref())?,
        out_mem: cu_devptr(out.buffer().as_ref())?,
        scores_out,
        scores_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        q_tokens,
        res_tokens,
        res_cap,
        scale,
        bits,
    };
    let rc = cap.attention_gen_quant(&args);
    if rc != 0 {
        anyhow::bail!("quant-window CUDA attention_gen_quant failed (rc={rc})");
    }
    Ok(())
}
