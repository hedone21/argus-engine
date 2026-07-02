//! CUDA half of the KIVI ATTENTION backend-cap — the [`CudaQuantAttnBackend`] that owns
//! `kivi_q2.cu` + `kivi_attn.cu`.
//!
//! The CUDA twin of the OpenCL [`KiviGpuBackend`](crate) (lib.rs), built on the same skeleton as the
//! already-done CUDA score reducer (`attn-score/src/gpu_reduce_cuda.rs`): it compiles both `.cu`
//! sources from the host's borrowed `CUcontext` at `make()` time (system `nvcc` → PTX, for
//! PTX-version parity with the engine's own kernels — NOT NVRTC), loads them with cudarc's low-level
//! module API, and dispatches each kernel on the engine's lent `CUstream`. The engine owns every KV
//! device buffer and lends raw `CUdeviceptr`s; this crate allocates only a transient per-call score
//! buffer (freed before return). Registered into `CUDA_QUANT_ATTN_REGS` (disjoint from the OpenCL
//! `QUANT_ATTN_REGS`) under the same name `"kivi_abi"`. Static-linkme only — no cdylib vtable twin.

use std::ffi::{CString, c_void};
use std::mem::ManuallyDrop;
use std::sync::Arc;

use argus_extension_api::{
    CUDA_QUANT_ATTN_REGS, CudaQuantAttnArgs, CudaQuantAttnBackend, CudaQuantAttnGatherArgs,
    CudaQuantAttnMakeArgs, CudaQuantAttnReg, CudaQuantDequantFlushArgs,
    CudaQuantScatterResidualArgs,
};
use cudarc::driver::{CudaContext, result as cuda_result, sys as cuda_sys};
use linkme::distributed_slice;

const KIVI_Q2_SRC: &str = include_str!("kivi_q2.cu");
const KIVI_ATTN_SRC: &str = include_str!("kivi_attn.cu");

/// LOCAL_SIZE in kivi_attn.cu — one block per Q head, this many threads, this much shared scratch.
const ATTN_BLOCK: u32 = 64;
/// Flat-kernel launch block size (kivi_q2.cu 1-thread-per-element kernels).
const FLAT_BLOCK: u32 = 256;

struct KiviCudaBackend {
    /// Borrowed engine context (ManuallyDrop → never destroy it; `from_raw_context` sets
    /// `is_primary=false`, whose Drop would `cuCtxDestroy`). CUDA analog of the OpenCL reducer.
    _ctx: ManuallyDrop<Arc<CudaContext>>,
    module_q2: cuda_sys::CUmodule,
    module_attn: cuda_sys::CUmodule,
    // 7 live kernels (None if get_function failed → the op returns -1 and has_quant_attn_kernel
    // reports false, so the engine gate falls back to the F32 dequant path — mirrors the OpenCL impl).
    deq_value_q2_f16: Option<cuda_sys::CUfunction>,
    deq_key_q2_f16: Option<cuda_sys::CUfunction>,
    scatter_residual_f16: Option<cuda_sys::CUfunction>,
    gather_update: Option<cuda_sys::CUfunction>,
    attn_q2: Option<cuda_sys::CUfunction>,
    attn_q4: Option<cuda_sys::CUfunction>,
    attn_q8: Option<cuda_sys::CUfunction>,
    is_nosub: bool,
}

// SAFETY: raw CUDA handles are only touched on the single inference thread (same assumption as the
// OpenCL KIVI backend + the CUDA score reducer). The handles belong to the borrowed host context.
unsafe impl Send for KiviCudaBackend {}
unsafe impl Sync for KiviCudaBackend {}

impl Drop for KiviCudaBackend {
    fn drop(&mut self) {
        // Unload only our own modules; the borrowed context is intentionally leaked (ManuallyDrop).
        unsafe {
            let _ = cuda_result::module::unload(self.module_q2);
            let _ = cuda_result::module::unload(self.module_attn);
        }
    }
}

impl CudaQuantAttnBackend for KiviCudaBackend {
    fn has_quant_attn_kernel(&self, bits: u8) -> bool {
        match bits {
            2 => self.attn_q2.is_some(),
            4 => self.attn_q4.is_some(),
            8 => self.attn_q8.is_some(),
            _ => false,
        }
    }

    fn is_nosub_device(&self) -> bool {
        self.is_nosub
    }

    fn attention_gen_quant(&self, args: &CudaQuantAttnArgs) -> i32 {
        match self.attention_inner(args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[kivi cuda] attention_gen_quant failed: {e}");
                -1
            }
        }
    }

    fn gather_update_quant(&self, args: &CudaQuantAttnGatherArgs) -> i32 {
        match self.gather_inner(args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[kivi cuda] gather_update_quant failed: {e}");
                -1
            }
        }
    }

    fn dequant_flush(&self, args: &CudaQuantDequantFlushArgs) -> i32 {
        match self.dequant_flush_inner(args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[kivi cuda] dequant_flush failed: {e}");
                -1
            }
        }
    }

    fn scatter_residual(&self, args: &CudaQuantScatterResidualArgs) -> i32 {
        match self.scatter_inner(args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[kivi cuda] scatter_residual failed: {e}");
                -1
            }
        }
    }
}

impl KiviCudaBackend {
    /// Fused dequant+attention. Mirrors the OpenCL `attention_gen` (lib.rs:297): pick the Q2/Q4/Q8
    /// kernel, launch one block per Q head (64 threads, 256B dynamic shared scratch, 16 params — the
    /// OpenCL `local float* scratch` trailing arg became `extern __shared__`), then read post-softmax
    /// scores back to the host slice if requested.
    fn attention_inner(&self, args: &CudaQuantAttnArgs) -> Result<(), String> {
        let kernel = match args.bits {
            2 => self.attn_q2.ok_or("Q2 attention kernel N/A")?,
            4 => self.attn_q4.ok_or("Q4 attention kernel N/A")?,
            8 => self.attn_q8.ok_or("Q8 attention kernel N/A")?,
            b => return Err(format!("unsupported KIVI bits: {b}")),
        };
        if args.num_heads_q == 0 || args.num_heads_kv == 0 {
            return Err("num_heads_q/kv must be > 0".to_string());
        }
        let stream = args.cu_stream as cuda_sys::CUstream;

        let want_scores = !args.scores_out.is_null() && args.scores_len > 0;
        let has_scores: i32 = want_scores as i32;
        let total_tokens = args.q_tokens + args.res_tokens;
        let score_stride_val: i32 = if want_scores {
            (args.scores_len / args.num_heads_q) as i32
        } else {
            total_tokens as i32
        };

        // Device score buffer (always ≥1 elem so the kernel's S arg is a valid pointer even when
        // has_scores=0, where the kernel never writes it). Freed on every path before return.
        let score_elems = if want_scores {
            args.num_heads_q * score_stride_val.max(0) as usize
        } else {
            1
        }
        .max(1);
        let score_dptr = unsafe { cuda_result::malloc_sync(score_elems * 4) }
            .map_err(|e| format!("malloc score buffer: {e}"))?;

        let run = || -> Result<(), String> {
            // Named stack locals: kernel_params holds pointers to each; the values are copied out at
            // enqueue. Order = kivi_attn.cu signature (16 params; no trailing scratch arg).
            let q_mem = args.q_mem;
            let qk_mem = args.qk_mem;
            let qv_mem = args.qv_mem;
            let res_k_mem = args.res_k_mem;
            let res_v_mem = args.res_v_mem;
            let out_mem = args.out_mem;
            let s_mem = score_dptr;
            let nhq = args.num_heads_q as i32;
            let nhkv = args.num_heads_kv as i32;
            let hd = args.head_dim as i32;
            let qt = args.q_tokens as i32;
            let rt = args.res_tokens as i32;
            let rc = args.res_cap as i32;
            let scale = args.scale;
            let score_stride = score_stride_val;
            let has = has_scores;

            let mut params: [*mut c_void; 16] = [
                &q_mem as *const _ as *mut c_void,
                &qk_mem as *const _ as *mut c_void,
                &qv_mem as *const _ as *mut c_void,
                &res_k_mem as *const _ as *mut c_void,
                &res_v_mem as *const _ as *mut c_void,
                &out_mem as *const _ as *mut c_void,
                &s_mem as *const _ as *mut c_void,
                &nhq as *const _ as *mut c_void,
                &nhkv as *const _ as *mut c_void,
                &hd as *const _ as *mut c_void,
                &qt as *const _ as *mut c_void,
                &rt as *const _ as *mut c_void,
                &rc as *const _ as *mut c_void,
                &scale as *const _ as *mut c_void,
                &score_stride as *const _ as *mut c_void,
                &has as *const _ as *mut c_void,
            ];

            let grid = (args.num_heads_q as u32, 1u32, 1u32);
            let block = (ATTN_BLOCK, 1u32, 1u32);
            let shared = ATTN_BLOCK * 4; // 64 floats of scratch
            // SAFETY: engine-owned device buffers live for the call; params point at stack locals
            // that outlive the synchronous enqueue.
            unsafe {
                cuda_result::launch_kernel(kernel, grid, block, shared, stream, &mut params)
                    .map_err(|e| format!("launch attn: {e}"))?;
            }
            if want_scores {
                // Blocking DtoH into the host slice (engine lends the default/null stream, so this is
                // ordered after the launch — mirror of the OpenCL blocking readback on the same queue).
                let scores =
                    unsafe { std::slice::from_raw_parts_mut(args.scores_out, args.scores_len) };
                unsafe { cuda_result::memcpy_dtoh_sync(scores, score_dptr) }
                    .map_err(|e| format!("score DtoH: {e}"))?;
            }
            Ok(())
        };

        let result = run();
        // Always free the transient score buffer.
        unsafe {
            let _ = cuda_result::free_sync(score_dptr);
        }
        result
    }

    /// Residual gather-update (lib.rs:174). One thread per (seq, head, dim) element.
    fn gather_inner(&self, args: &CudaQuantAttnGatherArgs) -> Result<(), String> {
        let kernel = self.gather_update.ok_or("gather_update kernel N/A")?;
        let stream = args.cu_stream as cuda_sys::CUstream;
        let total = args.seq_len * args.kv_heads * args.head_dim;

        let input_mem = args.input_mem;
        let residual_mem = args.residual_mem;
        let kv_heads = args.kv_heads as i32;
        let res_cap = args.res_cap as i32;
        let head_dim = args.head_dim as i32;
        let seq_len = args.seq_len as i32;
        let res_pos = args.res_pos as i32;
        let mut params: [*mut c_void; 7] = [
            &input_mem as *const _ as *mut c_void,
            &residual_mem as *const _ as *mut c_void,
            &kv_heads as *const _ as *mut c_void,
            &res_cap as *const _ as *mut c_void,
            &head_dim as *const _ as *mut c_void,
            &seq_len as *const _ as *mut c_void,
            &res_pos as *const _ as *mut c_void,
        ];
        launch_flat(kernel, total, stream, &mut params).map_err(|e| format!("launch gather: {e}"))
    }

    /// Q2 dequant-flush into the F16 view (lib.rs:209). Q2 only (Q4/Q8 flush is CPU in the engine).
    fn dequant_flush_inner(&self, args: &CudaQuantDequantFlushArgs) -> Result<(), String> {
        if args.bits != 2 {
            return Err(format!("dequant_flush unsupported for bits={}", args.bits));
        }
        let (kernel, total) = if args.is_key {
            (
                self.deq_key_q2_f16.ok_or("deq_key_q2_f16 kernel N/A")?,
                args.kv_heads * args.n_groups_or_tokens * args.head_dim,
            )
        } else {
            (
                self.deq_value_q2_f16.ok_or("deq_value_q2_f16 kernel N/A")?,
                args.kv_heads * args.n_groups_or_tokens * (args.head_dim / 32),
            )
        };
        let stream = args.cu_stream as cuda_sys::CUstream;
        let q_blocks_mem = args.q_blocks_mem;
        let attn_mem = args.attn_mem;
        let kv_heads = args.kv_heads as i32;
        let head_dim = args.head_dim as i32;
        let n = args.n_groups_or_tokens as i32;
        let tok_base = args.tok_base as i32;
        let block_start = args.block_start as i32;
        let mut params: [*mut c_void; 7] = [
            &q_blocks_mem as *const _ as *mut c_void,
            &attn_mem as *const _ as *mut c_void,
            &kv_heads as *const _ as *mut c_void,
            &head_dim as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
            &tok_base as *const _ as *mut c_void,
            &block_start as *const _ as *mut c_void,
        ];
        launch_flat(kernel, total, stream, &mut params).map_err(|e| format!("launch flush: {e}"))
    }

    /// Residual scatter into the F16 view (lib.rs:260). One thread per (head, token, dim) element.
    fn scatter_inner(&self, args: &CudaQuantScatterResidualArgs) -> Result<(), String> {
        let kernel = self
            .scatter_residual_f16
            .ok_or("scatter_residual_f16 kernel N/A")?;
        let stream = args.cu_stream as cuda_sys::CUstream;
        let total = args.kv_heads * args.res_pos * args.head_dim;
        let res_mem = args.res_mem;
        let attn_mem = args.attn_mem;
        let kv_heads = args.kv_heads as i32;
        let res_cap = args.res_cap as i32;
        let head_dim = args.head_dim as i32;
        let res_pos = args.res_pos as i32;
        let tok_base = args.tok_base as i32;
        let mut params: [*mut c_void; 7] = [
            &res_mem as *const _ as *mut c_void,
            &attn_mem as *const _ as *mut c_void,
            &kv_heads as *const _ as *mut c_void,
            &res_cap as *const _ as *mut c_void,
            &head_dim as *const _ as *mut c_void,
            &res_pos as *const _ as *mut c_void,
            &tok_base as *const _ as *mut c_void,
        ];
        launch_flat(kernel, total, stream, &mut params).map_err(|e| format!("launch scatter: {e}"))
    }
}

/// Launch a 1-thread-per-element flat kernel (`total` work items, `FLAT_BLOCK`-wide grid, no shared).
/// `total == 0` is a no-op (the kernels also early-return, but skip the empty launch entirely).
fn launch_flat(
    kernel: cuda_sys::CUfunction,
    total: usize,
    stream: cuda_sys::CUstream,
    params: &mut [*mut c_void],
) -> Result<(), cuda_result::DriverError> {
    if total == 0 {
        return Ok(());
    }
    let grid = ((total as u32).div_ceil(FLAT_BLOCK), 1u32, 1u32);
    let block = (FLAT_BLOCK, 1u32, 1u32);
    // SAFETY: engine-owned device buffers live for the call; params point at caller stack locals.
    unsafe { cuda_result::launch_kernel(kernel, grid, block, 0, stream, params) }
}

/// Compile one `.cu` source → PTX via system `nvcc` (parity with the engine's kernels + attn-score's
/// `compile_cu_to_ptx`; honors the same `LLMRS_NVCC_STD` / `LLMRS_NVCC_CCBIN` knobs). `tag` gives the
/// temp file a unique stem so the two KIVI modules never clobber each other or the score crate's.
fn compile_cu_to_ptx(src: &str, cc_major: i32, cc_minor: i32, tag: &str) -> Result<String, String> {
    use std::io::Write;
    let arch = format!("sm_{cc_major}{cc_minor}");
    let tmp = std::env::temp_dir();
    let cu = tmp.join(format!("argus_kivi_{tag}.cu"));
    let ptx = tmp.join(format!("argus_kivi_{tag}.ptx"));
    {
        let mut f = std::fs::File::create(&cu).map_err(|e| format!("create .cu: {e}"))?;
        f.write_all(src.as_bytes())
            .map_err(|e| format!("write .cu: {e}"))?;
    }
    let nvcc = [
        "/usr/local/cuda-11.4/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ]
    .iter()
    .find(|p| std::path::Path::new(p).exists())
    .map(|s| s.to_string())
    .unwrap_or_else(|| "nvcc".to_string());
    let std_flag = std::env::var("LLMRS_NVCC_STD").unwrap_or_else(|_| "c++17".to_string());
    let ccbin = std::env::var("LLMRS_NVCC_CCBIN").ok();

    let mut cmd = std::process::Command::new(&nvcc);
    cmd.args([
        "--ptx",
        "-allow-unsupported-compiler",
        &format!("-arch={arch}"),
        &format!("-std={std_flag}"),
    ]);
    if let Some(p) = &ccbin {
        cmd.arg("-ccbin").arg(p);
    }
    cmd.args(["-o", ptx.to_str().unwrap(), cu.to_str().unwrap()]);
    let out = cmd.output().map_err(|e| format!("run nvcc: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "nvcc kivi_{tag}.cu failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let ptx_bytes = std::fs::read(&ptx).map_err(|e| format!("read ptx: {e}"))?;
    let _ = std::fs::remove_file(&cu);
    let _ = std::fs::remove_file(&ptx);
    String::from_utf8(ptx_bytes).map_err(|e| format!("ptx utf8: {e}"))
}

/// Load a PTX module and resolve a function by name (None if absent, mirroring OpenCL `.ok()`).
fn get_fn(module: cuda_sys::CUmodule, name: &str) -> Option<cuda_sys::CUfunction> {
    let cname = CString::new(name).ok()?;
    unsafe { cuda_result::module::get_function(module, cname) }.ok()
}

/// make factory — reconstruct the borrowed context, compile both `.cu` modules, resolve the 7 kernels.
fn make_cuda_kivi(args: &CudaQuantAttnMakeArgs) -> Result<Box<dyn CudaQuantAttnBackend>, String> {
    if args.cu_context.is_null() {
        return Err("null cu_context".to_string());
    }
    let ctx = unsafe {
        CudaContext::from_raw_context(
            args.cu_device as usize,
            args.cu_device,
            args.cu_context as cuda_sys::CUcontext,
        )
    }
    .map_err(|e| format!("from_raw_context: {e}"))?;
    let ctx = ManuallyDrop::new(ctx);

    let ptx_q2 = compile_cu_to_ptx(KIVI_Q2_SRC, args.cc_major, args.cc_minor, "q2")?;
    let ptx_attn = compile_cu_to_ptx(KIVI_ATTN_SRC, args.cc_major, args.cc_minor, "attn")?;
    let ptx_q2_c = CString::new(ptx_q2).map_err(|e| format!("q2 ptx CString: {e}"))?;
    let ptx_attn_c = CString::new(ptx_attn).map_err(|e| format!("attn ptx CString: {e}"))?;

    // SAFETY: the context is current on this thread (from_raw_context bound it); the ptx images are
    // valid NUL-terminated PTX living across the loads.
    let module_q2 = unsafe { cuda_result::module::load_data(ptx_q2_c.as_ptr() as *const c_void) }
        .map_err(|e| format!("q2 module load: {e}"))?;
    let module_attn =
        unsafe { cuda_result::module::load_data(ptx_attn_c.as_ptr() as *const c_void) }.map_err(
            |e| {
                // Unload the first module before bailing so we don't leak it on the error path.
                unsafe {
                    let _ = cuda_result::module::unload(module_q2);
                }
                format!("attn module load: {e}")
            },
        )?;

    let use_4wave = std::env::var("LLMRS_F16_GEMV_USE_4WAVE")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);

    Ok(Box::new(KiviCudaBackend {
        deq_value_q2_f16: get_fn(module_q2, "kivi_dequantize_value_q2_f16"),
        deq_key_q2_f16: get_fn(module_q2, "kivi_dequantize_key_q2_f16"),
        scatter_residual_f16: get_fn(module_q2, "kivi_scatter_residual_f16"),
        gather_update: get_fn(module_q2, "kivi_gather_update"),
        attn_q2: get_fn(module_attn, "kernel_attn_gen_kivi_q2"),
        attn_q4: get_fn(module_attn, "kernel_attn_gen_kivi_q4"),
        attn_q8: get_fn(module_attn, "kernel_attn_gen_kivi_q8"),
        is_nosub: !use_4wave,
        module_q2,
        module_attn,
        _ctx: ctx,
    }))
}

/// Static registration — the engine resolves `"kivi_abi"` via `find_cuda_quant_attn`. KEEP the key
/// byte-exact (the `--backend-cap kivi_abi` selector). Force-linked with the OpenCL registration.
#[distributed_slice(CUDA_QUANT_ATTN_REGS)]
static KIVI_CUDA_QUANT_ATTN: CudaQuantAttnReg = CudaQuantAttnReg {
    name: "kivi_abi",
    make: make_cuda_kivi,
};

#[cfg(test)]
mod tests {
    #[test]
    fn cuda_kivi_registered() {
        assert!(argus_extension_api::find_cuda_quant_attn("kivi_abi").is_some());
    }
}
