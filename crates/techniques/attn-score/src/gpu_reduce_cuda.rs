//! CUDA half of the observer/score axis — the [`CudaScoreReduceBackend`] that owns `score_reduce.cu`.
//!
//! Mirrors the OpenCL [`gpu_reduce`](crate::gpu_reduce) exactly, one backend over: it compiles
//! `score_reduce.cu` from the host's borrowed `CUcontext` at `make()` time (system `nvcc` → PTX, for
//! PTX-version parity with the engine's own kernels — NOT NVRTC) and dispatches the fused per-token
//! reduce on the engine's lent `CUstream`. The engine owns the score device buffers and lends raw
//! `CUdeviceptr`s; this crate never allocates or frees them. Registered into `CUDA_SCORE_REDUCERS`
//! (disjoint from the OpenCL `SCORE_REDUCERS`) under the same name `"attn_score"`.

use std::ffi::{CString, c_void};
use std::mem::ManuallyDrop;
use std::sync::Arc;

use argus_extension_api::{
    CUDA_SCORE_REDUCERS, CudaScoreReduceArgs, CudaScoreReduceBackend, CudaScoreReduceMakeArgs,
    CudaScoreReduceReg,
};
use cudarc::driver::{CudaContext, result as cuda_result, sys as cuda_sys};
use linkme::distributed_slice;

/// The fused score-reduce kernels, compiled from source at make() time (mirrors `score_reduce.cl`).
const SCORE_REDUCE_SRC: &str = include_str!("score_reduce.cu");

struct AttnScoreCudaReducer {
    /// Borrowed engine context, reconstructed via `from_raw_context`. `ManuallyDrop` so its `Drop`
    /// never destroys the engine's `CUcontext` (`from_raw_context` sets `is_primary=false`, whose
    /// `Drop` calls `cuCtxDestroy`) — the CUDA analog of the OpenCL reducer's `from_raw_copied_ptr`.
    _ctx: ManuallyDrop<Arc<CudaContext>>,
    module: cuda_sys::CUmodule,
    kernel: cuda_sys::CUfunction,
    kernel_per_layer: cuda_sys::CUfunction,
}

// SAFETY: the raw CUDA handles are only touched on the single inference thread, behind the engine's
// single-threaded score accumulator (mirrors the OpenCL reducer's unsafe Send/Sync).
unsafe impl Send for AttnScoreCudaReducer {}
unsafe impl Sync for AttnScoreCudaReducer {}

impl Drop for AttnScoreCudaReducer {
    fn drop(&mut self) {
        // Unload only our own module; the borrowed context is intentionally leaked (ManuallyDrop).
        unsafe {
            let _ = cuda_result::module::unload(self.module);
        }
    }
}

impl CudaScoreReduceBackend for AttnScoreCudaReducer {
    fn name(&self) -> &str {
        "attn_score"
    }

    fn reduce(&self, args: &CudaScoreReduceArgs) -> i32 {
        match self.reduce_inner(args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[attn_score cuda] reduce failed: {e}");
                -1
            }
        }
    }

    fn reduce_per_layer(&self, args: &CudaScoreReduceArgs) -> i32 {
        match self.reduce_per_layer_inner(args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[attn_score cuda] reduce_per_layer failed: {e}");
                -1
            }
        }
    }
}

impl AttnScoreCudaReducer {
    fn reduce_inner(&self, args: &CudaScoreReduceArgs) -> Result<(), String> {
        if args.cu_stream.is_null() {
            return Err("null cu_stream".to_string());
        }
        let stream = args.cu_stream as cuda_sys::CUstream;

        // Locals must outlive the launch: `kernel_params` holds pointers to each of them, and
        // cuLaunchKernel copies the values out synchronously at enqueue.
        let score_buf = args.score_buf;
        let importance = args.importance;
        let head_importance = args.head_importance;
        let decay = args.decay_factor;
        let n_layers = args.n_layers as i32;
        let n_heads_q = args.n_heads_q as i32;
        let n_kv_heads = args.n_kv_heads as i32;
        let cache_seq_len = args.cache_seq_len as i32;
        let score_stride = args.score_stride as i32;
        let max_seq_len = args.max_seq_len as i32;

        // Signature order: kernel_score_fused_reduce(scores, importance, head_importance,
        // decay_factor, n_layers, n_heads_q, n_kv_heads, cache_seq_len, score_stride, max_seq_len).
        let mut params: [*mut c_void; 10] = [
            &score_buf as *const _ as *mut c_void,
            &importance as *const _ as *mut c_void,
            &head_importance as *const _ as *mut c_void,
            &decay as *const _ as *mut c_void,
            &n_layers as *const _ as *mut c_void,
            &n_heads_q as *const _ as *mut c_void,
            &n_kv_heads as *const _ as *mut c_void,
            &cache_seq_len as *const _ as *mut c_void,
            &score_stride as *const _ as *mut c_void,
            &max_seq_len as *const _ as *mut c_void,
        ];

        let grid = (args.cache_seq_len.div_ceil(64) as u32, 1u32, 1u32);
        let block = (64u32, 1u32, 1u32);
        // SAFETY: validated stream; engine-owned buffers live until the call returns; params point at
        // stack locals that outlive the synchronous enqueue.
        unsafe {
            cuda_result::launch_kernel(self.kernel, grid, block, 0, stream, &mut params)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn reduce_per_layer_inner(&self, args: &CudaScoreReduceArgs) -> Result<(), String> {
        if args.layer_flat_importance == 0 {
            return Err("null layer_flat_importance".to_string());
        }
        if args.cu_stream.is_null() {
            return Err("null cu_stream".to_string());
        }
        let stream = args.cu_stream as cuda_sys::CUstream;

        let score_buf = args.score_buf;
        let layer_flat = args.layer_flat_importance;
        let decay = args.decay_factor;
        let n_layers = args.n_layers as i32;
        let n_heads_q = args.n_heads_q as i32;
        let cache_seq_len = args.cache_seq_len as i32;
        let score_stride = args.score_stride as i32;
        let max_seq_len = args.max_seq_len as i32;

        // Signature order: kernel_score_fused_reduce_per_layer(scores, layer_flat, decay_factor,
        // n_layers, n_heads_q, cache_seq_len, score_stride, max_seq_len).
        let mut params: [*mut c_void; 8] = [
            &score_buf as *const _ as *mut c_void,
            &layer_flat as *const _ as *mut c_void,
            &decay as *const _ as *mut c_void,
            &n_layers as *const _ as *mut c_void,
            &n_heads_q as *const _ as *mut c_void,
            &cache_seq_len as *const _ as *mut c_void,
            &score_stride as *const _ as *mut c_void,
            &max_seq_len as *const _ as *mut c_void,
        ];

        let grid = (args.cache_seq_len.div_ceil(64) as u32, 1u32, 1u32);
        let block = (64u32, 1u32, 1u32);
        // SAFETY: as in `reduce_inner`.
        unsafe {
            cuda_result::launch_kernel(self.kernel_per_layer, grid, block, 0, stream, &mut params)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Compile `score_reduce.cu` → PTX via system `nvcc` (parity with the engine's
/// `CudaKernels::compile_with_nvcc`; honors the same `LLMRS_NVCC_STD` / `LLMRS_NVCC_CCBIN` knobs).
fn compile_cu_to_ptx(src: &str, cc_major: i32, cc_minor: i32) -> Result<String, String> {
    use std::io::Write;
    let arch = format!("sm_{cc_major}{cc_minor}");
    let tmp = std::env::temp_dir();
    let cu = tmp.join("argus_attn_score_reduce.cu");
    let ptx = tmp.join("argus_attn_score_reduce.ptx");
    {
        let mut f = std::fs::File::create(&cu).map_err(|e| format!("create .cu: {e}"))?;
        f.write_all(src.as_bytes())
            .map_err(|e| format!("write .cu: {e}"))?;
    }
    let nvcc = ["/usr/local/cuda-11.4/bin/nvcc", "/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"]
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
            "nvcc score_reduce.cu failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let ptx_bytes = std::fs::read(&ptx).map_err(|e| format!("read ptx: {e}"))?;
    let _ = std::fs::remove_file(&cu);
    let _ = std::fs::remove_file(&ptx);
    String::from_utf8(ptx_bytes).map_err(|e| format!("ptx utf8: {e}"))
}

/// make factory — reconstructs the borrowed context, compiles `score_reduce.cu` from it, loads the two
/// kernels. Returns an error string on any failure (the engine then falls back to CPU-readback scoring).
fn make_cuda_attn_score_reducer(
    args: &CudaScoreReduceMakeArgs,
) -> Result<Box<dyn CudaScoreReduceBackend>, String> {
    if args.cu_context.is_null() {
        return Err("null cu_context".to_string());
    }
    // Reconstruct + bind the engine's context on this thread. ManuallyDrop → never destroy it.
    let ctx = unsafe {
        CudaContext::from_raw_context(
            args.cu_device as usize,
            args.cu_device,
            args.cu_context as cuda_sys::CUcontext,
        )
    }
    .map_err(|e| format!("from_raw_context: {e}"))?;
    let ctx = ManuallyDrop::new(ctx);

    let ptx = compile_cu_to_ptx(SCORE_REDUCE_SRC, args.cc_major, args.cc_minor)?;
    let ptx_c = CString::new(ptx).map_err(|e| format!("ptx CString: {e}"))?;

    // SAFETY: the context is current on this thread (from_raw_context bound it); ptx_c is a valid
    // NUL-terminated PTX image living across the load.
    let module = unsafe { cuda_result::module::load_data(ptx_c.as_ptr() as *const c_void) }
        .map_err(|e| format!("module load_data: {e}"))?;
    let kernel = unsafe {
        cuda_result::module::get_function(
            module,
            CString::new("kernel_score_fused_reduce").unwrap(),
        )
    }
    .map_err(|e| format!("get_function reduce: {e}"))?;
    let kernel_per_layer = unsafe {
        cuda_result::module::get_function(
            module,
            CString::new("kernel_score_fused_reduce_per_layer").unwrap(),
        )
    }
    .map_err(|e| format!("get_function per_layer: {e}"))?;

    Ok(Box::new(AttnScoreCudaReducer {
        _ctx: ctx,
        module,
        kernel,
        kernel_per_layer,
    }))
}

/// Static registration — the engine resolves `"attn_score"` via `find_cuda_score_reducer`. Force-linked
/// alongside the CPU producer + OpenCL reducer through the same `use attn_score as _;`.
#[distributed_slice(CUDA_SCORE_REDUCERS)]
static ATTN_SCORE_CUDA_REDUCER: CudaScoreReduceReg = CudaScoreReduceReg {
    name: "attn_score",
    make: make_cuda_attn_score_reducer,
};

#[cfg(test)]
mod tests {
    #[test]
    fn cuda_reducer_registered() {
        assert!(argus_extension_api::find_cuda_score_reducer("attn_score").is_some());
    }
}
