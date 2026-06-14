//! Custom CUDA kernels for the CUDA backend.
//!
//! Kernels are compiled from `kernels.cu` using system `nvcc` at CudaBackend
//! initialization time. The resulting PTX is loaded via cudarc's module API.
//!
//! Why nvcc instead of NVRTC: JetPack 5.x ships CUDA 11.4 driver + 11.8 toolkit.
//! NVRTC 11.8 emits PTX 7.8, but the 11.4 driver only supports PTX ≤7.4.
//! Using the system nvcc (which matches the driver) avoids version mismatch.

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// CUDA kernel source embedded at compile time.
const KERNEL_SOURCE: &str = include_str!("kernels.cu");

/// Holds compiled CudaFunction handles for all custom kernels.
pub struct CudaKernels {
    pub rms_norm: CudaFunction,
    pub rope_inplace: CudaFunction,
    pub softmax: CudaFunction,
    pub silu_mul: CudaFunction,
    pub gelu_tanh_mul: CudaFunction,
    pub add_assign: CudaFunction,
    pub scale: CudaFunction,
    pub cast_f32_f16: CudaFunction,
    pub cast_f16_f32: CudaFunction,
    pub add_row_bias: CudaFunction,
    pub kv_scatter: CudaFunction,
    pub kv_scatter_batch: CudaFunction,
    pub kv_scatter_f32_batch: CudaFunction,
    pub gather_f16: CudaFunction,
    pub flash_attn_f32: CudaFunction,
    pub flash_attn_f16kv: CudaFunction,
    pub flash_prefill_f32_dk64: CudaFunction,
    pub flash_prefill_f32_dk128: CudaFunction,
    pub flash_prefill_f32_dk256: CudaFunction,
    pub flash_prefill_f16kv_dk64: CudaFunction,
    pub flash_prefill_f16kv_dk128: CudaFunction,
    pub flash_prefill_f16kv_dk256: CudaFunction,
}

impl CudaKernels {
    /// Compile all custom kernels for the given compute capability.
    ///
    /// Uses system `nvcc` to compile kernels.cu → PTX, then loads via cudarc.
    /// Falls back to NVRTC if nvcc is not available.
    pub fn compile(ctx: &Arc<CudaContext>, cc: (i32, i32)) -> Result<Self> {
        let arch = format!("sm_{}{}", cc.0, cc.1);
        let ptx = Self::compile_with_nvcc(&arch)?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| anyhow!("CUDA module load failed: {e}"))?;

        let load = |name: &str| -> Result<CudaFunction> {
            module
                .load_function(name)
                .map_err(|e| anyhow!("Failed to load kernel '{name}': {e}"))
        };

        Ok(Self {
            rms_norm: load("rms_norm_f32")?,
            rope_inplace: load("rope_inplace_f32")?,
            softmax: load("softmax_f32")?,
            silu_mul: load("silu_mul_f32")?,
            gelu_tanh_mul: load("gelu_tanh_mul_f32")?,
            add_assign: load("add_assign_f32")?,
            scale: load("scale_f32")?,
            cast_f32_f16: load("cast_f32_to_f16")?,
            cast_f16_f32: load("cast_f16_to_f32")?,
            add_row_bias: load("add_row_bias_f32")?,
            kv_scatter: load("kv_scatter_f32_to_f16")?,
            kv_scatter_batch: load("kv_scatter_f32_to_f16_batch")?,
            kv_scatter_f32_batch: load("kv_scatter_f32_to_f32_batch")?,
            gather_f16: load("gather_f16")?,
            flash_attn_f32: load("attention_gen_f32_naive")?,
            flash_attn_f16kv: load("attention_gen_f16kv_naive")?,
            flash_prefill_f32_dk64: load("flash_attn_prefill_f32_dk64")?,
            flash_prefill_f32_dk128: load("flash_attn_prefill_f32_dk128")?,
            flash_prefill_f32_dk256: load("flash_attn_prefill_f32_dk256")?,
            flash_prefill_f16kv_dk64: load("flash_attn_prefill_f16kv_dk64")?,
            flash_prefill_f16kv_dk128: load("flash_attn_prefill_f16kv_dk128")?,
            flash_prefill_f16kv_dk256: load("flash_attn_prefill_f16kv_dk256")?,
        })
    }

    /// Compile kernels.cu using system nvcc.
    ///
    /// Writes the embedded source to a temp file, invokes nvcc --ptx,
    /// and reads the resulting PTX. This ensures PTX version matches
    /// the system's CUDA driver.
    fn compile_with_nvcc(arch: &str) -> Result<Ptx> {
        use std::io::Write;
        let tmp_dir = std::env::temp_dir();
        let cu_path = tmp_dir.join("llmrs_kernels.cu");
        let ptx_path = tmp_dir.join("llmrs_kernels.ptx");

        // Write kernel source to temp file
        {
            let mut f = std::fs::File::create(&cu_path)
                .map_err(|e| anyhow!("Failed to create temp .cu file: {e}"))?;
            f.write_all(KERNEL_SOURCE.as_bytes())
                .map_err(|e| anyhow!("Failed to write kernel source: {e}"))?;
        }

        // Find nvcc — prefer the driver-matching version
        let nvcc = if std::path::Path::new("/usr/local/cuda-11.4/bin/nvcc").exists() {
            "/usr/local/cuda-11.4/bin/nvcc".to_string()
        } else if std::path::Path::new("/usr/local/cuda/bin/nvcc").exists() {
            "/usr/local/cuda/bin/nvcc".to_string()
        } else if std::path::Path::new("/opt/cuda/bin/nvcc").exists() {
            "/opt/cuda/bin/nvcc".to_string() // Arch Linux default
        } else {
            "nvcc".to_string() // hope it's in PATH
        };

        // C++ standard for host header parsing. Pinned to c++17 to dodge
        // g++ 16's C++20-only declarations (char8_t, `requires` clauses) that
        // older nvcc EDG frontends (e.g. 13.2) cannot parse. Override with
        // env LLMRS_NVCC_STD if a different standard is required.
        let std_flag = std::env::var("LLMRS_NVCC_STD").unwrap_or_else(|_| "c++17".to_string());

        // Optional host compiler override. Useful when a system-default g++
        // is incompatible with nvcc and a compatible compiler (e.g. g++-14)
        // is installed at a non-default path.
        let ccbin = std::env::var("LLMRS_NVCC_CCBIN").ok();

        eprintln!(
            "[CUDA] Compiling kernels with {nvcc} --ptx -arch={arch} -std={std_flag}{ccbin_note} ...",
            ccbin_note = ccbin
                .as_deref()
                .map(|p| format!(" -ccbin {p}"))
                .unwrap_or_default()
        );

        // `-allow-unsupported-compiler` allows nvcc to proceed when the host
        // C++ compiler is newer than nvcc's officially supported range
        // (e.g. g++ 16 + CUDA 13 on Arch Linux). nvcc only uses the host
        // compiler to parse declarations referenced by device code.
        let mut cmd = std::process::Command::new(&nvcc);
        cmd.args([
            "--ptx",
            "-allow-unsupported-compiler",
            &format!("-arch={arch}"),
            &format!("-std={std_flag}"),
        ]);
        if let Some(path) = &ccbin {
            cmd.arg("-ccbin").arg(path);
        }
        cmd.args(["-o", ptx_path.to_str().unwrap(), cu_path.to_str().unwrap()]);
        let output = cmd
            .output()
            .map_err(|e| anyhow!("Failed to run nvcc: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("nvcc compilation failed:\n{stderr}"));
        }

        // Read PTX
        let ptx_bytes =
            std::fs::read(&ptx_path).map_err(|e| anyhow!("Failed to read PTX output: {e}"))?;

        // Clean up temp files
        let _ = std::fs::remove_file(&cu_path);
        let _ = std::fs::remove_file(&ptx_path);

        eprintln!(
            "[CUDA] Kernel compilation done ({} bytes PTX)",
            ptx_bytes.len()
        );

        let ptx_str =
            String::from_utf8(ptx_bytes).map_err(|e| anyhow!("PTX is not valid UTF-8: {e}"))?;
        Ok(Ptx::from_src(ptx_str))
    }
}
