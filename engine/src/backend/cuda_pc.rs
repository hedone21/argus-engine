//! CUDA backend for Jetson (SM >= 7.2, UMA).
//!
//! Phase 4: NVRTC custom kernels — all ops run on GPU, no CPU fallback
//! except Q4_0 matmul (cuBLAS doesn't support Q4 natively).
//!
//! Key components:
//! - `CudaHostBuffer` (cuMemHostAlloc + DEVICEMAP): zero-copy CPU+GPU access
//! - cuBLAS `cublasGemmEx`: F32xF32, F16xF16 matmul
//! - NVRTC custom kernels: rms_norm, rope, softmax, silu_mul, attention_gen, etc.
//! - F32xF16 matmul: cast F32->F16 via custom kernel, then cuBLAS F16xF16

pub mod kernels;
pub mod memory;

use crate::backend::Backend;
use crate::buffer::{Buffer, DType};
use crate::memory::cuda::buffer::{CudaBuffer, CudaDeviceBuffer, CudaHostBuffer};
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg, result as cuda_result};
use kernels::CudaKernels;
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Per-category sync tag for `maybe_sync_cat`.
///
/// Ported from cuda_embedded (2026-04-19). Used to bisect which per-op
/// syncs are load-bearing on host-class CUDA versus pure overhead. The
/// category definitions are intentionally identical to cuda_embedded's
/// so CLI/policy strings stay portable between backends.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyncCat {
    /// `add_assign` (residual accumulate — runs 2x per layer).
    ElemAdd = 0,
    /// `rms_norm` and `rms_norm_oop` launches.
    RmsNorm = 1,
    /// `rope_inplace`.
    Rope = 2,
    /// cuBLAS / custom GEMV matmul variants in `matmul_transposed`,
    /// plus the pre-launch input-ordering guard at the top of that
    /// function. The cast F32->F16 that precedes `gemm_ex` is also
    /// folded in here (same-stream, not a CPU-visible read).
    Matmul = 3,
    /// `kv_scatter_f32_to_f16` (+ batch variant).
    KvScatter = 4,
    /// `attention_gen` (flash_attn_f32/f16kv) and
    /// `flash_attention_prefill`.
    Attention = 5,
    /// `gather` (embedding lookup).
    Gather = 6,
    /// Sync before dropping to a CPU fallback (unsupported dtype,
    /// missing device pointer). Keeping these syncs is load-bearing
    /// whenever a preceding GPU op's output feeds the fallback.
    FallbackPre = 7,
    /// FFN activation kernels: `silu_mul`, `gelu_tanh_mul`.
    ElemAct = 8,
    /// Miscellaneous elementwise: `scale`, `softmax`, `add_row_bias`,
    /// `cast_f16_f32`. Rarely exercised on the Llama decode path
    /// (softmax/add_row_bias never fire; scale/cast run at edges).
    ElemMisc = 9,
}

impl SyncCat {
    #[inline]
    fn bit(self) -> u32 {
        1u32 << (self as u32)
    }
}

/// Bitmask controlling which `SyncCat` launch sites still issue a
/// `synchronize()`. Stored as the raw `u32` inside an `AtomicU32`.
#[derive(Copy, Clone, Debug)]
pub struct SyncPolicy(u32);

impl SyncPolicy {
    pub const EMPTY: SyncPolicy = SyncPolicy(0);
    /// All categories active — identical to the pre-port behaviour.
    pub const ALL: SyncPolicy = SyncPolicy(0x3FF);
    /// llama.cpp-style minimal set: only the `FallbackPre` guard
    /// stays on so CPU fallback paths still observe in-flight GPU
    /// writes. On host discrete GPUs this is closer to llama.cpp's
    /// behaviour than on Jetson UMA.
    pub const LLAMACPP: SyncPolicy = SyncPolicy(1u32 << (SyncCat::FallbackPre as u32));
    /// Bisection result from cuda_embedded (Jetson UMA): `ElemAdd` +
    /// `FallbackPre`. Kept as a portable default starting point for
    /// host bisection.
    pub const MINIMAL: SyncPolicy =
        SyncPolicy((1u32 << (SyncCat::ElemAdd as u32)) | (1u32 << (SyncCat::FallbackPre as u32)));

    #[inline]
    pub fn contains(self, cat: SyncCat) -> bool {
        (self.0 & cat.bit()) != 0
    }

    #[inline]
    pub fn with(mut self, cat: SyncCat) -> Self {
        self.0 |= cat.bit();
        self
    }

    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn from_raw(v: u32) -> Self {
        SyncPolicy(v & 0x3FF)
    }

    /// Parse a policy string (see cuda_embedded::SyncPolicy::parse for the
    /// full grammar). Kept in sync so CLI strings are portable.
    pub fn parse(spec: &str) -> std::result::Result<Self, String> {
        let s = spec.trim();
        if s.eq_ignore_ascii_case("all") {
            return Ok(Self::ALL);
        }
        if s.eq_ignore_ascii_case("none") {
            return Ok(Self::EMPTY);
        }
        if s.eq_ignore_ascii_case("llamacpp") {
            return Ok(Self::LLAMACPP);
        }
        if s.eq_ignore_ascii_case("minimal") {
            return Ok(Self::MINIMAL);
        }
        let rest = s
            .strip_prefix("custom:")
            .ok_or_else(|| format!("unknown sync policy: '{s}'"))?;
        let mut mask = SyncPolicy::EMPTY;
        for tok in rest.split(',') {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            let lc = t.to_ascii_lowercase();
            if lc == "elementwise" || lc == "elem" {
                mask = mask
                    .with(SyncCat::ElemAdd)
                    .with(SyncCat::ElemAct)
                    .with(SyncCat::ElemMisc);
                continue;
            }
            let cat = match lc.as_str() {
                "elem_add" | "add_assign" | "elemadd" => SyncCat::ElemAdd,
                "elem_act" | "elemact" | "silu_mul" | "silu" => SyncCat::ElemAct,
                "elem_misc" | "elemmisc" | "misc" => SyncCat::ElemMisc,
                "rmsnorm" | "rms" => SyncCat::RmsNorm,
                "rope" => SyncCat::Rope,
                "matmul" | "mm" | "cublas" => SyncCat::Matmul,
                "kv_scatter" | "kv" | "kvscatter" => SyncCat::KvScatter,
                "attention" | "attn" => SyncCat::Attention,
                "gather" | "embed" => SyncCat::Gather,
                "fallback" | "fallback_pre" | "cpu_fallback" => SyncCat::FallbackPre,
                other => return Err(format!("unknown sync category: '{other}'")),
            };
            mask = mask.with(cat);
        }
        Ok(mask)
    }
}

/// Wrapper around a raw cuBLAS handle for safe sharing via Arc.
///
/// cuBLAS handles are thread-safe for concurrent read operations and
/// serialized internally for writes, so sharing via Arc is safe.
struct CublasHandle {
    handle: cublas_sys::cublasHandle_t,
}

// SAFETY: cuBLAS handles are thread-safe -- the library serializes concurrent
// API calls internally. We only share via Arc (no mutable aliasing).
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // SAFETY: We own this handle (created in CudaBackend::new) and
        // only destroy it once (here in Drop).
        unsafe {
            let _ = cublas_result::destroy_handle(self.handle);
        }
    }
}

/// CUDA backend targeting Jetson UMA devices.
///
/// Phase 4: cuBLAS matmul + NVRTC custom kernels for all other ops.
/// CPU fallback only for Q4_0 matmul and matmul_slice.
#[derive(Clone)]
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    device_name: String,
    compute_capability: (i32, i32),
    is_uma: bool,
    cublas: Arc<CublasHandle>,
    kernels: Arc<CudaKernels>,
    /// Cached F16 cast buffer for matmul_transposed F32×F16 path.
    cast_cache: Arc<std::sync::Mutex<Option<CudaHostBuffer>>>,
    /// Reusable device buffer for per-call attention score readback
    /// (Phase B: resilience/eviction GPU path). Holds normalized softmax weights
    /// in layout `[num_heads_q, score_stride]` F32. Allocated lazily on first
    /// score-enabled `attention_gen` call and grown on demand. `None` when
    /// score readback has not yet been requested.
    ///
    /// For UMA (Jetson) this uses `CudaHostBuffer` so kernel writes are
    /// automatically visible to the CPU without an explicit `memcpy_dtoh`.
    /// For discrete GPUs the same path works via managed-memory migration.
    score_tmp_buf: Arc<std::sync::Mutex<Option<CudaHostBuffer>>>,
    /// When `true`, launch-helper `maybe_sync_cat` calls are no-ops
    /// (suppresses every per-op sync). Explicit `Backend::synchronize()`
    /// and API-boundary syncs (read_buffer/copy_from/cast's CPU-read
    /// fallback) are unaffected. Toggled via `set_defer_sync` / CLI
    /// `--cuda-defer-sync`.
    defer_sync: Arc<AtomicBool>,
    /// Per-category sync bitmask (see `SyncPolicy`). When a
    /// `maybe_sync_cat(SyncCat::X)` call fires, it invokes
    /// `synchronize()` iff the `X` bit is set. Defaults to
    /// `SyncPolicy::ALL` for backward compatibility with the
    /// pre-port behaviour. `defer_sync=true` overrides this.
    sync_policy: Arc<AtomicU32>,
    /// LISWAP-2 prototype (plan: chasing-hopper, Phase 1).
    ///
    /// Lazily-created secondary CUDA stream used by `enqueue_write_async`
    /// to overlap H2D weight uploads with compute on the default stream.
    /// Created on first use behind the `LLMRS_CUDA_TRANSFER_STREAM` env
    /// gate (default ON). When unset, the async path falls back to the
    /// synchronous `copy_weight_from`.
    transfer_stream: Arc<std::sync::OnceLock<Arc<cudarc::driver::CudaStream>>>,

    /// B-5b Phase 2 Stage 1: CPU companion backend injected at construction
    /// time. Stage 2 will route the ~13 `cpu_fallback` sites in this module
    /// through `Backend::cpu_companion()` so they share a single CPU backend
    /// instance instead of constructing a fresh one per fallback call.
    cpu_companion: Arc<dyn Backend>,
}

impl CudaBackend {
    /// Initialize CUDA backend on device 0.
    ///
    /// Requirements:
    /// - Compute Capability >= 7.2 (Jetson AGX Xavier / Volta+)
    /// - Managed Memory support
    pub fn new() -> Result<Self> {
        Self::with_ordinal(0)
    }

    /// Initialize CUDA backend on a specific device ordinal.
    pub fn with_ordinal(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| anyhow!("CUDA context creation failed (ordinal={ordinal}): {e}"))?;

        // Query compute capability
        let cc_major = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| anyhow!("Failed to query CC major: {e}"))?;
        let cc_minor = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| anyhow!("Failed to query CC minor: {e}"))?;

        if cc_major < 7 || (cc_major == 7 && cc_minor < 2) {
            return Err(anyhow!(
                "CUDA backend requires SM >= 7.2 (Jetson Xavier+), got sm_{cc_major}{cc_minor}"
            ));
        }

        // Check managed memory support
        let managed_mem = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)
            .unwrap_or(0);
        if managed_mem == 0 {
            return Err(anyhow!("Device does not support managed (unified) memory"));
        }

        // Check UMA: integrated GPU (e.g., Jetson) shares physical memory with CPU.
        // CU_DEVICE_ATTRIBUTE_INTEGRATED = 1 means integrated (UMA), 0 means discrete.
        let integrated = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
            .unwrap_or(0);
        let is_uma = integrated != 0;

        // Get device name
        let cu_device = cuda_result::device::get(ordinal as i32)
            .map_err(|e| anyhow!("Failed to get CUDA device: {e}"))?;
        let device_name = cuda_result::device::get_name(cu_device)
            .unwrap_or_else(|_| format!("CUDA Device {ordinal}"));

        // Create cuBLAS handle and bind to the same stream as custom kernels.
        // Without this, cuBLAS uses stream 0 while custom kernels use
        // ctx.default_stream(), causing cross-stream ordering issues.
        let cublas_handle =
            cublas_result::create_handle().map_err(|e| anyhow!("cublasCreate_v2 failed: {e}"))?;
        unsafe {
            cublas_result::set_stream(cublas_handle, ctx.default_stream().cu_stream() as _)
                .map_err(|e| anyhow!("cublasSetStream failed: {e}"))?;
        }

        // Compile custom NVRTC kernels
        let cc = (cc_major, cc_minor);
        let kernels = CudaKernels::compile(&ctx, cc)?;

        eprintln!(
            "[CUDA] Device: {} (sm_{}{}, UMA={}, managed_mem={}, cuBLAS=ready, kernels=ready)",
            device_name, cc_major, cc_minor, is_uma, managed_mem
        );

        let backend = Self {
            ctx,
            device_name,
            compute_capability: (cc_major, cc_minor),
            is_uma,
            cublas: Arc::new(CublasHandle {
                handle: cublas_handle,
            }),
            kernels: Arc::new(kernels),
            cast_cache: Arc::new(std::sync::Mutex::new(None)),
            score_tmp_buf: Arc::new(std::sync::Mutex::new(None)),
            defer_sync: Arc::new(AtomicBool::new(false)),
            sync_policy: Arc::new(AtomicU32::new(SyncPolicy::ALL.raw())),
            // LISWAP-2 prototype (plan: chasing-hopper). Lazy-init in
            // `transfer_stream_or_init`.
            transfer_stream: Arc::new(std::sync::OnceLock::new()),
            // CPU companion for host fallback routing (S-2 sprint
            // 2026-05-24): shared singleton — feature detection runs once.
            // LAYER-EXEMPT: cross_backend_bootstrap — §13.8-P cpu_companion init.
            cpu_companion: crate::backend::cpu::cpu_singleton(),
        };

        // Run self-test to verify kernel launch + arg passing
        backend.self_test()?;

        Ok(backend)
    }

    /// Access the underlying CUDA context (for buffer allocation, kernel launches, etc).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Compute capability as (major, minor).
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// Enable or disable the experimental deferred-sync mode.
    ///
    /// When enabled, every `maybe_sync_cat` call becomes a no-op; the
    /// caller is responsible for syncing before any CPU-visible read
    /// (e.g. before sampling reads the logits). Does NOT affect
    /// `Backend::synchronize()`, `read_buffer`/`copy_from`, or the
    /// CPU-read fallback path inside `cast`.
    #[inline]
    pub fn set_defer_sync(&self, v: bool) {
        self.defer_sync.store(v, Ordering::Relaxed);
    }

    /// Query the current deferred-sync setting.
    #[inline]
    pub fn defer_sync_enabled(&self) -> bool {
        self.defer_sync.load(Ordering::Relaxed)
    }

    /// Set the per-category sync policy. `defer_sync=true` still
    /// takes precedence and suppresses sync for every category.
    #[inline]
    pub fn set_sync_policy(&self, policy: SyncPolicy) {
        self.sync_policy.store(policy.raw(), Ordering::Relaxed);
    }

    /// Current per-category sync policy.
    #[inline]
    pub fn sync_policy(&self) -> SyncPolicy {
        SyncPolicy::from_raw(self.sync_policy.load(Ordering::Relaxed))
    }

    /// Category-aware sync helper. Invokes `synchronize()` iff:
    /// - `defer_sync` is **off**, AND
    /// - the current `SyncPolicy` has `cat` set.
    #[inline]
    fn maybe_sync_cat(&self, cat: SyncCat) -> Result<()> {
        if self.defer_sync.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mask = self.sync_policy.load(Ordering::Relaxed);
        if (mask & cat.bit()) != 0 {
            self.synchronize()
        } else {
            Ok(())
        }
    }

    // ── LISWAP-2 prototype: async H2D transfer stream ───────────────────
    // Plan: .claude/plans/compiled-chasing-hopper.md (Phase 1).

    /// Whether the LISWAP-2 transfer stream is enabled at runtime.
    /// Reads `LLMRS_CUDA_TRANSFER_STREAM` once and caches. Default ON;
    /// "0" / "false" / "off" disables (caller falls back to sync copy).
    fn transfer_stream_env_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| match std::env::var("LLMRS_CUDA_TRANSFER_STREAM") {
            Ok(v) => {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "false" || v == "off")
            }
            Err(_) => true,
        })
    }

    /// Lazily build (or return) the secondary transfer stream. Returns
    /// `None` if the env-gate is disabled or stream creation failed —
    /// callers fall back to the synchronous path in that case.
    /// Note: cuda_pc has no graph capture path, so no capture check.
    fn transfer_stream_or_init(&self) -> Option<Arc<cudarc::driver::CudaStream>> {
        if !Self::transfer_stream_env_enabled() {
            return None;
        }
        if let Some(s) = self.transfer_stream.get() {
            return Some(s.clone());
        }
        match self.ctx.new_stream() {
            Ok(s) => {
                let _ = self.transfer_stream.set(s);
                self.transfer_stream.get().cloned()
            }
            Err(e) => {
                log::warn!(
                    "LLMRS_CUDA_TRANSFER_STREAM: secondary stream creation failed ({e}); \
                     falling back to synchronous `copy_weight_from`"
                );
                None
            }
        }
    }

    /// Attempt to get a CUdeviceptr from a Buffer.
    ///
    /// Returns Some(device_ptr) if the buffer is a CudaHostBuffer or CudaBuffer,
    /// None otherwise (e.g., SharedBuffer/MmapBuffer without CUDA registration).
    fn get_device_ptr(buf: &dyn Buffer) -> Option<cuda_sys::CUdeviceptr> {
        buf.as_any()
            .downcast_ref::<CudaHostBuffer>()
            .map(|hb| hb.device_ptr())
            .or_else(|| {
                buf.as_any()
                    .downcast_ref::<CudaBuffer>()
                    .map(|cb| cb.device_ptr())
            })
            .or_else(|| {
                buf.as_any()
                    .downcast_ref::<CudaDeviceBuffer>()
                    .map(|db| db.device_ptr())
            })
    }

    /// Get device pointer or return error with context.
    #[allow(dead_code)]
    fn require_device_ptr(buf: &dyn Buffer, name: &str) -> Result<cuda_sys::CUdeviceptr> {
        Self::get_device_ptr(buf)
            .ok_or_else(|| anyhow!("{name}: buffer has no CUDA device pointer"))
    }

    /// Run a basic self-test to verify kernel launch + arg passing works.
    /// Tests add_assign, scale, rms_norm with known data.
    pub fn self_test(&self) -> Result<()> {
        use crate::buffer::DType;
        use crate::memory::cuda::buffer::CudaHostBuffer;

        // === Test 1: add_assign [1,2,3] + [4,5,6] = [5,7,9] ===
        let a_buf = CudaHostBuffer::new(12, DType::F32)?;
        let b_buf = CudaHostBuffer::new(12, DType::F32)?;
        unsafe {
            let a = std::slice::from_raw_parts_mut(a_buf.as_mut_ptr() as *mut f32, 3);
            a.copy_from_slice(&[1.0, 2.0, 3.0]);
            let b = std::slice::from_raw_parts_mut(b_buf.as_mut_ptr() as *mut f32, 3);
            b.copy_from_slice(&[4.0, 5.0, 6.0]);
        }
        let a_ptr = a_buf.device_ptr();
        let b_ptr = b_buf.device_ptr();
        let k: i32 = 3;
        let stream = self.ctx.default_stream();
        unsafe {
            stream
                .launch_builder(&self.kernels.add_assign)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&k)
                .launch(Self::launch_config_1d(3))
                .map_err(|e| anyhow!("self-test add_assign launch failed: {e}"))?;
        }
        self.synchronize()?;
        let result = unsafe { std::slice::from_raw_parts(a_buf.as_ptr() as *const f32, 3) };
        eprint!(
            "[CUDA self-test] add_assign: [{:.1}, {:.1}, {:.1}]",
            result[0], result[1], result[2]
        );
        if (result[0] - 5.0).abs() > 0.01
            || (result[1] - 7.0).abs() > 0.01
            || (result[2] - 9.0).abs() > 0.01
        {
            eprintln!(" FAIL (expected [5.0, 7.0, 9.0])");
            return Err(anyhow!("self-test add_assign FAILED"));
        }
        eprintln!(" OK");

        // === Test 2: scale [2,4,6] * 0.5 = [1,2,3] ===
        let s_buf = CudaHostBuffer::new(12, DType::F32)?;
        unsafe {
            let s = std::slice::from_raw_parts_mut(s_buf.as_mut_ptr() as *mut f32, 3);
            s.copy_from_slice(&[2.0, 4.0, 6.0]);
        }
        let s_ptr = s_buf.device_ptr();
        let v: f32 = 0.5;
        unsafe {
            stream
                .launch_builder(&self.kernels.scale)
                .arg(&s_ptr)
                .arg(&v)
                .arg(&k)
                .launch(Self::launch_config_1d(3))
                .map_err(|e| anyhow!("self-test scale launch failed: {e}"))?;
        }
        self.synchronize()?;
        let result = unsafe { std::slice::from_raw_parts(s_buf.as_ptr() as *const f32, 3) };
        eprint!(
            "[CUDA self-test] scale:      [{:.1}, {:.1}, {:.1}]",
            result[0], result[1], result[2]
        );
        if (result[0] - 1.0).abs() > 0.01
            || (result[1] - 2.0).abs() > 0.01
            || (result[2] - 3.0).abs() > 0.01
        {
            eprintln!(" FAIL (expected [1.0, 2.0, 3.0])");
            return Err(anyhow!("self-test scale FAILED"));
        }
        eprintln!(" OK");

        // === Test 3: rms_norm [3,4] with weight [1,1] ===
        // rms = sqrt((9+16)/2) = sqrt(12.5) = 3.536
        // rms_inv = 1/3.536 = 0.2828
        // result = [3*0.2828, 4*0.2828] = [0.849, 1.131]
        let x_buf = CudaHostBuffer::new(8, DType::F32)?;
        let w_buf = CudaHostBuffer::new(8, DType::F32)?;
        unsafe {
            let x = std::slice::from_raw_parts_mut(x_buf.as_mut_ptr() as *mut f32, 2);
            x.copy_from_slice(&[3.0, 4.0]);
            let w = std::slice::from_raw_parts_mut(w_buf.as_mut_ptr() as *mut f32, 2);
            w.copy_from_slice(&[1.0, 1.0]);
        }
        let x_ptr = x_buf.device_ptr();
        let w_ptr = w_buf.device_ptr();
        let ncols: i32 = 2;
        let eps: f32 = 1e-5;
        let add_unit: i32 = 0;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1), // min warp size
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.kernels.rms_norm)
                .arg(&x_ptr) // dst
                .arg(&x_ptr) // x (in-place)
                .arg(&w_ptr) // weight
                .arg(&ncols)
                .arg(&eps)
                .arg(&add_unit)
                .launch(cfg)
                .map_err(|e| anyhow!("self-test rms_norm launch failed: {e}"))?;
        }
        self.synchronize()?;
        let result = unsafe { std::slice::from_raw_parts(x_buf.as_ptr() as *const f32, 2) };
        let expected_0 = 3.0 / (12.5f32 + 1e-5).sqrt();
        let expected_1 = 4.0 / (12.5f32 + 1e-5).sqrt();
        eprint!(
            "[CUDA self-test] rms_norm:   [{:.4}, {:.4}]",
            result[0], result[1]
        );
        if (result[0] - expected_0).abs() > 0.01 || (result[1] - expected_1).abs() > 0.01 {
            eprintln!(" FAIL (expected [{:.4}, {:.4}])", expected_0, expected_1);
            return Err(anyhow!(
                "self-test rms_norm FAILED: got [{}, {}]",
                result[0],
                result[1]
            ));
        }
        eprintln!(" OK");

        Ok(())
    }

    /// Standard 1D launch config: ceil(n/256) blocks, 256 threads.
    fn launch_config_1d(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl Backend for CudaBackend {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "CUDA"
    }

    fn device(&self) -> &str {
        &self.device_name
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn is_discrete_gpu(&self) -> bool {
        !self.is_uma
    }

    fn bind_current_thread(&self) -> Result<()> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| anyhow!("CUDA bind_to_thread failed: {e}"))
    }

    fn flash_attention_prefill(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        n_heads_q: usize,
        n_heads_kv: usize,
        seq_len: usize,
        cache_seq_len: usize,
        head_dim: usize,
        kv_capacity: usize,
        batch_size: usize,
        _is_head_major: bool,
    ) -> Result<bool> {
        let kv_dtype = k_cache.dtype();

        // Only support F32/F16 KV and head_dim in {64, 128, 256}
        if !matches!(head_dim, 64 | 128 | 256) || kv_dtype == DType::Q4_0 {
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            return Ok(false);
        }

        let q_ptr = Self::get_device_ptr(q.buffer().as_ref());
        let k_ptr = Self::get_device_ptr(k_cache.buffer().as_ref());
        let v_ptr = Self::get_device_ptr(v_cache.buffer().as_ref());
        let out_ptr = Self::get_device_ptr(out.buffer().as_ref());

        if let (Some(qp), Some(kp), Some(vp), Some(op)) = (q_ptr, k_ptr, v_ptr, out_ptr) {
            let block_m: u32 = 32;
            let cfg = LaunchConfig {
                grid_dim: (
                    seq_len.div_ceil(block_m as usize) as u32,
                    (n_heads_q * batch_size) as u32,
                    1,
                ),
                block_dim: (block_m, 1, 1),
                shared_mem_bytes: 0,
            };

            let nhq = n_heads_q as i32;
            let nkv = n_heads_kv as i32;
            let sl = seq_len as i32;
            let csl = cache_seq_len as i32;
            let cap = kv_capacity as i32;
            let bs = batch_size as i32;

            let kernel = match (kv_dtype, head_dim) {
                (DType::F32, 64) => &self.kernels.flash_prefill_f32_dk64,
                (DType::F32, 128) => &self.kernels.flash_prefill_f32_dk128,
                (DType::F32, 256) => &self.kernels.flash_prefill_f32_dk256,
                (DType::F16, 64) => &self.kernels.flash_prefill_f16kv_dk64,
                (DType::F16, 128) => &self.kernels.flash_prefill_f16kv_dk128,
                (DType::F16, 256) => &self.kernels.flash_prefill_f16kv_dk256,
                _ => {
                    self.maybe_sync_cat(SyncCat::FallbackPre)?;
                    return Ok(false);
                }
            };

            let stream = self.ctx.default_stream();
            // SAFETY: qp is valid F32 device ptr for Q [batch, seq_len, n_heads_q, head_dim].
            // kp/vp are valid F32 or F16 device ptrs for KV [batch, n_heads_kv, capacity, head_dim].
            // op is valid F32 device ptr for output, same layout as Q.
            // All dimensions are checked by callers (transformer layer).
            unsafe {
                stream
                    .launch_builder(kernel)
                    .arg(&qp)
                    .arg(&kp)
                    .arg(&vp)
                    .arg(&op)
                    .arg(&nhq)
                    .arg(&nkv)
                    .arg(&sl)
                    .arg(&csl)
                    .arg(&cap)
                    .arg(&bs)
                    .launch(cfg)
                    .map_err(|e| anyhow!("flash_attn_prefill launch failed: {e}"))?;
            }
            self.maybe_sync_cat(SyncCat::Attention)?;
            Ok(true)
        } else {
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            Ok(false)
        }
    }

    // --- Math ops ---

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        self.cpu_companion().matmul(a, b, out)
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        // Sync before cuBLAS: other ops may have written to input buffers
        // on a different stream than cuBLAS uses.
        self.maybe_sync_cat(SyncCat::Matmul)?;
        let a_dtype = a.dtype();
        let b_dtype = b.dtype();

        // Q4_0 weights always go through CPU fallback. Sync first so the
        // CPU read of `a`/`b` host pointers observes all in-flight GPU
        // writes — the matching guard at the top of every other fallback
        // path. Without it, `--cuda-sync-policy minimal` (default) skips
        // the upstream Matmul sync and the fallback sees stale activations,
        // producing non-deterministic Q4 prefill output (Bug A/B,
        // 2026-04-27).
        if b_dtype == DType::Q4_0 || a_dtype == DType::Q4_0 {
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            return self.cpu_companion().matmul_transposed(a, b, out);
        }

        // Try to get device pointers for cuBLAS
        let a_dev = Self::get_device_ptr(a.buffer().as_ref());
        let b_dev = Self::get_device_ptr(b.buffer().as_ref());
        let out_dev = Self::get_device_ptr(out.buffer().as_ref());

        // All three buffers must have device pointers for cuBLAS
        if let (Some(a_ptr), Some(b_ptr), Some(out_ptr)) = (a_dev, b_dev, out_dev) {
            let a_dims = a.shape().dims();
            let b_dims = b.shape().dims();
            let a_rank = a_dims.len();
            let b_rank = b_dims.len();

            let k = a_dims[a_rank - 1] as i32;
            let m: i32 = a_dims[..a_rank - 1].iter().product::<usize>() as i32;
            let n = b_dims[b_rank - 2] as i32;

            if a_dtype == DType::F32 && b_dtype == DType::F32 {
                // Pure F32: use sgemm
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                // SAFETY: All device pointers are valid CudaHostBuffer/CudaBuffer allocations.
                // Dimensions are checked by shape accessors. cuBLAS handle is valid.
                unsafe {
                    cublas_result::sgemm(
                        self.cublas.handle,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        b_ptr as *const f32,
                        k,
                        a_ptr as *const f32,
                        k,
                        &beta,
                        out_ptr as *mut f32,
                        n,
                    )
                    .map_err(|e| anyhow!("cublasSgemm failed: {e}"))?;
                }
                self.maybe_sync_cat(SyncCat::Matmul)?;
                Ok(())
            } else if a_dtype == DType::F32 && b_dtype == DType::F16 {
                // F32 activation x F16 weight: cast A to F16, then F16xF16 cuBLAS.
                // Reuse cached buffer to avoid per-call allocation.
                let k_elements = (m * k) as usize;
                let needed = k_elements * 2;
                let a_f16_buf = {
                    let mut cache = self.cast_cache.lock().unwrap();
                    match cache.take() {
                        Some(buf) if buf.size() >= needed => buf,
                        _ => CudaHostBuffer::new(needed, DType::F16)?,
                    }
                };
                let a_f16_ptr = a_f16_buf.device_ptr();

                // Launch F32->F16 cast kernel
                let cfg = Self::launch_config_1d(k_elements);
                let stream = self.ctx.default_stream();
                let k_i32 = k_elements as i32;
                // SAFETY: a_ptr is valid F32 device memory of size k_elements*4.
                // a_f16_ptr is valid F16 device memory of size k_elements*2.
                // Cast kernel reads k_elements floats and writes k_elements halfs.
                unsafe {
                    stream
                        .launch_builder(&self.kernels.cast_f32_f16)
                        .arg(&a_ptr)
                        .arg(&a_f16_ptr)
                        .arg(&k_i32)
                        .launch(cfg)
                        .map_err(|e| anyhow!("cast_f32_to_f16 kernel launch failed: {e}"))?;
                }

                // F16xF16 cuBLAS GemmEx
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                // SAFETY: a_f16_ptr and b_ptr are valid F16 device memory.
                // out_ptr is valid F32 device memory. Dimensions match.
                unsafe {
                    cublas_result::gemm_ex(
                        self.cublas.handle,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        b_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        k,
                        a_f16_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        k,
                        &beta as *const f32 as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_32F,
                        n,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .map_err(|e| anyhow!("cublasGemmEx (F32->F16 x F16) failed: {e}"))?;
                }
                self.maybe_sync_cat(SyncCat::Matmul)?;
                *self.cast_cache.lock().unwrap() = Some(a_f16_buf);
                Ok(())
            } else if a_dtype == DType::F16 && b_dtype == DType::F16 {
                // Both F16: use GemmEx with F32 compute for accuracy
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                // SAFETY: Both pointers are valid F16 device memory. out_ptr is valid F32.
                unsafe {
                    cublas_result::gemm_ex(
                        self.cublas.handle,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        b_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        k,
                        a_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        k,
                        &beta as *const f32 as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_32F,
                        n,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .map_err(|e| anyhow!("cublasGemmEx (F16xF16) failed: {e}"))?;
                }
                self.maybe_sync_cat(SyncCat::Matmul)?;
                Ok(())
            } else {
                // Unsupported dtype combination -> CPU fallback
                self.cpu_companion().matmul_transposed(a, b, out)
            }
        } else {
            // No device pointers available -> CPU fallback
            self.cpu_companion().matmul_transposed(a, b, out)
        }
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        rows: usize,
        cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        self.cpu_companion().matmul_slice(a, b, rows, cols, out)
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        let n = a.numel();
        let a_ptr = Self::require_device_ptr(a.buffer().as_ref(), "add_assign a")?;
        let b_ptr = Self::require_device_ptr(b.buffer().as_ref(), "add_assign b")?;
        let k = n as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: a_ptr and b_ptr are valid F32 device memory of n elements each.
        unsafe {
            stream
                .launch_builder(&self.kernels.add_assign)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&k)
                .launch(Self::launch_config_1d(n))
                .map_err(|e| anyhow!("add_assign kernel launch failed: {e}"))?;
        }
        self.maybe_sync_cat(SyncCat::ElemAdd)?;
        Ok(())
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        let n = x.numel();
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "scale x")?;
        let k = n as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: x_ptr is valid F32 device memory of n elements.
        unsafe {
            stream
                .launch_builder(&self.kernels.scale)
                .arg(&x_ptr)
                .arg(&v)
                .arg(&k)
                .launch(Self::launch_config_1d(n))
                .map_err(|e| anyhow!("scale kernel launch failed: {e}"))?;
        }
        self.maybe_sync_cat(SyncCat::ElemMisc)?;
        Ok(())
    }

    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        let n = a.numel();
        let a_ptr = Self::require_device_ptr(a.buffer().as_ref(), "silu_mul gate")?;
        let b_ptr = Self::require_device_ptr(b.buffer().as_ref(), "silu_mul up")?;
        let k = n as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: a_ptr (gate) and b_ptr (up) are valid F32 device memory of n elements.
        unsafe {
            stream
                .launch_builder(&self.kernels.silu_mul)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&k)
                .launch(Self::launch_config_1d(n))
                .map_err(|e| anyhow!("silu_mul kernel launch failed: {e}"))?;
        }
        self.maybe_sync_cat(SyncCat::ElemAct)?;
        Ok(())
    }

    fn gelu_tanh_mul(&self, gate: &mut Tensor, up: &Tensor) -> Result<()> {
        let n = gate.numel();
        let gate_ptr = Self::require_device_ptr(gate.buffer().as_ref(), "gelu_tanh_mul gate")?;
        let up_ptr = Self::require_device_ptr(up.buffer().as_ref(), "gelu_tanh_mul up")?;
        let k = n as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: gate_ptr and up_ptr are valid F32 device memory of n elements.
        unsafe {
            stream
                .launch_builder(&self.kernels.gelu_tanh_mul)
                .arg(&gate_ptr)
                .arg(&up_ptr)
                .arg(&k)
                .launch(Self::launch_config_1d(n))
                .map_err(|e| anyhow!("gelu_tanh_mul kernel launch failed: {e}"))?;
        }
        self.maybe_sync_cat(SyncCat::ElemAct)?;
        Ok(())
    }

    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        let ncols = dims[rank - 1] as i32;
        let nrows: usize = dims[..rank - 1].iter().product::<usize>().max(1);
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "rms_norm x")?;
        let w_ptr = Self::require_device_ptr(w.buffer().as_ref(), "rms_norm w")?;
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };
        let block_size = (ncols as usize).min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.ctx.default_stream();
        // SAFETY: x_ptr (in-place dst+src) and w_ptr are valid F32 device memory.
        // nrows * ncols == total elements.
        unsafe {
            stream
                .launch_builder(&self.kernels.rms_norm)
                .arg(&x_ptr) // dst
                .arg(&x_ptr) // x (in-place)
                .arg(&w_ptr) // weight
                .arg(&ncols)
                .arg(&eps)
                .arg(&add_unit_i32)
                .launch(cfg)
                .map_err(|e| anyhow!("rms_norm kernel launch failed: {e}"))?;
        }
        self.maybe_sync_cat(SyncCat::RmsNorm)?;
        Ok(())
    }

    fn rms_norm_oop(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        let ncols = dims[rank - 1] as i32;
        let nrows: usize = dims[..rank - 1].iter().product::<usize>().max(1);
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "rms_norm_oop x")?;
        let out_ptr = Self::require_device_ptr(out.buffer().as_ref(), "rms_norm_oop out")?;
        let w_ptr = Self::require_device_ptr(w.buffer().as_ref(), "rms_norm_oop w")?;
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };
        let block_size = (ncols as usize).min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.ctx.default_stream();
        // SAFETY: out_ptr (dst), x_ptr (src), w_ptr are valid F32 device memory.
        unsafe {
            stream
                .launch_builder(&self.kernels.rms_norm)
                .arg(&out_ptr) // dst
                .arg(&x_ptr) // x (read-only)
                .arg(&w_ptr) // weight
                .arg(&ncols)
                .arg(&eps)
                .arg(&add_unit_i32)
                .launch(cfg)
                .map_err(|e| anyhow!("rms_norm_oop kernel launch failed: {e}"))?;
        }
        self.maybe_sync_cat(SyncCat::RmsNorm)?;
        Ok(())
    }

    fn add_rms_norm_oop(
        &self,
        x: &mut Tensor,
        residual: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        // Fused: x += residual; out = rms_norm(x) * w
        self.add_assign(x, residual)?;
        self.rms_norm_oop(x, out, w, eps, add_unit)
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        let ncols = dims[rank - 1] as i32;
        let nrows: usize = dims[..rank - 1].iter().product::<usize>().max(1);
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "softmax x")?;
        let block_size = (ncols as usize).min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.ctx.default_stream();
        // SAFETY: x_ptr is valid F32 device memory of nrows*ncols elements.
        unsafe {
            stream
                .launch_builder(&self.kernels.softmax)
                .arg(&x_ptr)
                .arg(&ncols)
                .launch(cfg)
                .map_err(|e| anyhow!("softmax kernel launch failed: {e}"))?;
        }
        self.maybe_sync_cat(SyncCat::ElemMisc)?;
        Ok(())
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        if rank < 3 {
            return Err(anyhow!(
                "rope_inplace: expected at least 3 dims, got {rank}"
            ));
        }
        let head_dim = dims[rank - 1];
        let n_heads = dims[rank - 2];
        let seq_len: usize = dims[..rank - 1]
            .iter()
            .rev()
            .skip(1)
            .next()
            .copied()
            .unwrap_or(1);

        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "rope_inplace x")?;
        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let sl = seq_len as i32;
        let sp = start_pos as i32;
        let cfg = LaunchConfig {
            grid_dim: ((seq_len * n_heads) as u32, 1, 1),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.ctx.default_stream();
        unsafe {
            stream
                .launch_builder(&self.kernels.rope_inplace)
                .arg(&x_ptr)
                .arg(&hd)
                .arg(&nh)
                .arg(&sl)
                .arg(&sp)
                .arg(&theta)
                .launch(cfg)
                .map_err(|e| anyhow!("rope_inplace kernel launch failed: {e}"))?;
        }
        self.maybe_sync_cat(SyncCat::Rope)?;
        Ok(())
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        let src_dtype = src.dtype();
        let dst_dtype = dst.dtype();
        let n = src.numel();
        let src_ptr = Self::get_device_ptr(src.buffer().as_ref());
        let dst_ptr = Self::get_device_ptr(dst.buffer().as_ref());

        match (src_dtype, dst_dtype, src_ptr, dst_ptr) {
            (DType::F32, DType::F16, Some(sp), Some(dp)) => {
                let k = n as i32;
                let stream = self.ctx.default_stream();
                unsafe {
                    stream
                        .launch_builder(&self.kernels.cast_f32_f16)
                        .arg(&sp)
                        .arg(&dp)
                        .arg(&k)
                        .launch(Self::launch_config_1d(n))
                        .map_err(|e| anyhow!("cast_f32_f16 kernel launch failed: {e}"))?;
                }
                self.maybe_sync_cat(SyncCat::ElemMisc)?; // CPU may read dst immediately
                Ok(())
            }
            (DType::F16, DType::F32, Some(sp), Some(dp)) => {
                let k = n as i32;
                let stream = self.ctx.default_stream();
                unsafe {
                    stream
                        .launch_builder(&self.kernels.cast_f16_f32)
                        .arg(&sp)
                        .arg(&dp)
                        .arg(&k)
                        .launch(Self::launch_config_1d(n))
                        .map_err(|e| anyhow!("cast_f16_f32 kernel launch failed: {e}"))?;
                }
                self.maybe_sync_cat(SyncCat::ElemMisc)?;
                Ok(())
            }
            _ => {
                // Unsupported dtype or missing device ptr: sync then CPU fallback
                self.maybe_sync_cat(SyncCat::FallbackPre)?;
                self.cpu_companion().cast(src, dst)
            }
        }
    }

    fn add_row_bias(&self, x: &mut Tensor, bias: &Tensor) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        let ncols = dims[rank - 1] as i32;
        let total = x.numel();
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "add_row_bias x")?;
        let bias_ptr = Self::require_device_ptr(bias.buffer().as_ref(), "add_row_bias bias")?;
        let total_i32 = total as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: x_ptr is valid F32 device ptr of total elements; bias_ptr is valid F32 of ncols.
        unsafe {
            stream
                .launch_builder(&self.kernels.add_row_bias)
                .arg(&x_ptr)
                .arg(&bias_ptr)
                .arg(&ncols)
                .arg(&total_i32)
                .launch(Self::launch_config_1d(total))
                .map_err(|e| anyhow!("add_row_bias kernel launch failed: {e}"))?;
        }
        self.maybe_sync_cat(SyncCat::ElemMisc)?;
        Ok(())
    }

    fn kv_scatter_f32_to_f16(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        head_dim: usize,
        capacity: usize,
        write_pos: usize,
    ) -> Result<()> {
        let ks_ptr = Self::get_device_ptr(k_src.buffer().as_ref());
        let vs_ptr = Self::get_device_ptr(v_src.buffer().as_ref());
        let kd_ptr = Self::get_device_ptr(k_dst.buffer().as_ref());
        let vd_ptr = Self::get_device_ptr(v_dst.buffer().as_ref());

        if let (Some(ks), Some(vs), Some(kd), Some(vd)) = (ks_ptr, vs_ptr, kd_ptr, vd_ptr) {
            // Infer kv_heads from k_src shape: [..., kv_heads, head_dim]
            let dims = k_src.shape().dims().to_vec();
            let kv_heads = if dims.len() >= 2 {
                dims[dims.len() - 2]
            } else {
                1
            };
            let hd = head_dim as i32;
            let cap = capacity as i32;
            let wp = write_pos as i32;
            let cfg = LaunchConfig {
                grid_dim: (kv_heads as u32, 1, 1),
                block_dim: (head_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let stream = self.ctx.default_stream();
            // SAFETY: ks/vs are valid F32 device ptrs; kd/vd are valid F16 device ptrs.
            // Dimensions match kv_heads * head_dim for src, kv_heads * capacity * head_dim for dst.
            unsafe {
                stream
                    .launch_builder(&self.kernels.kv_scatter)
                    .arg(&ks)
                    .arg(&vs)
                    .arg(&kd)
                    .arg(&vd)
                    .arg(&hd)
                    .arg(&cap)
                    .arg(&wp)
                    .launch(cfg)
                    .map_err(|e| anyhow!("kv_scatter kernel launch failed: {e}"))?;
            }
            self.maybe_sync_cat(SyncCat::KvScatter)?;
            Ok(())
        } else {
            // Missing device ptr: sync then CPU fallback
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            self.cpu_companion()
                .kv_scatter_f32_to_f16(k_src, v_src, k_dst, v_dst, head_dim, capacity, write_pos)
        }
    }

    fn supports_kv_scatter_batch(&self) -> bool {
        true
    }

    fn kv_scatter_f32_to_f16_batch(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        write_pos_start: usize,
        seq_len: usize,
    ) -> Result<()> {
        let ks = Self::get_device_ptr(k_src.buffer().as_ref());
        let vs = Self::get_device_ptr(v_src.buffer().as_ref());
        let kd = Self::get_device_ptr(k_dst.buffer().as_ref());
        let vd = Self::get_device_ptr(v_dst.buffer().as_ref());

        if let (Some(ks), Some(vs), Some(kd), Some(vd)) = (ks, vs, kd, vd) {
            let hd = head_dim as i32;
            let cap = capacity as i32;
            let wps = write_pos_start as i32;
            let sl = seq_len as i32;
            let kvh = kv_heads as i32;
            let cfg = LaunchConfig {
                grid_dim: (kv_heads as u32, seq_len as u32, 1),
                block_dim: (head_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let stream = self.ctx.default_stream();
            unsafe {
                stream
                    .launch_builder(&self.kernels.kv_scatter_batch)
                    .arg(&ks)
                    .arg(&vs)
                    .arg(&kd)
                    .arg(&vd)
                    .arg(&kvh)
                    .arg(&hd)
                    .arg(&cap)
                    .arg(&wps)
                    .arg(&sl)
                    .launch(cfg)
                    .map_err(|e| anyhow!("kv_scatter_batch launch failed: {e}"))?;
            }
            self.maybe_sync_cat(SyncCat::KvScatter)?;
            Ok(())
        } else {
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            self.cpu_companion().kv_scatter_f32_to_f16_batch(
                k_src,
                v_src,
                k_dst,
                v_dst,
                kv_heads,
                head_dim,
                capacity,
                write_pos_start,
                seq_len,
            )
        }
    }

    fn supports_kv_scatter_f32_batch(&self) -> bool {
        true
    }

    fn kv_scatter_f32_to_f32_batch(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        write_pos_start: usize,
        seq_len: usize,
    ) -> Result<()> {
        let ks = Self::get_device_ptr(k_src.buffer().as_ref());
        let vs = Self::get_device_ptr(v_src.buffer().as_ref());
        let kd = Self::get_device_ptr(k_dst.buffer().as_ref());
        let vd = Self::get_device_ptr(v_dst.buffer().as_ref());

        if let (Some(ks), Some(vs), Some(kd), Some(vd)) = (ks, vs, kd, vd) {
            let hd = head_dim as i32;
            let cap = capacity as i32;
            let wps = write_pos_start as i32;
            let sl = seq_len as i32;
            let kvh = kv_heads as i32;
            let cfg = LaunchConfig {
                grid_dim: (kv_heads as u32, seq_len as u32, 1),
                block_dim: (head_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let stream = self.ctx.default_stream();
            unsafe {
                stream
                    .launch_builder(&self.kernels.kv_scatter_f32_batch)
                    .arg(&ks)
                    .arg(&vs)
                    .arg(&kd)
                    .arg(&vd)
                    .arg(&kvh)
                    .arg(&hd)
                    .arg(&cap)
                    .arg(&wps)
                    .arg(&sl)
                    .launch(cfg)
                    .map_err(|e| anyhow!("kv_scatter_f32_batch launch failed: {e}"))?;
            }
            self.maybe_sync_cat(SyncCat::KvScatter)?;
            Ok(())
        } else {
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            self.cpu_companion().kv_scatter_f32_to_f32_batch(
                k_src,
                v_src,
                k_dst,
                v_dst,
                kv_heads,
                head_dim,
                capacity,
                write_pos_start,
                seq_len,
            )
        }
    }

    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
        // GPU kernel only supports F16 embedding -> F32 output.
        // For other dtypes, sync and fall back to CPU.
        if src.dtype() != DType::F16 {
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            return self.cpu_companion().gather(src, indices, dst);
        }

        let src_ptr = Self::get_device_ptr(src.buffer().as_ref());
        let idx_ptr = Self::get_device_ptr(indices.buffer().as_ref());
        let dst_ptr = Self::get_device_ptr(dst.buffer().as_ref());

        if let (Some(sp), Some(ip), Some(dp)) = (src_ptr, idx_ptr, dst_ptr) {
            let dims = dst.shape().dims().to_vec();
            let rank = dims.len();
            let dim = dims[rank - 1];
            let n_tokens: usize = dims[..rank - 1].iter().product::<usize>().max(1);
            let dim_i32 = dim as i32;
            let y_blocks = dim.div_ceil(256) as u32;
            let cfg = LaunchConfig {
                grid_dim: (n_tokens as u32, y_blocks, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let stream = self.ctx.default_stream();
            // SAFETY: sp is valid F16 embed device ptr; ip is valid i32 indices; dp is valid F32 output.
            unsafe {
                stream
                    .launch_builder(&self.kernels.gather_f16)
                    .arg(&sp)
                    .arg(&ip)
                    .arg(&dp)
                    .arg(&dim_i32)
                    .launch(cfg)
                    .map_err(|e| anyhow!("gather_f16 kernel launch failed: {e}"))?;
            }
            self.maybe_sync_cat(SyncCat::Gather)?;
            Ok(())
        } else {
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            self.cpu_companion().gather(src, indices, dst)
        }
    }

    fn attention_gen(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        scores_out: Option<&mut [f32]>,
    ) -> Result<()> {
        let kv_dtype = k_cache.dtype();
        let q_ptr = Self::get_device_ptr(q.buffer().as_ref());
        let k_ptr = Self::get_device_ptr(k_cache.buffer().as_ref());
        let v_ptr = Self::get_device_ptr(v_cache.buffer().as_ref());
        let out_ptr = Self::get_device_ptr(out.buffer().as_ref());

        // Q4_0 KV cache is not supported by the GPU attention_gen kernels; always
        // take the CPU path for that dtype. Also, if any of the input tensors
        // lack a CUDA device pointer (non-CUDA buffer), we must fall back.
        let kv_dtype_ok = matches!(kv_dtype, DType::F32 | DType::F16);
        let all_ptrs = q_ptr.is_some() && k_ptr.is_some() && v_ptr.is_some() && out_ptr.is_some();
        if !kv_dtype_ok || !all_ptrs {
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            return self.cpu_companion().attention_gen(
                q,
                k_cache,
                v_cache,
                out,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
                scores_out,
            );
        }

        // GPU path (F32 / F16 KV). Phase B: if `scores_out` is requested, bind
        // a reusable device-visible score buffer to the kernel, then copy it
        // back to the caller-provided CPU slice after the launch syncs. When
        // `scores_out` is None the kernel receives a NULL score pointer and
        // avoids any extra global writes (zero overhead on the hot path).
        let qp = q_ptr.unwrap();
        let kp = k_ptr.unwrap();
        let vp = v_ptr.unwrap();
        let op = out_ptr.unwrap();

        // Prepare scores buffer + stride (row length per head) before kernel launch.
        // We pin the MutexGuard for the duration of the launch so the
        // underlying CudaHostBuffer lives at least until the device sync below.
        let (score_dptr, score_stride_i32, scratch_guard) = if let Some(ref slice) = scores_out {
            let stride = if num_heads_q == 0 {
                0
            } else {
                slice.len() / num_heads_q
            };
            if stride == 0 || stride < cache_seq_len {
                // Malformed caller buffer: disable GPU score export to avoid OOB.
                (0u64, 0i32, None)
            } else {
                let need_bytes = num_heads_q
                    .checked_mul(stride)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| anyhow!("score buffer size overflow"))?;
                let mut guard = self.score_tmp_buf.lock().unwrap();
                let need_realloc = guard
                    .as_ref()
                    .map(|b| b.size() < need_bytes)
                    .unwrap_or(true);
                if need_realloc {
                    *guard = Some(CudaHostBuffer::new(need_bytes, DType::F32)?);
                }
                let dptr = guard.as_ref().unwrap().device_ptr();
                (dptr, stride as i32, Some(guard))
            }
        } else {
            (0u64, 0i32, None)
        };

        // Shared memory: cache_seq_len floats for scores
        let shmem = (cache_seq_len * std::mem::size_of::<f32>()) as u32;
        let block_size = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (num_heads_q as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shmem,
        };
        let nhq = num_heads_q as i32;
        let nkv = num_heads_kv as i32;
        let hd = head_dim as i32;
        // KV cache is HeadMajor: capacity is derived from the k_cache tensor shape.
        let k_dims = k_cache.shape().dims().to_vec();
        let cap = if k_dims.len() >= 3 {
            k_dims[k_dims.len() - 2] as i32
        } else {
            cache_seq_len as i32
        };
        let csl = cache_seq_len as i32;
        let stream = self.ctx.default_stream();

        match kv_dtype {
            DType::F32 => {
                // SAFETY: qp, kp, vp, op are valid F32 device ptrs with correct dimensions.
                // score_dptr is either a valid device ptr (when scores_out is Some and
                // the scratch buffer is large enough) or 0 (NULL), which the kernel
                // treats as "skip score export".
                unsafe {
                    stream
                        .launch_builder(&self.kernels.flash_attn_f32)
                        .arg(&qp)
                        .arg(&kp)
                        .arg(&vp)
                        .arg(&op)
                        .arg(&nhq)
                        .arg(&nkv)
                        .arg(&hd)
                        .arg(&cap)
                        .arg(&csl)
                        .arg(&score_dptr)
                        .arg(&score_stride_i32)
                        .launch(cfg)
                        .map_err(|e| anyhow!("attention_gen_f32 kernel launch failed: {e}"))?;
                }
            }
            DType::F16 => {
                // SAFETY: kp and vp are valid F16 device ptrs; qp and op are F32.
                unsafe {
                    stream
                        .launch_builder(&self.kernels.flash_attn_f16kv)
                        .arg(&qp)
                        .arg(&kp)
                        .arg(&vp)
                        .arg(&op)
                        .arg(&nhq)
                        .arg(&nkv)
                        .arg(&hd)
                        .arg(&cap)
                        .arg(&csl)
                        .arg(&score_dptr)
                        .arg(&score_stride_i32)
                        .launch(cfg)
                        .map_err(|e| anyhow!("attention_gen_f16kv kernel launch failed: {e}"))?;
                }
            }
            _ => unreachable!("kv_dtype_ok gate already restricted to F32/F16"),
        }

        // If a CPU-side `scores_out` was requested and the kernel wrote into
        // the scratch device buffer, sync and copy back. UMA (Jetson) makes
        // this a zero-copy view; discrete GPUs trigger managed-memory
        // migration into pinned host memory on CPU read. We sync explicitly
        // here regardless of `defer_sync` because the caller expects the
        // slice to be valid immediately after return.
        if let (Some(scratch), Some(dst)) = (scratch_guard, scores_out) {
            let stride = score_stride_i32 as usize;
            let total = num_heads_q * stride;
            self.synchronize()?;
            // SAFETY: scratch is a CudaHostBuffer with at least
            // total*sizeof(f32) bytes (we sized it above). The kernel has
            // just written `num_heads_q` rows of `cache_seq_len` floats each
            // at stride `stride`. We read the full flat region.
            let host_ptr = scratch.as_ref().unwrap().as_ptr() as *const f32;
            unsafe {
                let src = std::slice::from_raw_parts(host_ptr, total);
                let copy_len = dst.len().min(total);
                dst[..copy_len].copy_from_slice(&src[..copy_len]);
            }
        } else {
            self.maybe_sync_cat(SyncCat::Attention)?;
        }
        Ok(())
    }

    // --- Memory ops ---

    fn synchronize(&self) -> Result<()> {
        self.ctx
            .default_stream()
            .synchronize()
            .map_err(|e| anyhow!("CUDA synchronize failed: {e}"))
    }

    fn copy_slice(
        &self,
        src: &Tensor,
        dst: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        let type_size = match src.dtype() {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::U8 => 1,
            DType::Q4_0 => std::mem::size_of::<crate::quant::BlockQ4_0>(),
            _ => 1,
        };
        let byte_count = count * type_size;
        let src_byte_offset = src_offset * type_size;
        let dst_byte_offset = dst_offset * type_size;

        let src_dev = Self::get_device_ptr(src.buffer().as_ref());
        let dst_dev = Self::get_device_ptr(dst.buffer().as_ref());

        if let (Some(s), Some(d)) = (src_dev, dst_dev) {
            // Device-to-device copy. Sync first so prior compute (RoPE, matmul)
            // that wrote into `src`/`dst` on the default stream is observed.
            self.maybe_sync_cat(SyncCat::KvScatter)?;
            unsafe {
                let res = cuda_sys::cuMemcpyDtoD_v2(
                    d + dst_byte_offset as cuda_sys::CUdeviceptr,
                    s + src_byte_offset as cuda_sys::CUdeviceptr,
                    byte_count,
                );
                if res != cuda_sys::CUresult::CUDA_SUCCESS {
                    anyhow::bail!("cuMemcpyDtoD_v2 failed: {:?}", res);
                }
            }
            self.maybe_sync_cat(SyncCat::KvScatter)?;
            Ok(())
        } else {
            // No device pointer on at least one side — host memcpy fallback (CPU buffers).
            self.maybe_sync_cat(SyncCat::FallbackPre)?;
            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            if src_ptr.is_null() || dst_ptr.is_null() {
                anyhow::bail!(
                    "copy_slice: missing device ptr and missing host ptr (src null={}, dst null={})",
                    src_ptr.is_null(),
                    dst_ptr.is_null()
                );
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(src_byte_offset),
                    dst_ptr.add(dst_byte_offset),
                    byte_count,
                );
            }
            Ok(())
        }
    }

    fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
        self.synchronize()?;
        if let Some(db) = t.buffer().as_any().downcast_ref::<CudaDeviceBuffer>() {
            db.copy_to_host(dst.as_mut_ptr(), dst.len())?;
            return Ok(());
        }
        let src_ptr = t.buffer().as_ptr();
        if src_ptr.is_null() {
            anyhow::bail!("Cannot read null buffer");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_from(&self, src: &Tensor) -> Result<Tensor> {
        self.synchronize()?;
        let size = src.size();
        let src_ptr = src.as_ptr();

        if self.is_discrete_gpu() {
            // Discrete GPU: use managed memory (cuMemAllocManaged).
            // CUDA driver auto-migrates pages to VRAM on first GPU access.
            let managed_buf = CudaBuffer::new(size, src.dtype())?;
            if !src_ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, managed_buf.as_mut_ptr(), size);
                }
            }
            Ok(Tensor::new(
                src.shape().clone(),
                Arc::new(managed_buf),
                Arc::new(self.clone()),
            ))
        } else {
            // UMA (Jetson): pinned host memory is zero-copy.
            let cuda_buf = CudaHostBuffer::new(size, src.dtype())?;
            let dst_ptr = cuda_buf.as_mut_ptr();
            if !src_ptr.is_null() && !dst_ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
                }
            }
            Ok(Tensor::new(
                src.shape().clone(),
                Arc::new(cuda_buf),
                Arc::new(self.clone()),
            ))
        }
    }

    /// LISWAP-2 prototype (plan: chasing-hopper, Phase 1). cuda_pc
    /// targets discrete GPUs, so the destination is always a pure
    /// device buffer (`CudaDeviceBuffer`). Submits a `cuMemcpyHtoDAsync`
    /// on the secondary transfer stream and records a `CudaEvent` for
    /// the dispatcher's worker to wait on.
    fn enqueue_write_async(&self, src: &Tensor) -> Result<(Tensor, crate::backend::GpuEvent)> {
        let Some(transfer_stream) = self.transfer_stream_or_init() else {
            let dst = self.copy_weight_from(src)?;
            return Ok((dst, crate::backend::GpuEvent::default()));
        };

        let size = src.size();
        let src_ptr = src.as_ptr();
        if src_ptr.is_null() {
            return Err(anyhow!(
                "enqueue_write_async: source has a null host pointer (size={size}, dtype={:?})",
                src.dtype()
            ));
        }

        let dev_buf = CudaDeviceBuffer::new(size, src.dtype())?;
        // SAFETY: dispatcher worker keeps `src` alive until the
        // recorded event fires. Stream is owned by this backend.
        dev_buf.copy_from_host_async(src_ptr, size, transfer_stream.cu_stream() as _)?;

        let event = self
            .ctx
            .new_event(None)
            .map_err(|e| anyhow!("CUDA event create failed: {e}"))?;
        event
            .record(&transfer_stream)
            .map_err(|e| anyhow!("cuEventRecord failed: {e}"))?;

        let dst_tensor = Tensor::new(
            src.shape().clone(),
            Arc::new(dev_buf),
            Arc::new(self.clone()),
        );
        Ok((
            dst_tensor,
            crate::backend::GpuEvent {
                #[cfg(feature = "opencl")]
                inner_cl: None,
                inner_cu: Some(event),
            },
        ))
    }

    fn wait_event_blocking(&self, evt: &crate::backend::GpuEvent) -> Result<()> {
        if let Some(e) = evt.inner_cu.as_ref() {
            e.synchronize()
                .map_err(|err| anyhow!("cuEventSynchronize failed: {err}"))?;
            return Ok(());
        }
        self.synchronize()
    }

    fn supports_async_transfer(&self) -> bool {
        Self::transfer_stream_env_enabled()
    }

    // B-5b Phase 2 Stage 1: CPU companion override. Stage 2 will route
    // the ~13 `cpu_fallback` paths in this module through this method.
    fn cpu_companion(&self) -> &dyn Backend {
        &*self.cpu_companion
    }

    // yield_after_layer: trait default body (S-2 sprint 2026-05-24).
}

#[cfg(test)]
mod tests {
    /// Reference implementation of tiled online-softmax flash attention (matches CUDA kernel logic).
    /// Used to verify numerical stability invariants without requiring a GPU.
    fn ref_flash_attn_prefill(
        q: &[f32],       // [seq_len, n_heads_q, head_dim]
        k: &[f32],       // [n_heads_kv, capacity, head_dim]
        v: &[f32],       // [n_heads_kv, capacity, head_dim]
        out: &mut [f32], // [seq_len, n_heads_q, head_dim]
        n_heads_q: usize,
        n_heads_kv: usize,
        seq_len: usize,
        cache_seq_len: usize,
        kv_capacity: usize,
        head_dim: usize,
        block_n: usize, // KV tile size
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = n_heads_q / n_heads_kv;

        for s in 0..seq_len {
            let causal_limit = (cache_seq_len as i64) - (seq_len as i64) + (s as i64);

            for h_q in 0..n_heads_q {
                let h_kv = h_q / gqa_ratio;
                let q_off = (s * n_heads_q + h_q) * head_dim;

                let mut o_acc = vec![0.0f32; head_dim];
                let mut m_i: f32 = f32::NEG_INFINITY;
                let mut l_i: f32 = 0.0;

                // Process KV in tiles of block_n, pairs of 2
                for kv_start in (0..cache_seq_len).step_by(block_n) {
                    let kv_end = (kv_start + block_n).min(cache_seq_len);
                    let tile_len = kv_end - kv_start;

                    let mut p = 0;
                    while p + 1 < tile_len {
                        let kp0 = kv_start + p;
                        let kp1 = kv_start + p + 1;

                        let s0 = if (kp0 as i64) <= causal_limit {
                            let k_off = h_kv * kv_capacity * head_dim + kp0 * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q[q_off + d] * k[k_off + d];
                            }
                            dot * scale
                        } else {
                            f32::NEG_INFINITY
                        };

                        let s1 = if (kp1 as i64) <= causal_limit {
                            let k_off = h_kv * kv_capacity * head_dim + kp1 * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q[q_off + d] * k[k_off + d];
                            }
                            dot * scale
                        } else {
                            f32::NEG_INFINITY
                        };

                        let m_new = m_i.max(s0.max(s1));
                        if m_new > f32::NEG_INFINITY {
                            let exp0 = (s0 - m_new).exp();
                            let exp1 = (s1 - m_new).exp();
                            let rescale = if m_i > f32::NEG_INFINITY {
                                (m_i - m_new).exp()
                            } else {
                                0.0
                            };
                            l_i = l_i * rescale + exp0 + exp1;
                            for d in 0..head_dim {
                                let v0 = v[h_kv * kv_capacity * head_dim + kp0 * head_dim + d];
                                let v1 = v[h_kv * kv_capacity * head_dim + kp1 * head_dim + d];
                                o_acc[d] = o_acc[d] * rescale + exp0 * v0 + exp1 * v1;
                            }
                            m_i = m_new;
                        }
                        p += 2;
                    }

                    // Odd remainder
                    if p < tile_len {
                        let kp = kv_start + p;
                        let s = if (kp as i64) <= causal_limit {
                            let k_off = h_kv * kv_capacity * head_dim + kp * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q[q_off + d] * k[k_off + d];
                            }
                            dot * scale
                        } else {
                            f32::NEG_INFINITY
                        };

                        let m_new = m_i.max(s);
                        if m_new > f32::NEG_INFINITY {
                            let exp_s = (s - m_new).exp();
                            let rescale = if m_i > f32::NEG_INFINITY {
                                (m_i - m_new).exp()
                            } else {
                                0.0
                            };
                            l_i = l_i * rescale + exp_s;
                            for d in 0..head_dim {
                                o_acc[d] = o_acc[d] * rescale
                                    + exp_s * v[h_kv * kv_capacity * head_dim + kp * head_dim + d];
                            }
                            m_i = m_new;
                        }
                    }
                }

                let o_off = (s * n_heads_q + h_q) * head_dim;
                if l_i > 0.0 {
                    let inv_l = 1.0 / l_i;
                    for d in 0..head_dim {
                        out[o_off + d] = o_acc[d] * inv_l;
                    }
                } else {
                    for d in 0..head_dim {
                        out[o_off + d] = 0.0;
                    }
                }
            }
        }
    }

    /// Naive attention (no tiling) for reference comparison.
    fn ref_naive_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        n_heads_q: usize,
        n_heads_kv: usize,
        seq_len: usize,
        cache_seq_len: usize,
        kv_capacity: usize,
        head_dim: usize,
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = n_heads_q / n_heads_kv;
        for s in 0..seq_len {
            let causal_limit = (cache_seq_len as i64) - (seq_len as i64) + (s as i64);
            for h_q in 0..n_heads_q {
                let h_kv = h_q / gqa_ratio;
                let q_off = (s * n_heads_q + h_q) * head_dim;
                // Compute all scores
                let mut scores = vec![f32::NEG_INFINITY; cache_seq_len];
                for t in 0..cache_seq_len {
                    if (t as i64) <= causal_limit {
                        let k_off = h_kv * kv_capacity * head_dim + t * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        scores[t] = dot * scale;
                    }
                }
                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut exps = vec![0.0f32; cache_seq_len];
                if max_s > f32::NEG_INFINITY {
                    for t in 0..cache_seq_len {
                        exps[t] = (scores[t] - max_s).exp();
                        sum_exp += exps[t];
                    }
                }
                // Weighted sum
                let o_off = (s * n_heads_q + h_q) * head_dim;
                if sum_exp > 0.0 {
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for t in 0..cache_seq_len {
                            acc += exps[t] * v[h_kv * kv_capacity * head_dim + t * head_dim + d];
                        }
                        out[o_off + d] = acc / sum_exp;
                    }
                } else {
                    for d in 0..head_dim {
                        out[o_off + d] = 0.0;
                    }
                }
            }
        }
    }

    // --- Test cases ---

    #[test]
    fn test_flash_prefill_no_nan_basic() {
        // Basic case: seq_len=4, head_dim=64, 2 heads, tile_size=8
        let head_dim = 64;
        let seq_len = 4;
        let n_heads_q = 2;
        let n_heads_kv = 2;
        let capacity = 16;
        let cache_seq_len = seq_len; // fresh prefill

        let q = vec![0.1f32; seq_len * n_heads_q * head_dim];
        let k = vec![0.2f32; n_heads_kv * capacity * head_dim];
        let v = vec![0.3f32; n_heads_kv * capacity * head_dim];
        let mut out_flash = vec![0.0f32; seq_len * n_heads_q * head_dim];
        let mut out_naive = vec![0.0f32; seq_len * n_heads_q * head_dim];

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out_flash,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            8,
        );
        ref_naive_attention(
            &q,
            &k,
            &v,
            &mut out_naive,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
        );

        for i in 0..out_flash.len() {
            assert!(!out_flash[i].is_nan(), "NaN at index {i}");
            assert!(
                (out_flash[i] - out_naive[i]).abs() < 1e-5,
                "mismatch at {i}: flash={} naive={}",
                out_flash[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn test_flash_prefill_no_nan_head_dim_256() {
        // Gemma3 4B config: head_dim=256, GQA
        let head_dim = 256;
        let seq_len = 7; // NOT multiple of BLOCK_M=32 -> tests padding
        let n_heads_q = 8;
        let n_heads_kv = 4; // GQA ratio 2
        let capacity = 32;
        let cache_seq_len = seq_len;

        let q: Vec<f32> = (0..seq_len * n_heads_q * head_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let k: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
            .map(|i| ((i % 83) as f32 - 41.0) * 0.01)
            .collect();
        let v: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
            .map(|i| ((i % 71) as f32 - 35.0) * 0.01)
            .collect();
        let mut out_flash = vec![0.0f32; seq_len * n_heads_q * head_dim];
        let mut out_naive = vec![0.0f32; seq_len * n_heads_q * head_dim];

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out_flash,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            8,
        );
        ref_naive_attention(
            &q,
            &k,
            &v,
            &mut out_naive,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
        );

        for i in 0..out_flash.len() {
            assert!(!out_flash[i].is_nan(), "NaN at index {i}");
            assert!(
                (out_flash[i] - out_naive[i]).abs() < 1e-4,
                "mismatch at {i}: flash={} naive={}",
                out_flash[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn test_flash_prefill_no_nan_single_token() {
        // Edge case: seq_len=1 (single token prefill)
        let head_dim = 64;
        let seq_len = 1;
        let n_heads_q = 4;
        let n_heads_kv = 4;
        let capacity = 8;
        let cache_seq_len = 1;

        let q = vec![1.0f32; seq_len * n_heads_q * head_dim];
        let k = vec![1.0f32; n_heads_kv * capacity * head_dim];
        let v = vec![0.5f32; n_heads_kv * capacity * head_dim];
        let mut out = vec![0.0f32; seq_len * n_heads_q * head_dim];

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            8,
        );

        for i in 0..out.len() {
            assert!(!out[i].is_nan(), "NaN at index {i}");
            // With single token, output should equal v (softmax of single element = 1.0)
            assert!(
                (out[i] - 0.5).abs() < 1e-5,
                "expected 0.5, got {} at {i}",
                out[i]
            );
        }
    }

    #[test]
    fn test_flash_prefill_no_nan_odd_tile_boundary() {
        // cache_seq_len that creates odd tile remainder for all block_n values
        for &block_n in &[8, 16, 32] {
            let head_dim = 64;
            let seq_len = 5;
            let n_heads_q = 2;
            let n_heads_kv = 2;
            let cache_seq_len = block_n + 3; // ensures odd remainder
            let capacity = cache_seq_len + 4;

            let q: Vec<f32> = (0..seq_len * n_heads_q * head_dim)
                .map(|i| (i as f32 * 0.01).sin())
                .collect();
            let k: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
                .map(|i| (i as f32 * 0.02).cos())
                .collect();
            let v: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
                .map(|i| (i as f32 * 0.03).sin())
                .collect();
            let mut out_flash = vec![0.0f32; seq_len * n_heads_q * head_dim];
            let mut out_naive = vec![0.0f32; seq_len * n_heads_q * head_dim];

            ref_flash_attn_prefill(
                &q,
                &k,
                &v,
                &mut out_flash,
                n_heads_q,
                n_heads_kv,
                seq_len,
                cache_seq_len,
                capacity,
                head_dim,
                block_n,
            );
            ref_naive_attention(
                &q,
                &k,
                &v,
                &mut out_naive,
                n_heads_q,
                n_heads_kv,
                seq_len,
                cache_seq_len,
                capacity,
                head_dim,
            );

            for i in 0..out_flash.len() {
                assert!(!out_flash[i].is_nan(), "NaN at block_n={block_n} index {i}");
                assert!(
                    (out_flash[i] - out_naive[i]).abs() < 1e-4,
                    "mismatch at block_n={block_n} index {i}: flash={} naive={}",
                    out_flash[i],
                    out_naive[i]
                );
            }
        }
    }

    #[test]
    fn test_flash_prefill_no_nan_all_masked() {
        // Edge case: cache_seq_len=0 -- no KV to attend, should produce zeros not NaN
        let head_dim = 64;
        let seq_len = 4;
        let n_heads_q = 2;
        let n_heads_kv = 2;
        let capacity = 8;
        let cache_seq_len = 0;

        let q = vec![1.0f32; seq_len * n_heads_q * head_dim];
        let k = vec![0.0f32; n_heads_kv * capacity * head_dim];
        let v = vec![0.0f32; n_heads_kv * capacity * head_dim];
        let mut out = vec![f32::NAN; seq_len * n_heads_q * head_dim]; // init with NaN to detect untouched

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            8,
        );

        for i in 0..out.len() {
            assert!(!out[i].is_nan(), "NaN at index {i} (all-masked case)");
            assert_eq!(out[i], 0.0, "expected 0.0 for all-masked at index {i}");
        }
    }

    #[test]
    fn test_flash_prefill_continuation_context() {
        // Continuation: existing KV cache + new tokens
        let head_dim = 128;
        let seq_len = 3; // new tokens
        let n_heads_q = 4;
        let n_heads_kv = 2; // GQA
        let capacity = 32;
        let cache_seq_len = 10; // already 7 cached + 3 new = 10

        let q: Vec<f32> = (0..seq_len * n_heads_q * head_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let k: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let mut out_flash = vec![0.0f32; seq_len * n_heads_q * head_dim];
        let mut out_naive = vec![0.0f32; seq_len * n_heads_q * head_dim];

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out_flash,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            16,
        );
        ref_naive_attention(
            &q,
            &k,
            &v,
            &mut out_naive,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
        );

        for i in 0..out_flash.len() {
            assert!(!out_flash[i].is_nan(), "NaN at index {i}");
            assert!(
                (out_flash[i] - out_naive[i]).abs() < 1e-4,
                "mismatch at {i}: flash={} naive={}",
                out_flash[i],
                out_naive[i]
            );
        }
    }
}
