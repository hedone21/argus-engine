#![allow(unused_unsafe)]
use crate::backend::Backend;
use crate::buffer::Buffer;
use crate::buffer::DType;
use crate::memory::Memory;
use crate::memory::opencl::unified::UnifiedBuffer;
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use ocl::core::Kernel as CoreKernel;
use ocl::core::{ClContextPtr, ClDeviceIdPtr};
use ocl::{Context, Device, Platform, Program, Queue, flags};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;

use self::gpu_self_meter::OpenClEventGpuMeter;

pub mod gpu_score;
pub mod gpu_self_meter;
pub mod host_ptr_pool;
pub mod host_ptr_pool_buffer;
pub mod memory;
pub mod plan;

/// `cl_khr_priority_hints` property name (`CL_QUEUE_PRIORITY_KHR`).
const CL_QUEUE_PRIORITY_KHR: u64 = 0x1096;
/// `cl_khr_priority_hints` priority values — bitmask, not ordinal.
const CL_QUEUE_PRIORITY_HIGH_KHR: u64 = 1 << 0;
const CL_QUEUE_PRIORITY_MED_KHR: u64 = 1 << 1;
const CL_QUEUE_PRIORITY_LOW_KHR: u64 = 1 << 2;
/// `CL_QUEUE_PROPERTIES` (legacy bitfield slot in the property list).
const CL_QUEUE_PROPERTIES: u64 = 0x1093;

unsafe extern "C" {
    fn clCreateCommandQueueWithProperties(
        context: *mut std::ffi::c_void,
        device: *mut std::ffi::c_void,
        properties: *const u64,
        errcode_ret: *mut i32,
    ) -> *mut std::ffi::c_void;
}

/// Replace `queue`'s underlying `cl_command_queue` with one that carries a
/// `CL_QUEUE_PRIORITY_KHR` hint, if the device advertises
/// `cl_khr_priority_hints`. The original queue is released by `DerefMut`
/// assignment (ocl-core's `CommandQueue` has a `Drop` impl).
fn try_apply_queue_priority(
    queue: &mut Queue,
    context: &Context,
    device: Device,
    queue_flags: Option<flags::CommandQueueProperties>,
    extensions: &str,
    priority: &str,
) -> Result<()> {
    if !extensions.contains("cl_khr_priority_hints") {
        return Err(anyhow!("device lacks cl_khr_priority_hints extension"));
    }
    let priority_val = match priority {
        "low" => CL_QUEUE_PRIORITY_LOW_KHR,
        "med" | "medium" => CL_QUEUE_PRIORITY_MED_KHR,
        "high" => CL_QUEUE_PRIORITY_HIGH_KHR,
        other => {
            return Err(anyhow!(
                "unknown priority '{}': expected low|medium|high|normal",
                other
            ));
        }
    };
    let flags_bits: u64 = queue_flags.map(|f| f.bits()).unwrap_or(0);
    // Property list is zero-terminated: [name, value, name, value, …, 0].
    let mut props: Vec<u64> = Vec::with_capacity(5);
    if flags_bits != 0 {
        props.push(CL_QUEUE_PROPERTIES);
        props.push(flags_bits);
    }
    props.push(CL_QUEUE_PRIORITY_KHR);
    props.push(priority_val);
    props.push(0);

    let ctx_ptr = <&Context as ClContextPtr>::as_ptr(&context);
    let dev_ptr = <Device as ClDeviceIdPtr>::as_ptr(&device);
    let mut err: i32 = 0;
    let raw =
        unsafe { clCreateCommandQueueWithProperties(ctx_ptr, dev_ptr, props.as_ptr(), &mut err) };
    if err != 0 || raw.is_null() {
        return Err(anyhow!(
            "clCreateCommandQueueWithProperties failed (err={})",
            err
        ));
    }
    // Wrap the raw handle into ocl-core's CommandQueue (takes ownership of
    // the refcount, Drop releases it). Then swap via Queue's DerefMut, which
    // releases the original default-priority queue.
    let new_core =
        unsafe { ocl::core::CommandQueue::from_raw_create_ptr(raw as ocl::ffi::cl_command_queue) };
    **queue = new_core;
    log::info!(
        "OpenCL queue: CL_QUEUE_PRIORITY_KHR = {} (cl_khr_priority_hints)",
        priority
    );
    Ok(())
}

/// Emit a one-time diagnostic to stderr when the prefill flash attention
/// dispatcher routes a head_dim=128 / F16 KV workload through the new
/// `flash_attn_f32_f16` DK=128 kernel. Matches the [GQA-Attn] style
/// per-run marker so short-run debugging can confirm Qwen 2.5-1.5B
/// reaches GPU prefill instead of the CPU fallback.
fn log_prefill_dk128_once() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!(
            "[Prefill] flash_attn dispatch: head_dim=128, DType=F16, BLOCK_M=32, subgroup=64"
        );
    });
}

/// Helper function to get the OpenCL memory handle from a tensor buffer.
/// Works with both UnifiedBuffer and legacy OpenCLBuffer.
pub fn get_cl_mem(buf: &dyn Buffer) -> Result<&ocl::core::Mem> {
    // First try UnifiedBuffer
    if let Some(unified) = buf.as_any().downcast_ref::<UnifiedBuffer>() {
        return Ok(unified.cl_buffer().as_core());
    }
    // Then try legacy OpenCLBuffer
    if let Some(ocl_buf) = buf
        .as_any()
        .downcast_ref::<crate::memory::opencl::device::OpenCLBuffer>()
    {
        return Ok(ocl_buf.buffer.as_core());
    }
    // ClSubBuffer (zero-copy sub-region of a parent CL buffer via clCreateSubBuffer)
    if let Some(sb) = buf
        .as_any()
        .downcast_ref::<crate::memory::opencl::sub::ClSubBuffer>()
    {
        return Ok(sb.cl_mem_ref());
    }
    // NoshuffleWeightBuffer (SOA replacement for an AOS Q4_0 weight — holds
    // q_buf/d_buf/q_img on behalf of the tensor). `cl_mem()` returns `d_buf`;
    // the same address doubles as the noshuffle SOA registry key.
    if let Some(ns) = buf
        .as_any()
        .downcast_ref::<crate::memory::opencl::noshuffle::NoshuffleWeightBuffer>()
    {
        return Ok(ns.d_buf());
    }
    // HostPtrPoolBuffer (Direction A Stage 3 zero-copy pool slot —
    // CL_MEM_ALLOC_HOST_PTR, filled via map/memcpy/unmap before use).
    if let Some(pp) = buf
        .as_any()
        .downcast_ref::<crate::backend::opencl::host_ptr_pool_buffer::HostPtrPoolBuffer>()
    {
        return pp
            .cl_mem()
            .ok_or_else(|| anyhow!("HostPtrPoolBuffer: cl_mem is None (unexpected)"));
    }
    // Generic fallback — Buffer trait의 cl_mem() method가 Some(...)을 반환하면
    // 사용 (새 buffer type 추가 시 OCP — 본 함수에 explicit downcast 추가 불필요).
    if let Some(m) = buf.cl_mem() {
        return Ok(m);
    }
    Err(anyhow!("Buffer is not an OpenCL buffer type"))
}

/// Returns a short label identifying the concrete buffer type backing `buf`.
///
/// Covers both `get_cl_mem`-recognised GPU-resident types and the CPU/host-
/// only buffer types that commonly appear in misrouted dispatch failures —
/// the latter set is where the diagnostic value lives, since they are what
/// `get_cl_mem` rejects. Add new buffer types here as they are introduced.
pub fn buffer_kind_label(buf: &dyn Buffer) -> &'static str {
    let any = buf.as_any();
    if any.is::<UnifiedBuffer>() {
        "UnifiedBuffer"
    } else if any.is::<crate::memory::opencl::device::OpenCLBuffer>() {
        "OpenCLBuffer"
    } else if any.is::<crate::memory::opencl::sub::ClSubBuffer>() {
        "ClSubBuffer"
    } else if any.is::<crate::memory::opencl::noshuffle::NoshuffleWeightBuffer>() {
        "NoshuffleWeightBuffer"
    } else if any.is::<crate::memory::host::shared::SharedBuffer>() {
        "SharedBuffer"
    } else if any.is::<crate::memory::host::shared::SharedBufferView>() {
        "SharedBufferView"
    } else if any.is::<crate::buffer::slice::SliceBuffer>() {
        "SliceBuffer"
    } else if any.is::<crate::memory::host::mmap::MmapBuffer>() {
        "MmapBuffer"
    } else if any.is::<crate::backend::opencl::host_ptr_pool_buffer::HostPtrPoolBuffer>() {
        "HostPtrPool"
    } else {
        "Unknown"
    }
}

// Fast math flags (excluding -cl-std which is determined at runtime)
const CL_FAST_MATH_FLAGS: &str =
    "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math";

/// Build compiler options string based on device's OpenCL C version.
/// Checks CL_DEVICE_OPENCL_C_VERSION (e.g. "OpenCL C 1.2") instead of the API version,
/// since a device can report OpenCL 3.0 API while only supporting CL C 1.2.
fn build_cl_opts(device: &Device) -> String {
    let cl_c_version_str = device
        .info(ocl::core::DeviceInfo::OpenclCVersion)
        .map(|v| v.to_string())
        .unwrap_or_default();
    // Parse "OpenCL C X.Y ..." → extract major.minor
    let supports_cl_c_2 = cl_c_version_str
        .split_whitespace()
        .nth(2) // "X.Y"
        .and_then(|ver| {
            let mut parts = ver.split('.');
            let major: u32 = parts.next()?.parse().ok()?;
            let minor: u32 = parts.next()?.parse().ok()?;
            Some((major, minor) >= (2, 0))
        })
        .unwrap_or(false);
    if supports_cl_c_2 {
        format!("-cl-std=CL2.0 {}", CL_FAST_MATH_FLAGS)
    } else {
        CL_FAST_MATH_FLAGS.to_string()
    }
}

/// Cached kernel objects wrapped in a struct for interior mutability.
/// Uses Mutex to make the raw kernel pointers thread-safe.
struct KernelCache {
    kernel_mul_mat_f32_f32: CoreKernel,
    kernel_mul_mat_f16_f32: CoreKernel,
    kernel_mul_mat_q4_0_f32: CoreKernel,
    kernel_rms_norm_opt: CoreKernel,
    kernel_softmax_opt: CoreKernel,
    kernel_rope_simple: CoreKernel,
    kernel_silu_mul_simple: CoreKernel,
    kernel_gelu_tanh_mul: CoreKernel,
    kernel_add_assign_simple: CoreKernel,
    kernel_add_row_bias: CoreKernel,
    kernel_scale_simple: CoreKernel,
    kernel_get_rows_q4_0: CoreKernel,
    kernel_get_rows_f32: CoreKernel,
    kernel_get_rows_f16: CoreKernel,
    kernel_attn_gen: CoreKernel,
    kernel_cast_f32_to_f16: CoreKernel,
    kernel_attn_gen_half: CoreKernel,
    kernel_quantize_f32_to_q4_0: CoreKernel,
    kernel_kv_scatter_f32_to_f16: CoreKernel,
    /// BATCH variant: writes `seq_len` positions in one kernel launch.
    /// Used by prefill path in `transformer_layer::forward.rs` when
    /// `supports_kv_scatter_batch()` returns true. See
    /// `kernel_kv_scatter_f32_to_f16_batch` in `engine/kernels/simple_ops.cl`.
    kernel_kv_scatter_f32_to_f16_batch: CoreKernel,
    kernel_rms_norm_oop: CoreKernel,
    kernel_add_rms_norm_oop: CoreKernel,
    /// Fused 3-input merge + RMSNorm for tensor-partition layer boundary.
    /// See `kernel_fused_norm_merge` in `engine/kernels/simple_ops.cl`.
    kernel_fused_norm_merge: CoreKernel,
    /// float4 variant (dim % 4 == 0). Selected dynamically in dispatch.
    kernel_fused_norm_merge_f4: Option<CoreKernel>,
    kernel_flash_attn_f32: Option<CoreKernel>,
    /// Prefill flash attention, F32 Q/KV, head_dim=256 variant
    /// (compiled from `flash_attn_f32.cl` with -DDK=256 -DDV=256 -DBLOCK_M=16).
    /// Added to eliminate the CPU fallback for Gemma3 models (head_dim=256)
    /// whose repeated per-layer `clEnqueueReadBuffer` churn exhausts NVIDIA
    /// OpenCL driver staging resources during multi-question eval-ll.
    kernel_flash_attn_f32_dk256: Option<CoreKernel>,
    /// Prefill flash attention, head_dim=64 variant
    /// (Q=F32, KV=F16, compiled with -DDK=64 -DDV=64).
    /// Formerly `kernel_flash_attn_f32_f16` (implicit dk64); renamed to
    /// match the decode naming after introducing the dk128 variant.
    kernel_flash_attn_f32_f16_dk64: Option<CoreKernel>,
    /// Prefill flash attention, head_dim=128 variant
    /// (Q=F32, KV=F16, compiled with -DDK=128 -DDV=128).
    /// Dispatched by `flash_attention_prefill_gpu` for models like
    /// Qwen 2.5-1.5B (head_dim=128, GQA).
    kernel_flash_attn_f32_f16_dk128: Option<CoreKernel>,
    /// Decode-specialized flash attention, head_dim=64 variant
    /// (Q=F32, KV=F16, compiled with -DDK=64 -DDV=64).
    kernel_flash_attn_f32_f16_q1_dk64: Option<CoreKernel>,
    /// Decode-specialized flash attention, head_dim=128 variant
    /// (Q=F32, KV=F16, compiled with -DDK=128 -DDV=128).
    kernel_flash_attn_f32_f16_q1_dk128: Option<CoreKernel>,
    // GEMM kernels for prefill (tiled matrix multiply, M > 1)
    kernel_mul_mm_f16_f32: Option<CoreKernel>,
    kernel_mul_mm_q4_0_f32: Option<CoreKernel>,
    kernel_mul_mm_f32_f32: Option<CoreKernel>,
    // F16 GEMV N_DST=4 variant for large-N matmuls (lm_head, FFN)
    kernel_mul_mat_f16_f32_l4: Option<CoreKernel>,
    // F16 GEMV single-token decode variant (N_DST=1, 64 threads/WG).
    // Matches llama.cpp's ne11*ne12<4 path. Dispatched when m=1.
    kernel_mul_mat_f16_f32_1row: Option<CoreKernel>,
    // true when f16 kernel is the nosub fallback (1D work group, N_DST=4 rows/WG)
    f16_is_nosub: bool,
    // Q4_0 noshuffle: SOA conversion kernel (Adreno-optimized, from cvt.cl)
    kernel_cvt_q4_0_noshuffle: Option<CoreKernel>,
    /// ENG-ALG-226 / INV-140: Fused single-dispatch SOA convert + 2D transpose
    /// kernel. When `Some(_)`, `convert_q4_0_to_noshuffle()` skips the 4-step
    /// host round-trip path. `None` falls back to the legacy path.
    /// Source: `engine/kernels/cvt_q4_0_noshuffle_fused.cl`.
    kernel_cvt_q4_0_noshuffle_fused: Option<CoreKernel>,
    /// Adreno-optimized Q4_0 GEMM for prefill (llama.cpp mul_mat_Ab_Bi_8x4).
    /// Consumes the same SOA q_buf / d_buf produced by `convert_q4_0_to_noshuffle`.
    /// Activations must be transposed to F16 (K × N_padded) via
    /// `kernel_transpose_32_16` before dispatch.
    kernel_mul_mat_ab_bi_8x4: Option<CoreKernel>,
    /// F32 → F16 activation transpose with N-padding to multiple of 8,
    /// used exclusively by the Adreno Q4_0 GEMM fast path.
    kernel_transpose_32_16: Option<CoreKernel>,
}

// SAFETY: OpenCL kernel objects are thread-safe for clSetKernelArg + clEnqueueNDRangeKernel
// when protected by a mutex. The underlying cl_kernel is not mutated by these operations,
// only the kernel arguments are set before each call.
unsafe impl Send for KernelCache {}
unsafe impl Sync for KernelCache {}

/// SOA layout entry for a Q4_0 weight tensor converted to noshuffle format.
/// Stored in the noshuffle registry, keyed by the original cl_mem pointer.
pub struct NoshuffleSoaEntry {
    /// SOA nibbles buffer (transposed, ushort-level column-major)
    pub q_buf: ocl::core::Mem,
    /// SOA scales buffer (transposed, half-level column-major)
    pub d_buf: ocl::core::Mem,
    /// image1d_buffer_t wrapping q_buf (R32UI format) for Adreno TP cache.
    /// None when image creation fails (e.g. CL_DEVICE_IMAGE_MAX_BUFFER_SIZE exceeded,
    /// image1d_buffer_t not supported). Falls back to standard Q4_0 matmul path.
    pub q_img: Option<ocl::core::Mem>,
    /// K dimension (elements per row)
    pub ne00: usize,
    /// M dimension (number of rows)
    pub ne01: usize,
}

/// OpenCL Backend with cached kernel objects for performance.
/// Kernels are created once during initialization and reused across all calls.
pub struct OpenCLBackend {
    pub context: Context,
    pub queue: Queue,
    pub device: Device,
    // Programs
    pub program: Program,
    pub simple_ops_program: Program,
    pub q4_0_program: Program,
    pub f16_program: Program,
    pub quant_q4_0_program: Program,
    pub get_rows_program: Program,

    // Flash attention programs (optional — compiled with head_dim-specific defines)
    pub flash_attn_f32_program: Option<Program>,
    /// Flash attention F32-Q / F32-KV program, head_dim=256 variant.
    /// Used by prefill (`flash_attn_f32` kernel) for Gemma3 1B / 4B.
    /// On NVIDIA OpenCL this replaces the CPU fallback whose per-layer
    /// `clEnqueueReadBuffer` calls exhaust driver staging resources.
    pub flash_attn_f32_program_dk256: Option<Program>,
    /// Flash attention F32-Q / F16-KV program, head_dim=64 variant.
    /// Used by prefill (`flash_attn_f32_f16` kernel) and decode
    /// (`flash_attn_f32_f16_q1` kernel) at head_dim=64.
    pub flash_attn_f32_f16_program_dk64: Option<Program>,
    /// Flash attention F32-Q / F16-KV program, head_dim=128 variant.
    /// Decode-only (no prefill dispatcher for this DK). Dispatched
    /// by `flash_attention_decode_gpu` for models with head_dim=128
    /// (e.g. Qwen 2.5-1.5B).
    pub flash_attn_f32_f16_program_dk128: Option<Program>,

    // F16 GEMV N_DST=4 for large-N decode (lm_head, FFN)
    pub f16_l4_program: Option<Program>,

    // F16 GEMV single-token decode (N_DST=1). llama.cpp parity.
    pub f16_1row_program: Option<Program>,

    // GEMM programs for prefill (tiled matmul, optional — fallback to GEMV)
    pub gemm_f16_program: Option<Program>,
    pub gemm_q4_0_program: Option<Program>,
    pub gemm_f32_program: Option<Program>,

    // Score-only attention program (Q*K^T + softmax, no V multiply)
    pub attention_scores_program: Option<Program>,

    // Q4_0 noshuffle programs (optional — Adreno-optimized SOA GEMV)
    pub cvt_noshuffle_program: Option<Program>,
    /// ENG-ALG-226: Fused SOA convert + transpose program (single dispatch).
    /// `None` ⇒ legacy 4-step host-transpose fallback in
    /// `convert_q4_0_to_noshuffle()`.
    pub cvt_noshuffle_fused_program: Option<Program>,
    // gemv_noshuffle programs are dimension-specific (LINE_STRIDE_A, BLOCK_STRIDE_A).
    // Built lazily per weight dimension via convert_q4_0_to_noshuffle().

    // Adreno Q4_0 GEMM path (llama.cpp-style): `mul_mat_Ab_Bi_8x4` + F32→F16
    // activation transpose from `transpose.cl`. Optional — falls back to
    // `mul_mm_q4_0_f32_l4_lm` when either program fails to compile.
    pub gemm_ab_bi_program: Option<Program>,
    pub transpose_program: Option<Program>,

    // Cached kernels — inference is single-threaded, no lock needed.
    // UnsafeCell avoids Mutex overhead (~170us/lock on Adreno).
    kernels: UnsafeCell<KernelCache>,

    // true on UMA devices (Adreno, Mali): CL_MEM_ALLOC_HOST_PTR for zero-copy.
    // false on discrete GPUs (NVIDIA): device-only buffers for correct behavior.
    pub use_zero_copy: bool,

    // Pre-allocated 1-element dummy buffer for attention_gen when scores_out is None.
    // Avoids per-call GPU buffer allocation (16 layers x every token).
    dummy_score_buf: ocl::core::Mem,

    // GPU-side attention score accumulator (optional).
    // When active, attention_gen writes to a persistent GPU buffer and
    // reduce_layer aggregates scores without CPU readback.
    // UnsafeCell follows the same pattern as `kernels` — single-threaded access.
    gpu_score_acc: UnsafeCell<Option<gpu_score::GpuScoreAccumulator>>,

    // Cached compiler options for building additional programs (e.g., score_reduce.cl).
    pub cl_opts: String,

    // CL_DEVICE_MAX_MEM_ALLOC_SIZE: maximum single buffer allocation (bytes).
    max_mem_alloc_size: usize,

    // Cached GEMV noshuffle kernels, keyed by (ne01,) dimensions.
    // Each unique dimension pair requires a different compile-time define set.
    // UnsafeCell: single-threaded access like other kernel caches.
    gemv_noshuffle_cache: UnsafeCell<HashMap<usize, (Program, CoreKernel)>>,

    // Q4_0 noshuffle SOA registry: maps original weight cl_mem pointer (as usize)
    // to pre-converted SOA buffers. Populated by prepare_noshuffle_buffers() at load time.
    // matmul_q4_0() auto-dispatches to noshuffle GEMV when a lookup succeeds.
    // UnsafeCell: single-threaded access (same pattern as kernels and gemv_noshuffle_cache).
    noshuffle_soa_registry: UnsafeCell<HashMap<usize, NoshuffleSoaEntry>>,

    // Persistent scratch for `matmul_q4_0_gemm_adreno`: F16 activation with N
    // padded to a multiple of 8. Grown on demand to max(K * N_padded / 2) bytes
    // across prefill. Single-threaded inference access, same pattern as other
    // UnsafeCell caches. Stored as (buffer, rgba16f_write_image,
    // rgba16f_read_image, capacity_bytes, image_texel_count). Reusing the same
    // `cl_mem` images across 196 Q4_0 matmul calls saves ~400 µs/call of
    // driver bookkeeping (Adreno clCreateImage on CL_MEM_IMAGE1D_BUFFER is
    // non-trivial).
    #[allow(clippy::type_complexity)]
    gemm_act_trans_buf:
        UnsafeCell<Option<(ocl::core::Mem, ocl::core::Mem, ocl::core::Mem, usize, usize)>>,

    // Per-layer cache key for the transposed activation above: `(a_buf_ptr,
    // m, k)`. When a matmul reuses the same activation pointer and shape
    // (e.g. Q/K/V all consume the post-RMSNorm residual within one layer),
    // the transpose pass is skipped and only the GEMM dispatch runs.
    // Invalidated whenever `gemm_act_trans_buf` is (re)allocated.
    gemm_act_trans_key: UnsafeCell<Option<(usize, usize, usize)>>,

    // ── Event-based per-op profiling (--profile-events) ──────────────────
    // When true, the queue was created with CL_QUEUE_PROFILING_ENABLE and all
    // enqueue_kernel_labeled() calls capture an event + label. Decode-step
    // flush aggregates (End-Start) ns per label. Unlike the legacy --profile
    // mode, no per-op clFinish is required.
    pub profile_events_enabled: bool,
    // Captured (label, event) pairs, cleared on each flush_and_aggregate_profile().
    // UnsafeCell: single-threaded inference access.
    profile_events: UnsafeCell<Vec<(&'static str, ocl::core::Event)>>,
    // Accumulated GPU ns per label (total over the whole run).
    profile_accum: UnsafeCell<HashMap<&'static str, u64>>,
    // Optional caller-provided label hint. When Some, it overrides the
    // static label passed to enqueue_kernel_labeled(). Used by forward_gen
    // and plan.rs to distinguish matmul_qkv / matmul_wo / matmul_ffn / lm_head
    // which all dispatch the same underlying GEMV/GEMM kernel.
    op_label_hint: UnsafeCell<Option<&'static str>>,

    // MSG-068 / MGR-DAT-076 (Phase 2): Engine process GPU self-utilization
    // meter. `flush_and_aggregate_profile()` pushes per-event (end - start) ns
    // into this accumulator when present. `CommandExecutor::send_heartbeat`
    // then drains it to compute `self_gpu_pct`. Populated only when the caller
    // opts in via `new_with_profile_events(true)`. Arc so both backend and
    // executor can hold a reference without lifetime gymnastics.
    gpu_self_meter: Option<Arc<OpenClEventGpuMeter>>,

    // ── WSWAP-5-TBT-DIAG: cl_mem allocation diagnostics ──────────────────
    // Counts cl_mem allocations per category (weight_q4_aos, weight_q4_soa_q,
    // weight_q4_soa_d, weight_q4_soa_img, auf_placeholder, kv_cache,
    // activation, ...) so we can compare the Q4 baseline cl_mem footprint
    // against the AUF SOA bypass swap path. Activated by env var
    // `LLMRS_CL_MEM_DIAG=1`. When disabled, all hooks are zero-cost
    // (single bool check + early return).
    cl_mem_diag_enabled: bool,
    cl_mem_diag: UnsafeCell<HashMap<&'static str, ClMemDiagBucket>>,

    // ── LISWAP-2 prototype: async H2D weight upload queue ────────────────
    // Lazily-created secondary command queue dedicated to the async swap
    // dispatcher's `enqueue_write_async` path. Created on first use so the
    // cost is paid only when `--swap-async-dispatch` is actually enabled.
    // Env-gate: `LLMRS_OPENCL_TRANSFER_QUEUE=0|false|off` disables it (then
    // `enqueue_write_async` falls through to the default sync copy path).
    // Plan: .claude/plans/compiled-chasing-hopper.md (Phase 1).
    transfer_queue: std::sync::OnceLock<Queue>,

    // ── LISWAP-3 prototype: ALLOC_HOST_PTR pool for swap (Direction A) ───
    // Lazily-created pool of pre-allocated `CL_MEM_ALLOC_HOST_PTR` cl_mem
    // slots dedicated to the production swap path's zero-copy upload (vs.
    // the staging copy used by `copy_weight_from`). Created on first use
    // so the up-front allocation cost is paid only when
    // `--swap-zero-copy` + `LLMRS_OPENCL_HOST_PTR_POOL=1` are both on.
    // Env-gate: `LLMRS_OPENCL_HOST_PTR_POOL=1|true|on` enables (default OFF —
    // measurement-driven decision pending). Plan: compiled-chasing-hopper.md
    // (Direction A track, Stage 3).
    host_ptr_pool: std::sync::OnceLock<Option<Arc<host_ptr_pool::HostPtrPool>>>,

    // ── LISWAP multi-context experiment: secondary cl_context for swap ───
    // Hypothesis: Adreno's in-order driver scheduler may serialize swap
    // Map/Unmap and forward kernels even on separate command queues if both
    // belong to the same `cl_context`. Allocating a second `cl_context` on
    // the same `cl_device_id` exposes whether driver-level queue scheduling
    // is context-scoped (then swap and forward are isolated) or
    // device-scoped (then this changes nothing). The DMA-BUF FD is shared
    // between both contexts so the underlying physical memory is identical.
    //
    // Env-gate: `LLMRS_OPENCL_SWAP_CONTEXT=1|true|on` enables (default OFF).
    // Both lazy-init via `swap_context_or_init()` / `swap_queue_or_init()`.
    swap_context: std::sync::OnceLock<Context>,
    swap_queue: std::sync::OnceLock<Queue>,

    // B-5b Phase 2 Stage 1: CPU companion backend injected at construction
    // time. Stage 2 will use this to remove the "GPU backend constructs a
    // fresh CpuBackend on the fly" pattern in fallback paths. For Stage 1
    // the field is owned but unused — Backend::cpu_companion() returns
    // `&*self.cpu_companion`.
    cpu_companion: Arc<dyn Backend>,

    // Sprint 2a Phase 2 (ENG-RPCMEM-020 ~ 024): backend-agnostic rpcmem
    // allocator. `Some` iff `--opencl-rpcmem` was active at construction time
    // **and** eager dlopen succeeded (Android only). `None` = legacy path
    // (UnifiedBuffer KV + GGUF mmap secondary). Exposed via
    // `Backend::get_extension(EXT_RPCMEM_ALLOCATOR)` so the precision swap
    // loader (`RpcmemSecondaryStore`) shares the same instance (INV-RPCMEM-002).
    rpcmem_allocator: Option<Arc<crate::memory::rpcmem::allocator::RpcmemAllocator>>,
}

/// Per-category cl_mem allocation bucket (WSWAP-5-TBT-DIAG).
#[derive(Debug, Default, Clone)]
pub struct ClMemDiagBucket {
    /// Number of `clCreateBuffer` / `clCreateImage` calls observed.
    pub count: usize,
    /// Sum of `size` arguments passed to those calls (bytes).
    pub total_bytes: usize,
    /// Number of explicit release events observed (currently only registry
    /// invalidations — destructor releases are not tracked).
    pub releases: usize,
    /// Sum of bytes released through the `releases` channel.
    pub released_bytes: usize,
}

// SAFETY: OpenCLBackend is only accessed from the inference thread.
// The UnsafeCell is safe because we guarantee single-threaded access
// during kernel dispatch (same as llama.cpp's lock-free approach).
unsafe impl Send for OpenCLBackend {}
unsafe impl Sync for OpenCLBackend {}

// Manual Debug impl
impl std::fmt::Debug for OpenCLBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenCLBackend")
            .field("device", &self.device)
            .finish()
    }
}

impl OpenCLBackend {
    pub fn new() -> Result<Self> {
        Self::new_with_profile_events(false)
    }

    /// Sprint 2a Phase 2 (ENG-RPCMEM-020 ~ 024): construct with both profile-
    /// events and `--opencl-rpcmem` toggles. When `opencl_rpcmem` is true the
    /// constructor eagerly attempts `RpcmemAllocator::new()`; on failure the
    /// flag is silently demoted (stderr warning 1회) so the host build keeps
    /// running with the standard UnifiedBuffer KV path (INV-RPCMEM-001/003).
    ///
    /// The eager init choice matches arch §3.1 — lazy init at the first KV
    /// alloc would surface dlopen failures inside the hot path, increasing
    /// debugging cost. Eager dlopen costs only 3 symbol lookups (~few ms).
    pub fn new_with_options(profile_events_enabled: bool, opencl_rpcmem: bool) -> Result<Self> {
        let mut backend = Self::new_with_profile_events(profile_events_enabled)?;
        if opencl_rpcmem {
            match crate::memory::rpcmem::allocator::RpcmemAllocator::new() {
                Ok(alloc) => {
                    backend.rpcmem_allocator = Some(Arc::new(alloc));
                    eprintln!(
                        "[OpenCL] --opencl-rpcmem: RpcmemAllocator init OK (libcdsprpc.so dlopen 성공)"
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[OpenCL] --opencl-rpcmem 강등: RpcmemAllocator init 실패 ({e}). \
                         UnifiedBuffer/GGUF mmap path 로 진행."
                    );
                    // backend.rpcmem_allocator already None from new path.
                }
            }
        }
        Ok(backend)
    }

    /// Read-only accessor — `Some` iff `--opencl-rpcmem` 가 활성이고 dlopen 성공.
    /// Sprint 2a Phase 2 의 OpenCLMemory::new_with_rpcmem 와 secondary loader
    /// 가 본 메서드로 동일 `Arc` 인스턴스를 공유한다 (INV-RPCMEM-002).
    pub fn rpcmem_allocator(
        &self,
    ) -> Option<Arc<crate::memory::rpcmem::allocator::RpcmemAllocator>> {
        self.rpcmem_allocator.as_ref().map(Arc::clone)
    }

    /// Build an OpenCLBackend with optional per-op event profiling.
    ///
    /// When `profile_events_enabled` is true, the command queue is created
    /// with `CL_QUEUE_PROFILING_ENABLE`, and every kernel dispatch that
    /// goes through `enqueue_kernel_labeled()` records an `ocl::core::Event`
    /// plus label. The decode loop periodically calls
    /// `flush_and_aggregate_profile()` which reads `End - Start` nanoseconds
    /// from each event and accumulates them into `profile_accum`.
    ///
    /// This is the replacement for the legacy `--profile` mode, which
    /// inserted two `clFinish()` calls per op and inflated decode ms/tok
    /// by ~54 ms on Adreno. Event-based measurement has near-zero CPU
    /// overhead because profiling info is collected by the GPU itself.
    pub fn new_with_profile_events(profile_events_enabled: bool) -> Result<Self> {
        // --- Device type selection via OCL_DEVICE_TYPE env var ---
        let device_type = match std::env::var("OCL_DEVICE_TYPE").as_deref().unwrap_or("gpu") {
            "cpu" => Some(flags::DEVICE_TYPE_CPU),
            "all" => None,
            _ => Some(flags::DEVICE_TYPE_GPU),
        };

        // --- Enumerate platforms (panic-free) ---
        // `ocl::Platform::list()` / `Platform::default()` call `clGetPlatformIDs`
        // and *panic* (via `.expect()`) when the ICD reports no platforms — e.g.
        // a CI host with the OpenCL loader installed but no driver registered.
        // Query the raw IDs so a platform-less host yields `Err` (callers — and
        // the GPU integration tests — then skip) instead of aborting the process.
        let platforms = Platform::list_from_core(
            ocl::core::get_platform_ids()
                .map_err(|e| anyhow!("No OpenCL platform available: {e}"))?,
        );
        if platforms.is_empty() {
            return Err(anyhow!(
                "No OpenCL platform available (clGetPlatformIDs returned 0)"
            ));
        }

        // --- Platform selection via OCL_PLATFORM env var ---
        let platform = if let Ok(name) = std::env::var("OCL_PLATFORM") {
            let name_lower = name.to_lowercase();
            platforms
                .iter()
                .copied()
                .find(|p| {
                    p.name()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains(&name_lower)
                })
                .ok_or_else(|| anyhow!("No OpenCL platform matching '{}'", name))?
        } else {
            // 기본: 요청 device type 을 가진 플랫폼 우선. 첫 플랫폼(`platforms[0]`)은
            // POCL(CPU)-first 호스트에서 GPU 플랫폼(NVIDIA)을 가리지 못한다. 스캔 중 Err 는
            // "해당 없음"으로 간주(POCL 의 clGetDeviceIDs 가 병렬 질의에서 간헐 Err).
            // 단일 플랫폼 디바이스(Adreno/Jetson)에선 기존 선택과 동일.
            device_type
                .and_then(|dtype| {
                    platforms.iter().copied().find(|p| {
                        Device::list(*p, Some(dtype))
                            .map(|d| !d.is_empty())
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(platforms[0])
        };

        let device = if let Some(dtype) = device_type {
            Device::list(platform, Some(dtype))?
                .into_iter()
                .next()
                .unwrap_or_else(|| Device::first(platform).expect("No OpenCL devices found"))
        } else {
            Device::first(platform)?
        };

        // --- Log platform/device info ---
        let platform_name = platform.name().unwrap_or_else(|_| "unknown".into());
        let device_name = device.name().unwrap_or_else(|_| "unknown".into());
        let device_version = device
            .version()
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "unknown".into());
        let cl_c_version = device
            .info(ocl::core::DeviceInfo::OpenclCVersion)
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "unknown".into());
        let extensions = device
            .info(ocl::core::DeviceInfo::Extensions)
            .map(|v| v.to_string())
            .unwrap_or_default();
        let has_khr_subgroups = extensions.contains("cl_khr_subgroups");
        let has_intel_subgroups = extensions.contains("cl_intel_subgroups");

        log::info!("OpenCL Platform: {}", platform_name);
        log::info!(
            "OpenCL Device: {} (API {}, CL C: {})",
            device_name,
            device_version,
            cl_c_version
        );
        log::info!(
            "Subgroup support: cl_khr_subgroups={}, cl_intel_subgroups={}",
            has_khr_subgroups,
            has_intel_subgroups
        );
        // Log memory limits
        if let Ok(global_mem) = device.info(ocl::core::DeviceInfo::GlobalMemSize) {
            log::info!(
                "GPU Global Memory: {} MB",
                global_mem.to_string().trim().parse::<u64>().unwrap_or(0) / (1024 * 1024)
            );
        }
        let max_mem_alloc_size: usize = device
            .info(ocl::core::DeviceInfo::MaxMemAllocSize)
            .ok()
            .and_then(|v| v.to_string().trim().parse::<u64>().ok())
            .unwrap_or(1024 * 1024 * 1024) as usize; // fallback 1GB
        log::info!(
            "GPU Max Alloc Size: {} MB",
            max_mem_alloc_size / (1024 * 1024)
        );

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        let queue_props = if profile_events_enabled {
            log::info!("OpenCL queue: CL_QUEUE_PROFILING_ENABLE (event-based per-op profiling)");
            Some(flags::QUEUE_PROFILING_ENABLE)
        } else {
            None
        };
        // Optional out-of-order execution: env-gated via LLMRS_OPENCL_OOO_QUEUE=1.
        // When enabled, the driver schedules commands by data dependency
        // rather than submission order. swap H2D and forward kernels that
        // touch different cl_mems can theoretically run in parallel.
        let queue_props = if std::env::var("LLMRS_OPENCL_OOO_QUEUE").is_ok() {
            log::info!(
                "OpenCL queue: CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE \
                 (LLMRS_OPENCL_OOO_QUEUE=1)"
            );
            Some(
                queue_props.unwrap_or(flags::CommandQueueProperties::empty())
                    | flags::QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            )
        } else {
            queue_props
        };
        let mut queue = Queue::new(&context, device, queue_props)?;
        // Optional: replace queue with a priority-hinted one via
        // `clCreateCommandQueueWithProperties` + `cl_khr_priority_hints`.
        // `OCL_QUEUE_PRIORITY` env var: "low" | "normal" | "high".
        // Unsupported priority or missing extension → warn + keep default queue.
        if let Ok(pri) = std::env::var("OCL_QUEUE_PRIORITY") {
            let pri_low = pri.to_lowercase();
            if pri_low != "normal"
                && let Err(e) = try_apply_queue_priority(
                    &mut queue,
                    &context,
                    device,
                    queue_props,
                    &extensions,
                    &pri_low,
                )
            {
                log::warn!("OCL_QUEUE_PRIORITY={}: {} — using default priority", pri, e);
            }
        }

        // --- Build compiler options based on device CL version ---
        let cl_opts = build_cl_opts(&device);
        log::info!("OpenCL compiler options: {}", cl_opts);

        // === Load kernel programs with fallback ===

        // Matmul F32: try original, fallback to nosub, then dummy
        let matmul_src = include_str!("../../kernels/mul_mv_f32_f32.cl");
        let matmul_fallback_src = include_str!("../../kernels/fallback/mul_mv_f32_f32_nosub.cl");
        let program = match Program::builder()
            .devices(device)
            .src(matmul_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                log::warn!("mul_mv_f32_f32.cl failed: {}. Trying fallback.", e);
                match Program::builder()
                    .devices(device)
                    .src(matmul_fallback_src)
                    .cmplr_opt(&cl_opts)
                    .build(&context)
                {
                    Ok(p) => {
                        log::info!("Using fallback/mul_mv_f32_f32_nosub.cl");
                        p
                    }
                    Err(e2) => {
                        log::warn!("Fallback matmul also failed: {}. Using dummy.", e2);
                        Program::builder()
                            .devices(device)
                            .src("__kernel void kernel_mul_mat_f32_f32() {}")
                            .build(&context)?
                    }
                }
            }
        };

        // Simple ops: try original, fallback to nosub file
        let simple_ops_src = include_str!("../../kernels/simple_ops.cl");
        let simple_ops_fallback_src = include_str!("../../kernels/fallback/simple_ops_nosub.cl");
        let simple_ops_program = match Program::builder()
            .devices(device)
            .src(simple_ops_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("simple_ops.cl compiled (subgroup path)");
                p
            }
            Err(e) => {
                log::warn!(
                    "simple_ops.cl failed: {}. Using fallback/simple_ops_nosub.cl",
                    e
                );
                Program::builder()
                    .devices(device)
                    .src(simple_ops_fallback_src)
                    .cmplr_opt(&cl_opts)
                    .build(&context)?
            }
        };

        // Q4_0 matmul: try original, fallback to nosub, then dummy
        let q4_0_src = include_str!("../../kernels/mul_mv_q4_0_f32.cl");
        let q4_0_fallback_src = include_str!("../../kernels/fallback/mul_mv_q4_0_f32_nosub.cl");
        let q4_0_program = match Program::builder()
            .devices(device)
            .src(q4_0_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                log::warn!("mul_mv_q4_0_f32.cl failed: {}. Trying fallback.", e);
                match Program::builder()
                    .devices(device)
                    .src(q4_0_fallback_src)
                    .cmplr_opt(&cl_opts)
                    .build(&context)
                {
                    Ok(p) => {
                        log::info!("Using fallback/mul_mv_q4_0_f32_nosub.cl");
                        p
                    }
                    Err(e2) => {
                        log::warn!("Fallback Q4_0 also failed: {}. Using dummy.", e2);
                        Program::builder()
                            .devices(device)
                            .src("__kernel void kernel_mul_mat_q4_0_f32() {}")
                            .build(&context)?
                    }
                }
            }
        };

        // Get rows: try original, fallback to dummy
        let get_rows_src = include_str!("../../kernels/get_rows.cl");
        let get_rows_program = match Program::builder()
            .devices(device)
            .src(get_rows_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                log::warn!("get_rows.cl failed: {}. Using dummy.", e);
                Program::builder()
                    .devices(device)
                    .src("__kernel void kernel_get_rows_q4_0() {} __kernel void kernel_get_rows_f32() {} __kernel void kernel_get_rows_f16() {}")
                    .build(&context)?
            }
        };

        // Quantize Q4_0: try original, fallback to dummy
        let q4_quant_src = include_str!("../../kernels/quantize_q4_0.cl");
        let quant_q4_0_program = match Program::builder()
            .devices(device)
            .src(q4_quant_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                log::warn!("quantize_q4_0.cl failed: {}. Using dummy.", e);
                Program::builder()
                    .devices(device)
                    .src("__kernel void kernel_quantize_f32_to_q4_0() {}")
                    .build(&context)?
            }
        };

        // F16 GEMV: two kernel variants with different dispatch geometries.
        //
        // 1. `mul_mv_f16_f32.cl` — 4-wave K-split: local=[64,4,1], N_DST=2, 128 rows/WG.
        //    4 subgroups split K dimension in parallel, partial sums reduced via local mem.
        //    Faster on Adreno/Mali due to better parallelism and activation reuse.
        //
        // 2. `fallback/mul_mv_f16_f32_nosub.cl` — subgroup-reduce: local=[64,1,1], N_DST=4, 4 rows/WG.
        //    Single subgroup per WG, uses sub_group_reduce_add(). Slower but works on any
        //    device with subgroup ops (or falls through to tree reduction).
        //
        // `f16_is_nosub` tracks which variant was loaded so dispatch code uses the
        // correct local/global geometry. Historically both paths shared a single
        // (nosub) dispatch, which silently corrupted the 4-wave variant — the
        // "broken on Adreno 830" story was a dispatch bug, not a kernel bug.
        let f16_src = include_str!("../../kernels/mul_mv_f16_f32.cl");
        let f16_fallback_src = include_str!("../../kernels/fallback/mul_mv_f16_f32_nosub.cl");
        let f16_is_nosub: bool;
        // 2026-04-22: nosub (single-SG, N_DST=4) is DEFAULT on Adreno 830 —
        // matches llama.cpp's design and measured −33% TBT at n=1024 (Qwen F16)
        // vs the 4-wave K-split variant. Opt back into 4-wave with
        // LLMRS_F16_GEMV_USE_4WAVE=1 (for A/B testing or other devices).
        let use_4wave = std::env::var("LLMRS_F16_GEMV_USE_4WAVE")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let primary_src = if use_4wave { f16_src } else { f16_fallback_src };
        let primary_tag = if use_4wave {
            "4-wave K-split, N_DST=2 (opt-in)"
        } else {
            "nosub single-SG, N_DST=4 (default)"
        };
        let f16_program = match Program::builder()
            .devices(device)
            .src(primary_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mv_f16_f32 compiled ({})", primary_tag);
                f16_is_nosub = !use_4wave;
                p
            }
            Err(e) => {
                log::warn!(
                    "primary F16 kernel ({}) failed: {}. Trying the other variant.",
                    primary_tag,
                    e
                );
                let alt_src = if use_4wave { f16_fallback_src } else { f16_src };
                let alt_is_nosub = use_4wave;
                match Program::builder()
                    .devices(device)
                    .src(alt_src)
                    .cmplr_opt(&cl_opts)
                    .build(&context)
                {
                    Ok(p) => {
                        log::info!("F16 GEMV compiled (alt path)");
                        f16_is_nosub = alt_is_nosub;
                        p
                    }
                    Err(e2) => {
                        log::warn!("Both F16 kernels failed: {}. Using dummy stub.", e2);
                        f16_is_nosub = true;
                        Program::builder()
                            .devices(device)
                            .src("__kernel void kernel_mul_mat_f16_f32() {}")
                            .build(&context)?
                    }
                }
            }
        };

        // F16 GEMV N_DST=4 variant for large-N decode (lm_head, FFN)
        let f16_l4_src = include_str!("../../kernels/mul_mv_f16_f32_l4.cl");
        let f16_l4_program = match Program::builder()
            .devices(device)
            .src(f16_l4_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mv_f16_f32_l4.cl compiled (GEMV N_DST=4 for large N)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "mul_mv_f16_f32_l4.cl failed: {}. Large-N will use N_DST=2 fallback.",
                    e
                );
                None
            }
        };

        // F16 GEMV 1row variant — matches llama.cpp's single-token decode path.
        // Activation ne11=1 case: 1 output row per WG, 64-lane subgroup reduce,
        // no local memory. llama.cpp selects this when ne11*ne12 < 4.
        // Tested 2026-04-22 to close the remaining n=2000 gap vs llama.cpp.
        let f16_1row_src = include_str!("../../kernels/mul_mv_f16_f32_1row_simple.cl");
        let f16_1row_program = match Program::builder()
            .devices(device)
            .src(f16_1row_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mv_f16_f32_1row.cl compiled (single-token decode)");
                Some(p)
            }
            Err(e) => {
                log::warn!("mul_mv_f16_f32_1row.cl failed: {}. Fallback to N_DST=4.", e);
                None
            }
        };

        // Flash attention kernels — compiled with head_dim-specific defines.
        // Each DK variant is a separate program because DK is a compile-time
        // constant (q_priv[DK_VEC] / o_acc[DV_VEC] are private arrays).
        let flash_attn_src = include_str!("../../kernels/flash_attn_f32_f16.cl");
        let flash_attn_f32_src = include_str!("../../kernels/flash_attn_f32.cl");

        // DK=64 (Llama 3.2 1B, other head_dim=64 models)
        let flash_attn_dk64_defines = "-DDK=64 -DDV=64 -DBLOCK_M=64 -DBLOCK_N=32";
        let flash_attn_f32_opts_dk64 = format!("{} {}", cl_opts, flash_attn_dk64_defines);

        let flash_attn_f32_program = match Program::builder()
            .devices(device)
            .src(flash_attn_f32_src)
            .cmplr_opt(&flash_attn_f32_opts_dk64)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32.cl compiled (BLOCK_M=64, BLOCK_N=32, DK=64)");
                Some(p)
            }
            Err(e) => {
                log::warn!("flash_attn_f32.cl failed: {}. GPU prefill F32 disabled.", e);
                None
            }
        };

        let flash_attn_f32_f16_program_dk64 = match Program::builder()
            .devices(device)
            .src(flash_attn_src)
            .cmplr_opt(&flash_attn_f32_opts_dk64)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=64)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "flash_attn_f32_f16.cl DK=64 failed: {}. GPU flash F16 KV disabled for DK=64.",
                    e
                );
                None
            }
        };

        // DK=128 (Qwen 2.5-1.5B). Same source file, different macros.
        // Supports both prefill (`flash_attn_f32_f16`) and decode
        // (`flash_attn_f32_f16_q1`) kernels.
        let flash_attn_dk128_defines = "-DDK=128 -DDV=128 -DBLOCK_M=32 -DBLOCK_N=32";
        let flash_attn_f32_opts_dk128 = format!("{} {}", cl_opts, flash_attn_dk128_defines);

        let flash_attn_f32_f16_program_dk128 = match Program::builder()
            .devices(device)
            .src(flash_attn_src)
            .cmplr_opt(&flash_attn_f32_opts_dk128)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=128, BLOCK_M=32)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "flash_attn_f32_f16.cl DK=128 failed: {}. GPU flash F16 KV disabled for DK=128 (Qwen prefill + decode fallback).",
                    e
                );
                None
            }
        };

        // DK=256 (Gemma3 1B / 4B). Built from the same F32 kernel source; only
        // the compile-time DK/DV/BLOCK_M/BLOCK_N constants differ.
        // * BLOCK_M=32: WG_SIZE = BLOCK_M = 32 matches NVIDIA warp size and
        //   keeps per-thread register pressure manageable for the
        //   q_priv[DK_VEC=64] + o_acc[DV_VEC=64] private arrays.
        // * BLOCK_N=16: 2 × 16 × 64 × sizeof(float4) = 32 KB of shared memory,
        //   under the 48 KB CUDA limit. BLOCK_N=32 overflows to 64 KB (ptxas
        //   "too much shared data"), BLOCK_N=8 reduces local tile reuse and
        //   showed numerical drift in long-prefill production runs.
        let flash_attn_dk256_defines = "-DDK=256 -DDV=256 -DBLOCK_M=32 -DBLOCK_N=16";
        let flash_attn_f32_opts_dk256 = format!("{} {}", cl_opts, flash_attn_dk256_defines);

        let flash_attn_f32_program_dk256 = match Program::builder()
            .devices(device)
            .src(flash_attn_f32_src)
            .cmplr_opt(&flash_attn_f32_opts_dk256)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32.cl compiled (BLOCK_M=16, BLOCK_N=32, DK=256)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "flash_attn_f32.cl DK=256 failed: {}. GPU prefill F32 head_dim=256 disabled (Gemma3 falls back to CPU attention).",
                    e
                );
                None
            }
        };

        // GEMM kernels for prefill — tiled matmul (BM=64, BN=64, BK=16)
        let gemm_f16_src = include_str!("../../kernels/mul_mm_f16_f32_l4_lm.cl");
        let gemm_f16_program = match Program::builder()
            .devices(device)
            .src(gemm_f16_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mm_f16_f32_l4_lm.cl compiled (GEMM prefill F16)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "mul_mm_f16_f32_l4_lm.cl failed: {}. Prefill will use GEMV.",
                    e
                );
                None
            }
        };

        let gemm_f32_src = include_str!("../../kernels/mul_mm_f32_f32_l4_lm.cl");
        let gemm_f32_program = match Program::builder()
            .devices(device)
            .src(gemm_f32_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mm_f32_f32_l4_lm.cl compiled (GEMM prefill F32)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "mul_mm_f32_f32_l4_lm.cl failed: {}. Prefill will use GEMV.",
                    e
                );
                None
            }
        };

        // Q4_0 tiled GEMM for prefill (interleaved BlockQ4_0 layout).
        let gemm_q4_0_src = include_str!("../../kernels/mul_mm_q4_0_f32_l4_lm.cl");
        let gemm_q4_0_program = match Program::builder()
            .devices(device)
            .src(gemm_q4_0_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mm_q4_0_f32_l4_lm.cl compiled (Q4_0 GEMM for prefill)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "mul_mm_q4_0_f32_l4_lm.cl failed: {}. Q4_0 prefill will fall back to GEMV.",
                    e
                );
                None
            }
        };

        // Score-only attention kernel (Q*K^T + softmax, no V multiply)
        let attn_scores_src = include_str!("../../kernels/attention_scores.cl");
        let attention_scores_program = Program::builder()
            .devices(device)
            .src(attn_scores_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
            .ok();
        if attention_scores_program.is_some() {
            log::info!("attention_scores.cl compiled (score-only decode kernel)");
        } else {
            log::warn!("attention_scores.cl failed to compile. Score-only kernel disabled.");
        }

        // Q4_0 noshuffle SOA conversion kernel (cvt.cl already contains the kernel)
        let cvt_noshuffle_src = include_str!("../../kernels/cvt.cl");
        let cvt_noshuffle_program = Program::builder()
            .devices(device)
            .src(cvt_noshuffle_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
            .ok();
        if cvt_noshuffle_program.is_some() {
            log::info!("cvt.cl compiled (Q4_0 noshuffle SOA conversion kernel)");
        } else {
            log::warn!("cvt.cl failed to compile. Q4_0 noshuffle disabled.");
        }

        // ENG-ALG-226 / INV-140: Fused convert+transpose kernel.
        // Compiled as a separate program so a register-spill failure on Adreno
        // (per-thread float4 budget) leaves the legacy `cvt.cl` path intact.
        // `convert_q4_0_to_noshuffle()` falls back to the 4-step path when this
        // program (and therefore `kernel_cvt_q4_0_noshuffle_fused`) is `None`.
        let cvt_noshuffle_fused_src = include_str!("../../kernels/cvt_q4_0_noshuffle_fused.cl");
        let cvt_noshuffle_fused_program = Program::builder()
            .devices(device)
            .src(cvt_noshuffle_fused_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
            .ok();
        if cvt_noshuffle_fused_program.is_some() {
            log::info!(
                "cvt_q4_0_noshuffle_fused.cl compiled (ENG-ALG-226 fused SOA convert+transpose)"
            );
        } else {
            log::warn!(
                "cvt_q4_0_noshuffle_fused.cl failed to compile; falling back to 4-step host transpose path."
            );
        }

        // Adreno Q4_0 GEMM fast path: `mul_mat_Ab_Bi_8x4.cl` consumes SOA
        // weights (from cvt.cl) and F16-transposed activations to compute
        // prefill matmul at ~4-5x the throughput of generic `mul_mm_q4_0`.
        let gemm_ab_bi_src = include_str!("../../kernels/mul_mat_Ab_Bi_8x4.cl");
        let gemm_ab_bi_program = Program::builder()
            .devices(device)
            .src(gemm_ab_bi_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
            .ok();
        if gemm_ab_bi_program.is_some() {
            log::info!("mul_mat_Ab_Bi_8x4.cl compiled (Adreno Q4_0 GEMM fast path for prefill)");
        } else {
            log::warn!(
                "mul_mat_Ab_Bi_8x4.cl failed to compile. Q4_0 prefill falls back to generic GEMM."
            );
        }

        // Activation transpose kernel (F32 → F16, pads N to multiple of 8).
        // Required pre-processing for `mul_mat_Ab_Bi_8x4`.
        let transpose_src = include_str!("../../kernels/transpose.cl");
        let transpose_program = Program::builder()
            .devices(device)
            .src(transpose_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
            .ok();
        if transpose_program.is_some() {
            log::info!("transpose.cl compiled (activation F32→F16 transpose)");
        } else {
            log::warn!("transpose.cl failed to compile. Adreno Q4_0 GEMM disabled.");
        }

        // Create and cache all kernel objects once
        let kernel_cache = KernelCache {
            kernel_mul_mat_f32_f32: ocl::core::create_kernel(&program, "kernel_mul_mat_f32_f32")?,
            kernel_mul_mat_f16_f32: ocl::core::create_kernel(
                &f16_program,
                "kernel_mul_mat_f16_f32",
            )?,
            kernel_mul_mat_q4_0_f32: ocl::core::create_kernel(
                &q4_0_program,
                "kernel_mul_mat_q4_0_f32",
            )?,
            kernel_rms_norm_opt: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_rms_norm_opt",
            )?,
            kernel_softmax_opt: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_softmax_opt",
            )?,
            kernel_rope_simple: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_rope_simple",
            )?,
            kernel_silu_mul_simple: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_silu_mul_simple",
            )?,
            kernel_gelu_tanh_mul: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_gelu_tanh_mul",
            )?,
            kernel_add_assign_simple: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_add_assign_simple",
            )?,
            kernel_add_row_bias: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_add_row_bias",
            )?,
            kernel_scale_simple: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_scale_simple",
            )?,
            kernel_get_rows_q4_0: ocl::core::create_kernel(
                &get_rows_program,
                "kernel_get_rows_q4_0",
            )?,
            kernel_get_rows_f32: ocl::core::create_kernel(
                &get_rows_program,
                "kernel_get_rows_f32",
            )?,
            kernel_get_rows_f16: ocl::core::create_kernel(
                &get_rows_program,
                "kernel_get_rows_f16",
            )?,
            kernel_attn_gen: ocl::core::create_kernel(&simple_ops_program, "kernel_attn_gen")?,
            kernel_cast_f32_to_f16: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_cast_f32_to_f16",
            )?,
            kernel_attn_gen_half: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_attn_gen_half",
            )?,
            kernel_quantize_f32_to_q4_0: ocl::core::create_kernel(
                &quant_q4_0_program,
                "kernel_quantize_f32_to_q4_0",
            )?,
            kernel_kv_scatter_f32_to_f16: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_kv_scatter_f32_to_f16",
            )?,
            kernel_kv_scatter_f32_to_f16_batch: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_kv_scatter_f32_to_f16_batch",
            )?,
            kernel_rms_norm_oop: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_rms_norm_oop",
            )?,
            kernel_add_rms_norm_oop: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_add_rms_norm_oop",
            )?,
            kernel_fused_norm_merge: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_fused_norm_merge",
            )?,
            // f4 variant exists only in the subgroup simple_ops.cl path, not in the
            // nosub fallback. Fall back to the scalar kernel on nosub platforms.
            kernel_fused_norm_merge_f4: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_fused_norm_merge_f4",
            )
            .ok(),
            kernel_flash_attn_f32: flash_attn_f32_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32").ok()),
            kernel_flash_attn_f32_dk256: flash_attn_f32_program_dk256
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32").ok()),
            kernel_flash_attn_f32_f16_dk64: flash_attn_f32_f16_program_dk64
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16").ok()),
            kernel_flash_attn_f32_f16_dk128: flash_attn_f32_f16_program_dk128
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16").ok()),
            kernel_flash_attn_f32_f16_q1_dk64: flash_attn_f32_f16_program_dk64
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16_q1").ok()),
            kernel_flash_attn_f32_f16_q1_dk128: flash_attn_f32_f16_program_dk128
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16_q1").ok()),
            kernel_mul_mm_f16_f32: gemm_f16_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mm_f16_f32_l4_lm").ok()),
            kernel_mul_mm_f32_f32: gemm_f32_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mm_f32_f32_l4_lm").ok()),
            kernel_mul_mm_q4_0_f32: gemm_q4_0_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mm_q4_0_f32_l4_lm").ok()),
            kernel_mul_mat_f16_f32_l4: f16_l4_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mat_f16_f32_l4").ok()),
            kernel_mul_mat_f16_f32_1row: f16_1row_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mat_f16_f32_1row").ok()),
            f16_is_nosub,
            kernel_cvt_q4_0_noshuffle: cvt_noshuffle_program.as_ref().and_then(|p| {
                ocl::core::create_kernel(p, "kernel_convert_block_q4_0_noshuffle").ok()
            }),
            kernel_cvt_q4_0_noshuffle_fused: cvt_noshuffle_fused_program.as_ref().and_then(|p| {
                ocl::core::create_kernel(p, "kernel_convert_block_q4_0_noshuffle_fused").ok()
            }),
            kernel_mul_mat_ab_bi_8x4: gemm_ab_bi_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mat_Ab_Bi_8x4").ok()),
            kernel_transpose_32_16: transpose_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_transpose_32_16").ok()),
            // GEMV noshuffle kernel is built per-dimension (lazy), so None initially
        };

        log::info!("OpenCL kernels cached successfully");

        // Auto-detect UMA (integrated GPU) vs discrete GPU for zero-copy decision.
        // CL_DEVICE_HOST_UNIFIED_MEMORY is the standard way; fall back to name heuristic.
        let use_zero_copy = if std::env::var("FORCE_DEVICE_ONLY").is_ok() {
            log::info!("FORCE_DEVICE_ONLY: disabling zero-copy, using device-only buffers");
            false
        } else {
            let unified_mem = device
                .info(ocl::core::DeviceInfo::HostUnifiedMemory)
                .map(|v| v.to_string().trim().to_lowercase())
                .unwrap_or_default();
            if unified_mem == "true" || unified_mem == "1" {
                true
            } else {
                let name_lower = device_name.to_lowercase();
                name_lower.contains("adreno")
                    || name_lower.contains("mali")
                    || name_lower.contains("powervr")
            }
        };
        log::info!(
            "Memory mode: {} (zero_copy={})",
            if use_zero_copy {
                "UMA/integrated"
            } else {
                "discrete"
            },
            use_zero_copy
        );

        // Pre-allocate 1-element dummy buffer for attention_gen (scores_out=None path)
        let dummy_score_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE,
                1,
                None,
            )?
        };

        Ok(Self {
            context,
            queue,
            device,
            program,
            simple_ops_program,
            q4_0_program,
            f16_program,
            quant_q4_0_program,
            get_rows_program,
            flash_attn_f32_program,
            flash_attn_f32_program_dk256,
            flash_attn_f32_f16_program_dk64,
            flash_attn_f32_f16_program_dk128,
            f16_l4_program,
            f16_1row_program,
            gemm_f16_program,
            gemm_f32_program,
            gemm_q4_0_program,
            attention_scores_program,
            cvt_noshuffle_program,
            cvt_noshuffle_fused_program,
            gemm_ab_bi_program,
            transpose_program,
            kernels: UnsafeCell::new(kernel_cache),
            use_zero_copy,
            dummy_score_buf,
            gpu_score_acc: UnsafeCell::new(None),
            cl_opts: cl_opts.clone(),
            max_mem_alloc_size,
            gemv_noshuffle_cache: UnsafeCell::new(HashMap::new()),
            noshuffle_soa_registry: UnsafeCell::new(HashMap::new()),
            gemm_act_trans_buf: UnsafeCell::new(None),
            gemm_act_trans_key: UnsafeCell::new(None),
            profile_events_enabled,
            profile_events: UnsafeCell::new(Vec::new()),
            profile_accum: UnsafeCell::new(HashMap::new()),
            op_label_hint: UnsafeCell::new(None),
            // MSG-068 Phase 2: meter는 profile-events 모드에서만 생성된다.
            // 호스트가 --profile-events 또는 --heartbeat-gpu-profile로 queue
            // profiling을 켰을 때만 의미가 있다.
            gpu_self_meter: if profile_events_enabled {
                Some(Arc::new(OpenClEventGpuMeter::new()))
            } else {
                None
            },
            // WSWAP-5-TBT-DIAG: cl_mem allocation diagnostics.
            cl_mem_diag_enabled: std::env::var("LLMRS_CL_MEM_DIAG")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            cl_mem_diag: UnsafeCell::new(HashMap::new()),
            // LISWAP-2 prototype (plan: chasing-hopper). Lazy-init in
            // `transfer_queue_or_init()`.
            transfer_queue: std::sync::OnceLock::new(),
            // LISWAP-3 prototype (Direction A, Stage 3). Lazy-init in
            // `host_ptr_pool_or_init()`.
            host_ptr_pool: std::sync::OnceLock::new(),
            // LISWAP multi-context experiment. Lazy-init in
            // `swap_context_or_init()` / `swap_queue_or_init()`.
            swap_context: std::sync::OnceLock::new(),
            swap_queue: std::sync::OnceLock::new(),
            // CPU companion for host fallback routing (S-2 sprint
            // 2026-05-24): shared singleton — feature detection runs once.
            // LAYER-EXEMPT: cross_backend_bootstrap — §13.8-P cpu_companion init.
            cpu_companion: crate::backend::cpu::cpu_singleton(),
            // Sprint 2a Phase 2: default off. `new_with_options(_, true)` 가
            // dlopen 후 본 field 를 `Some` 으로 set 한다. 기존 `new` /
            // `new_with_profile_events` 진입점은 None 유지 (legacy path).
            rpcmem_allocator: None,
        })
    }

    // ── WSWAP-5-TBT-DIAG: cl_mem allocation diagnostics helpers ─────────
    //
    // Hooks called from buffer/image allocation sites and registry
    // invalidations. All entry points short-circuit on
    // `!cl_mem_diag_enabled` so production is zero-cost when the env var
    // is unset. UnsafeCell is single-threaded inference access (same
    // pattern as the other diagnostic UnsafeCells in this struct).

    /// Returns true when `LLMRS_CL_MEM_DIAG=1` is set on backend creation.
    /// Public so generate.rs can gate its dump invocations cheaply.
    #[inline]
    pub fn cl_mem_diag_enabled(&self) -> bool {
        self.cl_mem_diag_enabled
    }

    /// Record a cl_mem (buffer or image) allocation under `category`.
    /// `size` is the user-requested byte count.
    #[inline]
    pub fn record_cl_mem_alloc(&self, category: &'static str, size: usize) {
        if !self.cl_mem_diag_enabled {
            return;
        }
        // SAFETY: single-threaded inference access. Mirrors the contract
        // documented on `noshuffle_soa_registry` etc.
        let map = unsafe { &mut *self.cl_mem_diag.get() };
        let entry = map.entry(category).or_default();
        entry.count += 1;
        entry.total_bytes += size;
    }

    /// Record an explicit cl_mem release under `category`. Currently only
    /// the noshuffle SOA registry invalidation path emits these — generic
    /// destructor drops are not instrumented.
    #[inline]
    pub fn record_cl_mem_release(&self, category: &'static str, size: usize) {
        if !self.cl_mem_diag_enabled {
            return;
        }
        let map = unsafe { &mut *self.cl_mem_diag.get() };
        let entry = map.entry(category).or_default();
        entry.releases += 1;
        entry.released_bytes += size;
    }

    /// Snapshot the current cl_mem diagnostic buckets. Returns a sorted
    /// `(category, bucket)` vector so consumer formatting is deterministic.
    pub fn cl_mem_diag_snapshot(&self) -> Vec<(&'static str, ClMemDiagBucket)> {
        if !self.cl_mem_diag_enabled {
            return Vec::new();
        }
        let map = unsafe { &*self.cl_mem_diag.get() };
        let mut out: Vec<(&'static str, ClMemDiagBucket)> =
            map.iter().map(|(k, v)| (*k, v.clone())).collect();
        out.sort_by_key(|(k, _)| *k);
        out
    }

    /// Dump the current cl_mem diagnostic buckets to stderr with `prefix`
    /// prepended to each line. Aggregate totals (count + bytes) are
    /// emitted on a final summary line. No-op when the diagnostic is off.
    pub fn dump_cl_mem_diagnostics(&self, prefix: &str) {
        if !self.cl_mem_diag_enabled {
            return;
        }
        let snap = self.cl_mem_diag_snapshot();
        let mut total_count: usize = 0;
        let mut total_bytes: usize = 0;
        let mut total_releases: usize = 0;
        let mut total_released_bytes: usize = 0;
        for (cat, b) in &snap {
            eprintln!(
                "[CL-MEM-DIAG]{prefix} {cat}: count={} alive_bytes={} (releases={} released_bytes={})",
                b.count,
                b.total_bytes.saturating_sub(b.released_bytes),
                b.releases,
                b.released_bytes
            );
            total_count += b.count;
            total_bytes += b.total_bytes;
            total_releases += b.releases;
            total_released_bytes += b.released_bytes;
        }
        eprintln!(
            "[CL-MEM-DIAG]{prefix} TOTAL: count={total_count} alive={} \
             bytes={total_bytes} released={total_releases} released_bytes={total_released_bytes}",
            total_count.saturating_sub(total_releases),
        );
    }

    // ── Event-based per-op profiling helpers (--profile-events) ─────────
    //
    // See `OpenCLBackend::new_with_profile_events()` for the design rationale.
    // All three helpers are no-ops when `profile_events_enabled` is false.

    /// Set a caller-provided label hint. Overrides the static label passed to
    /// `enqueue_kernel_labeled()` for the next dispatches, until cleared.
    /// Used by the forward pass to distinguish matmul_qkv / matmul_wo /
    /// matmul_ffn / lm_head which all dispatch the same GEMV/GEMM kernels.
    #[inline]
    pub fn set_op_label(&self, label: &'static str) {
        if !self.profile_events_enabled {
            return;
        }
        // SAFETY: single-threaded inference access (same pattern as `kernels`).
        unsafe {
            *self.op_label_hint.get() = Some(label);
        }
    }

    /// Clear a previously-set label hint.
    #[inline]
    pub fn clear_op_label(&self) {
        if !self.profile_events_enabled {
            return;
        }
        unsafe {
            *self.op_label_hint.get() = None;
        }
    }

    /// Enqueue an OpenCL kernel, capturing an event tagged with `default_label`
    /// (or the caller-provided hint, when set) when profile-events mode is on.
    ///
    /// Fallback path (profile off) = plain `ocl::core::enqueue_kernel` with no
    /// event — identical to the old sites being replaced.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_kernel_labeled(
        &self,
        kernel: &CoreKernel,
        default_label: &'static str,
        work_dim: u32,
        gws: &[usize; 3],
        lws: Option<[usize; 3]>,
    ) -> Result<()> {
        if self.profile_events_enabled {
            let label = unsafe { (*self.op_label_hint.get()).unwrap_or(default_label) };
            let mut ev: ocl::core::Event = ocl::core::Event::null();
            unsafe {
                ocl::core::enqueue_kernel(
                    &self.queue,
                    kernel,
                    work_dim,
                    None,
                    gws,
                    lws,
                    None::<&ocl::core::Event>,
                    Some(&mut ev),
                )?;
                let events = &mut *self.profile_events.get();
                events.push((label, ev));
            }
        } else {
            unsafe {
                ocl::core::enqueue_kernel(
                    &self.queue,
                    kernel,
                    work_dim,
                    None,
                    gws,
                    lws,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }
        Ok(())
    }

    /// Flush captured events into the per-label accumulator, reading
    /// `CL_PROFILING_COMMAND_{START,END}` from each event and summing
    /// `(end - start)` ns into `profile_accum[label]`.
    ///
    /// Caller must ensure the queue has been drained (e.g. via
    /// `synchronize()`) before calling — profiling info is only valid once
    /// the command has reached `CL_COMPLETE`. Events are released
    /// automatically when dropped from the Vec.
    ///
    /// No-op when profile-events mode is off.
    pub fn flush_and_aggregate_profile(&self) -> Result<()> {
        if !self.profile_events_enabled {
            return Ok(());
        }
        // SAFETY: single-threaded inference access.
        let (events_vec, accum) = unsafe {
            (
                &mut *self.profile_events.get(),
                &mut *self.profile_accum.get(),
            )
        };
        for (label, ev) in events_vec.drain(..) {
            let start_ns =
                match ocl::core::get_event_profiling_info(&ev, ocl::core::ProfilingInfo::Start) {
                    Ok(info) => info.time()?,
                    Err(e) => {
                        log::warn!("profile: get_event_profiling_info(start) failed: {}", e);
                        continue;
                    }
                };
            let end_ns =
                match ocl::core::get_event_profiling_info(&ev, ocl::core::ProfilingInfo::End) {
                    Ok(info) => info.time()?,
                    Err(e) => {
                        log::warn!("profile: get_event_profiling_info(end) failed: {}", e);
                        continue;
                    }
                };
            let delta = end_ns.saturating_sub(start_ns);
            *accum.entry(label).or_insert(0) += delta;
            // MSG-068 Phase 2: GPU self-util accumulator (opt-in). Sums raw
            // (end - start) ns regardless of label. CommandExecutor drains
            // this via GpuSelfMeter::sample() on each heartbeat.
            if let Some(meter) = self.gpu_self_meter.as_ref() {
                meter.record_busy_ns(delta);
            }
            // ev is dropped here -> clReleaseEvent
        }
        Ok(())
    }

    /// Clone the shared GPU self-utilization meter, if profile-events is enabled.
    ///
    /// Used by `CommandExecutor` to periodically drain accumulated GPU busy
    /// ns and compute `EngineStatus.self_gpu_pct` (MSG-068, MGR-DAT-076
    /// Phase 2). Returns `None` when the backend was constructed without
    /// profiling (`new_with_profile_events(false)`), in which case the
    /// executor falls back to `self_gpu_pct = 0.0` (INV-092).
    pub fn gpu_self_meter(&self) -> Option<Arc<OpenClEventGpuMeter>> {
        self.gpu_self_meter.clone()
    }

    /// Consume the accumulated per-label ns map, returning it as `HashMap<String, u64>`.
    /// Clears the internal accumulator (so callers can snapshot mid-run).
    ///
    /// Returns an empty map when profile-events is off.
    pub fn take_profile_accum(&self) -> HashMap<String, u64> {
        if !self.profile_events_enabled {
            return HashMap::new();
        }
        let accum = unsafe { &mut *self.profile_accum.get() };
        accum
            .drain()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<HashMap<_, _>>()
    }

    /// Initialize the GPU-side attention score accumulator.
    ///
    /// Resolves the `attn_score` GPU score-reduce policy from the technique registry (which compiles
    /// its own `score_reduce.cl` from this backend's borrowed context) and allocates the persistent
    /// score buffers. If the reducer is unregistered or its kernel fails to compile, this returns an
    /// error and the accumulator is not created (graceful degradation — the caller falls back to the
    /// per-token CPU score-readback path).
    ///
    /// Must be called before the decode loop starts.
    pub fn init_gpu_score_acc(
        &self,
        n_layers: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        decay: f32,
    ) -> Result<()> {
        if n_kv_heads > 16 {
            anyhow::bail!(
                "GpuScoreAccumulator: n_kv_heads={n_kv_heads} exceeds the fused reduce kernel limit \
                 of 16 (step_head_local[16] in attn-score's score_reduce.cl)"
            );
        }
        // Resolve the GPU score-reduce POLICY from the technique registry (the force-linked
        // `attn-score` plugin); it compiles its own `score_reduce.cl` from this backend's borrowed
        // context. The engine core no longer holds the reduce kernel — only the score buffers.
        let reducer = self.with_score_reduce_make_args(|make_args| {
            argus_extension_api::find_score_reducer("attn_score")
                .ok_or_else(|| {
                    anyhow!(
                        "ScoreReduceBackend 'attn_score' not registered — suspect linkme fat-LTO \
                         --gc-sections silent drop; ensure the `use attn_score as _;` force-link"
                    )
                })
                .and_then(|reg| {
                    (reg.make)(make_args).map_err(|e| anyhow!("attn_score reduce make failed: {e}"))
                })
        })?;
        let acc = gpu_score::GpuScoreAccumulator::new(
            self.queue.as_core(),
            self.context.as_core(),
            reducer,
            n_layers,
            n_heads_q,
            n_kv_heads,
            max_seq_len,
            decay,
        )?;
        // SAFETY: single-threaded access (same as kernels UnsafeCell pattern)
        unsafe {
            *self.gpu_score_acc.get() = Some(acc);
        }
        log::info!(
            "GPU score accumulator initialized (n_layers={}, n_heads_q={}, n_kv_heads={}, max_seq={})",
            n_layers,
            n_heads_q,
            n_kv_heads,
            max_seq_len
        );
        Ok(())
    }

    /// Get a reference to the GPU score accumulator (if initialized).
    pub fn gpu_score_acc(&self) -> Option<&gpu_score::GpuScoreAccumulator> {
        // SAFETY: single-threaded access
        unsafe { (*self.gpu_score_acc.get()).as_ref() }
    }

    /// Get a mutable reference to the GPU score accumulator (if initialized).
    ///
    /// # Safety
    /// Uses UnsafeCell — caller must ensure single-threaded access.
    /// The `&self -> &mut T` pattern is intentional (same as kernels UnsafeCell).
    #[allow(clippy::mut_from_ref)]
    pub fn gpu_score_acc_mut(&self) -> Option<&mut gpu_score::GpuScoreAccumulator> {
        // SAFETY: single-threaded inference, same as kernels UnsafeCell pattern
        unsafe { (*self.gpu_score_acc.get()).as_mut() }
    }

    /// Tiled GEMM for F16 weights: A(F32,[M,K]) x B^T(F16,[N,K]) -> Out(F32,[M,N])
    /// Uses mul_mm_f16_f32_l4_lm kernel with BM=64, BN=64, BK=16 tiling.
    fn matmul_gemm_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        let a_buf = get_cl_mem(a.buffer().as_ref())?;
        let b_buf = get_cl_mem(b.buffer().as_ref())?;
        let out_buf = get_cl_mem(out.buffer().as_ref())?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = kernels
            .kernel_mul_mm_f16_f32
            .as_ref()
            .ok_or_else(|| anyhow!("GEMM F16 kernel not available"))?;

        // All strides in ELEMENT counts (kernel divides by LOAD_VEC internally)
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne11 = m as i32;
        let ne12 = 1i32;
        let stride_a = k as i32;
        let stride_b = k as i32;
        let stride_d = n as i32;
        let batch_stride_a = (n * k) as i32;
        let batch_stride_b = (m * k) as i32;
        let batch_stride_d = (m * n) as i32;
        let r2 = 1i32;
        let r3 = 1i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?; // src0=weight(F16)
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(a_buf))?; // src1=activation(F32)
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&ne11))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&stride_a))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&stride_b))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&stride_d))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&batch_stride_a))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&batch_stride_b))?;
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&batch_stride_d))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&r3))?;

            // BM=64, BN=64, TM=4, TN=8 → 128 threads/WG
            // group_id(0) tiles N (weight rows), group_id(1) tiles M (activation rows)
            let local_work_size: [usize; 3] = [128, 1, 1];
            let global_work_size: [usize; 3] = [n.div_ceil(64) * 128, m.div_ceil(64), 1];

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    /// Tiled GEMM for F32 weights: A(F32,[M,K]) x B^T(F32,[N,K]) -> Out(F32,[M,N])
    fn matmul_gemm_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        let a_buf = get_cl_mem(a.buffer().as_ref())?;
        let b_buf = get_cl_mem(b.buffer().as_ref())?;
        let out_buf = get_cl_mem(out.buffer().as_ref())?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = kernels
            .kernel_mul_mm_f32_f32
            .as_ref()
            .ok_or_else(|| anyhow!("GEMM F32 kernel not available"))?;

        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne11 = m as i32;
        let ne12 = 1i32;
        let stride_a = k as i32;
        let stride_b = k as i32;
        let stride_d = n as i32;
        let batch_stride_a = (n * k) as i32;
        let batch_stride_b = (m * k) as i32;
        let batch_stride_d = (m * n) as i32;
        let r2 = 1i32;
        let r3 = 1i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(a_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&ne11))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&stride_a))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&stride_b))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&stride_d))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&batch_stride_a))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&batch_stride_b))?;
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&batch_stride_d))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&r3))?;

            let local_work_size: [usize; 3] = [128, 1, 1];
            let global_work_size: [usize; 3] = [n.div_ceil(64) * 128, m.div_ceil(64), 1];

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    /// Tiled GEMM for Q4_0 weights: A(F32,[M,K]) x B^T(Q4_0,[N,K]) -> Out(F32,[M,N])
    ///
    /// Uses interleaved BlockQ4_0 layout (single buffer; no separate q/d split).
    /// Matches `kernel_mul_mm_q4_0_f32_l4_lm` signature ported from llama.cpp.
    fn matmul_gemm_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        let a_buf = get_cl_mem(a.buffer().as_ref())?;
        let b_buf = get_cl_mem(b.buffer().as_ref())?;
        let out_buf = get_cl_mem(out.buffer().as_ref())?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = kernels
            .kernel_mul_mm_q4_0_f32
            .as_ref()
            .ok_or_else(|| anyhow!("GEMM Q4_0 kernel not available"))?;

        // All strides in ELEMENT counts (kernel divides by LOAD_VEC internally).
        // src0 = weight (Q4_0, [N,K]); src1 = activation (F32, [M,K]); dst = out (F32, [M,N]).
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne11 = m as i32;
        let ne12 = 1i32;
        let stride_a = k as i32;
        let stride_b = k as i32;
        let stride_d = n as i32;
        let batch_stride_a = (n * k) as i32;
        let batch_stride_b = (m * k) as i32;
        let batch_stride_d = (m * n) as i32;
        let r2 = 1i32;
        let r3 = 1i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?; // src0=weight(Q4_0)
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(a_buf))?; // src1=activation(F32)
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&ne11))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&stride_a))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&stride_b))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&stride_d))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&batch_stride_a))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&batch_stride_b))?;
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&batch_stride_d))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&r3))?;

            // BM=64, BN=64, TM=4, TN=8 → 128 threads/WG
            // group_id(0) tiles N (weight rows), group_id(1) tiles M (activation rows)
            let local_work_size: [usize; 3] = [128, 1, 1];
            let global_work_size: [usize; 3] = [n.div_ceil(64) * 128, m.div_ceil(64), 1];

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    /// F16 weight matmul: A(F32) x B^T(F16) -> Out(F32)
    /// Routes to GEMM kernel for prefill (M > 4) or GEMV kernel for decode.
    pub fn matmul_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        // Use tiled GEMM for prefill (M > 8), GEMV for decode/short sequences.
        // GEMM BM=64 tiling wastes work for small M; GEMV N_DST=4 is optimal there.
        const GEMM_THRESHOLD: usize = 8;
        if m > GEMM_THRESHOLD {
            let kernels = unsafe { &*self.kernels.get() };
            if kernels.kernel_mul_mm_f16_f32.is_some() {
                return self.matmul_gemm_f16(a, b, out);
            }
        }

        // GEMV path (decode or GEMM unavailable)
        let a_buf = get_cl_mem(a.buffer().as_ref()).map_err(|_| {
            anyhow!(
                "matmul_f16: A is not OpenCL buffer (kind={}, dtype={:?}, shape={:?})",
                buffer_kind_label(a.buffer().as_ref()),
                a.dtype(),
                a.shape().dims()
            )
        })?;
        let b_buf = get_cl_mem(b.buffer().as_ref()).map_err(|_| {
            anyhow!(
                "matmul_f16: B (weight) is not OpenCL buffer (kind={}, dtype={:?}, shape={:?})",
                buffer_kind_label(b.buffer().as_ref()),
                b.dtype(),
                b.shape().dims()
            )
        })?;
        let out_buf = get_cl_mem(out.buffer().as_ref()).map_err(|_| {
            anyhow!(
                "matmul_f16: Out is not OpenCL buffer (kind={}, dtype={:?}, shape={:?})",
                buffer_kind_label(out.buffer().as_ref()),
                out.dtype(),
                out.shape().dims()
            )
        })?;

        let kernels = unsafe { &*self.kernels.get() };
        let f16_nosub = kernels.f16_is_nosub;

        // 1row variant (mul_mv_f16_f32_1row.cl, N_DST=1 → 1 row/WG, single
        // 64-lane subgroup). Matches llama.cpp's decode path; dispatched when
        // activation batch m=1 (single-token decode). Disable with
        // LLMRS_F16_GEMV_DISABLE_1ROW=1 for A/B testing.
        let disable_1row = std::env::var("LLMRS_F16_GEMV_DISABLE_1ROW")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let use_1row = !disable_1row && m == 1 && kernels.kernel_mul_mat_f16_f32_1row.is_some();

        // L4 variant (mul_mv_f16_f32_l4.cl, N_DST=4 → 4 rows/WG) is also 4-wave
        // K-split, so it only works when the main 4-wave kernel is loaded.
        // Use it for large-N matmuls (lm_head, FFN) where halving WG count pays off.
        const LARGE_N_THRESHOLD: usize = 4096;
        let use_l4 = !use_1row
            && !f16_nosub
            && n > LARGE_N_THRESHOLD
            && kernels.kernel_mul_mat_f16_f32_l4.is_some();
        let kernel = if use_1row {
            kernels.kernel_mul_mat_f16_f32_1row.as_ref().unwrap()
        } else if use_l4 {
            kernels.kernel_mul_mat_f16_f32_l4.as_ref().unwrap()
        } else {
            &kernels.kernel_mul_mat_f16_f32
        };

        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne10 = k as i32;
        let ne12 = 1i32;
        let ne0 = n as i32;
        let ne1 = m as i32;
        let r2 = 1i32;
        let r3 = 1i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?; // src0=weight(F16)
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(a_buf))?; // src1=activation(F32)
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&r3))?;

            let (global_work_size, local_work_size) = if use_1row {
                // 1row kernel: local=[64,1,1], N_DST=1 → 1 row/WG (single SG).
                // Global: dim0 = n*64, dim1 = m=1, dim2 = 1.
                ([n * 64, m, 1], [64usize, 1, 1])
            } else if f16_nosub {
                // Nosub kernel: local=[64,1,1], N_DST=4 → 4 rows/WG.
                // Global: dim0 = ceil(n/4)*64, dim1 = m (batch), dim2 = 1.
                const NOSUB_N_DST: usize = 4;
                let n_groups = n.div_ceil(NOSUB_N_DST);
                ([n_groups * 64, m, 1], [64usize, 1, 1])
            } else if use_l4 {
                // 4-wave L4 kernel: local=[64,4,1], N_DST=4 → 4 rows/WG, with
                // 64 lanes × 4 waves cooperating on each row-group (256-way K parallelism).
                // Global: dim0 = ceil(n/4)*64, dim1 = m*4 (4 waves per batch), dim2 = 1.
                const L4_N_DST: usize = 4;
                let n_groups = n.div_ceil(L4_N_DST);
                ([n_groups * 64, m * 4, 1], [64usize, 4, 1])
            } else {
                // 4-wave kernel: local=[64,4,1], N_DST=2 → 2 rows/WG, with
                // 64 lanes × 4 waves cooperating on each row-pair (256-way K parallelism).
                // Global: dim0 = ceil(n/2)*64, dim1 = m*4 (4 waves per batch), dim2 = 1.
                const WAVE4_N_DST: usize = 2;
                let n_groups = n.div_ceil(WAVE4_N_DST);
                ([n_groups * 64, m * 4, 1], [64usize, 4, 1])
            };
            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    pub fn matmul_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        let a_buf = get_cl_mem(a.buffer().as_ref()).map_err(|_| {
            anyhow!(
                "matmul_q4_0: A is not OpenCL buffer (kind={}, dtype={:?}, shape={:?})",
                buffer_kind_label(a.buffer().as_ref()),
                a.dtype(),
                a.shape().dims()
            )
        })?;
        let b_buf = get_cl_mem(b.buffer().as_ref()).map_err(|_| {
            anyhow!(
                "matmul_q4_0: B (weight) is not OpenCL buffer (kind={}, dtype={:?}, shape={:?})",
                buffer_kind_label(b.buffer().as_ref()),
                b.dtype(),
                b.shape().dims()
            )
        })?;
        let out_buf = get_cl_mem(out.buffer().as_ref()).map_err(|_| {
            anyhow!(
                "matmul_q4_0: Out is not OpenCL buffer (kind={}, dtype={:?}, shape={:?})",
                buffer_kind_label(out.buffer().as_ref()),
                out.dtype(),
                out.shape().dims()
            )
        })?;

        // Auto-dispatch to noshuffle GEMV for decode (m==1) when image1d_buffer_t is available.
        // The noshuffle kernel uses Adreno TP cache via image reads (R32UI weight, RGBA32F act).
        // Only dispatches when q_img is Some (image creation succeeded at load time).
        // When q_img is None (image unsupported) or the registry is empty
        // (e.g. Phase 4-4.10 default AOS path), falls through to standard Q4_0 GEMV.
        if m == 1 {
            let b_key = b_buf.as_ptr() as usize;
            if let Some(entry) = self.lookup_noshuffle_soa(b_key)
                && let Some(ref q_img) = entry.q_img
            {
                return self.matmul_q4_0_noshuffle(
                    q_img,
                    &entry.d_buf,
                    a_buf,
                    out_buf,
                    entry.ne00,
                    entry.ne01,
                );
            }
            // q_img == None or no SOA entry → fall through to standard Q4_0 GEMV
        }

        // Adreno Q4_0 GEMM fast path (llama.cpp mul_mat_Ab_Bi_8x4).
        // Requires a noshuffle SOA entry (q_buf/d_buf transposed) and both
        // gemm_ab_bi + transpose programs. Triggers for every prefill-style
        // token count (m > 1) when available. SOA lookup miss, shape misfit,
        // or kernel unavailability all fall through to the generic tiled
        // GEMM / GEMV paths below.
        //
        // Shape requirements:
        //   * n % 4 == 0 — GEMM iterates 4 output rows per wg lane
        //   * k % 4 == 0 — weight nibble unpacking uses `vload4`
        //
        // `m` (token count) need not be a multiple of 4. The transpose kernel
        // tiles 4 rows per thread; when the last tile's tail row is OOB the
        // corresponding half lands in an `n >= n_no_padding` column of the
        // transposed activation. Downstream `mul_mat_Ab_Bi_8x4` guards writes
        // with `idx+3 < m*n_no_padding`, so garbage from OOB reads never
        // reaches the final output buffer.
        //
        // `n/4` does not have to divide the LWS(=128). Adreno OpenCL 2.0+
        // accepts non-uniform workgroup sizes for the trailing shape (tested
        // on Qwen 2.5-1.5B, n=8960 → n/4=2240 runs cleanly with [1,128,1]).
        // We keep a generic-GEMM fallback in this file so a stricter driver
        // can opt out via env if regressions appear.
        if m > 1 && n.is_multiple_of(4) && k.is_multiple_of(4) {
            let b_key = b_buf.as_ptr() as usize;
            if let Some(entry) = self.lookup_noshuffle_soa(b_key)
                && entry.ne00 == k
                && entry.ne01 == n
            {
                let kernels = unsafe { &*self.kernels.get() };
                if kernels.kernel_mul_mat_ab_bi_8x4.is_some()
                    && kernels.kernel_transpose_32_16.is_some()
                {
                    return self.matmul_q4_0_gemm_adreno(a_buf, out_buf, entry, m, n, k);
                }
            }
        }

        // Tiled GEMM for prefill (M >= 32). Reuses weights across BM=64 rows of output,
        // eliminating the 64x weight-reload overhead of GEMV for large batches.
        if m >= 32 {
            let kernels = unsafe { &*self.kernels.get() };
            if kernels.kernel_mul_mm_q4_0_f32.is_some() {
                return self.matmul_gemm_q4_0(a, b, out);
            }
        }

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_mul_mat_q4_0_f32;

        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1;
        let ne10 = k as i32;
        let ne12 = (k * m) as i32;
        let ne0 = n as i32;
        let ne1 = (n * m) as i32;
        let r2 = 1;
        let r3 = 1;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(a_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;

            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&r3))?;

            let local_work_size: [usize; 3] = [64, 1, 1];
            let group_size_0 = n.div_ceil(4);
            let global_work_size: [usize; 3] = [group_size_0 * local_work_size[0], m, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    /// Register a pre-converted noshuffle SOA entry for a weight tensor.
    ///
    /// `key` should be the original weight buffer's `cl_mem` pointer cast to `usize`.
    /// Called by `prepare_noshuffle_buffers()` after `convert_q4_0_to_noshuffle()`.
    pub fn register_noshuffle_soa(&self, key: usize, entry: NoshuffleSoaEntry) {
        // SAFETY: single-threaded inference access (same pattern as gemv_noshuffle_cache)
        let registry = unsafe { &mut *self.noshuffle_soa_registry.get() };
        registry.insert(key, entry);
    }

    /// Lookup a pre-converted noshuffle SOA entry by the original weight cl_mem pointer.
    ///
    /// Returns None if the weight was not pre-converted (fallback to standard Q4_0 GEMV).
    /// In the Phase 4-4.10 default AOS path the registry is empty so all lookups
    /// return None and callers transparently fall through.
    pub fn lookup_noshuffle_soa(&self, key: usize) -> Option<&NoshuffleSoaEntry> {
        // SAFETY: single-threaded inference access
        let registry = unsafe { &*self.noshuffle_soa_registry.get() };
        let r = registry.get(&key);
        if r.is_none() && std::env::var_os("LLMRS_NOSHUFFLE_SOA_TRACE").is_some() {
            // Phase 4-4.8 diagnostic: dump miss key + registry size + sample keys
            let sample: Vec<usize> = registry.keys().take(4).copied().collect();
            eprintln!(
                "[noshuffle-soa-trace] MISS key=0x{:x} registry_size={} sample_keys={:x?}",
                key,
                registry.len(),
                sample
            );
        }
        r
    }

    /// Clear the entire noshuffle SOA registry.
    ///
    /// Paired with `register_noshuffle_soa()`. Required on the runtime
    /// `SetPartitionRatio` directive path where `map_weights_for_cpu()` may
    /// replace a weight tensor's backing `UnifiedBuffer` with a new `cl_mem`.
    /// The old `cl_mem` key becomes stale, so `lookup_noshuffle_soa()` fails
    /// and `build_plan()` silently falls back to the AOS Q4_0 GEMV path
    /// (measured +102% TBT on Galaxy S25, verify v2 ISSUE-2).
    ///
    /// Dropping each `NoshuffleSoaEntry` releases its `q_buf` / `d_buf` /
    /// `q_img` (all `ocl::core::Mem`), allowing the driver to reclaim the
    /// SOA-transposed GPU allocations before the caller re-runs
    /// `prepare_noshuffle_buffers()` against the new keys.
    pub fn clear_noshuffle_soa_registry(&self) {
        // SAFETY: single-threaded inference access, matching the other
        // UnsafeCell-backed caches on this struct.
        let registry = unsafe { &mut *self.noshuffle_soa_registry.get() };
        // WSWAP-5-TBT-DIAG: account for the cl_mem references the registry
        // owns. Each `NoshuffleSoaEntry` carries q_buf / d_buf (and an
        // optional q_img view). Counting these as releases lets the diag
        // dump reflect the post-invalidation footprint accurately even
        // when the swap path subsequently re-registers fresh buffers.
        if self.cl_mem_diag_enabled {
            for entry in registry.values() {
                let q_bytes = (entry.ne00 / 32) * entry.ne01 * 16;
                let d_bytes = (entry.ne00 / 32) * entry.ne01 * 2;
                // We cannot tell here whether the entry came from the GGUF
                // SOA path (`weight_q4_soa_*`) or the AUF SOA bypass
                // (`auf_soa_*`); attribute releases to a single bucket so
                // the dump signals "registry drop" without double-counting
                // either source. Forensic detail is in the alloc-side
                // counters.
                self.record_cl_mem_release("noshuffle_registry_q", q_bytes);
                self.record_cl_mem_release("noshuffle_registry_d", d_bytes);
                if entry.q_img.is_some() {
                    self.record_cl_mem_release("noshuffle_registry_img", q_bytes);
                }
            }
        }
        registry.clear();
    }

    /// Convert Q4_0 AOS weight buffer to SOA layout for the noshuffle GEMV kernel.
    ///
    /// Called once per weight tensor at model load time and on every weight
    /// swap (Phase 6.5 stage `soa_reconvert`).
    /// Returns (q_buf, d_buf, q_img) — the SOA nibbles buffer, scales buffer,
    /// and an optional image1d_buffer_t wrapping q_buf (R32UI) for Adreno TP cache.
    /// **Transposed** so that the GEMV kernel can access them with coalesced reads.
    ///
    /// # Two paths (same byte-equal output, INV-140)
    ///
    /// 1. **Fused** (ENG-ALG-226, default when available):
    ///    `kernel_convert_block_q4_0_noshuffle_fused` performs nibble
    ///    rearrange + 2D transpose in a single GPU dispatch with zero host
    ///    round-trip. No `queue.finish()` — the caller must `synchronize()`
    ///    before the result is consumed by another stage (ENG-ALG-230/231,
    ///    INV-142).
    ///
    /// 2. **4-step legacy fallback** (when fused kernel is `None` or
    ///    alignment requirements aren't met):
    ///    GPU `kernel_convert_block_q4_0_noshuffle` (nibble rearrange,
    ///    row-major) → host read → CPU ushort-level transpose of q buffer
    ///    → CPU half-level transpose of d buffer → write_buffer × 2
    ///    → image1d_buffer_t creation.
    ///
    /// Both paths produce identical cl_mem contents (INV-140); the fused
    /// kernel was validated against the legacy path on host (NVIDIA OpenCL)
    /// via `engine/tests/spec/test_inv_140_fused_convert_byte_equal.rs`.
    ///
    /// # Arguments
    /// * `src` - Raw Q4_0 weight buffer (BlockQ4_0 AOS format, 18 B / block)
    /// * `num_blocks` - Total number of Q4_0 blocks (= num_rows * K / QK4_0)
    /// * `ne00` - K dimension (elements per row)
    /// * `ne01` - M dimension (number of rows)
    pub fn convert_q4_0_to_noshuffle(
        &self,
        src: &ocl::core::Mem,
        num_blocks: usize,
        ne00: usize,
        ne01: usize,
    ) -> Result<(ocl::core::Mem, ocl::core::Mem, Option<ocl::core::Mem>)> {
        let kernels = unsafe { &*self.kernels.get() };

        // Allocate SOA buffers (the layout below is the *post-transpose*
        // layout, written directly by the fused kernel, or written first
        // row-major by the legacy kernel and overwritten by the host
        // transpose in the 4-step path).
        //
        // dst_q: num_blocks * 16 bytes (QK4_0/2 = 16 nibble-bytes per block)
        let q_bytes = num_blocks * 16;
        let dst_q = unsafe {
            ocl::core::create_buffer::<_, u8>(
                self.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                q_bytes,
                None,
            )?
        };
        self.record_cl_mem_alloc("weight_q4_soa_q", q_bytes);

        // dst_d: num_blocks * 2 bytes (one f16 per block)
        let dst_d = unsafe {
            ocl::core::create_buffer::<_, u16>(
                self.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                num_blocks,
                None,
            )?
        };
        self.record_cl_mem_alloc("weight_q4_soa_d", num_blocks * 2);

        let cols_ushort = ne00 / 4; // K/4 ushort per row (post-transpose width)
        let blocks_per_row = ne00 / 32; // QK4_0 = 32
        let q_total_ushort = ne01 * cols_ushort;

        // ── Path selection ────────────────────────────────────────────────
        //
        // Fused path requires:
        //   * compiled kernel handle, and
        //   * ne00 % 32 == 0 (always true for Q4_0; one block has 32 elems),
        //   * ne00 % 4 == 0  (cols_ushort well-defined; implied above),
        //   * ne01 unrestricted — every row is independent.
        //
        // We additionally require num_blocks == ne01 * blocks_per_row to
        // catch caller mismatches (the kernel divides gid by blocks_per_row
        // to recover row, which would be wrong if num_blocks were inflated).
        // ENG-ALG-226 fused kernel is force-disabled on every backend until
        // the Adreno-specific corruption is understood and fixed. Both the
        // original per-row variant and a uint-store row-pair rewrite exhibit
        // the same ~43 % q_buf byte divergence vs the host CPU reference
        // (`q4_0_aos_to_adreno_soa`) on SM8750 / Adreno 830 — the legacy
        // 4-step path is byte-equal. The kernel source and the host probe
        // (`bin/test_q4_soa_byte_equal`) are kept so the regression can be
        // re-exercised once a fix lands. `LLMRS_FORCE_LEGACY_CVT=1` opts
        // out for diagnostic comparisons; `LLMRS_ENABLE_FUSED_CVT=1` opts
        // in to the (broken) fused path for the same purpose.
        let force_legacy = std::env::var("LLMRS_FORCE_LEGACY_CVT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let opt_in_fused = std::env::var("LLMRS_ENABLE_FUSED_CVT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let fused_eligible = !force_legacy
            && opt_in_fused
            && kernels.kernel_cvt_q4_0_noshuffle_fused.is_some()
            && ne00.is_multiple_of(32)
            && ne01.is_multiple_of(2)
            && num_blocks == ne01 * blocks_per_row;

        if fused_eligible {
            // ── ENG-ALG-226 fused path (v2: row-pair uint stores) ─────────
            // Each work-item processes 2 adjacent rows × 1 K-block, packing
            // outputs into 4-byte aligned uint stores. Global size therefore
            // halves. The row-pair restructure was needed to dodge an
            // Adreno-specific race where parallel ushort stores to adjacent
            // addresses corrupt each other's high byte (~43% q_buf
            // divergence on SM8750 / Adreno 830 with the prior per-row
            // dispatch).
            let fused_kernel = kernels.kernel_cvt_q4_0_noshuffle_fused.as_ref().unwrap();
            unsafe {
                ocl::core::set_kernel_arg(fused_kernel, 0, ocl::core::ArgVal::mem(src))?;
                ocl::core::set_kernel_arg(fused_kernel, 1, ocl::core::ArgVal::mem(&dst_q))?;
                ocl::core::set_kernel_arg(fused_kernel, 2, ocl::core::ArgVal::mem(&dst_d))?;
                ocl::core::set_kernel_arg(
                    fused_kernel,
                    3,
                    ocl::core::ArgVal::scalar(&(ne01 as u32)),
                )?;
                ocl::core::set_kernel_arg(
                    fused_kernel,
                    4,
                    ocl::core::ArgVal::scalar(&(blocks_per_row as u32)),
                )?;

                let global_work_size: [usize; 3] = [(ne01 / 2) * blocks_per_row, 1, 1];
                self.enqueue_kernel_labeled(fused_kernel, "load_time", 1, &global_work_size, None)?;
            }
            // NOTE: ENG-ALG-230 / INV-142 — no `queue.finish()` here. The
            // caller (`SwapExecutor::execute_on_slots`) issues a single
            // `backend.synchronize()` at the stage gate before consumers
            // (forward path, SOA registry rebuild) read the result.
        } else {
            // ── 4-step legacy fallback (kernel unavailable or unaligned) ──
            let cvt_kernel = kernels.kernel_cvt_q4_0_noshuffle.as_ref().ok_or_else(|| {
                anyhow!("Q4_0 noshuffle conversion kernel not available (neither fused nor legacy)")
            })?;

            // Step 1: GPU SOA conversion (nibble rearrange, keeps row-major block order)
            unsafe {
                ocl::core::set_kernel_arg(cvt_kernel, 0, ocl::core::ArgVal::mem(src))?;
                ocl::core::set_kernel_arg(cvt_kernel, 1, ocl::core::ArgVal::mem(&dst_q))?;
                ocl::core::set_kernel_arg(cvt_kernel, 2, ocl::core::ArgVal::mem(&dst_d))?;

                let global_work_size: [usize; 3] = [num_blocks, 1, 1];
                self.enqueue_kernel_labeled(cvt_kernel, "load_time", 1, &global_work_size, None)?;
            }
            self.queue.finish()?;

            // Step 2: CPU transpose of q buffer (ushort-level 2D transpose)
            //
            // Row-major layout (after SOA conversion):
            //   q[row * cols_ushort + col] where cols_ushort = K/4 (ushort per row)
            // Transposed layout (what GEMV expects):
            //   q_t[col * M + row] (column-major by ushort)
            //
            // When GEMV reads uint* from the transposed buffer, consecutive row-pairs
            // (row 2*gid, 2*gid+1) share a uint, enabling 2-row-per-fiber processing.
            {
                let mut q_host = vec![0u16; q_total_ushort];
                unsafe {
                    ocl::core::enqueue_read_buffer(
                        &self.queue,
                        &dst_q,
                        true,
                        0,
                        std::slice::from_raw_parts_mut(
                            q_host.as_mut_ptr() as *mut u8,
                            q_total_ushort * 2,
                        ),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }

                let mut q_transposed = vec![0u16; q_total_ushort];
                for row in 0..ne01 {
                    for col in 0..cols_ushort {
                        q_transposed[col * ne01 + row] = q_host[row * cols_ushort + col];
                    }
                }

                unsafe {
                    ocl::core::enqueue_write_buffer(
                        &self.queue,
                        &dst_q,
                        true,
                        0,
                        std::slice::from_raw_parts(
                            q_transposed.as_ptr() as *const u8,
                            q_total_ushort * 2,
                        ),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
            }

            // Step 3: CPU transpose of d buffer (half-level 2D transpose)
            //
            // Row-major: d[row * blocks_per_row + k] (half per block)
            // Transposed: d_t[k * M + row] (column-major by half)
            //
            // GEMV reads half2* from transposed d: half2[k*(M/2) + gid]
            //   .x = d_t[k*M + 2*gid]   = scale for even row
            //   .y = d_t[k*M + 2*gid+1] = scale for odd row
            {
                let mut d_host = vec![0u16; num_blocks];
                unsafe {
                    ocl::core::enqueue_read_buffer(
                        &self.queue,
                        &dst_d,
                        true,
                        0,
                        std::slice::from_raw_parts_mut(
                            d_host.as_mut_ptr() as *mut u8,
                            num_blocks * 2,
                        ),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }

                let mut d_transposed = vec![0u16; num_blocks];
                for row in 0..ne01 {
                    for k in 0..blocks_per_row {
                        d_transposed[k * ne01 + row] = d_host[row * blocks_per_row + k];
                    }
                }

                unsafe {
                    ocl::core::enqueue_write_buffer(
                        &self.queue,
                        &dst_d,
                        true,
                        0,
                        std::slice::from_raw_parts(
                            d_transposed.as_ptr() as *const u8,
                            num_blocks * 2,
                        ),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
            }
        }

        // Step 4: Create image1d_buffer_t wrapping transposed q_buf (R32UI).
        // Each texel = 1 uint = 2 ushort (row-pair), total width = q_total_ushort / 2 texels.
        // Gracefully falls back to None if image creation fails (unsupported device, size limit).
        let q_total_uint = q_total_ushort / 2;
        let q_img = {
            use ocl::core::{
                ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat,
                MemObjectType,
            };
            let q_img_fmt =
                ImageFormat::new(ImageChannelOrder::R, ImageChannelDataType::UnsignedInt32);
            let q_img_desc = ImageDescriptor::new(
                MemObjectType::Image1dBuffer,
                q_total_uint, // width in texels (each texel = 1 uint)
                0,
                0,
                0,
                0,
                0,
                // SAFETY: Mem::clone() calls clRetainMemObject, so the underlying buffer
                // stays alive even if this image is dropped first.
                Some(dst_q.clone()),
            );
            // SAFETY: dst_q is a valid CL buffer with q_total_uint * 4 bytes.
            // image1d_buffer_t is a read-only view over the same memory.
            let result = unsafe {
                ocl::core::create_image(
                    self.context.as_core(),
                    ocl::core::MEM_READ_ONLY,
                    &q_img_fmt,
                    &q_img_desc,
                    None::<&[u32]>,
                    None, // device_versions — fallback to context auto-detect (OpenCL 1.2+)
                )
            };
            match result {
                Ok(img) => {
                    log::info!(
                        "image1d_buffer_t created for noshuffle Q4_0: width={} texels, ne01={}",
                        q_total_uint,
                        ne01
                    );
                    self.record_cl_mem_alloc("weight_q4_soa_img", q_total_uint * 4);
                    Some(img)
                }
                Err(e) => {
                    log::warn!(
                        "image1d_buffer_t creation failed (ne01={}), fallback to buffer path: {}",
                        ne01,
                        e
                    );
                    None
                }
            }
        };

        log::info!(
            "Q4_0 noshuffle SOA conversion + transpose done ({}): {} blocks, ne00={}, ne01={}, q={} KB, d={} KB, img={}",
            if fused_eligible {
                "fused single-dispatch (ENG-ALG-226)"
            } else {
                "4-step host-transpose fallback"
            },
            num_blocks,
            ne00,
            ne01,
            q_bytes / 1024,
            num_blocks * 2 / 1024,
            if q_img.is_some() { "yes" } else { "no" },
        );
        Ok((dst_q, dst_d, q_img))
    }

    /// Dispatch the noshuffle GEMV kernel for Q4_0 weights in SOA layout.
    ///
    /// Uses image1d_buffer_t for both weight nibbles (R32UI) and activation (RGBA32F)
    /// to leverage Adreno TP (texture pipe) cache for coalesced reads.
    ///
    /// Compiles the dimension-specific kernel on first call and caches it
    /// (keyed by `ne01`). Subsequent calls with the same dimensions hit the cache.
    #[allow(clippy::map_entry)]
    ///
    /// # Arguments
    /// * `q_img` - image1d_buffer_t wrapping SOA nibbles (R32UI, from convert_q4_0_to_noshuffle)
    /// * `d_buf` - SOA scales buffer (half2*, from convert_q4_0_to_noshuffle)
    /// * `src1_buf` - Activation vector buffer (F32), wrapped into RGBA32F image internally
    /// * `dst_buf` - Output buffer (F32)
    /// * `ne00` - K dimension
    /// * `ne01` - M dimension (number of output rows, must be even)
    pub fn matmul_q4_0_noshuffle(
        &self,
        q_img: &ocl::core::Mem,
        d_buf: &ocl::core::Mem,
        src1_buf: &ocl::core::Mem,
        dst_buf: &ocl::core::Mem,
        ne00: usize,
        ne01: usize,
    ) -> Result<()> {
        // Compute strides (in uint units, matching the kernel's indexing on transposed SOA).
        //
        // After ushort-level 2D transpose of the SOA q buffer:
        //   Original: M rows x (K/4) cols of ushort (row-major)
        //   Transposed: (K/4) rows x M cols of ushort (column-major by ushort)
        //
        // When read as uint* (2 ushort = 1 uint), consecutive row-pairs share a uint.
        //   LINE_STRIDE_A = ne01/2 — stride between consecutive ushort columns in uint units
        //   BLOCK_STRIDE_A = 4 * ne01 — stride between consecutive K-blocks (each block = 8 ushort cols)
        //     = 8 ushort_cols * (ne01/2) uint_per_ushort_col = 4*ne01
        //   SIMDGROUP_WIDTH = 64 (Adreno half-wave)
        //
        // Reference: llama.cpp gemv_noshuffle_general.cl: LINE_STRIDE_A=M/2, BLOCK_STRIDE_A=N_SIMDGROUP*M
        let line_stride_a = ne01 / 2;
        let block_stride_a = 4 * ne01;
        let simdgroup_width: usize = 64;

        // SAFETY: single-threaded inference access (same pattern as kernels UnsafeCell)
        let cache = unsafe { &mut *self.gemv_noshuffle_cache.get() };
        if !cache.contains_key(&ne01) {
            let gemv_src = include_str!("../../kernels/gemv_noshuffle_q4_0.cl");
            let defines = format!(
                "{} -DLINE_STRIDE_A={} -DBLOCK_STRIDE_A={} -DSIMDGROUP_WIDTH={}",
                self.cl_opts, line_stride_a, block_stride_a, simdgroup_width
            );

            // Try with vector sub_group_broadcast first (Adreno 830+ / driver v47+).
            // Falls back to scalar path if the compiler rejects float8 broadcast.
            let defines_vec = format!("{} -DVECTOR_SUB_GROUP_BROADCAT", defines);
            let program_result = Program::builder()
                .devices(self.device)
                .src(gemv_src)
                .cmplr_opt(&defines_vec)
                .build(&self.context);

            let program = match program_result {
                Ok(p) => {
                    log::info!("GEMV noshuffle Q4_0: vector sub_group_broadcast enabled");
                    p
                }
                Err(_) => {
                    log::info!("GEMV noshuffle Q4_0: falling back to scalar sub_group_broadcast");
                    Program::builder()
                        .devices(self.device)
                        .src(gemv_src)
                        .cmplr_opt(&defines)
                        .build(&self.context)?
                }
            };

            let kernel = ocl::core::create_kernel(&program, "kernel_gemv_noshuffle_q4_0")?;
            log::info!(
                "GEMV noshuffle Q4_0 kernel compiled: ne01={}, LINE_STRIDE_A={}, BLOCK_STRIDE_A={}",
                ne01,
                line_stride_a,
                block_stride_a
            );
            cache.insert(ne01, (program, kernel));
        }
        let (_program, kernel) = cache.get(&ne01).unwrap();

        let ne00_i = ne00 as i32;
        let ne01_i = ne01 as i32;

        // Create activation image1d_buffer_t (RGBA32F) wrapping the F32 activation buffer.
        // Each texel = float4 (4 floats), so width = ne00 / 4.
        // This image is ephemeral (created per-dispatch, dropped after enqueue).
        // SAFETY: src1_buf has ne00 * sizeof(f32) bytes, and ne00 is always a multiple of 4
        // for Q4_0 (QK4_0=32, ne00 is a multiple of 32).
        let act_img = {
            use ocl::core::{
                ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat,
                MemObjectType,
            };
            let act_img_fmt =
                ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
            let act_img_desc = ImageDescriptor::new(
                MemObjectType::Image1dBuffer,
                ne00 / 4, // float4 texels -> K/4 width
                0,
                0,
                0,
                0,
                0,
                // SAFETY: Mem::clone() calls clRetainMemObject; the activation buffer
                // outlives this ephemeral image (dropped at end of this function).
                Some(src1_buf.clone()),
            );
            unsafe {
                ocl::core::create_image(
                    self.context.as_core(),
                    ocl::core::MEM_READ_ONLY,
                    &act_img_fmt,
                    &act_img_desc,
                    None::<&[f32]>,
                    None, // device_versions — fallback to context auto-detect
                )?
            }
        };

        unsafe {
            // arg 0: weight nibbles image (image1d_buffer_t, R32UI)
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_img))?;
            // arg 1: weight scales (global half2*)
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(d_buf))?;
            // arg 2: activation image (image1d_buffer_t, RGBA32F)
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(&act_img))?;
            // arg 3: output buffer (global float*)
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(dst_buf))?;
            // arg 4: ne00 (K dimension)
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&ne00_i))?;
            // arg 5: ne01 (M dimension)
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&ne01_i))?;

            // Dispatch: global=[ne01/2, N_SIMDGROUP=4, 1], local=[SIMDGROUP_WIDTH=64, 4, 1]
            let n_simdgroup: usize = 4;
            let global_work_size: [usize; 3] = [ne01 / 2, n_simdgroup, 1];
            let local_work_size: [usize; 3] = [simdgroup_width, n_simdgroup, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                2,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        // act_img is dropped here; clReleaseMemObject is called.
        // The enqueued kernel retains its own reference to the image/buffer.
        Ok(())
    }

    /// Adreno Q4_0 GEMM fast path (llama.cpp `mul_mat_Ab_Bi_8x4`).
    ///
    /// Two-stage dispatch:
    ///   1. `kernel_transpose_32_16`: F32 activation `[M_tok, K]` →
    ///      F16 scratch `[K, N_padded]` where `N_padded = (M_tok + 7) & ~7`.
    ///   2. `kernel_mul_mat_Ab_Bi_8x4`: consumes transposed activation +
    ///      SOA weight (`q_buf`/`d_buf`) and writes `[M_tok, N_out]` F32.
    ///
    /// Kernel shape-lookup notes (llama.cpp `ggml-opencl.cpp:9749`):
    ///   * per-shape tuned `local_work_size` is available but the generic
    ///     `[1, 128, 1]` from the default path is what actually ships in
    ///     llama.cpp's release builds for non-llama2 dims. Qwen2.5-1.5B
    ///     prefill shapes do not match any hard-coded entry, so we use
    ///     the default as well.
    ///
    /// # Arguments
    /// * `a_buf` — activation cl_mem (F32, `[m, k]`)
    /// * `out_buf` — output cl_mem (F32, `[m, n]`)
    /// * `entry` — SOA Q4_0 weight (must have `ne00 == k`, `ne01 == n`)
    /// * `m` — token count (rows of activation / output)
    /// * `n` — output dim (rows of weight)
    /// * `k` — input dim (cols of weight / activation)
    ///
    /// Register footprint (per Adreno reqd_sub_group_size("full") thread):
    ///   `c0..c3 half8` (16 halves = 8 float4) + `B half8` + `dequant half4` ≈
    ///   **~10 float4** — well below the 32 float4 spill threshold.
    #[allow(clippy::too_many_arguments)]
    fn matmul_q4_0_gemm_adreno(
        &self,
        a_buf: &ocl::core::Mem,
        out_buf: &ocl::core::Mem,
        entry: &NoshuffleSoaEntry,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        use ocl::core::{
            ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat, MemObjectType,
        };

        // Kernel preconditions:
        //   * m (token count) becomes padded to multiple of 8 for activation tiling.
        //   * n (out_dim) must be divisible by 4 (kernel processes 4 rows per wg lane
        //     via gws[1] = n/4). All Q4_0 weight rows from real models (dim / ffn)
        //     satisfy this — bail to generic path otherwise.
        //   * k (in_dim) must be divisible by 4 for vload4 / per-4 weight unpacking.
        if !n.is_multiple_of(4) || !k.is_multiple_of(4) {
            return Err(anyhow!(
                "Adreno Q4_0 GEMM requires n%4==0 && k%4==0, got n={}, k={}",
                n,
                k
            ));
        }

        let pad_extra = (8 - (m % 8)) % 8;
        let padded_m = m + pad_extra;

        let kernels = unsafe { &*self.kernels.get() };
        let transpose_kernel = kernels
            .kernel_transpose_32_16
            .as_ref()
            .ok_or_else(|| anyhow!("kernel_transpose_32_16 unavailable"))?;
        let gemm_kernel = kernels
            .kernel_mul_mat_ab_bi_8x4
            .as_ref()
            .ok_or_else(|| anyhow!("kernel_mul_mat_Ab_Bi_8x4 unavailable"))?;

        // ── Scratch buffer for the transposed F16 activation ─────────────
        // Size: k * padded_m * 2 bytes. The slot also holds two persistent
        // image views (write-only for transpose, read-only for GEMM). Both
        // images are only valid while the underlying buffer has at least
        // `texel_count * 8` bytes (RGBA16F = 4 halves = 8 bytes per texel);
        // on buffer realloc we rebuild both images too.
        let trans_bytes = k * padded_m * 2;
        let texel_count = k * padded_m / 4;
        // SAFETY: single-threaded inference access, same pattern as
        // `gemv_noshuffle_cache` / `noshuffle_soa_registry`.
        let trans_slot = unsafe { &mut *self.gemm_act_trans_buf.get() };
        let trans_key_slot = unsafe { &mut *self.gemm_act_trans_key.get() };
        let need_realloc = match trans_slot {
            Some((_, _, _, cap, cached_texels)) => {
                *cap < trans_bytes || *cached_texels < texel_count
            }
            None => true,
        };
        if need_realloc {
            let mem = unsafe {
                ocl::core::create_buffer::<_, u8>(
                    self.context.as_core(),
                    ocl::core::MEM_READ_WRITE,
                    trans_bytes,
                    None,
                )?
            };
            let fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::HalfFloat);
            let desc = ImageDescriptor::new(
                MemObjectType::Image1dBuffer,
                texel_count,
                0,
                0,
                0,
                0,
                0,
                Some(mem.clone()),
            );
            let write_img = unsafe {
                ocl::core::create_image(
                    self.context.as_core(),
                    ocl::core::MEM_WRITE_ONLY,
                    &fmt,
                    &desc,
                    None::<&[u16]>,
                    None,
                )?
            };
            let read_img = unsafe {
                ocl::core::create_image(
                    self.context.as_core(),
                    ocl::core::MEM_READ_ONLY,
                    &fmt,
                    &desc,
                    None::<&[u16]>,
                    None,
                )?
            };
            *trans_slot = Some((mem, write_img, read_img, trans_bytes, texel_count));
            *trans_key_slot = None; // any re-alloc invalidates the cache
        }
        let (act_trans_mem, act_out_img, b_in_img, _cap, _tex) = trans_slot.as_ref().unwrap();

        // Skip the transpose pass when the same activation (same ptr, same
        // shape) was transposed on the previous call. Q/K/V within one layer
        // share the post-RMSNorm residual, so ~3 of every 7 matmuls per
        // layer can reuse the existing F16 tile.
        let activation_key = (a_buf.as_ptr() as usize, m, k);
        let skip_transpose = matches!(*trans_key_slot, Some(prev) if prev == activation_key);

        // ── Stage 1: activation transpose F32 → F16 with N-padding ───────
        //
        // Kernel signature:
        //   kernel_transpose_32_16(
        //     __read_only image1d_buffer_t  input  (RGBA32F, width = m * k / 4),
        //     __write_only image1d_buffer_t output (RGBA16F, width = k * padded_m / 4),
        //     const uint rows       = m / 4,           // input row-tiles
        //     const uint cols       = k / 4,           // input col-tiles (K per texel/4)
        //     const uint padded_rows = padded_m / 4);  // output col-tiles
        //
        // gws = [k / 4, padded_m / 4]
        //
        // kernel_transpose_32_16 tiles 4×4 halves per thread. When `m` is
        // not a multiple of 4 the final tile reads past the valid region;
        // Adreno's `read_imageh` clamps to zero on OOB which is exactly
        // what `mul_mat_Ab_Bi_8x4` wants (N>m rows must contribute 0).
        // We still need the scratch padding zeros for n ∈ [m, padded_m).
        // Zero the scratch up-front (once per call, same ACE queue) so the
        // last F16 tile in the padded-N direction is clean irrespective of
        // image sampling semantics. Q4_0 prefill issues 7 matmul/layer and
        // `kernel_transpose_32_16` dominates less than 2% of the wall clock
        // so the zero-fill overhead is negligible.
        //
        // When the cache hit lets us skip the transpose, also skip the
        // zero-fill (the scratch already holds a valid tile).
        if !skip_transpose {
            let zero: u8 = 0;
            // Event arg types: `En` = new event (&mut Event), `Ewl` = wait-list
            // (&[Event]). Passing `None` with fully-specified turbofish lets
            // ocl-core monomorphize without an ambiguous trait resolution.
            type NullEvent<'a> = &'a mut ocl::core::Event;
            type WaitList<'a> = &'a [ocl::core::Event];
            unsafe {
                ocl::core::enqueue_fill_buffer::<u8, _, NullEvent<'_>, WaitList<'_>>(
                    &self.queue,
                    act_trans_mem,
                    zero,
                    0,
                    trans_bytes,
                    None,
                    None,
                    None,
                )?;
            }
        }

        if !skip_transpose {
            // Activation input image (RGBA32F view of a_buf). Ephemeral — the
            // caller owns the F32 buffer and may rebind `a_buf` across calls,
            // so we can't cache this image.
            let act_in_img = {
                let fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
                let desc = ImageDescriptor::new(
                    MemObjectType::Image1dBuffer,
                    m * k / 4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    Some(a_buf.clone()),
                );
                unsafe {
                    ocl::core::create_image(
                        self.context.as_core(),
                        ocl::core::MEM_READ_ONLY,
                        &fmt,
                        &desc,
                        None::<&[f32]>,
                        None,
                    )?
                }
            };

            let rows_i = (m / 4) as i32;
            let cols_i = (k / 4) as i32;
            let padded_rows_i = (padded_m / 4) as i32;

            unsafe {
                ocl::core::set_kernel_arg(
                    transpose_kernel,
                    0,
                    ocl::core::ArgVal::mem(&act_in_img),
                )?;
                ocl::core::set_kernel_arg(
                    transpose_kernel,
                    1,
                    ocl::core::ArgVal::mem(act_out_img),
                )?;
                ocl::core::set_kernel_arg(transpose_kernel, 2, ocl::core::ArgVal::scalar(&rows_i))?;
                ocl::core::set_kernel_arg(transpose_kernel, 3, ocl::core::ArgVal::scalar(&cols_i))?;
                ocl::core::set_kernel_arg(
                    transpose_kernel,
                    4,
                    ocl::core::ArgVal::scalar(&padded_rows_i),
                )?;

                let gws_t: [usize; 3] = [k / 4, padded_m / 4, 1];
                // Leave LWS to the driver. llama.cpp uses `[1, 16]` but that
                // constrains `padded_m/4 % 16 == 0`; Qwen 2.5-1.5B with M=131
                // pads to 136 → padded_m/4 = 34, which violates `gws % lws == 0`.
                self.enqueue_kernel_labeled(transpose_kernel, "matmul_transpose", 2, &gws_t, None)?;
            }
            *trans_key_slot = Some(activation_key);
        }

        // ── Stage 2: Q4_0 × F16 GEMM ─────────────────────────────────────
        //
        // `b_in_img` is the persistent RGBA16F read-only view of
        // `act_trans_mem` held in the cache slot.

        let m_i = n as i32; // kernel's m = output dim (rows of weight)
        let n_padded_i = padded_m as i32; // kernel's n = padded token count
        let k_i = k as i32;
        let n_no_padding_i = m as i32; // real token count

        unsafe {
            ocl::core::set_kernel_arg(gemm_kernel, 0, ocl::core::ArgVal::mem(&entry.q_buf))?;
            ocl::core::set_kernel_arg(gemm_kernel, 1, ocl::core::ArgVal::mem(&entry.d_buf))?;
            ocl::core::set_kernel_arg(gemm_kernel, 2, ocl::core::ArgVal::mem(b_in_img))?;
            ocl::core::set_kernel_arg(gemm_kernel, 3, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(gemm_kernel, 4, ocl::core::ArgVal::scalar(&m_i))?;
            ocl::core::set_kernel_arg(gemm_kernel, 5, ocl::core::ArgVal::scalar(&n_padded_i))?;
            ocl::core::set_kernel_arg(gemm_kernel, 6, ocl::core::ArgVal::scalar(&k_i))?;
            ocl::core::set_kernel_arg(gemm_kernel, 7, ocl::core::ArgVal::scalar(&n_no_padding_i))?;

            // gws = [ceil(N/8), M/4, 1] = [padded_m/8, n/4, 1]. llama.cpp
            // dispatches this same kernel with a *fixed* LWS of [1, 128, 1]
            // even for shapes where `n/4 % 128 != 0` (e.g. Qwen's FFN
            // n=8960 → n/4=2240, remainder 64). The dispatch works on
            // Adreno OpenCL 3.0 + CL2.0-compiled kernels because
            // non-uniform workgroups are supported; the `qcom_reqd_sub_group_size("full")`
            // attribute pins the lane count without constraining the WG shape.
            //
            // We match llama.cpp's behaviour by always using [1, 128, 1].
            // If Adreno's driver rejects the non-uniform remainder for
            // `n/4 % 128 != 0` we'll see `CL_INVALID_WORK_GROUP_SIZE` at
            // dispatch time; callers then re-run with the generic GEMM.
            let gws_g: [usize; 3] = [padded_m / 8, n / 4, 1];
            let lws_g: [usize; 3] = [1, 128, 1];
            self.enqueue_kernel_labeled(gemm_kernel, "matmul", 3, &gws_g, Some(lws_g))?;
        }

        // act_in_img / act_out_img / b_in_img drop here; the enqueued kernels
        // hold their own references via clRetainMemObject.
        Ok(())
    }

    /// GPU flash attention for prefill. Dispatches flash_attn_f32 or flash_attn_f32_f16 kernel.
    /// Returns Ok(true) if GPU kernel was dispatched, Ok(false) if unsupported (caller should
    /// fall back to CPU).
    ///
    /// Q: [batch, seq_len, n_heads_q, head_dim] F32 (on GPU)
    /// K/V cache: HeadMajor [1, kv_heads, capacity, head_dim] F32 or F16 (on GPU)
    /// Output: [batch, seq_len, n_heads_q * head_dim] F32 (on GPU)
    ///
    /// Supported head_dim values:
    /// - F32 KV: 64 only (`flash_attn_f32` compiled at DK=64).
    /// - F16 KV: 64 and 128 (two DK variants of `flash_attn_f32_f16`).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_prefill_gpu(
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
        is_head_major: bool,
    ) -> Result<bool> {
        let kv_dtype = k_cache.dtype();
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match kv_dtype {
            DType::F32 => match head_dim {
                64 => match &kernels.kernel_flash_attn_f32 {
                    Some(k) => k,
                    None => return Ok(false),
                },
                256 => match &kernels.kernel_flash_attn_f32_dk256 {
                    Some(k) => k,
                    None => return Ok(false),
                },
                _ => return Ok(false),
            },
            DType::F16 => match head_dim {
                64 => match &kernels.kernel_flash_attn_f32_f16_dk64 {
                    Some(k) => k,
                    None => return Ok(false),
                },
                128 => match &kernels.kernel_flash_attn_f32_f16_dk128 {
                    Some(k) => {
                        log_prefill_dk128_once();
                        k
                    }
                    None => return Ok(false),
                },
                _ => return Ok(false),
            },
            _ => return Ok(false), // Q4_0 not supported by flash attn kernel
        };

        let q_buf = get_cl_mem(q.buffer().as_ref())?;
        let k_buf = get_cl_mem(k_cache.buffer().as_ref())?;
        let v_buf = get_cl_mem(v_cache.buffer().as_ref())?;
        let o_buf = get_cl_mem(out.buffer().as_ref())?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let kv_elem_size: usize = match kv_dtype {
            DType::F16 => 2,
            _ => 4,
        };

        // Q strides in bytes (F32, row-major [batch, seq_len, n_heads_q, head_dim])
        let q_nb1 = (n_heads_q * head_dim * 4) as u64; // position stride
        let q_nb2 = (head_dim * 4) as u64; // head stride
        let q_nb3 = seq_len as u64 * q_nb1; // batch stride

        // KV strides in bytes
        let (k_nb1, k_nb2, k_nb3) = if is_head_major {
            // HeadMajor: [batch, kv_heads, capacity, head_dim]
            let pos_stride = (head_dim * kv_elem_size) as u64;
            let head_stride = (kv_capacity * head_dim * kv_elem_size) as u64;
            let batch_stride = n_heads_kv as u64 * head_stride;
            (pos_stride, head_stride, batch_stride)
        } else {
            // SeqMajor: [batch, capacity, kv_heads, head_dim]
            let pos_stride = (n_heads_kv * head_dim * kv_elem_size) as u64;
            let head_stride = (head_dim * kv_elem_size) as u64;
            let batch_stride = kv_capacity as u64 * pos_stride;
            (pos_stride, head_stride, batch_stride)
        };
        let (v_nb1, v_nb2, v_nb3) = (k_nb1, k_nb2, k_nb3);

        // Output strides in bytes (F32, [batch, seq_len, n_heads_q, head_dim])
        let o_nb1 = (head_dim * 4) as u64; // head stride
        let o_nb2 = (n_heads_q * head_dim * 4) as u64; // position stride
        let o_nb3 = seq_len as u64 * o_nb2; // batch stride

        let n_q = seq_len as i32;
        let n_kv = cache_seq_len as i32;
        let is_causal = 1i32;
        let n_head = n_heads_q as i32;
        let n_head_kv_arg = n_heads_kv as i32;
        let max_bias = 0.0f32;
        let m0 = 0.0f32;
        let m1 = 0.0f32;
        let n_head_log2 = 0i32;
        let logit_softcap = 0.0f32;
        let zero_u64 = 0u64;
        let zero_i32 = 0i32;

        unsafe {
            // Q, K, V, O buffers + offsets (args 0-7)
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(k_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(v_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::mem(o_buf))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&zero_u64))?;
            // Scalar params (args 8-12)
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&n_q))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&n_kv))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&is_causal))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&n_head))?;
            // Q strides (args 13-15)
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&q_nb1))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&q_nb2))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&q_nb3))?;
            // K strides (args 16-18)
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&k_nb1))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&k_nb2))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&k_nb3))?;
            // V strides (args 19-21)
            ocl::core::set_kernel_arg(kernel, 19, ocl::core::ArgVal::scalar(&v_nb1))?;
            ocl::core::set_kernel_arg(kernel, 20, ocl::core::ArgVal::scalar(&v_nb2))?;
            ocl::core::set_kernel_arg(kernel, 21, ocl::core::ArgVal::scalar(&v_nb3))?;
            // O strides (args 22-24)
            ocl::core::set_kernel_arg(kernel, 22, ocl::core::ArgVal::scalar(&o_nb1))?;
            ocl::core::set_kernel_arg(kernel, 23, ocl::core::ArgVal::scalar(&o_nb2))?;
            ocl::core::set_kernel_arg(kernel, 24, ocl::core::ArgVal::scalar(&o_nb3))?;
            // ALiBi params — unused (args 25-29)
            ocl::core::set_kernel_arg(kernel, 25, ocl::core::ArgVal::scalar(&max_bias))?;
            ocl::core::set_kernel_arg(kernel, 26, ocl::core::ArgVal::scalar(&m0))?;
            ocl::core::set_kernel_arg(kernel, 27, ocl::core::ArgVal::scalar(&m1))?;
            ocl::core::set_kernel_arg(kernel, 28, ocl::core::ArgVal::scalar(&n_head_log2))?;
            ocl::core::set_kernel_arg(kernel, 29, ocl::core::ArgVal::scalar(&logit_softcap))?;
            // n_head_kv (arg 30)
            ocl::core::set_kernel_arg(kernel, 30, ocl::core::ArgVal::scalar(&n_head_kv_arg))?;
            // mask = NULL (args 31-37)
            ocl::core::set_kernel_arg(kernel, 31, ocl::core::ArgVal::mem_null())?;
            ocl::core::set_kernel_arg(kernel, 32, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 33, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 34, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 35, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 36, ocl::core::ArgVal::scalar(&zero_i32))?;
            ocl::core::set_kernel_arg(kernel, 37, ocl::core::ArgVal::scalar(&zero_i32))?;
            // sinks = NULL (args 38-39)
            ocl::core::set_kernel_arg(kernel, 38, ocl::core::ArgVal::mem_null())?;
            ocl::core::set_kernel_arg(kernel, 39, ocl::core::ArgVal::scalar(&zero_u64))?;

            // Work size: [ceil(n_q/block_m) * lanes_per_wg, n_heads_q * batch_size, 1]
            // BLOCK_M = Q-rows per WG (matches the -DBLOCK_M compile-time macro).
            // lanes_per_wg = threads per WG:
            //   head_dim=64  → BLOCK_M=64, 1 thread / Q-row → lanes_per_wg=64 (dk64 program, unchanged)
            //   head_dim=128 → BLOCK_M=32, 2 threads / Q-row (A-3 B-1 subgroup split) → lanes_per_wg=64
            let (block_m, lanes_per_wg): (usize, usize) = match head_dim {
                64 => (64, 64),
                128 => (32, 64),
                // DK=256: WG_SIZE = BLOCK_M = 32 to match NVIDIA warp size.
                256 => (32, 32),
                _ => unreachable!("head_dim guard above"),
            };
            let n_groups_q = seq_len.div_ceil(block_m);
            let global_work_size: [usize; 3] =
                [n_groups_q * lanes_per_wg, n_heads_q * batch_size, 1];
            let local_work_size: [usize; 3] = [lanes_per_wg, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "flash_prefill",
                2,
                &global_work_size,
                Some(local_work_size),
            )?;

            // DK=256 stability guard: the kernel's large per-thread register
            // footprint (q_priv[64] + o_acc[64] float4 tiles) pressures the
            // NVIDIA driver when 34 back-to-back layers' dispatches share the
            // command queue without intermediate drains. An explicit finish
            // breaks the chain and keeps multi-question eval-ll stable.
            // No-op on head_dim ∈ {64, 128} where the existing kernels are
            // register-light, and on Adreno (isolated by the head_dim == 256
            // guard itself — no production Gemma3 path on Adreno today).
            if head_dim == 256 {
                ocl::core::finish(&self.queue)?;
            }
        }

        Ok(true)
    }

    /// Decode-specialized flash attention: single query per head, online softmax,
    /// zero intermediate score buffer. Dispatches `flash_attn_f32_f16_q1` from the
    /// DK variant program matching the runtime `head_dim`.
    ///
    /// Supports head_dim ∈ {64, 128} today. Requires F16 KV on GPU,
    /// HeadMajor KV layout, `seq_len=1`, and no score output.
    /// Returns Ok(true) if the kernel was dispatched; Ok(false) if preconditions
    /// are not met and the caller should fall back to the legacy attention kernel.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_decode_gpu(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        // Optional post-softmax score output. When `Some((buf, layer_offset, stride))`,
        // the kernel writes per-token weights into
        // `buf[layer_offset + head_idx * stride + t]` (matches the
        // `GpuScoreAccumulator` layout in `gpu_score.rs`). `None` forces the
        // dummy buffer + `write_scores=0` fast path.
        score_buf: Option<(&ocl::core::Mem, i32, i32)>,
    ) -> Result<bool> {
        // Only F16 KV on HeadMajor GPU buffer is supported.
        if k_cache.dtype() != DType::F16 {
            return Ok(false);
        }
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == n_heads_kv && k_shape[1] != k_shape[2];
        if !is_head_major {
            return Ok(false);
        }
        let kv_capacity = k_shape[2];

        // Head dim must match the DK/DV compile-time constant of one of the
        // compiled flash attention programs. Add new variants here when
        // adding a new head_dim (remember to compile the program in
        // `OpenCLBackend::new` and cache the kernel in `KernelCache`).
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match head_dim {
            64 => match &kernels.kernel_flash_attn_f32_f16_q1_dk64 {
                Some(k) => k,
                None => return Ok(false),
            },
            128 => match &kernels.kernel_flash_attn_f32_f16_q1_dk128 {
                Some(k) => k,
                None => return Ok(false),
            },
            _ => return Ok(false),
        };

        let q_buf = get_cl_mem(q.buffer().as_ref())?;
        let k_buf = get_cl_mem(k_cache.buffer().as_ref())?;
        let v_buf = get_cl_mem(v_cache.buffer().as_ref())?;
        let o_buf = get_cl_mem(out.buffer().as_ref())?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Q strides in bytes (F32 [batch=1, seq_len=1, n_heads_q, head_dim])
        let q_nb1 = (n_heads_q * head_dim * 4) as u64; // position stride (unused, seq=1)
        let q_nb2 = (head_dim * 4) as u64; // head stride
        let q_nb3 = q_nb1; // batch stride (seq=1)

        // KV strides in bytes — HeadMajor [1, kv_heads, capacity, head_dim]
        let kv_elem_size: u64 = 2; // F16
        let k_nb1 = (head_dim as u64) * kv_elem_size; // position stride within a head
        let k_nb2 = (kv_capacity * head_dim) as u64 * kv_elem_size; // head stride
        let k_nb3 = (n_heads_kv as u64) * k_nb2; // batch stride
        let (v_nb1, v_nb2, v_nb3) = (k_nb1, k_nb2, k_nb3);

        // Output strides in bytes (F32 [batch=1, seq_len=1, n_heads_q, head_dim])
        let o_nb1 = (head_dim * 4) as u64; // head stride (q1 uses this for head offset)
        let o_nb2 = (n_heads_q * head_dim * 4) as u64; // position stride (unused)
        let o_nb3 = o_nb2; // batch stride

        let n_q = 1i32;
        let n_kv = cache_seq_len as i32;
        let is_causal = 0i32; // single query, no causal mask needed
        let n_head = n_heads_q as i32;
        let n_head_kv_arg = n_heads_kv as i32;
        let max_bias = 0.0f32;
        let m0 = 0.0f32;
        let m1 = 0.0f32;
        let n_head_log2 = 0i32;
        let logit_softcap = 0.0f32;
        let zero_u64 = 0u64;
        let zero_i32 = 0i32;

        unsafe {
            // Q, K, V, O buffers + offsets (args 0-7)
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(k_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(v_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::mem(o_buf))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&zero_u64))?;
            // Scalar params (args 8-12)
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&n_q))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&n_kv))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&is_causal))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&n_head))?;
            // Q strides (args 13-15)
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&q_nb1))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&q_nb2))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&q_nb3))?;
            // K strides (args 16-18)
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&k_nb1))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&k_nb2))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&k_nb3))?;
            // V strides (args 19-21)
            ocl::core::set_kernel_arg(kernel, 19, ocl::core::ArgVal::scalar(&v_nb1))?;
            ocl::core::set_kernel_arg(kernel, 20, ocl::core::ArgVal::scalar(&v_nb2))?;
            ocl::core::set_kernel_arg(kernel, 21, ocl::core::ArgVal::scalar(&v_nb3))?;
            // O strides (args 22-24)
            ocl::core::set_kernel_arg(kernel, 22, ocl::core::ArgVal::scalar(&o_nb1))?;
            ocl::core::set_kernel_arg(kernel, 23, ocl::core::ArgVal::scalar(&o_nb2))?;
            ocl::core::set_kernel_arg(kernel, 24, ocl::core::ArgVal::scalar(&o_nb3))?;
            // ALiBi params — unused (args 25-29)
            ocl::core::set_kernel_arg(kernel, 25, ocl::core::ArgVal::scalar(&max_bias))?;
            ocl::core::set_kernel_arg(kernel, 26, ocl::core::ArgVal::scalar(&m0))?;
            ocl::core::set_kernel_arg(kernel, 27, ocl::core::ArgVal::scalar(&m1))?;
            ocl::core::set_kernel_arg(kernel, 28, ocl::core::ArgVal::scalar(&n_head_log2))?;
            ocl::core::set_kernel_arg(kernel, 29, ocl::core::ArgVal::scalar(&logit_softcap))?;
            // n_head_kv (arg 30)
            ocl::core::set_kernel_arg(kernel, 30, ocl::core::ArgVal::scalar(&n_head_kv_arg))?;
            // mask = NULL (args 31-37)
            ocl::core::set_kernel_arg(kernel, 31, ocl::core::ArgVal::mem_null())?;
            ocl::core::set_kernel_arg(kernel, 32, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 33, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 34, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 35, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 36, ocl::core::ArgVal::scalar(&zero_i32))?;
            ocl::core::set_kernel_arg(kernel, 37, ocl::core::ArgVal::scalar(&zero_i32))?;
            // sinks = NULL (args 38-39)
            ocl::core::set_kernel_arg(kernel, 38, ocl::core::ArgVal::mem_null())?;
            ocl::core::set_kernel_arg(kernel, 39, ocl::core::ArgVal::scalar(&zero_u64))?;
            // Post-softmax score output (args 40-43). The kernel requires a
            // valid buffer even when `write_scores=0`; bind the backend-wide
            // 1-element dummy in that case.
            let (s_buf, s_layer_offset, s_stride, write_scores) = match score_buf {
                Some((buf, off, stride)) => (buf, off, stride, 1i32),
                None => (&self.dummy_score_buf, 0i32, 0i32, 0i32),
            };
            ocl::core::set_kernel_arg(kernel, 40, ocl::core::ArgVal::mem(s_buf))?;
            ocl::core::set_kernel_arg(kernel, 41, ocl::core::ArgVal::scalar(&s_layer_offset))?;
            ocl::core::set_kernel_arg(kernel, 42, ocl::core::ArgVal::scalar(&s_stride))?;
            ocl::core::set_kernel_arg(kernel, 43, ocl::core::ArgVal::scalar(&write_scores))?;

            // Q1_WG_SIZE = 64 (compile-time constant in the kernel)
            const Q1_WG_SIZE: usize = 64;
            // Global = [Q1_WG_SIZE, n_heads_q * batch (= n_heads_q for decode batch=1), 1]
            // Each WG handles one (batch, head) pair.
            let global_work_size: [usize; 3] = [Q1_WG_SIZE, n_heads_q, 1];
            let local_work_size: [usize; 3] = [Q1_WG_SIZE, 1, 1];
            self.enqueue_kernel_labeled(
                kernel,
                "attention",
                2,
                &global_work_size,
                Some(local_work_size),
            )?;
        }

        Ok(true)
    }

    /// AUF SOA bypass helper — alloc q_buf/d_buf, **non-blocking**-write the
    /// pre-transposed payload, and (best-effort) build the
    /// `image1d_buffer_t` view. Counts the new cl_mem objects in the
    /// `auf_soa_*` diag buckets.
    ///
    /// # Async enqueue contract (ENG-ALG-230 / INV-142)
    ///
    /// Both `enqueue_write_buffer` calls use `blocking = false` so all 175 tensors
    /// in a full-ratio AUF swap are queued without a synchronous wait per tensor
    /// (~150 ms saved vs the previous blocking path). The OpenCL in-order command
    /// queue guarantees ordering: writes complete before any subsequent kernel
    /// dispatch on the same queue.
    ///
    /// **Callers must call `backend.synchronize()` (= `clFinish`) exactly once
    /// at the stage gate, after all `materialise_weight` calls complete and before
    /// `invalidate_noshuffle_soa_registry` / `restore_pre_converted_soa_registration`
    /// read the uploaded data (ENG-ALG-231).**
    ///
    /// The 4-step legacy fallback path in `convert_q4_0_to_noshuffle` retains
    /// `blocking = true` as a conservative choice — that path is only active when
    /// the fused kernel is unavailable, and the intermediate CPU transpose already
    /// serialises the write anyway.
    ///
    /// Caller is responsible for sizing checks; this helper assumes
    /// `q_bytes.len() == num_blocks * 16` and `d_bytes.len() == num_blocks * 2`.
    fn alloc_and_upload_soa_buffers(
        &self,
        q_bytes: &[u8],
        d_bytes: &[u8],
        ne00: usize,
        ne01: usize,
        num_blocks: usize,
    ) -> Result<(ocl::core::Mem, ocl::core::Mem, Option<ocl::core::Mem>)> {
        let expected_q = num_blocks * 16;
        let q_buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                self.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                expected_q,
                None,
            )?
        };
        self.record_cl_mem_alloc("auf_soa_q", expected_q);
        let d_buf = unsafe {
            ocl::core::create_buffer::<_, u16>(
                self.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                num_blocks,
                None,
            )?
        };
        self.record_cl_mem_alloc("auf_soa_d", num_blocks * 2);
        // ENG-ALG-230: non-blocking enqueue — the in-order queue guarantees
        // ordering. The caller (`execute_on_slots` stage gate) issues a single
        // `backend.synchronize()` after all layers are materialised.
        // SAFETY: `q_bytes` and `d_bytes` slices are valid for the duration of
        // this unsafe block; they are borrowed from the AUF mmap slice which
        // lives for the entire swap batch.
        unsafe {
            ocl::core::enqueue_write_buffer(
                &self.queue,
                &q_buf,
                false, // non-blocking — ENG-ALG-230
                0,
                q_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                &self.queue,
                &d_buf,
                false, // non-blocking — ENG-ALG-230
                0,
                d_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }

        // Build the optional image1d_buffer_t view over q_buf — same
        // signature as `convert_q4_0_to_noshuffle` step 4. Failure → None
        // (the GEMV path falls back to plain buffer reads).
        let cols_ushort = ne00 / 4;
        let q_total_uint = (ne01 * cols_ushort) / 2;
        let q_img = {
            use ocl::core::{
                ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat,
                MemObjectType,
            };
            let q_img_fmt =
                ImageFormat::new(ImageChannelOrder::R, ImageChannelDataType::UnsignedInt32);
            let q_img_desc = ImageDescriptor::new(
                MemObjectType::Image1dBuffer,
                q_total_uint,
                0,
                0,
                0,
                0,
                0,
                Some(q_buf.clone()),
            );
            // SAFETY: q_buf is a valid CL buffer with q_total_uint * 4 bytes;
            // image1d_buffer_t is a read-only view over the same memory.
            let result = unsafe {
                ocl::core::create_image(
                    self.context.as_core(),
                    ocl::core::MEM_READ_ONLY,
                    &q_img_fmt,
                    &q_img_desc,
                    None::<&[u32]>,
                    None,
                )
            };
            match result {
                Ok(img) => {
                    self.record_cl_mem_alloc("auf_soa_img", q_total_uint * 4);
                    Some(img)
                }
                Err(e) => {
                    log::warn!(
                        "AUF SOA bypass: image1d_buffer_t creation failed (ne01={}), \
                         falling back to buffer path: {}",
                        ne01,
                        e
                    );
                    None
                }
            }
        };

        Ok((q_buf, d_buf, q_img))
    }

    // ── LISWAP-2 prototype: async H2D transfer queue ─────────────────────
    // Plan: .claude/plans/compiled-chasing-hopper.md (Phase 1).
    //
    // The prototype uses a *secondary* command queue dedicated to H2D weight
    // uploads so that compute (Q4 GEMV / flash attn / FFN) on `self.queue`
    // can overlap with the in-flight write at the hardware level. cl_mem
    // handles are bound to the OpenCL *context*, not the queue, so a buffer
    // allocated through the standard `OpenCLMemory` (which holds
    // `self.queue`) is fully usable from the transfer queue too.

    /// Whether the LISWAP-2 transfer queue is enabled at runtime.
    /// Reads `LLMRS_OPENCL_TRANSFER_QUEUE` once and caches the answer.
    /// "0" / "false" / "off" (case-insensitive) → disabled (caller falls
    /// back to the synchronous `copy_weight_from` path). Anything else,
    /// including an absent env var, → enabled (default ON).
    fn transfer_queue_env_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| match std::env::var("LLMRS_OPENCL_TRANSFER_QUEUE") {
            Ok(v) => {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "false" || v == "off")
            }
            Err(_) => true,
        })
    }

    /// Lazily build (or return) the secondary transfer queue. Returns
    /// `None` if the env-gate is disabled or queue creation failed —
    /// callers fall back to the synchronous path in that case. Queue
    /// properties match the main queue (profiling stays consistent).
    fn transfer_queue_or_init(&self) -> Option<&Queue> {
        if !Self::transfer_queue_env_enabled() {
            return None;
        }
        // Fast path: already initialised.
        if let Some(q) = self.transfer_queue.get() {
            return Some(q);
        }
        // Slow path: try to create. Match the main queue's profiling
        // setting so per-event timings stay comparable; we deliberately
        // keep the default priority (Phase 5 may experiment with low).
        let queue_props = if self.profile_events_enabled {
            Some(flags::QUEUE_PROFILING_ENABLE)
        } else {
            None
        };
        // Match the main queue's OoO setting so commands across queues see
        // the same scheduling semantics.
        let queue_props = if std::env::var("LLMRS_OPENCL_OOO_QUEUE").is_ok() {
            Some(
                queue_props.unwrap_or(flags::CommandQueueProperties::empty())
                    | flags::QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            )
        } else {
            queue_props
        };
        match Queue::new(&self.context, self.device, queue_props) {
            Ok(q) => {
                // OnceLock::set returns Err if another thread won the
                // race; in either case `get()` will yield the live value.
                let _ = self.transfer_queue.set(q);
                self.transfer_queue.get()
            }
            Err(e) => {
                log::warn!(
                    "LLMRS_OPENCL_TRANSFER_QUEUE: secondary queue creation failed ({e}); \
                     falling back to synchronous `copy_weight_from`"
                );
                None
            }
        }
    }

    // ── LISWAP-3 prototype: ALLOC_HOST_PTR pool for swap ─────────────────
    //
    // The pool replaces the per-tensor `clCreateBuffer + enqueue_write_buffer`
    // staging cycle with a `clEnqueueMapBuffer(MAP_WRITE) + memcpy + Unmap`
    // path on a pre-allocated `CL_MEM_ALLOC_HOST_PTR | READ_ONLY` slot. On
    // Adreno UMA the unmap step flushes CPU caches and makes the bytes GPU-
    // visible without any device-side copy. Stage 1b microbench reported
    // -42.4% wall-clock on Galaxy S25 vs the staging baseline.
    //
    // Stage 3 keeps the pool path off-by-default so production swap behaviour
    // is unchanged. Caller (SwapExecutor) must check both the CLI flag and
    // the env-gate before requesting the pool path.

    /// Allocate an empty `CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY` cl_mem
    /// of `size` bytes (no initial data). Caller fills it via
    /// [`fill_host_ptr_buffer`][Self::fill_host_ptr_buffer].
    ///
    /// Used by `HostPtrPool::new` to build slot buffers up-front; the
    /// production swap path itself never calls this directly — it acquires
    /// a slot via the pool and reuses the pre-allocated cl_mem.
    ///
    /// Plan: `compiled-chasing-hopper.md` Direction A track.
    /// Allocate a DMA-BUF-backed cl_mem via Linux dma_heap + KHR
    /// external_memory_dma_buf extension. Returns `(cl_mem, dmabuf_fd,
    /// host_ptr)`. CPU writes go through `host_ptr` (cached, no Map/Unmap).
    /// GPU reads via `cl_mem` are DMA-BUF coherent without explicit sync.
    ///
    /// Caller is responsible for unmapping `host_ptr` and closing
    /// `dmabuf_fd` when the cl_mem is dropped. The cl_mem itself releases
    /// via the `ocl::core::Mem` Drop chain (`clReleaseMemObject`).
    ///
    /// # Errors
    /// - `/dev/dma_heap/system` not available (older kernel) → returns Err.
    /// - `clCreateBufferWithProperties` not supported → returns Err.
    pub fn alloc_dmabuf_heap_buffer(
        &self,
        size: usize,
    ) -> Result<(
        ocl::core::Mem,
        std::os::unix::io::RawFd,
        *mut std::ffi::c_void,
    )> {
        use std::ffi::c_void;
        use std::os::unix::io::RawFd;
        use std::ptr;

        const CL_MEM_READ_ONLY: u64 = 1 << 2;
        const CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR: u64 = 0x2067;

        // 1. Open /dev/dma_heap/system
        let path = c"/dev/dma_heap/system";
        let heap_fd: RawFd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY) };
        if heap_fd < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(anyhow!(
                "alloc_dmabuf_heap_buffer: open(/dev/dma_heap/system) failed: errno={errno}"
            ));
        }

        // 2. ioctl ALLOC. _IOWR('H', 0, struct dma_heap_allocation_data)
        // = ((1+2)<<30) | (sizeof << 16) | ('H' << 8) | 0
        #[repr(C)]
        struct DmaHeapAllocationData {
            len: u64,
            fd: u32,
            fd_flags: u32,
            heap_flags: u64,
        }
        let ioctl_alloc: u64 = (3u64 << 30)
            | ((std::mem::size_of::<DmaHeapAllocationData>() as u64) << 16)
            | ((b'H' as u64) << 8);
        let mut data = DmaHeapAllocationData {
            len: size as u64,
            fd: 0,
            fd_flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
            heap_flags: 0,
        };
        unsafe extern "C" {
            fn ioctl(fd: libc::c_int, request: libc::c_ulong, ...) -> libc::c_int;
        }
        let rc = unsafe { ioctl(heap_fd, ioctl_alloc as libc::c_ulong, &mut data) };
        let heap_close_errno = if rc < 0 {
            std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
        } else {
            0
        };
        unsafe { libc::close(heap_fd) };
        if rc < 0 {
            return Err(anyhow!(
                "alloc_dmabuf_heap_buffer: ioctl ALLOC failed (size={size}, errno={heap_close_errno})"
            ));
        }
        let dmabuf_fd: RawFd = data.fd as i32;

        // 3. mmap the DMA-BUF for CPU access
        let host_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                dmabuf_fd,
                0,
            )
        };
        if host_ptr == libc::MAP_FAILED {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            unsafe { libc::close(dmabuf_fd) };
            return Err(anyhow!(
                "alloc_dmabuf_heap_buffer: mmap failed (size={size}, errno={errno})"
            ));
        }

        // 4. clCreateBufferWithProperties with DMA-BUF property
        let props: [u64; 3] = [CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, dmabuf_fd as u64, 0];
        let ctx_ptr = <&Context as ClContextPtr>::as_ptr(&&self.context);
        unsafe extern "C" {
            fn clCreateBufferWithProperties(
                context: *mut c_void,
                properties: *const u64,
                flags: u64,
                size: usize,
                host_ptr: *mut c_void,
                errcode_ret: *mut i32,
            ) -> *mut c_void;
        }
        let mut errcode: i32 = 0;
        let raw = unsafe {
            clCreateBufferWithProperties(
                ctx_ptr as *mut _,
                props.as_ptr(),
                CL_MEM_READ_ONLY,
                size,
                ptr::null_mut(),
                &mut errcode,
            )
        };
        if errcode != 0 || raw.is_null() {
            unsafe {
                libc::munmap(host_ptr, size);
                libc::close(dmabuf_fd);
            }
            return Err(anyhow!(
                "alloc_dmabuf_heap_buffer: clCreateBufferWithProperties failed (errcode={errcode})"
            ));
        }
        // SAFETY: clCreateBufferWithProperties returned a valid cl_mem owning
        // one reference. ocl::core::Mem::from_raw_create_ptr takes ownership.
        let mem = unsafe { ocl::core::Mem::from_raw_create_ptr(raw as *mut _) };

        Ok((mem, dmabuf_fd, host_ptr))
    }

    /// Multi-context swap experiment env-gate. Cached after first call.
    /// `=1` / `=true` / `=on` (case-insensitive) enables the secondary
    /// `cl_context` path; any other value (including unset) keeps it
    /// disabled (default OFF).
    fn swap_context_env_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| match std::env::var("LLMRS_OPENCL_SWAP_CONTEXT") {
            Ok(v) => {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "on"
            }
            Err(_) => false,
        })
    }

    /// Lazily build (or return) the secondary `cl_context` dedicated to swap
    /// Map/Unmap operations. Returns `None` when the env-gate is disabled or
    /// context creation failed.
    ///
    /// The secondary context targets the same `cl_device_id` as the main
    /// context. DMA-BUF FDs are shared between both contexts so a single
    /// physical allocation is visible from either one's `cl_mem`.
    pub fn swap_context_or_init(&self) -> Option<&Context> {
        if !Self::swap_context_env_enabled() {
            return None;
        }
        if let Some(c) = self.swap_context.get() {
            return Some(c);
        }
        match Context::builder()
            .platform(Platform::default())
            .devices(self.device)
            .build()
        {
            Ok(c) => {
                let _ = self.swap_context.set(c);
                self.swap_context.get()
            }
            Err(e) => {
                log::warn!(
                    "LLMRS_OPENCL_SWAP_CONTEXT: secondary context creation failed ({e}); \
                     falling back to single-context swap path"
                );
                None
            }
        }
    }

    /// Lazily build (or return) the swap queue inside the secondary
    /// `cl_context`. Returns `None` when the secondary context is unavailable
    /// or queue creation failed.
    pub fn swap_queue_or_init(&self) -> Option<&Queue> {
        if !Self::swap_context_env_enabled() {
            return None;
        }
        if let Some(q) = self.swap_queue.get() {
            return Some(q);
        }
        let ctx = self.swap_context_or_init()?;
        // Match main queue profiling/OoO settings so cross-context comparisons
        // remain meaningful.
        let queue_props = if self.profile_events_enabled {
            Some(flags::QUEUE_PROFILING_ENABLE)
        } else {
            None
        };
        let queue_props = if std::env::var("LLMRS_OPENCL_OOO_QUEUE").is_ok() {
            Some(
                queue_props.unwrap_or(flags::CommandQueueProperties::empty())
                    | flags::QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            )
        } else {
            queue_props
        };
        match Queue::new(ctx, self.device, queue_props) {
            Ok(q) => {
                let _ = self.swap_queue.set(q);
                self.swap_queue.get()
            }
            Err(e) => {
                log::warn!(
                    "LLMRS_OPENCL_SWAP_CONTEXT: swap queue creation failed ({e}); \
                     falling back to single-context swap path"
                );
                None
            }
        }
    }

    /// Allocate a DMA-BUF-backed pair of `cl_mem` handles, one in the
    /// secondary swap `cl_context` and one in the main `cl_context`, both
    /// referencing the same DMA-BUF FD.
    ///
    /// Returns `(swap_mem, main_mem, dmabuf_fd, host_ptr)`. The caller fills
    /// the bytes via `fill_dmabuf_via_swap_queue` (uses `swap_mem` on the
    /// swap queue) and reads the bytes from forward kernels via `main_mem`
    /// on the main queue. Driver coherency between the two `cl_mem` views is
    /// guaranteed by the underlying DMA-BUF.
    ///
    /// On Drop the caller must `munmap` `host_ptr` and `close` `dmabuf_fd`
    /// (matches the lifecycle rules of `alloc_dmabuf_heap_buffer`).
    ///
    /// # Errors
    /// - secondary swap context unavailable (env-gate off) → returns Err.
    /// - `/dev/dma_heap/system` not available → returns Err.
    /// - `clCreateBufferWithProperties` failed on either context → Err.
    pub fn alloc_dmabuf_with_swap_context(
        &self,
        size: usize,
    ) -> Result<(
        ocl::core::Mem,
        ocl::core::Mem,
        std::os::unix::io::RawFd,
        *mut std::ffi::c_void,
    )> {
        use std::ffi::c_void;
        use std::os::unix::io::RawFd;
        use std::ptr;

        const CL_MEM_READ_ONLY: u64 = 1 << 2;
        const CL_MEM_READ_WRITE: u64 = 1 << 0;
        const CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR: u64 = 0x2067;

        let swap_ctx = self.swap_context_or_init().ok_or_else(|| {
            anyhow!(
                "alloc_dmabuf_with_swap_context: secondary cl_context unavailable \
                 (env-gate LLMRS_OPENCL_SWAP_CONTEXT off, or context build failed)"
            )
        })?;

        // 1. Open /dev/dma_heap/system
        let path = c"/dev/dma_heap/system";
        let heap_fd: RawFd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY) };
        if heap_fd < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(anyhow!(
                "alloc_dmabuf_with_swap_context: open(/dev/dma_heap/system) failed: errno={errno}"
            ));
        }

        // 2. ioctl ALLOC.
        #[repr(C)]
        struct DmaHeapAllocationData {
            len: u64,
            fd: u32,
            fd_flags: u32,
            heap_flags: u64,
        }
        let ioctl_alloc: u64 = (3u64 << 30)
            | ((std::mem::size_of::<DmaHeapAllocationData>() as u64) << 16)
            | ((b'H' as u64) << 8);
        let mut data = DmaHeapAllocationData {
            len: size as u64,
            fd: 0,
            fd_flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
            heap_flags: 0,
        };
        unsafe extern "C" {
            fn ioctl(fd: libc::c_int, request: libc::c_ulong, ...) -> libc::c_int;
        }
        let rc = unsafe { ioctl(heap_fd, ioctl_alloc as libc::c_ulong, &mut data) };
        let heap_close_errno = if rc < 0 {
            std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
        } else {
            0
        };
        unsafe { libc::close(heap_fd) };
        if rc < 0 {
            return Err(anyhow!(
                "alloc_dmabuf_with_swap_context: ioctl ALLOC failed (size={size}, errno={heap_close_errno})"
            ));
        }
        let dmabuf_fd: RawFd = data.fd as i32;

        // 3. mmap the DMA-BUF for CPU access.
        let host_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                dmabuf_fd,
                0,
            )
        };
        if host_ptr == libc::MAP_FAILED {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            unsafe { libc::close(dmabuf_fd) };
            return Err(anyhow!(
                "alloc_dmabuf_with_swap_context: mmap failed (size={size}, errno={errno})"
            ));
        }

        // 4. clCreateBufferWithProperties on swap_context first.
        let props: [u64; 3] = [CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, dmabuf_fd as u64, 0];
        unsafe extern "C" {
            fn clCreateBufferWithProperties(
                context: *mut c_void,
                properties: *const u64,
                flags: u64,
                size: usize,
                host_ptr: *mut c_void,
                errcode_ret: *mut i32,
            ) -> *mut c_void;
        }
        let swap_ctx_ptr = <&Context as ClContextPtr>::as_ptr(&swap_ctx);
        let mut errcode: i32 = 0;
        // swap_mem is mapped with CL_MAP_WRITE in `fill_dmabuf_via_swap_queue`,
        // so it must be allocated as READ_WRITE. main_mem stays READ_ONLY since
        // forward kernels only read.
        let raw_swap = unsafe {
            clCreateBufferWithProperties(
                swap_ctx_ptr as *mut _,
                props.as_ptr(),
                CL_MEM_READ_WRITE,
                size,
                ptr::null_mut(),
                &mut errcode,
            )
        };
        if errcode != 0 || raw_swap.is_null() {
            unsafe {
                libc::munmap(host_ptr, size);
                libc::close(dmabuf_fd);
            }
            return Err(anyhow!(
                "alloc_dmabuf_with_swap_context: clCreateBufferWithProperties (swap_ctx) failed (errcode={errcode})"
            ));
        }
        // SAFETY: clCreateBufferWithProperties returned a valid cl_mem owning
        // one reference. ocl::core::Mem::from_raw_create_ptr takes ownership.
        let swap_mem = unsafe { ocl::core::Mem::from_raw_create_ptr(raw_swap as *mut _) };

        // 5. clCreateBufferWithProperties on main_context with the same FD.
        // Driver dups the FD internally, so reuse is safe.
        let main_ctx_ptr = <&Context as ClContextPtr>::as_ptr(&&self.context);
        let mut errcode2: i32 = 0;
        let raw_main = unsafe {
            clCreateBufferWithProperties(
                main_ctx_ptr as *mut _,
                props.as_ptr(),
                CL_MEM_READ_ONLY,
                size,
                ptr::null_mut(),
                &mut errcode2,
            )
        };
        if errcode2 != 0 || raw_main.is_null() {
            // swap_mem already wraps raw_swap; let it Drop to release.
            drop(swap_mem);
            unsafe {
                libc::munmap(host_ptr, size);
                libc::close(dmabuf_fd);
            }
            return Err(anyhow!(
                "alloc_dmabuf_with_swap_context: clCreateBufferWithProperties (main_ctx) failed (errcode={errcode2})"
            ));
        }
        // SAFETY: same justification as `swap_mem` above.
        let main_mem = unsafe { ocl::core::Mem::from_raw_create_ptr(raw_main as *mut _) };

        Ok((swap_mem, main_mem, dmabuf_fd, host_ptr))
    }

    /// Fill a DMA-BUF-backed cl_mem via the secondary swap queue's
    /// `Map → memcpy → Unmap` cycle. All OpenCL calls are confined to the
    /// swap `cl_context` / queue, so the main forward queue is untouched —
    /// the multi-context isolation hypothesis under test.
    ///
    /// Optional cache-flush via `DMA_BUF_IOCTL_SYNC` when
    /// `LLMRS_DMABUF_SYNC=1` (default OFF — matches single-context DMA-BUF
    /// path so accuracy comparisons are apples-to-apples).
    ///
    /// # Safety
    /// - `src` must be valid (readable) for `size` bytes.
    /// - `swap_mem` must be a DMA-BUF-backed cl_mem from
    ///   `alloc_dmabuf_with_swap_context` (i.e., bound to the swap queue's
    ///   context).
    /// - `size` must be `<=` the buffer's allocated capacity.
    pub unsafe fn fill_dmabuf_via_swap_queue(
        &self,
        swap_mem: &ocl::core::Mem,
        host_ptr: *mut std::ffi::c_void,
        src: *const u8,
        size: usize,
    ) -> Result<()> {
        use ocl::ffi;
        const CL_TRUE: ffi::cl_bool = 1;
        const CL_MAP_WRITE: ffi::cl_map_flags = 1 << 1;

        let queue: &Queue = self.swap_queue_or_init().ok_or_else(|| {
            anyhow!(
                "fill_dmabuf_via_swap_queue: swap queue unavailable \
                 (env-gate LLMRS_OPENCL_SWAP_CONTEXT off, or queue build failed)"
            )
        })?;
        // CRITICAL: ocl 0.19 has `unsafe impl ClContextPtr for &Queue`,
        // so `q_ref: &Queue` + `q_ref.as_ptr()` resolves to ClContextPtr's
        // `as_ptr() -> cl_context`, NOT the cl_command_queue handle. Use the
        // explicit Deref-target type to disambiguate.
        let q_ref: &ocl::core::CommandQueue = queue;
        let q_ptr: ffi::cl_command_queue = q_ref.as_ptr();

        let mut errcode: ffi::cl_int = 0;
        // SAFETY (forwarded): caller asserts `swap_mem` is a DMA-BUF-backed
        // buffer in the swap queue's context and `size` is within capacity.
        let mapped_ptr = unsafe {
            ffi::clEnqueueMapBuffer(
                q_ptr,
                swap_mem.as_ptr(),
                CL_TRUE,
                CL_MAP_WRITE,
                0,
                size,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut errcode,
            )
        };
        if errcode != 0 || mapped_ptr.is_null() {
            return Err(anyhow!(
                "fill_dmabuf_via_swap_queue: clEnqueueMapBuffer failed (errcode={errcode}, ptr_null={})",
                mapped_ptr.is_null()
            ));
        }

        // SAFETY (forwarded): caller asserts `src` is valid for `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(src, mapped_ptr as *mut u8, size) };

        // SAFETY: `mapped_ptr` was returned by the map call above and is
        // still valid until the unmap completes.
        let rc = unsafe {
            ffi::clEnqueueUnmapMemObject(
                q_ptr,
                swap_mem.as_ptr(),
                mapped_ptr,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "fill_dmabuf_via_swap_queue: clEnqueueUnmapMemObject failed (rc={rc})"
            ));
        }

        // Note: optional `DMA_BUF_IOCTL_SYNC` is performed by the pool-level
        // caller via `ioctl_dmabuf_sync_write_if_enabled` because it owns
        // the FD. We intentionally don't propagate the FD into this fill
        // method to keep its signature compatible with non-DMA-BUF
        // callers that may reuse it later.
        let _ = host_ptr;

        // `LLMRS_HOST_PTR_SKIP_FINISH=1` matches `fill_host_ptr_buffer`'s
        // env-gate semantics. Default ON (we issue `clFinish`) so the swap
        // is observably complete before the caller drops the guard.
        if std::env::var("LLMRS_HOST_PTR_SKIP_FINISH").is_err() {
            // Use ocl::core::finish() with the raw queue ptr to avoid
            // method-resolution ambiguity between Queue::finish and
            // CommandQueueCore::finish via Deref.
            ocl::core::finish(q_ref)?;
        }
        Ok(())
    }

    /// Issue a `DMA_BUF_IOCTL_SYNC START|END WRITE` on the given FD when
    /// `LLMRS_DMABUF_SYNC=1`. Otherwise no-op. Used by the pool fill path
    /// to force CPU-cache flush after `memcpy`/`Map+Unmap`.
    pub fn ioctl_dmabuf_sync_write_if_enabled(&self, dmabuf_fd: std::os::unix::io::RawFd) {
        if std::env::var("LLMRS_DMABUF_SYNC").is_err() {
            return;
        }
        #[repr(C)]
        struct DmaBufSync {
            flags: u64,
        }
        const DMA_BUF_SYNC_WRITE: u64 = 2;
        const DMA_BUF_SYNC_START: u64 = 0;
        const DMA_BUF_SYNC_END: u64 = 1 << 2;
        let ioctl_sync: u64 = (1u64 << 30)
            | ((std::mem::size_of::<DmaBufSync>() as u64) << 16)
            | ((b'b' as u64) << 8);
        unsafe extern "C" {
            fn ioctl(fd: libc::c_int, request: libc::c_ulong, ...) -> libc::c_int;
        }
        let mut sync_start = DmaBufSync {
            flags: DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE,
        };
        let mut sync_end = DmaBufSync {
            flags: DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE,
        };
        unsafe {
            ioctl(dmabuf_fd, ioctl_sync as libc::c_ulong, &mut sync_start);
            ioctl(dmabuf_fd, ioctl_sync as libc::c_ulong, &mut sync_end);
        }
    }

    pub fn alloc_host_ptr_buffer_empty(&self, size: usize) -> Result<ocl::core::Mem> {
        const CL_MEM_READ_ONLY: u64 = 1 << 2;
        const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
        // Qualcomm cl_qcom_ext_host_ptr extension constants (from cl_ext.h).
        const CL_MEM_EXT_HOST_PTR_QCOM: u64 = 1 << 29;
        const CL_MEM_HOST_IOCOHERENT_QCOM: u32 = 0x40A9;
        const CL_MEM_HOST_WRITEBACK_QCOM: u32 = 0x40A5;

        // Path A' empirical probe: when LLMRS_OPENCL_IOCOHERENT=1, try
        // allocating with CL_MEM_EXT_HOST_PTR_QCOM + iocoherent cache policy.
        // Driver may reject if allocation_type=0 is not accepted; in that
        // case we fall back to plain ALLOC_HOST_PTR.
        if std::env::var("LLMRS_OPENCL_IOCOHERENT").is_ok() {
            #[repr(C)]
            struct ClMemExtHostPtr {
                allocation_type: u32,
                host_cache_policy: u32,
            }
            let cache_policy = if std::env::var("LLMRS_OPENCL_WRITEBACK").is_ok() {
                CL_MEM_HOST_WRITEBACK_QCOM
            } else {
                CL_MEM_HOST_IOCOHERENT_QCOM
            };
            let ext = ClMemExtHostPtr {
                allocation_type: 0, // probe: driver may accept 0 as "default"
                host_cache_policy: cache_policy,
            };
            let flags = (CL_MEM_READ_ONLY | CL_MEM_EXT_HOST_PTR_QCOM) as ocl::ffi::cl_bitfield;
            let ctx_ptr = <&Context as ClContextPtr>::as_ptr(&&self.context);
            let mut errcode: ocl::ffi::cl_int = 0;
            let raw = unsafe {
                ocl::ffi::clCreateBuffer(
                    ctx_ptr,
                    flags,
                    size,
                    &ext as *const _ as *mut std::ffi::c_void,
                    &mut errcode,
                )
            };
            if errcode == 0 && !raw.is_null() {
                eprintln!(
                    "[IOCOHERENT] cl_qcom_ext_host_ptr accepted: cache_policy=0x{:X}, size={}",
                    cache_policy, size
                );
                let mem = unsafe { ocl::core::Mem::from_raw_create_ptr(raw) };
                return Ok(mem);
            }
            eprintln!(
                "[IOCOHERENT] cl_qcom_ext_host_ptr rejected (errcode={}, ptr_null={}); \
                 falling back to plain ALLOC_HOST_PTR",
                errcode,
                raw.is_null()
            );
            // Fall through to standard path below.
        }

        // Phase 4 (2026-05-09): add CL_MEM_HOST_WRITE_ONLY hint. Adreno 830
        // measurement showed -35% wall-clock for 25MB clEnqueueWriteBuffer
        // (1.72ms → 1.11ms). Driver omits GPU L2 staging when host writes only.
        // Compatible with READ_ONLY (GPU read-only): host writes via clEnqueueWriteBuffer
        // remain allowed (not host-side read-only).
        const CL_MEM_HOST_WRITE_ONLY: u64 = 1 << 7;
        let flags = (CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY)
            as ocl::ffi::cl_bitfield;
        let ctx_ptr = <&Context as ClContextPtr>::as_ptr(&&self.context);
        let mut errcode: ocl::ffi::cl_int = 0;
        let raw = unsafe {
            ocl::ffi::clCreateBuffer(ctx_ptr, flags, size, std::ptr::null_mut(), &mut errcode)
        };
        if errcode != 0 || raw.is_null() {
            return Err(anyhow!(
                "alloc_host_ptr_buffer_empty: clCreateBuffer failed (size={size}, errcode={errcode})"
            ));
        }
        // SAFETY: clCreateBuffer succeeded with a non-null handle; we own
        // the freshly created refcount. `Mem::from_raw_create_ptr` takes
        // ownership; Drop will call `clReleaseMemObject`.
        let mem = unsafe { ocl::core::Mem::from_raw_create_ptr(raw) };
        Ok(mem)
    }

    /// Fill a previously-allocated `CL_MEM_ALLOC_HOST_PTR` cl_mem with
    /// `size` bytes from `src` via the
    /// `clEnqueueMapBuffer(MAP_WRITE) → memcpy → clEnqueueUnmapMemObject`
    /// cycle. The driver flushes CPU caches and makes the bytes GPU-
    /// visible on unmap. Blocks until the unmap completes.
    ///
    /// # Safety
    /// - `src` must be valid (readable) for `size` bytes for the duration
    ///   of the call.
    /// - `mem` must be a host-accessible buffer (created with
    ///   `CL_MEM_ALLOC_HOST_PTR` or `CL_MEM_USE_HOST_PTR`); calling on a
    ///   device-only buffer is undefined behaviour.
    /// - `size` must be `<=` the buffer's allocated capacity.
    pub unsafe fn fill_host_ptr_buffer(
        &self,
        mem: &ocl::core::Mem,
        src: *const u8,
        size: usize,
    ) -> Result<()> {
        use ocl::ffi;
        const CL_TRUE: ffi::cl_bool = 1;
        const CL_MAP_WRITE: ffi::cl_map_flags = 1 << 1;

        let q_ref: &ocl::core::CommandQueue = &self.queue;
        let q_ptr: ffi::cl_command_queue = q_ref.as_ptr();

        let mut errcode: ffi::cl_int = 0;
        // SAFETY (forwarded): caller asserts `mem` is `ALLOC_HOST_PTR`-allocated
        // and `size` is within its capacity.
        let mapped_ptr = unsafe {
            ffi::clEnqueueMapBuffer(
                q_ptr,
                mem.as_ptr(),
                CL_TRUE,
                CL_MAP_WRITE,
                0,
                size,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut errcode,
            )
        };
        if errcode != 0 || mapped_ptr.is_null() {
            return Err(anyhow!(
                "fill_host_ptr_buffer: clEnqueueMapBuffer failed (errcode={errcode}, ptr_null={})",
                mapped_ptr.is_null()
            ));
        }

        // SAFETY (forwarded): caller asserts `src` is valid for `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(src, mapped_ptr as *mut u8, size) };

        // SAFETY: `mapped_ptr` was returned by the map call above and is
        // still valid until the unmap completes.
        let rc = unsafe {
            ffi::clEnqueueUnmapMemObject(
                q_ptr,
                mem.as_ptr(),
                mapped_ptr,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "fill_host_ptr_buffer: clEnqueueUnmapMemObject failed (rc={rc})"
            ));
        }
        // Wait for the unmap to land so subsequent kernel reads see the
        // bytes. Stage 1b microbench used `queue.finish()` here too.
        //
        // LISWAP-3 v2 (env-gated): skip the explicit `queue.finish()` and
        // rely on OpenCL's same-queue dependency tracking — the next
        // kernel reading this cl_mem on `self.queue` will be automatically
        // ordered after the unmap. Removes `clFinish` × N (per-layer) full
        // barrier from the swap hot path. Set
        // `LLMRS_HOST_PTR_SKIP_FINISH=1` to enable.
        if std::env::var("LLMRS_HOST_PTR_SKIP_FINISH").is_err() {
            self.queue.finish()?;
        }
        Ok(())
    }

    /// Lazily build (or return) the LISWAP-3 host-ptr pool. Returns `None`
    /// when:
    /// - the env-gate `LLMRS_OPENCL_HOST_PTR_POOL` is disabled (default),
    /// - pool construction failed (e.g. memory pressure on first alloc).
    ///
    /// Caller (SwapExecutor) is contracted to fall back to the staging
    /// path on `None`. Idempotent: `OnceLock` caches the first attempt.
    ///
    /// `config` is consulted only on the **first** call; subsequent calls
    /// return the previously-built pool regardless of `config`. This
    /// matches the lifecycle expectation of a single CLI-configured pool
    /// per process.
    pub fn host_ptr_pool_or_init(
        &self,
        config: host_ptr_pool::HostPtrPoolConfig,
    ) -> Option<Arc<host_ptr_pool::HostPtrPool>> {
        if !host_ptr_pool::host_ptr_pool_env_enabled() {
            return None;
        }
        let cell = self.host_ptr_pool.get_or_init(|| {
            match host_ptr_pool::HostPtrPool::new(self, config) {
                Ok(pool) => Some(Arc::new(pool)),
                Err(e) => {
                    log::warn!(
                        "LLMRS_OPENCL_HOST_PTR_POOL: pool creation failed ({e}); \
                         falling back to staging swap path"
                    );
                    None
                }
            }
        });
        cell.clone()
    }

    /// Read-only accessor for an already-initialised pool. Returns `None`
    /// when `host_ptr_pool_or_init` has not yet been called or initialisation
    /// failed.
    pub fn host_ptr_pool_get(&self) -> Option<Arc<host_ptr_pool::HostPtrPool>> {
        self.host_ptr_pool.get().and_then(|opt| opt.clone())
    }
}

impl Backend for OpenCLBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // §13.8-L S-L-1: Profile hook trait methods — inherent impl 으로 위임.
    fn profile_events_enabled(&self) -> bool {
        self.profile_events_enabled
    }
    fn set_op_label(&self, label: &'static str) {
        Self::set_op_label(self, label)
    }
    fn clear_op_label(&self) {
        Self::clear_op_label(self)
    }

    // §13.8-L S-L-2: GpuScoreAccess trait method — inherent impl 으로 위임.
    // `&self -> &mut dyn` 패턴은 inherent `gpu_score_acc_mut` 의
    // UnsafeCell-backed 구현을 trait 으로 노출 (single-threaded 가정).
    fn gpu_score_acc(&self) -> Option<&dyn crate::backend::GpuScoreAccess> {
        Self::gpu_score_acc(self).map(|s| s as &dyn crate::backend::GpuScoreAccess)
    }
    fn gpu_score_acc_mut(&self) -> Option<&mut dyn crate::backend::GpuScoreAccess> {
        Self::gpu_score_acc_mut(self).map(|s| s as &mut dyn crate::backend::GpuScoreAccess)
    }

    // FORMAT Phase 2 Stage E: lend the live `cl_command_queue` so a
    // borrowed-context backend-cap plugin enqueues on the SAME in-order queue as
    // the engine (preserving the serialization the quant-window flush/score-readback path
    // relies on). The built-in quant-window impl ignores `args.cl_queue` and uses
    // `&self.queue`, so packing this is a no-op for the in-engine path.
    fn cl_command_queue_ptr(&self) -> *mut std::ffi::c_void {
        // `&Queue::as_ptr()` resolves via `ClContextPtr for &Queue` to a
        // `cl_context`; the explicit `&CommandQueue` target type yields the
        // `cl_command_queue` handle (cf. `fill_dmabuf_via_swap_queue`).
        let q_ref: &ocl::core::CommandQueue = &self.queue;
        q_ref.as_ptr()
    }

    // FORMAT Phase 2 Stage E: expose the device's nosub property on the Backend
    // trait so the quant-window FORMAT's native-attention gate reads it from the backend
    // (byte-identical) instead of the QuantAttn cap, which moves to a plugin.
    fn is_nosub_device(&self) -> bool {
        Self::is_nosub(self)
    }

    fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
        let buf =
            get_cl_mem(t.buffer().as_ref()).map_err(|_| anyhow::anyhow!("Not OpenCL buffer"))?;
        unsafe {
            ocl::core::enqueue_read_buffer(
                &self.queue,
                buf,
                true,
                0,
                dst,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn enqueue_read_buffer_async(
        &self,
        t: &Tensor,
        dst: &mut [u8],
    ) -> Result<crate::backend::GpuEvent> {
        let buf =
            get_cl_mem(t.buffer().as_ref()).map_err(|_| anyhow::anyhow!("Not OpenCL buffer"))?;
        let mut event: ocl::core::Event = ocl::core::Event::null();
        // SAFETY: non-blocking read with an explicit event output. The caller
        // is contracted (per trait docs) to keep `dst` alive until `wait_event`
        // returns for this event — matching OpenCL's device-timeline write.
        unsafe {
            ocl::core::enqueue_read_buffer(
                &self.queue,
                buf,
                false,
                0,
                dst,
                None::<&ocl::core::Event>,
                Some(&mut event),
            )?;
        }
        Ok(crate::backend::GpuEvent {
            inner_cl: Some(event),
            #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
            inner_cu: None,
        })
    }

    fn wait_event(&self, evt: &crate::backend::GpuEvent) -> Result<()> {
        if let Some(e) = evt.inner_cl.as_ref() {
            ocl::core::wait_for_event(e)?;
        }
        Ok(())
    }

    /// LISWAP-2 prototype (plan: chasing-hopper, Phase 1). Async H2D
    /// upload of a weight tensor onto the secondary `transfer_queue`.
    /// Returns the freshly allocated destination tensor plus an event
    /// that completes once the write is GPU-visible.
    ///
    /// Falls back to the synchronous default impl when the transfer
    /// queue is disabled (env-gate or creation failure) or when the
    /// destination has no `cl_mem` handle (host-only fallback).
    fn enqueue_write_async(&self, src: &Tensor) -> Result<(Tensor, crate::backend::GpuEvent)> {
        let Some(transfer_queue) = self.transfer_queue_or_init() else {
            // Default: sync via copy_weight_from + dummy event.
            let dst = self.copy_weight_from(src)?;
            return Ok((dst, crate::backend::GpuEvent::default()));
        };

        // Allocate the destination buffer through the regular memory
        // factory bound to the main queue. `cl_mem` handles are
        // context-scoped, not queue-scoped, so the secondary queue can
        // write to this buffer without any cross-queue migration cost.
        let size = src.size();
        let memory = crate::backend::opencl::memory::OpenCLMemory::new(
            self.context.clone(),
            self.queue.clone(),
            self.use_zero_copy,
        );
        let buffer = memory.alloc(size, src.dtype())?;
        let category = match src.dtype() {
            DType::Q4_0 => "weight_q4_aos_async",
            DType::F16 => "weight_f16_async",
            DType::F32 => "weight_f32_async",
            _ => "weight_other_async",
        };
        self.record_cl_mem_alloc(category, size);

        let new_tensor = Tensor::new(src.shape().clone(), buffer.clone(), src.backend().clone());

        // We require both a source host pointer (to feed the
        // non-blocking write) and a destination cl_mem handle. If
        // either is missing, defer to the sync path so we keep the
        // same correctness envelope as `copy_from`.
        let src_ptr = src.as_ptr();
        if src_ptr.is_null() {
            return Err(anyhow!(
                "enqueue_write_async: source has a null host pointer (size={size}, dtype={:?})",
                src.dtype()
            ));
        }
        let dst_mem = match get_cl_mem(buffer.as_ref()) {
            Ok(m) => m,
            Err(_) => {
                // Buffer doesn't expose cl_mem (mapped host-only). Fall
                // back to sync copy.
                let dst = self.copy_weight_from(src)?;
                return Ok((dst, crate::backend::GpuEvent::default()));
            }
        };
        let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, size) };

        let mut event: ocl::core::Event = ocl::core::Event::null();
        // SAFETY: non-blocking write with an explicit event output. The
        // caller (async swap dispatcher's worker thread) waits on this
        // event before publishing the new weights, so the destination
        // cl_mem outlives the in-flight write — and the source bytes
        // outlive it too because they come from the AUF/Mmap-backed
        // weight tensor whose lifetime spans the whole swap operation.
        unsafe {
            ocl::core::enqueue_write_buffer(
                transfer_queue,
                dst_mem,
                false,
                0,
                src_slice,
                None::<&ocl::core::Event>,
                Some(&mut event),
            )?;
        }
        // Submit the command immediately so the device starts the DMA
        // without waiting for the next clFlush. This matches OpenCL's
        // recommended pattern for cross-queue async work.
        ocl::core::flush(transfer_queue)?;

        Ok((
            new_tensor,
            crate::backend::GpuEvent {
                inner_cl: Some(event),
                #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
                inner_cu: None,
            },
        ))
    }

    /// Wait for an event recorded by `enqueue_write_async` (or
    /// `enqueue_read_buffer_async`) without draining the compute queue.
    /// `clWaitForEvents` blocks the *host* thread but not the device,
    /// which is exactly what the swap dispatcher's worker thread needs
    /// before flipping ArcSwap.
    fn wait_event_blocking(&self, evt: &crate::backend::GpuEvent) -> Result<()> {
        if let Some(e) = evt.inner_cl.as_ref() {
            ocl::core::wait_for_event(e)?;
            return Ok(());
        }
        // No CL event attached — could be a CUDA event on a multi-feature
        // build, or a dummy fallback event. In neither case does the
        // OpenCL backend have anything device-side to wait on, so we
        // fall through to the default sync barrier for safety.
        self.synchronize()
    }

    /// LISWAP-2 prototype: only true when the env-gate is on (and queue
    /// creation has not been pre-empted by a permanent failure). The
    /// async swap dispatcher uses this to decide between the dispatcher
    /// path and the regular sync `copy_weight_from`.
    fn supports_async_transfer(&self) -> bool {
        Self::transfer_queue_env_enabled()
    }

    fn name(&self) -> &str {
        "OpenCL"
    }
    fn device(&self) -> &str {
        "GPU"
    }

    fn copy_from(&self, src: &Tensor) -> Result<Tensor> {
        let size = src.size();
        // On UMA devices (Adreno): CL_MEM_ALLOC_HOST_PTR for zero-copy shared memory.
        // On discrete GPUs (NVIDIA): device-only buffers for correct behavior.
        let memory = crate::backend::opencl::memory::OpenCLMemory::new(
            self.context.clone(),
            self.queue.clone(),
            self.use_zero_copy,
        );
        let buffer = memory.alloc(size, src.dtype())?;
        // WSWAP-5-TBT-DIAG: classify the allocation by dtype. Q4_0 weights
        // landing through `copy_from` (or its `copy_weight_from` default
        // fall-through) come from the AUF placeholder path in
        // `materialise_auf_soa_weight` — every byte of these is "wasted"
        // because the GEMV kernel only reads the SOA registry q_buf/d_buf.
        // Counting this category lets the dump expose AUF placeholder
        // pressure separately from the SOA buffers proper.
        let category = match src.dtype() {
            DType::Q4_0 => "weight_q4_aos_copy",
            DType::F16 => "weight_f16_copy",
            DType::F32 => "weight_f32_copy",
            _ => "weight_other_copy",
        };
        self.record_cl_mem_alloc(category, size);

        // Share the source tensor's backend (Arc clone = cheap reference count)
        // instead of creating a new backend + cloning 15 kernel objects.
        // The Mutex<KernelCache> already provides thread safety.
        let new_tensor = Tensor::new(src.shape().clone(), buffer.clone(), src.backend().clone());

        // Device-to-Device Copy - try using get_cl_mem for both buffer types
        if let (Ok(src_mem), Ok(dst_mem)) = (
            get_cl_mem(src.buffer().as_ref()),
            get_cl_mem(buffer.as_ref()),
        ) {
            unsafe {
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue,
                    src_mem,
                    dst_mem,
                    0,
                    0,
                    size,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            return Ok(new_tensor);
        }

        // Host-to-Device Copy - source is CPU, destination is GPU
        let src_ptr = src.as_ptr();
        if !src_ptr.is_null() {
            let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, size) };
            if let Ok(dst_mem) = get_cl_mem(buffer.as_ref()) {
                unsafe {
                    ocl::core::enqueue_write_buffer(
                        &self.queue,
                        dst_mem,
                        true,
                        0,
                        src_slice,
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
            } else {
                return Err(anyhow!(
                    "Failed to get cl_mem handle for destination buffer"
                ));
            }
        }

        Ok(new_tensor)
    }

    fn copy_into(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        let size = src.size();
        assert_eq!(size, dst.size(), "copy_into: size mismatch");

        if let (Ok(src_mem), Ok(dst_mem)) = (
            get_cl_mem(src.buffer().as_ref()),
            get_cl_mem(dst.buffer().as_ref()),
        ) {
            // GPU→GPU: enqueue_copy_buffer (in-order queue ensures correctness)
            unsafe {
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue,
                    src_mem,
                    dst_mem,
                    0,
                    0,
                    size,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        } else {
            // Fallback: CPU memcpy
            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            if !src_ptr.is_null() && !dst_ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
                }
            }
        }
        Ok(())
    }

    fn write_buffer(&self, t: &mut Tensor, src: &[u8]) -> Result<()> {
        assert_eq!(
            src.len(),
            t.size(),
            "write_buffer: size mismatch ({} vs {})",
            src.len(),
            t.size()
        );
        if let Ok(dst_mem) = get_cl_mem(t.buffer().as_ref()) {
            // GPU buffer: blocking write to ensure src data is consumed before return.
            // For small transfers (e.g., 4-byte token id), overhead is negligible.
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &self.queue,
                    dst_mem,
                    true,
                    0,
                    src,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        } else {
            // Mapped / host-ptr buffer: direct memcpy
            let dst_ptr = t.as_mut_ptr();
            if dst_ptr.is_null() {
                anyhow::bail!("write_buffer: null pointer in destination tensor");
            }
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
            }
        }
        Ok(())
    }

    fn write_buffer_range(&self, t: &mut Tensor, src: &[u8], dst_offset: usize) -> Result<()> {
        let end = dst_offset
            .checked_add(src.len())
            .ok_or_else(|| anyhow::anyhow!("write_buffer_range: offset+len overflow"))?;
        if end > t.size() {
            anyhow::bail!(
                "write_buffer_range: out of bounds ({} + {} > {})",
                dst_offset,
                src.len(),
                t.size()
            );
        }
        if let Ok(dst_mem) = get_cl_mem(t.buffer().as_ref()) {
            // Partial blocking write starting at `dst_offset` bytes into the buffer.
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &self.queue,
                    dst_mem,
                    true,
                    dst_offset,
                    src,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        } else {
            // Mapped / host-ptr buffer: direct offset memcpy
            let dst_ptr = t.as_mut_ptr();
            if dst_ptr.is_null() {
                anyhow::bail!("write_buffer_range: null pointer in destination tensor");
            }
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr.add(dst_offset), src.len());
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        ocl::core::finish(&self.queue)?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        ocl::core::flush(&self.queue)?;
        Ok(())
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn is_discrete_gpu(&self) -> bool {
        !self.use_zero_copy
    }

    fn max_single_alloc(&self) -> usize {
        self.max_mem_alloc_size
    }

    /// ENG-ALG-221 / INV-130: drop every Q4_0 noshuffle SOA descriptor so the
    /// next `FullKernelPlan` rebuild (triggered by the `SwapExecutor`'s
    /// `ratio_generation` bump) re-registers against the new `cl_mem`
    /// addresses. See `clear_noshuffle_soa_registry` doc for the underlying
    /// mechanics. No-op on backends that never populated the registry.
    fn invalidate_noshuffle_soa_registry(&self) {
        self.clear_noshuffle_soa_registry();
    }

    /// ENG-ALG-222 / INV-131: register a noshuffle SOA descriptor for the
    /// given Q4_0 weight tensor. Adreno-only safety net — runs the same
    /// `convert_q4_0_to_noshuffle()` pipeline as `prepare_noshuffle_buffers()`
    /// but without swapping the tensor's backing buffer (the original AOS
    /// `cl_mem` stays alive so the registry key matches both this lookup
    /// path and any other consumer that holds the swap-installed tensor).
    ///
    /// Behaviour:
    /// - Non-Q4_0 tensor: NoOp.
    /// - Tensor without a GPU `cl_mem` (CPU buffer): NoOp.
    /// - Q4_0 noshuffle conversion kernel unavailable (driver lacks the SOA
    ///   path): NoOp — falls back to the standard Q4_0 GEMV path silently.
    /// - Otherwise: convert to SOA, register against the AOS `cl_mem` key.
    ///
    /// Called from `SwapExecutor::execute_on_slots` after the per-batch
    /// `clear_noshuffle_soa_registry()` and before the `ratio_generation`
    /// bump.
    fn ensure_noshuffle_soa_registered(&self, tensor: &Tensor) -> Result<()> {
        // Q4_0 only — other dtypes have no noshuffle SOA path.
        if tensor.dtype() != DType::Q4_0 {
            return Ok(());
        }

        // Need a GPU cl_mem to convert. Tensors that landed on CPU are
        // out of scope (this can occur in tests or when a swap target is
        // routed through the CPU backend).
        let Ok(cl_mem) = get_cl_mem(tensor.buffer().as_ref()) else {
            return Ok(());
        };

        // The conversion kernel is gated on the cvt_noshuffle program
        // compiling cleanly. If it is missing (driver-side build failure),
        // fall back to the AOS GEMV path silently — `convert_q4_0_to_noshuffle`
        // will surface the same error and we degrade rather than abort the
        // entire swap.
        if unsafe { (*self.kernels.get()).kernel_cvt_q4_0_noshuffle.is_none() } {
            return Ok(());
        }

        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            // Non-matmul Q4_0 tensors should not occur in the swap path
            // (norms are F32, biases are F32). Skip defensively.
            return Ok(());
        }
        let (ne01, ne00) = (dims[0], dims[1]);
        // QK4_0 = 32. Number of Q4_0 blocks across the entire tensor.
        if ne00 % 32 != 0 {
            return Ok(());
        }
        let num_blocks = ne01 * ne00 / 32;

        let (q_buf, d_buf, q_img) =
            self.convert_q4_0_to_noshuffle(cl_mem, num_blocks, ne00, ne01)?;

        // Key the registry by the AOS cl_mem address — the tensor that
        // landed in `LayerSlot` after the swap returns this address through
        // its `cl_mem()`, so `matmul_q4_0`'s `b_buf.as_ptr() as usize` will
        // hit. We do *not* swap the tensor for a `NoshuffleWeightBuffer`
        // here: that helper is paired with the model-load `prepare_noshuffle_
        // buffers()` flow and assumes mutable access to the tensor. Phase
        // 3.7a operates on already-installed `Arc<LayerWeights>` snapshots.
        let key = cl_mem.as_ptr() as usize;
        self.register_noshuffle_soa(
            key,
            NoshuffleSoaEntry {
                q_buf,
                d_buf,
                q_img,
                ne00,
                ne01,
            },
        );
        Ok(())
    }

    /// AUF SOA bypass — Phase 3.7b / Phase 5 Sprint C
    /// (WSWAP-5-AUF-PLACEHOLDER-DROP).
    ///
    /// Allocate a Q4_0 weight tensor whose backing buffer is a
    /// `NoshuffleWeightBuffer` directly populated from the AUF
    /// `WEIGHTS_ADRENO_SOA` payload — no AOS placeholder cl_mem.
    ///
    /// Behaviour matches the `ensure_noshuffle_soa_registered` short-circuits
    /// (non-Q4_0 / CPU buffer / missing cvt program). When preconditions are
    /// met:
    ///   1. Allocate `cl_mem` for `q_bytes` (`expected_q` bytes) and
    ///      `d_bytes` (`expected_d` bytes); blocking-write the AUF payload.
    ///   2. Build an `image1d_buffer_t` view over `q_buf` (silently None on
    ///      drivers that reject the descriptor).
    ///   3. Register a noshuffle SOA entry keyed on the `d_buf` cl_mem
    ///      address (mirrors `prepare_noshuffle_buffers(swap_to_placeholder=true)`
    ///      so the `matmul_q4_0` lookup hits via `Tensor::buffer().cl_mem()`).
    ///   4. Return a `Tensor` whose buffer is a `NoshuffleWeightBuffer`
    ///      owning the SOA handles, with the logical AOS shape and `Q4_0`
    ///      dtype.
    ///
    /// The previous version of this entry-point allocated a placeholder AOS
    /// cl_mem on top of the SOA registration purely to hand the swap pipeline
    /// a valid Tensor handle. Sprint C removes that 112-cl_mem / ~547 MiB
    /// drag (Llama 3.2 1B ratio=1.0 mixed measured by `ClMemDiagBucket`).
    fn alloc_pre_converted_soa_tensor(
        &self,
        shape: crate::shape::Shape,
        q_bytes: &[u8],
        d_bytes: &[u8],
        ne00: usize,
        ne01: usize,
    ) -> Result<Option<Tensor>> {
        // Skip silently when the noshuffle SOA program failed to build —
        // matches `ensure_noshuffle_soa_registered`'s degradation contract.
        if unsafe { (*self.kernels.get()).kernel_cvt_q4_0_noshuffle.is_none() } {
            return Ok(None);
        }

        // Validate payload sizes against the expected SOA layout.
        if !ne00.is_multiple_of(32) {
            return Ok(None);
        }
        let num_blocks = ne01 * ne00 / 32;
        let expected_q = num_blocks * 16;
        let expected_d = num_blocks * 2;
        if q_bytes.len() != expected_q || d_bytes.len() != expected_d {
            anyhow::bail!(
                "AUF SOA payload size mismatch: ne00={}, ne01={}, expected q={} d={}, got q={} d={}",
                ne00,
                ne01,
                expected_q,
                expected_d,
                q_bytes.len(),
                d_bytes.len()
            );
        }

        let (q_buf, d_buf, q_img) =
            self.alloc_and_upload_soa_buffers(q_bytes, d_bytes, ne00, ne01, num_blocks)?;

        // Register against the d_buf address — same key choice as
        // `prepare_noshuffle_buffers(swap_to_placeholder=true)`.
        let key = d_buf.as_ptr() as usize;
        self.register_noshuffle_soa(
            key,
            NoshuffleSoaEntry {
                q_buf: q_buf.clone(),
                d_buf: d_buf.clone(),
                q_img: q_img.as_ref().cloned(),
                ne00,
                ne01,
            },
        );

        // Logical AOS byte size. `NoshuffleWeightBuffer::size()` reports this
        // so consumers that inspect `Tensor::size()` see the weight footprint.
        let logical_bytes = num_blocks * 18;
        let placeholder = Arc::new(
            crate::memory::opencl::noshuffle::NoshuffleWeightBuffer::new(
                q_buf,
                d_buf,
                q_img,
                ne00,
                ne01,
                logical_bytes,
            ),
        ) as Arc<dyn Buffer>;

        // Backend Arc: clone the `OpenCLBackend` Arc that the swap path
        // already holds. The caller's `self` is `&dyn Backend`, but the
        // tensor needs an owned `Arc<dyn Backend>`. The swap executor passes
        // its own backend handle through `materialise_auf_soa_weight` which
        // is responsible for wrapping the `Arc` — here we cannot resurrect
        // it from `&self`. Returning a tensor without a backend Arc would
        // break later kernel dispatch, so we expect callers to substitute
        // the backend Arc before forwarding the tensor (handled in
        // `materialise_auf_soa_weight`).
        //
        // The convention adopted here: produce the tensor against a fresh
        // CPU `Backend` Arc as a transient placeholder for `Tensor::new`'s
        // `Arc<dyn Backend>` slot, knowing that the caller assigns the real
        // backend Arc immediately. This avoids exposing an `Arc<dyn Backend>`
        // accessor on `OpenCLBackend` while keeping the tensor self-contained.
        //
        // The forward path ignores `Tensor::backend()` for OpenCL dispatch —
        // it operates through `Backend::matmul_*` on the `LayerWeights`'
        // outer backend Arc. The cl_mem path does not consult tensor.backend.
        // placeholder Tensor용 CPU backend Arc; forward는 본 backend Arc 미참조.
        // LAYER-EXEMPT: cross_backend_bootstrap
        let cpu_be: Arc<dyn Backend> = crate::backend::cpu::cpu_singleton();
        Ok(Some(Tensor::new(shape, placeholder, cpu_be)))
    }

    /// Re-register a previously-built AUF SOA `NoshuffleWeightBuffer` after
    /// a `clear_noshuffle_soa_registry()` wipe (ENG-ALG-221 / INV-130).
    /// Falls through to `ensure_noshuffle_soa_registered` for tensors that
    /// are not SOA-backed.
    fn restore_pre_converted_soa_registration(&self, tensor: &Tensor) -> Result<()> {
        // Q4_0 only — non-Q4_0 noshuffle path is undefined.
        if tensor.dtype() != DType::Q4_0 {
            return Ok(());
        }

        if let Some(nb) = tensor
            .buffer()
            .as_any()
            .downcast_ref::<crate::memory::opencl::noshuffle::NoshuffleWeightBuffer>()
        {
            // SOA-backed: re-insert against the d_buf key (`cl_mem()` on
            // `NoshuffleWeightBuffer` returns d_buf, so the registry key
            // matches `b_buf.as_ptr() as usize` lookups in `matmul_q4_0`).
            let key = nb.d_buf().as_ptr() as usize;
            self.register_noshuffle_soa(
                key,
                NoshuffleSoaEntry {
                    q_buf: nb.q_buf().clone(),
                    d_buf: nb.d_buf().clone(),
                    q_img: nb.q_img().cloned(),
                    ne00: nb.ne00(),
                    ne01: nb.ne01(),
                },
            );
            return Ok(());
        }

        // GGUF AOS-backed (fallback path): re-run the standard SOA conversion.
        self.ensure_noshuffle_soa_registered(tensor)
    }

    /// LISWAP-6 — DMA-BUF alias `cl_mem` for swapped weights.
    ///
    /// Wraps `host_ptr + offset` in a `CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY`
    /// alias. The returned `RpcmemAliasBuffer` holds the supplied lifetime
    /// guards (`secondary_arc` + `layer_region`) so the underlying rpcmem
    /// allocation remains valid until every alias is dropped (Drop ordering:
    /// cl_mem → Arc<SecondaryMmap> → Arc<RpcmemLayerRegion> → rpcmem_free).
    ///
    /// Rejects `size == 0` (invalid alias); rejects null `host_ptr`.
    ///
    /// # Safety
    /// Caller must guarantee:
    /// - `host_ptr.add(offset)` is valid and points to at least `size` bytes
    ///   of correctly aligned, initialised memory.
    /// - The region remains live and unmoved for the entire lifetime of the
    ///   returned alias buffer. Lifetime is pinned by `secondary_arc` +
    ///   `layer_region`, both of which are moved into the returned
    ///   `RpcmemAliasBuffer` and dropped after the underlying `cl_mem`.
    /// - No mutation of the region while alias buffers exist (the secondary
    ///   mmap is read-only per `SecondaryMmap` contract).
    unsafe fn alloc_alias_weight_buffer(
        &self,
        host_ptr: *mut u8,
        offset: usize,
        size: usize,
        dtype: DType,
        secondary_arc: std::sync::Arc<dyn crate::memory::host::mmap::MmapKeepAlive>,
        layer_region: std::sync::Arc<dyn crate::memory::secondary::RpcmemRegionGuard>,
    ) -> Result<Option<std::sync::Arc<dyn crate::buffer::Buffer>>> {
        if size == 0 || host_ptr.is_null() {
            return Ok(None);
        }
        // SAFETY: `host_ptr + offset .. + size` lies within a region the
        // caller pins via `layer_region` (enforced by the fn-level safety
        // contract above). The `CL_MEM_USE_HOST_PTR` contract requires the
        // host memory to remain valid for the cl_mem's lifetime; the
        // lifetime guards installed in `RpcmemAliasBuffer` enforce this.
        let aliased_ptr = unsafe { host_ptr.add(offset) };
        let host_slice = unsafe { std::slice::from_raw_parts_mut(aliased_ptr, size) };
        let cl_buffer = unsafe {
            ocl::core::create_buffer(
                self.context.as_core(),
                ocl::core::MEM_READ_ONLY | ocl::core::MEM_USE_HOST_PTR,
                size,
                Some(host_slice),
            )
        }
        .map_err(|e| {
            anyhow::anyhow!(
                "alloc_alias_weight_buffer: clCreateBuffer USE_HOST_PTR failed (size={size}): {e}"
            )
        })?;

        let alias = crate::memory::rpcmem::opencl_alias::RpcmemAliasBuffer::new(
            cl_buffer,
            aliased_ptr,
            size,
            dtype,
            secondary_arc,
            layer_region,
        );
        Ok(Some(std::sync::Arc::new(alias)))
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
        is_head_major: bool,
    ) -> Result<bool> {
        self.flash_attention_prefill_gpu(
            q,
            k_cache,
            v_cache,
            out,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            head_dim,
            kv_capacity,
            batch_size,
            is_head_major,
        )
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        if b_dims[1] != k {
            return Err(anyhow!(
                "Dimension mismatch: A[{},{}] * B^T[{},{}]",
                m,
                k,
                n,
                b_dims[1]
            ));
        }

        // Use tiled GEMM for prefill (M > 8), GEMV for decode
        const GEMM_THRESHOLD: usize = 8;
        if m > GEMM_THRESHOLD {
            let kernels = unsafe { &*self.kernels.get() };
            if kernels.kernel_mul_mm_f32_f32.is_some() {
                return self.matmul_gemm_f32(a, b, out);
            }
        }

        let a_buf = get_cl_mem(a.buffer().as_ref()).map_err(|_| {
            anyhow!(
                "matmul (F32): A is not OpenCL buffer (kind={}, dtype={:?}, shape={:?})",
                buffer_kind_label(a.buffer().as_ref()),
                a.dtype(),
                a.shape().dims()
            )
        })?;
        let b_buf = get_cl_mem(b.buffer().as_ref()).map_err(|_| {
            anyhow!(
                "matmul (F32): B (weight) is not OpenCL buffer (kind={}, dtype={:?}, shape={:?})",
                buffer_kind_label(b.buffer().as_ref()),
                b.dtype(),
                b.shape().dims()
            )
        })?;
        let c_buf = get_cl_mem(out.buffer().as_ref()).map_err(|_| {
            anyhow!(
                "matmul (F32): Out is not OpenCL buffer (kind={}, dtype={:?}, shape={:?})",
                buffer_kind_label(out.buffer().as_ref()),
                out.dtype(),
                out.shape().dims()
            )
        })?;

        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let nb00 = 4u64;
        let nb01 = k as u64 * 4;
        let nb02 = n as u64 * k as u64 * 4;
        let nb03 = nb02;
        let ne10 = k as i32;
        let ne11 = m as i32;
        let ne12 = 1i32;
        let nb10 = 4u64;
        let nb11 = k as u64 * 4;
        let nb12 = m as u64 * k as u64 * 4;
        let nb13 = nb12;
        let ne0 = n as i32;
        let ne1 = m as i32;
        let r2 = 1i32;
        let r3 = 1i32;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_mul_mat_f32_f32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(a_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(c_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;

            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&nb00))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&nb01))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&nb02))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&nb03))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&ne11))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&nb10))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&nb11))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&nb12))?;
            ocl::core::set_kernel_arg(kernel, 19, ocl::core::ArgVal::scalar(&nb13))?;
            ocl::core::set_kernel_arg(kernel, 20, ocl::core::ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(kernel, 21, ocl::core::ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(kernel, 22, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 23, ocl::core::ArgVal::scalar(&r3))?;

            let local_size = 64usize;
            let global_work_size: [usize; 3] = [n * local_size, m.div_ceil(4), 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        match b.dtype() {
            DType::Q4_0 => self.matmul_q4_0(a, b, out),
            DType::F16 => self.matmul_f16(a, b, out),
            _ => self.matmul(a, b, out), // F32 path
        }
    }

    fn rms_norm(
        &self,
        x: &mut Tensor,
        weight: &Tensor,
        epsilon: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let w_buf = get_cl_mem(weight.buffer().as_ref())
            .map_err(|_| anyhow!("Weight is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_rms_norm_opt;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(w_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&epsilon))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&add_unit_i32))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;

            let global_work_size: [usize; 3] = [rows * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "rms_norm",
                1,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    fn rms_norm_oop(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        weight: &Tensor,
        epsilon: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let out_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;
        let w_buf = get_cl_mem(weight.buffer().as_ref())
            .map_err(|_| anyhow!("Weight is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_rms_norm_oop;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(w_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&epsilon))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&add_unit_i32))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;

            let global_work_size: [usize; 3] = [rows * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "rms_norm",
                1,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    fn add_rms_norm_oop(
        &self,
        x: &mut Tensor,
        residual: &Tensor,
        out: &mut Tensor,
        weight: &Tensor,
        epsilon: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let res_buf = get_cl_mem(residual.buffer().as_ref())
            .map_err(|_| anyhow!("Residual is not OpenCL buffer"))?;
        let out_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;
        let w_buf = get_cl_mem(weight.buffer().as_ref())
            .map_err(|_| anyhow!("Weight is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_add_rms_norm_oop;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(res_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(w_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&epsilon))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&add_unit_i32))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;

            let global_work_size: [usize; 3] = [rows * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "rms_norm",
                1,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_norm_merge(
        &self,
        prior_residual: &Tensor,
        gpu_partial: &Tensor,
        cpu_staging: &Tensor,
        norm_weight: &Tensor,
        out: &mut Tensor,
        residual_out: &mut Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        // Determine [rows, dim] from prior_residual shape — residual_out / out / weight
        // must have matching layout. In the decode path rows=1 (single-token boundary).
        let dims = prior_residual.shape().dims();
        if dims.is_empty() {
            anyhow::bail!("fused_norm_merge: prior_residual has 0 dims");
        }
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);

        let prior_buf = get_cl_mem(prior_residual.buffer().as_ref())
            .map_err(|_| anyhow!("prior_residual is not OpenCL buffer"))?;
        let gpu_buf = get_cl_mem(gpu_partial.buffer().as_ref())
            .map_err(|_| anyhow!("gpu_partial is not OpenCL buffer"))?;
        let cpu_buf = get_cl_mem(cpu_staging.buffer().as_ref())
            .map_err(|_| anyhow!("cpu_staging is not OpenCL buffer"))?;
        let w_buf = get_cl_mem(norm_weight.buffer().as_ref())
            .map_err(|_| anyhow!("norm_weight is not OpenCL buffer"))?;
        let out_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("out is not OpenCL buffer"))?;
        let res_out_buf = get_cl_mem(residual_out.buffer().as_ref())
            .map_err(|_| anyhow!("residual_out is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        // Prefer float4 kernel when dim aligns and the subgroup path is available.
        let kernel = if dim.is_multiple_of(4) {
            kernels
                .kernel_fused_norm_merge_f4
                .as_ref()
                .unwrap_or(&kernels.kernel_fused_norm_merge)
        } else {
            &kernels.kernel_fused_norm_merge
        };

        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(prior_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(gpu_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(cpu_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(w_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::mem(res_out_buf))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&eps))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&add_unit_i32))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;

            let global_work_size: [usize; 3] = [rows * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "fused_norm_merge",
                1,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        let dims = x.shape().dims();
        let (seq_len, num_heads, head_dim) = if dims.len() == 4 {
            (dims[1], dims[2], dims[3])
        } else if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            return Err(anyhow!("RoPE expects 3 or 4 dims"));
        };

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_rope_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(num_heads as i32)))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&(seq_len as i32)))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(start_pos as i32)))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&theta))?;

            let work_size = seq_len * num_heads * (head_dim / 2);
            self.enqueue_kernel_labeled(kernel, "rope", 1, &[work_size, 1, 1], None)?;
        }
        Ok(())
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        match (src.dtype(), dst.dtype()) {
            (DType::F32, DType::F16) => {
                let src_buf = get_cl_mem(src.buffer().as_ref())?;
                let dst_buf = get_cl_mem(dst.buffer().as_ref())?;
                let num_elements: usize = src.shape().dims().iter().product();
                let kernels = unsafe { &*self.kernels.get() };
                let kernel = &kernels.kernel_cast_f32_to_f16;
                unsafe {
                    ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(src_buf))?;
                    ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(dst_buf))?;
                    ocl::core::set_kernel_arg(
                        kernel,
                        2,
                        ocl::core::ArgVal::scalar(&(num_elements as i32)),
                    )?;
                    let gws: [usize; 3] = [num_elements.div_ceil(64) * 64, 1, 1];
                    let lws: [usize; 3] = [64, 1, 1];
                    self.enqueue_kernel_labeled(kernel, "cast", 1, &gws, Some(lws))?;
                }
                Ok(())
            }
            (DType::F32, DType::Q4_0) => {
                let src_buf = get_cl_mem(src.buffer().as_ref())?;
                let dst_buf = get_cl_mem(dst.buffer().as_ref())?;
                let num_elements: usize = src.shape().dims().iter().product();

                if !num_elements.is_multiple_of(32) {
                    return Err(anyhow!("Q4_0 cast requires size multiple of 32"));
                }

                let kernels = unsafe { &*self.kernels.get() };
                let kernel = &kernels.kernel_quantize_f32_to_q4_0;
                let num_blocks = num_elements / 32;

                unsafe {
                    ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(src_buf))?;
                    ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(dst_buf))?;
                    ocl::core::set_kernel_arg(
                        kernel,
                        2,
                        ocl::core::ArgVal::scalar(&(num_elements as i32)),
                    )?;

                    let local_size = 64;
                    let global_size = num_blocks.div_ceil(local_size) * local_size;

                    self.enqueue_kernel_labeled(
                        kernel,
                        "cast",
                        1,
                        &[global_size, 1, 1],
                        Some([local_size, 1, 1]),
                    )?;
                }
                Ok(())
            }
            _ => Err(anyhow!(
                "OpenCL cast: unsupported {:?} -> {:?}",
                src.dtype(),
                dst.dtype()
            )),
        }
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
        let k_src_mem = get_cl_mem(k_src.buffer().as_ref())?;
        let v_src_mem = get_cl_mem(v_src.buffer().as_ref())?;
        let k_dst_mem = get_cl_mem(k_dst.buffer().as_ref())?;
        let v_dst_mem = get_cl_mem(v_dst.buffer().as_ref())?;

        let n_elems: usize = k_src.shape().dims().iter().product(); // kv_heads * head_dim

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_kv_scatter_f32_to_f16;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(k_src_mem))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(v_src_mem))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(k_dst_mem))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(v_dst_mem))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&(capacity as i32)))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&(write_pos as i32)))?;

            let gws: [usize; 3] = [n_elems.div_ceil(64) * 64, 1, 1];
            let lws: [usize; 3] = [64, 1, 1];
            self.enqueue_kernel_labeled(kernel, "kv_update", 1, &gws, Some(lws))?;
        }
        Ok(())
    }

    fn supports_kv_scatter_batch(&self) -> bool {
        true
    }

    /// Batch F32->F16 KV scatter for prefill: single kernel launch writes
    /// `seq_len` positions for both K and V. Replaces the CPU per-token
    /// `copy_nonoverlapping` loop in `KVCache::update()` F16 HeadMajor path.
    ///
    /// NDRange: 3D `[head_dim, seq_len, kv_heads]`. Local size is
    /// `[min(head_dim, 64), 1, 1]` so typical (head_dim=64 or 128) fills one
    /// Adreno wavefront per workgroup without cross-SM reduction.
    fn kv_scatter_f32_to_f16_batch(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        n_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        write_pos_start: usize,
        seq_len: usize,
    ) -> Result<()> {
        if seq_len == 0 {
            return Ok(());
        }
        let k_src_mem = get_cl_mem(k_src.buffer().as_ref())?;
        let v_src_mem = get_cl_mem(v_src.buffer().as_ref())?;
        let k_dst_mem = get_cl_mem(k_dst.buffer().as_ref())?;
        let v_dst_mem = get_cl_mem(v_dst.buffer().as_ref())?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_kv_scatter_f32_to_f16_batch;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(k_src_mem))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(v_src_mem))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(k_dst_mem))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(v_dst_mem))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(n_kv_heads as i32)))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&(capacity as i32)))?;
            ocl::core::set_kernel_arg(
                kernel,
                7,
                ocl::core::ArgVal::scalar(&(write_pos_start as i32)),
            )?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&(seq_len as i32)))?;

            // Round head_dim up to the nearest multiple of the local-size (64 or head_dim).
            // For head_dim ∈ {64, 128, 256} this is exact; for smaller head_dim pick 32.
            let lx = if head_dim.is_multiple_of(64) {
                64
            } else if head_dim.is_multiple_of(32) {
                32
            } else {
                1
            };
            let gx = head_dim.div_ceil(lx) * lx;
            let gws: [usize; 3] = [gx, seq_len, n_kv_heads];
            let lws: [usize; 3] = [lx, 1, 1];
            self.enqueue_kernel_labeled(kernel, "kv_write", 3, &gws, Some(lws))?;
        }
        Ok(())
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
        // Flash-attention decode fast path: single-pass online softmax, vectorized
        // float4 K/V reads, no local-memory output reduction. Used when none of the
        // score-accumulator paths are active and the kernel's compile-time DK/DV
        // matches the current head_dim.
        //
        // When scores_out is requested, we still use flash attention for the output
        // vector and dispatch a separate lightweight score-only kernel (Q*K^T + softmax)
        // instead of falling back to the slow kernel_attn_gen_half.
        // Flash attention path — now also supports GPU score accumulator
        // writes directly from the Q1 kernel, so the legacy
        // `kernel_attn_gen_half` fallback is only needed for non-GPU-acc
        // score readback (plan.rs handles this via the Standard variant).
        let gpu_acc_score_triple: Option<(ocl::core::Mem, i32, i32)> = {
            let gpu_acc = unsafe { &*self.gpu_score_acc.get() };
            gpu_acc.as_ref().and_then(|acc| {
                if acc.is_active() {
                    let offset = acc.layer_offset_elems(acc.current_layer_idx()) as i32;
                    let stride = acc.score_stride() as i32;
                    // SAFETY: clone the cl_mem handle — OpenCL reference
                    // counting keeps the underlying allocation alive while
                    // the accumulator owns it.
                    Some((acc.score_buf_mem().clone(), offset, stride))
                } else {
                    None
                }
            })
        };

        let flash_score_arg = gpu_acc_score_triple
            .as_ref()
            .map(|(buf, off, stride)| (buf, *off, *stride));

        // When the CPU wants readback via `scores_out` (no GPU acc), we fall
        // back to the legacy path below since `scores_out` expects the result
        // in CPU memory. The GPU score accumulator path fully replaces the
        // slow kernel_attn_gen_half dispatch.
        let can_use_flash = gpu_acc_score_triple.is_some() || scores_out.is_none();
        if can_use_flash
            && self.flash_attention_decode_gpu(
                q,
                k_cache,
                v_cache,
                out,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
                flash_score_arg,
            )?
        {
            // Flash attention succeeded. For GPU acc path, scores are written
            // into the per-layer slice of the persistent score buffer; the
            // fused reduce runs at end_step(). For no-score path, we're done.
            return Ok(());
        }
        // Legacy path (scores_out readback requested and GPU acc inactive)
        // continues below with kernel_attn_gen_half + CPU readback.

        let q_buf =
            get_cl_mem(q.buffer().as_ref()).map_err(|_| anyhow!("Q is not OpenCL buffer"))?;
        let k_buf =
            get_cl_mem(k_cache.buffer().as_ref()).map_err(|_| anyhow!("K is not OpenCL buffer"))?;
        let v_buf =
            get_cl_mem(v_cache.buffer().as_ref()).map_err(|_| anyhow!("V is not OpenCL buffer"))?;
        let o_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();

        // Detect layout from shape: HeadMajor [batch, kv_heads, capacity, head_dim]
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };

        let (kv_pos_stride, kv_head_stride) = if is_head_major {
            (head_dim as i32, (capacity * head_dim) as i32)
        } else {
            ((num_heads_kv * head_dim) as i32, head_dim as i32)
        };

        let kernels = unsafe { &*self.kernels.get() };

        // Select kernel based on KV cache dtype
        let kernel = match k_cache.dtype() {
            DType::F16 => &kernels.kernel_attn_gen_half,
            _ => &kernels.kernel_attn_gen,
        };

        // GPU score accumulator path: when active, use persistent GPU score buffer
        // and skip CPU readback. Scores are reduced on-device and read only at eviction.
        // SAFETY: single-threaded access (same as kernels UnsafeCell pattern)
        let gpu_acc = unsafe { &*self.gpu_score_acc.get() };
        let use_gpu_acc = gpu_acc.as_ref().is_some_and(|acc| acc.is_active());

        let (write_scores, score_stride_val, score_layer_offset) = if use_gpu_acc {
            let acc = gpu_acc.as_ref().unwrap();
            let offset = acc.layer_offset_elems(acc.current_layer_idx()) as i32;
            (1i32, acc.score_stride() as i32, offset)
        } else {
            (
                scores_out.is_some() as i32,
                scores_out
                    .as_ref()
                    .map(|s| (s.len() / num_heads_q) as i32)
                    .unwrap_or(0),
                0i32,
            )
        };

        // GPU score buffer selection:
        // 1. GPU acc active -> persistent score_buf (no per-call alloc, no readback)
        // 2. scores_out requested -> per-call GPU alloc + CPU readback
        // 3. Neither -> dummy 1-element buffer
        let score_buf = if use_gpu_acc {
            None // using gpu_acc's persistent buffer
        } else if scores_out.is_some() {
            Some(unsafe {
                ocl::core::create_buffer::<_, f32>(
                    self.context.as_core(),
                    ocl::core::MEM_READ_WRITE | ocl::core::MEM_ALLOC_HOST_PTR,
                    num_heads_q * score_stride_val as usize,
                    None,
                )?
            })
        } else {
            None
        };
        let s_buf = if use_gpu_acc {
            gpu_acc.as_ref().unwrap().score_buf_mem()
        } else {
            score_buf.as_ref().unwrap_or(&self.dummy_score_buf)
        };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(k_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(v_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(o_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(s_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&(num_heads_q as i32)))?;
            ocl::core::set_kernel_arg(
                kernel,
                7,
                ocl::core::ArgVal::scalar(&(num_heads_kv as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                8,
                ocl::core::ArgVal::scalar(&(cache_seq_len as i32)),
            )?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&kv_pos_stride))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&kv_head_stride))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&write_scores))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&score_stride_val))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&score_layer_offset))?;
            ocl::core::set_kernel_arg(
                kernel,
                15,
                ocl::core::ArgVal::local::<f32>(&local_mem_size),
            )?;

            let global_work_size: [usize; 3] = [num_heads_q * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "attention",
                1,
                &global_work_size,
                Some(local_work_size),
            )?;
        }

        // Post-kernel score handling:
        // GPU acc path: scores stay in the per-layer slice of score_buf; a
        // single fused reduce kernel runs at end_step() (see gpu_score.rs).
        // Legacy path: blocking GPU->CPU readback of scores.
        if use_gpu_acc {
            // Nothing to do per layer — fused reduce runs at end_step().
        } else if let (Some(scores), Some(buf)) = (scores_out, &score_buf) {
            unsafe {
                ocl::core::enqueue_read_buffer(
                    &self.queue,
                    buf,
                    true, // blocking read
                    0,
                    scores,
                    None::<ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }

        Ok(())
    }

    fn silu_mul(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        let size4 = size / 4;

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf =
            get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_silu_mul_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size4 as i32)))?;

            self.enqueue_kernel_labeled(kernel, "silu_mul", 1, &[size4, 1, 1], None)?;
        }
        Ok(())
    }

    fn gelu_tanh_mul(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        let size4 = size / 4;

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf =
            get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_gelu_tanh_mul;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size4 as i32)))?;

            self.enqueue_kernel_labeled(kernel, "silu_mul", 1, &[size4, 1, 1], None)?;
        }
        Ok(())
    }

    fn add_assign(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        let size4 = size / 4;

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf =
            get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_add_assign_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size4 as i32)))?;

            self.enqueue_kernel_labeled(kernel, "add_assign", 1, &[size4, 1, 1], None)?;
        }
        Ok(())
    }

    fn add_row_bias(&self, x: &mut Tensor, bias: &Tensor) -> Result<()> {
        let x_dims = x.shape().dims();
        let dim = x_dims[x_dims.len() - 1];
        let total: usize = x_dims.iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let b_buf =
            get_cl_mem(bias.buffer().as_ref()).map_err(|_| anyhow!("Bias is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_add_row_bias;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(b_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&(total as i32)))?;

            let gws = total.div_ceil(64) * 64;
            self.enqueue_kernel_labeled(kernel, "add_assign", 1, &[gws, 1, 1], None)?;
        }
        Ok(())
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        _rows: usize,
        _cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        self.matmul_transposed(a, b, out)
    }

    fn scale(&self, x: &mut Tensor, val: f32) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_scale_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&val))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;

            self.enqueue_kernel_labeled(kernel, "other", 1, &[size, 1, 1], None)?;
        }
        Ok(())
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_softmax_opt;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;

            let global_work_size: [usize; 3] = [rows * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "other",
                1,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
        let dims = src.shape().dims();
        let k = dims[dims.len() - 1];

        let src_buf =
            get_cl_mem(src.buffer().as_ref()).map_err(|_| anyhow!("Src is not OpenCL buffer"))?;
        let idx_buf = get_cl_mem(indices.buffer().as_ref())
            .map_err(|_| anyhow!("Indices is not OpenCL buffer"))?;
        let dst_buf =
            get_cl_mem(dst.buffer().as_ref()).map_err(|_| anyhow!("Dst is not OpenCL buffer"))?;

        let ne00 = k as i32;
        let nb01 = match src.dtype() {
            DType::F32 => (k * 4) as u64,
            DType::F16 => (k * 2) as u64,
            DType::Q4_0 => (k / 32 * 18) as u64,
            _ => {
                return Err(anyhow!(
                    "Unsupported src dtype for gather: {:?}",
                    src.dtype()
                ));
            }
        };
        let nb02 = nb01 * dims[0] as u64;
        let nb03 = nb02;
        let ne10 = (indices.size() / 4) as i32; // u32 element count, not byte size
        let nb10 = 4u64;
        let nb11 = nb10 * ne10 as u64;
        let nb12 = nb11;
        let nb1 = (k * 4) as u64;
        let nb2 = nb1 * ne10 as u64;
        let nb3 = nb2;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match src.dtype() {
            DType::Q4_0 => &kernels.kernel_get_rows_q4_0,
            DType::F32 => &kernels.kernel_get_rows_f32,
            DType::F16 => &kernels.kernel_get_rows_f16,
            _ => return Err(anyhow!("Unsupported dtype")),
        };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(src_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(idx_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(dst_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&nb01))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&nb02))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&nb03))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&nb10))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&nb11))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&nb12))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&nb1))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&nb2))?;
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&nb3))?;

            let local_size = 64usize;
            let num_indices = indices.size() / 4; // u32 element count, not byte size
            let global_work_size: [usize; 3] = [num_indices * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "gather",
                1,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    fn buffer_shift(
        &self,
        tensor: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        if src_offset == dst_offset || count == 0 {
            return Ok(());
        }

        let type_size = match tensor.dtype() {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::U8 => 1,
            DType::Q4_0 => std::mem::size_of::<crate::quant::BlockQ4_0>(),
            _ => {
                return Err(anyhow!(
                    "Unsupported dtype for buffer_shift: {:?}",
                    tensor.dtype()
                ));
            }
        };

        // Prefer GPU copy (clEnqueueCopyBuffer) to avoid CPU cache pollution on
        // zero-copy (CL_MEM_ALLOC_HOST_PTR) buffers. This is critical for tensor
        // partition workloads where CPU memmove on mapped UMA buffers contends with
        // the CPU-side matmul, causing ~21ms TBT regression.
        if let Ok(buf_mem) = get_cl_mem(tensor.buffer().as_ref()) {
            let src_byte = src_offset * type_size;
            let dst_byte = dst_offset * type_size;
            let byte_count = count * type_size;

            // GPU copy path: use enqueue_copy_buffer via cl_mem handle.
            // No queue.finish() after enqueue — the in-order command queue guarantees
            // that subsequent kernel dispatches on the same queue are serialized after
            // this copy, so explicit synchronization would stall the GPU pipeline.
            let gpu_result = (|| -> Result<()> {
                // OpenCL spec: clEnqueueCopyBuffer with overlapping src/dst regions in
                // the same buffer is undefined behavior. Detect overlap and use a temp
                // buffer.
                let no_overlap =
                    (src_byte + byte_count <= dst_byte) || (dst_byte + byte_count <= src_byte);

                if no_overlap {
                    unsafe {
                        ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                            &self.queue,
                            buf_mem,
                            buf_mem,
                            src_byte,
                            dst_byte,
                            byte_count,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                    }
                } else {
                    // Overlap: 2-pass copy via temporary GPU buffer.
                    // Safe to drop temp after enqueue: clReleaseMemObject defers
                    // deallocation until pending commands referencing this buffer
                    // complete (OpenCL spec).
                    let temp = ocl::Buffer::<u8>::builder()
                        .queue(self.queue.clone())
                        .len(byte_count)
                        .build()?;
                    let temp_mem = temp.as_core();

                    unsafe {
                        ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                            &self.queue,
                            buf_mem,
                            temp_mem,
                            src_byte,
                            0,
                            byte_count,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                        ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                            &self.queue,
                            temp_mem,
                            buf_mem,
                            0,
                            dst_byte,
                            byte_count,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                    }
                }
                Ok(())
            })();

            match gpu_result {
                Ok(()) => return Ok(()),
                Err(e) => {
                    // Safety fallback: GPU copy failed, fall through to CPU memmove.
                    // Use std::sync::Once to emit the warning only once per process.
                    use std::sync::Once;
                    static WARN_ONCE: Once = Once::new();
                    WARN_ONCE.call_once(|| {
                        eprintln!(
                            "[buffer_shift] GPU enqueue_copy_buffer failed, \
                             falling back to CPU memmove: {e}"
                        );
                    });
                }
            }
        }

        // CPU fallback: pure-CPU buffers (SharedBuffer) or GPU copy failure.
        // Safety: as_mut_ptr() returns a valid mutable pointer for CPU-backed
        // buffers. std::ptr::copy handles overlapping regions correctly (memmove).
        let ptr = tensor.as_mut_ptr();
        if !ptr.is_null() {
            unsafe {
                std::ptr::copy(
                    ptr.add(src_offset * type_size),
                    ptr.add(dst_offset * type_size),
                    count * type_size,
                );
            }
            return Ok(());
        }

        Err(anyhow!(
            "buffer_shift: no GPU cl_mem and no CPU pointer available"
        ))
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
            _ => {
                return Err(anyhow!(
                    "Unsupported dtype for copy_slice: {:?}",
                    src.dtype()
                ));
            }
        };

        let src_m = get_cl_mem(src.buffer().as_ref());
        let dst_m = get_cl_mem(dst.buffer().as_ref());

        if let (Ok(sb), Ok(db)) = (src_m.as_ref(), dst_m.as_ref()) {
            let src_byte_off = src_offset * type_size;
            let dst_byte_off = dst_offset * type_size;
            let byte_len = count * type_size;

            unsafe {
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue,
                    *sb,
                    *db,
                    src_byte_off,
                    dst_byte_off,
                    byte_len,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            return Ok(());
        }

        if let Ok(db) = dst_m {
            let src_ptr = src.as_ptr();
            if !src_ptr.is_null() {
                let src_byte_off = src_offset * type_size;
                let dst_byte_off = dst_offset * type_size;
                let byte_len = count * type_size;
                unsafe {
                    ocl::core::enqueue_write_buffer(
                        &self.queue,
                        db,
                        true,
                        dst_byte_off,
                        std::slice::from_raw_parts(src_ptr.add(src_byte_off), byte_len),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
                return Ok(());
            }
        }

        if let Ok(sb) = src_m {
            let dst_ptr = dst.as_mut_ptr();
            if !dst_ptr.is_null() {
                let src_byte_off = src_offset * type_size;
                let dst_byte_off = dst_offset * type_size;
                let byte_len = count * type_size;
                unsafe {
                    ocl::core::enqueue_read_buffer(
                        &self.queue,
                        sb,
                        true,
                        src_byte_off,
                        std::slice::from_raw_parts_mut(dst_ptr.add(dst_byte_off), byte_len),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
                return Ok(());
            }
        }

        Err(anyhow!(
            "Unsupported copy_slice combination in OpenCL backend or null pointers"
        ))
    }

    // ── B-5b Phase 2 Stage 1: capability extension overrides ─────────────
    //
    // Stage 1 wires the owned `cpu_companion` and the empty
    // `OpenClSecondary` marker through the trait. Stage 2 will replace
    // existing `as_any().downcast_ref::<OpenCLBackend>()` and the direct
    // freestanding `fused_matmul_*` NEON call sites (see `cpu_kernels.rs`)
    // with these methods.

    fn cpu_companion(&self) -> &dyn Backend {
        &*self.cpu_companion
    }

    // yield_after_layer: trait default body (S-2 sprint 2026-05-24).

    // COLD-EXT: backend handle 노출 — qnn_oppkg swap path / secondary_mmap
    // loader / transformer loader 에서 `downcast_ref::<OpenCLBackend>()` 를
    // 대체하기 위한 string-keyed lookup. 자세한 정책은 `Backend::get_extension`
    // rustdoc + `arch/sprint_backend_extension_round.md` R-EXT-1 참조.
    fn get_extension(&self, name: &str) -> Option<&dyn std::any::Any> {
        match name {
            crate::backend::EXT_OPENCL_QUEUE | crate::backend::EXT_OPENCL_SECONDARY => {
                Some(self as &dyn std::any::Any)
            }
            // Sprint 2a Phase 2 (ENG-RPCMEM-024 / INV-RPCMEM-002): expose the
            // backend-agnostic rpcmem allocator. Returns `Some(&Arc<...>)` only
            // when `--opencl-rpcmem` was active and the eager dlopen succeeded;
            // otherwise `None` so the caller falls back to the standard mmap
            // path. Consumers downcast via `downcast_ref::<Arc<RpcmemAllocator>>()`
            // and clone the inner Arc.
            crate::backend::EXT_RPCMEM_ALLOCATOR => self
                .rpcmem_allocator
                .as_ref()
                .map(|a| a as &dyn std::any::Any),
            _ => None,
        }
    }
}

// ── Device capability queries (nosub / flash-decode availability) ──
impl OpenCLBackend {
    /// Returns true if this device lacks subgroup support (using nosub fallback kernels).
    /// Native quant-window attention (workgroup reduction) is preferred on these devices since
    /// the standard attention_gen also uses workgroup reduction.
    pub fn is_nosub(&self) -> bool {
        let kernels = unsafe { &*self.kernels.get() };
        kernels.f16_is_nosub
    }

    /// Returns true if the flash attention decode kernel is available for the
    /// given head_dim. Used by plan.rs to gate the `StandardFlash` attention
    /// variant, and by host tests to decide whether to exercise the flash path.
    /// Supported head_dim values match the DK variants compiled in `new()`.
    pub fn has_flash_decode_kernel(&self, head_dim: usize) -> bool {
        let kernels = unsafe { &*self.kernels.get() };
        match head_dim {
            64 => kernels.kernel_flash_attn_f32_f16_q1_dk64.is_some(),
            128 => kernels.kernel_flash_attn_f32_f16_q1_dk128.is_some(),
            _ => false,
        }
    }
}

// FORMAT Phase 2 Stage E: the OpenCL backend no longer implements `QuantAttnBackend` — the
// quant-window kernels moved to the `kivi` plugin. The backend only LENDS its live GPU context/queue
// so a `--backend-cap` plugin can build and dispatch its own kernels: `with_quant_attn_make_args`
// below packs the borrowed context/device/build_opts (used by `init.rs` to resolve a named
// cap), and `cl_command_queue_ptr` (above) lends the live queue per call.
impl OpenCLBackend {
    /// Build a [`argus_extension_api::QuantAttnMakeArgs`] from this backend's live GPU context and
    /// invoke `f` with it. Used by `--backend-cap <name>` to construct a named quantized
    /// attention capability via `resolve_quant_attn_capability`. The `build_opts` C string is
    /// owned only for the duration of `f` — `QuantAttnMakeArgs` is borrow-for-make (C7/D4) and a
    /// Copy POD, so the pointer must not escape `f`.
    pub fn with_quant_attn_make_args<R>(
        &self,
        f: impl FnOnce(&argus_extension_api::QuantAttnMakeArgs) -> R,
    ) -> R {
        let opts = std::ffi::CString::new(self.cl_opts.clone()).unwrap_or_default();
        let make_args = argus_extension_api::QuantAttnMakeArgs {
            cl_ctx: <&Context as ClContextPtr>::as_ptr(&&self.context),
            device: <Device as ClDeviceIdPtr>::as_ptr(&self.device),
            build_opts: opts.as_ptr(),
        };
        f(&make_args)
    }

    /// Build a [`argus_extension_api::ScoreReduceMakeArgs`] from this backend's live GPU context and
    /// invoke `f` with it. Used by `init_gpu_score_acc` to construct the `attn_score` GPU score
    /// reducer via `find_score_reducer`. Borrow-for-make: the `build_opts` C string is owned only for
    /// the duration of `f` (a Copy POD whose pointer must not escape `f`). Mirror of
    /// [`with_quant_attn_make_args`](Self::with_quant_attn_make_args) for the observer/score axis.
    pub fn with_score_reduce_make_args<R>(
        &self,
        f: impl FnOnce(&argus_extension_api::ScoreReduceMakeArgs) -> R,
    ) -> R {
        let opts = std::ffi::CString::new(self.cl_opts.clone()).unwrap_or_default();
        let make_args = argus_extension_api::ScoreReduceMakeArgs {
            cl_context: <&Context as ClContextPtr>::as_ptr(&&self.context),
            cl_device: <Device as ClDeviceIdPtr>::as_ptr(&self.device),
            build_opts: opts.as_ptr(),
        };
        f(&make_args)
    }
}

#[cfg(test)]
mod gpu_buffer_shift_tests {
    use super::*;
    use crate::shape::Shape;

    fn try_create_backend() -> Option<Arc<OpenCLBackend>> {
        OpenCLBackend::new().ok().map(Arc::new)
    }

    /// Allocate a GPU-only (non-zero-copy) buffer, write data, return Tensor.
    fn make_gpu_tensor(backend: &Arc<OpenCLBackend>, data: &[f32], shape: Vec<usize>) -> Tensor {
        let byte_len = data.len() * 4;
        let mem = memory::OpenCLMemory::new(
            backend.context.clone(),
            backend.queue.clone(),
            false, // device-only
        );
        let buf = mem.alloc(byte_len, DType::F32).unwrap();
        let tensor = Tensor::new(Shape::new(shape), buf, backend.clone());

        // Write data to GPU buffer
        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        tensor
    }

    /// Read f32 data back from a GPU tensor.
    fn read_gpu_tensor(backend: &Arc<OpenCLBackend>, tensor: &Tensor) -> Vec<f32> {
        let n = tensor.buffer().size() / 4;
        let mut result = vec![0.0f32; n];
        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        result
    }

    #[test]
    fn test_buffer_shift_gpu_no_overlap_f32() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // [1, 2, 3, 4, 5, 6, 7, 8] — shift elements [4..8] to [0..4] (no overlap)
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut tensor = make_gpu_tensor(&backend, &data, vec![8]);

        // src_offset=4, dst_offset=0, count=4 → no overlap (count <= src_offset)
        backend.buffer_shift(&mut tensor, 4, 0, 4).unwrap();

        let result = read_gpu_tensor(&backend, &tensor);
        assert_eq!(&result[0..4], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_buffer_shift_gpu_overlap_f32() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // [1, 2, 3, 4, 5, 6, 7, 8] — shift [2..8] to [0..6] (overlap: count=6 > src=2)
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut tensor = make_gpu_tensor(&backend, &data, vec![8]);

        backend.buffer_shift(&mut tensor, 2, 0, 6).unwrap();

        let result = read_gpu_tensor(&backend, &tensor);
        assert_eq!(&result[0..6], &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_buffer_shift_gpu_zero_count() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let data: Vec<f32> = (1..=4).map(|x| x as f32).collect();
        let mut tensor = make_gpu_tensor(&backend, &data, vec![4]);

        // No-op
        backend.buffer_shift(&mut tensor, 2, 0, 0).unwrap();

        let result = read_gpu_tensor(&backend, &tensor);
        assert_eq!(&result[..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_buffer_shift_gpu_same_offset() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let data: Vec<f32> = (1..=4).map(|x| x as f32).collect();
        let mut tensor = make_gpu_tensor(&backend, &data, vec![4]);

        // src == dst → no-op
        backend.buffer_shift(&mut tensor, 1, 1, 2).unwrap();

        let result = read_gpu_tensor(&backend, &tensor);
        assert_eq!(&result[..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_prune_prefix_opencl_buffer() {
        use crate::kv::kv_cache::KVCache;

        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let heads = 1;
        let dim = 4;
        let max_seq = 16;
        let mem = memory::OpenCLMemory::new(backend.context.clone(), backend.queue.clone(), false);

        let k_buf = mem.alloc(max_seq * heads * dim * 4, DType::F32).unwrap();
        let v_buf = mem.alloc(max_seq * heads * dim * 4, DType::F32).unwrap();
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            k_buf,
            backend.clone() as Arc<dyn Backend>,
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            v_buf,
            backend.clone() as Arc<dyn Backend>,
        );
        let mut cache = KVCache::new(k, v, max_seq);

        // Fill 8 positions: pos i → all values = (i+1) as f32
        let cpu_mem = crate::memory::host::shared::SharedBuffer::new(heads * dim * 4, DType::F32);
        for i in 0..8 {
            let val = (i + 1) as f32;
            let kb = Arc::new(crate::memory::host::shared::SharedBuffer::new(
                heads * dim * 4,
                DType::F32,
            ));
            let vb = Arc::new(crate::memory::host::shared::SharedBuffer::new(
                heads * dim * 4,
                DType::F32,
            ));
            unsafe {
                let kp = kb.as_mut_ptr() as *mut f32;
                let vp = vb.as_mut_ptr() as *mut f32;
                for j in 0..(heads * dim) {
                    *kp.add(j) = val;
                    *vp.add(j) = val * 10.0;
                }
            }
            let kt = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                kb,
                backend.clone() as Arc<dyn Backend>,
            );
            let vt = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                vb,
                backend.clone() as Arc<dyn Backend>,
            );
            cache.update(&kt, &vt).unwrap();
        }
        assert_eq!(cache.current_pos, 8);

        // Prune first 3 tokens
        cache.prune_prefix(3).unwrap();
        assert_eq!(cache.current_pos, 5);

        // Read K buffer back and verify: pos 0 should be old pos 3 (value 4.0)
        let k_data = {
            let cl_mem = get_cl_mem(cache.k_buffer.buffer().as_ref()).unwrap();
            let n = 8 * heads * dim; // read enough elements
            let mut buf = vec![0.0f32; n];
            unsafe {
                ocl::core::enqueue_read_buffer(
                    &backend.queue,
                    cl_mem,
                    true,
                    0,
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n * 4),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )
                .unwrap();
            }
            buf
        };

        // pos 0 = old pos 3 = value 4.0
        assert_eq!(k_data[0], 4.0);
        // pos 1 = old pos 4 = value 5.0
        assert_eq!(k_data[dim], 5.0);
        // pos 4 = old pos 7 = value 8.0
        assert_eq!(k_data[4 * dim], 8.0);

        drop(cpu_mem); // suppress unused warning
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod noshuffle_tests {
    use super::*;
    use crate::shape::Shape;

    fn try_create_backend() -> Option<Arc<OpenCLBackend>> {
        OpenCLBackend::new().ok().map(Arc::new)
    }

    /// Q4_0 block: 2 bytes d (f16) + 16 bytes qs (QK4_0/2 = 16 nibble bytes)
    const BLOCK_Q4_0_SIZE: usize = 18;
    const QK4_0: usize = 32;

    /// Build a minimal Q4_0 block from known quant values.
    ///
    /// `d_f32`: scale (will be converted to f16)
    /// `quants`: 32 signed 4-bit values (-8..7), stored as unsigned 0..15 in nibbles
    fn make_q4_0_block(d_f32: f32, quants: &[i8; 32]) -> [u8; BLOCK_Q4_0_SIZE] {
        let mut block = [0u8; BLOCK_Q4_0_SIZE];
        let d_f16 = half::f16::from_f32(d_f32);
        block[0..2].copy_from_slice(&d_f16.to_le_bytes());
        // Pack 32 quants into 16 bytes: qs[i] = (q[2i] & 0xF) | (q[2i+1] << 4)
        // In Q4_0, stored nibble = quant + 8 (unsigned 0..15)
        for i in 0..16 {
            let lo = (quants[i] + 8) as u8 & 0x0F;
            let hi = (quants[i + 16] + 8) as u8 & 0x0F;
            block[2 + i] = lo | (hi << 4);
        }
        block
    }

    /// Reference CPU dequant for a Q4_0 block -> 32 f32 values.
    fn dequant_q4_0_block(block: &[u8; BLOCK_Q4_0_SIZE]) -> [f32; 32] {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let mut out = [0.0f32; 32];
        for i in 0..16 {
            let byte = block[2 + i];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            out[i] = lo as f32 * d;
            out[i + 16] = hi as f32 * d;
        }
        out
    }

    /// Reference CPU matmul: weight (ne01 x ne00, Q4_0) * activation (ne00,) -> output (ne01,)
    fn reference_matmul_q4_0(
        blocks: &[[u8; BLOCK_Q4_0_SIZE]],
        activation: &[f32],
        ne00: usize,
        ne01: usize,
    ) -> Vec<f32> {
        let blocks_per_row = ne00 / QK4_0;
        let mut output = vec![0.0f32; ne01];
        for row in 0..ne01 {
            let mut sum = 0.0f32;
            for k_blk in 0..blocks_per_row {
                let block = &blocks[row * blocks_per_row + k_blk];
                let dequants = dequant_q4_0_block(block);
                for j in 0..QK4_0 {
                    sum += dequants[j] * activation[k_blk * QK4_0 + j];
                }
            }
            output[row] = sum;
        }
        output
    }

    /// Test that the SOA conversion + transpose produces the correct transposed layout.
    ///
    /// This verifies the nibble rearrangement and 2D transpose independently of GEMV.
    #[test]
    fn test_soa_conversion_transpose_layout() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // Small matrix: 4 rows, K=64 -> blocks_per_row=2, total 8 blocks
        let ne01 = 4usize;
        let ne00 = 64usize;
        let blocks_per_row = ne00 / QK4_0;
        let num_blocks = ne01 * blocks_per_row;

        // Create blocks with known patterns
        let mut all_blocks = Vec::new();
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let d = (row + 1) as f32 * 0.1 + k as f32 * 0.01;
                let mut quants = [0i8; 32];
                for j in 0..32 {
                    quants[j] = ((row * 100 + k * 32 + j) % 15) as i8 - 7;
                }
                all_blocks.push(make_q4_0_block(d, &quants));
            }
        }

        // Upload raw Q4_0 data to GPU
        let raw_bytes: Vec<u8> = all_blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let src_buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                raw_bytes.len(),
                None,
            )
            .unwrap()
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &src_buf,
                true,
                0,
                &raw_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        // Run SOA conversion + transpose
        let (q_buf, d_buf, _q_img) = backend
            .convert_q4_0_to_noshuffle(&src_buf, num_blocks, ne00, ne01)
            .unwrap();

        // Read back transposed q and d
        let q_total_ushort = ne01 * (ne00 / 4);
        let mut q_transposed = vec![0u16; q_total_ushort];
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &q_buf,
                true,
                0,
                std::slice::from_raw_parts_mut(
                    q_transposed.as_mut_ptr() as *mut u8,
                    q_total_ushort * 2,
                ),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        let mut d_transposed = vec![0u16; num_blocks];
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &d_buf,
                true,
                0,
                std::slice::from_raw_parts_mut(
                    d_transposed.as_mut_ptr() as *mut u8,
                    num_blocks * 2,
                ),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        // Verify d transpose: d_transposed[k * ne01 + row] == original d[row][k]
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let original_d = half::f16::from_le_bytes([
                    all_blocks[row * blocks_per_row + k][0],
                    all_blocks[row * blocks_per_row + k][1],
                ]);
                let transposed_d = half::f16::from_bits(d_transposed[k * ne01 + row]);
                assert_eq!(
                    original_d.to_bits(),
                    transposed_d.to_bits(),
                    "d mismatch at row={}, k={}: original={:?}, transposed={:?}",
                    row,
                    k,
                    original_d,
                    transposed_d
                );
            }
        }

        // Verify q transpose layout: each row's noshuffle block data should be
        // accessible at the correct transposed position.
        //
        // After noshuffle conversion, block[row][k] has 16 bytes = 8 ushort.
        // After transpose, ushort col c at row r is at q_transposed[c * ne01 + r].
        //
        // For row `row`, block `k`, the ushort column range is k*8..k*8+7.
        // We can verify by computing the expected noshuffle output for each block
        // and checking the transposed positions.
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let block = &all_blocks[row * blocks_per_row + k];
                // Compute expected noshuffle q bytes (same logic as kernel)
                let mut expected_q = [0u8; 16];
                for i in 0..8 {
                    let x0 = block[2 + 2 * i];
                    let x1 = block[2 + 2 * i + 1];
                    expected_q[i] = (x0 & 0x0F) | ((x1 & 0x0F) << 4);
                    expected_q[i + 8] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);
                }

                // expected_q as 8 ushort (LE)
                let mut expected_ushort = [0u16; 8];
                for i in 0..8 {
                    expected_ushort[i] =
                        expected_q[2 * i] as u16 | ((expected_q[2 * i + 1] as u16) << 8);
                }

                // Check transposed positions: col = k*8+j, row = row
                for j in 0..8 {
                    let col_ushort = k * 8 + j;
                    let transposed_val = q_transposed[col_ushort * ne01 + row];
                    assert_eq!(
                        expected_ushort[j], transposed_val,
                        "q mismatch at row={}, k={}, ushort_j={}: expected={:#06x}, got={:#06x}",
                        row, k, j, expected_ushort[j], transposed_val
                    );
                }
            }
        }

        eprintln!(
            "[PASS] SOA conversion + transpose layout verified for {}x{}",
            ne01, ne00
        );
    }

    /// Test that the GEMV indexing logic in the transposed layout produces correct
    /// uint values when accessed with the kernel's stride pattern.
    ///
    /// This simulates the kernel's memory access pattern without actually running
    /// the GPU kernel (no sub_group_broadcast needed).
    #[test]
    fn test_noshuffle_gemv_indexing() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // 4 rows, K=128 -> blocks_per_row=4
        let ne01 = 4usize;
        let ne00 = 128usize;
        let blocks_per_row = ne00 / QK4_0;
        let num_blocks = ne01 * blocks_per_row;

        // Create blocks with known scale and quant patterns
        let mut all_blocks = Vec::new();
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let d = 0.5f32;
                let mut quants = [0i8; 32];
                for j in 0..32 {
                    quants[j] = ((row * 1000 + k * 32 + j) % 15) as i8 - 7;
                }
                all_blocks.push(make_q4_0_block(d, &quants));
            }
        }

        // Upload and convert
        let raw_bytes: Vec<u8> = all_blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let src_buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                raw_bytes.len(),
                None,
            )
            .unwrap()
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &src_buf,
                true,
                0,
                &raw_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        let (q_buf, d_buf, _q_img) = backend
            .convert_q4_0_to_noshuffle(&src_buf, num_blocks, ne00, ne01)
            .unwrap();

        // Read back as uint (for q) and u16 (for d)
        let q_total_uint = ne01 * ne00 / 8; // ne01 * (K/4) ushort / 2
        let mut q_uint = vec![0u32; q_total_uint];
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &q_buf,
                true,
                0,
                std::slice::from_raw_parts_mut(q_uint.as_mut_ptr() as *mut u8, q_total_uint * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        let mut d_u16 = vec![0u16; num_blocks];
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &d_buf,
                true,
                0,
                std::slice::from_raw_parts_mut(d_u16.as_mut_ptr() as *mut u8, num_blocks * 2),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        let line_stride_a = ne01 / 2;
        let block_stride_a = 4 * ne01;

        // Simulate the kernel access pattern for each row-pair and K-block
        for gid in 0..ne01 / 2 {
            for k in 0..blocks_per_row {
                // Read scale (half2 = two consecutive u16 as half pair)
                let scale_idx = gid + k * line_stride_a;
                let scale_even = half::f16::from_bits(d_u16[scale_idx * 2]);
                let scale_odd = half::f16::from_bits(d_u16[scale_idx * 2 + 1]);

                // Expected scales
                let row_even = 2 * gid;
                let row_odd = 2 * gid + 1;
                let expected_d_even = half::f16::from_le_bytes([
                    all_blocks[row_even * blocks_per_row + k][0],
                    all_blocks[row_even * blocks_per_row + k][1],
                ]);
                let expected_d_odd = half::f16::from_le_bytes([
                    all_blocks[row_odd * blocks_per_row + k][0],
                    all_blocks[row_odd * blocks_per_row + k][1],
                ]);

                assert_eq!(
                    scale_even.to_bits(),
                    expected_d_even.to_bits(),
                    "scale even mismatch: gid={}, k={}",
                    gid,
                    k
                );
                assert_eq!(
                    scale_odd.to_bits(),
                    expected_d_odd.to_bits(),
                    "scale odd mismatch: gid={}, k={}",
                    gid,
                    k
                );

                // Read 8 uint values (matching kernel's regA pattern)
                for i in 0..8 {
                    let q_idx = gid + k * block_stride_a + line_stride_a * i;
                    assert!(
                        q_idx < q_total_uint,
                        "q_idx {} out of range (max {})",
                        q_idx,
                        q_total_uint
                    );
                    let val = q_uint[q_idx];
                    let lo_ushort = (val & 0xFFFF) as u16;
                    let hi_ushort = (val >> 16) as u16;

                    // lo_ushort should be from row_even, hi_ushort from row_odd
                    // Both at the same ushort column: k*8 + i
                    // Compute expected from noshuffle conversion
                    for (check_row, check_ushort, label) in
                        [(row_even, lo_ushort, "even"), (row_odd, hi_ushort, "odd")]
                    {
                        let block = &all_blocks[check_row * blocks_per_row + k];
                        let mut expected_q = [0u8; 16];
                        for ii in 0..8 {
                            let x0 = block[2 + 2 * ii];
                            let x1 = block[2 + 2 * ii + 1];
                            expected_q[ii] = (x0 & 0x0F) | ((x1 & 0x0F) << 4);
                            expected_q[ii + 8] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);
                        }
                        // ushort column i within this block's 8 ushorts
                        let expected_val =
                            expected_q[2 * i] as u16 | ((expected_q[2 * i + 1] as u16) << 8);
                        assert_eq!(
                            check_ushort, expected_val,
                            "q {} row mismatch: gid={}, k={}, i={}: got={:#06x}, expected={:#06x}",
                            label, gid, k, i, check_ushort, expected_val
                        );
                    }
                }
            }
        }

        eprintln!("[PASS] GEMV indexing verified for {}x{}", ne01, ne00);
    }

    /// End-to-end test: compare noshuffle GEMV output against reference CPU dequant.
    ///
    /// On macOS (Apple GPU), the noshuffle GEMV kernel may not compile due to missing
    /// sub_group_broadcast / cl_qcom_reqd_sub_group_size. In that case, this test
    /// is skipped gracefully.
    #[test]
    fn test_noshuffle_q4_0_correctness() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // 8 rows, K=128 (small but exercises multiple blocks and row-pairs)
        let ne01 = 8usize;
        let ne00 = 128usize;
        let blocks_per_row = ne00 / QK4_0;
        let num_blocks = ne01 * blocks_per_row;

        // Create blocks with non-trivial patterns
        let mut all_blocks = Vec::new();
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let d = 0.1 + 0.05 * row as f32 + 0.01 * k as f32;
                let mut quants = [0i8; 32];
                for j in 0..32 {
                    quants[j] = ((row * 7 + k * 3 + j * 5) % 15) as i8 - 7;
                }
                all_blocks.push(make_q4_0_block(d, &quants));
            }
        }

        // Create activation vector
        let mut activation = vec![0.0f32; ne00];
        for i in 0..ne00 {
            activation[i] = (i as f32 * 0.01).sin();
        }

        // Reference CPU result
        let reference = reference_matmul_q4_0(&all_blocks, &activation, ne00, ne01);

        // Upload Q4_0 data
        let raw_bytes: Vec<u8> = all_blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let src_buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                raw_bytes.len(),
                None,
            )
            .unwrap()
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &src_buf,
                true,
                0,
                &raw_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        // Convert to noshuffle SOA + transpose
        let (q_buf, d_buf, _q_img) = backend
            .convert_q4_0_to_noshuffle(&src_buf, num_blocks, ne00, ne01)
            .unwrap();

        // Upload activation
        let act_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                ne00,
                None,
            )
            .unwrap()
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &act_buf,
                true,
                0,
                std::slice::from_raw_parts(activation.as_ptr() as *const u8, ne00 * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        // Allocate output
        let dst_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                ne01,
                None,
            )
            .unwrap()
        };

        // Try running the noshuffle GEMV kernel
        match backend.matmul_q4_0_noshuffle(&q_buf, &d_buf, &act_buf, &dst_buf, ne00, ne01) {
            Ok(()) => {
                backend.queue.finish().unwrap();

                // Read back output
                let mut output = vec![0.0f32; ne01];
                unsafe {
                    ocl::core::enqueue_read_buffer(
                        &backend.queue,
                        &dst_buf,
                        true,
                        0,
                        std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u8, ne01 * 4),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )
                    .unwrap();
                }

                // Compare with reference
                let max_abs_error = reference
                    .iter()
                    .zip(output.iter())
                    .map(|(r, o)| (r - o).abs())
                    .fold(0.0f32, f32::max);

                eprintln!("Reference: {:?}", &reference);
                eprintln!("GPU:       {:?}", &output);
                eprintln!("Max abs error: {}", max_abs_error);

                // Q4_0 dequant introduces rounding, so allow moderate tolerance
                assert!(
                    max_abs_error < 0.01,
                    "GEMV output diverges from reference: max_abs_error={} (threshold=0.01)",
                    max_abs_error
                );

                eprintln!(
                    "[PASS] Noshuffle GEMV correctness verified for {}x{}",
                    ne01, ne00
                );
            }
            Err(e) => {
                eprintln!(
                    "[SKIPPED] Noshuffle GEMV kernel failed to compile/dispatch (expected on \
                     macOS without Adreno extensions): {}",
                    e
                );
                // SOA conversion + transpose tests above already validate the data layout.
            }
        }
    }

    #[test]
    fn test_image1d_buffer_creation_and_readback() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // Create a buffer with known data
        let data: Vec<u32> = (0..256).collect();
        let buf = unsafe {
            let b = ocl::core::create_buffer::<_, u32>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                data.len(),
                None,
            )
            .unwrap();
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &b,
                true,
                0,
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
            b
        };

        // Create image1d_buffer_t wrapping it (R32UI format)
        use ocl::core::{
            ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat, MemObjectType,
        };
        let img_fmt = ImageFormat::new(ImageChannelOrder::R, ImageChannelDataType::UnsignedInt32);
        let img_desc = ImageDescriptor::new(
            MemObjectType::Image1dBuffer,
            data.len(),
            0,
            0,
            0,
            0,
            0,
            Some(buf.clone()),
        );
        let result = unsafe {
            ocl::core::create_image(
                backend.context.as_core(),
                ocl::core::MEM_READ_ONLY,
                &img_fmt,
                &img_desc,
                None::<&[u32]>,
                None,
            )
        };

        match result {
            Ok(img) => {
                eprintln!(
                    "[PASS] image1d_buffer_t R32UI created: width={}",
                    data.len()
                );
                drop(img);
            }
            Err(e) => {
                eprintln!("[SKIPPED] image1d_buffer_t not supported: {}", e);
            }
        }

        // Also test RGBA32F format (for activation vectors)
        let float_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let float_buf = unsafe {
            let b = ocl::core::create_buffer::<_, f32>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                float_data.len(),
                None,
            )
            .unwrap();
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &b,
                true,
                0,
                std::slice::from_raw_parts(float_data.as_ptr() as *const u8, float_data.len() * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
            b
        };

        let act_fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
        let act_desc = ImageDescriptor::new(
            MemObjectType::Image1dBuffer,
            float_data.len() / 4, // RGBA = 4 floats per texel
            0,
            0,
            0,
            0,
            0,
            Some(float_buf.clone()),
        );
        let act_result = unsafe {
            ocl::core::create_image(
                backend.context.as_core(),
                ocl::core::MEM_READ_ONLY,
                &act_fmt,
                &act_desc,
                None::<&[f32]>,
                None,
            )
        };

        match act_result {
            Ok(img) => {
                eprintln!(
                    "[PASS] image1d_buffer_t RGBA32F created: width={}",
                    float_data.len() / 4
                );
                drop(img);
            }
            Err(e) => {
                eprintln!("[SKIPPED] image1d_buffer_t RGBA32F not supported: {}", e);
            }
        }
    }

    // --- Q4_0 tiled GEMM (mul_mm_q4_0_f32_l4_lm) correctness ---

    /// Build a Q4_0 weight Tensor [N, K] populated with random-ish deterministic quants.
    fn make_q4_0_weight_tensor(
        backend: &Arc<OpenCLBackend>,
        n: usize,
        k: usize,
        seed: u32,
    ) -> (Tensor, Vec<[u8; BLOCK_Q4_0_SIZE]>) {
        assert!(k.is_multiple_of(QK4_0), "K must be multiple of QK4_0=32");
        let blocks_per_row = k / QK4_0;
        let num_blocks = n * blocks_per_row;

        // Deterministic pseudo-random block generation (LCG for reproducibility).
        let mut rng_state: u32 = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut rng = || {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            rng_state
        };

        let mut all_blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let d = 0.01 + ((rng() % 256) as f32) * 0.001;
            let mut quants = [0i8; 32];
            for q in quants.iter_mut() {
                *q = ((rng() % 15) as i8) - 7;
            }
            all_blocks.push(make_q4_0_block(d, &quants));
        }

        // Upload to a Q4_0 DType Tensor.
        let raw_bytes: Vec<u8> = all_blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let mem = memory::OpenCLMemory::new(
            backend.context.clone(),
            backend.queue.clone(),
            false, // device-only
        );
        let buf = mem.alloc(raw_bytes.len(), DType::Q4_0).unwrap();
        let tensor = Tensor::new(Shape::new(vec![n, k]), buf, backend.clone());

        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                &raw_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        (tensor, all_blocks)
    }

    /// Build an F32 activation Tensor [M, K] with deterministic values.
    fn make_f32_activation_tensor(
        backend: &Arc<OpenCLBackend>,
        m: usize,
        k: usize,
    ) -> (Tensor, Vec<f32>) {
        let total = m * k;
        let mut data = vec![0.0f32; total];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i as f32) * 0.0137 + ((i % 19) as f32) * 0.05).sin() * 0.5;
        }
        let byte_len = total * 4;
        let mem = memory::OpenCLMemory::new(backend.context.clone(), backend.queue.clone(), false);
        let buf = mem.alloc(byte_len, DType::F32).unwrap();
        let tensor = Tensor::new(Shape::new(vec![m, k]), buf, backend.clone());

        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        (tensor, data)
    }

    /// Allocate a zero-initialised F32 output Tensor [M, N].
    fn make_f32_output_tensor(backend: &Arc<OpenCLBackend>, m: usize, n: usize) -> Tensor {
        let total = m * n;
        let byte_len = total * 4;
        let mem = memory::OpenCLMemory::new(backend.context.clone(), backend.queue.clone(), false);
        let buf = mem.alloc(byte_len, DType::F32).unwrap();
        let tensor = Tensor::new(Shape::new(vec![m, n]), buf, backend.clone());

        // Zero init.
        let zeros = vec![0.0f32; total];
        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(zeros.as_ptr() as *const u8, byte_len) };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        tensor
    }

    /// Read [M, N] F32 output back.
    fn read_f32_output(backend: &Arc<OpenCLBackend>, out: &Tensor) -> Vec<f32> {
        let n_elem = out.buffer().size() / 4;
        let mut result = vec![0.0f32; n_elem];
        let cl_mem = get_cl_mem(out.buffer().as_ref()).unwrap();
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n_elem * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        result
    }

    /// CPU reference GEMM for Q4_0 weight [N, K] x F32 activation [M, K] -> [M, N].
    /// Output layout: out[m * N + n] = sum_k weight[n, k] * act[m, k].
    fn reference_gemm_q4_0(
        blocks: &[[u8; BLOCK_Q4_0_SIZE]],
        activation: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let blocks_per_row = k / QK4_0;
        let mut out = vec![0.0f32; m * n];
        // Pre-dequant weights into a flat [N, K] f32 matrix for clarity.
        let mut w = vec![0.0f32; n * k];
        for row in 0..n {
            for kb in 0..blocks_per_row {
                let dq = dequant_q4_0_block(&blocks[row * blocks_per_row + kb]);
                for j in 0..QK4_0 {
                    w[row * k + kb * QK4_0 + j] = dq[j];
                }
            }
        }
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += w[ni * k + ki] * activation[mi * k + ki];
                }
                out[mi * n + ni] = acc;
            }
        }
        out
    }

    /// Verify tiled Q4_0 GEMM (`matmul_gemm_q4_0`) matches the CPU reference
    /// for a shape that activates the GEMM path (m >= 32).
    ///
    /// Shape chosen:
    ///   * m = 64 — exactly one BM tile
    ///   * n = 128 — two BN tiles
    ///   * k = 256 — 8 BK tiles
    ///
    /// All dimensions are multiples of 4 (the aligned fast path), but the
    /// kernel itself handles arbitrary `m` via the boundary check in the
    /// store loop.
    #[test]
    fn test_matmul_gemm_q4_0_basic() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        {
            let kernels = unsafe { &*backend.kernels.get() };
            if kernels.kernel_mul_mm_q4_0_f32.is_none() {
                eprintln!("[SKIPPED] mul_mm_q4_0_f32_l4_lm kernel not available");
                return;
            }
        }

        let m = 64usize;
        let n = 128usize;
        let k = 256usize;

        let (w_tensor, blocks) = make_q4_0_weight_tensor(&backend, n, k, 42);
        let (a_tensor, a_data) = make_f32_activation_tensor(&backend, m, k);
        let mut out_tensor = make_f32_output_tensor(&backend, m, n);

        // Force GEMM path by calling matmul_q4_0 (m >= 32 dispatches to matmul_gemm_q4_0
        // when no noshuffle SOA entry exists for the raw Q4_0 weight).
        backend
            .matmul_q4_0(&a_tensor, &w_tensor, &mut out_tensor)
            .expect("matmul_q4_0 GEMM path");
        backend.queue.finish().unwrap();

        let got = read_f32_output(&backend, &out_tensor);
        let expected = reference_gemm_q4_0(&blocks, &a_data, m, n, k);

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            let diff = (g - e).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            let denom = e.abs().max(1e-6);
            let rel = diff / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        eprintln!(
            "[gemm-q4_0] shape=[{},{},{}] max_abs={:.4e} max_rel={:.4e}",
            m, n, k, max_abs, max_rel
        );
        assert!(
            max_abs < 5e-2,
            "Q4_0 GEMM max_abs {:.4e} exceeds tolerance (f16 scale)",
            max_abs
        );
    }

    /// Boundary test: `m` not a multiple of BM=64. Exercises the store-loop
    /// row boundary guard and the tile-load zero-padding for OOB rows.
    #[test]
    fn test_matmul_gemm_q4_0_unaligned_m() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        {
            let kernels = unsafe { &*backend.kernels.get() };
            if kernels.kernel_mul_mm_q4_0_f32.is_none() {
                eprintln!("[SKIPPED] mul_mm_q4_0_f32_l4_lm kernel not available");
                return;
            }
        }

        let m = 33usize; // >= 32 gate, not aligned to BM=64 or BN=64
        let n = 96usize; // not aligned to 64
        let k = 128usize;

        let (w_tensor, blocks) = make_q4_0_weight_tensor(&backend, n, k, 7);
        let (a_tensor, a_data) = make_f32_activation_tensor(&backend, m, k);
        let mut out_tensor = make_f32_output_tensor(&backend, m, n);

        backend
            .matmul_q4_0(&a_tensor, &w_tensor, &mut out_tensor)
            .expect("matmul_q4_0 GEMM path (unaligned m)");
        backend.queue.finish().unwrap();

        let got = read_f32_output(&backend, &out_tensor);
        let expected = reference_gemm_q4_0(&blocks, &a_data, m, n, k);

        let mut max_abs = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            let diff = (g - e).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
        eprintln!(
            "[gemm-q4_0] unaligned shape=[{},{},{}] max_abs={:.4e}",
            m, n, k, max_abs
        );
        assert!(
            max_abs < 5e-2,
            "Q4_0 GEMM (unaligned m) max_abs {:.4e} exceeds tolerance",
            max_abs
        );
    }

    /// Realistic prefill shape (Qwen 2.5 1.5B FFN gate: n=8960, k=1536, m=256).
    /// Scaled down to keep host test fast but still tile-heavy.
    #[test]
    fn test_matmul_gemm_q4_0_prefill_shape() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        {
            let kernels = unsafe { &*backend.kernels.get() };
            if kernels.kernel_mul_mm_q4_0_f32.is_none() {
                eprintln!("[SKIPPED] mul_mm_q4_0_f32_l4_lm kernel not available");
                return;
            }
        }

        // Downscaled Qwen-like FFN projection: m (seq) = 128, n (out) = 768,
        // k (in) = 1024. k % QK4_0 = 0, n/k multiples of BM/BN.
        let m = 128usize;
        let n = 768usize;
        let k = 1024usize;

        let (w_tensor, blocks) = make_q4_0_weight_tensor(&backend, n, k, 99);
        let (a_tensor, a_data) = make_f32_activation_tensor(&backend, m, k);
        let mut out_tensor = make_f32_output_tensor(&backend, m, n);

        backend
            .matmul_q4_0(&a_tensor, &w_tensor, &mut out_tensor)
            .expect("matmul_q4_0 GEMM path (prefill-shape)");
        backend.queue.finish().unwrap();

        let got = read_f32_output(&backend, &out_tensor);
        let expected = reference_gemm_q4_0(&blocks, &a_data, m, n, k);

        // Larger k inflates absolute accumulated FP error but relative error
        // remains tight (F16 scales → ~1e-3 relative per term).
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            let diff = (g - e).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            let denom = e.abs().max(1e-4);
            let rel = diff / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        eprintln!(
            "[gemm-q4_0] prefill shape=[{},{},{}] max_abs={:.4e} max_rel={:.4e}",
            m, n, k, max_abs, max_rel
        );
        // Accumulated error over k=1024 mac's with F16 scales.
        assert!(
            max_abs < 2e-1,
            "Q4_0 GEMM (prefill) max_abs {:.4e} exceeds tolerance",
            max_abs
        );
    }

    /// Regression for verify v2 ISSUE-2 (SetPartitionRatio directive path).
    ///
    /// `map_weights_for_cpu()` replaces a weight's backing `UnifiedBuffer`,
    /// producing a new `cl_mem` pointer. The old SOA registry entry keyed by
    /// the pre-map pointer is then stale, so `lookup_noshuffle_soa()` misses
    /// and `build_plan()` silently falls back to the AOS GEMV path.
    ///
    /// The fix is `clear_noshuffle_soa_registry()` + a re-run of
    /// `prepare_noshuffle_buffers()` on the directive path. This test
    /// exercises the clear half: register under one key, clear, then confirm
    /// the old key misses and a brand-new key (simulating a post-map
    /// `cl_mem`) can be registered and looked up cleanly.
    #[test]
    fn test_clear_noshuffle_soa_registry() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // Build two independent SOA entries by running the same conversion
        // against two distinct source buffers. Each resulting (q_buf, d_buf,
        // q_img) triple owns real `ocl::core::Mem` handles, so the registry
        // entries are fully materialised and indistinguishable from the
        // production `prepare_noshuffle_buffers()` path.
        let ne01 = 4usize;
        let ne00 = 64usize;
        let blocks_per_row = ne00 / QK4_0;
        let num_blocks = ne01 * blocks_per_row;

        let build_entry = |seed: u8| -> (usize, NoshuffleSoaEntry) {
            let mut all_blocks = Vec::new();
            for row in 0..ne01 {
                for k in 0..blocks_per_row {
                    let d = (row + 1) as f32 * 0.1 + k as f32 * 0.01 + seed as f32 * 0.001;
                    let mut quants = [0i8; 32];
                    for (j, q) in quants.iter_mut().enumerate() {
                        *q = ((row * 100 + k * 32 + j + seed as usize) % 15) as i8 - 7;
                    }
                    all_blocks.push(make_q4_0_block(d, &quants));
                }
            }
            let raw_bytes: Vec<u8> = all_blocks.iter().flat_map(|b| b.iter().copied()).collect();
            let src_buf = unsafe {
                ocl::core::create_buffer::<_, u8>(
                    backend.context.as_core(),
                    ocl::core::MEM_READ_WRITE,
                    raw_bytes.len(),
                    None,
                )
                .unwrap()
            };
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &backend.queue,
                    &src_buf,
                    true,
                    0,
                    &raw_bytes,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )
                .unwrap();
            }
            let (q_buf, d_buf, q_img) = backend
                .convert_q4_0_to_noshuffle(&src_buf, num_blocks, ne00, ne01)
                .unwrap();
            let key = src_buf.as_ptr() as usize;
            (
                key,
                NoshuffleSoaEntry {
                    q_buf,
                    d_buf,
                    q_img,
                    ne00,
                    ne01,
                },
            )
        };

        let (key_a, entry_a) = build_entry(0);
        backend.register_noshuffle_soa(key_a, entry_a);
        assert!(
            backend.lookup_noshuffle_soa(key_a).is_some(),
            "entry must be visible immediately after register"
        );

        // Simulate the directive-path stall: `map_weights_for_cpu()` hands
        // the model a brand-new cl_mem, stranding `key_a`.
        backend.clear_noshuffle_soa_registry();
        assert!(
            backend.lookup_noshuffle_soa(key_a).is_none(),
            "clear_noshuffle_soa_registry() must evict all entries (old key must miss)"
        );

        // Re-registration with the *new* cl_mem key (different from `key_a`)
        // must succeed and be looked up by the plan builder.
        let (key_b, entry_b) = build_entry(1);
        assert_ne!(
            key_a, key_b,
            "test precondition: distinct source buffers produce distinct cl_mem keys"
        );
        backend.register_noshuffle_soa(key_b, entry_b);
        assert!(
            backend.lookup_noshuffle_soa(key_b).is_some(),
            "new entry must be looked up after clear + re-register"
        );
        // And the old key must still miss, i.e. the clear was not silently
        // re-populated during the second register call.
        assert!(
            backend.lookup_noshuffle_soa(key_a).is_none(),
            "old key must remain evicted after re-registration"
        );
    }

    /// `NoshuffleWeightBuffer` must expose `d_buf` through `get_cl_mem()` so
    /// that a weight tensor whose backing buffer was swapped post-SOA
    /// conversion still produces a stable registry key — i.e. the key used
    /// by `matmul_q4_0` (`b_buf.as_ptr() as usize`) resolves to the d_buf
    /// address that `prepare_noshuffle_buffers(keep_original=false)` chose
    /// when registering the entry.
    #[test]
    fn test_noshuffle_weight_buffer_key_stability() {
        use crate::memory::opencl::noshuffle::NoshuffleWeightBuffer;

        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let ne01 = 4usize;
        let ne00 = 64usize;
        let blocks_per_row = ne00 / QK4_0;
        let num_blocks = ne01 * blocks_per_row;

        // Build one SOA conversion output against a dummy Q4_0 source.
        let mut all_blocks = Vec::new();
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let d = (row + 1) as f32 * 0.1 + k as f32 * 0.01;
                let mut quants = [0i8; 32];
                for (j, q) in quants.iter_mut().enumerate() {
                    *q = ((row * 100 + k * 32 + j) % 15) as i8 - 7;
                }
                all_blocks.push(make_q4_0_block(d, &quants));
            }
        }
        let raw_bytes: Vec<u8> = all_blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let src_buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                raw_bytes.len(),
                None,
            )
            .unwrap()
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &src_buf,
                true,
                0,
                &raw_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        let (q_buf, d_buf, q_img) = backend
            .convert_q4_0_to_noshuffle(&src_buf, num_blocks, ne00, ne01)
            .unwrap();

        // Snapshot the d_buf address — this is what `prepare_noshuffle_buffers`
        // would register as the key when swapping in a placeholder.
        let d_buf_key = d_buf.as_ptr() as usize;

        // Register the entry under the d_buf key.
        backend.register_noshuffle_soa(
            d_buf_key,
            NoshuffleSoaEntry {
                q_buf: q_buf.clone(),
                d_buf: d_buf.clone(),
                q_img: q_img.as_ref().cloned(),
                ne00,
                ne01,
            },
        );

        // Swap the AOS source away — the only thing keeping the SOA view
        // alive is now the placeholder + registry entry.
        drop(src_buf);

        // Build the placeholder; it should expose `d_buf` through
        // `get_cl_mem()` so the registry key resolves.
        let placeholder = NoshuffleWeightBuffer::new(q_buf, d_buf, q_img, ne00, ne01, 1024);
        let reported = get_cl_mem(&placeholder).unwrap();
        assert_eq!(
            reported.as_ptr() as usize,
            d_buf_key,
            "placeholder must return d_buf from cl_mem() — registry key stability invariant"
        );
        assert!(
            backend.lookup_noshuffle_soa(d_buf_key).is_some(),
            "entry keyed by d_buf address must remain reachable via placeholder"
        );

        backend.clear_noshuffle_soa_registry();
    }

    /// Phase 3.7a — ENG-ALG-222 / INV-131. After a Q4_0 weight swap the
    /// `Backend::ensure_noshuffle_soa_registered()` trait entry point must
    /// (a) clear the registry of any stale entry keyed by the old cl_mem
    /// (Phase 3.6 path, exercised here as the precondition) and (b)
    /// re-register a fresh SOA descriptor against the **new** cl_mem
    /// address. The latter is the new safety net introduced by 3.7a.
    ///
    /// Test scope:
    /// - Build a Q4_0 weight tensor as if it had just been installed by
    ///   `SwapExecutor` (new cl_mem, no prior SOA entry).
    /// - Confirm the registry has no entry yet.
    /// - Call the trait method and confirm a new entry materialises keyed
    ///   on the tensor's cl_mem address.
    /// - Confirm idempotence — a second call replaces the entry without
    ///   panic.
    #[test]
    fn test_swap_soa_reconversion_registers_new_entry() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // Sanity: the cvt kernel must be available for this test to be
        // meaningful. On hosts where it is missing the trait method is a
        // no-op and we cannot exercise the registration contract.
        if unsafe { (*backend.kernels.get()).kernel_cvt_q4_0_noshuffle.is_none() } {
            eprintln!("[SKIPPED] cvt_noshuffle kernel unavailable");
            return;
        }

        // Mimic a freshly-swapped Q4_0 weight tensor. ne01 must be even
        // for the noshuffle GEMV path; this matches the Q/K/V projection
        // shapes in the production swap path.
        let n = 8usize;
        let k = 64usize;
        let (tensor, _blocks) = make_q4_0_weight_tensor(&backend, n, k, 7);

        let key_before = get_cl_mem(tensor.buffer().as_ref()).unwrap().as_ptr() as usize;

        // Precondition: the registry has no entry for this address.
        // Using a fresh tensor + cleared registry mirrors `SwapExecutor`'s
        // ordering (clear → ensure_registered).
        backend.clear_noshuffle_soa_registry();
        assert!(
            backend.lookup_noshuffle_soa(key_before).is_none(),
            "registry must be empty for a brand-new Q4_0 swap-installed tensor"
        );

        // Phase 3.7a entry point.
        let backend_dyn: &dyn Backend = backend.as_ref();
        backend_dyn
            .ensure_noshuffle_soa_registered(&tensor)
            .expect("ensure_noshuffle_soa_registered must succeed for Q4_0 weight tensor");

        let entry = backend
            .lookup_noshuffle_soa(key_before)
            .expect("INV-131: SOA descriptor must be registered against the new cl_mem");
        assert_eq!(entry.ne00, k);
        assert_eq!(entry.ne01, n);

        // Idempotence — calling the safety net twice must not panic; the
        // second call replaces the entry. (HashMap::insert overwrites.)
        backend_dyn
            .ensure_noshuffle_soa_registered(&tensor)
            .expect("ensure_noshuffle_soa_registered must be idempotent");
        assert!(
            backend.lookup_noshuffle_soa(key_before).is_some(),
            "entry must remain reachable after a redundant safety-net call"
        );

        // Cleanup so subsequent tests in this module observe a clean slate.
        backend.clear_noshuffle_soa_registry();
    }

    /// INV-131: tensor types that are not Q4_0 (e.g. norms, F16 fallbacks)
    /// must trip the trait method's NoOp branch — the registry should
    /// remain empty and the call must not error.
    #[test]
    fn test_swap_soa_reconversion_non_q4_0_is_noop() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // Build a tiny F32 tensor — no SOA conversion is defined for this
        // dtype, so the trait method must early-return Ok(()).
        let n_elems = 64usize;
        let bytes = n_elems * 4;
        let mem = memory::OpenCLMemory::new(backend.context.clone(), backend.queue.clone(), false);
        let buf = mem.alloc(bytes, DType::F32).unwrap();
        let tensor = Tensor::new(
            Shape::new(vec![1, n_elems]),
            buf,
            backend.clone() as Arc<dyn Backend>,
        );

        backend.clear_noshuffle_soa_registry();

        let backend_dyn: &dyn Backend = backend.as_ref();
        backend_dyn
            .ensure_noshuffle_soa_registered(&tensor)
            .expect("non-Q4_0 tensor must be a NoOp, not an error");

        // Registry must remain empty — non-Q4_0 tensors never produce
        // a SOA entry.
        let key = get_cl_mem(tensor.buffer().as_ref()).unwrap().as_ptr() as usize;
        assert!(
            backend.lookup_noshuffle_soa(key).is_none(),
            "non-Q4_0 tensor must not populate the noshuffle SOA registry"
        );
    }

    /// WSWAP-5-AUF-PLACEHOLDER-DROP: `alloc_pre_converted_soa_tensor`
    /// must return a tensor whose backing is a `NoshuffleWeightBuffer`,
    /// register a SOA entry keyed on the `d_buf` cl_mem, and **not**
    /// allocate any AOS placeholder cl_mem.
    #[test]
    fn test_alloc_pre_converted_soa_tensor_no_placeholder() {
        use crate::memory::opencl::noshuffle::NoshuffleWeightBuffer;

        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        if unsafe { (*backend.kernels.get()).kernel_cvt_q4_0_noshuffle.is_none() } {
            eprintln!("[SKIPPED] cvt_noshuffle kernel unavailable");
            return;
        }

        // Build a synthetic SOA payload by running the production
        // `convert_q4_0_to_noshuffle` over a known AOS source. This bypasses
        // the AUF I/O surface — what we exercise here is the registration +
        // tensor-shape contract.
        let n = 8usize;
        let k = 64usize;
        let blocks_per_row = k / QK4_0;
        let num_blocks = n * blocks_per_row;

        let (src_tensor, _blocks) = make_q4_0_weight_tensor(&backend, n, k, 11);
        let src_mem = get_cl_mem(src_tensor.buffer().as_ref()).unwrap();
        let (q_buf_ref, d_buf_ref, _q_img_ref) = backend
            .convert_q4_0_to_noshuffle(src_mem, num_blocks, k, n)
            .expect("reference SOA conversion must succeed");

        // Read the SOA bytes back so we can feed them into
        // `alloc_pre_converted_soa_tensor` as if they came from an AUF
        // payload. This mirrors `auf_tool::build_variant_payload` output.
        let q_size = num_blocks * 16;
        let d_size = num_blocks * 2;
        let mut q_bytes = vec![0u8; q_size];
        let mut d_bytes = vec![0u8; d_size];
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &q_buf_ref,
                true,
                0,
                &mut q_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &d_buf_ref,
                true,
                0,
                &mut d_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        backend.clear_noshuffle_soa_registry();

        // Sprint C entry point.
        let backend_dyn: &dyn Backend = backend.as_ref();
        let tensor = backend_dyn
            .alloc_pre_converted_soa_tensor(Shape::new(vec![n, k]), &q_bytes, &d_bytes, k, n)
            .expect("alloc_pre_converted_soa_tensor must succeed")
            .expect("driver-side cvt program is available, expected Some");

        // (1) The returned tensor must be backed by `NoshuffleWeightBuffer`.
        let nb = tensor
            .buffer()
            .as_any()
            .downcast_ref::<NoshuffleWeightBuffer>()
            .expect(
                "Sprint C contract: alloc_pre_converted_soa_tensor must back the tensor with \
                 NoshuffleWeightBuffer (no AOS placeholder cl_mem)",
            );
        assert_eq!(
            nb.ne00(),
            k,
            "NoshuffleWeightBuffer.ne00 must match logical K"
        );
        assert_eq!(
            nb.ne01(),
            n,
            "NoshuffleWeightBuffer.ne01 must match logical M"
        );
        assert_eq!(
            tensor.dtype(),
            DType::Q4_0,
            "weight tensor must report Q4_0 dtype to keep matmul dispatch unchanged"
        );
        // Logical AOS bytes for the size report (num_blocks * 18).
        assert_eq!(
            tensor.size(),
            num_blocks * 18,
            "Tensor::size() must report the logical AOS footprint"
        );

        // (2) Registry key is the `d_buf` address (matches
        // `prepare_noshuffle_buffers(swap_to_placeholder=true)`).
        let d_buf_key = nb.d_buf().as_ptr() as usize;
        let entry = backend
            .lookup_noshuffle_soa(d_buf_key)
            .expect("Sprint C: registry must be keyed on d_buf cl_mem address");
        assert_eq!(entry.ne00, k);
        assert_eq!(entry.ne01, n);

        // (3) `get_cl_mem(buffer)` must resolve to the d_buf — same key as
        // the registry lookup performed by `matmul_q4_0`.
        let lookup_key = get_cl_mem(tensor.buffer().as_ref()).unwrap().as_ptr() as usize;
        assert_eq!(
            lookup_key, d_buf_key,
            "Tensor::buffer().cl_mem() must equal the registered d_buf key"
        );

        // (4) `restore_pre_converted_soa_registration` after a clear must
        // re-insert the entry against the same d_buf key (Stage (d)
        // re-registration after the per-batch invalidate).
        backend.clear_noshuffle_soa_registry();
        assert!(
            backend.lookup_noshuffle_soa(d_buf_key).is_none(),
            "registry must be empty after clear"
        );
        backend_dyn
            .restore_pre_converted_soa_registration(&tensor)
            .expect("restore_pre_converted_soa_registration must re-register the SOA entry");
        let entry2 = backend
            .lookup_noshuffle_soa(d_buf_key)
            .expect("Sprint C: restore must re-register against d_buf key");
        assert_eq!(entry2.ne00, k);
        assert_eq!(entry2.ne01, n);

        backend.clear_noshuffle_soa_registry();
    }

    /// WSWAP-5-AUF-PLACEHOLDER-DROP: `restore_pre_converted_soa_registration`
    /// must fall through to `ensure_noshuffle_soa_registered` for tensors
    /// whose backing is *not* a `NoshuffleWeightBuffer` (GGUF AOS path).
    #[test]
    fn test_restore_pre_converted_soa_registration_aos_fallthrough() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        if unsafe { (*backend.kernels.get()).kernel_cvt_q4_0_noshuffle.is_none() } {
            eprintln!("[SKIPPED] cvt_noshuffle kernel unavailable");
            return;
        }

        // GGUF-style AOS Q4_0 weight tensor (no NoshuffleWeightBuffer).
        let n = 8usize;
        let k = 64usize;
        let (tensor, _blocks) = make_q4_0_weight_tensor(&backend, n, k, 23);
        let aos_key = get_cl_mem(tensor.buffer().as_ref()).unwrap().as_ptr() as usize;

        backend.clear_noshuffle_soa_registry();
        assert!(
            backend.lookup_noshuffle_soa(aos_key).is_none(),
            "registry must be empty before fallthrough exercise"
        );

        let backend_dyn: &dyn Backend = backend.as_ref();
        backend_dyn
            .restore_pre_converted_soa_registration(&tensor)
            .expect("restore must succeed by falling through to ensure_noshuffle_soa_registered");

        // ensure_noshuffle_soa_registered keys on the AOS cl_mem ptr.
        assert!(
            backend.lookup_noshuffle_soa(aos_key).is_some(),
            "GGUF AOS fallthrough must register a SOA entry against the AOS cl_mem key"
        );

        backend.clear_noshuffle_soa_registry();
    }
}

// ============================================================
// Tests: kv_scatter_f32_to_f16_batch
// ============================================================
#[cfg(test)]
mod kv_scatter_batch_tests {
    use super::*;
    use crate::shape::Shape;
    use half::f16;

    fn try_create_backend() -> Option<Arc<OpenCLBackend>> {
        OpenCLBackend::new().ok().map(Arc::new)
    }

    fn make_f32_src(backend: &Arc<OpenCLBackend>, data: &[f32], shape: Vec<usize>) -> Tensor {
        let byte_len = data.len() * 4;
        let mem = memory::OpenCLMemory::new(backend.context.clone(), backend.queue.clone(), false);
        let buf = mem.alloc(byte_len, DType::F32).unwrap();
        let tensor = Tensor::new(Shape::new(shape), buf, backend.clone());
        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        tensor
    }

    fn make_f16_zero_dst(
        backend: &Arc<OpenCLBackend>,
        n_elems: usize,
        shape: Vec<usize>,
    ) -> Tensor {
        let byte_len = n_elems * 2;
        let zeros: Vec<f16> = vec![f16::from_f32(0.0); n_elems];
        let mem = memory::OpenCLMemory::new(backend.context.clone(), backend.queue.clone(), false);
        let buf = mem.alloc(byte_len, DType::F16).unwrap();
        let tensor = Tensor::new(Shape::new(shape), buf, backend.clone());
        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(zeros.as_ptr() as *const u8, byte_len) };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        tensor
    }

    fn read_f16_tensor(backend: &Arc<OpenCLBackend>, tensor: &Tensor) -> Vec<f16> {
        let n = tensor.buffer().size() / 2;
        let mut result = vec![f16::from_f32(0.0); n];
        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n * 2),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        result
    }

    /// Build src F32 tensor layout `[seq_len, kv_heads, head_dim]` with a
    /// deterministic but per-position-distinct pattern so that offset bugs
    /// (swapped axes, wrong stride) fail loudly.
    fn make_pattern(seq_len: usize, kv_heads: usize, head_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * kv_heads * head_dim];
        for s in 0..seq_len {
            for h in 0..kv_heads {
                for d in 0..head_dim {
                    let idx = (s * kv_heads + h) * head_dim + d;
                    // Unique per (s, h, d) — fits in F16 range (±2^15) for reasonable sizes.
                    out[idx] = (s * 1000 + h * 100 + d) as f32 * 0.001;
                }
            }
        }
        out
    }

    /// Run batch scatter and verify against CPU reference.
    fn run_case(
        backend: &Arc<OpenCLBackend>,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        seq_len: usize,
        dst_pos: usize,
    ) {
        assert!(dst_pos + seq_len <= capacity, "case over-runs capacity");
        let n_dst = kv_heads * capacity * head_dim;

        let k_pattern = make_pattern(seq_len, kv_heads, head_dim);
        let mut v_pattern = make_pattern(seq_len, kv_heads, head_dim);
        // Offset V so K/V paths aren't trivially identical.
        for v in v_pattern.iter_mut() {
            *v += 0.5;
        }

        let k_src = make_f32_src(backend, &k_pattern, vec![1, seq_len, kv_heads * head_dim]);
        let v_src = make_f32_src(backend, &v_pattern, vec![1, seq_len, kv_heads * head_dim]);
        let mut k_dst = make_f16_zero_dst(backend, n_dst, vec![1, kv_heads, capacity, head_dim]);
        let mut v_dst = make_f16_zero_dst(backend, n_dst, vec![1, kv_heads, capacity, head_dim]);

        backend
            .kv_scatter_f32_to_f16_batch(
                &k_src, &v_src, &mut k_dst, &mut v_dst, kv_heads, head_dim, capacity, dst_pos,
                seq_len,
            )
            .unwrap();
        backend.synchronize().unwrap();

        let k_out = read_f16_tensor(backend, &k_dst);
        let v_out = read_f16_tensor(backend, &v_dst);

        assert_eq!(k_out.len(), n_dst);
        assert_eq!(v_out.len(), n_dst);

        // Verify: each (s, h, d) in src lands at HeadMajor offset
        //   h * capacity * head_dim + (dst_pos + s) * head_dim + d
        // Unwritten positions must remain zero.
        let mut max_abs_k: f32 = 0.0;
        let mut max_abs_v: f32 = 0.0;
        for h in 0..kv_heads {
            for p in 0..capacity {
                for d in 0..head_dim {
                    let off = h * capacity * head_dim + p * head_dim + d;
                    let k_val = k_out[off].to_f32();
                    let v_val = v_out[off].to_f32();
                    if p < dst_pos || p >= dst_pos + seq_len {
                        // Unwritten region must remain zero.
                        assert_eq!(
                            k_val, 0.0,
                            "k_dst untouched region non-zero at h={h} p={p} d={d}"
                        );
                        assert_eq!(
                            v_val, 0.0,
                            "v_dst untouched region non-zero at h={h} p={p} d={d}"
                        );
                    } else {
                        let s = p - dst_pos;
                        let src_idx = (s * kv_heads + h) * head_dim + d;
                        let k_exp = k_pattern[src_idx];
                        let v_exp = v_pattern[src_idx];
                        max_abs_k = max_abs_k.max((k_val - k_exp).abs());
                        max_abs_v = max_abs_v.max((v_val - v_exp).abs());
                    }
                }
            }
        }
        // F16 has ~11-bit mantissa → relative precision ~2^-10 ≈ 1e-3.
        // Scale tolerance by magnitude of the pattern (values up to seq*1.0 + heads*0.1 + dim*0.001).
        let max_mag = (seq_len as f32) + (kv_heads as f32) * 0.1 + (head_dim as f32) * 0.001;
        let tol = (max_mag * 1e-3).max(1e-3);
        assert!(
            max_abs_k < tol,
            "K max_abs {max_abs_k:.4e} exceeds F16 tol {tol:.4e} (max_mag={max_mag:.2})"
        );
        assert!(
            max_abs_v < tol,
            "V max_abs {max_abs_v:.4e} exceeds F16 tol {tol:.4e} (max_mag={max_mag:.2})"
        );
    }

    #[test]
    fn test_kv_scatter_batch_seq1_qwen() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        run_case(&backend, 2, 128, 512, 1, 0);
    }

    #[test]
    fn test_kv_scatter_batch_seq4_qwen() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        run_case(&backend, 2, 128, 512, 4, 0);
    }

    #[test]
    fn test_kv_scatter_batch_seq16_llama() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        run_case(&backend, 8, 64, 512, 16, 0);
    }

    #[test]
    fn test_kv_scatter_batch_seq256_qwen_prefill() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        run_case(&backend, 2, 128, 512, 256, 0);
    }

    #[test]
    fn test_kv_scatter_batch_append_offset() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        // Decode-after-prefill style: write positions 256..260.
        run_case(&backend, 2, 128, 512, 4, 256);
    }

    #[test]
    fn test_kv_scatter_batch_boundary() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        // Last-seq_len positions: dst_pos = capacity - seq_len.
        run_case(&backend, 8, 64, 512, 16, 512 - 16);
    }

    #[test]
    fn test_kv_scatter_batch_matches_single_scatter() {
        // Parity check: running seq_len=1 batch call vs seq_len iterations of
        // the single-position kernel must produce identical F16 output.
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };
        let kv_heads = 2usize;
        let head_dim = 128usize;
        let capacity = 256usize;
        let seq_len = 8usize;

        let k_pat = make_pattern(seq_len, kv_heads, head_dim);
        let v_pat: Vec<f32> = k_pat.iter().map(|x| x + 0.25).collect();

        // Batch path.
        let k_src_b = make_f32_src(&backend, &k_pat, vec![1, seq_len, kv_heads * head_dim]);
        let v_src_b = make_f32_src(&backend, &v_pat, vec![1, seq_len, kv_heads * head_dim]);
        let mut k_dst_b = make_f16_zero_dst(
            &backend,
            kv_heads * capacity * head_dim,
            vec![1, kv_heads, capacity, head_dim],
        );
        let mut v_dst_b = make_f16_zero_dst(
            &backend,
            kv_heads * capacity * head_dim,
            vec![1, kv_heads, capacity, head_dim],
        );
        backend
            .kv_scatter_f32_to_f16_batch(
                &k_src_b,
                &v_src_b,
                &mut k_dst_b,
                &mut v_dst_b,
                kv_heads,
                head_dim,
                capacity,
                0,
                seq_len,
            )
            .unwrap();
        backend.synchronize().unwrap();

        // Reference: seq_len calls to single-position scatter.
        let mut k_dst_r = make_f16_zero_dst(
            &backend,
            kv_heads * capacity * head_dim,
            vec![1, kv_heads, capacity, head_dim],
        );
        let mut v_dst_r = make_f16_zero_dst(
            &backend,
            kv_heads * capacity * head_dim,
            vec![1, kv_heads, capacity, head_dim],
        );
        for s in 0..seq_len {
            let off = s * kv_heads * head_dim;
            let k_slice = &k_pat[off..off + kv_heads * head_dim];
            let v_slice = &v_pat[off..off + kv_heads * head_dim];
            let k_s_s = make_f32_src(&backend, k_slice, vec![1, 1, kv_heads * head_dim]);
            let v_s_s = make_f32_src(&backend, v_slice, vec![1, 1, kv_heads * head_dim]);
            backend
                .kv_scatter_f32_to_f16(
                    &k_s_s,
                    &v_s_s,
                    &mut k_dst_r,
                    &mut v_dst_r,
                    head_dim,
                    capacity,
                    s,
                )
                .unwrap();
        }
        backend.synchronize().unwrap();

        let kb = read_f16_tensor(&backend, &k_dst_b);
        let vb = read_f16_tensor(&backend, &v_dst_b);
        let kr = read_f16_tensor(&backend, &k_dst_r);
        let vr = read_f16_tensor(&backend, &v_dst_r);
        assert_eq!(kb, kr, "batch vs single-scatter K mismatch");
        assert_eq!(vb, vr, "batch vs single-scatter V mismatch");
    }
}
