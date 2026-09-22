//! Pre-bound kernel execution plan for GPU decode.
//!
//! Eliminates per-dispatch overhead by pre-binding all static kernel arguments
//! (weights, workspace buffers, dimensions) at plan build time. During execution,
//! only dynamic scalars (start_pos, cache_seq_len, write_pos) are updated.
//!
//! This mirrors llama.cpp's tight enqueue loop where kernel arguments rarely change
//! between tokens, achieving near-zero CPU overhead per dispatch.

use anyhow::{Context, Result};
use ocl::core::Kernel as CoreKernel;
use ocl::core::Mem;
use std::sync::Arc;

use crate::backend::Backend;
use crate::layers::tensor_partition::PartitionContext;
use crate::partition_workspace::PartitionWsCell;

pub use super::tp_plan::{AttnPartitionStep, FfnPartitionStep, TpHostWeights};

thread_local! {
    /// LLMRS_OP_TRACE: per-token wall-clock accumulator (label -> microseconds).
    /// Activated by Plan::execute; dispatch_step reads/writes when Some(_).
    /// Labels are split into `{op}@enqueue` (CPU submission) and `{op}@gpu`
    /// (clFinish wait, approximates pure GPU execution) to isolate where slope
    /// against n_kv originates.
    static OP_TRACE_ACC: std::cell::RefCell<Option<std::collections::HashMap<String, u64>>> =
        const { std::cell::RefCell::new(None) };
}

/// Operation tag for profiling — maps to OpProfiler fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpTag {
    RmsNorm,
    MatmulQKV,
    Rope,
    KvScatter,
    Attention,
    MatmulWo,
    AddRmsNorm,
    AddAssign,
    MatmulGateUp,
    SiluMul,
    MatmulDown,
    FinalNorm,
    LmHead,
}

impl OpTag {
    /// Static label used by the event-based profiler (`--profile-events`).
    ///
    /// These names match the keys produced by the non-plan path (forward_gen
    /// via `OpenCLBackend::set_op_label`), so the resulting aggregate is
    /// self-consistent whether plan execution or the generic dispatch path
    /// was used. Must stay in sync with the label matrix used by the
    /// decode microbench profiling path.
    pub fn profile_label(&self) -> &'static str {
        match self {
            OpTag::RmsNorm => "rms_norm",
            OpTag::AddRmsNorm => "rms_norm",
            OpTag::FinalNorm => "rms_norm",
            OpTag::MatmulQKV => "matmul_qkv",
            OpTag::MatmulWo => "matmul_wo",
            OpTag::MatmulGateUp => "matmul_ffn",
            OpTag::MatmulDown => "matmul_ffn",
            OpTag::Rope => "rope",
            OpTag::KvScatter => "kv_update",
            OpTag::Attention => "attention",
            OpTag::AddAssign => "add_assign",
            OpTag::SiluMul => "silu_mul",
            OpTag::LmHead => "lm_head",
        }
    }
}

/// Dynamic argument that changes per token.
#[derive(Debug, Clone)]
pub enum DynamicArg {
    /// RoPE start position (i32)
    StartPos { arg_idx: u32 },
    /// Attention cache sequence length (i32)
    CacheSeqLen { arg_idx: u32 },
    /// KV scatter write position (i32)
    WritePos { arg_idx: u32 },
    /// KV cache capacity — changes on resize (i32)
    KvCapacity { arg_idx: u32 },
    // quant-window-specific dynamic args
    /// Residual write position (i32) — kivi_gather_update res_pos arg
    ResPos { arg_idx: u32 },
    /// Number of quantized tokens (i32) — kivi attention q2_tokens arg
    Q2Tokens { arg_idx: u32 },
    /// Number of valid residual tokens (i32) — kivi attention res_tokens arg
    ResTokens { arg_idx: u32 },
    /// Tok base offset for scatter (i32) — q2_tokens passed to scatter
    TokBase { arg_idx: u32 },
}

/// A single GPU kernel dispatch with pre-bound arguments.
pub struct KernelStep {
    /// Dedicated kernel object (cloned per step, args pre-bound)
    pub kernel: CoreKernel,
    /// Number of work dimensions (1, 2, or 3)
    pub ndim: u32,
    /// Global work size
    pub global_work_size: [usize; 3],
    /// Local work size (None = driver picks)
    pub local_work_size: Option<[usize; 3]>,
    /// Arguments that must be updated per token
    pub dynamic_args: Vec<DynamicArg>,
    /// Operation tag for profiling
    pub op_tag: OpTag,
    /// Buffers created during plan build that must be kept alive while the plan
    /// exists. Dropping them would invalidate the cl_mem handles pre-bound to
    /// the kernel, leading to SIGSEGV on dispatch.
    #[allow(dead_code)]
    pub retained_bufs: Vec<Mem>,
    /// Activation image1d_buffer_t recreation hint for the Q4_0 noshuffle GEMV.
    ///
    /// The noshuffle GEMV kernel reads its activation through an
    /// `image1d_buffer_t` (RGBA32F) wrapping a regular `cl_mem` buffer. On
    /// Adreno 830 (driver v47, OpenCL 3.0), an image created once at plan
    /// build time and reused across dispatches reads **stale** texels when
    /// the backing buffer is written by a kernel in the same queue between
    /// dispatches (the buffer path is coherent; the image-backed-by-buffer
    /// path is not). The non-plan `matmul_q4_0_noshuffle` path (mod.rs)
    /// dodges this by recreating the image every call.
    ///
    /// When `Some((src_buf, ne00, arg_idx))`, `FullKernelPlan::dispatch_step`
    /// allocates a fresh image wrapping `src_buf` of width `ne00/4`, sets it
    /// on argument `arg_idx`, and releases the ephemeral image after the
    /// enqueue returns (OpenCL refcounting keeps it alive until the driver
    /// consumes it). Static binding is still used for the nibble image and
    /// scale buffer (they never change between dispatches).
    ///
    /// Currently never set to `Some` — the per-layer noshuffle GEMV images
    /// do observe preceding buffer writes correctly on Adreno 830, and the
    /// `lm_head` F32 tied-embedding case (which originally motivated this
    /// hook) is now handled by the dtype-gated plan builder. Kept as a
    /// documented escape hatch for future coherency regressions.
    #[allow(dead_code)]
    pub noshuffle_act_rebuild: Option<(Mem, usize, u32)>,
}

// SAFETY: CoreKernel is a raw cl_kernel pointer. We guarantee single-threaded access
// during plan execution (same safety model as the UnsafeCell<KernelCache> in mod.rs).
unsafe impl Send for KernelStep {}
unsafe impl Sync for KernelStep {}

/// KV update variant — Standard scatter.
pub enum KvUpdateVariant {
    /// Standard F16 scatter: k,v → kv_cache
    Standard(KernelStep),
}

/// Attention variant — Standard half-precision or quant-window fused attention.
#[allow(clippy::large_enum_variant)]
pub enum AttentionVariant {
    /// Legacy kernel_attn_gen_half — supports score writes, any head_dim.
    Standard(KernelStep),
    /// Decode flash attention (flash_attn_f32_f16_q1). Single-pass online
    /// softmax, no score output. Selected at plan-build time when
    /// head_dim==64, F16 KV, HeadMajor, and no scores are needed.
    StandardFlash(KernelStep),
    /// Lanes-across-row decode flash attention (ticket 019): `main` is
    /// `flash_attn_f32_f16_q1_split` over `[64, n_heads_q, n_splits]`, `merge`
    /// is `flash_attn_q1_merge` over `[head_dim, n_heads_q]` (dispatched only
    /// when `n_splits > 1`; a single split finalises inside `main`). Same
    /// inputs and outputs (O, scores, ragged `kv_start`) as `StandardFlash`.
    SplitFlash { main: KernelStep, merge: KernelStep },
}

/// Number of KV splits for the lanes-across-row decode kernel (ticket 019).
/// `LLMRS_Q1_SPLITS` overrides the default so one binary can sweep it in an
/// order-interleaved on-device batch.
pub fn q1_splits() -> usize {
    static SPLITS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SPLITS.get_or_init(|| {
        std::env::var("LLMRS_Q1_SPLITS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(Q1_SPLITS_DEFAULT)
            .min(Q1_SPLITS_MAX)
    })
}

/// Default split count. S25 sweep 2026-09-18 (ticket 019, N = 128 / 2048 /
/// 4096 / 7168): 2 was best or tied at every length (55.4 / 60.5 / 66.2 /
/// 71.6-76.3 ms vs 55.5 / 66.0 / 78.5 / 101.9 ms for 1); 4 and 8 were level
/// with 2, 16 slightly worse.
pub const Q1_SPLITS_DEFAULT: usize = 2;

/// Which decode kernel the plan and runtime paths use. Default: the
/// lanes-across-row kernel (`flash_attn_f32_f16_q1_split`). `LLMRS_Q1_KERNEL=q1`
/// forces the original `flash_attn_f32_f16_q1` — the control arm of every
/// ticket 019 A/B.
pub fn q1_use_split_pair() -> bool {
    static KERNEL: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    !matches!(
        KERNEL
            .get_or_init(|| std::env::var("LLMRS_Q1_KERNEL").ok())
            .as_deref(),
        Some("q1")
    )
}

/// Upper bound on `LLMRS_Q1_SPLITS`: the merge kernel loops over the splits
/// and the scratch buffers scale with it.
pub const Q1_SPLITS_MAX: usize = 128;

/// Execution plan for a single transformer layer.
pub struct LayerKernelPlan {
    /// Steps 1-10: attention block, GPU-only or split with the CPU (ticket 021).
    pub attn: AttnVariant,
    /// Step 11-14: FFN — GPU-only (gate/up/silu/down) or split with the CPU.
    pub ffn: FfnVariant,
    /// Step 15+: typically `add_assign(x += down)`. Empty for a partitioned layer, whose FFN
    /// step lands `x` itself.
    pub steps_post_ffn: Vec<KernelStep>,
    /// Whether to call clFlush after this layer's steps
    pub flush_after: bool,
}

/// Attention-block execution strategy for a layer.
#[allow(clippy::large_enum_variant)]
pub enum AttnVariant {
    GpuOnly(GpuAttnSteps),
    /// Q heads / Wo columns split between GPU and CPU (`tp_plan`).
    Partitioned(Box<AttnPartitionStep>),
}

/// GPU-only attention block: steps 1-10.
pub struct GpuAttnSteps {
    /// Steps 1-6: RMSNorm, QKV matmul, RoPE Q, RoPE K
    pub steps_pre_kv: Vec<KernelStep>,
    /// Step 7: KV update (Standard scatter)
    pub kv_update: KvUpdateVariant,
    /// Step 8: Attention (Standard or flash)
    pub attention: AttentionVariant,
    /// Steps 9-10: Wo matmul, add+RMSNorm. FFN input (`residual`) is produced
    /// by the last step here.
    pub steps_post_attn_pre_ffn: Vec<KernelStep>,
}

/// FFN execution strategy for a layer.
///
/// `GpuOnly` is the historical plan path (4 KernelSteps dispatched inline).
/// `Partitioned` splits gate/up rows and down columns between GPU and CPU
/// (`tp_plan`, ticket 021).
#[allow(clippy::large_enum_variant)]
pub enum FfnVariant {
    /// Dense GPU FFN: gate + up + silu/gelu + down. `gate` and `up` are
    /// separate matmuls because the builder already runs them sequentially
    /// (no fused gate_up kernel on Adreno today).
    GpuOnly {
        gate: KernelStep,
        up: KernelStep,
        silu_mul: KernelStep,
        down: KernelStep,
    },
    Partitioned(Box<FfnPartitionStep>),
}

/// How a partitioned FFN lands its two partial sums.
pub enum PartitionMerge {
    /// `x += down_gpu + cpu_staging` in one kernel
    /// (`kernel_partition_fused_merge_residual_f4`).
    Fused { fused_step: KernelStep },
}

/// Dispatch shape of the F16 GEMV kernels (`kernel_mul_mat_f16_f32[_l4|_ld]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemvKind {
    /// 4-wave K-split, N_DST=2: local [64,4,1].
    Wave4,
    /// 4-wave K-split, N_DST=4 (large N): local [64,4,1].
    L4,
    /// Subgroup-less fallback, N_DST=4: local [64,1,1].
    Nosub,
}

impl GemvKind {
    /// Kernel `make_f16_matmul_step` picks for an `n`-row GEMV.
    pub fn select(n: usize, l4_available: bool, is_nosub: bool) -> Self {
        if is_nosub {
            GemvKind::Nosub
        } else if l4_available && n > LARGE_N_THRESHOLD {
            GemvKind::L4
        } else {
            GemvKind::Wave4
        }
    }

    /// `(global, local)` work size for `n` output rows, decode (m = 1).
    pub fn work_size(self, n: usize) -> ([usize; 3], [usize; 3]) {
        match self {
            GemvKind::Nosub => ([n.div_ceil(4) * 64, 1, 1], [64, 1, 1]),
            GemvKind::L4 => ([n.div_ceil(4) * 64, 4, 1], [64, 4, 1]),
            GemvKind::Wave4 => ([n.div_ceil(2) * 64, 4, 1], [64, 4, 1]),
        }
    }
}

/// Plan-path geometry snapshot for one KV cache (Phase α-K (3p) ④-a).
///
/// `execute` reads these four scalars **once per layer** via a single lock
/// on the `StandardFormat` wrapper (`plan_geometry()`), replacing the four
/// separate `KVCacheOps` getter calls that `execute<C>` makes against a `&mut C`.
/// Standard caches have no residual / quantized partition, so `res_pos` and
/// `q2_tokens` are always `0`; the fields exist for symmetry with the (deferred)
/// quant-window plan flip.
///
/// **Single-lock-snapshot contract**: a `PlanGeometry` value MUST be produced
/// under one lock acquisition so the four fields are mutually consistent (matters
/// for quant-window where `current_pos == q2_tokens + res_pos`; irrelevant for standard
/// but the contract is documented here so the quant-window flip preserves it).
pub(crate) struct PlanGeometry {
    pub current_pos: usize,
    pub capacity: usize,
    pub res_pos: usize,
    pub q2_tokens: usize,
    /// Per-KV-head first resident slot (`i32` per head) of a ragged cache
    /// (`KVCache::head_start_device`); `None` binds NULL = every head from slot 0.
    pub head_start: Option<ocl::core::Mem>,
}

/// Execution plan for the full model decode pass.
pub struct FullKernelPlan {
    /// Per-layer plans (indexed by layer number)
    pub layers: Vec<LayerKernelPlan>,
    /// Final RMSNorm step (model.norm)
    pub final_norm: KernelStep,
    /// lm_head matmul step.
    /// `None` when lm_head is kept on CPU (e.g. gemma3's 604 MB tied embedding).
    /// The caller must run CPU-side matmul when this is `None`.
    pub lm_head: Option<KernelStep>,
    /// KV cache capacity at plan creation time (for invalidation check)
    pub kv_capacity: usize,
    /// True when the attention step was bound to the backend's GPU score
    /// accumulator at build time. `execute()` uses this flag to run `end_step`
    /// after the final layer, mirroring the non-plan path in `transformer.rs`;
    /// the caller compares it with the accumulator's live state and rebuilds
    /// the plan on a mismatch.
    pub writes_gpu_scores: bool,
    /// ENG-ALG-219: `TransformerModel::ratio_generation` value captured at
    /// `build_plan` time (Acquire load). `execute()` compares the live counter
    /// against this snapshot; a mismatch means a weight swap occurred after
    /// plan construction and the pre-bound cl_mem handles are potentially
    /// stale. Returns `PlanInvalidated` so the caller can rebuild (INV-129).
    pub ratio_generation_at_build: u64,
    /// Shared global generation counter. Arc clone of
    /// `TransformerModel::ratio_generation` kept alive for the plan's
    /// lifetime — cheap clone, ensures the atomic is not dropped while this
    /// plan is cached.
    pub ratio_generation_counter: Arc<std::sync::atomic::AtomicU64>,
    /// `Some` when the output-perturbation query-row ring is armed: `execute()` copies each
    /// layer's rotated query row into the ring right after the RoPE steps. Set by the caller
    /// after `build_full_plan` ([`FullKernelPlan::set_q_row_copy`]); `None` costs one branch
    /// per layer.
    pub q_row_copy: Option<QRowPlanCopy>,
    /// `Some` when the layers are CPU–GPU partitioned (ticket 021): the shared partition
    /// workspace, whose controller closes each token after the last layer.
    pub tp: Option<Arc<PartitionWsCell>>,
}

/// Per-layer device copy of the step's rotated query row into the output-perturbation ring
/// (`crate::inference::q_rows`). The plan twin of `QRowCapture::capture` for one decode row:
/// layer `l` at absolute position `p` lands in ring row `l * rows + p % rows`.
pub struct QRowPlanCopy {
    /// The decode workspace's query buffer (`ws.q`), rotated in place by the layer's RoPE step.
    pub src: Mem,
    /// `[n_layers][rows][q_dim]` f32 ring.
    pub ring: Mem,
    pub n_layers: usize,
    pub rows: usize,
    /// `q_dim * size_of::<f32>()`.
    pub row_bytes: usize,
    /// Set when a copy failed to enqueue. The caller then leaves the ring's position stamps
    /// alone, so the next read of the ring is refused instead of served with a stale row.
    pub failed: std::sync::atomic::AtomicBool,
}

/// Error indicating the plan's pre-bound arguments are stale.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanInvalidated;

impl std::fmt::Display for PlanInvalidated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernel plan invalidated (KV cache resized)")
    }
}

impl std::error::Error for PlanInvalidated {}

/// INV-120 generation check as a free function — usable from tests without
/// constructing a partition step.
///
/// Returns `Err(PlanInvalidated)` when `counter.load(Acquire) != at_build`.
pub fn check_partition_generation(
    at_build: u64,
    counter: &Arc<std::sync::atomic::AtomicU64>,
) -> std::result::Result<(), PlanInvalidated> {
    use std::sync::atomic::Ordering;
    let live = counter.load(Ordering::Acquire);
    if live != at_build {
        Err(PlanInvalidated)
    } else {
        Ok(())
    }
}

/// ENG-ALG-219 global ratio_generation check as a free function — usable from
/// tests without constructing a full `FullKernelPlan`.
///
/// Compares `at_build` (captured at `build_plan` time from
/// `TransformerModel::ratio_generation`) against the current atomic value.
/// Returns `Err(PlanInvalidated)` when a weight swap has advanced the counter
/// past the captured snapshot, indicating that the plan's pre-bound cl_mem
/// handles may reference stale weight buffers (INV-129).
///
/// Atomic ordering: Acquire load, matching the Release in `SwapExecutor`'s
/// `ratio_generation.fetch_add(1, SeqCst)`.
pub fn check_global_generation(
    at_build: u64,
    counter: &Arc<std::sync::atomic::AtomicU64>,
) -> std::result::Result<(), PlanInvalidated> {
    use std::sync::atomic::Ordering;
    let live = counter.load(Ordering::Acquire);
    if live != at_build {
        Err(PlanInvalidated)
    } else {
        Ok(())
    }
}

impl FullKernelPlan {
    /// Arm the per-layer query-row copy (see [`QRowPlanCopy`]).
    pub fn set_q_row_copy(&mut self, copy: QRowPlanCopy) {
        self.q_row_copy = Some(copy);
    }

    /// Whether this plan copies query rows into the ring.
    pub fn captures_q_rows(&self) -> bool {
        self.q_row_copy.is_some()
    }

    /// False once any query-row copy failed to enqueue.
    pub fn q_row_copy_ok(&self) -> bool {
        self.q_row_copy
            .as_ref()
            .is_none_or(|c| !c.failed.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Dispatch a single kernel step, updating its dynamic args.
    ///
    /// When `backend.profile_events_enabled` is true, the dispatch goes
    /// through `backend.enqueue_kernel_labeled()` so a profiling event is
    /// captured for each kernel. Otherwise, falls back to the legacy raw
    /// `ocl::core::enqueue_kernel` path (zero overhead in the non-profiled
    /// build).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_step(
        backend: &crate::backend::opencl::OpenCLBackend,
        step: &KernelStep,
        start_pos: i32,
        cache_seq_len: i32,
        write_pos: i32,
        kv_capacity: i32,
        res_pos: i32,
        q2_tokens: i32,
        res_tokens: i32,
    ) {
        Self::dispatch_step_gws(
            backend,
            step,
            &step.global_work_size,
            start_pos,
            cache_seq_len,
            write_pos,
            kv_capacity,
            res_pos,
            q2_tokens,
            res_tokens,
        );
    }

    /// [`Self::dispatch_step`] with a per-dispatch global work size (the partitioned steps size
    /// their kernels to the current split).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_step_gws(
        backend: &crate::backend::opencl::OpenCLBackend,
        step: &KernelStep,
        global_work_size: &[usize; 3],
        start_pos: i32,
        cache_seq_len: i32,
        write_pos: i32,
        kv_capacity: i32,
        res_pos: i32,
        q2_tokens: i32,
        res_tokens: i32,
    ) {
        let queue = backend.queue.as_core();

        // Q4_0 noshuffle GEMV: recreate the activation `image1d_buffer_t` per
        // dispatch. Adreno's image cache is NOT coherent with kernel writes to
        // the underlying buffer across dispatches (see doc on `KernelStep
        // ::noshuffle_act_rebuild`). Holding the image alive until after
        // enqueue is sufficient because OpenCL's internal refcount keeps the
        // cl_mem valid until the driver consumes it.
        //
        // Kept outside the `dynamic_args` loop so the lifetime spans the
        // enqueue below.
        let _ephemeral_act_img: Option<Mem> =
            if let Some((ref src_buf, ne00, arg_idx)) = step.noshuffle_act_rebuild {
                use ocl::core::{
                    ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat,
                    MemObjectType,
                };
                let fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
                let desc = ImageDescriptor::new(
                    MemObjectType::Image1dBuffer,
                    ne00 / 4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    Some(src_buf.clone()),
                );
                let img_res = unsafe {
                    ocl::core::create_image(
                        &backend.context,
                        ocl::core::MEM_READ_ONLY,
                        &fmt,
                        &desc,
                        None::<&[f32]>,
                        None,
                    )
                };
                match img_res {
                    Ok(img) => {
                        let arg_res = unsafe {
                            ocl::core::set_kernel_arg(
                                &step.kernel,
                                arg_idx,
                                ocl::core::ArgVal::mem(&img),
                            )
                        };
                        if let Err(e) = arg_res {
                            log::error!(
                                "Plan set noshuffle act image arg failed: op={:?} arg_idx={}: {}",
                                step.op_tag,
                                arg_idx,
                                e
                            );
                        }
                        Some(img)
                    }
                    Err(e) => {
                        log::error!(
                            "Plan create noshuffle act image failed: op={:?}: {}",
                            step.op_tag,
                            e
                        );
                        None
                    }
                }
            } else {
                None
            };
        for dyn_arg in &step.dynamic_args {
            let (arg_idx, result) = unsafe {
                match dyn_arg {
                    DynamicArg::StartPos { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&start_pos),
                        ),
                    ),
                    DynamicArg::CacheSeqLen { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&cache_seq_len),
                        ),
                    ),
                    DynamicArg::WritePos { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&write_pos),
                        ),
                    ),
                    DynamicArg::KvCapacity { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&kv_capacity),
                        ),
                    ),
                    DynamicArg::ResPos { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&res_pos),
                        ),
                    ),
                    DynamicArg::Q2Tokens { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&q2_tokens),
                        ),
                    ),
                    DynamicArg::ResTokens { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&res_tokens),
                        ),
                    ),
                    DynamicArg::TokBase { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&q2_tokens),
                        ),
                    ),
                }
            };
            if let Err(e) = result {
                log::error!(
                    "Plan set_kernel_arg failed: op={:?} arg_idx={}: {}",
                    step.op_tag,
                    arg_idx,
                    e
                );
            }
        }
        let traced = OP_TRACE_ACC.with(|c| c.borrow().is_some());
        if traced {
            ocl::core::finish(queue).ok();
        }
        let t_op_start = std::time::Instant::now();
        if backend.profile_events_enabled {
            if let Err(e) = backend.enqueue_kernel_labeled(
                &step.kernel,
                step.op_tag.profile_label(),
                step.ndim,
                global_work_size,
                step.local_work_size,
            ) {
                log::error!(
                    "Plan enqueue_kernel_labeled failed: op={:?} gws={:?}: {}",
                    step.op_tag,
                    global_work_size,
                    e
                );
            }
        } else {
            unsafe {
                if let Err(e) = ocl::core::enqueue_kernel(
                    queue,
                    &step.kernel,
                    step.ndim,
                    None,
                    global_work_size,
                    step.local_work_size,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                ) {
                    log::error!(
                        "Plan enqueue_kernel failed: op={:?} gws={:?}: {}",
                        step.op_tag,
                        global_work_size,
                        e
                    );
                }
            }
        }
        if traced {
            let t_enqueue_end = std::time::Instant::now();
            ocl::core::finish(queue).ok();
            let t_gpu_end = std::time::Instant::now();
            let enqueue_us = t_enqueue_end.duration_since(t_op_start).as_nanos() as u64 / 1000;
            let gpu_us = t_gpu_end.duration_since(t_enqueue_end).as_nanos() as u64 / 1000;
            let label = step.op_tag.profile_label();
            OP_TRACE_ACC.with(|c| {
                if let Some(m) = c.borrow_mut().as_mut() {
                    *m.entry(format!("{}@enqueue", label)).or_insert(0) += enqueue_us;
                    *m.entry(format!("{}@gpu", label)).or_insert(0) += gpu_us;
                }
            });
        }
    }

    /// (3p) ④-a — `execute<C>` 의 production fmt-handle copy-fork (Phase α-K BC Step 3).
    ///
    /// `execute<C: KVCacheOps>(&mut [C])` 와 본문 byte-identical 이되, C-접촉 지점만
    /// `StandardFormat` concrete-handle(`plan_geometry`/`plan_advance`)로 교체한다 — vtable 0
    /// (static dispatch). 게이트 OFF(`LLMRS_KV_FMT` 미설정) 시 production 은 `execute<C>` 를
    /// 계속 쓰고 본 메서드는 미발화(byte-불변). legacy `execute<C>` 는 Step 5 까지 co-exist.
    ///
    /// plan path 는 GPU 전용이라 host 에서 기능 미발화 — bit-identical/avg_tbt acceptance 는
    /// device 세션(S25 OpenCL + Jetson CUDA).
    pub fn execute(
        &self,
        backend: &crate::backend::opencl::OpenCLBackend,
        start_pos: usize,
        handles: &[std::sync::Arc<crate::kv::standard_format::StandardFormat>],
    ) -> std::result::Result<(), PlanInvalidated> {
        // ENG-ALG-219: single Acquire load at entry — if a weight swap has
        // bumped ratio_generation since build_plan, the pre-bound cl_mem
        // handles may reference stale weight buffers (INV-129).
        check_global_generation(
            self.ratio_generation_at_build,
            &self.ratio_generation_counter,
        )?;

        let debug_sync = std::env::var("PLAN_DEBUG").is_ok();
        let op_trace = std::env::var_os("LLMRS_OP_TRACE").is_some();
        let queue = backend.queue.as_core();

        if op_trace {
            OP_TRACE_ACC.with(|c| {
                *c.borrow_mut() = Some(std::collections::HashMap::new());
            });
        }
        let mut trace_n_kv: i32 = 0;

        for (i, layer_plan) in self.layers.iter().enumerate() {
            let handle = &handles[i];
            // (3p) ④-a: single-lock geometry snapshot replaces the four
            // `KVCacheOps` getters that `execute<C>` reads off `&mut C`.
            let g = handle.plan_geometry();

            // Check for KV cache resize or capacity overflow (plan invalidation)
            if g.capacity != self.kv_capacity || g.current_pos >= g.capacity {
                return Err(PlanInvalidated);
            }

            let cache_seq_len = g.current_pos as i32;
            let write_pos = g.current_pos as i32;
            let start_pos_i32 = start_pos as i32;
            let kv_cap = g.capacity as i32;
            let rp = g.res_pos as i32;
            let q2t = g.q2_tokens as i32;
            let rt = rp; // res_tokens = res_pos before advance

            // Attention sees the token we just scattered
            let attn_seq_len = cache_seq_len + 1;
            if op_trace {
                trace_n_kv = attn_seq_len;
            }

            let a = match &layer_plan.attn {
                AttnVariant::Partitioned(step) => {
                    step.run(
                        backend,
                        start_pos,
                        g.current_pos,
                        kv_cap,
                        g.head_start.is_some(),
                    )?;
                    None
                }
                AttnVariant::GpuOnly(a) => Some(a),
            };
            if let Some(a) = a {
                // Steps 1-6: pre-KV steps
                for (si, step) in a.steps_pre_kv.iter().enumerate() {
                    Self::dispatch_step(
                        backend,
                        step,
                        start_pos_i32,
                        cache_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt,
                    );
                    if debug_sync {
                        ocl::core::finish(queue).ok();
                        eprintln!(
                            "[Plan] L{} pre_kv[{}] {:?} OK (pos={}, cap={})",
                            i, si, step.op_tag, start_pos, kv_cap
                        );
                    }
                }
                // Query-row capture (output-perturbation metric): the RoPE steps above left this
                // layer's rotated query row in `ws.q`; copy it out before the next layer overwrites it.
                if let Some(c) = self.q_row_copy.as_ref()
                    && i < c.n_layers
                {
                    let dst_off = (i * c.rows + start_pos % c.rows) * c.row_bytes;
                    if let Err(e) = unsafe {
                        ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                            queue,
                            &c.src,
                            &c.ring,
                            0,
                            dst_off,
                            c.row_bytes,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )
                    } {
                        log::error!("Plan q-row copy failed: layer={i} pos={start_pos}: {e}");
                        c.failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }

                // Step 7: KV update
                match &a.kv_update {
                    KvUpdateVariant::Standard(step) => {
                        Self::dispatch_step(
                            backend,
                            step,
                            start_pos_i32,
                            cache_seq_len,
                            write_pos,
                            kv_cap,
                            rp,
                            q2t,
                            rt,
                        );
                        if debug_sync {
                            ocl::core::finish(queue).ok();
                            eprintln!(
                                "[Plan] L{} kv_scatter OK (write_pos={}, cap={})",
                                i, write_pos, kv_cap
                            );
                        }
                    }
                }

                // Step 8: Attention — uses attn_seq_len (includes just-scattered token)
                // A ragged cache hands the kernel its per-head first resident slots (NULL = uniform).
                let bind_kv_start = |kernel: &ocl::core::Kernel, arg_idx: u32| {
                    let val = match g.head_start.as_ref() {
                        Some(m) => ocl::core::ArgVal::mem(m),
                        None => ocl::core::ArgVal::mem_null(),
                    };
                    if let Err(e) = unsafe { ocl::core::set_kernel_arg(kernel, arg_idx, val) } {
                        log::error!(
                            "Plan set kv_start arg failed: layer={i} arg_idx={arg_idx}: {e}"
                        );
                    }
                };
                match &a.attention {
                    AttentionVariant::Standard(step) => {
                        bind_kv_start(&step.kernel, 16);
                        if debug_sync {
                            eprintln!(
                                "[Plan] L{} attention dispatch (attn_seq_len={}, gws={:?}, lws={:?})",
                                i, attn_seq_len, step.global_work_size, step.local_work_size
                            );
                        }
                        Self::dispatch_step(
                            backend,
                            step,
                            start_pos_i32,
                            attn_seq_len,
                            write_pos,
                            kv_cap,
                            rp,
                            q2t,
                            rt,
                        );
                        // GPU score accumulator: per-layer scores now live in the
                        // layer's own slice of `score_buf` (offset pre-baked at
                        // plan-build time, see `LayerPlanConfig::gpu_score_layer_offset`).
                        // A single fused reduce kernel folds all layers into
                        // cumulative importance at `end_step()` after the final
                        // layer — no per-layer dispatch needed here.
                        if debug_sync {
                            eprintln!("[Plan] L{} attention enqueued, calling finish...", i);
                            ocl::core::finish(queue).ok();
                            eprintln!("[Plan] L{} attention OK (attn_seq_len={})", i, attn_seq_len);
                        }
                    }
                    AttentionVariant::StandardFlash(step) => {
                        bind_kv_start(&step.kernel, 44);
                        if debug_sync {
                            eprintln!(
                                "[Plan] L{} flash attention dispatch (attn_seq_len={}, gws={:?}, lws={:?})",
                                i, attn_seq_len, step.global_work_size, step.local_work_size
                            );
                        }
                        let trace_q1 = std::env::var_os("LLMRS_TRACE_Q1").is_some();
                        if trace_q1 {
                            ocl::core::finish(queue).ok();
                        }
                        let q1_start = std::time::Instant::now();
                        Self::dispatch_step(
                            backend,
                            step,
                            start_pos_i32,
                            attn_seq_len,
                            write_pos,
                            kv_cap,
                            rp,
                            q2t,
                            rt,
                        );
                        if trace_q1 {
                            ocl::core::finish(queue).ok();
                            let us = q1_start.elapsed().as_nanos() as u64 / 1000;
                            eprintln!("[Q1_TRACE] layer={} n_kv={} us={}", i, attn_seq_len, us);
                        }
                        // LLMRS_Q1_REPEAT=N: re-dispatch the Q1 kernel (N-1) additional
                        // times against the same KV state, measuring each repetition in
                        // isolation. The first (production) iteration follows matmul_qkv
                        // /rope/kv_update and reads KV "cold" from the just-written slot;
                        // subsequent reps read it "warm". Comparing rep=0 vs rep>=1
                        // slope against n_kv separates kernel-intrinsic cost from
                        // context-dependent cache/coherency effects.
                        let q1_repeat: u32 = std::env::var("LLMRS_Q1_REPEAT")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(1);
                        for rep in 1..q1_repeat {
                            ocl::core::finish(queue).ok();
                            let rep_start = std::time::Instant::now();
                            Self::dispatch_step(
                                backend,
                                step,
                                start_pos_i32,
                                attn_seq_len,
                                write_pos,
                                kv_cap,
                                rp,
                                q2t,
                                rt,
                            );
                            ocl::core::finish(queue).ok();
                            let rep_us = rep_start.elapsed().as_nanos() as u64 / 1000;
                            eprintln!(
                                "[Q1_REPEAT] layer={} n_kv={} rep={} us={}",
                                i, attn_seq_len, rep, rep_us
                            );
                        }
                        if debug_sync {
                            ocl::core::finish(queue).ok();
                            eprintln!(
                                "[Plan] L{} flash attention OK (attn_seq_len={})",
                                i, attn_seq_len
                            );
                        }
                    }
                    AttentionVariant::SplitFlash { main, merge } => {
                        bind_kv_start(&main.kernel, Q1_SPLIT_MAIN_KV_START_ARG);
                        bind_kv_start(&merge.kernel, Q1_MERGE_KV_START_ARG);
                        if debug_sync {
                            eprintln!(
                                "[Plan] L{} split flash attention dispatch (attn_seq_len={}, gws={:?}, lws={:?})",
                                i, attn_seq_len, main.global_work_size, main.local_work_size
                            );
                        }
                        // The pair is one attention op for `LLMRS_TRACE_Q1` / `LLMRS_Q1_REPEAT`
                        // (same env knobs as the `StandardFlash` arm) so the two kernels'
                        // numbers are directly comparable.
                        let dispatch_pair = || {
                            Self::dispatch_step(
                                backend,
                                main,
                                start_pos_i32,
                                attn_seq_len,
                                write_pos,
                                kv_cap,
                                rp,
                                q2t,
                                rt,
                            );
                            // A single split finalises inside the main kernel (`final_out`).
                            if main.global_work_size[2] <= 1 {
                                return;
                            }
                            Self::dispatch_step(
                                backend,
                                merge,
                                start_pos_i32,
                                attn_seq_len,
                                write_pos,
                                kv_cap,
                                rp,
                                q2t,
                                rt,
                            );
                        };
                        let trace_q1 = std::env::var_os("LLMRS_TRACE_Q1").is_some();
                        if trace_q1 {
                            ocl::core::finish(queue).ok();
                        }
                        let q1_start = std::time::Instant::now();
                        dispatch_pair();
                        if trace_q1 {
                            ocl::core::finish(queue).ok();
                            let us = q1_start.elapsed().as_nanos() as u64 / 1000;
                            eprintln!(
                                "[Q1_TRACE] layer={} n_kv={} us={} splits={}",
                                i, attn_seq_len, us, main.global_work_size[2]
                            );
                        }
                        let q1_repeat: u32 = std::env::var("LLMRS_Q1_REPEAT")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(1);
                        for rep in 1..q1_repeat {
                            ocl::core::finish(queue).ok();
                            let rep_start = std::time::Instant::now();
                            dispatch_pair();
                            ocl::core::finish(queue).ok();
                            let rep_us = rep_start.elapsed().as_nanos() as u64 / 1000;
                            eprintln!(
                                "[Q1_REPEAT] layer={} n_kv={} rep={} us={}",
                                i, attn_seq_len, rep, rep_us
                            );
                        }
                        if debug_sync {
                            ocl::core::finish(queue).ok();
                            eprintln!(
                                "[Plan] L{} split flash attention OK (attn_seq_len={})",
                                i, attn_seq_len
                            );
                        }
                    }
                }

                // Steps 9-10: post-attention pre-FFN (Wo matmul, add_rms_norm).
                for (si, step) in a.steps_post_attn_pre_ffn.iter().enumerate() {
                    Self::dispatch_step(
                        backend,
                        step,
                        start_pos_i32,
                        cache_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt,
                    );
                    if debug_sync {
                        ocl::core::finish(queue).ok();
                        eprintln!(
                            "[Plan] L{} post_attn_pre_ffn[{}] {:?} OK",
                            i, si, step.op_tag
                        );
                    }
                }
            }

            // Steps 11-14: FFN (GPU-only or cooperative partition).
            let skip_post_ffn = match &layer_plan.ffn {
                FfnVariant::GpuOnly {
                    gate,
                    up,
                    silu_mul,
                    down,
                } => {
                    for step in [gate, up, silu_mul, down] {
                        Self::dispatch_step(
                            backend,
                            step,
                            start_pos_i32,
                            cache_seq_len,
                            write_pos,
                            kv_cap,
                            rp,
                            q2t,
                            rt,
                        );
                        if debug_sync {
                            ocl::core::finish(queue).ok();
                            eprintln!("[Plan] L{} ffn {:?} OK", i, step.op_tag);
                        }
                    }
                    false
                }
                FfnVariant::Partitioned(step) => {
                    step.run(backend)?;
                    if debug_sync {
                        ocl::core::finish(queue).ok();
                        eprintln!("[Plan] L{} partition FFN OK", i);
                    }
                    // The fused merge already landed `x += down_gpu + cpu_partial`.
                    true
                }
            };

            // Step 15: add_assign (x += down).
            if !skip_post_ffn {
                for (si, step) in layer_plan.steps_post_ffn.iter().enumerate() {
                    Self::dispatch_step(
                        backend,
                        step,
                        start_pos_i32,
                        cache_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt,
                    );
                    if debug_sync {
                        ocl::core::finish(queue).ok();
                        eprintln!("[Plan] L{} post_ffn[{}] {:?} OK", i, si, step.op_tag);
                    }
                }
            }
            handle.plan_advance(1);

            if layer_plan.flush_after
                && let Err(e) = ocl::core::flush(queue)
            {
                log::error!("Plan flush failed: {}", e);
            }

            // Intra-token GPU yield hook. Plan path is decode-only, so
            // `is_decode = true` unconditionally.
            backend.yield_after_layer(i, true);
        }

        // GPU score accumulator: flush step-local scores into cumulative
        // importance and clear step buffers. Mirrors transformer.rs:979 for
        // the non-plan path. Uses the post-advance cache position (one past
        // the token just scattered) so `end_step` sees the same length the
        // runtime sees.
        if self.writes_gpu_scores
            && !handles.is_empty()
            && let Some(gpu_acc) = backend.gpu_score_acc_mut()
            && gpu_acc.is_active()
        {
            let cache_seq_len = handles[0].plan_geometry().current_pos;
            if let Err(e) = gpu_acc.end_step(queue, cache_seq_len) {
                log::error!(
                    "Plan gpu_score end_step failed: n_kv={}: {}",
                    cache_seq_len,
                    e
                );
            }
        }

        // Final norm
        if backend.profile_events_enabled {
            if let Err(e) = backend.enqueue_kernel_labeled(
                &self.final_norm.kernel,
                self.final_norm.op_tag.profile_label(),
                self.final_norm.ndim,
                &self.final_norm.global_work_size,
                self.final_norm.local_work_size,
            ) {
                log::error!("Plan enqueue_kernel_labeled final_norm failed: {}", e);
            }
        } else {
            unsafe {
                if let Err(e) = ocl::core::enqueue_kernel(
                    queue,
                    &self.final_norm.kernel,
                    self.final_norm.ndim,
                    None,
                    &self.final_norm.global_work_size,
                    self.final_norm.local_work_size,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                ) {
                    log::error!("Plan enqueue final_norm failed: {}", e);
                }
            }
        }

        // lm_head matmul (skipped when lm_head is on CPU or has an unsupported
        // dtype — see `build_full_plan` for the dtype gating).
        //
        // Route through `dispatch_step` so any future `noshuffle_act_rebuild`
        // hooks can participate. `dynamic_args` is empty here so there is no
        // per-token overhead from the dispatcher wrapping.
        if let Some(ref lm_head) = self.lm_head {
            Self::dispatch_step(backend, lm_head, 0, 0, 0, 0, 0, 0, 0);
        }
        if let Some(tp) = self.tp.as_ref() {
            if let Err(e) = ocl::core::flush(queue) {
                log::error!("Plan flush failed: {}", e);
            }
            super::tp_plan::end_token(tp)?;
        }

        if op_trace {
            OP_TRACE_ACC.with(|c| {
                if let Some(m) = c.borrow_mut().take() {
                    let mut entries: Vec<(String, u64)> = m.into_iter().collect();
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    let parts: Vec<String> = entries
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect();
                    eprintln!("[OP_TRACE] n_kv={} {}", trace_n_kv, parts.join(" "));
                }
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Layer plan builder
// ---------------------------------------------------------------------------

/// All references needed to build a pre-bound kernel plan for one layer.
pub struct LayerPlanConfig<'a> {
    // Context for buffer creation
    pub context: &'a ocl::Context,
    // Programs for kernel creation
    pub f16_program: &'a ocl::Program,
    pub f16_l4_program: Option<&'a ocl::Program>,
    pub simple_ops_program: &'a ocl::Program,
    /// AOS Q4_0 matmul program (`kernel_mul_mat_q4_0_f32`). Used for small-N
    /// matmuls where noshuffle GEMV under-utilizes the GPU (e.g. K/V projection).
    pub q4_0_program: &'a ocl::Program,
    // Buffer handles (cl_mem) — model weights
    pub x_buf: &'a Mem,
    pub wq_buf: &'a Mem,
    pub wk_buf: &'a Mem,
    pub wv_buf: &'a Mem,
    /// Optional QKV bias buffer for Q (F32). When `Some`, the plan
    /// builder appends a `kernel_add_row_bias` step after the Q matmul.
    pub bq_buf: Option<&'a Mem>,
    /// Optional QKV bias buffer for K (F32).
    pub bk_buf: Option<&'a Mem>,
    /// Optional QKV bias buffer for V (F32).
    pub bv_buf: Option<&'a Mem>,
    pub wo_buf: &'a Mem,
    pub w_gate_buf: &'a Mem,
    pub w_up_buf: &'a Mem,
    pub w_down_buf: &'a Mem,
    pub attn_norm_buf: &'a Mem,
    pub ffn_norm_buf: &'a Mem,
    // Workspace buffers
    pub q_buf: &'a Mem,
    pub k_buf: &'a Mem,
    pub v_buf: &'a Mem,
    pub out_attn_buf: &'a Mem,
    pub attn_out_buf: &'a Mem,
    pub gate_buf: &'a Mem,
    pub up_buf: &'a Mem,
    pub down_buf: &'a Mem,
    pub residual_buf: &'a Mem,
    // KV cache buffers
    pub k_cache_buf: &'a Mem,
    pub v_cache_buf: &'a Mem,
    // Dimensions
    pub dim: usize,
    pub n_heads_q: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub n_q: usize,
    pub n_k: usize,
    pub n_v: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// llama3 `rope_scaling` (identity when `factor == 1`) — see [`crate::rope`].
    pub rope_freq_scaling: crate::rope::RopeFreqScaling,
    pub kv_capacity: usize,
    // Attention layout info
    pub kv_pos_stride: i32,
    pub kv_head_stride: i32,
    /// Whether the device lacks subgroup support (nosub fallback path).
    pub is_nosub: bool,
    /// Flash attention F32-Q / F16-KV program handle for head_dim=64.
    /// Used to create `flash_attn_f32_f16_q1` at plan-build time when
    /// runtime preconditions hold and the layer's head_dim is 64.
    /// `None` forces the legacy path for head_dim=64 models.
    pub flash_attn_f32_f16_program_dk64: Option<&'a ocl::Program>,
    /// Flash attention F32-Q / F16-KV program handle for head_dim=128.
    /// Used for Qwen 2.5-1.5B and other head_dim=128 models.
    /// `None` forces the legacy path for head_dim=128 models.
    pub flash_attn_f32_f16_program_dk128: Option<&'a ocl::Program>,
    /// True if this decode plan must capture attention scores (heavy-hitter/heavy-hitter+ or
    /// an active GPU score accumulator). When true, the builder must use
    /// `AttentionVariant::Standard` because the flash kernel has no score
    /// output.
    pub needs_attention_scores: bool,
    /// Persistent GPU score buffer from the backend's `GpuScoreAccumulator`.
    /// When `Some` and `needs_attention_scores` is true, the legacy attention
    /// step is pre-bound to write softmax scores directly into this buffer
    /// (arg 4, `write_scores=1`, `score_stride` below). Per-layer reduction
    /// (`GpuScoreAccumulator::reduce_layer`) is then driven by `FullKernelPlan::execute`.
    pub gpu_score_buf: Option<&'a Mem>,
    /// Score stride (= `max_seq_len`) matching `gpu_score_buf` layout. Unused
    /// when `gpu_score_buf` is `None`.
    pub gpu_score_stride: i32,
    /// Base offset (in f32 elements) for this layer's slice of `gpu_score_buf`.
    /// The score buffer has layout `[n_layers, n_heads_q, score_stride]`, so
    /// layer `l`'s base offset is `l * n_heads_q * score_stride`. This is
    /// pre-baked into the attention kernel's `score_layer_offset` arg by the
    /// layer builder, avoiding per-token arg updates. Unused when
    /// `gpu_score_buf` is `None`.
    pub gpu_score_layer_offset: i32,
    // -- Q4_0 noshuffle matmul support --
    /// Pre-compiled noshuffle GEMV programs, keyed by ne01 (M dimension).
    /// When `Some`, matmul steps use Q4_0 noshuffle dispatch instead of F16.
    pub noshuffle_programs: Option<&'a std::collections::HashMap<usize, ocl::Program>>,
    /// Per-weight noshuffle SOA entries (q_img + d_buf + dimensions).
    /// When `noshuffle_programs` is `Some`, these must also be `Some`.
    pub wq_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wk_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wv_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wo_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_gate_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_up_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_down_noshuffle: Option<NoshufflePlanEntry<'a>>,
    /// Permanent-mapped host pointer for `residual_buf` (null when unmapped). The partitioned
    /// layers' CPU share reads each segment's normed input through it (ticket 021).
    pub residual_host_ptr: *const u8,
}

/// Lightweight reference to a noshuffle SOA entry for plan building.
/// Avoids coupling plan.rs to NoshuffleSoaEntry's full layout.
#[derive(Clone, Copy)]
pub struct NoshufflePlanEntry<'a> {
    /// image1d_buffer_t wrapping SOA nibbles (R32UI)
    pub q_img: &'a Mem,
    /// SOA scales buffer (half2*)
    pub d_buf: &'a Mem,
    /// K dimension (elements per row)
    pub ne00: usize,
    /// M dimension (number of output rows)
    pub ne01: usize,
}

/// N threshold for switching from N_DST=2 (128 rows/WG) to N_DST=4 (256 rows/WG) GEMV kernel.
/// At N > 4096, the L4 kernel halves WG count and reuses activation across 4 rows.
const LARGE_N_THRESHOLD: usize = 4096;

/// Helper: create a dedicated kernel and pre-bind F16 matmul arguments.
///
/// Must mirror `OpenClBackend::matmul_f16` dispatch exactly.
///
/// Three dispatch shapes depending on which kernel is loaded:
/// - 4-wave (default): local=[64,4,1], N_DST=2 → 128 rows/WG,
///   global=[ceil(n/128)*64, 4, 1] for m=1 decode.
/// - 4-wave L4 (large-N): local=[64,4,1], N_DST=4 → 256 rows/WG,
///   global=[ceil(n/256)*64, 4, 1]. Used when l4_program is provided and n > LARGE_N_THRESHOLD.
/// - Nosub fallback: local=[64,1,1], N_DST=4 → 4 rows/WG,
///   global=[ceil(n/4)*64, 1, 1].
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_f16_matmul_step(
    program: &ocl::Program,
    src_buf: &Mem,
    weight_buf: &Mem,
    dst_buf: &Mem,
    n: usize,
    k: usize,
    op_tag: OpTag,
    l4_program: Option<&ocl::Program>,
    is_nosub: bool,
) -> Result<KernelStep> {
    // Prefer L4 (N_DST=4, 4 rows/WG) for large-N when both conditions hold:
    // - 4-wave path is active (nosub fallback has incompatible dispatch)
    // - l4_program compiled successfully
    // - n exceeds the threshold where halving WG count pays off
    let kind = GemvKind::select(n, l4_program.is_some(), is_nosub);
    let use_l4 = kind == GemvKind::L4;
    let kernel = if use_l4 {
        ocl::core::create_kernel(l4_program.unwrap(), "kernel_mul_mat_f16_f32_l4")
            .context("create kernel_mul_mat_f16_f32_l4")?
    } else {
        ocl::core::create_kernel(program, "kernel_mul_mat_f16_f32")
            .context("create kernel_mul_mat_f16_f32")?
    };

    let ne00 = k as i32;
    let ne01 = n as i32;
    let ne02 = 1i32;
    let ne10 = k as i32;
    let ne12 = 1i32;
    let ne0 = n as i32;
    let ne1 = 1i32; // m=1 for decode
    let r2 = 1i32;
    let r3 = 1i32;

    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(weight_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(src_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(dst_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
        ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
        ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
        ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&ne10))?;
        ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
        ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&ne0))?;
        ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&ne1))?;
        ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&r2))?;
        ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&r3))?;
    }

    // Nosub: single-subgroup WG, 4 rows/WG. L4: 4 waves × 64 lanes, N_DST=4.
    // Wave4: 4 waves × 64 lanes, N_DST=2.
    let (global_work_size, local_work_size) = kind.work_size(n);

    Ok(KernelStep {
        kernel,
        ndim: 3,
        global_work_size,
        local_work_size: Some(local_work_size),
        dynamic_args: vec![],
        op_tag,
        retained_bufs: vec![],
        noshuffle_act_rebuild: None,
    })
}

/// Helper: create a dedicated kernel and pre-bind Q4_0 noshuffle GEMV arguments.
///
/// Must mirror `OpenClBackend::matmul_q4_0_noshuffle` dispatch exactly.
///
/// The noshuffle GEMV kernel uses image1d_buffer_t for both weight nibbles (R32UI)
/// and activation (RGBA32F). The activation image wraps the source F32 buffer and
/// is retained in `retained_bufs` so its cl_mem stays valid for the plan's lifetime.
///
/// Kernel args:
///   0: weight nibbles image (image1d_buffer_t, R32UI)
///   1: weight scales (global half2*)
///   2: activation image (image1d_buffer_t, RGBA32F)
///   3: output buffer (global float*)
///   4: ne00 (K dimension, i32)
///   5: ne01 (M dimension, i32)
///
/// Dispatch: global=[ne01/2, N_SIMDGROUP=4, 1], local=[64, 4, 1], ndim=2.
#[allow(clippy::too_many_arguments)]
fn make_q4_0_noshuffle_matmul_step(
    program: &ocl::Program,
    context: &ocl::core::Context,
    q_img: &Mem,
    d_buf: &Mem,
    src_buf: &Mem,
    dst_buf: &Mem,
    ne00: usize,
    ne01: usize,
    op_tag: OpTag,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(program, "kernel_gemv_noshuffle_q4_0")
        .context("create kernel_gemv_noshuffle_q4_0 for plan")?;

    // Create activation image1d_buffer_t (RGBA32F) wrapping the F32 source buffer.
    // Each texel = float4 (4 floats), so width = ne00 / 4.
    // SAFETY: src_buf has at least ne00 * sizeof(f32) bytes, and ne00 is always a
    // multiple of 4 for Q4_0 (QK4_0=32).
    let act_img = {
        use ocl::core::{
            ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat, MemObjectType,
        };
        let fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
        let desc = ImageDescriptor::new(
            MemObjectType::Image1dBuffer,
            ne00 / 4,
            0,
            0,
            0,
            0,
            0,
            Some(src_buf.clone()),
        );
        unsafe {
            ocl::core::create_image(
                context,
                ocl::core::MEM_READ_ONLY,
                &fmt,
                &desc,
                None::<&[f32]>,
                None,
            )?
        }
    };

    let ne00_i = ne00 as i32;
    let ne01_i = ne01 as i32;

    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(q_img))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(d_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(&act_img))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(dst_buf))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&ne00_i))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&ne01_i))?;
    }

    let simdgroup_width: usize = 64;
    let n_simdgroup: usize = 4;
    let global_work_size = [ne01 / 2, n_simdgroup, 1];
    let local_work_size = [simdgroup_width, n_simdgroup, 1];

    Ok(KernelStep {
        kernel,
        ndim: 2,
        global_work_size,
        local_work_size: Some(local_work_size),
        dynamic_args: vec![],
        op_tag,
        retained_bufs: vec![act_img],
        // Per-dispatch image rebuild is unnecessary for the per-layer
        // matmuls that share `residual_buf`/`gate_buf`/`x_buf` across
        // sequential kernel dispatches in the same queue — image reads
        // observe preceding buffer writes correctly on Adreno 830 for
        // those cases. Left as `None` here to avoid per-token image
        // allocation overhead (~4 μs × 112 calls). If a future case
        // surfaces a coherency issue with a specific activation buffer,
        // flip to `Some((src_buf.clone(), ne00, 2))` for just that step.
        noshuffle_act_rebuild: None,
    })
}

/// Build an AOS Q4_0 matmul step using `kernel_mul_mat_q4_0_f32`.
/// Better for small-N projections (e.g. K/V where N=n_kv_heads*head_dim is small)
/// because it dispatches `ceil(N/4)*64` threads along N, giving better GPU utilization
/// than noshuffle GEMV (which creates only N/2 threads along rows).
///
/// Weight is in AOS BlockQ4_0 layout (18 bytes per block: 2 bytes half scale + 16 bytes nibbles).
#[allow(clippy::too_many_arguments)]
fn make_q4_0_aos_matmul_step(
    program: &ocl::Program,
    weight_buf: &Mem,
    src_buf: &Mem,
    dst_buf: &Mem,
    k: usize,
    n: usize,
    op_tag: OpTag,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(program, "kernel_mul_mat_q4_0_f32")
        .context("create kernel_mul_mat_q4_0_f32 for plan")?;

    let m = 1i32;
    let ne00 = k as i32;
    let ne01 = n as i32;
    let ne02 = 1i32;
    let ne10 = k as i32;
    let ne12 = k as i32;
    let ne0 = n as i32;
    let ne1 = n as i32;
    let r2 = 1i32;
    let r3 = 1i32;
    let zero_u64 = 0u64;
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(weight_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(src_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(dst_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
        ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
        ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
        ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&ne10))?;
        ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
        ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&ne0))?;
        ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&ne1))?;
        ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&r2))?;
        ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&r3))?;
    }

    let global_work_size = [n.div_ceil(4) * 64, m as usize, 1];
    let local_work_size = [64, 1, 1];

    Ok(KernelStep {
        kernel,
        ndim: 3,
        global_work_size,
        local_work_size: Some(local_work_size),
        dynamic_args: vec![],
        op_tag,
        retained_bufs: vec![],
        noshuffle_act_rebuild: None,
    })
}

/// N threshold below which AOS Q4_0 matmul is preferred over noshuffle GEMV.
/// At small N, noshuffle's gws=[N/2, 4, 1] creates too few workgroups to saturate
/// Adreno 830's 16+ compute units. AOS's gws=[ceil(N/4)*64, 1, 1] scales better.
/// Empirical crossover (microbench_ops on Qwen 2.5-1.5B Q4_0):
///   N=256 (K/V proj):  aos 10μs  vs noshuffle 21μs → aos wins
///   N=1536 (Q/O proj): aos 35μs  vs noshuffle 24μs → noshuffle wins
///   N=8960 (FFN):      aos 164μs vs noshuffle 128μs → noshuffle wins
const SMALL_N_AOS_THRESHOLD: usize = 512;

/// Build noshuffle GEMV programs, keyed by ne01 (M dimension).
///
/// Each unique ne01 requires different compile-time defines (LINE_STRIDE_A,
/// BLOCK_STRIDE_A). Returns a HashMap that `make_q4_0_noshuffle_matmul_step`
/// indexes into for kernel creation.
///
/// Tries vector sub_group_broadcast first (Adreno 830+), falls back to scalar.
pub fn build_noshuffle_programs(
    device: &ocl::Device,
    context: &ocl::Context,
    cl_opts: &str,
    ne01_set: &[usize],
) -> Result<std::collections::HashMap<usize, ocl::Program>> {
    let gemv_src = include_str!("../../../kernels/gemv_noshuffle_q4_0.cl");
    let mut programs = std::collections::HashMap::new();

    for &ne01 in ne01_set {
        if programs.contains_key(&ne01) {
            continue;
        }
        let line_stride_a = ne01 / 2;
        let block_stride_a = 4 * ne01;
        let simdgroup_width: usize = 64;
        let defines = format!(
            "{} -DLINE_STRIDE_A={} -DBLOCK_STRIDE_A={} -DSIMDGROUP_WIDTH={}",
            cl_opts, line_stride_a, block_stride_a, simdgroup_width
        );

        // Try vector sub_group_broadcast first (Adreno 830+ / driver v47+)
        let defines_vec = format!("{} -DVECTOR_SUB_GROUP_BROADCAT", defines);
        let program = match ocl::Program::builder()
            .devices(device)
            .src(gemv_src)
            .cmplr_opt(&defines_vec)
            .build(context)
        {
            Ok(p) => p,
            Err(_) => ocl::Program::builder()
                .devices(device)
                .src(gemv_src)
                .cmplr_opt(&defines)
                .build(context)
                .with_context(|| format!("build noshuffle program for ne01={}", ne01))?,
        };
        programs.insert(ne01, program);
    }

    Ok(programs)
}

/// Build a pre-bound `KernelStep` that dispatches `flash_attn_f32_f16_q1`.
///
/// Arg layout mirrors `OpenCLBackend::flash_attention_decode_gpu` — see that
/// method in `engine/src/backend/opencl/mod.rs` for the canonical layout.
/// All 40 args are static except `n_kv` at index 10, which is patched per
/// decode step via `DynamicArg::CacheSeqLen`.
fn build_flash_attention_step(config: &LayerPlanConfig) -> Result<AttentionVariant> {
    // Pick the program matching this layer's head_dim. The caller
    // (`use_flash` gate) guarantees the matching program is `Some`.
    let program = match config.head_dim {
        64 => config.flash_attn_f32_f16_program_dk64,
        128 => config.flash_attn_f32_f16_program_dk128,
        _ => None,
    }
    .expect("caller must verify flash program is Some for this head_dim");

    let kernel = ocl::core::create_kernel(program, "flash_attn_f32_f16_q1")
        .context("create flash_attn_f32_f16_q1 for plan")?;

    let n_heads_q = config.n_heads_q;
    let n_heads_kv = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_capacity = config.kv_capacity;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Q strides (F32 [batch=1, seq=1, n_heads_q, head_dim]), bytes
    let q_nb1 = (n_heads_q * head_dim * 4) as u64;
    let q_nb2 = (head_dim * 4) as u64;
    let q_nb3 = q_nb1;

    // KV strides (F16 HeadMajor [1, n_heads_kv, capacity, head_dim]), bytes
    let kv_elem_size: u64 = 2;
    let k_nb1 = (head_dim as u64) * kv_elem_size;
    let k_nb2 = (kv_capacity * head_dim) as u64 * kv_elem_size;
    let k_nb3 = (n_heads_kv as u64) * k_nb2;

    // O strides (F32 [batch=1, seq=1, n_heads_q, head_dim]), bytes
    let o_nb1 = (head_dim * 4) as u64;
    let o_nb2 = (n_heads_q * head_dim * 4) as u64;
    let o_nb3 = o_nb2;

    let n_q = 1i32;
    let initial_n_kv = 0i32;
    let is_causal = 0i32;
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
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.q_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.k_cache_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(config.v_cache_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::mem(config.out_attn_buf))?;
        ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&scale))?;
        ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&n_q))?;
        ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&initial_n_kv))?;
        ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&is_causal))?;
        ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&n_head))?;
        ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&q_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&q_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 15, ocl::core::ArgVal::scalar(&q_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 16, ocl::core::ArgVal::scalar(&k_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 17, ocl::core::ArgVal::scalar(&k_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 18, ocl::core::ArgVal::scalar(&k_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 19, ocl::core::ArgVal::scalar(&k_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 20, ocl::core::ArgVal::scalar(&k_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 21, ocl::core::ArgVal::scalar(&k_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 22, ocl::core::ArgVal::scalar(&o_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 23, ocl::core::ArgVal::scalar(&o_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 24, ocl::core::ArgVal::scalar(&o_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 25, ocl::core::ArgVal::scalar(&max_bias))?;
        ocl::core::set_kernel_arg(&kernel, 26, ocl::core::ArgVal::scalar(&m0))?;
        ocl::core::set_kernel_arg(&kernel, 27, ocl::core::ArgVal::scalar(&m1))?;
        ocl::core::set_kernel_arg(&kernel, 28, ocl::core::ArgVal::scalar(&n_head_log2))?;
        ocl::core::set_kernel_arg(&kernel, 29, ocl::core::ArgVal::scalar(&logit_softcap))?;
        ocl::core::set_kernel_arg(&kernel, 30, ocl::core::ArgVal::scalar(&n_head_kv_arg))?;
        ocl::core::set_kernel_arg(&kernel, 31, ocl::core::ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 32, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 33, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 34, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 35, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 36, ocl::core::ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 37, ocl::core::ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 38, ocl::core::ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 39, ocl::core::ArgVal::scalar(&zero_u64))?;
    }

    // Post-softmax score output (args 40-43). When a GPU score buffer is
    // supplied via LayerPlanConfig, bind it here so the Q1 kernel can write
    // per-token weights directly — avoiding the legacy kernel_attn_gen_half
    // fallback that used to dominate decode wall-clock. Otherwise bind a
    // 1-element dummy buffer and disable writes.
    let (score_mem, score_stride_val, score_layer_offset_val, write_scores, retained_bufs): (
        Mem,
        i32,
        i32,
        i32,
        Vec<Mem>,
    ) = match config.gpu_score_buf {
        Some(buf) => (
            buf.clone(),
            config.gpu_score_stride,
            config.gpu_score_layer_offset,
            1i32,
            vec![],
        ),
        None => {
            let dummy = unsafe {
                ocl::core::create_buffer::<_, f32>(
                    config.context.as_core(),
                    ocl::core::MEM_READ_WRITE,
                    1,
                    None,
                )
            }
            .context("create dummy score buffer for flash plan")?;
            let d_clone = dummy.clone();
            (dummy, 0i32, 0i32, 0i32, vec![d_clone])
        }
    };
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 40, ocl::core::ArgVal::mem(&score_mem))?;
        ocl::core::set_kernel_arg(
            &kernel,
            41,
            ocl::core::ArgVal::scalar(&score_layer_offset_val),
        )?;
        ocl::core::set_kernel_arg(&kernel, 42, ocl::core::ArgVal::scalar(&score_stride_val))?;
        ocl::core::set_kernel_arg(&kernel, 43, ocl::core::ArgVal::scalar(&write_scores))?;
        // Ragged-cache head starts (arg 44): rebound per layer step from `PlanGeometry::head_start`.
        ocl::core::set_kernel_arg(&kernel, 44, ocl::core::ArgVal::mem_null())?;
    }

    const Q1_WG_SIZE: usize = 64;
    Ok(AttentionVariant::StandardFlash(KernelStep {
        kernel,
        ndim: 2,
        global_work_size: [Q1_WG_SIZE, n_heads_q, 1],
        local_work_size: Some([Q1_WG_SIZE, 1, 1]),
        dynamic_args: vec![DynamicArg::CacheSeqLen { arg_idx: 10 }],
        op_tag: OpTag::Attention,
        retained_bufs,
        noshuffle_act_rebuild: None,
    }))
}

/// Everything the split-KV decode kernel pair (ticket 019) needs bound.
/// Shared by the plan builder and the runtime path
/// (`OpenCLBackend::flash_attention_decode_split_gpu`) so the two cannot
/// drift apart on argument order.
pub(crate) struct FlashQ1SplitArgs<'a> {
    pub q: &'a Mem,
    pub k: &'a Mem,
    pub v: &'a Mem,
    pub o: &'a Mem,
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub kv_capacity: usize,
    /// Initial `n_kv`; the plan patches it per step through `DynamicArg::CacheSeqLen`.
    pub n_kv: i32,
    /// `(buffer, layer_offset, stride, write_scores)`; the buffer must be valid
    /// even when `write_scores == 0` (bind a 1-element dummy).
    pub score: (&'a Mem, i32, i32, i32),
    pub kv_start: Option<&'a Mem>,
    pub part_ml: &'a Mem,
    pub part_o: &'a Mem,
    pub n_splits: usize,
}

/// Arg index of `n_kv` in `flash_attn_f32_f16_q1_split` (same slot as q1).
pub(crate) const Q1_SPLIT_MAIN_NKV_ARG: u32 = 10;
/// Arg index of `kv_start` in `flash_attn_f32_f16_q1_split` (same slot as q1).
pub(crate) const Q1_SPLIT_MAIN_KV_START_ARG: u32 = 44;
/// Arg index of `n_kv` in `flash_attn_q1_merge`.
pub(crate) const Q1_MERGE_NKV_ARG: u32 = 2;
/// Arg index of `kv_start` in `flash_attn_q1_merge`.
pub(crate) const Q1_MERGE_KV_START_ARG: u32 = 13;

/// Scratch sizes (in f32 elements) for the split pair: `(part_ml, part_o)`.
pub(crate) fn q1_split_scratch_len(
    n_heads_q: usize,
    n_splits: usize,
    head_dim: usize,
) -> (usize, usize) {
    (n_heads_q * n_splits * 2, n_heads_q * n_splits * head_dim)
}

/// Bind all 48 arguments of `flash_attn_f32_f16_q1_split`.
///
/// # Safety
/// `kernel` must have been created from `flash_attn_f32_f16_q1_split` and the
/// buffers must outlive every enqueue of it.
pub(crate) unsafe fn bind_flash_q1_split_main_args(
    kernel: &CoreKernel,
    a: &FlashQ1SplitArgs<'_>,
) -> Result<()> {
    use ocl::core::{ArgVal, set_kernel_arg};
    let head_dim = a.head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    // Q strides (F32 [batch=1, seq=1, n_heads_q, head_dim]), bytes
    let q_nb1 = (a.n_heads_q * head_dim * 4) as u64;
    let q_nb2 = (head_dim * 4) as u64;
    let q_nb3 = q_nb1;
    // KV strides (F16 HeadMajor [1, n_heads_kv, capacity, head_dim]), bytes
    let k_nb1 = (head_dim as u64) * 2;
    let k_nb2 = (a.kv_capacity * head_dim) as u64 * 2;
    let k_nb3 = (a.n_heads_kv as u64) * k_nb2;
    // O strides (F32 [batch=1, seq=1, n_heads_q, head_dim]), bytes
    let o_nb1 = (head_dim * 4) as u64;
    let o_nb2 = (a.n_heads_q * head_dim * 4) as u64;
    let o_nb3 = o_nb2;

    let n_q = 1i32;
    let is_causal = 0i32;
    let n_head = a.n_heads_q as i32;
    let n_head_kv = a.n_heads_kv as i32;
    let zero_f32 = 0.0f32;
    let zero_u64 = 0u64;
    let zero_i32 = 0i32;
    let (s_buf, s_layer_offset, s_stride, write_scores) = a.score;

    unsafe {
        set_kernel_arg(kernel, 0, ArgVal::mem(a.q))?;
        set_kernel_arg(kernel, 1, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, 2, ArgVal::mem(a.k))?;
        set_kernel_arg(kernel, 3, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, 4, ArgVal::mem(a.v))?;
        set_kernel_arg(kernel, 5, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, 6, ArgVal::mem(a.o))?;
        set_kernel_arg(kernel, 7, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, 8, ArgVal::scalar(&scale))?;
        set_kernel_arg(kernel, 9, ArgVal::scalar(&n_q))?;
        set_kernel_arg(kernel, Q1_SPLIT_MAIN_NKV_ARG, ArgVal::scalar(&a.n_kv))?;
        set_kernel_arg(kernel, 11, ArgVal::scalar(&is_causal))?;
        set_kernel_arg(kernel, 12, ArgVal::scalar(&n_head))?;
        set_kernel_arg(kernel, 13, ArgVal::scalar(&q_nb1))?;
        set_kernel_arg(kernel, 14, ArgVal::scalar(&q_nb2))?;
        set_kernel_arg(kernel, 15, ArgVal::scalar(&q_nb3))?;
        set_kernel_arg(kernel, 16, ArgVal::scalar(&k_nb1))?;
        set_kernel_arg(kernel, 17, ArgVal::scalar(&k_nb2))?;
        set_kernel_arg(kernel, 18, ArgVal::scalar(&k_nb3))?;
        set_kernel_arg(kernel, 19, ArgVal::scalar(&k_nb1))?;
        set_kernel_arg(kernel, 20, ArgVal::scalar(&k_nb2))?;
        set_kernel_arg(kernel, 21, ArgVal::scalar(&k_nb3))?;
        set_kernel_arg(kernel, 22, ArgVal::scalar(&o_nb1))?;
        set_kernel_arg(kernel, 23, ArgVal::scalar(&o_nb2))?;
        set_kernel_arg(kernel, 24, ArgVal::scalar(&o_nb3))?;
        // ALiBi / softcap unused (args 25-29)
        set_kernel_arg(kernel, 25, ArgVal::scalar(&zero_f32))?;
        set_kernel_arg(kernel, 26, ArgVal::scalar(&zero_f32))?;
        set_kernel_arg(kernel, 27, ArgVal::scalar(&zero_f32))?;
        set_kernel_arg(kernel, 28, ArgVal::scalar(&zero_i32))?;
        set_kernel_arg(kernel, 29, ArgVal::scalar(&zero_f32))?;
        set_kernel_arg(kernel, 30, ArgVal::scalar(&n_head_kv))?;
        // mask = NULL (args 31-37)
        set_kernel_arg(kernel, 31, ArgVal::mem_null())?;
        set_kernel_arg(kernel, 32, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, 33, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, 34, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, 35, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, 36, ArgVal::scalar(&zero_i32))?;
        set_kernel_arg(kernel, 37, ArgVal::scalar(&zero_i32))?;
        // sinks = NULL (args 38-39)
        set_kernel_arg(kernel, 38, ArgVal::mem_null())?;
        set_kernel_arg(kernel, 39, ArgVal::scalar(&zero_u64))?;
        // scores (args 40-43)
        set_kernel_arg(kernel, 40, ArgVal::mem(s_buf))?;
        set_kernel_arg(kernel, 41, ArgVal::scalar(&s_layer_offset))?;
        set_kernel_arg(kernel, 42, ArgVal::scalar(&s_stride))?;
        set_kernel_arg(kernel, 43, ArgVal::scalar(&write_scores))?;
        // ragged head starts (arg 44)
        let kv_start_val = match a.kv_start {
            Some(m) => ArgVal::mem(m),
            None => ArgVal::mem_null(),
        };
        set_kernel_arg(kernel, Q1_SPLIT_MAIN_KV_START_ARG, kv_start_val)?;
        // split partials (args 45-46) and the single-split finalize flag (arg 47)
        set_kernel_arg(kernel, 45, ArgVal::mem(a.part_ml))?;
        set_kernel_arg(kernel, 46, ArgVal::mem(a.part_o))?;
        let final_out: i32 = (a.n_splits == 1) as i32;
        set_kernel_arg(kernel, 47, ArgVal::scalar(&final_out))?;
    }
    Ok(())
}

/// Bind all 17 arguments of `flash_attn_q1_merge`.
///
/// # Safety
/// `kernel` must have been created from `flash_attn_q1_merge` and the buffers
/// must outlive every enqueue of it.
pub(crate) unsafe fn bind_flash_q1_merge_args(
    kernel: &CoreKernel,
    a: &FlashQ1SplitArgs<'_>,
) -> Result<()> {
    use ocl::core::{ArgVal, set_kernel_arg};
    let o_nb1 = (a.head_dim * 4) as u64;
    let o_nb3 = (a.n_heads_q * a.head_dim * 4) as u64;
    let n_head = a.n_heads_q as i32;
    let n_head_kv = a.n_heads_kv as i32;
    let n_splits = a.n_splits as i32;
    let zero_u64 = 0u64;
    let (s_buf, s_layer_offset, s_stride, write_scores) = a.score;
    unsafe {
        set_kernel_arg(kernel, 0, ArgVal::mem(a.o))?;
        set_kernel_arg(kernel, 1, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, Q1_MERGE_NKV_ARG, ArgVal::scalar(&a.n_kv))?;
        set_kernel_arg(kernel, 3, ArgVal::scalar(&n_head))?;
        set_kernel_arg(kernel, 4, ArgVal::scalar(&n_head_kv))?;
        set_kernel_arg(kernel, 5, ArgVal::scalar(&o_nb1))?;
        set_kernel_arg(kernel, 6, ArgVal::scalar(&o_nb3))?;
        // sinks = NULL (args 7-8)
        set_kernel_arg(kernel, 7, ArgVal::mem_null())?;
        set_kernel_arg(kernel, 8, ArgVal::scalar(&zero_u64))?;
        set_kernel_arg(kernel, 9, ArgVal::mem(s_buf))?;
        set_kernel_arg(kernel, 10, ArgVal::scalar(&s_layer_offset))?;
        set_kernel_arg(kernel, 11, ArgVal::scalar(&s_stride))?;
        set_kernel_arg(kernel, 12, ArgVal::scalar(&write_scores))?;
        let kv_start_val = match a.kv_start {
            Some(m) => ArgVal::mem(m),
            None => ArgVal::mem_null(),
        };
        set_kernel_arg(kernel, Q1_MERGE_KV_START_ARG, kv_start_val)?;
        set_kernel_arg(kernel, 14, ArgVal::mem(a.part_ml))?;
        set_kernel_arg(kernel, 15, ArgVal::mem(a.part_o))?;
        set_kernel_arg(kernel, 16, ArgVal::scalar(&n_splits))?;
    }
    Ok(())
}

/// Work sizes for the split pair: `(main gws, main lws, merge gws, merge lws)`.
pub(crate) fn q1_split_work_sizes(
    n_heads_q: usize,
    head_dim: usize,
    n_splits: usize,
) -> ([usize; 3], [usize; 3], [usize; 3], [usize; 3]) {
    const Q1_WG_SIZE: usize = 64;
    (
        [Q1_WG_SIZE, n_heads_q, n_splits],
        [Q1_WG_SIZE, 1, 1],
        [head_dim, n_heads_q, 1],
        [head_dim, 1, 1],
    )
}

/// Build the pre-bound lanes-across-row decode attention pair (ticket 019).
///
/// Mirrors [`build_flash_attention_step`]: same program selection, same
/// score binding, `n_kv` patched per step, `kv_start` rebound per layer step
/// by the executor. The partial buffers are private to this layer.
fn build_split_flash_attention_steps(
    config: &LayerPlanConfig,
    n_splits: usize,
) -> Result<AttentionVariant> {
    let program = match config.head_dim {
        64 => config.flash_attn_f32_f16_program_dk64,
        128 => config.flash_attn_f32_f16_program_dk128,
        _ => None,
    }
    .expect("caller must verify flash program is Some for this head_dim");

    let main_kernel = ocl::core::create_kernel(program, "flash_attn_f32_f16_q1_split")
        .context("create flash_attn_f32_f16_q1_split for plan")?;
    let merge_kernel = ocl::core::create_kernel(program, "flash_attn_q1_merge")
        .context("create flash_attn_q1_merge for plan")?;

    let (ml_len, o_len) = q1_split_scratch_len(config.n_heads_q, n_splits, config.head_dim);
    let alloc = |len: usize, what: &str| -> Result<Mem> {
        unsafe {
            ocl::core::create_buffer::<_, f32>(
                config.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                len,
                None,
            )
        }
        .with_context(|| format!("create {what} scratch for split flash plan"))
    };
    let part_ml = alloc(ml_len, "part_ml")?;
    let part_o = alloc(o_len, "part_o")?;

    let mut retained_bufs = vec![part_ml.clone(), part_o.clone()];
    let (score_mem, score_stride_val, score_layer_offset_val, write_scores) =
        match config.gpu_score_buf {
            Some(buf) => (
                buf.clone(),
                config.gpu_score_stride,
                config.gpu_score_layer_offset,
                1i32,
            ),
            None => {
                let dummy = alloc(1, "dummy score")?;
                retained_bufs.push(dummy.clone());
                (dummy, 0i32, 0i32, 0i32)
            }
        };

    let args = FlashQ1SplitArgs {
        q: config.q_buf,
        k: config.k_cache_buf,
        v: config.v_cache_buf,
        o: config.out_attn_buf,
        n_heads_q: config.n_heads_q,
        n_heads_kv: config.n_kv_heads,
        head_dim: config.head_dim,
        kv_capacity: config.kv_capacity,
        n_kv: 0,
        score: (
            &score_mem,
            score_layer_offset_val,
            score_stride_val,
            write_scores,
        ),
        kv_start: None,
        part_ml: &part_ml,
        part_o: &part_o,
        n_splits,
    };
    unsafe {
        bind_flash_q1_split_main_args(&main_kernel, &args)?;
        bind_flash_q1_merge_args(&merge_kernel, &args)?;
    }

    let (main_gws, main_lws, merge_gws, merge_lws) =
        q1_split_work_sizes(config.n_heads_q, config.head_dim, n_splits);
    Ok(AttentionVariant::SplitFlash {
        main: KernelStep {
            kernel: main_kernel,
            ndim: 3,
            global_work_size: main_gws,
            local_work_size: Some(main_lws),
            dynamic_args: vec![DynamicArg::CacheSeqLen {
                arg_idx: Q1_SPLIT_MAIN_NKV_ARG,
            }],
            op_tag: OpTag::Attention,
            retained_bufs,
            noshuffle_act_rebuild: None,
        },
        merge: KernelStep {
            kernel: merge_kernel,
            ndim: 2,
            global_work_size: merge_gws,
            local_work_size: Some(merge_lws),
            dynamic_args: vec![DynamicArg::CacheSeqLen {
                arg_idx: Q1_MERGE_NKV_ARG,
            }],
            op_tag: OpTag::Attention,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        },
    })
}

/// Build a pre-bound `kernel_add_row_bias` step that adds the given bias
/// buffer to the given `x` buffer in-place. Used after QKV matmul steps
/// for models with `has_qkv_bias=true` (Qwen2 etc.).
///
/// Kernel signature (from simple_ops.cl:487):
///   kernel_add_row_bias(float* x, const float* bias, int dim, int total)
///
/// Dispatch: 1D, global = total.div_ceil(64) * 64, no local size.
/// For decode seq_len=1 batch=1, `total == dim == n_heads * head_dim`.
fn build_add_row_bias_step(
    simple_ops_program: &ocl::Program,
    x_buf: &Mem,
    bias_buf: &Mem,
    dim: usize,
    op_tag: OpTag,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(simple_ops_program, "kernel_add_row_bias")
        .context("create kernel_add_row_bias for plan")?;
    let dim_i32 = dim as i32;
    let total_i32 = dim as i32; // decode: 1 row × dim elements
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(bias_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&dim_i32))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&total_i32))?;
    }
    let gws = dim.div_ceil(64) * 64;
    Ok(KernelStep {
        kernel,
        ndim: 1,
        global_work_size: [gws, 1, 1],
        local_work_size: None,
        dynamic_args: vec![],
        op_tag,
        retained_bufs: vec![],
        noshuffle_act_rebuild: None,
    })
}

/// Build a pre-bound kernel execution plan for one transformer layer's decode
/// step (seq_len=1).
///
/// Creates dedicated kernel objects via `ocl::core::create_kernel` and pre-binds
/// all static arguments. Dynamic arguments (start_pos, cache_seq_len, write_pos)
/// are tagged with [`DynamicArg`] and set to initial value 0.
pub fn build_layer_plan(config: &LayerPlanConfig) -> Result<LayerKernelPlan> {
    let local_size = 64usize;
    let local_mem_bytes = local_size * std::mem::size_of::<f32>();
    let dim = config.dim;
    let k = dim; // matmul inner dim = hidden dim

    let mut steps_pre_kv = Vec::with_capacity(9);

    // -----------------------------------------------------------------------
    // 1. rms_norm_oop (x -> residual)
    // -----------------------------------------------------------------------
    {
        // float4 path for dim divisible by 4 (all decoder arches).
        let kernel_name = if dim.is_multiple_of(4) {
            "kernel_rms_norm_oop_f4"
        } else {
            "kernel_rms_norm_oop"
        };
        let kernel = ocl::core::create_kernel(config.simple_ops_program, kernel_name)
            .context("create kernel_rms_norm_oop")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.residual_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.attn_norm_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&config.rms_norm_eps))?;
            // add_unit = 0 (non-Gemma3; Plan does not support Gemma3 yet)
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&0i32))?;
            ocl::core::set_kernel_arg(
                &kernel,
                6,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        steps_pre_kv.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::RmsNorm,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        });
    }

    // -----------------------------------------------------------------------
    // 2. matmul Q (residual -> q)
    // -----------------------------------------------------------------------
    if let (Some(ns), Some(progs)) = (&config.wq_noshuffle, config.noshuffle_programs) {
        let prog = progs
            .get(&ns.ne01)
            .context("noshuffle program for wq ne01")?;
        steps_pre_kv.push(make_q4_0_noshuffle_matmul_step(
            prog,
            config.context.as_core(),
            ns.q_img,
            ns.d_buf,
            config.residual_buf,
            config.q_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else {
        steps_pre_kv.push(make_f16_matmul_step(
            config.f16_program,
            config.residual_buf,
            config.wq_buf,
            config.q_buf,
            config.n_q,
            k,
            OpTag::MatmulQKV,
            None,
            config.is_nosub,
        )?);
    }

    // Optional: add Q bias (Qwen2 etc.) — non-bias models have bq_buf = None.
    if let Some(bq) = config.bq_buf {
        steps_pre_kv.push(build_add_row_bias_step(
            config.simple_ops_program,
            config.q_buf,
            bq,
            config.n_q,
            OpTag::MatmulQKV,
        )?);
    }

    // -----------------------------------------------------------------------
    // 3. matmul K (residual -> k)
    // -----------------------------------------------------------------------
    // NOTE: Tried AOS path for small-N K proj (N <= SMALL_N_AOS_THRESHOLD) — in
    // isolated microbench 10μs vs noshuffle 21μs, but production pp4096 showed
    // +24ms/tok regression (cause unclear, possibly cache/pipe interaction with
    // surrounding ops). Reverted to noshuffle pending investigation.
    if false
        && let Some(ns) = &config.wk_noshuffle
        && ns.ne01 <= SMALL_N_AOS_THRESHOLD
    {
        steps_pre_kv.push(make_q4_0_aos_matmul_step(
            config.q4_0_program,
            config.wk_buf,
            config.residual_buf,
            config.k_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else if let (Some(ns), Some(progs)) = (&config.wk_noshuffle, config.noshuffle_programs) {
        let prog = progs
            .get(&ns.ne01)
            .context("noshuffle program for wk ne01")?;
        steps_pre_kv.push(make_q4_0_noshuffle_matmul_step(
            prog,
            config.context.as_core(),
            ns.q_img,
            ns.d_buf,
            config.residual_buf,
            config.k_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else {
        steps_pre_kv.push(make_f16_matmul_step(
            config.f16_program,
            config.residual_buf,
            config.wk_buf,
            config.k_buf,
            config.n_k,
            k,
            OpTag::MatmulQKV,
            None,
            config.is_nosub,
        )?);
    }

    // Optional: add K bias (Qwen2 etc.) — non-bias models have bk_buf = None.
    if let Some(bk) = config.bk_buf {
        steps_pre_kv.push(build_add_row_bias_step(
            config.simple_ops_program,
            config.k_buf,
            bk,
            config.n_k,
            OpTag::MatmulQKV,
        )?);
    }

    // -----------------------------------------------------------------------
    // 4. matmul V (residual -> v)
    // -----------------------------------------------------------------------
    if false
        && let Some(ns) = &config.wv_noshuffle
        && ns.ne01 <= SMALL_N_AOS_THRESHOLD
    {
        steps_pre_kv.push(make_q4_0_aos_matmul_step(
            config.q4_0_program,
            config.wv_buf,
            config.residual_buf,
            config.v_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else if let (Some(ns), Some(progs)) = (&config.wv_noshuffle, config.noshuffle_programs) {
        let prog = progs
            .get(&ns.ne01)
            .context("noshuffle program for wv ne01")?;
        steps_pre_kv.push(make_q4_0_noshuffle_matmul_step(
            prog,
            config.context.as_core(),
            ns.q_img,
            ns.d_buf,
            config.residual_buf,
            config.v_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else {
        steps_pre_kv.push(make_f16_matmul_step(
            config.f16_program,
            config.residual_buf,
            config.wv_buf,
            config.v_buf,
            config.n_v,
            k,
            OpTag::MatmulQKV,
            None,
            config.is_nosub,
        )?);
    }

    // Optional: add V bias (Qwen2 etc.) — non-bias models have bv_buf = None.
    if let Some(bv) = config.bv_buf {
        steps_pre_kv.push(build_add_row_bias_step(
            config.simple_ops_program,
            config.v_buf,
            bv,
            config.n_v,
            OpTag::MatmulQKV,
        )?);
    }

    // -----------------------------------------------------------------------
    // 5. rope Q (q inplace)
    // -----------------------------------------------------------------------
    {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_rope_simple")
            .context("create kernel_rope_simple (Q)")?;
        let seq_len_i32 = 1i32;
        let start_pos_init = 0i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.q_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                1,
                ocl::core::ArgVal::scalar(&(config.head_dim as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                2,
                ocl::core::ArgVal::scalar(&(config.n_heads_q as i32)),
            )?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&seq_len_i32))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&start_pos_init))?;
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&config.rope_theta))?;
            let fs = &config.rope_freq_scaling;
            ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&fs.factor))?;
            ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&fs.low_freq_factor))?;
            ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&fs.high_freq_factor))?;
            ocl::core::set_kernel_arg(
                &kernel,
                9,
                ocl::core::ArgVal::scalar(&fs.original_max_position_embeddings),
            )?;
        }
        let work_size = config.n_heads_q * (config.head_dim / 2);
        steps_pre_kv.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [work_size, 1, 1],
            local_work_size: None,
            dynamic_args: vec![DynamicArg::StartPos { arg_idx: 4 }],
            op_tag: OpTag::Rope,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        });
    }

    // -----------------------------------------------------------------------
    // 6. rope K (k inplace)
    // -----------------------------------------------------------------------
    {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_rope_simple")
            .context("create kernel_rope_simple (K)")?;
        let seq_len_i32 = 1i32;
        let start_pos_init = 0i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.k_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                1,
                ocl::core::ArgVal::scalar(&(config.head_dim as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                2,
                ocl::core::ArgVal::scalar(&(config.n_kv_heads as i32)),
            )?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&seq_len_i32))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&start_pos_init))?;
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&config.rope_theta))?;
            let fs = &config.rope_freq_scaling;
            ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&fs.factor))?;
            ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&fs.low_freq_factor))?;
            ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&fs.high_freq_factor))?;
            ocl::core::set_kernel_arg(
                &kernel,
                9,
                ocl::core::ArgVal::scalar(&fs.original_max_position_embeddings),
            )?;
        }
        let work_size = config.n_kv_heads * (config.head_dim / 2);
        steps_pre_kv.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [work_size, 1, 1],
            local_work_size: None,
            dynamic_args: vec![DynamicArg::StartPos { arg_idx: 4 }],
            op_tag: OpTag::Rope,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        });
    }

    // -----------------------------------------------------------------------
    // 7. kv_scatter (k,v -> cache) — Standard variant
    // -----------------------------------------------------------------------
    let kv_update = {
        let kernel =
            ocl::core::create_kernel(config.simple_ops_program, "kernel_kv_scatter_f32_to_f16")
                .context("create kernel_kv_scatter_f32_to_f16")?;
        let capacity_init = config.kv_capacity as i32;
        let write_pos_init = 0i32;
        let n_elems = config.n_kv_heads * config.head_dim;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.k_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.v_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.k_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.v_cache_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                4,
                ocl::core::ArgVal::scalar(&(config.head_dim as i32)),
            )?;
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&capacity_init))?;
            ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&write_pos_init))?;
        }
        KvUpdateVariant::Standard(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [n_elems.div_ceil(64) * 64, 1, 1],
            local_work_size: Some([64, 1, 1]),
            dynamic_args: vec![
                DynamicArg::KvCapacity { arg_idx: 5 },
                DynamicArg::WritePos { arg_idx: 6 },
            ],
            op_tag: OpTag::KvScatter,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        })
    };

    // -----------------------------------------------------------------------
    // 8. attention (q, kv_cache -> out_attn)
    // -----------------------------------------------------------------------
    // Precondition gate: select flash attention when all conditions hold at
    // plan-build time. These are static for a given model + KV layout, so we
    // pre-bake the choice instead of runtime-gating per step.
    //
    // HeadMajor predicate matches the runtime check in `attention_gen` —
    // kv_pos_stride == head_dim and kv_head_stride == capacity * head_dim.
    //
    // Implicit invariants (safe today, must be revisited on refactor):
    //   1. KV dtype is F16 — plan.rs is only invoked for F16 KV caches;
    //      Q4_0 and quant-window use separate code paths. Add an explicit
    //      `config.kv_dtype == DType::F16` check if a new KV dtype gets
    //      plan-routed.
    //   2. `is_head_major` is reverse-inferred from stride values set by
    //      the caller. This assumes HeadMajor is the only layout that
    //      sets `kv_pos_stride = head_dim` and
    //      `kv_head_stride = capacity * head_dim` (see `attention_gen`
    //      in mod.rs for the canonical assignment).
    let is_head_major = config.kv_pos_stride == config.head_dim as i32
        && config.kv_head_stride == (config.kv_capacity * config.head_dim) as i32;
    // Flash attention is gated per head_dim because each DK variant is a
    // separate compiled program. Add a new arm here when adding DK=256
    // (Gemma3) etc. The Q1 kernel now emits post-softmax scores directly
    // when a GPU score buffer is supplied, so `needs_attention_scores`
    // alone no longer forces the legacy path — it does only when scores
    // must land in a CPU-readback buffer (`gpu_score_buf == None`).
    let flash_program_available = match config.head_dim {
        64 => config.flash_attn_f32_f16_program_dk64.is_some(),
        128 => config.flash_attn_f32_f16_program_dk128.is_some(),
        _ => false,
    };
    let scores_need_legacy_readback =
        config.needs_attention_scores && config.gpu_score_buf.is_none();
    let use_flash = is_head_major && flash_program_available && !scores_need_legacy_readback;

    let attention = if use_flash {
        if q1_use_split_pair() {
            build_split_flash_attention_steps(config, q1_splits())?
        } else {
            build_flash_attention_step(config)?
        }
    } else {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_attn_gen_half")
            .context("create kernel_attn_gen_half")?;
        let scale = 1.0f32 / (config.head_dim as f32).sqrt();
        let cache_seq_len_init = 0i32;
        // When a GPU score accumulator buffer is supplied and scores are
        // required, bind it as arg 4 (`scores`) with `write_scores=1` and
        // the accumulator's stride. This mirrors the runtime path in
        // `OpenCLBackend::attention_gen` (mod.rs:~3858), allowing the plan
        // to accumulate per-step importance without round-tripping through
        // forward_gen. When no GPU buffer is supplied we retain the legacy
        // dummy binding.
        let (write_scores, score_stride) =
            if config.needs_attention_scores && config.gpu_score_buf.is_some() {
                (1i32, config.gpu_score_stride)
            } else {
                (0i32, 0i32)
            };
        let dummy_score_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                config.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                1,
                None,
            )
        }
        .context("create dummy score buffer for plan")?;
        let score_arg_buf: &Mem = if write_scores == 1 {
            config.gpu_score_buf.unwrap()
        } else {
            &dummy_score_buf
        };
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.q_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.k_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.v_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.out_attn_buf))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(score_arg_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::scalar(&(config.head_dim as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                6,
                ocl::core::ArgVal::scalar(&(config.n_heads_q as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                7,
                ocl::core::ArgVal::scalar(&(config.n_kv_heads as i32)),
            )?;
            ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&cache_seq_len_init))?;
            ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(
                &kernel,
                10,
                ocl::core::ArgVal::scalar(&config.kv_pos_stride),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                11,
                ocl::core::ArgVal::scalar(&config.kv_head_stride),
            )?;
            ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&write_scores))?;
            ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&score_stride))?;
            ocl::core::set_kernel_arg(
                &kernel,
                14,
                ocl::core::ArgVal::scalar(&config.gpu_score_layer_offset),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                15,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
            // Ragged-cache head starts (arg 16): rebound per layer step from `PlanGeometry`.
            ocl::core::set_kernel_arg(&kernel, 16, ocl::core::ArgVal::mem_null())?;
        }
        AttentionVariant::Standard(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [config.n_heads_q * local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![DynamicArg::CacheSeqLen { arg_idx: 8 }],
            op_tag: OpTag::Attention,
            retained_bufs: vec![dummy_score_buf],
            noshuffle_act_rebuild: None,
        })
    };

    // -----------------------------------------------------------------------
    // Steps 9-10: post-attention pre-FFN (Wo + add_rms_norm)
    // -----------------------------------------------------------------------
    let mut steps_post_attn_pre_ffn = Vec::with_capacity(2);

    // 9. matmul Wo (out_attn -> attn_out)
    if let (Some(ns), Some(progs)) = (&config.wo_noshuffle, config.noshuffle_programs) {
        let prog = progs
            .get(&ns.ne01)
            .context("noshuffle program for wo ne01")?;
        steps_post_attn_pre_ffn.push(make_q4_0_noshuffle_matmul_step(
            prog,
            config.context.as_core(),
            ns.q_img,
            ns.d_buf,
            config.out_attn_buf,
            config.attn_out_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulWo,
        )?);
    } else {
        steps_post_attn_pre_ffn.push(make_f16_matmul_step(
            config.f16_program,
            config.out_attn_buf,
            config.wo_buf,
            config.attn_out_buf,
            dim,
            dim,
            OpTag::MatmulWo,
            None,
            config.is_nosub,
        )?);
    }

    // 10. add_rms_norm_oop (x += attn_out, then norm -> residual)
    {
        let kernel_name = if dim.is_multiple_of(4) {
            "kernel_add_rms_norm_oop_f4"
        } else {
            "kernel_add_rms_norm_oop"
        };
        let kernel = ocl::core::create_kernel(config.simple_ops_program, kernel_name)
            .context("create kernel_add_rms_norm_oop")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.attn_out_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.residual_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.ffn_norm_buf))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&config.rms_norm_eps))?;
            // add_unit = 0 (non-Gemma3)
            ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&0i32))?;
            ocl::core::set_kernel_arg(
                &kernel,
                7,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        steps_post_attn_pre_ffn.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::AddRmsNorm,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        });
    }

    // -----------------------------------------------------------------------
    // Steps 11-14: FFN (GPU-only variant built here; partition variant routed
    // through `build_partitioned_layer_plan`).
    // -----------------------------------------------------------------------
    // 11. matmul gate (residual -> gate)
    let gate_step =
        if let (Some(ns), Some(progs)) = (&config.w_gate_noshuffle, config.noshuffle_programs) {
            let prog = progs
                .get(&ns.ne01)
                .context("noshuffle program for w_gate ne01")?;
            make_q4_0_noshuffle_matmul_step(
                prog,
                config.context.as_core(),
                ns.q_img,
                ns.d_buf,
                config.residual_buf,
                config.gate_buf,
                ns.ne00,
                ns.ne01,
                OpTag::MatmulGateUp,
            )?
        } else {
            make_f16_matmul_step(
                config.f16_program,
                config.residual_buf,
                config.w_gate_buf,
                config.gate_buf,
                config.ffn_hidden,
                k,
                OpTag::MatmulGateUp,
                config.f16_l4_program,
                config.is_nosub,
            )?
        };

    // 12. matmul up (residual -> up)
    let up_step =
        if let (Some(ns), Some(progs)) = (&config.w_up_noshuffle, config.noshuffle_programs) {
            let prog = progs
                .get(&ns.ne01)
                .context("noshuffle program for w_up ne01")?;
            make_q4_0_noshuffle_matmul_step(
                prog,
                config.context.as_core(),
                ns.q_img,
                ns.d_buf,
                config.residual_buf,
                config.up_buf,
                ns.ne00,
                ns.ne01,
                OpTag::MatmulGateUp,
            )?
        } else {
            make_f16_matmul_step(
                config.f16_program,
                config.residual_buf,
                config.w_up_buf,
                config.up_buf,
                config.ffn_hidden,
                k,
                OpTag::MatmulGateUp,
                config.f16_l4_program,
                config.is_nosub,
            )?
        };

    // 13. silu_mul (gate = silu(gate) * up)
    let silu_mul_step = {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_silu_mul_simple")
            .context("create kernel_silu_mul_simple")?;
        let size4 = (config.ffn_hidden / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.gate_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.up_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&size4))?;
        }
        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [config.ffn_hidden / 4, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::SiluMul,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        }
    };

    // 14. matmul down (gate -> down)
    let down_step =
        if let (Some(ns), Some(progs)) = (&config.w_down_noshuffle, config.noshuffle_programs) {
            let prog = progs
                .get(&ns.ne01)
                .context("noshuffle program for w_down ne01")?;
            make_q4_0_noshuffle_matmul_step(
                prog,
                config.context.as_core(),
                ns.q_img,
                ns.d_buf,
                config.gate_buf,
                config.down_buf,
                ns.ne00,
                ns.ne01,
                OpTag::MatmulDown,
            )?
        } else {
            make_f16_matmul_step(
                config.f16_program,
                config.gate_buf,
                config.w_down_buf,
                config.down_buf,
                dim,
                config.ffn_hidden,
                OpTag::MatmulDown,
                None,
                config.is_nosub,
            )?
        };

    let ffn = FfnVariant::GpuOnly {
        gate: gate_step,
        up: up_step,
        silu_mul: silu_mul_step,
        down: down_step,
    };

    // -----------------------------------------------------------------------
    // Step 15: post-FFN (residual add).
    // -----------------------------------------------------------------------
    let mut steps_post_ffn = Vec::with_capacity(1);
    {
        let kernel =
            ocl::core::create_kernel(config.simple_ops_program, "kernel_add_assign_simple")
                .context("create kernel_add_assign_simple")?;
        let size4 = (dim / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.down_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&size4))?;
        }
        steps_post_ffn.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [dim / 4, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::AddAssign,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        });
    }

    Ok(LayerKernelPlan {
        attn: AttnVariant::GpuOnly(GpuAttnSteps {
            steps_pre_kv,
            kv_update,
            attention,
            steps_post_attn_pre_ffn,
        }),
        ffn,
        steps_post_ffn,
        flush_after: false,
    })
}

// ---------------------------------------------------------------------------
// Full model plan builder
// ---------------------------------------------------------------------------

/// Config for building the full model plan (all layers + final norm + lm_head).
pub struct FullPlanConfig<'a> {
    pub context: &'a ocl::Context,
    pub f16_program: &'a ocl::Program,
    pub f16_l4_program: Option<&'a ocl::Program>,
    pub simple_ops_program: &'a ocl::Program,
    /// AOS Q4_0 matmul program (`kernel_mul_mat_q4_0_f32`). Used for small-N
    /// matmuls where noshuffle GEMV under-utilizes the GPU (e.g. K/V projection).
    pub q4_0_program: &'a ocl::Program,
    /// Flash attention program for head_dim=64. When `Some`, the layer
    /// builder may select `AttentionVariant::StandardFlash` for layers
    /// with head_dim=64.
    pub flash_attn_f32_f16_program_dk64: Option<&'a ocl::Program>,
    /// Flash attention program for head_dim=128. When `Some`, the layer
    /// builder may select `AttentionVariant::StandardFlash` for layers
    /// with head_dim=128 (e.g. Qwen 2.5-1.5B).
    pub flash_attn_f32_f16_program_dk128: Option<&'a ocl::Program>,
    /// True if this decode plan must capture attention scores (heavy-hitter / GPU
    /// score accumulator). Forces the legacy attention path because flash
    /// attention has no score output.
    pub needs_attention_scores: bool,
    /// Persistent GPU score buffer from `OpenCLBackend::gpu_score_acc()`.
    /// Propagated into each layer's `LayerPlanConfig::gpu_score_buf`.
    pub gpu_score_buf: Option<&'a Mem>,
    /// Score stride for the buffer above.
    pub gpu_score_stride: i32,
    // Per-layer weight buffers: Vec<(wq, wk, wv, wo, w_gate, w_up, w_down, attn_norm, ffn_norm)>
    pub layer_bufs: Vec<LayerBufs<'a>>,
    // Workspace buffers (shared across layers)
    pub x_buf: &'a Mem,
    pub q_buf: &'a Mem,
    pub k_buf: &'a Mem,
    pub v_buf: &'a Mem,
    pub out_attn_buf: &'a Mem,
    pub attn_out_buf: &'a Mem,
    pub gate_buf: &'a Mem,
    pub up_buf: &'a Mem,
    pub down_buf: &'a Mem,
    pub residual_buf: &'a Mem,
    /// Permanent-mapped host pointer backing `residual_buf`, if the
    /// residual tensor is an ALLOC_HOST_PTR UnifiedBuffer with `.map()`
    /// held for the plan's lifetime. `null` otherwise. Propagated into
    /// `LayerPlanConfig.residual_host_ptr` for partition plan builds.
    pub residual_host_ptr: *const u8,
    // Per-layer KV cache buffers
    pub kv_bufs: Vec<KvBufs<'a>>,
    // Final norm + lm_head
    pub final_norm_buf: &'a Mem,
    /// `None` when lm_head is kept on CPU (large tied embedding).
    pub lm_head_buf: Option<&'a Mem>,
    pub logits_buf: &'a Mem,
    // Model config
    pub dim: usize,
    pub n_heads_q: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// llama3 `rope_scaling` (identity when `factor == 1`) — see [`crate::rope`].
    pub rope_freq_scaling: crate::rope::RopeFreqScaling,
    pub kv_capacity: usize,
    pub kv_pos_stride: i32,
    pub kv_head_stride: i32,
    /// Whether the device lacks subgroup support (nosub fallback path).
    pub is_nosub: bool,
    // -- Q4_0 noshuffle matmul support --
    /// Pre-compiled noshuffle GEMV programs, keyed by ne01 (M dimension).
    /// When `Some`, layers with noshuffle entries use Q4_0 GEMV kernels.
    pub noshuffle_programs: Option<std::collections::HashMap<usize, ocl::Program>>,
    /// Per-layer noshuffle SOA entries for lm_head. `None` for F16 or CPU lm_head.
    pub lm_head_noshuffle: Option<NoshufflePlanEntry<'a>>,
    /// Tensor partition (ticket 021): `Some` = every layer is split between GPU and CPU.
    pub tp: Option<TpPlanConfig<'a>>,
    /// dtype of the lm_head weight tensor. Determines which GPU matmul
    /// variant the plan binds (noshuffle GEMV for Q4_0, F16 GEMV for F16).
    /// When the lm_head is stored as F32 (e.g. Llama 3.2 GGUF where
    /// `token_embd.weight` is F32 and the tied lm_head inherits that dtype),
    /// the plan **cannot** dispatch it on GPU — the F16 GEMV kernel reads
    /// 2 bytes per element and would read F32-interpreted-as-F16 halves,
    /// producing wildly scaled garbage logits. The caller should set this
    /// to the actual dtype of `lm_head_buf` so `build_full_plan` can skip
    /// emitting a broken GPU step and let `execute_plan` fall through to
    /// `lm_head_matmul_cpu`.
    pub lm_head_dtype: crate::buffer::DType,
    /// ENG-ALG-219: `TransformerModel::ratio_generation` Arc clone passed in
    /// from `build_plan`. The plan builder captures the current value with
    /// Acquire and stores both into `FullKernelPlan`. `execute()` uses this
    /// for the global weight-swap invalidation check (INV-129).
    pub ratio_generation: Arc<std::sync::atomic::AtomicU64>,
}

/// Per-layer weight buffer references.
pub struct LayerBufs<'a> {
    pub wq: &'a Mem,
    pub wk: &'a Mem,
    pub wv: &'a Mem,
    pub wo: &'a Mem,
    pub w_gate: &'a Mem,
    pub w_up: &'a Mem,
    pub w_down: &'a Mem,
    pub attn_norm: &'a Mem,
    pub ffn_norm: &'a Mem,
    /// Optional QKV bias buffers (F32). Present for models with
    /// `has_qkv_bias=true` (e.g. Qwen2). When `None`, the plan builder
    /// skips the `kernel_add_row_bias` step after each QKV matmul.
    pub bq: Option<&'a Mem>,
    pub bk: Option<&'a Mem>,
    pub bv: Option<&'a Mem>,
    // -- Q4_0 noshuffle SOA entries (optional, None for F16 weights) --
    pub wq_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wk_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wv_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wo_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_gate_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_up_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_down_noshuffle: Option<NoshufflePlanEntry<'a>>,
}

/// What `build_full_plan` needs for a CPU–GPU partitioned model (ticket 021).
pub struct TpPlanConfig<'a> {
    /// One per layer (every layer is partitioned).
    pub ctxs: Vec<&'a PartitionContext>,
    /// Host views of each layer's weights for the CPU share.
    pub host: Vec<Arc<TpHostWeights>>,
    pub workspace: Arc<PartitionWsCell>,
    pub kernels: Option<&'static crate::cpu_kernels::CpuKernelSet>,
}

/// Per-layer KV cache buffer references.
pub struct KvBufs<'a> {
    pub k_cache: &'a Mem,
    pub v_cache: &'a Mem,
}

/// Build a pre-bound plan for the full model decode pass (all layers + head).
// LAYER-EXEMPT: dispatch_orchestrator
pub fn build_full_plan(config: &FullPlanConfig) -> Result<FullKernelPlan> {
    let n_q = config.n_heads_q * config.head_dim;
    let n_k = config.n_kv_heads * config.head_dim;
    let n_v = n_k;

    let n_layers = config.layer_bufs.len();
    if let Some(tp) = config.tp.as_ref() {
        anyhow::ensure!(
            tp.ctxs.len() == n_layers && tp.host.len() == n_layers,
            "tensor partition must cover every layer"
        );
        super::tp_plan::prepare_runtime(
            &tp.workspace,
            &super::tp_plan::TpRuntimeGeom {
                n_layers,
                n_heads_q: config.n_heads_q,
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                ffn_hidden: config.ffn_hidden,
                kv_capacity: config.kv_capacity,
            },
            tp.ctxs[0].gpu_ratio,
            &tp.ctxs[0].cpu_backend,
        )?;
    }

    let mut layers = Vec::with_capacity(config.layer_bufs.len());
    for (i, (lb, kb)) in config
        .layer_bufs
        .iter()
        .zip(config.kv_bufs.iter())
        .enumerate()
    {
        let layer_config = LayerPlanConfig {
            context: config.context,
            f16_program: config.f16_program,
            f16_l4_program: config.f16_l4_program,
            simple_ops_program: config.simple_ops_program,
            q4_0_program: config.q4_0_program,
            x_buf: config.x_buf,
            wq_buf: lb.wq,
            wk_buf: lb.wk,
            wv_buf: lb.wv,
            bq_buf: lb.bq,
            bk_buf: lb.bk,
            bv_buf: lb.bv,
            wo_buf: lb.wo,
            w_gate_buf: lb.w_gate,
            w_up_buf: lb.w_up,
            w_down_buf: lb.w_down,
            attn_norm_buf: lb.attn_norm,
            ffn_norm_buf: lb.ffn_norm,
            q_buf: config.q_buf,
            k_buf: config.k_buf,
            v_buf: config.v_buf,
            out_attn_buf: config.out_attn_buf,
            attn_out_buf: config.attn_out_buf,
            gate_buf: config.gate_buf,
            up_buf: config.up_buf,
            down_buf: config.down_buf,
            residual_buf: config.residual_buf,
            k_cache_buf: kb.k_cache,
            v_cache_buf: kb.v_cache,
            dim: config.dim,
            n_heads_q: config.n_heads_q,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            ffn_hidden: config.ffn_hidden,
            n_q,
            n_k,
            n_v,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            rope_freq_scaling: config.rope_freq_scaling,
            kv_capacity: config.kv_capacity,
            kv_pos_stride: config.kv_pos_stride,
            kv_head_stride: config.kv_head_stride,
            is_nosub: config.is_nosub,
            flash_attn_f32_f16_program_dk64: config.flash_attn_f32_f16_program_dk64,
            flash_attn_f32_f16_program_dk128: config.flash_attn_f32_f16_program_dk128,
            needs_attention_scores: config.needs_attention_scores,
            gpu_score_buf: config.gpu_score_buf,
            gpu_score_stride: config.gpu_score_stride,
            // Per-layer offset into the [n_layers, n_heads_q, score_stride]
            // score buffer (in f32 elements). Pre-baked into the attention
            // kernel's `score_layer_offset` arg so each layer writes into its
            // own slice without per-token arg updates.
            gpu_score_layer_offset: (i as i32)
                * (config.n_heads_q as i32)
                * config.gpu_score_stride,
            noshuffle_programs: config.noshuffle_programs.as_ref(),
            wq_noshuffle: lb.wq_noshuffle,
            wk_noshuffle: lb.wk_noshuffle,
            wv_noshuffle: lb.wv_noshuffle,
            wo_noshuffle: lb.wo_noshuffle,
            w_gate_noshuffle: lb.w_gate_noshuffle,
            w_up_noshuffle: lb.w_up_noshuffle,
            w_down_noshuffle: lb.w_down_noshuffle,
            residual_host_ptr: config.residual_host_ptr,
        };
        let layer_plan = if let Some(tp) = config.tp.as_ref() {
            super::tp_plan::build_tp_layer(
                &layer_config,
                super::tp_plan::TpLayerInputs {
                    ctx: tp.ctxs[i],
                    host: tp.host[i].clone(),
                    ws: &tp.workspace,
                    layer: i,
                    kernels: tp.kernels,
                },
            )
            .with_context(|| format!("build partitioned plan for layer {}", i))?
        } else {
            build_layer_plan(&layer_config)
                .with_context(|| format!("build plan for layer {}", i))?
        };
        layers.push(layer_plan);
    }

    // Only the last layer needs clFlush before final norm + lm_head
    if let Some(last) = layers.last_mut() {
        last.flush_after = true;
    }

    // Final RMSNorm (in-place on x)
    let final_norm = {
        let kernel_name = if config.dim.is_multiple_of(4) {
            "kernel_rms_norm_opt_f4"
        } else {
            "kernel_rms_norm_opt"
        };
        let kernel = ocl::core::create_kernel(config.simple_ops_program, kernel_name)
            .context("create final kernel_rms_norm_opt")?;
        let local_size = 64usize;
        let local_mem_bytes = local_size * std::mem::size_of::<f32>();
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.final_norm_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(config.dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&config.rms_norm_eps))?;
            // add_unit = 0 (non-Gemma3)
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&0i32))?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1], // rows=1 for decode
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::FinalNorm,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        }
    };

    // lm_head matmul: x [1, dim] × lm_head [vocab, dim]^T → logits [1, vocab]
    // None when lm_head is on CPU (large tied embedding that exceeds GPU alloc
    // limit) OR when the weight dtype has no matching GPU GEMV variant — see
    // `lm_head_dtype` on FullPlanConfig for why an F32 lm_head must fall
    // through to CPU even though the buffer is uploaded to GPU.
    use crate::buffer::DType;
    let lm_head = if let Some(lm_head_buf) = config.lm_head_buf {
        if let (Some(ns), Some(progs)) = (&config.lm_head_noshuffle, &config.noshuffle_programs) {
            let prog = progs
                .get(&ns.ne01)
                .context("noshuffle program for lm_head ne01")?;
            Some(
                make_q4_0_noshuffle_matmul_step(
                    prog,
                    config.context.as_core(),
                    ns.q_img,
                    ns.d_buf,
                    config.x_buf,
                    config.logits_buf,
                    ns.ne00,
                    ns.ne01,
                    OpTag::LmHead,
                )
                .context("build lm_head noshuffle matmul step")?,
            )
        } else if config.lm_head_dtype == DType::F16 {
            Some(
                make_f16_matmul_step(
                    config.f16_program,
                    config.x_buf,
                    lm_head_buf,
                    config.logits_buf,
                    config.vocab_size,
                    config.dim,
                    OpTag::LmHead,
                    config.f16_l4_program,
                    config.is_nosub,
                )
                .context("build lm_head matmul step")?,
            )
        } else {
            // Unsupported dtype for the plan's GPU lm_head dispatch
            // (F32, tied-embedding case). Return None and let
            // `execute_plan` fall through to the CPU matmul path.
            log::info!(
                "plan: lm_head dtype {:?} has no GPU GEMV variant, falling back to CPU",
                config.lm_head_dtype
            );
            None
        }
    } else {
        None
    };

    // ENG-ALG-219: capture global ratio_generation at build time (Acquire).
    // Stored in the plan so execute() can detect weight swaps without a
    // second Arc clone per token (just one atomic load per execute call).
    use std::sync::atomic::Ordering;
    let ratio_generation_at_build = config.ratio_generation.load(Ordering::Acquire);
    let ratio_generation_counter = config.ratio_generation.clone();

    Ok(FullKernelPlan {
        layers,
        final_norm,
        lm_head,
        kv_capacity: config.kv_capacity,
        writes_gpu_scores: config.needs_attention_scores && config.gpu_score_buf.is_some(),
        ratio_generation_at_build,
        ratio_generation_counter,
        q_row_copy: None,
        tp: config.tp.as_ref().map(|tp| tp.workspace.clone()),
    })
}
