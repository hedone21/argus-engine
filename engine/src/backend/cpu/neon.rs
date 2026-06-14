// Hand-written ARM NEON SIMD: intrinsic-dense, perf-shaped, with helper routines kept
// for completeness. The lints below are noise in this style, so they are allowed
// module-wide rather than scattered across the intrinsic sites.
#![allow(
    unused_unsafe,
    dead_code,
    clippy::missing_safety_doc,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::doc_overindented_list_items
)]

use crate::backend::Backend;
use crate::backend::cpu::common::CpuBackendCommon;
use crate::buffer::DType;
use crate::quant::{BlockQ4_0, BlockQ8_0, QK4_0, QK8_0};
use crate::tensor::Tensor;
use crate::thread_pool::{self, WorkFn};
use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::arch::aarch64::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Mutex, OnceLock};

/// Runtime toggle: when true, F16 matmul uses Rayon instead of SpinPool.
/// Set via `--use-rayon` CLI flag for A/B benchmarking.
pub static USE_RAYON: AtomicBool = AtomicBool::new(false);

// Reusable Q8_0 workspace for matmul_transposed_q4_0.
// Eliminates per-matmul Vec<BlockQ8_0> allocation (~112 allocs/token).
// Safety: single-threaded inference model — only the main thread quantizes into this buffer
// before par_chunks_mut reads it via raw pointer.
thread_local! {
    static Q8_WORKSPACE: RefCell<Vec<BlockQ8_0>> = const { RefCell::new(Vec::new()) };
}

/// Wrapper to send `*const BlockQ8_0` into Rayon parallel closures.
/// Safety: the pointer targets a thread-local buffer that is fully initialized
/// before the parallel region and only read (never written) during it.
#[derive(Clone, Copy)]
struct SendSyncPtr(*const BlockQ8_0);
unsafe impl Send for SendSyncPtr {}
unsafe impl Sync for SendSyncPtr {}
impl SendSyncPtr {
    #[inline(always)]
    fn ptr(self) -> *const BlockQ8_0 {
        self.0
    }
}

// sdot_asm and prefetch_asm removed (unused)

pub struct CpuBackendNeon;

impl Default for CpuBackendNeon {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackendNeon {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackendNeon {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn name(&self) -> &str {
        "CPU (NEON)"
    }

    fn device(&self) -> &str {
        "CPU (AArch64)"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().matmul(a, b, out)
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        match b.dtype() {
            DType::F32 => self.matmul_transposed_f32(a, b, out),
            DType::F16 => self.matmul_transposed_f16(a, b, out),
            DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
            _ => CpuBackendCommon::new().matmul_transposed(a, b, out),
        }
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        _rows: usize,
        _cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        match b.dtype() {
            DType::F32 => self.matmul_transposed_f32(a, b, out),
            DType::F16 => self.matmul_transposed_f16(a, b, out),
            DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
            _ => CpuBackendCommon::new().matmul_transposed(a, b, out),
        }
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        CpuBackendCommon::new().add_assign(a, b)
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        CpuBackendCommon::new().scale(x, v)
    }

    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        let a_data = a.as_mut_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        unsafe { Self::swiglu_neon(a_data, b_data) };
        Ok(())
    }

    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let x_data = x.as_mut_slice::<f32>();
        let w_data = w.as_slice::<f32>();
        x_data.par_chunks_mut(dim).for_each(|row| {
            unsafe { Self::rms_norm_neon(row, w_data, eps, add_unit) };
        });
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
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let x_data = x.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();
        let w_data = w.as_slice::<f32>();
        for (x_row, out_row) in x_data.chunks(dim).zip(out_data.chunks_mut(dim)) {
            unsafe { Self::rms_norm_oop_neon(x_row, out_row, w_data, eps, add_unit) };
        }
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
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let x_data = x.as_mut_slice::<f32>();
        let res_data = residual.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();
        let w_data = w.as_slice::<f32>();
        for ((x_row, res_row), out_row) in x_data
            .chunks_mut(dim)
            .zip(res_data.chunks(dim))
            .zip(out_data.chunks_mut(dim))
        {
            unsafe { Self::add_rms_norm_oop_neon(x_row, res_row, out_row, w_data, eps, add_unit) };
        }
        Ok(())
    }

    fn gelu_tanh_mul(&self, gate: &mut Tensor, up: &Tensor) -> Result<()> {
        let gate_data = gate.as_mut_slice::<f32>();
        let up_data = up.as_slice::<f32>();
        unsafe { Self::gelu_tanh_mul_neon(gate_data, up_data) };
        Ok(())
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().softmax(x)
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        CpuBackendCommon::new().rope_inplace(x, start_pos, theta)
    }

    fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
        CpuBackendCommon::new().copy_from(t)
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().cast(src, dst)
    }

    #[allow(clippy::too_many_arguments)]
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
        // F16 KV cache: use NEON-optimized path (direct F16·F32 dot, vectorized softmax)
        if k_cache.dtype() == DType::F16 {
            return Self::attention_gen_f16_neon(
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
        // F32 / Q4_0: delegate to common implementation
        CpuBackendCommon::new().attention_gen(
            q,
            k_cache,
            v_cache,
            out,
            num_heads_q,
            num_heads_kv,
            head_dim,
            cache_seq_len,
            scores_out,
        )
    }

    // B-5b Phase 2 Stage 1: expose fused NEON kernels via the trait so Stage
    // 2 callers can reach them through `backend.cpu_kernels()` without the
    // direct `crate::backend::cpu::neon::fused_matmul_*` path.
    fn cpu_kernels(&self) -> Option<&'static crate::cpu_kernels::CpuKernelSet> {
        Some(&crate::cpu_kernels::CPU_KERNEL_SET)
    }

    // B-5b Phase 2 Stage 1: CPU companion = self (the CpuBackend IS the CPU
    // companion). See `Backend::cpu_companion` doc for why this is not a
    // default impl.
    fn cpu_companion(&self) -> &dyn Backend {
        self
    }
}

/// Size of one KV chunk (in tokens) used by the flash-decoding path.
///
/// Chosen so that one chunk of K+V in F16 occupies ~512 KB on Qwen2.5 1.5B
/// (head_dim=128): `1024 * 128 * 2 bytes/elem * 2 (K+V) = 512 KiB`, which is
/// ~25% of the Snapdragon 8 Elite Oryon L2 (2 MiB) and leaves room for other
/// threads.  Must be a multiple of 4 to keep the inner 4-way unrolled loop
/// aligned.  See `arch/cpu_flash_decoding.md` §4.1.
const FLASH_DECODE_CHUNK_SIZE: usize = 1024;

/// Minimum `cache_seq_len` at which the flash-decoding path activates.
///
/// Below this threshold the head-parallel Step 1 path is retained to avoid
/// its amortization overhead (partial-state buffer alloc + LSE merge) on
/// short contexts where GQA redundant reads are not the bottleneck.  Set
/// equal to `CHUNK_SIZE` so the single-chunk case (`n_chunks = 1`) is still
/// routed to flash decoding (merge is a trivial no-op).
const FLASH_DECODE_THRESHOLD: usize = 1024;

/// Per-(chunk, q-head) partial online-softmax state.
///
/// Aligned to a 64-byte cache line so neighbouring workers writing to
/// adjacent `PartialState` entries never share a cache line (false-sharing
/// prevention).  The flat `o_tilde` buffer (F32 `head_dim` lanes) is carried
/// out-of-band in `FlashDecodeCtx::partial_o` because `head_dim` is only
/// known at runtime; this struct tracks just the scalar `(m, l)` pair.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct PartialState {
    /// Chunk-local max logit.
    m: f32,
    /// Chunk-local softmax denominator `Σ exp(s_t − m)`.
    l: f32,
}

impl PartialState {
    const EMPTY: PartialState = PartialState {
        m: f32::NEG_INFINITY,
        l: 0.0,
    };
}

/// Flash-decoding dispatch context.  A single `FlashDecodeCtx` is shared
/// across SpinPool workers; each worker derives its `(chunk_idx, kv_h)` from
/// its task id and reads Q/K/V / writes its own slice of `partial_ml` and
/// `partial_o`.
///
/// # Safety
/// The raw pointers are required because `SpinPool::dispatch` takes a
/// `*const u8` context; `Send + Sync` is sound because:
/// * `q_ptr`, `k_ptr`, `v_ptr` point to read-only tensor slices that outlive
///   `dispatch`.
/// * `partial_ml_ptr` / `partial_o_ptr` point to mutable buffers whose
///   entries are partitioned across workers by `(chunk_idx, q_h)` — no two
///   workers touch the same element.
struct FlashDecodeCtx {
    q_ptr: *const f32,
    k_ptr: *const u16,
    v_ptr: *const u16,
    partial_ml_ptr: *mut PartialState,
    partial_o_ptr: *mut f32,
    num_heads_q: usize,
    num_heads_kv: usize,
    gqa_ratio: usize,
    head_dim: usize,
    capacity: usize,
    cache_seq_len: usize,
    chunk_size: usize,
    scale: f32,
}

// Safety: see struct docstring.  The raw pointer fields point to buffers
// whose lifetime is bounded by the `dispatch` call, and ownership regions
// are partitioned such that concurrent access is non-overlapping.
unsafe impl Send for FlashDecodeCtx {}
unsafe impl Sync for FlashDecodeCtx {}

/// Compute `[t_start, t_end)` for `chunk_idx`-th chunk of a sequence of
/// `cache_seq_len` tokens split at `chunk_size`-token granularity.
#[inline]
fn flash_chunk_range(chunk_idx: usize, cache_seq_len: usize, chunk_size: usize) -> (usize, usize) {
    let start = chunk_idx * chunk_size;
    let end = (start + chunk_size).min(cache_seq_len);
    (start, end)
}

/// Uniform-V fallback: `out_h = (1/seq_len) · Σ_t V[kv_h, t]`.
///
/// Used by the flash-decoding merge when every chunk partial is
/// `m_c = −∞` (all tokens for this Q-head were NaN) or the merged
/// denominator is non-finite / non-positive.  Semantics mirror the
/// head-parallel path's `fallback_uniform` branch.
///
/// # Safety
/// `v_ptr` must reference an F16 KV tensor of shape
/// `[_, _, capacity, head_dim]` with at least `cache_seq_len` valid
/// positions for `kv_h`; `out_h` must hold `head_dim` writable F32 lanes.
fn flash_uniform_fallback(
    out_h: &mut [f32],
    kv_h: usize,
    head_dim: usize,
    capacity: usize,
    cache_seq_len: usize,
    v_ptr: *const u16,
) {
    for x in out_h.iter_mut() {
        *x = 0.0;
    }
    if cache_seq_len == 0 {
        return;
    }
    let uniform = 1.0 / cache_seq_len as f32;
    unsafe {
        let o_ptr = out_h.as_mut_ptr();
        let w_v = vdupq_n_f32(uniform);
        for t in 0..cache_seq_len {
            let vp = v_ptr.add((kv_h * capacity + t) * head_dim);
            let mut i = 0;
            while i + 8 <= head_dim {
                let raw: uint16x8_t = vld1q_u16(vp.add(i));
                let fl: float32x4_t;
                let fh: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) fl, i = in(vreg) raw);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) fh, i = in(vreg) raw);
                vst1q_f32(o_ptr.add(i), vfmaq_f32(vld1q_f32(o_ptr.add(i)), w_v, fl));
                vst1q_f32(
                    o_ptr.add(i + 4),
                    vfmaq_f32(vld1q_f32(o_ptr.add(i + 4)), w_v, fh),
                );
                i += 8;
            }
            while i < head_dim {
                *o_ptr.add(i) += uniform * half::f16::from_bits(*vp.add(i)).to_f32();
                i += 1;
            }
        }
    }
}

impl CpuBackendNeon {
    /// NEON-optimized attention for F16 KV cache.
    ///
    /// Dispatches between two internal paths based on `cache_seq_len` and on
    /// whether post-softmax scores are requested:
    ///
    /// * **Head-parallel path** (`attention_gen_f16_neon_head_parallel`): Step 1
    ///   single-pass online softmax, one Q-head per rayon worker.  Optimal for
    ///   short contexts where GQA redundant KV reads are not the bottleneck.
    /// * **Flash-decoding path** (`attention_gen_f16_neon_flash`): Step 2
    ///   KV-chunk + kv_h parallelism via SpinPool.  All Q-heads in a GQA group
    ///   share the same K/V chunk load, reducing DRAM traffic by `gqa_ratio×`
    ///   at long contexts.
    ///
    /// Routing (see `arch/cpu_flash_decoding.md` §4.2, §7.2):
    /// * `cache_seq_len < FLASH_DECODE_THRESHOLD` → head-parallel.
    /// * `scores_out.is_some()` → head-parallel (post-softmax weights are
    ///   well-defined only with a global softmax, which the flash path does
    ///   not materialize per-token).
    /// * Otherwise → flash-decoding.
    #[allow(clippy::too_many_arguments)]
    fn attention_gen_f16_neon(
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
        if cache_seq_len < FLASH_DECODE_THRESHOLD || scores_out.is_some() {
            return Self::attention_gen_f16_neon_head_parallel(
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
        Self::attention_gen_f16_neon_flash(
            q,
            k_cache,
            v_cache,
            out,
            num_heads_q,
            num_heads_kv,
            head_dim,
            cache_seq_len,
        )
    }

    /// Step 1 head-parallel online-softmax path.
    ///
    /// Optimizations over the generic common.rs F16 path:
    /// 1. Direct F16·F32 dot via `vec_dot_f16_f32_4rows` — eliminates intermediate F32 buffer
    /// 2. Online softmax — K and V are scanned **once** in the same loop, eliminating
    ///    the 3-pass (QK^T → softmax → weighted V) structure. Running (m, l, o) state
    ///    is updated per token using the identity:
    ///        m_new = max(m, s)
    ///        rescale = exp(m - m_new)
    ///        l_new = l * rescale + exp(s - m_new)
    ///        o_new = o * rescale + V[t] * exp(s - m_new)
    ///    Final output = o / l.
    /// 3. Per-token NaN gate: NaN scores are treated as -inf (contribute 0) without
    ///    touching the running (m, l, o) state.
    #[allow(clippy::too_many_arguments)]
    fn attention_gen_f16_neon_head_parallel(
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
        let q_data = q.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = num_heads_q / num_heads_kv;

        let k_data = k_cache.as_slice::<half::f16>();
        let v_data = v_cache.as_slice::<half::f16>();

        // Wrap raw pointer for Send+Sync (Rayon par_chunks_mut capture).
        // Safety: each head h writes to non-overlapping region [h*stride .. h*stride+cache_seq_len].
        #[derive(Clone, Copy)]
        struct SendPtr(*mut f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        let scores_ptr = scores_out.as_ref().map(|s| SendPtr(s.as_ptr() as *mut f32));
        let scores_stride = scores_out
            .as_ref()
            .map(|s| s.len() / num_heads_q)
            .unwrap_or(0);
        let need_scores = scores_ptr.is_some();

        // Detect layout: HeadMajor [batch, kv_heads, capacity, head_dim]
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };

        // When diagnostics are requested we retain raw (pre-softmax) scores on the
        // stack for typical decode lengths, heap for longer sequences. Raw scores
        // are then converted to post-softmax weights in a short second pass using
        // the final running (m, l) from the online loop.
        const STACK_SCORES_MAX: usize = 4096;

        out_data
            .par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(h, out_h)| {
                let kv_h = h / gqa_ratio;
                let q_off = h * head_dim;
                let q_ptr = unsafe { q_data.as_ptr().add(q_off) };

                // Raw scores buffer only when diagnostics are needed.
                let mut stack_scores = [0.0f32; STACK_SCORES_MAX];
                let mut heap_scores: Vec<f32>;
                let raw_scores: Option<&mut [f32]> = if need_scores {
                    if cache_seq_len <= STACK_SCORES_MAX {
                        Some(&mut stack_scores[..cache_seq_len])
                    } else {
                        heap_scores = vec![0.0f32; cache_seq_len];
                        Some(&mut heap_scores[..])
                    }
                } else {
                    None
                };

                // Initialize accumulator `o` (out_h) to zero. Online softmax
                // will rescale and FMA V[t] into this buffer in-place.
                for x in out_h.iter_mut() {
                    *x = 0.0;
                }

                // Running softmax state.
                let mut m_run: f32 = f32::NEG_INFINITY;
                let mut l_run: f32 = 0.0;

                let k_f16_ptr = k_data.as_ptr() as *const u16;
                let v_f16_ptr = v_data.as_ptr() as *const u16;

                // Helper: apply a single (t, raw_s) to running state + V[t] FMA.
                // This closure-like sequence is inlined manually to keep `unsafe`
                // blocks tight and avoid closure capture.
                //
                // Step per token:
                //   if s is NaN or -inf: skip (no contribution, state untouched)
                //   m_new = max(m, s)
                //   alpha = exp(m - m_new)   (0 on first valid token since m=-inf)
                //   beta  = exp(s - m_new)
                //   l     = l * alpha + beta
                //   o     = o * alpha + V[t] * beta
                //   m     = m_new
                //
                // Vectorized o update: `o = o * alpha + V[t] * beta` — for each 4-lane
                // chunk, load V (F16→F32), multiply by beta, then FMA into alpha*o.
                let mut apply_token = |t: usize, raw_s: f32| {
                    // Skip non-finite logits without perturbing running state.
                    if !raw_s.is_finite() {
                        return;
                    }
                    let m_new = m_run.max(raw_s);
                    // On the very first valid token m_run is -inf so (m - m_new) = -inf,
                    // which gives alpha = 0.  The accumulator was pre-zeroed so this
                    // correctly discards the initial garbage (nothing to scale).
                    let alpha = if m_run.is_infinite() && m_run.is_sign_negative() {
                        0.0
                    } else {
                        (m_run - m_new).exp()
                    };
                    let beta = (raw_s - m_new).exp();
                    l_run = l_run * alpha + beta;
                    m_run = m_new;

                    // Rescale `o` by alpha and FMA in V[t] * beta.
                    let off = if is_head_major {
                        (kv_h * capacity + t) * head_dim
                    } else {
                        (t * num_heads_kv + kv_h) * head_dim
                    };
                    unsafe {
                        let vp = v_f16_ptr.add(off);
                        let o_ptr = out_h.as_mut_ptr();
                        let alpha_v = vdupq_n_f32(alpha);
                        let beta_v = vdupq_n_f32(beta);
                        let mut i = 0;
                        while i + 16 <= head_dim {
                            let raw0: uint16x8_t = vld1q_u16(vp.add(i));
                            let raw1: uint16x8_t = vld1q_u16(vp.add(i + 8));
                            let f0: float32x4_t;
                            let f1: float32x4_t;
                            let f2: float32x4_t;
                            let f3: float32x4_t;
                            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f0, i = in(vreg) raw0);
                            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f1, i = in(vreg) raw0);
                            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f2, i = in(vreg) raw1);
                            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f3, i = in(vreg) raw1);

                            // o = o * alpha + V * beta
                            let o0 = vld1q_f32(o_ptr.add(i));
                            let o1 = vld1q_f32(o_ptr.add(i + 4));
                            let o2 = vld1q_f32(o_ptr.add(i + 8));
                            let o3 = vld1q_f32(o_ptr.add(i + 12));
                            let o0s = vmulq_f32(o0, alpha_v);
                            let o1s = vmulq_f32(o1, alpha_v);
                            let o2s = vmulq_f32(o2, alpha_v);
                            let o3s = vmulq_f32(o3, alpha_v);
                            vst1q_f32(o_ptr.add(i), vfmaq_f32(o0s, beta_v, f0));
                            vst1q_f32(o_ptr.add(i + 4), vfmaq_f32(o1s, beta_v, f1));
                            vst1q_f32(o_ptr.add(i + 8), vfmaq_f32(o2s, beta_v, f2));
                            vst1q_f32(o_ptr.add(i + 12), vfmaq_f32(o3s, beta_v, f3));
                            i += 16;
                        }
                        while i + 8 <= head_dim {
                            let raw: uint16x8_t = vld1q_u16(vp.add(i));
                            let fl: float32x4_t;
                            let fh: float32x4_t;
                            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) fl, i = in(vreg) raw);
                            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) fh, i = in(vreg) raw);
                            let o0 = vld1q_f32(o_ptr.add(i));
                            let o1 = vld1q_f32(o_ptr.add(i + 4));
                            let o0s = vmulq_f32(o0, alpha_v);
                            let o1s = vmulq_f32(o1, alpha_v);
                            vst1q_f32(o_ptr.add(i), vfmaq_f32(o0s, beta_v, fl));
                            vst1q_f32(o_ptr.add(i + 4), vfmaq_f32(o1s, beta_v, fh));
                            i += 8;
                        }
                        while i < head_dim {
                            let v_f32 = half::f16::from_bits(*vp.add(i)).to_f32();
                            *o_ptr.add(i) = *o_ptr.add(i) * alpha + v_f32 * beta;
                            i += 1;
                        }
                    }
                };

                // --- Single-pass: Q·K[t] → online softmax update + V[t] FMA ---
                // Process 4 timesteps at a time via vec_dot_f16_f32_4rows for K
                // throughput; per-token online softmax is updated sequentially.
                let full_4 = cache_seq_len / 4;
                for chunk in 0..full_4 {
                    let t_base = chunk * 4;
                    let mut b_ptrs = [std::ptr::null::<u16>(); 4];
                    for (r, bp) in b_ptrs.iter_mut().enumerate() {
                        let t = t_base + r;
                        let off = if is_head_major {
                            (kv_h * capacity + t) * head_dim
                        } else {
                            (t * num_heads_kv + kv_h) * head_dim
                        };
                        *bp = unsafe { k_f16_ptr.add(off) };
                    }
                    let mut dots = [0.0f32; 4];
                    unsafe {
                        crate::quant::flash_neon::vec_dot_f16_f32_4rows(head_dim, q_ptr, b_ptrs, &mut dots);
                    }
                    for (r, &d) in dots.iter().enumerate() {
                        let s = d * scale;
                        let t = t_base + r;
                        if let Some(raw) = raw_scores.as_deref() {
                            // Safety: raw_scores has len == cache_seq_len.
                            unsafe {
                                *(raw.as_ptr() as *mut f32).add(t) = s;
                            }
                        }
                        apply_token(t, s);
                    }
                }
                for t in (full_4 * 4)..cache_seq_len {
                    let off = if is_head_major {
                        (kv_h * capacity + t) * head_dim
                    } else {
                        (t * num_heads_kv + kv_h) * head_dim
                    };
                    let k_ptr = unsafe { k_f16_ptr.add(off) };
                    let dot = unsafe { crate::quant::flash_neon::vec_dot_f16_f32(head_dim, q_ptr, k_ptr) };
                    let s = dot * scale;
                    if let Some(raw) = raw_scores.as_deref() {
                        unsafe {
                            *(raw.as_ptr() as *mut f32).add(t) = s;
                        }
                    }
                    apply_token(t, s);
                }

                // Finalize output = o / l (with fallback on pathological states).
                // m_run == -inf means every score was non-finite (all NaN) → uniform.
                // l_run non-finite / zero / negative also falls back to uniform.
                let fallback_uniform = !l_run.is_finite()
                    || l_run <= 0.0
                    || (m_run.is_infinite() && m_run.is_sign_negative());

                if fallback_uniform {
                    // Uniform average of V[t] over t, ignoring softmax.
                    // Matches the fallback semantics of the original 3-pass code.
                    for x in out_h.iter_mut() {
                        *x = 0.0;
                    }
                    let uniform = 1.0 / cache_seq_len as f32;
                    for t in 0..cache_seq_len {
                        let off = if is_head_major {
                            (kv_h * capacity + t) * head_dim
                        } else {
                            (t * num_heads_kv + kv_h) * head_dim
                        };
                        unsafe {
                            let vp = v_f16_ptr.add(off);
                            let o_ptr = out_h.as_mut_ptr();
                            let w_v = vdupq_n_f32(uniform);
                            let mut i = 0;
                            while i + 8 <= head_dim {
                                let raw: uint16x8_t = vld1q_u16(vp.add(i));
                                let fl: float32x4_t;
                                let fh: float32x4_t;
                                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) fl, i = in(vreg) raw);
                                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) fh, i = in(vreg) raw);
                                vst1q_f32(
                                    o_ptr.add(i),
                                    vfmaq_f32(vld1q_f32(o_ptr.add(i)), w_v, fl),
                                );
                                vst1q_f32(
                                    o_ptr.add(i + 4),
                                    vfmaq_f32(vld1q_f32(o_ptr.add(i + 4)), w_v, fh),
                                );
                                i += 8;
                            }
                            while i < head_dim {
                                *o_ptr.add(i) +=
                                    uniform * half::f16::from_bits(*vp.add(i)).to_f32();
                                i += 1;
                            }
                        }
                    }
                } else {
                    // Divide running o by l: out_h *= 1/l_run.
                    let inv_l = 1.0 / l_run;
                    unsafe {
                        let o_ptr = out_h.as_mut_ptr();
                        let inv_v = vdupq_n_f32(inv_l);
                        let mut i = 0;
                        while i + 4 <= head_dim {
                            let x = vld1q_f32(o_ptr.add(i));
                            vst1q_f32(o_ptr.add(i), vmulq_f32(x, inv_v));
                            i += 4;
                        }
                        while i < head_dim {
                            *o_ptr.add(i) *= inv_l;
                            i += 1;
                        }
                    }
                }

                // Diagnostic path: convert raw scores → post-softmax weights using
                // the final (m_run, l_run) state. This matches the contract of the
                // original 3-pass implementation (normalized probabilities).
                if let Some(SendPtr(ptr)) = scores_ptr {
                    let raw = raw_scores.expect("raw_scores set iff need_scores");
                    unsafe {
                        let dst = std::slice::from_raw_parts_mut(
                            ptr.add(h * scores_stride),
                            cache_seq_len,
                        );
                        if fallback_uniform {
                            let uniform = 1.0 / cache_seq_len as f32;
                            for w in dst.iter_mut() {
                                *w = uniform;
                            }
                        } else {
                            let inv_l = 1.0 / l_run;
                            for (i, w) in dst.iter_mut().enumerate() {
                                let s = raw[i];
                                if !s.is_finite() {
                                    *w = 0.0;
                                } else {
                                    *w = (s - m_run).exp() * inv_l;
                                }
                            }
                        }
                    }
                }
            });

        Ok(())
    }

    /// Step 2 flash-decoding path: KV-chunk + kv_h parallelism via SpinPool.
    ///
    /// Two-phase algorithm (see `arch/cpu_flash_decoding.md` §2):
    /// 1. **Per-chunk partial online softmax** (parallel): `n_chunks × num_heads_kv`
    ///    tasks dispatched to the SpinPool.  Each task iterates over every
    ///    Q-head in its GQA group (`gqa_ratio` heads) and for each Q-head
    ///    records a chunk-local triple `(m_c, l_c, Õ_c)` in shared scratch
    ///    buffers.  Because the inner loop reuses the same K/V chunk across
    ///    all Q-heads in the group, DRAM traffic shrinks by `gqa_ratio×`.
    /// 2. **LSE merge** (serial on main thread): for each Q-head combine the
    ///    partials back into a global softmax output:
    ///        M = max_c m_c,  L = Σ α_c · l_c,  O = Σ α_c · Õ_c,  out = O / L,
    ///    where `α_c = exp(m_c − M)` and `Õ_c = l_c · o_c` is the un-normalised
    ///    weighted-V accumulator the workers write out.  All-NaN chunks
    ///    (`m_c = −∞`, `l_c = 0`) contribute zero to both sums; if every
    ///    chunk is in that state the path falls back to a uniform-V average
    ///    matching the head-parallel semantics.
    ///
    /// The flash path never writes post-softmax diagnostic scores (caller must
    /// route `scores_out.is_some()` to `attention_gen_f16_neon_head_parallel`
    /// — this is enforced in the dispatcher).
    #[allow(clippy::too_many_arguments)]
    fn attention_gen_f16_neon_flash(
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
    ) -> Result<()> {
        debug_assert!(cache_seq_len >= FLASH_DECODE_THRESHOLD);
        debug_assert!(num_heads_q.is_multiple_of(num_heads_kv));

        let q_data = q.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();
        let k_data = k_cache.as_slice::<half::f16>();
        let v_data = v_cache.as_slice::<half::f16>();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = num_heads_q / num_heads_kv;

        // KV layout: HeadMajor [batch=1, kv_heads, capacity, head_dim].  The
        // flash path requires HeadMajor because chunk contiguity depends on
        // `(kv_h, t, d)` → offset being row-major in `t`.  (SeqMajor would
        // stride every chunk read by `num_heads_kv`, wiping out the DRAM
        // savings.)  Fall back to the head-parallel path when HeadMajor is
        // not detected.
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        if !is_head_major {
            return Self::attention_gen_f16_neon_head_parallel(
                q,
                k_cache,
                v_cache,
                out,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
                None,
            );
        }
        let capacity = k_shape[2];

        let chunk_size = FLASH_DECODE_CHUNK_SIZE;
        let n_chunks = cache_seq_len.div_ceil(chunk_size);

        // Scratch buffers.
        //
        // `partial_ml[chunk_idx * num_heads_q + q_h]` — scalar (m, l) pair
        //   (cache-line aligned; see `PartialState`).
        // `partial_o[chunk_idx * num_heads_q * head_dim + q_h * head_dim + d]`
        //   — un-normalised weighted-V accumulator Õ_c.
        let mut partial_ml: Vec<PartialState> = vec![PartialState::EMPTY; n_chunks * num_heads_q];
        let mut partial_o: Vec<f32> = vec![0.0f32; n_chunks * num_heads_q * head_dim];

        let ctx = FlashDecodeCtx {
            q_ptr: q_data.as_ptr(),
            k_ptr: k_data.as_ptr() as *const u16,
            v_ptr: v_data.as_ptr() as *const u16,
            partial_ml_ptr: partial_ml.as_mut_ptr(),
            partial_o_ptr: partial_o.as_mut_ptr(),
            num_heads_q,
            num_heads_kv,
            gqa_ratio,
            head_dim,
            capacity,
            cache_seq_len,
            chunk_size,
            scale,
        };

        let n_tasks = n_chunks * num_heads_kv;
        let pool = thread_pool::get_pool();
        // Safety:
        // * `flash_chunk_worker` interprets the `*const u8` ctx as
        //   `*const FlashDecodeCtx` (which is what we pass).
        // * `ctx` lives on the stack here and is borrowed immutably by
        //   `dispatch` (blocks until all tasks complete).
        // * Per-task writes are partitioned by `(chunk_idx, kv_h)` →
        //   `(chunk_idx, q_h ∈ gqa group of kv_h)`, so no two workers touch
        //   the same `partial_ml` or `partial_o` element.
        unsafe {
            pool.dispatch(
                n_tasks,
                Self::flash_chunk_worker as WorkFn,
                &ctx as *const FlashDecodeCtx as *const u8,
            );
        }

        // Phase 2: LSE merge (serial, main thread).
        Self::flash_merge_partials(
            &partial_ml,
            &partial_o,
            out_data,
            num_heads_q,
            gqa_ratio,
            head_dim,
            capacity,
            n_chunks,
            cache_seq_len,
            v_data.as_ptr() as *const u16,
        );

        Ok(())
    }

    /// SpinPool worker for Phase 1 of the flash-decoding path.
    ///
    /// `task_id` decodes as `(chunk_idx, kv_h) = (task_id / num_heads_kv,
    /// task_id % num_heads_kv)`.  The worker processes one `(chunk, kv_h)`
    /// tile across all `gqa_ratio` Q-heads in that kv-group, using per-token
    /// online softmax to produce `(m_c, l_c, Õ_c)` partials in the shared
    /// scratch buffers.
    ///
    /// # Safety
    /// `ctx` must point to a live `FlashDecodeCtx`.  `task_id` must lie in
    /// `0..n_chunks * num_heads_kv`.  The referenced tensor buffers must
    /// remain valid for the duration of the dispatch.  The partial output
    /// regions are written to exclusively by this task.
    unsafe fn flash_chunk_worker(ctx: *const u8, task_id: usize) {
        unsafe {
            let ctx = &*(ctx as *const FlashDecodeCtx);
            let chunk_idx = task_id / ctx.num_heads_kv;
            let kv_h = task_id % ctx.num_heads_kv;
            let (t_start, t_end) = flash_chunk_range(chunk_idx, ctx.cache_seq_len, ctx.chunk_size);
            debug_assert!(t_end > t_start);

            let q_h_begin = kv_h * ctx.gqa_ratio;
            let q_h_end = q_h_begin + ctx.gqa_ratio;
            let head_dim = ctx.head_dim;

            // Pre-zero the `o_tilde` scratch for each Q-head in this tile so
            // the online-softmax loop can FMA directly into it.
            for q_h in q_h_begin..q_h_end {
                let o_base = ctx
                    .partial_o_ptr
                    .add((chunk_idx * ctx.num_heads_q + q_h) * head_dim);
                std::ptr::write_bytes(o_base, 0, head_dim);
            }

            // Each Q-head scans the entire `[t_start, t_end)` range once.
            // This preserves the GQA-group inner loop so the chunk's K/V
            // stays resident in L1/L2 across all `gqa_ratio` Q-heads.
            for q_h in q_h_begin..q_h_end {
                let q_off = q_h * head_dim;
                let q_ptr = ctx.q_ptr.add(q_off);
                let o_base = ctx
                    .partial_o_ptr
                    .add((chunk_idx * ctx.num_heads_q + q_h) * head_dim);

                let mut m_run: f32 = f32::NEG_INFINITY;
                let mut l_run: f32 = 0.0;

                // Process 4 timesteps at a time via vec_dot_f16_f32_4rows.
                let chunk_len = t_end - t_start;
                let full_4 = chunk_len / 4;
                for block in 0..full_4 {
                    let t_base = t_start + block * 4;
                    let mut b_ptrs = [std::ptr::null::<u16>(); 4];
                    for (r, bp) in b_ptrs.iter_mut().enumerate() {
                        let t = t_base + r;
                        let off = (kv_h * ctx.capacity + t) * head_dim;
                        *bp = ctx.k_ptr.add(off);
                    }
                    let mut dots = [0.0f32; 4];
                    crate::quant::flash_neon::vec_dot_f16_f32_4rows(
                        head_dim, q_ptr, b_ptrs, &mut dots,
                    );
                    for (r, &d) in dots.iter().enumerate() {
                        let t = t_base + r;
                        let s = d * ctx.scale;
                        crate::quant::flash_neon::flash_apply_token(
                            s,
                            t,
                            kv_h,
                            ctx.capacity,
                            head_dim,
                            ctx.v_ptr,
                            o_base,
                            &mut m_run,
                            &mut l_run,
                        );
                    }
                }
                for t in (t_start + full_4 * 4)..t_end {
                    let off = (kv_h * ctx.capacity + t) * head_dim;
                    let k_ptr = ctx.k_ptr.add(off);
                    let dot = crate::quant::flash_neon::vec_dot_f16_f32(head_dim, q_ptr, k_ptr);
                    let s = dot * ctx.scale;
                    crate::quant::flash_neon::flash_apply_token(
                        s,
                        t,
                        kv_h,
                        ctx.capacity,
                        head_dim,
                        ctx.v_ptr,
                        o_base,
                        &mut m_run,
                        &mut l_run,
                    );
                }

                // Record chunk-local partial.  `o_base` already holds Õ_c.
                let slot = ctx.partial_ml_ptr.add(chunk_idx * ctx.num_heads_q + q_h);
                (*slot).m = m_run;
                (*slot).l = l_run;
            }
        }
    }

    /// Phase 2 of flash-decoding: combine chunk partials back into a global
    /// softmax output per Q-head.
    ///
    /// For each Q-head `q_h`, reads the `n_chunks` partials
    /// `(m_c, l_c, Õ_c)` and writes:
    /// ```text
    ///   M = max_c m_c
    ///   L = Σ_c exp(m_c − M) · l_c
    ///   O = Σ_c exp(m_c − M) · Õ_c          (un-normalised weighted V)
    ///   out[q_h] = O / L
    /// ```
    ///
    /// Fallback: if every partial has `m_c = −∞` (i.e. every token in every
    /// chunk was NaN) we emit the uniform-V average, mirroring the head-
    /// parallel path's `fallback_uniform` branch.
    #[allow(clippy::too_many_arguments)]
    fn flash_merge_partials(
        partial_ml: &[PartialState],
        partial_o: &[f32],
        out_data: &mut [f32],
        num_heads_q: usize,
        gqa_ratio: usize,
        head_dim: usize,
        capacity: usize,
        n_chunks: usize,
        cache_seq_len: usize,
        v_ptr: *const u16,
    ) {
        for q_h in 0..num_heads_q {
            let kv_h = q_h / gqa_ratio;
            let out_h = &mut out_data[q_h * head_dim..(q_h + 1) * head_dim];

            // Global max across finite chunk partials.
            let mut m_max = f32::NEG_INFINITY;
            for c in 0..n_chunks {
                let m_c = partial_ml[c * num_heads_q + q_h].m;
                if m_c.is_finite() && m_c > m_max {
                    m_max = m_c;
                }
            }

            if m_max == f32::NEG_INFINITY {
                // All chunks dead → uniform-V fallback matching Step 1
                // semantics: out_h = (1/seq_len) · Σ_t V[kv_h, t].
                flash_uniform_fallback(out_h, kv_h, head_dim, capacity, cache_seq_len, v_ptr);
                continue;
            }

            // Accumulate O and L via LSE.
            unsafe {
                let o_ptr = out_h.as_mut_ptr();
                // Zero O before summation.
                std::ptr::write_bytes(o_ptr, 0, head_dim);

                let mut l_total: f32 = 0.0;
                for c in 0..n_chunks {
                    let ml = partial_ml[c * num_heads_q + q_h];
                    if !ml.m.is_finite() {
                        continue;
                    }
                    let alpha = (ml.m - m_max).exp();
                    l_total += alpha * ml.l;

                    let o_c_ptr = partial_o.as_ptr().add((c * num_heads_q + q_h) * head_dim);
                    let alpha_v = vdupq_n_f32(alpha);
                    let mut i = 0;
                    while i + 4 <= head_dim {
                        let o_cur = vld1q_f32(o_ptr.add(i));
                        let o_add = vld1q_f32(o_c_ptr.add(i));
                        vst1q_f32(o_ptr.add(i), vfmaq_f32(o_cur, alpha_v, o_add));
                        i += 4;
                    }
                    while i < head_dim {
                        *o_ptr.add(i) += alpha * *o_c_ptr.add(i);
                        i += 1;
                    }
                }

                // Final normalisation or fallback on pathological L.
                if !l_total.is_finite() || l_total <= 0.0 {
                    flash_uniform_fallback(out_h, kv_h, head_dim, capacity, cache_seq_len, v_ptr);
                } else {
                    let inv_l = 1.0 / l_total;
                    let inv_v = vdupq_n_f32(inv_l);
                    let mut i = 0;
                    while i + 4 <= head_dim {
                        let x = vld1q_f32(o_ptr.add(i));
                        vst1q_f32(o_ptr.add(i), vmulq_f32(x, inv_v));
                        i += 4;
                    }
                    while i < head_dim {
                        *o_ptr.add(i) *= inv_l;
                        i += 1;
                    }
                }
            }
        }
    }

    fn matmul_transposed_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        if a_rank < 2 || b_rank < 2 {
            return Err(anyhow!("Tensors must be at least 2D for matmul"));
        }

        let k = a_shape[a_rank - 1];
        let k_b = b_shape[b_rank - 1];
        if k != k_b {
            return Err(anyhow!("Shape mismatch"));
        }

        let m: usize = a_shape[..a_rank - 1].iter().product();
        let n = b_shape[b_rank - 2];

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();

        // Heuristic: Serial for tiny matrices (Proven 3.6x faster for [1, 128, 256])
        if (m * n * k) < 100_000 {
            return self.matmul_transposed_f32_serial(a, b, out);
        }

        // Optimization: N-parallel for all M
        // Adaptive chunking: split N into exactly `num_threads` chunks to minimize Rayon overhead
        // for "medium" workloads (like M=1 generation).
        let num_threads = rayon::current_num_threads();
        let chunk_size = n.div_ceil(num_threads);
        let chunk_size = chunk_size.max(256); // Ensure at least 256 elements per task

        out_data
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * chunk_size;
                for (local_i, out_val) in chunk.iter_mut().enumerate() {
                    let idx = start_idx + local_i;
                    let i = idx / n;
                    let j = idx % n;

                    let a_ptr = unsafe { a_data.as_ptr().add(i * k) };
                    let b_ptr = unsafe { b_data.as_ptr().add(j * k) };

                    let mut k_idx = 0;

                    unsafe {
                        let mut sum_v = vdupq_n_f32(0.0);

                        while k_idx + 16 <= k {
                            let va0 = vld1q_f32(a_ptr.add(k_idx));
                            let vb0 = vld1q_f32(b_ptr.add(k_idx));
                            sum_v = vfmaq_f32(sum_v, va0, vb0);

                            let va1 = vld1q_f32(a_ptr.add(k_idx + 4));
                            let vb1 = vld1q_f32(b_ptr.add(k_idx + 4));
                            sum_v = vfmaq_f32(sum_v, va1, vb1);

                            let va2 = vld1q_f32(a_ptr.add(k_idx + 8));
                            let vb2 = vld1q_f32(b_ptr.add(k_idx + 8));
                            sum_v = vfmaq_f32(sum_v, va2, vb2);

                            let va3 = vld1q_f32(a_ptr.add(k_idx + 12));
                            let vb3 = vld1q_f32(b_ptr.add(k_idx + 12));
                            sum_v = vfmaq_f32(sum_v, va3, vb3);

                            k_idx += 16;
                        }

                        while k_idx + 4 <= k {
                            let va = vld1q_f32(a_ptr.add(k_idx));
                            let vb = vld1q_f32(b_ptr.add(k_idx));
                            sum_v = vfmaq_f32(sum_v, va, vb);
                            k_idx += 4;
                        }

                        // Reduction
                        let sum_s = vaddvq_f32(sum_v);
                        let mut sum = sum_s;

                        // Tail
                        while k_idx < k {
                            sum += *a_ptr.add(k_idx) * *b_ptr.add(k_idx);
                            k_idx += 1;
                        }
                        *out_val = sum;
                    }
                }
            });
        Ok(())
    }

    // Pure serial implementation for tiny matrices
    fn matmul_transposed_f32_serial(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 2];
        let m: usize = a_shape[..a_shape.len() - 1].iter().product();

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();

        for i in 0..m {
            let a_offset = i * k;
            for j in 0..n {
                let b_offset = j * k;
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a_data[a_offset + l] * b_data[b_offset + l];
                }
                out_data[i * n + j] = sum;
            }
        }
        Ok(())
    }

    /// NEON F16 matmul: A(F32) x B^T(F16) -> Out(F32)
    /// Multi-row GEMV: processes NR=4 output rows simultaneously,
    /// sharing activation loads across rows for better ILP and reduced overhead.
    fn matmul_transposed_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let n = b_shape[b_rank - 2];
        let m: usize = a_shape[..a_rank - 1].iter().product();

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<half::f16>();
        let out_data = out.as_mut_slice::<f32>();

        // M=1 decode: serial path uses F32×F16 dot directly (vec_dot_f16_f32_4rows),
        // avoiding F32→F16 conversion + heap alloc + SpinPool dispatch overhead.
        // Threshold: n*k < 2M covers Wk/Wv/Wq/Wo but keeps FFN gate/up/down parallel.
        let serial_threshold = if m == 1 { 2_000_000 } else { 100_000 };
        if (m * n * k) < serial_threshold {
            return self.matmul_transposed_f16_serial(a_data, b_data, out_data, m, n, k);
        }

        const NR: usize = 4;

        if USE_RAYON.load(AtomicOrdering::Relaxed) {
            // Rayon path: par_chunks_mut for A/B benchmarking vs SpinPool
            let b_ptr_base = b_data.as_ptr() as usize; // usize for Send
            for i in 0..m {
                let a_ptr = unsafe { a_data.as_ptr().add(i * k) } as usize;
                let row_out = &mut out_data[i * n..(i + 1) * n];
                row_out
                    .par_chunks_mut(NR)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let j = chunk_idx * NR;
                        let bp = b_ptr_base as *const u16;
                        let ap = a_ptr as *const f32;
                        if chunk.len() == NR {
                            let b_ptrs = [
                                unsafe { bp.add(j * k) },
                                unsafe { bp.add((j + 1) * k) },
                                unsafe { bp.add((j + 2) * k) },
                                unsafe { bp.add((j + 3) * k) },
                            ];
                            let mut results = [0.0f32; NR];
                            unsafe {
                                crate::quant::flash_neon::vec_dot_f16_f32_4rows(
                                    k,
                                    ap,
                                    b_ptrs,
                                    &mut results,
                                )
                            };
                            chunk.copy_from_slice(&results);
                        } else {
                            for t in 0..chunk.len() {
                                chunk[t] = unsafe {
                                    crate::quant::flash_neon::vec_dot_f16_f32(
                                        k,
                                        ap,
                                        bp.add((j + t) * k),
                                    )
                                };
                            }
                        }
                    });
            }
        } else {
            // SpinPool path (default)
            let pool = thread_pool::get_pool();

            let n_threads = rayon::current_num_threads();
            let target_chunks = n_threads * 8;
            let rows_per_chunk = n.div_ceil(target_chunks).div_ceil(NR) * NR;
            let rows_per_chunk = rows_per_chunk.max(NR);
            let n_chunks = n.div_ceil(rows_per_chunk);

            // Pre-convert A from F32 to F16 for the F16 matmul kernels
            // (one-time cost per forward-pass call; amortized across N).
            let mut a_f16_buf: Vec<u16> = vec![0u16; m * k];
            unsafe {
                for i in 0..m {
                    Self::f32_to_f16_neon(
                        a_data.as_ptr().add(i * k),
                        a_f16_buf.as_mut_ptr().add(i * k),
                        k,
                    );
                }
            }

            if m > 1 {
                // N-major GEMM with MR=4: groups of 4 A rows share B loads via F16 FMA.
                // chunk_id = n_chunk_idx * m_groups + m_group, where m_groups = ceil(m/4)
                let m_groups = m.div_ceil(4);
                let total_tasks = n_chunks * m_groups;
                let ctx = F16GemmCtx {
                    a_base: a_data.as_ptr(),
                    a_f16_base: a_f16_buf.as_ptr(),
                    b_base: b_data.as_ptr() as *const u16,
                    out_base: out_data.as_mut_ptr(),
                    m,
                    n,
                    k,
                    rows_per_chunk,
                };
                unsafe {
                    pool.dispatch(
                        total_tasks,
                        f16_gemm_chunk,
                        &ctx as *const F16GemmCtx as *const u8,
                    );
                }
            } else {
                // M=1: decode GEMV path (native F16 FMA)
                let ctx = F16GemvCtx {
                    a_ptr: a_data.as_ptr(),
                    a_f16_ptr: a_f16_buf.as_ptr(),
                    b_base: b_data.as_ptr() as *const u16,
                    out_ptr: out_data.as_mut_ptr(),
                    k,
                    n,
                    rows_per_chunk,
                };
                unsafe {
                    pool.dispatch(
                        n_chunks,
                        f16_gemv_chunk,
                        &ctx as *const F16GemvCtx as *const u8,
                    );
                }
            }
        }

        Ok(())
    }

    fn matmul_transposed_f16_serial(
        &self,
        a_data: &[f32],
        b_data: &[half::f16],
        out_data: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        const NR: usize = 4;
        let b_ptr_base = b_data.as_ptr() as *const u16;

        for i in 0..m {
            let a_ptr = unsafe { a_data.as_ptr().add(i * k) };
            let mut j = 0;
            while j + NR <= n {
                let b_ptrs = [
                    unsafe { b_ptr_base.add(j * k) },
                    unsafe { b_ptr_base.add((j + 1) * k) },
                    unsafe { b_ptr_base.add((j + 2) * k) },
                    unsafe { b_ptr_base.add((j + 3) * k) },
                ];
                let mut results = [0.0f32; NR];
                unsafe {
                    crate::quant::flash_neon::vec_dot_f16_f32_4rows(k, a_ptr, b_ptrs, &mut results);
                }
                out_data[i * n + j..i * n + j + NR].copy_from_slice(&results);
                j += NR;
            }
            while j < n {
                let b_ptr = unsafe { (b_data.as_ptr() as *const u16).add(j * k) };
                out_data[i * n + j] =
                    unsafe { crate::quant::flash_neon::vec_dot_f16_f32(k, a_ptr, b_ptr) };
                j += 1;
            }
        }
        Ok(())
    }

    // --- NEON vectorized exp/silu/swiglu (ported from llama.cpp ggml_v_expf) ---

    /// NEON vectorized exp(x) — polynomial approximation from ARM optimized routine.
    /// Maximum error: 1.45358 + 0.5 ULP. Matches llama.cpp's ggml_v_expf().
    #[inline(always)]
    unsafe fn v_expf(x: float32x4_t) -> float32x4_t {
        unsafe {
            let r = vdupq_n_f32(f32::from_bits(0x4b00_0000)); // 0x1.8p23
            let z = vfmaq_f32(r, x, vdupq_n_f32(f32::from_bits(0x3fb8_aa3b))); // 0x1.715476p+0
            let n = vsubq_f32(z, r);
            let b = vfmsq_f32(
                vfmsq_f32(x, n, vdupq_n_f32(f32::from_bits(0x3eb1_7200))), // 0x1.62e4p-1
                n,
                vdupq_n_f32(f32::from_bits(0x35bf_be8e)), // 0x1.7f7d1cp-20
            );
            let e = vshlq_n_u32::<23>(vreinterpretq_u32_f32(z));
            let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
            let c = vcagtq_f32(n, vdupq_n_f32(126.0));
            let u = vmulq_f32(b, b);
            let j = vfmaq_f32(
                vmulq_f32(vdupq_n_f32(f32::from_bits(0x3f7f_fff6)), b), // 0x1.ffffecp-1
                vfmaq_f32(
                    vfmaq_f32(
                        vdupq_n_f32(f32::from_bits(0x3f7f_fedb)), // 0x1.fffdb6p-2
                        vdupq_n_f32(f32::from_bits(0x3e2a_af33)), // 0x1.555e66p-3
                        b,
                    ),
                    vfmaq_f32(
                        vdupq_n_f32(f32::from_bits(0x3d2b_9f17)), // 0x1.573e2ep-5
                        vdupq_n_f32(f32::from_bits(0x3c07_2010)), // 0x1.0e4020p-7
                        b,
                    ),
                    u,
                ),
                u,
            );
            // Fast path: no overflow/underflow
            if vpaddd_u64(vreinterpretq_u64_u32(c)) == 0 {
                return vfmaq_f32(k, j, k);
            }
            // Slow path: handle overflow/underflow
            let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x8200_0000));
            let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f00_0000)));
            let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
            vbslq_f32(
                vcagtq_f32(n, vdupq_n_f32(192.0)),
                vmulq_f32(s1, s1),
                vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
            )
        }
    }

    /// NEON vectorized silu: x / (1 + exp(-x))
    /// Uses scalar exp() per lane for correctness on all ARM64 implementations.
    #[inline(always)]
    unsafe fn v_silu(x: float32x4_t) -> float32x4_t {
        unsafe {
            // Scalar exp per lane — avoids v_expf bit-manipulation issues on Apple Silicon
            let mut vals = [0.0f32; 4];
            vst1q_f32(vals.as_mut_ptr(), x);
            for v in vals.iter_mut() {
                *v = *v / (1.0 + (-*v).exp());
            }
            vld1q_f32(vals.as_ptr())
        }
    }

    /// NEON vectorized swiglu: silu(a) * b
    #[target_feature(enable = "neon")]
    unsafe fn swiglu_neon(a: &mut [f32], b: &[f32]) {
        unsafe {
            let n = a.len();
            let mut i = 0;
            while i + 4 <= n {
                let av = vld1q_f32(a.as_ptr().add(i));
                let bv = vld1q_f32(b.as_ptr().add(i));
                let result = vmulq_f32(Self::v_silu(av), bv);
                vst1q_f32(a.as_mut_ptr().add(i), result);
                i += 4;
            }
            // Scalar tail
            while i < n {
                let x = a[i];
                a[i] = (x / (1.0 + (-x).exp())) * b[i];
                i += 1;
            }
        }
    }

    /// NEON vectorized tanh — scalar per lane for correctness on all ARM64.
    /// Uses scalar tanh() to avoid v_expf bit-manipulation issues
    /// (same approach as v_silu — see Apple Silicon / Cortex-A720 errata).
    #[inline(always)]
    unsafe fn v_tanh(x: float32x4_t) -> float32x4_t {
        unsafe {
            let mut vals = [0.0f32; 4];
            vst1q_f32(vals.as_mut_ptr(), x);
            for v in vals.iter_mut() {
                *v = v.tanh();
            }
            vld1q_f32(vals.as_ptr())
        }
    }

    /// NEON vectorized GELU-tanh-mul: gate[i] = gelu_tanh(gate[i]) * up[i]
    /// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #[target_feature(enable = "neon")]
    unsafe fn gelu_tanh_mul_neon(gate: &mut [f32], up: &[f32]) {
        unsafe {
            let n = gate.len();
            let sqrt_2_over_pi = vdupq_n_f32((2.0_f32 / std::f32::consts::PI).sqrt());
            let coeff = vdupq_n_f32(0.044715);
            let half = vdupq_n_f32(0.5);
            let one = vdupq_n_f32(1.0);

            let mut i = 0;
            while i + 4 <= n {
                let x = vld1q_f32(gate.as_ptr().add(i));
                let u = vld1q_f32(up.as_ptr().add(i));

                // x^3
                let x2 = vmulq_f32(x, x);
                let x3 = vmulq_f32(x2, x);
                // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
                let inner = vmulq_f32(sqrt_2_over_pi, vfmaq_f32(x, coeff, x3));
                // 0.5 * x * (1 + tanh(inner)) * up
                let tanh_val = Self::v_tanh(inner);
                let result = vmulq_f32(vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_val)), u);
                vst1q_f32(gate.as_mut_ptr().add(i), result);
                i += 4;
            }
            // Scalar tail
            let sqrt_2_over_pi_s: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
            while i < n {
                let x = gate[i];
                let inner = sqrt_2_over_pi_s * (x + 0.044715 * x * x * x);
                gate[i] = 0.5 * x * (1.0 + inner.tanh()) * up[i];
                i += 1;
            }
        }
    }

    /// NEON vectorized RMS norm with add_unit support.
    /// Pass 1: sum of squares (2-accumulator unroll)
    /// Pass 2: x[i] = x[i] * scale * w_eff, where w_eff = (1+w[i]) or w[i]
    #[target_feature(enable = "neon")]
    unsafe fn rms_norm_neon(row: &mut [f32], w: &[f32], eps: f32, add_unit: bool) {
        unsafe {
            let dim = row.len();
            let ptr = row.as_mut_ptr();
            let w_ptr = w.as_ptr();

            // Pass 1: sum of squares with 2-accumulator unroll
            let mut sum_v0 = vdupq_n_f32(0.0);
            let mut sum_v1 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 8 <= dim {
                let v0 = vld1q_f32(ptr.add(i));
                let v1 = vld1q_f32(ptr.add(i + 4));
                sum_v0 = vfmaq_f32(sum_v0, v0, v0);
                sum_v1 = vfmaq_f32(sum_v1, v1, v1);
                i += 8;
            }
            while i + 4 <= dim {
                let v = vld1q_f32(ptr.add(i));
                sum_v0 = vfmaq_f32(sum_v0, v, v);
                i += 4;
            }
            sum_v0 = vaddq_f32(sum_v0, sum_v1);
            let mut sum_sq = vaddvq_f32(sum_v0);
            while i < dim {
                sum_sq += *ptr.add(i) * *ptr.add(i);
                i += 1;
            }

            let scale = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
            let scale_v = vdupq_n_f32(scale);

            // Pass 2: x[i] = x[i] * scale * w_eff
            i = 0;
            if add_unit {
                let one_v = vdupq_n_f32(1.0);
                while i + 4 <= dim {
                    let v = vld1q_f32(ptr.add(i));
                    let wv = vld1q_f32(w_ptr.add(i));
                    let w_eff = vaddq_f32(one_v, wv);
                    vst1q_f32(ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), w_eff));
                    i += 4;
                }
                while i < dim {
                    *ptr.add(i) = (*ptr.add(i) * scale) * (1.0 + *w_ptr.add(i));
                    i += 1;
                }
            } else {
                while i + 4 <= dim {
                    let v = vld1q_f32(ptr.add(i));
                    let wv = vld1q_f32(w_ptr.add(i));
                    vst1q_f32(ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), wv));
                    i += 4;
                }
                while i < dim {
                    *ptr.add(i) = (*ptr.add(i) * scale) * *w_ptr.add(i);
                    i += 1;
                }
            }
        }
    }

    /// NEON single-pass out-of-place RMS norm.
    /// Pass 1: read x → copy to out, compute sum-of-squares (saves one memcpy pass)
    /// Pass 2: out[i] = out[i] * scale * w_eff
    #[target_feature(enable = "neon")]
    unsafe fn rms_norm_oop_neon(x: &[f32], out: &mut [f32], w: &[f32], eps: f32, add_unit: bool) {
        unsafe {
            let dim = x.len();
            let x_ptr = x.as_ptr();
            let out_ptr = out.as_mut_ptr();
            let w_ptr = w.as_ptr();

            // Pass 1: copy x → out, accumulate sum-of-squares
            let mut sum_v0 = vdupq_n_f32(0.0);
            let mut sum_v1 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 8 <= dim {
                let v0 = vld1q_f32(x_ptr.add(i));
                let v1 = vld1q_f32(x_ptr.add(i + 4));
                vst1q_f32(out_ptr.add(i), v0);
                vst1q_f32(out_ptr.add(i + 4), v1);
                sum_v0 = vfmaq_f32(sum_v0, v0, v0);
                sum_v1 = vfmaq_f32(sum_v1, v1, v1);
                i += 8;
            }
            while i + 4 <= dim {
                let v = vld1q_f32(x_ptr.add(i));
                vst1q_f32(out_ptr.add(i), v);
                sum_v0 = vfmaq_f32(sum_v0, v, v);
                i += 4;
            }
            sum_v0 = vaddq_f32(sum_v0, sum_v1);
            let mut sum_sq = vaddvq_f32(sum_v0);
            while i < dim {
                let val = *x_ptr.add(i);
                *out_ptr.add(i) = val;
                sum_sq += val * val;
                i += 1;
            }

            let scale = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
            let scale_v = vdupq_n_f32(scale);

            // Pass 2: out[i] *= scale * w_eff
            i = 0;
            if add_unit {
                let one_v = vdupq_n_f32(1.0);
                while i + 4 <= dim {
                    let v = vld1q_f32(out_ptr.add(i));
                    let wv = vld1q_f32(w_ptr.add(i));
                    let w_eff = vaddq_f32(one_v, wv);
                    vst1q_f32(out_ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), w_eff));
                    i += 4;
                }
                while i < dim {
                    *out_ptr.add(i) = (*out_ptr.add(i) * scale) * (1.0 + *w_ptr.add(i));
                    i += 1;
                }
            } else {
                while i + 4 <= dim {
                    let v = vld1q_f32(out_ptr.add(i));
                    let wv = vld1q_f32(w_ptr.add(i));
                    vst1q_f32(out_ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), wv));
                    i += 4;
                }
                while i < dim {
                    *out_ptr.add(i) = (*out_ptr.add(i) * scale) * *w_ptr.add(i);
                    i += 1;
                }
            }
        }
    }

    /// NEON fused add + out-of-place RMS norm.
    /// Pass 1: x[i] += res[i], copy sum to out, accumulate sum-of-squares
    /// Pass 2: out[i] = out[i] * scale * w_eff
    #[target_feature(enable = "neon")]
    unsafe fn add_rms_norm_oop_neon(
        x: &mut [f32],
        res: &[f32],
        out: &mut [f32],
        w: &[f32],
        eps: f32,
        add_unit: bool,
    ) {
        unsafe {
            let dim = x.len();
            let x_ptr = x.as_mut_ptr();
            let res_ptr = res.as_ptr();
            let out_ptr = out.as_mut_ptr();
            let w_ptr = w.as_ptr();

            // Pass 1: x += res, copy to out, accumulate sum-of-squares
            let mut sum_v0 = vdupq_n_f32(0.0);
            let mut sum_v1 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 8 <= dim {
                let xv0 = vld1q_f32(x_ptr.add(i));
                let xv1 = vld1q_f32(x_ptr.add(i + 4));
                let rv0 = vld1q_f32(res_ptr.add(i));
                let rv1 = vld1q_f32(res_ptr.add(i + 4));
                let s0 = vaddq_f32(xv0, rv0);
                let s1 = vaddq_f32(xv1, rv1);
                vst1q_f32(x_ptr.add(i), s0);
                vst1q_f32(x_ptr.add(i + 4), s1);
                vst1q_f32(out_ptr.add(i), s0);
                vst1q_f32(out_ptr.add(i + 4), s1);
                sum_v0 = vfmaq_f32(sum_v0, s0, s0);
                sum_v1 = vfmaq_f32(sum_v1, s1, s1);
                i += 8;
            }
            while i + 4 <= dim {
                let xv = vld1q_f32(x_ptr.add(i));
                let rv = vld1q_f32(res_ptr.add(i));
                let s = vaddq_f32(xv, rv);
                vst1q_f32(x_ptr.add(i), s);
                vst1q_f32(out_ptr.add(i), s);
                sum_v0 = vfmaq_f32(sum_v0, s, s);
                i += 4;
            }
            sum_v0 = vaddq_f32(sum_v0, sum_v1);
            let mut sum_sq = vaddvq_f32(sum_v0);
            while i < dim {
                let val = *x_ptr.add(i) + *res_ptr.add(i);
                *x_ptr.add(i) = val;
                *out_ptr.add(i) = val;
                sum_sq += val * val;
                i += 1;
            }

            let scale = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
            let scale_v = vdupq_n_f32(scale);

            // Pass 2: out[i] *= scale * w_eff
            i = 0;
            if add_unit {
                let one_v = vdupq_n_f32(1.0);
                while i + 4 <= dim {
                    let v = vld1q_f32(out_ptr.add(i));
                    let wv = vld1q_f32(w_ptr.add(i));
                    let w_eff = vaddq_f32(one_v, wv);
                    vst1q_f32(out_ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), w_eff));
                    i += 4;
                }
                while i < dim {
                    *out_ptr.add(i) = (*out_ptr.add(i) * scale) * (1.0 + *w_ptr.add(i));
                    i += 1;
                }
            } else {
                while i + 4 <= dim {
                    let v = vld1q_f32(out_ptr.add(i));
                    let wv = vld1q_f32(w_ptr.add(i));
                    vst1q_f32(out_ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), wv));
                    i += 4;
                }
                while i < dim {
                    *out_ptr.add(i) = (*out_ptr.add(i) * scale) * *w_ptr.add(i);
                    i += 1;
                }
            }
        }
    }

    // --- F16 matmul kernels: native F16 FMA (FMLA .8H) with F16 accumulators ---

    /// Convert F32 slice to F16 using NEON fcvtn. One-time cost before the
    /// F16 inner loop (matmul_transposed_f16 pre-converts A row).
    #[target_feature(enable = "neon")]
    unsafe fn f32_to_f16_neon(src: *const f32, dst: *mut u16, len: usize) {
        unsafe {
            let mut i = 0;
            while i + 8 <= len {
                let v0 = vld1q_f32(src.add(i));
                let v1 = vld1q_f32(src.add(i + 4));
                let h0: uint16x4_t;
                let h1: uint16x4_t;
                std::arch::asm!("fcvtn {o:v}.4h, {i:v}.4s", o = lateout(vreg) h0, i = in(vreg) v0);
                std::arch::asm!("fcvtn {o:v}.4h, {i:v}.4s", o = lateout(vreg) h1, i = in(vreg) v1);
                vst1_u16(dst.add(i), h0);
                vst1_u16(dst.add(i + 4), h1);
                i += 8;
            }
            while i < len {
                *dst.add(i) = half::f16::from_f32(*src.add(i)).to_bits();
                i += 1;
            }
        }
    }

    /// RM=4 × NR=4 tiled GEMM with native F16 FMA and F16 accumulators.
    /// Processes 4 A rows × 4 B rows = 16 output cells per call.
    ///
    /// Pipeline (8 K-elem/iter): Load=8(4cy), FMA=16(4cy) → balanced @ 4cy
    /// Output: 16 / 4cy = 4.0 per cycle.
    ///
    /// F16 accumulators (uint16x8_t) use half the registers of F32,
    /// enabling the 4×4 tile that would be impossible with F32 (32 regs needed).
    #[target_feature(enable = "neon")]
    unsafe fn vec_dot_f16_native_4x4(
        k: usize,
        a_ptrs: [*const u16; 4],
        b_ptrs: [*const u16; 4],
        out: &mut [[f32; 4]; 4], // out[a_row][b_col]
    ) {
        unsafe {
            // 16 F16 accumulators: acc_AxBy = A row x × B col y
            let mut a0b0: uint16x8_t = vdupq_n_u16(0);
            let mut a0b1: uint16x8_t = vdupq_n_u16(0);
            let mut a0b2: uint16x8_t = vdupq_n_u16(0);
            let mut a0b3: uint16x8_t = vdupq_n_u16(0);
            let mut a1b0: uint16x8_t = vdupq_n_u16(0);
            let mut a1b1: uint16x8_t = vdupq_n_u16(0);
            let mut a1b2: uint16x8_t = vdupq_n_u16(0);
            let mut a1b3: uint16x8_t = vdupq_n_u16(0);
            let mut a2b0: uint16x8_t = vdupq_n_u16(0);
            let mut a2b1: uint16x8_t = vdupq_n_u16(0);
            let mut a2b2: uint16x8_t = vdupq_n_u16(0);
            let mut a2b3: uint16x8_t = vdupq_n_u16(0);
            let mut a3b0: uint16x8_t = vdupq_n_u16(0);
            let mut a3b1: uint16x8_t = vdupq_n_u16(0);
            let mut a3b2: uint16x8_t = vdupq_n_u16(0);
            let mut a3b3: uint16x8_t = vdupq_n_u16(0);

            let mut idx = 0;

            // K-loop macro: process 8 K-elements (1 K-step)
            macro_rules! k_step {
            ($off:expr) => {
                let av0: uint16x8_t = vld1q_u16(a_ptrs[0].add($off));
                let av1: uint16x8_t = vld1q_u16(a_ptrs[1].add($off));
                let av2: uint16x8_t = vld1q_u16(a_ptrs[2].add($off));
                let av3: uint16x8_t = vld1q_u16(a_ptrs[3].add($off));

                macro_rules! col {
                                    ($bi:expr, $c0:ident, $c1:ident, $c2:ident, $c3:ident) => {
                                        let bv: uint16x8_t = vld1q_u16(b_ptrs[$bi].add($off));
                                        std::arch::asm!(
                                            "fmla {d0:v}.8h, {a0:v}.8h, {b:v}.8h",
                                            "fmla {d1:v}.8h, {a1:v}.8h, {b:v}.8h",
                                            "fmla {d2:v}.8h, {a2:v}.8h, {b:v}.8h",
                                            "fmla {d3:v}.8h, {a3:v}.8h, {b:v}.8h",
                                            d0 = inout(vreg) $c0, d1 = inout(vreg) $c1,
                                            d2 = inout(vreg) $c2, d3 = inout(vreg) $c3,
                                            a0 = in(vreg) av0, a1 = in(vreg) av1,
                                            a2 = in(vreg) av2, a3 = in(vreg) av3,
                                            b = in(vreg) bv,
                                        );
                                    };
                                }
                col!(0, a0b0, a1b0, a2b0, a3b0);
                col!(1, a0b1, a1b1, a2b1, a3b1);
                col!(2, a0b2, a1b2, a2b2, a3b2);
                col!(3, a0b3, a1b3, a2b3, a3b3);
            };
        }

            // Main loop: 2x unrolled (16 K-elements per iteration)
            // Halves branch overhead; gives OOO engine 48 instructions to schedule
            while idx + 16 <= k {
                k_step!(idx);
                k_step!(idx + 8);
                idx += 16;
            }

            // 8-element tail
            if idx + 8 <= k {
                k_step!(idx);
                idx += 8;
            }

            // Reduce F16 accumulators → F32 scalars
            macro_rules! reduce_f16 {
            ($acc:expr) => {{
                let lo: float32x4_t;
                let hi: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) lo, i = in(vreg) $acc);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) hi, i = in(vreg) $acc);
                vaddvq_f32(vaddq_f32(lo, hi))
            }};
        }

            out[0] = [
                reduce_f16!(a0b0),
                reduce_f16!(a0b1),
                reduce_f16!(a0b2),
                reduce_f16!(a0b3),
            ];
            out[1] = [
                reduce_f16!(a1b0),
                reduce_f16!(a1b1),
                reduce_f16!(a1b2),
                reduce_f16!(a1b3),
            ];
            out[2] = [
                reduce_f16!(a2b0),
                reduce_f16!(a2b1),
                reduce_f16!(a2b2),
                reduce_f16!(a2b3),
            ];
            out[3] = [
                reduce_f16!(a3b0),
                reduce_f16!(a3b1),
                reduce_f16!(a3b2),
                reduce_f16!(a3b3),
            ];

            // Scalar tail (K not multiple of 8)
            while idx < k {
                for ar in 0..4 {
                    let a_val = half::f16::from_bits(*a_ptrs[ar].add(idx)).to_f32();
                    for bc in 0..4 {
                        out[ar][bc] += a_val * half::f16::from_bits(*b_ptrs[bc].add(idx)).to_f32();
                    }
                }
                idx += 1;
            }
        }
    }

    /// Native F16 FMA GEMV: 1 A row × 4 B rows → 4 F32 outputs.
    ///
    /// Matches llama.cpp's `ggml_vec_dot_f16` instruction stream (`FMLA Vd.8H`)
    /// while retaining our NR=4 A-reuse. Primary decode (M=1 GEMV) F16 matmul
    /// kernel, used by `f16_gemv_chunk` / `fused_gemv_chunk` and the
    /// `f16_gemm_chunk` tail-group fallback.
    ///
    /// Layout: 16 F16 accumulators = 4 B rows × 4 K-slots.
    /// K step = 32 per iter (llama.cpp `GGML_F16_STEP`).
    ///
    /// Instructions per iter: 4 A loads + 16 B loads + 16 FMA = 36 instrs.
    /// The earlier FMLAL-based kernel (widening F16×F16→F32) needed a pair
    /// of `fmlal`/`fmlal2` for the same flops, roughly 1.4× more instructions
    /// in the memory-bound GEMV regime, so the HW prefetcher stalled waiting
    /// for the load front to advance.
    ///
    /// F16 accumulator overflow: max K = 8960 (Qwen down_proj intermediate),
    /// 32 partial lanes → K_eff = 280. Worst case ≤280 × |10| × |2| = 5,600,
    /// well under F16 max 65,504 (~12× margin).
    #[target_feature(enable = "neon")]
    unsafe fn vec_dot_f16_native_gemv_4rows(
        k: usize,
        a_f16: *const u16,
        b_ptrs: [*const u16; 4],
        out: &mut [f32; 4],
    ) {
        unsafe {
            // 16 F16 accumulators: acc[row][k_slot] for row 0..3, k_slot 0..3.
            // Typed as uint16x8_t because inline `asm!` treats the bits opaquely;
            // `fmla .8h` reinterprets them as half-precision floats. This matches
            // `vec_dot_f16_native_4x4` and keeps us off the unstable `float16x8_t`
            // intrinsic surface.
            let mut r0s0: uint16x8_t = vdupq_n_u16(0);
            let mut r0s1: uint16x8_t = vdupq_n_u16(0);
            let mut r0s2: uint16x8_t = vdupq_n_u16(0);
            let mut r0s3: uint16x8_t = vdupq_n_u16(0);
            let mut r1s0: uint16x8_t = vdupq_n_u16(0);
            let mut r1s1: uint16x8_t = vdupq_n_u16(0);
            let mut r1s2: uint16x8_t = vdupq_n_u16(0);
            let mut r1s3: uint16x8_t = vdupq_n_u16(0);
            let mut r2s0: uint16x8_t = vdupq_n_u16(0);
            let mut r2s1: uint16x8_t = vdupq_n_u16(0);
            let mut r2s2: uint16x8_t = vdupq_n_u16(0);
            let mut r2s3: uint16x8_t = vdupq_n_u16(0);
            let mut r3s0: uint16x8_t = vdupq_n_u16(0);
            let mut r3s1: uint16x8_t = vdupq_n_u16(0);
            let mut r3s2: uint16x8_t = vdupq_n_u16(0);
            let mut r3s3: uint16x8_t = vdupq_n_u16(0);

            let mut idx = 0;

            // Main loop: 32 F16 elements per iteration (llama.cpp GGML_F16_STEP=32).
            // One A load set (4 × 8 lanes) is shared across all 4 B rows.
            while idx + 32 <= k {
                // Load 4 A vectors (32 f16 lanes total)
                let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
                let av1: uint16x8_t = vld1q_u16(a_f16.add(idx + 8));
                let av2: uint16x8_t = vld1q_u16(a_f16.add(idx + 16));
                let av3: uint16x8_t = vld1q_u16(a_f16.add(idx + 24));

                // Per-row macro: load 4 B vectors, issue 4 FMA .8h (independent
                // accumulators, independent operands → max ILP).
                macro_rules! fma_row {
                ($bp:expr, $s0:ident, $s1:ident, $s2:ident, $s3:ident) => {
                    let bv0: uint16x8_t = vld1q_u16($bp.add(idx));
                    let bv1: uint16x8_t = vld1q_u16($bp.add(idx + 8));
                    let bv2: uint16x8_t = vld1q_u16($bp.add(idx + 16));
                    let bv3: uint16x8_t = vld1q_u16($bp.add(idx + 24));
                    std::arch::asm!(
                        "fmla {d0:v}.8h, {a0:v}.8h, {b0:v}.8h",
                        "fmla {d1:v}.8h, {a1:v}.8h, {b1:v}.8h",
                        "fmla {d2:v}.8h, {a2:v}.8h, {b2:v}.8h",
                        "fmla {d3:v}.8h, {a3:v}.8h, {b3:v}.8h",
                        d0 = inout(vreg) $s0, d1 = inout(vreg) $s1,
                        d2 = inout(vreg) $s2, d3 = inout(vreg) $s3,
                        a0 = in(vreg) av0, a1 = in(vreg) av1,
                        a2 = in(vreg) av2, a3 = in(vreg) av3,
                        b0 = in(vreg) bv0, b1 = in(vreg) bv1,
                        b2 = in(vreg) bv2, b3 = in(vreg) bv3,
                    );
                };
            }
                fma_row!(b_ptrs[0], r0s0, r0s1, r0s2, r0s3);
                fma_row!(b_ptrs[1], r1s0, r1s1, r1s2, r1s3);
                fma_row!(b_ptrs[2], r2s0, r2s1, r2s2, r2s3);
                fma_row!(b_ptrs[3], r3s0, r3s1, r3s2, r3s3);

                idx += 32;
            }

            // 16-element tail (one half-step): 2 slots per row
            if idx + 16 <= k {
                let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
                let av1: uint16x8_t = vld1q_u16(a_f16.add(idx + 8));
                macro_rules! fma_row16 {
                ($bp:expr, $s0:ident, $s1:ident) => {
                    let bv0: uint16x8_t = vld1q_u16($bp.add(idx));
                    let bv1: uint16x8_t = vld1q_u16($bp.add(idx + 8));
                    std::arch::asm!(
                        "fmla {d0:v}.8h, {a0:v}.8h, {b0:v}.8h",
                        "fmla {d1:v}.8h, {a1:v}.8h, {b1:v}.8h",
                        d0 = inout(vreg) $s0, d1 = inout(vreg) $s1,
                        a0 = in(vreg) av0, a1 = in(vreg) av1,
                        b0 = in(vreg) bv0, b1 = in(vreg) bv1,
                    );
                };
            }
                fma_row16!(b_ptrs[0], r0s0, r0s1);
                fma_row16!(b_ptrs[1], r1s0, r1s1);
                fma_row16!(b_ptrs[2], r2s0, r2s1);
                fma_row16!(b_ptrs[3], r3s0, r3s1);
                idx += 16;
            }

            // 8-element tail: 1 slot per row
            if idx + 8 <= k {
                let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
                macro_rules! fma_row8 {
                ($bp:expr, $s0:ident) => {
                    let bv0: uint16x8_t = vld1q_u16($bp.add(idx));
                    std::arch::asm!(
                        "fmla {d:v}.8h, {a:v}.8h, {b:v}.8h",
                        d = inout(vreg) $s0,
                        a = in(vreg) av0,
                        b = in(vreg) bv0,
                    );
                };
            }
                fma_row8!(b_ptrs[0], r0s0);
                fma_row8!(b_ptrs[1], r1s0);
                fma_row8!(b_ptrs[2], r2s0);
                fma_row8!(b_ptrs[3], r3s0);
                idx += 8;
            }

            // Reduce: 4 F16 partials per row → F32 scalar.
            // Each partial is converted via fcvtl/fcvtl2 to two f32x4 vectors and
            // summed in F32 (preserves precision over an in-F16 pre-sum).
            macro_rules! reduce_f16_partial {
            ($acc:expr) => {{
                let lo: float32x4_t;
                let hi: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) lo, i = in(vreg) $acc);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) hi, i = in(vreg) $acc);
                vaddq_f32(lo, hi)
            }};
        }
            let r0 = vaddq_f32(
                vaddq_f32(reduce_f16_partial!(r0s0), reduce_f16_partial!(r0s1)),
                vaddq_f32(reduce_f16_partial!(r0s2), reduce_f16_partial!(r0s3)),
            );
            let r1 = vaddq_f32(
                vaddq_f32(reduce_f16_partial!(r1s0), reduce_f16_partial!(r1s1)),
                vaddq_f32(reduce_f16_partial!(r1s2), reduce_f16_partial!(r1s3)),
            );
            let r2 = vaddq_f32(
                vaddq_f32(reduce_f16_partial!(r2s0), reduce_f16_partial!(r2s1)),
                vaddq_f32(reduce_f16_partial!(r2s2), reduce_f16_partial!(r2s3)),
            );
            let r3 = vaddq_f32(
                vaddq_f32(reduce_f16_partial!(r3s0), reduce_f16_partial!(r3s1)),
                vaddq_f32(reduce_f16_partial!(r3s2), reduce_f16_partial!(r3s3)),
            );
            out[0] = vaddvq_f32(r0);
            out[1] = vaddvq_f32(r1);
            out[2] = vaddvq_f32(r2);
            out[3] = vaddvq_f32(r3);

            // Scalar tail (K not a multiple of 8)
            while idx < k {
                let a_val = half::f16::from_bits(*a_f16.add(idx)).to_f32();
                for r in 0..4 {
                    out[r] += a_val * half::f16::from_bits(*b_ptrs[r].add(idx)).to_f32();
                }
                idx += 1;
            }
        }
    }

    /// Native F16 FMA single-row dot product: A(F16) · B(F16) → scalar F32.
    ///
    /// Used for the `j < j_end` column tail of `f16_gemv_chunk` /
    /// `fused_gemv_chunk` when the chunk width is not a multiple of NR=4
    /// (e.g. tensor partition CPU slices with non-aligned out_dim), and for
    /// the MR=4 column tail / MR<4 tail group in `f16_gemm_chunk`.
    ///
    /// Uses 4 F16 K-slot accumulators following llama.cpp's pattern.
    #[target_feature(enable = "neon")]
    unsafe fn vec_dot_f16_native_gemv_1row(k: usize, a_f16: *const u16, b_ptr: *const u16) -> f32 {
        unsafe {
            let mut s0: uint16x8_t = vdupq_n_u16(0);
            let mut s1: uint16x8_t = vdupq_n_u16(0);
            let mut s2: uint16x8_t = vdupq_n_u16(0);
            let mut s3: uint16x8_t = vdupq_n_u16(0);

            let mut idx = 0;

            while idx + 32 <= k {
                let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
                let av1: uint16x8_t = vld1q_u16(a_f16.add(idx + 8));
                let av2: uint16x8_t = vld1q_u16(a_f16.add(idx + 16));
                let av3: uint16x8_t = vld1q_u16(a_f16.add(idx + 24));
                let bv0: uint16x8_t = vld1q_u16(b_ptr.add(idx));
                let bv1: uint16x8_t = vld1q_u16(b_ptr.add(idx + 8));
                let bv2: uint16x8_t = vld1q_u16(b_ptr.add(idx + 16));
                let bv3: uint16x8_t = vld1q_u16(b_ptr.add(idx + 24));
                std::arch::asm!(
                    "fmla {d0:v}.8h, {a0:v}.8h, {b0:v}.8h",
                    "fmla {d1:v}.8h, {a1:v}.8h, {b1:v}.8h",
                    "fmla {d2:v}.8h, {a2:v}.8h, {b2:v}.8h",
                    "fmla {d3:v}.8h, {a3:v}.8h, {b3:v}.8h",
                    d0 = inout(vreg) s0, d1 = inout(vreg) s1,
                    d2 = inout(vreg) s2, d3 = inout(vreg) s3,
                    a0 = in(vreg) av0, a1 = in(vreg) av1,
                    a2 = in(vreg) av2, a3 = in(vreg) av3,
                    b0 = in(vreg) bv0, b1 = in(vreg) bv1,
                    b2 = in(vreg) bv2, b3 = in(vreg) bv3,
                );
                idx += 32;
            }

            if idx + 16 <= k {
                let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
                let av1: uint16x8_t = vld1q_u16(a_f16.add(idx + 8));
                let bv0: uint16x8_t = vld1q_u16(b_ptr.add(idx));
                let bv1: uint16x8_t = vld1q_u16(b_ptr.add(idx + 8));
                std::arch::asm!(
                    "fmla {d0:v}.8h, {a0:v}.8h, {b0:v}.8h",
                    "fmla {d1:v}.8h, {a1:v}.8h, {b1:v}.8h",
                    d0 = inout(vreg) s0, d1 = inout(vreg) s1,
                    a0 = in(vreg) av0, a1 = in(vreg) av1,
                    b0 = in(vreg) bv0, b1 = in(vreg) bv1,
                );
                idx += 16;
            }

            if idx + 8 <= k {
                let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
                let bv0: uint16x8_t = vld1q_u16(b_ptr.add(idx));
                std::arch::asm!(
                    "fmla {d:v}.8h, {a:v}.8h, {b:v}.8h",
                    d = inout(vreg) s0,
                    a = in(vreg) av0,
                    b = in(vreg) bv0,
                );
                idx += 8;
            }

            // Reduce 4 F16 partials → F32 scalar (sum in F32 for precision)
            macro_rules! reduce_f16_partial {
            ($acc:expr) => {{
                let lo: float32x4_t;
                let hi: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) lo, i = in(vreg) $acc);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) hi, i = in(vreg) $acc);
                vaddq_f32(lo, hi)
            }};
        }
            let v = vaddq_f32(
                vaddq_f32(reduce_f16_partial!(s0), reduce_f16_partial!(s1)),
                vaddq_f32(reduce_f16_partial!(s2), reduce_f16_partial!(s3)),
            );
            let mut sum = vaddvq_f32(v);

            while idx < k {
                sum += half::f16::from_bits(*a_f16.add(idx)).to_f32()
                    * half::f16::from_bits(*b_ptr.add(idx)).to_f32();
                idx += 1;
            }
            sum
        }
    }

    /// NEON F16 weighted accumulate: out[i] += weight * v_f16[i]
    /// Fused fcvtl + vfmaq_f32 — operates directly on F16 data without temp buffer.
    ///
    /// # Safety
    /// `out_ptr` must point to at least `k` f32 values (read-write),
    /// `v_ptr` to at least `k` u16 (F16) values.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_mad_f16(k: usize, out_ptr: *mut f32, v_ptr: *const u16, weight: f32) {
        unsafe {
            let w_v = vdupq_n_f32(weight);
            let mut idx = 0;

            // Main loop: 16 elements per iteration
            while idx + 16 <= k {
                let v_raw0: uint16x8_t = vld1q_u16(v_ptr.add(idx));
                let v_raw1: uint16x8_t = vld1q_u16(v_ptr.add(idx + 8));

                let v0: float32x4_t;
                let v1: float32x4_t;
                let v2: float32x4_t;
                let v3: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) v0, i = in(vreg) v_raw0);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) v1, i = in(vreg) v_raw0);
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) v2, i = in(vreg) v_raw1);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) v3, i = in(vreg) v_raw1);

                vst1q_f32(
                    out_ptr.add(idx),
                    vfmaq_f32(vld1q_f32(out_ptr.add(idx)), w_v, v0),
                );
                vst1q_f32(
                    out_ptr.add(idx + 4),
                    vfmaq_f32(vld1q_f32(out_ptr.add(idx + 4)), w_v, v1),
                );
                vst1q_f32(
                    out_ptr.add(idx + 8),
                    vfmaq_f32(vld1q_f32(out_ptr.add(idx + 8)), w_v, v2),
                );
                vst1q_f32(
                    out_ptr.add(idx + 12),
                    vfmaq_f32(vld1q_f32(out_ptr.add(idx + 12)), w_v, v3),
                );

                idx += 16;
            }

            // 8-element tail
            while idx + 8 <= k {
                let v_raw: uint16x8_t = vld1q_u16(v_ptr.add(idx));
                let vlo: float32x4_t;
                let vhi: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) vlo, i = in(vreg) v_raw);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) vhi, i = in(vreg) v_raw);

                vst1q_f32(
                    out_ptr.add(idx),
                    vfmaq_f32(vld1q_f32(out_ptr.add(idx)), w_v, vlo),
                );
                vst1q_f32(
                    out_ptr.add(idx + 4),
                    vfmaq_f32(vld1q_f32(out_ptr.add(idx + 4)), w_v, vhi),
                );

                idx += 8;
            }

            // Scalar tail
            while idx < k {
                *out_ptr.add(idx) += weight * half::f16::from_bits(*v_ptr.add(idx)).to_f32();
                idx += 1;
            }
        }
    }

    fn matmul_transposed_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let n = b_shape[b_rank - 2];
        let m: usize = a_shape[..a_rank - 1].iter().product();

        // Heuristic: Serial for tiny matrices (Proven 9x faster for [1, 128, 256])
        if (m * n * k) < 100_000 {
            return self.matmul_transposed_q4_0_serial(a, b, out);
        }

        let a_data = a.as_slice::<f32>();
        let nb_k = k / QK4_0;
        let out_data = out.as_mut_slice::<f32>();
        let nb_k_q8 = k / QK8_0;
        let total_q8_blocks = m * nb_k_q8;

        // Reuse thread-local workspace to avoid per-matmul allocation.
        let a_q8_ptr: SendSyncPtr = Q8_WORKSPACE.with(|ws| {
            let mut a_q8 = ws.borrow_mut();
            if a_q8.len() < total_q8_blocks {
                let additional = total_q8_blocks - a_q8.len();
                a_q8.reserve(additional);
                unsafe {
                    a_q8.set_len(total_q8_blocks);
                }
            }

            for i in 0..m {
                let a_offset = i * k;
                let a_row = &a_data[a_offset..a_offset + k];
                let q8_row = &mut a_q8[i * nb_k_q8..(i + 1) * nb_k_q8];
                unsafe {
                    self.quantize_row_q8_0(a_row, q8_row, k);
                }
            }

            SendSyncPtr(a_q8.as_ptr())
        });

        // Hoist feature detection outside closure
        #[cfg(target_arch = "aarch64")]
        let use_i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        #[cfg(not(target_arch = "aarch64"))]
        let use_i8mm = false;

        #[cfg(target_arch = "aarch64")]
        let use_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        #[cfg(not(target_arch = "aarch64"))]
        let use_dotprod = false;

        let num_threads = rayon::current_num_threads();

        // Try to get interleaved weights for i8mm 4-row kernel
        let il_ptr = if use_i8mm && n.is_multiple_of(4) {
            get_interleaved_q4(b, nb_k, n)
        } else {
            None
        };

        if m == 1 {
            // === GEMV: adaptive chunk size, flat N-parallel ===
            let target_tasks = num_threads * 4;
            let chunk_size = if il_ptr.is_some() {
                // 4-row interleaved: round up to multiple of 4
                let cs = n.div_ceil(target_tasks);
                (cs + 3) & !3
            } else {
                let cs = n.div_ceil(target_tasks);
                (cs + 1) & !1
            };

            out_data.par_chunks_mut(chunk_size).enumerate().for_each(
                |(chunk_idx, chunk): (usize, &mut [f32])| {
                    let start_idx = chunk_idx * chunk_size;
                    let a_ptr = unsafe { a_q8_ptr.ptr() };

                    if let Some(il) = il_ptr {
                        // i8mm 4-row interleaved path — no weight vzip overhead
                        let il_base = il as *const u8;
                        let mut li = 0;
                        while li + 3 < chunk.len() && start_idx + li + 3 < n {
                            let j = start_idx + li;
                            let group = j / 4; // which 4-row group
                            let il_group_ptr = unsafe { il_base.add(group * nb_k * IL_BLOCK_SIZE) };
                            let mut results = [0.0f32; 4];
                            unsafe {
                                self.vec_dot_q4_0_q8_0_i8mm_4rows_il(
                                    nb_k,
                                    il_group_ptr,
                                    a_ptr,
                                    &mut results,
                                );
                            }
                            chunk[li..li + 4].copy_from_slice(&results);
                            li += 4;
                        }
                        // Tail: fallback to non-interleaved sdot
                        let b_base = b.as_ptr() as *const BlockQ4_0;
                        while li < chunk.len() {
                            let j = start_idx + li;
                            let mut sum = 0.0;
                            unsafe {
                                self.vec_dot_q4_0_q8_0_sdot(
                                    k,
                                    &mut sum,
                                    b_base.add(j * nb_k),
                                    a_ptr,
                                )
                            }
                            chunk[li] = sum;
                            li += 1;
                        }
                    } else if use_i8mm {
                        let b_base = b.as_ptr() as *const BlockQ4_0;
                        let mut li = 0;
                        while li < chunk.len() {
                            let j = start_idx + li;
                            if li + 1 < chunk.len() && j + 1 < n {
                                let (left, right) = chunk[li..].split_at_mut(1);
                                unsafe {
                                    self.vec_dot_q4_0_q8_0_i8mm(
                                        k,
                                        &mut left[0],
                                        &mut right[0],
                                        b_base.add(j * nb_k),
                                        b_base.add((j + 1) * nb_k),
                                        a_ptr,
                                    );
                                }
                                li += 2;
                            } else {
                                let mut sum = 0.0;
                                unsafe {
                                    self.vec_dot_q4_0_q8_0_sdot(
                                        k,
                                        &mut sum,
                                        b_base.add(j * nb_k),
                                        a_ptr,
                                    )
                                }
                                chunk[li] = sum;
                                li += 1;
                            }
                        }
                    } else if use_dotprod {
                        let b_base = b.as_ptr() as *const BlockQ4_0;
                        for (li, out_val) in chunk.iter_mut().enumerate() {
                            let mut sum = 0.0;
                            unsafe {
                                self.vec_dot_q4_0_q8_0_sdot(
                                    k,
                                    &mut sum,
                                    b_base.add((start_idx + li) * nb_k),
                                    a_ptr,
                                )
                            }
                            *out_val = sum;
                        }
                    } else {
                        let b_base = b.as_ptr() as *const BlockQ4_0;
                        for (li, out_val) in chunk.iter_mut().enumerate() {
                            let mut sum = 0.0;
                            unsafe {
                                self.vec_dot_q4_0_q8_0(
                                    k,
                                    &mut sum,
                                    b_base.add((start_idx + li) * nb_k),
                                    a_ptr,
                                )
                            }
                            *out_val = sum;
                        }
                    }
                },
            );
        } else {
            // === GEMM (M>1): N-major tiled ===
            // Weight rows are shared across M activation rows within each N-chunk.
            let nr = if il_ptr.is_some() { 4usize } else { 2 };
            let target_n_chunks = num_threads * 4;
            let rows_per_chunk = n.div_ceil(target_n_chunks).div_ceil(nr) * nr;
            let rows_per_chunk = rows_per_chunk.max(nr);
            let n_chunks = n.div_ceil(rows_per_chunk);

            let b_base_usize = b.as_ptr() as usize;
            let a_q8_usize = a_q8_ptr.ptr() as usize;
            let out_ptr_usize = out_data.as_mut_ptr() as usize;
            let il_usize = il_ptr.unwrap_or(0);

            (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
                let j_start = chunk_idx * rows_per_chunk;
                let j_end = (j_start + rows_per_chunk).min(n);
                let b_base = b_base_usize as *const BlockQ4_0;
                let a_q8_base = a_q8_usize as *const BlockQ8_0;
                let out_base = out_ptr_usize as *mut f32;
                let backend = CpuBackendNeon::new();

                let mut j = j_start;
                while j < j_end {
                    if il_usize != 0 && j + 3 < j_end {
                        // i8mm 4-row interleaved: weight data pre-arranged for smmla
                        let group = j / 4;
                        let il_group =
                            unsafe { (il_usize as *const u8).add(group * nb_k * IL_BLOCK_SIZE) };
                        for i in 0..m {
                            let a_row = unsafe { a_q8_base.add(i * nb_k_q8) };
                            let mut results = [0.0f32; 4];
                            unsafe {
                                backend.vec_dot_q4_0_q8_0_i8mm_4rows_il(
                                    nb_k,
                                    il_group,
                                    a_row,
                                    &mut results,
                                );
                                *out_base.add(i * n + j) = results[0];
                                *out_base.add(i * n + j + 1) = results[1];
                                *out_base.add(i * n + j + 2) = results[2];
                                *out_base.add(i * n + j + 3) = results[3];
                            }
                        }
                        j += 4;
                    } else if use_i8mm && j + 1 < j_end {
                        let b0 = unsafe { b_base.add(j * nb_k) };
                        let b1 = unsafe { b_base.add((j + 1) * nb_k) };
                        for i in 0..m {
                            let a_row = unsafe { a_q8_base.add(i * nb_k_q8) };
                            let mut s0 = 0.0f32;
                            let mut s1 = 0.0f32;
                            unsafe {
                                backend.vec_dot_q4_0_q8_0_i8mm(k, &mut s0, &mut s1, b0, b1, a_row);
                                *out_base.add(i * n + j) = s0;
                                *out_base.add(i * n + j + 1) = s1;
                            }
                        }
                        j += 2;
                    } else {
                        let b_ptr = unsafe { b_base.add(j * nb_k) };
                        for i in 0..m {
                            let a_row = unsafe { a_q8_base.add(i * nb_k_q8) };
                            let mut sum = 0.0f32;
                            unsafe {
                                if use_dotprod {
                                    backend.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_ptr, a_row);
                                } else {
                                    backend.vec_dot_q4_0_q8_0(k, &mut sum, b_ptr, a_row);
                                }
                                *out_base.add(i * n + j) = sum;
                            }
                        }
                        j += 1;
                    }
                }
            });
        }
        Ok(())
    }

    fn matmul_transposed_q4_0_serial(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: &mut Tensor,
    ) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();

        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 2];
        let m: usize = a_shape[..a_shape.len() - 1].iter().product();

        let a_data = a.as_slice::<f32>();
        let nb_k = k / QK4_0;

        // Safety: b is created as Q4_0 tensor, so data is BlockQ4_0
        let total_blocks = n * nb_k;
        let b_blocks =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const BlockQ4_0, total_blocks) };

        let out_data = out.as_mut_slice::<f32>();

        for i in 0..m {
            let a_offset = i * k;
            let a_row = &a_data[a_offset..a_offset + k];

            for j in 0..n {
                let b_offset = j * nb_k;
                let b_row_blocks = &b_blocks[b_offset..b_offset + nb_k];

                let mut sum = 0.0;
                for (bi, block) in b_row_blocks.iter().enumerate() {
                    let d = block.d.to_f32();
                    let a_slice = &a_row[bi * QK4_0..(bi + 1) * QK4_0];

                    let mut isum: f32 = 0.0;
                    for z in 0..(QK4_0 / 2) {
                        let b_val = block.qs[z];
                        let v0 = (b_val & 0x0F) as i8 - 8;
                        let v1 = (b_val >> 4) as i8 - 8;

                        isum += v0 as f32 * a_slice[z];
                        isum += v1 as f32 * a_slice[z + QK4_0 / 2];
                    }
                    sum += d * isum;
                }
                out_data[i * n + j] = sum;
            }
        }
        Ok(())
    }
    // Q8_0 Quantization (NEON)
    // Ported from llama.cpp/arch/arm/quants.c
    #[target_feature(enable = "neon")]
    pub unsafe fn quantize_row_q8_0(&self, x: &[f32], y: &mut [BlockQ8_0], k: usize) {
        assert!(k.is_multiple_of(QK8_0));
        let nb = k / QK8_0;

        let x_ptr = x.as_ptr();

        unsafe {
            for i in 0..nb {
                // Load 32 floats
                // 32 floats = 8 vectors of float32x4_t
                let src_ptr = x_ptr.add(i * QK8_0);

                // Compute max abs
                let mut max_abs = vdupq_n_f32(0.0);

                // Unroll 8 vectors
                let v0 = vld1q_f32(src_ptr);
                let v1 = vld1q_f32(src_ptr.add(4));
                let v2 = vld1q_f32(src_ptr.add(8));
                let v3 = vld1q_f32(src_ptr.add(12));
                let v4 = vld1q_f32(src_ptr.add(16));
                let v5 = vld1q_f32(src_ptr.add(20));
                let v6 = vld1q_f32(src_ptr.add(24));
                let v7 = vld1q_f32(src_ptr.add(28));

                max_abs = vmaxq_f32(max_abs, vabsq_f32(v0));
                max_abs = vmaxq_f32(max_abs, vabsq_f32(v1));
                max_abs = vmaxq_f32(max_abs, vabsq_f32(v2));
                max_abs = vmaxq_f32(max_abs, vabsq_f32(v3));
                max_abs = vmaxq_f32(max_abs, vabsq_f32(v4));
                max_abs = vmaxq_f32(max_abs, vabsq_f32(v5));
                max_abs = vmaxq_f32(max_abs, vabsq_f32(v6));
                max_abs = vmaxq_f32(max_abs, vabsq_f32(v7));

                let amax = vmaxvq_f32(max_abs);
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };

                y[i].d = half::f16::from_f32(d);

                let id_v = vdupq_n_f32(id);

                // Convert to i8
                // v0..v3 -> 16 floats -> 16 i32 -> 16 i16 -> 16 i8 (1 vector)
                // v4..v7 -> ... -> 16 i8 (1 vector)

                // Process first 16
                let i32_0 = vcvtnq_s32_f32(vmulq_f32(v0, id_v));
                let i32_1 = vcvtnq_s32_f32(vmulq_f32(v1, id_v));
                let i32_2 = vcvtnq_s32_f32(vmulq_f32(v2, id_v));
                let i32_3 = vcvtnq_s32_f32(vmulq_f32(v3, id_v));

                let i16_0 = vqmovn_s32(i32_0);
                let i16_1 = vqmovn_s32(i32_1);
                let i16_2 = vqmovn_s32(i32_2);
                let i16_3 = vqmovn_s32(i32_3);

                let i16_low = vcombine_s16(i16_0, i16_1);
                let i16_high = vcombine_s16(i16_2, i16_3);

                let i8_0 = vqmovn_s16(i16_low);
                let i8_1 = vqmovn_s16(i16_high);
                let i8_res_0 = vcombine_s8(i8_0, i8_1);

                // Process next 16
                let i32_4 = vcvtnq_s32_f32(vmulq_f32(v4, id_v));
                let i32_5 = vcvtnq_s32_f32(vmulq_f32(v5, id_v));
                let i32_6 = vcvtnq_s32_f32(vmulq_f32(v6, id_v));
                let i32_7 = vcvtnq_s32_f32(vmulq_f32(v7, id_v));

                let i16_4 = vqmovn_s32(i32_4);
                let i16_5 = vqmovn_s32(i32_5);
                let i16_6 = vqmovn_s32(i32_6);
                let i16_7 = vqmovn_s32(i32_7);

                let i16_low2 = vcombine_s16(i16_4, i16_5);
                let i16_high2 = vcombine_s16(i16_6, i16_7);

                let i8_2 = vqmovn_s16(i16_low2);
                let i8_3 = vqmovn_s16(i16_high2);
                let i8_res_1 = vcombine_s8(i8_2, i8_3);

                // Store
                vst1q_s8(y[i].qs.as_mut_ptr(), i8_res_0);
                vst1q_s8(y[i].qs.as_mut_ptr().add(16), i8_res_1);
            }
        }
    }

    // Dot Product Q4_0 * Q8_0 (NEON, dotprod 미지원 fallback)
    //
    // 최적화 (2026-04-13, P0-1):
    //   - 블록당 4회 `vaddlvq_s16` (각 호출이 i16x8→i64 horizontal reduction) 제거
    //   - 대신 `vmull_s8` → `vpaddlq_s16` → `vaddq_s32` 체인으로 int32x4_t 누적기에 유지
    //   - 블록 스케일은 `vmlaq_n_f32`로 float32x4_t sumv에 누적 (llama.cpp 패턴)
    //   - 최종 `vaddvq_f32` 1회만 수행하여 horizontal reduction 비용 최소화
    //
    // llama.cpp `ggml_vec_dot_q4_0_q8_0` (arch/arm/quants.c) 기본 NEON 경로와 동일 패턴.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q4_0_q8_0(
        &self,
        n: usize,
        s: &mut f32,
        vx: *const BlockQ4_0,
        vy: *const BlockQ8_0,
    ) {
        let nb = n / QK8_0;

        unsafe {
            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(8);

            // Float 누적기 2개 (2-block unroll → 독립 dependency chain으로 FMA 파이프 활용)
            let mut sumv0 = vdupq_n_f32(0.0);
            let mut sumv1 = vdupq_n_f32(0.0);

            let mut i = 0;
            while i + 1 < nb {
                let x0 = &*vx.add(i);
                let y0 = &*vy.add(i);
                let x1 = &*vx.add(i + 1);
                let y1 = &*vy.add(i + 1);

                let d0 = x0.d.to_f32() * y0.d.to_f32();
                let d1 = x1.d.to_f32() * y1.d.to_f32();

                // Block 0: unpack Q4_0 nibbles + sub 8
                let v0_0 = vld1q_u8(x0.qs.as_ptr());
                let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);
                let y0_l = vld1q_s8(y0.qs.as_ptr());
                let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));

                // Block 1: unpack Q4_0 nibbles + sub 8
                let v1_0 = vld1q_u8(x1.qs.as_ptr());
                let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1_0, m4b)), s8b);
                let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1_0, 4)), s8b);
                let y1_l = vld1q_s8(y1.qs.as_ptr());
                let y1_h = vld1q_s8(y1.qs.as_ptr().add(16));

                // Block 0 dot → int32x4_t: vmull_s8 × 4 → vpaddlq_s16 × 2 → vaddq_s32
                //   vdotq_s32(acc, a, b)와 등가 (dotprod 없는 fallback)
                let mul_l0_0 = vmull_s8(vget_low_s8(x0_l), vget_low_s8(y0_l));
                let mul_l1_0 = vmull_high_s8(x0_l, y0_l);
                let mul_h0_0 = vmull_s8(vget_low_s8(x0_h), vget_low_s8(y0_h));
                let mul_h1_0 = vmull_high_s8(x0_h, y0_h);
                // 각 i16x8 쌍을 i32x4 pairwise로 접고 누적. 블록당 horizontal reduction 없음.
                let p_low_0 = vaddq_s32(vpaddlq_s16(mul_l0_0), vpaddlq_s16(mul_l1_0));
                let p_high_0 = vaddq_s32(vpaddlq_s16(mul_h0_0), vpaddlq_s16(mul_h1_0));
                let p_0 = vaddq_s32(p_low_0, p_high_0);

                // Block 1 dot → int32x4_t
                let mul_l0_1 = vmull_s8(vget_low_s8(x1_l), vget_low_s8(y1_l));
                let mul_l1_1 = vmull_high_s8(x1_l, y1_l);
                let mul_h0_1 = vmull_s8(vget_low_s8(x1_h), vget_low_s8(y1_h));
                let mul_h1_1 = vmull_high_s8(x1_h, y1_h);
                let p_low_1 = vaddq_s32(vpaddlq_s16(mul_l0_1), vpaddlq_s16(mul_l1_1));
                let p_high_1 = vaddq_s32(vpaddlq_s16(mul_h0_1), vpaddlq_s16(mul_h1_1));
                let p_1 = vaddq_s32(p_low_1, p_high_1);

                // Float 누적: sumv += vcvtq_f32_s32(p) * d (scalar broadcast FMA)
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);

                i += 2;
            }

            // 루프 종료 후 horizontal reduction 1회 (float lane 4개 → scalar)
            let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

            // Tail (nb가 홀수일 때 마지막 1블록)
            if i < nb {
                let x = &*vx.add(i);
                let y = &*vy.add(i);
                let d = x.d.to_f32() * y.d.to_f32();

                let v0_0 = vld1q_u8(x.qs.as_ptr());
                let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);
                let y0_l = vld1q_s8(y.qs.as_ptr());
                let y0_h = vld1q_s8(y.qs.as_ptr().add(16));

                let mul_l0 = vmull_s8(vget_low_s8(x0_l), vget_low_s8(y0_l));
                let mul_l1 = vmull_high_s8(x0_l, y0_l);
                let mul_h0 = vmull_s8(vget_low_s8(x0_h), vget_low_s8(y0_h));
                let mul_h1 = vmull_high_s8(x0_h, y0_h);
                let p_low = vaddq_s32(vpaddlq_s16(mul_l0), vpaddlq_s16(mul_l1));
                let p_high = vaddq_s32(vpaddlq_s16(mul_h0), vpaddlq_s16(mul_h1));
                let p = vaddq_s32(p_low, p_high);

                // Tail은 lane-sum을 스칼라 가중 (단일 블록이므로 vmlaq_n_f32 대비 경미한 비용)
                sumf += d * vaddvq_s32(p) as f32;
            }

            *s = sumf;
        }
    }

    // Dot Product Q4_0 * Q8_0 (NEON + DotProd)
    // 2-block unrolling with vector FMA accumulation (matches llama.cpp pattern).
    // Key optimisation: dot products stay in int32x4_t lanes, converted to float32x4_t
    // and accumulated via vmlaq_n_f32, avoiding per-block scalar reduction (vaddvq_s32).
    // Final horizontal reduction happens once after the loop.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn vec_dot_q4_0_q8_0_sdot(
        &self,
        n: usize,
        s: &mut f32,
        vx: *const BlockQ4_0,
        vy: *const BlockQ8_0,
    ) {
        let nb = n / QK8_0;

        unsafe {
            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(8);

            let mut sumv0 = vdupq_n_f32(0.0);
            let mut sumv1 = vdupq_n_f32(0.0);

            // Main loop: process 2 blocks per iteration
            let mut i = 0;
            while i + 1 < nb {
                let x0 = &*vx.add(i);
                let x1 = &*vx.add(i + 1);
                let y0 = &*vy.add(i);
                let y1 = &*vy.add(i + 1);

                let d0 = x0.d.to_f32() * y0.d.to_f32();
                let d1 = x1.d.to_f32() * y1.d.to_f32();

                // Load and unpack Q4_0 block 0
                let v0_0 = vld1q_u8(x0.qs.as_ptr());
                let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);

                // Load and unpack Q4_0 block 1
                let v0_1 = vld1q_u8(x1.qs.as_ptr());
                let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_1, m4b)), s8b);
                let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4)), s8b);

                // Load Q8_0 block 0
                let y0_l = vld1q_s8(y0.qs.as_ptr());
                let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));

                // Load Q8_0 block 1
                let y1_l = vld1q_s8(y1.qs.as_ptr());
                let y1_h = vld1q_s8(y1.qs.as_ptr().add(16));

                // Dot product block 0: sdot low then high into p_0
                let mut p_0 = vdupq_n_s32(0);
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) p_0,
                    x = in(vreg) x0_l,
                    y = in(vreg) y0_l,
                );
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) p_0,
                    x = in(vreg) x0_h,
                    y = in(vreg) y0_h,
                );

                // Dot product block 1: sdot low then high into p_1
                let mut p_1 = vdupq_n_s32(0);
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) p_1,
                    x = in(vreg) x1_l,
                    y = in(vreg) y1_l,
                );
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) p_1,
                    x = in(vreg) x1_h,
                    y = in(vreg) y1_h,
                );

                // Vector FMA: accumulate int32x4_t → float32x4_t with scalar scale
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);

                i += 2;
            }

            // Horizontal reduction (single pass at end)
            let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

            // Handle odd remainder block (when nb is odd)
            if i < nb {
                let x = &*vx.add(i);
                let y = &*vy.add(i);
                let d = x.d.to_f32() * y.d.to_f32();

                let v0 = vld1q_u8(x.qs.as_ptr());
                let x_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), s8b);
                let x_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8b);

                let y_l = vld1q_s8(y.qs.as_ptr());
                let y_h = vld1q_s8(y.qs.as_ptr().add(16));

                let mut acc = vdupq_n_s32(0);
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) acc,
                    x = in(vreg) x_l,
                    y = in(vreg) y_l,
                );
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) acc,
                    x = in(vreg) x_h,
                    y = in(vreg) y_h,
                );

                sumf += d * vaddvq_s32(acc) as f32;
            }

            *s = sumf;
        }
    }

    /// 4-row sdot dot product: weight 4행 × activation 1행 → 결과 4개.
    /// Activation Q8_0 blocks are loaded once and reused across 4 weight rows,
    /// reducing memory bandwidth by ~4x in the GEMV (M=1) regime.
    ///
    /// Each iteration processes 2 Q4_0/Q8_0 block pairs (64 values) to keep
    /// the 8 accumulator registers busy and hide sdot latency.
    ///
    /// # Safety
    /// - `vx[0..4]` each point to `n / QK8_0` valid `BlockQ4_0` blocks
    /// - `vy` points to `n / QK8_0` valid `BlockQ8_0` blocks
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q4_0_q8_0_sdot_4rows(
        &self,
        n: usize,
        vx: [*const BlockQ4_0; 4],
        vy: *const BlockQ8_0,
        out: &mut [f32; 4],
    ) {
        let nb = n / QK8_0;

        unsafe {
            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(8);

            // 8 accumulators: 2 per row (for 2-block unrolling)
            let mut sum0a = vdupq_n_f32(0.0);
            let mut sum0b = vdupq_n_f32(0.0);
            let mut sum1a = vdupq_n_f32(0.0);
            let mut sum1b = vdupq_n_f32(0.0);
            let mut sum2a = vdupq_n_f32(0.0);
            let mut sum2b = vdupq_n_f32(0.0);
            let mut sum3a = vdupq_n_f32(0.0);
            let mut sum3b = vdupq_n_f32(0.0);

            let mut i = 0;
            while i + 1 < nb {
                // Load activation Q8_0 — shared across all 4 weight rows
                let y0 = &*vy.add(i);
                let y1 = &*vy.add(i + 1);
                let y0_l = vld1q_s8(y0.qs.as_ptr());
                let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));
                let y1_l = vld1q_s8(y1.qs.as_ptr());
                let y1_h = vld1q_s8(y1.qs.as_ptr().add(16));

                // Process each of the 4 weight rows
                macro_rules! do_row {
                    ($row:expr, $suma:ident, $sumb:ident) => {
                        let x0 = &*vx[$row].add(i);
                        let x1 = &*vx[$row].add(i + 1);

                        // Block 0: unpack Q4_0 nibbles
                        let v0_0 = vld1q_u8(x0.qs.as_ptr());
                        let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                        let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);

                        let d0 = x0.d.to_f32() * y0.d.to_f32();

                        let mut p_0 = vdupq_n_s32(0);
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_0,
                            x = in(vreg) x0_l,
                            y = in(vreg) y0_l,
                        );
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_0,
                            x = in(vreg) x0_h,
                            y = in(vreg) y0_h,
                        );
                        $suma = vmlaq_n_f32($suma, vcvtq_f32_s32(p_0), d0);

                        // Block 1: unpack Q4_0 nibbles
                        let v0_1 = vld1q_u8(x1.qs.as_ptr());
                        let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_1, m4b)), s8b);
                        let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4)), s8b);

                        let d1 = x1.d.to_f32() * y1.d.to_f32();

                        let mut p_1 = vdupq_n_s32(0);
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_1,
                            x = in(vreg) x1_l,
                            y = in(vreg) y1_l,
                        );
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_1,
                            x = in(vreg) x1_h,
                            y = in(vreg) y1_h,
                        );
                        $sumb = vmlaq_n_f32($sumb, vcvtq_f32_s32(p_1), d1);
                    };
                }

                do_row!(0, sum0a, sum0b);
                do_row!(1, sum1a, sum1b);
                do_row!(2, sum2a, sum2b);
                do_row!(3, sum3a, sum3b);

                i += 2;
            }

            // Handle odd remainder block
            if i < nb {
                let y0 = &*vy.add(i);
                let y0_l = vld1q_s8(y0.qs.as_ptr());
                let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));

                macro_rules! do_row_tail {
                    ($row:expr, $suma:ident) => {
                        let x0 = &*vx[$row].add(i);
                        let v0_0 = vld1q_u8(x0.qs.as_ptr());
                        let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                        let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);
                        let d0 = x0.d.to_f32() * y0.d.to_f32();
                        let mut p_0 = vdupq_n_s32(0);
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_0,
                            x = in(vreg) x0_l,
                            y = in(vreg) y0_l,
                        );
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_0,
                            x = in(vreg) x0_h,
                            y = in(vreg) y0_h,
                        );
                        $suma = vmlaq_n_f32($suma, vcvtq_f32_s32(p_0), d0);
                    };
                }

                do_row_tail!(0, sum0a);
                do_row_tail!(1, sum1a);
                do_row_tail!(2, sum2a);
                do_row_tail!(3, sum3a);
            }

            // Horizontal reduction
            out[0] = vaddvq_f32(vaddq_f32(sum0a, sum0b));
            out[1] = vaddvq_f32(vaddq_f32(sum1a, sum1b));
            out[2] = vaddvq_f32(vaddq_f32(sum2a, sum2b));
            out[3] = vaddvq_f32(vaddq_f32(sum3a, sum3b));
        }
    }

    /// 2-row i8mm dot product: weight 2행 × activation 1행 → 결과 2개.
    /// ARM FEAT_I8MM (smmla) 기반. i8mm 미지원 디바이스에서는 호출하지 않을 것.
    ///
    /// `smmla` 시맨틱 (2x8x2 matrix multiply-accumulate):
    ///   acc[0] += dot(a[0:7],  b[0:7])   — weight row 0 × activation low
    ///   acc[1] += dot(a[0:7],  b[8:15])  — weight row 0 × activation high (duplicate)
    ///   acc[2] += dot(a[8:15], b[0:7])   — weight row 1 × activation low
    ///   acc[3] += dot(a[8:15], b[8:15])  — weight row 1 × activation high (duplicate)
    ///
    /// GEMV에서 activation row가 1개이므로 b의 양쪽 절반에 동일 데이터를 복제.
    /// scale = [d_x0*d_y, 0, d_x1*d_y, 0]으로 중복 lane을 제거하여
    /// sumv[0] = row 0 결과, sumv[2] = row 1 결과를 누적.
    ///
    /// # Safety
    /// - `vx0`, `vx1`은 각각 `n / QK8_0`개의 `BlockQ4_0`을 가리키는 유효 포인터
    /// - `vy`는 `n / QK8_0`개의 `BlockQ8_0`을 가리키는 유효 포인터
    /// - 호출자가 FEAT_I8MM 지원을 런타임에 확인해야 함
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q4_0_q8_0_i8mm(
        &self,
        n: usize,
        s0: &mut f32,
        s1: &mut f32,
        vx0: *const BlockQ4_0,
        vx1: *const BlockQ4_0,
        vy: *const BlockQ8_0,
    ) {
        unsafe {
            let nb = n / QK8_0;
            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(0x08);

            // 2-block unroll with two independent float accumulators to hide
            // smmla → vcvt → vmla dependency chain (matches llama.cpp i8mm pattern).
            // Each sumv lane layout: [r0, r0, r1, r1] — scale vector zeroes duplicate
            // lanes so lane 0 holds row-0 sum and lane 2 holds row-1 sum.
            let mut sumv_a = vdupq_n_f32(0.0);
            let mut sumv_b = vdupq_n_f32(0.0);

            // Helper: build interleaved smmla operands for one block pair.
            // Returns (acc int32x4) via the closure-free macro-style inline asm.
            macro_rules! block_i8mm {
                ($i:expr, $sumv:ident) => {{
                    let b_x0 = &*vx0.add($i);
                    let b_x1 = &*vx1.add($i);
                    let b_y = &*vy.add($i);

                    // Unpack Q4_0 nibbles → signed int8 (subtract bias 8)
                    let v0_0 = vld1q_u8(b_x0.qs.as_ptr());
                    let v0_1 = vld1q_u8(b_x1.qs.as_ptr());

                    let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                    let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);
                    let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_1, m4b)), s8b);
                    let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4)), s8b);

                    // Load Q8_0 activation
                    let y_l = vld1q_s8(b_y.qs.as_ptr());
                    let y_h = vld1q_s8(b_y.qs.as_ptr().add(16));

                    let d_x0 = b_x0.d.to_f32();
                    let d_x1 = b_x1.d.to_f32();
                    let d_y = b_y.d.to_f32();

                    // Scale vector constructed in-register (avoids stack-to-load stall).
                    // Layout [d_x0*d_y, 0, d_x1*d_y, 0] masks out duplicate smmla lanes.
                    let s0x = d_x0 * d_y;
                    let s1x = d_x1 * d_y;
                    let scale = vsetq_lane_f32(
                        s1x,
                        vsetq_lane_f32(s0x, vdupq_n_f32(0.0), 0),
                        2,
                    );

                    // Interleave weight rows: a[0:7] = x0 half, a[8:15] = x1 half
                    let l0 = vreinterpretq_s8_s64(vzip1q_s64(
                        vreinterpretq_s64_s8(x0_l),
                        vreinterpretq_s64_s8(x1_l),
                    ));
                    let l1 = vreinterpretq_s8_s64(vzip2q_s64(
                        vreinterpretq_s64_s8(x0_l),
                        vreinterpretq_s64_s8(x1_l),
                    ));
                    let l2 = vreinterpretq_s8_s64(vzip1q_s64(
                        vreinterpretq_s64_s8(x0_h),
                        vreinterpretq_s64_s8(x1_h),
                    ));
                    let l3 = vreinterpretq_s8_s64(vzip2q_s64(
                        vreinterpretq_s64_s8(x0_h),
                        vreinterpretq_s64_s8(x1_h),
                    ));

                    // Duplicate activation for both halves (GEMV: single activation row)
                    let r0 = vreinterpretq_s8_s64(vzip1q_s64(
                        vreinterpretq_s64_s8(y_l),
                        vreinterpretq_s64_s8(y_l),
                    ));
                    let r1 = vreinterpretq_s8_s64(vzip2q_s64(
                        vreinterpretq_s64_s8(y_l),
                        vreinterpretq_s64_s8(y_l),
                    ));
                    let r2 = vreinterpretq_s8_s64(vzip1q_s64(
                        vreinterpretq_s64_s8(y_h),
                        vreinterpretq_s64_s8(y_h),
                    ));
                    let r3 = vreinterpretq_s8_s64(vzip2q_s64(
                        vreinterpretq_s64_s8(y_h),
                        vreinterpretq_s64_s8(y_h),
                    ));

                    let mut acc = vdupq_n_s32(0);
                    std::arch::asm!(
                        ".arch_extension i8mm",
                        "smmla {acc:v}.4s, {a0:v}.16b, {b0:v}.16b",
                        "smmla {acc:v}.4s, {a1:v}.16b, {b1:v}.16b",
                        "smmla {acc:v}.4s, {a2:v}.16b, {b2:v}.16b",
                        "smmla {acc:v}.4s, {a3:v}.16b, {b3:v}.16b",
                        acc = inout(vreg) acc,
                        a0 = in(vreg) l0,
                        b0 = in(vreg) r0,
                        a1 = in(vreg) l1,
                        b1 = in(vreg) r1,
                        a2 = in(vreg) l2,
                        b2 = in(vreg) r2,
                        a3 = in(vreg) l3,
                        b3 = in(vreg) r3,
                    );

                    // acc = [dot(x0,y), dot(x0,y), dot(x1,y), dot(x1,y)]
                    $sumv = vmlaq_f32($sumv, vcvtq_f32_s32(acc), scale);
                }};
            }

            let mut i = 0;
            while i + 1 < nb {
                block_i8mm!(i, sumv_a);
                block_i8mm!(i + 1, sumv_b);
                i += 2;
            }
            if i < nb {
                block_i8mm!(i, sumv_a);
            }

            let sumv = vaddq_f32(sumv_a, sumv_b);
            *s0 = vgetq_lane_f32(sumv, 0);
            *s1 = vgetq_lane_f32(sumv, 2);
        }
    }
}

// --- Q4_0 interleaved weight cache ---
// Lazily interleaves Q4_0 weights into smmla-ready 4-row blocks on first use.
// Key = tensor data pointer (stable for mmap/alloc'd weights), value = interleaved bytes.
static Q4_IL_CACHE: OnceLock<Mutex<HashMap<usize, Vec<u8>>>> = OnceLock::new();

/// Interleaved Q4_0 block for 4 rows: scales + smmla-ready byte pairs.
/// Layout per block (72 bytes, same size as 4 × BlockQ4_0):
///   [d0, d1, d2, d3]: 4 × f16 = 8 bytes
///   Pair 0+1: [r0.qs[0..8], r1.qs[0..8], r0.qs[8..16], r1.qs[8..16]] = 32 bytes
///   Pair 2+3: [r2.qs[0..8], r3.qs[0..8], r2.qs[8..16], r3.qs[8..16]] = 32 bytes
const IL_BLOCK_SIZE: usize = 72; // 8 + 32 + 32

/// Interleave Q4_0 weight matrix [N, K/32 blocks] into 4-row smmla-ready format.
/// N must be a multiple of 4.
fn interleave_q4_0_weights(src: *const BlockQ4_0, nb_k: usize, n: usize) -> Vec<u8> {
    assert!(
        n.is_multiple_of(4),
        "interleave requires N % 4 == 0, got N={n}"
    );
    let n_groups = n / 4;
    let mut out = vec![0u8; n_groups * nb_k * IL_BLOCK_SIZE];

    for g in 0..n_groups {
        let rows = [
            unsafe { src.add((g * 4) * nb_k) },
            unsafe { src.add((g * 4 + 1) * nb_k) },
            unsafe { src.add((g * 4 + 2) * nb_k) },
            unsafe { src.add((g * 4 + 3) * nb_k) },
        ];

        for bi in 0..nb_k {
            let dst_offset = (g * nb_k + bi) * IL_BLOCK_SIZE;
            let dst = &mut out[dst_offset..dst_offset + IL_BLOCK_SIZE];

            // 4 scales (f16, 2 bytes each)
            for r in 0..4 {
                let blk = unsafe { &*rows[r].add(bi) };
                dst[r * 2] = blk.d.to_bits() as u8;
                dst[r * 2 + 1] = (blk.d.to_bits() >> 8) as u8;
            }

            // Pair 0+1: interleave qs for smmla
            let b0 = unsafe { &*rows[0].add(bi) };
            let b1 = unsafe { &*rows[1].add(bi) };
            // First 16 bytes: [r0.qs[0..8], r1.qs[0..8]]
            dst[8..16].copy_from_slice(&b0.qs[0..8]);
            dst[16..24].copy_from_slice(&b1.qs[0..8]);
            // Next 16 bytes: [r0.qs[8..16], r1.qs[8..16]]
            dst[24..32].copy_from_slice(&b0.qs[8..16]);
            dst[32..40].copy_from_slice(&b1.qs[8..16]);

            // Pair 2+3
            let b2 = unsafe { &*rows[2].add(bi) };
            let b3 = unsafe { &*rows[3].add(bi) };
            dst[40..48].copy_from_slice(&b2.qs[0..8]);
            dst[48..56].copy_from_slice(&b3.qs[0..8]);
            dst[56..64].copy_from_slice(&b2.qs[8..16]);
            dst[64..72].copy_from_slice(&b3.qs[8..16]);
        }
    }
    out
}

/// Get interleaved weight data, lazily creating it on first call.
/// Returns pointer to interleaved data and whether it was used (N % 4 == 0).
fn get_interleaved_q4(b: &Tensor, nb_k: usize, n: usize) -> Option<usize> {
    if !n.is_multiple_of(4) {
        return None;
    }
    let key = b.as_ptr() as usize;
    let cache = Q4_IL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();

    if let Some(data) = map.get(&key) {
        return Some(data.as_ptr() as usize);
    }

    let data = interleave_q4_0_weights(b.as_ptr() as *const BlockQ4_0, nb_k, n);
    let ptr = data.as_ptr() as usize;
    map.insert(key, data);
    Some(ptr)
}

impl CpuBackendNeon {
    /// 4-row i8mm dot product on interleaved Q4_0 data.
    /// Weight data is pre-arranged so that vld1q + vand/vshr directly produces
    /// smmla-ready operands — no vzip needed for weights.
    ///
    /// il_ptr: pointer to interleaved block for rows [4*g, 4*g+1, 4*g+2, 4*g+3]
    /// nb_k: number of Q4_0 blocks per row (K / 32)
    /// vy: activation Q8_0 pointer (shared across all 4 rows)
    /// out: 4 output dot products
    #[cfg(target_arch = "aarch64")]
    unsafe fn vec_dot_q4_0_q8_0_i8mm_4rows_il(
        &self,
        nb_k: usize,
        il_ptr: *const u8,
        vy: *const BlockQ8_0,
        out: &mut [f32; 4],
    ) {
        unsafe {
            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(0x08);

            // 4 float accumulators: 2 per row-pair × 2-block unroll.
            // Two independent chains per row-pair break the smmla → vcvt → vmla
            // dependency, matching llama.cpp's i8mm 2-block unroll strategy.
            // Each sumv lane layout: [r_a, r_a, r_b, r_b] — scale zeroes duplicate lanes
            // so lane 0 holds the first row, lane 2 holds the second.
            let mut sumv01_a = vdupq_n_f32(0.0);
            let mut sumv01_b = vdupq_n_f32(0.0);
            let mut sumv23_a = vdupq_n_f32(0.0);
            let mut sumv23_b = vdupq_n_f32(0.0);

            // Single-block processing of an interleaved block, feeding two float
            // accumulators (one for row-pair 0+1, one for row-pair 2+3).
            macro_rules! il_block {
            ($i:expr, $s01:ident, $s23:ident) => {{
                let base = il_ptr.add($i * IL_BLOCK_SIZE);
                let b_y = &*vy.add($i);

                // 4 scales + activation scale
                let d0 = half::f16::from_bits(*(base as *const u16)).to_f32();
                let d1 = half::f16::from_bits(*(base.add(2) as *const u16)).to_f32();
                let d2 = half::f16::from_bits(*(base.add(4) as *const u16)).to_f32();
                let d3 = half::f16::from_bits(*(base.add(6) as *const u16)).to_f32();
                let d_y = b_y.d.to_f32();

                // Activation: load once, duplicate halves for smmla GEMV
                let y_l = vld1q_s8(b_y.qs.as_ptr());
                let y_h = vld1q_s8(b_y.qs.as_ptr().add(16));
                let r0 = vreinterpretq_s8_s64(vzip1q_s64(
                    vreinterpretq_s64_s8(y_l),
                    vreinterpretq_s64_s8(y_l),
                ));
                let r1 = vreinterpretq_s8_s64(vzip2q_s64(
                    vreinterpretq_s64_s8(y_l),
                    vreinterpretq_s64_s8(y_l),
                ));
                let r2 = vreinterpretq_s8_s64(vzip1q_s64(
                    vreinterpretq_s64_s8(y_h),
                    vreinterpretq_s64_s8(y_h),
                ));
                let r3 = vreinterpretq_s8_s64(vzip2q_s64(
                    vreinterpretq_s64_s8(y_h),
                    vreinterpretq_s64_s8(y_h),
                ));

                // Scale vectors built in-register (avoid stack store-to-load stall)
                let scale01 = vsetq_lane_f32(
                    d1 * d_y,
                    vsetq_lane_f32(d0 * d_y, vdupq_n_f32(0.0), 0),
                    2,
                );
                let scale23 = vsetq_lane_f32(
                    d3 * d_y,
                    vsetq_lane_f32(d2 * d_y, vdupq_n_f32(0.0), 0),
                    2,
                );

                // Pair 0+1
                let w01a = vld1q_u8(base.add(8));
                let w01b = vld1q_u8(base.add(24));
                let l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w01a, m4b)), s8b);
                let l1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w01b, m4b)), s8b);
                let l2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w01a, 4)), s8b);
                let l3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w01b, 4)), s8b);

                let mut acc01 = vdupq_n_s32(0);
                std::arch::asm!(
                    ".arch_extension i8mm",
                    "smmla {acc:v}.4s, {a0:v}.16b, {b0:v}.16b",
                    "smmla {acc:v}.4s, {a1:v}.16b, {b1:v}.16b",
                    "smmla {acc:v}.4s, {a2:v}.16b, {b2:v}.16b",
                    "smmla {acc:v}.4s, {a3:v}.16b, {b3:v}.16b",
                    acc = inout(vreg) acc01,
                    a0 = in(vreg) l0, b0 = in(vreg) r0,
                    a1 = in(vreg) l1, b1 = in(vreg) r1,
                    a2 = in(vreg) l2, b2 = in(vreg) r2,
                    a3 = in(vreg) l3, b3 = in(vreg) r3,
                );
                $s01 = vmlaq_f32($s01, vcvtq_f32_s32(acc01), scale01);

                // Pair 2+3: reuses r0..r3 activation halves
                let w23a = vld1q_u8(base.add(40));
                let w23b = vld1q_u8(base.add(56));
                let l0b = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w23a, m4b)), s8b);
                let l1b = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w23b, m4b)), s8b);
                let l2b = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w23a, 4)), s8b);
                let l3b = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w23b, 4)), s8b);

                let mut acc23 = vdupq_n_s32(0);
                std::arch::asm!(
                    ".arch_extension i8mm",
                    "smmla {acc:v}.4s, {a0:v}.16b, {b0:v}.16b",
                    "smmla {acc:v}.4s, {a1:v}.16b, {b1:v}.16b",
                    "smmla {acc:v}.4s, {a2:v}.16b, {b2:v}.16b",
                    "smmla {acc:v}.4s, {a3:v}.16b, {b3:v}.16b",
                    acc = inout(vreg) acc23,
                    a0 = in(vreg) l0b, b0 = in(vreg) r0,
                    a1 = in(vreg) l1b, b1 = in(vreg) r1,
                    a2 = in(vreg) l2b, b2 = in(vreg) r2,
                    a3 = in(vreg) l3b, b3 = in(vreg) r3,
                );
                $s23 = vmlaq_f32($s23, vcvtq_f32_s32(acc23), scale23);
            }};
        }

            let mut i = 0;
            while i + 1 < nb_k {
                il_block!(i, sumv01_a, sumv23_a);
                il_block!(i + 1, sumv01_b, sumv23_b);
                i += 2;
            }
            if i < nb_k {
                il_block!(i, sumv01_a, sumv23_a);
            }

            let sumv01 = vaddq_f32(sumv01_a, sumv01_b);
            let sumv23 = vaddq_f32(sumv23_a, sumv23_b);
            out[0] = vgetq_lane_f32(sumv01, 0);
            out[1] = vgetq_lane_f32(sumv01, 2);
            out[2] = vgetq_lane_f32(sumv23, 0);
            out[3] = vgetq_lane_f32(sumv23, 2);
        }
    }
}

// --- SpinPool work context for F16 GEMV ---

#[repr(C)]
struct F16GemvCtx {
    a_ptr: *const f32,
    a_f16_ptr: *const u16, // Pre-converted A in F16 for the F16 matmul kernels
    b_base: *const u16,
    out_ptr: *mut f32,
    k: usize,
    n: usize,
    rows_per_chunk: usize,
}

// --- N-major GEMM context (prefill, M > 1) ---

/// Context for N-major GEMM dispatch.
/// Instead of dispatching M times (one per activation row), we dispatch once
/// with M × n_chunks tasks. Consecutive chunk_ids share the same B rows,
/// so the same core reuses B data from L1 across all M activation rows.
#[repr(C)]
struct F16GemmCtx {
    a_base: *const f32,     // A[M, K] row-major
    a_f16_base: *const u16, // Pre-converted A[M, K] in F16 for F16 matmul kernels
    b_base: *const u16,     // B[N, K] row-major (F16, transposed)
    out_base: *mut f32,     // Out[M, N] row-major
    m: usize,
    n: usize,
    k: usize,
    rows_per_chunk: usize, // NR-aligned N rows per chunk
}

unsafe impl Send for F16GemmCtx {}
unsafe impl Sync for F16GemmCtx {}

// --- Fused multi-matmul dispatch (QKV / gate+up) ---

/// Context for fused multi-matmul dispatch.
/// Multiple matmuls share input but have different weights/outputs.
/// Chunks are distributed across all matmuls in a single dispatch.
#[repr(C)]
struct FusedGemvCtx {
    a_ptr: *const f32,
    a_f16_ptr: *const u16, // Pre-converted A in F16 for the F16 matmul kernels
    k: usize,
    rows_per_chunk: usize,
    n_matmuls: usize,
    // Per-matmul params (up to 3 for QKV)
    b_bases: [*const u16; 3],
    out_ptrs: [*mut f32; 3],
    ns: [usize; 3],
    chunk_offsets: [usize; 3], // cumulative chunk offset per matmul
}

/// Work function for fused multi-matmul: chunk_id spans across all matmuls.
///
/// # Safety
/// Called by SpinPool workers with valid FusedGemvCtx pointer.
unsafe fn fused_gemv_chunk(ctx_ptr: *const u8, chunk_id: usize) {
    unsafe {
        const NR: usize = 4;
        let ctx = &*(ctx_ptr as *const FusedGemvCtx);

        // Determine which matmul this chunk belongs to.
        let (mat_idx, local_chunk) = if chunk_id < ctx.chunk_offsets[1] {
            (0, chunk_id - ctx.chunk_offsets[0])
        } else if ctx.n_matmuls == 2 || chunk_id < ctx.chunk_offsets[2] {
            (1, chunk_id - ctx.chunk_offsets[1])
        } else {
            (2, chunk_id - ctx.chunk_offsets[2])
        };

        let b_base = ctx.b_bases[mat_idx];
        let out_ptr = ctx.out_ptrs[mat_idx];
        let n = ctx.ns[mat_idx];

        let j_start = local_chunk * ctx.rows_per_chunk;
        let j_end = (j_start + ctx.rows_per_chunk).min(n);

        let mut j = j_start;
        while j + NR <= j_end {
            let b_ptrs = [
                b_base.add(j * ctx.k),
                b_base.add((j + 1) * ctx.k),
                b_base.add((j + 2) * ctx.k),
                b_base.add((j + 3) * ctx.k),
            ];
            let mut results = [0.0f32; NR];
            CpuBackendNeon::vec_dot_f16_native_gemv_4rows(
                ctx.k,
                ctx.a_f16_ptr,
                b_ptrs,
                &mut results,
            );
            std::ptr::copy_nonoverlapping(results.as_ptr(), out_ptr.add(j), NR);
            j += NR;
        }
        while j < j_end {
            *out_ptr.add(j) = CpuBackendNeon::vec_dot_f16_native_gemv_1row(
                ctx.k,
                ctx.a_f16_ptr,
                b_base.add(j * ctx.k),
            );
            j += 1;
        }
    }
}

/// Dispatch multiple F16 matmuls (sharing the same input) as a single SpinPool dispatch.
/// Reduces dispatch overhead from N dispatches to 1.
///
/// # Safety
/// All tensor pointers must be valid. Input must be F32, weights F16, outputs F32.
/// All matmuls must share the same K dimension.
pub unsafe fn fused_matmul_f16(
    a_data: *const f32,
    k: usize,
    matmuls: &[(*const u16, *mut f32, usize)], // (weight_base, out_ptr, n_rows)
) {
    unsafe {
        use crate::thread_pool;
        const NR: usize = 4;

        let pool = thread_pool::get_pool();
        let n_threads = rayon::current_num_threads();
        let total_rows: usize = matmuls.iter().map(|m| m.2).sum();
        let target_chunks = n_threads * 8;
        let rows_per_chunk = total_rows.div_ceil(target_chunks).div_ceil(NR) * NR;
        let rows_per_chunk = rows_per_chunk.max(NR);

        // Pre-convert A from F32 to F16 for the F16 matmul kernels
        let mut a_f16_buf: Vec<u16> = vec![0u16; k];
        CpuBackendNeon::f32_to_f16_neon(a_data, a_f16_buf.as_mut_ptr(), k);

        let mut b_bases = [std::ptr::null::<u16>(); 3];
        let mut out_ptrs = [std::ptr::null_mut::<f32>(); 3];
        let mut ns = [0usize; 3];
        let mut chunk_offsets = [0usize; 3];
        let mut total_chunks = 0;

        for (i, &(b, o, n)) in matmuls.iter().enumerate().take(3) {
            b_bases[i] = b;
            out_ptrs[i] = o;
            ns[i] = n;
            chunk_offsets[i] = total_chunks;
            total_chunks += n.div_ceil(rows_per_chunk);
        }
        // Sentinel: unused matmul slots route past `total_chunks` so the dispatch
        // never falls through to them. `fused_gemv_chunk`'s `else` branch maps
        // chunk_id ≥ chunk_offsets[2] to mat 2 — without this, n_matmuls<3 leaves
        // chunk_offsets[1..n_matmuls..3] at 0 and chunks misroute into mat 2 with
        // zero-length output, producing silent zero-fill.
        for slot in chunk_offsets.iter_mut().skip(matmuls.len()) {
            *slot = total_chunks;
        }

        let ctx = FusedGemvCtx {
            a_ptr: a_data,
            a_f16_ptr: a_f16_buf.as_ptr(),
            k,
            rows_per_chunk,
            n_matmuls: matmuls.len(),
            b_bases,
            out_ptrs,
            ns,
            chunk_offsets,
        };

        pool.dispatch(
            total_chunks,
            fused_gemv_chunk,
            &ctx as *const FusedGemvCtx as *const u8,
        );
    }
}

/// Fused multi-matmul for Q4_0 decode (M=1): single Q8_0 quantization + single Rayon dispatch.
/// Combines gate+up (or QKV) into one parallel region, avoiding redundant quantization
/// and Rayon dispatch overhead.
///
/// `matmuls`: slice of (weight_base, out_ptr, n_rows) — up to 3 matmuls.
///
/// # Safety
/// All weight pointers must be valid BlockQ4_0 arrays. All out_ptrs must have capacity >= n_rows.
/// `a_data` must point to a valid f32 slice of length `k`.
pub unsafe fn fused_matmul_q4_0(
    a_data: *const f32,
    k: usize,
    matmuls: &[(*const BlockQ4_0, *mut f32, usize)],
) {
    unsafe {
        let nb_k = k / QK4_0;
        let nb_k_q8 = k / QK8_0;
        let backend = CpuBackendNeon::new();

        // 1. Quantize activation to Q8_0 — once for all matmuls (reuse thread-local workspace)
        let a_q8_usize: usize = Q8_WORKSPACE.with(|ws| {
            let mut a_q8 = ws.borrow_mut();
            if a_q8.len() < nb_k_q8 {
                let additional = nb_k_q8 - a_q8.len();
                a_q8.reserve(additional);
                a_q8.set_len(nb_k_q8);
            }
            let a_row = std::slice::from_raw_parts(a_data, k);
            backend.quantize_row_q8_0(a_row, &mut a_q8[..nb_k_q8], k);
            a_q8.as_ptr() as usize
        });

        // 2. Build fused context — cast pointers to usize for Rayon Send
        let mut b_bases = [0usize; 3];
        let mut out_ptrs = [0usize; 3];
        let mut ns = [0usize; 3];
        let n_matmuls = matmuls.len();

        for (i, &(b, o, n)) in matmuls.iter().enumerate().take(3) {
            b_bases[i] = b as usize;
            out_ptrs[i] = o as usize;
            ns[i] = n;
        }

        // Build cumulative row offsets for matmul boundaries
        let mut row_offsets = [0usize; 4];
        for i in 0..n_matmuls {
            row_offsets[i + 1] = row_offsets[i] + ns[i];
        }
        let total_rows = row_offsets[n_matmuls];

        // Adaptive chunk size: target ~4 tasks per thread
        let num_threads = rayon::current_num_threads();
        let target_tasks = num_threads * 4;
        let chunk_size = {
            let cs = total_rows.div_ceil(target_tasks);
            (cs + 1) & !1 // round up to even for i8mm 2-row pairing
        };
        let n_chunks = total_rows.div_ceil(chunk_size);

        #[cfg(target_arch = "aarch64")]
        let use_i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        #[cfg(not(target_arch = "aarch64"))]
        let use_i8mm = false;

        // 3. Rayon dispatch: write directly to output pointers (no intermediate buffer)
        (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
            let global_start = chunk_idx * chunk_size;
            let global_end = (global_start + chunk_size).min(total_rows);
            let a_q8_ptr = a_q8_usize as *const BlockQ8_0;

            if use_i8mm {
                let mut gi = global_start;
                while gi < global_end {
                    let mut mat_idx = 0;
                    while mat_idx + 1 < n_matmuls && gi >= row_offsets[mat_idx + 1] {
                        mat_idx += 1;
                    }
                    let j = gi - row_offsets[mat_idx];
                    let b_base = b_bases[mat_idx] as *const BlockQ4_0;
                    let out_base = out_ptrs[mat_idx] as *mut f32;

                    // i8mm 2-row: pair adjacent rows within same matmul
                    if gi + 1 < global_end && gi + 1 < row_offsets[mat_idx + 1] {
                        let b0 = unsafe { b_base.add(j * nb_k) };
                        let b1 = unsafe { b_base.add((j + 1) * nb_k) };
                        let mut s0 = 0.0f32;
                        let mut s1 = 0.0f32;
                        unsafe {
                            backend.vec_dot_q4_0_q8_0_i8mm(k, &mut s0, &mut s1, b0, b1, a_q8_ptr);
                            *out_base.add(j) = s0;
                            *out_base.add(j + 1) = s1;
                        }
                        gi += 2;
                    } else {
                        let b_ptr = unsafe { b_base.add(j * nb_k) };
                        let mut sum = 0.0f32;
                        unsafe {
                            backend.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_ptr, a_q8_ptr);
                            *out_base.add(j) = sum;
                        }
                        gi += 1;
                    }
                }
            } else {
                for gi in global_start..global_end {
                    let mut mat_idx = 0;
                    while mat_idx + 1 < n_matmuls && gi >= row_offsets[mat_idx + 1] {
                        mat_idx += 1;
                    }
                    let j = gi - row_offsets[mat_idx];
                    let b_base = b_bases[mat_idx] as *const BlockQ4_0;
                    let out_base = out_ptrs[mat_idx] as *mut f32;
                    let b_ptr = unsafe { b_base.add(j * nb_k) };

                    let mut sum = 0.0f32;
                    unsafe {
                        if std::arch::is_aarch64_feature_detected!("dotprod") {
                            backend.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_ptr, a_q8_ptr);
                        } else {
                            backend.vec_dot_q4_0_q8_0(k, &mut sum, b_ptr, a_q8_ptr);
                        }
                        *out_base.add(j) = sum;
                    }
                }
            }
        });
    }
}

/// Work function for SpinPool: processes a coarse chunk of output rows.
/// Each chunk contains multiple NR=4 row blocks to reduce atomic contention.
///
/// # Safety
/// Called by SpinPool workers with valid F16GemvCtx pointer.
unsafe fn f16_gemv_chunk(ctx_ptr: *const u8, chunk_id: usize) {
    unsafe {
        const NR: usize = 4;
        let ctx = &*(ctx_ptr as *const F16GemvCtx);
        let j_start = chunk_id * ctx.rows_per_chunk;
        let j_end = (j_start + ctx.rows_per_chunk).min(ctx.n);

        let mut j = j_start;
        while j + NR <= j_end {
            let b_ptrs = [
                ctx.b_base.add(j * ctx.k),
                ctx.b_base.add((j + 1) * ctx.k),
                ctx.b_base.add((j + 2) * ctx.k),
                ctx.b_base.add((j + 3) * ctx.k),
            ];
            let mut results = [0.0f32; NR];
            CpuBackendNeon::vec_dot_f16_native_gemv_4rows(
                ctx.k,
                ctx.a_f16_ptr,
                b_ptrs,
                &mut results,
            );
            std::ptr::copy_nonoverlapping(results.as_ptr(), ctx.out_ptr.add(j), NR);
            j += NR;
        }
        while j < j_end {
            *ctx.out_ptr.add(j) = CpuBackendNeon::vec_dot_f16_native_gemv_1row(
                ctx.k,
                ctx.a_f16_ptr,
                ctx.b_base.add(j * ctx.k),
            );
            j += 1;
        }
    }
}

/// N-major GEMM with RM=4 tiling + native F16 FMA.
/// Processes (n_chunk, m_group) where m_group = group of 4 A rows.
/// 4×4 micro-tile: B loaded once, shared across 4 A rows.
/// F16 accumulators enable 16-output tile in 16 registers.
///
/// Pipeline: Load=8(4cy), FMA=16(4cy) → balanced @ 4cy, 4.0 output/cy
///
/// # Safety
/// Called by SpinPool workers with valid F16GemmCtx pointer.
unsafe fn f16_gemm_chunk(ctx_ptr: *const u8, chunk_id: usize) {
    unsafe {
        const NR: usize = 4;
        const MR: usize = 4;
        let ctx = &*(ctx_ptr as *const F16GemmCtx);

        // MR=4 mapping: chunk_id = n_chunk_idx * m_groups + m_group
        let m_groups = ctx.m.div_ceil(MR);
        let n_chunk_idx = chunk_id / m_groups;
        let m_group = chunk_id % m_groups;
        let m_row0 = m_group * MR;
        let m_remaining = ctx.m - m_row0; // 1..=4 rows in this group

        let j_start = n_chunk_idx * ctx.rows_per_chunk;
        let j_end = (j_start + ctx.rows_per_chunk).min(ctx.n);

        if m_remaining >= MR {
            // Full MR=4 tile: use native F16 FMA 4×4 kernel
            let a_ptrs = [
                ctx.a_f16_base.add(m_row0 * ctx.k),
                ctx.a_f16_base.add((m_row0 + 1) * ctx.k),
                ctx.a_f16_base.add((m_row0 + 2) * ctx.k),
                ctx.a_f16_base.add((m_row0 + 3) * ctx.k),
            ];
            let out_ptrs = [
                ctx.out_base.add(m_row0 * ctx.n),
                ctx.out_base.add((m_row0 + 1) * ctx.n),
                ctx.out_base.add((m_row0 + 2) * ctx.n),
                ctx.out_base.add((m_row0 + 3) * ctx.n),
            ];

            let mut j = j_start;
            while j + NR <= j_end {
                let b_ptrs = [
                    ctx.b_base.add(j * ctx.k),
                    ctx.b_base.add((j + 1) * ctx.k),
                    ctx.b_base.add((j + 2) * ctx.k),
                    ctx.b_base.add((j + 3) * ctx.k),
                ];
                let mut results = [[0.0f32; NR]; MR];
                CpuBackendNeon::vec_dot_f16_native_4x4(ctx.k, a_ptrs, b_ptrs, &mut results);
                for r in 0..MR {
                    std::ptr::copy_nonoverlapping(results[r].as_ptr(), out_ptrs[r].add(j), NR);
                }
                j += NR;
            }
            // Tail: remaining B columns (< NR)
            while j < j_end {
                for r in 0..MR {
                    *out_ptrs[r].add(j) = CpuBackendNeon::vec_dot_f16_native_gemv_1row(
                        ctx.k,
                        a_ptrs[r],
                        ctx.b_base.add(j * ctx.k),
                    );
                }
                j += 1;
            }
        } else {
            // Tail group (1-3 rows): fallback to per-row GEMV kernel
            for r in 0..m_remaining {
                let a_f16 = ctx.a_f16_base.add((m_row0 + r) * ctx.k);
                let out_ptr = ctx.out_base.add((m_row0 + r) * ctx.n);
                let mut j = j_start;
                while j + NR <= j_end {
                    let b_ptrs = [
                        ctx.b_base.add(j * ctx.k),
                        ctx.b_base.add((j + 1) * ctx.k),
                        ctx.b_base.add((j + 2) * ctx.k),
                        ctx.b_base.add((j + 3) * ctx.k),
                    ];
                    let mut results = [0.0f32; NR];
                    CpuBackendNeon::vec_dot_f16_native_gemv_4rows(
                        ctx.k,
                        a_f16,
                        b_ptrs,
                        &mut results,
                    );
                    std::ptr::copy_nonoverlapping(results.as_ptr(), out_ptr.add(j), NR);
                    j += NR;
                }
                while j < j_end {
                    *out_ptr.add(j) = CpuBackendNeon::vec_dot_f16_native_gemv_1row(
                        ctx.k,
                        a_f16,
                        ctx.b_base.add(j * ctx.k),
                    );
                    j += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3 — NEON flash-attention **prefill** kernel.
// ---------------------------------------------------------------------------
//
// Tile-based Q·K^T → online softmax → P·V with per-row state. Task grid is
// `(tile_i, kv_h)`; each worker processes the full GQA Q-head group sharing
// that kv-head, re-using the K/V tile while iterating over Q-heads.
//
// Designed as a drop-in replacement for the scalar rayon path inside
// `layers::attention::flash_attention_forward_strided` when:
//   * `q_len >= PREFILL_FLASH_THRESHOLD`
//   * HeadMajor KV (`kv_head_stride == capacity * head_dim` contiguous KV rows)
//   * `head_dim % 8 == 0`
//   * `n_heads_q % n_heads_kv == 0`
//
// The caller passes F32 Q/K/V slices with the row stride conventions the
// scalar path uses.  F16 KV is still dequantized by the caller today (1st-
// PR scope); a future PR may add a direct-F16 NEON path.
//
// Invariants:
//   * Causal mask: `global_c <= global_r + q_start_pos`.
//   * Optional sliding window: `global_c + window_size > global_r + q_start_pos`.
//   * All `(m, l, O)` state is F32.
//   * `scores_out` diagnostic path is unsupported — caller routes to
//     scalar when requested.

/// Prefill flash threshold — below this `q_len` we keep the scalar rayon
/// path for short prompts where tile dispatch overhead dominates.
pub const PREFILL_FLASH_THRESHOLD: usize = 128;

/// Per-(tile_i, kv_h) task context for the prefill NEON kernel.
///
/// Shared across SpinPool workers by reference; each worker derives its
/// `(tile_i, kv_h)` from `task_id` and writes exclusively to its own
/// `out[tile_row_slice, gqa_group]` region.
///
/// # Safety
/// All raw pointers must outlive the enclosing dispatch.  Output regions
/// are partitioned such that no two workers touch the same element.
struct PrefillFlashCtx {
    q_ptr: *const f32,
    k_ptr: *const f32,
    v_ptr: *const f32,
    out_ptr: *mut f32,
    n_heads_kv: usize,
    gqa_ratio: usize,
    head_dim: usize,
    q_len: usize,
    kv_len: usize,
    q_row_stride: usize,
    k_pos_stride: usize,
    v_pos_stride: usize,
    out_row_stride: usize,
    kv_head_stride: usize,
    q_start_pos: usize,
    br: usize,
    bc: usize,
    scale: f32,
    window_size_plus1: u32, // 0 means "None" (full causal).
}

unsafe impl Send for PrefillFlashCtx {}
unsafe impl Sync for PrefillFlashCtx {}

/// Tile classification for causal-mask aware traversal.
///
/// - `Full`: every `(r, c)` in the tile is lower-triangular and within the
///   optional sliding window — mask-free inner kernel.
/// - `Skip`: every `(r, c)` is upper-triangular (or outside the window) —
///   the worker simply continues.
/// - `Diagonal`: boundary tile — per-row lane predicate needed.
#[derive(Copy, Clone, Debug, PartialEq)]
enum TileKind {
    Full,
    Diagonal,
    Skip,
}

/// Classify `(tile_i, tile_j)` relative to the causal triangle and optional
/// sliding window.  `r_start/r_end` are local query tile bounds,
/// `c_start/c_end` are global key tile bounds.  `q_start_pos` biases local
/// queries into global coordinates: `global_r = r + q_start_pos`.
#[inline]
fn classify_tile(
    r_start: usize,
    r_end: usize,
    c_start: usize,
    c_end: usize,
    q_start_pos: usize,
    window_size: Option<usize>,
) -> TileKind {
    // Upper-triangular entirely?  min global_r in tile = r_start + q_start_pos.
    // Every key masked iff c_start > max_global_r = (r_end - 1) + q_start_pos.
    // AND (for window) c_end + ws <= min_global_r + 1 (all keys too old).
    let max_global_r = (r_end - 1) + q_start_pos;
    let min_global_r = r_start + q_start_pos;

    // Skip if every (r, c) in the tile has global_c > global_r (upper triangle).
    let upper_skip = c_start > max_global_r;
    // Skip if every key falls outside the sliding window (all too old).
    let window_skip = match window_size {
        Some(ws) => c_end + ws <= min_global_r + 1,
        None => false,
    };
    if upper_skip || window_skip {
        return TileKind::Skip;
    }

    // Full iff every (r, c) satisfies causal AND window.
    // max_global_c = c_end - 1.  Causal full: c_end - 1 <= min_global_r,
    // i.e. c_end <= min_global_r + 1.
    let causal_full = c_end <= min_global_r + 1;
    // Window full: for every r, min_global_c in tile = c_start, must satisfy
    // c_start + ws > max_global_r.
    let window_full = match window_size {
        Some(ws) => c_start + ws > max_global_r,
        None => true,
    };
    if causal_full && window_full {
        TileKind::Full
    } else {
        TileKind::Diagonal
    }
}

/// NEON F32 dot-product kernel for a single (Q row, K row) pair of length
/// `head_dim`, unrolled to process 16 lanes per iteration.
///
/// # Safety
/// * `q`, `k` must point to at least `head_dim` F32 lanes.
/// * `head_dim % 4 == 0` (prefill caller guarantees `head_dim % 8 == 0`).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn qk_dot_f32_neon(q: *const f32, k: *const f32, head_dim: usize) -> f32 {
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= head_dim {
            let q0 = vld1q_f32(q.add(i));
            let q1 = vld1q_f32(q.add(i + 4));
            let q2 = vld1q_f32(q.add(i + 8));
            let q3 = vld1q_f32(q.add(i + 12));
            let k0 = vld1q_f32(k.add(i));
            let k1 = vld1q_f32(k.add(i + 4));
            let k2 = vld1q_f32(k.add(i + 8));
            let k3 = vld1q_f32(k.add(i + 12));
            acc0 = vfmaq_f32(acc0, q0, k0);
            acc1 = vfmaq_f32(acc1, q1, k1);
            acc2 = vfmaq_f32(acc2, q2, k2);
            acc3 = vfmaq_f32(acc3, q3, k3);
            i += 16;
        }
        while i + 8 <= head_dim {
            let q0 = vld1q_f32(q.add(i));
            let q1 = vld1q_f32(q.add(i + 4));
            let k0 = vld1q_f32(k.add(i));
            let k1 = vld1q_f32(k.add(i + 4));
            acc0 = vfmaq_f32(acc0, q0, k0);
            acc1 = vfmaq_f32(acc1, q1, k1);
            i += 8;
        }
        while i + 4 <= head_dim {
            let q0 = vld1q_f32(q.add(i));
            let k0 = vld1q_f32(k.add(i));
            acc0 = vfmaq_f32(acc0, q0, k0);
            i += 4;
        }
        let sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut tail = vaddvq_f32(sum);
        while i < head_dim {
            tail += *q.add(i) * *k.add(i);
            i += 1;
        }
        tail
    }
}

/// NEON FMA loop: `o[0..head_dim] *= alpha`.
///
/// # Safety
/// `o` must point to `head_dim` writable F32 lanes; `head_dim % 4 == 0`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn rescale_o_neon(o: *mut f32, alpha: f32, head_dim: usize) {
    unsafe {
        let a = vdupq_n_f32(alpha);
        let mut i = 0;
        while i + 16 <= head_dim {
            let o0 = vld1q_f32(o.add(i));
            let o1 = vld1q_f32(o.add(i + 4));
            let o2 = vld1q_f32(o.add(i + 8));
            let o3 = vld1q_f32(o.add(i + 12));
            vst1q_f32(o.add(i), vmulq_f32(o0, a));
            vst1q_f32(o.add(i + 4), vmulq_f32(o1, a));
            vst1q_f32(o.add(i + 8), vmulq_f32(o2, a));
            vst1q_f32(o.add(i + 12), vmulq_f32(o3, a));
            i += 16;
        }
        while i + 4 <= head_dim {
            let x = vld1q_f32(o.add(i));
            vst1q_f32(o.add(i), vmulq_f32(x, a));
            i += 4;
        }
        while i < head_dim {
            *o.add(i) *= alpha;
            i += 1;
        }
    }
}

/// NEON FMA loop: `o[0..head_dim] += beta * v[0..head_dim]`.
///
/// # Safety
/// `o`, `v` must point to `head_dim` F32 lanes; `head_dim % 4 == 0`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fma_o_v_neon(o: *mut f32, v: *const f32, beta: f32, head_dim: usize) {
    unsafe {
        let b = vdupq_n_f32(beta);
        let mut i = 0;
        while i + 16 <= head_dim {
            let o0 = vld1q_f32(o.add(i));
            let o1 = vld1q_f32(o.add(i + 4));
            let o2 = vld1q_f32(o.add(i + 8));
            let o3 = vld1q_f32(o.add(i + 12));
            let v0 = vld1q_f32(v.add(i));
            let v1 = vld1q_f32(v.add(i + 4));
            let v2 = vld1q_f32(v.add(i + 8));
            let v3 = vld1q_f32(v.add(i + 12));
            vst1q_f32(o.add(i), vfmaq_f32(o0, b, v0));
            vst1q_f32(o.add(i + 4), vfmaq_f32(o1, b, v1));
            vst1q_f32(o.add(i + 8), vfmaq_f32(o2, b, v2));
            vst1q_f32(o.add(i + 12), vfmaq_f32(o3, b, v3));
            i += 16;
        }
        while i + 4 <= head_dim {
            let o0 = vld1q_f32(o.add(i));
            let v0 = vld1q_f32(v.add(i));
            vst1q_f32(o.add(i), vfmaq_f32(o0, b, v0));
            i += 4;
        }
        while i < head_dim {
            *o.add(i) += beta * *v.add(i);
            i += 1;
        }
    }
}

/// Per-row online softmax state carried across K-tile iterations.
#[derive(Clone, Copy)]
struct PrefillRowState {
    m: f32,
    l: f32,
}

impl PrefillRowState {
    const EMPTY: PrefillRowState = PrefillRowState {
        m: f32::NEG_INFINITY,
        l: 0.0,
    };
}

/// Prefill SpinPool worker: process one `(tile_i, kv_h)` pair end-to-end.
///
/// The worker holds thread-local scratch:
/// * `o_tile[gqa_ratio * br * head_dim]`: weighted-V accumulator (F32).
/// * `state[gqa_ratio * br]`: per-row (m, l) state.
/// * `s_tile[br * bc]`: per-tile raw score buffer.
/// * `p_tile[br * bc]`: per-tile softmax-weighted probability buffer.
///
/// These are stack-allocated small buffers for the state, but `o_tile`
/// and `s_tile`/`p_tile` grow with tile dim so we Vec them once per worker
/// invocation (fast paths: Vec::with_capacity then reset to zero in-place).
///
/// # Safety
/// * `ctx` must be a live `*const PrefillFlashCtx`.
/// * `task_id < n_tile_rows * n_heads_kv`.
/// * The output region for `(tile_i, kv_h)` is disjoint from all other
///   workers' regions.
unsafe fn prefill_tile_worker(ctx: *const u8, task_id: usize) {
    unsafe {
        let ctx = &*(ctx as *const PrefillFlashCtx);
        let n_heads_kv = ctx.n_heads_kv;
        let gqa = ctx.gqa_ratio;
        let head_dim = ctx.head_dim;
        let br = ctx.br;
        let bc = ctx.bc;

        let tile_i = task_id / n_heads_kv;
        let kv_h = task_id % n_heads_kv;
        let r_start = tile_i * br;
        let r_end = (r_start + br).min(ctx.q_len);
        if r_start >= r_end {
            return;
        }
        let cur_br = r_end - r_start;
        let q_h_begin = kv_h * gqa;

        let rows = gqa * cur_br;
        // Per-row running softmax state for all `gqa × cur_br` queries in
        // this (tile_i, kv_h) task.
        let mut state = vec![PrefillRowState::EMPTY; rows];
        // Weighted-V accumulator, laid out as `[q_h_within_group][r][d]`.
        let mut o_tile = vec![0.0f32; rows * head_dim];
        // Per-(tile_j) S and P scratch.
        let mut s_tile = vec![0.0f32; cur_br * bc];
        let mut p_tile = vec![0.0f32; cur_br * bc];

        let tc = ctx.kv_len.div_ceil(bc);
        let ws = if ctx.window_size_plus1 == 0 {
            None
        } else {
            Some((ctx.window_size_plus1 - 1) as usize)
        };

        for tj in 0..tc {
            let c_start = tj * bc;
            let c_end = (c_start + bc).min(ctx.kv_len);
            let cur_bc = c_end - c_start;

            let kind = classify_tile(r_start, r_end, c_start, c_end, ctx.q_start_pos, ws);
            if kind == TileKind::Skip {
                continue;
            }

            // Process Q-heads in the GQA group.  Inner K-tile data is
            // shared across heads, maximising L2 reuse.
            for g in 0..gqa {
                let q_h = q_h_begin + g;
                let q_base = ctx.q_ptr.add(r_start * ctx.q_row_stride + q_h * head_dim);
                let k_base = ctx
                    .k_ptr
                    .add(kv_h * ctx.kv_head_stride + c_start * ctx.k_pos_stride);
                let v_base = ctx
                    .v_ptr
                    .add(kv_h * ctx.kv_head_stride + c_start * ctx.v_pos_stride);
                let o_group_base = o_tile.as_mut_ptr().add(g * cur_br * head_dim);
                let state_group = &mut state[g * cur_br..g * cur_br + cur_br];

                // ---- Kernel A: compute S = Q · K^T (scaled) ----
                for r in 0..cur_br {
                    let q_row = q_base.add(r * ctx.q_row_stride);
                    let s_row = s_tile.as_mut_ptr().add(r * bc);
                    for c in 0..cur_bc {
                        let k_row = k_base.add(c * ctx.k_pos_stride);
                        let dot = qk_dot_f32_neon(q_row, k_row, head_dim);
                        *s_row.add(c) = dot * ctx.scale;
                    }
                }

                // ---- Kernel B: per-row online softmax + P · V ----
                for r in 0..cur_br {
                    let global_r = r_start + r + ctx.q_start_pos;
                    let s_row = s_tile.as_ptr().add(r * bc);
                    let p_row = p_tile.as_mut_ptr().add(r * bc);

                    // Mask + row_max.
                    let mut row_max = f32::NEG_INFINITY;
                    let mut any_valid = false;
                    match kind {
                        TileKind::Full => {
                            // No mask; all lanes valid.
                            for c in 0..cur_bc {
                                let v = *s_row.add(c);
                                if v.is_finite() {
                                    if v > row_max {
                                        row_max = v;
                                    }
                                    any_valid = true;
                                }
                            }
                        }
                        TileKind::Diagonal => {
                            // Per-lane predicate.
                            for c in 0..cur_bc {
                                let global_c = c_start + c;
                                let causal_ok = global_c <= global_r;
                                let window_ok = match ws {
                                    Some(w) => global_c + w > global_r,
                                    None => true,
                                };
                                if causal_ok && window_ok {
                                    let v = *s_row.add(c);
                                    if v.is_finite() {
                                        if v > row_max {
                                            row_max = v;
                                        }
                                        any_valid = true;
                                    }
                                }
                            }
                        }
                        TileKind::Skip => unreachable!(),
                    }

                    if !any_valid {
                        // No finite valid key this tile — state untouched.
                        for c in 0..cur_bc {
                            *p_row.add(c) = 0.0;
                        }
                        continue;
                    }

                    let st = &mut state_group[r];
                    let m_prev = st.m;
                    let m_new = m_prev.max(row_max);

                    // Build P row.
                    let mut l_part = 0.0f32;
                    match kind {
                        TileKind::Full => {
                            for c in 0..cur_bc {
                                let sv = *s_row.add(c);
                                let p = if sv.is_finite() {
                                    (sv - m_new).exp()
                                } else {
                                    0.0
                                };
                                *p_row.add(c) = p;
                                l_part += p;
                            }
                        }
                        TileKind::Diagonal => {
                            for c in 0..cur_bc {
                                let global_c = c_start + c;
                                let causal_ok = global_c <= global_r;
                                let window_ok = match ws {
                                    Some(w) => global_c + w > global_r,
                                    None => true,
                                };
                                let sv = *s_row.add(c);
                                let p = if causal_ok && window_ok && sv.is_finite() {
                                    (sv - m_new).exp()
                                } else {
                                    0.0
                                };
                                *p_row.add(c) = p;
                                l_part += p;
                            }
                        }
                        TileKind::Skip => unreachable!(),
                    }

                    let alpha = if m_prev == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (m_prev - m_new).exp()
                    };

                    let o_row = o_group_base.add(r * head_dim);
                    if alpha != 1.0 {
                        rescale_o_neon(o_row, alpha, head_dim);
                    }

                    // O += Σ_c p[c] · V[c, :]
                    for c in 0..cur_bc {
                        let p = *p_row.add(c);
                        if p != 0.0 {
                            let v_row = v_base.add(c * ctx.v_pos_stride);
                            fma_o_v_neon(o_row, v_row, p, head_dim);
                        }
                    }

                    st.m = m_new;
                    st.l = st.l * alpha + l_part;
                }
            }
        }

        // Normalise and write out[r, q_h, :] for each (g, r) in this task.
        for g in 0..gqa {
            let q_h = q_h_begin + g;
            for r in 0..cur_br {
                let st = state[g * cur_br + r];
                let o_row = o_tile.as_ptr().add((g * cur_br + r) * head_dim);
                let out_row = ctx
                    .out_ptr
                    .add((r_start + r) * ctx.out_row_stride + q_h * head_dim);
                if st.l == 0.0 || !st.l.is_finite() {
                    // No valid key contributed — emit zero vector (matches
                    // the reference when the row is fully masked).
                    std::ptr::write_bytes(out_row, 0, head_dim);
                    continue;
                }
                let inv_l = 1.0 / st.l;
                let inv_v = vdupq_n_f32(inv_l);
                let mut i = 0;
                while i + 16 <= head_dim {
                    let o0 = vld1q_f32(o_row.add(i));
                    let o1 = vld1q_f32(o_row.add(i + 4));
                    let o2 = vld1q_f32(o_row.add(i + 8));
                    let o3 = vld1q_f32(o_row.add(i + 12));
                    vst1q_f32(out_row.add(i), vmulq_f32(o0, inv_v));
                    vst1q_f32(out_row.add(i + 4), vmulq_f32(o1, inv_v));
                    vst1q_f32(out_row.add(i + 8), vmulq_f32(o2, inv_v));
                    vst1q_f32(out_row.add(i + 12), vmulq_f32(o3, inv_v));
                    i += 16;
                }
                while i + 4 <= head_dim {
                    let x = vld1q_f32(o_row.add(i));
                    vst1q_f32(out_row.add(i), vmulq_f32(x, inv_v));
                    i += 4;
                }
                while i < head_dim {
                    *out_row.add(i) = *o_row.add(i) * inv_l;
                    i += 1;
                }
            }
        }
    }
}

/// Entry point: tile + online-softmax prefill flash attention on NEON.
///
/// Signature mirrors `layers::attention::flash_attention_forward_strided`.
/// See that function's doc-comment for stride conventions.  The caller
/// should perform the threshold / layout checks before dispatching here.
///
/// `br` / `bc` from the caller are *advisory*: the kernel picks its own
/// tile size based on `head_dim` to maximise L1/L2 residency.  The caller's
/// values are used only when they are already tile-friendly and large
/// enough to saturate NEON throughput (not enforced in this PR).
///
/// # Safety
/// Slices must correspond to the HeadMajor KV layout and the Q / out row
/// strides documented in `flash_attention_forward_strided`.
#[allow(clippy::too_many_arguments)]
pub fn flash_prefill_forward_f32_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    q_row_stride: usize,
    k_pos_stride: usize,
    v_pos_stride: usize,
    out_row_stride: usize,
    kv_head_stride: usize,
    q_start_pos: usize,
    window_size: Option<usize>,
) {
    debug_assert_eq!(n_heads_q % n_heads_kv, 0);
    debug_assert_eq!(head_dim % 8, 0);
    debug_assert!(q_len > 0);

    // Adaptive tile sizes per Architect §4.1.  `br × bc` chosen to fit the
    // (Q tile, K/V tile, S tile, O tile) working set inside L1/L2 while
    // giving enough FMA work to amortise dispatch.
    let (br, bc) = if head_dim >= 96 { (64, 64) } else { (128, 64) };

    let scale = 1.0 / (head_dim as f32).sqrt();
    let gqa_ratio = n_heads_q / n_heads_kv;

    // Encode `window_size` as a small unsigned so the ctx stays Copy/POD
    // and worker code can branch on `ws == 0` → None.  `ws + 1` avoids
    // collision with ws = 0 itself.
    let window_size_plus1 = match window_size {
        Some(w) => {
            debug_assert!(w < u32::MAX as usize - 1);
            (w as u32) + 1
        }
        None => 0,
    };

    let _ = n_heads_q; // field is only used for the debug_assert below.
    let ctx = PrefillFlashCtx {
        q_ptr: q.as_ptr(),
        k_ptr: k.as_ptr(),
        v_ptr: v.as_ptr(),
        out_ptr: out.as_mut_ptr(),
        n_heads_kv,
        gqa_ratio,
        head_dim,
        q_len,
        kv_len,
        q_row_stride,
        k_pos_stride,
        v_pos_stride,
        out_row_stride,
        kv_head_stride,
        q_start_pos,
        br,
        bc,
        scale,
        window_size_plus1,
    };

    let n_tile_rows = q_len.div_ceil(br);
    let n_tasks = n_tile_rows * n_heads_kv;
    let pool = thread_pool::get_pool();
    // Safety:
    // * `prefill_tile_worker` interprets `*const u8` as
    //   `*const PrefillFlashCtx` (what we pass).
    // * `ctx` is a stack value borrowed immutably by `dispatch` which
    //   blocks until every worker finishes.
    // * Task output regions are partitioned by `(tile_i, kv_h)` → distinct
    //   `(r_row, q_h_in_group)` output slices.
    unsafe {
        pool.dispatch(
            n_tasks,
            prefill_tile_worker as WorkFn,
            &ctx as *const PrefillFlashCtx as *const u8,
        );
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use std::sync::Arc;

    /// Scalar F32 reference dot product (ground truth).
    fn ref_dot_f32(a_f16: &[half::f16], b_f16: &[half::f16]) -> f32 {
        assert_eq!(a_f16.len(), b_f16.len());
        let mut sum = 0.0f32;
        for (&a, &b) in a_f16.iter().zip(b_f16.iter()) {
            sum += a.to_f32() * b.to_f32();
        }
        sum
    }

    /// Deterministic pseudo-random F16 vector in range [lo, hi].
    /// Uses a splitmix64-style hash to avoid pulling in `rand` as dev-dep.
    fn gen_f16_vec(len: usize, seed: u64, lo: f32, hi: f32) -> Vec<half::f16> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
        let mut out = Vec::with_capacity(len);
        let span = hi - lo;
        for _ in 0..len {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            // Map to [0, 1)
            let u = ((z >> 11) as f64 / (1u64 << 53) as f64) as f32;
            out.push(half::f16::from_f32(lo + u * span));
        }
        out
    }

    /// Assert `actual ≈ expected` under the standard numpy-style bound
    /// `|a - e| ≤ atol + rtol × |e|`. F16 accumulators have a non-trivial
    /// additive noise floor (`atol`) from ULP rounding during reduction, plus
    /// a multiplicative precision limit (`rtol`) from mantissa truncation.
    fn assert_close(actual: &[f32], expected: &[f32], rtol: f32, atol: f32, ctx: &str) {
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let bound = atol + rtol * e.abs();
            let diff = (a - e).abs();
            assert!(
                diff <= bound,
                "{}: element {} diff={:e} > bound={:e} (actual={:e}, expected={:e})",
                ctx,
                i,
                diff,
                bound,
                a,
                e
            );
        }
    }

    /// Test that `vec_dot_f16_native_gemv_4rows` matches a scalar F32 reference
    /// and stays within F16 precision tolerances across the K × N matrix of
    /// shapes encountered in Qwen 2.5 decode (including NR-unaligned shapes
    /// from tensor partition CPU slices).
    #[test]
    fn test_f16_native_gemv_4rows_matches_ref() {
        // (K, N) pairs cover Qwen 2.5-1.5B decode matmul shapes plus
        // partition-unaligned and small-K tail cases:
        //   K=1536 → Q/K/V/O/Gate/Up (hidden_size)
        //   K=8960 → Down (intermediate_size)
        //   K in {32,33,40,64,72,256,1024} → K-step-32 main/tail/scalar-tail
        //   N in {4,5,7,16} → NR=4 aligned + unaligned tail columns
        //   N in {1536,8960} → Qwen output dims
        //   N in {2688,2776} → typical tensor partition CPU-slice widths
        let cases: &[(usize, usize)] = &[
            (32, 4),
            (32, 5),
            (32, 7),
            (32, 16),
            (33, 4),
            (33, 7),
            (40, 16),
            (64, 16),
            (64, 2688),
            (72, 7),
            (256, 16),
            (256, 1536),
            (1024, 1536),
            (1024, 2776),
            (1536, 1536),
            (1536, 2688),
            (1536, 8960),
            (8960, 1536),
            (8960, 2691),
        ];

        for &(k, n) in cases {
            // A in RMS-norm-like range (~unit magnitude).
            let a = gen_f16_vec(k, 0xA1A1_u64 ^ (k as u64), -1.5, 1.5);
            // B in Qwen-weight-like range (~0.25 typical magnitude).
            let b_all = gen_f16_vec(n * k, 0xB2B2_u64 ^ (k as u64 * 131 + n as u64), -0.5, 0.5);

            // Reference: scalar F32 for each of N rows
            let mut expected = vec![0.0f32; n];
            for j in 0..n {
                expected[j] = ref_dot_f32(&a, &b_all[j * k..(j + 1) * k]);
            }

            // Run the NR=4 kernel + 1-row tail, mirroring f16_gemv_chunk dispatch
            let mut out = vec![0.0f32; n];
            let a_ptr = a.as_ptr() as *const u16;
            let b_ptr_base = b_all.as_ptr() as *const u16;
            let mut j = 0;
            while j + 4 <= n {
                let b_ptrs = unsafe {
                    [
                        b_ptr_base.add(j * k),
                        b_ptr_base.add((j + 1) * k),
                        b_ptr_base.add((j + 2) * k),
                        b_ptr_base.add((j + 3) * k),
                    ]
                };
                let mut res = [0.0f32; 4];
                unsafe {
                    CpuBackendNeon::vec_dot_f16_native_gemv_4rows(k, a_ptr, b_ptrs, &mut res);
                }
                out[j..j + 4].copy_from_slice(&res);
                j += 4;
            }
            while j < n {
                out[j] = unsafe {
                    CpuBackendNeon::vec_dot_f16_native_gemv_1row(k, a_ptr, b_ptr_base.add(j * k))
                };
                j += 1;
            }

            // F16 accumulation precision envelope: rtol × |expected| + atol.
            // rtol ≈ 1e-2 covers the multiplicative F16 mantissa loss
            // (~11 bits); atol scales with sqrt(K) × input_scale to absorb
            // near-zero cancellation cases where the accumulated F16 noise
            // floor dominates relative error. Empirical constants tuned for
            // Qwen-range inputs (|a| ≤ 1.5, |b| ≤ 0.5).
            let atol = 3e-3 * (k as f32).sqrt() + 1e-2;
            assert_close(&out, &expected, 1e-2, atol, &format!("k={}, n={}", k, n));
        }
    }

    /// Test that `vec_dot_f16_native_gemv_1row` matches scalar F32 reference
    /// in isolation (it is exercised indirectly by the 4rows test above, but
    /// this covers the standalone code path used by e.g. the GEMM column tail).
    #[test]
    fn test_f16_native_gemv_1row_matches_ref() {
        let k_cases = [32usize, 33, 40, 64, 72, 256, 1024, 1536, 8960];
        for &k in &k_cases {
            let a = gen_f16_vec(k, 0xC3C3_u64 ^ (k as u64), -1.5, 1.5);
            let b = gen_f16_vec(k, 0xD4D4_u64 ^ (k as u64), -0.5, 0.5);
            let expected = ref_dot_f32(&a, &b);
            let actual = unsafe {
                CpuBackendNeon::vec_dot_f16_native_gemv_1row(
                    k,
                    a.as_ptr() as *const u16,
                    b.as_ptr() as *const u16,
                )
            };
            // Same F16 accumulation precision budget as the 4rows test.
            let atol = 3e-3 * (k as f32).sqrt() + 1e-2;
            assert_close(
                std::slice::from_ref(&actual),
                std::slice::from_ref(&expected),
                1e-2,
                atol,
                &format!("k={}", k),
            );
        }
    }

    /// Test that `fused_matmul_f16` matches three separate matmul_transposed_f16
    /// calls. Specifically targets the Qwen2.5-1.5b QKV pattern (n=1536,256,256
    /// with k=1536) which exhibited garbage output on aarch64 NEON.
    #[test]
    fn test_fused_matmul_f16_matches_separate() {
        let cpu = CpuBackendNeon::new();
        // Qwen2.5-1.5b QKV decode: hidden=1536, q_dim=1536, kv_dim=256, head_dim=128
        let cases: &[(usize, &[usize])] = &[
            // (k, ns) — Qwen QKV
            (1536, &[1536, 256, 256]),
            // Llama 3.2 1B QKV
            (2048, &[2048, 512, 512]),
            // Qwen FFN gate+up (2 matmuls)
            (1536, &[8960, 8960]),
            // Single matmul (n_matmuls=1)
            (1536, &[1536]),
        ];

        for (case_idx, (k, ns)) in cases.iter().enumerate() {
            let k = *k;
            let n_matmuls = ns.len();
            // Generate input activation (F32, RMS-norm-like range).
            // Wider range than F16-quantized to mimic production residual values.
            let a_f32: Vec<f32> = (0..k)
                .map(|i| {
                    let mut state = (0xA0A0_u64 ^ (case_idx as u64) ^ (i as u64))
                        .wrapping_mul(0x9E3779B97F4A7C15);
                    state = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                    state = (state ^ (state >> 27)).wrapping_mul(0x94D049BB133111EB);
                    state ^= state >> 31;
                    let u = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
                    -3.0 + u * 6.0
                })
                .collect();

            // Generate weights (F16, small range).
            let mut weights: Vec<Vec<half::f16>> = Vec::new();
            for (i, &n) in ns.iter().enumerate() {
                weights.push(gen_f16_vec(
                    n * k,
                    0xB0B0_u64 ^ (case_idx as u64) ^ (i as u64 * 17),
                    -0.5,
                    0.5,
                ));
            }

            // Reference: scalar F32 dot per output row.
            let mut expected: Vec<Vec<f32>> = Vec::new();
            for (i, &n) in ns.iter().enumerate() {
                let mut out = vec![0.0f32; n];
                for j in 0..n {
                    let row = &weights[i][j * k..(j + 1) * k];
                    out[j] = a_f32
                        .iter()
                        .zip(row.iter())
                        .map(|(&a, &b)| a * b.to_f32())
                        .sum();
                }
                expected.push(out);
            }

            // Test: fused_matmul_f16
            let mut fused_outs: Vec<Vec<f32>> = ns.iter().map(|&n| vec![0.0f32; n]).collect();
            unsafe {
                let matmuls: Vec<(*const u16, *mut f32, usize)> = ns
                    .iter()
                    .enumerate()
                    .map(|(i, &n)| {
                        (
                            weights[i].as_ptr() as *const u16,
                            fused_outs[i].as_mut_ptr(),
                            n,
                        )
                    })
                    .collect();
                fused_matmul_f16(a_f32.as_ptr(), k, &matmuls);
            }

            // Test: separate matmul_transposed_f16
            let mut sep_outs: Vec<Vec<f32>> = ns.iter().map(|&n| vec![0.0f32; n]).collect();
            for (i, &n) in ns.iter().enumerate() {
                let a_tensor = Tensor::new(
                    Shape::new(vec![1, k]),
                    Arc::new(SharedBuffer::from_vec(
                        a_f32
                            .iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F32,
                    )),
                    Arc::new(CpuBackendNeon::new()),
                );
                let b_tensor = Tensor::new(
                    Shape::new(vec![n, k]),
                    Arc::new(SharedBuffer::from_vec(
                        weights[i]
                            .iter()
                            .flat_map(|h| h.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F16,
                    )),
                    Arc::new(CpuBackendNeon::new()),
                );
                let mut out_tensor = Tensor::new(
                    Shape::new(vec![1, n]),
                    Arc::new(SharedBuffer::new(n * 4, DType::F32)),
                    Arc::new(CpuBackendNeon::new()),
                );
                cpu.matmul_transposed(&a_tensor, &b_tensor, &mut out_tensor)
                    .unwrap();
                sep_outs[i].copy_from_slice(out_tensor.as_slice::<f32>());
            }

            // Compare fused vs separate (should be near-identical).
            for (i, &n) in ns.iter().enumerate() {
                let atol = 3e-3 * (k as f32).sqrt() + 1e-2;
                assert_close(
                    &fused_outs[i],
                    &sep_outs[i],
                    1e-2,
                    atol,
                    &format!(
                        "case={} mat={} fused-vs-separate (k={}, n={}, n_matmuls={})",
                        case_idx, i, k, n, n_matmuls
                    ),
                );
            }
            // Also compare fused vs scalar reference (catches both being wrong).
            for (i, &n) in ns.iter().enumerate() {
                let atol = 3e-3 * (k as f32).sqrt() + 1e-2;
                assert_close(
                    &fused_outs[i],
                    &expected[i],
                    2e-2,
                    atol * 2.0,
                    &format!(
                        "case={} mat={} fused-vs-ref (k={}, n={}, n_matmuls={})",
                        case_idx, i, k, n, n_matmuls
                    ),
                );
            }
        }
    }

    #[test]
    fn test_i8mm_dot_q4_0_q8_0() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            eprintln!("[SKIPPED] i8mm not supported on this device");
            return;
        }

        let backend = CpuBackendNeon;

        // Test various dimensions
        for nb in [1, 2, 4, 8, 64, 128] {
            let k = nb * QK4_0; // 32, 64, 128, 256, 2048, 4096

            // Create deterministic test data
            let mut rng_state = 42u64;
            let mut pseudo_rand = || -> i8 {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) % 15) as i8 - 7
            };

            let mut blocks_row0 = Vec::with_capacity(nb);
            let mut blocks_row1 = Vec::with_capacity(nb);
            for blk in 0..nb {
                let d0 = half::f16::from_f32(0.1 + 0.01 * blk as f32);
                let d1 = half::f16::from_f32(0.15 + 0.01 * blk as f32);
                let mut qs0 = [0u8; QK4_0 / 2];
                let mut qs1 = [0u8; QK4_0 / 2];
                for z in 0..QK4_0 / 2 {
                    let v0_lo = (pseudo_rand() + 8) as u8;
                    let v0_hi = (pseudo_rand() + 8) as u8;
                    qs0[z] = (v0_lo & 0x0F) | ((v0_hi & 0x0F) << 4);
                    let v1_lo = (pseudo_rand() + 8) as u8;
                    let v1_hi = (pseudo_rand() + 8) as u8;
                    qs1[z] = (v1_lo & 0x0F) | ((v1_hi & 0x0F) << 4);
                }
                blocks_row0.push(BlockQ4_0 { d: d0, qs: qs0 });
                blocks_row1.push(BlockQ4_0 { d: d1, qs: qs1 });
            }

            // Create activation (Q8_0)
            let mut act_blocks = Vec::with_capacity(nb);
            for blk in 0..nb {
                let d = half::f16::from_f32(0.2 + 0.005 * blk as f32);
                let mut qs = [0i8; QK8_0];
                for z in 0..QK8_0 {
                    qs[z] = pseudo_rand();
                }
                act_blocks.push(BlockQ8_0 { d, qs });
            }

            // Reference: existing sdot function
            let mut ref0 = 0.0f32;
            let mut ref1 = 0.0f32;
            unsafe {
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    backend.vec_dot_q4_0_q8_0_sdot(
                        k,
                        &mut ref0,
                        blocks_row0.as_ptr(),
                        act_blocks.as_ptr(),
                    );
                    backend.vec_dot_q4_0_q8_0_sdot(
                        k,
                        &mut ref1,
                        blocks_row1.as_ptr(),
                        act_blocks.as_ptr(),
                    );
                } else {
                    backend.vec_dot_q4_0_q8_0(
                        k,
                        &mut ref0,
                        blocks_row0.as_ptr(),
                        act_blocks.as_ptr(),
                    );
                    backend.vec_dot_q4_0_q8_0(
                        k,
                        &mut ref1,
                        blocks_row1.as_ptr(),
                        act_blocks.as_ptr(),
                    );
                }
            }

            // i8mm
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            unsafe {
                backend.vec_dot_q4_0_q8_0_i8mm(
                    k,
                    &mut s0,
                    &mut s1,
                    blocks_row0.as_ptr(),
                    blocks_row1.as_ptr(),
                    act_blocks.as_ptr(),
                );
            }

            let tol = 1e-4 * (k as f32).sqrt();
            assert!(
                (s0 - ref0).abs() < tol,
                "k={k} row0: i8mm={s0} ref={ref0} diff={}",
                (s0 - ref0).abs()
            );
            assert!(
                (s1 - ref1).abs() < tol,
                "k={k} row1: i8mm={s1} ref={ref1} diff={}",
                (s1 - ref1).abs()
            );
            eprintln!(
                "[PASS] k={k}: row0 diff={:.6}, row1 diff={:.6}",
                (s0 - ref0).abs(),
                (s1 - ref1).abs()
            );
        }
    }
}
