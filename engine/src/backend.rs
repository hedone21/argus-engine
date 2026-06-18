// ============================================================================
// Backend implementations (sub-modules). Moved from `backend/mod.rs` to the
// top-level `engine/src/backend.rs` per Step 4-A (Rust 2018+ pattern: trait
// definition lives next to its implementations, no `mod.rs` needed).
// ============================================================================

pub mod cpu;

// Optional GPU backends (mutually exclusive)
#[cfg(all(feature = "cuda", feature = "cuda-embedded"))]
compile_error!(
    "Features `cuda` and `cuda-embedded` are mutually exclusive — pick one (cuda = PC/discrete GPU, cuda-embedded = Jetson/UMA)"
);

#[cfg(feature = "cuda")]
pub mod cuda_pc;
#[cfg(feature = "cuda")]
pub use cuda_pc as cuda;

#[cfg(feature = "cuda-embedded")]
pub mod cuda_embedded;
#[cfg(feature = "cuda-embedded")]
pub use cuda_embedded as cuda;

#[cfg(feature = "opencl")]
pub mod opencl;

// ============================================================================
// Backend trait + GpuEvent (originally `core/backend.rs`).
// ============================================================================

use crate::buffer::DType;
use crate::tensor::Tensor;
use anyhow::Result;

/// Opaque async GPU event handle.
///
/// Originally introduced by tensor-partition's `LLMRS_PARTITION_ASYNC_READ`
/// path to overlap the residual DMA read with a subsequent GPU enqueue chain
/// (`enqueue_read_buffer_async` / `wait_event`). Extended for the async layer
/// swap dispatcher prototype (LISWAP-2 plan: `chasing-hopper`) to also carry
/// the H2D write event produced by `enqueue_write_async`. The CUDA variant
/// is added here so `wait_event_blocking` can wait on a CUDA-recorded event
/// without stalling the compute stream.
///
/// Both inner handles are `Option`s. `None` represents a dummy/no-op event
/// returned when the backend's async path falls back to a synchronous copy.
/// The struct itself is `Send + Sync` because both `ocl::core::Event` and
/// `cudarc::driver::CudaEvent` are `Send + Sync`, which lets the dispatcher
/// pass events to a worker thread.
#[derive(Default)]
pub struct GpuEvent {
    #[cfg(feature = "opencl")]
    pub(crate) inner_cl: Option<ocl::core::Event>,
    #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
    pub(crate) inner_cu: Option<cudarc::driver::CudaEvent>,
}

impl GpuEvent {
    /// Dummy event — `wait_event` / `wait_event_blocking` on this is a no-op.
    pub fn dummy() -> Self {
        Self::default()
    }

    /// LISWAP-6 Phase 5: alias path 검출용. cl/cu inner 모두 None 이면
    /// 진짜 wait할 GPU 작업이 없는 sentinel. `process_commit` 가 이를 보면
    /// `wait_event_blocking` 호출을 skip 해서 fall-through `synchronize()`
    /// (forward GPU op block) 을 회피한다.
    pub fn is_dummy(&self) -> bool {
        #[cfg(feature = "opencl")]
        let cl_none = self.inner_cl.is_none();
        #[cfg(not(feature = "opencl"))]
        let cl_none = true;
        #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
        let cu_none = self.inner_cu.is_none();
        #[cfg(not(any(feature = "cuda", feature = "cuda-embedded")))]
        let cu_none = true;
        cl_none && cu_none
    }
}

// ── Cold-path extension namespace (ggml `get_proc_address` 차용) ──────────
//
// `Backend::get_extension(name)` 의 `name` 컨벤션. 호출지가 자유 문자열을
// 박는 silent-None 회피용으로 본 const 만 사용한다.
//
// 본 entry 는 *cold path* 전용. forward / decode loop 안에서 호출 금지.
// (RAII guard / HashMap 비용 발생; 자세한 정책은 `Backend::get_extension`
// rustdoc 참조).
//
// Architect 라운드: `arch/sprint_backend_extension_round.md` R-EXT-1 (α).

/// Cold-path extension key — `OpenCLBackend` 핸들 (queue 접근용).
///
/// 반환 타입은 `&OpenCLBackend` 로 downcast 해서 사용한다.
pub const EXT_OPENCL_QUEUE: &str = "opencl_queue";

/// Cold-path extension key — OpenCL secondary slot (precision swap path).
///
/// 반환 타입은 `&OpenCLBackend` 로 downcast 해서 사용한다.
/// 현재는 `EXT_OPENCL_QUEUE` 와 동일한 핸들을 반환하지만, 향후
/// secondary store 추상화가 구체화되면 별도 타입을 반환할 수 있다.
pub const EXT_OPENCL_SECONDARY: &str = "opencl_secondary";

/// Cold-path extension key — backend-agnostic `Arc<RpcmemAllocator>` 핸들.
///
/// Sprint 2a Phase 2 (ENG-RPCMEM-024 / INV-RPCMEM-002) 에서 도입. OpenCL backend
/// 가 `--opencl-rpcmem` 활성 시 본 extension 을 통해 `Arc<RpcmemAllocator>` 를
/// expose 하며, `RpcmemSecondaryStore` loader 가 본 키로 lookup 한다. 반환
/// 타입은 `&Arc<crate::memory::rpcmem::allocator::RpcmemAllocator>` 로 downcast
/// 후 clone 하여 사용한다.
pub const EXT_RPCMEM_ALLOCATOR: &str = "rpcmem_allocator";

// §13.8-L S-L-3/S-L-2 — KIVI native attention / GPU score accumulator 추상화.
//
// Phase α-W-4 에서 두 trait 의 정의 본체를 `capability/` 서브모듈로 물리 이동했다.
// 기존 import path(`crate::backend::QuantAttnBackend` 등)를 보존하기 위해
// 여기서 re-export 한다 (byte-identical). `Backend` god-trait 의
// `gpu_score_acc`/`gpu_score_acc_mut` default 메서드는 본 re-export 된
// trait 를 그대로 참조한다.
pub use crate::capability::gpu_score::GpuScoreAccess;
pub use crate::capability::quant_attn::{
    QuantAttnArgs, QuantAttnBackend, QuantAttnGatherArgs, QuantDequantFlushArgs,
    QuantScatterResidualArgs,
};

pub trait Backend: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn name(&self) -> &str;
    fn device(&self) -> &str;

    // Basic Math
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        rows: usize,
        cols: usize,
        out: &mut Tensor,
    ) -> Result<()>;

    // In-place operations
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;
    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()>;

    /// Broadcast-add a 1D bias to each row of a tensor.
    /// x: [..., dim], bias: [dim]. Adds bias to every row of x.
    fn add_row_bias(&self, x: &mut Tensor, bias: &Tensor) -> Result<()> {
        let x_data = x.as_mut_slice::<f32>();
        let b_data = bias.as_slice::<f32>();
        let dim = b_data.len();
        for row in x_data.chunks_mut(dim) {
            for (v, &b) in row.iter_mut().zip(b_data.iter()) {
                *v += b;
            }
        }
        Ok(())
    }

    // Activation & Norm
    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;

    /// Fused FFN gate + up matmul followed by SiLU gating:
    ///     out[i] = silu(gate·x)[i] * (up·x)[i]
    ///
    /// Both matmuls share the same activation `x`, so a GPU backend
    /// can bundle them into one kernel and reuse the quantised
    /// activation cache. The `gate_scratch`/`up_scratch` tensors are
    /// temporaries used by the default (non-fused) fallback; GPU
    /// overrides may ignore them and write `out` directly.
    ///
    /// Default: 3 op fallback (gate matmul → up matmul → silu_mul).
    fn matmul_ffn_gate_up_silu(
        &self,
        x: &Tensor,
        w_gate: &Tensor,
        w_up: &Tensor,
        out: &mut Tensor,
        up_scratch: &mut Tensor,
    ) -> Result<()> {
        self.matmul_transposed(x, w_gate, out)?;
        self.matmul_transposed(x, w_up, up_scratch)?;
        self.silu_mul(out, up_scratch)
    }

    /// GELU tanh approximation fused with elementwise multiply.
    /// gate[i] = gelu_tanh(gate[i]) * up[i]
    /// Used by Gemma 3 FFN (hidden_activation = "gelu_pytorch_tanh").
    /// Default: scalar CPU fallback — backends may override for performance.
    fn gelu_tanh_mul(&self, gate: &mut Tensor, up: &Tensor) -> Result<()> {
        let gate_data = gate.as_mut_slice::<f32>();
        let up_data = up.as_slice::<f32>();
        let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
        for (g, &u) in gate_data.iter_mut().zip(up_data.iter()) {
            let x = *g;
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            *g = 0.5 * x * (1.0 + inner.tanh()) * u;
        }
        Ok(())
    }
    /// In-place RMS norm: x = x * w / rms(x).
    /// If `add_unit` is true (Gemma 3 style), applies `x * (1 + w) / rms(x)` instead.
    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()>;
    /// Out-of-place RMS norm: out = norm(x) * w. x is preserved.
    /// Default: copy x → out, then in-place rms_norm(out).
    fn rms_norm_oop(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        self.copy_into(x, out)?;
        self.rms_norm(out, w, eps, add_unit)
    }
    /// Fused add + out-of-place RMS norm: x += residual; out = norm(x) * w.
    /// Eliminates a separate add_assign dispatch.
    fn add_rms_norm_oop(
        &self,
        x: &mut Tensor,
        residual: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        self.add_assign(x, residual)?;
        self.rms_norm_oop(x, out, w, eps, add_unit)
    }

    /// Fused 3-input merge + out-of-place RMS norm for tensor-partition layer boundary.
    ///
    /// Replaces the decode-path per-layer merge sequence
    ///     (copy_slice gpu → buf, copy_slice cpu → staging, add_assign, residual += buf)
    /// plus the next layer's pre-attention RMSNorm with a single kernel dispatch,
    /// eliminating three inter-kernel barriers on the OpenCL in-order queue.
    ///
    /// Semantics:
    ///   residual_out[i] = prior_residual[i] + gpu_partial[i] + cpu_staging[i]
    ///   out[i]          = (residual_out[i] / rms(residual_out)) * w_eff[i]
    /// where `w_eff = (1 + norm_weight)` when `add_unit` is true (Gemma3) else `norm_weight`.
    ///
    /// Buffer aliasing: `residual_out` MAY alias `prior_residual` for an in-place residual
    /// update. `out` MUST NOT alias any input. The three partial inputs MUST be distinct
    /// buffers.
    ///
    /// Default implementation: composes the existing 3-step path for correctness and for
    /// backends that do not override (CPU, CUDA, and OpenCL fallback during bring-up).
    /// The OpenCL backend overrides with the fused kernel.
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
        // Default path: residual_out = prior + gpu + cpu; out = norm(residual_out) * w.
        // Copy prior into residual_out, then add gpu and cpu contributions.
        self.copy_into(prior_residual, residual_out)?;
        self.add_assign(residual_out, gpu_partial)?;
        self.add_assign(residual_out, cpu_staging)?;
        self.rms_norm_oop(residual_out, out, norm_weight, eps, add_unit)
    }
    fn softmax(&self, x: &mut Tensor) -> Result<()>;

    // Rotate
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()>;

    // Single-query attention for generation (GQA-aware)
    // Q: [num_heads_q, head_dim]
    // K/V cache: SeqMajor [cache_seq_len, num_heads_kv, head_dim] or HeadMajor [num_heads_kv, capacity, head_dim]
    // Output: [num_heads_q, head_dim]
    // scores_out: Optional [num_heads_q * stride] buffer for post-softmax attention scores.
    //             stride = scores_out.len() / num_heads_q. Each head's [0..cache_seq_len] filled.
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
        // Default CPU implementation — supports F32 and F16 KV cache.
        use crate::buffer::DType;
        let q_data = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const f32, q.size() / 4) };
        let out_data =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, out.size() / 4) };

        // For F16 KV, dequantize to temporary F32 buffers.
        let kv_dtype = k_cache.dtype();
        let (k_f32_buf, v_f32_buf);
        let (k_data, v_data): (&[f32], &[f32]) = match kv_dtype {
            DType::F32 => (
                unsafe {
                    std::slice::from_raw_parts(k_cache.as_ptr() as *const f32, k_cache.size() / 4)
                },
                unsafe {
                    std::slice::from_raw_parts(v_cache.as_ptr() as *const f32, v_cache.size() / 4)
                },
            ),
            DType::F16 => {
                use half::f16;
                let k_f16 = unsafe {
                    std::slice::from_raw_parts(k_cache.as_ptr() as *const f16, k_cache.size() / 2)
                };
                let v_f16 = unsafe {
                    std::slice::from_raw_parts(v_cache.as_ptr() as *const f16, v_cache.size() / 2)
                };
                k_f32_buf = k_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
                v_f32_buf = v_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
                (k_f32_buf.as_slice(), v_f32_buf.as_slice())
            }
            _ => anyhow::bail!(
                "attention_gen default impl: unsupported KV dtype {:?}",
                kv_dtype
            ),
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = num_heads_q / num_heads_kv;

        // Detect layout from shape: HeadMajor if shape[1] == num_heads_kv
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };
        let scores_stride = scores_out
            .as_ref()
            .map(|s| s.len() / num_heads_q)
            .unwrap_or(0);
        let scores_ptr = scores_out.map(|s| s.as_mut_ptr());

        for h in 0..num_heads_q {
            let kv_h = h / gqa_ratio;
            let q_off = h * head_dim;
            let q_vec = &q_data[q_off..q_off + head_dim];

            // Compute scores
            let mut scores = vec![0.0f32; cache_seq_len];
            #[allow(clippy::needless_range_loop)]
            for t in 0..cache_seq_len {
                let k_off = if is_head_major {
                    (kv_h * capacity + t) * head_dim
                } else {
                    (t * num_heads_kv + kv_h) * head_dim
                };
                let k_vec = &k_data[k_off..k_off + head_dim];
                let score: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                scores[t] = score * scale;
            }

            // Softmax
            let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max_val).exp();
                sum_exp += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum_exp;
            }

            // Copy scores to output buffer if requested
            if let Some(ptr) = scores_ptr {
                unsafe {
                    let dst =
                        std::slice::from_raw_parts_mut(ptr.add(h * scores_stride), cache_seq_len);
                    dst.copy_from_slice(&scores[..cache_seq_len]);
                }
            }

            // Weighted sum of V
            let out_off = h * head_dim;
            for d in 0..head_dim {
                out_data[out_off + d] = 0.0;
            }
            #[allow(clippy::needless_range_loop)]
            for t in 0..cache_seq_len {
                let weight = scores[t];
                let v_off = if is_head_major {
                    (kv_h * capacity + t) * head_dim
                } else {
                    (t * num_heads_kv + kv_h) * head_dim
                };
                let v_vec = &v_data[v_off..v_off + head_dim];
                for d in 0..head_dim {
                    out_data[out_off + d] += weight * v_vec[d];
                }
            }
        }
        Ok(())
    }

    // Memory Ops
    fn copy_from(&self, t: &Tensor) -> Result<Tensor>;

    /// Copy a weight tensor from CPU to this backend.
    ///
    /// Distinguishes static, read-only weights (uploaded once at model load
    /// time) from runtime activations. Backends that can benefit from
    /// device-local storage for weights (e.g. `cuda_embedded` with
    /// `--cuda-weights-device` to bypass Jetson UMA cache coherency issues)
    /// override this method. The default delegates to `copy_from`.
    fn copy_weight_from(&self, t: &Tensor) -> Result<Tensor> {
        self.copy_from(t)
    }

    /// Fused F32→F16 cast + HeadMajor scatter for KV cache update.
    /// Replaces 2× cast + 16× copy_slice with a single GPU kernel.
    /// Default: falls back to separate cast + copy.
    #[allow(unused_variables, clippy::too_many_arguments)]
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
        // CPU fallback: F32→F16 cast + HeadMajor scatter per head.
        // k_src: [1, seq_len, kv_heads * head_dim] F32
        // k_dst: [1, kv_heads, capacity, head_dim] F16 HeadMajor
        use half::f16;
        let kv_heads = k_src.shape().dims().last().copied().unwrap_or(0) / head_dim;
        let src_f32 =
            unsafe { std::slice::from_raw_parts(k_src.as_ptr() as *const f32, k_src.size() / 4) };
        let dst_f16 = unsafe {
            std::slice::from_raw_parts_mut(k_dst.as_mut_ptr() as *mut f16, k_dst.size() / 2)
        };
        for h in 0..kv_heads {
            let src_off = h * head_dim;
            let dst_off = h * capacity * head_dim + write_pos * head_dim;
            for d in 0..head_dim {
                dst_f16[dst_off + d] = f16::from_f32(src_f32[src_off + d]);
            }
        }
        let v_src_f32 =
            unsafe { std::slice::from_raw_parts(v_src.as_ptr() as *const f32, v_src.size() / 4) };
        let v_dst_f16 = unsafe {
            std::slice::from_raw_parts_mut(v_dst.as_mut_ptr() as *mut f16, v_dst.size() / 2)
        };
        for h in 0..kv_heads {
            let src_off = h * head_dim;
            let dst_off = h * capacity * head_dim + write_pos * head_dim;
            for d in 0..head_dim {
                v_dst_f16[dst_off + d] = f16::from_f32(v_src_f32[src_off + d]);
            }
        }
        Ok(())
    }

    /// Whether this backend has a native batch F32→F16 KV scatter implementation.
    ///
    /// Returns `false` by default. Backends implementing a real kernel for
    /// `kv_scatter_f32_to_f16_batch` must override this to return `true`.
    ///
    /// The default implementation of `kv_scatter_f32_to_f16_batch` uses host
    /// pointer slices via `as_ptr()`/`as_mut_ptr()`, which segfaults on
    /// device-only buffers (e.g. OpenCL without zero-copy mapping). Callers
    /// must check this flag before dispatching the batch path and fall back to
    /// the per-position `cast + update` path otherwise.
    fn supports_kv_scatter_batch(&self) -> bool {
        false
    }

    /// Batch F32->F16 KV scatter for prefill: writes seq_len positions in one kernel launch.
    /// k_src/v_src: contiguous [seq_len, kv_heads * head_dim] F32
    /// k_dst/v_dst: [kv_heads, capacity, head_dim] F16 HeadMajor
    /// Default: host-pointer fallback — ONLY safe when callers guard on `supports_kv_scatter_batch()`.
    #[allow(clippy::too_many_arguments)]
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
        // Default: iterate per position using single-position scatter
        use half::f16;
        let src_f32 =
            unsafe { std::slice::from_raw_parts(k_src.as_ptr() as *const f32, k_src.size() / 4) };
        let dst_f16 = unsafe {
            std::slice::from_raw_parts_mut(k_dst.as_mut_ptr() as *mut f16, k_dst.size() / 2)
        };
        let v_src_f32 =
            unsafe { std::slice::from_raw_parts(v_src.as_ptr() as *const f32, v_src.size() / 4) };
        let v_dst_f16 = unsafe {
            std::slice::from_raw_parts_mut(v_dst.as_mut_ptr() as *mut f16, v_dst.size() / 2)
        };
        for s in 0..seq_len {
            for h in 0..n_kv_heads {
                let src_off = (s * n_kv_heads + h) * head_dim;
                let dst_off = h * capacity * head_dim + (write_pos_start + s) * head_dim;
                for d in 0..head_dim {
                    dst_f16[dst_off + d] = f16::from_f32(src_f32[src_off + d]);
                    v_dst_f16[dst_off + d] = f16::from_f32(v_src_f32[src_off + d]);
                }
            }
        }
        Ok(())
    }

    /// Whether the backend has a dedicated F32->F32 batch KV scatter kernel.
    /// Default: false. CUDA-PC overrides to true.
    fn supports_kv_scatter_f32_batch(&self) -> bool {
        false
    }

    /// Batch F32->F32 KV scatter (HeadMajor). Same launch shape and semantics as
    /// `kv_scatter_f32_to_f16_batch` but no cast — for `kv-type=f32` caches on GPU.
    /// Avoids the per-(s,h) `cuMemcpyDtoD` storm produced by the generic
    /// `KVCache::update` path on discrete CUDA.
    /// Default: host-pointer fallback — ONLY safe when callers guard on
    /// `supports_kv_scatter_f32_batch()`.
    #[allow(clippy::too_many_arguments)]
    fn kv_scatter_f32_to_f32_batch(
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
        let src_k =
            unsafe { std::slice::from_raw_parts(k_src.as_ptr() as *const f32, k_src.size() / 4) };
        let src_v =
            unsafe { std::slice::from_raw_parts(v_src.as_ptr() as *const f32, v_src.size() / 4) };
        let dst_k = unsafe {
            std::slice::from_raw_parts_mut(k_dst.as_mut_ptr() as *mut f32, k_dst.size() / 4)
        };
        let dst_v = unsafe {
            std::slice::from_raw_parts_mut(v_dst.as_mut_ptr() as *mut f32, v_dst.size() / 4)
        };
        for s in 0..seq_len {
            for h in 0..n_kv_heads {
                let src_off = (s * n_kv_heads + h) * head_dim;
                let dst_off = h * capacity * head_dim + (write_pos_start + s) * head_dim;
                dst_k[dst_off..dst_off + head_dim]
                    .copy_from_slice(&src_k[src_off..src_off + head_dim]);
                dst_v[dst_off..dst_off + head_dim]
                    .copy_from_slice(&src_v[src_off..src_off + head_dim]);
            }
        }
        Ok(())
    }

    /// Copy data from src into dst buffer (same shape/size required).
    /// On GPU: just enqueue_copy_buffer, no new backend/kernel allocation.
    /// On CPU: memcpy.
    fn copy_into(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        let size = src.size();
        assert_eq!(size, dst.size(), "copy_into: size mismatch");
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        if !src_ptr.is_null() && !dst_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
            }
        }
        Ok(())
    }
    fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
        let src_ptr = t.buffer().as_ptr();
        if src_ptr.is_null() {
            anyhow::bail!("Cannot read null buffer (not mapped)");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    /// Non-blocking variant of `read_buffer`. Enqueues a DMA read and returns
    /// an opaque event handle that can be awaited via `wait_event`. Backends
    /// that support true async reads (OpenCL) override this. The default
    /// falls back to a synchronous blocking read and returns a dummy event.
    ///
    /// # Safety
    ///
    /// The caller must ensure `dst` remains valid until `wait_event` has
    /// returned for the returned event — OpenCL writes into `dst` on the
    /// device's timeline, not the host's.
    fn enqueue_read_buffer_async(&self, t: &Tensor, dst: &mut [u8]) -> Result<GpuEvent> {
        self.read_buffer(t, dst)?;
        Ok(GpuEvent::dummy())
    }

    /// Block until the event returned by `enqueue_read_buffer_async` has
    /// completed. Default: no-op (for backends that return a dummy event
    /// because the enqueue was already blocking).
    fn wait_event(&self, _evt: &GpuEvent) -> Result<()> {
        Ok(())
    }

    /// Submit an asynchronous host→device weight upload on the backend's
    /// transfer queue / stream, returning a fresh tensor backed by a new
    /// device buffer plus an event that completes once the H2D copy is
    /// GPU-visible.
    ///
    /// Used by the async layer swap dispatcher (LISWAP-2 prototype, plan
    /// `chasing-hopper`): the main thread submits + immediately returns,
    /// while a background worker calls `wait_event_blocking(&evt)` before
    /// installing the new weights into the layer slot. Hardware-level
    /// concurrency between this transfer and the compute queue/stream is
    /// the whole point of the prototype.
    ///
    /// Default implementation: synchronous fallback via `copy_weight_from`
    /// + dummy event. Backends that have a separate transfer queue/stream
    ///   (OpenCL: `transfer_queue`; CUDA: `transfer_stream`) override this.
    fn enqueue_write_async(&self, src: &Tensor) -> Result<(Tensor, GpuEvent)> {
        let dst = self.copy_weight_from(src)?;
        Ok((dst, GpuEvent::default()))
    }

    /// LISWAP-8 Phase B: write host bytes into an *existing* tensor's
    /// device buffer asynchronously on the transfer queue/stream.
    ///
    /// Unlike `enqueue_write_async`, this does NOT allocate a fresh device
    /// buffer — it overwrites the bytes of `dst`'s existing buffer in place.
    /// Enables a pre-allocated `LayerObjectPool`-backed swap path where
    /// every swap reuses a pool entry's GPU buffers (no `cuMemAlloc` /
    /// `cuMemFree` in the dispatch hot path → no CUDA driver context lock
    /// contention against forward `cuLaunchKernel` calls).
    ///
    /// Default: returns an error so backends opt in explicitly. CUDA
    /// embedded backend overrides this with a `CudaDeviceBuffer` downcast
    /// + `copy_from_host_async` on the existing device pointer.
    fn enqueue_write_into_async(
        &self,
        _dst: &Tensor,
        _src: *const u8,
        _len: usize,
    ) -> Result<GpuEvent> {
        anyhow::bail!("enqueue_write_into_async not implemented for this backend")
    }

    /// Block until the event returned by `enqueue_write_async` has
    /// completed, *without* blocking the compute queue/stream.
    ///
    /// Default: full barrier via `synchronize()` (correct, but loses any
    /// overlap with concurrent compute). Backends that returned a real
    /// event from `enqueue_write_async` override this with a per-event
    /// wait (`clWaitForEvents` / `cuEventSynchronize`).
    fn wait_event_blocking(&self, _evt: &GpuEvent) -> Result<()> {
        self.synchronize()
    }

    /// Whether this backend has a separate transfer queue/stream usable
    /// by `enqueue_write_async` for hardware-concurrent execution.
    ///
    /// `false` (default) means `enqueue_write_async` is the synchronous
    /// fallback above and there is no benefit over the regular
    /// `copy_weight_from` path. Callers (the async swap dispatcher) use
    /// this to decide whether to take the dispatcher path at all.
    fn supports_async_transfer(&self) -> bool {
        false
    }

    /// Write host bytes into a backend buffer (CPU→GPU upload).
    /// Default: memcpy from src slice to tensor's mapped pointer.
    /// GPU backends should override with enqueue_write_buffer.
    fn write_buffer(&self, t: &mut Tensor, src: &[u8]) -> Result<()> {
        let dst_ptr = t.as_mut_ptr();
        if dst_ptr.is_null() {
            anyhow::bail!("Cannot write to null buffer (not mapped)");
        }
        assert_eq!(
            src.len(),
            t.size(),
            "write_buffer: size mismatch ({} vs {})",
            src.len(),
            t.size()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
        }
        Ok(())
    }

    /// Write host bytes into a sub-range of a backend buffer starting at
    /// `dst_offset`. Used by callers that keep a max-capacity buffer but only
    /// need to upload the currently-valid prefix (e.g. `OffloadKVCache` staging
    /// a GPU KV view of length `current_pos` into a `max_seq_len`-sized buffer).
    ///
    /// Default: memcpy into the tensor's mapped pointer at `dst_offset`.
    /// GPU backends should override with a bounded `enqueue_write_buffer` to
    /// avoid requiring a full-capacity write.
    fn write_buffer_range(&self, t: &mut Tensor, src: &[u8], dst_offset: usize) -> Result<()> {
        let dst_ptr = t.as_mut_ptr();
        if dst_ptr.is_null() {
            anyhow::bail!("Cannot write to null buffer (not mapped)");
        }
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
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr.add(dst_offset), src.len());
        }
        Ok(())
    }

    // Type casting (e.g. F32 → F16)
    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()>;

    // Synchronization (for benchmarking)
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    /// Flush the command queue — submit enqueued commands to the device without waiting.
    /// On GPU backends this calls clFlush to prevent pipeline bubbles.
    fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// GPU 백엔드 여부. CPU: false, OpenCL/CUDA: true.
    fn is_gpu(&self) -> bool {
        false
    }

    /// Discrete GPU (non-UMA) 여부. UMA(Adreno/Jetson): false, discrete(NVIDIA desktop): true.
    fn is_discrete_gpu(&self) -> bool {
        false
    }

    /// Maximum single buffer allocation size in bytes.
    /// GPU backends return CL_DEVICE_MAX_MEM_ALLOC_SIZE; CPU returns usize::MAX.
    fn max_single_alloc(&self) -> usize {
        usize::MAX
    }

    /// Invalidate backend-internal caches keyed by weight tensor `cl_mem`
    /// addresses after a runtime weight swap (ENG-ALG-221 / INV-130).
    ///
    /// The OpenCL backend overrides this to clear
    /// `noshuffle_soa_registry` (HashMap keyed by old `cl_mem` addresses).
    /// `SwapExecutor::execute_on_slots` is contracted to call this exactly
    /// once per non-empty swap batch — before bumping `ratio_generation` so
    /// that the subsequent `FullKernelPlan` rebuild observes a clean slate
    /// and re-registers SOA descriptors against the new cl_mem keys.
    ///
    /// Device-only: the Adreno noshuffle Q4_0 GEMV path is the sole consumer.
    /// CPU / CUDA defaults are no-op. Callers may invoke unconditionally.
    fn invalidate_noshuffle_soa_registry(&self) {}

    /// Adreno SOA 재변환 — Phase 3.7a (ENG-ALG-222 / INV-131).
    ///
    /// Q4_0 weight swap 직후, swap된 layer의 weight tensor에 대해 noshuffle
    /// SOA descriptor를 backend의 noshuffle SOA registry에 등록한다.
    /// `invalidate_noshuffle_soa_registry()`로 stale entry를 비운 직후 호출되어야
    /// 하며, 다음 forward의 plan rebuild 진입 시점까지 모든 swap된 Q4_0 weight의
    /// 새 `cl_mem` 주소가 등록되어 있어야 한다 (INV-131).
    ///
    /// **호출 의무**: Q4_0 dtype + GPU buffer일 때만 의미가 있다. 다른 dtype 혹은
    /// CPU/CUDA backend, 그리고 Q4_0 noshuffle 커널이 컴파일되지 않은 OpenCL
    /// 환경에서는 NoOp이다. 호출자는 dtype 분기 없이 단일 호출을 사용한다.
    ///
    /// **에러 처리**: SOA 변환 자체가 실패한 경우 `Err`를 반환하며, 호출자는
    /// 해당 swap layer의 forward를 fallback (AOS GEMV)으로 진행할 수 있다.
    /// AOS fallback은 정확도가 부족하므로 batch swap에서 첫 실패 시 즉시 중단을
    /// 권장한다 (FullKernelPlan rebuild 시 lookup miss로 silent garbage).
    #[allow(unused_variables)]
    fn ensure_noshuffle_soa_registered(&self, tensor: &Tensor) -> Result<()> {
        Ok(())
    }

    /// AUF SOA bypass — Phase 3.7b (ENG-DAT-096 / Phase 4 LATENCY-AUF).
    ///
    /// Allocate a Q4_0 weight tensor whose backing buffer is the
    /// noshuffle SOA layout (`q_buf` + `d_buf` + optional `q_img`) sourced
    /// directly from the AUF `WEIGHTS_ADRENO_SOA` payload. The returned
    /// tensor reports the *logical* AOS shape and `Q4_0` dtype but its
    /// physical buffer is a `NoshuffleWeightBuffer` containing the SOA
    /// `cl_mem` handles. The runtime `convert_q4_0_to_noshuffle` pipeline is
    /// bypassed entirely.
    ///
    /// The AUF build pipeline (`auf_tool` `build_variant_payload`) is
    /// contracted to apply, in order, at build time:
    ///   1. nibble bit-unshuffle (the work of `kernel_convert_block_q4_0_noshuffle`)
    ///   2. ushort-level 2D transpose of the q nibbles buffer
    ///   3. half-level 2D transpose of the d (scale) buffer
    ///
    /// `q_bytes` and `d_bytes` are the raw payload slices (zero-copy from the
    /// AUF mmap; the backend takes ownership of fresh `cl_mem` buffers it
    /// allocates and uploads). `ne00`/`ne01` are the K and M dimensions of
    /// the logical weight matrix (rows = `ne01`, cols = `ne00`).
    ///
    /// **WSWAP-5-AUF-PLACEHOLDER-DROP** (Phase 5 Sprint C): replaces the
    /// previous "register_pre_converted_soa(&Tensor, …)" entry which paired
    /// with a placeholder AOS `cl_mem` allocation. The new contract returns
    /// the SOA-backed tensor itself, so callers do **not** allocate a
    /// throw-away cl_mem of the AOS shape just to provide a registry key
    /// (saves ~547 MiB / 112 cl_mem on Llama 3.2 1B ratio=1.0). The registry
    /// key falls naturally on the SOA `d_buf` address (same as
    /// `prepare_noshuffle_buffers(swap_to_placeholder=true)`).
    ///
    /// **Behaviour**:
    /// - Non-OpenCL backend → NoOp returning `None` (caller falls back to
    ///   the GGUF AOS materialisation path which targets CPU/CUDA correctly).
    /// - Q4_0 noshuffle conversion program unavailable (driver-side build
    ///   failure on host OpenCL) → returns `None`; caller is expected to
    ///   fall back to the GGUF AOS path.
    /// - Otherwise → allocate `cl_mem` for `q_bytes`/`d_bytes`, create the
    ///   optional `image1d_buffer_t` view, register a noshuffle SOA entry
    ///   keyed on the `d_buf` address, and return a tensor whose buffer is a
    ///   `NoshuffleWeightBuffer` that owns the same SOA handles.
    ///
    /// **Default**: returns `None` (CPU / CUDA / host-OpenCL builds without
    /// the SOA pipeline). The OpenCL backend overrides this.
    ///
    /// **Contract**: caller invokes this once per Q4_0 weight tensor inside
    /// the `materialise_auf_soa_weight` branch of
    /// `SwapExecutor::build_layer_from_mmap`, before any runtime registry
    /// invalidation. The registration is repeated by
    /// `restore_pre_converted_soa_registration` after the per-batch
    /// `invalidate_noshuffle_soa_registry()` call so that the ENG-ALG-221
    /// ordering is preserved.
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn alloc_pre_converted_soa_tensor(
        &self,
        shape: crate::shape::Shape,
        q_bytes: &[u8],
        d_bytes: &[u8],
        ne00: usize,
        ne01: usize,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Re-register an AUF SOA-backed tensor against the noshuffle registry
    /// after a per-batch `invalidate_noshuffle_soa_registry()` clear.
    ///
    /// `tensor` must be the result of `alloc_pre_converted_soa_tensor`. The
    /// backend recovers the q/d/q_img handles from the tensor's
    /// `NoshuffleWeightBuffer` backing and re-inserts a registry entry keyed
    /// on the buffer's `cl_mem()` address (which doubles as `d_buf`).
    ///
    /// **Default**: NoOp. CPU / CUDA backends never populated the registry.
    ///
    /// **Contract**: invoked once per AUF SOA-backed weight in
    /// `SwapExecutor::execute_on_slots` Stage (d), strictly between
    /// `invalidate_noshuffle_soa_registry()` and the `ratio_generation` bump.
    /// Falls through to GGUF `ensure_noshuffle_soa_registered` for tensors
    /// that are not SOA-backed.
    #[allow(unused_variables)]
    fn restore_pre_converted_soa_registration(&self, tensor: &Tensor) -> Result<()> {
        Ok(())
    }

    /// LISWAP-6 — DMA-BUF alias buffer for swap path (Adreno rpcmem).
    ///
    /// The OpenCL backend on Adreno (with `--opencl-rpcmem`) can convert a
    /// host-mapped pointer (returned by `RpcmemSecondaryStore::host_ptr_for`)
    /// into a `cl_mem` alias via
    /// `CL_MEM_USE_HOST_PTR`, eliminating the `clEnqueueWriteBuffer` H2D copy
    /// during weight swap. The returned buffer holds the supplied lifetime
    /// guards (`secondary_arc` + `layer_region`) so the rpcmem allocation
    /// remains valid until every alias is dropped.
    ///
    /// # Parameters
    /// - `host_ptr`: Base host pointer of the rpcmem region.
    /// - `offset`: Byte offset within the region where the tensor starts.
    /// - `size`: Tensor byte length.
    /// - `dtype`: Tensor dtype.
    /// - `secondary_arc`: Lifetime guard 1 — keeps the `RpcmemSecondaryStore`
    ///   (and thus its region map) alive.
    /// - `layer_region`: Lifetime guard 2 — pins this layer's rpcmem allocation.
    ///
    /// # Returns
    /// - `Ok(None)` (default) — backend does not support alias path; caller
    ///   falls back to the standard mmap+memcpy materialisation.
    /// - `Ok(Some(buf))` — alias `cl_mem` created with both guards installed.
    ///
    /// # Safety
    /// Caller must guarantee:
    /// - `host_ptr.add(offset)` is valid and points to at least `size` bytes
    ///   of correctly aligned, initialised memory readable by the GPU driver.
    /// - The memory region remains live and unmoved for the entire lifetime
    ///   of any returned alias `Buffer`. `secondary_arc` + `layer_region`
    ///   are the lifetime guards that ensure this; both must outlive the
    ///   returned buffer (which is enforced by storing them inside it).
    /// - No other code mutates the region while alias buffers exist (the
    ///   secondary mmap is read-only per `SecondaryMmap` contract).
    #[allow(unused_variables)]
    #[cfg(feature = "opencl")]
    unsafe fn alloc_alias_weight_buffer(
        &self,
        host_ptr: *mut u8,
        offset: usize,
        size: usize,
        dtype: DType,
        secondary_arc: std::sync::Arc<dyn crate::memory::host::mmap::MmapKeepAlive>,
        layer_region: std::sync::Arc<dyn crate::memory::secondary::RpcmemRegionGuard>,
    ) -> Result<Option<std::sync::Arc<dyn crate::buffer::Buffer>>> {
        Ok(None)
    }

    /// See the `opencl`-feature variant above for safety contract.
    ///
    /// # Safety
    /// Same as the `opencl`-feature variant — caller guarantees `host_ptr`
    /// validity and lifetime via `secondary_arc` + `layer_region` guards.
    /// This default returns `Ok(None)` and never dereferences `host_ptr`,
    /// but the signature mirrors the opencl variant for trait uniformity.
    #[allow(unused_variables)]
    #[cfg(not(feature = "opencl"))]
    unsafe fn alloc_alias_weight_buffer(
        &self,
        host_ptr: *mut u8,
        offset: usize,
        size: usize,
        dtype: DType,
        secondary_arc: std::sync::Arc<dyn crate::memory::host::mmap::MmapKeepAlive>,
        layer_region: std::sync::Arc<dyn crate::memory::secondary::RpcmemRegionGuard>,
    ) -> Result<Option<std::sync::Arc<dyn crate::buffer::Buffer>>> {
        Ok(None)
    }

    /// 14-node layer graph fast path (qnn_oppkg backend 등). Default false.
    ///
    /// QNN OpPackage M3 (ENG-QNN-211/INV-174): 본 method가 true를 반환하는
    /// backend는 `execute_layer_graph(...)`을 정상 구현해야 한다.
    /// transformer.rs forward 진입 시 1회 호출하여 분기 결정에 사용.
    /// idempotent — 동일 인스턴스에 대해 호출 결과가 항상 동일해야 한다 (INV-174).
    fn supports_layer_graph(&self) -> bool {
        false
    }

    /// 14-node single-layer graph dispatch (qnn_oppkg M3.3에서 본격 구현).
    ///
    /// Layer graph fast path (ENG-QNN-211/213/214/INV-175): `supports_layer_graph()`
    /// 가 true인 backend는 `transformer.rs::forward_into` layer loop가 layer
    /// 1개를 본 method 1회 호출로 처리하도록 사용한다. trait method (matmul/
    /// rope/attention_gen 등) 호출은 fallback debug 경로로만 남으며, fast path
    /// 정상 동작 시 호출 횟수는 0이어야 한다 (INV-175).
    ///
    /// Default: 미지원 backend는 `Err`.
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn execute_layer_graph(
        &self,
        layer_idx: usize,
        x: &Tensor,
        kv_cache_k: &mut Tensor,
        kv_cache_v: &mut Tensor,
        pos: usize,
        n_kv: usize,
        x_out: &mut Tensor,
    ) -> Result<()> {
        anyhow::bail!("execute_layer_graph not supported by this backend")
    }

    /// GPU prefill flash attention. Returns Ok(true) if GPU dispatched, Ok(false) for CPU fallback.
    /// Default: CPU fallback (returns false).
    #[allow(unused_variables, clippy::too_many_arguments)]
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
        Ok(false)
    }

    // Embedding Lookup / Gather
    // src: [Rows, Cols] (Embeddings)
    // indices: [NumIndices] (Indices)
    // dst: [NumIndices, Cols] (Output)
    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
        // Default CPU implementation — supports F32 and F16 src → F32 dst
        let idx_data = unsafe {
            std::slice::from_raw_parts(indices.as_ptr() as *const u32, indices.size() / 4)
        };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.size() / 4) };

        let cols = src.shape().dims()[1];
        let src_rows = src.shape().dims()[0];

        match src.dtype() {
            DType::F32 => {
                let src_data = unsafe {
                    std::slice::from_raw_parts(src.as_ptr() as *const f32, src_rows * cols)
                };
                for (i, &idx) in idx_data.iter().enumerate() {
                    let offset = idx as usize * cols;
                    let target_offset = i * cols;
                    if offset + cols <= src_data.len() && target_offset + cols <= dst_data.len() {
                        dst_data[target_offset..target_offset + cols]
                            .copy_from_slice(&src_data[offset..offset + cols]);
                    }
                }
            }
            DType::F16 => {
                let src_data = unsafe {
                    std::slice::from_raw_parts(src.as_ptr() as *const half::f16, src_rows * cols)
                };
                for (i, &idx) in idx_data.iter().enumerate() {
                    let offset = idx as usize * cols;
                    let target_offset = i * cols;
                    if offset + cols <= src_data.len() && target_offset + cols <= dst_data.len() {
                        for d in 0..cols {
                            dst_data[target_offset + d] = src_data[offset + d].to_f32();
                        }
                    }
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "gather: unsupported src dtype {:?}",
                    src.dtype()
                ));
            }
        }
        Ok(())
    }

    /// Shift data within a single tensor (overlap-safe, like memmove).
    /// Offsets and count are in element units (Q4_0: 1 element = 1 block = 18 bytes).
    /// Default implementation uses CPU memmove via `std::ptr::copy`.
    fn buffer_shift(
        &self,
        tensor: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        let type_size = match tensor.dtype() {
            crate::buffer::DType::F32 => 4,
            crate::buffer::DType::F16 => 2,
            crate::buffer::DType::U8 => 1,
            crate::buffer::DType::Q4_0 => std::mem::size_of::<crate::quant::BlockQ4_0>(),
            _ => 1,
        };

        let ptr = tensor.as_mut_ptr();
        if ptr.is_null() {
            anyhow::bail!("Null pointer in buffer_shift (default impl)");
        }
        unsafe {
            std::ptr::copy(
                ptr.add(src_offset * type_size),
                ptr.add(dst_offset * type_size),
                count * type_size,
            );
        }
        Ok(())
    }

    // New API: Copy slice from src to dst
    // src_offset/dst_offset are ELEMENT offsets (not bytes) if Tensor is typed, but here for simplicity let's assume they are ELEMENT offsets relative to Tensor's DType?
    // Actually, Tensors are somewhat untyped regarding pointer arithmetic in this trait unless we know DType size.
    // Let's assume count is number of ELEMENTS. And offsets are ELEMENTS.
    // Caller must ensure types match.
    fn copy_slice(
        &self,
        src: &Tensor,
        dst: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        let type_size = match src.dtype() {
            crate::buffer::DType::F32 => 4,
            crate::buffer::DType::F16 => 2,
            crate::buffer::DType::U8 => 1,
            crate::buffer::DType::Q4_0 => std::mem::size_of::<crate::quant::BlockQ4_0>(),
            _ => 1, // Fallback
        };

        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();

        if src_ptr.is_null() || dst_ptr.is_null() {
            anyhow::bail!(
                "Null pointer in copy_slice (default impl), likely OpenCL buffer mismatch"
            );
        }

        unsafe {
            // Calculate byte offsets
            let src_byte_offset = src_offset * type_size;
            let dst_byte_offset = dst_offset * type_size;
            let byte_count = count * type_size;

            std::ptr::copy_nonoverlapping(
                src_ptr.add(src_byte_offset),
                dst_ptr.add(dst_byte_offset),
                byte_count,
            );
        }
        Ok(())
    }

    /// Bind the backend's hardware context to the calling thread.
    ///
    /// Required before any allocation / kernel launch on a background thread
    /// for backends that use a thread-local hardware context (CUDA driver).
    /// CPU and OpenCL backends embed the context in their queue/handle clones
    /// and do not require explicit binding — default is a no-op.
    ///
    /// Introduced to remove the L3→L1 `CudaBackend` downcast in
    /// `LayerObjectPool::new` (Migration Step 3-B, V-27).
    fn bind_current_thread(&self) -> Result<()> {
        Ok(())
    }

    // ── B-5b Phase 2 Stage 1: capability extension surface ────────────────
    //
    // The four methods below replace four downcast / direct-call patterns
    // currently scattered across `plan.rs`, `forward_gen.rs`, `transformer.rs`,
    // and the CUDA backend's `cpu_fallback` path:
    //
    // 1. `cpu_companion()` — GPU backends (OpenCL/CUDA/QNN) currently grab a
    //    fresh `CpuBackend` instance whenever they need a host fallback. Stage
    //    2 will route those through the GPU backend's owned `cpu_companion`
    //    so the host fallback shares the same backend state as the rest of
    //    the inference loop. Default `&self` is correct for `CpuBackend`.
    //
    // 2. `cpu_kernels()` — `plan.rs:696,717` and `forward_gen.rs:234,264,
    //    1286,1307,1436,1461` call the freestanding NEON `fused_matmul_*`
    //    entry points directly. Stage 2 will replace those with
    //    `backend.cpu_kernels()?.fused_matmul_*` so the OpenCL plan path can
    //    pick up the fast path via its `cpu_companion`. Default returns
    //    `None` (no fused kernels available); the `CpuBackend` NEON impl
    //    overrides on aarch64.
    //
    // 3. `as_opencl_secondary()` — `qnn_oppkg::with_opencl_secondary` (line
    //    132) currently does `as_any().downcast_ref::<OpenCLBackend>()`.
    //    Stage 2 will swap that for `backend.as_opencl_secondary()`. Default
    //    `None`; `OpenCLBackend` overrides.
    //
    // 4. `yield_after_layer()` — Stage 2-B (HEAD this commit) routed
    //    `plan.rs:1841` and `transformer.rs:1841` through this trait method.
    //    The freestanding helper was renamed to
    //    `crate::resilience::gpu_yield::gpu_yield_impl` and is invoked from
    //    per-backend overrides. Default remains no-op (CPU backends).
    //
    // RPN-145~180 hot-path concern (architect pre-survey): the
    // `yield_after_layer` default impl still incurs one vtable lookup per
    // layer per token. Stage 2's S25 microbench gate will decide whether to
    // keep this shape or fall back to a freestanding helper for the yield
    // method specifically.

    /// CPU companion backend used by GPU backends for host fallback paths.
    ///
    /// - `CpuBackend` (NEON / AVX2 / generic): returns `self`.
    /// - `OpenCLBackend` / `CudaBackend` (PC + embedded): return their owned
    ///   `cpu_companion` injected at construction time.
    ///
    /// No default impl: returning `self` here would coerce `&Self` to
    /// `&dyn Backend`, which Rust forbids without a `Self: Sized` bound;
    /// adding that bound, however, would prevent dispatch through
    /// `&dyn Backend` — the exact call shape Stage 2 needs. Each impl
    /// therefore writes the one-line override explicitly.
    fn cpu_companion(&self) -> &dyn Backend;

    /// CPU NEON kernel function pointer set. Only `CpuBackend` (NEON) returns
    /// `Some` on aarch64; all GPU backends and the AVX2 / generic CPU paths
    /// return `None`. Stage 2 callers must handle the `None` case by falling
    /// back to per-matmul `matmul_transposed` dispatches.
    fn cpu_kernels(&self) -> Option<&'static crate::cpu_kernels::CpuKernelSet> {
        None
    }

    /// Intra-token cooperative yield hook.
    ///
    /// Called once per layer in the decode loop (and per layer in the OpenCL
    /// plan path). The default body implements env-driven yield:
    /// flush via `self.synchronize()` and sleep `LLMRS_DECODE_YIELD_US`
    /// microseconds every `LLMRS_DECODE_YIELD_EVERY` layers, gated by
    /// `is_decode`. CPU backends inherit the default — when env vars are
    /// unset, `yield_every() == 0` short-circuits before `synchronize()`,
    /// so the hook is effectively zero-cost.
    ///
    /// Backends may override this to use a backend-specific yield primitive
    /// (custom scheduler hint, finer-grained sync), but the default works
    /// for all backends. The freestanding `gpu_yield_impl` helper was
    /// folded into this default body (S-2 sprint 2026-05-24) so that
    /// `backend.rs` (L2) no longer crosses the cross-cutting boundary that
    /// INV-LAYER-001 prohibits — env-var caching lives in
    /// `crate::yield_policy` (also L2).
    ///
    /// B-5b Phase 2 Stage 2-A S25 microbench showed vtable cost below
    /// measurement noise.
    fn yield_after_layer(&self, layer_idx: usize, is_decode: bool) {
        if !is_decode {
            return;
        }
        let every = crate::yield_policy::yield_every();
        if every == 0 {
            return;
        }
        if !(layer_idx + 1).is_multiple_of(every) {
            return;
        }
        // flush + wait: kernels already dispatched must drain before the
        // sleep is useful (otherwise the sleep overlaps with the in-flight
        // burst and buys nothing). synchronize() errors are swallowed —
        // yield is a best effort, not a correctness hook.
        let _ = self.synchronize();
        let us = crate::yield_policy::yield_us();
        if us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(us));
        } else {
            std::thread::yield_now();
        }
    }

    // COLD-EXT: ─────────────────────────────────────────────────────────────
    //
    // 본 method 는 *cold path 전용*. forward / decode loop 안에서 호출 금지.
    // 차용 출처: ggml `ggml_backend_reg_get_proc_address(reg, name)` —
    // 표준 vtable 밖의 backend-specific 진입점을 string lookup 으로 통일.
    //
    // 호출지는 `EXT_OPENCL_QUEUE` 등 모듈 const 만 인자로 전달해야 한다
    // (자유 문자열 silent-None 방어). 사용 가능한 key 는
    // `engine/src/backend.rs` 상단 `EXT_*` const 열거 참조.
    //
    // Architect 라운드: `arch/sprint_backend_extension_round.md` R-EXT-1/3.

    // ────────────────────────────────────────────────────────────────────
    // §13.8-L hot path Profile hook (S-L-1):
    // `--profile-events` / `--cuda-profile` 가 op 라벨을 흘려넣을 때 사용.
    // GPU backend 만 override. CPU/QNN 등은 default no-op 으로 충분.
    // 이 3 메서드를 통해 forward_gen.rs 의 OpenCLBackend / CudaBackend
    // downcast 5건이 사라진다.

    /// `true` 면 backend 가 자체적으로 per-op GPU event 프로파일링을 수행
    /// 한다는 의미. forward_gen 의 wall-clock + `clFinish()` 경로를
    /// 비활성화해 double-counting 을 막는다. Default `false`.
    fn profile_events_enabled(&self) -> bool {
        false
    }

    /// caller-side label hint — backend 가 다음 dispatch 의 op 명을
    /// 알게 해서 `matmul_qkv` / `matmul_wo` / `matmul_ffn` / `lm_head` 같이
    /// 동일 GEMV/GEMM 커널이라도 구분 가능하게 한다. Default no-op.
    fn set_op_label(&self, _label: &'static str) {}

    /// Label clear hook. set_op_label 의 pair. Default no-op.
    fn clear_op_label(&self) {}

    /// §13.8-L S-L-2: GPU score accumulator 접근 hook. OpenCL backend 만
    /// 활성, 그 외 backend 는 default `None`. forward.rs / forward_gen.rs
    /// 의 `backend.as_any().downcast_ref::<OpenCLBackend>().and_then(...)`
    /// chain 2건이 본 trait method 호출로 통합된다.
    fn gpu_score_acc(&self) -> Option<&dyn GpuScoreAccess> {
        None
    }

    /// `gpu_score_acc` 의 mutable 버전. OpenCL backend 의 inherent
    /// `gpu_score_acc_mut(&self) -> Option<&mut ...>` 와 동일한 패턴
    /// (`UnsafeCell`-backed). single-threaded 추론 가정.
    #[allow(clippy::mut_from_ref)]
    fn gpu_score_acc_mut(&self) -> Option<&mut dyn GpuScoreAccess> {
        None
    }

    /// Raw OpenCL `cl_command_queue` handle (FORMAT Phase 2 Stage E). A
    /// borrowed-context backend-cap plugin (which has no `self.queue` of its
    /// own) must enqueue its kernels on the engine's live command queue so they
    /// stay serialized with surrounding engine ops over shared `cl_mem`; this
    /// accessor lends that raw handle so the call sites can pack it into the
    /// `QuantAttn*Args.cl_queue` ABI slot. OpenCL backend only; others return
    /// null. (The built-in OpenCL impl ignores the slot and uses `self.queue`
    /// directly, so populating it is a no-op for the in-engine path.)
    fn cl_command_queue_ptr(&self) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }

    /// Whether this backend's device uses the sub-group-less (nosub) attention
    /// path (FORMAT Phase 2 Stage E). The KIVI FORMAT gates its native fused
    /// attention on this device property (it is consistent with the standard
    /// `attention_gen` work-group reduction the nosub path also uses). Previously
    /// read off the QuantAttn cap; now a backend-owned device property so it stays
    /// byte-identical once the KIVI compute moves to a dlopen plugin. OpenCL
    /// returns its `f16_is_nosub`; others default `false`.
    fn is_nosub_device(&self) -> bool {
        false
    }

    /// Cold-path backend-specific extension lookup (string-keyed downcast).
    ///
    /// `downcast_ref::<OpenCLBackend>()` 등 outer-module downcast 를
    /// trait method 한 군데로 통일. Hot path (forward/decode loop) 진입
    /// 금지 — `Any` downcast 비용 + lookup branch 가 매 layer 호출에
    ///누적된다. Sprint scope = qnn_oppkg swap path / secondary_mmap loader
    /// / transformer loader 의 4건. KIVI / forward downcast 는 별도.
    ///
    /// 반환값을 사용할 때는 호출지에서 `downcast_ref::<...>()` 으로
    /// 구체 타입을 꺼낸다. 등록되지 않은 key 는 `None` 반환.
    fn get_extension(&self, _name: &str) -> Option<&dyn std::any::Any> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::Backend;
    use crate::backend::cpu::common::CpuBackendCommon;
    use crate::buffer::DType;
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn make_cpu_tensor(data: &[f32]) -> Tensor {
        let memory = Galloc::new();
        let buf = memory.alloc(data.len() * 4, DType::F32).unwrap();
        let mut t = Tensor::new(
            Shape::new(vec![data.len()]),
            buf,
            Arc::new(CpuBackendCommon::new()),
        );
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    #[test]
    fn test_gelu_tanh_known_values() {
        let backend = CpuBackendCommon::new();

        // gelu_tanh(0.0) * 1.0 ≈ 0.0
        let mut gate = make_cpu_tensor(&[0.0f32]);
        let up = make_cpu_tensor(&[1.0f32]);
        backend.gelu_tanh_mul(&mut gate, &up).unwrap();
        let result = gate.as_slice::<f32>()[0];
        assert!(
            result.abs() < 1e-6,
            "gelu_tanh(0)*1 should be ~0, got {}",
            result
        );

        // gelu_tanh(1.0) * 1.0 ≈ 0.8412
        let mut gate = make_cpu_tensor(&[1.0f32]);
        let up = make_cpu_tensor(&[1.0f32]);
        backend.gelu_tanh_mul(&mut gate, &up).unwrap();
        let result = gate.as_slice::<f32>()[0];
        assert!(
            (result - 0.8412).abs() < 1e-3,
            "gelu_tanh(1)*1 should be ~0.8412, got {}",
            result
        );

        // gelu_tanh(-1.0) * 1.0 ≈ -0.1588
        let mut gate = make_cpu_tensor(&[-1.0f32]);
        let up = make_cpu_tensor(&[1.0f32]);
        backend.gelu_tanh_mul(&mut gate, &up).unwrap();
        let result = gate.as_slice::<f32>()[0];
        assert!(
            (result - (-0.1588)).abs() < 1e-3,
            "gelu_tanh(-1)*1 should be ~-0.1588, got {}",
            result
        );
    }

    /// LISWAP-2 prototype Phase 1: the CPU backend has no transfer
    /// queue, so `enqueue_write_async` must fall through the default
    /// trait impl (which delegates to `copy_weight_from`) and produce
    /// byte-identical output.
    #[test]
    fn test_async_transfer_default_fallback_byte_equal() {
        let backend = CpuBackendCommon::new();
        assert!(
            !backend.supports_async_transfer(),
            "CPU backend must not advertise async transfer"
        );

        let src = make_cpu_tensor(&[1.0f32, -2.5, 3.25, 4.0, -0.125, 6.5, 7.0, 8.0]);
        let sync_dst = backend.copy_weight_from(&src).unwrap();
        let (async_dst, evt) = backend.enqueue_write_async(&src).unwrap();
        backend.wait_event_blocking(&evt).unwrap();

        let sync_slice = sync_dst.as_slice::<f32>();
        let async_slice = async_dst.as_slice::<f32>();
        assert_eq!(sync_slice.len(), async_slice.len());
        assert_eq!(
            sync_slice, async_slice,
            "default async fallback must be byte-equal to sync copy_weight_from"
        );
    }

    /// LISWAP-2 prototype Phase 1: with the OpenCL transfer queue
    /// enabled (default ON), `enqueue_write_async` must produce a
    /// device buffer whose readback is byte-equal to the sync
    /// `copy_weight_from` path. Skipped when the host has no usable
    /// OpenCL platform (e.g. CI without GPU drivers).
    #[cfg(feature = "opencl")]
    #[test]
    fn test_async_transfer_opencl_byte_equal() {
        // Force the env-gate ON so the test exercises the transfer queue.
        // Safe in tests because we never spawn other threads here.
        // SAFETY: single-threaded test process; no concurrent env reads.
        unsafe {
            std::env::set_var("LLMRS_OPENCL_TRANSFER_QUEUE", "1");
        }

        // `OpenCLBackend::new()` returns `Result` but the underlying
        // `Device::first` path can panic on hosts without an OpenCL
        // driver. Wrap with `catch_unwind` so we cleanly skip in those
        // environments (CI / sandboxed dev shells). `AssertUnwindSafe`
        // is fine because the closure has no shared mutable state.
        let backend = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            crate::backend::opencl::OpenCLBackend::new,
        )) {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => {
                eprintln!(
                    "[test] skipping OpenCL async transfer byte-equal: \
                     OpenCL backend unavailable ({e})"
                );
                return;
            }
            Err(_) => {
                eprintln!(
                    "[test] skipping OpenCL async transfer byte-equal: \
                     OpenCL backend init panicked (no devices?)"
                );
                return;
            }
        };

        // 1 KiB F32 tensor with a recognisable pattern. Small enough
        // to keep the test cheap on slower hosts.
        const N: usize = 256;
        let mut src_data = Vec::with_capacity(N);
        for i in 0..N {
            src_data.push(((i as f32) * 0.5) - 17.0);
        }
        let src = make_cpu_tensor(&src_data);

        // Sync reference path.
        let sync_dst = backend.copy_weight_from(&src).unwrap();
        let mut sync_buf = vec![0u8; sync_dst.size()];
        backend.read_buffer(&sync_dst, &mut sync_buf).unwrap();

        // Async path under test.
        let (async_dst, evt) = backend.enqueue_write_async(&src).unwrap();
        backend.wait_event_blocking(&evt).unwrap();
        // Drain the secondary queue too — `wait_event_blocking` only
        // waits on the recorded event, not on subsequent commands.
        backend.synchronize().unwrap();
        let mut async_buf = vec![0u8; async_dst.size()];
        backend.read_buffer(&async_dst, &mut async_buf).unwrap();

        assert_eq!(sync_buf.len(), async_buf.len());
        assert_eq!(
            sync_buf, async_buf,
            "OpenCL enqueue_write_async readback must equal sync copy_weight_from"
        );
        assert!(
            backend.supports_async_transfer(),
            "OpenCL backend should advertise async transfer when env gate is on"
        );
    }
}
