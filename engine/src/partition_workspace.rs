//! Partition workspace types — UMA hybrid CPU-GPU tensor partition data carriers.
//!
//! §13.8-G shared identifier promotion (B-5a sprint):
//! `PartitionWsCell` + `PartitionWorkspace`는 `layers/workspace.rs`에 정의되어
//! `backend/opencl/plan.rs`(L1 → L3 import 위반)에서 직접 import되었다. 두 struct는
//! 사실상 forward path의 *workspace data carrier*이며, 어느 도메인의 *상태 owner*가
//! 아니다 (UnsafeCell wrapper + per-decode lifetime). L3 inference 도메인의
//! identifier로 명시하여 backend가 inference를 import하는 *데이터 의존 방향*을
//! INV-LAYER-001 정상 경로로 정리한다.

use std::cell::UnsafeCell;
use std::sync::Arc;

use anyhow::Result;

use crate::buffer::{Buffer, DType};
use crate::memory::Memory;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Thin wrapper around `UnsafeCell<PartitionWorkspace>` that makes the
/// interior `Sync` for the restricted single-threaded-dispatch safety model
/// enforced by the OpenCL plan (`PartitionStep::run`) and forward_gen
/// (shared TLS thread). Do not use outside of that invariant.
pub struct PartitionWsCell(pub UnsafeCell<PartitionWorkspace>);

impl PartitionWsCell {
    pub fn new(ws: PartitionWorkspace) -> Self {
        Self(UnsafeCell::new(ws))
    }

    #[inline]
    pub fn get(&self) -> *mut PartitionWorkspace {
        self.0.get()
    }
}

// SAFETY: The LayerWorkspace single-threaded-dispatch contract (see
// `LayerWorkspace::partition_ws` doc) guarantees no aliased mutable access
// across threads. Both the forward_gen partition block and the plan-path
// `PartitionStep::run` execute on the same dispatch thread.
unsafe impl Send for PartitionWsCell {}
unsafe impl Sync for PartitionWsCell {}

/// Pre-allocated scratch tensors for partitioned FFN gate/up output (decode only).
///
/// Each pair (gpu, cpu) holds one partial result that together form the full
/// output dimension. Created once during partition setup and reused every token.
///
/// Only FFN gate/up are partitioned; attention and FFN down run GPU-only.
pub struct PartitionWorkspace {
    // --- FFN gate/up (Strategy B: CPU keeps its partial; GPU keeps its partial) ---
    /// GPU partial output for gate projection: [1, 1, gate_split_row]
    pub gate_gpu: Tensor,
    /// CPU partial output for gate projection: [1, 1, ffn_hidden - gate_split_row]
    pub gate_cpu: Tensor,
    /// GPU partial output for up projection: [1, 1, up_split_row]
    pub up_gpu: Tensor,
    /// CPU partial output for up projection: [1, 1, ffn_hidden - up_split_row]
    pub up_cpu: Tensor,
    /// CPU-side copy of residual for CPU matmul input: [1, 1, dim]
    /// UnifiedBuffer::as_ptr() returns null when unmapped, so we copy residual
    /// to this CPU buffer via read_buffer() before CPU matmul.
    pub residual_cpu: Tensor,

    // --- FFN down (Strategy B) ---
    /// GPU partial output for down projection: [1, 1, hidden_size]
    /// Contains `W_down[:, :split_col] @ silu(gate_gpu)*up_gpu`.
    pub down_partial_gpu: Tensor,
    /// CPU partial output for down projection: [1, 1, hidden_size]
    /// Contains `W_down[:, split_col:] @ silu(gate_cpu)*up_cpu`.
    pub down_partial_cpu: Tensor,
    /// GPU-side staging buffer to upload the CPU partial for the final
    /// elementwise add: [1, 1, hidden_size].
    pub cpu_merge_staging: Tensor,
    /// Host-visible 4-byte flag (CL_MEM_ALLOC_HOST_PTR, permanent-mapped).
    /// Written by the `kernel_add_rms_norm_oop_f4_sigflag` kernel variant
    /// via `atomic_xchg` once the `residual` output is globally visible;
    /// `PartitionStep::run` spin-polls this flag instead of calling
    /// `clFinish(queue)` when `LLMRS_PARTITION_POLL_FLAG=1`. Eliminates
    /// per-layer driver round-trip latency (~0.29 ms on Adreno 830).
    pub ready_flag: Tensor,
}

impl PartitionWorkspace {
    /// Allocate partition workspace buffers for FFN gate/up partitioned projections.
    ///
    /// `gate_split_row`/`up_split_row`: GPU-side split row count for FFN gate/up
    ///   projections (rest is CPU-side). Caller provides raw primitives so this
    ///   L2 workspace does not import L3 `PartitionContext`/`PartitionedWeight`.
    /// `ffn_hidden`: FFN intermediate dimension (gate/up out_dim).
    /// `hidden_size`: model hidden dimension (residual size).
    /// `gpu_alloc`: allocator closure for GPU-side buffers.
    /// `gpu_backend`: backend for GPU-side tensors.
    /// `cpu_backend`: backend for CPU-side tensors.
    pub fn new(
        gate_split_row: usize,
        up_split_row: usize,
        ffn_hidden: usize,
        hidden_size: usize,
        gpu_alloc: &dyn Fn(usize, DType) -> Result<Arc<dyn Buffer>>,
        gpu_backend: Arc<dyn crate::backend::Backend>,
        cpu_backend: Arc<dyn crate::backend::Backend>,
    ) -> Result<Self> {
        use crate::memory::galloc::Galloc;

        let cpu_mem = Galloc::new();

        // Helper: allocate a (gpu, cpu) tensor pair given split_row/out_dim.
        let alloc_pair = |sr: usize, out_dim: usize| -> Result<(Tensor, Tensor)> {
            let cr = out_dim - sr;
            let gpu_buf = gpu_alloc(sr * 4, DType::F32)?;
            let cpu_buf = cpu_mem.alloc(cr * 4, DType::F32)?;
            Ok((
                Tensor::new(Shape::new(vec![1, 1, sr]), gpu_buf, gpu_backend.clone()),
                Tensor::new(Shape::new(vec![1, 1, cr]), cpu_buf, cpu_backend.clone()),
            ))
        };

        // FFN gate/up only
        let (gate_gpu, gate_cpu) = alloc_pair(gate_split_row, ffn_hidden)?;
        let (up_gpu, up_cpu) = alloc_pair(up_split_row, ffn_hidden)?;

        // FFN down (Strategy B whole-slice):
        //   down_partial_gpu  [1, 1, hidden]  GPU F32
        //   down_partial_cpu  [1, 1, hidden]  CPU F32
        //   cpu_merge_staging [1, 1, hidden]  GPU F32  (upload target for CPU partial)
        let down_partial_gpu = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            gpu_alloc(hidden_size * 4, DType::F32)?,
            gpu_backend.clone(),
        );
        let cpu_merge_staging = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            gpu_alloc(hidden_size * 4, DType::F32)?,
            gpu_backend.clone(),
        );
        // Phase 1a: permanent-map `cpu_merge_staging` UnifiedBuffer so the
        // plan-path merge can memcpy directly from `down_partial_cpu` into the
        // GPU-visible host pointer, eliminating the per-layer
        // `enqueue_write_buffer` (~0.13 ms/layer × 16 = 2.1 ms/tok on S25).
        // Opt-out via LLMRS_PARTITION_ZCOPY_MERGE=0 for A/B benchmarking.
        #[cfg(feature = "opencl")]
        if std::env::var("LLMRS_PARTITION_ZCOPY_MERGE")
            .map(|v| v != "0")
            .unwrap_or(true)
            && let Some(ub) = cpu_merge_staging
                .buffer()
                .as_any()
                .downcast_ref::<crate::memory::opencl::unified::UnifiedBuffer>()
        {
            let _ = ub.map();
        }
        let down_partial_cpu = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            cpu_mem.alloc(hidden_size * 4, DType::F32)?,
            cpu_backend.clone(),
        );

        let residual_cpu = {
            let buf = cpu_mem.alloc(hidden_size * 4, DType::F32)?;
            Tensor::new(Shape::new(vec![1, 1, hidden_size]), buf, cpu_backend)
        };
        // Ready-flag: 4 bytes, GPU-visible + CPU-visible via ALLOC_HOST_PTR.
        // Permanent-mapped so `as_mut_ptr()` yields the host pointer the
        // CPU polls while the sigflag kernel signals completion.
        let ready_flag = Tensor::new(
            Shape::new(vec![1]),
            gpu_alloc(4, DType::F32)?,
            gpu_backend.clone(),
        );
        #[cfg(feature = "opencl")]
        if let Some(ub) = ready_flag
            .buffer()
            .as_any()
            .downcast_ref::<crate::memory::opencl::unified::UnifiedBuffer>()
        {
            let _ = ub.map();
        }

        Ok(Self {
            gate_gpu,
            gate_cpu,
            up_gpu,
            up_cpu,
            residual_cpu,
            down_partial_gpu,
            down_partial_cpu,
            cpu_merge_staging,
            ready_flag,
        })
    }
}
