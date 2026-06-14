use crate::backend::Backend;
use crate::buffer::DType;
use crate::memory::Memory;
use crate::shape::Shape;
use crate::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

// Backward-compat re-export (B-5a sprint, 1 sprint 한정). 다음 cleanup sprint에서 제거.
#[allow(deprecated)]
pub use crate::partition_workspace::{PartitionWorkspace, PartitionWsCell};

/// Pre-allocated workspace for a single transformer layer.
/// Reused across all tokens to avoid per-token memory allocation overhead.
pub struct LayerWorkspace {
    pub q: Tensor,        // [batch, 1, q_dim]
    pub k: Tensor,        // [batch, 1, k_dim]
    pub v: Tensor,        // [batch, 1, v_dim]
    pub out_attn: Tensor, // [batch, 1, q_dim]
    pub gate: Tensor,     // [batch, 1, ffn_hidden]
    pub up: Tensor,       // [batch, 1, ffn_hidden]
    pub down: Tensor,     // [batch, 1, dim]
    pub residual: Tensor, // [batch, 1, dim]
    pub attn_out: Tensor, // [batch, 1, dim]
    /// Pre-allocated scores buffer for attention (max_seq_len size)
    pub scores: Vec<f32>,
    /// Score buffer offset: kv_start_pos for local attention layers, 0 for global.
    /// Used to correctly map ws.scores[t] to cache position (score_offset + t).
    pub score_offset: usize,
    /// Pre-allocated KV cache cast buffers (F32→F16 conversion).
    /// Avoids GPU memory allocation per token per layer.
    pub k_cast: Option<Tensor>,
    pub v_cast: Option<Tensor>,
    /// Pre-allocated scratch buffers for CPU-GPU tensor partition (decode only).
    /// None when tensor partition is disabled.
    ///
    /// Wrapped in `Arc<UnsafeCell<..>>` so the OpenCL plan path
    /// (`build_partitioned_layer_plan`) can share ownership with the
    /// forward_gen partition path — both mutate the workspace through the
    /// same allocation. Safety: single-threaded dispatch (same model as
    /// `KernelStep`). The Arc clone that the plan retains bumps the refcount
    /// so the workspace outlives the plan even if `LayerWorkspace` is
    /// reallocated mid-flight (e.g. on UMA switch). See
    /// `backend/opencl/plan.rs::PartitionPlanContext` for the dispatch-side
    /// invariant enforcement.
    pub partition_ws: Option<Arc<PartitionWsCell>>,
    /// Fused-merge carry slots: when `LLMRS_PARTITION_FUSED_MERGE=1`, the
    /// previous layer's partition FFN end leaves its `down_partial_gpu` and
    /// `cpu_merge_staging` (CPU partial uploaded) tensors here so the next
    /// layer's entry can fuse them with the residual + attn_norm into a
    /// single kernel. Cleared at the start of each forward_into call and
    /// whenever the path is disabled. The tensors are shallow clones (Arc
    /// buffers) of `partition_ws.down_partial_gpu` / `cpu_merge_staging`.
    pub partition_prev_gpu_partial: Option<Tensor>,
    pub partition_prev_cpu_staging: Option<Tensor>,
}

impl LayerWorkspace {
    /// Extract all buffer Arcs (for keeping GPU buffers alive during switch).
    pub fn take_buffers(&self) -> Vec<Arc<dyn crate::buffer::Buffer>> {
        let mut bufs = vec![
            self.q.buffer().clone(),
            self.k.buffer().clone(),
            self.v.buffer().clone(),
            self.out_attn.buffer().clone(),
            self.gate.buffer().clone(),
            self.up.buffer().clone(),
            self.down.buffer().clone(),
            self.residual.buffer().clone(),
            self.attn_out.buffer().clone(),
        ];
        if let Some(ref t) = self.k_cast {
            bufs.push(t.buffer().clone());
        }
        if let Some(ref t) = self.v_cast {
            bufs.push(t.buffer().clone());
        }
        if let Some(ref pw_cell) = self.partition_ws {
            // SAFETY: single-threaded dispatch (see field doc).
            let pw: &PartitionWorkspace = unsafe { &*pw_cell.get() };
            bufs.push(pw.gate_gpu.buffer().clone());
            bufs.push(pw.gate_cpu.buffer().clone());
            bufs.push(pw.up_gpu.buffer().clone());
            bufs.push(pw.up_cpu.buffer().clone());
            bufs.push(pw.residual_cpu.buffer().clone());
            bufs.push(pw.down_partial_gpu.buffer().clone());
            bufs.push(pw.down_partial_cpu.buffer().clone());
            bufs.push(pw.cpu_merge_staging.buffer().clone());
        }
        bufs
    }

    /// Re-tag all tensors with a new backend without reallocating buffers.
    /// Used for UMA GPU↔CPU switch where buffers are already host-accessible.
    pub fn retag_backend(self, backend: Arc<dyn Backend>) -> Self {
        let retag = |t: Tensor| -> Tensor {
            Tensor::new(t.shape().clone(), t.buffer().clone(), backend.clone())
        };
        Self {
            q: retag(self.q),
            k: retag(self.k),
            v: retag(self.v),
            out_attn: retag(self.out_attn),
            gate: retag(self.gate),
            up: retag(self.up),
            down: retag(self.down),
            residual: retag(self.residual),
            attn_out: retag(self.attn_out),
            scores: self.scores,
            score_offset: self.score_offset,
            k_cast: self.k_cast.map(&retag),
            v_cast: self.v_cast.map(&retag),
            partition_ws: self.partition_ws.map(|pw_cell| {
                // `retag_backend` is only called on UMA GPU↔CPU switch, which
                // invalidates any active plan beforehand — so the Arc should
                // be unique here. Fall back to deep-cloning if unexpectedly
                // shared.
                let cell = Arc::try_unwrap(pw_cell).unwrap_or_else(|shared| {
                    // SAFETY: single-threaded dispatch invariant.
                    let pw: &PartitionWorkspace = unsafe { &*shared.get() };
                    PartitionWsCell::new(PartitionWorkspace {
                        gate_gpu: pw.gate_gpu.clone(),
                        gate_cpu: pw.gate_cpu.clone(),
                        up_gpu: pw.up_gpu.clone(),
                        up_cpu: pw.up_cpu.clone(),
                        residual_cpu: pw.residual_cpu.clone(),
                        down_partial_gpu: pw.down_partial_gpu.clone(),
                        down_partial_cpu: pw.down_partial_cpu.clone(),
                        cpu_merge_staging: pw.cpu_merge_staging.clone(),
                        ready_flag: pw.ready_flag.clone(),
                    })
                });
                let pw = cell.0.into_inner();
                Arc::new(PartitionWsCell::new(PartitionWorkspace {
                    gate_gpu: retag(pw.gate_gpu),
                    gate_cpu: retag(pw.gate_cpu),
                    up_gpu: retag(pw.up_gpu),
                    up_cpu: retag(pw.up_cpu),
                    residual_cpu: retag(pw.residual_cpu),
                    down_partial_gpu: retag(pw.down_partial_gpu),
                    down_partial_cpu: retag(pw.down_partial_cpu),
                    cpu_merge_staging: retag(pw.cpu_merge_staging),
                    ready_flag: retag(pw.ready_flag),
                }))
            }),
            partition_prev_gpu_partial: self.partition_prev_gpu_partial.map(&retag),
            partition_prev_cpu_staging: self.partition_prev_cpu_staging.map(&retag),
        }
    }

    pub fn new(
        config: WorkspaceConfig,
        memory: &dyn Memory,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let alloc = |shape: Vec<usize>| -> Result<Tensor> {
            let size: usize = shape.iter().product();
            let buf = memory.alloc(size * 4, DType::F32)?;
            Ok(Tensor::new(Shape::new(shape), buf, backend.clone()))
        };

        Ok(Self {
            q: alloc(vec![config.batch_size, 1, config.q_dim])?,
            k: alloc(vec![config.batch_size, 1, config.k_dim])?,
            v: alloc(vec![config.batch_size, 1, config.v_dim])?,
            out_attn: alloc(vec![config.batch_size, 1, config.q_dim])?,
            gate: alloc(vec![config.batch_size, 1, config.ffn_hidden])?,
            up: alloc(vec![config.batch_size, 1, config.ffn_hidden])?,
            down: alloc(vec![config.batch_size, 1, config.dim])?,
            residual: alloc(vec![config.batch_size, 1, config.dim])?,
            attn_out: alloc(vec![config.batch_size, 1, config.dim])?,
            scores: vec![0.0; config.n_heads * config.max_seq_len],
            score_offset: 0,
            k_cast: None, // Lazily initialized on first use with correct dtype
            v_cast: None,
            partition_ws: None, // Set externally when tensor partition is enabled
            partition_prev_gpu_partial: None,
            partition_prev_cpu_staging: None,
        })
    }

    /// Clear fused-merge carry slots. Called at the start of every
    /// `forward_into` to ensure layer 0 does not consume stale state from a
    /// previous token's final layer.
    pub fn reset_partition_prev(&mut self) {
        self.partition_prev_gpu_partial = None;
        self.partition_prev_cpu_staging = None;
    }
}

pub struct WorkspaceConfig {
    pub batch_size: usize,
    pub dim: usize,
    pub q_dim: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub ffn_hidden: usize,
    pub n_heads: usize,
    pub max_seq_len: usize,
}

/// Pre-allocated scratch buffers for partitioned FFN gate/up during prefill.
///
/// Sized for `[batch, seq_len, *]` (vs `PartitionWorkspace` which is decode-only
/// `[1, 1, *]`). Allocated once per prefill call and reused across all layers,
/// eliminating per-layer alloc/free churn (~56 alloc cycles on a 28-layer model).
///
/// All four partial buffers are sized at the exact `split_row` / `cpu_rows`
/// determined by the partition context — prefill workspaces are throw-away
/// (rebuilt every prefill call), so the geometry is always consistent with
/// the active `partition_ctx`.
pub struct PrefillPartitionScratch {
    /// CPU-host-visible copy of normalized FFN input `x` for CPU NEON matmul.
    /// Refilled once per layer via `read_buffer(x → x_cpu)`.
    pub x_cpu: Tensor,
    /// GPU partial output of `gate = matmul(x, gate.gpu_slice)`:
    /// shape `[batch, seq_len, gate.split_row]`.
    pub gate_gpu_partial: Tensor,
    /// CPU partial output of `gate = matmul(x_cpu, gate.cpu_slice)`:
    /// shape `[batch, seq_len, ffn_hidden - gate.split_row]`.
    pub gate_cpu_partial: Tensor,
    /// GPU partial output of up projection.
    pub up_gpu_partial: Tensor,
    /// CPU partial output of up projection.
    pub up_cpu_partial: Tensor,
    /// Scratch for `merge_partials_2d`: GPU partial readback target.
    /// Capacity = `total_rows * max(gate.split_row, up.split_row) * 4`.
    pub merge_gpu_temp: Vec<u8>,
    /// Scratch for `merge_partials_2d`: interleaved [total_rows, ffn_hidden] F32.
    /// Capacity = `total_rows * ffn_hidden * 4`.
    pub merge_buf: Vec<u8>,
}

/// Pre-allocated workspace for prefill (batch token processing).
/// Reuses GPU buffers across layers to avoid alloc/free churn that crashes NVIDIA's OpenCL driver.
pub struct PrefillWorkspace {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub out_attn: Tensor,
    pub attn_out_proj: Tensor,
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    pub residual: Tensor,
    pub residual_ffn: Tensor,
    /// Lazily initialized cast buffers for F16/Q4 KV cache
    pub k_cast: Option<Tensor>,
    pub v_cast: Option<Tensor>,
    /// Partition scratch — populated by `init_partition` when tensor partition is active.
    pub partition: Option<PrefillPartitionScratch>,
    seq_len: usize,
}

impl PrefillWorkspace {
    pub fn new(
        config: &WorkspaceConfig,
        seq_len: usize,
        memory: &dyn Memory,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let alloc = |shape: Vec<usize>| -> Result<Tensor> {
            let size: usize = shape.iter().product();
            let buf = memory.alloc(size * 4, DType::F32)?;
            Ok(Tensor::new(Shape::new(shape), buf, backend.clone()))
        };
        let b = config.batch_size;
        Ok(Self {
            q: alloc(vec![b, seq_len, config.q_dim])?,
            k: alloc(vec![b, seq_len, config.k_dim])?,
            v: alloc(vec![b, seq_len, config.v_dim])?,
            out_attn: alloc(vec![b, seq_len, config.q_dim])?,
            attn_out_proj: alloc(vec![b, seq_len, config.dim])?,
            gate: alloc(vec![b, seq_len, config.ffn_hidden])?,
            up: alloc(vec![b, seq_len, config.ffn_hidden])?,
            down: alloc(vec![b, seq_len, config.dim])?,
            residual: alloc(vec![b, seq_len, config.dim])?,
            residual_ffn: alloc(vec![b, seq_len, config.dim])?,
            k_cast: None,
            v_cast: None,
            partition: None,
            seq_len,
        })
    }

    /// Current sequence length this workspace is sized for.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Allocate the partition scratch buffers sized for the active
    /// `PartitionContext`. Idempotent — does nothing if `partition` is already
    /// populated. Called once per prefill (after PrefillWorkspace::new) when
    /// tensor partition is active for this model.
    ///
    /// `gate_split_row` / `up_split_row` come from `PartitionContext::gate.split_row`
    /// / `up.split_row`. `ffn_hidden` is the full FFN intermediate dimension.
    #[allow(clippy::too_many_arguments)]
    pub fn init_partition(
        &mut self,
        batch_size: usize,
        seq_len: usize,
        dim: usize,
        ffn_hidden: usize,
        gate_split_row: usize,
        up_split_row: usize,
        gpu_memory: &dyn Memory,
        gpu_backend: Arc<dyn Backend>,
        cpu_backend: Arc<dyn Backend>,
    ) -> Result<()> {
        if self.partition.is_some() {
            return Ok(());
        }

        use crate::memory::galloc::Galloc;
        let cpu_mem = Galloc::new();

        let total_rows = batch_size * seq_len;
        let gate_cpu_rows = ffn_hidden - gate_split_row;
        let up_cpu_rows = ffn_hidden - up_split_row;

        // x_cpu: CPU-host buffer the GPU `x` tensor is read into per layer.
        let x_cpu_buf = cpu_mem.alloc(total_rows * dim * 4, DType::F32)?;
        let x_cpu = Tensor::new(
            Shape::new(vec![batch_size, seq_len, dim]),
            x_cpu_buf,
            cpu_backend.clone(),
        );

        // GPU partials — alloc once via gpu_memory.
        let gate_gpu_buf = gpu_memory.alloc(total_rows * gate_split_row * 4, DType::F32)?;
        let gate_gpu_partial = Tensor::new(
            Shape::new(vec![batch_size, seq_len, gate_split_row]),
            gate_gpu_buf,
            gpu_backend.clone(),
        );
        let up_gpu_buf = gpu_memory.alloc(total_rows * up_split_row * 4, DType::F32)?;
        let up_gpu_partial = Tensor::new(
            Shape::new(vec![batch_size, seq_len, up_split_row]),
            up_gpu_buf,
            gpu_backend.clone(),
        );

        // CPU partials.
        let gate_cpu_buf = cpu_mem.alloc(total_rows * gate_cpu_rows * 4, DType::F32)?;
        let gate_cpu_partial = Tensor::new(
            Shape::new(vec![batch_size, seq_len, gate_cpu_rows]),
            gate_cpu_buf,
            cpu_backend.clone(),
        );
        let up_cpu_buf = cpu_mem.alloc(total_rows * up_cpu_rows * 4, DType::F32)?;
        let up_cpu_partial = Tensor::new(
            Shape::new(vec![batch_size, seq_len, up_cpu_rows]),
            up_cpu_buf,
            cpu_backend,
        );

        let max_gpu_bytes = total_rows * gate_split_row.max(up_split_row) * 4;
        let merged_bytes = total_rows * ffn_hidden * 4;

        self.partition = Some(PrefillPartitionScratch {
            x_cpu,
            gate_gpu_partial,
            gate_cpu_partial,
            up_gpu_partial,
            up_cpu_partial,
            merge_gpu_temp: vec![0u8; max_gpu_bytes],
            merge_buf: vec![0u8; merged_bytes],
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::galloc::Galloc;

    fn llama_1b_config() -> WorkspaceConfig {
        WorkspaceConfig {
            batch_size: 1,
            dim: 2048,
            q_dim: 2048,
            k_dim: 512,
            v_dim: 512,
            ffn_hidden: 8192,
            n_heads: 32,
            max_seq_len: 2048,
        }
    }

    fn minimal_config() -> WorkspaceConfig {
        WorkspaceConfig {
            batch_size: 1,
            dim: 64,
            q_dim: 64,
            k_dim: 16,
            v_dim: 16,
            ffn_hidden: 128,
            n_heads: 1,
            max_seq_len: 32,
        }
    }

    #[test]
    fn test_workspace_allocation_shapes() {
        let memory = Galloc::new();
        let backend = Arc::new(CpuBackend::new());
        let cfg = llama_1b_config();
        let ws = LayerWorkspace::new(cfg, &memory, backend).unwrap();

        assert_eq!(ws.q.shape().dims(), &[1, 1, 2048]);
        assert_eq!(ws.k.shape().dims(), &[1, 1, 512]);
        assert_eq!(ws.v.shape().dims(), &[1, 1, 512]);
        assert_eq!(ws.out_attn.shape().dims(), &[1, 1, 2048]);
        assert_eq!(ws.gate.shape().dims(), &[1, 1, 8192]);
        assert_eq!(ws.up.shape().dims(), &[1, 1, 8192]);
        assert_eq!(ws.down.shape().dims(), &[1, 1, 2048]);
        assert_eq!(ws.residual.shape().dims(), &[1, 1, 2048]);
        assert_eq!(ws.attn_out.shape().dims(), &[1, 1, 2048]);
    }

    #[test]
    fn test_workspace_scores_size() {
        let memory = Galloc::new();
        let backend = Arc::new(CpuBackend::new());
        let cfg = llama_1b_config();
        let ws = LayerWorkspace::new(cfg, &memory, backend).unwrap();

        // scores = n_heads * max_seq_len = 32 * 2048 = 65536
        assert_eq!(ws.scores.len(), 32 * 2048);
        // All initialized to zero
        assert!(ws.scores.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_workspace_small_config() {
        let memory = Galloc::new();
        let backend = Arc::new(CpuBackend::new());
        let cfg = minimal_config();
        let ws = LayerWorkspace::new(cfg, &memory, backend).unwrap();

        assert_eq!(ws.q.shape().dims(), &[1, 1, 64]);
        assert_eq!(ws.k.shape().dims(), &[1, 1, 16]);
        assert_eq!(ws.gate.shape().dims(), &[1, 1, 128]);
        assert_eq!(ws.scores.len(), 32); // 1 head * 32 max_seq
    }

    #[test]
    fn test_workspace_tensors_writable() {
        let memory = Galloc::new();
        let backend = Arc::new(CpuBackend::new());
        let cfg = minimal_config();
        let ws = LayerWorkspace::new(cfg, &memory, backend).unwrap();

        // All tensors should have non-null writable pointers
        assert!(!ws.q.as_mut_ptr().is_null(), "q pointer is null");
        assert!(!ws.k.as_mut_ptr().is_null(), "k pointer is null");
        assert!(!ws.v.as_mut_ptr().is_null(), "v pointer is null");
        assert!(
            !ws.out_attn.as_mut_ptr().is_null(),
            "out_attn pointer is null"
        );
        assert!(!ws.gate.as_mut_ptr().is_null(), "gate pointer is null");
        assert!(!ws.up.as_mut_ptr().is_null(), "up pointer is null");
        assert!(!ws.down.as_mut_ptr().is_null(), "down pointer is null");
        assert!(
            !ws.residual.as_mut_ptr().is_null(),
            "residual pointer is null"
        );
        assert!(
            !ws.attn_out.as_mut_ptr().is_null(),
            "attn_out pointer is null"
        );

        // Verify buffer sizes are correct (numel * 4 bytes for F32)
        assert_eq!(ws.q.size(), 64 * 4);
        assert_eq!(ws.gate.size(), 128 * 4);
    }
}
