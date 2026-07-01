//! CUDA GPU-side attention score accumulator — the discrete-GPU twin of
//! [`opencl/gpu_score.rs`](crate::backend::opencl::gpu_score).
//!
//! Maintains persistent VRAM buffers for importance scores and runs the fused
//! reduce kernel entirely on the device, eliminating the per-token GPU→CPU
//! blocking score readback (`attention_gen(scores_out)` → host copy). CPU
//! readback occurs only at eviction time (`sync_to_cpu`).
//!
//! The reduce *policy* (per-layer MAX + GQA averaging + forgetting-factor decay)
//! is owned by the `attn-score` plugin's [`CudaScoreReduceBackend`] — the engine
//! core holds no GPU scoring arithmetic, only the four score buffers. This mirrors
//! the OpenCL accumulator exactly; the only differences are the buffer type
//! (raw VRAM `CudaDeviceBuffer` vs `ocl::core::Mem`) and the transfer calls
//! (`cuMemcpyH/DtoD` vs `enqueue_{read,write}_buffer`).
//!
//! # Workflow (per decode token)
//! 1. For each layer `l`, the neutral forward-gen seam
//!    ([`forward_gen_fmt`](crate::layers::transformer_layer)) calls
//!    `set_current_layer_idx(l)` via the [`GpuScoreAccess`] trait, then
//!    `CudaBackend::attention_gen` binds `score_buf[l, :, :]` into the flash
//!    kernel's `scores_out` arg — the kernel writes post-softmax weights there
//!    (a byproduct of the softmax it already computes).
//! 2. At end of token, [`end_step`](Self::end_step) dispatches the fused reduce
//!    (MAX-across-layers of flat + per-KV-head aggregates, exponential decay,
//!    add into the cumulative buffers) on the engine's stream.
//!
//! At eviction time [`sync_to_cpu`](Self::sync_to_cpu) does the single blocking
//! readback and [`reset`](Self::reset) clears the cumulative buffers.

use std::cell::UnsafeCell;
use std::ffi::c_void;

use anyhow::{Result, anyhow};
use argus_extension_api::{CudaScoreReduceArgs, CudaScoreReduceBackend};

use crate::backend::GpuScoreAccess;
use crate::buffer::DType;
use crate::memory::cuda::buffer::CudaDeviceBuffer;

/// `UnsafeCell<Option<CudaGpuScoreAccumulator>>` newtype that is `Sync` — the
/// CUDA mirror of the OpenCL backend's bare `UnsafeCell` field. `CudaBackend`
/// is `#[derive(Clone)]` over all-`Arc` fields, so the accumulator lives behind
/// `Arc<CudaScoreAccCell>` (clones share one accumulator). Single-threaded
/// inference (INV-018) makes the `&self → &mut` access sound, same as the
/// `KernelCache` / OpenCL `gpu_score_acc` pattern.
pub struct CudaScoreAccCell(UnsafeCell<Option<CudaGpuScoreAccumulator>>);

// SAFETY: only ever touched on the single inference thread.
unsafe impl Send for CudaScoreAccCell {}
unsafe impl Sync for CudaScoreAccCell {}

impl CudaScoreAccCell {
    pub fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    /// # Safety
    /// Caller must guarantee single-threaded access (inference thread only).
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self) -> &mut Option<CudaGpuScoreAccumulator> {
        unsafe { &mut *self.0.get() }
    }
}

impl Default for CudaScoreAccCell {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-resident score accumulator (VRAM) that avoids per-token GPU→CPU readback.
pub struct CudaGpuScoreAccumulator {
    /// Persistent VRAM buffer for per-layer attention scores, layout
    /// `[n_layers, n_heads_q, score_stride]` (score_stride == max_seq_len).
    /// The flash kernel for layer `l` writes into
    /// `score_buf[l * n_heads_q * score_stride ..]`.
    score_buf: CudaDeviceBuffer,
    /// Cumulative flat importance `[max_seq_len]`.
    importance: CudaDeviceBuffer,
    /// Cumulative per-KV-head importance `[n_kv_heads * max_seq_len]`.
    head_importance: CudaDeviceBuffer,
    /// Cumulative per-`(layer, token)` FLAT importance `[n_layers * max_seq_len]`
    /// (faithful-H2O `(b)`; no cross-layer MAX). Always allocated; dispatched/
    /// synced only when `per_layer_flat` is armed.
    layer_flat_importance: CudaDeviceBuffer,
    per_layer_flat: bool,

    /// The GPU reduce POLICY, owned by the `attn-score` plugin.
    reducer: Box<dyn CudaScoreReduceBackend>,

    n_layers: usize,
    n_heads_q: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    score_stride: usize,
    decay_factor: f32,
    active: bool,
    steps_accumulated: usize,
    current_layer_idx: usize,
}

// SAFETY: accessed only from the inference thread (mirror of the OpenCL
// `GpuScoreAccumulator` unsafe Send/Sync).
unsafe impl Send for CudaGpuScoreAccumulator {}
unsafe impl Sync for CudaGpuScoreAccumulator {}

impl GpuScoreAccess for CudaGpuScoreAccumulator {
    fn is_active(&self) -> bool {
        self.active
    }
    fn set_active(&mut self, active: bool) {
        self.active = active;
    }
    fn current_layer_idx(&self) -> usize {
        self.current_layer_idx
    }
    fn set_current_layer_idx(&mut self, layer_idx: usize) {
        debug_assert!(
            layer_idx < self.n_layers,
            "layer_idx={} exceeds n_layers={}",
            layer_idx,
            self.n_layers
        );
        self.current_layer_idx = layer_idx;
    }
    fn n_heads_q(&self) -> usize {
        self.n_heads_q
    }
    fn n_layers(&self) -> usize {
        self.n_layers
    }
    fn layer_offset_elems(&self, layer_idx: usize) -> usize {
        layer_idx * self.n_heads_q * self.score_stride
    }
    fn score_stride(&self) -> usize {
        self.score_stride
    }
    fn steps_accumulated(&self) -> usize {
        self.steps_accumulated
    }
}

impl CudaGpuScoreAccumulator {
    /// Allocate the four persistent VRAM buffers (zero-initialized) over an
    /// already-resolved reduce policy. `n_kv_heads <= 16` is gated by the caller.
    /// Footprint for Qwen2.5-1.5B (n_layers=28, n_heads_q=12, max_seq=2048):
    /// score_buf = 28*12*2048*4B = 2.625 MiB — negligible on a discrete GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        reducer: Box<dyn CudaScoreReduceBackend>,
        n_layers: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        decay: f32,
    ) -> Result<Self> {
        debug_assert!(
            n_kv_heads <= 16,
            "n_kv_heads={n_kv_heads} exceeds the fused kernel limit of 16 (caller must gate)"
        );

        let score_stride = max_seq_len;

        // CudaDeviceBuffer::new does NOT zero-initialize (unlike CudaBuffer /
        // CudaHostBuffer), so zero every buffer explicitly — score_buf zero
        // matters because the fused reduce reads every layer; any layer that
        // didn't write would otherwise contribute stale VRAM contents.
        let alloc_zeroed = |elems: usize| -> Result<CudaDeviceBuffer> {
            let bytes = elems
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| anyhow!("score buffer size overflow ({elems} elems)"))?;
            let buf = CudaDeviceBuffer::new(bytes, DType::F32)?;
            let zeros = vec![0u8; bytes];
            buf.copy_from_host(zeros.as_ptr(), bytes)?;
            Ok(buf)
        };

        let score_buf = alloc_zeroed(n_layers * n_heads_q * score_stride)?;
        let importance = alloc_zeroed(max_seq_len)?;
        let head_importance = alloc_zeroed(n_kv_heads * max_seq_len)?;
        let layer_flat_importance = alloc_zeroed(n_layers * max_seq_len)?;

        Ok(Self {
            score_buf,
            importance,
            head_importance,
            layer_flat_importance,
            per_layer_flat: false,
            reducer,
            n_layers,
            n_heads_q,
            n_kv_heads,
            max_seq_len,
            score_stride,
            decay_factor: (1.0 - decay).clamp(0.0, 1.0),
            active: false,
            steps_accumulated: 0,
            current_layer_idx: 0,
        })
    }

    /// Raw `CUdeviceptr` (u64) of the persistent score buffer. `attention_gen`
    /// adds `layer_offset_elems(current_layer_idx) * 4` bytes to reach the
    /// current layer's slice.
    #[inline]
    pub fn score_buf_device_ptr(&self) -> u64 {
        self.score_buf.device_ptr()
    }

    /// Score stride (== max_seq_len) in f32 elements.
    #[inline]
    pub fn score_stride(&self) -> usize {
        self.score_stride
    }

    /// Base offset (f32 elements) of a layer's slice of `score_buf`.
    #[inline]
    pub fn layer_offset_elems(&self, layer_idx: usize) -> usize {
        layer_idx * self.n_heads_q * self.score_stride
    }

    /// Current layer index (set per layer by the neutral forward-gen seam).
    #[inline]
    pub fn current_layer_idx(&self) -> usize {
        self.current_layer_idx
    }

    /// Whether this accumulator is active (mirrors the trait method for concrete callers).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Dispatch the fused reduce over `score_buf` on the engine's stream.
    ///
    /// `cu_stream` is the raw `CUstream` (from `CudaStream::cu_stream()`), the
    /// same stream the attention kernels ran on — so stream ordering guarantees
    /// the reduce sees every layer's writes without an explicit sync.
    pub fn end_step(&mut self, cu_stream: *mut c_void, cache_seq_len: usize) -> Result<()> {
        if !self.active || cache_seq_len == 0 {
            return Ok(());
        }
        let args = CudaScoreReduceArgs {
            cu_stream,
            score_buf: self.score_buf.device_ptr(),
            importance: self.importance.device_ptr(),
            head_importance: self.head_importance.device_ptr(),
            decay_factor: self.decay_factor,
            n_layers: self.n_layers,
            n_heads_q: self.n_heads_q,
            n_kv_heads: self.n_kv_heads,
            cache_seq_len,
            score_stride: self.score_stride,
            max_seq_len: self.max_seq_len,
            layer_flat_importance: self.layer_flat_importance.device_ptr(),
        };
        let ret = self.reducer.reduce(&args);
        if ret != 0 {
            anyhow::bail!("attn_score CUDA reduce failed: code {ret}");
        }
        // Faithful-H2O (b): ALSO fold into the per-(layer, token) FLAT cumulative
        // (no cross-layer MAX). Gated on `per_layer_flat`; the kernel reads
        // `args.layer_flat_importance`.
        if self.per_layer_flat {
            let ret = self.reducer.reduce_per_layer(&args);
            if ret != 0 {
                anyhow::bail!("attn_score CUDA per-layer reduce failed: code {ret}");
            }
        }

        self.steps_accumulated += 1;
        self.current_layer_idx = 0;
        Ok(())
    }

    /// Blocking readback of the cumulative importance buffers. Returns
    /// `(flat_importance, head_importance)`. Callers MUST have synchronized the
    /// engine stream first (the reduce runs on a non-null stream, while
    /// `cuMemcpyDtoH` uses the null stream, so they are not implicitly ordered).
    pub fn sync_to_cpu(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut flat = vec![0.0f32; self.max_seq_len];
        let mut head = vec![0.0f32; self.n_kv_heads * self.max_seq_len];
        self.importance
            .copy_to_host(flat.as_mut_ptr() as *mut u8, self.max_seq_len * 4)?;
        self.head_importance.copy_to_host(
            head.as_mut_ptr() as *mut u8,
            self.n_kv_heads * self.max_seq_len * 4,
        )?;
        Ok((flat, head))
    }

    /// Seed the cumulative importance buffers with a prefill column-sum
    /// (faithful-H2O `(c)`) — the GPU twin of the CPU seed. The subsequent
    /// decode reduces accumulate on top. Must be called after `new`/`reset`.
    pub fn seed_cumulative(&self, flat: &[f32], head: &[f32]) -> Result<()> {
        if flat.len() != self.max_seq_len {
            anyhow::bail!(
                "seed_cumulative: flat.len()={} != max_seq_len={}",
                flat.len(),
                self.max_seq_len
            );
        }
        if head.len() != self.n_kv_heads * self.max_seq_len {
            anyhow::bail!(
                "seed_cumulative: head.len()={} != n_kv_heads*max_seq_len={}",
                head.len(),
                self.n_kv_heads * self.max_seq_len
            );
        }
        self.importance
            .copy_from_host(flat.as_ptr() as *const u8, flat.len() * 4)?;
        self.head_importance
            .copy_from_host(head.as_ptr() as *const u8, head.len() * 4)?;
        Ok(())
    }

    /// Arm the per-`(layer, token)` FLAT reduce (faithful-H2O `(b)`). Must be
    /// called before the first decode `end_step` / seed.
    pub fn enable_per_layer_flat(&mut self) {
        self.per_layer_flat = true;
    }

    /// Blocking readback of the per-`(layer, token)` FLAT cumulative
    /// (`[n_layers * max_seq_len]`, row-major `layer * max_seq_len + pos`).
    pub fn sync_layer_flat_to_cpu(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.n_layers * self.max_seq_len];
        self.layer_flat_importance.copy_to_host(
            out.as_mut_ptr() as *mut u8,
            self.n_layers * self.max_seq_len * 4,
        )?;
        Ok(out)
    }

    /// Seed the per-`(layer, token)` FLAT cumulative with the prefill column-sums
    /// (faithful-H2O `(b)` + `(c)`). `layer_flat.len() == n_layers * max_seq_len`.
    pub fn seed_layer_flat_cumulative(&self, layer_flat: &[f32]) -> Result<()> {
        let want = self.n_layers * self.max_seq_len;
        if layer_flat.len() != want {
            anyhow::bail!(
                "seed_layer_flat_cumulative: layer_flat.len()={} != n_layers*max_seq_len={}",
                layer_flat.len(),
                want
            );
        }
        self.layer_flat_importance
            .copy_from_host(layer_flat.as_ptr() as *const u8, layer_flat.len() * 4)?;
        Ok(())
    }

    /// Reset ONLY the per-`(layer, token)` FLAT cumulative buffer (faithful-H2O
    /// `(b)`), in lockstep with the CPU accumulator's post-eviction `reset()`.
    /// Touches neither the collapsed buffers nor the step counters.
    pub fn reset_layer_flat(&self) -> Result<()> {
        let bytes = self.n_layers * self.max_seq_len * 4;
        let zeros = vec![0u8; bytes];
        self.layer_flat_importance
            .copy_from_host(zeros.as_ptr(), bytes)
    }

    /// Reset the cumulative importance buffers (after eviction).
    pub fn reset(&mut self) -> Result<()> {
        let flat_bytes = self.max_seq_len * 4;
        let head_bytes = self.n_kv_heads * self.max_seq_len * 4;
        let layer_flat_bytes = self.n_layers * self.max_seq_len * 4;
        self.importance
            .copy_from_host(vec![0u8; flat_bytes].as_ptr(), flat_bytes)?;
        self.head_importance
            .copy_from_host(vec![0u8; head_bytes].as_ptr(), head_bytes)?;
        self.layer_flat_importance
            .copy_from_host(vec![0u8; layer_flat_bytes].as_ptr(), layer_flat_bytes)?;
        self.steps_accumulated = 0;
        self.current_layer_idx = 0;
        Ok(())
    }

    /// Steps accumulated since last reset.
    #[allow(dead_code)]
    pub fn steps_accumulated(&self) -> usize {
        self.steps_accumulated
    }

    /// Test-only: overwrite the entire `score_buf` from host data
    /// (`[n_layers, n_heads_q, score_stride]`, row-major). Used by the CUDA
    /// device test binary (`test_cuda_gpu_score_device`) to drive the reduce
    /// with controlled scores instead of a live attention pass.
    #[doc(hidden)]
    pub fn debug_fill_score_buf(&self, scores: &[f32]) -> Result<()> {
        let want = self.n_layers * self.n_heads_q * self.score_stride;
        if scores.len() != want {
            anyhow::bail!(
                "debug_fill_score_buf: scores.len()={} != n_layers*n_heads_q*score_stride={}",
                scores.len(),
                want
            );
        }
        self.score_buf
            .copy_from_host(scores.as_ptr() as *const u8, scores.len() * 4)
    }
}
