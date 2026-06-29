//! GPU-side attention score accumulator.
//!
//! Maintains persistent GPU buffers for importance scores and runs reduction
//! kernels entirely on the device, eliminating per-token GPU->CPU blocking
//! reads (~129ms/token). CPU readback occurs only at eviction time.
//!
//! # Workflow (fused variant, 2026-04-20)
//!
//! Per decode token:
//! 1. For each layer l, `attention_gen` writes post-softmax scores into the
//!    `[l, :, :]` slice of the per-token `score_buf`
//!    (`layer_offset(l) = l * n_heads_q * score_stride`).
//! 2. At end of token, `end_step()` dispatches a single fused reduce kernel
//!    that iterates all layers, computes MAX-across-layers of both flat and
//!    per-KV-head aggregates, applies exponential decay, and adds into the
//!    cumulative `importance` / `head_importance` buffers directly.
//!
//! This replaces the older per-layer reduce (28 dispatches/token on
//! Qwen2.5-1.5B) + `end_step` + 2×clear (= 31 dispatches) with a single
//! per-token dispatch, eliminating ~500-700 us of Adreno launch overhead.
//!
//! At eviction time:
//! 3. `sync_to_cpu()` does a single blocking readback of cumulative importance.
//! 4. `reset()` clears cumulative buffers after eviction.

use anyhow::Result;
use argus_extension_api::{ScoreReduceArgs, ScoreReduceBackend};

use crate::backend::GpuScoreAccess;

// §13.8-L S-L-2: GpuScoreAccess trait impl — inherent method 위임.
// trait method 가 forward.rs / forward_gen.rs hot path 의 OpenCLBackend
// downcast 2건을 제거하기 위해 정의되었음. OpenCL-specific 메서드
// (`score_buf_mem`, `end_step`, `sync_to_cpu`, `reset`) 는 trait 에서
// 제외 — OpenCL backend 내부 경로 (plan.rs 등) 가 inherent method 로
// 직접 호출.
impl GpuScoreAccess for GpuScoreAccumulator {
    fn is_active(&self) -> bool {
        GpuScoreAccumulator::is_active(self)
    }
    fn set_active(&mut self, active: bool) {
        GpuScoreAccumulator::set_active(self, active)
    }
    fn current_layer_idx(&self) -> usize {
        GpuScoreAccumulator::current_layer_idx(self)
    }
    fn set_current_layer_idx(&mut self, layer_idx: usize) {
        GpuScoreAccumulator::set_current_layer_idx(self, layer_idx)
    }
    fn n_heads_q(&self) -> usize {
        GpuScoreAccumulator::n_heads_q(self)
    }
    fn n_layers(&self) -> usize {
        GpuScoreAccumulator::n_layers(self)
    }
    fn layer_offset_elems(&self, layer_idx: usize) -> usize {
        GpuScoreAccumulator::layer_offset_elems(self, layer_idx)
    }
    fn score_stride(&self) -> usize {
        GpuScoreAccumulator::score_stride(self)
    }
    fn steps_accumulated(&self) -> usize {
        GpuScoreAccumulator::steps_accumulated(self)
    }
}

/// GPU-resident score accumulator that avoids per-token GPU->CPU readback.
pub struct GpuScoreAccumulator {
    /// Persistent GPU buffer for per-layer attention scores output.
    /// Layout: `[n_layers, n_heads_q, score_stride]` where score_stride == max_seq_len.
    /// The attention kernel for layer `l` writes into slice
    /// `score_buf[l * n_heads_q * score_stride .. (l+1) * n_heads_q * score_stride]`.
    score_buf: ocl::core::Mem,

    /// Cumulative flat importance: `[max_seq_len]`.
    /// Grows over decode steps with exponential decay.
    importance: ocl::core::Mem,

    /// Cumulative per-KV-head importance: `[n_kv_heads * max_seq_len]`.
    head_importance: ocl::core::Mem,

    /// Cumulative per-`(layer, token)` FLAT importance: `[n_layers * max_seq_len]` (faithful-H2O
    /// `H2OKVCache_LayerWise`, divergence `(b)`). Updated by the per-layer reduce kernel with NO
    /// cross-layer MAX, so each layer ranks heavy hitters on its OWN attention — the GPU twin of the
    /// CPU producer's `layer_flat_cum`. Always allocated (tiny vs `score_buf`); dispatched/synced only
    /// when `per_layer_flat` is armed.
    layer_flat_importance: ocl::core::Mem,
    /// Whether the per-layer FLAT reduce runs each `end_step` + is synced/seeded (faithful-H2O only).
    per_layer_flat: bool,

    // --- Reduce policy (the GPU score POLICY — per-layer MAX + GQA averaging + forgetting-factor decay —
    // owned by the `attn-score` plugin; the engine core holds no GPU scoring arithmetic). ---
    reducer: Box<dyn ScoreReduceBackend>,

    // --- Config ---
    n_layers: usize,
    n_heads_q: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    /// score_stride for the score_buf layout. Equal to max_seq_len.
    score_stride: usize,
    /// Exponential decay factor: `1.0 - decay`. Applied to cumulative scores each step.
    decay_factor: f32,
    active: bool,
    steps_accumulated: usize,
    /// Current layer index (0..n_layers). Non-plan callers (`forward_gen`)
    /// set this before `attention_gen` so the mod.rs arg binding can write
    /// to the correct layer slice of `score_buf`. The plan path pre-bakes
    /// the layer offset into each layer's kernel args and does not use this.
    current_layer_idx: usize,
}

// SAFETY: GpuScoreAccumulator is only accessed from the inference thread,
// same as KernelCache. Single-threaded access is guaranteed.
unsafe impl Send for GpuScoreAccumulator {}
unsafe impl Sync for GpuScoreAccumulator {}

impl GpuScoreAccumulator {
    /// Create a new GPU score accumulator over an already-resolved reduce policy.
    ///
    /// Allocates the persistent GPU buffers and zero-initializes them. The reduce kernel itself is
    /// owned by `reducer` (the `attn-score` plugin compiled it from the host's borrowed context in
    /// `OpenCLBackend::init_gpu_score_acc`); the engine core holds no GPU scoring arithmetic.
    /// `n_kv_heads > 16` is gated by the caller before the reducer is built.
    ///
    /// `n_layers` controls the per-layer partitioning of `score_buf`. Memory
    /// footprint for Qwen2.5-1.5B (n_layers=28, n_heads_q=12, max_seq=2048):
    /// 28 * 12 * 2048 * 4B = 2.625 MiB — acceptable on Adreno.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        queue: &ocl::core::CommandQueue,
        context: &ocl::core::Context,
        reducer: Box<dyn ScoreReduceBackend>,
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
        let score_buf_size = n_layers * n_heads_q * score_stride;
        let flat_size = max_seq_len;
        let head_size = n_kv_heads * max_seq_len;

        // Allocate GPU buffers
        let score_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context,
                ocl::core::MEM_READ_WRITE,
                score_buf_size,
                None,
            )?
        };

        let importance = unsafe {
            ocl::core::create_buffer::<_, f32>(context, ocl::core::MEM_READ_WRITE, flat_size, None)?
        };

        let head_importance = unsafe {
            ocl::core::create_buffer::<_, f32>(context, ocl::core::MEM_READ_WRITE, head_size, None)?
        };

        // Per-(layer, token) FLAT cumulative buffer (faithful-H2O (b)): [n_layers * max_seq_len].
        let layer_flat_size = n_layers * max_seq_len;
        let layer_flat_importance = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context,
                ocl::core::MEM_READ_WRITE,
                layer_flat_size,
                None,
            )?
        };

        // Zero-initialize cumulative buffers and score buffer (score_buf zero is
        // important because the fused reduce reads every layer; any layer that
        // didn't write for some reason would otherwise contribute stale data
        // from a prior allocation).
        let zeros_flat = vec![0.0f32; flat_size];
        let zeros_head = vec![0.0f32; head_size];
        let zeros_score = vec![0.0f32; score_buf_size];
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                &importance,
                true,
                0,
                &zeros_flat,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                queue,
                &head_importance,
                true,
                0,
                &zeros_head,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                queue,
                &score_buf,
                true,
                0,
                &zeros_score,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                queue,
                &layer_flat_importance,
                true,
                0,
                &vec![0.0f32; layer_flat_size],
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }

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

    /// Get the persistent GPU score buffer for use by `attention_gen`.
    /// The attention kernel writes post-softmax scores here, at the offset
    /// `current_layer_idx * n_heads_q * score_stride`.
    #[inline]
    pub fn score_buf_mem(&self) -> &ocl::core::Mem {
        &self.score_buf
    }

    /// Score stride (= max_seq_len) for the score buffer layout.
    #[inline]
    pub fn score_stride(&self) -> usize {
        self.score_stride
    }

    /// Number of query heads.
    #[inline]
    pub fn n_heads_q(&self) -> usize {
        self.n_heads_q
    }

    /// Number of layers (partition count of `score_buf`).
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Compute the base offset (in f32 elements) for a given layer's slice of
    /// `score_buf`. Callers that pre-bake the offset at plan-build time (plan.rs)
    /// use this; the runtime non-plan path reads `current_layer_idx()` instead.
    #[inline]
    pub fn layer_offset_elems(&self, layer_idx: usize) -> usize {
        layer_idx * self.n_heads_q * self.score_stride
    }

    /// Set the current layer index for the next `attention_gen` call (non-plan
    /// path only; the plan path pre-bakes per-layer offsets into kernel args).
    #[inline]
    pub fn set_current_layer_idx(&mut self, layer_idx: usize) {
        debug_assert!(
            layer_idx < self.n_layers,
            "layer_idx={} exceeds n_layers={}",
            layer_idx,
            self.n_layers
        );
        self.current_layer_idx = layer_idx;
    }

    /// Get the current layer index.
    #[inline]
    pub fn current_layer_idx(&self) -> usize {
        self.current_layer_idx
    }

    /// End the current decode step.
    ///
    /// Dispatches the fused reduce kernel that iterates all layers,
    /// aggregates per-layer scores (flat sum, per-head GQA avg), applies
    /// MAX across layers, and updates cumulative importance + head_importance
    /// with exponential decay — all in a single kernel.
    ///
    /// This replaces the former per-layer `reduce_layer` (28 dispatches) +
    /// `end_step` + 2 × clear (31 total) sequence with one dispatch.
    pub fn end_step(
        &mut self,
        queue: &ocl::core::CommandQueue,
        cache_seq_len: usize,
    ) -> Result<()> {
        if !self.active || cache_seq_len == 0 {
            return Ok(());
        }

        // Lend the engine-owned buffers + the live queue borrow-for-call; the plugin's reducer owns
        // the kernel and the policy (MAX / GQA / decay). All handles borrowed, never released here.
        // `cl_command_queue` / `cl_mem` are `*mut c_void` in this cl-sys; no cast needed.
        let args = ScoreReduceArgs {
            cl_queue: queue.as_ptr(),
            score_buf: self.score_buf.as_ptr(),
            importance: self.importance.as_ptr(),
            head_importance: self.head_importance.as_ptr(),
            decay_factor: self.decay_factor,
            n_layers: self.n_layers,
            n_heads_q: self.n_heads_q,
            n_kv_heads: self.n_kv_heads,
            cache_seq_len,
            score_stride: self.score_stride,
            max_seq_len: self.max_seq_len,
            layer_flat_importance: self.layer_flat_importance.as_ptr(),
        };
        let ret = self.reducer.reduce(&args);
        if ret != 0 {
            anyhow::bail!("attn_score GPU reduce failed: code {ret}");
        }
        // Faithful-H2O (b): ALSO fold this step into the per-(layer, token) FLAT cumulative (no
        // cross-layer MAX), so each layer ranks its own heavy hitters. The collapsed reduce above is
        // left running (harmless, keeps the non-faithful path available); only this extra dispatch is
        // gated. The kernel reads `args.layer_flat_importance`.
        if self.per_layer_flat {
            let ret = self.reducer.reduce_per_layer(&args);
            if ret != 0 {
                anyhow::bail!("attn_score GPU per-layer reduce failed: code {ret}");
            }
        }

        self.steps_accumulated += 1;
        // Advance the layer index counter back to 0 for the next token so the
        // non-plan path (which calls set_current_layer_idx per layer) starts
        // clean. The score_buf contents from this token are now folded into
        // `importance` / `head_importance`; the next token's per-layer writes
        // will simply overwrite the slices (fused reduce reads before any
        // subsequent writes of the same slot).
        self.current_layer_idx = 0;
        Ok(())
    }

    /// Read cumulative importance scores back to CPU.
    ///
    /// Returns `(flat_importance, head_importance)`.
    /// This is the only blocking GPU->CPU read and should be called only at eviction time.
    pub fn sync_to_cpu(&self, queue: &ocl::core::CommandQueue) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut flat = vec![0.0f32; self.max_seq_len];
        let mut head = vec![0.0f32; self.n_kv_heads * self.max_seq_len];

        unsafe {
            ocl::core::enqueue_read_buffer(
                queue,
                &self.importance,
                true,
                0,
                &mut flat,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_read_buffer(
                queue,
                &self.head_importance,
                true,
                0,
                &mut head,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }

        Ok((flat, head))
    }

    /// Seed the cumulative importance buffers with a prefill-attention column-sum (faithful-H2O `(c)`).
    ///
    /// On GPU, the eviction policy ranks on importance that is `import_gpu_scores`'d FROM these buffers
    /// (the pre-eviction sync OVERWRITES the CPU-side importance), so the prefill seed cannot live only
    /// on the CPU accumulator — it would be clobbered. We write the just-folded cumulative directly into
    /// the GPU buffers; the subsequent decode `end_step`s accumulate on top (the fused reduce reads the
    /// existing value, applies decay, then adds — so with the default no-decay config the seed persists).
    ///
    /// Must be called AFTER `new`/`reset` (which zero the buffers) and BEFORE the first decode `end_step`.
    /// `flat.len() == max_seq_len`, `head.len() == n_kv_heads * max_seq_len` (same layout `sync_to_cpu`
    /// reads back and `import_gpu_scores` consumes).
    pub fn seed_cumulative(
        &self,
        queue: &ocl::core::CommandQueue,
        flat: &[f32],
        head: &[f32],
    ) -> Result<()> {
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
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                &self.importance,
                true,
                0,
                flat,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                queue,
                &self.head_importance,
                true,
                0,
                head,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    /// Arm the per-`(layer, token)` FLAT reduce (faithful-H2O `(b)`): from now on each `end_step`
    /// also folds the step into `layer_flat_importance` (no cross-layer MAX), and the seed/sync paths
    /// below become active. Must be called before the first decode `end_step` / seed (mirror of the
    /// CPU `enable_per_layer_flat`).
    pub fn enable_per_layer_flat(&mut self) {
        self.per_layer_flat = true;
    }

    /// Read the cumulative per-`(layer, token)` FLAT importance back to CPU (`[n_layers * max_seq_len]`,
    /// row-major `layer * max_seq_len + pos`) — the GPU twin of the CPU producer's `layer_flat_cum`.
    /// Blocking; called only at eviction time alongside [`sync_to_cpu`](Self::sync_to_cpu).
    pub fn sync_layer_flat_to_cpu(&self, queue: &ocl::core::CommandQueue) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.n_layers * self.max_seq_len];
        unsafe {
            ocl::core::enqueue_read_buffer(
                queue,
                &self.layer_flat_importance,
                true,
                0,
                &mut out,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(out)
    }

    /// Seed the per-`(layer, token)` FLAT cumulative buffer with the prefill column-sums (faithful-H2O
    /// `(b)` + `(c)`): the GPU per-layer twin of [`seed_cumulative`](Self::seed_cumulative). The
    /// subsequent decode per-layer reduces accumulate on top (decay=1.0 default → the seed persists).
    /// `layer_flat.len() == n_layers * max_seq_len` (same layout `sync_layer_flat_to_cpu` reads back).
    pub fn seed_layer_flat_cumulative(
        &self,
        queue: &ocl::core::CommandQueue,
        layer_flat: &[f32],
    ) -> Result<()> {
        let want = self.n_layers * self.max_seq_len;
        if layer_flat.len() != want {
            anyhow::bail!(
                "seed_layer_flat_cumulative: layer_flat.len()={} != n_layers*max_seq_len={}",
                layer_flat.len(),
                want
            );
        }
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                &self.layer_flat_importance,
                true,
                0,
                layer_flat,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    /// Reset ONLY the per-`(layer, token)` FLAT cumulative buffer (faithful-H2O `(b)`), in lockstep
    /// with the CPU accumulator's post-eviction `reset()`. The GPU per-layer reduce accumulates
    /// on-device across decode steps; without this the buffer would monotonically grow since prefill
    /// at physical slots that compaction has shifted, so a 2nd eviction would rank GPU per-layer
    /// importance misaligned with the compacted cache (and diverge from the freshly-reset CPU twin).
    /// Touches NEITHER the collapsed `importance`/`head_importance` (their existing GPU behavior is
    /// unchanged — only the new per-layer buffer is reset) NOR `steps_accumulated`/`current_layer_idx`.
    pub fn reset_layer_flat(&self, queue: &ocl::core::CommandQueue) -> Result<()> {
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                &self.layer_flat_importance,
                true,
                0,
                &vec![0.0f32; self.n_layers * self.max_seq_len],
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    /// Reset cumulative importance (after eviction).
    pub fn reset(&mut self, queue: &ocl::core::CommandQueue) -> Result<()> {
        let flat_size = self.max_seq_len;
        let head_size = self.n_kv_heads * self.max_seq_len;
        let zeros_flat = vec![0.0f32; flat_size];
        let zeros_head = vec![0.0f32; head_size];
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                &self.importance,
                true,
                0,
                &zeros_flat,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                queue,
                &self.head_importance,
                true,
                0,
                &zeros_head,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            // Per-layer FLAT cumulative — cleared in lockstep (faithful-H2O (b)).
            ocl::core::enqueue_write_buffer(
                queue,
                &self.layer_flat_importance,
                true,
                0,
                &vec![0.0f32; self.n_layers * self.max_seq_len],
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        self.steps_accumulated = 0;
        self.current_layer_idx = 0;
        Ok(())
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Number of steps accumulated since last reset.
    #[allow(dead_code)]
    pub fn steps_accumulated(&self) -> usize {
        self.steps_accumulated
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[allow(clippy::float_equality_without_abs)]
    fn test_decay_factor_clamping() {
        // decay=0.0 -> factor=1.0 (no decay)
        assert!((1.0f32 - 0.0f32).clamp(0.0, 1.0) - 1.0 < f32::EPSILON);
        // decay=0.5 -> factor=0.5
        assert!((1.0f32 - 0.5f32).clamp(0.0, 1.0) - 0.5 < f32::EPSILON);
        // decay=1.0 -> factor=0.0 (full decay)
        assert!((1.0f32 - 1.0f32).clamp(0.0, 1.0) < f32::EPSILON);
    }
}
