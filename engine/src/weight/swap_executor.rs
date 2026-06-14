//! `SwapExecutor` — Phase 2 runtime layer swap (ENG-ALG-211).
//!
//! The executor materialises a fresh `LayerWeights` snapshot from the
//! secondary mmap for each target decoder layer and installs it into the
//! corresponding `LayerSlot` via `ArcSwap::store`. A single
//! `TransformerModel::ratio_generation` bump at the end of the batch
//! triggers plan invalidation (INV-120) exactly once (ENG-ALG-211 step (e)).
//!
//! **Scope (Stage 1)**: uniform, index-based target selection only — QCF
//! importance wiring lands in Phase 3 with `SwapDecider`.
//!
//! **Safety**: the executor is intended to run single-threaded relative to
//! other writers. Readers (forward pass) acquire `Arc<LayerWeights>`
//! snapshots on token entry (ENG-ALG-214-SNAP) and are therefore unaffected
//! by concurrent swaps — INV-121/INV-123.
//!
//! Spec: ENG-ALG-211, ENG-DAT-092/093/094, INV-120/121/123/124/125.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Measurement-only override (`LLMRS_SWAP_FORCE_EVERY_TICK=1`): skip the
/// INV-141 release_worker drain at `execute_on_slots` entry so the incremental
/// swap orchestrator fires every decode token at the user-requested K rate.
///
/// Resolved once per process. Logs a single warning to stderr on first read
/// when enabled. **Not production-safe** — bypassing the drain can let displaced
/// primary cl_mem accumulate on a slow release path, growing peak memory.
fn force_every_tick_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("LLMRS_SWAP_FORCE_EVERY_TICK")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// S-1+β: process-wide guard so caller emits the `force_every_tick`
/// `ConfigWarning` exactly once across the entire run, even when multiple
/// SwapExecutor instances re-check the env flag.
fn force_every_tick_warning_consumed() -> bool {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.set(()).is_err()
}

use anyhow::Result;
use argus_shared::DtypeTag;

use crate::backend::{Backend, GpuEvent};
use crate::buffer::{Buffer, DType};
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference layer type (TransformerLayer/WeightStagingPool)
use crate::layers::transformer_layer::TransformerLayer;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference staging pool trait
#[cfg(feature = "cuda-embedded")]
use crate::layers::staging_pool::WeightStagingPool;
use crate::memory::Memory;
use crate::memory::host::shared::SharedBuffer;
use crate::model_config::ModelConfig;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference loader/model type
use crate::models::loader::gguf::{qk_permute_shape, unpermute_qk_rows};
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference TransformerModel
use crate::models::transformer::TransformerModel;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference weight resource (LayerSlot/SecondaryMmap/SecondaryTensorInfo)
use crate::models::weights::{LayerSlot, LayerWeights, SecondaryMmap};
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference weight resource (SecondaryTensorInfo, opencl path)
#[cfg(feature = "opencl")]
use crate::models::weights::SecondaryTensorInfo;
use crate::tensor::Tensor;
use crate::weight::async_swap::AsyncSwapDispatcher;

/// Errors surfaced by `SwapExecutor::execute`.
#[derive(Debug)]
pub enum SwapError {
    /// A tensor expected by the secondary mmap index is missing.
    SecondaryTensorMissing { layer: usize, subname: String },
    /// Backend refused to allocate a host-side weight tensor.
    BufferAllocationFailed { layer: usize, source: anyhow::Error },
    /// Shape inferred from secondary dims does not match the primary layer.
    ShapeMismatch {
        layer: usize,
        subname: String,
        primary: Vec<usize>,
        secondary: Vec<u64>,
    },
    /// Wire-protocol `DtypeTag` that has no engine-internal `DType` mapping
    /// (reserved variants — INV-126). The variant name is included for
    /// diagnostics.
    UnsupportedDtype(DtypeTag),
    /// INV-141: `PrimaryReleaseWorker` still has pending drop jobs from the
    /// previous batch when the next batch attempts to start. Swap is rejected
    /// to prevent memory leak accumulation.
    ReleaseDrainTimeout { pending: usize, timeout_ms: u64 },
    /// INV-142: `backend.synchronize()` at the stage gate failed.
    ///
    /// This gate ensures all async `enqueue_write_buffer` calls (ENG-ALG-230)
    /// have completed before `invalidate_noshuffle_soa_registry` and the
    /// `ratio_generation` bump execute (ENG-ALG-231).
    StageGateSyncFailed { source: anyhow::Error },
    /// LISWAP-2 async path: `backend.enqueue_write_async` returned an error
    /// or `supports_async_transfer()` returned `false` despite the caller
    /// requesting the async path. Caller may retry the layer via sync path.
    AsyncTransferUnavailable { layer: usize, source: anyhow::Error },
    /// LISWAP-2 async path: `dispatcher.submit_commit` failed (channel closed).
    AsyncDispatchFailed { layer: usize, source: anyhow::Error },
}

impl std::fmt::Display for SwapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwapError::SecondaryTensorMissing { layer, subname } => write!(
                f,
                "secondary mmap: layer {layer} missing tensor '{subname}'"
            ),
            SwapError::BufferAllocationFailed { layer, source } => {
                write!(f, "swap layer {layer}: buffer allocation failed: {source}")
            }
            SwapError::ShapeMismatch {
                layer,
                subname,
                primary,
                secondary,
            } => write!(
                f,
                "swap layer {layer} tensor '{subname}': \
                 shape mismatch (primary={primary:?}, secondary={secondary:?})"
            ),
            SwapError::UnsupportedDtype(tag) => {
                write!(
                    f,
                    "unsupported target dtype: {tag:?} (INV-126 reserved variant)"
                )
            }
            SwapError::ReleaseDrainTimeout {
                pending,
                timeout_ms,
            } => write!(
                f,
                "primary release worker drain timeout: {pending} jobs remaining after {timeout_ms}ms"
            ),
            SwapError::StageGateSyncFailed { source } => {
                write!(f, "backend synchronize failed at stage gate: {source}")
            }
            SwapError::AsyncTransferUnavailable { layer, source } => {
                write!(f, "async transfer unavailable for layer {layer}: {source}")
            }
            SwapError::AsyncDispatchFailed { layer, source } => {
                write!(
                    f,
                    "async dispatch submit failed for layer {layer}: {source}"
                )
            }
        }
    }
}

impl std::error::Error for SwapError {}

/// Map a wire-protocol [`DtypeTag`] to the engine-internal [`DType`] used by
/// `SwapExecutor`. Returns `Err(SwapError::UnsupportedDtype)` for reserved
/// variants that are not yet executable (INV-126).
///
/// Only `DtypeTag::Q4_0` is currently mapped; all other variants are reserved
/// for forward-compat and will be rejected until engine support lands.
pub fn dtype_tag_to_dtype(tag: DtypeTag) -> Result<DType, SwapError> {
    match tag {
        DtypeTag::Q4_0 => Ok(DType::Q4_0),
        other => Err(SwapError::UnsupportedDtype(other)),
    }
}

/// `Result` returned by an async build of one target layer:
/// `(materialised TransformerLayer, fence GPU event)` on success.
type AsyncLayerBuild = Result<(TransformerLayer, GpuEvent), SwapError>;

/// One element of the per-batch async build collection: layer index paired
/// with its build result.
type AsyncLayerBuildPair = (usize, AsyncLayerBuild);

/// Outcome of a single layer swap inside one batch.
#[derive(Debug, Clone)]
pub struct SwappedLayer {
    pub layer_idx: usize,
    pub from_dtype: DType,
    pub to_dtype: DType,
}

/// Per-stage accumulated timings for one `execute` batch.
///
/// Collected via `Instant::now()` pairs at stage boundaries — µs-level
/// overhead, negligible relative to the ms-scale operations measured.
///
/// Stage mapping (WSWAP-4-LATENCY-β + WSWAP-5-COLD-UNIFORM):
/// - `prefault_ms` — (a-pre) WSWAP-5 batch-once `madvise(MADV_WILLNEED)` +
///   page-touch warmup over the secondary mmap weights region. Charged once
///   per `execute_on_slots` call (not per layer) so cold/warm runs are
///   directly comparable.
/// - `mmap_permute_ms` — (a) secondary slice + Q/K permutation + GPU upload
///   (`build_layer_from_mmap`, per-layer accumulated)
/// - `arc_swap_ms` — (b) `LayerSlot::swap_weights` atomic Arc store
/// - `madvise_ms` — (c) `madvise(MADV_DONTNEED)` on replaced pages
/// - `synchronize_ms` — stage gate: `backend.synchronize()` (`clFinish`)
///   called once after all materialise calls, before SOA rebuild (ENG-ALG-231)
/// - `soa_reconvert_ms` — (d) `ensure_noshuffle_soa_registered` loop
///   (Phase 3.7a, hypothesis: dominant bottleneck)
/// - `gen_bump_ms` — (e) `invalidate_noshuffle_soa_registry` +
///   `ratio_generation.fetch_add`
#[derive(Debug, Clone, Default)]
pub struct StageBreakdown {
    /// (a-pre) WSWAP-5 prefault: `madvise(MADV_WILLNEED)` + page-touch warmup.
    /// Single batch-level cost, NOT per layer. Stays at 0.0 on non-Linux
    /// targets / on no-op secondaries (CPU host tests).
    pub prefault_ms: f64,
    /// (a) secondary mmap slice + Q/K permutation + GPU copy_weight_from
    pub mmap_permute_ms: f64,
    /// (b) `LayerSlot::swap_weights` Arc store (per-layer accumulated)
    pub arc_swap_ms: f64,
    /// (c) `madvise(MADV_DONTNEED)` (per-layer accumulated)
    pub madvise_ms: f64,
    /// Stage gate: `backend.synchronize()` (`clFinish`) called once after all
    /// `materialise_weight` / `enqueue_write_buffer` calls, before
    /// `invalidate_noshuffle_soa_registry` (INV-142 / ENG-ALG-231).
    /// Near-zero on CPU/CUDA backends (trait default is `Ok(())`).
    pub synchronize_ms: f64,
    /// (d) `ensure_noshuffle_soa_registered` SOA re-conversion loop
    pub soa_reconvert_ms: f64,
    /// (e) `invalidate_noshuffle_soa_registry` + `ratio_generation` bump
    pub gen_bump_ms: f64,
}

impl StageBreakdown {
    /// Format as a compact single-line string for stderr diagnostics.
    ///
    /// Example:
    /// `prefault=12.3ms mmap_permute=123.4ms arc_swap=0.1ms madvise=0.2ms synchronize=0.5ms soa_reconvert=45.6ms gen_bump=0.1ms`
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path stage log formatter
    pub fn to_log_line(&self) -> String {
        format!(
            "prefault={:.1}ms mmap_permute={:.1}ms arc_swap={:.1}ms madvise={:.1}ms \
             synchronize={:.1}ms soa_reconvert={:.1}ms gen_bump={:.1}ms",
            self.prefault_ms,
            self.mmap_permute_ms,
            self.arc_swap_ms,
            self.madvise_ms,
            self.synchronize_ms,
            self.soa_reconvert_ms,
            self.gen_bump_ms,
        )
    }
}

/// Aggregated result of one `execute` batch (ENG-ALG-211 "SwapResult").
#[derive(Debug, Clone, Default)]
pub struct SwapReport {
    /// Layers whose snapshot was atomically replaced.
    pub swapped: Vec<SwappedLayer>,
    /// Layers we intended to swap but skipped (already at target dtype or
    /// outside the range / missing secondary handle).
    pub skipped: Vec<usize>,
    /// Wall clock for the full batch.
    pub latency_ms: f64,
    /// Value of `TransformerModel::ratio_generation` after the batch. `None`
    /// when no actual swap happened (no bump — ENG-ALG-211 step (e) contract).
    pub ratio_generation_after: Option<u64>,
    /// Per-stage timing breakdown (WSWAP-4-LATENCY-β).
    /// Always populated on a successful batch with at least one swapped layer;
    /// `None` when the entire batch was a no-op (no secondary mmap).
    pub stage_breakdown: Option<StageBreakdown>,
}

/// Stateless executor. One instance per swap request is fine.
pub struct SwapExecutor<'a> {
    /// Target dtype for all layers in this batch. Fixed per batch because
    /// ratio-based swap conceptually picks one secondary dtype per batch.
    pub target_dtype: DType,
    /// Model config — drives Q/K permutation gating (only Llama rearranges
    /// at load time, mirroring `gguf.rs`).
    pub config: &'a ModelConfig,
    /// Where we land the fresh weight tensors (CPU for host tests, the
    /// primary backend in production). Must match the existing layer's
    /// backend so downstream kernels remain compatible.
    pub backend: Arc<dyn Backend>,
    /// Memory allocator paired with `backend`. Used for the permutation
    /// fallback path when we have to materialise an owned buffer.
    pub memory: &'a dyn Memory,
    /// Optional async release worker (ENG-ALG-228 / ENG-DAT-100).
    ///
    /// When `Some`, Stage (c) enqueues displaced `LayerWeights` here instead
    /// of dropping inline. INV-141 is verified at the top of each batch.
    /// `None` → original inline drop path (host tests, CPU backend fallback).
    pub release_worker: Option<Arc<dyn crate::runtime_resources_access::ReleaseWorkerAccess>>,
    /// LISWAP-3 prototype: opt-in `CL_MEM_ALLOC_HOST_PTR` pool path.
    ///
    /// When `Some`, GPU AOS weight materialisation routes through the pool
    /// (zero-copy `map / memcpy / unmap` on a pre-allocated slot) instead
    /// of the staging `copy_weight_from` cycle. Slot exhaustion / size
    /// overflow falls back gracefully to the staging path. AUF SOA bypass
    /// path is unchanged — that path is already zero-copy by construction.
    /// `None` (default) → behaviour identical to the pre-Stage-3 baseline.
    /// Plan: `compiled-chasing-hopper.md` Direction A track, Stage 3.
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path pool field
    #[cfg(feature = "opencl")]
    pub host_ptr_pool: Option<Arc<crate::backend::opencl::host_ptr_pool::HostPtrPool>>,

    /// LISWAP-8 Phase B: pre-allocated layer object pool with background
    /// re-supply. When `Some` and `LLMRS_SWAP_LAYER_POOL=1` is set, the
    /// BG fetch dispatch path takes an entry from the pool instead of
    /// calling `cuMemAlloc` for each weight buffer. The background
    /// allocator thread refills the pool to keep the depth stable.
    /// Falls back to the regular bg_fetch path if the pool is exhausted.
    ///
    /// Stored as `Arc<dyn WeightStagingPool>` (Migration Step 3-B DIP).
    #[cfg(feature = "cuda-embedded")]
    pub layer_pool: Option<Arc<dyn WeightStagingPool>>,

    /// LISWAP-8 Hammer D: registered mmap region for alias-only swap.
    /// When `Some` and `LLMRS_SWAP_MMAP_ALIAS=1` is set, weights are
    /// installed as zero-copy aliases into the registered mmap region
    /// (no `cuMemAlloc`, no `cuMemcpyHtoDAsync`, no CPU memcpy). Q/K
    /// tensors that need runtime unpermute fall back to the bg_fetch
    /// build path.
    #[cfg(feature = "cuda-embedded")]
    pub mmap_registration: Option<Arc<crate::memory::cuda::mmap::CudaMmapRegistration>>,

    /// Dispatcher identity label used in log lines. Defaults to
    /// `"IntraForward"`. PhaseAware / Subsystem / Incremental caller 는
    /// `.with_kind("...")` chain 으로 override.
    pub kind: &'static str,
}

impl<'a> SwapExecutor<'a> {
    /// Construct an executor. `target_dtype` is the dtype we are swapping
    /// *to* (e.g. `Q4_0` for F16→Q4_0). Backend+memory must be consistent
    /// with the slots we will touch.
    pub fn new(
        target_dtype: DType,
        config: &'a ModelConfig,
        backend: Arc<dyn Backend>,
        memory: &'a dyn Memory,
    ) -> Self {
        Self {
            target_dtype,
            config,
            backend,
            memory,
            release_worker: None,
            #[cfg(feature = "opencl")]
            host_ptr_pool: None,
            #[cfg(feature = "cuda-embedded")]
            layer_pool: None,
            #[cfg(feature = "cuda-embedded")]
            mmap_registration: None,
            kind: "IntraForward",
        }
    }

    /// Construct an executor with an async release worker attached (ENG-ALG-228).
    ///
    /// Use this constructor in production paths where the `TransformerModel`
    /// owns a `PrimaryReleaseWorker`. Stage (c) will enqueue displaced layers
    /// to the worker instead of dropping inline.
    pub fn new_with_worker(
        target_dtype: DType,
        config: &'a ModelConfig,
        backend: Arc<dyn Backend>,
        memory: &'a dyn Memory,
        release_worker: Arc<dyn crate::runtime_resources_access::ReleaseWorkerAccess>,
    ) -> Self {
        Self {
            target_dtype,
            config,
            backend,
            memory,
            release_worker: Some(release_worker),
            #[cfg(feature = "opencl")]
            host_ptr_pool: None,
            #[cfg(feature = "cuda-embedded")]
            layer_pool: None,
            #[cfg(feature = "cuda-embedded")]
            mmap_registration: None,
            kind: "IntraForward",
        }
    }

    /// Override dispatcher identity used in log lines. PhaseAware /
    /// Subsystem / Incremental caller만 chain; default ("IntraForward")
    /// 사용 시 생략. expected values: "IntraForward" / "PhaseAware" /
    /// "Subsystem" / "Incremental".
    pub fn with_kind(mut self, kind: &'static str) -> Self {
        self.kind = kind;
        self
    }

    /// LISWAP-3 prototype builder — attach a `HostPtrPool` so the AOS
    /// materialisation path uses the zero-copy slot pool instead of the
    /// staging `copy_weight_from` cycle. Plan: `compiled-chasing-hopper.md`
    /// Direction A track, Stage 3. Setting this on a CPU executor is a
    /// no-op (the AOS path skips the pool when `is_cpu`).
    #[cfg(feature = "opencl")]
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path pool injection
    pub fn with_host_ptr_pool(
        mut self,
        pool: Arc<crate::backend::opencl::host_ptr_pool::HostPtrPool>,
    ) -> Self {
        self.host_ptr_pool = Some(pool);
        self
    }

    /// LISWAP-8 Phase B: attach a `WeightStagingPool` so the BG fetch path
    /// reuses pre-allocated layer buffers instead of calling `cuMemAlloc`
    /// on the dispatch thread. Activated by `LLMRS_SWAP_LAYER_POOL=1` env.
    #[cfg(feature = "cuda-embedded")]
    pub fn with_layer_pool(mut self, pool: Arc<dyn WeightStagingPool>) -> Self {
        self.layer_pool = Some(pool);
        self
    }

    /// LISWAP-8 Hammer D: attach a `CudaMmapRegistration` so the BG
    /// fetch path installs zero-copy alias buffers for swap weights.
    /// Activated by `LLMRS_SWAP_MMAP_ALIAS=1` env.
    #[cfg(feature = "cuda-embedded")]
    pub fn with_mmap_registration(
        mut self,
        registration: Arc<crate::memory::cuda::mmap::CudaMmapRegistration>,
    ) -> Self {
        self.mmap_registration = Some(registration);
        self
    }

    /// Select uniform index-spaced target layers (Stage 1 fallback, ENG-ALG-212).
    ///
    /// Deterministic: `ceil(ratio * num_layers)` layers evenly spaced across
    /// `[0, num_layers)`. Layers already at `target_dtype` are skipped by
    /// `execute` itself (idempotent).
    pub fn uniform_target_layers(ratio: f32, num_layers: usize) -> Vec<usize> {
        if num_layers == 0 {
            return Vec::new();
        }
        let clamped = ratio.clamp(0.0, 1.0);
        let count = ((clamped * num_layers as f32).round() as usize).min(num_layers);
        if count == 0 {
            return Vec::new();
        }
        if count == num_layers {
            return (0..num_layers).collect();
        }
        // Evenly spaced indices. For count==k and n layers, we use
        // floor((i + 0.5) * n / k) which stays stable under tie breaks.
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let idx = ((i as f32 + 0.5) * num_layers as f32 / count as f32) as usize;
            out.push(idx.min(num_layers - 1));
        }
        out.dedup();
        out
    }

    /// Execute a swap batch. `target_layers` is the caller-chosen set of
    /// decoder indices to convert. Out-of-range / already-swapped layers
    /// are silently skipped (ENG-DAT-C08, ENG-ALG-211).
    ///
    /// Returns `SwapReport` describing what landed. On success,
    /// `TransformerModel::ratio_generation` is bumped **exactly once** when
    /// at least one layer was actually swapped (ENG-ALG-211 step (e)).
    pub fn execute(
        &self,
        model: &TransformerModel,
        target_layers: &[usize],
    ) -> Result<SwapReport, SwapError> {
        self.execute_on_slots(
            model.layers.as_slice(),
            model.secondary_mmap.as_ref(),
            &model.ratio_generation,
            target_layers,
            None,
        )
    }

    /// Low-level execute that accepts raw slot/mmap/generation references.
    ///
    /// Called by `WeightSwapHandler` which holds these fields directly rather
    /// than through a `&TransformerModel`. The semantics are identical to
    /// `execute()`: ENG-ALG-211 batch swap with a single `ratio_generation`
    /// bump at the end.
    ///
    /// `async_dispatcher`: when `Some` **and** `backend.supports_async_transfer()`
    /// returns `true`, the async path is taken for each layer:
    /// - `build_layer_from_mmap_async` enqueues H2D writes on the transfer
    ///   queue/stream and returns `(LayerWeights, GpuEvent)`.
    /// - The `(slot, new_weights, event)` triplet is submitted to the
    ///   dispatcher instead of committing inline.
    /// - Stage gate `backend.synchronize()` (INV-142) is **skipped** — event
    ///   gating inside the worker thread replaces it.
    ///
    /// When `None` (or `supports_async_transfer()` returns `false`), the
    /// original synchronous path runs unchanged (backward-compatible).
    pub fn execute_on_slots(
        &self,
        layers: &[Arc<LayerSlot>],
        secondary_mmap: Option<&Arc<SecondaryMmap>>,
        ratio_generation: &Arc<std::sync::atomic::AtomicU64>,
        target_layers: &[usize],
        async_dispatcher: Option<&AsyncSwapDispatcher>,
    ) -> Result<SwapReport, SwapError> {
        let start = Instant::now();
        let mut report = SwapReport::default();

        // ── INV-141: drain previous batch's pending releases before starting ──
        // Ensures no primary cl_mem from batch N is still being dropped when
        // batch N+1 tries to swap. If drain times out, reject the batch to
        // prevent accumulating memory leaks.
        //
        // Measurement override (EuroSys 2027 §4.2): `LLMRS_SWAP_FORCE_EVERY_TICK=1`
        // skips this drain so `--swap-incremental-per-tick K` fires every decode
        // token at the user-requested rate instead of throttling on the previous
        // batch's release backlog. Memory-spike risk — measurement only.
        let force_every_tick = force_every_tick_enabled();
        if force_every_tick && !force_every_tick_warning_consumed() {
            crate::action_diag_helper::log_swap_config_warning(
                "force_every_tick",
                "LLMRS_SWAP_FORCE_EVERY_TICK=1 enabled — INV-141 release drain \
                 skipped, memory spike risk, measurement-only",
            );
        }
        if !force_every_tick && let Some(worker) = &self.release_worker {
            let pending = worker.pending_count();
            if pending > 0 {
                match worker.drain(Duration::from_millis(50)) {
                    Ok(()) => {}
                    Err(e) => {
                        return Err(SwapError::ReleaseDrainTimeout {
                            pending: e.pending,
                            timeout_ms: e.timeout_ms,
                        });
                    }
                }
            }
        }

        // ENG-DAT-C09: secondary handle absent → entire operation is a no-op.
        let Some(secondary) = secondary_mmap else {
            report.latency_ms = start.elapsed().as_secs_f64() * 1e3;
            return Ok(report);
        };

        // WSWAP-4-LATENCY-β: per-stage accumulated timings.
        let mut stages = StageBreakdown::default();

        // ── Stage (a-pre): WSWAP-5-COLD-UNIFORM prefault ─────────────────────
        // Issue `madvise(MADV_WILLNEED)` + explicit page-touch warmup over the
        // secondary mmap weights region exactly once per batch — before the
        // per-layer copy loop. The cost (a few ms for ~700 MB on Adreno UFS)
        // is paid up-front so subsequent `mmap_permute` reads always hit the
        // page cache. Without this, the 5th measurement showed a per-run
        // bimodal split (cold ~53 ms / warm ~36 ms per layer) driven entirely
        // by mmap demand-paging on cold runs.
        //
        // Charged as a single batch-level stage (NOT per layer) so cold and
        // warm runs are directly comparable. No-op on non-Linux targets.
        //
        // ENG-ALG-229: use `prefault_layers(target_layers)` instead of the
        // broad `prefault()` so only the weight pages for layers that will
        // actually be swapped are touched. This avoids reading ~40 ms worth
        // of pages for layers outside the swap batch at ratio < 1.0.
        //
        // LISWAP-6 / 2026-05-14: skip the batch prefault for the rpcmem alias
        // path. The alias path's first-touch cost is `ensure_layer_loaded`
        // (rpcmem alloc + memcpy from mmap), which is triggered lazily by
        // `try_alias_materialise`'s fallback at dispatch time. Letting the
        // async dispatcher (LISWAP-4 IntraForward) drive per-layer
        // `ensure_layer_loaded` on its worker thread lets first-touch overlap
        // with forward kernels — eliminating the 800+ ms batch stage we saw
        // collapse all hide opportunity.
        let alias_path_prefault = matches!(secondary.as_ref(), SecondaryMmap::Rpcmem(_));
        let t_pre0 = Instant::now();
        if !alias_path_prefault {
            secondary.prefault_layers(target_layers);
        }
        stages.prefault_ms = t_pre0.elapsed().as_secs_f64() * 1e3;

        // Determine whether to use the async path (LISWAP-2 prototype).
        // Requires both a dispatcher argument AND backend support.
        let use_async = async_dispatcher
            .map(|_| self.backend.supports_async_transfer())
            .unwrap_or(false);

        // ── env-gated peak monitoring (LLMRS_SWAP_DRAIN_DIAG=1) ──────────────
        // Track max release_worker queue depth observed across the batch.
        // Verifies the hypothesis "without drain backpressure, queue stays at
        // 1 layer naturally because release wall (~sub-ms drop on Adreno) is
        // hidden by the next layer build (~3 ms mmap_permute) / token gap
        // (~10 ms forward in pertick=1)". If max > 1 for a given mode, the
        // user's spike concern is real for that mode.
        let diag_enabled = std::env::var("LLMRS_SWAP_DRAIN_DIAG")
            .map(|v| v == "1")
            .unwrap_or(false);
        // ── Sub-batch reactive wait (was: pause/break, 2026-05-14) ─────────
        // Memory-spike avoidance: dispatching the next layer while the
        // previous primary release is still in flight would let
        // release_worker queue depth grow beyond 1 (violates
        // `feedback_no_memory_spike.md`).
        //
        // Previously this site `break`-ed the loop, which silently dropped
        // the remaining `chunk[i..]` because `IncrementalSwapPlan::drain_chunk`
        // had already popped them. Sync K>1 incremental then collapsed to
        // K=1-per-call AND silently lost the rest of the chunk (cold-fire
        // sync K=5 measurement, 2026-05-13: 7/28 layers actually swapped).
        //
        // New behaviour: **wait** for the previous release to drain instead
        // of breaking. Decode is paused (caller's main thread blocks here)
        // until the chunk completes — the caller asked for K layers per
        // tick, we honour that contract with queue depth ≤ 1 and zero
        // silent drops.
        let sub_batch_pause_diag = std::env::var("LLMRS_SUB_BATCH_PAUSE_DIAG")
            .map(|v| v == "1")
            .unwrap_or(false);
        // LISWAP-7 PoC: disable sub-batch wait so the whole chunk enqueues
        // immediately. Releases drain in the background dispatcher/release
        // worker — pending queue depth may grow beyond 1, so caller takes
        // responsibility for memory headroom.
        let sub_batch_no_wait = std::env::var("LLMRS_SUB_BATCH_NO_WAIT")
            .map(|v| v == "1")
            .unwrap_or(false);
        // LISWAP-7 Path 3 PoC: build all chunk layers in parallel via rayon
        // before the dispatch loop. mmap_permute still runs on the main-thread
        // pool but layers are processed concurrently, collapsing the sequential
        // accumulation seen in `stages.mmap_permute_ms` (1865ms @ K=32 → ~250ms
        // on 8-core Jetson estimate). NOT true forward/swap overlap — the main
        // thread still blocks until the parallel build completes — but isolates
        // the "CPU work" cost so we can quantify the remaining gap before going
        // to a fully background-thread design (length 1).
        let par_build = std::env::var("LLMRS_SWAP_PAR_BUILD")
            .map(|v| v == "1")
            .unwrap_or(false);

        // LISWAP-8 Phase A: single-thread background fetch. Main thread
        // submits a `ChunkDispatchJob` closure to the existing async swap
        // dispatcher (single worker `llmrs-async-swap`). The worker runs
        // build → wait_event → arc_swap → release-chain entirely off the
        // critical path. mmap_permute + cuMemAlloc + cuMemcpyAsync (which
        // currently block the main thread inside `build_layer_from_mmap_async`)
        // are all moved to the worker — hiding the per-layer 50 ms behind
        // forward (par_seq 2026-05-14 confirmed single-thread dispatcher
        // background work has zero impact on forward at pending=31).
        //
        // Mutually exclusive with `par_build` (par_iter multi-thread
        // pre-build) — `par_build` takes precedence if both are set.
        let bg_fetch = std::env::var("LLMRS_SWAP_BG_FETCH")
            .map(|v| v == "1")
            .unwrap_or(false);

        // LISWAP-8 Phase B: pre-allocated layer object pool path. Reads
        // `LLMRS_SWAP_LAYER_POOL=1` env. Only takes effect when both the
        // executor has a `layer_pool` attached AND `bg_fetch` is active
        // (pool dispatch piggy-backs on the bg_fetch closure flow).
        #[cfg(feature = "cuda-embedded")]
        let layer_pool_active = std::env::var("LLMRS_SWAP_LAYER_POOL")
            .map(|v| v == "1")
            .unwrap_or(false);

        // LISWAP-8 Hammer D: zero-copy mmap alias swap. When `Some`
        // registration is attached AND this env is set, each layer's
        // weight tensors are installed as `CudaMmapAliasBuffer` views
        // into the registered secondary mmap — no cuMemAlloc, no
        // cuMemcpyHtoDAsync, no CPU memcpy. PoC: ignores Q/K runtime
        // unpermute (accuracy-affecting for GGUF source; AUF secondary
        // already pre-unpermuted, safe).
        #[cfg(feature = "cuda-embedded")]
        let mmap_alias_active = std::env::var("LLMRS_SWAP_MMAP_ALIAS")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Pre-build cache (par_build only): layer_idx -> Result<(layer, evt)>.
        // Populated outside the loop so the loop only submits + accumulates
        // stats.
        let mut built_cache: std::collections::HashMap<
            usize,
            Result<(TransformerLayer, GpuEvent), SwapError>,
        > = std::collections::HashMap::new();
        if use_async && par_build {
            use rayon::prelude::*;
            // LISWAP-7 Path 3 PoC v2: when LLMRS_SWAP_SEQ_BUILD=1, route the
            // prebuild through sequential `.iter()` instead of `par_iter()`.
            // Used to disentangle multi-thread side-effects (rayon worker
            // activation / kernel mmap_sem contention) from the prebuild
            // cache reuse path itself. Single-threaded prebuild should match
            // baseline forward (67 ms) iff the multi-thread access is what
            // perturbs forward; otherwise the prebuild restructure alone
            // (cache lookup vs inline build) is the cause.
            let par_seq_mode = std::env::var("LLMRS_SWAP_SEQ_BUILD")
                .map(|v| v == "1")
                .unwrap_or(false);
            let t_par0 = Instant::now();
            let pairs: Vec<AsyncLayerBuildPair> = if par_seq_mode {
                target_layers
                    .iter()
                    .filter_map(|&idx| {
                        let slot = layers.get(idx)?;
                        if slot.current_dtype() == self.target_dtype {
                            return None;
                        }
                        slot.secondary_mmap_handle()?;
                        Some((idx, self.build_layer_from_mmap_async(secondary, slot, idx)))
                    })
                    .collect()
            } else {
                target_layers
                    .par_iter()
                    .filter_map(|&idx| {
                        let slot = layers.get(idx)?;
                        if slot.current_dtype() == self.target_dtype {
                            return None;
                        }
                        slot.secondary_mmap_handle()?;
                        Some((idx, self.build_layer_from_mmap_async(secondary, slot, idx)))
                    })
                    .collect()
            };
            stages.mmap_permute_ms += t_par0.elapsed().as_secs_f64() * 1e3;
            built_cache = pairs.into_iter().collect();
        }

        let mut max_release_pending: usize = 0;
        let mut max_dispatcher_pending: usize = 0;

        for (i, &layer_idx) in target_layers.iter().enumerate() {
            // Sub-batch reactive wait: spin until previous release completes
            // before dispatching the next layer. First layer (i == 0) is
            // unconditional. The wait keeps queue depth ≤ 1 → no memory
            // spike, and the loop always finishes the full chunk → no
            // silent layer drops in the plan.
            if i > 0
                && !sub_batch_no_wait
                && let Some(worker) = &self.release_worker
            {
                let wait_start = if sub_batch_pause_diag {
                    Some(Instant::now())
                } else {
                    None
                };
                while worker.pending_count() > 0 {
                    std::hint::spin_loop();
                }
                if let Some(t0) = wait_start {
                    let wait_ms = (t0.elapsed().as_secs_f64() * 1000.0) as f32;
                    if wait_ms > 0.1 {
                        crate::action_diag_helper::log_swap_sub_batch_wait(layer_idx, wait_ms);
                    }
                }
            }
            // Sample queue depths at the start of each layer iteration —
            // this is the moment when memory peak matters (about to allocate
            // new layer's cl_mems / build new tensor on top of any displaced
            // weights still pending release).
            if diag_enabled {
                if let Some(worker) = &self.release_worker {
                    let p = worker.pending_count();
                    if p > max_release_pending {
                        max_release_pending = p;
                    }
                }
                if use_async && let Some(d) = async_dispatcher {
                    let p = d.pending_count();
                    if p > max_dispatcher_pending {
                        max_dispatcher_pending = p;
                    }
                }
            }
            // ENG-DAT-C08: out-of-range silently skipped.
            let Some(slot) = layers.get(layer_idx) else {
                report.skipped.push(layer_idx);
                continue;
            };
            // Idempotent: already at target dtype.
            if slot.current_dtype() == self.target_dtype {
                report.skipped.push(layer_idx);
                continue;
            }
            // Secondary handle may have been stripped (no swap path). Treat
            // as skip rather than error — matches ENG-DAT-C09 spirit when
            // the per-slot handle was never installed.
            if slot.secondary_mmap_handle().is_none() {
                report.skipped.push(layer_idx);
                continue;
            }

            let from_dtype = slot.current_dtype();

            if use_async && bg_fetch && !par_build {
                // ── LISWAP-8 Phase A: full background fetch ──────────────────
                //
                // Hand off the *entire* per-layer pipeline (build →
                // wait_event → arc_swap → release-chain) to the dispatcher
                // worker thread via `submit_dispatch_chunk`. Main thread
                // does only a channel push (~µs) and returns to the next
                // layer / forward immediately.
                //
                // Compared to the LISWAP-2 async path below, this also
                // moves `build_layer_from_mmap_async` (mmap_permute +
                // cuMemAlloc + cuMemcpyAsync) to the worker — the dominant
                // remaining cost (~50 ms / layer) that the LISWAP-2 path
                // still runs on main.
                //
                // The closure captures owned Arc clones + `ModelConfig`
                // (Clone) so no `&'a` borrow of `SwapExecutor` leaks into
                // the `Send + 'static` boundary.
                let dispatcher = async_dispatcher.expect("use_async => dispatcher Some");

                let secondary_arc = Arc::clone(secondary);
                let slot_arc = Arc::clone(slot);
                let backend_arc = Arc::clone(&self.backend);
                let config_owned = self.config.clone();
                let target_dtype = self.target_dtype;
                let release_worker_clone = self.release_worker.clone();
                // Capture dispatcher identity label into BgFetch closure for log lines.
                let bg_kind: &'static str = self.kind;

                // LISWAP-8 Phase B: try to take a pool entry for this
                // layer. If `Some`, the worker closure overwrites the
                // entry's buffers in place via `enqueue_write_into_async`
                // (no cuMemAlloc on the dispatch path). If `None`, fall
                // back to the Phase A path (`build_layer_async_standalone`)
                // which allocates fresh device buffers.
                #[cfg(feature = "cuda-embedded")]
                let pool_entry: Option<TransformerLayer> = if layer_pool_active {
                    self.layer_pool.as_ref().and_then(|p| p.take())
                } else {
                    None
                };
                #[cfg(not(feature = "cuda-embedded"))]
                let pool_entry: Option<TransformerLayer> = None;

                // Hammer D: mmap alias path takes highest priority when
                // registration is attached and the env is set.
                #[cfg(feature = "cuda-embedded")]
                let mmap_reg: Option<
                    Arc<crate::memory::cuda::mmap::CudaMmapRegistration>,
                > = if mmap_alias_active {
                    self.mmap_registration.clone()
                } else {
                    None
                };
                #[cfg(not(feature = "cuda-embedded"))]
                let mmap_reg: Option<()> = None;

                let job = crate::weight::async_swap::ChunkDispatchJob {
                    run: Box::new(move || {
                        let result = {
                            #[cfg(feature = "cuda-embedded")]
                            {
                                if let Some(reg) = mmap_reg {
                                    build_layer_via_mmap_alias_standalone(
                                        &secondary_arc,
                                        &slot_arc,
                                        layer_idx,
                                        &backend_arc,
                                        &reg,
                                    )
                                } else if let Some(entry) = pool_entry {
                                    build_layer_via_pool_standalone(
                                        &secondary_arc,
                                        &slot_arc,
                                        layer_idx,
                                        &backend_arc,
                                        &config_owned,
                                        entry,
                                    )
                                } else {
                                    build_layer_async_standalone(
                                        &secondary_arc,
                                        &slot_arc,
                                        layer_idx,
                                        &backend_arc,
                                        &config_owned,
                                    )
                                }
                            }
                            #[cfg(not(feature = "cuda-embedded"))]
                            {
                                let _ = mmap_reg;
                                if let Some(entry) = pool_entry {
                                    build_layer_via_pool_standalone(
                                        &secondary_arc,
                                        &slot_arc,
                                        layer_idx,
                                        &backend_arc,
                                        &config_owned,
                                        entry,
                                    )
                                } else {
                                    build_layer_async_standalone(
                                        &secondary_arc,
                                        &slot_arc,
                                        layer_idx,
                                        &backend_arc,
                                        &config_owned,
                                    )
                                }
                            }
                        };
                        let (new_layer, write_event) = match result {
                            Ok(pair) => pair,
                            Err(e) => {
                                crate::action_diag_helper::log_swap_failed(
                                    bg_kind,
                                    &format!("BgFetch build failed: {e}"),
                                    Some(layer_idx),
                                    None,
                                );
                                return;
                            }
                        };

                        if !write_event.is_dummy()
                            && let Err(e) = backend_arc.wait_event_blocking(&write_event)
                        {
                            crate::action_diag_helper::log_swap_failed(
                                bg_kind,
                                &format!("BgFetch wait_event_blocking failed: {e}"),
                                Some(layer_idx),
                                None,
                            );
                            return;
                        }

                        let new_arc: Arc<LayerWeights> = Arc::new(new_layer);
                        let old = slot_arc.swap_weights(new_arc, target_dtype);

                        if let Some(rw) = &release_worker_clone
                            && let Ok(layer) = Arc::try_unwrap(old)
                        {
                            rw.enqueue_release(layer);
                        }
                    }),
                };

                let t_submit = Instant::now();
                if let Err(e) = dispatcher.submit_dispatch_chunk(job) {
                    crate::action_diag_helper::log_swap_failed(
                        self.kind,
                        &format!("BgFetch submit_dispatch_chunk failed: {e}"),
                        Some(layer_idx),
                        None,
                    );
                    report.skipped.push(layer_idx);
                    continue;
                }
                // arc_swap stage timing here is just the submit latency (ns
                // scale). The real arc_swap / mmap_permute / madvise run
                // inside the worker and are not attributable to main thread.
                stages.arc_swap_ms += t_submit.elapsed().as_secs_f64() * 1e3;

                report.swapped.push(SwappedLayer {
                    layer_idx,
                    from_dtype,
                    to_dtype: self.target_dtype,
                });
            } else if use_async {
                // ── LISWAP-2 async path — background commit activated (Phase 6.2) ──
                //
                // Background commit activated. See plan: compiled-chasing-hopper.md §Approach.
                //
                // Phase 5 found that the prior "async enqueue + main-thread per-event wait"
                // approach produced +1.27% on Adreno (no improvement). Root cause: main
                // thread blocked on `wait_event_blocking` per-layer → forward and write could
                // not overlap. Phase 6.2 removes that block entirely.
                //
                // New path:
                //   Stage (a): enqueue H2D writes on transfer queue/stream — non-blocking.
                //   Stage (b): hand off (slot, new_weights, event) to dispatcher worker.
                //              Worker waits on `write_event`, then ArcSwap-commits and
                //              chains `release_worker`. Main thread returns immediately to
                //              the next layer (and subsequently the next forward call).
                //
                // Concurrency note (INV-144): forward(N+1) may start before dispatcher
                // commits layer N. `slot.load_weights()` (ArcSwap snapshot) sees either
                // old or new weights — both valid. Plan completion calls `drain(2s)` to
                // guarantee all layers are fully committed before the plan is marked done.

                // Stage (a): enqueue H2D writes on transfer queue/stream — non-blocking.
                // LISWAP-7 Path 3: when `par_build` mode is on, the build was
                // already done in parallel before the loop; lookup-only here.
                let async_result = if par_build {
                    built_cache.remove(&layer_idx).unwrap_or_else(|| {
                        // Build was skipped in the par phase (slot already at
                        // target dtype, secondary missing, etc.). The current
                        // loop iter will hit the same skip checks above and
                        // bypass this branch — reaching here only on race or
                        // logic divergence. Surface as transient error.
                        Err(SwapError::AsyncTransferUnavailable {
                            layer: layer_idx,
                            source: anyhow::anyhow!("par_build cache miss for layer {layer_idx}"),
                        })
                    })
                } else {
                    let t_a0 = Instant::now();
                    let r = self.build_layer_from_mmap_async(secondary, slot, layer_idx);
                    stages.mmap_permute_ms += t_a0.elapsed().as_secs_f64() * 1e3;
                    r
                };

                let (new_layer, write_event) = match async_result {
                    Ok(pair) => pair,
                    Err(e) => {
                        crate::action_diag_helper::log_swap_failed(
                            self.kind,
                            &format!("AsyncSwap async transfer failed: {e}"),
                            Some(layer_idx),
                            None,
                        );
                        report.skipped.push(layer_idx);
                        continue;
                    }
                };

                // Hand off to dispatcher worker. Worker waits on `write_event`,
                // then commits arc_swap and chains primary release. Main thread
                // returns immediately to the next layer (and subsequently the
                // next forward call).
                let dispatcher = async_dispatcher.expect("use_async => dispatcher Some");
                let new_arc: Arc<LayerWeights> = Arc::new(new_layer);
                let job = crate::weight::async_swap::SwapCommitJob {
                    slot: Arc::clone(slot),
                    new_weights: new_arc,
                    new_dtype: self.target_dtype,
                    write_event: Arc::new(write_event),
                    release_worker: self.release_worker.clone(),
                    on_complete: None,
                    layer_idx: None,
                };

                // arc_swap_ms and madvise_ms accrue inside the dispatcher worker;
                // only record the submit latency (nanoseconds) on the main thread.
                let t_b0 = Instant::now();
                if let Err(e) = dispatcher.submit_commit(job) {
                    crate::action_diag_helper::log_swap_failed(
                        self.kind,
                        &format!("AsyncSwap dispatcher submit failed: {e}"),
                        Some(layer_idx),
                        None,
                    );
                    report.skipped.push(layer_idx);
                    continue;
                }
                stages.arc_swap_ms += t_b0.elapsed().as_secs_f64() * 1e3;
                // madvise_ms is 0 on main thread (release chained in worker).

                // ── No drain backpressure (verified 2026-05-11) ──────────────
                // 이전 진단의 "Adreno GPU flush 30~50 ms/layer" 가설은 잘못된
                // 인과관계 — drain polling/IPC overhead 자체가 wall cost를
                // 만든 것이고 release_worker drop 자체는 sub-ms (sync path
                // 진단의 첫 layer 55ms cold 후 0.03ms steady 와 일치).
                // backpressure 없이도 release_worker queue depth는 1 layer로
                // 자연 stable (다음 layer build / token gap이 release wall을
                // 흡수). 검증은 LLMRS_SWAP_DRAIN_DIAG=1 로 batch 끝 [SwapPeak]
                // log 확인.

                report.swapped.push(SwappedLayer {
                    layer_idx,
                    from_dtype,
                    to_dtype: self.target_dtype,
                });
            } else {
                // ── Synchronous path (original) ────────────────────────────────
                // Stage (a): secondary mmap slice + Q/K permutation + GPU upload.
                let t_a0 = Instant::now();
                let new_layer = self.build_layer_from_mmap(secondary, slot, layer_idx)?;
                stages.mmap_permute_ms += t_a0.elapsed().as_secs_f64() * 1e3;

                let new_arc: Arc<LayerWeights> = Arc::new(new_layer);

                // ── Stage (b): Atomic install — INV-123. `swap_weights` now uses
                // `ArcSwap::swap()` internally and returns the previous Arc after
                // `wait_for_readers` has flushed any in-flight ArcSwap hazard
                // pointers. The returned Arc is the only outstanding reference
                // held by this dispatch path — readers from prior forward passes
                // have already dropped their snapshots (the swap dispatcher runs
                // strictly between forwards). ────────────────────────────────────
                let t_b0 = Instant::now();
                let old = slot.swap_weights(new_arc, self.target_dtype);
                stages.arc_swap_ms += t_b0.elapsed().as_secs_f64() * 1e3;

                // ── Stage (c): Primary cl_mem release — WSWAP-5-PRIMARY-DROP.
                //
                // Take ownership of the inner `LayerWeights` from the returned
                // Arc and drop the primary weight tensors explicitly. Each
                // tensor's `Arc<dyn Buffer>` is then the unique owner of its
                // backing `OpenCLBuffer`, so the destructor releases the
                // underlying `cl_mem` immediately. This reclaims the F16
                // primary 2.4 GB / 145 cl_mem footprint that previously stayed
                // alive on Galaxy S25 ratio=1.0 mixed (`phase_5_tbt_diag.md`).
                //
                // Falls back to madvise-only when `Arc::try_unwrap` fails.
                // The latter only happens if some other holder kept the Arc —
                // none expected in production (forwards run sequentially against
                // the dispatcher), but the fallback preserves correctness if a
                // future caller ever holds a snapshot across a swap.
                //
                // ENG-ALG-211 step (c) refined; `Self::madvise_if_exclusive`
                // remains the fallback path so MADV_DONTNEED still fires on
                // mmap-backed primaries when we can't acquire unique ownership.
                // ── Stage (c): Primary cl_mem release / deferred enqueue ───────
                //
                // ENG-ALG-228: when a `PrimaryReleaseWorker` is attached, enqueue
                // the displaced `LayerWeights` for asynchronous drop on the worker
                // thread instead of blocking inline. The worker calls
                // `clReleaseMemObject` off the critical path (~1 ms × 7 tensors per
                // layer on Adreno). `stages.madvise_ms` then only records the
                // enqueue cost (nanoseconds) rather than the full release chain.
                //
                // Fallback: when no worker is set (host tests, CPU backend, partial
                // init), the original inline `release_primary_weights` path runs
                // unchanged. The `Err(arc)` branch is never affected — madvise is
                // always inline because it requires the live pointer.
                let t_c0 = Instant::now();
                match Arc::try_unwrap(old) {
                    Ok(layer) => {
                        if let Some(worker) = &self.release_worker {
                            // Async path: enqueue for background drop (ENG-ALG-228).
                            // No drain — release worker drop은 sub-ms이라 다음
                            // layer build (~3ms) 가 충분히 흡수. peak 검증은
                            // LLMRS_SWAP_DRAIN_DIAG=1 [SwapPeak] log 참고.
                            worker.enqueue_release(layer);
                        } else {
                            // Inline fallback (host tests, CPU backend).
                            Self::release_primary_weights(&self.backend, layer);
                        }
                    }
                    Err(arc) => {
                        // Non-exclusive ownership — best-effort reclaim only.
                        // Records strong_count for diagnostics on the AOS / GGUF
                        // path; the AUF SOA bypass path doesn't actually need
                        // madvise (its primary cl_mem was the F16 GGUF copy
                        // routed through `copy_weight_from`, not an mmap'd page).
                        Self::madvise_if_exclusive(&arc);
                    }
                }
                stages.madvise_ms += t_c0.elapsed().as_secs_f64() * 1e3;

                report.swapped.push(SwappedLayer {
                    layer_idx,
                    from_dtype,
                    to_dtype: self.target_dtype,
                });
            }
        }

        // (e) Single batch-level bump of the global ratio_generation counter.
        // Empty swaps do NOT bump (ENG-ALG-211: "if !swapped.is_empty()").
        if !report.swapped.is_empty() {
            // ── Stage (sync): backend.synchronize() — INV-142 / ENG-ALG-231 ──
            //
            // All `alloc_and_upload_soa_buffers` calls during Stage (a) use
            // non-blocking `enqueue_write_buffer` (ENG-ALG-230). A single
            // `clFinish` here drains the entire command queue before Stage (d)
            // consumers (SOA registry rebuild, next forward) read the uploaded
            // data.
            //
            // Ordering: synchronize → invalidate_noshuffle_soa_registry →
            //   ensure/restore_pre_converted_soa_registration → ratio_generation bump.
            //
            // CPU / CUDA backends: default impl returns `Ok(())` immediately.
            //
            // LISWAP-2 async path: synchronize() is SKIPPED. Per-event gating
            // inside the dispatcher worker thread replaces the full-barrier sync.
            // invalidate_noshuffle_soa_registry() and the gen_bump still run so
            // the next forward's plan rebuild observes a clean registry (safe:
            // AOS path never populated the registry, and AUF SOA path re-registers
            // after invalidate in Stage (d)).
            let t_sync0 = Instant::now();
            // 작업 C: alias 환경 (`SecondaryMmap::Rpcmem`) 에서는 H2D copy가
            // 발생하지 않아 (cl_mem이 rpcmem DMA-BUF + USE_HOST_PTR alias)
            // backend.synchronize()로 drain할 GPU 작업 자체가 없다. Phase 5
            // (process_commit) 와 동일한 패턴 — dummy event skip 과 같은 사유.
            // 비-alias path(memcpy)는 정상 sync 유지.
            let alias_path = matches!(secondary.as_ref(), SecondaryMmap::Rpcmem(_));
            if !use_async && !alias_path {
                self.backend
                    .synchronize()
                    .map_err(|e| SwapError::StageGateSyncFailed { source: e })?;
            }
            stages.synchronize_ms = t_sync0.elapsed().as_secs_f64() * 1e3;

            // ── Stage (e-pre): invalidate registry ───────────────────────────
            // ENG-ALG-221 / INV-130: invalidate the Adreno Q4_0 noshuffle SOA
            // registry before the generation bump so that the subsequent
            // `FullKernelPlan` rebuild (triggered by the bump via ENG-ALG-219)
            // re-registers SOA descriptors against the new `cl_mem` keys.
            // No-op on CPU / CUDA backends (trait default).
            //
            // Ordering rationale: clear must precede the bump so any concurrent
            // observer that takes the new generation cannot race against a
            // still-populated stale registry. The per-token `Arc<LayerWeights>`
            // snapshot (INV-121/123) already shields in-flight forwards that
            // captured the old generation.
            let t_e0 = Instant::now();
            self.backend.invalidate_noshuffle_soa_registry();
            stages.gen_bump_ms += t_e0.elapsed().as_secs_f64() * 1e3;

            // ── Stage (d): SOA re-conversion ─────────────────────────────────
            // ENG-ALG-222 / INV-131 — Phase 3.7a: re-convert Q4_0 weights to
            // SOA layout and register the entries against the new cl_mem
            // addresses **before** the generation bump. Without this safety
            // net the noshuffle GEMV kernel falls back to the AOS path on
            // Adreno (verified silently incorrect — Phase 3.6 measurements
            // showed only the first decoded token surviving). NoOp on
            // CPU / CUDA backends and on host OpenCL builds whose
            // `cvt_noshuffle` program failed to compile.
            //
            // AUF SOA bypass (Phase 3.7b): when the secondary source is an
            // AUF file with WEIGHTS_ADRENO_SOA, the bytes are already in SOA
            // layout. `ensure_noshuffle_soa_registered` is still called but
            // the backend registers directly without re-conversion. In this
            // case `soa_reconvert_ms` records only the registration overhead
            // (≈ 0 ms). `StageBreakdown::soa_reconvert_ms ≈ 0` signals the
            // fast-path to the caller.
            //
            // Ordering rationale: this MUST happen after `invalidate_*` (so
            // we re-populate a clean registry) and before the generation bump
            // (so the next forward seeing the new generation finds a fully
            // populated registry, eliminating any window where lookup misses).
            let t_d0 = Instant::now();
            let skip_soa_reconvert = secondary_mmap
                .map(|s| s.is_pre_converted_soa())
                .unwrap_or(false);
            if skip_soa_reconvert {
                // AUF WEIGHTS_ADRENO_SOA path: weights are already SOA-backed
                // at materialise time (`materialise_auf_soa_weight` →
                // `Backend::alloc_pre_converted_soa_tensor`). The
                // per-batch `invalidate_noshuffle_soa_registry()` above
                // wiped every entry — including the just-built ones — so we
                // re-register them here against the same `d_buf` keys via
                // `restore_pre_converted_soa_registration`. Tensors whose
                // buffer is *not* a `NoshuffleWeightBuffer` (e.g. AOS
                // fallback inside `materialise_auf_soa_weight`) re-enter
                // the GGUF SOA conversion path through the trait default.
                //
                // `soa_reconvert_ms` charge here is the registration cost
                // only (~µs / tensor). No host-side conversion runs and no
                // new cl_mem is allocated — this is the Phase 5 Sprint C
                // (WSWAP-5-AUF-PLACEHOLDER-DROP) shape of the bypass path.
                for swapped in &report.swapped {
                    let Some(slot) = layers.get(swapped.layer_idx) else {
                        continue;
                    };
                    let new_layer = slot.load_weights();
                    for tensor in [
                        &new_layer.wq,
                        &new_layer.wk,
                        &new_layer.wv,
                        &new_layer.wo,
                        &new_layer.w_gate,
                        &new_layer.w_up,
                        &new_layer.w_down,
                    ] {
                        if let Err(e) = self.backend.restore_pre_converted_soa_registration(tensor)
                        {
                            return Err(SwapError::BufferAllocationFailed {
                                layer: swapped.layer_idx,
                                source: e,
                            });
                        }
                    }
                }
            } else {
                // GGUF / AUF AOS path: intentionally do NOT register noshuffle
                // SOA entries here.
                //
                // Adreno regression: even with the fused kernel disabled (see
                // `OpenCLBackend::convert_q4_0_to_noshuffle`), routing the
                // swap-time tensor through the noshuffle GEMV path interacts
                // poorly with the swap-time `UnifiedBuffer` cl_mem and yields
                // garbage decode output. Default Q4_0 GGUF load works only
                // because `prepare_noshuffle_buffers` has a separate clone-
                // discard bug that drops the NoshuffleWeightBuffer
                // replacement, so matmul lookups miss and standard
                // `kernel_mul_mat_q4_0_f32` runs (correct on AOS bytes).
                //
                // Skipping registration here makes the swap path mirror the
                // accidentally-working Q4_0 GGUF default path. Re-enabling
                // requires fixing both the Adreno noshuffle GEMV regression
                // and the `prepare_noshuffle_buffers` clone discard.
                //
                // Diagnostic: `bin/test_q4_soa_byte_equal` exercises the
                // convert kernel byte-equal contract on-device and
                // currently confirms the legacy 4-step path is correct
                // while the fused path corrupts ~43 % of q_buf bytes.
            }
            stages.soa_reconvert_ms += t_d0.elapsed().as_secs_f64() * 1e3;

            // ── Stage (e-post): ratio_generation bump ─────────────────────────
            let t_e1 = Instant::now();
            let new_gen = ratio_generation.fetch_add(1, Ordering::SeqCst) + 1;
            stages.gen_bump_ms += t_e1.elapsed().as_secs_f64() * 1e3;
            report.ratio_generation_after = Some(new_gen);

            report.stage_breakdown = Some(stages);
        }

        // LISWAP-7 Path 3 / ε hypothesis: when par_build is on, drain
        // dispatcher + release_worker before returning so the next forward
        // starts with all background swap work complete (no in-flight commit
        // or cuMemFree). Tests whether the +67 ms/forward overhead observed
        // in 2026-05-14 par_build runs is caused by CUDA driver context lock
        // contention between background `cuMemFree` and forward
        // `cuLaunchKernel`.
        if par_build && use_async {
            if let Some(d) = async_dispatcher {
                let _ = d.drain(Duration::from_secs(5));
            }
            if let Some(worker) = &self.release_worker {
                let _ = worker.drain(Duration::from_secs(5));
            }
        }

        report.latency_ms = start.elapsed().as_secs_f64() * 1e3;

        // ── env-gated peak summary log ───────────────────────────────────────
        // Sampled at the start of each layer iter (above). Reports the worst
        // queue depth observed during the batch — the user's hard constraint
        // ("memory peak ≤ 1 layer") is satisfied iff max ≤ 1 here.
        if diag_enabled {
            let mode = if use_async { "async" } else { "sync" };
            crate::action_diag_helper::log_swap_batch_summary(
                self.kind,
                mode,
                target_layers.len(),
                max_release_pending,
                max_dispatcher_pending,
            );
        }

        Ok(report)
    }

    /// Build a fresh `TransformerLayer` from the secondary mmap for the
    /// given `layer_idx`, mirroring the primary loader's dtype handling and
    /// Q/K permutation gating.
    ///
    /// Non-swap fields (partition context) are preserved from the current
    /// snapshot so downstream tensor-partition state is retained.
    ///
    /// **AUF SOA fast path** (`secondary.is_pre_converted_soa()`):
    /// Q4_0 weights are routed through `materialise_auf_soa_weight` which
    /// delegates to `Backend::alloc_pre_converted_soa_tensor`. That call
    /// allocates the SOA `q_buf`/`d_buf`/optional `q_img` cl_mem, uploads
    /// the AUF payload, registers a noshuffle SOA entry keyed on the
    /// `d_buf` address, and returns a tensor whose backing buffer is a
    /// `NoshuffleWeightBuffer`. **No AOS placeholder cl_mem is allocated**
    /// (Phase 5 Sprint C — WSWAP-5-AUF-PLACEHOLDER-DROP). The Stage (d)
    /// `restore_pre_converted_soa_registration` re-registers the entry
    /// after the per-batch `invalidate_noshuffle_soa_registry()` clear so
    /// the ENG-ALG-221 ordering (clear → register → bump) is preserved.
    /// Non-Q4_0 layer tensors (none for the current swap target set) and
    /// non-OpenCL backends fall back to the AOS path.
    fn build_layer_from_mmap(
        &self,
        secondary: &Arc<SecondaryMmap>,
        slot: &LayerSlot,
        layer_idx: usize,
    ) -> Result<TransformerLayer, SwapError> {
        let old = slot.load_weights();

        let wq = self.materialise_weight(secondary, layer_idx, "attn_q.weight", &old.wq)?;
        let wk = self.materialise_weight(secondary, layer_idx, "attn_k.weight", &old.wk)?;
        let wv = self.materialise_weight(secondary, layer_idx, "attn_v.weight", &old.wv)?;
        let wo = self.materialise_weight(secondary, layer_idx, "attn_output.weight", &old.wo)?;
        let w_gate =
            self.materialise_weight(secondary, layer_idx, "ffn_gate.weight", &old.w_gate)?;
        let w_up = self.materialise_weight(secondary, layer_idx, "ffn_up.weight", &old.w_up)?;
        let w_down =
            self.materialise_weight(secondary, layer_idx, "ffn_down.weight", &old.w_down)?;

        // Norms are typically F32. We re-read them from secondary to avoid
        // relying on primary-lifetime buffers in the new snapshot. If the
        // secondary omits a norm (unlikely for standard GGUFs), fall back
        // to cloning the primary tensor — shape/dtype match is enforced by
        // `open_secondary` metadata validation.
        let attention_norm = self.clone_or_materialise_norm(
            secondary,
            layer_idx,
            "attn_norm.weight",
            &old.attention_norm,
        )?;
        let ffn_norm =
            self.clone_or_materialise_norm(secondary, layer_idx, "ffn_norm.weight", &old.ffn_norm)?;

        // Optional Gemma3 / Qwen norms — preserve existing tensors since
        // they are not part of the swap target set for this stage.
        //
        // DF-35-3 (ENG-ALG-219): tensor_partition × weight swap are mutually
        // exclusive. A weight swap installs new cl_mem handles for the weight
        // buffers; any pre-existing PartitionContext still points at the OLD
        // cl_mem slices and is therefore invalid after the swap. Force
        // partition_ctx to None so the plan-path correctly falls back to the
        // GPU-only FFN rather than dispatching a stale partition step.
        // The caller (generate.rs) will reinstate partition via
        // `prepare_tensor_partition` if `--tensor-partition` is active.
        Ok(TransformerLayer {
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            attention_norm,
            ffn_norm,
            qkv_bias: old.qkv_bias.clone(),
            q_norm: old.q_norm.clone(),
            k_norm: old.k_norm.clone(),
            pre_ffn_norm: old.pre_ffn_norm.clone(),
            post_ffn_norm: old.post_ffn_norm.clone(),
            partition_ctx: None, // DF-35-3: cleared on swap (see above)
        })
    }

    /// LISWAP-4 entry point for `IntraForwardSwapHook`: thin public wrapper
    /// over the private `build_layer_from_mmap_async`. Identical semantics —
    /// only exposed so the intra-forward hook in
    /// `models/weights/intra_forward_swap.rs` can re-use the async build path
    /// without re-implementing materialise / Q-K permutation.
    pub fn build_layer_from_mmap_async_for_hook(
        &self,
        secondary: &Arc<SecondaryMmap>,
        slot: &LayerSlot,
        layer_idx: usize,
    ) -> Result<(TransformerLayer, GpuEvent), SwapError> {
        self.build_layer_from_mmap_async(secondary, slot, layer_idx)
    }

    /// LISWAP-5 (B-2.4) entry point for `PhaseAwareSwapDispatcher`: build a
    /// single weight tensor from the secondary mmap and enqueue it on the
    /// async transfer queue. Returns the GPU-resident `Tensor` plus the H2D
    /// completion event.
    ///
    /// Used as the per-tensor chunk primitive — the caller (phase-aware
    /// dispatcher) accumulates per-tensor outputs into a `PartialLayer`
    /// staging slot and submits the layer's `SwapCommitJob` once the last
    /// chunk of that layer is enqueued.
    ///
    /// `subname` is the GGUF tensor sub-key (e.g. `attn_q.weight`,
    /// `ffn_gate.weight`). `primary` is the current GPU tensor for shape
    /// validation and Q/K permutation gating (mirrors the
    /// `build_layer_from_mmap_async` per-tensor branch).
    ///
    /// On CPU backends `enqueue_write_async` falls through to a synchronous
    /// `copy_weight_from`; the returned event is a dummy and `wait_event_blocking`
    /// is a no-op.
    pub fn build_tensor_from_mmap_async_for_hook(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
    ) -> Result<(Tensor, GpuEvent), SwapError> {
        let cpu_tensor = self.materialise_cpu_tensor(secondary, layer_idx, subname, primary)?;
        if self.backend.name().contains("CPU") {
            return Ok((cpu_tensor, GpuEvent::dummy()));
        }
        // LISWAP-6 Phase 5: alias path 면 cpu_tensor.buffer().cl_mem() 가 이미
        // GPU-visible (rpcmem DMA-BUF + USE_HOST_PTR alias). H2D copy 불필요 →
        // enqueue_write_async skip + dummy event 반환. process_commit 가
        // is_dummy() 체크로 wait_event_blocking 회피하여 forward 영향 0.
        if cpu_tensor.buffer().cl_mem().is_some() {
            return Ok((cpu_tensor, GpuEvent::dummy()));
        }
        self.backend.enqueue_write_async(&cpu_tensor).map_err(|e| {
            SwapError::AsyncTransferUnavailable {
                layer: layer_idx,
                source: e,
            }
        })
    }

    /// LISWAP-5 (B-2.4): same as `build_tensor_from_mmap_async_for_hook` but
    /// returns `Ok(None)` when the secondary lacks the tensor (e.g. an
    /// optional norm). Used by the phase-aware dispatcher to handle the
    /// post-attn / post-ffn norms that may or may not exist.
    pub fn build_optional_tensor_from_mmap_async_for_hook(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
    ) -> Result<Option<(Tensor, GpuEvent)>, SwapError> {
        if secondary.layer_tensor(layer_idx, subname).is_none() {
            return Ok(None);
        }
        Ok(Some(self.build_tensor_from_mmap_async_for_hook(
            secondary, layer_idx, subname, primary,
        )?))
    }

    /// LISWAP-2 async variant of `build_layer_from_mmap`.
    ///
    /// Enqueues H2D writes for each weight tensor on the backend's transfer
    /// queue/stream via `enqueue_write_async`, returning the completed
    /// `TransformerLayer` together with the **last** `GpuEvent`. Because the
    /// transfer queue/stream is in-order, waiting on the last event guarantees
    /// all preceding enqueues are GPU-visible.
    ///
    /// Norm tensors are materialised with the same async path. Optional tensors
    /// (qkv_bias, q_norm/k_norm/pre_ffn_norm/post_ffn_norm) are cloned from
    /// the existing snapshot — they are not part of the swap target and are
    /// already resident on the device.
    ///
    /// Returns `Err` if any `enqueue_write_async` call fails, allowing the
    /// caller to skip this layer and continue with others.
    fn build_layer_from_mmap_async(
        &self,
        secondary: &Arc<SecondaryMmap>,
        slot: &LayerSlot,
        layer_idx: usize,
    ) -> Result<(TransformerLayer, GpuEvent), SwapError> {
        let old = slot.load_weights();

        // Helper closure: materialise via enqueue_write_async (weight tensors).
        let enqueue_weight = |subname: &str,
                              primary: &Tensor|
         -> Result<(Tensor, GpuEvent), SwapError> {
            // Build cpu_tensor the same way materialise_weight does, then
            // enqueue asynchronously.
            let cpu_tensor = self.materialise_cpu_tensor(secondary, layer_idx, subname, primary)?;
            let is_cpu = self.backend.name().contains("CPU");
            if is_cpu {
                return Ok((cpu_tensor, GpuEvent::dummy()));
            }
            // LISWAP-6 Phase 5b: alias path 면 cpu_tensor.buffer().cl_mem() 가
            // 이미 GPU-visible (rpcmem DMA-BUF + USE_HOST_PTR alias). H2D copy
            // 불필요 → enqueue_write_async skip + dummy event 반환. LISWAP-1
            // (per-tick) / LISWAP-4 (intra-forward) 양쪽 모두 본 경로를 통하므로
            // alias 검출이 한 곳에서 처리된다 (build_tensor_from_mmap_async_for_hook
            // 와 동일 패턴, swap_executor.rs:907 참고).
            if cpu_tensor.buffer().cl_mem().is_some() {
                return Ok((cpu_tensor, GpuEvent::dummy()));
            }
            self.backend.enqueue_write_async(&cpu_tensor).map_err(|e| {
                SwapError::AsyncTransferUnavailable {
                    layer: layer_idx,
                    source: e,
                }
            })
        };

        // Helper closure: materialise norm via enqueue_write_async (or clone).
        let enqueue_norm = |subname: &str,
                            primary: &Tensor|
         -> Result<(Tensor, Option<GpuEvent>), SwapError> {
            if secondary.layer_tensor(layer_idx, subname).is_none() {
                return Ok((primary.clone(), None));
            }
            let cpu_tensor = self.materialise_cpu_tensor(secondary, layer_idx, subname, primary)?;
            let is_cpu = self.backend.name().contains("CPU");
            if is_cpu {
                return Ok((cpu_tensor, None));
            }
            // LISWAP-6 Phase 5b: alias path → enqueue_write_async skip. norm은
            // 보통 작아 alias 의 이득이 크지 않지만 일관성 유지를 위해 동일 처리.
            // last_event 선택 시 None 처리되므로 호출부 수정 불요 (line 1014
            // last_event = evt_fnorm.unwrap_or_else(|| evt_anorm.unwrap_or(evt_down))).
            if cpu_tensor.buffer().cl_mem().is_some() {
                return Ok((cpu_tensor, None));
            }
            let (gpu_tensor, evt) = self.backend.enqueue_write_async(&cpu_tensor).map_err(|e| {
                SwapError::AsyncTransferUnavailable {
                    layer: layer_idx,
                    source: e,
                }
            })?;
            Ok((gpu_tensor, Some(evt)))
        };

        let (wq, evt_wq) = enqueue_weight("attn_q.weight", &old.wq)?;
        let (wk, evt_wk) = enqueue_weight("attn_k.weight", &old.wk)?;
        let (wv, evt_wv) = enqueue_weight("attn_v.weight", &old.wv)?;
        let (wo, evt_wo) = enqueue_weight("attn_output.weight", &old.wo)?;
        let (w_gate, evt_gate) = enqueue_weight("ffn_gate.weight", &old.w_gate)?;
        let (w_up, evt_up) = enqueue_weight("ffn_up.weight", &old.w_up)?;
        let (w_down, evt_down) = enqueue_weight("ffn_down.weight", &old.w_down)?;

        let (attention_norm, evt_anorm) = enqueue_norm("attn_norm.weight", &old.attention_norm)?;
        let (ffn_norm, evt_fnorm) = enqueue_norm("ffn_norm.weight", &old.ffn_norm)?;

        // The last enqueued event covers all preceding in-order enqueues.
        // Fall back through earlier events until we find a non-dummy one.
        // In the common GPU case, evt_fnorm or evt_anorm will be the last.
        let last_event = evt_fnorm.unwrap_or_else(|| evt_anorm.unwrap_or(evt_down));
        // Keep earlier events for the ordering chain (all unused — in-order
        // queue guarantees they complete when last_event completes).
        let _ = (evt_wq, evt_wk, evt_wv, evt_wo, evt_gate, evt_up);

        Ok((
            TransformerLayer {
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                attention_norm,
                ffn_norm,
                qkv_bias: old.qkv_bias.clone(),
                q_norm: old.q_norm.clone(),
                k_norm: old.k_norm.clone(),
                pre_ffn_norm: old.pre_ffn_norm.clone(),
                post_ffn_norm: old.post_ffn_norm.clone(),
                partition_ctx: None, // DF-35-3: cleared on swap (see above)
            },
            last_event,
        ))
    }

    /// Materialise one tensor from the secondary mmap into a CPU-side tensor
    /// (without uploading to the GPU). Used by `build_layer_from_mmap_async`
    /// to separate the CPU-side permutation work from the H2D enqueue.
    ///
    /// The AUF SOA fast path is deliberately skipped here — `enqueue_write_async`
    /// targets the AOS path for the prototype and AUF SOA registration is
    /// handled synchronously in Stage (d). Production quality may revisit this.
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path swap materialise
    fn materialise_cpu_tensor(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
    ) -> Result<Tensor, SwapError> {
        let info = secondary.layer_tensor(layer_idx, subname).ok_or_else(|| {
            SwapError::SecondaryTensorMissing {
                layer: layer_idx,
                subname: subname.to_string(),
            }
        })?;

        let shape = primary.shape().clone();
        let expected_dims: Vec<usize> = primary.shape().dims().to_vec();
        let sec_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        if sec_rev != expected_dims {
            return Err(SwapError::ShapeMismatch {
                layer: layer_idx,
                subname: subname.to_string(),
                primary: expected_dims,
                secondary: info.dims.clone(),
            });
        }

        let data = secondary.tensor_bytes(info);
        let canonical_name = format!("blk.{layer_idx}.{subname}");
        let needs_unpermute = secondary.needs_qk_unpermute_at_swap()
            && qk_permute_shape(&canonical_name, self.config).is_some();
        let permuted_bytes: Option<Vec<u8>> = if secondary.needs_qk_unpermute_at_swap() {
            if let Some((n_head, head_dim)) = qk_permute_shape(&canonical_name, self.config) {
                let total_rows = shape.dims()[0];
                debug_assert_eq!(total_rows, n_head * head_dim);
                if !data.len().is_multiple_of(total_rows) {
                    return Err(SwapError::ShapeMismatch {
                        layer: layer_idx,
                        subname: subname.to_string(),
                        primary: expected_dims,
                        secondary: info.dims.clone(),
                    });
                }
                let row_size_bytes = data.len() / total_rows;
                Some(unpermute_qk_rows(data, n_head, head_dim, row_size_bytes))
            } else {
                None
            }
        } else {
            None
        };

        // ── LISWAP-6 alias fast path ────────────────────────────────────────
        // Rpcmem variant + no qk-unpermute → backend may produce a cl_mem
        // alias on top of the rpcmem region, eliminating the H2D copy.
        // Falls back to the standard borrowed/permuted path if either
        //   (a) the alias allocation fails, or
        //   (b) the backend lacks alias support.
        #[cfg(feature = "opencl")]
        if !needs_unpermute
            && let Some(t) =
                self.try_alias_materialise(secondary, layer_idx, subname, info, &shape)?
        {
            return Ok(t);
        }
        #[cfg(not(feature = "opencl"))]
        let _ = needs_unpermute;

        let cpu_buf: Arc<dyn Buffer> = if let Some(owned) = permuted_bytes {
            Arc::new(SharedBuffer::from_vec(owned, info.dtype))
        } else {
            Arc::new(crate::memory::host::mmap::MmapBuffer::borrow(
                data,
                info.dtype,
                secondary.clone(),
            ))
        };

        let cpu_backend = crate::backend::cpu::cpu_singleton();
        Ok(Tensor::new(shape, cpu_buf, cpu_backend))
    }

    /// LISWAP-6 — try to build a `RpcmemAliasBuffer`-backed tensor for this
    /// (layer, subname) combination. Returns `Ok(None)` if any precondition
    /// fails (secondary not Rpcmem, backend doesn't support alias, alloc
    /// returns None). Caller handles the fallback path.
    #[cfg(feature = "opencl")]
    fn try_alias_materialise(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        info: &SecondaryTensorInfo,
        shape: &crate::shape::Shape,
    ) -> Result<Option<Tensor>, SwapError> {
        // Only the Rpcmem variant carries a per-layer rpcmem region.
        let SecondaryMmap::Rpcmem(rpc) = secondary.as_ref() else {
            return Ok(None);
        };

        // LISWAP-6 Phase 1 — fast path: pre-built alias from the eager
        // prefault populates the (layer, subname) cache. Cache hits skip
        // both `ensure_layer_loaded` and `clCreateBuffer(USE_HOST_PTR)`.
        if let Some(alias_buf) = rpc.cached_alias(layer_idx, subname) {
            debug_assert_eq!(alias_buf.size(), info.len, "cached alias length mismatch");
            debug_assert_eq!(alias_buf.dtype(), info.dtype, "cached alias dtype mismatch");
            return Ok(Some(Tensor::new(
                shape.clone(),
                alias_buf,
                Arc::clone(&self.backend),
            )));
        }

        // Fallback: cache miss (norm tensors / non-prefaulted layers / store
        // constructed without backend). Allocate one cl_mem alias on demand.
        let entry = rpc.host_ptr_for(layer_idx, subname).map_err(|e| {
            SwapError::BufferAllocationFailed {
                layer: layer_idx,
                source: e,
            }
        })?;
        let Some(alias) = entry else {
            return Ok(None);
        };
        debug_assert_eq!(alias.len, info.len, "rpcmem region length mismatch");
        debug_assert_eq!(alias.dtype, info.dtype, "rpcmem region dtype mismatch");

        // SAFETY: `alias.host_ptr` + `alias.offset..+alias.len` is the
        // rpcmem region returned by `host_ptr_for`, length/dtype validated
        // by the debug_assert pair above. Lifetime is pinned by
        // `Arc::clone(secondary)` (whole store) + `alias.region` (this
        // layer's rpcmem allocation), both moved into the alias buffer per
        // `Backend::alloc_alias_weight_buffer` contract.
        let secondary_dyn: Arc<dyn crate::memory::host::mmap::MmapKeepAlive> = secondary.clone();
        let region_dyn: Arc<dyn crate::memory::secondary::RpcmemRegionGuard> = alias.region;
        let buf = unsafe {
            self.backend.alloc_alias_weight_buffer(
                alias.host_ptr,
                alias.offset,
                alias.len,
                alias.dtype,
                secondary_dyn,
                region_dyn,
            )
        }
        .map_err(|e| SwapError::BufferAllocationFailed {
            layer: layer_idx,
            source: e,
        })?;
        let Some(alias_buf) = buf else {
            return Ok(None);
        };
        Ok(Some(Tensor::new(
            shape.clone(),
            alias_buf,
            Arc::clone(&self.backend),
        )))
    }

    /// Dispatch a swap-target *weight* tensor through the right materialise
    /// path based on the secondary source format.
    ///
    /// - GGUF / non-pre-converted secondaries → standard `materialise_tensor`
    ///   (AOS bytes, runtime Q/K unpermute, runtime SOA reconversion).
    /// - AUF `WEIGHTS_ADRENO_SOA` (Phase 3.7b SOA bypass) → `materialise_auf_soa_weight`
    ///   which uploads pre-converted SOA bytes into a key-holder `cl_mem` of
    ///   the AOS shape. Q/K permutation is **already applied** at AUF build
    ///   time (`auf_tool::extract_weight_blobs`), so the runtime path is a
    ///   straight zero-copy upload. Non-Q4_0 entries (none for the current
    ///   swap-target set) fall back to the AOS path.
    fn materialise_weight(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
    ) -> Result<Tensor, SwapError> {
        // AUF SOA fast path is OpenCL-only and Q4_0-only. The dtype gate
        // checks the *target* dtype (post-swap), not the primary — when
        // swapping F16→Q4_0 the primary is F16 but the AUF SOA bypass
        // applies (it converts the secondary's pre-baked SOA payload into a
        // Q4_0 NoshuffleWeightBuffer). Falling back through
        // `materialise_tensor` for any mismatch keeps the host-test surface
        // (CPU backend, non-Q4_0 norms, etc.) on the original code path.
        let secondary_info = secondary.layer_tensor(layer_idx, subname);
        let secondary_dtype_is_q4_0 = secondary_info.is_some_and(|i| i.dtype == DType::Q4_0);
        if secondary.is_pre_converted_soa()
            && !self.backend.name().contains("CPU")
            && self.target_dtype == DType::Q4_0
            && secondary_dtype_is_q4_0
            && let Some(t) =
                self.materialise_auf_soa_weight(secondary, layer_idx, subname, primary)?
        {
            return Ok(t);
        }
        self.materialise_tensor(secondary, layer_idx, subname, primary, true)
    }

    /// Build a SOA-backed GPU tensor for an AUF `WEIGHTS_ADRENO_SOA` Q4_0
    /// weight without running the AOS materialisation pipeline and without
    /// allocating a placeholder AOS cl_mem.
    ///
    /// The returned `Tensor` reports the logical AOS shape and `Q4_0` dtype
    /// but its backing buffer is a `NoshuffleWeightBuffer` containing the
    /// `q_buf` / `d_buf` (and optional `q_img`) cl_mem handles. The AUF
    /// SOA payload (`q_bytes` || `d_bytes`) is uploaded directly into those
    /// fresh GPU buffers via blocking `enqueue_write_buffer`; no AOS
    /// placeholder is created (Phase 5 Sprint C
    /// WSWAP-5-AUF-PLACEHOLDER-DROP — saves 112 cl_mem / ~547 MiB on
    /// Llama 3.2 1B ratio=1.0 mixed).
    ///
    /// `Backend::alloc_pre_converted_soa_tensor` also registers a noshuffle
    /// SOA entry against the `d_buf` cl_mem address so the per-layer
    /// `matmul_q4_0` lookup hits in the steady state. The Stage (d)
    /// `restore_pre_converted_soa_registration` re-installs the entry after
    /// the per-batch `invalidate_noshuffle_soa_registry()` call so the
    /// invariant ordering (clear → register → bump) is preserved.
    ///
    /// Returns `Ok(None)` when:
    /// - The tensor is missing from the secondary index (caller falls back
    ///   to the GGUF AOS path which surfaces a proper error).
    /// - `split_pre_converted_soa` rejects the payload (non-Q4_0 dtype or
    ///   size not divisible by 18).
    /// - The size guards inferred from the primary tensor shape do not match
    ///   the AUF payload (defensive — any mismatch triggers AOS fallback).
    /// - The backend's `alloc_pre_converted_soa_tensor` returns `None`
    ///   (CPU / non-OpenCL / driver-side cvt program missing) — caller
    ///   falls back to `materialise_tensor`.
    fn materialise_auf_soa_weight(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
    ) -> Result<Option<Tensor>, SwapError> {
        let info = secondary.layer_tensor(layer_idx, subname).ok_or_else(|| {
            SwapError::SecondaryTensorMissing {
                layer: layer_idx,
                subname: subname.to_string(),
            }
        })?;
        let expected_dims: Vec<usize> = primary.shape().dims().to_vec();
        let sec_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        if sec_rev != expected_dims {
            return Err(SwapError::ShapeMismatch {
                layer: layer_idx,
                subname: subname.to_string(),
                primary: expected_dims,
                secondary: info.dims.clone(),
            });
        }
        // Split AUF payload into q_bytes / d_bytes. None → fall back to AOS path.
        let Some((q_bytes, d_bytes)) = secondary.split_pre_converted_soa(info) else {
            return Ok(None);
        };
        // Total elem count must be QK4_0-aligned; the SOA upload contract
        // requires exactly `num_blocks * 16` (q) + `num_blocks * 2` (d).
        let shape = primary.shape().clone();
        let dims = shape.dims();
        if dims.len() != 2 {
            return Ok(None);
        }
        let total_elems: usize = dims.iter().product();
        if !total_elems.is_multiple_of(32) {
            return Ok(None);
        }
        // Logical: dims[0] = ne01 (rows), dims[1] = ne00 (cols).
        let (ne01, ne00) = (dims[0], dims[1]);
        let num_blocks = ne01 * ne00 / 32;
        if q_bytes.len() != num_blocks * 16 || d_bytes.len() != num_blocks * 2 {
            return Ok(None);
        }

        // Delegate to the backend. This allocates two cl_mem (q_buf, d_buf)
        // + optional image1d view, uploads the SOA bytes verbatim, registers
        // a noshuffle SOA entry keyed on d_buf, and returns a tensor whose
        // backing is a `NoshuffleWeightBuffer` owning the same handles. No
        // placeholder AOS cl_mem is allocated.
        let opt = self
            .backend
            .alloc_pre_converted_soa_tensor(shape, q_bytes, d_bytes, ne00, ne01)
            .map_err(|e| SwapError::BufferAllocationFailed {
                layer: layer_idx,
                source: e,
            })?;
        Ok(opt)
    }

    /// Materialise one weight tensor from the secondary mmap, applying Q/K
    /// row permutation when `subname` identifies a Llama Q/K weight.
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path swap materialise (full)
    fn materialise_tensor(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
        is_weight: bool,
    ) -> Result<Tensor, SwapError> {
        // ── env-gated sub-stage profiling ───────────────────────────────────
        // Set LLMRS_SWAP_PROFILE_BREAKDOWN=1 to enable per-tensor μs timings.
        // When the env var is absent the profiling branch is never entered and
        // `Instant::now()` is never called, so production overhead is zero.
        let prof = std::env::var_os("LLMRS_SWAP_PROFILE_BREAKDOWN")
            .map(|v| v == "1")
            .unwrap_or(false);
        let t_total = if prof { Some(Instant::now()) } else { None };

        // ── sub-stage: lookup ───────────────────────────────────────────────
        let t_lookup = if prof { Some(Instant::now()) } else { None };
        let info = secondary.layer_tensor(layer_idx, subname).ok_or_else(|| {
            SwapError::SecondaryTensorMissing {
                layer: layer_idx,
                subname: subname.to_string(),
            }
        })?;
        let us_lookup = t_lookup.map(|t| t.elapsed().as_micros() as f64 / 1.0);

        // ── sub-stage: dim_check ────────────────────────────────────────────
        let t_dim = if prof { Some(Instant::now()) } else { None };
        let shape = primary.shape().clone();
        let expected_dims: Vec<usize> = primary.shape().dims().to_vec();
        // Secondary dims are stored in reverse order (GGUF innermost-first).
        let sec_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        if sec_rev != expected_dims {
            return Err(SwapError::ShapeMismatch {
                layer: layer_idx,
                subname: subname.to_string(),
                primary: expected_dims,
                secondary: info.dims.clone(),
            });
        }
        let us_dim = t_dim.map(|t| t.elapsed().as_micros() as f64 / 1.0);

        // ── sub-stage: bytes ────────────────────────────────────────────────
        // Raw bytes from the secondary mmap.
        let t_bytes = if prof { Some(Instant::now()) } else { None };
        let data = secondary.tensor_bytes(info);
        let tensor_size = data.len();
        let us_bytes = t_bytes.map(|t| t.elapsed().as_micros() as f64 / 1.0);

        // ── sub-stage: permute ──────────────────────────────────────────────
        // Build a canonical GGUF tensor name for the permutation gate, so we
        // reuse the same predicate as the primary loader without copying
        // regex logic.
        //
        // GGUF secondaries store Q/K weights in the on-disk permuted layout —
        // we must unpermute once. AUF secondaries (both WEIGHTS_ADRENO_SOA
        // and WEIGHTS_CPU_AOS) bake the unpermute step into the build-time
        // `auf_tool::extract_weight_blobs` pipeline, so the on-disk bytes are
        // already unpermuted. Calling `unpermute_qk_rows` here again would
        // double-apply the permutation and produce garbage on the post-swap
        // forward path. The `secondary.needs_qk_unpermute_at_swap()` check
        // distinguishes the two formats (Galaxy S25 Adreno + AOS variant
        // observed regression: 2026-05-01).
        let t_permute = if prof { Some(Instant::now()) } else { None };
        let canonical_name = format!("blk.{layer_idx}.{subname}");
        let needs_unpermute = secondary.needs_qk_unpermute_at_swap()
            && qk_permute_shape(&canonical_name, self.config).is_some();
        let permuted_bytes: Option<Vec<u8>> = if secondary.needs_qk_unpermute_at_swap() {
            if let Some((n_head, head_dim)) = qk_permute_shape(&canonical_name, self.config) {
                let total_rows = shape.dims()[0];
                debug_assert_eq!(total_rows, n_head * head_dim);
                if !data.len().is_multiple_of(total_rows) {
                    return Err(SwapError::ShapeMismatch {
                        layer: layer_idx,
                        subname: subname.to_string(),
                        primary: expected_dims,
                        secondary: info.dims.clone(),
                    });
                }
                let row_size_bytes = data.len() / total_rows;
                Some(unpermute_qk_rows(data, n_head, head_dim, row_size_bytes))
            } else {
                None
            }
        } else {
            // AUF: bytes already in the runtime layout — no transformation.
            None
        };
        let us_permute = t_permute.map(|t| t.elapsed().as_micros() as f64 / 1.0);

        // ── LISWAP-6 alias fast path (qk-unpermute 미발생 시) ───────────────
        // Rpcmem variant 가 alias-capable backend 와 결합되었을 때 H2D copy
        // 자체를 제거. needs_unpermute=true (Llama Q/K) 이거나 backend 미지원
        // 이면 알아서 standard path 로 fall through.
        let is_cpu_backend_early = self.backend.name().contains("CPU");
        #[cfg(feature = "opencl")]
        if !needs_unpermute
            && !is_cpu_backend_early
            && let Some(t) =
                self.try_alias_materialise(secondary, layer_idx, subname, info, &shape)?
        {
            // Emit profiling line so the alias path is visible in
            // `LLMRS_SWAP_PROFILE_BREAKDOWN=1` traces.
            if let Some(t_tot) = t_total {
                let us_total = t_tot.elapsed().as_micros() as f32;
                crate::action_diag_helper::log_swap_prof_breakdown(
                    layer_idx,
                    subname,
                    is_weight,
                    tensor_size,
                    us_lookup.unwrap_or(0.0) as f32,
                    us_dim.unwrap_or(0.0) as f32,
                    us_bytes.unwrap_or(0.0) as f32,
                    us_permute.unwrap_or(0.0) as f32,
                    0.0,
                    0.0,
                    0.0,
                    us_total,
                    "rpcmem-alias",
                );
            }
            return Ok(t);
        }
        #[cfg(not(feature = "opencl"))]
        let _ = (needs_unpermute, is_cpu_backend_early);

        // ── sub-stage: wrap ─────────────────────────────────────────────────
        // ENG-ALG-227: When no permutation is needed (AUF path), borrow the
        // mmap bytes directly instead of copying them into a heap Vec.  The
        // BorrowedMmapBuffer clones the secondary Arc (INV-143) so the mmap
        // region remains alive for the duration of the backend upload.
        //
        // When permutation *is* needed (GGUF Q/K row reorder), the bytes have
        // already been written into an owned Vec by `unpermute_qk_rows`, so we
        // wrap that in a SharedBuffer as before.
        let t_wrap = if prof { Some(Instant::now()) } else { None };
        let cpu_buf: Arc<dyn Buffer> = if let Some(owned) = permuted_bytes {
            // Permuted path: owned heap allocation required (bytes were reordered).
            Arc::new(SharedBuffer::from_vec(owned, info.dtype))
        } else {
            // Borrow path (AUF AOS / no permutation): zero-copy mmap borrow.
            Arc::new(crate::memory::host::mmap::MmapBuffer::borrow(
                data,
                info.dtype,
                secondary.clone(),
            ))
        };
        let us_wrap = t_wrap.map(|t| t.elapsed().as_micros() as f64 / 1.0);

        // ── sub-stage: cpu_tensor ───────────────────────────────────────────
        let t_cpu = if prof { Some(Instant::now()) } else { None };
        let cpu_backend = crate::backend::cpu::cpu_singleton();
        let cpu_tensor = Tensor::new(shape.clone(), cpu_buf, cpu_backend);
        let us_cpu = t_cpu.map(|t| t.elapsed().as_micros() as f64 / 1.0);

        let is_cpu = self.backend.name().contains("CPU");
        if is_cpu {
            // Emit profiling line even on early return path.
            if let Some(t_tot) = t_total {
                let us_total = t_tot.elapsed().as_micros() as f32;
                crate::action_diag_helper::log_swap_prof_breakdown(
                    layer_idx,
                    subname,
                    is_weight,
                    tensor_size,
                    us_lookup.unwrap_or(0.0) as f32,
                    us_dim.unwrap_or(0.0) as f32,
                    us_bytes.unwrap_or(0.0) as f32,
                    us_permute.unwrap_or(0.0) as f32,
                    us_wrap.unwrap_or(0.0) as f32,
                    us_cpu.unwrap_or(0.0) as f32,
                    0.0,
                    us_total,
                    "",
                );
            }
            return Ok(cpu_tensor);
        }

        // ── sub-stage: upload ───────────────────────────────────────────────
        // ── LISWAP-3 prototype zero-copy pool path ─────────────────────────
        // When `host_ptr_pool` is attached AND we are uploading a weight
        // tensor AND the OpenCL backend is reachable AND a slot acquire
        // succeeds, replace the staging `copy_weight_from` with a
        // `map(MAP_WRITE) → memcpy → unmap` cycle on a pre-allocated
        // ALLOC_HOST_PTR cl_mem. Any failure path (size overflow, slot
        // exhaustion, missing source pointer) falls back gracefully to the
        // staging path so prototype activation never reduces correctness.
        // Plan: compiled-chasing-hopper.md Direction A track, Stage 3.
        let t_upload = if prof { Some(Instant::now()) } else { None };

        #[cfg(feature = "opencl")]
        if is_weight
            && let Some(pool) = self.host_ptr_pool.as_ref()
            && let Some(t) = self.try_pool_materialise(secondary, &cpu_tensor, pool, layer_idx)?
        {
            if let Some(t_tot) = t_total {
                let us_upload = t_upload
                    .map(|t| t.elapsed().as_micros() as f32)
                    .unwrap_or(0.0);
                let us_total = t_tot.elapsed().as_micros() as f32;
                crate::action_diag_helper::log_swap_prof_breakdown(
                    layer_idx,
                    subname,
                    is_weight,
                    tensor_size,
                    us_lookup.unwrap_or(0.0) as f32,
                    us_dim.unwrap_or(0.0) as f32,
                    us_bytes.unwrap_or(0.0) as f32,
                    us_permute.unwrap_or(0.0) as f32,
                    us_wrap.unwrap_or(0.0) as f32,
                    us_cpu.unwrap_or(0.0) as f32,
                    us_upload,
                    us_total,
                    "",
                );
            }
            return Ok(t);
        }

        let result = if is_weight {
            self.backend.copy_weight_from(&cpu_tensor).map_err(|e| {
                SwapError::BufferAllocationFailed {
                    layer: layer_idx,
                    source: e,
                }
            })
        } else {
            self.backend
                .copy_from(&cpu_tensor)
                .map_err(|e| SwapError::BufferAllocationFailed {
                    layer: layer_idx,
                    source: e,
                })
        };

        if let Some(t_tot) = t_total {
            let us_upload = t_upload
                .map(|t| t.elapsed().as_micros() as f32)
                .unwrap_or(0.0);
            let us_total = t_tot.elapsed().as_micros() as f32;
            crate::action_diag_helper::log_swap_prof_breakdown(
                layer_idx,
                subname,
                is_weight,
                tensor_size,
                us_lookup.unwrap_or(0.0) as f32,
                us_dim.unwrap_or(0.0) as f32,
                us_bytes.unwrap_or(0.0) as f32,
                us_permute.unwrap_or(0.0) as f32,
                us_wrap.unwrap_or(0.0) as f32,
                us_cpu.unwrap_or(0.0) as f32,
                us_upload,
                us_total,
                "",
            );
        }

        result
    }

    /// LISWAP-3 prototype zero-copy pool materialise. Returns `Ok(Some)` when
    /// the pool path succeeded, `Ok(None)` when the caller should fall back
    /// to staging (slot exhaustion / size overflow / non-OpenCL backend),
    /// `Err` when an OpenCL operation hard-failed.
    ///
    /// Mmap region lifetime: `cpu_tensor` carries the secondary's `Arc<Buffer>`
    /// (either `BorrowedMmapBuffer` or `SharedBuffer` for the permuted path).
    /// Pulling that Arc into the resulting `HostPtrPoolBuffer` is unnecessary
    /// because the fill helper memcpy's bytes into the slot synchronously
    /// — the secondary mmap is no longer needed after `fill_host_ptr_buffer`
    /// returns. We still clone the SecondaryMmap Arc as a defence-in-depth
    /// guard so callers analysing buffer lifetimes see the same shape as the
    /// `BorrowedMmapBuffer` path.
    #[cfg(feature = "opencl")]
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path pool materialise
    fn try_pool_materialise(
        &self,
        secondary: &Arc<SecondaryMmap>,
        cpu_tensor: &Tensor,
        pool: &Arc<crate::backend::opencl::host_ptr_pool::HostPtrPool>,
        layer_idx: usize,
    ) -> Result<Option<Tensor>, SwapError> {
        // Backend must be OpenCL for the pool path to be meaningful.
        let Some(ocl_be) = self
            .backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
        else {
            return Ok(None);
        };
        let size = cpu_tensor.size();
        if size == 0 {
            return Ok(None);
        }
        // Source host pointer: produced by `BorrowedMmapBuffer` or
        // `SharedBuffer`; both are non-null on the AOS materialise path.
        let src_ptr = cpu_tensor.buffer().as_ptr();
        if src_ptr.is_null() {
            return Ok(None);
        }
        let Some(guard) = pool.acquire(size) else {
            // Slot exhaustion or size overflow — staging fallback.
            return Ok(None);
        };
        // Path priority for filling the slot:
        //   (a) Multi-context swap (LLMRS_OPENCL_SWAP_CONTEXT=1): fill via
        //       the swap queue's Map/Unmap on the swap-context cl_mem. The
        //       main forward queue is not touched, isolating swap from
        //       forward kernels at the driver-scheduling level (hypothesis
        //       under test).
        //   (b) Single-context DMA-BUF (LLMRS_OPENCL_DMABUF_HEAP=1): direct
        //       CPU memcpy to the mmap'd DMA-BUF host pointer. No OpenCL
        //       Map/Unmap, no `clFinish`.
        //   (c) Plain ALLOC_HOST_PTR fallback: Map/Unmap on the main queue.
        if let Some(swap_mem) = guard.swap_ctx_mem() {
            // Multi-context path. The slot's main-context `cl_mem`
            // (`guard.mem()`) is the read view used by forward kernels;
            // the swap-context `cl_mem` (`swap_mem`) is the write view
            // bound to the swap queue. Both alias the same DMA-BUF.
            let host_ptr = guard.dmabuf_host_ptr().unwrap_or(std::ptr::null_mut());
            // SAFETY: `swap_mem` is a DMA-BUF-backed cl_mem in the swap
            // queue's context (paired with the main-context `mem()` via the
            // shared FD); `src_ptr` is valid for `size` bytes for the
            // lifetime of `cpu_tensor`; `size <= pool.max_tensor_size`.
            unsafe { ocl_be.fill_dmabuf_via_swap_queue(swap_mem, host_ptr, src_ptr, size) }
                .map_err(|e| SwapError::BufferAllocationFailed {
                    layer: layer_idx,
                    source: e,
                })?;
            // Optional explicit cache flush (LLMRS_DMABUF_SYNC=1).
            if let Some(fd) = guard.dmabuf_fd() {
                ocl_be.ioctl_dmabuf_sync_write_if_enabled(fd);
            }
        } else if let Some(dmabuf_ptr) = guard.dmabuf_host_ptr() {
            // SAFETY: `src_ptr` is valid for `size` bytes for the lifetime
            // of `cpu_tensor`; `dmabuf_ptr` is an mmap'd DMA-BUF region of
            // `pool.max_tensor_size` bytes (>= `size`); the pool's
            // `in_use` flag enforces single-writer access for this slot.
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, dmabuf_ptr as *mut u8, size);
            }
            if let Some(fd) = guard.dmabuf_fd() {
                ocl_be.ioctl_dmabuf_sync_write_if_enabled(fd);
            }
        } else {
            // SAFETY: `src_ptr` is valid for `size` bytes for the lifetime of
            // `cpu_tensor` (which we hold a reference to here); `guard.mem()`
            // is an `ALLOC_HOST_PTR`-allocated cl_mem of `pool.max_tensor_size`
            // bytes (>= `size`) created via `alloc_host_ptr_buffer_empty`.
            unsafe { ocl_be.fill_host_ptr_buffer(guard.mem(), src_ptr, size) }.map_err(|e| {
                SwapError::BufferAllocationFailed {
                    layer: layer_idx,
                    source: e,
                }
            })?;
        }
        let dtype = cpu_tensor.dtype();
        let mmap_guard: Arc<dyn crate::memory::host::mmap::MmapKeepAlive> = secondary.clone();
        let buf: Arc<dyn crate::buffer::Buffer> = Arc::new(
            crate::backend::opencl::host_ptr_pool_buffer::HostPtrPoolBuffer::new(
                guard,
                size,
                dtype,
                Some(mmap_guard),
            ),
        );
        let tensor = Tensor::new(cpu_tensor.shape().clone(), buf, Arc::clone(&self.backend));
        Ok(Some(tensor))
    }

    /// Materialise a norm tensor if present in the secondary file; otherwise
    /// fall back to cloning the primary tensor. Norms are rarely stored
    /// differently across GGUFs so the fallback is the common path.
    fn clone_or_materialise_norm(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
    ) -> Result<Tensor, SwapError> {
        if secondary.layer_tensor(layer_idx, subname).is_some() {
            self.materialise_tensor(secondary, layer_idx, subname, primary, false)
        } else {
            Ok(primary.clone())
        }
    }

    /// WSWAP-5-PRIMARY-DROP: explicitly release the primary weight cl_mem
    /// backing the `LayerWeights` we just displaced.
    ///
    /// `old_layer` is moved by value (consumed via `Arc::try_unwrap`), so
    /// dropping it releases every owned `Arc<dyn Buffer>` whose refcount
    /// reaches zero. For OpenCL backends this triggers the
    /// `OpenCLBuffer::drop` destructor which calls `clReleaseMemObject` and
    /// returns the GPU memory to the driver. For CPU backends this is a
    /// regular heap free.
    ///
    /// Behaviour summary:
    /// - On Adreno UMA + AUF SOA bypass at ratio=1.0: each swapped layer
    ///   releases 7 F16 weight cl_mem (~17 MiB / layer for Llama 3.2 1B).
    ///   Total: ~16 layers × 7 × 17 MiB ≈ 2.0 GiB returned.
    /// - On CPU backend / host tests: heap free; behaviourally identical.
    /// - Norms (`attention_norm`, `ffn_norm`) are intentionally also
    ///   dropped — the new layer always materialises fresh norms either
    ///   from the secondary mmap or by cloning, so the old ones are
    ///   redundant. Optional Gemma3 norms (`q_norm`, `k_norm`,
    ///   `pre_ffn_norm`, `post_ffn_norm`) and `qkv_bias` are also released;
    ///   the new layer rebuilds them from the secondary or clones them.
    ///
    /// Diagnostic: emits a single `weight_swap_released` bucket count via
    /// `record_cl_mem_release` so a `LLMRS_CL_MEM_DIAG=1` dump can confirm
    /// the release path executed (replaces the implicit "destructor was
    /// not instrumented" gap noted in `phase_5_tbt_diag.md`).
    fn release_primary_weights(backend: &Arc<dyn Backend>, old_layer: LayerWeights) {
        // Tally per-tensor sizes before drop so the diagnostic tracks the
        // real reclaimed footprint regardless of which buffer concrete type
        // backed each tensor (OpenCLBuffer / UnifiedBuffer / SharedBuffer /
        // MmapBuffer / NoshuffleWeightBuffer).
        let mut released_bytes: usize = 0;
        let mut released_count: usize = 0;
        let mut tally = |t: &Tensor| {
            released_bytes += t.size();
            released_count += 1;
        };
        tally(&old_layer.wq);
        tally(&old_layer.wk);
        tally(&old_layer.wv);
        tally(&old_layer.wo);
        tally(&old_layer.w_gate);
        tally(&old_layer.w_up);
        tally(&old_layer.w_down);
        // Norms + biases are smaller but still counted so the dump reflects
        // the full primary layer cost. Most paths will leave these on F32
        // primaries (they aren't quantised in the secondary), so tracking
        // is informational rather than load-bearing.
        tally(&old_layer.attention_norm);
        tally(&old_layer.ffn_norm);
        if let Some(bias) = &old_layer.qkv_bias {
            tally(&bias.bq);
            tally(&bias.bk);
            tally(&bias.bv);
        }
        if let Some(t) = &old_layer.q_norm {
            tally(t);
        }
        if let Some(t) = &old_layer.k_norm {
            tally(t);
        }
        if let Some(t) = &old_layer.pre_ffn_norm {
            tally(t);
        }
        if let Some(t) = &old_layer.post_ffn_norm {
            tally(t);
        }
        // Tensor partition state on the OLD layer points at obsolete cl_mem
        // (the new layer uses `partition_ctx: None` per `build_layer_from_mmap`),
        // so dropping `old_layer` here also reclaims any partition slice
        // buffers — accounted under the same bucket for simplicity.

        // Drop fires destructors on every Arc<dyn Buffer> whose refcount
        // reaches zero. For OpenCL backends this is `clReleaseMemObject`.
        drop(old_layer);

        // Diagnostic charge — only OpenCLBackend exposes the diag hook;
        // CPU/CUDA backends quietly skip via the `as_any` downcast.
        record_swap_release(backend, released_count, released_bytes);
    }

    /// Advise the kernel that the primary pages backing `old` can be
    /// reclaimed. Safe only when we hold the last reference — otherwise an
    /// in-flight forward still uses the tensor. Best-effort, never panics.
    fn madvise_if_exclusive(old: &Arc<LayerWeights>) {
        // madvise(MADV_DONTNEED) is Linux-only; the whole guard is gated to
        // that target (on others this is a no-op).
        #[cfg(target_os = "linux")]
        {
            // strong_count is racy relative to readers that have not yet
            // cloned the Arc, but any such reader must have observed the new
            // snapshot (via `slot.load_weights()` after our `swap_weights`) to
            // be newly entering the forward path. Readers that saw the old
            // snapshot before the swap keep strong_count > 1 here — skip madvise
            // so an in-flight forward keeps its pages.
            if Arc::strong_count(old) > 1 {
                return;
            }
            Self::madvise_tensor(&old.wq);
            Self::madvise_tensor(&old.wk);
            Self::madvise_tensor(&old.wv);
            Self::madvise_tensor(&old.wo);
            Self::madvise_tensor(&old.w_gate);
            Self::madvise_tensor(&old.w_up);
            Self::madvise_tensor(&old.w_down);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = old;
    }

    #[cfg(target_os = "linux")]
    fn madvise_tensor(t: &Tensor) {
        // Only MmapBuffer-backed or host-managed buffers are amenable to
        // MADV_DONTNEED. Arbitrary device-only cl_mem allocations return a
        // null pointer and are skipped. See ENG-ALG-C05/C06.
        let ptr = t.buffer().as_ptr();
        if ptr.is_null() {
            return;
        }
        let len = t.size();
        if len == 0 {
            return;
        }
        // Align to a page boundary to satisfy the madvise contract; trim
        // head/tail partial pages rather than round out and stomp a sibling
        // tensor.
        let page = 4096usize;
        let addr = ptr as usize;
        let aligned = addr.div_ceil(page) * page;
        let tail = (addr + len) & !(page - 1);
        if tail <= aligned {
            return;
        }
        let adv_len = tail - aligned;
        // SAFETY: address range [aligned, aligned+adv_len) is a subset of
        // [addr, addr+len) which is live for the lifetime of `t`. MADV_DONTNEED
        // is a hint and failure is non-fatal.
        unsafe {
            let _ = libc::madvise(aligned as *mut libc::c_void, adv_len, libc::MADV_DONTNEED);
        }
    }
}

/// WSWAP-5-PRIMARY-DROP: charge the OpenCL diagnostic counter for primary
/// weight cl_mem released via `release_primary_weights`. NoOp on CPU /
/// CUDA backends and on builds without `LLMRS_CL_MEM_DIAG=1`.
///
/// Records a single bucket entry `weight_swap_released` regardless of which
/// concrete buffer type backed each tensor; the per-tensor classification
/// already lives in the alloc-side counters (`weight_f16_copy`,
/// `weight_q4_aos_copy`, etc.). The release-side tally lets a diag run
/// produce a closed-loop "alloc - release" picture without per-destructor
/// instrumentation (Sprint B `phase_5_tbt_diag.md` over-reported alive
/// bytes because the dump only counted allocs).
///
/// Public re-export used by `PrimaryReleaseWorker` (ENG-ALG-228) which runs
/// `release_primary_weights` on a background thread and needs to charge the
/// diagnostic from that thread with the same backend reference.
pub fn record_swap_release_pub(backend: &Arc<dyn Backend>, count: usize, bytes: usize) {
    record_swap_release(backend, count, bytes);
}

// LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path swap counter
fn record_swap_release(backend: &Arc<dyn Backend>, count: usize, bytes: usize) {
    if count == 0 {
        return;
    }
    #[cfg(feature = "opencl")]
    {
        if let Some(ocl_be) = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
        {
            // The diag bucket is keyed on a `&'static str`; we record one
            // bucket entry per swap layer with the cumulative byte tally.
            // `record_cl_mem_release` is a no-op when the env var is unset,
            // so we don't pay a HashMap update per swap on production runs.
            for _ in 0..count {
                ocl_be.record_cl_mem_release("weight_swap_released", bytes / count);
            }
            return;
        }
    }
    // CPU / CUDA / non-opencl builds: nothing to record.
    let _ = (backend, bytes);
}

// ── LISWAP-8: Background fetch standalone helpers ──────────────────────────
//
// Free-function variants of `build_layer_from_mmap_async` /
// `materialise_cpu_tensor` / `try_alias_materialise` that own their config
// + backend (no `&'a` borrow of `SwapExecutor`). Designed to be captured by
// `AsyncSwapDispatcher::submit_dispatch_chunk` closures so the entire
// `mmap_permute + cuMemAlloc + cuMemcpyAsync` pipeline runs on the worker
// thread (`llmrs-async-swap`).
//
// Gated by `LLMRS_SWAP_BG_FETCH=1` in `execute_on_slots`. par_seq data
// (2026-05-14) confirmed single-thread background dispatch keeps forward at
// baseline 67 ms (vs par_iter multi-thread +67 ms regression on Jetson UMA).
//
// `host_ptr_pool` is intentionally NOT plumbed through — the async / alias
// paths in `materialise_cpu_tensor` never consult the pool (it only affects
// the sync `materialise_tensor` AOS path).

pub(crate) fn build_layer_async_standalone(
    secondary: &Arc<SecondaryMmap>,
    slot: &LayerSlot,
    layer_idx: usize,
    backend: &Arc<dyn Backend>,
    config: &ModelConfig,
) -> Result<(TransformerLayer, GpuEvent), SwapError> {
    let old = slot.load_weights();

    let enqueue_weight =
        |subname: &str, primary: &Tensor| -> Result<(Tensor, GpuEvent), SwapError> {
            let cpu_tensor = materialise_cpu_tensor_standalone(
                secondary, layer_idx, subname, primary, backend, config,
            )?;
            if backend.name().contains("CPU") {
                return Ok((cpu_tensor, GpuEvent::dummy()));
            }
            if cpu_tensor.buffer().cl_mem().is_some() {
                return Ok((cpu_tensor, GpuEvent::dummy()));
            }
            backend.enqueue_write_async(&cpu_tensor).map_err(|e| {
                SwapError::AsyncTransferUnavailable {
                    layer: layer_idx,
                    source: e,
                }
            })
        };

    let enqueue_norm =
        |subname: &str, primary: &Tensor| -> Result<(Tensor, Option<GpuEvent>), SwapError> {
            if secondary.layer_tensor(layer_idx, subname).is_none() {
                return Ok((primary.clone(), None));
            }
            let cpu_tensor = materialise_cpu_tensor_standalone(
                secondary, layer_idx, subname, primary, backend, config,
            )?;
            if backend.name().contains("CPU") {
                return Ok((cpu_tensor, None));
            }
            if cpu_tensor.buffer().cl_mem().is_some() {
                return Ok((cpu_tensor, None));
            }
            let (gpu_tensor, evt) = backend.enqueue_write_async(&cpu_tensor).map_err(|e| {
                SwapError::AsyncTransferUnavailable {
                    layer: layer_idx,
                    source: e,
                }
            })?;
            Ok((gpu_tensor, Some(evt)))
        };

    let (wq, evt_wq) = enqueue_weight("attn_q.weight", &old.wq)?;
    let (wk, evt_wk) = enqueue_weight("attn_k.weight", &old.wk)?;
    let (wv, evt_wv) = enqueue_weight("attn_v.weight", &old.wv)?;
    let (wo, evt_wo) = enqueue_weight("attn_output.weight", &old.wo)?;
    let (w_gate, evt_gate) = enqueue_weight("ffn_gate.weight", &old.w_gate)?;
    let (w_up, evt_up) = enqueue_weight("ffn_up.weight", &old.w_up)?;
    let (w_down, evt_down) = enqueue_weight("ffn_down.weight", &old.w_down)?;

    let (attention_norm, evt_anorm) = enqueue_norm("attn_norm.weight", &old.attention_norm)?;
    let (ffn_norm, evt_fnorm) = enqueue_norm("ffn_norm.weight", &old.ffn_norm)?;

    let last_event = evt_fnorm.unwrap_or_else(|| evt_anorm.unwrap_or(evt_down));
    let _ = (evt_wq, evt_wk, evt_wv, evt_wo, evt_gate, evt_up);

    Ok((
        TransformerLayer {
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            attention_norm,
            ffn_norm,
            qkv_bias: old.qkv_bias.clone(),
            q_norm: old.q_norm.clone(),
            k_norm: old.k_norm.clone(),
            pre_ffn_norm: old.pre_ffn_norm.clone(),
            post_ffn_norm: old.post_ffn_norm.clone(),
            partition_ctx: None,
        },
        last_event,
    ))
}

/// LISWAP-8 Hammer D: build a `TransformerLayer` whose weight tensors are
/// zero-copy aliases into a CUDA-registered secondary mmap region.
///
/// **PoC**: For GGUF secondaries whose `needs_qk_unpermute_at_swap()` is
/// `true`, the alias bytes for `attn_q` / `attn_k` are still the on-disk
/// permuted layout. The PoC accepts this and measures the performance
/// envelope of a fully zero-copy swap; accuracy fixes (load-time
/// unpermute into a separately-registered owned region) are deferred.
/// For AUF secondaries (already unpermuted at build time) the alias is
/// directly correct.
#[cfg(feature = "cuda-embedded")]
pub(crate) fn build_layer_via_mmap_alias_standalone(
    secondary: &Arc<SecondaryMmap>,
    slot: &LayerSlot,
    layer_idx: usize,
    backend: &Arc<dyn Backend>,
    registration: &Arc<crate::memory::cuda::mmap::CudaMmapRegistration>,
) -> Result<(TransformerLayer, GpuEvent), SwapError> {
    use crate::memory::cuda::mmap::CudaMmapAliasBuffer;

    let old = slot.load_weights();

    let alias_tensor = |subname: &str, primary: &Tensor| -> Result<Tensor, SwapError> {
        let info = secondary.layer_tensor(layer_idx, subname).ok_or_else(|| {
            SwapError::SecondaryTensorMissing {
                layer: layer_idx,
                subname: subname.to_string(),
            }
        })?;
        let shape = primary.shape().clone();
        let expected_dims: Vec<usize> = primary.shape().dims().to_vec();
        let sec_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        if sec_rev != expected_dims {
            return Err(SwapError::ShapeMismatch {
                layer: layer_idx,
                subname: subname.to_string(),
                primary: expected_dims,
                secondary: info.dims.clone(),
            });
        }
        let alias =
            CudaMmapAliasBuffer::new(Arc::clone(registration), info.offset, info.len, info.dtype)
                .map_err(|e| SwapError::BufferAllocationFailed {
                layer: layer_idx,
                source: e,
            })?;
        Ok(Tensor::new(
            shape,
            Arc::new(alias) as Arc<dyn Buffer>,
            Arc::clone(backend),
        ))
    };

    let wq = alias_tensor("attn_q.weight", &old.wq)?;
    let wk = alias_tensor("attn_k.weight", &old.wk)?;
    let wv = alias_tensor("attn_v.weight", &old.wv)?;
    let wo = alias_tensor("attn_output.weight", &old.wo)?;
    let w_gate = alias_tensor("ffn_gate.weight", &old.w_gate)?;
    let w_up = alias_tensor("ffn_up.weight", &old.w_up)?;
    let w_down = alias_tensor("ffn_down.weight", &old.w_down)?;

    let attention_norm = if secondary
        .layer_tensor(layer_idx, "attn_norm.weight")
        .is_some()
    {
        alias_tensor("attn_norm.weight", &old.attention_norm)?
    } else {
        old.attention_norm.clone()
    };
    let ffn_norm = if secondary
        .layer_tensor(layer_idx, "ffn_norm.weight")
        .is_some()
    {
        alias_tensor("ffn_norm.weight", &old.ffn_norm)?
    } else {
        old.ffn_norm.clone()
    };

    Ok((
        TransformerLayer {
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            attention_norm,
            ffn_norm,
            qkv_bias: old.qkv_bias.clone(),
            q_norm: old.q_norm.clone(),
            k_norm: old.k_norm.clone(),
            pre_ffn_norm: old.pre_ffn_norm.clone(),
            post_ffn_norm: old.post_ffn_norm.clone(),
            partition_ctx: None,
        },
        GpuEvent::dummy(),
    ))
}

/// LISWAP-8 Phase B: pool-backed variant of `build_layer_async_standalone`.
///
/// Takes ownership of a pre-allocated `pool_entry: TransformerLayer` whose
/// weight buffers are sized for the secondary dtype. For each weight
/// tensor, materialises the secondary mmap bytes (with Q/K unpermute if
/// needed) into a CPU-side `Tensor`, then calls
/// `Backend::enqueue_write_into_async(pool_entry.<wq>, src_ptr, len)` to
/// overwrite the entry's existing device buffer in place. The returned
/// `TransformerLayer` reuses the pool entry's tensors (no fresh alloc).
///
/// Norms: when the secondary file carries the norm tensor we overwrite
/// the pool entry's norm buffer; otherwise we clone the displaced
/// primary's norm tensor (matching `build_layer_async_standalone`).
pub(crate) fn build_layer_via_pool_standalone(
    secondary: &Arc<SecondaryMmap>,
    slot: &LayerSlot,
    layer_idx: usize,
    backend: &Arc<dyn Backend>,
    config: &ModelConfig,
    pool_entry: TransformerLayer,
) -> Result<(TransformerLayer, GpuEvent), SwapError> {
    let old = slot.load_weights();

    // Tracks the most recent in-order event from the transfer stream/queue —
    // the last successful enqueue covers all prior ones (in-order semantics).
    let mut last_event: GpuEvent = GpuEvent::dummy();

    let mut write_into = |dst: &Tensor, subname: &str, primary: &Tensor| -> Result<(), SwapError> {
        let cpu = materialise_cpu_tensor_standalone(
            secondary, layer_idx, subname, primary, backend, config,
        )?;
        let src_ptr = cpu.buffer().as_ptr();
        if src_ptr.is_null() {
            return Err(SwapError::AsyncTransferUnavailable {
                layer: layer_idx,
                source: anyhow::anyhow!(
                    "pool path: cpu tensor for {subname} has null host pointer"
                ),
            });
        }
        let len = cpu.size();
        let evt = backend
            .enqueue_write_into_async(dst, src_ptr, len)
            .map_err(|e| SwapError::AsyncTransferUnavailable {
                layer: layer_idx,
                source: e,
            })?;
        last_event = evt;
        Ok(())
    };

    write_into(&pool_entry.wq, "attn_q.weight", &old.wq)?;
    write_into(&pool_entry.wk, "attn_k.weight", &old.wk)?;
    write_into(&pool_entry.wv, "attn_v.weight", &old.wv)?;
    write_into(&pool_entry.wo, "attn_output.weight", &old.wo)?;
    write_into(&pool_entry.w_gate, "ffn_gate.weight", &old.w_gate)?;
    write_into(&pool_entry.w_up, "ffn_up.weight", &old.w_up)?;
    write_into(&pool_entry.w_down, "ffn_down.weight", &old.w_down)?;

    let attention_norm = if secondary
        .layer_tensor(layer_idx, "attn_norm.weight")
        .is_some()
    {
        write_into(
            &pool_entry.attention_norm,
            "attn_norm.weight",
            &old.attention_norm,
        )?;
        pool_entry.attention_norm.clone()
    } else {
        old.attention_norm.clone()
    };
    let ffn_norm = if secondary
        .layer_tensor(layer_idx, "ffn_norm.weight")
        .is_some()
    {
        write_into(&pool_entry.ffn_norm, "ffn_norm.weight", &old.ffn_norm)?;
        pool_entry.ffn_norm.clone()
    } else {
        old.ffn_norm.clone()
    };

    Ok((
        TransformerLayer {
            wq: pool_entry.wq,
            wk: pool_entry.wk,
            wv: pool_entry.wv,
            wo: pool_entry.wo,
            w_gate: pool_entry.w_gate,
            w_up: pool_entry.w_up,
            w_down: pool_entry.w_down,
            attention_norm,
            ffn_norm,
            qkv_bias: old.qkv_bias.clone(),
            q_norm: old.q_norm.clone(),
            k_norm: old.k_norm.clone(),
            pre_ffn_norm: old.pre_ffn_norm.clone(),
            post_ffn_norm: old.post_ffn_norm.clone(),
            partition_ctx: None,
        },
        last_event,
    ))
}

// LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path standalone materialise
fn materialise_cpu_tensor_standalone(
    secondary: &Arc<SecondaryMmap>,
    layer_idx: usize,
    subname: &str,
    primary: &Tensor,
    backend: &Arc<dyn Backend>,
    config: &ModelConfig,
) -> Result<Tensor, SwapError> {
    let info = secondary.layer_tensor(layer_idx, subname).ok_or_else(|| {
        SwapError::SecondaryTensorMissing {
            layer: layer_idx,
            subname: subname.to_string(),
        }
    })?;

    let shape = primary.shape().clone();
    let expected_dims: Vec<usize> = primary.shape().dims().to_vec();
    let sec_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
    if sec_rev != expected_dims {
        return Err(SwapError::ShapeMismatch {
            layer: layer_idx,
            subname: subname.to_string(),
            primary: expected_dims,
            secondary: info.dims.clone(),
        });
    }

    let data = secondary.tensor_bytes(info);
    let canonical_name = format!("blk.{layer_idx}.{subname}");
    let needs_unpermute = secondary.needs_qk_unpermute_at_swap()
        && qk_permute_shape(&canonical_name, config).is_some();
    let permuted_bytes: Option<Vec<u8>> = if secondary.needs_qk_unpermute_at_swap() {
        if let Some((n_head, head_dim)) = qk_permute_shape(&canonical_name, config) {
            let total_rows = shape.dims()[0];
            debug_assert_eq!(total_rows, n_head * head_dim);
            if !data.len().is_multiple_of(total_rows) {
                return Err(SwapError::ShapeMismatch {
                    layer: layer_idx,
                    subname: subname.to_string(),
                    primary: expected_dims,
                    secondary: info.dims.clone(),
                });
            }
            let row_size_bytes = data.len() / total_rows;
            Some(unpermute_qk_rows(data, n_head, head_dim, row_size_bytes))
        } else {
            None
        }
    } else {
        None
    };

    #[cfg(feature = "opencl")]
    if !needs_unpermute
        && let Some(t) =
            try_alias_materialise_standalone(secondary, layer_idx, subname, info, &shape, backend)?
    {
        return Ok(t);
    }
    #[cfg(not(feature = "opencl"))]
    let _ = needs_unpermute;

    let cpu_buf: Arc<dyn Buffer> = if let Some(owned) = permuted_bytes {
        Arc::new(SharedBuffer::from_vec(owned, info.dtype))
    } else {
        Arc::new(crate::memory::host::mmap::MmapBuffer::borrow(
            data,
            info.dtype,
            secondary.clone(),
        ))
    };

    let cpu_backend = crate::backend::cpu::cpu_singleton();
    let _ = backend;
    Ok(Tensor::new(shape, cpu_buf, cpu_backend))
}

#[cfg(feature = "opencl")]
fn try_alias_materialise_standalone(
    secondary: &Arc<SecondaryMmap>,
    layer_idx: usize,
    subname: &str,
    info: &SecondaryTensorInfo,
    shape: &crate::shape::Shape,
    backend: &Arc<dyn Backend>,
) -> Result<Option<Tensor>, SwapError> {
    let SecondaryMmap::Rpcmem(rpc) = secondary.as_ref() else {
        return Ok(None);
    };

    if let Some(alias_buf) = rpc.cached_alias(layer_idx, subname) {
        debug_assert_eq!(alias_buf.size(), info.len, "cached alias length mismatch");
        debug_assert_eq!(alias_buf.dtype(), info.dtype, "cached alias dtype mismatch");
        return Ok(Some(Tensor::new(
            shape.clone(),
            alias_buf,
            Arc::clone(backend),
        )));
    }

    let entry =
        rpc.host_ptr_for(layer_idx, subname)
            .map_err(|e| SwapError::BufferAllocationFailed {
                layer: layer_idx,
                source: e,
            })?;
    let Some(alias) = entry else {
        return Ok(None);
    };
    debug_assert_eq!(alias.len, info.len, "rpcmem region length mismatch");
    debug_assert_eq!(alias.dtype, info.dtype, "rpcmem region dtype mismatch");

    // SAFETY: `alias.host_ptr` + `alias.offset..+alias.len` is the rpcmem
    // region returned by `host_ptr_for`, length/dtype validated by the
    // debug_assert pair above. Lifetime is pinned by `Arc::clone(secondary)`
    // (whole store) + `alias.region` (this layer's rpcmem allocation), both
    // moved into the alias buffer per `Backend::alloc_alias_weight_buffer`
    // contract.
    let secondary_dyn: Arc<dyn crate::memory::host::mmap::MmapKeepAlive> = secondary.clone();
    let region_dyn: Arc<dyn crate::memory::secondary::RpcmemRegionGuard> = alias.region;
    let buf = unsafe {
        backend.alloc_alias_weight_buffer(
            alias.host_ptr,
            alias.offset,
            alias.len,
            alias.dtype,
            secondary_dyn,
            region_dyn,
        )
    }
    .map_err(|e| SwapError::BufferAllocationFailed {
        layer: layer_idx,
        source: e,
    })?;
    let Some(alias_buf) = buf else {
        return Ok(None);
    };
    Ok(Some(Tensor::new(
        shape.clone(),
        alias_buf,
        Arc::clone(backend),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_target_layers_matches_spec() {
        // Spec ENG-ALG-213 uniform fallback shape: ceil(ratio * n) layers,
        // evenly spaced. These are not-yet-plan-integration shape tests to
        // guard against refactors that silently change the selection.
        assert!(SwapExecutor::uniform_target_layers(0.0, 16).is_empty());
        assert_eq!(
            SwapExecutor::uniform_target_layers(1.0, 16).len(),
            16,
            "ratio=1.0 selects all layers"
        );
        assert_eq!(SwapExecutor::uniform_target_layers(0.25, 16).len(), 4);
        assert_eq!(SwapExecutor::uniform_target_layers(0.5, 16).len(), 8);
        // Degenerate size.
        assert!(SwapExecutor::uniform_target_layers(0.5, 0).is_empty());
    }

    // INV-126: dtype_tag_to_dtype must accept Q4_0 and reject all reserved variants.

    #[test]
    fn dtype_tag_q4_0_maps_to_dtype_q4_0() {
        let result = dtype_tag_to_dtype(DtypeTag::Q4_0);
        assert!(
            matches!(result, Ok(DType::Q4_0)),
            "Q4_0 should be executable"
        );
    }

    #[test]
    fn dtype_tag_f16_is_unsupported() {
        let result = dtype_tag_to_dtype(DtypeTag::F16);
        assert!(
            matches!(result, Err(SwapError::UnsupportedDtype(DtypeTag::F16))),
            "F16 is reserved, must return UnsupportedDtype"
        );
    }

    #[test]
    fn dtype_tag_f32_is_unsupported() {
        let result = dtype_tag_to_dtype(DtypeTag::F32);
        assert!(
            matches!(result, Err(SwapError::UnsupportedDtype(DtypeTag::F32))),
            "F32 is reserved, must return UnsupportedDtype"
        );
    }

    #[test]
    fn dtype_tag_q8_0_is_unsupported() {
        let result = dtype_tag_to_dtype(DtypeTag::Q8_0);
        assert!(
            matches!(result, Err(SwapError::UnsupportedDtype(DtypeTag::Q8_0))),
            "Q8_0 is reserved, must return UnsupportedDtype"
        );
    }

    #[test]
    fn unsupported_dtype_error_display_contains_inv_126() {
        let err = dtype_tag_to_dtype(DtypeTag::F16).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("INV-126"),
            "Error message should reference INV-126, got: {msg}"
        );
    }

    #[test]
    fn stage_breakdown_log_line_includes_prefault_stage() {
        // WSWAP-5-COLD-UNIFORM: the stage breakdown log line must surface the
        // prefault stage so device measurements can plot it next to the other
        // stages. Refactors that drop the prefix will trip this guard.
        let stages = StageBreakdown {
            prefault_ms: 12.34,
            mmap_permute_ms: 100.0,
            arc_swap_ms: 0.5,
            madvise_ms: 0.1,
            synchronize_ms: 0.7,
            soa_reconvert_ms: 50.0,
            gen_bump_ms: 0.2,
        };
        let line = stages.to_log_line();
        assert!(
            line.starts_with("prefault=12.3ms"),
            "log line must lead with prefault stage, got: {line}"
        );
        assert!(
            line.contains("mmap_permute=100.0ms"),
            "log line must contain mmap_permute stage, got: {line}"
        );
        assert!(
            line.contains("synchronize=0.7ms"),
            "log line must contain synchronize stage (ENG-ALG-231), got: {line}"
        );
    }

    #[test]
    fn stage_breakdown_prefault_default_zero() {
        // Default-constructed StageBreakdown (e.g. CPU host tests, no-op
        // secondaries) must report prefault=0.0 so downstream parsers do not
        // see a missing field.
        let stages = StageBreakdown::default();
        assert_eq!(stages.prefault_ms, 0.0);
    }

    // ── WSWAP-5-PRIMARY-DROP: explicit primary cl_mem release ─────────────
    //
    // The Sprint C-2 contract is: after `LayerSlot::swap_weights` returns
    // the previous Arc, the swap path can `try_unwrap` it deterministically
    // so the tensor destructors fire and the primary `cl_mem` is reclaimed.
    // The host-side proxy for this behaviour is the buffer Arc strong count:
    // each old weight tensor's `Arc<dyn Buffer>` must reach refcount 1
    // (the LayerWeights itself is the sole holder) when the swap path takes
    // ownership. Combined with `Arc::try_unwrap` on the LayerWeights, this
    // implies every backing buffer is uniquely owned by the dispatcher and
    // its destructor will run on `drop(layer)`.

    use crate::backend::Backend;
    use crate::layers::transformer_layer::TransformerLayer;
    use crate::memory::host::shared::SharedBuffer;
    use crate::models::weights::LayerSlot;
    use crate::shape::Shape;
    use std::sync::Arc;

    fn build_test_layer(be: &Arc<dyn Backend>) -> TransformerLayer {
        // Build a minimal `TransformerLayer` whose weight tensors each carry
        // a fresh `Arc<dyn Buffer>` so we can probe refcounts after a swap.
        let make = |sz: usize| -> Tensor {
            let buf: Arc<dyn crate::buffer::Buffer> = Arc::new(SharedBuffer::new(sz, DType::F32));
            Tensor::new(Shape::new(vec![sz / 4]), buf, be.clone())
        };
        TransformerLayer {
            wq: make(32),
            wk: make(32),
            wv: make(32),
            wo: make(32),
            w_gate: make(32),
            w_up: make(32),
            w_down: make(32),
            attention_norm: make(8),
            ffn_norm: make(8),
            qkv_bias: None,
            q_norm: None,
            k_norm: None,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            partition_ctx: None,
        }
    }

    #[test]
    fn slot_swap_weights_returns_previous_arc_uniquely_owned() {
        // ENG-ALG-211 step (b/c) refined contract: the returned Arc has
        // refcount==1 (no extant snapshots, swap dispatcher runs strictly
        // between forwards). `Arc::try_unwrap` therefore succeeds on the
        // production path and the inner `LayerWeights` can be dropped to
        // release every buffer Arc. This mirrors the WSWAP-5-PRIMARY-DROP
        // primary cl_mem release semantics on the device.
        let be: Arc<dyn Backend> = crate::backend::cpu::cpu_singleton();
        let initial = Arc::new(build_test_layer(&be));
        let slot = LayerSlot::new(
            (*initial).clone(),
            DType::F32,
            None, // No secondary mmap; not exercised by this test.
            0,
        );
        // Capture the wq buffer Arc from the initial install.
        let initial_wq_buf = Arc::clone(slot.load_weights().wq.buffer());
        let initial_wq_count_before = Arc::strong_count(&initial_wq_buf);
        drop(initial); // release builder snapshot — slot still holds one ref

        // Pre-swap: slot holds one Arc<LayerWeights>; load_weights briefly
        // borrows another, but we drop it immediately.
        // After swap: returned Arc must be the only outstanding reference.
        let new_layer = build_test_layer(&be);
        let returned = slot.swap_weights(Arc::new(new_layer), DType::F32);
        assert_eq!(
            Arc::strong_count(&returned),
            1,
            "WSWAP-5-PRIMARY-DROP contract: swap_weights must return the \
             previous Arc with strong_count==1 so the dispatcher can \
             try_unwrap and release primary cl_mem deterministically",
        );

        // try_unwrap must succeed.
        let inner = match Arc::try_unwrap(returned) {
            Ok(layer) => layer,
            Err(_) => panic!("Arc::try_unwrap must succeed on a uniquely-owned snapshot"),
        };

        // After try_unwrap and before drop, the wq buffer Arc has 2 refs
        // (inner.wq.buffer + the local capture above).
        let post_unwrap = Arc::strong_count(&initial_wq_buf);
        assert_eq!(
            post_unwrap, 2,
            "buffer Arc held by both inner LayerWeights and the local capture"
        );

        // Drop inner — every buffer Arc decrements to 1 (only the local
        // capture remains for `wq`). Other tensors had no external capture
        // so their buffer Arcs go to 0 and the destructors run.
        drop(inner);
        assert_eq!(
            Arc::strong_count(&initial_wq_buf),
            1,
            "after drop(inner): only the local probe holds the wq buffer; \
             a real OpenCL buffer would now be reclaimed",
        );
        // initial_wq_count_before is informational; assert it sanity-checks.
        assert!(initial_wq_count_before >= 1);
    }

    #[test]
    fn release_primary_weights_runs_destructors() {
        // Direct test of the SwapExecutor::release_primary_weights helper:
        // dropping the LayerWeights must drop every contained Tensor and
        // therefore decrement every buffer Arc. We verify by capturing weak
        // references to the buffers and ensuring they all upgrade to None
        // after the helper consumes the layer.
        use std::sync::Weak;

        let be: Arc<dyn Backend> = crate::backend::cpu::cpu_singleton();
        let layer = build_test_layer(&be);
        // Capture weak refs to the 7 weight buffers.
        let weaks: Vec<Weak<dyn crate::buffer::Buffer>> = [
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &layer.wo,
            &layer.w_gate,
            &layer.w_up,
            &layer.w_down,
        ]
        .iter()
        .map(|t| Arc::downgrade(t.buffer()))
        .collect();

        SwapExecutor::release_primary_weights(&be, layer);

        for (i, w) in weaks.iter().enumerate() {
            assert!(
                w.upgrade().is_none(),
                "WSWAP-5-PRIMARY-DROP: buffer {i} must be fully released \
                 after release_primary_weights — Weak::upgrade returning \
                 Some implies a lingering Arc reference",
            );
        }
    }
}
