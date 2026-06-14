//! `RuntimeResourcesAccess` traits — cross-L3 vocabulary inversion
//! (§13.8-O 본질 해소, design doc `arch/weights_pressure_split.md §7.4` 후속 sprint).
//!
//! Inference L3 (`TransformerModel`) 가 pressure-owned 자원
//! (`QuantNoiseTable` / `PrimaryReleaseWorker`) 의 concrete type 을 field 로
//! 직접 보유하던 패턴을 trait object 로 추상화. struct 정의에서 pressure
//! 타입이 사라져 cross-L3 vocabulary marker 가 자연 해소된다.
//!
//! 위치 결정: 본 trait 들은 inference owner ↔ pressure impl 을 잇는
//! cross-cutting 추상화라 top-level L2 file 에 정의 (`backend.rs` /
//! `kv_cache_ops.rs` / `layer_boundary_hook.rs` 와 같은 위계).

use std::time::Duration;

// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O cross-cutting trait surface 가 inference payload (LayerWeights) 를 받기 위한 signature 정합 (kv_cache_ops 의 Tensor 노출과 같은 위계).
use crate::models::weights::LayerWeights;

/// Per-layer quantization noise factor table accessor (ENG-DAT-095).
///
/// Inference owner (`TransformerModel`) holds an `Arc<dyn QuantNoiseAccess>`
/// installed via `weight::setup_runtime_resources`. Consumers
/// (`WeightSwapDecider`, `compute_qcf_weight_swap`, `QcfHelpers`) accept
/// `&dyn QuantNoiseAccess` to avoid coupling to the concrete
/// `QuantNoiseTable` type.
pub trait QuantNoiseAccess: Send + Sync {
    /// ε for `layer_id` if computed; `None` when no secondary is present
    /// or the layer index is out of range.
    fn epsilon(&self, layer_id: usize) -> Option<f32>;

    /// Number of per-layer ε entries. `0` when secondary mmap is absent.
    fn len(&self) -> usize;

    /// `true` when ε was computed from secondary tensors (Frobenius); `false`
    /// for `empty()` or `uniform_ones()` fallbacks.
    fn is_computed(&self) -> bool;

    /// Raw slice view of the per-layer ε array (length == `len()`).
    fn as_slice(&self) -> &[f32];

    /// Convenience: `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Asynchronous primary cl_mem drop worker accessor (ENG-ALG-228 /
/// ENG-DAT-100).
///
/// Inference owner holds an `Arc<dyn ReleaseWorkerAccess>` installed by
/// `weight::setup_runtime_resources`. Both inference observers
/// (`swap_dispatch`) and pressure consumers (`SwapExecutor`) interact via
/// this trait object so the inference struct definition does not surface
/// the concrete `PrimaryReleaseWorker` type.
pub trait ReleaseWorkerAccess: Send + Sync {
    /// Enqueue a `LayerWeights` for asynchronous drop on the worker thread.
    ///
    /// `pending_count()` is incremented before the dispatch; on send failure
    /// the worker logs and decrements without panicking.
    fn enqueue_release(&self, layer: LayerWeights);

    /// Number of drop jobs still in flight (INV-141 observation).
    fn pending_count(&self) -> usize;

    /// Block until all pending jobs complete or `deadline` elapses.
    /// `Err(DrainError)` on timeout (used by `SwapExecutor::execute_on_slots`
    /// to enforce INV-141 before starting a new swap batch).
    fn drain(&self, deadline: Duration) -> Result<(), DrainError>;
}

/// Error returned by [`ReleaseWorkerAccess::drain`] on timeout.
///
/// Defined here so the cross-cutting trait is self-contained — downstream
/// consumers (`SwapExecutor`, `async_swap`) import `DrainError` from
/// `crate::runtime_resources_access` instead of the pressure-side
/// `release_worker` module.
#[derive(Debug)]
pub struct DrainError {
    pub pending: usize,
    pub timeout_ms: u64,
}

impl std::fmt::Display for DrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "primary release worker drain timeout: {} jobs remaining after {}ms",
            self.pending, self.timeout_ms
        )
    }
}

impl std::error::Error for DrainError {}
