//! Migration Step 3-B (V-27): `WeightStagingPool` trait — abstraction over
//! pre-allocated [`TransformerLayer`] suppliers used by the swap path.
//!
//! Decouples [`crate::models::weights::SwapExecutor`] from the concrete
//! `LayerObjectPool` impl so the BG-fetch dispatch path can take entries
//! through a trait object. Enables:
//! - Mocking the pool in unit tests (DIP).
//! - Adding alternate pool strategies (CPU staging, OpenCL host_ptr) without
//!   touching the executor (OCP).
//!
//! The trait lives in `L3-inference` alongside `TransformerLayer` so the
//! `take()` return type is import-natural. The concrete `LayerObjectPool`
//! impl stays in `models/weights/` (also L3-inference) — no file relocation
//! was needed to resolve V-27; the downcast to `CudaBackend` was replaced
//! by [`crate::backend::Backend::bind_current_thread`].

use crate::layers::transformer_layer::TransformerLayer;

/// Supplier of pre-allocated `TransformerLayer` instances drained on demand
/// by the swap executor's BG-fetch dispatch path.
///
/// Implementations are expected to refill in the background; `take()` is
/// non-blocking and returns `None` when the pool is exhausted (the caller
/// then falls back to inline allocation).
pub trait WeightStagingPool: Send + Sync {
    /// Pop one pre-allocated layer. Non-blocking. Returns `None` when the
    /// pool is empty — callers should fall back to inline allocation.
    fn take(&self) -> Option<TransformerLayer>;

    /// Current ready depth — diagnostics only.
    fn depth(&self) -> usize;

    /// Steady-state depth the background refill aims for — diagnostics only.
    fn target_depth(&self) -> usize;
}
