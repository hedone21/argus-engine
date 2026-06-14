// ── LayerBoundaryHook trait (L2 격상) ────────────────────────────────────────
//!
//! `TransformerModel::forward_into`의 layer loop에서 layer i compute 직후,
//! layer i+1 진입 직전에 호출되는 trait (ENG-ALG-235).
//!
//! 위치 근거: L2 공유 어휘 — inference(transformer) 와 pressure(intra_forward_swap)
//! 양 도메인이 공유하는 인터페이스 정의. §13.8-G B-5b (KVCacheOps 패턴)와 동일 동기.
//!
//! Spec: ENG-ALG-235, INV-147.

use std::sync::Arc;

use crate::backend::GpuEvent;

/// `TransformerModel::forward_into`의 layer loop에서 layer i compute 직후,
/// layer i+1 진입 직전에 호출되는 trait (ENG-ALG-235).
///
/// hook이 `None`인 경우 forward path는 `Option::is_some` branch 1회만 추가되며,
/// hot-path overhead는 measurement noise 이하여야 한다 (INV-147).
///
/// Pre-conditions:
/// - `0 <= idx < num_layers`
/// - hook은 `x` activation tensor를 mutate하지 않음
/// - hook은 forward thread context에서 실행됨 — blocking 작업 금지
pub trait LayerBoundaryHook: Send + Sync {
    /// Called after layer `idx` finished writing into `x`, before layer
    /// `idx + 1` reads `x`. `seq_len = 1` for decode, `> 1` for prefill.
    fn on_layer_boundary(&self, idx: usize, seq_len: usize);

    /// Wait-gate query: trait-object dispatchable shim for
    /// `IntraForwardSwapHook::pending_event_for`. Default is `None` so
    /// hook implementations that don't carry pending events (e.g. `NoOpHook`)
    /// pay zero overhead at the wait gate (ENG-ALG-238 / INV-149).
    #[inline]
    fn pending_event_for_dyn(&self, _idx: usize) -> Option<Arc<GpuEvent>> {
        None
    }
}

/// No-op implementation of `LayerBoundaryHook`.
///
/// Used as the default hook when no intra-forward swap plan is active,
/// incurring zero overhead on the forward hot path (INV-147).
pub struct NoOpHook;

impl LayerBoundaryHook for NoOpHook {
    fn on_layer_boundary(&self, _idx: usize, _seq_len: usize) {}
}
