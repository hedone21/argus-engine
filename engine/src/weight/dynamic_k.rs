//! `DynamicKController` — timing-based safe `K` for incremental weight swap.
//!
//! Decides `--swap-incremental-per-tick` value at runtime based on measured
//! per-layer drop cost vs forward wall budget. Spike avoidance is the absolute
//! priority (cf. `feedback_no_memory_spike.md`):
//!
//! - K is calibrated **once** in Phase 0 with a synchronous K=1 dispatch.
//! - From Phase 1 onward, `K` is monotonically non-increasing — never probes
//!   upward (probing = spike risk).
//! - Reactive pause (`should_pause`) is an independent last-line defence: if
//!   the release queue still has in-flight layers, the caller skips swap this
//!   token without decreasing K.
//! - There is no static upper cap; `compute_safe_k` is the only governor.
//!   `SwapExecutor::execute_on_slots` further bounds per-tick dispatch via the
//!   sub-batch reactive pause (`i > 0 + release_pending > 0 → break`).
//!
//! Algorithm (option C, timing-based, NO probing):
//!
//! Phase 0 (first swap batch, K=1 sync):
//!   safe_k = max(1, floor(fwd_min_ms * margin / drop_ms_per_layer))
//!
//! Phase 1+ (every subsequent token, async):
//!   if pending > 0 → pause (skip swap this token, K unchanged)
//!   else if fwd_ms < fwd_min_ms → recompute safe_k, k = min(k, safe_k)

const DEFAULT_MARGIN: f32 = 0.5;

/// Timing-based controller for the per-tick swap chunk size.
pub struct DynamicKController {
    k: usize,
    margin: f32,
    drop_ms_per_layer: f32,
    fwd_min_ms: f32,
    calibrated: bool,
}

impl DynamicKController {
    pub fn new() -> Self {
        Self {
            k: 1,
            margin: DEFAULT_MARGIN,
            drop_ms_per_layer: 0.0,
            fwd_min_ms: f32::INFINITY,
            calibrated: false,
        }
    }

    /// Current per-tick chunk size. Always `>= 1` after `new`.
    #[inline]
    pub fn current_k(&self) -> usize {
        self.k
    }

    /// `true` once `calibrate` has been called.
    #[inline]
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Phase 0 result injection. `drop_ms` is the wall-clock cost of releasing
    /// one displaced primary layer; `fwd_ms` is the forward wall of the token
    /// during which the calibration ran. Both are positive — `drop_ms <= 0` is
    /// clamped to 1e-3 (defensive: never divide by zero).
    pub fn calibrate(&mut self, drop_ms: f32, fwd_ms: f32) {
        let drop_ms = drop_ms.max(1.0e-3);
        self.drop_ms_per_layer = drop_ms;
        self.fwd_min_ms = fwd_ms.max(0.0);
        self.k = self.compute_safe_k().max(1);
        self.calibrated = true;
    }

    /// Phase 1+ observation. Tracks the EMA *minimum* forward wall — when the
    /// forward gets shorter than anything seen before, `K` may need to shrink.
    /// `K` is never increased here (monotone non-increasing).
    pub fn observe_forward(&mut self, fwd_ms: f32) {
        if !self.calibrated {
            return;
        }
        if fwd_ms < self.fwd_min_ms {
            self.fwd_min_ms = fwd_ms;
            let new_safe = self.compute_safe_k().max(1);
            self.k = self.k.min(new_safe);
        }
    }

    /// Reactive pause check. `true` when the release queue is non-empty —
    /// caller should skip swap this token to avoid stacking pressure.
    #[inline]
    pub fn should_pause(&self, release_pending: usize) -> bool {
        release_pending > 0
    }

    fn compute_safe_k(&self) -> usize {
        if self.drop_ms_per_layer <= 0.0 {
            return 1;
        }
        let raw = (self.fwd_min_ms * self.margin / self.drop_ms_per_layer).floor();
        if raw.is_finite() && raw >= 1.0 {
            raw as usize
        } else {
            1
        }
    }
}

impl Default for DynamicKController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_picks_safe_k_from_timing() {
        // fwd=10ms, drop=0.5ms, margin=0.5 → 10 * 0.5 / 0.5 = 10.
        let mut c = DynamicKController::new();
        c.calibrate(0.5, 10.0);
        assert!(c.is_calibrated());
        assert_eq!(c.current_k(), 10);
    }

    #[test]
    fn calibration_picks_safe_k_one_when_fwd_is_tight() {
        // fwd=1ms, drop=0.5ms, margin=0.5 → 1 * 0.5 / 0.5 = 1.
        let mut c = DynamicKController::new();
        c.calibrate(0.5, 1.0);
        assert_eq!(c.current_k(), 1);
    }

    #[test]
    fn forward_shrink_drops_k() {
        let mut c = DynamicKController::new();
        c.calibrate(0.5, 10.0);
        assert_eq!(c.current_k(), 10);
        // Forward suddenly drops to 1ms — safe_k recomputes to 1.
        c.observe_forward(1.0);
        assert_eq!(c.current_k(), 1);
    }

    #[test]
    fn forward_grow_keeps_k_unchanged() {
        let mut c = DynamicKController::new();
        c.calibrate(0.5, 10.0);
        assert_eq!(c.current_k(), 10);
        // Larger forward times never raise k.
        c.observe_forward(50.0);
        assert_eq!(c.current_k(), 10);
        c.observe_forward(20.0);
        assert_eq!(c.current_k(), 10);
    }

    #[test]
    fn k_is_monotone_non_increasing() {
        let mut c = DynamicKController::new();
        c.calibrate(0.5, 10.0);
        assert_eq!(c.current_k(), 10);
        c.observe_forward(1.0);
        assert_eq!(c.current_k(), 1);
        c.observe_forward(0.1);
        assert_eq!(c.current_k(), 1);
        c.observe_forward(100.0);
        assert_eq!(c.current_k(), 1);
    }

    #[test]
    fn should_pause_pending_semantics() {
        let c = DynamicKController::new();
        assert!(!c.should_pause(0));
        assert!(c.should_pause(1));
        assert!(c.should_pause(7));
    }

    #[test]
    fn observe_before_calibrate_is_noop() {
        let mut c = DynamicKController::new();
        c.observe_forward(1.0);
        assert!(!c.is_calibrated());
        assert_eq!(c.current_k(), 1);
    }

    #[test]
    fn large_fwd_small_drop_yields_large_k() {
        // fwd=1000ms, drop=0.1ms, margin=0.5 → safe = 5000. No static cap.
        let mut c = DynamicKController::new();
        c.calibrate(0.1, 1000.0);
        assert_eq!(c.current_k(), 5000);
    }

    #[test]
    fn calibration_handles_zero_drop() {
        // Defensive: drop_ms <= 0 clamps to 1e-3, no divide-by-zero.
        let mut c = DynamicKController::new();
        c.calibrate(0.0, 10.0);
        // drop clamped to 1e-3 → 10 * 0.5 / 1e-3 = 5000.
        assert_eq!(c.current_k(), 5000);
    }
}
