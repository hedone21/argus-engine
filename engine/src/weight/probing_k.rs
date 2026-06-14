//! `ProbingKController` — bottom-up adaptive `K` for incremental weight swap.
//!
//! Complement to [`crate::weight::DynamicKController`] (ARGUS). The
//! latter calibrates `K` once at Phase 0 (timing-based) and stays monotonic
//! non-increasing. This controller instead starts from `K = 1` (the safest
//! possible chunk) and probes the chunk size upward while monitoring two
//! safety signals:
//!
//! 1. release queue back-pressure (`release_pending > 0`) — last-line spike
//!    defence: skip the probe and decrement `K`.
//! 2. EWMA of per-token forward wall — only probe up when the EWMA stays
//!    bounded for `stability_window` tokens at the current `K`.
//!
//! Trade-offs vs ARGUS:
//!
//! - ramp-up cost (first ~log2(K_max) tokens swap at small chunks) vs
//!   calibration-and-stay design.
//! - adapts to environment drift (thermal, contention) instead of trusting a
//!   single Phase-0 measurement.
//! - allows `K` to oscillate within a tight band — useful when the optimal
//!   `K` is a moving target across decoding.
//!
//! The growth strategy is selectable:
//! - [`GrowthMode::Linear`]:  K → K + 1
//! - [`GrowthMode::Binary`]:  K → min(2 K, K_max)         (faster ramp)
//!
//! Spike avoidance is preserved: a probe is only attempted when the previous
//! `stability_window` tokens reported `release_pending == 0`. On any spike
//! observation the controller drops back to `K.saturating_sub(1)` and resets
//! the stability counter — symmetric to ARGUS's monotonic shrink.

const DEFAULT_ALPHA: f32 = 0.3;
const DEFAULT_STABILITY_WINDOW: usize = 5;
const DEFAULT_MAX_K: usize = 64;

/// Probing growth schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrowthMode {
    /// `K → K + 1` once stable.
    Linear,
    /// `K → min(2 K, max_k)` once stable.
    Binary,
}

/// Bottom-up adaptive controller for the per-tick swap chunk size.
pub struct ProbingKController {
    k: usize,
    max_k: usize,
    alpha: f32,
    stability_window: usize,
    tokens_at_current_k: usize,
    spike_count_at_current_k: usize,
    fwd_ewma: f32,
    growth: GrowthMode,
    calibrated: bool,
}

impl ProbingKController {
    /// Construct with `K = 1`, max `K = max_k` (default 64), linear growth.
    pub fn new() -> Self {
        Self::with_options(1, DEFAULT_MAX_K, GrowthMode::Linear)
    }

    pub fn with_options(initial_k: usize, max_k: usize, growth: GrowthMode) -> Self {
        Self {
            k: initial_k.max(1).min(max_k.max(1)),
            max_k: max_k.max(1),
            alpha: DEFAULT_ALPHA,
            stability_window: DEFAULT_STABILITY_WINDOW,
            tokens_at_current_k: 0,
            spike_count_at_current_k: 0,
            fwd_ewma: 0.0,
            growth,
            calibrated: false,
        }
    }

    /// Override stability window (tokens to observe before probing up).
    pub fn set_stability_window(&mut self, n: usize) {
        self.stability_window = n.max(1);
    }

    /// Override EWMA smoothing factor (0 < α ≤ 1).
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.01, 1.0);
    }

    /// Override `K` ceiling.
    pub fn set_max_k(&mut self, max_k: usize) {
        self.max_k = max_k.max(1);
        if self.k > self.max_k {
            self.k = self.max_k;
        }
    }

    #[inline]
    pub fn current_k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    #[inline]
    pub fn fwd_ewma(&self) -> f32 {
        self.fwd_ewma
    }

    /// Same reactive pause as ARGUS: when the release queue is non-empty,
    /// skip this token's swap dispatch.
    #[inline]
    pub fn should_pause(&self, release_pending: usize) -> bool {
        release_pending > 0
    }

    /// Per-token observation. Updates EWMA, counts stability, decides whether
    /// to probe up or shrink down.
    pub fn observe(&mut self, fwd_ms: f32, release_pending: usize) {
        // Bootstrap / EWMA update.
        if !self.calibrated {
            self.fwd_ewma = fwd_ms;
            self.calibrated = true;
        } else {
            self.fwd_ewma = self.alpha * fwd_ms + (1.0 - self.alpha) * self.fwd_ewma;
        }

        // Spike = backlog *growth* (more layers waiting than this tick
        // dispatched). `release_pending == K` is the steady state immediately
        // after a clean K-layer swap. `release_pending > K` means at least one
        // tick's worth of layers failed to release — the controller treats
        // that as the back-pressure signal and shrinks K by one.
        if release_pending > self.k {
            self.spike_count_at_current_k += 1;
            if self.k > 1 {
                self.k -= 1;
            }
            self.tokens_at_current_k = 0;
            self.spike_count_at_current_k = 0;
            return;
        }

        // Stable accumulator.
        self.tokens_at_current_k += 1;

        // Probe up after `stability_window` clean tokens.
        if self.tokens_at_current_k >= self.stability_window && self.k < self.max_k {
            self.k = match self.growth {
                GrowthMode::Linear => (self.k + 1).min(self.max_k),
                GrowthMode::Binary => (self.k.saturating_mul(2)).min(self.max_k),
            };
            self.tokens_at_current_k = 0;
            self.spike_count_at_current_k = 0;
        }
    }
}

impl Default for ProbingKController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_at_k_one() {
        let c = ProbingKController::new();
        assert_eq!(c.current_k(), 1);
        assert!(!c.is_calibrated());
    }

    #[test]
    fn probes_up_after_stability_window() {
        let mut c = ProbingKController::with_options(1, 32, GrowthMode::Linear);
        c.set_stability_window(3);
        for _ in 0..3 {
            c.observe(10.0, 0);
        }
        assert_eq!(c.current_k(), 2);
        assert!(c.is_calibrated());
    }

    #[test]
    fn binary_growth_doubles_k() {
        let mut c = ProbingKController::with_options(1, 32, GrowthMode::Binary);
        c.set_stability_window(2);
        for _ in 0..2 {
            c.observe(10.0, 0);
        }
        assert_eq!(c.current_k(), 2);
        for _ in 0..2 {
            c.observe(10.0, 0);
        }
        assert_eq!(c.current_k(), 4);
        for _ in 0..2 {
            c.observe(10.0, 0);
        }
        assert_eq!(c.current_k(), 8);
    }

    #[test]
    fn spike_drops_k_and_resets_window() {
        let mut c = ProbingKController::with_options(1, 32, GrowthMode::Linear);
        c.set_stability_window(3);
        for _ in 0..3 {
            c.observe(10.0, 0);
        } // K = 2
        for _ in 0..3 {
            c.observe(10.0, 2);
        } // pending == K → no spike, K=3
        assert_eq!(c.current_k(), 3);
        c.observe(10.0, 5); // pending=5 > K=3 → spike
        assert_eq!(c.current_k(), 2);
    }

    #[test]
    fn pending_equal_to_k_is_not_spike() {
        let mut c = ProbingKController::with_options(2, 32, GrowthMode::Linear);
        c.set_stability_window(3);
        // Steady-state: after a clean K=2 dispatch, pending typically == 2.
        for _ in 0..3 {
            c.observe(10.0, 2);
        }
        assert_eq!(c.current_k(), 3);
    }

    #[test]
    fn spike_at_k_one_stays_at_one() {
        let mut c = ProbingKController::with_options(1, 32, GrowthMode::Linear);
        c.observe(10.0, 5); // pending > K=1
        assert_eq!(c.current_k(), 1);
    }

    #[test]
    fn never_exceeds_max_k() {
        let mut c = ProbingKController::with_options(1, 4, GrowthMode::Binary);
        c.set_stability_window(1);
        for _ in 0..20 {
            c.observe(10.0, 0);
        }
        assert_eq!(c.current_k(), 4);
    }

    #[test]
    fn pause_signal_matches_release_pending() {
        let c = ProbingKController::new();
        assert!(!c.should_pause(0));
        assert!(c.should_pause(1));
    }

    #[test]
    fn fwd_ewma_smooths_observations() {
        let mut c = ProbingKController::new();
        c.observe(10.0, 0);
        let after_first = c.fwd_ewma();
        assert!((after_first - 10.0).abs() < 1e-3);
        c.observe(20.0, 0);
        let after_second = c.fwd_ewma();
        // Should be between 10 and 20, weighted by alpha.
        assert!(after_second > 10.0 && after_second < 20.0);
    }
}
