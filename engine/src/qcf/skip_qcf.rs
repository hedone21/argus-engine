//! SWIFT layer-skip proxy: tracks speculative decoding acceptance rate.
//!
//! The proxy is `1 - acceptance_rate` (rejection rate), averaged over a
//! sliding window. Zero additional computation — uses existing verify results.

use super::QcfMetric;
use std::collections::VecDeque;

/// Tracks SWIFT speculative decoding acceptance rates as a proxy for skip quality.
///
/// Maintains a sliding window of recent `(accepted, drafted)` pairs and
/// computes a moving-average rejection rate as the proxy value.
pub struct SkipQcfTracker {
    /// Recent rejection rates (1 - acceptance_rate per step).
    window: VecDeque<f32>,
    /// Maximum window size for moving average.
    window_size: usize,
    /// Cumulative totals for lifetime statistics.
    total_accepted: usize,
    total_drafted: usize,
}

impl SkipQcfTracker {
    /// Create a new tracker with the given window size (default: 50).
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            total_accepted: 0,
            total_drafted: 0,
        }
    }

    /// Record a verify result: `accepted` tokens out of `drafted` tokens.
    pub fn record(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        self.total_accepted += accepted;
        self.total_drafted += drafted;

        let rejection_rate = 1.0 - (accepted as f32 / drafted as f32);
        self.window.push_back(rejection_rate);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
    }

    /// Current proxy value: moving-average rejection rate.
    pub fn current_proxy(&self) -> QcfMetric {
        let raw_value = if self.window.is_empty() {
            0.0
        } else {
            self.window.iter().sum::<f32>() / self.window.len() as f32
        };

        QcfMetric {
            action: "swift".to_string(),
            raw_value,
            normalized_value: raw_value, // rejection rate is already normalized
            per_head: None,
            tokens_affected: self.total_drafted,
        }
    }

    /// Lifetime acceptance rate (not windowed).
    pub fn lifetime_acceptance_rate(&self) -> f32 {
        if self.total_drafted == 0 {
            1.0
        } else {
            self.total_accepted as f32 / self.total_drafted as f32
        }
    }

    /// Number of recorded steps in the window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.window.clear();
        self.total_accepted = 0;
        self.total_drafted = 0;
    }
}

impl Default for SkipQcfTracker {
    fn default() -> Self {
        Self::new(50)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tracker() {
        let tracker = SkipQcfTracker::new(10);
        let metric = tracker.current_proxy();
        assert_eq!(metric.raw_value, 0.0);
        assert_eq!(metric.action, "swift");
        assert_eq!(tracker.lifetime_acceptance_rate(), 1.0);
    }

    #[test]
    fn test_perfect_acceptance() {
        let mut tracker = SkipQcfTracker::new(10);
        tracker.record(5, 5); // 100% acceptance → 0% rejection
        tracker.record(3, 3);

        let metric = tracker.current_proxy();
        assert!((metric.raw_value - 0.0).abs() < 1e-6);
        assert!((tracker.lifetime_acceptance_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_total_rejection() {
        let mut tracker = SkipQcfTracker::new(10);
        tracker.record(0, 5); // 0% acceptance → 100% rejection

        let metric = tracker.current_proxy();
        assert!((metric.raw_value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_partial_acceptance() {
        let mut tracker = SkipQcfTracker::new(10);
        tracker.record(3, 5); // 60% acceptance → 40% rejection

        let metric = tracker.current_proxy();
        assert!((metric.raw_value - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_window_rolling() {
        let mut tracker = SkipQcfTracker::new(3);
        tracker.record(5, 5); // rejection = 0.0
        tracker.record(0, 5); // rejection = 1.0
        tracker.record(5, 5); // rejection = 0.0
        // Window: [0.0, 1.0, 0.0] → avg = 0.333...

        let metric = tracker.current_proxy();
        assert!(
            (metric.raw_value - 1.0 / 3.0).abs() < 0.01,
            "expected ~0.333, got {}",
            metric.raw_value
        );

        // Add one more → oldest (0.0) drops out
        tracker.record(0, 5); // rejection = 1.0
        // Window: [1.0, 0.0, 1.0] → avg = 0.666...
        let metric = tracker.current_proxy();
        assert!(
            (metric.raw_value - 2.0 / 3.0).abs() < 0.01,
            "expected ~0.666, got {}",
            metric.raw_value
        );
        assert_eq!(tracker.window_len(), 3);
    }

    #[test]
    fn test_zero_drafted_ignored() {
        let mut tracker = SkipQcfTracker::new(10);
        tracker.record(0, 0); // Should be ignored
        assert_eq!(tracker.window_len(), 0);
    }

    #[test]
    fn test_lifetime_rate() {
        let mut tracker = SkipQcfTracker::new(10);
        tracker.record(3, 5);
        tracker.record(4, 5);
        // Lifetime: 7/10 = 0.7
        assert!((tracker.lifetime_acceptance_rate() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut tracker = SkipQcfTracker::new(10);
        tracker.record(3, 5);
        tracker.record(4, 5);
        tracker.reset();

        assert_eq!(tracker.window_len(), 0);
        assert_eq!(tracker.current_proxy().raw_value, 0.0);
        assert_eq!(tracker.lifetime_acceptance_rate(), 1.0);
    }
}
