//! `IncrementalSwapPlan` — Layer-Incremental Swap Stage 1 MVP (ENG-ALG-232).
//!
//! Holds the remaining target layers and per-tick dispatch budget.
//! One `drain()` call per decode token until `is_done()`.
//!
//! Spec: ENG-ALG-232, INV-145.

use std::collections::VecDeque;

/// ENG-ALG-232: Incremental swap dispatch plan.
///
/// Holds the remaining target layers and per-tick dispatch budget.
/// One `drain()` call per decode token until `is_done()`.
///
/// **Invariants (INV-145)**:
/// - `drain()` is monotonically decreasing: each call reduces `remaining.len()`
///   or leaves it unchanged (already empty).
/// - Drained layers never reappear.
/// - Empty plan returns empty `Vec` from `drain()` — no-op compatible.
///
/// **Per-tick = 0 policy**: `IncrementalSwapPlan` must not be constructed
/// with `per_tick == 0`. If constructed defensively (e.g. via test), `drain()`
/// always returns empty and `is_done()` returns `false` — the plan can never
/// self-retire. The caller is responsible for retiring via `Option::take()`.
pub struct IncrementalSwapPlan {
    remaining: VecDeque<usize>,
    per_tick: usize,
    started_at_token: usize,
}

impl IncrementalSwapPlan {
    /// Create a new plan.
    ///
    /// **Preconditions (caller-enforced)**:
    /// - `per_tick > 0` — callers must use the single-shot path when `per_tick == 0`.
    /// - `target_layers` is non-empty — callers should skip construction when empty.
    ///
    /// If `per_tick == 0`, the plan is constructed but `drain()` always returns empty
    /// (defensive — the plan never progresses).
    pub fn new(target_layers: Vec<usize>, per_tick: usize, started_at_token: usize) -> Self {
        Self {
            remaining: VecDeque::from(target_layers),
            per_tick,
            started_at_token,
        }
    }

    /// Drain up to `per_tick` layers from the front of the queue.
    ///
    /// Returns a `Vec<usize>` of layer indices to swap this tick.
    /// Returns empty `Vec` if:
    /// - `per_tick == 0` (defensive: should not occur in production)
    /// - `remaining` is already empty
    ///
    /// INV-145: each call strictly reduces `remaining.len()` (when non-zero
    /// and `per_tick > 0`), or leaves it at zero.
    pub fn drain_chunk(&mut self) -> Vec<usize> {
        if self.per_tick == 0 {
            return Vec::new();
        }
        let n = self.per_tick.min(self.remaining.len());
        self.remaining.drain(..n).collect()
    }

    /// Returns `true` when all layers have been drained.
    ///
    /// The decode loop should call `Option::take()` immediately after detecting
    /// `is_done() == true` to prevent further `drain_chunk()` calls (INV-145).
    pub fn is_done(&self) -> bool {
        self.remaining.is_empty()
    }

    /// Remaining layer count (for diagnostics/logging).
    pub fn remaining_count(&self) -> usize {
        self.remaining.len()
    }

    /// Token index at which this plan was committed (for diagnostics/logging).
    pub fn started_at_token(&self) -> usize {
        self.started_at_token
    }

    /// Update the per-tick chunk budget. Used by `DynamicKController` to inject
    /// the calibrated/observed `K` value before each `drain_chunk` call.
    ///
    /// `per_tick == 0` is accepted and turns `drain_chunk` into a no-op for
    /// that tick (used by the reactive-pause branch).
    pub fn set_per_tick(&mut self, per_tick: usize) {
        self.per_tick = per_tick;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_progression_per_tick_2() {
        let mut plan = IncrementalSwapPlan::new(vec![0, 1, 2, 3, 4], 2, 0);
        assert_eq!(plan.remaining_count(), 5);
        assert!(!plan.is_done());

        let c0 = plan.drain_chunk();
        assert_eq!(c0, vec![0, 1]);
        assert_eq!(plan.remaining_count(), 3);
        assert!(!plan.is_done());

        let c1 = plan.drain_chunk();
        assert_eq!(c1, vec![2, 3]);
        assert_eq!(plan.remaining_count(), 1);
        assert!(!plan.is_done());

        let c2 = plan.drain_chunk();
        assert_eq!(c2, vec![4]);
        assert_eq!(plan.remaining_count(), 0);
        assert!(plan.is_done());

        // After done: further drain returns empty
        let c3 = plan.drain_chunk();
        assert!(c3.is_empty());
        assert!(plan.is_done());
    }

    #[test]
    fn empty_layers_is_done_immediately() {
        let plan = IncrementalSwapPlan::new(vec![], 2, 0);
        assert!(plan.is_done());
    }

    #[test]
    fn per_tick_zero_always_empty_never_done() {
        // Defensive: per_tick=0 should not be constructed in production,
        // but if it is, drain always returns empty and is_done stays false.
        let mut plan = IncrementalSwapPlan::new(vec![0, 1], 0, 0);
        assert!(!plan.is_done());
        let chunk = plan.drain_chunk();
        assert!(chunk.is_empty());
        assert!(!plan.is_done()); // plan can never self-retire
    }

    #[test]
    fn started_at_token_preserved() {
        let plan = IncrementalSwapPlan::new(vec![0], 1, 42);
        assert_eq!(plan.started_at_token(), 42);
    }

    #[test]
    fn remaining_count_monotone() {
        let mut plan = IncrementalSwapPlan::new(vec![0, 1, 2, 3], 1, 0);
        let mut prev = plan.remaining_count();
        while !plan.is_done() {
            plan.drain_chunk();
            let cur = plan.remaining_count();
            assert!(cur < prev, "remaining must strictly decrease each drain");
            prev = cur;
        }
        assert_eq!(plan.remaining_count(), 0);
    }
}
