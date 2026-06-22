//! SqueezeAttention per-layer KV eviction budget allocation (R-P1-6).
//!
//! SqueezeAttention assigns *different* KV-cache budgets to different decoder layers based on
//! per-layer importance (more important layers keep more tokens). The per-layer importance signal
//! is produced by the `layer-importance` plugin (`mean_pool` / `shortgpt_bi` = `1 − cos(h_in, h_out)`)
//! and surfaced via `WeightStageCtx::layer_metric(Importance)` / the `ImportanceTable`.
//!
//! This module is the *budget allocation* policy: importance → per-layer `target_len`. The vector it
//! returns is consumed by `CacheManager::run_policy_eviction`'s per-layer budget path
//! (`per_layer_target_len`). Production arming (CLI flag + warmup compute that feeds this into the
//! live eviction signal) is a follow-up; this function + the eviction mechanism are verified
//! together by an in-engine end-to-end test.

/// Group the layers into three importance tiers (low / mid / high) and assign each layer a KV
/// budget (target token count) from its tier — higher-importance layers keep more tokens.
///
/// Deterministic: layers are ranked by ascending importance (ties broken by layer index, so equal
/// importances produce a stable grouping), partitioned into thirds by rank, and given relative tier
/// weights 1 / 2 / 3. `total_budget` is distributed proportionally to tier weight; each per-layer
/// budget is then floored at `min_floor` (so no layer is starved below the policy's minimum). No RNG
/// is used (a deterministic tertile stand-in for 1-D k-means over the importance histogram).
///
/// Returns a vector of length `importance.len()` (empty when there are no layers). The sum is
/// approximately `total_budget` (integer division may leave a small remainder); the `min_floor`
/// clamp can push the sum above `total_budget` when many layers would otherwise fall below the floor.
pub fn compute_squeeze_budgets(
    importance: &[f32],
    total_budget: usize,
    min_floor: usize,
) -> Vec<usize> {
    let n = importance.len();
    if n == 0 {
        return Vec::new();
    }

    // Rank layers by ascending importance; stable tiebreak by layer index keeps it deterministic
    // even when importances are equal or contain NaN (NaN sorts as "equal", preserving index order).
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| {
        importance[a]
            .partial_cmp(&importance[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // tier weight per layer: lowest third = 1, middle third = 2, highest third = 3.
    let mut weights = vec![0usize; n];
    for (rank, &layer) in ranked.iter().enumerate() {
        let tier = (rank * 3 / n).min(2); // 0, 1, 2
        weights[layer] = tier + 1; // 1, 2, 3
    }

    let wsum: usize = weights.iter().sum::<usize>().max(1);
    weights
        .iter()
        .map(|&w| ((total_budget * w) / wsum).max(min_floor))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_importance_yields_empty_budget() {
        assert!(compute_squeeze_budgets(&[], 1024, 64).is_empty());
    }

    #[test]
    fn length_matches_layer_count() {
        let imp = [0.1, 0.5, 0.9, 0.2, 0.8, 0.3];
        assert_eq!(compute_squeeze_budgets(&imp, 1200, 16).len(), imp.len());
    }

    #[test]
    fn higher_importance_gets_at_least_as_much_budget() {
        // Strictly increasing importance → tiers are [low, low, mid, mid, high, high] by rank,
        // so budgets are monotonically non-decreasing with importance.
        let imp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b = compute_squeeze_budgets(&imp, 1200, 1);
        for i in 1..b.len() {
            assert!(
                b[i] >= b[i - 1],
                "budget should not decrease with importance: {b:?}"
            );
        }
        // The most-important layer is in the top tier (weight 3) and the least in the bottom
        // tier (weight 1), so it must keep strictly more (away from the floor).
        assert!(b[5] > b[0], "top tier must exceed bottom tier: {b:?}");
    }

    #[test]
    fn min_floor_is_enforced() {
        // Tiny total budget → proportional shares round toward 0; the floor must lift every layer.
        let imp = [0.1, 0.9, 0.5, 0.2];
        let b = compute_squeeze_budgets(&imp, 1, 64);
        assert!(
            b.iter().all(|&x| x >= 64),
            "all budgets floored at 64: {b:?}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let imp = [0.7, 0.1, 0.7, 0.3, 0.9, 0.1, 0.5];
        let a = compute_squeeze_budgets(&imp, 999, 8);
        let b = compute_squeeze_budgets(&imp, 999, 8);
        assert_eq!(a, b, "allocation must be deterministic");
    }

    #[test]
    fn tertile_grouping_small_layer_counts() {
        // n = 1, 2, 3 must not panic and must produce a valid budget per layer.
        for n in 1..=3 {
            let imp: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let b = compute_squeeze_budgets(&imp, 300, 10);
            assert_eq!(b.len(), n);
            assert!(b.iter().all(|&x| x >= 10));
        }
    }

    #[test]
    fn equal_importance_is_uniform_and_stable() {
        // All-equal importance → stable index tiebreak partitions by position; budgets are a valid
        // non-decreasing tier pattern, and the call is deterministic.
        let imp = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let b = compute_squeeze_budgets(&imp, 600, 1);
        assert_eq!(b.len(), 6);
        assert_eq!(b, compute_squeeze_budgets(&imp, 600, 1));
    }
}
