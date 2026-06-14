//! Top-K importance token retention metric (ARGUS QCF improvement #5).
//!
//! Measures the fraction of "high-importance" tokens that survived a lossy
//! cache action. Strong catastrophic-detection signal candidate (e.g., NIAH
//! needle token loss).

use std::collections::HashSet;

/// Result of top-K retention computation.
#[derive(Debug, Clone)]
pub struct TopKRetentionResult {
    /// `|top_K ∩ retained| / K`, in `[0, 1]`. Higher = better retention.
    pub retention_binary: f32,
    /// Importance-weighted retention: `Σ_{t ∈ top_K ∩ retained} α_t / Σ_{t ∈ top_K} α_t`.
    pub retention_weighted: f32,
    /// Indices of the top-K importance tokens (descending importance).
    pub topk_indices: Vec<usize>,
}

/// Compute top-K retention given importance scores and the evicted token set.
///
/// - `importance_scores`: per-token importance (length T = current_pos).
/// - `evicted_set`: set of token positions that were removed by the action.
/// - `k`: how many top-importance tokens to consider.
///
/// If `importance_scores` is empty or `k` is 0, returns retention 1.0 (no risk).
pub fn compute_topk_retention(
    importance_scores: &[f32],
    evicted_set: &HashSet<usize>,
    k: usize,
) -> TopKRetentionResult {
    let n = importance_scores.len();
    let k = k.min(n);
    if k == 0 {
        return TopKRetentionResult {
            retention_binary: 1.0,
            retention_weighted: 1.0,
            topk_indices: Vec::new(),
        };
    }

    let mut indexed: Vec<(usize, f32)> = importance_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let topk_indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
    let topk_scores: Vec<f32> = indexed.iter().take(k).map(|(_, s)| *s).collect();

    let retained_count = topk_indices
        .iter()
        .filter(|t| !evicted_set.contains(t))
        .count();
    let retention_binary = retained_count as f32 / k as f32;

    let total_imp: f32 = topk_scores.iter().sum();
    let retained_imp: f32 = topk_indices
        .iter()
        .zip(topk_scores.iter())
        .filter(|(t, _)| !evicted_set.contains(t))
        .map(|(_, imp)| *imp)
        .sum();
    let retention_weighted = if total_imp > 0.0 {
        retained_imp / total_imp
    } else {
        1.0
    };

    TopKRetentionResult {
        retention_binary,
        retention_weighted,
        topk_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_evicted() -> HashSet<usize> {
        HashSet::new()
    }

    fn evicted_set(indices: &[usize]) -> HashSet<usize> {
        indices.iter().copied().collect()
    }

    #[test]
    fn test_all_retained() {
        // evicted_set is empty → all top-K tokens retained → retention = 1.0
        let importance = vec![0.5f32, 0.3, 0.8, 0.1, 0.6];
        let result = compute_topk_retention(&importance, &empty_evicted(), 3);
        assert!(
            (result.retention_binary - 1.0).abs() < 1e-6,
            "expected binary=1.0, got {}",
            result.retention_binary
        );
        assert!(
            (result.retention_weighted - 1.0).abs() < 1e-6,
            "expected weighted=1.0, got {}",
            result.retention_weighted
        );
        // top-3 by importance: indices 2(0.8), 4(0.6), 0(0.5)
        assert_eq!(result.topk_indices.len(), 3);
        assert!(result.topk_indices.contains(&2));
        assert!(result.topk_indices.contains(&4));
        assert!(result.topk_indices.contains(&0));
    }

    #[test]
    fn test_all_evicted() {
        // All top-K tokens evicted → retention = 0.0
        let importance = vec![0.5f32, 0.3, 0.8, 0.1, 0.6];
        // top-3 are indices 2,4,0
        let evicted = evicted_set(&[0, 2, 4]);
        let result = compute_topk_retention(&importance, &evicted, 3);
        assert!(
            (result.retention_binary - 0.0).abs() < 1e-6,
            "expected binary=0.0, got {}",
            result.retention_binary
        );
        assert!(
            (result.retention_weighted - 0.0).abs() < 1e-6,
            "expected weighted=0.0, got {}",
            result.retention_weighted
        );
    }

    #[test]
    fn test_half_retained() {
        // Evict exactly half of top-K → binary = 0.5
        let importance = vec![1.0f32, 1.0, 1.0, 1.0]; // uniform importance
        // top-4: all indices 0..4; evict 2 of them
        let evicted = evicted_set(&[0, 1]);
        let result = compute_topk_retention(&importance, &evicted, 4);
        assert!(
            (result.retention_binary - 0.5).abs() < 1e-6,
            "expected binary=0.5, got {}",
            result.retention_binary
        );
        // uniform importance → weighted == binary
        assert!(
            (result.retention_weighted - 0.5).abs() < 1e-6,
            "expected weighted=0.5 (uniform), got {}",
            result.retention_weighted
        );
    }

    #[test]
    fn test_weighted_emphasis() {
        // Non-uniform importance: the most important token survives, less important evicted.
        // weighted retention should be > binary retention.
        let importance = vec![10.0f32, 1.0, 1.0]; // token 0 dominates
        // top-3: [0,1,2]; evict tokens 1 and 2 (the less important ones)
        let evicted = evicted_set(&[1, 2]);
        let result = compute_topk_retention(&importance, &evicted, 3);
        assert!(
            result.retention_weighted > result.retention_binary,
            "weighted ({}) should > binary ({}) when most important token survives",
            result.retention_weighted,
            result.retention_binary
        );
        // binary = 1/3 ≈ 0.333; weighted = 10/12 ≈ 0.833
        assert!(
            (result.retention_binary - 1.0 / 3.0).abs() < 1e-5,
            "expected binary≈0.333, got {}",
            result.retention_binary
        );
        let expected_weighted = 10.0f32 / 12.0;
        assert!(
            (result.retention_weighted - expected_weighted).abs() < 1e-5,
            "expected weighted≈{expected_weighted}, got {}",
            result.retention_weighted
        );
    }

    #[test]
    fn test_k_exceeds_n() {
        // k > importance.len() → effective k = n
        let importance = vec![0.5f32, 0.3, 0.8];
        let result = compute_topk_retention(&importance, &empty_evicted(), 100);
        assert_eq!(result.topk_indices.len(), 3, "effective k should be n=3");
        assert!(
            (result.retention_binary - 1.0).abs() < 1e-6,
            "all retained → binary=1.0"
        );
    }

    #[test]
    fn test_empty_importance() {
        // Empty importance slice → no-op, retention 1.0
        let result = compute_topk_retention(&[], &empty_evicted(), 5);
        assert!(
            (result.retention_binary - 1.0).abs() < 1e-6,
            "empty importance → binary=1.0"
        );
        assert!(
            (result.retention_weighted - 1.0).abs() < 1e-6,
            "empty importance → weighted=1.0"
        );
        assert!(result.topk_indices.is_empty());
    }

    #[test]
    fn test_k_zero() {
        // k=0 → no risk, retention 1.0
        let importance = vec![0.5f32, 0.3, 0.8];
        let result = compute_topk_retention(&importance, &evicted_set(&[0, 1, 2]), 0);
        assert!(
            (result.retention_binary - 1.0).abs() < 1e-6,
            "k=0 → binary=1.0"
        );
        assert!(
            (result.retention_weighted - 1.0).abs() < 1e-6,
            "k=0 → weighted=1.0"
        );
        assert!(result.topk_indices.is_empty());
    }
}
