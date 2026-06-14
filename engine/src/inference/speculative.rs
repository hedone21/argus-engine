//! SWIFT speculative decoding framework.
//!
//! Provides draft-verify cycle infrastructure for layer-skip based
//! speculative decoding. The same model is used as both draft (with skip)
//! and verifier (full layers).
//!
//! Reference: SWIFT (arXiv 2024)

use crate::inference::skip_config::SkipConfig;

/// Result of a draft generation phase.
#[derive(Debug, Clone)]
pub struct DraftResult {
    /// Generated draft tokens.
    pub tokens: Vec<u32>,
    /// Confidence (max softmax probability) for each draft token.
    pub confidences: Vec<f32>,
}

/// Result of verification phase.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Number of draft tokens accepted (matching verifier output).
    pub accepted_count: usize,
    /// If rejected, the corrected token from the verifier.
    pub corrected_token: Option<u32>,
}

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Skip configuration for draft model.
    pub skip_config: SkipConfig,
    /// Maximum draft tokens per speculative step. Default: 25.
    pub max_draft_steps: usize,
    /// Confidence threshold below which draft stops early. Default: 0.8.
    pub stop_threshold: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            skip_config: SkipConfig::default(),
            max_draft_steps: 25,
            stop_threshold: 0.8,
        }
    }
}

impl SpeculativeConfig {
    pub fn new(skip_config: SkipConfig, max_draft_steps: usize, stop_threshold: f32) -> Self {
        Self {
            skip_config,
            max_draft_steps,
            stop_threshold,
        }
    }
}

/// Rollback KV cache positions after rejected draft tokens.
///
/// When draft tokens are rejected during verification, the KV cache
/// must be rewound to the position before the draft started.
///
/// `accepted`: number of accepted draft tokens (these stay in cache)
/// `total_draft`: total draft tokens generated (all were appended to KV cache)
pub fn rollback_kv_positions(
    kv_current_positions: &mut [usize],
    accepted: usize,
    total_draft: usize,
) {
    let rollback = total_draft - accepted;
    for pos in kv_current_positions.iter_mut() {
        *pos = pos.saturating_sub(rollback);
    }
}

/// Accept/reject draft tokens against verifier output (greedy decoding).
///
/// For each position, if `argmax(target_logits[i]) == draft_tokens[i]`, accept.
/// On first mismatch, return the corrected token and stop.
pub fn verify_greedy(draft_tokens: &[u32], target_tokens: &[u32]) -> VerifyResult {
    let mut accepted = 0;
    let mut corrected = None;

    for (i, &draft) in draft_tokens.iter().enumerate() {
        if i < target_tokens.len() && target_tokens[i] == draft {
            accepted += 1;
        } else {
            corrected = target_tokens.get(i).copied();
            break;
        }
    }

    VerifyResult {
        accepted_count: accepted,
        corrected_token: corrected,
    }
}

/// Skip layer optimizer using random search.
///
/// Evaluates skip configurations by comparing draft model output
/// against reference tokens (matchness score).
pub struct SkipOptimizer {
    /// Target skip ratio (fraction of sub-layers to skip). Default: 0.45.
    pub skip_ratio: f32,
    /// Maximum optimization iterations. Default: 200.
    pub max_iter: usize,
    /// Early stopping threshold for matchness. Default: 0.95.
    pub early_stop: f32,
}

impl Default for SkipOptimizer {
    fn default() -> Self {
        Self {
            skip_ratio: 0.45,
            max_iter: 200,
            early_stop: 0.95,
        }
    }
}

impl SkipOptimizer {
    pub fn new(skip_ratio: f32, max_iter: usize, early_stop: f32) -> Self {
        Self {
            skip_ratio,
            max_iter,
            early_stop,
        }
    }

    /// Generate a random skip configuration by perturbing the given config.
    ///
    /// Swaps one sub-layer skip (add or remove a random layer).
    pub fn perturb(config: &SkipConfig, num_layers: usize, rng_seed: u64) -> SkipConfig {
        let mut new_config = config.clone();
        // Simple hash-based pseudo-random
        let idx = (rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 33) as usize;
        let layer = 1 + (idx % (num_layers.saturating_sub(2)).max(1));
        let is_attn = (idx / num_layers).is_multiple_of(2);

        if is_attn {
            if new_config.attn_skip.contains(&layer) {
                new_config.attn_skip.remove(&layer);
            } else {
                new_config.attn_skip.insert(layer);
            }
        } else if new_config.mlp_skip.contains(&layer) {
            new_config.mlp_skip.remove(&layer);
        } else {
            new_config.mlp_skip.insert(layer);
        }

        new_config
    }

    /// Compute matchness between predicted and reference tokens.
    ///
    /// matchness = count(predicted == reference) / total
    pub fn matchness(predicted: &[u32], reference: &[u32]) -> f32 {
        if reference.is_empty() {
            return 0.0;
        }
        let matches = predicted
            .iter()
            .zip(reference.iter())
            .filter(|(p, r)| p == r)
            .count();
        matches as f32 / reference.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_greedy_all_accepted() {
        let draft = vec![1, 2, 3, 4];
        let target = vec![1, 2, 3, 4];
        let result = verify_greedy(&draft, &target);
        assert_eq!(result.accepted_count, 4);
        assert_eq!(result.corrected_token, None);
    }

    #[test]
    fn test_verify_greedy_first_rejected() {
        let draft = vec![1, 2, 3];
        let target = vec![5, 2, 3];
        let result = verify_greedy(&draft, &target);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.corrected_token, Some(5));
    }

    #[test]
    fn test_verify_greedy_partial_accept() {
        let draft = vec![1, 2, 99, 4];
        let target = vec![1, 2, 3, 4];
        let result = verify_greedy(&draft, &target);
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.corrected_token, Some(3));
    }

    #[test]
    fn test_rollback_kv() {
        let mut positions = vec![100, 100, 100];
        rollback_kv_positions(&mut positions, 2, 5);
        // Rollback 3 positions
        assert_eq!(positions, vec![97, 97, 97]);
    }

    #[test]
    fn test_rollback_kv_all_accepted() {
        let mut positions = vec![50, 50];
        rollback_kv_positions(&mut positions, 5, 5);
        // No rollback
        assert_eq!(positions, vec![50, 50]);
    }

    #[test]
    fn test_matchness() {
        assert!((SkipOptimizer::matchness(&[1, 2, 3, 4], &[1, 2, 3, 4]) - 1.0).abs() < 1e-5);
        assert!((SkipOptimizer::matchness(&[1, 2, 9, 4], &[1, 2, 3, 4]) - 0.75).abs() < 1e-5);
        assert!((SkipOptimizer::matchness(&[9, 9, 9, 9], &[1, 2, 3, 4]) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_perturb_changes_config() {
        let config = SkipConfig::uniform_init(16, 0.3);
        let original_skips = config.total_skips();
        let perturbed = SkipOptimizer::perturb(&config, 16, 42);
        // Perturbation should change exactly one skip
        let diff = (perturbed.total_skips() as i32 - original_skips as i32).unsigned_abs();
        assert!(diff <= 1, "perturbation changed {diff} skips, expected ≤ 1");
    }

    #[test]
    fn test_perturb_respects_boundaries() {
        let config = SkipConfig::new();
        // Test many seeds — layer 0 and L-1 should never appear
        for seed in 0..100 {
            let p = SkipOptimizer::perturb(&config, 16, seed);
            assert!(
                !p.attn_skip.contains(&0),
                "seed={seed}: layer 0 in attn_skip"
            );
            assert!(
                !p.attn_skip.contains(&15),
                "seed={seed}: layer 15 in attn_skip"
            );
        }
    }

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.max_draft_steps, 25);
        assert!((config.stop_threshold - 0.8).abs() < 1e-5);
        assert!(!config.skip_config.is_active());
    }

    #[test]
    fn test_draft_result_empty() {
        let draft = DraftResult {
            tokens: vec![],
            confidences: vec![],
        };
        assert!(draft.tokens.is_empty());
    }
}
