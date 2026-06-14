//! SWIFT layer skip configuration for speculative decoding.
//!
//! Reference: SWIFT (arXiv 2024) — self-speculative decoding via layer skipping.
//!
//! Allows attention and MLP sub-layers to be independently skipped per layer.
//! Layer 0 and layer L-1 are always executed (SWIFT constraint).

use std::collections::HashSet;

/// Configuration specifying which transformer sub-layers to skip.
///
/// Both attention and MLP can be independently skipped per layer.
/// When a sub-layer is skipped, the input passes through as identity (residual only).
#[derive(Debug, Clone, Default)]
pub struct SkipConfig {
    /// Layer indices where attention is skipped.
    pub attn_skip: HashSet<usize>,
    /// Layer indices where MLP (FFN) is skipped.
    pub mlp_skip: HashSet<usize>,
}

impl SkipConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate that layer 0 and layer L-1 are never skipped (SWIFT constraint).
    pub fn validate(&self, num_layers: usize) -> bool {
        if num_layers == 0 {
            return true;
        }
        let last = num_layers - 1;
        !self.attn_skip.contains(&0)
            && !self.attn_skip.contains(&last)
            && !self.mlp_skip.contains(&0)
            && !self.mlp_skip.contains(&last)
    }

    /// Uniform initialization: skip alternating layers (odd indices).
    ///
    /// `skip_ratio` determines the fraction of (L-2)*2 sub-layers to skip.
    /// Distributes skips evenly: first attention, then MLP, at odd-indexed layers.
    pub fn uniform_init(num_layers: usize, skip_ratio: f32) -> Self {
        if num_layers <= 2 {
            return Self::new();
        }
        let total_candidates = (num_layers - 2) * 2; // exclude first and last
        let num_skip = (total_candidates as f32 * skip_ratio).round() as usize;

        let mut config = Self::new();
        let mut count = 0;
        // Skip at odd-indexed layers (1, 3, 5, ...), alternating attn/mlp
        for i in (1..num_layers - 1).step_by(2) {
            if count >= num_skip {
                break;
            }
            config.attn_skip.insert(i);
            count += 1;
            if count >= num_skip {
                break;
            }
            config.mlp_skip.insert(i);
            count += 1;
        }
        // If still need more, skip at even-indexed layers (2, 4, 6, ...)
        for i in (2..num_layers - 1).step_by(2) {
            if count >= num_skip {
                break;
            }
            config.attn_skip.insert(i);
            count += 1;
            if count >= num_skip {
                break;
            }
            config.mlp_skip.insert(i);
            count += 1;
        }
        config
    }

    /// Check if attention should be skipped for the given layer.
    pub fn skip_attn(&self, layer_id: usize) -> bool {
        self.attn_skip.contains(&layer_id)
    }

    /// Check if MLP should be skipped for the given layer.
    pub fn skip_mlp(&self, layer_id: usize) -> bool {
        self.mlp_skip.contains(&layer_id)
    }

    /// Total number of skipped sub-layers.
    pub fn total_skips(&self) -> usize {
        self.attn_skip.len() + self.mlp_skip.len()
    }

    /// Whether any sub-layer is skipped (any skip active).
    pub fn is_active(&self) -> bool {
        !self.attn_skip.is_empty() || !self.mlp_skip.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_no_skip() {
        let config = SkipConfig::default();
        assert!(!config.is_active());
        assert_eq!(config.total_skips(), 0);
        assert!(!config.skip_attn(0));
        assert!(!config.skip_mlp(5));
    }

    #[test]
    fn test_validate_valid() {
        let mut config = SkipConfig::new();
        config.attn_skip.insert(3);
        config.mlp_skip.insert(5);
        assert!(config.validate(16));
    }

    #[test]
    fn test_validate_first_layer_skip() {
        let mut config = SkipConfig::new();
        config.attn_skip.insert(0);
        assert!(!config.validate(16));
    }

    #[test]
    fn test_validate_last_layer_skip() {
        let mut config = SkipConfig::new();
        config.mlp_skip.insert(15);
        assert!(!config.validate(16));
    }

    #[test]
    fn test_uniform_init_16_layers() {
        let config = SkipConfig::uniform_init(16, 0.5);
        // (16-2)*2 = 28 candidates, 50% = 14 skips
        assert_eq!(config.total_skips(), 14);
        // Layer 0 and 15 never skipped
        assert!(!config.skip_attn(0));
        assert!(!config.skip_mlp(0));
        assert!(!config.skip_attn(15));
        assert!(!config.skip_mlp(15));
        assert!(config.validate(16));
    }

    #[test]
    fn test_uniform_init_small() {
        let config = SkipConfig::uniform_init(2, 0.5);
        // Only 2 layers → nothing to skip
        assert_eq!(config.total_skips(), 0);
    }

    #[test]
    fn test_uniform_init_zero_ratio() {
        let config = SkipConfig::uniform_init(16, 0.0);
        assert_eq!(config.total_skips(), 0);
    }

    #[test]
    fn test_skip_queries() {
        let mut config = SkipConfig::new();
        config.attn_skip.insert(3);
        config.attn_skip.insert(7);
        config.mlp_skip.insert(5);

        assert!(config.skip_attn(3));
        assert!(config.skip_attn(7));
        assert!(!config.skip_attn(5));
        assert!(config.skip_mlp(5));
        assert!(!config.skip_mlp(3));
        assert_eq!(config.total_skips(), 3);
    }

    #[test]
    fn test_is_active() {
        let mut config = SkipConfig::new();
        assert!(!config.is_active());
        config.attn_skip.insert(1);
        assert!(config.is_active());
    }
}
