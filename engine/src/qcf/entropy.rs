//! Shannon entropy of attention/importance distribution (ARGUS QCF improvement #6).
//!
//! Sparse distributions (low entropy) are hypothesised to predict catastrophic
//! quality loss under aggressive eviction (e.g., NIAH needle in low-entropy
//! attention).

/// Result of normalized entropy computation.
#[derive(Debug, Clone, Copy)]
pub struct EntropyResult {
    /// Raw Shannon entropy in nats: `-Σ p_i log p_i`.
    pub entropy: f32,
    /// Length-normalized entropy: `entropy / ln(T)`, in `[0, 1]`.
    /// Uniform distribution → 1.0; one-hot → 0.0.
    pub entropy_normalized: f32,
}

/// Compute Shannon entropy of `scores` after L1 normalization to a probability
/// distribution. Negative or zero entries are treated as zero probability.
///
/// Returns `(0, 0)` for empty input or all-zero distribution.
pub fn compute_normalized_entropy(scores: &[f32]) -> EntropyResult {
    let n = scores.len();
    if n == 0 {
        return EntropyResult {
            entropy: 0.0,
            entropy_normalized: 0.0,
        };
    }
    let total: f32 = scores.iter().map(|&s| s.max(0.0)).sum();
    if total <= 0.0 {
        return EntropyResult {
            entropy: 0.0,
            entropy_normalized: 0.0,
        };
    }
    let mut entropy = 0.0f32;
    for &s in scores {
        let s = s.max(0.0);
        if s > 0.0 {
            let p = s / total;
            entropy -= p * p.ln();
        }
    }
    let max_entropy = (n as f32).ln();
    let entropy_normalized = if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    EntropyResult {
        entropy,
        entropy_normalized,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let r = compute_normalized_entropy(&[]);
        assert_eq!(r.entropy, 0.0);
        assert_eq!(r.entropy_normalized, 0.0);
    }

    #[test]
    fn all_zero() {
        let r = compute_normalized_entropy(&[0.0, 0.0, 0.0]);
        assert_eq!(r.entropy, 0.0);
        assert_eq!(r.entropy_normalized, 0.0);
    }

    #[test]
    fn uniform_normalized_one() {
        // Uniform distribution: entropy = ln(N), normalized = 1.
        let r = compute_normalized_entropy(&[1.0, 1.0, 1.0, 1.0]);
        assert!(
            (r.entropy - (4f32).ln()).abs() < 1e-5,
            "entropy={}, expected={}",
            r.entropy,
            (4f32).ln()
        );
        assert!(
            (r.entropy_normalized - 1.0).abs() < 1e-5,
            "entropy_normalized={}, expected=1.0",
            r.entropy_normalized
        );
    }

    #[test]
    fn one_hot_normalized_zero() {
        let r = compute_normalized_entropy(&[10.0, 0.0, 0.0, 0.0]);
        assert!(r.entropy < 1e-5, "entropy={}, expected ~0", r.entropy);
        assert!(
            r.entropy_normalized < 1e-5,
            "entropy_normalized={}, expected ~0",
            r.entropy_normalized
        );
    }

    #[test]
    fn negative_treated_as_zero() {
        let r = compute_normalized_entropy(&[-1.0, 1.0, 1.0]);
        // Effective: [0, 1, 1] → entropy = ln(2)
        assert!(
            (r.entropy - 2f32.ln()).abs() < 1e-5,
            "entropy={}, expected={}",
            r.entropy,
            2f32.ln()
        );
    }

    #[test]
    fn single_element() {
        // n=1: max_entropy = ln(1) = 0 → normalized = 0 (degenerate)
        let r = compute_normalized_entropy(&[3.5]);
        assert_eq!(r.entropy, 0.0);
        assert_eq!(r.entropy_normalized, 0.0);
    }

    #[test]
    fn skewed_between() {
        let r = compute_normalized_entropy(&[10.0, 1.0, 1.0]);
        assert!(
            r.entropy_normalized > 0.0,
            "entropy_normalized should be > 0 for skewed distribution"
        );
        assert!(
            r.entropy_normalized < 1.0,
            "entropy_normalized should be < 1 for skewed distribution"
        );
    }
}
