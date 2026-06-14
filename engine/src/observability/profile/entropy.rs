/// Attention entropy tracker (optional probe).
///
/// Computes Shannon entropy of post-softmax attention distributions
/// to measure attention focus vs. diffuseness.
///
/// - **Low entropy**: attention concentrated on few tokens → clear heavy hitters.
/// - **High entropy**: attention spread broadly → eviction is risky.
#[derive(Default)]
pub struct EntropyTracker {
    records: Vec<EntropyRecord>,
}

pub struct EntropyRecord {
    pub step: usize,
    /// Average entropy across all heads for this step.
    pub avg_entropy: f32,
    /// Min entropy across heads (most focused head).
    pub min_entropy: f32,
    /// Max entropy across heads (most diffuse head).
    pub max_entropy: f32,
}

impl EntropyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn records(&self) -> &[EntropyRecord] {
        &self.records
    }

    /// Compute entropy from post-softmax attention scores.
    ///
    /// `scores`: flat buffer `[n_heads * stride]` containing attention weights.
    /// `stride`: distance between head score arrays (>= cache_seq_len).
    /// `cache_seq_len`: number of valid positions.
    /// `n_heads`: number of query heads.
    pub fn record_from_scores(
        &mut self,
        step: usize,
        scores: &[f32],
        stride: usize,
        cache_seq_len: usize,
        n_heads: usize,
    ) {
        if n_heads == 0 || cache_seq_len == 0 {
            return;
        }

        let mut sum_entropy = 0.0f32;
        let mut min_h = f32::MAX;
        let mut max_h = f32::MIN;

        for h in 0..n_heads {
            let offset = h * stride;
            let h_val = shannon_entropy(&scores[offset..offset + cache_seq_len]);
            sum_entropy += h_val;
            min_h = min_h.min(h_val);
            max_h = max_h.max(h_val);
        }

        self.records.push(EntropyRecord {
            step,
            avg_entropy: sum_entropy / n_heads as f32,
            min_entropy: min_h,
            max_entropy: max_h,
        });
    }

    pub fn to_json(&self) -> serde_json::Value {
        let records: Vec<serde_json::Value> = self
            .records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "step": r.step,
                    "avg_entropy": (r.avg_entropy * 1000.0).round() / 1000.0,
                    "min_entropy": (r.min_entropy * 1000.0).round() / 1000.0,
                    "max_entropy": (r.max_entropy * 1000.0).round() / 1000.0,
                })
            })
            .collect();
        serde_json::json!({ "records": records })
    }
}

/// Shannon entropy: H = -Σ p * log₂(p), with 0·log(0) = 0.
fn shannon_entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution over 4 tokens: H = log2(4) = 2.0
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&probs);
        assert!((h - 2.0).abs() < 1e-5, "got {}", h);
    }

    #[test]
    fn test_shannon_entropy_peaked() {
        // One-hot distribution: H = 0
        let probs = vec![1.0, 0.0, 0.0, 0.0];
        let h = shannon_entropy(&probs);
        assert!((h - 0.0).abs() < 1e-5, "got {}", h);
    }

    #[test]
    fn test_shannon_entropy_binary() {
        // Binary: [0.5, 0.5] → H = 1.0
        let probs = vec![0.5, 0.5];
        let h = shannon_entropy(&probs);
        assert!((h - 1.0).abs() < 1e-5, "got {}", h);
    }

    #[test]
    fn test_shannon_entropy_skewed() {
        // Skewed: [0.9, 0.1] → H ≈ 0.469
        let probs = vec![0.9, 0.1];
        let h = shannon_entropy(&probs);
        let expected = -(0.9 * 0.9f32.log2() + 0.1 * 0.1f32.log2());
        assert!((h - expected).abs() < 1e-5, "got {}", h);
    }

    #[test]
    fn test_entropy_tracker_empty() {
        let t = EntropyTracker::new();
        assert!(t.records().is_empty());
    }

    #[test]
    fn test_entropy_tracker_record_from_scores() {
        let mut t = EntropyTracker::new();
        // 2 heads, stride=4, cache_seq_len=4
        // Head 0: uniform [0.25, 0.25, 0.25, 0.25] → H = 2.0
        // Head 1: peaked [1.0, 0.0, 0.0, 0.0] → H = 0.0
        let scores = vec![
            0.25, 0.25, 0.25, 0.25, // head 0
            1.0, 0.0, 0.0, 0.0, // head 1
        ];
        t.record_from_scores(0, &scores, 4, 4, 2);

        assert_eq!(t.records().len(), 1);
        let r = &t.records()[0];
        assert_eq!(r.step, 0);
        assert!((r.avg_entropy - 1.0).abs() < 1e-5); // (2.0 + 0.0) / 2
        assert!((r.min_entropy - 0.0).abs() < 1e-5);
        assert!((r.max_entropy - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_entropy_tracker_multiple_steps() {
        let mut t = EntropyTracker::new();
        // 1 head, stride=2, cache_seq_len=2
        let scores1 = vec![0.5, 0.5]; // H = 1.0
        let scores2 = vec![0.9, 0.1]; // H ≈ 0.469

        t.record_from_scores(0, &scores1, 2, 2, 1);
        t.record_from_scores(1, &scores2, 2, 2, 1);

        assert_eq!(t.records().len(), 2);
        assert!((t.records()[0].avg_entropy - 1.0).abs() < 1e-5);
        assert!(t.records()[1].avg_entropy < 1.0); // more peaked
    }

    #[test]
    fn test_entropy_tracker_zero_heads_noop() {
        let mut t = EntropyTracker::new();
        t.record_from_scores(0, &[], 0, 0, 0);
        assert!(t.records().is_empty());
    }

    #[test]
    fn test_entropy_tracker_to_json() {
        let mut t = EntropyTracker::new();
        let scores = vec![0.25, 0.25, 0.25, 0.25];
        t.record_from_scores(0, &scores, 4, 4, 1);

        let json = t.to_json();
        let records = json["records"].as_array().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["step"], 0);
        assert!((records[0]["avg_entropy"].as_f64().unwrap() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_tracker_to_json_empty() {
        let t = EntropyTracker::new();
        let json = t.to_json();
        let records = json["records"].as_array().unwrap();
        assert!(records.is_empty());
    }
}
