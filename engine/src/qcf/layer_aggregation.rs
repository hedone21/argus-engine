//! Layer-level QCF aggregation (ARGUS QCF improvement #1).
//!
//! Mirrors the per-head aggregation API but operates over per-layer scalars.

/// Layer-level aggregation strategy.
#[derive(Debug, Clone)]
pub enum LayerAggregationMode {
    /// Arithmetic mean.
    Mean,
    /// Maximum across layers.
    Max,
    /// Softmax-weighted (low temperature → max-like).
    Defensive { temperature: f32 },
    /// Mean of the top fraction of layers (e.g., 0.3 = last 30%).
    LateFocused { fraction: f32 },
}

/// Aggregate per-layer QCF scalars into a single value.
///
/// Empty input returns `0.0`.
pub fn aggregate_layers(per_layer: &[f32], mode: &LayerAggregationMode) -> f32 {
    if per_layer.is_empty() {
        return 0.0;
    }
    match mode {
        LayerAggregationMode::Mean => per_layer.iter().sum::<f32>() / per_layer.len() as f32,
        LayerAggregationMode::Max => per_layer.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        LayerAggregationMode::Defensive { temperature } => {
            let temp = temperature.max(1e-6);
            let max_val = per_layer.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = per_layer
                .iter()
                .map(|&v| ((v - max_val) / temp).exp())
                .sum();
            if exp_sum < 1e-12 {
                return per_layer.iter().sum::<f32>() / per_layer.len() as f32;
            }
            per_layer
                .iter()
                .map(|&v| {
                    let w = ((v - max_val) / temp).exp() / exp_sum;
                    w * v
                })
                .sum()
        }
        LayerAggregationMode::LateFocused { fraction } => {
            let n = per_layer.len();
            let frac = fraction.clamp(0.0, 1.0);
            let take = ((n as f32) * frac).ceil().max(1.0) as usize;
            let start = n - take.min(n);
            per_layer[start..].iter().sum::<f32>() / (n - start) as f32
        }
    }
}

/// QCF D7 — dispersion ratio of a per-layer scalar series.
///
/// `D7 = (max(L) - min(L)) / (max(L) + eps)` where `L` is the per-layer scalar
/// (worst-head or mean-head). EuroSys'27 §3 main QCF metric.
///
/// `eps` = 1e-6 (hard-coded). Empty input → returns 0.0.
pub fn compute_d7(per_layer: &[f32]) -> f32 {
    if per_layer.is_empty() {
        return 0.0;
    }
    let max_l = per_layer.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_l = per_layer.iter().copied().fold(f32::INFINITY, f32::min);
    let eps = 1e-6_f32;
    (max_l - min_l) / (max_l + eps)
}

/// QCF C1 — D7 with population standard deviation as a magnitude tie-breaker.
///
/// `C1 = D7(L) + std_pop(L)` where `std_pop` uses divisor `n` (matching NumPy's
/// `np.std` default). EuroSys'27 §3 supplementary refinement.
///
/// Empty input → returns 0.0.
pub fn compute_c1(per_layer: &[f32]) -> f32 {
    if per_layer.is_empty() {
        return 0.0;
    }
    let n = per_layer.len() as f32;
    let mean = per_layer.iter().sum::<f32>() / n;
    let var = per_layer.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let std_pop = var.sqrt();
    compute_d7(per_layer) + std_pop
}

/// Compute auto sample-layer indices: `[0, n/4, n/2, 3n/4, n-1]` for `n >= 5`,
/// otherwise all layers.
///
/// Produces 5 evenly-spaced indices: `[0, n/4, n/2, 3*n/4, n-1]` using
/// integer arithmetic.  For `n=32`: `[0, 8, 16, 24, 31]`.
pub fn compute_auto_sample_layers(n_layers: usize) -> Vec<usize> {
    if n_layers == 0 {
        return Vec::new();
    }
    if n_layers <= 5 {
        return (0..n_layers).collect();
    }
    // 5 equally-spaced points: 0, n/4, n/2, 3n/4, n-1.
    // The first four use integer n/4 steps; the last is always n-1.
    let step = n_layers / 4;
    let mut indices: Vec<usize> = (0..4).map(|i| i * step).collect();
    indices.push(n_layers - 1);
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_layers_empty() {
        assert_eq!(aggregate_layers(&[], &LayerAggregationMode::Max), 0.0);
        assert_eq!(aggregate_layers(&[], &LayerAggregationMode::Mean), 0.0);
    }

    #[test]
    fn aggregate_layers_max() {
        let v = vec![0.1, 0.5, 0.2, 0.4];
        assert!((aggregate_layers(&v, &LayerAggregationMode::Max) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn aggregate_layers_mean() {
        let v = vec![0.1, 0.2, 0.3];
        assert!((aggregate_layers(&v, &LayerAggregationMode::Mean) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn aggregate_layers_defensive_emphasises_max() {
        let v = vec![0.1, 0.1, 0.9];
        let mean = aggregate_layers(&v, &LayerAggregationMode::Mean);
        let def = aggregate_layers(&v, &LayerAggregationMode::Defensive { temperature: 0.1 });
        assert!(def > mean);
        assert!(def > 0.5);
    }

    #[test]
    fn aggregate_layers_late_focused_30pct() {
        // 10 layers, fraction=0.3 → last 3 layers (ceil(10*0.3)=3).
        let v: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let r = aggregate_layers(&v, &LayerAggregationMode::LateFocused { fraction: 0.3 });
        // Last 3 = [7, 8, 9] → mean = 8.0
        assert!((r - 8.0).abs() < 1e-6);
    }

    #[test]
    fn aggregate_layers_late_focused_full() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        // fraction=1.0 → all layers
        let r = aggregate_layers(&v, &LayerAggregationMode::LateFocused { fraction: 1.0 });
        assert!((r - 2.5).abs() < 1e-6);
    }

    #[test]
    fn aggregate_layers_late_focused_zero_takes_one() {
        // fraction=0 → ceil(0) clamped to >=1 → last layer only
        let v = vec![1.0, 2.0, 3.0];
        let r = aggregate_layers(&v, &LayerAggregationMode::LateFocused { fraction: 0.0 });
        assert!((r - 3.0).abs() < 1e-6);
    }

    #[test]
    fn compute_d7_empty() {
        assert_eq!(compute_d7(&[]), 0.0);
    }

    #[test]
    fn compute_d7_uniform() {
        // Uniform layers → max == min → D7 == 0
        let v = vec![0.3, 0.3, 0.3, 0.3];
        assert!((compute_d7(&v)).abs() < 1e-5);
    }

    #[test]
    fn compute_d7_concentrated() {
        // [0.1, 0.5, 0.3] → (0.5 - 0.1) / (0.5 + 1e-6) ≈ 0.79999..
        let v = vec![0.1, 0.5, 0.3];
        let d7 = compute_d7(&v);
        assert!((d7 - 0.8).abs() < 1e-3);
    }

    #[test]
    fn compute_d7_full_concentration() {
        // [0.0, 0.0, 1.0] → (1 - 0) / (1 + eps) ≈ 1.0
        let v = vec![0.0, 0.0, 1.0];
        let d7 = compute_d7(&v);
        assert!((d7 - 1.0).abs() < 1e-3);
    }

    #[test]
    fn compute_c1_empty() {
        assert_eq!(compute_c1(&[]), 0.0);
    }

    #[test]
    fn compute_c1_uniform() {
        // Uniform → D7 = 0, std = 0 → C1 = 0
        let v = vec![0.4, 0.4, 0.4];
        assert!((compute_c1(&v)).abs() < 1e-5);
    }

    #[test]
    fn compute_c1_concentrated() {
        // [0.0, 0.0, 1.0] → D7 ≈ 1.0, mean = 1/3, var_pop = 2/9, std ≈ 0.4714
        // C1 ≈ 1.4714
        let v = vec![0.0_f32, 0.0, 1.0];
        let c1 = compute_c1(&v);
        let expected = 1.0 + (2.0_f32 / 9.0).sqrt();
        assert!((c1 - expected).abs() < 1e-3);
    }

    #[test]
    fn auto_sample_layers_small() {
        assert_eq!(compute_auto_sample_layers(0), Vec::<usize>::new());
        assert_eq!(compute_auto_sample_layers(3), vec![0, 1, 2]);
        assert_eq!(compute_auto_sample_layers(5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn auto_sample_layers_llama3_8b() {
        // 32 layers → [0, 8, 16, 24, 31]
        assert_eq!(compute_auto_sample_layers(32), vec![0, 8, 16, 24, 31]);
    }

    #[test]
    fn auto_sample_layers_llama3_3b() {
        // 28 layers → [0, 7, 14, 21, 27]
        assert_eq!(compute_auto_sample_layers(28), vec![0, 7, 14, 21, 27]);
    }

    #[test]
    fn auto_sample_layers_gemma_1b() {
        // 26 layers → [0, ~6, ~13, ~19, 25]
        let v = compute_auto_sample_layers(26);
        assert_eq!(v[0], 0);
        assert_eq!(v[4], 25);
        assert_eq!(v.len(), 5);
    }
}
