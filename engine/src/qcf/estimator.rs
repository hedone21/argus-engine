//! Piecewise-linear degradation estimator with EMA correction.
//!
//! Converts raw proxy values to estimated PPL increase via offline-calibrated
//! coefficients. Supports JSON calibration files and runtime EMA updates.

use super::QcfMetric;
use std::collections::HashMap;

/// Piecewise-linear function with one breakpoint.
///
/// ```text
/// f(x) = slope_low × x                          if x < breakpoint
///       = slope_low × breakpoint
///         + slope_high × (x - breakpoint)        if x >= breakpoint
/// ```
#[derive(Debug, Clone)]
pub struct PiecewiseLinear {
    pub breakpoint: f32,
    pub slope_low: f32,
    pub slope_high: f32,
}

impl PiecewiseLinear {
    pub fn new(breakpoint: f32, slope_low: f32, slope_high: f32) -> Self {
        Self {
            breakpoint,
            slope_low,
            slope_high,
        }
    }

    /// Simple linear (no breakpoint).
    pub fn linear(slope: f32) -> Self {
        Self {
            breakpoint: f32::INFINITY,
            slope_low: slope,
            slope_high: slope,
        }
    }

    /// Evaluate the piecewise-linear function at `x`.
    pub fn evaluate(&self, x: f32) -> f32 {
        if x < self.breakpoint {
            self.slope_low * x
        } else {
            self.slope_low * self.breakpoint + self.slope_high * (x - self.breakpoint)
        }
    }
}

/// Converts proxy values to estimated PPL increase (degradation).
///
/// Each action has a calibrated `PiecewiseLinear` mapping.
/// Runtime observations can be used to correct via EMA.
pub struct DegradationEstimator {
    /// Per-action calibration curves.
    curves: HashMap<String, PiecewiseLinear>,
    /// Maximum degradation estimate (clamp ceiling).
    d_max: f32,
    /// EMA smoothing factor for runtime correction (0 = no correction).
    ema_alpha: f32,
    /// Running EMA correction factors per action.
    ema_corrections: HashMap<String, f32>,
}

impl DegradationEstimator {
    /// Create an estimator with default linear slopes (no calibration data).
    ///
    /// Default slopes: all actions map 1:1 (slope=1.0).
    pub fn with_defaults(d_max: f32) -> Self {
        // 곡선 키는 IPC estimates 키와 일치해야 한다 (qcf_runtime.rs
        // estimates.insert 및 ENG-ALG-050 키 정렬 노트 참조). B안에서 live
        // 미장착이라 동작 무변화이나, 키를 정렬해 두어 향후 캘리브레이션
        // 장착 시 silent 미적용(키 불일치 → 곡선 fallback)을 방지한다.
        let mut curves = HashMap::new();
        curves.insert("kv.evict_h2o".to_string(), PiecewiseLinear::linear(1.0));
        curves.insert("kv.evict_sliding".to_string(), PiecewiseLinear::linear(1.0));
        curves.insert(
            "kv.evict_streaming".to_string(),
            PiecewiseLinear::linear(1.0),
        );
        curves.insert("kv.merge_d2o".to_string(), PiecewiseLinear::linear(1.0));
        curves.insert("kv.quant_dynamic".to_string(), PiecewiseLinear::linear(1.0));
        curves.insert("weight.skip".to_string(), PiecewiseLinear::linear(1.0));

        Self {
            curves,
            d_max,
            ema_alpha: 0.0,
            ema_corrections: HashMap::new(),
        }
    }

    /// Create from a calibration map (action → PiecewiseLinear).
    pub fn new(curves: HashMap<String, PiecewiseLinear>, d_max: f32, ema_alpha: f32) -> Self {
        Self {
            curves,
            d_max,
            ema_alpha,
            ema_corrections: HashMap::new(),
        }
    }

    /// Load calibration from a JSON file.
    ///
    /// Expected format:
    /// ```json
    /// {
    ///   "d_max": 5.0,
    ///   "ema_alpha": 0.1,
    ///   "actions": {
    ///     "kv.evict_h2o": { "breakpoint": 0.3, "slope_low": 2.0, "slope_high": 8.0 },
    ///     "kv.quant_dynamic": { "breakpoint": 0.05, "slope_low": 10.0, "slope_high": 50.0 }
    ///   }
    /// }
    /// ```
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;

        let d_max = json["d_max"].as_f64().unwrap_or(5.0) as f32;
        let ema_alpha = json["ema_alpha"].as_f64().unwrap_or(0.1) as f32;

        let mut curves = HashMap::new();
        if let Some(actions) = json["actions"].as_object() {
            for (name, cfg) in actions {
                let bp = cfg["breakpoint"].as_f64().unwrap_or(f64::INFINITY) as f32;
                let sl = cfg["slope_low"].as_f64().unwrap_or(1.0) as f32;
                let sh = cfg["slope_high"].as_f64().unwrap_or(1.0) as f32;
                curves.insert(name.clone(), PiecewiseLinear::new(bp, sl, sh));
            }
        }

        Ok(Self {
            curves,
            d_max,
            ema_alpha,
            ema_corrections: HashMap::new(),
        })
    }

    /// Estimate degradation (PPL increase) from a proxy metric.
    ///
    /// Returns `α(proxy_value)` clamped to `[0, d_max]`, with optional EMA correction.
    pub fn estimate(&self, metric: &QcfMetric) -> f32 {
        let curve = self
            .curves
            .get(&metric.action)
            .cloned()
            .unwrap_or_else(|| PiecewiseLinear::linear(1.0));

        let base = curve.evaluate(metric.raw_value);

        // Apply EMA correction if available
        let correction = self
            .ema_corrections
            .get(&metric.action)
            .copied()
            .unwrap_or(1.0);

        (base * correction).clamp(0.0, self.d_max)
    }

    /// Update EMA correction from observed (proxy, actual_degradation) pair.
    ///
    /// `actual_d`: the measured PPL increase from periodic calibration.
    pub fn update_ema(&mut self, action: &str, proxy_value: f32, actual_d: f32) {
        if self.ema_alpha <= 0.0 || proxy_value < 1e-8 {
            return;
        }

        let curve = self
            .curves
            .get(action)
            .cloned()
            .unwrap_or_else(|| PiecewiseLinear::linear(1.0));

        let predicted = curve.evaluate(proxy_value);
        if predicted < 1e-8 {
            return;
        }

        let observed_ratio = actual_d / predicted;
        let current = self.ema_corrections.get(action).copied().unwrap_or(1.0);
        let updated = (1.0 - self.ema_alpha) * current + self.ema_alpha * observed_ratio;
        self.ema_corrections.insert(action.to_string(), updated);
    }

    /// Get the current EMA correction factor for an action.
    pub fn ema_correction(&self, action: &str) -> f32 {
        self.ema_corrections.get(action).copied().unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piecewise_linear_below_breakpoint() {
        let pw = PiecewiseLinear::new(0.3, 2.0, 8.0);
        assert!((pw.evaluate(0.1) - 0.2).abs() < 1e-6);
        assert!((pw.evaluate(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_piecewise_linear_above_breakpoint() {
        let pw = PiecewiseLinear::new(0.3, 2.0, 8.0);
        // f(0.5) = 2.0*0.3 + 8.0*(0.5-0.3) = 0.6 + 1.6 = 2.2
        assert!((pw.evaluate(0.5) - 2.2).abs() < 1e-5);
    }

    #[test]
    fn test_piecewise_linear_at_breakpoint() {
        let pw = PiecewiseLinear::new(0.3, 2.0, 8.0);
        // At breakpoint: both branches agree
        assert!((pw.evaluate(0.3) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_linear_no_breakpoint() {
        let pw = PiecewiseLinear::linear(3.0);
        assert!((pw.evaluate(0.5) - 1.5).abs() < 1e-6);
        assert!((pw.evaluate(2.0) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimator_defaults() {
        let est = DegradationEstimator::with_defaults(5.0);
        let metric = QcfMetric {
            action: "kv.evict_h2o".to_string(),
            raw_value: 0.3,
            normalized_value: 0.3,
            per_head: None,
            tokens_affected: 10,
        };
        let d = est.estimate(&metric);
        assert!((d - 0.3).abs() < 1e-6); // slope=1.0 → d = proxy
    }

    #[test]
    fn test_estimator_d_max_clamp() {
        let est = DegradationEstimator::with_defaults(2.0);
        let metric = QcfMetric {
            action: "kv.evict_h2o".to_string(),
            raw_value: 5.0, // Would give d=5.0 but clamped to 2.0
            normalized_value: 5.0,
            per_head: None,
            tokens_affected: 10,
        };
        let d = est.estimate(&metric);
        assert!((d - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimator_unknown_action() {
        let est = DegradationEstimator::with_defaults(5.0);
        let metric = QcfMetric {
            action: "unknown_action".to_string(),
            raw_value: 0.5,
            normalized_value: 0.5,
            per_head: None,
            tokens_affected: 1,
        };
        let d = est.estimate(&metric);
        // Falls back to linear slope=1.0
        assert!((d - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_estimator_custom_curves() {
        let mut curves = HashMap::new();
        curves.insert(
            "kv.evict_h2o".to_string(),
            PiecewiseLinear::new(0.3, 2.0, 10.0),
        );

        let est = DegradationEstimator::new(curves, 5.0, 0.0);
        let metric = QcfMetric {
            action: "kv.evict_h2o".to_string(),
            raw_value: 0.1, // Below breakpoint: 2.0 * 0.1 = 0.2
            normalized_value: 0.1,
            per_head: None,
            tokens_affected: 5,
        };
        assert!((est.estimate(&metric) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_ema_correction() {
        let mut est = DegradationEstimator::new(
            {
                let mut m = HashMap::new();
                m.insert("kv.evict_h2o".to_string(), PiecewiseLinear::linear(2.0));
                m
            },
            10.0,
            0.5, // Aggressive EMA for test
        );

        // Predicted: 2.0 * 0.3 = 0.6
        // Actual: 1.2 → ratio = 1.2/0.6 = 2.0
        est.update_ema("kv.evict_h2o", 0.3, 1.2);
        let correction = est.ema_correction("kv.evict_h2o");
        // EMA: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        assert!((correction - 1.5).abs() < 1e-5);

        // Now estimate with correction
        let metric = QcfMetric {
            action: "kv.evict_h2o".to_string(),
            raw_value: 0.3,
            normalized_value: 0.3,
            per_head: None,
            tokens_affected: 5,
        };
        let d = est.estimate(&metric);
        // 2.0 * 0.3 * 1.5 = 0.9
        assert!((d - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_ema_no_update_when_alpha_zero() {
        let mut est = DegradationEstimator::new(
            {
                let mut m = HashMap::new();
                m.insert("kv.evict_h2o".to_string(), PiecewiseLinear::linear(1.0));
                m
            },
            5.0,
            0.0, // No EMA
        );

        est.update_ema("kv.evict_h2o", 0.3, 1.2);
        assert_eq!(est.ema_correction("kv.evict_h2o"), 1.0); // Unchanged
    }

    #[test]
    fn test_load_json() {
        let dir = std::env::temp_dir().join("proxy_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("calibration.json");

        let json = r#"{
            "d_max": 3.0,
            "ema_alpha": 0.2,
            "actions": {
                "kv.evict_h2o": { "breakpoint": 0.5, "slope_low": 1.5, "slope_high": 5.0 },
                "kv.quant_dynamic": { "breakpoint": 0.1, "slope_low": 10.0, "slope_high": 30.0 }
            }
        }"#;
        std::fs::write(&path, json).unwrap();

        let est = DegradationEstimator::load(path.to_str().unwrap()).unwrap();
        let metric = QcfMetric {
            action: "kv.evict_h2o".to_string(),
            raw_value: 0.2,
            normalized_value: 0.2,
            per_head: None,
            tokens_affected: 5,
        };
        // slope_low=1.5, x=0.2 < bp=0.5 → 1.5 * 0.2 = 0.3
        assert!((est.estimate(&metric) - 0.3).abs() < 1e-5);

        std::fs::remove_dir_all(dir).ok();
    }
}
