//! KiviHook: StepHook implementation for KIVI quantization flush metric collection.
//!
//! Encapsulates the KIVI flush proxy collection logic previously embedded in
//! `run_kivi_eval_ll` (generate.rs).
//!
//! KIVI does not perform eviction — instead, when the FP32 residual buffer fills,
//! it batch-quantizes those tokens to Q2 and records the NMSE/OPR degradation as
//! a `QcfMetric`. KiviHook collects those metrics from `take_flush_proxies()` after
//! each prefill and decode step.

use super::hook::{CacheSnapshot, StepHook};
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::kv::kivi_cache::KiviCache;

/// KiviCache snapshot for choice-level restore.
///
/// Uses `Clone` because `KiviCache` implements `Clone` and all fields are heap-allocated.
pub struct KiviCacheSnapshot {
    caches: Vec<KiviCache>,
}

impl CacheSnapshot<KiviCache> for KiviCacheSnapshot {
    fn restore_to(&self, caches: &mut [KiviCache]) {
        caches.clone_from_slice(&self.caches);
    }
}

/// StepHook for KIVI quantization flush metric collection.
///
/// After each prefill and decode step, drains `take_flush_proxies()` from
/// the sample layers (default: layer 0 only, matching legacy behaviour).
/// Metrics from non-sample layers are drained and discarded.
///
/// Does not perform eviction; `PostStepResult` is always the default (no eviction).
pub struct KiviHook {
    /// QCF metric collection config (used for aggregation strategy).
    pub qcf_config: crate::qcf_types::QcfConfig,
    /// Whether to compute and dump experimental QCF metrics (ARGUS Step 5).
    pub experimental_enabled: bool,
    /// Sample layer indices for multi-layer flush proxy (ARGUS Step 5).
    /// Empty → use [0] for backward compat.
    pub qcf_sample_layers: Vec<usize>,
    /// Optional attention score accumulator (for entropy dump).
    pub score_accumulator: Option<AttentionScoreAccumulator>,

    /// Running count of flush events for the current question (layer 0 legacy,
    /// dumped as `kivi_flush_count`).
    flush_count: usize,

    // Schema v3: per-layer accumulation. Buffers indexed by `sample_layers` position.
    /// Per-layer worst-head accumulator (max OPR observed at each sampled layer).
    per_layer_max: Vec<f32>,
    /// Per-layer running sum (for mean-head dump).
    per_layer_sum: Vec<f32>,
    /// Per-layer flush count (denominator of mean-head).
    per_layer_count: Vec<usize>,
}

impl KiviHook {
    pub fn new(
        qcf_config: crate::qcf_types::QcfConfig,
        experimental_enabled: bool,
        qcf_sample_layers: Vec<usize>,
        score_accumulator: Option<AttentionScoreAccumulator>,
    ) -> Self {
        let n = qcf_sample_layers.len().max(1);
        Self {
            qcf_config,
            experimental_enabled,
            qcf_sample_layers,
            score_accumulator,
            flush_count: 0,
            per_layer_max: vec![0.0; n],
            per_layer_sum: vec![0.0; n],
            per_layer_count: vec![0; n],
        }
    }

    /// Drain flush proxies from sample layers, track max OPR and flush count.
    ///
    /// Legacy path (experimental_enabled=false): layer 0 only.
    /// Experimental path: all sample_layers, with per-layer buffers.
    fn collect_flush_proxies(&mut self, caches: &mut [KiviCache]) {
        if caches.is_empty() {
            return;
        }

        if self.experimental_enabled {
            let sample_layers: Vec<usize> = if self.qcf_sample_layers.is_empty() {
                vec![0]
            } else {
                self.qcf_sample_layers.clone()
            };

            // Ensure per-layer buffers are sized correctly.
            if self.per_layer_max.len() != sample_layers.len() {
                self.per_layer_max = vec![0.0; sample_layers.len()];
                self.per_layer_sum = vec![0.0; sample_layers.len()];
                self.per_layer_count = vec![0; sample_layers.len()];
            }

            let sample_set: std::collections::HashSet<usize> =
                sample_layers.iter().copied().collect();

            for (cache_idx, cache) in caches.iter_mut().enumerate() {
                if sample_set.contains(&cache_idx) {
                    let pos = sample_layers.iter().position(|&v| v == cache_idx).unwrap();
                    for metric in cache.take_flush_proxies() {
                        if metric.action == "kivi_opr" {
                            if cache_idx == 0 {
                                self.flush_count += 1;
                            }
                            self.per_layer_max[pos] = self.per_layer_max[pos].max(metric.raw_value);
                            self.per_layer_sum[pos] += metric.raw_value;
                            self.per_layer_count[pos] += 1;
                        }
                        // "aw_vopr" and NMSE proxies are drained but not exposed in v3.
                    }
                } else {
                    cache.take_flush_proxies();
                }
            }
        } else {
            // Legacy path: layer 0 only, no per-layer tracking.
            for metric in caches[0].take_flush_proxies() {
                if metric.action == "kivi_opr" {
                    self.flush_count += 1;
                }
            }
            for cache in caches[1..].iter_mut() {
                cache.take_flush_proxies();
            }
        }
    }
}

impl StepHook<KiviCache> for KiviHook {
    fn post_prefill(&mut self, caches: &mut [KiviCache]) {
        self.collect_flush_proxies(caches);
    }

    fn reset_caches(&mut self, caches: &mut [KiviCache]) {
        for cache in caches.iter_mut() {
            cache.reset();
        }
        self.flush_count = 0;
        for v in self.per_layer_max.iter_mut() {
            *v = 0.0;
        }
        for v in self.per_layer_sum.iter_mut() {
            *v = 0.0;
        }
        for v in self.per_layer_count.iter_mut() {
            *v = 0;
        }
        if let Some(ref mut acc) = self.score_accumulator {
            acc.reset();
        }
    }

    fn snapshot(&self, caches: &[KiviCache]) -> Box<dyn CacheSnapshot<KiviCache>> {
        Box::new(KiviCacheSnapshot {
            caches: caches.to_vec(),
        })
    }

    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator> {
        self.score_accumulator.as_mut()
    }

    fn extra_question_fields(&self, caches: &[KiviCache]) -> serde_json::Value {
        let (q2_tokens, res_pos) = if caches.is_empty() {
            (0, 0)
        } else {
            (caches[0].q2_tokens, caches[0].res_pos)
        };
        let mut obj = serde_json::json!({
            "kivi_q2_tokens": q2_tokens,
            "kivi_res_pos": res_pos,
            "kivi_flush_count": self.flush_count,
        });

        if self.experimental_enabled {
            let indices: Vec<usize> = if self.qcf_sample_layers.is_empty() {
                vec![0]
            } else {
                self.qcf_sample_layers.clone()
            };
            let n = indices.len();
            let layer_worst_head: Vec<f32> = self.per_layer_max.iter().copied().take(n).collect();
            let layer_mean_head: Vec<f32> = self
                .per_layer_sum
                .iter()
                .zip(self.per_layer_count.iter())
                .take(n)
                .map(|(s, c)| if *c > 0 { *s / (*c as f32) } else { 0.0 })
                .collect();

            let max_or_zero = |s: &[f32]| -> f32 {
                if s.is_empty() {
                    0.0
                } else {
                    s.iter().copied().fold(f32::NEG_INFINITY, f32::max).max(0.0)
                }
            };
            let mean_or_zero = |s: &[f32]| -> f32 {
                if s.is_empty() {
                    0.0
                } else {
                    s.iter().sum::<f32>() / s.len() as f32
                }
            };

            use crate::qcf::{compute_c1, compute_d7};
            obj["schema_version"] = serde_json::json!(3);
            obj["action_family"] = serde_json::json!("kivi");
            obj["n_layers"] = serde_json::json!(n);
            obj["qcf_record_worst_head_max"] = serde_json::json!(max_or_zero(&layer_worst_head));
            obj["qcf_record_worst_head_mean"] = serde_json::json!(mean_or_zero(&layer_worst_head));
            obj["qcf_record_mean_head_max"] = serde_json::json!(max_or_zero(&layer_mean_head));
            obj["qcf_record_mean_head_mean"] = serde_json::json!(mean_or_zero(&layer_mean_head));
            obj["qcf_d7_worst_head"] = serde_json::json!(compute_d7(&layer_worst_head));
            obj["qcf_d7_mean_head"] = serde_json::json!(compute_d7(&layer_mean_head));
            obj["qcf_c1_worst_head"] = serde_json::json!(compute_c1(&layer_worst_head));
            obj["qcf_c1_mean_head"] = serde_json::json!(compute_c1(&layer_mean_head));
            obj["qcf_layer_worst_head"] = serde_json::json!(layer_worst_head);
            obj["qcf_layer_mean_head"] = serde_json::json!(layer_mean_head);
        }

        obj
    }

    fn extra_config_fields(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::kivi_cache::KiviCache;
    use crate::qcf_types::QcfConfig;

    fn make_hook() -> KiviHook {
        KiviHook::new(QcfConfig::default(), false, vec![0], None)
    }

    fn make_hook_experimental(sample_layers: Vec<usize>) -> KiviHook {
        KiviHook::new(QcfConfig::default(), true, sample_layers, None)
    }

    #[test]
    fn test_extra_question_fields_empty() {
        let hook = make_hook();
        let fields = hook.extra_question_fields(&[]);
        assert_eq!(fields["kivi_q2_tokens"], 0);
        assert_eq!(fields["kivi_res_pos"], 0);
    }

    #[test]
    fn test_extra_question_fields_with_cache() {
        let hook = make_hook();
        let cache = KiviCache::new(8, 64, 512, 32);
        // Initial state: q2_tokens=0, res_pos=0
        let fields = hook.extra_question_fields(&[cache]);
        assert_eq!(fields["kivi_q2_tokens"], 0);
        assert_eq!(fields["kivi_res_pos"], 0);
    }

    #[test]
    fn test_flush_count_tracking_and_reset() {
        let mut hook = make_hook();
        let cache = KiviCache::new(8, 64, 512, 32);

        assert_eq!(hook.flush_count, 0);

        // Schema v3: kivi_flush_count is always emitted (no longer gated on > 0).
        hook.flush_count = 5;
        let fields = hook.extra_question_fields(&[cache]);
        assert_eq!(fields["kivi_flush_count"], 5);

        let cache2 = KiviCache::new(8, 64, 512, 32);
        hook.reset_caches(&mut [cache2]);
        assert_eq!(hook.flush_count, 0);
    }

    #[test]
    fn test_score_accumulator_returns_none() {
        let mut hook = make_hook();
        assert!(hook.score_accumulator().is_none());
    }

    #[test]
    fn test_score_accumulator_returns_some_when_provided() {
        use crate::inference::attention_scores::AttentionScoreAccumulator;
        // new(max_seq_len, n_heads, total_layers, last_n_layers, decay)
        let acc = AttentionScoreAccumulator::new(512, 32, 16, 0, 1.0);
        let mut hook = KiviHook::new(QcfConfig::default(), false, vec![0], Some(acc));
        assert!(
            hook.score_accumulator().is_some(),
            "score_accumulator() should return Some when acc is injected"
        );
    }

    #[test]
    fn test_extra_config_fields_empty() {
        let hook = make_hook();
        let fields = hook.extra_config_fields();
        // Should be an empty JSON object.
        assert!(fields.as_object().map(|o| o.is_empty()).unwrap_or(false));
    }

    #[test]
    fn test_snapshot_and_restore_empty() {
        let hook = make_hook();
        let snapshot = hook.snapshot(&[]);
        snapshot.restore_to(&mut []);
    }

    #[test]
    fn test_snapshot_and_restore_single_cache() {
        let hook = make_hook();
        let original = KiviCache::new(8, 64, 512, 32);
        let snap = hook.snapshot(std::slice::from_ref(&original));

        // Modify a second copy then restore.
        let mut modified = vec![KiviCache::new(8, 64, 512, 32)];
        snap.restore_to(&mut modified);
        // After restore, q2_tokens and res_pos should match the original.
        assert_eq!(modified[0].q2_tokens, original.q2_tokens);
        assert_eq!(modified[0].res_pos, original.res_pos);
    }

    #[test]
    fn test_reset_caches_resets_flush_count() {
        let mut hook = make_hook();
        hook.flush_count = 5;
        let mut caches = vec![KiviCache::new(8, 64, 512, 32)];
        hook.reset_caches(&mut caches);
        assert_eq!(hook.flush_count, 0);
    }

    // ── ARGUS Step 5 tests ──────────────────────────────────────────────────

    #[test]
    fn test_schema_v3_fields_when_experimental() {
        let hook = make_hook_experimental(vec![0, 4, 8]);
        let fields = hook.extra_question_fields(&[]);

        assert_eq!(fields["schema_version"], 3);
        assert_eq!(fields["action_family"], "kivi");
        assert_eq!(fields["n_layers"], 3);

        assert!(
            fields["qcf_layer_worst_head"].is_array(),
            "qcf_layer_worst_head must be an array"
        );
        assert!(
            fields["qcf_layer_mean_head"].is_array(),
            "qcf_layer_mean_head must be an array"
        );
        assert_eq!(fields["qcf_layer_worst_head"].as_array().unwrap().len(), 3);
        assert_eq!(fields["qcf_layer_mean_head"].as_array().unwrap().len(), 3);

        for k in [
            "qcf_record_worst_head_max",
            "qcf_record_worst_head_mean",
            "qcf_record_mean_head_max",
            "qcf_record_mean_head_mean",
            "qcf_d7_worst_head",
            "qcf_d7_mean_head",
            "qcf_c1_worst_head",
            "qcf_c1_mean_head",
        ] {
            assert!(fields[k].is_number(), "{k} must be a number");
        }
    }

    #[test]
    fn test_no_schema_v3_fields_when_not_experimental() {
        let hook = make_hook();
        let fields = hook.extra_question_fields(&[]);
        for k in [
            "schema_version",
            "qcf_layer_worst_head",
            "qcf_layer_mean_head",
            "qcf_d7_worst_head",
        ] {
            assert!(
                fields.get(k).is_none(),
                "{k} must be absent when experimental_enabled=false"
            );
        }
    }

    #[test]
    fn test_reset_caches_clears_per_layer_buffers() {
        let mut hook = make_hook_experimental(vec![0, 4]);
        hook.per_layer_max[0] = 0.9;
        hook.per_layer_sum[0] = 1.8;
        hook.per_layer_count[0] = 2;

        let mut caches = vec![KiviCache::new(8, 64, 512, 32)];
        hook.reset_caches(&mut caches);

        assert_eq!(hook.per_layer_max[0], 0.0);
        assert_eq!(hook.per_layer_sum[0], 0.0);
        assert_eq!(hook.per_layer_count[0], 0);
    }

    #[test]
    fn test_experimental_empty_sample_layers_fallback_to_layer0() {
        // Empty qcf_sample_layers → runtime fallback to [0] in dump.
        let hook = make_hook_experimental(vec![]);
        let fields = hook.extra_question_fields(&[]);
        // schema v3: n_layers reflects the fallback length (1).
        assert_eq!(fields["n_layers"], 1);
    }
}
