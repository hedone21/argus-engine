//! `QuantWindowFlushHook`: the `StepHook` the quant-window eval mode runs on.
//!
//! The quant-window cache does not evict — when its FP32 residual buffer fills it batch-quantizes
//! those tokens to Q2. This hook is what lets `run_eval_ll_generic` drive that cache: it owns the
//! per-choice snapshot/restore, the optional score accumulator, and the two occupancy numbers the
//! eval record carries.
//!
//! It used to also collect the flush's NMSE/OPR degradation proxies and dump them as QCF columns.
//! That went with the QCF metric family; the cache no longer produces the proxies and the hook no
//! longer reports them. Everything that made the quant-window eval mode *run* is unchanged.

use super::hook::{CacheSnapshot, StepHook};
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::kv::quant_window_cache::QuantizedRecentWindowCache;

/// QuantizedRecentWindowCache snapshot for choice-level restore.
///
/// Uses `Clone` because `QuantizedRecentWindowCache` implements `Clone` and all fields are heap-allocated.
pub struct QuantWindowCacheSnapshot {
    caches: Vec<QuantizedRecentWindowCache>,
}

impl CacheSnapshot<QuantizedRecentWindowCache> for QuantWindowCacheSnapshot {
    fn restore_to(&self, caches: &mut [QuantizedRecentWindowCache]) {
        caches.clone_from_slice(&self.caches);
    }
}

/// StepHook for the quant-window eval mode.
///
/// Does not perform eviction; `PostStepResult` is always the default (no eviction).
pub struct QuantWindowFlushHook {
    /// Optional attention score accumulator.
    pub score_accumulator: Option<AttentionScoreAccumulator>,
}

impl QuantWindowFlushHook {
    pub fn new(score_accumulator: Option<AttentionScoreAccumulator>) -> Self {
        Self { score_accumulator }
    }
}

impl StepHook<QuantizedRecentWindowCache> for QuantWindowFlushHook {
    fn post_prefill(&mut self, _caches: &mut [QuantizedRecentWindowCache]) {}

    fn reset_caches(&mut self, caches: &mut [QuantizedRecentWindowCache]) {
        for cache in caches.iter_mut() {
            cache.reset();
        }
        if let Some(ref mut acc) = self.score_accumulator {
            acc.reset();
        }
    }

    fn snapshot(
        &self,
        caches: &[QuantizedRecentWindowCache],
    ) -> Box<dyn CacheSnapshot<QuantizedRecentWindowCache>> {
        Box::new(QuantWindowCacheSnapshot {
            caches: caches.to_vec(),
        })
    }

    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator> {
        self.score_accumulator.as_mut()
    }

    fn extra_question_fields(&self, caches: &[QuantizedRecentWindowCache]) -> serde_json::Value {
        let (q2_tokens, res_pos) = if caches.is_empty() {
            (0, 0)
        } else {
            (caches[0].q2_tokens, caches[0].res_pos)
        };
        serde_json::json!({
            "quant_q2_tokens": q2_tokens,
            "quant_res_pos": res_pos,
        })
    }

    fn extra_config_fields(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::quant_window_cache::QuantizedRecentWindowCache;

    fn make_hook() -> QuantWindowFlushHook {
        QuantWindowFlushHook::new(None)
    }

    #[test]
    fn occupancy_reads_zero_with_no_caches() {
        let fields = make_hook().extra_question_fields(&[]);
        assert_eq!(fields["quant_q2_tokens"], 0);
        assert_eq!(fields["quant_res_pos"], 0);
    }

    #[test]
    fn occupancy_reads_the_first_cache() {
        let cache = QuantizedRecentWindowCache::new(8, 64, 512, 32);
        let fields = make_hook().extra_question_fields(&[cache]);
        assert_eq!(fields["quant_q2_tokens"], 0);
        assert_eq!(fields["quant_res_pos"], 0);
    }

    #[test]
    fn the_record_carries_no_qcf_columns() {
        let cache = QuantizedRecentWindowCache::new(8, 64, 512, 32);
        let fields = make_hook().extra_question_fields(&[cache]);
        let obj = fields.as_object().expect("object");
        assert!(
            obj.keys()
                .all(|k| !k.contains("qcf") && k != "quant_flush_count"),
            "a QCF column survived the removal: {:?}",
            obj.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn score_accumulator_is_absent_unless_supplied() {
        assert!(make_hook().score_accumulator().is_none());
    }

    #[test]
    fn score_accumulator_is_returned_when_supplied() {
        let acc = AttentionScoreAccumulator::new(512, 32, 16, 0, 1.0);
        let mut hook = QuantWindowFlushHook::new(Some(acc));
        assert!(hook.score_accumulator().is_some());
    }

    #[test]
    fn reset_clears_the_caches() {
        let mut hook = make_hook();
        let mut caches = [QuantizedRecentWindowCache::new(8, 64, 512, 32)];
        hook.reset_caches(&mut caches);
        assert_eq!(caches[0].res_pos, 0);
    }
}
