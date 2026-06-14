//! Bridge between existing `EvictionPolicy` and the new `CachePressureHandler` trait.
//!
//! Wraps any `Box<dyn EvictionPolicy>` so that H2O and SlidingWindow policies
//! work seamlessly inside a `CachePressurePipeline`.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use crate::kv::cache_manager::{CacheManager, ScoreContext};
use crate::kv::eviction::EvictionPolicy;
use anyhow::Result;

/// Minimum number of tokens that must be evicted to justify compaction overhead.
///
/// When `current_pos - target_len` falls below this threshold, eviction is
/// skipped and `ActionResult::NoOp` is returned. This prevents the "sticky
/// eviction" pattern where 18~22 tokens trigger a full sort+memcpy cycle for
/// only ~33 tokens removed, whose cost offsets the attention savings entirely.
pub const MIN_EVICT_TOKENS: usize = 64;

/// Adapts an `EvictionPolicy` to the `CachePressureHandler` interface.
///
/// The wrapped policy's `evict()` / `evict_with_scores()` is called with
/// `target_len = current_pos * target_ratio`.
pub struct EvictionHandler {
    policy: Box<dyn EvictionPolicy>,
    target_ratio: f32,
}

impl EvictionHandler {
    /// Create a new eviction handler wrapping the given policy.
    ///
    /// `target_ratio` controls how aggressively to evict:
    /// - 0.8 = keep 80% of tokens (mild)
    /// - 0.5 = keep 50% of tokens (aggressive)
    pub fn new(policy: Box<dyn EvictionPolicy>, target_ratio: f32) -> Self {
        Self {
            policy,
            target_ratio: target_ratio.clamp(0.1, 0.99),
        }
    }
}

impl CachePressureHandler for EvictionHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
        if ctx.caches.is_empty() {
            return Ok(ActionResult::NoOp);
        }

        let current_pos = ctx.caches[0].current_pos;
        // Use signal's target_ratio if provided, otherwise fall back to handler config.
        let effective_ratio = ctx.target_ratio.unwrap_or(self.target_ratio);
        let target_len = (((current_pos as f32) * effective_ratio) as usize).max(1);

        // α-K 2b: guard + per-policy dispatch + 결과 조립은 registry 경로와 공유하는
        // 단일 코어(CacheManager::run_policy_eviction)에 위임한다. EHH 는 target_len(≥1)을
        // 미리 해소해 넘기므로 코어의 `target_len > 0` 한정 guard 가 기존 거동과 동일하다.
        let scores = match (ctx.importance, ctx.head_importance) {
            (Some(flat), Some(head)) if ctx.n_kv_heads > 0 => ScoreContext::PerHead {
                flat,
                head,
                n_kv_heads: ctx.n_kv_heads,
            },
            (Some(flat), _) => ScoreContext::Flat { importance: flat },
            _ => ScoreContext::None,
        };

        let result = CacheManager::run_policy_eviction(
            self.policy.as_ref(),
            ctx.caches,
            target_len,
            scores,
        )?;

        if result.evicted {
            Ok(ActionResult::Evicted {
                tokens_removed: result.tokens_removed,
                new_pos: result.new_pos,
            })
        } else {
            Ok(ActionResult::NoOp)
        }
    }

    fn name(&self) -> &str {
        self.policy.name()
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::PressureLevel;
    use crate::kv::eviction::h2o::H2OPolicy;
    use crate::kv::eviction::no_eviction::NoEvictionPolicy;
    use crate::kv::eviction::sliding_window::SlidingWindowPolicy;
    use crate::kv::kv_cache::KVCache;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn make_caches(n_layers: usize, pos: usize) -> Vec<KVCache> {
        let max_seq = 100;
        let backend = Arc::new(CpuBackend::new());
        (0..n_layers)
            .map(|_| {
                let buf_size = max_seq * 4 * 4;
                let k = Tensor::new(
                    Shape::new(vec![1, max_seq, 1, 4]),
                    Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                    backend.clone(),
                );
                let v = Tensor::new(
                    Shape::new(vec![1, max_seq, 1, 4]),
                    Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                    backend.clone(),
                );
                let mut cache = KVCache::new(k, v, max_seq);
                cache.current_pos = pos;
                cache
            })
            .collect()
    }

    #[test]
    fn test_wraps_sliding_window() {
        // pos=100, target_ratio=0.3 → target_len=30, tokens_to_remove=70 >= MIN_EVICT_TOKENS(64).
        let handler = EvictionHandler::new(
            Box::new(SlidingWindowPolicy::new(10, 0)), // window=10, prefix=0
            0.3,
        );

        let mut caches = make_caches(4, 100);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted {
                tokens_removed,
                new_pos,
            } => {
                assert!(tokens_removed > 0);
                assert!(new_pos < 100);
                // SlidingWindow may clamp to internal min_keep; just verify significant reduction.
                assert!(
                    tokens_removed >= 64,
                    "Must remove >= MIN_EVICT_TOKENS(64), removed {}",
                    tokens_removed
                );
            }
            _ => panic!("Expected Evicted, got {:?}", result),
        }

        // All layers should have the same position
        let pos = ctx.caches[0].current_pos;
        for cache in ctx.caches.iter() {
            assert_eq!(cache.current_pos, pos);
        }
    }

    #[test]
    fn test_wraps_h2o_with_scores() {
        // pos=100, target_ratio=0.3 → target_len=30, tokens_to_remove=70 >= MIN_EVICT_TOKENS(64).
        let handler = EvictionHandler::new(
            Box::new(H2OPolicy::new(0.5, 0)), // prefix=0(clamped), keep_ratio=0.5
            0.3,
        );

        let mut caches = make_caches(4, 100);
        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[20] = 9.0;
        importance[30] = 8.0;
        for i in 4..100 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted {
                tokens_removed,
                new_pos,
            } => {
                assert!(tokens_removed > 0);
                assert!(new_pos < 100);
                // H2O: target = 100 * 0.3 = 30
                assert_eq!(new_pos, 30);
            }
            _ => panic!("Expected Evicted, got {:?}", result),
        }
    }

    #[test]
    fn test_noop_when_below_target() {
        // target_len = 1 * 0.99 = 0 → clamped to 1.
        // current_pos=1 <= target_len=1 → NoOp.
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(50, 0)), 0.99);

        let mut caches = make_caches(2, 1);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Warning,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(matches!(result, ActionResult::NoOp));
        assert_eq!(ctx.caches[0].current_pos, 1); // unchanged
    }

    #[test]
    fn test_noop_on_empty_caches() {
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(10, 0)), 0.5);

        let mut caches: Vec<KVCache> = vec![];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(matches!(result, ActionResult::NoOp));
    }

    #[test]
    fn test_name_delegates_to_policy() {
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(10, 0)), 0.5);
        assert_eq!(handler.name(), "sliding");

        let handler = EvictionHandler::new(Box::new(H2OPolicy::new(0.5, 0)), 0.5);
        assert_eq!(handler.name(), "h2o");

        let handler = EvictionHandler::new(Box::new(NoEvictionPolicy::new()), 0.5);
        assert_eq!(handler.name(), "none");
    }

    #[test]
    fn test_skip_when_tokens_to_remove_below_threshold() {
        // current_pos=100, target_ratio=0.95 → target_len=95, tokens_to_remove=5 < MIN_EVICT_TOKENS=64
        // Expected: NoOp, current_pos unchanged.
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(200, 0)), 0.95);

        let mut caches = make_caches(2, 100);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(
            matches!(result, ActionResult::NoOp),
            "Expected NoOp when tokens_to_remove({}) < MIN_EVICT_TOKENS({}), got {:?}",
            100usize.saturating_sub(95),
            MIN_EVICT_TOKENS,
            result,
        );
        // current_pos must be unchanged
        assert_eq!(ctx.caches[0].current_pos, 100);
    }

    #[test]
    fn test_skip_does_not_fire_when_above_threshold() {
        // current_pos=100, target_ratio=0.3 → target_len=30, tokens_to_remove=70 >= MIN_EVICT_TOKENS=64
        // Expected: Evicted, not NoOp.
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(200, 0)), 0.3);

        let mut caches = make_caches(2, 100);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(
            matches!(result, ActionResult::Evicted { .. }),
            "Expected Evicted when tokens_to_remove >= MIN_EVICT_TOKENS, got {:?}",
            result,
        );
        assert!(ctx.caches[0].current_pos < 100);
    }

    #[test]
    fn test_target_ratio_clamping() {
        // ratio=0.0 should clamp to 0.1.
        // pos=100, clamped ratio=0.1 → target_len=10, tokens_to_remove=90 >= MIN_EVICT_TOKENS(64).
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(10, 0)), 0.0);

        let mut caches = make_caches(1, 100);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted { new_pos, .. } => {
                assert!(new_pos > 0, "Should keep at least 1 token");
            }
            _ => panic!("Expected Evicted"),
        }
    }

    #[test]
    fn test_h2o_fallback_without_scores() {
        // H2O without importance scores → fallback to sliding-window-like behavior.
        // pos=100, target_ratio=0.3 → tokens_to_remove=70 >= MIN_EVICT_TOKENS(64).
        let handler = EvictionHandler::new(Box::new(H2OPolicy::new(0.5, 0)), 0.3);

        let mut caches = make_caches(2, 100);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None, // no scores
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted {
                tokens_removed,
                new_pos,
            } => {
                assert!(tokens_removed > 0);
                assert!(new_pos < 100);
            }
            _ => panic!("Expected Evicted"),
        }
    }
}
