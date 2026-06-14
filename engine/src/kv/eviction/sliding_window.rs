use super::EvictionPolicy;
use crate::kv::kv_cache::KVCache;
use anyhow::Result;

/// Sliding window eviction policy.
///
/// Keeps only the most recent `window_size` tokens in the cache.
/// Optionally protects a prefix of `protected_prefix` tokens (e.g., system prompt)
/// from being evicted.
///
/// ## Example
/// ```text
/// window_size=4, protected_prefix=2
///
/// Before (current_pos=8):
/// [P0][P1][T2][T3][T4][T5][T6][T7]
///  protected   ───── evict ─────  keep
///
/// After (current_pos=6):
/// [P0][P1][T4][T5][T6][T7][_][_]
/// ```
pub struct SlidingWindowPolicy {
    window_size: usize,
    protected_prefix: usize,
}

impl SlidingWindowPolicy {
    pub fn new(window_size: usize, protected_prefix: usize) -> Self {
        // Enforce a minimum protected prefix of 4 to act as an Attention Sink
        let protected_prefix = protected_prefix.max(4);
        Self {
            window_size,
            protected_prefix,
        }
    }
}

impl EvictionPolicy for SlidingWindowPolicy {
    fn should_evict(&self, cache: &KVCache, _mem_available: usize) -> bool {
        cache.current_pos > self.window_size + self.protected_prefix
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        let current = cache.current_pos;
        // Determine how many tokens to keep
        let max_keep = self.window_size + self.protected_prefix;
        let min_keep = (self.protected_prefix + 16).min(max_keep);
        let keep = target_len.clamp(min_keep, max_keep);

        if current <= keep {
            return Ok(());
        }

        // Number of tokens to remove from the non-protected region
        let removable_start = self.protected_prefix;
        let removable_count = current - removable_start;
        let tokens_to_keep_after_prefix = keep.saturating_sub(self.protected_prefix);

        if tokens_to_keep_after_prefix >= removable_count {
            return Ok(()); // Nothing to remove
        }

        let prune_count = removable_count - tokens_to_keep_after_prefix;

        // We need to prune from position `protected_prefix`, not from position 0.
        // But KVCache::prune_prefix removes from position 0.
        // If protected_prefix == 0, we just prune from the front.
        // If protected_prefix > 0, we need to shift data in a more complex way.
        //
        // For now, handle the common case: prune_prefix removes the oldest tokens.
        // Protected prefix: the first `protected_prefix` tokens are preserved.
        // We effectively remove tokens from index `protected_prefix` to
        // `protected_prefix + prune_count - 1`, keeping both the prefix and the tail.

        if self.protected_prefix == 0 {
            cache.prune_prefix(prune_count)?;
        } else {
            let remaining = current - self.protected_prefix - prune_count;
            cache.shift_positions(
                self.protected_prefix + prune_count,
                self.protected_prefix,
                remaining,
            )?;
            cache.current_pos -= prune_count;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "sliding"
    }

    /// (3c-evict) keep-list. `evict()` 의 보존 영역 `[0..prefix) ∪ [prefix+prune_count..current)`
    /// 을 prefix-포함 ascending 으로 산출. no-op 분기(`evict()` early-return)는 전체 보존 `[0..current)`.
    /// `protected_prefix >= 4`(생성자 clamp)라 `evict()` 의 prefix==0 `prune_prefix` 분기는 dead —
    /// 본 keep-list 는 prefix>0 의 `shift_positions(prefix+prune_count, prefix, remaining)` 와 등가.
    fn plan_keep(
        &self,
        current_pos: usize,
        target_len: usize,
        _importance: Option<&[f32]>,
    ) -> Option<(Vec<usize>, Vec<crate::format::Merge>)> {
        let current = current_pos;
        let max_keep = self.window_size + self.protected_prefix;
        let min_keep = (self.protected_prefix + 16).min(max_keep);
        let keep = target_len.clamp(min_keep, max_keep);

        if current <= keep {
            return Some(((0..current).collect(), Vec::new()));
        }

        let removable_count = current - self.protected_prefix;
        let tokens_to_keep_after_prefix = keep.saturating_sub(self.protected_prefix);

        if tokens_to_keep_after_prefix >= removable_count {
            return Some(((0..current).collect(), Vec::new()));
        }

        let prune_count = removable_count - tokens_to_keep_after_prefix;

        let mut keep_list: Vec<usize> = (0..self.protected_prefix).collect();
        keep_list.extend((self.protected_prefix + prune_count)..current);
        Some((keep_list, Vec::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache_with_data(num_tokens: usize) -> KVCache {
        let max_seq = 100;
        let heads = 1;
        let dim = 4;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * heads * dim * 4;

        let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

        // Fill with recognizable data
        unsafe {
            let k_ptr = k_buf.as_mut_ptr() as *mut f32;
            let v_ptr = v_buf.as_mut_ptr() as *mut f32;
            for i in 0..num_tokens * dim {
                *k_ptr.add(i) = (i / dim + 1) as f32; // pos+1 pattern
                *v_ptr.add(i) = ((i / dim + 1) * 10) as f32;
            }
        }

        let k = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            v_buf,
            backend.clone(),
        );
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = num_tokens;
        cache
    }

    #[test]
    fn test_should_evict() {
        let policy = SlidingWindowPolicy::new(10, 0); // prefix becomes 4
        let mut cache = make_cache_with_data(8);
        assert!(!policy.should_evict(&cache, 0)); // 8 <= 14

        cache.current_pos = 15;
        assert!(policy.should_evict(&cache, 0)); // 15 > 14
    }

    #[test]
    fn test_should_evict_with_prefix() {
        let policy = SlidingWindowPolicy::new(10, 5);
        let mut cache = make_cache_with_data(14);
        assert!(!policy.should_evict(&cache, 0)); // 14 <= 10+5

        cache.current_pos = 16;
        assert!(policy.should_evict(&cache, 0)); // 16 > 15
    }

    #[test]
    fn test_evict_no_prefix() {
        // window=10, prefix defaults to 4
        // target=15, min_keep = 4+16=20, max_keep = 14
        // Wait, max_keep < min_keep -> min_keep is clamped to max_keep=14.
        // So keep=14.
        let policy = SlidingWindowPolicy::new(10, 0);
        let mut cache = make_cache_with_data(20);
        // prune_count = 20 - 14 = 6.
        // We prune from index 4 to 9. (6 tokens)

        policy.evict(&mut cache, 5).unwrap();
        assert_eq!(cache.current_pos, 14);

        // Verify data: position 0..3 should be protected
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 1.0);
        assert_eq!(k_data[12], 4.0);

        // position 4 should now be old position 10 (value=11.0)
        assert_eq!(k_data[16], 11.0);
    }

    #[test]
    fn test_evict_with_protected_prefix() {
        let policy = SlidingWindowPolicy::new(4, 4); // window=4, prefix=4 => max=8
        let mut cache = make_cache_with_data(12);
        // min_keep = min(4+16=20, 8) = 8.
        // keep = 8. Removing 4 tokens (indices 4..7).

        policy.evict(&mut cache, 6).unwrap();
        assert_eq!(cache.current_pos, 8);

        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 1.0); // Original pos 0
        assert_eq!(k_data[12], 4.0); // Original pos 3

        // Position 4 should now be old position 8 (value=9.0)
        assert_eq!(k_data[16], 9.0);
        assert_eq!(k_data[28], 12.0);
    }

    #[test]
    fn test_evict_no_action_needed() {
        let policy = SlidingWindowPolicy::new(20, 0);
        let mut cache = make_cache_with_data(10);

        policy.evict(&mut cache, 20).unwrap();
        assert_eq!(cache.current_pos, 10); // No change
    }

    #[test]
    fn test_name() {
        let policy = SlidingWindowPolicy::new(10, 0);
        assert_eq!(policy.name(), "sliding");
    }

    #[test]
    fn test_minimum_protected_prefix_enforced() {
        // Even when constructed with prefix=0, 1, 2, 3, the internal value is .max(4) = 4
        for input_prefix in 0..=3 {
            let policy = SlidingWindowPolicy::new(10, input_prefix);
            // Threshold for eviction is window_size + protected_prefix = 10 + 4 = 14
            let mut cache = make_cache_with_data(14);
            assert!(
                !policy.should_evict(&cache, 0),
                "prefix={}: 14 should NOT trigger eviction (threshold=14)",
                input_prefix
            );

            cache.current_pos = 15;
            assert!(
                policy.should_evict(&cache, 0),
                "prefix={}: 15 should trigger eviction (threshold=14)",
                input_prefix
            );
        }

        // prefix=5 stays as 5 (already >= 4)
        let policy = SlidingWindowPolicy::new(10, 5);
        let mut cache = make_cache_with_data(15);
        assert!(!policy.should_evict(&cache, 0)); // 15 <= 10+5
        cache.current_pos = 16;
        assert!(policy.should_evict(&cache, 0)); // 16 > 15
    }
}
