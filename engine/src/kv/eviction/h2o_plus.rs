use super::EvictionPolicy;
use crate::kv::kv_cache::KVCache;
use anyhow::Result;

/// H2O+ (GQA-Aware Per-Head Eviction) — extends H2O with per-KV-head token selection.
///
/// Like H2O, uses the 3-partition model: [Protected Prefix] [Heavy Hitters] [Recent Window].
/// Unlike H2O, each KV head independently selects its own heavy hitters based on
/// GQA-grouped attention scores, enabling different token retention per head.
///
/// All heads keep the same NUMBER of tokens (target_len), but select DIFFERENT
/// tokens as heavy hitters. This preserves the single `current_pos` invariant.
pub struct H2OPlusPolicy {
    /// Fraction of available budget allocated to heavy hitters (0.0 to 1.0).
    keep_ratio: f32,
    /// Number of prefix tokens to always protect (attention sinks).
    protected_prefix: usize,
}

impl H2OPlusPolicy {
    pub fn new(keep_ratio: f32, protected_prefix: usize) -> Self {
        let protected_prefix = protected_prefix.max(4);
        Self {
            keep_ratio: keep_ratio.clamp(0.0, 1.0),
            protected_prefix,
        }
    }
}

impl EvictionPolicy for H2OPlusPolicy {
    fn should_evict(&self, _cache: &KVCache, _mem_available: usize) -> bool {
        false
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        // Fallback: without scores, use sliding window (same as H2O)
        let current = cache.current_pos;
        let keep = target_len.max(self.protected_prefix + 2);

        if current <= keep {
            return Ok(());
        }

        let available = keep.saturating_sub(self.protected_prefix);
        let actual_recent = available.min(current - self.protected_prefix);
        let prune_count = current - self.protected_prefix - actual_recent;

        if prune_count == 0 {
            return Ok(());
        }

        cache.shift_positions(
            self.protected_prefix + prune_count,
            self.protected_prefix,
            actual_recent,
        )?;
        cache.current_pos = self.protected_prefix + actual_recent;
        Ok(())
    }

    fn evict_with_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: &[f32],
    ) -> Result<()> {
        // Flat-score fallback: same algorithm as H2O (all heads same eviction)
        let current = cache.current_pos;
        let keep = target_len.max(self.protected_prefix + 2);

        if current <= keep {
            return Ok(());
        }

        let available = keep.saturating_sub(self.protected_prefix);
        let hh_budget = (available as f32 * self.keep_ratio) as usize;
        let recent_budget = available - hh_budget;
        let recent_start = current
            .saturating_sub(recent_budget)
            .max(self.protected_prefix);

        let evictable_start = self.protected_prefix;
        let mut token_scores: Vec<(usize, f32)> = (evictable_start..recent_start)
            .map(|pos| (pos, importance.get(pos).copied().unwrap_or(0.0)))
            .collect();

        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut hh_positions: Vec<usize> = token_scores
            .iter()
            .take(hh_budget)
            .map(|(pos, _)| *pos)
            .collect();
        hh_positions.sort();

        let recent_len = current - recent_start;
        let mut keep_all: Vec<usize> = Vec::with_capacity(hh_positions.len() + recent_len);
        keep_all.extend_from_slice(&hh_positions);
        keep_all.extend(recent_start..current);

        cache.compact_keep_positions(&keep_all, self.protected_prefix)?;

        cache.current_pos = self.protected_prefix + keep_all.len();
        Ok(())
    }

    fn evict_with_head_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        _flat_importance: &[f32],
        head_importance: &[f32],
        n_kv_heads: usize,
    ) -> Result<()> {
        let current = cache.current_pos;
        let keep = target_len.max(self.protected_prefix + 2);

        if current <= keep {
            return Ok(());
        }

        let available = keep.saturating_sub(self.protected_prefix);
        let hh_budget = (available as f32 * self.keep_ratio) as usize;
        let recent_budget = available - hh_budget;
        let recent_start = current
            .saturating_sub(recent_budget)
            .max(self.protected_prefix);
        let actual_recent = current - recent_start;
        let max_seq = head_importance.len() / n_kv_heads;

        log::debug!(
            "H2O+: per-head eviction, keeping {}/{} tokens (prefix={}, hh={}, recent={}, kv_heads={})",
            keep,
            current,
            self.protected_prefix,
            hh_budget,
            actual_recent,
            n_kv_heads,
        );

        // Each KV head independently selects its heavy hitters
        for kv_h in 0..n_kv_heads {
            let mut head_tokens: Vec<(usize, f32)> = (self.protected_prefix..recent_start)
                .map(|pos| (pos, head_importance[kv_h * max_seq + pos]))
                .collect();

            head_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut hh_positions: Vec<usize> = head_tokens
                .iter()
                .take(hh_budget)
                .map(|(pos, _)| *pos)
                .collect();
            hh_positions.sort();

            // Compact this head only — batch consecutive positions together
            let mut keep_all: Vec<usize> = Vec::with_capacity(hh_positions.len() + actual_recent);
            keep_all.extend_from_slice(&hh_positions);
            keep_all.extend(recent_start..current);
            cache.compact_keep_positions_for_head(kv_h, &keep_all, self.protected_prefix)?;
        }

        cache.current_pos = self.protected_prefix + hh_budget + actual_recent;

        log::debug!("H2O+: compacted cache to {} tokens", cache.current_pos);
        Ok(())
    }

    fn name(&self) -> &str {
        "h2o_plus"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv_cache_ops::KVLayout;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache(pos: usize, n_kv_heads: usize) -> KVCache {
        let max_seq = 100;
        let head_dim = 4;
        let backend = Arc::new(CpuBackend::new());
        // Shape uses SeqMajor convention: [batch, seq, kv_heads, head_dim]
        // new() reads shape[2]=kv_heads, shape[3]=head_dim
        let buf_size = n_kv_heads * max_seq * head_dim * std::mem::size_of::<f32>();
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, n_kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, n_kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let mut cache = KVCache::new(k, v, max_seq).with_layout(KVLayout::HeadMajor);
        cache.current_pos = pos;
        cache
    }

    fn write_head_marker(cache: &mut KVCache, head: usize, pos: usize, val: f32) {
        let off = cache.offset(pos, head);
        unsafe {
            let k_f32 = cache.k_buffer.as_mut_ptr() as *mut f32;
            let v_f32 = cache.v_buffer.as_mut_ptr() as *mut f32;
            *k_f32.add(off) = val;
            *v_f32.add(off) = val;
        }
    }

    fn read_head_marker(cache: &KVCache, head: usize, pos: usize) -> (f32, f32) {
        let off = cache.offset(pos, head);
        let k_val = cache.k_buffer.as_slice::<f32>()[off];
        let v_val = cache.v_buffer.as_slice::<f32>()[off];
        (k_val, v_val)
    }

    #[test]
    fn test_should_evict_always_false() {
        let policy = H2OPlusPolicy::new(0.5, 0);
        assert!(!policy.should_evict(&make_cache(20, 2), 0));
    }

    #[test]
    fn test_name() {
        assert_eq!(H2OPlusPolicy::new(0.5, 0).name(), "h2o_plus");
    }

    #[test]
    fn test_evict_fallback() {
        let policy = H2OPlusPolicy::new(0.5, 0);
        let mut cache = make_cache(20, 2);
        policy.evict(&mut cache, 10).unwrap();
        assert!(cache.current_pos <= 10);
    }

    #[test]
    fn test_noop_when_below_target() {
        let policy = H2OPlusPolicy::new(0.5, 0);
        let mut cache = make_cache(10, 2);
        let head_importance = vec![1.0f32; 2 * 100];
        let flat = vec![1.0f32; 100];

        policy
            .evict_with_head_scores(&mut cache, 15, &flat, &head_importance, 2)
            .unwrap();
        assert_eq!(cache.current_pos, 10);
    }

    #[test]
    fn test_current_pos_uniform_after_eviction() {
        let policy = H2OPlusPolicy::new(0.5, 0);
        let n_kv_heads = 2;
        let mut cache = make_cache(20, n_kv_heads);

        let mut head_importance = vec![0.01f32; n_kv_heads * 100];
        // Different high-score tokens per head
        head_importance[5] = 10.0; // KV0: tok5 high
        head_importance[100 + 8] = 10.0; // KV1: tok8 high

        let flat = vec![1.0f32; 100];

        policy
            .evict_with_head_scores(&mut cache, 10, &flat, &head_importance, n_kv_heads)
            .unwrap();
        assert_eq!(cache.current_pos, 10);
    }

    #[test]
    fn test_per_head_preserves_prefix() {
        let policy = H2OPlusPolicy::new(0.5, 5);
        let n_kv_heads = 2;
        let mut cache = make_cache(30, n_kv_heads);

        // Write markers for prefix
        for h in 0..n_kv_heads {
            for i in 0..30 {
                write_head_marker(&mut cache, h, i, (i + 1) as f32 + h as f32 * 100.0);
            }
        }

        let mut head_importance = vec![0.01f32; n_kv_heads * 100];
        for i in 5..30 {
            head_importance[i] = (30 - i) as f32;
            head_importance[100 + i] = (30 - i) as f32;
        }
        let flat = vec![1.0f32; 100];

        policy
            .evict_with_head_scores(&mut cache, 15, &flat, &head_importance, n_kv_heads)
            .unwrap();

        assert_eq!(cache.current_pos, 15);

        // Prefix intact for both heads
        for h in 0..n_kv_heads {
            for i in 0..5 {
                let (k, v) = read_head_marker(&cache, h, i);
                let expected = (i + 1) as f32 + h as f32 * 100.0;
                assert_eq!(k, expected, "head {} prefix pos {} marker wrong", h, i);
                assert_eq!(v, expected);
            }
        }
    }

    #[test]
    fn test_per_head_preserves_recent() {
        let policy = H2OPlusPolicy::new(0.5, 0); // prefix=4(clamped)
        let n_kv_heads = 2;
        let mut cache = make_cache(20, n_kv_heads);

        for h in 0..n_kv_heads {
            for i in 0..20 {
                write_head_marker(&mut cache, h, i, (i + 1) as f32 + h as f32 * 100.0);
            }
        }

        let mut head_importance = vec![0.01f32; n_kv_heads * 100];
        head_importance[5] = 10.0;
        head_importance[100 + 8] = 10.0;
        let flat = vec![1.0f32; 100];

        // target=10, keep=10, available=6, hh=3, recent=3
        // recent_start = max(4, 20-3) = 17
        // Recent tokens: 17,18,19
        policy
            .evict_with_head_scores(&mut cache, 10, &flat, &head_importance, n_kv_heads)
            .unwrap();

        assert_eq!(cache.current_pos, 10);

        // Last 3 positions should be recent tokens (originally pos 17,18,19) for BOTH heads
        for h in 0..n_kv_heads {
            for i in 0..3 {
                let (k, _) = read_head_marker(&cache, h, cache.current_pos - 3 + i);
                let expected = (17 + i + 1) as f32 + h as f32 * 100.0;
                assert_eq!(k, expected, "head {} recent pos {} marker wrong", h, i);
            }
        }
    }

    #[test]
    fn test_per_head_different_hh() {
        let policy = H2OPlusPolicy::new(0.5, 0); // prefix=4(clamped)
        let n_kv_heads = 2;
        let mut cache = make_cache(20, n_kv_heads);

        for h in 0..n_kv_heads {
            for i in 0..20 {
                write_head_marker(&mut cache, h, i, (i * 100) as f32 + h as f32);
            }
        }

        let mut head_importance = vec![0.01f32; n_kv_heads * 100];
        // KV head 0: prefers tokens 5, 6, 7 (high scores)
        head_importance[5] = 10.0;
        head_importance[6] = 9.0;
        head_importance[7] = 8.0;
        // KV head 1: prefers tokens 10, 11, 12 (high scores)
        head_importance[100 + 10] = 10.0;
        head_importance[100 + 11] = 9.0;
        head_importance[100 + 12] = 8.0;

        let flat = vec![1.0f32; 100];

        // target=10, keep=10, available=6, hh=3, recent=3
        // recent_start = max(4, 20-3) = 17
        policy
            .evict_with_head_scores(&mut cache, 10, &flat, &head_importance, n_kv_heads)
            .unwrap();

        assert_eq!(cache.current_pos, 10);

        // Head 0: [prefix 0-3] [HH: 5,6,7] [recent: 17,18,19]
        let (k, _) = read_head_marker(&cache, 0, 4);
        assert_eq!(k, 500.0); // token 5's marker
        let (k, _) = read_head_marker(&cache, 0, 5);
        assert_eq!(k, 600.0); // token 6
        let (k, _) = read_head_marker(&cache, 0, 6);
        assert_eq!(k, 700.0); // token 7

        // Head 1: [prefix 0-3] [HH: 10,11,12] [recent: 17,18,19]
        let (k, _) = read_head_marker(&cache, 1, 4);
        assert_eq!(k, 1001.0); // token 10's marker (10*100 + 1.0 for head 1)
        let (k, _) = read_head_marker(&cache, 1, 5);
        assert_eq!(k, 1101.0); // token 11
        let (k, _) = read_head_marker(&cache, 1, 6);
        assert_eq!(k, 1201.0); // token 12
    }

    #[test]
    fn test_per_head_eviction_basic() {
        let policy = H2OPlusPolicy::new(0.5, 0); // prefix=4(clamped)
        let n_kv_heads = 2;
        let mut cache = make_cache(20, n_kv_heads);

        let mut head_importance = vec![0.5f32; n_kv_heads * 100];
        for i in 4..20 {
            head_importance[i] = (20 - i) as f32;
            head_importance[100 + i] = i as f32;
        }
        let flat = vec![1.0f32; 100];

        policy
            .evict_with_head_scores(&mut cache, 10, &flat, &head_importance, n_kv_heads)
            .unwrap();

        assert_eq!(cache.current_pos, 10);
    }

    #[test]
    fn test_flat_scores_fallback() {
        let policy = H2OPlusPolicy::new(0.5, 0);
        let mut cache = make_cache(20, 2);

        let mut importance = vec![0.01f32; 100];
        importance[5] = 10.0;
        importance[8] = 9.0;
        importance[12] = 8.0;

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();
        assert_eq!(cache.current_pos, 10);
    }
}
