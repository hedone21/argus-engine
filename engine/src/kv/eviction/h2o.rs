use super::EvictionPolicy;
use crate::kv::kv_cache::KVCache;
use anyhow::Result;

/// H2O (Heavy-Hitter Oracle) eviction policy — attention score-based KV cache eviction.
///
/// Implements the 3-partition model from the H2O paper:
///   [Protected Prefix] [Heavy Hitters (score-ranked)] [Recent Window]
///
/// - **Protected Prefix**: First N tokens (attention sinks / system prompt) are never evicted.
/// - **Heavy Hitters**: Tokens with highest cumulative attention scores.
/// - **Recent Window**: Most recent M tokens, always protected.
///
/// Budget allocation (following the paper): after reserving prefix, the remaining
/// `keep` slots are split between HH and Recent by `keep_ratio`:
///   - `hh_budget = available * keep_ratio`
///   - `recent_budget = available - hh_budget`
///
/// With `keep_ratio=0.5` (default), this produces the paper's recommended 50:50 split.
///
/// Reference: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
/// Large Language Models" (Zhang et al., 2023)
pub struct H2OPolicy {
    /// Fraction of available budget allocated to heavy hitters (0.0 to 1.0).
    /// Default 0.5 = paper's 50:50 HH:Recent split.
    keep_ratio: f32,
    /// Number of prefix tokens to always protect (attention sinks)
    protected_prefix: usize,
}

impl H2OPolicy {
    pub fn new(keep_ratio: f32, protected_prefix: usize) -> Self {
        let protected_prefix = protected_prefix.max(4);
        Self {
            keep_ratio: keep_ratio.clamp(0.0, 1.0),
            protected_prefix,
        }
    }
}

impl EvictionPolicy for H2OPolicy {
    fn should_evict(&self, _cache: &KVCache, _mem_available: usize) -> bool {
        // H2O is signal-driven: eviction is triggered exclusively by external
        // resilience signals, never by automatic cache/memory checks.
        // Score accumulation happens every token, but eviction decisions
        // come from the resilience manager.
        false
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        let current = cache.current_pos;
        let keep = target_len.max(self.protected_prefix + 2);

        if current <= keep {
            return Ok(());
        }

        // Fallback: without attention scores, keep prefix + most recent tokens
        let available = keep.saturating_sub(self.protected_prefix);
        let recent_budget = available; // All budget goes to recent in fallback mode
        let actual_recent = recent_budget.min(current - self.protected_prefix);
        let prune_count = current - self.protected_prefix - actual_recent;

        if prune_count == 0 {
            return Ok(());
        }

        log::debug!(
            "H2O (fallback): sliding window eviction, removing {} tokens (keep={})",
            prune_count,
            keep,
        );

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
        let current = cache.current_pos;
        let keep = target_len.max(self.protected_prefix + 2);

        if current <= keep {
            return Ok(());
        }

        // Budget allocation (paper's approach):
        // available = keep - prefix
        // hh_budget = available * keep_ratio  (default 0.5 = 50:50)
        // recent_budget = available - hh_budget
        let available = keep.saturating_sub(self.protected_prefix);
        let hh_budget = (available as f32 * self.keep_ratio) as usize;
        let recent_budget = available - hh_budget;
        let actual_recent = recent_budget.min(current - self.protected_prefix);

        // Recent window boundary
        let recent_start = current
            .saturating_sub(actual_recent)
            .max(self.protected_prefix);
        let actual_recent = current - recent_start;

        // 1. Rank evictable tokens (prefix..recent_start) by importance
        let evictable_start = self.protected_prefix;

        log::debug!(
            "H2O: score-based eviction, keeping {}/{} tokens (prefix={}, hh={}, recent={})",
            keep,
            current,
            self.protected_prefix,
            hh_budget,
            actual_recent,
        );
        log::debug!(
            "H2O: budget: available={}, keep_ratio={:.2}, evictable_range=[{}..{}), evictable_count={}",
            available,
            self.keep_ratio,
            evictable_start,
            recent_start,
            recent_start - evictable_start,
        );
        let mut token_scores: Vec<(usize, f32)> = (evictable_start..recent_start)
            .map(|pos| (pos, importance.get(pos).copied().unwrap_or(0.0)))
            .collect();

        // 2. Sort by importance descending (highest first = keep)
        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Select top hh_budget heavy hitters, then sort by position to preserve order
        let mut hh_positions: Vec<usize> = token_scores
            .iter()
            .take(hh_budget)
            .map(|(pos, _)| *pos)
            .collect();
        hh_positions.sort();

        if log::log_enabled!(log::Level::Debug) {
            let selected_hh: Vec<_> = token_scores.iter().take(hh_budget).collect();
            let evicted: Vec<_> = token_scores.iter().skip(hh_budget).collect();
            log::debug!(
                "H2O: HH selected (pos, score): {:?}",
                &selected_hh[..selected_hh.len().min(10)]
            );
            if !evicted.is_empty() {
                log::debug!(
                    "H2O: evicted (pos, score): {:?}",
                    &evicted[..evicted.len().min(10)]
                );
            }
        }

        // 4. Build final keep list: [heavy hitters] ++ [recent positions]
        let recent_len = current - recent_start;
        let mut keep_all: Vec<usize> = Vec::with_capacity(hh_positions.len() + recent_len);
        keep_all.extend_from_slice(&hh_positions);
        keep_all.extend(recent_start..current);

        // 5. Compact the KV cache: [prefix..., heavy hitters in order..., recent in order...]
        cache.compact_keep_positions(&keep_all, self.protected_prefix)?;

        cache.current_pos = self.protected_prefix + keep_all.len();

        log::debug!("H2O: compacted cache to {} tokens", cache.current_pos);

        Ok(())
    }

    fn name(&self) -> &str {
        "h2o"
    }

    /// (3c-evict) keep-list. `evict_with_scores`(importance Some) 또는 `evict`(None)의 보존 영역을
    /// **prefix-포함 ascending** 으로 산출. score 경로는 `evict_with_scores` 가 이미 호출하는
    /// `compact_keep_positions(keep_all, prefix)`(line ~169)의 `keep_all = [hh ++ recent]` 와 동일
    /// 토큰을 prefix 와 합쳐 `[0..prefix) ++ hh_positions ++ [recent_start..current)` 로 반환한다 —
    /// `compact(keep, write_start=0)` 의 prefix batch 가 src==dst no-op 이므로 등가. merges 없음.
    fn plan_keep(
        &self,
        current_pos: usize,
        target_len: usize,
        importance: Option<&[f32]>,
    ) -> Option<(Vec<usize>, Vec<crate::format::Merge>)> {
        let current = current_pos;
        let keep = target_len.max(self.protected_prefix + 2);
        let prefix = self.protected_prefix;

        if current <= keep {
            return Some(((0..current).collect(), Vec::new()));
        }

        match importance {
            // score-free fallback (`evict`): prefix + most-recent.
            None => {
                let available = keep.saturating_sub(prefix);
                let recent_budget = available;
                let actual_recent = recent_budget.min(current - prefix);
                let prune_count = current - prefix - actual_recent;
                if prune_count == 0 {
                    return Some(((0..current).collect(), Vec::new()));
                }
                let mut keep_list: Vec<usize> = (0..prefix).collect();
                keep_list.extend((prefix + prune_count)..current);
                Some((keep_list, Vec::new()))
            }
            // score-based (`evict_with_scores`): prefix + heavy hitters + recent window.
            Some(imp) => {
                let available = keep.saturating_sub(prefix);
                let hh_budget = (available as f32 * self.keep_ratio) as usize;
                let recent_budget = available - hh_budget;
                let actual_recent = recent_budget.min(current - prefix);
                let recent_start = current.saturating_sub(actual_recent).max(prefix);
                let evictable_start = prefix;

                // 1. (pos, score) over evictable range. 2. stable sort desc (== evict_with_scores).
                let mut token_scores: Vec<(usize, f32)> = (evictable_start..recent_start)
                    .map(|pos| (pos, imp.get(pos).copied().unwrap_or(0.0)))
                    .collect();
                token_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // 3. top-hh_budget by score, then re-sort by position (== evict_with_scores).
                let mut hh_positions: Vec<usize> = token_scores
                    .iter()
                    .take(hh_budget)
                    .map(|(pos, _)| *pos)
                    .collect();
                hh_positions.sort();

                // 4. prefix + HH(position order) + recent(position order), all ascending.
                let mut keep_list: Vec<usize> = (0..prefix).collect();
                keep_list.extend_from_slice(&hh_positions);
                keep_list.extend(recent_start..current);
                Some((keep_list, Vec::new()))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::useless_vec)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache(pos: usize) -> KVCache {
        let max_seq = 100;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * 4 * 4; // max_seq * batch * heads * dim * sizeof(f32)
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
    }

    fn write_marker(cache: &mut KVCache, pos: usize, val: f32) {
        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let bytes_per_pos = heads * dim * std::mem::size_of::<f32>();
        let k_ptr = cache.k_buffer.as_mut_ptr();
        let v_ptr = cache.v_buffer.as_mut_ptr();
        unsafe {
            let k_f32 = k_ptr.add(pos * bytes_per_pos) as *mut f32;
            let v_f32 = v_ptr.add(pos * bytes_per_pos) as *mut f32;
            *k_f32 = val;
            *v_f32 = val;
        }
    }

    fn read_marker(cache: &KVCache, pos: usize) -> (f32, f32) {
        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let bytes_per_pos = heads * dim * std::mem::size_of::<f32>();
        let k_ptr = cache.k_buffer.as_mut_ptr();
        let v_ptr = cache.v_buffer.as_mut_ptr();
        unsafe {
            let k_val = *(k_ptr.add(pos * bytes_per_pos) as *const f32);
            let v_val = *(v_ptr.add(pos * bytes_per_pos) as *const f32);
            (k_val, v_val)
        }
    }

    // ── should_evict tests ──

    #[test]
    fn test_should_evict_always_false() {
        let policy = H2OPolicy::new(0.5, 0);
        assert!(!policy.should_evict(&make_cache(20), 0));
        assert!(!policy.should_evict(&make_cache(1000), 0));
        assert!(!policy.should_evict(&make_cache(1), 0));
    }

    // ── evict (fallback) tests ──

    #[test]
    fn test_evict_fallback_keeps_recent() {
        let _ = env_logger::try_init();
        // keep_ratio=0.5, prefix=4(clamped)
        let policy = H2OPolicy::new(0.5, 0);
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        // target_len=10 → keep=10, available=10-4=6, all recent
        // prune = 20-4-6 = 10
        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 10);

        // Prefix intact
        for i in 0..4 {
            let (k, _) = read_marker(&cache, i);
            assert_eq!(k, (i + 1) as f32);
        }
        // Most recent tokens at the end
        let (k, _) = read_marker(&cache, 9);
        assert_eq!(k, 20.0);
    }

    #[test]
    fn test_evict_below_threshold_noop() {
        let policy = H2OPolicy::new(0.5, 0);
        let mut cache = make_cache(10);
        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 10);
    }

    // ── evict_with_scores tests ──

    #[test]
    fn test_evict_with_scores_budget_split() {
        // Verify that prefix + hh_budget + recent_budget == keep for various keep_ratio values.
        // Each case runs end-to-end eviction and checks the resulting current_pos.
        struct Case {
            label: &'static str,
            keep_ratio: f32,
            current: usize,
            target: usize,
            // prefix is always 4 (clamped from 0)
            expected_pos: usize,
        }

        let cases = [
            Case {
                label: "50:50 split (paper default)",
                keep_ratio: 0.5,
                current: 20,
                target: 10,
                // available=6, hh=3, recent=3 → prefix(4)+3+3=10
                expected_pos: 10,
            },
            Case {
                label: "80:20 split (high HH ratio)",
                keep_ratio: 0.8,
                current: 20,
                target: 14,
                // available=10, hh=8, recent=2 → prefix(4)+8+2=14
                expected_pos: 14,
            },
        ];

        for c in &cases {
            let policy = H2OPolicy::new(c.keep_ratio, 0); // prefix=4 (clamped)
            let mut cache = make_cache(c.current);
            for i in 0..c.current {
                write_marker(&mut cache, i, (i + 1) as f32);
            }
            let mut importance = vec![0.0f32; 100];
            for i in 4..c.current {
                importance[i] = (c.current - i) as f32;
            }
            policy
                .evict_with_scores(&mut cache, c.target, &importance)
                .unwrap();
            assert_eq!(
                cache.current_pos, c.expected_pos,
                "case='{}': expected current_pos={}",
                c.label, c.expected_pos
            );
        }
    }

    #[test]
    fn test_evict_preserves_prefix() {
        let policy = H2OPolicy::new(0.5, 5); // prefix=5
        let mut cache = make_cache(30);

        for i in 0..30 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        let mut importance = vec![0.0f32; 100];
        for (idx, imp) in importance[5..30].iter_mut().enumerate() {
            *imp = (25 - idx) as f32; // 30 - (i) where i starts at 5 → (30-5)-idx = 25-idx
        }

        // target_len=15, keep=15, available=15-5=10
        // hh_budget=5, recent=5
        policy
            .evict_with_scores(&mut cache, 15, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 15);

        // Prefix intact
        for i in 0..5 {
            let (k, v) = read_marker(&cache, i);
            assert_eq!(k, (i + 1) as f32);
            assert_eq!(v, (i + 1) as f32);
        }
    }

    #[test]
    fn test_no_eviction_when_below_target() {
        let policy = H2OPolicy::new(0.5, 0);
        let mut cache = make_cache(10);
        let importance = vec![1.0f32; 100];

        policy
            .evict_with_scores(&mut cache, 15, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10); // no eviction
    }

    #[test]
    fn test_order_preservation() {
        // Verify final layout is [prefix][hh in position order][recent in position order]
        let policy = H2OPolicy::new(0.5, 0); // prefix=4(clamped)
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i * 100) as f32);
        }

        // Give distinctive scores to specific tokens
        let mut importance = vec![0.0f32; 100];
        importance[5] = 10.0;
        importance[10] = 9.0;
        importance[7] = 8.0;
        for imp in importance[4..20].iter_mut() {
            if *imp == 0.0 {
                *imp = 0.01;
            }
        }

        // target=10, keep=10, available=6, hh=3, recent=3
        // recent_start = max(4, 20-3) = 17
        // Top 3 HH from evictable [4..17]: pos 5(10.0), 10(9.0), 7(8.0)
        // Final layout: [0,1,2,3] [5,7,10] [17,18,19]
        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10);

        // All markers should be in ascending position order
        let mut prev_marker = -1.0f32;
        for i in 0..cache.current_pos {
            let (k, _) = read_marker(&cache, i);
            assert!(
                k > prev_marker,
                "Position {} has marker {} <= prev {}",
                i,
                k,
                prev_marker
            );
            prev_marker = k;
        }

        // Last 3 should be recent tokens (originally 17, 18, 19)
        for i in 0..3 {
            let (k, _) = read_marker(&cache, cache.current_pos - 3 + i);
            assert_eq!(k, ((17 + i) * 100) as f32);
        }
    }

    #[test]
    fn test_evict_fallback_works() {
        let policy = H2OPolicy::new(0.5, 0);
        let mut cache = make_cache(20);
        policy.evict(&mut cache, 10).unwrap();
        assert!(cache.current_pos < 20);
    }

    #[test]
    fn test_name() {
        assert_eq!(H2OPolicy::new(0.5, 0).name(), "h2o");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Group A: Score-based token identification accuracy
    // ═══════════════════════════════════════════════════════════════════

    /// Helper: create a cache with `n_heads` KV heads and `head_dim` dimensions.
    fn make_cache_multihead(pos: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        let max_seq = 100;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * kv_heads * head_dim * std::mem::size_of::<f32>();
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = pos;
        cache
    }

    /// Helper: write marker to ALL elements of a position (fills entire position with val).
    fn write_marker_full(cache: &mut KVCache, pos: usize, val: f32) {
        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let elems_per_pos = heads * dim;
        let k_ptr = cache.k_buffer.as_mut_ptr();
        let v_ptr = cache.v_buffer.as_mut_ptr();
        unsafe {
            let k_f32 = k_ptr as *mut f32;
            let v_f32 = v_ptr as *mut f32;
            for e in 0..elems_per_pos {
                *k_f32.add(pos * elems_per_pos + e) = val;
                *v_f32.add(pos * elems_per_pos + e) = val + 0.5; // V offset for distinguishing K/V
            }
        }
    }

    /// Helper: read markers from ALL elements of a position.
    /// Returns (first_k, first_v) to identify which original token is here.
    fn read_marker_full(cache: &KVCache, pos: usize) -> (f32, f32) {
        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let elems_per_pos = heads * dim;
        let k_ptr = cache.k_buffer.as_mut_ptr();
        let v_ptr = cache.v_buffer.as_mut_ptr();
        unsafe {
            let k_val = *(k_ptr as *const f32).add(pos * elems_per_pos);
            let v_val = *(v_ptr as *const f32).add(pos * elems_per_pos);
            (k_val, v_val)
        }
    }

    /// Helper: collect all surviving token markers after eviction.
    fn collect_markers(cache: &KVCache) -> Vec<f32> {
        (0..cache.current_pos)
            .map(|i| read_marker(cache, i).0)
            .collect()
    }

    #[test]
    fn test_hh_selects_highest_scores() {
        // Core test: verify that the top-K scored tokens from evictable region
        // are exactly the ones that survive as HH.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        // Each position gets a unique marker = (pos+1)*100
        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // Importance: assign distinct scores to evictable region
        // Evictable will be [4..17) with target=10, keep_ratio=0.5
        // hh_budget=3, recent_budget=3
        let mut importance = vec![0.0f32; 100];
        // Top 3 scores in evictable: pos 7 (9.0), pos 12 (8.5), pos 5 (8.0)
        importance[7] = 9.0;
        importance[12] = 8.5;
        importance[5] = 8.0;
        // Lower scores for other evictable tokens
        importance[4] = 1.0;
        importance[6] = 2.0;
        importance[8] = 3.0;
        importance[9] = 0.5;
        importance[10] = 1.5;
        importance[11] = 2.5;
        importance[13] = 0.1;
        importance[14] = 0.2;
        importance[15] = 0.3;
        importance[16] = 0.4;

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10);

        let markers = collect_markers(&cache);

        // Expected layout: [prefix 0-3] [HH sorted by pos: 5, 7, 12] [recent: 17, 18, 19]
        // Markers:          [100..400]   [600, 800, 1300]             [1800, 1900, 2000]
        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix (pos 0,1,2,3)
            600.0, 800.0, 1300.0, // HH (originally pos 5, 7, 12)
            1800.0, 1900.0, 2000.0, // recent (pos 17, 18, 19)
        ];
        assert_eq!(
            markers, expected,
            "HH must be exactly the top-3 scored tokens from evictable region"
        );
    }

    #[test]
    fn test_hh_ignores_recent_region_scores() {
        // Even if recent tokens have highest scores, they stay as "recent",
        // and HH is selected only from evictable region.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        let mut importance = vec![0.0f32; 100];
        // Give VERY high scores to recent region tokens (17, 18, 19)
        importance[17] = 999.0;
        importance[18] = 998.0;
        importance[19] = 997.0;
        // Moderate scores in evictable region
        importance[6] = 5.0;
        importance[10] = 4.0;
        importance[14] = 3.0;
        for i in 4..17 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        // target=10, hh=3, recent=3, recent_start=17
        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        let markers = collect_markers(&cache);

        // HH should be 6, 10, 14 (top-3 from evictable [4..17))
        // NOT 17, 18, 19 (those are in recent, not HH candidates)
        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix
            700.0, 1100.0, 1500.0, // HH from evictable: pos 6, 10, 14
            1800.0, 1900.0, 2000.0, // recent: pos 17, 18, 19
        ];
        assert_eq!(
            markers, expected,
            "Recent tokens must not participate in HH selection"
        );
    }

    #[test]
    fn test_hh_ignores_prefix_region_scores() {
        // Prefix tokens (even with high scores) are never in HH candidate set.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        let mut importance = vec![0.0f32; 100];
        // Give highest scores to prefix tokens
        importance[0] = 100.0;
        importance[1] = 99.0;
        importance[2] = 98.0;
        importance[3] = 97.0;
        // Evictable region scores
        importance[5] = 5.0;
        importance[9] = 4.0;
        importance[13] = 3.0;
        for i in 4..17 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        let markers = collect_markers(&cache);

        // Prefix stays at [0-3], HH from evictable only
        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix stays
            600.0, 1000.0, 1400.0, // HH: pos 5(5.0), 9(4.0), 13(3.0)
            1800.0, 1900.0, 2000.0, // recent
        ];
        assert_eq!(
            markers, expected,
            "Prefix tokens must not be in HH candidate pool"
        );
    }

    #[test]
    fn test_low_score_tokens_evicted() {
        // Verify that the tokens with lowest scores are the ones removed.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // Give each evictable token a descending score by position
        // pos 4=16, pos 5=15, ..., pos 16=4
        let mut importance = vec![0.0f32; 100];
        for i in 4..17 {
            importance[i] = (20 - i) as f32;
        }

        // target=10, hh=3, recent=3, evictable=[4..17)
        // Top 3 from evictable by score: pos 4(16), pos 5(15), pos 6(14)
        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        let markers = collect_markers(&cache);

        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix
            500.0, 600.0, 700.0, // HH: pos 4(16), 5(15), 6(14) — highest scores
            1800.0, 1900.0, 2000.0, // recent
        ];
        assert_eq!(
            markers, expected,
            "Tokens with lowest scores must be evicted, highest kept"
        );
    }

    #[test]
    fn test_score_ranking_patterns() {
        // Verify HH selection under three distinct score distributions over the evictable
        // region [4..14). All cases: current=16, target=8, hh=2, recent=2, recent_start=14.
        struct Case {
            label: &'static str,
            // importance[4..14] assigned by closure; values outside this range stay 0.0
            scores: [f32; 10], // index 0 → pos 4, index 9 → pos 13
            // expected markers for the 8 slots after eviction
            expected: [f32; 8],
        }

        let cases = [
            Case {
                label: "descending (older tokens score higher)",
                // pos 4=10, pos 5=9, ..., pos 13=1 → top-2: pos 4, 5
                scores: [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                expected: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 150.0, 160.0],
            },
            Case {
                label: "ascending (newer evictable tokens score higher)",
                // pos 4=1, pos 5=2, ..., pos 13=10 → top-2: pos 12, 13
                scores: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                expected: [10.0, 20.0, 30.0, 40.0, 130.0, 140.0, 150.0, 160.0],
            },
            Case {
                label: "V-shape (both ends high, middle low)",
                // pos 4=9.0, pos 13=8.0 → top-2: pos 4, 13
                scores: [9.0, 2.0, 1.0, 0.5, 0.1, 0.1, 0.5, 1.0, 2.0, 8.0],
                expected: [10.0, 20.0, 30.0, 40.0, 50.0, 140.0, 150.0, 160.0],
            },
        ];

        for c in &cases {
            let policy = H2OPolicy::new(0.5, 0); // prefix=4
            let mut cache = make_cache(16);
            for i in 0..16 {
                write_marker(&mut cache, i, (i + 1) as f32 * 10.0);
            }
            let mut importance = vec![0.0f32; 100];
            for (offset, &score) in c.scores.iter().enumerate() {
                importance[4 + offset] = score;
            }
            policy
                .evict_with_scores(&mut cache, 8, &importance)
                .unwrap();
            let markers = collect_markers(&cache);
            assert_eq!(markers, c.expected.to_vec(), "case='{}'", c.label);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Group B: Boundary conditions and budget calculation
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_keep_ratio_boundary() {
        // keep_ratio=0.0 → hh_budget=0 (pure sliding window, no HH partition).
        // keep_ratio=1.0 → recent_budget=0 (pure HH, recent tokens can be evicted by score).
        // Both cases: current=20, prefix=4 (clamped from 0), target=10, available=6.
        struct Case {
            label: &'static str,
            keep_ratio: f32,
            expected: Vec<f32>,
        }

        let cases = [
            Case {
                label: "keep_ratio=0.0: high-score tokens must still be evicted (no HH partition)",
                keep_ratio: 0.0,
                // hh=0, recent=6, recent_start=14 → prefix + recent[14..20]
                expected: vec![
                    100.0, 200.0, 300.0, 400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0,
                ],
            },
            Case {
                label: "keep_ratio=1.0: recent tokens with low scores must be evicted",
                keep_ratio: 1.0,
                // hh=6, recent=0, evictable=[4..20), top-6 by score: pos 4-9
                expected: vec![
                    100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
                ],
            },
        ];

        for c in &cases {
            let policy = H2OPolicy::new(c.keep_ratio, 0); // prefix=4 (clamped)
            let mut cache = make_cache(20);
            for i in 0..20 {
                write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
            }
            // Shared importance: old evictable tokens (pos 4-9) score high, newer ones low.
            // ratio=0.0 ignores HH entirely (all budget → recent);
            // ratio=1.0 selects top-6 from [4..20) → pos 4-9.
            let mut importance = vec![0.0f32; 100];
            importance[4] = 10.0;
            importance[5] = 9.0;
            importance[6] = 8.0;
            importance[7] = 7.0;
            importance[8] = 6.0;
            importance[9] = 5.0;
            for i in 10..20 {
                importance[i] = 0.01;
            }
            policy
                .evict_with_scores(&mut cache, 10, &importance)
                .unwrap();
            let markers = collect_markers(&cache);
            assert_eq!(markers, c.expected, "case='{}'", c.label);
        }
    }

    #[test]
    fn test_evictable_boundary_token() {
        // Verify the exact boundary: recent_start-1 is evictable, recent_start is recent.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // target=10, hh=3, recent=3, recent_start=17
        // pos 16 should be evictable, pos 17 should be recent
        let mut importance = vec![0.0f32; 100];
        // Give pos 16 the highest score in evictable region
        importance[16] = 50.0;
        importance[5] = 40.0;
        importance[10] = 30.0;
        for i in 4..17 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        let markers = collect_markers(&cache);

        // pos 16 should be kept as HH (evictable, highest score)
        // pos 17 should be kept as recent
        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix
            600.0, 1100.0, 1700.0, // HH: pos 5(40), 10(30), 16(50) sorted by pos
            1800.0, 1900.0, 2000.0, // recent: 17, 18, 19
        ];
        assert_eq!(
            markers, expected,
            "Token at recent_start-1 must be evictable, at recent_start must be recent"
        );
    }

    #[test]
    fn test_budget_rounding_odd_available() {
        // available=7 with keep_ratio=0.5 → hh=(7*0.5)=3 (truncated), recent=4
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        let mut importance = vec![0.0f32; 100];
        for i in 4..20 {
            importance[i] = (20 - i) as f32;
        }

        // target=11, keep=11, available=11-4=7
        // hh = (7*0.5) as usize = 3, recent = 7-3 = 4
        // recent_start = max(4, 20-4) = 16
        policy
            .evict_with_scores(&mut cache, 11, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 11); // 4 + 3 + 4

        let markers = collect_markers(&cache);

        // Top 3 from evictable [4..16): pos 4(16), 5(15), 6(14)
        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix
            500.0, 600.0, 700.0, // HH: pos 4, 5, 6
            1700.0, 1800.0, 1900.0, 2000.0, // recent: pos 16, 17, 18, 19
        ];
        assert_eq!(markers, expected);
    }

    #[test]
    fn test_tie_breaking_prefers_earlier_position() {
        // When multiple tokens have identical scores, stable sort preserves
        // original position order → earlier positions are selected first.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // All evictable tokens have the SAME score
        let mut importance = vec![0.0f32; 100];
        for i in 4..17 {
            importance[i] = 5.0; // identical scores
        }

        // target=10, hh=3, recent=3, evictable=[4..17) with 13 tokens all at 5.0
        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        let markers = collect_markers(&cache);

        // stable sort → first 3 from evictable (pos 4, 5, 6) become HH
        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix
            500.0, 600.0, 700.0, // HH: pos 4, 5, 6 (first 3 by position due to stable sort)
            1800.0, 1900.0, 2000.0, // recent
        ];
        assert_eq!(
            markers, expected,
            "Tie-breaking must select earlier positions (stable sort)"
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Group C: Compaction data integrity
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compaction_noncontiguous_hh_exact_data() {
        // HH at widely spaced positions: verify data at each position after compaction.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        // Write unique marker for each position
        for i in 0..20 {
            write_marker_full(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        let mut importance = vec![0.0f32; 100];
        importance[6] = 10.0;
        importance[11] = 9.0;
        importance[15] = 8.0;
        for i in 4..17 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        // Verify K and V separately
        for i in 0..cache.current_pos {
            let (k, v) = read_marker_full(&cache, i);
            assert_eq!(v, k + 0.5, "K/V mismatch at pos {}: K={}, V={}", i, k, v);
        }

        let markers: Vec<f32> = (0..cache.current_pos)
            .map(|i| read_marker_full(&cache, i).0)
            .collect();

        // pos 6→4, pos 11→5, pos 15→6, then recent 17→7, 18→8, 19→9
        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix unchanged
            700.0, 1200.0, 1600.0, // HH compacted from pos 6, 11, 15
            1800.0, 1900.0, 2000.0, // recent
        ];
        assert_eq!(markers, expected);
    }

    #[test]
    fn test_compaction_adjacent_hh_no_unnecessary_shift() {
        // HH tokens are right after prefix: pos 4, 5, 6 → no shift needed.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker_full(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // Make pos 4, 5, 6 the highest scoring in evictable
        let mut importance = vec![0.0f32; 100];
        importance[4] = 10.0;
        importance[5] = 9.0;
        importance[6] = 8.0;
        for i in 7..17 {
            importance[i] = 0.01;
        }

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        let markers: Vec<f32> = (0..cache.current_pos)
            .map(|i| read_marker_full(&cache, i).0)
            .collect();

        // HH = [4,5,6] are already in place (no shift needed)
        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix
            500.0, 600.0, 700.0, // HH (no shift)
            1800.0, 1900.0, 2000.0, // recent
        ];
        assert_eq!(markers, expected);
    }

    #[test]
    fn test_compaction_multihead_cache() {
        // Multiple KV heads: verify shift_positions correctly moves all heads.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let kv_heads = 4;
        let head_dim = 4;
        let mut cache = make_cache_multihead(20, kv_heads, head_dim);

        // Write per-head markers: for each position, each head gets a unique value
        for pos in 0..20 {
            let elems_per_pos = kv_heads * head_dim;
            let k_ptr = cache.k_buffer.as_mut_ptr();
            let v_ptr = cache.v_buffer.as_mut_ptr();
            unsafe {
                for h in 0..kv_heads {
                    let offset = pos * elems_per_pos + h * head_dim;
                    let marker = (pos * 100 + h * 10) as f32;
                    *(k_ptr as *mut f32).add(offset) = marker;
                    *(v_ptr as *mut f32).add(offset) = marker + 0.5;
                }
            }
        }

        let mut importance = vec![0.0f32; 100];
        importance[6] = 10.0;
        importance[10] = 9.0;
        importance[14] = 8.0;
        for i in 4..17 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10);

        // Verify per-head data integrity
        let elems_per_pos = kv_heads * head_dim;
        let k_ptr = cache.k_buffer.as_mut_ptr();
        let v_ptr = cache.v_buffer.as_mut_ptr();

        // Expected original positions at each slot after compaction:
        // [0,1,2,3, 6,10,14, 17,18,19]
        let expected_orig_pos = [0, 1, 2, 3, 6, 10, 14, 17, 18, 19];

        for (slot, &orig_pos) in expected_orig_pos.iter().enumerate() {
            for h in 0..kv_heads {
                let offset = slot * elems_per_pos + h * head_dim;
                let expected_marker = (orig_pos * 100 + h * 10) as f32;
                unsafe {
                    let k_val = *(k_ptr as *const f32).add(offset);
                    let v_val = *(v_ptr as *const f32).add(offset);
                    assert_eq!(
                        k_val, expected_marker,
                        "K mismatch at slot={} head={}: got {} expected {}",
                        slot, h, k_val, expected_marker
                    );
                    assert_eq!(
                        v_val,
                        expected_marker + 0.5,
                        "V mismatch at slot={} head={}: got {} expected {}",
                        slot,
                        h,
                        v_val,
                        expected_marker + 0.5
                    );
                }
            }
        }
    }

    #[test]
    fn test_k_v_buffers_stay_synchronized() {
        // Write different patterns to K and V, verify both compacted identically.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        // K gets positive markers, V gets negative markers
        for i in 0..20 {
            let shape = cache.k_buffer.shape().dims();
            let elems_per_pos = shape[2] * shape[3];
            let k_ptr = cache.k_buffer.as_mut_ptr();
            let v_ptr = cache.v_buffer.as_mut_ptr();
            unsafe {
                *(k_ptr as *mut f32).add(i * elems_per_pos) = (i + 1) as f32;
                *(v_ptr as *mut f32).add(i * elems_per_pos) = -((i + 1) as f32);
            }
        }

        let mut importance = vec![0.0f32; 100];
        importance[7] = 10.0;
        importance[11] = 9.0;
        importance[15] = 8.0;
        for i in 4..17 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        // Expected orig positions: [0,1,2,3, 7,11,15, 17,18,19]
        let expected_orig = [0, 1, 2, 3, 7, 11, 15, 17, 18, 19];

        let shape = cache.k_buffer.shape().dims();
        let elems_per_pos = shape[2] * shape[3];
        let k_ptr = cache.k_buffer.as_mut_ptr();
        let v_ptr = cache.v_buffer.as_mut_ptr();

        for (slot, &orig) in expected_orig.iter().enumerate() {
            unsafe {
                let k_val = *(k_ptr as *const f32).add(slot * elems_per_pos);
                let v_val = *(v_ptr as *const f32).add(slot * elems_per_pos);
                assert_eq!(k_val, (orig + 1) as f32, "K at slot {}", slot);
                assert_eq!(v_val, -((orig + 1) as f32), "V at slot {}", slot);
            }
        }
    }

    #[test]
    fn test_repeated_eviction_data_integrity() {
        // Two rounds of eviction on the same cache.
        // After 1st eviction, importance array is NOT realigned.
        // 2nd eviction uses stale importance → test the resulting behavior.
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // 1st eviction: 20→10
        let mut importance = vec![0.0f32; 100];
        importance[5] = 10.0;
        importance[9] = 9.0;
        importance[13] = 8.0;
        for i in 4..17 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10);

        // After 1st eviction, cache layout is:
        // [0,1,2,3, orig5, orig9, orig13, orig17, orig18, orig19]
        // markers: [100, 200, 300, 400, 600, 1000, 1400, 1800, 1900, 2000]
        let markers_after_1st = collect_markers(&cache);
        assert_eq!(
            markers_after_1st,
            vec![
                100.0, 200.0, 300.0, 400.0, 600.0, 1000.0, 1400.0, 1800.0, 1900.0, 2000.0
            ]
        );

        // 2nd eviction: 10→7 with NEW importance (reflecting actual positions now)
        // After 1st eviction, new positions:
        //   slot 4 = orig5 (marker 600)
        //   slot 5 = orig9 (marker 1000)
        //   slot 6 = orig13 (marker 1400)
        //   slot 7 = orig17 (marker 1800)
        //   slot 8 = orig18 (marker 1900)
        //   slot 9 = orig19 (marker 2000)
        //
        // target=7, keep=7, available=3, hh=1, recent=2
        // recent_start = max(4, 10-2) = 8
        // evictable = [4..8) = slots 4,5,6,7
        let mut importance2 = vec![0.0f32; 100];
        importance2[5] = 20.0; // slot 5 has highest score → keep as HH
        importance2[4] = 1.0;
        importance2[6] = 2.0;
        importance2[7] = 3.0;

        policy
            .evict_with_scores(&mut cache, 7, &importance2)
            .unwrap();

        assert_eq!(cache.current_pos, 7);

        let markers_after_2nd = collect_markers(&cache);
        // prefix [0-3], HH slot 5 (marker 1000), recent slots 8,9 (markers 1900, 2000)
        let expected = vec![
            100.0, 200.0, 300.0, 400.0,  // prefix
            1000.0, // HH: slot 5 (originally pos 9)
            1900.0, 2000.0, // recent: slots 8,9 (originally pos 18,19)
        ];
        assert_eq!(
            markers_after_2nd, expected,
            "2nd eviction must correctly compact based on current slot positions"
        );
    }

    #[test]
    fn test_aggressive_eviction_large_to_small() {
        // 50 tokens → 10 tokens (aggressive 80% eviction)
        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(50);

        for i in 0..50 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        // target=10, available=6, hh=3, recent=3
        // recent_start = max(4, 50-3) = 47
        // evictable = [4..47) = 43 tokens
        let mut importance = vec![0.0f32; 100];
        importance[20] = 100.0;
        importance[30] = 99.0;
        importance[40] = 98.0;
        for i in 4..47 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10);

        let markers = collect_markers(&cache);
        let expected = vec![
            1.0, 2.0, 3.0, 4.0, // prefix
            21.0, 31.0, 41.0, // HH: pos 20, 30, 40
            48.0, 49.0, 50.0, // recent: pos 47, 48, 49
        ];
        assert_eq!(markers, expected);
    }

    #[test]
    fn test_custom_prefix_size() {
        // Custom prefix=8, verify correct partitioning.
        let policy = H2OPolicy::new(0.5, 8); // prefix=8
        let mut cache = make_cache(30);

        for i in 0..30 {
            write_marker(&mut cache, i, (i + 1) as f32 * 10.0);
        }

        // target=16, keep=16, available=16-8=8, hh=4, recent=4
        // recent_start = max(8, 30-4) = 26
        // evictable = [8..26) = 18 tokens
        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[15] = 9.0;
        importance[20] = 8.0;
        importance[24] = 7.0;
        for i in 8..26 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        policy
            .evict_with_scores(&mut cache, 16, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 16);

        let markers = collect_markers(&cache);
        let expected = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, // prefix 0-7
            110.0, 160.0, 210.0, 250.0, // HH: pos 10, 15, 20, 24
            270.0, 280.0, 290.0, 300.0, // recent: pos 26, 27, 28, 29
        ];
        assert_eq!(markers, expected);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 3: Score reset after eviction — verify reset is necessary
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_reset_prevents_score_position_misalignment() {
        // After eviction, shift_positions() moves KV data:
        //   Original pos 10 → new pos 5
        // But the importance array is NOT rearranged.
        // Without reset: importance[5] still holds OLD pos 5's score.
        // With reset: importance[5] = 0, re-accumulated correctly for new data.
        //
        // This test demonstrates the misalignment that reset prevents.

        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // Give tokens at pos 5, 9, 13 highest scores (they become HH)
        let mut importance = vec![0.0f32; 100];
        importance[5] = 10.0;
        importance[9] = 9.0;
        importance[13] = 8.0;
        for i in 4..17 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();
        assert_eq!(cache.current_pos, 10);

        // After eviction, layout: [0,1,2,3, orig5, orig9, orig13, orig17, orig18, orig19]
        // New slot 4 = orig pos 5 (marker 600)
        // New slot 5 = orig pos 9 (marker 1000)
        // New slot 6 = orig pos 13 (marker 1400)

        // WITHOUT reset: importance[4] = 0.01 (was old pos 4's score, a non-HH token)
        // but slot 4 now holds orig pos 5's data (marker 600, was HH with score 10.0)
        // This is a MISALIGNMENT: score doesn't match the data.
        assert!(
            (importance[4] - 0.01).abs() < 1e-6,
            "importance[4] = {} (stale score for old pos 4, not new data at slot 4)",
            importance[4]
        );

        // The data at slot 4 is actually orig pos 5 (had score 10.0), not pos 4 (score 1.0).
        let (k_val, _) = read_marker(&cache, 4);
        assert_eq!(k_val, 600.0, "Slot 4 holds orig pos 5's data");

        // After reset: importance[4] = 0.0 → will be re-accumulated correctly
        // for the data that is actually at slot 4 (orig pos 5).
        importance.fill(0.0); // simulating reset
        assert_eq!(importance[4], 0.0, "After reset, slot 4 score = 0 (clean)");
    }

    #[test]
    fn test_reset_allows_former_hh_to_be_evicted() {
        // Round 1: Token at pos 5 has high score → selected as HH
        // After eviction + reset, it re-accumulates with LOW scores
        // Round 2: It should now be evictable (no "HH lock-in")

        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // Round 1: pos 5 is heavy hitter
        let mut importance1 = vec![0.01f32; 100];
        importance1[5] = 10.0;
        importance1[9] = 9.0;
        importance1[13] = 8.0;

        policy
            .evict_with_scores(&mut cache, 10, &importance1)
            .unwrap();
        assert_eq!(cache.current_pos, 10);

        // After round 1: slots [0,1,2,3, orig5, orig9, orig13, orig17, orig18, orig19]
        // Slot 4 = orig5 (was HH in round 1)
        let (k4, _) = read_marker(&cache, 4);
        assert_eq!(k4, 600.0); // orig pos 5

        // Reset (simulating what generate.rs does)
        importance1.fill(0.0);

        // Round 2: Simulate new tokens arriving (pos 10-14 unused, pos 7-9 are recent)
        // Now give slot 4 (orig pos 5) a LOW score → it should be evictable
        let mut importance2 = vec![0.0f32; 100];
        importance2[4] = 0.01; // slot 4 (former HH) now has low score
        importance2[5] = 20.0; // slot 5 (orig9) high score
        importance2[6] = 0.01; // slot 6 (orig13) low
        importance2[7] = 0.01; // slot 7 (orig17)

        // target=7: prefix=4, available=3, hh=1, recent=2
        // recent_start = max(4, 10-2) = 8
        // evictable = [4..8): slots 4,5,6,7
        // HH = top-1 by score: slot 5 (score 20.0)
        policy
            .evict_with_scores(&mut cache, 7, &importance2)
            .unwrap();

        assert_eq!(cache.current_pos, 7);

        // Slot 4 (former HH, orig pos 5) should be EVICTED because its score is now 0.01
        // New layout: [prefix 0-3] [HH: slot5=orig9] [recent: slot8=orig18, slot9=orig19]
        let markers = collect_markers(&cache);
        let expected = vec![
            100.0, 200.0, 300.0, 400.0,  // prefix
            1000.0, // HH: orig9 (slot 5, score 20.0)
            1900.0, 2000.0, // recent: orig18, orig19
        ];
        assert_eq!(
            markers, expected,
            "Former HH (orig5) must be evictable after reset + low re-score"
        );
    }

    #[test]
    fn test_without_reset_stale_scores_cause_wrong_eviction() {
        // Demonstrate: without reset, stale importance scores cause incorrect HH selection.
        //
        // Round 1: evict 20→10, positions shift
        // Round 2: use OLD importance array (not reset) → wrong tokens selected as HH

        let policy = H2OPolicy::new(0.5, 0); // prefix=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // Round 1 importance: pos 5, 9, 13 are HH
        let mut importance = vec![0.01f32; 100];
        importance[5] = 10.0;
        importance[9] = 9.0;
        importance[13] = 8.0;

        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();
        assert_eq!(cache.current_pos, 10);

        // After round 1: [0,1,2,3, orig5, orig9, orig13, orig17, orig18, orig19]
        // DO NOT RESET importance (simulate the bug)

        // Round 2: use stale importance for 2nd eviction (10→7)
        // Stale importance[4] = 0.01 (was pos 4's score, but data is now orig5)
        // Stale importance[5] = 10.0 (was pos 5's score, but data is now orig9)
        // Stale importance[6] = 0.01 (was pos 6's score, but data is now orig13)
        // Stale importance[7] = 0.01 (was pos 7's score, but data is now orig17)
        //
        // target=7: hh=1, recent=2, recent_start=8
        // evictable=[4..8), stale scores: 0.01, 10.0, 0.01, 0.01
        // HH selection (top-1): slot 5 (stale score 10.0)
        //
        // But slot 5 has data from orig9 (marker 1000).
        // The stale score 10.0 was for OLD pos 5 (marker 600), which is now at slot 4.
        // This is a WRONG HH selection based on stale data.

        policy
            .evict_with_scores(&mut cache, 7, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 7);

        let markers = collect_markers(&cache);
        // With stale scores: HH=slot5 (stale score 10.0, but actually orig9's data)
        // The "correct" selection should depend on actual attention patterns,
        // but stale scores select based on old position mapping.
        //
        // This test documents the behavior — stale scores cause misaligned selection.
        // Result: prefix [100,200,300,400] + HH slot5 [1000] + recent [1900,2000]
        let stale_result = vec![
            100.0, 200.0, 300.0, 400.0,  // prefix
            1000.0, // HH: slot 5 (stale score 10.0 ← was pos 5)
            1900.0, 2000.0, // recent
        ];
        assert_eq!(
            markers, stale_result,
            "Stale scores select HH based on old position mapping"
        );

        // Note: orig5 (marker 600, now at slot 4) had the REAL high importance
        // but got evicted because stale importance[4]=0.01 (was old pos 4's score).
        // This demonstrates why reset is necessary.
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 4: Budget calculation edge cases
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_budget_calculation_exact_values() {
        // Verify exact budget allocation for multiple (keep_ratio, prefix, current, target) combos.
        #[allow(dead_code)]
        struct Case {
            keep_ratio: f32,
            prefix: usize,
            current: usize,
            target: usize,
            expected_hh: usize,
            expected_recent: usize,
        }

        let cases = vec![
            Case {
                keep_ratio: 0.5,
                prefix: 4,
                current: 20,
                target: 10,
                expected_hh: 3,
                expected_recent: 3,
                // available=10-4=6, hh=3, recent=3
            },
            Case {
                keep_ratio: 0.8,
                prefix: 4,
                current: 20,
                target: 14,
                expected_hh: 8,
                expected_recent: 2,
                // available=14-4=10, hh=8, recent=2
            },
            Case {
                keep_ratio: 0.0,
                prefix: 4,
                current: 20,
                target: 10,
                expected_hh: 0,
                expected_recent: 6,
                // available=6, hh=0, recent=6 (pure sliding window)
            },
            Case {
                keep_ratio: 1.0,
                prefix: 4,
                current: 20,
                target: 10,
                expected_hh: 6,
                expected_recent: 0,
                // available=6, hh=6, recent=0 (pure HH)
            },
            Case {
                keep_ratio: 0.5,
                prefix: 8,
                current: 30,
                target: 16,
                expected_hh: 4,
                expected_recent: 4,
                // available=16-8=8, hh=4, recent=4
            },
        ];

        for (idx, c) in cases.iter().enumerate() {
            let keep = c.target.max(c.prefix + 2);
            let available = keep.saturating_sub(c.prefix);
            let hh_budget = (available as f32 * c.keep_ratio) as usize;
            let recent_budget = available - hh_budget;

            assert_eq!(
                hh_budget, c.expected_hh,
                "Case {}: hh_budget = {} (expected {})",
                idx, hh_budget, c.expected_hh
            );
            assert_eq!(
                recent_budget, c.expected_recent,
                "Case {}: recent_budget = {} (expected {})",
                idx, recent_budget, c.expected_recent
            );
            assert_eq!(
                c.prefix + hh_budget + recent_budget,
                keep,
                "Case {}: prefix + hh + recent must equal keep",
                idx
            );
        }
    }

    #[test]
    fn test_evictable_fewer_than_hh_budget() {
        // Edge case: hh_budget > evictable tokens → keep all evictable as HH.
        let policy = H2OPolicy::new(0.9, 0); // prefix=4
        let mut cache = make_cache(10);

        for i in 0..10 {
            write_marker(&mut cache, i, (i + 1) as f32 * 100.0);
        }

        // target=9, keep=9, available=9-4=5
        // hh_budget = (5*0.9) = 4, recent_budget = 1
        // recent_start = max(4, 10-1) = 9
        // evictable = [4..9) = 5 tokens
        // Top-4 of 5 evictable → keeps 4 as HH, evicts 1
        let mut importance = vec![0.0f32; 100];
        importance[4] = 5.0;
        importance[5] = 4.0;
        importance[6] = 3.0;
        importance[7] = 2.0;
        importance[8] = 1.0; // lowest → evicted

        policy
            .evict_with_scores(&mut cache, 9, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 9);
        let markers = collect_markers(&cache);
        let expected = vec![
            100.0, 200.0, 300.0, 400.0, // prefix
            500.0, 600.0, 700.0, 800.0,  // HH: pos 4,5,6,7
            1000.0, // recent: pos 9
        ];
        assert_eq!(markers, expected);
    }
}
