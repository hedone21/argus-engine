// NEON-optimized score accumulation removed: scalar path is used for all
// architectures. The NEON specialization produced incorrect results on ARM
// (all-zero importance / NaN sum) while the performance difference was
// negligible (~0.66ms). See git history for the removed neon_scores module.

/// Accumulates per-token attention importance scores across layers.
///
/// During decode, each layer's post-softmax attention weights are aggregated
/// into a per-token importance score. H2O uses these scores to decide
/// which tokens to keep vs evict.
pub struct AttentionScoreAccumulator {
    /// Per-token cumulative importance scores, indexed by cache position.
    /// Updated once per step via `end_step()`.
    importance: Vec<f32>,
    /// Per-token step-local importance buffer.
    /// Within a single decode step, each layer's score is aggregated here
    /// using MAX (per-layer independence), then flushed to `importance` in `end_step()`.
    step_importance: Vec<f32>,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Which layers to track. Empty means track all layers.
    tracked_layers: Vec<usize>,
    /// Exponential decay factor (0.0 = no decay, 1.0 = full decay).
    decay: f32,
    /// Whether accumulation is active.
    active: bool,
    // ── GQA-aware fields ──
    /// Number of KV heads for GQA grouping. 0 = GQA mode disabled.
    n_kv_heads: usize,
    /// Per-KV-head cumulative importance: `[n_kv_heads * max_seq_len]`, row-major.
    head_importance: Vec<f32>,
    /// Per-KV-head step-local buffer (same layout).
    head_step_importance: Vec<f32>,
    // ── CAOTE fields ──
    /// Last tracked layer's per-KV-head attention from the most recent decode step.
    /// Layout: `[n_kv_heads * max_seq_len]`, row-major. Overwritten each layer
    /// (not MAX), so after a decode step it holds the last tracked layer's values.
    /// These are proper softmax-derived scores (sum ≈ 1.0 per head).
    last_layer_head_attn: Vec<f32>,
    // ── Time-normalization fields ──
    /// Per-token count of steps in which this position was active.
    step_count: Vec<u32>,
    /// Time-normalized importance: `importance[t] / step_count[t]`.
    /// Computed at each `end_step()` when `time_normalize` is enabled.
    normalized: Vec<f32>,
    /// If true, `importance_scores()` returns time-normalized values.
    time_normalize: bool,
}

impl AttentionScoreAccumulator {
    pub fn new(
        max_seq_len: usize,
        _n_heads: usize,
        total_layers: usize,
        last_n_layers: usize,
        decay: f32,
    ) -> Self {
        let tracked_layers = if last_n_layers == 0 || last_n_layers >= total_layers {
            Vec::new()
        } else {
            ((total_layers - last_n_layers)..total_layers).collect()
        };

        Self {
            importance: vec![0.0; max_seq_len],
            step_importance: vec![0.0; max_seq_len],
            max_seq_len,
            tracked_layers,
            decay: decay.clamp(0.0, 1.0),
            active: false,
            n_kv_heads: 0,
            head_importance: Vec::new(),
            head_step_importance: Vec::new(),
            last_layer_head_attn: Vec::new(),
            step_count: vec![0; max_seq_len],
            normalized: vec![0.0; max_seq_len],
            time_normalize: false,
        }
    }

    /// Create a GQA-aware accumulator that tracks per-KV-head importance.
    ///
    /// In addition to flat per-token importance (backward compatible),
    /// maintains a 2D `[n_kv_heads, max_seq_len]` importance matrix where
    /// Q-head scores are averaged within each GQA group.
    pub fn new_gqa(
        max_seq_len: usize,
        _n_heads: usize,
        n_kv_heads: usize,
        total_layers: usize,
        last_n_layers: usize,
        decay: f32,
    ) -> Self {
        let tracked_layers = if last_n_layers == 0 || last_n_layers >= total_layers {
            Vec::new()
        } else {
            ((total_layers - last_n_layers)..total_layers).collect()
        };

        let head_buf_size = n_kv_heads * max_seq_len;
        Self {
            importance: vec![0.0; max_seq_len],
            step_importance: vec![0.0; max_seq_len],
            max_seq_len,
            tracked_layers,
            decay: decay.clamp(0.0, 1.0),
            active: false,
            n_kv_heads,
            head_importance: vec![0.0; head_buf_size],
            head_step_importance: vec![0.0; head_buf_size],
            last_layer_head_attn: vec![0.0; head_buf_size],
            step_count: vec![0; max_seq_len],
            normalized: vec![0.0; max_seq_len],
            time_normalize: false,
        }
    }

    /// Enable time-normalized scoring.
    /// When enabled, `importance_scores()` returns `importance[t] / step_count[t]`
    /// instead of raw cumulative scores, removing the time-in-cache bias.
    pub fn set_time_normalize(&mut self, enable: bool) {
        self.time_normalize = enable;
    }

    /// Returns whether this layer should be tracked.
    #[inline]
    pub fn should_track_layer(&self, layer_idx: usize) -> bool {
        self.active && (self.tracked_layers.is_empty() || self.tracked_layers.contains(&layer_idx))
    }

    /// Called once per decode step before layer iteration.
    /// Applies decay to cumulative importance and clears step-local buffer.
    pub fn begin_step(&mut self) {
        if !self.active {
            return;
        }
        if self.decay > 0.0 {
            let factor = 1.0 - self.decay;
            for v in self.importance.iter_mut() {
                *v *= factor;
            }
            for v in self.head_importance.iter_mut() {
                *v *= factor;
            }
        }
        self.step_importance.fill(0.0);
        self.head_step_importance.fill(0.0);
        self.last_layer_head_attn.fill(0.0);
    }

    /// Accumulate post-softmax attention scores from one layer.
    ///
    /// Uses per-layer MAX aggregation: for each token, the step-local importance
    /// is the maximum across all layers (not sum). This ensures tokens critical
    /// to ANY layer are preserved, matching the H2O paper's per-layer independence.
    ///
    /// Within a single layer, scores across query heads are summed first to get
    /// a single per-token score for that layer.
    ///
    /// `scores`: flat buffer `[n_heads_q * stride]`, where stride >= cache_seq_len.
    /// `stride`: distance between head score arrays.
    /// `cache_seq_len`: number of valid token positions covered by this scores buffer.
    /// `n_heads_q`: number of query heads.
    /// `score_offset`: cache position of scores[t=0]. 0 for global attention; kv_start_pos
    ///   for local (sliding-window) attention layers so that scores are mapped to the correct
    ///   absolute cache positions.
    pub fn accumulate_layer(
        &mut self,
        scores: &[f32],
        stride: usize,
        cache_seq_len: usize,
        n_heads_q: usize,
        score_offset: usize,
    ) {
        let len = cache_seq_len.min(self.max_seq_len);

        for t in 0..len {
            let pos = score_offset + t;
            if pos >= self.max_seq_len {
                break;
            }
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            self.step_importance[pos] = self.step_importance[pos].max(layer_score);
        }
    }

    /// GQA-aware accumulation: in addition to flat per-token scores,
    /// computes per-KV-head importance by averaging Q-head scores within
    /// each GQA group.
    ///
    /// `scores`: flat buffer `[n_heads_q * stride]`.
    /// `n_heads_q`: total query heads (must be divisible by `n_kv_heads`).
    /// `score_offset`: cache position of scores[t=0]. 0 for global attention; kv_start_pos
    ///   for local (sliding-window) attention layers.
    pub fn accumulate_layer_gqa(
        &mut self,
        scores: &[f32],
        stride: usize,
        cache_seq_len: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
        score_offset: usize,
    ) {
        let len = cache_seq_len.min(self.max_seq_len);

        let n_rep = n_heads_q / n_kv_heads;
        let inv_rep = 1.0 / n_rep as f32;

        for t in 0..len {
            let pos = score_offset + t;
            if pos >= self.max_seq_len {
                break;
            }
            // Flat accumulation (backward compatible with H2O)
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            self.step_importance[pos] = self.step_importance[pos].max(layer_score);

            // Per-KV-head: average Q-heads within each GQA group
            for kv_h in 0..n_kv_heads {
                let mut group_score = 0.0f32;
                for r in 0..n_rep {
                    group_score += scores[(kv_h * n_rep + r) * stride + t];
                }
                group_score *= inv_rep;
                let idx = kv_h * self.max_seq_len + pos;
                self.head_step_importance[idx] = self.head_step_importance[idx].max(group_score);
                // CAOTE: overwrite (not MAX) to keep the last tracked layer's raw attention.
                // NaN guard: softmax can produce NaN when all logits are -inf (e.g.
                // masked tokens). f32::max() silently swallows NaN for the other
                // accumulators, but direct assignment propagates it here.
                self.last_layer_head_attn[idx] = if group_score.is_nan() {
                    0.0
                } else {
                    group_score
                };
            }
        }
    }

    /// Called once per decode step after all layers have been processed.
    /// Flushes step-local importance (per-layer MAX) into cumulative importance.
    pub fn end_step(&mut self) {
        if !self.active {
            return;
        }

        for t in 0..self.max_seq_len {
            let step_val = self.step_importance[t];
            self.importance[t] += step_val;
            // Track step count: increment for positions that were in cache this step
            if step_val > 0.0 {
                self.step_count[t] += 1;
            }
            // Compute time-normalized score
            if self.time_normalize {
                let count = self.step_count[t].max(1) as f32;
                self.normalized[t] = self.importance[t] / count;
            }
        }
        for (cum, &step) in self
            .head_importance
            .iter_mut()
            .zip(self.head_step_importance.iter())
        {
            *cum += step;
        }
    }

    /// Get the importance scores slice.
    /// When time-normalization is enabled, returns `importance[t] / step_count[t]`
    /// (average per-step importance), removing the time-in-cache bias.
    pub fn importance_scores(&self) -> &[f32] {
        if self.time_normalize {
            &self.normalized
        } else {
            &self.importance
        }
    }

    /// Get raw cumulative scores regardless of normalization setting.
    pub fn raw_importance_scores(&self) -> &[f32] {
        &self.importance
    }

    /// Import cumulative scores computed on GPU.
    ///
    /// Called after `GpuScoreAccumulator::sync_to_cpu()` to transfer GPU-accumulated
    /// importance into this CPU-side accumulator. Overwrites (not adds to) the
    /// current importance scores since the GPU accumulator already includes decay
    /// and cumulative aggregation.
    ///
    /// `flat`: `[max_seq_len]` cumulative flat importance from GPU.
    /// `head`: `[n_kv_heads * max_seq_len]` cumulative per-head importance from GPU.
    pub fn import_gpu_scores(&mut self, flat: &[f32], head: &[f32]) {
        let len = flat.len().min(self.importance.len());
        self.importance[..len].copy_from_slice(&flat[..len]);

        if self.n_kv_heads > 0 {
            let head_len = head.len().min(self.head_importance.len());
            self.head_importance[..head_len].copy_from_slice(&head[..head_len]);

            // Also populate last_layer_head_attn from GPU head importance.
            // On GPU backends, accumulate_layer_gqa() cannot read GPU-only
            // score buffers, so last_layer_head_attn remains empty.
            // Using cumulative head importance as a proxy provides reasonable
            // QCF estimates (proportional to attention distribution).
            let attn_len = head_len.min(self.last_layer_head_attn.len());
            self.last_layer_head_attn[..attn_len].copy_from_slice(&head[..attn_len]);
        }

        // Recompute time-normalized scores if enabled
        if self.time_normalize {
            for t in 0..len {
                let count = self.step_count[t].max(1) as f32;
                self.normalized[t] = self.importance[t] / count;
            }
        }
    }

    /// Reset all accumulated scores (e.g., after eviction).
    pub fn reset(&mut self) {
        self.importance.fill(0.0);
        self.step_importance.fill(0.0);
        self.head_importance.fill(0.0);
        self.head_step_importance.fill(0.0);
        self.last_layer_head_attn.fill(0.0);
        self.step_count.fill(0);
        self.normalized.fill(0.0);
    }

    /// Activate accumulation.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Returns per-KV-head importance scores if GQA mode is active.
    /// Layout: `[n_kv_heads * max_seq_len]`, row-major.
    pub fn head_importance_scores(&self) -> Option<&[f32]> {
        if self.n_kv_heads > 0 {
            Some(&self.head_importance)
        } else {
            None
        }
    }

    /// Number of KV heads (0 = GQA mode disabled).
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Returns the last tracked layer's per-KV-head attention from the most
    /// recent decode step, if GQA mode is active.
    ///
    /// Layout: `[n_kv_heads * max_seq_len]`, row-major.
    /// Values are softmax-derived (sum ≈ 1.0 per head for active positions).
    pub fn last_step_head_attn(&self) -> Option<&[f32]> {
        if self.n_kv_heads > 0 {
            Some(&self.last_layer_head_attn)
        } else {
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulate_single_layer() {
        let mut acc = AttentionScoreAccumulator::new(8, 2, 4, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        // 2 heads, stride=8, cache_seq_len=4
        // head0: [0.1, 0.2, 0.3, 0.4, 0,0,0,0]
        // head1: [0.4, 0.3, 0.2, 0.1, 0,0,0,0]
        let scores = vec![
            0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0,
        ];
        acc.accumulate_layer(&scores, 8, 4, 2, 0);
        acc.end_step();

        // Per-token: sum across heads → layer_score, then MAX with step_importance
        // token 0: 0.1+0.4=0.5, token 1: 0.2+0.3=0.5, token 2: 0.3+0.2=0.5, token 3: 0.4+0.1=0.5
        let imp = acc.importance_scores();
        assert!((imp[0] - 0.5).abs() < 1e-6);
        assert!((imp[1] - 0.5).abs() < 1e-6);
        assert!((imp[2] - 0.5).abs() < 1e-6);
        assert!((imp[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_accumulate_multi_layer_uses_max() {
        // With per-layer MAX: step_importance = MAX(layer1_score, layer2_score)
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        let scores1 = vec![0.1, 0.2, 0.3, 0.4]; // layer 0 scores
        let scores2 = vec![0.4, 0.1, 0.1, 0.4]; // layer 1 scores

        acc.accumulate_layer(&scores1, 4, 4, 1, 0);
        acc.accumulate_layer(&scores2, 4, 4, 1, 0);
        acc.end_step();

        // Per-layer MAX: max(0.1,0.4)=0.4, max(0.2,0.1)=0.2, max(0.3,0.1)=0.3, max(0.4,0.4)=0.4
        let imp = acc.importance_scores();
        assert!((imp[0] - 0.4).abs() < 1e-6);
        assert!((imp[1] - 0.2).abs() < 1e-6);
        assert!((imp[2] - 0.3).abs() < 1e-6);
        assert!((imp[3] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_cumulative_across_steps() {
        // importance accumulates across multiple decode steps
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        acc.set_active(true);

        // Step 1
        acc.begin_step();
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 4, 4, 1, 0);
        acc.end_step();

        // Step 2
        acc.begin_step();
        acc.accumulate_layer(&[4.0, 3.0, 2.0, 1.0], 4, 4, 1, 0);
        acc.end_step();

        // importance = step1 + step2 = [5.0, 5.0, 5.0, 5.0]
        let imp = acc.importance_scores();
        assert!((imp[0] - 5.0).abs() < 1e-6);
        assert!((imp[1] - 5.0).abs() < 1e-6);
        assert!((imp[2] - 5.0).abs() < 1e-6);
        assert!((imp[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_decay() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.5);
        acc.set_active(true);

        // Step 1: accumulate
        acc.begin_step();
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 4, 4, 1, 0);
        acc.end_step();
        // importance = [1, 2, 3, 4]

        // Step 2: begin_step applies decay, then end_step adds new scores
        acc.begin_step(); // decay 0.5 → importance = [0.5, 1.0, 1.5, 2.0]
        // No accumulation in this step, just check decay effect
        let imp = acc.importance_scores();
        assert!((imp[0] - 0.5).abs() < 1e-6);
        assert!((imp[1] - 1.0).abs() < 1e-6);
        assert!((imp[2] - 1.5).abs() < 1e-6);
        assert!((imp[3] - 2.0).abs() < 1e-6);
    }

    /// A2SF (arXiv 2407.20485) 격리 게이트 — `decay=0.0` bit-identical 회귀.
    ///
    /// `--score-decay` flag 도입 후에도 **decay 미주입(0.0)** 이면 `begin_step()` 의
    /// `if self.decay > 0.0` 가드가 진입하지 않아 누적 결과가 종전과 정확히 동일해야 한다.
    /// 이 테스트는 decay=0.0 으로 3-step 누적 후 score 가 순수 합산(감쇠 0)임을 명시 단언한다 —
    /// flag 가 누적 로직을 한 줄도 바꾸지 않았음의 회귀 anchor(KV roadmap 항목 0 §4.2 게이트 (1)).
    #[test]
    fn test_a2sf_decay_zero_bit_identical() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        acc.set_active(true);

        // 3-step 누적 (각 step 동일 score). decay=0.0 → 순수 SUM.
        for _ in 0..3 {
            acc.begin_step();
            acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 4, 4, 1, 0);
            acc.end_step();
        }
        // 감쇠 0 → importance = 3 × [1,2,3,4] = [3,6,9,12] (정확히, 부동소수 오차 없음).
        let imp = acc.importance_scores();
        assert_eq!(imp[0], 3.0, "decay=0.0 순수 SUM (감쇠 미진입)");
        assert_eq!(imp[1], 6.0);
        assert_eq!(imp[2], 9.0);
        assert_eq!(imp[3], 12.0);
    }

    /// A2SF 격리 게이트 — `decay=0.8` 2-step 감쇠 동작.
    ///
    /// §4.2 게이트 (2): decay=0.8 일 때 begin_step 감쇠 factor=(1−0.8)=0.2 가 누적에 적용됨을
    /// 확인. step1 score 가 step2 begin_step 에서 0.2× 로 감쇠된 뒤 step2 score 가 더해진다.
    #[test]
    fn test_a2sf_decay_point_eight_two_step() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.8);
        acc.set_active(true);

        // Step 1: importance = [1, 2, 3, 4]
        acc.begin_step();
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 4, 4, 1, 0);
        acc.end_step();

        // Step 2: begin_step 이 factor=0.2 감쇠 → [0.2, 0.4, 0.6, 0.8],
        //         이후 [10, 10, 10, 10] 누적 → [10.2, 10.4, 10.6, 10.8].
        acc.begin_step();
        acc.accumulate_layer(&[10.0, 10.0, 10.0, 10.0], 4, 4, 1, 0);
        acc.end_step();

        let imp = acc.importance_scores();
        assert!(
            (imp[0] - 10.2).abs() < 1e-5,
            "factor=0.2 감쇠 + 누적, got {}",
            imp[0]
        );
        assert!((imp[1] - 10.4).abs() < 1e-5);
        assert!((imp[2] - 10.6).abs() < 1e-5);
        assert!((imp[3] - 10.8).abs() < 1e-5);
    }

    #[test]
    fn test_should_track_layer() {
        // Track last 2 of 4 layers => layers 2, 3
        let mut acc = AttentionScoreAccumulator::new(4, 1, 4, 2, 0.0);
        acc.set_active(true);

        assert!(!acc.should_track_layer(0));
        assert!(!acc.should_track_layer(1));
        assert!(acc.should_track_layer(2));
        assert!(acc.should_track_layer(3));
    }

    #[test]
    fn test_reset() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        acc.set_active(true);

        acc.begin_step();
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 4, 4, 1, 0);
        acc.end_step();
        acc.reset();

        assert!(acc.importance_scores().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_inactive_no_accumulation() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.0);
        // active defaults to false

        assert!(!acc.should_track_layer(0));
        assert!(!acc.is_active());

        // begin_step and end_step should be no-ops when inactive
        acc.begin_step();
        acc.end_step();
    }

    #[test]
    fn test_end_step_without_begin_step() {
        // end_step should work even if begin_step wasn't called (step_importance starts zeroed)
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        acc.set_active(true);
        acc.end_step();
        assert!(acc.importance_scores().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_per_layer_max_preserves_critical_tokens() {
        // Token critical to only one layer should still be preserved
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        // Layer 0: token 0 is critical (score=9.0), token 1 low (0.1)
        acc.accumulate_layer(&[9.0, 0.1, 0.1, 0.1], 4, 4, 1, 0);
        // Layer 1: token 1 is critical (score=9.0), token 0 low (0.1)
        acc.accumulate_layer(&[0.1, 9.0, 0.1, 0.1], 4, 4, 1, 0);
        acc.end_step();

        // MAX: both token 0 and token 1 should have high importance (9.0)
        let imp = acc.importance_scores();
        assert!((imp[0] - 9.0).abs() < 1e-6);
        assert!((imp[1] - 9.0).abs() < 1e-6);
        assert!((imp[2] - 0.1).abs() < 1e-6);
    }

    // ── GQA mode tests ──

    #[test]
    fn test_accumulate_gqa_groups_q_heads() {
        // 4 Q-heads, 2 KV-heads → n_rep=2 (heads 0,1 → KV0; heads 2,3 → KV1)
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 4, 2, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        // stride=4, cache_seq_len=4
        // Q-head 0: [1.0, 0.0, 0.0, 0.0]  ← KV group 0
        // Q-head 1: [0.0, 1.0, 0.0, 0.0]  ← KV group 0
        // Q-head 2: [0.0, 0.0, 1.0, 0.0]  ← KV group 1
        // Q-head 3: [0.0, 0.0, 0.0, 1.0]  ← KV group 1
        let scores = vec![
            1.0, 0.0, 0.0, 0.0, // Q-head 0
            0.0, 1.0, 0.0, 0.0, // Q-head 1
            0.0, 0.0, 1.0, 0.0, // Q-head 2
            0.0, 0.0, 0.0, 1.0, // Q-head 3
        ];
        acc.accumulate_layer_gqa(&scores, 4, 4, 4, 2, 0);
        acc.end_step();

        // KV head 0: avg(Q0, Q1) per token
        //   token 0: (1.0+0.0)/2 = 0.5
        //   token 1: (0.0+1.0)/2 = 0.5
        //   token 2: 0.0, token 3: 0.0
        let head_imp = acc.head_importance_scores().unwrap();
        assert!((head_imp[0] - 0.5).abs() < 1e-6); // KV0, tok0
        assert!((head_imp[1] - 0.5).abs() < 1e-6); // KV0, tok1
        assert!((head_imp[2] - 0.0).abs() < 1e-6); // KV0, tok2
        assert!((head_imp[3] - 0.0).abs() < 1e-6); // KV0, tok3

        // KV head 1: avg(Q2, Q3) per token
        //   token 2: (1.0+0.0)/2 = 0.5
        //   token 3: (0.0+1.0)/2 = 0.5
        assert!((head_imp[4] - 0.0).abs() < 1e-6); // KV1, tok0
        assert!((head_imp[5] - 0.0).abs() < 1e-6); // KV1, tok1
        assert!((head_imp[6] - 0.5).abs() < 1e-6); // KV1, tok2
        assert!((head_imp[7] - 0.5).abs() < 1e-6); // KV1, tok3
    }

    #[test]
    fn test_gqa_also_updates_flat() {
        // GQA accumulation should also update flat importance (backward compat)
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 4, 2, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        let scores = vec![
            1.0, 0.0, 0.0, 0.0, // Q0
            0.0, 1.0, 0.0, 0.0, // Q1
            0.0, 0.0, 1.0, 0.0, // Q2
            0.0, 0.0, 0.0, 1.0, // Q3
        ];
        acc.accumulate_layer_gqa(&scores, 4, 4, 4, 2, 0);
        acc.end_step();

        // Flat importance = sum across all Q-heads per token
        let imp = acc.importance_scores();
        assert!((imp[0] - 1.0).abs() < 1e-6);
        assert!((imp[1] - 1.0).abs() < 1e-6);
        assert!((imp[2] - 1.0).abs() < 1e-6);
        assert!((imp[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gqa_decay() {
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 2, 2, 1, 0, 0.5);
        acc.set_active(true);

        // Step 1
        acc.begin_step();
        // 2 Q-heads, 2 KV-heads (n_rep=1)
        let scores = vec![
            2.0, 4.0, 0.0, 0.0, // Q0 → KV0
            0.0, 0.0, 6.0, 8.0, // Q1 → KV1
        ];
        acc.accumulate_layer_gqa(&scores, 4, 4, 2, 2, 0);
        acc.end_step();

        // head_importance: KV0=[2,4,0,0], KV1=[0,0,6,8]

        // Step 2: decay 0.5
        acc.begin_step();
        let head_imp = acc.head_importance_scores().unwrap();
        assert!((head_imp[0] - 1.0).abs() < 1e-6); // 2*0.5
        assert!((head_imp[1] - 2.0).abs() < 1e-6); // 4*0.5
        assert!((head_imp[4] - 0.0).abs() < 1e-6); // 0*0.5
        assert!((head_imp[6] - 3.0).abs() < 1e-6); // 6*0.5
        assert!((head_imp[7] - 4.0).abs() < 1e-6); // 8*0.5
    }

    #[test]
    fn test_gqa_reset() {
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 2, 2, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        let scores = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        acc.accumulate_layer_gqa(&scores, 4, 4, 2, 2, 0);
        acc.end_step();
        acc.reset();

        assert!(acc.importance_scores().iter().all(|&v| v == 0.0));
        assert!(
            acc.head_importance_scores()
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn test_head_importance_accessor() {
        // Non-GQA: returns None
        let acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        assert!(acc.head_importance_scores().is_none());
        assert_eq!(acc.n_kv_heads(), 0);

        // GQA: returns Some
        let acc = AttentionScoreAccumulator::new_gqa(4, 4, 2, 1, 0, 0.0);
        assert!(acc.head_importance_scores().is_some());
        assert_eq!(acc.head_importance_scores().unwrap().len(), 2 * 4);
        assert_eq!(acc.n_kv_heads(), 2);
    }

    #[test]
    fn test_gqa_multi_layer_max() {
        // Per-layer MAX should apply to per-head importance too
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 2, 2, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        // Layer 0: KV0 high on tok0, KV1 low
        let scores_l0 = vec![
            9.0, 0.1, 0.1, 0.1, // Q0 → KV0
            0.1, 0.1, 0.1, 0.1, // Q1 → KV1
        ];
        acc.accumulate_layer_gqa(&scores_l0, 4, 4, 2, 2, 0);

        // Layer 1: KV0 low, KV1 high on tok2
        let scores_l1 = vec![
            0.1, 0.1, 0.1, 0.1, // Q0 → KV0
            0.1, 0.1, 9.0, 0.1, // Q1 → KV1
        ];
        acc.accumulate_layer_gqa(&scores_l1, 4, 4, 2, 2, 0);
        acc.end_step();

        let head_imp = acc.head_importance_scores().unwrap();
        // KV0: max(layer0, layer1) per token
        //   tok0: max(9.0, 0.1) = 9.0
        assert!((head_imp[0] - 9.0).abs() < 1e-6);
        //   tok1: max(0.1, 0.1) = 0.1
        assert!((head_imp[1] - 0.1).abs() < 1e-6);

        // KV1: max(layer0, layer1) per token
        //   tok2: max(0.1, 9.0) = 9.0
        assert!((head_imp[4 + 2] - 9.0).abs() < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: MAX vs SUM aggregation — verify MAX preserves layer-critical tokens
    // ═══════════════════════════════════════════════════════════════════

    /// Helper: simulate SUM aggregation (instead of MAX) for comparison.
    /// Returns per-token importance after one step with SUM across layers.
    fn simulate_sum_aggregation(
        layer_scores: &[Vec<f32>],
        n_heads_q: usize,
        max_seq_len: usize,
        cache_seq_len: usize,
    ) -> Vec<f32> {
        let stride = max_seq_len;
        let mut importance = vec![0.0f32; max_seq_len];
        for layer in layer_scores {
            // Sum across heads per token (same as accumulate_layer)
            for t in 0..cache_seq_len {
                let mut layer_score = 0.0f32;
                for h in 0..n_heads_q {
                    layer_score += layer[h * stride + t];
                }
                // SUM instead of MAX
                importance[t] += layer_score;
            }
        }
        importance
    }

    #[test]
    fn test_max_vs_sum_divergent_hh_ranking() {
        // Scenario where MAX and SUM produce DIFFERENT HH rankings.
        //
        // Token A: critical in layer 0 only (score=9.0), negligible in layer 1
        // Token B: uniformly moderate in both layers (score=4.5 each)
        // Token C: critical in layer 1 only (score=8.0), negligible in layer 0
        //
        // Per-token layer scores (1 head for simplicity):
        //   Layer 0: A=9.0, B=4.5, C=0.1
        //   Layer 1: A=0.1, B=4.5, C=8.0
        //
        // MAX: A=9.0, B=4.5, C=8.0 → ranking: A > C > B
        // SUM: A=9.1, B=9.0, C=8.1 → ranking: A > B > C
        //
        // If we need to keep only 2 tokens (HH budget=2):
        //   MAX keeps A, C — both layer-critical tokens preserved ✓
        //   SUM keeps A, B — C (layer-1 critical) lost ✗

        let max_seq = 8;
        let cache_seq = 3;
        let n_heads = 1;
        let stride = max_seq;

        // MAX aggregation (current implementation)
        let mut acc = AttentionScoreAccumulator::new(max_seq, n_heads, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        let layer0 = {
            let mut s = vec![0.0f32; n_heads * stride];
            s[0] = 9.0; // Token A
            s[1] = 4.5; // Token B
            s[2] = 0.1; // Token C
            s
        };
        let layer1 = {
            let mut s = vec![0.0f32; n_heads * stride];
            s[0] = 0.1; // Token A
            s[1] = 4.5; // Token B
            s[2] = 8.0; // Token C
            s
        };

        acc.accumulate_layer(&layer0, stride, cache_seq, n_heads, 0);
        acc.accumulate_layer(&layer1, stride, cache_seq, n_heads, 0);
        acc.end_step();

        let max_imp = acc.importance_scores();
        // MAX: A=max(9.0, 0.1)=9.0, B=max(4.5, 4.5)=4.5, C=max(0.1, 8.0)=8.0
        assert!((max_imp[0] - 9.0).abs() < 1e-6, "MAX A={}", max_imp[0]);
        assert!((max_imp[1] - 4.5).abs() < 1e-6, "MAX B={}", max_imp[1]);
        assert!((max_imp[2] - 8.0).abs() < 1e-6, "MAX C={}", max_imp[2]);

        // MAX ranking: A(9.0) > C(8.0) > B(4.5) — both layer-critical tokens rank high
        assert!(max_imp[0] > max_imp[2]); // A > C
        assert!(max_imp[2] > max_imp[1]); // C > B

        // SUM aggregation (simulated)
        let sum_imp = simulate_sum_aggregation(
            &[layer0.clone(), layer1.clone()],
            n_heads,
            max_seq,
            cache_seq,
        );
        // SUM: A=9.0+0.1=9.1, B=4.5+4.5=9.0, C=0.1+8.0=8.1
        assert!((sum_imp[0] - 9.1).abs() < 1e-6, "SUM A={}", sum_imp[0]);
        assert!((sum_imp[1] - 9.0).abs() < 1e-6, "SUM B={}", sum_imp[1]);
        assert!((sum_imp[2] - 8.1).abs() < 1e-6, "SUM C={}", sum_imp[2]);

        // SUM ranking: A(9.1) > B(9.0) > C(8.1) — B (uniformly moderate) outranks C
        assert!(sum_imp[0] > sum_imp[1]); // A > B
        assert!(sum_imp[1] > sum_imp[2]); // B > C ← C (layer-1 critical) pushed down!

        // CONCLUSION: MAX preserves layer-critical tokens better.
        // If HH budget=2: MAX keeps {A, C}, SUM keeps {A, B}.
        // Token C (critical for layer 1) survives under MAX but not SUM.
    }

    #[test]
    fn test_max_preserves_single_layer_critical_token() {
        // Token only critical in 1 of 4 layers: score 50.0 in layer 2, 0.01 elsewhere.
        // MAX correctly gives it importance=50.0, ensuring it ranks high.
        let max_seq = 4;
        let cache_seq = 2;

        let mut acc = AttentionScoreAccumulator::new(max_seq, 1, 4, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        // 4 layers, token 0 is critical in layer 2 only
        for layer_idx in 0..4 {
            let mut s = vec![0.0f32; max_seq];
            if layer_idx == 2 {
                s[0] = 50.0; // critical
                s[1] = 0.01;
            } else {
                s[0] = 0.01;
                s[1] = 5.0; // moderately important in other layers
            }
            acc.accumulate_layer(&s, max_seq, cache_seq, 1, 0);
        }
        acc.end_step();

        let imp = acc.importance_scores();
        // MAX: token 0 = max(0.01, 0.01, 50.0, 0.01) = 50.0
        // MAX: token 1 = max(5.0, 5.0, 0.01, 5.0) = 5.0
        assert!((imp[0] - 50.0).abs() < 1e-6);
        assert!((imp[1] - 5.0).abs() < 1e-6);

        // Token 0 ranks higher despite being critical in only 1 layer
        assert!(imp[0] > imp[1]);
    }

    #[test]
    fn test_two_stage_aggregation_within_step_max_across_steps_sum() {
        // Verify the two-stage aggregation:
        //   Within step: MAX across layers
        //   Across steps: SUM of per-step MAX values
        let max_seq = 4;

        let mut acc = AttentionScoreAccumulator::new(max_seq, 1, 2, 0, 0.0);
        acc.set_active(true);

        // Step 1: layer0=[3.0, 1.0], layer1=[1.0, 5.0]
        acc.begin_step();
        acc.accumulate_layer(&[3.0, 1.0, 0.0, 0.0], max_seq, 2, 1, 0);
        acc.accumulate_layer(&[1.0, 5.0, 0.0, 0.0], max_seq, 2, 1, 0);
        acc.end_step();

        // step_max: token0=max(3,1)=3, token1=max(1,5)=5
        // importance after step 1: [3.0, 5.0]
        assert!((acc.importance_scores()[0] - 3.0).abs() < 1e-6);
        assert!((acc.importance_scores()[1] - 5.0).abs() < 1e-6);

        // Step 2: layer0=[4.0, 2.0], layer1=[2.0, 1.0]
        acc.begin_step();
        acc.accumulate_layer(&[4.0, 2.0, 0.0, 0.0], max_seq, 2, 1, 0);
        acc.accumulate_layer(&[2.0, 1.0, 0.0, 0.0], max_seq, 2, 1, 0);
        acc.end_step();

        // step_max: token0=max(4,2)=4, token1=max(2,1)=2
        // importance = step1 + step2: [3+4, 5+2] = [7.0, 7.0]
        assert!((acc.importance_scores()[0] - 7.0).abs() < 1e-6);
        assert!((acc.importance_scores()[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_post_softmax_score_total_equals_n_heads() {
        // When scores are valid post-softmax (each head sums to 1.0),
        // accumulate_layer sums across heads → per-token total across all tokens = n_heads.
        let max_seq = 4;
        let n_heads = 4;
        let cache_seq = 4;
        let stride = max_seq;

        // Create valid softmax distributions for each head
        let mut scores = vec![0.0f32; n_heads * stride];
        for h in 0..n_heads {
            let base = h * stride;
            // uniform distribution: each token gets 0.25
            for t in 0..cache_seq {
                scores[base + t] = 1.0 / cache_seq as f32;
            }
        }

        let mut acc = AttentionScoreAccumulator::new(max_seq, n_heads, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq, n_heads, 0);
        acc.end_step();

        let imp = acc.importance_scores();
        // Each token: sum of n_heads * 0.25 = n_heads * 0.25 = 1.0
        // Total across all tokens: n_heads * 1.0 = 4.0
        let total: f32 = imp[..cache_seq].iter().sum();
        assert!(
            (total - n_heads as f32).abs() < 1e-5,
            "Total importance {} != n_heads {}",
            total,
            n_heads
        );
    }

    // ── Experiment: HH meaninglessness proof ──

    /// Simulate Llama 3.2 1B attention patterns and dump score distribution.
    ///
    /// Calibrated to Round 15 observations: BOS≈3003, prompt avg≈3.3, gen avg≈33.
    /// Outputs CSV to `experiments/analysis/score_distribution.csv`.
    #[test]
    fn experiment_hh_proof_score_distribution() {
        use std::collections::HashSet;
        use std::io::Write;

        let max_seq = 2048usize;
        let n_heads_q = 32usize;
        let n_layers = 16usize;
        let prompt_len = 128usize;
        let cache_at_eviction = 1024usize;
        let decode_steps = cache_at_eviction - prompt_len; // 896

        // "Structural" prompt tokens (sentence boundaries, punctuation) — slightly
        // higher attention than average prompt tokens.
        let structural: HashSet<usize> = [5, 20, 45, 70, 100, 125].iter().copied().collect();

        let mut acc = AttentionScoreAccumulator::new(max_seq, n_heads_q, n_layers, 0, 0.0);
        acc.set_active(true);

        for step in 0..decode_steps {
            acc.begin_step();
            let cache_len = prompt_len + step + 1;

            for layer in 0..n_layers {
                let mut scores = vec![0.0f32; n_heads_q * max_seq];

                for h in 0..n_heads_q {
                    let off = h * max_seq;

                    // ── Unnormalized attention weights ──
                    // BOS: strong attention sink (varies slightly by head/layer)
                    scores[off] = 100.0 + (layer as f32) * 2.0 + (h as f32) * 0.5;

                    // Prompt tokens: very low attention
                    for t in 1..prompt_len {
                        scores[off + t] = if structural.contains(&t) {
                            // Structural tokens: 3x normal prompt attention
                            0.3 + 0.05 * ((h + layer) % 4) as f32
                        } else {
                            0.1 + 0.01 * ((h * 7 + t * 3) % 10) as f32 / 10.0
                        };
                    }

                    // Generated tokens: exponential recency bias
                    for t in prompt_len..cache_len {
                        let distance = (cache_len - 1 - t) as f32;
                        scores[off + t] = 0.5
                            + 8.0 * (-distance / 80.0).exp()
                            + 0.1 * ((h + t + layer) % 5) as f32 / 5.0;
                    }

                    // Normalize to probability distribution (sum=1)
                    let sum: f32 = scores[off..off + cache_len].iter().sum();
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for t in 0..cache_len {
                            scores[off + t] *= inv;
                        }
                    }
                }

                acc.accumulate_layer(&scores, max_seq, cache_len, n_heads_q, 0);
            }

            acc.end_step();
        }

        // ── Write CSV ──
        let importance = acc.importance_scores();
        let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let output_dir = project_root.join("experiments").join("analysis");
        std::fs::create_dir_all(&output_dir).ok();
        let csv_path = output_dir.join("score_distribution.csv");

        let mut f = std::fs::File::create(&csv_path).expect("create CSV");
        writeln!(f, "position,score,token_type").unwrap();
        for t in 0..cache_at_eviction {
            let token_type = if t == 0 {
                "bos"
            } else if structural.contains(&t) {
                "structural"
            } else if t < prompt_len {
                "prompt"
            } else {
                "generated"
            };
            writeln!(f, "{},{:.6},{}", t, importance[t], token_type).unwrap();
        }

        // ── Compute & print statistics ──
        let bos = importance[0];
        let prompt_scores: Vec<f32> = (1..prompt_len).map(|t| importance[t]).collect();
        let struct_scores: Vec<f32> = structural
            .iter()
            .filter(|&&t| t > 0 && t < prompt_len)
            .map(|&t| importance[t])
            .collect();
        let gen_scores: Vec<f32> = (prompt_len..cache_at_eviction)
            .map(|t| importance[t])
            .collect();

        let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let std_dev = |v: &[f32]| {
            let m = avg(v);
            (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };

        let prompt_avg = avg(&prompt_scores);
        let struct_avg = avg(&struct_scores);
        let gen_avg = avg(&gen_scores);
        let gen_std = std_dev(&gen_scores);

        // Non-BOS scores for overall distribution analysis
        let non_bos: Vec<f32> = (1..cache_at_eviction).map(|t| importance[t]).collect();
        let non_bos_avg = avg(&non_bos);
        let non_bos_std = std_dev(&non_bos);
        let cv = non_bos_std / non_bos_avg;

        // Tokens above 1σ and 2σ (excluding BOS)
        let above_1s = non_bos
            .iter()
            .filter(|&&s| s > non_bos_avg + non_bos_std)
            .count();
        let above_2s = non_bos
            .iter()
            .filter(|&&s| s > non_bos_avg + 2.0 * non_bos_std)
            .count();

        // Shannon entropy (normalized)
        let score_sum: f32 = non_bos.iter().sum();
        let entropy: f64 = non_bos
            .iter()
            .filter(|&&s| s > 0.0)
            .map(|&s| {
                let p = s as f64 / score_sum as f64;
                -p * p.ln()
            })
            .sum();
        let max_entropy = (non_bos.len() as f64).ln();
        let normalized_entropy = entropy / max_entropy;

        // Gini coefficient
        let mut sorted_non_bos = non_bos.clone();
        sorted_non_bos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_non_bos.len() as f64;
        let gini_sum: f64 = sorted_non_bos
            .iter()
            .enumerate()
            .map(|(i, &s)| (2.0 * (i as f64 + 1.0) - n - 1.0) * s as f64)
            .sum();
        let gini = gini_sum / (n * score_sum as f64);

        // H2O simulation: what would H2O select?
        let protected_prefix = 4usize;
        let target = 512usize;
        let keep_ratio = 0.5f32;
        let available = target.saturating_sub(protected_prefix);
        let hh_budget = (available as f32 * keep_ratio) as usize;
        let recent_budget = available - hh_budget;
        let recent_start = cache_at_eviction - recent_budget;
        let evictable_range = protected_prefix..recent_start;

        let mut token_scores: Vec<(usize, f32)> = evictable_range
            .clone()
            .map(|pos| (pos, importance[pos]))
            .collect();
        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let hh_selected: Vec<(usize, f32)> = token_scores.iter().take(hh_budget).cloned().collect();
        let evicted: Vec<(usize, f32)> = token_scores.iter().skip(hh_budget).cloned().collect();

        let hh_avg = avg(&hh_selected.iter().map(|x| x.1).collect::<Vec<_>>());
        let evicted_avg = avg(&evicted.iter().map(|x| x.1).collect::<Vec<_>>());

        // How many HH are prompt vs generated?
        let hh_prompt_count = hh_selected.iter().filter(|x| x.0 < prompt_len).count();
        let hh_gen_count = hh_selected.iter().filter(|x| x.0 >= prompt_len).count();

        // Random baseline: average score of 254 random evictable tokens
        // Use deterministic "random" = every other token
        let random_selected: Vec<f32> = evictable_range
            .clone()
            .step_by(2)
            .take(hh_budget)
            .map(|pos| importance[pos])
            .collect();
        let random_avg = avg(&random_selected);

        // Position-score correlation (Experiment 3)
        // Pearson correlation for generated tokens only
        let gen_positions: Vec<f64> = (prompt_len..cache_at_eviction).map(|t| t as f64).collect();
        let gen_values: Vec<f64> = gen_scores.iter().map(|&s| s as f64).collect();
        let gn = gen_positions.len() as f64;
        let pos_mean = gen_positions.iter().sum::<f64>() / gn;
        let val_mean = gen_values.iter().sum::<f64>() / gn;
        let mut cov = 0.0f64;
        let mut var_pos = 0.0f64;
        let mut var_val = 0.0f64;
        for i in 0..gen_positions.len() {
            let dp = gen_positions[i] - pos_mean;
            let dv = gen_values[i] - val_mean;
            cov += dp * dv;
            var_pos += dp * dp;
            var_val += dv * dv;
        }
        let pearson_r = cov / (var_pos.sqrt() * var_val.sqrt());

        // Spearman rank correlation
        let mut gen_rank_pairs: Vec<(usize, f64)> = gen_values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        gen_rank_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut score_ranks = vec![0.0f64; gen_rank_pairs.len()];
        for (rank, &(idx, _)) in gen_rank_pairs.iter().enumerate() {
            score_ranks[idx] = rank as f64;
        }
        let pos_ranks: Vec<f64> = (0..gen_positions.len()).map(|i| i as f64).collect();
        let rank_n = pos_ranks.len() as f64;
        let pos_rank_mean = pos_ranks.iter().sum::<f64>() / rank_n;
        let score_rank_mean = score_ranks.iter().sum::<f64>() / rank_n;
        let mut rank_cov = 0.0f64;
        let mut rank_var_pos = 0.0f64;
        let mut rank_var_val = 0.0f64;
        for i in 0..pos_ranks.len() {
            let dp = pos_ranks[i] - pos_rank_mean;
            let dv = score_ranks[i] - score_rank_mean;
            rank_cov += dp * dv;
            rank_var_pos += dp * dp;
            rank_var_val += dv * dv;
        }
        let spearman_rho = rank_cov / (rank_var_pos.sqrt() * rank_var_val.sqrt());

        println!("\n{}", "=".repeat(60));
        println!("  HH Proof Experiment: Score Distribution Analysis");
        println!("{}", "=".repeat(60));
        println!("\n[Distribution]");
        println!("  BOS score:       {:.1}", bos);
        println!(
            "  Prompt avg:      {:.3} (n={})",
            prompt_avg,
            prompt_scores.len()
        );
        println!(
            "  Structural avg:  {:.3} (n={})",
            struct_avg,
            struct_scores.len()
        );
        println!("  Generated avg:   {:.3} (n={})", gen_avg, gen_scores.len());
        println!("  Generated std:   {:.3}", gen_std);
        println!("  BOS/Prompt:      {:.0}x", bos / prompt_avg);
        println!("  BOS/Generated:   {:.0}x", bos / gen_avg);
        println!("\n[Non-BOS Statistics]");
        println!("  Mean:  {:.3}", non_bos_avg);
        println!("  Std:   {:.3}", non_bos_std);
        println!("  CV:    {:.3}", cv);
        println!(
            "  >1σ:   {} ({:.1}%)",
            above_1s,
            100.0 * above_1s as f32 / non_bos.len() as f32
        );
        println!(
            "  >2σ:   {} ({:.1}%)",
            above_2s,
            100.0 * above_2s as f32 / non_bos.len() as f32
        );
        println!("\n[Information Theory]");
        println!("  Shannon entropy:    {:.4}", entropy);
        println!("  Max entropy:        {:.4}", max_entropy);
        println!(
            "  Normalized entropy: {:.4} (1.0 = uniform)",
            normalized_entropy
        );
        println!("  Gini coefficient:   {:.4} (0.0 = equal)", gini);
        println!(
            "\n[H2O Simulation (prefix={}, target={}, kr={})]",
            protected_prefix, target, keep_ratio
        );
        println!("  HH budget:      {}", hh_budget);
        println!("  Recent budget:   {}", recent_budget);
        println!("  HH avg score:    {:.3}", hh_avg);
        println!("  Evicted avg:     {:.3}", evicted_avg);
        println!("  HH/Evicted:      {:.2}x", hh_avg / evicted_avg);
        println!("  Random avg:      {:.3}", random_avg);
        println!("  HH/Random:       {:.2}x", hh_avg / random_avg);
        println!(
            "  HH from prompt:  {} ({:.1}%)",
            hh_prompt_count,
            100.0 * hh_prompt_count as f32 / hh_budget as f32
        );
        println!(
            "  HH from gen:     {} ({:.1}%)",
            hh_gen_count,
            100.0 * hh_gen_count as f32 / hh_budget as f32
        );
        if let Some(top) = hh_selected.first() {
            println!("  HH top-1:        pos={} score={:.3}", top.0, top.1);
        }
        if let Some(bot) = hh_selected.last() {
            println!("  HH bottom-1:     pos={} score={:.3}", bot.0, bot.1);
        }
        if let Some(top_ev) = evicted.first() {
            println!("  Evicted top-1:   pos={} score={:.3}", top_ev.0, top_ev.1);
        }
        println!("\n[Position-Score Correlation (Generated only)]");
        println!("  Pearson r:   {:.4}", pearson_r);
        println!("  Spearman ρ:  {:.4}", spearman_rho);
        println!(
            "  Interpretation: {}",
            if pearson_r.abs() > 0.7 {
                "STRONG — scores ≈ recency → HH redundant with sliding"
            } else if pearson_r.abs() > 0.4 {
                "MODERATE — partial recency correlation"
            } else {
                "WEAK — scores independent of position"
            }
        );
        println!("\n  CSV: {}", csv_path.display());
    }

    /// Compare raw vs time-normalized scoring on the same attention patterns.
    /// Generates both CSVs and prints comparison statistics.
    #[test]
    fn experiment_time_normalized_comparison() {
        use std::collections::HashSet;
        use std::io::Write;

        let max_seq = 2048usize;
        let n_heads_q = 32usize;
        let n_layers = 16usize;
        let prompt_len = 128usize;
        let cache_at_eviction = 1024usize;
        let decode_steps = cache_at_eviction - prompt_len;

        let structural: HashSet<usize> = [5, 20, 45, 70, 100, 125].iter().copied().collect();

        // Run two accumulators in parallel: raw vs time-normalized
        let mut acc_raw = AttentionScoreAccumulator::new(max_seq, n_heads_q, n_layers, 0, 0.0);
        acc_raw.set_active(true);

        let mut acc_norm = AttentionScoreAccumulator::new(max_seq, n_heads_q, n_layers, 0, 0.0);
        acc_norm.set_active(true);
        acc_norm.set_time_normalize(true);

        for step in 0..decode_steps {
            acc_raw.begin_step();
            acc_norm.begin_step();
            let cache_len = prompt_len + step + 1;

            for layer in 0..n_layers {
                let mut scores = vec![0.0f32; n_heads_q * max_seq];

                for h in 0..n_heads_q {
                    let off = h * max_seq;
                    scores[off] = 100.0 + (layer as f32) * 2.0 + (h as f32) * 0.5;
                    for t in 1..prompt_len {
                        scores[off + t] = if structural.contains(&t) {
                            0.3 + 0.05 * ((h + layer) % 4) as f32
                        } else {
                            0.1 + 0.01 * ((h * 7 + t * 3) % 10) as f32 / 10.0
                        };
                    }
                    for t in prompt_len..cache_len {
                        let distance = (cache_len - 1 - t) as f32;
                        scores[off + t] = 0.5
                            + 8.0 * (-distance / 80.0).exp()
                            + 0.1 * ((h + t + layer) % 5) as f32 / 5.0;
                    }
                    let sum: f32 = scores[off..off + cache_len].iter().sum();
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for t in 0..cache_len {
                            scores[off + t] *= inv;
                        }
                    }
                }

                acc_raw.accumulate_layer(&scores, max_seq, cache_len, n_heads_q, 0);
                acc_norm.accumulate_layer(&scores, max_seq, cache_len, n_heads_q, 0);
            }

            acc_raw.end_step();
            acc_norm.end_step();
        }

        let raw_scores = acc_raw.importance_scores();
        let norm_scores = acc_norm.importance_scores();

        // Write normalized CSV
        let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let output_dir = project_root.join("experiments").join("analysis");
        std::fs::create_dir_all(&output_dir).ok();
        let csv_path = output_dir.join("score_distribution_normalized.csv");

        let mut f = std::fs::File::create(&csv_path).expect("create CSV");
        writeln!(f, "position,raw_score,norm_score,token_type").unwrap();
        for t in 0..cache_at_eviction {
            let token_type = if t == 0 {
                "bos"
            } else if structural.contains(&t) {
                "structural"
            } else if t < prompt_len {
                "prompt"
            } else {
                "generated"
            };
            writeln!(
                f,
                "{},{:.6},{:.6},{}",
                t, raw_scores[t], norm_scores[t], token_type
            )
            .unwrap();
        }

        // ── Statistics helper closures ──
        let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;

        let gen_raw: Vec<f32> = (prompt_len..cache_at_eviction)
            .map(|t| raw_scores[t])
            .collect();
        let gen_norm: Vec<f32> = (prompt_len..cache_at_eviction)
            .map(|t| norm_scores[t])
            .collect();

        // Pearson correlation for generated tokens
        let compute_pearson = |values: &[f32]| -> f64 {
            let positions: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
            let vals: Vec<f64> = values.iter().map(|&s| s as f64).collect();
            let n = positions.len() as f64;
            let pm = positions.iter().sum::<f64>() / n;
            let vm = vals.iter().sum::<f64>() / n;
            let mut cov = 0.0f64;
            let mut vp = 0.0f64;
            let mut vv = 0.0f64;
            for i in 0..positions.len() {
                let dp = positions[i] - pm;
                let dv = vals[i] - vm;
                cov += dp * dv;
                vp += dp * dp;
                vv += dv * dv;
            }
            let denom = (vp * vv).sqrt();
            if denom > 0.0 { cov / denom } else { 0.0 }
        };

        let raw_pearson = compute_pearson(&gen_raw);
        let norm_pearson = compute_pearson(&gen_norm);

        // H2O simulation for both
        let protected_prefix = 4usize;
        let target = 512usize;
        let keep_ratio = 0.5f32;
        let available = target - protected_prefix;
        let hh_budget = (available as f32 * keep_ratio) as usize;
        let recent_budget = available - hh_budget;
        let recent_start = cache_at_eviction - recent_budget;

        let simulate_h2o = |scores: &[f32]| -> (f32, f32, usize, usize) {
            let mut token_scores: Vec<(usize, f32)> = (protected_prefix..recent_start)
                .map(|pos| (pos, scores[pos]))
                .collect();
            token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let hh: Vec<f32> = token_scores.iter().take(hh_budget).map(|x| x.1).collect();
            let ev: Vec<f32> = token_scores.iter().skip(hh_budget).map(|x| x.1).collect();
            let hh_from_prompt = token_scores
                .iter()
                .take(hh_budget)
                .filter(|x| x.0 < prompt_len)
                .count();
            let hh_oldest_gen = token_scores
                .iter()
                .take(hh_budget)
                .filter(|x| x.0 >= prompt_len && x.0 < prompt_len + hh_budget)
                .count();
            (avg(&hh), avg(&ev), hh_from_prompt, hh_oldest_gen)
        };

        let (raw_hh_avg, raw_ev_avg, raw_hh_prompt, raw_hh_oldest) = simulate_h2o(raw_scores);
        let (norm_hh_avg, norm_ev_avg, norm_hh_prompt, norm_hh_oldest) = simulate_h2o(norm_scores);

        println!("\n{}", "=".repeat(60));
        println!("  Time-Normalized vs Raw Scoring Comparison");
        println!("{}", "=".repeat(60));
        println!("\n[Position-Score Correlation (Generated only)]");
        println!("  Raw  Pearson r:  {:.4}", raw_pearson);
        println!("  Norm Pearson r:  {:.4}", norm_pearson);
        println!(
            "  Improvement:     {:.4} → {:.4} (Δ={:+.4})",
            raw_pearson,
            norm_pearson,
            norm_pearson - raw_pearson
        );
        println!("\n[Score Distribution (Generated)]");
        println!(
            "  Raw  range: {:.3} - {:.3} (avg {:.3})",
            gen_raw.iter().cloned().reduce(f32::min).unwrap(),
            gen_raw.iter().cloned().reduce(f32::max).unwrap(),
            avg(&gen_raw)
        );
        println!(
            "  Norm range: {:.3} - {:.3} (avg {:.3})",
            gen_norm.iter().cloned().reduce(f32::min).unwrap(),
            gen_norm.iter().cloned().reduce(f32::max).unwrap(),
            avg(&gen_norm)
        );
        println!(
            "\n[H2O Simulation (prefix={}, target={}, kr={})]",
            protected_prefix, target, keep_ratio
        );
        println!(
            "  Raw:  HH avg={:.3}, Evicted avg={:.3}, ratio={:.2}x",
            raw_hh_avg,
            raw_ev_avg,
            raw_hh_avg / raw_ev_avg
        );
        println!(
            "  Norm: HH avg={:.3}, Evicted avg={:.3}, ratio={:.2}x",
            norm_hh_avg,
            norm_ev_avg,
            norm_hh_avg / norm_ev_avg
        );
        println!(
            "  Raw  HH composition: {} from prompt, {} oldest-gen (of {})",
            raw_hh_prompt, raw_hh_oldest, hh_budget
        );
        println!(
            "  Norm HH composition: {} from prompt, {} oldest-gen (of {})",
            norm_hh_prompt, norm_hh_oldest, hh_budget
        );
        println!("\n  CSV: {}", csv_path.display());

        // ── Assertions ──
        // Time normalization should substantially reduce the magnitude of correlation
        assert!(
            norm_pearson.abs() < raw_pearson.abs(),
            "Normalization should reduce position-score correlation: raw={:.4}, norm={:.4}",
            raw_pearson,
            norm_pearson
        );
    }

    // ── GPU score import tests ──

    #[test]
    fn test_import_gpu_scores_flat() {
        let mut acc = AttentionScoreAccumulator::new(8, 2, 4, 0, 0.0);
        acc.set_active(true);

        // Simulate GPU-accumulated scores
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let head = vec![];
        acc.import_gpu_scores(&flat, &head);

        let imp = acc.importance_scores();
        assert!((imp[0] - 1.0).abs() < 1e-6);
        assert!((imp[7] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_import_gpu_scores_gqa() {
        // 4 Q-heads, 2 KV-heads, max_seq=4
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 4, 2, 1, 0, 0.0);
        acc.set_active(true);

        let flat = vec![10.0, 20.0, 30.0, 40.0];
        // head layout: [n_kv_heads * max_seq_len] = [2 * 4] = 8
        let head = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        acc.import_gpu_scores(&flat, &head);

        assert!((acc.importance_scores()[0] - 10.0).abs() < 1e-6);
        assert!((acc.importance_scores()[3] - 40.0).abs() < 1e-6);

        let head_imp = acc.head_importance_scores().unwrap();
        assert!((head_imp[0] - 1.0).abs() < 1e-6);
        assert!((head_imp[7] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_import_gpu_scores_overwrites() {
        // import_gpu_scores should overwrite existing CPU scores
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        acc.set_active(true);

        // First accumulate via CPU path
        acc.begin_step();
        acc.accumulate_layer(&[100.0, 200.0, 300.0, 400.0], 4, 4, 1, 0);
        acc.end_step();

        assert!((acc.importance_scores()[0] - 100.0).abs() < 1e-6);

        // GPU import overwrites
        let flat = vec![1.0, 2.0, 3.0, 4.0];
        acc.import_gpu_scores(&flat, &[]);

        assert!((acc.importance_scores()[0] - 1.0).abs() < 1e-6);
        assert!((acc.importance_scores()[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_import_gpu_scores_with_time_normalize() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        acc.set_active(true);
        acc.set_time_normalize(true);

        // Simulate some steps to build step_count
        acc.begin_step();
        acc.accumulate_layer(&[1.0, 1.0, 1.0, 1.0], 4, 4, 1, 0);
        acc.end_step(); // step_count[0..4] = 1
        acc.begin_step();
        acc.accumulate_layer(&[1.0, 1.0, 0.0, 0.0], 4, 4, 1, 0);
        acc.end_step(); // step_count = [2, 2, 1, 1]

        // GPU import with known values
        let flat = vec![10.0, 20.0, 30.0, 40.0];
        acc.import_gpu_scores(&flat, &[]);

        // With time_normalize: importance[t] / step_count[t]
        let imp = acc.importance_scores();
        assert!((imp[0] - 5.0).abs() < 1e-6); // 10/2
        assert!((imp[1] - 10.0).abs() < 1e-6); // 20/2
        assert!((imp[2] - 30.0).abs() < 1e-6); // 30/1
        assert!((imp[3] - 40.0).abs() < 1e-6); // 40/1
    }

    // ── NEON vectorization correctness tests ──
    // These verify the NEON path (on aarch64) and scalar path (on x86) produce
    // identical results by using sizes that exercise both the 4-wide main loop
    // and the scalar tail.

    /// Helper: scalar reference implementation of accumulate_layer.
    fn accumulate_layer_scalar(
        step_importance: &mut [f32],
        scores: &[f32],
        stride: usize,
        len: usize,
        n_heads_q: usize,
        score_offset: usize,
    ) {
        for t in 0..len {
            let pos = score_offset + t;
            if pos >= step_importance.len() {
                break;
            }
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            step_importance[pos] = step_importance[pos].max(layer_score);
        }
    }

    /// Helper: scalar reference implementation of accumulate_layer_gqa.
    fn accumulate_layer_gqa_scalar(
        step_importance: &mut [f32],
        head_step_importance: &mut [f32],
        last_layer_head_attn: &mut [f32],
        scores: &[f32],
        stride: usize,
        len: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        score_offset: usize,
    ) {
        let n_rep = n_heads_q / n_kv_heads;
        let inv_rep = 1.0 / n_rep as f32;
        for t in 0..len {
            let pos = score_offset + t;
            if pos >= max_seq_len {
                break;
            }
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            step_importance[pos] = step_importance[pos].max(layer_score);

            for kv_h in 0..n_kv_heads {
                let mut group_score = 0.0f32;
                for r in 0..n_rep {
                    group_score += scores[(kv_h * n_rep + r) * stride + t];
                }
                group_score *= inv_rep;
                let idx = kv_h * max_seq_len + pos;
                head_step_importance[idx] = head_step_importance[idx].max(group_score);
                last_layer_head_attn[idx] = group_score;
            }
        }
    }

    #[test]
    fn test_accumulate_layer_vectorized_vs_scalar() {
        // 13 tokens: exercises 3 full NEON iterations (12) + 1 scalar tail
        let max_seq = 16;
        let cache_seq = 13;
        let n_heads_q = 8;
        let stride = max_seq;

        // Deterministic scores
        let mut scores = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            for t in 0..cache_seq {
                scores[h * stride + t] = ((h * 13 + t * 7 + 3) % 100) as f32 / 100.0;
            }
        }

        // Scalar reference
        let mut step_ref = vec![0.0f32; max_seq];
        accumulate_layer_scalar(&mut step_ref, &scores, stride, cache_seq, n_heads_q, 0);

        // Through the struct (uses NEON on aarch64, scalar on x86)
        let mut acc = AttentionScoreAccumulator::new(max_seq, n_heads_q, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq, n_heads_q, 0);

        for t in 0..cache_seq {
            assert!(
                (acc.step_importance[t] - step_ref[t]).abs() < 1e-5,
                "mismatch at t={}: got {}, expected {}",
                t,
                acc.step_importance[t],
                step_ref[t]
            );
        }
    }

    #[test]
    fn test_accumulate_layer_gqa_vectorized_vs_scalar() {
        // 11 tokens: 2 full NEON iterations (8) + 3 scalar tail
        let max_seq = 16;
        let cache_seq = 11;
        let n_heads_q = 32;
        let n_kv_heads = 8;
        let stride = max_seq;

        let mut scores = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            for t in 0..cache_seq {
                scores[h * stride + t] = ((h * 11 + t * 5 + 7) % 97) as f32 / 97.0;
            }
        }

        // Scalar reference
        let mut step_ref = vec![0.0f32; max_seq];
        let mut head_step_ref = vec![0.0f32; n_kv_heads * max_seq];
        let mut last_attn_ref = vec![0.0f32; n_kv_heads * max_seq];
        accumulate_layer_gqa_scalar(
            &mut step_ref,
            &mut head_step_ref,
            &mut last_attn_ref,
            &scores,
            stride,
            cache_seq,
            n_heads_q,
            n_kv_heads,
            max_seq,
            0,
        );

        // Through the struct
        let mut acc = AttentionScoreAccumulator::new_gqa(max_seq, n_heads_q, n_kv_heads, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer_gqa(&scores, stride, cache_seq, n_heads_q, n_kv_heads, 0);

        for t in 0..cache_seq {
            assert!(
                (acc.step_importance[t] - step_ref[t]).abs() < 1e-5,
                "flat mismatch at t={}: got {}, expected {}",
                t,
                acc.step_importance[t],
                step_ref[t]
            );
        }
        for kv_h in 0..n_kv_heads {
            for t in 0..cache_seq {
                let idx = kv_h * max_seq + t;
                assert!(
                    (acc.head_step_importance[idx] - head_step_ref[idx]).abs() < 1e-5,
                    "head_step mismatch kv_h={} t={}: got {}, expected {}",
                    kv_h,
                    t,
                    acc.head_step_importance[idx],
                    head_step_ref[idx]
                );
                assert!(
                    (acc.last_layer_head_attn[idx] - last_attn_ref[idx]).abs() < 1e-5,
                    "last_attn mismatch kv_h={} t={}: got {}, expected {}",
                    kv_h,
                    t,
                    acc.last_layer_head_attn[idx],
                    last_attn_ref[idx]
                );
            }
        }
    }

    #[test]
    fn test_end_step_vectorized_time_normalize() {
        // 7 tokens: 1 full NEON iteration (4) + 3 scalar tail, with time normalization
        let max_seq = 7;
        let n_heads_q = 4;
        let n_kv_heads = 2;

        let mut acc = AttentionScoreAccumulator::new_gqa(max_seq, n_heads_q, n_kv_heads, 1, 0, 0.0);
        acc.set_time_normalize(true);
        acc.set_active(true);

        // Step 1
        acc.begin_step();
        let mut scores1 = vec![0.0f32; n_heads_q * max_seq];
        for h in 0..n_heads_q {
            for t in 0..5 {
                scores1[h * max_seq + t] = (t + 1) as f32 * 0.1 + h as f32 * 0.01;
            }
        }
        acc.accumulate_layer_gqa(&scores1, max_seq, 5, n_heads_q, n_kv_heads, 0);
        acc.end_step();

        // Step 2
        acc.begin_step();
        let mut scores2 = vec![0.0f32; n_heads_q * max_seq];
        for h in 0..n_heads_q {
            for t in 0..7 {
                scores2[h * max_seq + t] = (7 - t) as f32 * 0.15 + h as f32 * 0.02;
            }
        }
        acc.accumulate_layer_gqa(&scores2, max_seq, 7, n_heads_q, n_kv_heads, 0);
        acc.end_step();

        // Verify: positions 0..5 have step_count=2, positions 5..7 have step_count=1
        assert_eq!(acc.step_count[0], 2);
        assert_eq!(acc.step_count[4], 2);
        assert_eq!(acc.step_count[5], 1);
        assert_eq!(acc.step_count[6], 1);

        // Time-normalized scores should be importance / step_count
        let imp = acc.importance_scores();
        let raw = acc.raw_importance_scores();
        for t in 0..max_seq {
            let count = acc.step_count[t].max(1) as f32;
            let expected = raw[t] / count;
            assert!(
                (imp[t] - expected).abs() < 1e-5,
                "time_normalize mismatch at t={}: got {}, expected {}",
                t,
                imp[t],
                expected,
            );
        }

        // Head importance should be sum of head_step from both steps
        let head_imp = acc.head_importance_scores().unwrap();
        for i in 0..head_imp.len() {
            assert!(head_imp[i].is_finite(), "head_imp[{}] not finite", i);
        }
    }

    #[test]
    fn test_accumulate_layer_exact_multiple_of_4() {
        // 8 tokens: exactly 2 NEON iterations, no scalar tail
        let max_seq = 8;
        let cache_seq = 8;
        let n_heads_q = 4;
        let stride = max_seq;

        let mut scores = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            for t in 0..cache_seq {
                scores[h * stride + t] = (h as f32 + 1.0) * (t as f32 + 1.0) * 0.1;
            }
        }

        let mut step_ref = vec![0.0f32; max_seq];
        accumulate_layer_scalar(&mut step_ref, &scores, stride, cache_seq, n_heads_q, 0);

        let mut acc = AttentionScoreAccumulator::new(max_seq, n_heads_q, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq, n_heads_q, 0);

        for t in 0..cache_seq {
            assert!(
                (acc.step_importance[t] - step_ref[t]).abs() < 1e-5,
                "t={}: got {}, expected {}",
                t,
                acc.step_importance[t],
                step_ref[t]
            );
        }
    }

    #[test]
    fn test_accumulate_layer_fewer_than_4_tokens() {
        // 3 tokens: no NEON iterations at all, pure scalar tail
        let max_seq = 8;
        let cache_seq = 3;
        let n_heads_q = 2;
        let stride = max_seq;

        let scores = vec![
            1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, // head 0
            4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, // head 1
        ];

        let mut acc = AttentionScoreAccumulator::new(max_seq, n_heads_q, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq, n_heads_q, 0);

        // sum: [5.0, 7.0, 9.0]
        assert!((acc.step_importance[0] - 5.0).abs() < 1e-6);
        assert!((acc.step_importance[1] - 7.0).abs() < 1e-6);
        assert!((acc.step_importance[2] - 9.0).abs() < 1e-6);
    }

    /// Verify that NaN scores in accumulate_layer_gqa are handled gracefully.
    /// NaN in attention scores (from softmax on NaN Q*K^T) should not poison
    /// head_step_importance or last_layer_head_attn — they should remain at
    /// their prior values (or 0 if this is the first layer).
    #[test]
    fn test_accumulate_gqa_nan_scores_handled() {
        let max_seq = 8;
        let n_heads_q = 4;
        let n_kv_heads = 2;
        let mut acc = AttentionScoreAccumulator::new_gqa(max_seq, n_heads_q, n_kv_heads, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        let stride = max_seq;
        let cache_seq_len = 4;

        // Layer 0: valid scores (softmax-like, sum to ~1.0 per head)
        let mut scores_l0 = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            scores_l0[h * stride] = 0.4;
            scores_l0[h * stride + 1] = 0.3;
            scores_l0[h * stride + 2] = 0.2;
            scores_l0[h * stride + 3] = 0.1;
        }
        acc.accumulate_layer_gqa(&scores_l0, stride, cache_seq_len, n_heads_q, n_kv_heads, 0);

        // Verify layer 0 accumulated correctly
        assert!(
            acc.step_importance[0] > 0.0,
            "flat step_importance[0] should be > 0 after L0"
        );
        let idx0 = 0usize; // kv_h=0, t=0 → 0 * max_seq = 0
        assert!(
            acc.head_step_importance[idx0] > 0.0,
            "head_step_importance[0] should be > 0 after L0"
        );
        assert!(
            acc.last_layer_head_attn[idx0] > 0.0,
            "last_layer_head_attn[0] should be > 0 after L0"
        );

        // Layer 1: all NaN scores (simulates NaN cascade from Q*K^T)
        let scores_l1 = vec![f32::NAN; n_heads_q * stride];
        acc.accumulate_layer_gqa(&scores_l1, stride, cache_seq_len, n_heads_q, n_kv_heads, 0);

        // After NaN layer: flat step_importance should retain L0 values (max(0.4*4, NaN) = 0.4*4)
        // f32::max(valid, NaN) = valid (IEEE 754)
        assert!(
            acc.step_importance[0] > 0.0,
            "flat step_importance[0] must survive NaN layer: got {}",
            acc.step_importance[0]
        );
        // head_step_importance should retain L0 values (max(valid, NaN) = valid)
        assert!(
            acc.head_step_importance[idx0] > 0.0,
            "head_step_importance[0] must survive NaN layer: got {}",
            acc.head_step_importance[idx0]
        );
        // last_layer_head_attn: NaN guard → 0 (overwritten, not MAX)
        assert_eq!(
            acc.last_layer_head_attn[idx0], 0.0,
            "last_layer_head_attn should be 0 after NaN layer (NaN guard)"
        );

        // end_step should flush correctly
        acc.end_step();
        assert!(
            acc.importance[0] > 0.0,
            "cumulative importance[0] should be > 0 after end_step"
        );
        let hi = acc.head_importance_scores().unwrap();
        assert!(
            hi[idx0] > 0.0,
            "cumulative head_importance[0] should be > 0 after end_step"
        );
    }
}
