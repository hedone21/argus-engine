/// Token lifetime record for Gantt chart visualization.
pub struct TokenLifetime {
    /// Decode step when this token entered the cache (0 = prompt token).
    pub birth_step: usize,
    /// Decode step when evicted (None = still alive at end of generation).
    pub death_step: Option<usize>,
    /// Importance score at eviction time (0.0 if still alive).
    pub importance_at_death: f32,
    /// Whether this is a protected prefix token.
    pub is_prefix: bool,
}

/// H2O attention score snapshot tracker.
///
/// Records per-step importance score snapshots and eviction events
/// for post-hoc visualization (heatmaps, partition evolution, etc.).
#[derive(Default)]
pub struct ScoreTracker {
    snapshots: Vec<ScoreSnapshot>,
    evictions: Vec<EvictionEvent>,
    lifetimes: Vec<TokenLifetime>,
    snapshot_interval: usize,
    track_per_head: bool,
}

/// A snapshot of attention importance scores at a single decode step.
pub struct ScoreSnapshot {
    pub step: usize,
    pub token_id: u32,
    pub cache_len: usize,
    /// Per-position importance scores `[cache_len]`.
    pub importance: Vec<f32>,
    /// Optional per-KV-head importance `[n_kv_heads][cache_len]` (row-major).
    pub head_importance: Option<Vec<f32>>,
    /// Number of KV heads (for interpreting head_importance layout).
    pub n_kv_heads: usize,
    /// Maps physical cache position → token birth step (decode step when token entered cache).
    /// `position_map[i]` = the decode step at which the token at cache position `i` was generated.
    /// Prompt tokens all have birth_step = 0.
    pub position_map: Option<Vec<usize>>,
}

/// An eviction event captured by the profiler.
pub struct EvictionEvent {
    pub step: usize,
    pub policy: String,
    pub before_len: usize,
    pub after_len: usize,
    pub evicted_count: usize,
    pub partition: PartitionInfo,
    /// Physical cache positions that were evicted (empty if unavailable).
    pub evicted_indices: Vec<usize>,
    /// Importance scores snapshot taken immediately BEFORE eviction.
    /// `pre_eviction_scores[i]` = importance at cache position `i` (len = before_len).
    pub pre_eviction_scores: Vec<f32>,
}

/// Partition boundaries at the time of eviction.
pub struct PartitionInfo {
    /// Protected prefix boundary: `[0..prefix_end)`.
    pub prefix_end: usize,
    /// Number of heavy hitter tokens selected.
    pub hh_count: usize,
    /// Start of the recent window: `[recent_start..)`.
    pub recent_start: usize,
}

impl ScoreTracker {
    pub fn new(snapshot_interval: usize, track_per_head: bool) -> Self {
        Self {
            snapshots: Vec::new(),
            evictions: Vec::new(),
            lifetimes: Vec::new(),
            snapshot_interval: snapshot_interval.max(1),
            track_per_head,
        }
    }

    pub fn snapshot_interval(&self) -> usize {
        self.snapshot_interval
    }

    pub fn track_per_head(&self) -> bool {
        self.track_per_head
    }

    pub fn snapshots(&self) -> &[ScoreSnapshot] {
        &self.snapshots
    }

    pub fn evictions(&self) -> &[EvictionEvent] {
        &self.evictions
    }

    pub fn lifetimes(&self) -> &[TokenLifetime] {
        &self.lifetimes
    }

    /// Register tokens that entered the cache (call during prefill and each decode step).
    pub fn record_token_births(
        &mut self,
        birth_step: usize,
        count: usize,
        protected_prefix: usize,
    ) {
        let current_len = self.lifetimes.len();
        for i in 0..count {
            self.lifetimes.push(TokenLifetime {
                birth_step,
                death_step: None,
                importance_at_death: 0.0,
                is_prefix: current_len + i < protected_prefix,
            });
        }
    }

    /// Record token deaths from an eviction event.
    ///
    /// `position_birth_map[pos]` = birth_step of the token at cache position `pos`.
    /// `evicted_indices` = physical positions that were evicted.
    /// `pre_scores[pos]` = importance at cache position `pos` before eviction.
    pub fn record_token_deaths(
        &mut self,
        step: usize,
        evicted_indices: &[usize],
        position_birth_map: &[usize],
        pre_scores: &[f32],
    ) {
        for &pos in evicted_indices {
            let birth = position_birth_map.get(pos).copied().unwrap_or(0);
            let importance = pre_scores.get(pos).copied().unwrap_or(0.0);
            // Find the lifetime entry by birth_step match
            // (multiple tokens can share birth_step=0 for prompt tokens, so find the first alive one)
            for lt in self.lifetimes.iter_mut() {
                if lt.birth_step == birth && lt.death_step.is_none() && !lt.is_prefix {
                    lt.death_step = Some(step);
                    lt.importance_at_death = importance;
                    break;
                }
            }
        }
    }

    /// Take a snapshot of current importance scores.
    ///
    /// Only records if `step % snapshot_interval == 0`.
    /// Reads from `importance[..cache_len]` and optionally `head_importance`.
    #[allow(clippy::too_many_arguments)]
    pub fn take_snapshot(
        &mut self,
        step: usize,
        token_id: u32,
        cache_len: usize,
        importance: &[f32],
        head_importance: Option<&[f32]>,
        n_kv_heads: usize,
        position_map: Option<&[usize]>,
    ) {
        if !step.is_multiple_of(self.snapshot_interval) {
            return;
        }

        let len = cache_len.min(importance.len());
        let imp = importance[..len].to_vec();

        let head_imp = if self.track_per_head {
            head_importance.map(|hi| {
                let head_len = n_kv_heads * cache_len;
                hi[..head_len.min(hi.len())].to_vec()
            })
        } else {
            None
        };

        let pos_map = position_map.map(|pm| pm[..cache_len.min(pm.len())].to_vec());

        self.snapshots.push(ScoreSnapshot {
            step,
            token_id,
            cache_len,
            importance: imp,
            head_importance: head_imp,
            n_kv_heads,
            position_map: pos_map,
        });
    }

    pub fn record_eviction(&mut self, event: EvictionEvent) {
        self.evictions.push(event);
    }

    pub fn to_json(&self) -> serde_json::Value {
        let snapshots: Vec<serde_json::Value> = self
            .snapshots
            .iter()
            .map(|s| {
                let mut obj = serde_json::json!({
                    "step": s.step,
                    "token_id": s.token_id,
                    "cache_len": s.cache_len,
                    "importance": s.importance,
                });
                if let Some(ref hi) = s.head_importance {
                    obj["head_importance"] = serde_json::json!(hi);
                    obj["n_kv_heads"] = serde_json::json!(s.n_kv_heads);
                }
                if let Some(ref pm) = s.position_map {
                    obj["position_map"] = serde_json::json!(pm);
                }
                obj
            })
            .collect();

        let evictions: Vec<serde_json::Value> = self
            .evictions
            .iter()
            .map(|e| {
                serde_json::json!({
                    "step": e.step,
                    "policy": e.policy,
                    "before_len": e.before_len,
                    "after_len": e.after_len,
                    "evicted_count": e.evicted_count,
                    "partition": {
                        "prefix_end": e.partition.prefix_end,
                        "hh_count": e.partition.hh_count,
                        "recent_start": e.partition.recent_start,
                    },
                    "evicted_indices": e.evicted_indices,
                    "pre_eviction_scores": e.pre_eviction_scores,
                })
            })
            .collect();

        let lifetimes: Vec<serde_json::Value> = self
            .lifetimes
            .iter()
            .map(|lt| {
                serde_json::json!({
                    "birth_step": lt.birth_step,
                    "death_step": lt.death_step,
                    "importance_at_death": (lt.importance_at_death * 10000.0).round() / 10000.0,
                    "is_prefix": lt.is_prefix,
                })
            })
            .collect();

        serde_json::json!({
            "snapshot_interval": self.snapshot_interval,
            "snapshot_count": self.snapshots.len(),
            "eviction_count": self.evictions.len(),
            "snapshots": snapshots,
            "evictions": evictions,
            "lifetimes": lifetimes,
        })
    }
}

/// Reconstruct H2O evicted indices from pre-eviction state (profiling utility).
///
/// Mirrors the H2O selection logic: prefix is protected, recent window is protected,
/// evictable tokens (prefix..recent_start) are ranked by importance — bottom ones are evicted.
pub fn compute_h2o_evicted_indices(
    before_len: usize,
    target_len: usize,
    protected_prefix: usize,
    keep_ratio: f32,
    importance: &[f32],
) -> Vec<usize> {
    let keep = target_len.max(protected_prefix + 2);
    if before_len <= keep {
        return Vec::new();
    }
    let available = keep.saturating_sub(protected_prefix);
    let hh_budget = (available as f32 * keep_ratio) as usize;
    let recent_budget = available - hh_budget;
    let actual_recent = recent_budget.min(before_len - protected_prefix);
    let recent_start = before_len
        .saturating_sub(actual_recent)
        .max(protected_prefix);

    let mut token_scores: Vec<(usize, f32)> = (protected_prefix..recent_start)
        .map(|pos| (pos, importance.get(pos).copied().unwrap_or(0.0)))
        .collect();
    token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Evicted = those NOT selected as heavy hitters
    let mut evicted: Vec<usize> = token_scores
        .iter()
        .skip(hh_budget)
        .map(|(pos, _)| *pos)
        .collect();
    evicted.sort();
    evicted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_tracker_default() {
        let t = ScoreTracker::default();
        assert_eq!(t.snapshot_interval(), 0);
        assert!(!t.track_per_head());
        assert!(t.snapshots().is_empty());
        assert!(t.evictions().is_empty());
    }

    #[test]
    fn test_score_tracker_new() {
        let t = ScoreTracker::new(5, true);
        assert_eq!(t.snapshot_interval(), 5);
        assert!(t.track_per_head());
    }

    #[test]
    fn test_snapshot_interval_min_one() {
        let t = ScoreTracker::new(0, false);
        assert_eq!(t.snapshot_interval(), 1);
    }

    #[test]
    fn test_take_snapshot_basic() {
        let mut t = ScoreTracker::new(1, false);
        let scores = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        t.take_snapshot(0, 42, 5, &scores, None, 0, None);

        assert_eq!(t.snapshots().len(), 1);
        let s = &t.snapshots()[0];
        assert_eq!(s.step, 0);
        assert_eq!(s.token_id, 42);
        assert_eq!(s.cache_len, 5);
        assert_eq!(s.importance.len(), 5);
        assert!((s.importance[0] - 0.5).abs() < 1e-6);
        assert!(s.head_importance.is_none());
    }

    #[test]
    fn test_take_snapshot_interval_skip() {
        let mut t = ScoreTracker::new(3, false);
        let scores = vec![0.1; 10];

        for step in 0..10 {
            t.take_snapshot(step, 0, 10, &scores, None, 0, None);
        }

        // Only steps 0, 3, 6, 9 should be captured
        assert_eq!(t.snapshots().len(), 4);
        assert_eq!(t.snapshots()[0].step, 0);
        assert_eq!(t.snapshots()[1].step, 3);
        assert_eq!(t.snapshots()[2].step, 6);
        assert_eq!(t.snapshots()[3].step, 9);
    }

    #[test]
    fn test_take_snapshot_clones_data() {
        let mut t = ScoreTracker::new(1, false);
        let mut scores = vec![1.0, 2.0, 3.0];
        t.take_snapshot(0, 0, 3, &scores, None, 0, None);

        // Modify original — snapshot should be unchanged
        scores[0] = 999.0;
        assert!((t.snapshots()[0].importance[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_take_snapshot_cache_len_bounds() {
        let mut t = ScoreTracker::new(1, false);
        // cache_len larger than scores buffer — should clamp
        let scores = vec![0.1, 0.2, 0.3];
        t.take_snapshot(0, 0, 10, &scores, None, 0, None);

        assert_eq!(t.snapshots()[0].importance.len(), 3);
    }

    #[test]
    fn test_take_snapshot_with_head_importance() {
        let mut t = ScoreTracker::new(1, true);
        let scores = vec![0.5, 0.3, 0.2];
        // 2 KV heads * 3 cache_len = 6 values
        let head_scores = vec![0.6, 0.2, 0.2, 0.4, 0.4, 0.2];
        t.take_snapshot(0, 42, 3, &scores, Some(&head_scores), 2, None);

        assert_eq!(t.snapshots().len(), 1);
        let hi = t.snapshots()[0].head_importance.as_ref().unwrap();
        assert_eq!(hi.len(), 6);
        assert!((hi[0] - 0.6).abs() < 1e-6);
        assert_eq!(t.snapshots()[0].n_kv_heads, 2);
    }

    #[test]
    fn test_take_snapshot_head_importance_not_tracked() {
        let mut t = ScoreTracker::new(1, false); // track_per_head = false
        let scores = vec![0.5, 0.3];
        let head_scores = vec![0.6, 0.2, 0.4, 0.4];
        t.take_snapshot(0, 0, 2, &scores, Some(&head_scores), 2, None);

        // head_importance should be None because track_per_head is false
        assert!(t.snapshots()[0].head_importance.is_none());
    }

    #[test]
    fn test_record_eviction() {
        let mut t = ScoreTracker::new(1, false);
        t.record_eviction(EvictionEvent {
            step: 50,
            policy: "h2o".to_string(),
            before_len: 256,
            after_len: 200,
            evicted_count: 56,
            partition: PartitionInfo {
                prefix_end: 4,
                hh_count: 98,
                recent_start: 102,
            },
            evicted_indices: vec![],
            pre_eviction_scores: vec![],
        });

        assert_eq!(t.evictions().len(), 1);
        let e = &t.evictions()[0];
        assert_eq!(e.step, 50);
        assert_eq!(e.policy, "h2o");
        assert_eq!(e.before_len, 256);
        assert_eq!(e.after_len, 200);
        assert_eq!(e.evicted_count, 56);
        assert_eq!(e.partition.prefix_end, 4);
        assert_eq!(e.partition.hh_count, 98);
        assert_eq!(e.partition.recent_start, 102);
    }

    #[test]
    fn test_record_multiple_evictions() {
        let mut t = ScoreTracker::new(1, false);
        for i in 0..3 {
            t.record_eviction(EvictionEvent {
                step: i * 100,
                policy: "h2o".to_string(),
                before_len: 256 - i * 20,
                after_len: 200 - i * 20,
                evicted_count: 56,
                partition: PartitionInfo {
                    prefix_end: 4,
                    hh_count: 98 - i * 10,
                    recent_start: 102 - i * 10,
                },
                evicted_indices: vec![],
                pre_eviction_scores: vec![],
            });
        }
        assert_eq!(t.evictions().len(), 3);
        assert_eq!(t.evictions()[2].step, 200);
    }

    #[test]
    fn test_to_json_empty() {
        let t = ScoreTracker::new(1, false);
        let json = t.to_json();

        assert_eq!(json["snapshot_interval"], 1);
        assert_eq!(json["snapshot_count"], 0);
        assert_eq!(json["eviction_count"], 0);
        assert!(json["snapshots"].as_array().unwrap().is_empty());
        assert!(json["evictions"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_to_json_with_snapshots() {
        let mut t = ScoreTracker::new(1, false);
        let scores = vec![0.8, 0.15, 0.05];
        t.take_snapshot(0, 100, 3, &scores, None, 0, None);

        let json = t.to_json();
        assert_eq!(json["snapshot_count"], 1);

        let snap = &json["snapshots"][0];
        assert_eq!(snap["step"], 0);
        assert_eq!(snap["token_id"], 100);
        assert_eq!(snap["cache_len"], 3);

        let imp = snap["importance"].as_array().unwrap();
        assert_eq!(imp.len(), 3);
        assert!((imp[0].as_f64().unwrap() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_to_json_with_head_importance() {
        let mut t = ScoreTracker::new(1, true);
        let scores = vec![0.5, 0.5];
        let head_scores = vec![0.6, 0.4, 0.3, 0.7];
        t.take_snapshot(0, 10, 2, &scores, Some(&head_scores), 2, None);

        let json = t.to_json();
        let snap = &json["snapshots"][0];
        assert_eq!(snap["n_kv_heads"], 2);

        let hi = snap["head_importance"].as_array().unwrap();
        assert_eq!(hi.len(), 4);
    }

    #[test]
    fn test_to_json_with_evictions() {
        let mut t = ScoreTracker::new(1, false);
        t.record_eviction(EvictionEvent {
            step: 42,
            policy: "d2o".to_string(),
            before_len: 512,
            after_len: 384,
            evicted_count: 128,
            partition: PartitionInfo {
                prefix_end: 8,
                hh_count: 188,
                recent_start: 196,
            },
            evicted_indices: vec![10, 20, 30],
            pre_eviction_scores: vec![0.1; 512],
        });

        let json = t.to_json();
        assert_eq!(json["eviction_count"], 1);

        let ev = &json["evictions"][0];
        assert_eq!(ev["step"], 42);
        assert_eq!(ev["policy"], "d2o");
        assert_eq!(ev["before_len"], 512);
        assert_eq!(ev["after_len"], 384);
        assert_eq!(ev["evicted_count"], 128);
        assert_eq!(ev["partition"]["prefix_end"], 8);
        assert_eq!(ev["partition"]["hh_count"], 188);
        assert_eq!(ev["partition"]["recent_start"], 196);
    }

    #[test]
    fn test_to_json_combined() {
        let mut t = ScoreTracker::new(2, true);
        let scores = vec![0.5, 0.3, 0.2];
        let head_scores = vec![0.6, 0.2, 0.2, 0.4, 0.4, 0.2];

        // Steps 0,1,2,3,4 — interval 2 captures 0, 2, 4
        for step in 0..5 {
            t.take_snapshot(step, step as u32, 3, &scores, Some(&head_scores), 2, None);
        }
        t.record_eviction(EvictionEvent {
            step: 3,
            policy: "h2o".to_string(),
            before_len: 3,
            after_len: 2,
            evicted_count: 1,
            partition: PartitionInfo {
                prefix_end: 1,
                hh_count: 1,
                recent_start: 2,
            },
            evicted_indices: vec![],
            pre_eviction_scores: vec![],
        });

        let json = t.to_json();
        assert_eq!(json["snapshot_count"], 3); // steps 0, 2, 4
        assert_eq!(json["eviction_count"], 1);
        assert_eq!(json["snapshot_interval"], 2);
    }

    // ── New tests for Improvements 1, 2, 4 ──

    #[test]
    fn test_eviction_event_with_indices_and_scores() {
        let mut t = ScoreTracker::new(1, false);
        let pre_scores = vec![0.8, 0.01, 0.05, 0.7, 0.02, 0.6];
        t.record_eviction(EvictionEvent {
            step: 10,
            policy: "h2o".to_string(),
            before_len: 6,
            after_len: 4,
            evicted_count: 2,
            partition: PartitionInfo {
                prefix_end: 1,
                hh_count: 2,
                recent_start: 4,
            },
            evicted_indices: vec![1, 4],
            pre_eviction_scores: pre_scores.clone(),
        });

        let e = &t.evictions()[0];
        assert_eq!(e.evicted_indices, vec![1, 4]);
        assert_eq!(e.pre_eviction_scores.len(), 6);

        // Verify JSON includes new fields
        let json = t.to_json();
        let ev = &json["evictions"][0];
        let indices = ev["evicted_indices"].as_array().unwrap();
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 1);
        assert_eq!(indices[1], 4);
        assert!(ev["pre_eviction_scores"].as_array().unwrap().len() == 6);
    }

    #[test]
    fn test_compute_h2o_evicted_indices_basic() {
        // 10 tokens: prefix=2, target=6, keep_ratio=0.5
        // available = 6 - 2 = 4, hh_budget = 2, recent_budget = 2
        // recent_start = 10 - 2 = 8
        // Evictable range: [2..8), 6 tokens ranked by importance
        // Top 2 (HH) survive, bottom 4 evicted
        let importance = vec![0.0, 0.0, 0.3, 0.1, 0.5, 0.2, 0.4, 0.05, 0.0, 0.0];
        let evicted = compute_h2o_evicted_indices(10, 6, 2, 0.5, &importance);

        // Evictable: pos 2(0.3), 3(0.1), 4(0.5), 5(0.2), 6(0.4), 7(0.05)
        // Top 2 by importance: pos 4(0.5), 6(0.4) → kept as HH
        // Evicted: pos 2, 3, 5, 7
        assert_eq!(evicted, vec![2, 3, 5, 7]);
    }

    #[test]
    fn test_compute_h2o_evicted_indices_no_eviction_needed() {
        let importance = vec![0.5; 5];
        let evicted = compute_h2o_evicted_indices(5, 10, 2, 0.5, &importance);
        assert!(evicted.is_empty()); // before_len <= keep
    }

    #[test]
    fn test_snapshot_with_position_map() {
        let mut t = ScoreTracker::new(1, false);
        let scores = vec![0.5, 0.3, 0.2];
        let pos_map = vec![0, 5, 10]; // token born at steps 0, 5, 10
        t.take_snapshot(0, 42, 3, &scores, None, 0, Some(&pos_map));

        let snap = &t.snapshots()[0];
        assert!(snap.position_map.is_some());
        assert_eq!(snap.position_map.as_ref().unwrap(), &vec![0, 5, 10]);

        // Verify JSON includes position_map
        let json = t.to_json();
        let pm = json["snapshots"][0]["position_map"].as_array().unwrap();
        assert_eq!(pm.len(), 3);
        assert_eq!(pm[0], 0);
        assert_eq!(pm[1], 5);
        assert_eq!(pm[2], 10);
    }

    #[test]
    fn test_token_lifetime_tracking() {
        let mut t = ScoreTracker::new(1, false);

        // Register 5 prompt tokens (birth=0), prefix=2
        t.record_token_births(0, 5, 2);
        assert_eq!(t.lifetimes().len(), 5);
        assert!(t.lifetimes()[0].is_prefix);
        assert!(t.lifetimes()[1].is_prefix);
        assert!(!t.lifetimes()[2].is_prefix);

        // Register 3 decode tokens
        t.record_token_births(1, 1, 2); // token at step 1
        t.record_token_births(2, 1, 2); // token at step 2
        t.record_token_births(3, 1, 2); // token at step 3
        assert_eq!(t.lifetimes().len(), 8);

        // Evict positions 2 and 3 (non-prefix) at step 5
        let birth_map = vec![0, 0, 0, 0, 0, 1, 2, 3]; // 5 prompt + 3 decode
        let pre_scores = vec![0.8, 0.7, 0.01, 0.02, 0.5, 0.3, 0.4, 0.6];
        t.record_token_deaths(5, &[2, 3], &birth_map, &pre_scores);

        // Token at position 2 (birth=0, non-prefix) should be dead
        let dead: Vec<_> = t
            .lifetimes()
            .iter()
            .filter(|lt| lt.death_step.is_some())
            .collect();
        assert_eq!(dead.len(), 2);
        assert_eq!(dead[0].death_step, Some(5));
        assert!((dead[0].importance_at_death - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_token_lifetime_json() {
        let mut t = ScoreTracker::new(1, false);
        t.record_token_births(0, 2, 1); // 2 tokens, prefix=1
        t.record_token_births(5, 1, 1); // 1 decode token

        let json = t.to_json();
        let lifetimes = json["lifetimes"].as_array().unwrap();
        assert_eq!(lifetimes.len(), 3);
        assert!(lifetimes[0]["is_prefix"].as_bool().unwrap());
        assert!(!lifetimes[1]["is_prefix"].as_bool().unwrap());
        assert_eq!(lifetimes[2]["birth_step"], 5);
        assert!(lifetimes[2]["death_step"].is_null());
    }

    #[test]
    fn test_prefix_tokens_never_die() {
        let mut t = ScoreTracker::new(1, false);
        t.record_token_births(0, 4, 2); // 4 tokens, 2 prefix

        // H2O never evicts prefix positions, so evicted_indices never contains 0 or 1.
        // Only non-prefix positions (2, 3) can be evicted.
        let birth_map = vec![0, 0, 0, 0];
        let pre_scores = vec![0.1, 0.2, 0.3, 0.4];
        t.record_token_deaths(10, &[2, 3], &birth_map, &pre_scores);

        // Positions 2, 3 are non-prefix → should be dead
        let dead_count = t
            .lifetimes()
            .iter()
            .filter(|lt| lt.death_step.is_some())
            .count();
        assert_eq!(dead_count, 2);
        // Prefix tokens remain alive
        assert!(t.lifetimes()[0].death_step.is_none());
        assert!(t.lifetimes()[1].death_step.is_none());
    }
}
