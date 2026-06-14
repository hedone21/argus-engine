//! Inference profiling framework.
//!
//! Provides modular, zero-overhead-when-disabled profiling for LLM inference.
//! Each probe collects data independently and exports to a unified JSON file.
//!
//! # Usage
//! ```ignore
//! let profiler = InferenceProfiler::new(ProfileConfig { .. });
//! // decode loop:
//! profiler.on_step_begin(step);
//! // ... forward pass ...
//! profiler.on_step_end(step, token_id, &score_acc, forward_us, sample_us, total_us, cache_len);
//! // ... after eviction ...
//! profiler.on_eviction(event);
//! // end:
//! profiler.export_json(&metadata)?;
//! ```

pub mod cache;
pub mod entropy;
pub mod latency;
pub mod op_trace;
pub mod ops;
pub mod quality_metrics;
pub mod scores;

use std::path::PathBuf;

pub use ops::OpProfiler;
pub use scores::{EvictionEvent, PartitionInfo, compute_h2o_evicted_indices};

use cache::CacheTracker;
use entropy::EntropyTracker;
use latency::LatencyTracker;
use scores::ScoreTracker;

/// Configuration for the inference profiler.
pub struct ProfileConfig {
    /// Score snapshot interval (1 = every step, 10 = every 10th step).
    pub score_snapshot_interval: usize,
    /// Track per-KV-head scores (for H2O+ analysis).
    pub track_per_head: bool,
    /// Which probes to enable (e.g., ["ops", "latency", "scores"]).
    pub enabled_probes: Vec<String>,
    /// Output directory for profile JSON files.
    pub output_dir: PathBuf,
}

/// Metadata about the inference run, written into the profile JSON.
pub struct ProfileMetadata {
    pub model: String,
    pub backend: String,
    pub eviction_policy: String,
    pub max_seq_len: usize,
    pub prompt_len: usize,
    pub generated_tokens: usize,
}

/// Unified inference profiler that holds all probes.
pub struct InferenceProfiler {
    /// Per-op timing breakdown (always active when profiler is created).
    pub ops: OpProfiler,
    /// Attention score snapshots for H2O visualization.
    pub scores: ScoreTracker,
    /// Per-token decode latency.
    pub latency: LatencyTracker,
    /// Attention entropy (optional).
    pub entropy: Option<EntropyTracker>,
    /// KV cache utilization (optional).
    pub cache: Option<CacheTracker>,
    /// Configuration.
    pub config: ProfileConfig,
}

impl InferenceProfiler {
    pub fn new(config: ProfileConfig) -> Self {
        let scores = if config.enabled_probes.iter().any(|p| p == "scores") {
            ScoreTracker::new(config.score_snapshot_interval, config.track_per_head)
        } else {
            ScoreTracker::default()
        };

        let latency = if config.enabled_probes.iter().any(|p| p == "latency") {
            LatencyTracker::new()
        } else {
            LatencyTracker::default()
        };

        let entropy = if config.enabled_probes.iter().any(|p| p == "entropy") {
            Some(EntropyTracker::new())
        } else {
            None
        };

        let cache = if config.enabled_probes.iter().any(|p| p == "cache") {
            Some(CacheTracker::new())
        } else {
            None
        };

        Self {
            ops: OpProfiler::new(),
            scores,
            latency,
            entropy,
            cache,
            config,
        }
    }

    /// Called at the beginning of each decode step.
    pub fn on_step_begin(&mut self, _step: usize) {
        // Reserved for future per-step initialization.
    }

    /// Called at the end of each decode step after forward pass and sampling.
    #[allow(clippy::too_many_arguments)]
    pub fn on_step_end(
        &mut self,
        step: usize,
        token_id: u32,
        forward_us: u64,
        sample_us: u64,
        total_us: u64,
        cache_len: usize,
        importance: Option<&[f32]>,
        head_importance: Option<&[f32]>,
        n_kv_heads: usize,
        position_map: Option<&[usize]>,
    ) {
        self.latency
            .record(step, forward_us, sample_us, total_us, cache_len);

        if let Some(imp) = importance {
            self.scores.take_snapshot(
                step,
                token_id,
                cache_len,
                imp,
                head_importance,
                n_kv_heads,
                position_map,
            );
        }
    }

    /// Record an eviction event.
    pub fn on_eviction(&mut self, event: EvictionEvent) {
        self.scores.record_eviction(event);
    }

    /// Check if a specific probe is enabled.
    pub fn is_probe_enabled(&self, name: &str) -> bool {
        self.config.enabled_probes.iter().any(|p| p == name)
    }

    /// Export all collected data to a JSON file.
    pub fn export_json(&self, metadata: &ProfileMetadata) -> anyhow::Result<PathBuf> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let output = serde_json::json!({
            "metadata": {
                "model": metadata.model,
                "backend": metadata.backend,
                "eviction_policy": metadata.eviction_policy,
                "max_seq_len": metadata.max_seq_len,
                "prompt_len": metadata.prompt_len,
                "generated_tokens": metadata.generated_tokens,
                "timestamp": timestamp,
            },
            "ops": self.ops.to_json(),
            "latency": self.latency.to_json(),
            "scores": self.scores.to_json(),
            "entropy": self.entropy.as_ref().map(|e| e.to_json()),
            "cache": self.cache.as_ref().map(|c| c.to_json()),
        });

        let filename = format!(
            "profile_{}_{}_{}_{}.json",
            metadata.backend, metadata.eviction_policy, metadata.generated_tokens, timestamp
        );
        let path = self.config.output_dir.join(&filename);
        std::fs::create_dir_all(&self.config.output_dir)?;
        std::fs::write(&path, serde_json::to_string_pretty(&output)?)?;

        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ProfileConfig {
        ProfileConfig {
            score_snapshot_interval: 1,
            track_per_head: false,
            enabled_probes: vec![
                "ops".to_string(),
                "latency".to_string(),
                "scores".to_string(),
            ],
            output_dir: PathBuf::from("/tmp/argus_engine_profile_test"),
        }
    }

    #[test]
    fn test_profiler_creation_with_default_probes() {
        let p = InferenceProfiler::new(default_config());
        assert!(p.is_probe_enabled("ops"));
        assert!(p.is_probe_enabled("latency"));
        assert!(p.is_probe_enabled("scores"));
        assert!(!p.is_probe_enabled("entropy"));
        assert!(!p.is_probe_enabled("cache"));
        assert!(p.entropy.is_none());
        assert!(p.cache.is_none());
    }

    #[test]
    fn test_profiler_creation_with_optional_probes() {
        let config = ProfileConfig {
            enabled_probes: vec!["entropy".to_string(), "cache".to_string()],
            ..default_config()
        };
        let p = InferenceProfiler::new(config);
        assert!(p.entropy.is_some());
        assert!(p.cache.is_some());
    }

    #[test]
    fn test_on_step_end_records_latency() {
        let mut p = InferenceProfiler::new(default_config());
        p.on_step_end(0, 42, 5000, 100, 5100, 128, None, None, 0, None);
        p.on_step_end(1, 43, 5200, 80, 5280, 129, None, None, 0, None);

        let records = p.latency.records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].step, 0);
        assert_eq!(records[0].forward_us, 5000);
        assert_eq!(records[0].cache_len, 128);
        assert_eq!(records[1].step, 1);
        assert_eq!(records[1].cache_len, 129);
    }

    #[test]
    fn test_on_eviction_records_event() {
        let mut p = InferenceProfiler::new(default_config());
        p.on_eviction(EvictionEvent {
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

        assert_eq!(p.scores.evictions().len(), 1);
        assert_eq!(p.scores.evictions()[0].step, 50);
        assert_eq!(p.scores.evictions()[0].evicted_count, 56);
    }

    #[test]
    fn test_export_json_creates_file() {
        let dir = std::env::temp_dir().join("argus_engine_profile_test_export");
        let _ = std::fs::remove_dir_all(&dir);

        let config = ProfileConfig {
            output_dir: dir.clone(),
            ..default_config()
        };
        let mut p = InferenceProfiler::new(config);
        p.ops.matmul_qkv = 1000;
        p.ops.count = 10;
        p.on_step_end(0, 42, 5000, 100, 5100, 128, None, None, 0, None);

        let metadata = ProfileMetadata {
            model: "llama-3.2-1b".to_string(),
            backend: "cpu".to_string(),
            eviction_policy: "h2o".to_string(),
            max_seq_len: 2048,
            prompt_len: 64,
            generated_tokens: 100,
        };

        let path = p.export_json(&metadata).unwrap();
        assert!(path.exists());

        // Parse and verify structure
        let content = std::fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(json["metadata"]["model"], "llama-3.2-1b");
        assert_eq!(json["metadata"]["backend"], "cpu");
        assert_eq!(json["ops"]["count"], 10);
        assert_eq!(json["ops"]["breakdown"]["matmul_qkv"]["total_us"], 1000);

        let latency_records = json["latency"]["records"].as_array().unwrap();
        assert_eq!(latency_records.len(), 1);
        assert_eq!(latency_records[0]["forward_us"], 5000);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_export_json_optional_probes_null_when_disabled() {
        let dir = std::env::temp_dir().join("argus_engine_profile_test_null");
        let _ = std::fs::remove_dir_all(&dir);

        let config = ProfileConfig {
            output_dir: dir.clone(),
            enabled_probes: vec!["ops".to_string()],
            ..default_config()
        };
        let p = InferenceProfiler::new(config);

        let metadata = ProfileMetadata {
            model: "test".to_string(),
            backend: "cpu".to_string(),
            eviction_policy: "none".to_string(),
            max_seq_len: 512,
            prompt_len: 10,
            generated_tokens: 0,
        };

        let path = p.export_json(&metadata).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert!(json["entropy"].is_null());
        assert!(json["cache"].is_null());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_on_step_end_records_score_snapshot() {
        let mut p = InferenceProfiler::new(default_config());
        let scores = vec![0.8, 0.15, 0.05];
        p.on_step_end(0, 42, 5000, 100, 5100, 3, Some(&scores), None, 0, None);

        assert_eq!(p.scores.snapshots().len(), 1);
        assert_eq!(p.scores.snapshots()[0].step, 0);
        assert_eq!(p.scores.snapshots()[0].token_id, 42);
        assert!((p.scores.snapshots()[0].importance[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_on_step_end_no_snapshot_without_scores() {
        let mut p = InferenceProfiler::new(default_config());
        p.on_step_end(0, 42, 5000, 100, 5100, 128, None, None, 0, None);

        // No importance → no snapshot
        assert!(p.scores.snapshots().is_empty());
        // But latency is still recorded
        assert_eq!(p.latency.records().len(), 1);
    }

    #[test]
    fn test_on_step_end_with_head_importance() {
        let config = ProfileConfig {
            track_per_head: true,
            ..default_config()
        };
        let mut p = InferenceProfiler::new(config);
        let scores = vec![0.5, 0.5];
        let head_scores = vec![0.6, 0.4, 0.3, 0.7];
        p.on_step_end(
            0,
            10,
            5000,
            100,
            5100,
            2,
            Some(&scores),
            Some(&head_scores),
            2,
            None,
        );

        let snap = &p.scores.snapshots()[0];
        assert!(snap.head_importance.is_some());
        assert_eq!(snap.head_importance.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn test_full_lifecycle_scores_and_eviction() {
        let mut p = InferenceProfiler::new(default_config());

        // Simulate 5 decode steps
        for step in 0..5 {
            let scores: Vec<f32> = (0..10).map(|i| (10 - i) as f32 * 0.1).collect();
            p.on_step_end(
                step,
                step as u32,
                5000,
                100,
                5100,
                10,
                Some(&scores),
                None,
                0,
                None,
            );
        }

        // Eviction at step 3
        p.on_eviction(EvictionEvent {
            step: 3,
            policy: "h2o".to_string(),
            before_len: 10,
            after_len: 8,
            evicted_count: 2,
            partition: PartitionInfo {
                prefix_end: 2,
                hh_count: 3,
                recent_start: 5,
            },
            evicted_indices: vec![],
            pre_eviction_scores: vec![],
        });

        assert_eq!(p.scores.snapshots().len(), 5);
        assert_eq!(p.scores.evictions().len(), 1);
        assert_eq!(p.latency.records().len(), 5);

        // Verify JSON export contains all data
        let json = p.scores.to_json();
        assert_eq!(json["snapshot_count"], 5);
        assert_eq!(json["eviction_count"], 1);
    }
}
