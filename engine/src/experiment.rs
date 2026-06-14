use argus_shared::EngineDirective;
use serde::{Deserialize, Serialize};
use std::io::{BufWriter, Write};

// ── Schedule ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ExperimentSchedule {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub directives: Vec<DirectiveEntry>,
}

#[derive(Debug, Deserialize)]
pub struct DirectiveEntry {
    pub at_token: usize,
    pub directive: EngineDirective,
}

impl ExperimentSchedule {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let schedule: Self = serde_json::from_str(&content)?;
        Ok(schedule)
    }

    /// Return all directives scheduled at the given decode token position.
    pub fn directives_at(&self, token_pos: usize) -> impl Iterator<Item = &DirectiveEntry> {
        self.directives
            .iter()
            .filter(move |e| e.at_token == token_pos)
    }
}

// ── SystemSampler ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct SystemMetrics {
    pub rss_mb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_pct: Option<f64>,
    pub cpu_mhz: Vec<u64>,
    pub thermal_mc: Vec<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_mhz: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_pct: Option<f64>,
}

pub struct SystemSampler {
    interval: usize,
    prev_cpu_times: Option<(u64, u64)>, // (process_ticks, total_elapsed_ticks)
}

impl SystemSampler {
    pub fn new(interval: usize) -> Self {
        Self {
            interval,
            prev_cpu_times: None,
        }
    }

    /// Sample system metrics if this token position is on the sampling interval.
    /// Returns None if interval=0 or position doesn't match interval.
    pub fn sample(&mut self, token_pos: usize) -> Option<SystemMetrics> {
        if self.interval == 0 {
            return None;
        }
        if !token_pos.is_multiple_of(self.interval) {
            return None;
        }
        Some(SystemMetrics {
            rss_mb: self.read_rss(),
            cpu_pct: self.read_cpu_util(),
            cpu_mhz: Self::read_cpu_freqs(),
            thermal_mc: Self::read_thermal(),
            gpu_mhz: Self::read_gpu_freq(),
            gpu_pct: Self::read_gpu_util(),
        })
    }

    /// Snapshot without interval check (for summary start/end).
    pub fn snapshot(&mut self) -> SystemMetrics {
        SystemMetrics {
            rss_mb: self.read_rss(),
            cpu_pct: self.read_cpu_util(),
            cpu_mhz: Self::read_cpu_freqs(),
            thermal_mc: Self::read_thermal(),
            gpu_mhz: Self::read_gpu_freq(),
            gpu_pct: Self::read_gpu_util(),
        }
    }

    /// Read process RSS from /proc/self/statm (field 1 = resident pages).
    fn read_rss(&self) -> f64 {
        let page_size = 4096u64; // standard Linux page size
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| {
                s.split_whitespace()
                    .nth(1) // resident pages
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(|pages| (pages * page_size) as f64 / (1024.0 * 1024.0))
            })
            .unwrap_or(0.0)
    }

    /// Read process CPU utilization from /proc/self/stat.
    /// Returns delta-based percentage since last call, or None on first call.
    fn read_cpu_util(&mut self) -> Option<f64> {
        let proc_ticks = std::fs::read_to_string("/proc/self/stat")
            .ok()
            .and_then(|s| {
                let fields: Vec<&str> = s.split_whitespace().collect();
                // fields[13]=utime, fields[14]=stime (both in clock ticks)
                if fields.len() > 14 {
                    let utime: u64 = fields[13].parse().ok()?;
                    let stime: u64 = fields[14].parse().ok()?;
                    Some(utime + stime)
                } else {
                    None
                }
            })?;

        // Read total CPU ticks from /proc/stat (first "cpu" line)
        let total_ticks = std::fs::read_to_string("/proc/stat").ok().and_then(|s| {
            let line = s.lines().next()?;
            if !line.starts_with("cpu ") {
                return None;
            }
            let sum: u64 = line
                .split_whitespace()
                .skip(1) // skip "cpu"
                .filter_map(|v| v.parse::<u64>().ok())
                .sum();
            Some(sum)
        })?;

        let result = if let Some((prev_proc, prev_total)) = self.prev_cpu_times {
            let d_proc = proc_ticks.saturating_sub(prev_proc);
            let d_total = total_ticks.saturating_sub(prev_total);
            if d_total > 0 {
                // Multiply by number of CPUs since /proc/stat is aggregate
                let num_cpus = num_cpus_online();
                Some((d_proc as f64 / d_total as f64) * 100.0 * num_cpus as f64)
            } else {
                None
            }
        } else {
            None
        };

        self.prev_cpu_times = Some((proc_ticks, total_ticks));
        result
    }

    /// Read per-core CPU frequencies from sysfs (MHz).
    fn read_cpu_freqs() -> Vec<u64> {
        let mut freqs = Vec::new();
        for i in 0.. {
            let path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", i);
            match std::fs::read_to_string(&path) {
                Ok(s) => {
                    if let Ok(khz) = s.trim().parse::<u64>() {
                        freqs.push(khz / 1000); // kHz → MHz
                    }
                }
                Err(_) => break,
            }
        }
        freqs
    }

    /// Read thermal zone temperatures from sysfs (millidegrees Celsius).
    fn read_thermal() -> Vec<i64> {
        let mut temps = Vec::new();
        for i in 0.. {
            let path = format!("/sys/class/thermal/thermal_zone{}/temp", i);
            match std::fs::read_to_string(&path) {
                Ok(s) => {
                    if let Ok(mc) = s.trim().parse::<i64>() {
                        temps.push(mc);
                    }
                }
                Err(_) => break,
            }
        }
        temps
    }

    /// Read GPU frequency (Android Adreno via kgsl).
    fn read_gpu_freq() -> Option<u64> {
        std::fs::read_to_string("/sys/class/kgsl/kgsl-3d0/gpuclk")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|hz| hz / 1_000_000) // Hz → MHz
    }

    /// Read GPU utilization (Android Adreno via kgsl).
    fn read_gpu_util() -> Option<f64> {
        std::fs::read_to_string("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
    }

    /// Read CPU governor (for summary).
    pub fn read_governor() -> String {
        std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    }
}

fn num_cpus_online() -> usize {
    std::fs::read_to_string("/proc/stat")
        .map(|s| {
            s.lines()
                .filter(|line| {
                    line.starts_with("cpu")
                        && line.as_bytes().get(3).is_some_and(|b| b.is_ascii_digit())
                })
                .count()
        })
        .unwrap_or(1)
}

// ── JSONL Records ─────────────────────────────────────────────

#[derive(Serialize)]
pub struct TokenRecord<'a> {
    pub pos: usize,
    pub token_id: u32,
    pub text: String,
    pub tbt_ms: f64,
    pub forward_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal: Option<&'a str>,
    pub actions: Vec<String>,
    pub cache_pos: usize,
    pub throttle_ms: u64,
    pub top_logits: Vec<(u32, f32)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sys: Option<SystemMetrics>,
}

#[derive(Serialize)]
pub struct SummaryRecord {
    pub _summary: bool,
    pub total_tokens: usize,
    pub ttft_ms: f64,
    pub avg_tbt_ms: f64,
    pub avg_forward_ms: f64,
    pub total_throttle_ms: u64,
    pub eviction_count: usize,
    pub evicted_tokens_total: usize,
    pub final_cache_pos: usize,
    pub max_seq_len: usize,
    pub prompt: String,
    pub schedule_name: String,
    pub eviction_policy: String,
    pub backend: String,
    pub sample_interval: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sys_start: Option<SystemMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sys_end: Option<SystemMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub governor: Option<String>,
}

// ── JSONL Writer ──────────────────────────────────────────────

pub struct JsonlWriter {
    writer: BufWriter<std::fs::File>,
}

impl JsonlWriter {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn write_token(&mut self, record: &TokenRecord) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.writer, record)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    pub fn write_summary(&mut self, record: &SummaryRecord) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.writer, record)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
}

// ── Top-K Logits Extraction ───────────────────────────────────

/// Extract top-K (token_id, logit_value) pairs from raw logits.
/// Logits must not have been modified by temperature/softmax yet.
pub fn extract_top_k_logits(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(logits.len());
    if k == 0 {
        return vec![];
    }

    // Partial sort: find top-K by maintaining a min-heap (via sorted vec for simplicity)
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (i, &val) in logits.iter().enumerate() {
        if top.len() < k {
            top.push((i as u32, val));
            if top.len() == k {
                top.sort_by(|a, b| a.1.total_cmp(&b.1));
            }
        } else if val > top[0].1 {
            top[0] = (i as u32, val);
            top.sort_by(|a, b| a.1.total_cmp(&b.1));
        }
    }

    // Return in descending order
    top.reverse();
    top
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_directives_at() {
        let json = r#"{
            "name": "test",
            "directives": [
                {"at_token": 10, "directive": {"seq_id": 1, "commands": [{"type": "kv.evict_h2o", "keep_ratio": 0.5}]}},
                {"at_token": 10, "directive": {"seq_id": 2, "commands": [{"type": "throttle", "delay_ms": 30}]}},
                {"at_token": 20, "directive": {"seq_id": 3, "commands": [{"type": "restore_defaults"}]}}
            ]
        }"#;
        let schedule: ExperimentSchedule = serde_json::from_str(json).unwrap();
        assert_eq!(schedule.name, "test");
        assert_eq!(schedule.directives_at(10).count(), 2);
        assert_eq!(schedule.directives_at(20).count(), 1);
        assert_eq!(schedule.directives_at(5).count(), 0);
    }

    #[test]
    fn test_extract_top_k_logits() {
        let logits = vec![1.0, 5.0, 3.0, 8.0, 2.0, 7.0];
        let top3 = extract_top_k_logits(&logits, 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].0, 3); // index 3, value 8.0
        assert_eq!(top3[1].0, 5); // index 5, value 7.0
        assert_eq!(top3[2].0, 1); // index 1, value 5.0
    }

    #[test]
    fn test_extract_top_k_logits_k_larger_than_len() {
        let logits = vec![1.0, 2.0];
        let top = extract_top_k_logits(&logits, 10);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_system_sampler_interval_zero_returns_none() {
        let mut sampler = SystemSampler::new(0);
        assert!(sampler.sample(0).is_none());
        assert!(sampler.sample(5).is_none());
    }

    #[test]
    fn test_system_sampler_interval_respects_interval() {
        let mut sampler = SystemSampler::new(3);
        assert!(sampler.sample(0).is_some()); // 0 % 3 == 0
        assert!(sampler.sample(1).is_none()); // 1 % 3 != 0
        assert!(sampler.sample(2).is_none()); // 2 % 3 != 0
        assert!(sampler.sample(3).is_some()); // 3 % 3 == 0
    }

    #[test]
    fn test_system_sampler_snapshot_always_returns() {
        let mut sampler = SystemSampler::new(0); // interval=0
        let metrics = sampler.snapshot();
        // Should still return metrics (interval irrelevant for snapshot)
        assert!(metrics.rss_mb >= 0.0);
    }

    #[test]
    fn test_token_record_serialization() {
        let record = TokenRecord {
            pos: 0,
            token_id: 42,
            text: "hello".to_string(),
            tbt_ms: 45.2,
            forward_ms: 44.8,
            signal: None,
            actions: vec![],
            cache_pos: 10,
            throttle_ms: 0,
            top_logits: vec![(42, 12.3), (100, 11.1)],
            sys: None,
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"pos\":0"));
        assert!(json.contains("\"token_id\":42"));
        assert!(!json.contains("\"signal\"")); // skipped when None
        assert!(!json.contains("\"sys\"")); // skipped when None
    }

    #[test]
    fn test_summary_record_serialization() {
        let record = SummaryRecord {
            _summary: true,
            total_tokens: 128,
            ttft_ms: 1200.0,
            avg_tbt_ms: 45.0,
            avg_forward_ms: 44.0,
            total_throttle_ms: 0,
            eviction_count: 0,
            evicted_tokens_total: 0,
            final_cache_pos: 128,
            max_seq_len: 2048,
            prompt: "test".to_string(),
            schedule_name: "baseline".to_string(),
            eviction_policy: "none".to_string(),
            backend: "cpu".to_string(),
            sample_interval: 1,
            sys_start: None,
            sys_end: None,
            governor: Some("performance".to_string()),
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"_summary\":true"));
        assert!(json.contains("\"total_tokens\":128"));
    }
}
