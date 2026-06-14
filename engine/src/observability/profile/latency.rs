/// Per-token decode latency tracker.
#[derive(Default)]
pub struct LatencyTracker {
    records: Vec<LatencyRecord>,
}

pub struct LatencyRecord {
    pub step: usize,
    pub forward_us: u64,
    pub sample_us: u64,
    pub total_us: u64,
    pub cache_len: usize,
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(
        &mut self,
        step: usize,
        forward_us: u64,
        sample_us: u64,
        total_us: u64,
        cache_len: usize,
    ) {
        self.records.push(LatencyRecord {
            step,
            forward_us,
            sample_us,
            total_us,
            cache_len,
        });
    }

    pub fn records(&self) -> &[LatencyRecord] {
        &self.records
    }

    /// Compute summary statistics.
    pub fn summary(&self) -> Option<LatencySummary> {
        if self.records.is_empty() {
            return None;
        }
        let n = self.records.len() as f64;
        let avg_forward = self.records.iter().map(|r| r.forward_us).sum::<u64>() as f64 / n;
        let avg_total = self.records.iter().map(|r| r.total_us).sum::<u64>() as f64 / n;
        let min_forward = self.records.iter().map(|r| r.forward_us).min().unwrap();
        let max_forward = self.records.iter().map(|r| r.forward_us).max().unwrap();

        Some(LatencySummary {
            count: self.records.len(),
            avg_forward_us: avg_forward,
            avg_total_us: avg_total,
            min_forward_us: min_forward,
            max_forward_us: max_forward,
        })
    }

    pub fn to_json(&self) -> serde_json::Value {
        let records: Vec<serde_json::Value> = self
            .records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "step": r.step,
                    "forward_us": r.forward_us,
                    "sample_us": r.sample_us,
                    "total_us": r.total_us,
                    "cache_len": r.cache_len,
                })
            })
            .collect();

        let summary = self.summary().map(|s| {
            serde_json::json!({
                "count": s.count,
                "avg_forward_us": s.avg_forward_us,
                "avg_total_us": s.avg_total_us,
                "min_forward_us": s.min_forward_us,
                "max_forward_us": s.max_forward_us,
            })
        });

        serde_json::json!({
            "records": records,
            "summary": summary,
        })
    }
}

pub struct LatencySummary {
    pub count: usize,
    pub avg_forward_us: f64,
    pub avg_total_us: f64,
    pub min_forward_us: u64,
    pub max_forward_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_tracker_empty() {
        let t = LatencyTracker::new();
        assert!(t.records().is_empty());
        assert!(t.summary().is_none());
    }

    #[test]
    fn test_latency_tracker_record_and_access() {
        let mut t = LatencyTracker::new();
        t.record(0, 5000, 100, 5100, 128);
        t.record(1, 5200, 80, 5280, 129);
        t.record(2, 4800, 120, 4920, 130);

        assert_eq!(t.records().len(), 3);
        assert_eq!(t.records()[0].step, 0);
        assert_eq!(t.records()[0].forward_us, 5000);
        assert_eq!(t.records()[0].sample_us, 100);
        assert_eq!(t.records()[0].total_us, 5100);
        assert_eq!(t.records()[0].cache_len, 128);
        assert_eq!(t.records()[2].step, 2);
        assert_eq!(t.records()[2].cache_len, 130);
    }

    #[test]
    fn test_latency_tracker_summary() {
        let mut t = LatencyTracker::new();
        t.record(0, 4000, 100, 4100, 128);
        t.record(1, 6000, 200, 6200, 129);
        t.record(2, 5000, 150, 5150, 130);

        let s = t.summary().unwrap();
        assert_eq!(s.count, 3);
        assert!((s.avg_forward_us - 5000.0).abs() < 1.0);
        assert!((s.avg_total_us - 5150.0).abs() < 1.0);
        assert_eq!(s.min_forward_us, 4000);
        assert_eq!(s.max_forward_us, 6000);
    }

    #[test]
    fn test_latency_tracker_summary_single_record() {
        let mut t = LatencyTracker::new();
        t.record(0, 3000, 50, 3050, 64);

        let s = t.summary().unwrap();
        assert_eq!(s.count, 1);
        assert_eq!(s.min_forward_us, 3000);
        assert_eq!(s.max_forward_us, 3000);
        assert!((s.avg_forward_us - 3000.0).abs() < 1.0);
    }

    #[test]
    fn test_latency_tracker_to_json_structure() {
        let mut t = LatencyTracker::new();
        t.record(0, 5000, 100, 5100, 128);
        t.record(1, 5200, 80, 5280, 129);

        let json = t.to_json();

        let records = json["records"].as_array().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["step"], 0);
        assert_eq!(records[0]["forward_us"], 5000);
        assert_eq!(records[0]["sample_us"], 100);
        assert_eq!(records[0]["total_us"], 5100);
        assert_eq!(records[0]["cache_len"], 128);
        assert_eq!(records[1]["step"], 1);
    }

    #[test]
    fn test_latency_tracker_to_json_includes_summary() {
        let mut t = LatencyTracker::new();
        t.record(0, 4000, 100, 4100, 128);
        t.record(1, 6000, 200, 6200, 256);

        let json = t.to_json();
        let summary = &json["summary"];
        assert!(!summary.is_null());
        assert_eq!(summary["count"], 2);
        assert_eq!(summary["min_forward_us"], 4000);
        assert_eq!(summary["max_forward_us"], 6000);
    }

    #[test]
    fn test_latency_tracker_to_json_empty() {
        let t = LatencyTracker::new();
        let json = t.to_json();

        let records = json["records"].as_array().unwrap();
        assert!(records.is_empty());
        assert!(json["summary"].is_null());
    }

    #[test]
    fn test_latency_tracker_cache_len_increases() {
        let mut t = LatencyTracker::new();
        for i in 0..100 {
            t.record(i, 5000, 100, 5100, 128 + i);
        }
        assert_eq!(t.records().len(), 100);
        assert_eq!(t.records()[0].cache_len, 128);
        assert_eq!(t.records()[99].cache_len, 227);
    }
}
