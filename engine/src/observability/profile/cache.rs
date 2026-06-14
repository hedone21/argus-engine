/// KV cache utilization tracker (optional probe).
///
/// Records per-step cache capacity, occupancy, and memory usage
/// to analyze cache sizing, dynamic growth, and eviction impact.
#[derive(Default)]
pub struct CacheTracker {
    records: Vec<CacheRecord>,
}

pub struct CacheRecord {
    pub step: usize,
    pub capacity: usize,
    pub current_pos: usize,
    pub utilization: f32,
    pub memory_bytes: usize,
}

impl CacheTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn records(&self) -> &[CacheRecord] {
        &self.records
    }

    pub fn record(
        &mut self,
        step: usize,
        capacity: usize,
        current_pos: usize,
        memory_bytes: usize,
    ) {
        let utilization = if capacity > 0 {
            current_pos as f32 / capacity as f32
        } else {
            0.0
        };
        self.records.push(CacheRecord {
            step,
            capacity,
            current_pos,
            utilization,
            memory_bytes,
        });
    }

    pub fn to_json(&self) -> serde_json::Value {
        let records: Vec<serde_json::Value> = self
            .records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "step": r.step,
                    "capacity": r.capacity,
                    "current_pos": r.current_pos,
                    "utilization": (r.utilization * 1000.0).round() / 1000.0,
                    "memory_bytes": r.memory_bytes,
                })
            })
            .collect();
        serde_json::json!({ "records": records })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_tracker_empty() {
        let t = CacheTracker::new();
        assert!(t.records().is_empty());
    }

    #[test]
    fn test_cache_tracker_record() {
        let mut t = CacheTracker::new();
        t.record(0, 512, 128, 1024 * 1024);

        assert_eq!(t.records().len(), 1);
        let r = &t.records()[0];
        assert_eq!(r.step, 0);
        assert_eq!(r.capacity, 512);
        assert_eq!(r.current_pos, 128);
        assert!((r.utilization - 0.25).abs() < 1e-5);
        assert_eq!(r.memory_bytes, 1024 * 1024);
    }

    #[test]
    fn test_cache_tracker_utilization_calculation() {
        let mut t = CacheTracker::new();
        t.record(0, 1000, 500, 0);
        t.record(1, 1000, 1000, 0);
        t.record(2, 2000, 1000, 0); // after growth

        assert!((t.records()[0].utilization - 0.5).abs() < 1e-5);
        assert!((t.records()[1].utilization - 1.0).abs() < 1e-5);
        assert!((t.records()[2].utilization - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_cache_tracker_zero_capacity() {
        let mut t = CacheTracker::new();
        t.record(0, 0, 0, 0);
        assert!((t.records()[0].utilization - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_cache_tracker_growth_and_eviction() {
        let mut t = CacheTracker::new();
        // Initial: small cache
        t.record(0, 128, 64, 32768);
        // Grows
        t.record(10, 256, 140, 65536);
        // After eviction: pos drops
        t.record(20, 256, 100, 65536);

        assert_eq!(t.records().len(), 3);
        assert!(t.records()[2].utilization < t.records()[1].utilization);
    }

    #[test]
    fn test_cache_tracker_to_json() {
        let mut t = CacheTracker::new();
        t.record(0, 512, 256, 1048576);

        let json = t.to_json();
        let records = json["records"].as_array().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["step"], 0);
        assert_eq!(records[0]["capacity"], 512);
        assert_eq!(records[0]["current_pos"], 256);
        assert_eq!(records[0]["utilization"], 0.5);
        assert_eq!(records[0]["memory_bytes"], 1048576);
    }

    #[test]
    fn test_cache_tracker_to_json_empty() {
        let t = CacheTracker::new();
        let json = t.to_json();
        assert!(json["records"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_cache_tracker_many_records() {
        let mut t = CacheTracker::new();
        for i in 0..500 {
            t.record(i, 2048, 128 + i, (128 + i) * 8192);
        }
        assert_eq!(t.records().len(), 500);
        assert_eq!(t.records()[499].current_pos, 627);
    }
}
