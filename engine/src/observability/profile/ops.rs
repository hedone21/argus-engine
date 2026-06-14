/// Compute `v / total * 100.0`, returning `0.0` when `total == 0`.
fn pct_of(v: u64, total: u64) -> f64 {
    if total > 0 {
        v as f64 / total as f64 * 100.0
    } else {
        0.0
    }
}

/// Per-operation profiler for forward_gen timing breakdown.
/// Accumulates microseconds per operation across layers and tokens.
#[derive(Default)]
pub struct OpProfiler {
    pub rms_norm: u64,
    pub matmul_qkv: u64,
    pub rope: u64,
    pub kv_update: u64,
    pub attention: u64,
    pub matmul_wo: u64,
    pub matmul_ffn: u64,
    pub silu_mul: u64,
    pub add_assign: u64,
    pub copy_residual: u64,
    pub cast: u64,
    /// Final LM head matmul (only populated by `--profile-events`).
    /// The legacy `--profile` path does not time lm_head separately.
    pub lm_head: u64,
    /// Generic matmul bucket used by `--profile-events` when the caller did
    /// not set a more specific hint (matmul_qkv/matmul_wo/matmul_ffn/lm_head).
    /// Only populated by event-based profiling.
    pub matmul: u64,
    /// Residual/elementwise operations that did not match a specific label.
    /// Only populated by event-based profiling.
    pub gather: u64,
    /// Fallback bucket: any kernel dispatch whose event label did not match
    /// a known OpProfiler field. Only populated by event-based profiling.
    pub other: u64,
    pub count: u64,
    /// Prefill-specific per-op breakdown (populated during prefill pass).
    pub prefill: PrefillOpProfiler,
}

/// Per-operation profiler for the prefill (multi-token) forward pass.
/// Accumulates microseconds per operation across all layers of one prefill run.
#[derive(Default, Clone)]
pub struct PrefillOpProfiler {
    pub rms_norm: u64,
    pub matmul_qkv: u64,
    pub rope: u64,
    pub kv_write: u64,
    /// GPU flash attention prefill (when dispatched).
    pub flash_prefill_gpu: u64,
    /// CPU attention fallback (when GPU not dispatched).
    pub flash_prefill_cpu: u64,
    pub matmul_wo: u64,
    pub ffn_gate: u64,
    pub ffn_up: u64,
    pub ffn_down: u64,
    pub silu_mul: u64,
    pub add_assign: u64,
    /// Number of GPU→CPU attention fallbacks (per-layer count).
    pub cpu_fallback_count: u64,
    /// Number of layers profiled (incremented per layer).
    pub layer_count: u64,
}

impl PrefillOpProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all accumulators (e.g. between prefill chunks).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn ops(&self) -> [(&'static str, u64); 10] {
        [
            ("matmul_qkv", self.matmul_qkv),
            ("matmul_wo", self.matmul_wo),
            ("ffn_gate", self.ffn_gate),
            ("ffn_up", self.ffn_up),
            ("ffn_down", self.ffn_down),
            ("flash_prefill_gpu", self.flash_prefill_gpu),
            ("flash_prefill_cpu", self.flash_prefill_cpu),
            ("rms_norm", self.rms_norm),
            ("rope", self.rope),
            ("kv_write", self.kv_write),
        ]
    }

    fn total(&self) -> u64 {
        self.ops().iter().map(|(_, v)| v).sum::<u64>() + self.silu_mul + self.add_assign
    }

    pub fn print_report(&self) {
        let total = self.total();
        let n = self.layer_count.max(1);
        eprintln!(
            "\n[Profile/Prefill] Per-op breakdown ({} layers, {} CPU attn fallbacks):",
            n, self.cpu_fallback_count
        );
        eprintln!(
            "  {:<24} {:>10} {:>10} {:>8}",
            "Operation", "Total(us)", "Avg(us)", "%"
        );
        eprintln!("  {:-<24} {:-<10} {:-<10} {:-<8}", "", "", "", "");
        for (name, val) in &self.ops() {
            eprintln!(
                "  {:<24} {:>10} {:>10} {:>7.1}%",
                name,
                val,
                val / n,
                pct_of(*val, total)
            );
        }
        eprintln!(
            "  {:<24} {:>10} {:>10} {:>7.1}%",
            "silu_mul",
            self.silu_mul,
            self.silu_mul / n,
            pct_of(self.silu_mul, total)
        );
        eprintln!(
            "  {:<24} {:>10} {:>10} {:>7.1}%",
            "add_assign",
            self.add_assign,
            self.add_assign / n,
            pct_of(self.add_assign, total)
        );
        eprintln!("  {:<24} {:>10} {:>10}", "TOTAL", total, total / n);
    }

    pub fn to_json(&self) -> serde_json::Value {
        let total = self.total();
        let n = self.layer_count.max(1);

        let mut breakdown = serde_json::Map::new();
        for (name, val) in &self.ops() {
            breakdown.insert(
                name.to_string(),
                serde_json::json!({
                    "total_us": val,
                    "avg_us": val / n,
                    "pct": (pct_of(*val, total) * 10.0).round() / 10.0,
                }),
            );
        }
        breakdown.insert(
            "silu_mul".to_string(),
            serde_json::json!({
                "total_us": self.silu_mul,
                "avg_us": self.silu_mul / n,
                "pct": (pct_of(self.silu_mul, total) * 10.0).round() / 10.0,
            }),
        );
        breakdown.insert(
            "add_assign".to_string(),
            serde_json::json!({
                "total_us": self.add_assign,
                "avg_us": self.add_assign / n,
                "pct": (pct_of(self.add_assign, total) * 10.0).round() / 10.0,
            }),
        );

        serde_json::json!({
            "layer_count": self.layer_count,
            "cpu_fallback_count": self.cpu_fallback_count,
            "total_us": total,
            "breakdown": breakdown,
        })
    }
}

impl OpProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all decode operations as (name, value) pairs sorted by typical impact.
    fn ops(&self) -> [(&'static str, u64); 15] {
        [
            ("matmul_qkv", self.matmul_qkv),
            ("matmul_wo", self.matmul_wo),
            ("matmul_ffn", self.matmul_ffn),
            ("lm_head", self.lm_head),
            ("matmul", self.matmul),
            ("attention", self.attention),
            ("rms_norm", self.rms_norm),
            ("rope", self.rope),
            ("silu_mul", self.silu_mul),
            ("add_assign", self.add_assign),
            ("copy_residual", self.copy_residual),
            ("kv_update", self.kv_update),
            ("cast", self.cast),
            ("gather", self.gather),
            ("other", self.other),
        ]
    }

    /// Merge event-based per-label GPU ns into the microsecond accumulators.
    ///
    /// Called once per decode step by the `--profile-events` code path.
    /// Input map uses the labels defined in
    /// `.agent/research/2026-04-14_decode_microbench_plan.md` (C-2 table).
    /// Unknown labels fall into the `other` bucket so no dispatch is dropped.
    pub fn merge_from_events(&mut self, events: &std::collections::HashMap<String, u64>) {
        for (label, ns) in events {
            let us = ns / 1_000;
            match label.as_str() {
                "rms_norm" => self.rms_norm += us,
                "matmul_qkv" => self.matmul_qkv += us,
                "matmul_wo" => self.matmul_wo += us,
                "matmul_ffn" => self.matmul_ffn += us,
                "lm_head" => self.lm_head += us,
                "matmul" => self.matmul += us,
                "rope" => self.rope += us,
                "kv_update" => self.kv_update += us,
                "attention" => self.attention += us,
                "silu_mul" => self.silu_mul += us,
                "add_assign" => self.add_assign += us,
                "cast" => self.cast += us,
                "gather" => self.gather += us,
                // flash_prefill counted towards prefill bucket? For decode
                // path it will not fire. Accept into "other" if it leaks.
                "flash_prefill" => self.prefill.flash_prefill_gpu += us,
                // load_time events fire only during model initialization and
                // should not contaminate decode numbers — silently drop.
                "load_time" => {}
                _ => self.other += us,
            }
        }
    }

    fn total(&self) -> u64 {
        self.ops().iter().map(|(_, v)| v).sum()
    }

    pub fn print_report(&self) {
        let total = self.total();
        let n = if self.count > 0 { self.count } else { 1 };
        eprintln!(
            "\n[Profile] Decode per-op breakdown (accumulated over {} layer-calls):",
            n
        );
        eprintln!(
            "  {:<20} {:>10} {:>10} {:>8}",
            "Operation", "Total(us)", "Avg(us)", "%"
        );
        eprintln!("  {:-<20} {:-<10} {:-<10} {:-<8}", "", "", "", "");
        for (name, val) in &self.ops() {
            eprintln!(
                "  {:<20} {:>10} {:>10} {:>7.1}%",
                name,
                val,
                val / n,
                pct_of(*val, total)
            );
        }
        eprintln!("  {:<20} {:>10} {:>10}", "TOTAL", total, total / n,);

        // Print prefill breakdown if any data was collected.
        if self.prefill.layer_count > 0 {
            self.prefill.print_report();
        }
    }

    /// Serialize to JSON for profile export.
    pub fn to_json(&self) -> serde_json::Value {
        let total = self.total();
        let n = self.count.max(1);

        let mut breakdown = serde_json::Map::new();
        for (name, val) in &self.ops() {
            breakdown.insert(
                name.to_string(),
                serde_json::json!({
                    "total_us": val,
                    "avg_us": val / n,
                    "pct": (pct_of(*val, total) * 10.0).round() / 10.0,
                }),
            );
        }

        serde_json::json!({
            "count": self.count,
            "total_us": total,
            "breakdown": breakdown,
            "prefill": self.prefill.to_json(),
        })
    }
}

// ── OpInstrument impls ────────────────────────────────────────────────────────

impl crate::instrument::OpInstrument for OpProfiler {
    fn record_op_us(&mut self, op_name: &'static str, elapsed_us: u64) {
        match op_name {
            "rms_norm" => self.rms_norm += elapsed_us,
            "matmul_qkv" => self.matmul_qkv += elapsed_us,
            "rope" => self.rope += elapsed_us,
            "kv_update" => self.kv_update += elapsed_us,
            "attention" => self.attention += elapsed_us,
            "matmul_wo" => self.matmul_wo += elapsed_us,
            "matmul_ffn" => self.matmul_ffn += elapsed_us,
            "silu_mul" => self.silu_mul += elapsed_us,
            "add_assign" => self.add_assign += elapsed_us,
            "copy_residual" => self.copy_residual += elapsed_us,
            "cast" => self.cast += elapsed_us,
            "lm_head" => self.lm_head += elapsed_us,
            "matmul" => self.matmul += elapsed_us,
            "gather" => self.gather += elapsed_us,
            _ => self.other += elapsed_us,
        }
    }

    // record_cpu_fallback / record_layer_end: default no-op (decode profiler 불필요)

    fn merge_events(&mut self, events: &std::collections::HashMap<String, u64>) {
        self.merge_from_events(events);
    }

    fn note_token(&mut self) {
        self.count += 1;
    }
}

impl crate::instrument::OpInstrument for PrefillOpProfiler {
    fn record_op_us(&mut self, op_name: &'static str, elapsed_us: u64) {
        match op_name {
            "rms_norm" => self.rms_norm += elapsed_us,
            "matmul_qkv" => self.matmul_qkv += elapsed_us,
            "rope" => self.rope += elapsed_us,
            "kv_write" => self.kv_write += elapsed_us,
            "flash_prefill_gpu" => self.flash_prefill_gpu += elapsed_us,
            "flash_prefill_cpu" => self.flash_prefill_cpu += elapsed_us,
            "matmul_wo" => self.matmul_wo += elapsed_us,
            "ffn_gate" => self.ffn_gate += elapsed_us,
            "ffn_up" => self.ffn_up += elapsed_us,
            "ffn_down" => self.ffn_down += elapsed_us,
            "silu_mul" => self.silu_mul += elapsed_us,
            "add_assign" => self.add_assign += elapsed_us,
            _ => {} // prefill profiler는 unknown op 무시
        }
    }

    fn record_cpu_fallback(&mut self, _head_dim: usize, _dtype_str: &str, _reason: &'static str) {
        self.cpu_fallback_count += 1;
    }

    fn record_layer_end(&mut self) {
        self.layer_count += 1;
    }

    // merge_events / note_token: default no-op (prefill profiler 불필요)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pct_of_zero_total_returns_zero() {
        assert_eq!(pct_of(0, 0), 0.0);
        assert_eq!(pct_of(100, 0), 0.0);
    }

    #[test]
    fn test_pct_of_basic() {
        assert!((pct_of(25, 100) - 25.0).abs() < f64::EPSILON);
        assert!((pct_of(0, 100) - 0.0).abs() < f64::EPSILON);
        assert!((pct_of(100, 100) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_op_profiler_default_is_zero() {
        let p = OpProfiler::new();
        assert_eq!(p.total(), 0);
        assert_eq!(p.count, 0);
    }

    #[test]
    fn test_op_profiler_total() {
        let mut p = OpProfiler::new();
        p.matmul_qkv = 100;
        p.attention = 200;
        p.matmul_ffn = 300;
        assert_eq!(p.total(), 600);
    }

    #[test]
    fn test_op_profiler_to_json_structure() {
        let mut p = OpProfiler::new();
        p.matmul_qkv = 1000;
        p.attention = 2000;
        p.matmul_ffn = 3000;
        p.rms_norm = 500;
        p.count = 10;

        let json = p.to_json();

        assert_eq!(json["count"], 10);
        assert_eq!(json["total_us"], 6500);

        let breakdown = json["breakdown"].as_object().unwrap();
        assert_eq!(breakdown.len(), 15); // all 15 ops present

        // Verify specific op values
        assert_eq!(breakdown["matmul_qkv"]["total_us"], 1000);
        assert_eq!(breakdown["matmul_qkv"]["avg_us"], 100); // 1000 / 10
        assert_eq!(breakdown["attention"]["total_us"], 2000);
    }

    #[test]
    fn test_op_profiler_to_json_percentages() {
        let mut p = OpProfiler::new();
        p.matmul_qkv = 500;
        p.matmul_ffn = 500;
        p.count = 1;

        let json = p.to_json();
        let breakdown = json["breakdown"].as_object().unwrap();

        // Each should be 50%
        assert_eq!(breakdown["matmul_qkv"]["pct"], 50.0);
        assert_eq!(breakdown["matmul_ffn"]["pct"], 50.0);
        // Others should be 0%
        assert_eq!(breakdown["attention"]["pct"], 0.0);
    }

    #[test]
    fn test_op_profiler_to_json_zero_count() {
        let mut p = OpProfiler::new();
        p.matmul_qkv = 100;
        // count = 0 → should use 1 to avoid division by zero
        let json = p.to_json();
        assert_eq!(json["breakdown"]["matmul_qkv"]["avg_us"], 100);
    }

    #[test]
    fn test_op_profiler_to_json_zero_total() {
        let p = OpProfiler::new();
        let json = p.to_json();
        assert_eq!(json["total_us"], 0);
        // All percentages should be 0 when total is 0
        let breakdown = json["breakdown"].as_object().unwrap();
        for (_name, val) in breakdown {
            assert_eq!(val["pct"], 0.0);
        }
    }

    #[test]
    fn test_op_profiler_to_json_has_prefill_field() {
        let p = OpProfiler::new();
        let json = p.to_json();
        // prefill field must always be present, even when empty
        assert!(json["prefill"].is_object());
        assert_eq!(json["prefill"]["layer_count"], 0);
        assert_eq!(json["prefill"]["cpu_fallback_count"], 0);
        assert_eq!(json["prefill"]["total_us"], 0);
    }

    // ── PrefillOpProfiler tests ──────────────────────────────────────

    #[test]
    fn test_prefill_op_profiler_default_is_zero() {
        let p = PrefillOpProfiler::new();
        assert_eq!(p.layer_count, 0);
        assert_eq!(p.cpu_fallback_count, 0);
        assert_eq!(p.matmul_qkv, 0);
        assert_eq!(p.flash_prefill_gpu, 0);
        assert_eq!(p.flash_prefill_cpu, 0);
    }

    #[test]
    fn test_prefill_op_profiler_total() {
        let mut p = PrefillOpProfiler::new();
        p.matmul_qkv = 1000;
        p.flash_prefill_gpu = 2000;
        p.ffn_gate = 500;
        p.ffn_up = 500;
        p.ffn_down = 800;
        // total = sum of ops() + silu_mul + add_assign
        let total = p.matmul_qkv
            + p.flash_prefill_gpu
            + p.ffn_gate
            + p.ffn_up
            + p.ffn_down
            + p.matmul_wo
            + p.rms_norm
            + p.rope
            + p.kv_write
            + p.flash_prefill_cpu
            + p.silu_mul
            + p.add_assign;
        assert_eq!(p.total(), total);
    }

    #[test]
    fn test_prefill_op_profiler_to_json_structure() {
        let mut p = PrefillOpProfiler::new();
        p.matmul_qkv = 3000;
        p.flash_prefill_gpu = 5000;
        p.ffn_gate = 1000;
        p.ffn_up = 1000;
        p.ffn_down = 1500;
        p.layer_count = 16;
        p.cpu_fallback_count = 0;

        let json = p.to_json();
        assert_eq!(json["layer_count"], 16);
        assert_eq!(json["cpu_fallback_count"], 0);

        let breakdown = json["breakdown"].as_object().unwrap();
        assert_eq!(breakdown["matmul_qkv"]["total_us"], 3000);
        assert_eq!(breakdown["matmul_qkv"]["avg_us"], 3000 / 16);
        assert_eq!(breakdown["flash_prefill_gpu"]["total_us"], 5000);
        assert_eq!(breakdown["flash_prefill_cpu"]["total_us"], 0);
        assert!(breakdown.contains_key("ffn_gate"));
        assert!(breakdown.contains_key("ffn_up"));
        assert!(breakdown.contains_key("ffn_down"));
        assert!(breakdown.contains_key("silu_mul"));
        assert!(breakdown.contains_key("add_assign"));
    }

    #[test]
    fn test_prefill_op_profiler_cpu_fallback_tracking() {
        let mut p = PrefillOpProfiler::new();
        p.cpu_fallback_count = 3;
        p.flash_prefill_cpu = 9000;
        p.layer_count = 16;

        let json = p.to_json();
        assert_eq!(json["cpu_fallback_count"], 3);
        assert_eq!(json["breakdown"]["flash_prefill_cpu"]["total_us"], 9000);
        assert_eq!(json["breakdown"]["flash_prefill_cpu"]["avg_us"], 9000 / 16);
    }

    #[test]
    fn test_prefill_op_profiler_reset() {
        let mut p = PrefillOpProfiler::new();
        p.matmul_qkv = 5000;
        p.layer_count = 8;
        p.cpu_fallback_count = 2;

        p.reset();

        assert_eq!(p.matmul_qkv, 0);
        assert_eq!(p.layer_count, 0);
        assert_eq!(p.cpu_fallback_count, 0);
    }

    #[test]
    fn test_prefill_op_profiler_zero_layer_count_no_div_zero() {
        let mut p = PrefillOpProfiler::new();
        p.matmul_qkv = 100;
        // layer_count = 0 → avg_us should use 1 as divisor
        let json = p.to_json();
        assert_eq!(json["breakdown"]["matmul_qkv"]["avg_us"], 100);
    }
}
