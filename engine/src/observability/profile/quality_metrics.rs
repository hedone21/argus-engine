//! Env-gated runtime overhead profiler for QCF/NLL quality metrics.
//!
//! Enabled by setting `ARGUS_PROFILE_QUALITY=1` (or `true`) in the environment.
//! When disabled, every instrumentation site costs only one branch on a cached
//! `OnceLock<bool>` (~1 ns).
//!
//! Usage at a measurement site:
//! ```ignore
//! use crate::observability::profile::quality_metrics::{Timer, QCF_KV_UNIFIED};
//! pub fn compute_qcf_kv(...) -> ... {
//!     let _t = Timer::start(&QCF_KV_UNIFIED);
//!     // ... function body ...
//! }
//! ```
//!
//! Call `print_report()` once near process exit to flush a stderr summary.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

const ENV_VAR: &str = "ARGUS_PROFILE_QUALITY";

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Returns true iff the env var requests profiling. Result is cached.
#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var(ENV_VAR)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Lock-free counter for one measurement category.
pub struct Bucket {
    pub name: &'static str,
    total_ns: AtomicU64,
    calls: AtomicU64,
}

impl Bucket {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            total_ns: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        }
    }

    fn record(&self, ns: u64) {
        self.total_ns.fetch_add(ns, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot `(total_ns, calls)`.
    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.total_ns.load(Ordering::Relaxed),
            self.calls.load(Ordering::Relaxed),
        )
    }
}

/// RAII timer that records elapsed nanoseconds into a [`Bucket`] on drop.
///
/// When profiling is disabled, both fields stay `None` and `Drop` is a no-op.
pub struct Timer<'a> {
    bucket_and_start: Option<(&'a Bucket, Instant)>,
}

impl<'a> Timer<'a> {
    #[inline]
    pub fn start(bucket: &'a Bucket) -> Self {
        if enabled() {
            Self {
                bucket_and_start: Some((bucket, Instant::now())),
            }
        } else {
            Self {
                bucket_and_start: None,
            }
        }
    }
}

impl Drop for Timer<'_> {
    #[inline]
    fn drop(&mut self) {
        if let Some((bucket, t0)) = self.bucket_and_start {
            bucket.record(t0.elapsed().as_nanos() as u64);
        }
    }
}

// ── Buckets (process-lifetime) ───────────────────────────────────

/// Output-error QCF for KV cache eviction/merge actions.
pub static QCF_KV_UNIFIED: Bucket = Bucket::new("qcf_kv_unified");
/// KIVI dynamic-quantization dry-run estimate (residual NMSE or bits proxy).
pub static QCF_KV_DRYRUN: Bucket = Bucket::new("qcf_kv_dryrun");
/// Weight-swap QCF (per-ratio sweep + actual-swap measurement).
pub static QCF_WEIGHT_SWAP: Bucket = Bucket::new("qcf_weight_swap");
/// Layer-skip QCF (importance-table based).
pub static QCF_LAYER_SKIP: Bucket = Bucket::new("qcf_layer_skip");
/// Numerically stable log-softmax for eval-LL token scoring.
pub static NLL: Bucket = Bucket::new("nll");
/// Per-token decode (forward + sample) baseline.
pub static DECODE_TOTAL: Bucket = Bucket::new("decode_total");

const ALL_BUCKETS: &[&Bucket] = &[
    &QCF_KV_UNIFIED,
    &QCF_KV_DRYRUN,
    &QCF_WEIGHT_SWAP,
    &QCF_LAYER_SKIP,
    &NLL,
    &DECODE_TOTAL,
];

/// Print a stderr summary of all buckets. No-op when profiling is disabled.
pub fn print_report() {
    if !enabled() {
        return;
    }

    eprintln!("[QualityProfile] {}=1", ENV_VAR);

    let mut quality_total_ns: u64 = 0;
    let (decode_ns, decode_calls) = DECODE_TOTAL.snapshot();

    for b in ALL_BUCKETS {
        let (ns, calls) = b.snapshot();
        if calls == 0 {
            continue;
        }
        let total_ms = ns as f64 / 1.0e6;
        let avg_us = ns as f64 / calls as f64 / 1.0e3;
        eprintln!(
            "  {:<18} {:>8} calls  {:>10.3} ms total  {:>10.3} µs/call",
            b.name, calls, total_ms, avg_us
        );
        if !std::ptr::eq(*b, &DECODE_TOTAL) {
            quality_total_ns += ns;
        }
    }

    if decode_calls > 0 && decode_ns > 0 {
        let pct = quality_total_ns as f64 / decode_ns as f64 * 100.0;
        eprintln!(
            "  ─ quality_overhead = {:.3} ms / {:.3} ms = {:.2}% of decode_total",
            quality_total_ns as f64 / 1.0e6,
            decode_ns as f64 / 1.0e6,
            pct
        );
    }
}

/// Register an `atexit` hook that calls [`print_report`] once at process exit.
/// Idempotent and a no-op when profiling is disabled.
pub fn install_atexit_once() {
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::Relaxed) {
        return;
    }
    if !enabled() {
        return;
    }
    #[cfg(unix)]
    extern "C" fn cleanup() {
        print_report();
    }
    #[cfg(unix)]
    unsafe {
        libc::atexit(cleanup);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_record_accumulates() {
        let b = Bucket::new("test_bucket");
        b.record(1000);
        b.record(2000);
        let (ns, calls) = b.snapshot();
        assert_eq!(ns, 3000);
        assert_eq!(calls, 2);
    }

    #[test]
    fn test_timer_disabled_no_record() {
        // The static ENABLED is set by env at first call; we cannot easily
        // toggle it inside a test. Instead, exercise the Bucket API directly
        // and confirm Drop with None is harmless.
        let b = Bucket::new("noop");
        {
            let _t = Timer {
                bucket_and_start: None,
            };
        }
        let (ns, calls) = b.snapshot();
        assert_eq!(ns, 0);
        assert_eq!(calls, 0);
    }

    #[test]
    fn test_timer_records_when_some() {
        let b = Box::leak(Box::new(Bucket::new("forced")));
        {
            let _t = Timer {
                bucket_and_start: Some((b, Instant::now())),
            };
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        let (ns, calls) = b.snapshot();
        assert_eq!(calls, 1);
        assert!(ns >= 50_000, "expected ≥ 50µs recorded, got {ns} ns");
    }
}
