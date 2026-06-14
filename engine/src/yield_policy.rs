//! Intra-token cooperative yield policy (L2).
//!
//! Env-var-driven knobs cached via [`OnceLock`] so the hot path is a
//! branch + cmp. Used by `Backend::yield_after_layer` default body to
//! flush the command queue every N layers and sleep for M microseconds.
//! This creates scheduling windows for higher-priority GPU contexts that
//! would otherwise be starved during a token's kernel chain.
//!
//! Env vars (read once via `OnceLock`):
//!
//! - `LLMRS_DECODE_YIELD_EVERY` — layer interval (0 disables, default 0).
//! - `LLMRS_DECODE_YIELD_US` — sleep microseconds (default 500).
//!
//! Promoted out of `resilience/gpu_yield.rs` (S-2 sprint 2026-05-24) to
//! L2 so that the `Backend` trait default body in `backend.rs` (L2) can
//! read it without crossing the cross-cutting boundary that
//! INV-LAYER-001/002 prohibits.

use std::sync::OnceLock;

/// Layer interval for `yield_after_layer` to fire (`0` = disabled).
#[inline]
pub fn yield_every() -> usize {
    static C: OnceLock<usize> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("LLMRS_DECODE_YIELD_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Sleep microseconds per yield (`0` = `thread::yield_now` instead of sleep).
#[inline]
pub fn yield_us() -> u64 {
    static C: OnceLock<u64> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("LLMRS_DECODE_YIELD_US")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(500)
    })
}

/// Fast check: is intra-token yield configured? Callers can skip the
/// per-layer hook entirely when this returns false.
#[inline]
pub fn intra_token_yield_enabled() -> bool {
    yield_every() > 0
}
