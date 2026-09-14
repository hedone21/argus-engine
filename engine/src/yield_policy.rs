//! Intra-token cooperative yield policy (L2).
//!
//! Env-var-seeded, runtime-settable knob cached in an [`AtomicUsize`] so the hot path is
//! a load + branch + cmp. Used by `Backend::yield_after_layer` default body to flush the
//! command queue every N layers and sleep for M microseconds. This creates scheduling
//! windows for higher-priority GPU contexts that would otherwise be starved during a
//! token's kernel chain.
//!
//! `yield_every` starts at an env seed but can be changed for the life of the process via
//! [`set_yield_every`] (tickets/015: a Manager `gpu.share` command translates its intent
//! into a rung through [`every_for_share`] and calls the setter). [`restore_default_yield_every`]
//! puts it back to the env seed.
//!
//! Env vars (each read once, to seed the atomic and to seed `yield_us`):
//!
//! - `LLMRS_DECODE_YIELD_EVERY` — layer interval (0 disables, default 0).
//! - `LLMRS_DECODE_YIELD_US` — sleep microseconds (default 0 — `thread::yield_now`
//!   instead of a sleep; tickets/015 T0-3: the dynamic arm is switched on by
//!   [`set_yield_every`] rather than env, so a nonzero default would silently reintroduce
//!   the sleep-bearing arm 012 rejected).
//!
//! Promoted out of `resilience/gpu_yield.rs` (S-2 sprint 2026-05-24) to
//! L2 so that the `Backend` trait default body in `backend.rs` (L2) can
//! read it without crossing the cross-cutting boundary that
//! INV-LAYER-001/002 prohibits.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The env seed for `yield_every`, read once. Also what [`restore_default_yield_every`]
/// restores to.
fn env_seed_every() -> usize {
    static SEED: OnceLock<usize> = OnceLock::new();
    *SEED.get_or_init(|| {
        std::env::var("LLMRS_DECODE_YIELD_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// The mutable cell backing `yield_every`, seeded from the env on first touch.
fn every_cell() -> &'static AtomicUsize {
    static C: OnceLock<AtomicUsize> = OnceLock::new();
    C.get_or_init(|| AtomicUsize::new(env_seed_every()))
}

/// Layer interval for `yield_after_layer` to fire (`0` = disabled).
#[inline]
pub fn yield_every() -> usize {
    every_cell().load(Ordering::Relaxed)
}

/// Set the layer interval at runtime, independent of the env seed. Idempotent; the last
/// call before a read wins, and nothing here fences against the read side — `yield_after_layer`
/// tolerates a stale value for at most one layer's worth of staleness.
pub fn set_yield_every(every: usize) {
    every_cell().store(every, Ordering::Relaxed);
}

/// Put `yield_every` back to what the process started with.
pub fn restore_default_yield_every() {
    every_cell().store(env_seed_every(), Ordering::Relaxed);
}

/// Translate a `gpu.share` intent (`foreground` in `[0.0, 1.0]`) into a layer interval.
///
/// Two rungs only — `0` (off) and `2` (012's `e2s0` arm) — because 012 measured exactly
/// three points (`EVERY ∈ {0, 8, 2}`) and `EVERY=8` fell short of its own `2D` bar
/// (tickets/012 §Result). Interpolating a value 012 never measured is out of scope
/// (tickets/015 §3): add a rung only after measuring it.
#[inline]
pub fn every_for_share(foreground: f32) -> usize {
    if foreground > 0.0 { 2 } else { 0 }
}

/// Sleep microseconds per yield (`0` = `thread::yield_now` instead of sleep).
#[inline]
pub fn yield_us() -> u64 {
    static C: OnceLock<u64> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("LLMRS_DECODE_YIELD_US")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Fast check: is intra-token yield configured? Callers can skip the
/// per-layer hook entirely when this returns false.
#[inline]
pub fn intra_token_yield_enabled() -> bool {
    yield_every() > 0
}

/// Serializes tests that mutate the process-global yield state. Shared beyond this
/// module: `command_dispatcher`'s `gpu_share_sets_the_yield_and_restore_releases_it` test
/// drives the same statics through `set_yield_every`/`restore_default_yield_every` and
/// must not interleave with the tests below.
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutation-proof rather than process-environment-mutating: memory
    /// `verification-guard-limits` forbids tests that poke the process environment,
    /// since a `OnceLock`-seeded value cannot observe a post-init change there anyway.
    #[test]
    fn setter_changes_what_the_hook_reads() {
        let _g = TEST_LOCK.lock().unwrap();
        set_yield_every(2);
        assert_eq!(yield_every(), 2);
        restore_default_yield_every();
    }

    #[test]
    fn round_trip_zero_two_zero() {
        let _g = TEST_LOCK.lock().unwrap();
        set_yield_every(0);
        assert_eq!(yield_every(), 0);
        set_yield_every(2);
        assert_eq!(yield_every(), 2);
        set_yield_every(0);
        assert_eq!(yield_every(), 0);
        assert!(!intra_token_yield_enabled());
    }

    #[test]
    fn default_is_the_env_seed_without_a_setter() {
        let _g = TEST_LOCK.lock().unwrap();
        // The test process runs with no LLMRS_DECODE_YIELD_EVERY set.
        assert_eq!(yield_every(), 0);
        restore_default_yield_every();
        assert_eq!(yield_every(), 0);
    }

    #[test]
    fn share_ladder_has_two_rungs() {
        assert_eq!(every_for_share(0.0), 0);
        assert_eq!(every_for_share(0.01), 2);
        assert_eq!(every_for_share(0.5), 2);
        assert_eq!(every_for_share(1.0), 2);
    }

    #[test]
    fn yield_us_defaults_to_zero() {
        // The test process runs with no LLMRS_DECODE_YIELD_US set.
        assert_eq!(yield_us(), 0);
    }
}
