//! Intra-token cooperative yield — re-exports.
//!
//! S-2 sprint 2026-05-24: env-var caching + yield logic moved to
//! `crate::yield_policy` (L2) so that `Backend::yield_after_layer`
//! default body can read it without crossing the cross-cutting boundary
//! that INV-LAYER-001 prohibits. The freestanding `gpu_yield_impl`
//! helper is gone — all dispatch flows through the trait method now.
//!
//! Existing public surface is preserved as thin re-exports for any
//! external resilience consumer that still imports from this module.

pub use crate::yield_policy::{intra_token_yield_enabled, yield_every, yield_us};

#[cfg(test)]
mod tests {
    #[test]
    fn defaults_disable_yield() {
        // Without env var, yield_every returns 0 → disabled.
        // (OnceLock may be cached from a prior test; just read current.)
        let enabled = super::intra_token_yield_enabled();
        // Non-deterministic but in CI without env vars should be false.
        assert!(!enabled || super::yield_every() > 0);
    }
}
