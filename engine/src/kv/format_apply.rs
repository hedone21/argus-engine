//! D3 — engine-side executor for [`KVFormatPlan`] (the format / precision axis).
//!
//! `apply_format_plan` is the format twin of the eviction executor `execute_kv_plan`. It applies a
//! plugin-produced [`KVFormatPlan`] to the engine's KV container.
//!
//! Honesty contract (the reason this exists rather than silently no-op'ing): a plan whose effective
//! format varies ACROSS heads or ACROSS tokens within a single layer cannot be stored by any current
//! container — [`KVCache`] holds a single dtype per layer buffer and the quant-window container a
//! single bit-width per layer — so such a plan is REJECTED with
//! [`FormatApplyError::HeterogeneousUnsupported`] instead of being mis-stored. "Expressible (a
//! well-formed plan value) != executable (the engine can re-materialize it)".
//!
//! Scope (this change): Gate-0 no-op + heterogeneous rejection + the uniform-per-layer re-encode
//! reported as not-yet-wired ([`FormatApplyError::UniformReencodeNotWired`]); the per-layer
//! re-allocation/re-encode execution (L1) is deferred. The signature takes `&KVCache` because the
//! in-scope behavior only inspects + rejects/no-ops; the L1 mutating path will promote it to `&mut`.
//!
//! DORMANCY (verified — do not mistake this for a live runtime guarantee): `apply_format_plan` has
//! **no production caller** today; it is exercised only by the unit tests below. The format-honesty
//! that production actually relies on is the *construction-time* twin
//! [`per_layer_storage_from_policy`](crate::session::bin_setup::per_layer_storage_from_policy)
//! (bin_setup.rs:565, rejecting override-bearing plans at ~:586 *before* allocation). This module is the
//! *runtime* (post-allocation) re-encode executor — kept rather than deleted so the L1-deferred shape
//! stays documented and the format axis retains a structural twin of the eviction executor
//! `execute_kv_plan` (kv::eviction::stage_registry). Until it is wired to a production caller, its
//! honesty arms are a *dormant* contract, not a runtime guarantee.

use crate::buffer::DType;
use crate::kv::kv_cache::KVCache;
use argus_extension_api::{KVFormatPlan, KeepSpec};

/// Why a [`KVFormatPlan`] could not be applied to the current container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatApplyError {
    /// The plan assigns different formats to different heads, or to a token SUBSET within one layer.
    /// No current container can hold heterogeneous-within-layer precision (one dtype per [`KVCache`]
    /// layer buffer; one bit-width per quant-window layer), so it is rejected rather than mis-stored.
    /// Faithful per-head / per-token precision needs a heterogeneous-membership store (L2).
    HeterogeneousUnsupported,
    /// The plan names a format the current backend cannot decode (g2 backend-capability feedback).
    UnsupportedFormat(String),
    /// A uniform-per-layer precision change that is well-formed, but whose execution (per-layer
    /// re-allocation + re-encode) is not yet wired (L1, deferred).
    UniformReencodeNotWired,
}

impl std::fmt::Display for FormatApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatApplyError::HeterogeneousUnsupported => write!(
                f,
                "KVFormatPlan assigns heterogeneous-within-layer precision (per-head or per-token); \
                 no current container can store it — needs a heterogeneous-membership store (L2)"
            ),
            FormatApplyError::UnsupportedFormat(name) => {
                write!(
                    f,
                    "KVFormatPlan names a format the backend cannot decode: {name}"
                )
            }
            FormatApplyError::UniformReencodeNotWired => write!(
                f,
                "uniform-per-layer precision change is well-formed but per-layer re-encode is not yet \
                 wired (L1, deferred)"
            ),
        }
    }
}

impl std::error::Error for FormatApplyError {}

/// The current stored-format name for `cache`, derived from its KV dtype (mirror of the floor's
/// `register_kv_format!` names). Used to detect the Gate-0 no-op (`base` == current format).
fn current_format_name(cache: &KVCache) -> &'static str {
    match cache.kv_dtype() {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::Q4_0 => "q4_0",
        _ => "unknown",
    }
}

/// Applies a [`KVFormatPlan`] to `cache` for one layer. See the module docs for the honesty contract.
///
/// Returns `Ok(())` only for the Gate-0 no-op (base == current stored format, no overrides). Any
/// heterogeneous-within-layer plan is rejected with [`FormatApplyError::HeterogeneousUnsupported`];
/// a uniform-per-layer change is reported as [`FormatApplyError::UniformReencodeNotWired`] (L1).
pub fn apply_format_plan(
    cache: &KVCache,
    plan: &KVFormatPlan,
    _layer: usize,
    _n_layers: usize,
) -> Result<(), FormatApplyError> {
    // Gate-0: base == current stored format AND no overrides => byte-identical no-op.
    if plan.overrides.is_empty() && plan.base.0 == current_format_name(cache) {
        return Ok(());
    }
    // Heterogeneous-within-layer? A `PerHead` override, or a `LayerWide` override that covers only a
    // token SUBSET (not the whole resident layer), assigns a different format to part of a layer —
    // unholdable by any current single-precision-per-layer container. Reject honestly.
    let resident = cache.current_pos();
    for ov in &plan.overrides {
        let heterogeneous = match &ov.region {
            KeepSpec::PerHead(_) => true,
            KeepSpec::LayerWide(positions) => positions.len() != resident,
        };
        if heterogeneous {
            return Err(FormatApplyError::HeterogeneousUnsupported);
        }
    }
    // Otherwise the plan is uniform-per-layer (a base change, or a whole-layer override). Well-formed,
    // but executing it requires per-layer re-allocation/re-encode that is not yet wired (L1, deferred).
    Err(FormatApplyError::UniformReencodeNotWired)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use argus_extension_api::{FormatId, FormatOverride, MergeAxis};
    use std::sync::Arc;

    const MAX_SEQ: usize = 32;
    const HD: usize = 4;
    const N_KV: usize = 2;

    /// An F16 KVCache with `resident` tokens written (current_pos = resident).
    fn cache_f16(resident: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let buf = || {
            Arc::new(SharedBuffer::new(
                N_KV * MAX_SEQ * HD * std::mem::size_of::<half::f16>(),
                DType::F16,
            ))
        };
        let shape = Shape::new(vec![1, MAX_SEQ, N_KV, HD]);
        let mut c = KVCache::new(
            Tensor::new(shape.clone(), buf(), backend.clone()),
            Tensor::new(shape, buf(), backend),
            MAX_SEQ,
        );
        c.set_current_pos(resident);
        c
    }

    /// Gate-0: base == current stored format + no overrides => Ok (byte-identical no-op).
    #[test]
    fn apply_format_plan_gate0_noop_ok() {
        let c = cache_f16(8);
        let plan = KVFormatPlan {
            base: FormatId("f16".into()),
            overrides: vec![],
        };
        assert_eq!(apply_format_plan(&c, &plan, 0, 1), Ok(()));
    }

    /// Per-token SUBSET override (two-tier) is heterogeneous-within-layer => rejected, not mis-stored.
    #[test]
    fn apply_format_plan_per_token_subset_rejected() {
        let c = cache_f16(8); // resident = 8, override covers only {2,3} => subset
        let plan = KVFormatPlan {
            base: FormatId("q2".into()),
            overrides: vec![FormatOverride {
                region: KeepSpec::LayerWide(vec![2, 3]),
                format: FormatId("f16".into()),
                side: MergeAxis::Both,
            }],
        };
        assert_eq!(
            apply_format_plan(&c, &plan, 0, 1),
            Err(FormatApplyError::HeterogeneousUnsupported)
        );
    }

    /// Per-head override is heterogeneous-within-layer => rejected (no per-head precision container).
    #[test]
    fn apply_format_plan_per_head_rejected() {
        let c = cache_f16(8);
        let plan = KVFormatPlan {
            base: FormatId("q2".into()),
            overrides: vec![FormatOverride {
                region: KeepSpec::PerHead(vec![vec![], vec![2]]),
                format: FormatId("f16".into()),
                side: MergeAxis::Both,
            }],
        };
        assert_eq!(
            apply_format_plan(&c, &plan, 0, 1),
            Err(FormatApplyError::HeterogeneousUnsupported)
        );
    }

    /// A uniform-per-layer change (whole-resident-layer override) is well-formed but not yet wired.
    #[test]
    fn apply_format_plan_uniform_reencode_not_wired() {
        let c = cache_f16(4); // resident = 4
        let plan = KVFormatPlan {
            base: FormatId("f16".into()),
            overrides: vec![FormatOverride {
                region: KeepSpec::LayerWide(vec![0, 1, 2, 3]), // spans the whole resident layer
                format: FormatId("q4_0".into()),
                side: MergeAxis::Both,
            }],
        };
        assert_eq!(
            apply_format_plan(&c, &plan, 0, 1),
            Err(FormatApplyError::UniformReencodeNotWired)
        );
    }

    /// Dormancy pin: a BARE per-layer base swap (no overrides, base != current stored format) is the
    /// canonical L1 re-encode request. The dormant executor must HONESTLY report it as not-yet-wired
    /// — never silently return `Ok(())` (a no-op that would falsely claim the swap happened). This is
    /// a DISTINCT code path from `apply_format_plan_uniform_reencode_not_wired` (which routes through a
    /// whole-layer override): here `overrides` is empty, so Gate-0 is the only `Ok` gate and it does
    /// NOT fire (base `q4_0` != current `f16`), the override loop is skipped, and control falls through
    /// to the `UniformReencodeNotWired` tail. Non-tautological: flipping Gate-0 to ignore the base
    /// mismatch (a plausible future "optimization") would make this return `Ok`, and this test fails.
    #[test]
    fn apply_format_plan_bare_base_change_reports_not_wired_not_silent_noop() {
        let c = cache_f16(8); // current stored format = f16
        let plan = KVFormatPlan {
            base: FormatId("q4_0".into()), // != current f16 → Gate-0 does not fire
            overrides: vec![],             // no overrides → override loop is skipped entirely
        };
        // Must be the honest not-wired report, NOT a silent Ok(()) no-op.
        assert_eq!(
            apply_format_plan(&c, &plan, 0, 1),
            Err(FormatApplyError::UniformReencodeNotWired)
        );
    }
}
