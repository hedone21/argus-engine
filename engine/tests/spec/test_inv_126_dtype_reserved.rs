//! INV-126 — `DtypeTag` reserved variant rejection.
//!
//! `dtype_tag_to_dtype(tag)` must return `Err(SwapError::UnsupportedDtype)`
//! for any `DtypeTag` other than `Q4_0`.  Callers (dispatch_swap_weights) rely
//! on this function to gate the execution path — a reserved dtype must never
//! reach `SwapExecutor::execute`.
//!
//! Spec: INV-126, MSG-082, ENG-ALG-214-ROUTE.

use argus_engine::weight::swap_executor::{SwapError, dtype_tag_to_dtype};
use argus_shared::DtypeTag;

// ── INV-126: Only Q4_0 is executable in Phase 3 ──────────────────────────────

/// `DtypeTag::Q4_0` is the sole executable variant — must succeed.
#[test]
fn inv_126_q4_0_is_accepted() {
    let result = dtype_tag_to_dtype(DtypeTag::Q4_0);
    assert!(result.is_ok(), "Q4_0 must be accepted, got {result:?}");
}

/// `DtypeTag::F16` is reserved — must be rejected without panic.
#[test]
fn inv_126_f16_is_rejected() {
    let result = dtype_tag_to_dtype(DtypeTag::F16);
    assert!(
        matches!(result, Err(SwapError::UnsupportedDtype(DtypeTag::F16))),
        "F16 must produce UnsupportedDtype, got {result:?}"
    );
}

/// `DtypeTag::F32` is reserved — must be rejected without panic.
#[test]
fn inv_126_f32_is_rejected() {
    let result = dtype_tag_to_dtype(DtypeTag::F32);
    assert!(
        matches!(result, Err(SwapError::UnsupportedDtype(DtypeTag::F32))),
        "F32 must produce UnsupportedDtype, got {result:?}"
    );
}

/// `DtypeTag::Q8_0` is reserved — must be rejected without panic.
#[test]
fn inv_126_q8_0_is_rejected() {
    let result = dtype_tag_to_dtype(DtypeTag::Q8_0);
    assert!(
        matches!(result, Err(SwapError::UnsupportedDtype(DtypeTag::Q8_0))),
        "Q8_0 must produce UnsupportedDtype, got {result:?}"
    );
}

/// Rejection carries the original tag so callers can log it (MSG-082 wire-stability).
#[test]
fn inv_126_rejection_preserves_tag_for_logging() {
    let tag = DtypeTag::F16;
    let err = dtype_tag_to_dtype(tag).unwrap_err();
    match err {
        SwapError::UnsupportedDtype(t) => {
            assert_eq!(t, DtypeTag::F16, "rejected tag must be preserved in error");
        }
        other => panic!("Expected UnsupportedDtype, got {other:?}"),
    }
}

/// All reserved variants are enumerated — each must fail, only Q4_0 succeeds.
#[test]
fn inv_126_exhaustive_reserved_variants() {
    let reserved = [DtypeTag::F16, DtypeTag::F32, DtypeTag::Q8_0];
    for tag in reserved {
        let result = dtype_tag_to_dtype(tag);
        assert!(
            result.is_err(),
            "DtypeTag {tag:?} must be rejected (INV-126)"
        );
    }
    // Q4_0 is the only valid one
    assert!(dtype_tag_to_dtype(DtypeTag::Q4_0).is_ok());
}
