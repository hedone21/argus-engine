//! INV-LAYER-007: `session::DecodeLoopBuilder` typestate enforces that
//! `.build()` is only callable when a `Forward` has been supplied.
//!
//! Negative test: `compile_fail/forward_missing.rs` — `Builder::new().build()`
//! must fail to compile (no `build()` method on `DecodeLoopBuilder<NoForward>`).
//!
//! Positive test: `compile_pass/forward_minimal.rs` — a Forward implementor
//! that overrides only `prefill` + `step` (relying on default `finalize` and
//! `on_kv_prune`) must compile.
//!
//! Tooling: `trybuild` (dev-dep). Stderr snapshots are written on first run
//! via `TRYBUILD=overwrite cargo test`. Re-generate after rustc upgrades.

#[test]
fn typestate_negative_forward_missing() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/spec/compile_fail/forward_missing.rs");
    // trybuild panics on mismatch; this assert documents intent for coverage tooling.
    assert!(
        std::path::Path::new("tests/spec/compile_fail/forward_missing.rs").exists(),
        "INV-LAYER-007: compile_fail fixture must exist"
    );
}

#[test]
fn typestate_positive_forward_minimal() {
    let t = trybuild::TestCases::new();
    t.pass("tests/spec/compile_pass/forward_minimal.rs");
    // Phase 4-3 C2: ModelForward concrete impl also satisfies the
    // `HasForward` typestate (i.e. `Builder::with_forward(ModelForward)`
    // reaches `.build()`). Bundled here so a single trybuild invocation
    // covers both fixtures.
    t.pass("tests/spec/compile_pass/model_forward_minimal.rs");
    assert!(
        std::path::Path::new("tests/spec/compile_pass/forward_minimal.rs").exists(),
        "INV-LAYER-007: compile_pass fixture must exist"
    );
    assert!(
        std::path::Path::new("tests/spec/compile_pass/model_forward_minimal.rs").exists(),
        "Phase 4-3 C2: model_forward compile_pass fixture must exist"
    );
}
