//! ENG-DAT-094 — `SecondaryMmap` unit checks.
//!
//! Phase 1 scope: exercise the pure-Rust helpers (tensor-name parser, Debug
//! output) and the LoadError machinery. End-to-end `open_secondary` against
//! real GGUF pairs requires committed fixture files and is covered by the
//! integration test `test_eng_alg_210_initial_load.rs` (skipped when the
//! fixtures are absent).

use argus_engine::models::weights::LoadError;

#[test]
fn load_error_display_metadata_mismatch() {
    let err = LoadError::MetadataMismatch {
        field: "hidden_size",
        primary: "2048".into(),
        secondary: "1024".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("hidden_size"));
    assert!(msg.contains("2048"));
    assert!(msg.contains("1024"));
}

#[test]
fn load_error_display_shape_mismatch() {
    let err = LoadError::ShapeMismatch {
        name: "blk.0.attn_q.weight".into(),
        primary: vec![2048, 2048],
        secondary: vec![2048, 1024],
    };
    let msg = format!("{err}");
    assert!(msg.contains("blk.0.attn_q.weight"));
}

#[test]
fn load_error_display_missing_tensor() {
    let err = LoadError::TensorMissing {
        name: "blk.3.ffn_down.weight".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("blk.3.ffn_down.weight"));
    assert!(msg.contains("missing"));
}

#[test]
fn load_error_source_chain_set_for_unavailable() {
    use std::error::Error;

    let inner = anyhow::anyhow!("file not found: /tmp/missing.gguf");
    let err = LoadError::SecondaryUnavailable {
        path: "/tmp/missing.gguf".into(),
        source: inner,
    };
    assert!(err.source().is_some());

    // Display contains the path plus the inner message.
    let msg = format!("{err}");
    assert!(msg.contains("/tmp/missing.gguf"));
}
