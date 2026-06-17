//! Spec test for `--kv-mode` (FORMAT-axis mode/knob declaration: the closed clap
//! `KvMode` enum is gone — `--kv-mode` is a runtime String resolved against the
//! engine KV-mode registry. `--kv-mode kivi` must still parse and resolve).

use argus_engine::session::cli::Args;
use argus_engine::session::mode::{mode_caps, resolve_kv_mode};
use clap::Parser;

fn parse(argv: &[&str]) -> Args {
    let mut full = vec!["generate"];
    full.extend_from_slice(argv);
    Args::try_parse_from(full).expect("parse failed")
}

#[test]
fn default_is_standard() {
    let args = parse(&["--model-path", "/tmp/x.gguf", "--prompt", "hi"]);
    assert_eq!(args.kv_mode_args.kv_mode, "standard");
    assert_eq!(args.effective_kv_mode(), "standard");
    // default name resolves to a registered mode with non-quantized caps.
    assert!(resolve_kv_mode(args.effective_kv_mode()).is_some());
    assert!(!mode_caps("standard").unwrap().is_quantized_kv);
}

#[test]
fn explicit_kivi_parses() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "kivi",
        "--kivi-bits",
        "4",
        "--kivi-residual-len",
        "64",
    ]);
    assert_eq!(args.effective_kv_mode(), "kivi");
    // `--kv-mode kivi` must still resolve against the registry, with quantized caps.
    assert!(resolve_kv_mode("kivi").is_some());
    assert!(mode_caps("kivi").unwrap().is_quantized_kv);
    assert_eq!(args.kv_mode_args.kivi_bits, 4);
    assert_eq!(args.kv_mode_args.kivi_residual_len, 64);
}

#[test]
fn explicit_offload_parses() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "offload",
        "--kv-offload-storage",
        "tmpfs",
        "--kv-offload-path",
        "/tmp/kv",
        "--kv-max-prefetch-depth",
        "4",
    ]);
    assert_eq!(args.effective_kv_mode(), "offload");
    assert!(mode_caps("offload").unwrap().supports_offload);
    assert_eq!(args.kv_mode_args.kv_offload_storage, "tmpfs");
    assert_eq!(args.kv_mode_args.kv_offload_path, "/tmp/kv");
    assert_eq!(args.kv_mode_args.kv_max_prefetch_depth, 4);
}

#[test]
fn unknown_kv_mode_parses_but_does_not_resolve() {
    // String arg parses any value (clap can't enumerate at compile time); the build
    // funnel fail-fasts on resolve. Pin that an unknown name has no registry entry.
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "bogus",
    ]);
    assert_eq!(args.effective_kv_mode(), "bogus");
    assert!(resolve_kv_mode("bogus").is_none());
}

#[test]
fn effective_kivi_bits_reads_new_field() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "kivi",
        "--kivi-bits",
        "4",
    ]);
    assert_eq!(args.effective_kivi_bits(), 4);
}

#[test]
fn effective_kivi_residual_size_reads_new_field() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "kivi",
        "--kivi-residual-len",
        "64",
    ]);
    assert_eq!(args.effective_kivi_residual_size(), 64);
}

#[test]
fn effective_kv_offload_storage_reads_new_field_when_offload() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "offload",
        "--kv-offload-storage",
        "tmpfs",
    ]);
    assert_eq!(args.effective_kv_offload_storage(), "tmpfs");
}

#[test]
fn effective_kv_offload_storage_empty_when_not_offload() {
    let args = parse(&["--model-path", "/tmp/x.gguf", "--prompt", "hi"]);
    assert_eq!(args.effective_kv_offload_storage(), "");
}

#[test]
fn legacy_kivi_flag_no_longer_parses() {
    // 옵션 C 후: `--kivi` 같은 legacy flag는 clap parse error.
    let result = Args::try_parse_from([
        "generate",
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kivi",
    ]);
    assert!(result.is_err(), "legacy --kivi must error after 옵션 C");
}

#[test]
fn legacy_kv_offload_flag_no_longer_parses() {
    let result = Args::try_parse_from([
        "generate",
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-offload",
        "mmap",
    ]);
    assert!(
        result.is_err(),
        "legacy --kv-offload must error after 옵션 C"
    );
}
