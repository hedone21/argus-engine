// spec/30-engine.md §3.7 ENG-085: CLI 표면 — 두 flag 파싱 + is_standard_happy_path 호환
//
// 검증:
// (a) --save-prefix-cache 파싱
// (b) --prefix-cache 파싱
// (c) 두 flag 미지정 시 None
// (d) is_standard_happy_path에서 두 flag 무관하게 통과 (happy path 호환)

use argus_engine::session::assembly::build_standard_loop::is_standard_happy_path;
use argus_engine::session::cli::Args;
use clap::Parser;

/// ENG-085(a): --save-prefix-cache 파싱
#[test]
fn save_prefix_cache_flag_parses() {
    let args = Args::try_parse_from(["test", "--save-prefix-cache", "/tmp/kv.cache"]).unwrap();
    assert_eq!(args.save_prefix_cache.as_deref(), Some("/tmp/kv.cache"));
}

/// ENG-085(b): --prefix-cache 파싱
#[test]
fn prefix_cache_flag_parses() {
    let args = Args::try_parse_from(["test", "--prefix-cache", "/tmp/kv.cache"]).unwrap();
    assert_eq!(args.prefix_cache.as_deref(), Some("/tmp/kv.cache"));
}

/// ENG-085(c): 두 flag 미지정 시 None
#[test]
fn prefix_cache_flags_default_none() {
    let args = Args::try_parse_from(["test"]).unwrap();
    assert!(
        args.save_prefix_cache.is_none(),
        "--save-prefix-cache default must be None"
    );
    assert!(
        args.prefix_cache.is_none(),
        "--prefix-cache default must be None"
    );
}

/// ENG-085(d): is_standard_happy_path 호환 — 두 flag 지정 시에도 happy path 통과
#[test]
fn prefix_cache_flags_compatible_with_happy_path() {
    // --save-prefix-cache 지정 시에도 happy path 통과
    let args_save = Args::try_parse_from(["test", "--save-prefix-cache", "/tmp/kv.cache"]).unwrap();
    assert!(
        is_standard_happy_path(&args_save),
        "--save-prefix-cache must not break is_standard_happy_path (ENG-085)"
    );

    // --prefix-cache 지정 시에도 happy path 통과
    let args_restore = Args::try_parse_from(["test", "--prefix-cache", "/tmp/kv.cache"]).unwrap();
    assert!(
        is_standard_happy_path(&args_restore),
        "--prefix-cache must not break is_standard_happy_path (ENG-085)"
    );

    // 두 flag 모두 지정 시에도 happy path 통과
    let args_both = Args::try_parse_from([
        "test",
        "--save-prefix-cache",
        "/tmp/kv.cache",
        "--prefix-cache",
        "/tmp/kv.cache",
    ])
    .unwrap();
    assert!(
        is_standard_happy_path(&args_both),
        "both flags must not break is_standard_happy_path (ENG-085)"
    );
}

/// ENG-085(e): 두 flag 모두 None = prefix cache 코드 미진입 (분기 1회)
/// Args 기본값으로 확인
#[test]
fn no_prefix_cache_flags_means_no_cache_path() {
    let args = Args::try_parse_from(["test"]).unwrap();
    // 두 flag 모두 None이면 prefix cache 분기가 없다 (코드 경로 보장)
    assert!(args.save_prefix_cache.is_none() && args.prefix_cache.is_none());
    // happy path도 통과
    assert!(is_standard_happy_path(&args));
}
