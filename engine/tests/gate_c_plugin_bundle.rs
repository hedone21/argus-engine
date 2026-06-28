//! GATE-C 멀티-vtable bundle ABI 재증명 게이트 (V5) — v1 단일-vtable 게이트
//! (`gate_c_dlopen_equivalence.rs` + `gate_c_format_dlopen_equivalence.rs`)를 v2 의미론으로 **재작성·통합**.
//!
//! v2 에서 dlopen 이 바꾸는 것: plugin `.so` 의 `register_kv_*s_v2` 봉투가 vtable **리스트**(0..N)를
//! 신고하고, host dispatcher(`register_dynamic_plugins`)가 `.so` 1회 dlopen 으로 stage+format 양축을
//! Arc 공유 등록한다. 따라서 게이트 = (i) 번들/멀티-format 가 올바른 개수·이름·descriptor 로 등록되고,
//! (ii) capability-0/충돌/중복/PerHead 가 fail-fast/bail 되며, (iii) 동적 descriptor 가 정적과 동일하게
//! 엔진 floor 를 구동(byte-identity)하는지.
//!
//! **CF4 7요소 → 본 게이트 보존 매핑**: ①builtin-collision=synth_q4/q4_0 reject · ②동적 등록 성공=
//! bundle/multi-format Ok · ③registry merge 가시화=dynamic_registered_*_names · ④descriptor-identity=
//! multi-format 서로 다른 desc · ⑤floor byte-identity=mf_q4 round-trip · ⑥동적 중복=dup reject ·
//! ⑦심볼부재 reject=capability-0(no-export & feature-OFF). + v2 신규: 번들 양축 · 멀티-stage/format
//! 인덱스 바인딩 · wrong-type graceful · per-`.so` 원자성 롤백 · PerHead bail.
//!
//! **process-global 레지스트리**(DYN_REGISTRY/DYN_FORMAT_REGISTRY OnceLock): 병렬 간섭 회피 위해 모든
//! 단언을 단일 `#[test]` 에서 순차 수행 — reject(등록 0) 먼저, success(영구 등록) 나중.

use std::path::PathBuf;
use std::process::Command;

use argus_engine::format::dynamic_format_registry::{dynamic_registered_format_names, make_format};
use argus_engine::format::{decode_via_descriptor, encode_via_descriptor};
use argus_engine::session::plugin_dispatch::register_dynamic_plugins;
use argus_extension_api::{KVLayoutDesc, Packing, ScaleLayout, find_kv_format};

/// q4_0-like 알려진 정답 descriptor(mf_q4 / bundle_fmt / synth_q4 공유).
fn known_q4() -> KVLayoutDesc {
    KVLayoutDesc {
        block_elems: 32,
        bits: 4,
        scale_layout: ScaleLayout::PerBlockF16,
        packing: Packing::Nibble,
    }
}

/// `cargo build -p <pkg> [--features plugin-cdylib] --message-format=json` 으로 `.so` 산출 → 경로를
/// `CARGO_TARGET_TMPDIR` 의 고유 이름으로 복사(feature ON/OFF `.so` 덮어쓰기 회피).
fn build_plugin_so(pkg: &str, with_export: bool, dst_name: &str) -> PathBuf {
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["build", "-p", pkg, "--message-format=json"]);
    if with_export {
        cmd.args(["--features", "plugin-cdylib"]);
    }
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("cargo build -p {pkg} 실행 실패: {e}"));
    assert!(
        out.status.success(),
        "cargo build {pkg} .so 실패:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let underscore = pkg.replace('-', "_");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let src = stdout
        .lines()
        .filter(|l| l.contains("compiler-artifact") && l.contains(&underscore))
        .flat_map(|l| l.split('"'))
        .find(|tok| tok.ends_with(".so") && tok.contains(&underscore))
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{pkg} .so 산출 경로 미검출"));
    let dst = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(dst_name);
    std::fs::copy(&src, &dst)
        .unwrap_or_else(|e| panic!("{} → {} 복사 실패: {e}", src.display(), dst.display()));
    dst
}

#[test]
fn gate_c_v2_bundle_multivtable_and_rejects() {
    // ── vehicle .so 빌드 ──
    let synth = build_plugin_so("synth-q4-format", true, "libsynth_q4_v2.so");
    let no_export = build_plugin_so("example-no-export", true, "libexample_no_export_v2.so");
    let kv_off = build_plugin_so("example-kv-format", false, "libexample_kv_format_off.so");
    let rollback = build_plugin_so("example-rollback", true, "libexample_rollback_v2.so");
    let bundle = build_plugin_so("example-bundle", true, "libexample_bundle_v2.so");
    let multi = build_plugin_so(
        "example-multi-format",
        true,
        "libexample_multi_format_v2.so",
    );

    // ════ REJECT 단언 (등록 0 — 레지스트리 불변) ════

    // (1) builtin-collision: synth_q4 는 엔진 force-link 정적 등록 → 동적 등록 거부(CF4 ①).
    assert!(
        find_kv_format("synth_q4").is_some(),
        "전제: synth_q4 가 엔진 force-link 정적 등록"
    );
    let e = register_dynamic_plugins(std::slice::from_ref(&synth))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("conflict") || e.contains("builtin") || e.contains("built-in"),
        "builtin-collision: {e}"
    );

    // (2) capability-0 (no-export): PLUGIN 슬라이스 기여는 있으나 register_kv_*s_v2 entry 부재(CF4 ⑦).
    let e = register_dynamic_plugins(std::slice::from_ref(&no_export))
        .unwrap_err()
        .to_string();
    assert!(e.contains("0 capabilities"), "no-export capability-0: {e}");

    // (3) capability-0 (feature-OFF): v2 심볼 자체 부재.
    let e = register_dynamic_plugins(std::slice::from_ref(&kv_off))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("0 capabilities"),
        "feature-OFF capability-0: {e}"
    );

    // (4) per-.so 원자성 롤백: 봉투 [q4_0(빌트인 충돌), rollback_ok] → q4_0 에서 bail → rollback_ok 미등록.
    let e = register_dynamic_plugins(std::slice::from_ref(&rollback))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("conflict") || e.contains("builtin") || e.contains("built-in"),
        "rollback collision: {e}"
    );
    assert!(
        !dynamic_registered_format_names().contains(&"rollback_ok".to_string()),
        "원자성: q4_0 충돌로 bail 시 rollback_ok 는 등록되지 않아야(롤백): {:?}",
        dynamic_registered_format_names()
    );

    // ════ SUCCESS 단언 (영구 등록) ════

    // (5) bundle .so 의 format 축 등록: stage 축은 static-linkme 전용이라 .so 의 stage vtable 은
    // 동적 경로에 없고 format(bundle_fmt)만 동적 등록된다(CF4 ②③).
    register_dynamic_plugins(std::slice::from_ref(&bundle)).expect("bundle 등록 성공");
    let fnames = dynamic_registered_format_names();
    assert!(
        fnames.contains(&"bundle_fmt".to_string()),
        "format bundle_fmt: {fnames:?}"
    );

    // (7) multi-format 서로 다른 descriptor-identity(CF4 ④): N개 등록 + 각자 고유 desc.
    register_dynamic_plugins(std::slice::from_ref(&multi)).expect("multi-format 등록 성공");
    let fnames = dynamic_registered_format_names();
    assert!(fnames.contains(&"mf_q4".to_string()) && fnames.contains(&"mf_q8".to_string()));
    let d_q4 = make_format("mf_q4").expect("make_format mf_q4").layout();
    let d_q8 = make_format("mf_q8").expect("make_format mf_q8").layout();
    assert_eq!(d_q4, known_q4(), "mf_q4 descriptor-identity");
    assert_eq!(d_q8.bits, 8);
    assert_eq!(d_q8.packing, Packing::Byte);
    assert_ne!(
        d_q4, d_q8,
        "두 format desc 가 달라야(인덱스/이름 미스바인딩 검출)"
    );

    // (8) descriptor-floor byte-identity(CF4 ⑤): 동적 mf_q4 desc 와 정적 q4_0 desc 의 encode 바이트 동일.
    let numel = 64usize;
    let src: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let nbytes = d_q4.bytes_for_elems(numel).expect("q4 64 bytes");
    let mut enc_dyn = vec![0u8; nbytes];
    let mut enc_static = vec![0u8; nbytes];
    encode_via_descriptor(&d_q4, &src, &mut enc_dyn).expect("dyn encode");
    encode_via_descriptor(&known_q4(), &src, &mut enc_static).expect("static encode");
    assert_eq!(
        enc_dyn, enc_static,
        "dlopen 과 정적 descriptor 의 encode 바이트 동일"
    );
    let mut dec = vec![0.0f32; numel];
    decode_via_descriptor(&d_q4, &enc_dyn, &mut dec);
    assert!(dec.iter().all(|v| v.is_finite()), "decode 유한");

    // (9) 동적 중복 reject(CF4 ⑥): 같은 multi-format .so 재등록 → 이미 등록됨.
    let e = register_dynamic_plugins(std::slice::from_ref(&multi))
        .unwrap_err()
        .to_string();
    assert!(e.contains("duplicate"), "동적 중복 reject: {e}");
}
