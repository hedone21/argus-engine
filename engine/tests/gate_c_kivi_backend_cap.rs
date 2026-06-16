//! GATE-C (kivi) — Backend 축 capability dlopen 재증명 게이트, `kivi` 플러그인 짝
//! (design D2/D7/D8, CB5). `gate_c_backend_cap_dlopen.rs`(synthetic `example-backend-cap`)의
//! 자매 게이트로, 첫 *비-example* backend-cap 플러그인 `kivi`(이름 `kivi_abi`)가 ABI 경계
//! (register_backend_caps_v2 봉투 → category 다리 → `DynKiviAttentionBackend` 어댑터 → make/dispatch)를
//! **정확히** 넘는지 증명한다.
//!
//! **host-검증 범위(C12)**: `kivi` 플러그인은 GPU 수학을 하지 않는다 — `KiviAttnArgs` 스칼라로 결정적
//! sentinel 을 계산해 `scores_out[0]` 에 기록하므로, host 가 args struct 의 필드 정렬·값이 ABI 경계를
//! 정확히 넘었는지 확인할 수 있다. 실제 KIVI 커널 실행은 본 게이트 밖(엔진 잔류, S25 device C12 영역).
//!
//! **process-global 레지스트리**(DYN_BACKEND_REGISTRY OnceLock): 단일 `#[test]` 에서 순차 수행.

use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;

use argus_engine::capability::dynamic_backend_registry::{
    dynamic_registered_backend_cap_names, resolve_kivi_capability,
};
use argus_engine::session::plugin_dispatch::register_dynamic_plugins;
use argus_extension_api::{KiviAttnArgs, KiviGatherArgs, KiviMakeArgs};

/// `cargo build -p <pkg> [--features plugin-cdylib] --message-format=json` 으로 `.so` 산출 → 경로를
/// `CARGO_TARGET_TMPDIR` 의 고유 이름으로 복사. (`gate_c_backend_cap_dlopen.rs` 의 헬퍼와 동일.)
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
fn gate_c_kivi_backend_cap_dlopen_round_trip() {
    // ── 1. kivi backend-cap .so 빌드 (cdylib + C-ABI export). ──
    let so = build_plugin_so("kivi", true, "libkivi_backend_cap.so");

    // ── 2. dlopen → register_backend_caps_v2 → DYN_BACKEND_REGISTRY 등록(3축 dispatcher). ──
    // .so 는 stage/format 0개 + backend-cap 1개 → capability-0 bail 없이 통과.
    register_dynamic_plugins(std::slice::from_ref(&so)).expect("kivi backend-cap .so 등록 실패");

    // ── 3. 등록 가시화. ──
    let names = dynamic_registered_backend_cap_names();
    assert!(
        names.iter().any(|n| n == "kivi_abi"),
        "kivi_abi 미등록: {names:?}"
    );

    // ── 4. category 다리(D7) → DynKiviAttentionBackend 어댑터 생성(resolve, vtable.make 호출). ──
    let make_args = KiviMakeArgs {
        cl_ctx: std::ptr::null_mut(),
        device: std::ptr::null_mut(),
        build_opts: std::ptr::null(),
    };
    let cap = resolve_kivi_capability("kivi_abi", &make_args)
        .expect("resolve_kivi_capability None — category 다리/make 실패");

    // ── 5. 어댑터 메서드 round-trip — bool 쿼리(vtable.has/nosub). ──
    assert!(
        cap.has_kivi_attn_kernel(2),
        "has_kivi_attn_kernel(2) != true"
    );
    assert!(cap.has_kivi_attn_kernel(4));
    assert!(cap.has_kivi_attn_kernel(8));
    assert!(
        !cap.has_kivi_attn_kernel(3),
        "has_kivi_attn_kernel(3) != false"
    );
    assert!(!cap.is_nosub_device(), "is_nosub_device != false");

    // ── 6. attention_gen_kivi dispatch round-trip — KiviAttnArgs 가 ABI 경계를 정확히 넘었는지 ──
    //       sentinel 로 검증. 더미 non-null cl_mem(synthetic 은 null 검사만 — GPU 안 씀).
    let mut dummy = 0u8;
    let dummy_mem = (&mut dummy as *mut u8) as *mut c_void;
    let mut scores = [0.0f32; 1];
    let attn = KiviAttnArgs {
        cl_queue: std::ptr::null_mut(),
        q_mem: dummy_mem,
        qk_mem: dummy_mem,
        qv_mem: dummy_mem,
        res_k_mem: dummy_mem,
        res_v_mem: dummy_mem,
        out_mem: dummy_mem,
        scores_out: scores.as_mut_ptr(),
        scores_len: 1,
        num_heads_q: 32,
        num_heads_kv: 8,
        head_dim: 64,
        q_tokens: 1,
        res_tokens: 16,
        res_cap: 128,
        scale: 0.125,
        bits: 2,
    };
    let rc = cap.attention_gen_kivi(&attn);
    assert_eq!(rc, 0, "attention_gen_kivi rc != 0 (마샬링 실패)");
    // synthetic sentinel = num_heads_q*1000 + head_dim + bits*0.5 + scale (동일 f32 연산 순서).
    let expected = 32.0_f32 * 1000.0 + 64.0 + 2.0 * 0.5 + 0.125;
    assert_eq!(
        scores[0], expected,
        "scores_out sentinel 불일치 — KiviAttnArgs scalar 가 ABI 경계를 잘못 넘음"
    );

    // ── 7. null mem → 어댑터→vtable→plugin 이 -1(마샬링 실패 감지). ──
    let bad = KiviAttnArgs {
        q_mem: std::ptr::null_mut(),
        ..attn
    };
    assert_eq!(cap.attention_gen_kivi(&bad), -1, "null q_mem 인데 rc != -1");

    // ── 8. kivi_gather_update round-trip. ──
    let gather = KiviGatherArgs {
        cl_queue: std::ptr::null_mut(),
        input_mem: dummy_mem,
        residual_mem: dummy_mem,
        kv_heads: 8,
        res_cap: 128,
        head_dim: 64,
        seq_len: 1,
        res_pos: 0,
    };
    assert_eq!(
        cap.kivi_gather_update(&gather),
        0,
        "kivi_gather_update rc != 0"
    );

    // ── 9. 미지 이름 → None (graceful unknown). ──
    assert!(
        resolve_kivi_capability("nonexistent_cap", &make_args).is_none(),
        "미지 이름이 None 아님"
    );
}
