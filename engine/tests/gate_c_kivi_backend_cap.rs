//! GATE-C (kivi) — Backend 축 capability dlopen 재증명 게이트, `kivi` 플러그인 짝
//! (design D2/D7/D8, CB5). `gate_c_backend_cap_dlopen.rs`(synthetic `example-backend-cap`)의
//! 자매 게이트로, 첫 *비-example* backend-cap 플러그인 `kivi`(이름 `kivi_abi`)가 ABI 경계
//! (register_backend_caps_v2 봉투 → category 다리 → `DynQuantAttnBackend` 어댑터 → make/dispatch)를
//! **정확히** 넘는지 증명한다.
//!
//! **host-검증 범위(C12, FORMAT Phase 2 Stage E)**: `kivi` 플러그인은 이제 실제 OpenCL 커널을 `make()`
//! 의 borrowed cl_context 에서 컴파일한다. host 게이트엔 GPU 컨텍스트가 없어 make 가 degraded
//! 핸들(inner=None)을 반환하므로, 본 게이트는 ABI 라운드트립(register → category 다리 →
//! `DynQuantAttnBackend` 어댑터 → make/vtable dispatch)과 **panic=abort 안전성**(degraded 경로가
//! C-ABI 를 넘어 -1 을 반환하고 프로세스를 죽이지 않음)을 증명한다. 실제 커널 정확성은 on-device
//! byte-identical 게이트(Adreno S25)가 검증하고, ABI 마샬링 sentinel 라운드트립은 synthetic
//! `example-backend-cap` 자매 게이트(`gate_c_backend_cap_dlopen.rs`)가 커버한다.
//!
//! **process-global 레지스트리**(DYN_BACKEND_REGISTRY OnceLock): 단일 `#[test]` 에서 순차 수행.

use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;

use argus_engine::capability::dynamic_backend_registry::{
    dynamic_registered_backend_cap_names, resolve_quant_attn_capability,
};
use argus_engine::session::plugin_dispatch::register_dynamic_plugins;
use argus_extension_api::{QuantAttnArgs, QuantAttnGatherArgs, QuantAttnMakeArgs};

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

    // ── 4. category 다리(D7) → DynQuantAttnBackend 어댑터 생성(resolve, vtable.make 호출). ──
    let make_args = QuantAttnMakeArgs {
        cl_ctx: std::ptr::null_mut(),
        device: std::ptr::null_mut(),
        build_opts: std::ptr::null(),
    };
    let cap = resolve_quant_attn_capability("kivi_abi", &make_args)
        .expect("resolve_quant_attn_capability None — category 다리/make 실패");

    // ── 5. Degraded round-trip — host 엔 GPU 컨텍스트가 없어 make 가 커널을 컴파일하지 못해
    //       inner=None(degraded). ABI 경계(어댑터→vtable→plugin)는 정상 동작하되 capability
    //       쿼리는 false 를 보고한다. (실제 true 경로는 on-device 게이트에서만 가능.)
    assert!(
        !cap.has_quant_attn_kernel(2),
        "host(no GPU): has_quant_attn_kernel(2) should be false (degraded)"
    );
    assert!(!cap.has_quant_attn_kernel(4));
    assert!(!cap.has_quant_attn_kernel(8));
    assert!(!cap.has_quant_attn_kernel(3));
    assert!(!cap.is_nosub_device());

    // ── 6. attention_gen_quant dispatch — degraded 경로가 C-ABI 를 넘어 -1 을 반환하고
    //       panic=abort 로 프로세스를 죽이지 않는지(first-of-kind 안전성) 증명한다. ──
    let mut dummy = 0u8;
    let dummy_mem = (&mut dummy as *mut u8) as *mut c_void;
    let mut scores = [0.0f32; 1];
    let attn = QuantAttnArgs {
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
    assert_eq!(
        cap.attention_gen_quant(&attn),
        -1,
        "degraded attention_gen_quant must return -1 across the ABI (no panic/abort)"
    );

    // ── 7. gather_update_quant — degraded 도 -1(panic 없이 C-ABI 를 넘는다). ──
    let gather = QuantAttnGatherArgs {
        cl_queue: std::ptr::null_mut(),
        input_mem: dummy_mem,
        residual_mem: dummy_mem,
        kv_heads: 8,
        res_cap: 128,
        head_dim: 64,
        seq_len: 1,
        res_pos: 0,
    };
    assert_eq!(cap.gather_update_quant(&gather), -1);

    // ── 8. 미지 이름 → None (graceful unknown). ──
    assert!(
        resolve_quant_attn_capability("nonexistent_cap", &make_args).is_none(),
        "미지 이름이 None 아님"
    );
}
