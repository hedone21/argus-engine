//! INV-RPCMEM-006 — `--opencl-rpcmem` + `--backend qnn_oppkg | qnngpu` 동시 지정 시
//! 후자가 우선되며 전자는 무시 (stderr 경고 1회). Sprint 2a 호환 mutex.
//! Sprint 2b 에서 qnn_oppkg backend 삭제 시 본 mutex 도 자연 만료.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.5 (ENG-RPCMEM-041), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/rpcmem_allocator.md` §3.1, `arch/opencl_backend.md` §5.

#![allow(dead_code, unused_imports)]

use argus_engine::session::cli::Args;
use clap::Parser;

#[test]
fn cli_opencl_rpcmem_demoted_when_qnn_oppkg_backend_used() {
    // INV-RPCMEM-006: --backend qnn_oppkg + --opencl-rpcmem 동시 지정.
    // Args::parse_from 은 OK 반환하고 (parse 실패 아님),
    // effective_opencl_rpcmem() 는 false 를 반환해야 한다.
    let args = Args::try_parse_from([
        "argus_engine",
        "--backend",
        "qnn_oppkg",
        "--opencl-rpcmem",
        "--model-path",
        "/tmp/dummy.gguf",
    ])
    .expect("Args parse 는 성공해야 함 — opencl-rpcmem + qnn_oppkg 는 parse error 아님");

    // 파싱 자체는 true 로 저장되지만 effective 는 false 여야 한다.
    assert_eq!(
        args.effective_opencl_rpcmem(),
        false,
        "INV-RPCMEM-006: --backend qnn_oppkg 사용 시 effective_opencl_rpcmem() 는 false 반환해야 함."
    );
}

#[test]
fn cli_opencl_rpcmem_demoted_when_qnngpu_backend_used() {
    let args = Args::try_parse_from([
        "argus_engine",
        "--backend",
        "qnngpu",
        "--opencl-rpcmem",
        "--model-path",
        "/tmp/dummy.gguf",
    ])
    .expect("Args parse 는 성공해야 함");

    assert_eq!(
        args.effective_opencl_rpcmem(),
        false,
        "INV-RPCMEM-006: --backend qnngpu 사용 시 effective_opencl_rpcmem() 는 false 반환해야 함."
    );
}

#[test]
fn cli_opencl_rpcmem_active_when_backend_opencl() {
    // INV-RPCMEM-006: --backend opencl + --opencl-rpcmem 조합 → effective true.
    let args = Args::try_parse_from([
        "argus_engine",
        "--backend",
        "opencl",
        "--opencl-rpcmem",
        "--model-path",
        "/tmp/dummy.gguf",
    ])
    .expect("Args parse 성공");

    assert_eq!(
        args.effective_opencl_rpcmem(),
        true,
        "INV-RPCMEM-006: --backend opencl + --opencl-rpcmem 조합 시 effective_opencl_rpcmem() 는 true 반환해야 함."
    );
}

#[test]
fn cli_opencl_rpcmem_false_by_default() {
    // 기본값 확인 — --opencl-rpcmem 없으면 false.
    let args = Args::try_parse_from(["argus_engine", "--model-path", "/tmp/dummy.gguf"])
        .expect("Args parse 성공");

    assert_eq!(
        args.effective_opencl_rpcmem(),
        false,
        "INV-RPCMEM-006: --opencl-rpcmem 기본값은 false 여야 함."
    );
}
