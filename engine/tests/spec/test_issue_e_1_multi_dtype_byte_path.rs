//! Spec 테스트: ISSUE-E-1 회귀 격리 — multi-dtype 모드에서 1-D tensor의 dtype 후보 처리.
//!
//! Sprint E (2026-04-27) 디바이스 측정에서 발견된 정확성 회귀:
//!   F16 primary GGUF + mixed.auf (v0.2 multi-dtype Q4_0+F16) + `--secondary-dtype q4_0`
//!   조합에서 첫 token이 EOS(128001)로 깨졌다. v0.1.1 single-dtype AUF에서는 동일
//!   조건이 정상 PASS이므로 multi-dtype writer 경로 한정 회귀이다.
//!
//! Root cause:
//!   `build_dtype_candidates`가 1-D tensor (RMSNorm weight, shape=[2048])에도 Q4_0
//!   변환을 적용하여 norm 별 Q4 entry를 추가했다. F16 primary가 secondary swap을
//!   요청하면 reader의 dtype filter가 1-D Q4_0 1152B (2048/32 × 18) bytes를 norm slot
//!   에 bind하여 RMSNorm scale가 garbage가 된다 (정상은 F16 4096B 또는 F32 8192B).
//!
//! Fix:
//!   1-D tensor (`shape_logical.len() < 2`)는 multi-dtype 모드에서도 src_dtype 1개만
//!   동봉. `engine/src/auf/dtype_convert.rs::build_dtype_candidates` 가드.
//!
//! 본 테스트는 fix를 회귀 격리한다:
//!   1. 1-D F16 tensor + multi-dtype 모드 → src_dtype 1개만 (Q4 entry 생성 안 됨)
//!   2. 1-D F32 tensor + multi-dtype 모드 → src_dtype 1개만
//!   3. 2-D Q4_0 tensor + multi-dtype 모드 → 두 dtype 모두 entry 생성 (정상)
//!   4. 0-D scalar (rank=0)도 src_dtype 1개로 처리
//!
//! spec/Sprint 참조:
//! - Sprint F (2026-04-27) ISSUE-E-1 hotfix
//! - `results/data/weight_swap/v0_2_multi_quant_validation.md` §5

use argus_engine::auf::TensorDType;
use argus_engine::auf::dtype_convert::build_dtype_candidates;

/// (R-1) RMSNorm-style 1-D F16 tensor: multi-dtype 모드 [Q4_0, F16] 명시 시에도
/// **Q4_0 entry는 생성되지 않아야 한다** (src_dtype F16 1개만 반환).
#[test]
fn rms_norm_1d_f16_keeps_only_src_dtype() {
    // 2048-element F16 vector (typical RMSNorm scale, hidden_dim=2048).
    let bytes = vec![0u8; 2048 * 2]; // 4096 bytes
    let shape: Vec<u64> = vec![2048]; // 1-D
    let cands = [TensorDType::Q4_0, TensorDType::F16];

    let out = build_dtype_candidates(
        "blk.0.attn_norm.weight",
        &bytes,
        TensorDType::F16,
        &shape,
        Some(&cands),
        true, // quiet
    )
    .expect("build_dtype_candidates must succeed for 1-D");

    assert_eq!(
        out.len(),
        1,
        "1-D tensor must produce only src_dtype entry (got {} entries)",
        out.len()
    );
    assert_eq!(out[0].0, TensorDType::F16, "must keep src_dtype = F16");
    assert_eq!(
        out[0].1.len(),
        4096,
        "F16 bytes preserved (2048 * 2 = 4096)"
    );
}

/// (R-2) RMSNorm-style 1-D F32 tensor: multi-dtype 모드에서도 src_dtype F32 1개만.
#[test]
fn rms_norm_1d_f32_keeps_only_src_dtype() {
    let bytes = vec![0u8; 2048 * 4]; // 8192 bytes
    let shape: Vec<u64> = vec![2048];
    let cands = [TensorDType::Q4_0, TensorDType::F32];

    let out = build_dtype_candidates(
        "output_norm.weight",
        &bytes,
        TensorDType::F32,
        &shape,
        Some(&cands),
        true,
    )
    .expect("build_dtype_candidates must succeed for 1-D F32");

    assert_eq!(out.len(), 1, "1-D F32 must keep src_dtype only");
    assert_eq!(out[0].0, TensorDType::F32);
    assert_eq!(out[0].1.len(), 8192);
}

/// (R-3) 2-D layer weight Q4_0 source: multi-dtype 모드 [Q4_0, F16] → 두 entry 생성.
///
/// 이 케이스는 fix와 무관하게 정상 동작이어야 한다 (회귀 방지).
#[test]
fn layer_weight_2d_q4_0_yields_both_dtypes() {
    // 32×32 Q4_0 = 32 blocks × 18 bytes = 576 bytes (1 block per row).
    let n_blocks = 32 * 32 / 32;
    let bytes = vec![0u8; n_blocks * 18];
    let shape: Vec<u64> = vec![32, 32]; // 2-D
    let cands = [TensorDType::Q4_0, TensorDType::F16];

    let out = build_dtype_candidates(
        "blk.0.attn_q.weight",
        &bytes,
        TensorDType::Q4_0,
        &shape,
        Some(&cands),
        true,
    )
    .expect("2-D Q4_0 multi-dtype must succeed");

    assert_eq!(out.len(), 2, "2-D tensor must produce both dtype entries");
    assert_eq!(out[0].0, TensorDType::Q4_0, "first entry = src_dtype");
    assert_eq!(out[1].0, TensorDType::F16, "second entry = converted F16");

    // F16 size: 32 * 32 * 2 = 2048 bytes
    assert_eq!(out[1].1.len(), 32 * 32 * 2);
}

/// (R-4) candidate_dtypes = None (single-dtype, v0.1.x 호환): src_dtype 1개.
#[test]
fn single_dtype_mode_unchanged() {
    let bytes = vec![0u8; 100];
    let shape: Vec<u64> = vec![10, 10];

    let out = build_dtype_candidates("test", &bytes, TensorDType::F32, &shape, None, true).unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, TensorDType::F32);
    assert_eq!(out[0].1, bytes);
}

/// (R-5) 0-D scalar (shape rank=0): src_dtype 1개만 (1-D 가드와 동일 분기).
#[test]
fn zero_d_scalar_keeps_only_src_dtype() {
    let bytes = vec![0u8; 4]; // single F32 scalar
    let shape: Vec<u64> = vec![]; // 0-D
    let cands = [TensorDType::F32, TensorDType::F16];

    let out = build_dtype_candidates(
        "scalar",
        &bytes,
        TensorDType::F32,
        &shape,
        Some(&cands),
        true,
    )
    .expect("0-D must succeed via 1-D guard");

    assert_eq!(out.len(), 1, "0-D must keep src_dtype only");
    assert_eq!(out[0].0, TensorDType::F32);
}

/// (R-6) 1-D tensor + multi-dtype 모드에서 src_dtype이 후보 목록에 **없어도**
/// 결과는 src_dtype 1개. 후보 목록 무관하게 1-D는 source 보존이 spec.
#[test]
fn one_d_ignores_candidate_list_entirely() {
    let bytes = vec![0u8; 2048 * 4];
    let shape: Vec<u64> = vec![2048];
    // 후보 목록에 F32(src)는 없고 Q4_0과 F16만.
    let cands = [TensorDType::Q4_0, TensorDType::F16];

    let out = build_dtype_candidates(
        "blk.0.attn_norm.weight",
        &bytes,
        TensorDType::F32,
        &shape,
        Some(&cands),
        true,
    )
    .expect("1-D must succeed even when src_dtype not in cands");

    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].0,
        TensorDType::F32,
        "1-D guard must preserve src_dtype regardless of cands"
    );
    assert_eq!(out[0].1.len(), 8192);
}

/// (R-7) 2-D tensor + multi-dtype 모드 + cols % 32 != 0 (Q4_0 변환 불가):
/// Q4_0 후보는 silent skip되고 변환 가능한 dtype만 entry 생성. Fix는 1-D만 영향
/// 미치고 2-D 변환 실패 fallback은 기존 동작 보존.
#[test]
fn two_d_q4_failure_silent_skip_unchanged() {
    // 4×30 F32: cols=30이 32 배수 아님 → Q4_0 변환 reject.
    let bytes = vec![0u8; 4 * 30 * 4];
    let shape: Vec<u64> = vec![4, 30];
    let cands = [TensorDType::Q4_0, TensorDType::F16];

    let out = build_dtype_candidates(
        "test_2d",
        &bytes,
        TensorDType::F32,
        &shape,
        Some(&cands),
        true,
    )
    .expect("2-D with Q4 reject must still succeed (only F16 entry)");

    // Q4_0 변환 실패 → skip. F16만 남음. (src=F32이지만 cands에 F32 없으므로 F16만.)
    assert_eq!(out.len(), 1, "Q4 reject + F16 success → 1 entry (F16 only)");
    assert_eq!(out[0].0, TensorDType::F16);
}
