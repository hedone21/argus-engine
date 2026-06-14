/// AUF v0.2 dtype 변환 파이프라인 (ENG-ALG-224, Sprint C-A).
///
/// AUF Writer가 같은 layer/tensor에 대해 여러 dtype variant를 동봉할 때 사용하는
/// 결정적 dequant→requant 헬퍼 모음이다. 결정성 의무(ENG-DAT-096.13, INV-138)는
/// 다음에 의해 보장된다.
///
/// 1. F16 ↔ F32 변환은 IEEE 754 round-half-to-even에 따라 비트 단위 결정적.
/// 2. Q4_0 quantize는 `models::loader::convert::quantize_q4_0`이 제공하는 단순
///    max-abs scaling 루프 — float reduction 순서가 고정 (rows × blocks 외측,
///    blocks 내부 16-pair 순회)되어 결정적.
/// 3. Q4_0 dequantize는 `core::quant::BlockQ4_0::dequantize` (per-block linear
///    decode)로 결정적.
///
/// 따라서 같은 입력 + 같은 dtype 조합 → byte-identical AUF가 보장된다.
///
/// Spec 참조:
/// - `spec/32-engine-algorithms.md` §3.12.18 (ENG-ALG-224)
/// - `spec/33-engine-data.md` §3.22.14~16 (ENG-DAT-097/098/099)
/// - `spec/41-invariants.md` §3.18 (INV-137~139)
use crate::auf::error::{AufError, AufResult};
use crate::auf::tensor_index::TensorDType;
use crate::quant::convert::{f16_to_f32, quantize_q4_0};
use crate::quant::{BlockQ4_0, QK4_0};
use half::f16;

/// `src_dtype`로 인코딩된 `src_bytes`를 `dst_dtype` 바이트로 변환한다.
///
/// 변환 경로 (현재 지원):
/// - F32 ↔ F16 ↔ Q4_0 (양방향)
/// - 동일 dtype: zero-copy clone
///
/// `shape_logical`는 outermost-first 2-D `[rows, cols]`. Q4_0 dequant/quant는
/// `cols % QK4_0 == 0`을 요구한다. 1-D tensor는 `[1, n]`로 취급한다.
///
/// 결정성: 동일 (`src_bytes`, `src_dtype`, `dst_dtype`, `shape`) 입력에 대해
/// byte-identical 출력. 호스트 결정적 (ENG-DAT-096.13).
pub fn convert_tensor_dtype(
    src_bytes: &[u8],
    src_dtype: TensorDType,
    dst_dtype: TensorDType,
    shape_logical: &[u64],
) -> AufResult<Vec<u8>> {
    if src_dtype == dst_dtype {
        return Ok(src_bytes.to_vec());
    }

    let (rows, cols) = shape_to_2d(shape_logical)?;
    let numel = rows.checked_mul(cols).ok_or_else(|| {
        AufError::Other(format!("element count overflow: shape={shape_logical:?}"))
    })?;

    // 1) src → F32 (intermediate).
    let f32_intermediate = match src_dtype {
        TensorDType::F32 => bytes_to_f32(src_bytes, numel)?,
        TensorDType::F16 => f16_bytes_to_f32(src_bytes, numel)?,
        TensorDType::Q4_0 => q4_0_bytes_to_f32(src_bytes, numel)?,
        other => {
            return Err(AufError::Other(format!(
                "convert_tensor_dtype: unsupported src dtype {other:?}"
            )));
        }
    };

    // 2) F32 → dst.
    match dst_dtype {
        TensorDType::F32 => Ok(f32_to_bytes(&f32_intermediate)),
        TensorDType::F16 => Ok(f32_to_f16_bytes(&f32_intermediate)),
        TensorDType::Q4_0 => q4_0_quantize_bytes(&f32_intermediate, rows, cols),
        other => Err(AufError::Other(format!(
            "convert_tensor_dtype: unsupported dst dtype {other:?}"
        ))),
    }
}

/// `(dtype, bytes)` candidate 페어를 multi-dtype 모드로 생성한다.
///
/// AUF v0.2 multi-dtype writer가 각 tensor에 동봉할 candidate dtype 별 bytes를
/// 결정하는 정책 함수. ENG-ALG-224 (`spec/32-engine-algorithms.md` §3.12.18.1)
/// writer 의무 + ISSUE-E-1 회귀 격리 (Sprint F).
///
/// # Behavior
///
/// - `candidate_dtypes = None` (single-dtype, v0.1.x 호환): `src_dtype` 1개만 동봉.
/// - `candidate_dtypes = Some(cands)` + `shape_logical.len() < 2`
///   (1-D / scalar tensor, 예: RMSNorm weight): **`src_dtype` 1개만 동봉**.
///   1-D tensor를 Q4_0 등 다른 dtype으로 quantize하면 의미적 garbage가 되어
///   primary forward의 norm 경로에서 첫 token EOS를 유발한다 (Sprint E ISSUE-E-1).
///   Reader 측 dtype filter도 1-D Q4 entry를 무비판적으로 선택하므로 writer 측에서
///   엔트리 자체를 만들지 않는 것이 안전망.
/// - `candidate_dtypes = Some(cands)` + `shape_logical.len() >= 2`: 각 dtype별로
///   `convert_tensor_dtype`로 변환. 변환 실패 (Q4_0 cols % 32 != 0 등)는 해당 dtype
///   skip + 경고 로그. 결과가 비면 src_dtype 1개라도 fallback 동봉.
///
/// # Determinism
///
/// 동일 (`src_bytes`, `src_dtype`, `shape_logical`, `candidate_dtypes`) 입력에 대해
/// byte-identical 출력을 보장한다 (`convert_tensor_dtype`이 호스트 결정적이고,
/// 본 함수는 입력 순회 외 비결정적 요소가 없다).
///
/// # Errors
///
/// 항상 `Ok(...)`를 반환한다 — 변환 실패는 silent skip + 경고. spec 테스트용으로
/// `quiet=true`이면 경고도 억제.
pub fn build_dtype_candidates(
    name: &str,
    src_bytes: &[u8],
    src_dtype: TensorDType,
    shape_logical: &[u64],
    candidate_dtypes: Option<&[TensorDType]>,
    quiet: bool,
) -> AufResult<Vec<(TensorDType, Vec<u8>)>> {
    let cands = match candidate_dtypes {
        None => return Ok(vec![(src_dtype, src_bytes.to_vec())]),
        Some(c) => c,
    };

    // ISSUE-E-1 fix (Sprint F): 1-D tensor (norm 등)는 src_dtype 1개만 동봉.
    //
    // RMSNorm weight (`*_norm.weight`)는 항상 1-D (shape rank=1)이며, primary forward
    // 경로는 F16/F32만 사용한다. multi-dtype 모드에서 norm을 Q4_0으로 quantize한 entry를
    // 동봉하면, F16 primary + `--secondary-dtype q4_0` swap 시 reader가 dtype filter로
    // norm Q4 entry를 선택하여 1-D Q4_0 1152B (= 2048/32 × 18) bytes를 norm slot에 bind
    // 한다. 이는 정상 F16 4096B 또는 F32 8192B와 dtype/size 불일치이며, primary forward
    // 에서 RMSNorm scale가 garbage가 되어 첫 token이 EOS로 깨진다 (Sprint E ISSUE-E-1
    // 디바이스 회귀).
    //
    // 1-D는 logical shape rank == 1로 식별한다 (이름 기반 휴리스틱보다 안정적).
    // RoPE freqs / token_embd / lm_head은 모두 2-D 이상이므로 영향 없음.
    if shape_logical.len() < 2 {
        return Ok(vec![(src_dtype, src_bytes.to_vec())]);
    }

    let mut out: Vec<(TensorDType, Vec<u8>)> = Vec::with_capacity(cands.len());
    for &dt in cands {
        if dt == src_dtype {
            out.push((dt, src_bytes.to_vec()));
            continue;
        }
        // Q4_0 변환은 cols % 32 == 0이 필요. 만족 못하면 원본만 동봉하고 경고.
        match convert_tensor_dtype(src_bytes, src_dtype, dt, shape_logical) {
            Ok(bytes) => out.push((dt, bytes)),
            Err(e) => {
                if !quiet {
                    eprintln!(
                        "[auf-tool] Warning: tensor '{}' dtype convert {:?}→{:?} failed: {}; \
                         dropping this dtype candidate",
                        name, src_dtype, dt, e
                    );
                }
                // skip — 해당 dtype candidate은 이 tensor에서 누락. INV-137은 동일 (layer,kind)에
                // 대해 모든 후보가 같은 shape이어야 한다는 제약일 뿐, dtype별 entry 수가 동일해야
                // 한다는 의무는 없다 (정합성은 reader 측 lookup으로 보장).
            }
        }
    }

    if out.is_empty() {
        // 모든 candidate이 reject되면 source 1개라도 fallback으로 보장.
        out.push((src_dtype, src_bytes.to_vec()));
    }
    Ok(out)
}

/// outermost-first logical shape에서 (rows, cols)를 추출한다.
///
/// 1-D tensor는 `[1, n]`으로 처리. 0-D / 3-D 이상은 거부.
fn shape_to_2d(shape_logical: &[u64]) -> AufResult<(usize, usize)> {
    match shape_logical.len() {
        1 => Ok((1, shape_logical[0] as usize)),
        2 => Ok((shape_logical[0] as usize, shape_logical[1] as usize)),
        other => Err(AufError::Other(format!(
            "convert_tensor_dtype: shape rank {other} unsupported (expected 1 or 2)"
        ))),
    }
}

fn bytes_to_f32(src: &[u8], numel: usize) -> AufResult<Vec<f32>> {
    let need = numel * 4;
    if src.len() < need {
        return Err(AufError::Other(format!(
            "F32 bytes too small: need {need}, have {}",
            src.len()
        )));
    }
    let mut out = vec![0.0f32; numel];
    // SAFETY: src is little-endian f32 array of length numel (verified above).
    let src_f32 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, numel) };
    out.copy_from_slice(src_f32);
    Ok(out)
}

fn f16_bytes_to_f32(src: &[u8], numel: usize) -> AufResult<Vec<f32>> {
    let need = numel * 2;
    if src.len() < need {
        return Err(AufError::Other(format!(
            "F16 bytes too small: need {need}, have {}",
            src.len()
        )));
    }
    let mut out = vec![0.0f32; numel];
    f16_to_f32(src, &mut out, numel);
    Ok(out)
}

fn q4_0_bytes_to_f32(src: &[u8], numel: usize) -> AufResult<Vec<f32>> {
    if !numel.is_multiple_of(QK4_0) {
        return Err(AufError::Other(format!(
            "Q4_0 numel ({numel}) not multiple of QK4_0 ({QK4_0})"
        )));
    }
    let n_blocks = numel / QK4_0;
    let need = n_blocks * std::mem::size_of::<BlockQ4_0>();
    if src.len() < need {
        return Err(AufError::Other(format!(
            "Q4_0 bytes too small: need {need}, have {}",
            src.len()
        )));
    }
    // SAFETY: BlockQ4_0 is `#[repr(C)]` size 18; src has at least n_blocks*18 bytes.
    let blocks = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const BlockQ4_0, n_blocks) };
    let mut out = vec![0.0f32; numel];
    for (i, blk) in blocks.iter().enumerate() {
        let dst_block: &mut [f32; QK4_0] =
            (&mut out[i * QK4_0..(i + 1) * QK4_0]).try_into().unwrap();
        blk.dequantize(dst_block);
    }
    Ok(out)
}

fn f32_to_bytes(src: &[f32]) -> Vec<u8> {
    let n = src.len();
    let mut out = vec![0u8; n * 4];
    // SAFETY: dst has exactly n*4 bytes.
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, n) };
    dst.copy_from_slice(src);
    out
}

fn f32_to_f16_bytes(src: &[f32]) -> Vec<u8> {
    let n = src.len();
    let mut out = vec![0u8; n * 2];
    // SAFETY: dst has exactly n*2 bytes (n f16s).
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f16, n) };
    for (i, &v) in src.iter().enumerate() {
        dst[i] = f16::from_f32(v);
    }
    out
}

fn q4_0_quantize_bytes(src: &[f32], rows: usize, cols: usize) -> AufResult<Vec<u8>> {
    if !cols.is_multiple_of(QK4_0) {
        return Err(AufError::Other(format!(
            "Q4_0 quantize: cols ({cols}) not multiple of QK4_0 ({QK4_0})"
        )));
    }
    let blocks = quantize_q4_0(src, rows, cols);
    let block_size = std::mem::size_of::<BlockQ4_0>(); // 18
    let total = blocks.len() * block_size;
    // SAFETY: BlockQ4_0 is `#[repr(C)]` size 18 with no padding; bytes are well-defined.
    let view = unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, total) };
    Ok(view.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 동일 dtype은 zero-conversion (clone)이어야 한다.
    #[test]
    fn same_dtype_passthrough() {
        let bytes = vec![1u8, 2, 3, 4];
        let out = convert_tensor_dtype(&bytes, TensorDType::F32, TensorDType::F32, &[1]).unwrap();
        assert_eq!(out, bytes);
    }

    /// F32 → F16 → F32 round-trip은 representable f16 값에 대해 bit-exact (within f16 precision).
    #[test]
    fn f32_to_f16_round_trip() {
        let src: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 2.0, 100.0];
        let f32_bytes = f32_to_bytes(&src);
        let f16_bytes =
            convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::F16, &[6]).unwrap();
        assert_eq!(f16_bytes.len(), 12); // 6 × 2

        let back =
            convert_tensor_dtype(&f16_bytes, TensorDType::F16, TensorDType::F32, &[6]).unwrap();
        assert_eq!(back.len(), 24); // 6 × 4

        // SAFETY: back is f32 little-endian.
        let back_f32 =
            unsafe { std::slice::from_raw_parts(back.as_ptr() as *const f32, src.len()) };
        for (i, (&a, &b)) in src.iter().zip(back_f32.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "f16 round-trip mismatch at {i}: {a} vs {b}"
            );
        }
    }

    /// Q4_0 → F32 → Q4_0 round-trip은 정확히 같은 bytes (Q4_0 quantize는 max-abs 결정적).
    #[test]
    fn q4_0_round_trip_deterministic() {
        // 32 floats one block. Symmetric range [-7d, 7d].
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let blocks = quantize_q4_0(&src, 1, 32);
        let block_size = std::mem::size_of::<BlockQ4_0>();
        let q4_bytes_in: Vec<u8> = unsafe {
            std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * block_size)
        }
        .to_vec();

        let f32_back =
            convert_tensor_dtype(&q4_bytes_in, TensorDType::Q4_0, TensorDType::F32, &[1, 32])
                .unwrap();

        let q4_bytes_back =
            convert_tensor_dtype(&f32_back, TensorDType::F32, TensorDType::Q4_0, &[1, 32]).unwrap();

        assert_eq!(
            q4_bytes_in, q4_bytes_back,
            "Q4_0→F32→Q4_0 must be byte-identical (determinism)"
        );
    }

    /// F16 → Q4_0 변환이 결정적임을 검증 (두 번 호출 → 동일 bytes).
    #[test]
    fn f16_to_q4_0_deterministic() {
        let src_f32: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let f16_bytes = f32_to_f16_bytes(&src_f32);

        let q4_a = convert_tensor_dtype(&f16_bytes, TensorDType::F16, TensorDType::Q4_0, &[2, 32])
            .unwrap();
        let q4_b = convert_tensor_dtype(&f16_bytes, TensorDType::F16, TensorDType::Q4_0, &[2, 32])
            .unwrap();

        assert_eq!(q4_a, q4_b, "same input must produce byte-identical output");
        assert_eq!(q4_a.len(), 2 * 18); // 2 blocks × 18B
    }

    /// Q4_0 → F16 변환은 (1) Q4_0 dequant 결정적 + (2) F16 cast 결정적이므로 결정적.
    #[test]
    fn q4_0_to_f16_deterministic() {
        let src_f32: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.2).collect();
        let blocks = quantize_q4_0(&src_f32, 1, 32);
        let block_size = std::mem::size_of::<BlockQ4_0>();
        let q4_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * block_size)
        }
        .to_vec();

        let f16_a =
            convert_tensor_dtype(&q4_bytes, TensorDType::Q4_0, TensorDType::F16, &[1, 32]).unwrap();
        let f16_b =
            convert_tensor_dtype(&q4_bytes, TensorDType::Q4_0, TensorDType::F16, &[1, 32]).unwrap();

        assert_eq!(f16_a, f16_b);
        assert_eq!(f16_a.len(), 32 * 2);
    }

    /// shape rank > 2 is rejected.
    #[test]
    fn rank_3_rejected() {
        let bytes = vec![0u8; 64];
        let err = convert_tensor_dtype(&bytes, TensorDType::F32, TensorDType::F16, &[2, 4, 8]);
        assert!(err.is_err());
    }

    /// Q4_0 변환 시 cols가 32 배수가 아니면 거부.
    #[test]
    fn q4_0_cols_must_be_multiple_of_32() {
        let src_f32 = vec![0.0f32; 30];
        let f32_bytes = f32_to_bytes(&src_f32);
        let err = convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::Q4_0, &[1, 30]);
        assert!(err.is_err());
    }

    /// 1-D shape는 `[1, n]`로 처리되어 동작한다.
    #[test]
    fn one_d_shape_supported() {
        let src_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let f32_bytes = f32_to_bytes(&src_f32);
        let f16_bytes =
            convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::F16, &[4]).unwrap();
        assert_eq!(f16_bytes.len(), 8);
    }
}
