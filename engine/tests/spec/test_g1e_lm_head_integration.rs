//! Sprint G-1-E — AUF lm_head Q4_0 통합 정확성 검증.
//!
//! **G-1-F update (INV-135 v2)**: lm_head Q4_0 entry는 모든 backend variant에서
//! AOS 18B/block layout으로 동봉된다. ADRENO_SOA section 안에서도 lm_head만 AOS,
//! 나머지 layer weight는 SOA. 이유: lm_head q_buf size가
//! `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` 한계 초과 → image1d_buffer_t 생성 실패 →
//! 빠른 SOA path 발동 불가 → AOS layout으로 통일.
//!
//! Covers (v2):
//! 1. ADRENO_SOA + lm_head: payload.bytes = raw AOS Q4_0 (18B/block).
//! 2. payload.bytes.len() == NUM_BLOCKS * 18 (AOS 그대로).
//! 3. Roundtrip (ADRENO_SOA): writer AOS → reader AOS bytes 일치.
//! 4. 후방 호환: bit2=0 AUF → lm_head_q4_0_payload() Ok(None) (INV-136).
//! 5. 숫자 정확성: quantize_q4_0 결정성 (두 번 호출 → byte-level 동일).
//! 6. ADRENO_SOA + CPU_AOS 양쪽: lm_head payload는 동일 AOS bytes.
//! 7. CPU_AOS variant roundtrip: payload bytes == AOS Q4_0.
//! 8. shape/dtype/alignment accessor 정확성.
//! 9. ADRENO_SOA + CPU_AOS 동시 variant — 각 accessor가 올바른 bytes 반환.
//!
//! Spec: INV-135 v2, INV-136, G-1-F fix.

use argus_engine::auf::reader::{BackendTag, open_from_bytes};
use argus_engine::auf::section::{TAG_WEIGHTS_ADRENO_SOA, TAG_WEIGHTS_CPU_AOS};
use argus_engine::auf::tensor_index::{
    LAYER_IDX_CROSS, TensorDType, TensorEntry, TensorIndex, TensorKind,
};
use argus_engine::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
use argus_engine::auf::writer::AufWriter;
use argus_engine::auf::{AufMeta, q4_0_aos_to_adreno_soa};

// ── Fixture constants ─────────────────────────────────────────────────────────

/// vocab_size (multiple of 32, small for fast tests).
const VOCAB: usize = 64;
/// hidden_dim (multiple of 32).
const HIDDEN: usize = 128;
/// Total Q4_0 blocks for (VOCAB × HIDDEN).
const NUM_BLOCKS: usize = VOCAB * HIDDEN / 32; // = 256

// ── Fixture helpers ───────────────────────────────────────────────────────────

fn make_meta() -> AufMeta {
    AufMeta {
        architecture: "llama".to_owned(),
        n_layers: 2,
        n_heads_q: 4,
        n_kv_heads: 2,
        head_dim: 8,
        hidden_dim: HIDDEN as u32,
        ffn_dim: 64,
        vocab_size: VOCAB as u32,
        max_seq_len: 128,
        rope_theta: 10000.0,
        rotary_dim: 8,
        rope_scaling: 1.0,
        rms_norm_epsilon: 1e-5,
        default_dtype: None,
    }
}

fn make_tokenizer() -> AufTokenizer {
    AufTokenizer {
        kind: TOKENIZER_KIND_BPE,
        tokens: vec![b"a".to_vec(), b"b".to_vec()],
        merges: vec![],
        bos_id: 1,
        eos_id: 2,
        pad_id: -1,
        unk_id: 0,
        chat_template: None,
    }
}

/// Deterministic AOS Q4_0 bytes for (VOCAB × HIDDEN).
fn make_aos_q4_0_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(NUM_BLOCKS * 18);
    for b in 0..NUM_BLOCKS {
        // scale: little-endian f16 = b+1
        let scale = ((b as u16) + 1).to_le_bytes();
        out.extend_from_slice(&scale);
        // 16 nibble bytes: pseudo-random per block
        for i in 0..16u8 {
            out.push((b as u8).wrapping_add(i.wrapping_mul(7)));
        }
    }
    assert_eq!(out.len(), NUM_BLOCKS * 18);
    out
}

/// Build an AUF with TAG_WEIGHTS_ADRENO_SOA section containing an lm_head AOS entry.
///
/// **G-1-F update (INV-135 v2)**: lm_head는 image1d_buffer_t 한계 때문에 SOA 변환
/// 불가. ADRENO_SOA section 내부에서도 AOS bytes 그대로 동봉.
fn build_auf_adreno_soa_with_lm_head(aos_bytes: &[u8]) -> Vec<u8> {
    let lm_head_size = aos_bytes.len() as u64;
    let mut tag_arr = [0u8; 24];
    let tb = TAG_WEIGHTS_ADRENO_SOA.as_bytes();
    tag_arr[..tb.len().min(24)].copy_from_slice(&tb[..tb.len().min(24)]);

    let tidx = TensorIndex {
        variant_tags: vec![tag_arr],
        entries: vec![TensorEntry {
            layer_idx: LAYER_IDX_CROSS,
            kind: TensorKind::LmHead.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![VOCAB as u64, HIDDEN as u64],
            alignment: 65536,
            variant_offsets: vec![0],
            variant_sizes: vec![lm_head_size],
        }],
    };

    AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
        .with_lm_head_q4_0(true)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, aos_bytes.to_vec())
        .build()
        .unwrap()
}

/// Build an AUF with TAG_WEIGHTS_CPU_AOS section containing an lm_head AOS entry.
fn build_auf_cpu_aos_with_lm_head(aos_bytes: &[u8]) -> Vec<u8> {
    let lm_head_size = aos_bytes.len() as u64;
    let mut tag_arr = [0u8; 24];
    let tb = TAG_WEIGHTS_CPU_AOS.as_bytes();
    tag_arr[..tb.len().min(24)].copy_from_slice(&tb[..tb.len().min(24)]);

    let tidx = TensorIndex {
        variant_tags: vec![tag_arr],
        entries: vec![TensorEntry {
            layer_idx: LAYER_IDX_CROSS,
            kind: TensorKind::LmHead.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![VOCAB as u64, HIDDEN as u64],
            alignment: 65536,
            variant_offsets: vec![0],
            variant_sizes: vec![lm_head_size],
        }],
    };

    AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
        .with_lm_head_q4_0(true)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, aos_bytes.to_vec())
        .build()
        .unwrap()
}

// ── 1 + 2: ADRENO_SOA + lm_head — AOS layout & bytes 크기 (INV-135 v2) ─────────

/// **G-1-F (INV-135 v2)**: ADRENO_SOA section 내부에서도 lm_head는 AOS layout으로
/// 동봉된다 (image 한계 때문에 SOA 변환 적용 불가). reader는 AOS bytes를 그대로
/// 노출한다.
#[test]
fn adreno_soa_payload_is_aos_for_lm_head() {
    let aos = make_aos_q4_0_bytes();
    let auf_bytes = build_auf_adreno_soa_with_lm_head(&aos);
    let view = open_from_bytes(auf_bytes, BackendTag::AdrenoSoa).unwrap();

    assert!(
        view.header.has_lm_head_q4_0(),
        "capability bit 2 must be set"
    );
    let payload = view
        .lm_head_q4_0_payload(VOCAB, HIDDEN)
        .unwrap()
        .expect("should return Some(LmHeadPayload)");

    // 크기 == N * 18 (AOS).
    let expected_size = NUM_BLOCKS * 18;
    assert_eq!(
        payload.bytes.len(),
        expected_size,
        "ADRENO_SOA + lm_head payload bytes must equal NUM_BLOCKS*18={} (got {})",
        expected_size,
        payload.bytes.len(),
    );

    // INV-135 v2: AOS layout (raw GGUF bytes 그대로). SOA 변환 미적용.
    assert_eq!(
        payload.bytes,
        &aos[..],
        "ADRENO_SOA + lm_head: payload bytes must equal raw AOS Q4_0 (no SOA transform)"
    );
    assert_eq!(payload.variant_tag, TAG_WEIGHTS_ADRENO_SOA);
}

// ── 3. SOA 크기 불변 (`soa_conversion_byte_invariant_for_layer_weights` 참조) ─

// ── 4. Roundtrip ─────────────────────────────────────────────────────────────

/// CPU_AOS roundtrip: writer stores AOS → reader returns same AOS bytes.
#[test]
fn cpu_aos_roundtrip_bytes_match() {
    let aos = make_aos_q4_0_bytes();
    let auf_bytes = build_auf_cpu_aos_with_lm_head(&aos);
    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();

    let payload = view
        .lm_head_q4_0_payload(VOCAB, HIDDEN)
        .unwrap()
        .expect("should have lm_head entry");

    assert_eq!(
        payload.bytes,
        &aos[..],
        "CPU_AOS roundtrip: payload bytes must equal original AOS bytes"
    );
    assert_eq!(payload.variant_tag, TAG_WEIGHTS_CPU_AOS);
    assert_eq!(payload.bytes.len(), NUM_BLOCKS * 18);
}

/// **G-1-F (INV-135 v2)**: ADRENO_SOA + lm_head AOS roundtrip.
///
/// writer stores AOS → reader returns AOS bytes == original AOS bytes.
/// (구 spec에서는 SOA bytes 검증, INV-135 v2부터 AOS 검증.)
#[test]
fn adreno_soa_roundtrip_bytes_match_aos() {
    let aos = make_aos_q4_0_bytes();
    let auf_bytes = build_auf_adreno_soa_with_lm_head(&aos);
    let view = open_from_bytes(auf_bytes, BackendTag::AdrenoSoa).unwrap();
    let payload = view
        .lm_head_q4_0_payload(VOCAB, HIDDEN)
        .unwrap()
        .expect("should have lm_head entry");

    assert_eq!(
        payload.bytes,
        &aos[..],
        "ADRENO_SOA + lm_head roundtrip: payload bytes must equal original AOS Q4_0 \
         (no SOA transform applied to lm_head)"
    );
    assert_eq!(payload.bytes.len(), NUM_BLOCKS * 18);
}

/// **Sanity**: SOA 변환 자체는 layer weight에 대해 여전히 byte-equivalent.
/// q4_0_aos_to_adreno_soa(aos) total bytes == aos bytes (불변).
/// lm_head는 적용 대상 아님이지만 layer weight (kind != LmHead) 검증을 위해 보존.
#[test]
fn soa_conversion_byte_invariant_for_layer_weights() {
    let aos = make_aos_q4_0_bytes();
    let (q_buf, d_buf) = q4_0_aos_to_adreno_soa(&aos, HIDDEN, VOCAB);
    assert_eq!(
        q_buf.len() + d_buf.len(),
        aos.len(),
        "SOA q+d total must equal AOS total bytes"
    );
    assert_eq!(q_buf.len(), NUM_BLOCKS * 16, "q_buf must be N*16");
    assert_eq!(d_buf.len(), NUM_BLOCKS * 2, "d_buf must be N*2");
}

// ── 5. 후방 호환 ──────────────────────────────────────────────────────────────

/// INV-136: bit2=0 (v0.1.0 AUF) → lm_head_q4_0_payload() = Ok(None).
#[test]
fn backward_compat_bit2_zero_cpu_aos_returns_none() {
    let auf_bytes = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 64])
        .build()
        .unwrap();
    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
    assert!(!view.header.has_lm_head_q4_0());
    assert!(view.lm_head_q4_0_payload(VOCAB, HIDDEN).unwrap().is_none());
}

/// INV-136: bit2=0 + ADRENO_SOA backend → Ok(None).
#[test]
fn backward_compat_bit2_zero_adreno_soa_returns_none() {
    let auf_bytes = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
        .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![0u8; 128])
        .build()
        .unwrap();
    let view = open_from_bytes(auf_bytes, BackendTag::AdrenoSoa).unwrap();
    assert!(!view.header.has_lm_head_q4_0());
    assert!(view.lm_head_q4_0_payload(VOCAB, HIDDEN).unwrap().is_none());
}

// ── 6. 숫자 정확성: quantize_q4_0 결정성 ─────────────────────────────────────

/// quantize_q4_0 결정성: 동일 F32 입력 → 두 번 호출 → byte-level identical.
/// G-1-B writer가 동일 GGUF에서 두 번 빌드 시 byte-identical AUF를 만드는 전제 검증.
#[test]
fn quantize_q4_0_is_deterministic() {
    use argus_engine::models::loader::convert::quantize_q4_0;
    use argus_engine::quant::BlockQ4_0;

    let f32_data: Vec<f32> = (0..VOCAB * HIDDEN)
        .map(|i| (i as f32) * 0.001 - 4.0)
        .collect();

    let blocks_a = quantize_q4_0(&f32_data, VOCAB, HIDDEN);
    let blocks_b = quantize_q4_0(&f32_data, VOCAB, HIDDEN);

    let n = blocks_a.len() * std::mem::size_of::<BlockQ4_0>();
    let bytes_a = unsafe { std::slice::from_raw_parts(blocks_a.as_ptr() as *const u8, n) };
    let bytes_b = unsafe { std::slice::from_raw_parts(blocks_b.as_ptr() as *const u8, n) };

    assert_eq!(bytes_a, bytes_b, "quantize_q4_0 must be deterministic");
    assert_eq!(n, NUM_BLOCKS * 18, "output size must be NUM_BLOCKS * 18");
}

/// AUF CPU_AOS payload bytes == quantize_q4_0 direct output.
/// G-1-B writer path와 직접 quantize 경로의 결과 동일성 검증.
#[test]
fn auf_cpu_aos_payload_matches_direct_quantize() {
    use argus_engine::models::loader::convert::quantize_q4_0;
    use argus_engine::quant::BlockQ4_0;

    // Step A: direct quantize → bytes.
    let f32_data: Vec<f32> = (0..VOCAB * HIDDEN)
        .map(|i| (i as f32) * 0.001 - 4.0)
        .collect();
    let blocks = quantize_q4_0(&f32_data, VOCAB, HIDDEN);
    let n = blocks.len() * std::mem::size_of::<BlockQ4_0>();
    let direct_bytes =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, n) }.to_vec();

    // Step B: write direct_bytes to AUF CPU_AOS → read back via reader.
    let auf_bytes = build_auf_cpu_aos_with_lm_head(&direct_bytes);
    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
    let payload = view.lm_head_q4_0_payload(VOCAB, HIDDEN).unwrap().unwrap();

    assert_eq!(
        payload.bytes,
        &direct_bytes[..],
        "AUF CPU_AOS payload must match direct quantize_q4_0 output"
    );
}

/// **G-1-F (INV-135 v2)**: ADRENO_SOA + lm_head 부수 검증 — q/d split이 의미가 없음.
///
/// 구 spec에서는 `payload.bytes = q_buf||d_buf`로 split이 가능했지만, INV-135 v2에서는
/// AOS bytes 그대로 동봉되므로 split 후의 byte 패턴은 SOA 변환 결과와 무관. 본 테스트는
/// payload size invariant (N*18)만 보존됨을 확인한다.
#[test]
fn adreno_soa_payload_size_invariant_aos() {
    let aos = make_aos_q4_0_bytes();
    let auf_bytes = build_auf_adreno_soa_with_lm_head(&aos);
    let view = open_from_bytes(auf_bytes, BackendTag::AdrenoSoa).unwrap();
    let payload = view.lm_head_q4_0_payload(VOCAB, HIDDEN).unwrap().unwrap();

    // INV-135 v2: ADRENO_SOA section 내부의 lm_head는 AOS, size N*18 불변.
    assert_eq!(payload.bytes.len(), NUM_BLOCKS * 18);
    // payload bytes가 AOS bytes 자체임을 다시 확인 (SOA 변환 결과와 다름).
    assert_eq!(
        payload.bytes,
        &aos[..],
        "ADRENO_SOA + lm_head: payload bytes must remain AOS (no SOA q/d split)"
    );
    // SOA 변환 결과는 lm_head AOS payload와 다름을 sanity 확인.
    let (q_soa, _) = q4_0_aos_to_adreno_soa(&aos, HIDDEN, VOCAB);
    let q_len = NUM_BLOCKS * 16;
    assert_ne!(
        &payload.bytes[..q_len],
        &q_soa[..],
        "AOS와 SOA-converted q_buf는 byte pattern이 다른 게 정상 (sanity check)"
    );
}

// ── 7. shape/dtype/alignment accessor ────────────────────────────────────────

/// LmHeadPayload: shape, dtype, alignment, variant_tag 정확성 (CPU_AOS).
#[test]
fn lm_head_payload_accessors_cpu_aos() {
    let aos = make_aos_q4_0_bytes();
    let auf_bytes = build_auf_cpu_aos_with_lm_head(&aos);
    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
    let payload = view.lm_head_q4_0_payload(VOCAB, HIDDEN).unwrap().unwrap();

    assert_eq!(payload.shape, [VOCAB, HIDDEN]);
    assert_eq!(payload.dtype, TensorDType::Q4_0);
    assert_eq!(payload.alignment, 65536);
    assert_eq!(payload.variant_tag, TAG_WEIGHTS_CPU_AOS);
}

/// LmHeadPayload: shape, dtype, alignment, variant_tag 정확성 (ADRENO_SOA).
#[test]
fn lm_head_payload_accessors_adreno_soa() {
    let aos = make_aos_q4_0_bytes();
    let auf_bytes = build_auf_adreno_soa_with_lm_head(&aos);
    let view = open_from_bytes(auf_bytes, BackendTag::AdrenoSoa).unwrap();
    let payload = view.lm_head_q4_0_payload(VOCAB, HIDDEN).unwrap().unwrap();

    assert_eq!(payload.shape, [VOCAB, HIDDEN]);
    assert_eq!(payload.dtype, TensorDType::Q4_0);
    assert_eq!(payload.alignment, 65536);
    assert_eq!(payload.variant_tag, TAG_WEIGHTS_ADRENO_SOA);
}

// ── 8. vocab/hidden shape mismatch 거부 ──────────────────────────────────────

/// lm_head_q4_0_payload: vocab_size mismatch → Err(LmHeadShapeMismatch).
#[test]
fn shape_mismatch_vocab_returns_err() {
    use argus_engine::auf::error::AufError;
    let aos = make_aos_q4_0_bytes();
    let auf_bytes = build_auf_cpu_aos_with_lm_head(&aos);
    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();

    // Call with wrong vocab_size.
    let err = view.lm_head_q4_0_payload(VOCAB + 1, HIDDEN).unwrap_err();
    assert!(
        matches!(err, AufError::LmHeadShapeMismatch { .. }),
        "expected LmHeadShapeMismatch for vocab mismatch, got: {err}"
    );
}

/// lm_head_q4_0_payload: hidden_dim mismatch → Err(LmHeadShapeMismatch).
#[test]
fn shape_mismatch_hidden_returns_err() {
    use argus_engine::auf::error::AufError;
    let aos = make_aos_q4_0_bytes();
    let auf_bytes = build_auf_cpu_aos_with_lm_head(&aos);
    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();

    let err = view.lm_head_q4_0_payload(VOCAB, HIDDEN + 32).unwrap_err();
    assert!(
        matches!(err, AufError::LmHeadShapeMismatch { .. }),
        "expected LmHeadShapeMismatch for hidden mismatch, got: {err}"
    );
}

// ── 9. 대형 shape 크기 검증 ─────────────────────────────────────────────────

/// Llama 3.2 1B 크기 (128256 × 2048) Q4_0 bytes 크기 계산 검증.
///
/// 의문 2의 "131072 bytes" 보고가 단위 오류임을 수치로 확인.
/// 실제 크기: 128256 * (2048/32) * 18 = 128256 * 64 * 18 = 147,750,912 bytes ≈ 140 MB.
/// 131,072 = 128KB — 명백히 잘못된 값으로, 실제 AUF 로드 시 위 fix가 올바른 크기를 검증.
#[test]
fn llama_1b_lm_head_size_calculation() {
    const LLAMA_VOCAB: usize = 128256;
    const LLAMA_HIDDEN: usize = 2048;
    const LLAMA_BLOCKS: usize = LLAMA_VOCAB * LLAMA_HIDDEN / 32;
    const LLAMA_BYTES: usize = LLAMA_BLOCKS * 18;

    // 147,750,912 bytes ≈ 140 MB
    assert_eq!(
        LLAMA_BYTES, 147_750_912,
        "Llama 1B lm_head Q4_0 bytes must be 147,750,912"
    );
    // LLAMA_BYTES = 147_750_912 ≈ 140 MB. Compare at runtime (const assertion causes clippy lint).
    let bytes_runtime = LLAMA_BYTES;
    assert!(
        bytes_runtime > 100_000_000,
        "must be > 100 MB (reporter's '131072 bytes' = 128 KB is ~1000x too small)"
    );

    // SOA 크기 == AOS 크기 (불변).
    let q_len = LLAMA_BLOCKS * 16;
    let d_len = LLAMA_BLOCKS * 2;
    assert_eq!(
        q_len + d_len,
        LLAMA_BYTES,
        "SOA total == AOS total for Llama 1B"
    );
}
