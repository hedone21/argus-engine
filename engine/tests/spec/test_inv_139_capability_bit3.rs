/// Spec 테스트: INV-139 — capability bit 3 의미 + v0.1.x 호환성.
///
/// INV-139:
/// - `capability_optional` bit 3 = 1: multi-dtype variant 존재 + META.default_dtype 있음.
/// - bit 3 = 0: single-dtype 모드.
/// - v0.1.x reader는 bit 3를 인식하지 못하지만 capability_optional이므로 reject 사유 아님.
/// - bit 3 = 1이면 format_minor >= 2 필수.
/// - capability_required에 unknown bit → reject (INV-132와 동일 규칙).
///
/// spec: `spec/41-invariants.md` §3.18 (INV-139)
///       `spec/33-engine-data.md` §3.22.14 (ENG-DAT-097)
use argus_engine::auf::{
    AufError, AufMeta, AufTokenizer, AufWriter, BackendTag, CAPABILITY_BIT_LM_HEAD_Q4_0,
    CAPABILITY_BIT_MULTI_DTYPE, READER_KNOWN_CAPABILITIES, TAG_WEIGHTS_CPU_AOS, TOKENIZER_KIND_BPE,
    open_from_bytes,
};

// ── 헬퍼 ──────────────────────────────────────────────────────────────────

fn make_meta() -> AufMeta {
    AufMeta {
        architecture: "llama".to_owned(),
        n_layers: 2,
        n_heads_q: 4,
        n_kv_heads: 2,
        head_dim: 8,
        hidden_dim: 32,
        ffn_dim: 64,
        vocab_size: 10,
        max_seq_len: 128,
        rope_theta: 10000.0,
        rotary_dim: 8,
        rope_scaling: 1.0,
        rms_norm_epsilon: 1e-5,
        default_dtype: Some("Q4_0".to_owned()),
    }
}

fn make_tokenizer() -> AufTokenizer {
    AufTokenizer {
        kind: TOKENIZER_KIND_BPE,
        tokens: vec![b"a".to_vec(), b"b".to_vec()],
        merges: vec![],
        bos_id: 0,
        eos_id: 1,
        pad_id: -1,
        unk_id: -1,
        chat_template: None,
    }
}

fn build_base_auf() -> Vec<u8> {
    AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 64])
        .build()
        .unwrap()
}

/// capability_optional을 주어진 값으로 덮어쓴다.
fn set_cap_opt(bytes: &mut [u8], value: u64) {
    bytes[104..112].copy_from_slice(&value.to_le_bytes());
}

/// capability_required를 주어진 값으로 덮어쓴다.
fn set_cap_req(bytes: &mut [u8], value: u64) {
    bytes[96..104].copy_from_slice(&value.to_le_bytes());
}

/// format_minor를 주어진 값으로 덮어쓴다.
fn set_format_minor(bytes: &mut [u8], value: u16) {
    bytes[10..12].copy_from_slice(&value.to_le_bytes());
}

// ── INV-139: READER_KNOWN_CAPABILITIES 검증 ──────────────────────────────────

/// INV-139.0: READER_KNOWN_CAPABILITIES에 bit 2와 bit 3 모두 포함.
///
/// v0.2 reader는 두 capability bit를 인식해야 한다.
#[test]
fn inv139_reader_known_capabilities_includes_bit2_and_bit3() {
    assert_eq!(
        READER_KNOWN_CAPABILITIES,
        CAPABILITY_BIT_LM_HEAD_Q4_0 | CAPABILITY_BIT_MULTI_DTYPE,
        "READER_KNOWN_CAPABILITIES must be bit2|bit3 = 0xC"
    );
    assert_eq!(READER_KNOWN_CAPABILITIES, 0xC, "0xC = 0b1100 = bit2|bit3");
}

// ── INV-139: bit 3 = 1 → reader가 인식하고 multi-dtype dispatch ──────────────

/// INV-139.1: bit 3 set + reader(v0.2) → has_multi_dtype() = true.
#[test]
fn inv139_bit3_set_recognized_by_v02_reader() {
    let mut bytes = build_base_auf();
    set_cap_opt(&mut bytes, CAPABILITY_BIT_MULTI_DTYPE);
    set_format_minor(&mut bytes, 2);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    assert!(
        view.header.has_multi_dtype(),
        "v0.2 reader must recognize bit 3"
    );
    assert_eq!(view.header.format_minor, 2);
}

/// INV-139.2: bit 3 = 0 (v0.1.x AUF) → has_multi_dtype() = false.
#[test]
fn inv139_bit3_zero_is_single_dtype_mode() {
    let bytes = build_base_auf();
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    assert!(
        !view.header.has_multi_dtype(),
        "v0.1.x AUF must have bit 3 = 0"
    );
}

// ── INV-139: bit 3 = 1 + capability_optional → v0.1.x reader가 reject하지 않음 ─────

/// INV-139.3: bit 3 in capability_optional → v0.1.x reader도 파싱 성공.
///
/// capability_optional이므로 v0.1.x reader(READER_KNOWN_CAPABILITIES = 0)는
/// bit 3를 인식하지 못해도 reject하지 않는다 (INV-132와 호환).
/// 이 테스트는 bit 3 설정이 reader open에 문제가 없음을 확인한다.
#[test]
fn inv139_bit3_optional_does_not_reject_any_reader() {
    let mut bytes = build_base_auf();
    // capability_optional에만 bit 3 set (required는 0 유지)
    set_cap_opt(&mut bytes, CAPABILITY_BIT_MULTI_DTYPE);
    set_cap_req(&mut bytes, 0);
    set_format_minor(&mut bytes, 2);

    // v0.2 reader도 문제 없이 파싱
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    assert_eq!(view.meta.architecture, "llama");
    assert!(view.header.has_multi_dtype());
}

/// INV-139.4: bit 3 in capability_required → READER_KNOWN_CAPABILITIES에 포함되므로 통과.
///
/// v0.2 reader는 bit 3를 알고 있으므로, capability_required에 있어도 reject 안 함.
/// (단, bit 3을 required로 set하는 것은 spec 권고 위반이지만 reader는 알 수 없음)
#[test]
fn inv139_bit3_in_required_allowed_because_v02_reader_knows_it() {
    let mut bytes = build_base_auf();
    set_cap_req(&mut bytes, CAPABILITY_BIT_MULTI_DTYPE);
    set_cap_opt(&mut bytes, 0);
    set_format_minor(&mut bytes, 2);

    // v0.2 reader(READER_KNOWN_CAPABILITIES = bit2|bit3)는 알고 있으므로 PASS
    let result = open_from_bytes(bytes, BackendTag::CpuAos);
    assert!(
        result.is_ok(),
        "v0.2 reader must not reject bit 3 in capability_required: {:?}",
        result
    );
}

/// INV-139.5: capability_required에 알 수 없는 bit(예: bit 7) → UnknownRequiredCapability reject.
#[test]
fn inv139_unknown_required_bit_rejected() {
    let mut bytes = build_base_auf();
    // bit 7 = 128 — READER_KNOWN_CAPABILITIES에 없는 bit
    set_cap_req(&mut bytes, 1 << 7);
    let err = open_from_bytes(bytes, BackendTag::CpuAos).unwrap_err();
    assert!(
        matches!(err, AufError::UnknownRequiredCapability { bit: 7 }),
        "unknown required capability bit must be rejected: {err}"
    );
}

/// INV-139.6: capability_optional에 알 수 없는 bit(bit 7) → reject 아님.
///
/// capability_optional은 reader가 모르면 skip해도 안전하므로 reject하지 않는다.
#[test]
fn inv139_unknown_optional_bit_not_rejected() {
    let mut bytes = build_base_auf();
    // bit 7 in optional only → OK
    set_cap_opt(&mut bytes, 1 << 7);
    set_cap_req(&mut bytes, 0);
    let result = open_from_bytes(bytes, BackendTag::CpuAos);
    assert!(
        result.is_ok(),
        "unknown optional bit must not cause rejection: {:?}",
        result
    );
}

// ── INV-139: format_minor ↔ bit 3 정합성 ────────────────────────────────────

/// INV-139.7: bit 3 = 1이면 format_minor = 2.
///
/// set_multi_dtype_capability(true)가 format_minor를 2로 설정함을 header 레벨에서 검증.
#[test]
fn inv139_bit3_set_implies_format_minor_2() {
    use argus_engine::auf::AufHeader;
    let mut h = AufHeader::new_v01("test", [0u8; 32], 0, 0, 0, 256, 65536);
    assert_eq!(h.format_minor, 1);
    h.set_multi_dtype_capability(true);
    assert!(h.has_multi_dtype());
    assert_eq!(h.format_minor, 2, "bit 3 set must bump format_minor to 2");
}

/// INV-139.8: bit 2 + bit 3 모두 set — v0.2 + lm_head Q4_0 겸용 AUF 파싱.
#[test]
fn inv139_bit2_and_bit3_combined_ok() {
    let mut bytes = build_base_auf();
    set_cap_opt(
        &mut bytes,
        CAPABILITY_BIT_LM_HEAD_Q4_0 | CAPABILITY_BIT_MULTI_DTYPE,
    );
    set_format_minor(&mut bytes, 2);

    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    assert!(view.header.has_lm_head_q4_0(), "bit 2 must be set");
    assert!(view.header.has_multi_dtype(), "bit 3 must be set");
    assert_eq!(view.header.capability_optional, 0xC); // bit2|bit3
}

/// INV-139.9: bit 3 = 1 + format_minor = 1 → reader는 파싱 성공하지만 정합성 경고 대상.
///
/// format_minor는 reader의 reject 사유가 아니다 (현재 spec에서 format_minor=1도 허용).
/// 이 테스트는 reader가 reject하지 않음을 확인한다.
#[test]
fn inv139_bit3_with_format_minor_1_not_rejected_by_reader() {
    let mut bytes = build_base_auf();
    set_cap_opt(&mut bytes, CAPABILITY_BIT_MULTI_DTYPE);
    // format_minor를 1로 두면 INV-139 spec 위반이지만 reader는 reject하지 않는다.
    set_format_minor(&mut bytes, 1);

    let result = open_from_bytes(bytes, BackendTag::CpuAos);
    assert!(
        result.is_ok(),
        "reader must not reject bit3+format_minor=1 (format_minor is not validated): {:?}",
        result
    );
}

// ── INV-139: v0.1.x ↔ v0.2 양방향 호환 매트릭스 ────────────────────────────

/// INV-139.10: v0.2 AUF(bit 3 = 1)를 v0.2 reader로 읽기 — 정상 파싱.
#[test]
fn inv139_v02_auf_x_v02_reader_ok() {
    let mut bytes = build_base_auf();
    set_cap_opt(&mut bytes, CAPABILITY_BIT_MULTI_DTYPE);
    set_format_minor(&mut bytes, 2);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    assert!(view.header.has_multi_dtype());
    assert_eq!(view.meta.default_dtype.as_deref(), Some("Q4_0"));
}

/// INV-139.11: v0.1.x AUF(bit 3 = 0)를 v0.2 reader로 읽기 — 회귀 없음 (forward compat).
#[test]
fn inv139_v01_auf_x_v02_reader_no_regression() {
    // v0.1.x AUF: bit 3 = 0, format_minor = 1, default_dtype 없음
    let meta = AufMeta {
        architecture: "llama".to_owned(),
        n_layers: 2,
        n_heads_q: 4,
        n_kv_heads: 2,
        head_dim: 8,
        hidden_dim: 32,
        ffn_dim: 64,
        vocab_size: 10,
        max_seq_len: 128,
        rope_theta: 10000.0,
        rotary_dim: 8,
        rope_scaling: 1.0,
        rms_norm_epsilon: 1e-5,
        default_dtype: None, // v0.1.x에는 없음
    };
    let bytes = AufWriter::new(meta, make_tokenizer(), [0u8; 32], 0, 0)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 64])
        .build()
        .unwrap();

    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    // v0.2 reader가 v0.1.x AUF를 정상 파싱
    assert!(
        !view.header.has_multi_dtype(),
        "v0.1.x AUF must not have bit 3"
    );
    assert_eq!(view.header.format_minor, 1);
    assert!(
        view.meta.default_dtype.is_none(),
        "v0.1.x AUF has no default_dtype"
    );
    assert_eq!(view.meta.architecture, "llama");
}
