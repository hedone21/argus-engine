/// Spec 테스트: INV-138 — default_dtype 의무 + writer 안정 정렬.
///
/// INV-138:
///   (a) capability_optional bit 3 = 1이면 META JSON에 `default_dtype` 필드가 반드시 존재.
///   (b) writer는 entries를 (layer_idx ASC, kind ASC, is_default DESC, dtype ASC)로 안정 정렬하여
///       default_dtype entry가 동일 (layer_idx, kind) 그룹의 가장 앞에 오도록 보장.
///   (c) reader dtype selection precedence: 호출자 명시 > META.default_dtype > first-match.
///
/// spec: `spec/41-invariants.md` §3.18 (INV-138)
///       `spec/33-engine-data.md` §3.22.15 (ENG-DAT-098), §3.22.16 (ENG-DAT-099)
///       `spec/32-engine-algorithms.md` §3.12.18 (ENG-ALG-224)
use argus_engine::auf::{
    AufMeta, AufTokenizer, AufWriter, BackendTag, CAPABILITY_BIT_MULTI_DTYPE, TAG_WEIGHTS_CPU_AOS,
    TOKENIZER_KIND_BPE, TensorDType, TensorEntry, TensorIndex, TensorKind, open_from_bytes,
};

// ── 헬퍼 ──────────────────────────────────────────────────────────────────

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

fn make_meta_with_default_dtype(default_dtype: Option<&str>) -> AufMeta {
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
        default_dtype: default_dtype.map(|s| s.to_owned()),
    }
}

fn make_variant_tag_buf(s: &str) -> [u8; 24] {
    let mut buf = [0u8; 24];
    let b = s.as_bytes();
    buf[..b.len().min(24)].copy_from_slice(&b[..b.len().min(24)]);
    buf
}

/// bit 3 set + format_minor=2로 AUF 바이트를 빌드한다.
fn build_v02_auf(meta: AufMeta, entries: Vec<TensorEntry>) -> Vec<u8> {
    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries,
    };
    let mut bytes = AufWriter::new(meta, make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 64])
        .build()
        .unwrap();

    // capability_optional bit 3 수동 set
    let cap_opt = u64::from_le_bytes(bytes[104..112].try_into().unwrap());
    bytes[104..112].copy_from_slice(&(cap_opt | CAPABILITY_BIT_MULTI_DTYPE).to_le_bytes());
    // format_minor = 2
    bytes[10..12].copy_from_slice(&2u16.to_le_bytes());

    bytes
}

// ── INV-138(a): default_dtype 의무 ──────────────────────────────────────────

/// INV-138.1: bit 3 = 1 + META에 default_dtype 있음 → 정상 파싱.
#[test]
fn inv138_bit3_set_with_default_dtype_ok() {
    let meta = make_meta_with_default_dtype(Some("Q4_0"));
    let bytes = build_v02_auf(meta, vec![]);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    assert!(view.header.has_multi_dtype(), "bit 3 must be set");
    assert_eq!(
        view.meta.default_dtype.as_deref(),
        Some("Q4_0"),
        "default_dtype must be preserved"
    );
}

/// INV-138.2: v0.1.x AUF(bit 3 = 0) — default_dtype 없어도 정상 파싱 (하위 호환).
#[test]
fn inv138_v01_auf_without_default_dtype_ok() {
    let meta = make_meta_with_default_dtype(None);
    let bytes = AufWriter::new(meta, make_tokenizer(), [0u8; 32], 0, 0)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 64])
        .build()
        .unwrap();
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    assert!(!view.header.has_multi_dtype(), "bit 3 must be 0");
    assert!(
        view.meta.default_dtype.is_none(),
        "default_dtype must be None in v0.1.x AUF"
    );
}

/// INV-138.3: default_dtype round-trip — META JSON에서 올바르게 직렬화/역직렬화.
#[test]
fn inv138_default_dtype_round_trip_in_meta_json() {
    for dtype_str in &["F32", "F16", "BF16", "Q4_0", "Q4_1", "Q8_0", "U8"] {
        let meta = make_meta_with_default_dtype(Some(dtype_str));
        let bytes = meta.to_json_bytes().unwrap();
        let meta2 = AufMeta::from_json_bytes(&bytes).unwrap();
        assert_eq!(
            meta2.default_dtype.as_deref(),
            Some(*dtype_str),
            "dtype_str={dtype_str}"
        );
    }
}

// ── INV-138(b): writer 정렬 의무 ─────────────────────────────────────────────

/// INV-138.4: AUF v0.2 entries 정렬 — (layer_idx ASC, kind ASC, is_default DESC, dtype ASC).
///
/// default_dtype("Q4_0")에 해당하는 entry가 동일 (layer, kind) 그룹의 가장 앞에 와야 한다.
/// 이 테스트는 AufWriter가 아직 v0.2 정렬을 지원하지 않으므로(Sprint C 이전),
/// TensorIndex를 직접 구성하여 정렬 결과를 검증한다.
#[test]
fn inv138_default_dtype_entry_is_first_in_group() {
    // default_dtype = "Q4_0". F16 entry를 먼저 넣었다가, Q4_0이 앞에 와야 하는 조건.
    // 현재(Sprint B)에서는 writer 정렬 기능이 없으므로 수동 정렬된 entries를 사용한다.
    // 이 테스트는 정렬된 entries가 올바른지 검증 (INV-138 호환성 보증).
    let q4_first = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnQ.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(), // default_dtype = Q4_0 → 첫 번째
        shape: vec![32, 64],
        alignment: 64,
        variant_offsets: vec![0],
        variant_sizes: vec![256],
    };
    let f16_second = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnQ.as_u32(),
        dtype: TensorDType::F16.as_u32(), // 두 번째
        shape: vec![32, 64],
        alignment: 64,
        variant_offsets: vec![256],
        variant_sizes: vec![512],
    };

    let meta = make_meta_with_default_dtype(Some("Q4_0"));
    // entries는 직접 정렬 순서로 주입 (writer 정렬 기능 Sprint C에서 구현 예정)
    let bytes = build_v02_auf(meta, vec![q4_first, f16_second]);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    let entries = view.tensor_index.entries_for(0, TensorKind::AttnQ.as_u32());
    assert_eq!(entries.len(), 2);

    // INV-138(b): first-match가 default_dtype(Q4_0)이어야 함
    assert_eq!(
        entries[0].dtype,
        TensorDType::Q4_0.as_u32(),
        "default_dtype entry must be first in group (INV-138 writer sort)"
    );
    assert_eq!(
        entries[1].dtype,
        TensorDType::F16.as_u32(),
        "F16 entry must be second"
    );
}

/// INV-138.5: v0.1.x reader가 v0.2 AUF의 entries를 읽을 때 first-match = default_dtype.
///
/// writer 정렬 의무(INV-138(b)) 덕분에, v0.1.x reader는 single-dtype 모드로 동작하더라도
/// first-match가 default_dtype entry임이 보장된다.
#[test]
fn inv138_v01x_reader_first_match_is_default_dtype() {
    let q4_first = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::FfnGate.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![64, 32],
        alignment: 64,
        variant_offsets: vec![0],
        variant_sizes: vec![512],
    };
    let f16_second = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::FfnGate.as_u32(),
        dtype: TensorDType::F16.as_u32(),
        shape: vec![64, 32],
        alignment: 64,
        variant_offsets: vec![512],
        variant_sizes: vec![1024],
    };

    let meta = make_meta_with_default_dtype(Some("Q4_0"));
    let bytes = build_v02_auf(meta, vec![q4_first, f16_second]);

    // v0.1.x reader처럼 동작: bit 3은 optional이므로 파싱은 성공.
    // entries_for()의 첫 번째 entry = Q4_0 (default_dtype).
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    let entries = view
        .tensor_index
        .entries_for(0, TensorKind::FfnGate.as_u32());
    assert!(
        !entries.is_empty(),
        "v0.1.x reader must see at least one entry"
    );
    assert_eq!(
        entries[0].dtype,
        TensorDType::Q4_0.as_u32(),
        "first-match must be Q4_0 (default_dtype), ensuring v0.1.x compat"
    );
}

// ── INV-138(c): dtype selection precedence ──────────────────────────────────

/// INV-138.6: find_entry_by_dtype()는 명시 dtype을 우선 선택한다.
///
/// precedence: 호출자 명시 dtype > META.default_dtype > first-match.
/// 여기서는 find_entry_by_dtype(F16)이 Q4_0(default)보다 우선한다.
#[test]
fn inv138_explicit_dtype_wins_over_default() {
    let q4_first = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnV.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![16, 64],
        alignment: 64,
        variant_offsets: vec![0],
        variant_sizes: vec![128],
    };
    let f16_second = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnV.as_u32(),
        dtype: TensorDType::F16.as_u32(),
        shape: vec![16, 64],
        alignment: 64,
        variant_offsets: vec![128],
        variant_sizes: vec![256],
    };

    let meta = make_meta_with_default_dtype(Some("Q4_0"));
    let bytes = build_v02_auf(meta, vec![q4_first, f16_second]);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    // 명시 dtype = F16 → Q4_0(default)보다 우선
    let explicit = view.tensor_index.find_entry_by_dtype(
        0,
        TensorKind::AttnV.as_u32(),
        TensorDType::F16.as_u32(),
    );
    assert!(
        explicit.is_some(),
        "explicit F16 lookup must succeed even though default is Q4_0"
    );
    assert_eq!(explicit.unwrap().dtype, TensorDType::F16.as_u32());
}

/// INV-138.7: default_dtype = None인 v0.1.x AUF에서 META JSON에 default_dtype 미포함.
#[test]
fn inv138_v01_meta_no_default_dtype_field() {
    let meta = make_meta_with_default_dtype(None);
    let json_bytes = meta.to_json_bytes().unwrap();
    let json_str = std::str::from_utf8(&json_bytes).unwrap();
    assert!(
        !json_str.contains("default_dtype"),
        "v0.1.x META JSON must not include default_dtype field: {json_str}"
    );
}

/// INV-138.8: 같은 meta + entries + writer로 두 번 build → byte-identical (determinism).
///
/// INV-138(b) writer 정렬 의무는 byte determinism을 내포한다.
/// 현재 Sprint B에서는 AufWriter가 고정 순서 entries를 그대로 직렬화하므로 deterministic.
#[test]
fn inv138_writer_determinism() {
    let entries = vec![
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![32, 64],
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![256],
        },
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::F16.as_u32(),
            shape: vec![32, 64],
            alignment: 64,
            variant_offsets: vec![256],
            variant_sizes: vec![512],
        },
    ];

    let meta = make_meta_with_default_dtype(Some("Q4_0"));
    let bytes1 = build_v02_auf(meta.clone(), entries.clone());
    let bytes2 = build_v02_auf(meta, entries);

    assert_eq!(
        bytes1, bytes2,
        "same input must produce byte-identical AUF (determinism)"
    );
}
