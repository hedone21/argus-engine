/// Spec 테스트: INV-137 — AUF v0.2 multi-dtype shape 일치 의무.
///
/// INV-137: capability_optional bit 3(MULTI_DTYPE_VARIANTS) = 1이면, 동일 (`layer_idx`, `kind`)에
/// 등록된 모든 dtype 후보 entry는 동일 `shape_rank`와 동일 `shape` 값을 가져야 한다.
/// lm_head(`kind=11`)도 multi-dtype 후보 그룹에 포함된다.
///
/// spec: `spec/41-invariants.md` §3.18 (INV-137)
///       `spec/33-engine-data.md` §3.22.14 (ENG-DAT-097)
use argus_engine::auf::{
    AufMeta, AufTokenizer, AufWriter, BackendTag, CAPABILITY_BIT_MULTI_DTYPE, LAYER_IDX_CROSS,
    TAG_WEIGHTS_CPU_AOS, TOKENIZER_KIND_BPE, TensorDType, TensorEntry, TensorIndex, TensorKind,
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

fn make_variant_tag_buf(s: &str) -> [u8; 24] {
    let mut buf = [0u8; 24];
    let b = s.as_bytes();
    buf[..b.len().min(24)].copy_from_slice(&b[..b.len().min(24)]);
    buf
}

/// capability bit 3 = 1이 설정된 AUF 바이트를 빌드한다.
///
/// `entries`를 TENSOR_INDEX에 직접 주입하여 shape 일치/불일치 케이스를 커버한다.
fn build_multi_dtype_auf(entries: Vec<TensorEntry>) -> Vec<u8> {
    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries,
    };
    let mut bytes = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 64])
        .build()
        .unwrap();

    // capability_optional bit 3 수동 set (AufWriter는 Sprint C 이전에는 자동 set 미지원)
    // header capability_optional은 offset 104..112
    let cap_opt = u64::from_le_bytes(bytes[104..112].try_into().unwrap());
    let new_cap_opt = cap_opt | CAPABILITY_BIT_MULTI_DTYPE;
    bytes[104..112].copy_from_slice(&new_cap_opt.to_le_bytes());
    // format_minor를 2로 set (offset 10..12)
    bytes[10..12].copy_from_slice(&2u16.to_le_bytes());

    bytes
}

// ── INV-137 테스트 ──────────────────────────────────────────────────────────

/// INV-137.1: 동일 (layer_idx, kind)에 dtype이 다른 entry 2개가 lookup 가능하다.
///
/// multi-dtype 모드에서 Q4_0과 F16 두 entry가 모두 TensorIndex에 등록되고
/// entries_for()로 둘 다 조회 가능해야 한다.
#[test]
fn inv137_multi_dtype_entries_both_lookable() {
    let q4_entry = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnQ.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![32, 64],
        alignment: 64,
        variant_offsets: vec![0],
        variant_sizes: vec![256],
    };
    let f16_entry = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnQ.as_u32(),
        dtype: TensorDType::F16.as_u32(),
        shape: vec![32, 64],
        alignment: 64,
        variant_offsets: vec![256],
        variant_sizes: vec![512],
    };

    let bytes = build_multi_dtype_auf(vec![q4_entry, f16_entry]);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    assert!(
        view.header.has_multi_dtype(),
        "capability bit 3 must be set"
    );

    let candidates = view.tensor_index.entries_for(0, TensorKind::AttnQ.as_u32());
    assert_eq!(
        candidates.len(),
        2,
        "must find both Q4_0 and F16 entries for (layer=0, kind=AttnQ)"
    );

    let dtypes: Vec<u32> = candidates.iter().map(|e| e.dtype).collect();
    assert!(
        dtypes.contains(&TensorDType::Q4_0.as_u32()),
        "Q4_0 entry must be present"
    );
    assert!(
        dtypes.contains(&TensorDType::F16.as_u32()),
        "F16 entry must be present"
    );
}

/// INV-137.2: 동일 (layer_idx, kind)에 dtype이 같은 entry가 2개인 경우 entries_for가 2개 반환.
///
/// 이는 spec 위반이지만 reader는 reject하지 않고 그대로 반환한다. reader는 검증 의무 없음.
/// (writer build 시 자동 충족이 주 방어선, INV-137)
#[test]
fn inv137_same_dtype_entries_returned_as_is() {
    let e1 = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnK.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![16, 64],
        alignment: 64,
        variant_offsets: vec![0],
        variant_sizes: vec![128],
    };
    let e2 = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnK.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(), // 동일 dtype
        shape: vec![16, 64],
        alignment: 64,
        variant_offsets: vec![128],
        variant_sizes: vec![128],
    };

    let bytes = build_multi_dtype_auf(vec![e1, e2]);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    let candidates = view.tensor_index.entries_for(0, TensorKind::AttnK.as_u32());
    assert_eq!(candidates.len(), 2);
}

/// INV-137.3: lm_head도 multi-dtype 후보 그룹에 포함된다 (Sprint A' 반전).
///
/// lm_head(kind=11)에 Q4_0 + F16 entry를 등록하고 둘 다 조회 가능한지 확인.
#[test]
fn inv137_lm_head_multi_dtype_entries_lookable() {
    let lm_q4 = TensorEntry {
        layer_idx: LAYER_IDX_CROSS,
        kind: TensorKind::LmHead.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![10, 32],
        alignment: 65536,
        variant_offsets: vec![0],
        variant_sizes: vec![180],
    };
    let lm_f16 = TensorEntry {
        layer_idx: LAYER_IDX_CROSS,
        kind: TensorKind::LmHead.as_u32(),
        dtype: TensorDType::F16.as_u32(),
        shape: vec![10, 32], // 동일 shape — INV-137 준수
        alignment: 65536,
        variant_offsets: vec![180],
        variant_sizes: vec![640],
    };

    let bytes = build_multi_dtype_auf(vec![lm_q4, lm_f16]);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    let lm_entries = view
        .tensor_index
        .entries_for(LAYER_IDX_CROSS, TensorKind::LmHead.as_u32());
    assert_eq!(
        lm_entries.len(),
        2,
        "lm_head must allow multi-dtype candidates (Sprint A' rev)"
    );

    // shape 일치 확인 (INV-137)
    let shapes: Vec<&Vec<u64>> = lm_entries.iter().map(|e| &e.shape).collect();
    assert_eq!(
        shapes[0], shapes[1],
        "all lm_head multi-dtype entries must have identical shape (INV-137)"
    );
}

/// INV-137.4: find_entry_by_dtype()로 특정 dtype의 lm_head entry를 조회한다.
#[test]
fn inv137_find_lm_head_by_dtype() {
    let lm_q4 = TensorEntry {
        layer_idx: LAYER_IDX_CROSS,
        kind: TensorKind::LmHead.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![10, 32],
        alignment: 65536,
        variant_offsets: vec![0],
        variant_sizes: vec![180],
    };
    let lm_f16 = TensorEntry {
        layer_idx: LAYER_IDX_CROSS,
        kind: TensorKind::LmHead.as_u32(),
        dtype: TensorDType::F16.as_u32(),
        shape: vec![10, 32],
        alignment: 65536,
        variant_offsets: vec![256],
        variant_sizes: vec![640],
    };

    let bytes = build_multi_dtype_auf(vec![lm_q4, lm_f16]);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    let found_q4 = view.tensor_index.find_entry_by_dtype(
        LAYER_IDX_CROSS,
        TensorKind::LmHead.as_u32(),
        TensorDType::Q4_0.as_u32(),
    );
    assert!(found_q4.is_some(), "must find lm_head Q4_0 entry");
    assert_eq!(found_q4.unwrap().variant_sizes[0], 180);

    let found_f16 = view.tensor_index.find_entry_by_dtype(
        LAYER_IDX_CROSS,
        TensorKind::LmHead.as_u32(),
        TensorDType::F16.as_u32(),
    );
    assert!(found_f16.is_some(), "must find lm_head F16 entry");
    assert_eq!(found_f16.unwrap().variant_sizes[0], 640);

    // BF16은 없음
    let not_found = view.tensor_index.find_entry_by_dtype(
        LAYER_IDX_CROSS,
        TensorKind::LmHead.as_u32(),
        TensorDType::BF16.as_u32(),
    );
    assert!(not_found.is_none(), "BF16 entry must not be found");
}

/// INV-137.5: bit 3 = 0인 단일 dtype AUF에서 entries_for는 entry 1개만 반환한다.
#[test]
fn inv137_single_dtype_mode_returns_one_entry() {
    let e = TensorEntry {
        layer_idx: 0,
        kind: TensorKind::FfnGate.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![4096, 2048],
        alignment: 64,
        variant_offsets: vec![0],
        variant_sizes: vec![1024],
    };
    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries: vec![e],
    };
    let bytes = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 64])
        .build()
        .unwrap();

    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    // bit 3 = 0
    assert!(!view.header.has_multi_dtype());

    let candidates = view
        .tensor_index
        .entries_for(0, TensorKind::FfnGate.as_u32());
    assert_eq!(
        candidates.len(),
        1,
        "single-dtype mode: exactly 1 entry per (layer, kind)"
    );
}
