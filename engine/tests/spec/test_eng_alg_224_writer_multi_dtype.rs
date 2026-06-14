//! Spec 테스트: ENG-ALG-224 — AUF v0.2 multi-dtype writer 알고리즘 (Sprint C).
//!
//! ENG-ALG-224 (`spec/32-engine-algorithms.md` §3.12.18.1) writer 의무:
//!   1. dequant→requant 파이프라인이 결정적이다 (ENG-DAT-096.13). 같은 입력 + 같은 옵션 →
//!      byte-identical AUF.
//!   2. 동일 (layer_idx, kind) 그룹 내 entry는 (layer ASC, kind ASC, is_default DESC, dtype ASC)
//!      로 안정 정렬되어 default_dtype entry가 그룹 첫 번째에 온다 (INV-138 호환 의무).
//!   3. multi-dtype 모드 활성화 시 capability_optional bit 3 set + format_minor = 2.
//!   4. META.default_dtype 필드가 자동으로 set된다.
//!   5. dequant→requant는 `convert_tensor_dtype`를 호출하며 호스트 결정적이다.
//!
//! 본 테스트는 binary crate `auf_tool::cmd_build`를 직접 호출할 수 없으므로
//! writer + tensor_index + convert_tensor_dtype을 직접 호출하여 ENG-ALG-224의
//! 핵심 보증을 검증한다.
//!
//! INV-137 (multi-dtype shape 일치)도 함께 검증한다.

use argus_engine::auf::dtype_convert::convert_tensor_dtype;
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

fn make_meta_with_default(default_dtype: Option<&str>) -> AufMeta {
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

/// 테스트 fixture — 32-element f32 layer weight를 Q4_0/F16/F32 dtype별 bytes로 변환.
fn fixture_dtype_bytes() -> [(TensorDType, Vec<u8>); 3] {
    let f32_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let f32_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4).to_vec()
    };

    let f16_bytes =
        convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::F16, &[1, 32]).unwrap();
    let q4_bytes =
        convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::Q4_0, &[1, 32]).unwrap();

    [
        (TensorDType::Q4_0, q4_bytes),
        (TensorDType::F16, f16_bytes),
        (TensorDType::F32, f32_bytes),
    ]
}

// ── (1) 결정성 ────────────────────────────────────────────────────────────────

/// ENG-ALG-224.1: 같은 입력 + 같은 옵션으로 두 번 build → byte-identical.
///
/// dequant→requant 파이프라인의 결정성과 sort_by_key의 stability를 함께 보증한다.
#[test]
fn eng_alg_224_byte_determinism_two_runs() {
    let dtypes = fixture_dtype_bytes();

    // 동일한 multi-dtype tensor entries를 두 번 직접 구성하여 byte-identical 검증.
    let build = || -> Vec<u8> {
        let q4 = &dtypes[0];
        let f16 = &dtypes[1];

        let mut entries: Vec<TensorEntry> = vec![
            TensorEntry {
                layer_idx: 0,
                kind: TensorKind::AttnQ.as_u32(),
                dtype: q4.0.as_u32(),
                shape: vec![1, 32],
                alignment: 64,
                variant_offsets: vec![0],
                variant_sizes: vec![q4.1.len() as u64],
            },
            TensorEntry {
                layer_idx: 0,
                kind: TensorKind::AttnQ.as_u32(),
                dtype: f16.0.as_u32(),
                shape: vec![1, 32],
                alignment: 64,
                variant_offsets: vec![q4.1.len() as u64],
                variant_sizes: vec![f16.1.len() as u64],
            },
        ];

        // INV-138 정렬 키: (layer ASC, kind ASC, is_default(=Q4_0) DESC, dtype ASC)
        let default_u32 = TensorDType::Q4_0.as_u32();
        entries.sort_by_key(|e| {
            let not_default: u8 = if e.dtype == default_u32 { 0 } else { 1 };
            (e.layer_idx, e.kind, not_default, e.dtype)
        });

        let tidx = TensorIndex {
            variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_CPU_AOS)],
            entries,
        };

        let mut payload = Vec::new();
        payload.extend_from_slice(&q4.1);
        payload.extend_from_slice(&f16.1);

        AufWriter::new(
            make_meta_with_default(Some("Q4_0")),
            make_tokenizer(),
            [0u8; 32],
            0,
            0,
        )
        .with_multi_dtype(true)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, payload)
        .build()
        .unwrap()
    };

    let bytes_a = build();
    let bytes_b = build();
    assert_eq!(
        bytes_a, bytes_b,
        "two builds with same input must produce byte-identical AUF (ENG-DAT-096.13)"
    );
}

/// ENG-ALG-224.2: dequant→requant 파이프라인 자체의 결정성.
///
/// 같은 src_bytes + (src_dtype, dst_dtype, shape) → byte-identical 결과.
#[test]
fn eng_alg_224_convert_pipeline_deterministic() {
    let f32_data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let f32_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4).to_vec()
    };

    let q4_a =
        convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::Q4_0, &[2, 32]).unwrap();
    let q4_b =
        convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::Q4_0, &[2, 32]).unwrap();
    assert_eq!(q4_a, q4_b);

    let f16_a =
        convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::F16, &[2, 32]).unwrap();
    let f16_b =
        convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::F16, &[2, 32]).unwrap();
    assert_eq!(f16_a, f16_b);
}

// ── (2) INV-137: multi-dtype shape 일치 ─────────────────────────────────────────

/// ENG-ALG-224.3 / INV-137: 동일 (layer_idx, kind) 그룹의 모든 dtype entry는 동일 shape.
///
/// auf_tool은 같은 source tensor의 shape을 모든 dtype candidate에 그대로 전파하므로
/// shape 불일치가 발생해서는 안 된다. 본 테스트는 정상 케이스를 verify로 검증한다.
#[test]
fn eng_alg_224_inv137_shape_consistency() {
    let dtypes = fixture_dtype_bytes();
    let q4 = &dtypes[0];
    let f16 = &dtypes[1];

    let entries = vec![
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: q4.0.as_u32(),
            shape: vec![1, 32], // 동일
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![q4.1.len() as u64],
        },
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: f16.0.as_u32(),
            shape: vec![1, 32], // 동일
            alignment: 64,
            variant_offsets: vec![q4.1.len() as u64],
            variant_sizes: vec![f16.1.len() as u64],
        },
    ];
    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries,
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&q4.1);
    payload.extend_from_slice(&f16.1);

    let bytes = AufWriter::new(
        make_meta_with_default(Some("Q4_0")),
        make_tokenizer(),
        [0u8; 32],
        0,
        0,
    )
    .with_multi_dtype(true)
    .with_tensor_index(tidx)
    .add_weights_section(TAG_WEIGHTS_CPU_AOS, payload)
    .build()
    .unwrap();

    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    let group = view.tensor_index.entries_for(0, TensorKind::AttnQ.as_u32());
    assert_eq!(group.len(), 2);
    assert_eq!(group[0].shape, group[1].shape, "INV-137 shape match");
}

// ── (3) INV-138: writer 정렬 의무 ───────────────────────────────────────────────

/// ENG-ALG-224.4 / INV-138(b): default_dtype entry가 그룹 첫 번째에 와야 한다.
///
/// 입력 entries에서 F16이 먼저 등록되어도, INV-138 정렬 후 Q4_0(default)이 첫 번째.
#[test]
fn eng_alg_224_inv138_default_dtype_first_after_sort() {
    // 입력 순서: F16 먼저, Q4_0 두 번째.
    let dtypes = fixture_dtype_bytes();
    let q4 = &dtypes[0];
    let f16 = &dtypes[1];

    let mut entries = vec![
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::FfnGate.as_u32(),
            dtype: f16.0.as_u32(),
            shape: vec![1, 32],
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![f16.1.len() as u64],
        },
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::FfnGate.as_u32(),
            dtype: q4.0.as_u32(),
            shape: vec![1, 32],
            alignment: 64,
            variant_offsets: vec![f16.1.len() as u64],
            variant_sizes: vec![q4.1.len() as u64],
        },
    ];

    // ENG-ALG-224 정렬 적용 (auf_tool::build_tensor_index와 동일 키).
    let default_u32 = TensorDType::Q4_0.as_u32();
    entries.sort_by_key(|e| {
        let not_default: u8 = if e.dtype == default_u32 { 0 } else { 1 };
        (e.layer_idx, e.kind, not_default, e.dtype)
    });

    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries,
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&f16.1);
    payload.extend_from_slice(&q4.1);

    let bytes = AufWriter::new(
        make_meta_with_default(Some("Q4_0")),
        make_tokenizer(),
        [0u8; 32],
        0,
        0,
    )
    .with_multi_dtype(true)
    .with_tensor_index(tidx)
    .add_weights_section(TAG_WEIGHTS_CPU_AOS, payload)
    .build()
    .unwrap();

    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    let group = view
        .tensor_index
        .entries_for(0, TensorKind::FfnGate.as_u32());
    assert_eq!(group.len(), 2);
    assert_eq!(
        group[0].dtype,
        TensorDType::Q4_0.as_u32(),
        "INV-138: default_dtype (Q4_0) must be first in group"
    );
    assert_eq!(group[1].dtype, TensorDType::F16.as_u32());
}

// ── (4) capability bit 3 + format_minor 자동 활성화 ────────────────────────────

/// ENG-ALG-224.5: writer.with_multi_dtype(true) → bit 3 set + format_minor = 2.
#[test]
fn eng_alg_224_capability_bit3_and_format_minor_auto() {
    let bytes = AufWriter::new(
        make_meta_with_default(Some("Q4_0")),
        make_tokenizer(),
        [0u8; 32],
        0,
        0,
    )
    .with_multi_dtype(true)
    .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 64])
    .build()
    .unwrap();

    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    assert!(
        view.header.has_multi_dtype(),
        "capability_optional bit 3 must be set"
    );
    assert_eq!(view.header.format_minor, 2, "format_minor must be 2 (v0.2)");
    assert!(
        view.header.capability_optional & CAPABILITY_BIT_MULTI_DTYPE != 0,
        "MULTI_DTYPE_VARIANTS bit must be set"
    );
}

/// ENG-ALG-224.6: with_multi_dtype(false) → bit 3 unset + format_minor 1.
#[test]
fn eng_alg_224_single_dtype_keeps_v01_compat() {
    let bytes = AufWriter::new(
        make_meta_with_default(None),
        make_tokenizer(),
        [0u8; 32],
        0,
        0,
    )
    .with_multi_dtype(false)
    .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 64])
    .build()
    .unwrap();
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    assert!(!view.header.has_multi_dtype());
    assert_eq!(view.header.format_minor, 1);
}

// ── (5) Q4_0↔F32 round-trip via writer build ─────────────────────────────────

/// ENG-ALG-224.7: F32 → F16 → F32 round-trip이 모든 representable f16 값에서 정확.
#[test]
fn eng_alg_224_f32_f16_round_trip_in_pipeline() {
    let f32_src: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0, 100.0, -100.0, 2.5];
    let n = f32_src.len();
    let f32_bytes: Vec<u8> =
        unsafe { std::slice::from_raw_parts(f32_src.as_ptr() as *const u8, n * 4).to_vec() };

    let f16_bytes =
        convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::F16, &[n as u64]).unwrap();
    let back =
        convert_tensor_dtype(&f16_bytes, TensorDType::F16, TensorDType::F32, &[n as u64]).unwrap();
    let back_f32 = unsafe { std::slice::from_raw_parts(back.as_ptr() as *const f32, n) };
    for (i, (&a, &b)) in f32_src.iter().zip(back_f32.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-2,
            "f16 round-trip mismatch at idx {i}: {a} vs {b}"
        );
    }
}

/// ENG-ALG-224.8: Q4_0 self round-trip은 byte-identical (Q4_0 quant 결정적).
#[test]
fn eng_alg_224_q4_0_self_round_trip_byte_identical() {
    // 32 floats one block. 일정한 분포로 정해진 input.
    let f32_src: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.07).collect();
    let f32_bytes: Vec<u8> =
        unsafe { std::slice::from_raw_parts(f32_src.as_ptr() as *const u8, 32 * 4).to_vec() };

    let q4_v1 =
        convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::Q4_0, &[1, 32]).unwrap();
    let f32_back =
        convert_tensor_dtype(&q4_v1, TensorDType::Q4_0, TensorDType::F32, &[1, 32]).unwrap();
    let q4_v2 =
        convert_tensor_dtype(&f32_back, TensorDType::F32, TensorDType::Q4_0, &[1, 32]).unwrap();

    assert_eq!(
        q4_v1, q4_v2,
        "Q4_0 self round-trip must be byte-identical (ENG-DAT-096.13)"
    );
}
