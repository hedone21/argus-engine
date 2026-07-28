//! AUF end-to-end shape propagation sanity (ENG-ALG-223 / ENG-DAT-096).
//!
//! 목적: auf_tool build → AUF 파일 → SecondaryMmap 로드까지의 전 경로에서
//! TensorEntry.shape가 올바르게 채워지고, swap_executor의 shape 검증 단계를
//! 통과하는지 호스트에서 검증한다. 디바이스 측정 이전에 동일한 차단 조건을
//! 사전 발견하는 안전망 역할.
//!
//! 검증 항목:
//! 1. AUF에 저장된 logical shape가 SecondaryMmap.layer_tensor()에서 GGUF
//!    order(innermost-first)로 올바르게 복원되는지.
//! 2. 복원된 dims를 swap_executor 방식(rev → primary 비교)으로 검증하면 일치.
//! 3. 빈 shape(vec![])이 SecondaryMmap에 들어오면 swap_executor 방식으로
//!    shape mismatch를 재현할 수 있는지.
//! 4. dtype round-trip (Q4_0, F16, F32).

use argus_engine::auf::q4_0_aos_to_adreno_soa;
use argus_engine::auf::reader::open_from_bytes;
use argus_engine::auf::section::{TAG_WEIGHTS_ADRENO_SOA, TAG_WEIGHTS_CPU_AOS};
use argus_engine::auf::tensor_index::{TensorDType, TensorEntry, TensorIndex, TensorKind};
use argus_engine::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
use argus_engine::auf::writer::AufWriter;
use argus_engine::auf::{AufMeta, BackendTag};
use argus_engine::buffer::DType;
use argus_engine::model_config::{ModelArch, ModelConfig};
use argus_engine::models::weights::build_auf_secondary_from_view;
use std::path::Path;

// ── fixture helpers ──────────────────────────────────────────────────────────

fn make_meta(n_layers: u32, head_dim: u32, hidden_dim: u32, ffn_dim: u32) -> AufMeta {
    AufMeta {
        architecture: "llama".to_owned(),
        n_layers,
        n_heads_q: 2,
        n_kv_heads: 1,
        head_dim,
        hidden_dim,
        ffn_dim,
        vocab_size: 4,
        max_seq_len: 32,
        rope_theta: 10000.0,
        rotary_dim: head_dim,
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

fn make_config(n_layers: usize, head_dim: usize, hidden_dim: usize, ffn_dim: usize) -> ModelConfig {
    ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: hidden_dim,
        num_hidden_layers: n_layers,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim,
        intermediate_size: ffn_dim,
        vocab_size: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_ids: vec![2],
        weight_prefix: String::new(),
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
    }
}

/// 단일 variant tag 배열 생성 헬퍼.
fn tag_buf(s: &str) -> [u8; 24] {
    let mut buf = [0u8; 24];
    let b = s.as_bytes();
    buf[..b.len().min(24)].copy_from_slice(&b[..b.len().min(24)]);
    buf
}

// ── 테스트 1: shape 올바른 경우 → SwapExecutor 방식 검증 통과 ──────────────

/// AUF에 logical order (outermost-first) shape을 저장하면 SecondaryMmap에서
/// GGUF order (innermost-first)로 복원되고, swap_executor 방식으로 다시
/// outermost-first로 변환하면 primary shape와 일치해야 한다.
///
/// 시나리오:
///   - attn_q: primary shape (outermost-first) = [64, 32]
///   - AUF shape (logical/outermost-first) = [64, 32]
///   - secondary_mmap dims (innermost-first) = [32, 64]
///   - swap_executor sec_rev (outermost-first) = [64, 32] → primary와 일치
#[test]
fn auf_e2e_shape_round_trip_matches_primary() {
    // primary shape (outermost-first): rows=64, cols=32
    let primary_shape_logical: Vec<usize> = vec![64, 32];
    // AUF logical shape (same as primary outermost-first)
    let auf_logical_shape: Vec<u64> = vec![64, 32];
    // payload: 64 * 32 bytes = 2048 bytes (F32 스텁, 크기만 맞춤)
    let tensor_payload: Vec<u8> = vec![0xABu8; 2048];
    let weights_payload = tensor_payload.clone();

    let tensor_index = TensorIndex {
        variant_tags: vec![tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries: vec![TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::F32.as_u32(),
            // logical order (outermost-first): 실제 auf_tool build가 채워야 하는 값
            shape: auf_logical_shape,
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![2048],
        }],
    };

    let auf_bytes = AufWriter::new(make_meta(1, 32, 64, 128), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tensor_index)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, weights_payload)
        .build()
        .unwrap();

    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
    let config = make_config(1, 32, 64, 128);

    let secondary = build_auf_secondary_from_view(
        view,
        &config,
        Path::new("/fake/e2e_test.auf"),
        BackendTag::CpuAos,
        argus_engine::models::weights::SecondaryDtypeChoice::Auto,
    )
    .expect("build_auf_secondary_from_view should succeed");

    let info = secondary
        .layer_tensor(0, "attn_q.weight")
        .expect("attn_q.weight must be in layer 0");

    // secondary_mmap.rs:593이 .rev()를 적용하므로 dims는 innermost-first.
    let expected_dims_innermost_first: Vec<u64> = vec![32, 64];
    assert_eq!(
        info.dims, expected_dims_innermost_first,
        "SecondaryTensorInfo::dims should be innermost-first (GGUF order)"
    );

    // swap_executor.rs:544 방식: sec_rev = info.dims.iter().rev()
    let sec_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
    assert_eq!(
        sec_rev, primary_shape_logical,
        "swap_executor rev(dims) must equal primary shape — shape mismatch would block swap"
    );
}

// ── 테스트 2: 빈 shape → swap_executor 방식으로 mismatch 재현 ───────────────

/// TensorEntry.shape가 비어 있으면 SecondaryTensorInfo::dims도 비어 있고
/// swap_executor의 shape 검증에서 mismatch가 발생함을 확인한다.
///
/// 이 테스트는 3차 차단의 원인을 재현하여 회귀 가드로 동작한다.
/// (현재 fix 이후에도 빈 shape를 직접 주입하면 mismatch가 발생해야 함)
#[test]
fn auf_e2e_empty_shape_causes_mismatch() {
    let tensor_payload: Vec<u8> = vec![0u8; 2048];
    let weights_payload = tensor_payload.clone();

    // 의도적으로 빈 shape 주입 (fix 이전의 auf_tool build 동작 재현)
    let tensor_index = TensorIndex {
        variant_tags: vec![tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries: vec![TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::F32.as_u32(),
            shape: vec![], // 빈 shape — 3차 차단의 원인
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![2048],
        }],
    };

    let auf_bytes = AufWriter::new(make_meta(1, 32, 64, 128), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tensor_index)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, weights_payload)
        .build()
        .unwrap();

    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
    let config = make_config(1, 32, 64, 128);

    let secondary = build_auf_secondary_from_view(
        view,
        &config,
        Path::new("/fake/e2e_empty_shape.auf"),
        BackendTag::CpuAos,
        argus_engine::models::weights::SecondaryDtypeChoice::Auto,
    )
    .expect("build should succeed even with empty shape");

    let info = secondary
        .layer_tensor(0, "attn_q.weight")
        .expect("tensor entry exists");

    // 빈 shape → dims도 비어 있음
    assert!(
        info.dims.is_empty(),
        "empty shape must propagate to empty dims"
    );

    // swap_executor 방식: sec_rev = [] ≠ primary [64, 32] → mismatch
    let sec_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
    let primary: Vec<usize> = vec![64, 32];
    assert_ne!(
        sec_rev, primary,
        "empty dims rev must not match primary shape — this confirms the mismatch condition"
    );
}

// ── 테스트 3: dtype round-trip ───────────────────────────────────────────────

/// TensorEntry.dtype → auf_dtype_to_engine → SecondaryTensorInfo::dtype
/// Q4_0, F16, F32 각각 round-trip 확인.
#[test]
fn auf_e2e_dtype_round_trip() {
    let cases: &[(TensorDType, DType, usize)] = &[
        (TensorDType::F32, DType::F32, 64),   // 16 f32 values
        (TensorDType::F16, DType::F16, 32),   // 16 f16 values
        (TensorDType::Q4_0, DType::Q4_0, 18), // 1 Q4_0 block
    ];

    for (auf_dtype, expected_engine_dtype, payload_size) in cases {
        let payload = vec![0u8; *payload_size];
        let tensor_index = TensorIndex {
            variant_tags: vec![tag_buf(TAG_WEIGHTS_CPU_AOS)],
            entries: vec![TensorEntry {
                layer_idx: 0,
                kind: TensorKind::FfnGate.as_u32(),
                dtype: auf_dtype.as_u32(),
                shape: vec![1, *payload_size as u64],
                alignment: 64,
                variant_offsets: vec![0],
                variant_sizes: vec![*payload_size as u64],
            }],
        };

        let auf_bytes = AufWriter::new(
            make_meta(1, 8, 16, *payload_size as u32),
            make_tokenizer(),
            [0u8; 32],
            0,
            0,
        )
        .with_tensor_index(tensor_index)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, payload)
        .build()
        .unwrap();

        let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
        let config = make_config(1, 8, 16, *payload_size);

        let secondary = build_auf_secondary_from_view(
            view,
            &config,
            Path::new("/fake/dtype_test.auf"),
            BackendTag::CpuAos,
            argus_engine::models::weights::SecondaryDtypeChoice::Auto,
        )
        .expect("build should succeed");

        let info = secondary
            .layer_tensor(0, "ffn_gate.weight")
            .expect("ffn_gate.weight must be present");

        assert_eq!(
            info.dtype, *expected_engine_dtype,
            "dtype mismatch for {auf_dtype:?}: expected {expected_engine_dtype:?}"
        );
    }
}

// ── 테스트 4: multi-layer shape propagation ──────────────────────────────────

/// 2-layer AUF에서 각 레이어의 shape가 독립적으로 올바르게 전파되는지 확인.
#[test]
fn auf_e2e_multi_layer_shape_propagation() {
    // layer 0: attn_q shape [64, 32]  (payload 2048 bytes)
    // layer 1: attn_q shape [128, 64] (payload 8192 bytes)
    let payload0 = vec![0xAAu8; 2048];
    let payload1 = vec![0xBBu8; 8192];
    let weights_payload = {
        let mut v = payload0.clone();
        v.extend_from_slice(&payload1);
        v
    };

    let tensor_index = TensorIndex {
        variant_tags: vec![tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries: vec![
            TensorEntry {
                layer_idx: 0,
                kind: TensorKind::AttnQ.as_u32(),
                dtype: TensorDType::F32.as_u32(),
                shape: vec![64, 32], // logical outermost-first
                alignment: 64,
                variant_offsets: vec![0],
                variant_sizes: vec![2048],
            },
            TensorEntry {
                layer_idx: 1,
                kind: TensorKind::AttnQ.as_u32(),
                dtype: TensorDType::F32.as_u32(),
                shape: vec![128, 64], // logical outermost-first
                alignment: 64,
                variant_offsets: vec![2048],
                variant_sizes: vec![8192],
            },
        ],
    };

    let auf_bytes = AufWriter::new(
        make_meta(2, 64, 128, 256),
        make_tokenizer(),
        [0u8; 32],
        0,
        0,
    )
    .with_tensor_index(tensor_index)
    .add_weights_section(TAG_WEIGHTS_CPU_AOS, weights_payload)
    .build()
    .unwrap();

    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
    let config = make_config(2, 64, 128, 256);

    let secondary = build_auf_secondary_from_view(
        view,
        &config,
        Path::new("/fake/multi_layer.auf"),
        BackendTag::CpuAos,
        argus_engine::models::weights::SecondaryDtypeChoice::Auto,
    )
    .expect("build_auf_secondary_from_view should succeed");

    // layer 0: dims должны быть [32, 64] (innermost-first)
    let info0 = secondary
        .layer_tensor(0, "attn_q.weight")
        .expect("layer 0 attn_q.weight");
    assert_eq!(
        info0.dims,
        vec![32u64, 64],
        "layer 0 dims must be innermost-first"
    );
    // swap_executor 검증: rev → [64, 32] = primary
    let sec_rev0: Vec<usize> = info0.dims.iter().rev().map(|&d| d as usize).collect();
    assert_eq!(sec_rev0, vec![64, 32]);

    // layer 1: dims должны быть [64, 128] (innermost-first)
    let info1 = secondary
        .layer_tensor(1, "attn_q.weight")
        .expect("layer 1 attn_q.weight");
    assert_eq!(
        info1.dims,
        vec![64u64, 128],
        "layer 1 dims must be innermost-first"
    );
    let sec_rev1: Vec<usize> = info1.dims.iter().rev().map(|&d| d as usize).collect();
    assert_eq!(sec_rev1, vec![128, 64]);

    // bytes integrity
    let bytes0 = secondary.tensor_bytes(info0);
    assert_eq!(bytes0.len(), 2048);
    assert_eq!(bytes0[0], 0xAAu8);

    let bytes1 = secondary.tensor_bytes(info1);
    assert_eq!(bytes1.len(), 8192);
    assert_eq!(bytes1[0], 0xBBu8);
}

// ── 테스트 5: AUF SOA bypass — split round-trip + size invariant ─────────────

/// Phase 4 LATENCY-AUF / Phase 3.7b SOA bypass: AUF `WEIGHTS_ADRENO_SOA`
/// payload (q_buf+d_buf concat)를 `SecondaryMmap::split_pre_converted_soa`로
/// 분리했을 때 builder 출력과 byte-equal해야 한다. 디바이스 측정 진입 전
/// SOA bytes 오해석으로 인한 정확성 회귀 (4차 차단의 원인)를 호스트에서
/// 사전 차단하는 회귀 가드.
#[test]
fn auf_soa_bypass_split_matches_builder_output() {
    // 4 rows × 64 cols → 8 Q4_0 blocks → AOS 144 bytes.
    let ne00 = 64usize;
    let ne01 = 4usize;
    let n_blocks = ne01 * ne00 / 32;
    let mut aos = Vec::with_capacity(n_blocks * 18);
    for b in 0..n_blocks {
        let s = (0x0100u16 + b as u16).to_le_bytes();
        aos.extend_from_slice(&s);
        for i in 0..16u8 {
            aos.push(((b * 7 + i as usize) & 0xFF) as u8);
        }
    }
    assert_eq!(aos.len(), n_blocks * 18);

    let (q_expected, d_expected) = q4_0_aos_to_adreno_soa(&aos, ne00, ne01);
    assert_eq!(q_expected.len() + d_expected.len(), aos.len());

    let mut soa_payload = Vec::with_capacity(aos.len());
    soa_payload.extend_from_slice(&q_expected);
    soa_payload.extend_from_slice(&d_expected);

    // Logical shape (outermost-first): [ne01, ne00] = [4, 64].
    let auf_logical_shape: Vec<u64> = vec![ne01 as u64, ne00 as u64];
    let payload_size = soa_payload.len() as u64;

    let tensor_index = TensorIndex {
        variant_tags: vec![tag_buf(TAG_WEIGHTS_ADRENO_SOA)],
        entries: vec![TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: auf_logical_shape,
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![payload_size],
        }],
    };

    let auf_bytes = AufWriter::new(make_meta(1, 32, 64, 128), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tensor_index)
        .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, soa_payload)
        .build()
        .expect("AUF build should succeed");

    let view = open_from_bytes(auf_bytes, BackendTag::AdrenoSoa)
        .expect("open_from_bytes with AdrenoSoa should succeed");
    let config = make_config(1, 32, 64, 128);

    let secondary = build_auf_secondary_from_view(
        view,
        &config,
        Path::new("/fake/soa_bypass.auf"),
        BackendTag::AdrenoSoa,
        argus_engine::models::weights::SecondaryDtypeChoice::Auto,
    )
    .expect("build_auf_secondary_from_view should succeed for SOA");

    assert!(
        secondary.is_pre_converted_soa(),
        "SecondaryMmap should signal pre-converted SOA for AdrenoSoa variant"
    );

    let info = secondary
        .layer_tensor(0, "attn_q.weight")
        .expect("attn_q.weight must be present");
    assert_eq!(info.dtype, DType::Q4_0);

    // SOA bypass split round-trip
    let (q_actual, d_actual) = secondary
        .split_pre_converted_soa(info)
        .expect("split_pre_converted_soa must return Some for Q4_0 SOA payload");
    assert_eq!(
        q_actual, q_expected,
        "split q_bytes must match builder q_buf"
    );
    assert_eq!(
        d_actual, d_expected,
        "split d_bytes must match builder d_buf"
    );

    // Size invariant: q + d = original AOS size (block count × 18B).
    assert_eq!(
        q_actual.len() + d_actual.len(),
        n_blocks * 18,
        "AUF SOA payload length must equal AOS Q4_0 size — placeholder cl_mem assumption"
    );
}

/// GGUF secondary는 SOA 변환을 적용하지 않으므로
/// `split_pre_converted_soa`가 항상 `None`이어야 한다 (음성 가드).
#[test]
fn split_pre_converted_soa_returns_none_for_cpu_aos_variant() {
    // CPU_AOS variant: payload는 raw AOS bytes 그대로.
    let ne00 = 32usize;
    let ne01 = 1usize;
    let n_blocks = 1;
    let mut aos = vec![0x10u8, 0x00];
    for i in 0..16u8 {
        aos.push(i);
    }
    assert_eq!(aos.len(), n_blocks * 18);

    let tensor_index = TensorIndex {
        variant_tags: vec![tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries: vec![TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![ne01 as u64, ne00 as u64],
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![18],
        }],
    };

    let auf_bytes = AufWriter::new(make_meta(1, 32, 64, 128), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tensor_index)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, aos)
        .build()
        .expect("build OK");
    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).expect("open OK");
    let secondary = build_auf_secondary_from_view(
        view,
        &make_config(1, 32, 64, 128),
        Path::new("/fake/cpu_aos.auf"),
        BackendTag::CpuAos,
        argus_engine::models::weights::SecondaryDtypeChoice::Auto,
    )
    .expect("build OK");

    assert!(!secondary.is_pre_converted_soa());
    let info = secondary.layer_tensor(0, "attn_q.weight").expect("present");
    assert!(
        secondary.split_pre_converted_soa(info).is_none(),
        "CPU_AOS variant must NOT expose SOA split (would corrupt forward path)"
    );
}
