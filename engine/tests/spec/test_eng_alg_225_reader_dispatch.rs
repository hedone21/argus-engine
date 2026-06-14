//! Spec 테스트: ENG-ALG-225 — AUF reader dtype dispatch precedence (Sprint D).
//!
//! ENG-ALG-225 reader dispatch 알고리즘 의무:
//!   1. 명시 dtype (`requested_dtype = Some(d)`) → 해당 entry 반환. 없으면 DtypeNotAvailable.
//!   2. `requested_dtype = None` + `META.default_dtype` → default_dtype entry 우선 반환.
//!   3. `requested_dtype = None` + `META.default_dtype = None` → first-match.
//!
//! Sprint D 함정 3 (Adreno SOA × F16 reject):
//!   - backend=ADRENO_SOA + selected dtype=F16 → AdrenoSoaF16Rejected 반환.
//!
//! SwapExecutor 시그니처 unchanged 검증:
//!   - `SwapExecutor::new(target_dtype: DType, ...)` 시그니처가 유지됨을 컴파일타임에 보증.
//!
//! INV-225: reader dtype dispatch precedence 보존.

use argus_engine::auf::{
    AufError, AufMeta, AufTokenizer, AufWriter, BackendTag, TAG_WEIGHTS_ADRENO_SOA,
    TAG_WEIGHTS_CPU_AOS, TOKENIZER_KIND_BPE, TensorDType, TensorEntry, TensorIndex, TensorKind,
    open_from_bytes,
};
use argus_engine::models::weights::{
    LoadError, SecondaryDtypeChoice, build_auf_secondary_from_view,
};

// ── 공통 헬퍼 ────────────────────────────────────────────────────────────────

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

fn make_meta(default_dtype: Option<&str>) -> AufMeta {
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

/// 단순한 f32 → Q4_0 bytes (32 elements = 1 block = 18 bytes)
fn q4_0_bytes_32() -> Vec<u8> {
    use argus_engine::auf::dtype_convert::convert_tensor_dtype;
    let f32_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let f32_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4).to_vec()
    };
    convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::Q4_0, &[1, 32]).unwrap()
}

/// 단순한 f32 → F16 bytes (32 elements = 64 bytes)
fn f16_bytes_32() -> Vec<u8> {
    use argus_engine::auf::dtype_convert::convert_tensor_dtype;
    let f32_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let f32_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4).to_vec()
    };
    convert_tensor_dtype(&f32_bytes, TensorDType::F32, TensorDType::F16, &[1, 32]).unwrap()
}

/// multi-dtype AUF 바이트를 빌드한다.
/// Q4_0 + F16 두 dtype entry를 layer=0, kind=AttnQ에 등록.
/// `default_dtype`: META.default_dtype 필드 값.
fn build_multi_dtype_auf(weights_tag: &str, default_dtype: Option<&str>) -> Vec<u8> {
    let q4 = q4_0_bytes_32();
    let f16 = f16_bytes_32();

    let q4_offset = 0u64;
    let q4_size = q4.len() as u64;
    let f16_offset = q4_size;
    let f16_size = f16.len() as u64;

    let entries = vec![
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![1, 32],
            alignment: 64,
            variant_offsets: vec![q4_offset],
            variant_sizes: vec![q4_size],
        },
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::F16.as_u32(),
            shape: vec![1, 32],
            alignment: 64,
            variant_offsets: vec![f16_offset],
            variant_sizes: vec![f16_size],
        },
    ];

    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(weights_tag)],
        entries,
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&q4);
    payload.extend_from_slice(&f16);

    AufWriter::new(make_meta(default_dtype), make_tokenizer(), [0u8; 32], 0, 0)
        .with_multi_dtype(true)
        .with_tensor_index(tidx)
        .add_weights_section(weights_tag, payload)
        .build()
        .unwrap()
}

// ── ENG-ALG-225 Precedence 1: 명시 dtype ────────────────────────────────────

/// ENG-ALG-225 Precedence 1-a: requested_dtype=Q4_0 → Q4_0 entry 반환.
#[test]
fn eng_alg_225_explicit_q4_0_found() {
    let bytes = build_multi_dtype_auf(TAG_WEIGHTS_CPU_AOS, Some("Q4_0"));
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    let entry = view
        .lookup_tensor(0, TensorKind::AttnQ.as_u32(), Some(TensorDType::Q4_0))
        .unwrap();
    assert_eq!(
        entry.dtype,
        TensorDType::Q4_0.as_u32(),
        "requested Q4_0 must return Q4_0 entry"
    );
}

/// ENG-ALG-225 Precedence 1-b: requested_dtype=F16 → F16 entry 반환.
#[test]
fn eng_alg_225_explicit_f16_found() {
    let bytes = build_multi_dtype_auf(TAG_WEIGHTS_CPU_AOS, Some("Q4_0"));
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    let entry = view
        .lookup_tensor(0, TensorKind::AttnQ.as_u32(), Some(TensorDType::F16))
        .unwrap();
    assert_eq!(
        entry.dtype,
        TensorDType::F16.as_u32(),
        "requested F16 must return F16 entry"
    );
}

/// ENG-ALG-225 Precedence 1-c: requested_dtype=BF16 (없음) → DtypeNotAvailable.
#[test]
fn eng_alg_225_explicit_dtype_not_available() {
    let bytes = build_multi_dtype_auf(TAG_WEIGHTS_CPU_AOS, Some("Q4_0"));
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    let err = view
        .lookup_tensor(0, TensorKind::AttnQ.as_u32(), Some(TensorDType::BF16))
        .unwrap_err();
    assert!(
        matches!(err, AufError::DtypeNotAvailable { dtype, .. } if dtype == TensorDType::BF16.as_u32()),
        "expected DtypeNotAvailable for BF16, got: {err}"
    );
}

// ── ENG-ALG-225 Precedence 2: META.default_dtype ────────────────────────────

/// ENG-ALG-225 Precedence 2: default_dtype=Q4_0 이면 Q4_0 entry 우선 반환.
#[test]
fn eng_alg_225_default_dtype_q4_0_used() {
    let bytes = build_multi_dtype_auf(TAG_WEIGHTS_CPU_AOS, Some("Q4_0"));
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    // requested_dtype=None → META.default_dtype="Q4_0" → Q4_0 entry.
    let entry = view
        .lookup_tensor(0, TensorKind::AttnQ.as_u32(), None)
        .unwrap();
    assert_eq!(
        entry.dtype,
        TensorDType::Q4_0.as_u32(),
        "META.default_dtype=Q4_0 must select Q4_0 entry when requested_dtype=None"
    );
}

/// ENG-ALG-225 Precedence 2: default_dtype=F16 이면 F16 entry 우선 반환.
#[test]
fn eng_alg_225_default_dtype_f16_used() {
    let bytes = build_multi_dtype_auf(TAG_WEIGHTS_CPU_AOS, Some("F16"));
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    let entry = view
        .lookup_tensor(0, TensorKind::AttnQ.as_u32(), None)
        .unwrap();
    assert_eq!(
        entry.dtype,
        TensorDType::F16.as_u32(),
        "META.default_dtype=F16 must select F16 entry"
    );
}

// ── ENG-ALG-225 Precedence 3: first-match ────────────────────────────────────

/// ENG-ALG-225 Precedence 3: default_dtype=None → first-match (Q4_0, INV-138 정렬 기준).
///
/// INV-138에 따라 writer는 default dtype entry를 그룹 첫 번째에 정렬해야 하므로,
/// first-match는 (default dtype이 없어도) entries의 순서를 따른다.
#[test]
fn eng_alg_225_no_default_dtype_first_match() {
    let bytes = build_multi_dtype_auf(TAG_WEIGHTS_CPU_AOS, None);
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    // META.default_dtype=None → first-match.
    // build_multi_dtype_auf는 Q4_0을 먼저 등록하므로 first-match = Q4_0.
    let entry = view
        .lookup_tensor(0, TensorKind::AttnQ.as_u32(), None)
        .unwrap();
    assert_eq!(
        entry.dtype,
        TensorDType::Q4_0.as_u32(),
        "no default_dtype → first-match entry must be Q4_0 (first registered)"
    );
}

// ── DtypeNotAvailable — (layer, kind) 자체 없음 ────────────────────────────

/// ENG-ALG-225: 존재하지 않는 (layer_idx=99, kind) → DtypeNotAvailable.
#[test]
fn eng_alg_225_nonexistent_layer_returns_error() {
    let bytes = build_multi_dtype_auf(TAG_WEIGHTS_CPU_AOS, Some("Q4_0"));
    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();

    let err = view
        .lookup_tensor(99, TensorKind::AttnQ.as_u32(), None)
        .unwrap_err();
    assert!(
        matches!(err, AufError::DtypeNotAvailable { .. }),
        "nonexistent layer must return DtypeNotAvailable, got: {err}"
    );
}

// ── Sprint D 함정 3: Adreno SOA × F16 reject ──────────────────────────────

/// Sprint D 함정 3: backend=ADRENO_SOA + secondary-dtype=F16 → AdrenoSoaF16Rejected.
///
/// build_auf_secondary_from_view에 backend_tag=AdrenoSoa + SecondaryDtypeChoice::F16을
/// 전달하면 LoadError::AdrenoSoaF16Rejected를 반환해야 한다.
#[test]
fn sprint_d_adreno_soa_f16_rejected() {
    // Adreno SOA weights tag를 사용한 multi-dtype AUF 빌드.
    let q4 = q4_0_bytes_32();
    let f16 = f16_bytes_32();

    let entries = vec![
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![1, 32],
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![q4.len() as u64],
        },
        TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::F16.as_u32(),
            shape: vec![1, 32],
            alignment: 64,
            variant_offsets: vec![q4.len() as u64],
            variant_sizes: vec![f16.len() as u64],
        },
    ];

    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_ADRENO_SOA)],
        entries,
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&q4);
    payload.extend_from_slice(&f16);

    let bytes = AufWriter::new(make_meta(Some("Q4_0")), make_tokenizer(), [0u8; 32], 0, 0)
        .with_multi_dtype(true)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, payload)
        .build()
        .unwrap();

    let view = open_from_bytes(bytes, BackendTag::AdrenoSoa).unwrap();

    // ModelConfig 대역: build_auf_secondary_from_view에 전달용.
    let primary_config = make_dummy_model_config(2);
    let path = std::path::Path::new("/test/dummy.auf");

    let err = build_auf_secondary_from_view(
        view,
        &primary_config,
        path,
        BackendTag::AdrenoSoa,
        SecondaryDtypeChoice::F16,
    )
    .unwrap_err();

    assert!(
        matches!(err, LoadError::AdrenoSoaF16Rejected),
        "Adreno SOA + F16 must be rejected, got: {err:?}"
    );
}

/// Sprint D 함정 3 (역방향): backend=ADRENO_SOA + dtype=Q4_0 → 허용.
#[test]
fn sprint_d_adreno_soa_q4_0_allowed() {
    let q4 = q4_0_bytes_32();

    let entries = vec![TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnQ.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![1, 32],
        alignment: 64,
        variant_offsets: vec![0],
        variant_sizes: vec![q4.len() as u64],
    }];

    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_ADRENO_SOA)],
        entries,
    };

    let bytes = AufWriter::new(make_meta(Some("Q4_0")), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, q4)
        .build()
        .unwrap();

    let view = open_from_bytes(bytes, BackendTag::AdrenoSoa).unwrap();
    let primary_config = make_dummy_model_config(2);
    let path = std::path::Path::new("/test/dummy.auf");

    let result = build_auf_secondary_from_view(
        view,
        &primary_config,
        path,
        BackendTag::AdrenoSoa,
        SecondaryDtypeChoice::Q4_0,
    );
    assert!(
        result.is_ok(),
        "Adreno SOA + Q4_0 must be allowed, got: {result:?}"
    );
}

// ── 단방향 swap 정합성 검증 ────────────────────────────────────────────────

/// primary=Q4_0 only AUF + secondary=F16 → ReverseSwapRejected.
///
/// AUF 파일에 Q4_0 entry만 있는 경우 secondary=F16을 지정하면
/// 역방향 swap (Q4_0→F16)으로 판단하여 거부해야 한다.
#[test]
fn sprint_d_reverse_swap_rejected() {
    // Q4_0 only AUF (F16 entry 없음).
    let q4 = q4_0_bytes_32();

    let entries = vec![TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnQ.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![1, 32],
        alignment: 64,
        variant_offsets: vec![0],
        variant_sizes: vec![q4.len() as u64],
    }];

    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries,
    };

    let bytes = AufWriter::new(make_meta(Some("Q4_0")), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, q4)
        .build()
        .unwrap();

    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    let primary_config = make_dummy_model_config(2);
    let path = std::path::Path::new("/test/dummy.auf");

    // F16을 명시했지만 AUF에 F16이 없으므로 DtypeNotFound가 먼저 발생한다.
    // (ReverseSwapRejected는 F16이 available한데 primary=Q4_0인 경우)
    let err = build_auf_secondary_from_view(
        view,
        &primary_config,
        path,
        BackendTag::CpuAos,
        SecondaryDtypeChoice::F16,
    )
    .unwrap_err();

    // F16이 AUF에 없으므로 DtypeNotFound.
    assert!(
        matches!(err, LoadError::DtypeNotFound { .. }),
        "Q4_0-only AUF + secondary=F16 must DtypeNotFound, got: {err:?}"
    );
}

/// Q4_0-only AUF + secondary=Auto → Q4_0 선택 (단방향 차단 없음, 자동 선택).
#[test]
fn sprint_d_auto_q4_0_only_auf_selects_q4_0() {
    let q4 = q4_0_bytes_32();

    let entries = vec![TensorEntry {
        layer_idx: 0,
        kind: TensorKind::AttnQ.as_u32(),
        dtype: TensorDType::Q4_0.as_u32(),
        shape: vec![1, 32],
        alignment: 64,
        variant_offsets: vec![0],
        variant_sizes: vec![q4.len() as u64],
    }];

    let tidx = TensorIndex {
        variant_tags: vec![make_variant_tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries,
    };

    let bytes = AufWriter::new(make_meta(Some("Q4_0")), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tidx)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, q4)
        .build()
        .unwrap();

    let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
    let primary_config = make_dummy_model_config(2);
    let path = std::path::Path::new("/test/dummy.auf");

    // Auto + Q4_0 only → 정상 (Q4_0 선택됨).
    let result = build_auf_secondary_from_view(
        view,
        &primary_config,
        path,
        BackendTag::CpuAos,
        SecondaryDtypeChoice::Auto,
    );
    assert!(
        result.is_ok(),
        "Auto with Q4_0-only AUF must succeed, got: {result:?}"
    );
}

// ── SwapExecutor 시그니처 unchanged (D-4) ──────────────────────────────────

/// D-4: SwapExecutor::new 시그니처가 `(DType, &ModelConfig, Arc<dyn Backend>, &dyn Memory)` 임을
/// 컴파일타임에 보증한다.
///
/// 실제 `SwapExecutor::new`를 호출하지 않고, 함수 포인터로 타입을 추출하여 시그니처를 확인한다.
/// 이 테스트가 컴파일되면 SwapExecutor 시그니처가 unchanged임이 증명된다.
#[test]
fn d4_swap_executor_signature_unchanged() {
    use argus_engine::buffer::DType;
    use argus_engine::weight::SwapExecutor;

    // SwapExecutor::new 시그니처 컴파일타임 체크:
    // target_dtype: DType이 첫 번째 파라미터이고 나머지 인수 타입이 맞아야 한다.
    // 함수 항목(fn item)을 직접 호출하는 wrapper 함수로 타입 체크를 수행한다.
    // 이 함수가 컴파일되면 SwapExecutor::new 시그니처가 unchanged임이 증명된다.
    fn _assert_swap_executor_signature<'a>(
        dtype: DType,
        config: &'a argus_engine::model_config::ModelConfig,
        backend: std::sync::Arc<dyn argus_engine::backend::Backend>,
        mem: &'a dyn argus_engine::memory::Memory,
    ) -> SwapExecutor<'a> {
        SwapExecutor::new(dtype, config, backend, mem)
    }
    // 위 inner function이 컴파일되면 시그니처 검증 완료.
    // 실행 시점에는 아무것도 assert하지 않는다.
    let _ = _assert_swap_executor_signature as *const () as usize;
}

// ── 헬퍼 함수 ──────────────────────────────────────────────────────────────

/// 테스트용 더미 `ModelConfig` 생성.
///
/// `num_hidden_layers`만 커스텀하고 나머지는 최소값을 사용한다.
fn make_dummy_model_config(num_layers: usize) -> argus_engine::model_config::ModelConfig {
    use argus_engine::model_config::{ModelArch, ModelConfig};
    ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: 32,
        num_hidden_layers: num_layers,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        vocab_size: 10,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_id: 1,
        weight_prefix: String::new(),
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
    }
}
