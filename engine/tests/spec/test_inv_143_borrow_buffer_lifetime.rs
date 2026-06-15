//! INV-143 — BorrowedMmapBuffer mmap lifetime invariant.
//!
//! 대응 spec: `spec/41-invariants.md` §3.19 (INV-143)
//! 대응 alg : `spec/32-engine-algorithms.md` §3.12.19.2 (ENG-ALG-227)
//! 대응 arch: `arch/weight_swap.md` §7.3
//! Backlog : WSWAP-6-B
//!
//! ## 불변식 요약
//!
//! `BorrowedMmapBuffer`는 자신이 생존하는 동안 secondary `Arc<SecondaryMmap>`의
//! clone을 보관해야 한다.  이로써 mmap 슬라이스 포인터가 backend
//! `copy_weight_from`/`copy_from` 호출 사이클을 통과하는 동안 secondary mmap이
//! drop되어 SIGBUS를 유발하지 않음을 보증한다.
//!
//! ## 검증 항목
//!
//! - [S1] `BorrowedMmapBuffer` 생성 직후 secondary Arc strong_count >= 2.
//! - [S2] `BorrowedMmapBuffer` drop 후 secondary Arc strong_count 1 감소.
//! - [S3] `as_ptr()`, `size()`, `dtype()` getter 정확성.
//! - [S4] AOS path materialise 결과 Tensor가 CPU backend에서 byte-equal.
//!
//! ## 구현 메모
//!
//! S4는 real AUF secondary 없이 CPU backend `copy_from`만으로 검증한다.
//! `copy_from`은 `src.as_ptr()` → `copy_nonoverlapping`이므로 borrow buffer의
//! `as_ptr()` 정확성이 byte-equal을 보장한다.

use std::sync::Arc;

use argus_engine::auf::{
    AufMeta, BackendTag,
    section::TAG_WEIGHTS_CPU_AOS,
    tensor_index::TensorIndex,
    tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE},
    writer::AufWriter,
};
use argus_engine::backend::Backend;
use argus_engine::buffer::{Buffer, DType};
use argus_engine::memory::host::mmap::MmapBuffer;
use argus_engine::model_config::{ModelArch, ModelConfig};
use argus_engine::models::weights::SecondaryMmap;
use argus_engine::models::weights::secondary_mmap::{
    SecondaryDtypeChoice, build_auf_secondary_from_view,
};
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;

// ── Fixture helpers ──────────────────────────────────────────────────────────

fn make_auf_meta(n_layers: u32) -> AufMeta {
    AufMeta {
        architecture: "llama".to_owned(),
        n_layers,
        n_heads_q: 2,
        n_kv_heads: 1,
        head_dim: 8,
        hidden_dim: 16,
        ffn_dim: 32,
        vocab_size: 4,
        max_seq_len: 64,
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

fn make_model_config(n_layers: usize) -> ModelConfig {
    let m = make_auf_meta(n_layers as u32);
    ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: m.hidden_dim as usize,
        num_hidden_layers: m.n_layers as usize,
        num_attention_heads: m.n_heads_q as usize,
        num_key_value_heads: m.n_kv_heads as usize,
        head_dim: m.head_dim as usize,
        intermediate_size: m.ffn_dim as usize,
        vocab_size: m.vocab_size as usize,
        rms_norm_eps: m.rms_norm_epsilon,
        rope_theta: m.rope_theta,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_id: 2,
        weight_prefix: String::new(),
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
    }
}

/// Build a 24-byte NUL-padded tag buffer.
fn tag_buf(s: &str) -> [u8; 24] {
    let mut buf = [0u8; 24];
    let b = s.as_bytes();
    buf[..b.len().min(24)].copy_from_slice(&b[..b.len().min(24)]);
    buf
}

/// Build a minimal AUF byte payload with an empty TensorIndex (CPU_AOS variant
/// tag) and an empty CPU_AOS weights section. Sufficient for
/// `build_auf_secondary_from_view` to succeed on a model with 0 tensor entries.
fn build_auf_bytes(n_layers: u32) -> Vec<u8> {
    let tensor_index = TensorIndex {
        variant_tags: vec![tag_buf(TAG_WEIGHTS_CPU_AOS)],
        entries: vec![],
    };
    AufWriter::new(make_auf_meta(n_layers), make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tensor_index)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![])
        .build()
        .unwrap()
}

/// Build an `Arc<SecondaryMmap>` backed by an in-memory AUF payload.
fn make_secondary(n_layers: usize) -> Arc<SecondaryMmap> {
    let auf_bytes = build_auf_bytes(n_layers as u32);
    let view = argus_engine::auf::reader::open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
    let config = make_model_config(n_layers);
    let secondary = build_auf_secondary_from_view(
        view,
        &config,
        std::path::Path::new("test.auf"),
        BackendTag::CpuAos,
        SecondaryDtypeChoice::Auto,
    )
    .unwrap();
    Arc::new(secondary)
}

// ── S1: Arc strong_count >= 2 while BorrowedMmapBuffer is alive ──────────────

#[test]
fn s1_arc_refcount_incremented_while_buffer_alive() {
    let data = [0xAAu8, 0xBBu8, 0xCCu8, 0xDDu8];
    let secondary = make_secondary(1);

    // Before creating the borrow buffer, strong_count should be 1 (only `secondary`).
    let count_before = Arc::strong_count(&secondary);
    assert_eq!(count_before, 1, "baseline strong_count should be 1");

    let buf = MmapBuffer::borrow(&data, DType::F16, secondary.clone());

    // After construction, BorrowedMmapBuffer clones the Arc → count >= 2.
    let count_during = Arc::strong_count(&secondary);
    assert!(
        count_during >= 2,
        "strong_count must be >= 2 while BorrowedMmapBuffer is alive (got {count_during})"
    );

    // Keep buf alive for the assertion above.
    drop(buf);
}

// ── S2: Arc strong_count decreases after BorrowedMmapBuffer is dropped ───────

#[test]
fn s2_arc_refcount_decremented_on_drop() {
    let data = [1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8];
    let secondary = make_secondary(1);

    let count_before = Arc::strong_count(&secondary);

    let buf = MmapBuffer::borrow(&data, DType::F32, secondary.clone());
    let count_during = Arc::strong_count(&secondary);
    assert!(
        count_during > count_before,
        "count should increase after construction"
    );

    drop(buf);

    let count_after = Arc::strong_count(&secondary);
    assert_eq!(
        count_after, count_before,
        "strong_count must return to {count_before} after drop (got {count_after})"
    );
}

// ── S3: as_ptr / size / dtype getter correctness ─────────────────────────────

#[test]
fn s3_getter_accuracy() {
    let data: Vec<u8> = (0u8..16).collect();
    let secondary = make_secondary(1);

    let buf = MmapBuffer::borrow(&data, DType::Q4_0, secondary.clone());

    assert_eq!(
        buf.as_ptr(),
        data.as_ptr(),
        "as_ptr() must equal data.as_ptr()"
    );
    assert_eq!(buf.size(), data.len(), "size() must equal data.len()");
    assert_eq!(
        buf.dtype(),
        DType::Q4_0,
        "dtype() must equal the stored dtype"
    );
    // MmapBuffer (post-unification) returns the underlying ptr from `as_mut_ptr()`
    // matching the original safetensors/GGUF loader contract — read-only via
    // mmap pages; mutation would segfault. The previous `BorrowedMmapBuffer`
    // returned null_mut(); the new unified contract trusts callers (backend
    // `copy_weight_from`/`copy_from`) to use only `as_ptr()`.
    assert_eq!(
        buf.as_mut_ptr() as *const u8,
        data.as_ptr(),
        "as_mut_ptr() must point to the same address as as_ptr() (matches MmapBuffer contract)"
    );
    assert!(buf.cl_mem().is_none(), "cl_mem() must be None");
    assert!(buf.sync_device().is_ok(), "sync_device() must be Ok(())");
    assert!(
        buf.is_host_managed(),
        "is_host_managed() should default to true"
    );
    assert!(
        !buf.is_gpu_buffer(),
        "is_gpu_buffer() should default to false"
    );
}

// ── S4: CPU backend copy_from preserves bytes (AOS borrow path correctness) ──

#[test]
fn s4_cpu_copy_from_borrow_buffer_byte_equal() {
    use argus_engine::backend::cpu::CpuBackend;

    // 32 bytes of distinguishable content.
    let data: Vec<u8> = (0u8..32)
        .map(|x| x.wrapping_mul(7).wrapping_add(3))
        .collect();
    let secondary = make_secondary(1);

    let borrow_buf: Arc<dyn Buffer> =
        Arc::new(MmapBuffer::borrow(&data, DType::F32, secondary.clone()));

    let cpu_backend = Arc::new(CpuBackend::new()) as Arc<dyn Backend>;
    let shape = Shape::new(vec![8, 1]); // 8 × f32 = 32 bytes
    let src_tensor = Tensor::new(shape, borrow_buf, cpu_backend.clone());

    let dst_tensor = cpu_backend
        .copy_from(&src_tensor)
        .expect("copy_from must succeed");

    assert_eq!(
        dst_tensor.size(),
        data.len(),
        "destination tensor size mismatch"
    );

    let dst_bytes = unsafe { std::slice::from_raw_parts(dst_tensor.as_ptr(), dst_tensor.size()) };
    assert_eq!(
        dst_bytes,
        data.as_slice(),
        "CPU copy_from must produce byte-equal output for BorrowedMmapBuffer"
    );

    assert!(
        Arc::strong_count(&secondary) >= 1,
        "secondary Arc must remain alive after copy"
    );
}

// ── Additional: Arc is the last reference — secondary drops with buffer ───────

#[test]
fn arc_is_sole_keeper_after_outer_drop() {
    let data = [0xFFu8; 4];
    let secondary = make_secondary(1);

    let buf: Arc<dyn Buffer> = Arc::new(MmapBuffer::borrow(&data, DType::U8, secondary.clone()));

    let weak = Arc::downgrade(&secondary);
    drop(secondary);

    assert!(
        weak.upgrade().is_some(),
        "secondary mmap must still be alive while BorrowedMmapBuffer holds the Arc clone"
    );

    drop(buf);
    assert!(
        weak.upgrade().is_none(),
        "secondary mmap must be dropped once BorrowedMmapBuffer is dropped"
    );
}
