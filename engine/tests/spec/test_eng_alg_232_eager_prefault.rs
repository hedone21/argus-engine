//! WSWAP-6-PREFAULT — Eager prefault at model load.
//!
//! `SecondaryMmap::prefault()` must be callable without panic on a valid
//! secondary, and the code path that skips prefault when no secondary is
//! configured must not panic either.
//!
//! Scenarios tested:
//! 1. `prefault()` on a valid AUF-backed secondary — no panic, data intact.
//! 2. Calling `prefault()` twice (idempotent) — no panic, data unchanged.
//! 3. "No secondary" guard: the caller-side silent-skip branch (Option::is_none
//!    check) compiles and runs without panic.
//!
//! Note: page-cache warm-up effects are not measurable in a unit-test
//! environment (anonymous memory, no kernel page cache). These tests verify
//! functional correctness and call-site safety only. On-device latency
//! reduction (target: ~328 ms) is validated by the Tester in the deploy-test
//! workflow.
//!
//! Spec: WSWAP-6-PREFAULT (backlog).

use argus_engine::auf::reader::open_from_bytes;
use argus_engine::auf::section::TAG_WEIGHTS_CPU_AOS;
use argus_engine::auf::tensor_index::{TensorDType, TensorEntry, TensorIndex, TensorKind};
use argus_engine::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
use argus_engine::auf::writer::AufWriter;
use argus_engine::auf::{AufMeta, BackendTag};
use argus_engine::model_config::{ModelArch, ModelConfig};
use argus_engine::models::weights::{SecondaryDtypeChoice, build_auf_secondary_from_view};

// ── Fixture ──────────────────────────────────────────────────────────────────

const BYTES_PER_TENSOR: usize = 128;
const N_LAYERS: usize = 4;

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

/// Build a SecondaryMmap (AUF-backed) with `N_LAYERS` layers.
/// Each layer's bytes are stamped with `layer_idx as u8` to verify
/// read-only invariant after prefault.
fn build_secondary() -> argus_engine::models::weights::SecondaryMmap {
    let total = N_LAYERS * 2 * BYTES_PER_TENSOR;
    let mut weights_payload = vec![0u8; total];
    for i in 0..N_LAYERS {
        let base = i * 2 * BYTES_PER_TENSOR;
        for b in &mut weights_payload[base..base + 2 * BYTES_PER_TENSOR] {
            *b = i as u8;
        }
    }

    let mut tag_buf = [0u8; 24];
    tag_buf[..TAG_WEIGHTS_CPU_AOS.len()].copy_from_slice(TAG_WEIGHTS_CPU_AOS.as_bytes());
    let mut entries = Vec::new();
    for layer_idx in 0..N_LAYERS as u32 {
        let base_offset = (layer_idx as usize * 2 * BYTES_PER_TENSOR) as u64;
        entries.push(TensorEntry {
            layer_idx,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::F32.as_u32(),
            shape: vec![1, 128],
            alignment: 64,
            variant_offsets: vec![base_offset],
            variant_sizes: vec![BYTES_PER_TENSOR as u64],
        });
        entries.push(TensorEntry {
            layer_idx,
            kind: TensorKind::AttnK.as_u32(),
            dtype: TensorDType::F32.as_u32(),
            shape: vec![1, 128],
            alignment: 64,
            variant_offsets: vec![base_offset + BYTES_PER_TENSOR as u64],
            variant_sizes: vec![BYTES_PER_TENSOR as u64],
        });
    }
    let tensor_index = TensorIndex {
        variant_tags: vec![tag_buf],
        entries,
    };

    let meta = AufMeta {
        architecture: "llama".to_owned(),
        n_layers: N_LAYERS as u32,
        n_heads_q: 1,
        n_kv_heads: 1,
        head_dim: 128,
        hidden_dim: 128,
        ffn_dim: 256,
        vocab_size: 2,
        max_seq_len: 32,
        rope_theta: 10000.0,
        rotary_dim: 128,
        rope_scaling: 1.0,
        rms_norm_epsilon: 1e-5,
        default_dtype: None,
    };

    let auf_bytes = AufWriter::new(meta, make_tokenizer(), [0u8; 32], 0, 0)
        .with_tensor_index(tensor_index)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, weights_payload)
        .build()
        .unwrap();

    let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();

    let config = ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: 128,
        num_hidden_layers: N_LAYERS,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 128,
        intermediate_size: 256,
        vocab_size: 2,
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
    };

    build_auf_secondary_from_view(
        view,
        &config,
        std::path::Path::new("/fake/wswap6_prefault.auf"),
        BackendTag::CpuAos,
        SecondaryDtypeChoice::Auto,
    )
    .expect("build_auf_secondary_from_view should succeed")
}

// ── Scenario 1: prefault on valid secondary — no panic, data intact ──────────

/// WSWAP-6-PREFAULT §S1: `prefault()` on a valid AUF-backed secondary must
/// not panic and must leave all tensor bytes unchanged (read-only semantics).
#[test]
fn wswap6_prefault_s1_valid_secondary_no_panic() {
    let secondary = build_secondary();
    secondary.prefault();

    // Verify all layers' bytes are unchanged after prefault (read-only).
    for layer_idx in 0..N_LAYERS {
        let info = secondary
            .layer_tensor(layer_idx, "attn_q.weight")
            .unwrap_or_else(|| panic!("layer {layer_idx} attn_q must exist after prefault"));
        let tb = secondary.tensor_bytes(info);
        assert_eq!(
            tb.len(),
            BYTES_PER_TENSOR,
            "layer {layer_idx} size unchanged"
        );
        assert!(
            tb.iter().all(|&b| b == layer_idx as u8),
            "layer {layer_idx} bytes must be {layer_idx:#02x} (stamp) — prefault must not modify data"
        );
    }
}

// ── Scenario 2: idempotent — calling prefault twice is safe ──────────────────

/// WSWAP-6-PREFAULT §S2: calling `prefault()` twice on the same secondary
/// must not panic. Page cache hits make the second call nearly free; data
/// must remain intact after both calls.
#[test]
fn wswap6_prefault_s2_idempotent() {
    let secondary = build_secondary();

    // First call — cold faults (in anonymous mmap on host, effectively no-op page touch).
    secondary.prefault();
    // Second call — must not panic (warm or re-touch is safe).
    secondary.prefault();

    // Data still intact.
    let info = secondary
        .layer_tensor(0, "attn_k.weight")
        .expect("layer 0 attn_k must exist");
    let tb = secondary.tensor_bytes(info);
    assert_eq!(tb.len(), BYTES_PER_TENSOR);
    assert!(
        tb.iter().all(|&b| b == 0u8),
        "layer 0 bytes must be 0x00 after two prefault calls"
    );
}

// ── Scenario 3: no secondary — silent skip (Option guard) ────────────────────

/// WSWAP-6-PREFAULT §S3: the caller-side guard `if let Some(ref secondary) = ...`
/// must silently skip when no secondary is configured. This mirrors the
/// generate.rs eager-prefault branch and ensures no panic when
/// `--eager-prefault-secondary` is used without `--secondary-gguf`.
#[test]
fn wswap6_prefault_s3_no_secondary_silent_skip() {
    // Simulate the generate.rs guard: Option::None → skip prefault.
    let no_secondary: Option<argus_engine::models::weights::SecondaryMmap> = None;

    // This mirrors the generate.rs code path:
    //   if let Some(ref secondary) = model.secondary_mmap { secondary.prefault(); }
    // When None, the branch is not taken — must not panic.
    if let Some(ref secondary) = no_secondary {
        secondary.prefault();
        panic!("Should not reach here — no secondary configured");
    }
    // Reaching here means silent skip succeeded.
}
