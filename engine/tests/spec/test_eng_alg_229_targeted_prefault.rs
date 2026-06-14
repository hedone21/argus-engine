//! ENG-ALG-229 — Targeted prefault for swap target layers.
//!
//! `prefault_layers(target_layers)` must touch only the byte ranges of the
//! requested layers instead of the entire WEIGHTS section. This avoids reading
//! ~40 ms of pages for layers outside the swap batch at ratio < 1.0.
//!
//! Scenarios tested:
//! 1. Subset of layers (0, 2, 4): functional verification, no panic.
//! 2. Empty `target_layers` — no-op.
//! 3. Out-of-range indices — silently skipped (ENG-DAT-C08 spirit).
//! 4. Full layer list: functionally equivalent to `prefault()` (all layers
//!    accessible via `tensor_bytes` without panic after either call).
//!
//! Spec: ENG-ALG-229.

use argus_engine::auf::reader::open_from_bytes;
use argus_engine::auf::section::TAG_WEIGHTS_CPU_AOS;
use argus_engine::auf::tensor_index::{TensorDType, TensorEntry, TensorIndex, TensorKind};
use argus_engine::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
use argus_engine::auf::writer::AufWriter;
use argus_engine::auf::{AufMeta, BackendTag};
use argus_engine::model_config::{ModelArch, ModelConfig};
use argus_engine::models::weights::{SecondaryDtypeChoice, build_auf_secondary_from_view};

// ── Common fixture builder ───────────────────────────────────────────────────

const BYTES_PER_TENSOR: usize = 128;

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

/// Build a SecondaryMmap (AUF-backed) with `n_layers` layers.
///
/// Each layer contains two tensors (AttnQ, AttnK). The byte content of layer `i`
/// is stamped with `i as u8` throughout, so the test can verify data integrity
/// after prefault (read-only semantics).
fn build_secondary_n_layers(n_layers: usize) -> argus_engine::models::weights::SecondaryMmap {
    // Weights payload: n_layers × 2 tensors × BYTES_PER_TENSOR bytes.
    // Tensor layout: [layer0_attn_q, layer0_attn_k, layer1_attn_q, ...]
    let total = n_layers * 2 * BYTES_PER_TENSOR;
    let mut weights_payload = vec![0u8; total];
    for i in 0..n_layers {
        let base = i * 2 * BYTES_PER_TENSOR;
        for b in &mut weights_payload[base..base + 2 * BYTES_PER_TENSOR] {
            *b = i as u8;
        }
    }

    // TensorIndex: 2 entries per layer (AttnQ, AttnK), section-local offsets.
    let mut tag_buf = [0u8; 24];
    tag_buf[..TAG_WEIGHTS_CPU_AOS.len()].copy_from_slice(TAG_WEIGHTS_CPU_AOS.as_bytes());
    let mut entries = Vec::new();
    for layer_idx in 0..n_layers as u32 {
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
        n_layers: n_layers as u32,
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
        num_hidden_layers: n_layers,
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
        std::path::Path::new("/fake/eng_alg_229.auf"),
        BackendTag::CpuAos,
        SecondaryDtypeChoice::Auto,
    )
    .expect("build_auf_secondary_from_view should succeed")
}

// ── Scenario 1: subset of layers ─────────────────────────────────────────────

/// ENG-ALG-229 §S1: calling `prefault_layers` with a subset {0, 2, 4} on a
/// 6-layer secondary must not panic, and the byte data for all layers must
/// remain accessible and unmodified afterward.
#[test]
fn eng_alg_229_s1_prefault_subset_no_panic() {
    let secondary = build_secondary_n_layers(6);

    // Touch only layers 0, 2, 4 (simulates ratio ≈ 0.5 swap batch).
    secondary.prefault_layers(&[0, 2, 4]);

    // Verify data integrity for all layers — prefault is read-only.
    for layer_idx in 0..6 {
        let info = secondary
            .layer_tensor(layer_idx, "attn_q.weight")
            .unwrap_or_else(|| panic!("layer {layer_idx} attn_q must exist"));
        let tb = secondary.tensor_bytes(info);
        assert_eq!(tb.len(), BYTES_PER_TENSOR, "layer {layer_idx} attn_q size");
        assert!(
            tb.iter().all(|&b| b == layer_idx as u8),
            "layer {layer_idx} bytes must be {layer_idx} (stamp) after prefault_layers"
        );
    }
}

// ── Scenario 2: empty target — no-op ─────────────────────────────────────────

/// ENG-ALG-229 §S2: `prefault_layers(&[])` is a no-op — must not panic and
/// must leave the data untouched.
#[test]
fn eng_alg_229_s2_prefault_empty_noop() {
    let secondary = build_secondary_n_layers(4);

    // Empty target — absolute no-op.
    secondary.prefault_layers(&[]);

    // Data still reachable.
    let info = secondary
        .layer_tensor(0, "attn_q.weight")
        .expect("layer 0 attn_q must exist");
    let tb = secondary.tensor_bytes(info);
    assert_eq!(tb.len(), BYTES_PER_TENSOR);
}

// ── Scenario 3: out-of-range indices silently skipped ────────────────────────

/// ENG-ALG-229 §S3: out-of-range layer indices are silently skipped — no panic,
/// no effect on in-range layers.
#[test]
fn eng_alg_229_s3_prefault_out_of_range_silent_skip() {
    let secondary = build_secondary_n_layers(3);

    // Layer indices 99 and 1000 are out of range for a 3-layer secondary.
    secondary.prefault_layers(&[99, 1000]);

    // Also mix valid and invalid indices.
    secondary.prefault_layers(&[1, 500, 2]);

    // Data for valid layer 1 is intact.
    let info = secondary
        .layer_tensor(1, "attn_q.weight")
        .expect("layer 1 attn_q must exist");
    let tb = secondary.tensor_bytes(info);
    assert!(tb.iter().all(|&b| b == 1u8), "layer 1 bytes must be 0x01");
}

// ── Scenario 4: full layer list ≡ prefault() ─────────────────────────────────

/// ENG-ALG-229 §S4: `prefault_layers` with the full layer index must be
/// functionally equivalent to `prefault()` — all layers accessible and
/// data unchanged by either call.
///
/// We verify the functional equivalence by calling both on the same fixture
/// and asserting that `tensor_bytes` returns consistent data after each.
#[test]
fn eng_alg_229_s4_full_prefault_layers_equivalent_to_prefault() {
    const N: usize = 5;
    let secondary = build_secondary_n_layers(N);

    // Call the targeted variant with all layers.
    let all_layers: Vec<usize> = (0..N).collect();
    secondary.prefault_layers(&all_layers);

    // Also call the broad prefault().
    secondary.prefault();

    // Verify all tensors accessible and data intact after both calls.
    for layer_idx in 0..N {
        for subname in &["attn_q.weight", "attn_k.weight"] {
            let info = secondary
                .layer_tensor(layer_idx, subname)
                .unwrap_or_else(|| panic!("layer {layer_idx} {subname} must exist"));
            let tb = secondary.tensor_bytes(info);
            assert_eq!(tb.len(), BYTES_PER_TENSOR);
            assert!(
                tb.iter().all(|&b| b == layer_idx as u8),
                "layer {layer_idx} {subname} bytes must be {layer_idx}"
            );
        }
    }
}
