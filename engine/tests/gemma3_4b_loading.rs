//! Gemma 3 4B multimodal wrapper loading smoke test.
//!
//! Exercises the original 2-shard safetensors + original wrapper config.json
//! (Gemma3ForConditionalGeneration) path introduced in Task 2–6.
//!
//! Skips silently if the 4B model directory is unavailable.

use std::path::PathBuf;

const MODEL_DIR_ENV: &str = "ARGUS_GEMMA3_4B_DIR";
const DEFAULT_DIR: &str = "models/gemma3-4b";

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var(MODEL_DIR_ENV).unwrap_or_else(|_| DEFAULT_DIR.to_string());
    let p = PathBuf::from(&dir);
    if p.join("config.json").exists() {
        Some(p)
    } else {
        None
    }
}

/// Test 1: multimodal wrapper config is correctly flattened.
///
/// With config.json = Gemma3ForConditionalGeneration (original wrapper),
/// `ModelConfig::from_json` must:
///  - detect the multimodal layout and extract text_config fields
///  - set arch = Gemma3
///  - set hidden_size = 2560, num_hidden_layers = 34, etc.
///  - set weight_prefix = "language_model."
#[test]
fn gemma3_4b_config_flattens_and_sets_prefix() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: 4B model dir not found at {DEFAULT_DIR}");
        return;
    };

    let cfg = argus_engine::model_config::ModelConfig::from_json(&dir)
        .expect("ModelConfig::from_json failed");

    assert_eq!(
        cfg.arch,
        argus_engine::model_config::ModelArch::Gemma3,
        "Expected arch Gemma3, got {:?}",
        cfg.arch
    );
    assert_eq!(cfg.hidden_size, 2560, "hidden_size mismatch");
    assert_eq!(cfg.num_hidden_layers, 34, "num_hidden_layers mismatch");
    assert_eq!(cfg.num_attention_heads, 8, "num_attention_heads mismatch");
    assert_eq!(cfg.num_key_value_heads, 4, "num_key_value_heads mismatch");
    assert_eq!(cfg.head_dim, 256, "head_dim mismatch");

    if cfg.weight_prefix.is_empty() {
        // If this fires, the flattened workaround config.json is still active
        // (Step 1 of Task 7 was not completed).
        panic!(
            "weight_prefix is empty — original wrapper config.json was not restored.\n\
             Ensure config.json.orig was copied over config.json before running this test."
        );
    }
    assert_eq!(
        cfg.weight_prefix, "language_model.",
        "weight_prefix mismatch: got {:?}",
        cfg.weight_prefix
    );
}

/// Test 2: `SafetensorsSource::open` resolves through the 2-shard index and
/// the prefix-aware mapper maps layer-0 Wq to the correct tensor name.
///
/// With model.safetensors renamed/removed and model.safetensors.index.json
/// present, `open` must fall through to the shard-index path.  The mapper
/// must prepend "language_model." so the resolved name ends with
/// "language_model.model.layers.0.self_attn.q_proj.weight".
#[test]
fn gemma3_4b_safetensors_resolves_layer0_qproj() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: 4B model dir not found at {DEFAULT_DIR}");
        return;
    };

    // Verify that model.safetensors does NOT exist (workaround file was moved aside).
    let single = dir.join("model.safetensors");
    if single.exists() {
        panic!(
            "model.safetensors still exists at {:?}.\n\
             Move it aside so the shard-index path is exercised:\n\
             mv model.safetensors model.safetensors.workaround.bak",
            single
        );
    }
    // Verify the 2-shard index is present.
    assert!(
        dir.join("model.safetensors.index.json").exists(),
        "model.safetensors.index.json not found — cannot test shard loading"
    );

    let src = argus_engine::models::loader::safetensors::SafetensorsSource::open(
        dir.to_str().expect("non-UTF8 path"),
        argus_engine::buffer::DType::F16,
    )
    .expect("SafetensorsSource::open failed");

    let name = src.resolve_name(&argus_engine::models::loader::TensorId::LayerWeight {
        layer: 0,
        kind: argus_engine::models::loader::LayerWeightKind::Wq,
    });

    assert!(
        name.ends_with("model.layers.0.self_attn.q_proj.weight"),
        "resolve_name returned unexpected name: {name}"
    );
    assert!(
        name.starts_with("language_model."),
        "weight_prefix 'language_model.' not prepended — got: {name}"
    );
}
