use crate::rope::RopeFreqScaling;
use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

/// Discriminator for model architecture dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Qwen2,
    Gemma3,
}

/// Unified model configuration parsed from HuggingFace config.json.
/// All architecture-specific derivations are resolved at construction time.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: ModelArch,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Per-dimension frequency rescaling from `config.json`'s `rope_scaling`
    /// ([`RopeFreqScaling::NONE`] when the model declares none). Every Llama 3.1/3.2 checkpoint
    /// sets this; dropping it silently runs a different model — see [`crate::rope`].
    pub rope_freq_scaling: RopeFreqScaling,
    pub has_qkv_bias: bool,
    pub tie_word_embeddings: bool,
    /// Every id that terminates generation, in the order the config listed them. NEVER empty —
    /// an absent config value yields `[u32::MAX]` (an id no tokenizer emits), matching the old
    /// scalar fallback.
    ///
    /// A set, not a scalar, because the Instruct variants of Llama 3.1/3.2 ship three:
    /// `<|end_of_text|>`, `<|eom_id|>` (end of a tool/multi-step message) and `<|eot_id|>` (end of
    /// turn — the one that actually ends an assistant reply). Generation must stop on ANY of them.
    /// Test [`ModelConfig::is_eos`] rather than comparing against [`ModelConfig::primary_eos`].
    pub eos_token_ids: Vec<u32>,
    /// Safetensors tensor name prefix (e.g., "language_model." for Gemma3 multimodal wrappers).
    /// Empty string for standard flat layouts (Llama, Qwen2, Gemma3 1B).
    pub weight_prefix: String,

    // Gemma 3 specific fields (None for Llama/Qwen2)
    pub rope_local_theta: Option<f64>,
    pub sliding_window: Option<usize>,
    pub sliding_window_pattern: Option<usize>,
    pub query_pre_attn_scalar: Option<usize>,
    pub embed_scale: Option<f32>,
}

/// `eos_token_id` as HuggingFace writes it: a scalar for base checkpoints, an ARRAY for the
/// Instruct variants of Llama 3.1/3.2. Accepting only the scalar made every Instruct checkpoint
/// unloadable — `invalid type: sequence, expected u32`, thrown while parsing `config.json`, so the
/// model never reached the loader at all.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum RawEosTokenIds {
    One(u32),
    Many(Vec<u32>),
}

/// `rope_scaling` as HuggingFace writes it. `rope_type` was spelled `type` before transformers
/// 4.44, and checkpoints with the old key are still in circulation, so both are accepted.
#[derive(Deserialize, Clone)]
struct RawRopeScaling {
    rope_type: Option<String>,
    #[serde(rename = "type")]
    legacy_type: Option<String>,
    factor: Option<f64>,
    low_freq_factor: Option<f64>,
    high_freq_factor: Option<f64>,
    original_max_position_embeddings: Option<usize>,
}

/// Raw HuggingFace config.json — supports Llama, Qwen2, and Gemma3 via Option fields.
/// 필수처럼 보이는 숫자 필드도 Option으로 선언하여 multimodal wrapper JSON
/// (최상위에 hidden_size 없이 text_config로 감싸는 구조)도 파싱 가능하게 한다.
#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct RawHfConfig {
    architectures: Option<Vec<String>>,
    model_type: Option<String>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    intermediate_size: Option<usize>,
    vocab_size: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_scaling: Option<RawRopeScaling>,
    tie_word_embeddings: Option<bool>,
    eos_token_id: Option<RawEosTokenIds>,
    // Gemma 3 specific
    rope_local_base_freq: Option<f64>,
    sliding_window: Option<usize>,
    sliding_window_pattern: Option<usize>,
    query_pre_attn_scalar: Option<usize>,
    hidden_activation: Option<String>,
    /// Multimodal wrapper용 — text 전용 서브 config. Gemma3 4B의 "text_config"에 해당.
    text_config: Option<Box<RawHfConfig>>,
}

impl ModelConfig {
    /// Parse config.json and auto-detect architecture.
    pub fn from_json(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let file = File::open(&config_path)
            .map_err(|e| anyhow!("Cannot open {}: {}", config_path.display(), e))?;
        let raw: RawHfConfig = serde_json::from_reader(file)?;

        // Multimodal wrapper 감지: text_config가 존재하거나 architectures가 *ForConditionalGeneration이면,
        // text_config를 top-level로 flatten하고 weight prefix를 설정한다.
        let is_multimodal = raw.architectures.as_ref().is_some_and(|archs| {
            archs
                .iter()
                .any(|a| a.ends_with("ForConditionalGeneration"))
        }) || raw.text_config.is_some();

        let (mut raw, weight_prefix): (RawHfConfig, String) = if is_multimodal {
            let arch_hint = raw
                .architectures
                .as_ref()
                .and_then(|a| a.first())
                .cloned()
                .unwrap_or_else(|| "<unknown>".to_string());
            let tc = *raw.text_config.clone().ok_or_else(|| {
                anyhow!("config.json has architecture '{arch_hint}' (multimodal wrapper) but 'text_config' field is missing")
            })?;
            let mut flat = tc;
            if flat.text_config.is_some() {
                return Err(anyhow!(
                    "nested text_config in multimodal wrapper is not supported (single-wrapper only)"
                ));
            }
            // 현재 지원 범위는 Gemma3 multimodal wrapper 한정. 향후 Llava 등 다른 wrapper를 지원하려면
            // architectures 주입을 감지된 wrapper 패밀리별로 분기해야 함.
            if flat.architectures.is_none() {
                flat.architectures = Some(vec!["Gemma3ForCausalLM".to_string()]);
            }
            flat.text_config = None;
            (flat, "language_model.".to_string())
        } else {
            (raw, String::new())
        };

        let arch = Self::detect_arch(&raw)?;

        // Gemma3TextConfig defaults applied when a multimodal wrapper omits attention shape
        // fields. Mirrors HuggingFace transformers Gemma3TextConfig __init__ defaults — these
        // values match gemma-3-4b. Other Gemma3 sizes include these fields explicitly in
        // their text_config, so defaults are only consulted for 4B-style wrappers.
        // Scoped to Gemma3 so future Llava/Qwen2-VL wrappers don't accidentally inherit them.
        if !weight_prefix.is_empty() && matches!(arch, ModelArch::Gemma3) {
            raw.num_attention_heads.get_or_insert(8);
            raw.num_key_value_heads.get_or_insert(4);
            raw.head_dim.get_or_insert(256);
            raw.vocab_size.get_or_insert(262208);
        }

        let hidden_size = raw
            .hidden_size
            .ok_or_else(|| anyhow!("config.json: missing 'hidden_size'"))?;
        let num_hidden_layers = raw
            .num_hidden_layers
            .ok_or_else(|| anyhow!("config.json: missing 'num_hidden_layers'"))?;
        let num_attention_heads = raw
            .num_attention_heads
            .ok_or_else(|| anyhow!("config.json: missing 'num_attention_heads'"))?;
        let num_key_value_heads = raw
            .num_key_value_heads
            .ok_or_else(|| anyhow!("config.json: missing 'num_key_value_heads'"))?;
        let intermediate_size = raw
            .intermediate_size
            .ok_or_else(|| anyhow!("config.json: missing 'intermediate_size'"))?;
        let vocab_size = raw
            .vocab_size
            .ok_or_else(|| anyhow!("config.json: missing 'vocab_size'"))?;

        let head_dim = raw.head_dim.unwrap_or(hidden_size / num_attention_heads);

        let has_qkv_bias = match arch {
            ModelArch::Qwen2 => true,
            ModelArch::Llama | ModelArch::Gemma3 => false,
        };

        // Gemma 3 specific fields
        let (
            rope_local_theta,
            sliding_window,
            sliding_window_pattern,
            query_pre_attn_scalar,
            embed_scale,
        ) = match arch {
            ModelArch::Gemma3 => (
                Some(raw.rope_local_base_freq.unwrap_or(10000.0)),
                raw.sliding_window,
                raw.sliding_window_pattern,
                raw.query_pre_attn_scalar,
                Some((hidden_size as f32).sqrt()),
            ),
            _ => (None, None, None, None, None),
        };

        Ok(Self {
            arch,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-5),
            rope_theta: raw.rope_theta.unwrap_or(10000.0),
            rope_freq_scaling: Self::parse_rope_scaling(raw.rope_scaling.as_ref()),
            has_qkv_bias,
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
            eos_token_ids: match raw.eos_token_id {
                Some(RawEosTokenIds::One(v)) => vec![v],
                // An empty array carries no more information than an absent key, so it takes the
                // same fallback rather than breaking the non-empty invariant.
                Some(RawEosTokenIds::Many(v)) if !v.is_empty() => v,
                _ => vec![u32::MAX],
            },
            rope_local_theta,
            sliding_window,
            sliding_window_pattern,
            query_pre_attn_scalar,
            embed_scale,
            weight_prefix,
        })
    }

    /// `rope_scaling` -> [`RopeFreqScaling`].
    ///
    /// Only `rope_type: "llama3"` is implemented. Any OTHER type is reported LOUDLY rather than
    /// ignored in silence: an unapplied rescaling does not fail, it quietly changes what the model
    /// computes, which is the hardest class of bug to notice from the outside.
    fn parse_rope_scaling(raw: Option<&RawRopeScaling>) -> RopeFreqScaling {
        let Some(rs) = raw else {
            return RopeFreqScaling::NONE;
        };
        let kind = rs
            .rope_type
            .as_deref()
            .or(rs.legacy_type.as_deref())
            .unwrap_or("default");
        match kind {
            "llama3" => RopeFreqScaling {
                factor: rs.factor.unwrap_or(1.0) as f32,
                low_freq_factor: rs.low_freq_factor.unwrap_or(1.0) as f32,
                high_freq_factor: rs.high_freq_factor.unwrap_or(4.0) as f32,
                original_max_position_embeddings: rs
                    .original_max_position_embeddings
                    .unwrap_or(8192) as f32,
            },
            "default" => RopeFreqScaling::NONE,
            other => {
                eprintln!(
                    "[rope] config.json declares rope_scaling type '{other}', which this engine \
                     does not implement — position encoding will NOT match the reference \
                     implementation for this model. Only 'llama3' is supported."
                );
                RopeFreqScaling::NONE
            }
        }
    }

    /// Does `id` end generation? This is the check every stop condition should use — the config may
    /// name several terminators and only one of them typically ends a chat turn.
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_token_ids.contains(&id)
    }

    /// A single representative id, for the few interfaces that cannot carry a set (AUF metadata,
    /// the `--eos-token-id` default).
    ///
    /// **Lossy on purpose.** HuggingFace lists the ids in ascending numeric order, not priority
    /// order, so for Llama 3.1 Instruct this returns `<|end_of_text|>` and NOT the `<|eot_id|>` that
    /// actually terminates an assistant turn. Anything that stops generation must use
    /// [`Self::is_eos`]; reach for this only where a set genuinely does not fit.
    pub fn primary_eos(&self) -> u32 {
        self.eos_token_ids.first().copied().unwrap_or(u32::MAX)
    }

    fn detect_arch(raw: &RawHfConfig) -> Result<ModelArch> {
        // Try architectures field first
        if let Some(archs) = &raw.architectures {
            for a in archs {
                match a.as_str() {
                    "LlamaForCausalLM" => return Ok(ModelArch::Llama),
                    "Qwen2ForCausalLM" => return Ok(ModelArch::Qwen2),
                    "Gemma3ForCausalLM" => return Ok(ModelArch::Gemma3),
                    _ => {}
                }
            }
        }
        // Fallback to model_type
        if let Some(mt) = &raw.model_type {
            match mt.as_str() {
                "llama" => return Ok(ModelArch::Llama),
                "qwen2" => return Ok(ModelArch::Qwen2),
                "gemma3_text" | "gemma3" => return Ok(ModelArch::Gemma3),
                _ => {}
            }
        }
        Err(anyhow!(
            "Unsupported model architecture: {:?} / {:?}",
            raw.architectures,
            raw.model_type
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_parse_llama_config() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/llama3.2-1b");
        if !dir.exists() {
            eprintln!("Skipping: model dir not found at {}", dir.display());
            return;
        }
        let config = ModelConfig::from_json(&dir).unwrap();
        assert_eq!(config.arch, ModelArch::Llama);
        assert!(!config.has_qkv_bias);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.primary_eos(), 128001);
    }

    #[test]
    fn test_parse_gemma3_config() {
        let json = r#"{
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "hidden_size": 1152,
            "num_hidden_layers": 26,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 6912,
            "vocab_size": 262144,
            "rms_norm_eps": 0.000001,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "sliding_window": 512,
            "sliding_window_pattern": 6,
            "query_pre_attn_scalar": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "tie_word_embeddings": true,
            "eos_token_id": 1
        }"#;

        // Write to a temp file in /tmp
        let tmp_dir = std::path::PathBuf::from("/tmp/argus_engine_test_gemma3_config");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let config_path = tmp_dir.join("config.json");
        let mut f = std::fs::File::create(&config_path).unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let config = ModelConfig::from_json(&tmp_dir).unwrap();
        assert_eq!(config.arch, ModelArch::Gemma3);
        assert!(!config.has_qkv_bias);
        assert_eq!(config.hidden_size, 1152);
        assert_eq!(config.num_hidden_layers, 26);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.intermediate_size, 6912);
        assert_eq!(config.vocab_size, 262144);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-10);
        assert!((config.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.primary_eos(), 1);

        // Gemma3 specific fields
        let local_theta = config
            .rope_local_theta
            .expect("rope_local_theta should be set");
        assert!((local_theta - 10000.0).abs() < 1.0);
        assert_eq!(config.sliding_window, Some(512));
        assert_eq!(config.sliding_window_pattern, Some(6));
        assert_eq!(config.query_pre_attn_scalar, Some(256));

        let embed_scale = config.embed_scale.expect("embed_scale should be set");
        let expected_scale = (1152_f32).sqrt();
        assert!(
            (embed_scale - expected_scale).abs() < 1e-3,
            "embed_scale={} expected={}",
            embed_scale,
            expected_scale
        );

        // Llama/Qwen2 fields should be None for Gemma3 — verify non-Gemma fields still work
        // (embed_scale is only Some for Gemma3)
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_parse_gemma3_multimodal_config_flattens_text_config() {
        let json = r#"{
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "text_config": {
                "hidden_size": 2560,
                "num_hidden_layers": 34,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "intermediate_size": 10240,
                "vocab_size": 262144,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "sliding_window": 1024,
                "sliding_window_pattern": 6,
                "query_pre_attn_scalar": 256,
                "model_type": "gemma3_text",
                "eos_token_id": 1
            },
            "vision_config": { "hidden_size": 1152 }
        }"#;
        let tmp_dir = std::path::PathBuf::from("/tmp/argus_engine_test_gemma3_4b_config");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let config_path = tmp_dir.join("config.json");
        let mut f = std::fs::File::create(&config_path).unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let config = ModelConfig::from_json(&tmp_dir).unwrap();
        assert_eq!(config.arch, ModelArch::Gemma3);
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 34);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.intermediate_size, 10240);
        assert_eq!(config.weight_prefix, "language_model.");
        assert_eq!(config.sliding_window, Some(1024));
        assert_eq!(config.query_pre_attn_scalar, Some(256));
    }

    #[test]
    fn test_parse_gemma3_1b_has_empty_weight_prefix() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/gemma3-1b");
        if !dir.exists() {
            eprintln!("Skipping: model dir not found at {}", dir.display());
            return;
        }
        let config = ModelConfig::from_json(&dir).unwrap();
        assert_eq!(config.weight_prefix, "");
    }

    #[test]
    fn test_parse_llama_has_empty_weight_prefix() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/llama3.2-1b");
        if !dir.exists() {
            eprintln!("Skipping: model dir not found at {}", dir.display());
            return;
        }
        let config = ModelConfig::from_json(&dir).unwrap();
        assert_eq!(config.weight_prefix, "");
    }

    #[test]
    fn test_parse_multimodal_nested_text_config_errors() {
        let json = r#"{
            "architectures": ["Gemma3ForConditionalGeneration"],
            "text_config": {
                "architectures": ["Gemma3ForCausalLM"],
                "hidden_size": 2560,
                "num_hidden_layers": 34,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "intermediate_size": 10240,
                "vocab_size": 262144,
                "text_config": {"hidden_size": 1}
            }
        }"#;
        let tmp_dir = std::path::PathBuf::from("/tmp/argus_engine_test_nested_text_config");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let config_path = tmp_dir.join("config.json");
        let mut f = std::fs::File::create(&config_path).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        let err = ModelConfig::from_json(&tmp_dir).unwrap_err();
        assert!(
            err.to_string().contains("nested text_config"),
            "expected nested text_config error, got: {err}"
        );
    }

    #[test]
    fn test_parse_multimodal_missing_text_config_errors_with_arch_name() {
        let json = r#"{
            "architectures": ["Gemma3ForConditionalGeneration"]
        }"#;
        let tmp_dir = std::path::PathBuf::from("/tmp/argus_engine_test_missing_text_config");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let config_path = tmp_dir.join("config.json");
        let mut f = std::fs::File::create(&config_path).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        let err = ModelConfig::from_json(&tmp_dir).unwrap_err();
        let s = err.to_string();
        assert!(
            s.contains("Gemma3ForConditionalGeneration"),
            "expected arch name in error, got: {err}"
        );
        assert!(
            s.contains("text_config"),
            "expected text_config mention, got: {err}"
        );
    }

    #[test]
    fn test_parse_qwen2_config() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/qwen2.5-1.5b");
        if !dir.exists() {
            eprintln!("Skipping: model dir not found at {}", dir.display());
            return;
        }
        let config = ModelConfig::from_json(&dir).unwrap();
        assert_eq!(config.arch, ModelArch::Qwen2);
        assert!(config.has_qkv_bias);
        // head_dim derived: 1536 / 12 = 128
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_key_value_heads, 2);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.primary_eos(), 151643);
    }

    /// `eos_token_id` arrives as a scalar OR an array, and the array form is what every Llama 3.1/3.2
    /// Instruct checkpoint ships. Rejecting it failed at `serde_json::from_reader`, i.e. the model
    /// was unloadable — not degraded, unusable.
    #[test]
    fn eos_token_id_parses_as_either_a_scalar_or_an_array() {
        use std::io::Write;
        let write = |name: &str, eos: &str| {
            let dir = std::env::temp_dir().join(format!("argus_eos_{name}"));
            std::fs::create_dir_all(&dir).unwrap();
            let json = format!(
                r#"{{"architectures":["LlamaForCausalLM"],"hidden_size":16,
                   "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                   "intermediate_size":32,"vocab_size":64,"eos_token_id":{eos}}}"#
            );
            std::fs::File::create(dir.join("config.json"))
                .unwrap()
                .write_all(json.as_bytes())
                .unwrap();
            ModelConfig::from_json(&dir).unwrap()
        };

        assert_eq!(write("scalar", "128001").eos_token_ids, vec![128001]);

        // Llama 3.1 Instruct: <|end_of_text|>, <|eom_id|>, <|eot_id|>.
        let many = write("array", "[128001, 128008, 128009]");
        assert_eq!(many.eos_token_ids, vec![128001, 128008, 128009]);
        // `<|eot_id|>` ends an assistant turn but is NOT the first entry — the whole reason a stop
        // condition has to test membership instead of comparing against `primary_eos`.
        assert!(many.is_eos(128009));
        assert_ne!(many.primary_eos(), 128009);

        // Absent and empty both fall back to the sentinel, keeping the non-empty invariant.
        assert_eq!(write("empty", "[]").eos_token_ids, vec![u32::MAX]);
    }
}
