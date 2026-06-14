//! INV-003: config.json의 architectures가 지원 목록에 없으면 로딩 거부.
//!
//! 원본: 00-overview SYS-032
//! 검증: 지원되지 않는 architecture 문자열이 포함된 config.json으로
//!       ModelConfig::from_json() 호출 시 Err 반환 확인.
//!       지원되는 architecture (Llama, Qwen2, Gemma3) 는 성공 확인.

use std::io::Write;

use argus_engine::model_config::{ModelArch, ModelConfig};

/// 임시 디렉토리에 config.json을 작성하고 ModelConfig::from_json()을 호출하는 헬퍼.
fn parse_config_json(json: &str) -> anyhow::Result<ModelConfig> {
    let tmp_dir = std::env::temp_dir().join(format!(
        "argus_engine_test_inv003_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let config_path = tmp_dir.join("config.json");
    let mut f = std::fs::File::create(&config_path).unwrap();
    f.write_all(json.as_bytes()).unwrap();
    let result = ModelConfig::from_json(&tmp_dir);
    let _ = std::fs::remove_dir_all(&tmp_dir);
    result
}

/// Minimal valid config JSON (Llama architecture) for reuse.
fn llama_config_json() -> &'static str {
    r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 5632,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    }"#
}

// ═══════════════════════════════════════════════════════════════
// INV-003: Unsupported architecture => rejected
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_003_unsupported_architecture_rejected() {
    let json = r#"{
        "architectures": ["GPT2ForCausalLM"],
        "model_type": "gpt2",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": 50257
    }"#;
    let result = parse_config_json(json);
    assert!(
        result.is_err(),
        "INV-003: unsupported architecture must be rejected"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Unsupported"),
        "Error message should mention 'Unsupported', got: {}",
        err_msg
    );
}

#[test]
fn test_inv_003_unknown_architecture_string_rejected() {
    let json = r#"{
        "architectures": ["MistralForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 32000
    }"#;
    let result = parse_config_json(json);
    assert!(
        result.is_err(),
        "INV-003: MistralForCausalLM is not in the supported list"
    );
}

#[test]
fn test_inv_003_empty_architectures_no_model_type_rejected() {
    let json = r#"{
        "architectures": [],
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 5632,
        "vocab_size": 32000
    }"#;
    let result = parse_config_json(json);
    assert!(
        result.is_err(),
        "INV-003: empty architectures with no model_type fallback must be rejected"
    );
}

#[test]
fn test_inv_003_no_architectures_no_model_type_rejected() {
    let json = r#"{
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 5632,
        "vocab_size": 32000
    }"#;
    let result = parse_config_json(json);
    assert!(
        result.is_err(),
        "INV-003: missing architectures and model_type must be rejected"
    );
}

#[test]
fn test_inv_003_multiple_unknown_architectures_rejected() {
    let json = r#"{
        "architectures": ["GPTNeoXForCausalLM", "FalconForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 5632,
        "vocab_size": 32000
    }"#;
    let result = parse_config_json(json);
    assert!(
        result.is_err(),
        "INV-003: multiple unsupported architectures must be rejected"
    );
}

// ═══════════════════════════════════════════════════════════════
// INV-003 positive: supported architectures accepted
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_003_llama_accepted() {
    let result = parse_config_json(llama_config_json());
    assert!(result.is_ok(), "LlamaForCausalLM must be accepted");
    assert_eq!(result.unwrap().arch, ModelArch::Llama);
}

#[test]
fn test_inv_003_qwen2_accepted() {
    let json = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "intermediate_size": 8960,
        "vocab_size": 151936
    }"#;
    let result = parse_config_json(json);
    assert!(result.is_ok(), "Qwen2ForCausalLM must be accepted");
    assert_eq!(result.unwrap().arch, ModelArch::Qwen2);
}

#[test]
fn test_inv_003_gemma3_accepted() {
    let json = r#"{
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text",
        "hidden_size": 1152,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 6912,
        "vocab_size": 262144
    }"#;
    let result = parse_config_json(json);
    assert!(result.is_ok(), "Gemma3ForCausalLM must be accepted");
    assert_eq!(result.unwrap().arch, ModelArch::Gemma3);
}

#[test]
fn test_inv_003_model_type_fallback_llama() {
    // No architectures field, but model_type "llama" should work as fallback
    let json = r#"{
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 5632,
        "vocab_size": 32000
    }"#;
    let result = parse_config_json(json);
    assert!(
        result.is_ok(),
        "model_type fallback to 'llama' must be accepted"
    );
    assert_eq!(result.unwrap().arch, ModelArch::Llama);
}

#[test]
fn test_inv_003_unknown_model_type_rejected() {
    let json = r#"{
        "model_type": "phi3",
        "hidden_size": 3072,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "intermediate_size": 8192,
        "vocab_size": 32064
    }"#;
    let result = parse_config_json(json);
    assert!(
        result.is_err(),
        "INV-003: unsupported model_type must be rejected"
    );
}
