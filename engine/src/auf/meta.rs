/// AUF META section — 모델 architecture JSON payload (ENG-DAT-096.4).
///
/// META section은 JSON-in-binary 형식이다. 파싱 결과는 `AufMeta` 구조체.
use crate::auf::error::{AufError, AufResult};
use serde::{Deserialize, Serialize};

/// META section JSON 구조.
///
/// 직렬화 형식: UTF-8 JSON (`serde_json`). size 필드로 길이 명시.
///
/// **unknown-key 정책**: `#[serde(deny_unknown_fields)]`를 **설정하지 않는다**.
/// v0.1.x reader가 v0.2+ AUF의 META JSON을 읽을 때 `default_dtype` 같은 신규 필드를
/// silently ignore하여 호환성을 보존한다 (INV-139).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AufMeta {
    /// 모델 아키텍처 (예: "llama", "qwen2").
    pub architecture: String,
    /// decoder layer 수.
    pub n_layers: u32,
    /// Q head 수.
    pub n_heads_q: u32,
    /// KV head 수.
    pub n_kv_heads: u32,
    /// head당 차원.
    pub head_dim: u32,
    /// hidden dim.
    pub hidden_dim: u32,
    /// FFN intermediate dim.
    pub ffn_dim: u32,
    /// vocab 크기.
    pub vocab_size: u32,
    /// 최대 시퀀스 길이.
    pub max_seq_len: u32,
    /// RoPE theta.
    pub rope_theta: f64,
    /// RoPE rotary_dim (rotary 적용 차원 수). 0이면 head_dim 전체.
    pub rotary_dim: u32,
    /// RoPE scaling factor (1.0 = 없음).
    pub rope_scaling: f64,
    /// RMSNorm epsilon.
    pub rms_norm_epsilon: f64,
    /// AUF v0.2 multi-dtype 기본 dtype (INV-138, ENG-DAT-098).
    ///
    /// `capability_optional` bit 3(MULTI_DTYPE_VARIANTS) = 1이면 의무 필드.
    /// bit 3 = 0이면 `None`으로 직렬화를 생략한다.
    ///
    /// 값은 `TensorDType` enum 이름 중 하나:
    /// `"F32"` / `"F16"` / `"BF16"` / `"Q4_0"` / `"Q4_1"` / `"Q8_0"` / `"U8"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_dtype: Option<String>,
}

impl AufMeta {
    /// JSON 바이트열에서 파싱한다.
    pub fn from_json_bytes(bytes: &[u8]) -> AufResult<Self> {
        serde_json::from_slice(bytes).map_err(|e| AufError::Other(format!("META JSON parse: {e}")))
    }

    /// JSON 바이트열로 직렬화한다.
    pub fn to_json_bytes(&self) -> AufResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| AufError::Other(format!("META JSON serialize: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_meta() -> AufMeta {
        AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 16,
            n_heads_q: 32,
            n_kv_heads: 8,
            head_dim: 64,
            hidden_dim: 2048,
            ffn_dim: 8192,
            vocab_size: 128256,
            max_seq_len: 2048,
            rope_theta: 500000.0,
            rotary_dim: 64,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        }
    }

    #[test]
    fn round_trip_json() {
        let meta = example_meta();
        let bytes = meta.to_json_bytes().unwrap();
        let meta2 = AufMeta::from_json_bytes(&bytes).unwrap();
        assert_eq!(meta2.architecture, "llama");
        assert_eq!(meta2.n_layers, 16);
        assert_eq!(meta2.n_heads_q, 32);
        assert_eq!(meta2.n_kv_heads, 8);
        assert_eq!(meta2.head_dim, 64);
        assert_eq!(meta2.vocab_size, 128256);
        assert!((meta2.rope_theta - 500000.0).abs() < 1.0);
    }

    #[test]
    fn invalid_json_returns_err() {
        let bad = b"not json";
        let err = AufMeta::from_json_bytes(bad).unwrap_err();
        assert!(matches!(err, AufError::Other(_)));
    }

    /// B-1: unknown key silently ignore 검증 (INV-139 호환성 보증).
    ///
    /// v0.1.x reader가 v0.2 META JSON(default_dtype 등 신규 필드 포함)을 읽을 때
    /// `#[serde(deny_unknown_fields)]`가 없으면 unknown 필드를 무시하고 파싱 성공해야 한다.
    #[test]
    fn unknown_key_silently_ignored() {
        // v0.2 META JSON에 default_dtype과 추가 미지 필드가 포함된 경우
        let json = r#"{
            "architecture": "llama",
            "n_layers": 16,
            "n_heads_q": 32,
            "n_kv_heads": 8,
            "head_dim": 64,
            "hidden_dim": 2048,
            "ffn_dim": 8192,
            "vocab_size": 128256,
            "max_seq_len": 2048,
            "rope_theta": 500000.0,
            "rotary_dim": 64,
            "rope_scaling": 1.0,
            "rms_norm_epsilon": 1e-5,
            "default_dtype": "Q4_0",
            "unknown_future_field": 42,
            "another_unknown": true
        }"#;
        // v0.1.x reader가 알지 못하는 필드가 포함되어도 파싱 성공해야 한다.
        let meta = AufMeta::from_json_bytes(json.as_bytes()).unwrap();
        assert_eq!(meta.architecture, "llama");
        assert_eq!(meta.n_layers, 16);
        // default_dtype은 v0.2 필드이므로 v0.1.x 구조체에 없으면 무시,
        // v0.2 구조체에 있으면 Some("Q4_0")으로 파싱된다.
        // 이 테스트는 파싱 자체가 성공하는지만 검증한다.
    }

    /// B-3: default_dtype 필드 round-trip.
    #[test]
    fn default_dtype_round_trip() {
        let meta = AufMeta {
            default_dtype: Some("Q4_0".to_owned()),
            ..example_meta()
        };
        let bytes = meta.to_json_bytes().unwrap();
        let json_str = std::str::from_utf8(&bytes).unwrap();
        assert!(
            json_str.contains("default_dtype"),
            "default_dtype must be serialized when Some: {json_str}"
        );
        let meta2 = AufMeta::from_json_bytes(&bytes).unwrap();
        assert_eq!(meta2.default_dtype, Some("Q4_0".to_owned()));
    }

    /// B-3: default_dtype = None이면 JSON에서 생략된다 (skip_serializing_if).
    #[test]
    fn default_dtype_none_omitted_from_json() {
        let meta = example_meta(); // default_dtype = None
        let bytes = meta.to_json_bytes().unwrap();
        let json_str = std::str::from_utf8(&bytes).unwrap();
        assert!(
            !json_str.contains("default_dtype"),
            "default_dtype must be omitted when None: {json_str}"
        );
    }

    /// B-3: 지원 dtype 문자열 모두 round-trip.
    #[test]
    fn default_dtype_all_valid_values() {
        for dtype_str in &["F32", "F16", "BF16", "Q4_0", "Q4_1", "Q8_0", "U8"] {
            let meta = AufMeta {
                default_dtype: Some((*dtype_str).to_owned()),
                ..example_meta()
            };
            let bytes = meta.to_json_bytes().unwrap();
            let meta2 = AufMeta::from_json_bytes(&bytes).unwrap();
            assert_eq!(
                meta2.default_dtype.as_deref(),
                Some(*dtype_str),
                "dtype_str={dtype_str}"
            );
        }
    }
}
