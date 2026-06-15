//! OpenAI-compatible request/response types + message rendering for the chat
//! server (`/v1/chat/completions`).
//!
//! The OpenAI chat API is **stateless**: each request carries the full
//! conversation in `messages[]`. [`render_messages`] replays that array through
//! the model's [`ChatTemplate`] into a single prompt string (closed system/user/
//! assistant turns, then an open assistant header for generation). The server
//! then resets the [`ChatSession`](super::session::ChatSession), prefills the
//! prompt, and generates.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::inference::sampling::SamplingConfig;
use crate::session::chat_template::ChatTemplate;

// ─── Request ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct OaiMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

/// OpenAI `stop` may be a single string or an array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopField::One(s) => vec![s],
            StopField::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<OaiMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stop: Option<StopField>,
}

impl OpenAiChatRequest {
    pub fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

// ─── Response (non-streaming) ────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: RespMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct RespMessage {
    pub role: &'static str, // "assistant"
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ─── Response (streaming chunk) ──────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

pub fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Render the full OpenAI `messages[]` array into a single prompt string: each
/// message as a closed turn, then an open assistant header for generation.
/// Robust to the final role (always opens an assistant turn).
pub fn render_messages(tpl: &ChatTemplate, messages: &[OaiMessage]) -> Result<String> {
    if messages.is_empty() {
        return Err(anyhow!("messages[] must not be empty"));
    }
    let mut out = String::new();
    for m in messages {
        match m.role.as_str() {
            "system" => out.push_str(&tpl.render_system(&m.content)),
            "user" => out.push_str(&tpl.render_user(&m.content)),
            "assistant" => out.push_str(&tpl.render_assistant(&m.content)),
            other => {
                return Err(anyhow!(
                    "unsupported message role '{other}' (expected system|user|assistant)"
                ));
            }
        }
    }
    out.push_str(tpl.assistant_header());
    Ok(out)
}

/// Build the effective [`SamplingConfig`] for a request: start from the server's
/// base config (CLI defaults) and apply per-request `temperature` / `top_p`.
pub fn effective_sampling(base: &SamplingConfig, req: &OpenAiChatRequest) -> SamplingConfig {
    let mut cfg = base.clone();
    if let Some(t) = req.temperature {
        cfg.temperature = t.max(0.0);
    }
    if let Some(p) = req.top_p {
        cfg.top_p = p.clamp(0.0, 1.0);
    }
    cfg
}

/// Extra stop token ids from the request's `stop` strings. Only single-token stop
/// strings are honored (the decode-loop stop check matches one sampled token);
/// multi-token stop sequences are skipped (documented v1 limitation). Template
/// EOT/EOS stops are added separately via `build_chat_stop_ids`.
pub fn extra_stop_ids(req: &OpenAiChatRequest, tk: &Tokenizer) -> Vec<u32> {
    let Some(stop) = req.stop.clone() else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    for s in stop.into_vec() {
        if s.is_empty() {
            continue;
        }
        if let Ok(enc) = tk.encode(s.as_str(), false) {
            let toks = enc.get_ids();
            if toks.len() == 1 {
                ids.push(toks[0]);
            }
            // multi-token stop strings are not matchable token-wise — skipped.
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::ModelArch;

    fn msg(role: &str, content: &str) -> OaiMessage {
        OaiMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn render_messages_llama_full_conversation() {
        let t = ChatTemplate::new(ModelArch::Llama).unwrap();
        let msgs = vec![
            msg("system", "You are helpful."),
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
            msg("user", "Bye"),
        ];
        let rendered = render_messages(&t, &msgs).unwrap();
        let expected = concat!(
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(rendered, expected);
    }

    #[test]
    fn render_messages_empty_errors() {
        let t = ChatTemplate::new(ModelArch::Qwen2).unwrap();
        assert!(render_messages(&t, &[]).is_err());
    }

    #[test]
    fn render_messages_unknown_role_errors() {
        let t = ChatTemplate::new(ModelArch::Qwen2).unwrap();
        assert!(render_messages(&t, &[msg("tool", "x")]).is_err());
    }

    #[test]
    fn effective_sampling_overrides() {
        let base = SamplingConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 40,
            repetition_penalty: 1.1,
            repetition_window: 64,
        };
        let req: OpenAiChatRequest =
            serde_json::from_str(r#"{"messages":[],"temperature":0.7,"top_p":0.9}"#).unwrap();
        let cfg = effective_sampling(&base, &req);
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_p, 0.9);
        assert_eq!(cfg.top_k, 40); // unchanged
        assert_eq!(cfg.repetition_penalty, 1.1); // unchanged
    }

    #[test]
    fn stop_field_accepts_string_or_array() {
        let one: OpenAiChatRequest =
            serde_json::from_str(r#"{"messages":[],"stop":"END"}"#).unwrap();
        assert!(matches!(one.stop, Some(StopField::One(_))));
        let many: OpenAiChatRequest =
            serde_json::from_str(r#"{"messages":[],"stop":["A","B"]}"#).unwrap();
        assert!(matches!(many.stop, Some(StopField::Many(_))));
    }
}
