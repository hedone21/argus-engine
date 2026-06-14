/// AUF TOKENIZER section 직렬화/역직렬화 (ENG-DAT-096.7).
///
/// 구조:
/// ```text
/// [0..16)   magic = "ARGUS_TOK\0\0\0\0\0\0\0" (16B)
/// [16..20)  schema_version: u32 = 1
/// [20..24)  tokenizer_kind: u32  (0 = bpe, 1+ reserved)
/// [24..28)  vocab_size: u32
/// [28..32)  merges_count: u32
/// [32..40)  special_tokens_offset: u64  (section-local)
/// [40..48)  chat_template_offset: u64   (section-local, 0이면 부재)
/// [48..56)  tokens_blob_offset: u64     (section-local)
/// [56..64)  merges_blob_offset: u64     (section-local, 0이면 부재)
/// [64..)    payload blobs
/// ```
use crate::auf::error::{AufError, AufResult};

/// TOKENIZER section magic (16B).
pub const TOKENIZER_MAGIC: &[u8; 16] = b"ARGUS_TOK\0\0\0\0\0\0\0";

/// tokenizer_kind: BPE.
pub const TOKENIZER_KIND_BPE: u32 = 0;

/// TOKENIZER section 헤더 크기 (64B).
pub const TOKENIZER_HEADER_SIZE: usize = 64;

/// 파싱된 TOKENIZER section 내용.
#[derive(Debug, Clone)]
pub struct AufTokenizer {
    /// tokenizer 종류 (v0.1: 0=BPE만 지원).
    pub kind: u32,
    /// vocab 토큰 문자열 목록 (vocab_size개).
    pub tokens: Vec<Vec<u8>>,
    /// BPE merge 쌍 문자열 목록 (merges_count개, "A B" 형태).
    pub merges: Vec<String>,
    /// special token IDs (bos, eos, pad, unk), -1이면 미설정.
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
    pub unk_id: i32,
    /// chat template 문자열 (부재 시 None).
    pub chat_template: Option<String>,
}

impl AufTokenizer {
    /// section payload 바이트열에서 파싱한다.
    pub fn from_bytes(payload: &[u8]) -> AufResult<Self> {
        if payload.len() < TOKENIZER_HEADER_SIZE {
            return Err(AufError::TokenizerFormat {
                detail: "payload too small for header".to_owned(),
            });
        }

        // magic 검증
        if &payload[0..16] != TOKENIZER_MAGIC {
            return Err(AufError::TokenizerFormat {
                detail: "magic mismatch".to_owned(),
            });
        }

        let schema_version = u32::from_le_bytes(payload[16..20].try_into().unwrap());
        if schema_version != 1 {
            return Err(AufError::TokenizerFormat {
                detail: format!("unsupported schema_version={schema_version}"),
            });
        }

        let kind = u32::from_le_bytes(payload[20..24].try_into().unwrap());
        if kind != TOKENIZER_KIND_BPE {
            return Err(AufError::TokenizerFormat {
                detail: format!(
                    "tokenizer_kind={kind} is not supported in schema_version=1 (only kind=0 BPE)"
                ),
            });
        }

        let vocab_size = u32::from_le_bytes(payload[24..28].try_into().unwrap()) as usize;
        let merges_count = u32::from_le_bytes(payload[28..32].try_into().unwrap()) as usize;
        let special_tokens_offset =
            u64::from_le_bytes(payload[32..40].try_into().unwrap()) as usize;
        let chat_template_offset = u64::from_le_bytes(payload[40..48].try_into().unwrap()) as usize;
        let tokens_blob_offset = u64::from_le_bytes(payload[48..56].try_into().unwrap()) as usize;
        let merges_blob_offset = u64::from_le_bytes(payload[56..64].try_into().unwrap()) as usize;

        // tokens_blob
        let tokens = read_length_prefixed_strings(payload, tokens_blob_offset, vocab_size)
            .map_err(|e| AufError::TokenizerFormat {
                detail: format!("tokens_blob: {e}"),
            })?;

        // merges_blob
        let merges = if merges_count > 0 && merges_blob_offset > 0 {
            let raw = read_length_prefixed_strings(payload, merges_blob_offset, merges_count)
                .map_err(|e| AufError::TokenizerFormat {
                    detail: format!("merges_blob: {e}"),
                })?;
            raw.into_iter()
                .map(|b| {
                    String::from_utf8(b).map_err(|_| "merges entry is not valid UTF-8".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| AufError::TokenizerFormat { detail: e })?
        } else {
            Vec::new()
        };

        // special tokens (12B: bos_id i32, eos_id i32, pad_id i32, unk_id i32)
        if payload.len() < special_tokens_offset + 16 {
            return Err(AufError::TokenizerFormat {
                detail: "special_tokens_offset out of bounds".to_owned(),
            });
        }
        let bos_id = i32::from_le_bytes(
            payload[special_tokens_offset..special_tokens_offset + 4]
                .try_into()
                .unwrap(),
        );
        let eos_id = i32::from_le_bytes(
            payload[special_tokens_offset + 4..special_tokens_offset + 8]
                .try_into()
                .unwrap(),
        );
        let pad_id = i32::from_le_bytes(
            payload[special_tokens_offset + 8..special_tokens_offset + 12]
                .try_into()
                .unwrap(),
        );
        let unk_id = i32::from_le_bytes(
            payload[special_tokens_offset + 12..special_tokens_offset + 16]
                .try_into()
                .unwrap(),
        );

        // chat_template (optional)
        let chat_template = if chat_template_offset > 0 {
            Some(
                read_length_prefixed_utf8(payload, chat_template_offset).map_err(|e| {
                    AufError::TokenizerFormat {
                        detail: format!("chat_template: {e}"),
                    }
                })?,
            )
        } else {
            None
        };

        Ok(AufTokenizer {
            kind,
            tokens,
            merges,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            chat_template,
        })
    }

    /// section payload 바이트열로 직렬화한다.
    pub fn to_bytes(&self) -> Vec<u8> {
        // payload 영역 먼저 구성
        let mut payload_parts: Vec<Vec<u8>> = Vec::new();

        // [0] tokens_blob
        let tokens_blob = build_length_prefixed_strings(&self.tokens);
        // [1] merges_blob (빈 경우 empty)
        let merges_blob: Vec<u8> = if !self.merges.is_empty() {
            build_length_prefixed_strings(
                &self
                    .merges
                    .iter()
                    .map(|s| s.as_bytes().to_vec())
                    .collect::<Vec<_>>(),
            )
        } else {
            Vec::new()
        };
        // [2] special_tokens (16B: bos eos pad unk each i32)
        let mut special = [0u8; 16];
        special[0..4].copy_from_slice(&self.bos_id.to_le_bytes());
        special[4..8].copy_from_slice(&self.eos_id.to_le_bytes());
        special[8..12].copy_from_slice(&self.pad_id.to_le_bytes());
        special[12..16].copy_from_slice(&self.unk_id.to_le_bytes());
        // [3] chat_template (optional)
        let chat_blob: Vec<u8> = if let Some(ct) = &self.chat_template {
            let b = ct.as_bytes();
            let mut v = Vec::with_capacity(4 + b.len());
            v.extend_from_slice(&(b.len() as u32).to_le_bytes());
            v.extend_from_slice(b);
            v
        } else {
            Vec::new()
        };

        // header 내 offset 계산 (section-local, 헤더=64B 이후)
        let tokens_blob_offset = TOKENIZER_HEADER_SIZE as u64;
        payload_parts.push(tokens_blob);

        let merges_blob_offset = if !self.merges.is_empty() {
            let off = tokens_blob_offset + payload_parts[0].len() as u64;
            payload_parts.push(merges_blob);
            off
        } else {
            payload_parts.push(Vec::new());
            0u64
        };

        let special_tokens_offset =
            tokens_blob_offset + payload_parts[0].len() as u64 + payload_parts[1].len() as u64;
        payload_parts.push(special.to_vec());

        let chat_template_offset = if self.chat_template.is_some() {
            let off = special_tokens_offset + 16;
            payload_parts.push(chat_blob);
            off
        } else {
            payload_parts.push(Vec::new());
            0u64
        };

        // 헤더 직렬화
        let mut hdr = Vec::with_capacity(TOKENIZER_HEADER_SIZE);
        hdr.extend_from_slice(TOKENIZER_MAGIC);
        hdr.extend_from_slice(&1u32.to_le_bytes()); // schema_version
        hdr.extend_from_slice(&self.kind.to_le_bytes());
        hdr.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        hdr.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        hdr.extend_from_slice(&special_tokens_offset.to_le_bytes());
        hdr.extend_from_slice(&chat_template_offset.to_le_bytes());
        hdr.extend_from_slice(&tokens_blob_offset.to_le_bytes());
        hdr.extend_from_slice(&merges_blob_offset.to_le_bytes());
        assert_eq!(hdr.len(), TOKENIZER_HEADER_SIZE);

        let mut out = hdr;
        for part in payload_parts {
            out.extend_from_slice(&part);
        }
        out
    }
}

/// section-local offset에서 `count`개 length-prefixed 문자열 읽기.
fn read_length_prefixed_strings(
    payload: &[u8],
    offset: usize,
    count: usize,
) -> Result<Vec<Vec<u8>>, String> {
    let mut pos = offset;
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        if pos + 4 > payload.len() {
            return Err(format!("entry {i}: length prefix out of bounds"));
        }
        let len = u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + len > payload.len() {
            return Err(format!("entry {i}: data out of bounds (len={len})"));
        }
        result.push(payload[pos..pos + len].to_vec());
        pos += len;
    }
    Ok(result)
}

/// section-local offset에서 length-prefixed UTF-8 문자열 읽기.
fn read_length_prefixed_utf8(payload: &[u8], offset: usize) -> Result<String, String> {
    if offset + 4 > payload.len() {
        return Err("length prefix out of bounds".to_owned());
    }
    let len = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    if offset + 4 + len > payload.len() {
        return Err(format!("data out of bounds (len={len})"));
    }
    String::from_utf8(payload[offset + 4..offset + 4 + len].to_vec())
        .map_err(|_| "invalid UTF-8".to_owned())
}

/// 바이트열 슬라이스를 length-prefixed 블롭으로 직렬화.
fn build_length_prefixed_strings(strings: &[Vec<u8>]) -> Vec<u8> {
    let total: usize = strings.iter().map(|s| 4 + s.len()).sum();
    let mut out = Vec::with_capacity(total);
    for s in strings {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokenizer() -> AufTokenizer {
        AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"hello".to_vec(), b"world".to_vec(), b"<eos>".to_vec()],
            merges: vec!["he llo".to_owned(), "wor ld".to_owned()],
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            unk_id: 0,
            chat_template: Some("<|user|>{{input}}<|assistant|>".to_owned()),
        }
    }

    #[test]
    fn round_trip() {
        let tok = make_tokenizer();
        let bytes = tok.to_bytes();
        let tok2 = AufTokenizer::from_bytes(&bytes).unwrap();
        assert_eq!(tok2.kind, TOKENIZER_KIND_BPE);
        assert_eq!(tok2.tokens.len(), 3);
        assert_eq!(tok2.tokens[0], b"hello");
        assert_eq!(tok2.tokens[2], b"<eos>");
        assert_eq!(tok2.merges.len(), 2);
        assert_eq!(tok2.merges[0], "he llo");
        assert_eq!(tok2.bos_id, 1);
        assert_eq!(tok2.eos_id, 2);
        assert_eq!(tok2.pad_id, -1);
        assert_eq!(tok2.unk_id, 0);
        assert_eq!(
            tok2.chat_template.as_deref(),
            Some("<|user|>{{input}}<|assistant|>")
        );
    }

    #[test]
    fn round_trip_no_merges_no_template() {
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec()],
            merges: vec![],
            bos_id: -1,
            eos_id: -1,
            pad_id: -1,
            unk_id: -1,
            chat_template: None,
        };
        let bytes = tok.to_bytes();
        let tok2 = AufTokenizer::from_bytes(&bytes).unwrap();
        assert_eq!(tok2.tokens.len(), 1);
        assert_eq!(tok2.merges.len(), 0);
        assert!(tok2.chat_template.is_none());
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = make_tokenizer().to_bytes();
        bytes[0] = 0xFF; // magic 오염
        let err = AufTokenizer::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AufError::TokenizerFormat { .. }));
    }

    #[test]
    fn unsupported_kind_rejected() {
        let mut bytes = make_tokenizer().to_bytes();
        // tokenizer_kind offset = 20..24
        bytes[20] = 1; // kind=1 (unigram, unsupported)
        let err = AufTokenizer::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AufError::TokenizerFormat { .. }));
    }
}
