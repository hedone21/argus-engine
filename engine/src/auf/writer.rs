/// AUF Writer — AUF v0.1 파일 빌드 (ENG-ALG-223 §3.12.17.2).
///
/// 보증:
/// - atomic file replace (`tempfile + rename`) — ENG-ALG-C11.
/// - section offset은 cursor 기반 단조 증가 → overlap 자동 방지 (INV-134).
/// - WEIGHTS_* section은 64KB align (THP 친화).
use std::path::Path;

use crate::auf::error::{AufError, AufResult};
use crate::auf::header::{AufHeader, HEADER_SIZE};
use crate::auf::meta::AufMeta;
use crate::auf::section::{
    SECTION_REQUIRED, SECTION_STRIPPABLE, SectionEntry, SectionTable, TAG_META, TAG_TENSOR_INDEX,
    TAG_TOKENIZER,
};
use crate::auf::tensor_index::TensorIndex;
use crate::auf::tokenizer::AufTokenizer;

/// WEIGHTS_* payload alignment (64KB).
pub const WEIGHTS_ALIGNMENT: u64 = 65536;
/// META/TOKENIZER/TENSOR_INDEX payload alignment (8B).
pub const META_ALIGNMENT: u64 = 8;

/// AUF 파일 빌더.
pub struct AufWriter {
    meta: AufMeta,
    tokenizer: AufTokenizer,
    source_hash: [u8; 32],
    source_size: u64,
    source_mtime: u64,
    /// (tag, payload) 순서 보존
    weights_sections: Vec<(String, Vec<u8>)>,
    /// 외부에서 설정한 tensor_index (없으면 빈 index)
    tensor_index: Option<TensorIndex>,
    /// 생성 도구 이름
    created_by: String,
    /// lm_head Q4_0 사전 변환이 포함되었는지 (v0.1.1, Sprint G-1).
    ///
    /// `true`이면 `build()`에서 `capability_optional` bit 2 set + format_patch = 1.
    /// `false`이면 v0.1.0 byte-level 호환 출력.
    lm_head_q4_0_present: bool,
    /// AUF v0.2 multi-dtype variant 모드 활성화 (Sprint C, INV-139).
    ///
    /// `true`이면 `build()`에서 `capability_optional` bit 3 set + format_minor = 2.
    /// 이 값은 `cmd_build`가 `--dtypes`에 dtype을 2개 이상 지정한 경우 자동 활성화한다.
    multi_dtype_enabled: bool,
}

impl AufWriter {
    pub fn new(
        meta: AufMeta,
        tokenizer: AufTokenizer,
        source_hash: [u8; 32],
        source_size: u64,
        source_mtime: u64,
    ) -> Self {
        AufWriter {
            meta,
            tokenizer,
            source_hash,
            source_size,
            source_mtime,
            weights_sections: Vec::new(),
            tensor_index: None,
            created_by: "argus_engine v0.1.0".to_owned(),
            lm_head_q4_0_present: false,
            multi_dtype_enabled: false,
        }
    }

    /// AUF v0.2 `MULTI_DTYPE_VARIANTS` capability를 선언한다 (Sprint C, INV-139).
    ///
    /// `true`이면 `build()`가 `capability_optional` bit 3 set + format_minor = 2.
    /// `false` (기본값)이면 single-dtype 모드 (v0.1.x 호환).
    pub fn with_multi_dtype(mut self, enabled: bool) -> Self {
        self.multi_dtype_enabled = enabled;
        self
    }

    /// `LM_HEAD_PRECOMPUTED_Q4_0` capability를 선언한다 (v0.1.1, Sprint G-1).
    ///
    /// `true`이면 `build()`가 `capability_optional` bit 2 set + format_patch = 1.
    /// `false` (기본값)이면 v0.1.0 호환 출력.
    pub fn with_lm_head_q4_0(mut self, enabled: bool) -> Self {
        self.lm_head_q4_0_present = enabled;
        self
    }

    /// `created_by` 문자열 설정 (최대 32B).
    pub fn with_created_by(mut self, s: &str) -> Self {
        self.created_by = s.to_owned();
        self
    }

    /// TENSOR_INDEX를 설정한다.
    pub fn with_tensor_index(mut self, idx: TensorIndex) -> Self {
        self.tensor_index = Some(idx);
        self
    }

    /// WEIGHTS_* section을 추가한다. tag는 `"WEIGHTS_ADRENO_SOA"` 등.
    ///
    /// 동일 tag를 두 번 추가하면 `build()` 시 에러 반환.
    pub fn add_weights_section(mut self, tag: &str, payload: Vec<u8>) -> Self {
        self.weights_sections.push((tag.to_owned(), payload));
        self
    }

    /// AUF 파일 바이트열을 빌드한다.
    ///
    /// section layout 결정 → header 직렬화 → section table 직렬화 → payload 직렬화.
    pub fn build(self) -> AufResult<Vec<u8>> {
        // WEIGHTS_* section이 없으면 useless asset 경고 (하지만 테스트 편의를 위해 허용)
        let section_count = 3 + self.weights_sections.len();

        // (1) payload 직렬화
        let meta_bytes = self.meta.to_json_bytes()?;
        let tok_bytes = self.tokenizer.to_bytes();
        let tidx_bytes = if let Some(idx) = &self.tensor_index {
            idx.to_bytes()
        } else {
            TensorIndex {
                variant_tags: Vec::new(),
                entries: Vec::new(),
            }
            .to_bytes()
        };

        // (2) section table 크기
        let section_table_offset = HEADER_SIZE as u64; // 256
        let section_table_size = section_count as u64 * 48;

        // (3) payload_start_offset = align_up(256 + section_table_size, 64KB)
        let after_table = section_table_offset + section_table_size;
        let payload_start_offset = align_up(after_table, WEIGHTS_ALIGNMENT);

        // (4) section layout 결정 (cursor 기반 단조 증가)
        let mut cursor = payload_start_offset;
        let mut sections: Vec<SectionEntry> = Vec::with_capacity(section_count);

        // META
        let meta_offset = cursor;
        sections.push(SectionEntry::new(
            TAG_META,
            meta_offset,
            meta_bytes.len() as u64,
            SECTION_REQUIRED,
            1,
        )?);
        cursor = align_up(cursor + meta_bytes.len() as u64, META_ALIGNMENT);

        // TOKENIZER
        let tok_offset = cursor;
        sections.push(SectionEntry::new(
            TAG_TOKENIZER,
            tok_offset,
            tok_bytes.len() as u64,
            SECTION_REQUIRED,
            1,
        )?);
        cursor = align_up(cursor + tok_bytes.len() as u64, META_ALIGNMENT);

        // TENSOR_INDEX
        let tidx_offset = cursor;
        sections.push(SectionEntry::new(
            TAG_TENSOR_INDEX,
            tidx_offset,
            tidx_bytes.len() as u64,
            SECTION_REQUIRED,
            1,
        )?);
        // 다음은 WEIGHTS_* → 64KB align
        cursor = align_up(cursor + tidx_bytes.len() as u64, WEIGHTS_ALIGNMENT);

        // WEIGHTS_* sections
        let mut weights_offsets: Vec<(u64, u64)> = Vec::new();
        // 중복 tag 검사
        {
            let mut seen = std::collections::HashSet::new();
            for (tag, _) in &self.weights_sections {
                if !seen.insert(tag.clone()) {
                    return Err(AufError::Other(format!(
                        "Duplicate WEIGHTS section tag: {tag}"
                    )));
                }
            }
        }
        for (tag, payload) in &self.weights_sections {
            let off = cursor;
            let sz = payload.len() as u64;
            sections.push(SectionEntry::new(tag, off, sz, SECTION_STRIPPABLE, 1)?);
            weights_offsets.push((off, sz));
            cursor = align_up(cursor + sz, WEIGHTS_ALIGNMENT);
        }

        let total_size = cursor; // 마지막 cursor가 파일 끝 (align-up 후)

        // (5) header 생성
        let mut header = AufHeader::new_v01(
            &self.created_by,
            self.source_hash,
            self.source_size,
            self.source_mtime,
            section_count as u32,
            section_table_offset,
            payload_start_offset,
        );
        // v0.1.1 (Sprint G-1): lm_head Q4_0 사전 변환 capability 선언.
        // 미선언 시 v0.1.0 byte-level 호환 (format_patch = 0, bit 2 = 0).
        header.set_lm_head_q4_0_capability(self.lm_head_q4_0_present);
        // v0.2 (Sprint C, INV-139): multi-dtype variant capability 선언.
        // 활성화 시 bit 3 set + format_minor = 2. 미활성화 시 v0.1.x 호환.
        header.set_multi_dtype_capability(self.multi_dtype_enabled);

        // (6) 파일 바이트 조립
        let mut out: Vec<u8> = vec![0u8; total_size as usize];
        out[..HEADER_SIZE].copy_from_slice(&header.to_bytes());

        let table = SectionTable { entries: sections };
        let table_bytes = table.to_bytes();
        let table_start = section_table_offset as usize;
        out[table_start..table_start + table_bytes.len()].copy_from_slice(&table_bytes);

        // META payload
        out[meta_offset as usize..meta_offset as usize + meta_bytes.len()]
            .copy_from_slice(&meta_bytes);
        // TOKENIZER payload
        out[tok_offset as usize..tok_offset as usize + tok_bytes.len()].copy_from_slice(&tok_bytes);
        // TENSOR_INDEX payload
        out[tidx_offset as usize..tidx_offset as usize + tidx_bytes.len()]
            .copy_from_slice(&tidx_bytes);
        // WEIGHTS_* payloads
        for ((off, _sz), (_, payload)) in weights_offsets.iter().zip(&self.weights_sections) {
            out[*off as usize..*off as usize + payload.len()].copy_from_slice(payload);
        }

        Ok(out)
    }

    /// AUF 바이트열을 파일에 atomic write (tempfile + rename, ENG-ALG-C11).
    pub fn write_to_file(self, path: &Path) -> AufResult<()> {
        use std::io::Write;

        let bytes = self.build()?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let tmp_path = parent.join(format!(
            ".auf_tmp_{}.auf.tmp",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));

        {
            let mut f = std::fs::File::create(&tmp_path).map_err(AufError::Io)?;
            f.write_all(&bytes).map_err(AufError::Io)?;
            f.sync_all().map_err(AufError::Io)?;
        }
        std::fs::rename(&tmp_path, path).map_err(AufError::Io)?;
        Ok(())
    }
}

/// `val`을 `align`의 배수로 올림 정렬.
pub fn align_up(val: u64, align: u64) -> u64 {
    if align == 0 {
        return val;
    }
    val.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auf::tokenizer::TOKENIZER_KIND_BPE;

    fn make_meta() -> AufMeta {
        AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 2,
            n_heads_q: 4,
            n_kv_heads: 2,
            head_dim: 8,
            hidden_dim: 32,
            ffn_dim: 64,
            vocab_size: 3,
            max_seq_len: 128,
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
            tokens: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            merges: vec![],
            bos_id: 0,
            eos_id: 1,
            pad_id: -1,
            unk_id: -1,
            chat_template: None,
        }
    }

    #[test]
    fn build_produces_valid_auf() {
        let payload = vec![42u8; 512];
        let bytes = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
            .add_weights_section("WEIGHTS_CPU_AOS", payload.clone())
            .build()
            .unwrap();

        // 최소 크기 확인 (헤더 256 + section table + payload_start 64KB + payloads)
        assert!(bytes.len() >= 65536 + 512);
        // magic 확인
        assert_eq!(&bytes[0..8], b"ARGUS_W\0");
        // format_major = 0
        assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()), 0);
        // format_minor = 1
        assert_eq!(u16::from_le_bytes(bytes[10..12].try_into().unwrap()), 1);
    }

    #[test]
    fn weights_payload_at_64kb_aligned_offset() {
        let payload = vec![0u8; 256];
        let bytes = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
            .add_weights_section("WEIGHTS_CPU_AOS", payload)
            .build()
            .unwrap();

        // payload_start_offset 확인
        let pso = u64::from_le_bytes(bytes[128..136].try_into().unwrap());
        assert_eq!(pso % 65536, 0, "payload_start_offset must be 64KB aligned");

        // section table: 3 required + 1 weights = 4 entries, offset = 256
        // entry[3] (WEIGHTS_CPU_AOS) offset = 256 + 48*4 = 256+192=448 에서 시작
        let entry3_start = 256 + 3 * 48; // WEIGHTS section entry
        let weights_offset = u64::from_le_bytes(
            bytes[entry3_start + 16..entry3_start + 24]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            weights_offset % 65536,
            0,
            "WEIGHTS payload offset must be 64KB aligned"
        );
    }

    #[test]
    fn duplicate_weights_tag_rejected() {
        let err = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
            .add_weights_section("WEIGHTS_CPU_AOS", vec![0u8; 64])
            .add_weights_section("WEIGHTS_CPU_AOS", vec![0u8; 64])
            .build()
            .unwrap_err();
        assert!(matches!(err, AufError::Other(_)));
    }

    #[test]
    fn multiple_variants_ok() {
        let bytes = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
            .add_weights_section("WEIGHTS_CPU_AOS", vec![1u8; 64])
            .add_weights_section("WEIGHTS_ADRENO_SOA", vec![2u8; 128])
            .build()
            .unwrap();
        // section count = 5 (META + TOKENIZER + TENSOR_INDEX + 2 WEIGHTS)
        let section_count = u32::from_le_bytes(bytes[112..116].try_into().unwrap());
        assert_eq!(section_count, 5);
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 65536), 0);
        assert_eq!(align_up(1, 65536), 65536);
        assert_eq!(align_up(65536, 65536), 65536);
        assert_eq!(align_up(65537, 65536), 131072);
    }

    #[test]
    fn write_to_file_atomic() {
        use std::io::Read;
        let payload = vec![7u8; 128];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.auf");
        AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 100, 999)
            .add_weights_section("WEIGHTS_CPU_AOS", payload)
            .write_to_file(&path)
            .unwrap();
        assert!(path.exists());
        let mut f = std::fs::File::open(&path).unwrap();
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"ARGUS_W\0");
    }
}
