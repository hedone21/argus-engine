/// AUF Reader — mmap 기반 파일 파싱 + 무결성 검증 (ENG-ALG-223 §3.12.17.1).
///
/// 검증 순서 (INV-132, INV-133, INV-134):
/// 1. 파일 크기 ≥ 256B
/// 2. magic 검증 → format_major 검증 → capability_required 검증 (INV-132)
/// 3. section table 파싱 + 무결성 검증 (INV-134)
/// 4. required section 존재 확인 (INV-133)
/// 5. backend WEIGHTS_* section 존재 확인 (INV-133)
/// 6. META, TOKENIZER, TENSOR_INDEX 파싱 (lazy 가능)
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::auf::error::{AufError, AufResult};
use crate::auf::header::AufHeader;
use crate::auf::meta::AufMeta;
use crate::auf::section::{
    SectionTable, TAG_META, TAG_TENSOR_INDEX, TAG_TOKENIZER, TAG_WEIGHTS_ADRENO_SOA,
    TAG_WEIGHTS_CPU_AOS, TAG_WEIGHTS_CUDA_AOS,
};
use crate::auf::tensor_index::{TensorDType, TensorEntry, TensorIndex};
use crate::auf::tokenizer::AufTokenizer;

/// backend 식별자.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendTag {
    AdrenoSoa,
    CudaAos,
    CpuAos,
    /// strip 등 backend check를 우회하는 모드.
    Any,
}

impl BackendTag {
    pub fn weights_section_tag(self) -> Option<&'static str> {
        match self {
            BackendTag::AdrenoSoa => Some(TAG_WEIGHTS_ADRENO_SOA),
            BackendTag::CudaAos => Some(TAG_WEIGHTS_CUDA_AOS),
            BackendTag::CpuAos => Some(TAG_WEIGHTS_CPU_AOS),
            BackendTag::Any => None,
        }
    }
}

/// lm_head Q4_0 사전 변환 payload 뷰 (G-1-A spec, INV-135/136).
///
/// `AufView::lm_head_q4_0_payload()` 성공 시 반환되는 zero-copy 슬라이스 컨테이너.
/// `bytes`는 backend variant section 내부의 lm_head payload를 직접 가리킨다.
/// lifetime은 `AufView`에 귀속된다.
#[derive(Debug)]
pub struct LmHeadPayload<'a> {
    /// backend variant section 내의 lm_head Q4_0 raw bytes (zero-copy).
    pub bytes: &'a [u8],
    /// `[vocab_size, hidden_dim]`.
    pub shape: [usize; 2],
    /// 항상 `TensorDType::Q4_0` (INV-135 보장).
    pub dtype: TensorDType,
    /// WEIGHTS section alignment (bytes). 현재는 65536 (64KB) 고정.
    pub alignment: usize,
    /// 이 payload가 추출된 backend variant tag.
    pub variant_tag: &'static str,
}

/// META.default_dtype 문자열을 `TensorDType`으로 변환한다.
///
/// 알 수 없는 문자열이면 `None`을 반환한다 (graceful fallback).
fn dtype_str_to_tensor_dtype(s: &str) -> Option<TensorDType> {
    match s {
        "F32" => Some(TensorDType::F32),
        "F16" => Some(TensorDType::F16),
        "BF16" => Some(TensorDType::BF16),
        "Q4_0" => Some(TensorDType::Q4_0),
        "Q4_1" => Some(TensorDType::Q4_1),
        "Q8_0" => Some(TensorDType::Q8_0),
        "U8" => Some(TensorDType::U8),
        _ => None,
    }
}

/// AUF 파일 view — mmap 보유 + 파싱된 메타데이터 + WEIGHTS payload byte slice.
pub struct AufView {
    /// 파일 mmap (lifetime 보장용).
    _mmap: Mmap,
    pub header: AufHeader,
    pub section_table: SectionTable,
    pub meta: AufMeta,
    pub tokenizer: AufTokenizer,
    pub tensor_index: TensorIndex,
    /// WEIGHTS_* payload의 파일 내 byte range (offset, size).
    /// `BackendTag::Any`이면 None.
    pub weights_range: Option<(u64, u64)>,
}

impl AufView {
    /// WEIGHTS payload byte slice를 반환한다. `BackendTag::Any`이면 None.
    ///
    /// WEIGHTS payload는 mmap slice이므로 별도 복사 없이 zero-copy 접근 가능.
    /// 반환된 slice의 lifetime은 `AufView`에 귀속된다.
    pub fn weights_bytes(&self) -> Option<&[u8]> {
        self.weights_range
            .map(|(offset, size)| &self._mmap[offset as usize..][..size as usize])
    }

    /// lm_head Q4_0 사전 변환 payload를 반환한다 (INV-135/136).
    ///
    /// 호출 시 `vocab_size` / `hidden_dim`은 model config의 값이어야 한다.
    /// backend variant는 AufView를 `open()` 시 지정한 `BackendTag`에서 결정된다.
    ///
    /// # 반환값
    ///
    /// - `Ok(None)` — capability bit 2 = 0 (INV-136). 호출자는 runtime fallback
    ///   (`quantize_lm_head_to_q4_0`) 을 사용한다.
    /// - `Ok(Some(payload))` — entry 검증 통과, zero-copy bytes 반환.
    /// - `Err(...)` — capability bit 2 = 1이지만 INV-135 위반 (entry 부재, dtype 불일치,
    ///   shape 불일치, variant payload 부재).
    ///
    /// # Backend variant 분기
    ///
    /// `weights_range`가 `Some((offset, size))`이면 WEIGHTS section이 특정 backend variant에
    /// 대해 열려 있으므로 `tensor_index.variant_index_for_tag(variant_tag)` 를 통해
    /// 올바른 variant column을 선택한다. `BackendTag::Any`이면 weights_range=None이므로
    /// 이 accessor를 사용해서는 안 된다 (strip 용도).
    pub fn lm_head_q4_0_payload(
        &self,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> AufResult<Option<LmHeadPayload<'_>>> {
        // INV-136: bit 2 = 0 → None (runtime fallback).
        if !self.header.has_lm_head_q4_0() {
            return Ok(None);
        }

        // INV-135: bit 2 = 1이면 TENSOR_INDEX에 entry가 정확히 1개 존재해야 한다.
        let entry = self
            .tensor_index
            .find_lm_head_entry()
            .ok_or(AufError::LmHeadEntryMissing)?;

        // dtype 검증 (Q4_0 = 3).
        if entry.dtype != TensorDType::Q4_0.as_u32() {
            return Err(AufError::LmHeadDtypeMismatch {
                found_dtype: entry.dtype,
            });
        }

        // shape 검증: entry.shape은 outermost-first [vocab_size, hidden_dim].
        let expected_shape = [vocab_size as u64, hidden_dim as u64];
        if entry.shape.len() != 2
            || entry.shape[0] != expected_shape[0]
            || entry.shape[1] != expected_shape[1]
        {
            return Err(AufError::LmHeadShapeMismatch {
                expected: [vocab_size, hidden_dim],
                found: entry.shape.clone(),
            });
        }

        // backend variant 선택: weights_range가 Some인 경우에만 유효한 variant가 있다.
        let (weights_offset, _weights_size) = self.weights_range.ok_or_else(|| {
            AufError::Other(
                "lm_head_q4_0_payload: BackendTag::Any does not support lm_head accessor"
                    .to_owned(),
            )
        })?;

        // variant tag 문자열 확인 (weights_range가 있으면 반드시 열린 variant가 있음).
        // section_table에서 해당 offset의 tag를 역조회한다.
        let variant_tag: &'static str = self.resolve_variant_tag(weights_offset)?;

        let variant_idx = self
            .tensor_index
            .variant_index_for_tag(variant_tag)
            .ok_or_else(|| AufError::LmHeadVariantPayloadMissing {
                variant_tag: variant_tag.to_owned(),
            })?;

        let var_offset = entry
            .variant_offsets
            .get(variant_idx)
            .copied()
            .unwrap_or(u64::MAX);
        let var_size = entry.variant_sizes.get(variant_idx).copied().unwrap_or(0);

        if var_offset == u64::MAX || var_size == 0 {
            return Err(AufError::LmHeadVariantPayloadMissing {
                variant_tag: variant_tag.to_owned(),
            });
        }

        // weights_bytes()는 WEIGHTS section의 section-local slice.
        // var_offset은 section-local이므로 직접 인덱싱한다.
        let weights_bytes = self
            .weights_bytes()
            .expect("weights_range Some implies weights_bytes Some");

        let start = var_offset as usize;
        let end = start + var_size as usize;
        if end > weights_bytes.len() {
            return Err(AufError::Other(format!(
                "lm_head payload [{start}..{end}) exceeds WEIGHTS section size {}",
                weights_bytes.len()
            )));
        }

        Ok(Some(LmHeadPayload {
            bytes: &weights_bytes[start..end],
            shape: [vocab_size, hidden_dim],
            dtype: TensorDType::Q4_0,
            alignment: crate::auf::writer::WEIGHTS_ALIGNMENT as usize,
            variant_tag,
        }))
    }

    /// ENG-ALG-225 precedence: 명시 dtype > META.default_dtype > first-match.
    ///
    /// `layer_idx` / `kind`에 해당하는 `TensorEntry`를 dtype 우선순위에 따라 조회한다.
    ///
    /// # Precedence (ENG-ALG-225)
    ///
    /// 1. `requested_dtype = Some(d)` — 해당 dtype의 entry를 명시 조회한다.
    ///    존재하지 않으면 `AufError::DtypeNotAvailable`을 반환한다.
    /// 2. `requested_dtype = None` + `META.default_dtype = Some(s)` — `s`를 파싱하여
    ///    해당 dtype의 entry를 조회한다. 파싱 실패 또는 entry 부재 시 first-match로 fallback.
    /// 3. `requested_dtype = None` + `META.default_dtype = None` — first-match (entries_for의 첫 번째).
    ///
    /// # Errors
    ///
    /// - `requested_dtype`이 명시되었지만 해당 entry가 없으면 `AufError::DtypeNotAvailable`.
    /// - `(layer_idx, kind)` 자체에 entry가 0개이면 `AufError::DtypeNotAvailable`.
    pub fn lookup_tensor(
        &self,
        layer_idx: u32,
        kind: u32,
        requested_dtype: Option<TensorDType>,
    ) -> AufResult<&TensorEntry> {
        if let Some(dtype) = requested_dtype {
            // Precedence 1: 명시 dtype 조회.
            self.tensor_index
                .find_entry_by_dtype(layer_idx, kind, dtype.as_u32())
                .ok_or(AufError::DtypeNotAvailable {
                    layer_idx,
                    kind,
                    dtype: dtype.as_u32(),
                })
        } else {
            let candidates = self.tensor_index.entries_for(layer_idx, kind);
            if candidates.is_empty() {
                return Err(AufError::DtypeNotAvailable {
                    layer_idx,
                    kind,
                    dtype: u32::MAX,
                });
            }

            // Precedence 2: META.default_dtype가 있으면 그 dtype을 시도한다.
            if let Some(ref default_str) = self.meta.default_dtype {
                let default_dtype_opt = dtype_str_to_tensor_dtype(default_str);
                if let Some(default_dtype) = default_dtype_opt
                    && let Some(entry) = self.tensor_index.find_entry_by_dtype(
                        layer_idx,
                        kind,
                        default_dtype.as_u32(),
                    )
                {
                    return Ok(entry);
                }
                // default_dtype 파싱 실패 또는 해당 entry 부재 → first-match fallback.
            }

            // Precedence 3: first-match.
            Ok(candidates[0])
        }
    }

    /// `weights_range.offset`으로 어떤 WEIGHTS variant tag가 열렸는지 역조회한다.
    ///
    /// section_table에서 offset이 일치하는 entry의 tag를 찾아 static str로 반환한다.
    fn resolve_variant_tag(&self, weights_offset: u64) -> AufResult<&'static str> {
        use crate::auf::section::{
            TAG_WEIGHTS_ADRENO_SOA, TAG_WEIGHTS_CPU_AOS, TAG_WEIGHTS_CUDA_AOS,
        };
        for e in &self.section_table.entries {
            if e.offset == weights_offset {
                let tag = e.tag();
                return match tag {
                    TAG_WEIGHTS_ADRENO_SOA => Ok(TAG_WEIGHTS_ADRENO_SOA),
                    TAG_WEIGHTS_CUDA_AOS => Ok(TAG_WEIGHTS_CUDA_AOS),
                    TAG_WEIGHTS_CPU_AOS => Ok(TAG_WEIGHTS_CPU_AOS),
                    other => Err(AufError::Other(format!(
                        "resolve_variant_tag: unknown WEIGHTS tag '{other}'"
                    ))),
                };
            }
        }
        Err(AufError::Other(
            "resolve_variant_tag: no section entry matches weights_range offset".to_owned(),
        ))
    }

    /// mmap 전체 바이트 slice를 반환한다 (stripper 등 내부 용도).
    pub fn raw_bytes(&self) -> &[u8] {
        &self._mmap[..]
    }

    /// 파일 전체 크기 (mmap 길이).
    pub fn file_size(&self) -> u64 {
        self._mmap.len() as u64
    }
}

impl std::fmt::Debug for AufView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AufView")
            .field("format_major", &self.header.format_major)
            .field("format_minor", &self.header.format_minor)
            .field("section_count", &self.header.section_count)
            .field("weights_range", &self.weights_range)
            .finish()
    }
}

/// AUF 파일을 mmap으로 열고 검증 + 파싱하여 `AufView`를 반환한다.
///
/// ENG-ALG-223 §3.12.17.1 Reader 알고리즘 구현.
pub fn open(path: &Path, backend: BackendTag) -> AufResult<AufView> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file) }?;
    open_from_mmap(mmap, backend)
}

/// 바이트열에서 AufView를 파싱한다 (테스트 및 단위 검증 친화).
///
/// 내부적으로 임시 파일을 통해 mmap을 생성한다.
pub fn open_from_bytes(bytes: Vec<u8>, backend: BackendTag) -> AufResult<AufView> {
    use std::io::Write;
    let mut f = tempfile::tempfile().map_err(AufError::Io)?;
    f.write_all(&bytes).map_err(AufError::Io)?;
    let mmap = unsafe { MmapOptions::new().map(&f) }?;
    open_from_mmap(mmap, backend)
}

/// mmap에서 AufView를 파싱하는 실제 로직.
fn open_from_mmap(mmap: Mmap, backend: BackendTag) -> AufResult<AufView> {
    let bytes: &[u8] = &mmap;
    let file_size = bytes.len() as u64;

    // (1) 헤더 파싱 + 검증 (INV-132)
    let header = AufHeader::from_bytes(bytes)?;
    header.validate()?;

    // format_major=0 경고 (실험적)
    if header.format_major == 0 {
        log::warn!("AUF format_major=0 is experimental");
    }

    // (2) section table 영역 경계 확인
    let section_table_end = header
        .section_table_offset
        .checked_add(header.section_count as u64 * 48)
        .ok_or_else(|| AufError::Other("section_table_offset overflow".to_owned()))?;
    if section_table_end > file_size {
        return Err(AufError::SectionTableTruncated);
    }

    let section_table = SectionTable::from_bytes(
        &bytes[header.section_table_offset as usize..],
        header.section_count,
    )?;

    // (3) section 무결성 검증 (INV-134)
    section_table.validate_unique_tags()?;
    section_table.validate_flags()?;
    section_table.validate_no_compressed()?;
    section_table.validate_ranges(header.payload_start_offset, file_size)?;

    // (4) required section 존재 확인 (INV-133)
    let meta_entry = section_table
        .find(TAG_META)
        .ok_or_else(|| AufError::RequiredSectionMissing {
            tag: TAG_META.to_owned(),
        })?
        .clone();
    let tok_entry = section_table
        .find(TAG_TOKENIZER)
        .ok_or_else(|| AufError::RequiredSectionMissing {
            tag: TAG_TOKENIZER.to_owned(),
        })?
        .clone();
    let tidx_entry = section_table
        .find(TAG_TENSOR_INDEX)
        .ok_or_else(|| AufError::RequiredSectionMissing {
            tag: TAG_TENSOR_INDEX.to_owned(),
        })?
        .clone();

    // (5) backend WEIGHTS_* 확인 (INV-133)
    let weights_range = if let Some(weights_tag) = backend.weights_section_tag() {
        let entry =
            section_table
                .find(weights_tag)
                .ok_or_else(|| AufError::WeightsSectionMissing {
                    weights_tag: weights_tag.to_owned(),
                })?;
        Some((entry.offset, entry.size))
    } else {
        None // BackendTag::Any
    };

    // (6) META 파싱
    let meta_bytes = &bytes[meta_entry.offset as usize..][..meta_entry.size as usize];
    let meta = AufMeta::from_json_bytes(meta_bytes)?;

    // (7) TOKENIZER 파싱
    let tok_bytes = &bytes[tok_entry.offset as usize..][..tok_entry.size as usize];
    let tokenizer = AufTokenizer::from_bytes(tok_bytes)?;

    // (8) TENSOR_INDEX 파싱
    let tidx_bytes = &bytes[tidx_entry.offset as usize..][..tidx_entry.size as usize];
    let tensor_index = TensorIndex::from_bytes(tidx_bytes)?;

    Ok(AufView {
        _mmap: mmap,
        header,
        section_table,
        meta,
        tokenizer,
        tensor_index,
        weights_range,
    })
}

// tempfile crate 없이 동작할 수 있도록 fallback — engine의 Cargo.toml에 tempfile이 없으면 표준 라이브러리로 대체
// (실제로는 writer 테스트에서 tempfile을 사용하므로 Cargo.toml에 추가 필요)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auf::meta::AufMeta;
    use crate::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
    use crate::auf::writer::AufWriter;

    fn make_meta() -> AufMeta {
        AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 2,
            n_heads_q: 4,
            n_kv_heads: 2,
            head_dim: 8,
            hidden_dim: 32,
            ffn_dim: 64,
            vocab_size: 10,
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
            tokens: vec![b"hello".to_vec(), b"world".to_vec()],
            merges: vec![],
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            unk_id: 0,
            chat_template: None,
        }
    }

    fn build_auf_bytes(weights_payload: &[u8], weights_tag: &str) -> Vec<u8> {
        AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
            .add_weights_section(weights_tag, weights_payload.to_vec())
            .build()
            .unwrap()
    }

    #[test]
    fn reader_cpu_aos_ok() {
        let payload = vec![0u8; 128];
        let bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
        assert_eq!(view.meta.architecture, "llama");
        assert_eq!(view.tokenizer.tokens.len(), 2);
        let wr = view.weights_range.unwrap();
        assert_eq!(wr.1, 128);
    }

    #[test]
    fn reader_any_skips_weights_check() {
        let payload = vec![0u8; 64];
        let bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        let view = open_from_bytes(bytes, BackendTag::Any).unwrap();
        assert!(view.weights_range.is_none());
    }

    #[test]
    fn reader_wrong_backend_fails() {
        let payload = vec![0u8; 64];
        let bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        let err = open_from_bytes(bytes, BackendTag::AdrenoSoa).unwrap_err();
        assert!(matches!(err, AufError::WeightsSectionMissing { .. }));
    }

    #[test]
    fn reader_missing_meta_fails() {
        // META 없는 AUF 만들기 — raw bytes로 직접 조작은 복잡하므로
        // magic을 오염시켜 magic mismatch로 테스트
        let payload = vec![0u8; 64];
        let mut bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        bytes[0] = 0xFF;
        let err = open_from_bytes(bytes, BackendTag::CpuAos).unwrap_err();
        assert!(matches!(err, AufError::MagicMismatch));
    }

    #[test]
    fn reader_bad_format_major_fails() {
        let payload = vec![0u8; 64];
        let mut bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        // format_major offset=8, set to 1
        bytes[8] = 1;
        bytes[9] = 0;
        let err = open_from_bytes(bytes, BackendTag::CpuAos).unwrap_err();
        assert!(matches!(
            err,
            AufError::UnsupportedFormatMajor { found: 1, .. }
        ));
    }

    // ── lm_head Q4_0 accessor 테스트 (INV-135/136) ─────────────────────────

    /// make_meta() 기준: vocab_size=10, hidden_dim=32.
    /// Q4_0 blocks = 10 * (32/32) = 10. 각 18B → 총 180B.
    const LM_HEAD_Q4_0_SIZE: usize = 10 * 18; // 180B

    /// lm_head entry가 포함된 TENSOR_INDEX를 빌드한다.
    fn make_lm_head_tensor_index(
        weights_tag: &str,
        lm_head_offset: u64,
        lm_head_size: u64,
        dtype: u32,
        shape: Vec<u64>,
    ) -> crate::auf::tensor_index::TensorIndex {
        use crate::auf::tensor_index::{LAYER_IDX_CROSS, TensorEntry, TensorIndex, TensorKind};
        let mut variant_tag = [0u8; 24];
        let tag_bytes = weights_tag.as_bytes();
        variant_tag[..tag_bytes.len().min(24)]
            .copy_from_slice(&tag_bytes[..tag_bytes.len().min(24)]);
        TensorIndex {
            variant_tags: vec![variant_tag],
            entries: vec![TensorEntry {
                layer_idx: LAYER_IDX_CROSS,
                kind: TensorKind::LmHead.as_u32(),
                dtype,
                shape,
                alignment: 65536,
                variant_offsets: vec![lm_head_offset],
                variant_sizes: vec![lm_head_size],
            }],
        }
    }

    /// lm_head Q4_0 payload를 포함한 AUF 바이트를 빌드한다.
    /// weights_payload = layer_weights + lm_head_payload (lm_head은 끝에 동봉).
    fn build_auf_with_lm_head(
        lm_head_bytes: &[u8],
        weights_tag: &str,
        dtype: u32,
        shape: Vec<u64>,
    ) -> Vec<u8> {
        // layer dummy payload (32B) + lm_head payload가 뒤에 붙는다고 가정.
        // lm_head_offset = layer_dummy_size (section-local).
        let layer_dummy: Vec<u8> = vec![0xABu8; 32];
        let lm_head_offset = layer_dummy.len() as u64;
        let lm_head_size = lm_head_bytes.len() as u64;

        let mut combined = layer_dummy;
        combined.extend_from_slice(lm_head_bytes);

        let tidx =
            make_lm_head_tensor_index(weights_tag, lm_head_offset, lm_head_size, dtype, shape);
        AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
            .with_lm_head_q4_0(true)
            .with_tensor_index(tidx)
            .add_weights_section(weights_tag, combined)
            .build()
            .unwrap()
    }

    /// INV-136: capability bit 2 = 0 (v0.1.0 AUF) → accessor Ok(None).
    #[test]
    fn lm_head_payload_bit2_zero_returns_none() {
        let payload = vec![0u8; 128];
        let bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
        // bit 2 = 0 (default)
        assert!(!view.header.has_lm_head_q4_0());
        let result = view.lm_head_q4_0_payload(10, 32).unwrap();
        assert!(result.is_none(), "bit2=0 should return None");
    }

    /// INV-135 happy path: bit 2 = 1 + entry 정상 → Ok(Some(payload)).
    #[test]
    fn lm_head_payload_bit2_set_entry_ok() {
        use crate::auf::tensor_index::TensorDType;
        let lm_head_data: Vec<u8> = (0..LM_HEAD_Q4_0_SIZE as u8).collect();
        let bytes = build_auf_with_lm_head(
            &lm_head_data,
            "WEIGHTS_CPU_AOS",
            TensorDType::Q4_0.as_u32(),
            vec![10, 32],
        );
        let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
        assert!(view.header.has_lm_head_q4_0());

        let payload = view
            .lm_head_q4_0_payload(10, 32)
            .unwrap()
            .expect("expected Some(LmHeadPayload)");

        assert_eq!(payload.shape, [10, 32]);
        assert_eq!(payload.dtype, TensorDType::Q4_0);
        assert_eq!(payload.bytes.len(), LM_HEAD_Q4_0_SIZE);
        assert_eq!(payload.variant_tag, "WEIGHTS_CPU_AOS");
        // bytes 정확성: lm_head_data와 일치
        assert_eq!(payload.bytes, &lm_head_data[..]);
    }

    /// INV-135: bit 2 = 1이지만 TENSOR_INDEX에 lm_head entry 없음 → Err(LmHeadEntryMissing).
    #[test]
    fn lm_head_payload_entry_missing() {
        // bit 2 = 1이지만 tensor_index는 빈 (lm_head entry 없음).
        let empty_tidx = crate::auf::tensor_index::TensorIndex {
            variant_tags: vec![[0u8; 24]],
            entries: vec![],
        };
        let bytes = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
            .with_lm_head_q4_0(true)
            .with_tensor_index(empty_tidx)
            .add_weights_section("WEIGHTS_CPU_AOS", vec![0u8; 64])
            .build()
            .unwrap();
        let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
        let err = view.lm_head_q4_0_payload(10, 32).unwrap_err();
        assert!(
            matches!(err, AufError::LmHeadEntryMissing),
            "expected LmHeadEntryMissing, got: {err}"
        );
    }

    /// INV-135: bit 2 = 1 + entry 존재 + dtype != Q4_0 → Err(LmHeadDtypeMismatch).
    #[test]
    fn lm_head_payload_dtype_mismatch() {
        use crate::auf::tensor_index::TensorDType;
        let lm_head_data = vec![0u8; LM_HEAD_Q4_0_SIZE];
        let bytes = build_auf_with_lm_head(
            &lm_head_data,
            "WEIGHTS_CPU_AOS",
            TensorDType::F16.as_u32(), // 잘못된 dtype
            vec![10, 32],
        );
        let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
        let err = view.lm_head_q4_0_payload(10, 32).unwrap_err();
        assert!(
            matches!(err, AufError::LmHeadDtypeMismatch { found_dtype: 1 }),
            "expected LmHeadDtypeMismatch, got: {err}"
        );
    }

    /// INV-135: bit 2 = 1 + entry 존재 + shape 불일치 → Err(LmHeadShapeMismatch).
    #[test]
    fn lm_head_payload_shape_mismatch() {
        use crate::auf::tensor_index::TensorDType;
        let lm_head_data = vec![0u8; LM_HEAD_Q4_0_SIZE];
        let bytes = build_auf_with_lm_head(
            &lm_head_data,
            "WEIGHTS_CPU_AOS",
            TensorDType::Q4_0.as_u32(),
            vec![99, 32], // vocab_size=99 vs. expected 10
        );
        let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
        let err = view.lm_head_q4_0_payload(10, 32).unwrap_err(); // vocab_size=10
        assert!(
            matches!(
                err,
                AufError::LmHeadShapeMismatch {
                    expected: [10, 32],
                    ..
                }
            ),
            "expected LmHeadShapeMismatch, got: {err}"
        );
    }
}
