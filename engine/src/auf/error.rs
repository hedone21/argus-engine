/// AUF 파일 파싱/직렬화 에러 타입.
///
/// 모든 에러는 `Result::Err`로 반환되며, panic 없이 진단 메시지를 포함한다 (ENG-ALG-C10).
use std::fmt;

#[derive(Debug)]
pub enum AufError {
    /// 파일이 AUF가 아니다 (magic 불일치).
    MagicMismatch,
    /// format_major가 reader 지원 범위를 초과한다.
    UnsupportedFormatMajor { found: u16, max_supported: u16 },
    /// capability_required 필드에 reader가 인식하지 못하는 비트가 set되어 있다.
    UnknownRequiredCapability { bit: u32 },
    /// required section이 파일에 없다.
    RequiredSectionMissing { tag: String },
    /// backend에 맞는 WEIGHTS_* section이 없다.
    WeightsSectionMissing { weights_tag: String },
    /// section offset/size가 파일 범위를 벗어나거나 payload_start_offset 미만이다 (INV-134).
    SectionRangeInvalid { tag: String, detail: String },
    /// 두 section의 byte range가 overlap된다 (INV-134).
    SectionOverlap { tag_a: String, tag_b: String },
    /// 동일 tag가 두 번 등장한다 (INV-134).
    DuplicateSectionTag { tag: String },
    /// section flag 모순 (REQUIRED + STRIPPABLE 동시 set).
    ContradictoryFlags { tag: String },
    /// SECTION_COMPRESSED가 set되었으나 v0.1 reader는 압축 미지원.
    CompressedSectionUnsupported { tag: String },
    /// section table 영역이 파일 크기를 벗어난다.
    SectionTableTruncated,
    /// 파일이 헤더(256B) 미만이다.
    FileTooSmall,
    /// TOKENIZER section 포맷이 잘못되었다.
    TokenizerFormat { detail: String },
    /// TENSOR_INDEX section 포맷이 잘못되었다.
    TensorIndexFormat { detail: String },
    /// WEIGHTS_* section이 존재하지만 TENSOR_INDEX가 해당 variant를 cover하지 않는다.
    ///
    /// `auf-tool build`가 TensorIndex를 채우지 않았거나, 수동 조작으로 빈 인덱스가 주입된 경우.
    TensorIndexMissingVariant { weights_tag: String },
    /// capability_optional bit 2 = 1이지만 TENSOR_INDEX에 lm_head Q4_0 엔트리가 없다 (INV-135).
    LmHeadEntryMissing,
    /// lm_head entry가 존재하지만 dtype이 Q4_0이 아니다 (INV-135).
    LmHeadDtypeMismatch { found_dtype: u32 },
    /// lm_head entry가 존재하지만 shape이 model config와 일치하지 않는다 (INV-135).
    LmHeadShapeMismatch {
        expected: [usize; 2],
        found: Vec<u64>,
    },
    /// capability_optional bit 2 = 1이지만 lm_head entry에 해당 variant payload가 없다 (INV-135).
    LmHeadVariantPayloadMissing { variant_tag: String },
    /// 요청한 dtype의 entry가 TENSOR_INDEX에 없다 (ENG-ALG-225).
    DtypeNotAvailable {
        layer_idx: u32,
        kind: u32,
        dtype: u32,
    },
    /// backend와 dtype 조합이 호환되지 않는다 (Sprint D 함정 3: Adreno SOA × F16).
    ///
    /// Adreno SOA backend에서 F16 dtype secondary를 사용할 수 없다.
    /// SOA layout은 Q4_0 전용이며, F16 weights에 대한 SOA 변환이 정의되지 않았다.
    /// F16 secondary가 필요하다면 CPU_AOS 또는 CUDA_AOS backend로 전환하라.
    BackendDtypeIncompatible { backend: String, dtype: String },
    /// 단방향 swap 정합성 위반: primary=Q4_0인데 secondary=F16을 지정했다 (역방향 swap 차단).
    ///
    /// weight swap은 F16→Q4_0 단방향만 지원한다. primary가 이미 Q4_0인 경우
    /// secondary로 F16을 사용하면 역방향(Q4_0→F16)이 되므로 거부된다.
    ReverseSwapRejected {
        primary_dtype: String,
        secondary_dtype: String,
    },
    /// 기타 IO/파싱 에러.
    Io(std::io::Error),
    /// 기타 메시지 에러.
    Other(String),
}

impl fmt::Display for AufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AufError::MagicMismatch => {
                write!(f, "Not an AUF file (magic mismatch)")
            }
            AufError::UnsupportedFormatMajor {
                found,
                max_supported,
            } => write!(
                f,
                "AUF format_major={found} but reader supports up to {max_supported}. Update argus_engine."
            ),
            AufError::UnknownRequiredCapability { bit } => write!(
                f,
                "AUF requires capability bit {bit} which is not understood by this reader"
            ),
            AufError::RequiredSectionMissing { tag } => {
                write!(f, "{tag} section missing — file is not a valid AUF")
            }
            AufError::WeightsSectionMissing { weights_tag } => write!(
                f,
                "AUF does not contain {weights_tag}. Run 'auf-tool repack --add {weights_tag}' to add it from source GGUF."
            ),
            AufError::SectionRangeInvalid { tag, detail } => {
                write!(f, "Section {tag} has invalid offset/size: {detail}")
            }
            AufError::SectionOverlap { tag_a, tag_b } => {
                write!(
                    f,
                    "Section {tag_a} overlaps with {tag_b} (offset/size invalid)"
                )
            }
            AufError::DuplicateSectionTag { tag } => {
                write!(f, "Duplicate section tag: {tag}")
            }
            AufError::ContradictoryFlags { tag } => {
                write!(
                    f,
                    "Section {tag} has contradictory flags REQUIRED+STRIPPABLE"
                )
            }
            AufError::CompressedSectionUnsupported { tag } => write!(
                f,
                "Section {tag} uses compression which is not supported in format_minor=1"
            ),
            AufError::SectionTableTruncated => {
                write!(f, "AUF section table extends beyond file size")
            }
            AufError::FileTooSmall => write!(f, "AUF file is too small to contain a valid header"),
            AufError::TokenizerFormat { detail } => {
                write!(f, "TOKENIZER section format error: {detail}")
            }
            AufError::TensorIndexFormat { detail } => {
                write!(f, "TENSOR_INDEX section format error: {detail}")
            }
            AufError::TensorIndexMissingVariant { weights_tag } => write!(
                f,
                "AUF file invariant violation: TENSOR_INDEX does not list variant '{}' — \
                 empty index detected. Rebuild with 'auf-tool build'.",
                weights_tag
            ),
            AufError::LmHeadEntryMissing => write!(
                f,
                "AUF invariant violation (INV-135): capability_optional bit 2 is set but \
                 TENSOR_INDEX contains no lm_head entry (kind=11, layer_idx=u32::MAX, dtype=Q4_0)"
            ),
            AufError::LmHeadDtypeMismatch { found_dtype } => write!(
                f,
                "AUF invariant violation (INV-135): lm_head entry dtype={found_dtype} \
                 (expected Q4_0=3)"
            ),
            AufError::LmHeadShapeMismatch { expected, found } => write!(
                f,
                "AUF invariant violation (INV-135): lm_head entry shape={found:?} \
                 does not match model config [vocab_size={}, hidden_dim={}]",
                expected[0], expected[1]
            ),
            AufError::LmHeadVariantPayloadMissing { variant_tag } => write!(
                f,
                "AUF invariant violation (INV-135): lm_head entry has no payload \
                 for variant '{variant_tag}'"
            ),
            AufError::DtypeNotAvailable {
                layer_idx,
                kind,
                dtype,
            } => write!(
                f,
                "AUF tensor not available: layer_idx={layer_idx}, kind={kind}, dtype={dtype} \
                 not found in TENSOR_INDEX (ENG-ALG-225)"
            ),
            AufError::BackendDtypeIncompatible { backend, dtype } => write!(
                f,
                "AUF backend/dtype incompatible: backend={backend} does not support dtype={dtype}. \
                 Adreno SOA backend only supports Q4_0 secondary. \
                 Use --backend cpu or switch to a non-SOA AUF variant."
            ),
            AufError::ReverseSwapRejected {
                primary_dtype,
                secondary_dtype,
            } => write!(
                f,
                "AUF reverse swap rejected: primary={primary_dtype}, secondary={secondary_dtype}. \
                 Weight swap only supports F16→Q4_0 direction. \
                 primary=Q4_0 with secondary=F16 would be a reverse (Q4_0→F16) swap which is unsupported."
            ),
            AufError::Io(e) => write!(f, "AUF I/O error: {e}"),
            AufError::Other(msg) => write!(f, "AUF error: {msg}"),
        }
    }
}

impl std::error::Error for AufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let AufError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for AufError {
    fn from(e: std::io::Error) -> Self {
        AufError::Io(e)
    }
}

pub type AufResult<T> = Result<T, AufError>;
