/// AUF 파일 헤더 (256B 고정, little-endian).
///
/// ENG-DAT-096.1 spec 구현.
use crate::auf::error::{AufError, AufResult};

/// reader가 지원하는 최대 format_major.
pub const READER_MAX_FORMAT_MAJOR: u16 = 0;

/// capability_optional bit 2: AUF에 lm_head Q4_0 사전 변환 entry가 포함됨 (v0.1.1+).
///
/// bit 2 = 1이면 TENSOR_INDEX에 `kind=11(lm_head), layer_idx=u32::MAX, dtype=Q4_0` 엔트리
/// 가 정확히 1개 존재해야 한다 (INV-135). 없거나 부정합이면 reader는 `Result::Err`를 반환.
/// bit 2 = 0 (또는 AUF 부재)이면 lm_head Q4_0 accessor는 `Ok(None)`을 반환 (INV-136).
pub const CAPABILITY_BIT_LM_HEAD_Q4_0: u64 = 1 << 2;

/// capability_optional bit 3: AUF v0.2 multi-dtype variant (INV-139, ENG-DAT-097).
///
/// bit 3 = 1이면:
///   - 동일 (`layer_idx`, `kind`) 쌍에 dtype이 다른 entry가 2개 이상 존재함.
///   - META JSON에 `default_dtype` 필드가 의무적으로 존재해야 한다 (INV-138).
///   - format_minor >= 2 이어야 한다 (INV-139).
///   - v0.1.x reader는 이 bit를 인식하지 못하지만 `capability_optional`이므로 reject 사유가 아니다.
///     first-match 규칙으로 default_dtype entry를 안전하게 로드한다 (INV-138 writer 정렬 보장).
///
/// bit 3 = 0이면 single-dtype 모드 (각 (`layer_idx`, `kind`)에 entry 1개씩).
pub const CAPABILITY_BIT_MULTI_DTYPE: u64 = 1 << 3;

/// reader가 인식하는 capability bit set.
///
/// v0.1.0: 모두 0 (READER_KNOWN_CAPABILITIES = 0).
/// v0.1.1: CAPABILITY_BIT_LM_HEAD_Q4_0 (bit 2) 추가 — optional capability이므로
///   capability_optional에만 set, capability_required에는 set하지 않는다.
///   따라서 구 reader (bit 2 미인식)는 lm_head accessor를 None으로 처리하고 안전하게 로드.
/// v0.2.0: CAPABILITY_BIT_MULTI_DTYPE (bit 3) 추가 — optional capability.
///   capability_optional에만 set. v0.1.x reader는 무시하고 first-match로 fallback.
pub const READER_KNOWN_CAPABILITIES: u64 = CAPABILITY_BIT_LM_HEAD_Q4_0 | CAPABILITY_BIT_MULTI_DTYPE;

/// AUF 헤더 magic bytes: `"ARGUS_W\0"`.
pub const AUF_MAGIC: &[u8; 8] = b"ARGUS_W\0";

/// 헤더 전체 크기 (고정).
pub const HEADER_SIZE: usize = 256;

/// AUF 파일 헤더.
///
/// 파일 오프셋 0~255에 고정 배치된다. 모든 multi-byte 필드는 little-endian.
#[derive(Debug, Clone)]
pub struct AufHeader {
    /// `"ARGUS_W\0"` (8B).
    pub magic: [u8; 8],
    /// breaking change 시 증가 (v0.1 = 0).
    pub format_major: u16,
    /// additive 변경 시 증가 (v0.1 = 1).
    pub format_minor: u16,
    /// reserved (v0.1 = 0).
    pub format_patch: u16,
    // _pad0: u16 — serialization 전용, 구조체 내부에서는 생략
    /// UTF-8 생성 도구 이름, 우측 NUL 패딩 (32B). 예: `"argus_engine v0.4.0"`.
    pub created_by: [u8; 32],
    /// 원본 GGUF hybrid source hash (32B). §3.22.6.
    pub source_hash: [u8; 32],
    /// 원본 GGUF 파일 바이트 크기.
    pub source_size: u64,
    /// 원본 GGUF mtime (Unix epoch seconds).
    pub source_mtime: u64,
    /// reader가 모르면 reject할 capability bit set.
    pub capability_required: u64,
    /// reader가 모르면 skip해도 안전한 capability bit set.
    pub capability_optional: u64,
    /// section table 엔트리 수.
    pub section_count: u32,
    // _pad1: u32 — serialization 전용
    /// section table 시작 byte offset.
    pub section_table_offset: u64,
    /// 모든 section payload는 이 offset 이상.
    pub payload_start_offset: u64,
    // _reserved: [u8; 120] — serialization 전용
}

impl AufHeader {
    /// 256B 직렬화된 바이트열에서 헤더를 파싱한다.
    ///
    /// magic, format_major, capability_required 검증은 [`validate`] 를 별도 호출해야 한다.
    pub fn from_bytes(bytes: &[u8]) -> AufResult<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(AufError::FileTooSmall);
        }
        let b = &bytes[..HEADER_SIZE];

        let magic: [u8; 8] = b[0..8].try_into().unwrap();
        let format_major = u16::from_le_bytes(b[8..10].try_into().unwrap());
        let format_minor = u16::from_le_bytes(b[10..12].try_into().unwrap());
        let format_patch = u16::from_le_bytes(b[12..14].try_into().unwrap());
        // _pad0 = b[14..16]
        let created_by: [u8; 32] = b[16..48].try_into().unwrap();
        let source_hash: [u8; 32] = b[48..80].try_into().unwrap();
        let source_size = u64::from_le_bytes(b[80..88].try_into().unwrap());
        let source_mtime = u64::from_le_bytes(b[88..96].try_into().unwrap());
        let capability_required = u64::from_le_bytes(b[96..104].try_into().unwrap());
        let capability_optional = u64::from_le_bytes(b[104..112].try_into().unwrap());
        let section_count = u32::from_le_bytes(b[112..116].try_into().unwrap());
        // _pad1 = b[116..120]
        let section_table_offset = u64::from_le_bytes(b[120..128].try_into().unwrap());
        let payload_start_offset = u64::from_le_bytes(b[128..136].try_into().unwrap());
        // _reserved = b[136..256]

        Ok(AufHeader {
            magic,
            format_major,
            format_minor,
            format_patch,
            created_by,
            source_hash,
            source_size,
            source_mtime,
            capability_required,
            capability_optional,
            section_count,
            section_table_offset,
            payload_start_offset,
        })
    }

    /// 헤더를 256B little-endian 바이트열로 직렬화한다.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0..8].copy_from_slice(&self.magic);
        b[8..10].copy_from_slice(&self.format_major.to_le_bytes());
        b[10..12].copy_from_slice(&self.format_minor.to_le_bytes());
        b[12..14].copy_from_slice(&self.format_patch.to_le_bytes());
        // _pad0 = [14..16] = 0
        b[16..48].copy_from_slice(&self.created_by);
        b[48..80].copy_from_slice(&self.source_hash);
        b[80..88].copy_from_slice(&self.source_size.to_le_bytes());
        b[88..96].copy_from_slice(&self.source_mtime.to_le_bytes());
        b[96..104].copy_from_slice(&self.capability_required.to_le_bytes());
        b[104..112].copy_from_slice(&self.capability_optional.to_le_bytes());
        b[112..116].copy_from_slice(&self.section_count.to_le_bytes());
        // _pad1 = [116..120] = 0
        b[120..128].copy_from_slice(&self.section_table_offset.to_le_bytes());
        b[128..136].copy_from_slice(&self.payload_start_offset.to_le_bytes());
        // _reserved = [136..256] = 0
        b
    }

    /// 파일 식별 및 format_major, capability_required 검증 (INV-132).
    ///
    /// - magic 불일치 → [`AufError::MagicMismatch`]
    /// - format_major > READER_MAX_FORMAT_MAJOR → [`AufError::UnsupportedFormatMajor`]
    /// - capability_required에 알 수 없는 bit → [`AufError::UnknownRequiredCapability`]
    pub fn validate(&self) -> AufResult<()> {
        if &self.magic != AUF_MAGIC {
            return Err(AufError::MagicMismatch);
        }
        if self.format_major > READER_MAX_FORMAT_MAJOR {
            return Err(AufError::UnsupportedFormatMajor {
                found: self.format_major,
                max_supported: READER_MAX_FORMAT_MAJOR,
            });
        }
        let unknown_required = self.capability_required & !READER_KNOWN_CAPABILITIES;
        if unknown_required != 0 {
            // 가장 낮은 미인식 bit를 보고
            let bit = unknown_required.trailing_zeros();
            return Err(AufError::UnknownRequiredCapability { bit });
        }
        Ok(())
    }

    /// capability_optional bit 2 = 1인지 확인한다.
    ///
    /// `true`이면 이 AUF에 lm_head Q4_0 사전 변환 entry가 포함되어 있다 (INV-135).
    /// `false`이면 lm_head accessor는 `Ok(None)`을 반환해야 한다 (INV-136).
    pub fn has_lm_head_q4_0(&self) -> bool {
        self.capability_optional & CAPABILITY_BIT_LM_HEAD_Q4_0 != 0
    }

    /// capability_optional bit 3 = 1인지 확인한다 (INV-139, ENG-DAT-097).
    ///
    /// `true`이면 이 AUF v0.2 AUF이며 동일 (`layer_idx`, `kind`)에 dtype이 다른 entry가
    /// 존재한다. META JSON에 `default_dtype` 필드가 의무 존재 (INV-138).
    /// `false`이면 single-dtype 모드 (v0.1.x 호환).
    pub fn has_multi_dtype(&self) -> bool {
        self.capability_optional & CAPABILITY_BIT_MULTI_DTYPE != 0
    }

    /// `LM_HEAD_PRECOMPUTED_Q4_0` capability bit를 설정/해제한다 (writer용).
    ///
    /// `enabled = true`: bit 2 set + format_patch = 1 (v0.1.1).
    /// `enabled = false`: bit 2 clear + format_patch = 0 (v0.1.0 호환).
    /// 다른 capability bit는 보존한다.
    pub fn set_lm_head_q4_0_capability(&mut self, enabled: bool) {
        if enabled {
            self.capability_optional |= CAPABILITY_BIT_LM_HEAD_Q4_0;
            self.format_patch = 1;
        } else {
            self.capability_optional &= !CAPABILITY_BIT_LM_HEAD_Q4_0;
            self.format_patch = 0;
        }
    }

    /// `MULTI_DTYPE_VARIANTS` capability bit를 설정/해제한다 (writer용, INV-139).
    ///
    /// `enabled = true`: bit 3 set + format_minor = 2 (v0.2).
    /// `enabled = false`: bit 3 clear. format_minor는 호출자가 별도 관리.
    /// 다른 capability bit는 보존한다.
    pub fn set_multi_dtype_capability(&mut self, enabled: bool) {
        if enabled {
            self.capability_optional |= CAPABILITY_BIT_MULTI_DTYPE;
            self.format_minor = 2;
        } else {
            self.capability_optional &= !CAPABILITY_BIT_MULTI_DTYPE;
        }
    }

    /// v0.1 기본 헤더를 생성한다 (writer 용도).
    pub fn new_v01(
        created_by: &str,
        source_hash: [u8; 32],
        source_size: u64,
        source_mtime: u64,
        section_count: u32,
        section_table_offset: u64,
        payload_start_offset: u64,
    ) -> Self {
        let mut cb = [0u8; 32];
        let bytes = created_by.as_bytes();
        let len = bytes.len().min(32);
        cb[..len].copy_from_slice(&bytes[..len]);

        AufHeader {
            magic: *AUF_MAGIC,
            format_major: 0,
            format_minor: 1,
            format_patch: 0,
            created_by: cb,
            source_hash,
            source_size,
            source_mtime,
            capability_required: 0,
            capability_optional: 0,
            section_count,
            section_table_offset,
            payload_start_offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header() -> AufHeader {
        AufHeader::new_v01("argus_engine test", [0u8; 32], 1024, 1000000, 3, 256, 65536)
    }

    #[test]
    fn round_trip_bytes() {
        let h = make_header();
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), 256);
        let h2 = AufHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h2.magic, *AUF_MAGIC);
        assert_eq!(h2.format_major, 0);
        assert_eq!(h2.format_minor, 1);
        assert_eq!(h2.format_patch, 0);
        assert_eq!(h2.section_count, 3);
        assert_eq!(h2.section_table_offset, 256);
        assert_eq!(h2.payload_start_offset, 65536);
    }

    #[test]
    fn validate_ok() {
        let h = make_header();
        assert!(h.validate().is_ok());
    }

    #[test]
    fn validate_magic_mismatch() {
        let mut h = make_header();
        h.magic = *b"BADMAGIC";
        let err = h.validate().unwrap_err();
        assert!(matches!(err, AufError::MagicMismatch));
    }

    #[test]
    fn validate_unsupported_format_major() {
        let mut h = make_header();
        h.format_major = 1;
        let err = h.validate().unwrap_err();
        assert!(matches!(
            err,
            AufError::UnsupportedFormatMajor {
                found: 1,
                max_supported: 0
            }
        ));
    }

    #[test]
    fn validate_unknown_required_capability() {
        let mut h = make_header();
        h.capability_required = 1; // bit 0 unknown
        let err = h.validate().unwrap_err();
        assert!(matches!(
            err,
            AufError::UnknownRequiredCapability { bit: 0 }
        ));
    }

    #[test]
    fn file_too_small() {
        let bytes = [0u8; 10];
        let err = AufHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AufError::FileTooSmall));
    }

    #[test]
    fn created_by_truncated_to_32() {
        // 32B 초과 문자열은 앞 32B만 저장
        let long_str = "a".repeat(100);
        let h = AufHeader::new_v01(&long_str, [0u8; 32], 0, 0, 0, 256, 65536);
        assert_eq!(&h.created_by, b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    }

    /// B-2: CAPABILITY_BIT_MULTI_DTYPE 상수 값 검증.
    #[test]
    fn capability_bit_multi_dtype_is_bit3() {
        assert_eq!(CAPABILITY_BIT_MULTI_DTYPE, 1 << 3);
        assert_eq!(CAPABILITY_BIT_MULTI_DTYPE, 8u64);
    }

    /// B-2: READER_KNOWN_CAPABILITIES에 bit 2와 bit 3이 포함됨.
    #[test]
    fn reader_known_capabilities_includes_bit2_and_bit3() {
        const { assert!(READER_KNOWN_CAPABILITIES & CAPABILITY_BIT_LM_HEAD_Q4_0 != 0) };
        const { assert!(READER_KNOWN_CAPABILITIES & CAPABILITY_BIT_MULTI_DTYPE != 0) };
        assert_eq!(
            READER_KNOWN_CAPABILITIES,
            (1 << 2) | (1 << 3),
            "READER_KNOWN_CAPABILITIES must be exactly bit2|bit3"
        );
    }

    /// B-2: has_multi_dtype() accessor.
    #[test]
    fn has_multi_dtype_accessor() {
        let mut h = make_header();
        assert!(!h.has_multi_dtype(), "bit 3 initially clear");
        h.capability_optional |= CAPABILITY_BIT_MULTI_DTYPE;
        assert!(h.has_multi_dtype(), "bit 3 set after OR");
    }

    /// B-2: set_multi_dtype_capability() writer helper — bit 3 set + format_minor = 2.
    #[test]
    fn set_multi_dtype_capability_sets_format_minor_2() {
        let mut h = make_header();
        assert_eq!(h.format_minor, 1);
        h.set_multi_dtype_capability(true);
        assert!(h.has_multi_dtype());
        assert_eq!(
            h.format_minor, 2,
            "format_minor must be 2 when multi-dtype enabled"
        );
    }

    /// B-2: set_multi_dtype_capability(false) — bit 3 clear, format_minor unchanged.
    #[test]
    fn set_multi_dtype_capability_clear() {
        let mut h = make_header();
        h.set_multi_dtype_capability(true);
        h.set_multi_dtype_capability(false);
        assert!(!h.has_multi_dtype());
    }

    /// B-2: format_minor 1 (v0.1.x) AUF는 validate() 통과.
    #[test]
    fn format_minor_1_is_valid() {
        let mut h = make_header();
        h.format_minor = 1;
        assert!(h.validate().is_ok());
    }

    /// B-2: format_minor 2 (v0.2) AUF는 validate() 통과.
    #[test]
    fn format_minor_2_is_valid() {
        let mut h = make_header();
        h.set_multi_dtype_capability(true);
        assert_eq!(h.format_minor, 2);
        assert!(h.validate().is_ok());
    }

    /// B-2: bit 3 set + capability_required = 0 (optional) → validate PASS.
    ///
    /// bit 3는 capability_optional에만 set. READER_KNOWN_CAPABILITIES에 포함되므로
    /// capability_required에 있어도 reject하지 않는다.
    #[test]
    fn bit3_optional_not_required() {
        let mut h = make_header();
        h.capability_optional |= CAPABILITY_BIT_MULTI_DTYPE;
        h.capability_required = 0; // optional only
        assert!(
            h.validate().is_ok(),
            "bit 3 in optional must not cause rejection"
        );
    }

    /// B-2: capability_required bit 3가 set되어 있어도, READER_KNOWN_CAPABILITIES에 포함되므로 통과.
    ///
    /// READER_KNOWN_CAPABILITIES에 bit 3가 추가되었으므로 required에 set해도 reject하지 않는다.
    #[test]
    fn bit3_in_required_passes_because_known() {
        let mut h = make_header();
        h.capability_required = CAPABILITY_BIT_MULTI_DTYPE;
        // bit 3는 이제 READER_KNOWN_CAPABILITIES에 포함되므로 알 수 없는 bit가 아님 → PASS
        assert!(h.validate().is_ok());
    }

    /// B-2: round-trip with format_minor=2 + multi_dtype bit.
    #[test]
    fn round_trip_with_multi_dtype_capability() {
        let mut h = make_header();
        h.set_multi_dtype_capability(true);
        h.set_lm_head_q4_0_capability(true);
        let bytes = h.to_bytes();
        let h2 = AufHeader::from_bytes(&bytes).unwrap();
        assert!(h2.has_multi_dtype());
        assert!(h2.has_lm_head_q4_0());
        assert_eq!(h2.format_minor, 2);
        assert_eq!(
            h2.capability_optional,
            CAPABILITY_BIT_LM_HEAD_Q4_0 | CAPABILITY_BIT_MULTI_DTYPE
        );
    }
}
