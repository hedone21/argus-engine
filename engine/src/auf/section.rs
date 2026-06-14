/// AUF Section Table (ENG-DAT-096.2/3/4).
///
/// `SectionEntry` 48B 고정 크기. `SectionTable`은 entry 배열 + 조회 헬퍼.
use crate::auf::error::{AufError, AufResult};

/// Section Entry 크기 (바이트).
pub const SECTION_ENTRY_SIZE: usize = 48;

/// Section Entry tag 필드 크기 (바이트).
///
/// spec §3.22.2는 16B로 명시했으나, 카탈로그에 `WEIGHTS_ADRENO_SOA`(18자) 등
/// 16B 초과 태그가 등장한다. 충돌 방지를 위해 tag 필드를 24B로 확장하고
/// `_reserved`(8B → 0B)를 tag 확장에 흡수한다. entry 총 크기 48B 유지.
///
/// Layout (revised):
/// [0..24)  tag: [u8; 24]    (UTF-8 ASCII, 우측 NUL 패딩)
/// [24..32) offset: u64
/// [32..40) size: u64
/// [40..44) flags: u32
/// [44..48) version: u32
pub const SECTION_TAG_SIZE: usize = 24;

/// `SECTION_REQUIRED` flag bit 0: reader가 인식하지 못하면 reject.
pub const SECTION_REQUIRED: u32 = 1 << 0;
/// `SECTION_STRIPPABLE` flag bit 1: `auf-tool strip`이 안전하게 제거 가능.
pub const SECTION_STRIPPABLE: u32 = 1 << 1;
/// `SECTION_COMPRESSED` flag bit 2: payload 압축됨 (v0.1에서는 reserved, 사용 금지).
pub const SECTION_COMPRESSED: u32 = 1 << 2;

// ── Section tag 상수 ──

pub const TAG_META: &str = "META";
pub const TAG_TOKENIZER: &str = "TOKENIZER";
pub const TAG_TENSOR_INDEX: &str = "TENSOR_INDEX";
pub const TAG_WEIGHTS_ADRENO_SOA: &str = "WEIGHTS_ADRENO_SOA";
pub const TAG_WEIGHTS_CUDA_AOS: &str = "WEIGHTS_CUDA_AOS";
pub const TAG_WEIGHTS_CPU_AOS: &str = "WEIGHTS_CPU_AOS";

/// Section Table Entry (48B, little-endian).
///
/// ENG-DAT-096.2 spec 구현.
///
/// 실제 layout (tag 24B 확장, _reserved 흡수):
/// ```text
/// [0..24)  tag: [u8; 24]    (UTF-8 ASCII, 우측 NUL 패딩)
/// [24..32) offset: u64
/// [32..40) size: u64
/// [40..44) flags: u32
/// [44..48) version: u32
/// ```
#[derive(Debug, Clone)]
pub struct SectionEntry {
    /// UTF-8 ASCII section 식별자, 우측 NUL 패딩 (24B, `SECTION_TAG_SIZE`).
    pub tag_raw: [u8; SECTION_TAG_SIZE],
    /// 파일 시작 기준 payload 시작 byte offset.
    pub offset: u64,
    /// payload 바이트 크기.
    pub size: u64,
    /// section flag bit set (ENG-DAT-096.3).
    pub flags: u32,
    /// section 자체 버전.
    pub version: u32,
}

impl SectionEntry {
    /// 48B 바이트열에서 단일 entry를 파싱한다.
    pub fn from_bytes(b: &[u8]) -> AufResult<Self> {
        if b.len() < SECTION_ENTRY_SIZE {
            return Err(AufError::SectionTableTruncated);
        }
        let tag_raw: [u8; SECTION_TAG_SIZE] = b[0..SECTION_TAG_SIZE].try_into().unwrap();
        let offset = u64::from_le_bytes(b[24..32].try_into().unwrap());
        let size = u64::from_le_bytes(b[32..40].try_into().unwrap());
        let flags = u32::from_le_bytes(b[40..44].try_into().unwrap());
        let version = u32::from_le_bytes(b[44..48].try_into().unwrap());
        Ok(SectionEntry {
            tag_raw,
            offset,
            size,
            flags,
            version,
        })
    }

    /// entry를 48B little-endian 바이트열로 직렬화한다.
    pub fn to_bytes(&self) -> [u8; SECTION_ENTRY_SIZE] {
        let mut b = [0u8; SECTION_ENTRY_SIZE];
        b[0..SECTION_TAG_SIZE].copy_from_slice(&self.tag_raw);
        b[24..32].copy_from_slice(&self.offset.to_le_bytes());
        b[32..40].copy_from_slice(&self.size.to_le_bytes());
        b[40..44].copy_from_slice(&self.flags.to_le_bytes());
        b[44..48].copy_from_slice(&self.version.to_le_bytes());
        b
    }

    /// tag를 UTF-8 문자열로 반환한다 (NUL 트리밍).
    pub fn tag(&self) -> &str {
        let end = self
            .tag_raw
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(SECTION_TAG_SIZE);
        std::str::from_utf8(&self.tag_raw[..end]).unwrap_or("")
    }

    /// SECTION_REQUIRED 비트 확인.
    pub fn is_required(&self) -> bool {
        self.flags & SECTION_REQUIRED != 0
    }

    /// SECTION_STRIPPABLE 비트 확인.
    pub fn is_strippable(&self) -> bool {
        self.flags & SECTION_STRIPPABLE != 0
    }

    /// SECTION_COMPRESSED 비트 확인.
    pub fn is_compressed(&self) -> bool {
        self.flags & SECTION_COMPRESSED != 0
    }

    /// tag 문자열로부터 `SectionEntry`를 생성한다 (writer 용도).
    ///
    /// tag는 최대 `SECTION_TAG_SIZE` (24B) ASCII여야 한다.
    pub fn new(tag: &str, offset: u64, size: u64, flags: u32, version: u32) -> AufResult<Self> {
        let tag_bytes = tag.as_bytes();
        if tag_bytes.len() > SECTION_TAG_SIZE {
            return Err(AufError::Other(format!(
                "Section tag too long (max {SECTION_TAG_SIZE}B): {tag}"
            )));
        }
        let mut tag_raw = [0u8; SECTION_TAG_SIZE];
        tag_raw[..tag_bytes.len()].copy_from_slice(tag_bytes);
        Ok(SectionEntry {
            tag_raw,
            offset,
            size,
            flags,
            version,
        })
    }
}

/// AUF Section Table — section entry 배열 + 조회 헬퍼.
#[derive(Debug, Clone)]
pub struct SectionTable {
    pub entries: Vec<SectionEntry>,
}

impl SectionTable {
    /// section table 바이트 영역에서 `count`개 entry를 파싱한다.
    pub fn from_bytes(bytes: &[u8], count: u32) -> AufResult<Self> {
        let count = count as usize;
        let required = count * SECTION_ENTRY_SIZE;
        if bytes.len() < required {
            return Err(AufError::SectionTableTruncated);
        }
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * SECTION_ENTRY_SIZE;
            let entry = SectionEntry::from_bytes(&bytes[start..start + SECTION_ENTRY_SIZE])?;
            entries.push(entry);
        }
        Ok(SectionTable { entries })
    }

    /// section table을 바이트 벡터로 직렬화한다.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.entries.len() * SECTION_ENTRY_SIZE);
        for entry in &self.entries {
            out.extend_from_slice(&entry.to_bytes());
        }
        out
    }

    /// 직렬화 크기 (바이트).
    pub fn serialized_size(&self) -> usize {
        self.entries.len() * SECTION_ENTRY_SIZE
    }

    /// tag 이름으로 entry를 조회한다.
    pub fn find(&self, tag: &str) -> Option<&SectionEntry> {
        self.entries.iter().find(|e| e.tag() == tag)
    }

    /// flag 무결성 검증: REQUIRED + STRIPPABLE 동시 set 금지 (ENG-ALG-223 §3.12.17.6).
    pub fn validate_flags(&self) -> AufResult<()> {
        for e in &self.entries {
            if e.is_required() && e.is_strippable() {
                return Err(AufError::ContradictoryFlags {
                    tag: e.tag().to_owned(),
                });
            }
        }
        Ok(())
    }

    /// v0.1 reader: SECTION_COMPRESSED section 금지 검증.
    pub fn validate_no_compressed(&self) -> AufResult<()> {
        for e in &self.entries {
            if e.is_compressed() {
                return Err(AufError::CompressedSectionUnsupported {
                    tag: e.tag().to_owned(),
                });
            }
        }
        Ok(())
    }

    /// 중복 tag 금지 검증 (INV-134).
    pub fn validate_unique_tags(&self) -> AufResult<()> {
        let mut seen = std::collections::HashSet::new();
        for e in &self.entries {
            let tag = e.tag().to_owned();
            if !seen.insert(tag.clone()) {
                return Err(AufError::DuplicateSectionTag { tag });
            }
        }
        Ok(())
    }

    /// section offset/size 무결성 검증 (INV-134).
    ///
    /// - 각 section은 `[payload_start_offset, file_size)` 범위 내.
    /// - 어떤 두 section도 byte range overlap 금지.
    pub fn validate_ranges(&self, payload_start_offset: u64, file_size: u64) -> AufResult<()> {
        for e in &self.entries {
            if e.offset < payload_start_offset {
                return Err(AufError::SectionRangeInvalid {
                    tag: e.tag().to_owned(),
                    detail: format!(
                        "offset={} < payload_start_offset={}",
                        e.offset, payload_start_offset
                    ),
                });
            }
            let end =
                e.offset
                    .checked_add(e.size)
                    .ok_or_else(|| AufError::SectionRangeInvalid {
                        tag: e.tag().to_owned(),
                        detail: "offset+size overflow".to_owned(),
                    })?;
            if end > file_size {
                return Err(AufError::SectionRangeInvalid {
                    tag: e.tag().to_owned(),
                    detail: format!("offset+size={end} > file_size={file_size}"),
                });
            }
        }

        // overlap 검사 O(n^2) — section 수가 6 미만이므로 허용
        for i in 0..self.entries.len() {
            for j in (i + 1)..self.entries.len() {
                let a = &self.entries[i];
                let b = &self.entries[j];
                let a_end = a.offset + a.size;
                let b_end = b.offset + b.size;
                if a.offset < b_end && b.offset < a_end {
                    return Err(AufError::SectionOverlap {
                        tag_a: a.tag().to_owned(),
                        tag_b: b.tag().to_owned(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(tag: &str, offset: u64, size: u64, flags: u32) -> SectionEntry {
        SectionEntry::new(tag, offset, size, flags, 1).unwrap()
    }

    #[test]
    fn entry_round_trip() {
        let e = make_entry("META", 65536, 100, SECTION_REQUIRED);
        let bytes = e.to_bytes();
        assert_eq!(bytes.len(), 48);
        let e2 = SectionEntry::from_bytes(&bytes).unwrap();
        assert_eq!(e2.tag(), "META");
        assert_eq!(e2.offset, 65536);
        assert_eq!(e2.size, 100);
        assert_eq!(e2.flags, SECTION_REQUIRED);
        assert_eq!(e2.version, 1);
    }

    #[test]
    fn tag_nul_trimming() {
        let e = make_entry("TENSOR_INDEX", 0, 0, 0);
        assert_eq!(e.tag(), "TENSOR_INDEX");
    }

    #[test]
    fn tag_too_long() {
        // SECTION_TAG_SIZE=24, 25자 태그는 에러
        let result = SectionEntry::new("WEIGHTS_ADRENO_SOA_EXTRA_X", 0, 0, 0, 1);
        assert!(result.is_err());
        // 24자 태그는 OK (WEIGHTS_ADRENO_SOA = 18자, OK)
        let result2 = SectionEntry::new("WEIGHTS_ADRENO_SOA", 0, 0, 0, 1);
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap().tag(), "WEIGHTS_ADRENO_SOA");
    }

    #[test]
    fn table_round_trip() {
        let entries = vec![
            make_entry("META", 65536, 100, SECTION_REQUIRED),
            make_entry("TOKENIZER", 65636, 200, SECTION_REQUIRED),
        ];
        let table = SectionTable { entries };
        let bytes = table.to_bytes();
        let table2 = SectionTable::from_bytes(&bytes, 2).unwrap();
        assert_eq!(table2.entries.len(), 2);
        assert_eq!(table2.entries[0].tag(), "META");
        assert_eq!(table2.entries[1].tag(), "TOKENIZER");
    }

    #[test]
    fn validate_contradictory_flags() {
        let e = make_entry("META", 65536, 100, SECTION_REQUIRED | SECTION_STRIPPABLE);
        let table = SectionTable { entries: vec![e] };
        let err = table.validate_flags().unwrap_err();
        assert!(matches!(err, AufError::ContradictoryFlags { .. }));
    }

    #[test]
    fn validate_duplicate_tags() {
        let entries = vec![
            make_entry("META", 65536, 100, SECTION_REQUIRED),
            make_entry("META", 65636, 200, SECTION_REQUIRED),
        ];
        let table = SectionTable { entries };
        let err = table.validate_unique_tags().unwrap_err();
        assert!(matches!(err, AufError::DuplicateSectionTag { .. }));
    }

    #[test]
    fn validate_ranges_ok() {
        let entries = vec![
            make_entry("META", 65536, 100, SECTION_REQUIRED),
            make_entry("TOKENIZER", 65700, 200, SECTION_REQUIRED),
        ];
        let table = SectionTable { entries };
        assert!(table.validate_ranges(65536, 1_000_000).is_ok());
    }

    #[test]
    fn validate_ranges_below_payload_start() {
        let entries = vec![make_entry("META", 100, 100, SECTION_REQUIRED)];
        let table = SectionTable { entries };
        let err = table.validate_ranges(65536, 1_000_000).unwrap_err();
        assert!(matches!(err, AufError::SectionRangeInvalid { .. }));
    }

    #[test]
    fn validate_ranges_overflow_file() {
        let entries = vec![make_entry("META", 65536, 1_000_000, SECTION_REQUIRED)];
        let table = SectionTable { entries };
        let err = table.validate_ranges(65536, 100_000).unwrap_err();
        assert!(matches!(err, AufError::SectionRangeInvalid { .. }));
    }

    #[test]
    fn validate_ranges_overlap() {
        let entries = vec![
            make_entry("META", 65536, 500, SECTION_REQUIRED),
            make_entry("TOKENIZER", 65800, 500, SECTION_REQUIRED),
        ];
        let table = SectionTable { entries };
        let err = table.validate_ranges(65536, 1_000_000).unwrap_err();
        assert!(matches!(err, AufError::SectionOverlap { .. }));
    }

    #[test]
    fn validate_no_compressed_passes() {
        let entries = vec![make_entry("META", 65536, 100, SECTION_REQUIRED)];
        let table = SectionTable { entries };
        assert!(table.validate_no_compressed().is_ok());
    }

    #[test]
    fn validate_compressed_rejects() {
        let entries = vec![make_entry(
            "META",
            65536,
            100,
            SECTION_REQUIRED | SECTION_COMPRESSED,
        )];
        let table = SectionTable { entries };
        let err = table.validate_no_compressed().unwrap_err();
        assert!(matches!(err, AufError::CompressedSectionUnsupported { .. }));
    }
}
