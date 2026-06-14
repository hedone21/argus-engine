/// AUF Stripper — dead variant 제거 (ENG-ALG-223 §3.12.17.3).
///
/// Strip 의미:
/// - required section은 절대 제거 불가.
/// - strippable section만 제거 가능.
/// - 단순 truncation 금지 — rewrite (cursor 기반 새 파일 생성).
/// - atomic rename (POSIX rename).
/// - source_hash/created_by는 원본 헤더에서 보존.
use std::path::Path;

use crate::auf::error::{AufError, AufResult};
use crate::auf::header::{AufHeader, HEADER_SIZE};
use crate::auf::reader::{BackendTag, open};
use crate::auf::section::SectionTable;
use crate::auf::writer::{WEIGHTS_ALIGNMENT, align_up};

/// AUF 파일에서 지정된 tag만 유지하고 나머지를 제거한다.
///
/// - `keep_tags`: 유지할 section tag 목록. required section은 자동 포함.
/// - `no_backup`: true이면 `.auf.bak` 백업 파일을 생성하지 않는다.
///
/// 처리 절차:
/// 1. 기존 AUF 읽기 (`BackendTag::Any`).
/// 2. keep_tags 검증 (required section 제거 시도 → 에러).
/// 3. 백업 (선택).
/// 4. 새 AUF rewrite (원본 헤더 source_hash/created_by 보존).
/// 5. atomic rename.
pub fn strip(in_path: &Path, keep_tags: &[&str], no_backup: bool) -> AufResult<()> {
    let view = open(in_path, BackendTag::Any)?;
    let keep_set: std::collections::HashSet<&str> = keep_tags.iter().copied().collect();

    // keep할 sections 결정
    let mut sections_to_keep = Vec::new();
    for entry in &view.section_table.entries {
        let tag = entry.tag();
        if keep_set.contains(tag) {
            sections_to_keep.push(entry.clone());
        } else if entry.is_required() {
            // required section은 keep_tags에 없어도 자동 보존
            sections_to_keep.push(entry.clone());
        } else if !entry.is_strippable() {
            // non-strippable, non-required section을 제거하려 하면 에러
            return Err(AufError::Other(format!(
                "Section '{tag}' is not strippable and was not in keep_tags"
            )));
        }
        // is_strippable이고 keep_set에 없으면 제거 (삽입 안 함)
    }

    // 백업
    if !no_backup {
        let bak_path = in_path.with_extension("auf.bak");
        std::fs::copy(in_path, &bak_path).map_err(AufError::Io)?;
    }

    // 새 AUF 파일 빌드 (원본 bytes + 헤더에서 source_hash/created_by 보존)
    let new_bytes = rewrite_sections_into_bytes(view.raw_bytes(), &view.header, &sections_to_keep)?;

    // atomic rename
    let tmp_path = {
        let parent = in_path.parent().unwrap_or_else(|| Path::new("."));
        parent.join(format!(
            ".auf_strip_{}.auf.tmp",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    };
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp_path).map_err(AufError::Io)?;
        f.write_all(&new_bytes).map_err(AufError::Io)?;
        f.sync_all().map_err(AufError::Io)?;
    }
    std::fs::rename(&tmp_path, in_path).map_err(AufError::Io)?;
    Ok(())
}

/// rewrite_with_sections의 실제 구현 — raw 파일 bytes를 추가 인자로 받는다.
pub fn rewrite_sections_into_bytes(
    original_bytes: &[u8],
    header: &AufHeader,
    keep_entries: &[crate::auf::section::SectionEntry],
) -> AufResult<Vec<u8>> {
    use crate::auf::section::SectionEntry;

    let section_count = keep_entries.len();
    let section_table_offset = HEADER_SIZE as u64;
    let section_table_size = section_count as u64 * 48;
    let after_table = section_table_offset + section_table_size;
    let payload_start_offset = align_up(after_table, WEIGHTS_ALIGNMENT);

    let mut cursor = payload_start_offset;
    let mut new_entries: Vec<SectionEntry> = Vec::with_capacity(section_count);
    let mut copy_plan: Vec<(u64, u64, u64)> = Vec::new();

    for old_entry in keep_entries {
        let new_offset = cursor;
        let alignment = if old_entry.tag().starts_with("WEIGHTS_") {
            WEIGHTS_ALIGNMENT
        } else {
            8u64
        };
        new_entries.push(SectionEntry::new(
            old_entry.tag(),
            new_offset,
            old_entry.size,
            old_entry.flags,
            old_entry.version,
        )?);
        copy_plan.push((new_offset, old_entry.offset, old_entry.size));
        cursor = align_up(new_offset + old_entry.size, alignment);
    }

    let total_size = cursor;
    let mut out = vec![0u8; total_size as usize];

    // 헤더 (source_hash/created_by 보존)
    let created_by_str = {
        let end = header.created_by.iter().position(|&b| b == 0).unwrap_or(32);
        std::str::from_utf8(&header.created_by[..end])
            .unwrap_or("")
            .to_owned()
    };
    let new_header = AufHeader::new_v01(
        &created_by_str,
        header.source_hash,
        header.source_size,
        header.source_mtime,
        section_count as u32,
        section_table_offset,
        payload_start_offset,
    );
    out[..HEADER_SIZE].copy_from_slice(&new_header.to_bytes());

    // section table
    let table = SectionTable {
        entries: new_entries,
    };
    let table_bytes = table.to_bytes();
    out[HEADER_SIZE..HEADER_SIZE + table_bytes.len()].copy_from_slice(&table_bytes);

    // payload 복사
    for (new_off, old_off, size) in copy_plan {
        let src_start = old_off as usize;
        let src_end = src_start + size as usize;
        let dst_start = new_off as usize;
        let dst_end = dst_start + size as usize;
        if src_end > original_bytes.len() || dst_end > out.len() {
            return Err(AufError::SectionRangeInvalid {
                tag: "strip copy".to_owned(),
                detail: format!(
                    "src {src_start}..{src_end} or dst {dst_start}..{dst_end} out of bounds"
                ),
            });
        }
        out[dst_start..dst_end].copy_from_slice(&original_bytes[src_start..src_end]);
    }

    Ok(out)
}

/// AUF 파일의 특정 section tag만 유지하는 public API (bytes 기반, 테스트 친화).
///
/// `keep_tags`에 포함되지 않고 strippable인 section을 제거한다.
/// required section은 자동 보존.
pub fn strip_bytes(original: &[u8], keep_tags: &[&str]) -> AufResult<Vec<u8>> {
    use crate::auf::header::AufHeader;
    use crate::auf::section::SectionTable;

    if original.len() < HEADER_SIZE {
        return Err(AufError::FileTooSmall);
    }
    let header = AufHeader::from_bytes(original)?;
    header.validate()?;

    let section_table_end =
        header.section_table_offset as usize + header.section_count as usize * 48;
    if section_table_end > original.len() {
        return Err(AufError::SectionTableTruncated);
    }
    let section_table = SectionTable::from_bytes(
        &original[header.section_table_offset as usize..],
        header.section_count,
    )?;
    section_table.validate_unique_tags()?;
    section_table.validate_flags()?;
    section_table.validate_ranges(header.payload_start_offset, original.len() as u64)?;

    let keep_set: std::collections::HashSet<&str> = keep_tags.iter().copied().collect();
    let mut sections_to_keep = Vec::new();
    for entry in &section_table.entries {
        let tag = entry.tag();
        if entry.is_required() || keep_set.contains(tag) {
            sections_to_keep.push(entry.clone());
        } else if !entry.is_strippable() {
            return Err(AufError::Other(format!(
                "Section '{tag}' is not strippable"
            )));
        }
    }

    rewrite_sections_into_bytes(original, &header, &sections_to_keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auf::meta::AufMeta;
    use crate::auf::reader::BackendTag;
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

    fn build_two_variant_auf() -> Vec<u8> {
        AufWriter::new(make_meta(), make_tokenizer(), [99u8; 32], 1234, 5678)
            .add_weights_section("WEIGHTS_CPU_AOS", vec![1u8; 128])
            .add_weights_section("WEIGHTS_ADRENO_SOA", vec![2u8; 256])
            .build()
            .unwrap()
    }

    #[test]
    fn strip_removes_variant_keeps_required() {
        let original = build_two_variant_auf();
        // CPU_AOS만 유지
        let stripped = strip_bytes(&original, &["WEIGHTS_CPU_AOS"]).unwrap();

        // stripped AUF를 파싱해서 section 확인
        let header = AufHeader::from_bytes(&stripped).unwrap();
        let section_table = crate::auf::section::SectionTable::from_bytes(
            &stripped[header.section_table_offset as usize..],
            header.section_count,
        )
        .unwrap();

        let tags: Vec<_> = section_table
            .entries
            .iter()
            .map(|e| e.tag().to_owned())
            .collect();
        assert!(tags.contains(&"META".to_owned()));
        assert!(tags.contains(&"TOKENIZER".to_owned()));
        assert!(tags.contains(&"TENSOR_INDEX".to_owned()));
        assert!(tags.contains(&"WEIGHTS_CPU_AOS".to_owned()));
        assert!(!tags.contains(&"WEIGHTS_ADRENO_SOA".to_owned()));
    }

    #[test]
    fn stripped_auf_still_valid() {
        let original = build_two_variant_auf();
        let stripped = strip_bytes(&original, &["WEIGHTS_CPU_AOS"]).unwrap();

        // reader로 열 수 있어야 함
        let view = crate::auf::reader::open_from_bytes(stripped, BackendTag::CpuAos).unwrap();
        assert_eq!(view.meta.architecture, "llama");
        // source_hash 보존 확인
        assert_eq!(view.header.source_hash, [99u8; 32]);
        assert_eq!(view.header.source_size, 1234);
        assert_eq!(view.header.source_mtime, 5678);
    }

    #[test]
    fn strip_adreno_only() {
        let original = build_two_variant_auf();
        let stripped = strip_bytes(&original, &["WEIGHTS_ADRENO_SOA"]).unwrap();
        let view = crate::auf::reader::open_from_bytes(stripped, BackendTag::AdrenoSoa).unwrap();
        let wr = view.weights_range.unwrap();
        assert_eq!(wr.1, 256); // ADRENO_SOA payload size
    }

    #[test]
    fn strip_keeps_payload_integrity() {
        let cpu_payload = (0u8..=127u8).collect::<Vec<_>>();
        let adreno_payload = (128u8..=255u8).collect::<Vec<_>>();
        let original = AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
            .add_weights_section("WEIGHTS_CPU_AOS", cpu_payload.clone())
            .add_weights_section("WEIGHTS_ADRENO_SOA", adreno_payload.clone())
            .build()
            .unwrap();

        let stripped = strip_bytes(&original, &["WEIGHTS_CPU_AOS"]).unwrap();
        let view = crate::auf::reader::open_from_bytes(stripped, BackendTag::CpuAos).unwrap();
        let wb = view.weights_bytes().unwrap();
        assert_eq!(wb, cpu_payload.as_slice());
    }

    #[test]
    fn strip_both_variants_keep_only_required() {
        let original = build_two_variant_auf();
        // 아무 WEIGHTS도 유지하지 않음
        let stripped = strip_bytes(&original, &[]).unwrap();
        let header = AufHeader::from_bytes(&stripped).unwrap();
        // section_count = 3 (META + TOKENIZER + TENSOR_INDEX)
        assert_eq!(header.section_count, 3);
    }

    #[test]
    fn section_count_decreases_after_strip() {
        let original = build_two_variant_auf();
        let header_original = AufHeader::from_bytes(&original).unwrap();
        assert_eq!(header_original.section_count, 5);

        let stripped = strip_bytes(&original, &["WEIGHTS_CPU_AOS"]).unwrap();
        let header_stripped = AufHeader::from_bytes(&stripped).unwrap();
        assert_eq!(header_stripped.section_count, 4);
    }
}
