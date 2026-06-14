/// AUF TENSOR_INDEX section 직렬화/역직렬화 (ENG-DAT-096.8).
///
/// 구조:
/// ```text
/// [0..16)   magic = "ARGUS_TIDX\0\0\0\0\0\0" (16B)
/// [16..20)  schema_version: u32 = 1
/// [20..24)  variant_count: u32
/// [24..28)  tensor_count: u32
/// [28..32)  _pad: u32
/// [32..32+variant_count*16)  variant tag 배열 (각 16B)
/// [그 이후] tensor entry 배열 (가변 길이)
/// ```
use crate::auf::error::{AufError, AufResult};

/// TENSOR_INDEX magic (16B).
pub const TENSOR_INDEX_MAGIC: &[u8; 16] = b"ARGUS_TIDX\0\0\0\0\0\0";

/// `tensor_count` / `variant_count` 고정 헤더 크기 (32B).
pub const TENSOR_INDEX_FIXED_HEADER: usize = 32;

/// cross-layer tensor (embedding/final_norm/lm_head)의 layer_idx 예약값.
pub const LAYER_IDX_CROSS: u32 = u32::MAX;

/// DType 열거형 (ENG-DAT-096.8 dtype 코드).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TensorDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    Q4_0 = 3,
    Q4_1 = 4,
    Q8_0 = 5,
    U8 = 6,
}

impl TensorDType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::BF16),
            3 => Some(Self::Q4_0),
            4 => Some(Self::Q4_1),
            5 => Some(Self::Q8_0),
            6 => Some(Self::U8),
            _ => None,
        }
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// TensorKind 열거형 (ENG-DAT-096.8 kind 코드).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TensorKind {
    AttnQ = 0,
    AttnK = 1,
    AttnV = 2,
    AttnO = 3,
    FfnGate = 4,
    FfnUp = 5,
    FfnDown = 6,
    AttnNorm = 7,
    FfnNorm = 8,
    Embedding = 9,
    FinalNorm = 10,
    LmHead = 11,
    AttnQBias = 12,
    AttnKBias = 13,
    AttnVBias = 14,
}

impl TensorKind {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::AttnQ),
            1 => Some(Self::AttnK),
            2 => Some(Self::AttnV),
            3 => Some(Self::AttnO),
            4 => Some(Self::FfnGate),
            5 => Some(Self::FfnUp),
            6 => Some(Self::FfnDown),
            7 => Some(Self::AttnNorm),
            8 => Some(Self::FfnNorm),
            9 => Some(Self::Embedding),
            10 => Some(Self::FinalNorm),
            11 => Some(Self::LmHead),
            12 => Some(Self::AttnQBias),
            13 => Some(Self::AttnKBias),
            14 => Some(Self::AttnVBias),
            _ => None,
        }
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// 단일 tensor entry.
///
/// variant_offsets/variant_sizes는 variant_count 길이. 해당 variant가 없으면 u64::MAX.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub layer_idx: u32,
    pub kind: u32,
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub alignment: u64,
    /// 각 backend variant의 section-local payload offset (없으면 u64::MAX).
    pub variant_offsets: Vec<u64>,
    /// 각 backend variant의 payload byte size.
    pub variant_sizes: Vec<u64>,
}

impl TensorEntry {
    /// 직렬화 크기 계산.
    pub fn serialized_size(shape_rank: usize, variant_count: usize) -> usize {
        // layer_idx(4) + kind(4) + dtype(4) + shape_rank(4) = 16
        // shape: shape_rank * 8
        // alignment(8)
        // variant_offsets: variant_count * 8
        // variant_sizes: variant_count * 8
        16 + shape_rank * 8 + 8 + variant_count * 8 * 2
    }

    /// 바이트열에서 tensor entry를 파싱한다. `variant_count`는 TENSOR_INDEX 헤더에서 결정.
    pub fn from_bytes(bytes: &[u8], variant_count: usize) -> AufResult<(Self, usize)> {
        let mut pos = 0;

        macro_rules! require {
            ($need:expr, $ctx:literal) => {
                if bytes.len() < pos + $need {
                    return Err(AufError::TensorIndexFormat {
                        detail: format!(
                            "{}: need {} bytes at pos {}, have {}",
                            $ctx,
                            $need,
                            pos,
                            bytes.len()
                        ),
                    });
                }
            };
        }

        require!(16, "entry fixed header");
        let layer_idx = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let kind = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let dtype = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let shape_rank = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        require!(shape_rank * 8, "shape");
        let mut shape = Vec::with_capacity(shape_rank);
        for _ in 0..shape_rank {
            let dim = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;
            shape.push(dim);
        }

        require!(8, "alignment");
        let alignment = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;

        require!(variant_count * 8, "variant_offsets");
        let mut variant_offsets = Vec::with_capacity(variant_count);
        for _ in 0..variant_count {
            variant_offsets.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()));
            pos += 8;
        }

        require!(variant_count * 8, "variant_sizes");
        let mut variant_sizes = Vec::with_capacity(variant_count);
        for _ in 0..variant_count {
            variant_sizes.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()));
            pos += 8;
        }

        Ok((
            TensorEntry {
                layer_idx,
                kind,
                dtype,
                shape,
                alignment,
                variant_offsets,
                variant_sizes,
            },
            pos,
        ))
    }

    /// 바이트 벡터로 직렬화한다.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.layer_idx.to_le_bytes());
        out.extend_from_slice(&self.kind.to_le_bytes());
        out.extend_from_slice(&self.dtype.to_le_bytes());
        out.extend_from_slice(&(self.shape.len() as u32).to_le_bytes());
        for &dim in &self.shape {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&self.alignment.to_le_bytes());
        for &off in &self.variant_offsets {
            out.extend_from_slice(&off.to_le_bytes());
        }
        for &sz in &self.variant_sizes {
            out.extend_from_slice(&sz.to_le_bytes());
        }
        out
    }
}

/// TENSOR_INDEX section 파싱 결과.
#[derive(Debug, Clone)]
pub struct TensorIndex {
    /// variant_count개의 backend tag (WEIGHTS_* tag와 일치).
    /// 각 tag는 24B, NUL 패딩 (SECTION_TAG_SIZE와 동일).
    pub variant_tags: Vec<[u8; 24]>,
    pub entries: Vec<TensorEntry>,
}

impl TensorIndex {
    /// TENSOR_INDEX section payload 바이트열에서 파싱한다.
    pub fn from_bytes(payload: &[u8]) -> AufResult<Self> {
        if payload.len() < TENSOR_INDEX_FIXED_HEADER {
            return Err(AufError::TensorIndexFormat {
                detail: "payload too small for header".to_owned(),
            });
        }

        if &payload[0..16] != TENSOR_INDEX_MAGIC {
            return Err(AufError::TensorIndexFormat {
                detail: "magic mismatch".to_owned(),
            });
        }

        let schema_version = u32::from_le_bytes(payload[16..20].try_into().unwrap());
        if schema_version != 1 {
            return Err(AufError::TensorIndexFormat {
                detail: format!("unsupported schema_version={schema_version}"),
            });
        }

        let variant_count = u32::from_le_bytes(payload[20..24].try_into().unwrap()) as usize;
        let tensor_count = u32::from_le_bytes(payload[24..28].try_into().unwrap()) as usize;
        // _pad = [28..32]
        let mut pos = 32;

        // variant tag 배열 (각 24B, SECTION_TAG_SIZE와 동일)
        let variant_tags_size = variant_count * 24;
        if payload.len() < pos + variant_tags_size {
            return Err(AufError::TensorIndexFormat {
                detail: "variant tags extend beyond payload".to_owned(),
            });
        }
        let mut variant_tags = Vec::with_capacity(variant_count);
        for _ in 0..variant_count {
            let tag: [u8; 24] = payload[pos..pos + 24].try_into().unwrap();
            variant_tags.push(tag);
            pos += 24;
        }

        // tensor entries
        let mut entries = Vec::with_capacity(tensor_count);
        for i in 0..tensor_count {
            if pos >= payload.len() {
                return Err(AufError::TensorIndexFormat {
                    detail: format!("tensor entry {i} out of bounds"),
                });
            }
            let (entry, consumed) = TensorEntry::from_bytes(&payload[pos..], variant_count)
                .map_err(|e| AufError::TensorIndexFormat {
                    detail: format!("tensor entry {i}: {e}"),
                })?;
            entries.push(entry);
            pos += consumed;
        }

        Ok(TensorIndex {
            variant_tags,
            entries,
        })
    }

    /// TENSOR_INDEX section payload 바이트 벡터로 직렬화한다.
    pub fn to_bytes(&self) -> Vec<u8> {
        let variant_count = self.variant_tags.len();
        let tensor_count = self.entries.len();

        let mut out = Vec::new();
        out.extend_from_slice(TENSOR_INDEX_MAGIC); // 16B
        out.extend_from_slice(&1u32.to_le_bytes()); // schema_version
        out.extend_from_slice(&(variant_count as u32).to_le_bytes());
        out.extend_from_slice(&(tensor_count as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // _pad

        for tag in &self.variant_tags {
            out.extend_from_slice(tag);
        }
        for entry in &self.entries {
            out.extend_from_slice(&entry.to_bytes());
        }
        out
    }

    /// `kind=LmHead(11), layer_idx=u32::MAX` 조건을 만족하는 entry를 반환한다.
    ///
    /// G-1-A spec (INV-135): capability bit 2 = 1이면 이 entry가 정확히 1개 존재해야 한다.
    /// 0개이면 `None` (호출자가 INV-135 에러로 처리), 2개 이상이면 첫 번째 반환 (spec 위반).
    pub fn find_lm_head_entry(&self) -> Option<&TensorEntry> {
        self.entries
            .iter()
            .find(|e| e.kind == TensorKind::LmHead.as_u32() && e.layer_idx == LAYER_IDX_CROSS)
    }

    /// backend variant tag 문자열로 variant index를 조회한다.
    pub fn variant_index_for_tag(&self, weights_tag: &str) -> Option<usize> {
        self.variant_tags.iter().position(|t| {
            let end = t.iter().position(|&b| b == 0).unwrap_or(24);
            std::str::from_utf8(&t[..end]).ok() == Some(weights_tag)
        })
    }

    /// variant tag 이름 목록 (NUL 트리밍).
    pub fn variant_tag_strings(&self) -> Vec<&str> {
        self.variant_tags
            .iter()
            .map(|t| {
                let end = t.iter().position(|&b| b == 0).unwrap_or(24);
                std::str::from_utf8(&t[..end]).unwrap_or("")
            })
            .collect()
    }

    /// 동일 (`layer_idx`, `kind`)에 해당하는 모든 entry를 반환한다 (B-4, INV-137).
    ///
    /// AUF v0.2 multi-dtype 모드에서는 같은 (`layer_idx`, `kind`) 쌍에 dtype이 다른
    /// entry가 여러 개 존재할 수 있다. 이 메서드는 모든 후보 entry를 순서대로 반환한다.
    ///
    /// AUF v0.1.x single-dtype 모드에서는 최대 1개를 반환한다.
    pub fn entries_for(&self, layer_idx: u32, kind: u32) -> Vec<&TensorEntry> {
        self.entries
            .iter()
            .filter(|e| e.layer_idx == layer_idx && e.kind == kind)
            .collect()
    }

    /// 동일 (`layer_idx`, `kind`, `dtype`)에 해당하는 entry를 반환한다 (INV-137).
    ///
    /// multi-dtype 모드에서 특정 dtype의 entry를 명시적으로 조회한다.
    pub fn find_entry_by_dtype(
        &self,
        layer_idx: u32,
        kind: u32,
        dtype: u32,
    ) -> Option<&TensorEntry> {
        self.entries
            .iter()
            .find(|e| e.layer_idx == layer_idx && e.kind == kind && e.dtype == dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_variant_tag(s: &str) -> [u8; 24] {
        let mut tag = [0u8; 24];
        let b = s.as_bytes();
        tag[..b.len().min(24)].copy_from_slice(&b[..b.len().min(24)]);
        tag
    }

    fn make_entry(layer_idx: u32, kind: u32, variant_count: usize) -> TensorEntry {
        TensorEntry {
            layer_idx,
            kind,
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![4096, 4096],
            alignment: 64 * 1024,
            variant_offsets: vec![0u64; variant_count],
            variant_sizes: vec![1024u64; variant_count],
        }
    }

    fn make_index() -> TensorIndex {
        TensorIndex {
            variant_tags: vec![
                make_variant_tag("WEIGHTS_ADRENO_SOA"),
                make_variant_tag("WEIGHTS_CPU_AOS"),
            ],
            entries: vec![
                make_entry(0, TensorKind::AttnQ.as_u32(), 2),
                make_entry(0, TensorKind::AttnK.as_u32(), 2),
                make_entry(LAYER_IDX_CROSS, TensorKind::Embedding.as_u32(), 2),
            ],
        }
    }

    #[test]
    fn round_trip() {
        let idx = make_index();
        let bytes = idx.to_bytes();
        let idx2 = TensorIndex::from_bytes(&bytes).unwrap();
        assert_eq!(idx2.variant_tags.len(), 2);
        assert_eq!(idx2.entries.len(), 3);
        assert_eq!(idx2.entries[0].layer_idx, 0);
        assert_eq!(idx2.entries[0].kind, TensorKind::AttnQ.as_u32());
        assert_eq!(idx2.entries[2].layer_idx, LAYER_IDX_CROSS);
        assert_eq!(idx2.variant_tag_strings()[0], "WEIGHTS_ADRENO_SOA");
        assert_eq!(idx2.variant_tag_strings()[1], "WEIGHTS_CPU_AOS");
    }

    #[test]
    fn variant_index_lookup() {
        let idx = make_index();
        let bytes = idx.to_bytes();
        let idx2 = TensorIndex::from_bytes(&bytes).unwrap();
        assert_eq!(idx2.variant_index_for_tag("WEIGHTS_ADRENO_SOA"), Some(0));
        assert_eq!(idx2.variant_index_for_tag("WEIGHTS_CPU_AOS"), Some(1));
        assert_eq!(idx2.variant_index_for_tag("WEIGHTS_CUDA_AOS"), None);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = make_index().to_bytes();
        bytes[0] = 0xFF;
        let err = TensorIndex::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AufError::TensorIndexFormat { .. }));
    }

    #[test]
    fn tensor_entry_round_trip() {
        let e = make_entry(5, TensorKind::FfnGate.as_u32(), 2);
        let bytes = e.to_bytes();
        let (e2, consumed) = TensorEntry::from_bytes(&bytes, 2).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(e2.layer_idx, 5);
        assert_eq!(e2.kind, TensorKind::FfnGate.as_u32());
        assert_eq!(e2.shape, vec![4096, 4096]);
        assert_eq!(e2.variant_offsets.len(), 2);
        assert_eq!(e2.variant_sizes[0], 1024);
    }

    #[test]
    fn layer_idx_cross_reserved() {
        assert_eq!(LAYER_IDX_CROSS, u32::MAX);
        let e = make_entry(LAYER_IDX_CROSS, TensorKind::Embedding.as_u32(), 1);
        let bytes = e.to_bytes();
        let (e2, _) = TensorEntry::from_bytes(&bytes, 1).unwrap();
        assert_eq!(e2.layer_idx, u32::MAX);
    }

    /// B-4: entries_for() — 동일 (layer_idx, kind)의 multi-dtype entries 반환.
    #[test]
    fn entries_for_multi_dtype() {
        // layer=0, kind=AttnQ에 Q4_0과 F16 두 entry 등록
        let q4_entry = TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![4096, 4096],
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![1024],
        };
        let f16_entry = TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::F16.as_u32(),
            shape: vec![4096, 4096],
            alignment: 64,
            variant_offsets: vec![1024],
            variant_sizes: vec![2048],
        };
        let other_entry = TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnK.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![4096, 4096],
            alignment: 64,
            variant_offsets: vec![3072],
            variant_sizes: vec![1024],
        };
        let idx = TensorIndex {
            variant_tags: vec![make_variant_tag("WEIGHTS_CPU_AOS")],
            entries: vec![q4_entry, f16_entry, other_entry],
        };

        let entries = idx.entries_for(0, TensorKind::AttnQ.as_u32());
        assert_eq!(entries.len(), 2, "must return both Q4_0 and F16 entries");
        assert_eq!(entries[0].dtype, TensorDType::Q4_0.as_u32());
        assert_eq!(entries[1].dtype, TensorDType::F16.as_u32());

        // AttnK는 1개만
        let k_entries = idx.entries_for(0, TensorKind::AttnK.as_u32());
        assert_eq!(k_entries.len(), 1);

        // 없는 (layer, kind) → 빈 vec
        let empty = idx.entries_for(99, TensorKind::AttnQ.as_u32());
        assert!(empty.is_empty());
    }

    /// B-4: find_entry_by_dtype() — 특정 dtype entry 조회.
    #[test]
    fn find_entry_by_dtype_works() {
        let q4_entry = TensorEntry {
            layer_idx: 1,
            kind: TensorKind::FfnGate.as_u32(),
            dtype: TensorDType::Q4_0.as_u32(),
            shape: vec![8192, 4096],
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![512],
        };
        let f16_entry = TensorEntry {
            layer_idx: 1,
            kind: TensorKind::FfnGate.as_u32(),
            dtype: TensorDType::F16.as_u32(),
            shape: vec![8192, 4096],
            alignment: 64,
            variant_offsets: vec![512],
            variant_sizes: vec![1024],
        };
        let idx = TensorIndex {
            variant_tags: vec![make_variant_tag("WEIGHTS_CPU_AOS")],
            entries: vec![q4_entry, f16_entry],
        };

        let found_q4 =
            idx.find_entry_by_dtype(1, TensorKind::FfnGate.as_u32(), TensorDType::Q4_0.as_u32());
        assert!(found_q4.is_some());
        assert_eq!(found_q4.unwrap().variant_sizes[0], 512);

        let found_f16 =
            idx.find_entry_by_dtype(1, TensorKind::FfnGate.as_u32(), TensorDType::F16.as_u32());
        assert!(found_f16.is_some());
        assert_eq!(found_f16.unwrap().variant_sizes[0], 1024);

        // BF16은 없음
        let not_found =
            idx.find_entry_by_dtype(1, TensorKind::FfnGate.as_u32(), TensorDType::BF16.as_u32());
        assert!(not_found.is_none());
    }
}
