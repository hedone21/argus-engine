/// AUF (Argus Unified Format) v0.1 — self-contained weight asset reader/writer/stripper.
///
/// # 개요
///
/// AUF는 GGUF의 derived but independent self-contained 자산 포맷이다.
/// 단일 파일에 모델 메타데이터, tokenizer, tensor index, backend별 사전 변환된 weight payload를 포함한다.
///
/// # 컴포넌트
///
/// - [`header`]: 256B 고정 헤더 (`AufHeader`). magic, format version, source hash.
/// - [`section`]: section table (`SectionTable`, `SectionEntry`). flag bits.
/// - [`meta`]: META section JSON payload (`AufMeta`).
/// - [`tokenizer`]: TOKENIZER section binary payload (`AufTokenizer`).
/// - [`tensor_index`]: TENSOR_INDEX section (`TensorIndex`, `TensorEntry`).
/// - [`reader`]: `open()` — mmap 기반 파일 파싱 + 전 검증 (INV-132, INV-133, INV-134).
/// - [`writer`]: `AufWriter` — section layout 결정 + atomic file write.
/// - [`stripper`]: `strip_bytes()` / `strip()` — dead variant 제거.
/// - [`source_hash`]: hybrid sha256 계산 (`compute_source_hash`).
/// - [`error`]: 통합 에러 타입 (`AufError`, `AufResult`).
///
/// # Spec 참조
///
/// - ENG-DAT-096 (`spec/33-engine-data.md` §3.22)
/// - ENG-ALG-223 (`spec/32-engine-algorithms.md` §3.12.17)
/// - INV-132~134 (`spec/41-invariants.md` §3.16)
pub mod dtype_convert;
pub mod error;
pub mod header;
pub mod meta;
pub mod q4_0_soa;
pub mod reader;
pub mod section;
pub mod source_hash;
pub mod stripper;
pub mod tensor_index;
pub mod tokenizer;
pub mod writer;

// 편의 re-export
pub use dtype_convert::{build_dtype_candidates, convert_tensor_dtype};
pub use error::{AufError, AufResult};
pub use header::{
    AufHeader, CAPABILITY_BIT_LM_HEAD_Q4_0, CAPABILITY_BIT_MULTI_DTYPE, HEADER_SIZE,
    READER_KNOWN_CAPABILITIES, READER_MAX_FORMAT_MAJOR,
};
pub use meta::AufMeta;
pub use q4_0_soa::{QK4_0, q4_0_aos_to_adreno_soa, q4_0_aos_to_soa_unshuffled};
pub use reader::{AufView, BackendTag, LmHeadPayload, open, open_from_bytes};
pub use section::{
    SECTION_COMPRESSED, SECTION_REQUIRED, SECTION_STRIPPABLE, SectionEntry, SectionTable, TAG_META,
    TAG_TENSOR_INDEX, TAG_TOKENIZER, TAG_WEIGHTS_ADRENO_SOA, TAG_WEIGHTS_CPU_AOS,
    TAG_WEIGHTS_CUDA_AOS,
};
pub use source_hash::{compute_source_hash, compute_source_hash_from_bytes};
pub use stripper::{strip, strip_bytes};
pub use tensor_index::{LAYER_IDX_CROSS, TensorDType, TensorEntry, TensorIndex, TensorKind};
pub use tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
pub use writer::AufWriter;
