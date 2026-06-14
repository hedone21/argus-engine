//! L2 Format 축 추상화 — base trait 거주 모듈 (§4, C2).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §4 / §2.1 C2.
//!
//! format = 축(표현/precision)이다. base trait 정의는 여기(L2 응집 모듈, `capability/` 동급)에
//! 두고, **impl 은 데이터에 내재**한다(KV → `kv/`(현 `pressure/`), weight → `models/weights/`).
//! **guard rail: impl 은 여기 금지** (§2.1).
//!
//! Phase α-W 는 `WeightFormat`(신설, 현 코드 trait 부재)만 입주시킨다. `KVCacheFormat` 은
//! **Phase α-K substep (1)** 에서 `format/kv_cache_format.rs` 로 입주한다(C2 연혁 — `Merge`
//! compact arg 동반). 단 substep (1) 은 purely additive·unwired 이라 현 `kv_cache_ops.rs` 의
//! `KVCacheOps` 와 **공존**한다(rename 이 아니라 신규 trait 신설; 기존 경로 무변).

pub mod builtin_kv_formats;
pub mod dtype_layout;
pub mod dynamic_format_registry;
pub mod kv_cache_format;
pub mod kv_snapshot;
pub mod selective_read;
pub mod weight_format;

pub use builtin_kv_formats::{builtin_format_dtype, ensure_builtin_kv_formats_registered};
pub use dtype_layout::{
    decode_via_descriptor, dequant_to_f32_tensor, dequant_via_descriptor, dtype_to_layout_desc,
    encode_via_descriptor, layout_desc_to_builtin_dtype,
};
pub use kv_cache_format::{AttnDims, KVCacheFormat, Merge};
pub use kv_snapshot::SnapshotRestore;
pub use selective_read::SelectiveRead;
pub use weight_format::{LayerDispatch, PartitionShare, WeightFormat};
