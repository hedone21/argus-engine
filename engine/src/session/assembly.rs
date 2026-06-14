//! Phase 4-4-a: [`crate::session::DecodeLoop`] 조립자 헬퍼.
//!
//! `SessionInitCtx`에서 분기별 `DecodeLoop`를 조립하는 책임을 모았다. 본
//! phase는 standard happy path 한정 — `build_standard_loop` 만 제공한다.
//! chat / kivi / offload 변형은 Phase 4-5에서 추가 예정.
//!
//! ## `build_standard_loop` happy path 정의
//!
//! Phase 4-3 [`crate::session::forward::ModelForward`]는 다음을 미지원하므로
//! 표준 path 안의 모든 케이스를 흡수하지 못한다 (Phase 4-4.5 별도 sprint):
//! - chunked prefill (긴 prompt 메모리 spike 회피)
//! - score_accumulator / skip_config / importance_collector / variance_collector / profiler
//!
//! 따라서 [`is_standard_happy_path`] 가드를 통과한 args만 `build_standard_loop`로
//! 위임 가능하며, 미통과 args는 generate.rs의 기존 prefill+decode path를
//! fallback으로 사용한다.

pub mod build_bench_kivi_loop;
pub mod build_bench_loop;
pub mod build_standard_loop;

pub use build_bench_kivi_loop::build_bench_kivi_loop;
pub use build_bench_loop::{
    SwapWiringConfig, build_bench_loop, build_local_pressure_source, build_resilience_cache_manager,
};
pub use build_standard_loop::{build_standard_loop, is_standard_happy_path};
