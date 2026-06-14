//! Phase 4-C: PPL (`--ppl <text>`) 모드 추출.
//!
//! `bin/generate.rs::main()`의 ppl_main 분기 (170 LOC) + run_ppl free fn
//! (603 LOC) + run_kivi_ppl free fn (345 LOC) = 1,118 LOC를 session/ppl/
//! 로 외과적 이동. β light-lift 패턴 (4-A/4-B와 동일).
//!
//! 공유 의존: `session::qcf_runtime::{run_qcf_warmup_workflow,
//! dispatch_swap_weights, dump_layer_weights_to_dir, ...}` (Phase 4-B-1 +
//! 4-C-1에서 lib 노출).

pub mod args;
pub mod runner;

pub use args::PplRunCtx;
pub use runner::{run_kivi_ppl, run_ppl, run_ppl_dispatch};
