//! Phase 4-B: eval-ll (`--eval-ll`) 모드 추출.
//!
//! `bin/generate.rs::main()`의 eval_ll 분기 (l.1642~1992, 350 LOC)를
//! 외과적으로 이동. Light-lift 패턴 (4-A와 동일): EvalLlRunCtx로 외부
//! state packaging 후 `run_eval_ll(ctx)` 단일 호출.
//!
//! 공유 의존: `session::qcf_runtime::{run_qcf_warmup_workflow, run_layer_swap}`
//! (Phase 4-B-1에서 lib 이동).

pub mod args;
pub mod eval_loop;
pub mod eviction_hook;
pub mod fmt_bridge;
pub mod helpers;
pub mod hook;
pub mod kivi_hook;
pub mod output;
pub mod qcf_helpers;
pub mod runner;

pub use args::EvalLlRunCtx;
pub use eval_loop::run_eval_ll_generic;
pub use eviction_hook::EvictionHook;
pub use fmt_bridge::EvalCacheKind;
pub use helpers::{build_eval_ll_warmup_text, load_eval_questions};
pub use hook::{CacheSnapshot, PostStepResult, StepHook};
pub use kivi_hook::KiviHook;
pub use output::{EvalConfig, EvalOutput, EvalQuestion};
pub use qcf_helpers::{QcfSwapDumpContext, dump_qcf_swap_json};
pub use runner::run_eval_ll;
