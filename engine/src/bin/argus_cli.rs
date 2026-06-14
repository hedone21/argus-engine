//! argus-cli — single-prompt inference.
//!
//! ARGUS CLI 패밀리의 단일 추론 엔트리. legacy `generate` 의 standard happy
//! path + production resilience 를 지원한다. chat / experiment / ppl / eval /
//! dump / prompt-batch / weight swap / KIVI / offload / profile /
//! tensor-partition 는 아직 미구현이며 명시적으로 reject 한다.
//!
//! 공용 셋업(SessionInitCtx → tokenizer → KV alloc → resilience)은
//! [`build_inference_ctx`](argus_engine::session::bin_setup) 로 argus_bench 와 공유한다.
//! argus-bench(experiment-output + resilience runtime effect)는 별도 bin.

use anyhow::bail;
use argus_engine::session::bin_setup::build_inference_ctx;
use argus_engine::session::cli::{Args, KvMode};
use argus_engine::session::is_standard_happy_path;
use argus_engine::session::standard_happy::run_standard_happy_path;
use clap::Parser;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = Args::parse();

    // backlog P3 (2026-05-25): `--swap` shorthand → legacy 4 flag normalize.
    args.normalize_swap_shorthand();

    // v1-1: resilience default-on. `--no-resilience` 가 명시되면 effective=false.
    args.enable_resilience = !args.no_resilience;

    reject_unsupported_modes_v0(&args)?;

    if !is_standard_happy_path(&args) {
        bail!(
            "argus-cli v0: this combination of args is not yet supported. \
             happy path requires: qcf_dump=none, skip_ratio=0, d2o_layer_alloc=off, \
             profile=off, profile_events=off, eviction_policy=none, tensor_partition=0, \
             swap_intra_forward=off, swap_layer_immediate=off, swap_phase_aware=off."
        );
    }
    if args.num_tokens < 1 {
        bail!("argus-cli v0: --num-tokens must be >= 1");
    }

    let ctx = build_inference_ctx(args)?;
    run_standard_happy_path(ctx)
}

/// v0 에서 미구현인 mode 진입 flag 를 검사하여 즉시 reject 한다.
/// 모든 거부 메시지는 향후 갈 곳 (argus-chat / argus-bench / argus-eval) 을 명시.
fn reject_unsupported_modes_v0(args: &Args) -> anyhow::Result<()> {
    if args.chat {
        bail!("argus-cli v0: --chat moved to argus-chat (planned)");
    }
    if args.chat_socket.is_some() || args.chat_tcp.is_some() {
        bail!("argus-cli v0: --chat-socket / --chat-tcp moved to argus-chat (planned)");
    }
    if args.experiment_schedule.is_some() {
        bail!(
            "argus-cli v0: --experiment-schedule moved to argus-eval experiment; use argus-eval --experiment-schedule"
        );
    }
    if args.experiment_output.is_some() {
        bail!("argus-cli v0: --experiment-output moved to argus-bench (planned)");
    }
    if args.ppl.is_some() {
        bail!("argus-cli v0: --ppl moved to argus-eval --ppl");
    }
    if args.eval_ll || args.eval_batch.is_some() || args.eval_continuation.is_some() {
        bail!(
            "argus-cli v0: --eval-ll / --eval-batch / --eval-continuation moved to argus-eval --eval-ll"
        );
    }
    if args.dump_importance {
        bail!("argus-cli v0: --dump-importance moved to argus-eval --dump-importance");
    }
    if args.qcf_dump.is_some() {
        bail!("argus-cli v0: --qcf-dump moved to argus-eval (--qcf-dump with --eval-ll or --ppl)");
    }
    if !matches!(args.effective_kv_mode(), KvMode::Standard) {
        bail!("argus-cli v0: only --kv-mode standard supported (KIVI/Offload planned for v1)");
    }
    if args.secondary_gguf.is_some()
        || args.force_swap_ratio.is_some()
        || args.swap_incremental_per_tick > 0
        || args.swap_intra_forward
        || args.swap_layer_immediate
        || args.swap_phase_aware
    {
        bail!("argus-cli v0: weight swap options not yet supported (planned for v1)");
    }
    if args.profile || args.profile_events {
        bail!("argus-cli v0: --profile / --profile-events not yet supported (planned for v1)");
    }
    if args.tensor_partition > 0.0 {
        bail!("argus-cli v0: --tensor-partition not yet supported (planned for v1)");
    }
    Ok(())
}
