//! argus-chat — OpenAI-compatible multi-turn chat server.
//!
//! Serves `POST /v1/chat/completions` (streaming + non-streaming) over HTTP,
//! backed by the multi-turn [`ChatSession`](argus_engine::session::chat::session::ChatSession).
//! Supports all three KV modes (Standard / KIVI / Offload, via `--kv-mode`) and
//! integrates production resilience (manager IPC) into the decode loop —
//! throttle / suspend / target-TBT directives + per-token heartbeats apply during
//! generation, exactly as the standard happy path.
//!
//! Shares the session bootstrap (`build_inference_prelude`) with argus-cli /
//! argus-bench / argus-eval. `--interactive` runs a local stdin REPL instead of
//! the HTTP server.

use anyhow::bail;
use argus_engine::session::bin_setup::{InferencePrelude, build_inference_prelude};
use argus_engine::session::chat::build::build_chat_session;
use argus_engine::session::chat::repl::{ChatReplArgs, run_chat_repl_v2};
use argus_engine::session::chat::server::serve;
use argus_engine::session::chat_template::ChatTemplate;
use argus_engine::session::cli::Args;
use clap::Parser;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = Args::parse();

    // `--swap` shorthand → legacy 4 flag normalize (argus_cli 와 동일).
    args.normalize_swap_shorthand();
    // resilience default-on. `--no-resilience` 명시 시 effective=false.
    args.enable_resilience = !args.no_resilience;

    reject_unsupported_for_chat(&args)?;

    let InferencePrelude {
        init, tokenizer, ..
    } = build_inference_prelude(&args)?;

    let arch = init.model.config.arch;
    let eos = args.eos_token_id.unwrap_or(init.model.config.eos_token_id);
    let vocab = init.model.config.vocab_size;
    let base_sampling = init.sampling_config.clone();
    let max_seq_len = args.max_seq_len;

    // Template guard: Gemma3 has no chat template → fail fast with a clear message.
    ChatTemplate::new(arch)?;

    let mut session = build_chat_session(init, &args)?;

    if args.interactive {
        let repl_args = ChatReplArgs {
            model_arch: arch,
            tokenizer: &tokenizer,
            eos_token_id: eos,
            vocab_size: vocab,
            sampling_config: &base_sampling,
            max_seq_len,
            system_prompt: args.system_prompt.as_deref(),
            initial_user_prompt: if args.prompt.trim().is_empty() {
                None
            } else {
                Some(args.prompt.as_str())
            },
            chat_socket: args.chat_socket.as_deref(),
            chat_tcp: args.chat_tcp.as_deref(),
            repetition_window: base_sampling.repetition_window,
            max_new_tokens: args.num_tokens,
        };
        run_chat_repl_v2(&repl_args, &mut session)
    } else {
        serve(&args, session, tokenizer, arch, eos, vocab, base_sampling)
    }
}

/// Reject modes that argus-chat does not support. argus-chat ACCEPTS `--chat*`,
/// all `--kv-mode` values, `--eviction-policy`, `--system-prompt`, `--listen`,
/// `--interactive`; it rejects eval / ppl / experiment / profiling / weight-swap /
/// tensor-partition / skip / d2o-layer-alloc (handled by argus-eval/argus-bench).
fn reject_unsupported_for_chat(args: &Args) -> anyhow::Result<()> {
    if args.experiment_schedule.is_some() || args.experiment_output.is_some() {
        bail!("argus-chat: experiment modes belong to argus-bench / argus-eval");
    }
    if args.ppl.is_some() {
        bail!("argus-chat: --ppl belongs to argus-eval --ppl");
    }
    if args.eval_ll || args.eval_batch.is_some() || args.eval_continuation.is_some() {
        bail!("argus-chat: --eval-* belongs to argus-eval --eval-ll");
    }
    if args.dump_importance {
        bail!("argus-chat: --dump-importance belongs to argus-eval");
    }
    if args.qcf_dump.is_some() {
        bail!("argus-chat: --qcf-dump belongs to argus-eval");
    }
    if args.secondary_gguf.is_some()
        || args.force_swap_ratio.is_some()
        || args.swap_incremental_per_tick > 0
        || args.swap_intra_forward
        || args.swap_layer_immediate
        || args.swap_phase_aware
    {
        bail!("argus-chat: weight swap is not supported (use argus-bench)");
    }
    if args.profile || args.profile_events {
        bail!("argus-chat: --profile / --profile-events not supported");
    }
    if args.tensor_partition > 0.0 {
        bail!("argus-chat: --tensor-partition not supported");
    }
    if args.skip_ratio.unwrap_or(0.0) > 0.0 {
        bail!("argus-chat: --skip-ratio not supported");
    }
    if args.d2o_layer_alloc() {
        bail!("argus-chat: --d2o-layer-alloc (variance measurement) belongs to argus-eval");
    }
    Ok(())
}
