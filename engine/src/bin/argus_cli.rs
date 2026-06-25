//! argus-cli — single-prompt inference.
//!
//! ARGUS CLI 패밀리의 단일 추론 엔트리. legacy `generate` 의 standard happy path +
//! production resilience + score-free eviction(`eviction none|sliding|streaming` 및
//! `--load-plugin` stage)을 지원한다. 단일 프롬프트는 turn 경계가 없으므로 eviction 은 KV
//! 점유율 high-water 구동(`KvFillPressureSource`)으로 발동한다(`standard_happy` →
//! `build_standard_loop`).
//!
//! 미구현이라 명시적으로 reject 하는 모드(각각 argus-chat·argus-bench·argus-eval 로 안내):
//! chat, experiment, ppl, eval, dump, prompt-batch, weight swap, quant-window, offload,
//! KV-offload(`--swap-dir`), profile, tensor-partition, score-based eviction(h2o·d2o —
//! attention-score accumulator 필요).
//!
//! Note(streaming/sliding window 크기): eviction 의 retain window(streaming 은
//! `--set sink=` + `--set recent_window=`, sliding 은 `--set window=`)는 high-water(85% × `--max-seq-len`)
//! 보다 **작아야** 한다. window 가 high-water 이상이면 prune 후에도 pos 가 high-water 밑으로
//! 내려가지 못해 재발화(re-arm)가 끊기고, KV 가 다시 차 `KV Cache overflow` 로 종료된다. 예:
//! `eviction plugin --name streaming --set sink=4 --set recent_window=128` 은 `--max-seq-len` 을 ~512 이상으로 둔다
//! (128 ≪ 0.85×512=435).
//!
//! 공용 셋업(SessionInitCtx → tokenizer → KV alloc → resilience)은
//! [`build_inference_ctx`](argus_engine::session::bin_setup) 로 argus_bench 와 공유한다.
//! argus-bench(experiment-output + resilience runtime effect)는 별도 bin.

use anyhow::bail;
use argus_engine::session::bin_setup::build_inference_ctx;
use argus_engine::session::cli::Args;
use argus_engine::session::standard_happy::run_standard_happy_path;
use clap::Parser;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = Args::parse();

    // backlog P3 (2026-05-25): `--swap` shorthand → legacy 4 flag normalize.
    args.normalize_swap_shorthand();

    // v1-1: resilience default-on. `--no-resilience` 가 명시되면 effective=false.
    args.enable_resilience = !args.no_resilience;

    // 미지원 모드 reject. score-free eviction(none/sliding/streaming/`--load-plugin` stage)은
    // 통과하고 standard happy path 의 공용 디코드 경로가 처리한다. 미지원(qcf/skip/d2o-layer-alloc/
    // profile/partition/swap/quant-window/offload/score-based eviction)은 reject 유지.
    reject_unsupported_modes_v0(&args)?;

    if args.num_tokens < 1 {
        bail!("argus-cli: --num-tokens must be >= 1");
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
    if !args.dump_kinds().is_empty() || args.dump_dir.is_some() {
        bail!(
            "argus-cli v0: --dump <kinds> / --dump-dir is an argus-eval diagnostic; \
             use argus-eval --eval-ll --dump <kind> --dump-dir <dir>"
        );
    }
    if args.qcf_dump.is_some() {
        bail!("argus-cli v0: --qcf-dump moved to argus-eval (--qcf-dump with --eval-ll or --ppl)");
    }
    if args.effective_kv_mode() != "standard" {
        bail!(
            "argus-cli v0: only --kv-mode standard supported (quant-window/Offload planned for v1)"
        );
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
    // is_standard_happy_path 가 막던 나머지 가드를 명시 reject 로 이전 (eviction 만 해제).
    if args.skip_ratio.unwrap_or(0.0) > 0.0 {
        bail!("argus-cli v0: --skip-ratio not yet supported (planned for v1)");
    }
    // KV offload(`--swap-dir`)는 OffloadStage 배선이 필요 → argus-bench.
    if args.swap_dir.is_some() {
        bail!("argus-cli: --swap-dir (KV offload) belongs to argus-bench");
    }
    // Score-based eviction (importance / attention-score accumulator dependent) needs the
    // AttentionScoreAccumulator wired (argus-bench connects score_cell to ModelForward +
    // EvictionStage); argus-cli is score-free only (none / sliding / streaming + `--load-plugin <.so>
    // eviction plugin --name <stage>`). The capability is read generically from the stage's declared
    // StageCaps — no per-name list (this also subsumes the old `--d2o-layer-alloc` reject, since d2o
    // is score-based). Note: a dynamically-loaded stage's caps don't cross the `.so` ABI yet (caps
    // are Phase 2), so a score-based dynamic stage is assumed score-free here — the same gap as
    // before, to be closed when the stage ABI carries caps.
    if argus_engine::kv::eviction::stage_registry::stage_is_score_based(args.eviction_policy()) {
        bail!(
            "argus-cli: score-based eviction '{}' belongs to argus-bench (needs attention-score \
             accumulator); argus-cli supports none / sliding / streaming and --load-plugin stages",
            args.eviction_policy()
        );
    }
    Ok(())
}
