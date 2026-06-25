//! argus-bench — resilience benchmark / verify 엔진 bin.
//!
//! argus_cli (single-prompt happy path) 와 같은 추론 골격을 공유하되,
//! `--experiment-output` per-token JSONL 산출 + resilience directive(throttle /
//! target_tbt / suspend …) 의 런타임 효과 측정을 지원한다. verify 하네스
//! (`verify/verify.py`) 가 이 bin 을 baseline/action 으로 띄워 SystemSignal →
//! Policy → EngineCommand → Engine 전 경로를 검증한다.
//!
//! ## 단계적 흡수 (argus-bench AB sub-sprint)
//!
//! - **AB-0 (current)**: experiment-output JSONL + `[Experiment] Done` summary,
//!   throttle·target_tbt·suspend 런타임 효과. eviction·quant·offload·partition·
//!   weight-swap 는 후속 단계에서 stage 로 재배선.
//! - AB-1: EvictionStage (plan.evict) 재배선.
//! - AB-2: quant-window dynamic quant. AB-3: KvOffload. AB-4: tensor partition.
//! - AB-6: weight-swap 8종 SwapStage glue.

use anyhow::bail;
use argus_engine::session::bin_setup::{build_inference_ctx, build_quant_window_bench_ctx};
use argus_engine::session::cli::Args;
use argus_engine::session::experiment_run::{
    run_experiment_path, run_quant_window_experiment_path,
};
use argus_engine::session::mode::{mode_caps, resolve_kv_mode_checked};
use clap::Parser;

/// Whether the requested `--kv-mode` runs a quantized-KV pipeline (reads declared
/// `ModeCaps` — no concrete-technique name). Replaces `matches!(.., KvMode::QuantWindow)`.
fn is_quantized_kv_mode(args: &Args) -> bool {
    mode_caps(args.effective_kv_mode()).is_some_and(|c| c.is_quantized_kv)
}

/// Whether the requested `--kv-mode` owns an offload cache container. Replaces
/// `matches!(.., KvMode::Offload)`.
fn is_offload_mode(args: &Args) -> bool {
    mode_caps(args.effective_kv_mode()).is_some_and(|c| c.supports_offload)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = Args::parse();

    // `--swap` shorthand → legacy 4 flag normalize (argus_cli 와 동일).
    args.normalize_swap_shorthand();

    // resilience default-on. `--no-resilience` 명시 시 effective=false.
    args.enable_resilience = !args.no_resilience;

    // FORMAT-axis Phase 1: fail-fast on an unknown `--kv-mode` name. `KvMode` is now a
    // free `String`, so clap no longer rejects a typo at parse time — restore that via
    // the shared registry funnel (same one chat/eval use), else a typo'd mode would
    // silently fall through to the standard pipeline.
    resolve_kv_mode_checked(args.effective_kv_mode())?;

    reject_unsupported_modes_ab0(&args)?;

    // AB-1: standard happy path + resilience runtime effect + CLI `eviction
    // <policy>` (mid-decode KvEvict* directive). throttle/target_tbt/suspend/
    // eviction 은 런타임 directive 이므로 가드를 통과한다. 미지원 모드(skip /
    // d2o-layer-alloc / qcf / profile / partition / swap)만 차단.
    if !bench_supported(&args) {
        bail!(
            "argus-bench AB-1: this combination of args is not yet supported. \
             supported: eviction <none|sliding|streaming|h2o|h2o_plus> + resilience. \
             blocked: qcf_dump, skip_ratio, d2o_layer_alloc, profile, profile_events, \
             tensor_partition, weight-swap (→ AB-3)."
        );
    }
    if args.num_tokens < 1 {
        bail!("argus-bench: --num-tokens must be >= 1");
    }

    // AB-2: quantized-KV dynamic-quant 분기. quantized-KV mode(`ModeCaps.is_quantized_kv`)
    // 또는 `--kv-dynamic-quant`(orphan flag 재배선) 진입 시 QuantWindowForward + QuantWindowBitTransitionStage 경로.
    // 비양자화(Standard)는 ModelForward 경로. `--kv-dynamic-quant` 는 mode 가 아니므로 명시적 OR.
    if is_quantized_kv_mode(&args) || args.kv_dynamic_quant {
        let ctx = build_quant_window_bench_ctx(args)?;
        return run_quant_window_experiment_path(ctx);
    }

    let ctx = build_inference_ctx(args)?;
    run_experiment_path(ctx)
}

/// AB-1 bench 지원 args 가드. [`is_standard_happy_path`](argus_engine::session::is_standard_happy_path)
/// 와 동일하되 **eviction_policy != "none" 을 허용**한다 (resilience eviction은
/// `build_bench_loop` 가 처리). AB-4: `tensor_partition > 0` 허용(PartitionStage).
/// AB-6: `--secondary-gguf` 허용(SwapWeights directive → WeightSwapStage). 아직
/// 미지원인 skip / d2o-layer-alloc / qcf / profile / swap mode flag 는 그대로 차단.
fn bench_supported(args: &Args) -> bool {
    args.qcf_dump.is_none()
        && args.skip_ratio.unwrap_or(0.0) == 0.0
        && !args.profile
        && !args.profile_events
    // `--d2o-layer-alloc` is no longer gated here: it is a d2o-private knob that rides the StageArgs
    // blob, the per-layer variance machinery is runtime-dead, and the d2o plugin ignores it — so
    // accepting it is a no-op. swap mode 3종(intra_forward/phase_aware/layer_immediate)은 §5.9.2 허용.
}

/// AB-0 에서 미구현인 mode 진입 flag 를 검사하여 즉시 reject 한다.
/// argus_cli 의 reject 와 동일하되 `--experiment-output` 은 허용한다 (argus-bench
/// 의 핵심 기능). eviction/swap/quant-window/offload/partition 은 AB-1..AB-6 에서 해제.
fn reject_unsupported_modes_ab0(args: &Args) -> anyhow::Result<()> {
    if args.chat {
        bail!("argus-bench AB-0: --chat moved to argus-chat (planned)");
    }
    if args.chat_socket.is_some() || args.chat_tcp.is_some() {
        bail!("argus-bench AB-0: --chat-socket / --chat-tcp moved to argus-chat (planned)");
    }
    if args.experiment_schedule.is_some() {
        bail!(
            "argus-bench AB-0: --experiment-schedule (static schedule) unsupported; use mock_manager/manager IPC resilience"
        );
    }
    if args.ppl.is_some() {
        bail!("argus-bench AB-0: --ppl moved to argus-eval --ppl");
    }
    if args.eval_ll || args.eval_batch.is_some() || args.eval_continuation.is_some() {
        bail!(
            "argus-bench AB-0: --eval-ll / --eval-batch / --eval-continuation moved to argus-eval --eval-ll"
        );
    }
    if args.dump_importance {
        bail!("argus-bench AB-0: --dump-importance moved to argus-eval --dump-importance");
    }
    if !args.dump_kinds().is_empty() || args.dump_dir.is_some() {
        bail!(
            "argus-bench AB-0: --dump <kinds> / --dump-dir is an argus-eval diagnostic; \
             use argus-eval --eval-ll --dump <kind> --dump-dir <dir>"
        );
    }
    if args.qcf_dump.is_some() {
        bail!(
            "argus-bench AB-0: --qcf-dump moved to argus-eval (--qcf-dump with --eval-ll or --ppl)"
        );
    }
    // AB-2: quantized-KV mode 해제 (QuantWindowForward + QuantWindowBitTransitionStage 배선 완료). Offload(AB-3)는 bail 유지.
    if is_offload_mode(args) {
        bail!("argus-bench AB-3: --kv-mode Offload lands in AB-3");
    }
    // AB-6: `--secondary-gguf` 해제 — SwapWeights runtime directive 경로의 secondary 공급원
    // (WeightSwapStage + build_bench_loop SwapWiringConfig 배선 완료). CLI 정적 swap
    // (force_swap_ratio/incremental_per_tick)은 차단 유지. swap mode 3종
    // (intra-forward/phase-aware/layer-immediate)은 §5.9.2 IntraForward hook 실배선
    // 완료(5862325c)로 차단 해제 — SwapWiringConfig.default_mode 로 흘러 SwapWeights
    // directive 도착 시 해당 mode로 commit된다.
    if args.force_swap_ratio.is_some() || args.swap_incremental_per_tick > 0 {
        bail!(
            "argus-bench AB-6: CLI-static weight swap not supported; \
             use SwapWeights runtime directive (mock_manager/manager IPC) with --secondary-gguf"
        );
    }
    if args.profile || args.profile_events {
        bail!("argus-bench AB-0: --profile / --profile-events not yet supported");
    }
    // AB-4: --tensor-partition 해제 — PartitionStage + build_bench_loop 배선 완료.
    Ok(())
}
