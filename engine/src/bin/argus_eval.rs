//! argus-eval — log-likelihood / perplexity / importance-dump 측정 엔트리.
//!
//! ARGUS CLI 패밀리의 eval 측정 bin. 폐기된 legacy `generate` 의 `--eval-ll` /
//! `--ppl` / `--dump-importance` 모드를 신규 bin 으로 정착시킨다 (Phase γ-3a,
//! `arch/inference_pipeline.md` §13). 신규 메커니즘 0 — 전부 기존 orphan runner
//! 재배선. 동작 의미는 legacy 의 해당 모드와 등가.
//!
//! ## 표면 (flag-based dispatch, clap subcommand 아님)
//!
//! - `ll` — `--eval-ll` / `--eval-batch` / `--eval-continuation`
//! - `ppl` — `--ppl <text>` + `ppl_*` 패밀리
//! - `dump importance` — `--dump-importance`
//! - `dump qcf` — `--qcf-dump <path>` (modifier — `ll`/`ppl` 에 결합)
//! - `experiment` — `--experiment-schedule` (정적 directive schedule, `ScheduleCommandSource` 경유)
//!
//! quant-window(`--kv-mode kivi`)는 ll/ppl 양쪽에서 지원 — 별 `QuantizedRecentWindowCache` 경로(§13.6).
//!
//! ## resilience default-off
//!
//! argus_cli/argus_bench 와 달리 `enable_resilience = !no_resilience` 인버전
//! 라인을 **생략**한다 → `enable_resilience` 기본 false, `--enable-resilience`
//! opt-in 만 동작(handoff 합의). eval ctx 에는 IPC adapter 슬롯이 없으므로
//! opt-in 시 `--enable-resilience` 는 score_accumulator 강제 활성에만 관여한다.
//!
//! ## AUF 제약
//!
//! AUF 단일파일 모델은 tokenizer 자동 해석 부재 — `--tokenizer-path` 명시 필수
//! (`session::eval_setup` 모듈 헤더 참조).

use anyhow::bail;
use argus_engine::experiment::ExperimentSchedule;
use argus_engine::session::bin_setup::build_inference_ctx;
use argus_engine::session::cli::Args;
use argus_engine::session::eval_setup;
use argus_engine::session::experiment::ScheduleCommandSource;
use argus_engine::session::mode::{mode_caps, resolve_kv_mode_checked};
use argus_engine::session::run_experiment_schedule_path;
use clap::Parser;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = Args::parse();

    // `--swap` shorthand → legacy 4 flag normalize (argus_cli/bench 와 동일).
    args.normalize_swap_shorthand();

    // ★ argus_cli/bench 와 달리 인버전 라인 생략 → enable_resilience 기본 false
    //   (handoff default-off — `--enable-resilience` opt-in 자연 동작).

    reject_unsupported_modes_eval(&args)?;
    if !eval_supported(&args) {
        bail!(
            "argus-eval: this combination of args is not yet supported. \
             supported: eval-ll/ppl/dump-importance + eviction-policy + qcf-dump + \
             skip-ratio/skip-layers + weight-swap + --kv-mode kivi. \
             blocked: profile, profile_events, tensor_partition, chat."
        );
    }
    if args.num_tokens < 1 {
        bail!("argus-eval: --num-tokens must be >= 1");
    }

    // FORMAT-axis Phase 1: fail-fast on an unknown `--kv-mode` name (the eval runners
    // only read `mode_caps()`, which folds an unknown name to None → silently
    // classifies as the Standard runner — wrong numbers for a measurement tool). This
    // also runs the KV_MODES gc-sections self-test, which the eval prologue
    // (`build_eval_base`) otherwise skips.
    resolve_kv_mode_checked(args.effective_kv_mode())?;

    let mode = classify_eval_mode(&args)?;
    dispatch_eval(mode, args)
}

/// argus-eval 의 dispatch 모드. mode 우선순위는 legacy main 분기 순서를 보존:
/// quant-window-eval → quant-window-ppl → dump_importance → eval_ll → ppl → experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalMode {
    /// `--eval-ll` (Standard KV).
    EvalLl,
    /// `--eval-ll --kv-mode kivi`.
    EvalLlQuantWindow,
    /// `--ppl` (Standard KV).
    Ppl,
    /// `--ppl --kv-mode kivi`.
    PplQuantWindow,
    /// `--dump-importance`.
    DumpImportance,
    /// `--experiment-schedule` — 정적 schedule 을 `ScheduleCommandSource` 로 실행.
    Experiment,
}

/// args 로부터 정확히 1개 eval 모드를 결정한다.
///
/// mode 게이트 flag: `--eval-ll`(+continuation/batch) / `--ppl` /
/// `--dump-importance` / `--experiment-schedule`. 정확히 1개 모드만 활성이어야
/// 한다 — 0개·복수면 bail(안내). `--qcf-dump` 는 modifier 라 mode 카운트에 포함
/// 하지 않는다. quant-window 변형은 mode 결정 후 `effective_kv_mode()` 로 분기.
fn classify_eval_mode(args: &Args) -> anyhow::Result<EvalMode> {
    let eval_ll_active =
        args.eval_ll || args.eval_batch.is_some() || args.eval_continuation.is_some();
    let ppl_active = args.ppl.is_some();
    let dump_active = args.dump_importance;
    let experiment_active = args.experiment_schedule.is_some();

    let n_modes = eval_ll_active as usize
        + ppl_active as usize
        + dump_active as usize
        + experiment_active as usize;
    if n_modes == 0 {
        bail!(
            "argus-eval: no eval mode selected. Pass exactly one of: \
             --eval-ll (with --eval-batch or --eval-continuation), --ppl <text>, \
             --dump-importance, --experiment-schedule."
        );
    }
    if n_modes > 1 {
        bail!(
            "argus-eval: eval modes are mutually exclusive (selected {}). Pass exactly one of \
             --eval-ll / --ppl / --dump-importance / --experiment-schedule.",
            n_modes
        );
    }

    // 우선순위 보존: experiment(γ-3b) 는 별도 bail, quant-window 변형은 kv-mode 로 분기.
    if experiment_active {
        return Ok(EvalMode::Experiment);
    }
    // site #4: classify by declared cap (`ModeCaps.is_quantized_kv`), not a concrete
    // `matches!(.., KvMode::QuantWindow)`. The `EvalMode::*QuantWindow` variant *names* are
    // eval-harness residue (deferred — orthogonal to the selection mechanism).
    let quantized = mode_caps(args.effective_kv_mode()).is_some_and(|c| c.is_quantized_kv);
    if eval_ll_active {
        return Ok(if quantized {
            EvalMode::EvalLlQuantWindow
        } else {
            EvalMode::EvalLl
        });
    }
    if ppl_active {
        return Ok(if quantized {
            EvalMode::PplQuantWindow
        } else {
            EvalMode::Ppl
        });
    }
    debug_assert!(dump_active);
    Ok(EvalMode::DumpImportance)
}

/// 결정된 모드를 해당 runner 로 dispatch 한다.
fn dispatch_eval(mode: EvalMode, args: Args) -> anyhow::Result<()> {
    match mode {
        EvalMode::EvalLl => {
            let ctx = eval_setup::build_eval_ll_ctx(args)?;
            argus_engine::session::eval::run_eval_ll(ctx)
        }
        EvalMode::EvalLlQuantWindow => eval_setup::run_eval_ll_quant_window(args),
        EvalMode::Ppl => {
            let ctx = eval_setup::build_ppl_ctx(args)?;
            argus_engine::session::ppl::run_ppl_dispatch(ctx)
        }
        EvalMode::PplQuantWindow => {
            let ppl_path = args
                .ppl
                .clone()
                .expect("PplKivi mode implies args.ppl.is_some()");
            eval_setup::run_ppl_quant_window(args, &ppl_path)
        }
        EvalMode::DumpImportance => {
            let ctx = eval_setup::build_dump_importance_ctx(args)?;
            argus_engine::session::dump_importance::run_dump_importance(ctx)
        }
        EvalMode::Experiment => {
            let schedule_path = args
                .experiment_schedule
                .clone()
                .expect("Experiment mode implies experiment_schedule.is_some()");
            let schedule = ExperimentSchedule::load(&schedule_path)?;
            let scs = ScheduleCommandSource::new(schedule);
            let ctx = build_inference_ctx(args)?;
            run_experiment_schedule_path(ctx, scs)
        }
    }
}

/// bin-local 허용목록 가드. argus_bench 의 `bench_supported`(argus_bench.rs:60)
/// 패턴 미러. eval/ppl 의 핵심 사용례(eviction/qcf/skip/swap/quant-window)는 허용하고,
/// eval 측정과 무관·충돌하는 모드만 차단한다.
///
/// 허용: eviction-policy, qcf-dump, skip-ratio/skip-layers, weight swap 계열,
/// quant-window kv-mode. 차단(→ `reject_unsupported_modes_eval` 가 안내): profile,
/// tensor-partition, chat.
fn eval_supported(args: &Args) -> bool {
    !args.profile
        && !args.profile_events
        && args.tensor_partition == 0.0
        && !args.chat
        && args.chat_socket.is_none()
        && args.chat_tcp.is_none()
}

/// eval 표면 밖 모드를 명시 reject 하며 행선지를 안내한다 (argus_cli 가드 미러).
/// eviction/qcf/skip/swap/quant-window 는 `eval_supported` 가 통과시키므로 reject 하지
/// 않는다.
fn reject_unsupported_modes_eval(args: &Args) -> anyhow::Result<()> {
    if args.chat {
        bail!("argus-eval: --chat moved to argus-chat (planned)");
    }
    if args.chat_socket.is_some() || args.chat_tcp.is_some() {
        bail!("argus-eval: --chat-socket / --chat-tcp moved to argus-chat (planned)");
    }
    if args.profile || args.profile_events {
        bail!(
            "argus-eval: --profile / --profile-events oversimplify eval measurement (sync overhead); not supported"
        );
    }
    if args.tensor_partition > 0.0 {
        bail!("argus-eval: --tensor-partition is a decode-only measurement mode, not eval");
    }
    // W-ALLOC honesty: `--kv-format` is a global Args flag but only argus-cli/argus-bench honor a
    // per-layer KV format POLICY (N-way mixed precision); eval allocates a uniform KV dtype. Fail
    // fast on a policy name instead of silently dropping it to uniform (no-silent-no-op contract).
    if let Some(fmt) = args.kv_format.as_deref().filter(|s| !s.is_empty())
        && argus_engine::format::is_registered_kv_format_policy(fmt)
    {
        bail!(
            "argus-eval: --kv-format '{fmt}' is a per-layer KV format policy (N-way mixed precision) \
             supported only by argus-cli / argus-bench; eval uses a uniform KV dtype. Use --kv-type \
             for eval, or run mixed precision via argus-cli / argus-bench."
        );
    }
    // Generic diagnostic dumps (`--dump <kind>`): validate kind names, require an
    // output directory, and enforce the eval-ll + CPU-backend preconditions.
    // Read-only (INV-147); a requested-but-unsupported dump must fail fast rather
    // than be silently dropped (no-silent-no-op contract).
    let dump_kinds = args.dump_kinds();
    if !dump_kinds.is_empty() {
        argus_engine::session::eval::dump::validate_dump_kinds(dump_kinds)?;
        if args.dump_dir.is_none() {
            bail!(
                "argus-eval: --dump <kinds> requires --dump-dir <dir> \
                 (each kind writes <dir>/<kind>.jsonl)"
            );
        }
        let eval_ll_active =
            args.eval_ll || args.eval_batch.is_some() || args.eval_continuation.is_some();
        if !eval_ll_active {
            bail!(
                "argus-eval: --dump is only supported with --eval-ll \
                 (--eval-batch / --eval-continuation)"
            );
        }
        if mode_caps(args.effective_kv_mode()).is_some_and(|c| c.is_quantized_kv) {
            bail!(
                "argus-eval: --dump is not supported with a quantized KV mode (--kv-mode); \
                 use the standard --eval-ll path"
            );
        }
        // Both dumps capture per-layer attention on the CPU path only (the GPU flash /
        // GPU score kernels short-circuit it), so require a CPU backend rather than
        // emit a buffer of zeros.
        if args.backend != "cpu" {
            bail!(
                "argus-eval: --dump requires --backend cpu \
                 (per-layer attention capture is CPU-only)"
            );
        }
        // evict_importance profiles a real eviction event — it needs an eviction policy.
        if args.dump_enabled(argus_engine::session::eval::dump::DUMP_EVICT_IMPORTANCE) {
            if args.eviction_policy() == "none" {
                bail!(
                    "argus-eval: --dump evict_importance requires an eviction policy \
                     (e.g. `eviction plugin --name h2o`) and a KV budget to profile"
                );
            }
            // The per-layer-head buffer is filled only for tracked layers; a restricted
            // tracking window (`--set tracked_layers=N`) would leave untracked layers as
            // zeros — a structurally-valid but mostly-empty buffer that silently corrupts
            // the IMP-3 analysis. Require full-layer tracking (the default) rather than
            // force it (which would change the policy's own ranking → unfaithful dump).
            if args.h2o_tracked_layers() != 0 {
                bail!(
                    "argus-eval: --dump evict_importance requires all layers tracked; \
                     remove `--set tracked_layers=N` (the dump needs full per-layer resolution)"
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// 최소 인자로 Args 를 만들고 클로저로 모드 flag 를 설정한다.
    fn make_args(extra: &[&str]) -> Args {
        let mut argv = vec!["argus-eval", "--model-path", "/tmp/model.gguf"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).expect("Args parse")
    }

    // ── eval_supported: 허용 케이스 ──────────────────────────────────
    #[test]
    fn eval_supported_allows_eviction_qcf_skip() {
        // eviction 은 `eviction <policy>` nested subcommand — flag 뒤(끝)에 둔다.
        let args = make_args(&[
            "--eval-ll",
            "--skip-ratio",
            "0.2",
            "--qcf-dump",
            "/tmp/q.json",
            "eviction",
            "plugin",
            "--name",
            "h2o",
        ]);
        assert!(eval_supported(&args));
        assert_eq!(args.eviction_policy(), "h2o");
    }

    #[test]
    fn eval_supported_allows_quant_window() {
        let args = make_args(&["--ppl", "/tmp/ref.txt", "--kv-mode", "kivi"]);
        assert!(eval_supported(&args));
    }

    #[test]
    fn eval_supported_allows_plain_eval_ll() {
        let args = make_args(&["--eval-ll", "--eval-continuation", "world"]);
        assert!(eval_supported(&args));
    }

    // ── eval_supported: 차단 케이스 ──────────────────────────────────
    #[test]
    fn eval_supported_blocks_profile() {
        let args = make_args(&["--eval-ll", "--profile"]);
        assert!(!eval_supported(&args));
    }

    #[test]
    fn eval_supported_blocks_tensor_partition() {
        let args = make_args(&["--ppl", "/tmp/ref.txt", "--tensor-partition", "0.5"]);
        assert!(!eval_supported(&args));
    }

    #[test]
    fn eval_supported_blocks_chat() {
        let args = make_args(&["--eval-ll", "--chat"]);
        assert!(!eval_supported(&args));
    }

    // ── reject_unsupported_modes_eval ────────────────────────────────
    #[test]
    fn reject_blocks_profile_with_message() {
        let args = make_args(&["--eval-ll", "--profile"]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        assert!(err.to_string().contains("--profile"));
    }

    /// W-ALLOC honesty: a per-layer KV format policy name is rejected (fail-fast), not silently
    /// dropped to a uniform dtype (eval does not honor `--kv-format` policies).
    #[test]
    fn reject_blocks_kv_format_policy() {
        let args = make_args(&["--eval-ll", "--kv-format", "mixed_precision"]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        assert!(err.to_string().contains("mixed_precision"));
    }

    /// A plain builtin format name is NOT a policy → not rejected by this guard.
    #[test]
    fn reject_allows_non_policy_kv_format() {
        let args = make_args(&["--eval-ll", "--kv-format", "f16"]);
        assert!(reject_unsupported_modes_eval(&args).is_ok());
    }

    #[test]
    fn reject_allows_eviction_and_qcf() {
        let args = make_args(&[
            "--eval-ll",
            "--qcf-dump",
            "/tmp/q.json",
            "eviction",
            "plugin",
            "--name",
            "sliding",
        ]);
        assert!(reject_unsupported_modes_eval(&args).is_ok());
        assert_eq!(args.eviction_policy(), "sliding");
    }

    // ── classify_eval_mode: 상호배제 + 우선순위 ──────────────────────
    #[test]
    fn classify_eval_ll_standard() {
        let args = make_args(&["--eval-ll", "--eval-continuation", "x"]);
        assert_eq!(classify_eval_mode(&args).unwrap(), EvalMode::EvalLl);
    }

    #[test]
    fn classify_eval_ll_quant_window() {
        let args = make_args(&["--eval-ll", "--eval-continuation", "x", "--kv-mode", "kivi"]);
        assert_eq!(
            classify_eval_mode(&args).unwrap(),
            EvalMode::EvalLlQuantWindow
        );
    }

    #[test]
    fn classify_ppl_standard() {
        let args = make_args(&["--ppl", "/tmp/ref.txt"]);
        assert_eq!(classify_eval_mode(&args).unwrap(), EvalMode::Ppl);
    }

    #[test]
    fn classify_ppl_quant_window() {
        let args = make_args(&["--ppl", "/tmp/ref.txt", "--kv-mode", "kivi"]);
        assert_eq!(classify_eval_mode(&args).unwrap(), EvalMode::PplQuantWindow);
    }

    #[test]
    fn classify_dump_importance() {
        let args = make_args(&["--dump-importance"]);
        assert_eq!(classify_eval_mode(&args).unwrap(), EvalMode::DumpImportance);
    }

    #[test]
    fn classify_experiment() {
        let args = make_args(&["--experiment-schedule", "/tmp/s.json"]);
        assert_eq!(classify_eval_mode(&args).unwrap(), EvalMode::Experiment);
    }

    #[test]
    fn classify_zero_modes_bails() {
        let args = make_args(&[]);
        assert!(classify_eval_mode(&args).is_err());
    }

    #[test]
    fn classify_multiple_modes_bails() {
        let args = make_args(&["--eval-ll", "--dump-importance"]);
        let err = classify_eval_mode(&args).unwrap_err();
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn classify_ppl_and_eval_ll_bails() {
        let args = make_args(&["--eval-ll", "--ppl", "/tmp/ref.txt"]);
        assert!(classify_eval_mode(&args).is_err());
    }

    /// qcf-dump 는 modifier — mode 카운트에 포함하지 않으므로 ll 단독은 EvalLl.
    #[test]
    fn classify_qcf_dump_is_modifier_not_mode() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--qcf-dump",
            "/tmp/q.json",
        ]);
        assert_eq!(classify_eval_mode(&args).unwrap(), EvalMode::EvalLl);
    }

    // ── generic --dump diagnostic guard ──────────────────────────────────
    #[test]
    fn reject_dump_unknown_kind() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--dump",
            "bogus",
            "--dump-dir",
            "/tmp/d",
        ]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        assert!(err.to_string().contains("bogus"), "{}", err);
    }

    #[test]
    fn reject_dump_without_dump_dir() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--dump",
            "answer_attention",
        ]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        assert!(err.to_string().contains("--dump-dir"), "{}", err);
    }

    #[test]
    fn reject_dump_answer_attention_requires_cpu_backend() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--dump",
            "answer_attention",
            "--dump-dir",
            "/tmp/d",
            "--backend",
            "opencl",
        ]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        assert!(err.to_string().contains("cpu"), "{}", err);
    }

    #[test]
    fn reject_dump_requires_eval_ll_mode() {
        // --ppl is not an eval-ll mode → dump is rejected (no silent no-op).
        let args = make_args(&[
            "--ppl",
            "/tmp/ref.txt",
            "--dump",
            "answer_attention",
            "--dump-dir",
            "/tmp/d",
        ]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        assert!(err.to_string().contains("--eval-ll"), "{}", err);
    }

    #[test]
    fn reject_dump_with_quant_window_kv_mode() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--dump",
            "answer_attention",
            "--dump-dir",
            "/tmp/d",
            "--kv-mode",
            "kivi",
        ]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("quantized") || msg.contains("kv-mode"),
            "{}",
            err
        );
    }

    /// Happy path: answer_attention on cpu + eval-ll + dump-dir passes the guard,
    /// and the path accessor resolves to `<dir>/<kind>.jsonl`.
    #[test]
    fn accept_dump_answer_attention_cpu_eval_ll() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--dump",
            "answer_attention",
            "--dump-dir",
            "/tmp/d",
        ]);
        // Default backend on the host (non-android) is "cpu".
        assert!(reject_unsupported_modes_eval(&args).is_ok());
        assert!(args.dump_enabled("answer_attention"));
        assert_eq!(
            args.dump_path("answer_attention").unwrap(),
            std::path::PathBuf::from("/tmp/d/answer_attention.jsonl")
        );
    }

    /// No `--dump` → guard is a no-op and the accessors report nothing enabled
    /// (production path untouched — INV-147).
    #[test]
    fn no_dump_is_inert() {
        let args = make_args(&["--eval-ll", "--eval-continuation", "x"]);
        assert!(reject_unsupported_modes_eval(&args).is_ok());
        assert!(args.dump_kinds().is_empty());
        assert!(!args.dump_enabled("answer_attention"));
        assert_eq!(args.dump_path("answer_attention"), None);
    }

    #[test]
    fn reject_dump_evict_importance_without_eviction_policy() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--dump",
            "evict_importance",
            "--dump-dir",
            "/tmp/d",
        ]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        assert!(err.to_string().contains("eviction policy"), "{}", err);
    }

    #[test]
    fn accept_dump_evict_importance_with_eviction_policy() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--dump",
            "evict_importance",
            "--dump-dir",
            "/tmp/d",
            "eviction",
            "plugin",
            "--name",
            "h2o",
        ]);
        assert!(reject_unsupported_modes_eval(&args).is_ok());
        assert!(args.dump_enabled("evict_importance"));
        assert_eq!(
            args.dump_path("evict_importance").unwrap(),
            std::path::PathBuf::from("/tmp/d/evict_importance.jsonl")
        );
    }

    /// A restricted tracking window would leave untracked layers as zeros in the
    /// dump → reject (the dump needs full per-layer resolution).
    #[test]
    fn reject_dump_evict_importance_with_partial_tracked_layers() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--dump",
            "evict_importance",
            "--dump-dir",
            "/tmp/d",
            "eviction",
            "plugin",
            "--name",
            "h2o",
            "--set",
            "tracked_layers=8",
        ]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        assert!(err.to_string().contains("all layers tracked"), "{}", err);
    }

    /// Both dump kinds require a CPU backend (per-layer capture is CPU-only).
    #[test]
    fn reject_dump_evict_importance_requires_cpu() {
        let args = make_args(&[
            "--eval-ll",
            "--eval-continuation",
            "x",
            "--dump",
            "evict_importance",
            "--dump-dir",
            "/tmp/d",
            "--backend",
            "opencl",
            "eviction",
            "plugin",
            "--name",
            "h2o",
        ]);
        let err = reject_unsupported_modes_eval(&args).unwrap_err();
        assert!(err.to_string().contains("cpu"), "{}", err);
    }
}
