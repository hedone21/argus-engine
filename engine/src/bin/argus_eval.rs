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
//! KIVI(`--kv-mode kivi`)는 ll/ppl 양쪽에서 지원 — 별 `KiviCache` 경로(§13.6).
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
use argus_engine::session::cli::{Args, KvMode};
use argus_engine::session::eval_setup;
use argus_engine::session::experiment::ScheduleCommandSource;
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

    let mode = classify_eval_mode(&args)?;
    dispatch_eval(mode, args)
}

/// argus-eval 의 dispatch 모드. mode 우선순위는 legacy main 분기 순서를 보존:
/// KIVI-eval → KIVI-ppl → dump_importance → eval_ll → ppl → experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalMode {
    /// `--eval-ll` (Standard KV).
    EvalLl,
    /// `--eval-ll --kv-mode kivi`.
    EvalLlKivi,
    /// `--ppl` (Standard KV).
    Ppl,
    /// `--ppl --kv-mode kivi`.
    PplKivi,
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
/// 하지 않는다. KIVI 변형은 mode 결정 후 `effective_kv_mode()` 로 분기.
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

    // 우선순위 보존: experiment(γ-3b) 는 별도 bail, KIVI 변형은 kv-mode 로 분기.
    if experiment_active {
        return Ok(EvalMode::Experiment);
    }
    let is_kivi = matches!(args.effective_kv_mode(), KvMode::Kivi);
    if eval_ll_active {
        return Ok(if is_kivi {
            EvalMode::EvalLlKivi
        } else {
            EvalMode::EvalLl
        });
    }
    if ppl_active {
        return Ok(if is_kivi {
            EvalMode::PplKivi
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
        EvalMode::EvalLlKivi => eval_setup::run_eval_ll_kivi(args),
        EvalMode::Ppl => {
            let ctx = eval_setup::build_ppl_ctx(args)?;
            argus_engine::session::ppl::run_ppl_dispatch(ctx)
        }
        EvalMode::PplKivi => {
            let ppl_path = args
                .ppl
                .clone()
                .expect("PplKivi mode implies args.ppl.is_some()");
            eval_setup::run_ppl_kivi(args, &ppl_path)
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
/// 패턴 미러. eval/ppl 의 핵심 사용례(eviction/qcf/skip/swap/KIVI)는 허용하고,
/// eval 측정과 무관·충돌하는 모드만 차단한다.
///
/// 허용: eviction-policy, qcf-dump, skip-ratio/skip-layers, weight swap 계열,
/// KIVI kv-mode. 차단(→ `reject_unsupported_modes_eval` 가 안내): profile,
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
/// eviction/qcf/skip/swap/KIVI 는 `eval_supported` 가 통과시키므로 reject 하지
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
            "h2o",
        ]);
        assert!(eval_supported(&args));
        assert_eq!(args.eviction_policy(), "h2o");
    }

    #[test]
    fn eval_supported_allows_kivi() {
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

    #[test]
    fn reject_allows_eviction_and_qcf() {
        let args = make_args(&[
            "--eval-ll",
            "--qcf-dump",
            "/tmp/q.json",
            "eviction",
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
    fn classify_eval_ll_kivi() {
        let args = make_args(&["--eval-ll", "--eval-continuation", "x", "--kv-mode", "kivi"]);
        assert_eq!(classify_eval_mode(&args).unwrap(), EvalMode::EvalLlKivi);
    }

    #[test]
    fn classify_ppl_standard() {
        let args = make_args(&["--ppl", "/tmp/ref.txt"]);
        assert_eq!(classify_eval_mode(&args).unwrap(), EvalMode::Ppl);
    }

    #[test]
    fn classify_ppl_kivi() {
        let args = make_args(&["--ppl", "/tmp/ref.txt", "--kv-mode", "kivi"]);
        assert_eq!(classify_eval_mode(&args).unwrap(), EvalMode::PplKivi);
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
}
