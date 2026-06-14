//! Phase 4-B-2: `run_eval_ll` — `bin/generate.rs::main()`의 eval_ll 분기
//! (l.1642~1992) 본문을 외과적으로 이동.
//!
//! lift-and-shift: BatchRunCtx와 동일 패턴. EvalLlRunCtx field destructure
//! 후 local var로 풀어 본문이 원본 outer-scope를 그대로 참조하던 패턴 보존.

use anyhow::Result;

use crate::session::cli::parse_qcf_sample_layers;
use crate::session::eval::args::EvalLlRunCtx;
use crate::session::eval::helpers::{build_eval_ll_warmup_text, load_eval_questions};
use crate::session::qcf_runtime::{
    QcfWarmupConfig, QcfWarmupCtx, run_layer_swap, run_qcf_warmup_workflow,
};

pub fn run_eval_ll(ctx: EvalLlRunCtx) -> Result<()> {
    let EvalLlRunCtx {
        args,
        backend,
        memory,
        cpu_backend_arc,
        gpu_backend_arc,
        model,
        tokenizer,
        mut kv_caches,
        cache_manager,
        score_accumulator,
        skip_config,
        prompt,
        hidden_size,
        vocab_size,
        max_seq_len,
        num_layers,
        kv_type: _kv_type,
        actual_protected_prefix,
        score_based_eviction,
        swap_algorithm,
        importance_formula,
        importance_compare,
        swap_only_layers,
    } = ctx;

    let questions = load_eval_questions(&args, &prompt)?;

    // ── QCF-dump prelude: --eval-ll + --qcf-dump + --force-swap-ratio ────
    // When all three flags are active we run warmup prefill → ImportanceTable
    // → WeightSwapDecider → SwapExecutor before the eval loop.  This mirrors
    // the PPL/generation QCF-dump workflow (line ~2417) but uses the eval
    // questions' prompt text instead of a corpus file for the warmup input.
    let eval_ll_qcf_start = std::time::Instant::now();
    let mut eval_ll_qcf_importance: Option<crate::qcf::ImportanceTable> = None;
    let mut eval_ll_qcf_decision: Option<crate::weight::decider::SwapDecision> = None;
    let mut eval_ll_qcf_dpllm_epsilon: Option<Vec<f32>> = None;
    let mut eval_ll_qcf_dpllm_epsilon_multi: Option<Vec<f32>> = None;
    let mut eval_ll_qcf_dpllm_epsilon_abs: Option<Vec<f32>> = None;
    let mut eval_ll_qcf_dpllm_epsilon_qcf: Option<Vec<f32>> = None;
    let mut eval_ll_qcf_direct_attn_f4: Option<Vec<f32>> = None;
    let mut eval_ll_qcf_direct_attn_f5: Option<Vec<f32>> = None;
    let mut eval_ll_qcf_direct_attn_f5_decode_only: Option<Vec<f32>> = None;
    let mut eval_ll_qcf_direct_attn_f5_prefill_decode: Option<Vec<f32>> = None;

    if args.qcf_dump.is_some()
        && let Some(force_ratio) = args.force_swap_ratio
    {
        let warmup_n = args.qcf_warmup_tokens.max(1);
        // Concatenate question prompts (separated by "\n\n") and take the
        // first warmup_n tokens. Empty result → soft skip (no abort).
        let warmup_ids = build_eval_ll_warmup_text(&questions, warmup_n, &tokenizer);

        if warmup_ids.is_empty() {
            eprintln!(
                "[QCF-dump] WARNING: eval-ll warmup token sequence is empty; \
                 prelude skipped (swap will use uniform fallback)"
            );
        } else {
            let result = run_qcf_warmup_workflow(
                QcfWarmupCtx {
                    model: &model,
                    backend: &backend,
                    memory: memory.as_ref(),
                    kv_caches: &mut kv_caches,
                    vocab_size,
                    warmup_ids: &warmup_ids,
                    gpu_backend: gpu_backend_arc.as_ref(),
                    cpu_backend: &cpu_backend_arc,
                },
                QcfWarmupConfig {
                    force_ratio: Some(force_ratio),
                    swap_algorithm,
                    execute_swap: !args.qcf_trajectory,
                    importance_formula,
                    importance_three_way: importance_compare,
                    swap_only_layers: swap_only_layers.as_deref(),
                    decode_x_steps: args.decode_x_steps,
                    log_prefix: " eval-ll",
                },
            )?;
            eval_ll_qcf_decision = result.decision;
            eval_ll_qcf_importance = Some(result.importance);
            eval_ll_qcf_dpllm_epsilon = result.dpllm_epsilon;
            eval_ll_qcf_dpllm_epsilon_multi = result.dpllm_epsilon_multi;
            eval_ll_qcf_dpllm_epsilon_abs = result.dpllm_epsilon_abs;
            eval_ll_qcf_dpllm_epsilon_qcf = result.dpllm_epsilon_qcf;
            eval_ll_qcf_direct_attn_f4 = result.direct_attn_f4;
            eval_ll_qcf_direct_attn_f5 = result.direct_attn_f5;
            eval_ll_qcf_direct_attn_f5_decode_only = result.direct_attn_f5_decode_only;
            eval_ll_qcf_direct_attn_f5_prefill_decode = result.direct_attn_f5_prefill_decode;
        }
    }

    let ratio_mode = args.kv_budget_ratio() > 0.0;
    let budget_mode = args.kv_budget() > 0 || ratio_mode;

    // For ratio mode, effective_budget is computed per-question inside eval_loop.
    // Pass 0 here; the loop will use kv_budget_ratio × prompt_len.
    let effective_budget = if ratio_mode { 0 } else { args.kv_budget() };

    eprintln!(
        "[Eval-LL] {} questions, policy={}, kv_budget={}, kv_budget_ratio={}, mode={}",
        questions.len(),
        args.eviction_policy(),
        args.kv_budget(),
        args.kv_budget_ratio(),
        if budget_mode {
            if ratio_mode {
                "ratio-per-question"
            } else {
                "chunked"
            }
        } else {
            "full-prefill"
        }
    );

    let qcf_mode_enum = match args.qcf_mode.as_str() {
        "caote" => crate::qcf_types::QcfMode::Caote,
        "both" => crate::qcf_types::QcfMode::Both,
        _ => crate::qcf_types::QcfMode::Attn,
    };
    let qcf_config = crate::qcf_types::QcfConfig {
        mode: qcf_mode_enum,
        ..crate::qcf_types::QcfConfig::default()
    };

    let eval_config = crate::session::eval::EvalConfig {
        max_seq_len,
        effective_budget,
        kv_budget_ratio: args.kv_budget_ratio(),
        greedy: args.greedy,
        kv_type: args.kv_type.clone(),
        qcf_mode: args.qcf_mode.clone(),
        vocab_size,
        hidden_size,
    };

    // For ratio mode, hook starts with budget=0; eval_loop updates it per-question.
    let hook_budget = if ratio_mode { 0 } else { effective_budget };
    let is_d2o = args.eviction_policy() == "d2o";

    // ARGUS Step 6: resolve --qcf-sample-layers from CLI.
    // When --enable-qcf-experimental is off, always use [0] (legacy, no overhead).
    let eviction_hook_sample_layers = if args.enable_qcf_experimental {
        parse_qcf_sample_layers(&args.qcf_sample_layers, num_layers)
            .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
    } else {
        vec![0]
    };

    let mut hook = crate::session::eval::EvictionHook::new(
        cache_manager,
        score_accumulator,
        qcf_config,
        hook_budget,
        actual_protected_prefix,
        score_based_eviction,
        args.h2o_keep_ratio(),
        is_d2o,
        args.kv_type.clone(),
        backend.clone(),
        args.enable_qcf_experimental,
        eviction_hook_sample_layers,
    );

    // ── Trajectory mode dispatch ──────────────────────────────────────────
    // When `--qcf-trajectory` is active alongside `--qcf-dump` and
    // `--force-swap-ratio`, we run eval-ll K+1 times (K = decision layer
    // count): step 0 with no swap (baseline), then step t (1..=K) after
    // cumulatively applying `selected_layers[t-1]`. Each step's full
    // EvalOutput is captured into `trajectory_outputs` and emitted under
    // the `trajectory` field of the dump JSON.
    let trajectory_mode = args.qcf_trajectory
        && args.qcf_dump.is_some()
        && args.force_swap_ratio.is_some()
        && eval_ll_qcf_decision
            .as_ref()
            .map(|d| !d.selected_layers.is_empty())
            .unwrap_or(false);
    let ordered_layers: Vec<usize> = if trajectory_mode {
        eval_ll_qcf_decision
            .as_ref()
            .unwrap()
            .selected_layers
            .clone()
    } else {
        Vec::new()
    };
    let n_steps = if trajectory_mode {
        ordered_layers.len() + 1
    } else {
        1
    };
    let mut trajectory_outputs: Vec<crate::session::eval::EvalOutput> = Vec::with_capacity(n_steps);

    if trajectory_mode {
        eprintln!(
            "[QCF-trajectory] mode enabled: K={} (algo={}, ratio={:.2})",
            ordered_layers.len(),
            swap_algorithm.short_name(),
            args.force_swap_ratio.unwrap_or(0.0),
        );
    }

    for step in 0..n_steps {
        if trajectory_mode {
            eprintln!(
                "[QCF-trajectory] step {}/{}: cumulative swap = {:?}",
                step,
                ordered_layers.len(),
                &ordered_layers[..step]
            );
        }

        let step_out = crate::session::eval::run_eval_ll_generic(
            &model,
            &tokenizer,
            &backend,
            &*memory,
            &mut kv_caches,
            &mut hook,
            &questions,
            &eval_config,
            skip_config.as_ref(),
        )?;
        trajectory_outputs.push(step_out);

        if trajectory_mode && step < ordered_layers.len() {
            let layer_to_swap = ordered_layers[step];
            let report = run_layer_swap(
                &model,
                &[layer_to_swap],
                gpu_backend_arc.as_ref(),
                &cpu_backend_arc,
                None,
                #[cfg(feature = "opencl")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "[QCF-trajectory] swap layer {} failed: {}",
                    layer_to_swap,
                    e
                )
            })?;
            eprintln!(
                "[QCF-trajectory] swapped layer {}: latency {:.1}ms",
                layer_to_swap, report.latency_ms
            );
        }
    }

    // For downstream non-trajectory stdout printing, expose the last step's
    // EvalOutput as `output` (in non-trajectory mode this is the only step).
    let output = trajectory_outputs
        .last()
        .expect("at least one eval-ll step ran")
        .clone();

    // ── QCF-dump JSON (eval-ll mode) ──────────────────────────────────────
    if let Some(ref dump_path) = args.qcf_dump {
        use crate::session::eval::qcf_helpers::{
            QcfSwapDumpContext, TrajectoryStep, dump_qcf_swap_json,
        };

        let empty_swap: Vec<usize> = Vec::new();
        let (swap_set, qcf_predicted, fallback_used) = if let Some(ref dec) = eval_ll_qcf_decision {
            (
                dec.selected_layers.as_slice(),
                dec.qcf_swap_estimate,
                dec.fallback_used,
            )
        } else {
            (empty_swap.as_slice(), 0.0f32, false)
        };

        let secondary_path_str = args.secondary_gguf.as_ref().and_then(|p| p.to_str());
        let model_arch = if args.model_path.to_lowercase().contains("qwen") {
            "qwen2"
        } else {
            "llama"
        };
        let total_wall = eval_ll_qcf_start.elapsed().as_secs_f64();

        // Build trajectory steps when in trajectory mode.
        let trajectory_steps: Vec<TrajectoryStep> = if trajectory_mode {
            trajectory_outputs
                .iter()
                .enumerate()
                .map(|(t, eo)| TrajectoryStep {
                    step: t,
                    swapped_layers: ordered_layers[..t].to_vec(),
                    layer_added: if t > 0 {
                        Some(ordered_layers[t - 1])
                    } else {
                        None
                    },
                    eval_ll_output: eo,
                })
                .collect()
        } else {
            Vec::new()
        };
        let trajectory_ref: Option<&[TrajectoryStep]> = if trajectory_mode {
            Some(trajectory_steps.as_slice())
        } else {
            None
        };
        let eval_ll_output_ref = if trajectory_mode { None } else { Some(&output) };

        let ctx = QcfSwapDumpContext {
            model_arch,
            model_path: &args.model_path,
            secondary_path: secondary_path_str,
            primary_dtype: "F16",
            secondary_dtype: "Q4_0",
            num_layers: model.layers.len(),
            force_swap_ratio: args.force_swap_ratio,
            swap_algorithm: args.force_swap_ratio.map(|_| swap_algorithm.short_name()),
            swap_set,
            qcf_swap_predicted: qcf_predicted,
            fallback_used,
            importance_table: eval_ll_qcf_importance.as_ref(),
            noise_table: Some(model.quant_noise.as_ref()),
            ppl: None,
            avg_nll: None,
            n_eval_tokens: 0,
            wall_time_s: total_wall,
            warmup_tokens: args.qcf_warmup_tokens,
            backend: &args.backend,
            kv_type: &args.kv_type,
            ppl_corpus: None,
            eval_ll_output: eval_ll_output_ref,
            trajectory: trajectory_ref,
            dpllm_epsilon: eval_ll_qcf_dpllm_epsilon.as_deref(),
            dpllm_epsilon_multi: eval_ll_qcf_dpllm_epsilon_multi.as_deref(),
            dpllm_epsilon_abs: eval_ll_qcf_dpllm_epsilon_abs.as_deref(),
            dpllm_epsilon_qcf: eval_ll_qcf_dpllm_epsilon_qcf.as_deref(),
            direct_attn_f4: eval_ll_qcf_direct_attn_f4.as_deref(),
            direct_attn_f5: eval_ll_qcf_direct_attn_f5.as_deref(),
            direct_attn_f5_decode_only: eval_ll_qcf_direct_attn_f5_decode_only.as_deref(),
            direct_attn_f5_prefill_decode: eval_ll_qcf_direct_attn_f5_prefill_decode.as_deref(),
        };

        dump_qcf_swap_json(dump_path, &ctx)?;
        eprintln!(
            "[QCF-dump] eval-ll JSON written to {}{}",
            dump_path.display(),
            if trajectory_mode {
                " (trajectory schema_v2)"
            } else {
                ""
            }
        );
    }

    let mut json_val = serde_json::from_str::<serde_json::Value>(&output.to_json()?)?;
    json_val["config"] = serde_json::json!({
        "model": args.model_path,
        "eviction_policy": args.eviction_policy(),
        "kv_budget": args.kv_budget(),
        "kv_budget_ratio": args.kv_budget_ratio(),
        "max_seq_len": max_seq_len,
        "kv_type": args.kv_type,
        "h2o_keep_ratio": args.h2o_keep_ratio(),
        "h2o_decay": args.h2o_decay(),
        "time_normalized": !args.h2o_raw_scores(),
        "skip_layers": args.skip_layers,
        "skip_ratio": args.skip_ratio,
    });
    println!("{}", serde_json::to_string_pretty(&json_val)?);
    Ok(())
}
