//! Phase 4-B-2: `run_eval_ll` — `bin/generate.rs::main()`의 eval_ll 분기
//! (l.1642~1992) 본문을 외과적으로 이동.
//!
//! lift-and-shift: BatchRunCtx와 동일 패턴. EvalLlRunCtx field destructure
//! 후 local var로 풀어 본문이 원본 outer-scope를 그대로 참조하던 패턴 보존.

use anyhow::Result;

use crate::inference::signal_runtime::SignalRuntime;
use crate::session::eval::args::EvalLlRunCtx;
use crate::session::eval::helpers::{load_eval_questions, resolve_token_spans_from_text};
/// Whether the eval eviction path routes `policy` to the faithful per-head prefill-keepset (PFA)
/// executor instead of the generic score-fed eviction. True for the happy path (`"none"`) AND for an
/// explicit `eviction plugin --name <pfa_stage>` — the registered `TensorKind::PrefillAttention`
/// -reading stage. The second case is the footgun fix: without it, `eviction plugin --name pyramidkv`
/// falls through to the generic path, which only ever hands the stage a flat `importance()` signal,
/// so pyramidkv silently degrades to the layer-wide fallback (the pyramid BUDGET stays correct, but
/// the SELECTION becomes H2O-style, not SnapKV per-head). Any other policy keeps its own score-fed
/// path. `pfa_stage` = `resolve_prefill_keepset_arming().stage_name`.
fn routes_to_prefill_keepset(policy: &str, pfa_stage: &str) -> bool {
    policy == "none" || policy == pfa_stage
}

pub fn run_eval_ll(ctx: EvalLlRunCtx) -> Result<()> {
    let EvalLlRunCtx {
        args,
        backend,
        memory,
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
        num_layers: _,
        kv_type: _kv_type,
        actual_protected_prefix,
        score_based_eviction,
    } = ctx;

    let mut questions = load_eval_questions(&args, &prompt)?;

    // R2: a fixture may ship needle/gold spans as *text* (model-agnostic) rather
    // than raw token indices; resolve those against each question's own canonical
    // tokenization so positions stay consistent with scoring. Gated on a dump
    // being requested so the no-dump path does zero extra tokenization (INV-147).
    // Raw token positions, when present, are an explicit override (left as-is).
    if !args.dump_kinds().is_empty() {
        resolve_token_spans_from_text(&mut questions, &tokenizer);
    }

    // ── IMP-2: answer_attention diagnostic dump (read-only; INV-147) ──────
    // Standalone pass over a fresh uncompressed reference cache — decoupled from
    // the scoring loop below, so NLL/MC are byte-identical whether it runs or not.
    if args.dump_enabled(crate::session::eval::dump::DUMP_ANSWER_ATTENTION) {
        let out_path = args
            .dump_path(crate::session::eval::dump::DUMP_ANSWER_ATTENTION)
            .ok_or_else(|| anyhow::anyhow!("--dump <kind> requires --dump-dir"))?;
        crate::session::eval::run_answer_attention_dump(
            &model,
            &tokenizer,
            &backend,
            memory.clone(),
            &questions,
            max_seq_len,
            vocab_size,
            &out_path,
        )?;
    }

    // ── IMP-4: answer_attention_steps diagnostic dump (read-only; INV-147) ─
    // Same standalone reference pass as answer_attention, but per-output-step (the trajectory):
    // one JSONL record per (question, step). Decoupled from scoring → NLL/MC byte-identical.
    if args.dump_enabled(crate::session::eval::dump::DUMP_ANSWER_ATTENTION_STEPS) {
        let out_path = args
            .dump_path(crate::session::eval::dump::DUMP_ANSWER_ATTENTION_STEPS)
            .ok_or_else(|| anyhow::anyhow!("--dump <kind> requires --dump-dir"))?;
        crate::session::eval::run_answer_attention_steps_dump(
            &model,
            &tokenizer,
            &backend,
            memory.clone(),
            &questions,
            max_seq_len,
            vocab_size,
            &out_path,
            args.answer_attention_steps_per_head,
            args.answer_attention_steps_full(),
            args.answer_attention_steps_predict_row,
        )?;
    }

    // ── Output-perturbation candidate scoring (read-only; INV-147) ────────
    // Standalone pass on its own uncompressed cache: prefill, then score the candidate pool at the
    // decision point immediately after it. Decoupled from scoring → NLL/MC byte-identical.
    if args.dump_enabled(crate::session::eval::dump::DUMP_APERTURB) {
        let out_path = args
            .dump_path(crate::session::eval::dump::DUMP_APERTURB)
            .ok_or_else(|| anyhow::anyhow!("--dump <kind> requires --dump-dir"))?;
        crate::session::eval::run_aperturb_dump(
            &model,
            &tokenizer,
            &backend,
            memory.clone(),
            &questions,
            max_seq_len,
            vocab_size,
            &out_path,
            args.aperturb_tensor_dir.as_deref(),
            args.aperturb_basis.as_deref(),
            args.aperturb_basis_out.as_deref(),
        )?;
    }

    let ratio_mode = args.kv_budget_ratio() > 0.0;
    let budget_mode = args.kv_budget() > 0 || ratio_mode;

    // For ratio mode, effective_budget is computed per-question inside eval_loop.
    // Pass 0 here; the loop will use kv_budget_ratio × prompt_len.
    let effective_budget = if ratio_mode { 0 } else { args.kv_budget() };

    // REQ-3: warn ONCE at setup if an engine KV budget disagrees with faithful h2o's own absolute
    // hh+recent(+prefix). h2o's `partition` ignores --kv-budget/target_len, so a mismatch silently
    // makes --kv-budget non-authoritative (the cache settles at hh+recent+prefix, not the cap) — see
    // the worked example in the request spec §2(b). No-op for any non-h2o policy / no budget set.
    if let Some(w) = crate::session::cli::h2o_budget_mismatch_warning(
        args.eviction_policy(),
        args.h2o_hh_size(),
        args.h2o_recent_size(),
        actual_protected_prefix,
        args.kv_budget(),
        args.kv_budget_ratio(),
    ) {
        eprintln!("{w}");
    }

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

    // R4: warn early if `--dump evict_importance` was requested without a KV
    // budget — eviction never fires (full-prefill), so the dump would be silently
    // empty. (The policy's keep_ratio does not set a budget; only --kv-budget /
    // --kv-budget-ratio do.)
    if let Some(w) = crate::session::eval::dump::evict_importance_empty_dump_warning(
        args.dump_enabled(crate::session::eval::dump::DUMP_EVICT_IMPORTANCE),
        budget_mode,
    ) {
        eprintln!("{w}");
    }

    // Budget-unit guard for `prefill_streaming` (variant b). The streaming cap `B`
    // must be an *absolute* token count (no ratio, non-zero) — see
    // `EvictTiming::budget_unit_error`. Reject up front rather than silently degrade
    // to a different operating point.
    if let Some(msg) = args
        .evict_timing()
        .budget_unit_error(args.kv_budget(), ratio_mode)
    {
        anyhow::bail!("{msg}");
    }

    // `--evict-timing prefill_end` only changes anything when an eviction actually
    // fires, which (like the dump above) needs a KV budget. Warn rather than silently
    // run today's batched full-prefill under a non-default flag. (Streaming already
    // bailed above if it has no absolute budget, so this only catches prefill_end.)
    if args.evict_timing().accumulates_context_scores()
        && !args.evict_timing().evicts_on_overflow()
        && !budget_mode
    {
        eprintln!(
            "[evict-timing] WARNING: --evict-timing prefill_end has no effect without a KV budget \
             (--kv-budget / --kv-budget-ratio) → no eviction fires → prefill stays batched"
        );
    }

    let eval_config = crate::session::eval::EvalConfig {
        max_seq_len,
        effective_budget,
        kv_budget_ratio: args.kv_budget_ratio(),
        greedy: args.greedy,
        kv_type: args.kv_type.clone(),
        vocab_size,
        hidden_size,
        evict_timing: args.evict_timing(),
        // Faithful-H2O (c): AUTO-on for the h2o policy.
        faithful_h2o: args.eviction_policy() == "h2o",
    };

    // For ratio mode, hook starts with budget=0; eval_loop updates it per-question.
    let hook_budget = if ratio_mode { 0 } else { effective_budget };
    // Whether the selected stage emits weighted merges (weighted-merge) → merge-compensation QCF estimator +
    // K readback. Read off the plugin's StageCaps, not a "d2o" name match (B1-1).
    let produces_merge_plan =
        crate::kv::eviction::stage_registry::stage_produces_merge_plan(args.eviction_policy());

    let mut hook = crate::session::eval::EvictionHook::new(
        cache_manager,
        // Route the eval-ll score signal through the coherence conduit (P1). The GPU half is already
        // armed inside `build_eval_score_accumulator`; the runtime only wraps the CPU accumulator.
        score_accumulator.map(|acc| SignalRuntime::new(Some(acc))),
        hook_budget,
        actual_protected_prefix,
        score_based_eviction,
        args.keep_ratio(),
        produces_merge_plan,
        args.kv_type.clone(),
        backend.clone(),
        args.dump_enabled(crate::session::eval::dump::DUMP_EVICT_IMPORTANCE),
        // variant b: cap the resident cache at the budget and evict per-overflow
        // during token-by-token prefill (vs. a single post_prefill cut).
        args.evict_timing().evicts_on_overflow(),
    );

    // Direction-B unification: a `TensorKind::PrefillAttention`-reading stage (pyramidkv) MUST run
    // through its faithful per-head PFA path, never the generic score-fed eviction path — the latter
    // only ever hands the stage a flat `importance()` signal, so pyramidkv silently degrades to a
    // layer-wide keep-set (the pyramid BUDGET stays correct, but the SELECTION is H2O-style, not
    // SnapKV per-head). Arm the prefill keep-set when the policy is EITHER the happy path ("none") OR
    // an explicit `eviction plugin --name <that same stage>` (which previously fell through to the
    // degraded generic path — the "faithful vs degraded invocation" footgun). `post_prefill` applies
    // the per-head keep-set and RETURNS before the generic eviction, so there is no double eviction;
    // any other policy is unchanged (byte-identical).
    if let Some(arming) = crate::kv::eviction::stage_registry::resolve_prefill_keepset_arming() {
        let policy = args.eviction_policy();
        // `eviction plugin --name pyramidkv` — the explicit invocation of the registered PFA stage.
        let explicit_pfa_plugin = policy == arming.stage_name;
        if routes_to_prefill_keepset(&policy, &arming.stage_name)
            && let Some(stage) =
                crate::kv::eviction::stage_registry::make_prefill_keepset_stage(&arming.stage_name)
        {
            // Keep budget: on the happy path it is `--eviction-target-ratio`; for an explicit plugin
            // the user sets the budget via `--kv-budget-ratio`, so honor that (else the target-ratio
            // default). pyramidkv derives its per-layer pyramid cr from this keep fraction.
            let target_ratio = if explicit_pfa_plugin && args.kv_budget_ratio() > 0.0 {
                args.kv_budget_ratio()
            } else {
                args.eviction_target_ratio()
            };
            eprintln!(
                "[prefill-keepset] '{}' active — PFA producer arms q_window={} \
                 (SnapKV per-head keep-set applied at post_prefill, keep_ratio={target_ratio:.3})",
                arming.stage_name, arming.q_window
            );
            hook = hook.with_prefill_keepset(
                stage,
                arming.q_window,
                model.config.num_attention_heads,
                target_ratio,
                args.protected_prefix()
                    .unwrap_or(arming.default_protected_prefix),
            );
        }
    }

    // IMP-1: open the evict_importance dump writer (one JSONL record per question
    // whose eviction fires). The eval loop drains the hook's captured snapshot.
    let mut evict_writer = if args.dump_enabled(crate::session::eval::dump::DUMP_EVICT_IMPORTANCE) {
        let path = args
            .dump_path(crate::session::eval::dump::DUMP_EVICT_IMPORTANCE)
            .ok_or_else(|| anyhow::anyhow!("--dump <kind> requires --dump-dir"))?;
        Some(crate::session::eval::dump::JsonlDumpWriter::create(path)?)
    } else {
        None
    };

    let output = crate::session::eval::run_eval_ll_generic(
        &model,
        &tokenizer,
        &backend,
        &*memory,
        &mut kv_caches,
        &mut hook,
        &questions,
        &eval_config,
        skip_config.as_ref(),
        evict_writer.as_mut(),
    )?;
    if let Some(writer) = evict_writer {
        let path = writer.path().to_path_buf();
        let n = writer.finish()?;
        eprintln!(
            "[dump:evict_importance] wrote {} record(s) → {}",
            n,
            path.display()
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
        "h2o_keep_ratio": args.keep_ratio(),
        "h2o_decay": args.h2o_decay(),
        // Faithful-H2O: time_normalize is forced OFF for h2o, and the budget is absolute
        // (hh_size/recent_size, not keep_ratio) — surface the effective values honestly.
        "faithful_h2o": args.eviction_policy() == "h2o",
        "h2o_hh_size": args.h2o_hh_size(),
        "h2o_recent_size": args.h2o_recent_size(),
        "time_normalized": args.eviction_policy() != "h2o" && !args.h2o_raw_scores(),
        "skip_layers": args.skip_layers,
        "skip_ratio": args.skip_ratio,
    });
    println!("{}", serde_json::to_string_pretty(&json_val)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::routes_to_prefill_keepset;

    /// The footgun fix: an explicit `eviction plugin --name pyramidkv` (policy == the registered PFA
    /// stage) MUST route to the faithful per-head prefill-keepset path, same as the "none" happy path
    /// — otherwise it silently ran the degraded layer-wide `importance()` fallback. A non-PFA policy
    /// (h2o/streaming) keeps its own score-fed path.
    #[test]
    fn routes_pfa_stage_and_happy_path_but_not_other_policies() {
        assert!(
            routes_to_prefill_keepset("none", "pyramidkv"),
            "happy path arms the keep-set"
        );
        assert!(
            routes_to_prefill_keepset("pyramidkv", "pyramidkv"),
            "explicit `eviction plugin --name pyramidkv` must route to faithful, not degrade"
        );
        assert!(
            !routes_to_prefill_keepset("h2o", "pyramidkv"),
            "h2o keeps its own score-fed path"
        );
        assert!(
            !routes_to_prefill_keepset("streaming", "pyramidkv"),
            "streaming keeps its own path"
        );
    }
}
