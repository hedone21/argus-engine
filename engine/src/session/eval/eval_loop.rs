//! Generic eval-LL loop: `run_eval_ll_generic<C: EvalCacheKind>`.
//!
//! Replaces `run_eval_ll` and `run_quant_window_eval_ll` in `generate.rs`.
//! The two modes differ only in their `StepHook` implementations and
//! `KVCacheOps` type parameters — all other logic is shared here.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::inference::sampling;
use crate::inference::skip_config::SkipConfig;
use crate::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::qcf::ImportanceCollector;
use crate::qcf_types::SubLayer;
use crate::shape::Shape;
use crate::tensor::Tensor;

use super::fmt_bridge::EvalCacheKind;
use super::hook::StepHook;
use super::output::{EvalConfig, EvalOutput, EvalQuestion};

/// Run log-likelihood evaluation for a list of multiple-choice questions.
///
/// This is the unified entry point for both eviction (`EvictionHook`) and
/// quant-window (`QuantWindowFlushHook`) eval modes. The caller provides:
///
/// - `kv_caches`: per-layer KV cache (either `KVCache` or `QuantizedRecentWindowCache`)
/// - `hook`: cache-management and QCF-collection logic
/// - `questions`: pre-parsed questions in grouped format
/// - `eval_config`: loop configuration (budgets, sizes, flags)
/// - `skip_config`: optional layer-skip configuration (triggers importance 2-pass)
///
/// # JSON output
///
/// Returns an `EvalOutput` whose `to_json()` matches the format previously
/// produced by `run_eval_ll` and `run_quant_window_eval_ll` exactly.
#[allow(clippy::too_many_arguments)]
pub fn run_eval_ll_generic<C: EvalCacheKind>(
    model: &TransformerModel,
    tokenizer: &tokenizers::Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut Vec<C>,
    hook: &mut dyn StepHook<C>,
    questions: &[EvalQuestion],
    eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
    mut evict_dump: Option<&mut super::dump::JsonlDumpWriter>,
) -> Result<EvalOutput> {
    let vocab_size = eval_config.vocab_size;
    let hidden_size = eval_config.hidden_size;
    let max_seq_len = eval_config.max_seq_len;

    // ── Pre-allocate decode buffers (reused across all questions / choices) ──
    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut decode_logits =
        Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());

    let q_dim = model.config.num_attention_heads * model.config.head_dim;
    let k_dim = model.config.num_key_value_heads * model.config.head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;
    let mut gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len,
        },
        memory,
        backend.clone(),
    )?;

    // Pre-allocate prefill workspace for GPU buffer reuse across questions.
    // Allocated once at max_seq_len; prevents NVIDIA OpenCL driver crash from
    // accumulated alloc/free cycles during multi-question eval-ll.
    let mut prefill_ws = if backend.is_gpu() {
        use crate::layers::workspace::PrefillWorkspace;
        PrefillWorkspace::new(
            &WorkspaceConfig {
                batch_size: 1,
                dim: hidden_size,
                q_dim,
                k_dim,
                v_dim,
                ffn_hidden,
                n_heads: model.config.num_attention_heads,
                max_seq_len,
            },
            max_seq_len,
            memory,
            backend.clone(),
        )
        .ok()
    } else {
        None
    };

    // Single-token CPU input tensor (reused for decode steps)
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );

    // Pre-allocate GPU input tensor for decode loops (avoids per-token GPU alloc)
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let mut gpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        gpu_gen_buf,
        backend.clone() as Arc<dyn Backend>,
    );

    // ── Importance 2-pass (only when skip_config is active) ──
    let (importance_table, layer_skip_qcf, layer_skip_opr, layer_skip_set_len) =
        run_importance_pass(
            model,
            tokenizer,
            backend,
            memory,
            kv_caches,
            hook,
            questions,
            vocab_size,
            eval_config,
            skip_config,
        )?;

    // ── Per-question evaluation loop ──
    let mut results: Vec<serde_json::Value> = Vec::new();
    let overall_start = std::time::Instant::now();

    let qcf_layer_skip = layer_skip_opr.map(|v| v as f64);
    let qcf_layer_skip_layers = layer_skip_opr.map(|_| layer_skip_set_len);

    let trace_alloc = std::env::var("LLM_OCL_ALLOC_TRACE").is_ok();
    if trace_alloc {
        #[cfg(feature = "opencl")]
        {
            let (live, bytes, allocs, releases) =
                crate::memory::opencl::device::snapshot_alloc_counters();
            eprintln!(
                "[OCL-Trace] pre-loop live={} bytes={:.1}MB total_allocs={} total_releases={}",
                live,
                bytes as f64 / (1024.0 * 1024.0),
                allocs,
                releases,
            );
        }
    }

    for (q_idx, question) in questions.iter().enumerate() {
        let q_start = std::time::Instant::now();

        // Reset caches and hook state for this question
        hook.reset_caches(kv_caches);
        // Flush GPU queue to release deferred OpenCL buffers from previous question.
        // Without this, NVIDIA's runtime accumulates pending buffer releases → OOM.
        backend.synchronize()?;

        if trace_alloc {
            #[cfg(feature = "opencl")]
            {
                let (live, bytes, allocs, releases) =
                    crate::memory::opencl::device::snapshot_alloc_counters();
                eprintln!(
                    "[OCL-Trace] Q{} begin id={} live={} bytes={:.1}MB total_allocs={} total_releases={}",
                    q_idx + 1,
                    question.id,
                    live,
                    bytes as f64 / (1024.0 * 1024.0),
                    allocs,
                    releases,
                );
            }
        }

        // Tokenize prompt
        let prompt_enc = tokenizer
            .encode(question.prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let prompt_ids: &[u32] = prompt_enc.get_ids();
        let prompt_len = prompt_ids.len();

        if prompt_len > max_seq_len {
            eprintln!(
                "[Eval-LL] {}: prompt too long ({} > {}), skipping",
                question.id, prompt_len, max_seq_len
            );
            continue;
        }

        // ── Per-question budget (ratio mode) ──
        let q_eval_config;
        let effective_eval_config = if eval_config.kv_budget_ratio > 0.0 {
            let budget = ((prompt_len as f32) * eval_config.kv_budget_ratio) as usize;
            let budget = budget.max(1);
            hook.set_effective_budget(budget);
            q_eval_config = EvalConfig {
                effective_budget: budget,
                ..eval_config.clone()
            };
            &q_eval_config
        } else {
            eval_config
        };

        // ── Prefill ──
        let (prompt_logits_cpu, start_pos_after_prompt) = run_prefill(
            model,
            backend,
            memory,
            kv_caches,
            hook,
            prompt_ids,
            &mut decode_logits,
            &mut x_gen,
            &mut gen_ws,
            &cpu_gen_input,
            vocab_size,
            effective_eval_config,
            skip_config,
            prefill_ws.as_mut(),
        )?;

        // ── Score collection probe: capture attention weights for QCF-ATTN v2 ──
        // Batch prefill doesn't populate score_accumulator (workspace: None).
        // Re-feed the last token as a decode step to capture per-head attention.
        //
        // IMPORTANT: kv_cache.update() always writes at current_pos and increments it,
        // ignoring start_pos (which only affects RoPE). The probe adds an extra entry
        // at position prompt_len with RoPE for position prompt_len-1. We must restore
        // current_pos afterwards so eviction sees the correct token count.
        //
        // The probe re-feeds the *post-question* last token, so the importance it
        // produces is query-informed. `--evict-timing prefill_end` suppresses it
        // (`runs_query_probe()` = false) and instead ranks on the context importance
        // accumulated by the token-by-token prefill above — a query-agnostic decision.
        if effective_eval_config.evict_timing.runs_query_probe()
            && hook.needs_score_probe(kv_caches)
        {
            // Save current_pos before probe (will be prompt_len)
            let saved_positions: Vec<usize> = kv_caches.iter().map(|c| c.cur_pos()).collect();

            let last_token_id = prompt_ids[prompt_len - 1];
            // SAFETY: cpu_gen_input was allocated with 4 bytes (one u32).
            unsafe {
                *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token_id;
            }
            // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
            backend.write_buffer(&mut gpu_gen_input, unsafe {
                std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
            })?;

            if let Some(acc) = hook.score_accumulator() {
                acc.begin_step();
            }

            // Phase α-K ①-c: forward_into → fmt round-trip. cache_self_need(quant-window AWQE)는 wrap 전 산출.
            let probe_need = kv_caches.first().is_some_and(|c| c.needs_scores());
            C::forward_fmt_roundtrip(kv_caches, |fmts| {
                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &gpu_gen_input,
                    start_pos: start_pos_after_prompt - 1,
                    fmts,
                    backend,
                    memory,
                    logits_out: &mut decode_logits,
                    x_gen: Some(&mut x_gen),
                    workspace: Some(&mut gen_ws),
                    logits_last_only: false,
                    score_accumulator: hook.score_accumulator(),
                    query_stats_accumulator: None,
                    skip_config,
                    importance_collector: None,
                    cache_self_need_scores: probe_need,
                    layer_boundary_hook: None,
                    read_stage: None,
                    prefill_attn: None,
                })
            })?;

            // Restore current_pos: undo the probe's kv_cache.update() increment.
            // The extra entry at position prompt_len is beyond current_pos and invisible.
            for (cache, &pos) in kv_caches.iter_mut().zip(saved_positions.iter()) {
                cache.set_cur_pos(pos);
            }
        }

        // ── post_prefill hook (eviction if cache exceeds budget) ──
        hook.post_prefill(kv_caches);

        // ── IMP-1: drain + write the evict_importance dump record ──
        // Read-only diagnostic; the hook captured it during post_prefill (above)
        // without perturbing eviction (INV-147). No record when eviction did not fire.
        if let Some(writer) = evict_dump.as_deref_mut()
            && let Some(snap) = hook.take_evict_importance_dump()
        {
            super::dump::write_evict_importance_record(
                writer,
                &snap,
                &question.id,
                question.gold_token_positions.as_deref(),
                question.needle_token_positions.as_deref(),
            )?;
        }

        // IMPORTANT: Do NOT update start_pos_after_prompt to current_pos after eviction.
        // After shift_positions(), cached K vectors retain their original RoPE positions.
        // The next decode token's RoPE must continue from prompt_len (original position)
        // to maintain correct relative distances. Using current_pos (compacted) would
        // create a RoPE discontinuity where cached tokens appear as "future" tokens.

        // ── Snapshot KV cache after prompt ──
        let snap = hook.snapshot(kv_caches);

        // ── Score each choice ──
        let mut choice_nlls: Vec<f64> = Vec::new();
        let mut choice_byte_lens: Vec<usize> = Vec::new();
        let mut choice_token_lens: Vec<usize> = Vec::new();

        for choice_text in &question.choices {
            let full_text = format!("{}{}", question.prompt, choice_text);
            let full_enc = tokenizer
                .encode(full_text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let full_ids: Vec<u32> = full_enc.get_ids().to_vec();

            // Safety: prompt_ids.len() may be larger than full_ids when the
            // tokenizer merges tokens differently. Guard against underflow.
            let cont_start = prompt_ids.len().min(full_ids.len());
            let cont_ids: Vec<u32> = full_ids[cont_start..].to_vec();

            if cont_ids.is_empty() {
                choice_nlls.push(f64::INFINITY);
                choice_byte_lens.push(choice_text.len());
                choice_token_lens.push(0);
                continue;
            }

            // First continuation token is scored from the prompt logits
            let mut total_nll =
                -sampling::compute_log_prob(&prompt_logits_cpu, cont_ids[0], vocab_size);

            // Multi-token: decode remaining tokens one-by-one
            if cont_ids.len() > 1 {
                // Restore KV cache to post-prompt state for each choice
                snap.restore_to(kv_caches);

                for (sp, token_pair) in (start_pos_after_prompt..).zip(cont_ids.windows(2)) {
                    let input_token = token_pair[0];
                    let target_token = token_pair[1];

                    // Write token id into CPU buffer
                    // SAFETY: cpu_gen_input was allocated with size 4 bytes (one u32).
                    unsafe {
                        *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
                    }
                    // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
                    backend.write_buffer(&mut gpu_gen_input, unsafe {
                        std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
                    })?;

                    // Score accumulator begin_step only during choice decode
                    if let Some(acc) = hook.score_accumulator() {
                        acc.begin_step();
                    }

                    let dec_need = kv_caches.first().is_some_and(|c| c.needs_scores());
                    C::forward_fmt_roundtrip(kv_caches, |fmts| {
                        model.forward_into(TransformerModelForwardArgs {
                            input_tokens: &gpu_gen_input,
                            start_pos: sp,
                            fmts,
                            backend,
                            memory,
                            logits_out: &mut decode_logits,
                            x_gen: Some(&mut x_gen),
                            workspace: Some(&mut gen_ws),
                            logits_last_only: false,
                            score_accumulator: hook.score_accumulator(),
                            query_stats_accumulator: None,
                            skip_config,
                            importance_collector: None,
                            cache_self_need_scores: dec_need,
                            layer_boundary_hook: None,
                            read_stage: None,
                            prefill_attn: None,
                        })
                    })?;

                    // No eviction during choice decode. Cache may grow slightly
                    // beyond budget (by continuation length), but snapshot restore
                    // resets it between choices. This gives deterministic, isolated
                    // per-choice evaluation without cross-choice score contamination.

                    // Read logits and accumulate NLL
                    let mut step_logits = vec![0.0f32; vocab_size];
                    // SAFETY: buffer is exactly vocab_size * 4 bytes of f32.
                    unsafe {
                        let ptr = step_logits.as_mut_ptr() as *mut u8;
                        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
                        backend.read_buffer(&decode_logits, slice)?;
                    }
                    total_nll -= sampling::compute_log_prob(&step_logits, target_token, vocab_size);
                }
            }

            choice_nlls.push(total_nll);
            choice_byte_lens.push(choice_text.len());
            choice_token_lens.push(cont_ids.len());
        }

        // Restore for consistency (not strictly required — hook.reset_caches next iter)
        snap.restore_to(kv_caches);

        // ── Determine predicted choice ──
        // Byte-length-normalized NLL (acc_norm, lm-eval style)
        let predicted_norm: usize = choice_nlls
            .iter()
            .zip(choice_byte_lens.iter())
            .enumerate()
            .min_by(|(_, (a, al)), (_, (b, bl))| {
                let a_norm = *a / (**al).max(1) as f64;
                let b_norm = *b / (**bl).max(1) as f64;
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Raw prediction (acc, no normalization)
        let predicted_raw: usize = choice_nlls
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let elapsed_q = q_start.elapsed().as_secs_f64();

        eprintln!(
            "[Eval-LL] {}/{} {} — norm={} raw={} nlls=[{}] {:.1}s",
            q_idx + 1,
            questions.len(),
            question.id,
            predicted_norm,
            predicted_raw,
            choice_nlls
                .iter()
                .map(|v| format!("{:.3}", v))
                .collect::<Vec<_>>()
                .join(","),
            elapsed_q,
        );

        // ── Per-question result JSON ──
        let final_cache_pos = kv_caches.iter().map(|c| c.cur_pos()).max().unwrap_or(0);

        let extra = hook.extra_question_fields(kv_caches);

        let mut result_obj = serde_json::json!({
            "id": question.id,
            "choice_nlls": choice_nlls,
            "choice_byte_lens": choice_byte_lens,
            "choice_token_lens": choice_token_lens,
            "predicted": predicted_norm,
            "predicted_raw": predicted_raw,
            "n_choices": question.choices.len(),
            "n_prompt_tokens": prompt_len,
            "final_cache_pos": final_cache_pos,
            "qcf_layer_skip": qcf_layer_skip,
            "qcf_layer_skip_layers": qcf_layer_skip_layers,
        });

        // Merge hook-specific fields (qcf, qcf_quant_legacy, qcf_value_aware, effective_budget, etc.)
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                result_obj[k] = v.clone();
            }
        }

        results.push(result_obj);
    }

    let wall_time_s = overall_start.elapsed().as_secs_f64();

    // ── Build layer_importance JSON ──
    let layer_importance_json = importance_table.map(|table| {
        serde_json::json!(
            table
                .entries()
                .iter()
                .map(|e| serde_json::json!({
                    "layer": e.layer_id,
                    "sublayer": format!("{:?}", e.sublayer),
                    "importance": e.importance,
                    "opr": e.opr,
                }))
                .collect::<Vec<serde_json::Value>>()
        )
    });

    // Normalize layer_skip_qcf: skipped / remaining = raw / (1 - raw)
    let layer_skip_qcf_normalized = layer_skip_qcf.map(|qcf| {
        const NORMALIZED_CAP: f32 = 100.0;
        if qcf >= 1.0 - 1e-7 {
            NORMALIZED_CAP
        } else {
            qcf / (1.0 - qcf)
        }
    });

    Ok(EvalOutput {
        results,
        config: serde_json::json!(hook.extra_config_fields()),
        wall_time_s,
        layer_importance: layer_importance_json,
        layer_skip_qcf,
        layer_skip_qcf_normalized,
        qcf_layer_skip,
        qcf_layer_skip_layers,
    })
}

/// Importance 2-pass: runs a forward pass on the first question's prompt to
/// measure per-layer importance when a `SkipConfig` is active.
///
/// Returns `(importance_table, layer_skip_qcf, layer_skip_opr, skip_set_len)`.
/// All fields are `None` / `0` when `skip_config` is `None` or questions is empty.
type ImportancePassResult = (
    Option<crate::qcf::ImportanceTable>,
    Option<f32>,
    Option<f32>,
    usize,
);

#[allow(clippy::too_many_arguments)]
fn run_importance_pass<C: EvalCacheKind>(
    model: &TransformerModel,
    tokenizer: &tokenizers::Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut Vec<C>,
    hook: &mut dyn StepHook<C>,
    questions: &[EvalQuestion],
    vocab_size: usize,
    _eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
) -> Result<ImportancePassResult> {
    let sc = match skip_config {
        None => return Ok((None, None, None, 0)),
        Some(sc) => sc,
    };
    if questions.is_empty() {
        return Ok((None, None, None, 0));
    }

    let first_q = &questions[0];
    let prompt_enc = tokenizer
        .encode(first_q.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let prompt_ids_imp: Vec<u32> = prompt_enc.get_ids().to_vec();
    let imp_len = prompt_ids_imp.len();

    // Reset caches before importance measurement
    hook.reset_caches(kv_caches);

    let cpu_buf = Galloc::new().alloc(imp_len * 4, DType::U8)?;
    // SAFETY: allocated exactly imp_len u32 words above.
    unsafe {
        let ptr = cpu_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(prompt_ids_imp.as_ptr(), ptr, imp_len);
    }
    let cpu_input = Tensor::new(
        Shape::new(vec![1, imp_len]),
        cpu_buf,
        Arc::new(CpuBackend::new()),
    );
    let input_tensor = backend.copy_from(&cpu_input)?;

    let imp_logits_buf = memory.alloc(imp_len * vocab_size * 4, DType::F32)?;
    let mut imp_logits = Tensor::new(
        Shape::new(vec![1, imp_len, vocab_size]),
        imp_logits_buf,
        backend.clone(),
    );

    let mut collector = ImportanceCollector::new();
    C::forward_fmt_roundtrip(kv_caches, |fmts| {
        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            fmts,
            backend,
            memory,
            logits_out: &mut imp_logits,
            x_gen: None,
            workspace: None,
            logits_last_only: false,
            score_accumulator: None,
            query_stats_accumulator: None,
            skip_config: None, // intentionally None for importance measurement
            importance_collector: Some(&mut collector),
            cache_self_need_scores: false,
            layer_boundary_hook: None,
            read_stage: None,
            prefill_attn: None,
        })
    })?;

    let table = collector.build();

    let skip_set: Vec<(usize, SubLayer)> = sc
        .attn_skip
        .union(&sc.mlp_skip)
        .map(|&l| (l, SubLayer::Full))
        .collect();
    let qcf = table.compute_qcf_weight(&skip_set);
    let opr_skip = table.compute_opr_skip(&skip_set);
    let skip_set_len = skip_set.len();

    eprintln!(
        "[Skip] Importance measured on {} tokens, layer_skip_qcf={:.4}",
        imp_len, qcf
    );

    // Reset caches again before actual evaluation
    hook.reset_caches(kv_caches);

    Ok((Some(table), Some(qcf), Some(opr_skip), skip_set_len))
}

/// Decide whether prefill runs token-by-token (per-step score accumulation) instead
/// of a single batched pass. Pure so the timing policy — and the `INV-147` guarantee
/// that the default leaves prefill batched — is provable without a model.
///
/// Two independent reasons to go token-by-token:
/// 1. `needs_cache_scores`: the cache itself consumes per-step attention scores
///    (quant-window AWQE/AW-VOPR) — unchanged, timing-independent.
/// 2. `--evict-timing prefill_end`: accumulate query-agnostic context importance, but
///    only when a *score-ranking* eviction will actually fire — `hook_ranks_on_scores`
///    with a positive budget the prompt exceeds. Positional policies and the
///    no-eviction case keep the fast batched path.
fn prefill_runs_token_by_token(
    needs_cache_scores: bool,
    evict_timing: super::evict_timing::EvictTiming,
    hook_ranks_on_scores: bool,
    effective_budget: usize,
    prompt_len: usize,
) -> bool {
    if prompt_len <= 1 {
        return false;
    }
    if needs_cache_scores {
        return true;
    }
    evict_timing.accumulates_context_scores()
        && hook_ranks_on_scores
        && effective_budget > 0
        && prompt_len > effective_budget
}

/// Run prompt processing (prefill or chunked prefill + decode) for one question.
///
/// Returns `(prompt_logits_cpu, start_pos_after_prompt)`.
///
/// - `prompt_logits_cpu`: logits for the last prompt token (for scoring cont_ids[0])
/// - `start_pos_after_prompt`: KV cache position after processing all prompt tokens
#[allow(clippy::too_many_arguments)]
fn run_prefill<C: EvalCacheKind>(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut Vec<C>,
    hook: &mut dyn StepHook<C>,
    prompt_ids: &[u32],
    decode_logits: &mut Tensor,
    x_gen: &mut Tensor,
    gen_ws: &mut LayerWorkspace,
    cpu_gen_input: &Tensor,
    vocab_size: usize,
    eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
    prefill_ws: Option<&mut crate::layers::workspace::PrefillWorkspace>,
) -> Result<(Vec<f32>, usize)> {
    // When KV caches need attention scores (quant-window+AWQE/AW-VOPR), use token-by-token
    // prefill so each quant-window flush has scores from the previous step.
    let needs_scores = !kv_caches.is_empty() && kv_caches[0].needs_scores();

    if prefill_runs_token_by_token(
        needs_scores,
        eval_config.evict_timing,
        hook.ranks_on_scores(),
        eval_config.effective_budget,
        prompt_ids.len(),
    ) {
        return run_token_by_token_prefill(
            model,
            backend,
            memory,
            kv_caches,
            hook,
            prompt_ids,
            decode_logits,
            x_gen,
            gen_ws,
            cpu_gen_input,
            vocab_size,
            eval_config,
            skip_config,
        );
    }

    run_full_prefill(
        model,
        backend,
        memory,
        kv_caches,
        hook,
        prompt_ids,
        vocab_size,
        eval_config,
        skip_config,
        prefill_ws,
    )
}

/// Token-by-token prefill for quant-window AWQE/AW-VOPR: processes each token individually
/// so `set_attn_scores()` is called between tokens, giving quant-window flushes access to
/// attention scores for weighted error computation.
#[allow(clippy::too_many_arguments)]
fn run_token_by_token_prefill<C: EvalCacheKind>(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut Vec<C>,
    hook: &mut dyn StepHook<C>,
    prompt_ids: &[u32],
    decode_logits: &mut Tensor,
    x_gen: &mut Tensor,
    gen_ws: &mut LayerWorkspace,
    cpu_gen_input: &Tensor,
    vocab_size: usize,
    _eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
) -> Result<(Vec<f32>, usize)> {
    let prompt_len = prompt_ids.len();

    // Pre-allocate GPU input tensor (avoids per-token GPU alloc in loop)
    let gpu_buf = memory.alloc(4, DType::U8)?;
    let mut gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        gpu_buf,
        backend.clone() as Arc<dyn Backend>,
    );

    for (i, &token_id) in prompt_ids.iter().enumerate() {
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token_id;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        // Reset the per-step accumulator buffers before each token's forward so
        // importance sums correctly across the prompt (the eviction accumulator under
        // `--evict-timing prefill_end`). No-op for the quant-window hook, whose
        // `score_accumulator()` is None — so this stays byte-identical there.
        if let Some(acc) = hook.score_accumulator() {
            acc.begin_step();
        }

        let tbt_need = kv_caches.first().is_some_and(|c| c.needs_scores());
        C::forward_fmt_roundtrip(kv_caches, |fmts| {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &gen_input,
                start_pos: i,
                fmts,
                backend,
                memory,
                logits_out: decode_logits,
                x_gen: Some(x_gen),
                workspace: Some(gen_ws),
                logits_last_only: false,
                score_accumulator: hook.score_accumulator(),
                query_stats_accumulator: None,
                skip_config,
                importance_collector: None,
                cache_self_need_scores: tbt_need,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
            })
        })?;
    }

    let mut prompt_logits_cpu = vec![0.0f32; vocab_size];
    unsafe {
        let ptr = prompt_logits_cpu.as_mut_ptr() as *mut u8;
        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
        backend.read_buffer(decode_logits, slice)?;
    }

    Ok((prompt_logits_cpu, prompt_len))
}

/// Full prefill: forward all prompt tokens in a single batched pass.
#[allow(clippy::too_many_arguments)]
fn run_full_prefill<C: EvalCacheKind>(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut Vec<C>,
    hook: &mut dyn StepHook<C>,
    prompt_ids: &[u32],
    vocab_size: usize,
    _eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
    _prefill_ws: Option<&mut crate::layers::workspace::PrefillWorkspace>,
) -> Result<(Vec<f32>, usize)> {
    let prompt_len = prompt_ids.len();

    let cpu_indices_buf = Galloc::new().alloc(prompt_len * 4, DType::U8)?;
    // SAFETY: allocated prompt_len u32 words above.
    unsafe {
        let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, prompt_len);
    }
    let cpu_input = Tensor::new(
        Shape::new(vec![1, prompt_len]),
        cpu_indices_buf,
        Arc::new(CpuBackend::new()),
    );
    let input_tensor = backend.copy_from(&cpu_input)?;

    // Only allocate for last position's logits (saves ~3GB vs full prompt × vocab).
    let prefill_logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut prefill_logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        prefill_logits_buf,
        backend.clone(),
    );

    let fp_need = kv_caches.first().is_some_and(|c| c.needs_scores());
    C::forward_fmt_roundtrip(kv_caches, |fmts| {
        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            fmts,
            backend,
            memory,
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            logits_last_only: true,
            score_accumulator: hook.score_accumulator(),
            query_stats_accumulator: None,
            skip_config,
            importance_collector: None,
            cache_self_need_scores: fp_need,
            layer_boundary_hook: None,
            read_stage: None,
            prefill_attn: None,
        })
    })?;

    // Read logits (only last position — much smaller than full prompt × vocab)
    let mut prompt_logits_cpu = vec![0.0f32; vocab_size];
    unsafe {
        let ptr = prompt_logits_cpu.as_mut_ptr() as *mut u8;
        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
        backend.read_buffer(&prefill_logits, slice)?;
    }

    // Explicitly drop GPU buffers and flush queue to free VRAM.
    drop(prefill_logits);
    drop(input_tensor);
    backend.synchronize()?;

    Ok((prompt_logits_cpu, prompt_len))
}

/// Chunked prefill: forward first `effective_budget` tokens as a batch, then
/// decode the remaining prompt tokens one-by-one.
///
/// Used when eviction is active (prompt > budget) to accumulate heavy-hitter attention
/// scores during the decode phase. This gives post_prefill high-quality scores
/// for eviction decisions, matching the quality of decode-step accumulation.
#[allow(clippy::too_many_arguments, dead_code, clippy::ptr_arg)]
fn run_chunked_prefill<C: EvalCacheKind>(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut Vec<C>,
    hook: &mut dyn StepHook<C>,
    prompt_ids: &[u32],
    decode_logits: &mut Tensor,
    x_gen: &mut Tensor,
    gen_ws: &mut LayerWorkspace,
    cpu_gen_input: &Tensor,
    vocab_size: usize,
    eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
    _prefill_ws: Option<&mut crate::layers::workspace::PrefillWorkspace>,
) -> Result<(Vec<f32>, usize)> {
    let effective_budget = eval_config.effective_budget;
    let first_chunk_len = effective_budget;

    // ── First chunk: batched prefill ──
    let cpu_buf = Galloc::new().alloc(first_chunk_len * 4, DType::U8)?;
    // SAFETY: allocated first_chunk_len u32 words above.
    unsafe {
        let ptr = cpu_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, first_chunk_len);
    }
    let cpu_input = Tensor::new(
        Shape::new(vec![1, first_chunk_len]),
        cpu_buf,
        Arc::new(CpuBackend::new()),
    );
    let input_tensor = backend.copy_from(&cpu_input)?;

    // Only allocate for last position's logits — the first chunk logits are
    // never read (only the final decode-step logits are returned). This saves
    // ~116 MB of GPU memory for vocab_size=151936 × budget=200.
    let prefill_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut prefill_logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        prefill_buf,
        backend.clone(),
    );

    if let Some(acc) = hook.score_accumulator() {
        acc.begin_step();
    }

    let fp_need = kv_caches.first().is_some_and(|c| c.needs_scores());
    C::forward_fmt_roundtrip(kv_caches, |fmts| {
        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            fmts,
            backend,
            memory,
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            logits_last_only: true,
            score_accumulator: hook.score_accumulator(),
            query_stats_accumulator: None,
            skip_config,
            importance_collector: None,
            cache_self_need_scores: fp_need,
            layer_boundary_hook: None,
            read_stage: None,
            prefill_attn: None,
        })
    })?;

    // Explicitly drop GPU buffers and flush queue to free VRAM before the
    // decode loop. Mirrors the cleanup in run_full_prefill that prevents
    // NVIDIA OpenCL driver crashes from accumulated deferred buffer releases.
    drop(prefill_logits);
    drop(input_tensor);
    backend.synchronize()?;

    let mut start_pos = first_chunk_len;

    // Pre-allocate GPU input tensor for the decode loop (avoids per-token GPU alloc)
    let gpu_buf = memory.alloc(4, DType::U8)?;
    let mut gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        gpu_buf,
        backend.clone() as Arc<dyn Backend>,
    );

    // ── Decode remaining prompt tokens one-by-one (no eviction during decode) ──
    // Let KV cache grow to full prompt length so score-based importance accumulates
    // across ALL decode steps. Eviction happens once after the loop (in post_prefill),
    // giving the eviction stage the full importance picture before selecting heavy hitters.
    // This matches the OLD eval-ll behavior at 9f033a3.
    for &token_id in &prompt_ids[first_chunk_len..] {
        // SAFETY: cpu_gen_input was allocated with 4 bytes (one u32).
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token_id;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        if let Some(acc) = hook.score_accumulator() {
            acc.begin_step();
        }

        let cp_need = kv_caches.first().is_some_and(|c| c.needs_scores());
        C::forward_fmt_roundtrip(kv_caches, |fmts| {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &gen_input,
                start_pos,
                fmts,
                backend,
                memory,
                logits_out: decode_logits,
                x_gen: Some(x_gen),
                workspace: Some(gen_ws),
                logits_last_only: false,
                score_accumulator: hook.score_accumulator(),
                query_stats_accumulator: None,
                skip_config,
                importance_collector: None,
                cache_self_need_scores: cp_need,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
            })
        })?;
        start_pos += 1;
    }

    // Read logits from last decode step
    let mut logits_cpu = vec![0.0f32; vocab_size];
    // SAFETY: decode_logits is vocab_size f32 values = vocab_size * 4 bytes.
    unsafe {
        let ptr = logits_cpu.as_mut_ptr() as *mut u8;
        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
        backend.read_buffer(decode_logits, slice)?;
    }

    Ok((logits_cpu, start_pos))
}

#[cfg(test)]
mod tests {
    use super::prefill_runs_token_by_token;
    use crate::session::eval::evict_timing::EvictTiming;
    use crate::session::eval::output::{EvalConfig, EvalQuestion};

    // ── prefill-driver selection (mutation-provable INV-147 + prefill_end gating) ──

    /// `INV-147`: with the default timing, the eviction path NEVER switches to
    /// token-by-token — it stays on today's batched full prefill (+ probe), no matter
    /// the budget or prompt length. A regression that made the default accumulate
    /// would flip these and fail the test.
    #[test]
    fn default_timing_keeps_batched_prefill() {
        // Over budget, score-based, long prompt — the prefill_end trigger conditions —
        // yet the default must stay batched.
        assert!(!prefill_runs_token_by_token(
            false,
            EvictTiming::PostPrefillProbe,
            true,
            8,
            100
        ));
        // And without a budget.
        assert!(!prefill_runs_token_by_token(
            false,
            EvictTiming::PostPrefillProbe,
            true,
            0,
            100
        ));
    }

    /// `prefill_end` goes token-by-token exactly when a score-ranking eviction will
    /// fire: a positive budget the prompt exceeds, on a score-based policy.
    #[test]
    fn prefill_end_token_by_token_only_when_eviction_fires() {
        let pe = EvictTiming::PrefillEnd;
        // score-based, over budget → token-by-token (accumulate context importance).
        assert!(prefill_runs_token_by_token(false, pe, true, 8, 100));
        // positional policy (ranks_on_scores=false) → batched (scores would be unused).
        assert!(!prefill_runs_token_by_token(false, pe, false, 8, 100));
        // no budget → no eviction → batched.
        assert!(!prefill_runs_token_by_token(false, pe, true, 0, 100));
        // prompt within budget → no eviction → batched.
        assert!(!prefill_runs_token_by_token(false, pe, true, 100, 100));
        // single-token prompt is never chunked.
        assert!(!prefill_runs_token_by_token(false, pe, true, 1, 1));
    }

    /// Cache-driven score need (quant-window) forces token-by-token regardless of the
    /// eviction timing — that path is unchanged by this feature.
    #[test]
    fn cache_score_need_forces_token_by_token_for_any_timing() {
        for timing in [EvictTiming::PostPrefillProbe, EvictTiming::PrefillEnd] {
            assert!(prefill_runs_token_by_token(true, timing, false, 0, 10));
        }
        // ...but still not for a trivial 1-token prompt.
        assert!(!prefill_runs_token_by_token(
            true,
            EvictTiming::PostPrefillProbe,
            false,
            0,
            1
        ));
    }

    fn make_config() -> EvalConfig {
        EvalConfig {
            max_seq_len: 2048,
            effective_budget: 0,
            kv_budget_ratio: 0.0,
            greedy: true,
            kv_type: "f32".to_string(),
            qcf_mode: "attn".to_string(),
            vocab_size: 32000,
            hidden_size: 2048,
            evict_timing: crate::session::eval::EvictTiming::default(),
        }
    }

    #[test]
    fn test_eval_config_fields() {
        let cfg = make_config();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.max_seq_len, 2048);
    }

    #[test]
    fn test_eval_question_structure() {
        let q = EvalQuestion {
            id: "q1".to_string(),
            prompt: "The capital of France is".to_string(),
            choices: vec![" Paris".to_string(), " London".to_string()],
            ..Default::default()
        };
        assert_eq!(q.choices.len(), 2);
        assert_eq!(q.id, "q1");
    }

    /// Test that EvalConfig with budget_mode=false (effective_budget=0) selects full prefill.
    #[test]
    fn test_budget_mode_false_when_zero() {
        let cfg = make_config();
        let prompt_len = 100;
        let budget_mode = cfg.effective_budget > 0;
        assert!(!budget_mode);
        // full prefill: no chunking regardless of prompt length
        assert!(!(budget_mode && prompt_len > cfg.effective_budget));
    }

    /// Test that EvalConfig with effective_budget > 0 and prompt_len > budget triggers chunked.
    #[test]
    fn test_budget_mode_triggers_chunked() {
        let mut cfg = make_config();
        cfg.effective_budget = 50;
        let prompt_len = 100;
        let budget_mode = cfg.effective_budget > 0;
        assert!(budget_mode && prompt_len > cfg.effective_budget);
    }

    #[test]
    fn test_predicted_norm_tie_returns_first() {
        // When all NLLs are equal, predicted_norm should return 0.
        let choice_nlls: Vec<f64> = vec![1.0, 1.0, 1.0];
        let choice_byte_lens: Vec<usize> = vec![5, 5, 5];
        let predicted_norm: usize = choice_nlls
            .iter()
            .zip(choice_byte_lens.iter())
            .enumerate()
            .min_by(|(_, (a, al)), (_, (b, bl))| {
                let a_norm = *a / (**al).max(1) as f64;
                let b_norm = *b / (**bl).max(1) as f64;
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(predicted_norm, 0);
    }

    #[test]
    fn test_predicted_norm_selects_minimum() {
        let choice_nlls: Vec<f64> = vec![2.0, 1.0, 3.0];
        let choice_byte_lens: Vec<usize> = vec![4, 4, 4];
        let predicted_norm: usize = choice_nlls
            .iter()
            .zip(choice_byte_lens.iter())
            .enumerate()
            .min_by(|(_, (a, al)), (_, (b, bl))| {
                let a_norm = *a / (**al).max(1) as f64;
                let b_norm = *b / (**bl).max(1) as f64;
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(predicted_norm, 1);
    }

    #[test]
    fn test_predicted_raw_selects_minimum() {
        let choice_nlls: Vec<f64> = vec![3.0, 0.5, 2.0];
        let predicted_raw: usize = choice_nlls
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(predicted_raw, 1);
    }

    #[test]
    fn test_layer_skip_qcf_normalized_cap() {
        // qcf = 1.0 should cap at 100.0
        let qcf: f32 = 1.0;
        const NORMALIZED_CAP: f32 = 100.0;
        let normalized = if qcf >= 1.0 - 1e-7 {
            NORMALIZED_CAP
        } else {
            qcf / (1.0 - qcf)
        };
        assert!((normalized - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_layer_skip_qcf_normalized_midrange() {
        // qcf = 0.5 → normalized = 0.5 / 0.5 = 1.0
        let qcf: f32 = 0.5;
        const NORMALIZED_CAP: f32 = 100.0;
        let normalized = if qcf >= 1.0 - 1e-7 {
            NORMALIZED_CAP
        } else {
            qcf / (1.0 - qcf)
        };
        assert!((normalized - 1.0).abs() < 1e-5);
    }
}
