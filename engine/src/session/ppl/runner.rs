//! Phase 4-C-2: `run_ppl_dispatch` + `run_ppl` + `run_quant_window_ppl` —
//! `bin/generate.rs::main()`의 PPL 분기 + 두 free fn 본문 외과적 이동.

use std::sync::Arc;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::capability::quant_attn::QuantAttnBackend;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::inference::sampling::{self};
use crate::kv::cache_manager::CacheManager;
use crate::kv::kv_cache::KVCache;
use crate::kv::quant_window_cache::QuantizedRecentWindowCache;
use crate::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::TransformerModel;
use crate::models::transformer::TransformerModelForwardArgs;
use crate::session::cli::Args;
use crate::session::eval::EvalCacheKind;
use crate::session::ppl::args::PplResult;
use crate::session::ppl::args::PplRunCtx;
use crate::session::qcf_runtime::{
    dispatch_swap_weights, dump_layer_weights_to_dir, run_layer_swap,
};
use crate::shape::Shape;
use crate::tensor::Tensor;

/// PPL 모드 dispatch entry point. main()에서 호출.
/// 본문은 원본 ppl_main 분기를 그대로 이동한다.
pub fn run_ppl_dispatch(ctx: PplRunCtx) -> Result<()> {
    let PplRunCtx {
        args,
        backend,
        memory,
        model,
        tokenizer,
        mut kv_caches,
        mut cache_manager,
        mut score_accumulator,
        skip_config,
        hidden_size,
        vocab_size,
        max_seq_len,
        num_layers: _num_layers,
        kv_heads: _kv_heads,
        head_dim: _head_dim,
        actual_protected_prefix,
        score_based_eviction,
        qcf_warmup_importance,
        qcf_swap_decision,
        qcf_workflow_start,
        auto_eviction,
        swap_algorithm,
    } = ctx;

    let ppl_path = args
        .ppl
        .as_deref()
        .expect("run_ppl_dispatch only called when args.ppl is Some");

    // LISWAP-PPL diagnostic: dump weights immediately after model load
    // (before any swap), useful for Q4-native baseline comparison.
    if let Some(ref dump_dir) = args.dump_q4_after_load {
        dump_layer_weights_to_dir(&model, &backend, dump_dir)?;
    }

    // LISWAP-PPL Scenario E (warmup-then-measure):
    //   Pass 1: drive the weight swap to completion with no NLL logging.
    //   Reset KV caches + score_accumulator so the measurement pass sees a
    //   fresh cache, then run the measurement pass with the swap trigger
    //   disabled. The cache reset isolates the "cache mismatch" hypothesis
    //   from the "weight quantization-path mismatch" hypothesis.
    if args.ppl_warmup_swap {
        if args.ppl_swap_at_token.is_none() {
            anyhow::bail!(
                "--ppl-warmup-swap requires --ppl-swap-at-token (the warmup pass needs a swap trigger)"
            );
        }
        if model.secondary_mmap.is_none() {
            anyhow::bail!(
                "--ppl-warmup-swap requires --secondary-gguf (weights must be available for swap)"
            );
        }

        let mut warmup_args = args.clone();
        // Suppress CSV/JSON outputs on the warmup pass.
        warmup_args.ppl_nll_csv = None;
        warmup_args.qcf_dump = None;
        eprintln!("[PPL-Swap] === Pass 1: warmup (driving swap to completion) ===");
        let _warmup_dummy = run_ppl(
            &warmup_args,
            &model,
            &tokenizer,
            &backend,
            &*memory,
            &mut kv_caches,
            &mut cache_manager,
            &mut score_accumulator,
            vocab_size,
            hidden_size,
            max_seq_len,
            ppl_path,
            auto_eviction,
            score_based_eviction,
            actual_protected_prefix,
            skip_config.as_ref(),
            /* warmup_only */ true,
        )?;

        // LISWAP-PPL diagnostic: dump weights right after swap completion
        // (before cache reset), so each layer's GPU buffer can be compared
        // byte-for-byte against the Q4-native baseline dump.
        if let Some(ref dump_dir) = args.dump_q4_after_swap {
            dump_layer_weights_to_dir(&model, &backend, dump_dir)?;
        }

        // Reset KV cache positions. The underlying tensor buffers stay
        // allocated; we only rewind the write head + high-water mark so
        // the next prefill starts from pos 0.
        for cache in kv_caches.iter_mut() {
            cache.current_pos = 0;
            cache.high_water_pos = 0;
        }
        if let Some(acc) = score_accumulator.as_mut() {
            acc.reset();
        }
        eprintln!("[PPL-Swap] === Pass 2: measurement (swap disabled, fresh KV cache) ===");
    }

    // Measurement pass. When warmup_swap was active, disable further swap
    // triggers and clear the warmup flag locally so this pass is a pure
    // teacher-forcing PPL run on the already-swapped weights.
    // When `--ppl-measure-prefill-tokens` is set, pass 2 uses that prefill
    // length instead of `--ppl-prefill-tokens` (Scenario F: large prefill
    // shrinks the decode loop, isolating batch vs single-step path).
    let mut measure_args_owned;
    let measure_args: &Args = if args.ppl_warmup_swap {
        measure_args_owned = args.clone();
        measure_args_owned.ppl_swap_at_token = None;
        measure_args_owned.ppl_warmup_swap = false;
        if let Some(measure_prefill) = args.ppl_measure_prefill_tokens {
            measure_args_owned.ppl_prefill_tokens = Some(measure_prefill);
        }
        &measure_args_owned
    } else {
        &args
    };
    let ppl_result = run_ppl(
        measure_args,
        &model,
        &tokenizer,
        &backend,
        &*memory,
        &mut kv_caches,
        &mut cache_manager,
        &mut score_accumulator,
        vocab_size,
        hidden_size,
        max_seq_len,
        ppl_path,
        auto_eviction,
        score_based_eviction,
        actual_protected_prefix,
        skip_config.as_ref(),
        /* warmup_only */ false,
    )?;

    // --qcf-dump: write JSON after PPL measurement completes.
    if let Some(ref dump_path) = args.qcf_dump {
        use crate::session::eval::qcf_helpers::{QcfSwapDumpContext, dump_qcf_swap_json};

        let empty_swap: Vec<usize> = Vec::new();
        let (swap_set, qcf_predicted, fallback_used) = if let Some(ref dec) = qcf_swap_decision {
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
        let total_wall = qcf_workflow_start.elapsed().as_secs_f64() + ppl_result.wall_time_s;

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
            importance_table: qcf_warmup_importance.as_ref(),
            noise_table: Some(model.quant_noise.as_ref()),
            ppl: Some(ppl_result.ppl),
            avg_nll: Some(ppl_result.avg_nll),
            n_eval_tokens: ppl_result.n_eval_tokens,
            wall_time_s: total_wall,
            warmup_tokens: args.qcf_warmup_tokens,
            backend: &args.backend,
            kv_type: &args.kv_type,
            ppl_corpus: Some(ppl_path),
            eval_ll_output: None,
            trajectory: None,
            dpllm_epsilon: None,
            dpllm_epsilon_multi: None,
            dpllm_epsilon_abs: None,
            dpllm_epsilon_qcf: None,
            direct_attn_f4: None,
            direct_attn_f5: None,
            direct_attn_f5_decode_only: None,
            direct_attn_f5_prefill_decode: None,
        };

        dump_qcf_swap_json(dump_path, &ctx)?;
        eprintln!("[QCF-dump] JSON written to {}", dump_path.display());
    }

    Ok(())
}

// ─── Phase 4-C-2: PPL evaluation free fns (lift from bin/generate.rs) ───

#[allow(clippy::too_many_arguments)]
pub fn run_quant_window_ppl(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    // Phase α-W-4 §3.3: quant-window native attention handle (caller 가 caps 에서 pull).
    // OpenCL backend 면 Some, 그 외 None.
    quant_attn: Option<Arc<dyn QuantAttnBackend>>,
    memory: &Arc<dyn Memory>,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
    residual_size: usize,
    text_file: &str,
) -> anyhow::Result<()> {
    let hidden_size = model.config.hidden_size;
    let vocab_size = model.config.vocab_size;

    // ── 1. Read and tokenize reference text ──
    let text = std::fs::read_to_string(text_file)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", text_file, e))?;
    let encoding = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
    let all_ids: Vec<u32> = encoding.get_ids().to_vec();
    let total_tokens = all_ids.len();

    if total_tokens < 2 {
        anyhow::bail!("PPL requires at least 2 tokens, got {}", total_tokens);
    }

    let eval_tokens = total_tokens.min(max_seq_len);
    if total_tokens > max_seq_len {
        eprintln!(
            "[quant-window-PPL] Warning: text has {} tokens, truncating to max_seq_len={}",
            total_tokens, max_seq_len
        );
    }
    let token_ids = &all_ids[..eval_tokens];

    eprintln!(
        "[quant-window-PPL] {} tokens, quant_residual_size={}, max_seq_len={}",
        eval_tokens, residual_size, max_seq_len
    );

    // ── 2. Create QuantizedRecentWindowCache per layer ──
    let mut kv_caches: Vec<QuantizedRecentWindowCache> = (0..num_layers)
        .map(|_| {
            QuantizedRecentWindowCache::new_gpu(
                kv_heads,
                head_dim,
                max_seq_len,
                residual_size,
                2,
                backend.clone(),
                quant_attn.clone(),
                // ppl runner does not resolve a CUDA quant-attn cap → CPU mode on CUDA.
                None,
                memory.clone(),
            )
        })
        .collect();

    // ── 3. Pre-allocate decode buffers ──
    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut decode_logits =
        Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
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
            max_seq_len: args.max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf_kp = memory.alloc(4, DType::U8)?;
    let mut gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf_kp, backend.clone());
    let mut logits_cpu = vec![0.0f32; vocab_size];

    let mut total_nll: f64 = 0.0;
    let mut nll_count: usize = 0;
    let mut qcf_metrics: Vec<serde_json::Value> = Vec::new();
    let mut flush_count: usize = 0;
    let overall_start = std::time::Instant::now();

    // ── 4. Prefill phase ──
    let prefill_len = eval_tokens.min(max_seq_len);
    eprintln!("[quant-window-PPL] Prefill: {} tokens", prefill_len);

    {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(prefill_len * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in token_ids[..prefill_len].iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, prefill_len]), input_buf, cpu_backend);
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(prefill_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, prefill_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        // Phase α-K ①-e: run_quant_window_ppl prefill flip — `QuantizedRecentWindowCache::forward_fmt_roundtrip` 로 forward 1회
        // 동안만 `Vec<QuantizedRecentWindowCache>` → `Arc<QuantWindowFormat>` wrap → `forward_into` → concrete 복귀
        // (①-c eval 미러). multi-token prefill 은 QuantWindowFormat::attention_into 의 신규 prefill arm
        // (seq_len>1 → `prefill_attention` 재사용, quant_window_format.rs:106)을 경유 — OLD forward_prefill<C>
        // 의 quant-window 경로(get_view → flash)와 bit-identical. AWQE 는 run_quant_window_ppl 미활성
        // (set_awqe_enabled 미호출)이라 `cache_self_need_scores`=false(forward_gen.rs:409 OR 항 = false);
        // prefill 은 score 누적 안 하므로 어차피 무관. roundtrip 종료 후 take_flush_proxies/q2_tokens/
        // res_pos 접근은 concrete Vec 복귀 후라 borrow 충돌 없음.
        let cache_self_need_scores = kv_caches.first().is_some_and(|c| c.needs_scores());
        QuantizedRecentWindowCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: 0,
                fmts,
                backend,
                memory: memory.as_ref(),
                logits_out: &mut prefill_logits,
                x_gen: None,
                workspace: None,
                logits_last_only: false,
                score_accumulator: None,
                query_stats_accumulator: None,
                skip_config: None,
                importance_collector: None,
                cache_self_need_scores,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: None,
            })
        })?;

        // Collect flush QCF metrics from prefill
        for metric in kv_caches[0].take_flush_proxies() {
            qcf_metrics.push(serde_json::json!({
                "flush": flush_count,
                "action": metric.action,
                "raw_value": metric.raw_value,
                "normalized_value": metric.normalized_value,
                "tokens_quantized": metric.tokens_affected,
            }));
            flush_count += 1;
        }
        for cache in kv_caches[1..].iter_mut() {
            cache.take_flush_proxies();
        }

        // Read all prefill logits to CPU
        let mut all_logits = vec![0.0f32; prefill_len * vocab_size];
        unsafe {
            let ptr = all_logits.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, all_logits.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }

        // Score tokens 1..prefill_len: logits[i] predicts token[i+1]
        for i in 0..prefill_len - 1 {
            let offset = i * vocab_size;
            let lp = sampling::compute_log_prob(
                &all_logits[offset..offset + vocab_size],
                token_ids[i + 1],
                vocab_size,
            );
            total_nll -= lp;
            nll_count += 1;
        }

        eprintln!(
            "[quant-window-PPL] Prefill NLL: {:.4}, count={}, running PPL={:.4}, Q2_tokens={}, res_pos={}",
            total_nll,
            nll_count,
            (total_nll / nll_count as f64).exp(),
            kv_caches[0].q2_tokens,
            kv_caches[0].res_pos,
        );
    }

    // ── 5. Decode phase (teacher-forcing) ──
    for i in prefill_len..eval_tokens - 1 {
        let input_token = token_ids[i];
        let target_token = token_ids[i + 1];

        // Feed true token
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input_gpu, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        // Phase α-K ①-e: run_quant_window_ppl decode flip — forward_fmt_roundtrip + forward_into.
        // decode(seq_len=1)는 QuantWindowFormat::attention_into 의 decode arm(attention_native / F32-view
        // fallback)을 경유 — ①-c eval quant-window 와 동일 경로(host nll Δ~1e-6=★2 carve-out bit-identical
        // 검증됨). `cache_self_need_scores` 는 AWQE 미활성으로 false(decode 의 need_scores OR 항).
        let cache_self_need_scores = kv_caches.first().is_some_and(|c| c.needs_scores());
        QuantizedRecentWindowCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &gen_input_gpu,
                start_pos: i,
                fmts,
                backend,
                memory: memory.as_ref(),
                logits_out: &mut decode_logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut gen_ws),
                logits_last_only: false,
                score_accumulator: None,
                query_stats_accumulator: None,
                skip_config: None,
                importance_collector: None,
                cache_self_need_scores,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: None,
            })
        })?;

        // Collect flush QCF from decode step
        for metric in kv_caches[0].take_flush_proxies() {
            qcf_metrics.push(serde_json::json!({
                "flush": flush_count,
                "action": metric.action,
                "raw_value": metric.raw_value,
                "normalized_value": metric.normalized_value,
                "tokens_quantized": metric.tokens_affected,
            }));
            flush_count += 1;
        }
        for cache in kv_caches[1..].iter_mut() {
            cache.take_flush_proxies();
        }

        // Read logits and score target
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
            backend.read_buffer(&decode_logits, slice)?;
        }
        let lp = sampling::compute_log_prob(&logits_cpu, target_token, vocab_size);
        total_nll -= lp;
        nll_count += 1;

        // Progress
        if (i + 1) % 200 == 0 {
            let ppl = (total_nll / nll_count as f64).exp();
            eprintln!(
                "[quant-window-PPL] step {}/{}: NLL={:.4}, PPL={:.4}, cache_pos={}, Q2_tokens={}",
                i + 1,
                eval_tokens,
                total_nll,
                ppl,
                kv_caches[0].current_pos(),
                kv_caches[0].q2_tokens,
            );
        }
    }

    // ── 6. Output results ──
    let wall_time = overall_start.elapsed().as_secs_f64();
    let ppl = (total_nll / nll_count as f64).exp();
    let tok_per_sec = nll_count as f64 / wall_time;

    // Separate QCF (NMSE) and OPR metrics from flush proxies
    let qcf_quant_nmse_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kv.quant_flush_opr"))
        .filter_map(|m| m["raw_value"].as_f64())
        .sum();
    let qcf_attn_normalized_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kv.quant_flush_opr"))
        .filter_map(|m| m["normalized_value"].as_f64())
        .sum();

    // quant-window OPR: per-flush events and summary stats
    let qcf_quant_events: Vec<&serde_json::Value> = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() == Some("kv.quant_flush_opr"))
        .collect();
    let n_quant_flushes = qcf_quant_events.len();
    let opr_raw_values: Vec<f64> = qcf_quant_events
        .iter()
        .filter_map(|m| m["raw_value"].as_f64())
        .collect();
    let qcf_quant_opr_sum: f64 = opr_raw_values.iter().sum();
    let qcf_quant_opr_max: f64 = opr_raw_values.iter().cloned().fold(0.0f64, f64::max);
    let qcf_quant_opr_total: Option<f64> = if opr_raw_values.is_empty() {
        None
    } else {
        Some(qcf_quant_opr_sum / opr_raw_values.len() as f64)
    };
    let qcf_quant_opr_events: Option<usize> = if opr_raw_values.is_empty() {
        None
    } else {
        Some(opr_raw_values.len())
    };

    let output = serde_json::json!({
        "ppl": ppl,
        "total_nll": total_nll,
        "token_count": nll_count,
        "tokens_per_second": tok_per_sec,
        "wall_time_s": wall_time,
        "qcf_metrics": qcf_metrics,
        "flush_count": qcf_metrics.len(),
        "n_quant_flushes": n_quant_flushes,
        "qcf_quant_events": qcf_quant_events,
        "qcf_quant_nmse_total": qcf_quant_nmse_total,
        "qcf_attn_total": qcf_quant_nmse_total,
        "qcf_attn_normalized_total": qcf_attn_normalized_total,
        "qcf_quant_opr_sum": qcf_quant_opr_sum,
        "qcf_quant_opr_max": qcf_quant_opr_max,
        "qcf_quant_opr_total": qcf_quant_opr_total,
        "qcf_quant_opr_events": qcf_quant_opr_events,
        "final_cache_pos": kv_caches[0].current_pos(),
        "quant_q2_tokens": kv_caches[0].q2_tokens,
        "quant_res_pos": kv_caches[0].res_pos,
        "config": {
            "model": args.model_path,
            "text_file": text_file,
            "eviction_policy": "quant_window",
            "quant_residual_size": residual_size,
            "max_seq_len": max_seq_len,
            "kv_type": "q2+f32_residual",
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    eprintln!(
        "\n[quant-window-PPL] Final: PPL={:.4}, NLL={:.4}, tokens={}, {:.1} tok/s, {:.1}s, Q2_tokens={}",
        ppl, total_nll, nll_count, tok_per_sec, wall_time, kv_caches[0].q2_tokens
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_ppl(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut Vec<KVCache>,
    cache_manager: &mut CacheManager,
    score_accumulator: &mut Option<AttentionScoreAccumulator>,
    vocab_size: usize,
    hidden_size: usize,
    max_seq_len: usize,
    text_file: &str,
    auto_eviction: bool,
    score_based_eviction: bool,
    protected_prefix: usize,
    skip_config: Option<&crate::inference::skip_config::SkipConfig>,
    // LISWAP-PPL Scenario E: when true, return early as soon as the swap plan
    // completes. NLL/CSV/JSON outputs are suppressed. Used by `--ppl-warmup-swap`
    // to drive the swap to completion before the actual measurement pass.
    warmup_only: bool,
) -> anyhow::Result<PplResult> {
    // ── 1. Read and tokenize reference text ──
    let text = std::fs::read_to_string(text_file)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", text_file, e))?;
    let encoding = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
    let all_ids: Vec<u32> = encoding.get_ids().to_vec();
    let total_tokens = all_ids.len();

    if total_tokens < 2 {
        anyhow::bail!("PPL requires at least 2 tokens, got {}", total_tokens);
    }

    let eval_tokens = total_tokens.min(max_seq_len);
    if total_tokens > max_seq_len {
        eprintln!(
            "[PPL] Warning: text has {} tokens, truncating to max_seq_len={}",
            total_tokens, max_seq_len
        );
    }
    let token_ids = &all_ids[..eval_tokens];

    eprintln!(
        "[PPL] {} tokens, policy={}, kv_budget={}, kv_type={}",
        eval_tokens,
        args.eviction_policy(),
        args.kv_budget(),
        args.kv_type
    );

    // ── 2. Pre-allocate decode buffers ──
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
            max_seq_len: args.max_seq_len,
        },
        memory,
        backend.clone(),
    )?;
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf_ppl = memory.alloc(4, DType::U8)?;
    let mut gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf_ppl, backend.clone());
    let mut logits_cpu = vec![0.0f32; vocab_size];

    // ── 3. Determine prefill chunk size ──
    let has_budget = args.kv_budget() > 0 || args.kv_budget_ratio() > 0.0;
    if auto_eviction && !has_budget {
        eprintln!(
            "[PPL] Warning: eviction enabled without --kv-budget. \
             Results may not be reproducible. Use --kv-budget N for deterministic experiments."
        );
    }
    let prefill_chunk = if let Some(forced) = args.ppl_prefill_tokens {
        // LISWAP-PPL / 측정: 명시적 prefill 길이 강제. budget 로직보다 우선.
        // budget이 있어도 prefill은 절단되지 않음 — eviction은 decode 중 budget 기준 발동.
        let clamped = forced.clamp(2, eval_tokens);
        eprintln!(
            "[PPL] prefill override: --ppl-prefill-tokens={} (budget과 독립, \
             eviction은 budget={} 기준 decode 중 발동)",
            clamped,
            if has_budget { args.kv_budget() } else { 0 }
        );
        clamped
    } else if has_budget {
        let budget = if args.kv_budget_ratio() > 0.0 {
            ((eval_tokens as f32) * args.kv_budget_ratio()) as usize
        } else {
            args.kv_budget()
        };
        let truncated = budget.min(eval_tokens).max(2);
        if truncated < eval_tokens {
            eprintln!(
                "[PPL] prefill truncated by budget: {} → {} tokens \
                 (use --ppl-prefill-tokens={} to fix prefill length independently)",
                eval_tokens, truncated, eval_tokens
            );
        }
        truncated
    } else if auto_eviction && args.eviction_policy() == "sliding" {
        args.eviction_window().min(eval_tokens)
    } else {
        eval_tokens
    };

    let effective_budget = if args.kv_budget_ratio() > 0.0 {
        ((eval_tokens as f32) * args.kv_budget_ratio()) as usize
    } else if args.kv_budget() > 0 {
        args.kv_budget()
    } else {
        max_seq_len // No budget → no eviction trigger
    };

    if has_budget {
        eprintln!(
            "[PPL] Effective budget: {} tokens (deterministic eviction)",
            effective_budget
        );
    }

    // Headroom-based threshold: evict only when cache exceeds budget + headroom.
    // This prevents 1-by-1 evictions every step and ensures batch evictions (~2 total).
    // Example: budget=1500 → headroom=375 → threshold=1875.
    let eviction_headroom = (effective_budget / 4).max(16);
    let eviction_threshold = effective_budget.saturating_add(eviction_headroom);

    let mut total_nll: f64 = 0.0;
    let mut nll_count: usize = 0;
    // PPL v3: collect QCF for every eviction event
    let mut qcf_events: Vec<serde_json::Value> = Vec::new();
    // score-decay 측정(KV roadmap 항목 0 §4.2): --dump-a2sf 지정 시 eviction 직전 + run 종료 시 score
    // accumulator importance 에서 BOS/non-BOS ratio + HH(top-k) 집합 스냅샷을 누적(읽기 전용).
    let mut score_decay_snapshots: Vec<crate::session::score_decay_dump::ScoreDecaySnapshot> =
        Vec::new();
    let overall_start = std::time::Instant::now();

    // LISWAP-PPL: per-token NLL log + token-index-triggered weight swap.
    // (phase, token_idx, token_id, nll, swap_state, layers_swapped)
    let mut per_token_log: Vec<(&'static str, usize, u32, f64, &'static str, usize)> = Vec::new();
    let log_per_token = args.ppl_nll_csv.is_some();
    let mut ppl_swap_plan: Option<crate::weight::IncrementalSwapPlan> = None;
    // dispatch_swap_weights 시그니처 호환용 (PPL 경로에서는 manager 보고 안 함).
    let mut ppl_swap_report_unused: Option<(f32, usize, std::time::Instant, f32)> = None;
    let mut layers_swapped_so_far: usize = 0;
    let ppl_cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let ppl_swap_logged = std::sync::atomic::AtomicBool::new(false);

    if args.ppl_swap_at_token.is_some() && model.secondary_mmap.is_none() {
        anyhow::bail!("--ppl-swap-at-token requires --secondary-gguf to load secondary weights");
    }

    // ── 4. Prefill phase ──
    let prefill_len = prefill_chunk.min(eval_tokens);
    eprintln!("[PPL] Prefill: {} tokens", prefill_len);

    {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(prefill_len * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in token_ids[..prefill_len].iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, prefill_len]), input_buf, cpu_backend);
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(prefill_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, prefill_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        if let Some(acc) = score_accumulator.as_mut() {
            acc.begin_step();
        }

        // Phase α-K ①-d: forward_into → fmt round-trip (run_ppl prefill). begin_step 선행(위) 유지.
        // KVCache → cache_self_need_scores=false. score-feed 는 prefill(workspace=None)이라 자연 skip.
        KVCache::forward_fmt_roundtrip(kv_caches, |fmts| {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: 0,
                fmts,
                backend,
                memory,
                logits_out: &mut prefill_logits,
                x_gen: None,
                workspace: None,
                logits_last_only: false,
                score_accumulator: score_accumulator.as_mut(),
                query_stats_accumulator: None,
                skip_config,
                importance_collector: None,
                cache_self_need_scores: false,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: None,
            })
        })?;

        // Read all prefill logits to CPU
        let mut all_logits = vec![0.0f32; prefill_len * vocab_size];
        unsafe {
            let ptr = all_logits.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, all_logits.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }

        // Score tokens 1..prefill_len: logits[i] predicts token[i+1]
        for i in 0..prefill_len - 1 {
            let offset = i * vocab_size;
            let lp = sampling::compute_log_prob(
                &all_logits[offset..offset + vocab_size],
                token_ids[i + 1],
                vocab_size,
            );
            total_nll -= lp;
            nll_count += 1;
            if log_per_token {
                per_token_log.push(("prefill", i, token_ids[i + 1], -lp, "none", 0));
            }
        }

        eprintln!(
            "[PPL] Prefill NLL: {:.4}, count={}, running PPL={:.4}",
            total_nll,
            nll_count,
            (total_nll / nll_count as f64).exp()
        );
    }

    // ── 5. Decode phase (teacher-forcing) ──
    for (decode_idx, i) in (prefill_len..eval_tokens - 1).enumerate() {
        let input_token = token_ids[i];
        let target_token = token_ids[i + 1];

        // ── LISWAP-PPL: token-index-triggered weight swap ──────────────────
        // dispatch_swap_weights 가 commit 한 IncrementalSwapPlan 을 매 decode
        // step 마다 K=ppl_swap_per_tick 만큼 drain. dynamic-K controller 와
        // async dispatcher 는 측정 결정론을 위해 사용하지 않는다.
        if Some(decode_idx) == args.ppl_swap_at_token && ppl_swap_plan.is_none() {
            dispatch_swap_weights(
                model,
                args.ppl_swap_ratio,
                argus_shared::DtypeTag::Q4_0,
                None, // importance None → fallback uniform
                decode_idx,
                &mut ppl_swap_plan,
                &mut ppl_swap_report_unused,
            );
            if !ppl_swap_logged.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[PPL-Swap] triggered at decode_idx={}, ratio={}, per_tick={}",
                    decode_idx, args.ppl_swap_ratio, args.ppl_swap_per_tick
                );
            }
        }
        let mut swap_state: &'static str = if layers_swapped_so_far > 0 {
            "post_swap"
        } else {
            "none"
        };
        let plan_done = if let Some(plan) = ppl_swap_plan.as_mut() {
            plan.set_per_tick(args.ppl_swap_per_tick);
            let chunk = plan.drain_chunk();
            if !chunk.is_empty() {
                let t_swap = std::time::Instant::now();
                match run_layer_swap(
                    model,
                    &chunk,
                    Some(backend),
                    &ppl_cpu_backend,
                    None,
                    #[cfg(feature = "opencl")]
                    None,
                    #[cfg(feature = "cuda-embedded")]
                    None,
                    #[cfg(feature = "cuda-embedded")]
                    None,
                ) {
                    Ok(report) => {
                        layers_swapped_so_far += report.swapped.len();
                        swap_state = "swapping";
                        eprintln!(
                            "[PPL-Swap] tick decode_idx={} chunk={:?} swapped={} remaining={} latency={:.1}ms",
                            decode_idx,
                            &chunk,
                            report.swapped.len(),
                            plan.remaining_count(),
                            t_swap.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    Err(e) => {
                        eprintln!("[PPL-Swap] run_layer_swap error: {}", e);
                    }
                }
            }
            plan.is_done()
        } else {
            false
        };
        if plan_done {
            eprintln!(
                "[PPL-Swap] plan complete at decode_idx={}, total_swapped={}",
                decode_idx, layers_swapped_so_far
            );
            ppl_swap_plan = None;
            swap_state = "post_swap";

            if warmup_only {
                // LISWAP-PPL Scenario E (warmup pass): swap is complete, return
                // before scoring the current token so the caller can reset KV
                // caches and run the measurement pass from scratch with the
                // already-swapped weights.
                eprintln!(
                    "[PPL-Swap] warmup_only=true → returning at decode_idx={} (no further scoring)",
                    decode_idx
                );
                let wall_time = overall_start.elapsed().as_secs_f64();
                return Ok(PplResult {
                    ppl: 0.0,
                    avg_nll: 0.0,
                    n_eval_tokens: 0,
                    wall_time_s: wall_time,
                });
            }
        }

        // Score accumulator begin step
        if let Some(acc) = score_accumulator.as_mut() {
            acc.begin_step();
        }

        // Feed true token
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input_gpu, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        // Phase α-K ①-d: forward_into → fmt round-trip (run_ppl decode; workspace=Some →
        // forward_gen_fmt, 발산 A 무관). score-feed 활성(heavy-hitter 누적).
        KVCache::forward_fmt_roundtrip(kv_caches, |fmts| {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &gen_input_gpu,
                start_pos: i,
                fmts,
                backend,
                memory,
                logits_out: &mut decode_logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut gen_ws),
                logits_last_only: false,
                score_accumulator: score_accumulator.as_mut(),
                query_stats_accumulator: None,
                skip_config,
                importance_collector: None,
                cache_self_need_scores: false,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: None,
            })
        })?;

        // Read logits and score target
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
            backend.read_buffer(&decode_logits, slice)?;
        }
        let lp = sampling::compute_log_prob(&logits_cpu, target_token, vocab_size);
        total_nll -= lp;
        nll_count += 1;
        if log_per_token {
            per_token_log.push((
                "decode",
                i,
                target_token,
                -lp,
                swap_state,
                layers_swapped_so_far,
            ));
        }

        // ── Budget-based eviction (deterministic, experiment-reproducible) ──
        // Eviction triggers when cache_pos exceeds eviction_threshold (budget + headroom).
        // Using headroom prevents 1-by-1 evictions: evictions occur in ~2 large batches
        // rather than 500+ tiny steps, preserving PPL measurement validity.
        // This is deterministic: same text + same budget = same eviction positions.
        // No dependency on memory pressure or hardware state.
        if auto_eviction && has_budget {
            let before_len = kv_caches[0].current_pos;
            if before_len > eviction_threshold {
                let ratio = effective_budget as f32 / before_len as f32;

                // GPU score sync before the eviction reads importance (mirrors eval-ll's
                // EvictionHook; no-op on CPU / when the GPU accumulator is unarmed). On a GPU
                // backend `init_gpu_score_acc` sets `need_scores=false`, so the CPU accumulator
                // is stale until synced — this fixes BOTH the a2sf snapshot below AND the
                // score-based eviction ranking (`extract_scores` further down), keeping the dump
                // from perturbing the measurement.
                if let Some(acc) = score_accumulator.as_mut() {
                    crate::kv::eviction::score_fed::sync_gpu_scores_to_cpu(acc, backend.as_ref());
                }

                // score-decay 측정: eviction(+ acc.reset) 직전 importance 스냅샷(읽기 전용). budget=top-k.
                if args.dump_a2sf.is_some()
                    && let Some(acc) = score_accumulator.as_ref()
                    && acc.is_active()
                {
                    score_decay_snapshots.push(
                        crate::session::score_decay_dump::compute_score_decay_snapshot(
                            acc.importance_scores(),
                            before_len,
                            effective_budget,
                            i as i64,
                        ),
                    );
                }

                // GPU V buffer readback for QCF-value-aware computation.
                let v_cpu_data: Option<Vec<f32>> = if args.kv_type == "f32"
                    && !kv_caches.is_empty()
                    && kv_caches[0].v_buffer.buffer().as_ptr().is_null()
                {
                    let v_elems = kv_caches[0].v_buffer.buffer().size() / 4;
                    let mut v_buf = vec![0.0f32; v_elems];
                    let byte_slice = unsafe {
                        std::slice::from_raw_parts_mut(v_buf.as_mut_ptr() as *mut u8, v_elems * 4)
                    };
                    match backend.read_buffer(&kv_caches[0].v_buffer, byte_slice) {
                        Ok(()) => Some(v_buf),
                        Err(_) => None,
                    }
                } else {
                    None
                };
                let can_compute_qcf = args.kv_type == "f32"
                    && !kv_caches.is_empty()
                    && (v_cpu_data.is_some() || !kv_caches[0].v_buffer.buffer().as_ptr().is_null());

                // Perform eviction — shared score-fed body (extract → route; force, ratio).
                // GPU scores were synced into the CPU accumulator above, so `extract_scores`
                // reads real importance on both CPU and GPU backends.
                use crate::kv::eviction::score_fed;
                let extracted = if score_based_eviction {
                    score_accumulator
                        .as_ref()
                        .and_then(score_fed::extract_scores)
                } else {
                    None
                };
                let result = {
                    let (scores, last_attn, per_layer) = extracted
                        .as_ref()
                        .map(|e| e.as_args())
                        .unwrap_or((None, None, None));
                    score_fed::route_evict(
                        cache_manager,
                        kv_caches,
                        scores,
                        last_attn,
                        per_layer,
                        true,
                        ratio,
                    )?
                };

                if result.evicted {
                    let eviction_ratio = result.tokens_removed as f32 / before_len as f32;
                    let ppl_at_event = (total_nll / nll_count as f64).exp();

                    let qcf_value_aware_value = if can_compute_qcf
                        && let Some(acc) = score_accumulator.as_ref()
                        && let Some(head_attn) = acc.last_step_head_attn()
                    {
                        use crate::qcf::{QcfKvParams, VDataSource, compute_qcf_kv};
                        use crate::qcf_types::AggregationMode;
                        use argus_extension_api::{StageParams, find_qcf_estimator};
                        let target_len = ((before_len as f32) * ratio) as usize;
                        let cache = &kv_caches[0];
                        let v_cpu_bytes: Option<&[u8]> = v_cpu_data.as_deref().map(|s| {
                            // Reinterpret &[f32] as &[u8] so the unified helper handles it.
                            unsafe {
                                std::slice::from_raw_parts(
                                    s.as_ptr() as *const u8,
                                    std::mem::size_of_val(s),
                                )
                            }
                        });
                        // Resolve the estimator (h2o when score-based, else sliding) by name; the
                        // engine no longer enumerates techniques here. PPL eval only ever triggers
                        // Sliding/heavy-hitter — never weighted-merge — so `k_source` stays None.
                        let (est_name, est_params) = if score_based_eviction {
                            (
                                "h2o",
                                StageParams {
                                    keep_ratio: args.keep_ratio(),
                                    protected_prefix,
                                    ..Default::default()
                                },
                            )
                        } else {
                            ("sliding", StageParams::default())
                        };
                        let estimator = (find_qcf_estimator(est_name)
                            .expect("h2o/sliding QCF estimator registered")
                            .make)(est_params, &[]);
                        match VDataSource::from_buffer(&cache.v_buffer, v_cpu_bytes) {
                            Some(v_source) => {
                                let params = QcfKvParams {
                                    estimator: &*estimator,
                                    target_len,
                                    v_source,
                                    // PPL eval site only triggers Sliding/heavy-hitter,
                                    // never weighted-merge — `k_source` is unused.
                                    k_source: None,
                                    attention_scores: acc.importance_scores(),
                                    head_attn: Some(head_attn),
                                    n_kv_heads: cache.kv_heads(),
                                    head_dim: cache.head_dim(),
                                    current_pos: before_len,
                                    capacity: cache.capacity(),
                                    layout: cache.layout(),
                                    aggregation: AggregationMode::Mean,
                                    beta: 1.0,
                                };
                                let (qcf, _) = compute_qcf_kv(&params);
                                qcf as f64
                            }
                            None => 0.0,
                        }
                    } else {
                        0.0
                    };

                    qcf_events.push(serde_json::json!({
                        "step": i,
                        "tokens_evicted": result.tokens_removed,
                        "eviction_ratio": eviction_ratio,
                        "qcf_value_aware": qcf_value_aware_value,
                        "ppl_at_step": ppl_at_event,
                    }));

                    // IMPORTANT: Do NOT reset start_pos to current_pos after eviction.
                    // After shift_positions(), cached K vectors retain their original RoPE
                    // positions. start_pos must continue incrementing from the original
                    // position to maintain correct RoPE relative distances. Using current_pos
                    // (compacted) creates a RoPE discontinuity where cached tokens appear
                    // as "future" tokens, causing severe NLL degradation.
                    // start_pos continues via `start_pos += 1` in the main loop.
                    if let Some(acc) = score_accumulator.as_mut() {
                        acc.reset();
                        // Reset the GPU accumulator in lockstep (mirrors eval-ll/bench). Without this
                        // the GPU importance keeps accumulating since prefill while the CPU twin above
                        // starts fresh, so a 2nd eviction's synced scores (and the a2sf dump) would
                        // diverge from the CPU oracle. No-op on CPU / when unarmed.
                        score_fed::reset_gpu_scores(acc, backend.as_ref());
                    }
                    eprintln!(
                        "[PPL] Eviction at step {}: {} → {} tokens (removed {})",
                        i, before_len, result.new_pos, result.tokens_removed
                    );
                }
            }
        }

        // Progress
        if (i + 1) % 200 == 0 {
            let ppl = (total_nll / nll_count as f64).exp();
            eprintln!(
                "[PPL] step {}/{}: NLL={:.4}, PPL={:.4}, cache_pos={}",
                i + 1,
                eval_tokens,
                total_nll,
                ppl,
                kv_caches[0].current_pos
            );
        }
    }

    // score-decay 측정: run 종료 시점 스냅샷(step=-1) + 누적 스냅샷을 JSON 파일로 덤프(읽기 전용).
    if let Some(dump_path) = args.dump_a2sf.as_ref() {
        // Sync GPU scores into the CPU accumulator before the final snapshot reads them
        // (no-op on CPU / when the GPU accumulator is unarmed). This is a dump-only,
        // post-run read, so it never touches the measured eviction path.
        if let Some(acc) = score_accumulator.as_mut() {
            crate::kv::eviction::score_fed::sync_gpu_scores_to_cpu(acc, backend.as_ref());
        }
        if let Some(acc) = score_accumulator.as_ref()
            && acc.is_active()
        {
            let pos = kv_caches[0].current_pos;
            let top_k = if effective_budget > 0 {
                effective_budget
            } else {
                pos
            };
            score_decay_snapshots.push(
                crate::session::score_decay_dump::compute_score_decay_snapshot(
                    acc.importance_scores(),
                    pos,
                    top_k,
                    -1,
                ),
            );
        }
        let dump = serde_json::json!({
            "score_decay": args.h2o_decay(),
            "eviction_policy": args.eviction_policy(),
            "snapshots": score_decay_snapshots,
        });
        std::fs::write(dump_path, serde_json::to_string_pretty(&dump)?)?;
        eprintln!(
            "[score-decay] dumped {} snapshot(s) (score_decay={}) → {}",
            score_decay_snapshots.len(),
            args.h2o_decay(),
            dump_path.display()
        );
        // BOS ratio 스모크 가시성: run-end 스냅샷을 stderr 마커로도 노출(파싱 가능).
        if let Some(last) = score_decay_snapshots.last() {
            eprintln!(
                "[score-decay] bos_ratio={:.4} bos_score={:.4} non_bos_mean={:.4} hh_topk_len={}",
                last.bos_ratio,
                last.bos_score,
                last.non_bos_mean,
                last.hh_topk.len()
            );
        }
    }

    // ── 6. Output results ──
    let wall_time = overall_start.elapsed().as_secs_f64();
    let ppl = (total_nll / nll_count as f64).exp();
    let avg_nll = total_nll / nll_count as f64;
    let tok_per_sec = nll_count as f64 / wall_time;

    // Compute summary stats from all eviction events (v3)
    let n_evictions = qcf_events.len();
    let qcf_sum_value_aware: f64 = qcf_events
        .iter()
        .filter_map(|e| e["qcf_value_aware"].as_f64())
        .sum();
    let qcf_max_value_aware: f64 = qcf_events
        .iter()
        .filter_map(|e| e["qcf_value_aware"].as_f64())
        .fold(0.0f64, f64::max);

    let output = serde_json::json!({
        "ppl": ppl,
        "total_nll": total_nll,
        "token_count": nll_count,
        "tokens_per_second": tok_per_sec,
        "wall_time_s": wall_time,
        "n_evictions": n_evictions,
        "qcf_events": qcf_events,
        "qcf_sum_value_aware": qcf_sum_value_aware,
        "qcf_max_value_aware": qcf_max_value_aware,
        "config": {
            "model": args.model_path,
            "text_file": text_file,
            "eviction_policy": args.eviction_policy(),
            "kv_budget": args.kv_budget(),
            "kv_type": args.kv_type,
            "max_seq_len": max_seq_len,
            "eviction_target_ratio": args.eviction_target_ratio(),
            "h2o_keep_ratio": args.keep_ratio(),
            "protected_prefix": protected_prefix,
            "skip_layers": args.skip_layers,
            "skip_ratio": args.skip_ratio,
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    eprintln!(
        "\n[PPL] Final: PPL={:.4}, NLL={:.4}, tokens={}, {:.1} tok/s, {:.1}s",
        ppl, total_nll, nll_count, tok_per_sec, wall_time
    );

    // LISWAP-PPL: per-token NLL CSV dump (token_idx is text-absolute, identical
    // across scenarios for direct curve comparison).
    if let Some(csv_path) = args.ppl_nll_csv.as_ref() {
        use std::io::Write;
        let mut f = std::fs::File::create(csv_path)?;
        writeln!(f, "phase,token_idx,token_id,nll,swap_state,layers_swapped")?;
        for (phase, idx, id, nll, state, n) in &per_token_log {
            writeln!(f, "{},{},{},{:.6},{},{}", phase, idx, id, nll, state, n)?;
        }
        f.flush()?;
        eprintln!(
            "[PPL] Per-token NLL CSV: {} ({} rows)",
            csv_path.display(),
            per_token_log.len()
        );
    }

    Ok(PplResult {
        ppl,
        avg_nll,
        n_eval_tokens: nll_count,
        wall_time_s: wall_time,
    })
}
