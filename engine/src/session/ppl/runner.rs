//! Phase 4-C-2: `run_ppl_dispatch` + `run_ppl` + `run_quant_window_ppl` —
//! `bin/generate.rs::main()`의 PPL 분기 + 두 free fn 본문 외과적 이동.

use std::sync::Arc;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::capability::quant_attn::QuantAttnBackend;
use crate::inference::sampling::{self};
use crate::inference::signal_runtime::SignalRuntime;
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
        auto_eviction,
    } = ctx;

    let ppl_path = args
        .ppl
        .as_deref()
        .expect("run_ppl_dispatch only called when args.ppl is Some");
    run_ppl(
        &args,
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
    )?;

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
                cache_self_need_scores,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: None,
                head_mask: None,
                duo_heads: None,
                q_rows: None,
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
                cache_self_need_scores,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: None,
                head_mask: None,
                duo_heads: None,
                q_rows: None,
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

    let output = serde_json::json!({
        "ppl": ppl,
        "total_nll": total_nll,
        "token_count": nll_count,
        "tokens_per_second": tok_per_sec,
        "wall_time_s": wall_time,
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
    score_accumulator: &mut Option<SignalRuntime>,
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
    let mut eviction_events: Vec<serde_json::Value> = Vec::new();
    // score-decay 측정(KV roadmap 항목 0 §4.2): --dump-a2sf 지정 시 eviction 직전 + run 종료 시 score
    // accumulator importance 에서 BOS/non-BOS ratio + HH(top-k) 집합 스냅샷을 누적(읽기 전용).
    let mut score_decay_snapshots: Vec<crate::session::score_decay_dump::ScoreDecaySnapshot> =
        Vec::new();
    let overall_start = std::time::Instant::now();

    // Per-token NLL log: (phase, token_idx, token_id, nll).
    let mut per_token_log: Vec<(&'static str, usize, u32, f64)> = Vec::new();
    let log_per_token = args.ppl_nll_csv.is_some();

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

        if let Some(acc) = score_accumulator.as_mut().and_then(|rt| rt.acc_mut()) {
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
                score_accumulator: score_accumulator.as_mut().and_then(|rt| rt.acc_mut()),
                query_stats_accumulator: None,
                skip_config,
                cache_self_need_scores: false,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: None,
                head_mask: None,
                duo_heads: None,
                q_rows: None,
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
                per_token_log.push(("prefill", i, token_ids[i + 1], -lp));
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
    for i in prefill_len..eval_tokens - 1 {
        let input_token = token_ids[i];
        let target_token = token_ids[i + 1];

        // Score accumulator begin step
        if let Some(acc) = score_accumulator.as_mut().and_then(|rt| rt.acc_mut()) {
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
                score_accumulator: score_accumulator.as_mut().and_then(|rt| rt.acc_mut()),
                query_stats_accumulator: None,
                skip_config,
                cache_self_need_scores: false,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: None,
                head_mask: None,
                duo_heads: None,
                q_rows: None,
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
            per_token_log.push(("decode", i, target_token, -lp));
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
                if let Some(rt) = score_accumulator.as_mut() {
                    rt.ensure_coherent(backend.as_ref());
                }

                // score-decay 측정: eviction(+ acc.reset) 직전 importance 스냅샷(읽기 전용). budget=top-k.
                if args.dump_a2sf.is_some()
                    && let Some(acc) = score_accumulator.as_ref().and_then(|rt| rt.view())
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

                // Perform eviction — shared score-fed body (extract → route; force, ratio).
                // GPU scores were synced into the CPU accumulator above, so `extract_scores`
                // reads real importance on both CPU and GPU backends.
                use crate::kv::eviction::score_fed;
                let extracted = if score_based_eviction {
                    score_accumulator.as_ref().and_then(|rt| rt.extract())
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

                    eviction_events.push(serde_json::json!({
                        "step": i,
                        "tokens_evicted": result.tokens_removed,
                        "eviction_ratio": eviction_ratio,
                        "ppl_at_step": ppl_at_event,
                    }));

                    // IMPORTANT: Do NOT reset start_pos to current_pos after eviction.
                    // After shift_positions(), cached K vectors retain their original RoPE
                    // positions. start_pos must continue incrementing from the original
                    // position to maintain correct RoPE relative distances. Using current_pos
                    // (compacted) creates a RoPE discontinuity where cached tokens appear
                    // as "future" tokens, causing severe NLL degradation.
                    // start_pos continues via `start_pos += 1` in the main loop.
                    if let Some(rt) = score_accumulator.as_mut() {
                        // Reset the CPU accumulator AND the GPU accumulator in lockstep (mirrors
                        // eval-ll/bench). Without this the GPU importance keeps accumulating since
                        // prefill while the CPU twin starts fresh, so a 2nd eviction's synced scores
                        // (and the a2sf dump) would diverge from the CPU oracle. No-op on CPU / unarmed.
                        rt.reset(backend.as_ref());
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
        if let Some(rt) = score_accumulator.as_mut() {
            rt.ensure_coherent(backend.as_ref());
        }
        if let Some(acc) = score_accumulator.as_ref().and_then(|rt| rt.view())
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
    let n_evictions = eviction_events.len();

    let output = serde_json::json!({
        "ppl": ppl,
        "total_nll": total_nll,
        "token_count": nll_count,
        "tokens_per_second": tok_per_sec,
        "wall_time_s": wall_time,
        "n_evictions": n_evictions,
        "eviction_events": eviction_events,
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
        writeln!(f, "phase,token_idx,token_id,nll")?;
        for (phase, idx, id, nll) in &per_token_log {
            writeln!(f, "{},{},{},{:.6}", phase, idx, id, nll)?;
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
