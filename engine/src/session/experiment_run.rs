//! AB-0: argus_bench experiment 경로.
//!
//! [`run_standard_happy_path`](crate::session::standard_happy) 와 동일한
//! prefill→sample→run 골격에 per-token JSONL writer + `[Experiment] Done`
//! summary + suspend 로그를 더한다. verify 하네스가 소비하는 산출물:
//! - `--experiment-output` JSONL: token record(token_id) + `_summary` record.
//!   verify 는 token record 를 세고(`count_decoded_tokens`) token_id 를
//!   재디코딩(accuracy)하며 `_summary.avg_tbt_ms` 로 performance 를 본다.
//! - stderr: `[Resilience] Inference suspended ...` (Suspend), `[Experiment] Done`.
//!
//! resilience directive(throttle/target_tbt/suspend) 의 런타임 효과는
//! [`DecodeLoop::run`](crate::session::DecodeLoop) 가 `ExecutionPlan` 을 읽어
//! 적용하므로 본 경로는 별도 처리하지 않는다 (avg_tbt 가 그 효과를 반영).

use tokenizers::Tokenizer;

use crate::experiment::{JsonlWriter, SummaryRecord, SystemSampler, TokenRecord};
use crate::inference::sampling::{self, SamplingConfig};
use crate::session::DecodeLoop;
use crate::session::assembly::{
    SwapWiringConfig, build_bench_loop, build_bench_quant_window_loop, build_local_pressure_source,
    build_resilience_cache_manager,
};
use crate::session::bin_setup::QuantWindowBenchCtx;
use crate::session::cli::Args;
use crate::session::decode_loop::StopReason;
use crate::session::experiment::ScheduleCommandSource;
use crate::session::standard_happy::StandardHappyCtx;

/// AB-2 §5.7.7: quant-window bench 경로 — `run_experiment_path` 의 quant-window 형제.
///
/// `build_bench_quant_window_loop` 로 `QuantWindowForward` + `QuantWindowBitTransitionStage` 배선 `DecodeLoop` 를 조립한 뒤,
/// Standard 경로와 동일한 prefill→sample→run→summary 공통부([`run_decode_loop_experiment`])를 탄다.
/// eviction/swap/partition 미배선(§5.7.7) — `cache_manager`/`pressure_source` 무주입.
pub fn run_quant_window_experiment_path(ctx: QuantWindowBenchCtx) -> anyhow::Result<()> {
    let QuantWindowBenchCtx {
        args,
        backend,
        memory,
        hardware: _,
        model,
        quant_attn,
        tokenizer,
        tokens,
        max_seq_len,
        sampling_config,
        vocab_size,
        initial_bits,
        residual_size,
        resilience,
    } = ctx;

    eprintln!(
        "[argus-bench] quant-window experiment path → DecodeLoop+KiviForward (tokens={}, budget={}, bits={})",
        tokens.len(),
        args.num_tokens,
        initial_bits,
    );

    // R-P0-3: quant-window KV bytes per token ≈ quantized element size (initial_bits/8) across all
    // layers, captured before the model is consumed. First-order — ignores the small fixed f16
    // residual window; the quantized portion dominates at scale. `final_cache_pos * this` is peak_kv_mb.
    let per_token_kv_bytes = model.config.num_hidden_layers as f64
        * 2.0
        * model.config.num_key_value_heads as f64
        * model.config.head_dim as f64
        * (initial_bits as f64 / 8.0);

    let decode_loop = build_bench_quant_window_loop(
        backend,
        memory,
        model,
        &quant_attn,
        initial_bits,
        residual_size,
        max_seq_len,
        sampling_config.clone(),
        resilience,
    )?;

    run_decode_loop_experiment(
        decode_loop,
        &args,
        &tokenizer,
        &tokens,
        max_seq_len,
        &sampling_config,
        vocab_size,
        per_token_kv_bytes,
    )
}

pub fn run_experiment_path(ctx: StandardHappyCtx) -> anyhow::Result<()> {
    let StandardHappyCtx {
        args,
        backend,
        memory,
        hardware,
        model,
        tokenizer,
        kv_caches,
        max_seq_len,
        sampling_config,
        vocab_size,
        resilience,
        tokens,
    } = ctx;

    use crate::hardware::DeviceTarget;
    let cpu_backend_arc = hardware
        .resolve(DeviceTarget::Cpu)
        .expect("Cpu always resolves")
        .0
        .clone();

    eprintln!(
        "[argus-bench] experiment path → DecodeLoop+ModelForward (tokens={}, budget={})",
        tokens.len(),
        args.num_tokens
    );

    // AB-1: CLI `eviction <policy>` 로 resilience force-eviction CacheManager 구성
    // (eviction=none 이면 None → happy-path 동등). plan.evict directive 가 오면
    // decode 루프가 forward.try_evict 로 mid-decode prune.
    let cache_manager = build_resilience_cache_manager(&args, &backend)?;
    // β-5: graded 압력 source 는 cache_manager 가 있을 때(eviction/swap 활성)만 주입한다 —
    // happy-path(eviction=none + swap-dir 없음 → cache_manager=None)는 무주입해 per-token
    // /proc 읽기를 차단한다(G4). source 가 있어도 pressure 소비자(Persistent EvictionStage)가
    // 등록돼 있어야 실제 발화하며, N-step 캐시로 syscall 빈도를 제한한다.
    let pressure_source = cache_manager
        .as_ref()
        .map(|_| build_local_pressure_source(&args, &backend));

    // §5.9.1 Track A: score-based policy(h2o/h2o_plus/d2o)면 AttentionScoreAccumulator 생성.
    // 비-score 정책은 더미 None 셀을 넘긴다.
    let score_cell = {
        use crate::inference::attention_scores::AttentionScoreAccumulator;
        use std::sync::{Arc, Mutex};
        let policy = args.eviction_policy();
        if crate::kv::eviction::stage_registry::stage_is_score_based(policy) {
            // EXPLICIT-REQUIRED: faithful h2o needs `--set hh_size/recent_size` (clean reject, no
            // default budget). No-op for any non-h2o policy.
            args.require_h2o_budgets()?;
            let n_layers = model.config.num_hidden_layers;
            let n_kv_heads = model.config.num_key_value_heads;
            crate::inference::attention_scores::ensure_score_producers_registered()?;
            let n_heads = model.config.num_attention_heads;
            let mut acc = AttentionScoreAccumulator::new_gqa(
                max_seq_len,
                n_heads,
                n_kv_heads,
                n_layers,
                args.h2o_tracked_layers(),
                args.h2o_decay(),
            );
            acc.set_active(true);
            // Faithful-H2O (a): force time_normalize OFF for h2o so the large prefill column-sum base
            // is not divided by the decode-only step count. Non-h2o policies keep their behavior.
            let faithful = args.eviction_policy() == "h2o";
            acc.set_time_normalize(!faithful && !args.h2o_raw_scores());
            // Faithful-H2O (b): per-layer FLAT importance (no cross-layer MAX) so each layer evicts
            // on its own heavy hitters. Opt-in (h2o only) → off elsewhere = byte-identical.
            if faithful {
                acc.enable_per_layer_flat();
            }
            // GPU-side accumulator init (OpenCL only) — mirrors `eval_setup`/`chat`. Arming the GPU
            // score path makes `gpu_score_active=true`, so the flash kernel emits per-key scores
            // on-device (a byproduct of the softmax it already computes) instead of forcing a
            // per-token GPU→CPU readback — that readback is what makes bench score-based decode ~2×.
            // The bench `EvictionStage` syncs these back (`score_fed::sync_gpu_scores_to_cpu`) before
            // a score-fed eviction reads them, so the keep-set matches the CPU path.
            #[cfg(feature = "opencl")]
            if let Some(ocl_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            {
                // Log the init result (eval_setup 동형, chat 의 `let _ =` 와 의도적으로 다름): bench 는
                // measurement tool 이라 init 실패 시 silent 하게 느린 CPU-accumulate 로 폴백하면 그 자체가
                // 이 fix 가 없애려는 ~2× 로 보여 측정을 오염시킨다. set_active 는 Ok 일 때만 호출해
                // 실패 후 stale buffer 무장도 피한다.
                match ocl_be.init_gpu_score_acc(
                    n_layers,
                    n_heads,
                    n_kv_heads,
                    max_seq_len,
                    args.h2o_decay(),
                ) {
                    Ok(()) => {
                        if let Some(gpu_acc) = ocl_be.gpu_score_acc_mut() {
                            gpu_acc.set_active(true);
                        }
                        eprintln!(
                            "[GPU Score] Accumulator initialized — per-token readback eliminated"
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[GPU Score] Failed to initialize (falling back to CPU path): {e}"
                        );
                    }
                }
            }
            // CUDA twin of the GPU-accumulator init (discrete-GPU / Jetson).
            #[cfg(feature = "cuda")]
            if let Some(cuda_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::cuda_pc::CudaBackend>()
            {
                match cuda_be.init_gpu_score_acc(
                    n_layers,
                    n_heads,
                    n_kv_heads,
                    max_seq_len,
                    args.h2o_decay(),
                ) {
                    Ok(()) => {
                        if let Some(gpu_acc) = cuda_be.gpu_score_acc_mut() {
                            gpu_acc.set_active(true);
                        }
                        eprintln!(
                            "[GPU Score] CUDA accumulator initialized — per-token readback eliminated"
                        );
                    }
                    Err(e) => {
                        eprintln!("[GPU Score] CUDA init failed (falling back to CPU path): {e}");
                    }
                }
            }
            Arc::new(Mutex::new(Some(acc)))
        } else {
            Arc::new(Mutex::new(None))
        }
    };

    // R-P0-3: KV bytes per occupied token, captured before build_bench_loop consumes kv_caches.
    // A buffer holds `capacity` token slots, so `(k+v bytes) / capacity` is the exact per-token
    // cost — reading real buffer sizes accounts for f16 / q4_0 / opaque formats alike, and is
    // independent of how much of `max_seq_len` is currently allocated. `final_cache_pos * this`
    // is peak_kv_mb.
    let per_token_kv_bytes: f64 = kv_caches
        .iter()
        .filter(|c| c.capacity() > 0)
        .map(|c| (c.k_buffer.size() + c.v_buffer.size()) as f64 / c.capacity() as f64)
        .sum();

    // bin_setup이 dispatch한 kv_caches를 소비(과거엔 drop 후 typed 재할당).
    let decode_loop = build_bench_loop(
        backend.clone(),
        memory.clone(),
        cpu_backend_arc.clone(),
        hardware.clone(),
        model,
        kv_caches,
        max_seq_len,
        sampling_config.clone(),
        !args.no_gpu_plan,
        resilience,
        cache_manager,
        pressure_source,
        args.eviction_target_ratio(),
        None, // γ-3b: argus-bench 는 schedule 없음 (IPC resilience 만)
        // AB-6: swap dispatch 설정. `--swap` 미지정 시 Incremental(LISWAP-6 production winner).
        SwapWiringConfig {
            default_mode: args
                .swap
                .unwrap_or(crate::session::cli::SwapMode::Incremental),
            phase_chunk_size_bytes: args.swap_phase_aware_chunk_mb * 1024 * 1024,
            phase_max_chunks_per_token: args.swap_phase_aware_max_chunks_per_token,
        },
        score_cell,
        // Faithful-H2O (c): arm the prefill seed when eviction == "h2o".
        args.eviction_policy() == "h2o",
    )?;

    run_decode_loop_experiment(
        decode_loop,
        &args,
        &tokenizer,
        &tokens,
        max_seq_len,
        &sampling_config,
        vocab_size,
        per_token_kv_bytes,
    )
}

/// AB-2 §5.7.7: prefill→sample→run→summary 공통부 (`run_experiment_path` / `run_quant_window_experiment_path`
/// 공유). 조립된 `DecodeLoop` 를 받아 generation + JSONL/summary 산출까지 수행한다.
#[allow(clippy::too_many_arguments)]
fn run_decode_loop_experiment(
    mut decode_loop: DecodeLoop,
    args: &Args,
    tokenizer: &Tokenizer,
    tokens: &[u32],
    max_seq_len: usize,
    sampling_config: &SamplingConfig,
    vocab_size: usize,
    // R-P0-3: KV-cache bytes per occupied token (across all layers). `peak_kv_mb` is
    // `final_cache_pos * per_token_kv_bytes`. 0.0 → `peak_kv_mb` is omitted.
    per_token_kv_bytes: f64,
) -> anyhow::Result<()> {
    let mut sys_sampler = SystemSampler::new(args.experiment_sample_interval);
    let sys_start = args
        .experiment_output
        .as_ref()
        .map(|_| sys_sampler.snapshot());

    let t_prefill = std::time::Instant::now();
    let mut last_logits = decode_loop.prefill(tokens)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let first_token = sampling::sample(&mut last_logits, tokens, vocab_size, sampling_config, None);

    let t_decode = std::time::Instant::now();
    let result = decode_loop.run(args.num_tokens - 1, first_token)?;
    let decode_total_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

    // Suspend 시 break → CommandRequested. legacy 와 동일 문자열을 emit 하여
    // verify thermal_emergency_suspend 의 stderr_pattern 을 충족한다.
    if result.stopped_by == StopReason::CommandRequested {
        eprintln!("\n[Resilience] Inference suspended by system signal");
    }

    let mut final_tokens: Vec<u32> = tokens.to_vec();
    final_tokens.push(first_token);
    final_tokens.extend_from_slice(&result.tokens_generated);
    let decoded = tokenizer
        .decode(&final_tokens, true)
        .unwrap_or_else(|_| String::from("[decode error]"));
    // `--bench-json`: keep stdout pure JSON (emitted below), so the generated text and the
    // human-readable metric lines go to stderr. Without it, behavior is unchanged.
    if args.bench_json {
        eprintln!("{}", decoded);
    } else {
        println!("{}", decoded);
    }

    let decode_tokens = result.tokens_generated.len();
    let total_gen = 1 + decode_tokens;
    let decode_per_tok = if decode_tokens > 0 {
        decode_total_ms / decode_tokens as f64
    } else {
        0.0
    };
    let avg_tbt = (prefill_ms + decode_total_ms) / total_gen as f64;
    let decode_tok_s = 1000.0 / decode_per_tok.max(0.001);
    macro_rules! metric_line {
        ($($arg:tt)*) => {
            if args.bench_json { eprintln!($($arg)*) } else { println!($($arg)*) }
        };
    }
    metric_line!("TTFT: {:.2} ms", prefill_ms);
    if decode_tokens > 0 {
        metric_line!(
            "Decode: {:.2} ms/tok ({:.1} tok/s) [{} tokens]",
            decode_per_tok,
            decode_tok_s,
            decode_tokens,
        );
    }
    metric_line!(
        "Avg TBT: {:.2} ms ({:.1} tokens/sec)",
        avg_tbt,
        1000.0 / avg_tbt.max(0.001),
    );

    // R-P0-3: single-line metrics JSON to stdout for the validation harness (Gate 3).
    // `tokens_per_sec` is the steady-state decode throughput (reciprocal of decode_ms_per_tok);
    // `peak_kv_mb` is the KV-cache footprint at final occupancy (omitted when geometry is unknown).
    if args.bench_json {
        let mut metrics = serde_json::json!({
            "decode_ms_per_tok": decode_per_tok,
            "prefill_ms": prefill_ms,
            "tokens_per_sec": decode_tok_s,
            "avg_tbt_ms": avg_tbt,
            "decode_tokens": decode_tokens,
            "prompt_len": tokens.len(),
            "final_cache_pos": result.final_cache_pos,
            "eviction_policy": args.eviction_policy(),
            "kv_mode": args.effective_kv_mode(),
        });
        if per_token_kv_bytes > 0.0 {
            // peak_kv_mb 는 KV-cache 점유(final_cache_pos) 기반 — eviction 후 누적 final_pos 로
            // 계산하면 over-report 된다(eviction-후 RoPE-drift 수정으로 둘이 분리됨).
            let peak_kv_mb = result.final_cache_pos as f64 * per_token_kv_bytes / (1024.0 * 1024.0);
            metrics["peak_kv_mb"] = serde_json::json!(peak_kv_mb);
        }
        println!("{}", serde_json::to_string(&metrics)?);
    }

    // ── experiment JSONL: per-token record + _summary ──
    if let Some(path) = args.experiment_output.as_ref() {
        let prompt_len = tokens.len();
        let generated: Vec<u32> = std::iter::once(first_token)
            .chain(result.tokens_generated.iter().copied())
            .collect();

        let mut writer = JsonlWriter::new(path)?;
        for (i, &token_id) in generated.iter().enumerate() {
            let pos = prompt_len + i;
            // per-token wall-clock 분해는 보존하지 않는다 — verify 는 token_id 와
            // record 수만 소비하므로 평균값으로 채운다.
            let (tbt_ms, forward_ms) = if i == 0 {
                (prefill_ms, prefill_ms)
            } else {
                (decode_per_tok, decode_per_tok)
            };
            let record = TokenRecord {
                pos,
                token_id,
                text: String::new(),
                tbt_ms,
                forward_ms,
                signal: None,
                actions: Vec::new(),
                cache_pos: pos,
                throttle_ms: 0,
                top_logits: Vec::new(),
                sys: sys_sampler.sample(pos),
            };
            writer.write_token(&record)?;
        }

        let prompt_text = tokenizer
            .decode(tokens, true)
            .unwrap_or_else(|_| String::new());
        let summary = SummaryRecord {
            _summary: true,
            total_tokens: total_gen,
            ttft_ms: prefill_ms,
            avg_tbt_ms: avg_tbt,
            avg_forward_ms: decode_per_tok,
            total_throttle_ms: 0,
            eviction_count: 0,
            evicted_tokens_total: 0,
            final_cache_pos: result.final_cache_pos,
            max_seq_len,
            prompt: prompt_text,
            schedule_name: String::new(),
            eviction_policy: args.eviction_policy().to_string(),
            backend: args.backend.clone(),
            sample_interval: args.experiment_sample_interval,
            sys_start,
            sys_end: Some(sys_sampler.snapshot()),
            governor: Some(SystemSampler::read_governor()),
        };
        writer.write_summary(&summary)?;

        eprintln!(
            "[Experiment] Done: {} tokens, avg TBT {:.2}ms, {} evictions",
            total_gen, avg_tbt, 0
        );
    }

    eprintln!(
        "[argus-bench] generated={} (first={} + run={}) stopped_by={:?} final_pos={}",
        total_gen, first_token, decode_tokens, result.stopped_by, result.final_pos
    );
    Ok(())
}

/// γ-3b: argus-eval experiment 모드 — 정적 `ScheduleCommandSource` 를 β-4 CommandSource
/// seam 에 주입하여 generation 실행. `run_experiment_path` 와 동일한 prefill→decode 골격에
/// schedule-driven directive 를 더한다.
///
/// ## JSONL 산출
///
/// `--experiment-output` 이 지정된 경우 `run_experiment_path` 와 동일한 JSONL + `_summary`
/// 레코드를 기록한다. verify 하네스 호환.
pub fn run_experiment_schedule_path(
    ctx: StandardHappyCtx,
    schedule_source: ScheduleCommandSource,
) -> anyhow::Result<()> {
    let StandardHappyCtx {
        args,
        backend,
        memory,
        hardware,
        model,
        tokenizer,
        kv_caches,
        max_seq_len,
        sampling_config,
        vocab_size,
        resilience,
        tokens,
    } = ctx;

    use crate::hardware::DeviceTarget;
    let cpu_backend_arc = hardware
        .resolve(DeviceTarget::Cpu)
        .expect("Cpu always resolves")
        .0
        .clone();

    eprintln!(
        "[argus-eval] experiment path → ScheduleCommandSource (tokens={}, budget={})",
        tokens.len(),
        args.num_tokens
    );

    let mut sys_sampler = SystemSampler::new(args.experiment_sample_interval);
    let sys_start = args
        .experiment_output
        .as_ref()
        .map(|_| sys_sampler.snapshot());

    let cache_manager = build_resilience_cache_manager(&args, &backend)?;
    let pressure_source = cache_manager
        .as_ref()
        .map(|_| build_local_pressure_source(&args, &backend));

    // §5.9.1 Track A: schedule 모드도 score-based policy 시 accumulator 생성.
    let score_cell = {
        use crate::inference::attention_scores::AttentionScoreAccumulator;
        use std::sync::{Arc, Mutex};
        let policy = args.eviction_policy();
        if crate::kv::eviction::stage_registry::stage_is_score_based(policy) {
            // EXPLICIT-REQUIRED: faithful h2o needs `--set hh_size/recent_size` (clean reject, no
            // default budget). No-op for any non-h2o policy.
            args.require_h2o_budgets()?;
            let n_layers = model.config.num_hidden_layers;
            let n_kv_heads = model.config.num_key_value_heads;
            crate::inference::attention_scores::ensure_score_producers_registered()?;
            let n_heads = model.config.num_attention_heads;
            let mut acc = AttentionScoreAccumulator::new_gqa(
                max_seq_len,
                n_heads,
                n_kv_heads,
                n_layers,
                args.h2o_tracked_layers(),
                args.h2o_decay(),
            );
            acc.set_active(true);
            // Faithful-H2O (a): force time_normalize OFF for h2o so the large prefill column-sum base
            // is not divided by the decode-only step count. Non-h2o policies keep their behavior.
            let faithful = args.eviction_policy() == "h2o";
            acc.set_time_normalize(!faithful && !args.h2o_raw_scores());
            // Faithful-H2O (b): per-layer FLAT importance (no cross-layer MAX) so each layer evicts
            // on its own heavy hitters. Opt-in (h2o only) → off elsewhere = byte-identical.
            if faithful {
                acc.enable_per_layer_flat();
            }
            // GPU-side accumulator init (OpenCL only) — mirrors `eval_setup`/`chat`. Arming the GPU
            // score path makes `gpu_score_active=true`, so the flash kernel emits per-key scores
            // on-device (a byproduct of the softmax it already computes) instead of forcing a
            // per-token GPU→CPU readback — that readback is what makes bench score-based decode ~2×.
            // The bench `EvictionStage` syncs these back (`score_fed::sync_gpu_scores_to_cpu`) before
            // a score-fed eviction reads them, so the keep-set matches the CPU path.
            #[cfg(feature = "opencl")]
            if let Some(ocl_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            {
                // Log the init result (eval_setup 동형, chat 의 `let _ =` 와 의도적으로 다름): bench 는
                // measurement tool 이라 init 실패 시 silent 하게 느린 CPU-accumulate 로 폴백하면 그 자체가
                // 이 fix 가 없애려는 ~2× 로 보여 측정을 오염시킨다. set_active 는 Ok 일 때만 호출해
                // 실패 후 stale buffer 무장도 피한다.
                match ocl_be.init_gpu_score_acc(
                    n_layers,
                    n_heads,
                    n_kv_heads,
                    max_seq_len,
                    args.h2o_decay(),
                ) {
                    Ok(()) => {
                        if let Some(gpu_acc) = ocl_be.gpu_score_acc_mut() {
                            gpu_acc.set_active(true);
                        }
                        eprintln!(
                            "[GPU Score] Accumulator initialized — per-token readback eliminated"
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[GPU Score] Failed to initialize (falling back to CPU path): {e}"
                        );
                    }
                }
            }
            // CUDA twin of the GPU-accumulator init (discrete-GPU / Jetson).
            #[cfg(feature = "cuda")]
            if let Some(cuda_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::cuda_pc::CudaBackend>()
            {
                match cuda_be.init_gpu_score_acc(
                    n_layers,
                    n_heads,
                    n_kv_heads,
                    max_seq_len,
                    args.h2o_decay(),
                ) {
                    Ok(()) => {
                        if let Some(gpu_acc) = cuda_be.gpu_score_acc_mut() {
                            gpu_acc.set_active(true);
                        }
                        eprintln!(
                            "[GPU Score] CUDA accumulator initialized — per-token readback eliminated"
                        );
                    }
                    Err(e) => {
                        eprintln!("[GPU Score] CUDA init failed (falling back to CPU path): {e}");
                    }
                }
            }
            Arc::new(Mutex::new(Some(acc)))
        } else {
            Arc::new(Mutex::new(None))
        }
    };

    let mut decode_loop = build_bench_loop(
        backend.clone(),
        memory.clone(),
        cpu_backend_arc.clone(),
        hardware.clone(),
        model,
        kv_caches,
        max_seq_len,
        sampling_config.clone(),
        !args.no_gpu_plan,
        resilience,
        cache_manager,
        pressure_source,
        args.eviction_target_ratio(),
        Some(schedule_source),
        // AB-6: swap dispatch 설정 (schedule 모드도 secondary 보유 시 swap 활성).
        SwapWiringConfig {
            default_mode: args
                .swap
                .unwrap_or(crate::session::cli::SwapMode::Incremental),
            phase_chunk_size_bytes: args.swap_phase_aware_chunk_mb * 1024 * 1024,
            phase_max_chunks_per_token: args.swap_phase_aware_max_chunks_per_token,
        },
        score_cell,
        // Faithful-H2O (c): arm the prefill seed when eviction == "h2o".
        args.eviction_policy() == "h2o",
    )?;

    let t_prefill = std::time::Instant::now();
    let mut last_logits = decode_loop.prefill(&tokens)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let first_token = sampling::sample(
        &mut last_logits,
        &tokens,
        vocab_size,
        &sampling_config,
        None,
    );

    let t_decode = std::time::Instant::now();
    let result = decode_loop.run(args.num_tokens - 1, first_token)?;
    let decode_total_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

    if result.stopped_by == StopReason::CommandRequested {
        eprintln!("\n[Resilience] Inference suspended by system signal");
    }

    let mut final_tokens: Vec<u32> = tokens.clone();
    final_tokens.push(first_token);
    final_tokens.extend_from_slice(&result.tokens_generated);
    let decoded = tokenizer
        .decode(&final_tokens, true)
        .unwrap_or_else(|_| String::from("[decode error]"));
    println!("{}", decoded);

    let decode_tokens = result.tokens_generated.len();
    let total_gen = 1 + decode_tokens;
    let decode_per_tok = if decode_tokens > 0 {
        decode_total_ms / decode_tokens as f64
    } else {
        0.0
    };
    let avg_tbt = (prefill_ms + decode_total_ms) / total_gen as f64;
    println!("TTFT: {:.2} ms", prefill_ms);
    if decode_tokens > 0 {
        println!(
            "Decode: {:.2} ms/tok ({:.1} tok/s) [{} tokens]",
            decode_per_tok,
            1000.0 / decode_per_tok.max(0.001),
            decode_tokens,
        );
    }
    println!(
        "Avg TBT: {:.2} ms ({:.1} tokens/sec)",
        avg_tbt,
        1000.0 / avg_tbt.max(0.001),
    );

    if let Some(path) = args.experiment_output.as_ref() {
        let prompt_len = tokens.len();
        let generated: Vec<u32> = std::iter::once(first_token)
            .chain(result.tokens_generated.iter().copied())
            .collect();

        let mut writer = JsonlWriter::new(path)?;
        for (i, &token_id) in generated.iter().enumerate() {
            let pos = prompt_len + i;
            let (tbt_ms, forward_ms) = if i == 0 {
                (prefill_ms, prefill_ms)
            } else {
                (decode_per_tok, decode_per_tok)
            };
            let record = TokenRecord {
                pos,
                token_id,
                text: String::new(),
                tbt_ms,
                forward_ms,
                signal: None,
                actions: Vec::new(),
                cache_pos: pos,
                throttle_ms: 0,
                top_logits: Vec::new(),
                sys: sys_sampler.sample(pos),
            };
            writer.write_token(&record)?;
        }

        let prompt_text = tokenizer
            .decode(&tokens, true)
            .unwrap_or_else(|_| String::new());
        let summary = SummaryRecord {
            _summary: true,
            total_tokens: total_gen,
            ttft_ms: prefill_ms,
            avg_tbt_ms: avg_tbt,
            avg_forward_ms: decode_per_tok,
            total_throttle_ms: 0,
            eviction_count: 0,
            evicted_tokens_total: 0,
            final_cache_pos: result.final_cache_pos,
            max_seq_len,
            prompt: prompt_text,
            schedule_name: String::new(),
            eviction_policy: args.eviction_policy().to_string(),
            backend: args.backend.clone(),
            sample_interval: args.experiment_sample_interval,
            sys_start,
            sys_end: Some(sys_sampler.snapshot()),
            governor: Some(SystemSampler::read_governor()),
        };
        writer.write_summary(&summary)?;

        eprintln!(
            "[Experiment] Done: {} tokens, avg TBT {:.2}ms",
            total_gen, avg_tbt,
        );
    }

    eprintln!(
        "[argus-eval] experiment generated={} (first={} + run={}) stopped_by={:?} final_pos={}",
        total_gen, first_token, decode_tokens, result.stopped_by, result.final_pos
    );
    Ok(())
}
