//! Phase 4-B-1: `session::qcf_runtime` — bin/generate.rs에서 분산된
//! shared helper fn (`run_qcf_warmup_workflow`, `run_layer_swap` +
//! `read_allow_boundary_env`, `QcfWarmupResult` struct)을 lib level로
//! 이동하여 session/eval/, session/ppl/, session/batch/, standard
//! generate 분기에서 공통 호출 가능하게 한다.
//!
//! lift-and-shift: 본문 변경 없음. `crate::X` → `crate::X` 만 적용.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::format::KVCacheFormat;
use crate::kv::kv_cache::KVCache;
use crate::kv::quant_window_format::QuantWindowFormat;
use crate::kv::standard_format::StandardFormat;
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::TransformerModelForwardArgs;
use crate::qcf_collector::ImportanceLookup;
use crate::session::eval::EvalCacheKind;
use crate::shape::Shape;
use crate::tensor::Tensor;

pub struct QcfWarmupResult {
    pub importance: crate::qcf::ImportanceTable,
    pub decision: Option<crate::weight::SwapDecision>,
    /// Per-layer DP-LLM proxy ε (single-tensor relative `attn_output`).
    /// `Some` only in compare mode.
    pub dpllm_epsilon: Option<Vec<f32>>,
    /// §4 candidate A: per-layer DP-LLM ε summed across 6 attn+MLP tensors.
    pub dpllm_epsilon_multi: Option<Vec<f32>>,
    /// §4 candidate D: per-layer DP-LLM ε without the `‖W·x‖` normalisation
    /// (absolute L2 of the activation difference).
    pub dpllm_epsilon_abs: Option<Vec<f32>>,
    /// §4 candidate E: QCF-style multiplicative composition `ε_v × ε_o`.
    pub dpllm_epsilon_qcf: Option<Vec<f32>>,
    /// §4.2 F4: cascade-aware single output perturbation.
    /// `‖(W_o^F16 − W_o^Q4) · V_out^F16‖_F / ‖W_o^F16 · V_out^F16‖_F`.
    pub direct_attn_f4: Option<Vec<f32>>,
    /// §4.2 F5: direct attention output relative L2 perturbation.
    /// `‖O^F16 − O^Q4‖_F / ‖O^F16‖_F`.
    pub direct_attn_f5: Option<Vec<f32>>,
    /// §4.2 decode-only F5: per-layer F5 evaluated with X = the N decode-step
    /// raws (T = N) only (no prefill X). `Some` only when `--decode-x-steps >
    /// 0` and a secondary GGUF was loaded.
    pub direct_attn_f5_decode_only: Option<Vec<f32>>,
    /// §4.2 prefill+decode F5: per-layer F5 evaluated with X =
    /// concat(prefill raws, decode raws) (T = 256 + N). `Some` only when
    /// `--decode-x-steps > 0` and a secondary GGUF was loaded.
    pub direct_attn_f5_prefill_decode: Option<Vec<f32>>,
}

pub fn run_layer_swap(
    model: &crate::models::transformer::TransformerModel,
    target_layers: &[usize],
    gpu_backend: Option<&Arc<dyn Backend>>,
    cpu_backend: &Arc<dyn Backend>,
    async_dispatcher: Option<&crate::weight::AsyncSwapDispatcher>,
    #[cfg(feature = "opencl")] host_ptr_pool: Option<
        Arc<crate::backend::opencl::host_ptr_pool::HostPtrPool>,
    >,
    #[cfg(feature = "cuda-embedded")] layer_pool: Option<
        Arc<dyn crate::layers::staging_pool::WeightStagingPool>,
    >,
    #[cfg(feature = "cuda-embedded")] mmap_registration: Option<
        Arc<crate::memory::cuda::mmap::CudaMmapRegistration>,
    >,
) -> Result<crate::weight::SwapReport, crate::weight::SwapError> {
    let swap_memory = Galloc::new();
    let swap_backend: Arc<dyn Backend> =
        gpu_backend.cloned().unwrap_or_else(|| cpu_backend.clone());
    // ENG-ALG-228: attach the model's async release worker so Stage (c) enqueues
    // displaced LayerWeights for background drop instead of blocking inline.
    let executor = crate::weight::SwapExecutor::new_with_worker(
        DType::Q4_0,
        &model.config,
        swap_backend,
        &swap_memory,
        Arc::clone(&model.release_worker),
    );
    // LISWAP-3 prototype: if a host_ptr pool is supplied, attach it so the
    // AOS materialise path uses the zero-copy slot pool.
    #[cfg(feature = "opencl")]
    let executor = match host_ptr_pool {
        Some(pool) => executor.with_host_ptr_pool(pool),
        None => executor,
    };
    // LISWAP-8 Phase B: attach pre-allocated layer object pool when set.
    #[cfg(feature = "cuda-embedded")]
    let executor = match layer_pool {
        Some(pool) => executor.with_layer_pool(pool),
        None => executor,
    };
    // LISWAP-8 Hammer D: attach mmap registration when set.
    #[cfg(feature = "cuda-embedded")]
    let executor = match mmap_registration {
        Some(reg) => executor.with_mmap_registration(reg),
        None => executor,
    };
    executor.execute_on_slots(
        model.layers.as_slice(),
        model.secondary_mmap.as_ref(),
        &model.ratio_generation,
        target_layers,
        async_dispatcher,
    )
}

/// F16 recall 전용 `SwapExecutor` 진입점 (ENG-ALG-240).
///
/// `run_layer_swap`과 동형이나 `DType::F16`을 target으로 하고,
/// `f16_secondary`(recall용 F16 view)를 직접 공급한다.
/// `execute_on_slots`의 idempotent skip(`current_dtype() == F16`)이 이미
/// F16인 layer를 자동 스킵한다(INV-193, 변경 불요).
/// `materialise_weight`의 SOA fast-path는 Q4_0 조건이라 F16은 자동 AOS 폴백(변경 불요).
pub fn run_layer_recall(
    model: &crate::models::transformer::TransformerModel,
    f16_secondary: &Arc<crate::models::weights::secondary_mmap::SecondaryMmap>,
    target_layers: &[usize],
    cpu_backend: &Arc<dyn Backend>,
    async_dispatcher: Option<&crate::weight::AsyncSwapDispatcher>,
) -> Result<crate::weight::SwapReport, crate::weight::SwapError> {
    let swap_memory = Galloc::new();
    let executor = crate::weight::SwapExecutor::new_with_worker(
        DType::F16,
        &model.config,
        cpu_backend.clone(),
        &swap_memory,
        Arc::clone(&model.release_worker),
    );
    executor.execute_on_slots(
        model.layers.as_slice(),
        Some(f16_secondary),
        &model.ratio_generation,
        target_layers,
        async_dispatcher,
    )
}

/// Re-map weight tensors for CPU access after a weight swap.
///
/// Required when running on GPU with `--secondary-layout aos +
/// --resilience-prealloc-switch`: `SwapExecutor::materialise_tensor` lands an
/// unmapped `UnifiedBuffer` in the new `LayerWeights` snapshot, and the next
/// `switch_hw cpu` directive segfaults on a null host pointer.  Idempotent —
/// already-mapped tensors short-circuit in `map_one`.
pub fn read_allow_boundary_env() -> bool {
    std::env::var("LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Inference-engine references + KV caches + warmup input.
///
/// Groups the "where to run + on what data" inputs for
/// [`run_qcf_warmup_workflow`].
pub struct QcfWarmupCtx<'a> {
    pub model: &'a crate::models::transformer::TransformerModel,
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub kv_caches: &'a mut Vec<KVCache>,
    pub vocab_size: usize,
    pub warmup_ids: &'a [u32],
    pub gpu_backend: Option<&'a Arc<dyn Backend>>,
    pub cpu_backend: &'a Arc<dyn Backend>,
}

/// Behaviour knobs for [`run_qcf_warmup_workflow`] (scalar/option values).
pub struct QcfWarmupConfig<'a> {
    pub force_ratio: Option<f32>,
    pub swap_algorithm: crate::weight::SwapAlgorithm,
    pub execute_swap: bool,
    pub importance_formula: crate::qcf_types::ImportanceFormula,
    pub importance_three_way: bool,
    pub swap_only_layers: Option<&'a [usize]>,
    pub decode_x_steps: usize,
    pub log_prefix: &'a str,
}

pub fn run_qcf_warmup_workflow(
    ctx: QcfWarmupCtx<'_>,
    cfg: QcfWarmupConfig<'_>,
) -> anyhow::Result<QcfWarmupResult> {
    use crate::qcf::ImportanceCollector;
    use crate::weight::WeightSwapDecider;

    let QcfWarmupCtx {
        model,
        backend,
        memory,
        kv_caches,
        vocab_size,
        warmup_ids,
        gpu_backend,
        cpu_backend,
    } = ctx;
    let QcfWarmupConfig {
        force_ratio,
        swap_algorithm,
        execute_swap,
        importance_formula,
        importance_three_way,
        swap_only_layers,
        decode_x_steps,
        log_prefix,
    } = cfg;

    let actual_warmup_len = warmup_ids.len();
    eprintln!(
        "[QCF-dump]{} warmup prefill: {} tokens (formula={}, three_way={}, decode_x_steps={})",
        log_prefix,
        actual_warmup_len,
        importance_formula.as_str(),
        importance_three_way,
        decode_x_steps,
    );

    // ── Warmup prefill with ImportanceCollector (F16 prefill pass) ────────────
    let mut collector =
        ImportanceCollector::new_with_formula(importance_formula, importance_three_way);
    let last_token_logits_argmax: u32;
    {
        let warmup_buf = Galloc::new().alloc(actual_warmup_len * 4, DType::U8)?;
        unsafe {
            let ptr = warmup_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in warmup_ids.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_warmup = Tensor::new(
            Shape::new(vec![1, actual_warmup_len]),
            warmup_buf,
            cpu_backend.clone(),
        );
        let warmup_input = backend.copy_from(&cpu_warmup)?;

        let warmup_logits_buf = memory.alloc(actual_warmup_len * vocab_size * 4, DType::F32)?;
        let mut warmup_logits = Tensor::new(
            Shape::new(vec![1, actual_warmup_len, vocab_size]),
            warmup_logits_buf,
            backend.clone(),
        );

        // Phase α-K ①-d: forward_into → fmt round-trip (warmup prefill, importance 부착).
        KVCache::forward_fmt_roundtrip(kv_caches, |fmts| {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &warmup_input,
                start_pos: 0,
                fmts,
                backend,
                memory,
                logits_out: &mut warmup_logits,
                x_gen: None,
                workspace: None,
                logits_last_only: false,
                score_accumulator: None,
                query_stats_accumulator: None,
                skip_config: None,
                importance_collector: Some(&mut collector),
                cache_self_need_scores: false,
                layer_boundary_hook: None,
                read_stage: None,
            })
        })?;
        backend.synchronize()?;

        // Read the argmax of the last token's logits — first decode-step input.
        last_token_logits_argmax = if decode_x_steps > 0 {
            // Copy the final `[vocab_size]` slice of warmup_logits to host.
            let host_logits = cpu_backend.copy_from(&warmup_logits)?;
            let last_offset = (actual_warmup_len - 1) * vocab_size;
            let host_data = unsafe {
                std::slice::from_raw_parts(
                    host_logits.buffer().as_ptr() as *const f32,
                    actual_warmup_len * vocab_size,
                )
            };
            let last_slice = &host_data[last_offset..last_offset + vocab_size];
            let mut best_idx: u32 = 0;
            let mut best_val: f32 = f32::NEG_INFINITY;
            for (i, &v) in last_slice.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            best_idx
        } else {
            0
        };
    }

    // ── Optional decode-X pass (§4.2 EuroSys'27) ──────────────────────────────
    // After prefill, run `decode_x_steps` greedy decode forwards. Capture
    // per-layer hidden state (T = decode_x_steps per layer after concat) in
    // a fresh `DirectAttn` collector. KV cache is reset after this pass.
    let raws_decode_opt: Option<Vec<(Vec<f32>, usize, usize)>> = if decode_x_steps > 0 {
        if model.secondary_mmap.is_none() {
            eprintln!(
                "[QCF-dump]{} decode-x: SKIPPED (no secondary GGUF loaded)",
                log_prefix
            );
            None
        } else {
            eprintln!(
                "[QCF-dump]{} decode-x: running {} greedy decode steps (seed token id {})",
                log_prefix, decode_x_steps, last_token_logits_argmax,
            );

            let mut collector_decode = ImportanceCollector::new_with_formula(
                crate::qcf_types::ImportanceFormula::DirectAttn,
                false,
            );

            let mut next_tok: u32 = last_token_logits_argmax;

            for step in 0..decode_x_steps {
                let decode_buf = Galloc::new().alloc(4, DType::U8)?;
                unsafe {
                    let ptr = decode_buf.as_mut_ptr() as *mut u32;
                    *ptr = next_tok;
                }
                let cpu_decode =
                    Tensor::new(Shape::new(vec![1, 1]), decode_buf, cpu_backend.clone());
                let decode_input = backend.copy_from(&cpu_decode)?;

                let decode_logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
                let mut decode_logits = Tensor::new(
                    Shape::new(vec![1, 1, vocab_size]),
                    decode_logits_buf,
                    backend.clone(),
                );

                // Phase α-K ①-d: forward_into → fmt round-trip (decode-X, seq_len=1 workspace=None
                // → forward_into 의 발산 A fallthrough = 구 layer.forward→forward_prefill bit-identical).
                KVCache::forward_fmt_roundtrip(kv_caches, |fmts| {
                    model.forward_into(TransformerModelForwardArgs {
                        input_tokens: &decode_input,
                        start_pos: actual_warmup_len + step,
                        fmts,
                        backend,
                        memory,
                        logits_out: &mut decode_logits,
                        x_gen: None,
                        workspace: None,
                        logits_last_only: true,
                        score_accumulator: None,
                        query_stats_accumulator: None,
                        skip_config: None,
                        importance_collector: Some(&mut collector_decode),
                        cache_self_need_scores: false,
                        layer_boundary_hook: None,
                        read_stage: None,
                    })
                })?;
                backend.synchronize()?;

                // argmax for next decode step
                let host_logits = cpu_backend.copy_from(&decode_logits)?;
                let host_data = unsafe {
                    std::slice::from_raw_parts(
                        host_logits.buffer().as_ptr() as *const f32,
                        vocab_size,
                    )
                };
                let mut best_idx: u32 = 0;
                let mut best_val: f32 = f32::NEG_INFINITY;
                for (i, &v) in host_data.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_idx = i as u32;
                    }
                }
                next_tok = best_idx;
            }

            // The collector cached one `[1 × d]` snapshot per layer for *every*
            // decode step (N × n_layers snapshots in chronological order).
            // Rearrange into a per-layer concat: layer i → T = decode_x_steps.
            // build_with_raws returns (table, x_means, raws_per_layer); raws is
            // `Vec<(Vec<f32>, seq_len=1, dim)>` of length N * n_layers.
            let (_table, _xm, raws_chrono) = collector_decode.build_with_raws();
            let n_layers = kv_caches.len();
            let mut per_layer_concat: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(n_layers);
            if raws_chrono.len() == decode_x_steps * n_layers && n_layers > 0 {
                let d = raws_chrono[0].2;
                for li in 0..n_layers {
                    let mut buf: Vec<f32> = Vec::with_capacity(decode_x_steps * d);
                    for step in 0..decode_x_steps {
                        let idx = step * n_layers + li;
                        let (data, t, _d) = &raws_chrono[idx];
                        debug_assert_eq!(*t, 1);
                        buf.extend_from_slice(&data[..d.min(data.len())]);
                    }
                    per_layer_concat.push((buf, decode_x_steps, d));
                }
            } else {
                eprintln!(
                    "[QCF-dump]{} decode-x: WARN raws layout mismatch (chrono len={}, expected {} × {})",
                    log_prefix,
                    raws_chrono.len(),
                    decode_x_steps,
                    n_layers
                );
            }

            // Reset KV cache so the regular flow starts from a clean prefill state.
            for kv in kv_caches.iter_mut() {
                kv.current_pos = 0;
            }

            Some(per_layer_concat)
        }
    } else {
        None
    };

    // ── Build ImportanceTable (+ optional DP-LLM ε variants) + reset KV cache ────
    let direct_attn_primary = matches!(
        importance_formula,
        crate::qcf_types::ImportanceFormula::DirectAttn
    );
    let cache_raw = importance_three_way || direct_attn_primary;
    let mut direct_attn_f5_decode_only: Option<Vec<f32>> = None;
    let mut direct_attn_f5_prefill_decode: Option<Vec<f32>> = None;
    let (
        imp_table,
        dpllm_epsilon,
        dpllm_epsilon_multi,
        dpllm_epsilon_abs,
        dpllm_epsilon_qcf,
        direct_attn_f4,
        direct_attn_f5,
    ) = if cache_raw {
        let (table, x_means, raws) = collector.build_with_raws();
        let sec_opt = model.secondary_mmap.as_ref();
        let eps_single = if importance_three_way {
            sec_opt.map(|sec| {
                crate::weight::noise_table::compute_input_aware_epsilon(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        let eps_multi = if importance_three_way {
            sec_opt.map(|sec| {
                crate::weight::noise_table::compute_input_aware_epsilon_multitensor(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        let eps_abs = if importance_three_way {
            sec_opt.map(|sec| {
                crate::weight::noise_table::compute_input_aware_epsilon_absolute(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        let eps_qcf = if importance_three_way {
            sec_opt.map(|sec| {
                crate::weight::noise_table::compute_input_aware_epsilon_qcf(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        // Cascade attention F4 + F5 (compute when raws are available, regardless
        // of whether primary is DirectAttn or 3-way compare).
        let (f4, f5) = if let Some(sec) = sec_opt {
            let n_heads = model.config.num_attention_heads;
            let n_kv_heads = model.config.num_key_value_heads;
            let d_head = model.config.head_dim;
            let pairs = crate::weight::noise_table::compute_cascade_attn_perturbation(
                &model.layers,
                sec,
                &raws,
                n_heads,
                n_kv_heads,
                d_head,
            );
            let f4_vec: Vec<f32> = pairs.iter().map(|(a, _)| *a).collect();
            let f5_vec: Vec<f32> = pairs.iter().map(|(_, b)| *b).collect();
            (Some(f4_vec), Some(f5_vec))
        } else {
            (None, None)
        };
        // §4.2 decode-X F5: compute with decode-only raws (T = N) AND
        // prefill+decode concat raws (T = 256 + N).
        if let (Some(sec), Some(raws_dec)) = (sec_opt, raws_decode_opt.as_ref()) {
            let n_heads = model.config.num_attention_heads;
            let n_kv_heads = model.config.num_key_value_heads;
            let d_head = model.config.head_dim;

            // (1) decode-only
            let pairs_dec = crate::weight::noise_table::compute_cascade_attn_perturbation(
                &model.layers,
                sec,
                raws_dec,
                n_heads,
                n_kv_heads,
                d_head,
            );
            direct_attn_f5_decode_only =
                Some(pairs_dec.iter().map(|(_, b)| *b).collect::<Vec<f32>>());

            // (2) prefill + decode concat: raws[i] = concat(prefill_raws[i], decode_raws[i])
            // along the T dimension.
            if raws.len() == raws_dec.len() {
                let mut raws_merged: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(raws.len());
                for (p, d_entry) in raws.iter().zip(raws_dec.iter()) {
                    let (p_data, p_t, p_d) = p;
                    let (d_data, d_t, d_d) = d_entry;
                    if *p_d != *d_d {
                        eprintln!(
                            "[QCF-dump]{} decode-x merge: WARN dim mismatch prefill={} decode={}",
                            log_prefix, p_d, d_d
                        );
                        continue;
                    }
                    let dim = *p_d;
                    let mut merged: Vec<f32> = Vec::with_capacity((*p_t + *d_t) * dim);
                    merged.extend_from_slice(p_data);
                    merged.extend_from_slice(d_data);
                    raws_merged.push((merged, *p_t + *d_t, dim));
                }
                let pairs_pd = crate::weight::noise_table::compute_cascade_attn_perturbation(
                    &model.layers,
                    sec,
                    &raws_merged,
                    n_heads,
                    n_kv_heads,
                    d_head,
                );
                direct_attn_f5_prefill_decode =
                    Some(pairs_pd.iter().map(|(_, b)| *b).collect::<Vec<f32>>());
            }
        }
        (table, eps_single, eps_multi, eps_abs, eps_qcf, f4, f5)
    } else {
        (collector.build(), None, None, None, None, None, None)
    };
    eprintln!(
        "[QCF-dump]{} ImportanceTable built: {} entries (dpllm_epsilon={}, multi={}, abs={}, qcf={})",
        log_prefix,
        imp_table.len(),
        if dpllm_epsilon.is_some() {
            "computed"
        } else {
            "skipped"
        },
        if dpllm_epsilon_multi.is_some() {
            "computed"
        } else {
            "skipped"
        },
        if dpllm_epsilon_abs.is_some() {
            "computed"
        } else {
            "skipped"
        },
        if dpllm_epsilon_qcf.is_some() {
            "computed"
        } else {
            "skipped"
        },
    );
    for kv in kv_caches.iter_mut() {
        kv.current_pos = 0;
    }

    // ── Optional swap with importance-guided decider ──────────────────────────
    let decision = if let Some(ratio) = force_ratio {
        let ratio = ratio.clamp(0.0, 1.0);
        eprintln!(
            "[QCF-dump]{} swap algorithm: {} (execute_swap={})",
            log_prefix,
            swap_algorithm.short_name(),
            execute_swap,
        );
        // MW-C: ImportanceTable → flat per-layer 투영. noise 는 is_computed()
        // 일 때만 Some(slice) (구 fallback 게이트 보존). ratio→budget 환산도
        // 호출자로 이동 (currently_swapped=∅ 이므로 차감 0).
        let n_layers = model.layers.len();
        let importance_flat = crate::weight::decider::flatten_importance(&imp_table, n_layers);
        let noise_flat = if model.quant_noise.is_computed() {
            Some(model.quant_noise.as_slice())
        } else {
            None
        };
        let budget = (ratio * n_layers as f32).floor() as usize;
        let decider = WeightSwapDecider {
            importance: Some(&importance_flat),
            noise: noise_flat,
            n_decoder_layers: n_layers,
            currently_swapped: &[],
            allow_boundary_layers: read_allow_boundary_env(),
            algorithm: swap_algorithm,
        };
        let decider_decision = decider.decide(budget);

        // §4 ground-truth path: when `--swap-only-layers` is set, override the
        // decider's selection with the explicit list. The decider's
        // `qcf_swap_estimate` is recomputed against this override so the dump
        // JSON reports the QCF prediction for the actually-swapped set.
        let decision = if let Some(only) = swap_only_layers {
            let override_layers: Vec<usize> = only
                .iter()
                .copied()
                .filter(|i| *i < model.layers.len())
                .collect();
            let qcf_override = crate::weight::compute_qcf_weight_swap(
                &override_layers,
                model.quant_noise.as_slice(),
                Some(&importance_flat),
                model.layers.len(),
            );
            eprintln!(
                "[QCF-dump]{} swap-only override: layers={:?} (ignoring algorithm/ratio decision)",
                log_prefix, override_layers,
            );
            crate::weight::SwapDecision {
                selected_layers: override_layers,
                qcf_swap_estimate: qcf_override,
                fallback_used: false,
            }
        } else {
            decider_decision
        };

        // Trajectory mode (`--qcf-trajectory`): return the decision without
        // executing the swap — the caller drives swap one layer at a time
        // around per-step eval-ll measurements.
        if !execute_swap {
            return Ok(QcfWarmupResult {
                importance: imp_table,
                decision: Some(decision),
                dpllm_epsilon,
                dpllm_epsilon_multi,
                dpllm_epsilon_abs,
                dpllm_epsilon_qcf,
                direct_attn_f4,
                direct_attn_f5,
                direct_attn_f5_decode_only,
                direct_attn_f5_prefill_decode,
            });
        }

        if decision.selected_layers.is_empty() {
            eprintln!(
                "[QCF-dump]{} swap: ratio={:.2} → 0 layers selected (qcf=0.0)",
                log_prefix, ratio,
            );
        } else {
            let report = run_layer_swap(
                model,
                &decision.selected_layers,
                gpu_backend,
                cpu_backend,
                None,
                // LISWAP-3: QCF dump path does not exercise the pool yet —
                // Stage 3 prototype only wires --force-swap-ratio paths.
                #[cfg(feature = "opencl")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
            )
            .map_err(|e| anyhow::anyhow!("[QCF-dump]{} swap failed: {}", log_prefix, e))?;
            eprintln!(
                "[QCF-dump]{} swap: ratio={:.2}, layers={}/{}, qcf_pred={:.4}, \
                 fallback={}, latency={:.1}ms",
                log_prefix,
                ratio,
                report.swapped.len(),
                model.layers.len(),
                decision.qcf_swap_estimate,
                decision.fallback_used,
                report.latency_ms,
            );
        }
        Some(decision)
    } else {
        None
    };

    Ok(QcfWarmupResult {
        importance: imp_table,
        decision,
        dpllm_epsilon,
        dpllm_epsilon_multi,
        dpllm_epsilon_abs,
        dpllm_epsilon_qcf,
        direct_attn_f4,
        direct_attn_f5,
        direct_attn_f5_decode_only,
        direct_attn_f5_prefill_decode,
    })
}

// ─── Phase 4-C-1: PPL/standard generate 분기와 공유하는 swap dispatch + ─────
//  weight dump helpers. 본문 변경 없음.

pub fn dispatch_swap_weights(
    model: &crate::models::transformer::TransformerModel,
    ratio: f32,
    target_dtype: argus_shared::DtypeTag,
    importance_table: Option<&dyn crate::qcf_collector::ImportanceLookup>,
    decode_token_index: usize,
    swap_plan_out: &mut Option<crate::weight::IncrementalSwapPlan>,
    manager_report_out: &mut Option<(f32, usize, std::time::Instant, f32)>,
) {
    use crate::weight::{
        IncrementalSwapPlan, SwapDecision, WeightSwapDecider, compute_qcf_weight_swap,
    };
    use argus_shared::DtypeTag;

    // ── 1. Validation ──────────────────────────────────────────────────────
    if model.secondary_mmap.is_none() {
        eprintln!("[WeightSwap] Rejected: no_secondary (ENG-DAT-C09)");
        return;
    }
    if ratio <= 0.0 || ratio > 1.0 {
        eprintln!("[WeightSwap] Rejected: invalid_ratio ({:.4})", ratio);
        return;
    }
    if target_dtype != DtypeTag::Q4_0 {
        eprintln!(
            "[WeightSwap] Rejected: unsupported_dtype ({:?}) (INV-126)",
            target_dtype
        );
        return;
    }

    // ── 1b. In-flight plan check ───────────────────────────────────────────
    // Reject if a plan is already in flight (CLI or manager). Prevents
    // concurrent plan conflict (spec: manager signal accept only when no plan).
    if swap_plan_out.is_some() {
        eprintln!(
            "[WeightSwap] Rejected: incremental plan already in-flight (ratio={:.2}). \
             Wait for current plan to complete before sending a new SwapWeights signal.",
            ratio
        );
        return;
    }

    // ── 2. Collect currently-swapped layers ────────────────────────────────
    let n_layers = model.layers.len();
    let currently_swapped: Vec<usize> = (0..n_layers)
        .filter(|&i| model.layers[i].current_dtype() == DType::Q4_0)
        .collect();

    // ── 3. Decider ─────────────────────────────────────────────────────────
    let allow_boundary = read_allow_boundary_env();
    eprintln!(
        "[Decider] allow_boundary_layers={} (ratio={:.4})",
        allow_boundary, ratio
    );
    // MW-C: ImportanceLookup → flat per-layer 투영. noise 는 is_computed()
    // 일 때만 Some(slice) (구 fallback 게이트 보존). ratio→budget 환산도
    // 호출자로 이동.
    let importance_flat =
        importance_table.map(|imp| crate::weight::decider::flatten_importance(imp, n_layers));
    let noise_flat = if model.quant_noise.is_computed() {
        Some(model.quant_noise.as_slice())
    } else {
        None
    };
    let target_count = (ratio * n_layers as f32).floor() as usize;
    let budget = target_count.saturating_sub(currently_swapped.len());
    let decider = WeightSwapDecider {
        importance: importance_flat.as_deref(),
        noise: noise_flat,
        n_decoder_layers: n_layers,
        currently_swapped: &currently_swapped,
        allow_boundary_layers: allow_boundary,
        algorithm: crate::weight::SwapAlgorithm::ImportanceAware,
    };
    let decision: SwapDecision = decider.decide(budget);

    if decision.selected_layers.is_empty() {
        eprintln!(
            "[WeightSwap] No layers to swap (ratio={:.2}, already_swapped={})",
            ratio,
            currently_swapped.len()
        );
        // Empty swap is Ok per spec (already fully swapped); no plan committed.
        return;
    }

    // ── 4. Compute QCF estimate for the planned layers ─────────────────────
    let qcf_swap_estimated = compute_qcf_weight_swap(
        &decision.selected_layers,
        model.quant_noise.as_slice(),
        importance_flat.as_deref(),
        n_layers,
    );

    // ── 5. Commit incremental plan (K=2, same as CLI --swap-incremental-per-tick 2) ──
    let n_planned = decision.selected_layers.len();
    let per_tick = 2usize; // LISWAP-6: K=2 hard upper cap for manager path
    let ticks_est = n_planned.div_ceil(per_tick);
    eprintln!(
        "[WeightSwap] manager path: ratio={:.2}, {} target layers, per_tick={} ({} ticks estimated), qcf_estimated={:.4}",
        ratio, n_planned, per_tick, ticks_est, qcf_swap_estimated,
    );

    *swap_plan_out = Some(IncrementalSwapPlan::new(
        decision.selected_layers,
        per_tick,
        decode_token_index,
    ));
    *manager_report_out = Some((
        ratio,
        n_planned,
        std::time::Instant::now(),
        qcf_swap_estimated,
    ));
}

/// Groups all inputs needed by [`compute_qcf_estimates`] so that the caller
/// constructs a single struct instead of passing many individual arguments.
///
/// v1 anchor: `generate.rs`(`d5ed71d2^`) L3676-3695 `QcfEstimateContext`. Adapted:
/// `kv_caches: &[KVCache]` → `kv_handles: &[Arc<StandardFormat>]`.
pub struct QcfEstimateContext<'a> {
    /// Standard format KV handles (one per layer). Empty → no KV-based estimates.
    pub kv_handles: &'a [Arc<StandardFormat>],
    /// KIVI handles for `kv.quant_dynamic` estimate. Empty → quant estimate skipped.
    pub quant_window_handles: &'a [Arc<QuantWindowFormat>],
    /// Attention score lookup for H2O / D2O / streaming actions. `None` → uniform fallback.
    pub importance: Option<&'a dyn ImportanceLookup>,
    /// (sink_size, window_size) for StreamingLLM dry-run. None = skip.
    pub streaming_config: Option<(usize, usize)>,
    /// Pre-built importance table for LayerSkip dry-run. None = skip.
    pub importance_table: Option<&'a crate::qcf::ImportanceTable>,
    /// Total number of transformer layers (needed for LayerSkip).
    pub num_layers: usize,
    /// §5.9.1 Track A: per-token attention importance scores from `AttentionScoreAccumulator`.
    /// `Some` → `kv.evict_h2o` / `kv.merge_d2o` QCF 산출 언블록 (requires_scores=true 통과).
    /// `None` → 기존 uniform fallback 유지 (h2o/d2o 키 absent).
    /// **QCF_kv 전용** — `importance`(QCF_weight, weight 가족 layer importance)와 절대 합치지 말 것
    /// (QCF_kv ⊥ QCF_weight, CLAUDE.md QCF 명명 컨벤션).
    pub token_scores: Option<&'a [f32]>,
}

/// Compute dry-run QCF estimates for all 6 lossy actions (ENG-ALG-050).
/// Read-only: does not modify KV caches or KIVI caches.
///
/// v1 anchor: `generate.rs`(`d5ed71d2^`) L3696-3850 `compute_qcf_estimates`. Byte-level
/// lift-and-shift: only adaptation = `&[KVCache]` → `&[Arc<StandardFormat>]` handle access
/// via `with_cache_mut`.
///
/// Returns estimates map (keys: `kv.evict_sliding` / `kv.evict_h2o` / `kv.merge_d2o` /
/// `kv.evict_streaming` / `kv.quant_dynamic` / `weight.skip`).
pub fn compute_qcf_estimates(
    ctx: &QcfEstimateContext<'_>,
) -> std::collections::HashMap<String, f32> {
    use crate::qcf::{AggregationMode, QcfActionType, QcfKvParams, VDataSource, compute_qcf_kv};
    use std::collections::HashMap;
    let mut estimates = HashMap::new();

    // ── 1-4. StandardFormat handle-based eviction/merge QCF via unified formula ──
    //
    // ISSUE-6 guard: OpenCL zero-copy KV (UnifiedBuffer / CL_MEM_ALLOC_HOST_PTR)는
    // unmapped 상태에서 `as_ptr()=null`을 반환한다. host pointer가 null인 경우
    // backend read-back fallback으로 layer-0 v/k buffer를 Vec<u8>으로 읽어온 뒤
    // `VDataSource::from_buffer(&buf, Some(&bytes))` 경로를 사용한다.
    // read_buffer 실패 시에만 eprintln 경고(graceful skip).
    if !ctx.kv_handles.is_empty() {
        // Read current_pos / v_buffer / k_buffer from the first handle (layer-0 geometry).
        let current_pos = ctx.kv_handles[0].current_pos();

        // GPU 백엔드는 as_ptr() 이 non-null(zero-copy DMA-BUF / rpcmem 매핑) 이어도 GPU 가
        // attention 중 V 를 쓴 뒤 CPU 캐시가 invalidate 되지 않아 stale(0) 을 읽는다 (ARM UMA
        // 캐시 비일관성). as_ptr 직접 접근은 cache-coherent 를 보장하지 못하므로 GPU 면 항상
        // read_buffer(D2H) 로 강제한다 — 그래야 o_before=Σα·V 가 실데이터가 되어 QCF≠0.
        // CPU 백엔드만 as_ptr 직접 접근(fast path)이 coherent 하다.
        let v_host_readable = ctx.kv_handles[0]
            .with_cache_mut(|c| !c.v_buffer.backend().is_gpu() && !c.v_buffer.as_ptr().is_null());

        if current_pos > 0 {
            // Try to obtain host-readable byte slices for layer-0 v/k buffers.
            // Fast path: CPU backend, as_ptr() valid and cache-coherent.
            // Slow path: GPU backend (cache-incoherent as_ptr) 또는 device-only buffer →
            // read_buffer fallback (1 D2H per RequestQcf cold call).
            let (v_bytes_opt, k_bytes_opt): (Option<Vec<u8>>, Option<Vec<u8>>) = if v_host_readable
            {
                (None, None) // fast path: VDataSource::from_buffer(_, None) will use as_ptr
            } else {
                // Fallback: read layer-0 v_buffer (and k_buffer) from device.
                let (v_res, k_res) = ctx.kv_handles[0].with_cache_mut(|cache| {
                    let backend = cache.v_buffer.backend().clone();
                    let v_size = cache.v_buffer.buffer().size();
                    let k_size = cache.k_buffer.buffer().size();
                    let mut v_bytes = vec![0u8; v_size];
                    let mut k_bytes = vec![0u8; k_size];
                    let v_ok = backend.read_buffer(&cache.v_buffer, &mut v_bytes);
                    let k_ok = backend.read_buffer(&cache.k_buffer, &mut k_bytes);
                    (v_ok.map(|_| v_bytes), k_ok.map(|_| k_bytes))
                });
                match v_res {
                    Ok(v_bytes) => (Some(v_bytes), k_res.ok()),
                    Err(e) => {
                        eprintln!(
                            "[QCF] KV-based estimates skipped: v_buffer read_buffer failed: {e}"
                        );
                        (None, None)
                    }
                }
            };

            let can_compute = v_host_readable || v_bytes_opt.is_some();

            if can_compute {
                let keep_ratio = 0.5f32;
                let target_len = (current_pos as f32 * keep_ratio) as usize;
                let protected_prefix = 4usize;

                // §5.9.1 Track A: ctx.token_scores 가 Some 이면 acc.importance_scores() 사용
                // (kv.evict_h2o / kv.merge_d2o requires_scores=true 통과).
                // None 이면 uniform fallback (v1 fallback_scores 등가 — h2o/d2o 키 absent).
                let scores_opt: Option<Vec<f32>> = ctx
                    .token_scores
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_vec());
                let fallback_scores = vec![1.0 / current_pos.max(1) as f32; current_pos];
                let attention_scores_owned: Vec<f32> =
                    scores_opt.clone().unwrap_or(fallback_scores);

                if target_len < current_pos {
                    let mut actions: Vec<(&'static str, QcfActionType, bool)> = vec![
                        (
                            "kv.evict_sliding",
                            QcfActionType::EvictSliding { target_len },
                            false,
                        ),
                        (
                            "kv.evict_h2o",
                            QcfActionType::EvictH2o {
                                target_len,
                                keep_ratio: 0.5,
                                protected_prefix,
                            },
                            true,
                        ),
                        (
                            "kv.merge_d2o",
                            QcfActionType::MergeD2o {
                                target_len,
                                keep_ratio: 0.5,
                                protected_prefix,
                            },
                            true,
                        ),
                    ];
                    if let Some((sink_size, window_size)) = ctx.streaming_config {
                        actions.push((
                            "kv.evict_streaming",
                            QcfActionType::EvictStreaming {
                                sink_size,
                                window_size,
                            },
                            false,
                        ));
                    }

                    // Run each action inside with_cache_mut to borrow v_buffer/k_buffer safely.
                    // When v_bytes_opt is Some (device-only fallback), pass the byte slices to
                    // VDataSource::from_buffer so the QCF computation uses host copies.
                    for (id, action, requires_scores) in actions {
                        if requires_scores && scores_opt.is_none() {
                            continue;
                        }
                        let v_bytes_ref: Option<&[u8]> = v_bytes_opt.as_deref();
                        let k_bytes_ref: Option<&[u8]> = k_bytes_opt.as_deref();
                        let qcf_opt = ctx.kv_handles[0].with_cache_mut(|cache| {
                            let v_source = VDataSource::from_buffer(&cache.v_buffer, v_bytes_ref)?;
                            let k_source = if matches!(action, QcfActionType::MergeD2o { .. }) {
                                VDataSource::from_buffer(&cache.k_buffer, k_bytes_ref)
                            } else {
                                None
                            };
                            let params = QcfKvParams {
                                action: action.clone(),
                                v_source,
                                k_source,
                                attention_scores: &attention_scores_owned,
                                head_attn: None,
                                n_kv_heads: cache.kv_heads(),
                                head_dim: cache.head_dim(),
                                current_pos,
                                capacity: cache.capacity(),
                                layout: cache.layout(),
                                aggregation: AggregationMode::Mean,
                                beta: 1.0,
                            };
                            let (qcf, _) = compute_qcf_kv(&params);
                            Some(qcf)
                        });
                        if let Some(qcf) = qcf_opt {
                            estimates.insert(id.to_string(), qcf);
                        }
                    }
                }
            }
        }
    }

    // ── 5. KIVI dynamic quantization QCF ──
    if !ctx.quant_window_handles.is_empty() {
        let mut total_qcf = 0.0f32;
        let mut count = 0u32;
        for handle in ctx.quant_window_handles {
            let qcf = handle.with_cache_mut(|c| c.estimate_dryrun_qcf());
            if qcf > 0.0 {
                total_qcf += qcf;
                count += 1;
            }
        }
        if count > 0 {
            let avg_qcf = total_qcf / count as f32;
            estimates.insert("kv.quant_dynamic".to_string(), avg_qcf.min(1.0));
        }
    }

    // ── 6. LayerSkip QCF: importance-table based skip cost estimate ──
    if let Some(table) = ctx.importance_table {
        let total_sublayers = ctx.num_layers * 2;
        let skip_count = total_sublayers / 4;
        if skip_count > 0 {
            let (qcf_skip, _skip_set) = table.estimate_qcf_for_count(skip_count, ctx.num_layers);
            estimates.insert("weight.skip".to_string(), qcf_skip);
        }
    }

    estimates
}

/// LISWAP-PPL diagnostic: dump every layer's weight tensors (wq/wk/wv/wo/
/// w_gate/w_up/w_down) to raw bin files under `out_dir`. File naming:
/// `layer{NN}_{tensor}_{dtype}.bin` (e.g. `layer00_wq_Q4_0.bin`). Each file
/// holds the raw GPU buffer bytes for that tensor at the moment of the call.
///
/// Two such dumps (one from a Q4-native model load, one from an F16 model
/// after swap completion) can be byte-compared on the host to determine
/// whether the swap path produces bit-identical Q4 weights.
pub fn dump_layer_weights_to_dir(
    model: &crate::models::transformer::TransformerModel,
    backend: &Arc<dyn Backend>,
    out_dir: &std::path::Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let n = model.layers.len();
    eprintln!(
        "[Q4-DUMP] dumping {} layer weights to {}",
        n,
        out_dir.display()
    );
    for (i, slot) in model.layers.iter().enumerate() {
        let weights = slot.load_weights();
        let dtype = slot.current_dtype();
        let tensors: [(&str, &crate::tensor::Tensor); 7] = [
            ("wq", &weights.wq),
            ("wk", &weights.wk),
            ("wv", &weights.wv),
            ("wo", &weights.wo),
            ("w_gate", &weights.w_gate),
            ("w_up", &weights.w_up),
            ("w_down", &weights.w_down),
        ];
        for (name, t) in tensors {
            let nbytes = t.buffer().size();
            if nbytes == 0 {
                eprintln!(
                    "[Q4-DUMP] layer{:02} {} dtype={:?} SKIP (size=0)",
                    i, name, dtype
                );
                continue;
            }
            let mut bytes = vec![0u8; nbytes];
            // For OpenCL/CUDA tensors `buffer().as_ptr()` is the cl_mem/cu_ptr
            // handle and may look like a host nullptr — backend.read_buffer
            // does the device→host copy via the backend-specific path, so we
            // rely on its return value rather than pre-checking as_ptr.
            match backend.read_buffer(t, &mut bytes) {
                Ok(()) => {
                    let fname = format!("layer{:02}_{}_{:?}.bin", i, name, dtype);
                    let path = out_dir.join(&fname);
                    std::fs::write(&path, &bytes)?;
                    eprintln!(
                        "[Q4-DUMP] layer{:02} {:8} dtype={:>5?} bytes={:8} → {}",
                        i, name, dtype, nbytes, fname
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[Q4-DUMP] layer{:02} {} dtype={:?} SKIP (read_buffer failed: {})",
                        i, name, dtype, e
                    );
                }
            }
        }
    }
    // Also dump model-level tensors that are NOT inside per-layer slots and
    // therefore are NOT touched by weight swap: embed_tokens, final norm, and
    // lm_head. These three are the most likely sources of E ≠ D NLL drift
    // because (a) the F16 model's lm_head is typically tied to embed_tokens
    // and (b) any missing lm_head is derived via F16→Q4_0 quantization at
    // load time, whose result may not match a standalone Q4_0 GGUF's lm_head
    // byte-for-byte.
    let model_tensors: [(&str, &crate::tensor::Tensor); 3] = [
        ("embed_tokens", &model.embed_tokens),
        ("norm", &model.norm),
        ("lm_head", &model.lm_head),
    ];
    for (name, t) in model_tensors {
        let nbytes = t.buffer().size();
        if nbytes == 0 {
            eprintln!("[Q4-DUMP] model.{} SKIP (size=0)", name);
            continue;
        }
        let dt = t.dtype();
        let mut bytes = vec![0u8; nbytes];
        match backend.read_buffer(t, &mut bytes) {
            Ok(()) => {
                let fname = format!("model_{}_{:?}.bin", name, dt);
                let path = out_dir.join(&fname);
                std::fs::write(&path, &bytes)?;
                eprintln!(
                    "[Q4-DUMP] model.{:14} dtype={:>5?} bytes={:8} → {}",
                    name, dt, nbytes, fname
                );
            }
            Err(e) => {
                // The lm_head can live on a CPU backend even when the main
                // backend is GPU (`lm_head_on_cpu`) — fall back to CpuBackend
                // for that case so we still get a dump file out.
                let cpu_be: Arc<dyn Backend> = crate::backend::cpu::cpu_singleton();
                match cpu_be.read_buffer(t, &mut bytes) {
                    Ok(()) => {
                        let fname = format!("model_{}_{:?}.bin", name, dt);
                        let path = out_dir.join(&fname);
                        std::fs::write(&path, &bytes)?;
                        eprintln!(
                            "[Q4-DUMP] model.{:14} dtype={:>5?} bytes={:8} → {} (via CPU fallback)",
                            name, dt, nbytes, fname
                        );
                    }
                    Err(e2) => {
                        eprintln!(
                            "[Q4-DUMP] model.{} SKIP (read_buffer failed: gpu={}, cpu={})",
                            name, e, e2
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "[Q4-DUMP] complete: {} layers + 3 model tensors dumped to {}",
        n,
        out_dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::kv_cache::KVCache;
    use crate::kv::standard_format::StandardFormat;
    use crate::kv_cache_ops::KVLayout;
    use crate::memory::host::shared::SharedBuffer;
    use crate::qcf::{AggregationMode, QcfActionType, QcfKvParams, VDataSource, compute_qcf_kv};
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    use super::{QcfEstimateContext, compute_qcf_estimates};

    /// HeadMajor F32 KVCache (CpuBackend, as_ptr 유효).
    fn make_standard_format(
        capacity: usize,
        kv_heads: usize,
        head_dim: usize,
        n_tokens: usize,
    ) -> Arc<StandardFormat> {
        let total = kv_heads * capacity * head_dim;
        let mk_buf = |data: Vec<f32>| {
            let buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
            let mut t = Tensor::new(
                Shape::new(vec![1, kv_heads, capacity, head_dim]),
                buf,
                Arc::new(CpuBackend::new()),
            );
            t.as_mut_slice::<f32>().copy_from_slice(&data);
            t
        };
        let v_data: Vec<f32> = (0..total).map(|i| i as f32 * 0.01 + 0.1).collect();
        let k_data: Vec<f32> = (0..total).map(|i| i as f32 * 0.02 + 0.2).collect();
        let k = mk_buf(k_data);
        let v = mk_buf(v_data);
        let mut cache = KVCache::new(k, v, capacity);
        // SeqMajor/HeadMajor: SharedBuffer로 만든 캐시는 SeqMajor 기본값.
        // current_pos만 n_tokens로 설정한다.
        for _ in 0..n_tokens {
            cache.advance_pos(1);
        }
        Arc::new(StandardFormat::new(0, cache))
    }

    /// AB-5 fallback 검증: `VDataSource::from_buffer(&tensor, Some(&bytes))` 경로가
    /// `from_buffer(&tensor, None)` (as_ptr 직접 접근)와 동일한 QCF를 반환하는지 확인.
    ///
    /// 이것이 device-only 버퍼의 read-back fallback 경로와 동등하다
    /// (device bytes를 Vec<u8>으로 읽어온 후 cpu_bytes=Some 경로로 주입).
    #[test]
    fn test_vdata_source_cpu_bytes_matches_direct() {
        let kv_heads = 2usize;
        let head_dim = 4usize;
        let capacity = 16usize;
        let current_pos = 8usize;
        let target_len = 4usize;

        let v_data: Vec<f32> = (0..kv_heads * capacity * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let scores: Vec<f32> = vec![1.0 / current_pos as f32; current_pos];

        // Path A: from_buffer(_, None) — direct as_ptr
        let buf_a = Arc::new(SharedBuffer::new(v_data.len() * 4, DType::F32));
        let mut t_a = Tensor::new(
            Shape::new(vec![1, kv_heads, capacity, head_dim]),
            buf_a,
            Arc::new(CpuBackend::new()),
        );
        t_a.as_mut_slice::<f32>().copy_from_slice(&v_data);
        let src_direct = VDataSource::from_buffer(&t_a, None)
            .expect("direct path should succeed for CpuBackend");

        // Path B: from_buffer(_, Some(&bytes)) — cpu_bytes injected (simulates read-back)
        let v_bytes: Vec<u8> = v_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let src_fallback =
            VDataSource::from_buffer(&t_a, Some(&v_bytes)).expect("cpu_bytes path should succeed");

        let (qcf_direct, _) = compute_qcf_kv(&QcfKvParams {
            action: QcfActionType::EvictSliding { target_len },
            v_source: src_direct,
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads: kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        });
        let (qcf_fallback, _) = compute_qcf_kv(&QcfKvParams {
            action: QcfActionType::EvictSliding { target_len },
            v_source: src_fallback,
            k_source: None,
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads: kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
            beta: 1.0,
        });

        assert!(
            (qcf_direct - qcf_fallback).abs() < 1e-6,
            "cpu_bytes fallback QCF ({qcf_fallback}) should match direct QCF ({qcf_direct})"
        );
    }

    /// `compute_qcf_estimates`가 CPU-backed StandardFormat(as_ptr 유효)에서
    /// KV 기반 estimates를 정상 반환하는지 확인.
    #[test]
    fn test_compute_qcf_estimates_cpu_kv_returns_sliding() {
        let handle = make_standard_format(32, 2, 4, 16);
        let ctx = QcfEstimateContext {
            kv_handles: &[handle],
            quant_window_handles: &[],
            importance: None,
            streaming_config: None,
            importance_table: None,
            num_layers: 4,
            token_scores: None,
        };
        let estimates = compute_qcf_estimates(&ctx);
        // current_pos=16, keep_ratio=0.5 → target_len=8 < 16 → sliding should appear.
        // h2o/d2o require scores (scores_opt=None) → only sliding expected.
        assert!(
            estimates.contains_key("kv.evict_sliding"),
            "expected kv.evict_sliding in estimates, got keys: {:?}",
            estimates.keys().collect::<Vec<_>>()
        );
        let qcf = estimates["kv.evict_sliding"];
        assert!(qcf >= 0.0, "sliding QCF should be non-negative, got {qcf}");
        // QCF = 0 when V is all-zero; QCF ≤ 1 always. Accept both 0 and >0.
        assert!(qcf <= 1.0, "sliding QCF should be <= 1.0, got {qcf}");
    }

    /// §5.9.1 Track A: `token_scores` Some → `kv.evict_h2o` / `kv.merge_d2o` 키가 포함되어야 함.
    /// `token_scores` None → h2o/d2o 는 requires_scores=true 로 skip.
    #[test]
    fn test_compute_qcf_estimates_token_scores_unlocks_h2o_d2o() {
        let handle = make_standard_format(32, 2, 4, 16);
        let n_tokens = 16usize;
        // 16-token uniform score
        let scores: Vec<f32> = vec![1.0 / n_tokens as f32; n_tokens];

        let ctx_with_scores = QcfEstimateContext {
            kv_handles: std::slice::from_ref(&handle),
            quant_window_handles: &[],
            importance: None,
            streaming_config: None,
            importance_table: None,
            num_layers: 4,
            token_scores: Some(&scores),
        };
        let estimates_with = compute_qcf_estimates(&ctx_with_scores);
        assert!(
            estimates_with.contains_key("kv.evict_h2o"),
            "token_scores Some → kv.evict_h2o expected, got keys: {:?}",
            estimates_with.keys().collect::<Vec<_>>()
        );
        assert!(
            estimates_with.contains_key("kv.merge_d2o"),
            "token_scores Some → kv.merge_d2o expected"
        );

        let ctx_no_scores = QcfEstimateContext {
            kv_handles: &[handle],
            quant_window_handles: &[],
            importance: None,
            streaming_config: None,
            importance_table: None,
            num_layers: 4,
            token_scores: None,
        };
        let estimates_without = compute_qcf_estimates(&ctx_no_scores);
        assert!(
            !estimates_without.contains_key("kv.evict_h2o"),
            "token_scores None → kv.evict_h2o absent, got keys: {:?}",
            estimates_without.keys().collect::<Vec<_>>()
        );
        assert!(
            !estimates_without.contains_key("kv.merge_d2o"),
            "token_scores None → kv.merge_d2o absent"
        );
    }

    /// `compute_qcf_estimates`가 current_pos=0인 경우 estimates를 반환하지 않는지 확인.
    #[test]
    fn test_compute_qcf_estimates_empty_cache_returns_no_kv_estimates() {
        let handle = make_standard_format(32, 2, 4, 0);
        let ctx = QcfEstimateContext {
            kv_handles: &[handle],
            quant_window_handles: &[],
            importance: None,
            streaming_config: None,
            importance_table: None,
            num_layers: 4,
            token_scores: None,
        };
        let estimates = compute_qcf_estimates(&ctx);
        let kv_keys: Vec<_> = estimates.keys().filter(|k| k.starts_with("kv.")).collect();
        assert!(
            kv_keys.is_empty(),
            "empty cache should produce no kv estimates, got: {:?}",
            kv_keys
        );
    }
}
