//! eval/ppl/dump-importance 진입에 필요한 RunCtx 조립층 (Phase γ-3a).
//!
//! `argus-eval` bin 이 `--eval-ll` / `--ppl` / `--dump-importance` 진입 시,
//! 폐기된 `legacy/generate.rs::main()` 이 인라인으로 만들던 eval 전용 상태
//! (`cache_manager` / `score_accumulator` / `skip_config` /
//! `actual_protected_prefix` / KV capacity = max_seq_len 선할당)를 재현한다.
//!
//! ## bin_setup 과의 차이
//!
//! [`build_inference_ctx`](crate::session::bin_setup::build_inference_ctx) 는
//! happy-path 전용(`is_standard_happy_path` 전제)이라 [`StandardHappyCtx`] 만
//! 만든다. eval ctx 는 eviction/swap/score 상태를 요구하므로 별 조립층을 둔다.
//! 단 [`SessionInitCtx::build`] 를 **직접 호출**해 `caps`(KIVI 라우팅 필수) /
//! `swap_algorithm` / `importance_formula` 등을 보존한다(bin_setup destructure 가
//! drop 하는 값들).
//!
//! ## AUF 제약
//!
//! [`resolve_tokenizer_path`](crate::session::bin_setup::resolve_tokenizer_path)
//! 는 AUF 단일파일 분기가 없다(`AufTokenizer`→HF 브리지 부재). **AUF 모델은
//! `--tokenizer-path` 명시 필수** — 미지정 시 `model.auf/tokenizer.json` 을
//! dir 로 취급해 로드 실패한다.

use std::sync::Arc;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::hardware::DeviceTarget;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::inference::skip_config::SkipConfig;
use crate::kv::cache_manager::CacheManager;
use crate::kv::d2o_handler::{D2OConfig, D2OHandler};
use crate::kv::eviction::EvictMethod;
use crate::kv::eviction::EvictionPolicy;
use crate::kv::eviction::h2o::H2OPolicy;
use crate::kv::eviction::h2o_plus::H2OPlusPolicy;
use crate::kv::eviction::no_eviction::NoEvictionPolicy;
use crate::kv::eviction::sliding_window::SlidingWindowPolicy;
use crate::kv::eviction::streaming_llm::StreamingLLMPolicy;
use crate::kv::{CachePressurePipeline, PressureLevel, PressureStageConfig};
use crate::models::transformer::TransformerModel;
use crate::resilience::sys_monitor::{LinuxSystemMonitor, NoOpMonitor, SystemMonitor};
use crate::session::bin_setup::{
    alloc_standard_kv_caches, check_vocab_compatibility, resolve_tokenizer_path,
};
use crate::session::cli::Args;
use crate::session::dump_importance::DumpImportanceCtx;
use crate::session::eval::args::EvalLlRunCtx;
use crate::session::init::SessionInitCtx;
use crate::session::ppl::PplRunCtx;

/// `SessionInitCtx::build` 직후의 공통 결과 + eval 파생 상태.
///
/// model/backend/tokenizer 등 owned 자원을 한곳에 모아 mode 별 빌더가 소비한다.
/// `caps` 는 KIVI 경로 thread-through 용으로 보존된다.
struct EvalBase {
    backend: Arc<dyn crate::backend::Backend>,
    memory: Arc<dyn crate::memory::Memory>,
    cpu_backend_arc: Arc<dyn crate::backend::Backend>,
    gpu_backend_arc: Option<Arc<dyn crate::backend::Backend>>,
    caps: Arc<crate::capability::CapabilityRegistry>,
    model: TransformerModel,
    tokenizer: Tokenizer,
    prompt: String,
    kv_type: DType,
    max_seq_len: usize,
    swap_algorithm: crate::weight::SwapAlgorithm,
    importance_formula: crate::qcf_types::ImportanceFormula,
    importance_compare: bool,
    swap_only_layers: Option<Vec<usize>>,
}

/// 플러그인 등록 → 내장 KV format self-test → `SessionInitCtx::build` →
/// tokenizer resolve/load/검증 → prompt encode 까지 수행해 [`EvalBase`] 를 만든다.
///
/// bin_setup 의 보조 fn(`register_dynamic_plugins` /
/// `ensure_builtin_kv_formats_registered` / `resolve_tokenizer_path` /
/// `check_vocab_compatibility`)을 재사용한다. KV alloc 은 mode 별 빌더가
/// capacity=max_seq_len 으로 직접 호출한다.
fn build_eval_base(args: &Args) -> Result<EvalBase> {
    // bin_setup::build_inference_ctx 의 prologue 와 동일 — plugin 등록 + format self-test.
    crate::session::plugin_dispatch::register_dynamic_plugins(&args.load_plugin)?;
    crate::format::ensure_builtin_kv_formats_registered()?;

    let init = SessionInitCtx::build(args)?;

    // legacy generate.rs:114-126 미러 — hardware resolver 에서 cpu/gpu backend arc 파생.
    let cpu_backend_arc = init
        .hardware
        .resolve(DeviceTarget::Cpu)
        .expect("Cpu always resolves")
        .0
        .clone();
    let gpu_backend_arc: Option<Arc<dyn Backend>> = init
        .hardware
        .resolve(DeviceTarget::Gpu)
        .map(|(b, _)| b.clone());

    let tokenizer_path = resolve_tokenizer_path(args, &init.model_path, init.is_gguf);
    eprintln!("[Tokenizer] {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Cannot load tokenizer from {}: {}", tokenizer_path, e))?;
    check_vocab_compatibility(&tokenizer, &init.model, &tokenizer_path)?;

    let prompt = if let Some(path) = &args.prompt_file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt file {}: {}", path, e))?
    } else {
        args.prompt.clone()
    };
    eprintln!("Prompt: {}", prompt);

    let kv_type = match args.kv_type.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "q4" => DType::Q4_0,
        other => anyhow::bail!("Unsupported KV type: {other}. Use f32, f16, or q4."),
    };

    Ok(EvalBase {
        backend: init.backend,
        memory: init.memory,
        cpu_backend_arc,
        gpu_backend_arc,
        caps: init.caps,
        model: init.model,
        tokenizer,
        prompt,
        kv_type,
        max_seq_len: args.max_seq_len,
        swap_algorithm: init.swap_algorithm,
        importance_formula: init.importance_formula,
        importance_compare: init.importance_compare,
        swap_only_layers: init.swap_only_layers,
    })
}

/// score(importance) 기반 eviction 정책인가 — accumulator score 를 evict 에 흘려보내야 하는지 결정.
///
/// h2o/h2o_plus/d2o 가 기존 대상. R-KV(KV roadmap 항목 0 측정, feature `rkv`)는 fusion Z 가
/// importance I 를 항(λ·I)으로 포함하므로 score-based 로 분류 → `force_evict_with_scores` 경로로
/// accumulator importance 가 stage 에 흐른다. feature OFF 시 "rkv" 정책명 자체가 부재(불변).
pub fn is_score_based_eviction(args: &Args) -> bool {
    matches!(args.eviction_policy(), "h2o" | "h2o_plus" | "d2o")
        || (cfg!(feature = "rkv") && args.eviction_policy() == "rkv")
}

/// eval/ppl 의 protected_prefix 기본값을 legacy generate.rs:849-860 등가로 계산.
///
/// `--protected-prefix` 명시 시 그 값. 미지정 시 정책별 기본:
/// - h2o/h2o_plus/d2o/rkv → 4 (attention sinks only — 전체 prompt 보호 시 score-based
///   eviction 이 무의미해진다).
/// - streaming → `sink_size`.
/// - sliding/none → prompt 전체 보호(legacy 동작).
pub fn eval_protected_prefix(args: &Args, prompt_len: usize) -> usize {
    args.protected_prefix().unwrap_or_else(|| {
        if is_score_based_eviction(args) {
            4
        } else if args.eviction_policy() == "streaming" {
            args.sink_size()
        } else {
            prompt_len
        }
    })
}

/// legacy generate.rs:862-961 의 cache_manager 인라인 조립을 재현.
///
/// d2o 는 `CachePressurePipeline`(Pipeline 모드), 그 외는 `EvictionPolicy`(Legacy
/// 모드). Manager-directed eviction 용 H2o/Sliding policy 도 등록한다.
/// `build_resilience_cache_manager`(build_bench_loop.rs) 는 `Option` 반환 +
/// protected_prefix override 미지원이라 eval 에 부적합 — legacy 를 직접 재현한다.
fn build_eval_cache_manager(
    args: &Args,
    backend: &Arc<dyn Backend>,
    actual_protected_prefix: usize,
) -> Result<CacheManager> {
    let monitor: Box<dyn SystemMonitor> = if backend.is_discrete_gpu() {
        Box::new(NoOpMonitor)
    } else {
        Box::new(LinuxSystemMonitor)
    };
    let threshold_bytes = args.memory_threshold_mb() * 1024 * 1024;

    let mut cache_manager = if args.eviction_policy() == "d2o" {
        let d2o_handler = D2OHandler::new(D2OConfig {
            keep_ratio: args.d2o_keep_ratio(),
            protected_prefix: actual_protected_prefix,
            target_ratio: args.eviction_target_ratio(),
            ema_beta: args.d2o_ema_beta(),
            merge_e: args.d2o_merge_e(),
            use_layer_allocation: args.d2o_layer_alloc(),
            protected_layers: args.d2o_protected_layers().unwrap_or_default(),
            merge_axis: args.d2o_merge_axis(),
        });
        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(d2o_handler),
        }]);
        CacheManager::with_pipeline(pipeline, monitor, threshold_bytes)
    } else if cfg!(feature = "rkv") && args.eviction_policy() == "rkv" {
        // R-KV(KV roadmap 항목 0 측정, P2a): RkvStage(KVCacheStage)를 StageBackedPolicy 로
        // 감싸 EvictionPolicy 표면으로 노출 → CacheManager::new 등록(d2o if-branch 동형).
        // λ 는 CLI(--lambda)에서, α/τ 는 측정 상수. importance 는 score accumulator 가
        // force_evict_with_scores 로 흘려보낸다(아래 score_based_eviction 포함).
        #[cfg(feature = "rkv")]
        {
            use crate::kv::eviction::stage_registry::StageBackedPolicy;
            use crate::kv::rkv_stage::{RkvConfig, RkvStage};
            let stage = RkvStage::new(RkvConfig {
                lambda: args.rkv_lambda(),
                ..RkvConfig::default()
            });
            let policy: Box<dyn EvictionPolicy> = Box::new(StageBackedPolicy::new(Box::new(stage)));
            CacheManager::new(
                policy,
                monitor,
                threshold_bytes,
                args.eviction_target_ratio(),
            )
        }
        // feature OFF 에서는 위 cfg!() 가 false 라 도달 불가하나, 컴파일러 만족을 위해 분기를 닫는다.
        #[cfg(not(feature = "rkv"))]
        unreachable!("rkv branch gated by cfg!(feature = \"rkv\")")
    } else {
        let policy: Box<dyn EvictionPolicy> = match args.eviction_policy() {
            "none" => Box::new(NoEvictionPolicy::new()),
            "sliding" => Box::new(SlidingWindowPolicy::new(
                args.eviction_window(),
                actual_protected_prefix,
            )),
            "streaming" => {
                let window = if args.streaming_window() > 0 {
                    args.streaming_window()
                } else if args.kv_budget() > 0 {
                    args.kv_budget().saturating_sub(args.sink_size())
                } else {
                    args.eviction_window()
                };
                Box::new(StreamingLLMPolicy::new(args.sink_size(), window))
            }
            "h2o" => Box::new(H2OPolicy::new(
                args.h2o_keep_ratio(),
                actual_protected_prefix,
            )),
            "h2o_plus" => Box::new(H2OPlusPolicy::new(
                args.h2o_keep_ratio(),
                actual_protected_prefix,
            )),
            other => anyhow::bail!(
                "Unknown eviction policy: '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o",
                other
            ),
        };
        CacheManager::new(
            policy,
            monitor,
            threshold_bytes,
            args.eviction_target_ratio(),
        )
    };

    if let Some(dir) = args.swap_dir.clone() {
        eprintln!("[Resilience] KV swap enabled: dir={}", dir.display());
        cache_manager.enable_swap(dir);
    }

    // Manager-directed eviction 용 policy 등록 (legacy generate.rs:944-958).
    // attention sinks(4)만 보호 — Manager 가 언제/얼마나 evict 할지 결정한다.
    let resilience_protected_prefix = 4usize;
    cache_manager.register_policy(
        EvictMethod::H2o,
        Box::new(H2OPolicy::new(
            args.h2o_keep_ratio(),
            resilience_protected_prefix,
        )),
    );
    cache_manager.register_policy(
        EvictMethod::Sliding,
        Box::new(SlidingWindowPolicy::new(
            args.eviction_window(),
            resilience_protected_prefix,
        )),
    );

    Ok(cache_manager)
}

/// legacy generate.rs:971-1021 의 score_accumulator 인라인 조립을 재현.
///
/// 생성 조건: score-based(h2o/h2o_plus/d2o) || CAOTE || `--enable-resilience` ||
/// eviction!=none. GQA 변형(`new_gqa`) + `set_active(true)` +
/// `set_time_normalize` + (OpenCL 시) `init_gpu_score_acc`.
fn build_eval_score_accumulator(
    args: &Args,
    backend: &Arc<dyn Backend>,
    model: &TransformerModel,
    max_seq_len: usize,
) -> Option<AttentionScoreAccumulator> {
    let qcf_mode = match args.qcf_mode.as_str() {
        "caote" => crate::qcf::QcfMode::Caote,
        "both" => crate::qcf::QcfMode::Both,
        _ => crate::qcf::QcfMode::Attn,
    };
    let needs_caote = qcf_mode.has_caote();
    let needs_score_based = matches!(args.eviction_policy(), "h2o" | "d2o" | "h2o_plus");
    let has_eviction_policy = args.eviction_policy() != "none";
    let needs_accumulator =
        needs_score_based || needs_caote || args.enable_resilience || has_eviction_policy;
    if !needs_accumulator {
        return None;
    }

    // GQA mode required for last_step_head_attn() (QCF-ATTN v2 + CAOTE).
    let use_gqa = args.eviction_policy() == "h2o_plus" || needs_caote || has_eviction_policy;
    let mut acc = if use_gqa {
        AttentionScoreAccumulator::new_gqa(
            max_seq_len,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            model.config.num_hidden_layers,
            args.h2o_tracked_layers(),
            args.h2o_decay(),
        )
    } else {
        AttentionScoreAccumulator::new(
            max_seq_len,
            model.config.num_attention_heads,
            model.config.num_hidden_layers,
            args.h2o_tracked_layers(),
            args.h2o_decay(),
        )
    };
    acc.set_active(true);
    acc.set_time_normalize(!args.h2o_raw_scores());

    // OpenCL backend 면 GPU-side accumulator 초기화 — per-token GPU→CPU readback 제거.
    #[cfg(feature = "opencl")]
    if let Some(ocl_be) = backend
        .as_any()
        .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
    {
        match ocl_be.init_gpu_score_acc(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            max_seq_len,
            args.h2o_decay(),
        ) {
            Ok(()) => {
                if let Some(gpu_acc) = ocl_be.gpu_score_acc_mut() {
                    gpu_acc.set_active(true);
                }
                eprintln!("[GPU Score] Accumulator initialized — per-token readback eliminated");
            }
            Err(e) => {
                eprintln!("[GPU Score] Failed to initialize (falling back to CPU path): {e}");
            }
        }
    }
    let _ = backend; // opencl feature off 일 때 미사용 경고 회피.

    Some(acc)
}

/// legacy generate.rs:1065-1097 의 skip_config 인라인 조립을 재현.
///
/// `--skip-layers` 명시 → 해당 layer attn/mlp skip. `--skip-ratio` → SWIFT
/// uniform_init. 둘 다 SWIFT 제약(layer 0/마지막 skip 불가)을 validate 한다.
fn build_eval_skip_config(args: &Args, model: &TransformerModel) -> Result<Option<SkipConfig>> {
    if let Some(ref layers) = args.skip_layers {
        let mut sc = SkipConfig::new();
        for &l in layers {
            sc.attn_skip.insert(l);
            sc.mlp_skip.insert(l);
        }
        if !sc.validate(model.config.num_hidden_layers) {
            anyhow::bail!("Cannot skip layer 0 or last layer (SWIFT constraint)");
        }
        eprintln!(
            "[Skip] Explicit layers: {:?} ({} sub-layers skipped)",
            layers,
            sc.total_skips()
        );
        Ok(Some(sc))
    } else if let Some(ratio) = args.skip_ratio {
        let sc = SkipConfig::uniform_init(model.config.num_hidden_layers, ratio);
        if !sc.validate(model.config.num_hidden_layers) {
            anyhow::bail!(
                "uniform_init produced invalid SkipConfig (layer 0 or last layer skipped)"
            );
        }
        eprintln!(
            "[Skip] Uniform ratio={:.1}% → {} sub-layers skipped",
            ratio * 100.0,
            sc.total_skips()
        );
        Ok(Some(sc))
    } else {
        Ok(None)
    }
}

/// `--eval-ll` (Standard KV) 진입용 [`EvalLlRunCtx`] 를 조립한다.
///
/// pre: `eval_supported(&args)` 통과 + Standard KV mode (KIVI 는 별 경로 §13.6).
/// post: KV capacity == max_seq_len, cache_manager 의 protected_prefix == eval 정책.
pub fn build_eval_ll_ctx(args: Args) -> Result<EvalLlRunCtx> {
    let base = build_eval_base(&args)?;
    let EvalBase {
        backend,
        memory,
        cpu_backend_arc,
        gpu_backend_arc,
        caps: _caps, // Standard eval 은 caps 미소비 (KIVI 만 thread-through).
        model,
        tokenizer,
        prompt,
        kv_type,
        max_seq_len,
        swap_algorithm,
        importance_formula,
        importance_compare,
        swap_only_layers,
    } = base;

    // model move 전에 config 파생 값 캡처.
    let hidden_size = model.config.hidden_size;
    let vocab_size = model.config.vocab_size;
    let num_layers = model.config.num_hidden_layers;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;

    let prompt_enc = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let prompt_len = prompt_enc.get_ids().len();
    let actual_protected_prefix = eval_protected_prefix(&args, prompt_len);
    let score_based_eviction = is_score_based_eviction(&args);

    // eval 모드는 grow() 스파이크 회피로 capacity=max_seq_len 전량 선할당 (legacy:408).
    let kv_caches = alloc_standard_kv_caches(
        &backend,
        memory.clone(),
        num_layers,
        max_seq_len,
        max_seq_len,
        kv_heads,
        head_dim,
        kv_type,
    )?;

    let cache_manager = build_eval_cache_manager(&args, &backend, actual_protected_prefix)?;
    let score_accumulator = build_eval_score_accumulator(&args, &backend, &model, max_seq_len);
    let skip_config = build_eval_skip_config(&args, &model)?;

    Ok(EvalLlRunCtx {
        args,
        backend,
        memory,
        cpu_backend_arc,
        gpu_backend_arc,
        model,
        tokenizer,
        kv_caches,
        cache_manager,
        score_accumulator,
        skip_config,
        prompt,
        hidden_size,
        vocab_size,
        max_seq_len,
        num_layers,
        // kv_type 은 runner.rs:34 에서 `_kv_type` 으로 destructure 되는 dead field —
        // runner 시그니처 무변경 원칙으로 채우되 소비처는 없다.
        kv_type,
        actual_protected_prefix,
        score_based_eviction,
        swap_algorithm,
        importance_formula,
        importance_compare,
        swap_only_layers,
    })
}

/// `--ppl` (Standard KV) 진입용 [`PplRunCtx`] 를 조립한다.
///
/// legacy generate.rs:1698-1777 등가 — `--qcf-dump` 시 ppl 진입 전 warmup
/// workflow 를 돌려 `qcf_warmup_importance` / `qcf_swap_decision` 3필드를 채운다.
pub fn build_ppl_ctx(args: Args) -> Result<PplRunCtx> {
    let base = build_eval_base(&args)?;
    let EvalBase {
        backend,
        memory,
        cpu_backend_arc,
        gpu_backend_arc,
        caps: _caps,
        model,
        tokenizer,
        prompt,
        kv_type,
        max_seq_len,
        swap_algorithm,
        importance_formula,
        importance_compare,
        swap_only_layers,
    } = base;

    let hidden_size = model.config.hidden_size;
    let vocab_size = model.config.vocab_size;
    let num_layers = model.config.num_hidden_layers;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;

    let actual_protected_prefix = eval_protected_prefix(&args, prompt.len());
    let score_based_eviction = is_score_based_eviction(&args);
    let auto_eviction = args.eviction_policy() != "none" && args.experiment_schedule.is_none();

    let mut kv_caches = alloc_standard_kv_caches(
        &backend,
        memory.clone(),
        num_layers,
        max_seq_len,
        max_seq_len,
        kv_heads,
        head_dim,
        kv_type,
    )?;

    let cache_manager = build_eval_cache_manager(&args, &backend, actual_protected_prefix)?;
    let score_accumulator = build_eval_score_accumulator(&args, &backend, &model, max_seq_len);
    let skip_config = build_eval_skip_config(&args, &model)?;

    // ── QCF dump warmup prelude (legacy generate.rs:1698-1747) ──
    // --qcf-dump 시 ppl 진입 전 warmup prefill → ImportanceTable → swap 결정.
    let qcf_workflow_start = std::time::Instant::now();
    let mut qcf_warmup_importance: Option<crate::qcf::ImportanceTable> = None;
    let mut qcf_swap_decision: Option<crate::weight::SwapDecision> = None;

    if args.qcf_dump.is_some() && (args.ppl.is_some() || !prompt.is_empty()) {
        let warmup_n = args.qcf_warmup_tokens.max(1);
        let warmup_tokens: Vec<u32> = if let Some(ref ppl_path) = args.ppl {
            let text = std::fs::read_to_string(ppl_path)
                .map_err(|e| anyhow::anyhow!("Failed to read PPL file for warmup: {e}"))?;
            let enc = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Warmup tokenize error: {e}"))?;
            enc.get_ids().iter().take(warmup_n).copied().collect()
        } else {
            let enc = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Warmup tokenize error: {e}"))?;
            enc.get_ids().iter().take(warmup_n).copied().collect()
        };
        if warmup_tokens.is_empty() {
            anyhow::bail!(
                "--qcf-dump: warmup token sequence is empty (prompt or PPL text too short)"
            );
        }
        let result = crate::session::qcf_runtime::run_qcf_warmup_workflow(
            crate::session::qcf_runtime::QcfWarmupCtx {
                model: &model,
                backend: &backend,
                memory: memory.as_ref(),
                kv_caches: &mut kv_caches,
                vocab_size,
                warmup_ids: &warmup_tokens,
                gpu_backend: gpu_backend_arc.as_ref(),
                cpu_backend: &cpu_backend_arc,
            },
            crate::session::qcf_runtime::QcfWarmupConfig {
                force_ratio: args.force_swap_ratio,
                swap_algorithm,
                execute_swap: true,
                importance_formula,
                importance_three_way: importance_compare,
                swap_only_layers: swap_only_layers.as_deref(),
                decode_x_steps: args.decode_x_steps,
                log_prefix: "",
            },
        )?;
        qcf_swap_decision = result.decision;
        qcf_warmup_importance = Some(result.importance);
    }

    Ok(PplRunCtx {
        args,
        backend,
        memory,
        model,
        tokenizer,
        kv_caches,
        cache_manager,
        score_accumulator,
        skip_config,
        hidden_size,
        vocab_size,
        max_seq_len,
        num_layers,
        kv_heads,
        head_dim,
        actual_protected_prefix,
        score_based_eviction,
        qcf_warmup_importance,
        qcf_swap_decision,
        qcf_workflow_start,
        auto_eviction,
        swap_algorithm,
    })
}

/// `--dump-importance` 진입용 [`DumpImportanceCtx`] 를 조립한다 (8필드, 최소).
///
/// legacy generate.rs:1634-1647 등가. eviction/score/skip 상태 불필요.
pub fn build_dump_importance_ctx(args: Args) -> Result<DumpImportanceCtx> {
    let base = build_eval_base(&args)?;
    let vocab_size = base.model.config.vocab_size;
    let model_path = args.model_path.clone();
    // KV capacity=max_seq_len 선할당 (eval 모드 일관 — grow 스파이크 회피).
    let kv_caches = alloc_standard_kv_caches(
        &base.backend,
        base.memory.clone(),
        base.model.config.num_hidden_layers,
        base.max_seq_len,
        base.max_seq_len,
        base.model.config.num_key_value_heads,
        base.model.config.head_dim,
        base.kv_type,
    )?;
    Ok(DumpImportanceCtx {
        backend: base.backend,
        memory: base.memory,
        model: base.model,
        tokenizer: base.tokenizer,
        kv_caches,
        prompt: base.prompt,
        vocab_size,
        model_path,
    })
}

/// `--eval-ll --kv-mode kivi` 진입 — KiviCache 기반 log-likelihood eval.
///
/// legacy generate.rs:274-369 재현. `EvalLlRunCtx` 를 경유하지 않고
/// `run_eval_ll_generic` 을 KiviHook + `Vec<KiviCache>` 로 직접 호출한다.
/// `caps.get::<dyn KiviAttentionBackend>()` 를 closure 밖에서 1회 pull —
/// chat 의 `build_chat_kivi`(session/chat/session.rs:412) thread-through 미러.
///
/// 빌드부터 run 까지 한 함수에 둔다 (KIVI 는 ctx struct 표면이 없음).
pub fn run_eval_ll_kivi(args: Args) -> Result<()> {
    use crate::backend::KiviAttentionBackend;
    use crate::kv::kivi_cache::KiviCache;
    use crate::session::cli::parse_qcf_sample_layers;
    use crate::session::eval::{EvalConfig, KiviHook, run_eval_ll_generic};

    let base = build_eval_base(&args)?;
    let EvalBase {
        backend,
        memory,
        caps,
        model,
        tokenizer,
        prompt,
        max_seq_len,
        ..
    } = base;

    let questions = crate::session::eval::load_eval_questions(&args, &prompt)?;
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let num_layers = model.config.num_hidden_layers;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;

    let eval_config = EvalConfig {
        max_seq_len,
        effective_budget: 0,
        kv_budget_ratio: 0.0,
        greedy: args.greedy,
        kv_type: format!("q{}+f32_residual", args.effective_kivi_bits()),
        qcf_mode: args.qcf_mode.clone(),
        vocab_size,
        hidden_size,
    };
    let qcf_config = crate::qcf_types::QcfConfig::default();
    let kivi_bits = args.effective_kivi_bits();

    // KIVI native attention handle 을 caps 에서 1회 pull (closure 밖). OpenCL 면 Some.
    let kivi_cap = caps.get::<dyn KiviAttentionBackend>();
    let mut kv_caches: Vec<KiviCache> = (0..num_layers)
        .map(|_| {
            KiviCache::new_gpu(
                kv_heads,
                head_dim,
                max_seq_len,
                args.effective_kivi_residual_size(),
                kivi_bits,
                backend.clone(),
                kivi_cap.clone(),
                memory.clone(),
            )
        })
        .collect();

    // AWQE / AW-VOPR metric collection (env-gated, measurement-only).
    if std::env::var("LLMRS_KIVI_AWQE").is_ok() {
        for cache in kv_caches.iter_mut() {
            cache.set_awqe_enabled(true);
        }
        eprintln!("[KIVI] AWQE + AW-VOPR enabled (LLMRS_KIVI_AWQE)");
    }

    let kivi_n_layers = kv_caches.len();
    let kivi_sample_layers = if args.enable_qcf_experimental {
        parse_qcf_sample_layers(&args.qcf_sample_layers, kivi_n_layers)
            .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
    } else {
        vec![0]
    };
    let kivi_score_acc = if args.enable_qcf_experimental {
        let mut acc = AttentionScoreAccumulator::new_gqa(
            max_seq_len,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            kivi_n_layers,
            0,
            1.0,
        );
        acc.set_active(true);
        Some(acc)
    } else {
        None
    };

    let mut hook = KiviHook::new(
        qcf_config,
        args.enable_qcf_experimental,
        kivi_sample_layers,
        kivi_score_acc,
    );
    let output = run_eval_ll_generic(
        &model,
        &tokenizer,
        &backend,
        memory.as_ref(),
        &mut kv_caches,
        &mut hook,
        &questions,
        &eval_config,
        None,
    )?;
    let mut json_val = serde_json::from_str::<serde_json::Value>(&output.to_json()?)?;
    json_val["config"] = serde_json::json!({
        "model": args.model_path,
        "eviction_policy": "kivi",
        "kivi_bits": args.effective_kivi_bits(),
        "kivi_residual_size": args.effective_kivi_residual_size(),
        "max_seq_len": max_seq_len,
        "kv_type": format!("q{}+f32_residual", args.effective_kivi_bits()),
    });
    println!("{}", serde_json::to_string_pretty(&json_val)?);
    Ok(())
}

/// `--ppl --kv-mode kivi` 진입 — KiviCache 기반 perplexity eval.
///
/// legacy generate.rs:372-389 재현. `run_kivi_ppl` free fn 을 caps 에서 pull 한
/// KIVI handle 과 함께 직접 호출한다.
pub fn run_ppl_kivi(args: Args, ppl_path: &str) -> Result<()> {
    use crate::backend::KiviAttentionBackend;

    let base = build_eval_base(&args)?;
    let kivi = base.caps.get::<dyn KiviAttentionBackend>();
    crate::session::ppl::run_kivi_ppl(
        &args,
        &base.model,
        &base.tokenizer,
        &base.backend,
        kivi,
        &base.memory,
        base.model.config.num_key_value_heads,
        base.model.config.head_dim,
        base.model.config.num_hidden_layers,
        base.max_seq_len,
        args.effective_kivi_residual_size(),
        ppl_path,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// 최소 인자 + nested eviction subcommand(끝)로 Args 를 만든다.
    fn parse_args(extra: &[&str]) -> Args {
        let mut argv = vec!["argus-eval", "--model-path", "/tmp/model.gguf"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).expect("Args parse")
    }

    /// eviction=none + protected-prefix 미지정 → prompt 전체 보호 (legacy 동작).
    #[test]
    fn protected_prefix_none_defaults_to_prompt_len() {
        let args = parse_args(&["--eval-ll"]);
        assert_eq!(args.eviction_policy(), "none");
        assert_eq!(eval_protected_prefix(&args, 37), 37);
    }

    /// sliding + 미지정 → prompt 전체 보호.
    #[test]
    fn protected_prefix_sliding_defaults_to_prompt_len() {
        let args = parse_args(&["--eval-ll", "eviction", "sliding"]);
        assert_eq!(args.eviction_policy(), "sliding");
        assert_eq!(eval_protected_prefix(&args, 50), 50);
    }

    /// h2o/h2o_plus/d2o + 미지정 → 4 (attention sinks only).
    #[test]
    fn protected_prefix_score_based_defaults_to_4() {
        // clap subcommand 는 kebab-case(`h2o-plus`), canonical policy_name 은 snake_case(`h2o_plus`).
        for (subcmd, policy) in [("h2o", "h2o"), ("h2o-plus", "h2o_plus"), ("d2o", "d2o")] {
            let args = parse_args(&["--eval-ll", "eviction", subcmd]);
            assert_eq!(args.eviction_policy(), policy);
            assert_eq!(
                eval_protected_prefix(&args, 100),
                4,
                "policy {policy} should default protected_prefix to 4"
            );
        }
    }

    /// --protected-prefix 명시 시 정책과 무관하게 그 값.
    #[test]
    fn protected_prefix_explicit_overrides_default() {
        let args = parse_args(&["--eval-ll", "--protected-prefix", "12", "eviction", "h2o"]);
        assert_eq!(eval_protected_prefix(&args, 100), 12);
    }
}
