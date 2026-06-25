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
//! 단 [`SessionInitCtx::build`] 를 **직접 호출**해 `caps`(quant-window 라우팅 필수) /
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
use crate::kv::eviction::EvictionPolicy;
use crate::kv::eviction::stage_registry::{
    StageBackedPolicy, make_stage_with_args, stage_default_protected_prefix, stage_is_score_based,
};
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
/// `caps` 는 quant-window 경로 thread-through 용으로 보존된다.
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
    stage_is_score_based(args.eviction_policy())
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
        // Score-based stages declare a protected-prefix (4 attention sinks); score-free stages
        // declare 0 → the engine protects the whole prompt (legacy sliding/none default). Streaming
        // derives its own sink and ignores StageParams.protected_prefix, so no per-name branch is
        // needed.
        match stage_default_protected_prefix(args.eviction_policy()) {
            0 => prompt_len,
            cap => cap,
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

    let mut cache_manager = {
        let name = args.eviction_policy();
        // Every policy (none/sliding/streaming/h2o/h2o_plus/d2o/rkv) resolves through the plugin
        // registry by name (static linkme + dynamic `--load-plugin`), with its private knobs riding
        // the opaque StageArgs blob — this site names no plugin.
        let policy: Box<dyn EvictionPolicy> = {
            let streaming_window = if args.streaming_window() > 0 {
                args.streaming_window()
            } else if args.kv_budget() > 0 {
                args.kv_budget().saturating_sub(args.sink_size())
            } else {
                args.eviction_window()
            };
            let params = argus_extension_api::StageParams {
                eviction_window: args.eviction_window(),
                protected_prefix: actual_protected_prefix,
                keep_ratio: args.keep_ratio(),
                sink_size: args.sink_size(),
                streaming_window,
            };
            let extra_owned = args.stage_args();
            let extra: Vec<argus_extension_api::PluginArg> = extra_owned
                .iter()
                .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
                .collect();
            let stage = make_stage_with_args(name, &params, &extra).ok_or_else(|| {
                anyhow::anyhow!(
                    "argus-eval: unknown eviction policy '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o (or --load-plugin <.so>).",
                    name
                )
            })?;
            Box::new(StageBackedPolicy::new(stage))
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

    Ok(cache_manager)
}

/// legacy generate.rs:971-1021 의 score_accumulator 인라인 조립을 재현.
///
/// 생성 조건: score-based(h2o/h2o_plus/d2o) || value-aware || `--enable-resilience` ||
/// eviction!=none. GQA 변형(`new_gqa`) + `set_active(true)` +
/// `set_time_normalize` + (OpenCL 시) `init_gpu_score_acc`.
fn build_eval_score_accumulator(
    args: &Args,
    backend: &Arc<dyn Backend>,
    model: &TransformerModel,
    max_seq_len: usize,
) -> Option<AttentionScoreAccumulator> {
    // `--qcf-mode caote|both` forces the score accumulator + GQA even with no eviction policy.
    // The value-aware QcfMode variant was name-only residue (no distinct arithmetic), so gate on the CLI
    // string directly (B1-2).
    let needs_caote = matches!(args.qcf_mode.as_str(), "caote" | "both");
    let needs_score_based = stage_is_score_based(args.eviction_policy());
    let has_eviction_policy = args.eviction_policy() != "none";
    let needs_accumulator =
        needs_score_based || needs_caote || args.enable_resilience || has_eviction_policy;
    if !needs_accumulator {
        return None;
    }

    // GQA mode required for last_step_head_attn() (QCF-ATTN v2 + value-aware) and for any per-head stage.
    // Any non-none eviction policy already enables it (has_eviction_policy), so no per-name branch.
    let use_gqa = needs_caote || has_eviction_policy;
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

    // IMP-1: arm the non-collapsed per-(layer, KV-head) importance buffer when the
    // `evict_importance` dump is requested. No-op in flat mode; eviction policies use
    // GQA (use_gqa above), so the dump captures real per-KV-head data. The flat
    // importance the policy ranks on is unchanged either way (INV-147).
    if args.dump_enabled(crate::session::eval::dump::DUMP_EVICT_IMPORTANCE) {
        acc.enable_layer_head_dump();
    }

    // OpenCL backend 면 GPU-side accumulator 초기화 — per-token GPU→CPU readback 제거.
    // Caps-driven arming: only arm the GPU score path when the eviction policy actually consumes
    // scores (`StageCaps.reads` non-empty). A score-free policy (sliding/streaming) or a
    // measurement-only `--qcf-mode caote`/`--enable-resilience` build still gets the CPU accumulator
    // above, but never writes/reduces GPU scores it would never read.
    #[cfg(feature = "opencl")]
    if needs_score_based
        && let Some(ocl_be) = backend
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
/// `--skip-layers` 명시 → 해당 layer attn/mlp skip. `--skip-ratio` → uniform_init.
/// 둘 다 layer 0/마지막 skip 불가 제약을 validate 한다.
fn build_eval_skip_config(args: &Args, model: &TransformerModel) -> Result<Option<SkipConfig>> {
    if let Some(ref layers) = args.skip_layers {
        let mut sc = SkipConfig::new();
        for &l in layers {
            sc.attn_skip.insert(l);
            sc.mlp_skip.insert(l);
        }
        if !sc.validate(model.config.num_hidden_layers) {
            anyhow::bail!("Cannot skip layer 0 or last layer (boundary constraint)");
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
/// pre: `eval_supported(&args)` 통과 + Standard KV mode (quant-window 는 별 경로 §13.6).
/// post: KV capacity == max_seq_len, cache_manager 의 protected_prefix == eval 정책.
pub fn build_eval_ll_ctx(args: Args) -> Result<EvalLlRunCtx> {
    let base = build_eval_base(&args)?;
    let EvalBase {
        backend,
        memory,
        cpu_backend_arc,
        gpu_backend_arc,
        caps: _caps, // Standard eval 은 caps 미소비 (quant-window 만 thread-through).
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
    let kv_layout = crate::kv_cache_ops::KVLayout::from_cli(&args.kv_layout)
        .ok_or_else(|| anyhow::anyhow!("Unsupported --kv-layout: '{}'", args.kv_layout))?;
    let kv_caches = alloc_standard_kv_caches(
        &backend,
        memory.clone(),
        num_layers,
        max_seq_len,
        max_seq_len,
        kv_heads,
        head_dim,
        kv_type,
        kv_layout,
    )?;

    let cache_manager = build_eval_cache_manager(&args, &backend, actual_protected_prefix)?;
    crate::inference::attention_scores::ensure_score_producers_registered()?;
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

    let kv_layout = crate::kv_cache_ops::KVLayout::from_cli(&args.kv_layout)
        .ok_or_else(|| anyhow::anyhow!("Unsupported --kv-layout: '{}'", args.kv_layout))?;
    let mut kv_caches = alloc_standard_kv_caches(
        &backend,
        memory.clone(),
        num_layers,
        max_seq_len,
        max_seq_len,
        kv_heads,
        head_dim,
        kv_type,
        kv_layout,
    )?;

    let cache_manager = build_eval_cache_manager(&args, &backend, actual_protected_prefix)?;
    crate::inference::attention_scores::ensure_score_producers_registered()?;
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
    let kv_layout = crate::kv_cache_ops::KVLayout::from_cli(&args.kv_layout)
        .ok_or_else(|| anyhow::anyhow!("Unsupported --kv-layout: '{}'", args.kv_layout))?;
    let kv_caches = alloc_standard_kv_caches(
        &base.backend,
        base.memory.clone(),
        base.model.config.num_hidden_layers,
        base.max_seq_len,
        base.max_seq_len,
        base.model.config.num_key_value_heads,
        base.model.config.head_dim,
        base.kv_type,
        kv_layout,
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

/// `--eval-ll --kv-mode kivi` 진입 — QuantizedRecentWindowCache 기반 log-likelihood eval.
///
/// legacy generate.rs:274-369 재현. `EvalLlRunCtx` 를 경유하지 않고
/// `run_eval_ll_generic` 을 QuantWindowFlushHook + `Vec<QuantizedRecentWindowCache>` 로 직접 호출한다.
/// `caps.get::<dyn QuantAttnBackend>()` 를 closure 밖에서 1회 pull —
/// chat 의 `build_chat_quant_window_forward`(session/chat/session.rs, KvModeReg.build) thread-through 미러.
///
/// 빌드부터 run 까지 한 함수에 둔다 (quant-window 는 ctx struct 표면이 없음).
pub fn run_eval_ll_quant_window(args: Args) -> Result<()> {
    use crate::backend::QuantAttnBackend;
    use crate::kv::quant_window_cache::QuantizedRecentWindowCache;
    use crate::session::cli::parse_qcf_sample_layers;
    use crate::session::eval::{EvalConfig, QuantWindowFlushHook, run_eval_ll_generic};

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
        kv_type: format!("q{}+f32_residual", args.effective_quant_window_bits()),
        qcf_mode: args.qcf_mode.clone(),
        vocab_size,
        hidden_size,
    };
    let qcf_config = crate::qcf_types::QcfConfig::default();
    let quant_window_bits = args.effective_quant_window_bits();

    // quant-window native attention handle 을 caps 에서 1회 pull (closure 밖). OpenCL 면 Some.
    let quant_attn_cap = caps.get::<dyn QuantAttnBackend>();
    let mut kv_caches: Vec<QuantizedRecentWindowCache> = (0..num_layers)
        .map(|_| {
            QuantizedRecentWindowCache::new_gpu(
                kv_heads,
                head_dim,
                max_seq_len,
                args.effective_quant_window_residual_size(),
                quant_window_bits,
                backend.clone(),
                quant_attn_cap.clone(),
                memory.clone(),
            )
        })
        .collect();

    // AWQE / AW-VOPR metric collection (env-gated, measurement-only).
    if std::env::var("LLMRS_KIVI_AWQE").is_ok() {
        for cache in kv_caches.iter_mut() {
            cache.set_awqe_enabled(true);
        }
        eprintln!("[quant-window] AWQE + AW-VOPR enabled (LLMRS_KIVI_AWQE)");
    }

    let kivi_n_layers = kv_caches.len();
    let kivi_sample_layers = if args.enable_qcf_experimental {
        parse_qcf_sample_layers(&args.qcf_sample_layers, kivi_n_layers)
            .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
    } else {
        vec![0]
    };
    let kivi_score_acc = if args.enable_qcf_experimental {
        crate::inference::attention_scores::ensure_score_producers_registered()?;
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

    let mut hook = QuantWindowFlushHook::new(
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
        None, // evict_importance dump: quant-window path has no eviction hook
    )?;
    let mut json_val = serde_json::from_str::<serde_json::Value>(&output.to_json()?)?;
    json_val["config"] = serde_json::json!({
        "model": args.model_path,
        "eviction_policy": "quant_window",
        "quant_bits": args.effective_quant_window_bits(),
        "quant_residual_size": args.effective_quant_window_residual_size(),
        "max_seq_len": max_seq_len,
        "kv_type": format!("q{}+f32_residual", args.effective_quant_window_bits()),
    });
    println!("{}", serde_json::to_string_pretty(&json_val)?);
    Ok(())
}

/// `--ppl --kv-mode kivi` 진입 — QuantizedRecentWindowCache 기반 perplexity eval.
///
/// legacy generate.rs:372-389 재현. `run_quant_window_ppl` free fn 을 caps 에서 pull 한
/// quant-window handle 과 함께 직접 호출한다.
pub fn run_ppl_quant_window(args: Args, ppl_path: &str) -> Result<()> {
    use crate::backend::QuantAttnBackend;

    let base = build_eval_base(&args)?;
    let quant_attn = base.caps.get::<dyn QuantAttnBackend>();
    crate::session::ppl::run_quant_window_ppl(
        &args,
        &base.model,
        &base.tokenizer,
        &base.backend,
        quant_attn,
        &base.memory,
        base.model.config.num_key_value_heads,
        base.model.config.head_dim,
        base.model.config.num_hidden_layers,
        base.max_seq_len,
        args.effective_quant_window_residual_size(),
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
        let args = parse_args(&["--eval-ll", "eviction", "plugin", "--name", "sliding"]);
        assert_eq!(args.eviction_policy(), "sliding");
        assert_eq!(eval_protected_prefix(&args, 50), 50);
    }

    /// h2o/h2o_plus/d2o + 미지정 → 4 (attention sinks only).
    #[test]
    fn protected_prefix_score_based_defaults_to_4() {
        // Stages are now selected generically by registry name (snake_case `h2o_plus`).
        for policy in ["h2o", "h2o_plus", "d2o"] {
            let args = parse_args(&["--eval-ll", "eviction", "plugin", "--name", policy]);
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
        let args = parse_args(&[
            "--eval-ll",
            "--protected-prefix",
            "12",
            "eviction",
            "plugin",
            "--name",
            "h2o",
        ]);
        assert_eq!(eval_protected_prefix(&args, 100), 12);
    }
}
