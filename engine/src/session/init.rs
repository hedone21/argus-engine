use std::sync::Arc;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::capability::CapabilityRegistry;
use crate::hardware::Hardware;
use crate::inference::sampling::SamplingConfig;
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::TransformerModel;
use crate::observability::rss_trace::{dump_smaps, io_trace, rss_trace};
use crate::session::cli::Args;

/// Session 초기화 컨텍스트 (Phase 4-1 외곽 추출).
///
/// `SessionInitCtx::build(&args)`가 generate.rs main()의 args 검증,
/// 환경변수 전파, Rayon 초기화, Backend/Memory init, 모델 로드, weight probe를
/// 수행한다. build() 완료 후 ctx는 모든 하위 decode 경로에서 필요한 값들을 보유한다.
///
/// - `args`는 main()이 owned 보유; build()는 `&Args` borrow + 필요 필드만 clone.
/// - `model`은 owned으로 보유 (quant-window/SwitchHw가 model.layers를 mutation).
/// - `is_gpu`, `weights_on_gpu`는 main()이 mutation할 수 있도록 pub으로 노출.
pub struct SessionInitCtx {
    /// sampling 파라미터 (greedy override 처리 완료).
    pub sampling_config: SamplingConfig,
    /// 모델 파일 경로 (GGUF 또는 safetensors 디렉토리).
    pub model_path: String,
    /// 모델이 GGUF 포맷이면 true.
    pub is_gguf: bool,

    /// 주 backend (CPU 또는 GPU). "현재 활성 device" = decode-loop local.
    pub backend: Arc<dyn Backend>,
    /// 주 메모리 할당자.
    pub memory: Arc<dyn Memory>,
    /// 현재 GPU가 primary backend이면 true (main()이 SwitchHw 후 mutation).
    pub is_gpu: bool,
    /// 모델 weights가 GPU cl_mem에 있으면 true.
    pub weights_on_gpu: bool,

    /// hardware 축 resolver — 흩어진 cpu/gpu backend·memory secondary Arc를 흡수.
    /// `hardware.resolve(target)` 로 (backend, memory) 페어를 푼다 (Phase α-W-2).
    pub hardware: Arc<Hardware>,

    /// L2 backend capability 레지스트리 (Phase α-W-4 §3.3). OpenCL backend 면
    /// `QuantAttnBackend` handle 이 등록되어 있고, 그 외(CPU-only/CUDA)는 빈
    /// registry 다. 소비자(QuantizedRecentWindowCache 생성 caller)가 construction 시점에 1회
    /// `caps.get::<dyn QuantAttnBackend>()` pull 한다.
    pub caps: Arc<CapabilityRegistry>,

    /// swap layer 선택 알고리즘 (--swap-algorithm).
    pub swap_algorithm: crate::weight::SwapAlgorithm,
    /// layer 중요도 계산 공식 (--importance-formula).
    pub importance_formula: crate::qcf_types::ImportanceFormula,
    /// `compare` 모드 활성 여부 (importance_formula = MeanPool이지만 3-way 수집).
    pub importance_compare: bool,
    /// 명시적 swap 대상 layer 목록 (--swap-only-layers, § 4 ground-truth study).
    pub swap_only_layers: Option<Vec<usize>>,

    /// 로드된 모델 (quant-window/SwitchHw가 model.layers를 swap하므로 owned).
    pub model: TransformerModel,
}

impl SessionInitCtx {
    pub fn build(args: &Args) -> anyhow::Result<Self> {
        // ENG-DAT-C18: --swap-incremental-per-tick > 0 / --swap-intra-forward /
        // --swap-phase-aware are mutually exclusive (LISWAP-1 vs LISWAP-4 vs
        // LISWAP-5 — ratio_generation bump + dispatcher ownership conflict).
        // Reject combinations explicitly so engine never starts in an ambiguous
        // swap-policy state.
        let swap_modes_active = (args.swap_incremental_per_tick > 0) as usize
            + args.swap_intra_forward as usize
            + args.swap_phase_aware as usize
            + args.swap_layer_immediate as usize;
        if swap_modes_active > 1 {
            anyhow::bail!(
                "--swap-incremental-per-tick (= {}) / --swap-intra-forward (= {}) / \
                 --swap-phase-aware (= {}) / --swap-layer-immediate (= {}) are mutually \
                 exclusive (ENG-DAT-C18). Pick one:\n\
                 (a) --swap-incremental-per-tick=N                                 (LISWAP-1)\n\
                 (b) --swap-intra-forward=true                                     (LISWAP-4)\n\
                 (c) --swap-phase-aware=true                                       (LISWAP-5)\n\
                 (d) --swap-layer-immediate=true                                   (LISWAP-6 P6)\n\
                 (e) (none)                                                        (single-shot)",
                args.swap_incremental_per_tick,
                args.swap_intra_forward,
                args.swap_phase_aware,
                args.swap_layer_immediate
            );
        }

        // --swap-no-throttle: forwards to env so SwapExecutor::execute_on_slots
        // skips the INV-141 release_worker drain. Measurement-only (EuroSys 2027
        // §4.2 layer-count predictor accuracy). Sets the env only if unset so the
        // env-based invocation path (LLMRS_SWAP_FORCE_EVERY_TICK=1) keeps working
        // independently. The executor logs a stderr warning on first read.
        if args.swap_no_throttle && std::env::var_os("LLMRS_SWAP_FORCE_EVERY_TICK").is_none() {
            // SAFETY: set before any worker thread that might read the variable.
            // generate.rs::main runs single-threaded up to this point (CLI parse +
            // Rayon pool init below). Writes after thread spawn would be UB on
            // some platforms; this write precedes the pool builder on line 1247.
            unsafe { std::env::set_var("LLMRS_SWAP_FORCE_EVERY_TICK", "1") };
        }

        // Configure Rayon thread pool: 0 = auto-detect CPU cores
        let num_threads = if args.threads > 0 {
            args.threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
        eprintln!("[Config] Using {} threads", num_threads);

        // --chat conflict validation.
        // Standard / kivi / offload paths are each supported; experiment/eval
        // modes and advanced GPU features remain incompatible.
        if args.chat {
            // §4.3 / site #5: conflict policy reads declared caps (`ModeCaps`), not a
            // concrete-technique `matches!`. Exactly as StageCaps removed
            // `matches!(name, "h2o" | ..)` from the eviction path.
            let caps = crate::session::mode::mode_caps(args.effective_kv_mode());
            let is_quantized = caps.is_some_and(|c| c.is_quantized_kv);
            let kv_offload_active = caps.is_some_and(|c| c.supports_offload);
            let has_eviction = args.eviction_policy() != "none";
            if is_quantized && kv_offload_active {
                anyhow::bail!("--chat: --kivi and --kv-offload are mutually exclusive");
            }
            if is_quantized && has_eviction {
                anyhow::bail!("--chat: --kivi cannot combine with eviction in v1 (pick one)");
            }
            if kv_offload_active && has_eviction {
                anyhow::bail!("--chat: --kv-offload cannot combine with eviction in v1 (pick one)");
            }
            let conflicts: &[(&str, bool)] = &[
                ("--eval-ll", args.eval_ll),
                ("--ppl", args.ppl.is_some()),
                ("--eval-batch", args.eval_batch.is_some()),
                ("--tensor-partition", args.tensor_partition > 0.0),
                ("--cuda-graph", args.cuda_graph),
                ("--dump-importance", args.dump_importance),
                ("--experiment-schedule", args.experiment_schedule.is_some()),
            ];
            if let Some((flag, _)) = conflicts.iter().find(|(_, enabled)| *enabled) {
                anyhow::bail!(
                    "--chat is incompatible with {} (v1 supports standard / --kivi / --kv-offload / eviction paths)",
                    flag
                );
            }
        }

        // --profile and --profile-events are mutually exclusive.
        // Both probe the decode path but use incompatible mechanisms:
        //   --profile          : CPU wall clock + per-op clFinish (adds ~54 ms/tok)
        //   --profile-events   : GPU profiling events (near-zero overhead)
        if args.profile && args.profile_events {
            anyhow::bail!(
                "--profile and --profile-events are mutually exclusive. \
                 Use --profile-events for absolute GPU per-op timing (Adreno/llama.cpp comparison), \
                 or --profile for legacy CPU-wall-clock relative ranking."
            );
        }

        // --greedy overrides temperature to 0 (args is &Args so we apply inline)
        let effective_temperature = if args.greedy { 0.0 } else { args.temperature };
        let sampling_config = SamplingConfig {
            temperature: effective_temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            repetition_penalty: args.repetition_penalty,
            repetition_window: args.repetition_window,
        };

        let model_path = &args.model_path;

        // Propagate GPU queue priority to OpenCLBackend via env var (same
        // convention as OCL_PLATFORM / OCL_DEVICE_TYPE). CLI wins over an
        // already-set env var so a flag in a script overrides the shell env.
        if args.gpu_priority != "normal" {
            unsafe {
                std::env::set_var("OCL_QUEUE_PRIORITY", &args.gpu_priority);
            }
        }

        // 1. Setup
        eprintln!("[Profile] Event: ModelLoadStart");
        eprintln!("Loading model from {}", model_path);

        // Backend initialization: primary backend + secondary for SwitchHw resilience.
        // GPU secondary is auto-initialized when available (soft failure OK).
        #[allow(clippy::type_complexity)]
        let (backend, memory, gpu_backend_arc, gpu_memory_arc, is_gpu, caps): (
            Arc<dyn Backend>,
            Arc<dyn Memory>,
            Option<Arc<dyn Backend>>,
            Option<Arc<dyn Memory>>,
            bool,
            CapabilityRegistry,
        ) = match args.backend.as_str() {
            "cpu" => {
                let cpu = Arc::new(CpuBackend::new()) as Arc<dyn Backend>;
                let cpu_mem: Arc<dyn Memory> = Arc::new(Galloc::new());
                // Try to init GPU as secondary for SwitchHw resilience
                #[cfg(feature = "opencl")]
                let (gpu_be, gpu_mem_arc) =
                    match crate::backend::opencl::OpenCLBackend::new_with_profile_events(
                        // MSG-068 Phase 2: heartbeat-gpu-profile도 같은 queue
                        // profiling 인프라를 사용하므로 어느 한쪽이 켜지면 활성화.
                        args.profile_events || args.heartbeat_gpu_profile,
                    ) {
                        Ok(gpu_concrete) => {
                            let gpu_concrete = Arc::new(gpu_concrete);
                            let gm: Arc<dyn Memory> =
                                Arc::new(crate::backend::opencl::memory::OpenCLMemory::new(
                                    gpu_concrete.context.clone(),
                                    gpu_concrete.queue.clone(),
                                    !args.no_zero_copy,
                                ));
                            let g = gpu_concrete as Arc<dyn Backend>;
                            eprintln!(
                                "[Backend] CPU primary, GPU secondary available (SwitchHw ready)"
                            );
                            (Some(g), Some(gm))
                        }
                        Err(e) => {
                            eprintln!("[Backend] CPU only (GPU init failed: {})", e);
                            (None, None)
                        }
                    };
                #[cfg(not(feature = "opencl"))]
                let (gpu_be, gpu_mem_arc): (
                    Option<Arc<dyn Backend>>,
                    Option<Arc<dyn Memory>>,
                ) = (None, None);
                // CPU primary: backend(=CpuBackend) 는 quant-window native attention capability
                // 미보유 → 빈 registry. GPU secondary 가 있어도 primary backend 가 GPU
                // 가 아니므로(decode 가 GPU new_gpu 로 진입 안 함) register 하지 않는다.
                (
                    cpu,
                    cpu_mem,
                    gpu_be,
                    gpu_mem_arc,
                    false,
                    CapabilityRegistry::new(),
                )
            }
            #[cfg(feature = "opencl")]
            "opencl" | "gpu" => {
                // ENG-RPCMEM-042 / INV-RPCMEM-006 — Sprint 2a Phase 2:
                // --opencl-rpcmem 은 --backend opencl 전용.
                // 실질 mutex 는 아래 opencl_rpcmem 인자로 new_with_options 에 전달.
                let gpu_concrete =
                    Arc::new(crate::backend::opencl::OpenCLBackend::new_with_options(
                        // MSG-068 Phase 2: heartbeat-gpu-profile도 같은 queue
                        // profiling 인프라를 사용하므로 어느 한쪽이 켜지면 활성화.
                        args.profile_events || args.heartbeat_gpu_profile,
                        // ENG-RPCMEM-042: pass --opencl-rpcmem flag.
                        args.opencl_rpcmem,
                    )?);
                // Zero-copy is the default. `--no-zero-copy` disables it unless
                // another feature requires host-accessible buffers (resilience
                // SwitchHw, tensor partition, prefill CPU interleave) — those
                // force-enable regardless of `--no-zero-copy`.
                let mut effective_zero_copy = !args.no_zero_copy
                    || args.resilience_prealloc_switch
                    || args.tensor_partition > 0.0
                    || args.prefill_cpu_chunk_size > 0
                    || args.enable_resilience;
                if args.no_zero_copy
                    && (args.resilience_prealloc_switch
                        || args.tensor_partition > 0.0
                        || args.prefill_cpu_chunk_size > 0
                        || args.enable_resilience)
                {
                    eprintln!(
                        "[Config] --no-zero-copy ignored: required for CPU-accessible buffers"
                    );
                }
                // LLMRS_FORCE_DEVICE_ALLOC: RSS diagnostic flag.
                // Forces effective_zero_copy=false so OpenCLMemory::alloc() creates
                // OpenCLBuffer (READ_WRITE device-only) instead of UnifiedBuffer
                // (CL_MEM_ALLOC_HOST_PTR).  This lets the Tester measure the RSS
                // contribution of ALLOC_HOST_PTR vs device-only allocations.
                //
                // Independent from FORCE_DEVICE_ONLY (backend-level flag) — both can
                // be set simultaneously to ensure all paths use device-only memory.
                // When only LLMRS_FORCE_DEVICE_ALLOC is set, the primary alloc path
                // (OpenCLMemory::alloc) goes device-only but any backend-level zero-copy
                // overrides (e.g. --zero-copy CLI flag processed above) are suppressed.
                if std::env::var("LLMRS_FORCE_DEVICE_ALLOC").is_ok() {
                    effective_zero_copy = false;
                    eprintln!(
                        "[RSS-diag] LLMRS_FORCE_DEVICE_ALLOC set: effective_zero_copy forced to false \
                         (primary memory = device-only)"
                    );
                }
                // ENG-RPCMEM-042 / INV-RPCMEM-002: OpenCLMemory 에 동일 Arc<RpcmemAllocator>
                // 주입. new_with_rpcmem 은 None 일 때 기존 new() 와 동일 동작.
                let rpcmem_alloc = gpu_concrete.rpcmem_allocator();
                let gpu_mem: Arc<dyn Memory> = Arc::new(
                    crate::backend::opencl::memory::OpenCLMemory::new_with_rpcmem(
                        gpu_concrete.context.clone(),
                        gpu_concrete.queue.clone(),
                        effective_zero_copy,
                        rpcmem_alloc,
                    ),
                );
                // Stage E: the engine ships NO built-in QuantAttn cap — the OpenCL quant-window kernels
                // moved to the `kivi` plugin. Only `--backend-cap <name>` registers a cap (the
                // resolved plugin, built from the live GPU context). With no flag, no cap is
                // registered; GPU quant-window then requires `--backend-cap kivi_abi --load-plugin
                // <kivi>.so`, and `QuantizedRecentWindowCache`'s own GPU path surfaces the missing
                // cap (the engine core names no technique). R4 불변식: the resolved cap is built
                // from `gpu_concrete`'s context via `with_quant_attn_make_args`, so it shares the
                // same ocl instance the primary backend (and the cache) uses.
                // gpu_score 는 결정 #5 에 따라 register 하지 않는다(소비자 0, β 까지).
                let mut caps = CapabilityRegistry::new();
                if let Some(name) = args.backend_cap.as_deref() {
                    // --backend-cap <name>: resolve a named quantized-attention capability
                    // (static QUANT_ATTN_REGS first, then dynamic --load-plugin) using the live
                    // GPU context, and register it.
                    let cap = gpu_concrete
                        .with_quant_attn_make_args(|make_args| {
                            crate::capability::dynamic_backend_registry::resolve_quant_attn_capability(
                                name, make_args,
                            )
                        })
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Unknown --backend-cap '{name}' (not found in static \
                                 QUANT_ATTN_REGS or dynamic registration — check --load-plugin)"
                            )
                        })?;
                    caps.register::<dyn crate::backend::QuantAttnBackend>(cap);
                }
                let gpu: Arc<dyn Backend> = gpu_concrete;
                // GPU is primary; keep a ref as secondary for SwitchHw round-trip
                (
                    gpu.clone(),
                    gpu_mem.clone(),
                    Some(gpu),
                    Some(gpu_mem),
                    true,
                    caps,
                )
            }
            #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
            "cuda" => {
                let gpu_concrete = Arc::new(crate::backend::cuda::CudaBackend::new()?);
                let gpu_mem: Arc<dyn Memory> = if gpu_concrete.is_discrete_gpu() {
                    Arc::new(crate::backend::cuda::memory::CudaMemory::managed())
                } else {
                    Arc::new(crate::backend::cuda::memory::CudaMemory::new())
                };
                // --cuda-profile: event-based per-op profiler. Only wired on
                // the cuda-embedded backend (PC cuda path has its own
                // profiling story and doesn't expose enable_profiler).
                #[cfg(feature = "cuda-embedded")]
                if args.cuda_profile {
                    gpu_concrete.enable_profiler(4096)?;
                }
                // --cuda-sync-policy: fine-grained per-category bisection.
                // Parsed before weights-device so a misconfigured string
                // errors out before the long model-load path. `all` is a
                // no-op (matches the AtomicU32 default from `new()`); other
                // values override the policy bitmask.
                //
                // Resolves through `crate::backend::cuda` which aliases to
                // cuda_pc (feature = "cuda") or cuda_embedded (feature =
                // "cuda-embedded"); the two modules share the same
                // SyncPolicy API shape.
                #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
                {
                    use crate::backend::cuda::SyncPolicy;
                    let policy = SyncPolicy::parse(&args.cuda_sync_policy).map_err(|e| {
                        anyhow::anyhow!(
                            "--cuda-sync-policy: {e}. Valid: all | none | llamacpp | minimal | custom:<cats>"
                        )
                    })?;
                    gpu_concrete.set_sync_policy(policy);
                    if !args.cuda_sync_policy.eq_ignore_ascii_case("all") {
                        eprintln!(
                            "[CUDA] --cuda-sync-policy={} (mask=0x{:02x})",
                            args.cuda_sync_policy,
                            policy.raw()
                        );
                    }
                }
                // --cuda-weights-device: route weight uploads through a pure
                // device allocation (cuMemAlloc + explicit H2D). Must be set
                // before the model loader runs so every `copy_weight_from`
                // call sees the flag.
                #[cfg(feature = "cuda-embedded")]
                if args.cuda_weights_device {
                    if gpu_concrete.is_discrete_gpu() {
                        eprintln!(
                            "[CUDA] --cuda-weights-device ignored on discrete GPU (managed memory already migrates weights to VRAM)"
                        );
                    } else {
                        gpu_concrete.set_weights_device(true);
                        eprintln!(
                            "[CUDA] --cuda-weights-device enabled: weight tensors allocated via cuMemAlloc (device-only); activations/KV remain host-pinned"
                        );
                    }
                }
                // P4b: `--backend-cap <name>` resolves the CUDA quant-attn cap (KIVI `kivi_abi`)
                // from the force-linked static `CUDA_QUANT_ATTN_REGS`, built from this backend's live
                // CUcontext. Registered as `dyn CudaQuantAttnBackend` (a distinct TypeId from the
                // OpenCL `dyn QuantAttnBackend`, pulled separately by the quant-window setup). No
                // dlopen path — the CUDA axis is static-linkme only. Discrete-GPU (`cuda`) only —
                // matches the P3 score-axis scoping; cuda-embedded has no KV-plugin dispatch.
                // `mut` used only by the `cuda` register below; cuda-embedded leaves caps empty.
                #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
                let mut caps = CapabilityRegistry::new();
                #[cfg(feature = "cuda")]
                if let Some(name) = args.backend_cap.as_deref() {
                    let cap: Option<Arc<dyn argus_extension_api::CudaQuantAttnBackend>> =
                        gpu_concrete.with_cuda_quant_attn_make_args(|make_args| {
                            argus_extension_api::find_cuda_quant_attn(name)
                                .and_then(|reg| (reg.make)(make_args).ok())
                                .map(Arc::from)
                        });
                    let cap = cap.ok_or_else(|| {
                        anyhow::anyhow!(
                            "Unknown/failed --backend-cap '{name}' for CUDA (not in static \
                             CUDA_QUANT_ATTN_REGS, or its nvcc/PTX build failed)"
                        )
                    })?;
                    caps.register::<dyn argus_extension_api::CudaQuantAttnBackend>(cap);
                }
                // cuda-embedded: no CUDA KV-plugin dispatch. Fail loudly rather than silently
                // ignoring the flag, so the user isn't misled into thinking the cap is active.
                #[cfg(feature = "cuda-embedded")]
                if args.backend_cap.is_some() {
                    anyhow::bail!(
                        "--backend-cap is not supported on the cuda-embedded backend \
                         (CUDA KV-plugin dispatch is discrete-GPU / `cuda` only)"
                    );
                }
                let gpu: Arc<dyn Backend> = gpu_concrete;
                (
                    gpu.clone(),
                    gpu_mem.clone(),
                    Some(gpu),
                    Some(gpu_mem),
                    true,
                    caps,
                )
            }
            _ => anyhow::bail!(
                "Unknown backend: {}. Use cpu, opencl, or cuda.",
                args.backend
            ),
        };
        // --backend-cap is honored on OpenCL (dlopen QuantAttn cap) and CUDA (force-linked
        // CudaQuantAttn cap, P4b); the CPU arm registers an empty registry, so warn there.
        if args.backend_cap.is_some() && !matches!(args.backend.as_str(), "opencl" | "gpu" | "cuda")
        {
            eprintln!(
                "[init] --backend-cap '{}' ignored: only the OpenCL/CUDA backends expose a \
                 quant-window attention capability (backend = {})",
                args.backend_cap.as_deref().unwrap_or(""),
                args.backend
            );
        }
        // Phase α-W-4 §3.3: registry freeze. 이후 소비자는 read-only pull 만.
        let caps = Arc::new(caps);
        // cpu_backend_arc: always available for migration and SwitchHw fallback.
        let cpu_backend_arc: Arc<dyn Backend> = if args.backend == "cpu" {
            backend.clone()
        } else {
            Arc::new(CpuBackend::new())
        };
        let cpu_memory_arc: Arc<dyn Memory> = if args.backend == "cpu" {
            memory.clone()
        } else {
            Arc::new(Galloc::new())
        };
        let w_dtype = match args.weight_dtype.as_str() {
            "f16" => DType::F16,
            "q4" | "q4_0" => DType::Q4_0,
            _ => anyhow::bail!(
                "Unknown weight-dtype: {}. Use f16 or q4.",
                args.weight_dtype
            ),
        };
        eprintln!("[Config] Weight dtype: {:?}", w_dtype);
        // Validate --force-swap-ratio requires --secondary-gguf.
        if args.force_swap_ratio.is_some() && args.secondary_gguf.is_none() {
            anyhow::bail!(
                "--force-swap-ratio requires --secondary-gguf to be set (no secondary weight file)"
            );
        }

        // Parse --swap-algorithm (used by --qcf-dump warmup-swap path; U5 ablation).
        let swap_algorithm = crate::weight::SwapAlgorithm::from_cli(&args.swap_algorithm)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "--swap-algorithm: unknown value '{}'. Valid: imp, seq, rev, uni, anti",
                    args.swap_algorithm
                )
            })?;

        // Parse --importance-formula (§4 EuroSys'27 study). `compare` enables
        // three_way collector + post-warmup weight-perturbation proxy ε computation.
        let (importance_formula, importance_compare) = match args.importance_formula.as_str() {
            "mean_pool" => (crate::qcf_types::ImportanceFormula::MeanPool, false),
            "shortgpt_bi" => (crate::qcf_types::ImportanceFormula::PerTokenCosineBi, false),
            "dpllm_proxy" => (
                crate::qcf_types::ImportanceFormula::WeightPerturbProxy,
                false,
            ),
            "dpllm_multi" => (
                crate::qcf_types::ImportanceFormula::WeightPerturbMulti,
                false,
            ),
            "dpllm_abs" => (crate::qcf_types::ImportanceFormula::WeightPerturbAbs, false),
            "dpllm_qcf" => (crate::qcf_types::ImportanceFormula::WeightPerturbQcf, false),
            "direct_attn" => (crate::qcf_types::ImportanceFormula::DirectAttn, false),
            "compare" => (crate::qcf_types::ImportanceFormula::MeanPool, true),
            other => anyhow::bail!(
                "--importance-formula: unknown value '{}'. Valid: mean_pool, shortgpt_bi, dpllm_proxy, dpllm_multi, dpllm_abs, dpllm_qcf, direct_attn, compare",
                other
            ),
        };

        // Parse --swap-only-layers (§4 ground-truth study). CSV of layer indices.
        // Order is preserved (trajectory mode swaps in the listed order); duplicates
        // are dropped while keeping the first occurrence.
        let swap_only_layers: Option<Vec<usize>> = match args.swap_only_layers.as_deref() {
            None | Some("") => None,
            Some(csv) => {
                let mut v = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for tok in csv.split(',') {
                    let t = tok.trim();
                    if t.is_empty() {
                        continue;
                    }
                    let idx: usize = t.parse().map_err(|_| {
                        anyhow::anyhow!("--swap-only-layers: '{}' is not a non-negative integer", t)
                    })?;
                    if seen.insert(idx) {
                        v.push(idx);
                    }
                }
                Some(v)
            }
        };

        let primary_path_buf = std::path::PathBuf::from(model_path);
        let primary_format = crate::models::loader::detect_primary_format(&primary_path_buf)?;
        let is_gguf = matches!(primary_format, crate::models::loader::PrimaryFormat::Gguf);
        let is_auf = matches!(primary_format, crate::models::loader::PrimaryFormat::Auf);

        if args.secondary_gguf.is_some() {
            eprintln!(
                "[Deprecated] --secondary-gguf는 향후 제거됩니다. \
                 AUF single-file (--model-path foo.auf)로 전환하세요. \
                 기존 호출은 그대로 동작합니다."
            );
        }

        let mut model = if is_gguf || is_auf {
            if is_gguf && args.weight_dtype != "f16" {
                eprintln!("[Warning] --weight-dtype ignored for GGUF models (dtype from file)");
            }
            if is_auf && args.weight_dtype != "f16" {
                eprintln!(
                    "[Warning] --weight-dtype ignored for AUF models; \
                     use --primary-dtype <auto|f16|q4_0|...> instead"
                );
            }
            let load_cfg = crate::models::loader::LoadConfig {
                primary_source: primary_path_buf,
                primary_format,
                primary_variant_choice: args.primary_variant.into(),
                primary_dtype_choice: args.primary_dtype.into(),
                primary_eos_override: args.eos_token_id,
                default_dtype: w_dtype,
                secondary_source: args.secondary_gguf.clone(),
                disable_self_secondary: args.no_self_secondary,
                secondary_dtype_choice: args.secondary_dtype.into(),
                secondary_layout_choice: args.secondary_layout.into(),
            };
            TransformerModel::load_from_config(&load_cfg, backend.clone(), &*memory)?
        } else {
            TransformerModel::load_with_dtype(model_path, backend.clone(), &*memory, w_dtype)?
        };
        // T1: model weights loaded into memory (MmapBuffer + GPU copy if applicable).
        rss_trace("model_loaded");
        io_trace("model_loaded");
        // LLMRS_DUMP_SMAPS_T1: dump /proc/self/smaps at T1 for VMA analysis.
        // Tester pulls this file to analyse kgsl/ion/dmabuf VMA distribution.
        if std::env::var("LLMRS_DUMP_SMAPS_T1").is_ok() {
            dump_smaps("T1_model_loaded");
        }

        // WSWAP-6-PREFAULT: eager prefault of the secondary weight file.
        //
        // When --eager-prefault-secondary is set, touch all secondary weight pages
        // immediately after model load so that subsequent swap invocations hit the
        // page cache instead of incurring cold page faults (~328 ms on Galaxy S25,
        // §3.1 swap_overhead_s25.md). This is a one-time upfront cost traded for
        // per-swap latency elimination.
        //
        // Memory commit ≈ AUF/GGUF secondary size (e.g. 1.2 GB for Q4_0 1.5B).
        // Default OFF to protect memory-constrained environments.
        if args.eager_prefault_secondary {
            if let Some(ref secondary) = model.secondary_mmap {
                io_trace("before_prefault");
                let t0 = std::time::Instant::now();
                secondary.prefault();
                eprintln!(
                    "[Eager-Prefault] secondary weights prefaulted in {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
                io_trace("after_prefault");
            } else {
                eprintln!("[Eager-Prefault] no secondary configured, skipping");
            }
        }

        // When CPU primary + GPU secondary: migrate weights to GPU zero-copy memory.
        // Creates UnifiedBuffer (CL_MEM_ALLOC_HOST_PTR) + mapped: single VMA,
        // as_ptr() valid for CPU, cl_mem() valid for GPU.
        #[cfg(feature = "opencl")]
        if !is_gpu && let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc) {
            match model.migrate_weights_to_gpu(gpu_mem.as_ref(), gpu_be) {
                Ok(n) => eprintln!(
                    "[Backend] Migrated {} weight tensors to GPU zero-copy (ALLOC_HOST_PTR)",
                    n
                ),
                Err(e) => eprintln!("[Backend] Weight migration skipped: {}", e),
            }
        }

        // When GPU primary + resilience/partition enabled: ensure weights are CPU-accessible.
        // Maps UnifiedBuffer weights or reads device-only OpenCLBuffer into new UnifiedBuffer.
        // Single VMA (ALLOC_HOST_PTR) — no PSS double-counting on Adreno.
        //
        // Preload conditions (in order of cost/necessity):
        //   1. `--resilience-prealloc-switch`  → SwitchHw needs CPU-accessible weights
        //   2. `--tensor-partition <r>` where r is NOT the GPU-only fast-path ratio
        //      (r < GPU_ONLY_THRESHOLD and r > 0) → CPU matmul needs host pointers
        //   3. `--prefill-cpu-chunk-size > 0`  → CPU-side prefill chunk uses weights
        //
        // Notably `--enable-resilience` alone does NOT trigger preload. The
        // IPC directives that require CPU-accessible weights (`SetPartitionRatio`
        // with a non-GPU-only ratio, `SwitchHw` to CPU) now lazily invoke
        // `map_weights_for_cpu()` at the first activation point (see the
        // directive handler below). This avoids the ~200 ms startup cost and
        // the 400+ MB RSS uplift that hit every run which only enabled the
        // manager channel but never actually used CPU-side weight access.
        // §13.8-B B-1: cfg-gate-free wrapper (`map_weights_for_host_access`).
        // OpenCL 미빌드 또는 backend가 OpenCL 아닐 때 wrapper가 Ok(0) 반환.
        // tensor_partition helper는 layers::tensor_partition 모듈에 있어 opencl
        // off 시점에는 import 되지 않으므로 partition 조건만 cfg-gate 유지한다.
        #[cfg(feature = "opencl")]
        let cli_partition_needs_cpu_weights = args.tensor_partition > 0.0
            && !crate::layers::tensor_partition::is_gpu_only_ratio(args.tensor_partition);
        #[cfg(not(feature = "opencl"))]
        let cli_partition_needs_cpu_weights = false;
        if is_gpu
            && (args.resilience_prealloc_switch
                || cli_partition_needs_cpu_weights
                || args.prefill_cpu_chunk_size > 0)
        {
            match model.map_weights_for_host_access(&backend) {
                Ok(n) if n > 0 => eprintln!(
                    "[Backend] Mapped {} weight tensors for dual CPU/GPU access",
                    n
                ),
                Ok(_) => {} // No remap needed (already host-accessible / non-OpenCL backend).
                Err(e) => eprintln!("[Backend] Weight mapping failed (switch may crash): {}", e),
            }
        }

        // Sprint F/G-1-D (2026-04-26): one-shot lm_head Q4_0 load.
        //
        // Mode `auto` (default):
        //   1. AUF secondary with lm_head Q4_0 entry (capability bit 2 = 1)
        //      → zero-copy AUF path (~0 ms).
        //   2. AUF capability bit 2 = 0, or non-AUF secondary present
        //      → runtime quantize fallback (Sprint F, ~hundreds of ms).
        //   3. No secondary at all → skip (preserve legacy F16 behaviour).
        //
        // Mode `q4_0`: force runtime quantize regardless of AUF entry (debug).
        // Mode `none`/`off`: skip entirely (legacy F16).
        // LISWAP-PPL: PPL mode + --secondary-gguf with the default `auto` policy
        // would runtime-quantize lm_head F16 → Q4_0 (see `LmHeadAufResolution::
        // NotAuf` branch below). That diverges from a Q4-native GGUF baseline,
        // whose lm_head is loaded as F16 from the file. The result is a
        // systematic +~0.07 NLL gap that has nothing to do with the swap path
        // (root cause documented in `notes/handoff_liswap_ppl_lm_head_2026_05_12`).
        // For PPL measurements we silently switch the default to `none` so that
        // F16+swap and Q4-native are bit-identical. Power users can still force
        // the old behaviour with `--quantize-lm-head q4_0`.
        let qlm = {
            let raw = args.quantize_lm_head.to_ascii_lowercase();
            if args.ppl.is_some()
                && args.secondary_gguf.is_some()
                && (raw == "auto" || raw.is_empty())
            {
                eprintln!(
                    "[Notice] PPL mode + --secondary-gguf: auto-disabling lm_head Q4_0 \
                     quantization (would create a systematic +~0.07 NLL gap vs Q4-native \
                     baseline). Pass `--quantize-lm-head q4_0` to override."
                );
                "none".to_string()
            } else {
                raw
            }
        };
        match qlm.as_str() {
            "none" | "off" => {
                // F16 preserved — no action.
            }
            "auto" | "" => {
                if args.secondary_gguf.is_some() {
                    // Try AUF lm_head entry first.
                    // Note: payload.bytes borrows model.secondary_mmap (mmap lifetime).
                    // We extract the bytes into an owned Vec before calling
                    // load_lm_head_from_auf (which mutably borrows model) to satisfy
                    // the borrow checker.
                    let vocab_size = model.config.vocab_size;
                    let hidden_size = model.config.hidden_size;
                    // Resolve AUF lm_head payload: (bytes_owned, shape, variant_tag, is_none_ok)
                    // or None (GGUF secondary).
                    enum LmHeadAufResolution {
                        /// AUF entry found — owned bytes ready for load.
                        Found {
                            bytes: Vec<u8>,
                            shape: [usize; 2],
                            variant_tag: &'static str,
                        },
                        /// AUF present but no lm_head entry (bit 2 = 0).
                        AbsentFallback,
                        /// INV-135 violation.
                        Error(crate::auf::AufError),
                        /// Non-AUF secondary or no secondary.
                        NotAuf,
                    }
                    let resolution = {
                        match model
                            .secondary_mmap
                            .as_ref()
                            .and_then(|sm| sm.as_auf_view())
                            .map(|view| view.lm_head_q4_0_payload(vocab_size, hidden_size))
                        {
                            Some(Ok(Some(payload))) => LmHeadAufResolution::Found {
                                bytes: payload.bytes.to_vec(),
                                shape: payload.shape,
                                variant_tag: payload.variant_tag,
                            },
                            Some(Ok(None)) => LmHeadAufResolution::AbsentFallback,
                            Some(Err(e)) => LmHeadAufResolution::Error(e),
                            None => LmHeadAufResolution::NotAuf,
                        }
                    };

                    match resolution {
                        LmHeadAufResolution::Found {
                            bytes,
                            shape,
                            variant_tag,
                        } => {
                            // AUF path: lm_head Q4_0 entry found — load from owned bytes (~0 ms quantize).
                            eprintln!(
                                "[Backend] lm_head: loading from AUF Q4_0 entry (~0 ms quantize, variant={variant_tag})"
                            );
                            // Build a synthetic LmHeadPayload with owned bytes.
                            let payload = crate::auf::LmHeadPayload {
                                bytes: &bytes,
                                shape,
                                dtype: crate::auf::TensorDType::Q4_0,
                                alignment: 65536,
                                variant_tag,
                            };
                            model
                                .load_lm_head_from_auf(&payload, &backend)
                                .map_err(|e| {
                                    anyhow::anyhow!("--quantize-lm-head AUF load failed: {e}")
                                })?;
                        }
                        LmHeadAufResolution::AbsentFallback => {
                            // AUF present but no lm_head entry (bit 2 = 0, v0.1.0) → runtime fallback.
                            eprintln!(
                                "[Backend] lm_head: AUF entry absent (capability bit 2 = 0), runtime quantize"
                            );
                            let t_q = std::time::Instant::now();
                            match model.quantize_lm_head_to_q4_0(&backend) {
                                Ok(true) => eprintln!(
                                    "[Backend] Quantized lm_head → Q4_0 in {:.1} ms (mode=auto/runtime-fallback)",
                                    t_q.elapsed().as_secs_f64() * 1000.0,
                                ),
                                Ok(false) => {} // already Q4_0
                                Err(e) => {
                                    anyhow::bail!("--quantize-lm-head runtime fallback failed: {e}")
                                }
                            }
                        }
                        LmHeadAufResolution::Error(e) => {
                            // INV-135 violation (entry/dtype/shape mismatch) → fail-fast.
                            anyhow::bail!(
                                "--quantize-lm-head AUF invariant violation (INV-135): {e}"
                            );
                        }
                        LmHeadAufResolution::NotAuf => {
                            // Non-AUF secondary (GGUF) or no secondary at all → runtime quantize.
                            let t_q = std::time::Instant::now();
                            match model.quantize_lm_head_to_q4_0(&backend) {
                                Ok(true) => eprintln!(
                                    "[Backend] Quantized lm_head → Q4_0 in {:.1} ms (mode=auto)",
                                    t_q.elapsed().as_secs_f64() * 1000.0,
                                ),
                                Ok(false) => {} // already Q4_0
                                Err(e) => anyhow::bail!("--quantize-lm-head failed: {e}"),
                            }
                        }
                    }
                }
                // No secondary → skip (plain F16 run, legacy behaviour preserved).
            }
            "q4_0" | "q4" => {
                // Forced runtime quantize — AUF entry ignored (regression / debug mode).
                eprintln!(
                    "[Backend] lm_head: forced runtime quantize (AUF entry ignored, mode=q4_0)"
                );
                let t_q = std::time::Instant::now();
                match model.quantize_lm_head_to_q4_0(&backend) {
                    Ok(true) => eprintln!(
                        "[Backend] Quantized lm_head → Q4_0 in {:.1} ms (mode=q4_0)",
                        t_q.elapsed().as_secs_f64() * 1000.0,
                    ),
                    Ok(false) => eprintln!("[Backend] lm_head already Q4_0 — quantize skipped"),
                    Err(e) => anyhow::bail!("--quantize-lm-head failed: {e}"),
                }
            }
            other => anyhow::bail!(
                "Unknown --quantize-lm-head value: {}. Use 'auto', 'none', or 'q4_0'.",
                other
            ),
        }

        // CUDA: migrate weights to pinned host memory for cuBLAS access.
        // Unlike OpenCL (CL_MEM_USE_HOST_PTR zero-copy wrap), CUDA requires a memcpy into
        // cuMemHostAlloc'd buffers to get device pointers for cuBLAS.
        #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
        if args.backend == "cuda" {
            match model.migrate_weights_to_cuda(&backend) {
                Ok(n) => eprintln!(
                    "[Backend] Migrated {} weight tensors to CUDA pinned memory",
                    n
                ),
                Err(e) => eprintln!("[Backend] CUDA weight migration failed: {}", e),
            }
        }

        // Phase α-W-2: 흩어진 4 secondary Arc 를 단일 Hardware resolver 로 흡수.
        // primary `backend`/`memory` 는 위에서 이미 cpu/gpu 로컬의 clone 으로 set 되어
        // 있으므로, 4 Arc 를 move 하는 Hardware::new 호출이 안전하다.
        // 매핑: cpu=cpu_backend_arc / gpu=gpu_backend_arc / host=cpu_memory_arc /
        //       device=gpu_memory_arc (UMA/discrete 직교는 resolve 내부에서 처리).
        //
        // Phase α-W-5: tensor partition 이 `prepare_tensor_partition(.., &hardware)` 로
        // hardware resolver 를 직접 요구하므로 생성을 partition 블록 직전으로 끌어올린다.
        // 762~865 사이는 backend/model/args/is_gpu 만 by-ref 로 쓰고 4 Arc 를 by-value
        // consume 하지 않으므로 이동이 안전하다.
        let hardware = Arc::new(Hardware::new(
            cpu_backend_arc,
            gpu_backend_arc,
            None,
            cpu_memory_arc,
            gpu_memory_arc,
        ));

        // Tensor partition: split FFN gate/up weights for CPU-GPU cooperative inference.
        // Requires weights to be CPU-accessible (after map_weights_for_cpu).
        //
        // When `args.tensor_partition` is inside the GPU-only fast-path band
        // (>= GPU_ONLY_THRESHOLD), `prepare_tensor_partition` is a no-op and
        // leaves `partition_ctx = None`; forward() then takes the dense GPU
        // path. We still call it so the semantics (and the "Prepared 0" log)
        // are explicit.
        if args.tensor_partition > 0.0 && args.tensor_partition < 1.0 {
            match model.prepare_tensor_partition(args.tensor_partition, &hardware) {
                Ok(0) => eprintln!(
                    "[Partition] ratio={:.3} treated as GPU-only (>= {:.3}); partition path disabled",
                    args.tensor_partition,
                    crate::layers::tensor_partition::GPU_ONLY_THRESHOLD,
                ),
                Ok(n) => eprintln!(
                    "[Partition] Prepared {} weights with ratio {:.2}",
                    n, args.tensor_partition
                ),
                Err(e) => eprintln!("[Partition] Failed to prepare tensor partition: {}", e),
            }
        }

        // Q4_0 noshuffle SOA conversion: pre-convert all Q4_0 weights to Adreno-optimized
        // SOA layout. After this, matmul_q4_0 auto-dispatches to noshuffle GEMV for decode.
        // Check actual weight dtype (GGUF may load Q4_0 even when w_dtype=F16).
        //
        // LLMRS_SKIP_NOSHUFFLE_SOA: RSS diagnostic flag.
        // When set, skip SOA conversion entirely (registry stays empty).
        // matmul_q4_0() fallback path (engine/src/backend/opencl/mod.rs:1961):
        //   lookup_noshuffle_soa() returns None → standard Q4_0 GEMV kernel runs.
        // So decode still works, just slightly slower. RSS measurement is valid
        // for all tokens when this flag is set.
        #[cfg(feature = "opencl")]
        if is_gpu {
            let actual_q4 = w_dtype == DType::Q4_0
                || model
                    .layers
                    .first()
                    .is_some_and(|l| l.load_weights().wq.dtype() == DType::Q4_0);
            if actual_q4 {
                // Phase 4-4.10: noshuffle SOA conversion is now opt-in.
                //
                // Background: noshuffle SOA reclaims ≈702.8 MiB on Qwen 2.5-1.5B
                // Q4_0 but assumes GPU-only inference. Active weights converted to
                // SOA cannot back CPU fallback paths (switch_hw cpu, tensor
                // partition CPU split, prefill CPU chunking) without silent garbage.
                // It also relies on `kernel_gemv_noshuffle_q4_0` which currently
                // regresses Adreno decode TBT by +13% vs the standard Q4_0 GEMV
                // (Phase 4-4.8 measurement).
                //
                // Defaulting to AOS keeps the CPU fallback story intact and avoids
                // the noshuffle GEMV regression. Memory savings can be opted back
                // in once the `.cl` kernel is tuned for Adreno (backlog Path B).
                //
                // Env flags:
                //   LLMRS_ENABLE_NOSHUFFLE_SOA=1 → opt in to SOA conversion
                //   LLMRS_SKIP_NOSHUFFLE_SOA=1   → legacy override, always skip
                let soa_opt_in = std::env::var_os("LLMRS_ENABLE_NOSHUFFLE_SOA").is_some();
                let soa_skip = std::env::var_os("LLMRS_SKIP_NOSHUFFLE_SOA").is_some();
                if !soa_opt_in || soa_skip {
                    let reason = if soa_skip {
                        "LLMRS_SKIP_NOSHUFFLE_SOA set"
                    } else {
                        "default AOS (set LLMRS_ENABLE_NOSHUFFLE_SOA=1 to opt in)"
                    };
                    eprintln!(
                        "[NoShuffle] Skipped: {} — decode uses standard Q4_0 GEMV",
                        reason
                    );
                } else {
                    // Keep the AOS cl_mem alive when any runtime path still needs
                    // CPU-accessible weights: resilience pre-warm, a non-GPU-only
                    // tensor partition, prefill CPU chunking, or plain lazy
                    // activation via `--enable-resilience`. In those cases
                    // `map_weights_for_cpu()` will either have already run (lines
                    // ~988 above) or will run on demand against the original AOS
                    // allocation. Dropping it would strand the fallback path.
                    let keep_for_cpu = args.resilience_prealloc_switch
                        || cli_partition_needs_cpu_weights
                        || args.prefill_cpu_chunk_size > 0
                        || args.enable_resilience;
                    match model.prepare_noshuffle_buffers(&backend, keep_for_cpu) {
                        Ok(n) => {
                            eprintln!("[Backend] Noshuffle SOA prepared: {} weight tensors", n)
                        }
                        Err(e) => eprintln!("[Backend] Noshuffle preparation skipped: {}", e),
                    }
                    // WSWAP-5-TBT-DIAG: dump cl_mem footprint immediately after
                    // primary noshuffle prep so the Q4 baseline allocation
                    // pattern is recorded *before* any AUF SOA bypass swap path
                    // adds placeholder cl_mems on top.
                    #[cfg(feature = "opencl")]
                    // COLD-EXT: diagnostic dump after noshuffle prep (1회).
                    if let Some(ocl_be) = backend
                        .get_extension(crate::backend::EXT_OPENCL_QUEUE)
                        .and_then(|a| a.downcast_ref::<crate::backend::opencl::OpenCLBackend>())
                    {
                        ocl_be.dump_cl_mem_diagnostics(" stage=after_noshuffle_prep");
                    }
                }
            }
        }

        // Check if model weights are on GPU (cl_mem accessible) — needed for CPU→GPU switch.
        #[cfg(feature = "opencl")]
        let weights_on_gpu = {
            let layer0 = model.layers[0].load_weights();
            crate::backend::opencl::get_cl_mem(layer0.wq.buffer().as_ref()).is_ok()
        };
        #[cfg(not(feature = "opencl"))]
        let weights_on_gpu = false;

        // (Phase α-W-5) `hardware` 는 partition 블록 직전(~line 761)에서 이미 생성됨.

        Ok(SessionInitCtx {
            sampling_config,
            model_path: model_path.clone(),
            is_gguf,
            backend,
            memory,
            is_gpu,
            weights_on_gpu,
            hardware,
            caps,
            swap_algorithm,
            importance_formula,
            importance_compare,
            swap_only_layers,
            model,
        })
    }
}
