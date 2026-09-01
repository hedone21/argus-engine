//! argus-bench AB-1: resilience eviction 지원 [`DecodeLoop`] 조립자.
//!
//! [`build_standard_loop`](super::build_standard_loop) 와 동일한 ModelForward·
//! sampler·resilience 골격에, CLI `eviction <policy>` 로 구성한 [`CacheManager`]
//! 를 [`DecodeLoopBuilder::with_cache_manager`] 로 주입한다. decode 루프가
//! `plan.evict` (resilience KvEvict directive) 를 받으면 `forward.try_evict` 로
//! mid-decode prune 한다.
//!
//! happy path(AB-0 시나리오, `eviction=none`) 는 `cache_manager=None` 으로
//! 흘러 [`build_standard_loop`] 와 동등하게 동작한다.

use std::sync::{Arc, Mutex};

use anyhow::Result;

use argus_extension_api::{
    KVMutationStage, MutationPhase, StageCaps, StageParams, find_mutation_stage,
};

use crate::backend::Backend;
use crate::inference::sampling::SamplingConfig;
use crate::kv::cache_manager::CacheManager;
use crate::kv::eviction::EvictionPolicy;
use crate::kv::eviction::stage_registry::stage_default_protected_prefix;
use crate::kv::kv_cache::KVCache;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::resilience::sys_monitor::{LinuxSystemMonitor, NoOpMonitor};
use crate::session::cli::Args;
use crate::session::command_dispatcher::CommandDispatcher;
use crate::session::experiment::ScheduleCommandSource;
use crate::session::forward::ModelForward;
use crate::session::pipeline_registry::PipelineRegistry;
use crate::session::resilience_adapter::ResilienceAdapter;
use crate::session::{DecodeLoop, DecodeLoopBuilder, GreedySampler, RepetitionPenaltySampler};

/// The `(name, StageParams, owned extra-args)` for the configured `eviction <policy>` — shared by the
/// v2 [`build_resilience_cache_manager`] (which builds the v2 `KVMutationStage`) and the v3
/// [`resolve_mutation_driver`] (which builds the v3 `KVMutationStage`) so a migrated technique's v3
/// stage is constructed from byte-identical params to its v2 stage. The keep-set is then identical by
/// the Phase-1 decision-equivalence gate, making the driver a faithful replacement for `EvictionStage`.
fn eviction_policy_params(args: &Args) -> (String, StageParams, Vec<(String, String)>) {
    let policy_name = args.eviction_policy().to_string();
    let (params, extra) = stage_params_for(args, &policy_name);
    (policy_name, params, extra)
}

/// The same derivation for a technique named explicitly rather than by `eviction <policy>` — what
/// `--aperturb-select` needs, since each of its candidates declares its own protected-prefix
/// default and must be instantiated with that one, not with the configured policy's.
fn stage_params_for(args: &Args, policy_name: &str) -> (StageParams, Vec<(String, String)>) {
    // Score-based stages declare a protected-prefix (4 sinks); score-free ones declare 0 → protect 4
    // sinks by default. No per-name branch.
    let actual_protected_prefix = args.protected_prefix().unwrap_or_else(|| {
        match stage_default_protected_prefix(policy_name) {
            0 => 4,
            cap => cap,
        }
    });
    let streaming_window = if args.streaming_window() > 0 {
        args.streaming_window()
    } else if args.kv_budget() > 0 {
        args.kv_budget().saturating_sub(args.sink_size())
    } else {
        args.eviction_window()
    };
    let params = StageParams {
        eviction_window: args.eviction_window(),
        protected_prefix: actual_protected_prefix,
        keep_ratio: args.keep_ratio(),
        sink_size: args.sink_size(),
        streaming_window,
    };
    (params, args.stage_args())
}

/// A resolved v3 mutation-stage selection for the production driver: the stage instance plus the
/// (caps, phase) the driver needs at construction (the same `MutationStageReg` data the registry
/// carries pre-`make`). Returned by [`resolve_mutation_driver`].
pub struct MutationDriverSelection {
    pub stage: Box<dyn KVMutationStage>,
    pub caps: StageCaps,
    pub phase: MutationPhase,
}

/// Resolve the configured `eviction <policy>` to a native v3 [`KVMutationStage`] selection, or `None`
/// when the policy is `none` or has no v3 registration (a v2-only / dynamically-loaded `.so` stage —
/// the caller falls back to the v2 `EvictionStage` path). The v3 stage is built from the SAME
/// `eviction_policy_params` the v2 `CacheManager` uses, so the driver applies the identical keep-set.
pub fn resolve_mutation_driver(args: &Args) -> Option<MutationDriverSelection> {
    let (policy_name, params, extra_owned) = eviction_policy_params(args);
    if policy_name == "none" {
        return None;
    }
    let reg = find_mutation_stage(&policy_name)?;
    let extra: Vec<argus_extension_api::PluginArg> = extra_owned
        .iter()
        .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
        .collect();
    Some(MutationDriverSelection {
        stage: (reg.make)(params, &extra),
        caps: reg.caps,
        phase: reg.phase,
    })
}

/// Resolve `--aperturb-select` into the engine's own compression chooser, or `None` when the flag
/// is absent (every existing path).
///
/// Each name is resolved through the same registry `eviction <policy>` uses, and instantiated with
/// the same [`StageParams`] — a candidate must be the technique it would have been had it been
/// configured directly, or the choice is between things that do not exist. The output-projection
/// basis is loaded from `--aperturb-basis` when given, and factored from the weights otherwise.
pub fn resolve_aperturb_selector(
    args: &Args,
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
) -> Result<Option<Arc<crate::kv::aperturb_select::Selector>>> {
    use crate::kv::aperturb_select::{Candidate, Selector};
    if args.aperturb_select.is_empty() {
        return Ok(None);
    }
    crate::kv::eviction::stage_registry::ensure_builtin_stages_registered()?;
    let mut candidates = Vec::with_capacity(args.aperturb_select.len());
    for name in &args.aperturb_select {
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let reg = find_mutation_stage(name).ok_or_else(|| {
            anyhow::anyhow!(
                "--aperturb-select names '{name}', which is not a registered mutation stage. \
                 registered: {:?}",
                argus_extension_api::registered_mutation_names()
            )
        })?;
        anyhow::ensure!(
            !reg.caps.whole_model,
            "--aperturb-select: '{name}' is a whole-model stage; the planner drives one layer at a \
             time, so its cross-layer read would see nothing and its keep-set would not be the one \
             it really produces"
        );
        // A compression budget arrives mid-decode, so a candidate has to be a technique that acts
        // there. A `PrefillEnd` stage (PyramidKV/SnapKV) decides once, off the prefill attention it
        // is registered to read; asked at `KvMutate` it takes its own degraded fallback, and ranking
        // that would rank something the technique never does.
        anyhow::ensure!(
            reg.phase == MutationPhase::KvMutate,
            "--aperturb-select: '{name}' fires at {:?}, not KvMutate; a compression budget arrives \
             mid-decode and this technique does not act there",
            reg.phase
        );
        // The planner supplies the same reads a mid-decode mutation gets — importance, per-head
        // scores, last-step attention, and host-resident K/V. Anything else would reach the
        // candidate as `None` and send it down a fallback path.
        if let Some(kind) = reg.caps.reads.iter().find(|k| {
            matches!(
                k,
                argus_extension_api::TensorKind::PrefillAttention
                    | argus_extension_api::TensorKind::Query
                    | argus_extension_api::TensorKind::QueryStats
            )
        }) {
            anyhow::bail!(
                "--aperturb-select: '{name}' reads {kind:?}, which the mid-decode planner cannot \
                 supply; it would answer from its fallback path instead of from itself"
            );
        }
        // Each candidate gets the params it would have been configured with, including its own
        // declared protected-prefix default.
        let (params, extra_owned) = stage_params_for(args, name);
        let extra: Vec<argus_extension_api::PluginArg> = extra_owned
            .iter()
            .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
            .collect();
        candidates.push(Candidate::new(name, (reg.make)(params, &extra), reg.caps));
    }

    let (basis, residual_max) = crate::session::aperturb_basis::load_or_factor(
        model,
        backend,
        args.aperturb_basis.as_deref(),
        args.aperturb_basis_out.as_deref(),
    )?;
    eprintln!(
        "[aperturb-select] basis rank {} over {} layers (worst residual {:.1e})",
        basis.rank(),
        basis.n_layers(),
        residual_max,
    );
    Ok(Some(Arc::new(Selector::new(
        candidates,
        Arc::new(basis),
        model.config.num_attention_heads,
    )?)))
}

/// CLI `eviction <policy>` + `--swap-dir` 로 resilience-driven force eviction /
/// KvOffload 용 [`CacheManager`] 를 구성한다. `eviction=none` 이고 `--swap-dir`
/// 도 없으면 `None`.
///
/// - AB-1 eviction: score-free `force_evict`(verify eviction 시나리오는
///   `functional_only`) — [`AttentionScoreAccumulator`](crate::inference::attention_scores::AttentionScoreAccumulator)
///   미장착, heavy-hitter 는 score 부재 시 recency degrade(chat 과 동일).
/// - AB-3 KvOffload: `--swap-dir` 지정 시 `enable_swap` 으로 SwapHandler 등록.
///   eviction=none + swap-dir 만 있는 경우 [`NoEvictionPolicy`] CacheManager 위에
///   swap 만 활성(offload/recall directive 가 cm.offload/recall 호출).
///
/// d2o 는 layer-alloc/variance 머신을 요구하므로 AB-1 범위(non-layer-alloc)만 지원.
pub fn build_resilience_cache_manager(
    args: &Args,
    backend: &Arc<dyn Backend>,
) -> Result<Option<CacheManager>> {
    let swap_dir = args.swap_dir.clone();
    if args.eviction_policy() == "none" && swap_dir.is_none() {
        return Ok(None);
    }

    // The policy name + StageParams + extra-args, shared verbatim with the v3 `resolve_mutation_driver`
    // (protected-prefix / streaming-window derivation lives in the helper) so the v2 stage built here
    // and the v3 stage the driver builds receive byte-identical params.
    let (policy_name, params, extra_owned) = eviction_policy_params(args);

    let monitor: Box<dyn crate::resilience::sys_monitor::SystemMonitor> =
        if backend.is_discrete_gpu() {
            Box::new(NoOpMonitor)
        } else {
            Box::new(LinuxSystemMonitor)
        };
    let threshold_bytes = args.memory_threshold_mb() * 1024 * 1024;
    let target_ratio = args.eviction_target_ratio();

    // linkme fat-LTO 생존 self-test: 빌트인 stage 미등록 시 fail-fast.
    crate::kv::eviction::stage_registry::ensure_builtin_stages_registered()?;

    let mut cm = {
        // Every policy (none/sliding/streaming/h2o/h2o_plus/d2o/rkv) resolves through the plugin
        // registry by name (static linkme + dynamic --load-plugin); eviction=none + swap-dir (AB-3)
        // flows through make_stage("none") = a no-op stage. This site names no plugin.
        let policy: Box<dyn EvictionPolicy> = {
            // Technique-private knobs ride the opaque StageArgs blob (each plugin parses its own keys;
            // the engine knows none). Built generically by `Args::stage_args()`.
            let extra: Vec<argus_extension_api::PluginArg> = extra_owned
                .iter()
                .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
                .collect();
            crate::kv::eviction::stage_registry::make_stage_backed_policy(&policy_name, &params, &extra).ok_or_else(|| {
                anyhow::anyhow!(
                    "argus-bench: unknown eviction policy '{}'. Use: none, sliding, streaming, h2o, d2o{} (or --load-plugin <.so>).",
                    policy_name,
                    if cfg!(feature = "caote") { ", caote" } else { "" }
                )
            })?
        };
        CacheManager::new(policy, monitor, threshold_bytes, target_ratio)
    };

    if let Some(dir) = swap_dir {
        // legacy generate.rs:935 미러 — KvOffload directive 가 cm.offload 호출 시
        // SwapHandler 가 활성화되어 있어야 한다.
        eprintln!("[Resilience] KV swap enabled: dir={}", dir.display());
        cm.enable_swap(dir);
    }
    Ok(Some(cm))
}

/// β-5: argus-bench 용 memory-only [`LocalPressureSource`] 를 구성한다.
///
/// [`build_resilience_cache_manager`] 의 monitor/threshold 구성과 동일 의미(discrete GPU 면
/// `NoOpMonitor` = 압력 없음, 그 외 `LinuxSystemMonitor`). eviction/swap 도 resilience 도 없는
/// happy-path(`eviction=none` + swap-dir 없음 + resilience 없음)에서는 호출처가 `None` 으로 흘려
/// **무주입**한다 (per-token syscall 차단, G4). 본 함수는 source 객체만 만들고, 주입 여부는
/// 호출처(`experiment_run.rs`)가 결정한다.
pub fn build_local_pressure_source(
    args: &Args,
    backend: &Arc<dyn Backend>,
) -> Arc<dyn crate::pipeline::PressureSource> {
    let monitor: Arc<dyn crate::resilience::sys_monitor::SystemMonitor> =
        if backend.is_discrete_gpu() {
            Arc::new(NoOpMonitor)
        } else {
            Arc::new(LinuxSystemMonitor)
        };
    let threshold_bytes = args.memory_threshold_mb() * 1024 * 1024;
    Arc::new(crate::session::local_pressure::LocalPressureSource::new(
        monitor,
        threshold_bytes,
    ))
}

/// [`build_standard_loop`](super::build_standard_loop) 와 동일 골격 + resilience
/// eviction `CacheManager` 주입. `cache_manager=None` 이면 happy-path 와 동등.
///
/// `kv_caches`: `bin_setup`이 `--kv-format`/`--kv-type` dispatch로 이미 할당한
/// KV cache (typed 또는 opaque). builder는 재할당하지 않고 소비한다.
///
/// `schedule_source`: γ-3b experiment 모드용 `ScheduleCommandSource`. `resilience`
/// 와 상호 배타 — `resilience.is_some()` 이면 cmd_source 슬롯은 `ResilienceAdapter`
/// 가 점유하므로 `schedule_source` 는 무시된다. experiment 모드는 resilience=None.
#[allow(clippy::too_many_arguments)]
pub fn build_bench_loop(
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    // AB-4: PartitionStage 의 companion resolve 용 hardware (init.rs:822 보유분 전달).
    _hardware: Arc<crate::hardware::Hardware>,
    model: TransformerModel,
    kv_caches: Vec<KVCache>,
    max_seq_len: usize,
    sampling_config: SamplingConfig,
    plan_enabled: bool,
    resilience: Option<ResilienceAdapter>,
    cache_manager: Option<CacheManager>,
    // β-5: graded 압력 source (memory-only). None → 무주입(happy-path per-token syscall 0).
    pressure_source: Option<Arc<dyn crate::pipeline::PressureSource>>,
    // β-5: pressure-driven Persistent EvictionStage 의 force_evict target ratio
    // (CLI `--eviction-target-ratio` — CM 내부 값과 동일 출처를 호출자가 보장).
    pressure_evict_ratio: f32,
    // CLI `--protected-prefix` (keep-set attention-sink guard). `None` = user omitted → resolve to the
    // keep-set stage's `default_protected_prefix` (0 for kvpress-faithful pyramidkv).
    protected_prefix: Option<usize>,
    // γ-3b: 정적 directive schedule source. None → 무주입(bench/happy-path).
    schedule_source: Option<ScheduleCommandSource>,
    // AB-6 §5.6.7: WeightSwapStage 의 swap dispatch 설정 (CLI `--swap`/`--swap-phase-aware-*`
    // normalize 결과). secondary 보유 모델일 때만 `EngineSwapRuntime` 을 구성한다.
    // §5.9.1 Track A: score-based policy(h2o/h2o_plus/d2o) 시 호출자가 생성한 accumulator cell.
    // 비-score 조립처는 `Arc::new(Mutex::new(None))` 더미를 넘긴다.
    score_cell: Arc<Mutex<Option<crate::inference::signal_runtime::SignalRuntime>>>,
    // Faithful-H2O `(c)`: when true (eviction == "h2o"), arm a full-prompt-window PFA producer + the
    // prefill seed so the score accumulator reflects prefill attention, not just decode.
    faithful_h2o: bool,
    // The engine's own compression choice (`--aperturb-select`): a resolved candidate pool the
    // dispatcher hands each `KvCompress` budget to. `Some` arms the trailing query-row capture the
    // metric scores on and routes compressions through the selector instead of the CLI-configured
    // policy; `None` leaves the forward byte-identical and the method-drop path untouched.
    aperturb_selector: Option<Arc<crate::kv::aperturb_select::Selector>>,
) -> Result<DecodeLoop> {
    let vocab_size = model.config.vocab_size;
    // Captured before `model` is moved into `mf` — the PFA handle row count for the prefill keep-set.
    let n_heads_q = model.config.num_attention_heads;
    // decode loop가 실제로 쥐는 KV 저장 형태를 진입 시점에 보고
    // (build_standard_loop 와 동일 — alloc-시점 로그는 drop돼도 찍혀 증거 못 됨).
    let kv_is_opaque = kv_caches.first().is_some_and(|c| c.is_opaque());
    eprintln!(
        "[DecodeLoop] kv storage = {} (layers={}, cap={})",
        if kv_is_opaque {
            "OPAQUE (descriptor-driven)"
        } else {
            "typed"
        },
        kv_caches.len(),
        max_seq_len,
    );
    // AB-6: swap_backend resolve 용 — mf 가 `backend` 를 move 하므로 그 전에 Arc clone 보유.
    let backend_arc = Arc::clone(&backend);
    // The query-row ring is allocated from the same memory as the workspaces, so it is
    // device-resident wherever they are; `mf` moves both, so hold them first.
    let memory_arc = Arc::clone(&memory);
    let head_dim = model.config.head_dim;
    let n_layers_for_q = model.config.num_hidden_layers;
    // §5.9.2 Track B: ModelForward 와 WeightSwapStage(dispatcher 경유)가 공유할 hook cell 1개.
    // 양측에 Arc clone 으로 넘긴다. 초기값 None — IntraForward/LayerImmediate commit 이 Some 설치.
    let hook_cell: Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>> =
        Arc::new(Mutex::new(None));

    // §5.9.1 Track A: score_cell 은 호출자(argus_bench 진입부)가 score-based policy 여부를 판단해
    // 생성 후 전달한다. 비-score 조립처는 `Arc::new(Mutex::new(None))` 더미를 넘긴다.

    let mut mf = ModelForward::new(
        backend,
        memory,
        cpu_backend,
        Arc::new(model),
        kv_caches,
        max_seq_len,
        plan_enabled,
        Arc::clone(&hook_cell),
        Arc::clone(&score_cell),
    )?;
    // Faithful-H2O (c): arm a full-prompt-window (`usize::MAX` clamps to seq_len) PFA producer + the
    // prefill seed. The dummy cell is never consumed in the bench loop (no PrefillKeepSetStage); the
    // PFA buffer is used only to fold prefill column-sums into `score_cell` at the final chunk.
    if faithful_h2o {
        mf.set_prefill_attn(Arc::new(Mutex::new(None)), usize::MAX);
        mf.arm_faithful_prefill_seed();
    }

    // Direction-B unification: the caps-driven prefill-keepset arming — the SAME shared decision
    // `build_standard_loop` (argus-cli) uses. Arms a PFA producer so a per-head SnapKV/PyramidKV runs
    // faithfully in bench too (previously cli-only); the PrefillEnd consumer is submitted to the registry
    // below. Gated on `cache_manager.is_none()` ⇒ the eviction policy is "none" (the happy path), which
    // also excludes faithful_h2o (its h2o policy always builds a CacheManager). An explicit
    // `eviction <policy>` keeps its own KvMutate path — no double eviction. `None` (no PFA-reading stage
    // linked) leaves this dormant = byte-identical.
    let keepset_arming = (cache_manager.is_none() && !faithful_h2o)
        .then(crate::kv::eviction::stage_registry::resolve_prefill_keepset_arming)
        .flatten();
    let pfa_cell: Arc<Mutex<Option<Vec<Vec<f32>>>>> = Arc::new(Mutex::new(None));
    if let Some(arming) = &keepset_arming {
        mf.set_prefill_attn(pfa_cell.clone(), arming.q_window);
        eprintln!(
            "[prefill-keepset] '{}' active — PFA producer arms q_window={} \
             (SnapKV per-head keep-set staged at PrefillEnd)",
            arming.stage_name, arming.q_window
        );
    }

    // The engine's own compression choice: arm the trailing query-row capture. It is the one thing
    // the metric cannot recompute — the rows are what the forward already produced, and reusing
    // them is why a decision costs no forward pass. Unarmed (the default) both forks lend `None`
    // and the forward is byte-identical.
    let q_rows_cell: Arc<Mutex<Option<crate::inference::q_rows::QRowCapture>>> =
        Arc::new(Mutex::new(None));
    if aperturb_selector.is_some() {
        let cap = crate::inference::q_rows::QRowCapture::new(
            Arc::clone(&backend_arc),
            memory_arc.as_ref(),
            n_layers_for_q,
            crate::session::aperturb_basis::APERTURB_ROWS,
            n_heads_q * head_dim,
        )?;
        *q_rows_cell.lock().expect("q-rows cell poisoned") = Some(cap);
        mf.set_q_rows(Arc::clone(&q_rows_cell));
    }

    // β-3: pos-환류용 layer-0 fmt handle (§5.2.1 (가)). coercion: Arc<StandardFormat> →
    // Arc<dyn KVCacheFormat>.
    let kv_pos_handle: Option<Arc<dyn crate::format::KVCacheFormat>> = mf
        .fmt_caches()
        .first()
        .map(|f| f.clone() as Arc<dyn crate::format::KVCacheFormat>);

    // β-4: EvictionStage 가 prune 할 전체 layer handle (W1 — enumerate 순서 == layer idx).
    let kv_handles: Vec<Arc<crate::kv::standard_format::StandardFormat>> = mf.fmt_caches().to_vec();

    // AB-4: PartitionStage 가 re-slice 할 전체 layer slot handle (model.layers.clone()).

    // AB-6 §5.6.3/§5.6.7: WeightSwapStage 가 swap 할 model handle (register 시점 보유,
    // model 측 접근 seam — secondary_mmap/quant_noise/current_dtype). swap_runtime 은 아래에서
    // secondary 보유 시에만 greenfield 구성한다.
    let swap_model: Arc<TransformerModel> = Arc::clone(mf.model());
    let _has_secondary = swap_model.secondary_mmap.is_some();

    // β-4 (매핑 문서 4부): resilience adapter 에 held-handle 주입 → heartbeat snapshot 의
    // kv_cache_tokens/capacity 를 layer-0 handle 에서 query (poll 인자 제거 대체).
    let resilience = match (resilience, kv_pos_handle.clone()) {
        (Some(mut adapter), Some(h)) => {
            adapter.set_kv_handle(h);
            adapter.set_kv_byte_handles(
                kv_handles
                    .iter()
                    .map(|h| {
                        h.clone() as Arc<dyn crate::session::resilience_adapter::KvBytesHandle>
                    })
                    .collect(),
            );
            Some(adapter)
        }
        (other, _) => other,
    };

    // β-4: dispatcher 와 driver 가 공유하는 registry. dispatcher.submit(OneShot EvictionStage) →
    // driver 의 KvMutate dispatch(β-3 배선)가 소비.
    let registry = Arc::new(PipelineRegistry::new());

    // Direction-B: the PrefillEnd keep-set consumer (pyramidkv) — submitted to the SAME registry as the
    // pressure-driven persistent EvictionStage below (phase-disjoint: PrefillEnd vs KvMutate → submit
    // order immaterial). Mirrors `build_standard_loop`. Reads the per-layer PFA the armed `mf` fills and
    // applies the SnapKV per-head keep-set via the shared `apply_prefill_keepset` executor.
    if let Some(arming) = &keepset_arming
        && let Some(stage) =
            crate::kv::eviction::stage_registry::make_prefill_keepset_stage(&arming.stage_name)
    {
        registry.submit(Arc::new(
            crate::stages::kv::prefill_keepset::PrefillKeepSetStage::new(
                kv_handles.clone(),
                stage,
                pfa_cell.clone(),
                n_heads_q,
                pressure_evict_ratio,
                protected_prefix.unwrap_or(arming.default_protected_prefix),
            ),
        ));
    }

    // β-4: EngineCommand 분배자. **resilience-on 이면 CM 유무와 무관하게 구성** — control
    // 디렉티브(Throttle/SetTargetTbt/Suspend 등)는 CM 없이 소비 가능하고, v1 도 eviction=none +
    // resilience-on 에서 control 을 적용했다(미구성 시 디렉티브 무소비 드롭 = v1 회귀, β-4 device
    // smoke 실증 2026-06-10). evict 디렉티브는 CM=None 이면 dispatcher 내부에서 inert —
    // v1 (a.5) 의 `cache_manager=None` 스킵과 등가. 둘 다 None 이면 미구성(happy-path 거동-0).
    // β-5: CM 을 Arc<Mutex> 로 한 번 들어 dispatcher(OneShot 구성)와 Persistent stage 가 공유.
    let shared_cm = cache_manager.map(|cm| Arc::new(Mutex::new(cm)));

    // γ-3b: schedule_source 가 있어도 dispatcher 를 구성해야 evict directive 가 OneShot
    // EvictionStage 로 submit 된다 (설계 §13.4 "schedule.is_some() OR 추가").
    let dispatcher = if resilience.is_some() || shared_cm.is_some() || schedule_source.is_some() {
        let dispatcher = CommandDispatcher::new(
            Arc::clone(&registry),
            kv_handles.clone(),
            shared_cm.clone(),
            // §5.9.1 Track A: ModelForward + EvictionStage 와 공유하는 score cell.
            Arc::clone(&score_cell),
        );
        // bench GPU-score 경로: dispatcher 가 submit 하는 score-fed EvictionStage 에 decode 와 동일한
        // backend Arc(= 동일 OpenCL = 동일 GPU score buffer)를 넘긴다. `gpu_score_active` 면 run_eviction
        // 이 score 읽기 직전 GPU→CPU sync. backend_arc 는 mf 가 move 한 backend 의 clone(build 진입부
        // 보유). non-opencl 이면 sync 가 no-op.
        let dispatcher = dispatcher.with_backend(Some(Arc::clone(&backend_arc)));
        // A configured pool makes the engine choose; without one the dispatcher keeps method-drop.
        Some(match aperturb_selector.as_ref() {
            Some(sel) => {
                eprintln!(
                    "[aperturb-select] active — candidate pool [{}], {} query rows per decision",
                    sel.names().join(", "),
                    crate::session::aperturb_basis::APERTURB_ROWS,
                );
                dispatcher.with_aperturb_selector(
                    Arc::clone(sel),
                    Arc::clone(&q_rows_cell),
                    Arc::clone(&pfa_cell),
                )
            }
            None => dispatcher,
        })
    } else {
        if aperturb_selector.is_some() {
            // Nothing polls commands, so no `KvCompress` can arrive and the pool would never be
            // asked. Say so: a silent selector reads exactly like one that was asked and declined.
            eprintln!(
                "[aperturb-select] inert — no command source is configured (resilience off and no \
                 schedule), so no compression budget can reach the engine"
            );
        }
        None
    };

    // β-5: pressure-driven Persistent EvictionStage — CM + graded source 가 둘 다 있을 때만
    // 상주 등록. band ≥ Warning 상향 에지에서 에피소드당 1회 force_evict (stage 내부
    // edge-trigger). source 부재(None) 면 StepInfo.pressure 가 항상 0(Normal) → 등록해도
    // 영구 무발화이므로 미등록 (의도 명시). ratio = CLI `--eviction-target-ratio`
    // (method-drop 시맨틱과 동일하게 정책은 CM 의 CLI 구성).
    if let (Some(cm), Some(_)) = (&shared_cm, &pressure_source) {
        let persistent = crate::stages::kv::eviction::EvictionStage::persistent(
            kv_handles,
            Arc::clone(cm),
            pressure_evict_ratio,
            crate::kv::PressureLevel::Warning,
        );
        registry.submit(Arc::new(persistent));
    }

    let use_stateful =
        sampling_config.repetition_penalty != 1.0 || sampling_config.temperature != 0.0;
    let builder = DecodeLoopBuilder::new()
        .with_forward(mf)
        .with_pipeline(Arc::clone(&registry));
    let builder = match kv_pos_handle {
        Some(h) => builder.with_kv_pos_handle(h),
        None => builder,
    };
    let builder = if use_stateful {
        builder.with_sampler(RepetitionPenaltySampler::new(sampling_config, vocab_size))
    } else {
        builder.with_sampler(GreedySampler)
    };
    let builder = match resilience {
        Some(adapter) => builder.with_resilience(adapter),
        None => match schedule_source {
            // γ-3b: resilience 없을 때만 schedule cmd_source 주입.
            Some(scs) => builder.with_cmd_source(scs),
            None => builder,
        },
    };
    let builder = match dispatcher {
        Some(d) => builder.with_command_dispatcher(d),
        None => builder,
    };
    let builder = match pressure_source {
        Some(s) => builder.with_pressure_source(s),
        None => builder,
    };
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// P0-6 selector: a v3-native built-in resolves to a driver selection; `none` and a non-v3
    /// (unknown / `.so`) name return `None` so the caller keeps the v2 `EvictionStage` fallback.
    #[test]
    fn resolve_mutation_driver_selects_v3_for_native_builtin() {
        // `eviction` 미지정(=none) → 드라이버 없음.
        let args = Args::parse_from(["argus_engine"]);
        assert!(resolve_mutation_driver(&args).is_none(), "none → no driver");

        // sliding 은 Phase-1 v3-native 빌트인 → KvMutate 에서 드라이버 선택.
        let args = Args::parse_from(["argus_engine", "eviction", "plugin", "--name", "sliding"]);
        let sel = resolve_mutation_driver(&args).expect("sliding registers a v3 KVMutationStage");
        assert_eq!(sel.phase, MutationPhase::KvMutate);
        assert_eq!(sel.stage.name(), "sliding");

        // 미등록(비-v3 / `.so`) 이름 → None → 호출처가 v2 EvictionStage 로 폴백.
        let args = Args::parse_from([
            "argus_engine",
            "eviction",
            "plugin",
            "--name",
            "definitely_not_a_stage",
        ]);
        assert!(
            resolve_mutation_driver(&args).is_none(),
            "non-v3 name → v2 fallback"
        );
    }
}
