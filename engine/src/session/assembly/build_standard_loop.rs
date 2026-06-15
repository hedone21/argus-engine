//! Phase 4-4-a: standard happy path용 [`DecodeLoop`] 조립자.
//!
//! [`build_standard_loop`]는 unpack된 args (backend / memory / cpu_backend /
//! model) 를 받아 표준 `KVCache` 기반 [`ModelForward`] + greedy sampler
//! [`DecodeLoop`]을 반환한다. `model`은 owned consume — `Arc::new(model)`로
//! 1회 변환하여 [`ModelForward`]에 위임 (Q2-B 결정의 변형 α: ctx struct
//! 의존 없이 unpack-args 시그니처).
//!
//! ## 왜 ctx-consume이 아닌 unpack-args인가
//!
//! `bin/generate.rs` line 91~109에서 `SessionInitCtx::build(&args)?` 직후
//! 즉시 ctx unpack이 발생한다. 따라서 표준 path 진입 시점 (line 3032)에
//! ctx struct는 더 이상 존재하지 않으며, unpack된 `backend` / `memory` /
//! `model` 변수만 사용 가능하다. ctx struct를 보존하려면 모든 분기
//! (chat/eval-ll/ppl/batch)에 ctx-borrow 패턴을 적용해야 하므로 4-4 범위
//! 초과. → 헬퍼 시그니처를 unpack-args 형태로 한정하여 영향 범위를
//! standard path 한정으로 유지.
//!
//! Happy path 진입 조건은 [`is_standard_happy_path`] 참조. chunked prefill /
//! optional collector 의존 케이스는 Phase 4-4.5에서 흡수 예정.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use argus_shared::Level;

use crate::backend::Backend;
use crate::format::KVCacheFormat;
use crate::inference::sampling::SamplingConfig;
use crate::kv::cache_manager::CacheManager;
use crate::kv::kv_cache::KVCache;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::pipeline::PressureSource;
use crate::session::cli::Args;
use crate::session::command_dispatcher::CommandDispatcher;
use crate::session::forward::ModelForward;
use crate::session::local_pressure::KvFillPressureSource;
use crate::session::pipeline_registry::PipelineRegistry;
use crate::session::resilience_adapter::ResilienceAdapter;
use crate::session::{DecodeLoop, DecodeLoopBuilder, GreedySampler, RepetitionPenaltySampler};
use crate::stages::kv::eviction::EvictionStage;

/// argus-cli score-free eviction 의 KV-fill high-water (점유율 percent). `pos >= 85% * max_seq_len`
/// 에서 [`KvFillPressureSource`] 가 Warning 밴드를 보고해 Persistent EvictionStage 를 1회 발화시킨다.
/// 15% headroom 은 decode loop 의 8-step pressure 캐시 지연(`PRESSURE_QUERY_INTERVAL`)을 덮는다.
const KV_FILL_HIGH_WATER_PCT: u32 = 85;

/// Phase 4-4-a: standard generate happy path 진입 가드.
///
/// 다음을 모두 만족할 때만 `true`. 미통과 args는 generate.rs 기존 path 사용.
///
/// - `args.qcf_dump.is_none()`               — `--qcf-dump` 비활성 (importance_collector 미장착)
/// - `args.skip_ratio.unwrap_or(0.0) == 0.0` — `--skip-ratio=0` (skip_config 미장착)
/// - `!args.d2o_layer_alloc()`                 — `--d2o-layer-alloc` 비활성 (variance_collector 미장착)
/// - `!args.profile && !args.profile_events` — profile 비활성 (profiler 미장착)
/// - `args.eviction_policy() == "none"`        — eviction 비활성 (score_accumulator 미장착)
/// - `args.tensor_partition == 0.0`          — Phase 4-4.7: tensor_partition 활성 시
///   plan path가 build_plan에서 None을 반환 → sticky_disabled lock-out → 매 step
///   forward_into fallback이라 성능 저하. happy path에서는 partition 차단.
/// - `!args.swap_intra_forward && !args.swap_layer_immediate && !args.swap_phase_aware`
///   — Phase 4-4.7: weight swap intra-forward / phase-aware는 plan path가 미지원
///   (production generate.rs l.4192-4199 가드와 동치).
///
/// Phase 4-4.7에서 `repetition_penalty == 1.0` 가드가 제거되었다. 대신
/// [`build_standard_loop`]가 `sampling_config`에 따라
/// [`GreedySampler`] 또는 [`RepetitionPenaltySampler`] 중 적절한 sampler를
/// 자동 선택하여 production `sampling::sample` 호출과 paradigm equivalent
/// 결과를 보장한다.
///
/// 호출자는 추가로 `prompt_len <= MAX_NON_CHUNKED_PREFILL_LEN`도 검증해야 한다
/// (chunked prefill 미지원). 그 가드는 generate.rs 호출 site에서 처리.
pub fn is_standard_happy_path(args: &Args) -> bool {
    args.qcf_dump.is_none()
        && args.skip_ratio.unwrap_or(0.0) == 0.0
        && !args.d2o_layer_alloc()
        && !args.profile
        && !args.profile_events
        && args.eviction_policy() == "none"
        && args.tensor_partition == 0.0
        && !args.swap_intra_forward
        && !args.swap_layer_immediate
        && !args.swap_phase_aware
}

/// Phase 4-4-a: unpack-args 형태로 standard `DecodeLoop` 조립.
///
/// **model consume 패턴 (Q2-B α변형)**: `model: TransformerModel` owned 인자를
/// `Arc::new(model)`로 1회 변환하여 [`ModelForward`]에 위임. 호출자
/// (generate.rs main)는 본 헬퍼 호출 후 `model` 변수를 다시 사용할 수 없다.
/// chat/eval-ll/ppl/batch는 early-return 구조이므로 표준 path 진입 시점에
/// 다른 분기로 흐를 가능성 없음 (자연스러운 모순 없음).
///
/// - `kv_caches`: `bin_setup`이 `--kv-format`/`--kv-type` dispatch로 이미 할당한
///   KV cache (typed 또는 opaque). builder는 재할당하지 않고 소비한다 —
///   과거에는 builder가 `alloc_standard_kv_caches`로 typed를 재할당하여 `ctx`의
///   opaque 선택을 덮어썼다(`--kv-format`이 decode 경로에 도달 못 함).
/// - `max_seq_len`: lazy `PrefillWorkspace` cap
/// - `resilience`: P3.3 — `Some(adapter)` 이면 3 slot에 주입, `None` 이면 NoOp default.
#[allow(clippy::too_many_arguments)]
pub fn build_standard_loop(
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    model: TransformerModel,
    kv_caches: Vec<KVCache>,
    max_seq_len: usize,
    sampling_config: SamplingConfig,
    plan_enabled: bool,
    resilience: Option<ResilienceAdapter>,
    // 선택적 read stage 이름(`--read-stage`). None = full read(현행 byte-identical).
    read_stage_name: Option<&str>,
    // score-free eviction `CacheManager`(sliding/streaming/`--load-plugin` stage). `None` =
    // 순수 happy-path(eviction none) — 아래 eviction 배선 전체 미진입 = 기존과 byte-identical.
    cache_manager: Option<CacheManager>,
    // force_evict target ratio(CLI `--eviction-target-ratio`). `cache_manager=None` 이면 무시.
    eviction_target_ratio: f32,
) -> Result<DecodeLoop> {
    let vocab_size = model.config.vocab_size;
    // decode loop가 실제로 쥐는 KV 저장 형태를 진입 시점에 보고한다.
    // bin_setup의 alloc-시점 "KV format" 로그는 caches가 drop돼도 찍히므로 증거가
    // 못 된다(과거 false-positive e2e의 원인). ModelForward가 소비하기 직전의
    // 이 identity가 진짜 decode 경로 증거다.
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
    // §5.9.2 Track B: happy/standard 경로는 swap 미구성(dispatcher model=None) → hook 설치가
    // 일어나지 않는다. ModelForward 생성자가 요구하는 cell 은 항상 None 인 더미를 넘긴다.
    let hook_cell: Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>> =
        Arc::new(Mutex::new(None));
    // §5.9.1 Track A: happy/standard 경로는 score-based eviction 미구성 → 더미 None cell.
    let score_cell: Arc<
        Mutex<Option<crate::inference::attention_scores::AttentionScoreAccumulator>>,
    > = Arc::new(Mutex::new(None));
    let mut mf = ModelForward::new(
        backend,
        memory,
        cpu_backend,
        Arc::new(model),
        kv_caches,
        max_seq_len,
        plan_enabled,
        Arc::clone(&hook_cell),
        score_cell,
    )?;

    // 선택적 read stage 주입. 미지정(None)이면 미진입 = read_stage 슬롯 None 유지
    // (full read, INV-147 byte-identical). 모르는 이름이면 등록 목록과 함께 에러.
    if let Some(name) = read_stage_name {
        // fat-LTO --gc-sections silent drop fail-fast. 빌트인 Quest 누락 시 즉시 Err.
        crate::kv::read::read_stage_registry::ensure_builtin_read_stages_registered()?;
        match technique_api::find_read_stage(name) {
            Some(reg) => {
                // standard happy path 의 활성 format 은 StandardFormat = SelectiveRead 지원이라
                // 폴백 경고가 발생하지 않는다. (미지원 format 폴백 경고는 transformer.rs seam 의
                // as_selective_read()==None 자동 처리 — opaque/KIVI 진입 시.)
                let stage = (reg.make)(technique_api::ReadStageParams::default());
                eprintln!(
                    "[read-stage] '{name}' active — read_plan called per layer right before decode attention"
                );
                mf.set_read_stage(stage);
            }
            None => {
                let names = technique_api::registered_read_names();
                anyhow::bail!("unknown --read-stage '{name}'. registered read stages: {names:?}");
            }
        }
    }

    // KV format handles — eviction stage prune 대상 + resilience heartbeat + eviction 후
    // pos-환류(with_kv_pos_handle). mf 가 with_forward 로 move 되기 전에 읽는다.
    let kv_handles: Vec<Arc<crate::kv::standard_format::StandardFormat>> = mf.fmt_caches().to_vec();
    let kv_pos_handle: Option<Arc<dyn KVCacheFormat>> = mf
        .fmt_caches()
        .first()
        .map(|f| f.clone() as Arc<dyn KVCacheFormat>);

    // eviction(CM) 또는 resilience 가 있을 때만 registry 를 만든다. 둘 다 없으면(순수 happy-path,
    // eviction none + no-resilience) registry/dispatcher/pos-handle/pressure 미배선 = 기존과
    // byte-identical 조립(INV: 회귀 0).
    let needs_registry = resilience.is_some() || cache_manager.is_some();
    let registry = needs_registry.then(|| Arc::new(PipelineRegistry::new()));

    // score-free eviction(sliding/streaming/`--load-plugin` stage): KV-fill 압력 구동 Persistent
    // EvictionStage + [`KvFillPressureSource`]. chat 의 turn-경계 evict 와 달리 단일 프롬프트는 turn
    // 경계가 없으므로, KV 점유율이 high-water 를 넘으면 Warning 밴드를 보고하는 source 로 stage 를
    // 구동한다(메모리 압력·manager 무관 — free-RAM 머신에서도 발화). score-based(h2o/d2o)는
    // accumulator 가 필요해 argus-cli 진입부에서 reject — 여기 도달하는 CM 은 항상 score-free.
    let pressure_source: Option<Arc<dyn PressureSource>> = match cache_manager {
        Some(cm) => {
            let registry = registry
                .as_ref()
                .expect("registry: cache_manager.is_some()");
            let shared_cm = Arc::new(Mutex::new(cm));
            registry.submit(Arc::new(EvictionStage::persistent(
                kv_handles.clone(),
                shared_cm,
                eviction_target_ratio,
                Level::Warning,
            )));
            kv_pos_handle.clone().map(|h| {
                Arc::new(KvFillPressureSource::new(
                    h,
                    max_seq_len,
                    KV_FILL_HIGH_WATER_PCT,
                )) as Arc<dyn PressureSource>
            })
        }
        None => None,
    };

    // β-4: resilience-on 이면 dispatcher 를 구성한다 — control 디렉티브(Throttle/SetTargetTbt/
    // Suspend 등)는 CM 없이 소비 가능하고, v1 은 argus_cli resilience-on 에서 이를 적용했다
    // (dispatcher 부재 시 디렉티브 무소비 드롭 = v1 회귀, β-4 device smoke 실증 2026-06-10).
    // dispatcher 의 cache_manager 는 None(inert): argus-cli eviction 은 위 KV-fill Persistent
    // stage 가 구동하므로 manager evict 디렉티브 경로는 사용하지 않는다(v1 (a.5) 스킵 등가).
    // heartbeat kv snapshot 은 held-handle query — layer-0 handle 주입.
    let (resilience, dispatcher) = match resilience {
        Some(mut adapter) => {
            if let Some(h) = kv_pos_handle.clone() {
                adapter.set_kv_handle(h);
            }
            let registry = registry.as_ref().expect("registry: resilience.is_some()");
            // happy/chat 경로는 partition/swap/quant 미구성 (빈 slots + None hardware/model/
            // swap_runtime + 빈 kivi_handles). report_tx=None (AB-5: happy-path resilience 미배선).
            let dispatcher = CommandDispatcher::new(
                Arc::clone(registry),
                kv_handles.clone(),
                None,
                Vec::new(),
                None,
                None,
                None,
                None,
                Vec::new(),
                None, // report_tx: AB-5
                // §5.9.2 Track B: happy 경로는 swap 미구성 → 더미 cell (항상 None).
                Arc::clone(&hook_cell),
                // §5.9.1 Track A: happy 경로는 score-based eviction 미구성 → 더미 None cell.
                Arc::new(Mutex::new(None)),
            );
            (Some(adapter), Some(dispatcher))
        }
        None => (None, None),
    };

    // Phase 4-4.7: sampler 자동 선택. production `sampling::sample`은
    // temperature=0 + repetition_penalty=1.0이면 raw argmax와 동치이므로 두 조건이
    // 모두 만족될 때만 GreedySampler 사용. 그 외는 RepetitionPenaltySampler가
    // 내부 VecDeque ring buffer + scratch logits로 production 호출을 충실히 모사.
    let use_stateful =
        sampling_config.repetition_penalty != 1.0 || sampling_config.temperature != 0.0;
    let mut builder = DecodeLoopBuilder::new().with_forward(mf);
    builder = if use_stateful {
        builder.with_sampler(RepetitionPenaltySampler::new(sampling_config, vocab_size))
    } else {
        builder.with_sampler(GreedySampler)
    };
    // P3.3: resilience adapter 주입 (Some → 3 slot 주입, None → NoOp default 유지)
    if let Some(adapter) = resilience {
        builder = builder.with_resilience(adapter);
    }
    // β-4/eviction: 공유 registry + dispatcher(resilience-on) + pos-환류 handle + KV-fill pressure
    // 배선. needs_registry 일 때만 진입 — 순수 happy-path 는 미배선(기존과 byte-identical).
    if let Some(registry) = registry {
        builder = builder.with_pipeline(registry);
        if let Some(dispatcher) = dispatcher {
            builder = builder.with_command_dispatcher(dispatcher);
        }
        if let Some(h) = kv_pos_handle {
            builder = builder.with_kv_pos_handle(h);
        }
        if let Some(src) = pressure_source {
            builder = builder.with_pressure_source(src);
        }
    }
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// `Args::parse_from(["argus_engine"])` — clap의 default_value_t 적용된 baseline.
    /// `Default` derive 미보유 → parse_from으로 default 인스턴스 생성.
    fn default_args() -> Args {
        Args::parse_from(["argus_engine"])
    }

    #[test]
    fn happy_path_with_no_repetition_penalty() {
        let mut args = default_args();
        args.repetition_penalty = 1.0;
        assert!(
            is_standard_happy_path(&args),
            "기본 args + --repetition-penalty 1.0 → happy path 진입"
        );
    }

    /// Phase 4-4.7: rep_penalty 가드가 제거됨. 기본 CLI (default 1.1) 도
    /// happy path 진입 가능 — `build_standard_loop`가 `RepetitionPenaltySampler`
    /// 를 자동 선택하여 production `sampling::sample` 호출과 동치 결과를 낸다.
    #[test]
    fn accepts_default_repetition_penalty() {
        let args = default_args();
        assert!(
            is_standard_happy_path(&args),
            "Phase 4-4.7: default repetition_penalty=1.1 happy path 진입 허용"
        );
    }

    #[test]
    fn rejects_skip_ratio() {
        let mut args = default_args();
        args.skip_ratio = Some(0.1);
        assert!(!is_standard_happy_path(&args));
    }

    #[test]
    fn rejects_profile() {
        let mut args = default_args();
        args.profile = true;
        assert!(!is_standard_happy_path(&args));
    }

    #[test]
    fn rejects_qcf_dump() {
        let mut args = default_args();
        args.qcf_dump = Some(std::path::PathBuf::from("/tmp/qcf.json"));
        assert!(!is_standard_happy_path(&args));
    }

    #[test]
    fn rejects_d2o_layer_alloc() {
        let mut args = default_args();
        args.eviction = Some(crate::session::cli::TopLevelCmd::Eviction {
            policy: crate::session::cli::EvictionCmd::D2o(crate::session::cli::D2oArgs {
                keep_ratio: 0.75,
                ema_beta: 0.7,
                merge_e: 0.1,
                layer_alloc: true,
                protected_layers: None,
                merge_axis: "both".to_string(),
            }),
        });
        assert!(!is_standard_happy_path(&args));
    }

    #[test]
    fn rejects_non_none_eviction() {
        let mut args = default_args();
        args.eviction = Some(crate::session::cli::TopLevelCmd::Eviction {
            policy: crate::session::cli::EvictionCmd::Sliding(crate::session::cli::SlidingArgs {
                window: 1024,
            }),
        });
        assert!(!is_standard_happy_path(&args));
    }

    /// `skip_ratio = Some(0.0)`는 비활성과 동등 (CLI 명시 했지만 0.0 = no skip)
    #[test]
    fn accepts_skip_ratio_zero() {
        let mut args = default_args();
        args.skip_ratio = Some(0.0);
        assert!(is_standard_happy_path(&args));
    }
}
