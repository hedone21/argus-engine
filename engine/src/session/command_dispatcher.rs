//! `CommandDispatcher` + `LoopControl` — v2 §5.4 A-1 의 2-source 명령 분배자 (Phase β-4).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.4 (2-source 모델) +
//! `arch/beta4_command_channel_mapping.md` (18-variant × 19필드 전수 매핑 = 구현 명세).
//!
//! [`CommandSource::poll`](super::traits::CommandSource) 가 pure 생산한 [`EngineCommand`] 들을
//! 받아 v2 §5.4 의 3분류로 분배한다:
//!
//! - **① OneShot EvictionStage** — evict-family 4종(KvEvictH2o/KvEvictSliding/KvStreaming/
//!   KvMergeD2o) → `registry.submit(EvictionStage::one_shot(...))` (method-drop 시맨틱 — directive
//!   의 method 는 무시하고 `keep_ratio`→`target_ratio` 만 사용, 정책은 CM 의 CLI 구성, 3부).
//! - **② LoopControl** — control 7종(throttle/tbt/suspend/resume/restore/qcf/prefill) + 과도기
//!   5종(offload/recall/quant/swap/partition/layer-skip — deprecated, 등가 보존 G1).
//! - **③ Hardware resolve seam** — SwitchHw/PrepareComputeUnit (seam 만, run() 인라인 소비 없음).
//!
//! **sticky 등가 (2부)**: v1 `evict_plan` sticky carry + driver `evict_applied` 1회-게이트 =
//! OneShot Consumed GC 1회성. directive 1회 = submit 1회 = 발화 1회 = GC. RestoreDefaults →
//! 재제출 가능 reset. v1 `evict_applied` 는 dispatcher 내부 sticky 상태로 흡수된다.

use std::sync::{Arc, Mutex};

use argus_shared::{EngineCapability, EngineCommand, EngineMessage, QcfEstimate, WeightSwapReport};

use crate::hardware::Hardware;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::kv::cache_manager::CacheManager;
use crate::kv::kivi_format::KIVIFormat;
use crate::kv::standard_format::StandardFormat;
use crate::models::transformer::TransformerModel;
use crate::models::weights::LayerSlot;
use crate::qcf_collector::ImportanceLookup;
use crate::session::pipeline_registry::PipelineRegistry;
use crate::session::swap_runtime::EngineSwapRuntime;
use crate::stages::kv::eviction::EvictionStage;
use crate::stages::kv::kivi_quant::KiviQuantStage;
use crate::stages::kv::offload::OffloadStage;
use crate::stages::weight::partition::PartitionStage;
use crate::stages::weight::weight_recall::WeightRecallStage;
use crate::stages::weight::weight_swap::WeightSwapStage;

/// External command channel (manager IPC, schedule, stdin, ...).
///
/// **Phase β-7**: moved here from the deleted `session::traits` — this is the
/// dispatcher's input seam.
///
/// **β-4 retarget (v2 §5.4 A-1)**: `poll` 은 **pure 생산자**다 — drain 한
/// [`EngineCommand`] 들을 그대로 반환할 뿐, `ExecutionPlan` 으로 번역하지 않고
/// registry 도 모른다. 번역(① OneShot Stage submit / ② LoopControl / ③ Hardware seam)은
/// [`CommandDispatcher`] 책임이다.
///
/// heartbeat 등 부수효과(매핑 문서 4부 채택안 (가))는 source 구현체 내부에 잔존한다 —
/// `kv_snap` 운반은 poll 인자가 아니라 source 가 register 시점 보유한 held-handle query 로
/// 교체된다(`ManagerCommandSource`). pure poll 은 `ctx`/`kv_snap` 인자가 없다.
pub trait CommandSource {
    /// Per-step poll — 도착한 manager command 들을 drain 하여 반환한다 (pure).
    /// Default Noop 은 빈 `Vec` 을 반환.
    fn poll(&mut self) -> anyhow::Result<Vec<EngineCommand>>;
}

/// Outbound reporting channel (engine → manager).
///
/// **Phase β-7**: moved here from the deleted `session::traits`. Implemented by
/// `ResilienceAdapter` (the surviving `CommandExecutor` side).
pub trait EngineReport {
    fn send_capability(&mut self, _cap: EngineCapability) {}
    fn send_qcf_estimate(&mut self, _qcf: QcfEstimate) {}
    fn send_swap_report(&mut self, _report: WeightSwapReport) {}
}

/// ExecutionPlan 축소판 — driver-local 루프 제어 상태 (v2 §5.4 ② channel).
///
/// `CommandDispatcher::dispatch` 가 매 step 갱신하고, `DecodeLoop::run` 이 읽어 sleep/break/pacing
/// 한다. v1 `ExecutionPlan` 의 control 필드와 1:1 (매핑 문서 1.2/1.3).
///
/// **과도기 필드(layer_skip)** 는 대응 Stage 미구현이라 deprecated 로 잔존한다(G1).
/// **partition 은 AB-4 에서 OneShot `PartitionStage`, swap 은 AB-6 에서 OneShot `WeightSwapStage`,
/// quant 는 AB-2 에서 OneShot `KiviQuantStage`, offload/recall 은 AB-3 에서 OneShot `OffloadStage`
/// 로 이전됨** — 삭제된 필드: `partition_ratio`/`swap_weights`/`kv_quant_bits`/`offload_ratio`/
/// `recall_offload` (§5.5/§5.6/§5.7/§5.10).
#[derive(Debug, Clone, Default)]
pub struct LoopControl {
    // ── ② control 핵심 (run() live 소비) ──
    /// Throttle delay between tokens (ms). 0 = no throttle. (Throttle)
    pub throttle_delay_ms: u64,
    /// Target TBT in ms. 0 = disabled. (SetTargetTbt)
    pub target_tbt_ms: u64,
    /// True iff `target_tbt_ms` was explicitly set at least once. (SetTargetTbt / RestoreDefaults)
    pub target_tbt_set: bool,
    /// Whether inference should be suspended → loop break (G6 보존). (Suspend)
    pub suspended: bool,

    // ── ② control 비-live (seam 잔존, run() 미소비) ──
    /// Whether inference should resume from suspension. (Resume — executor 내부 state 만)
    pub resumed: bool,

    // ── ② RestoreDefaults 묶음 ──
    /// Whether to restore all action-induced state to defaults.
    pub restore_defaults: bool,

    // ── 과도기 (deprecated, G1) ──
    /// Layer skip ratio. (LayerSkip — sticky, forward skip_config 측)
    pub layer_skip: Option<f32>,

    // ── ③ Hardware resolve seam (run() 미소비) ──
    /// Device to switch to. (SwitchHw)
    pub switch_device: Option<String>,
    /// Device to pre-warm. (PrepareComputeUnit)
    pub prepare_device: Option<String>,
}

/// v2 §5.4 A-1 의 명령 분배자. `EngineCommand` 를 ① OneShot Stage submit / ② LoopControl /
/// ③ Hardware seam 으로 분배한다.
///
/// **L4 동층 합성** (INV-LAYER-006 BANNED 비해당): dispatcher 는 driver(`DecodeLoop`)와 같은 L4 에서
/// 합성되며, registry(L4)·CacheManager(L3, `Arc<Mutex>`)·held-handle(`Arc<StandardFormat>`) 를 보유한다.
pub struct CommandDispatcher {
    /// ① evict directive 가 EvictionStage 를 submit 할 stage registry (driver 와 공유).
    registry: Arc<PipelineRegistry>,
    /// ① EvictionStage 가 prune 할 KV handle (register 시점 보유, INV-STAGE-LAYER-HANDLE).
    kv_handles: Vec<Arc<StandardFormat>>,
    /// ① EvictionStage 들이 공유하는 단일 CacheManager (CLI 정책·sticky eviction 상태).
    /// `None` 이면 evict directive 가 와도 submit 안 함(happy/chat 동등 — eviction 미구성).
    cache_manager: Option<Arc<Mutex<CacheManager>>>,
    /// ① PartitionStage 가 re-slice 할 전체 layer slot handle (register 시점 보유, AB-4 §5.5.1).
    /// 비어 있으면 SetPartitionRatio directive 가 와도 submit 안 함(미구성 — happy/chat).
    layer_slots: Vec<Arc<LayerSlot>>,
    /// ① PartitionStage 의 companion backend resolve 용 (AB-4 §5.5.8). `None` 이면 partition
    /// directive 무시(model/hardware 미배선 — host 단위테스트 등).
    hardware: Option<Arc<Hardware>>,
    /// ① WeightSwapStage 가 swap 할 model handle (register 시점 보유, AB-6 §5.6.3 model seam).
    /// `None` 이면 swap directive 무시(미배선 — host 단위테스트 등).
    model: Option<Arc<TransformerModel>>,
    /// ① WeightSwapStage 의 swap 자원 묶음(AB-6 §5.6.7). `None` 이면 swap directive 무시
    /// (secondary 부재 — happy/chat).
    swap_runtime: Option<Arc<EngineSwapRuntime>>,
    /// ① WeightSwapStage 의 decider importance 입력(AB-6 §5.6.1). `None` 이면 uniform fallback.
    importance: Option<Arc<dyn ImportanceLookup>>,
    /// §5.9.2 Track B: WeightSwapStage(IntraForward/LayerImmediate)가 hook 을 설치할 공유 cell.
    /// ModelForward 와 동일 cell 을 assembly 가 만들어 양측에 `Arc` clone 으로 넘긴다. swap
    /// 미구성 조립처는 `Arc::new(Mutex::new(None))` 더미 — submit 자체가 안 일어나 무영향.
    hook_cell: Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>>,
    /// §5.9.1 Track A: score-based eviction 의 attention score accumulator 공유 cell.
    /// ModelForward(begin_step + 주입) + EvictionStage(read + reset) 와 동일 cell 을 공유한다.
    /// `compute_and_send_qcf` 에서 active acc 의 `importance_scores()` 를 QCF `token_scores` 로 전달.
    /// `submit_evict` 에서 EvictionStage 생성 시 score_cell 전달(scored 경로 선택).
    /// score-based 미구성 조립처는 `Arc::new(Mutex::new(None))` 더미(QCF uniform fallback 유지).
    score_cell: Arc<Mutex<Option<AttentionScoreAccumulator>>>,
    /// ① KiviQuantStage 가 transition 할 KIVI handle (register 시점 보유, AB-2 §5.7.8). 비어 있으면
    /// KvQuantDynamic directive 가 와도 submit 안 함(미구성 — non-KIVI: Standard/Offload).
    kivi_handles: Vec<Arc<KIVIFormat>>,
    /// AB-5 §5.8.2: RequestQcf dispatch 시 QcfEstimate 를 manager 로 송출하는 채널.
    /// `None` 이면 미배선(resilience-off / host 단위테스트) → RequestQcf 가 무송출(inert).
    report_tx: Option<std::sync::mpsc::Sender<EngineMessage>>,
    /// ② 누적 루프 제어 상태 (sticky control — throttle/tbt 유지, evict 는 OneShot 으로 분리).
    control: LoopControl,

    // ── sticky 상태 (2부 — v1 executor 의 sticky carry/게이트 흡수) ──
    /// 상태 A 등가(v1 `evict_applied`): active 구간당 evict OneShot 1회만 submit.
    /// evict directive 도착 시 true, RestoreDefaults 도착 시 false 로 재무장.
    evict_armed: bool,
    /// AB-4 §5.5.2: partition sticky last-applied 게이트. 같은 ratio 의 재submit 을 막고
    /// (idempotent re-slice 비용 절감), 값 변경 시 재적용한다(evict 의 bool armed 와 다름 —
    /// partition 은 값이 곧 상태이므로 값 비교가 정확한 게이트). RestoreDefaults 시 `None` 으로
    /// reset(재무장) + Full 복원 submit.
    last_partition_ratio: Option<f32>,
    /// AB-2 §5.7.3: quant sticky last-applied 게이트. 같은 bits 의 재submit 을 막고(`transition_bits`
    /// 자체 no-op 이기도 하나 loop 진입 자체 차단 = 비용 절감), 값 변경 시 재적용한다(partition 의
    /// `last_partition_ratio` 와 동형 — 값이 곧 상태이므로 값 비교가 정확한 게이트, evict bool armed
    /// 복사 금지). RestoreDefaults 시 `None` 으로 reset(재무장) — **16bit 복원 transition 없음**
    /// (v1 등가, partition `submit_partition_full` 과 비대칭).
    last_quant_bits: Option<u8>,
    /// AB-3 §5.10.3: offload 가 한 번이라도 적용됐는가 상태. KvOffload directive 도착 시 true,
    /// RestoreDefaults recall 완료 후(submit_offload_recall) false. RestoreDefaults 시 recall
    /// submit 여부의 게이트 — offload 미적용이면 불필요 disk 접근 0.
    offload_armed: bool,
}

impl CommandDispatcher {
    /// dispatcher 생성. `cache_manager` 가 `None` 이면 evict directive 는 무시되고(미구성),
    /// `layer_slots` 가 비었거나 `hardware` 가 `None` 이면 partition directive 는 무시된다.
    /// `model`/`swap_runtime` 이 `None` 이면 swap directive 는 무시된다(AB-6 §5.6.4).
    /// `report_tx` 가 `None` 이면 RequestQcf 무송출(inert — AB-5 §5.8.2).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        registry: Arc<PipelineRegistry>,
        kv_handles: Vec<Arc<StandardFormat>>,
        cache_manager: Option<Arc<Mutex<CacheManager>>>,
        layer_slots: Vec<Arc<LayerSlot>>,
        hardware: Option<Arc<Hardware>>,
        model: Option<Arc<TransformerModel>>,
        swap_runtime: Option<Arc<EngineSwapRuntime>>,
        importance: Option<Arc<dyn ImportanceLookup>>,
        // AB-2 §5.7.8: KiviQuantStage 가 transition 할 KIVI handle. Standard/Offload 경로는 빈 Vec
        // → KvQuantDynamic directive 무시 (inert — evict CM=None 동형).
        kivi_handles: Vec<Arc<KIVIFormat>>,
        // AB-5 §5.8.2: RequestQcf dispatch 시 QcfEstimate 를 송출할 채널. None → inert.
        report_tx: Option<std::sync::mpsc::Sender<EngineMessage>>,
        // §5.9.2 Track B: WeightSwapStage 에 넘길 layer-boundary hook cell (ModelForward 공유).
        hook_cell: Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>>,
        // §5.9.1 Track A: score-based eviction 의 accumulator cell (ModelForward 공유).
        // score-based 미구성 조립처는 `Arc::new(Mutex::new(None))` 더미.
        score_cell: Arc<Mutex<Option<AttentionScoreAccumulator>>>,
    ) -> Self {
        Self {
            registry,
            kv_handles,
            cache_manager,
            layer_slots,
            hardware,
            model,
            swap_runtime,
            importance,
            kivi_handles,
            report_tx,
            hook_cell,
            score_cell,
            control: LoopControl::default(),
            evict_armed: false,
            last_partition_ratio: None,
            last_quant_bits: None,
            offload_armed: false,
        }
    }

    /// 마지막 [`Self::dispatch`] 가 갱신한 누적 [`LoopControl`] 읽기.
    pub fn control(&self) -> &LoopControl {
        &self.control
    }

    /// 도착한 command 들을 분배하고 갱신된 [`LoopControl`] 을 반환한다.
    ///
    /// 구 `CommandExecutor::apply_command`(executor.rs:360-571) + `poll` 후처리(:344-355) 로직 이동:
    /// - **transient reset**: control 의 1-step 필드(evict 트리거 제외 transient)는 매 dispatch 진입
    ///   시 초기화하되, sticky 필드(throttle/tbt/quant/partition)는 carry. v1 `ExecutionPlan::default`
    ///   에서 시작 후 sticky carry 하던 것과 등가.
    /// - **suspend override**: suspended 면 evict 미submit + device seam clear (v1 :344-352 등가).
    pub fn dispatch(&mut self, cmds: Vec<EngineCommand>) -> &LoopControl {
        // transient(매 step 새로 결정되는) 필드만 초기화 — sticky(throttle/tbt/quant/partition)는 carry.
        self.control.suspended = false;
        self.control.resumed = false;
        self.control.restore_defaults = false;
        self.control.switch_device = None;
        self.control.prepare_device = None;

        for cmd in &cmds {
            self.apply(cmd);
        }

        // 상태 A/B 재무장 시맨틱 (2부 핵심 등가 명제, line 186-194): v1 의 disarm 은 **빈 batch 가
        // 아니라 RestoreDefaults** 에서만 일어난다. v1 evict_plan 이 sticky carry 되어 빈 poll 에서도
        // plan.evict 가 Some 유지 → evict_applied 가 reset 되지 않기 때문(decode_loop.rs:279 의
        // plan.evict.is_none() 분기는 evict_plan=None 일 때만 = RestoreDefaults 후). 따라서 disarm 은
        // apply 의 RestoreDefaults arm 에서만 수행한다(빈 batch 에서 disarm 금지 — active 구간 유지).

        // suspend override (v1 poll :344-352): suspend 면 device seam 무효 + throttle 0.
        if self.control.suspended {
            self.control.switch_device = None;
            self.control.prepare_device = None;
            self.control.throttle_delay_ms = 0;
            self.control.resumed = false;
        }

        &self.control
    }

    /// 단일 command 분배.
    fn apply(&mut self, cmd: &EngineCommand) {
        match cmd {
            // ── ① evict-family 4종 → OneShot EvictionStage submit (method-drop, 3부) ──
            EngineCommand::KvEvictH2o { keep_ratio }
            | EngineCommand::KvEvictSliding { keep_ratio }
            | EngineCommand::KvMergeD2o { keep_ratio } => {
                self.submit_evict(*keep_ratio);
            }
            EngineCommand::KvStreaming { .. } => {
                // StreamingLLM 은 target_len 무시 → target_ratio=0.0 (CM CLI sink/window 정책 사용).
                self.submit_evict(0.0);
            }

            // ── ② control 7종 → LoopControl ──
            EngineCommand::Throttle { delay_ms } => {
                self.control.throttle_delay_ms = *delay_ms;
            }
            EngineCommand::SetTargetTbt { target_ms } => {
                self.control.target_tbt_ms = *target_ms;
                self.control.target_tbt_set = true;
            }
            EngineCommand::Suspend => {
                self.control.suspended = true;
            }
            EngineCommand::Resume => {
                self.control.resumed = true;
                self.control.throttle_delay_ms = 0;
            }
            EngineCommand::RestoreDefaults => {
                // v1 RestoreDefaults(:484-502) 등가 — reset 묶음.
                self.control.restore_defaults = true;
                self.control.throttle_delay_ms = 0;
                self.control.target_tbt_ms = 0;
                // target_tbt_set 은 false 로 (CLI fallback branch — v1 :489 주석 등가).
                self.control.target_tbt_set = false;
                // AB-2 §5.7.3: quant guard clear 만 (재무장 — 다음 KvQuantDynamic 이 어떤 bits 든
                // 재적용). **16bit 복원 transition submit 없음** (v1 등가 — partition
                // `submit_partition_full` 과 비대칭).
                self.last_quant_bits = None;
                // 상태 C 재무장: 다음 KvEvict* directive 가 새 OneShot submit 가능 (2부).
                self.evict_armed = false;
                // AB-4 §5.5.2: partition 을 GPU-only(Full)로 복원 + last reset(재무장). v1
                // (`generate.rs:3132 RestoreDefaults: re-split...`)이 partition 을 GPU-only 로
                // 되돌린 것과 등가 — Full 복원 OneShot submit 후 last=None 으로 두면 다음
                // SetPartitionRatio 가 어떤 ratio 든 재적용된다.
                self.submit_partition_full();
                // AB-3 §5.10.3: offload 가 적용됐으면 recall OneShot submit.
                self.submit_offload_recall();
            }
            // AB-5 §5.8.2: query directive — dispatcher 직결 compute+send. LoopControl 경유 0.
            EngineCommand::RequestQcf => {
                self.compute_and_send_qcf();
            }
            // SetPrefillPolicy: DecodeLoop run() 미소비 (prefill 정책은 CLI Args 경로 전용 —
            // init.rs 가 args.prefill_* 를 직접 read). LoopControl.prefill_* 는 dead write 였어
            // 삭제됨(2026-06-13 census) → directive 수신 시 no-op (MSG 표면은 존속).
            EngineCommand::SetPrefillPolicy { .. } => {}

            // ── ① offload → OneShot OffloadStage submit (AB-3 §5.10) ──
            EngineCommand::KvOffload { ratio } => {
                self.submit_offload(ratio.clamp(0.0, 1.0));
            }
            // ── ① quant → OneShot KiviQuantStage submit (AB-2 §5.7.3) ──
            EngineCommand::KvQuantDynamic { target_bits } => {
                self.submit_kv_quant(*target_bits);
            }
            // ── ① partition → OneShot PartitionStage submit (AB-4 §5.5.2) ──
            EngineCommand::SetPartitionRatio { ratio } => {
                self.submit_partition(*ratio);
            }
            EngineCommand::LayerSkip { skip_ratio } => {
                self.control.layer_skip = Some(*skip_ratio);
            }
            // ── ① swap → OneShot WeightSwapStage submit (AB-6 §5.6.4, transient) ──
            EngineCommand::SwapWeights {
                ratio,
                target_dtype,
            } => {
                self.submit_swap(*ratio, *target_dtype);
            }
            // ── ① recall → OneShot WeightRecallStage submit (§5.6.8, ENG-ALG-240) ──
            // RestoreDefaults 는 무발화(INV-192) — 이 arm 만 발화.
            EngineCommand::RecallWeights { ratio } => {
                self.submit_recall(*ratio);
            }

            // ── ③ Hardware resolve seam ──
            EngineCommand::SwitchHw { device } => {
                self.control.switch_device = Some(device.clone());
            }
            EngineCommand::PrepareComputeUnit { device } => {
                self.control.prepare_device = Some(device.clone());
            }
        }
    }

    /// ① evict directive 1건을 OneShot `EvictionStage` 로 submit (method-drop).
    ///
    /// 상태 A/B 등가(2부): `evict_armed` 게이트로 active 구간당 1회만 submit. CacheManager 미구성
    /// (`None`)이거나 handle 이 없으면 no-op(happy/chat 동등 — v1 `cache_manager=None` 분기).
    /// §5.9.1 Track A: score_cell 이 구성된 경우 `EvictionStage::one_shot_scored` 경로 사용 —
    /// run_eviction 이 acc.importance_scores() 를 추출해 force_evict_with_scores 호출, 직후 acc.reset().
    fn submit_evict(&mut self, target_ratio: f32) {
        if self.evict_armed {
            return; // 이미 active 구간 내 submit 됨 (재적용 방지 — v1 evict_applied 등가).
        }
        let Some(cm) = self.cache_manager.as_ref() else {
            return; // 미구성 — directive 무시 (happy/chat).
        };
        if self.kv_handles.is_empty() {
            return;
        }
        self.evict_armed = true;
        let stage = EvictionStage::one_shot_scored(
            self.kv_handles.clone(),
            Arc::clone(cm),
            target_ratio,
            Arc::clone(&self.score_cell),
        );
        self.registry.submit(Arc::new(stage));
    }

    /// ① KvQuantDynamic directive 1건을 OneShot `KiviQuantStage` 로 submit (AB-2 §5.7.3).
    ///
    /// sticky last-applied 게이트: 같은 bits 면 재submit 0(transition_bits 자체 no-op 이기도 하나
    /// loop 진입 차단 = 비용 절감), 값 변경 시 재적용(새 OneShot → transition). `kivi_handles` 가
    /// 비면 미구성(non-KIVI: Standard/Offload) — directive 무시(evict CM=None 동형). partition 의
    /// `last_partition_ratio` 게이트와 동형(evict bool armed 복사 금지 — 값 비교 게이트).
    fn submit_kv_quant(&mut self, target_bits: u8) {
        if self.last_quant_bits == Some(target_bits) {
            return; // 값 무변경 → 재submit 0 (sticky 핵심 — partition last-applied 동형).
        }
        if self.kivi_handles.is_empty() {
            return; // 미구성 (non-KIVI: Standard/Offload — KIVI handle 부재).
        }
        self.last_quant_bits = Some(target_bits);
        let stage = KiviQuantStage::one_shot(self.kivi_handles.clone(), target_bits);
        self.registry.submit(Arc::new(stage));
    }

    /// ① SetPartitionRatio directive 1건을 OneShot `PartitionStage` 로 submit (AB-4 §5.5.2).
    ///
    /// sticky last-applied 게이트: 같은 ratio 면 재submit 0(idempotent re-slice 비용 절감),
    /// 값 변경 시 재적용(새 OneShot → re-slice). `layer_slots` 가 비었거나 `hardware` 가 `None`
    /// 이면 미구성 — directive 무시(happy/chat).
    fn submit_partition(&mut self, ratio: f32) {
        if self.last_partition_ratio == Some(ratio) {
            return; // 값 무변경 → 재submit 0 (sticky 핵심 — evict bool armed 와 다른 값 비교 게이트).
        }
        if self.layer_slots.is_empty() {
            return; // 미구성 (happy/chat — partition 미배선).
        }
        let Some(hw) = self.hardware.as_ref() else {
            return; // hardware 미배선 (host 단위테스트 등).
        };
        self.last_partition_ratio = Some(ratio);
        let stage = PartitionStage::one_shot(self.layer_slots.clone(), Arc::clone(hw), ratio);
        self.registry.submit(Arc::new(stage));
    }

    /// RestoreDefaults 시 partition 을 GPU-only(Full)로 복원 + last reset (AB-4 §5.5.2).
    ///
    /// `last_partition_ratio == None` 이면 애초에 partition 이 적용된 적 없으므로 복원 불요
    /// (no-op). 적용된 적 있으면 Full 복원 OneShot submit 후 `last=None`(재무장 — 다음
    /// SetPartitionRatio 가 어떤 ratio 든 재적용).
    fn submit_partition_full(&mut self) {
        if self.last_partition_ratio.is_none() {
            return; // partition 미적용 — 복원 불요.
        }
        self.last_partition_ratio = None;
        let (false, Some(hw)) = (self.layer_slots.is_empty(), self.hardware.as_ref()) else {
            return; // 미구성 — submit 생략 (last 는 이미 None 으로 reset 됨).
        };
        // ratio=1.0 → apply_partition_dispatch 가 GPU-only fast path(LayerDispatch::Full) 적용.
        let stage = PartitionStage::one_shot(self.layer_slots.clone(), Arc::clone(hw), 1.0);
        self.registry.submit(Arc::new(stage));
    }

    /// ① KvOffload directive 1건을 OneShot `OffloadStage` 로 submit (AB-3 §5.10).
    ///
    /// **transient 시맨틱** — armed 게이트 없음(매 directive = 새 submit, WeightSwapStage 동형).
    /// `cache_manager` 미구성(`None`)이거나 handle 이 없으면 no-op(happy/chat 동등). submit 후
    /// `offload_armed = true` — RestoreDefaults 가 recall 을 게이트할 때 사용.
    fn submit_offload(&mut self, ratio: f32) {
        let Some(cm) = self.cache_manager.as_ref() else {
            return; // 미구성 — directive 무시.
        };
        if self.kv_handles.is_empty() {
            return;
        }
        self.offload_armed = true;
        let stage = OffloadStage::offload(self.kv_handles.clone(), Arc::clone(cm), ratio);
        self.registry.submit(Arc::new(stage));
    }

    /// ① RestoreDefaults → offload 적용 이력 있으면 OneShot recall `OffloadStage` submit (AB-3 §5.10.3).
    ///
    /// `offload_armed` 이면 recall Stage submit + `offload_armed = false`. 미적용이면 no-op
    /// (불필요 disk 접근 0). `cache_manager`/handle 미구성이어도 offload_armed 가 true 면
    /// disarm — submit 는 생략(핸들 부재 시 이미 submit 이 안 됐으므로 armed=true 는 이론상
    /// 없으나 방어적 처리).
    fn submit_offload_recall(&mut self) {
        if !self.offload_armed {
            return; // offload 미적용 → recall 불요.
        }
        self.offload_armed = false;
        let Some(cm) = self.cache_manager.as_ref() else {
            return;
        };
        if self.kv_handles.is_empty() {
            return;
        }
        let stage = OffloadStage::recall(self.kv_handles.clone(), Arc::clone(cm));
        self.registry.submit(Arc::new(stage));
    }

    /// AB-5 §5.8.2: RequestQcf 수신 시 dispatcher 직결 QcfEstimate 산출·송출.
    ///
    /// `report_tx` 가 `None` 이면 inert(미배선 — resilience-off / host 단위테스트).
    /// `report_tx` 가 `Some` 이면 `compute_qcf_estimates` 로 dry-run 산출 후 `EngineMessage::QcfEstimate` 송출.
    /// §5.9.1 Track A: score_cell 이 active 면 `importance_scores()` 를 `token_scores` 로 전달해
    /// kv.evict_h2o / kv.merge_d2o 키를 산출한다. `&self` — query 이므로 mutation 없음(acc no-op).
    fn compute_and_send_qcf(&self) {
        let Some(tx) = self.report_tx.as_ref() else {
            return; // 미배선 → inert (drop).
        };
        // §5.9.1 Track A: score_cell lock → active acc면 importance_scores() 복사 후 즉시 해제.
        // 단일 스레드(INV-018) → lock contention 0. 복사 후 guard 해제로 forward 와 lock 충돌 없음.
        let token_scores_owned: Option<Vec<f32>> = {
            let guard = self.score_cell.lock().expect("score_cell mutex poisoned");
            guard
                .as_ref()
                .filter(|acc| acc.is_active())
                .map(|acc| acc.importance_scores().to_vec())
        };
        let ctx = crate::session::qcf_runtime::QcfEstimateContext {
            kv_handles: &self.kv_handles,
            kivi_handles: &self.kivi_handles,
            importance: self.importance.as_deref(),
            streaming_config: None,
            importance_table: None,
            num_layers: self.kv_handles.len().max(self.kivi_handles.len()),
            // §5.9.1 Track A: active score 면 token_scores 전달(h2o/d2o QCF 언블록).
            // None 이면 기존 uniform fallback 유지(QCF_kv ⊥ QCF_weight 분리 보존).
            token_scores: token_scores_owned.as_deref(),
        };
        let est = crate::session::qcf_runtime::compute_qcf_estimates(&ctx);
        let _ = tx.send(EngineMessage::QcfEstimate(argus_shared::QcfEstimate {
            estimates: est,
            layer_swap: None,
        }));
    }

    /// ① RecallWeights directive 1건을 OneShot `WeightRecallStage` 로 submit (§5.6.8).
    ///
    /// submit_swap 과 동형이나 `target_dtype` 이 없다(방향 고정 = F16, ENG-ALG-240).
    /// `model`/`swap_runtime` 이 `None` 이거나 `layer_slots` 가 비면 미구성 — directive 무시.
    fn submit_recall(&mut self, ratio: f32) {
        let Some(model) = self.model.as_ref() else {
            return; // 미구성 (secondary 부재 — happy/chat).
        };
        let Some(rt) = self.swap_runtime.as_ref() else {
            return; // 미구성 (swap 자원 미배선).
        };
        if self.layer_slots.is_empty() {
            return; // 미구성.
        }
        let stage = WeightRecallStage::one_shot(
            Arc::clone(model),
            Arc::clone(rt),
            ratio,
            Arc::clone(&self.hook_cell),
        );
        self.registry.submit(Arc::new(stage));
    }

    /// ① SwapWeights directive 1건을 OneShot `WeightSwapStage` 로 submit (AB-6 §5.6.4).
    ///
    /// **transient 시맨틱**: partition 의 last-applied 게이트도 evict 의 armed 게이트도 **없다**
    /// (landmine — partition 게이트 복사 금지, §5.6.4). 같은 directive 가 재도착하면 새 Stage 를
    /// submit 하되, 그 Stage 의 commit §2 in-flight 가드(swap_runtime 공유 마커)가 미완 plan 이
    /// 살아 있으면 reject 한다(R-1 동시 활성화 차단 = 정확한 게이트). `model`/`swap_runtime` 이
    /// `None` 이거나 `layer_slots` 가 비면 미구성 — directive 무시(happy/chat).
    fn submit_swap(&mut self, ratio: f32, target_dtype: argus_shared::DtypeTag) {
        let Some(model) = self.model.as_ref() else {
            return; // 미구성 (secondary 부재 — happy/chat).
        };
        let Some(rt) = self.swap_runtime.as_ref() else {
            return; // 미구성 (swap 자원 미배선).
        };
        if self.layer_slots.is_empty() {
            return; // 미구성.
        }
        let stage = WeightSwapStage::one_shot(
            Arc::clone(model),
            Arc::clone(rt),
            self.importance.clone(),
            ratio,
            target_dtype,
            Arc::clone(&self.hook_cell),
        );
        self.registry.submit(Arc::new(stage));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::format::KVCacheFormat;
    use crate::kv::eviction::sliding_window::SlidingWindowPolicy;
    use crate::kv::kv_cache::KVCache;
    use crate::memory::host::shared::SharedBuffer;
    use crate::resilience::sys_monitor::NoOpMonitor;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 32;
    const MAX_SEQ: usize = 128;
    const N_TOKENS: usize = 120;

    fn make_handle(n_tokens: usize) -> Arc<StandardFormat> {
        let total = MAX_SEQ * KV_HEADS * HEAD_DIM;
        let k_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend);
        let mut cache = KVCache::new(k, v, MAX_SEQ);
        cache.current_pos = n_tokens;
        Arc::new(StandardFormat::new(0, cache))
    }

    fn make_cm() -> Arc<Mutex<CacheManager>> {
        let policy = Box::new(SlidingWindowPolicy::new(10, 4));
        Arc::new(Mutex::new(CacheManager::new(
            policy,
            Box::new(NoOpMonitor),
            usize::MAX,
            0.3,
        )))
    }

    fn make_dispatcher() -> (
        CommandDispatcher,
        Arc<PipelineRegistry>,
        Arc<StandardFormat>,
    ) {
        let registry = Arc::new(PipelineRegistry::new());
        let handle = make_handle(N_TOKENS);
        let cm = make_cm();
        // partition/swap/quant 미구성 (빈 slots + None hardware/model/swap_runtime + 빈 kivi_handles):
        // evict 전용 dispatcher. report_tx=None (AB-5 단위테스트는 미배선).
        let d = CommandDispatcher::new(
            Arc::clone(&registry),
            vec![handle.clone()],
            Some(cm),
            Vec::new(),
            None,
            None,
            None,
            None,
            Vec::new(),
            None,                       // report_tx: AB-5
            Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
            Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
        );
        (d, registry, handle)
    }

    // ── AB-4 (B): partition handle 을 구성한 dispatcher helper ──

    fn cpu_be() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    fn f32_weight(be: &Arc<dyn Backend>, out_dim: usize, in_dim: usize) -> Tensor {
        let buf: Arc<dyn crate::buffer::Buffer> =
            Arc::new(SharedBuffer::new(out_dim * in_dim * 4, DType::F32));
        Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, be.clone())
    }

    fn ffn_slot(be: &Arc<dyn Backend>, idx: usize) -> Arc<LayerSlot> {
        use crate::layers::transformer_layer::TransformerLayer;
        let small = f32_weight(be, 1, 1);
        let layer = TransformerLayer {
            wq: small.clone(),
            wk: small.clone(),
            wv: small.clone(),
            wo: small.clone(),
            w_gate: f32_weight(be, 512, 256),
            w_up: f32_weight(be, 512, 256),
            w_down: f32_weight(be, 256, 512),
            attention_norm: small.clone(),
            ffn_norm: small,
            qkv_bias: None,
            q_norm: None,
            k_norm: None,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            partition_ctx: None,
        };
        Arc::new(LayerSlot::new(layer, DType::F32, None, idx))
    }

    fn make_partition_dispatcher() -> (CommandDispatcher, Arc<PipelineRegistry>) {
        let registry = Arc::new(PipelineRegistry::new());
        let be = cpu_be();
        let slots: Vec<Arc<LayerSlot>> = (0..2).map(|i| ffn_slot(&be, i)).collect();
        let host: Arc<dyn crate::memory::Memory> = Arc::new(crate::memory::galloc::Galloc::new());
        let hw = Arc::new(Hardware::new(be.clone(), None, None, host, None));
        let d = CommandDispatcher::new(
            Arc::clone(&registry),
            Vec::new(),
            None,
            slots,
            Some(hw),
            None,
            None,
            None,
            Vec::new(),
            None,                       // report_tx: AB-5
            Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
            Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
        );
        (d, registry)
    }

    // ── AB-6: swap handle(model + swap_runtime) 을 구성한 dispatcher helper ──

    fn make_swap_model(be: &Arc<dyn Backend>, n_layers: usize) -> Arc<TransformerModel> {
        use crate::memory::Memory;
        use crate::model_config::{ModelArch, ModelConfig};
        let mem = crate::memory::galloc::Galloc::new();
        let (dim, vocab) = (4usize, 8usize);
        let embed = Tensor::new(
            Shape::new(vec![vocab, dim]),
            mem.alloc(vocab * dim * 4, DType::F32).unwrap(),
            be.clone(),
        );
        let norm = Tensor::new(
            Shape::new(vec![dim]),
            mem.alloc(dim * 4, DType::F32).unwrap(),
            be.clone(),
        );
        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: n_layers,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 2,
            arch: ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        };
        let rr = crate::weight::setup_runtime_resources(be.clone());
        Arc::new(TransformerModel {
            config,
            layers: (0..n_layers).map(|i| ffn_slot(be, i)).collect(),
            embed_tokens: embed,
            norm: norm.clone(),
            lm_head: norm,
            lm_head_on_cpu: false,
            gpu_embed_tokens: None,
            cpu_backend: None,
            preload_pool: std::sync::OnceLock::new(),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            quant_noise: rr.quant_noise.clone(),
            release_worker: rr.release_worker.clone(),
        })
    }

    fn make_swap_runtime(be: &Arc<dyn Backend>) -> Arc<EngineSwapRuntime> {
        use crate::model_config::{ModelArch, ModelConfig};
        let dispatcher = Arc::new(crate::weight::AsyncSwapDispatcher::new(be.clone()));
        let rr = crate::weight::setup_runtime_resources(be.clone());
        let config = Arc::new(ModelConfig {
            vocab_size: 8,
            hidden_size: 4,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: 4,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 2,
            arch: ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        });
        Arc::new(EngineSwapRuntime::new(
            be.clone(),
            dispatcher,
            config,
            rr.release_worker.clone(),
            crate::session::cli::SwapMode::Incremental,
            1024 * 1024,
            4,
            None,
        ))
    }

    fn make_swap_dispatcher() -> (CommandDispatcher, Arc<PipelineRegistry>) {
        let registry = Arc::new(PipelineRegistry::new());
        let be = cpu_be();
        let slots: Vec<Arc<LayerSlot>> = (0..2).map(|i| ffn_slot(&be, i)).collect();
        let d = CommandDispatcher::new(
            Arc::clone(&registry),
            Vec::new(),
            None,
            slots,
            None,
            Some(make_swap_model(&be, 2)),
            Some(make_swap_runtime(&be)),
            None,
            Vec::new(),
            None,                       // report_tx: AB-5
            Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
            Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
        );
        (d, registry)
    }

    // ── AB-2: kivi_handles 를 구성한 dispatcher helper (CPU KiviCache) ──

    fn make_quant_dispatcher() -> (CommandDispatcher, Arc<PipelineRegistry>) {
        use crate::kv::kivi_cache::KiviCache;
        use crate::kv::kivi_format::KIVIFormat;
        let registry = Arc::new(PipelineRegistry::new());
        // CPU KiviCache(bits=16 initial — --kv-dynamic-quant 진입 동형). head_dim/residual = QKKV 배수.
        let kivi_handles: Vec<Arc<KIVIFormat>> = (0..2)
            .map(|i| {
                Arc::new(KIVIFormat::new(
                    i,
                    KiviCache::new_with_bits(1, 32, 128, 32, 16),
                ))
            })
            .collect();
        let d = CommandDispatcher::new(
            Arc::clone(&registry),
            Vec::new(),
            None,
            Vec::new(),
            None,
            None,
            None,
            None,
            kivi_handles,
            None,                       // report_tx: AB-5
            Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
            Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
        );
        (d, registry)
    }

    // ── ② control: throttle/tbt/suspend ──

    #[test]
    fn throttle_sets_control() {
        let (mut d, _r, _h) = make_dispatcher();
        let c = d.dispatch(vec![EngineCommand::Throttle { delay_ms: 50 }]);
        assert_eq!(c.throttle_delay_ms, 50);
        // sticky: 다음 dispatch 에서도 유지.
        let c2 = d.dispatch(vec![]);
        assert_eq!(c2.throttle_delay_ms, 50);
    }

    #[test]
    fn set_target_tbt_dirty_flag() {
        let (mut d, _r, _h) = make_dispatcher();
        let c0 = d.dispatch(vec![]);
        assert!(!c0.target_tbt_set);
        let c1 = d.dispatch(vec![EngineCommand::SetTargetTbt { target_ms: 250 }]);
        assert!(c1.target_tbt_set);
        assert_eq!(c1.target_tbt_ms, 250);
        // RestoreDefaults → set=false, ms=0.
        let c2 = d.dispatch(vec![EngineCommand::RestoreDefaults]);
        assert!(!c2.target_tbt_set);
        assert_eq!(c2.target_tbt_ms, 0);
        assert!(c2.restore_defaults);
    }

    #[test]
    fn suspend_overrides_device_seam() {
        let (mut d, _r, _h) = make_dispatcher();
        let c = d.dispatch(vec![
            EngineCommand::SwitchHw {
                device: "cpu".to_string(),
            },
            EngineCommand::Suspend,
        ]);
        assert!(c.suspended);
        assert!(c.switch_device.is_none(), "suspend → device seam clear");
    }

    // ── ① evict OneShot submit + sticky 1회성 (상태 A) ──

    #[test]
    fn evict_submits_one_shot_once() {
        let (mut d, registry, _h) = make_dispatcher();
        assert_eq!(registry.len(), 0);
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(registry.len(), 1, "첫 evict directive → OneShot 1개 submit");
        // sticky carry(빈 batch)에도 재submit 안 함 (상태 A — evict_armed 게이트).
        d.dispatch(vec![]);
        assert_eq!(registry.len(), 1, "빈 batch — 재submit 없음");
        // 같은 directive 반복도 재submit 안 함.
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(
            registry.len(),
            1,
            "active 구간 내 directive 반복 — 재submit 없음"
        );
    }

    #[test]
    fn restore_defaults_rearms_evict() {
        let (mut d, registry, _h) = make_dispatcher();
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(registry.len(), 1);
        // RestoreDefaults → 재무장 (다음 KvEvict* 가 새 OneShot submit 가능).
        d.dispatch(vec![EngineCommand::RestoreDefaults]);
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(
            registry.len(),
            2,
            "RestoreDefaults 후 재무장 → 새 OneShot submit"
        );
    }

    /// v1 sticky carry 등가: evict directive 없는 batch(빈/다른 directive)에서는 disarm 되지
    /// 않는다 — v1 evict_plan 이 sticky carry 되어 plan.evict 가 Some 유지되므로 evict_applied 도
    /// reset 안 됨. disarm 은 오직 RestoreDefaults 에서만 (2부 핵심 등가 명제).
    #[test]
    fn evict_stays_armed_without_restore_defaults() {
        let (mut d, registry, _h) = make_dispatcher();
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(registry.len(), 1);
        // evict directive 없는 batch (다른 directive) → 여전히 armed (sticky carry 등가, disarm 아님).
        d.dispatch(vec![EngineCommand::Throttle { delay_ms: 10 }]);
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(
            registry.len(),
            1,
            "RestoreDefaults 없이는 disarm 안 됨 — 재submit 없음"
        );
    }

    // ── method-drop (3부): KvEvictH2o directive 라도 stage 는 method 안 받음 (target_ratio 만) ──

    #[test]
    fn method_drop_h2o_directive_submits_target_ratio_only() {
        let (mut d, registry, handle) = make_dispatcher();
        // CLI 정책 = sliding(make_cm). KvEvictH2o directive 가 와도 sliding 으로 prune 됨.
        d.dispatch(vec![EngineCommand::KvEvictH2o { keep_ratio: 0.3 }]);
        assert_eq!(registry.len(), 1);
        // stage 발화 (KvMutate dispatch) → sliding prune.
        let mut profiler = crate::observability::profile::OpProfiler::new();
        let mut ctx = crate::pipeline::StageContext {
            step: crate::pipeline::StepInfo {
                pos: N_TOKENS,
                decode_step: 0,
                pressure: crate::pipeline::Pressure::new(0),
                prev_token: 0,
            },
            profiler: &mut profiler,
        };
        use crate::pipeline::{LifecyclePhase, PipelineDispatcher};
        registry.dispatch(LifecyclePhase::KvMutate, &mut ctx);
        assert!(
            handle.current_pos() < N_TOKENS,
            "method-drop: directive method 무시, CM 정책(sliding)으로 prune"
        );
    }

    // ── 과도기 등가 (LayerSkip) — AB-2/3/4/6 으로 이전된 quant/offload/partition/swap 제외 ──

    #[test]
    fn transitional_fields_equivalence() {
        let (mut d, _r, _h) = make_dispatcher();
        // AB-2: KvQuantDynamic, AB-3: KvOffload/recall, AB-4: SetPartitionRatio, AB-6: SwapWeights 는
        // LoopControl 필드가 아니라 OneShot Stage submit 으로 이전됨. 본 테스트는 잔존 과도기
        // 필드(layer_skip)만.
        let c = d.dispatch(vec![EngineCommand::LayerSkip { skip_ratio: 0.25 }]);
        assert_eq!(c.layer_skip, Some(0.25));
        // sticky: layer_skip 는 다음 dispatch 에 carry.
        let c2 = d.dispatch(vec![]);
        assert_eq!(c2.layer_skip, Some(0.25), "layer_skip sticky");
    }

    // ── AB-3: offload OneShot submit + transient 시맨틱 ──

    /// KvOffload directive → OneShot OffloadStage 1개 submit (transient).
    /// 같은 directive 재도착 → 새 submit (armed 게이트 없음 — WeightSwapStage 동형).
    #[test]
    fn offload_submits_one_shot_each_directive() {
        let (mut d, registry, _h) = make_dispatcher();
        assert_eq!(registry.len(), 0);
        d.dispatch(vec![EngineCommand::KvOffload { ratio: 0.5 }]);
        assert_eq!(registry.len(), 1, "첫 KvOffload → OneShot 1개 submit");
        // transient: 같은 directive 재도착 → 새 submit (armed 게이트 없음).
        d.dispatch(vec![EngineCommand::KvOffload { ratio: 0.5 }]);
        assert_eq!(
            registry.len(),
            2,
            "transient: 같은 directive 재도착 → 새 submit"
        );
        // 빈 batch 는 carry 0 (offload_ratio 필드 삭제 — sticky 아님).
        d.dispatch(vec![]);
        assert_eq!(registry.len(), 2, "빈 batch — carry 0 (transient)");
    }

    /// offload 적용 후 RestoreDefaults → recall OneShot submit.
    /// offload 미적용 상태 RestoreDefaults → recall submit 없음(no-op).
    #[test]
    fn restore_defaults_submits_recall_when_offloaded() {
        let (mut d, registry, _h) = make_dispatcher();
        // offload 미적용 상태 RestoreDefaults → recall submit 없음.
        d.dispatch(vec![EngineCommand::RestoreDefaults]);
        assert_eq!(registry.len(), 0, "offload 미적용 → recall submit 없음");
        // offload 적용 후 RestoreDefaults → recall OneShot submit.
        d.dispatch(vec![EngineCommand::KvOffload { ratio: 0.5 }]);
        assert_eq!(registry.len(), 1, "KvOffload → offload OneShot submit");
        d.dispatch(vec![EngineCommand::RestoreDefaults]);
        assert_eq!(
            registry.len(),
            2,
            "offload 적용 후 RestoreDefaults → recall OneShot submit"
        );
        // recall 후 armed=false → 다음 RestoreDefaults 에서 재recall 없음.
        d.dispatch(vec![EngineCommand::RestoreDefaults]);
        assert_eq!(
            registry.len(),
            2,
            "recall 후 RestoreDefaults → recall 없음 (armed=false)"
        );
    }

    // ── AB-2: quant OneShot submit + sticky last-applied 게이트 (§5.7.9 승계) ──

    /// quant 미구성(빈 kivi_handles) dispatcher 는 KvQuantDynamic 을 무시 (non-KIVI 경로).
    #[test]
    fn quant_unconfigured_ignores_directive() {
        let (mut d, registry, _h) = make_dispatcher(); // 빈 kivi_handles
        d.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 4 }]);
        assert_eq!(registry.len(), 0, "quant 미구성 → directive 무시");
    }

    /// 새 bits → OneShot KiviQuantStage 1개 submit. 같은 bits 반복/빈 batch → 재submit 0,
    /// 값 변경 → 재submit (partition `last_partition_ratio` 게이트 동형). RestoreDefaults 후 재무장.
    /// **16bit 복원 submit 부재** 단언(partition `submit_partition_full` 과 비대칭).
    #[test]
    fn quant_sticky_resubmits_on_value_change() {
        let (mut d, registry) = make_quant_dispatcher();
        assert_eq!(registry.len(), 0);
        // 첫 directive → OneShot 1 submit.
        d.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 4 }]);
        assert_eq!(registry.len(), 1, "첫 quant directive → OneShot 1개 submit");
        // 같은 bits 반복 → 재submit 0 (sticky last-applied).
        d.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 4 }]);
        assert_eq!(registry.len(), 1, "같은 bits 반복 — 재submit 없음");
        // 빈 batch 에도 재submit 0 (sticky carry 아님 — last-applied 비교).
        d.dispatch(vec![]);
        assert_eq!(registry.len(), 1, "빈 batch — 재submit 없음");
        // 값 변경(4 → 8) → 새 OneShot submit.
        d.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 8 }]);
        assert_eq!(registry.len(), 2, "bits 변경 → 새 OneShot submit");
        // RestoreDefaults → 재무장 (16bit 복원 submit 없음 — registry 불변).
        d.dispatch(vec![EngineCommand::RestoreDefaults]);
        assert_eq!(
            registry.len(),
            2,
            "RestoreDefaults → 16bit 복원 submit 없음 (partition 과 비대칭)"
        );
        // 재무장: 같은 bits(8)도 last=None reset 후라 재적용된다.
        d.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 8 }]);
        assert_eq!(
            registry.len(),
            3,
            "RestoreDefaults 후 재무장 → 어떤 bits 든 재적용"
        );
    }

    // ── exhaustive match 컴파일 강제 (게이트 2): variant 가 apply 에서 누락되면 컴파일 실패 ──
    // (apply 의 match 가 non-exhaustive `_` 없이 전 arm 을 다루므로, variant 추가 시 컴파일 에러.)

    #[test]
    fn cache_manager_none_ignores_evict() {
        let registry = Arc::new(PipelineRegistry::new());
        let handle = make_handle(N_TOKENS);
        let mut d = CommandDispatcher::new(
            Arc::clone(&registry),
            vec![handle],
            None,
            Vec::new(),
            None,
            None,
            None,
            None,
            Vec::new(),
            None,                       // report_tx: AB-5
            Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
            Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
        );
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(registry.len(), 0, "CM 미구성 → evict directive 무시");
    }

    // ── AB-4 (B): partition OneShot submit + sticky last-applied 게이트 ──

    /// 새 ratio → OneShot PartitionStage 1개 submit.
    #[test]
    fn partition_submits_one_shot_on_new_ratio() {
        let (mut d, registry) = make_partition_dispatcher();
        assert_eq!(registry.len(), 0);
        d.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }]);
        assert_eq!(
            registry.len(),
            1,
            "첫 partition directive → OneShot 1개 submit"
        );
    }

    /// 같은 ratio 반복 → 재submit 0 (sticky last-applied 게이트).
    #[test]
    fn partition_same_ratio_no_resubmit() {
        let (mut d, registry) = make_partition_dispatcher();
        d.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }]);
        assert_eq!(registry.len(), 1);
        // 같은 값 반복 → 무시 (idempotent re-slice 비용 절감).
        d.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }]);
        assert_eq!(registry.len(), 1, "같은 ratio 반복 — 재submit 없음");
        // 빈 batch 에도 재submit 없음 (sticky carry 아님 — last-applied 비교).
        d.dispatch(vec![]);
        assert_eq!(registry.len(), 1, "빈 batch — 재submit 없음");
    }

    /// 값 변경 → 재submit (partition 은 값이 곧 상태라 값 변경 시 재적용).
    #[test]
    fn partition_changed_ratio_resubmits() {
        let (mut d, registry) = make_partition_dispatcher();
        d.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }]);
        assert_eq!(registry.len(), 1);
        // 값 변경 (0.3 → 0.5) → 새 OneShot submit (evict 의 bool armed 와 다른 시맨틱).
        d.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.5 }]);
        assert_eq!(registry.len(), 2, "ratio 변경 → 새 OneShot submit");
    }

    /// RestoreDefaults → last reset + Full 복원 OneShot submit + 재무장.
    #[test]
    fn restore_defaults_resets_partition() {
        let (mut d, registry) = make_partition_dispatcher();
        d.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }]);
        assert_eq!(registry.len(), 1);
        // RestoreDefaults → Full 복원 OneShot submit (partition 적용된 상태였으므로).
        d.dispatch(vec![EngineCommand::RestoreDefaults]);
        assert_eq!(
            registry.len(),
            2,
            "RestoreDefaults → Full 복원 OneShot submit"
        );
        // 재무장: 같은 ratio(0.3)도 last=None reset 후라 재적용된다.
        d.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }]);
        assert_eq!(
            registry.len(),
            3,
            "RestoreDefaults 후 재무장 → 어떤 ratio 든 재적용"
        );
    }

    /// partition 미적용 상태에서 RestoreDefaults → Full 복원 submit 없음 (no-op).
    #[test]
    fn restore_defaults_without_partition_is_noop() {
        let (mut d, registry) = make_partition_dispatcher();
        d.dispatch(vec![EngineCommand::RestoreDefaults]);
        assert_eq!(
            registry.len(),
            0,
            "partition 미적용 → Full 복원 불요 (no-op)"
        );
    }

    /// partition 미구성(빈 slots) dispatcher 는 SetPartitionRatio 를 무시.
    #[test]
    fn partition_unconfigured_ignores_directive() {
        let (mut d, registry, _h) = make_dispatcher(); // 빈 slots + None hardware
        d.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }]);
        assert_eq!(registry.len(), 0, "partition 미구성 → directive 무시");
    }

    // ── AB-6: swap OneShot submit + transient 시맨틱 (§5.6.4) ──

    /// swap 미구성(model/swap_runtime=None) dispatcher 는 SwapWeights 를 무시.
    #[test]
    fn swap_unconfigured_ignores_directive() {
        let (mut d, registry, _h) = make_dispatcher(); // model=None, swap_runtime=None
        d.dispatch(vec![EngineCommand::SwapWeights {
            ratio: 0.9,
            target_dtype: argus_shared::DtypeTag::Q4_0,
        }]);
        assert_eq!(registry.len(), 0, "swap 미구성 → directive 무시");
    }

    /// SwapWeights arm → registry 에 WeightSwapStage 등록(첫 directive → 1 submit).
    #[test]
    fn swap_directive_submits_one_shot_stage() {
        let (mut d, registry) = make_swap_dispatcher();
        assert_eq!(registry.len(), 0);
        d.dispatch(vec![EngineCommand::SwapWeights {
            ratio: 0.9,
            target_dtype: argus_shared::DtypeTag::Q4_0,
        }]);
        assert_eq!(registry.len(), 1, "swap directive → OneShot 1개 submit");
    }

    /// transient 시맨틱(§5.6.4): partition 의 last-applied 게이트도 evict 의 armed 게이트도 없다.
    /// 같은 directive 재도착 → 새 submit (재submit 차단은 dispatcher 가 아니라 Stage in-flight
    /// 가드 담당 — landmine: partition 게이트 복사 금지). 따라서 dispatcher 레벨에서는 매 directive
    /// 가 submit 을 늘린다.
    #[test]
    fn swap_transient_resubmits_each_directive() {
        let (mut d, registry) = make_swap_dispatcher();
        d.dispatch(vec![EngineCommand::SwapWeights {
            ratio: 0.9,
            target_dtype: argus_shared::DtypeTag::Q4_0,
        }]);
        assert_eq!(registry.len(), 1);
        // 같은 directive 재도착 → 새 submit (값 비교 게이트 없음).
        d.dispatch(vec![EngineCommand::SwapWeights {
            ratio: 0.9,
            target_dtype: argus_shared::DtypeTag::Q4_0,
        }]);
        assert_eq!(
            registry.len(),
            2,
            "transient: 같은 directive 재도착 → 새 submit (게이트 없음)"
        );
        // 빈 batch 는 carry 0 (swap_weights 필드 삭제 — sticky 아님).
        d.dispatch(vec![]);
        assert_eq!(registry.len(), 2, "빈 batch — carry 0 (transient)");
    }
}
