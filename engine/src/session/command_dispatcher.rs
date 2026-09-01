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

use argus_shared::{CommandResult, EngineCommand};

use crate::inference::signal_runtime::SignalRuntime;
use crate::kv::cache_manager::CacheManager;
use crate::kv::standard_format::StandardFormat;
use crate::session::pipeline_registry::PipelineRegistry;
use crate::stages::kv::aperturb_select_stage::AperturbSelectStage;
use crate::stages::kv::eviction::EvictionStage;

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

    /// Report what became of the commands the matching [`Self::poll`] returned,
    /// in the same order, so the source can answer the directives they came from.
    ///
    /// The driver calls this after [`CommandDispatcher::dispatch`], because a
    /// command's outcome is not known at poll time. Sources with no outbound
    /// channel (schedule replay, tests) keep the default no-op.
    fn report_results(&mut self, _results: Vec<CommandResult>) {}
}

/// ExecutionPlan 축소판 — driver-local 루프 제어 상태 (v2 §5.4 ② channel).
///
/// `CommandDispatcher::dispatch` 가 매 step 갱신하고, `DecodeLoop::run` 이 읽어 sleep/break/pacing
/// 한다. v1 `ExecutionPlan` 의 control 필드와 1:1 (매핑 문서 1.2/1.3).
///
/// **과도기 필드(layer_skip)** 는 대응 Stage 미구현이라 deprecated 로 잔존한다(G1).
/// **partition 은 AB-4 에서 OneShot `PartitionStage`, swap 은 AB-6 에서 OneShot `WeightSwapStage`,
/// quant 는 AB-2 에서 OneShot `QuantWindowBitTransitionStage`, offload/recall 은 AB-3 에서 OneShot `OffloadStage`
/// 로 이전됨** — 삭제된 필드: `partition_ratio`/`swap_weights`/`kv_quant_bits`/`offload_ratio`/
/// `recall_offload` (§5.5/§5.6/§5.7/§5.10).
#[derive(Debug, Clone, Default)]
pub struct LoopControl {
    // ── ② control 핵심 (run() live 소비) ──
    /// Whether inference should be suspended → loop break (G6 보존). (Suspend)
    pub suspended: bool,

    // ── ② control 비-live (seam 잔존, run() 미소비) ──
    /// Whether inference should resume from suspension. (Resume — executor 내부 state 만)
    pub resumed: bool,

    // ── ② RestoreDefaults 묶음 ──
    /// Whether to restore all action-induced state to defaults.
    pub restore_defaults: bool,
    // ── 과도기 (deprecated, G1) ──

    // ── ③ Hardware resolve seam (run() 미소비) ──
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
    /// §5.9.1 Track A: score-based eviction 의 attention score accumulator 공유 cell.
    /// ModelForward(begin_step + 주입) + EvictionStage(read + reset) 와 동일 cell 을 공유한다.
    /// `compute_and_send_qcf` 에서 active acc 의 `importance_scores()` 를 QCF `token_scores` 로 전달.
    /// `submit_evict` 에서 EvictionStage 생성 시 score_cell 전달(scored 경로 선택).
    /// score-based 미구성 조립처는 `Arc::new(Mutex::new(None))` 더미(QCF uniform fallback 유지).
    score_cell: Arc<Mutex<Option<SignalRuntime>>>,
    /// ② 누적 루프 제어 상태 (sticky control — throttle/tbt 유지, evict 는 OneShot 으로 분리).
    control: LoopControl,

    // ── sticky 상태 (2부 — v1 executor 의 sticky carry/게이트 흡수) ──
    /// Last budget an evict OneShot was submitted for in this active window, or `None`
    /// before the first one and after a `RestoreDefaults` re-arm.
    ///
    /// This was a bare `evict_armed: bool` (v1 `evict_applied` equivalence: at most one
    /// OneShot per active window). A bool gate is value-BLIND: the second directive of a
    /// tightening sequence — 0.50 then 0.35 then 0.25 as pressure rises — was dropped and
    /// answered `Ok`, so the manager saw success, saw the cache unchanged, and escalated
    /// into a budget it could never reach. Comparing the value instead keeps the
    /// once-per-window property for a REPEATED budget (which is what the equivalence
    /// actually protects) while letting a DIFFERENT budget through. Same shape as
    /// `last_partition_ratio` / `last_quant_bits` / `last_reencode_format`.
    last_evict_ratio: Option<f32>,
    /// bench GPU-score 경로용 backend. `submit_evict` 가 `EvictionStage::one_shot_scored` 에 넘겨,
    /// score-fed eviction 이 score 를 읽기 직전 GPU 누적 score 를 CPU accumulator 로 sync 하게 한다
    /// (`init_gpu_score_acc` 로 `gpu_score_active=true` 일 때 decode 가 CPU accumulate 를 건너뛰므로).
    /// 기본 `None`(ctor param 아님 — `reencode_fired_cell` 처럼 호출처 무변경); build_bench_loop 가
    /// `with_backend` 로 OpenCL backend 를 주입한다. `None` 이면 기존 CPU accumulate 경로 무변.
    backend: Option<Arc<dyn crate::backend::Backend>>,
    /// The engine's own compression choice, when one is configured: a resolved candidate pool plus
    /// the query rows the metric scores them on. `Some` makes a `KvCompress` submit an
    /// [`AperturbSelectStage`] instead of the single CLI-configured policy — the contract says how
    /// much KV may remain, and this is what decides by what technique.
    ///
    /// `None` (the default, and every path that configures no pool) keeps the method-drop
    /// behaviour: the `CacheManager`'s one policy prunes to the budget.
    aperturb: Option<AperturbSelection>,
    /// Per-command outcomes of the last [`Self::dispatch`], in the order the commands
    /// arrived. Drained by [`Self::finalize_results`] so the driver can hand them back to
    /// the `CommandSource` that produced the commands.
    last_results: Vec<CommandResult>,
    /// Where in `last_results` a KV-compression sits, and how full the cache was when it
    /// was submitted, so [`Self::finalize_results`] can say what the stage achieved
    /// instead of what it was asked for. `None` when no compression was submitted this
    /// step. See [`Self::finalize_results`] for why submit-time is too early to answer.
    pending_compress: Option<PendingCompress>,
    /// Index the command currently being applied will occupy in `last_results`. Scratch
    /// for `apply`, which does not otherwise know where its answer lands.
    result_idx: usize,
}

/// The engine's own compression chooser, and the query rows it scores candidates on.
///
/// The rows come from the live forward — `ModelForward` captures into this same cell — so a
/// decision measures what this session actually computed rather than a re-derivation of it.
type AperturbSelection = (
    Arc<crate::kv::aperturb_select::Selector>,
    Arc<Mutex<Option<crate::inference::q_rows::QRowCapture>>>,
);

/// A KV compression submitted this step, awaiting its post-apply reading.
struct PendingCompress {
    /// Index into `CommandDispatcher::last_results`.
    result_idx: usize,
    /// Retained fraction the directive asked for.
    target_ratio: f32,
    /// Resident tokens at submit time — the denominator of what was achieved.
    tokens_before: usize,
}

impl CommandDispatcher {
    /// dispatcher 생성. `cache_manager` 가 `None` 이면 evict directive 는 무시되고(미구성),
    /// `layer_slots` 가 비었거나 `hardware` 가 `None` 이면 partition directive 는 무시된다.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        registry: Arc<PipelineRegistry>,
        kv_handles: Vec<Arc<StandardFormat>>,
        cache_manager: Option<Arc<Mutex<CacheManager>>>,
        // §5.9.1 Track A: score-based eviction 의 accumulator cell (ModelForward 공유).
        // score-based 미구성 조립처는 `Arc::new(Mutex::new(None))` 더미.
        score_cell: Arc<Mutex<Option<SignalRuntime>>>,
    ) -> Self {
        Self {
            registry,
            kv_handles,
            cache_manager,
            score_cell,
            control: LoopControl::default(),
            last_evict_ratio: None,
            backend: None,
            aperturb: None,
            last_results: Vec::new(),
            pending_compress: None,
            result_idx: 0,
        }
    }

    /// bench GPU-score 경로: score-fed `EvictionStage`(submit_evict)가 GPU 누적 score 를 CPU 로
    /// sync 할 backend 를 주입한다. build_bench_loop 가 decode 와 동일한 OpenCL `Arc`(= 동일 GPU
    /// score buffer)를 넘긴다. 미호출(chat/standard/test)이면 `None` → 기존 CPU accumulate 경로 무변.
    pub fn with_backend(mut self, backend: Option<Arc<dyn crate::backend::Backend>>) -> Self {
        self.backend = backend;
        self
    }

    /// Let the engine choose its own compression technique: a `KvCompress` then submits an
    /// [`AperturbSelectStage`](crate::stages::kv::aperturb_select_stage::AperturbSelectStage) over
    /// `selector`'s candidate pool instead of applying the one configured policy. `q_rows` is the
    /// cell `ModelForward` captures into, shared so the decision reads the rows this session's own
    /// forward produced.
    pub fn with_aperturb_selector(
        mut self,
        selector: Arc<crate::kv::aperturb_select::Selector>,
        q_rows: Arc<Mutex<Option<crate::inference::q_rows::QRowCapture>>>,
    ) -> Self {
        self.aperturb = Some((selector, q_rows));
        self
    }

    /// 마지막 [`Self::dispatch`] 가 갱신한 누적 [`LoopControl`] 읽기.
    pub fn control(&self) -> &LoopControl {
        &self.control
    }

    /// Take the per-command outcomes of the last [`Self::dispatch`], leaving the
    /// dispatcher empty. Same length and order as that call's `cmds`.
    ///
    /// Call this AFTER the `KvMutate` dispatch. `dispatch` only *submits* a compression
    /// as a one-shot stage; the stage runs later in the same step, and it can decline —
    /// `run_policy_eviction` no-ops below `MIN_EVICT_TOKENS` rather than shave off a
    /// handful of tokens. Answering at submit time therefore reported `Ok` for a cache
    /// nothing had touched. Here the resident token count is read back and compared with
    /// what the directive asked for, so a compression that did not happen, or landed
    /// short, answers `Partial` with the fraction it actually reached.
    pub fn finalize_results(&mut self) -> Vec<CommandResult> {
        if let Some(p) = self.pending_compress.take()
            && let Some(r) = self.compress_outcome(&p)
        {
            self.last_results[p.result_idx] = r;
        }
        std::mem::take(&mut self.last_results)
    }

    /// Read back what a submitted compression achieved, or `None` to keep the submit-time
    /// answer (no handle to measure with, or an empty cache to begin with).
    fn compress_outcome(&self, p: &PendingCompress) -> Option<CommandResult> {
        use crate::format::KVCacheFormat;
        let after = self.kv_handles.first()?.current_pos();
        if p.tokens_before == 0 {
            return None;
        }
        let achieved = after as f32 / p.tokens_before as f32;
        // One token of slack: a target that lands on a fraction cannot be hit exactly, and
        // `target_len` is `(pos * ratio) as usize` floored then `.max(1)`.
        let slack = 1.0 / p.tokens_before as f32;
        if achieved <= p.target_ratio + slack {
            return Some(CommandResult::Ok);
        }
        Some(CommandResult::Partial {
            achieved,
            reason: if after == p.tokens_before {
                format!(
                    "eviction declined: fewer than {} tokens would have been removed",
                    crate::kv::MIN_EVICT_TOKENS
                )
            } else {
                "eviction stopped short of the requested budget".to_string()
            },
        })
    }

    /// 도착한 command 들을 분배하고 갱신된 [`LoopControl`] 을 반환한다.
    ///
    /// 구 `CommandExecutor::apply_command`(executor.rs:360-571) + `poll` 후처리(:344-355) 로직 이동:
    /// - **transient reset**: control 의 1-step 필드(evict 트리거 제외 transient)는 매 dispatch 진입
    ///   시 초기화하되, sticky 필드(throttle/tbt/quant/partition)는 carry. v1 `ExecutionPlan::default`
    ///   에서 시작 후 sticky carry 하던 것과 등가.
    /// - **suspend override**: suspended 면 evict 미submit + device seam clear (v1 :344-352 등가).
    pub fn dispatch(&mut self, cmds: Vec<EngineCommand>) -> &LoopControl {
        // transient(매 step 새로 결정되는) 필드만 초기화 — sticky(last_evict_ratio)는 carry.
        self.control.suspended = false;
        self.control.resumed = false;
        self.control.restore_defaults = false;

        self.last_results = Vec::with_capacity(cmds.len());
        self.pending_compress = None;
        for cmd in &cmds {
            self.result_idx = self.last_results.len();
            let r = self.apply(cmd);
            self.last_results.push(r);
        }

        &self.control
    }

    /// 단일 command 분배 + 그 결과 판정.
    ///
    /// The returned [`CommandResult`] is what the Manager is told. `Rejected` means the
    /// engine cannot carry the command out **in this configuration** — an unconfigured
    /// subsystem — and is how a Manager discovers the engine's real action set, since the
    /// contract has no capability exchange. A compression's `Ok` here is provisional:
    /// [`Self::finalize_results`] replaces it once the stage it submitted has run.
    fn apply(&mut self, cmd: &EngineCommand) -> CommandResult {
        match cmd {
            // ① KV 압축 → OneShot EvictionStage submit. 어떤 기법으로 줄일지는 계약이 말하지
            // 않는다 — CM 이 보유한 CLI 구성 기법이 예산까지 prune 한다.
            EngineCommand::KvCompress { budget } => self.submit_compress(*budget),

            // ② lifecycle → LoopControl
            EngineCommand::Suspend => {
                self.control.suspended = true;
                CommandResult::Ok
            }
            EngineCommand::Resume => {
                self.control.resumed = true;
                CommandResult::Ok
            }
            EngineCommand::RestoreDefaults => {
                self.control.restore_defaults = true;
                // 재무장: 다음 KvCompress 가 새 OneShot submit 가능.
                self.last_evict_ratio = None;
                CommandResult::Ok
            }
        }
    }

    /// ① evict directive 1건을 OneShot `EvictionStage` 로 submit (method-drop).
    ///
    /// 상태 A/B 등가(2부): 같은 budget 은 active 구간당 1회만 submit. CacheManager 미구성
    /// (`None`)이거나 handle 이 없으면 no-op(happy/chat 동등 — v1 `cache_manager=None` 분기).
    /// §5.9.1 Track A: score_cell 이 구성된 경우 `EvictionStage::one_shot_scored` 경로 사용 —
    /// run_eviction 이 acc.importance_scores() 를 추출해 force_evict_with_scores 호출, 직후 acc.reset().
    fn submit_compress(&mut self, target_ratio: f32) -> CommandResult {
        if self.last_evict_ratio == Some(target_ratio) {
            // 같은 budget 재요청 — 이미 이 active 구간에서 submit 됐다 (v1 evict_applied 등가).
            // 요청한 상태가 이미 성립하므로 실패가 아니다. 값이 **다르면** 아래로 내려가
            // 새 OneShot 을 submit 한다 — 그것이 bool 게이트와의 차이다.
            return CommandResult::Ok;
        }
        if self.kv_handles.is_empty() {
            return CommandResult::Rejected {
                reason: "no kv cache handles are registered".to_string(),
            };
        }
        // The contract names a budget, not a technique. When a candidate pool is configured the
        // engine picks the technique itself; otherwise it applies the one the CLI configured.
        let stage: Arc<dyn crate::pipeline::PipelineStage> = match self.aperturb.as_ref() {
            Some((selector, q_rows)) => Arc::new(AperturbSelectStage::new(
                self.kv_handles.clone(),
                Arc::clone(selector),
                Arc::clone(q_rows),
                target_ratio,
                Arc::clone(&self.score_cell),
                self.backend.clone(),
            )),
            None => {
                let Some(cm) = self.cache_manager.as_ref() else {
                    return CommandResult::Rejected {
                        reason: "kv cache manager is not configured".to_string(),
                    };
                };
                Arc::new(EvictionStage::one_shot_scored(
                    self.kv_handles.clone(),
                    Arc::clone(cm),
                    target_ratio,
                    Arc::clone(&self.score_cell),
                    self.backend.clone(),
                ))
            }
        };
        self.last_evict_ratio = Some(target_ratio);
        {
            use crate::format::KVCacheFormat;
            self.pending_compress = Some(PendingCompress {
                result_idx: self.result_idx,
                target_ratio,
                tokens_before: self.kv_handles.first().map_or(0, |h| h.current_pos()),
            });
        }
        self.registry.submit(stage);
        CommandResult::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::format::KVCacheFormat;
    use crate::kv::eviction::stage_registry::sliding_backed_policy;
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
        let policy = sliding_backed_policy(10, 4);
        Arc::new(Mutex::new(CacheManager::new(
            policy,
            Box::new(NoOpMonitor),
            usize::MAX,
            0.3,
        )))
    }

    /// A dispatcher with a cache manager and one layer handle — enough to submit a
    /// compression.
    fn make_dispatcher() -> (
        CommandDispatcher,
        Arc<PipelineRegistry>,
        Arc<StandardFormat>,
    ) {
        let registry = Arc::new(PipelineRegistry::new());
        let handle = make_handle(N_TOKENS);
        let d = CommandDispatcher::new(
            Arc::clone(&registry),
            vec![handle.clone()],
            Some(make_cm()),
            Arc::new(Mutex::new(None)),
        );
        (d, registry, handle)
    }

    /// The same, with no cache manager: nothing can compress.
    fn bare_dispatcher() -> (CommandDispatcher, Arc<PipelineRegistry>) {
        let registry = Arc::new(PipelineRegistry::new());
        let d = CommandDispatcher::new(
            Arc::clone(&registry),
            vec![make_handle(N_TOKENS)],
            None,
            Arc::new(Mutex::new(None)),
        );
        (d, registry)
    }

    fn results_of(d: &mut CommandDispatcher, cmds: Vec<EngineCommand>) -> Vec<CommandResult> {
        d.dispatch(cmds);
        d.finalize_results()
    }

    fn compress(budget: f32) -> EngineCommand {
        EngineCommand::KvCompress { budget }
    }

    fn is_rejected(r: &CommandResult) -> bool {
        matches!(r, CommandResult::Rejected { .. })
    }

    /// These tests exercise the dispatcher alone — nothing runs the `KvMutate` phase, so a
    /// submitted compression legitimately finalizes as `Partial` ("the cache did not
    /// move"). What they assert about a compression is therefore that it was ACCEPTED;
    /// `unapplied_compression_reports_partial` covers the other half.
    fn is_accepted(r: &CommandResult) -> bool {
        !is_rejected(r)
    }

    /// A configured candidate pool is what decides, and it needs no `CacheManager` — the technique
    /// no longer comes from one. Mutation-proof: leaving the `cache_manager` guard ahead of the
    /// selector branch makes this `Rejected` and submits nothing.
    #[test]
    fn a_configured_pool_compresses_without_a_cache_manager() {
        use crate::kv::aperturb_select::{Candidate, Selector};

        let registry = Arc::new(PipelineRegistry::new());
        let d = CommandDispatcher::new(
            Arc::clone(&registry),
            vec![make_handle(N_TOKENS)],
            None, // no cache manager — the selector is the whole configuration
            Arc::new(Mutex::new(None)),
        );
        let basis = Arc::new(
            crate::aperturb::OutputBasis::from_layers(vec![vec![1.0f32]], 1, 1, None).unwrap(),
        );
        let selector = Arc::new(
            Selector::new(
                vec![Candidate::new(
                    "none",
                    argus_extension_api::find_mutation_stage("none")
                        .map(|r| (r.make)(Default::default(), &[]))
                        .expect("the built-in no-eviction stage is registered"),
                    argus_extension_api::StageCaps::SCORE_FREE,
                )],
                basis,
                1,
            )
            .unwrap(),
        );
        let mut d = d.with_aperturb_selector(selector, Arc::new(Mutex::new(None)));
        let r = results_of(&mut d, vec![compress(0.5)]);
        assert!(
            is_accepted(&r[0]),
            "a pool-configured compression must not be rejected for a missing cache manager: {:?}",
            r[0]
        );
        assert_eq!(registry.len(), 1, "one selection stage submitted");
    }

    #[test]
    fn compress_submits_one_shot_once_per_budget() {
        let (mut d, registry, _h) = make_dispatcher();
        assert_eq!(registry.len(), 0);
        assert!(is_accepted(&results_of(&mut d, vec![compress(0.5)])[0]));
        assert_eq!(registry.len(), 1, "첫 압축 → OneShot 1개 submit");
        d.dispatch(vec![]);
        assert_eq!(registry.len(), 1, "빈 batch — 재submit 없음");
        let again = results_of(&mut d, vec![compress(0.5)]);
        assert!(matches!(again[..], [CommandResult::Ok]));
        assert_eq!(registry.len(), 1, "같은 budget 반복 — 재submit 없음");
    }

    /// 예산을 조이는 연속 directive 는 매번 새 OneShot 을 submit 한다. 값-무관 bool 게이트
    /// 시절에는 두 번째부터 조용히 버려지고 `Ok` 로 보고됐다.
    #[test]
    fn tightening_budget_resubmits() {
        let (mut d, registry, _h) = make_dispatcher();
        for (i, budget) in [0.50f32, 0.35, 0.25].into_iter().enumerate() {
            let r = results_of(&mut d, vec![compress(budget)]);
            assert!(is_accepted(&r[0]), "{budget}: {r:?}");
            assert_eq!(
                registry.len(),
                i + 1,
                "budget {budget} 는 새 OneShot 을 submit"
            );
        }
    }

    /// `RestoreDefaults` 는 재무장한다 — 그 뒤 같은 budget 도 다시 submit 된다.
    #[test]
    fn restore_defaults_rearms_compression() {
        let (mut d, registry, _h) = make_dispatcher();
        results_of(&mut d, vec![compress(0.5)]);
        assert_eq!(registry.len(), 1);
        let r = results_of(&mut d, vec![EngineCommand::RestoreDefaults]);
        assert!(matches!(r[..], [CommandResult::Ok]));
        results_of(&mut d, vec![compress(0.5)]);
        assert_eq!(registry.len(), 2, "RestoreDefaults 후 재submit 가능");
    }

    /// 압축을 제출했는데 stage 가 캐시를 건드리지 않았으면 `Ok` 가 아니라 `Partial` 이다.
    /// 이 유닛 테스트는 `KvMutate` 를 돌리지 않으므로 `MIN_EVICT_TOKENS` 로 거절당한
    /// 실전 케이스와 관측 결과가 같다 — 매니저가 「적용됐다」로 오독하면 안 되는 상황.
    #[test]
    fn unapplied_compression_reports_partial() {
        let (mut d, registry, handle) = make_dispatcher();
        let before = handle.current_pos();
        let r = results_of(&mut d, vec![compress(0.5)]);
        assert_eq!(registry.len(), 1, "stage 는 submit 됐다");
        match &r[0] {
            CommandResult::Partial { achieved, reason } => {
                assert!((*achieved - 1.0).abs() < 1e-6, "achieved==1.0: {achieved}");
                assert!(reason.contains("declined"), "이유가 실려야 한다: {reason}");
            }
            other => panic!("미적용 압축은 Partial 이어야 한다, got {other:?}"),
        }
        assert_eq!(
            before,
            handle.current_pos(),
            "이 테스트는 stage 를 안 돌린다"
        );
    }

    /// 압축할 수단이 없으면 `Rejected` — 계약에 capability 교환이 없으므로 이것이 매니저가
    /// 엔진의 액션 집합을 배우는 유일한 경로다.
    #[test]
    fn compress_without_cache_manager_is_rejected() {
        let (mut d, registry) = bare_dispatcher();
        let r = results_of(&mut d, vec![compress(0.5)]);
        assert!(is_rejected(&r[0]), "{r:?}");
        assert_eq!(registry.len(), 0, "Rejected 는 stage 를 submit 하지 않는다");
    }

    #[test]
    fn lifecycle_commands_are_ok_and_drive_control() {
        let (mut d, _r, _h) = make_dispatcher();
        let r = results_of(&mut d, vec![EngineCommand::Suspend]);
        assert!(matches!(r[..], [CommandResult::Ok]));
        assert!(d.control().suspended, "Suspend → LoopControl");

        let r = results_of(&mut d, vec![EngineCommand::Resume]);
        assert!(matches!(r[..], [CommandResult::Ok]));
        assert!(
            !d.control().suspended,
            "다음 dispatch 가 transient 를 초기화"
        );
        assert!(d.control().resumed);

        let r = results_of(&mut d, vec![EngineCommand::RestoreDefaults]);
        assert!(matches!(r[..], [CommandResult::Ok]));
        assert!(d.control().restore_defaults);
    }

    /// `finalize_results` 는 비운다 — 다음 dispatch 가 이전 결과를 물려받지 않는다.
    #[test]
    fn finalize_results_drains() {
        let (mut d, _r, _h) = make_dispatcher();
        d.dispatch(vec![EngineCommand::Suspend]);
        assert_eq!(d.finalize_results().len(), 1);
        assert!(d.finalize_results().is_empty(), "두 번째 호출은 비어 있다");
    }

    /// 명령 1건 = 결과 1건, 순서 보존.
    #[test]
    fn results_match_commands_one_to_one() {
        let (mut d, _r, _h) = make_dispatcher();
        let r = results_of(
            &mut d,
            vec![EngineCommand::Suspend, compress(0.5), EngineCommand::Resume],
        );
        assert_eq!(r.len(), 3);
        assert!(matches!(r[0], CommandResult::Ok));
        assert!(is_accepted(&r[1]));
        assert!(matches!(r[2], CommandResult::Ok));
    }
}
