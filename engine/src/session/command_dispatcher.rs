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

use crate::inference::prefill_attn::PrefillAttn;
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
    /// Tokens this context would hold if nothing had been compressed — the denominator a
    /// `KvCompress` budget is a fraction of. The contract names it: "the fraction of the
    /// **uncompressed KV byte** footprint to retain … not a token count and not a token
    /// ratio" (`argus-shared::EngineCommand::KvCompress`).
    ///
    /// It cannot be read off the cache. Compaction renumbers `current_pos`, so a cache that
    /// has already been compressed reports fewer positions than the conversation produced,
    /// and taking the budget against *that* makes the command **compound instead of
    /// restate**: a Manager walking 0.5, 0.25, 0.9, 0.85 … multiplies those together and
    /// ratchets the cache toward nothing, because every value that differs from the last one
    /// clears `last_evict_ratio` and applies afresh. Measured on the archived S25 runs, a
    /// dithering thermal ramp produced 111 such directives in one cell.
    ///
    /// Accumulating the **positive** deltas of `current_pos` separates the two motions that
    /// share that field: growth is appended tokens, a drop is a compaction and contributes
    /// nothing.
    ///
    /// Sampled twice a step — in `dispatch` (before `KvMutate`) and in `finalize_results`
    /// (after it), both of which the decode loop runs every step whether or not a command
    /// arrived. The second one is what keeps the token the forward appends in the same step
    /// as a compaction from being swallowed by the drop.
    ///
    /// ⚠ A compaction that lands **before the first sample** — a `PrefillEnd` prune — is
    /// invisible, so the anchor is then the post-prefill length rather than the prompt's.
    /// The two are mutually exclusive on the contract path: a configured candidate pool
    /// stands the standing `PrefillEnd` consumer down.
    logical_len: usize,
    /// `current_pos` at the last sample, to difference against. See [`Self::logical_len`].
    last_seen_pos: usize,
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
    Arc<Mutex<Option<PrefillAttn>>>,
);

/// A KV compression submitted this step, awaiting its post-apply reading.
struct PendingCompress {
    /// Index into `CommandDispatcher::last_results`.
    result_idx: usize,
    /// Retained fraction the directive asked for, in the contract's units — a fraction of
    /// [`CommandDispatcher::logical_len`], not of what was resident.
    budget: f32,
    /// The budget's denominator at submit time. Reported achievement uses it too, so the
    /// Manager reads an answer in the units it asked in.
    logical_len: usize,
    /// Resident tokens at submit time. Only for telling "the stage removed nothing" apart
    /// from "the stage stopped short".
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
            logical_len: 0,
            last_seen_pos: 0,
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
        prefill_attn: Arc<Mutex<Option<PrefillAttn>>>,
    ) -> Self {
        self.aperturb = Some((selector, q_rows, prefill_attn));
        self
    }

    /// The query-row ring the pool measures on, when one is configured.
    ///
    /// Handed out so the decode loop can tell the ring about a compaction: the ring's clock and
    /// the cache's `current_pos` separate at every prune, and nothing else sees both numbers.
    pub fn aperturb_q_rows(
        &self,
    ) -> Option<&Arc<Mutex<Option<crate::inference::q_rows::QRowCapture>>>> {
        self.aperturb.as_ref().map(|(_, q, _)| q)
    }

    /// The prompt-attention capture the pool's prefill-end candidates decide off, when one is
    /// configured.
    ///
    /// Handed out for the same reason as [`Self::aperturb_q_rows`]: its columns are cache
    /// positions, a compaction renumbers them, and the decode loop is the only place that sees
    /// every compaction — including the ones no candidate pool performed.
    pub fn aperturb_prefill_attn(&self) -> Option<&Arc<Mutex<Option<PrefillAttn>>>> {
        self.aperturb.as_ref().map(|(_, _, p)| p)
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
        // 두 번째 표집 — 이 호출은 `KvMutate` **직후**이고 명령이 0건이어도 매 step 온다
        // (decode_loop:362). 압축이 방금 재번호했으니 여기서 기준점을 새로 잡아 둬야, 같은
        // step 의 forward 가 붙일 토큰이 다음 표집에서 **증가로** 보인다. 이게 없으면 압축이
        // 일어난 step 의 토큰 하나가 매번 사라진다.
        self.observe_context();
        std::mem::take(&mut self.last_results)
    }

    /// Read back what a submitted compression achieved, or `None` to keep the submit-time
    /// answer (no handle to measure with, or an empty cache to begin with).
    fn compress_outcome(&self, p: &PendingCompress) -> Option<CommandResult> {
        use crate::format::KVCacheFormat;
        let after = self.kv_handles.first()?.current_pos();
        if p.logical_len == 0 {
            return None;
        }
        // Answered in the contract's units — a fraction of the uncompressed footprint — so
        // the Manager can compare it against the budget it sent without knowing what was
        // resident when the directive landed.
        let achieved = after as f32 / p.logical_len as f32;
        // One token of slack: a target that lands on a fraction cannot be hit exactly, and
        // `target_len` is `(logical_len * budget) as usize` floored then `.max(1)`.
        let slack = 1.0 / p.logical_len as f32;
        if achieved <= p.budget + slack {
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
        // 이 호출이 곧 「디코드 한 스텝」이다 (decode_loop 가 명령 유무와 무관하게 매 step
        // 부른다) — 문맥 길이를 여기서 표집한다. 이번 step 이 제출할 압축보다 **먼저** 봐야
        // 그 압축의 분모가 압축 전 길이가 된다.
        self.observe_context();
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

    /// Fold this step's cache growth into [`Self::logical_len`].
    ///
    /// `current_pos` moves for two unrelated reasons and only the sign tells them apart: it
    /// rises when the forward appends a token and falls when a compaction renumbers what is
    /// left. Taking the positive part keeps the first and discards the second, which is what
    /// makes the budget's denominator survive compression.
    ///
    /// A `current_pos` of **0** is a new sequence, not a compaction — a compaction floors its
    /// target at one token (`target_len … .max(1)`), so it can never land there.
    fn observe_context(&mut self) {
        use crate::format::KVCacheFormat;
        let Some(pos) = self.kv_handles.first().map(|h| h.current_pos()) else {
            return;
        };
        if pos == 0 {
            self.logical_len = 0;
        } else {
            self.logical_len += pos.saturating_sub(self.last_seen_pos);
        }
        self.last_seen_pos = pos;
    }

    /// ① evict directive 1건을 OneShot `EvictionStage` 로 submit (method-drop).
    ///
    /// 상태 A/B 등가(2부): 같은 budget 은 active 구간당 1회만 submit. CacheManager 미구성
    /// (`None`)이거나 handle 이 없으면 no-op(happy/chat 동등 — v1 `cache_manager=None` 분기).
    /// §5.9.1 Track A: score_cell 이 구성된 경우 `EvictionStage::one_shot_scored` 경로 사용 —
    /// run_eviction 이 acc.importance_scores() 를 추출해 force_evict_with_scores 호출, 직후 acc.reset().
    fn submit_compress(&mut self, budget: f32) -> CommandResult {
        use crate::format::KVCacheFormat;
        if self.last_evict_ratio == Some(budget) {
            // 같은 budget 재요청 — 이미 이 active 구간에서 submit 됐다 (v1 evict_applied 등가).
            // 요청한 상태가 이미 성립하므로 실패가 아니다. 값이 **다르면** 아래로 내려가
            // 새 OneShot 을 submit 한다 — 그것이 bool 게이트와의 차이다.
            return CommandResult::Ok;
        }
        let Some(h0) = self.kv_handles.first() else {
            return CommandResult::Rejected {
                reason: "no kv cache handles are registered".to_string(),
            };
        };
        // The budget is a fraction of what this context would occupy **uncompressed**, not of
        // what is resident now. Against the resident length a repeated budget would compound;
        // against this one it restates, which is what makes the command idempotent.
        let resident = h0.current_pos();
        let target_len = ((self.logical_len as f32 * budget) as usize).max(1);
        if target_len >= resident {
            // The cache already fits. Nothing to remove, so nothing to score — and scoring is
            // the expensive half: it recomputes the trailing query rows against every
            // candidate. This is the guard that makes a Manager which re-sends a **loosened**
            // budget every tick cost nothing.
            //
            // Answering `Ok` is not a silent drop: the state the directive names holds. The
            // value is recorded so an unchanged repeat short-circuits above, while any
            // tightening still falls through — the property the bool gate got wrong.
            self.last_evict_ratio = Some(budget);
            return CommandResult::Ok;
        }
        // The stages take a fraction of the resident length, which is what they can act on.
        // Converting here keeps the contract's denominator at the boundary and leaves
        // `force_evict` / `--eviction-target-ratio` meaning exactly what they meant before.
        let target_ratio = target_len as f32 / resident as f32;
        // The contract names a budget, not a technique. When a candidate pool is configured the
        // engine picks the technique itself; otherwise it applies the one the CLI configured.
        let stage: Arc<dyn crate::pipeline::PipelineStage> = match self.aperturb.as_ref() {
            Some((selector, q_rows, prefill_attn)) => Arc::new(AperturbSelectStage::new(
                self.kv_handles.clone(),
                Arc::clone(selector),
                Arc::clone(q_rows),
                target_ratio,
                Arc::clone(&self.score_cell),
                Arc::clone(prefill_attn),
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
        self.last_evict_ratio = Some(budget);
        self.pending_compress = Some(PendingCompress {
            result_idx: self.result_idx,
            budget,
            logical_len: self.logical_len,
            tokens_before: resident,
        });
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
        let mut d = d.with_aperturb_selector(
            selector,
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
        );
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

    /// 호환성 보장: **첫 압축까지는 새 분모가 옛 분모와 정확히 같다.**
    ///
    /// `logical_len` 은 첫 표집을 기준선으로 잡고 그 뒤의 증가만 더하므로, dispatcher 가
    /// 아직 아무것도 압축하지 않았다면 언제나 `current_pos` 와 같다 — 프리필 끝에서 prune 이
    /// 돌았더라도(기준선이 prune 된 값이 될 뿐) 마찬가지다. 그래서 지시 1건짜리 실행
    /// (`--aperturb-select` 실측 스케줄이 그렇다)은 이 변경 **전후로 바이트 동일**하다.
    ///
    /// mutation-proof: `observe_context` 를 `dispatch` 에서 빼면 `logical_len` 이 0 에 머물러
    /// 목표가 1 토큰이 되고, 아래 achieved 단정이 깨진다.
    #[test]
    fn the_first_budget_targets_exactly_what_the_old_denominator_did() {
        let (mut d, registry, h) = make_dispatcher();
        let resident = h.current_pos();
        assert!(is_accepted(&results_of(&mut d, vec![compress(0.5)])[0]));
        assert_eq!(registry.len(), 1);
        // 이것이 보장의 전부다: 첫 압축을 제출하는 순간 분모가 남은 길이와 **같다**.
        // 그러니 `target_len` 이 옛 규칙 `(resident * budget)` 과 글자 그대로 같은 값이다.
        assert_eq!(
            d.logical_len, resident,
            "첫 압축까지 문맥 길이는 남은 길이와 같아야 한다 (기존 측정 불변의 근거)"
        );
        // 디코드가 더 붙어도 압축 전이면 계속 같다.
        h.with_cache_mut(|c| c.advance_pos(5));
        d.dispatch(vec![]);
        assert_eq!(d.logical_len, h.current_pos(), "압축 전에는 계속 일치한다");
    }

    /// 예산의 분모는 **압축 전 문맥 길이**이지 남아 있는 길이가 아니다.
    ///
    /// 남은 길이로 재면 명령이 **누적**된다 — 0.5 뒤의 0.6 이 「원래의 60%」가 아니라
    /// 「남은 것의 60%」가 되어 캐시가 계단식으로 접힌다. 아카이브 S25 런에서 thermal 떨림이
    /// 한 셀에 그런 지시를 111건 냈다.
    ///
    /// mutation-proof: `submit_compress` 의 분모를 `logical_len` → `resident` 로 되돌리면
    /// 0.6 이 `0.6*60 = 36 < 60` 이라 새 stage 를 submit 해 아래 단정이 깨진다.
    #[test]
    fn the_budget_is_a_fraction_of_the_uncompressed_context() {
        let (mut d, registry, h) = make_dispatcher();
        assert!(is_accepted(&results_of(&mut d, vec![compress(0.5)])[0]));
        assert_eq!(registry.len(), 1, "0.5 → 60 토큰 목표, submit 된다");
        // 이 유닛 테스트는 KvMutate 를 안 돌리므로 stage 가 했을 압축을 손으로 반영한다.
        h.with_cache_mut(|c| c.set_current_pos(60));

        // 0.6 은 **느슨해진** 예산이다. 압축 전 120 기준이면 목표 72 ≥ 남은 60 이라
        // 지울 것이 없다 — 채점도 하지 않는다.
        let r = results_of(&mut d, vec![compress(0.6)]);
        assert!(matches!(r[..], [CommandResult::Ok]), "{r:?}");
        assert_eq!(
            registry.len(),
            1,
            "이미 예산 안이면 stage 를 안 만든다 (채점이 비싼 쪽이다)"
        );

        // 반면 진짜로 조이는 예산은 그대로 통과한다 — 0.25*120 = 30 < 60.
        assert!(is_accepted(&results_of(&mut d, vec![compress(0.25)])[0]));
        assert_eq!(registry.len(), 2, "조이는 예산은 여전히 submit 된다");
    }

    /// 분모는 디코드가 붙인 만큼 **자라고**, 압축이 재번호해도 **줄지 않는다**.
    ///
    /// mutation-proof: `observe_context` 에서 `saturating_sub` 대신 `pos` 를 그대로 대입하면
    /// 압축 뒤 분모가 30 으로 떨어져, 아래 0.55 가 목표 66 → 120 미만이라 submit 돼 버린다.
    #[test]
    fn the_denominator_grows_with_decode_and_survives_compaction() {
        let (mut d, registry, h) = make_dispatcher();
        results_of(&mut d, vec![compress(0.25)]); // 0.25*120 = 30
        assert_eq!(registry.len(), 1);
        h.with_cache_mut(|c| c.set_current_pos(30)); // stage 가 압축했다
        d.finalize_results(); // 실제 루프처럼 압축 직후 표집한다 (decode_loop:362)
        h.with_cache_mut(|c| c.advance_pos(90)); // 디코드가 90 토큰 더 붙였다 → 남은 120

        // 문맥은 120 + 90 = 210 토큰을 만들었다. 압축이 그 사실을 지우지 않는다.
        let r = results_of(&mut d, vec![compress(0.6)]);
        assert!(matches!(r[..], [CommandResult::Ok]), "{r:?}");
        assert_eq!(
            registry.len(),
            1,
            "0.6*210 = 126 ≥ 남은 120 — 지울 것이 없다"
        );

        assert!(is_accepted(&results_of(&mut d, vec![compress(0.5)])[0]));
        assert_eq!(registry.len(), 2, "0.5*210 = 105 < 120 — 조인다");
    }

    /// `current_pos` 가 0 이면 압축이 아니라 **새 시퀀스**다 — 압축은 목표를 1 로 바닥치므로
    /// 0 에 닿을 수 없다. 분모를 안 비우면 다음 대화가 이전 대화 길이를 물려받는다.
    #[test]
    fn an_empty_cache_resets_the_denominator() {
        let (mut d, registry, h) = make_dispatcher();
        results_of(&mut d, vec![compress(0.5)]);
        assert_eq!(registry.len(), 1);
        h.with_cache_mut(|c| c.set_current_pos(0)); // 새 시퀀스
        d.dispatch(vec![]); // 표집
        h.with_cache_mut(|c| c.advance_pos(40)); // 새 프리필 40 토큰

        assert!(is_accepted(&results_of(&mut d, vec![compress(0.25)])[0]));
        assert_eq!(
            registry.len(),
            2,
            "0.25*40 = 10 < 40 — 새 문맥 기준으로 조인다"
        );
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
