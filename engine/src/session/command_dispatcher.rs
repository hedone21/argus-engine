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

/// The smallest reduction, as a fraction of the resident length, that is worth starting a
/// decision for. A directive asking for less is skipped in [`CommandDispatcher::submit_compress`]
/// before any stage is built.
///
/// **0.20.** The ceiling is set by the tightest real directive that must still get through: over
/// the measured camera 8K cell's twelve decisions the smallest requested reduction is 27.2 %
/// (decision #12, `target_len` 816 against a 1121-token layer mean), so a floor above that would
/// skip all twelve. 0.20 keeps a 7.2-point margin under it while still catching the camera 1K
/// cell's 15.0 % directive, the one measured skip target in the three cells (1K 1 / 4K 0-1 / 8K 0).
///
/// This is NOT `crate::kv::MIN_EVICT_TOKENS`: that constant is an absolute token count on the
/// cache-manager path and is not wired into the pool path at all.
const MIN_EVICT_FRACTION: f32 = 0.20;

/// The gate's floor in tokens: the reduction a directive has to reach before an eviction is
/// submitted for it, as a truncated share of the resident length.
///
/// This is the one place that arithmetic lives. The gate in `handle_kv_compress` and T6's boundary
/// derivation both call it, so changing the threshold moves the tested boundary with it. While the
/// two spelled it out separately, a mutation that changed only the gate (truncation → `.ceil()`)
/// left every acceptance criterion in ticket 008 green while T6 quietly probed a point one token
/// off the real boundary — task 9 asks for the boundary to be DERIVED from the gate, and a copy is
/// not a derivation (ticket 008, 5차 수리).
///
/// Truncating, not rounding: this is a floor the ask must clear, and the truncation is what makes
/// the largest still-skipped fraction `(floor(resident * F) - 1) / resident`, the value
/// [`skip_frac_display`] then has to truncate rather than round to keep the skip line honest.
fn evict_floor_tokens(resident: usize) -> usize {
    (resident as f32 * MIN_EVICT_FRACTION) as usize
}

/// The reduction fraction as the skip line prints it: truncated at three decimals, never rounded.
///
/// The gate's own threshold is [`evict_floor_tokens`], a truncation, so
/// the largest reduction that still skips is `(floor(resident * F) - 1) / resident` — which for a
/// resident length of a few thousand tokens sits just under the floor rather than well under it
/// (4096 → 0.19971, 8192 → 0.19983). Handing that to `{:.3}` directly ROUNDS it to `0.200` and
/// leaves a line that says it skipped a directive whose fraction equals the floor it was skipped
/// against. Truncating keeps the printed pair honest: a skipped directive always prints
/// `frac < floor`.
///
/// The `Partial` reason's `{:.1}` percentage goes through here too. Task 8″ first exempted it —
/// "a different precision, it cannot collide with the floor it quotes" — and that was measured to
/// be false: the largest still-skipped ask renders `20.0%` for every resident length from 2000 up
/// (2000, 4096 and 8192 all print it; 1999 is the last that does not), so the sentence the Manager
/// receives reads "the requested reduction was 20.0% … below the 20.0% floor" — the same
/// self-contradiction in the reason that the truncation removes from the line
/// (ticket 008, 7차 수리 정정). Narrowing to f32 first does not save it either: the raw f32 value
/// prints `20.0%` at all three of those residents. The percentage and the printed `frac` now come
/// from one truncated value, so they can never disagree about the same directive.
///
/// The ratio is taken in f64 and narrowed at the end. Truncation has no slack to absorb a
/// representation error: `1 - 102/120` is exactly 0.15, but in f32 it lands at 0.14999998 and would
/// print `frac=0.149` for the very directive whose reason string calls it 15.0 %. In f64 the error
/// (~1e-16) is orders below the 1/resident by which a skipped fraction clears the floor, so the
/// truncation only ever removes decimals, never a unit of the last one.
fn skip_frac_display(target_len: usize, resident: usize) -> f32 {
    let frac = 1.0 - target_len as f64 / resident as f64;
    ((frac * 1000.0).floor() / 1000.0) as f32
}

/// The skip line itself, whole: the pinned format literal, its four fields and the truncated
/// display fraction live here and nowhere else, and the gate only prints what this returns.
///
/// Task 8″ first put the fraction alone behind [`skip_frac_display`], and a mutation pass showed
/// that was not enough: swapping the `eprintln!` argument back to the raw fraction brought the
/// `frac=0.200 floor=0.200` line back while every check in ticket 008 still passed — T6 called the
/// value helper directly, and the host schedule never reaches the `resident >= 2000` range where
/// truncating and rounding differ. With the whole line behind one function T6 asserts on the
/// rendered line, so that mutation has no argument left to swap (ticket 008, 4차 수리 ②).
///
/// The literal is a contract, not a detail: ticket 011 greps the on-device logs for it and 008's
/// acceptance criterion 1′ pins it with `grep -qF`. Changing it means raising the ticket.
fn skip_log_line(target_len: usize, resident: usize) -> String {
    format!(
        "[kv-compress] skip target_len={} resident={} frac={:.3} floor={:.3}",
        target_len,
        resident,
        skip_frac_display(target_len, resident),
        MIN_EVICT_FRACTION
    )
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
        if self.kv_handles.is_empty() {
            return None;
        }
        // Read back across every layer, rounded up — the unit `p.tokens_before` is in (it is the
        // `resident` the submit computed the same way). Reading layer 0 alone made a per-layer
        // winner's compression look like a miss: camera 8K decision #12 landed the 28-layer mean
        // exactly on its 816-token budget while layer 0 still showed 1061, and this answered the
        // Manager `Partial`.
        let after =
            crate::kv::layer_mean_resident(self.kv_handles.iter().map(|h| h.resident_tokens()));
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
    /// - **batch fold**: 한 스텝에 복수 `KvCompress` 가 도착하면 (프리필 동안 적체된 지시가
    ///   한 스텝에 함께 도착하는 등), `RestoreDefaults` 로 구분된 구간별로 가장 조인(최소) budget 1건만
    ///   `submit_compress` 로 stage 를 제출하고, 나머지는 stage 없이 `CommandResult::Ok` 로 접는다.
    pub fn dispatch(&mut self, cmds: Vec<EngineCommand>) -> &LoopControl {
        // 이 호출이 곧 「디코드 한 스텝」이다 (decode_loop 가 명령 유무와 무관하게 매 step
        // 부른다) — 문맥 길이를 여기서 표집한다. 이번 step 이 제출할 압축보다 **먼저** 봐야
        // 그 압축의 분모가 압축 전 길이가 된다.
        self.observe_context();
        // transient(매 step 새로 결정되는) 필드만 초기화 — sticky(last_evict_ratio)는 carry.
        self.control.suspended = false;
        self.control.resumed = false;
        self.control.restore_defaults = false;

        let mut is_folded = vec![false; cmds.len()];

        if self.can_compress() {
            // RestoreDefaults 기준으로 구간을 나눈다 (RestoreDefaults 가 last_evict_ratio 를 재무장하므로).
            let mut seg_start = 0;
            for i in 0..=cmds.len() {
                if i == cmds.len() || matches!(cmds[i], EngineCommand::RestoreDefaults) {
                    let seg = seg_start..i;
                    seg_start = i + 1;

                    let compress_indices: Vec<(usize, f32)> = seg
                        .filter_map(|idx| match cmds[idx] {
                            EngineCommand::KvCompress { budget } => Some((idx, budget)),
                            _ => None,
                        })
                        .collect();

                    if compress_indices.len() > 1 {
                        let mut chosen_idx = compress_indices[0].0;
                        let mut min_budget = compress_indices[0].1;
                        for &(idx, budget) in &compress_indices[1..] {
                            if budget.total_cmp(&min_budget).is_le() {
                                chosen_idx = idx;
                                min_budget = budget;
                            }
                        }

                        let mut folded_budgets = Vec::new();
                        for &(idx, budget) in &compress_indices {
                            if idx != chosen_idx {
                                is_folded[idx] = true;
                                folded_budgets.push(budget);
                            }
                        }

                        let folded_str = folded_budgets
                            .iter()
                            .map(|b| format!("{:.3}", b))
                            .collect::<Vec<_>>()
                            .join(" ");

                        eprintln!(
                            "[dispatch] {} kv.compress directives arrived in one step; folded to budget={:.3} ({} answered Ok)",
                            compress_indices.len(),
                            min_budget,
                            folded_str,
                        );
                    }
                }
            }
        }

        self.last_results = Vec::with_capacity(cmds.len());
        self.pending_compress = None;
        for (i, cmd) in cmds.iter().enumerate() {
            self.result_idx = self.last_results.len();
            let r = if is_folded[i] {
                CommandResult::Ok
            } else {
                self.apply(cmd)
            };
            self.last_results.push(r);
        }

        &self.control
    }

    /// 현재 구성에서 KV 압축이 가능한 상태인지 판정한다.
    fn can_compress(&self) -> bool {
        !self.kv_handles.is_empty() && (self.aperturb.is_some() || self.cache_manager.is_some())
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
                crate::yield_policy::restore_default_yield_every();
                CommandResult::Ok
            }

            // ④ gpu.share → yield_policy setter (tickets/015). 의도(foreground)를 손잡이 값
            // (EVERY)으로 번역하는 것은 `every_for_share` 뿐이다 — dispatcher 는 사다리를 모른다.
            EngineCommand::GpuShare { foreground } => {
                crate::yield_policy::set_yield_every(crate::yield_policy::every_for_share(
                    *foreground,
                ));
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
        // A repeated budget names a state, not an action: it is answered `Ok` below when the
        // cache is still within it (the `target_len >= resident` guard), and re-applied when
        // decode has grown the cache back past it. Short-circuiting every repeat, as this once
        // did, made a budget hold only at the instant it was sent — a Manager parked at its
        // floor watched the cache regrow from 1.4K to 4K tokens (idle 8K, 2026-09-02). With the
        // denominator fixed at the uncompressed length there is nothing for a re-application to
        // compound.
        if self.kv_handles.is_empty() {
            return CommandResult::Rejected {
                reason: "no kv cache handles are registered".to_string(),
            };
        }
        // The budget is a fraction of what this context would occupy **uncompressed**, not of
        // what is resident now. Against the resident length a repeated budget would compound;
        // against this one it restates, which is what makes the command idempotent.
        let resident =
            crate::kv::layer_mean_resident(self.kv_handles.iter().map(|h| h.resident_tokens()));
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
        // The directive asks for a reduction too small to pay for the decision it would start.
        // Everything downstream of this line is the expensive half — the pool re-reads K/V to the
        // host, rebuilds the observation window, plans every candidate and scores them — and on
        // the measured camera 1K cell one decision of four asked for 15 % and cost 0.221 s of
        // stage time to deliver it. Skipping here rather than inside the stage is what makes it
        // free: `AperturbSelectStage`'s own `target_len >= resident` guard already sits past
        // `take_inner` and the score sync.
        //
        // Unlike the branch above, "the state the directive names holds" is FALSE here — the
        // cache is outside the budget and stays outside it — so the honest answer is `Partial`,
        // in the same denominator `compress_outcome` reports in.
        //
        // Like the `target_len >= resident` branch above it, this one answers BEFORE capability is
        // established: the "kv cache manager is not configured" `Rejected` sits further down, so an
        // assembly with neither a pool nor a cache manager (the standard/argus-cli loop) answers a
        // sub-floor directive `Partial` where it used to answer `Rejected`, and prints the skip line
        // below in a session with no compression path wired at all. That widens a wart the `Ok`
        // branch above already had rather than inventing one, and the Manager only logs either
        // answer; reordering the two is a contract change ticket 008 was not asked to make
        // (008 result §14-2).
        let asked_reduction = resident - target_len;
        if asked_reduction < evict_floor_tokens(resident) {
            // The same truncation the printed line goes through, not the raw fraction: at any
            // resident length from 2000 up the largest still-skipped ask rounds to the floor
            // itself and the reason contradicts itself (ticket 008, 7차 수리 정정).
            let frac = skip_frac_display(target_len, resident);
            // The whole line — literal, fields and the truncated display fraction — is
            // `skip_log_line`'s, which is the string T6 asserts on. Nothing is formatted here.
            eprintln!("{}", skip_log_line(target_len, resident));
            // `last_evict_ratio` is deliberately NOT recorded. It means "the budget an eviction
            // was submitted for in this window", and nothing was submitted; recording it would
            // also suppress the same budget on the next tick, when decode has regrown the cache
            // and the identical value now asks for a reduction that clears the floor — the
            // regrow-unchecked regression the doc comment on that field describes. Re-evaluating
            // every tick costs one subtraction and one compare, both ahead of any stage.
            //
            // Note the field currently has no reader in any conditional — the value-comparing
            // short-circuit its doc comment describes was removed (see this function's header), so
            // today the choice is inert either way. It is insurance for that short-circuit's
            // return, which is exactly when recording an unsubmitted budget would freeze it.
            return CommandResult::Partial {
                achieved: if self.logical_len == 0 {
                    0.0
                } else {
                    resident as f32 / self.logical_len as f32
                },
                reason: format!(
                    "eviction declined: the requested reduction was {:.1}% of the \
                     resident length, below the {:.1}% floor",
                    frac * 100.0,
                    MIN_EVICT_FRACTION * 100.0
                ),
            };
        }
        // The configured-technique stage takes a fraction, which is what it can act on. Converting
        // here keeps the contract's denominator at the boundary. The pool stage takes the count and
        // forms its fraction when it runs: two directives in one step would otherwise both take
        // theirs against the length at submit time, and the second would apply a stale fraction to
        // a cache the first had already compacted.
        //
        // `--eviction-target-ratio` still means what it always meant — `EvictionHandler` prefers
        // `ctx.target_ratio` over its own configured one (`eviction_handler.rs`), so the two never
        // meet. What DID move with ticket 008 is this manager-directed ratio's denominator: it is
        // now the layer mean, while the handler multiplies it back out by layer 0's `current_pos`.
        // The round trip is exact only while the layers agree, and on a ragged cache layer 0 is the
        // longest, so the handler then keeps slightly MORE than the mean asked for. The path this
        // ticket measures is the pool one below, which takes `target_len` directly and is
        // unaffected; nothing pins this branch in either direction (008 result §14-1).
        let target_ratio = target_len as f32 / resident as f32;
        // The contract names a budget, not a technique. When a candidate pool is configured the
        // engine picks the technique itself; otherwise it applies the one the CLI configured.
        let stage: Arc<dyn crate::pipeline::PipelineStage> = match self.aperturb.as_ref() {
            Some((selector, q_rows, prefill_attn)) => Arc::new(AperturbSelectStage::new(
                self.kv_handles.clone(),
                Arc::clone(selector),
                Arc::clone(q_rows),
                target_len,
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

    /// A repeated budget names a state. While the cache is within it — here the first
    /// compaction is simulated by moving the cursor under the target, since nothing runs
    /// `KvMutate` in these tests — the repeat is `Ok` and submits nothing; once decode has
    /// grown the cache back past it, the same value submits again. Mutation-proof: restoring
    /// the old `last_evict_ratio == budget` short-circuit keeps the registry at 1 on the last
    /// step, which is the cache regrowing unchecked under a Manager parked at its floor.
    #[test]
    fn a_repeated_budget_reapplies_only_when_the_cache_has_regrown_past_it() {
        let (mut d, registry, h) = make_dispatcher();
        assert_eq!(registry.len(), 0);
        assert!(is_accepted(&results_of(&mut d, vec![compress(0.5)])[0]));
        assert_eq!(registry.len(), 1, "첫 압축 → OneShot 1개 submit");
        d.dispatch(vec![]);
        assert_eq!(registry.len(), 1, "빈 batch — 재submit 없음");
        let full = h.current_pos();
        h.with_cache_mut(|c| c.set_current_pos(full / 2));
        let again = results_of(&mut d, vec![compress(0.5)]);
        assert!(matches!(again[..], [CommandResult::Ok]), "{again:?}");
        assert_eq!(
            registry.len(),
            1,
            "예산 안 — 같은 budget 반복은 재submit 없음"
        );
        h.with_cache_mut(|c| c.set_current_pos(full));
        let regrown = results_of(&mut d, vec![compress(0.5)]);
        assert!(is_accepted(&regrown[0]), "{regrown:?}");
        assert_eq!(
            registry.len(),
            2,
            "예산 밖으로 다시 자람 — 같은 budget 이 재적용된다"
        );
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
    /// 압축 뒤 분모가 30 으로 떨어지고 디코드 90 을 더해 210 이 아니라 **120** 이 된다. 그러면
    /// 아래 `compress(0.6)` 이 목표 `0.6*120 = 72 < 120` 이라 submit 되고, 이 테스트에서는
    /// stage 를 아무도 돌리지 않으므로 답이 `Ok` 가 아니라 `Partial` 로 뒤집혀 그 블록의
    /// 단정이 깨진다(실측: 뮤테이션 뒤 `compress(0.6)` 의 `matches!(… Ok)` 에서 패닉).
    /// ⇒ 뮤테이션을 잡는 것은 `compress(0.6)` 블록이고, **마지막 예산 줄은 아예 도달하지도
    /// 않는다** — 그래서 그 줄을 0.5 에서 0.4 로 옮겨도(티켓 008 의 감축률 하한 때문) 판별력이
    /// 하나도 줄지 않는다.
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

        // 0.4 (구 0.5): 조이는 예산이라는 점은 같고, 티켓 008 의 감축률 하한
        // (`MIN_EVICT_FRACTION` = 0.20) 위에 있다. 0.5 는 목표 105/상주 120 = 감축률 12.5 % 라
        // 새 게이트에 걸려 스킵된다 — 이 테스트가 세우는 것은 **분모**(210)이지 하한이 아니므로
        // 예산만 옮긴다. 하한 자체는 T1·T2 가 세운다.
        assert!(is_accepted(&results_of(&mut d, vec![compress(0.4)])[0]));
        assert_eq!(
            registry.len(),
            2,
            "0.4*210 = 84 < 120 이고 감축률 30 % ≥ 20 % — 조인다"
        );
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

    /// `GpuShare` touches `yield_policy` global state directly (tickets/015 T2-c), same as
    /// `yield_policy::tests::LOCK` — serialize against those tests too. No cache manager
    /// needed: `bare_dispatcher` proves the command doesn't route through KV at all.
    #[test]
    fn gpu_share_sets_the_yield_and_restore_releases_it() {
        let _g = crate::yield_policy::TEST_LOCK.lock().unwrap();
        let (mut d, _registry) = bare_dispatcher();

        let r = results_of(&mut d, vec![EngineCommand::GpuShare { foreground: 1.0 }]);
        assert!(matches!(r[..], [CommandResult::Ok]));
        assert_eq!(crate::yield_policy::yield_every(), 2);

        let r = results_of(&mut d, vec![EngineCommand::RestoreDefaults]);
        assert!(matches!(r[..], [CommandResult::Ok]));
        assert_eq!(crate::yield_policy::yield_every(), 0);
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

    /// 한 배치 안의 복수 `KvCompress` 는 가장 조인(최소) budget 하나만 stage 를 submit 하고,
    /// 나머지는 stage 없이 Ok 로 접힌다 (프리필 동안 적체된 복수 지시의 단일 결정 축약).
    #[test]
    fn a_batch_of_budgets_submits_one_stage_for_the_tightest() {
        let (mut d, registry, _h) = make_dispatcher();
        d.dispatch(vec![compress(0.75), compress(0.6), compress(0.5)]);
        assert_eq!(registry.len(), 1, "3건 중 최소 예산 1건만 stage submit");
        assert_eq!(
            d.pending_compress.as_ref().map(|p| p.budget),
            Some(0.5),
            "제출된 stage 의 목표가 0.5 기준이어야 한다"
        );
        assert_eq!(
            d.last_evict_ratio,
            Some(0.5),
            "last_evict_ratio 는 제출된 최소값"
        );
        let r = d.finalize_results();
        assert_eq!(r.len(), 3, "결과 3건 일대일 대응");
        assert!(r.iter().all(is_accepted), "3개 모두 accepted: {r:?}");
    }

    /// 도착 순서가 조임→느슨이어도 최소 budget 하나만 submit 된다.
    #[test]
    fn a_batch_of_budgets_is_order_independent() {
        let (mut d, registry, _h) = make_dispatcher();
        d.dispatch(vec![compress(0.5), compress(0.75)]);
        assert_eq!(registry.len(), 1, "역순이어도 0.5 하나만 submit");
        assert_eq!(
            d.pending_compress.as_ref().map(|p| p.budget),
            Some(0.5),
            "제출된 stage 의 목표가 0.5 기준"
        );
        assert_eq!(d.last_evict_ratio, Some(0.5));
        let r = d.finalize_results();
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(is_accepted), "모두 accepted: {r:?}");
    }

    /// **T1 (ticket 008).** A directive whose requested reduction is under
    /// [`MIN_EVICT_FRACTION`] never reaches a stage.
    ///
    /// `logical_len` and `resident` are both 120 here, so `budget=0.85` resolves to
    /// `target_len = 102` and asks for `1 - 102/120 = 15.0 %` — the camera 1K cell's first
    /// directive, the one measured skip target of the three cells (it cost 0.221 s of stage time
    /// to remove 77 of 512 tokens). The answer is `Partial`, not `Ok`: unlike the
    /// `target_len >= resident` branch above it, the state the directive names does NOT hold.
    ///
    /// Mutation-proof: dropping the gate submits a stage and the `registry.len()` assert fails;
    /// answering `Ok` instead of `Partial` fails the `matches!`. `is_accepted` is deliberately not
    /// used — it is `!is_rejected`, so a `Partial` would pass it.
    #[test]
    fn a_directive_below_the_evict_fraction_is_skipped() {
        let (mut d, registry, _h) = make_dispatcher();
        let r = results_of(&mut d, vec![compress(0.85)]);
        assert_eq!(
            registry.len(),
            0,
            "a 15 % ask is below the 20 % floor — no stage may be submitted"
        );
        assert!(
            matches!(r[..], [CommandResult::Partial { .. }]),
            "a skipped directive is answered Partial, not Ok: {r:?}"
        );
        let CommandResult::Partial { reason, .. } = &r[0] else {
            unreachable!()
        };
        // Pinned whole: the reason is what the Manager logs, and a `\`-continued literal is one
        // `cargo fmt` away from carrying the source indentation into it.
        assert_eq!(
            reason,
            "eviction declined: the requested reduction was 15.0% of the resident length, \
             below the 20.0% floor"
        );
        assert!(
            !reason.contains("tokens would have been removed"),
            "the token-count reason belongs to a different gate: {reason}"
        );
        assert_eq!(
            d.last_evict_ratio, None,
            "nothing was submitted, so no budget is recorded"
        );
    }

    /// **T2 (ticket 008).** A directive just above the floor still submits.
    ///
    /// `budget=0.73` against `logical_len = resident = 120` resolves to `target_len = 87`, an ask
    /// of `1 - 87/120 = 27.5 %`. That is the camera 8K cell's tightest real directive (#12 asked
    /// 27.2 %) and the reason [`MIN_EVICT_FRACTION`] cannot be raised: a floor above it would skip
    /// all twelve of that cell's decisions.
    ///
    /// Mutation-proof: raising the constant to 0.30 makes this ask fall under the floor and the
    /// `registry.len()` assert fails.
    #[test]
    fn a_directive_at_the_eight_k_margin_still_submits() {
        let (mut d, registry, _h) = make_dispatcher();
        let r = results_of(&mut d, vec![compress(0.73)]);
        assert_eq!(
            registry.len(),
            1,
            "a 27.5 % ask clears the 20 % floor — the stage must be submitted"
        );
        assert!(is_accepted(&r[0]), "{r:?}");
        assert_eq!(d.last_evict_ratio, Some(0.73));
    }

    /// **T6 (ticket 008, task 8″).** The skip line's `frac` never prints at the floor.
    ///
    /// The gate's threshold is [`evict_floor_tokens`], a truncation, so
    /// the tightest ask it still skips removes exactly one token less than that threshold. On a
    /// resident length of a few thousand tokens that fraction lands at 0.1995–0.1998, which
    /// `{:.3}` ROUNDS up to `0.200` — the very floor the directive was skipped against, leaving a
    /// self-contradicting line in the log. [`skip_frac_display`] truncates instead, so the printed
    /// pair stays strictly ordered.
    ///
    /// The boundary is derived by calling the gate's own [`evict_floor_tokens`] rather than by
    /// re-spelling its arithmetic here, and each resident is checked to BE a boundary (removing one
    /// more token no longer skips) so the case under test cannot silently drift away from the
    /// floor. Copying the threshold instead is what let a gate-only mutation pass unnoticed
    /// (ticket 008, 5차 수리).
    ///
    /// The assertion reads the fields back out of the line [`skip_log_line`] renders, not out of
    /// [`skip_frac_display`]: a helper that truncates is worth nothing once the line stops calling
    /// it, and that is precisely the mutation an earlier round of this ticket could not catch.
    ///
    /// Mutation-proof, and only just: make the line format the fraction untruncated and 4096
    /// (0.19971) and 8192 (0.19983) round up to `0.200`, failing here. Neither 1178 nor 2000 can
    /// be relied on to catch that. 1178 (0.19864) stays under the floor either way — 0.199 rounded
    /// against 0.198 truncated — and 2000 (0.19950) sits exactly on the rounding boundary, so
    /// whether it reaches `0.200` depends on how the untruncated value got there: in f64 it does,
    /// but narrowed through f32 first it lands at 0.19949999 and prints 0.199. Only 4096 and 8192
    /// round up to the floor in every form, which is why all four residents are here rather than
    /// one.
    ///
    /// Two things the four boundary residents cannot pin on their own, asserted directly at the
    /// top (ticket 008, 5차 수리): the threshold's VALUE and rounding direction, because the
    /// boundary derivation reduces to the identity `floor_tokens - 1 < floor_tokens` and moves
    /// with the threshold instead of holding it still; and the `f64` ratio, because narrowing it
    /// changes the third decimal only where the true fraction sits near a display boundary, which
    /// none of the four boundary points do.
    ///
    /// Two more (ⓒ and ⓓ, 6차 수리): the two integers are asserted to carry what they are labelled
    /// with, since the fraction is computed from the arguments rather than read back and a swapped
    /// pair renders a correct `frac` beside two wrong integers; and the gate's comparison operator
    /// is exercised through the dispatcher at the token either side of the floor, since the
    /// boundary derivation above is an identity that a `<` → `<=` change carries along with it.
    #[test]
    fn the_skip_line_frac_never_prints_at_the_floor() {
        // ⓐ The threshold's value and the direction it rounds.
        //
        // Deriving the boundary from `evict_floor_tokens` keeps this test and the gate on one
        // arithmetic, but what that derivation can assert -- `resident - target_len <
        // floor_tokens` for `target_len = resident - (floor_tokens - 1)` -- is
        // `floor_tokens - 1 < floor_tokens`, true for every threshold. Swap the truncation for
        // `.ceil()` and the probed boundary simply moves with it: all four residents below still
        // print 0.199 against 0.200 and nothing fires. So pin the number itself.
        //
        // 1178 * 0.20 = 235.6: this one point separates the truncation (235) from BOTH `.ceil()`
        // and `.round()` (236).
        assert_eq!(
            evict_floor_tokens(1178),
            235,
            "the gate's floor truncates: 1178 * 0.20 = 235.6 -> 235, not 236"
        );
        // A second point, because one is thin in a way 1178 cannot show. 4096 * 0.20 = 819.2 has
        // a fractional part below .5, so `.round()` agrees with the truncation here and only
        // `.ceil()` parts from it -- a second regime for the rounding mode. More to the point, it
        // is the only one of the two that moves if a clamp is ever grafted onto the floor (a
        // `.max`/`.min`, or `crate::kv::MIN_EVICT_TOKENS` finding its way back in, which task B-7
        // forbids): any absolute clamp between 236 and 819 leaves 1178 reading 235 and is
        // invisible there. 4096 is also the resident range of ticket 011's 4K cell, where a
        // `.ceil()` gate would change which directives are skipped at all.
        assert_eq!(
            evict_floor_tokens(4096),
            819,
            "the gate's floor truncates: 4096 * 0.20 = 819.2 -> 819, not 820"
        );

        // ⓑ The ratio has to be taken in f64, and only the rendered line can say so.
        //
        // T1's own directive is the case: `target_len` 102 against 120 resident tokens, the 15.0 %
        // ask the reason string quotes. In f64 `1 - 102/120` lands a hair above 0.15 and truncates
        // to 0.150; narrowed to f32 it lands at 0.14999998 and truncates to 0.149, so the line
        // would call a 15.0 % directive 0.149 -- the same disagreement between the two renderings
        // that task 8" exists to remove. None of the four boundary residents below notices: at
        // every one of them the third decimal reads 0.198/0.199 in either width.
        let t1_line = skip_log_line(102, 120);
        assert!(
            t1_line.contains("frac=0.150"),
            "the reduction fraction must be taken in f64: 1 - 102/120 is 0.15 and prints \
             frac=0.150, where f32's 0.14999998 truncates to frac=0.149. Line: {t1_line:?}"
        );

        // 1178 is the camera 1K cell's layer mean; the other three bracket the 011 cells, whose
        // residents are exactly the range where rounding reaches the floor.
        for resident in [1178usize, 2000, 4096, 8192] {
            let floor_tokens = evict_floor_tokens(resident);
            assert!(
                floor_tokens >= 2,
                "resident {resident} has no boundary to test"
            );
            // One token under the threshold: the largest reduction — i.e. the smallest
            // `target_len` — the gate still skips.
            let target_len = resident - (floor_tokens - 1);
            assert!(
                resident - target_len < floor_tokens,
                "resident {resident}: target_len {target_len} must still be skipped"
            );
            assert!(
                resident - (target_len - 1) >= floor_tokens,
                "resident {resident}: removing one more token must clear the floor, \
                 or target_len {target_len} is not the boundary"
            );

            // The rendered line, not the arithmetic behind it: read the two fields back out of
            // exactly the bytes the gate prints.
            let line = skip_log_line(target_len, resident);
            assert!(
                line.starts_with("[kv-compress] skip "),
                "the skip line lost its prefix: {line:?}"
            );
            let field = |name: &str| -> f64 {
                line.split_whitespace()
                    .find_map(|f| f.strip_prefix(name))
                    .unwrap_or_else(|| panic!("no `{name}` field in {line:?}"))
                    .parse()
                    .unwrap_or_else(|e| panic!("unparsable `{name}` field in {line:?}: {e}"))
            };
            let int_field = |name: &str| -> usize {
                line.split_whitespace()
                    .find_map(|f| f.strip_prefix(name))
                    .unwrap_or_else(|| panic!("no `{name}` field in {line:?}"))
                    .parse()
                    .unwrap_or_else(|e| panic!("unparsable `{name}` field in {line:?}: {e}"))
            };

            // ⓒ The two integers carry the values they are labelled with (ticket 008, 6차 수리).
            //
            // The fraction assertions below are blind to the pair being swapped: `frac` is
            // computed from the arguments, not from what was printed, so transposing the two
            // `format!` arguments renders `target_len=1178 resident=943` with a perfectly correct
            // `frac` beside it and every other check in this ticket stays green. Ticket 011 greps
            // exactly these two integers off the device to recompute the gate's margin, so a
            // silent swap would move numbers into an on-device analysis rather than break a build.
            assert_eq!(
                int_field("target_len="),
                target_len,
                "the skip line's `target_len=` field must carry target_len: {line:?}"
            );
            assert_eq!(
                int_field("resident="),
                resident,
                "the skip line's `resident=` field must carry resident: {line:?}"
            );

            // ⓔ The `floor=` field is the floor actually in force (ticket 008, 7차 수리).
            //
            // Nothing else in this ticket reads that field for its VALUE. The comparison below
            // only asks that it sit above the printed fraction, which any larger constant does,
            // so handing `skip_log_line` a literal `0.25f32` in place of `MIN_EVICT_FRACTION`
            // leaves this test, criterion 5's (vi)/(vi′) regexes and every grep in criterion 1′
            // green — 1′ pins the format string, and a format string does not name its
            // arguments. Ticket 011 reads `floor=` off the device to recover the floor that run
            // enforced and recompute each directive's margin against it, so a field that has
            // drifted from the constant puts a wrong number into an on-device analysis instead
            // of breaking a build.
            //
            // Read back at the width it is printed: `{:.3}` of `0.20f32` is "0.200", which parses
            // back to exactly `MIN_EVICT_FRACTION`. That width is also this assertion's limit —
            // a floor within 0.0005 of F renders the same three decimals and cannot be told apart
            // from the rendered line by anyone, this test included.
            let printed_floor: f32 = line
                .split_whitespace()
                .find_map(|f| f.strip_prefix("floor="))
                .unwrap_or_else(|| panic!("no `floor=` field in {line:?}"))
                .parse()
                .unwrap_or_else(|e| panic!("unparsable `floor=` field in {line:?}: {e}"));
            assert_eq!(
                printed_floor, MIN_EVICT_FRACTION,
                "the skip line's `floor=` field must carry the floor the gate enforces, \
                 not a literal of its own: {line:?}"
            );

            let shown_frac = field("frac=");
            let shown_floor = field("floor=");
            assert!(
                shown_frac < shown_floor,
                "resident {resident}: skipped target_len {target_len} prints \
                 frac={shown_frac:.3} floor={shown_floor:.3} — a skipped directive must print \
                 strictly below its floor. Line: {line:?}"
            );

            // ⓗ The reason's percentage is truncated too (ticket 008, 7차 수리 정정).
            //
            // Task 8″ exempted the `Partial` reason on the grounds that `{:.1}` is a different
            // precision and could not collide with the floor it quotes. Measured, that is false:
            // the raw fraction renders `20.0%` at every resident length from 2000 up — 2000,
            // 4096 and 8192 all do, in f32 and in f64 alike — so the sentence the Manager reads
            // says "the requested reduction was 20.0% of the resident length, below the 20.0%
            // floor". That is the same self-contradiction the truncated `frac` exists to remove,
            // in the field a human actually reads.
            //
            // It has to be driven through the dispatcher: the reason is formatted inside the gate
            // and no helper returns it. The budget lands the gate on this same boundary
            // `target_len` (`logical_len == resident` for a single fresh handle), which the
            // second assertion re-confirms by pinning the percentage to the line's own fraction.
            //
            // Mutation-proof: put the raw fraction back in the reason and 2000, 4096 and 8192 all
            // print `20.0% … below the 20.0% floor` and fail the first assertion; 1178 prints
            // 19.9 % against the line's `frac=0.198` and fails the second. 512 and 1999, where
            // raw and truncated agree, would catch neither — the four residents here are the
            // ones that do.
            let budget = ((target_len as f64 + 0.5) / resident as f64) as f32;
            let registry = Arc::new(PipelineRegistry::new());
            let mut d = CommandDispatcher::new(
                Arc::clone(&registry),
                vec![make_handle(resident)],
                Some(make_cm()),
                Arc::new(Mutex::new(None)),
            );
            let r = results_of(&mut d, vec![compress(budget)]);
            let [CommandResult::Partial { reason, .. }] = &r[..] else {
                panic!("resident {resident}: a sub-floor directive is answered Partial: {r:?}");
            };
            assert_eq!(
                registry.len(),
                0,
                "resident {resident}: nothing may be submitted"
            );
            let pct: Vec<f64> = reason
                .split_whitespace()
                .filter_map(|w| w.strip_suffix('%'))
                .map(|w| {
                    w.parse().unwrap_or_else(|e| {
                        panic!("unparsable percentage in reason {reason:?}: {e}")
                    })
                })
                .collect();
            let [asked_pct, floor_pct] = pct[..] else {
                panic!("the reason must quote exactly two percentages: {reason:?}");
            };
            assert!(
                asked_pct < floor_pct,
                "resident {resident}: the reason says the reduction was {asked_pct}% and that \
                 this is below the {floor_pct}% floor — a skipped directive's reason must not \
                 contradict itself: {reason:?}"
            );
            assert_eq!(
                format!("{asked_pct:.1}"),
                format!("{:.1}", shown_frac * 100.0),
                "resident {resident}: the reason quotes a different fraction than the line \
                 prints: {reason:?} vs {line:?}"
            );
        }

        // ⓓ The gate's comparison operator, driven through the dispatcher (ticket 008, 6차 수리).
        //
        // Everything above derives its boundary from `evict_floor_tokens` and then asserts
        // `resident - target_len < floor_tokens` for `target_len = resident - (floor_tokens - 1)`.
        // That is `floor_tokens - 1 < floor_tokens`: an identity, true whichever way the gate
        // compares. Turning the gate's `<` into `<=` moves the real boundary by one token and
        // nothing above notices, because the point being probed moves with it.
        //
        // So take the two tokens either side of the boundary through `submit_compress` itself and
        // read the outcome off the registry, the way T1 and T2 do. `make_dispatcher` gives one
        // 120-token layer, so `logical_len = resident = 120` and the floor is 24 tokens: an ask of
        // exactly 24 has REACHED the floor and must submit, an ask of 23 has not and must skip.
        assert_eq!(
            evict_floor_tokens(120),
            24,
            "the harness's boundary: 120 * 0.20 = 24 tokens"
        );

        // `budget = 0.805` → `target_len = (120 * 0.805) as usize = 96`, an ask of exactly 24.
        let (mut d, registry, _h) = make_dispatcher();
        let r = results_of(&mut d, vec![compress(0.805)]);
        assert_eq!(
            registry.len(),
            1,
            "an ask of exactly {} tokens has reached the floor and must submit: {r:?}",
            evict_floor_tokens(120)
        );
        assert!(is_accepted(&r[0]), "{r:?}");

        // `budget = 0.81` → `target_len = (120 * 0.81) as usize = 97`, an ask of 23: one token
        // short of the floor.
        let (mut d, registry, _h) = make_dispatcher();
        let r = results_of(&mut d, vec![compress(0.81)]);
        assert_eq!(
            registry.len(),
            0,
            "an ask one token short of the floor must skip: {r:?}"
        );
        let [CommandResult::Partial { reason, .. }] = &r[..] else {
            panic!("a skipped directive is answered Partial: {r:?}");
        };
        assert!(
            reason.starts_with("eviction declined:"),
            "the skip branch's reason: {reason}"
        );
    }

    /// **T3 (ticket 008).** The outcome is read back across every layer, and so is the
    /// submit-time length it is compared against.
    ///
    /// This transposes camera 8K decision #12, where a winner that budgets per layer (pyramidkv
    /// clamps layer 0 at `q - window`) left layer 0 at 1061 while the 28-layer mean landed exactly
    /// on its 816-token budget — and the engine answered the Manager `Partial` for a compression
    /// that had hit its target. Scaled to this harness (`MAX_SEQ = 128`):
    ///
    /// | | layer 0 | layer 1 | mean (round up) |
    /// |---|---|---|---|
    /// | at submit | 120 | 80 (`head_start` 40) | **100** |
    /// | after | 60 | 40 (`head_start` 80) | **50** |
    ///
    /// With `logical_len = 120` and `budget = 0.45`, `50/120 = 0.417 <= 0.45 + 1/120` is `Ok`
    /// while layer 0's `60/120 = 0.500` is not.
    ///
    /// The second half pins the OTHER operand. Nothing runs `KvMutate` in these tests, so a
    /// submission left alone finds the cache exactly as it was, and `compress_outcome` must say
    /// "removed nothing" — which it can only do if `after` and `PendingCompress::tokens_before`
    /// are the same unit. An implementation that averages `after` but leaves the submit-time
    /// `resident` on layer 0 compares 100 against 120, misses that branch, and fails here.
    ///
    /// The third half (T3′, 6차 수리) covers the one `achieved` neither of the first two reaches:
    /// the skip gate's, computed before any stage exists and never passing through
    /// `compress_outcome` at all. It also carries ⓕ (7차 수리): this is the only dispatcher in the
    /// suite where `resident` and `logical_len` differ, so it is the only place that can say which
    /// of the two the declined percentage is quoted against.
    #[test]
    fn the_outcome_reads_back_across_layers_not_from_layer_zero() {
        fn two_layer_dispatcher() -> (CommandDispatcher, Arc<StandardFormat>, Arc<StandardFormat>) {
            let l0 = make_handle(N_TOKENS); // 120 resident
            let l1 = make_handle(N_TOKENS);
            l1.with_cache_mut(|c| c.set_head_starts(&[40])); // 80 resident
            let d = CommandDispatcher::new(
                Arc::new(PipelineRegistry::new()),
                vec![l0.clone(), l1.clone()],
                Some(make_cm()),
                Arc::new(Mutex::new(None)),
            );
            (d, l0, l1)
        }

        // ── half 1: `achieved` is the layer mean ──────────────────────────────────────────
        let (mut d, l0, l1) = two_layer_dispatcher();
        assert_eq!(l0.resident_tokens(), 120);
        assert_eq!(l1.resident_tokens(), 80);
        d.dispatch(vec![compress(0.45)]);
        assert_eq!(
            d.pending_compress.as_ref().map(|p| p.tokens_before),
            Some(100),
            "the submit-time length is the layer mean, not layer 0's 120"
        );
        // Stand in for the compaction the `KvMutate` phase would have applied.
        l0.with_cache_mut(|c| c.set_current_pos(60));
        l1.with_cache_mut(|c| c.set_head_starts(&[80]));
        assert_eq!(l1.resident_tokens(), 40);
        let r = d.finalize_results();
        assert!(
            matches!(r[..], [CommandResult::Ok]),
            "mean 50/120 is inside budget 0.45; layer 0's 60/120 is not: {r:?}"
        );

        // ── half 2: `tokens_before` is the same unit as `after` ───────────────────────────
        let (mut d, _l0, _l1) = two_layer_dispatcher();
        d.dispatch(vec![compress(0.45)]);
        let r = d.finalize_results();
        let [CommandResult::Partial { achieved, reason }] = &r[..] else {
            panic!("an unapplied compression reports Partial: {r:?}");
        };
        assert!(
            (achieved - 100.0 / 120.0).abs() < 1e-6,
            "achieved is the layer mean over logical_len: {achieved}"
        );
        assert!(
            reason.contains("tokens would have been removed"),
            "an untouched cache must read as 'removed nothing', which needs `after` and \
             `tokens_before` in one unit: {reason}"
        );

        // ── half 3 (T3′): the SKIP branch answers in the same unit ────────────────────────
        //
        // The two halves above both run through `compress_outcome`. The gate that ticket 008 adds
        // answers `Partial { achieved }` from its own line, before any stage exists, and that
        // `achieved` is the number the Manager actually receives for every directive the engine
        // declines — on the measured camera 1K cell, one directive in four. Reverting that one
        // numerator to `self.kv_handles[0].resident_tokens()` left all eight of this ticket's
        // acceptance criteria green (adversarial pass, 2026-09-10): nothing read it.
        //
        // `budget = 0.75` against `logical_len = 120` gives `target_len = 90`. The layer mean is
        // 100, so the ask is 10 tokens against a 20-token floor — skipped — while layer 0 alone
        // would read 120 resident and answer a fraction of 1.0 for a cache that is 100 tokens deep
        // on average.
        let (mut d, l0, l1) = two_layer_dispatcher();
        assert_eq!(l0.resident_tokens(), 120);
        assert_eq!(l1.resident_tokens(), 80);
        let r = results_of(&mut d, vec![compress(0.75)]);
        let [CommandResult::Partial { achieved, reason }] = &r[..] else {
            panic!("a sub-floor directive is answered Partial by the gate: {r:?}");
        };
        assert_eq!(
            *achieved,
            100.0f32 / 120.0f32,
            "the skipped directive's `achieved` is the layer mean over logical_len"
        );
        assert_ne!(
            *achieved,
            120.0f32 / 120.0f32,
            "…not layer 0's resident length over logical_len"
        );

        // ── ⓕ (7차 수리): the reason's percentage is taken against `resident` ──────────────
        //
        // The sentence the Manager receives names the same denominator the gate compared
        // against: the resident layer mean. Swapping it for `self.logical_len` is invisible
        // everywhere else in this ticket — T1 and T6ⓓ both run `make_dispatcher`, one layer at
        // 120 tokens, where `resident == logical_len` and the two denominators render the same
        // percentage. They part only once the cache has actually been compressed below the
        // logical length, which is every real directive after the first eviction: here
        // `1 - 90/100` against `1 - 90/120 = 25.0 %`.
        //
        // The `logical_len` form is not merely a different number, it is self-contradicting: it
        // would tell the Manager the ask was "25.0% of the resident length, below the 20.0%
        // floor" — 25 % is not below 20 %, and the directive was skipped precisely because the
        // real share was. This test's dispatcher is the only place in the suite where the
        // two lengths differ, which is why the clause lives here rather than in T6 with the rest
        // of the 7차 pins (T6's `two_layer_dispatcher` equivalent does not exist: the helper is
        // local to this test).
        //
        // The share prints 9.9 %, not 10.0 %: since 7차 수리 정정 the percentage goes through
        // [`skip_frac_display`], and `1 - 90/100` lands a hair BELOW 0.1 in f64
        // (0.09999999999999998), which truncation takes to 0.099 — the same 0.099 the skip line
        // beside it prints for this directive. Truncation can only ever understate the ask, so
        // the reason stays strictly under the floor it quotes, which is the invariant; the price
        // is that an ask sitting exactly on a display boundary reads one decimal light. The
        // alternative — the raw fraction — is what made the reason say "20.0% … below the 20.0%
        // floor" at every resident length from 2000 up (T6ⓗ).
        assert_eq!(
            reason,
            "eviction declined: the requested reduction was 9.9% of the resident length, \
             below the 20.0% floor",
            "the declined share is target_len against the resident layer mean (1 - 90/100), \
             not against logical_len (1 - 90/120 would print 25.0%, above the floor it claims \
             to be under), and it is truncated the way the line beside it is"
        );
    }

    /// 배치 중간에 `RestoreDefaults` 가 있으면 앞뒤를 별개 구간으로 보고 각각 최소를 취한다.
    #[test]
    fn batch_compression_split_by_restore_defaults() {
        let (mut d, registry, _h) = make_dispatcher();
        let r = results_of(
            &mut d,
            vec![
                compress(0.75),
                compress(0.6),
                EngineCommand::RestoreDefaults,
                compress(0.5),
                compress(0.4),
            ],
        );
        assert_eq!(
            registry.len(),
            2,
            "RestoreDefaults 앞뒤 구간에서 각각 1건씩 submit"
        );
        assert_eq!(r.len(), 5);
        assert!(r.iter().all(is_accepted), "모두 accepted: {r:?}");
        assert_eq!(d.last_evict_ratio, Some(0.4));
    }
}
