//! Chat 세션 multi-turn 상태. Phase 4-5-d.
//!
//! [`ChatSession`]은 [`DecodeLoop`]를 owned 보유한다 (1회 build, turn마다 재사용).
//! turn마다 build/drop 금지 — multi-turn KV pos 누적 보존이 핵심 invariant (R1).
//!
//! `/reset` 처리: [`ChatSession::reset`]이 KV cache + score_accumulator +
//! decode_loop.pos를 atomic하게 3단 clear한다 (R2).
//!
//! stats_line 포맷 (D5, G1 enforce):
//! - Standard: `kv_pos={kv_pos}/{max_seq_len} policy={policy_name} evicted_total={evicted_total}`
//! - NonEvicting: `kv_pos={kv_pos}/{max_seq_len} {fragment}` — `fragment` 은 모드 build closure 가
//!   공급한다(kivi: `mode=kivi bits={bits} residual={residual_size}`,
//!   offload: `mode=offload store={mode} prefetch_depth={max_prefetch_depth}`).

use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::buffer::DType;
use crate::capability::quant_attn::QuantAttnBackend;
use crate::format::KVCacheFormat;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::inference::sampling::SamplingConfig;
use crate::kv::cache_manager::CacheManager;
use crate::kv::eviction::stage_registry::StageBackedPolicy;
use crate::kv::standard_format::StandardFormat;
use crate::resilience::sys_monitor::{LinuxSystemMonitor, NoOpMonitor};
use crate::session::DecodeLoopBuilder;
use crate::session::chat::sampler::SharedSamplingConfig;
use crate::session::chat::stop_condition::{ChatStopSlot, ChatStopStage, StopCondition};
use crate::session::chat::stream_stage::{ChatStreamSlot, ChatStreamStage};
use crate::session::command_dispatcher::CommandDispatcher;
use crate::session::decode_loop::DecodeLoop;
use crate::session::decode_loop::{DecodeResult, HasForward};
use crate::session::forward::{
    ModelForward, OffloadForward, QuantWindowForward, alloc_offload_kv_caches,
    alloc_quant_window_kv_caches,
};
use crate::session::pipeline_registry::PipelineRegistry;
use crate::session::resilience_adapter::ResilienceAdapter;

/// `ChatKvMode::Standard` variant inner payload.
///
/// `CacheManager` + `AttentionScoreAccumulator`로 인해 ~376 bytes로 enum 전체가
/// 비대해지는 것을 막기 위해 별도 struct로 추출하고 `Box`로 wrap한다.
pub struct ChatKvModeStandard {
    pub cache_manager: Option<CacheManager>,
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    /// score-based policy (h2o, h2o_plus, d2o)인지 여부.
    pub score_based: bool,
    pub policy_name: String,
    pub target_ratio: f32,
    pub evicted_total: usize,
}

/// chat 모드의 KV-type 분기 (eviction-capable vs not).
///
/// stats_line 포맷 + ensure_capacity 동작이 분기된다. Standard만 eviction
/// (CacheManager)을 자체 관리한다. 그 외 모드(quant-window/offload)는 overflow 시
/// bail (eviction 미지원) — 엔진 chat 계층은 구체 모드를 명명하지 않고, 모드의
/// build closure 가 자신의 `/stats` fragment 를 공급한다(이름-비분기 불변식).
pub enum ChatKvMode {
    Standard(Box<ChatKvModeStandard>),
    /// eviction 없는 모드 일반형. `stats_fragment` 는 build closure 가 만든
    /// `/stats` 줄 조각(예: `mode=kivi bits=4 residual=32`).
    NonEvicting {
        stats_fragment: String,
    },
}

/// Chat 세션. DecodeLoop을 owned 보유하여 turn 사이 KV pos를 보존한다.
///
/// # Invariant (R1)
///
/// `DecodeLoop`는 chat 세션 시작 시 1회 build되고 세션 종료 시 drop된다.
/// turn마다 build/drop하면 KV cache가 소실된다.
pub struct ChatSession {
    decode_loop: DecodeLoop,
    pub kv_mode: ChatKvMode,
    /// KV pos 외부 read용 cache. DecodeLoop.pos와 항상 동기화된다.
    pub pos: usize,
    max_seq_len: usize,
    /// β-6: turn별 stop condition 을 `ChatStopStage`(DecodeEnd 구독)에 전달하는 공유 슬롯.
    /// `run_turn` 이 turn 시작 시 arm, run 후 자동 disarm(RAII guard).
    stop_slot: Arc<ChatStopSlot>,
    /// Per-token streaming callback slot (SSE). The chat server arms it around a
    /// turn; empty for the interactive REPL / non-streaming requests (no-op stage).
    stream_slot: Arc<ChatStreamSlot>,
    /// Shared sampling config handle. The chat server overwrites it per request
    /// (OpenAI `temperature`/`top_p`); the in-loop [`ChatSampler`] reads it each
    /// step. `None` when the decode loop uses a non-`ChatSampler` (test mocks).
    sampling: Option<SharedSamplingConfig>,
}

impl ChatSession {
    /// production 조립자(`chat/build.rs`)가 mode-agnostic 하게 부품을 모아 세션을 만든다.
    /// 필드가 private 이므로 build.rs 가 직접 struct literal 을 쓸 수 없어 이 crate-private
    /// 생성자를 경유한다(decode_loop 은 이미 `finish_chat_loop` 로 stop/stream stage 가 배선됨).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        decode_loop: DecodeLoop,
        kv_mode: ChatKvMode,
        max_seq_len: usize,
        stop_slot: Arc<ChatStopSlot>,
        stream_slot: Arc<ChatStreamSlot>,
        sampling: Option<SharedSamplingConfig>,
    ) -> Self {
        Self {
            decode_loop,
            kv_mode,
            pos: 0,
            max_seq_len,
            stop_slot,
            stream_slot,
            sampling,
        }
    }

    /// spec test용 직접 생성자. 호출자가 DecodeLoop + kv_mode를 직접 조립한다.
    ///
    /// β-6: stop 판정을 `ChatStopStage` 로 수렴하므로, 내부에서 registry + ChatStopStage 를
    /// 구성해 decode_loop 에 `with_pipeline` 으로 주입한다. caller 가 미리 조립한 decode_loop 의
    /// 기존 registry 는 무시되고 이 stop-registry 로 교체된다(spec 의 빈-registry decode_loop 전제).
    #[doc(hidden)]
    pub fn new_for_test(decode_loop: DecodeLoop, kv_mode: ChatKvMode, max_seq_len: usize) -> Self {
        let (decode_loop, stop_slot, stream_slot) = install_stop_stage(decode_loop);
        Self {
            decode_loop,
            kv_mode,
            pos: 0,
            max_seq_len,
            stop_slot,
            stream_slot,
            sampling: None,
        }
    }
}

/// β-6: `ChatStopStage`(DecodeEnd) + `ChatStreamStage`(DecodeEnd) 를 등록한 새 registry 와 두
/// 공유 슬롯을 만든다. **submit 순서가 곧 dispatch 순서** — stop 을 먼저 등록해야 stop 토큰이
/// 스트리밍되지 않는다(stop 시 dispatch break → stream stage 미발화).
pub(crate) fn make_chat_registry() -> (
    Arc<PipelineRegistry>,
    Arc<ChatStopSlot>,
    Arc<ChatStreamSlot>,
) {
    let stop_slot = ChatStopSlot::new();
    let stream_slot = ChatStreamSlot::new();
    let registry = Arc::new(PipelineRegistry::new());
    registry.submit(Arc::new(ChatStopStage::new(Arc::clone(&stop_slot))));
    registry.submit(Arc::new(ChatStreamStage::new(Arc::clone(&stream_slot))));
    (registry, stop_slot, stream_slot)
}

/// 이미 build 된 decode_loop 에 chat registry 를 `with_pipeline_registry` 로 주입한다
/// (test/`new_for_test` 전용 — post-build 교체). production 빌더는 build **전** `with_pipeline`
/// 으로 동일 registry 를 주입해 `with_resilience` 의 TickStage 와 공존시킨다(`finish_chat_loop`).
fn install_stop_stage(
    decode_loop: DecodeLoop,
) -> (DecodeLoop, Arc<ChatStopSlot>, Arc<ChatStreamSlot>) {
    let (registry, stop_slot, stream_slot) = make_chat_registry();
    let decode_loop = decode_loop.with_pipeline_registry(registry);
    (decode_loop, stop_slot, stream_slot)
}

/// production chat 빌더 공통 마무리: 단일 registry 를 build **전** 주입하고(`with_pipeline`),
/// resilience 가 있으면 happy/chat-minimal `CommandDispatcher` + `with_resilience` 를 배선한다.
/// build() 가 동일 registry 에 `TickStage`(PostSample) 를 submit 하므로 stop/stream/tick stage 가
/// 한 registry 에 공존한다. eviction directive 는 chat 자체 CacheManager(루프 밖)가 담당하므로
/// dispatcher 의 cache_manager 는 None(inert) — [`build_standard_loop`] happy 경로와 동형.
pub(crate) fn finish_chat_loop(
    builder: DecodeLoopBuilder<HasForward>,
    registry: Arc<PipelineRegistry>,
    resilience: Option<ResilienceAdapter>,
    kv_handles: Vec<Arc<StandardFormat>>,
) -> DecodeLoop {
    let builder = builder.with_pipeline(Arc::clone(&registry));
    let builder = match resilience {
        Some(adapter) => {
            let dispatcher = CommandDispatcher::new(
                registry,
                kv_handles,
                None, // cache_manager: chat evicts out-of-loop (ensure_capacity/on_turn_end)
                Vec::new(), // layer_slots (no partition)
                None, // hardware
                None, // model (no weight swap)
                None, // swap_runtime
                None, // importance
                Vec::new(), // quant_window_handles (manager quant inert — chat-managed)
                None, // report_tx
                Arc::new(Mutex::new(None)), // hook_cell dummy
                Arc::new(Mutex::new(None)), // score_cell dummy
            );
            builder
                .with_resilience(adapter)
                .with_command_dispatcher(dispatcher)
        }
        None => builder,
    };
    builder.build()
}

impl ChatSession {
    /// turn 시작 시 prompt prefill. pos 갱신.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let logits = self.decode_loop.prefill(tokens)?;
        self.pos = self.decode_loop.pos_snapshot();
        Ok(logits)
    }

    /// turn 본체 inner decode. stop condition까지 토큰 누적.
    ///
    /// **finalize를 호출하지 않는다.** multi-turn 재사용이 핵심 invariant.
    ///
    /// β-6: stop 판정은 `ChatStopStage`(DecodeEnd 구독)가 담당한다. turn별 stop condition 을
    /// 공유 슬롯에 arm 한 뒤(RAII guard — run 후 자동 disarm) `run_until_stop` 을 호출한다.
    pub fn run_turn(&mut self, first_token: u32, stop: &dyn StopCondition) -> Result<DecodeResult> {
        let result = {
            // guard 수명 = decode 동기 실행 구간. drop 시 슬롯 clear (dangling 방지).
            let _guard = self.stop_slot.arm(stop);
            self.decode_loop.run_until_stop(first_token)?
        };
        self.pos = self.decode_loop.pos_snapshot();
        Ok(result)
    }

    /// `/reset` 처리. KV cache + score_accumulator + decode_loop.pos를 atomic하게 clear.
    ///
    /// # Reset 순서
    /// 1. Forward 내부 KV caches reset (`Forward::reset_kv`)
    /// 2. score_accumulator reset (Standard 모드만)
    /// 3. decode_loop.reset_pos()
    /// 4. self.pos = 0
    pub fn reset(&mut self) -> Result<()> {
        // 1. Forward 내부 KV reset
        self.decode_loop.forward_mut().reset_kv()?;

        // 2. score_accumulator + evicted_total reset (Standard 모드만)
        if let ChatKvMode::Standard(s) = &mut self.kv_mode {
            if let Some(acc) = s.score_accumulator.as_mut() {
                acc.reset();
            }
            s.evicted_total = 0;
        }

        // 3. decode_loop pos reset
        self.decode_loop.reset_pos();

        // 3b. sampler per-sequence state reset (rep penalty ring). Stateless chat
        //     server reuses one loop across requests — drop prior token history.
        self.decode_loop.reset_sampler();

        // 4. external pos cache clear
        self.pos = 0;

        Ok(())
    }

    /// turn 시작 전 KV capacity 보장.
    ///
    /// - Standard: CacheManager::force_evict → 재확인. 여전히 부족하면 bail.
    /// - QuantWindow/Offload: pos + additional > max_seq_len이면 bail (eviction 미지원).
    pub fn ensure_capacity(&mut self, additional: usize) -> Result<()> {
        match &self.kv_mode {
            ChatKvMode::Standard(s) => {
                if self.pos + additional <= self.max_seq_len {
                    return Ok(());
                }
                if s.cache_manager.is_none() {
                    anyhow::bail!(
                        "context would exceed max_seq_len={} (pos={}, incoming_reserve={}). \
                         Use /reset or increase --max-seq-len.",
                        self.max_seq_len,
                        self.pos,
                        additional
                    );
                }
                // force_evict 실행.
                // Borrow 분리: ChatKvMode 필드를 먼저 복사한 뒤 forward에 접근한다.
                let (target_ratio, score_based) = if let ChatKvMode::Standard(s) = &self.kv_mode {
                    (s.target_ratio, s.score_based)
                } else {
                    unreachable!()
                };

                let (removed, new_pos) = {
                    let scores_vec: Option<Vec<f32>> =
                        if let ChatKvMode::Standard(s) = &self.kv_mode {
                            if score_based {
                                s.score_accumulator
                                    .as_ref()
                                    .filter(|a| a.is_active())
                                    .map(|a| a.importance_scores().to_vec())
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                    // value-aware a_i: last-layer last-step per-head attention (value-aware policies).
                    let attn_vec: Option<Vec<f32>> = if let ChatKvMode::Standard(s) = &self.kv_mode
                    {
                        if score_based {
                            s.score_accumulator
                                .as_ref()
                                .filter(|a| a.is_active())
                                .and_then(|a| a.last_step_head_attn().map(|s| s.to_vec()))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let cm = if let ChatKvMode::Standard(s) = &self.kv_mode {
                        let cm_ref = s.cache_manager.as_ref().expect("checked above");
                        // SAFETY: cm_ref는 self.kv_mode 안에 있고, forward_mut()은
                        // self.decode_loop을 빌린다 (kv_mode와 서로 다른 필드).
                        // 두 필드가 disjoint임이 구조적으로 보장되지만 borrow
                        // checker는 self를 전체로 봄 — 포인터로 우회한다.
                        let cm_ptr: *const CacheManager = cm_ref;
                        unsafe { &*cm_ptr }
                    } else {
                        unreachable!()
                    };
                    self.decode_loop.forward_mut().try_evict(
                        cm,
                        scores_vec.as_deref(),
                        attn_vec.as_deref(),
                        true,
                        target_ratio,
                    )?
                };

                if removed > 0 {
                    if let ChatKvMode::Standard(s) = &mut self.kv_mode {
                        s.evicted_total += removed;
                    }
                    self.pos = new_pos;
                }

                // 재확인
                if self.pos + additional <= self.max_seq_len {
                    Ok(())
                } else {
                    anyhow::bail!(
                        "context would exceed max_seq_len={} even after eviction \
                         (pos={}, incoming_reserve={}). Use /reset or increase --max-seq-len.",
                        self.max_seq_len,
                        self.pos,
                        additional
                    );
                }
            }
            ChatKvMode::NonEvicting { .. } => {
                if self.pos + additional > self.max_seq_len {
                    anyhow::bail!(
                        "context would exceed max_seq_len={} (pos={}, incoming_reserve={}). \
                         Use /reset or increase --max-seq-len.",
                        self.max_seq_len,
                        self.pos,
                        additional
                    );
                }
                Ok(())
            }
        }
    }

    /// turn 종료 후 opportunistic eviction (Standard 모드만).
    ///
    /// pos가 KV capacity의 90% 이상이면 force_evict, 미만이면 maybe_evict.
    /// generate.rs::StandardTurnExec::on_turn_end (l.10288~10303)과 동등.
    pub fn on_turn_end(&mut self) -> Result<()> {
        let has_cm = matches!(
            &self.kv_mode,
            ChatKvMode::Standard(s) if s.cache_manager.is_some()
        );
        if !has_cm {
            return Ok(());
        }

        // KV capacity는 pos로 근사한다 (ModelForward 내부 cache.capacity()를
        // 직접 읽는 대신 max_seq_len을 proxy로 사용 — 할당 크기와 동일).
        let at_pressure = self.pos >= self.max_seq_len.saturating_mul(9) / 10;

        let (target_ratio, score_based) = if let ChatKvMode::Standard(s) = &self.kv_mode {
            (s.target_ratio, s.score_based)
        } else {
            return Ok(());
        };

        let scores_vec: Option<Vec<f32>> = if let ChatKvMode::Standard(s) = &self.kv_mode {
            if score_based {
                s.score_accumulator
                    .as_ref()
                    .filter(|a| a.is_active())
                    .map(|a| a.importance_scores().to_vec())
            } else {
                None
            }
        } else {
            None
        };

        // value-aware a_i: last-layer last-step per-head attention (value-aware policies).
        let attn_vec: Option<Vec<f32>> = if let ChatKvMode::Standard(s) = &self.kv_mode {
            if score_based {
                s.score_accumulator
                    .as_ref()
                    .filter(|a| a.is_active())
                    .and_then(|a| a.last_step_head_attn().map(|s| s.to_vec()))
            } else {
                None
            }
        } else {
            None
        };

        let cm_ptr: *const CacheManager = if let ChatKvMode::Standard(s) = &self.kv_mode {
            match s.cache_manager.as_ref() {
                Some(cm) => cm as *const CacheManager,
                None => return Ok(()),
            }
        } else {
            return Ok(());
        };

        // SAFETY: cm_ptr은 self.kv_mode의 일부이고, forward_mut()은 self.decode_loop을
        // 빌린다 — 두 필드는 disjoint. borrow checker가 self 전체를 잠그므로 포인터 우회.
        let cm: &CacheManager = unsafe { &*cm_ptr };

        let (removed, new_pos) = self.decode_loop.forward_mut().try_evict(
            cm,
            scores_vec.as_deref(),
            attn_vec.as_deref(),
            at_pressure,
            target_ratio,
        )?;

        if removed > 0 {
            if let ChatKvMode::Standard(s) = &mut self.kv_mode {
                s.evicted_total += removed;
            }
            self.pos = new_pos;
            eprintln!(
                "[Chat/Evict] on_turn_end: removed={} new_pos={}",
                removed, new_pos
            );
        }
        Ok(())
    }

    /// `/stats` 출력용 stats_line (D5, G1 enforce — 라인 포맷 원본 보존).
    pub fn stats_line(&self) -> String {
        match &self.kv_mode {
            ChatKvMode::Standard(s) => {
                format!(
                    "kv_pos={}/{} policy={} evicted_total={}",
                    self.pos, self.max_seq_len, s.policy_name, s.evicted_total
                )
            }
            ChatKvMode::NonEvicting { stats_fragment } => {
                format!(
                    "kv_pos={}/{} {}",
                    self.pos, self.max_seq_len, stats_fragment
                )
            }
        }
    }

    /// 현재 KV pos.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// max_seq_len.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Shared per-token streaming slot — the chat server arms it (with a callback)
    /// around a streaming turn; [`ChatStreamStage`] emits each kept token to it.
    pub fn stream_slot(&self) -> Arc<ChatStreamSlot> {
        Arc::clone(&self.stream_slot)
    }

    /// Overwrite the shared sampling config (OpenAI per-request `temperature`/
    /// `top_p`). No-op if the decode loop was built without a [`ChatSampler`]
    /// (test mocks). The change is observed by the in-loop sampler on its next step.
    pub fn set_sampling(&self, cfg: SamplingConfig) {
        if let Some(shared) = &self.sampling
            && let Ok(mut g) = shared.lock()
        {
            *g = cfg;
        }
    }

    /// Cancellation flag for the inner decode loop. The chat server flips it when
    /// a streaming client disconnects, so generation stops early
    /// (`StopReason::StopFlag`).
    pub fn stop_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.decode_loop.stop_flag()
    }
}

// ─── 3 forward-build 함수 (KvModeReg.build fn-ptr) ────────────────────────────
//
// 각 함수는 [`ModeBuildCtx`] 를 받아 whole-pipeline `Box<dyn Forward>` (+ resilience 핸들/
// eviction 라벨/`ChatKvMode` stats 페이로드)를 [`ChatModeBuild`] 로 반환한다. mode-agnostic 한
// ChatSession 조립(sampler/registry/finish_chat_loop/resilience)은 `chat/build.rs` 가 이
// 결과 *둘레*에서 수행한다 — dispatch 지점이 더는 mode 정체성에 match 하지 않는다.

use crate::session::mode::{ChatModeBuild, ModeBuildCtx};

/// "standard" mode forward-build (KvModeReg.build) — `ModelForward` + eviction
/// (CacheManager/score-accumulator). Standard 는 유일한 eviction mode 라 eviction 구성을
/// 여기서 mode-agnostically 소유한다. (옛 `build_chat_standard` forward 본문 verbatim.)
pub(crate) fn build_chat_standard_forward(ctx: ModeBuildCtx<'_>) -> Result<ChatModeBuild> {
    let max_seq_len = ctx.max_seq_len;
    let args = ctx.args;

    let kv_dtype = parse_kv_type(&args.kv_type)?;
    let initial = {
        let r = args.initial_kv_capacity();
        if r > 0 {
            r.min(max_seq_len)
        } else {
            256.min(max_seq_len)
        }
    };
    let kv_layout = crate::kv_cache_ops::KVLayout::from_cli(&args.kv_layout)
        .ok_or_else(|| anyhow::anyhow!("Unsupported --kv-layout: '{}'", args.kv_layout))?;
    let kv_caches = crate::session::bin_setup::alloc_standard_kv_caches(
        &ctx.backend,
        ctx.memory.clone(),
        ctx.num_layers,
        initial,
        max_seq_len,
        ctx.kv_heads,
        ctx.head_dim,
        kv_dtype,
        kv_layout,
    )?;

    // eviction setup — generate.rs build_chat_eviction 로직 이관
    let (cache_manager, score_accumulator, score_based, policy_name) =
        build_chat_eviction_internal(&ctx, max_seq_len)?;

    let target_ratio = args.eviction_target_ratio();

    // ModelForward 생성
    // §5.9.2 Track B: chat 경로는 swap 미구성 → hook 더미 cell (항상 None).
    let hook_cell: Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>> =
        Arc::new(Mutex::new(None));
    // §5.9.1 Track A: chat 경로는 v1 EvictionHook 기반 score 처리 유지(eval_loop 동형) →
    // ModelForward 의 score_cell 은 더미 None (chat 측 score acc 는 별도 ChatKvMode 안).
    let score_cell: Arc<
        Mutex<Option<crate::inference::attention_scores::AttentionScoreAccumulator>>,
    > = Arc::new(Mutex::new(None));
    let mf = ModelForward::new(
        ctx.backend,
        ctx.memory,
        ctx.cpu_backend,
        ctx.model,
        kv_caches,
        max_seq_len,
        false, // chat 모드는 plan path 비활성 (D4: eviction + plan 공존 미지원)
        hook_cell,
        score_cell,
    )?;

    // Resilience: heartbeat KV handle (layer-0 StandardFormat) + dispatcher kv_handles.
    // `fmt_caches()` is populated by `ModelForward::new`. Read before `mf` is moved.
    let kv_handles = mf.fmt_caches().to_vec();
    let kv_handle = mf
        .fmt_caches()
        .first()
        .map(|f| Arc::clone(f) as Arc<dyn KVCacheFormat>);

    Ok(ChatModeBuild {
        forward: Box::new(mf),
        kv_handles,
        kv_handle,
        quant_handle: None,
        eviction_policy: policy_name.clone(),
        kv_mode: ChatKvMode::Standard(Box::new(ChatKvModeStandard {
            cache_manager,
            score_accumulator,
            score_based,
            policy_name,
            target_ratio,
            evicted_total: 0,
        })),
    })
}

/// "kivi" mode forward-build (KvModeReg.build) — quant-window-private 구성 일체
/// (alloc_quant_window_kv_caches / QuantWindowForward::new / bits·residual / caps.get::<QuantAttnBackend>())가
/// 여기로 이동했다. dispatch 지점은 더는 "kivi" 를 NAME 하지 않는다.
pub(crate) fn build_chat_quant_window_forward(ctx: ModeBuildCtx<'_>) -> Result<ChatModeBuild> {
    let max_seq_len = ctx.max_seq_len;
    let bits = ctx.args.effective_quant_window_bits();
    let residual_size = ctx.args.effective_quant_window_residual_size();
    // §4.3: cap pull moved into the quant_attn closure (gated by `needs_quant_attn`).
    let quant_attn = ctx.caps.get::<dyn QuantAttnBackend>();

    eprintln!(
        "[Chat/quant-window] bits={}, residual_size={}, max_seq_len={}",
        bits, residual_size, max_seq_len
    );

    let kv_caches = alloc_quant_window_kv_caches(
        ctx.num_layers,
        ctx.kv_heads,
        ctx.head_dim,
        max_seq_len,
        residual_size,
        bits,
        &ctx.backend,
        &quant_attn,
        &ctx.memory,
    );

    let fwd = QuantWindowForward::new(
        ctx.backend,
        ctx.memory,
        ctx.model,
        kv_caches,
        bits,
        residual_size,
        max_seq_len,
    )?;

    // §4.5: pos/capacity via base `kv_handle`, bit-width via the neutral `quant_handle`.
    let quant_window_handle = fwd.quant_window_caches().first().cloned();
    let kv_handle = quant_window_handle
        .clone()
        .map(|h| h as Arc<dyn KVCacheFormat>);
    let quant_handle = quant_window_handle
        .map(|h| h as Arc<dyn crate::session::resilience_adapter::QuantStageHandle>);

    Ok(ChatModeBuild {
        forward: Box::new(fwd),
        kv_handles: Vec::new(), // quant-window: no StandardFormat eviction handles.
        kv_handle,
        quant_handle,
        eviction_policy: String::new(), // quant-window: no in-loop eviction policy.
        kv_mode: ChatKvMode::NonEvicting {
            stats_fragment: format!("mode=kivi bits={bits} residual={residual_size}"),
        },
    })
}

/// "offload" mode forward-build (KvModeReg.build) — `OffloadForward` + offload cache
/// container. (옛 `build_chat_offload` forward 본문 verbatim.)
pub(crate) fn build_chat_offload_forward(ctx: ModeBuildCtx<'_>) -> Result<ChatModeBuild> {
    let max_seq_len = ctx.max_seq_len;
    let args = ctx.args;
    let kv_dtype = parse_kv_type(&args.kv_type)?;
    let offload_mode = args.effective_kv_offload_storage();
    let disk_dir = args.swap_dir.clone();
    let max_prefetch_depth = args.kv_mode_args.kv_max_prefetch_depth;

    let token_bytes = ctx.kv_heads * ctx.head_dim * kv_dtype.size();
    let disk_dir_ref = disk_dir.as_deref();

    eprintln!(
        "[Chat/Offload] mode={}, dtype={:?}, layers={}, token_bytes={}, max_seq={}",
        offload_mode, kv_dtype, ctx.num_layers, token_bytes, max_seq_len
    );

    let kv_caches = alloc_offload_kv_caches(
        ctx.num_layers,
        &offload_mode,
        kv_dtype,
        ctx.kv_heads,
        ctx.head_dim,
        max_seq_len,
        token_bytes,
        disk_dir_ref,
        &ctx.backend,
        &ctx.memory,
    )?;

    let prefetch =
        crate::kv::offload::prefetch::PrefetchController::new(max_prefetch_depth, ctx.num_layers);

    let fwd = OffloadForward::new(
        ctx.backend,
        ctx.memory,
        ctx.model,
        kv_caches,
        prefetch,
        max_seq_len,
    )?;

    Ok(ChatModeBuild {
        forward: Box::new(fwd),
        kv_handles: Vec::new(),
        // Offload has no KV-format handle accessor → heartbeat KV handle is skipped.
        kv_handle: None,
        quant_handle: None,
        eviction_policy: String::new(),
        kv_mode: ChatKvMode::NonEvicting {
            stats_fragment: format!(
                "mode=offload store={offload_mode} prefetch_depth={max_prefetch_depth}"
            ),
        },
    })
}

fn parse_kv_type(s: &str) -> Result<DType> {
    match s {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "q4" => Ok(DType::Q4_0),
        other => anyhow::bail!("Unsupported KV type: {other}. Use f32, f16, or q4."),
    }
}

// ─── 내부 헬퍼 ───────────────────────────────────────────────────────────────

/// generate.rs build_chat_eviction (l.10317~10439) 이관. `ModeBuildCtx` + `ctx.args` 직독.
///
/// Returns (cache_manager, score_accumulator, score_based, policy_name).
#[allow(clippy::type_complexity)]
fn build_chat_eviction_internal(
    ctx: &ModeBuildCtx<'_>,
    max_seq_len: usize,
) -> Result<(
    Option<CacheManager>,
    Option<AttentionScoreAccumulator>,
    bool,
    String,
)> {
    let args = ctx.args;
    let eviction_policy = args.eviction_policy().to_string();
    if eviction_policy == "none" {
        return Ok((None, None, false, "none".to_string()));
    }

    // Score-based stages declare a protected-prefix (4 sinks); score-free ones declare 0 → chat
    // protects 4 sinks by default (streaming derives its own sink and ignores the value). No
    // per-name branch.
    let actual_protected_prefix = args.protected_prefix().unwrap_or_else(|| {
        match crate::kv::eviction::stage_registry::stage_default_protected_prefix(&eviction_policy)
        {
            0 => 4,
            cap => cap,
        }
    });

    let monitor: Box<dyn crate::resilience::sys_monitor::SystemMonitor> =
        if ctx.backend.is_discrete_gpu() {
            Box::new(NoOpMonitor)
        } else {
            Box::new(LinuxSystemMonitor)
        };
    let threshold_bytes = (args.memory_threshold_mb() as u64 * 1024 * 1024) as usize;

    // linkme fat-LTO 생존 self-test: 빌트인 stage 미등록 시 fail-fast.
    crate::kv::eviction::stage_registry::ensure_builtin_stages_registered()?;

    let cache_manager = {
        let name = eviction_policy.as_str();
        // Every policy (none/sliding/streaming/h2o/h2o_plus/d2o) resolves through the plugin registry
        // by name (static linkme + dynamic --load-plugin), with its private knobs riding the opaque
        // StageArgs blob (built generically by `Args::stage_args`). Names no plugin.
        let policy: Box<dyn crate::kv::eviction::EvictionPolicy> = {
            // streaming window 유도는 StageParams 5필드 밖이라 caller(여기)에서 해소해 baked.
            // 비-streaming 정책의 make 는 이 필드를 무시한다.
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
            let stage_args = args.stage_args();
            let extra: Vec<argus_extension_api::PluginArg> = stage_args
                .iter()
                .map(|(k, v)| argus_extension_api::PluginArg { key: k, val: v })
                .collect();
            // 정적(linkme) + 동적(--load-plugin dlopen) 통합 조회. miss = unknown.
            let stage =
                crate::kv::eviction::stage_registry::make_stage_with_args(name, &params, &extra)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Unknown eviction policy for --chat: '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o{} (or --load-plugin <.so>)",
                            name,
                            if cfg!(feature = "caote") { ", caote" } else { "" }
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

    // caote 는 value-aware(crit_i = a_i·‖v_i − o_h‖) — V 는 ctx.tensor(Value)로 직접 읽지만
    // 가중치 a_i 는 importance 가 공급돼야 한다. score_based=true 여야 decode 루프가
    // force_evict_with_scores 로 importance 를 흘려보내 KVStageCtx(Some(importance)) 가 된다
    // (미공급 시 weight=0 → degenerate). attn-weight(last_attn) 정밀화는 Tier 2 deferred.
    let score_based = crate::kv::eviction::stage_registry::stage_is_score_based(&eviction_policy);

    crate::inference::attention_scores::ensure_score_producers_registered()?;
    let mut acc = AttentionScoreAccumulator::new_gqa(
        max_seq_len,
        ctx.model.config.num_attention_heads,
        ctx.model.config.num_key_value_heads,
        ctx.model.config.num_hidden_layers,
        args.h2o_tracked_layers(),
        args.h2o_decay(),
    );
    acc.set_active(true);
    acc.set_time_normalize(!args.h2o_raw_scores());

    // GPU-side accumulator init (OpenCL only). Caps-driven arming: only when the policy consumes
    // scores (`score_based`); a score-free policy still runs eviction but never reads importance, so
    // arming the GPU score path for it would write/reduce scores that are never read.
    #[cfg(feature = "opencl")]
    if score_based
        && let Some(ocl_be) = ctx
            .backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
    {
        let _ = ocl_be.init_gpu_score_acc(
            ctx.model.config.num_hidden_layers,
            ctx.model.config.num_attention_heads,
            ctx.model.config.num_key_value_heads,
            max_seq_len,
            args.h2o_decay(),
        );
        if let Some(gpu_acc) = ocl_be.gpu_score_acc_mut() {
            gpu_acc.set_active(true);
        }
    }

    Ok((Some(cache_manager), Some(acc), score_based, eviction_policy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::sampling::StepCtx;
    use crate::session::chat::stop_condition::StopCondition as StopConditionTrait;
    use crate::session::decode_loop::StopReason;
    use crate::session::forward::Forward;

    // ─── Mock Forward ──────────────────────────────────────────────────────

    /// 간단한 sequence generator. prefill → first_token → step_count+1 순으로 emit.
    struct MockSeqForward {
        vocab: usize,
        step_count: usize,
        reset_count: usize,
    }

    impl Forward for MockSeqForward {
        fn prefill(&mut self, _tokens: &[u32], _start_pos: usize) -> anyhow::Result<Vec<f32>> {
            let mut logits = vec![0.0f32; self.vocab];
            logits[0] = 1.0;
            Ok(logits)
        }

        fn step(&mut self, _ctx: &StepCtx, _token: u32) -> anyhow::Result<Vec<f32>> {
            self.step_count += 1;
            let mut logits = vec![0.0f32; self.vocab];
            let target = self.step_count % self.vocab;
            logits[target] = 1.0;
            Ok(logits)
        }

        fn reset_kv(&mut self) -> anyhow::Result<()> {
            self.reset_count += 1;
            self.step_count = 0;
            Ok(())
        }
    }

    // ─── Mock StopCondition ────────────────────────────────────────────────

    /// stop_id에 해당하는 토큰이 생성되면 종료.
    struct TokenStop {
        stop_id: u32,
        max_pos: usize,
    }

    impl StopConditionTrait for TokenStop {
        fn should_stop(&self, sampled: u32, pos: usize) -> bool {
            sampled == self.stop_id || pos >= self.max_pos
        }
    }

    // ─── ChatSession factory (mock용) ──────────────────────────────────────

    /// mock Forward로 ChatSession(Standard 모드) 생성.
    fn make_mock_session(max_seq_len: usize) -> ChatSession {
        let fwd = MockSeqForward {
            vocab: 16,
            step_count: 0,
            reset_count: 0,
        };
        let decode_loop = DecodeLoopBuilder::new()
            .with_forward(fwd)
            .with_kv_capacity(max_seq_len)
            .build();
        let (decode_loop, stop_slot, stream_slot) = super::install_stop_stage(decode_loop);
        ChatSession {
            decode_loop,
            kv_mode: ChatKvMode::Standard(Box::new(ChatKvModeStandard {
                cache_manager: None,
                score_accumulator: None,
                score_based: false,
                policy_name: "none".to_string(),
                target_ratio: 1.0,
                evicted_total: 0,
            })),
            pos: 0,
            max_seq_len,
            stop_slot,
            stream_slot,
            sampling: None,
        }
    }

    // ─── G2: multi-turn pos 누적 보존 ─────────────────────────────────────

    /// G2: turn 1 후 pos > 0이고, ChatSession이 살아있어 turn 2 prefill을 받을 수 있다.
    ///
    /// DecodeLoop::prefill은 pos = tokens.len() (절대값)으로 설정한다.
    /// R1 invariant: ChatSession이 turn 사이 drop되지 않아야 한다.
    #[test]
    fn g2_multi_turn_pos_preserved() {
        let mut session = make_mock_session(2048);
        let stop = TokenStop {
            stop_id: 3,
            max_pos: 100,
        };

        // turn 1: prefill + decode
        let prompt = &[1u32, 2, 3];
        let logits = session.prefill(prompt).unwrap();
        assert_eq!(logits.len(), 16);
        assert_eq!(session.pos(), 3);

        // GreedySampler: logits[0]=1.0 → first_token = 0
        let result1 = session.run_turn(0, &stop).unwrap();
        let pos_after_turn1 = session.pos();
        assert!(pos_after_turn1 > 0, "turn 1 후 pos > 0");
        assert_eq!(result1.stopped_by, StopReason::StopConditionMet);

        // R1 검증: ChatSession이 drop되지 않고 turn 2 prefill 수신 가능.
        // DecodeLoop::prefill은 pos += tokens.len() (누적)이므로
        // 2nd turn prefill 후 pos = pos_after_turn1 + prompt2.len().
        let prompt2 = &[10u32, 11];
        let _ = session.prefill(prompt2).unwrap();
        let expected_pos = pos_after_turn1 + prompt2.len();
        assert_eq!(
            session.pos(),
            expected_pos,
            "prefill accumulates pos (multi-turn)"
        );
    }

    // ─── G3: /reset 동작 ──────────────────────────────────────────────────

    /// G3: reset 후 pos == 0. score_acc (evicted_total) 도 0.
    #[test]
    fn g3_reset_clears_pos_and_acc() {
        let mut session = make_mock_session(2048);
        let stop = TokenStop {
            stop_id: 99,
            max_pos: 5,
        };

        let _ = session.prefill(&[1u32, 2]).unwrap();
        let _ = session.run_turn(0, &stop).unwrap();
        assert!(session.pos() > 0);

        // evicted_total을 수동으로 설정하여 reset 후 0이 되는지 검증
        if let ChatKvMode::Standard(s) = &mut session.kv_mode {
            s.evicted_total = 42;
        }

        session.reset().unwrap();
        assert_eq!(session.pos(), 0, "reset 후 pos == 0");

        // evicted_total도 0
        if let ChatKvMode::Standard(s) = &session.kv_mode {
            assert_eq!(s.evicted_total, 0, "reset 후 evicted_total == 0");
        }
    }

    /// G3 보조: reset 후 KV forward의 reset_kv가 호출됐는지 확인.
    /// MockSeqForward::reset_count로 간접 검증.
    #[test]
    fn g3_reset_calls_forward_reset_kv() {
        // reset_kv 호출 여부를 decode_loop.forward_mut()으로 확인하기 위해
        // 직접 mock session을 구성한다.
        let fwd = MockSeqForward {
            vocab: 8,
            step_count: 5,
            reset_count: 0,
        };
        let decode_loop = DecodeLoopBuilder::new()
            .with_forward(fwd)
            .with_kv_capacity(2048)
            .build();
        let (decode_loop, stop_slot, stream_slot) = super::install_stop_stage(decode_loop);
        let mut session = ChatSession {
            decode_loop,
            kv_mode: ChatKvMode::NonEvicting {
                stats_fragment: "mode=kivi bits=4 residual=32".to_string(),
            },
            pos: 10,
            max_seq_len: 2048,
            stop_slot,
            stream_slot,
            sampling: None,
        };

        session.reset().unwrap();
        assert_eq!(session.pos(), 0);
    }

    // ─── G4: ensure_capacity 분기 ─────────────────────────────────────────

    /// G4: QuantWindow 모드에서 pos + additional > max_seq_len이면 bail.
    #[test]
    fn g4_kivi_ensure_capacity_bails_on_overflow() {
        let fwd = MockSeqForward {
            vocab: 8,
            step_count: 0,
            reset_count: 0,
        };
        let decode_loop = DecodeLoopBuilder::new()
            .with_forward(fwd)
            .with_kv_capacity(10)
            .build();
        let (decode_loop, stop_slot, stream_slot) = super::install_stop_stage(decode_loop);
        let mut session = ChatSession {
            decode_loop,
            kv_mode: ChatKvMode::NonEvicting {
                stats_fragment: "mode=kivi bits=4 residual=32".to_string(),
            },
            pos: 9,
            max_seq_len: 10,
            stop_slot,
            stream_slot,
            sampling: None,
        };
        // pos=9, additional=2 → 9+2=11 > 10 → bail
        let result = session.ensure_capacity(2);
        assert!(result.is_err(), "overflow 시 bail 예상");
    }

    /// G4: Offload 모드에서도 overflow bail.
    #[test]
    fn g4_offload_ensure_capacity_bails_on_overflow() {
        let fwd = MockSeqForward {
            vocab: 8,
            step_count: 0,
            reset_count: 0,
        };
        let decode_loop = DecodeLoopBuilder::new()
            .with_forward(fwd)
            .with_kv_capacity(10)
            .build();
        let (decode_loop, stop_slot, stream_slot) = super::install_stop_stage(decode_loop);
        let mut session = ChatSession {
            decode_loop,
            kv_mode: ChatKvMode::NonEvicting {
                stats_fragment: "mode=offload store=raw prefetch_depth=2".to_string(),
            },
            pos: 9,
            max_seq_len: 10,
            stop_slot,
            stream_slot,
            sampling: None,
        };
        let result = session.ensure_capacity(2);
        assert!(result.is_err(), "offload overflow 시 bail 예상");
    }

    /// G4: Standard 모드, cache_manager=None이면 overflow 시 bail.
    #[test]
    fn g4_standard_no_cache_manager_bails_on_overflow() {
        let mut session = make_mock_session(10);
        session.pos = 9;
        let result = session.ensure_capacity(2);
        assert!(result.is_err(), "no cache_manager + overflow → bail");
    }

    /// G4: Standard 모드, 여유 있으면 Ok.
    #[test]
    fn g4_standard_ok_when_capacity_sufficient() {
        let mut session = make_mock_session(10);
        session.pos = 5;
        let result = session.ensure_capacity(2);
        assert!(result.is_ok(), "pos=5, additional=2, max=10 → Ok");
    }

    // ─── β-6 commit A 핀 4: turn-boundary try_evict 직접 호출 보존 ─────────

    /// β-6 핀 4: turn-boundary score-fed try_evict 는 **decode loop 밖 경로**다.
    /// `ChatSession::ensure_capacity`/`on_turn_end` 가 `decode_loop.forward_mut().try_evict(cm, ...)`
    /// 를 직접 호출하는 이 경로는 수렴(commit B) 에서 **stage 화하지 않고 보존**한다 — 이 테스트가
    /// try_evict 직접 호출이 실재함을 핀한다. 통합 후에도 이 호출이 그대로 살아 있어야 한다.
    #[test]
    fn turn_boundary_try_evict_called_directly_on_overflow() {
        use crate::kv::cache_manager::CacheManager;
        use crate::kv::eviction::stage_registry::sliding_backed_policy;
        use crate::resilience::sys_monitor::NoOpMonitor;
        use crate::session::forward::Forward as ForwardTrait;
        use std::cell::Cell;
        use std::rc::Rc;

        // try_evict 호출 횟수를 기록하는 mock Forward. removed=1, new_pos=pos-1 반환.
        struct EvictCountForward {
            vocab: usize,
            evict_calls: Rc<Cell<usize>>,
        }
        impl ForwardTrait for EvictCountForward {
            fn prefill(&mut self, _t: &[u32], _start_pos: usize) -> anyhow::Result<Vec<f32>> {
                Ok(vec![0.0f32; self.vocab])
            }
            fn step(&mut self, _c: &StepCtx, _t: u32) -> anyhow::Result<Vec<f32>> {
                Ok(vec![0.0f32; self.vocab])
            }
            fn try_evict(
                &mut self,
                _cm: &CacheManager,
                _scores: Option<&[f32]>,
                _last_attn: Option<&[f32]>,
                _force: bool,
                _target_ratio: f32,
            ) -> anyhow::Result<(usize, usize)> {
                self.evict_calls.set(self.evict_calls.get() + 1);
                // overflow 해소: pos 를 max_seq_len 밑으로 끌어내려 재확인 통과.
                Ok((5, 4))
            }
        }

        let evict_calls = Rc::new(Cell::new(0usize));
        let fwd = EvictCountForward {
            vocab: 8,
            evict_calls: evict_calls.clone(),
        };
        let decode_loop = DecodeLoopBuilder::new()
            .with_forward(fwd)
            .with_kv_capacity(10)
            .build();

        let (decode_loop, stop_slot, stream_slot) = super::install_stop_stage(decode_loop);
        // cache_manager=Some → ensure_capacity overflow 시 try_evict 직접 호출 경로 진입.
        let policy = sliding_backed_policy(4, 2);
        let cm = CacheManager::new(policy, Box::new(NoOpMonitor), usize::MAX, 0.5);
        let mut session = ChatSession {
            decode_loop,
            kv_mode: ChatKvMode::Standard(Box::new(ChatKvModeStandard {
                cache_manager: Some(cm),
                score_accumulator: None,
                score_based: false,
                policy_name: "sliding".to_string(),
                target_ratio: 0.5,
                evicted_total: 0,
            })),
            pos: 9,
            max_seq_len: 10,
            stop_slot,
            stream_slot,
            sampling: None,
        };

        // pos=9, additional=2 → 11 > 10 → overflow → try_evict 직접 호출.
        session.ensure_capacity(2).unwrap();
        assert_eq!(
            evict_calls.get(),
            1,
            "turn-boundary try_evict 가 decode loop 밖에서 직접 1회 호출됨"
        );
        // try_evict 반환 new_pos=4 로 pos 갱신 → evicted_total 누적.
        assert_eq!(session.pos(), 4, "try_evict new_pos 로 pos 갱신");
        if let ChatKvMode::Standard(s) = &session.kv_mode {
            assert_eq!(s.evicted_total, 5, "removed 누적");
        }
    }

    // ─── D5/G1: stats_line 포맷 보존 ─────────────────────────────────────

    #[test]
    fn g1_stats_line_standard_format() {
        let mut session = make_mock_session(2048);
        session.pos = 42;
        // evicted_total 수동 설정
        if let ChatKvMode::Standard(s) = &mut session.kv_mode {
            s.evicted_total = 10;
            s.policy_name = "sliding".to_string();
        }
        let line = session.stats_line();
        assert_eq!(line, "kv_pos=42/2048 policy=sliding evicted_total=10");
    }

    #[test]
    fn g1_stats_line_kivi_format() {
        let fwd = MockSeqForward {
            vocab: 8,
            step_count: 0,
            reset_count: 0,
        };
        let decode_loop = DecodeLoopBuilder::new()
            .with_forward(fwd)
            .with_kv_capacity(512)
            .build();
        let (decode_loop, stop_slot, stream_slot) = super::install_stop_stage(decode_loop);
        let session = ChatSession {
            decode_loop,
            kv_mode: ChatKvMode::NonEvicting {
                stats_fragment: "mode=kivi bits=4 residual=32".to_string(),
            },
            pos: 100,
            max_seq_len: 512,
            stop_slot,
            stream_slot,
            sampling: None,
        };
        let line = session.stats_line();
        assert_eq!(line, "kv_pos=100/512 mode=kivi bits=4 residual=32");
    }

    #[test]
    fn g1_stats_line_offload_format() {
        let fwd = MockSeqForward {
            vocab: 8,
            step_count: 0,
            reset_count: 0,
        };
        let decode_loop = DecodeLoopBuilder::new()
            .with_forward(fwd)
            .with_kv_capacity(512)
            .build();
        let (decode_loop, stop_slot, stream_slot) = super::install_stop_stage(decode_loop);
        let session = ChatSession {
            decode_loop,
            kv_mode: ChatKvMode::NonEvicting {
                stats_fragment: "mode=offload store=raw prefetch_depth=4".to_string(),
            },
            pos: 77,
            max_seq_len: 512,
            stop_slot,
            stream_slot,
            sampling: None,
        };
        let line = session.stats_line();
        assert_eq!(
            line,
            "kv_pos=77/512 mode=offload store=raw prefetch_depth=4"
        );
    }
}
