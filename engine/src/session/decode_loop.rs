//! [`DecodeLoop`] + typestate builder (Phase 4-2).
//!
//! The decode loop owns all six pipeline trait objects and the stop flag.
//! [`DecodeLoopBuilder`] uses a typestate marker (`NoForward` → `HasForward`)
//! so `.build()` is only callable once a [`Forward`] has been supplied
//! (INV-LAYER-007).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::format::KVCacheFormat;
use crate::inference::sampling::{GreedySampler, StepCtx, TokenSampler};
use crate::observability::profile::OpProfiler;
use crate::pipeline::StopReason as StageStopReason;
use crate::pipeline::{
    LifecyclePhase, PipelineDispatcher, Pressure, PressureSource, StageContext, StepInfo,
};
use crate::session::command_dispatcher::{CommandDispatcher, CommandSource};
use crate::session::forward::Forward;
use crate::session::pipeline_registry::PipelineRegistry;

use super::defaults::NoOpCommandSource;

/// Why [`DecodeLoop::run`] returned.
///
/// **Phase β-7**: this is the *driver-result* vocabulary — it carries
/// [`StopReason::StopFlag`], which the v2 stage-level
/// [`crate::pipeline::StopReason`] (4-variant) deliberately omits: a Stage can
/// never *return* `StopFlag` because it is a driver-internal `stop_flag` break
/// (cancellation observed but not produced by any stage). The two enums stay
/// separate for exactly this reason — see `arch/pipeline_stage_design_v2.md`
/// §5.2.1 (다). [`DecodeLoop::map_stage_stop`] maps the 4 stage variants into
/// the matching driver variants; `StopFlag` is set only by the driver loop guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    BudgetExhausted,
    StopFlag,
    EosToken,
    CommandRequested,
    /// Phase 4-5-c: [`DecodeLoop::run_until_stop`]에서 StopCondition이 true를 반환.
    StopConditionMet,
}

/// Decode result returned by [`DecodeLoop::run`].
///
/// **Phase β-7**: moved here from the deleted `session::traits` (it is the
/// `run`/`run_until_stop` return type). Its `stopped_by` uses the driver-result
/// [`StopReason`] above (StopFlag-bearing), distinct from the stage vocabulary.
#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub tokens_generated: Vec<u32>,
    /// True cumulative sequence position (= the next RoPE `start_pos`). This is **not**
    /// collapsed when an in-loop eviction compacts the KV cache — surviving keys keep the RoPE
    /// phase baked at their original absolute positions, so the position the loop continues from
    /// must stay cumulative. Hence `final_pos >= final_cache_pos` whenever eviction fired.
    pub final_pos: usize,
    /// KV-cache occupancy at exit (== `final_pos` when no eviction handle is wired). Memory
    /// telemetry (e.g. `peak_kv_mb`) must use this occupancy, not the cumulative `final_pos`,
    /// otherwise it over-reports KV footprint after an eviction-decrease.
    pub final_cache_pos: usize,
    pub stopped_by: StopReason,
}

/// Typestate marker — Forward not yet supplied. `.build()` is unavailable.
pub struct NoForward;

/// Typestate marker — Forward supplied. `.build()` is callable.
pub struct HasForward(Box<dyn Forward>);

/// Decode loop. Owned trait objects only — see INV-LAYER-006.
pub struct DecodeLoop {
    forward: Box<dyn Forward>,
    cmd_source: Box<dyn CommandSource>,
    sampler: Box<dyn TokenSampler>,
    // β-7: v1 eviction/swap/observers/report 슬롯 제거. eviction → PipelineStage(KvMutate/
    // WeightMutate, stages/kv/eviction.rs) 로 수렴, tick/observe → TickStage/PipelineStage 로,
    // report(EngineReport) 슬롯은 dead 였다(IPC 송출은 resilience_init 가 executor 직접 호출).
    stop_flag: Arc<AtomicBool>,
    // β-4 (v2 §5.4 A-1): 2-source 명령 분배자. cmd_source.poll() 이 pure 생산한 EngineCommand 를
    // 받아 ① OneShot EvictionStage submit / ② LoopControl / ③ Hardware seam 으로 분배한다.
    // L4 동층 합성 — registry/KV handle/공유 cache-manager 는 dispatcher 가 보유(driver 미보유,
    // INV-LAYER-006 BANNED 비해당). None(happy/chat) → cmd_source 도 NoOp 라 빈 Vec → dispatch
    // no-op = 거동-0.
    dispatcher: Option<CommandDispatcher>,
    // β-2/β-6: L2 PipelineStage dispatch 배선. default = empty registry (len==0 fast-path 로
    // happy-path 거동-0). β-6 에서 run()/run_until_stop 이 공유 core(run_steps)로 통합되어
    // 양쪽 모두 dispatch 한다. chat 은 ChatStopStage(DecodeEnd) 등록 registry 를 주입한다.
    pipeline: Arc<PipelineRegistry>,
    // β-3: pos-환류용 held handle (§5.2.1 (가) — StageOutcome 무변경, EvictionStage 가 cache 의
    // current_pos 를 prune 한 뒤 driver 가 동일 layer-0 handle 의 current_pos 를 query 해 loop pos
    // 를 동기화). trait object(INV-LAYER-006). None(happy/chat/기존 전부) → 환류 skip + 빈 registry
    // = 거동-0. β-4 cutover 전제 배선 — registry submit 은 β-4 CommandDispatcher.
    kv_pos_handle: Option<Arc<dyn KVCacheFormat>>,
    // β-1 계약 (v2 §5.2.1): DecodeLoop 이 OpProfiler 1개 소유, dispatch 호출부마다
    // &mut 재대여 (StageContext.profiler).
    profiler: OpProfiler,
    // β-5: graded system 압력 source (memory-only `LocalPressureSource` 등). None(happy/chat) →
    // StepInfo.pressure = Pressure::default()(=0) — per-token syscall 차단(G4 happy-path 무주입).
    // build_bench_loop 만 주입한다. trait object (INV-LAYER-006 — Option<Arc<dyn _>> 합치).
    pressure_source: Option<Arc<dyn PressureSource>>,
    // β-5 N-step 캐시: 매 step /proc 읽기 금지. decode_step % PRESSURE_QUERY_INTERVAL == 0 일 때만
    // source 재query, 그 외 step 은 이 캐시값을 재사용한다. source 부재 시 항상 default(0).
    cached_pressure: Pressure,
    pos: usize,
    // eviction-후 RoPE-drift 수정: 마지막으로 관측한 KV-cache 점유(`kv_pos_handle.current_pos()`).
    // `pos`(진짜 누적 RoPE 위치)와 **분리**된 값이다 — eviction 이 cache 를 compaction 하면 occupancy 는
    // 줄지만 `pos` 는 누적값을 유지한다. 직전 관측 대비 점유 감소(shrink)를 감지해 GPU plan invalidation
    // 을 1회만 트리거하는 용도(handle 미배선 happy/chat 경로에선 사용 안 됨).
    kv_occupancy: usize,
    decode_step: usize,
    prev_token: u32,
    kv_capacity: usize,
    // A `FormatReencodeStage` is armed (PrefillEnd per-layer KV re-encode). After that phase the
    // forward's fused GPU plan must be invalidated (a dtype flip is invisible to the capacity-keyed
    // guard). `false`(happy/chat/eval/모든 기존 경로) → no invalidation call = 거동-0.
    kv_reencode_armed: bool,
}

/// β-5: `PressureSource` query 주기 (N-step 캐시 N). 매 decode step 마다 `/proc/meminfo` 를
/// 읽으면 happy-path 대비 syscall 오버헤드가 누적되므로, N step 마다 1회만 재query 한다.
const PRESSURE_QUERY_INTERVAL: usize = 8;

/// Build a StepCtx without borrowing the whole DecodeLoop — lets the loop
/// pass `&mut self.<field>` to trait methods on the same line.
fn step_ctx<'a>(
    pos: usize,
    prev_token: u32,
    kv_capacity: usize,
    decode_step: usize,
    stop: &'a AtomicBool,
) -> StepCtx<'a> {
    StepCtx {
        pos,
        prev_token,
        kv_capacity,
        decode_step,
        stop_requested: stop,
    }
}

impl DecodeLoop {
    /// β-2: 현재 step 스냅샷으로 phase 1건 dispatch. 빈 registry 는
    /// PipelineRegistry 내부 len==0 fast-path 로 무lock 즉시 반환 (거동-0).
    ///
    /// β-5: `pressure_source` 가 주입돼 있으면 N-step 캐시(`PRESSURE_QUERY_INTERVAL`)로 graded
    /// 압력을 갱신해 `StepInfo.pressure` 에 싣는다. source 부재(happy/chat) 면 `cached_pressure` 가
    /// 항상 `default()`(=0) 라 거동-0 (per-token syscall 0).
    fn dispatch_phase(&mut self, phase: LifecyclePhase) -> Option<StageStopReason> {
        self.refresh_pressure_if_due();
        let step = StepInfo {
            pos: self.pos,
            decode_step: self.decode_step,
            pressure: self.cached_pressure,
            prev_token: self.prev_token,
        };
        let mut ctx = StageContext {
            step,
            profiler: &mut self.profiler,
        };
        self.pipeline.dispatch(phase, &mut ctx)
    }

    /// β-5: N-step 캐시 갱신. source 가 있고 `decode_step` 이 `PRESSURE_QUERY_INTERVAL` 의 배수일
    /// 때만 `/proc` 을 재query 한다. source 부재면 no-op(캐시 = default 유지).
    fn refresh_pressure_if_due(&mut self) {
        if let Some(src) = &self.pressure_source
            && self.decode_step.is_multiple_of(PRESSURE_QUERY_INTERVAL)
        {
            self.cached_pressure = src.pressure();
        }
    }

    /// β-3/AB-3 KV-occupancy 추적 + GPU plan invalidation (§5.2.1 (가), §5.10.4 A안):
    /// eviction/offload stage dispatch 후 held handle 의 `current_pos`(= KV-cache 점유)가 직전 관측
    /// 대비 줄었으면, 보유 중인 GPU kernel plan 의 geometry 가 stale 해지므로 invalidate 한다.
    ///
    /// ★`self.pos` 는 **건드리지 않는다**. (이전엔 감소 시 `self.pos = current_pos` 로 collapse 했으나
    /// 그것이 eviction-후 RoPE-drift 버그였다.) compaction 은 생존 키를 낮은 연속 슬롯으로 **memmove**
    /// 할 뿐 재회전하지 않으므로(W-REROTATE), 생존 키는 원 절대위치의 RoPE phase 를 유지한다. 따라서
    /// 다음 토큰의 RoPE `start_pos`(= `self.pos`)는 **진짜 누적 시퀀스 위치**를 유지해야 한다 — 압축된
    /// `current_pos` 로 collapse 하면 query 가 생존 키보다 낮은 위치로 회전해 학습된 상대거리가 깨진다
    /// (`session/eval/eval_loop.rs:299-303`·`session/ppl/runner.rs:1160-1166` 이 같은 메커니즘에서
    /// 명시적으로 금지: "Do NOT reset start_pos to current_pos after eviction ... severe NLL
    /// degradation"). write-slot / attention seq_len 은 forward 가 cache 의 `current_pos` 에서 직접
    /// 읽으므로(CPU `forward_gen_fmt.rs`, GPU `backend/opencl/plan.rs`) `pos` 와 occupancy 는 독립이다.
    ///
    /// - 감소(eviction/offload prune): `on_kv_prune(occupancy)` 로 GPU plan invalidate (CPU 는 plan
    ///   부재라 no-op). occupancy 가 매 step 정상 증가(+1)와 분리돼 shrink 일 때만 1회 발화한다.
    /// - 증가(recall/offload restore)·불변: occupancy 만 추적, `on_kv_prune` 미호출 — recall 은 새 토큰을
    ///   추가하지 않으므로 누적 위치(`self.pos`)도 불변, plan 은 다음 step forward 가 자연 재빌드.
    /// - `kv_pos_handle == None`(happy/chat/기존 전부) 이면 즉시 return → **거동-0**.
    fn invalidate_plan_on_kv_shrink(&mut self) {
        let Some(h) = &self.kv_pos_handle else {
            return;
        };
        let occupancy = h.current_pos();
        if occupancy < self.kv_occupancy {
            // KV-cache 가 직전 관측 대비 줄었다(eviction/offload prune 으로 생존 키를 낮은 슬롯으로
            // compaction). 보유 plan 의 geometry 가 stale → 다음 step lazy rebuild 로 강하. self.pos 는
            // 의도적으로 불변(위 닥 — 원위치 RoPE phase 보존).
            self.forward.on_kv_prune(occupancy);
        }
        self.kv_occupancy = occupancy;
    }

    /// v2 StopReason(pipeline.rs 4-variant) → v1 StopReason(traits.rs) 수렴 매핑
    /// (v2 §5.2.1 (다)). StopFlag 는 v2 에 없음 — driver 루프 가드 v1 유지.
    fn map_stage_stop(r: StageStopReason) -> StopReason {
        match r {
            StageStopReason::EosToken => StopReason::EosToken,
            StageStopReason::BudgetExhausted => StopReason::BudgetExhausted,
            StageStopReason::StopConditionMet => StopReason::StopConditionMet,
            StageStopReason::CommandRequested => StopReason::CommandRequested,
        }
    }

    /// Run prefill over the prompt. Returns logits for the last token so the
    /// caller can sample the first generated token before invoking
    /// [`Self::run`]. `pos` is advanced to `tokens.len()`. `prev_token` is
    /// staged from `tokens.last()` for `on_prefill_end` observer context only;
    /// [`Self::run`] overwrites it with `first_token`.
    pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        // PrefillStart: prefill 엔 break 할 루프 없음 — Stop 은 per-token phase 전용
        // (chunked prefill 시 PrefillChunkBoundary 와 함께 재고).
        let _ = self.dispatch_phase(LifecyclePhase::PrefillStart);
        let start_pos = self.pos;
        let logits = self.forward.prefill(tokens, start_pos)?;
        self.pos += tokens.len();
        self.prev_token = *tokens.last().unwrap_or(&0);
        // Phase 4-4.7: production fallback (generate.rs)이
        // `sampling::sample(&mut logits, &tokens, ...)` 호출 시 `tokens`(prompt
        // 전체)를 rep history로 사용한다. paradigm equivalence를 위해 prefill
        // 시점에 prompt를 sampler에게 전부 통보. GreedySampler는 default no-op
        // 이라 무영향, RepetitionPenaltySampler만 ring buffer에 prompt suffix를
        // 적재한다.
        for &t in tokens {
            self.sampler.observe_token(t);
        }
        // PrefillEnd: prefill 산출(logits) 직후, Ok(logits) 직전 (β-7: v1 on_prefill_end
        // observer 루프 제거 — prefill-end 관찰은 PrefillEnd stage 구독으로 수렴).
        // eviction-후 RoPE-drift 수정: PrefillEnd(keepset prune) 전에 prefill 점유를 seed 해, prune
        // 을 shrink 로 감지할 수 있게 한다(첫 decode plan 은 lazy 라 여기 on_kv_prune 은 통상 no-op이나
        // 정확성을 위해 정확한 직전-점유를 기록). kv_pos_handle==None 이면 self.pos 로 폴백(거동-0).
        self.kv_occupancy = self
            .kv_pos_handle
            .as_ref()
            .map_or(self.pos, |h| h.current_pos());
        let _ = self.dispatch_phase(LifecyclePhase::PrefillEnd);
        // R-P1-1: PrefillEnd stage(PrefillKeepSetStage)가 prefix 를 prune 했으면 GPU plan 만
        // invalidate 한다(점유 감소 추적). self.pos 는 보정하지 않는다 — 생존 키가 원위치 RoPE phase 를
        // 유지하므로 첫 decode 토큰의 start_pos 는 진짜 prompt_len 이어야 한다(invalidate_plan_on_kv_shrink
        // 닥 참조). kv_pos_handle==None(prune 없음/미무장)이면 거동-0.
        self.invalidate_plan_on_kv_shrink();
        // L1-runtime: a FormatReencodeStage may have re-encoded layers at PrefillEnd (a dtype flip
        // the capacity-keyed plan guard cannot see). Invalidate the fused plan + clear the F16 sticky
        // lock so the first decode step's lazy build re-checks per-layer dtype. The plan is built
        // lazily *after* this point, so today this is a defensive no-op; it is the correct guard for a
        // post-plan-build (mid-decode) re-encode. Unarmed (every other path) → no call = 거동-0.
        if self.kv_reencode_armed {
            self.forward.on_kv_reencode();
        }
        Ok(logits)
    }

    /// Run up to `budget` decode steps starting from `first_token` (the token
    /// already sampled from `prefill`'s last logits). Returns sampled tokens
    /// (does **not** include `first_token`) + reason loop exited. Caller is
    /// responsible for prepending `first_token` to the final output sequence.
    pub fn run(&mut self, budget: usize, first_token: u32) -> anyhow::Result<DecodeResult> {
        let result = self.run_steps(budget, first_token)?;
        self.forward.finalize()?;
        // Finalize: forward.finalize() 후, DecodeResult 구성 직전 (β-7: v1 observer finalize
        // 체인 제거 — finalize 관찰은 Finalize stage 구독으로 수렴).
        let _ = self.dispatch_phase(LifecyclePhase::Finalize);
        Ok(result)
    }

    /// β-6: run()/run_until_stop 공유 decode core. 현 run() 본문에서 finalize 체인을 제외한
    /// 부분이다. `budget` step 까지 `first_token` 부터 decode 하고 [`DecodeResult`] 를 반환한다.
    /// finalize(`forward.finalize`/Finalize phase)는 호출하지 **않는다** —
    /// 호출자(run = 호출, run_until_stop = 미호출)가 결정한다.
    fn run_steps(&mut self, budget: usize, first_token: u32) -> anyhow::Result<DecodeResult> {
        self.prev_token = first_token;
        // Phase 4-4.7: stateful samplers (RepetitionPenaltySampler 등)이 production
        // fallback `tokens.push(first_token)`과 동치 history를 갖도록 첫 토큰을 통보.
        // [`super::defaults::GreedySampler`]는 default no-op이라 무영향.
        self.sampler.observe_token(first_token);
        let stop = Arc::clone(&self.stop_flag);
        // budget == usize::MAX(run_until_stop) 면 with_capacity 가 OOM 이므로 0 으로 시작.
        let mut generated = Vec::with_capacity(budget.min(4096));
        // budget ∞(run_until_stop) 면 BudgetExhausted 도달 불가 — stop stage/StopFlag 가 종결.
        // v1 run_until_stop 의 기본 stopped_by(StopConditionMet)와 수렴: stop stage 미발화 +
        // budget 무한이면 무한 루프이므로 실 종결은 항상 Stop/StopFlag 가 설정한다.
        let mut stopped_by = StopReason::BudgetExhausted;

        for _ in 0..budget {
            if stop.load(Ordering::Acquire) {
                stopped_by = StopReason::StopFlag;
                break;
            }
            // 매 step 전체 wall-clock 시작점 (target_tbt pacing 기준 — throttle
            // delay 도 이 측정에 포함된다).
            let t_iter = Instant::now();

            // DecodeStart: t_iter 직후, (a) command poll 전.
            if let Some(r) = self.dispatch_phase(LifecyclePhase::DecodeStart) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

            // (a) command poll + dispatch (β-4 v2 §5.4 A-1)
            // cmd_source.poll() 은 pure 생산(EngineCommand drain). dispatcher 가 ① evict 4종을
            // OneShot EvictionStage 로 registry.submit, ② control 7종을 LoopControl 로, ③ device
            // seam 으로 분배한다. dispatcher=None(happy/chat) 이면 cmd_source 도 NoOp → 빈 Vec →
            // 분배 대상 0 = 거동-0. throttle_delay/target_tbt 의 sticky 는 LoopControl 이 보존.
            let cmds = self.cmd_source.poll()?;
            let (suspended, throttle_delay_ms, target_tbt_ms): (bool, u64, u64) =
                if let Some(d) = self.dispatcher.as_mut() {
                    let control = d.dispatch(cmds);
                    (
                        control.suspended,
                        control.throttle_delay_ms,
                        control.target_tbt_ms,
                    )
                } else {
                    (false, 0, 0)
                };
            if suspended {
                // G6: suspend = loop break 보존 (pause/park 전환 금지).
                stopped_by = StopReason::CommandRequested;
                break;
            }
            if throttle_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
            }

            // (a.5) 제거: evict 는 dispatcher 가 OneShot EvictionStage 로 registry 에 submit 했고,
            // 아래 KvMutate dispatch + pos-환류(β-3 배선)가 이를 소비한다. v1 try_evict 인라인
            // 경로는 dispatcher.submit + EvictionStage.on_phase 로 cutover 됨.

            // KvMutate: (b) v1 eviction 직전 (§5.2.1 (나) — command-poll 직후·
            // (a.6) AB-3 완료: KvOffload/recall 은 OffloadStage(OneShot, KvMutate phase) 로 이전됨.
            // forward 직전). v2 EvictionStage(command-driven OneShot) 가 여기서 발화한다 — UER 로
            // cache 를 prune 한 뒤 pos-환류로 loop pos 동기화 (§5.2.1 (가)).
            if let Some(r) = self.dispatch_phase(LifecyclePhase::KvMutate) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }
            self.invalidate_plan_on_kv_shrink();

            // WeightMutate: KvMutate 직후, forward 직전 (§5.2.1 (나)). pressure band-driven
            // Persistent EvictionStage 등 eviction 후속 발화 슬롯. dispatch 후 pos-환류로 loop pos
            // 동기화 (β-7: v1 (b) eviction/(c) swap before/(e) swap after 인라인 슬롯 제거 —
            // eviction 은 EvictionStage, swap 은 SwapStage 로 수렴).
            if let Some(r) = self.dispatch_phase(LifecyclePhase::WeightMutate) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }
            self.invalidate_plan_on_kv_shrink();

            // PreForward: WeightMutate 후, (d) forward 직전.
            if let Some(r) = self.dispatch_phase(LifecyclePhase::PreForward) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

            // (d) forward
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            let logits = self.forward.step(&ctx, self.prev_token)?;

            // PostForward: forward 직후, PreSample 전 (β-7: v1 (e) swap after 인라인 슬롯
            // 제거 — swap commit/release 는 SwapStage 로 수렴).
            if let Some(r) = self.dispatch_phase(LifecyclePhase::PostForward) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

            // PreSample: PostForward 후, (f) sampler.sample 직전.
            if let Some(r) = self.dispatch_phase(LifecyclePhase::PreSample) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

            // (f) sample
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            let sampled = self.sampler.sample(&ctx, &logits);
            // Phase 4-4.7: sample 직후 history 갱신. production fallback에서
            // `tokens.push(next_token_id)`가 다음 step의 rep window에 들어가는
            // 것과 동치.
            self.sampler.observe_token(sampled);

            // PostSample: sampled 직후. β-6 commit C: v1 (f2) tick_sink 는 TickStage(PostSample
            // 구독)로 이관됐다 — TickStage 가 ResilienceAdapter per-token tick 을 발화한다(stop
            // 토큰 포함, v1 "stop 토큰에도 tick" 보존). chat stop 은 ChatStopStage(DecodeEnd).
            if let Some(r) = self.dispatch_phase(LifecyclePhase::PostSample) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }
            // β-7: v1 (g) observer.on_step_end 루프 제거 — per-step 관찰은 PostSample/DecodeEnd
            // stage 구독으로 수렴.

            // β-6: bookkeeping 을 push 앞으로 이동(재배열). DecodeEnd 가 증가된
            // pos/prev_token 으로 발화하고, Stop(chat StopConditionMet 등) 이면 token 을 **push
            // 하지 않고** break 한다 — v1 run_until_stop 의 stop 시맨틱(stop 토큰 미push, pos 는
            // 증가) 과 정합. 비-Stop 경로는 push 위치만 이동(매 iteration push 됨, 거동-0).
            self.prev_token = sampled;
            self.pos += 1;
            self.decode_step += 1;

            // DecodeEnd: bookkeeping 후, push·(h) pacing 전.
            // StepInfo 는 증가된 pos/decode_step + prev_token=sampled. ChatStopStage 가 여기서
            // should_stop(prev_token, pos) 를 평가한다(미포함 = chat stop 시맨틱).
            if let Some(r) = self.dispatch_phase(LifecyclePhase::DecodeEnd) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

            // DecodeEnd Stop 이 아니면 token push (run/run_until_stop 공통).
            generated.push(sampled);

            // (h) target TBT pacing — resilience SetTargetTbt 가 설정한 목표
            // wall-clock 에 도달하도록 step 끝에서 sleep. target_tbt_ms == 0
            // (미설정/release) 이면 no-op이라 비-resilience 경로는 무영향.
            // (a) 에서 LoopControl 로부터 읽은 sticky target_tbt_ms 를 사용.
            if target_tbt_ms > 0 {
                let elapsed_ms = t_iter.elapsed().as_secs_f64() * 1000.0;
                let target_ms = target_tbt_ms as f64;
                if elapsed_ms < target_ms {
                    std::thread::sleep(std::time::Duration::from_secs_f64(
                        (target_ms - elapsed_ms) / 1000.0,
                    ));
                }
            }
        }

        // final_pos = 진짜 누적 위치(self.pos). 메모리 telemetry 용 점유는 cache 에서 직접 읽는다
        // (handle 미배선 시 둘은 동일 = 거동-0).
        let final_cache_pos = self
            .kv_pos_handle
            .as_ref()
            .map_or(self.pos, |h| h.current_pos());
        Ok(DecodeResult {
            tokens_generated: generated,
            final_pos: self.pos,
            final_cache_pos,
            stopped_by,
        })
    }

    /// Phase 4-5-c. Chat 모드 inner decode loop — β-6 단일 driver 통합.
    ///
    /// `run()` 과 동일한 [`Self::run_steps`] core 를 `budget = usize::MAX` 로 돌린다. chat 차이는
    /// 분기 신설이 아니라 (i) registry 등록물([`ChatStopStage`] — `DecodeEnd` 에서 stop 판정) (ii)
    /// finalize 미호출 wrapper 뿐이다. stop 판정은 stage 가 담당하므로 본 메서드는 stop 인자를
    /// 받지 않는다 — 호출자([`ChatSession::run_turn`])가 turn별 stop condition 을 `ChatStopSlot` 에
    /// arm 한 뒤 호출한다.
    ///
    /// **TurnStart/TurnEnd 발화**: run_steps 전후로 dispatch 한다(turn 경계 stage 슬롯).
    ///
    /// **finalize 는 호출하지 않는다.** chat 세션은 여러 turn 에 걸쳐 같은 DecodeLoop 를
    /// 재사용하므로 finalize 는 전체 세션 종료 시 1회만 호출해야 한다.
    pub fn run_until_stop(&mut self, first_token: u32) -> anyhow::Result<DecodeResult> {
        // TurnStart: run_steps 진입 전. budget ∞ 라 stop 은 ChatStopStage(DecodeEnd) 또는
        // StopFlag 가 종결한다(미발화 시 무한 루프이므로 stop stage 등록이 chat 의 invariant).
        let _ = self.dispatch_phase(LifecyclePhase::TurnStart);
        let result = self.run_steps(usize::MAX, first_token)?;
        // TurnEnd: run_steps 후. finalize 미호출(세션 재사용).
        let _ = self.dispatch_phase(LifecyclePhase::TurnEnd);
        Ok(result)
    }

    /// Phase 4-5-c. Chat /reset 명령 처리용.
    ///
    /// pos, decode_step, prev_token을 0으로 reset한다.
    /// KV cache 자체는 caller (ChatSession::reset)가 `forward_mut()`으로
    /// 접근하여 reset한다.
    pub fn reset_pos(&mut self) {
        self.pos = 0;
        self.decode_step = 0;
        self.prev_token = 0;
    }

    /// Phase 4-5-c. Chat /reset + stateless chat 서버용. sampler 의 per-sequence
    /// state(rep penalty ring 등)를 clear 한다. `GreedySampler` 는 default no-op.
    pub fn reset_sampler(&mut self) {
        self.sampler.reset();
    }

    /// prefix cache restore 후 pos 를 외부에서 설정한다 (ENG-082, session::standard_happy 전용).
    ///
    /// `reset_pos`(0 고정)와 달리 임의 `pos` 를 허용한다. decode_step / prev_token 은 0 유지.
    pub(crate) fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// prefix cache full/partial restore 후 sampler token history 를 초기화한다.
    ///
    /// `DecodeLoop::prefill`은 완료 직후 `sampler.observe_token(t)` for all prompt tokens 를
    /// 호출하여 RepetitionPenaltySampler ring buffer 를 채운다. restore 경로는 prefill 을
    /// 건너뛰므로 이 메서드로 동일한 history 를 주입해야 한다 (INV-SAMPLING-HISTORY).
    ///
    /// `tokens`: prompt 토큰열 전체. GreedySampler 는 observe_token 이 no-op 이라 무비용.
    pub(crate) fn seed_sampler_history(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.sampler.observe_token(t);
        }
    }

    /// prefix cache save 위임 (ENG-085, INV-189). prefill 완료 직후 호출.
    ///
    /// `forward.save_kv_prefix(...)` 에 위임한다. ModelForward 이외 Forward 는 default no-op.
    pub(crate) fn save_kv_prefix(
        &self,
        path: &std::path::Path,
        model_hash: &[u8; 32],
        tokenizer_hash: &[u8; 32],
        token_ids: &[u32],
        last_logits: &[f32],
        backend: &dyn crate::backend::Backend,
    ) -> anyhow::Result<()> {
        self.forward.save_kv_prefix(
            path,
            model_hash,
            tokenizer_hash,
            token_ids,
            last_logits,
            backend,
        )
    }

    /// Phase 4-5-c. ChatSession::reset이 forward의 KV cache에 접근하기 위한 accessor.
    pub fn forward_mut(&mut self) -> &mut dyn Forward {
        &mut *self.forward
    }

    /// Phase 4-5-d: 현재 KV pos 읽기 (ChatSession 외부 캐시 동기화용).
    pub fn pos_snapshot(&self) -> usize {
        self.pos
    }

    /// Borrow the stop flag handle (caller can install a signal handler).
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_flag)
    }

    /// β-6: build 후 stage registry 교체(consuming). `ChatSession` 이 build 된 decode_loop 에
    /// `ChatStopStage` 등록 registry 를 주입할 때 사용한다(빌드 전 `with_pipeline` 과 동일 효과,
    /// 단 이미 build 된 인스턴스용). 기존 registry 는 폐기된다.
    pub fn with_pipeline_registry(mut self, registry: Arc<PipelineRegistry>) -> Self {
        self.pipeline = registry;
        self
    }
}

/// Builder with typestate marker. `F = NoForward` is the entry state;
/// `with_forward` transitions to `HasForward`, after which `.build()` exists.
pub struct DecodeLoopBuilder<F = NoForward> {
    forward: F,
    cmd_source: Option<Box<dyn CommandSource>>,
    sampler: Option<Box<dyn TokenSampler>>,
    stop_flag: Option<Arc<AtomicBool>>,
    dispatcher: Option<CommandDispatcher>,
    pipeline: Option<Arc<PipelineRegistry>>,
    // β-6 commit C: with_resilience 가 보관한 shared ResilienceAdapter — build() 에서 TickStage
    // (PostSample) 를 registry 로 submit 한다(with_pipeline 호출 순서 무관 보장).
    resilience_tick: Option<Arc<std::sync::Mutex<super::resilience_adapter::ResilienceAdapter>>>,
    kv_pos_handle: Option<Arc<dyn KVCacheFormat>>,
    pressure_source: Option<Arc<dyn PressureSource>>,
    kv_capacity: usize,
    kv_reencode_armed: bool,
}

impl Default for DecodeLoopBuilder<NoForward> {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeLoopBuilder<NoForward> {
    /// Start a new builder. Forward is required before `.build()`.
    pub fn new() -> Self {
        Self {
            forward: NoForward,
            cmd_source: None,
            sampler: None,
            stop_flag: None,
            dispatcher: None,
            pipeline: None,
            kv_pos_handle: None,
            pressure_source: None,
            resilience_tick: None,
            kv_capacity: 0,
            kv_reencode_armed: false,
        }
    }

    /// Supply the required Forward. Transitions to `HasForward` state.
    pub fn with_forward<T: Forward + 'static>(self, fwd: T) -> DecodeLoopBuilder<HasForward> {
        self.with_forward_boxed(Box::new(fwd))
    }

    /// Supply an already-boxed Forward (`Box<dyn Forward>`). The KV-mode registry's
    /// `build` fn-ptr returns a boxed forward (the concrete type is mode-private), so
    /// the mode-agnostic chat assembly wires it through this seam. Transitions to
    /// `HasForward` state.
    pub fn with_forward_boxed(self, fwd: Box<dyn Forward>) -> DecodeLoopBuilder<HasForward> {
        DecodeLoopBuilder {
            forward: HasForward(fwd),
            cmd_source: self.cmd_source,
            sampler: self.sampler,
            stop_flag: self.stop_flag,
            dispatcher: self.dispatcher,
            pipeline: self.pipeline,
            kv_pos_handle: self.kv_pos_handle,
            pressure_source: self.pressure_source,
            resilience_tick: self.resilience_tick,
            kv_capacity: self.kv_capacity,
            kv_reencode_armed: self.kv_reencode_armed,
        }
    }
}

// Optional setters available in any state. Order-independent.
impl<F> DecodeLoopBuilder<F> {
    pub fn with_cmd_source<T: CommandSource + 'static>(mut self, c: T) -> Self {
        self.cmd_source = Some(Box::new(c));
        self
    }
    pub fn with_sampler<T: TokenSampler + 'static>(mut self, t: T) -> Self {
        self.sampler = Some(Box::new(t));
        self
    }
    pub fn with_stop_flag(mut self, f: Arc<AtomicBool>) -> Self {
        self.stop_flag = Some(f);
        self
    }
    pub fn with_kv_capacity(mut self, c: usize) -> Self {
        self.kv_capacity = c;
        self
    }
    /// β-4 (v2 §5.4 A-1): 2-source 명령 분배자 [`CommandDispatcher`] 주입.
    /// 미주입(None) 이면 `cmd_source` 가 생산한 command 가 분배되지 않는다 (happy/chat — NoOp source
    /// 라 빈 Vec → 거동-0). dispatcher 는 registry·KV handle·CacheManager 를 보유해 evict directive 를
    /// OneShot EvictionStage 로 submit 하고 control 을 LoopControl 에 누적한다.
    pub fn with_command_dispatcher(mut self, d: CommandDispatcher) -> Self {
        self.dispatcher = Some(d);
        self
    }
    /// β-2: L2 stage registry 주입. 미주입 시 빈 registry (거동-0).
    pub fn with_pipeline(mut self, p: Arc<PipelineRegistry>) -> Self {
        self.pipeline = Some(p);
        self
    }

    /// β-3: pos-환류용 held handle 주입(§5.2.1 (가)). EvictionStage 가 prune 한 cache 의
    /// `current_pos` 를 driver 가 query 하는 layer-0 handle (`Arc<StandardFormat>` →
    /// `Arc<dyn KVCacheFormat>` coercion). 미주입(None) 이면 환류 skip → 거동-0.
    pub fn with_kv_pos_handle(mut self, h: Arc<dyn KVCacheFormat>) -> Self {
        self.kv_pos_handle = Some(h);
        self
    }

    /// Arm post-`PrefillEnd` GPU-plan invalidation for a `FormatReencodeStage` (L1-runtime per-layer
    /// re-encode). `build_standard_loop` calls this iff a format-reencode stage was submitted. Unset
    /// (every other path) → the driver makes no `on_kv_reencode` call = 거동-0.
    pub fn with_kv_reencode_invalidation(mut self) -> Self {
        self.kv_reencode_armed = true;
        self
    }

    /// β-5: graded system 압력 source 주입(`LocalPressureSource` 등). 미주입(happy/chat) 이면
    /// `StepInfo.pressure` = `Pressure::default()`(=0) 로 흘러 per-token syscall 0 (G4 happy-path
    /// 무주입). `build_bench_loop` 만 주입한다. driver 가 N-step 캐시로 query 한다.
    pub fn with_pressure_source(mut self, s: Arc<dyn PressureSource>) -> Self {
        self.pressure_source = Some(s);
        self
    }

    /// P3.3 + β-6 commit C: [`ResilienceAdapter`]를 cmd_source slot 에 주입하고, per-token tick 은
    /// [`TickStage`](crate::stages::system::tick::TickStage)(PostSample)로 build() 시점에 registry
    /// 에 submit 한다.
    ///
    /// 단일 인스턴스를 `Arc<Mutex<ResilienceAdapter>>`로 감싸 wrapper·stage 가 공유하므로
    /// ownership 이전 없이 충족한다. per-token Mutex lock 은 (poll 1회 + tick 1회) 정도라 contention
    /// 무시 가능. **with_pipeline 호출 순서 무관**: shared Arc 를 보관했다가 build() 에서 (그때
    /// 확정된) registry 로 submit 한다.
    ///
    /// β-7: v1 report slot(EngineReport/ReportWrapper) 제거 — IPC 송출(capability/qcf/swap_report)
    /// 은 decode loop 슬롯을 경유한 적이 없다(resilience_init 가 `CommandExecutor` 직접 호출).
    pub fn with_resilience(
        mut self,
        adapter: super::resilience_adapter::ResilienceAdapter,
    ) -> Self {
        use super::resilience_adapter::CmdSrcWrapper;
        use std::sync::Mutex;
        let shared = Arc::new(Mutex::new(adapter));
        self.cmd_source = Some(Box::new(CmdSrcWrapper(Arc::clone(&shared))));
        // β-6 commit C: tick 은 build() 에서 TickStage 로 submit (TickWrapper 제거).
        self.resilience_tick = Some(shared);
        self
    }
}

impl DecodeLoopBuilder<HasForward> {
    /// Assemble the decode loop. Optional components default to no-op impls
    /// from [`super::defaults`].
    pub fn build(self) -> DecodeLoop {
        let pipeline = self
            .pipeline
            .unwrap_or_else(|| Arc::new(PipelineRegistry::new()));
        // β-6 commit C: resilience 주입 시 per-token tick 을 TickStage(PostSample) 로 registry 에
        // submit. with_pipeline 호출 순서와 무관하게 build() 시점에 확정된 registry 로 등록한다.
        if let Some(adapter) = self.resilience_tick {
            pipeline.submit(Arc::new(crate::stages::system::tick::TickStage::new(
                adapter,
            )));
        }
        DecodeLoop {
            forward: self.forward.0,
            cmd_source: self
                .cmd_source
                .unwrap_or_else(|| Box::new(NoOpCommandSource)),
            sampler: self.sampler.unwrap_or_else(|| Box::new(GreedySampler)),
            stop_flag: self
                .stop_flag
                .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
            dispatcher: self.dispatcher,
            pipeline,
            kv_pos_handle: self.kv_pos_handle,
            profiler: OpProfiler::new(),
            pressure_source: self.pressure_source,
            cached_pressure: Pressure::default(),
            pos: 0,
            kv_occupancy: 0,
            decode_step: 0,
            prev_token: 0,
            kv_capacity: self.kv_capacity,
            kv_reencode_armed: self.kv_reencode_armed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward stub: prefill returns logits[0]=1.0, step returns logits with
    /// the next index biased highest so GreedySampler picks `step_count + 1`.
    struct MockForward {
        vocab: usize,
        step_count: usize,
    }

    impl Forward for MockForward {
        fn prefill(&mut self, _tokens: &[u32], _start_pos: usize) -> anyhow::Result<Vec<f32>> {
            let mut logits = vec![0.0_f32; self.vocab];
            logits[0] = 1.0;
            Ok(logits)
        }
        fn step(&mut self, _ctx: &StepCtx, _token: u32) -> anyhow::Result<Vec<f32>> {
            self.step_count += 1;
            let mut logits = vec![0.0_f32; self.vocab];
            let target = self.step_count % self.vocab;
            logits[target] = 1.0;
            Ok(logits)
        }
    }

    /// commit-3 wiring: `with_kv_reencode_invalidation()` makes the driver call
    /// `forward.on_kv_reencode()` after PrefillEnd (so a `FormatReencodeStage`'s dtype flip
    /// invalidates the fused GPU plan); the default leaves it unarmed (거동-0). Pins the live seam
    /// without a GPU: a recording Forward counts the calls.
    #[test]
    fn kv_reencode_armed_invokes_forward_after_prefill() {
        struct ReencodeRecorder {
            vocab: usize,
            calls: Arc<std::sync::Mutex<usize>>,
        }
        impl Forward for ReencodeRecorder {
            fn prefill(&mut self, _t: &[u32], _s: usize) -> anyhow::Result<Vec<f32>> {
                Ok(vec![0.0_f32; self.vocab])
            }
            fn step(&mut self, _c: &StepCtx, _t: u32) -> anyhow::Result<Vec<f32>> {
                Ok(vec![0.0_f32; self.vocab])
            }
            fn on_kv_reencode(&mut self) {
                *self.calls.lock().unwrap() += 1;
            }
        }

        let armed_calls = Arc::new(std::sync::Mutex::new(0usize));
        let mut armed = DecodeLoopBuilder::new()
            .with_forward(ReencodeRecorder {
                vocab: 8,
                calls: armed_calls.clone(),
            })
            .with_kv_reencode_invalidation()
            .build();
        armed.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(
            *armed_calls.lock().unwrap(),
            1,
            "armed → on_kv_reencode called once after PrefillEnd"
        );

        let unarmed_calls = Arc::new(std::sync::Mutex::new(0usize));
        let mut unarmed = DecodeLoopBuilder::new()
            .with_forward(ReencodeRecorder {
                vocab: 8,
                calls: unarmed_calls.clone(),
            })
            .build();
        unarmed.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(
            *unarmed_calls.lock().unwrap(),
            0,
            "unarmed (default) → no on_kv_reencode call (거동-0)"
        );
    }

    #[test]
    fn builder_runs_full_budget_with_defaults() {
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .build();
        let last_logits = loop_.prefill(&[1, 2, 3]).unwrap();
        // MockForward::prefill sets logits[0]=1.0 → GreedySampler picks 0.
        assert_eq!(last_logits.len(), 16);
        let result = loop_.run(5, 0).unwrap();
        assert_eq!(result.tokens_generated.len(), 5);
        assert_eq!(result.stopped_by, StopReason::BudgetExhausted);
        assert_eq!(result.final_pos, 3 + 5);
        // GreedySampler picks the highest logit each step. step_count = 1..=5
        // produces logits with target = step_count % vocab.
        assert_eq!(result.tokens_generated, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn stop_flag_short_circuits_loop() {
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .build();
        let stop = loop_.stop_flag();
        stop.store(true, Ordering::Release);
        let result = loop_.run(5, 0).unwrap();
        assert_eq!(result.tokens_generated.len(), 0);
        assert_eq!(result.stopped_by, StopReason::StopFlag);
    }

    #[test]
    fn custom_sampler_overrides_default() {
        struct AlwaysZero;
        impl TokenSampler for AlwaysZero {
            fn sample(&mut self, _ctx: &StepCtx, _logits: &[f32]) -> u32 {
                0
            }
        }
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_sampler(AlwaysZero)
            .build();
        let result = loop_.run(3, 7).unwrap();
        assert_eq!(result.tokens_generated, vec![0, 0, 0]);
    }

    /// Phase 4-4.7: TokenSampler::observe_token이 run() 진입 시 first_token 1회 +
    /// 매 step sampled 1회씩 호출되는지 검증. C2 (prompt seeding) 진입 후에는
    /// prefill 단계에서도 prompt 길이만큼 추가 호출되므로 expected가 갱신될 예정.
    #[test]
    fn observe_token_invoked_on_first_and_each_step() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrd};
        struct CountingSampler {
            observe_count: Arc<AtomicUsize>,
            next: u32,
        }
        impl TokenSampler for CountingSampler {
            fn sample(&mut self, _ctx: &StepCtx, _logits: &[f32]) -> u32 {
                let t = self.next;
                self.next = self.next.wrapping_add(1);
                t
            }
            fn observe_token(&mut self, _token: u32) {
                self.observe_count.fetch_add(1, AtomicOrd::Relaxed);
            }
        }
        let observe_count = Arc::new(AtomicUsize::new(0));
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_sampler(CountingSampler {
                observe_count: observe_count.clone(),
                next: 0,
            })
            .build();
        let _ = loop_.prefill(&[1, 2, 3]).unwrap();
        let _ = loop_.run(3, 7).unwrap();
        // C2 prompt seeding 활성: prompt 3회 + first_token 1회 + sampled 3회 = 7.
        assert_eq!(observe_count.load(AtomicOrd::Relaxed), 7);
    }

    // β-7: v1 `with_eviction`/`EvictionStage` trait 슬롯 검증 테스트
    // (eviction_pruned_advances_on_kv_prune_hook) 제거 — 슬롯이 삭제되고 eviction 은
    // stages::EvictionStage(PipelineStage)로 수렴했다. 등가 검증은
    // `tests/beta3_eviction_stage_equivalence.rs` 가 담당한다.

    // Phase 4-5-c / β-6: run_until_stop 테스트.
    //
    // β-6 통합 후 run_until_stop 은 stop 인자를 받지 않는다 — chat 의 ChatStopStage(DecodeEnd
    // 구독) 가 stop 판정을 담당한다. 테스트는 ChatStopStage 를 registry 에 등록하고 ChatStopSlot
    // 에 stop condition 을 arm 한 뒤 호출한다(ChatSession::run_turn 과 동일 배선).

    /// ChatStopStage + slot 을 구성해 `with_pipeline` 한 DecodeLoop builder 를 만든다.
    /// 반환된 slot 에 caller 가 stop condition 을 arm 한다.
    fn build_loop_with_stop_stage(
        vocab: usize,
    ) -> (
        DecodeLoopBuilder<HasForward>,
        Arc<crate::session::chat::stop_condition::ChatStopSlot>,
    ) {
        use crate::session::chat::stop_condition::{ChatStopSlot, ChatStopStage};
        let slot = ChatStopSlot::new();
        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(ChatStopStage::new(Arc::clone(&slot))));
        let builder = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(registry);
        (builder, slot)
    }

    /// StopCondition이 특정 token에서 true를 반환하면 loop가 종료된다.
    /// MockForward (vocab=16): step_count=1 → token 1, step_count=2 → token 2, ...
    /// stop_id=3으로 설정하면 3번째 step에서 token 3 sample 후 종료.
    /// generated에는 stop token이 포함되지 않는다 (1, 2만).
    #[test]
    fn run_until_stop_terminates_on_stop_token() {
        use crate::session::chat::stop_condition::ChatStopCondition;

        let (builder, slot) = build_loop_with_stop_stage(16);
        let mut loop_ = builder.build();
        let _ = loop_.prefill(&[0]).unwrap();
        // step_count=1→token1, 2→token2, 3→token3(stop)
        let cond = ChatStopCondition::new(vec![3], 2048);
        let result = {
            let _guard = slot.arm(&cond);
            loop_.run_until_stop(0).unwrap()
        };
        // token 3은 stop이므로 generated에 미포함 → [1, 2]
        assert_eq!(result.tokens_generated, vec![1, 2]);
        assert_eq!(result.stopped_by, StopReason::StopConditionMet);
    }

    /// max_pos overflow 안전망: pos >= max_pos이면 loop 종료. (G5: 마지막 토큰 미포함 보존)
    #[test]
    fn run_until_stop_terminates_on_max_pos() {
        use crate::session::chat::stop_condition::ChatStopCondition;

        let (builder, slot) = build_loop_with_stop_stage(16);
        let mut loop_ = builder.build();
        // prefill 1 token → pos=1
        let _ = loop_.prefill(&[0]).unwrap();
        // max_pos=3 → pos≥3이면 stop. 1+2 steps → pos=3 → break (step 2 token not pushed)
        let cond = ChatStopCondition::new(vec![], 3);
        let result = {
            let _guard = slot.arm(&cond);
            loop_.run_until_stop(0).unwrap()
        };
        // step1: sampled=1, pos→2, stop?(pos=2<3)→no → push. step2: sampled=2, pos→3, stop?(pos=3>=3)→yes
        assert_eq!(result.tokens_generated, vec![1]);
        assert_eq!(result.final_pos, 3);
    }

    // ── β-6 commit A: chat 거동 고정 (수렴 전 핀) ──────────────────────────

    /// β-6 핀 2: stop 토큰에 대해서도 tick(PostSample phase)이 **발화**한다. v1 run_until_stop
    /// 시맨틱 census: stop 체크는 tick/bookkeeping **후** 이므로 stop 토큰의 tick 도 1회 발화하고
    /// pos 만 증가하며 push 만 안 된다. commit C 후 tick 은 PostSample phase 로 이관됐으므로
    /// (TickStage), PostSample 발화 횟수 == 구 tick 횟수로 등가 검증한다. ChatStopStage(DecodeEnd)
    /// 가 PostSample 보다 뒤이므로 stop 토큰의 PostSample 도 발화.
    ///
    /// β-7: v1 (g) observer.on_step_end 슬롯이 삭제됐으므로 obs 단언은 제거 — per-step 관찰은
    /// PostSample stage 발화로 대표 검증한다.
    #[test]
    fn run_until_stop_fires_tick_on_stop_token() {
        use crate::pipeline::{LifecyclePhase as P2, PipelineStage as PS2, StageOutcome as SO2};
        use crate::session::chat::stop_condition::{
            ChatStopCondition, ChatStopSlot, ChatStopStage,
        };
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrd};

        // PostSample 발화 횟수를 세는 stage (구 tick_sink.on_token_generated 등가 — TickStage 도
        // PostSample 구독이라 발화 횟수가 동일).
        struct CountPostSampleStage {
            count: Arc<AtomicUsize>,
        }
        impl PS2 for CountPostSampleStage {
            fn name(&self) -> &str {
                "count.post_sample"
            }
            fn on_phase(
                &self,
                phase: &P2,
                _ctx: &mut crate::pipeline::StageContext<'_>,
            ) -> anyhow::Result<SO2> {
                if *phase == P2::PostSample {
                    self.count.fetch_add(1, AtomicOrd::Relaxed);
                }
                Ok(SO2::Continue)
            }
        }

        let tick_count = Arc::new(AtomicUsize::new(0));
        let slot = ChatStopSlot::new();
        let registry = Arc::new(PipelineRegistry::new());
        // CountPostSampleStage(PostSample) 를 먼저, ChatStopStage(DecodeEnd) 를 나중에 등록.
        registry.submit(Arc::new(CountPostSampleStage {
            count: tick_count.clone(),
        }));
        registry.submit(Arc::new(ChatStopStage::new(Arc::clone(&slot))));
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(registry)
            .build();
        let _ = loop_.prefill(&[0]).unwrap();
        // step1→token1(push), step2→token2(push), step3→token3=stop(미push, but tick 발화).
        let cond = ChatStopCondition::new(vec![3], 2048);
        let result = {
            let _guard = slot.arm(&cond);
            loop_.run_until_stop(0).unwrap()
        };
        assert_eq!(result.tokens_generated, vec![1, 2], "stop 토큰 미push");
        // 3 step (token1, token2, token3=stop) 모두 PostSample(tick) 발화 — stop 토큰 포함.
        assert_eq!(
            tick_count.load(AtomicOrd::Relaxed),
            3,
            "stop 토큰에도 PostSample(tick) 발화 (총 3 step)"
        );
    }

    /// β-6 핀 3: stop 체크가 pos-증가-후 타이밍. should_stop 에 전달되는 pos 인자가
    /// 매 호출에서 **증가된** pos (bookkeeping 후) 임을 recording StopCondition 으로 핀한다.
    /// v1: prev_token=sampled; pos+=1; decode_step+=1; → should_stop(sampled, self.pos).
    /// 통합 후 DecodeEnd 구독 stage 도 `ctx.step.pos == 증가된 pos` 를 봐야 한다.
    #[test]
    fn run_until_stop_checks_stop_with_post_increment_pos() {
        use crate::session::chat::stop_condition::StopCondition;
        use std::sync::Mutex as StdMutex;

        // (sampled, pos) 호출 인자를 전부 기록하고 항상 false 반환 (max_pos 로만 종료).
        struct RecordStop {
            log: Arc<StdMutex<Vec<(u32, usize)>>>,
            max_pos: usize,
        }
        impl StopCondition for RecordStop {
            fn should_stop(&self, sampled: u32, pos: usize) -> bool {
                self.log.lock().unwrap().push((sampled, pos));
                pos >= self.max_pos
            }
        }

        let log: Arc<StdMutex<Vec<(u32, usize)>>> = Arc::new(StdMutex::new(Vec::new()));
        let (builder, slot) = build_loop_with_stop_stage(16);
        let mut loop_ = builder.build();
        // prefill 1 token → pos=1. 이후 step 마다 pos: 2, 3, ...
        let _ = loop_.prefill(&[0]).unwrap();
        let cond = RecordStop {
            log: log.clone(),
            max_pos: 4, // pos>=4 에서 종료 → step1(pos=2), step2(pos=3), step3(pos=4=stop).
        };
        {
            let _guard = slot.arm(&cond);
            let _ = loop_.run_until_stop(0).unwrap();
        }

        let calls = log.lock().unwrap().clone();
        // step1: sampled=1, pos=2 (1→2 증가 후). step2: sampled=2, pos=3. step3: sampled=3, pos=4.
        assert_eq!(
            calls,
            vec![(1, 2), (2, 3), (3, 4)],
            "should_stop 의 pos 인자 == 증가된 pos (post-increment 타이밍)"
        );
    }

    /// reset_pos가 pos, decode_step, prev_token을 0으로 초기화한다.
    #[test]
    fn reset_pos_clears_loop_state() {
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 8,
                step_count: 0,
            })
            .build();
        let _ = loop_.prefill(&[1, 2, 3]).unwrap();
        let _ = loop_.run(2, 0).unwrap();
        // run 후 pos=5, decode_step=2, prev_token=?
        loop_.reset_pos();
        // reset 후 pos=0
        // forward_mut으로 Forward에 접근 가능함을 컴파일로 검증
        let _fwd: &mut dyn Forward = loop_.forward_mut();
    }

    // ── β-2 phase 배선 테스트 ──────────────────────────────────────────────

    use crate::pipeline::{
        LifecyclePhase as Phase, PipelineStage, StageContext as SCtx, StageOutcome,
    };
    use crate::session::pipeline_registry::PipelineRegistry;
    use std::sync::Mutex;

    /// 모든 phase를 기록하는 Persistent stage.
    struct RecordStage {
        log: Arc<Mutex<Vec<Phase>>>,
    }

    impl PipelineStage for RecordStage {
        fn name(&self) -> &str {
            "RecordStage"
        }
        fn on_phase(&self, phase: &Phase, _ctx: &mut SCtx<'_>) -> anyhow::Result<StageOutcome> {
            self.log.lock().unwrap().push(phase.clone());
            Ok(StageOutcome::Continue)
        }
    }

    /// 지정 phase에서 Stop(EosToken)을 반환하는 stage.
    struct StopAtStage {
        target: Phase,
        fired: Arc<Mutex<bool>>,
    }

    impl PipelineStage for StopAtStage {
        fn name(&self) -> &str {
            "StopAtStage"
        }
        fn on_phase(&self, phase: &Phase, _ctx: &mut SCtx<'_>) -> anyhow::Result<StageOutcome> {
            if phase == &self.target {
                *self.fired.lock().unwrap() = true;
                Ok(StageOutcome::Stop(crate::pipeline::StopReason::EosToken))
            } else {
                Ok(StageOutcome::Continue)
            }
        }
    }

    /// phase, pos, decode_step 을 기록하는 stage.
    struct SnapshotStage {
        log: Arc<Mutex<Vec<(Phase, usize, usize)>>>,
    }

    impl PipelineStage for SnapshotStage {
        fn name(&self) -> &str {
            "SnapshotStage"
        }
        fn on_phase(&self, phase: &Phase, ctx: &mut SCtx<'_>) -> anyhow::Result<StageOutcome> {
            self.log
                .lock()
                .unwrap()
                .push((phase.clone(), ctx.step.pos, ctx.step.decode_step));
            Ok(StageOutcome::Continue)
        }
    }

    /// prefill(&[1,2,3]) + run(2, 0) 시 정확한 phase 시퀀스를 검증한다.
    #[test]
    fn phase_emission_sequence_exact() {
        let log: Arc<Mutex<Vec<Phase>>> = Arc::new(Mutex::new(Vec::new()));
        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(RecordStage { log: log.clone() }));

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(Arc::clone(&registry))
            .build();

        loop_.prefill(&[1, 2, 3]).unwrap();
        loop_.run(2, 0).unwrap();

        let observed = log.lock().unwrap().clone();
        // β-3: 각 DecodeStart 직후 KvMutate, v1 eviction 후 WeightMutate 발화
        // (§5.2.1 (나) — command-poll 직후·forward 직전).
        let expected = vec![
            Phase::PrefillStart,
            Phase::PrefillEnd,
            Phase::DecodeStart,
            Phase::KvMutate,
            Phase::WeightMutate,
            Phase::PreForward,
            Phase::PostForward,
            Phase::PreSample,
            Phase::PostSample,
            Phase::DecodeEnd,
            Phase::DecodeStart,
            Phase::KvMutate,
            Phase::WeightMutate,
            Phase::PreForward,
            Phase::PostForward,
            Phase::PreSample,
            Phase::PostSample,
            Phase::DecodeEnd,
            Phase::Finalize,
        ];
        assert_eq!(observed, expected, "phase 시퀀스 불일치");
    }

    /// PostSample 에서 Stop(EosToken) 반환 시 tokens_generated 가 비어 있고
    /// stopped_by 가 v1 StopReason::EosToken 으로 수렴된다.
    #[test]
    fn stage_stop_at_post_sample_maps_to_v1_reason() {
        let fired = Arc::new(Mutex::new(false));
        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(StopAtStage {
            target: Phase::PostSample,
            fired: fired.clone(),
        }));

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(Arc::clone(&registry))
            .build();

        loop_.prefill(&[1]).unwrap();
        let result = loop_.run(5, 0).unwrap();

        assert_eq!(result.stopped_by, StopReason::EosToken, "v1 EosToken 수렴");
        assert!(
            result.tokens_generated.is_empty(),
            "PostSample 미push break — 토큰 없음"
        );
        assert!(*fired.lock().unwrap(), "StopAtStage 발화 확인");
    }

    /// β-6 통합: run_until_stop 이 run() 과 동일 phase 를 발화한다(TurnStart → per-token 6종
    /// → TurnEnd). β-2 의 "run_until_stop 미발화" 테스트를 통합 거동 검증으로 갱신.
    /// RecordStage(순회 우선) + ChatStopStage(stop 판정) 를 함께 등록 — token1 에서 stop.
    #[test]
    fn run_until_stop_dispatches_turn_and_decode_phases() {
        use crate::session::chat::stop_condition::{
            ChatStopCondition, ChatStopSlot, ChatStopStage,
        };

        let log: Arc<Mutex<Vec<Phase>>> = Arc::new(Mutex::new(Vec::new()));
        let slot = ChatStopSlot::new();
        let registry = Arc::new(PipelineRegistry::new());
        // RecordStage 를 먼저 등록 → DecodeEnd 에서 RecordStage(Continue, 기록) 후 ChatStopStage(Stop).
        registry.submit(Arc::new(RecordStage { log: log.clone() }));
        registry.submit(Arc::new(ChatStopStage::new(Arc::clone(&slot))));

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(Arc::clone(&registry))
            .build();

        // prefill 1회 → PrefillStart + PrefillEnd = 2 발화.
        loop_.prefill(&[0]).unwrap();
        log.lock().unwrap().clear(); // prefill phase 제거 — run_until_stop 발화만 본다.

        // step1 → token1 = stop. DecodeEnd 에서 ChatStopStage Stop → break.
        let cond = ChatStopCondition::new(vec![1], 2048);
        let result = {
            let _guard = slot.arm(&cond);
            loop_.run_until_stop(0).unwrap()
        };
        assert_eq!(
            result.tokens_generated,
            Vec::<u32>::new(),
            "token1=stop 미push"
        );
        assert_eq!(result.stopped_by, StopReason::StopConditionMet);

        let observed = log.lock().unwrap().clone();
        // TurnStart → 1 step (DecodeStart..DecodeEnd) → DecodeEnd 에서 Stop break → TurnEnd.
        let expected = vec![
            Phase::TurnStart,
            Phase::DecodeStart,
            Phase::KvMutate,
            Phase::WeightMutate,
            Phase::PreForward,
            Phase::PostForward,
            Phase::PreSample,
            Phase::PostSample,
            Phase::DecodeEnd,
            Phase::TurnEnd,
        ];
        assert_eq!(observed, expected, "run_until_stop phase 시퀀스 (통합)");
    }

    /// phase 별 StepInfo(pos, decode_step) 스냅샷 값을 검증한다.
    #[test]
    fn step_info_snapshot_values() {
        let log: Arc<Mutex<Vec<(Phase, usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(SnapshotStage { log: log.clone() }));

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(Arc::clone(&registry))
            .build();

        // prefill(&[1,2,3]) → pos=0→3
        loop_.prefill(&[1, 2, 3]).unwrap();
        // run(1, 0) → 1 decode step
        loop_.run(1, 0).unwrap();

        let observed = log.lock().unwrap().clone();

        // PrefillStart: pos=0, decode_step=0 (prefill 직전)
        let prefill_start = observed
            .iter()
            .find(|(p, _, _)| p == &Phase::PrefillStart)
            .unwrap();
        assert_eq!(
            (prefill_start.1, prefill_start.2),
            (0, 0),
            "PrefillStart pos==0"
        );

        // PrefillEnd: pos=3, decode_step=0 (pos 갱신 후)
        let prefill_end = observed
            .iter()
            .find(|(p, _, _)| p == &Phase::PrefillEnd)
            .unwrap();
        assert_eq!((prefill_end.1, prefill_end.2), (3, 0), "PrefillEnd pos==3");

        // DecodeStart: pos=3, decode_step=0 (루프 진입 시)
        let decode_start = observed
            .iter()
            .find(|(p, _, _)| p == &Phase::DecodeStart)
            .unwrap();
        assert_eq!(
            (decode_start.1, decode_start.2),
            (3, 0),
            "DecodeStart pos==3, step==0"
        );

        // DecodeEnd: pos=4, decode_step=1 (bookkeeping 후)
        let decode_end = observed
            .iter()
            .find(|(p, _, _)| p == &Phase::DecodeEnd)
            .unwrap();
        assert_eq!(
            (decode_end.1, decode_end.2),
            (4, 1),
            "DecodeEnd pos==4, step==1"
        );
    }

    // ── eviction-후 RoPE-position drift 회귀 테스트 ────────────────────────

    /// 회귀: in-loop eviction(KvMutate, sliding prefix+window = non-contiguous keep)이 KV-cache 를
    /// compaction 한 뒤에도 driver 의 `self.pos`(= 다음 forward 의 RoPE `start_pos`)는 **압축된
    /// current_pos 로 collapse 되지 않고 진짜 누적 위치를 유지**해야 한다. compaction 은 생존 키를
    /// memmove 만 하고 재회전하지 않으므로(원위치 RoPE phase 보존), 압축 count 로 collapse 하면 다음
    /// 토큰 query 가 생존 키보다 낮은 위치로 회전해 학습된 상대거리가 깨진다(eval/ppl 이 명시 금지하는
    /// "severe NLL degradation"). 동시에 GPU plan invalidation(`on_kv_prune`)은 점유 shrink 시 1회
    /// 발화해야 한다.
    ///
    /// **비-tautology**: forward 가 받은 `ctx.pos` 를 기록해 검증한다. 구 collapse 동작에선 이 값이
    /// 압축 count(≈14)였고, 본 단언(== N_TOKENS=120)은 그때 FAIL 한다.
    #[test]
    fn eviction_keeps_cumulative_rope_pos_not_collapsed_to_compacted() {
        use crate::backend::Backend;
        use crate::backend::cpu::CpuBackend;
        use crate::buffer::DType;
        use crate::format::KVCacheFormat;
        use crate::inference::sampling::StepCtx;
        use crate::kv::cache_manager::CacheManager;
        use crate::kv::eviction::stage_registry::sliding_backed_policy;
        use crate::kv::kv_cache::KVCache;
        use crate::kv::standard_format::StandardFormat;
        use crate::memory::host::shared::SharedBuffer;
        use crate::resilience::sys_monitor::NoOpMonitor;
        use crate::session::forward::Forward;
        use crate::shape::Shape;
        use crate::stages::kv::eviction::EvictionStage;
        use crate::tensor::Tensor;
        use std::sync::Mutex as StdMutex;

        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 32;
        const MAX_SEQ: usize = 128;
        const N_TOKENS: usize = 120; // ratio=0.3 → remove=84 ≥ MIN_EVICT_TOKENS(64).

        // forward 가 매 step 받는 `ctx.pos`(= RoPE start_pos)와 `on_kv_prune(new_pos)` 호출을 기록하는
        // forward stub. step 은 KVCache 를 advance 하지 않으므로(mock) handle.current_pos 는 eviction
        // 후 값으로 고정 — 단일-step(budget=1) 검증과 함께 reconcile 재발동 아티팩트를 회피한다.
        struct RecordingForward {
            vocab: usize,
            step_positions: Arc<StdMutex<Vec<usize>>>,
            prune_calls: Arc<StdMutex<Vec<usize>>>,
        }
        impl Forward for RecordingForward {
            fn prefill(&mut self, _tokens: &[u32], _start_pos: usize) -> anyhow::Result<Vec<f32>> {
                let mut l = vec![0.0_f32; self.vocab];
                l[0] = 1.0;
                Ok(l)
            }
            fn step(&mut self, ctx: &StepCtx, _token: u32) -> anyhow::Result<Vec<f32>> {
                self.step_positions.lock().unwrap().push(ctx.pos);
                let mut l = vec![0.0_f32; self.vocab];
                l[1] = 1.0;
                Ok(l)
            }
            fn on_kv_prune(&mut self, new_pos: usize) {
                self.prune_calls.lock().unwrap().push(new_pos);
            }
        }

        // 실물 F32 KVCache (current_pos = N_TOKENS) 를 StandardFormat 으로 wrap.
        let total = MAX_SEQ * KV_HEADS * HEAD_DIM;
        let k_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend);
        let mut cache = KVCache::new(k, v, MAX_SEQ);
        cache.current_pos = N_TOKENS;
        let handle = Arc::new(StandardFormat::new(0, cache));

        // OneShot EvictionStage: sliding(window=10, prefix=4) → prefix[0..4)+recent[110..120) keep =
        // non-contiguous(중간 구멍) → 생존자의 원위치 ≠ 압축 슬롯.
        let policy = sliding_backed_policy(10, 4);
        let cm = CacheManager::new(policy, Box::new(NoOpMonitor), usize::MAX, 0.3);
        let stage = EvictionStage::one_shot(vec![handle.clone()], Arc::new(Mutex::new(cm)), 0.3);

        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(stage));
        assert_eq!(registry.len(), 1);

        let step_positions = Arc::new(StdMutex::new(Vec::<usize>::new()));
        let prune_calls = Arc::new(StdMutex::new(Vec::<usize>::new()));
        let pos_handle: Arc<dyn KVCacheFormat> = handle.clone();
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(RecordingForward {
                vocab: 16,
                step_positions: step_positions.clone(),
                prune_calls: prune_calls.clone(),
            })
            .with_kv_capacity(2048)
            .with_pipeline(Arc::clone(&registry))
            .with_kv_pos_handle(pos_handle)
            .build();

        // prefill N_TOKENS → driver pos = N_TOKENS. budget=1: 첫 step 의 KvMutate 가 eviction 을 발화.
        let prompt: Vec<u32> = (0..N_TOKENS as u32).collect();
        loop_.prefill(&prompt).unwrap();
        let result = loop_.run(1, 0).unwrap();

        // 비-vacuous 가드: eviction 이 실제로 cache 를 compaction 했다.
        let occupancy = handle.current_pos();
        assert!(
            occupancy < N_TOKENS,
            "eviction 이 cache 를 prune 해야 함 (occupancy={occupancy})"
        );

        // ★핵심(비-tautology): eviction 직후 step 의 RoPE start_pos 는 압축 count 가 아니라 진짜 누적
        // 위치(N_TOKENS)다. 구 collapse 동작에선 occupancy(≈14)였다 → 본 단언이 FAIL.
        let seen = step_positions.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec![N_TOKENS],
            "eviction-후 forward start_pos == 진짜 누적 위치(압축 count 아님)"
        );
        assert!(
            seen[0] > occupancy,
            "start_pos(누적, {}) 와 cache occupancy(압축, {occupancy}) 는 분리되어야 함",
            seen[0]
        );

        // GPU plan invalidation: 점유 shrink 1회 → on_kv_prune 1회(압축 occupancy 로).
        let prunes = prune_calls.lock().unwrap().clone();
        assert_eq!(
            prunes,
            vec![occupancy],
            "on_kv_prune 은 shrink 당 1회, 압축 occupancy 로 발화"
        );

        // loop pos 는 누적 유지 → step 진행으로 +1.
        assert_eq!(
            result.final_pos,
            N_TOKENS + 1,
            "final_pos 는 누적 위치(collapse 아님)에서 진행"
        );
        // telemetry 점유는 누적이 아니라 압축 occupancy.
        assert_eq!(
            result.final_cache_pos, occupancy,
            "final_cache_pos 는 cache 점유(occupancy) 여야 함"
        );
        // OneShot Consumed → registry GC.
        assert_eq!(registry.len(), 0, "OneShot Consumed 후 registry GC");
    }

    /// 회귀(강화): eviction 1회 후 후속 step 들에서 cache 가 정상 성장(+1/step)할 때
    /// `on_kv_prune` 이 **재발화하지 않는다**. 이게 shrink-detection 설계의 핵심 — 단순히
    /// "collapse 금지"만 하고 트리거를 `new_pos < self.pos` 로 두면 eviction 후 pos·occupancy 가
    /// 영구 분리돼 매 step on_kv_prune 이 발화(= GPU plan 매 step 무효화) 한다. budget>=2 + cache 를
    /// 실제로 advance 하는 forward 로 그 trap 부재를 핀한다.
    #[test]
    fn on_kv_prune_fires_once_per_eviction_not_every_step() {
        use crate::backend::Backend;
        use crate::backend::cpu::CpuBackend;
        use crate::buffer::DType;
        use crate::format::KVCacheFormat;
        use crate::inference::sampling::StepCtx;
        use crate::kv::cache_manager::CacheManager;
        use crate::kv::eviction::stage_registry::sliding_backed_policy;
        use crate::kv::kv_cache::KVCache;
        use crate::kv::standard_format::StandardFormat;
        use crate::memory::host::shared::SharedBuffer;
        use crate::resilience::sys_monitor::NoOpMonitor;
        use crate::session::forward::Forward;
        use crate::shape::Shape;
        use crate::stages::kv::eviction::EvictionStage;
        use crate::tensor::Tensor;
        use std::sync::Mutex as StdMutex;

        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 32;
        const MAX_SEQ: usize = 256;
        const N_TOKENS: usize = 120;
        const BUDGET: usize = 4;

        // step() 가 실제 KVCache 를 +1 advance 하는 forward(= production write_kv 의 점유 증가 모사).
        // 이로써 eviction 후 후속 step 에서 occupancy 가 kv_occupancy 위로 자라며, shrink 미감지 →
        // on_kv_prune 미발화를 검증할 수 있다.
        struct AdvancingForward {
            vocab: usize,
            cache: Arc<StandardFormat>,
            prune_calls: Arc<StdMutex<Vec<usize>>>,
        }
        impl Forward for AdvancingForward {
            fn prefill(&mut self, _tokens: &[u32], _start_pos: usize) -> anyhow::Result<Vec<f32>> {
                let mut l = vec![0.0_f32; self.vocab];
                l[0] = 1.0;
                Ok(l)
            }
            fn step(&mut self, _ctx: &StepCtx, _token: u32) -> anyhow::Result<Vec<f32>> {
                // production forward.step 의 write_kv 처럼 점유를 1 늘린다.
                self.cache.with_cache_mut(|c| c.current_pos += 1);
                let mut l = vec![0.0_f32; self.vocab];
                l[1] = 1.0;
                Ok(l)
            }
            fn on_kv_prune(&mut self, new_pos: usize) {
                self.prune_calls.lock().unwrap().push(new_pos);
            }
        }

        let total = MAX_SEQ * KV_HEADS * HEAD_DIM;
        let k_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend);
        let mut cache = KVCache::new(k, v, MAX_SEQ);
        cache.current_pos = N_TOKENS;
        let handle = Arc::new(StandardFormat::new(0, cache));

        // OneShot eviction: 첫 KvMutate 에서 1회만 prune (Consumed→GC). 후속 step 엔 eviction 없음.
        let policy = sliding_backed_policy(10, 4);
        let cm = CacheManager::new(policy, Box::new(NoOpMonitor), usize::MAX, 0.3);
        let stage = EvictionStage::one_shot(vec![handle.clone()], Arc::new(Mutex::new(cm)), 0.3);
        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(stage));

        let prune_calls = Arc::new(StdMutex::new(Vec::<usize>::new()));
        let pos_handle: Arc<dyn KVCacheFormat> = handle.clone();
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(AdvancingForward {
                vocab: 16,
                cache: handle.clone(),
                prune_calls: prune_calls.clone(),
            })
            .with_kv_capacity(MAX_SEQ)
            .with_pipeline(Arc::clone(&registry))
            .with_kv_pos_handle(pos_handle)
            .build();

        let prompt: Vec<u32> = (0..N_TOKENS as u32).collect();
        loop_.prefill(&prompt).unwrap();
        let result = loop_.run(BUDGET, 0).unwrap();

        // eviction 이 실제로 발생(비-vacuous): 첫 step 후 점유는 prune 값 + 후속 advance.
        let prunes = prune_calls.lock().unwrap().clone();
        // ★핵심: on_kv_prune 은 eviction 1회에 대해 정확히 1회. budget=4 step 중 후속 3 step 은
        // cache 가 성장(occupancy > kv_occupancy)하므로 재발화하지 않는다.
        assert_eq!(
            prunes.len(),
            1,
            "on_kv_prune 은 eviction 당 1회만 (매 step 재발화 금지) — got {prunes:?}"
        );
        // 첫 prune 은 압축 occupancy(< N_TOKENS)로 발화.
        assert!(
            prunes[0] < N_TOKENS,
            "prune occupancy 는 압축값(<{N_TOKENS}) — got {}",
            prunes[0]
        );
        // self.pos 는 누적 유지: prefill(120) + budget(4) step.
        assert_eq!(
            result.final_pos,
            N_TOKENS + BUDGET,
            "final_pos 누적(collapse 아님)"
        );
        // 최종 점유 = prune 후 occupancy + 후속 step advance 수.
        assert_eq!(
            result.final_cache_pos,
            handle.current_pos(),
            "final_cache_pos == 실제 cache 점유"
        );
        assert!(
            result.final_cache_pos < result.final_pos,
            "eviction 후 점유({}) < 누적 위치({})",
            result.final_cache_pos,
            result.final_pos
        );
    }
}
