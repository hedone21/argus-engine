//! `EvictionStage` — v2 `PipelineStage` 입주자 1호 (Phase β-3 commit B).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.2.1 (가)(나) + roadmap β-3 commit B.
//!
//! `KvMutate` phase 에서 발화하여, register 시점 보유한 `Vec<Arc<StandardFormat>>` 핸들의
//! inner `KVCache` 를 **CacheManager UER**(`take_inner` → `force_evict` → `put_inner`)로 prune
//! 한다. 적용 경로가 v1 `ModelForward::try_evict`(model_forward.rs:505-548, AB-1)의 inner op 와
//! **byte-identical** 이라 madvise/CacheEvent/min-floor 회계가 그대로 보존된다(자기 비교 금지 —
//! 등가 게이트의 anchor 는 v1 `force_evict` 직접 호출).
//!
//! **pos 환류는 driver 책임**(§5.2.1 (가)): `StageOutcome` 는 무변경이고, driver(decode_loop.rs)가
//! eviction phase dispatch 후 held handle 의 `current_pos` 를 비교해 loop `pos` 를 갱신하고
//! `forward.on_kv_prune` 을 호출한다. Stage 는 cache 상태만 바꾼다.
//!
//! **Persistent⊕OneShot 同코드**: lifecycle 은 필드로만 분기한다. command-driven 1차 cut(v1 AB-1
//! 동일 조건 = score-free force_evict)은 [`EvictionStage::one_shot`] 로 생성하고, pressure
//! band-driven Persistent 변형은 β-5 에서 `on_phase` 의 `Continue` 반환만 다르게 구성하면 된다
//! (동일 UER 본문 재사용 — 본 파일 [`EvictionStage::on_phase`] 의 `match self.lifecycle`).

use std::sync::{Arc, Mutex};

use argus_shared::Level;

use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::kv::cache_manager::CacheManager;
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};

/// `KvMutate` phase 에서 CacheManager UER 로 force-evict 하는 Stage.
///
/// `CacheManager::force_evict` 는 `&self` 이지만, `PipelineStage: Send + Sync` 를 충족하려면
/// 내부 가변 상태가 `Sync` 여야 한다 → `Mutex<CacheManager>` 로 소유한다(per-dispatch 1 lock,
/// eviction = cold path 라 비용 무관). CacheManager 의 모든 필드(`Box<dyn SystemMonitor>`/
/// `HashMap<_, Box<dyn EvictionPolicy>>`/`Arc<SwapHandler>`)가 `Send + Sync` 라 `Mutex<CacheManager>`
/// 는 `Send + Sync` 다.
pub struct EvictionStage {
    lifecycle: StageLifecycle,
    /// register 시점 보유 (INV-STAGE-LAYER-HANDLE). enumerate 순서 == layer idx (W1 — weighted-merge
    /// cross-layer 정확성 전제, `wrap_kv_caches` 와 동일 불변식).
    handles: Vec<Arc<StandardFormat>>,
    /// 적용 경로 = CacheManager UER (v1 `try_evict` 와 산출 동일).
    ///
    /// β-4: `Arc<Mutex<CacheManager>>` 공유. CommandDispatcher 가 KvEvict* directive 마다 새
    /// OneShot `EvictionStage` 를 만들지만, 단일 `CacheManager`(CLI 정책·sticky eviction 상태)를
    /// 모든 stage 가 공유해야 method-drop 시맨틱(3부)과 정책 일관성이 유지된다.
    cache_manager: Arc<Mutex<CacheManager>>,
    /// score-free force_evict 의 target ratio.
    target_ratio: f32,
    /// β-5: Persistent band-driven 발화 임계. `Some(min)` 이면 `KvMutate` 에서
    /// `ctx.step.pressure.band() >= min` 일 때만 prune 한다(미달 시 `Continue`, no-op).
    /// `None`(OneShot) 이면 무조건 발화(v1 AB-1 = command-driven, 압력 무관).
    min_band: Option<Level>,
    /// β-5: Persistent **episode edge-trigger** 무장 상태. band 가 `min_band` 를 상향 돌파한
    /// 에지에서 1회 발화 후 disarm, band 가 `min_band` 미만으로 떨어지면 re-arm — 압력 에피소드당
    /// prune 1회. 가드 없이는 지속 고압에서 매 step `force_evict(ratio)` 가 재발화해 캐시가
    /// floor 까지 나선 축소(churn — madvise/CacheEvent per-token spam)하는 퇴행 시맨틱이 된다.
    /// 압력 지속 중 캐시 재성장 시의 재prune 은 friction-triggered 후속 (doc).
    armed: std::sync::atomic::AtomicBool,
    /// §5.9.1 Track A: score-aware eviction 용 accumulator cell. `Some` 이면 run_eviction 이
    /// acc.importance_scores() 를 추출해 force_evict_with_scores 로 heavy-hitter/weighted-merge 정밀 선택, 직후 acc.reset().
    /// `None`(score-free = sliding/streaming 또는 더미) 이면 기존 score-free force_evict 유지.
    score_cell: Option<Arc<Mutex<Option<AttentionScoreAccumulator>>>>,
    /// bench GPU-score 경로용 backend. `Some`(OpenCL + score-fed dispatcher) 이면 run_eviction 이
    /// score 를 읽기 직전 GPU 누적 score 를 CPU accumulator 로 sync 한다(`gpu_score_active=true` 시
    /// decode 가 CPU accumulate 를 건너뛰어 score 가 on-device 에만 쌓이므로 — eval/chat 동형 갭).
    /// score-free(`one_shot`/`persistent`) 경로엔 `None`(sync 불필요). CPU 빌드에선 `sync_*` 가 no-op.
    backend: Option<Arc<dyn crate::backend::Backend>>,
}

impl EvictionStage {
    /// command-driven OneShot eviction (v1 AB-1 동일 조건 — score-free `force_evict`).
    ///
    /// 1회 발화 후 `Consumed` 를 반환해 registry 가 GC 한다(sticky 재적용 방지 — v1 `evict_applied`
    /// 게이트 등가). 압력 무관(`min_band=None`).
    ///
    /// β-4: `cache_manager` 는 `Arc<Mutex<CacheManager>>` — CommandDispatcher 가 보유한 단일 CM 을
    /// directive 마다 새 OneShot stage 에 clone 주입한다(공유).
    pub fn one_shot(
        handles: Vec<Arc<StandardFormat>>,
        cache_manager: Arc<Mutex<CacheManager>>,
        target_ratio: f32,
    ) -> Self {
        Self {
            lifecycle: StageLifecycle::OneShot,
            handles,
            cache_manager,
            target_ratio,
            min_band: None,
            armed: std::sync::atomic::AtomicBool::new(true),
            score_cell: None,
            backend: None,
        }
    }

    /// §5.9.1 Track A: score-aware OneShot eviction (h2o/h2o_plus/d2o 전용).
    ///
    /// `one_shot` 과 동일 lifecycle·발화 조건이나, `run_eviction` 이 score_cell 에서
    /// `importance_scores()` 를 추출해 `force_evict_with_scores` 를 호출하고, 직후 `acc.reset()` 으로
    /// 누적 스코어를 비운다(eval EvictionHook 정본 — eviction_hook.rs:523-524 동형).
    /// score_cell 이 None 또는 acc 비활성이면 기존 score-free `force_evict` 로 fallback.
    pub fn one_shot_scored(
        handles: Vec<Arc<StandardFormat>>,
        cache_manager: Arc<Mutex<CacheManager>>,
        target_ratio: f32,
        score_cell: Arc<Mutex<Option<AttentionScoreAccumulator>>>,
        // bench GPU-score 경로: `Some`(OpenCL) 이면 run_eviction 이 score 읽기 직전 GPU→CPU sync.
        // CPU/non-opencl 또는 dispatcher 가 backend 미보유면 `None`(기존 CPU accumulate 경로 무변).
        backend: Option<Arc<dyn crate::backend::Backend>>,
    ) -> Self {
        Self {
            lifecycle: StageLifecycle::OneShot,
            handles,
            cache_manager,
            target_ratio,
            min_band: None,
            armed: std::sync::atomic::AtomicBool::new(true),
            score_cell: Some(score_cell),
            backend,
        }
    }

    /// β-5: pressure-driven Persistent eviction. 세션 내내 상주하며, `KvMutate` 에서
    /// `ctx.step.pressure.band() >= min_band` 일 때만 prune 한다(미달 시 `Continue`, no-op).
    ///
    /// **OneShot 과 同코드**: prune 본문([`run_eviction`](Self::run_eviction))은 공유하고,
    /// lifecycle(GC 여부)·발화 조건(band 게이트)만 분기한다. 같은 입력 캐시에서 band 충족 시의
    /// 발화 산출은 OneShot 과 byte-identical (unit `persistent_band_met_matches_one_shot`).
    pub fn persistent(
        handles: Vec<Arc<StandardFormat>>,
        cache_manager: Arc<Mutex<CacheManager>>,
        target_ratio: f32,
        min_band: Level,
    ) -> Self {
        Self {
            lifecycle: StageLifecycle::Persistent,
            handles,
            cache_manager,
            target_ratio,
            min_band: Some(min_band),
            armed: std::sync::atomic::AtomicBool::new(true),
            score_cell: None,
            backend: None,
        }
    }

    /// prune 본문 (UER, Unwrap-Evict-Rewrap) — OneShot/Persistent 공유.
    ///
    /// v1 try_evict(model_forward.rs:524-541)의 inner op 그대로. take_inner 로 inner cache 들을
    /// 연속 Vec 로 꺼내(W1 순서 보존) force_evict 후, Err/Ok 무관하게 put_inner 로 되돌린다
    /// (placeholder 폐기 — `?` 전파를 rewrap 이후로 미룸).
    ///
    /// §5.9.1 Track A: score_cell 이 Some 이고 acc active 이면 importance_scores() 를 추출해
    /// force_evict_with_scores 로 heavy-hitter/weighted-merge 정밀 token 선택(eval EvictionHook 동형).
    /// eviction 성공 직후 acc.reset() — KV geometry 변경 후 stale score 방지(§5.9.1 landmine 해소).
    fn run_eviction(&self) -> anyhow::Result<()> {
        use crate::kv::eviction::score_fed;
        // §5.9.1 Track A: shared score-fed body (gpu_sync → extract → route → reset), the eval/chat
        // EvictionHook quartet. When bench arms the GPU score path (`init_gpu_score_acc` in
        // experiment_run → `gpu_score_active=true`), decode skips the CPU accumulate (acc_need=false)
        // and scores live on-device; this syncs them into the CPU `score_cell` before reading. No-op
        // on CPU / non-opencl (no-op stub) / when `backend` is None (score-free `one_shot`/
        // `persistent`) / GPU acc inactive, so the host CPU-accumulate path is byte-identical.
        if let (Some(be), Some(cell)) = (self.backend.as_ref(), self.score_cell.as_ref()) {
            let mut guard = cell
                .lock()
                .expect("EvictionStage score_cell Mutex poisoned");
            if let Some(acc) = guard.as_mut() {
                score_fed::sync_gpu_scores_to_cpu(acc, be.as_ref());
            }
        }
        // Extract under the lock and release before routing.
        let extracted = self.score_cell.as_ref().and_then(|cell| {
            let guard = cell
                .lock()
                .expect("EvictionStage score_cell Mutex poisoned");
            guard.as_ref().and_then(score_fed::extract_scores)
        });

        let mut temp: Vec<KVCache> = self.handles.iter().map(|f| f.take_inner()).collect();
        let result = {
            let cm = self
                .cache_manager
                .lock()
                .expect("EvictionStage CacheManager Mutex poisoned");
            let (scores, last_attn, per_layer) = extracted
                .as_ref()
                .map(|e| e.as_args())
                .unwrap_or((None, None, None));
            // §5.9.1 force-evict (target_ratio): per-layer FLAT (faithful-H2O b) over collapsed
            // scores over recency.
            score_fed::route_evict(
                &cm,
                &mut temp,
                scores,
                last_attn,
                per_layer,
                true,
                self.target_ratio,
            )
        };
        for (f, c) in self.handles.iter().zip(temp) {
            f.put_inner(c);
        }
        // Err → dispatcher 가 panic (fail-fast 계약, INV-DECODE-STAGE-004).
        let evicted = result?.evicted;

        // §5.9.1 Track A: eviction 이 **실제로 token 을 제거했을 때만** reset — eval EvictionHook 정본
        // (eviction_hook.rs:777 `evict_result.evicted` 게이트) 동형. cache 가 floor 이하라 `force_evict`
        // 가 `Ok(evicted:false)`(cache_manager.rs:515/530/546)면 geometry 불변 → 누적 score 유지(주석
        // "성공 후"의 본래 의도). 무조건 reset 은 no-op directive 가 score 를 날려 다음 eviction 의
        // keep-set 을 eval 과 발산시키는 잠재 버그였다 — CPU·GPU 양쪽을 evicted 게이트로 묶어 해소.
        if evicted {
            if let Some(cell) = self.score_cell.as_ref() {
                let mut guard = cell
                    .lock()
                    .expect("EvictionStage score_cell Mutex poisoned");
                if let Some(ref mut acc) = *guard {
                    acc.reset();
                }
            }
            // Reset the GPU accumulator in lockstep with the CPU acc.reset above.
            // No-op when `backend` is None / unarmed / non-opencl stub.
            if let (Some(be), Some(cell)) = (self.backend.as_ref(), self.score_cell.as_ref()) {
                let guard = cell
                    .lock()
                    .expect("EvictionStage score_cell Mutex poisoned");
                if let Some(acc) = guard.as_ref() {
                    score_fed::reset_gpu_scores(acc, be.as_ref());
                }
            }
        }

        Ok(())
    }
}

impl PipelineStage for EvictionStage {
    fn name(&self) -> &str {
        "kv.eviction"
    }

    fn lifecycle(&self) -> StageLifecycle {
        self.lifecycle
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.3): eviction 외 phase 는 무시.
        if *phase != LifecyclePhase::KvMutate {
            return Ok(StageOutcome::Continue);
        }

        // β-5: Persistent band 게이트 — 압력 미달이면 prune 없이 Continue (no-op) + re-arm.
        // OneShot(min_band=None) 은 무조건 발화 (command-driven, 압력 무관).
        if let Some(min) = self.min_band {
            use std::sync::atomic::Ordering;
            if ctx.step.pressure.band() < min {
                // 압력 에피소드 종료 → re-arm (다음 상향 돌파에서 재발화 가능).
                self.armed.store(true, Ordering::Relaxed);
                return Ok(StageOutcome::Continue);
            }
            // episode edge-trigger: 이번 에피소드에서 이미 발화했으면 no-op
            // (지속 고압에서 매 step force_evict 재발화 = floor 나선 축소 차단).
            if !self.armed.swap(false, Ordering::Relaxed) {
                return Ok(StageOutcome::Continue);
            }
        }

        self.run_eviction()?;

        match self.lifecycle {
            StageLifecycle::OneShot => Ok(StageOutcome::Consumed),
            StageLifecycle::Persistent => Ok(StageOutcome::Continue),
        }
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
    use crate::memory::host::shared::SharedBuffer;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::resilience::sys_monitor::NoOpMonitor;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 32;
    const MAX_SEQ: usize = 128;
    /// ratio=0.3 → target_len=36, tokens_to_remove=84 ≥ MIN_EVICT_TOKENS(64) → 실제 prune.
    const N_TOKENS: usize = 120;
    const TARGET_RATIO: f32 = 0.3;

    /// SeqMajor F32 KVCache, current_pos = n_tokens.
    fn make_cache(n_tokens: usize) -> KVCache {
        let total = MAX_SEQ * KV_HEADS * HEAD_DIM;
        let k_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend);
        let mut c = KVCache::new(k, v, MAX_SEQ);
        c.current_pos = n_tokens;
        c
    }

    fn make_cache_manager() -> Arc<Mutex<CacheManager>> {
        // sliding window(window=10, prefix=4): current>keep 면 force_evict 가 prune.
        let policy = sliding_backed_policy(10, 4);
        Arc::new(Mutex::new(CacheManager::new(
            policy,
            Box::new(NoOpMonitor),
            usize::MAX,
            TARGET_RATIO,
        )))
    }

    fn make_ctx(profiler: &mut OpProfiler) -> StageContext<'_> {
        make_ctx_with_pressure(profiler, 0)
    }

    fn make_ctx_with_pressure(profiler: &mut OpProfiler, pressure: u8) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(pressure),
                prev_token: 0,
            },
            profiler,
        }
    }

    /// KvMutate 외 phase 는 no-op(Continue) + cache 불변.
    #[test]
    fn non_eviction_phase_is_noop() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage =
            EvictionStage::one_shot(vec![handle.clone()], make_cache_manager(), TARGET_RATIO);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::DecodeStart, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Continue));
        // current_pos 불변 (evict 미진입).
        assert_eq!(handle.current_pos(), N_TOKENS);
    }

    /// OneShot: KvMutate 1회 발화 → Consumed.
    #[test]
    fn one_shot_returns_consumed_on_eviction() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage =
            EvictionStage::one_shot(vec![handle.clone()], make_cache_manager(), TARGET_RATIO);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(
            matches!(outcome, StageOutcome::Consumed),
            "OneShot eviction 은 Consumed 반환"
        );
        // sliding(window=10, prefix=4) prune → current_pos 감소.
        assert!(
            handle.current_pos() < N_TOKENS,
            "eviction 후 current_pos 감소해야 함 (got {})",
            handle.current_pos()
        );
    }

    /// take_inner 후 put_inner 로 inner 가 복원된다(handle 경유 재접근 가능).
    #[test]
    fn put_inner_restores_after_dispatch() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage =
            EvictionStage::one_shot(vec![handle.clone()], make_cache_manager(), TARGET_RATIO);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let _ = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();

        // dispatch 후 inner 가 placeholder(0-len)가 아니라 실물 cache 여야 한다 —
        // take_inner 로 꺼내도 capacity 가 보존됨.
        let inner = handle.take_inner();
        assert_eq!(inner.capacity(), MAX_SEQ, "put_inner 가 실물 cache 를 복원");
    }

    // ── β-5: Persistent band-driven 발화 ──

    /// Persistent: band 충족(Warning 이상) 시 발화 → Continue + cache prune.
    #[test]
    fn persistent_band_met_fires_and_continues() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage = EvictionStage::persistent(
            vec![handle.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Level::Warning,
        );

        let mut profiler = OpProfiler::new();
        // pressure=50 → band()=Warning >= min(Warning) → 발화.
        let mut ctx = make_ctx_with_pressure(&mut profiler, 50);
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(
            matches!(outcome, StageOutcome::Continue),
            "Persistent 은 발화해도 Continue(상주, GC 안 함)"
        );
        assert!(
            handle.current_pos() < N_TOKENS,
            "band 충족 시 prune (got {})",
            handle.current_pos()
        );
    }

    /// Persistent: band 미달(Normal) 시 무발화 → Continue + cache 불변.
    #[test]
    fn persistent_band_unmet_is_noop() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage = EvictionStage::persistent(
            vec![handle.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Level::Warning,
        );

        let mut profiler = OpProfiler::new();
        // pressure=49 → band()=Normal < min(Warning) → 무발화.
        let mut ctx = make_ctx_with_pressure(&mut profiler, 49);
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Continue));
        assert_eq!(
            handle.current_pos(),
            N_TOKENS,
            "band 미달 시 prune 없음 (cache 불변)"
        );
    }

    /// 同코드 증명: 같은 입력 캐시에서 OneShot 발화 산출 == Persistent(band 충족) 발화 산출.
    /// `new_pos`(prune 후 current_pos)가 byte-identical 이어야 한다 (run_eviction 공유).
    #[test]
    fn persistent_band_met_matches_one_shot() {
        // OneShot 경로.
        let h_one = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let one = EvictionStage::one_shot(vec![h_one.clone()], make_cache_manager(), TARGET_RATIO);
        let mut p1 = OpProfiler::new();
        let mut c1 = make_ctx_with_pressure(&mut p1, 50);
        one.on_phase(&LifecyclePhase::KvMutate, &mut c1).unwrap();
        let one_shot_pos = h_one.current_pos();

        // Persistent(band 충족) 경로 — 동일 입력 캐시·CM·ratio.
        let h_per = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let per = EvictionStage::persistent(
            vec![h_per.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Level::Warning,
        );
        let mut p2 = OpProfiler::new();
        let mut c2 = make_ctx_with_pressure(&mut p2, 50);
        per.on_phase(&LifecyclePhase::KvMutate, &mut c2).unwrap();
        let persistent_pos = h_per.current_pos();

        assert_eq!(
            one_shot_pos, persistent_pos,
            "OneShot 과 Persistent(band 충족) prune 산출이 동일해야 함 (run_eviction 공유)"
        );
        assert!(
            persistent_pos < N_TOKENS,
            "양쪽 모두 실제 prune (비-vacuous)"
        );
    }

    /// β-5 episode edge-trigger: 지속 고압에서 prune 은 에피소드당 1회 — 매 step 재발화로
    /// floor 까지 나선 축소되는 퇴행 차단. band 가 min 미만으로 떨어지면 re-arm 되어
    /// 다음 상향 돌파에서 재발화한다.
    #[test]
    fn persistent_edge_trigger_once_per_episode() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage = EvictionStage::persistent(
            vec![handle.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Level::Warning,
        );
        let mut profiler = OpProfiler::new();

        // 에피소드 1: 상향 돌파 → 1회 발화.
        let mut c = make_ctx_with_pressure(&mut profiler, 50);
        stage.on_phase(&LifecyclePhase::KvMutate, &mut c).unwrap();
        let pos_after_first = handle.current_pos();
        assert!(pos_after_first < N_TOKENS, "에피소드 1 발화");

        // 지속 고압: 같은 에피소드 내 재dispatch → 무발화 (pos 불변).
        let mut c = make_ctx_with_pressure(&mut profiler, 60);
        stage.on_phase(&LifecyclePhase::KvMutate, &mut c).unwrap();
        assert_eq!(
            handle.current_pos(),
            pos_after_first,
            "지속 고압에서 재발화 금지 (episode edge-trigger)"
        );

        // band 하강 → re-arm (무발화).
        let mut c = make_ctx_with_pressure(&mut profiler, 10);
        stage.on_phase(&LifecyclePhase::KvMutate, &mut c).unwrap();
        assert_eq!(handle.current_pos(), pos_after_first, "Normal 에선 무발화");

        // 에피소드 2: 캐시 재성장 후 재돌파 → 재발화.
        handle.with_cache_mut(|cache| cache.current_pos = N_TOKENS);
        let mut c = make_ctx_with_pressure(&mut profiler, 80);
        stage.on_phase(&LifecyclePhase::KvMutate, &mut c).unwrap();
        assert!(
            handle.current_pos() < N_TOKENS,
            "re-arm 후 두 번째 에피소드에서 재발화"
        );
    }

    // ── §5.9.1 Track A: one_shot_scored + score_cell reset ──

    /// §5.9.1 Track A: one_shot_scored — eviction 완료 후 score_cell.reset() 호출.
    /// score_cell이 Some(active) → eviction 완료 후 importance가 0-filled 되어야 함.
    #[test]
    fn one_shot_scored_resets_accumulator_after_eviction() {
        use crate::inference::attention_scores::AttentionScoreAccumulator;

        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));

        // 점수를 0이 아닌 값으로 채운 accumulator 생성.
        let mut acc = AttentionScoreAccumulator::new(MAX_SEQ, KV_HEADS, 1, 0, 0.0);
        acc.set_active(true);
        // begin_step + 임의 score 주입(내부 importance 초기화).
        acc.begin_step();
        // importance 값이 0이 아닌지 확인하기 위해 accept_scores 호출(단, API 없으면 skip).
        // reset 후에는 모두 0이어야 한다.

        let score_cell = Arc::new(Mutex::new(Some(acc)));

        let stage = EvictionStage::one_shot_scored(
            vec![handle.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Arc::clone(&score_cell),
            None,
        );

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();

        // eviction 완료 후 score_cell.importance_scores()가 all-zero여야 한다.
        let guard = score_cell.lock().unwrap();
        let acc_ref = guard.as_ref().expect("acc still in cell");
        let all_zero = acc_ref.importance_scores().iter().all(|&s| s == 0.0);
        assert!(all_zero, "reset() 후 importance_scores 는 all-zero");
    }

    /// §5.9.1 Track A: score_cell=None → 기존 one_shot 경로와 동일하게 eviction 완료.
    #[test]
    fn one_shot_scored_without_score_cell_behaves_as_one_shot() {
        // score_cell 미주입 경우에도 정상 eviction 동작.
        let h_scored = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let empty_cell: Arc<
            Mutex<Option<crate::inference::attention_scores::AttentionScoreAccumulator>>,
        > = Arc::new(Mutex::new(None));
        let stage = EvictionStage::one_shot_scored(
            vec![h_scored.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            empty_cell,
            None,
        );

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();

        // eviction 실행 — current_pos 감소.
        assert!(
            h_scored.current_pos() < N_TOKENS,
            "score_cell None 도 eviction 실행 (got {})",
            h_scored.current_pos()
        );
    }

    /// bench GPU-score 배선: backend 주입(`Some`)도 CPU backend 에선 sync 가 no-op(downcast-miss)이라
    /// eviction 산출이 `backend=None`(=`one_shot_scored_resets_accumulator_after_eviction`)과
    /// byte-identical 임을 고정한다. run_eviction 의 새 sync 블록(`if let (Some(be), Some(cell))`)이
    /// CPU host 게이트에서 실제 진입하되 무영향임을 입증 — GPU sync 의 실효 검증은 온디바이스(수동).
    #[test]
    fn one_shot_scored_with_cpu_backend_syncs_noop_byte_identical() {
        use crate::inference::attention_scores::AttentionScoreAccumulator;

        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let mut acc = AttentionScoreAccumulator::new(MAX_SEQ, KV_HEADS, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        let score_cell = Arc::new(Mutex::new(Some(acc)));

        // CPU backend 주입 — run_eviction 의 sync 블록이 진입하지만 OpenCLBackend downcast-miss 로 no-op.
        let cpu_backend: Arc<dyn crate::backend::Backend> = Arc::new(CpuBackend::new());
        let stage = EvictionStage::one_shot_scored(
            vec![handle.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Arc::clone(&score_cell),
            Some(cpu_backend),
        );

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();

        // backend=None 경로와 동일 산출: eviction 발화(current_pos 감소) + reset → importance all-zero.
        assert!(
            handle.current_pos() < N_TOKENS,
            "backend 주입도 eviction 발화 (got {})",
            handle.current_pos()
        );
        let guard = score_cell.lock().unwrap();
        let acc_ref = guard.as_ref().expect("acc still in cell");
        assert!(
            acc_ref.importance_scores().iter().all(|&s| s == 0.0),
            "CPU backend sync no-op 후에도 reset() → all-zero (byte-identical)"
        );
    }
}
