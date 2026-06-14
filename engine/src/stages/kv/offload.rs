//! `OffloadStage` — AB-3 KvOffload/recall OneShot PipelineStage (KV 축).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.10.
//!
//! 1차 거울 = EvictionStage(β-3) — 동일 핸들(`Vec<Arc<StandardFormat>>` +
//! `Arc<Mutex<CacheManager>>`)과 UER 본문(`take_inner → cm.op → put_inner`)을 공유하며,
//! `cm.op` 만 `force_evict` → `cm.offload(ratio)` / `cm.recall()` 로 달라진다.
//!
//! **lifecycle = OneShot**: directive 1회 = submit 1회 = 발화 1회 = Consumed.
//! sticky 게이트 없음(transient — WeightSwapStage §5.6.4 동형).
//!
//! **pos 환류**: offload(`prune_prefix`)는 `current_pos` 를 감소시키므로 driver 의
//! `reconcile_kv_pos_after_eviction`(EvictionStage 와 동일 메커니즘)이 자동 흡수.
//! recall 의 pos 증가는 driver 의 일반화된 환류(§5.10.4 A안)가 흡수.

use std::sync::{Arc, Mutex};

use crate::kv::cache_manager::CacheManager;
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};

/// offload(disk → KV) / recall(KV → disk) 방향.
pub enum OffloadDirection {
    /// LRU prefix 를 디스크로 offload. ratio = offload 목표 비율.
    Offload { ratio: f32 },
    /// offload 된 prefix 를 KV 로 recall.
    Recall,
}

/// `KvMutate` phase 에서 CacheManager UER 로 offload/recall 하는 Stage.
///
/// EvictionStage 거울 — 동일 핸들 구조와 UER 본문, 다른 CacheManager op.
/// v1 `ModelForward::try_offload`(model_forward.rs:694-715)/`try_recall`(:717-735) 의
/// inner op 와 **byte-identical**.
pub struct OffloadStage {
    handles: Vec<Arc<StandardFormat>>,
    cache_manager: Arc<Mutex<CacheManager>>,
    direction: OffloadDirection,
}

impl OffloadStage {
    /// KvOffload directive → offload OneShot. transient — 매 directive 마다 새 Stage.
    pub fn offload(
        handles: Vec<Arc<StandardFormat>>,
        cache_manager: Arc<Mutex<CacheManager>>,
        ratio: f32,
    ) -> Self {
        Self {
            handles,
            cache_manager,
            direction: OffloadDirection::Offload { ratio },
        }
    }

    /// RestoreDefaults recall → recall OneShot.
    pub fn recall(
        handles: Vec<Arc<StandardFormat>>,
        cache_manager: Arc<Mutex<CacheManager>>,
    ) -> Self {
        Self {
            handles,
            cache_manager,
            direction: OffloadDirection::Recall,
        }
    }

    /// UER 본문 — v1 try_offload/try_recall inner op byte-identical.
    ///
    /// take_inner → cm.offload(ratio) or cm.recall() → put_inner → n 반환.
    /// swap 미활성(swap_handler=None) 이면 cm 이 0 반환(graceful, fail-fast 아님).
    fn run(&self) -> anyhow::Result<usize> {
        let mut temp: Vec<KVCache> = self.handles.iter().map(|f| f.take_inner()).collect();
        let result = {
            let mut cm = self
                .cache_manager
                .lock()
                .expect("OffloadStage CacheManager Mutex poisoned");
            match &self.direction {
                OffloadDirection::Offload { ratio } => cm.offload(&mut temp, *ratio),
                OffloadDirection::Recall => cm.recall(&mut temp),
            }
        };
        for (f, c) in self.handles.iter().zip(temp) {
            f.put_inner(c);
        }
        result
    }
}

impl PipelineStage for OffloadStage {
    fn name(&self) -> &str {
        "kv.offload"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::OneShot
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter: KvMutate 외 phase 는 무시.
        if *phase != LifecyclePhase::KvMutate {
            return Ok(StageOutcome::Continue);
        }

        let n = self.run()?;

        // 로그 verbatim (§5.10.5 — verify YAML 글자단위 계약).
        match &self.direction {
            OffloadDirection::Offload { ratio } => {
                eprintln!(
                    "[Resilience] KvOffload: ratio={:.2}, {} tokens swapped",
                    ratio, n
                );
            }
            OffloadDirection::Recall => {
                if n > 0 {
                    eprintln!("[Resilience] Recalled {} tokens from swap", n);
                }
            }
        }

        Ok(StageOutcome::Consumed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::format::KVCacheFormat;
    use crate::kv::eviction::sliding_window::SlidingWindowPolicy;
    use crate::memory::host::shared::SharedBuffer;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::resilience::sys_monitor::NoOpMonitor;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 32;
    const MAX_SEQ: usize = 128;
    const N_TOKENS: usize = 120;

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

    /// swap 미활성 CacheManager (swap_handler=None).
    fn make_cm_no_swap() -> Arc<Mutex<CacheManager>> {
        let policy = Box::new(SlidingWindowPolicy::new(10, 4));
        Arc::new(Mutex::new(CacheManager::new(
            policy,
            Box::new(NoOpMonitor),
            usize::MAX,
            0.3,
        )))
    }

    fn make_ctx(profiler: &mut OpProfiler) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(0),
                prev_token: 0,
            },
            profiler,
        }
    }

    /// ① KvMutate 외 phase 는 no-op(Continue), cache 불변.
    #[test]
    fn non_kv_mutate_phase_is_noop() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage = OffloadStage::offload(vec![handle.clone()], make_cm_no_swap(), 0.5);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::DecodeStart, &mut ctx)
            .unwrap();
        assert!(
            matches!(outcome, StageOutcome::Continue),
            "non-KvMutate phase → Continue"
        );
        assert_eq!(
            handle.current_pos(),
            N_TOKENS,
            "cache 불변 (offload 미진입)"
        );
    }

    /// ② offload 발화 → Consumed (swap 미활성이라도 graceful, n=0).
    #[test]
    fn offload_fires_returns_consumed() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage = OffloadStage::offload(vec![handle.clone()], make_cm_no_swap(), 0.5);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(
            matches!(outcome, StageOutcome::Consumed),
            "offload KvMutate → Consumed"
        );
    }

    /// ③ recall 발화 → Consumed.
    #[test]
    fn recall_fires_returns_consumed() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage = OffloadStage::recall(vec![handle.clone()], make_cm_no_swap());

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(
            matches!(outcome, StageOutcome::Consumed),
            "recall KvMutate → Consumed"
        );
    }

    /// ④ swap 미활성 CM → offload n=0 graceful (panic 없이 정상 완료, Consumed 반환).
    #[test]
    fn swap_inactive_cm_offload_n0_graceful() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        // make_cm_no_swap: swap_handler=None → cm.offload 가 Ok(0) 반환.
        let stage = OffloadStage::offload(vec![handle.clone()], make_cm_no_swap(), 0.5);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        // panic 없이 완료해야 한다.
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        // cache current_pos 불변 (offload 0건).
        assert_eq!(handle.current_pos(), N_TOKENS);
    }
}
