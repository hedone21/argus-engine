//! `KiviQuantStage` — `KvQuantDynamic` runtime directive 의 OneShot `PipelineStage` (AB-2).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.7.
//!
//! `KvMutate` phase 에서 발화하여, register 시점 보유한 `Vec<Arc<KIVIFormat>>` 핸들의 inner
//! `KiviCache` 양자화 bit-width 를 런타임 전환(F16↔Q2/Q4/Q8)한다(`transition_bits` fan-out).
//! EvictionStage 와 형제(둘 다 KV 축 `stages/kv/`, OneShot, register 시점 concrete handle 보유,
//! `KvMutate` self-filter, 발화 후 `Consumed`)이나 mutate 대상이 다르다 — eviction 은 KV 토큰 prune,
//! quant 는 KV cache 표현(bit-width) 전환.
//!
//! **sticky 재적용 방지는 dispatcher 책임**(§5.7.3): CommandDispatcher 의 `last_quant_bits`
//! 비교 게이트가 같은 bits 의 재submit 을 막는다. Stage 자체는 발화 후 무조건 `Consumed`.
//!
//! **pos 환류 없음**(§5.7.2): quant 는 bit-width 만 바꾸고 토큰 수·pos 는 불변이라 `StageOutcome`
//! 는 무변경이고 driver 후처리가 0 이다.
//!
//! **graceful continue**(§5.7.5): transition 실패는 *구성/타이밍 조건*이지 프로그래밍 오류가
//! 아니므로(WeightSwapStage reject 와 동일 카테고리) `?` 전파/panic 하지 않고 로그 후 다음 layer 로
//! 계속한다(v1 `generate.rs`(d5ed71d2^) L4392-4407 등가).

use std::sync::Arc;

use crate::kv::kivi_format::KIVIFormat;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};

/// `KvMutate` phase 에서 KIVI cache bit-width 를 전환하는 OneShot Stage.
///
/// `KIVIFormat::with_cache_mut(&self)` 가 `Mutex<KiviCache>` interior-mutate 라 `&self` Stage 가
/// transition 가능(EvictionStage 의 `Mutex<CacheManager>` 와 달리 추가 Mutex 불요 — KIVIFormat 이
/// 이미 Mutex 보유).
pub struct KiviQuantStage {
    /// register 시점 보유 (INV-STAGE-LAYER-HANDLE). enumerate 순서 == layer idx
    /// (`kivi_forward.kivi_caches()` 동형). EvictionStage 의 `Vec<Arc<StandardFormat>>` 동형.
    handles: Vec<Arc<KIVIFormat>>,
    /// directive 가 지정한 목표 bit-width (2/4/8/16). `transition_bits` 가 same-bits 면 자체 no-op.
    target_bits: u8,
}

impl KiviQuantStage {
    /// `KvQuantDynamic` directive 1건 → OneShot bit-width transition (§5.7.2).
    ///
    /// 1회 발화 후 `Consumed` 를 반환해 registry 가 GC 한다. sticky 재적용 방지(같은 bits 무시,
    /// 값 변경 재적용)는 dispatcher 의 `last_quant_bits` 게이트가 담당한다(§5.7.3).
    pub fn one_shot(handles: Vec<Arc<KIVIFormat>>, target_bits: u8) -> Self {
        Self {
            handles,
            target_bits,
        }
    }

    /// transition 본문 (§5.7.5) — `KvMutate` 발화 시 1회 실행.
    ///
    /// v1 등가 anchor = `generate.rs`(d5ed71d2^) L4392-4407. per-cache `transition_bits` 적용,
    /// Err 시 `?` 전파/panic 하지 않고 로그 후 다음 layer 로 graceful continue. 전 cache 순회 후
    /// marker 라인 1회 출력(verify YAML 글자단위 계약 — `direct_cmd_kvquant_to_q4.yaml:26`).
    fn run_kivi_quant(&self) {
        let bits = self.target_bits;
        for h in &self.handles {
            if let Err(e) = h.with_cache_mut(|c| c.transition_bits(bits)) {
                eprintln!("[KIVI-Resilience] transition_bits({bits}) error: {e}");
            }
        }
        // marker 라인 (글자단위 — v1 L4406 verbatim). `{}`(Display) u8 → "4bit" 등.
        eprintln!("[KIVI-Resilience] Transitioned KV cache to {bits}bit");
    }
}

impl PipelineStage for KiviQuantStage {
    fn name(&self) -> &str {
        "kv.kivi_quant"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::OneShot
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.7.2): quant 외 phase 는 무시 (EvictionStage 의 KvMutate self-filter
        // (eviction.rs:142) verbatim 동형).
        if *phase != LifecyclePhase::KvMutate {
            return Ok(StageOutcome::Continue);
        }
        self.run_kivi_quant();
        // OneShot GC — transition 후 무조건 Consumed (sticky 게이트는 dispatcher).
        Ok(StageOutcome::Consumed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::kv::kivi_cache::KiviCache;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};

    // KiviCache 제약: residual_size·head_dim 모두 QKKV(=32) 의 배수여야 한다.
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 32;
    const MAX_SEQ: usize = 128;
    const RES: usize = 32;

    /// CPU KiviCache(initial bits) 를 KIVIFormat handle 로 wrap.
    fn make_handle(idx: usize, bits: u8) -> Arc<KIVIFormat> {
        let cache = KiviCache::new_with_bits(KV_HEADS, HEAD_DIM, MAX_SEQ, RES, bits);
        Arc::new(KIVIFormat::new(idx, cache))
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

    /// KvMutate 외 phase 는 no-op(Continue) + bits 불변.
    #[test]
    fn non_kvmutate_phase_is_noop() {
        let handle = make_handle(0, 16);
        let stage = KiviQuantStage::one_shot(vec![handle.clone()], 4);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::DecodeStart, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Continue));
        // bits 불변 (transition 미진입).
        assert_eq!(handle.current_bits(), 16);
    }

    /// OneShot: KvMutate 1회 발화 → Consumed + bits 실전환.
    #[test]
    fn one_shot_transitions_bits_and_consumes() {
        let h0 = make_handle(0, 16);
        let h1 = make_handle(1, 16);
        let stage = KiviQuantStage::one_shot(vec![h0.clone(), h1.clone()], 4);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(
            matches!(outcome, StageOutcome::Consumed),
            "OneShot quant 은 Consumed 반환"
        );
        // 전 cache 가 16 → 4 로 실전환.
        assert_eq!(h0.current_bits(), 4, "layer 0 bits transition");
        assert_eq!(h1.current_bits(), 4, "layer 1 bits transition");
    }

    /// same-bits transition 은 self no-op (transition_bits 가 같은 bits 면 early return) + Consumed.
    #[test]
    fn same_bits_is_self_noop_and_consumes() {
        let handle = make_handle(0, 4);
        let stage = KiviQuantStage::one_shot(vec![handle.clone()], 4);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        assert_eq!(handle.current_bits(), 4, "same-bits → 불변 (no-op)");
    }

    /// 빈 handle: 발화해도 panic 없이 Consumed (marker 라인만 출력, transition 0회).
    #[test]
    fn empty_handles_consumes_without_panic() {
        let stage = KiviQuantStage::one_shot(Vec::new(), 4);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
    }
}
