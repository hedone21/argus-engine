//! `PartitionStage` — `SetPartitionRatio` runtime directive 의 OneShot `PipelineStage` (AB-4).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.5.
//!
//! `PreForward` phase 에서 발화하여, register 시점 보유한 `Vec<Arc<LayerSlot>>` 의 weight slot 을
//! 런타임 re-slice 한다(`apply_partition_dispatch` fan-out, §5.5.3). EvictionStage 와 형제(둘 다
//! OneShot, register 시점 concrete handle 보유, self-filter, 발화 후 `Consumed`)이나 mutate
//! 대상이 다르다 — eviction 은 KV cache, partition 은 weight slot dispatch mode.
//!
//! **sticky 재적용 방지는 dispatcher 책임**(§5.5.2): CommandDispatcher 의 `last_partition_ratio`
//! 비교 게이트가 같은 ratio 의 재submit 을 막는다. Stage 자체는 발화 후 무조건 `Consumed`.
//!
//! **pos 환류 없음**(§5.5.1): partition 은 weight 만 바꾸고 KV/pos 는 불변이라 `StageOutcome` 는
//! 무변경이고 driver 후처리가 0 이다. plan path 의 stale `cl_mem` 차단은 INV-120 gen-counter
//! (`apply_dispatch` 내부 bump → `PlanInvalidated` 런타임 자동 무효화)가 담당 — Stage 는 slot 만
//! mutate 하고 plan 은 만지지 않는다.

use std::sync::Arc;

use crate::hardware::Hardware;
use crate::layers::tensor_partition::apply_partition_dispatch;
use crate::models::weights::LayerSlot;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};

/// `PreForward` phase 에서 weight slot 을 re-slice 하는 OneShot Stage.
///
/// `LayerSlot::apply_dispatch(&self)` 가 ArcSwap RCU 라 `&self` Stage 가 mutate 가능
/// (Mutex 불요 — slot 자체가 Sync interior-mutable). `Hardware` 도 read-only resolver 라
/// `Arc<Hardware>` 공유가 안전하다.
pub struct PartitionStage {
    /// register 시점 보유 (INV-STAGE-LAYER-HANDLE). `model.layers.clone()` — enumerate 순서 ==
    /// layer idx (`apply_partition_dispatch` fan-out 순서 == CLI 정적 경로와 동일).
    slots: Vec<Arc<LayerSlot>>,
    /// companion backend resolve 용. partition fan-out 의 cpu backend (`resolve(Cpu)`)와 lazy
    /// host-map 의 gpu backend (`resolve(Gpu)`)를 모두 여기서 푼다.
    hw: Arc<Hardware>,
    /// directive 가 지정한 GPU share. `>= GPU_ONLY_THRESHOLD` 면 GPU-only fast path.
    gpu_ratio: f32,
}

impl PartitionStage {
    /// `SetPartitionRatio` directive 1건 → OneShot re-slice (§5.5.1).
    ///
    /// 1회 발화 후 `Consumed` 를 반환해 registry 가 GC 한다. sticky 재적용 방지(같은 ratio 무시,
    /// 값 변경 재적용)는 dispatcher 의 `last_partition_ratio` 게이트가 담당한다(§5.5.2).
    pub fn one_shot(slots: Vec<Arc<LayerSlot>>, hw: Arc<Hardware>, gpu_ratio: f32) -> Self {
        Self {
            slots,
            hw,
            gpu_ratio,
        }
    }

    /// fan-out 본문 (§5.5.3) — `PreForward` 발화 시 1회 실행.
    ///
    /// partition 활성(`!is_gpu_only_ratio`) 시 최초 1회 lazy host-map(weight 가 CPU companion
    /// matmul 에 host pointer 를 제공하도록, §5.5.4) 후 re-slice. 이미 host-mapped 인 weight 는
    /// `map_weight_for_host` 가 no-op(changed=false) 이라 반복 호출이 안전하다(OneShot 이라 실 1회).
    ///
    /// disable(`ratio <= 0.0`)은 legacy(`generate.rs:2730-2745`) 동형 — partition_ctx 클리어
    /// (`LayerDispatch::Full` fan-out) + `[Partition] Disabled (ratio={})`. `ratio >= 1.0` 은
    /// `apply_partition_dispatch` 의 GPU-only fast path 가 이미 Full 로 클리어한다.
    fn run_partition(&self) -> anyhow::Result<()> {
        use crate::layers::tensor_partition::is_gpu_only_ratio;

        // Disable: ratio<=0.0 → partition off (partition_ctx 클리어). legacy 와 달리
        // ratio>=1.0 은 apply_partition_dispatch 의 GPU-only fast path 가 처리하므로 여기선
        // ratio<=0.0 만 별도 disable 로그. host-map 불필요(CPU 측 미사용).
        if self.gpu_ratio <= 0.0 {
            use crate::format::weight_format::{LayerDispatch, WeightFormat};
            for slot in &self.slots {
                slot.apply_dispatch(LayerDispatch::Full, &self.hw)?;
            }
            eprintln!("[Partition] Disabled (ratio={})", self.gpu_ratio);
            return Ok(());
        }

        // partition path(비 GPU-only)는 weight 가 host-accessible 이어야 한다.
        // CLI 정적 경로는 init.rs 에서 미리 map 하나, 런타임 directive 경로는 여기서 lazy map.
        // GPU-only fast path 는 partition path 를 안 타므로 host-map 불필요.
        #[cfg(feature = "opencl")]
        if !is_gpu_only_ratio(self.gpu_ratio) {
            use crate::hardware::DeviceTarget;
            // OpenCL GPU backend 가 있을 때만 lazy map(CPU/CUDA-UMA/non-OpenCL 은 weight 가 이미
            // host-pointer-accessible → `map_layer_slots_for_host_access` 가 GPU backend 부재로
            // skip 하므로, resolve(Gpu) 가 None 이면 호출 자체를 생략).
            if let Some((gpu_backend, _)) = self.hw.resolve(DeviceTarget::Gpu) {
                let mapped = crate::models::transformer::map_layer_slots_for_host_access(
                    &self.slots,
                    gpu_backend,
                )?;
                if mapped > 0 {
                    eprintln!(
                        "[Partition] Lazy-mapped {} weight tensors for host access",
                        mapped
                    );
                }
            }
        }
        #[cfg(not(feature = "opencl"))]
        let _ = is_gpu_only_ratio(self.gpu_ratio);

        let n = apply_partition_dispatch(&self.slots, self.gpu_ratio, &self.hw)?;
        if n > 0 {
            eprintln!(
                "[Partition] Re-split {} weights with ratio {:.2}",
                n, self.gpu_ratio
            );
        }
        Ok(())
    }
}

impl PipelineStage for PartitionStage {
    fn name(&self) -> &str {
        "weight.partition"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::OneShot
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.5.1): PreForward 외 phase 는 무시.
        // PartitionStage 는 PreForward 잔류 — re-slice 가 forward plan 준비 성격이고
        // AB-4 동결 직후라 이동하지 않음(§5.6.5).
        if *phase != LifecyclePhase::PreForward {
            return Ok(StageOutcome::Continue);
        }
        self.run_partition()?;
        // OneShot GC — re-slice 후 무조건 Consumed (sticky 게이트는 dispatcher).
        Ok(StageOutcome::Consumed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::layers::transformer_layer::TransformerLayer;
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;
    use crate::memory::host::shared::SharedBuffer;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    fn cpu_backend() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    fn cpu_only_hardware(be: &Arc<dyn Backend>) -> Arc<Hardware> {
        let host: Arc<dyn Memory> = Arc::new(Galloc::new());
        Arc::new(Hardware::new(be.clone(), None, None, host, None))
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

    fn f32_weight(be: &Arc<dyn Backend>, out_dim: usize, in_dim: usize) -> Tensor {
        let buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(out_dim * in_dim * 4, DType::F32));
        Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, be.clone())
    }

    fn ffn_slot(be: &Arc<dyn Backend>, idx: usize) -> Arc<LayerSlot> {
        // FFN weights large enough for split_weight; attention weights small.
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

    fn slots(be: &Arc<dyn Backend>, n: usize) -> Vec<Arc<LayerSlot>> {
        (0..n).map(|i| ffn_slot(be, i)).collect()
    }

    /// PreForward 외 phase 는 no-op(Continue) + slot dispatch 불변.
    #[test]
    fn non_preforward_phase_is_noop() {
        let be = cpu_backend();
        let hw = cpu_only_hardware(&be);
        let stage = PartitionStage::one_shot(slots(&be, 2), Arc::clone(&hw), 0.5);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Continue));
        for slot in &stage.slots {
            assert!(
                crate::format::weight_format::WeightFormat::idx(slot.as_ref()) < 2,
                "slot intact"
            );
        }
    }

    /// OneShot: PreForward 1회 발화 → Consumed + partition_ctx 설치(비 GPU-only).
    #[test]
    fn partition_ratio_installs_ctx_and_consumes() {
        let be = cpu_backend();
        let hw = cpu_only_hardware(&be);
        let stage = PartitionStage::one_shot(slots(&be, 2), Arc::clone(&hw), 0.5);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::PreForward, &mut ctx)
            .unwrap();
        assert!(
            matches!(outcome, StageOutcome::Consumed),
            "OneShot partition 은 Consumed 반환"
        );
        for slot in &stage.slots {
            assert!(
                slot.load_weights().partition_ctx.is_some(),
                "ratio<threshold → partition_ctx 설치"
            );
        }
    }

    /// GPU-only ratio: Full fan-out → partition_ctx 미설치 + Consumed.
    #[test]
    fn gpu_only_ratio_leaves_ctx_cleared() {
        let be = cpu_backend();
        let hw = cpu_only_hardware(&be);
        let stage = PartitionStage::one_shot(slots(&be, 2), Arc::clone(&hw), 0.999);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::PreForward, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        for slot in &stage.slots {
            assert!(
                slot.load_weights().partition_ctx.is_none(),
                "GPU-only fast path → partition_ctx 미설치"
            );
        }
        let _ = DType::F32;
    }

    /// disable(ratio=0.0): 설치된 partition_ctx 를 클리어 + Consumed (legacy Disabled 동형).
    #[test]
    fn ratio_zero_disables_partition() {
        let be = cpu_backend();
        let hw = cpu_only_hardware(&be);
        let s = slots(&be, 2);

        // 먼저 partition 설치 (ratio=0.5).
        let install = PartitionStage::one_shot(s.clone(), Arc::clone(&hw), 0.5);
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        install
            .on_phase(&LifecyclePhase::PreForward, &mut ctx)
            .unwrap();
        for slot in &s {
            assert!(slot.load_weights().partition_ctx.is_some(), "설치 확인");
        }

        // ratio=0.0 → disable (partition_ctx 클리어).
        let disable = PartitionStage::one_shot(s.clone(), Arc::clone(&hw), 0.0);
        let outcome = disable
            .on_phase(&LifecyclePhase::PreForward, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        for slot in &s {
            assert!(
                slot.load_weights().partition_ctx.is_none(),
                "ratio=0.0 → partition_ctx 클리어 (Disabled)"
            );
        }
    }
}
