//! L2 hardware 축 resolver — `Hardware`.
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §3.5.
//!
//! `Hardware` 는 **hardware 축**(연산 위치)의 좌표를 해석하는 read-only resolver 다.
//! `Backend`(§3.1)가 단일 연산기라면, `Hardware` 는 전환·분산을 위해 여러 backend + memory 를
//! 묶고 `resolve(target) → (backend, memory)` 로 좌표를 푼다. 활성 backend 를 mutable 소유하지
//! 않는다("현재 device" = decode-loop local). backend ⊥ memory 직교(UMA/discrete 캡슐화).
//!
//! 본 파일은 **Phase α-W 신설 추가 타입**이다 — 아직 `session/init.rs` 의 4 Arc
//! (`cpu_backend_arc`/`gpu_backend_arc`/`cpu_memory_arc`/`gpu_memory_arc`)가 이 타입으로
//! 전환되지 않았다. 흡수 배선은 Phase α-W-2 다. 따라서 이 파일의 추가는 기존 동작을 바꾸지 않는다.

use std::sync::Arc;

use crate::backend::Backend;
use crate::memory::Memory;

/// hardware 축 좌표 = 연산 위치(추상 역할). 구체 backend 는 registry resolve (G5-1, §3.5).
///
/// OpenCL↔CUDA 는 같은 `Gpu` 위치의 플랫폼별 구현체이고 feature 배타(`lib.rs:1`
/// `compile_error!`)라 한 바이너리에 공존 불가 → `Gpu` 하나가 컴파일된 GPU backend 로 resolve.
/// `--opencl-rpcmem` 은 device 가 아니라 memory interop(DMA-BUF alias)이라 variant 아님.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTarget {
    /// NEON/AVX2 — 항상 존재 → `resolve` 가 항상 `Some` (universal floor).
    Cpu,
    /// OpenCL or CUDA (feature 배타 → registry 가 컴파일된 것 택1). CPU-only 빌드면 resolve `None`.
    Gpu,
    /// HeteroLLM GPU∥NPU 재진입분. backend 미보유 시 resolve `None` (qnn_oppkg 제거 2026-05-26)
    /// 지만 partition `SliceSpec` 가 spec 레벨 전제라 variant 보유.
    Npu,
}

/// compute backend 레지스트리 — `{ cpu, gpu?, npu? }`.
struct BackendRegistry {
    cpu: Arc<dyn Backend>,
    gpu: Option<Arc<dyn Backend>>,
    npu: Option<Arc<dyn Backend>>,
}

/// data memory 레지스트리 — `{ host, device? }` (UMA 면 device==host → `None` = host 재사용).
struct MemoryRegistry {
    host: Arc<dyn Memory>,
    device: Option<Arc<dyn Memory>>,
}

/// `Hardware::resolve` 결과 — (compute backend, data memory) 페어 참조 (§3.5).
pub type ResolvedDevice<'a> = (&'a Arc<dyn Backend>, &'a Arc<dyn Memory>);

/// hardware 축 resolver (read-only). backend ⊥ memory 직교 (§3.5).
pub struct Hardware {
    backends: BackendRegistry,
    memories: MemoryRegistry,
}

impl Hardware {
    /// 흩어진 4(+2 primary) Arc 를 흡수해 생성한다 (배선은 α-W-2). `device == None` 이면 UMA 로
    /// 보고 host memory 를 재사용한다.
    pub fn new(
        cpu: Arc<dyn Backend>,
        gpu: Option<Arc<dyn Backend>>,
        npu: Option<Arc<dyn Backend>>,
        host: Arc<dyn Memory>,
        device: Option<Arc<dyn Memory>>,
    ) -> Self {
        Self {
            backends: BackendRegistry { cpu, gpu, npu },
            memories: MemoryRegistry { host, device },
        }
    }

    /// hardware 축 좌표를 (backend, memory) 로 푼다. UMA/discrete 분기를 이 한 곳에 가둔다.
    ///
    /// 부재 target 은 `None` (C1 확정, §3.5): `Cpu` 는 항상 `Some`, `Gpu`/`Npu` 는 미컴파일
    /// (feature)·미보유(backend 부재) 시 `None`. **부재 시 무엇을 할지는 호출자 정책**
    /// (§1.2 Mechanism over policy — switch 는 `resolve(Gpu).or_else(|| resolve(Cpu))`,
    /// partition setup 은 loud-fail).
    pub fn resolve(&self, target: DeviceTarget) -> Option<ResolvedDevice<'_>> {
        let device_mem = self.memories.device.as_ref().unwrap_or(&self.memories.host);
        match target {
            DeviceTarget::Cpu => Some((&self.backends.cpu, &self.memories.host)),
            DeviceTarget::Gpu => Some((self.backends.gpu.as_ref()?, device_mem)),
            DeviceTarget::Npu => Some((self.backends.npu.as_ref()?, device_mem)),
        }
    }
}
