//! `WeightFormat` base trait + dispatch spec (§4.2).
//!
//! KV(`KVCacheFormat`, §4.1)와 대칭이다. base-trait-handle 을 든 Stage 는 dispatch 모드만 알고
//! weight content 는 모른다. precision swap 등 Format mutation 은 concrete-handle Stage
//! (`WeightSwapStage` with `Arc<LayerSlot>`)가 concrete method 로 직접 수행한다.
//! `LayerDispatch::Partition` 의 분산 대상 backend 는 `Hardware`(§3.5)에서 resolve 하고, 슬라이스
//! 마다 다른 (format, hardware) 좌표를 가질 수 있다 — partition = format(표현) × hardware(위치)의 곱.
//!
//! **dispatch 타입은 `technique-api` 거주**: `LayerDispatch`/`PartitionShare` 는
//! plugin 결정 표면이라 api crate 에 있고 여기서 re-export 한다. 엔진 측에는 executor 변형 표면인
//! `WeightFormat` trait 과 `DeviceTarget` mirror From 변환만 둔다. per-slice 저장 format 은 plugin
//! 결정이 아니라 executor 가 weight dtype 에서 파생하므로 `PartitionShare` 표면에서 제외(api 정의 doc).

use anyhow::Result;

use crate::hardware::Hardware;

pub use technique_api::{LayerDispatch, PartitionShare};

/// weight layer 의 dispatch 모드(Full / Skip / Partition)를 적용하는 base trait.
pub trait WeightFormat: Send + Sync {
    fn idx(&self) -> usize;

    /// construction tier — companion 을 `Hardware` 로 resolve (§4.2). `Partition` 은 N-HW composite.
    fn apply_dispatch(&self, d: LayerDispatch, hw: &Hardware) -> Result<()>;
    // view() 없음 — read 는 concrete-handle(load_weights) 경유 (§4.2 연혁: 런타임 weight 구조체가
    //               `TransformerLayer` 단 하나 → 제네릭 read view 는 1-adapter 가설적 seam).
    // apply_storage(spec) 없음 — precision swap 등은 concrete-handle Stage(`Arc<LayerSlot>`) 직접 호출.
}

/// api 표면 `technique_api::DeviceTarget` ↔ 엔진 `hardware::DeviceTarget` 1:1 mirror.
/// 두 enum 변종이 어긋나면 round-trip 테스트(drift 게이트)가 깨진다.
impl From<technique_api::DeviceTarget> for crate::hardware::DeviceTarget {
    fn from(d: technique_api::DeviceTarget) -> Self {
        use technique_api::DeviceTarget as A;
        match d {
            A::Cpu => crate::hardware::DeviceTarget::Cpu,
            A::Gpu => crate::hardware::DeviceTarget::Gpu,
            A::Npu => crate::hardware::DeviceTarget::Npu,
        }
    }
}

impl From<crate::hardware::DeviceTarget> for technique_api::DeviceTarget {
    fn from(d: crate::hardware::DeviceTarget) -> Self {
        use crate::hardware::DeviceTarget as E;
        match d {
            E::Cpu => technique_api::DeviceTarget::Cpu,
            E::Gpu => technique_api::DeviceTarget::Gpu,
            E::Npu => technique_api::DeviceTarget::Npu,
        }
    }
}

#[cfg(test)]
mod drift_tests {
    /// api↔engine DeviceTarget 변종 drift 게이트: 모든 변종 round-trip.
    #[test]
    fn device_target_mirror_roundtrip() {
        use crate::hardware::DeviceTarget as E;
        for e in [E::Cpu, E::Gpu, E::Npu] {
            let a: technique_api::DeviceTarget = e.into();
            let back: E = a.into();
            assert_eq!(e, back);
        }
    }
}
