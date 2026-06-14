//! `RuntimeResources` + `setup_runtime_resources` — inference 측의 ctor 호출
//! 위계 어긋남 해소 helper (§13.8-O 우선순위 #2, design doc `arch/weights_pressure_split.md §7.4`).
//!
//! 종래에는 `TransformerModel` ctor 가 직접 `QuantNoiseTable::empty()` /
//! `PrimaryReleaseWorker::spawn(...)` 를 호출하여 inference 가 pressure-owned
//! 자원의 생성자를 알고 있는 형태였다. 본 helper 로 ctor 호출을 pressure
//! 도메인에 정착시키고, inference 는 `RuntimeResources` 를 받아 field 에
//! install 하는 형태로 정합.
//!
//! 2026-05-27 후속 sprint (`RuntimeResourcesAccess` trait inversion): field
//! 타입을 trait object (`Arc<dyn QuantNoiseAccess>` / `Arc<dyn
//! ReleaseWorkerAccess>`) 로 격상. struct 정의에서 pressure 타입 노출 0건 →
//! cross-L3 vocabulary marker 자연 해소.

use std::sync::Arc;

use crate::backend::Backend;
use crate::runtime_resources_access::{QuantNoiseAccess, ReleaseWorkerAccess};

use super::noise_table::QuantNoiseTable;
use super::release_worker::PrimaryReleaseWorker;

/// Pressure-owned runtime resources required by `TransformerModel` at init.
///
/// Constructed by [`setup_runtime_resources`] and consumed by the inference
/// loader / test helpers via struct field install. The fields are exposed as
/// trait objects so the inference struct definition does not reference
/// pressure-side concrete types directly.
pub struct RuntimeResources {
    pub quant_noise: Arc<dyn QuantNoiseAccess>,
    pub release_worker: Arc<dyn ReleaseWorkerAccess>,
}

/// Initialize pressure-owned runtime resources for a fresh `TransformerModel`.
///
/// - `quant_noise` is initialized empty; populate later via
///   `super::noise_table::compute_quant_noise` once `secondary_mmap` is known
///   (cf. `TransformerModel::load_gguf_with_secondary`).
/// - `release_worker` spawns its background thread immediately. The worker
///   retains a clone of `backend` for diagnostic calls; the returned `Arc`
///   must outlive any `SwapExecutor` borrow.
pub fn setup_runtime_resources(backend: Arc<dyn Backend>) -> RuntimeResources {
    RuntimeResources {
        quant_noise: Arc::new(QuantNoiseTable::empty()),
        release_worker: Arc::new(PrimaryReleaseWorker::spawn(backend)),
    }
}
