//! GPU score accumulator capability (§3.3).
//!
//! Phase α-W-4 에서 `backend.rs` 의 `GpuScoreAccess` trait 정의를 이리로 이동했다.
//! `backend.rs` 는 본 정의를 re-export 하므로 기존 `crate::backend::GpuScoreAccess`
//! import path 가 그대로 동작한다(byte-identical).

/// §13.8-L S-L-2 — GPU score accumulator 추상화.
///
/// `forward_into` / `forward_gen` 같은 hot path 가 OpenCL backend 의
/// `GpuScoreAccumulator` inherent struct 에 접근하기 위한 trait 입니다.
/// 본 trait 는 *backend-agnostic* 인 read/write API 만 노출하고,
/// OpenCL-specific 한 `score_buf_mem()` (returns `&ocl::core::Mem`),
/// `end_step(...)`, `sync_to_cpu(queue)`, `reset(queue)` 등 raw OpenCL
/// 자원에 닿는 메서드는 OpenCL backend 안에서 inherent method 그대로
/// 사용합니다 (`OpenCLBackend::execute_plan` 등 OpenCL 내부 경로).
///
/// `&self -> &mut Self` 반환은 OpenCL backend 의 `UnsafeCell`-backed
/// accumulator 와 일치 (single-threaded 추론 가정).
pub trait GpuScoreAccess: Send + Sync {
    fn is_active(&self) -> bool;
    fn set_active(&mut self, active: bool);
    fn current_layer_idx(&self) -> usize;
    fn set_current_layer_idx(&mut self, layer_idx: usize);
    fn n_heads_q(&self) -> usize;
    fn n_layers(&self) -> usize;
    fn layer_offset_elems(&self, layer_idx: usize) -> usize;
    fn score_stride(&self) -> usize;
    fn steps_accumulated(&self) -> usize;
}
