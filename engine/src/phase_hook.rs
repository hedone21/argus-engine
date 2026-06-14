// ── PhaseHook trait + DdrPhase enum (L2 격상) ────────────────────────────────
//!
//! op boundary phase 경계에서 불리는 hook 계약 (LISWAP-5 / §13.8-N).
//!
//! 위치 근거: L2 공유 어휘 — observability(`op_trace`, hook fire 측)와
//! weight(`phase_aware_swap`, hook impl 측) 양 도메인이 공유하는 인터페이스
//! 정의. §13.8-G `OpKind` / `LayerBoundaryHook` 격상과 동일 동기 (cross-cutting
//! trait/enum 을 producer·consumer 어느 한쪽 도메인에 묶지 않고 L2 top-level 로).
//!
//! 실제 프로파일링 로직(op boundary 검사, hook fire 호출처, `OpKind::ddr_phase`
//! 분류 매핑)은 `observability/profile/op_trace.rs` 에 잔존한다. 본 파일은
//! "phase 경계에서 불리는 hook" 계약과 그 분류 enum 정의만 소유한다.
//!
//! BC re-export: `op_trace.rs` 가 `pub use crate::phase_hook::{PhaseHook, DdrPhase}`
//! 로 re-export 하여 기존 `crate::observability::profile::op_trace::{PhaseHook, DdrPhase}`
//! 경로는 무파손 유지된다.

use crate::op_kind::OpKind;

/// LISWAP-5 phase boundary hook. `op_trace::start_op` / `record`가 OpKind와
/// 함께 콜백을 호출. 등록은 `op_trace::set_phase_hook` (process-wide singleton).
/// hook 미등록 시 zero-overhead (atomic load 1회 + 분기).
///
/// PhaseHook trait 정의는 L2(`phase_hook.rs`) — observability(op_trace)가 hook을
/// fire하고, weight(phase_aware_swap)가 이를 impl 한다. 양 도메인이 공유하는
/// cross-cutting 어휘이므로 producer·consumer 어느 쪽에도 묶지 않고 L2에 둔다.
pub trait PhaseHook: Send + Sync {
    /// op 시작 직전 호출. ddr-heavy 진입 시 in-flight chunk 완료 대기.
    fn on_op_start(&self, kind: OpKind);
    /// op 끝난 직후 호출. cache-fit 끝났으면 다음 chunk dispatch.
    fn on_op_end(&self, kind: OpKind);
}

/// DDR-bandwidth phase classification for phase-aware async swap (LISWAP-5 / B).
/// Phase R 측정 (qnn_phase_r_summary.md §5)에 따른 분류.
///
/// `Heavy`: weight matmul 등 DDR을 많이 쓰는 op. swap 메모리 트래픽과 contention.
/// `CacheFit`: weight 작거나 L2 fit. swap 트래픽과 거의 무간섭 (Phase R Scenario B에서 1.04× of max).
/// `Medium`: attention 등. partial overlap. 보수적으로 swap 회피.
///
/// `OpKind::ddr_phase` 분류 매핑은 producer 도메인(op_trace.rs)에 잔존한다 —
/// 본 enum은 그 매핑이 반환하는 순수 데이터 분류만 정의한다.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DdrPhase {
    Heavy,
    CacheFit,
    Medium,
}
