//! weight-mutate Stage 거주지 (§2.1).
//!
//! 입주자:
//! - `PartitionStage` (AB-4, `partition.rs`) — `SetPartitionRatio` runtime directive 의 OneShot
//!   re-slice. concrete-handle `Vec<Arc<LayerSlot>>` + `Arc<Hardware>`(§5.5).
//!
//! `WeightSwapStage` / `WeightRecallStage` 는 QCF 기반 레이어 선택(importance × ε)에 의존했고,
//! QCF metric family 와 함께 제거되었다. Partition 은 그 결정과 무관한 slot dispatch-mode 변경이라
//! 그대로 남는다.

pub mod partition;
