//! weight-mutate Stage 거주지 (§2.1).
//!
//! 입주자:
//! - `PartitionStage` (AB-4, `partition.rs`) — `SetPartitionRatio` runtime directive 의 OneShot
//!   re-slice. concrete-handle `Vec<Arc<LayerSlot>>` + `Arc<Hardware>`(§5.5).
//! - `WeightSwapStage` (AB-6, `weight_swap.rs`) — `SwapWeights` runtime directive 의 OneShot
//!   precision swap(F16→Q4_0). held-handle `Arc<TransformerModel>`(model 측 접근 seam) +
//!   `Arc<EngineSwapRuntime>` + `Option<Arc<dyn ImportanceLookup>>`(§5.6). commit 본문은
//!   `session::swap_runtime::handle_swap_weights` §1~7 byte-identical 이전(등가 anchor 유지).
//!   PartitionStage 와 형제이나 join 표면이 0(weight slot dispatch mode 변경 vs precision swap)이라
//!   별도 파일. Incremental 만 Stage multi-tick drain, 나머지 3-mode 는 hook 설치만(§5.6.3).

pub mod partition;
pub mod weight_recall;
pub mod weight_swap;
