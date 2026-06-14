//! Precision swap orchestration domain (Sprint C, 2026-05-26).
//!
//! Contains all stateful weight swap orchestrators, controllers, and policy
//! modules moved from `models/weights/` per the INV-LAYER-003 §13.8-O
//! domain-boundary alignment.
//!
//! Inference-side weight resource definitions (LayerSlot, SecondaryMmap, etc.)
//! remain in `models/weights/` as loader artifacts.
//!
//! Spec: ENG-ALG-211/215/228/235~238, INV-120/121/123/127/147~150.

pub mod async_swap;
pub mod decider;
pub mod dynamic_k;
pub mod incremental_plan;
pub mod intra_forward_swap;
pub mod noise_table;
pub mod phase_aware_swap;
pub mod probing_k;
pub mod release_worker;
pub mod setup;
pub mod stage_ctx;
pub mod stage_registry;
pub mod swap_executor;
pub mod weight_swap_handler;

pub use async_swap::{AsyncSwapDispatcher, SwapCommitJob, SwapJob};
pub use decider::{SwapAlgorithm, SwapDecision, WeightSwapDecider, compute_qcf_weight_swap};
pub use dynamic_k::DynamicKController;
pub use incremental_plan::IncrementalSwapPlan;
pub use intra_forward_swap::{IntraForwardSwapHook, IntraForwardSwapPlan};
pub use noise_table::{QuantNoiseTable, compute_quant_noise};
pub use phase_aware_swap::{PhaseAwareSwapDispatcher, WeightChunk};
pub use probing_k::{GrowthMode, ProbingKController};
pub use release_worker::PrimaryReleaseWorker;
pub use setup::{RuntimeResources, setup_runtime_resources};
pub use stage_ctx::WeightStageModelCtx;
pub use swap_executor::{
    StageBreakdown, SwapError, SwapExecutor, SwapReport, SwappedLayer, dtype_tag_to_dtype,
};
pub use weight_swap_handler::WeightSwapHandler;
