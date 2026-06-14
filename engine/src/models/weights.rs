//! Inference-side weight resource definitions (loader artifacts).
//!
//! Contains `LayerSlot`, `SecondaryMmap`, `RpcmemSecondaryStore`, and the
//! backing types that the loader produces and inference uses for forward pass.
//!
//! Precision swap orchestration (SwapExecutor, WeightSwapDecider, etc.) has
//! been moved to `pressure/weights/` (Sprint C, 2026-05-26).
//!
//! Spec: ENG-DAT-090/092/094/095, INV-123/124.

pub mod backing;
#[cfg(feature = "cuda-embedded")]
pub mod layer_object_pool; // §13.8-B 결정 보류로 잔존 (§6 risk D)
pub mod rpcmem_secondary;
pub mod secondary_mmap;
pub mod slot;

pub use backing::{AufBacking, GgufBacking, WeightSectionView};
pub use secondary_mmap::{
    LayerTensorSlice, LoadError, SecondaryDtypeChoice, SecondaryLayoutChoice, SecondaryMmap,
    SecondaryTensorInfo, build_auf_secondary_from_view, open_secondary, open_secondary_auf,
    open_secondary_with_backend, open_secondary_with_dtype, open_secondary_with_options,
};
pub use slot::{LayerSlot, LayerWeights};
