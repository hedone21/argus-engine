//! Pressure-owned weight runtime resources.
//!
//! What remains after the QCF-driven precision-swap orchestration was removed: the per-layer
//! quantization-noise table, the background worker that releases primary weights under memory
//! pressure, and the `RuntimeResources` bundle `TransformerModel` installs at construction. None of
//! these decided anything — they are the resources the swap machinery happened to be built on, and
//! the loader and the model still need them.
//!
//! Inference-side weight resource definitions (LayerSlot, SecondaryMmap, etc.) remain in
//! `models/weights/` as loader artifacts.

pub mod noise_table;
pub mod release_worker;
pub mod setup;

pub use noise_table::{QuantNoiseTable, compute_quant_noise};
pub use release_worker::PrimaryReleaseWorker;
pub use setup::{RuntimeResources, setup_runtime_resources};
