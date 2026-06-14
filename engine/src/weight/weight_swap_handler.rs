//! `WeightSwapHandler` — weight-precision downgrade orchestrator (ENG-ALG-214-ROUTE).
//!
//! Phase 3 refactoring: `CachePressureHandler` impl removed per ENG-ALG-214-ROUTE "1안".
//! `EngineCommand::SwapWeights` is now dispatched directly in `generate.rs`
//! (see `dispatch_swap_weights`), bypassing the `CachePressurePipeline`.
//!
//! This module is kept as a thin orchestrator to preserve unit test surface area
//! and reusability.  `WeightSwapModelRef` + `WeightSwapHandler::execute_swap`
//! can still be used from integration tests or any caller that has the model
//! references but not access to the full `TransformerModel`.
//!
//! Spec: ENG-ALG-211, ENG-ALG-214-ROUTE, INV-121/123, WSWAP-2-HANDLER.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::memory::Memory;
use crate::model_config::ModelConfig;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure handler → inference weight resource (LayerSlot/SecondaryMmap)
use crate::models::weights::{LayerSlot, SecondaryMmap};
use crate::weight::swap_executor::SwapExecutor;

use crate::action_result::ActionResult;

/// Shared model references needed by `WeightSwapHandler` without requiring
/// `Arc<TransformerModel>` (the model is owned directly in `generate.rs`).
///
/// Constructed once at handler registration time and kept alive for the
/// duration of the `CachePressurePipeline`.
pub struct WeightSwapModelRef {
    /// Decoder layer slots shared via Arc. The forward path also holds
    /// these slots (via `TransformerModel::layers`), so the Arc keeps the
    /// backing data alive even after a swap until all in-flight forwards
    /// complete (INV-121/INV-123).
    ///
    /// LISWAP-2 Phase 6.1: each slot is itself wrapped in `Arc<LayerSlot>`
    /// so that the async swap dispatcher (Phase 6.2) can take ownership of
    /// a slot handle across thread boundaries.  Forward / handler paths
    /// transparently use `Arc::deref` on the inner slot.
    pub layers: Arc<Vec<Arc<LayerSlot>>>,
    /// Secondary GGUF handle. `None` disables the swap path (ENG-DAT-C09).
    pub secondary_mmap: Option<Arc<SecondaryMmap>>,
    /// Global swap generation counter shared with `TransformerModel`.
    /// Bumped by `SwapExecutor::execute_on_slots` after each batch.
    pub ratio_generation: Arc<std::sync::atomic::AtomicU64>,
    /// Model configuration (needed by `SwapExecutor` for Q/K permutation).
    pub config: ModelConfig,
    /// Backend on which fresh weight tensors should land.
    pub backend: Arc<dyn Backend>,
}

/// Thin orchestrator for weight swap execution (ENG-ALG-214-ROUTE "1안").
///
/// Phase 3: `CachePressureHandler` impl has been removed.  The Pipeline
/// path is no longer used for weight swaps — `EngineCommand::SwapWeights`
/// is dispatched directly in `generate.rs` via `dispatch_swap_weights`.
///
/// This struct is retained as a reusable helper for contexts that hold raw
/// model references (e.g., integration tests) rather than a full
/// `TransformerModel` Arc.
pub struct WeightSwapHandler {
    model_ref: Arc<WeightSwapModelRef>,
    memory: Arc<dyn Memory>,
    /// Target dtype to swap *to* (default `Q4_0`).
    target_dtype: DType,
}

impl WeightSwapHandler {
    /// Construct a handler.
    pub fn new(
        model_ref: Arc<WeightSwapModelRef>,
        memory: Arc<dyn Memory>,
        target_dtype: DType,
    ) -> Self {
        Self {
            model_ref,
            memory,
            target_dtype,
        }
    }

    /// Execute a weight swap for the given `target_layers`.
    ///
    /// Returns `ActionResult::WeightSwapped` if at least one layer was
    /// swapped, `ActionResult::NoOp` otherwise.  Callers are responsible
    /// for layer selection (use `WeightSwapDecider` from Phase 3).
    pub fn execute_swap(&self, target_layers: &[usize]) -> Result<ActionResult> {
        if target_layers.is_empty() || self.model_ref.secondary_mmap.is_none() {
            return Ok(ActionResult::NoOp);
        }

        let num_layers = self.model_ref.layers.len();
        let executor = SwapExecutor::new(
            self.target_dtype,
            &self.model_ref.config,
            self.model_ref.backend.clone(),
            self.memory.as_ref(),
        );

        match executor.execute_on_slots(
            self.model_ref.layers.as_slice(),
            self.model_ref.secondary_mmap.as_ref(),
            &self.model_ref.ratio_generation,
            target_layers,
            None, // async_dispatcher: sync path (Phase 3 upgrade pending)
        ) {
            Ok(report) => {
                let layers_changed = report.swapped.len();
                if layers_changed == 0 {
                    return Ok(ActionResult::NoOp);
                }
                log::info!(
                    "[WeightSwapHandler] swapped {}/{} layers in {:.1}ms",
                    layers_changed,
                    num_layers,
                    report.latency_ms,
                );
                Ok(ActionResult::WeightSwapped {
                    layers_changed,
                    freed_bytes: 0,
                    duration_ms: report.latency_ms,
                })
            }
            Err(e) => Err(anyhow::anyhow!("[WeightSwapHandler] execute failed: {}", e)),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::ModelConfig;

    fn make_minimal_config() -> ModelConfig {
        // Use a unique per-call temp dir to avoid race conditions between parallel tests.
        use std::time::{SystemTime, UNIX_EPOCH};
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let dir = std::path::PathBuf::from(format!(
            "/tmp/argus_engine_weight_swap_handler_test_{}_{:x}",
            std::process::id(),
            unique,
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let json = r#"{
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
            "model_type": "llama"
        }"#;
        std::fs::write(dir.join("config.json"), json).unwrap();
        ModelConfig::from_json(&dir).unwrap()
    }

    /// execute_swap with empty target_layers → NoOp (ENG-ALG-214-ROUTE: no-op when needed=0).
    #[test]
    fn execute_swap_empty_target_is_noop() {
        let model_ref = Arc::new(WeightSwapModelRef {
            layers: Arc::new(Vec::new()),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            config: make_minimal_config(),
            backend: crate::backend::cpu::cpu_singleton(),
        });
        let memory = Arc::new(crate::memory::galloc::Galloc::new());
        let handler = WeightSwapHandler::new(model_ref, memory, DType::Q4_0);

        let result = handler.execute_swap(&[]).unwrap();
        assert!(matches!(result, ActionResult::NoOp));
    }

    /// execute_swap with no secondary mmap → NoOp (ENG-DAT-C09).
    #[test]
    fn execute_swap_no_secondary_is_noop() {
        let model_ref = Arc::new(WeightSwapModelRef {
            layers: Arc::new(Vec::new()),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            config: make_minimal_config(),
            backend: crate::backend::cpu::cpu_singleton(),
        });
        let memory = Arc::new(crate::memory::galloc::Galloc::new());
        let handler = WeightSwapHandler::new(model_ref, memory, DType::Q4_0);

        let result = handler.execute_swap(&[0, 1, 2]).unwrap();
        assert!(matches!(result, ActionResult::NoOp));
    }
}
