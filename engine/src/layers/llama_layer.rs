//! Backward-compatibility shim — re-exports from `transformer_layer`.
//!
//! All logic has moved to `crate::layers::transformer_layer`.
//! This module preserves the old type names as aliases.

pub use super::transformer_layer::{QkvBias, TransformerLayer};

/// Backward-compatible alias for [`TransformerLayer`].
pub type LlamaLayer = TransformerLayer;
