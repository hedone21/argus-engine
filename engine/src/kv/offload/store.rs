//! OffloadStore trait: abstraction for KV cache layer storage backends.

use anyhow::Result;

/// Backend-agnostic KV cache storage for a single layer.
///
/// Implementation: `RawStore` (in-memory, no compression).
/// Each store instance holds data for one transformer layer (K + V).
pub trait OffloadStore: Send {
    /// Write full KV data to storage (used during migration).
    fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()>;

    /// Load KV data from storage into pre-allocated buffers.
    /// Returns the number of tokens loaded.
    fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize>;

    /// Append a single token's K/V data (used during decode).
    fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()>;

    /// Current storage size in bytes.
    fn storage_size(&self) -> usize;

    /// Number of tokens currently stored.
    fn stored_tokens(&self) -> usize;

    /// Reset storage to empty state.
    fn clear(&mut self);
}
