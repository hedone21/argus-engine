//! RawStore: zero-overhead KV cache storage without compression.
//!
//! Stores K/V data as raw `Vec<u8>` — no shuffle, no compression.
//! Ideal when data is high-entropy (e.g. F16 KV activations) and
//! compression yields ~1.0x ratio with non-trivial CPU cost.

use super::store::OffloadStore;
use anyhow::Result;

/// Uncompressed in-memory KV cache store for a single layer.
pub struct RawStore {
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    num_tokens: usize,
    token_bytes: usize,
}

impl RawStore {
    /// Create a new RawStore.
    ///
    /// - `token_bytes`: bytes per token for K (or V), i.e. kv_heads × head_dim × dtype_size
    pub fn new(token_bytes: usize) -> Self {
        Self {
            k_data: Vec::new(),
            v_data: Vec::new(),
            num_tokens: 0,
            token_bytes,
        }
    }
}

impl OffloadStore for RawStore {
    fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()> {
        let expected = num_tokens * self.token_bytes;
        anyhow::ensure!(k_data.len() == expected, "K data size mismatch");
        anyhow::ensure!(v_data.len() == expected, "V data size mismatch");

        self.k_data.clear();
        self.v_data.clear();
        self.k_data.extend_from_slice(k_data);
        self.v_data.extend_from_slice(v_data);
        self.num_tokens = num_tokens;
        Ok(())
    }

    fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize> {
        if self.num_tokens == 0 {
            return Ok(0);
        }
        let total_bytes = self.num_tokens * self.token_bytes;
        k_buf[..total_bytes].copy_from_slice(&self.k_data[..total_bytes]);
        v_buf[..total_bytes].copy_from_slice(&self.v_data[..total_bytes]);
        Ok(self.num_tokens)
    }

    fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()> {
        anyhow::ensure!(k_token.len() == self.token_bytes, "K token size mismatch");
        anyhow::ensure!(v_token.len() == self.token_bytes, "V token size mismatch");

        self.k_data.extend_from_slice(k_token);
        self.v_data.extend_from_slice(v_token);
        self.num_tokens += 1;
        Ok(())
    }

    fn storage_size(&self) -> usize {
        self.k_data.len() + self.v_data.len()
    }

    fn stored_tokens(&self) -> usize {
        self.num_tokens
    }

    fn clear(&mut self) {
        self.k_data.clear();
        self.v_data.clear();
        self.num_tokens = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_store_basic() {
        let token_bytes = 8 * 64 * 2; // 8 heads × 64 dim × F16
        let num_tokens = 32;
        let mut store = RawStore::new(token_bytes);

        let k_data: Vec<u8> = (0..num_tokens * token_bytes)
            .map(|i| (i % 256) as u8)
            .collect();
        let v_data: Vec<u8> = (0..num_tokens * token_bytes)
            .map(|i| ((i + 128) % 256) as u8)
            .collect();

        store.store(&k_data, &v_data, num_tokens).unwrap();
        assert_eq!(store.stored_tokens(), num_tokens);
        assert_eq!(store.storage_size(), k_data.len() + v_data.len());

        let mut k_buf = vec![0u8; k_data.len()];
        let mut v_buf = vec![0u8; v_data.len()];
        let loaded = store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(loaded, num_tokens);
        assert_eq!(&k_buf, &k_data, "K roundtrip mismatch");
        assert_eq!(&v_buf, &v_data, "V roundtrip mismatch");
    }

    #[test]
    fn test_raw_store_append_token() {
        let token_bytes = 64;
        let mut store = RawStore::new(token_bytes);

        let mut all_k = Vec::new();
        let mut all_v = Vec::new();

        for i in 0..50u8 {
            let k_tok = vec![i; token_bytes];
            let v_tok = vec![i.wrapping_add(100); token_bytes];
            store.append_token(&k_tok, &v_tok).unwrap();
            all_k.extend_from_slice(&k_tok);
            all_v.extend_from_slice(&v_tok);
        }

        assert_eq!(store.stored_tokens(), 50);

        let mut k_buf = vec![0u8; all_k.len()];
        let mut v_buf = vec![0u8; all_v.len()];
        let loaded = store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(loaded, 50);
        assert_eq!(&k_buf, &all_k, "K append mismatch");
        assert_eq!(&v_buf, &all_v, "V append mismatch");
    }

    #[test]
    fn test_raw_store_clear() {
        let token_bytes = 64;
        let mut store = RawStore::new(token_bytes);

        let k = vec![0xABu8; 16 * token_bytes];
        let v = vec![0xCDu8; 16 * token_bytes];
        store.store(&k, &v, 16).unwrap();

        store.clear();
        assert_eq!(store.stored_tokens(), 0);
        assert_eq!(store.storage_size(), 0);
    }

    #[test]
    fn test_raw_store_empty() {
        let store = RawStore::new(64);
        let mut k = vec![0u8; 64];
        let mut v = vec![0u8; 64];
        let n = store.load_into(&mut k, &mut v).unwrap();
        assert_eq!(n, 0);
    }
}
