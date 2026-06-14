//! DiskStore: file-based KV cache storage for disk offloading.
//!
//! Each layer has two files (K and V) written sequentially.
//! Uses standard buffered I/O (no mmap dependency).

use super::store::OffloadStore;
use anyhow::Result;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// File-backed KV cache storage for a single layer.
///
/// K and V data are stored in separate binary files.
/// Supports incremental append (decode) and bulk load (recall).
pub struct DiskStore {
    dir: PathBuf,
    layer_id: usize,
    k_file: File,
    v_file: File,
    stored_tokens: usize,
    bytes_per_token: usize, // per K or V: kv_heads * head_dim * dtype_size
}

impl DiskStore {
    /// Create a new DiskStore for the given layer.
    ///
    /// - `dir`: directory to store files (created if needed)
    /// - `layer_id`: layer index (used in filename)
    /// - `bytes_per_token`: bytes per token for K or V (kv_heads * head_dim * dtype_size)
    pub fn new(dir: PathBuf, layer_id: usize, bytes_per_token: usize) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        let k_path = dir.join(format!("layer{layer_id}_k.bin"));
        let v_path = dir.join(format!("layer{layer_id}_v.bin"));

        let k_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&k_path)?;
        let v_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&v_path)?;

        Ok(Self {
            dir,
            layer_id,
            k_file,
            v_file,
            stored_tokens: 0,
            bytes_per_token,
        })
    }
}

impl OffloadStore for DiskStore {
    fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()> {
        self.k_file.seek(SeekFrom::Start(0))?;
        self.k_file.set_len(0)?;
        self.k_file.write_all(k_data)?;
        self.k_file.flush()?;

        self.v_file.seek(SeekFrom::Start(0))?;
        self.v_file.set_len(0)?;
        self.v_file.write_all(v_data)?;
        self.v_file.flush()?;

        self.stored_tokens = num_tokens;
        Ok(())
    }

    fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize> {
        let expected_bytes = self.stored_tokens * self.bytes_per_token;
        if k_buf.len() < expected_bytes || v_buf.len() < expected_bytes {
            anyhow::bail!(
                "buffer too small: need {expected_bytes}, got k={} v={}",
                k_buf.len(),
                v_buf.len()
            );
        }

        let mut k_file = &self.k_file;
        k_file.seek(SeekFrom::Start(0))?;
        k_file.read_exact(&mut k_buf[..expected_bytes])?;

        let mut v_file = &self.v_file;
        v_file.seek(SeekFrom::Start(0))?;
        v_file.read_exact(&mut v_buf[..expected_bytes])?;

        Ok(self.stored_tokens)
    }

    fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()> {
        self.k_file.seek(SeekFrom::End(0))?;
        self.k_file.write_all(k_token)?;
        self.k_file.flush()?;

        self.v_file.seek(SeekFrom::End(0))?;
        self.v_file.write_all(v_token)?;
        self.v_file.flush()?;

        self.stored_tokens += 1;
        Ok(())
    }

    fn storage_size(&self) -> usize {
        self.stored_tokens * self.bytes_per_token * 2 // K + V
    }

    fn stored_tokens(&self) -> usize {
        self.stored_tokens
    }

    fn clear(&mut self) {
        let _ = self.k_file.set_len(0);
        let _ = self.v_file.set_len(0);
        self.stored_tokens = 0;
    }
}

impl Drop for DiskStore {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.dir.join(format!("layer{}_k.bin", self.layer_id)));
        let _ = fs::remove_file(self.dir.join(format!("layer{}_v.bin", self.layer_id)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_dir(name: &str) -> PathBuf {
        env::temp_dir().join(format!("argus_engine_test_disk_store_{name}"))
    }

    #[test]
    fn test_disk_store_roundtrip() {
        let dir = temp_dir("roundtrip");
        let _ = fs::remove_dir_all(&dir);
        let bpt = 16; // bytes per token

        let mut store = DiskStore::new(dir.clone(), 0, bpt).unwrap();

        let k_data: Vec<u8> = (0..64).collect(); // 4 tokens * 16 bytes
        let v_data: Vec<u8> = (64..128).collect();
        store.store(&k_data, &v_data, 4).unwrap();

        assert_eq!(store.stored_tokens(), 4);
        assert_eq!(store.storage_size(), 4 * 16 * 2);

        let mut k_buf = vec![0u8; 64];
        let mut v_buf = vec![0u8; 64];
        let n = store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(k_buf, k_data);
        assert_eq!(v_buf, v_data);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_store_append() {
        let dir = temp_dir("append");
        let _ = fs::remove_dir_all(&dir);
        let bpt = 8;

        let mut store = DiskStore::new(dir.clone(), 0, bpt).unwrap();

        // Append 3 tokens one by one
        for i in 0u8..3 {
            let k = vec![i * 10; bpt];
            let v = vec![i * 10 + 1; bpt];
            store.append_token(&k, &v).unwrap();
        }

        assert_eq!(store.stored_tokens(), 3);

        let mut k_buf = vec![0u8; 3 * bpt];
        let mut v_buf = vec![0u8; 3 * bpt];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        // Verify first token K data
        assert_eq!(&k_buf[..bpt], &vec![0u8; bpt]);
        // Verify second token K data
        assert_eq!(&k_buf[bpt..2 * bpt], &vec![10u8; bpt]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_store_clear() {
        let dir = temp_dir("clear");
        let _ = fs::remove_dir_all(&dir);
        let bpt = 8;

        let mut store = DiskStore::new(dir.clone(), 0, bpt).unwrap();
        let k = vec![1u8; 16];
        let v = vec![2u8; 16];
        store.store(&k, &v, 2).unwrap();

        assert_eq!(store.stored_tokens(), 2);
        store.clear();
        assert_eq!(store.stored_tokens(), 0);
        assert_eq!(store.storage_size(), 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_store_empty_load() {
        let dir = temp_dir("empty");
        let _ = fs::remove_dir_all(&dir);
        let bpt = 8;

        let store = DiskStore::new(dir.clone(), 0, bpt).unwrap();
        let mut k_buf = vec![0u8; 0];
        let mut v_buf = vec![0u8; 0];
        let n = store.load_into(&mut k_buf, &mut v_buf).unwrap();
        assert_eq!(n, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_store_large_data() {
        let dir = temp_dir("large");
        let _ = fs::remove_dir_all(&dir);
        let bpt = 512; // 8 heads * 64 dim * 1 byte (simulated)

        let mut store = DiskStore::new(dir.clone(), 0, bpt).unwrap();

        let num_tokens = 128;
        let k_data: Vec<u8> = (0..num_tokens * bpt).map(|i| (i % 256) as u8).collect();
        let v_data: Vec<u8> = (0..num_tokens * bpt)
            .map(|i| ((i + 128) % 256) as u8)
            .collect();
        store.store(&k_data, &v_data, num_tokens).unwrap();

        let mut k_buf = vec![0u8; num_tokens * bpt];
        let mut v_buf = vec![0u8; num_tokens * bpt];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(k_buf, k_data);
        assert_eq!(v_buf, v_data);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_store_overwrite() {
        let dir = temp_dir("overwrite");
        let _ = fs::remove_dir_all(&dir);
        let bpt = 8;

        let mut store = DiskStore::new(dir.clone(), 0, bpt).unwrap();

        // First write: 4 tokens
        let k1 = vec![0xAAu8; 4 * bpt];
        let v1 = vec![0xBBu8; 4 * bpt];
        store.store(&k1, &v1, 4).unwrap();
        assert_eq!(store.stored_tokens(), 4);

        // Overwrite with 2 tokens — must replace, not append
        let k2 = vec![0xCCu8; 2 * bpt];
        let v2 = vec![0xDDu8; 2 * bpt];
        store.store(&k2, &v2, 2).unwrap();
        assert_eq!(store.stored_tokens(), 2);
        assert_eq!(store.storage_size(), 2 * bpt * 2);

        let mut k_buf = vec![0u8; 2 * bpt];
        let mut v_buf = vec![0u8; 2 * bpt];
        let n = store.load_into(&mut k_buf, &mut v_buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(k_buf, k2);
        assert_eq!(v_buf, v2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_store_drop_cleanup() {
        let dir = temp_dir("drop_cleanup");
        let _ = fs::remove_dir_all(&dir);
        let bpt = 8;

        let layer_id = 3;
        {
            let mut store = DiskStore::new(dir.clone(), layer_id, bpt).unwrap();
            store.store(&vec![1u8; bpt], &vec![2u8; bpt], 1).unwrap();
            // Files should exist
            assert!(dir.join(format!("layer{layer_id}_k.bin")).exists());
            assert!(dir.join(format!("layer{layer_id}_v.bin")).exists());
        }
        // After drop, files should be cleaned up
        assert!(!dir.join(format!("layer{layer_id}_k.bin")).exists());
        assert!(!dir.join(format!("layer{layer_id}_v.bin")).exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_store_append_after_store() {
        let dir = temp_dir("append_after_store");
        let _ = fs::remove_dir_all(&dir);
        let bpt = 8;

        let mut store = DiskStore::new(dir.clone(), 0, bpt).unwrap();

        // Bulk store 2 tokens
        let k_init = vec![0xAAu8; 2 * bpt];
        let v_init = vec![0xBBu8; 2 * bpt];
        store.store(&k_init, &v_init, 2).unwrap();

        // Append 1 more token
        let k_tok = vec![0xCCu8; bpt];
        let v_tok = vec![0xDDu8; bpt];
        store.append_token(&k_tok, &v_tok).unwrap();

        assert_eq!(store.stored_tokens(), 3);

        let mut k_buf = vec![0u8; 3 * bpt];
        let mut v_buf = vec![0u8; 3 * bpt];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        // First 2 tokens from store, 3rd from append
        assert_eq!(&k_buf[..2 * bpt], &k_init);
        assert_eq!(&k_buf[2 * bpt..], &k_tok);
        assert_eq!(&v_buf[..2 * bpt], &v_init);
        assert_eq!(&v_buf[2 * bpt..], &v_tok);

        let _ = fs::remove_dir_all(&dir);
    }
}
