//! Swap handler — offloads cold KV cache data to disk on memory pressure.
//!
//! Two operating modes:
//!
//! 1. **Lossy** (default, backward-compatible): simply prunes the oldest tokens
//!    from the cache without writing them anywhere. Use `SwapHandler::new(ratio)`
//!    or `SwapHandler::default()`.
//! 2. **Disk write-back** (lossless recall possible): dumps the prune-prefix
//!    region of each layer's K/V buffer to `swap_dir/cache_L{layer}_s{seq}_{start}-{end}_{k,v}.bin`
//!    before pruning, and can restore them via `recall_caches`. The `s{seq}` segment is a monotonic
//!    per-ledger counter that keeps repeated offloads of the same layer + count from clobbering each
//!    other (see `SwapState::next_seq`). Enable with
//!    `SwapHandler::with_disk(ratio, swap_dir)`.
//!
//! LRU strategy: always offloads the oldest tokens first.
//! Currently only F32/F16 KV dtypes are supported for disk write-back.
//! Q4_0/quant-window caches fall back to lossy prune with a stderr warning.

use super::{ActionResult, CachePressureHandler, HandlerContext, PressureLevel};
use crate::buffer::DType;
use crate::kv::kv_cache::KVCache;
use crate::kv_cache_ops::KVLayout;
use anyhow::Result;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Record of a single offload operation.
///
/// Each record stores enough metadata to re-insert the prefix region
/// back into the KV buffer during `recall_caches`.
pub struct SwapRecord {
    pub layer_idx: usize,
    pub token_count: usize,
    pub k_path: PathBuf,
    pub v_path: PathBuf,
    pub dtype: DType,
    /// Bytes between consecutive heads in the on-disk layout.
    pub head_stride_bytes: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

/// In-memory ledger of pending swap records.
#[derive(Default)]
pub struct SwapState {
    pub records: Vec<SwapRecord>,
    /// Monotonic offload counter, appended to every backing-file name. Without it the name is keyed
    /// only on `(layer_idx, offload_count)`, so a second offload of the same layer with the same count
    /// (before its recall) would reuse the same path and clobber the first dump — leaving two records
    /// pointing at one file, so the first recall reads the second dump's bytes (KV corruption). The
    /// seq makes each dump's path unique. It lives in the shared ledger so handlers that share a
    /// `state` (and thus a `swap_dir`) draw from one sequence.
    next_seq: u64,
}

/// Offloads old KV cache tokens to disk (or just prunes them in lossy mode).
pub struct SwapHandler {
    /// Fraction of tokens to offload (0.0–1.0). Default: 0.5.
    pub offload_ratio: f32,
    /// When `Some`, writes raw prefix bytes to `swap_dir` before pruning so
    /// they can be later restored with `recall_caches`. When `None`, performs
    /// the legacy lossy prune.
    pub swap_dir: Option<PathBuf>,
    /// Shared ledger of outstanding swap records.
    pub state: Arc<Mutex<SwapState>>,
}

impl SwapHandler {
    /// Lossy swap handler (no disk write). Kept for backward compatibility.
    pub fn new(offload_ratio: f32) -> Self {
        Self {
            offload_ratio,
            swap_dir: None,
            state: Arc::new(Mutex::new(SwapState::default())),
        }
    }

    /// Disk-backed swap handler. Prefixes are dumped into `swap_dir` on offload
    /// and can be recalled via `recall_caches`.
    pub fn with_disk(offload_ratio: f32, swap_dir: PathBuf) -> Self {
        Self {
            offload_ratio,
            swap_dir: Some(swap_dir),
            state: Arc::new(Mutex::new(SwapState::default())),
        }
    }

    /// Update the offload ratio (used by CacheManager::offload to honor the
    /// `ratio` field on `KvOffload` directives).
    pub fn set_ratio(&mut self, ratio: f32) {
        self.offload_ratio = ratio.clamp(0.0, 1.0);
    }

    /// Perform an offload on the given caches regardless of pressure level.
    /// Called directly from CacheManager for `KvOffload` directives.
    /// Returns the total number of tokens offloaded across all layers.
    pub fn offload_caches(&self, caches: &mut [KVCache]) -> Result<usize> {
        let mut total = 0usize;
        for (layer_idx, cache) in caches.iter_mut().enumerate() {
            total += self.offload_one(layer_idx, cache)?;
        }
        Ok(total)
    }

    /// Restore previously offloaded tokens back into their caches.
    /// Returns the total number of tokens recalled.
    ///
    /// On capacity overflow or any recoverable error the record is skipped
    /// with a warning and `Ok(0)` is contributed for that layer rather than
    /// returning `Err`.
    pub fn recall_caches(&self, caches: &mut [KVCache]) -> Result<usize> {
        let records = {
            let mut guard = self.state.lock().unwrap();
            std::mem::take(&mut guard.records)
        };

        if records.is_empty() {
            return Ok(0);
        }

        let mut total = 0usize;
        for rec in records {
            if rec.layer_idx >= caches.len() {
                eprintln!(
                    "[SwapHandler] recall: layer_idx {} out of range (n_layers={}), skipping",
                    rec.layer_idx,
                    caches.len()
                );
                continue;
            }
            match Self::recall_one(&mut caches[rec.layer_idx], &rec) {
                Ok(n) => total += n,
                Err(e) => {
                    eprintln!(
                        "[SwapHandler] recall failed for layer {}: {}",
                        rec.layer_idx, e
                    );
                }
            }
            // Best-effort cleanup; we keep going if unlink fails.
            let _ = fs::remove_file(&rec.k_path);
            let _ = fs::remove_file(&rec.v_path);
        }
        Ok(total)
    }

    // ── Internals ─────────────────────────────────────────────────

    /// Offload the LRU prefix of a single layer's cache.
    /// Falls back to lossy prune on unsupported dtypes (Q4_0/quant-window) or when
    /// `swap_dir` is unset.
    fn offload_one(&self, layer_idx: usize, cache: &mut KVCache) -> Result<usize> {
        let total = cache.current_pos;
        if total == 0 {
            return Ok(0);
        }
        let offload_count = ((total as f32 * self.offload_ratio) as usize).max(1);
        self.offload_prefix(layer_idx, cache, offload_count)
    }

    /// Offload an explicit `offload_count`-token LRU prefix of one layer, decoupled from
    /// `offload_ratio`. The `CacheHandle::offload` residency op routes here; `offload_one` computes a
    /// ratio-driven count then delegates. Guards `current_pos == 0` and `offload_count >= current_pos`
    /// (never offload everything / nothing). Same disk/lossy body as the ratio-driven path.
    pub(crate) fn offload_prefix(
        &self,
        layer_idx: usize,
        cache: &mut KVCache,
        offload_count: usize,
    ) -> Result<usize> {
        let total = cache.current_pos;
        if total == 0 || offload_count == 0 || offload_count >= total {
            return Ok(0); // Don't offload everything / nothing.
        }

        let dtype = cache.k_buffer.dtype();
        let supported = matches!(dtype, DType::F32 | DType::F16);

        if let Some(dir) = self.swap_dir.as_ref() {
            if !supported {
                eprintln!(
                    "[SwapHandler] dtype {:?} not supported for disk swap (layer {}), falling back to lossy prune",
                    dtype, layer_idx
                );
                cache.prune_prefix(offload_count)?;
                return Ok(offload_count);
            }
            if cache.layout() != KVLayout::HeadMajor {
                eprintln!(
                    "[SwapHandler] non-HeadMajor layout at layer {}; falling back to lossy prune",
                    layer_idx
                );
                cache.prune_prefix(offload_count)?;
                return Ok(offload_count);
            }

            // Ensure directory exists (best effort).
            if let Err(e) = fs::create_dir_all(dir) {
                eprintln!(
                    "[SwapHandler] create_dir_all({}) failed: {} — falling back to lossy prune",
                    dir.display(),
                    e
                );
                cache.prune_prefix(offload_count)?;
                return Ok(offload_count);
            }

            let head_dim = cache.head_dim();
            let n_heads = cache.kv_heads();
            let capacity = cache.capacity();
            let elem_bytes = dtype.size();
            let bytes_per_pos_per_head = head_dim * elem_bytes;
            let bytes_this_dump = offload_count * bytes_per_pos_per_head * n_heads;

            let mut k_bytes = vec![0u8; bytes_this_dump];
            let mut v_bytes = vec![0u8; bytes_this_dump];

            // HeadMajor layout: per-head contiguous [capacity * head_dim] elements.
            // Pull the [0..offload_count] prefix from each head into k_bytes/v_bytes.
            // We use read_buffer over the full tensor then copy per-head ranges.
            let k_all_size = cache.k_buffer.size();
            let v_all_size = cache.v_buffer.size();
            let mut k_all = vec![0u8; k_all_size];
            let mut v_all = vec![0u8; v_all_size];
            let backend = cache.k_buffer.backend().clone();
            backend.read_buffer(&cache.k_buffer, &mut k_all)?;
            backend.read_buffer(&cache.v_buffer, &mut v_all)?;

            let head_stride_bytes_src = capacity * head_dim * elem_bytes;
            let head_stride_bytes_dst = offload_count * head_dim * elem_bytes;
            for h in 0..n_heads {
                let src_base = h * head_stride_bytes_src;
                let dst_base = h * head_stride_bytes_dst;
                let cp = head_stride_bytes_dst;
                k_bytes[dst_base..dst_base + cp].copy_from_slice(&k_all[src_base..src_base + cp]);
                v_bytes[dst_base..dst_base + cp].copy_from_slice(&v_all[src_base..src_base + cp]);
            }

            // Draw a unique sequence number so this dump's backing files never collide with an earlier
            // (still-outstanding) dump of the same layer + count (see `SwapState::next_seq`).
            let seq = {
                let mut guard = self.state.lock().unwrap();
                let s = guard.next_seq;
                guard.next_seq += 1;
                s
            };
            let k_path = dir.join(format!(
                "cache_L{}_s{}_{}-{}_k.bin",
                layer_idx, seq, 0, offload_count
            ));
            let v_path = dir.join(format!(
                "cache_L{}_s{}_{}-{}_v.bin",
                layer_idx, seq, 0, offload_count
            ));
            if let Err(e) = fs::write(&k_path, &k_bytes) {
                eprintln!(
                    "[SwapHandler] write {} failed: {} — falling back to lossy prune",
                    k_path.display(),
                    e
                );
                cache.prune_prefix(offload_count)?;
                return Ok(offload_count);
            }
            if let Err(e) = fs::write(&v_path, &v_bytes) {
                eprintln!(
                    "[SwapHandler] write {} failed: {} — falling back to lossy prune",
                    v_path.display(),
                    e
                );
                let _ = fs::remove_file(&k_path);
                cache.prune_prefix(offload_count)?;
                return Ok(offload_count);
            }

            // Register the record before prune so recall can find it.
            {
                let mut guard = self.state.lock().unwrap();
                guard.records.push(SwapRecord {
                    layer_idx,
                    token_count: offload_count,
                    k_path,
                    v_path,
                    dtype,
                    head_stride_bytes: head_stride_bytes_dst,
                    n_heads,
                    head_dim,
                });
            }

            cache.prune_prefix(offload_count)?;
            Ok(offload_count)
        } else {
            // Lossy path (default / backward-compatible).
            cache.prune_prefix(offload_count)?;
            Ok(offload_count)
        }
    }

    /// Read one record's files and insert them back as the new prefix.
    fn recall_one(cache: &mut KVCache, rec: &SwapRecord) -> Result<usize> {
        if cache.layout() != KVLayout::HeadMajor {
            anyhow::bail!("recall requires HeadMajor layout");
        }
        if cache.head_dim() != rec.head_dim || cache.kv_heads() != rec.n_heads {
            anyhow::bail!(
                "recall: head layout mismatch (cache {}x{}, record {}x{})",
                cache.kv_heads(),
                cache.head_dim(),
                rec.n_heads,
                rec.head_dim
            );
        }
        if cache.k_buffer.dtype() != rec.dtype {
            anyhow::bail!(
                "recall: dtype mismatch (cache {:?}, record {:?})",
                cache.k_buffer.dtype(),
                rec.dtype
            );
        }

        let count = rec.token_count;
        let existing = cache.current_pos;
        let mut capacity = cache.capacity();
        if existing + count > capacity {
            // Grow the dynamic cache to hold the recalled prefix + current
            // tokens. Without this, recall silently dropped data when the
            // cache had grown just enough to fit decode after offload but
            // not enough to hold offloaded + current (e.g. offload 20, then
            // decode 49 tokens → capacity=64 < 49 + 20 = 69).
            // Phase α-K BC 5-E: KVCache inherent `ensure_capacity` 직접 호출 (KVCacheOps import 제거).
            if let Err(e) = cache.ensure_capacity(existing + count) {
                eprintln!(
                    "[SwapHandler] recall for layer {}: grow to {} failed: {}; skipping",
                    rec.layer_idx,
                    existing + count,
                    e
                );
                return Ok(0);
            }
            capacity = cache.capacity();
        }

        // Shift existing tokens forward by `count` to make room for the prefix.
        if existing > 0 {
            cache.shift_positions(0, count, existing)?;
        }

        // Load raw bytes.
        let k_bytes = fs::read(&rec.k_path)?;
        let v_bytes = fs::read(&rec.v_path)?;

        let head_dim = rec.head_dim;
        let n_heads = rec.n_heads;
        let elem_bytes = rec.dtype.size();
        let head_stride_bytes_dst = capacity * head_dim * elem_bytes;
        let per_head_bytes = count * head_dim * elem_bytes;

        if k_bytes.len() != per_head_bytes * n_heads || v_bytes.len() != per_head_bytes * n_heads {
            anyhow::bail!(
                "recall: file size mismatch (k={}, v={}, expected={})",
                k_bytes.len(),
                v_bytes.len(),
                per_head_bytes * n_heads
            );
        }

        let backend = cache.k_buffer.backend().clone();
        for h in 0..n_heads {
            let dst_base = h * head_stride_bytes_dst;
            let src_base = h * per_head_bytes;
            backend.write_buffer_range(
                &mut cache.k_buffer,
                &k_bytes[src_base..src_base + per_head_bytes],
                dst_base,
            )?;
            backend.write_buffer_range(
                &mut cache.v_buffer,
                &v_bytes[src_base..src_base + per_head_bytes],
                dst_base,
            )?;
        }

        cache.current_pos = existing + count;
        if cache.current_pos > cache.high_water_pos {
            cache.high_water_pos = cache.current_pos;
        }
        Ok(count)
    }

    /// Recall the first outstanding record for `layer_idx` back into `cache` (the `CacheHandle::recall`
    /// residency op routes here). Pops that layer's record from the ledger, restores its prefix, and
    /// cleans up the backing files. `Ok(0)` when this layer has no outstanding record. The all-layers
    /// `recall_caches` is the ratio/pressure-driven twin.
    pub(crate) fn recall_layer(&self, layer_idx: usize, cache: &mut KVCache) -> Result<usize> {
        let rec = {
            let mut guard = self.state.lock().unwrap();
            guard
                .records
                .iter()
                .position(|r| r.layer_idx == layer_idx)
                .map(|i| guard.records.remove(i))
        };
        match rec {
            Some(rec) => {
                // Clean up the backing files UNCONDITIONALLY (mirroring `recall_caches`), even when
                // recall_one fails (e.g. a dtype mismatch after an intervening re-encode): the record
                // was already popped, so leaving the files would leak them. Run the restore, drop the
                // files regardless, then surface the restore result to the transactional caller.
                let result = Self::recall_one(cache, &rec);
                let _ = fs::remove_file(&rec.k_path);
                let _ = fs::remove_file(&rec.v_path);
                result
            }
            None => Ok(0),
        }
    }
}

impl Default for SwapHandler {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl CachePressureHandler for SwapHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
        // Only activate on Warning+ pressure
        if ctx.pressure_level < PressureLevel::Warning {
            return Ok(ActionResult::NoOp);
        }

        let mut total_swapped = 0;
        for (layer_idx, cache) in ctx.caches.iter_mut().enumerate() {
            total_swapped += self.offload_one(layer_idx, cache)?;
        }

        if total_swapped > 0 {
            Ok(ActionResult::Swapped {
                tokens_swapped: total_swapped,
            })
        } else {
            Ok(ActionResult::NoOp)
        }
    }

    fn name(&self) -> &str {
        "swap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::PressureLevel;
    use crate::kv::kv_cache::KVCache;
    use crate::kv_cache_ops::KVLayout;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache(num_tokens: usize) -> KVCache {
        let max_seq = 100;
        let heads = 1;
        let dim = 4;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * heads * dim * 4;

        let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

        let k = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, max_seq, heads, dim]), v_buf, backend);
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = num_tokens;
        cache
    }

    /// HeadMajor cache initialized with recognizable F32 data for byte-level checks.
    /// Uses `KVCache::new()` with SeqMajor-shaped Tensor, then flips layout —
    /// `with_layout` is the supported recipe for tests outside `kv_cache.rs`.
    fn make_hm_cache_with_data(num_tokens: usize, heads: usize, dim: usize) -> KVCache {
        let max_seq = 32;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * heads * dim * 4;
        let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, max_seq, heads, dim]), v_buf, backend);
        let mut cache = KVCache::new(k, v, max_seq).with_layout(KVLayout::HeadMajor);
        cache.current_pos = num_tokens;
        // Fill with deterministic F32 data: buffer[i] = i as f32.
        {
            let k_slice = cache.k_buffer.as_mut_slice::<f32>();
            for (i, v) in k_slice.iter_mut().enumerate() {
                *v = i as f32;
            }
            let v_slice = cache.v_buffer.as_mut_slice::<f32>();
            for (i, v) in v_slice.iter_mut().enumerate() {
                *v = -(i as f32);
            }
        }
        cache
    }

    #[test]
    fn test_swap_normal_noop() {
        let handler = SwapHandler::default();
        let mut caches = vec![make_cache(50)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Normal,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
        assert_eq!(ctx.caches[0].current_pos, 50);
    }

    #[test]
    fn test_swap_warning_offloads() {
        let handler = SwapHandler::new(0.5);
        let mut caches = vec![make_cache(50)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Warning,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(result.is_action());
        assert_eq!(ctx.caches[0].current_pos, 25); // 50 - 25 = 25
    }

    #[test]
    fn test_swap_emergency_offloads() {
        let handler = SwapHandler::new(0.75);
        let mut caches = vec![make_cache(40)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(result.is_action());
        assert_eq!(ctx.caches[0].current_pos, 10); // 40 - 30 = 10
    }

    #[test]
    fn test_swap_empty_cache() {
        let handler = SwapHandler::default();
        let mut caches = vec![make_cache(0)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Warning,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_swap_name() {
        assert_eq!(SwapHandler::default().name(), "swap");
    }

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "argus_engine_swap_test_{}_{}_{}",
            tag,
            std::process::id(),
            nanos
        ));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_offload_writes_to_disk() {
        let dir = unique_tmp_dir("offload_write");
        let handler = SwapHandler::with_disk(0.5, dir.clone());
        let mut caches = vec![make_hm_cache_with_data(10, 2, 4)];

        let n = handler.offload_caches(&mut caches).unwrap();
        assert_eq!(n, 5); // 10 * 0.5 = 5
        assert_eq!(caches[0].current_pos, 5);

        let guard = handler.state.lock().unwrap();
        assert_eq!(guard.records.len(), 1);
        let rec = &guard.records[0];
        assert_eq!(rec.layer_idx, 0);
        assert_eq!(rec.token_count, 5);
        assert!(rec.k_path.exists());
        assert!(rec.v_path.exists());

        // Cleanup
        drop(guard);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_recall_restores_data() {
        let dir = unique_tmp_dir("offload_recall");
        let handler = SwapHandler::with_disk(0.5, dir.clone());
        let mut caches = vec![make_hm_cache_with_data(10, 2, 4)];

        // Snapshot original per-head prefix (first 5 positions of each head).
        let capacity = caches[0].capacity();
        let head_dim = caches[0].head_dim();
        let n_heads = caches[0].kv_heads();
        let mut orig_k = Vec::<f32>::new();
        let mut orig_v = Vec::<f32>::new();
        {
            let k_slice: &[f32] = caches[0].k_buffer.as_slice();
            let v_slice: &[f32] = caches[0].v_buffer.as_slice();
            for h in 0..n_heads {
                let base = h * capacity * head_dim;
                for pos in 0..5 {
                    let off = base + pos * head_dim;
                    orig_k.extend_from_slice(&k_slice[off..off + head_dim]);
                    orig_v.extend_from_slice(&v_slice[off..off + head_dim]);
                }
            }
        }

        handler.offload_caches(&mut caches).unwrap();
        assert_eq!(caches[0].current_pos, 5);

        let recalled = handler.recall_caches(&mut caches).unwrap();
        assert_eq!(recalled, 5);
        assert_eq!(caches[0].current_pos, 10);

        // Verify prefix (positions 0..5) matches original after recall.
        let k_slice: &[f32] = caches[0].k_buffer.as_slice();
        let v_slice: &[f32] = caches[0].v_buffer.as_slice();
        let mut ok = true;
        let mut idx = 0usize;
        for h in 0..n_heads {
            let base = h * capacity * head_dim;
            for pos in 0..5 {
                let off = base + pos * head_dim;
                for d in 0..head_dim {
                    if (k_slice[off + d] - orig_k[idx]).abs() > 1e-6 {
                        ok = false;
                    }
                    if (v_slice[off + d] - orig_v[idx]).abs() > 1e-6 {
                        ok = false;
                    }
                    idx += 1;
                }
            }
        }
        assert!(ok, "recalled prefix must match original bytes");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_recall_without_offload_noop() {
        let dir = unique_tmp_dir("offload_noop");
        let handler = SwapHandler::with_disk(0.5, dir.clone());
        let mut caches = vec![make_hm_cache_with_data(10, 2, 4)];

        let before = caches[0].current_pos;
        let n = handler.recall_caches(&mut caches).unwrap();
        assert_eq!(n, 0);
        assert_eq!(caches[0].current_pos, before);

        let _ = fs::remove_dir_all(&dir);
    }

    /// recall_layer cleans up the popped record's backing files even when recall_one FAILS (a layout /
    /// dtype mismatch, e.g. after an intervening re-encode). Mutation-proof: reverting recall_layer to
    /// `?`-propagate-before-cleanup leaves the two .bin files on disk, failing the final assertions.
    #[test]
    fn test_recall_layer_cleans_files_on_error() {
        let dir = unique_tmp_dir("recall_layer_err");
        let handler = SwapHandler::with_disk(0.5, dir.clone());
        let mut caches = vec![make_hm_cache_with_data(10, 2, 4)];
        handler.offload_caches(&mut caches).unwrap();
        let (kp, vp) = {
            let g = handler.state.lock().unwrap();
            (g.records[0].k_path.clone(), g.records[0].v_path.clone())
        };
        assert!(
            kp.exists() && vp.exists(),
            "offload wrote the backing files"
        );

        // Recall into a cache with a DIFFERENT head_dim (8 != recorded 4) → recall_one bails on the
        // layout mismatch, but recall_layer must still drop the popped record's files.
        let mut other = make_hm_cache_with_data(0, 2, 8);
        let res = handler.recall_layer(0, &mut other);
        assert!(res.is_err(), "recall_one must fail on the layout mismatch");
        assert!(
            handler.state.lock().unwrap().records.is_empty(),
            "the record was popped"
        );
        assert!(
            !kp.exists() && !vp.exists(),
            "backing files cleaned up on the recall error path (no leak)"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Per-head `[0..count]` K/V prefix snapshot in the live HeadMajor stride (`h*capacity*head_dim`),
    /// the layout `offload_prefix` dumps and `recall_one` restores. Used to capture the exact bytes a
    /// dump should round-trip.
    fn prefix_kv(cache: &KVCache, count: usize) -> (Vec<f32>, Vec<f32>) {
        let capacity = cache.capacity();
        let head_dim = cache.head_dim();
        let n_heads = cache.kv_heads();
        let k: &[f32] = cache.k_buffer.as_slice();
        let v: &[f32] = cache.v_buffer.as_slice();
        let (mut ks, mut vs) = (Vec::new(), Vec::new());
        for h in 0..n_heads {
            let base = h * capacity * head_dim;
            for pos in 0..count {
                let off = base + pos * head_dim;
                ks.extend_from_slice(&k[off..off + head_dim]);
                vs.extend_from_slice(&v[off..off + head_dim]);
            }
        }
        (ks, vs)
    }

    /// RES-1 regression: two offloads of the SAME layer with the SAME count (before recall) must NOT
    /// share backing files. With the pre-fix `cache_L{layer}_{0}-{count}` naming the second dump
    /// clobbers the first's file and both records point at it, so recalling the first record returns
    /// the SECOND dump's bytes (KV corruption). The `s{seq}` segment gives each dump a unique path —
    /// for BOTH the K and V backing files — so each record round-trips its own bytes. Mutation-proof:
    /// dropping the seq makes the first recall read pattern B and the `expected_a` assertion fails.
    #[test]
    fn test_repeat_offload_same_layer_count_no_clobber() {
        let dir = unique_tmp_dir("repeat_offload");
        let handler = SwapHandler::with_disk(0.5, dir.clone());
        let (heads, dim, count) = (2usize, 4usize, 3usize);

        // Dump #1: pattern A (the make_hm_cache_with_data fill: k[i]=i, v[i]=-i).
        let mut cache = make_hm_cache_with_data(10, heads, dim);
        let expected_a = prefix_kv(&cache, count);
        let n1 = handler.offload_prefix(0, &mut cache, count).unwrap();
        assert_eq!(n1, count);

        // Overwrite the whole buffer with a DISTINCT pattern B and re-arm current_pos, then dump #2 of
        // the same layer + count without an intervening recall.
        {
            let k_slice = cache.k_buffer.as_mut_slice::<f32>();
            for (i, x) in k_slice.iter_mut().enumerate() {
                *x = 7000.0 + i as f32;
            }
            let v_slice = cache.v_buffer.as_mut_slice::<f32>();
            for (i, x) in v_slice.iter_mut().enumerate() {
                *x = -(7000.0 + i as f32);
            }
        }
        cache.current_pos = 10;
        let expected_b = prefix_kv(&cache, count);
        let n2 = handler.offload_prefix(0, &mut cache, count).unwrap();
        assert_eq!(n2, count);

        // Both dumps are outstanding, same (layer=0, count=3) — but on DISTINCT backing files.
        // Check BOTH the K and V paths: the seq is baked into each independently, so a regression
        // that uniquified only one of them would still corrupt the other.
        let (a_kpath, b_kpath, a_vpath, b_vpath) = {
            let g = handler.state.lock().unwrap();
            assert_eq!(g.records.len(), 2, "both dumps recorded");
            (
                g.records[0].k_path.clone(),
                g.records[1].k_path.clone(),
                g.records[0].v_path.clone(),
                g.records[1].v_path.clone(),
            )
        };
        assert_ne!(
            a_kpath, b_kpath,
            "the two same-(layer,count) dumps must not share a K backing file"
        );
        assert_ne!(
            a_vpath, b_vpath,
            "the two same-(layer,count) dumps must not share a V backing file"
        );

        // recall_layer pops layer-0 records in insertion order: first the A dump, then the B dump.
        // Each must restore ITS OWN pattern into a fresh empty cache.
        let mut fresh_a = make_hm_cache_with_data(0, heads, dim);
        assert_eq!(handler.recall_layer(0, &mut fresh_a).unwrap(), count);
        assert_eq!(
            prefix_kv(&fresh_a, count),
            expected_a,
            "first record must restore dump A (not the clobbering dump B)"
        );

        let mut fresh_b = make_hm_cache_with_data(0, heads, dim);
        assert_eq!(handler.recall_layer(0, &mut fresh_b).unwrap(), count);
        assert_eq!(
            prefix_kv(&fresh_b, count),
            expected_b,
            "second record must restore dump B"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// `recall_caches` (the all-layers / pressure-driven path) restores MULTIPLE outstanding records
    /// in a single call, each back into ITS OWN layer's cache. The existing `test_recall_restores_data`
    /// only exercises one record; this covers the loop over several. The records are deliberately
    /// recorded OUT of layer order (offloaded as 2, 0, 1) and each layer carries a distinct pattern, so
    /// a mis-dispatch — indexing by loop position instead of `rec.layer_idx`, or reading a sibling's
    /// backing file — routes a record's bytes into the wrong layer and trips a per-layer byte assert.
    /// Mutation-proof two ways: hardcoding the recall target to `caches[0]` makes layer 0 over-stack to
    /// length 20; dispatching by the loop index (`caches[i]`) instead of `rec.layer_idx` mismatches the
    /// per-layer prefix bytes (this is why the out-of-order offload matters — identity order can't tell
    /// the two dispatch strategies apart).
    #[test]
    fn test_recall_caches_multi_layer_restores_each_own_bytes() {
        let dir = unique_tmp_dir("recall_caches_multi");
        let handler = SwapHandler::with_disk(0.5, dir.clone());
        let (heads, dim, n_layers) = (2usize, 4usize, 3usize);

        // Build N layers, each stamped with a layer-distinct pattern so any cross-layer mixup is
        // visible (the gap between tags >> the buffer length, so no two layers can alias a value).
        let mut caches: Vec<KVCache> = (0..n_layers)
            .map(|_| make_hm_cache_with_data(10, heads, dim))
            .collect();
        for (l, cache) in caches.iter_mut().enumerate() {
            let tag = (l as f32 + 1.0) * 100_000.0;
            let k_slice = cache.k_buffer.as_mut_slice::<f32>();
            for (i, x) in k_slice.iter_mut().enumerate() {
                *x = tag + i as f32;
            }
            let v_slice = cache.v_buffer.as_mut_slice::<f32>();
            for (i, x) in v_slice.iter_mut().enumerate() {
                *x = -(tag + i as f32);
            }
        }

        // Snapshot each layer's OWN prefix (first 5 positions) before offload.
        let expected: Vec<(Vec<f32>, Vec<f32>)> = caches.iter().map(|c| prefix_kv(c, 5)).collect();

        // Offload in a NON-identity order so the ledger's records are NOT in layer order
        // (records become [L2, L0, L1]). This is what makes the test able to distinguish a correct
        // `caches[rec.layer_idx]` dispatch from a positional `caches[loop_index]` bug — under identity
        // order the two are indistinguishable. offload_count per layer = 10 * 0.5 = 5.
        let mut offloaded = 0usize;
        for &l in &[2usize, 0, 1] {
            offloaded += handler.offload_prefix(l, &mut caches[l], 5).unwrap();
        }
        assert_eq!(offloaded, 5 * n_layers, "every layer offloaded its half");
        {
            let g = handler.state.lock().unwrap();
            assert_eq!(
                g.records.len(),
                n_layers,
                "one outstanding record per layer"
            );
            assert_eq!(
                g.records.iter().map(|r| r.layer_idx).collect::<Vec<_>>(),
                vec![2usize, 0, 1],
                "records are intentionally NOT in identity layer order"
            );
        }
        for cache in &caches {
            assert_eq!(cache.current_pos, 5);
        }

        // ONE recall_caches call restores ALL layers at once.
        let recalled = handler.recall_caches(&mut caches).unwrap();
        assert_eq!(recalled, 5 * n_layers, "all layers recalled in one call");
        assert!(
            handler.state.lock().unwrap().records.is_empty(),
            "ledger drained after recall_caches"
        );

        // Each layer must hold back exactly its OWN prefix — not a sibling's.
        for (l, cache) in caches.iter().enumerate() {
            assert_eq!(cache.current_pos, 10, "layer {} restored to full length", l);
            assert_eq!(
                prefix_kv(cache, 5),
                expected[l],
                "layer {} must restore its OWN prefix bytes, not another layer's",
                l
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
