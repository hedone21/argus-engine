use crate::buffer::opaque::OpaqueBuffer;
use crate::buffer::{Buffer, DType};
use crate::format::encode_via_descriptor;
use crate::memory::Memory;
use crate::shape::Shape;
use crate::tensor::Tensor;
use anyhow::Result;
use argus_extension_api::KVLayoutDesc;
use std::sync::Arc;

// BC re-export: KVLayout, QuantWindowRawBuffers 를 L2(kv_cache_ops)로 격상. KVCacheOps trait 은
// 5-F 에서 폐기됨(inherent 전환 완료) — kv_cache_ops.rs 는 이제 layout/raw-buffer 타입 전용.
pub use crate::kv_cache_ops::{KVLayout, QuantWindowRawBuffers};

/// Extension trait for KV caches that support prefetch pipelines.
///
/// Implementors: `OffloadKVCache`.
/// Used by `forward_into_offload` to overlap I/O with compute.
///
/// **KVCacheOps 비의존 (Phase α-K BC Step 2, B-3 offload 분리)** — 구
/// `PrefetchableCache: KVCacheOps` supertrait 를 제거해 prefetch 계약(preload/release/retain)을
/// KV 저장 형태와 분리한다. cache geometry/mutation 이 필요한 offload forward 는 concrete
/// `OffloadKVCache`(KVCacheOps 직접 impl)를 쓰므로 trait 표면에 KVCacheOps 를 끌어올 필요가 없다.
/// 이로써 offload cluster 가 `KVCacheOps` 폐기((4)) 차단자에서 빠진다(SSOT §9.1 B-3).
pub trait PrefetchableCache {
    /// Pre-load data from external storage into memory buffers.
    fn preload(&mut self) -> Result<()>;

    /// Release memory buffers to free RAM (only 2 layers active at once).
    fn release_buffers(&mut self);

    /// Reset preloaded flag at token boundary.
    fn reset_preload(&mut self);

    /// Re-arm preloaded state for cross-token buffer retention.
    ///
    /// Call after `get_view()` on layers whose buffers should survive the token
    /// boundary. The next token's `update()` dual-writes into the still-valid
    /// attn buffers, so `preload()` becomes a no-op (early-return on
    /// `self.preloaded == true`).
    fn retain_preload(&mut self) {}
}

pub struct KVCache {
    pub k_buffer: Tensor,
    pub v_buffer: Tensor,
    pub current_pos: usize,
    /// High-water mark: the maximum `current_pos` ever reached.
    ///
    /// Invariant: `current_pos <= high_water_pos <= capacity`.
    /// Used by `release_unused_pages()` to limit madvise to the
    /// region that was actually written, avoiding spurious page faults
    /// in the never-touched tail of the buffer.
    pub high_water_pos: usize,
    pub max_seq_len: usize,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
    pub(crate) layout: KVLayout,
    memory: Option<Arc<dyn Memory>>,
}

impl KVCache {
    /// Create a KVCache with full pre-allocation (capacity = max_seq_len).
    /// Growth is disabled. Layout defaults to SeqMajor for backward compatibility.
    pub fn new(k: Tensor, v: Tensor, max_seq_len: usize) -> Self {
        let shape = k.shape().dims();
        let kv_heads = shape[2];
        let head_dim = shape[3];
        Self {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            high_water_pos: 0,
            max_seq_len,
            capacity: max_seq_len,
            kv_heads,
            head_dim,
            layout: KVLayout::SeqMajor,
            memory: None,
        }
    }

    /// Create a KVCache with dynamic grow-on-demand allocation.
    /// Starts with `initial_capacity` and doubles up to `max_seq_len`.
    /// Layout defaults to SeqMajor for backward compatibility.
    pub fn new_dynamic(
        k: Tensor,
        v: Tensor,
        initial_capacity: usize,
        max_seq_len: usize,
        kv_heads: usize,
        head_dim: usize,
        memory: Arc<dyn Memory>,
    ) -> Self {
        Self {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            high_water_pos: 0,
            max_seq_len,
            capacity: initial_capacity,
            kv_heads,
            head_dim,
            layout: KVLayout::SeqMajor,
            memory: Some(memory),
        }
    }

    /// Current physical buffer capacity in tokens.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn layout(&self) -> KVLayout {
        self.layout
    }

    /// Set the KV cache layout. Must be called before any data is written.
    pub fn with_layout(mut self, layout: KVLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    // ── opaque(.so block-quant) 저장 지원 (floor 패턴 — typed arm 무변) ──

    /// `k_buffer` 가 [`OpaqueBuffer`](`DType` 없는 .so format 저장)면 true. true 인 호출처는
    /// byte-회계/copy/scatter 의 descriptor-keyed opaque arm 을 탄다(typed arm 은 무변).
    pub(crate) fn is_opaque(&self) -> bool {
        self.k_buffer
            .buffer()
            .as_any()
            .downcast_ref::<OpaqueBuffer>()
            .is_some()
    }

    /// opaque 저장의 sidecar [`KVLayoutDesc`]. `is_opaque()` 가 true 인 arm 전용(아니면 panic).
    pub(crate) fn opaque_desc(&self) -> KVLayoutDesc {
        self.k_buffer
            .buffer()
            .as_any()
            .downcast_ref::<OpaqueBuffer>()
            .expect("opaque_desc() is only called on opaque storage (is_opaque guard)")
            .descriptor()
    }

    /// opaque (head, 단일 pos) 당 raw 바이트 = `(head_dim / block_elems) * block_bytes`.
    /// `copy_slice`/`buffer_shift` 가 U8 type_size=1 이므로 count 단위는 byte.
    pub(crate) fn opaque_bytes_per_head(&self) -> usize {
        let desc = self.opaque_desc();
        let be = desc.block_elems as usize;
        let bb = desc
            .block_bytes()
            .expect("opaque is a block-quant descriptor (Dense not allowed)");
        (self.head_dim / be) * bb
    }

    /// Read-only handle to the grow-on-demand allocator, if this cache is dynamic.
    ///
    /// Returns the backend-appropriate `Memory` (host `Galloc` / `OpenCLMemory` /
    /// `CudaMemory`) used internally for `ensure_capacity`. Exposed so a wrapping
    /// `KVCacheFormat` impl (`StandardFormat`) can allocate its cast scratch from the
    /// *same* allocator the cache itself uses — keeping the format⊥hardware seam intact
    /// without threading a second `Memory` through the trait signature. `None` for
    /// fully pre-allocated caches built via `new()`.
    pub(crate) fn memory(&self) -> Option<Arc<dyn Memory>> {
        self.memory.clone()
    }

    /// Element offset for position `pos` of head `head`.
    ///
    /// - SeqMajor: `pos * kv_heads * head_dim + head * head_dim`
    /// - HeadMajor: `head * capacity * head_dim + pos * head_dim`
    #[inline]
    pub fn offset(&self, pos: usize, head: usize) -> usize {
        match self.layout {
            KVLayout::SeqMajor => pos * self.kv_heads * self.head_dim + head * self.head_dim,
            KVLayout::HeadMajor => head * self.capacity * self.head_dim + pos * self.head_dim,
        }
    }

    /// Elements between consecutive positions for the same head.
    #[inline]
    pub fn pos_stride(&self) -> usize {
        match self.layout {
            KVLayout::SeqMajor => self.kv_heads * self.head_dim,
            KVLayout::HeadMajor => self.head_dim,
        }
    }

    /// Block offset for Q4_0 data at (pos, head).
    ///
    /// Returns the block index into a `&[BlockQ4_0]` slice.
    /// `blocks_per_pos` = head_dim / QK4_0 (e.g. 64/32 = 2).
    #[inline]
    pub fn q4_block_offset(&self, pos: usize, head: usize, blocks_per_pos: usize) -> usize {
        match self.layout {
            KVLayout::SeqMajor => (pos * self.kv_heads + head) * blocks_per_pos,
            KVLayout::HeadMajor => (head * self.capacity + pos) * blocks_per_pos,
        }
    }

    /// Elements between consecutive heads for the same position.
    #[inline]
    pub fn head_stride(&self) -> usize {
        match self.layout {
            KVLayout::SeqMajor => self.head_dim,
            KVLayout::HeadMajor => self.capacity * self.head_dim,
        }
    }

    /// Grow the KV cache buffers to at least `min_capacity` tokens.
    /// Uses doubling strategy clamped to `max_seq_len`.
    fn grow(&mut self, min_capacity: usize) -> Result<()> {
        let memory = self.memory.as_ref().ok_or_else(|| {
            anyhow::anyhow!("KVCache::grow() called on a non-dynamic cache (created with new())")
        })?;

        let new_cap = (self.capacity * 2).max(min_capacity).min(self.max_seq_len);
        let dtype = self.k_buffer.dtype();
        let backend = self.k_buffer.backend().clone();

        let n_values = new_cap * self.kv_heads * self.head_dim;
        let buf_size = if self.is_opaque() {
            // opaque: descriptor-keyed raw bytes (G1 bytes_for_elems). typed arm 무변.
            self.opaque_desc()
                .bytes_for_elems(n_values)
                .expect("opaque KV grow: block-aligned accounting")
        } else {
            match dtype {
                crate::buffer::DType::Q4_0 => {
                    (n_values / crate::quant::QK4_0)
                        * std::mem::size_of::<crate::quant::BlockQ4_0>()
                }
                _ => n_values * dtype.size(),
            }
        };

        let new_shape = match self.layout {
            KVLayout::SeqMajor => Shape::new(vec![1, new_cap, self.kv_heads, self.head_dim]),
            KVLayout::HeadMajor => Shape::new(vec![1, self.kv_heads, new_cap, self.head_dim]),
        };

        // Allocate new buffers (alloc_kv for madvise-capable path). opaque 는 새 U8 버퍼를
        // OpaqueBuffer 로 재포장해 sidecar descriptor 를 보존(미포장 시 desc 소실 → garbage).
        let wrap = |buf: Arc<dyn Buffer>| -> Arc<dyn Buffer> {
            if self.is_opaque() {
                Arc::new(OpaqueBuffer::new(buf, self.opaque_desc()))
            } else {
                buf
            }
        };
        let new_k_buf = wrap(memory.alloc_kv(buf_size, dtype)?);
        let mut new_k = Tensor::new(new_shape.clone(), new_k_buf, backend.clone());
        let new_v_buf = wrap(memory.alloc_kv(buf_size, dtype)?);
        let mut new_v = Tensor::new(new_shape, new_v_buf, backend.clone());

        // Copy existing data
        if self.current_pos > 0 {
            match self.layout {
                KVLayout::SeqMajor => {
                    if self.is_opaque() {
                        anyhow::bail!("opaque KV grow: SeqMajor unsupported (HeadMajor only)");
                    }
                    // Contiguous: all positions × heads × dim
                    let copy_count = if dtype == crate::buffer::DType::Q4_0 {
                        let blocks_per_pos = self.kv_heads * self.head_dim / crate::quant::QK4_0;
                        self.current_pos * blocks_per_pos
                    } else {
                        self.current_pos * self.kv_heads * self.head_dim
                    };
                    backend.copy_slice(&self.k_buffer, &mut new_k, 0, 0, copy_count)?;
                    backend.copy_slice(&self.v_buffer, &mut new_v, 0, 0, copy_count)?;
                }
                KVLayout::HeadMajor => {
                    if self.is_opaque() {
                        // opaque per-head byte copy (U8 type_size=1, byte stride = cap*bytes_per_head).
                        let bph = self.opaque_bytes_per_head();
                        let old_hs = self.capacity * bph;
                        let new_hs = new_cap * bph;
                        let cph = self.current_pos * bph;
                        for h in 0..self.kv_heads {
                            backend.copy_slice(
                                &self.k_buffer,
                                &mut new_k,
                                h * old_hs,
                                h * new_hs,
                                cph,
                            )?;
                            backend.copy_slice(
                                &self.v_buffer,
                                &mut new_v,
                                h * old_hs,
                                h * new_hs,
                                cph,
                            )?;
                        }
                        log::info!(
                            "[KVCache] Growing capacity: {} → {} tokens (opaque)",
                            self.capacity,
                            new_cap
                        );
                        self.k_buffer = new_k;
                        self.v_buffer = new_v;
                        self.capacity = new_cap;
                        return Ok(());
                    }
                    // Per-head copy: capacity changes head_stride
                    let old_head_stride = self.capacity * self.head_dim;
                    let new_head_stride = new_cap * self.head_dim;
                    let copy_per_head = self.current_pos * self.head_dim;

                    if dtype == crate::buffer::DType::Q4_0 {
                        let qk = crate::quant::QK4_0;
                        let old_hs = old_head_stride / qk;
                        let new_hs = new_head_stride / qk;
                        let cph = copy_per_head / qk;
                        for h in 0..self.kv_heads {
                            backend.copy_slice(
                                &self.k_buffer,
                                &mut new_k,
                                h * old_hs,
                                h * new_hs,
                                cph,
                            )?;
                            backend.copy_slice(
                                &self.v_buffer,
                                &mut new_v,
                                h * old_hs,
                                h * new_hs,
                                cph,
                            )?;
                        }
                    } else {
                        for h in 0..self.kv_heads {
                            backend.copy_slice(
                                &self.k_buffer,
                                &mut new_k,
                                h * old_head_stride,
                                h * new_head_stride,
                                copy_per_head,
                            )?;
                            backend.copy_slice(
                                &self.v_buffer,
                                &mut new_v,
                                h * old_head_stride,
                                h * new_head_stride,
                                copy_per_head,
                            )?;
                        }
                    }
                }
            }
        }

        log::info!(
            "[KVCache] Growing capacity: {} → {} tokens",
            self.capacity,
            new_cap
        );

        self.k_buffer = new_k;
        self.v_buffer = new_v;
        self.capacity = new_cap;

        Ok(())
    }

    /// Shrink the KV cache buffers to fit `current_pos`, releasing excess memory.
    /// This is the inverse of `grow()` — reallocates smaller buffers and copies data.
    /// Returns the number of bytes freed.
    pub fn shrink_to_fit(&mut self) -> Result<usize> {
        // opaque(.so block-quant) shrink: descriptor-keyed byte 회계.
        if self.is_opaque() {
            return self.shrink_to_fit_opaque();
        }
        let memory = match self.memory.as_ref() {
            Some(m) => m,
            None => return Ok(0),
        };

        // Compute target capacity: 1.5x current_pos, 64-aligned, minimum 64.
        // Using 1.5x headroom (instead of next_power_of_two) avoids edge cases where
        // e.g. current_pos=2049 rounds up to 4096 and prevents any shrink.
        let new_cap = ((self.current_pos * 3 / 2) + 63) & !63;
        let new_cap = new_cap.max(64);
        if new_cap >= self.capacity {
            return Ok(0);
        }

        let dtype = self.k_buffer.dtype();
        let backend = self.k_buffer.backend().clone();

        let n_values = new_cap * self.kv_heads * self.head_dim;
        let buf_size = match dtype {
            crate::buffer::DType::Q4_0 => {
                (n_values / crate::quant::QK4_0) * std::mem::size_of::<crate::quant::BlockQ4_0>()
            }
            _ => n_values * dtype.size(),
        };

        let old_n_values = self.capacity * self.kv_heads * self.head_dim;
        let old_buf_size = match dtype {
            crate::buffer::DType::Q4_0 => {
                (old_n_values / crate::quant::QK4_0)
                    * std::mem::size_of::<crate::quant::BlockQ4_0>()
            }
            _ => old_n_values * dtype.size(),
        };

        let new_shape = match self.layout {
            KVLayout::SeqMajor => Shape::new(vec![1, new_cap, self.kv_heads, self.head_dim]),
            KVLayout::HeadMajor => Shape::new(vec![1, self.kv_heads, new_cap, self.head_dim]),
        };

        let new_k_buf = memory.alloc_kv(buf_size, dtype)?;
        let mut new_k = Tensor::new(new_shape.clone(), new_k_buf, backend.clone());
        let new_v_buf = memory.alloc_kv(buf_size, dtype)?;
        let mut new_v = Tensor::new(new_shape, new_v_buf, backend.clone());

        // Copy existing data (same logic as grow()). **Q4_0: copy_slice 의 count/offset 은
        // element 가 아니라 block 단위다** (type_size = sizeof(BlockQ4_0) = 18B). element 로
        // 넘기면 ~QK4_0 배 over-read → OOB(SIGSEGV). grow() 와 동일하게 변환한다.
        if self.current_pos > 0 {
            match self.layout {
                KVLayout::SeqMajor => {
                    let copy_count = if dtype == crate::buffer::DType::Q4_0 {
                        let blocks_per_pos = self.kv_heads * self.head_dim / crate::quant::QK4_0;
                        self.current_pos * blocks_per_pos
                    } else {
                        self.current_pos * self.kv_heads * self.head_dim
                    };
                    backend.copy_slice(&self.k_buffer, &mut new_k, 0, 0, copy_count)?;
                    backend.copy_slice(&self.v_buffer, &mut new_v, 0, 0, copy_count)?;
                }
                KVLayout::HeadMajor => {
                    let old_head_stride = self.capacity * self.head_dim;
                    let new_head_stride = new_cap * self.head_dim;
                    let copy_per_head = self.current_pos * self.head_dim;
                    if dtype == crate::buffer::DType::Q4_0 {
                        let qk = crate::quant::QK4_0;
                        let old_hs = old_head_stride / qk;
                        let new_hs = new_head_stride / qk;
                        let cph = copy_per_head / qk;
                        for h in 0..self.kv_heads {
                            backend.copy_slice(
                                &self.k_buffer,
                                &mut new_k,
                                h * old_hs,
                                h * new_hs,
                                cph,
                            )?;
                            backend.copy_slice(
                                &self.v_buffer,
                                &mut new_v,
                                h * old_hs,
                                h * new_hs,
                                cph,
                            )?;
                        }
                    } else {
                        for h in 0..self.kv_heads {
                            backend.copy_slice(
                                &self.k_buffer,
                                &mut new_k,
                                h * old_head_stride,
                                h * new_head_stride,
                                copy_per_head,
                            )?;
                            backend.copy_slice(
                                &self.v_buffer,
                                &mut new_v,
                                h * old_head_stride,
                                h * new_head_stride,
                                copy_per_head,
                            )?;
                        }
                    }
                }
            }
        }

        let freed = old_buf_size.saturating_sub(buf_size) * 2; // K + V
        // steady-state eviction 에서 디코드 토큰마다 발화하므로 `log::debug!` 로 강등한다
        // (RUST_LOG 로 제어 — 데모/공개 저장소 출력 청결, per-token stderr 도배 제거).
        log::debug!(
            "[KVCache] Shrunk capacity: {} → {} tokens (freed {} bytes)",
            self.capacity,
            new_cap,
            freed
        );

        self.k_buffer = new_k;
        self.v_buffer = new_v;
        self.capacity = new_cap;
        self.high_water_pos = self.current_pos;

        Ok(freed)
    }

    /// opaque(.so block-quant) `shrink_to_fit` — grow 의 opaque arm 과 대칭(descriptor-keyed byte
    /// 회계 + HeadMajor per-head byte copy, U8 type_size=1). HeadMajor 전용(SeqMajor 는 no-op). (D2)
    fn shrink_to_fit_opaque(&mut self) -> Result<usize> {
        let memory = match self.memory.as_ref() {
            Some(m) => m.clone(),
            None => return Ok(0),
        };
        if self.layout != KVLayout::HeadMajor {
            return Ok(0); // SeqMajor opaque 미지원 — 보수적 no-op
        }
        let new_cap = ((self.current_pos * 3 / 2) + 63) & !63;
        let new_cap = new_cap.max(64);
        if new_cap >= self.capacity {
            return Ok(0);
        }
        let backend = self.k_buffer.backend().clone();
        let desc = self.opaque_desc();
        let bph = self.opaque_bytes_per_head(); // bytes per (head, pos)
        let n_values = new_cap * self.kv_heads * self.head_dim;
        let buf_size = desc
            .bytes_for_elems(n_values)
            .expect("opaque shrink: block-aligned accounting");
        let old_n_values = self.capacity * self.kv_heads * self.head_dim;
        let old_buf_size = desc
            .bytes_for_elems(old_n_values)
            .expect("opaque shrink: block-aligned accounting");

        let new_shape = Shape::new(vec![1, self.kv_heads, new_cap, self.head_dim]);
        let mk = || -> Result<Tensor> {
            let inner = memory.alloc_kv(buf_size, DType::U8)?;
            let op: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner, desc));
            Ok(Tensor::new(new_shape.clone(), op, backend.clone()))
        };
        let mut new_k = mk()?;
        let mut new_v = mk()?;

        if self.current_pos > 0 {
            let old_hs = self.capacity * bph;
            let new_hs = new_cap * bph;
            let cph = self.current_pos * bph;
            for h in 0..self.kv_heads {
                backend.copy_slice(&self.k_buffer, &mut new_k, h * old_hs, h * new_hs, cph)?;
                backend.copy_slice(&self.v_buffer, &mut new_v, h * old_hs, h * new_hs, cph)?;
            }
        }
        let freed = old_buf_size.saturating_sub(buf_size) * 2; // K + V
        self.k_buffer = new_k;
        self.v_buffer = new_v;
        self.capacity = new_cap;
        self.high_water_pos = self.current_pos;
        Ok(freed)
    }

    /// Append new K/V data to the cache.
    ///
    /// Input tensors are always seq-major `[batch, seq_len, kv_heads, head_dim]`
    /// (from the model's matmul output). For HeadMajor layout, this scatters
    /// per-head data to the correct positions.
    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let seq_len = new_k.shape().dims()[1];

        if self.current_pos + seq_len > self.capacity {
            if self.current_pos + seq_len > self.max_seq_len {
                return Err(anyhow::anyhow!("KV Cache overflow"));
            }
            self.grow(self.current_pos + seq_len)?;
        }

        // opaque(.so block-quant) write: cast 없이 F32 입력을 encode_via_descriptor 로 quant 후
        // HeadMajor byte offset 에 scatter (구 OpaqueKvFormat::scatter_token 흡수).
        if self.is_opaque() {
            return self.update_opaque(new_k, new_v);
        }

        let backend = self.k_buffer.backend().clone();
        let is_q4 = self.k_buffer.dtype() == crate::buffer::DType::Q4_0;

        match self.layout {
            KVLayout::SeqMajor => {
                // Contiguous: single copy per buffer
                let (offset, count) = if is_q4 {
                    let bpp = self.kv_heads * self.head_dim / crate::quant::QK4_0;
                    (self.current_pos * bpp, seq_len * bpp)
                } else {
                    let row = self.kv_heads * self.head_dim;
                    (self.current_pos * row, seq_len * row)
                };
                backend.copy_slice(new_k, &mut self.k_buffer, 0, offset, count)?;
                backend.copy_slice(new_v, &mut self.v_buffer, 0, offset, count)?;
            }
            KVLayout::HeadMajor => {
                // Input is seq-major: [batch, seq_len, kv_heads, head_dim]
                // Scatter to head-major: each head's data at head * capacity * head_dim + pos * head_dim
                if is_q4 {
                    let qk = crate::quant::QK4_0;
                    let src_row = self.kv_heads * self.head_dim / qk;
                    let dst_head_stride = self.capacity * self.head_dim / qk;
                    let blocks_per_head = self.head_dim / qk;
                    let dst_pos_blocks = self.current_pos * blocks_per_head;

                    for s in 0..seq_len {
                        for h in 0..self.kv_heads {
                            let src_off = s * src_row + h * blocks_per_head;
                            let dst_off =
                                h * dst_head_stride + dst_pos_blocks + s * blocks_per_head;
                            backend.copy_slice(
                                new_k,
                                &mut self.k_buffer,
                                src_off,
                                dst_off,
                                blocks_per_head,
                            )?;
                            backend.copy_slice(
                                new_v,
                                &mut self.v_buffer,
                                src_off,
                                dst_off,
                                blocks_per_head,
                            )?;
                        }
                    }
                } else {
                    // Direct memcpy for CPU buffers (avoids Backend::copy_slice overhead).
                    // For OpenCL buffers (as_ptr may be null or host-mapped),
                    // fall back to backend.copy_slice which handles GPU↔GPU copy.
                    let type_size = match self.k_buffer.dtype() {
                        crate::buffer::DType::F16 => 2,
                        crate::buffer::DType::F32 => 4,
                        _ => 0,
                    };

                    // Direct CPU memcpy is faster than GPU enqueue_copy_buffer for
                    // tiny copies (128 bytes per head). On ARM UMA (CL_MEM_ALLOC_HOST_PTR),
                    // host pointers directly access GPU memory without DMA transfer.
                    //
                    // Why: discrete-GPU CudaBuffer (cuMemAllocManaged) returns a non-null
                    // pointer that *looks* host-addressable but actually points to
                    // GPU-resident pages. Direct host memcpy bypasses driver migration
                    // and reads/writes stale data, corrupting the KV cache (PPL explodes).
                    // Force backend.copy_slice for GPU buffers so the backend's
                    // device-to-device path runs.
                    let k_dst = self.k_buffer.as_mut_ptr();
                    let k_src = new_k.as_ptr();
                    let host_addressable =
                        !self.k_buffer.buffer().is_gpu_buffer() && !new_k.buffer().is_gpu_buffer();
                    let can_direct_copy =
                        type_size > 0 && !k_dst.is_null() && !k_src.is_null() && host_addressable;

                    let src_row = self.kv_heads * self.head_dim;
                    let dst_head_stride = self.capacity * self.head_dim;

                    if can_direct_copy {
                        let bytes_per_head = self.head_dim * type_size;
                        let v_src = new_v.as_ptr();
                        let v_dst = self.v_buffer.as_mut_ptr();

                        unsafe {
                            for s in 0..seq_len {
                                for h in 0..self.kv_heads {
                                    let src_byte = (s * src_row + h * self.head_dim) * type_size;
                                    let dst_byte = (h * dst_head_stride
                                        + (self.current_pos + s) * self.head_dim)
                                        * type_size;
                                    std::ptr::copy_nonoverlapping(
                                        k_src.add(src_byte),
                                        k_dst.add(dst_byte),
                                        bytes_per_head,
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        v_src.add(src_byte),
                                        v_dst.add(dst_byte),
                                        bytes_per_head,
                                    );
                                }
                            }
                        }
                    } else {
                        for s in 0..seq_len {
                            for h in 0..self.kv_heads {
                                let src_off = s * src_row + h * self.head_dim;
                                let dst_off =
                                    h * dst_head_stride + (self.current_pos + s) * self.head_dim;
                                backend.copy_slice(
                                    new_k,
                                    &mut self.k_buffer,
                                    src_off,
                                    dst_off,
                                    self.head_dim,
                                )?;
                                backend.copy_slice(
                                    new_v,
                                    &mut self.v_buffer,
                                    src_off,
                                    dst_off,
                                    self.head_dim,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        self.current_pos += seq_len;
        self.high_water_pos = self.high_water_pos.max(self.current_pos);

        Ok(())
    }

    /// opaque(.so block-quant) write — F32 입력 토큰들을 `encode_via_descriptor` 로 quant 후
    /// HeadMajor block layout 의 byte offset 에 scatter. HeadMajor 전용.
    ///
    /// byte offset(head h, position pos) = `h * capacity * bytes_per_head + pos * bytes_per_head`
    /// 여기서 `bytes_per_head = (head_dim / block_elems) * block_bytes`. 입력 `new_k`/`new_v` 는
    /// seq-major F32 `[batch, seq_len, kv_heads, head_dim]` (`StandardFormat::write_inner` 가
    /// opaque 면 cast 없이 그대로 전달). interior-mut 는 엔진 `Buffer` 관용구(`as_mut_ptr`).
    fn update_opaque(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        if self.layout != KVLayout::HeadMajor {
            anyhow::bail!("opaque KV update: HeadMajor only");
        }
        let seq_len = new_k.shape().dims()[1];
        let desc = self.opaque_desc();
        let bph = self.opaque_bytes_per_head();
        let head_stride = self.capacity * bph;
        let row = self.kv_heads * self.head_dim; // f32 elements per token (input)
        let base_pos = self.current_pos;
        for (buf, src_t) in [(&self.k_buffer, new_k), (&self.v_buffer, new_v)] {
            let src = src_t.as_slice::<f32>();
            let total = buf.buffer().size();
            // SAFETY: opaque 버퍼는 total 바이트 유효(self 수명 동안 Arc 보유). 각 (h,pos) sub-slice 는
            // non-overlapping. 같은 token 의 K/V 는 서로 다른 버퍼라 aliasing 없음.
            let all: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(buf.buffer().as_mut_ptr(), total) };
            for s in 0..seq_len {
                let pos = base_pos + s;
                if pos >= self.capacity {
                    anyhow::bail!("opaque KV update: capacity {} exceeded", self.capacity);
                }
                for h in 0..self.kv_heads {
                    let lo = s * row + h * self.head_dim;
                    let off = h * head_stride + pos * bph;
                    encode_via_descriptor(
                        &desc,
                        &src[lo..lo + self.head_dim],
                        &mut all[off..off + bph],
                    )?;
                }
            }
        }
        self.current_pos += seq_len;
        self.high_water_pos = self.high_water_pos.max(self.current_pos);
        Ok(())
    }

    pub fn get_view(&self, _seq_len: usize) -> (Tensor, Tensor) {
        (self.k_buffer.clone(), self.v_buffer.clone())
    }

    /// Remove the first `count` tokens from the cache, shifting remaining data forward.
    ///
    /// Before: [A][B][C][D][E] (current_pos=5)
    /// prune_prefix(2)
    /// After:  [C][D][E][_][_] (current_pos=3)
    pub fn prune_prefix(&mut self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if count > self.current_pos {
            return Err(anyhow::anyhow!(
                "Cannot prune {} tokens, only {} in cache",
                count,
                self.current_pos
            ));
        }

        let remaining = self.current_pos - count;

        if remaining == 0 {
            self.current_pos = 0;
            self.high_water_pos = 0;
            return Ok(());
        }

        let backend = self.k_buffer.backend().clone();
        let is_q4 = self.k_buffer.dtype() == crate::buffer::DType::Q4_0;

        match self.layout {
            KVLayout::SeqMajor => {
                if self.is_opaque() {
                    anyhow::bail!("opaque KV prune_prefix: SeqMajor unsupported (HeadMajor only)");
                }
                let (src_offset, move_count) = if is_q4 {
                    let bpp = self.kv_heads * self.head_dim / crate::quant::QK4_0;
                    (count * bpp, remaining * bpp)
                } else {
                    let epp = self.kv_heads * self.head_dim;
                    (count * epp, remaining * epp)
                };
                backend.buffer_shift(&mut self.k_buffer, src_offset, 0, move_count)?;
                backend.buffer_shift(&mut self.v_buffer, src_offset, 0, move_count)?;
            }
            KVLayout::HeadMajor => {
                if self.is_opaque() {
                    // opaque per-head byte shift (U8 type_size=1).
                    let bph = self.opaque_bytes_per_head();
                    let head_stride = self.capacity * bph;
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + count * bph,
                            base,
                            remaining * bph,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + count * bph,
                            base,
                            remaining * bph,
                        )?;
                    }
                    self.current_pos = remaining;
                    self.release_unused_pages();
                    return Ok(());
                }
                // Per-head shift: each head's data is contiguous
                if is_q4 {
                    let qk = crate::quant::QK4_0;
                    let head_stride = self.capacity * self.head_dim / qk;
                    let bph = self.head_dim / qk; // blocks per head per position
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + count * bph,
                            base,
                            remaining * bph,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + count * bph,
                            base,
                            remaining * bph,
                        )?;
                    }
                } else {
                    let head_stride = self.capacity * self.head_dim;
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + count * self.head_dim,
                            base,
                            remaining * self.head_dim,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + count * self.head_dim,
                            base,
                            remaining * self.head_dim,
                        )?;
                    }
                }
            }
        }

        self.current_pos = remaining;
        self.release_unused_pages();
        Ok(())
    }

    /// Returns the memory usage in bytes for currently stored KV data.
    pub fn memory_usage_bytes(&self) -> usize {
        if self.is_opaque() {
            // opaque: descriptor-keyed raw bytes (G1 bytes_for_elems).
            let per_buffer = self
                .opaque_desc()
                .bytes_for_elems(self.current_pos * self.kv_heads * self.head_dim)
                .unwrap_or(0);
            return per_buffer * 2; // K + V
        }
        let is_q4 = self.k_buffer.dtype() == crate::buffer::DType::Q4_0;

        let per_buffer = if is_q4 {
            let blocks_per_pos = self.kv_heads * self.head_dim / crate::quant::QK4_0;
            let block_size = std::mem::size_of::<crate::quant::BlockQ4_0>();
            self.current_pos * blocks_per_pos * block_size
        } else {
            let type_size = self.k_buffer.dtype().size();
            self.current_pos * self.kv_heads * self.head_dim * type_size
        };

        per_buffer * 2 // K + V
    }

    /// Layout-aware position shift for eviction policies.
    ///
    /// Moves `count` positions worth of data from `src_pos` to `dst_pos`
    /// within both K and V buffers. Handles per-head scattering for HeadMajor.
    pub fn shift_positions(&mut self, src_pos: usize, dst_pos: usize, count: usize) -> Result<()> {
        if count == 0 || src_pos == dst_pos {
            return Ok(());
        }

        let backend = self.k_buffer.backend().clone();
        let is_q4 = self.k_buffer.dtype() == crate::buffer::DType::Q4_0;

        match self.layout {
            KVLayout::SeqMajor => {
                if self.is_opaque() {
                    anyhow::bail!(
                        "opaque KV shift_positions: SeqMajor unsupported (HeadMajor only)"
                    );
                }
                let (src_off, dst_off, move_count) = if is_q4 {
                    let bpp = self.kv_heads * self.head_dim / crate::quant::QK4_0;
                    (src_pos * bpp, dst_pos * bpp, count * bpp)
                } else {
                    let epp = self.kv_heads * self.head_dim;
                    (src_pos * epp, dst_pos * epp, count * epp)
                };
                backend.buffer_shift(&mut self.k_buffer, src_off, dst_off, move_count)?;
                backend.buffer_shift(&mut self.v_buffer, src_off, dst_off, move_count)?;
            }
            KVLayout::HeadMajor => {
                if self.is_opaque() {
                    // opaque per-head byte shift (U8 type_size=1).
                    let bph = self.opaque_bytes_per_head();
                    let head_stride = self.capacity * bph;
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + src_pos * bph,
                            base + dst_pos * bph,
                            count * bph,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + src_pos * bph,
                            base + dst_pos * bph,
                            count * bph,
                        )?;
                    }
                    return Ok(());
                }
                if is_q4 {
                    let qk = crate::quant::QK4_0;
                    let head_stride = self.capacity * self.head_dim / qk;
                    let bph = self.head_dim / qk;
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + src_pos * bph,
                            base + dst_pos * bph,
                            count * bph,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + src_pos * bph,
                            base + dst_pos * bph,
                            count * bph,
                        )?;
                    }
                } else {
                    let head_stride = self.capacity * self.head_dim;
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + src_pos * self.head_dim,
                            base + dst_pos * self.head_dim,
                            count * self.head_dim,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + src_pos * self.head_dim,
                            base + dst_pos * self.head_dim,
                            count * self.head_dim,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Move `count` positions from `src_pos` to `dst_pos` for a **single** KV head.
    ///
    /// Unlike `shift_positions()` which moves the same positions for all heads,
    /// this method only modifies data belonging to the specified `head`.
    /// Used by per-head eviction policies (e.g., H2O+) where each KV head
    /// independently selects which tokens to keep.
    ///
    /// **Requires HeadMajor layout** — panics on SeqMajor because per-head
    /// data is interleaved and cannot be moved independently.
    pub fn shift_positions_for_head(
        &mut self,
        head: usize,
        src_pos: usize,
        dst_pos: usize,
        count: usize,
    ) -> Result<()> {
        assert_eq!(
            self.layout,
            KVLayout::HeadMajor,
            "shift_positions_for_head requires HeadMajor layout"
        );

        if count == 0 || src_pos == dst_pos {
            return Ok(());
        }

        let backend = self.k_buffer.backend().clone();
        let is_q4 = self.k_buffer.dtype() == crate::buffer::DType::Q4_0;

        if self.is_opaque() {
            // opaque per-head byte shift (U8 type_size=1).
            let bph = self.opaque_bytes_per_head();
            let head_stride = self.capacity * bph;
            let base = head * head_stride;
            backend.buffer_shift(
                &mut self.k_buffer,
                base + src_pos * bph,
                base + dst_pos * bph,
                count * bph,
            )?;
            backend.buffer_shift(
                &mut self.v_buffer,
                base + src_pos * bph,
                base + dst_pos * bph,
                count * bph,
            )?;
            return Ok(());
        }

        if is_q4 {
            let qk = crate::quant::QK4_0;
            let head_stride = self.capacity * self.head_dim / qk;
            let bph = self.head_dim / qk;
            let base = head * head_stride;
            backend.buffer_shift(
                &mut self.k_buffer,
                base + src_pos * bph,
                base + dst_pos * bph,
                count * bph,
            )?;
            backend.buffer_shift(
                &mut self.v_buffer,
                base + src_pos * bph,
                base + dst_pos * bph,
                count * bph,
            )?;
        } else {
            let head_stride = self.capacity * self.head_dim;
            let base = head * head_stride;
            backend.buffer_shift(
                &mut self.k_buffer,
                base + src_pos * self.head_dim,
                base + dst_pos * self.head_dim,
                count * self.head_dim,
            )?;
            backend.buffer_shift(
                &mut self.v_buffer,
                base + src_pos * self.head_dim,
                base + dst_pos * self.head_dim,
                count * self.head_dim,
            )?;
        }

        Ok(())
    }

    /// Release physical pages for the unused portion of KV buffers via `madvise(MADV_DONTNEED)`.
    ///
    /// After eviction reduces `current_pos`, the buffer region beyond `current_pos`
    /// is no longer needed. This method advises the OS to reclaim those physical pages,
    /// reducing RSS and increasing MemAvailable — without reallocating the buffer.
    ///
    /// Only applies to CPU heap buffers (`SharedBuffer` / `Vec<u8>`).
    /// GPU-managed buffers (`UnifiedBuffer` with `CL_MEM_ALLOC_HOST_PTR`) are skipped
    /// because their physical pages are pinned by the OpenCL driver.
    ///
    /// Returns the total bytes released (page-aligned) across K and V buffers.
    pub fn release_unused_pages(&mut self) -> usize {
        // Shrink buffers when significantly underutilized (works for any buffer type).
        // Reallocates smaller buffers, guaranteeing physical memory release.
        // Threshold: 75% — e.g. H2O keep_ratio=0.5 evicts to capacity/2, which now triggers shrink.
        if self.memory.is_some() && self.current_pos * 4 < self.capacity * 3 {
            return self.shrink_to_fit().unwrap_or_else(|e| {
                eprintln!("[KVCache] shrink_to_fit failed: {}", e);
                0
            });
        }

        // madvise path: only for host-managed buffers (SharedBuffer, MmapBuffer).
        // GPU-managed buffers (UnifiedBuffer with CL_MEM_ALLOC_HOST_PTR) are driver-pinned.
        if !self.k_buffer.buffer().is_host_managed() {
            return 0;
        }

        let type_size = match self.k_buffer.dtype() {
            DType::F16 => 2,
            DType::F32 => 4,
            _ => return 0,
        };

        let hwm = self.high_water_pos;
        let mut total_released = 0usize;

        match self.layout {
            KVLayout::SeqMajor => {
                // Contiguous: [pos0_all_heads | pos1_all_heads | ... | posN_all_heads]
                // Only advise up to high_water_pos — never-touched tail pages are already free.
                let row_bytes = self.kv_heads * self.head_dim * type_size;
                let used_bytes = self.current_pos * row_bytes;
                let hwm_bytes = hwm * row_bytes;
                total_released += madvise_dontneed(self.k_buffer.as_ptr(), used_bytes, hwm_bytes);
                total_released += madvise_dontneed(self.v_buffer.as_ptr(), used_bytes, hwm_bytes);
            }
            KVLayout::HeadMajor => {
                // Per-head: [head0: pos0..posN | head1: pos0..posN | ...]
                let head_stride_bytes = self.capacity * self.head_dim * type_size;
                let used_per_head_bytes = self.current_pos * self.head_dim * type_size;
                let hwm_per_head_bytes = hwm * self.head_dim * type_size;
                for h in 0..self.kv_heads {
                    let base = h * head_stride_bytes;
                    let from = base + used_per_head_bytes;
                    let to = base + hwm_per_head_bytes;
                    total_released += madvise_dontneed(self.k_buffer.as_ptr(), from, to);
                    total_released += madvise_dontneed(self.v_buffer.as_ptr(), from, to);
                }
            }
        }

        if total_released > 0 {
            log::debug!(
                "[KVCache] released {} bytes of unused pages (pos={}, hwm={}, cap={})",
                total_released,
                self.current_pos,
                hwm,
                self.capacity,
            );
        }

        // After releasing, high_water_pos shrinks to current_pos.
        // Pages between current_pos and old hwm have been advised away.
        self.high_water_pos = self.current_pos;

        total_released
    }

    /// Compact the KV cache by moving `keep` positions to a contiguous region
    /// starting at `write_start`.
    ///
    /// `keep` must be sorted in ascending order. Consecutive source positions are
    /// merged into a single `shift_positions` call, reducing the number of
    /// `buffer_shift` invocations by up to 200x compared to per-token shifting.
    ///
    /// This method does **not** update `current_pos`; the caller is responsible
    /// for setting it to `write_start + keep.len()` after compaction.
    pub fn compact_keep_positions(&mut self, keep: &[usize], write_start: usize) -> Result<()> {
        if keep.is_empty() {
            return Ok(());
        }

        let mut write_pos = write_start;
        let mut batch_src_start = keep[0];
        let mut batch_dst_start = write_pos;
        let mut batch_count = 1usize;
        write_pos += 1;

        for &src_pos in &keep[1..] {
            if src_pos == batch_src_start + batch_count {
                // Extend current batch
                batch_count += 1;
            } else {
                // Flush current batch
                if batch_src_start != batch_dst_start {
                    self.shift_positions(batch_src_start, batch_dst_start, batch_count)?;
                }
                // Start new batch
                batch_src_start = src_pos;
                batch_dst_start = write_pos;
                batch_count = 1;
            }
            write_pos += 1;
        }

        // Flush final batch
        if batch_src_start != batch_dst_start {
            self.shift_positions(batch_src_start, batch_dst_start, batch_count)?;
        }

        Ok(())
    }

    /// Per-head variant of `compact_keep_positions` for H2O+ per-head eviction.
    ///
    /// Only moves data for the specified `head`; other heads are untouched.
    /// `keep` must be sorted in ascending order.
    ///
    /// **Requires HeadMajor layout** — delegates to `shift_positions_for_head`.
    pub fn compact_keep_positions_for_head(
        &mut self,
        head: usize,
        keep: &[usize],
        write_start: usize,
    ) -> Result<()> {
        if keep.is_empty() {
            return Ok(());
        }

        let mut write_pos = write_start;
        let mut batch_src_start = keep[0];
        let mut batch_dst_start = write_pos;
        let mut batch_count = 1usize;
        write_pos += 1;

        for &src_pos in &keep[1..] {
            if src_pos == batch_src_start + batch_count {
                // Extend current batch
                batch_count += 1;
            } else {
                // Flush current batch
                if batch_src_start != batch_dst_start {
                    self.shift_positions_for_head(
                        head,
                        batch_src_start,
                        batch_dst_start,
                        batch_count,
                    )?;
                }
                // Start new batch
                batch_src_start = src_pos;
                batch_dst_start = write_pos;
                batch_count = 1;
            }
            write_pos += 1;
        }

        // Flush final batch
        if batch_src_start != batch_dst_start {
            self.shift_positions_for_head(head, batch_src_start, batch_dst_start, batch_count)?;
        }

        Ok(())
    }

    // ── KVCacheOps primitives moved to inherent (Phase α-K BC 5-E) ───────────
    // KVCacheOps trait 은 5-F 까지 생존하되, 본문은 inherent 로 이전하고 trait 메서드는
    // 이 inherent 를 위임 호출한다(byte-identical, 단일 본문). fmt 경로(StandardFormat)와
    // 비-forward 소비자(swap_handler/batch)가 trait 경유 없이 직접 호출하도록 한다.
    // (`capacity`/`kv_heads`/`head_dim`/`layout`/`update`/`memory_usage_bytes` 는 이미 inherent.)

    /// Number of valid tokens currently in the cache.
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    /// Override the current position counter (pos==0 도 high_water 리셋).
    pub fn set_current_pos(&mut self, pos: usize) {
        self.current_pos = pos;
        if pos == 0 {
            self.high_water_pos = 0;
        }
    }

    /// The DType that the caller should pass to `update()`.
    pub fn kv_dtype(&self) -> DType {
        self.k_buffer.dtype()
    }

    /// K/V tensors for attention covering `[0..current_pos]`.
    ///
    /// 기존 inherent `get_view(&self, seq_len)`(:534)와 시그니처가 달라 동명 정의가 불가하므로
    /// `view` 로 신설한다(Phase α-K BC 5-E R3). trait `get_view(&mut self)` 가 이를 위임한다.
    pub fn view(&mut self) -> (Tensor, Tensor) {
        (self.k_buffer.clone(), self.v_buffer.clone())
    }

    /// Direct access to underlying K/V buffers for zero-overhead scatter writes.
    pub fn get_buffers_mut(&mut self) -> Option<(&mut Tensor, &mut Tensor)> {
        Some((&mut self.k_buffer, &mut self.v_buffer))
    }

    /// Advance position counter without performing any data copy.
    pub fn advance_pos(&mut self, n: usize) {
        self.current_pos += n;
        self.high_water_pos = self.high_water_pos.max(self.current_pos);
    }

    /// Ensure the cache has capacity for at least `min_tokens` total tokens.
    pub fn ensure_capacity(&mut self, min_tokens: usize) -> Result<bool> {
        if min_tokens <= self.capacity {
            return Ok(false);
        }
        if min_tokens > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "KV Cache overflow: need {} tokens but max_seq_len={}",
                min_tokens,
                self.max_seq_len
            ));
        }
        self.grow(min_tokens)?;
        Ok(true)
    }
}

// ── madvise helpers ─────────────────────────────────────────────────────────

/// Advise the OS to release physical pages in `[from_offset..to_offset)` relative to `base_ptr`.
///
/// Addresses are rounded to page boundaries (start up, end down) to satisfy madvise requirements.
/// Returns the number of bytes actually advised for release.
fn madvise_dontneed(base_ptr: *const u8, from_offset: usize, to_offset: usize) -> usize {
    if from_offset >= to_offset || base_ptr.is_null() {
        return 0;
    }

    let page_size = page_size();
    let abs_start = base_ptr as usize + from_offset;
    let abs_end = base_ptr as usize + to_offset;

    // Round start UP (don't release partially-used page), end DOWN
    let aligned_start = round_up(abs_start, page_size);
    let aligned_end = round_down(abs_end, page_size);

    if aligned_start >= aligned_end {
        return 0;
    }

    let len = aligned_end - aligned_start;
    // SAFETY: The range [aligned_start..aligned_end) lies within the buffer's allocation.
    // MADV_DONTNEED on anonymous private mappings releases physical pages; re-access
    // triggers zero-fill page faults (safe for KV cache — overwritten before read).
    let ret =
        unsafe { libc::madvise(aligned_start as *mut libc::c_void, len, libc::MADV_DONTNEED) };
    if ret == 0 { len } else { 0 }
}

#[inline]
fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

#[inline]
fn round_down(x: usize, align: usize) -> usize {
    x & !(align - 1)
}

/// Cached page size (4096 on most ARM64/x86_64 systems).
fn page_size() -> usize {
    static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        // SAFETY: sysconf(_SC_PAGESIZE) is always safe and returns a positive value on Linux.
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if ps > 0 { ps as usize } else { 4096 }
    })
}

/// Get the maximum current_pos across all KV caches.
/// Returns 0 if the slice is empty.
#[inline]
pub fn max_cache_pos(caches: &[KVCache]) -> usize {
    caches.iter().map(|c| c.current_pos).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::memory::galloc::Galloc;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use std::sync::Arc;

    fn make_cache(max_seq_len: usize, heads: usize, dim: usize) -> KVCache {
        let size_bytes = max_seq_len * heads * dim * 4; // F32
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let backend = Arc::new(CpuBackend::new());

        let k = Tensor::new(
            Shape::new(vec![1, max_seq_len, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq_len, heads, dim]),
            v_buf,
            backend.clone(),
        );
        KVCache::new(k, v, max_seq_len)
    }

    fn make_dynamic_cache(
        initial_capacity: usize,
        max_seq_len: usize,
        heads: usize,
        dim: usize,
    ) -> KVCache {
        let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
        let backend = Arc::new(CpuBackend::new());

        let size_bytes = initial_capacity * heads * dim * 4;
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));

        let k = Tensor::new(
            Shape::new(vec![1, initial_capacity, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, initial_capacity, heads, dim]),
            v_buf,
            backend.clone(),
        );
        KVCache::new_dynamic(k, v, initial_capacity, max_seq_len, heads, dim, memory)
    }

    fn make_token_tensor(value: f32, heads: usize, dim: usize) -> (Tensor, Tensor) {
        let backend = Arc::new(CpuBackend::new());
        let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut f32;
            for j in 0..(heads * dim) {
                *ptr.add(j) = value;
            }
        }
        let k = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
        let vbuf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
        unsafe {
            let ptr = vbuf.as_mut_ptr() as *mut f32;
            for j in 0..(heads * dim) {
                *ptr.add(j) = value * 10.0;
            }
        }
        let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), vbuf, backend);
        (k, v)
    }

    #[test]
    fn test_prune_prefix_basic() {
        let mut cache = make_cache(100, 1, 4);

        let backend = Arc::new(CpuBackend::new());
        for i in 0..10 {
            let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let val = (i + 1) as f32;
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for j in 0..4 {
                    *ptr.add(j) = val;
                }
            }
            let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
            let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            unsafe {
                let ptr = vbuf.as_mut_ptr() as *mut f32;
                for j in 0..4 {
                    *ptr.add(j) = val * 10.0;
                }
            }
            let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
            cache.update(&t, &vt).unwrap();
        }
        assert_eq!(cache.current_pos, 10);

        cache.prune_prefix(3).unwrap();
        assert_eq!(cache.current_pos, 7);

        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 4.0);
        assert_eq!(k_data[4], 5.0);

        let v_data = cache.v_buffer.as_slice::<f32>();
        assert_eq!(v_data[0], 40.0);
        assert_eq!(v_data[4], 50.0);
    }

    #[test]
    fn test_prune_prefix_zero() {
        let mut cache = make_cache(100, 1, 4);
        cache.current_pos = 5;
        cache.prune_prefix(0).unwrap();
        assert_eq!(cache.current_pos, 5);
    }

    #[test]
    fn test_prune_prefix_over_count() {
        let mut cache = make_cache(100, 1, 4);
        cache.current_pos = 5;
        assert!(cache.prune_prefix(10).is_err());
    }

    #[test]
    fn test_prune_prefix_all() {
        let mut cache = make_cache(100, 1, 4);
        cache.current_pos = 5;
        cache.prune_prefix(5).unwrap();
        assert_eq!(cache.current_pos, 0);
    }

    #[test]
    fn test_memory_usage_bytes() {
        let cache = make_cache(100, 2, 64);
        assert_eq!(cache.memory_usage_bytes(), 0);

        let mut cache = make_cache(100, 2, 64);
        cache.current_pos = 10;
        assert_eq!(cache.memory_usage_bytes(), 10 * 2 * 64 * 4 * 2);
    }

    #[test]
    fn test_cache_creation() {
        let cache = make_cache(64, 4, 8);
        assert_eq!(cache.current_pos, 0);
        assert_eq!(cache.max_seq_len, 64);
        assert_eq!(cache.capacity(), 64);
        assert_eq!(cache.k_buffer.shape().dims(), &[1, 64, 4, 8]);
        assert_eq!(cache.v_buffer.shape().dims(), &[1, 64, 4, 8]);
    }

    #[test]
    fn test_update_overflow() {
        let mut cache = make_cache(4, 1, 4);
        let backend = Arc::new(CpuBackend::new());

        for _ in 0..4 {
            let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
            let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
            cache.update(&t, &vt).unwrap();
        }
        assert_eq!(cache.current_pos, 4);

        let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
        let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
        let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
        let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
        assert!(cache.update(&t, &vt).is_err());
    }

    #[test]
    fn test_get_view() {
        let mut cache = make_cache(100, 2, 4);
        cache.current_pos = 10;
        let (k_view, v_view) = cache.get_view(10);
        assert_eq!(k_view.shape().dims(), &[1, 100, 2, 4]);
        assert_eq!(v_view.shape().dims(), &[1, 100, 2, 4]);
        assert_eq!(k_view.dtype(), DType::F32);
        assert_eq!(v_view.dtype(), DType::F32);
    }

    #[test]
    fn test_new_backward_compat() {
        let cache = make_cache(128, 4, 64);
        assert_eq!(cache.capacity(), 128);
        assert_eq!(cache.max_seq_len, 128);
        assert!(cache.memory.is_none());
    }

    #[test]
    fn test_dynamic_growth_basic() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 64, heads, dim);

        assert_eq!(cache.capacity(), 4);

        // Insert 4 tokens (fills initial capacity)
        for i in 0..4 {
            let (k, v) = make_token_tensor((i + 1) as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 4);
        assert_eq!(cache.capacity(), 4);

        // Insert 1 more → triggers growth
        let (k, v) = make_token_tensor(5.0, heads, dim);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos, 5);
        assert!(cache.capacity() >= 5);

        // Verify data integrity
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 1.0); // first token
        assert_eq!(k_data[4], 2.0); // second token (offset by dim=4)
        assert_eq!(k_data[16], 5.0); // fifth token
    }

    #[test]
    fn test_dynamic_growth_doubling() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 256, heads, dim);

        // Fill 4 tokens
        for i in 0..4 {
            let (k, v) = make_token_tensor(i as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.capacity(), 4);

        // Insert 1 more → 4 * 2 = 8
        let (k, v) = make_token_tensor(100.0, heads, dim);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.capacity(), 8);

        // Fill up to 8
        for _ in 0..3 {
            let (k, v) = make_token_tensor(0.0, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 8);
        assert_eq!(cache.capacity(), 8);

        // Insert 1 more → 8 * 2 = 16
        let (k, v) = make_token_tensor(0.0, heads, dim);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.capacity(), 16);
    }

    #[test]
    fn test_dynamic_growth_capped() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 6, heads, dim);

        // Fill 4 tokens
        for i in 0..4 {
            let (k, v) = make_token_tensor(i as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }

        // Insert 1 more → min(4*2, 6) = 6
        let (k, v) = make_token_tensor(0.0, heads, dim);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.capacity(), 6);
    }

    #[test]
    fn test_dynamic_overflow() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 6, heads, dim);

        // Fill to max_seq_len
        for i in 0..6 {
            let (k, v) = make_token_tensor(i as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 6);

        // One more should overflow
        let (k, v) = make_token_tensor(0.0, heads, dim);
        assert!(cache.update(&k, &v).is_err());
    }

    #[test]
    fn test_ensure_capacity_grows_when_needed() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 64, heads, dim);

        // Fill to capacity
        for i in 0..4 {
            let (k, v) = make_token_tensor(i as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.capacity(), 4);
        assert_eq!(cache.current_pos(), 4);

        // ensure_capacity within current → no grow, returns false
        assert!(!cache.ensure_capacity(3).unwrap());
        assert_eq!(cache.capacity(), 4);

        // ensure_capacity exactly at boundary → no grow
        assert!(!cache.ensure_capacity(4).unwrap());

        // ensure_capacity beyond → grow, returns true
        assert!(cache.ensure_capacity(5).unwrap());
        assert!(cache.capacity() >= 5);

        // Data integrity after grow
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 0.0);
        assert_eq!(k_data[dim], 1.0);
    }

    #[test]
    fn test_ensure_capacity_overflow() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 8, heads, dim);

        // Beyond max_seq_len → error
        assert!(cache.ensure_capacity(9).is_err());
    }

    #[test]
    fn test_dynamic_with_eviction() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 32, heads, dim);

        // Fill 8 tokens (triggers growth from 4 → 8)
        for i in 0..8 {
            let (k, v) = make_token_tensor((i + 1) as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 8);
        assert_eq!(cache.capacity(), 8);

        // Prune first 4 tokens
        cache.prune_prefix(4).unwrap();
        assert_eq!(cache.current_pos, 4);

        // Verify data shifted correctly: pos 0 should have value 5.0
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 5.0);
        assert_eq!(k_data[4], 6.0);

        // Can insert more without growing (capacity=8, current_pos=4)
        for i in 0..4 {
            let (k, v) = make_token_tensor((100 + i) as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 8);
        assert_eq!(cache.capacity(), 8); // no growth needed
    }

    #[test]
    fn test_non_dynamic_grow_fails() {
        let mut cache = make_cache(4, 1, 4);
        // Fill to capacity
        let backend = Arc::new(CpuBackend::new());
        for _ in 0..4 {
            let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
            let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
            cache.update(&t, &vt).unwrap();
        }
        // Non-dynamic cache: capacity == max_seq_len, so overflow is immediate
        let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
        let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
        let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
        let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
        assert!(cache.update(&t, &vt).is_err());
    }

    // ── Phase 0: Layout accessor tests ──

    #[test]
    fn test_layout_default_is_seq_major() {
        let cache = make_cache(64, 4, 8);
        assert_eq!(cache.layout(), KVLayout::SeqMajor);
    }

    #[test]
    fn test_offset_seq_major() {
        let cache = make_cache(64, 4, 8); // heads=4, dim=8, capacity=64
        // SeqMajor: pos * kv_heads * head_dim + head * head_dim
        assert_eq!(cache.offset(0, 0), 0);
        assert_eq!(cache.offset(0, 1), 8);
        assert_eq!(cache.offset(0, 3), 24);
        assert_eq!(cache.offset(1, 0), 32); // 1 * 4 * 8
        assert_eq!(cache.offset(5, 2), 5 * 32 + 16);
    }

    #[test]
    fn test_offset_head_major() {
        let cache = make_head_major_cache(64, 4, 8);

        // HeadMajor: head * capacity * head_dim + pos * head_dim
        assert_eq!(cache.offset(0, 0), 0);
        assert_eq!(cache.offset(1, 0), 8); // pos=1, head=0 → 0 + 1*8
        assert_eq!(cache.offset(0, 1), 512); // head=1 → 1 * 64 * 8
        assert_eq!(cache.offset(5, 2), 2 * 512 + 5 * 8);
    }

    #[test]
    fn test_strides_seq_major() {
        let cache = make_cache(64, 4, 8);
        assert_eq!(cache.pos_stride(), 32); // kv_heads * head_dim
        assert_eq!(cache.head_stride(), 8); // head_dim
    }

    #[test]
    fn test_strides_head_major() {
        let cache = make_head_major_cache(64, 4, 8);

        assert_eq!(cache.pos_stride(), 8); // head_dim
        assert_eq!(cache.head_stride(), 512); // capacity * head_dim
    }

    #[test]
    fn test_accessors() {
        let cache = make_cache(64, 4, 8);
        assert_eq!(cache.kv_heads(), 4);
        assert_eq!(cache.head_dim(), 8);
    }

    // ── Phase 1: HeadMajor internal operations ──

    fn make_head_major_cache(max_seq_len: usize, heads: usize, dim: usize) -> KVCache {
        let size_bytes = max_seq_len * heads * dim * 4;
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(
            Shape::new(vec![1, heads, max_seq_len, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, heads, max_seq_len, dim]), v_buf, backend);
        KVCache {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            high_water_pos: 0,
            max_seq_len,
            capacity: max_seq_len,
            kv_heads: heads,
            head_dim: dim,
            layout: KVLayout::HeadMajor,
            memory: None,
        }
    }

    fn make_head_major_dynamic_cache(
        initial_capacity: usize,
        max_seq_len: usize,
        heads: usize,
        dim: usize,
    ) -> KVCache {
        let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
        let backend = Arc::new(CpuBackend::new());
        let size_bytes = initial_capacity * heads * dim * 4;
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let k = Tensor::new(
            Shape::new(vec![1, heads, initial_capacity, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, heads, initial_capacity, dim]),
            v_buf,
            backend,
        );
        KVCache {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            high_water_pos: 0,
            max_seq_len,
            capacity: initial_capacity,
            kv_heads: heads,
            head_dim: dim,
            layout: KVLayout::HeadMajor,
            memory: Some(memory),
        }
    }

    /// Read a value from head-major cache at (pos, head, d=0)
    fn hm_read(cache: &KVCache, pos: usize, head: usize) -> f32 {
        let off = cache.offset(pos, head);
        cache.k_buffer.as_slice::<f32>()[off]
    }

    #[test]
    fn test_hm_update_single_token() {
        let heads = 2;
        let dim = 4;
        let mut cache = make_head_major_cache(16, heads, dim);

        // Input: seq-major [1, 1, 2, 4] — head0=[1,1,1,1], head1=[2,2,2,2]
        let backend = Arc::new(CpuBackend::new());
        let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut f32;
            for d in 0..dim {
                *ptr.add(d) = 1.0;
            } // head 0
            for d in 0..dim {
                *ptr.add(dim + d) = 2.0;
            } // head 1
        }
        let k = Tensor::new(
            Shape::new(vec![1, 1, heads, dim]),
            buf.clone(),
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend);
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.current_pos, 1);
        // Head 0, pos 0 should be 1.0
        assert_eq!(hm_read(&cache, 0, 0), 1.0);
        // Head 1, pos 0 should be 2.0
        assert_eq!(hm_read(&cache, 0, 1), 2.0);
    }

    #[test]
    fn test_hm_update_multi_token() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        // 3 tokens: seq-major [1,1,3,2]
        let backend = Arc::new(CpuBackend::new());
        let n = 3 * heads * dim;
        let buf = Arc::new(SharedBuffer::new(n * 4, DType::F32));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut f32;
            // Token 0: h0=[10,11], h1=[20,21]
            *ptr.add(0) = 10.0;
            *ptr.add(1) = 11.0;
            *ptr.add(2) = 20.0;
            *ptr.add(3) = 21.0;
            // Token 1: h0=[30,31], h1=[40,41]
            *ptr.add(4) = 30.0;
            *ptr.add(5) = 31.0;
            *ptr.add(6) = 40.0;
            *ptr.add(7) = 41.0;
            // Token 2: h0=[50,51], h1=[60,61]
            *ptr.add(8) = 50.0;
            *ptr.add(9) = 51.0;
            *ptr.add(10) = 60.0;
            *ptr.add(11) = 61.0;
        }
        let k = Tensor::new(
            Shape::new(vec![1, 3, heads, dim]),
            buf.clone(),
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, 3, heads, dim]), buf, backend);
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.current_pos, 3);
        // Head 0: [10, 30, 50] at positions 0, 1, 2
        assert_eq!(hm_read(&cache, 0, 0), 10.0);
        assert_eq!(hm_read(&cache, 1, 0), 30.0);
        assert_eq!(hm_read(&cache, 2, 0), 50.0);
        // Head 1: [20, 40, 60] at positions 0, 1, 2
        assert_eq!(hm_read(&cache, 0, 1), 20.0);
        assert_eq!(hm_read(&cache, 1, 1), 40.0);
        assert_eq!(hm_read(&cache, 2, 1), 60.0);
    }

    #[test]
    fn test_hm_prune_prefix() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        // Insert 5 tokens with distinct values per head
        let backend = Arc::new(CpuBackend::new());
        for i in 0..5 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i * 10 + d) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = (i * 100 + d) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 5);

        // Head 0 pos 0: [0,1], pos 2: [20,21]
        assert_eq!(hm_read(&cache, 0, 0), 0.0);
        assert_eq!(hm_read(&cache, 2, 0), 20.0);

        // Prune first 2 tokens
        cache.prune_prefix(2).unwrap();
        assert_eq!(cache.current_pos, 3);

        // After prune: old pos 2 is now pos 0
        // Head 0: [20, 30, 40]
        assert_eq!(hm_read(&cache, 0, 0), 20.0);
        assert_eq!(hm_read(&cache, 1, 0), 30.0);
        assert_eq!(hm_read(&cache, 2, 0), 40.0);
        // Head 1: [200, 300, 400]
        assert_eq!(hm_read(&cache, 0, 1), 200.0);
        assert_eq!(hm_read(&cache, 1, 1), 300.0);
        assert_eq!(hm_read(&cache, 2, 1), 400.0);
    }

    #[test]
    fn test_hm_dynamic_growth() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_dynamic_cache(4, 64, heads, dim);
        assert_eq!(cache.capacity(), 4);

        // Fill 4 tokens
        let backend = Arc::new(CpuBackend::new());
        for i in 0..4 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i + 1) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = ((i + 1) * 10) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.capacity(), 4);

        // 5th token triggers grow
        let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut f32;
            for d in 0..dim {
                *ptr.add(d) = 5.0;
            }
            for d in 0..dim {
                *ptr.add(dim + d) = 50.0;
            }
        }
        let k = Tensor::new(
            Shape::new(vec![1, 1, heads, dim]),
            buf.clone(),
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend);
        cache.update(&k, &v).unwrap();
        assert!(cache.capacity() >= 5);
        assert_eq!(cache.current_pos, 5);

        // Verify data integrity after growth
        assert_eq!(hm_read(&cache, 0, 0), 1.0);
        assert_eq!(hm_read(&cache, 3, 0), 4.0);
        assert_eq!(hm_read(&cache, 4, 0), 5.0);
        assert_eq!(hm_read(&cache, 0, 1), 10.0);
        assert_eq!(hm_read(&cache, 4, 1), 50.0);
    }

    #[test]
    fn test_hm_shift_positions() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        let backend = Arc::new(CpuBackend::new());
        for i in 0..5 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i * 10 + d) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = (i * 100 + d) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }

        // shift_positions: move positions 3..5 to 1..3
        cache.shift_positions(3, 1, 2).unwrap();

        // Head 0: pos 1 now has old pos 3 value (30)
        assert_eq!(hm_read(&cache, 1, 0), 30.0);
        assert_eq!(hm_read(&cache, 2, 0), 40.0);
        // Head 1: pos 1 now has old pos 3 value (300)
        assert_eq!(hm_read(&cache, 1, 1), 300.0);
        assert_eq!(hm_read(&cache, 2, 1), 400.0);
    }

    /// Cross-layout: verify SeqMajor and HeadMajor produce same logical data
    #[test]
    fn test_cross_layout_equivalence() {
        let heads = 2;
        let dim = 4;
        let backend = Arc::new(CpuBackend::new());

        let mut sm_cache = make_cache(16, heads, dim);
        let mut hm_cache = make_head_major_cache(16, heads, dim);

        // Insert same 5 tokens into both
        for i in 0..5 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..(heads * dim) {
                    *ptr.add(d) = (i * 100 + d) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            sm_cache.update(&k, &v).unwrap();

            // Same data for hm
            let buf2 = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf2.as_mut_ptr() as *mut f32;
                for d in 0..(heads * dim) {
                    *ptr.add(d) = (i * 100 + d) as f32;
                }
            }
            let k2 = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf2.clone(),
                backend.clone(),
            );
            let v2 = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf2, backend.clone());
            hm_cache.update(&k2, &v2).unwrap();
        }

        // Compare logical values at each (pos, head, d)
        let sm_k = sm_cache.k_buffer.as_slice::<f32>();
        let hm_k = hm_cache.k_buffer.as_slice::<f32>();
        for pos in 0..5 {
            for h in 0..heads {
                for d in 0..dim {
                    let sm_off = sm_cache.offset(pos, h) + d;
                    let hm_off = hm_cache.offset(pos, h) + d;
                    assert_eq!(
                        sm_k[sm_off], hm_k[hm_off],
                        "Mismatch at pos={}, head={}, d={}",
                        pos, h, d
                    );
                }
            }
        }

        // Prune and compare
        sm_cache.prune_prefix(2).unwrap();
        hm_cache.prune_prefix(2).unwrap();
        let sm_k = sm_cache.k_buffer.as_slice::<f32>();
        let hm_k = hm_cache.k_buffer.as_slice::<f32>();
        for pos in 0..3 {
            for h in 0..heads {
                for d in 0..dim {
                    let sm_off = sm_cache.offset(pos, h) + d;
                    let hm_off = hm_cache.offset(pos, h) + d;
                    assert_eq!(
                        sm_k[sm_off], hm_k[hm_off],
                        "After prune: mismatch at pos={}, head={}, d={}",
                        pos, h, d
                    );
                }
            }
        }
    }

    // ── Per-head shift tests ──

    #[test]
    fn test_shift_positions_for_head_basic() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        let backend = Arc::new(CpuBackend::new());
        for i in 0..5 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i * 10 + d) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = (i * 100 + d) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }
        // Head 0: pos0=0, pos1=10, pos2=20, pos3=30, pos4=40
        // Head 1: pos0=0, pos1=100, pos2=200, pos3=300, pos4=400

        // Move pos 3 → pos 1 for HEAD 0 ONLY
        cache.shift_positions_for_head(0, 3, 1, 1).unwrap();

        // Head 0: pos 1 should now have value 30 (from pos 3)
        assert_eq!(hm_read(&cache, 1, 0), 30.0);
        // Head 1: pos 1 should be UNCHANGED (still 100)
        assert_eq!(hm_read(&cache, 1, 1), 100.0);

        // Head 0: pos 0 should be unchanged
        assert_eq!(hm_read(&cache, 0, 0), 0.0);
        // Head 1: pos 3 should be unchanged
        assert_eq!(hm_read(&cache, 3, 1), 300.0);
    }

    #[test]
    fn test_shift_positions_for_head_multi_count() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        let backend = Arc::new(CpuBackend::new());
        for i in 0..6 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i * 10) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = (i * 100) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }

        // Move positions 4..6 → 2..4 for head 1 only
        cache.shift_positions_for_head(1, 4, 2, 2).unwrap();

        // Head 1: pos 2 = old pos 4 = 400, pos 3 = old pos 5 = 500
        assert_eq!(hm_read(&cache, 2, 1), 400.0);
        assert_eq!(hm_read(&cache, 3, 1), 500.0);
        // Head 0: unchanged at pos 2, 3
        assert_eq!(hm_read(&cache, 2, 0), 20.0);
        assert_eq!(hm_read(&cache, 3, 0), 30.0);
    }

    #[test]
    #[should_panic(expected = "HeadMajor")]
    fn test_shift_positions_for_head_panics_seq_major() {
        let mut cache = make_cache(16, 2, 4);
        cache.current_pos = 5;
        cache.shift_positions_for_head(0, 3, 1, 1).unwrap();
    }

    #[test]
    fn test_shift_positions_for_head_noop() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);
        cache.current_pos = 5;

        // count=0 should be noop
        cache.shift_positions_for_head(0, 3, 1, 0).unwrap();
        // src==dst should be noop
        cache.shift_positions_for_head(0, 3, 3, 1).unwrap();
    }

    // ── madvise / release_unused_pages tests ──

    /// Helper: make a large F16 SeqMajor cache (buffer > 1 page).
    fn make_large_f16_cache(max_seq_len: usize, heads: usize, dim: usize) -> KVCache {
        let size_bytes = max_seq_len * heads * dim * 2; // F16
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F16));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F16));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(
            Shape::new(vec![1, max_seq_len, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, max_seq_len, heads, dim]), v_buf, backend);
        KVCache::new(k, v, max_seq_len)
    }

    fn make_large_f16_headmajor_cache(max_seq_len: usize, heads: usize, dim: usize) -> KVCache {
        let size_bytes = max_seq_len * heads * dim * 2; // F16
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F16));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F16));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(
            Shape::new(vec![1, heads, max_seq_len, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, heads, max_seq_len, dim]), v_buf, backend);
        KVCache {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            high_water_pos: 0,
            max_seq_len,
            capacity: max_seq_len,
            kv_heads: heads,
            head_dim: dim,
            layout: KVLayout::HeadMajor,
            memory: None,
        }
    }

    /// Read VmRSS from /proc/self/status in bytes.
    #[cfg(target_os = "linux")]
    fn read_rss_bytes() -> usize {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let kb: usize = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return kb * 1024;
            }
        }
        0
    }

    #[test]
    fn test_release_unused_pages_returns_bytes_seqmajor() {
        // 8192 positions * 8 heads * 64 dim * 2 bytes = 8 MB per buffer
        let mut cache = make_large_f16_cache(8192, 8, 64);
        cache.current_pos = 8192;
        cache.high_water_pos = 8192;

        // Touch all pages to ensure physical allocation
        let ptr = cache.k_buffer.as_mut_ptr();
        if !ptr.is_null() {
            for i in (0..cache.k_buffer.size()).step_by(4096) {
                unsafe { std::ptr::write_volatile(ptr.add(i), 0xABu8) };
            }
        }

        // Evict most tokens: keep 100 out of 8192
        cache.current_pos = 100;
        let released = cache.release_unused_pages();

        // With SeqMajor, single contiguous region.
        // used = 100 * 8 * 64 * 2 = 102400 bytes (~25 pages)
        // total = 8192 * 8 * 64 * 2 = 8388608 bytes (~2048 pages)
        // Released should be significant (> 7 MB from K+V)
        assert!(
            released > 7_000_000,
            "Expected >7MB released, got {} bytes",
            released
        );
    }

    #[test]
    fn test_release_unused_pages_returns_bytes_headmajor() {
        // 8192 positions * 8 heads * 64 dim * 2 bytes = 8 MB per buffer
        let mut cache = make_large_f16_headmajor_cache(8192, 8, 64);
        cache.current_pos = 8192;
        cache.high_water_pos = 8192;

        // Touch all pages
        let ptr = cache.k_buffer.as_mut_ptr();
        if !ptr.is_null() {
            for i in (0..cache.k_buffer.size()).step_by(4096) {
                unsafe { std::ptr::write_volatile(ptr.add(i), 0xABu8) };
            }
        }

        cache.current_pos = 100;
        let released = cache.release_unused_pages();

        // HeadMajor: per head = 8192 * 64 * 2 = 1048576 bytes = 1MB
        // used per head = 100 * 64 * 2 = 12800 bytes (~3 pages used)
        // ~253 pages released per head, 8 heads = ~2024 pages per buffer, x2 (K+V)
        assert!(
            released > 7_000_000,
            "Expected >7MB released, got {} bytes",
            released
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_release_unused_pages_rss_reduction() {
        // Allocate large cache, touch pages, then release and check RSS drops
        let mut cache = make_large_f16_cache(16384, 8, 64);
        cache.current_pos = 16384;
        cache.high_water_pos = 16384;

        // Touch all pages in both K and V
        for buf_ptr in [cache.k_buffer.as_mut_ptr(), cache.v_buffer.as_mut_ptr()] {
            if !buf_ptr.is_null() {
                let size = 16384 * 8 * 64 * 2;
                for i in (0..size).step_by(4096) {
                    unsafe { std::ptr::write_volatile(buf_ptr.add(i), 0xCDu8) };
                }
            }
        }

        let rss_before = read_rss_bytes();

        // Evict most: keep 10 tokens out of 16384
        cache.current_pos = 10;
        cache.release_unused_pages();

        let rss_after = read_rss_bytes();

        // KV buffer total = 16384 * 8 * 64 * 2 * 2 (K+V) = 32 MB
        // After eviction: ~0.02 MB used, ~32 MB released
        // RSS should drop by at least 20 MB (allowing margin for test runtime overhead)
        let rss_drop = rss_before.saturating_sub(rss_after);
        assert!(
            rss_drop > 20_000_000,
            "Expected RSS drop > 20MB, got {} MB drop (before={}, after={})",
            rss_drop / (1024 * 1024),
            rss_before / (1024 * 1024),
            rss_after / (1024 * 1024),
        );
    }

    #[test]
    fn test_release_unused_pages_regrowth_integrity() {
        // After release, new data written to madvise'd region should work
        let mut cache = make_large_f16_cache(4096, 2, 64);
        cache.current_pos = 4096;
        cache.high_water_pos = 4096;

        // Touch all pages
        let k_ptr = cache.k_buffer.as_mut_ptr();
        if !k_ptr.is_null() {
            for i in (0..cache.k_buffer.size()).step_by(4096) {
                unsafe { std::ptr::write_volatile(k_ptr.add(i), 0xABu8) };
            }
        }

        // Evict to 10 tokens and release
        cache.current_pos = 10;
        cache.release_unused_pages();

        // "Regrow": write new data into the released region
        cache.current_pos = 2000;
        let k_ptr = cache.k_buffer.as_mut_ptr();
        if !k_ptr.is_null() {
            let type_size = 2; // F16
            let row_size = 2 * 64 * type_size; // heads * dim * type_size
            // Write a marker at position 1000
            let offset = 1000 * row_size;
            unsafe {
                std::ptr::write_volatile(k_ptr.add(offset), 0x42u8);
                // Read it back
                let val = std::ptr::read_volatile(k_ptr.add(offset));
                assert_eq!(
                    val, 0x42u8,
                    "Data written to released region should be readable"
                );
            }
        }
    }

    #[test]
    fn test_release_unused_pages_noop_on_full_cache() {
        let mut cache = make_large_f16_cache(1024, 2, 64);
        cache.current_pos = 1024; // Full cache
        cache.high_water_pos = 1024;
        let released = cache.release_unused_pages();
        assert_eq!(released, 0, "Full cache should release 0 bytes");
    }

    #[test]
    fn test_release_unused_pages_noop_on_small_eviction() {
        let mut cache = make_large_f16_cache(1024, 2, 64);
        cache.current_pos = 1024;
        cache.high_water_pos = 1024;

        // Touch pages
        let ptr = cache.k_buffer.as_mut_ptr();
        if !ptr.is_null() {
            for i in (0..cache.k_buffer.size()).step_by(4096) {
                unsafe { std::ptr::write_volatile(ptr.add(i), 0xABu8) };
            }
        }

        // Evict just 1 token — likely < 1 page for most configs
        cache.current_pos = 1023;
        let released = cache.release_unused_pages();
        // 1 token = 2 * 64 * 2 = 256 bytes, well under page size → 0 released
        assert_eq!(released, 0, "Evicting < 1 page should release 0 bytes");
    }

    #[test]
    fn test_release_unused_pages_headmajor_boundary_safety() {
        // Verify HeadMajor madvise doesn't corrupt adjacent head data
        let heads = 4;
        let dim = 64;
        let cap = 4096;
        let mut cache = make_large_f16_headmajor_cache(cap, heads, dim);
        cache.current_pos = cap;
        cache.high_water_pos = cap;

        let type_size = 2; // F16
        let head_stride = cap * dim * type_size;

        // Write a marker at the START of each head region (position 0)
        let k_ptr = cache.k_buffer.as_mut_ptr();
        if k_ptr.is_null() {
            return;
        }
        for h in 0..heads {
            unsafe {
                let head_base = k_ptr.add(h * head_stride);
                std::ptr::write_volatile(head_base, (0x10 + h) as u8);
            }
        }

        // Evict to 10 tokens and release unused pages
        cache.current_pos = 10;
        cache.release_unused_pages();

        // Verify markers at head starts are preserved (they're in the "used" region)
        for h in 0..heads {
            unsafe {
                let head_base = k_ptr.add(h * head_stride);
                let val = std::ptr::read_volatile(head_base);
                assert_eq!(
                    val,
                    (0x10 + h) as u8,
                    "Head {} start marker should be preserved after madvise",
                    h
                );
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_prune_prefix_calls_release_unused_pages() {
        // prune_prefix should release pages (verified via RSS)
        // Use large buffers (32 MB total) and touch both K and V for clear signal.
        let mut cache = make_large_f16_cache(16384, 8, 64);
        cache.current_pos = 16384;
        cache.high_water_pos = 16384;

        // Touch all pages in both K and V buffers
        for buf_ptr in [cache.k_buffer.as_mut_ptr(), cache.v_buffer.as_mut_ptr()] {
            if !buf_ptr.is_null() {
                let size = 16384 * 8 * 64 * 2;
                for i in (0..size).step_by(4096) {
                    unsafe { std::ptr::write_volatile(buf_ptr.add(i), 0xABu8) };
                }
            }
        }

        let rss_before = read_rss_bytes();
        cache.prune_prefix(16000).unwrap(); // Keep 384 tokens
        let rss_after = read_rss_bytes();

        assert_eq!(cache.current_pos, 384);
        // 32 MB total, ~97% evicted → expect > 20 MB RSS drop
        let rss_drop = rss_before.saturating_sub(rss_after);
        assert!(
            rss_drop > 20_000_000,
            "Expected RSS drop > 20MB from prune_prefix, got {} MB (before={}, after={})",
            rss_drop / (1024 * 1024),
            rss_before / (1024 * 1024),
            rss_after / (1024 * 1024),
        );
    }

    #[test]
    fn test_release_unused_pages_noop_on_q4_dtype() {
        // Q4_0 dtype should be skipped (returns 0)
        use crate::quant::QK4_0;
        let heads = 2;
        let dim = 64;
        let max_seq = 1024;
        let blocks_per_pos = heads * dim / QK4_0;
        let block_size = std::mem::size_of::<crate::quant::BlockQ4_0>();
        let buf_size = max_seq * blocks_per_pos * block_size;
        let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::Q4_0));
        let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::Q4_0));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, max_seq, heads, dim]), v_buf, backend);
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = 512;
        let released = cache.release_unused_pages();
        assert_eq!(released, 0, "Q4_0 dtype should not trigger madvise");
    }

    /// Regression (2026-06-09): **dynamic** Q4_0 HeadMajor cache 가 underutilized 되어
    /// `shrink_to_fit` 이 발동할 때, `copy_slice` 의 count/offset 은 element 가 아니라
    /// **block** 단위여야 한다 (Q4_0 `type_size = sizeof(BlockQ4_0) = 18B`). element 로
    /// 넘기면 실제 버퍼의 ~32배(QK4_0)를 읽어 **OOB → SIGSEGV**.
    /// `grow()` 는 block 변환을 했으나 `shrink_to_fit()` 에 누락되어, argus_bench h2o
    /// eviction → `release_unused_pages` → `shrink_to_fit` 경로에서 q4_0 한정 segfault.
    /// (f16/f32 는 element 단위가 곧 정답이라 무사, opaque 는 byte 회계라 무사.)
    /// 기존 `noop_on_q4_dtype` 는 non-dynamic(`memory=None`)이라 shrink 경로 미진입 →
    /// 이 버그를 못 잡았다.
    #[test]
    fn test_shrink_to_fit_q4_0_headmajor_no_oob_and_preserves_data() {
        use crate::quant::QK4_0;
        let heads = 2;
        let dim = 64;
        let init_cap = 256;
        let max_seq = 1024;
        let blocks_per_pos = heads * dim / QK4_0;
        let block_size = std::mem::size_of::<crate::quant::BlockQ4_0>();
        let buf_bytes = init_cap * blocks_per_pos * block_size;

        let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
        let backend = Arc::new(CpuBackend::new());
        let k_buf = Arc::new(SharedBuffer::new(buf_bytes, DType::Q4_0));
        let v_buf = Arc::new(SharedBuffer::new(buf_bytes, DType::Q4_0));
        let k = Tensor::new(
            Shape::new(vec![1, heads, init_cap, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, heads, init_cap, dim]), v_buf, backend);
        let mut cache = KVCache::new_dynamic(k, v, init_cap, max_seq, heads, dim, memory)
            .with_layout(KVLayout::HeadMajor);

        // current_pos*4 < cap*3 (underutilized) AND new_cap < cap → shrink 발동.
        let keep = 64usize;
        let new_cap = ((keep * 3 / 2) + 63) & !63; // 128
        cache.current_pos = keep;
        cache.high_water_pos = keep;

        // (head h, pos p) 첫 블록의 첫 바이트에 고유 marker. HeadMajor Q4_0 block offset:
        //   block(h,p) = h * (cap*dim/QK4_0) + p * (dim/QK4_0)
        let hd_blocks = dim / QK4_0;
        let old_head_stride_blocks = init_cap * dim / QK4_0;
        let marker = |h: usize, p: usize| ((h * keep + p) % 251 + 1) as u8;
        {
            let kb = cache.k_buffer.as_mut_slice::<u8>();
            for h in 0..heads {
                for p in 0..keep {
                    let off = (h * old_head_stride_blocks + p * hd_blocks) * block_size;
                    kb[off] = marker(h, p);
                }
            }
        }

        // 버그 시 이 줄에서 SIGSEGV (copy_slice element-count over-read).
        let freed = cache.release_unused_pages();

        assert_eq!(cache.capacity, new_cap, "Q4_0 HeadMajor shrink new_cap");
        assert!(freed > 0, "shrink 으로 메모리 회수");

        // 새 capacity 기준 offset 에서 marker 보존 검증 (copy_slice 가 올바른 block 을 옮겼는지).
        let new_head_stride_blocks = new_cap * dim / QK4_0;
        let kb = cache.k_buffer.as_slice::<u8>();
        for h in 0..heads {
            for p in 0..keep {
                let off = (h * new_head_stride_blocks + p * hd_blocks) * block_size;
                assert_eq!(kb[off], marker(h, p), "Q4_0 (h={h},p={p}) marker 보존");
            }
        }
    }

    /// SeqMajor Q4_0 shrink 도 동일 block-count 회계 (fix 의 SeqMajor arm). no-OOB + 회수.
    #[test]
    fn test_shrink_to_fit_q4_0_seqmajor_no_oob() {
        use crate::quant::QK4_0;
        let heads = 2;
        let dim = 64;
        let init_cap = 256;
        let max_seq = 1024;
        let blocks_per_pos = heads * dim / QK4_0;
        let block_size = std::mem::size_of::<crate::quant::BlockQ4_0>();
        let buf_bytes = init_cap * blocks_per_pos * block_size;

        let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
        let backend = Arc::new(CpuBackend::new());
        let k_buf = Arc::new(SharedBuffer::new(buf_bytes, DType::Q4_0));
        let v_buf = Arc::new(SharedBuffer::new(buf_bytes, DType::Q4_0));
        let k = Tensor::new(
            Shape::new(vec![1, init_cap, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, init_cap, heads, dim]), v_buf, backend);
        // SeqMajor 가 기본 layout (with_layout 미호출).
        let mut cache = KVCache::new_dynamic(k, v, init_cap, max_seq, heads, dim, memory);

        let keep = 64usize;
        let new_cap = ((keep * 3 / 2) + 63) & !63;
        cache.current_pos = keep;
        cache.high_water_pos = keep;

        // marker: SeqMajor Q4_0 block(p) = p * blocks_per_pos (첫 바이트).
        {
            let kb = cache.k_buffer.as_mut_slice::<u8>();
            for p in 0..keep {
                kb[p * blocks_per_pos * block_size] = ((p % 251) + 1) as u8;
            }
        }

        let freed = cache.release_unused_pages(); // 버그 시 SIGSEGV
        assert_eq!(cache.capacity, new_cap);
        assert!(freed > 0);
        let kb = cache.k_buffer.as_slice::<u8>();
        for p in 0..keep {
            assert_eq!(kb[p * blocks_per_pos * block_size], ((p % 251) + 1) as u8);
        }
    }

    #[test]
    fn test_round_up_and_round_down() {
        assert_eq!(super::round_up(0, 4096), 0);
        assert_eq!(super::round_up(1, 4096), 4096);
        assert_eq!(super::round_up(4095, 4096), 4096);
        assert_eq!(super::round_up(4096, 4096), 4096);
        assert_eq!(super::round_up(4097, 4096), 8192);

        assert_eq!(super::round_down(0, 4096), 0);
        assert_eq!(super::round_down(1, 4096), 0);
        assert_eq!(super::round_down(4095, 4096), 0);
        assert_eq!(super::round_down(4096, 4096), 4096);
        assert_eq!(super::round_down(4097, 4096), 4096);
    }

    #[test]
    fn test_madvise_dontneed_null_ptr() {
        // Null pointer should return 0 without crashing
        let released = super::madvise_dontneed(std::ptr::null(), 0, 4096);
        assert_eq!(released, 0);
    }

    #[test]
    fn test_madvise_dontneed_empty_range() {
        let buf = vec![0u8; 8192];
        let released = super::madvise_dontneed(buf.as_ptr(), 4096, 4096);
        assert_eq!(released, 0, "from == to should release 0");
        let released = super::madvise_dontneed(buf.as_ptr(), 4096, 100);
        assert_eq!(released, 0, "from > to should release 0");
    }

    // ── compact_keep_positions tests ──

    /// Write value `val` at (pos, all heads, d=0) in a HeadMajor F32 cache.
    fn hm_write_pos(cache: &mut KVCache, pos: usize, val: f32) {
        let heads = cache.kv_heads;
        let dim = cache.head_dim;
        let cap = cache.capacity;
        let k_slice = cache.k_buffer.as_mut_slice::<f32>();
        let v_slice = cache.v_buffer.as_mut_slice::<f32>();
        for h in 0..heads {
            let base = h * cap * dim;
            k_slice[base + pos * dim] = val;
            v_slice[base + pos * dim] = val * 10.0;
        }
    }

    /// Read the K value at (pos, head=0, d=0) in a HeadMajor F32 cache.
    fn hm_read_k(cache: &KVCache, pos: usize) -> f32 {
        let off = cache.offset(pos, 0);
        cache.k_buffer.as_slice::<f32>()[off]
    }

    /// Read the V value at (pos, head=0, d=0) in a HeadMajor F32 cache.
    fn hm_read_v_d0(cache: &KVCache, pos: usize) -> f32 {
        let off = cache.offset(pos, 0);
        cache.v_buffer.as_slice::<f32>()[off]
    }

    /// Make a HeadMajor F32 cache pre-filled with distinct values (pos+1) at every position.
    fn make_filled_hm_cache(max_seq: usize, heads: usize, dim: usize) -> KVCache {
        let mut cache = make_head_major_cache(max_seq, heads, dim);
        for pos in 0..max_seq {
            hm_write_pos(&mut cache, pos, (pos + 1) as f32);
        }
        cache.current_pos = max_seq;
        cache
    }

    #[test]
    fn test_compact_keep_positions_consecutive() {
        // keep = [2, 3, 4] — a single consecutive run
        // write_start = 0
        // Expected: pos 0←2, 1←3, 2←4 in a single batch shift
        let mut cache = make_filled_hm_cache(10, 2, 4);
        let keep = vec![2usize, 3, 4];
        cache.compact_keep_positions(&keep, 0).unwrap();
        cache.current_pos = keep.len();

        // pos 0 should have old pos 2 value = 3.0
        assert_eq!(hm_read_k(&cache, 0), 3.0);
        assert_eq!(hm_read_k(&cache, 1), 4.0);
        assert_eq!(hm_read_k(&cache, 2), 5.0);
        assert_eq!(hm_read_v_d0(&cache, 0), 30.0);
    }

    #[test]
    fn test_compact_keep_positions_scattered() {
        // keep = [1, 3, 5, 7] — scattered, no consecutive pairs
        // write_start = 0
        let mut cache = make_filled_hm_cache(10, 2, 4);
        let keep = vec![1usize, 3, 5, 7];
        cache.compact_keep_positions(&keep, 0).unwrap();
        cache.current_pos = keep.len();

        // Each is shifted individually: dst 0←1, 1←3, 2←5, 3←7
        assert_eq!(hm_read_k(&cache, 0), 2.0); // was pos 1 → value 2
        assert_eq!(hm_read_k(&cache, 1), 4.0); // was pos 3 → value 4
        assert_eq!(hm_read_k(&cache, 2), 6.0); // was pos 5 → value 6
        assert_eq!(hm_read_k(&cache, 3), 8.0); // was pos 7 → value 8
    }

    #[test]
    fn test_compact_keep_positions_one_gap() {
        // Classic H2O scenario: keep HH at [0,2,4] and recent [7,8,9]
        // write_start = 0 (no protected prefix)
        // Batching: [0] in-place, [2] single, [4] single, [7,8,9] batch of 3
        let mut cache = make_filled_hm_cache(10, 2, 4);
        let keep = vec![0usize, 2, 4, 7, 8, 9];
        cache.compact_keep_positions(&keep, 0).unwrap();
        cache.current_pos = keep.len();

        assert_eq!(hm_read_k(&cache, 0), 1.0); // pos 0, in-place
        assert_eq!(hm_read_k(&cache, 1), 3.0); // pos 1 ← old pos 2
        assert_eq!(hm_read_k(&cache, 2), 5.0); // pos 2 ← old pos 4
        assert_eq!(hm_read_k(&cache, 3), 8.0); // pos 3 ← old pos 7
        assert_eq!(hm_read_k(&cache, 4), 9.0); // pos 4 ← old pos 8
        assert_eq!(hm_read_k(&cache, 5), 10.0); // pos 5 ← old pos 9
    }

    #[test]
    fn test_compact_keep_positions_all_in_place() {
        // keep = [0, 1, 2] with write_start = 0 — nothing to move
        let mut cache = make_filled_hm_cache(10, 2, 4);
        cache.compact_keep_positions(&[0, 1, 2], 0).unwrap();
        cache.current_pos = 3;

        // Values unchanged
        assert_eq!(hm_read_k(&cache, 0), 1.0);
        assert_eq!(hm_read_k(&cache, 1), 2.0);
        assert_eq!(hm_read_k(&cache, 2), 3.0);
    }

    #[test]
    fn test_compact_keep_positions_empty() {
        // Empty keep list → no-op, no panic
        let mut cache = make_filled_hm_cache(10, 2, 4);
        cache.compact_keep_positions(&[], 0).unwrap();
        // Values at original positions are untouched
        assert_eq!(hm_read_k(&cache, 0), 1.0);
    }

    #[test]
    fn test_compact_keep_positions_with_write_start() {
        // Simulate protected prefix: write_start = 2, keep = [4, 5, 8, 9]
        // 4,5 consecutive batch → shift to 2,3; 8,9 consecutive batch → shift to 4,5
        let mut cache = make_filled_hm_cache(12, 2, 4);
        let keep = vec![4usize, 5, 8, 9];
        cache.compact_keep_positions(&keep, 2).unwrap();
        cache.current_pos = 2 + keep.len();

        assert_eq!(hm_read_k(&cache, 2), 5.0); // old pos 4
        assert_eq!(hm_read_k(&cache, 3), 6.0); // old pos 5
        assert_eq!(hm_read_k(&cache, 4), 9.0); // old pos 8
        assert_eq!(hm_read_k(&cache, 5), 10.0); // old pos 9
        // Protected prefix (pos 0,1) untouched
        assert_eq!(hm_read_k(&cache, 0), 1.0);
        assert_eq!(hm_read_k(&cache, 1), 2.0);
    }

    #[test]
    fn test_compact_keep_positions_for_head() {
        // Per-head version: only head 0 is modified; head 1 untouched.
        // Cache: heads=2, dim=2, max_seq=10
        // Fill each head differently: head 0 = (pos+1)*1.0, head 1 = (pos+1)*100.0
        let heads = 2;
        let dim = 2;
        let cap = 10;
        let size_bytes = cap * heads * dim * 4;
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(Shape::new(vec![1, heads, cap, dim]), k_buf, backend.clone());
        let v = Tensor::new(Shape::new(vec![1, heads, cap, dim]), v_buf, backend);
        let mut cache = KVCache {
            k_buffer: k,
            v_buffer: v,
            current_pos: cap,
            high_water_pos: cap,
            max_seq_len: cap,
            capacity: cap,
            kv_heads: heads,
            head_dim: dim,
            layout: KVLayout::HeadMajor,
            memory: None,
        };

        // Write distinct values per head
        {
            let k_slice = cache.k_buffer.as_mut_slice::<f32>();
            // head 0: stride = cap * dim = 20
            for pos in 0..cap {
                k_slice[pos * dim] = (pos + 1) as f32; // head 0
                k_slice[cap * dim + pos * dim] = (pos + 1) as f32 * 100.0; // head 1
            }
        }

        // Compact head 0 only: keep positions [2, 3, 7, 8, 9] → write_start=0
        let keep = vec![2usize, 3, 7, 8, 9];
        cache.compact_keep_positions_for_head(0, &keep, 0).unwrap();

        {
            let k_slice = cache.k_buffer.as_slice::<f32>();
            // Head 0 at pos 0 ← old pos 2 = 3.0; pos 1 ← old pos 3 = 4.0
            assert_eq!(k_slice[0], 3.0, "head0 pos0 should be old pos2");
            assert_eq!(k_slice[dim], 4.0, "head0 pos1 should be old pos3");
            // pos 2,3,4 ← old pos 7,8,9 (consecutive batch)
            assert_eq!(k_slice[2 * dim], 8.0, "head0 pos2 should be old pos7");
            assert_eq!(k_slice[3 * dim], 9.0, "head0 pos3 should be old pos8");
            assert_eq!(k_slice[4 * dim], 10.0, "head0 pos4 should be old pos9");

            // Head 1 is completely untouched
            for pos in 0..cap {
                let expected = (pos + 1) as f32 * 100.0;
                assert_eq!(
                    k_slice[cap * dim + pos * dim],
                    expected,
                    "head1 pos{} should be unchanged",
                    pos
                );
            }
        }
    }

    // ── high_water_pos tracking tests ──

    /// Test that high_water_pos is correctly maintained across
    /// update → eviction → release_unused_pages flow.
    #[test]
    fn test_high_water_pos_tracking() {
        let mut cache = make_cache(100, 1, 4);

        // Initially both counters are zero
        assert_eq!(cache.current_pos, 0);
        assert_eq!(cache.high_water_pos, 0);

        // update() should advance high_water_pos monotonically
        for i in 0..10 {
            let (k, v) = make_token_tensor((i + 1) as f32, 1, 4);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 10);
        assert_eq!(cache.high_water_pos, 10);

        // Simulate eviction: current_pos drops, high_water_pos stays
        cache.current_pos = 5;
        assert_eq!(
            cache.high_water_pos, 10,
            "hwm must not decrease on eviction"
        );

        // release_unused_pages uses hwm as upper bound, then resets hwm to current_pos
        let released = cache.release_unused_pages();
        // Buffer is F32: 5 tokens * 1 head * 4 dim * 4 bytes = 80 bytes — below page granularity.
        // released may be 0 due to alignment, but high_water_pos must be reset.
        let _ = released; // don't assert bytes — page alignment makes it zero on small caches
        assert_eq!(
            cache.high_water_pos, 5,
            "hwm reset to current_pos after release"
        );

        // Continuing to update advances hwm again
        for i in 0..3 {
            let (k, v) = make_token_tensor((100 + i) as f32, 1, 4);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 8);
        assert_eq!(cache.high_water_pos, 8);

        // advance_pos also tracks hwm
        cache.advance_pos(4);
        assert_eq!(cache.current_pos, 12);
        assert_eq!(cache.high_water_pos, 12);

        // set_current_pos(0) resets hwm
        cache.set_current_pos(0);
        assert_eq!(cache.high_water_pos, 0, "set_current_pos(0) must reset hwm");

        // set_current_pos to a non-zero value does NOT reset hwm (only zero resets)
        cache.current_pos = 20;
        cache.high_water_pos = 20;
        cache.set_current_pos(10);
        assert_eq!(
            cache.high_water_pos, 20,
            "set_current_pos(non-zero) must not change hwm"
        );
    }

    /// Test that prune_prefix(all) resets high_water_pos via the remaining==0 path.
    #[test]
    fn test_high_water_pos_reset_on_prune_all() {
        let mut cache = make_cache(100, 1, 4);

        for i in 0..5 {
            let (k, v) = make_token_tensor((i + 1) as f32, 1, 4);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.high_water_pos, 5);

        // prune_prefix(all) → remaining == 0 path
        cache.prune_prefix(5).unwrap();
        assert_eq!(cache.current_pos, 0);
        assert_eq!(cache.high_water_pos, 0, "prune_prefix(all) must reset hwm");
    }

    /// Test that release_unused_pages with hwm as upper bound narrows the madvise range
    /// compared to using capacity as the upper bound.
    #[test]
    fn test_high_water_pos_limits_madvise_range() {
        // Large cache: 8192 positions, but only touch the first 1024 positions.
        // high_water_pos = 1024 → madvise range is current_pos..1024 (not ..8192).
        let mut cache = make_large_f16_cache(8192, 8, 64);

        // Only touch first 1024 positions worth of pages
        let row_bytes = 8 * 64 * 2; // heads * dim * F16
        let touch_bytes = 1024 * row_bytes;
        let ptr = cache.k_buffer.as_mut_ptr();
        if !ptr.is_null() {
            for i in (0..touch_bytes).step_by(4096) {
                unsafe { std::ptr::write_volatile(ptr.add(i), 0xBBu8) };
            }
        }

        // Set hwm to 1024 (matching the touched region), current_pos to 100
        cache.current_pos = 1024;
        cache.high_water_pos = 1024;
        cache.current_pos = 100;

        // release_unused_pages should only advise 100..1024 (not 100..8192)
        let released = cache.release_unused_pages();
        // Expected range: (1024 - 100) * 8 * 64 * 2 * 2 (K+V) = ~2 MB
        // Full-capacity range would be (8192 - 100) * 8 * 64 * 2 * 2 = ~16 MB
        // We assert > 1 MB to confirm madvise fired over meaningful range
        assert!(
            released > 1_000_000,
            "Expected >1MB released (hwm-bounded), got {} bytes",
            released
        );
        // And hwm must be reset to current_pos after release
        assert_eq!(cache.high_water_pos, 100);
    }
}
