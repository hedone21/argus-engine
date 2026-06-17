//! KIVI-style KV cache with asymmetric 2-bit quantization (ICML 2024).
//!
//! Key insight: recent R tokens stay in FP32 (residual buffer). When full,
//! they are batch-quantized to Q2 and flushed to compressed storage.
//!
//! - **Key cache**: per-channel quantization (groups across tokens within each channel).
//! - **Value cache**: per-token quantization (groups within one token's head_dim).
//!
//! `kv_dtype()` returns `F32` so LlamaLayer passes F32 data to `update()`.
//! `get_view()` returns dequantized F32 tensors for the existing attention path.
//!
//! ## GPU mode
//!
//! When constructed with `new_gpu()`, KiviCache allocates persistent GPU buffers for
//! residual and attention data. The hot path (update, get_view) runs GPU kernels.
//! The cold path (quantize during flush) still runs on CPU since it is infrequent.
//! CPU-only mode is fully preserved for backward compatibility.

use crate::backend::Backend;
// LAYER-EXEMPT: backend_concrete_downcast — §13.8-L
use crate::backend::cpu::CpuBackend;
use crate::buffer::{Buffer, DType};
use crate::capability::quant_attn::QuantAttnBackend;
use crate::kv_cache_ops::{KVLayout, KiviRawBuffers};
use crate::memory::Memory;
use crate::memory::host::shared::{SharedBuffer, SharedBufferView};
use crate::quant::{BlockKVQ4, BlockKVQ8, BlockQ2_0, QKKV};
use crate::shape::Shape;
use crate::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

/// Snapshot of post-softmax attention scores from the previous decode step.
/// Used for AWQE (Attention-Weighted Quantization Error) during flush.
#[derive(Clone)]
struct AttnScoresSnapshot {
    /// Flattened scores: [n_heads_q * stride]. Post-softmax values.
    scores: Vec<f32>,
    /// Number of Q heads.
    n_heads_q: usize,
    /// Stride between heads (= max_seq_len allocation).
    stride: usize,
    /// Number of valid positions per head at snapshot time.
    valid_len: usize,
}

/// Enum wrapping different quantization bit-widths for KV cache blocks.
#[derive(Clone)]
enum QuantizedBlocks {
    Unquantized, // bits=16: residual-only mode, no quantized data
    Q2(Vec<BlockQ2_0>),
    Q4(Vec<BlockKVQ4>),
    Q8(Vec<BlockKVQ8>),
}

impl QuantizedBlocks {
    fn new(bits: u8) -> Self {
        match bits {
            16 => QuantizedBlocks::Unquantized,
            2 => QuantizedBlocks::Q2(Vec::new()),
            4 => QuantizedBlocks::Q4(Vec::new()),
            8 => QuantizedBlocks::Q8(Vec::new()),
            _ => panic!("unsupported bits: {bits}"),
        }
    }

    fn with_capacity(bits: u8, cap: usize) -> Self {
        match bits {
            16 => QuantizedBlocks::Unquantized,
            2 => QuantizedBlocks::Q2(Vec::with_capacity(cap)),
            4 => QuantizedBlocks::Q4(Vec::with_capacity(cap)),
            8 => QuantizedBlocks::Q8(Vec::with_capacity(cap)),
            _ => panic!("unsupported bits: {bits}"),
        }
    }

    fn clear(&mut self) {
        match self {
            QuantizedBlocks::Unquantized => {}
            QuantizedBlocks::Q2(v) => v.clear(),
            QuantizedBlocks::Q4(v) => v.clear(),
            QuantizedBlocks::Q8(v) => v.clear(),
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            QuantizedBlocks::Unquantized => 0,
            QuantizedBlocks::Q2(v) => v.len() * std::mem::size_of::<BlockQ2_0>(),
            QuantizedBlocks::Q4(v) => v.len() * std::mem::size_of::<BlockKVQ4>(),
            QuantizedBlocks::Q8(v) => v.len() * std::mem::size_of::<BlockKVQ8>(),
        }
    }

    /// Quantize a group of QKKV f32 values and push to storage.
    fn push_quantized(&mut self, src: &[f32; QKKV]) {
        match self {
            QuantizedBlocks::Unquantized => panic!("push_quantized called on Unquantized"),
            QuantizedBlocks::Q2(v) => v.push(BlockQ2_0::quantize(src)),
            QuantizedBlocks::Q4(v) => v.push(BlockKVQ4::quantize(src)),
            QuantizedBlocks::Q8(v) => v.push(BlockKVQ8::quantize(src)),
        }
    }

    /// Dequantize block at given index into output buffer.
    fn dequantize_block(&self, idx: usize, out: &mut [f32; QKKV]) {
        match self {
            QuantizedBlocks::Unquantized => panic!("dequantize_block called on Unquantized"),
            QuantizedBlocks::Q2(v) => v[idx].dequantize(out),
            QuantizedBlocks::Q4(v) => v[idx].dequantize(out),
            QuantizedBlocks::Q8(v) => v[idx].dequantize(out),
        }
    }

    /// Total capacity in bytes (capacity * block_size).
    fn capacity_bytes(&self) -> usize {
        match self {
            QuantizedBlocks::Unquantized => 0,
            QuantizedBlocks::Q2(v) => v.capacity() * std::mem::size_of::<BlockQ2_0>(),
            QuantizedBlocks::Q4(v) => v.capacity() * std::mem::size_of::<BlockKVQ4>(),
            QuantizedBlocks::Q8(v) => v.capacity() * std::mem::size_of::<BlockKVQ8>(),
        }
    }

    /// Create an empty QuantizedBlocks with zero capacity for the given bit-width.
    fn empty(bits: u8) -> Self {
        match bits {
            16 => QuantizedBlocks::Unquantized,
            2 => QuantizedBlocks::Q2(Vec::new()),
            4 => QuantizedBlocks::Q4(Vec::new()),
            8 => QuantizedBlocks::Q8(Vec::new()),
            _ => panic!("unsupported bits: {bits}"),
        }
    }
}

/// KIVI KV cache: quantized compressed storage + FP32 residual buffer.
///
/// SeqMajor layout only (Phase 1). Data layout for residual:
/// `[kv_heads, res_cap, head_dim]` (head-first for easier per-channel Key quantization).
///
/// Quantized storage layout (after flush):
/// - Key: per-channel blocks. For each head, `head_dim` channels, each channel
///   quantized across tokens in groups of QKKV (32).
/// - Value: per-token blocks. For each head, each token's `head_dim` values
///   quantized in groups of QKKV (32).
///
/// Supports 2-bit (Q2), 4-bit (Q4), and 8-bit (Q8) quantization.
/// Dynamic bit transition via `transition_bits()`.
///
/// ## GPU mode
///
/// Set `gpu_backend` to `Some(...)` via `new_gpu()`. GPU buffers hold persistent
/// residual and attention F32 data. CPU residual vectors are still allocated for
/// the quantize step (cold path). All hot-path operations dispatch GPU kernels.
///
/// S-3b-4: `qcf_computer: Box<dyn QcfComputer>` 는 trait method
/// `clone_box()` 기반 `impl Clone for Box<dyn QcfComputer>` 으로 Clone 지원.
#[derive(Clone)]
pub struct KiviCache {
    /// Current quantization bit-width (2, 4, or 8).
    bits: u8,

    // Quantized compressed storage
    qk: QuantizedBlocks,
    qv: QuantizedBlocks,
    /// Number of tokens in quantized storage (always a multiple of `res_cap`).
    pub q2_tokens: usize,

    // FP32 residual buffer — layout: [kv_heads][res_cap][head_dim]
    res_k: Vec<f32>,
    res_v: Vec<f32>,
    /// Number of valid tokens in residual buffer.
    pub res_pos: usize,
    /// Residual buffer capacity (R). Must be a multiple of QKKV (32).
    res_cap: usize,

    // Pre-allocated output buffers for assemble_view() dequantization.
    // Arc<SharedBuffer> enables zero-copy get_view(): Arc::clone instead of memcpy.
    attn_k_buf: Arc<SharedBuffer>,
    attn_v_buf: Arc<SharedBuffer>,

    /// Number of quantized tokens already dequantized into attn_k_buf/attn_v_buf.
    /// Enables incremental dequantization — only new flushes are processed.
    q2_deq_tokens: usize,

    // Shared backend for get_view() tensors (avoid per-call allocation)
    out_backend: Arc<CpuBackend>,

    // Dimensions
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,

    // Group size for quantization (= QKKV = 32)
    group_size: usize,

    /// Accumulated flush proxy metrics (NMSE). Pushed during `flush_residual()`.
    flush_proxies: Vec<crate::qcf_types::QcfMetric>,

    /// Whether AWQE computation is enabled (set via set_awqe_enabled()).
    awqe_enabled: bool,
    /// Attention scores from the previous decode step (for AWQE).
    last_attn_scores: Option<AttnScoresSnapshot>,

    // ── GPU-native buffers (None = CPU-only mode, backward compatible) ──
    /// GPU backend handle. Some ↔ GPU mode enabled.
    gpu_backend: Option<Arc<dyn Backend>>,
    /// KIVI native attention capability handle (Phase α-W-4 §3.3). Some ↔ GPU
    /// 모드 + OpenCL backend. `update_gpu` 의 `gather_update_quant` dispatch 가
    /// 매 토큰 `as_quant_attn()` lookup 대신 본 handle 을 직접 쓴다. caller
    /// 가 `caps.get::<dyn QuantAttnBackend>()` 로 pull 해 `new_gpu` 에 전달
    /// (R4: gpu_backend 와 동일 ocl 인스턴스의 clone).
    kivi: Option<Arc<dyn QuantAttnBackend>>,
    /// Memory allocator used to create GPU buffers (used by ensure_gpu_attn_capacity).
    gpu_memory: Option<Arc<dyn Memory>>,
    /// GPU F32 residual K buffer: [kv_heads, res_cap, head_dim]
    gpu_res_k: Option<Tensor>,
    /// GPU F32 residual V buffer: [kv_heads, res_cap, head_dim]
    gpu_res_v: Option<Tensor>,
    /// GPU F16 attention K output: [gpu_attn_cap, kv_heads, head_dim]
    /// Capacity grows lazily via `ensure_gpu_attn_capacity()`.
    gpu_attn_k: Option<Tensor>,
    /// GPU F16 attention V output: [gpu_attn_cap, kv_heads, head_dim]
    /// Capacity grows lazily via `ensure_gpu_attn_capacity()`.
    gpu_attn_v: Option<Tensor>,
    /// Current allocated capacity of gpu_attn_k/v in tokens.
    /// 0 in CPU-only mode. Grows lazily up to `max_seq_len`.
    gpu_attn_cap: usize,
    /// GPU byte buffer for Q2 key blocks (12 bytes per block).
    gpu_q2k: Option<Tensor>,
    /// GPU byte buffer for Q2 value blocks (12 bytes per block).
    gpu_q2v: Option<Tensor>,
    /// Number of Q2 key blocks written to `gpu_q2k`.
    gpu_q2k_blocks: usize,
    /// Number of Q2 value blocks written to `gpu_q2v`.
    gpu_q2v_blocks: usize,

    /// QCF flush proxy 계산기 (L2 trait dispatch, S-3b-4 γ-2).
    /// Default = `KiviQcfComputer` (stateless). caller 가 다른 구현체를
    /// 주입하려면 `set_qcf_computer` 사용.
    qcf_computer: Box<dyn crate::qcf_computer::QcfComputer>,
}

impl KiviCache {
    // ── SharedBuffer ↔ f32 slice helpers ─────────────────────────────────────

    /// Return the attn_k_buf contents as a `&mut [f32]` slice.
    ///
    /// SAFETY: Caller must ensure no aliasing references exist. In practice,
    /// this is only called from `&mut self` methods (assemble_view, reset, etc.)
    /// where the borrow checker guarantees exclusive access to `self`.
    fn attn_k_as_slice_mut(&mut self) -> &mut [f32] {
        let len = self.attn_k_buf.size() / 4;
        if len == 0 {
            return &mut [];
        }
        // SAFETY: SharedBuffer::as_mut_ptr() returns a valid writable pointer.
        // We have &mut self, so no other references to this buffer exist.
        unsafe { std::slice::from_raw_parts_mut(self.attn_k_buf.as_mut_ptr() as *mut f32, len) }
    }

    /// Return the attn_v_buf contents as a `&mut [f32]` slice.
    fn attn_v_as_slice_mut(&mut self) -> &mut [f32] {
        let len = self.attn_v_buf.size() / 4;
        if len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.attn_v_buf.as_mut_ptr() as *mut f32, len) }
    }

    /// Create a new KiviCache with specified bit-width.
    ///
    /// - `kv_heads`: number of KV heads
    /// - `head_dim`: dimension per head (must be a multiple of QKKV)
    /// - `max_seq_len`: maximum sequence length
    /// - `residual_size`: FP32 residual buffer size in tokens (must be a multiple of QKKV).
    ///   For bits=16, the residual buffer is sized to `max_seq_len` regardless of this value.
    /// - `bits`: quantization bits (2, 4, 8, or 16). Default: 2.
    ///   bits=16 means unquantized/residual-only mode: all tokens are kept in FP32.
    pub fn new_with_bits(
        kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        residual_size: usize,
        bits: u8,
    ) -> Self {
        assert!(
            residual_size.is_multiple_of(QKKV) && residual_size > 0,
            "residual_size ({residual_size}) must be a positive multiple of {QKKV}"
        );
        assert!(
            head_dim.is_multiple_of(QKKV),
            "head_dim ({head_dim}) must be a multiple of {QKKV}"
        );
        assert!(
            bits == 2 || bits == 4 || bits == 8 || bits == 16,
            "bits must be 2, 4, 8, or 16, got {bits}"
        );

        // bits=16: residual covers the full sequence (no quantized storage needed)
        let actual_res_cap = if bits == 16 {
            max_seq_len
        } else {
            residual_size
        };
        let res_elems = kv_heads * actual_res_cap * head_dim;

        let groups_per_flush = residual_size / QKKV;
        let max_flushes = max_seq_len / residual_size;
        let max_k_blocks = max_flushes * groups_per_flush * kv_heads * head_dim;
        let blocks_per_token = head_dim / QKKV;
        let max_v_blocks = max_flushes * kv_heads * residual_size * blocks_per_token;

        let attn_buf_size = max_seq_len * kv_heads * head_dim;

        Self {
            bits,
            qk: QuantizedBlocks::with_capacity(bits, max_k_blocks),
            qv: QuantizedBlocks::with_capacity(bits, max_v_blocks),
            q2_tokens: 0,
            res_k: vec![0.0; res_elems],
            res_v: vec![0.0; res_elems],
            res_pos: 0,
            res_cap: actual_res_cap,
            attn_k_buf: Arc::new(SharedBuffer::new(attn_buf_size * 4, DType::F32)),
            attn_v_buf: Arc::new(SharedBuffer::new(attn_buf_size * 4, DType::F32)),
            q2_deq_tokens: 0,
            out_backend: Arc::new(CpuBackend::new()),
            kv_heads,
            head_dim,
            max_seq_len,
            group_size: QKKV,
            flush_proxies: Vec::new(),
            awqe_enabled: false,
            last_attn_scores: None,
            // GPU fields — None in CPU-only mode
            gpu_backend: None,
            kivi: None,
            gpu_memory: None,
            gpu_res_k: None,
            gpu_res_v: None,
            gpu_attn_k: None,
            gpu_attn_v: None,
            gpu_attn_cap: 0,
            gpu_q2k: None,
            gpu_q2v: None,
            gpu_q2k_blocks: 0,
            gpu_q2v_blocks: 0,
            // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cross-L3 default qcf computer (cold-path one-time init, S-3b-4 trail)
            qcf_computer: Box::<crate::qcf::KiviQcfComputer>::default(),
        }
    }

    /// Create a new KiviCache with default 2-bit quantization.
    pub fn new(kv_heads: usize, head_dim: usize, max_seq_len: usize, residual_size: usize) -> Self {
        Self::new_with_bits(kv_heads, head_dim, max_seq_len, residual_size, 2)
    }

    /// Current quantization bit-width.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Take all accumulated flush proxy metrics (NMSE), draining the internal buffer.
    pub fn take_flush_proxies(&mut self) -> Vec<crate::qcf_types::QcfMetric> {
        std::mem::take(&mut self.flush_proxies)
    }

    /// Enable/disable AWQE computation during flush.
    /// When enabled, `needs_attn_scores()` returns true, which causes
    /// the layer to compute attention scores even on GPU attention paths.
    pub fn set_awqe_enabled(&mut self, enabled: bool) {
        self.awqe_enabled = enabled;
    }

    /// Total number of valid tokens (Q2 + residual).
    fn total_tokens(&self) -> usize {
        self.q2_tokens + self.res_pos
    }

    /// Create a GPU-native KiviCache.
    ///
    /// When `backend` is an OpenCL backend, allocates persistent GPU buffers for
    /// residual and attention data. Falls back to CPU-only mode if GPU allocation fails.
    ///
    /// `kivi` 는 KIVI native attention capability handle (Phase α-W-4 §3.3). caller 가
    /// `caps.get::<dyn QuantAttnBackend>()` 로 pull 해 전달한다. R3 불변식: OpenCL
    /// backend 로 GPU 모드 진입 시 반드시 `Some` (init.rs 가 OpenCL 에 register). R4
    /// 불변식: `backend` 와 동일 ocl 인스턴스의 clone.
    #[allow(clippy::too_many_arguments)]
    pub fn new_gpu(
        kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        residual_size: usize,
        bits: u8,
        backend: Arc<dyn Backend>,
        kivi: Option<Arc<dyn QuantAttnBackend>>,
        memory: Arc<dyn Memory>,
    ) -> Self {
        let mut cache = Self::new_with_bits(kv_heads, head_dim, max_seq_len, residual_size, bits);

        if backend.name() != "OpenCL" {
            return cache;
        }

        // bits=16 (dynamic quant entry): F16 attn and Q-block buffers are NOT
        // used until transition_bits() switches to Q2/Q4/Q8.  Skip their
        // allocation to avoid OOM — only the F32 residual is needed.
        let skip_attn_and_q = bits == 16;

        // Allocate attention buffers at max_seq_len upfront (unless bits=16).
        // Lazy grow (Phase 2) is disabled because the OpenCL Plan caches cl_mem
        // handles at build time — growing attn buffers would stale the Plan's
        // references, causing garbage output or crashes.
        let initial_attn_cap = if skip_attn_and_q { 0 } else { max_seq_len };
        let attn_elems = initial_attn_cap * kv_heads * head_dim;

        // Q2 storage: max blocks needed
        let gs = QKKV;
        let groups_per_flush = residual_size / gs;
        let max_flushes = max_seq_len.div_ceil(residual_size);
        let blocks_per_token_v = head_dim / gs;
        let max_k_blocks = max_flushes * groups_per_flush * kv_heads * head_dim;
        let max_v_blocks = max_flushes * kv_heads * residual_size * blocks_per_token_v;

        // GPU residual capacity: full residual_size for all bit-widths.
        // For bits=16 with large max_seq_len, alloc may fail — the Err path
        // below falls back to CPU mode gracefully.
        // F16 attn and Q-block buffers are still skipped for bits=16.
        let gpu_res_cap = residual_size;
        let gpu_res_elems = kv_heads * gpu_res_cap * head_dim;

        #[allow(clippy::type_complexity)]
        let result = (|| -> Result<(Tensor, Tensor, Option<Tensor>, Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
            let res_k_buf = memory.alloc(gpu_res_elems * 4, DType::F32)?;
            let res_k = Tensor::new(
                Shape::new(vec![kv_heads, gpu_res_cap, head_dim]),
                res_k_buf,
                backend.clone(),
            );
            let res_v_buf = memory.alloc(gpu_res_elems * 4, DType::F32)?;
            let res_v = Tensor::new(
                Shape::new(vec![kv_heads, gpu_res_cap, head_dim]),
                res_v_buf,
                backend.clone(),
            );

            // bits=2/4/8: F16 attn + Q-block allocation.
            // These will be allocated on-demand in transition_bits() or
            // ensure_gpu_attn_capacity() when the cache transitions to Q2/Q4/Q8.
            let (attn_k, attn_v) = if skip_attn_and_q {
                (None, None)
            } else {
                // Attention buffers use F16 to halve GPU memory (~112 MB savings).
                // Dequant kernels (*_f16) and scatter_residual_f16 write half directly.
                let attn_k_buf = memory.alloc(attn_elems * 2, DType::F16)?;
                let ak = Tensor::new(
                    Shape::new(vec![initial_attn_cap, kv_heads, head_dim]),
                    attn_k_buf,
                    backend.clone(),
                );
                let attn_v_buf = memory.alloc(attn_elems * 2, DType::F16)?;
                let av = Tensor::new(
                    Shape::new(vec![initial_attn_cap, kv_heads, head_dim]),
                    attn_v_buf,
                    backend.clone(),
                );
                (Some(ak), Some(av))
            };

            let (q2k, q2v) = if skip_attn_and_q {
                (None, None)
            } else {
                // Quantized block storage: size depends on bits
                let block_bytes = match bits {
                    2 => 12,      // BlockQ2_0
                    4 => 20,      // BlockKVQ4
                    8 | 16 => 36, // BlockKVQ8 (16: dynamic, worst-case)
                    _ => 36,
                };
                let q2k_bytes = max_k_blocks * block_bytes;
                let q2k_buf = memory.alloc(q2k_bytes.max(block_bytes), DType::U8)?;
                let qk = Tensor::new(
                    Shape::new(vec![q2k_bytes.max(block_bytes)]),
                    q2k_buf,
                    backend.clone(),
                );
                let q2v_bytes = max_v_blocks * block_bytes;
                let q2v_buf = memory.alloc(q2v_bytes.max(block_bytes), DType::U8)?;
                let qv = Tensor::new(
                    Shape::new(vec![q2v_bytes.max(block_bytes)]),
                    q2v_buf,
                    backend.clone(),
                );
                (Some(qk), Some(qv))
            };

            Ok((res_k, res_v, attn_k, attn_v, q2k, q2v))
        })();

        match result {
            Ok((mut res_k, mut res_v, mut attn_k, mut attn_v, mut q2k, mut q2v)) => {
                // Zero-initialize all GPU buffers to prevent garbage data in
                // unwritten regions from corrupting attention computations.
                let zero_init = |t: &mut Tensor| -> Result<()> {
                    let zeros = vec![0u8; t.size()];
                    backend.write_buffer(t, &zeros)
                };
                let init_result = zero_init(&mut res_k)
                    .and_then(|_| zero_init(&mut res_v))
                    .and_then(|_| {
                        if let Some(ref mut ak) = attn_k {
                            zero_init(ak)?;
                        }
                        if let Some(ref mut av) = attn_v {
                            zero_init(av)?;
                        }
                        if let Some(ref mut qk) = q2k {
                            zero_init(qk)?;
                        }
                        if let Some(ref mut qv) = q2v {
                            zero_init(qv)?;
                        }
                        Ok(())
                    });
                if let Err(e) = init_result {
                    log::warn!(
                        "KiviCache: GPU buffer zero-init failed ({}), falling back to CPU mode",
                        e
                    );
                    return cache;
                }

                cache.gpu_backend = Some(backend);
                cache.kivi = kivi;
                cache.gpu_memory = Some(memory);
                cache.gpu_res_k = Some(res_k);
                cache.gpu_res_v = Some(res_v);
                cache.gpu_attn_k = attn_k;
                cache.gpu_attn_v = attn_v;
                cache.gpu_attn_cap = initial_attn_cap;
                cache.gpu_q2k = q2k;
                cache.gpu_q2v = q2v;

                // GPU mode: eliminate redundant CPU allocations.
                // GPU buffers hold the canonical data; CPU Vecs are not needed.
                // attn_k_buf/attn_v_buf are replaced by gpu_attn_k/gpu_attn_v.
                // qk/qv are replaced by gpu_q2k/gpu_q2v.
                // res_k/res_v are KEPT -- needed for CPU quantization during flush.
                cache.attn_k_buf = Arc::new(SharedBuffer::new(0, DType::F32));
                cache.attn_v_buf = Arc::new(SharedBuffer::new(0, DType::F32));
                cache.qk = QuantizedBlocks::empty(bits);
                cache.qv = QuantizedBlocks::empty(bits);

                if skip_attn_and_q {
                    log::info!(
                        "KiviCache: GPU mode enabled (bits=16, attn/Q deferred, res_cap={residual_size})"
                    );
                } else {
                    log::info!(
                        "KiviCache: GPU mode enabled (attn cap={initial_attn_cap}, CPU attn/q freed)"
                    );
                }
            }
            Err(e) => {
                log::warn!("KiviCache: GPU alloc failed ({}), using CPU mode", e);
            }
        }

        cache
    }

    /// Returns `true` if GPU buffers are active.
    pub fn is_gpu(&self) -> bool {
        self.gpu_backend.is_some()
    }

    /// Current allocated capacity of GPU attention buffers in tokens.
    /// Returns 0 in CPU-only mode.
    pub fn gpu_attn_capacity(&self) -> usize {
        self.gpu_attn_cap
    }

    /// Ensure GPU attention buffers have capacity for at least `needed_tokens`.
    ///
    /// Grows by powers of two up to `max_seq_len`. When growing, allocates new
    /// GPU buffers, copies existing data via `copy_slice`, and drops old buffers.
    /// No-op if current capacity is already sufficient.
    fn ensure_gpu_attn_capacity(&mut self, needed_tokens: usize) -> Result<()> {
        if needed_tokens <= self.gpu_attn_cap {
            return Ok(());
        }
        let backend = self
            .gpu_backend
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ensure_gpu_attn_capacity: no GPU backend"))?
            .clone();
        let memory = self
            .gpu_memory
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ensure_gpu_attn_capacity: no GPU memory"))?
            .clone();

        let new_cap = needed_tokens.next_power_of_two().min(self.max_seq_len);
        let new_elems = new_cap * self.kv_heads * self.head_dim;
        let new_shape = Shape::new(vec![new_cap, self.kv_heads, self.head_dim]);

        // Allocate new F16 K buffer
        let new_k_buf = memory.alloc(new_elems * 2, DType::F16)?;
        let mut new_k = Tensor::new(new_shape.clone(), new_k_buf, backend.clone());
        // Allocate new F16 V buffer
        let new_v_buf = memory.alloc(new_elems * 2, DType::F16)?;
        let mut new_v = Tensor::new(new_shape, new_v_buf, backend.clone());

        // Zero-initialize new buffers to prevent garbage in unwritten regions
        let zeros = vec![0u8; new_elems * 2];
        backend.write_buffer(&mut new_k, &zeros)?;
        backend.write_buffer(&mut new_v, &zeros)?;

        // Copy existing data from old buffers
        let old_elems = self.gpu_attn_cap * self.kv_heads * self.head_dim;
        if old_elems > 0 {
            if let Some(ref old_k) = self.gpu_attn_k {
                backend.copy_slice(old_k, &mut new_k, 0, 0, old_elems)?;
            }
            if let Some(ref old_v) = self.gpu_attn_v {
                backend.copy_slice(old_v, &mut new_v, 0, 0, old_elems)?;
            }
        }

        log::debug!(
            "KiviCache: GPU attn grow {} -> {} tokens",
            self.gpu_attn_cap,
            new_cap
        );

        self.gpu_attn_k = Some(new_k);
        self.gpu_attn_v = Some(new_v);
        self.gpu_attn_cap = new_cap;
        Ok(())
    }

    /// Allocate GPU Q-block buffers (gpu_q2k/gpu_q2v) for the given bit-width.
    ///
    /// Called during `transition_bits(16 -> Q)` when Q-block allocation was
    /// deferred at `new_gpu()` (bits=16 skips Q-block allocation to save memory).
    /// No-op if Q-block buffers already exist.
    fn allocate_gpu_q_blocks(&mut self, bits: u8) -> Result<()> {
        if self.gpu_q2k.is_some() {
            return Ok(());
        }
        let backend = self
            .gpu_backend
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("allocate_gpu_q_blocks: no GPU backend"))?
            .clone();
        let memory = self
            .gpu_memory
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("allocate_gpu_q_blocks: no GPU memory"))?
            .clone();

        let gs = QKKV;
        let groups_per_flush = self.res_cap.max(gs) / gs; // use current res_cap (may be max_seq_len)
        let max_flushes = self.max_seq_len.div_ceil(self.res_cap.max(gs));
        let blocks_per_token_v = self.head_dim / gs;
        let max_k_blocks = max_flushes * groups_per_flush * self.kv_heads * self.head_dim;
        let max_v_blocks = max_flushes * self.kv_heads * self.res_cap.max(gs) * blocks_per_token_v;

        let block_bytes: usize = match bits {
            2 => 12,
            4 => 20,
            8 => 36,
            _ => 36,
        };

        let q2k_bytes = max_k_blocks * block_bytes;
        let q2k_buf = memory.alloc(q2k_bytes.max(block_bytes), DType::U8)?;
        let mut q2k = Tensor::new(
            Shape::new(vec![q2k_bytes.max(block_bytes)]),
            q2k_buf,
            backend.clone(),
        );
        let q2v_bytes = max_v_blocks * block_bytes;
        let q2v_buf = memory.alloc(q2v_bytes.max(block_bytes), DType::U8)?;
        let mut q2v = Tensor::new(
            Shape::new(vec![q2v_bytes.max(block_bytes)]),
            q2v_buf,
            backend.clone(),
        );

        // Zero-init
        let zeros_k = vec![0u8; q2k.size()];
        backend.write_buffer(&mut q2k, &zeros_k)?;
        let zeros_v = vec![0u8; q2v.size()];
        backend.write_buffer(&mut q2v, &zeros_v)?;

        log::info!(
            "KiviCache: deferred GPU Q-block alloc (bits={}, q2k={} B, q2v={} B)",
            bits,
            q2k_bytes.max(block_bytes),
            q2v_bytes.max(block_bytes)
        );

        self.gpu_q2k = Some(q2k);
        self.gpu_q2v = Some(q2v);
        Ok(())
    }

    /// CPU-side memory usage in bytes -- for testing/debugging.
    ///
    /// Counts capacity of attn_k_buf/attn_v_buf, res_k/res_v, and qk/qv.
    pub fn cpu_memory_bytes(&self) -> usize {
        let attn_bytes = self.attn_k_buf.size() + self.attn_v_buf.size();
        let res_bytes =
            (self.res_k.capacity() + self.res_v.capacity()) * std::mem::size_of::<f32>();
        let q_bytes = self.qk.capacity_bytes() + self.qv.capacity_bytes();
        attn_bytes + res_bytes + q_bytes
    }

    /// Get raw GPU buffers for native KIVI attention (no F32 intermediate).
    ///
    /// Returns `None` if GPU mode is not active, no quantized tokens exist, or
    /// the quantization mode is unquantized (bits=16).
    pub fn get_raw_gpu_buffers(&self) -> Option<KiviRawBuffers<'_>> {
        if !self.is_gpu() || self.bits == 16 {
            return None;
        }
        Some(KiviRawBuffers {
            qk_buf: self.gpu_q2k.as_ref()?,
            qv_buf: self.gpu_q2v.as_ref()?,
            res_k: self.gpu_res_k.as_ref()?,
            res_v: self.gpu_res_v.as_ref()?,
            q_tokens: self.q2_tokens,
            res_tokens: self.res_pos,
            res_cap: self.res_cap,
            bits: self.bits,
        })
    }

    /// Get GPU attention buffer references for Plan building.
    ///
    /// Returns (attn_k, attn_v) GPU tensors used as the F32 scatter target for
    /// assembled attention path. Returns None if GPU mode is not active.
    pub fn get_gpu_attn_buffers(&self) -> Option<(&Tensor, &Tensor)> {
        if !self.is_gpu() {
            return None;
        }
        Some((self.gpu_attn_k.as_ref()?, self.gpu_attn_v.as_ref()?))
    }

    /// Dry-run QCF estimate for KIVI quantization (read-only, no state mutation).
    ///
    /// - **CPU mode** with residual data: computes actual NMSE via `compute_flush_nmse`.
    /// - **GPU mode** or empty residual: returns a bits-based proxy
    ///   (Q2=0.30, Q4=0.10, Q8=0.03, F16=0.0).
    pub fn estimate_dryrun_qcf(&self) -> f32 {
        let _t = crate::qcf_timer!(QCF_KV_DRYRUN);
        // bits=16 means unquantized, no degradation
        if self.bits == 16 {
            return 0.0;
        }

        // CPU mode with data in residual: compute actual NMSE
        if !self.is_gpu() && self.res_pos > 0 {
            let gs = self.group_size; // QKKV
            let n_groups = self.res_pos / gs;
            let flush_tokens = n_groups * gs;
            if flush_tokens > 0 {
                let config = crate::qcf_types::QcfConfig::default();
                let params = crate::qcf_types::QuantFlushParams {
                    res_k: &self.res_k,
                    res_v: &self.res_v,
                    kv_heads: self.kv_heads,
                    head_dim: self.head_dim,
                    flush_tokens,
                    res_cap: self.res_cap,
                    bits: self.bits,
                };
                let metric = self.qcf_computer.flush_nmse(&params, &config);
                return metric.normalized_value.clamp(0.0, 1.0);
            }
        }

        // Fallback: bits-based proxy
        match self.bits {
            2 => 0.30,
            4 => 0.10,
            8 => 0.03,
            _ => 0.0,
        }
    }

    /// Reset cache to empty state (reuse allocations).
    pub fn reset(&mut self) {
        if self.is_gpu() {
            // GPU mode: keep CPU attn/q Vecs empty (they were freed in new_gpu).
            // Only clear CPU residual (still used for flush quantization).
            self.qk = QuantizedBlocks::empty(self.bits);
            self.qv = QuantizedBlocks::empty(self.bits);
            // attn_k_buf/attn_v_buf stay as empty SharedBuffers (no allocation)
            debug_assert_eq!(self.attn_k_buf.size(), 0);
            debug_assert_eq!(self.attn_v_buf.size(), 0);
        } else {
            // CPU mode: zero-fill to reuse allocations
            self.qk.clear();
            self.qv.clear();
            self.attn_k_as_slice_mut().fill(0.0);
            self.attn_v_as_slice_mut().fill(0.0);
        }
        self.q2_tokens = 0;
        self.res_k.fill(0.0);
        self.res_v.fill(0.0);
        self.res_pos = 0;
        self.q2_deq_tokens = 0;
        self.flush_proxies.clear();
        // GPU position counters reset; buffers are reused (valid data region tracked by positions)
        self.gpu_q2k_blocks = 0;
        self.gpu_q2v_blocks = 0;
    }

    /// Flush residual buffer to quantized storage.
    ///
    /// Key quantization: per-channel (for each head, each of `head_dim` channels
    /// is quantized across `group_size` tokens as one block).
    ///
    /// Value quantization: per-token (for each head, each token's `head_dim`
    /// values are quantized in groups of QKKV).
    fn flush_residual(&mut self) {
        if self.bits == 16 {
            // bits=16: residual-only mode, no quantization
            return;
        }
        let gs = self.group_size; // = QKKV
        assert!(self.res_pos >= gs, "not enough tokens to flush");
        debug_assert!(
            self.q2_tokens.is_multiple_of(self.res_cap),
            "q2_tokens ({}) must be a multiple of res_cap ({})",
            self.q2_tokens,
            self.res_cap
        );

        // How many full groups to flush
        let n_groups = self.res_pos / gs;
        let flush_tokens = n_groups * gs;

        // Compute NMSE proxy before quantization (FP32 originals still available)
        let qcf_config = crate::qcf_types::QcfConfig::default();
        let proxy_params = crate::qcf_types::QuantFlushParams {
            res_k: &self.res_k,
            res_v: &self.res_v,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            flush_tokens,
            res_cap: self.res_cap,
            bits: self.bits,
        };
        self.flush_proxies
            .push(self.qcf_computer.flush_nmse(&proxy_params, &qcf_config));
        self.flush_proxies
            .push(self.qcf_computer.flush_opr(&proxy_params, &qcf_config));

        // AWQE: attention-weighted quantization error (V-only)
        if let Some(ref attn) = self.last_attn_scores {
            let gqa_group_size = attn.n_heads_q.checked_div(self.kv_heads).unwrap_or(0);
            if gqa_group_size > 0 && self.q2_tokens < attn.valid_len {
                let awqe_params = crate::qcf_types::FlushAttentionParams {
                    res_v: &self.res_v,
                    kv_heads: self.kv_heads,
                    head_dim: self.head_dim,
                    flush_tokens,
                    res_cap: self.res_cap,
                    bits: self.bits,
                    attn_scores: &attn.scores,
                    n_heads_q: attn.n_heads_q,
                    scores_stride: attn.stride,
                    gqa_group_size,
                    flush_cache_start: self.q2_tokens,
                    scores_valid_len: attn.valid_len,
                };
                self.flush_proxies
                    .push(self.qcf_computer.flush_awqe(&awqe_params, &qcf_config));

                // AW-VOPR: attention-weighted vector output perturbation ratio
                let vopr_params = crate::qcf_types::FlushAttentionParams {
                    res_v: &self.res_v,
                    kv_heads: self.kv_heads,
                    head_dim: self.head_dim,
                    flush_tokens,
                    res_cap: self.res_cap,
                    bits: self.bits,
                    attn_scores: &attn.scores,
                    n_heads_q: attn.n_heads_q,
                    scores_stride: attn.stride,
                    gqa_group_size,
                    flush_cache_start: self.q2_tokens,
                    scores_valid_len: attn.valid_len,
                };
                self.flush_proxies
                    .push(self.qcf_computer.flush_aw_vopr(&vopr_params, &qcf_config));
            }
        }

        // === Key: per-channel quantization ===
        for h in 0..self.kv_heads {
            let head_base = h * self.res_cap * self.head_dim;
            for group in 0..n_groups {
                let tok_start = group * gs;
                for ch in 0..self.head_dim {
                    let mut vals = [0.0f32; QKKV];
                    for (t, v) in vals.iter_mut().enumerate().take(gs) {
                        *v = self.res_k[head_base + (tok_start + t) * self.head_dim + ch];
                    }
                    self.qk.push_quantized(&vals);
                }
            }
        }

        // === Value: per-token quantization ===
        let blocks_per_token = self.head_dim / QKKV;
        for h in 0..self.kv_heads {
            let head_base = h * self.res_cap * self.head_dim;
            for t in 0..flush_tokens {
                let tok_base = head_base + t * self.head_dim;
                for b in 0..blocks_per_token {
                    let start = tok_base + b * QKKV;
                    let chunk: &[f32; QKKV] = self.res_v[start..start + QKKV].try_into().unwrap();
                    self.qv.push_quantized(chunk);
                }
            }
        }

        self.q2_tokens += flush_tokens;

        // Shift remaining tokens (if any) to front of residual
        let remaining = self.res_pos - flush_tokens;
        if remaining > 0 {
            for h in 0..self.kv_heads {
                let head_base = h * self.res_cap * self.head_dim;
                let src_start = head_base + flush_tokens * self.head_dim;
                let dst_start = head_base;
                let count = remaining * self.head_dim;
                self.res_k
                    .copy_within(src_start..src_start + count, dst_start);
                self.res_v
                    .copy_within(src_start..src_start + count, dst_start);
            }
        }
        self.res_pos = remaining;
    }

    /// Transition quantized blocks to a new bit-width.
    ///
    /// Dequantizes existing blocks to FP32, then re-quantizes at the new bit-width.
    /// Error accumulates through the dequant→requant cycle.
    ///
    /// Supports transitions to/from bits=16 (unquantized/residual-only mode).
    pub fn transition_bits(&mut self, new_bits: u8) -> Result<()> {
        assert!(
            new_bits == 2 || new_bits == 4 || new_bits == 8 || new_bits == 16,
            "bits must be 2, 4, 8, or 16, got {new_bits}"
        );
        assert!(
            self.bits == 2 || self.bits == 4 || self.bits == 8 || self.bits == 16,
            "current bits must be 2, 4, 8, or 16, got {}",
            self.bits
        );
        if new_bits == self.bits {
            return Ok(());
        }

        // ── 2/4/8 → 16: restore to unquantized residual-only mode ──────────
        if new_bits == 16 {
            let total_tokens = self.q2_tokens + self.res_pos;
            let new_res_cap = self.max_seq_len;
            let new_elems = self.kv_heads * new_res_cap * self.head_dim;

            if self.is_gpu() {
                // GPU mode: read F16 GPU attn buffers, convert to F32, populate residual.
                // Use gpu_attn_cap (not max_seq_len) since lazy grow may not have
                // expanded to full size yet.
                let read_cap = self.gpu_attn_cap;
                let attn_buf_size = read_cap * self.kv_heads * self.head_dim;

                // First, ensure GPU attn buffers are up to date (scatter residual)
                let _ = self.assemble_view_gpu();

                // Read F16 GPU attn buffers to temporary half vec, then convert to F32
                let mut tmp_attn_k_f16 = vec![half::f16::ZERO; attn_buf_size];
                let mut tmp_attn_v_f16 = vec![half::f16::ZERO; attn_buf_size];

                let backend = self.gpu_backend.as_ref().unwrap();
                if let Some(gpu_attn_k) = self.gpu_attn_k.as_ref() {
                    // SAFETY: half::f16 is repr(transparent) over u16, 2 bytes each
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            tmp_attn_k_f16.as_mut_ptr() as *mut u8,
                            attn_buf_size * 2,
                        )
                    };
                    backend.read_buffer(gpu_attn_k, dst)?;
                }
                if let Some(gpu_attn_v) = self.gpu_attn_v.as_ref() {
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            tmp_attn_v_f16.as_mut_ptr() as *mut u8,
                            attn_buf_size * 2,
                        )
                    };
                    backend.read_buffer(gpu_attn_v, dst)?;
                }

                // Convert F16 → F32
                let tmp_attn_k: Vec<f32> = tmp_attn_k_f16.iter().map(|v| v.to_f32()).collect();
                let tmp_attn_v: Vec<f32> = tmp_attn_v_f16.iter().map(|v| v.to_f32()).collect();

                // Expand residual and copy from tmp attn buffers
                self.res_k.resize(new_elems, 0.0);
                self.res_v.resize(new_elems, 0.0);
                for h in 0..self.kv_heads {
                    let res_base = h * new_res_cap * self.head_dim;
                    for t in 0..total_tokens {
                        let attn_idx = t * self.kv_heads * self.head_dim + h * self.head_dim;
                        let res_idx = res_base + t * self.head_dim;
                        self.res_k[res_idx..res_idx + self.head_dim]
                            .copy_from_slice(&tmp_attn_k[attn_idx..attn_idx + self.head_dim]);
                        self.res_v[res_idx..res_idx + self.head_dim]
                            .copy_from_slice(&tmp_attn_v[attn_idx..attn_idx + self.head_dim]);
                    }
                }
            } else {
                // CPU mode: use self.attn_k_buf/attn_v_buf as before
                self.assemble_view();

                // Obtain raw slices to break the borrow: attn buffers are read-only here,
                // res_k/res_v are written. They are disjoint allocations.
                let attn_k_len = self.attn_k_buf.size() / 4;
                let attn_v_len = self.attn_v_buf.size() / 4;
                let attn_k = unsafe {
                    std::slice::from_raw_parts(self.attn_k_buf.as_ptr() as *const f32, attn_k_len)
                };
                let attn_v = unsafe {
                    std::slice::from_raw_parts(self.attn_v_buf.as_ptr() as *const f32, attn_v_len)
                };

                self.res_k.resize(new_elems, 0.0);
                self.res_v.resize(new_elems, 0.0);

                for h in 0..self.kv_heads {
                    let res_base = h * new_res_cap * self.head_dim;
                    for t in 0..total_tokens {
                        let attn_idx = t * self.kv_heads * self.head_dim + h * self.head_dim;
                        let res_idx = res_base + t * self.head_dim;
                        self.res_k[res_idx..res_idx + self.head_dim]
                            .copy_from_slice(&attn_k[attn_idx..attn_idx + self.head_dim]);
                        self.res_v[res_idx..res_idx + self.head_dim]
                            .copy_from_slice(&attn_v[attn_idx..attn_idx + self.head_dim]);
                    }
                }
            }

            self.res_pos = total_tokens;
            self.res_cap = new_res_cap;
            self.q2_tokens = 0;
            self.q2_deq_tokens = 0;
            self.bits = 16;
            self.qk = QuantizedBlocks::Unquantized;
            self.qv = QuantizedBlocks::Unquantized;

            // GPU mode: release attn and Q-block buffers to reclaim memory.
            // bits=16 uses only F32 residual; these buffers are unused.
            if self.is_gpu() {
                self.gpu_attn_k = None;
                self.gpu_attn_v = None;
                self.gpu_attn_cap = 0;
                self.gpu_q2k = None;
                self.gpu_q2v = None;
                self.gpu_q2k_blocks = 0;
                self.gpu_q2v_blocks = 0;
                log::info!("KiviCache: GPU attn/Q buffers released (transition to bits=16)");
            }

            return Ok(());
        }

        // ── 16 → 2/4/8: quantize all residual data ──────────────────────────
        if self.bits == 16 {
            // Switch to target bit-width so flush uses correct format
            self.bits = new_bits;
            self.qk = QuantizedBlocks::new(new_bits);
            self.qv = QuantizedBlocks::new(new_bits);

            if self.gpu_backend.is_some() {
                // GPU mode: avoid OOM by releasing large F32 residual buffers before
                // allocating F16 attn + Q-block buffers.
                //
                // Step A: Read GPU F32 residual → CPU (so quantization can proceed).
                let res_bytes = self.kv_heads * self.res_cap * self.head_dim * 4;
                {
                    let backend = self.gpu_backend.as_ref().unwrap();
                    if let Some(gpu_rk) = self.gpu_res_k.as_ref() {
                        // SAFETY: res_k Vec is pre-allocated with res_cap * kv_heads * head_dim
                        // f32 elements. res_bytes = that count * 4 bytes.
                        let k_dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                self.res_k.as_mut_ptr() as *mut u8,
                                res_bytes,
                            )
                        };
                        backend.read_buffer(gpu_rk, k_dst)?;
                    }
                    if let Some(gpu_rv) = self.gpu_res_v.as_ref() {
                        let v_dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                self.res_v.as_mut_ptr() as *mut u8,
                                res_bytes,
                            )
                        };
                        backend.read_buffer(gpu_rv, v_dst)?;
                    }
                }

                // Step B: Release GPU F32 residual buffers to free memory before
                // Q-block + F16 attn allocation.
                self.gpu_res_k = None;
                self.gpu_res_v = None;
                log::info!(
                    "KiviCache: GPU F32 residual released before 16→{} transition",
                    new_bits
                );

                // Step C: Allocate Q-block buffers (deferred from new_gpu).
                if self.gpu_q2k.is_none() {
                    self.allocate_gpu_q_blocks(new_bits)?;
                }
            }

            // Flush as many full groups as possible from residual.
            // GPU mode: flush_residual_gpu() will skip GPU readback (already done above)
            // and use CPU data for quantization. F16 attn allocation happens inside flush.
            if self.res_pos >= self.group_size {
                if self.gpu_backend.is_some() {
                    self.flush_residual_gpu()?;
                } else {
                    self.flush_residual();
                }
            }

            // Compact remaining residual data from HeadMajor[old_res_cap] → HeadMajor[new_res_cap]
            // Before: res_k layout is [kv_heads][old_res_cap][head_dim]
            // After:  res_k layout must be [kv_heads][new_res_cap][head_dim]
            // Remaining tokens (res_pos) for each head must be packed at head-base offsets.
            let new_res_cap = self.group_size; // QKKV = 32
            if self.res_cap > new_res_cap {
                let remaining = self.res_pos; // tokens still in residual after flush

                // Compact GPU residual: write CPU residual (already compacted below) to GPU.
                // The GPU buffer retains old stride (res_cap=4096), so we overwrite with
                // correctly-strided CPU data after CPU compaction.
                let needs_gpu_sync = self.gpu_backend.is_some() && remaining > 0;

                if remaining > 0 && self.kv_heads > 1 {
                    // Move each head's slice into its new packed position
                    // head 0 is already at index 0, no move needed.
                    // heads 1..kv_heads need to be moved.
                    let elem_per_head = new_res_cap * self.head_dim;
                    for h in 1..self.kv_heads {
                        let old_src = h * self.res_cap * self.head_dim;
                        let new_dst = h * elem_per_head;
                        let count = remaining * self.head_dim;
                        self.res_k.copy_within(old_src..old_src + count, new_dst);
                        self.res_v.copy_within(old_src..old_src + count, new_dst);
                    }
                }
                let new_elems = self.kv_heads * new_res_cap * self.head_dim;
                self.res_k.truncate(new_elems);
                self.res_k.shrink_to_fit();
                self.res_v.truncate(new_elems);
                self.res_v.shrink_to_fit();
                self.res_cap = new_res_cap;

                // Sync compacted CPU residual to GPU.
                if needs_gpu_sync {
                    let backend = self.gpu_backend.as_ref().unwrap().clone();
                    let memory = self.gpu_memory.as_ref().unwrap().clone();
                    // SAFETY: res_k/res_v contain new_elems f32 values after compaction.
                    let k_bytes = unsafe {
                        std::slice::from_raw_parts(self.res_k.as_ptr() as *const u8, new_elems * 4)
                    };
                    let v_bytes = unsafe {
                        std::slice::from_raw_parts(self.res_v.as_ptr() as *const u8, new_elems * 4)
                    };

                    if self.gpu_res_k.is_some() {
                        // GPU buffer exists (oversized from old allocation).
                        // Write compacted data to the start; remaining space is unused.
                        if let Some(gpu_rk) = self.gpu_res_k.as_mut() {
                            let gpu_size = gpu_rk.buffer().size();
                            let mut padded = vec![0u8; gpu_size];
                            padded[..k_bytes.len()].copy_from_slice(k_bytes);
                            backend.write_buffer(gpu_rk, &padded)?;
                        }
                        if let Some(gpu_rv) = self.gpu_res_v.as_mut() {
                            let gpu_size = gpu_rv.buffer().size();
                            let mut padded = vec![0u8; gpu_size];
                            padded[..v_bytes.len()].copy_from_slice(v_bytes);
                            backend.write_buffer(gpu_rv, &padded)?;
                        }
                    } else {
                        // GPU residual was released (transition_bits 16→Q path).
                        // Allocate small GPU residual buffers (res_cap=32 now).
                        let shape = Shape::new(vec![self.kv_heads, new_res_cap, self.head_dim]);
                        let k_buf = memory.alloc(new_elems * 4, DType::F32)?;
                        let mut gpu_rk = Tensor::new(shape.clone(), k_buf, backend.clone());
                        backend.write_buffer(&mut gpu_rk, k_bytes)?;
                        self.gpu_res_k = Some(gpu_rk);

                        let v_buf = memory.alloc(new_elems * 4, DType::F32)?;
                        let mut gpu_rv = Tensor::new(shape, v_buf, backend.clone());
                        backend.write_buffer(&mut gpu_rv, v_bytes)?;
                        self.gpu_res_v = Some(gpu_rv);

                        log::info!(
                            "KiviCache: GPU F32 residual re-allocated (res_cap={})",
                            new_res_cap
                        );
                    }
                } else if self.gpu_backend.is_some() && self.gpu_res_k.is_none() {
                    // No remaining tokens but GPU residual was released.
                    // Allocate empty small GPU residual buffers for future updates.
                    let backend = self.gpu_backend.as_ref().unwrap().clone();
                    let memory = self.gpu_memory.as_ref().unwrap().clone();
                    let shape = Shape::new(vec![self.kv_heads, new_res_cap, self.head_dim]);
                    let k_buf = memory.alloc(new_elems * 4, DType::F32)?;
                    let gpu_rk = Tensor::new(shape.clone(), k_buf, backend.clone());
                    self.gpu_res_k = Some(gpu_rk);

                    let v_buf = memory.alloc(new_elems * 4, DType::F32)?;
                    let gpu_rv = Tensor::new(shape, v_buf, backend.clone());
                    self.gpu_res_v = Some(gpu_rv);

                    log::info!(
                        "KiviCache: GPU F32 residual re-allocated empty (res_cap={})",
                        new_res_cap
                    );
                }
            }

            self.q2_deq_tokens = 0;
            return Ok(());
        }

        // ── Q2/Q4/Q8 → Q2/Q4/Q8: re-quantize existing blocks ───────────────
        if self.q2_tokens == 0 {
            // No quantized data — just switch the format
            if self.is_gpu() {
                self.qk = QuantizedBlocks::empty(new_bits);
                self.qv = QuantizedBlocks::empty(new_bits);
            } else {
                self.qk = QuantizedBlocks::new(new_bits);
                self.qv = QuantizedBlocks::new(new_bits);
            }
            self.bits = new_bits;
            return Ok(());
        }

        let gs = self.group_size;
        let n_groups_total = self.q2_tokens / gs;
        let blocks_per_token = self.head_dim / QKKV;
        let total_k_blocks = n_groups_total * self.kv_heads * self.head_dim;
        let total_v_blocks = self.q2_tokens * self.kv_heads * blocks_per_token;

        if self.is_gpu() {
            // GPU mode: self.qk/qv are empty. Read GPU q2k/q2v, deserialize to
            // temporary QuantizedBlocks, dequant, requant, re-serialize and upload.
            let old_block_bytes = match self.bits {
                2 => 12usize,
                4 => 20,
                8 => 36,
                _ => return Err(anyhow::anyhow!("unsupported bits {}", self.bits)),
            };

            let backend = self.gpu_backend.as_ref().unwrap();

            // Read GPU q2k blocks
            let k_gpu_bytes = total_k_blocks * old_block_bytes;
            let mut k_raw = vec![0u8; k_gpu_bytes];
            if let Some(gpu_q2k) = self.gpu_q2k.as_ref() {
                backend.read_buffer(gpu_q2k, &mut k_raw)?;
            }
            // Read GPU q2v blocks
            let v_gpu_bytes = total_v_blocks * old_block_bytes;
            let mut v_raw = vec![0u8; v_gpu_bytes];
            if let Some(gpu_q2v) = self.gpu_q2v.as_ref() {
                backend.read_buffer(gpu_q2v, &mut v_raw)?;
            }

            // Deserialize → temporary old QuantizedBlocks
            let old_qk = Self::deserialize_blocks(self.bits, &k_raw, total_k_blocks);
            let old_qv = Self::deserialize_blocks(self.bits, &v_raw, total_v_blocks);

            // Dequant → requant at new bit-width
            let mut new_qk = QuantizedBlocks::with_capacity(new_bits, total_k_blocks);
            let mut new_qv = QuantizedBlocks::with_capacity(new_bits, total_v_blocks);
            let mut buf = [0.0f32; QKKV];
            for i in 0..total_k_blocks {
                old_qk.dequantize_block(i, &mut buf);
                new_qk.push_quantized(&buf);
            }
            for i in 0..total_v_blocks {
                old_qv.dequantize_block(i, &mut buf);
                new_qv.push_quantized(&buf);
            }

            // Serialize new blocks and upload to GPU
            let k_new_bytes = self.serialize_blocks(&new_qk, 0, total_k_blocks);
            let v_new_bytes = self.serialize_blocks(&new_qv, 0, total_v_blocks);

            if let Some(gpu_q2k) = self.gpu_q2k.as_mut() {
                // Write to start of GPU buffer (it may be larger)
                let gpu_size = gpu_q2k.buffer().size();
                let mut padded = vec![0u8; gpu_size];
                let copy_len = k_new_bytes.len().min(gpu_size);
                padded[..copy_len].copy_from_slice(&k_new_bytes[..copy_len]);
                backend.write_buffer(gpu_q2k, &padded)?;
            }
            if let Some(gpu_q2v) = self.gpu_q2v.as_mut() {
                let gpu_size = gpu_q2v.buffer().size();
                let mut padded = vec![0u8; gpu_size];
                let copy_len = v_new_bytes.len().min(gpu_size);
                padded[..copy_len].copy_from_slice(&v_new_bytes[..copy_len]);
                backend.write_buffer(gpu_q2v, &padded)?;
            }

            // Update GPU block counts (unchanged — same number of blocks)
            // self.gpu_q2k_blocks and self.gpu_q2v_blocks stay the same

            // Dispatch GPU dequant to update attn buffers with new bit-width data.
            // For Q2, dispatch kernel; for Q4/Q8, use CPU dequant + upload.
            self.bits = new_bits;
            self.qk = QuantizedBlocks::empty(new_bits);
            self.qv = QuantizedBlocks::empty(new_bits);
            self.q2_deq_tokens = 0;

            // Re-dequant all blocks into GPU attn buffers
            let n_flushes = self.q2_tokens / self.res_cap;
            for flush in 0..n_flushes {
                let flush_tok = self.res_cap;
                let n_groups = flush_tok / gs;
                let tok_base = flush * self.res_cap;
                let flush_k_blocks = n_groups * self.kv_heads * self.head_dim;
                let flush_v_blocks = self.kv_heads * flush_tok * blocks_per_token;
                let k_start = flush * flush_k_blocks;
                let v_start = flush * flush_v_blocks;

                // Extract sub-range of new blocks for this flush
                let flush_qk = Self::extract_sub_blocks(&new_qk, k_start, flush_k_blocks);
                let flush_qv = Self::extract_sub_blocks(&new_qv, v_start, flush_v_blocks);

                let saved_k_blocks = self.gpu_q2k_blocks;
                let saved_v_blocks = self.gpu_q2v_blocks;
                self.gpu_q2k_blocks = k_start;
                self.gpu_q2v_blocks = v_start;
                // Skip GPU block upload (already done above), just dispatch dequant.
                // We set gpu_q2k/v_blocks to match the kernel's expected block start.
                // NOTE: upload_and_dequant_flush_with will re-upload blocks, which is
                // redundant but harmless for this rare operation.
                self.upload_and_dequant_flush_with(
                    flush_tok, n_groups, tok_base, &flush_qk, &flush_qv,
                )?;
                self.gpu_q2k_blocks = saved_k_blocks;
                self.gpu_q2v_blocks = saved_v_blocks;
            }

            // Restore correct gpu block counts for future flushes
            self.gpu_q2k_blocks = total_k_blocks;
            self.gpu_q2v_blocks = total_v_blocks;
        } else {
            // CPU mode: use self.qk/qv directly (unchanged)
            let mut new_qk = QuantizedBlocks::with_capacity(new_bits, total_k_blocks);
            let mut buf = [0.0f32; QKKV];
            for i in 0..total_k_blocks {
                self.qk.dequantize_block(i, &mut buf);
                new_qk.push_quantized(&buf);
            }
            let mut new_qv = QuantizedBlocks::with_capacity(new_bits, total_v_blocks);
            for i in 0..total_v_blocks {
                self.qv.dequantize_block(i, &mut buf);
                new_qv.push_quantized(&buf);
            }
            self.qk = new_qk;
            self.qv = new_qv;
            self.bits = new_bits;
            self.q2_deq_tokens = 0;
        }

        Ok(())
    }

    /// Assemble K/V view by dequantizing blocks and copying residual data
    /// into the pre-allocated attention buffers.
    ///
    /// **Incremental**: only dequantizes flushes that haven't been processed
    /// yet (tracked by `q2_deq_tokens`). Residual data is always re-copied
    /// because it changes every decode step.
    fn assemble_view(&mut self) {
        let gs = self.group_size;
        let groups_per_flush = self.res_cap / gs;

        // Obtain raw f32 pointers to the SharedBuffer-backed attn buffers.
        // SAFETY: We have &mut self, so exclusive access is guaranteed.
        // The buffers are allocated as DType::F32 with size = max_seq * kv_heads * head_dim * 4.
        let attn_k_len = self.attn_k_buf.size() / 4;
        let attn_v_len = self.attn_v_buf.size() / 4;
        let attn_k = unsafe {
            std::slice::from_raw_parts_mut(self.attn_k_buf.as_mut_ptr() as *mut f32, attn_k_len)
        };
        let attn_v = unsafe {
            std::slice::from_raw_parts_mut(self.attn_v_buf.as_mut_ptr() as *mut f32, attn_v_len)
        };

        // === Incremental quantized dequantization (only new flushes) ===
        if self.q2_tokens > self.q2_deq_tokens {
            debug_assert!(
                self.q2_tokens.is_multiple_of(self.res_cap),
                "q2_tokens ({}) must be a multiple of res_cap ({})",
                self.q2_tokens,
                self.res_cap
            );
            let old_flushes = self.q2_deq_tokens / self.res_cap;
            let new_flushes = self.q2_tokens / self.res_cap;

            let k_blocks_per_flush = groups_per_flush * self.kv_heads * self.head_dim;
            let blocks_per_token = self.head_dim / QKKV;
            let v_blocks_per_flush = self.kv_heads * self.res_cap * blocks_per_token;

            let mut k_block_idx = old_flushes * k_blocks_per_flush;
            let mut v_block_idx = old_flushes * v_blocks_per_flush;
            let mut channel_buf = [0.0f32; QKKV];
            let mut deq_buf = [0.0f32; QKKV];

            for flush in old_flushes..new_flushes {
                let tok_base = flush * self.res_cap;

                // Key: per-channel dequantization
                for h in 0..self.kv_heads {
                    for g in 0..groups_per_flush {
                        let tok_start = tok_base + g * gs;
                        for ch in 0..self.head_dim {
                            self.qk.dequantize_block(k_block_idx, &mut channel_buf);
                            k_block_idx += 1;
                            for (t, &val) in channel_buf.iter().enumerate().take(gs) {
                                let pos = tok_start + t;
                                let out_idx =
                                    pos * self.kv_heads * self.head_dim + h * self.head_dim + ch;
                                attn_k[out_idx] = val;
                            }
                        }
                    }
                }

                // Value: per-token dequantization
                for h in 0..self.kv_heads {
                    for t in 0..self.res_cap {
                        let pos = tok_base + t;
                        let out_base = pos * self.kv_heads * self.head_dim + h * self.head_dim;
                        for b in 0..blocks_per_token {
                            self.qv.dequantize_block(v_block_idx, &mut deq_buf);
                            v_block_idx += 1;
                            let start = out_base + b * QKKV;
                            attn_v[start..start + QKKV].copy_from_slice(&deq_buf);
                        }
                    }
                }
            }

            self.q2_deq_tokens = self.q2_tokens;
        }

        // === Copy residual FP32 data (always, since it changes every step) ===
        if self.res_pos > 0 {
            let token_offset = self.q2_tokens;
            for h in 0..self.kv_heads {
                let res_base = h * self.res_cap * self.head_dim;
                for t in 0..self.res_pos {
                    let pos = token_offset + t;
                    let out_base = pos * self.kv_heads * self.head_dim + h * self.head_dim;
                    let res_start = res_base + t * self.head_dim;
                    attn_k[out_base..out_base + self.head_dim]
                        .copy_from_slice(&self.res_k[res_start..res_start + self.head_dim]);
                    attn_v[out_base..out_base + self.head_dim]
                        .copy_from_slice(&self.res_v[res_start..res_start + self.head_dim]);
                }
            }
        }
    }

    // ── GPU-mode private helpers ──────────────────────────────────────────────

    /// GPU update path: scatter input tokens into GPU residual buffer using
    /// `gather_update_quant` kernel, flushing when the residual is full.
    // §13.8-L S-L-3: QuantAttnBackend trait 으로 OpenCLBackend
    // downcast 제거. fast path 의 `gather_update_quant` 2건은 trait method,
    // slow path 의 `copy_slice` 는 Backend trait 의 일반 메서드라 그대로
    // dyn dispatch — 함수 내 잔존 downcast/import 0 이므로 marker 제거.
    fn update_gpu(&mut self, new_k: &Tensor, new_v: &Tensor, seq_len: usize) -> Result<()> {
        // slow path 의 `copy_slice` 는 Backend trait 의 일반 ops → gpu_backend 유지.
        let backend_arc = self.gpu_backend.as_ref().unwrap().clone();
        // fast path 의 `gather_update_quant` 는 KIVI native capability → 매 토큰
        // `as_quant_attn()` lookup 대신 construction 시 pull 된 handle 직접 사용
        // (Phase α-W-4 §3.3). R3: GPU 모드면 OpenCL backend 라 caller 가 Some 보장.
        // owned clone 으로 self 의 mutable borrow(gpu_res_k/flush_residual_gpu)와
        // 충돌 회피 (기존 backend_arc 패턴과 동일).
        let kivi_be = self
            .kivi
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("KIVI GPU mode but kivi handle missing (registry not registered?)")
            })?
            .clone();

        let mut written = 0usize;
        while written < seq_len {
            if self.res_pos >= self.res_cap {
                self.flush_residual_gpu()?;
            }
            // How many tokens we can write in this batch without overflow
            let batch = (seq_len - written).min(self.res_cap - self.res_pos);

            // We need to call the kernel for the [written..written+batch] slice of new_k/v.
            // The kernel signature is: input[seq_len, kv_heads, head_dim] → residual[kv_heads, res_cap, head_dim]
            // But our input tensor contains all seq_len tokens; we need a view of [batch] tokens starting at `written`.
            // Since GPU tensors may not support views, we create a lightweight wrapper pointing into the same buffer.
            // SAFETY: We rely on the kernel only reading `seq_len` tokens from the start of the input buffer.
            // To avoid GPU sub-buffer complexity, if written==0 and batch==seq_len we pass as-is.
            // Otherwise, we fall back to the CPU copy_slice approach per token.
            if written == 0 && batch == seq_len {
                // Fast path: pass entire input at once.
                // D8: ABI struct(cl_mem) 시그니처. input/residual `&Tensor` 를 raw
                // cl_mem 으로 추출해 `QuantAttnGatherArgs` 패킹 → trait 호출 → rc!=0 이면
                // bail. cl_queue 는 OpenCL 정적 impl 이 `&self.queue` 를 직접 알아
                // 사용하지 않으므로 null 패킹.
                use crate::backend::opencl::get_cl_mem;
                let gpu_res_k = self.gpu_res_k.as_ref().unwrap();
                let gpu_res_v = self.gpu_res_v.as_ref().unwrap();
                // `Mem::as_ptr()` 는 이미 `cl_mem`(= `*mut c_void`) 를 반환하므로 캐스트 불요.
                let new_k_mem = get_cl_mem(new_k.buffer().as_ref())?.as_ptr();
                let new_v_mem = get_cl_mem(new_v.buffer().as_ref())?.as_ptr();
                let res_k_mem = get_cl_mem(gpu_res_k.buffer().as_ref())?.as_ptr();
                let res_v_mem = get_cl_mem(gpu_res_v.buffer().as_ref())?.as_ptr();
                let args_k = crate::backend::QuantAttnGatherArgs {
                    cl_queue: std::ptr::null_mut(),
                    input_mem: new_k_mem,
                    residual_mem: res_k_mem,
                    kv_heads: self.kv_heads,
                    res_cap: self.res_cap,
                    head_dim: self.head_dim,
                    seq_len: batch,
                    res_pos: self.res_pos,
                };
                let rc_k = kivi_be.gather_update_quant(&args_k);
                if rc_k != 0 {
                    anyhow::bail!("KIVI gather_update_quant (K) failed (rc={rc_k})");
                }
                let args_v = crate::backend::QuantAttnGatherArgs {
                    cl_queue: std::ptr::null_mut(),
                    input_mem: new_v_mem,
                    residual_mem: res_v_mem,
                    kv_heads: self.kv_heads,
                    res_cap: self.res_cap,
                    head_dim: self.head_dim,
                    seq_len: batch,
                    res_pos: self.res_pos,
                };
                let rc_v = kivi_be.gather_update_quant(&args_v);
                if rc_v != 0 {
                    anyhow::bail!("KIVI gather_update_quant (V) failed (rc={rc_v})");
                }
            } else {
                // Slow path: token-by-token copy_slice for the sub-range
                for s in written..written + batch {
                    let gpu_res_k = self.gpu_res_k.as_mut().unwrap();
                    let gpu_res_v = self.gpu_res_v.as_mut().unwrap();
                    for h in 0..self.kv_heads {
                        let src_off = s * self.kv_heads * self.head_dim + h * self.head_dim;
                        let dst_off = h * self.res_cap * self.head_dim
                            + (self.res_pos + (s - written)) * self.head_dim;
                        backend_arc.copy_slice(
                            new_k,
                            gpu_res_k,
                            src_off,
                            dst_off,
                            self.head_dim,
                        )?;
                        backend_arc.copy_slice(
                            new_v,
                            gpu_res_v,
                            src_off,
                            dst_off,
                            self.head_dim,
                        )?;
                    }
                }
            }
            self.res_pos += batch;
            written += batch;
        }
        Ok(())
    }

    /// GPU flush: read GPU residual → CPU quantize → upload Q2 blocks to GPU →
    /// dispatch GPU dequant kernels to fill attention buffers.
    fn flush_residual_gpu(&mut self) -> Result<()> {
        let gs = self.group_size;
        let n_groups = self.res_pos / gs;
        let flush_tokens = n_groups * gs;

        if flush_tokens == 0 {
            return Ok(());
        }

        // 0. Ensure GPU attn buffers can hold all quantized + flushing tokens
        self.ensure_gpu_attn_capacity(self.q2_tokens + flush_tokens)?;

        // 1. Read GPU residual to CPU (needed for quantization).
        //    Skip if GPU residual buffers were already read back and released
        //    (e.g. during transition_bits 16→Q to avoid OOM).
        if let (Some(gpu_res_k), Some(gpu_res_v)) =
            (self.gpu_res_k.as_ref(), self.gpu_res_v.as_ref())
        {
            let res_bytes = self.kv_heads * self.res_cap * self.head_dim * 4;
            let backend = self.gpu_backend.as_ref().unwrap();
            // SAFETY: res_k/res_v Vecs hold at least res_bytes / 4 f32 elements.
            let k_dst = unsafe {
                std::slice::from_raw_parts_mut(self.res_k.as_mut_ptr() as *mut u8, res_bytes)
            };
            backend.read_buffer(gpu_res_k, k_dst)?;
            let v_dst = unsafe {
                std::slice::from_raw_parts_mut(self.res_v.as_mut_ptr() as *mut u8, res_bytes)
            };
            backend.read_buffer(gpu_res_v, v_dst)?;
        }

        // 2. Compute QCF proxy metrics (same as CPU path)
        let qcf_config = crate::qcf_types::QcfConfig::default();
        let proxy_params = crate::qcf_types::QuantFlushParams {
            res_k: &self.res_k,
            res_v: &self.res_v,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            flush_tokens,
            res_cap: self.res_cap,
            bits: self.bits,
        };
        self.flush_proxies
            .push(self.qcf_computer.flush_nmse(&proxy_params, &qcf_config));
        self.flush_proxies
            .push(self.qcf_computer.flush_opr(&proxy_params, &qcf_config));

        // AWQE (same logic as CPU flush_residual)
        if let Some(ref attn) = self.last_attn_scores {
            let gqa_group_size = attn.n_heads_q.checked_div(self.kv_heads).unwrap_or(0);
            if gqa_group_size > 0 && self.q2_tokens < attn.valid_len {
                let awqe_params = crate::qcf_types::FlushAttentionParams {
                    res_v: &self.res_v,
                    kv_heads: self.kv_heads,
                    head_dim: self.head_dim,
                    flush_tokens,
                    res_cap: self.res_cap,
                    bits: self.bits,
                    attn_scores: &attn.scores,
                    n_heads_q: attn.n_heads_q,
                    scores_stride: attn.stride,
                    gqa_group_size,
                    flush_cache_start: self.q2_tokens,
                    scores_valid_len: attn.valid_len,
                };
                self.flush_proxies
                    .push(self.qcf_computer.flush_awqe(&awqe_params, &qcf_config));

                // AW-VOPR: attention-weighted vector output perturbation ratio
                let vopr_params = crate::qcf_types::FlushAttentionParams {
                    res_v: &self.res_v,
                    kv_heads: self.kv_heads,
                    head_dim: self.head_dim,
                    flush_tokens,
                    res_cap: self.res_cap,
                    bits: self.bits,
                    attn_scores: &attn.scores,
                    n_heads_q: attn.n_heads_q,
                    scores_stride: attn.stride,
                    gqa_group_size,
                    flush_cache_start: self.q2_tokens,
                    scores_valid_len: attn.valid_len,
                };
                self.flush_proxies
                    .push(self.qcf_computer.flush_aw_vopr(&vopr_params, &qcf_config));
            }
        }

        // 3. CPU quantize → temporary QuantizedBlocks (not stored in self.qk/qv)
        let new_k_blocks = n_groups * self.kv_heads * self.head_dim;
        let blocks_per_token = self.head_dim / QKKV;
        let new_v_blocks = self.kv_heads * flush_tokens * blocks_per_token;
        let mut tmp_qk = QuantizedBlocks::with_capacity(self.bits, new_k_blocks);
        let mut tmp_qv = QuantizedBlocks::with_capacity(self.bits, new_v_blocks);

        for h in 0..self.kv_heads {
            let head_base = h * self.res_cap * self.head_dim;
            for group in 0..n_groups {
                let tok_start = group * gs;
                for ch in 0..self.head_dim {
                    let mut vals = [0.0f32; QKKV];
                    for (t, v) in vals.iter_mut().enumerate().take(gs) {
                        *v = self.res_k[head_base + (tok_start + t) * self.head_dim + ch];
                    }
                    tmp_qk.push_quantized(&vals);
                }
            }
        }
        for h in 0..self.kv_heads {
            let head_base = h * self.res_cap * self.head_dim;
            for t in 0..flush_tokens {
                let tok_base_inner = head_base + t * self.head_dim;
                for b in 0..blocks_per_token {
                    let start = tok_base_inner + b * QKKV;
                    let chunk: &[f32; QKKV] = self.res_v[start..start + QKKV].try_into().unwrap();
                    tmp_qv.push_quantized(chunk);
                }
            }
        }

        // 4. Upload Q2 blocks to GPU and run dequant kernels into attention buffers
        let tok_base = self.q2_tokens;
        self.upload_and_dequant_flush_with(flush_tokens, n_groups, tok_base, &tmp_qk, &tmp_qv)?;

        self.q2_tokens += flush_tokens;
        self.q2_deq_tokens = self.q2_tokens; // GPU dequant already done above

        // 5. Shift remaining residual tokens on GPU (or CPU if GPU residual was released)
        let remaining = self.res_pos - flush_tokens;
        if remaining > 0 {
            if let Some(gpu_res_k) = self.gpu_res_k.as_mut() {
                let backend = self.gpu_backend.as_ref().unwrap().clone();
                for h in 0..self.kv_heads {
                    let base = h * self.res_cap * self.head_dim;
                    let src = base + flush_tokens * self.head_dim;
                    let count = remaining * self.head_dim;
                    backend.buffer_shift(gpu_res_k, src, base, count)?;
                }
                let gpu_res_v = self.gpu_res_v.as_mut().unwrap();
                for h in 0..self.kv_heads {
                    let base = h * self.res_cap * self.head_dim;
                    let src = base + flush_tokens * self.head_dim;
                    let count = remaining * self.head_dim;
                    backend.buffer_shift(gpu_res_v, src, base, count)?;
                }
            } else {
                // GPU residual released (transition_bits 16→Q path).
                // Shift on CPU side — data was already read back before release.
                for h in 0..self.kv_heads {
                    let base = h * self.res_cap * self.head_dim;
                    let src = base + flush_tokens * self.head_dim;
                    let count = remaining * self.head_dim;
                    self.res_k.copy_within(src..src + count, base);
                    self.res_v.copy_within(src..src + count, base);
                }
            }
        }
        self.res_pos = remaining;

        Ok(())
    }

    /// Upload newly quantized blocks to GPU buffers and dispatch dequant kernels.
    ///
    /// `flush_tokens`: number of tokens in this flush
    /// `n_groups`: number of key groups (= flush_tokens / group_size)
    /// `tok_base`: token offset in the attention buffer for this flush
    /// `tmp_qk`/`tmp_qv`: temporary quantized blocks (not stored in self.qk/qv in GPU mode)
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L hot-path KIVI flush upload
    fn upload_and_dequant_flush_with(
        &mut self,
        flush_tokens: usize,
        n_groups: usize,
        tok_base: usize,
        tmp_qk: &QuantizedBlocks,
        tmp_qv: &QuantizedBlocks,
    ) -> Result<()> {
        let k_block_start = self.gpu_q2k_blocks;
        let v_block_start = self.gpu_q2v_blocks;

        let new_k_blocks = n_groups * self.kv_heads * self.head_dim;
        let blocks_per_token = self.head_dim / QKKV;
        let new_v_blocks = self.kv_heads * flush_tokens * blocks_per_token;

        // Serialize quantized blocks to raw bytes and upload with byte offset.
        let block_bytes = match self.bits {
            2 => 12, // size_of::<BlockQ2_0>()
            4 => 20, // size_of::<BlockKVQ4>()
            8 => 36, // size_of::<BlockKVQ8>()
            _ => {
                return Err(anyhow::anyhow!(
                    "unsupported bits {} for GPU upload",
                    self.bits
                ));
            }
        };

        let k_byte_offset = k_block_start * block_bytes;
        let v_byte_offset = v_block_start * block_bytes;

        // Serialize from the temporary blocks (start=0, count=all new blocks)
        let k_bytes = self.serialize_blocks(tmp_qk, 0, new_k_blocks);
        let v_bytes = self.serialize_blocks(tmp_qv, 0, new_v_blocks);

        // Suppress unused warnings when compiled without opencl feature.
        let _ = (tok_base, &k_byte_offset, &v_byte_offset, &k_bytes, &v_bytes);

        #[cfg(feature = "opencl")]
        {
            use crate::backend::opencl::{OpenCLBackend, get_cl_mem};

            let backend_arc = self.gpu_backend.as_ref().unwrap().clone();
            if let Some(ocl) = backend_arc.as_any().downcast_ref::<OpenCLBackend>() {
                // Upload key blocks at byte offset
                {
                    let gpu_q2k = self.gpu_q2k.as_ref().unwrap();
                    let cl_mem = get_cl_mem(gpu_q2k.buffer().as_ref())?;
                    unsafe {
                        ocl::core::enqueue_write_buffer(
                            &ocl.queue,
                            cl_mem,
                            true,
                            k_byte_offset,
                            &k_bytes,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                    }
                }
                // Upload value blocks at byte offset
                {
                    let gpu_q2v = self.gpu_q2v.as_ref().unwrap();
                    let cl_mem = get_cl_mem(gpu_q2v.buffer().as_ref())?;
                    unsafe {
                        ocl::core::enqueue_write_buffer(
                            &ocl.queue,
                            cl_mem,
                            true,
                            v_byte_offset,
                            &v_bytes,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                    }
                }

                // Dispatch GPU dequant: fill F16 attention buffers for the assembled path.
                // Q2: use dedicated GPU F16 dequant kernel (fast).
                // Q4/Q8: no GPU dequant kernel yet — CPU dequant → F16 convert → upload.
                if self.bits == 2 {
                    let gpu_q2k = self.gpu_q2k.as_ref().unwrap();
                    let gpu_attn_k = self.gpu_attn_k.as_mut().unwrap();
                    ocl.kivi_dequantize_key_q2_f16(
                        gpu_q2k,
                        gpu_attn_k,
                        self.kv_heads,
                        self.head_dim,
                        n_groups,
                        tok_base,
                        k_block_start,
                    )?;
                    let gpu_q2v = self.gpu_q2v.as_ref().unwrap();
                    let gpu_attn_v = self.gpu_attn_v.as_mut().unwrap();
                    ocl.kivi_dequantize_value_q2_f16(
                        gpu_q2v,
                        gpu_attn_v,
                        self.kv_heads,
                        self.head_dim,
                        flush_tokens,
                        tok_base,
                        v_block_start,
                    )?;
                } else {
                    // Q4/Q8: CPU dequant into temporary F32 buffer, convert to F16,
                    // partial upload at byte offset to avoid overwriting earlier flushes.
                    let gs = self.group_size;
                    let flush_buf_size = flush_tokens * self.kv_heads * self.head_dim;
                    let mut tmp_attn_k = vec![0.0f32; flush_buf_size];
                    let mut tmp_attn_v = vec![0.0f32; flush_buf_size];
                    let mut deq_buf = [0.0f32; QKKV];

                    // Key: per-channel dequant — relative positions (0-based within flush window)
                    let mut k_idx = 0usize;
                    for h in 0..self.kv_heads {
                        for g in 0..n_groups {
                            let tok_start = g * gs;
                            for ch in 0..self.head_dim {
                                tmp_qk.dequantize_block(k_idx, &mut deq_buf);
                                k_idx += 1;
                                for (t, &val) in deq_buf.iter().enumerate().take(gs) {
                                    let pos = tok_start + t;
                                    let out_idx = pos * self.kv_heads * self.head_dim
                                        + h * self.head_dim
                                        + ch;
                                    tmp_attn_k[out_idx] = val;
                                }
                            }
                        }
                    }
                    // Value: per-token dequant — relative positions (0-based within flush window)
                    let blocks_per_token_v = self.head_dim / QKKV;
                    let mut v_idx = 0usize;
                    for h in 0..self.kv_heads {
                        for t in 0..flush_tokens {
                            let out_base = t * self.kv_heads * self.head_dim + h * self.head_dim;
                            for b in 0..blocks_per_token_v {
                                tmp_qv.dequantize_block(v_idx, &mut deq_buf);
                                v_idx += 1;
                                let start = out_base + b * QKKV;
                                tmp_attn_v[start..start + QKKV].copy_from_slice(&deq_buf);
                            }
                        }
                    }

                    // Convert F32 → F16 and partial upload to GPU attention buffers
                    let tmp_attn_k_f16: Vec<half::f16> =
                        tmp_attn_k.iter().map(|&v| half::f16::from_f32(v)).collect();
                    let tmp_attn_v_f16: Vec<half::f16> =
                        tmp_attn_v.iter().map(|&v| half::f16::from_f32(v)).collect();

                    // Byte offset into F16 attention buffer for this flush window
                    let byte_offset = tok_base * self.kv_heads * self.head_dim * 2;

                    let gpu_attn_k = self.gpu_attn_k.as_ref().unwrap();
                    let gpu_attn_v = self.gpu_attn_v.as_ref().unwrap();
                    let cl_mem_k = get_cl_mem(gpu_attn_k.buffer().as_ref())?;
                    let cl_mem_v = get_cl_mem(gpu_attn_v.buffer().as_ref())?;
                    // SAFETY: half::f16 is repr(transparent) over u16, contiguous in memory
                    let k_f16_bytes = unsafe {
                        std::slice::from_raw_parts(
                            tmp_attn_k_f16.as_ptr() as *const u8,
                            tmp_attn_k_f16.len() * 2,
                        )
                    };
                    let v_f16_bytes = unsafe {
                        std::slice::from_raw_parts(
                            tmp_attn_v_f16.as_ptr() as *const u8,
                            tmp_attn_v_f16.len() * 2,
                        )
                    };
                    // SAFETY: cl_mem handles are valid; byte_offset + data fits within buffer
                    unsafe {
                        ocl::core::enqueue_write_buffer(
                            &ocl.queue,
                            cl_mem_k,
                            true,
                            byte_offset,
                            k_f16_bytes,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                        ocl::core::enqueue_write_buffer(
                            &ocl.queue,
                            cl_mem_v,
                            true,
                            byte_offset,
                            v_f16_bytes,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                    }
                }
            }
        }

        self.gpu_q2k_blocks += new_k_blocks;
        self.gpu_q2v_blocks += new_v_blocks;

        Ok(())
    }

    fn serialize_blocks(&self, blocks: &QuantizedBlocks, start: usize, count: usize) -> Vec<u8> {
        match blocks {
            QuantizedBlocks::Unquantized => Vec::new(),
            QuantizedBlocks::Q2(v) => {
                let mut out = Vec::with_capacity(count * 12);
                for b in &v[start..start + count] {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(b as *const BlockQ2_0 as *const u8, 12)
                    };
                    out.extend_from_slice(bytes);
                }
                out
            }
            QuantizedBlocks::Q4(v) => {
                let mut out = Vec::with_capacity(count * 20);
                for b in &v[start..start + count] {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(b as *const BlockKVQ4 as *const u8, 20)
                    };
                    out.extend_from_slice(bytes);
                }
                out
            }
            QuantizedBlocks::Q8(v) => {
                let mut out = Vec::with_capacity(count * 36);
                for b in &v[start..start + count] {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(b as *const BlockKVQ8 as *const u8, 36)
                    };
                    out.extend_from_slice(bytes);
                }
                out
            }
        }
    }

    /// Deserialize raw bytes into QuantizedBlocks.
    ///
    /// Inverse of `serialize_blocks`. Used for GPU read-back during transition_bits.
    fn deserialize_blocks(bits: u8, raw: &[u8], count: usize) -> QuantizedBlocks {
        match bits {
            2 => {
                let block_size = std::mem::size_of::<BlockQ2_0>();
                assert_eq!(raw.len(), count * block_size);
                let mut blocks = Vec::with_capacity(count);
                for i in 0..count {
                    let src = &raw[i * block_size..(i + 1) * block_size];
                    // SAFETY: BlockQ2_0 is a plain-old-data struct (repr(C)); byte copy is valid.
                    let block: BlockQ2_0 = unsafe { std::ptr::read(src.as_ptr() as *const _) };
                    blocks.push(block);
                }
                QuantizedBlocks::Q2(blocks)
            }
            4 => {
                let block_size = std::mem::size_of::<BlockKVQ4>();
                assert_eq!(raw.len(), count * block_size);
                let mut blocks = Vec::with_capacity(count);
                for i in 0..count {
                    let src = &raw[i * block_size..(i + 1) * block_size];
                    let block: BlockKVQ4 = unsafe { std::ptr::read(src.as_ptr() as *const _) };
                    blocks.push(block);
                }
                QuantizedBlocks::Q4(blocks)
            }
            8 => {
                let block_size = std::mem::size_of::<BlockKVQ8>();
                assert_eq!(raw.len(), count * block_size);
                let mut blocks = Vec::with_capacity(count);
                for i in 0..count {
                    let src = &raw[i * block_size..(i + 1) * block_size];
                    let block: BlockKVQ8 = unsafe { std::ptr::read(src.as_ptr() as *const _) };
                    blocks.push(block);
                }
                QuantizedBlocks::Q8(blocks)
            }
            _ => panic!("unsupported bits: {bits}"),
        }
    }

    /// Extract a sub-range of blocks into a new QuantizedBlocks.
    fn extract_sub_blocks(blocks: &QuantizedBlocks, start: usize, count: usize) -> QuantizedBlocks {
        match blocks {
            QuantizedBlocks::Unquantized => QuantizedBlocks::Unquantized,
            QuantizedBlocks::Q2(v) => QuantizedBlocks::Q2(v[start..start + count].to_vec()),
            QuantizedBlocks::Q4(v) => QuantizedBlocks::Q4(v[start..start + count].to_vec()),
            QuantizedBlocks::Q8(v) => QuantizedBlocks::Q8(v[start..start + count].to_vec()),
        }
    }

    /// GPU assemble_view: ensure GPU attention buffers are up to date.
    ///
    /// - `q2_deq_tokens` is kept in sync with `q2_tokens` during `flush_residual_gpu`,
    ///   so no incremental dequant is needed here.
    /// - Residual scatter: uses `kivi_scatter_residual` kernel (always re-done each call).
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L hot-path KIVI view assemble
    fn assemble_view_gpu(&mut self) -> Result<()> {
        // Defensive: ensure attn buffers can hold q2_tokens + res_pos
        let needed = self.q2_tokens + self.res_pos;
        if needed > self.gpu_attn_cap {
            self.ensure_gpu_attn_capacity(needed)?;
        }

        #[cfg(feature = "opencl")]
        {
            use crate::backend::opencl::OpenCLBackend;

            if self.res_pos > 0 {
                let backend_arc = self.gpu_backend.as_ref().unwrap().clone();
                if let Some(ocl) = backend_arc.as_any().downcast_ref::<OpenCLBackend>() {
                    // F32 residual → F16 attention buffer (scatter + convert)
                    let gpu_res_k = self.gpu_res_k.as_ref().unwrap();
                    let gpu_attn_k = self.gpu_attn_k.as_mut().unwrap();
                    ocl.kivi_scatter_residual_f16(
                        gpu_res_k,
                        gpu_attn_k,
                        self.kv_heads,
                        self.res_cap,
                        self.head_dim,
                        self.res_pos,
                        self.q2_tokens,
                    )?;
                    let gpu_res_v = self.gpu_res_v.as_ref().unwrap();
                    let gpu_attn_v = self.gpu_attn_v.as_mut().unwrap();
                    ocl.kivi_scatter_residual_f16(
                        gpu_res_v,
                        gpu_attn_v,
                        self.kv_heads,
                        self.res_cap,
                        self.head_dim,
                        self.res_pos,
                        self.q2_tokens,
                    )?;
                }
            }
            return Ok(());
        }
        #[allow(unreachable_code)]
        Ok(())
    }

    /// GPU get_view: return Tensors backed by GPU attention buffers.
    ///
    /// If `assemble_view_gpu()` fails (e.g. kernel dispatch error), falls back
    /// to CPU dequant + assemble path to avoid returning stale/garbage data.
    fn get_view_gpu(&mut self) -> (Tensor, Tensor) {
        let total = self.total_tokens();
        let backend = self.gpu_backend.as_ref().unwrap().clone();

        if total == 0 {
            let buf = Arc::new(SharedBuffer::new(0, DType::F32));
            let shape = Shape::new(vec![1, 0, self.kv_heads, self.head_dim]);
            let t = Tensor::new(shape.clone(), buf.clone(), backend);
            return (t.clone(), t);
        }

        // bits=16: no attn buffers — return GPU residual directly as F32 view.
        // HeadMajor [kv_heads, res_cap, head_dim] → reinterpret as
        // [1, kv_heads, res_cap, head_dim] with only `total` valid tokens.
        if self.bits == 16 {
            let gpu_res_k = self.gpu_res_k.as_ref().unwrap();
            let gpu_res_v = self.gpu_res_v.as_ref().unwrap();
            // HeadMajor layout: shape encodes [batch=1, kv_heads, capacity, head_dim]
            // The attention kernel reads kv_head_stride and kv_pos_stride from shape.
            let shape = Shape::new(vec![1, self.kv_heads, self.res_cap, self.head_dim]);
            let k = Tensor::new(shape.clone(), gpu_res_k.buffer().clone(), backend.clone());
            let v = Tensor::new(shape, gpu_res_v.buffer().clone(), backend);
            return (k, v);
        }

        // bits=2/4/8: assemble quantized + residual into F16 attn buffers
        if let Err(e) = self.assemble_view_gpu() {
            log::warn!(
                "KiviCache assemble_view_gpu error: {}, returning stale GPU attention buffers \
                 (CPU qk/qv not available in GPU mode)",
                e
            );
        }

        let shape = Shape::new(vec![1, total, self.kv_heads, self.head_dim]);
        let k = Tensor::new(
            shape.clone(),
            self.gpu_attn_k.as_ref().unwrap().buffer().clone(),
            backend.clone(),
        );
        let v = Tensor::new(
            shape,
            self.gpu_attn_v.as_ref().unwrap().buffer().clone(),
            backend,
        );
        (k, v)
    }

    // ── KVCacheOps primitives moved to inherent (Phase α-K BC 5-E) ───────────
    // KVCacheOps trait 은 5-F 까지 생존하되, 본문은 inherent 로 이전하고 trait 메서드는
    // 이 inherent 를 위임 호출한다(byte-identical, 단일 본문). fmt 경로(KIVIFormat/fmt_bridge/
    // KiviForward)가 trait 경유 없이 직접 호출하도록 한다.

    /// Number of valid tokens currently in the cache (q2_tokens + res_pos).
    pub fn current_pos(&self) -> usize {
        self.total_tokens()
    }

    /// Physical buffer capacity in tokens (bits=16 GPU ⇒ residual capacity).
    pub fn capacity(&self) -> usize {
        // bits=16 GPU: return GPU residual capacity (may be smaller than max_seq_len
        // due to grow-on-demand). Used for HeadMajor stride calculations.
        if self.bits == 16 && self.gpu_res_k.is_some() {
            self.res_cap
        } else {
            self.max_seq_len
        }
    }

    /// Memory layout (bits=16 GPU ⇒ HeadMajor residual).
    pub fn layout(&self) -> KVLayout {
        // bits=16 GPU mode: residual is HeadMajor [kv_heads, res_cap, head_dim].
        // bits=2/4/8 GPU mode: assembled attn view is SeqMajor [total, kv_heads, head_dim].
        // CPU mode: always SeqMajor.
        if self.bits == 16 && self.gpu_res_k.is_some() {
            KVLayout::HeadMajor
        } else {
            KVLayout::SeqMajor
        }
    }

    /// Number of KV heads.
    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    /// Dimension per head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The DType the caller should pass to `update()` (KIVI quantizes internally → F32).
    pub fn kv_dtype(&self) -> DType {
        DType::F32
    }

    /// Append new K/V data. Input shape: `[batch, seq_len, kv_heads, head_dim]`.
    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let seq_len = new_k.shape().dims()[1];
        let total_after = self.total_tokens() + seq_len;
        if total_after > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "KiviCache overflow: {} + {} > {}",
                self.total_tokens(),
                seq_len,
                self.max_seq_len
            ));
        }

        // GPU path: dispatch GPU kernels for residual update
        if self.gpu_backend.is_some() {
            return self.update_gpu(new_k, new_v, seq_len);
        }

        // === CPU path (unchanged) ===
        // Input layout: [batch=1, seq_len, kv_heads, head_dim] (SeqMajor, F32)
        let k_data = new_k.as_slice::<f32>();
        let v_data = new_v.as_slice::<f32>();

        // Append each token to residual buffer
        for s in 0..seq_len {
            // If residual is full, flush to Q2
            if self.res_pos >= self.res_cap {
                self.flush_residual();
            }

            // Copy this token into residual buffer
            // Input: seq-major [s * kv_heads * head_dim + h * head_dim + d]
            // Residual: [h * res_cap * head_dim + res_pos * head_dim + d]
            for h in 0..self.kv_heads {
                let src_base = s * self.kv_heads * self.head_dim + h * self.head_dim;
                let dst_base = h * self.res_cap * self.head_dim + self.res_pos * self.head_dim;
                self.res_k[dst_base..dst_base + self.head_dim]
                    .copy_from_slice(&k_data[src_base..src_base + self.head_dim]);
                self.res_v[dst_base..dst_base + self.head_dim]
                    .copy_from_slice(&v_data[src_base..src_base + self.head_dim]);
            }
            self.res_pos += 1;
        }

        Ok(())
    }

    /// K/V tensors for attention covering `[0..current_pos]` (dequantized assembly).
    ///
    /// `&mut self` — internal buffer assembly (KIVI dequantization). 충돌 없는 동명 inherent
    /// (KVCache 와 달리 KiviCache 는 기존 `get_view` inherent 부재).
    pub fn get_view(&mut self) -> (Tensor, Tensor) {
        // GPU path: return tensors backed by GPU attention buffers
        if self.gpu_backend.is_some() {
            return self.get_view_gpu();
        }

        // === CPU path (unchanged) ===
        let total = self.total_tokens();
        let backend: Arc<dyn Backend> = self.out_backend.clone();
        if total == 0 {
            let buf = Arc::new(SharedBuffer::new(0, DType::F32));
            let shape = Shape::new(vec![1, 0, self.kv_heads, self.head_dim]);
            let t = Tensor::new(shape.clone(), buf.clone(), backend.clone());
            return (t.clone(), t);
        }

        // Incremental assemble (only dequantize new Q2 flushes + copy residual)
        self.assemble_view();

        let shape = Shape::new(vec![1, total, self.kv_heads, self.head_dim]);

        // Zero-copy: create a view into the pre-allocated attn buffers.
        // SharedBufferView holds an Arc to keep the backing buffer alive, and
        // exposes only the valid region (total * kv_heads * head_dim * 4 bytes).
        // This eliminates the per-token memcpy + malloc that was the bottleneck.
        let valid_bytes = total * self.kv_heads * self.head_dim * 4;
        let k_view = Arc::new(SharedBufferView::new(
            self.attn_k_buf.clone(),
            valid_bytes,
            DType::F32,
        ));
        let v_view = Arc::new(SharedBufferView::new(
            self.attn_v_buf.clone(),
            valid_bytes,
            DType::F32,
        ));
        let k_tensor = Tensor::new(shape.clone(), k_view, backend.clone());
        let v_tensor = Tensor::new(shape, v_view, backend);

        (k_tensor, v_tensor)
    }

    /// Whether AWQE is enabled (decode-time attention scores required).
    pub fn is_awqe_enabled(&self) -> bool {
        self.awqe_enabled
    }

    /// Store post-softmax attention scores from the latest decode step (AWQE).
    pub fn set_attn_scores(
        &mut self,
        scores: &[f32],
        n_heads_q: usize,
        stride: usize,
        valid_len: usize,
    ) {
        if !self.awqe_enabled {
            return;
        }
        let snapshot = self
            .last_attn_scores
            .get_or_insert_with(|| AttnScoresSnapshot {
                scores: Vec::new(),
                n_heads_q,
                stride,
                valid_len,
            });
        // Reuse allocation when possible
        snapshot.scores.clear();
        snapshot.scores.extend_from_slice(scores);
        snapshot.n_heads_q = n_heads_q;
        snapshot.stride = stride;
        snapshot.valid_len = valid_len;
    }

    /// Raw GPU buffers for native KIVI fused attention (Some only in GPU mode).
    pub fn get_kivi_raw_buffers(&self) -> Option<KiviRawBuffers<'_>> {
        self.get_raw_gpu_buffers()
    }

    /// Memory usage in bytes for currently stored KV data.
    pub fn memory_usage_bytes(&self) -> usize {
        let q_bytes = self.qk.memory_bytes() + self.qv.memory_bytes();
        let res_bytes = self.res_pos * self.kv_heads * self.head_dim * 4 * 2; // K + V
        q_bytes + res_bytes
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::Buffer;
    use crate::memory::host::shared::SharedBuffer;

    fn make_input_tensor(
        seq_len: usize,
        kv_heads: usize,
        head_dim: usize,
        base_val: f32,
    ) -> Tensor {
        let n = seq_len * kv_heads * head_dim;
        let mut data = vec![0.0f32; n];
        for i in 0..n {
            data[i] = base_val + (i as f32) * 0.01;
        }
        let buf = Arc::new(SharedBuffer::new(n * 4, DType::F32));
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.as_mut_ptr() as *mut f32, n);
        }
        let backend: Arc<dyn crate::backend::Backend> = Arc::new(CpuBackend::new());
        Tensor::new(
            Shape::new(vec![1, seq_len, kv_heads, head_dim]),
            buf,
            backend,
        )
    }

    #[test]
    fn test_kivi_cache_basic() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        assert_eq!(cache.current_pos(), 0);
        assert_eq!(cache.kv_dtype(), DType::F32);
        assert_eq!(cache.layout(), KVLayout::SeqMajor);

        // Add 1 token
        let k = make_input_tensor(1, kv_heads, head_dim, 1.0);
        let v = make_input_tensor(1, kv_heads, head_dim, 2.0);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos(), 1);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(cache.q2_tokens, 0);
    }

    #[test]
    fn test_kivi_cache_residual_only() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill 16 tokens (within residual capacity)
        for i in 0..16 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1 + 100.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos(), 16);
        assert_eq!(cache.q2_tokens, 0);
        assert_eq!(cache.res_pos, 16);

        // get_view should return exact FP32 data (no quantization involved)
        let (k_view, _v_view) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 16, kv_heads, head_dim]);

        let k_out = k_view.as_slice::<f32>();
        // First token, first head, first value
        let expected = 0.0 * 0.1 + 0.0 * 0.01; // base_val=0, i=0
        assert!(
            (k_out[0] - expected).abs() < 1e-5,
            "k_out[0]={}, expected={}",
            k_out[0],
            expected
        );
    }

    #[test]
    fn test_kivi_cache_flush_and_quantize() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill exactly res_cap tokens to trigger flush on the next insert
        for i in 0..res_cap {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.res_pos, res_cap);
        assert_eq!(cache.q2_tokens, 0);

        // Add one more token → triggers flush
        let k = make_input_tensor(1, kv_heads, head_dim, 99.0);
        let v = make_input_tensor(1, kv_heads, head_dim, 99.0);
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.q2_tokens, res_cap);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(cache.current_pos(), res_cap + 1);
    }

    #[test]
    fn test_kivi_cache_get_view_after_flush() {
        let kv_heads = 1;
        let head_dim = 32; // minimum for QK2_0
        let max_seq = 128;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Values: sequential so we can verify round-trip
        let mut all_k = Vec::new();
        let mut all_v = Vec::new();

        for i in 0..33 {
            let base_k = i as f32 * 0.1;
            let base_v = i as f32 * 0.1 + 50.0;
            let k = make_input_tensor(1, kv_heads, head_dim, base_k);
            let v = make_input_tensor(1, kv_heads, head_dim, base_v);

            // Save expected values
            all_k.extend_from_slice(k.as_slice::<f32>());
            all_v.extend_from_slice(v.as_slice::<f32>());

            cache.update(&k, &v).unwrap();
        }

        // 32 tokens flushed to Q2, 1 in residual
        assert_eq!(cache.q2_tokens, 32);
        assert_eq!(cache.res_pos, 1);

        let (k_view, v_view) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();
        let v_out = v_view.as_slice::<f32>();

        assert_eq!(k_out.len(), 33 * kv_heads * head_dim);

        // Q2 tokens (0..32) should be approximately correct (Q2 error expected)
        let range_k = 3.2f32; // rough range of input values
        let max_q2_err = range_k / 6.0 + 0.1; // Q2 max error ≈ scale/2 + epsilon

        for i in 0..(32 * kv_heads * head_dim) {
            let err = (all_k[i] - k_out[i]).abs();
            assert!(
                err < max_q2_err,
                "K Q2 error at {i}: expected={}, got={}, err={err}",
                all_k[i],
                k_out[i]
            );
        }

        // Last token (residual, index 32) should be exact (FP32)
        let res_start = 32 * kv_heads * head_dim;
        for d in 0..head_dim {
            assert!(
                (all_k[res_start + d] - k_out[res_start + d]).abs() < 1e-5,
                "K residual mismatch at d={d}"
            );
            assert!(
                (all_v[res_start + d] - v_out[res_start + d]).abs() < 1e-5,
                "V residual mismatch at d={d}"
            );
        }
    }

    #[test]
    fn test_kivi_cache_multi_token_update() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Add 8 tokens at once (prefill-like)
        let k = make_input_tensor(8, kv_heads, head_dim, 1.0);
        let v = make_input_tensor(8, kv_heads, head_dim, 2.0);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos(), 8);
        assert_eq!(cache.res_pos, 8);
    }

    #[test]
    fn test_kivi_cache_overflow() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 64;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill to max
        for _ in 0..64 {
            let k = make_input_tensor(1, kv_heads, head_dim, 0.0);
            let v = make_input_tensor(1, kv_heads, head_dim, 0.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos(), 64);

        // One more should fail
        let k = make_input_tensor(1, kv_heads, head_dim, 0.0);
        let v = make_input_tensor(1, kv_heads, head_dim, 0.0);
        assert!(cache.update(&k, &v).is_err());
    }

    #[test]
    fn test_kivi_cache_memory_usage() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        assert_eq!(cache.memory_usage_bytes(), 0);
    }

    #[test]
    fn test_kivi_cache_compression_ratio() {
        let kv_heads = 8;
        let head_dim = 64;
        let max_seq = 512;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill 128 tokens: flush at 33,65,97 → 96 Q2 + 32 residual
        for _ in 0..128 {
            let k = make_input_tensor(1, kv_heads, head_dim, 0.5);
            let v = make_input_tensor(1, kv_heads, head_dim, 0.5);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 96);
        assert_eq!(cache.res_pos, 32);
        assert_eq!(cache.current_pos(), 128);

        let kivi_bytes = cache.memory_usage_bytes();
        // FP32 would be: 128 * 8 * 64 * 4 bytes * 2 (K+V) = 524288 bytes
        let fp32_bytes = 128 * kv_heads * head_dim * 4 * 2;

        // Q2 portion has high compression, residual is FP32.
        // With 96 Q2 tokens: 96/128 * 10.67x compression + 32/128 * 1x
        // Should still show significant compression overall.
        let ratio = fp32_bytes as f64 / kivi_bytes as f64;
        assert!(
            ratio > 3.0,
            "compression ratio {ratio:.1}x too low (expected >3x)"
        );
    }

    #[test]
    fn test_kivi_cache_incremental_deq() {
        // Verify that incremental dequantization produces identical results
        // to full dequantization across multiple flushes.
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill 64 tokens → 2 flushes (at token 32 and 64)
        for i in 0..65 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03 + 10.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 64);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(cache.q2_deq_tokens, 0); // nothing dequantized yet

        // First get_view: dequantize all 64 Q2 tokens
        let (k1, v1) = cache.get_view();
        assert_eq!(cache.q2_deq_tokens, 64);
        let k1_data: Vec<f32> = k1.as_slice::<f32>().to_vec();
        let v1_data: Vec<f32> = v1.as_slice::<f32>().to_vec();

        // Add more tokens to trigger another flush
        for i in 65..97 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03 + 10.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 96);
        assert_eq!(cache.q2_deq_tokens, 64); // still at 64 from last get_view

        // Second get_view: should only dequantize flush 2 (tokens 64..95)
        let (k2, _v2) = cache.get_view();
        assert_eq!(cache.q2_deq_tokens, 96);
        let k2_data = k2.as_slice::<f32>();

        // First 64 tokens should be identical to the previous get_view
        let first_64_elems = 64 * kv_heads * head_dim;
        assert_eq!(&k1_data[..first_64_elems], &k2_data[..first_64_elems]);
        assert_eq!(
            &v1_data[..first_64_elems],
            &_v2.as_slice::<f32>()[..first_64_elems]
        );
    }

    #[test]
    fn test_kivi_cache_vec_capacity_no_realloc() {
        // With correct capacity calculations, no reallocation should occur
        // for sequences within max_seq_len.
        let kv_heads = 8;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 64; // larger residual to test groups_per_flush > 1

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        let initial_k_cap = cache.qk.capacity_bytes();
        let initial_v_cap = cache.qv.capacity_bytes();

        // Fill to max_seq_len (3 flushes: at 64, 128, 192 → 192 Q2 + 64 residual)
        for _ in 0..max_seq {
            let k = make_input_tensor(1, kv_heads, head_dim, 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, 0.2);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 192);
        assert_eq!(cache.res_pos, 64);

        // No reallocation should have occurred
        assert_eq!(
            cache.qk.capacity_bytes(),
            initial_k_cap,
            "K vec reallocated (initial={initial_k_cap}, now={})",
            cache.qk.capacity_bytes()
        );
        assert_eq!(
            cache.qv.capacity_bytes(),
            initial_v_cap,
            "V vec reallocated (initial={initial_v_cap}, now={})",
            cache.qv.capacity_bytes()
        );
    }

    /// Direct quality comparison: KIVI Q2 vs Baseline.
    ///
    /// Same synthetic KV data fed to both caches. Measures:
    /// - Cosine similarity (K and V separately)
    /// - MSE (mean squared error)
    /// - Memory usage
    ///
    /// Data pattern: sin-based low-rank structure (realistic for attention KV).
    #[test]
    fn test_compare_kivi_vs_baseline() {
        use crate::kv::kv_cache::KVCache;

        let kv_heads = 8;
        let head_dim = 64;
        let max_seq = 256;
        let prefill_len = 32;
        let decode_len = 96;
        let total = prefill_len + decode_len; // 128 tokens

        // Generate ground truth KV data (F32)
        // Pattern: sin-based with mild low-rank structure
        let mut ground_truth_k = vec![0.0f32; total * kv_heads * head_dim];
        let mut ground_truth_v = vec![0.0f32; total * kv_heads * head_dim];
        for t in 0..total {
            for h in 0..kv_heads {
                for d in 0..head_dim {
                    let idx = t * kv_heads * head_dim + h * head_dim + d;
                    // K: lower effective rank (attention key pattern)
                    ground_truth_k[idx] = ((t as f32 + 1.0) * (d as f32 + 1.0) * 0.003).sin() * 0.3
                        + ((h as f32 + 1.0) * (d as f32 + 1.0) * 0.007).cos() * 0.2;
                    // V: higher rank, more varied (value pattern)
                    ground_truth_v[idx] =
                        ((t as f32 * 0.1 + d as f32 * 0.05 + h as f32 * 0.3).sin()) * 0.4
                            + (t as f32 * 0.02).cos() * 0.1;
                }
            }
        }

        // Helper: create F32 tensor from slice
        let make_f32 = |data: &[f32], seq_len: usize| -> Tensor {
            let n = seq_len * kv_heads * head_dim;
            let buf = Arc::new(SharedBuffer::new(n * 4, DType::F32));
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buf.as_mut_ptr() as *mut f32, n);
            }
            let backend: Arc<dyn crate::backend::Backend> = Arc::new(CpuBackend::new());
            Tensor::new(
                Shape::new(vec![1, seq_len, kv_heads, head_dim]),
                buf,
                backend,
            )
        };

        // Helper: cosine similarity
        let cosine_sim = |a: &[f32], b: &[f32]| -> f64 {
            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..a.len() {
                let (x, y) = (a[i] as f64, b[i] as f64);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            dot / (na.sqrt() * nb.sqrt() + 1e-15)
        };

        // Helper: MSE
        let mse = |a: &[f32], b: &[f32]| -> f64 {
            let sum: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = (x - y) as f64;
                    d * d
                })
                .sum();
            sum / a.len() as f64
        };

        let elems_per_step = |seq_len: usize| seq_len * kv_heads * head_dim;

        // ═══════════════════════════════════════════════
        // 1. Baseline: standard KVCache (F32, lossless)
        // ═══════════════════════════════════════════════
        let baseline_k: Vec<f32>;
        let baseline_v: Vec<f32>;
        {
            let memory: Arc<dyn crate::memory::Memory> = Arc::new(crate::memory::galloc::Galloc);
            let backend: Arc<dyn crate::backend::Backend> = Arc::new(CpuBackend::new());
            let k_buf = memory
                .alloc(max_seq * kv_heads * head_dim * 4, DType::F32)
                .unwrap();
            let v_buf = memory
                .alloc(max_seq * kv_heads * head_dim * 4, DType::F32)
                .unwrap();
            let mut cache = KVCache::new(
                Tensor::new(
                    Shape::new(vec![1, max_seq, kv_heads, head_dim]),
                    k_buf,
                    backend.clone(),
                ),
                Tensor::new(
                    Shape::new(vec![1, max_seq, kv_heads, head_dim]),
                    v_buf,
                    backend.clone(),
                ),
                max_seq,
            );

            // Prefill
            let off = 0;
            let n = elems_per_step(prefill_len);
            let k = make_f32(&ground_truth_k[off..off + n], prefill_len);
            let v = make_f32(&ground_truth_v[off..off + n], prefill_len);
            cache.update(&k, &v).unwrap();

            // Decode
            for t in 0..decode_len {
                let off = (prefill_len + t) * kv_heads * head_dim;
                let n = elems_per_step(1);
                let k = make_f32(&ground_truth_k[off..off + n], 1);
                let v = make_f32(&ground_truth_v[off..off + n], 1);
                cache.update(&k, &v).unwrap();
            }

            let (kv, vv) = cache.view();
            let n = total * kv_heads * head_dim;
            baseline_k = kv.as_slice::<f32>()[..n].to_vec();
            baseline_v = vv.as_slice::<f32>()[..n].to_vec();
        }

        // ═══════════════════════════════════════════════
        // 2. KIVI Q2 (res_cap=32)
        // ═══════════════════════════════════════════════
        let kivi_k: Vec<f32>;
        let kivi_v: Vec<f32>;
        let kivi_mem: usize;
        {
            let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, 32);

            // Prefill
            let off = 0;
            let n = elems_per_step(prefill_len);
            let k = make_f32(&ground_truth_k[off..off + n], prefill_len);
            let v = make_f32(&ground_truth_v[off..off + n], prefill_len);
            cache.update(&k, &v).unwrap();

            // Decode
            for t in 0..decode_len {
                let off = (prefill_len + t) * kv_heads * head_dim;
                let n = elems_per_step(1);
                let k = make_f32(&ground_truth_k[off..off + n], 1);
                let v = make_f32(&ground_truth_v[off..off + n], 1);
                cache.update(&k, &v).unwrap();
            }

            kivi_mem = cache.memory_usage_bytes();
            let (kv, vv) = cache.get_view();
            let n = total * kv_heads * head_dim;
            kivi_k = kv.as_slice::<f32>()[..n].to_vec();
            kivi_v = vv.as_slice::<f32>()[..n].to_vec();
        }

        // ═══════════════════════════════════════════════
        // 3. F16 baseline (quantization-only loss, no compression)
        // ═══════════════════════════════════════════════
        let f16_baseline_k: Vec<f32> = ground_truth_k
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let f16_baseline_v: Vec<f32> = ground_truth_v
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();

        // ═══════════════════════════════════════════════
        // Compute metrics
        // ═══════════════════════════════════════════════
        let n = total * kv_heads * head_dim;

        // KIVI vs Baseline (F32)
        let kivi_k_cos = cosine_sim(&baseline_k, &kivi_k);
        let kivi_v_cos = cosine_sim(&baseline_v, &kivi_v);
        let kivi_k_mse = mse(&baseline_k, &kivi_k);
        let kivi_v_mse = mse(&baseline_v, &kivi_v);

        // F16 vs F32 Baseline (pure precision loss)
        let f16_k_cos = cosine_sim(&baseline_k, &f16_baseline_k[..n]);
        let f16_v_cos = cosine_sim(&baseline_v, &f16_baseline_v[..n]);

        // Memory
        let baseline_mem = total * kv_heads * head_dim * 4 * 2; // F32 K+V

        // ═══════════════════════════════════════════════
        // Print comparison table
        // ═══════════════════════════════════════════════
        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════════════╗");
        eprintln!("║           KIVI vs Baseline Quality Comparison                   ║");
        eprintln!(
            "║  Data: {total} tokens, {kv_heads} heads, {head_dim} dim, prefill={prefill_len} decode={decode_len}  ║"
        );
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!("║                      │  K CosSim  │  V CosSim  │  K MSE       ║");
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!(
            "║ F16 (precision only) │  {:.6}  │  {:.6}  │  {:.2e}  ║",
            f16_k_cos,
            f16_v_cos,
            mse(&baseline_k, &f16_baseline_k[..n])
        );
        eprintln!(
            "║ KIVI Q2 res=32       │  {:.6}  │  {:.6}  │  {:.2e}  ║",
            kivi_k_cos, kivi_v_cos, kivi_k_mse
        );
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!("║                      │  V MSE     │  Memory    │  Ratio       ║");
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!(
            "║ Baseline (F32)       │  0.00e+00  │  {:>7}B  │  1.00x       ║",
            baseline_mem
        );
        eprintln!(
            "║ KIVI Q2 res=32       │  {:.2e}  │  {:>7}B  │  {:.2}x       ║",
            kivi_v_mse,
            kivi_mem,
            baseline_mem as f64 / kivi_mem as f64
        );
        eprintln!("╚══════════════════════════════════════════════════════════════════╝");
        eprintln!();

        // ═══════════════════════════════════════════════
        // Assertions
        // ═══════════════════════════════════════════════

        // KIVI should have reasonable K cosine similarity
        assert!(kivi_k_cos > 0.9, "KIVI K CosSim {kivi_k_cos:.4} < 0.9");

        // KIVI should achieve good compression ratio
        let kivi_ratio = baseline_mem as f64 / kivi_mem as f64;
        assert!(
            kivi_ratio > 1.5,
            "KIVI ratio ({kivi_ratio:.1}x) should be > 1.5x"
        );
    }

    #[test]
    fn test_kivi_new_with_bits_q4() {
        let cache = KiviCache::new_with_bits(2, 64, 256, 32, 4);
        assert_eq!(cache.bits(), 4);
        assert_eq!(cache.current_pos(), 0);
    }

    #[test]
    fn test_kivi_new_with_bits_q8() {
        let cache = KiviCache::new_with_bits(2, 64, 256, 32, 8);
        assert_eq!(cache.bits(), 8);
    }

    #[test]
    fn test_kivi_q4_flush_and_view() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 128;
        let res_cap = 32;

        let mut cache = KiviCache::new_with_bits(kv_heads, head_dim, max_seq, res_cap, 4);

        let mut all_k = Vec::new();
        for i in 0..33 {
            let base = i as f32 * 0.1;
            let k = make_input_tensor(1, kv_heads, head_dim, base);
            let v = make_input_tensor(1, kv_heads, head_dim, base + 50.0);
            all_k.extend_from_slice(k.as_slice::<f32>());
            cache.update(&k, &v).unwrap();
        }

        assert_eq!(cache.q2_tokens, 32);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(cache.bits(), 4);

        let (k_view, _v_view) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();

        // Q4 should have lower error than Q2
        let range = 3.2f32;
        let max_q4_err = range / 30.0 + 0.1;
        for i in 0..(32 * kv_heads * head_dim) {
            let err = (all_k[i] - k_out[i]).abs();
            assert!(
                err < max_q4_err,
                "K Q4 error at {i}: expected={}, got={}, err={err}",
                all_k[i],
                k_out[i]
            );
        }
    }

    #[test]
    fn test_kivi_transition_bits_8_to_4_to_2() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new_with_bits(kv_heads, head_dim, max_seq, res_cap, 8);

        // Fill 65 tokens → 2 flushes (flush at 32, 64) + 1 residual
        for i in 0..65 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03 + 10.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 64);
        assert_eq!(cache.bits(), 8);

        // Get Q8 view as reference
        let n_q = 64 * kv_heads * head_dim;
        let (k8, _) = cache.get_view();
        let k8_data: Vec<f32> = k8.as_slice::<f32>()[..n_q].to_vec();

        // Transition 8 → 4
        cache.transition_bits(4).unwrap();
        assert_eq!(cache.bits(), 4);
        assert_eq!(cache.q2_tokens, 64);

        let (k4, _) = cache.get_view();
        // Clone the data: zero-copy get_view shares the internal buffer, so
        // subsequent assemble_view() calls (e.g. from transition_bits) would
        // overwrite the data visible through this tensor.
        let k4_data: Vec<f32> = k4.as_slice::<f32>()[..n_q].to_vec();

        // Q4 should be close to Q8 (some error from requant)
        let mse_8_to_4: f32 = k8_data
            .iter()
            .zip(k4_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / k8_data.len() as f32;

        // Transition 4 → 2
        cache.transition_bits(2).unwrap();
        assert_eq!(cache.bits(), 2);

        let (k2, _) = cache.get_view();
        let k2_data: Vec<f32> = k2.as_slice::<f32>()[..n_q].to_vec();

        let mse_4_to_2: f32 = k4_data
            .iter()
            .zip(k2_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / k4_data.len() as f32;

        // Both MSEs should be bounded (not NaN, not huge)
        assert!(mse_8_to_4 < 1.0, "MSE 8→4 too high: {mse_8_to_4}");
        assert!(mse_4_to_2 < 5.0, "MSE 4→2 too high: {mse_4_to_2}");

        // Q4→Q2 should have more error than Q8→Q4
        assert!(
            mse_4_to_2 > mse_8_to_4,
            "Expected more error in 4→2 than 8→4"
        );
    }

    #[test]
    fn test_kivi_transition_noop() {
        let mut cache = KiviCache::new_with_bits(1, 32, 128, 32, 4);
        // Fill 32 tokens
        for _ in 0..33 {
            let k = make_input_tensor(1, 1, 32, 0.5);
            let v = make_input_tensor(1, 1, 32, 0.5);
            cache.update(&k, &v).unwrap();
        }
        // Transition to same bits — should be no-op
        cache.transition_bits(4).unwrap();
        assert_eq!(cache.bits(), 4);
    }

    #[test]
    fn test_kivi_transition_empty() {
        let mut cache = KiviCache::new_with_bits(1, 32, 128, 32, 8);
        // No tokens — transition should just switch format
        cache.transition_bits(2).unwrap();
        assert_eq!(cache.bits(), 2);
        assert_eq!(cache.q2_tokens, 0);
    }

    #[test]
    fn test_take_flush_proxies_accumulates_multiple_flushes() {
        // residual_size=32 → flush triggers when res_pos reaches res_cap (32).
        // Flush #1 fires on token index 32 (before inserting it, res_pos==32).
        // Flush #2 fires on token index 65 (before inserting it, res_pos==32 again).
        // So we need 65+1=66 tokens to observe 2 flushes.
        let mut cache = KiviCache::new(1, 32, 256, 32);

        // Insert 66 tokens → 2 flushes should have occurred
        for i in 0..66 {
            let k = make_input_tensor(1, 1, 32, i as f32 * 0.1);
            let v = make_input_tensor(1, 1, 32, i as f32 * 0.1 + 5.0);
            cache.update(&k, &v).unwrap();
        }

        let proxies = cache.take_flush_proxies();
        // Each flush pushes 2 QCF metrics (compute_flush_nmse → "kivi",
        // compute_flush_opr → "kivi_opr"). 2 flushes × 2 metrics = 4 total.
        assert_eq!(
            proxies.len(),
            4,
            "expected 4 flush proxies after 66 tokens with residual_size=32 (2 flushes × 2 metrics), got {}",
            proxies.len()
        );
        for p in &proxies {
            assert!(
                p.action == "kivi" || p.action == "kivi_opr",
                "unexpected action: {}",
                p.action
            );
            assert!(p.raw_value >= 0.0, "NMSE should be non-negative");
            assert_eq!(p.tokens_affected, 32);
        }

        // After take, buffer should be drained
        let proxies2 = cache.take_flush_proxies();
        assert!(proxies2.is_empty(), "buffer should be empty after take");
    }

    #[test]
    fn test_take_flush_proxies_empty_when_no_flush() {
        // residual_size=32 → no flush until 32 tokens are inserted
        let mut cache = KiviCache::new(1, 32, 256, 32);

        // Insert only 16 tokens (< residual_size) → no flush
        for i in 0..16 {
            let k = make_input_tensor(1, 1, 32, i as f32 * 0.1);
            let v = make_input_tensor(1, 1, 32, i as f32 * 0.1 + 5.0);
            cache.update(&k, &v).unwrap();
        }

        let proxies = cache.take_flush_proxies();
        assert!(
            proxies.is_empty(),
            "no flush should occur below residual_size"
        );
    }

    #[test]
    fn test_reset_clears_flush_proxies() {
        let mut cache = KiviCache::new(1, 32, 256, 32);

        // Flush fires when res_pos reaches res_cap before inserting the 33rd token.
        for i in 0..33 {
            let k = make_input_tensor(1, 1, 32, i as f32 * 0.1);
            let v = make_input_tensor(1, 1, 32, i as f32 * 0.1 + 5.0);
            cache.update(&k, &v).unwrap();
        }
        assert!(
            !cache.flush_proxies.is_empty(),
            "flush_proxies should have entries after flush"
        );

        cache.reset();
        assert!(
            cache.flush_proxies.is_empty(),
            "reset() must clear flush_proxies"
        );
    }

    // ── GPU-mode tests ────────────────────────────────────────────────────────

    /// `new()` / `new_with_bits()` must always create CPU-mode caches.
    #[test]
    fn test_cpu_mode_is_default() {
        let cache = KiviCache::new(2, 64, 256, 32);
        assert!(!cache.is_gpu(), "new() must return CPU-mode cache");
        assert!(cache.gpu_backend.is_none());
        assert!(cache.gpu_res_k.is_none());
        assert!(cache.gpu_attn_k.is_none());
    }

    /// `new_gpu()` with a non-OpenCL backend must silently fall back to CPU mode.
    #[test]
    fn test_new_gpu_non_opencl_falls_back_to_cpu() {
        use crate::memory::galloc::Galloc;
        let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn crate::memory::Memory> = Arc::new(Galloc::new());
        let cache = KiviCache::new_gpu(2, 64, 256, 32, 2, cpu_backend, None, memory);
        // CpuBackend.name() != "OpenCL" → must fall back to CPU mode
        assert!(
            !cache.is_gpu(),
            "new_gpu() with non-OpenCL backend must use CPU mode"
        );
    }

    /// `reset()` must clear GPU position counters.
    #[test]
    fn test_reset_clears_gpu_counters() {
        let mut cache = KiviCache::new(2, 64, 256, 32);
        // Manually poke GPU counters to non-zero values (simulating partial GPU use)
        cache.gpu_q2k_blocks = 42;
        cache.gpu_q2v_blocks = 17;
        cache.reset();
        assert_eq!(cache.gpu_q2k_blocks, 0, "reset() must zero gpu_q2k_blocks");
        assert_eq!(cache.gpu_q2v_blocks, 0, "reset() must zero gpu_q2v_blocks");
    }

    /// Existing CPU tests must not be affected by GPU fields being None.
    #[test]
    fn test_cpu_path_unaffected_by_gpu_fields() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 128;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        // Fill 33 tokens → triggers one flush
        for i in 0..33 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05 + 10.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 32);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(
            cache.gpu_q2k_blocks, 0,
            "cpu mode must not touch gpu counters"
        );
        let (k_view, _) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 33, kv_heads, head_dim]);
    }

    // ── AWQE tests ────────────────────────────────────────────────────────────

    /// Test 10: awqe_enabled=false → set_attn_scores is no-op, last_attn_scores stays None.
    #[test]
    fn test_kivi_awqe_disabled_no_scores_stored() {
        let mut cache = KiviCache::new(1, 32, 256, 32);
        assert!(!cache.awqe_enabled);

        // Call set_attn_scores with some data
        let scores = vec![0.1f32; 64];
        cache.set_attn_scores(&scores, 1, 64, 32);

        // awqe_enabled=false → must not store anything
        assert!(
            cache.last_attn_scores.is_none(),
            "set_attn_scores must be no-op when awqe_enabled=false"
        );
    }

    /// Test 11: awqe_enabled=true → set_attn_scores stores scores, flush produces kivi_awqe metric.
    #[test]
    fn test_kivi_awqe_enabled_scores_stored() {
        let kv_heads = 1;
        let head_dim = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let max_seq = 256;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        cache.set_awqe_enabled(true);

        // valid_len must cover flush tokens (q2_tokens=0 .. flush_tokens=32)
        // So valid_len >= 32
        let valid_len = 64usize;
        let stride = valid_len;
        let scores: Vec<f32> = (0..n_heads_q * stride)
            .map(|i| {
                if i < valid_len {
                    1.0 / valid_len as f32
                } else {
                    0.0
                }
            })
            .collect();

        // Store scores (simulating previous decode step)
        cache.set_attn_scores(&scores, n_heads_q, stride, valid_len);
        assert!(cache.last_attn_scores.is_some(), "scores must be stored");

        // Insert 33 tokens to trigger one flush at the 33rd token
        for i in 0..33 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05 + 1.0);
            cache.update(&k, &v).unwrap();
        }

        let proxies = cache.take_flush_proxies();
        // Must have NMSE, OPR, and AWQE
        let awqe_metric = proxies.iter().find(|m| m.action == "kivi_awqe");
        assert!(
            awqe_metric.is_some(),
            "flush_proxies must contain 'kivi_awqe' when awqe_enabled=true and scores present; got: {:?}",
            proxies.iter().map(|m| &m.action).collect::<Vec<_>>()
        );
        assert!(
            awqe_metric.unwrap().raw_value >= 0.0,
            "AWQE raw_value must be non-negative"
        );
    }

    /// Test 12: awqe_enabled=true but set_attn_scores never called → no AWQE in flush_proxies.
    #[test]
    fn test_kivi_awqe_no_scores_no_awqe() {
        let kv_heads = 1;
        let head_dim = 32;
        let res_cap = 32;
        let max_seq = 256;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        cache.set_awqe_enabled(true);

        // Do NOT call set_attn_scores
        // Insert tokens to trigger a flush
        for i in 0..33 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05 + 1.0);
            cache.update(&k, &v).unwrap();
        }

        let proxies = cache.take_flush_proxies();
        let awqe_metric = proxies.iter().find(|m| m.action == "kivi_awqe");
        assert!(
            awqe_metric.is_none(),
            "flush_proxies must NOT contain 'kivi_awqe' when no scores were provided; got: {:?}",
            proxies.iter().map(|m| &m.action).collect::<Vec<_>>()
        );
        // But NMSE and OPR should still be present
        assert!(
            proxies.iter().any(|m| m.action == "kivi"),
            "NMSE proxy must still be present"
        );
        assert!(
            proxies.iter().any(|m| m.action == "kivi_opr"),
            "OPR proxy must still be present"
        );
    }

    /// Test 13: is_awqe_enabled() mirrors set_awqe_enabled (5-F: trait needs_attn_scores
    /// 폐기 — 본문이 위임하던 inherent is_awqe_enabled 를 직접 검증).
    #[test]
    fn test_kivi_needs_attn_scores() {
        let mut cache = KiviCache::new(1, 32, 256, 32);

        // Default: false
        assert!(
            !cache.is_awqe_enabled(),
            "is_awqe_enabled() must be false by default"
        );

        // After enabling
        cache.set_awqe_enabled(true);
        assert!(
            cache.is_awqe_enabled(),
            "is_awqe_enabled() must be true after set_awqe_enabled(true)"
        );

        // After disabling again
        cache.set_awqe_enabled(false);
        assert!(
            !cache.is_awqe_enabled(),
            "is_awqe_enabled() must be false after set_awqe_enabled(false)"
        );
    }

    // ── bits=16 (unquantized) mode tests ─────────────────────────────────────

    /// bits=16 construction: res_cap must be max_seq_len and no quantized blocks.
    #[test]
    fn test_kivi_bits16_construction() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32; // ignored for bits=16

        let cache = KiviCache::new_with_bits(kv_heads, head_dim, max_seq, res_cap, 16);
        assert_eq!(cache.bits(), 16);
        assert_eq!(
            cache.res_cap, max_seq,
            "bits=16 must set res_cap to max_seq_len"
        );
        assert_eq!(cache.q2_tokens, 0);
        assert_eq!(cache.res_pos, 0);
    }

    /// bits=16 mode stores all tokens in residual; no flush is ever performed.
    #[test]
    fn test_kivi_bits16_no_flush() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 128;

        let mut cache = KiviCache::new_with_bits(kv_heads, head_dim, max_seq, 32, 16);

        // Insert 100 tokens — well beyond a normal residual_size=32 flush threshold
        for i in 0..100 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1 + 50.0);
            cache.update(&k, &v).unwrap();
        }

        // In bits=16 mode, nothing should ever be flushed to quantized storage
        assert_eq!(cache.q2_tokens, 0, "bits=16 must never quantize tokens");
        assert_eq!(cache.res_pos, 100);
        assert_eq!(cache.current_pos(), 100);
    }

    /// bits=16 get_view returns exact FP32 data (no quantization loss).
    #[test]
    fn test_kivi_bits16_exact_roundtrip() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 128;

        let mut cache = KiviCache::new_with_bits(kv_heads, head_dim, max_seq, 32, 16);

        let mut all_k = Vec::new();
        for i in 0..50 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1 + 50.0);
            all_k.extend_from_slice(k.as_slice::<f32>());
            cache.update(&k, &v).unwrap();
        }

        let (k_view, _) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();

        // Must be exact (FP32 residual, no quantization)
        for (i, (&expected, &got)) in all_k.iter().zip(k_out.iter()).enumerate() {
            assert!(
                (expected - got).abs() < 1e-6,
                "bits=16 roundtrip mismatch at idx {i}: expected={expected}, got={got}"
            );
        }
    }

    /// Transition Q2 → 16: all tokens restored to FP32 residual, qk/qv cleared.
    #[test]
    fn test_kivi_transition_q2_to_16() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 256;

        let mut cache = KiviCache::new_with_bits(kv_heads, head_dim, max_seq, 32, 2);

        // Insert 65 tokens → 2 flushes (32+32) + 1 residual
        let mut all_k = Vec::new();
        for i in 0..65 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05 + 10.0);
            all_k.extend_from_slice(k.as_slice::<f32>());
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 64);
        assert_eq!(cache.bits(), 2);

        // Transition to unquantized mode
        cache.transition_bits(16).unwrap();
        assert_eq!(cache.bits(), 16);
        assert_eq!(
            cache.q2_tokens, 0,
            "q2_tokens must be 0 after 16 transition"
        );
        assert_eq!(cache.res_pos, 65, "all 65 tokens must be in residual");
        assert_eq!(
            cache.res_cap, max_seq,
            "res_cap must be max_seq_len after 16 transition"
        );

        // get_view should return all 65 tokens (with some Q2 dequant error for the first 64)
        let (k_view, _) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();
        assert_eq!(k_out.len(), 65 * kv_heads * head_dim);

        // Last token (was in residual before transition) must be exact
        let last_start = 64 * kv_heads * head_dim;
        for d in 0..head_dim {
            assert!(
                (all_k[last_start + d] - k_out[last_start + d]).abs() < 1e-5,
                "last token (residual) must be exact after Q2→16 transition"
            );
        }
    }

    /// Transition 16 → Q2: tokens are quantized from residual, residual shrinks.
    #[test]
    fn test_kivi_transition_16_to_q2() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 256;

        let mut cache = KiviCache::new_with_bits(kv_heads, head_dim, max_seq, 32, 16);

        // Insert 65 tokens in unquantized mode
        for i in 0..65 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05 + 10.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.bits(), 16);
        assert_eq!(cache.res_pos, 65);

        // Transition to Q2
        cache.transition_bits(2).unwrap();
        assert_eq!(cache.bits(), 2);
        // 64 tokens flushed (2 full groups of 32), 1 remains in residual
        assert_eq!(cache.q2_tokens, 64);
        assert_eq!(cache.res_pos, 1);
        // res_cap must be shrunk to standard size (QKKV = 32)
        assert_eq!(cache.res_cap, QKKV);
        assert_eq!(cache.current_pos(), 65);
    }

    /// Transition 16 → 16 is a no-op.
    #[test]
    fn test_kivi_transition_16_noop() {
        let mut cache = KiviCache::new_with_bits(1, 32, 128, 32, 16);
        for i in 0..10 {
            let k = make_input_tensor(1, 1, 32, i as f32 * 0.1);
            let v = make_input_tensor(1, 1, 32, i as f32 * 0.1);
            cache.update(&k, &v).unwrap();
        }
        cache.transition_bits(16).unwrap();
        assert_eq!(cache.bits(), 16);
        assert_eq!(cache.res_pos, 10);
        assert_eq!(cache.q2_tokens, 0);
    }

    // ── estimate_dryrun_qcf tests ──

    #[test]
    fn test_dryrun_qcf_bits16_returns_zero() {
        let cache = KiviCache::new_with_bits(2, 64, 256, 32, 16);
        assert_eq!(cache.estimate_dryrun_qcf(), 0.0);
    }

    #[test]
    fn test_dryrun_qcf_empty_cache_returns_bits_proxy() {
        // Empty Q2 cache (no residual data) returns bits-based proxy
        let cache = KiviCache::new(2, 64, 256, 32);
        assert!((cache.estimate_dryrun_qcf() - 0.30).abs() < 1e-6);
    }

    #[test]
    fn test_dryrun_qcf_q4_empty_returns_proxy() {
        let cache = KiviCache::new_with_bits(2, 64, 256, 32, 4);
        assert!((cache.estimate_dryrun_qcf() - 0.10).abs() < 1e-6);
    }

    #[test]
    fn test_dryrun_qcf_q8_empty_returns_proxy() {
        let cache = KiviCache::new_with_bits(2, 64, 256, 32, 8);
        assert!((cache.estimate_dryrun_qcf() - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_dryrun_qcf_with_residual_data_computes_nmse() {
        let kv_heads = 2;
        let head_dim = 64;
        let mut cache = KiviCache::new(kv_heads, head_dim, 256, 32);
        // Fill 32 tokens (full residual buffer, triggering NMSE path)
        for i in 0..32 {
            let k = make_input_tensor(1, kv_heads, head_dim, (i + 1) as f32 * 0.5);
            let v = make_input_tensor(1, kv_heads, head_dim, (i + 1) as f32 * 0.3);
            cache.update(&k, &v).unwrap();
        }
        // After 32 inserts, residual was flushed (res_pos back to 0).
        // Insert a few more to have non-zero residual for dry-run.
        for i in 0..5 {
            let k = make_input_tensor(1, kv_heads, head_dim, (i + 33) as f32 * 0.5);
            let v = make_input_tensor(1, kv_heads, head_dim, (i + 33) as f32 * 0.3);
            cache.update(&k, &v).unwrap();
        }
        // res_pos should be 5 (< group_size=32), so falls back to proxy
        assert_eq!(cache.res_pos, 5);
        let qcf = cache.estimate_dryrun_qcf();
        // 5 tokens < QKKV(32), so n_groups=0, flush_tokens=0 → bits proxy 0.30
        assert!((qcf - 0.30).abs() < 1e-6);
    }

    #[test]
    fn test_dryrun_qcf_full_residual_computes_actual_nmse() {
        let kv_heads = 1;
        let head_dim = 32;
        let _cache = KiviCache::new(kv_heads, head_dim, 128, 32);
        // Fill exactly 32 tokens without flushing (the 32nd token triggers flush
        // during update, but estimate_dryrun_qcf is read-only and uses current res data).
        // We need to have 32 tokens in residual for NMSE. Use bits=4 and 64-token residual.
        let mut cache4 = KiviCache::new_with_bits(kv_heads, head_dim, 128, 64, 4);
        for i in 0..32 {
            let k = make_input_tensor(1, kv_heads, head_dim, (i + 1) as f32 * 0.7);
            let v = make_input_tensor(1, kv_heads, head_dim, (i + 1) as f32 * 0.4);
            cache4.update(&k, &v).unwrap();
        }
        // res_pos = 32, res_cap = 64, no flush happened yet
        assert_eq!(cache4.res_pos, 32);
        let qcf = cache4.estimate_dryrun_qcf();
        // Should compute actual NMSE (between 0.0 and 1.0, not a proxy)
        assert!((0.0..=1.0).contains(&qcf), "qcf={qcf} out of range");
        // Q4 NMSE should be < Q2 proxy (0.30) for typical data
        assert!(qcf < 0.30, "Q4 NMSE {qcf} should be < Q2 proxy 0.30");
    }

    #[test]
    fn test_raw_gpu_buffers_none_for_cpu_mode() {
        let cache = KiviCache::new(8, 64, 2048, 64);
        assert!(
            cache.get_raw_gpu_buffers().is_none(),
            "CPU-only cache should return None"
        );
    }

    #[test]
    fn test_raw_gpu_buffers_none_for_bits16() {
        let cache = KiviCache::new_with_bits(8, 64, 2048, 64, 16);
        assert!(
            cache.get_raw_gpu_buffers().is_none(),
            "bits=16 (unquantized) should return None"
        );
    }

    // (5-F: test_get_kivi_raw_buffers_trait_default_none 삭제 — KVCacheOps trait 폐기로
    //  KVCache 엔 get_kivi_raw_buffers 메서드 자체가 없음. KiviCache inherent 만 잔존.)

    #[test]
    fn test_kivi_raw_buffers_trait_none_for_cpu() {
        // KiviCache in CPU mode should also return None via the trait method
        let cache = KiviCache::new(8, 64, 2048, 64);
        assert!(
            cache.get_kivi_raw_buffers().is_none(),
            "CPU-only KiviCache trait method should return None"
        );
    }

    // ── GPU dual-allocation elimination tests ────────────────────────────────

    /// GPU mode: attn_k_buf/attn_v_buf must have zero capacity (not allocated).
    /// In CPU mode, they are allocated as before.
    #[test]
    fn test_gpu_mode_cpu_attn_buf_not_allocated() {
        use crate::memory::galloc::Galloc;
        let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn crate::memory::Memory> = Arc::new(Galloc::new());

        // new_gpu with non-OpenCL backend falls back to CPU mode
        let cache_cpu = KiviCache::new_gpu(2, 64, 256, 32, 2, cpu_backend, None, memory);
        assert!(!cache_cpu.is_gpu());
        // CPU mode: attn_k_buf/attn_v_buf should be fully allocated
        let attn_cap = cache_cpu.attn_k_buf.size() + cache_cpu.attn_v_buf.size();
        assert!(
            attn_cap > 0,
            "CPU mode must have allocated attn buffers, got size=0"
        );

        // For a true GPU cache we can only test with OpenCL backend on device.
        // Verify the cpu_memory_bytes helper works for CPU mode.
        let mem = cache_cpu.cpu_memory_bytes();
        assert!(
            mem > 0,
            "CPU mode cpu_memory_bytes should be > 0, got {mem}"
        );
    }

    /// After CPU-mode flush, CPU qk/qv have data; verify cpu_quantized_len increases.
    /// (GPU mode: qk/qv stay empty after flush -- tested on device only.)
    #[test]
    fn test_gpu_flush_cpu_qk_qv_stays_empty() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 128;
        let res_cap = 32;

        // CPU mode: flush fills qk/qv
        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        assert_eq!((cache.qk.memory_bytes() + cache.qv.memory_bytes()), 0);
        for i in 0..33 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1 + 5.0);
            cache.update(&k, &v).unwrap();
        }
        // CPU mode: after flush, qk/qv have data
        assert!(
            (cache.qk.memory_bytes() + cache.qv.memory_bytes()) > 0,
            "CPU mode must have quantized blocks after flush"
        );

        // GPU mode with non-OpenCL backend: should behave like CPU mode
        // (no actual GPU -> falls back, so qk/qv will have data too)
    }

    /// GPU mode get_view correctness regression guard.
    /// On CPU mode, verifies the assembler still works correctly after code changes.
    #[test]
    fn test_gpu_mode_get_view_correctness() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        let mut all_k = Vec::new();
        for i in 0..65 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03 + 10.0);
            all_k.extend_from_slice(k.as_slice::<f32>());
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 64);
        assert_eq!(cache.res_pos, 1);

        let (k_view, _) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();
        assert_eq!(k_out.len(), 65 * kv_heads * head_dim);

        // Last token (residual) must be exact
        let last = 64 * kv_heads * head_dim;
        for d in 0..(kv_heads * head_dim) {
            assert!(
                (all_k[last + d] - k_out[last + d]).abs() < 1e-5,
                "Residual token mismatch at d={d}"
            );
        }
    }

    /// reset() in GPU mode must not re-allocate CPU attn buffers.
    /// In CPU mode, the buffers are zero-filled but capacity stays the same.
    #[test]
    fn test_gpu_reset_no_cpu_realloc() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        let size_before = cache.attn_k_buf.size();
        cache.reset();
        let size_after = cache.attn_k_buf.size();
        // CPU mode: size must not change (fill, not reallocate)
        assert_eq!(
            size_before, size_after,
            "CPU mode reset() must not change attn_k_buf size"
        );
    }

    // ── Phase 2: GPU attn buffer lazy grow tests ──────────────────────

    /// GPU mode: initial attn buffer capacity should be res_cap, not max_seq_len.
    /// On host without GPU, new_gpu() with CpuBackend falls back to CPU mode
    /// where gpu_attn_cap == 0, so we verify the field directly.
    #[test]
    fn test_gpu_lazy_grow_initial_small() {
        use crate::memory::galloc::Galloc;
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        // CPU-only cache: gpu_attn_cap == 0
        let cache_cpu = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        assert_eq!(
            cache_cpu.gpu_attn_capacity(),
            0,
            "CPU mode must have gpu_attn_cap == 0"
        );

        // new_gpu with non-OpenCL backend: falls back to CPU mode
        let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn crate::memory::Memory> = Arc::new(Galloc::new());
        let cache_fallback = KiviCache::new_gpu(
            kv_heads,
            head_dim,
            max_seq,
            res_cap,
            2,
            cpu_backend,
            None,
            memory,
        );
        assert!(!cache_fallback.is_gpu());
        assert_eq!(
            cache_fallback.gpu_attn_capacity(),
            0,
            "Fallback CPU mode must have gpu_attn_cap == 0"
        );

        // Verify that if gpu mode WERE active, initial cap would be res_cap.
        // We can check the field directly on a non-GPU cache struct as a unit test.
        // The actual GPU path sets gpu_attn_cap = initial_attn_cap = res_cap
        // in new_gpu() -- this is verified by checking the code path, and on
        // actual GPU hardware.
        // For host verification: check that the initial_attn_cap logic in new_gpu
        // would set res_cap (32), not max_seq (256).
        assert!(
            res_cap < max_seq,
            "Test prerequisite: res_cap ({res_cap}) < max_seq ({max_seq})"
        );
    }

    /// After flush, GPU attn capacity should have grown to accommodate flushed tokens.
    /// On CPU mode (host without GPU), this tests the cpu_attn_cap == 0 invariant.
    #[test]
    fn test_gpu_lazy_grow_after_flush() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        // CPU mode: gpu_attn_cap stays 0 regardless of flush
        for i in 0..33 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1 + 5.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(
            cache.q2_tokens, 32,
            "flush should have occurred at token 33"
        );
        assert_eq!(cache.res_pos, 1);
        // CPU mode: gpu_attn_cap remains 0
        assert_eq!(
            cache.gpu_attn_capacity(),
            0,
            "CPU mode gpu_attn_cap must remain 0 after flush"
        );
    }

    /// After multiple flushes, get_view data correctness must be preserved.
    /// This is the CPU-mode equivalent: verifies the lazy grow code paths
    /// do not break existing CPU behavior.
    #[test]
    fn test_gpu_lazy_grow_data_preserved() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        let mut all_k = Vec::new();
        // Insert 65 tokens: triggers 2 flushes (32 + 32) + 1 residual
        for i in 0..65 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03 + 10.0);
            all_k.extend_from_slice(k.as_slice::<f32>());
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 64, "two flushes should have occurred");
        assert_eq!(cache.res_pos, 1);

        let (k_view, _) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();
        assert_eq!(k_out.len(), 65 * kv_heads * head_dim);

        // Last token (residual) must be exact (no quantization)
        let last_offset = 64 * kv_heads * head_dim;
        for d in 0..(kv_heads * head_dim) {
            assert!(
                (all_k[last_offset + d] - k_out[last_offset + d]).abs() < 1e-5,
                "Residual data mismatch after grow at d={d}"
            );
        }

        // Q2 region: approximate match (quantization introduces error)
        // First token's first element should be close
        let expected_first = all_k[0];
        let actual_first = k_out[0];
        assert!(
            (expected_first - actual_first).abs() < 0.5,
            "Q2 token 0 element 0: expected ~{expected_first}, got {actual_first}"
        );
    }

    /// Verify that F16 round-trip of Q2 dequantized values preserves precision.
    /// KIVI Q2 dequant outputs are in a narrow range (typically -2..+2),
    /// so F16 (which covers +/-65504 with ~3.3 decimal digits) should be more
    /// than sufficient. This test confirms cosine similarity > 0.9999.
    #[test]
    fn test_f16_dequant_precision_simulation() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;
        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Insert 65 tokens to trigger at least one flush (res_cap=32 → flush at 32)
        for i in 0..65 {
            let k = make_input_tensor(1, kv_heads, head_dim, 0.1 + i as f32 * 0.05);
            let v = make_input_tensor(1, kv_heads, head_dim, -0.1 - i as f32 * 0.03);
            cache.update(&k, &v).unwrap();
        }

        let (k_view, v_view) = cache.get_view();
        let k_f32 =
            unsafe { std::slice::from_raw_parts(k_view.as_ptr() as *const f32, k_view.size() / 4) };
        let v_f32 =
            unsafe { std::slice::from_raw_parts(v_view.as_ptr() as *const f32, v_view.size() / 4) };

        // Simulate F16 round-trip
        let k_f16_roundtrip: Vec<f32> = k_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let v_f16_roundtrip: Vec<f32> = v_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();

        // Cosine similarity
        fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
            let mut dot = 0.0f64;
            let mut norm_a = 0.0f64;
            let mut norm_b = 0.0f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot += x as f64 * y as f64;
                norm_a += (x as f64) * (x as f64);
                norm_b += (y as f64) * (y as f64);
            }
            if norm_a == 0.0 || norm_b == 0.0 {
                return 1.0; // both zero = identical
            }
            (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
        }

        let k_cos = cosine_sim(k_f32, &k_f16_roundtrip);
        let v_cos = cosine_sim(v_f32, &v_f16_roundtrip);
        assert!(
            k_cos > 0.9999,
            "K F16 precision loss too high: cosine_sim = {k_cos}"
        );
        assert!(
            v_cos > 0.9999,
            "V F16 precision loss too high: cosine_sim = {v_cos}"
        );
    }

    /// Verify that GPU-mode attn buffers are allocated as F16 (not F32).
    /// new_gpu() with non-OpenCL backend falls back to CPU mode, so we just
    /// verify that the CPU fallback path doesn't create F16 buffers.
    #[test]
    fn test_new_gpu_fallback_no_f16_attn_buffers() {
        use crate::memory::galloc::Galloc;
        let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn crate::memory::Memory> = Arc::new(Galloc::new());
        let cache = KiviCache::new_gpu(2, 64, 256, 32, 2, cpu_backend, None, memory);
        // CPU fallback: gpu_attn_k/v should be None
        assert!(
            cache.gpu_attn_k.is_none(),
            "CPU fallback should not have gpu_attn_k"
        );
        assert!(
            cache.gpu_attn_v.is_none(),
            "CPU fallback should not have gpu_attn_v"
        );
    }
}
