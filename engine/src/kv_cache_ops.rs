//! KV cache 공통 trait 및 레이아웃 타입 (L2 shared identifier).
//!
//! `KVCacheOps` trait은 `KVCache` (F32/F16/Q4_0 + eviction)와 `KiviCache`
//! (Q2 + residual) 양쪽을 추상화하는 OCP extension point다. L1 backend
//! (`backend/opencl/plan.rs` 등)가 직접 import할 수 있도록 L2에 위치한다
//! (§13.8-G shared identifier promotion).
//!
//! `KVLayout`, `KiviRawBuffers`는 `KVCacheOps` method signature에서 사용되므로
//! 함께 위치한다.

use crate::tensor::Tensor;

/// KV cache 메모리 레이아웃.
///
/// - `SeqMajor`: `[batch, seq_pos, kv_heads, head_dim]` — positions contiguous across heads.
/// - `HeadMajor`: `[batch, kv_heads, seq_pos, head_dim]` — each head's positions contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVLayout {
    SeqMajor,
    HeadMajor,
}

/// Raw GPU buffer references for native KIVI fused attention.
///
/// Provides direct access to quantized KV blocks and F32 residual buffers
/// without intermediate dequantization. Used by `attention_gen_kivi` kernel.
pub struct KiviRawBuffers<'a> {
    /// Quantized key blocks (Q2/Q4/Q8 packed).
    pub qk_buf: &'a Tensor,
    /// Quantized value blocks (Q2/Q4/Q8 packed).
    pub qv_buf: &'a Tensor,
    /// F32 residual keys [kv_heads, res_cap, head_dim].
    pub res_k: &'a Tensor,
    /// F32 residual values [kv_heads, res_cap, head_dim].
    pub res_v: &'a Tensor,
    /// Number of tokens in quantized storage.
    pub q_tokens: usize,
    /// Number of valid tokens in residual buffer.
    pub res_tokens: usize,
    /// Residual buffer capacity.
    pub res_cap: usize,
    /// Quantization bit-width (2, 4, or 8).
    pub bits: u8,
}
