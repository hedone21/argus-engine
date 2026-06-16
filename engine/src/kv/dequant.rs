//! Shared KV dequantization helpers.
//!
//! These were formerly defined inside `d2o_handler.rs`, but they are not D2O-specific: the
//! [`StageCtx`](argus_extension_api::StageCtx) `tensor(Key)`/`tensor(Value)` handles (in
//! `eviction/stage_registry.rs`) delegate to [`dequantize_k`]/[`dequantize_v`] so every stage reads
//! raw K/V identically. When D2O / R-KV were extracted to out-of-tree plugin crates, these
//! dequant helpers stayed in the engine core (the plugins read K via `ctx.dequant_k`, and carry
//! their own `cosine_similarity`).

use crate::buffer::DType;
use crate::kv::kv_cache::KVCache;
use crate::quant::{BlockQ4_0, QK4_0};
use half::f16;

/// Dequantize a K vector at (pos, head) into the output buffer.
/// Works for F32, F16, and Q4_0 dtypes.
///
/// `pub(crate)`: `StageCtx::dequant_k` (the `KeyHandle` in `stage_registry.rs`) delegates to this
/// canonical implementation, so all stages — including the out-of-tree `d2o` plugin reading K via
/// `ctx.dequant_k` — see bit-identical raw K.
pub(crate) fn dequantize_k(
    cache: &KVCache,
    pos: usize,
    head: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    match cache.k_buffer.dtype() {
        DType::F32 => {
            let k = cache.k_buffer.as_slice::<f32>();
            let off = cache.offset(pos, head);
            out[..head_dim].copy_from_slice(&k[off..off + head_dim]);
        }
        DType::F16 => {
            let k = cache.k_buffer.as_slice::<f16>();
            let off = cache.offset(pos, head);
            for d in 0..head_dim {
                out[d] = k[off + d].to_f32();
            }
        }
        DType::Q4_0 => {
            let k = cache.k_buffer.as_slice::<BlockQ4_0>();
            let blocks_per_pos = head_dim / QK4_0;
            let block_off = cache.q4_block_offset(pos, head, blocks_per_pos);
            for bi in 0..blocks_per_pos {
                let mut tmp = [0.0f32; QK4_0];
                k[block_off + bi].dequantize(&mut tmp);
                let dst = bi * QK4_0;
                out[dst..dst + QK4_0].copy_from_slice(&tmp);
            }
        }
        _ => {}
    }
}

/// Dequantize a V vector at (pos, head) into the output buffer.
/// Works for F32, F16, and Q4_0 dtypes — exact mirror of [`dequantize_k`] on `v_buffer`.
///
/// V uses the IDENTICAL `[1, kv_heads, capacity, head_dim]` layout and `offset`/`q4_block_offset`
/// as K (confirmed by `apply_weighted_merges` which dispatches K/V independently over the same
/// offsets). `pub(crate)`: `StageCtx::tensor(Value)` (the `ValueHandle`) delegates here — CAOTE's `v_i`.
pub(crate) fn dequantize_v(
    cache: &KVCache,
    pos: usize,
    head: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    match cache.v_buffer.dtype() {
        DType::F32 => {
            let v = cache.v_buffer.as_slice::<f32>();
            let off = cache.offset(pos, head);
            out[..head_dim].copy_from_slice(&v[off..off + head_dim]);
        }
        DType::F16 => {
            let v = cache.v_buffer.as_slice::<f16>();
            let off = cache.offset(pos, head);
            for d in 0..head_dim {
                out[d] = v[off + d].to_f32();
            }
        }
        DType::Q4_0 => {
            let v = cache.v_buffer.as_slice::<BlockQ4_0>();
            let blocks_per_pos = head_dim / QK4_0;
            let block_off = cache.q4_block_offset(pos, head, blocks_per_pos);
            for bi in 0..blocks_per_pos {
                let mut tmp = [0.0f32; QK4_0];
                v[block_off + bi].dequantize(&mut tmp);
                let dst = bi * QK4_0;
                out[dst..dst + QK4_0].copy_from_slice(&tmp);
            }
        }
        _ => {}
    }
}
