//! D3 — engine-side executor for [`KVFormatPlan`] (the format / precision axis).
//!
//! `apply_format_plan` is the format twin of the eviction executor `execute_kv_plan`. It applies a
//! plugin-produced [`KVFormatPlan`] to the engine's KV container.
//!
//! Honesty contract (the reason this exists rather than silently no-op'ing): a plan whose effective
//! format varies ACROSS heads or ACROSS tokens within a single layer cannot be stored by any current
//! container — [`KVCache`] holds a single dtype per layer buffer and the quant-window container a
//! single bit-width per layer — so such a plan is REJECTED with
//! [`FormatApplyError::HeterogeneousUnsupported`] instead of being mis-stored. "Expressible (a
//! well-formed plan value) != executable (the engine can re-materialize it)".
//!
//! Scope: Gate-0 no-op + heterogeneous rejection + the L1 uniform-per-layer re-encode for a BARE
//! per-layer base swap (no overrides, `base` != current stored format), executed on the host (CPU) by
//! dequantizing the resident tokens from the old format and requantizing into a freshly-allocated
//! new-format buffer (typed floor formats f32/f16/q4_0). A whole-layer *override* re-encode and any
//! heterogeneous-within-layer plan remain unexecuted (the latter rejected with
//! [`FormatApplyError::HeterogeneousUnsupported`]). The signature is `&mut KVCache` because the L1
//! path swaps the layer's K/V buffers in place. Device-resident (GPU) buffers are handled by staging
//! through a host mirror (`reencode_typed_device`): the bytes are downloaded (`host_snapshot` /
//! `read_buffer`), re-encoded on the CPU, and uploaded to fresh device buffers (`alloc_kv` /
//! `write_buffer`) — needs the cache's grow allocator (`memory`), else it is skipped upstream.
//!
//! LIVENESS: `apply_format_plan`'s first production caller is
//! [`FormatReencodeStage`](crate::stages::kv::format_reencode::FormatReencodeStage) — a `PrefillEnd`
//! pipeline stage armed when `--kv-format` resolves to a registered `KVFormatPolicy`. This module is
//! the *runtime* (post-allocation) re-encode executor, the format twin of the eviction executor
//! `execute_kv_plan` (kv::eviction::stage_registry). Production format-honesty *also* relies, at
//! construction time, on the twin
//! [`per_layer_storage_from_policy`](crate::session::bin_setup::per_layer_storage_from_policy)
//! (bin_setup.rs:578, rejecting override-bearing plans at bin_setup.rs:599 *before* allocation). The
//! stage feeds this fn only typed-floor layers it can re-encode — host-resident, or device-resident
//! with a grow allocator (it skips opaque / non-floor / allocator-less-device layers, which the
//! construction-time allocator owns), so the honesty arms below are a live runtime guarantee on that
//! path — not a dormant contract.

use crate::buffer::DType;
use crate::kv::dequant::{dequantize_k, dequantize_v};
use crate::kv::kv_cache::KVCache;
use crate::memory::host::shared::SharedBuffer;
use crate::quant::{BlockQ4_0, QK4_0};
use crate::tensor::Tensor;
use argus_extension_api::{KVFormatPlan, KeepSpec};
use half::f16;
use std::sync::Arc;

/// Why a [`KVFormatPlan`] could not be applied to the current container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatApplyError {
    /// The plan assigns different formats to different heads, or to a token SUBSET within one layer.
    /// No current container can hold heterogeneous-within-layer precision (one dtype per [`KVCache`]
    /// layer buffer; one bit-width per quant-window layer), so it is rejected rather than mis-stored.
    /// Faithful per-head / per-token precision needs a heterogeneous-membership store (L2).
    HeterogeneousUnsupported,
    /// The plan names a format the current backend cannot decode (g2 backend-capability feedback).
    UnsupportedFormat(String),
    /// A uniform-per-layer precision change that is well-formed, but whose execution (per-layer
    /// re-allocation + re-encode) is not yet wired (L1, deferred).
    UniformReencodeNotWired,
}

impl std::fmt::Display for FormatApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatApplyError::HeterogeneousUnsupported => write!(
                f,
                "KVFormatPlan assigns heterogeneous-within-layer precision (per-head or per-token); \
                 no current container can store it — needs a heterogeneous-membership store (L2)"
            ),
            FormatApplyError::UnsupportedFormat(name) => {
                write!(
                    f,
                    "KVFormatPlan names a format the backend cannot decode: {name}"
                )
            }
            FormatApplyError::UniformReencodeNotWired => write!(
                f,
                "uniform-per-layer precision change is well-formed but per-layer re-encode is not yet \
                 wired (L1, deferred)"
            ),
        }
    }
}

impl std::error::Error for FormatApplyError {}

/// The current stored-format name for `cache`, derived from its KV dtype (mirror of the floor's
/// `register_kv_format!` names). Used to detect the Gate-0 no-op (`base` == current format).
fn current_format_name(cache: &KVCache) -> &'static str {
    match cache.kv_dtype() {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::Q4_0 => "q4_0",
        _ => "unknown",
    }
}

/// Applies a [`KVFormatPlan`] to `cache` for one layer. See the module docs for the honesty contract.
///
/// Returns `Ok(())` only for the Gate-0 no-op (base == current stored format, no overrides). Any
/// heterogeneous-within-layer plan is rejected with [`FormatApplyError::HeterogeneousUnsupported`];
/// a uniform-per-layer change is reported as [`FormatApplyError::UniformReencodeNotWired`] (L1).
pub fn apply_format_plan(
    cache: &mut KVCache,
    plan: &KVFormatPlan,
    _layer: usize,
    _n_layers: usize,
) -> Result<(), FormatApplyError> {
    // Gate-0: base == current stored format AND no overrides => byte-identical no-op.
    if plan.overrides.is_empty() && plan.base.0 == current_format_name(cache) {
        return Ok(());
    }
    // Heterogeneous-within-layer? A `PerHead` override, or a `LayerWide` override that covers only a
    // token SUBSET (not the whole resident layer), assigns a different format to part of a layer —
    // unholdable by any current single-precision-per-layer container. Reject honestly.
    let resident = cache.current_pos();
    for ov in &plan.overrides {
        let heterogeneous = match &ov.region {
            KeepSpec::PerHead(_) => true,
            KeepSpec::LayerWide(positions) => positions.len() != resident,
        };
        if heterogeneous {
            return Err(FormatApplyError::HeterogeneousUnsupported);
        }
    }
    // A BARE per-layer base swap (overrides empty — Gate-0 already returned for base == current, so
    // here base != current) is the canonical L1 uniform re-encode: execute it on the host (CPU).
    if plan.overrides.is_empty() {
        return reencode_uniform(cache, &plan.base.0);
    }
    // A whole-layer *override* (non-empty overrides, each covering the full resident layer) is also
    // uniform-per-layer and well-formed, but its re-encode is a separate slice (not yet wired).
    Err(FormatApplyError::UniformReencodeNotWired)
}

/// True for the typed floor formats the host dequant/requant path can read & write directly.
fn is_typed_floor(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::Q4_0)
}

/// Maps a floor format name (mirror of `current_format_name`) to its typed [`DType`], or `None` for
/// formats this host path cannot materialize (opaque/.so codecs such as q2_0, or q8_0/q4_1).
fn typed_floor_dtype(name: &str) -> Option<DType> {
    match name {
        "f32" => Some(DType::F32),
        "f16" => Some(DType::F16),
        "q4_0" => Some(DType::Q4_0),
        _ => None,
    }
}

/// Byte size of a typed buffer holding `n_values` elements in `dtype` (Q4_0 is block-quantized, so
/// it is sized in 18-byte [`BlockQ4_0`] blocks — `DType::size()` reports 1 for Q4_0 and must not be
/// used here). Mirrors the typed arm of `KVCache::grow` / `alloc_mixed_kv_caches`.
fn typed_byte_size(n_values: usize, dtype: DType) -> usize {
    match dtype {
        DType::Q4_0 => (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>(),
        _ => n_values * dtype.size(),
    }
}

/// Uniform per-layer re-encode of `cache` to the format named `target_name`.
///
/// Host-resident caches re-encode in place (CPU dequant/requant). Device-resident (GPU) caches stage
/// through a host mirror (`reencode_typed` → `reencode_typed_device`): the bytes are downloaded,
/// re-encoded on the CPU, and uploaded to fresh device buffers. Rejects (honestly, never silently) a
/// non-typed-floor source/target (opaque codecs) and a Q4_0 target whose `head_dim` is not a multiple
/// of `QK4_0`.
fn reencode_uniform(cache: &mut KVCache, target_name: &str) -> Result<(), FormatApplyError> {
    let source = cache.kv_dtype();
    if !is_typed_floor(source) {
        return Err(FormatApplyError::UnsupportedFormat(format!(
            "re-encode source '{}' is not a typed floor format (f32/f16/q4_0)",
            current_format_name(cache)
        )));
    }
    let target = typed_floor_dtype(target_name)
        .ok_or_else(|| FormatApplyError::UnsupportedFormat(target_name.to_string()))?;
    if source == target {
        // Defensive: equal source/target needs no work (Gate-0 normally catches base == current).
        return Ok(());
    }
    reencode_typed(cache, target)
}

/// Re-encode every resident token of `cache` from its current typed format into a fresh `target`
/// buffer, then swap the new buffers in. Capacity / kv_heads / head_dim / layout / current_pos are
/// preserved (only the stored dtype changes), so the engine's `offset()` math is unchanged.
fn reencode_typed(cache: &mut KVCache, target: DType) -> Result<(), FormatApplyError> {
    let kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();
    let capacity = cache.capacity();
    let resident = cache.current_pos();

    // Q4_0 represents a head's `head_dim` values as `head_dim / QK4_0` blocks; a non-multiple
    // head_dim cannot be tiled into whole blocks.
    if target == DType::Q4_0 && !head_dim.is_multiple_of(QK4_0) {
        return Err(FormatApplyError::UnsupportedFormat(format!(
            "q4_0 re-encode needs head_dim a multiple of {QK4_0}, got {head_dim}"
        )));
    }

    // Device-resident cache (Adreno UMA: host ptr is null until mapped): the host dequant below reads
    // via `as_slice`, so stage through a host mirror — download the bytes, re-encode on the CPU, and
    // upload to fresh device buffers. The result is byte-identical to a construction-time device alloc
    // of `target` (the host re-encode is deterministic: `BlockQ4_0::quantize` etc.).
    if cache.k_buffer.buffer().is_gpu_buffer() {
        return reencode_typed_device(cache, target);
    }

    let backend = cache.k_buffer.backend().clone();
    let shape = cache.k_buffer.shape().clone();
    let n_values = capacity * kv_heads * head_dim;
    let bytes = typed_byte_size(n_values, target);

    // New-format K/V buffers, host-backed (SharedBuffer) and sized for the FULL capacity so the
    // unchanged `offset()`/`q4_block_offset()` indexing addresses them correctly.
    let mut new_k = Tensor::new(
        shape.clone(),
        Arc::new(SharedBuffer::new(bytes, target)),
        backend.clone(),
    );
    let mut new_v = Tensor::new(shape, Arc::new(SharedBuffer::new(bytes, target)), backend);

    // Per (pos, head): dequant the old format to f32, then requant into the new format. This is the
    // exact inverse of `dequantize_k`/`_v`; the Q4_0 write uses `BlockQ4_0::quantize`, so the result
    // is byte-identical to a direct re-quantization of the dequantized inputs.
    let mut row = vec![0.0f32; head_dim];
    for pos in 0..resident {
        for head in 0..kv_heads {
            dequantize_k(cache, pos, head, head_dim, &mut row);
            write_typed_row(&mut new_k, cache, pos, head, head_dim, target, &row);
            dequantize_v(cache, pos, head, head_dim, &mut row);
            write_typed_row(&mut new_v, cache, pos, head, head_dim, target, &row);
        }
    }

    cache.k_buffer = new_k;
    cache.v_buffer = new_v;
    Ok(())
}

/// Device-resident re-encode: download the cache to a geometry-identical host mirror
/// ([`KVCache::host_snapshot`] — a `read_buffer` of both buffers), re-encode that mirror on the CPU
/// path ([`reencode_typed`] recursion lands in the host branch since the mirror is host-backed), then
/// upload the new-format host buffers to fresh device buffers via the cache's grow allocator and swap.
///
/// The uploaded bytes are byte-identical to a construction-time device alloc of `target`: the host
/// re-encode is deterministic (`BlockQ4_0::quantize` etc.), and the device alloc + `write_buffer`
/// reproduces those exact bytes. The fused GPU plan is F16-only, so after a non-F16 re-encode the
/// caller invalidates it (`on_kv_reencode`) and decode falls to the dyn path (the q4_0-KV-on-GPU
/// fallback) — see `decode_loop.rs` / `model_forward.rs`.
fn reencode_typed_device(cache: &mut KVCache, target: DType) -> Result<(), FormatApplyError> {
    let backend = cache.k_buffer.backend().clone();
    let memory = cache.memory().ok_or_else(|| {
        FormatApplyError::UnsupportedFormat(
            "device re-encode needs the cache's grow allocator (memory=None)".into(),
        )
    })?;

    // Flush any pending device writes, then download device → host (read_buffer).
    backend
        .synchronize()
        .map_err(|e| FormatApplyError::UnsupportedFormat(format!("device synchronize: {e}")))?;
    let mut mirror = cache
        .host_snapshot()
        .map_err(|e| FormatApplyError::UnsupportedFormat(format!("device→host snapshot: {e}")))?;

    // Re-encode the host mirror on the CPU path (host buffers → `as_slice` dequant works).
    reencode_typed(&mut mirror, target)?;

    // Upload the new-format host buffers to fresh device buffers, swap (current_pos/geometry kept).
    cache.k_buffer = upload_host_to_device(&mirror.k_buffer, memory.as_ref(), &backend, target)?;
    cache.v_buffer = upload_host_to_device(&mirror.v_buffer, memory.as_ref(), &backend, target)?;
    Ok(())
}

/// Allocate a fresh device buffer of `target` format via `memory` and copy `host_t`'s bytes into it
/// (`write_buffer`), returning a device [`Tensor`] of the same shape. Mirror of
/// `read_device_tensor_to_host` (the inverse host→device copy).
fn upload_host_to_device(
    host_t: &Tensor,
    memory: &dyn crate::memory::Memory,
    backend: &Arc<dyn crate::backend::Backend>,
    target: DType,
) -> Result<Tensor, FormatApplyError> {
    let bytes = host_t.size();
    let dev_buf = memory
        .alloc_kv(bytes, target)
        .map_err(|e| FormatApplyError::UnsupportedFormat(format!("device alloc_kv: {e}")))?;
    let mut dev_t = Tensor::new(host_t.shape().clone(), dev_buf, backend.clone());
    // SAFETY: `host_t.buffer()` is a host SharedBuffer with exactly `bytes` valid bytes; `write_buffer`
    // copies exactly `bytes` and does not retain the pointer past the call.
    let src = unsafe { std::slice::from_raw_parts(host_t.buffer().as_ptr(), bytes) };
    backend
        .write_buffer(&mut dev_t, src)
        .map_err(|e| FormatApplyError::UnsupportedFormat(format!("device write_buffer: {e}")))?;
    Ok(dev_t)
}

/// Write one (pos, head) row of `head_dim` f32 values into `buf` encoded as `dtype`, at the same
/// offset the cache uses for reads (`offset`/`q4_block_offset` on the still-old `cache` geometry).
fn write_typed_row(
    buf: &mut Tensor,
    cache: &KVCache,
    pos: usize,
    head: usize,
    head_dim: usize,
    dtype: DType,
    row: &[f32],
) {
    match dtype {
        DType::F32 => {
            let off = cache.offset(pos, head);
            let dst = buf.as_mut_slice::<f32>();
            dst[off..off + head_dim].copy_from_slice(&row[..head_dim]);
        }
        DType::F16 => {
            let off = cache.offset(pos, head);
            let dst = buf.as_mut_slice::<f16>();
            for d in 0..head_dim {
                dst[off + d] = f16::from_f32(row[d]);
            }
        }
        DType::Q4_0 => {
            let blocks_per_pos = head_dim / QK4_0;
            let block_off = cache.q4_block_offset(pos, head, blocks_per_pos);
            let dst = buf.as_mut_slice::<BlockQ4_0>();
            for bi in 0..blocks_per_pos {
                let mut blk = [0.0f32; QK4_0];
                blk.copy_from_slice(&row[bi * QK4_0..(bi + 1) * QK4_0]);
                dst[block_off + bi] = BlockQ4_0::quantize(&blk);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use argus_extension_api::{FormatId, FormatOverride, MergeAxis};
    use std::sync::Arc;

    const MAX_SEQ: usize = 32;
    const HD: usize = 4;
    const N_KV: usize = 2;

    /// An F16 KVCache with `resident` tokens written (current_pos = resident).
    fn cache_f16(resident: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let buf = || {
            Arc::new(SharedBuffer::new(
                N_KV * MAX_SEQ * HD * std::mem::size_of::<half::f16>(),
                DType::F16,
            ))
        };
        let shape = Shape::new(vec![1, MAX_SEQ, N_KV, HD]);
        let mut c = KVCache::new(
            Tensor::new(shape.clone(), buf(), backend.clone()),
            Tensor::new(shape, buf(), backend),
            MAX_SEQ,
        );
        c.set_current_pos(resident);
        c
    }

    /// Gate-0: base == current stored format + no overrides => Ok (byte-identical no-op).
    #[test]
    fn apply_format_plan_gate0_noop_ok() {
        let mut c = cache_f16(8);
        let plan = KVFormatPlan {
            base: FormatId("f16".into()),
            overrides: vec![],
        };
        assert_eq!(apply_format_plan(&mut c, &plan, 0, 1), Ok(()));
    }

    /// Per-token SUBSET override (two-tier) is heterogeneous-within-layer => rejected, not mis-stored.
    #[test]
    fn apply_format_plan_per_token_subset_rejected() {
        let mut c = cache_f16(8); // resident = 8, override covers only {2,3} => subset
        let plan = KVFormatPlan {
            base: FormatId("q2".into()),
            overrides: vec![FormatOverride {
                region: KeepSpec::LayerWide(vec![2, 3]),
                format: FormatId("f16".into()),
                side: MergeAxis::Both,
            }],
        };
        assert_eq!(
            apply_format_plan(&mut c, &plan, 0, 1),
            Err(FormatApplyError::HeterogeneousUnsupported)
        );
    }

    /// Per-head override is heterogeneous-within-layer => rejected (no per-head precision container).
    #[test]
    fn apply_format_plan_per_head_rejected() {
        let mut c = cache_f16(8);
        let plan = KVFormatPlan {
            base: FormatId("q2".into()),
            overrides: vec![FormatOverride {
                region: KeepSpec::PerHead(vec![vec![], vec![2]]),
                format: FormatId("f16".into()),
                side: MergeAxis::Both,
            }],
        };
        assert_eq!(
            apply_format_plan(&mut c, &plan, 0, 1),
            Err(FormatApplyError::HeterogeneousUnsupported)
        );
    }

    /// A uniform-per-layer change (whole-resident-layer override) is well-formed but not yet wired.
    #[test]
    fn apply_format_plan_uniform_reencode_not_wired() {
        let mut c = cache_f16(4); // resident = 4
        let plan = KVFormatPlan {
            base: FormatId("f16".into()),
            overrides: vec![FormatOverride {
                region: KeepSpec::LayerWide(vec![0, 1, 2, 3]), // spans the whole resident layer
                format: FormatId("q4_0".into()),
                side: MergeAxis::Both,
            }],
        };
        assert_eq!(
            apply_format_plan(&mut c, &plan, 0, 1),
            Err(FormatApplyError::UniformReencodeNotWired)
        );
    }

    /// L1 (now WIRED): a BARE per-layer base swap (no overrides, base != current stored format) is the
    /// canonical uniform re-encode. The executor now PERFORMS it — dequant every resident token from
    /// the old format and requant into a freshly-allocated new-format buffer — rather than reporting it
    /// not-wired. This is a DISTINCT code path from `apply_format_plan_uniform_reencode_not_wired`
    /// (which routes through a whole-layer *override*, still not wired): here `overrides` is empty, so
    /// Gate-0 is the only `Ok` gate and it does NOT fire (base `q4_0` != current `f16`), the override
    /// loop is skipped, and control reaches the bare-base-swap re-encode.
    ///
    /// Pins the live behavior: (a) the stored dtype flips to the target, and (b) the re-encoded Q4_0
    /// blocks are BYTE-EXACT to a direct `BlockQ4_0::quantize` of the dequantized f16 inputs, with the
    /// round-trip dequant faithful within q4_0 tolerance. Non-tautological / mutation-proof: reverting
    /// the body to the old stub (`Err(UniformReencodeNotWired)`) makes `apply_format_plan` return `Err`
    /// AND leaves `kv_dtype() == F16`, so both the `Ok(())` and the dtype-flip assertions fail.
    #[test]
    fn apply_format_plan_bare_base_change_executes_round_trip_faithful() {
        const HD: usize = 64; // q4_0-valid: a multiple of QK4_0 (=32)
        const NKV: usize = 2;
        const CAP: usize = 8;
        const RESIDENT: usize = 5;

        let backend = Arc::new(CpuBackend::new());
        let nbytes = NKV * CAP * HD * std::mem::size_of::<f16>();
        let shape = Shape::new(vec![1, CAP, NKV, HD]);
        let mut c = KVCache::new(
            Tensor::new(
                shape.clone(),
                Arc::new(SharedBuffer::new(nbytes, DType::F16)),
                backend.clone(),
            ),
            Tensor::new(
                shape,
                Arc::new(SharedBuffer::new(nbytes, DType::F16)),
                backend,
            ),
            CAP,
        );
        c.set_current_pos(RESIDENT);

        // Write a known, q4-friendly pattern into the resident region of both buffers; record the
        // f16-rounded f32 originals (what a faithful round-trip must reproduce within q4_0 error).
        let pat = |pos: usize, head: usize, d: usize, salt: f32| {
            salt + pos as f32 * 0.13 + head as f32 * 0.31 + (d as f32 - HD as f32 / 2.0) * 0.05
        };
        let mut orig_k = vec![0.0f32; RESIDENT * NKV * HD];
        let mut orig_v = vec![0.0f32; RESIDENT * NKV * HD];
        for pos in 0..RESIDENT {
            for head in 0..NKV {
                let off = c.offset(pos, head);
                let idx = (pos * NKV + head) * HD;
                {
                    let ks = c.k_buffer.as_mut_slice::<f16>();
                    for d in 0..HD {
                        let x = f16::from_f32(pat(pos, head, d, 1.0));
                        ks[off + d] = x;
                        orig_k[idx + d] = x.to_f32();
                    }
                }
                {
                    let vs = c.v_buffer.as_mut_slice::<f16>();
                    for d in 0..HD {
                        let x = f16::from_f32(pat(pos, head, d, -0.7));
                        vs[off + d] = x;
                        orig_v[idx + d] = x.to_f32();
                    }
                }
            }
        }

        let plan = KVFormatPlan {
            base: FormatId("q4_0".into()), // != current f16 → not a Gate-0 no-op; bare base swap
            overrides: vec![],             // no overrides → reaches the bare-base-swap re-encode
        };
        // Executes the L1 re-encode and returns Ok — NOT the old not-wired Err.
        assert_eq!(apply_format_plan(&mut c, &plan, 0, 1), Ok(()));
        // (a) the stored format flipped to the target.
        assert_eq!(c.kv_dtype(), DType::Q4_0);

        // (b) byte-exact vs a direct BlockQ4_0::quantize of the dequantized inputs + faithful round-trip.
        let bpp = HD / QK4_0;
        let mut got = vec![0.0f32; HD];
        for pos in 0..RESIDENT {
            for head in 0..NKV {
                let idx = (pos * NKV + head) * HD;
                let bo = c.q4_block_offset(pos, head, bpp);
                let k_blocks = c.k_buffer.as_slice::<BlockQ4_0>();
                let v_blocks = c.v_buffer.as_slice::<BlockQ4_0>();
                for bi in 0..bpp {
                    let mut blk = [0.0f32; QK4_0];
                    blk.copy_from_slice(&orig_k[idx + bi * QK4_0..idx + (bi + 1) * QK4_0]);
                    let expect = BlockQ4_0::quantize(&blk);
                    assert_eq!(k_blocks[bo + bi].d, expect.d, "K block {bi} scale");
                    assert_eq!(k_blocks[bo + bi].qs, expect.qs, "K block {bi} nibbles");

                    blk.copy_from_slice(&orig_v[idx + bi * QK4_0..idx + (bi + 1) * QK4_0]);
                    let expect = BlockQ4_0::quantize(&blk);
                    assert_eq!(v_blocks[bo + bi].d, expect.d, "V block {bi} scale");
                    assert_eq!(v_blocks[bo + bi].qs, expect.qs, "V block {bi} nibbles");
                }
                // round-trip fidelity: dequant of the re-encoded K within q4_0 tolerance of the orig.
                dequantize_k(&c, pos, head, HD, &mut got);
                for bi in 0..bpp {
                    let slice = &orig_k[idx + bi * QK4_0..idx + (bi + 1) * QK4_0];
                    let max_abs = slice.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let tol = max_abs / 7.0 + 1e-3; // symmetric q4_0 step = max_abs/7
                    for j in 0..QK4_0 {
                        let o = orig_k[idx + bi * QK4_0 + j];
                        let g = got[bi * QK4_0 + j];
                        assert!((g - o).abs() <= tol, "K round-trip |{g} - {o}| > {tol}");
                    }
                }
            }
        }
    }
}
