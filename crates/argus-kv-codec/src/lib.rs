//! KV-cache quantization codecs for argus-engine.
//!
//! Asymmetric block codecs used by the engine's quantized KV-cache container and
//! its out-of-tree plugins: 2-bit (`BlockQ2_0`), 4-bit (`BlockKVQ4`) and 8-bit
//! (`BlockKVQ8`). Kept in a no-engine-dependency leaf crate so both the engine and
//! KV-cache plugins can share one byte-identical codec without a cargo cycle. The
//! `#[repr(C)]` layouts and 12/20/36-byte sizes are part of the on-device contract
//! and are asserted at compile time.

use half::f16;

// ── Q2_0: asymmetric 2-bit quantization ─────────────────────────────────────

pub const QK2_0: usize = 32;

/// Asymmetric 2-bit quantization block.
///
/// Each block quantizes 32 f32 values into 2-bit unsigned integers [0..3].
/// Formula: `q = round((x - min) / scale)`, `scale = (max - min) / 3`.
/// Dequantize: `x ≈ q * scale + min`.
///
/// Layout: d (scale, f16) + m (minimum, f16) + qs (32×2bit = 8 bytes) = 12 bytes.
/// Compression: 0.375 bytes/element vs Q4_0's 0.5625 (33% smaller).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ2_0 {
    pub d: f16,              // scale = (max - min) / 3
    pub m: f16,              // minimum (zero point)
    pub qs: [u8; QK2_0 / 4], // 32 × 2-bit packed into 8 bytes
}

const _: () = assert!(std::mem::size_of::<BlockQ2_0>() == 12);

impl BlockQ2_0 {
    /// Quantize 32 f32 values into a Q2_0 block (asymmetric 2-bit).
    pub fn quantize(src: &[f32; QK2_0]) -> Self {
        let min_val = src.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let d = range / 3.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; QK2_0 / 4];
        for (i, qs_byte) in qs.iter_mut().enumerate() {
            let mut byte = 0u8;
            for j in 0..4 {
                let idx = i * 4 + j;
                let q = ((src[idx] - min_val) * id).round().clamp(0.0, 3.0) as u8;
                byte |= q << (j * 2);
            }
            *qs_byte = byte;
        }

        Self {
            d: f16::from_f32(d),
            m: f16::from_f32(min_val),
            qs,
        }
    }

    /// Dequantize Q2_0 block back to 32 f32 values.
    pub fn dequantize(&self, out: &mut [f32; QK2_0]) {
        let d = self.d.to_f32();
        let m = self.m.to_f32();
        for i in 0..(QK2_0 / 4) {
            let byte = self.qs[i];
            for j in 0..4 {
                let q = ((byte >> (j * 2)) & 0x03) as f32;
                out[i * 4 + j] = q * d + m;
            }
        }
    }
}

/// Quantize a contiguous f32 slice into Q2_0 blocks.
/// `src.len()` must be a multiple of QK2_0 (32).
/// Returns packed Q2_0 block data as bytes.
pub fn quantize_slice_q2(src: &[f32]) -> Vec<BlockQ2_0> {
    assert!(
        src.len().is_multiple_of(QK2_0),
        "quantize_slice_q2: length {} not a multiple of {}",
        src.len(),
        QK2_0
    );
    let n_blocks = src.len() / QK2_0;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let chunk: &[f32; QK2_0] = src[i * QK2_0..(i + 1) * QK2_0].try_into().unwrap();
        blocks.push(BlockQ2_0::quantize(chunk));
    }
    blocks
}

/// Dequantize Q2_0 blocks back to f32.
pub fn dequantize_slice_q2(blocks: &[BlockQ2_0], out: &mut [f32]) {
    assert_eq!(blocks.len() * QK2_0, out.len());
    let mut buf = [0.0f32; QK2_0];
    for (i, block) in blocks.iter().enumerate() {
        block.dequantize(&mut buf);
        out[i * QK2_0..(i + 1) * QK2_0].copy_from_slice(&buf);
    }
}

// ── KV cache asymmetric quantization blocks (multi-bit) ──────────────────────

/// Group size for KV cache quantization (same as QK2_0).
pub const QKKV: usize = 32;

/// 4-bit asymmetric KV cache quantization block.
///
/// 32 values → 16 bytes (nibble-packed) + 4 bytes (scale f16 + min f16) = 20 bytes.
/// Compression: 0.625 bytes/element.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockKVQ4 {
    pub d: f16,             // scale = (max - min) / 15
    pub m: f16,             // minimum value
    pub qs: [u8; QKKV / 2], // 32 × 4-bit packed into 16 bytes
}

const _: () = assert!(std::mem::size_of::<BlockKVQ4>() == 20);

impl BlockKVQ4 {
    /// Quantize 32 f32 values into asymmetric 4-bit.
    pub fn quantize(src: &[f32; QKKV]) -> Self {
        let min_val = src.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let d = range / 15.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; QKKV / 2];
        for (i, qs_byte) in qs.iter_mut().enumerate() {
            let lo = ((src[i * 2] - min_val) * id).round().clamp(0.0, 15.0) as u8;
            let hi = ((src[i * 2 + 1] - min_val) * id).round().clamp(0.0, 15.0) as u8;
            *qs_byte = lo | (hi << 4);
        }

        Self {
            d: f16::from_f32(d),
            m: f16::from_f32(min_val),
            qs,
        }
    }

    /// Dequantize 4-bit block back to 32 f32 values.
    pub fn dequantize(&self, out: &mut [f32; QKKV]) {
        let d = self.d.to_f32();
        let m = self.m.to_f32();
        for (i, &byte) in self.qs.iter().enumerate() {
            let lo = (byte & 0x0F) as f32;
            let hi = (byte >> 4) as f32;
            out[i * 2] = lo * d + m;
            out[i * 2 + 1] = hi * d + m;
        }
    }
}

/// 8-bit asymmetric KV cache quantization block.
///
/// 32 values → 32 bytes + 4 bytes (scale f16 + min f16) = 36 bytes.
/// Compression: 1.125 bytes/element.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockKVQ8 {
    pub d: f16,         // scale = (max - min) / 255
    pub m: f16,         // minimum value
    pub qs: [u8; QKKV], // 32 × 8-bit
}

const _: () = assert!(std::mem::size_of::<BlockKVQ8>() == 36);

impl BlockKVQ8 {
    /// Quantize 32 f32 values into asymmetric 8-bit.
    pub fn quantize(src: &[f32; QKKV]) -> Self {
        let min_val = src.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let d = range / 255.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; QKKV];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((src[i] - min_val) * id).round().clamp(0.0, 255.0) as u8;
        }

        Self {
            d: f16::from_f32(d),
            m: f16::from_f32(min_val),
            qs,
        }
    }

    /// Dequantize 8-bit block back to 32 f32 values.
    pub fn dequantize(&self, out: &mut [f32; QKKV]) {
        let d = self.d.to_f32();
        let m = self.m.to_f32();
        for (i, &q) in self.qs.iter().enumerate() {
            out[i] = q as f32 * d + m;
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    // ── Q2_0 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_block_q2_0_round_trip() {
        // Spread of values: 0..31 mapped to [0.0, 3.1]
        let src: [f32; QK2_0] = std::array::from_fn(|i| i as f32 * 0.1);
        let block = BlockQ2_0::quantize(&src);
        let mut dst = [0.0f32; QK2_0];
        block.dequantize(&mut dst);

        // 2-bit has only 4 levels → max error ≈ scale/2 ≈ range/(2*3)
        let range = 3.1f32;
        let max_err = range / 6.0 + 0.01; // ~0.527, +epsilon for f16
        for i in 0..QK2_0 {
            assert!(
                (src[i] - dst[i]).abs() < max_err,
                "q2 round-trip error at {i}: src={}, dst={}, err={}",
                src[i],
                dst[i],
                (src[i] - dst[i]).abs()
            );
        }
    }

    #[test]
    fn test_block_q2_0_zeros() {
        let src = [0.0f32; QK2_0];
        let block = BlockQ2_0::quantize(&src);
        let mut dst = [0.0f32; QK2_0];
        block.dequantize(&mut dst);
        for val in dst {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_block_q2_0_constant() {
        // All same value → d=0, all should dequantize to that value
        let src = [42.0f32; QK2_0];
        let block = BlockQ2_0::quantize(&src);
        let mut dst = [0.0f32; QK2_0];
        block.dequantize(&mut dst);
        for val in dst {
            assert!(
                (val - 42.0).abs() < 0.1,
                "constant q2: expected ~42.0, got {val}"
            );
        }
    }

    #[test]
    fn test_block_q2_0_negative_range() {
        // Negative range: [-10, -1]
        let src: [f32; QK2_0] = std::array::from_fn(|i| -10.0 + (i as f32 * 9.0 / 31.0));
        let block = BlockQ2_0::quantize(&src);
        let mut dst = [0.0f32; QK2_0];
        block.dequantize(&mut dst);

        let range = 9.0f32;
        let max_err = range / 6.0 + 0.05;
        for i in 0..QK2_0 {
            assert!(
                (src[i] - dst[i]).abs() < max_err,
                "negative range q2 error at {i}: src={}, dst={}",
                src[i],
                dst[i]
            );
        }
    }

    #[test]
    fn test_block_q2_0_manual_pack() {
        // Manually verify bit packing: values [0, 1, 2, 3, ...] repeating
        let mut src = [0.0f32; QK2_0];
        // Range 0..3 exactly → d=1.0, m=0.0
        for i in 0..QK2_0 {
            src[i] = (i % 4) as f32;
        }
        let block = BlockQ2_0::quantize(&src);
        let mut dst = [0.0f32; QK2_0];
        block.dequantize(&mut dst);

        // Should be exact (or near-exact due to f16)
        for i in 0..QK2_0 {
            assert!(
                (src[i] - dst[i]).abs() < 0.01,
                "manual pack q2 at {i}: src={}, dst={}",
                src[i],
                dst[i]
            );
        }
    }

    #[test]
    fn test_block_q2_0_dequantize_known() {
        // Construct a block with known values and verify dequantize
        let block = BlockQ2_0 {
            d: f16::from_f32(2.0),
            m: f16::from_f32(-1.0),
            qs: [0b11_10_01_00; QK2_0 / 4], // q = [0, 1, 2, 3] repeating
        };
        let mut out = [0.0f32; QK2_0];
        block.dequantize(&mut out);
        // Expected: q*d + m = [0*2-1, 1*2-1, 2*2-1, 3*2-1] = [-1, 1, 3, 5]
        for i in (0..QK2_0).step_by(4) {
            assert!((out[i] - (-1.0)).abs() < 0.01);
            assert!((out[i + 1] - 1.0).abs() < 0.01);
            assert!((out[i + 2] - 3.0).abs() < 0.01);
            assert!((out[i + 3] - 5.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantize_dequantize_slice_q2() {
        let n = QK2_0 * 4; // 128 elements = 4 blocks
        let src: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let blocks = quantize_slice_q2(&src);
        assert_eq!(blocks.len(), 4);
        let mut dst = vec![0.0f32; n];
        dequantize_slice_q2(&blocks, &mut dst);

        let range = 12.7f32;
        let max_err = range / 6.0 + 0.1;
        for i in 0..n {
            assert!(
                (src[i] - dst[i]).abs() < max_err,
                "slice q2 error at {i}: src={}, dst={}",
                src[i],
                dst[i]
            );
        }
    }

    #[test]
    #[should_panic(expected = "not a multiple")]
    fn test_quantize_slice_q2_bad_len() {
        let src = vec![0.0f32; 33]; // not multiple of 32
        quantize_slice_q2(&src);
    }

    // ── BlockKVQ4 tests ──────────────────────────────────────────────────

    #[test]
    fn test_kvq4_round_trip() {
        let src: [f32; QKKV] = std::array::from_fn(|i| i as f32 * 0.1);
        let block = BlockKVQ4::quantize(&src);
        let mut dst = [0.0f32; QKKV];
        block.dequantize(&mut dst);
        let range = 3.1f32;
        let max_err = range / 30.0 + 0.02; // 4-bit: 16 levels, max error ≈ range/30
        for i in 0..QKKV {
            assert!(
                (src[i] - dst[i]).abs() < max_err,
                "KVQ4 error at {i}: src={}, dst={}, err={}",
                src[i],
                dst[i],
                (src[i] - dst[i]).abs()
            );
        }
    }

    #[test]
    fn test_kvq4_zeros() {
        let src = [0.0f32; QKKV];
        let block = BlockKVQ4::quantize(&src);
        let mut dst = [0.0f32; QKKV];
        block.dequantize(&mut dst);
        for val in dst {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_kvq4_known_values() {
        let block = BlockKVQ4 {
            d: f16::from_f32(1.0),
            m: f16::from_f32(-2.0),
            qs: [0x31; QKKV / 2], // lo=1, hi=3 repeating
        };
        let mut out = [0.0f32; QKKV];
        block.dequantize(&mut out);
        for i in (0..QKKV).step_by(2) {
            assert!((out[i] - (-1.0)).abs() < 0.01, "lo: {} != -1.0", out[i]); // 1*1 + (-2) = -1
            assert!((out[i + 1] - 1.0).abs() < 0.01, "hi: {} != 1.0", out[i + 1]); // 3*1 + (-2) = 1
        }
    }

    // ── BlockKVQ8 tests ──────────────────────────────────────────────────

    #[test]
    fn test_kvq8_round_trip() {
        let src: [f32; QKKV] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.5);
        let block = BlockKVQ8::quantize(&src);
        let mut dst = [0.0f32; QKKV];
        block.dequantize(&mut dst);
        let range = 15.5f32;
        let max_err = range / 510.0 + 0.02; // 8-bit: 256 levels
        for i in 0..QKKV {
            assert!(
                (src[i] - dst[i]).abs() < max_err,
                "KVQ8 error at {i}: src={}, dst={}, err={}",
                src[i],
                dst[i],
                (src[i] - dst[i]).abs()
            );
        }
    }

    #[test]
    fn test_kvq8_zeros() {
        let src = [0.0f32; QKKV];
        let block = BlockKVQ8::quantize(&src);
        let mut dst = [0.0f32; QKKV];
        block.dequantize(&mut dst);
        for val in dst {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_kvq8_high_precision() {
        // 8-bit should have much lower error than 2-bit
        let src: [f32; QKKV] = std::array::from_fn(|i| (i as f32) * 0.1);
        let q8 = BlockKVQ8::quantize(&src);
        let q2 = BlockQ2_0::quantize(&src);
        let mut dst_q8 = [0.0f32; QKKV];
        let mut dst_q2 = [0.0f32; QKKV];
        q8.dequantize(&mut dst_q8);
        q2.dequantize(&mut dst_q2);
        let mse_q8: f32 = src
            .iter()
            .zip(dst_q8.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / QKKV as f32;
        let mse_q2: f32 = src
            .iter()
            .zip(dst_q2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / QKKV as f32;
        assert!(
            mse_q8 < mse_q2,
            "Q8 MSE ({mse_q8}) should be < Q2 MSE ({mse_q2})"
        );
    }

    #[test]
    fn test_kv_block_sizes() {
        assert_eq!(std::mem::size_of::<BlockKVQ4>(), 20);
        assert_eq!(std::mem::size_of::<BlockKVQ8>(), 36);
    }
}
