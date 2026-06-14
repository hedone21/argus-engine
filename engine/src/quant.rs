use half::f16;

pub mod convert;
pub mod f16_bulk;

#[cfg(target_arch = "aarch64")]
pub mod flash_neon;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; QK4_0 / 2],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub d: f16,
    pub m: f16,
    pub qs: [u8; QK4_1 / 2],
}

// Ensure sizes are correct
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

impl BlockQ4_0 {
    /// Quantize 32 f32 values into a Q4_0 block.
    ///
    /// Symmetric quantization: finds max absolute value, scales to [-8, 7] range.
    pub fn quantize(src: &[f32; QK4_0]) -> Self {
        let max_abs = src.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = max_abs / 7.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut qs = [0u8; QK4_0 / 2];
        for i in 0..(QK4_0 / 2) {
            let v0 = (src[i] * id).round().clamp(-8.0, 7.0) as i8;
            let v1 = (src[i + QK4_0 / 2] * id).round().clamp(-8.0, 7.0) as i8;
            qs[i] = ((v0 + 8) as u8) | (((v1 + 8) as u8) << 4);
        }
        Self {
            d: f16::from_f32(d),
            qs,
        }
    }

    pub fn dequantize(&self, out: &mut [f32; QK4_0]) {
        let d = self.d.to_f32();
        for i in 0..(QK4_0 / 2) {
            let b = self.qs[i];
            let v0 = (b & 0x0F) as i8 - 8;
            let v1 = (b >> 4) as i8 - 8;

            out[i] = v0 as f32 * d;
            out[i + QK4_0 / 2] = v1 as f32 * d;
        }
    }
}

impl BlockQ4_1 {
    pub fn dequantize(&self, out: &mut [f32; QK4_1]) {
        let d = self.d.to_f32();
        let m = self.m.to_f32();
        for i in 0..(QK4_1 / 2) {
            let b = self.qs[i];
            let v0 = (b & 0x0F) as f32;
            let v1 = (b >> 4) as f32;

            out[i] = v0 * d + m;
            out[i + QK4_1 / 2] = v1 * d + m;
        }
    }
}

// ── Q2_0: KIVI-style asymmetric 2-bit quantization ──────────────────────────

pub const QK2_0: usize = 32;

/// Asymmetric 2-bit quantization block (KIVI paper, ICML 2024).
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

pub const QK8_0: usize = 32;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; QK8_0],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

impl BlockQ8_0 {
    pub fn dequantize(&self, out: &mut [f32; QK8_0]) {
        let d = self.d.to_f32();
        for (i, o) in out.iter_mut().enumerate().take(QK8_0) {
            *o = self.qs[i] as f32 * d;
        }
    }
}

// ── KV cache asymmetric quantization blocks (KIVI multi-bit) ─────────────────

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

    #[test]
    fn test_block_q4_0_dequantize() {
        let mut out = [0.0; QK4_0];

        // Scale = 2.0 (f16 corresponding to 2.0 is roughly 0x4000)
        let mut block = BlockQ4_0 {
            d: f16::from_f32(2.0),
            qs: [0; QK4_0 / 2],
        };

        // Let's set some specific nibbles
        // b = 0x1A (0b0001_1010)
        // v0 = 0x0A (10) -> 10 - 8 = 2
        // v1 = 0x01 (1) -> 1 - 8 = -7
        block.qs[0] = 0x1A;

        // Max nibble: 0xFF
        // v0 = 15 - 8 = 7
        // v1 = 15 - 8 = 7
        block.qs[1] = 0xFF;

        // Min nibble: 0x00
        // v0 = 0 - 8 = -8
        // v1 = 0 - 8 = -8
        block.qs[2] = 0x00;

        block.dequantize(&mut out);

        // Verification for qs[0] -> block 0 and 16
        assert_eq!(out[0], 2.0 * 2.0); // 4.0
        assert_eq!(out[16], -7.0 * 2.0); // -14.0

        // Verification for qs[1] -> block 1 and 17
        assert_eq!(out[1], 7.0 * 2.0); // 14.0
        assert_eq!(out[17], 7.0 * 2.0); // 14.0

        // Verification for qs[2] -> block 2 and 18
        assert_eq!(out[2], -8.0 * 2.0); // -16.0
        assert_eq!(out[18], -8.0 * 2.0); // -16.0

        // Unset ones should be (0 - 8) * 2.0 = -16.0
        assert_eq!(out[3], -16.0);
    }

    #[test]
    fn test_block_q4_0_zero_scale() {
        let mut out = [0.0; QK4_0];
        let block = BlockQ4_0 {
            d: f16::from_f32(0.0),
            qs: [0x55; QK4_0 / 2], // Some non-zero pattern
        };
        block.dequantize(&mut out);
        for val in out {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_block_q4_1_dequantize() {
        let mut out = [0.0; QK4_1];

        // Scale = 2.0, Min = -5.0
        let mut block = BlockQ4_1 {
            d: f16::from_f32(2.0),
            m: f16::from_f32(-5.0),
            qs: [0; QK4_1 / 2],
        };

        // b = 0x1A
        // v0 = 0x0A (10)
        // v1 = 0x01 (1)
        block.qs[0] = 0x1A;

        block.dequantize(&mut out);

        // out = v * d + m
        // out[0] = 10 * 2.0 - 5.0 = 15.0
        // out[16] = 1 * 2.0 - 5.0 = -3.0
        assert_eq!(out[0], 15.0);
        assert_eq!(out[16], -3.0);
    }

    #[test]
    fn test_block_q8_0_dequantize() {
        let mut out = [0.0; QK8_0];

        let mut block = BlockQ8_0 {
            d: f16::from_f32(0.5),
            qs: [0; QK8_0],
        };

        block.qs[0] = 10;
        block.qs[1] = -5;
        block.qs[31] = 127; // max i8

        block.dequantize(&mut out);

        assert_eq!(out[0], 5.0);
        assert_eq!(out[1], -2.5);
        assert_eq!(out[31], 63.5);
    }

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(std::mem::size_of::<BlockQ2_0>(), 12);
    }

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

    #[test]
    fn test_block_q4_0_quantize_round_trip() {
        // Values within representable range
        let src: [f32; QK4_0] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.5);
        let block = BlockQ4_0::quantize(&src);
        let mut dst = [0.0f32; QK4_0];
        block.dequantize(&mut dst);
        for i in 0..QK4_0 {
            assert!(
                (src[i] - dst[i]).abs() < 1.5,
                "round-trip error at {i}: src={}, dst={}",
                src[i],
                dst[i]
            );
        }
    }

    #[test]
    fn test_block_q4_0_quantize_zeros() {
        let src = [0.0f32; QK4_0];
        let block = BlockQ4_0::quantize(&src);
        let mut dst = [0.0f32; QK4_0];
        block.dequantize(&mut dst);
        for val in dst {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_block_q4_1_zero_scale() {
        let mut out = [0.0; QK4_1];
        let block = BlockQ4_1 {
            d: f16::from_f32(0.0),
            m: f16::from_f32(3.0),
            qs: [0xAB; QK4_1 / 2], // Non-zero nibbles
        };
        block.dequantize(&mut out);
        // With d=0, all values should equal m (=3.0): v * 0.0 + 3.0 = 3.0
        for val in out {
            assert_eq!(
                val, 3.0,
                "Q4_1 zero scale should produce m for all elements"
            );
        }
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
