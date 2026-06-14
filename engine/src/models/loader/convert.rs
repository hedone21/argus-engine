//! Dtype conversion helpers for model weight loading.
//!
//! These functions convert raw safetensors byte slices into the target dtype
//! (F16, F32, Q4_0). Extracted from `transformer.rs::load_with_dtype` so they
//! can be reused by any `TensorSource` implementation.

use half::f16;

// S-D1.1 (2026-05-24, INV-LAYER-002 해소): `f16_to_f32` 와 `quantize_q4_0` 는
// L2 `quant::convert` 로 격상되었다. 본 모듈은 backward-compat 을 위한 re-export.
pub use crate::quant::convert::{f16_to_f32, quantize_q4_0};

/// Convert BF16 raw bytes to F16 values in a pre-allocated destination buffer.
///
/// # Safety
/// `src` must contain `num_elements * 2` bytes of valid BF16 data.
/// `dst` must have at least `num_elements` entries.
pub fn bf16_to_f16_buf(src: &[u8], dst: &mut [f16], num_elements: usize) {
    let src_u16 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, num_elements) };
    for (i, &b) in src_u16.iter().enumerate() {
        dst[i] = f16::from_f32(half::bf16::from_bits(b).to_f32());
    }
}

/// Convert F32 raw bytes to F16 values in a pre-allocated destination buffer.
///
/// # Safety
/// `src` must contain `num_elements * 4` bytes of valid F32 data.
/// `dst` must have at least `num_elements` entries.
pub fn f32_to_f16_buf(src: &[u8], dst: &mut [f16], num_elements: usize) {
    let src_f32 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, num_elements) };
    for (i, &v) in src_f32.iter().enumerate() {
        dst[i] = f16::from_f32(v);
    }
}

/// Convert BF16 raw bytes to F32 values in a pre-allocated destination buffer.
///
/// # Safety
/// `src` must contain `num_elements * 2` bytes of valid BF16 data.
/// `dst` must have at least `num_elements` entries.
pub fn bf16_to_f32(src: &[u8], dst: &mut [f32], num_elements: usize) {
    let src_u16 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, num_elements) };
    for (i, &b) in src_u16.iter().enumerate() {
        dst[i] = half::bf16::from_bits(b).to_f32();
    }
}

/// Release mmap pages after tensor data has been converted.
///
/// Calls `MADV_DONTNEED` on the page-aligned region so the kernel can reclaim
/// the source (e.g. BF16) pages that are no longer needed.
#[cfg(unix)]
pub fn release_source_pages(data: &[u8]) {
    const PAGE_SIZE: usize = 4096;
    let start = data.as_ptr() as usize;
    let end = start + data.len();
    let aligned_start = (start + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let aligned_end = end & !(PAGE_SIZE - 1);
    if aligned_end > aligned_start {
        unsafe {
            libc::madvise(
                aligned_start as *mut libc::c_void,
                aligned_end - aligned_start,
                libc::MADV_DONTNEED,
            );
        }
    }
}

/// Dequantize Q4_1 block data to F32.
///
/// Q4_1 block layout (20 bytes, 32 values):
///   d: f16 (2B) — scale
///   m: f16 (2B) — minimum
///   qs: [u8; 16] — 32 x 4-bit unsigned nibbles packed
///
/// Dequant: x[i] = nibble_i * d + m
pub fn dequant_q4_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK: usize = 32; // values per block
    const BLOCK_SIZE: usize = 20; // 2 + 2 + 16 bytes

    assert!(
        num_elements.is_multiple_of(QK),
        "Q4_1 dequant: num_elements ({}) must be a multiple of {}",
        num_elements,
        QK
    );
    let n_blocks = num_elements / QK;
    assert!(
        data.len() >= n_blocks * BLOCK_SIZE,
        "Q4_1 dequant: data too short ({} bytes for {} blocks)",
        data.len(),
        n_blocks
    );

    let mut out = vec![0.0f32; num_elements];

    for bi in 0..n_blocks {
        let block = &data[bi * BLOCK_SIZE..];
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let m = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let qs = &block[4..4 + 16];

        let base = bi * QK;
        for j in 0..16 {
            let byte = qs[j];
            let lo = (byte & 0x0F) as f32;
            let hi = (byte >> 4) as f32;
            out[base + j] = lo * d + m;
            out[base + j + 16] = hi * d + m;
        }
    }

    out
}

/// Dequantize Q4_K super-block data to F32.
///
/// Q4_K super-block layout (144 bytes, 256 values):
///   d: f16 (2B) — super-block scale
///   dmin: f16 (2B) — super-block minimum
///   scales: [u8; 12] — 8 sub-blocks x (6-bit scale + 6-bit min), packed
///   qs: [u8; 128] — 256 x 4-bit unsigned nibbles packed
///
/// Sub-block scale/min unpacking (from llama.cpp ggml-common.h):
///   For j in 0..3:
///     sc[j]   = scales[j] & 63
///     m[j]    = scales[j+4] & 63
///   For j in 4..7:
///     sc[j]   = (scales[j+4] & 0xF) << 2 | (scales[j-4] >> 6)
///     m[j]    = (scales[j+4] >> 4) << 2  | (scales[j-4+4] >> 6)
///
/// Dequant for sub-block j, value i:
///   q = 4-bit nibble from qs
///   x = d * sc[j] * q - dmin * m[j]
pub fn dequant_q4_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK: usize = 256; // values per super-block
    const BLOCK_SIZE: usize = 144; // 2 + 2 + 12 + 128

    assert!(
        num_elements.is_multiple_of(QK),
        "Q4_K dequant: num_elements ({}) must be a multiple of {}",
        num_elements,
        QK
    );
    let n_blocks = num_elements / QK;
    assert!(
        data.len() >= n_blocks * BLOCK_SIZE,
        "Q4_K dequant: data too short ({} bytes for {} blocks)",
        data.len(),
        n_blocks
    );

    let mut out = vec![0.0f32; num_elements];

    for bi in 0..n_blocks {
        let block = &data[bi * BLOCK_SIZE..];
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let scales_raw = &block[4..16]; // 12 bytes
        let qs = &block[16..144]; // 128 bytes

        // Unpack 8 sub-block scales and mins (6-bit each)
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];

        // Sub-blocks 0..3: straightforward 6-bit extraction
        for j in 0..4 {
            sc[j] = scales_raw[j] & 63;
            mn[j] = scales_raw[j + 4] & 63;
        }
        // Sub-blocks 4..7: combined from high bits of [0..3] and [8..11]
        for j in 4..8 {
            sc[j] = (scales_raw[j + 4] & 0xF) << 2 | (scales_raw[j - 4] >> 6);
            mn[j] = (scales_raw[j + 4] >> 4) << 2 | (scales_raw[j] >> 6);
        }

        let base = bi * QK;

        // Each sub-block has 32 values, 8 sub-blocks total
        for j in 0..8 {
            let d_sc = d * sc[j] as f32;
            let d_mn = dmin * mn[j] as f32;
            let qs_offset = j * 16; // 32 nibbles = 16 bytes

            for i in 0..16 {
                let byte = qs[qs_offset + i];
                let lo = (byte & 0x0F) as f32;
                let hi = (byte >> 4) as f32;
                out[base + j * 32 + i] = lo * d_sc - d_mn;
                out[base + j * 32 + i + 16] = hi * d_sc - d_mn;
            }
        }
    }

    out
}

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_to_f32_roundtrip() {
        // bf16(1.0) = 0x3F80
        let bf16_bytes: [u8; 4] = [0x80, 0x3F, 0x00, 0x40]; // 1.0, 2.0 in BF16
        let mut dst = [0.0f32; 2];
        bf16_to_f32(&bf16_bytes, &mut dst, 2);
        assert!((dst[0] - 1.0).abs() < 1e-2);
        assert!((dst[1] - 2.0).abs() < 1e-2);
    }

    #[test]
    fn test_f16_to_f32_roundtrip() {
        let val = f16::from_f32(3.14);
        let bytes = val.to_bits().to_le_bytes();
        let mut src = [0u8; 2];
        src.copy_from_slice(&bytes);
        let mut dst = [0.0f32; 1];
        f16_to_f32(&src, &mut dst, 1);
        assert!((dst[0] - 3.14).abs() < 0.01);
    }

    #[test]
    fn test_bf16_to_f16_roundtrip() {
        // bf16(1.0) = 0x3F80
        let bf16_bytes: [u8; 2] = [0x80, 0x3F];
        let mut dst = [f16::from_f32(0.0); 1];
        bf16_to_f16_buf(&bf16_bytes, &mut dst, 1);
        assert!((dst[0].to_f32() - 1.0).abs() < 1e-2);
    }

    #[test]
    #[allow(clippy::useless_vec)]
    fn test_f32_to_f16_roundtrip() {
        let val = 2.5f32;
        // Use Vec<f32> to guarantee 4-byte alignment for safe transmutation
        let src_f32 = vec![val];
        let bytes =
            unsafe { std::slice::from_raw_parts(src_f32.as_ptr() as *const u8, src_f32.len() * 4) };
        let mut dst = [f16::from_f32(0.0); 1];
        f32_to_f16_buf(bytes, &mut dst, 1);
        assert!((dst[0].to_f32() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_quantize_q4_0_basic() {
        // 1 row, 32 cols (1 block)
        let mut data = vec![0.0f32; 32];
        data[0] = 7.0;
        data[16] = -7.0;
        let blocks = quantize_q4_0(&data, 1, 32);
        assert_eq!(blocks.len(), 1);
        // d should be ~1.0 (max_abs=7.0, d=7.0/7.0=1.0)
        assert!((blocks[0].d.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dequant_q4_1_basic() {
        // Construct one Q4_1 block (20 bytes, 32 values)
        // d=1.0 (f16), m=0.5 (f16)
        // All nibbles = 3 => dequant = 3 * 1.0 + 0.5 = 3.5
        let mut block = [0u8; 20];
        let d_bits = f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bits = f16::from_f32(0.5).to_bits().to_le_bytes();
        block[0] = d_bits[0];
        block[1] = d_bits[1];
        block[2] = m_bits[0];
        block[3] = m_bits[1];
        // Fill qs with nibble pattern: lo=3, hi=3 => byte = 0x33
        for i in 0..16 {
            block[4 + i] = 0x33;
        }

        let result = dequant_q4_1(&block, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - 3.5).abs() < 0.01, "Expected ~3.5, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q4_1_asymmetric() {
        // d=2.0, m=1.0
        // lo nibbles = 0, hi nibbles = 15
        // lo values: 0 * 2.0 + 1.0 = 1.0
        // hi values: 15 * 2.0 + 1.0 = 31.0
        let mut block = [0u8; 20];
        let d_bits = f16::from_f32(2.0).to_bits().to_le_bytes();
        let m_bits = f16::from_f32(1.0).to_bits().to_le_bytes();
        block[0] = d_bits[0];
        block[1] = d_bits[1];
        block[2] = m_bits[0];
        block[3] = m_bits[1];
        // lo=0, hi=15 => byte = 0xF0
        for i in 0..16 {
            block[4 + i] = 0xF0;
        }

        let result = dequant_q4_1(&block, 32);
        // First 16 values (lo nibbles = 0): 0 * 2.0 + 1.0 = 1.0
        for i in 0..16 {
            assert!(
                (result[i] - 1.0).abs() < 0.01,
                "result[{}] = {}, expected 1.0",
                i,
                result[i]
            );
        }
        // Last 16 values (hi nibbles = 15): 15 * 2.0 + 1.0 = 31.0
        for i in 16..32 {
            assert!(
                (result[i] - 31.0).abs() < 0.01,
                "result[{}] = {}, expected 31.0",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_dequant_q4_k_basic() {
        // Construct one Q4_K super-block (144 bytes, 256 values)
        // d=1.0, dmin=0.0, all scales=1, all mins=0, all nibbles=5
        // => dequant = 1.0 * 1 * 5 - 0.0 * 0 = 5.0
        let mut block = [0u8; 144];
        let d_bits = f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = f16::from_f32(0.0).to_bits().to_le_bytes();
        block[0] = d_bits[0];
        block[1] = d_bits[1];
        block[2] = dmin_bits[0];
        block[3] = dmin_bits[1];

        // Set all sub-block scales to 1 (6-bit), mins to 0
        // For sub-blocks 0..3: scales_raw[j] = sc[j] (low 6 bits), rest 0
        // For sub-blocks 4..7: need scales_raw[j+4] low nibble for scale
        //   sc[4] = (scales_raw[8] & 0xF) << 2 | (scales_raw[0] >> 6)
        //   For sc=1: we need (scales_raw[8] & 0xF) << 2 | (scales_raw[0] >> 6) = 1
        //   Simplest: scales_raw[0..4] = 1 (sc[0..3]=1, no high bits)
        //   scales_raw[4..8] = 0 (mn[0..3]=0)
        //   For sc[4..7]=1: (scales_raw[j+4] & 0xF) << 2 | (scales_raw[j-4] >> 6) = 1
        //   Since scales_raw[0..4] have no high bits (value=1), >> 6 = 0
        //   So we need (scales_raw[8..12] & 0xF) << 2 = 1, but 1 is not divisible by 4.
        //   Use sc=4 instead: (x & 0xF) << 2 = 4 => x & 0xF = 1 => scales_raw[8..12] = 1
        // Let's use a simpler approach: all sc=1 for sub-blocks 0..3, sc=0 for 4..7
        // and verify only sub-blocks 0..3
        //
        // Even simpler: set d=2.0, all sc=1 for sub-blocks 0..3, nibbles=3
        // => result = 2.0 * 1 * 3 - 0 = 6.0 for first 128 values

        // Reset: just use straightforward approach with sub-blocks 0..3 only
        // scales_raw[0..4] = 1 (sc=1 for sub-blocks 0..3)
        // scales_raw[4..8] = 0 (mn=0 for sub-blocks 0..3)
        // scales_raw[8..12] = 0 (sc=0 and mn=0 for sub-blocks 4..7)
        for i in 0..4 {
            block[4 + i] = 1; // sc[0..3] = 1
        }
        // scales_raw[4..12] = 0 (already zeroed)

        // Fill all qs nibbles with 5: byte = 0x55
        for i in 0..128 {
            block[16 + i] = 0x55;
        }

        let result = dequant_q4_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-blocks 0..3 (first 128 values): d * 1 * 5 - 0 = 5.0
        for i in 0..128 {
            assert!(
                (result[i] - 5.0).abs() < 0.01,
                "result[{}] = {}, expected 5.0",
                i,
                result[i]
            );
        }
        // Sub-blocks 4..7 (last 128 values): d * 0 * 5 - 0 = 0.0
        for i in 128..256 {
            assert!(
                (result[i] - 0.0).abs() < 0.01,
                "result[{}] = {}, expected 0.0",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_dequant_q4_k_with_min() {
        // Test with nonzero dmin to verify subtraction
        let mut block = [0u8; 144];
        let d_bits = f16::from_f32(2.0).to_bits().to_le_bytes();
        let dmin_bits = f16::from_f32(1.0).to_bits().to_le_bytes();
        block[0] = d_bits[0];
        block[1] = d_bits[1];
        block[2] = dmin_bits[0];
        block[3] = dmin_bits[1];

        // Sub-block 0: sc=3, mn=2
        // scales_raw[0] = 3 (sc[0] = 3 & 63 = 3)
        // scales_raw[4] = 2 (mn[0] = 2 & 63 = 2)
        block[4] = 3;
        block[8] = 2;
        // All other sub-blocks: sc=0, mn=0

        // Nibbles for sub-block 0 (first 32 values): all 7 => byte = 0x77
        for i in 0..16 {
            block[16 + i] = 0x77;
        }
        // Rest zeroed

        let result = dequant_q4_k(&block, 256);

        // Sub-block 0: d * sc * q - dmin * mn = 2.0 * 3 * 7 - 1.0 * 2 = 42.0 - 2.0 = 40.0
        for i in 0..32 {
            assert!(
                (result[i] - 40.0).abs() < 0.1,
                "result[{}] = {}, expected 40.0",
                i,
                result[i]
            );
        }
    }
}
