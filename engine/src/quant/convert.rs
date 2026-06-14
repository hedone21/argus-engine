//! Pure dtype conversion helpers (L2 quant).
//!
//! `f16_to_f32` 와 `quantize_q4_0` 는 weight loader / auf 양쪽에서 공유되는
//! pure byte→tensor 변환 helper다. `BlockQ4_0`/`QK4_0`/`half::f16` 외부 의존이
//! 없어서 L2 quant 모듈에 자연스럽게 속한다 (S-D1.1, 2026-05-24,
//! INV-LAYER-002 해소).
//!
//! 이전 위치: `models::loader::convert` (L3). 이 모듈에서 정의하고
//! `models::loader::convert` 에는 backward-compat 용 `pub use` re-export를 둔다.

use crate::quant::{BlockQ4_0, QK4_0};
use half::f16;

/// Convert F16 raw bytes to F32 values in a pre-allocated destination buffer.
///
/// # Safety
/// `src` must contain `num_elements * 2` bytes of valid F16 data.
/// `dst` must have at least `num_elements` entries.
pub fn f16_to_f32(src: &[u8], dst: &mut [f32], num_elements: usize) {
    let src_u16 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, num_elements) };
    for (i, &b) in src_u16.iter().enumerate() {
        dst[i] = f16::from_bits(b).to_f32();
    }
}

/// Quantize F32 data into Q4_0 blocks.
///
/// `rows` and `cols` define the 2D shape. `cols` must be a multiple of `QK4_0` (32).
/// Returns a Vec of `BlockQ4_0` with `rows * (cols / QK4_0)` entries.
pub fn quantize_q4_0(f32_data: &[f32], rows: usize, cols: usize) -> Vec<BlockQ4_0> {
    let nb_k = cols / QK4_0;
    let mut blocks = Vec::with_capacity(rows * nb_k);
    for j in 0..rows {
        for bi in 0..nb_k {
            let offset = j * cols + bi * QK4_0;
            let src = &f32_data[offset..offset + QK4_0];
            let mut block = BlockQ4_0 {
                d: f16::from_f32(0.0),
                qs: [0; 16],
            };
            let max_val = src.iter().map(|v| v.abs()).fold(0.0f32, |x, y| x.max(y));
            let d = max_val / 7.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            block.d = f16::from_f32(d);
            for z in 0..16 {
                let v0 = (src[z] * id).round().clamp(-8.0, 7.0) as i8;
                let v1 = (src[z + 16] * id).round().clamp(-8.0, 7.0) as i8;
                block.qs[z] = (v0 + 8) as u8 | (((v1 + 8) as u8) << 4);
            }
            blocks.push(block);
        }
    }
    blocks
}
