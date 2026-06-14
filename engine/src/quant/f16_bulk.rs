//! Bulk F16 → F32 conversion (L2 utility).
//!
//! NEON `fcvtl`/`fcvtl2` SIMD impl on aarch64, scalar fallback elsewhere.
//! Pulled out of `backend::cpu::neon` so that L3 callers (transformer_layer,
//! KV dequant paths) can use it without crossing the L3→L1 boundary that
//! INV-LAYER-003 prohibits.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON bulk F16→F32 conversion using `fcvtl`/`fcvtl2`.
/// 16 elements per iteration (2× vld1q_u16 + 4× fcvtl/fcvtl2 + 4× vst1q_f32).
///
/// # Safety
/// `src` must point to at least `n` u16 (F16) values,
/// `dst` to at least `n` f32 values (write-only).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bulk_f16_to_f32(src: *const u16, dst: *mut f32, n: usize) {
    unsafe {
        let mut i = 0;
        while i + 16 <= n {
            let raw0: uint16x8_t = vld1q_u16(src.add(i));
            let raw1: uint16x8_t = vld1q_u16(src.add(i + 8));
            let f0: float32x4_t;
            let f1: float32x4_t;
            let f2: float32x4_t;
            let f3: float32x4_t;
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f0, i = in(vreg) raw0);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f1, i = in(vreg) raw0);
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f2, i = in(vreg) raw1);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f3, i = in(vreg) raw1);
            vst1q_f32(dst.add(i), f0);
            vst1q_f32(dst.add(i + 4), f1);
            vst1q_f32(dst.add(i + 8), f2);
            vst1q_f32(dst.add(i + 12), f3);
            i += 16;
        }
        while i + 8 <= n {
            let raw: uint16x8_t = vld1q_u16(src.add(i));
            let lo: float32x4_t;
            let hi: float32x4_t;
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) lo, i = in(vreg) raw);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) hi, i = in(vreg) raw);
            vst1q_f32(dst.add(i), lo);
            vst1q_f32(dst.add(i + 4), hi);
            i += 8;
        }
        while i < n {
            *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
            i += 1;
        }
    }
}

/// Scalar fallback for non-aarch64 hosts (cargo test on x86, etc.).
///
/// # Safety
/// `src`/`dst` must be valid for `n` elements.
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn bulk_f16_to_f32(src: *const u16, dst: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_basic() {
        let f32_in: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 1e-3, 1e3, -1e3];
        let f16_bits: Vec<u16> = f32_in
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let mut out = vec![0.0f32; f16_bits.len()];
        unsafe {
            bulk_f16_to_f32(f16_bits.as_ptr(), out.as_mut_ptr(), f16_bits.len());
        }
        for (i, (&a, &b)) in f32_in.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "idx {i}: {a} vs {b} (f16 precision loss expected ~1e-3)"
            );
        }
    }

    #[test]
    fn lengths_16_8_residual() {
        // Hits all three branches: ≥16 fast loop, 8-tail, <8 scalar.
        let lengths = [0, 1, 7, 8, 15, 16, 17, 23, 24, 31, 32, 33];
        for &n in &lengths {
            let f32_in: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
            let f16_bits: Vec<u16> = f32_in
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            let mut out = vec![0.0f32; n];
            unsafe {
                bulk_f16_to_f32(f16_bits.as_ptr(), out.as_mut_ptr(), n);
            }
            for (i, (&a, &b)) in f32_in.iter().zip(out.iter()).enumerate() {
                assert!((a - b).abs() < 1e-2, "n={n} idx={i}: {a} vs {b}");
            }
        }
    }
}
