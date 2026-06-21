//! Flash attention NEON helpers + F16/F32 dot product utilities (L2).
//!
//! Pulled out of `backend::cpu::neon::CpuBackendNeon` so that:
//! - `layers/transformer_layer/forward_gen.rs` can invoke `vec_dot_f16_f32`
//!   for KV-F16 attention score computation without the L3→L1 downcast
//!   that INV-LAYER-003 prohibits.
//!
//! Module is gated by `target_arch = "aarch64"` — all callers are themselves
//! `#[cfg(target_arch = "aarch64")]` gated. NEON intrinsics + inline asm
//! (`fcvtl`/`fcvtl2`) are the primary implementation; no scalar fallback is
//! provided because no non-aarch64 caller exists.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

/// Apply a single `(t, raw_s)` token to the running online-softmax state
/// `(m_run, l_run)` and FMA `V[t] * beta` into the accumulator `o_base`.
///
/// NaN/−inf scores are treated as weight-0: the running state is left
/// untouched and the accumulator is not modified. This mirrors the
/// per-token NaN gate of the head-parallel path.
///
/// # Safety
/// `v_ptr` must point to an F16 KV tensor of shape
/// `[_, num_heads_kv, capacity, head_dim]` with at least `t+1` valid
/// positions for `kv_h`. `o_base` must point to `head_dim` writable
/// F32 lanes.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_apply_token(
    raw_s: f32,
    t: usize,
    kv_h: usize,
    capacity: usize,
    head_dim: usize,
    v_ptr: *const u16,
    o_base: *mut f32,
    m_run: &mut f32,
    l_run: &mut f32,
) {
    unsafe {
        if !raw_s.is_finite() {
            return;
        }
        let m_new = m_run.max(raw_s);
        let alpha = if m_run.is_infinite() && m_run.is_sign_negative() {
            0.0
        } else {
            (*m_run - m_new).exp()
        };
        let beta = (raw_s - m_new).exp();
        *l_run = *l_run * alpha + beta;
        *m_run = m_new;

        let vp = v_ptr.add((kv_h * capacity + t) * head_dim);
        let alpha_v = vdupq_n_f32(alpha);
        let beta_v = vdupq_n_f32(beta);
        let mut i = 0;
        while i + 16 <= head_dim {
            let raw0: uint16x8_t = vld1q_u16(vp.add(i));
            let raw1: uint16x8_t = vld1q_u16(vp.add(i + 8));
            let f0: float32x4_t;
            let f1: float32x4_t;
            let f2: float32x4_t;
            let f3: float32x4_t;
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f0, i = in(vreg) raw0);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f1, i = in(vreg) raw0);
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f2, i = in(vreg) raw1);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f3, i = in(vreg) raw1);

            let o0 = vld1q_f32(o_base.add(i));
            let o1 = vld1q_f32(o_base.add(i + 4));
            let o2 = vld1q_f32(o_base.add(i + 8));
            let o3 = vld1q_f32(o_base.add(i + 12));
            let o0s = vmulq_f32(o0, alpha_v);
            let o1s = vmulq_f32(o1, alpha_v);
            let o2s = vmulq_f32(o2, alpha_v);
            let o3s = vmulq_f32(o3, alpha_v);
            vst1q_f32(o_base.add(i), vfmaq_f32(o0s, beta_v, f0));
            vst1q_f32(o_base.add(i + 4), vfmaq_f32(o1s, beta_v, f1));
            vst1q_f32(o_base.add(i + 8), vfmaq_f32(o2s, beta_v, f2));
            vst1q_f32(o_base.add(i + 12), vfmaq_f32(o3s, beta_v, f3));
            i += 16;
        }
        while i + 8 <= head_dim {
            let raw: uint16x8_t = vld1q_u16(vp.add(i));
            let fl: float32x4_t;
            let fh: float32x4_t;
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) fl, i = in(vreg) raw);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) fh, i = in(vreg) raw);
            let o0 = vld1q_f32(o_base.add(i));
            let o1 = vld1q_f32(o_base.add(i + 4));
            let o0s = vmulq_f32(o0, alpha_v);
            let o1s = vmulq_f32(o1, alpha_v);
            vst1q_f32(o_base.add(i), vfmaq_f32(o0s, beta_v, fl));
            vst1q_f32(o_base.add(i + 4), vfmaq_f32(o1s, beta_v, fh));
            i += 8;
        }
        while i < head_dim {
            let v_f32 = half::f16::from_bits(*vp.add(i)).to_f32();
            *o_base.add(i) = *o_base.add(i) * alpha + v_f32 * beta;
            i += 1;
        }
    }
}

/// NEON multi-row dot product: compute 4 dot products simultaneously.
/// A(F32)[K] · B0..B3(F16)[K] → 4 output scalars.
/// Uses 16-element stride with software prefetch for weight data.
/// 16 accumulators (4 rows × 4 each) for maximum ILP.
///
/// # Safety
/// `a_ptr` must point to at least `k` f32 values. Each `b_ptrs[r]` must
/// point to at least `k` u16 (F16) values.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_f16_f32_4rows(
    k: usize,
    a_ptr: *const f32,
    b_ptrs: [*const u16; 4],
    out: &mut [f32; 4],
) {
    unsafe {
        // 4 rows × 4 accumulators = 16 registers (leaves 16 for temps)
        let mut s00 = vdupq_n_f32(0.0);
        let mut s01 = vdupq_n_f32(0.0);
        let mut s02 = vdupq_n_f32(0.0);
        let mut s03 = vdupq_n_f32(0.0);
        let mut s10 = vdupq_n_f32(0.0);
        let mut s11 = vdupq_n_f32(0.0);
        let mut s12 = vdupq_n_f32(0.0);
        let mut s13 = vdupq_n_f32(0.0);
        let mut s20 = vdupq_n_f32(0.0);
        let mut s21 = vdupq_n_f32(0.0);
        let mut s22 = vdupq_n_f32(0.0);
        let mut s23 = vdupq_n_f32(0.0);
        let mut s30 = vdupq_n_f32(0.0);
        let mut s31 = vdupq_n_f32(0.0);
        let mut s32 = vdupq_n_f32(0.0);
        let mut s33 = vdupq_n_f32(0.0);

        let mut idx = 0;
        // Prefetch distance: 128 bytes ahead (~2 cache lines = 64 F16 values)
        const PF: usize = 64;

        // Main loop: 16 elements per iteration × 4 rows
        while idx + 16 <= k {
            // Software prefetch weight data for all 4 rows
            std::arch::asm!(
                "prfm pldl1strm, [{b0}]",
                "prfm pldl1strm, [{b1}]",
                "prfm pldl1strm, [{b2}]",
                "prfm pldl1strm, [{b3}]",
                b0 = in(reg) b_ptrs[0].add(idx + PF),
                b1 = in(reg) b_ptrs[1].add(idx + PF),
                b2 = in(reg) b_ptrs[2].add(idx + PF),
                b3 = in(reg) b_ptrs[3].add(idx + PF),
            );

            // Load 16 F32 activations (shared)
            let a0 = vld1q_f32(a_ptr.add(idx));
            let a1 = vld1q_f32(a_ptr.add(idx + 4));
            let a2 = vld1q_f32(a_ptr.add(idx + 8));
            let a3 = vld1q_f32(a_ptr.add(idx + 12));

            // Macro: load 16 F16 → 4×F32, FMA into 4 accumulators
            macro_rules! dot16_row {
            ($bp:expr, $sa:ident, $sb:ident, $sc:ident, $sd:ident) => {
                let bra: uint16x8_t = vld1q_u16($bp.add(idx));
                let brb: uint16x8_t = vld1q_u16($bp.add(idx + 8));
                let c0: float32x4_t;
                let c1: float32x4_t;
                let c2: float32x4_t;
                let c3: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) c0, i = in(vreg) bra);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) c1, i = in(vreg) bra);
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) c2, i = in(vreg) brb);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) c3, i = in(vreg) brb);
                $sa = vfmaq_f32($sa, a0, c0);
                $sb = vfmaq_f32($sb, a1, c1);
                $sc = vfmaq_f32($sc, a2, c2);
                $sd = vfmaq_f32($sd, a3, c3);
            };
        }
            dot16_row!(b_ptrs[0], s00, s01, s02, s03);
            dot16_row!(b_ptrs[1], s10, s11, s12, s13);
            dot16_row!(b_ptrs[2], s20, s21, s22, s23);
            dot16_row!(b_ptrs[3], s30, s31, s32, s33);

            idx += 16;
        }

        // 8-element tail
        while idx + 8 <= k {
            let a0 = vld1q_f32(a_ptr.add(idx));
            let a1 = vld1q_f32(a_ptr.add(idx + 4));

            macro_rules! dot8_row {
            ($bp:expr, $s0:ident, $s1:ident) => {
                let br: uint16x8_t = vld1q_u16($bp.add(idx));
                let bl: float32x4_t;
                let bh: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) bl, i = in(vreg) br);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) bh, i = in(vreg) br);
                $s0 = vfmaq_f32($s0, a0, bl);
                $s1 = vfmaq_f32($s1, a1, bh);
            };
        }
            dot8_row!(b_ptrs[0], s00, s01);
            dot8_row!(b_ptrs[1], s10, s11);
            dot8_row!(b_ptrs[2], s20, s21);
            dot8_row!(b_ptrs[3], s30, s31);
            idx += 8;
        }

        // Reduce 4 accumulators per row → scalar
        out[0] = vaddvq_f32(vaddq_f32(vaddq_f32(s00, s01), vaddq_f32(s02, s03)));
        out[1] = vaddvq_f32(vaddq_f32(vaddq_f32(s10, s11), vaddq_f32(s12, s13)));
        out[2] = vaddvq_f32(vaddq_f32(vaddq_f32(s20, s21), vaddq_f32(s22, s23)));
        out[3] = vaddvq_f32(vaddq_f32(vaddq_f32(s30, s31), vaddq_f32(s32, s33)));

        // Scalar tail
        while idx < k {
            let a_val = *a_ptr.add(idx);
            for r in 0..4 {
                out[r] += a_val * half::f16::from_bits(*b_ptrs[r].add(idx)).to_f32();
            }
            idx += 1;
        }
    }
}

/// Single-row NEON dot product: A(F32) · B(F16) → scalar F32.
/// Fused fcvtl + vfmaq_f32 — no intermediate F32 buffer needed.
///
/// # Safety
/// `a_ptr` must point to at least `k` f32 values, `b_ptr` to at least `k` u16 (F16) values.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_f16_f32(k: usize, a_ptr: *const f32, b_ptr: *const u16) -> f32 {
    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let mut k_idx = 0;

        // Main loop: 16 elements per iteration
        while k_idx + 16 <= k {
            let b_raw0: uint16x8_t = vld1q_u16(b_ptr.add(k_idx));
            let b_raw1: uint16x8_t = vld1q_u16(b_ptr.add(k_idx + 8));

            let b_f32_0: float32x4_t;
            let b_f32_1: float32x4_t;
            let b_f32_2: float32x4_t;
            let b_f32_3: float32x4_t;

            std::arch::asm!("fcvtl {out:v}.4s, {inp:v}.4h", out = lateout(vreg) b_f32_0, inp = in(vreg) b_raw0);
            std::arch::asm!("fcvtl2 {out:v}.4s, {inp:v}.8h", out = lateout(vreg) b_f32_1, inp = in(vreg) b_raw0);
            std::arch::asm!("fcvtl {out:v}.4s, {inp:v}.4h", out = lateout(vreg) b_f32_2, inp = in(vreg) b_raw1);
            std::arch::asm!("fcvtl2 {out:v}.4s, {inp:v}.8h", out = lateout(vreg) b_f32_3, inp = in(vreg) b_raw1);

            let a0 = vld1q_f32(a_ptr.add(k_idx));
            let a1 = vld1q_f32(a_ptr.add(k_idx + 4));
            let a2 = vld1q_f32(a_ptr.add(k_idx + 8));
            let a3 = vld1q_f32(a_ptr.add(k_idx + 12));

            sum0 = vfmaq_f32(sum0, a0, b_f32_0);
            sum1 = vfmaq_f32(sum1, a1, b_f32_1);
            sum2 = vfmaq_f32(sum2, a2, b_f32_2);
            sum3 = vfmaq_f32(sum3, a3, b_f32_3);

            k_idx += 16;
        }

        // 8-element tail
        while k_idx + 8 <= k {
            let b_raw: uint16x8_t = vld1q_u16(b_ptr.add(k_idx));
            let b_lo: float32x4_t;
            let b_hi: float32x4_t;
            std::arch::asm!("fcvtl {out:v}.4s, {inp:v}.4h", out = lateout(vreg) b_lo, inp = in(vreg) b_raw);
            std::arch::asm!("fcvtl2 {out:v}.4s, {inp:v}.8h", out = lateout(vreg) b_hi, inp = in(vreg) b_raw);

            let a0 = vld1q_f32(a_ptr.add(k_idx));
            let a1 = vld1q_f32(a_ptr.add(k_idx + 4));

            sum0 = vfmaq_f32(sum0, a0, b_lo);
            sum1 = vfmaq_f32(sum1, a1, b_hi);

            k_idx += 8;
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        let mut sum = vaddvq_f32(sum0);

        while k_idx < k {
            sum += *a_ptr.add(k_idx) * half::f16::from_bits(*b_ptr.add(k_idx)).to_f32();
            k_idx += 1;
        }

        sum
    }
}
