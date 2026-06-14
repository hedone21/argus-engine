//! Flash attention NEON helpers + F16/F32 dot product utilities (L2).
//!
//! Pulled out of `backend::cpu::neon::CpuBackendNeon` so that:
//! - `backend/opencl/plan.rs` can invoke `flash_partial_kv_range_f16` and
//!   `merge_two_partials_f32` for the GPU↔CPU hybrid attention split
//!   (`AttentionVariant::HybridKvSplit`) without crossing the L1↔L1
//!   cross-backend boundary that INV-LAYER-001 prohibits.
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

/// Compute a partial flash-attention result over the KV range
/// `[kv_start, kv_end)` for a single-token decode.
///
/// The output is **un-normalised** on purpose: the caller merges the CPU
/// partial with a complementary GPU partial via [`merge_two_partials_f32`].
/// Per Q-head the function writes:
///   * `partial_ml_out[h*2 + 0] = m_final`  (running max logit)
///   * `partial_ml_out[h*2 + 1] = l_final`  (running softmax denom)
///   * `partial_o_out[h*head_dim ..]      = Õ = Σ exp(s_t − m_final) · V[t]`
///
/// Empty ranges (`kv_start >= kv_end`) produce `m = −∞`, `l = 0`, `o = 0`,
/// which makes them a no-op contributor in the downstream merge.
///
/// GQA: `kv_h = q_h / (n_heads_q / n_heads_kv)`. Q-heads are parallelised
/// with Rayon across the head axis; each head is independent in this
/// formulation, so there is no cross-thread synchronisation.
///
/// # Preconditions
/// * HeadMajor KV layout: `K/V[kv_h, t, d]` with stride
///   `(capacity * head_dim, head_dim, 1)`.
/// * F16 K/V (passed as `*const u16` = f16 bits).
/// * `kv_end <= capacity`.
/// * `n_heads_q` is divisible by `n_heads_kv` (standard GQA).
///
/// # Safety
/// All input/output pointers must reference buffers with at least the shapes
/// documented above and remain valid for the duration of the call. The
/// output regions (`partial_ml_out`, `partial_o_out`) must not alias any
/// input. Aliasing between `partial_ml_out` and `partial_o_out` would be UB
/// but neither this function nor any realistic caller does so.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_partial_kv_range_f16(
    q_f32: *const f32,
    k_f16: *const u16,
    v_f16: *const u16,
    partial_ml_out: *mut f32,
    partial_o_out: *mut f32,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    capacity: usize,
    kv_start: usize,
    kv_end: usize,
    inv_sqrt_dk: f32,
) {
    use rayon::prelude::*;

    debug_assert!(n_heads_q.is_multiple_of(n_heads_kv));
    debug_assert!(kv_end <= capacity);
    debug_assert!(head_dim == 64 || head_dim == 128);

    let gqa_ratio = n_heads_q / n_heads_kv;

    // 빈 범위 처리: 모든 head를 (m=-inf, l=0, o=0)로 초기화 후 조기 반환.
    if kv_start >= kv_end {
        unsafe {
            for h in 0..n_heads_q {
                *partial_ml_out.add(h * 2) = f32::NEG_INFINITY;
                *partial_ml_out.add(h * 2 + 1) = 0.0;
                std::ptr::write_bytes(partial_o_out.add(h * head_dim), 0, head_dim);
            }
        }
        return;
    }

    // Rayon par_iter용 포인터 번들. rustc 2021의 disjoint closure capture가
    // 필드 단위로 raw pointer를 `&*const _` 형태로 캡처하는 것을 피하기 위해
    // `usize`로 감싸둔다. head 축 partition이므로 동시 쓰기 충돌 없음.
    #[derive(Clone, Copy)]
    struct HeadPtrs {
        q: usize,
        k: usize,
        v: usize,
        ml: usize,
        o: usize,
    }

    let ptrs = HeadPtrs {
        q: q_f32 as usize,
        k: k_f16 as usize,
        v: v_f16 as usize,
        ml: partial_ml_out as usize,
        o: partial_o_out as usize,
    };

    (0..n_heads_q).into_par_iter().for_each(move |q_h| {
        let kv_h = q_h / gqa_ratio;
        let q_off = q_h * head_dim;
        // SAFETY: q_h < n_heads_q 범위. head 별로 독립된 slice에만 쓴다.
        unsafe {
            let q_base = ptrs.q as *const f32;
            let k_base = ptrs.k as *const u16;
            let v_base = ptrs.v as *const u16;
            let ml_base = ptrs.ml as *mut f32;
            let o_all = ptrs.o as *mut f32;
            let q_ptr = q_base.add(q_off);
            let o_base = o_all.add(q_h * head_dim);
            // 누산 전 0으로 초기화 (online softmax는 FMA into-self 패턴).
            std::ptr::write_bytes(o_base, 0, head_dim);

            let mut m_run: f32 = f32::NEG_INFINITY;
            let mut l_run: f32 = 0.0;

            // `flash_chunk_worker`와 동일한 4-way unroll 구조.
            let range_len = kv_end - kv_start;
            let full_4 = range_len / 4;
            for block in 0..full_4 {
                let t_base = kv_start + block * 4;
                let mut b_ptrs = [std::ptr::null::<u16>(); 4];
                for (r, bp) in b_ptrs.iter_mut().enumerate() {
                    let t = t_base + r;
                    let off = (kv_h * capacity + t) * head_dim;
                    *bp = k_base.add(off);
                }
                let mut dots = [0.0f32; 4];
                vec_dot_f16_f32_4rows(head_dim, q_ptr, b_ptrs, &mut dots);
                for (r, &d) in dots.iter().enumerate() {
                    let t = t_base + r;
                    let s = d * inv_sqrt_dk;
                    flash_apply_token(
                        s, t, kv_h, capacity, head_dim, v_base, o_base, &mut m_run, &mut l_run,
                    );
                }
            }
            for t in (kv_start + full_4 * 4)..kv_end {
                let off = (kv_h * capacity + t) * head_dim;
                let k_ptr = k_base.add(off);
                let dot = vec_dot_f16_f32(head_dim, q_ptr, k_ptr);
                let s = dot * inv_sqrt_dk;
                flash_apply_token(
                    s, t, kv_h, capacity, head_dim, v_base, o_base, &mut m_run, &mut l_run,
                );
            }

            *ml_base.add(q_h * 2) = m_run;
            *ml_base.add(q_h * 2 + 1) = l_run;
        }
    });
}

/// Merge two partial `(m, l, Õ)` flash-attention results into a single
/// normalised output.
///
/// For each Q-head `h`, given partials `(m0, l0, o0_unnorm)` and
/// `(m1, l1, o1_unnorm)`:
///   * `M = max(m0, m1)`
///   * `α0 = exp(m0 − M)`, `α1 = exp(m1 − M)`
///   * `L = α0 · l0 + α1 · l1`
///   * `O = α0 · o0 + α1 · o1`
///   * `out[h] = O / L`
///
/// Edge cases (matching [`flash_partial_kv_range_f16`]):
///   * `m0 = −∞` AND `m1 = −∞` → the merge writes zeros for that head.
///     This matches "no valid tokens anywhere" semantics; callers that want
///     a uniform-V fallback must handle that at a higher level (analogous
///     to the existing `flash_uniform_fallback` in `flash_merge_partials`).
///   * Exactly one partial is `m = −∞` → the other partial is used as-is
///     (after normalisation by its own `l`).
///   * Non-finite or non-positive `L` → output is zero (same pathological
///     guard as `flash_merge_partials`).
///
/// # Safety
/// All pointers must reference buffers of the documented shape and must not
/// alias `out`. `out` must be writable for `n_heads_q * head_dim` f32 lanes.
pub unsafe fn merge_two_partials_f32(
    partial0_ml: *const f32,
    partial0_o: *const f32,
    partial1_ml: *const f32,
    partial1_o: *const f32,
    out: *mut f32,
    n_heads_q: usize,
    head_dim: usize,
) {
    unsafe {
        for h in 0..n_heads_q {
            let m0 = *partial0_ml.add(h * 2);
            let l0 = *partial0_ml.add(h * 2 + 1);
            let m1 = *partial1_ml.add(h * 2);
            let l1 = *partial1_ml.add(h * 2 + 1);

            let out_h = out.add(h * head_dim);

            // 두 partial 모두 dead → 제로 폴백.
            if !m0.is_finite() && !m1.is_finite() {
                std::ptr::write_bytes(out_h, 0, head_dim);
                continue;
            }

            // Global max. -inf는 자연스럽게 작은 쪽으로 처리됨.
            let m_max = if m0.is_finite() && m1.is_finite() {
                m0.max(m1)
            } else if m0.is_finite() {
                m0
            } else {
                m1
            };

            let alpha0 = if m0.is_finite() {
                (m0 - m_max).exp()
            } else {
                0.0
            };
            let alpha1 = if m1.is_finite() {
                (m1 - m_max).exp()
            } else {
                0.0
            };

            let l_total = alpha0 * l0 + alpha1 * l1;

            // 분모 불량 → 제로 폴백 (flash_merge_partials와 동일 semantics).
            if !l_total.is_finite() || l_total <= 0.0 {
                std::ptr::write_bytes(out_h, 0, head_dim);
                continue;
            }

            let inv_l = 1.0 / l_total;
            let scale0 = alpha0 * inv_l;
            let scale1 = alpha1 * inv_l;

            let o0_ptr = partial0_o.add(h * head_dim);
            let o1_ptr = partial1_o.add(h * head_dim);

            // NEON FMA 루프: out = scale0*o0 + scale1*o1.
            let s0_v = vdupq_n_f32(scale0);
            let s1_v = vdupq_n_f32(scale1);
            let mut i = 0;
            while i + 4 <= head_dim {
                let a = vld1q_f32(o0_ptr.add(i));
                let b = vld1q_f32(o1_ptr.add(i));
                let t0 = vmulq_f32(a, s0_v);
                let t1 = vfmaq_f32(t0, b, s1_v);
                vst1q_f32(out_h.add(i), t1);
                i += 4;
            }
            while i < head_dim {
                *out_h.add(i) = scale0 * *o0_ptr.add(i) + scale1 * *o1_ptr.add(i);
                i += 1;
            }
        }
    }
}
