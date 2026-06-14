//------------------------------------------------------------------------------
// cvt_q4_0_noshuffle_fused.cl  —  ENG-ALG-226 / INV-140
//
// Fused single-dispatch kernel that combines:
//   1) AOS block_q4_0 → noshuffle SOA (nibble rearrange, equivalent to
//      `kernel_convert_block_q4_0_noshuffle` in cvt.cl).
//   2) Row-major → column-major 2D transpose for both dst_q (ushort) and
//      dst_d (half), so the GEMV kernel can issue coalesced reads.
//
// This kernel replaces the legacy 4-step path:
//   GPU convert → host read → CPU transpose → host write × 2
// for Phase 6.5 weight-swap overhead reduction (Galaxy S25 measurement:
// soa_reconvert stage 758 ms / 25 layers / 7 tensors → ~100–150 ms).
//
// Layout contract (matches `OpenCLBackend::convert_q4_0_to_noshuffle` 4-step
// fallback after CPU transpose):
//
//   Source AOS Q4_0:
//     Each block of QK4_0=32 elements is `half d` + `uchar qs[16]`
//     (18 bytes total). num_blocks = ne01 * (ne00 / 32) blocks total.
//     Block layout in memory is row-major:
//       block(row, k) at index `row * blocks_per_row + k`,
//       blocks_per_row = ne00 / 32.
//
//   Destination dst_q (column-major ushort, total ne01 * cols_ushort ushorts):
//     cols_ushort = ne00 / 4   (each block = 16 bytes = 8 ushort).
//     For block (row, k), output position of the i-th ushort is:
//       col          = k * 8 + i,           i ∈ [0, 8)
//       q_t[col * ne01 + row]               (col-major)
//
//   Destination dst_d (column-major half, total ne01 * blocks_per_row halfs):
//     For block (row, k):
//       d_t[k * ne01 + row]
//
// Nibble rearrangement (same as cvt.cl noshuffle):
//   For i ∈ [0, 8):
//     uchar x0 = qs[2*i + 0];
//     uchar x1 = qs[2*i + 1];
//     out_q_lo = (x0 & 0x0F) | ((x1 & 0x0F) << 4);   → out byte i + 0
//     out_q_hi = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);   → out byte i + 8
//   The 16 output bytes are then re-grouped as 8 little-endian ushorts:
//     ushort idx u (u ∈ [0,8)) = (out_byte[2u + 1] << 8) | out_byte[2u + 0]
//
// Dispatch:
//   global_size = (ne01/2) * blocks_per_row, 1-D.
//   Each work-item handles a ROW PAIR (2 adjacent rows) at one k-block, so
//   the 8 q-output ushorts for the two rows can be packed and written as
//   8 uint stores (4-byte aligned). This avoids the Adreno race on
//   parallel ushort stores to adjacent addresses (verified on SM8750 /
//   Adreno 830: per-row work-item version had ~43% q_buf bytes diverge
//   from CPU reference; row-pair uint-write fixes it).
//   ne01 is required to be even for this kernel; the host falls back to
//   the 4-step legacy path otherwise.
//   No subgroup reductions, no SLM. Pure scalar per-thread fan-out write.
//------------------------------------------------------------------------------

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK4_0 32

typedef ushort uint16_t;
typedef uchar  uint8_t;

struct block_q4_0
{
    half  d;
    uchar qs[QK4_0 / 2];   // 16 bytes
};

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0_noshuffle_fused
//
// Args:
//   src        — global block_q4_0* (AOS, num_blocks blocks)
//   dst_q      — global ushort* (col-major, ne01 * cols_ushort ushorts)
//   dst_d      — global half*   (col-major, ne01 * blocks_per_row halfs)
//   ne01       — M (number of rows)
//   blocks_per_row — ne00 / 32 (Q4_0 blocks per row)
//
// Note: cols_ushort = blocks_per_row * 8.  We do not pass cols_ushort
// separately — it is derived implicitly from the (k, i) iteration.
//------------------------------------------------------------------------------
__kernel void kernel_convert_block_q4_0_noshuffle_fused(
    __global const struct block_q4_0 * src,
    __global ushort * dst_q,
    __global half   * dst_d,
    const uint ne01,
    const uint blocks_per_row)
{
    const uint gid = get_global_id(0);

    // Decompose linear pair index into (pair, k).
    //   gid = pair * blocks_per_row + k    where pair = row_pair_idx, 2*pair = even row
    const uint pair = gid / blocks_per_row;
    const uint k    = gid - pair * blocks_per_row;
    const uint half_ne01 = ne01 >> 1;

    // Bounds check (host clamps but be defensive).
    if (pair >= half_ne01) {
        return;
    }

    const uint row_a = pair * 2u;     // even row
    const uint row_b = row_a + 1u;    // odd row

    __global const struct block_q4_0 * b_a = src + row_a * blocks_per_row + k;
    __global const struct block_q4_0 * b_b = src + row_b * blocks_per_row + k;

    // ── d: pack two halves into a uint store at d_t[k*ne01 + row_a..row_a+1] ──
    // dst_d is `half *`; viewing it as `uint *` aligns each write at a row
    // pair boundary. Address: (k * ne01 + row_a) / 2 = k * half_ne01 + pair.
    __global uint * dst_d_uint = (__global uint *)dst_d;
    const uint d_packed = ((uint)as_ushort(b_a->d))
                        | ((uint)as_ushort(b_b->d) << 16);
    dst_d_uint[k * half_ne01 + pair] = d_packed;

    // ── q: nibble rearrange + col-major scatter (uint stores per row pair) ──
    //
    // For each output column u ∈ [0, 8) we produce one ushort for row_a and
    // one ushort for row_b, then pack them into a single uint and store at
    // address (k*8 + u) * (ne01/2) + pair. This guarantees 4-byte aligned
    // stores from each work-item and prevents the Adreno-specific race that
    // corrupts the high byte of every ushort when adjacent ushorts are
    // written by parallel work-items (verified ~43% q_buf divergence with
    // the per-row variant; uint-pair writes fixed it).
    //
    // Per-block unshuffle:
    //   lo(i) = (qs[2i] & 0x0F) | ((qs[2i+1] & 0x0F) << 4)        i ∈ [0, 8)
    //   hi(i) = ((qs[2i] & 0xF0) >> 4) | (qs[2i+1] & 0xF0)
    // ushort layout (per row, packed little-endian):
    //   u=0..3 → (lo(2u), lo(2u+1))
    //   u=4..7 → (hi(2(u-4)), hi(2(u-4)+1))
    __global uint * dst_q_uint = (__global uint *)dst_q;
    const uint base_col = k * 8u;

    // Helper macro: pack one ushort from two unshuffle outputs (lo/hi).
    // Each ushort is built from FOUR source bytes drawn from a single block.
    //
    // Args: BL = block pointer (b_a or b_b), MASK_HI = 0/1 selecting low
    // nibble (0) or high nibble (1) variant, IDX_BASE = 2*u or 2*(u-4).
    // Returns ushort value.
    #define PACK_USHORT(BL, MASK_HI, IDX_BASE) ({                           \
        const uchar _x0a = (BL)->qs[2*((IDX_BASE) + 0) + 0];                \
        const uchar _x1a = (BL)->qs[2*((IDX_BASE) + 0) + 1];                \
        const uchar _x0b = (BL)->qs[2*((IDX_BASE) + 1) + 0];                \
        const uchar _x1b = (BL)->qs[2*((IDX_BASE) + 1) + 1];                \
        uchar _lo, _hi;                                                     \
        if ((MASK_HI) == 0) {                                               \
            _lo = (uchar)((_x0a & 0x0F) | ((_x1a & 0x0F) << 4));            \
            _hi = (uchar)((_x0b & 0x0F) | ((_x1b & 0x0F) << 4));            \
        } else {                                                            \
            _lo = (uchar)(((_x0a & 0xF0) >> 4) | (_x1a & 0xF0));            \
            _hi = (uchar)(((_x0b & 0xF0) >> 4) | (_x1b & 0xF0));            \
        }                                                                   \
        ((ushort)_lo | ((ushort)_hi << 8));                                 \
    })

    // u in [0, 4): low-nibble half (lo bytes)
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const ushort v_a = PACK_USHORT(b_a, 0, 2*u);
        const ushort v_b = PACK_USHORT(b_b, 0, 2*u);
        const uint packed = (uint)v_a | ((uint)v_b << 16);
        dst_q_uint[(base_col + (uint)u) * half_ne01 + pair] = packed;
    }

    // u in [4, 8): high-nibble half (hi bytes)
    #pragma unroll
    for (int u = 4; u < 8; ++u) {
        const ushort v_a = PACK_USHORT(b_a, 1, 2*(u - 4));
        const ushort v_b = PACK_USHORT(b_b, 1, 2*(u - 4));
        const uint packed = (uint)v_a | ((uint)v_b << 16);
        dst_q_uint[(base_col + (uint)u) * half_ne01 + pair] = packed;
    }

    #undef PACK_USHORT
}
