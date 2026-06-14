//! AOS → Adreno noshuffle SOA Q4_0 변환 (build-time pipeline).
//!
//! 빌드 타임에 적용되어, 런타임 GPU 측 `OpenCLBackend::convert_q4_0_to_noshuffle`
//! 을 우회한다. backend의 3단계 파이프라인:
//!   1. nibble bit unshuffle (`kernel_convert_block_q4_0_noshuffle`, cvt.cl)
//!   2. ushort-level 2D transpose of q nibbles
//!   3. half-level   2D transpose of d scales
//!
//! 와 동등한 byte-identical 결과를 반환하는 순수 CPU 구현이다.
//!
//! 입력: GGUF 표준 Q4_0 AOS bytes (block당 18B = 2B scale + 16B packed nibbles).
//! 출력: `(q_buf, d_buf)` — `cl_mem`에 직접 업로드 가능한 transposed SOA.
//!
//! Spec: ENG-DAT-096, Phase 4 LATENCY-AUF.

/// QK4_0 양자화 그룹 크기 (32개 element가 하나의 block).
pub const QK4_0: usize = 32;

/// AOS Q4_0 weight bytes를 Adreno noshuffle SOA로 완전 변환한다.
///
/// `ne00` (cols, K dim) 과 `ne01` (rows, M dim) 은 logical weight matrix의
/// 차원이며, GGUF Q/K permute는 호출 전에 적용되어 있어야 한다.
///
/// # Panics
/// - `ne00`이 32의 배수가 아닐 때.
/// - `blocks.len()`이 `ne01 * ne00 / 32 * 18`과 다를 때 (debug build 한정).
pub fn q4_0_aos_to_adreno_soa(blocks: &[u8], ne00: usize, ne01: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(ne00 % QK4_0, 0, "ne00 must be a multiple of QK4_0 (=32)");
    let blocks_per_row = ne00 / QK4_0;
    let cols_ushort = ne00 / 4; // ushort count per row (q, pre-transpose)
    let n_blocks_total = ne01 * blocks_per_row;
    debug_assert_eq!(blocks.len(), n_blocks_total * 18);

    // Step 1: unshuffle nibbles per block.
    let (q_unshuf, d_unshuf) = q4_0_aos_to_soa_unshuffled(blocks);

    // Step 2: ushort-level 2D transpose of q.
    //   src layout (row-major): q[row * cols_ushort + col] (ushort)
    //   dst layout (column-major view of dst): q_t[col * ne01 + row]
    debug_assert_eq!(q_unshuf.len(), n_blocks_total * 16);
    let q_total_ushort = ne01 * cols_ushort;
    let mut q_t = vec![0u16; q_total_ushort];
    {
        // SAFETY: q_unshuf is exactly q_total_ushort * 2 bytes; reinterpreting
        // little-endian byte pairs as u16 mirrors the runtime path which calls
        // `enqueue_read_buffer` directly into a `Vec<u16>`.
        let q_src =
            unsafe { std::slice::from_raw_parts(q_unshuf.as_ptr() as *const u16, q_total_ushort) };
        for row in 0..ne01 {
            for col in 0..cols_ushort {
                q_t[col * ne01 + row] = q_src[row * cols_ushort + col];
            }
        }
    }
    let mut q_out = vec![0u8; q_total_ushort * 2];
    for (i, &val) in q_t.iter().enumerate() {
        q_out[i * 2..i * 2 + 2].copy_from_slice(&val.to_le_bytes());
    }

    // Step 3: half-level 2D transpose of d.
    //   src layout (row-major): d[row * blocks_per_row + k] (half)
    //   dst layout (column-major view of dst): d_t[k * ne01 + row]
    debug_assert_eq!(d_unshuf.len(), n_blocks_total * 2);
    let mut d_t = vec![0u16; n_blocks_total];
    {
        // SAFETY: d_unshuf is exactly n_blocks_total * 2 bytes (one f16 per block).
        let d_src =
            unsafe { std::slice::from_raw_parts(d_unshuf.as_ptr() as *const u16, n_blocks_total) };
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                d_t[k * ne01 + row] = d_src[row * blocks_per_row + k];
            }
        }
    }
    let mut d_out = vec![0u8; n_blocks_total * 2];
    for (i, &val) in d_t.iter().enumerate() {
        d_out[i * 2..i * 2 + 2].copy_from_slice(&val.to_le_bytes());
    }

    (q_out, d_out)
}

/// Step 1만 수행: AOS 18B 블록 → row-major (q_unshuffled, d) 분리.
///
/// `kernel_convert_block_q4_0_noshuffle` (cvt.cl) 의 nibble bit-rearrange를
/// 동등하게 수행한다:
///   for i in 0..QK4_0/4 {
///     x0 = qs[2*i+0]; x1 = qs[2*i+1];
///     q[i + 0]       = (x0 & 0x0F) | ((x1 & 0x0F) << 4);  // even-positioned
///     q[i + QK4_0/4] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);  // odd-positioned
///   }
///
/// 결과는 row-major 즉 transpose 적용 전 layout. transpose 미적용 변환만
/// 필요한 호출자(예: 단위 테스트 스텁)에 노출된다. Builder 본 경로는
/// `q4_0_aos_to_adreno_soa`만 사용하라.
pub fn q4_0_aos_to_soa_unshuffled(blocks: &[u8]) -> (Vec<u8>, Vec<u8>) {
    const BLOCK_SIZE: usize = 18; // 2 (f16 scale) + 16 (nibbles)
    const QK_HALF: usize = QK4_0 / 2; // 16
    const QK_QUARTER: usize = QK4_0 / 4; // 8
    let n_blocks = blocks.len() / BLOCK_SIZE;
    let mut q_buf = vec![0u8; n_blocks * QK_HALF];
    let mut d_buf = vec![0u8; n_blocks * 2];
    for b in 0..n_blocks {
        let off = b * BLOCK_SIZE;
        d_buf[b * 2..(b + 1) * 2].copy_from_slice(&blocks[off..off + 2]);
        let qs = &blocks[off + 2..off + BLOCK_SIZE];
        let q_dst = &mut q_buf[b * QK_HALF..(b + 1) * QK_HALF];
        for i in 0..QK_QUARTER {
            let x0 = qs[2 * i];
            let x1 = qs[2 * i + 1];
            q_dst[i] = (x0 & 0x0F) | ((x1 & 0x0F) << 4);
            q_dst[i + QK_QUARTER] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);
        }
    }
    (q_buf, d_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 단일 블록: nibble unshuffle 정합성 — 핵심 변환 로직 검증.
    #[test]
    fn unshuffle_single_block_matches_cvt_kernel() {
        // 1개 block: scale=0x1234 (LE), qs = [0x10, 0x32, 0x54, 0x76, ..., 0xFE]
        // → 위치 i=0: x0=0x10, x1=0x32 → q[0] = 0x20 ((0x0&...)|(2<<4))
        //                                 q[8] = 0x31 (1>>4 | 0x30)
        let mut block = vec![0x34u8, 0x12u8]; // scale (little-endian f16)
        for i in 0..16u8 {
            block.push(i * 2); // 0x00, 0x02, 0x04, ..., 0x1E
        }
        assert_eq!(block.len(), 18);

        let (q, d) = q4_0_aos_to_soa_unshuffled(&block);
        assert_eq!(d, vec![0x34, 0x12]);
        // i=0: x0=0, x1=2 → q[0]=(0&0xF)|((2&0xF)<<4)=0x20
        //                  q[8]=((0&0xF0)>>4)|(2&0xF0)=0
        assert_eq!(q[0], 0x20);
        assert_eq!(q[8], 0x00);
        // i=1: x0=4, x1=6 → q[1]=(4&0xF)|((6&0xF)<<4)=0x64
        //                  q[9]=((4&0xF0)>>4)|(6&0xF0)=0
        assert_eq!(q[1], 0x64);
        assert_eq!(q[9], 0x00);
    }

    /// 2x32 (ne01=2, ne00=32) — nrows=2, cols=32, n_blocks_per_row=1.
    /// transpose 검증: q_t는 (cols_ushort=8, rows=2) layout.
    #[test]
    fn adreno_soa_transposes_two_rows_one_block() {
        let mut blocks = Vec::new();
        // row 0: scale=0x0001, qs increasing
        blocks.extend_from_slice(&[0x01, 0x00]);
        for i in 0..16u8 {
            blocks.push(i);
        }
        // row 1: scale=0x0002, qs decreasing
        blocks.extend_from_slice(&[0x02, 0x00]);
        for i in (0..16u8).rev() {
            blocks.push(i);
        }
        assert_eq!(blocks.len(), 36);

        let (q_out, d_out) = q4_0_aos_to_adreno_soa(&blocks, 32, 2);
        // q_total_ushort = 2 * 8 = 16, q_out = 32 bytes.
        assert_eq!(q_out.len(), 32);
        // d_total = 2 blocks * 2B = 4 bytes. transpose: blocks_per_row=1, M=2
        //   d_t[0*2+0]=row0 scale=0x0001, d_t[0*2+1]=row1 scale=0x0002.
        assert_eq!(d_out.len(), 4);
        assert_eq!(&d_out[0..2], &[0x01, 0x00]);
        assert_eq!(&d_out[2..4], &[0x02, 0x00]);

        // q transpose check: read u16s.
        let q_u16: Vec<u16> = q_out
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(q_u16.len(), 16);
        // First column (col=0): q_t[0..2] = (row0_q[0]_ushort, row1_q[0]_ushort).
        // Run unshuffle on row 0 manually to derive row0_q[0]:
        let (row0_q_unshuf, _) = q4_0_aos_to_soa_unshuffled(&blocks[0..18]);
        let row0_q_ushort = u16::from_le_bytes([row0_q_unshuf[0], row0_q_unshuf[1]]);
        let (row1_q_unshuf, _) = q4_0_aos_to_soa_unshuffled(&blocks[18..36]);
        let row1_q_ushort = u16::from_le_bytes([row1_q_unshuf[0], row1_q_unshuf[1]]);
        assert_eq!(q_u16[0], row0_q_ushort);
        assert_eq!(q_u16[1], row1_q_ushort);
    }

    /// Round-trip: builder 결과는 backend의 `convert_q4_0_to_noshuffle`이
    /// 만드는 transposed SOA와 완전히 동일한 byte sequence를 가진다는 invariant.
    /// 여기서는 reference algorithm을 직접 구현하여 비교한다.
    #[test]
    fn adreno_soa_matches_reference_pipeline() {
        // 4 rows × 64 cols = 4 × 2 blocks/row = 8 blocks total.
        let ne00 = 64usize;
        let ne01 = 4usize;
        let n_blocks = ne01 * ne00 / QK4_0;
        let mut blocks = Vec::with_capacity(n_blocks * 18);
        for b in 0..n_blocks {
            // scale: 0x0010 + b
            let s = (0x0010u16 + b as u16).to_le_bytes();
            blocks.extend_from_slice(&s);
            // qs: pseudo-random byte pattern, deterministic per block.
            for i in 0..16u8 {
                blocks.push(((b as u8) * 16).wrapping_add(i));
            }
        }
        assert_eq!(blocks.len(), n_blocks * 18);

        let (q_actual, d_actual) = q4_0_aos_to_adreno_soa(&blocks, ne00, ne01);

        // ── Reference pipeline ──────────────────────────────────────────────
        let blocks_per_row = ne00 / QK4_0;
        let cols_ushort = ne00 / 4;

        // Step 1: per-block nibble unshuffle (already covered by separate test).
        let (q_unshuf_ref, d_unshuf_ref) = q4_0_aos_to_soa_unshuffled(&blocks);

        // Step 2 reference: independent transpose loop.
        let q_total_ushort = ne01 * cols_ushort;
        let mut q_t_ref = vec![0u16; q_total_ushort];
        let q_src_ref = unsafe {
            std::slice::from_raw_parts(q_unshuf_ref.as_ptr() as *const u16, q_total_ushort)
        };
        for r in 0..ne01 {
            for c in 0..cols_ushort {
                q_t_ref[c * ne01 + r] = q_src_ref[r * cols_ushort + c];
            }
        }
        let mut q_ref_bytes = vec![0u8; q_total_ushort * 2];
        for (i, &v) in q_t_ref.iter().enumerate() {
            q_ref_bytes[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }

        // Step 3 reference: half transpose.
        let mut d_t_ref = vec![0u16; n_blocks];
        let d_src_ref =
            unsafe { std::slice::from_raw_parts(d_unshuf_ref.as_ptr() as *const u16, n_blocks) };
        for r in 0..ne01 {
            for k in 0..blocks_per_row {
                d_t_ref[k * ne01 + r] = d_src_ref[r * blocks_per_row + k];
            }
        }
        let mut d_ref_bytes = vec![0u8; n_blocks * 2];
        for (i, &v) in d_t_ref.iter().enumerate() {
            d_ref_bytes[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }

        assert_eq!(q_actual, q_ref_bytes, "q transpose must match reference");
        assert_eq!(d_actual, d_ref_bytes, "d transpose must match reference");
    }

    /// Size invariant: 변환 결과 (q_buf + d_buf)의 총 byte 수는 입력 AOS
    /// payload (블록당 18B)와 동일해야 한다 — `materialise_auf_soa_weight`의
    /// placeholder cl_mem 크기 가정과 일치.
    #[test]
    fn adreno_soa_total_bytes_invariant() {
        let ne00 = 128usize;
        let ne01 = 16usize;
        let n_blocks = ne01 * ne00 / QK4_0;
        let blocks = vec![0xA5u8; n_blocks * 18];

        let (q, d) = q4_0_aos_to_adreno_soa(&blocks, ne00, ne01);
        assert_eq!(q.len() + d.len(), n_blocks * 18);
        assert_eq!(q.len(), n_blocks * 16);
        assert_eq!(d.len(), n_blocks * 2);
    }
}
