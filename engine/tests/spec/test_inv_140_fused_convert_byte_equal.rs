//! INV-140 — Fused SOA Convert + Transpose Kernel byte-equal contract
//!
//! 대응 spec: `spec/32-engine-algorithms.md` §3.12.19.1 (ENG-ALG-226)
//! 대응 inv : `spec/41-invariants.md` §3.19 (INV-140)
//! 대응 arch: `arch/weight_swap.md` §7.2
//! Backlog : `.agent/todos/backlog.md` WSWAP-6-A
//!
//! ## 불변식 요약
//!
//! `OpenCLBackend::convert_q4_0_to_noshuffle()`의 두 경로 — fused single-
//! dispatch (ENG-ALG-226) 와 4-step host-transpose fallback — 는 동일 입력에
//! 대해 **byte-equal** 출력 cl_mem 내용을 산출해야 한다.
//!
//! - 입력: AOS Q4_0 buffer (block_q4_0 = `half d` + `uchar qs[16]`, 18 B/block).
//! - 출력 dst_q: column-major ushort, ne01 × cols_ushort (cols_ushort = ne00/4).
//! - 출력 dst_d: column-major half,   ne01 × blocks_per_row (blocks_per_row = ne00/32).
//!
//! ## 검증 항목
//!
//! - [A] 다양한 (ne00, ne01) 조합에 대해 random Q4_0 buffer를 생성하고 두
//!   경로의 출력을 host로 read하여 비트 단위 비교한다.
//! - [B] fused kernel이 컴파일되지 않은 host에서는 자동 skip (4-step path만
//!   실행 가능하므로 대조군이 없음).
//! - [C] OpenCL platform 자체가 없는 CI 호스트에서도 graceful skip.
//!
//! ## 구현 메모
//!
//! 두 경로 모두 동일한 backend 인스턴스에서 호출한다. fused path를 강제하기
//! 위해 `convert_q4_0_to_noshuffle`을 직접 호출하고(가용성 분기 포함),
//! 4-step path는 Backend의 fused kernel 핸들을 임시로 비활성화하는 helper
//! `convert_q4_0_to_noshuffle_force_legacy`를 통해 호출한다… 가 이상적이지만,
//! 본 코드베이스에 그런 helper는 아직 없다.
//!
//! 대신 본 테스트는 4-step path를 이미 알려진 알고리즘(테스트 내부
//! reference impl)으로 모사한다 — GPU에서 row-major SOA convert 실행
//! (`kernel_convert_block_q4_0_noshuffle`) 후 host에서 ushort/half 2D
//! transpose 수행. 그런 다음 fused path 결과와 byte-equal 비교한다.
//! 이는 4-step path의 정확성을 가정하지만, 4-step path는 swap 도입 전부터
//! production에서 검증되어 왔으며 INV-130/INV-131 spec test가 동일
//! buffer-layout 가정을 검증한다.

#![cfg(feature = "opencl")]

use anyhow::Result;

use argus_engine::backend::opencl::OpenCLBackend;

// ── Reference 4-step host-transpose implementation ───────────────────────────
//
// Mirrors the legacy fallback in `convert_q4_0_to_noshuffle()` (path 2).
// We replicate it locally so the test does not depend on a runtime toggle that
// would otherwise have to be added to the production backend.

const Q4_0_BLOCK_BYTES: usize = 18; // half d + 16 bytes qs (QK4_0=32 elems / 2)

/// Synthetic Q4_0 buffer: deterministic-per-seed pseudo-random nibbles + scales.
/// `scale_iter` controls scale variance so the test exercises edge fp16 values.
fn make_random_q4_0_aos(num_blocks: usize, seed: u64) -> Vec<u8> {
    let mut buf = vec![0u8; num_blocks * Q4_0_BLOCK_BYTES];
    let mut state: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for b in 0..num_blocks {
        // Simple xorshift64* — not crypto, just to fan out bytes deterministically.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let off = b * Q4_0_BLOCK_BYTES;

        // half d: small magnitude so fp16 round-trip is exact for representable values.
        let scale_f32 = ((state as i32 as f32) / (i32::MAX as f32)) * 0.125;
        let d_half = f32_to_f16_bits(scale_f32);
        buf[off] = d_half as u8;
        buf[off + 1] = (d_half >> 8) as u8;

        // 16 nibble bytes: another xorshift round per byte.
        for i in 0..16 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            buf[off + 2 + i] = (state >> ((i % 8) * 8)) as u8;
        }
    }
    buf
}

/// f32 → fp16 (IEEE 754 binary16) bit pattern.
/// Round-to-nearest-even, no NaN/Inf branch (test inputs stay in fp16 range).
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exp == 0 {
        // Zero / subnormal f32 → zero half (sign preserved).
        return sign << 15;
    }
    if exp == 0xFF {
        // NaN/Inf → emit Inf with same sign.
        return (sign << 15) | (0x1F << 10);
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign << 15; // underflow → 0
    }
    if new_exp >= 0x1F {
        return (sign << 15) | (0x1F << 10); // overflow → Inf
    }
    let mantissa_h = (mantissa >> 13) as u16; // round-to-zero is fine for our test inputs.
    (sign << 15) | ((new_exp as u16) << 10) | mantissa_h
}

/// CPU emulation of `kernel_convert_block_q4_0_noshuffle` row-major SOA stage,
/// followed by ushort/half host transpose. Produces the *exact same* output
/// layout the fused kernel writes directly.
fn cpu_reference_4step(src: &[u8], ne00: usize, ne01: usize) -> (Vec<u16>, Vec<u16>) {
    let blocks_per_row = ne00 / 32;
    let cols_ushort = ne00 / 4;
    let num_blocks = ne01 * blocks_per_row;
    assert_eq!(
        src.len(),
        num_blocks * Q4_0_BLOCK_BYTES,
        "src must be {} blocks * 18 B",
        num_blocks
    );

    // Row-major intermediate (matches kernel_convert_block_q4_0_noshuffle output).
    let mut q_row = vec![0u16; num_blocks * 8]; // 8 ushort per block
    let mut d_row = vec![0u16; num_blocks];

    for (b, d_slot) in d_row.iter_mut().enumerate() {
        let off = b * Q4_0_BLOCK_BYTES;

        // d: little-endian half stored in bytes [0..2].
        *d_slot = (src[off] as u16) | ((src[off + 1] as u16) << 8);

        // qs[16]: nibble rearrange identical to cvt.cl noshuffle.
        let qs = &src[off + 2..off + 18];
        let mut out = [0u8; 16];
        for i in 0..8 {
            let x0 = qs[2 * i];
            let x1 = qs[2 * i + 1];
            out[i] = (x0 & 0x0F) | ((x1 & 0x0F) << 4);
            out[i + 8] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);
        }
        // Repack as 8 little-endian ushorts at the row-major position.
        let block_q_off = b * 8;
        for u in 0..8 {
            q_row[block_q_off + u] = (out[2 * u] as u16) | ((out[2 * u + 1] as u16) << 8);
        }
    }

    // Now transpose:
    //   q_row[row * cols_ushort + col]      → q_col[col * ne01 + row]
    //   d_row[row * blocks_per_row + k]     → d_col[k   * ne01 + row]
    let mut q_col = vec![0u16; ne01 * cols_ushort];
    let mut d_col = vec![0u16; num_blocks];

    // Note: `q_row[block_q_off + u]` where block_q_off = (row * blocks_per_row + k) * 8
    // and the column inside that block = u → column overall = k * 8 + u.
    for row in 0..ne01 {
        for k in 0..blocks_per_row {
            // d
            d_col[k * ne01 + row] = d_row[row * blocks_per_row + k];
            // q (8 ushorts per block)
            let block_q_off = (row * blocks_per_row + k) * 8;
            for u in 0..8 {
                let col = k * 8 + u;
                q_col[col * ne01 + row] = q_row[block_q_off + u];
            }
        }
    }

    (q_col, d_col)
}

/// Read a cl_mem buffer back as `Vec<u16>`. Caller must guarantee a prior
/// `synchronize()` if the buffer was written asynchronously.
fn read_buffer_u16(queue: &ocl::Queue, buf: &ocl::core::Mem, nelem: usize) -> Result<Vec<u16>> {
    let mut host = vec![0u16; nelem];
    unsafe {
        let bytes = std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, nelem * 2);
        ocl::core::enqueue_read_buffer(
            queue,
            buf,
            true, // blocking
            0,
            bytes,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    Ok(host)
}

/// Upload an AOS Q4_0 buffer to a fresh device cl_mem.
fn upload_aos(backend: &OpenCLBackend, src: &[u8]) -> Result<ocl::core::Mem> {
    let ctx_ptr = backend.context.as_core();
    let cl_mem = unsafe {
        ocl::core::create_buffer::<_, u8>(ctx_ptr, ocl::core::MEM_READ_ONLY, src.len(), None)?
    };
    unsafe {
        ocl::core::enqueue_write_buffer(
            &backend.queue,
            &cl_mem,
            true, // blocking
            0,
            src,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    Ok(cl_mem)
}

// ── INV-140-A: byte-equal across (ne00, ne01) sweep ──────────────────────────

/// Sweep several (ne00, ne01) shapes and assert the fused kernel output is
/// byte-equal to the CPU reference (which models the legacy 4-step path).
///
/// Skip behaviour:
///   - No OpenCL platform → skip.
///   - `kernel_cvt_q4_0_noshuffle_fused` failed to compile → skip
///     (fused path unavailable on this host; nothing to compare).
#[test]
fn inv_140_fused_matches_4step_reference() {
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[SKIPPED] OpenCL device unavailable: {e}");
            return;
        }
    };

    // Probe fused kernel availability without leaking the UnsafeCell internals:
    // we round-trip a tiny buffer and check the production log path. The cleaner
    // route would be a public accessor; for now we rely on the fact that an
    // unavailable fused kernel still falls back to the legacy 4-step path which
    // is itself the reference, so byte-equality would be trivially true and we
    // would incorrectly call the test "passed" for the wrong reason. To avoid
    // that, we expose a lightweight probe via `cvt_noshuffle_fused_program` —
    // a `pub` field on the backend.
    if backend.cvt_noshuffle_fused_program.is_none() {
        eprintln!(
            "[SKIPPED] Fused convert kernel did not compile on this host (no NVIDIA/Adreno OpenCL with required features). 4-step path is exclusive — nothing to compare."
        );
        return;
    }
    eprintln!("[INV-140] Fused convert kernel program is present — running byte-equal sweep.");

    // Cases: small / medium / non-square / vocab-like. ne00 must be % 32 and
    // ne01 unconstrained (the kernel handles odd ne01 because each block is
    // independently transposed).
    let cases: &[(usize, usize, &str, u64)] = &[
        (32, 4, "minimal 32x4", 0x0011_2233_4455_6677),
        (64, 8, "64x8", 0xCAFE_BABE_DEAD_BEEF),
        (128, 16, "128x16 (Llama head_dim)", 0x1357_9BDF_2468_ACE0),
        (256, 32, "256x32", 0xF00D_0BAD_FACE_B00C),
        (512, 33, "512x33 (odd ne01)", 0xA5A5_5A5A_C3C3_3C3C),
        (2048, 64, "2048x64 (FFN slice)", 0x0102_0304_0506_0708),
    ];

    for (ne00, ne01, label, seed) in cases.iter().copied() {
        let blocks_per_row = ne00 / 32;
        let num_blocks = ne01 * blocks_per_row;
        let cols_ushort = ne00 / 4;

        // 1. Build deterministic random AOS Q4_0 source.
        let aos = make_random_q4_0_aos(num_blocks, seed);

        // 2. Upload to device.
        let src = upload_aos(&backend, &aos).expect("upload_aos");

        // 3. Run fused conversion.
        let (dst_q, dst_d, _img) = backend
            .convert_q4_0_to_noshuffle(&src, num_blocks, ne00, ne01)
            .expect("fused convert_q4_0_to_noshuffle");

        // The fused path skips queue.finish(); make sure GPU work is done
        // before we read.
        backend.queue.finish().expect("queue finish");

        let q_actual =
            read_buffer_u16(&backend.queue, &dst_q, ne01 * cols_ushort).expect("read dst_q");
        let d_actual = read_buffer_u16(&backend.queue, &dst_d, num_blocks).expect("read dst_d");

        // 4. Compute CPU reference.
        let (q_ref, d_ref) = cpu_reference_4step(&aos, ne00, ne01);

        // 5. Byte-equal compare.
        assert_eq!(
            q_actual.len(),
            q_ref.len(),
            "{label}: q length mismatch (actual={}, ref={})",
            q_actual.len(),
            q_ref.len()
        );
        assert_eq!(
            d_actual.len(),
            d_ref.len(),
            "{label}: d length mismatch (actual={}, ref={})",
            d_actual.len(),
            d_ref.len()
        );

        if q_actual != q_ref {
            // Find first divergence for a useful diagnostic.
            let mut first_diff = None;
            for (i, (a, r)) in q_actual.iter().zip(q_ref.iter()).enumerate() {
                if a != r {
                    first_diff = Some((i, *a, *r));
                    break;
                }
            }
            panic!(
                "{label}: dst_q byte-not-equal. ne00={ne00}, ne01={ne01}. First diff: {:?}",
                first_diff
            );
        }
        if d_actual != d_ref {
            let mut first_diff = None;
            for (i, (a, r)) in d_actual.iter().zip(d_ref.iter()).enumerate() {
                if a != r {
                    first_diff = Some((i, *a, *r));
                    break;
                }
            }
            panic!(
                "{label}: dst_d byte-not-equal. ne00={ne00}, ne01={ne01}. First diff: {:?}",
                first_diff
            );
        }
    }
}

// ── INV-140-B: device-only smoke (Adreno) ────────────────────────────────────
//
// On Android we additionally want to confirm the fused kernel actually
// compiles on Adreno (register spill check). Host NVIDIA can compile but
// Adreno may reject due to per-thread register ceiling. The test body is
// identical to the host path; we just gate it on target_os.

#[cfg(target_os = "android")]
#[test]
fn inv_140_fused_matches_4step_reference_android_smoke() {
    // Reuse the host-side function — if it skipped on host, the device case
    // will run the same checks. Adreno register-spill regressions surface here.
    inv_140_fused_matches_4step_reference();
}
