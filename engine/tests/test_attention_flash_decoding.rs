//! Tests for the NEON flash-decoding (`KV-split`) attention path.
//!
//! Step 2 of the long-context optimization plan introduces a flash-decoding
//! style KV-split inside `attention_gen_f16_neon` to reduce GQA redundant KV
//! reads at long contexts.  The public signature of `Backend::attention_gen`
//! is unchanged; dispatch between the Step 1 head-parallel path and the
//! Step 2 flash path happens internally based on `cache_seq_len` and on
//! `scores_out.is_some()`.
//!
//! Test matrix covers:
//!   * GQA ratios 1, 2, 4, 6 (Qwen / Llama configurations).
//!   * Short sequences (< THRESHOLD)     → Step 1 head-parallel path.
//!   * Chunk-boundary / non-multiple lengths (1023/1024/1025/2047/2048/...).
//!   * Long sequences up to 4096 with multiple chunks.
//!   * NaN injection:  single token, full chunk, every token.
//!   * `scores_out = Some(...)` forces the head-parallel path irrespective
//!     of `cache_seq_len` (see arch/cpu_flash_decoding.md §7.2).
//!
//! The reference used for correctness is an inline F32 3-pass attention
//! (identical to the reference in `test_attention_online_softmax.rs`) fed
//! with the same F16-quantized KV values the NEON path sees.  Permitted
//! error: `NMSE < 1e-4`.
//!
//! Compiled only on aarch64 because `CpuBackendNeon` is gated on
//! `target_arch = "aarch64"`.  On non-aarch64 hosts the binary still
//! compiles but exposes no test functions.

#![cfg(target_arch = "aarch64")]

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::neon::CpuBackendNeon;
use argus_engine::buffer::DType;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random F32 values in [-1, 1).
fn gen_data(n: usize, seed: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        data.push(((s >> 16) as i16 as f32) / 32768.0);
    }
    data
}

fn make_f32_tensor(backend: &Arc<dyn Backend>, shape: Vec<usize>, data: &[f32]) -> Tensor {
    let memory = Galloc::new();
    let n = data.len();
    let buf = memory.alloc(n * 4, DType::F32).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.as_mut_ptr(), n * 4);
    }
    Tensor::new(Shape::new(shape), buf, backend.clone())
}

fn make_f32_zeros(backend: &Arc<dyn Backend>, shape: Vec<usize>) -> Tensor {
    let n: usize = shape.iter().product();
    let memory = Galloc::new();
    let buf = memory.alloc(n * 4, DType::F32).unwrap();
    unsafe {
        std::ptr::write_bytes(buf.as_mut_ptr(), 0, n * 4);
    }
    Tensor::new(Shape::new(shape), buf, backend.clone())
}

fn make_f16_tensor_from_f32(backend: &Arc<dyn Backend>, shape: Vec<usize>, data: &[f32]) -> Tensor {
    let n = data.len();
    let f16: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
    let memory = Galloc::new();
    let buf = memory.alloc(n * 2, DType::F16).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(f16.as_ptr() as *const u8, buf.as_mut_ptr(), n * 2);
    }
    Tensor::new(Shape::new(shape), buf, backend.clone())
}

/// Reference attention (3-pass, F32 throughout, HeadMajor layout).
///
/// Input layouts (F32):
///   q:  [n_heads_q, head_dim]
///   k:  [n_heads_kv, capacity, head_dim]    (first `seq_len` positions used)
///   v:  [n_heads_kv, capacity, head_dim]
fn reference_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    capacity: usize,
    seq_len: usize,
) -> Vec<f32> {
    let gqa = n_heads_q / n_heads_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_heads_q * head_dim];
    for h in 0..n_heads_q {
        let kv_h = h / gqa;
        let q_off = h * head_dim;
        let q_vec = &q[q_off..q_off + head_dim];

        // Pass 1: raw scores, with NaN -> -inf sanitization.
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let off = (kv_h * capacity + t) * head_dim;
            let k_vec = &k[off..off + head_dim];
            let dot: f32 = q_vec
                .iter()
                .zip(k_vec.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
            let s = dot * scale;
            scores[t] = if s.is_finite() { s } else { f32::NEG_INFINITY };
        }

        // Pass 2: softmax (with uniform fallback if everything is -inf).
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if m == f32::NEG_INFINITY {
            let u = 1.0 / seq_len as f32;
            for s in scores.iter_mut() {
                *s = u;
            }
        } else {
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            if sum <= 0.0 || !sum.is_finite() {
                let u = 1.0 / seq_len as f32;
                for s in scores.iter_mut() {
                    *s = u;
                }
            } else {
                let inv = 1.0 / sum;
                for s in scores.iter_mut() {
                    *s *= inv;
                }
            }
        }

        // Pass 3: weighted V sum.
        let o = &mut out[q_off..q_off + head_dim];
        for t in 0..seq_len {
            let w = scores[t];
            if w == 0.0 {
                continue;
            }
            let off = (kv_h * capacity + t) * head_dim;
            let v_vec = &v[off..off + head_dim];
            for d in 0..head_dim {
                o[d] += w * v_vec[d];
            }
        }
    }
    out
}

fn nmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (*x as f64) - (*y as f64);
        num += diff * diff;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        num as f32
    } else {
        (num / den) as f32
    }
}

/// Shared harness: build F16 KV tensors from F32 payloads, run NEON attention,
/// and return `(neon_output, reference_output)` for inspection.
fn run_case_raw(
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    capacity: usize,
    seed: u32,
    k_mutator: impl FnOnce(&mut [f32]),
) -> (Vec<f32>, Vec<f32>) {
    assert!(capacity >= seq_len);
    let backend: Arc<dyn Backend> = Arc::new(CpuBackendNeon::new());

    let q_data = gen_data(n_heads_q * head_dim, seed);
    let mut k_data = gen_data(n_heads_kv * capacity * head_dim, seed.wrapping_add(1));
    let v_data = gen_data(n_heads_kv * capacity * head_dim, seed.wrapping_add(2));
    k_mutator(&mut k_data);

    // Reference consumes the *quantized* F32→F16→F32 values so that the
    // comparison isolates algorithmic error, not F16 round-off.
    let k_f16_as_f32: Vec<f32> = k_data
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    let v_f16_as_f32: Vec<f32> = v_data
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();

    let q = make_f32_tensor(&backend, vec![n_heads_q, head_dim], &q_data);
    let k = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &k_data);
    let v = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &v_data);
    let mut out = make_f32_zeros(&backend, vec![n_heads_q, head_dim]);

    backend
        .attention_gen(
            &q, &k, &v, &mut out, n_heads_q, n_heads_kv, head_dim, seq_len, None,
        )
        .unwrap();

    let reference = reference_attention_f32(
        &q_data,
        &k_f16_as_f32,
        &v_f16_as_f32,
        n_heads_q,
        n_heads_kv,
        head_dim,
        capacity,
        seq_len,
    );
    (out.as_slice::<f32>().to_vec(), reference)
}

fn run_case(
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    capacity: usize,
    seed: u32,
) -> f32 {
    let (out, reference) = run_case_raw(
        n_heads_q,
        n_heads_kv,
        head_dim,
        seq_len,
        capacity,
        seed,
        |_| {},
    );
    nmse(&out, &reference)
}

// ---------------------------------------------------------------------------
// GQA ratio sweep (Stage 1 correctness matrix)
// ---------------------------------------------------------------------------

#[test]
fn flash_gqa_ratio_1_mha() {
    // Multi-head attention (no GQA).  Short seq keeps head-parallel path active.
    let nm = run_case(8, 8, 64, 128, 256, 0x1111);
    assert!(nm < 1e-4, "NMSE {} for GQA=1 seq=128", nm);
}

#[test]
fn flash_gqa_ratio_2() {
    let nm = run_case(8, 4, 64, 512, 1024, 0x2222);
    assert!(nm < 1e-4, "NMSE {} for GQA=2 seq=512", nm);
}

#[test]
fn flash_gqa_ratio_4_llama() {
    // Llama 3.2 1B shape: n_heads_q=32, n_heads_kv=8, head_dim=64.  Use
    // seq_len=2048 to force the flash path (>= THRESHOLD).
    let nm = run_case(32, 8, 64, 2048, 2048, 0x3333);
    assert!(nm < 1e-4, "NMSE {} for GQA=4 Llama seq=2048", nm);
}

#[test]
fn flash_gqa_ratio_6_qwen() {
    // Qwen2.5 1.5B shape: n_heads_q=12, n_heads_kv=2, head_dim=128.  Long
    // seq hits the multi-chunk flash path.
    let nm = run_case(12, 2, 128, 2048, 2048, 0x4444);
    assert!(nm < 1e-4, "NMSE {} for GQA=6 Qwen seq=2048", nm);
}

// ---------------------------------------------------------------------------
// Sequence length sweep
// ---------------------------------------------------------------------------

#[test]
fn flash_seq_32() {
    // Below threshold → head-parallel path.
    let nm = run_case(4, 2, 64, 32, 64, 0xa001);
    assert!(nm < 1e-4, "NMSE {} for seq=32", nm);
}

#[test]
fn flash_seq_128() {
    let nm = run_case(4, 2, 64, 128, 128, 0xa002);
    assert!(nm < 1e-4, "NMSE {} for seq=128", nm);
}

#[test]
fn flash_seq_512() {
    let nm = run_case(4, 2, 64, 512, 512, 0xa003);
    assert!(nm < 1e-4, "NMSE {} for seq=512", nm);
}

#[test]
fn flash_seq_1024() {
    // At / around the threshold: first sequence length where flash may be
    // active.  Should match reference regardless.
    let nm = run_case(4, 2, 128, 1024, 1024, 0xa004);
    assert!(nm < 1e-4, "NMSE {} for seq=1024", nm);
}

#[test]
fn flash_seq_1100_non_multiple() {
    // Non-multiple of chunk_size — exercises the ragged last chunk.
    let nm = run_case(4, 2, 128, 1100, 1100, 0xa005);
    assert!(nm < 1e-4, "NMSE {} for seq=1100", nm);
}

#[test]
fn flash_seq_2048() {
    let nm = run_case(12, 2, 128, 2048, 2048, 0xa006);
    assert!(nm < 1e-4, "NMSE {} for seq=2048", nm);
}

#[test]
fn flash_seq_4096() {
    // Multi-chunk flash path (n_chunks = 4 with chunk=1024).
    let nm = run_case(12, 2, 128, 4096, 4096, 0xa007);
    assert!(nm < 1e-4, "NMSE {} for seq=4096", nm);
}

// ---------------------------------------------------------------------------
// Chunk boundary corner cases
// ---------------------------------------------------------------------------

#[test]
fn flash_chunk_boundary_eq_chunk_size() {
    // seq_len == chunk_size: single-chunk flash path (merge is a no-op).
    let nm = run_case(12, 2, 128, 1024, 1024, 0xb001);
    assert!(nm < 1e-4, "NMSE {} at seq=chunk_size", nm);
}

#[test]
fn flash_chunk_boundary_plus_one() {
    // Two chunks with the second holding a single token.
    let nm = run_case(12, 2, 128, 1025, 1100, 0xb002);
    assert!(nm < 1e-4, "NMSE {} at seq=chunk_size+1", nm);
}

#[test]
fn flash_chunk_boundary_minus_one() {
    // One chunk of chunk_size - 1 tokens → single-chunk flash path.
    let nm = run_case(12, 2, 128, 1023, 1024, 0xb003);
    assert!(nm < 1e-4, "NMSE {} at seq=chunk_size-1", nm);
}

#[test]
fn flash_chunk_boundary_two_chunks_full() {
    // Exactly two full chunks.
    let nm = run_case(12, 2, 128, 2047, 2048, 0xb004);
    assert!(nm < 1e-4, "NMSE {} at seq=2*chunk_size-1", nm);
}

#[test]
fn flash_chunk_boundary_three_chunks_ragged() {
    // Three chunks, last is partial.  Exercises general ragged last-chunk.
    let nm = run_case(12, 2, 128, 2500, 2500, 0xb005);
    assert!(nm < 1e-4, "NMSE {} at seq=2500", nm);
}

// ---------------------------------------------------------------------------
// NaN injection
// ---------------------------------------------------------------------------

#[test]
fn flash_nan_single_token() {
    // Poison one K token inside chunk 1 (i.e. the flash path must sanitize
    // per-token even when the chunk is otherwise valid).
    let n_heads_kv = 2;
    let capacity = 2048;
    let head_dim = 128;
    let seq_len = 2048;
    let (out, reference) = run_case_raw(12, n_heads_kv, head_dim, seq_len, capacity, 0xc001, |k| {
        // Poison K[kv_h=0, t=1500] — deep inside chunk 1.
        let off = (0 * capacity + 1500) * head_dim;
        for d in 0..head_dim {
            k[off + d] = f32::NAN;
        }
    });
    for (i, &x) in out.iter().enumerate() {
        assert!(x.is_finite(), "out[{}] = {} not finite", i, x);
    }
    let nm = nmse(&out, &reference);
    assert!(nm < 1e-4, "NMSE {} for single-token NaN", nm);
}

#[test]
fn flash_nan_full_chunk() {
    // Poison every K token in one full chunk for kv_h=0.  The flash path
    // must handle that chunk's (m_c = -inf, l_c = 0) partial cleanly during
    // the LSE merge, and non-poisoned chunks must still deliver a finite
    // weighted sum.
    let n_heads_kv = 2;
    let capacity = 4096;
    let head_dim = 128;
    let seq_len = 4096;
    let chunk_start = 1024; // chunk index 1
    let chunk_end = 2048;
    let (out, reference) = run_case_raw(12, n_heads_kv, head_dim, seq_len, capacity, 0xc002, |k| {
        for t in chunk_start..chunk_end {
            let off = (0 * capacity + t) * head_dim;
            for d in 0..head_dim {
                k[off + d] = f32::NAN;
            }
        }
    });
    for (i, &x) in out.iter().enumerate() {
        assert!(x.is_finite(), "out[{}] = {} not finite", i, x);
    }
    let nm = nmse(&out, &reference);
    assert!(nm < 1e-4, "NMSE {} for full-chunk NaN", nm);
}

#[test]
fn flash_nan_every_token_uniform_fallback() {
    // Every K token poisoned → reference collapses to uniform-V.  Both
    // head-parallel (short) and flash (long) paths must agree.
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let seq_len = 64; // short: head-parallel path
    let capacity = 64;
    let backend: Arc<dyn Backend> = Arc::new(CpuBackendNeon::new());

    let q_data = gen_data(n_heads_q * head_dim, 0xc003);
    let mut k_data = gen_data(n_heads_kv * capacity * head_dim, 0xc004);
    let v_data = gen_data(n_heads_kv * capacity * head_dim, 0xc005);
    for x in k_data.iter_mut() {
        *x = f32::NAN;
    }

    let q = make_f32_tensor(&backend, vec![n_heads_q, head_dim], &q_data);
    let k = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &k_data);
    let v = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &v_data);
    let mut out = make_f32_zeros(&backend, vec![n_heads_q, head_dim]);

    backend
        .attention_gen(
            &q, &k, &v, &mut out, n_heads_q, n_heads_kv, head_dim, seq_len, None,
        )
        .unwrap();

    for (i, &x) in out.as_slice::<f32>().iter().enumerate() {
        assert!(x.is_finite(), "fallback out[{}] = {} not finite", i, x);
    }
    let uniform = 1.0 / seq_len as f32;
    let mut expected = vec![0.0f32; n_heads_q * head_dim];
    for h in 0..n_heads_q {
        let kv_h = h / (n_heads_q / n_heads_kv);
        for t in 0..seq_len {
            let v_off = (kv_h * capacity + t) * head_dim;
            for d in 0..head_dim {
                let vv = half::f16::from_f32(v_data[v_off + d]).to_f32();
                expected[h * head_dim + d] += uniform * vv;
            }
        }
    }
    let nm = nmse(out.as_slice::<f32>(), &expected);
    assert!(nm < 1e-4, "fallback NMSE {} too large", nm);
}

#[test]
fn flash_nan_every_token_long_seq_uniform_fallback() {
    // Same as above but with a long seq → flash path.  Must still collapse
    // cleanly to uniform without NaN leaking through the merge.
    let n_heads_q = 12;
    let n_heads_kv = 2;
    let head_dim = 128;
    let seq_len = 2048;
    let capacity = 2048;
    let backend: Arc<dyn Backend> = Arc::new(CpuBackendNeon::new());

    let q_data = gen_data(n_heads_q * head_dim, 0xc006);
    let mut k_data = gen_data(n_heads_kv * capacity * head_dim, 0xc007);
    let v_data = gen_data(n_heads_kv * capacity * head_dim, 0xc008);
    for x in k_data.iter_mut() {
        *x = f32::NAN;
    }

    let q = make_f32_tensor(&backend, vec![n_heads_q, head_dim], &q_data);
    let k = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &k_data);
    let v = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &v_data);
    let mut out = make_f32_zeros(&backend, vec![n_heads_q, head_dim]);

    backend
        .attention_gen(
            &q, &k, &v, &mut out, n_heads_q, n_heads_kv, head_dim, seq_len, None,
        )
        .unwrap();

    for (i, &x) in out.as_slice::<f32>().iter().enumerate() {
        assert!(x.is_finite(), "fallback out[{}] = {} not finite", i, x);
    }
    let uniform = 1.0 / seq_len as f32;
    let mut expected = vec![0.0f32; n_heads_q * head_dim];
    for h in 0..n_heads_q {
        let kv_h = h / (n_heads_q / n_heads_kv);
        for t in 0..seq_len {
            let v_off = (kv_h * capacity + t) * head_dim;
            for d in 0..head_dim {
                let vv = half::f16::from_f32(v_data[v_off + d]).to_f32();
                expected[h * head_dim + d] += uniform * vv;
            }
        }
    }
    let nm = nmse(out.as_slice::<f32>(), &expected);
    assert!(nm < 1e-3, "long-seq fallback NMSE {} too large", nm);
}

// ---------------------------------------------------------------------------
// Routing invariants
// ---------------------------------------------------------------------------

#[test]
fn flash_short_bit_exact_head_parallel() {
    // When cache_seq_len < THRESHOLD the dispatcher MUST use the Step 1
    // head-parallel path.  Since we cannot force the flash path from the
    // outside, this test only asserts that the result on a short seq is
    // bit-exact against a second call on the same inputs (idempotence) —
    // any non-determinism from the flash path would show up as drift.
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let seq_len = 256; // well below any sensible THRESHOLD
    let capacity = 256;
    let backend: Arc<dyn Backend> = Arc::new(CpuBackendNeon::new());

    let q_data = gen_data(n_heads_q * head_dim, 0xd001);
    let k_data = gen_data(n_heads_kv * capacity * head_dim, 0xd002);
    let v_data = gen_data(n_heads_kv * capacity * head_dim, 0xd003);

    let q = make_f32_tensor(&backend, vec![n_heads_q, head_dim], &q_data);
    let k = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &k_data);
    let v = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &v_data);

    let mut out_a = make_f32_zeros(&backend, vec![n_heads_q, head_dim]);
    let mut out_b = make_f32_zeros(&backend, vec![n_heads_q, head_dim]);

    backend
        .attention_gen(
            &q, &k, &v, &mut out_a, n_heads_q, n_heads_kv, head_dim, seq_len, None,
        )
        .unwrap();
    backend
        .attention_gen(
            &q, &k, &v, &mut out_b, n_heads_q, n_heads_kv, head_dim, seq_len, None,
        )
        .unwrap();

    let a = out_a.as_slice::<f32>();
    let b = out_b.as_slice::<f32>();
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "short seq not bit-exact across calls at idx {}",
            i
        );
    }
}

#[test]
fn flash_needs_scores_routes_head_parallel() {
    // `scores_out = Some(...)` forces the head-parallel path (diagnostic
    // contract: per-head post-softmax weights must sum to 1).  Validate by
    // checking the scores row-sum property even on a long seq that would
    // otherwise hit the flash path.
    let n_heads_q = 12;
    let n_heads_kv = 2;
    let head_dim = 128;
    let seq_len = 2048;
    let capacity = 2048;
    let backend: Arc<dyn Backend> = Arc::new(CpuBackendNeon::new());

    let q_data = gen_data(n_heads_q * head_dim, 0xd004);
    let k_data = gen_data(n_heads_kv * capacity * head_dim, 0xd005);
    let v_data = gen_data(n_heads_kv * capacity * head_dim, 0xd006);

    let q = make_f32_tensor(&backend, vec![n_heads_q, head_dim], &q_data);
    let k = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &k_data);
    let v = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &v_data);
    let mut out = make_f32_zeros(&backend, vec![n_heads_q, head_dim]);

    let mut scores = vec![0.0f32; n_heads_q * seq_len];

    backend
        .attention_gen(
            &q,
            &k,
            &v,
            &mut out,
            n_heads_q,
            n_heads_kv,
            head_dim,
            seq_len,
            Some(&mut scores),
        )
        .unwrap();

    // Each row of post-softmax weights must sum to 1 and be in [0, 1].
    for h in 0..n_heads_q {
        let row = &scores[h * seq_len..(h + 1) * seq_len];
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "head {} scores sum {} != 1 (flash path must not be taken when scores requested)",
            h,
            sum
        );
        for (i, &w) in row.iter().enumerate() {
            assert!(
                w >= 0.0 && w <= 1.0 + 1e-5 && w.is_finite(),
                "head {} score[{}] = {} out of range",
                h,
                i,
                w
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Smoke test: head_dim=64 Llama configuration at 4K
// ---------------------------------------------------------------------------

#[test]
fn flash_llama_4k() {
    // Full Llama 3.2 1B shape at 4K: exercises the flash path with 8 KV
    // heads × 4 chunks.
    let nm = run_case(32, 8, 64, 4096, 4096, 0xe001);
    assert!(nm < 1e-4, "NMSE {} for Llama 4K", nm);
}
