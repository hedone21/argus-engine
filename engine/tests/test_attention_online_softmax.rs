//! Tests for the NEON online-softmax `attention_gen_f16_neon` path.
//!
//! The NEON implementation fuses QK^T, softmax, and weighted V into a single
//! pass using the online-softmax identity.  These tests validate:
//!   - Mathematical equivalence with a straightforward F32 3-pass reference
//!     (NMSE < 1e-4).
//!   - Correct handling of NaN in K values — the per-token NaN gate must
//!     discard poisoned logits without corrupting the running state.
//!   - Correctness across several (seq_len, head_dim, n_heads_q, n_heads_kv)
//!     configurations that exercise both the 4-way unrolled inner loop and
//!     the scalar tail.
//!
//! These tests are compiled only on aarch64 because `CpuBackendNeon` is
//! gated on `target_arch = "aarch64"`.  On non-aarch64 hosts the binary
//! is still produced but contains no test functions.

#![cfg(target_arch = "aarch64")]

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::neon::CpuBackendNeon;
use argus_engine::buffer::DType;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use std::sync::Arc;

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

/// Build an F32 tensor from `data` on the given backend.
fn make_f32_tensor(backend: &Arc<dyn Backend>, shape: Vec<usize>, data: &[f32]) -> Tensor {
    let memory = Galloc::new();
    let n = data.len();
    let buf = memory.alloc(n * 4, DType::F32).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.as_mut_ptr(), n * 4);
    }
    Tensor::new(Shape::new(shape), buf, backend.clone())
}

/// Zero-initialized F32 tensor of the given shape.
fn make_f32_zeros(backend: &Arc<dyn Backend>, shape: Vec<usize>) -> Tensor {
    let n: usize = shape.iter().product();
    let memory = Galloc::new();
    let buf = memory.alloc(n * 4, DType::F32).unwrap();
    unsafe {
        std::ptr::write_bytes(buf.as_mut_ptr(), 0, n * 4);
    }
    Tensor::new(Shape::new(shape), buf, backend.clone())
}

/// Build an F16 tensor from the given F32 data (converted element-wise).
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
/// Input slice layouts:
///   q:  [n_heads_q, head_dim]
///   k:  [n_heads_kv, capacity, head_dim]    (first `seq_len` positions used)
///   v:  [n_heads_kv, capacity, head_dim]
/// Output:
///   out: [n_heads_q, head_dim]
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

        // Pass 2: softmax.
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if m == f32::NEG_INFINITY {
            // Fallback: uniform weights.
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

/// Normalized mean squared error between two equally-sized slices.
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
/// and compare against the F32 reference.  Returns the measured NMSE.
fn run_case(
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    capacity: usize,
    seed: u32,
) -> f32 {
    assert!(capacity >= seq_len);
    let backend: Arc<dyn Backend> = Arc::new(CpuBackendNeon::new());

    let q_data = gen_data(n_heads_q * head_dim, seed);
    let k_data = gen_data(n_heads_kv * capacity * head_dim, seed.wrapping_add(1));
    let v_data = gen_data(n_heads_kv * capacity * head_dim, seed.wrapping_add(2));

    // NEON expects F16 KV; reference uses F32 equivalents of the same F16 values
    // to keep the comparison fair (no double precision drift from f32→f16→f32).
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
    nmse(out.as_slice::<f32>(), &reference)
}

#[test]
fn online_softmax_small_seq() {
    // seq_len=32: exactly 8 chunks of 4 timesteps, no remainder.
    let nm = run_case(4, 2, 64, 32, 64, 0x1234);
    assert!(nm < 1e-4, "NMSE too large for seq_len=32: {}", nm);
}

#[test]
fn online_softmax_medium_seq() {
    let nm = run_case(4, 2, 64, 128, 256, 0x5678);
    assert!(nm < 1e-4, "NMSE too large for seq_len=128: {}", nm);
}

#[test]
fn online_softmax_long_seq() {
    let nm = run_case(4, 2, 64, 512, 1024, 0x9abc);
    assert!(nm < 1e-4, "NMSE too large for seq_len=512: {}", nm);
}

#[test]
fn online_softmax_seq_with_remainder() {
    // seq_len=35 triggers the (full_4 * 4)..cache_seq_len scalar tail.
    let nm = run_case(4, 2, 64, 35, 64, 0xdead);
    assert!(nm < 1e-4, "NMSE too large for seq_len=35: {}", nm);
}

#[test]
fn online_softmax_gqa_8x2() {
    // GQA ratio = 4 — 8 Q-heads share 2 KV-heads.
    let nm = run_case(8, 2, 64, 100, 128, 0xbeef);
    assert!(nm < 1e-4, "NMSE too large for 8x2 GQA: {}", nm);
}

#[test]
fn online_softmax_with_scores_out() {
    // Diagnostic path: scores_out=Some must receive post-softmax (normalized)
    // probabilities.  Each row must sum to 1.
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let seq_len = 64;
    let capacity = 128;
    let backend: Arc<dyn Backend> = Arc::new(CpuBackendNeon::new());

    let q_data = gen_data(n_heads_q * head_dim, 7);
    let k_data = gen_data(n_heads_kv * capacity * head_dim, 11);
    let v_data = gen_data(n_heads_kv * capacity * head_dim, 13);

    let q = make_f32_tensor(&backend, vec![n_heads_q, head_dim], &q_data);
    let k = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &k_data);
    let v = make_f16_tensor_from_f32(&backend, vec![1, n_heads_kv, capacity, head_dim], &v_data);
    let mut out = make_f32_zeros(&backend, vec![n_heads_q, head_dim]);

    // Scores buffer: stride = seq_len per head.
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

    for h in 0..n_heads_q {
        let row = &scores[h * seq_len..(h + 1) * seq_len];
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "head {} scores sum {} != 1",
            h,
            sum
        );
        for (i, &w) in row.iter().enumerate() {
            assert!(
                w >= 0.0 && w.is_finite(),
                "head {} score[{}] = {} not a valid probability",
                h,
                i,
                w
            );
        }
    }
}

#[test]
fn online_softmax_nan_in_k_skipped() {
    // Poison K[t=5] for kv_head 0 with NaN.  The token should be treated as
    // weight-0 and the remaining output must stay finite and match a reference
    // that also excludes that token from the softmax.
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let seq_len = 32;
    let capacity = 32;
    let backend: Arc<dyn Backend> = Arc::new(CpuBackendNeon::new());

    let q_data = gen_data(n_heads_q * head_dim, 21);
    let mut k_data = gen_data(n_heads_kv * capacity * head_dim, 23);
    let v_data = gen_data(n_heads_kv * capacity * head_dim, 29);

    // Inject NaN into all head_dim lanes of K[kv_h=0, t=5].  A Q·K dot
    // touching a NaN lane produces NaN, which the NEON path must sanitize.
    let poison_off = (0 * capacity + 5) * head_dim;
    for d in 0..head_dim {
        k_data[poison_off + d] = f32::NAN;
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

    // All output values must be finite (not NaN / not Inf).
    for (i, &x) in out.as_slice::<f32>().iter().enumerate() {
        assert!(x.is_finite(), "out[{}] = {} not finite", i, x);
    }

    // Compare against reference that also sees the NaN; both must converge
    // to identical semantics (token skipped from softmax).
    let k_f16_as_f32: Vec<f32> = k_data
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    let v_f16_as_f32: Vec<f32> = v_data
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
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
    let nm = nmse(out.as_slice::<f32>(), &reference);
    assert!(
        nm < 1e-4,
        "NMSE {} too large when a K token is NaN-poisoned",
        nm
    );
}

#[test]
fn online_softmax_all_nan_falls_back_uniform() {
    // Corner case: every K row NaN-poisoned → reference collapses to uniform
    // average of V.  The NEON path's fallback branch must produce finite
    // output, not NaN.
    let n_heads_q = 2;
    let n_heads_kv = 1;
    let head_dim = 64;
    let seq_len = 16;
    let capacity = 16;
    let backend: Arc<dyn Backend> = Arc::new(CpuBackendNeon::new());

    let q_data = gen_data(n_heads_q * head_dim, 31);
    let mut k_data = gen_data(n_heads_kv * capacity * head_dim, 37);
    let v_data = gen_data(n_heads_kv * capacity * head_dim, 41);
    // Poison every lane.
    for v in k_data.iter_mut() {
        *v = f32::NAN;
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

    // Fallback = uniform average of V over t for each kv head.
    let uniform = 1.0 / seq_len as f32;
    let mut expected = vec![0.0f32; n_heads_q * head_dim];
    for h in 0..n_heads_q {
        let kv_h = h / (n_heads_q / n_heads_kv);
        for t in 0..seq_len {
            let v_off = (kv_h * capacity + t) * head_dim;
            for d in 0..head_dim {
                let v_f16 = half::f16::from_f32(v_data[v_off + d]).to_f32();
                expected[h * head_dim + d] += uniform * v_f16;
            }
        }
    }
    let nm = nmse(out.as_slice::<f32>(), &expected);
    assert!(nm < 1e-4, "fallback NMSE {} too large", nm);
}
