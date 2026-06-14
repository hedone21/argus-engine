//! Host integration tests for flash attention prefill at head_dim=256
//! with F32 KV (Gemma3 4B / Gemma3 1B shape).
//!
//! Motivation: NVIDIA host OpenCL dev environment + Gemma3 4B crashes with
//! `CL_OUT_OF_RESOURCES` during eval-ll because head_dim=256 is not in the
//! compiled DK set (64, 128). The CPU fallback pulls K/V back per layer via
//! `clEnqueueReadBuffer`, and the accumulated driver staging pressure
//! exhausts resources after several questions. Adding a GPU DK=256 dispatch
//! eliminates the fallback path and the associated read_buffer churn.
//!
//! Tests:
//! - `prefill_dk256_f32_self_consistent`: deterministic, non-zero output on
//!   two identical dispatches (Gemma3 shape).
//! - `prefill_dk256_f32_vs_cpu_reference`: GPU output matches
//!   `flash_attention_forward_strided` (the CPU fallback this kernel replaces).
//! - `prefill_dk256_gemma3_4b_shape`: matches 4B production config
//!   (n_heads_q=16, n_heads_kv=4, seq_len sweep).
//!
//! Hosts where the DK=256 program fails to compile skip cleanly via
//! `Ok(false)` from the dispatcher.

#![cfg(feature = "opencl")]

use argus_engine::backend::Backend;
use argus_engine::backend::opencl::OpenCLBackend;
use argus_engine::backend::opencl::memory::OpenCLMemory;
use argus_engine::buffer::DType;
use argus_engine::layers::attention::flash_attention_forward_strided;
use argus_engine::memory::Memory;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use std::sync::Arc;

fn upload_f32(
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    data: &[f32],
    shape: Vec<usize>,
) -> Tensor {
    let buf = memory.alloc(data.len() * 4, DType::F32).unwrap();
    let mut t = Tensor::new(Shape::new(shape), buf, backend.clone());
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}

fn backend_and_memory() -> Option<(Arc<dyn Backend>, Arc<dyn Memory>)> {
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return None;
        }
    };
    let ocl_arc: Arc<dyn Backend> = Arc::new(backend);
    let ocl = ocl_arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCLBackend downcast");
    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));
    Some((ocl_arc, memory))
}

/// Generate deterministic Q, K, V for reproducibility.
fn make_inputs(
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    capacity: usize,
    cache_seq_len: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_elems = seq_len * n_heads_q * head_dim;
    let q: Vec<f32> = (0..q_elems)
        .map(|i| ((i as f32) * 0.0013).sin() + 0.05)
        .collect();

    // HeadMajor: [1, n_heads_kv, capacity, head_dim]
    let kv_total = n_heads_kv * capacity * head_dim;
    let mut k = vec![0.0f32; kv_total];
    let mut v = vec![0.0f32; kv_total];
    for h in 0..n_heads_kv {
        for p in 0..cache_seq_len {
            for d in 0..head_dim {
                let idx = h * capacity * head_dim + p * head_dim + d;
                let seed = (h as f32) * 0.91 + (p as f32) * 0.07 + (d as f32) * 0.011;
                k[idx] = (seed * 0.0017).cos() * 0.5;
                v[idx] = (seed * 0.0039).sin() * 0.5;
            }
        }
    }
    (q, k, v)
}

#[allow(clippy::too_many_arguments)]
fn run_gpu_prefill(
    ocl_arc: &Arc<dyn Backend>,
    memory: &dyn Memory,
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    cache_seq_len: usize,
    capacity: usize,
) -> Option<Vec<f32>> {
    let ocl = ocl_arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCLBackend downcast");

    let q = upload_f32(
        ocl_arc,
        memory,
        q_data,
        vec![1, seq_len, n_heads_q, head_dim],
    );
    let k = upload_f32(
        ocl_arc,
        memory,
        k_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let v = upload_f32(
        ocl_arc,
        memory,
        v_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );

    let out_len = seq_len * n_heads_q * head_dim;
    let zero = vec![0.0f32; out_len];
    let mut out = upload_f32(
        ocl_arc,
        memory,
        &zero,
        vec![1, seq_len, n_heads_q, head_dim],
    );

    let dispatched = ocl
        .flash_attention_prefill_gpu(
            &q,
            &k,
            &v,
            &mut out,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            head_dim,
            capacity,
            1,
            true, // is_head_major
        )
        .expect("prefill flash call");
    if !dispatched {
        return None;
    }
    ocl_arc.synchronize().unwrap();

    let out_bytes = out_len * std::mem::size_of::<f32>();
    let mut raw = vec![0u8; out_bytes];
    ocl_arc.read_buffer(&out, &mut raw).unwrap();
    let out_vec =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, out_len) }.to_vec();
    Some(out_vec)
}

#[allow(clippy::too_many_arguments)]
fn run_cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    cache_seq_len: usize,
    capacity: usize,
) -> Vec<f32> {
    let out_len = seq_len * n_heads_q * head_dim;
    let mut out = vec![0.0f32; out_len];

    // HeadMajor strides matching transformer_layer::forward prefill fallback.
    let q_stride = n_heads_q * head_dim;
    let k_pos_stride = head_dim;
    let kv_head_stride = capacity * head_dim;

    flash_attention_forward_strided(
        q,
        k,
        v,
        &mut out,
        n_heads_q,
        n_heads_kv,
        seq_len,
        cache_seq_len,
        head_dim,
        q_stride,
        k_pos_stride,
        k_pos_stride,
        q_stride,
        kv_head_stride,
        0, // q_start_pos
        32,
        32,
        None, // no sliding window
    );
    out
}

/// Determinism + sanity check: two identical dispatches produce identical,
/// non-zero, finite output.
#[test]
fn prefill_dk256_f32_self_consistent() {
    let Some((ocl_arc, memory)) = backend_and_memory() else {
        return;
    };

    let n_heads_q = 4usize;
    let n_heads_kv = 2usize;
    let head_dim = 256usize;
    let seq_len = 16usize;
    let cache_seq_len = 16usize;
    let capacity = 64usize;

    let (q, k, v) = make_inputs(
        n_heads_q,
        n_heads_kv,
        head_dim,
        seq_len,
        capacity,
        cache_seq_len,
    );

    let a = run_gpu_prefill(
        &ocl_arc,
        &*memory,
        &q,
        &k,
        &v,
        n_heads_q,
        n_heads_kv,
        head_dim,
        seq_len,
        cache_seq_len,
        capacity,
    )
    .expect("DK=256 kernel must dispatch for head_dim=256 F32 KV");
    let b = run_gpu_prefill(
        &ocl_arc,
        &*memory,
        &q,
        &k,
        &v,
        n_heads_q,
        n_heads_kv,
        head_dim,
        seq_len,
        cache_seq_len,
        capacity,
    )
    .expect("DK=256 kernel must dispatch on second call");

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "nondeterministic DK=256 prefill output at [{i}]: {x} vs {y}"
        );
    }
    assert!(
        a.iter().any(|&v| v.abs() > 1e-6),
        "DK=256 prefill produced all-zero output — dispatch no-op?"
    );
    assert!(
        a.iter().all(|&v| v.is_finite()),
        "DK=256 prefill produced NaN/Inf"
    );
}

/// Numerical correctness: GPU output matches CPU reference within tolerance.
#[test]
fn prefill_dk256_f32_vs_cpu_reference() {
    let Some((ocl_arc, memory)) = backend_and_memory() else {
        return;
    };

    let n_heads_q = 4usize;
    let n_heads_kv = 2usize;
    let head_dim = 256usize;

    // First-prefill scenarios only: seq_len == cache_seq_len (matches eval-ll
    // full-prefill where the cache is empty before this call). Chunked-prefill
    // with pre-filled cache uses a different causal semantics and is exercised
    // by on-device end-to-end tests.
    for &(seq_len, cache_seq_len) in &[(8usize, 8usize), (16, 16), (64, 64), (128, 128)] {
        let capacity = (cache_seq_len + 63) & !63;
        let (q, k, v) = make_inputs(
            n_heads_q,
            n_heads_kv,
            head_dim,
            seq_len,
            capacity,
            cache_seq_len,
        );

        let gpu_out = run_gpu_prefill(
            &ocl_arc,
            &*memory,
            &q,
            &k,
            &v,
            n_heads_q,
            n_heads_kv,
            head_dim,
            seq_len,
            cache_seq_len,
            capacity,
        )
        .unwrap_or_else(|| {
            panic!("DK=256 kernel must dispatch (seq_len={seq_len}, kv_len={cache_seq_len})")
        });

        let cpu_out = run_cpu_reference(
            &q,
            &k,
            &v,
            n_heads_q,
            n_heads_kv,
            head_dim,
            seq_len,
            cache_seq_len,
            capacity,
        );

        assert_eq!(gpu_out.len(), cpu_out.len());
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let abs = (g - c).abs();
            let rel = abs / c.abs().max(1e-5);
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                abs < 5e-3 || rel < 5e-3,
                "DK=256 GPU vs CPU mismatch at [{i}] seq_len={seq_len} kv_len={cache_seq_len}: gpu={g} cpu={c} abs={abs} rel={rel}"
            );
        }
        eprintln!(
            "[test] dk256 seq_len={seq_len} kv_len={cache_seq_len}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
        );
    }
}

/// Gemma3 4B production shape: 16 Q heads, 4 KV heads, head_dim=256.
#[test]
fn prefill_dk256_gemma3_4b_shape() {
    let Some((ocl_arc, memory)) = backend_and_memory() else {
        return;
    };

    let n_heads_q = 16usize;
    let n_heads_kv = 4usize;
    let head_dim = 256usize;

    // First-prefill scenarios: seq_len == cache_seq_len.
    for &(seq_len, cache_seq_len) in &[(32usize, 32usize), (128, 128), (256, 256)] {
        let capacity = (cache_seq_len + 63) & !63;
        let (q, k, v) = make_inputs(
            n_heads_q,
            n_heads_kv,
            head_dim,
            seq_len,
            capacity,
            cache_seq_len,
        );

        let gpu_out = run_gpu_prefill(
            &ocl_arc,
            &*memory,
            &q,
            &k,
            &v,
            n_heads_q,
            n_heads_kv,
            head_dim,
            seq_len,
            cache_seq_len,
            capacity,
        )
        .unwrap_or_else(|| panic!("Gemma3 4B DK=256 kernel must dispatch (seq_len={seq_len})"));

        let cpu_out = run_cpu_reference(
            &q,
            &k,
            &v,
            n_heads_q,
            n_heads_kv,
            head_dim,
            seq_len,
            cache_seq_len,
            capacity,
        );

        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let abs = (g - c).abs();
            let rel = abs / c.abs().max(1e-5);
            assert!(
                abs < 5e-3 || rel < 5e-3,
                "Gemma3 4B DK=256 mismatch at [{i}] seq_len={seq_len} kv_len={cache_seq_len}: gpu={g} cpu={c} abs={abs} rel={rel}"
            );
        }
        eprintln!("[test] dk256 gemma3-4b seq_len={seq_len} kv_len={cache_seq_len}: OK");
    }
}
