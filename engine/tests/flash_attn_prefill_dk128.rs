//! Host integration tests for flash attention prefill at head_dim ∈ {64, 128}.
//!
//! - `prefill_dk128_self_consistent`: dispatches `flash_attention_prefill_gpu`
//!   twice with identical Qwen 2.5-1.5B shape inputs (head_dim=128,
//!   n_heads_q=12, n_heads_kv=2) and asserts deterministic, non-zero output.
//! - `prefill_dk64_regression`: repeats the same self-consistency check for
//!   the existing head_dim=64 / Llama 3.2 1B shape to guard against
//!   regressions introduced by the DK=128 dispatcher split.
//!
//! Numerical correctness vs. a CPU reference is covered by on-device
//! end-to-end inference sanity. Hosts where the DK=128 program fails to
//! compile (e.g. macOS Apple OpenCL, NVIDIA OpenCL with strict compiler
//! settings) skip cleanly — the kernel builder returns `Ok(false)`.

#![cfg(feature = "opencl")]

use argus_engine::backend::Backend;
use argus_engine::backend::opencl::OpenCLBackend;
use argus_engine::backend::opencl::memory::OpenCLMemory;
use argus_engine::buffer::DType;
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

fn upload_f16(
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    data: &[u16],
    shape: Vec<usize>,
) -> Tensor {
    let buf = memory.alloc(data.len() * 2, DType::F16).unwrap();
    let mut t = Tensor::new(Shape::new(shape), buf, backend.clone());
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}

/// Run prefill flash_attention_gpu twice and assert deterministic output with
/// non-trivial values. Skips if the backend returns `Ok(false)` on the first
/// call (kernel compile failed on this host).
fn run_prefill_self_consistency(
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    cache_seq_len: usize,
) {
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
        }
    };
    let ocl_arc: Arc<dyn Backend> = Arc::new(backend);
    let ocl = ocl_arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCLBackend downcast");

    let capacity = (cache_seq_len + 63) & !63;
    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));

    // Deterministic Q: seq_len queries, n_heads_q heads, head_dim each.
    let q_elems = seq_len * n_heads_q * head_dim;
    let q_data: Vec<f32> = (0..q_elems)
        .map(|i| ((i as f32) * 0.003).sin() + 0.1)
        .collect();

    // Deterministic F16 KV: [1, n_heads_kv, capacity, head_dim].
    let kv_total = n_heads_kv * capacity * head_dim;
    let mut k_data = vec![0u16; kv_total];
    let mut v_data = vec![0u16; kv_total];
    for h in 0..n_heads_kv {
        for p in 0..cache_seq_len {
            for d in 0..head_dim {
                let idx = h * capacity * head_dim + p * head_dim + d;
                let seed = (h * 19 + p * 5 + d * 2) as f32;
                k_data[idx] = half::f16::from_f32((seed * 0.0015).cos()).to_bits();
                v_data[idx] = half::f16::from_f32((seed * 0.0037).sin()).to_bits();
            }
        }
    }

    let q = upload_f32(
        &ocl_arc,
        &*memory,
        &q_data,
        vec![1, seq_len, n_heads_q, head_dim],
    );
    let k = upload_f16(
        &ocl_arc,
        &*memory,
        &k_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let v = upload_f16(
        &ocl_arc,
        &*memory,
        &v_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );

    let out_len = seq_len * n_heads_q * head_dim;
    let zero = vec![0.0f32; out_len];
    let mut out_a = upload_f32(
        &ocl_arc,
        &*memory,
        &zero,
        vec![1, seq_len, n_heads_q, head_dim],
    );
    let mut out_b = upload_f32(
        &ocl_arc,
        &*memory,
        &zero,
        vec![1, seq_len, n_heads_q, head_dim],
    );

    let ok_a = ocl
        .flash_attention_prefill_gpu(
            &q,
            &k,
            &v,
            &mut out_a,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            head_dim,
            capacity,
            1,
            true, // is_head_major
        )
        .expect("prefill flash call a");
    if !ok_a {
        eprintln!(
            "Skipping: prefill flash DK={} unavailable on this host",
            head_dim
        );
        return;
    }
    ocl_arc.synchronize().unwrap();

    let ok_b = ocl
        .flash_attention_prefill_gpu(
            &q,
            &k,
            &v,
            &mut out_b,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            head_dim,
            capacity,
            1,
            true,
        )
        .expect("prefill flash call b");
    assert!(
        ok_b,
        "prefill flash must dispatch for second call if first call dispatched"
    );
    ocl_arc.synchronize().unwrap();

    let out_bytes = out_len * std::mem::size_of::<f32>();
    let mut raw_a = vec![0u8; out_bytes];
    let mut raw_b = vec![0u8; out_bytes];
    ocl_arc.read_buffer(&out_a, &mut raw_a).unwrap();
    ocl_arc.read_buffer(&out_b, &mut raw_b).unwrap();

    let a: &[f32] = unsafe { std::slice::from_raw_parts(raw_a.as_ptr() as *const f32, out_len) };
    let b: &[f32] = unsafe { std::slice::from_raw_parts(raw_b.as_ptr() as *const f32, out_len) };
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "nondeterministic prefill flash DK={head_dim} output at [{i}]: {x} vs {y}"
        );
    }
    assert!(
        a.iter().any(|&v| v.abs() > 1e-6),
        "prefill flash DK={head_dim} produced all-zero output — likely a dispatch no-op"
    );
    assert!(
        a.iter().all(|&v| v.is_finite()),
        "prefill flash DK={head_dim} produced NaN/Inf — kernel numerical failure"
    );
}

/// Qwen 2.5-1.5B prefill shape: 12 Q heads, 2 KV heads, head_dim=128.
/// Sweeps across a range of KV lengths to exercise multi-BLOCK_N tiling.
#[test]
fn prefill_dk128_self_consistent() {
    for &(seq_len, kv_len) in &[(64usize, 64usize), (64, 512), (128, 1024), (256, 2048)] {
        eprintln!("[test] dk128 seq_len={seq_len} kv_len={kv_len}");
        run_prefill_self_consistency(12, 2, 128, seq_len, kv_len);
    }
}

/// Llama 3.2 1B prefill shape regression guard: 32 Q heads, 8 KV heads,
/// head_dim=64. Ensures the dispatcher split did not break the existing
/// head_dim=64 path.
#[test]
fn prefill_dk64_regression() {
    for &(seq_len, kv_len) in &[(64usize, 64usize), (64, 512), (128, 1024)] {
        eprintln!("[test] dk64 seq_len={seq_len} kv_len={kv_len}");
        run_prefill_self_consistency(32, 8, 64, seq_len, kv_len);
    }
}

/// Unsupported head_dim must return Ok(false) without panicking.
#[test]
fn prefill_unsupported_head_dim_returns_false() {
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
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

    let n_heads_q = 4usize;
    let n_heads_kv = 1usize;
    let head_dim = 96usize; // not in {64, 128}
    let seq_len = 8usize;
    let cache_seq_len = 8usize;
    let capacity = 64usize;

    let q = upload_f32(
        &ocl_arc,
        &*memory,
        &vec![0.0f32; seq_len * n_heads_q * head_dim],
        vec![1, seq_len, n_heads_q, head_dim],
    );
    let k = upload_f16(
        &ocl_arc,
        &*memory,
        &vec![0u16; n_heads_kv * capacity * head_dim],
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let v = upload_f16(
        &ocl_arc,
        &*memory,
        &vec![0u16; n_heads_kv * capacity * head_dim],
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let mut out = upload_f32(
        &ocl_arc,
        &*memory,
        &vec![0.0f32; seq_len * n_heads_q * head_dim],
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
            true,
        )
        .expect("prefill call must not error");
    assert!(
        !dispatched,
        "head_dim={head_dim} is not in the compiled DK set; dispatcher must return Ok(false)"
    );
}
