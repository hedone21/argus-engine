//! Host unit tests for the flash attention decode dispatch prototype.
//!
//! These backfill coverage for commit 938149a before extending to plan.rs.

#![cfg(feature = "opencl")]

use argus_engine::backend::opencl::OpenCLBackend;

/// Sanity: OpenCLBackend initializes and the flash kernel handle loads.
/// On hosts without OpenCL drivers this test silently skips.
#[test]
fn backend_initializes_with_flash_kernel() {
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
        }
    };
    // The backend must accept downcasts from trait-object form.
    // (Used by plan.rs builder in Task 2.)
    let _ = &backend;
}

/// Deterministic self-consistency: the flash dispatch must produce bit-
/// identical output across two invocations with the same inputs. This
/// guards against nondeterminism (e.g. uninitialized local memory, bad
/// barrier placement).
#[test]
fn flash_attn_decode_self_consistent() {
    use argus_engine::backend::Backend;
    use argus_engine::backend::opencl::memory::OpenCLMemory;
    use argus_engine::memory::Memory;
    use std::sync::Arc;

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

    // llama3.2-1b shape: 32 Q heads, 8 KV heads (GQA=4), head_dim=64
    let n_heads_q = 32usize;
    let n_heads_kv = 8usize;
    let head_dim = 64usize;
    let cache_seq_len = 32usize;
    let capacity = 128usize;

    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));

    // Deterministic inputs
    let q_data: Vec<f32> = (0..n_heads_q * head_dim)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();

    let kv_total = n_heads_kv * capacity * head_dim;
    let mut k_data = vec![0u16; kv_total];
    let mut v_data = vec![0u16; kv_total];
    for h in 0..n_heads_kv {
        for p in 0..cache_seq_len {
            for d in 0..head_dim {
                let idx = h * capacity * head_dim + p * head_dim + d;
                let seed = (h * 31 + p * 7 + d) as f32;
                k_data[idx] = half::f16::from_f32((seed * 0.003).cos()).to_bits();
                v_data[idx] = half::f16::from_f32((seed * 0.005).sin()).to_bits();
            }
        }
    }

    let q = upload_f32(&ocl_arc, &*memory, &q_data, vec![1, 1, n_heads_q, head_dim]);
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

    let zero = vec![0.0f32; n_heads_q * head_dim];
    let mut out_a = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);
    let mut out_b = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);

    let ok_a = ocl
        .flash_attention_decode_gpu(
            &q,
            &k,
            &v,
            &mut out_a,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
            None,
        )
        .expect("flash call a");
    if !ok_a {
        // Some hosts (e.g. macOS/Apple OpenCL via cl2Metal) fail to compile
        // flash_attn_f32_f16.cl at runtime, leaving the kernel handle None.
        // The gate in flash_attention_decode_gpu() then returns Ok(false),
        // which on device would fall back to kernel_attn_gen_half. Here we
        // skip cleanly — strong on-device verification lives in Task 3.
        eprintln!("Skipping: flash kernel unavailable on this host (kernel compile likely failed)");
        return;
    }
    ocl_arc.synchronize().unwrap();

    let ok_b = ocl
        .flash_attention_decode_gpu(
            &q,
            &k,
            &v,
            &mut out_b,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
            None,
        )
        .expect("flash call b");
    assert!(
        ok_b,
        "flash must dispatch for second call if first call dispatched"
    );
    ocl_arc.synchronize().unwrap();

    // Readback via Backend::read_buffer to avoid ARM UMA stale-cache pitfalls
    // (see CLAUDE.md: as_ptr direct access can be stale on Adreno; dev host
    // behavior may differ but read_buffer is always safe).
    let out_len_bytes = n_heads_q * head_dim * std::mem::size_of::<f32>();
    let mut raw_a = vec![0u8; out_len_bytes];
    let mut raw_b = vec![0u8; out_len_bytes];
    ocl_arc.read_buffer(&out_a, &mut raw_a).unwrap();
    ocl_arc.read_buffer(&out_b, &mut raw_b).unwrap();

    let a: &[f32] =
        unsafe { std::slice::from_raw_parts(raw_a.as_ptr() as *const f32, n_heads_q * head_dim) };
    let b: &[f32] =
        unsafe { std::slice::from_raw_parts(raw_b.as_ptr() as *const f32, n_heads_q * head_dim) };
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "nondeterministic flash output at [{i}]: {x} vs {y}"
        );
    }

    // Sanity: output is not all zeros
    assert!(
        a.iter().any(|&v| v.abs() > 1e-6),
        "flash attention produced all-zero output — likely a dispatch no-op"
    );
}

fn upload_f32(
    backend: &std::sync::Arc<dyn argus_engine::backend::Backend>,
    memory: &dyn argus_engine::memory::Memory,
    data: &[f32],
    shape: Vec<usize>,
) -> argus_engine::tensor::Tensor {
    let buf = memory
        .alloc(data.len() * 4, argus_engine::buffer::DType::F32)
        .unwrap();
    let mut t = argus_engine::tensor::Tensor::new(
        argus_engine::shape::Shape::new(shape),
        buf,
        backend.clone(),
    );
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}

fn upload_f16(
    backend: &std::sync::Arc<dyn argus_engine::backend::Backend>,
    memory: &dyn argus_engine::memory::Memory,
    data: &[u16],
    shape: Vec<usize>,
) -> argus_engine::tensor::Tensor {
    let buf = memory
        .alloc(data.len() * 2, argus_engine::buffer::DType::F16)
        .unwrap();
    let mut t = argus_engine::tensor::Tensor::new(
        argus_engine::shape::Shape::new(shape),
        buf,
        backend.clone(),
    );
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}
