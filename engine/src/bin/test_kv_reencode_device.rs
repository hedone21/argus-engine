//! On-device verification for the W-FORMAT-HET L1-runtime GPU KV re-encode
//! (`apply_format_plan` → `reencode_typed_device`).
//!
//! Builds a **device-resident** F16 KV cache via the OpenCL backend's own KV
//! allocator (so the buffers are real `UnifiedBuffer`s, `is_gpu_buffer()==true`),
//! re-encodes it to q4_0 through the production `apply_format_plan` path (download
//! → host re-encode → upload), reads the result back, and asserts it is
//! **byte-identical** to the host (CPU) re-encode of the same data. The host
//! re-encode is the deterministic reference (`BlockQ4_0::quantize`), equal to a
//! construction-time q4_0 allocation — so byte-equality proves the device
//! download/upload round-trip + re-encode is correct on the actual Adreno GPU.
//!
//! Run on device (Adreno) via `scripts/run_device.py`. Exit 0 = byte-equal;
//! non-zero = mismatch reported on stderr.

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use argus_engine::backend::Backend;
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::backend::opencl::OpenCLBackend;
    use std::sync::Arc;

    const QK4_0: usize = 32;
    const Q4_BLOCK: usize = 18; // BlockQ4_0 byte size

    // (kv_heads, head_dim, n_tokens) — a few shapes; head_dim a multiple of QK4_0 (=32).
    let configs: [(usize, usize, usize); 4] =
        [(2, 64, 32), (4, 128, 96), (8, 128, 256), (1, 32, 7)];

    let gpu_backend: Arc<dyn Backend> = Arc::new(OpenCLBackend::new()?);
    let memory = gpu_backend
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCL backend")
        .make_kv_memory();
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    println!(
        "[test_kv_reencode_device] backend={} configs={}",
        gpu_backend.name(),
        configs.len()
    );

    let mut all_ok = true;
    for (kv_heads, head_dim, n) in configs {
        all_ok &= run_case(
            &gpu_backend,
            &memory,
            &cpu,
            kv_heads,
            head_dim,
            n,
            QK4_0,
            Q4_BLOCK,
        )?;
    }

    if all_ok {
        println!("[test_kv_reencode_device] PASS — device GPU re-encode is byte-identical to host");
        Ok(())
    } else {
        anyhow::bail!("[test_kv_reencode_device] FAIL — device re-encode bytes differ from host")
    }
}

/// One shape: build a device F16 cache, re-encode to q4_0 on the GPU path, and compare the read-back
/// bytes to the host (CPU) re-encode of the identical data. Returns `true` iff byte-equal (K and V).
#[cfg(feature = "opencl")]
#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu_backend: &std::sync::Arc<dyn argus_engine::backend::Backend>,
    memory: &std::sync::Arc<dyn argus_engine::memory::Memory>,
    cpu: &std::sync::Arc<dyn argus_engine::backend::Backend>,
    kv_heads: usize,
    head_dim: usize,
    n: usize,
    qk4_0: usize,
    q4_block: usize,
) -> anyhow::Result<bool> {
    use argus_engine::buffer::{Buffer, DType};
    use argus_engine::kv::format_apply::apply_format_plan;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;
    use argus_extension_api::{FormatId, KVFormatPlan};
    use half::f16;
    use std::sync::Arc;

    let total = n * kv_heads * head_dim;
    let f16_bytes = total * 2;
    let q4_bytes = (total / qk4_0) * q4_block;
    let shape = Shape::new(vec![1, n, kv_heads, head_dim]);

    // Known F16 source data, distinct K/V salts (so a K-into-V swap is caught).
    let pat = |i: usize, salt: f32| salt + (i % 97) as f32 * 0.013 - 0.31;
    let mut k_f16 = vec![0u8; f16_bytes];
    let mut v_f16 = vec![0u8; f16_bytes];
    {
        // SAFETY: k_f16/v_f16 are `f16_bytes == total*2` bytes → exactly `total` f16 elements.
        let kk = unsafe { std::slice::from_raw_parts_mut(k_f16.as_mut_ptr() as *mut f16, total) };
        let vv = unsafe { std::slice::from_raw_parts_mut(v_f16.as_mut_ptr() as *mut f16, total) };
        for i in 0..total {
            kk[i] = f16::from_f32(pat(i, 0.5));
            vv[i] = f16::from_f32(pat(i, -1.3));
        }
    }

    let plan = KVFormatPlan {
        base: FormatId("q4_0".into()),
        overrides: Vec::new(),
    };

    // ── GPU path: device-resident F16 cache → re-encode to q4_0 → read back. ──
    let mut gpu_cache = {
        let k_buf = memory.alloc_kv(f16_bytes, DType::F16)?;
        let v_buf = memory.alloc_kv(f16_bytes, DType::F16)?;
        let mut k_t = Tensor::new(shape.clone(), k_buf, gpu_backend.clone());
        let mut v_t = Tensor::new(shape.clone(), v_buf, gpu_backend.clone());
        gpu_backend.write_buffer(&mut k_t, &k_f16)?;
        gpu_backend.write_buffer(&mut v_t, &v_f16)?;
        let mut c = KVCache::new_dynamic(k_t, v_t, n, n, kv_heads, head_dim, Arc::clone(memory));
        c.set_current_pos(n);
        c
    };
    anyhow::ensure!(
        gpu_cache.k_buffer.buffer().is_gpu_buffer(),
        "GPU cache must be device-resident (is_gpu_buffer) — re-encode would not exercise the device path"
    );
    apply_format_plan(&mut gpu_cache, &plan, 0, 1)
        .map_err(|e| anyhow::anyhow!("device re-encode failed: {e}"))?;
    anyhow::ensure!(
        gpu_cache.kv_dtype() == DType::Q4_0,
        "device cache dtype is {:?}, expected Q4_0",
        gpu_cache.kv_dtype()
    );
    anyhow::ensure!(
        gpu_cache.current_pos() == n,
        "current_pos changed: {} != {n}",
        gpu_cache.current_pos()
    );
    let mut gpu_k_q4 = vec![0u8; q4_bytes];
    let mut gpu_v_q4 = vec![0u8; q4_bytes];
    gpu_backend.read_buffer(&gpu_cache.k_buffer, &mut gpu_k_q4)?;
    gpu_backend.read_buffer(&gpu_cache.v_buffer, &mut gpu_v_q4)?;

    // ── Host reference: same F16 data on a host cache → re-encode to q4_0. ──
    let (host_k_q4, host_v_q4) = {
        let k_buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(f16_bytes, DType::F16));
        let v_buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(f16_bytes, DType::F16));
        let mut k_t = Tensor::new(shape.clone(), k_buf, cpu.clone());
        let mut v_t = Tensor::new(shape.clone(), v_buf, cpu.clone());
        cpu.write_buffer(&mut k_t, &k_f16)?;
        cpu.write_buffer(&mut v_t, &v_f16)?;
        let mut c = KVCache::new(k_t, v_t, n);
        c.set_current_pos(n);
        apply_format_plan(&mut c, &plan, 0, 1)
            .map_err(|e| anyhow::anyhow!("host reference re-encode failed: {e}"))?;
        let mut hk = vec![0u8; q4_bytes];
        let mut hv = vec![0u8; q4_bytes];
        cpu.read_buffer(&c.k_buffer, &mut hk)?;
        cpu.read_buffer(&c.v_buffer, &mut hv)?;
        (hk, hv)
    };

    // ── Byte-equality (device re-encode == host reference == construction-time q4_0). ──
    let tag = format!("h{kv_heads}xd{head_dim}xn{n}");
    let report = |axis: &str, gpu: &[u8], host: &[u8]| -> bool {
        if gpu == host {
            return true;
        }
        let n_diff = (0..gpu.len().min(host.len()))
            .filter(|&i| gpu[i] != host[i])
            .count();
        let first: Vec<usize> = (0..gpu.len().min(host.len()))
            .filter(|&i| gpu[i] != host[i])
            .take(8)
            .collect();
        eprintln!(
            "[test_kv_reencode_device] {tag} {axis}: MISMATCH ({n_diff}/{} bytes); first offsets {first:?}",
            gpu.len()
        );
        false
    };
    let ok = report("K", &gpu_k_q4, &host_k_q4) & report("V", &gpu_v_q4, &host_v_q4);
    if ok {
        println!("[test_kv_reencode_device] {tag}: byte-equal (K+V, {q4_bytes} bytes each)");
    }
    Ok(ok)
}

#[cfg(not(feature = "opencl"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("test_kv_reencode_device requires the `opencl` feature")
}
