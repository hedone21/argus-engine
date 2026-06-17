use argus_engine::backend::Backend;
#[cfg(target_arch = "x86_64")]
use argus_engine::backend::cpu::CpuBackendAVX2;
use argus_engine::backend::cpu::{CpuBackend, CpuBackendCommon};
#[cfg(feature = "opencl")]
use argus_engine::backend::opencl::OpenCLBackend;
#[cfg(feature = "opencl")]
use argus_engine::backend::opencl::get_cl_mem;
use argus_engine::buffer::{Buffer, DType};
use argus_engine::kv::kivi_cache::KiviCache;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::quant::{BlockQ4_0, BlockQ4_1, QK4_0, QK4_1};
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use clap::Parser;
use half::f16;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "auto,scalar", value_delimiter = ',')]
    backends: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum OpType {
    MatMulTransposed,
    MatMulSlice,
    Softmax,
    RMSNorm,
    RoPE,
    KiviAttention,
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
struct TestResult {
    op: OpType,
    status: String,
    dtype: String,
    shape: String,
    backend: String,
    duration: Duration,
    error: f32,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("=== LLM RS2 Backend Test Suite (Expanded) ===");

    let mut backends: Vec<Arc<dyn Backend>> = Vec::new();
    // Phase α-W-4 §3.3: OpenCL backend 의 KIVI native attention handle (R4: KIVI
    // test 가 new_gpu 에 넘길 때 backend 와 동일 ocl 인스턴스의 clone).
    #[cfg(feature = "opencl")]
    let mut opencl_kivi_handle: Option<
        Arc<dyn argus_engine::capability::kivi_attention::QuantAttnBackend>,
    > = None;
    for name in &args.backends {
        match name.to_lowercase().as_str() {
            "auto" | "cpu" => backends.push(Arc::new(CpuBackend::new())),
            "scalar" | "common" => backends.push(Arc::new(CpuBackendCommon::new())),
            "avx2" => {
                #[cfg(target_arch = "x86_64")]
                if std::is_x86_feature_detected!("avx2") {
                    backends.push(Arc::new(CpuBackendAVX2::new()))
                } else {
                    println!("Warning: AVX2 requested but not supported or not x86_64. Skipping.");
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    println!("Warning: AVX2 requested but not x86_64. Skipping.");
                }
            }
            "opencl" | "gpu" => {
                #[cfg(feature = "opencl")]
                match OpenCLBackend::new() {
                    Ok(b) => {
                        // Phase α-W-4 §3.3: KIVI native attention handle 을 동일 ocl
                        // 인스턴스에서 파생해 보관 (R4: new_gpu 의 backend 인자와 같은
                        // 객체). KIVI test 루프(아래)가 이 handle 을 new_gpu 에 전달.
                        let b = Arc::new(b);
                        opencl_kivi_handle = Some(b.clone()
                            as Arc<dyn argus_engine::capability::kivi_attention::QuantAttnBackend>);
                        backends.push(b as Arc<dyn Backend>);
                    }
                    Err(e) => println!(
                        "Warning: Failed to initialize OpenCL backend: {}. Skipping.",
                        e
                    ),
                }
                #[cfg(not(feature = "opencl"))]
                {
                    println!("Warning: OpenCL not enabled in this build. Skipping.");
                }
            }
            _ => println!(
                "Warning: Unknown backend '{}'. Usage: auto, scalar, avx2, opencl",
                name
            ),
        }
    }

    if backends.is_empty() {
        println!("No valid backends selected. Defaulting to Auto (CPU) and Scalar.");
        backends.push(Arc::new(CpuBackend::new()));
        backends.push(Arc::new(CpuBackendCommon::new()));
    }

    let memory = Galloc::new();
    let mut all_results = Vec::new();

    let shapes = vec![
        // Small test shapes first for OpenCL debugging
        (1, 64, 32),
        (4, 128, 64),
        (1, 256, 128),
        (4, 256, 128),
        (8, 256, 128),
        (32, 256, 128),
        (64, 256, 128),
        // Llama 7B Realistic Benchmarks (Hidden Size = 4096)
        (1, 4096, 4096), // Single token - works reliably
                         // (32, 4096, 4096), // Disabled - needs separate test run due to OOM with many shapes
    ];

    for backend in &backends {
        println!("\n--- Running Tests for Backend: {} ---", backend.name());

        for (m, k, n) in &shapes {
            let (m, k, n) = (*m, *k, *n);

            // Skip MatMul Standard
            // if k < 512 { ... }

            // Test MatMulTransposed - F32
            run_matmul_test(
                &mut all_results,
                backend.clone(),
                &memory,
                OpType::MatMulTransposed,
                DType::F32,
                m,
                k,
                n,
            );

            if k % 32 == 0 {
                run_matmul_test(
                    &mut all_results,
                    backend.clone(),
                    &memory,
                    OpType::MatMulTransposed,
                    DType::Q4_0,
                    m,
                    k,
                    n,
                );
                // run_matmul_test(&mut all_results, backend.clone(), &memory, OpType::MatMulTransposed, DType::Q4_1, m, k, n);
            }
            run_matmul_test(
                &mut all_results,
                backend.clone(),
                &memory,
                OpType::MatMulSlice,
                DType::F32,
                m,
                k,
                n,
            );

            // Additional Ops Benchmarks - run on all shapes for debugging
            if k >= 64 {
                run_matmul_test(
                    &mut all_results,
                    backend.clone(),
                    &memory,
                    OpType::RMSNorm,
                    DType::F32,
                    m,
                    k,
                    n,
                );
                run_matmul_test(
                    &mut all_results,
                    backend.clone(),
                    &memory,
                    OpType::Softmax,
                    DType::F32,
                    m,
                    k,
                    n,
                );
                run_matmul_test(
                    &mut all_results,
                    backend.clone(),
                    &memory,
                    OpType::RoPE,
                    DType::F32,
                    m,
                    k,
                    n,
                );
            }
        }
    }

    // KIVI Attention oracle tests (GPU-only)
    for backend in &backends {
        if backend.name() != "OpenCL" {
            continue;
        }
        println!(
            "\n--- Running KIVI Attention Oracle Tests for Backend: {} ---",
            backend.name()
        );
        for bits in [2u8, 4, 8] {
            run_kivi_attention_test(
                &mut all_results,
                backend.clone(),
                opencl_kivi_handle.clone(),
                bits,
            );
        }
    }

    print_comparison_table(&all_results, &backends);

    Ok(())
}

fn print_comparison_table(results: &[TestResult], backends: &[Arc<dyn Backend>]) {
    #[derive(PartialEq, Eq, Hash, Ord, PartialOrd)]
    struct RowKey {
        op: OpType,
        shape: String,
        dtype: String,
    }

    let mut map: BTreeMap<RowKey, BTreeMap<String, (Duration, f32, String)>> = BTreeMap::new();

    for res in results {
        let key = RowKey {
            op: res.op,
            shape: res.shape.clone(),
            dtype: res.dtype.clone(),
        };

        map.entry(key).or_default().insert(
            res.backend.clone(),
            (res.duration, res.error, res.status.clone()),
        );
    }

    println!("\n{:=<200}", "");
    print!("{:<18} | {:<16} | {:<6} |", "Op", "Shape", "DTy");
    for backend in backends {
        let short_name = backend
            .name()
            .replace("CPU (", "")
            .replace(")", "")
            .replace("Auto - ", "");
        print!(" {:^32} |", short_name);
    }
    println!();

    print!("{:<18} | {:<16} | {:<6} |", "", "", "");
    for _ in backends {
        print!(" {:<18} | {:<11} |", "Duration", "Error");
    }
    println!("\n{:-<200}", "");

    for (key, backend_map) in map {
        print!(
            "{:<18} | {:<16} | {:<6} |",
            key.op.to_string(),
            key.shape,
            key.dtype
        );

        let ref_dur_secs = backend_map
            .get("CPU (Scalar)")
            .map(|(d, _, _)| d.as_secs_f64());

        for backend in backends {
            let b_name = backend.name();
            if let Some((dur, err, status)) = backend_map.get(b_name) {
                if status == "PASS" {
                    let dur_secs = dur.as_secs_f64();
                    let dur_ms = dur_secs * 1000.0;

                    let speedup_str = if let Some(ref_secs) = ref_dur_secs {
                        if dur_secs > 0.0 {
                            format!("(x{:.2})", ref_secs / dur_secs)
                        } else {
                            "(x0.00)".to_string()
                        }
                    } else {
                        "".to_string()
                    };

                    let dur_display = format!("{:.2}ms {}", dur_ms, speedup_str);
                    print!(" {:>18} | {:<11.6} |", dur_display, err);
                } else {
                    print!(" {:^32} |", status);
                }
            } else {
                print!(" {:^32} |", "N/A");
            }
        }
        println!();
    }
    println!("{:=<200}", "");
}

#[allow(clippy::too_many_arguments)]
fn run_matmul_test(
    results: &mut Vec<TestResult>,
    backend: Arc<dyn Backend>,
    memory: &Galloc,
    op: OpType,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
) {
    let shape_str = format!("[{}, {}, {}]", m, k, n);
    match perform_matmul_test(backend.clone(), memory, op, dtype, m, k, n) {
        Ok((diff, dur)) => {
            // Softmax outputs are normalized to [0,1], so a single accumulation
            // tolerance applies regardless of n; F32 exp + reduction-order skew
            // for n<=4096 stays well under 2e-3. MatMul-family compares a raw
            // A·B dot product whose magnitude grows with k, so it keeps the
            // looser 1e-2 bound.
            let tol = if op == OpType::Softmax { 2e-3 } else { 1e-2 };
            let status = if diff > tol { "FAIL" } else { "PASS" };
            results.push(TestResult {
                op,
                status: status.to_string(),
                dtype: format!("{:?}", dtype),
                shape: shape_str,
                backend: backend.name().to_string(),
                duration: dur,
                error: diff,
            });
        }
        Err(e) => {
            eprintln!("ERROR in {} {:?} {}: {}", op, dtype, shape_str, e);
            results.push(TestResult {
                op,
                status: "ERROR".to_string(),
                dtype: format!("{:?}", dtype),
                shape: shape_str,
                backend: backend.name().to_string(),
                duration: Duration::from_secs(0),
                error: -1.0,
            });
        }
    }
}

fn perform_matmul_test(
    backend: Arc<dyn Backend>,
    memory: &Galloc,
    op: OpType,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
) -> anyhow::Result<(f32, Duration)> {
    // A: [M, K]
    let a_size = m * k * 4;
    let buf_a = memory.alloc(a_size, DType::F32)?;
    let mut a_vec = vec![0.0f32; m * k];
    for (i, v) in a_vec.iter_mut().enumerate() {
        *v = ((i % 100) as f32 * 0.01) - 0.5;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(a_vec.as_ptr(), buf_a.as_mut_ptr() as *mut f32, a_vec.len());
    }
    let a = Tensor::new(Shape::new(vec![m, k]), buf_a, backend.clone());

    let mut b_vec_f32 = vec![0.0f32; n * k];
    for (i, v) in b_vec_f32.iter_mut().enumerate() {
        *v = ((i % 123) as f32 * 0.01) - 0.5;
    }

    // Helpers to hold blocks for verification if needed
    let mut q4_0_blocks: Vec<BlockQ4_0> = Vec::new();
    let mut q4_1_blocks: Vec<BlockQ4_1> = Vec::new();

    // Prepare B Tensor
    let b = match (op, dtype) {
        (OpType::MatMulTransposed, DType::F32) => {
            let buf_b = memory.alloc(n * k * 4, DType::F32)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b_vec_f32.as_ptr(),
                    buf_b.as_mut_ptr() as *mut f32,
                    n * k,
                );
            }
            Tensor::new(Shape::new(vec![n, k]), buf_b, backend.clone())
        }
        (OpType::MatMulTransposed, DType::Q4_0) => {
            q4_0_blocks = quantize_q4_0(&b_vec_f32, n, k);
            let b_size_bytes = q4_0_blocks.len() * std::mem::size_of::<BlockQ4_0>();
            let buf_b = memory.alloc(b_size_bytes, DType::Q4_0)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    q4_0_blocks.as_ptr(),
                    buf_b.as_mut_ptr() as *mut BlockQ4_0,
                    q4_0_blocks.len(),
                );
            }
            Tensor::new(Shape::new(vec![n, k]), buf_b, backend.clone())
        }
        (OpType::MatMulTransposed, DType::Q4_1) => {
            q4_1_blocks = quantize_q4_1(&b_vec_f32, n, k);
            let b_size_bytes = q4_1_blocks.len() * std::mem::size_of::<BlockQ4_1>();
            let buf_b = memory.alloc(b_size_bytes, DType::Q4_1)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    q4_1_blocks.as_ptr(),
                    buf_b.as_mut_ptr() as *mut BlockQ4_1,
                    q4_1_blocks.len(),
                );
            }
            Tensor::new(Shape::new(vec![n, k]), buf_b, backend.clone())
        }
        (OpType::MatMulSlice, DType::F32) => {
            // Just allocate dummy B, but must be 2D [N, K] for matmul_transposed to accept it.
            let buf_b = memory.alloc(n * k * 4, DType::F32)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b_vec_f32.as_ptr(),
                    buf_b.as_mut_ptr() as *mut f32,
                    n * k,
                );
            }
            Tensor::new(Shape::new(vec![n, k]), buf_b, backend.clone())
        }
        (OpType::Softmax, _) => {
            // B unused
            let buf_b = memory.alloc(4, DType::F32)?;
            Tensor::new(Shape::new(vec![1]), buf_b, backend.clone())
        }
        (OpType::RMSNorm, _) | (OpType::RoPE, _) => {
            // RMSNorm: W is [Dim] = [K].
            // RoPE: unused but technically we might pass something?
            // Safest to alloc K elements to avoid range checks failing if B is used.
            let buf_b = memory.alloc(k * 4, DType::F32)?;
            // Fill with 1.0
            let vec_b = vec![1.0f32; k];
            unsafe {
                std::ptr::copy_nonoverlapping(vec_b.as_ptr(), buf_b.as_mut_ptr() as *mut f32, k);
            }
            Tensor::new(Shape::new(vec![k]), buf_b, backend.clone())
        }
        _ => return Err(anyhow::anyhow!("Unsupported config")),
    };

    let buf_c = memory.alloc(m * n * 4, DType::F32)?;
    // Softmax is an in-place op on C: seed C with a deterministic input so the
    // reference softmax below is reproducible. Other ops use C as output-only,
    // so they leave it uninitialized (their reference compares the computed
    // value against an independently derived A·B sum).
    let c_input_vec: Vec<f32> = if op == OpType::Softmax {
        let v: Vec<f32> = (0..m * n).map(|i| ((i % 97) as f32 * 0.05) - 2.0).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), buf_c.as_mut_ptr() as *mut f32, v.len());
        }
        v
    } else {
        Vec::new()
    };
    let c = Tensor::new(Shape::new(vec![m, n]), buf_c, backend.clone());

    // For OpenCL backend, we need to transfer tensors to GPU
    let is_opencl = backend.name() == "OpenCL";
    let (a_gpu, b_gpu, mut c_gpu) = if is_opencl {
        (
            backend.copy_from(&a)?,
            backend.copy_from(&b)?,
            backend.copy_from(&c)?,
        )
    } else {
        (a.clone(), b.clone(), c)
    };

    // Keep original A for verification (a still exists)

    let start = Instant::now();

    let iterations = 10; // Reduced from 50 to prevent OOM on large tensors
    for _ in 0..iterations {
        match op {
            OpType::MatMulTransposed => backend.matmul_transposed(&a_gpu, &b_gpu, &mut c_gpu)?,
            OpType::MatMulSlice => backend.matmul_slice(&a_gpu, &b_gpu, m, n, &mut c_gpu)?,
            OpType::RMSNorm => backend.rms_norm(&mut c_gpu, &b_gpu, 1e-5, false)?,
            OpType::Softmax => backend.softmax(&mut c_gpu)?,
            OpType::RoPE => {
                // RoPE requires a (Seq, Head, Dim) or (Batch, Seq, Head, Dim)
                // tensor with an even head_dim. The 2D [m, n] activation `c_gpu`
                // never satisfies that contract, so we always RoPE on a freshly
                // shaped 4D tensor. When n is a multiple of 128 we use the
                // production head_dim; otherwise we fall back to a single head
                // of width n (always even for the test shapes), which keeps the
                // op exercising a valid 4D layout instead of erroring on a 2D
                // input. (No reference comparison is done for RoPE — this only
                // verifies the op runs without a dimensionality error.)
                let head_dim = if n.is_multiple_of(128) { 128 } else { n };
                let num_heads = n / head_dim;
                let r_shape = Shape::new(vec![1, m, num_heads, head_dim]);
                let r_buf = memory.alloc(m * n * 4, DType::F32)?;
                let r_tensor = Tensor::new(r_shape, r_buf, backend.clone());
                let mut r_gpu = if is_opencl {
                    backend.copy_from(&r_tensor)?
                } else {
                    r_tensor
                };
                backend.rope_inplace(&mut r_gpu, 0, 10000.0)?;
            }
            OpType::KiviAttention => {
                // KIVI tests are handled by run_kivi_attention_test(), not here
                return Err(anyhow::anyhow!(
                    "KiviAttention not handled in perform_matmul_test"
                ));
            }
        }
        backend.synchronize()?;
    }
    let dur = start.elapsed();

    // Softmax is not idempotent: the timing loop above ran it `iterations`
    // times in place, so c_gpu now holds softmax^N(input), which does not match
    // the single-pass reference. Re-seed a fresh input and run exactly one pass
    // for the correctness check (timing is already captured in `dur`).
    if op == OpType::Softmax {
        let buf_clean = memory.alloc(m * n * 4, DType::F32)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_input_vec.as_ptr(),
                buf_clean.as_mut_ptr() as *mut f32,
                c_input_vec.len(),
            );
        }
        let clean = Tensor::new(Shape::new(vec![m, n]), buf_clean, backend.clone());
        c_gpu = if is_opencl {
            backend.copy_from(&clean)?
        } else {
            clean
        };
        backend.softmax(&mut c_gpu)?;
        backend.synchronize()?;
    }

    // Verify - read back results from GPU if OpenCL
    let c_data: Vec<f32> = if is_opencl {
        #[cfg(feature = "opencl")]
        {
            let buf = c_gpu.buffer();
            // Production OpenCL path uses UnifiedBuffer (CL_MEM_ALLOC_HOST_PTR);
            // the legacy OpenCLBuffer only appears on some host drivers. Read
            // via blocking clEnqueueReadBuffer for both — on ARM UMA the mapped
            // host pointer is cache-incoherent, so an explicit read is required
            // to avoid stale data (CLAUDE.md zero-copy note).
            let mut data = vec![0u8; m * n * 4];
            let read_ok = if let Some(unified) =
                buf.as_any()
                    .downcast_ref::<argus_engine::memory::opencl::unified::UnifiedBuffer>()
            {
                unified.cl_buffer().read(&mut data).enq()?;
                true
            } else if let Some(cl_buf) =
                buf.as_any()
                    .downcast_ref::<argus_engine::memory::opencl::device::OpenCLBuffer>()
            {
                cl_buf.buffer.read(&mut data).enq()?;
                true
            } else {
                false
            };
            if read_ok {
                let mut result = vec![0.0f32; m * n];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        result.as_mut_ptr() as *mut u8,
                        m * n * 4,
                    );
                }
                result
            } else {
                return Err(anyhow::anyhow!(
                    "test_backend: unsupported OpenCL buffer type for readback"
                ));
            }
        }
        #[cfg(not(feature = "opencl"))]
        {
            vec![0.0f32; m * n]
        }
    } else {
        c_gpu.as_slice::<f32>().to_vec()
    };

    let r_m = m / 2;
    let r_n = n / 2;
    let mut ref_sum = 0.0;

    match (op, dtype) {
        (OpType::MatMulTransposed, DType::F32) => {
            for idx_k in 0..k {
                ref_sum += a_vec[r_m * k + idx_k] * b_vec_f32[r_n * k + idx_k];
            }
        }
        (OpType::MatMulTransposed, DType::Q4_0) => {
            // Dequantize B row r_n
            let nb_k = k / QK4_0;
            let row_blocks = &q4_0_blocks[r_n * nb_k..(r_n + 1) * nb_k];
            let mut deq_buf = vec![0.0f32; k];
            for (i, block) in row_blocks.iter().enumerate() {
                let mut temp = [0.0f32; 32];
                block.dequantize(&mut temp);
                deq_buf[i * 32..(i + 1) * 32].copy_from_slice(&temp);
            }
            for idx_k in 0..k {
                ref_sum += a_vec[r_m * k + idx_k] * deq_buf[idx_k];
            }
        }
        (OpType::MatMulTransposed, DType::Q4_1) => {
            let nb_k = k / QK4_1;
            let row_blocks = &q4_1_blocks[r_n * nb_k..(r_n + 1) * nb_k];
            let mut deq_buf = vec![0.0f32; k];
            for (i, block) in row_blocks.iter().enumerate() {
                let mut temp = [0.0f32; 32];
                block.dequantize(&mut temp);
                deq_buf[i * 32..(i + 1) * 32].copy_from_slice(&temp);
            }
            for idx_k in 0..k {
                ref_sum += a_vec[r_m * k + idx_k] * deq_buf[idx_k];
            }
        }
        (OpType::MatMulSlice, DType::F32) => {
            // Uses same layout as MatMulTransposed logic in our implementation
            for idx_k in 0..k {
                ref_sum += a_vec[r_m * k + idx_k] * b_vec_f32[r_n * k + idx_k];
            }
        }
        (OpType::Softmax, _) => {
            // Softmax operates row-wise over the last dim (length n). Reference
            // for cell [r_m, r_n] = numerically-stable softmax of input row r_m.
            let row = &c_input_vec[r_m * n..(r_m + 1) * n];
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
            ref_sum = (row[r_n] - max_val).exp() / sum_exp;
        }
        _ => {}
    }

    let val = c_data[r_m * n + r_n];
    let diff = (ref_sum - val).abs();
    Ok((diff, dur))
}

fn quantize_q4_0(data: &[f32], n: usize, k: usize) -> Vec<BlockQ4_0> {
    let nb_k = k / QK4_0;
    let mut blocks = Vec::with_capacity(n * nb_k);

    for j in 0..n {
        for bi in 0..nb_k {
            let offset = j * k + bi * QK4_0;
            let src = &data[offset..offset + QK4_0];
            let mut block = BlockQ4_0 {
                d: f16::from_f32(0.0),
                qs: [0; 16],
            };

            let max_val = src.iter().map(|v| v.abs()).fold(0.0f32, |x, y| x.max(y));
            let d = max_val / 7.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };

            block.d = f16::from_f32(d);
            for z in 0..16 {
                let v0 = (src[z] * id).round().clamp(-8.0, 7.0) as i8;
                let v1 = (src[z + 16] * id).round().clamp(-8.0, 7.0) as i8;
                let b0 = (v0 + 8) as u8;
                let b1 = (v1 + 8) as u8;
                block.qs[z] = b0 | (b1 << 4);
            }
            blocks.push(block);
        }
    }
    blocks
}

fn quantize_q4_1(data: &[f32], n: usize, k: usize) -> Vec<BlockQ4_1> {
    let nb_k = k / QK4_1;
    let mut blocks = Vec::with_capacity(n * nb_k);

    for j in 0..n {
        for bi in 0..nb_k {
            let offset = j * k + bi * QK4_1;
            let src = &data[offset..offset + QK4_1];
            let mut block = BlockQ4_1 {
                d: f16::from_f32(0.0),
                m: f16::from_f32(0.0),
                qs: [0; 16],
            };

            let min_val = src.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = src.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let d = (max_val - min_val) / 15.0;
            let m_val = min_val;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };

            block.d = f16::from_f32(d);
            block.m = f16::from_f32(m_val);

            for z in 0..16 {
                let v0 = ((src[z] - m_val) * id).round().clamp(0.0, 15.0) as u8;
                let v1 = ((src[z + 16] - m_val) * id).round().clamp(0.0, 15.0) as u8;
                block.qs[z] = v0 | (v1 << 4);
            }
            blocks.push(block);
        }
    }
    blocks
}

/// Simple deterministic PRNG for reproducible test data (xorshift32).
fn xorshift32(state: &mut u32) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    // Map to [-0.5, 0.5)
    (x as f32 / u32::MAX as f32) - 0.5
}

/// Run a KIVI attention oracle test for a specific bit-width.
///
/// Creates identical CPU and GPU KiviCaches, fills with the same data
/// (enough tokens to trigger multiple flushes), then compares:
///   CPU: get_view() -> naive single-token attention
///   GPU: attention_gen_kivi() native kernel
fn run_kivi_attention_test(
    results: &mut Vec<TestResult>,
    backend: Arc<dyn Backend>,
    kivi: Option<Arc<dyn argus_engine::capability::kivi_attention::QuantAttnBackend>>,
    bits: u8,
) {
    let kv_heads: usize = 8;
    let head_dim: usize = 64;
    let n_heads_q: usize = 32; // GQA ratio = 4
    let residual_size: usize = 32; // GROUP_SIZE = QKKV
    let test_tokens: usize = 128; // 4 flushes
    let max_seq_len: usize = 256;

    let shape_str = format!("Q{}b h{} t{}", bits, kv_heads, test_tokens);

    match perform_kivi_attention_test(
        backend.clone(),
        kivi,
        bits,
        kv_heads,
        head_dim,
        n_heads_q,
        residual_size,
        test_tokens,
        max_seq_len,
    ) {
        Ok((error, dur)) => {
            // Q2 has much higher quantization error, use relaxed threshold
            let threshold = match bits {
                2 => 0.15,
                4 => 0.08,
                8 => 0.05,
                _ => 0.05,
            };
            let status = if error > threshold { "FAIL" } else { "PASS" };
            println!(
                "  KIVI Q{} attention: {} (L2 error = {:.6}, threshold = {:.3})",
                bits, status, error, threshold
            );
            results.push(TestResult {
                op: OpType::KiviAttention,
                status: status.to_string(),
                dtype: format!("Q{}", bits),
                shape: shape_str,
                backend: backend.name().to_string(),
                duration: dur,
                error,
            });
        }
        Err(e) => {
            eprintln!("  KIVI Q{} attention: ERROR — {}", bits, e);
            results.push(TestResult {
                op: OpType::KiviAttention,
                status: "ERROR".to_string(),
                dtype: format!("Q{}", bits),
                shape: shape_str,
                backend: backend.name().to_string(),
                duration: Duration::from_secs(0),
                error: -1.0,
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn perform_kivi_attention_test(
    backend: Arc<dyn Backend>,
    kivi: Option<Arc<dyn argus_engine::capability::kivi_attention::QuantAttnBackend>>,
    bits: u8,
    kv_heads: usize,
    head_dim: usize,
    n_heads_q: usize,
    residual_size: usize,
    test_tokens: usize,
    max_seq_len: usize,
) -> anyhow::Result<(f32, Duration)> {
    #[cfg(not(feature = "opencl"))]
    {
        let _ = (
            backend,
            kivi,
            bits,
            kv_heads,
            head_dim,
            n_heads_q,
            residual_size,
            test_tokens,
            max_seq_len,
        );
        anyhow::bail!("OpenCL feature not enabled");
    }

    #[cfg(feature = "opencl")]
    {
        let ocl = backend
            .as_any()
            .downcast_ref::<OpenCLBackend>()
            .ok_or_else(|| anyhow::anyhow!("Backend is not OpenCL"))?;

        // Check if kernel is available
        if !ocl.has_kivi_attn_kernel(bits) {
            anyhow::bail!(
                "KIVI Q{} attention kernel not available on this device",
                bits
            );
        }

        // Create GPU memory allocator
        let gpu_memory: Arc<dyn Memory> =
            Arc::new(argus_engine::backend::opencl::memory::OpenCLMemory::new(
                ocl.context.clone(),
                ocl.queue.clone(),
                false, // device-only buffers
            ));

        // === Create CPU and GPU caches ===
        let mut cpu_cache =
            KiviCache::new_with_bits(kv_heads, head_dim, max_seq_len, residual_size, bits);
        let mut gpu_cache = KiviCache::new_gpu(
            kv_heads,
            head_dim,
            max_seq_len,
            residual_size,
            bits,
            backend.clone(),
            kivi,
            gpu_memory.clone(),
        );
        assert!(gpu_cache.is_gpu(), "GPU cache creation failed");

        // === Generate deterministic test data and fill both caches ===
        let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mut rng_state: u32 = 42;

        for t in 0..test_tokens {
            // Generate one token of K/V data: [1, 1, kv_heads, head_dim]
            let n_elems = kv_heads * head_dim;
            let mut k_data = vec![0.0f32; n_elems];
            let mut v_data = vec![0.0f32; n_elems];
            for i in 0..n_elems {
                k_data[i] = xorshift32(&mut rng_state) * 0.5;
                v_data[i] = xorshift32(&mut rng_state) * 0.5;
            }

            // CPU tensor (for CPU cache update)
            let cpu_k_buf = Arc::new(argus_engine::memory::host::shared::SharedBuffer::new(
                n_elems * 4,
                DType::F32,
            ));
            let cpu_v_buf = Arc::new(argus_engine::memory::host::shared::SharedBuffer::new(
                n_elems * 4,
                DType::F32,
            ));
            unsafe {
                std::ptr::copy_nonoverlapping(
                    k_data.as_ptr(),
                    cpu_k_buf.as_mut_ptr() as *mut f32,
                    n_elems,
                );
                std::ptr::copy_nonoverlapping(
                    v_data.as_ptr(),
                    cpu_v_buf.as_mut_ptr() as *mut f32,
                    n_elems,
                );
            }
            let cpu_k = Tensor::new(
                Shape::new(vec![1, 1, kv_heads, head_dim]),
                cpu_k_buf,
                cpu_backend.clone(),
            );
            let cpu_v = Tensor::new(
                Shape::new(vec![1, 1, kv_heads, head_dim]),
                cpu_v_buf,
                cpu_backend.clone(),
            );
            cpu_cache.update(&cpu_k, &cpu_v)?;

            // GPU tensor (for GPU cache update)
            let gpu_k_buf = gpu_memory.alloc(n_elems * 4, DType::F32)?;
            let gpu_v_buf = gpu_memory.alloc(n_elems * 4, DType::F32)?;
            let mut gpu_k_t = Tensor::new(
                Shape::new(vec![1, 1, kv_heads, head_dim]),
                gpu_k_buf,
                backend.clone(),
            );
            let mut gpu_v_t = Tensor::new(
                Shape::new(vec![1, 1, kv_heads, head_dim]),
                gpu_v_buf,
                backend.clone(),
            );
            // Write data to GPU buffers
            backend.write_buffer(&mut gpu_k_t, unsafe {
                std::slice::from_raw_parts(k_data.as_ptr() as *const u8, n_elems * 4)
            })?;
            backend.write_buffer(&mut gpu_v_t, unsafe {
                std::slice::from_raw_parts(v_data.as_ptr() as *const u8, n_elems * 4)
            })?;
            gpu_cache.update(&gpu_k_t, &gpu_v_t)?;

            // Note: both CPU and GPU caches auto-flush during update() when residual is full.
            // Verify positions are in sync
            if t < 5 || t == test_tokens - 1 {
                assert_eq!(
                    cpu_cache.current_pos(),
                    gpu_cache.current_pos(),
                    "Position mismatch at token {}: cpu={}, gpu={}",
                    t,
                    cpu_cache.current_pos(),
                    gpu_cache.current_pos()
                );
            }
        }

        let total_tokens = cpu_cache.current_pos();
        let q_tokens = cpu_cache.q2_tokens;
        let res_tokens = cpu_cache.res_pos;

        println!(
            "  KIVI Q{}: total={}, q_tokens={}, res_tokens={}",
            bits, total_tokens, q_tokens, res_tokens
        );

        // === Generate random Q vector: [n_heads_q, head_dim] ===
        let q_elems = n_heads_q * head_dim;
        let mut q_data = vec![0.0f32; q_elems];
        for v in q_data.iter_mut() {
            *v = xorshift32(&mut rng_state) * 0.3;
        }

        // === CPU reference: get_view() -> naive single-token attention ===
        let (k_view, v_view) = cpu_cache.get_view();
        // k_view/v_view shape: [1, total_tokens, kv_heads, head_dim] (SeqMajor)
        let k_cpu = k_view.as_slice::<f32>();
        let v_cpu = v_view.as_slice::<f32>();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = n_heads_q / kv_heads;
        let mut cpu_out = vec![0.0f32; q_elems];

        for qh in 0..n_heads_q {
            let kv_h = qh / gqa_ratio;
            let q_slice = &q_data[qh * head_dim..(qh + 1) * head_dim];

            // Compute attention scores: Q * K^T / sqrt(d) for each token
            let mut scores = vec![0.0f32; total_tokens];
            for t in 0..total_tokens {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    // SeqMajor layout: [t * kv_heads * head_dim + kv_h * head_dim + d]
                    let k_val = k_cpu[t * kv_heads * head_dim + kv_h * head_dim + d];
                    dot += q_slice[d] * k_val;
                }
                scores[t] = dot * scale;
            }

            // Softmax
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }

            // Weighted sum: scores * V -> out
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..total_tokens {
                    let v_val = v_cpu[t * kv_heads * head_dim + kv_h * head_dim + d];
                    val += scores[t] * v_val;
                }
                cpu_out[qh * head_dim + d] = val;
            }
        }

        // === GPU: attention_gen_kivi() ===
        let gpu_raw = gpu_cache
            .get_raw_gpu_buffers()
            .ok_or_else(|| anyhow::anyhow!("No GPU raw buffers (q_tokens=0?)"))?;

        // Upload Q to GPU: shape [1, n_heads_q, head_dim] (single token decode)
        let q_gpu_buf = gpu_memory.alloc(q_elems * 4, DType::F32)?;
        let mut q_gpu = Tensor::new(
            Shape::new(vec![1, n_heads_q, head_dim]),
            q_gpu_buf,
            backend.clone(),
        );
        backend.write_buffer(&mut q_gpu, unsafe {
            std::slice::from_raw_parts(q_data.as_ptr() as *const u8, q_elems * 4)
        })?;

        // Output buffer
        let out_gpu_buf = gpu_memory.alloc(q_elems * 4, DType::F32)?;
        let out_gpu = Tensor::new(
            Shape::new(vec![1, n_heads_q, head_dim]),
            out_gpu_buf,
            backend.clone(),
        );

        // D8: inherent attention_gen_kivi 가 raw cl_mem(`*mut c_void`) 시그니처로
        // 바뀌어, `&Tensor` 6개에서 cl_mem 을 추출해 넘긴다. `Mem::as_ptr()` 가 이미
        // `cl_mem`(= `*mut c_void`) 라 캐스트 불요.
        let q_mem = get_cl_mem(q_gpu.buffer().as_ref())?.as_ptr();
        let qk_mem = get_cl_mem(gpu_raw.qk_buf.buffer().as_ref())?.as_ptr();
        let qv_mem = get_cl_mem(gpu_raw.qv_buf.buffer().as_ref())?.as_ptr();
        let res_k_mem = get_cl_mem(gpu_raw.res_k.buffer().as_ref())?.as_ptr();
        let res_v_mem = get_cl_mem(gpu_raw.res_v.buffer().as_ref())?.as_ptr();
        let out_mem = get_cl_mem(out_gpu.buffer().as_ref())?.as_ptr();
        let start = Instant::now();
        // SAFETY: cl_mem 6개는 위에서 live GPU 버퍼로부터 추출한 유효 핸들(borrow-only).
        unsafe {
            ocl.attention_gen_kivi(
                q_mem,
                qk_mem,
                qv_mem,
                res_k_mem,
                res_v_mem,
                out_mem,
                n_heads_q,
                kv_heads,
                head_dim,
                gpu_raw.q_tokens,
                gpu_raw.res_tokens,
                gpu_raw.res_cap,
                scale,
                None, // no scores output
                bits,
            )?;
        }
        backend.synchronize()?;
        let dur = start.elapsed();

        // Read GPU output back to CPU
        let mut gpu_out = vec![0.0f32; q_elems];
        backend.read_buffer(&out_gpu, unsafe {
            std::slice::from_raw_parts_mut(gpu_out.as_mut_ptr() as *mut u8, q_elems * 4)
        })?;

        // === Compare: L2 error (per-element RMSE) ===
        let mut sq_diff_sum = 0.0f64;
        let mut ref_sq_sum = 0.0f64;
        for i in 0..q_elems {
            let diff = (cpu_out[i] - gpu_out[i]) as f64;
            sq_diff_sum += diff * diff;
            ref_sq_sum += (cpu_out[i] as f64) * (cpu_out[i] as f64);
        }
        // Relative L2 error = sqrt(sum(diff^2)) / sqrt(sum(ref^2))
        let l2_error = if ref_sq_sum > 0.0 {
            (sq_diff_sum / ref_sq_sum).sqrt() as f32
        } else {
            (sq_diff_sum / q_elems as f64).sqrt() as f32
        };

        // Also print per-head max absolute error for diagnosis
        let mut max_abs_err: f32 = 0.0;
        for i in 0..q_elems {
            let abs_err = (cpu_out[i] - gpu_out[i]).abs();
            max_abs_err = max_abs_err.max(abs_err);
        }
        println!(
            "    rel-L2={:.6}, max-abs-err={:.6}, cpu[0..4]={:?}, gpu[0..4]={:?}",
            l2_error,
            max_abs_err,
            &cpu_out[..4.min(q_elems)],
            &gpu_out[..4.min(q_elems)],
        );

        Ok((l2_error, dur))
    }
}
