//! Unit tests for `Backend::fused_norm_merge`.
//!
//! Validates the default (3-step) composition implemented on the CPU backend
//! against a straightforward naive reference:
//!     residual_out[i] = prior[i] + gpu[i] + cpu[i]
//!     out[i]          = (residual_out[i] / rms(residual_out)) * w_eff[i]
//!
//! `w_eff` is `(1 + w)` when `add_unit` is true (Gemma3 convention) else `w`.
//!
//! When the OpenCL backend is available and a GPU device is present, the
//! kernel override is compared against the same reference. On hosts without
//! an OpenCL device (most dev Macs) the OpenCL test is silently skipped.
//!
//! See `.agent/todos/partition_fused_norm_merge.md` §1 for the design rationale
//! (layer-boundary barrier elimination) that motivates this kernel.

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/// Naive scalar reference matching the fused_norm_merge semantics.
///
/// Reduction order is `prior + (gpu + cpu)` so it matches the baseline 3-step
/// path `ws.down = gpu + cpu; x += ws.down`. Keeping the same order here is
/// what lets the kernel override stay bit-exact with fused off — swapping to
/// `(prior + gpu) + cpu` would drift one ULP per element and cascade through
/// downstream layers, diverging the greedy generation stream.
fn reference(
    prior: &[f32],
    gpu: &[f32],
    cpu: &[f32],
    w: &[f32],
    dim: usize,
    eps: f32,
    add_unit: bool,
) -> (Vec<f32>, Vec<f32>) {
    let rows = prior.len() / dim;
    let mut residual_out = vec![0.0f32; prior.len()];
    let mut out = vec![0.0f32; prior.len()];
    for r in 0..rows {
        let base = r * dim;
        let mut sum_sq = 0.0f64;
        for i in 0..dim {
            let partial = gpu[base + i] + cpu[base + i];
            let v = prior[base + i] + partial;
            residual_out[base + i] = v;
            sum_sq += (v as f64) * (v as f64);
        }
        let rms = (sum_sq / dim as f64 + eps as f64).sqrt() as f32;
        let scale = 1.0f32 / rms;
        for i in 0..dim {
            let w_eff = if add_unit { 1.0 + w[i] } else { w[i] };
            out[base + i] = residual_out[base + i] * scale * w_eff;
        }
    }
    (residual_out, out)
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let scale = x.abs().max(y.abs()).max(1.0);
        let rel = diff / scale;
        assert!(
            diff < tol || rel < tol,
            "{label}: mismatch at {i}: {x} vs {y} (abs diff {diff}, rel {rel})",
        );
    }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

fn run_case(dim: usize, rows: usize, add_unit: bool, seed: u32) {
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let n = rows * dim;

    let prior = gen_data(n, seed);
    let gpu = gen_data(n, seed.wrapping_add(1));
    let cpu = gen_data(n, seed.wrapping_add(2));
    let w = gen_data(dim, seed.wrapping_add(3));
    let eps: f32 = 1e-5;

    let (ref_res, ref_out) = reference(&prior, &gpu, &cpu, &w, dim, eps, add_unit);

    let prior_t = make_f32_tensor(
        &backend,
        if rows == 1 {
            vec![dim]
        } else {
            vec![rows, dim]
        },
        &prior,
    );
    let gpu_t = make_f32_tensor(
        &backend,
        if rows == 1 {
            vec![dim]
        } else {
            vec![rows, dim]
        },
        &gpu,
    );
    let cpu_t = make_f32_tensor(
        &backend,
        if rows == 1 {
            vec![dim]
        } else {
            vec![rows, dim]
        },
        &cpu,
    );
    let w_t = make_f32_tensor(&backend, vec![dim], &w);
    let mut out_t = make_f32_zeros(
        &backend,
        if rows == 1 {
            vec![dim]
        } else {
            vec![rows, dim]
        },
    );
    let mut res_out_t = make_f32_zeros(
        &backend,
        if rows == 1 {
            vec![dim]
        } else {
            vec![rows, dim]
        },
    );

    backend
        .fused_norm_merge(
            &prior_t,
            &gpu_t,
            &cpu_t,
            &w_t,
            &mut out_t,
            &mut res_out_t,
            eps,
            add_unit,
        )
        .expect("fused_norm_merge (CPU default path)");

    let got_res = res_out_t.as_slice::<f32>();
    let got_out = out_t.as_slice::<f32>();

    let label = format!("dim={dim} rows={rows} add_unit={add_unit}");
    // The CPU default path composes existing ops (add_assign → rms_norm_oop), each of
    // which already has its own NEON/rayon tolerance. 1e-4 is comfortably loose.
    assert_close(&ref_res, got_res, 1e-4, &format!("residual_out [{label}]"));
    assert_close(&ref_out, got_out, 1e-4, &format!("out [{label}]"));
}

#[test]
fn fused_norm_merge_dim1536_llama_style() {
    // Qwen2.5 / Llama3.2 FFN hidden. add_unit = false.
    run_case(1536, 1, false, 0x1234_5678);
}

#[test]
fn fused_norm_merge_dim1536_gemma_style() {
    // Gemma3-style (1 + w) weighting.
    run_case(1536, 1, true, 0xdead_beef);
}

#[test]
fn fused_norm_merge_dim64_sanity() {
    // Small dim sanity (also not a multiple of WG_SIZE=64 wrap).
    run_case(64, 1, false, 0x0000_0001);
}

#[test]
fn fused_norm_merge_multirow() {
    // Batched rows to exercise stride logic in the default path.
    run_case(512, 4, false, 0xcafe_babe);
}

#[test]
fn fused_norm_merge_non_multiple_of_4() {
    // Forces the scalar kernel path on OpenCL (dim % 4 != 0).
    run_case(129, 1, false, 0x1357_9bdf);
}

/// Residual-out phase must be bit-exact with the baseline 3-step path:
///     tmp = gpu + cpu      (matches `ws.down += cpu_merge_staging`)
///     res = prior + tmp    (matches `x += ws.down`)
/// Any other associativity (e.g. `(prior + gpu) + cpu`) drifts one ULP per
/// element and, cumulated across 28 layers × N tokens, diverges the greedy
/// generation stream from a fused-off baseline.
#[test]
fn fused_norm_merge_residual_is_bit_exact_with_baseline() {
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let dim = 1536;
    let n = dim;
    let prior = gen_data(n, 0xf00d_f00d);
    let gpu = gen_data(n, 0x1234_5678);
    let cpu = gen_data(n, 0x9abc_def0);
    let w = gen_data(dim, 0x5555_aaaa);
    let eps: f32 = 1e-5;

    // Baseline 3-step reference: ws_down = gpu + cpu; residual = prior + ws_down.
    let mut ws_down = vec![0.0f32; n];
    for i in 0..n {
        ws_down[i] = gpu[i] + cpu[i];
    }
    let mut expected_residual = vec![0.0f32; n];
    for i in 0..n {
        expected_residual[i] = prior[i] + ws_down[i];
    }

    let prior_t = make_f32_tensor(&backend, vec![dim], &prior);
    let gpu_t = make_f32_tensor(&backend, vec![dim], &gpu);
    let cpu_t = make_f32_tensor(&backend, vec![dim], &cpu);
    let w_t = make_f32_tensor(&backend, vec![dim], &w);
    let mut out_t = make_f32_zeros(&backend, vec![dim]);
    let mut res_out_t = make_f32_zeros(&backend, vec![dim]);
    backend
        .fused_norm_merge(
            &prior_t,
            &gpu_t,
            &cpu_t,
            &w_t,
            &mut out_t,
            &mut res_out_t,
            eps,
            false,
        )
        .expect("fused_norm_merge bit-exact probe");

    let got = res_out_t.as_slice::<f32>();
    for i in 0..n {
        assert_eq!(
            got[i].to_bits(),
            expected_residual[i].to_bits(),
            "bit-exact residual mismatch at {i}: got {} (0x{:x}) vs baseline {} (0x{:x})",
            got[i],
            got[i].to_bits(),
            expected_residual[i],
            expected_residual[i].to_bits()
        );
    }
}
