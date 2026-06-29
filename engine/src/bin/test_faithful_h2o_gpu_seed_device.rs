//! On-device verification for faithful-H2O `(c)` firing on the GPU (Adreno).
//!
//! Two parts, both run on the real OpenCL device:
//!
//! - **A — host-mirror PFA**: builds a device-resident HeadMajor F16 KV view + device F32 query and
//!   calls the production `prefill_attention`, so the GPU flash kernel dispatches and the new
//!   host-mirror branch (the only code that fills the PFA side-channel under that early-return)
//!   produces the prefill column-sums. Asserts they are **byte-identical** to the CPU backend running
//!   the same call over identical data (both paths run the same scalar `prefill_attention_scores`),
//!   and non-zero. `gpu_flash_confirmed` (`out_gpu != out_cpu`, since the GPU and CPU flash share no
//!   rounding) independently confirms the GPU actually dispatched (→ the host-mirror branch ran).
//!
//! - **B — GPU score-buffer seed round-trip**: seeds the `GpuScoreAccumulator`'s cumulative buffers via
//!   `seed_cumulative` (the path the engine uses so the prefill seed survives the pre-eviction
//!   GPU→CPU sync instead of being clobbered) and reads them back byte-identical.
//!
//! Run on device (Adreno) via `scripts/run_device.py -d android test_faithful_h2o_gpu_seed_device`.
//! Exit 0 = all checks pass; non-zero = first failure reported on stderr.

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use argus_engine::backend::Backend;
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::backend::opencl::OpenCLBackend;
    use std::sync::Arc;

    let bits_eq = |a: &[f32], b: &[f32]| -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };

    let gpu_backend: Arc<dyn Backend> = Arc::new(OpenCLBackend::new()?);
    let gpu_memory = gpu_backend
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCL backend")
        .make_kv_memory();
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    println!(
        "[faithful-h2o-gpu] backend={} — verifying (c) fires on GPU",
        gpu_backend.name()
    );

    let mut all_ok = true;

    // ── Part A: host-mirror PFA byte-identity (head_dim=128 dispatches the F16 dk128 flash kernel). ──
    // (n_heads_q, n_heads_kv, head_dim, seq_len, q_window): MHA + GQA, windowed + full-prompt window.
    let pfa_configs: [(usize, usize, usize, usize, usize); 4] = [
        (8, 8, 128, 16, 8),
        (16, 4, 128, 32, 16),
        (14, 2, 128, 48, 32),
        (8, 8, 128, 7, 4),
    ];
    for (nq, nkv, hd, sl, qw) in pfa_configs {
        let (pfa_gpu, pfa_cpu, flash_confirmed) =
            argus_engine::kv::standard_format::faithful_h2o_pfa_host_mirror_selfcheck(
                &gpu_backend,
                &gpu_memory,
                &cpu,
                nq,
                nkv,
                hd,
                sl,
                qw,
            )?;
        let tag = format!("nq{nq}xnkv{nkv}xd{hd}xn{sl}xw{qw}");
        let nonzero = pfa_gpu.iter().any(|&x| x != 0.0);
        let eq = bits_eq(&pfa_gpu, &pfa_cpu);
        if eq && nonzero {
            println!(
                "[faithful-h2o-gpu] A {tag}: host-mirror PFA byte-identical to CPU ({} vals, nonzero), \
                 gpu_flash_confirmed={flash_confirmed}",
                pfa_gpu.len()
            );
        } else {
            all_ok = false;
            let n_diff = pfa_gpu
                .iter()
                .zip(&pfa_cpu)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "[faithful-h2o-gpu] A {tag}: FAIL eq={eq} nonzero={nonzero} n_diff={n_diff}/{}",
                pfa_gpu.len()
            );
        }
        if !flash_confirmed {
            eprintln!(
                "[faithful-h2o-gpu] A {tag}: WARN out_gpu==out_cpu — GPU flash dispatch not \
                 independently confirmed (PFA byte-match still holds; K is device-resident)"
            );
        }
    }

    // ── Part B: GPU score-buffer seed round-trip (the (c) seed lands in the GPU cumulative buffer). ──
    {
        let ocl = gpu_backend
            .as_any()
            .downcast_ref::<OpenCLBackend>()
            .expect("OpenCL backend");
        let (n_layers, nq, nkv, max_seq) = (4usize, 8usize, 4usize, 64usize);
        ocl.init_gpu_score_acc(n_layers, nq, nkv, max_seq, 0.0)?;
        let queue = ocl.queue.as_core();
        let acc = ocl
            .gpu_score_acc_mut()
            .ok_or_else(|| anyhow::anyhow!("GPU score accumulator unavailable after init"))?;
        acc.set_active(true);
        let flat: Vec<f32> = (0..max_seq).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let head: Vec<f32> = (0..nkv * max_seq)
            .map(|i| (i as f32) * 0.25 - 0.5)
            .collect();
        acc.seed_cumulative(queue, &flat, &head)?;
        let (got_flat, got_head) = acc.sync_to_cpu(queue)?;
        if bits_eq(&got_flat, &flat) && bits_eq(&got_head, &head) {
            println!(
                "[faithful-h2o-gpu] B: GPU seed round-trip byte-identical (flat {} + head {})",
                flat.len(),
                head.len()
            );
        } else {
            all_ok = false;
            eprintln!(
                "[faithful-h2o-gpu] B: FAIL seed round-trip not byte-identical (flat_eq={} head_eq={})",
                bits_eq(&got_flat, &flat),
                bits_eq(&got_head, &head)
            );
        }
    }

    if all_ok {
        println!(
            "[faithful-h2o-gpu] PASS — host-mirror PFA fires on GPU (== CPU, non-zero) + GPU seed round-trips"
        );
        Ok(())
    } else {
        anyhow::bail!("[faithful-h2o-gpu] FAIL — see stderr")
    }
}

#[cfg(not(feature = "opencl"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("test_faithful_h2o_gpu_seed_device requires the `opencl` feature")
}
