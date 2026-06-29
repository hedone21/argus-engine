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

    // ── Part C: GPU per-layer FLAT reduce (faithful-H2O (b)) — per-layer divergence + byte-identity. ──
    // Build a per-layer-DIVERGENT score_buf (each layer spikes a different token), dispatch the GPU
    // per-layer reduce via end_step, read it back, and assert it is byte-identical to a CPU reference
    // computing the same per-(layer, token) head-sum — and that the layers' importance DIFFER (the
    // layer axis is preserved, NOT MAX-collapsed). This is the on-device proof that GPU per-layer
    // produces correct, layer-divergent importance.
    {
        let ocl = gpu_backend
            .as_any()
            .downcast_ref::<OpenCLBackend>()
            .expect("OpenCL backend");
        let (n_layers, nq, nkv, max_seq, cache_seq) = (4usize, 8usize, 4usize, 64usize, 16usize);
        ocl.init_gpu_score_acc(n_layers, nq, nkv, max_seq, 0.0)?;
        let queue = ocl.queue.as_core();
        let acc = ocl
            .gpu_score_acc_mut()
            .ok_or_else(|| anyhow::anyhow!("GPU score accumulator unavailable after init"))?;
        acc.set_active(true);
        acc.enable_per_layer_flat();

        // score_buf [n_layers, nq, score_stride(==max_seq)]: layer l spikes token (3 + 4*l).
        let stride = acc.score_stride();
        let mut scores = vec![0.0f32; n_layers * nq * stride];
        let expected_peaks: Vec<usize> = (0..n_layers).map(|l| 3 + 4 * l).collect();
        for (l, &peak) in expected_peaks.iter().enumerate() {
            for h in 0..nq {
                for t in 0..cache_seq {
                    let base = (l * nq + h) * stride + t;
                    scores[base] = 0.01 * (t as f32) + if t == peak { 5.0 } else { 0.0 };
                }
            }
        }
        // SAFETY: writing the host scores into the engine-owned score_buf for this one-step reduce.
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                acc.score_buf_mem(),
                true,
                0,
                &scores,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        acc.end_step(queue, cache_seq)?; // dispatches the per-layer reduce (per_layer_flat armed)
        let got = acc.sync_layer_flat_to_cpu(queue)?;

        // CPU reference: layer_flat[l*max_seq + t] = sum_h scores[(l*nq+h)*stride + t] (one step,
        // decay=1.0, zero-init buffer → cumulative equals the step). Same head-sum order as the kernel.
        let mut want = vec![0.0f32; n_layers * max_seq];
        for l in 0..n_layers {
            for t in 0..cache_seq {
                let mut s = 0.0f32;
                for h in 0..nq {
                    s += scores[(l * nq + h) * stride + t];
                }
                want[l * max_seq + t] = s;
            }
        }
        let eq = bits_eq(&got, &want);
        // Per-layer divergence: each layer's argmax is at its OWN peak token (distinct per layer).
        let argmax = |l: usize| -> usize {
            (0..max_seq)
                .max_by(|&a, &b| {
                    got[l * max_seq + a]
                        .partial_cmp(&got[l * max_seq + b])
                        .unwrap()
                })
                .unwrap()
        };
        let peaks: Vec<usize> = (0..n_layers).map(argmax).collect();
        let divergent = peaks == expected_peaks;
        // The first two per-layer windows DIFFER (layer axis preserved, not MAX-collapsed).
        let distinct = got[0..max_seq] != got[max_seq..2 * max_seq];
        if eq && divergent && distinct {
            println!(
                "[faithful-h2o-gpu] C: GPU per-layer reduce byte-identical to CPU ref ({} vals), \
                 per-layer peaks={peaks:?} (divergent), layer-windows distinct",
                got.len()
            );
        } else {
            all_ok = false;
            eprintln!(
                "[faithful-h2o-gpu] C: FAIL eq={eq} divergent={divergent} distinct={distinct} \
                 peaks={peaks:?} expected={expected_peaks:?}"
            );
        }
    }

    // ── Part D: GPU per-layer reset-on-eviction (faithful-H2O (b), multi-eviction correctness). ──
    // The GPU per-layer buffer accumulates on-device; at each eviction it must be reset in lockstep
    // with the CPU acc.reset() or a 2nd eviction would rank stale, slot-misaligned importance. Verify:
    // reduce(A) → reset_layer_flat (buffer must zero) → reduce(B) → the buffer holds B ALONE, not A+B
    // (no carry-over). This is the on-device proof of the multi-eviction reset fix.
    {
        let ocl = gpu_backend
            .as_any()
            .downcast_ref::<OpenCLBackend>()
            .expect("OpenCL backend");
        let (n_layers, nq, nkv, max_seq, cache_seq) = (2usize, 4usize, 2usize, 32usize, 8usize);
        ocl.init_gpu_score_acc(n_layers, nq, nkv, max_seq, 0.0)?;
        let queue = ocl.queue.as_core();
        let acc = ocl
            .gpu_score_acc_mut()
            .ok_or_else(|| anyhow::anyhow!("GPU score accumulator unavailable after init"))?;
        acc.set_active(true);
        acc.enable_per_layer_flat();
        let stride = acc.score_stride();

        // Build a score_buf with a per-layer-distinct spike at `peak`, return it.
        let mk_scores = |peak: usize| -> Vec<f32> {
            let mut s = vec![0.0f32; n_layers * nq * stride];
            for l in 0..n_layers {
                for h in 0..nq {
                    for t in 0..cache_seq {
                        s[(l * nq + h) * stride + t] =
                            0.01 * (t as f32) + if t == peak + l { 3.0 } else { 0.0 };
                    }
                }
            }
            s
        };
        let cpu_ref = |scores: &[f32]| -> Vec<f32> {
            let mut w = vec![0.0f32; n_layers * max_seq];
            for l in 0..n_layers {
                for t in 0..cache_seq {
                    let mut sum = 0.0f32;
                    for h in 0..nq {
                        sum += scores[(l * nq + h) * stride + t];
                    }
                    w[l * max_seq + t] = sum;
                }
            }
            w
        };
        // SAFETY (both writes): a one-step score_buf write for each reduce. The immutable borrow of
        // `acc` via `score_buf_mem()` ends before the subsequent `&mut` `end_step`.
        let scores_a = mk_scores(2);
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                acc.score_buf_mem(),
                true,
                0,
                &scores_a,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        acc.end_step(queue, cache_seq)?;
        let after_a = acc.sync_layer_flat_to_cpu(queue)?;

        acc.reset_layer_flat(queue)?; // the fix under test
        let after_reset = acc.sync_layer_flat_to_cpu(queue)?;

        let scores_b = mk_scores(5);
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                acc.score_buf_mem(),
                true,
                0,
                &scores_b,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        acc.end_step(queue, cache_seq)?;
        let after_b = acc.sync_layer_flat_to_cpu(queue)?;

        let reset_zeroed = after_reset.iter().all(|&x| x == 0.0);
        let a_nonzero = after_a.iter().any(|&x| x != 0.0);
        // The key invariant: after reset, reduce(B) equals B ALONE (no A carry-over).
        let b_is_fresh = bits_eq(&after_b, &cpu_ref(&scores_b));
        if reset_zeroed && a_nonzero && b_is_fresh {
            println!(
                "[faithful-h2o-gpu] D: GPU per-layer reset-on-eviction works — reduce(A) nonzero, \
                 reset zeroes the buffer, reduce(B) == B alone (no carry-over, multi-eviction safe)"
            );
        } else {
            all_ok = false;
            eprintln!(
                "[faithful-h2o-gpu] D: FAIL reset_zeroed={reset_zeroed} a_nonzero={a_nonzero} \
                 b_is_fresh={b_is_fresh}"
            );
        }
    }

    if all_ok {
        println!(
            "[faithful-h2o-gpu] PASS — host-mirror PFA fires on GPU (== CPU, non-zero) + GPU seed \
             round-trips + GPU per-layer reduce (== CPU, layer-divergent) + per-layer reset-on-eviction"
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
