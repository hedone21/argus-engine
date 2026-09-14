//! On-device A/B for the decision-time observation-window attention (Adreno).
//!
//! `kv::aperturb_select::window_attention` is the largest single item in a compression decision's
//! stall (mean 1.352 s of a 2.5–3 s decision on an S25, measured 2026-09-03). This bin runs the
//! CPU reference and the OpenCL kernel over the same synthetic cache at the production geometry
//! and reports both the divergence and the wall clock.
//!
//! Correctness bar is the one the consumers actually read: the two paths must agree on the
//! top-scoring column of every (layer, query head), because that is what a candidate's top-k
//! ranks from. The float gap is reported but is not the gate.
//!
//! Run: `python scripts/run_device.py -d android2 test_window_attn_gpu_device`
//! Exit 0 = pass.

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("built without the opencl feature — nothing to check");
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use std::sync::Arc;

    use argus_engine::backend::Backend;
    use argus_engine::backend::opencl::OpenCLBackend;
    use argus_engine::backend::opencl::memory::OpenCLMemory;
    use argus_engine::kv::aperturb_select::window_attention_selfcheck;
    use argus_engine::memory::Memory;

    let ocl = OpenCLBackend::new()?;
    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));
    let backend: Arc<dyn Backend> = Arc::new(ocl);

    // Qwen2.5-1.5B, the §5.2 model: 28 layers, 12 query heads over 2 KV heads, head_dim 128.
    // `current_pos` walks the range the 8K cell actually decided at (`tokens_before` ran
    // 4096 → 1023 over its 18 decisions), at the capacity that cell allocated.
    const LAYERS: usize = 28;
    const HQ: usize = 12;
    const HKV: usize = 2;
    const HD: usize = 128;
    const CAP: usize = 8192;
    const ROWS: usize = 64;

    // `APERTURB_ROWS` — the metric's own row count, which is what A1 exports.
    const MROWS: usize = 16;

    let mut failed = 0usize;
    for &(pos, mrows, ragged) in &[
        // The pooled-attention cases, unchanged. `metric_rows == ROWS` here, which means they do
        // NOT execute A1's tail seam — the mapping from window row to metric row is the identity.
        (4096usize, ROWS, false),
        (4096, ROWS, true),
        (1559, ROWS, true), // the mean `tokens_before` of the 8K argus_full cell
        (1023, ROWS, true),
        (199, ROWS, true), // unaligned to the work-group size
        // The tail seam, at production's own 16-of-64 and at two shapes that break "export whole
        // row blocks": 12 starts mid-block at WATT_B=8, and 5 fits inside one block.
        (4096, MROWS, true),
        (1559, MROWS, true),
        (1023, MROWS, true),
        (1023, 12, true),
        (199, 5, true),
    ] {
        let r = window_attention_selfcheck(
            &backend,
            memory.as_ref(),
            LAYERS,
            HQ,
            HKV,
            HD,
            CAP,
            pos,
            ROWS,
            mrows,
            ragged,
        )?;
        let verdict = if !r.gpu_ran {
            failed += 1;
            "DECLINED"
        } else if r.argmax_mismatch > 0 || r.max_rel >= 1e-3 {
            failed += 1;
            "MISMATCH"
        } else if !r.z_gpu_ran || r.z_rows_compared == 0 {
            // Not folded into MISMATCH: an export that never happened, or one that compared
            // nothing, is a wiring failure and reports a perfect score if it is not named.
            failed += 1;
            "NO-EXPORT"
        } else if r.z_argmax_unexplained > 0 || r.z_max_abs >= 1e-3 || r.z_max_abs <= 0.0 {
            // The gate is `z_argmax_unexplained` and `z_max_abs`, NOT `z_argmax_mismatch` and NOT
            // `z_max_rel` (contract §3′). A raw-argmax flip can be two columns 2 ULP apart that
            // the GPU's summation order collapses to one float — 1 of 21504 triples does exactly
            // that on Adreno (`tickets/010-evidence/z_tie_diagnostic_2026-09-11.log`), and
            // demanding zero would be demanding the bit-identity this ticket gave up. A relative
            // gap is unbounded here because raw logits pass through zero. Both are printed.
            //
            // `z_max_abs` is gated on BOTH sides — `0 < z_max_abs < 1e-3` (contract §3″). The
            // lower bound is the provenance check: a z the host recomputed for itself would be
            // bit-identical to `kernel::logits_into` and read exactly 0.0, while the kernel sums
            // `dot()` over float4s where the CPU blocks TB=8/VW=4, so across >20M compared
            // elements they cannot agree everywhere (measured floor 1.19e-7 on Adreno).
            failed += 1;
            "Z-MISMATCH"
        } else {
            "ok"
        };
        println!(
            "[window-gpu] pos={pos:5} metric_rows={mrows:3} ragged={ragged:5} {verdict:10} \
             cpu={:.3}s gpu={:.3}s speedup={:5.1}x  max_rel={:.2e} argmax_mismatch={} loose={} \
             z_gpu_ran={} z_rows_compared={} z_argmax_mismatch={} z_argmax_unexplained={} \
             z_max_abs={:.2e} z_max_rel={:.2e}",
            r.cpu_s,
            r.gpu_s,
            if r.gpu_s > 0.0 {
                r.cpu_s / r.gpu_s
            } else {
                0.0
            },
            r.max_rel,
            r.argmax_mismatch,
            r.loose_cols,
            r.z_gpu_ran,
            r.z_rows_compared,
            r.z_argmax_mismatch,
            r.z_argmax_unexplained,
            r.z_max_abs,
            r.z_max_rel,
        );
    }
    if failed > 0 {
        anyhow::bail!("{failed} case(s) failed");
    }
    println!("[window-gpu] PASS");
    Ok(())
}
