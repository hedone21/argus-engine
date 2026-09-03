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

    let mut failed = 0usize;
    for &(pos, ragged) in &[
        (4096usize, false),
        (4096, true),
        (1559, true), // the mean `tokens_before` of the 8K argus_full cell
        (1023, true),
        (199, true), // unaligned to the work-group size
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
            ragged,
        )?;
        let verdict = if !r.gpu_ran {
            failed += 1;
            "DECLINED"
        } else if r.argmax_mismatch > 0 || r.max_rel >= 1e-3 {
            failed += 1;
            "MISMATCH"
        } else {
            "ok"
        };
        println!(
            "[window-gpu] pos={pos:5} ragged={ragged:5} {verdict:8} \
             cpu={:.3}s gpu={:.3}s speedup={:5.1}x  max_rel={:.2e} argmax_mismatch={} loose={}",
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
        );
    }
    if failed > 0 {
        anyhow::bail!("{failed} case(s) failed");
    }
    println!("[window-gpu] PASS");
    Ok(())
}
