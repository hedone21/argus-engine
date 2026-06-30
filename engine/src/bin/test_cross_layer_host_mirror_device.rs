//! On-device verification (Adreno) for the whole-model cross-layer GPU **host-mirror K readback**.
//!
//! TriAttention's global mode reads EVERY layer's resident keys through an [`EngineCrossLayerStageCtx`].
//! On a device-resident cache those keys have no host pointer, so the ctx host-mirrors them (flush +
//! `read_buffer` device→host via `KVCache::host_snapshot`) before dequantizing. This bin runs that path
//! on the real OpenCL device: it builds device-resident F32 KV caches with known keys, reads them back
//! through the ctx, and asserts the result is **bit-identical** to reading byte-identical host caches
//! through a CPU ctx — the on-device proof that the device→host DMA preserves the exact bytes the CPU
//! path sees (so the cross-layer keep-set scored on GPU keys matches the CPU reference).
//!
//! Run on device (Adreno): `python scripts/run_device.py -d android test_cross_layer_host_mirror_device`.
//! Exit 0 = all checks pass; non-zero = first failure on stderr.

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use argus_engine::backend::Backend;
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
    println!(
        "[cross-layer-gpu] backend={} — verifying whole-model host-mirror K readback",
        gpu_backend.name()
    );

    let mut all_ok = true;
    // (n_layers, n_kv, head_dim, resident): MHA + GQA shapes, dk128 + dk64, small + odd resident.
    let configs: [(usize, usize, usize, usize); 3] =
        [(4, 2, 128, 16), (2, 4, 64, 20), (3, 8, 128, 7)];
    for (nl, nkv, hd, res) in configs {
        let (gpu_mirror, cpu_ref, gpu_confirmed) =
            argus_engine::stages::kv::mutation::cross_layer_host_mirror_selfcheck(
                &gpu_backend,
                &gpu_memory,
                nl,
                nkv,
                hd,
                res,
            )?;
        let tag = format!("L{nl}xkv{nkv}xd{hd}xn{res}");
        let nonzero = gpu_mirror.iter().any(|&x| x != 0.0);
        let eq = bits_eq(&gpu_mirror, &cpu_ref);
        if eq && nonzero && gpu_confirmed {
            println!(
                "[cross-layer-gpu] {tag}: host-mirror K byte-identical to CPU ({} vals, nonzero, \
                 device-resident)",
                gpu_mirror.len()
            );
        } else {
            all_ok = false;
            let n_diff = gpu_mirror
                .iter()
                .zip(&cpu_ref)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "[cross-layer-gpu] {tag}: FAIL eq={eq} nonzero={nonzero} gpu_confirmed={gpu_confirmed} \
                 n_diff={n_diff}/{}",
                gpu_mirror.len()
            );
        }
    }

    if all_ok {
        println!(
            "[cross-layer-gpu] PASS — whole-model host-mirror K readback byte-identical to CPU on device"
        );
        Ok(())
    } else {
        anyhow::bail!("[cross-layer-gpu] FAIL — see stderr")
    }
}

#[cfg(not(feature = "opencl"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("test_cross_layer_host_mirror_device requires the `opencl` feature")
}
