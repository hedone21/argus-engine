//! Diagnostic: compare `convert_q4_0_to_noshuffle` GPU kernel output against
//! `q4_0_aos_to_adreno_soa` Rust reference algorithm for byte-equality on
//! the active OpenCL device (typically Adreno).
//!
//! Mirrors the host-only INV-140 spec test
//! (`tests/spec/test_inv_140_fused_convert_byte_equal.rs`) but runs on the
//! actual device so we can isolate Adreno-specific divergence — INV-140
//! covers correctness on host NVIDIA OpenCL only and missed the Adreno
//! regression that broke AUF AOS swap (commit 4416994 / 2026-05-01).
//!
//! Exit codes: 0 = byte-equal across all cases; non-zero = first mismatch
//! reported on stderr.

use argus_engine::auf::q4_0_soa::q4_0_aos_to_adreno_soa;
use argus_engine::backend::Backend;
#[cfg(feature = "opencl")]
use argus_engine::backend::opencl::{OpenCLBackend, get_cl_mem};
use argus_engine::buffer::DType;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use clap::Parser;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Cases to test as comma-separated `ne00xne01` pairs (e.g. `64x32,1536x1536`).
    #[arg(
        long,
        default_value = "64x32,128x64,1536x256,1536x1536,1536x8960,8960x1536"
    )]
    cases: String,

    /// Random seed.
    #[arg(long, default_value_t = 0xC0FFEE)]
    seed: u64,

    /// Maximum number of mismatching bytes to report per buffer.
    #[arg(long, default_value_t = 8)]
    max_mismatch_report: usize,

    /// Force legacy 4-step path (disable fused kernel even if available).
    #[arg(long, default_value_t = false)]
    legacy: bool,
}

/// Pseudo-random Q4_0 AOS buffer (block_q4_0 layout: 2 byte d + 16 byte qs = 18 byte/block).
fn make_q4_0_aos(num_blocks: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut bytes = Vec::with_capacity(num_blocks * 18);
    for b in 0..num_blocks {
        // d as half (LE).
        let d_u16 = (b as u16).wrapping_mul(0x4321) ^ ((seed >> 16) as u16);
        bytes.extend_from_slice(&d_u16.to_le_bytes());
        // qs: 16 bytes of pseudo-random nibble pairs.
        for _ in 0..16 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            bytes.push((state >> 33) as u8);
        }
    }
    bytes
}

#[cfg(feature = "opencl")]
fn run_case(
    backend: &OpenCLBackend,
    backend_arc: &Arc<dyn Backend>,
    memory: &Galloc,
    ne00: usize,
    ne01: usize,
    seed: u64,
    max_report: usize,
) -> anyhow::Result<bool> {
    assert!(ne00.is_multiple_of(32), "ne00 must be multiple of 32");
    let blocks_per_row = ne00 / 32;
    let num_blocks = ne01 * blocks_per_row;
    let aos_bytes = make_q4_0_aos(num_blocks, seed);

    println!(
        "[test_q4_soa_byte_equal] case ne00={} ne01={} blocks={}",
        ne00, ne01, num_blocks
    );

    // ── (1) Reference (Rust): q4_0_aos_to_adreno_soa ──────────────────────────
    let (ref_q, ref_d) = q4_0_aos_to_adreno_soa(&aos_bytes, ne00, ne01);

    // ── (2) GPU kernel: convert_q4_0_to_noshuffle ─────────────────────────────
    // Upload AOS bytes to a fresh cl_mem via copy_weight_from path (matches the
    // swap-time materialise_tensor pipeline that landed the bug).
    let buf = memory.alloc(aos_bytes.len(), DType::Q4_0)?;
    unsafe {
        std::ptr::copy_nonoverlapping(aos_bytes.as_ptr(), buf.as_mut_ptr(), aos_bytes.len());
    }
    let shape = Shape::new(vec![ne01, ne00]);
    let cpu_be: Arc<dyn Backend> = Arc::new(argus_engine::backend::cpu::CpuBackend::new());
    let cpu_tensor = Tensor::new(shape.clone(), buf, cpu_be);
    let gpu_tensor = backend_arc.copy_weight_from(&cpu_tensor)?;
    let src_mem = get_cl_mem(gpu_tensor.buffer().as_ref())?;

    let (dst_q, dst_d, _q_img) =
        backend.convert_q4_0_to_noshuffle(src_mem, num_blocks, ne00, ne01)?;
    backend.synchronize()?;

    // Read GPU output back to host.
    let mut gpu_q = vec![0u8; ref_q.len()];
    let mut gpu_d = vec![0u8; ref_d.len()];
    unsafe {
        ocl::core::enqueue_read_buffer(
            &backend.queue,
            &dst_q,
            true,
            0,
            &mut gpu_q,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_read_buffer(
            &backend.queue,
            &dst_d,
            true,
            0,
            &mut gpu_d,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }

    // ── (3) byte-by-byte compare ──────────────────────────────────────────────
    let mut q_mismatches = 0usize;
    let mut d_mismatches = 0usize;
    for (i, (a, b)) in gpu_q.iter().zip(ref_q.iter()).enumerate() {
        if a != b {
            q_mismatches += 1;
            if q_mismatches <= max_report {
                eprintln!(
                    "  q[{:8}] gpu=0x{:02x} ref=0x{:02x}  (col_ushort_idx={}, row_pair_byte={})",
                    i,
                    a,
                    b,
                    i / (ne01 * 2),
                    i % (ne01 * 2),
                );
            }
        }
    }
    for (i, (a, b)) in gpu_d.iter().zip(ref_d.iter()).enumerate() {
        if a != b {
            d_mismatches += 1;
            if d_mismatches <= max_report {
                eprintln!(
                    "  d[{:8}] gpu=0x{:02x} ref=0x{:02x}  (block_col_idx={}, row_pair_byte={})",
                    i,
                    a,
                    b,
                    i / (ne01 * 2),
                    i % (ne01 * 2),
                );
            }
        }
    }

    if q_mismatches == 0 && d_mismatches == 0 {
        println!(
            "  ✓ byte-equal: q={}KB d={}KB",
            gpu_q.len() / 1024,
            gpu_d.len() / 1024
        );
        Ok(true)
    } else {
        eprintln!(
            "  ✗ MISMATCH: q={}/{} bytes ({:.2}%), d={}/{} bytes ({:.2}%)",
            q_mismatches,
            gpu_q.len(),
            100.0 * q_mismatches as f64 / gpu_q.len() as f64,
            d_mismatches,
            gpu_d.len(),
            100.0 * d_mismatches as f64 / gpu_d.len() as f64,
        );
        Ok(false)
    }
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let backend_arc: Arc<dyn Backend> = Arc::new(OpenCLBackend::new()?);
    let backend = backend_arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCL backend");
    let memory = Galloc::new();

    println!(
        "[test_q4_soa_byte_equal] backend={} seed=0x{:x}",
        backend_arc.name(),
        args.seed
    );

    let mut all_ok = true;
    for case_str in args.cases.split(',') {
        let case_str = case_str.trim();
        let (ne00_s, ne01_s) = case_str
            .split_once('x')
            .ok_or_else(|| anyhow::anyhow!("Invalid case format: '{}'", case_str))?;
        let ne00: usize = ne00_s.parse()?;
        let ne01: usize = ne01_s.parse()?;
        let ok = run_case(
            backend,
            &backend_arc,
            &memory,
            ne00,
            ne01,
            args.seed,
            args.max_mismatch_report,
        )?;
        all_ok &= ok;
    }

    if all_ok {
        println!("[test_q4_soa_byte_equal] all cases byte-equal");
        Ok(())
    } else {
        anyhow::bail!("[test_q4_soa_byte_equal] one or more cases mismatched");
    }
}

#[cfg(not(feature = "opencl"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("test_q4_soa_byte_equal requires the `opencl` feature")
}
