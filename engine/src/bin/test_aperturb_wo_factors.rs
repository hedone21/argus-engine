//! Check the engine's rank-`r` output-projection factorization against an exact reference, on real
//! model weights.
//!
//! ```text
//! cargo run --release --bin test_aperturb_wo_factors -- <reference.bin>
//! ```
//!
//! The reference file carries, per layer, the `o_proj` weight and the factors a float64 LAPACK thin
//! SVD produced from it — the same computation the paper harness runs. The engine has no such
//! solver, so this is the only way to know whether its own iteration lands on the same truncation.
//!
//! **What is compared is not the vectors.** Where the metric cuts the spectrum, `σ_{r+1}/σ_r` is
//! ~0.97–0.995 on real weights, so individual directions near the cut are barely determined and two
//! correct implementations will disagree about them. What the measurement depends on is
//! `M_r = V_r Σ_r² V_rᵀ`, the truncation of the Gram, and that *is* determined: mixing between two
//! near-degenerate directions moves it only in proportion to the gap it crosses. So the verdict is
//! `‖M̃_r − M_r‖_F / ‖M_r‖_F`, which maps to the per-cell error in the metric roughly one-to-one.
//!
//! Manual, like every other test in this directory that needs a real model: it reads a multi-megabyte
//! file this repository does not carry.

use std::io::Read;

use argus_engine::aperturb::subspace::{FactorConfig, top_right_singular};

struct Layer {
    index: usize,
    w: Vec<f32>,
    v_r: Vec<f64>,
    sigma: Vec<f64>,
    tail: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or(
        "usage: test_aperturb_wo_factors <reference.bin>  (produced by dump_wo_reference.py)",
    )?;
    let mut buf = Vec::new();
    std::fs::File::open(&path)?.read_to_end(&mut buf)?;

    let mut o = 0usize;
    let take = |n: usize, o: &mut usize| {
        let s = &buf[*o..*o + n];
        *o += n;
        s
    };
    if take(8, &mut o) != b"ARGUSWOR" {
        return Err("not a reference file".into());
    }
    let u32_at = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap()) as usize;
    let f64_at = |s: &[u8]| f64::from_le_bytes(s.try_into().unwrap());
    let version = u32_at(take(4, &mut o));
    let n_layers = u32_at(take(4, &mut o));
    if version != 1 {
        return Err(format!("reference version {version}, expected 1").into());
    }
    let d_out = u32_at(take(4, &mut o));
    let d_in = u32_at(take(4, &mut o));
    let r = u32_at(take(4, &mut o));
    let _pad = take(4, &mut o);

    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let index = u32_at(take(4, &mut o));
        let _pad = take(4, &mut o);
        let w: Vec<f32> = take(d_out * d_in * 4, &mut o)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let v_r: Vec<f64> = take(d_in * r * 8, &mut o)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        let sigma: Vec<f64> = take(r * 8, &mut o)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        let tail = f64_at(take(8, &mut o));
        layers.push(Layer {
            index,
            w,
            v_r,
            sigma,
            tail,
        });
    }

    println!("reference: {n_layers} layers, d_out={d_out} d_in={d_in} r={r}");
    let cfg = FactorConfig::default();
    println!(
        "engine:    block {}r, energy tol {:e}, cap {} iterations, f64\n",
        cfg.block_mult, cfg.energy_tol, cfg.max_iters
    );

    let mut worst_m = 0.0f64;
    let mut fails = 0usize;
    for l in &layers {
        let t0 = std::time::Instant::now();
        let (v, s, rep) = match top_right_singular(&l.w, d_out, d_in, r, &cfg, l.index) {
            Ok(x) => x,
            Err(e) => {
                println!("layer {:3}  FAILED: {e}", l.index);
                fails += 1;
                continue;
            }
        };
        let dt = t0.elapsed().as_secs_f64();

        // ‖M̃_r − M_r‖_F / ‖M_r‖_F, expanded so neither operator is materialized:
        //   ‖A − B‖² = tr(A²) + tr(B²) − 2 tr(AB), and for A = Σ λ_i v_i v_iᵀ these are sums over
        //   λ, μ and the r×r cross Gram (v_iᵀ u_j)².
        let cross = |a: &[f64], b: &[f64]| -> Vec<f64> {
            let mut g = vec![0.0f64; r * r];
            for x in 0..d_in {
                let (ra, rb) = (&a[x * r..(x + 1) * r], &b[x * r..(x + 1) * r]);
                for i in 0..r {
                    for j in 0..r {
                        g[i * r + j] += ra[i] * rb[j];
                    }
                }
            }
            g
        };
        let g = cross(&v, &l.v_r);
        let lam: Vec<f64> = s.iter().map(|x| x * x).collect();
        let mu: Vec<f64> = l.sigma.iter().map(|x| x * x).collect();
        let tr_aa: f64 = lam.iter().map(|x| x * x).sum();
        let tr_bb: f64 = mu.iter().map(|x| x * x).sum();
        let mut tr_ab = 0.0f64;
        for i in 0..r {
            for j in 0..r {
                tr_ab += lam[i] * mu[j] * g[i * r + j] * g[i * r + j];
            }
        }
        let m_rel = ((tr_aa + tr_bb - 2.0 * tr_ab).max(0.0)).sqrt() / tr_bb.sqrt();
        worst_m = worst_m.max(m_rel);

        let sig_rel = s
            .iter()
            .zip(&l.sigma)
            .map(|(a, b)| (a - b).abs() / b)
            .fold(0.0f64, f64::max);
        // The subspace angle, reported only to show how much worse it is than the thing that matters.
        let sin_theta = (1.0
            - (0..r)
                .map(|i| (0..r).map(|j| g[i * r + j] * g[i * r + j]).sum::<f64>())
                .fold(f64::INFINITY, f64::min))
        .max(0.0)
        .sqrt();

        println!(
            "layer {:3}  M_rel {:.2e}  sigma_rel {:.2e}  sin(theta) {:.2e}  \
             | iters {:2} gain {:.1e} res {:.1e} orth {:.1e} defl {:.4}  | {:.1}s  tail {:.3e}",
            l.index,
            m_rel,
            sig_rel,
            sin_theta,
            rep.iters,
            rep.energy_gain,
            rep.residual,
            rep.orthogonality,
            rep.deflation_ratio,
            dt,
            l.tail
        );
    }

    // The parity bar the fixture asserts on is 1e-5 relative for the l2 readout; the per-cell error
    // in the metric tracks M_rel to within a small factor, so this is the number that has to clear
    // it, with room.
    println!("\nworst M_rel = {worst_m:.3e}   (parity bar 1e-5; failures {fails})");
    if fails > 0 || worst_m > 1e-5 {
        return Err(
            format!("factorization does not meet the parity bar: M_rel {worst_m:e}").into(),
        );
    }
    println!("PASS");
    Ok(())
}
