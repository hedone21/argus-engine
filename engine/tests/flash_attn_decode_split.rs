//! Host parity test for the split-KV decode attention pair (ticket 019).
//!
//! `flash_attention_decode_split_gpu` (`flash_attn_f32_f16_q1_split` +
//! `flash_attn_q1_merge`) must reproduce, for every `n_splits`, what the
//! original single-work-group `flash_attention_decode_q1_gpu` produces:
//! the attention output O, the post-softmax score row per head (including
//! the zeroed hole columns of a ragged cache), and untouched columns beyond
//! `n_kv`. Both are also checked against an F32 three-pass CPU reference so a
//! shared bug cannot hide.
//!
//! Skips cleanly on hosts without an OpenCL device (the plan path is covered
//! on-device by the ticket's A/B batch).

#![cfg(feature = "opencl")]
// The reference is written as plain index loops on purpose: it mirrors the
// kernel's row/column arithmetic so a reader can check them side by side.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use argus_engine::backend::Backend;
use argus_engine::backend::opencl::OpenCLBackend;
use argus_engine::backend::opencl::memory::OpenCLMemory;
use argus_engine::memory::Memory;
use std::sync::Arc;

/// (n_heads_q, n_heads_kv, head_dim): Qwen2.5-1.5B and Llama-3.2-1B shapes.
const SHAPES: [(usize, usize, usize); 2] = [(12, 2, 128), (32, 8, 64)];
const N_KV: [usize; 8] = [1, 63, 64, 65, 300, 1000, 4096, 8191];
const SPLITS: [usize; 5] = [1, 2, 3, 16, 64];
const CAPACITY: usize = 8192;
/// Relative tolerance on O and on score weights (sum-order differences only).
const REL_TOL: f32 = 1e-4;
/// Sentinel written to score columns the kernels must not touch.
const SENTINEL: f32 = -7777.5;

#[derive(Clone, Copy, Debug)]
enum Ragged {
    Uniform,
    /// `kv_start[h] = (h * 37) % (n_kv + 1)` — small holes, one head may be empty.
    Small,
    /// `kv_start[h] = n_kv * h / n_heads_kv` — large holes.
    Large,
}

fn kv_starts(r: Ragged, n_heads_kv: usize, n_kv: usize) -> Option<Vec<i32>> {
    match r {
        Ragged::Uniform => None,
        Ragged::Small => Some(
            (0..n_heads_kv)
                .map(|h| ((h * 37) % (n_kv + 1)) as i32)
                .collect(),
        ),
        Ragged::Large => Some(
            (0..n_heads_kv)
                .map(|h| (n_kv * h / n_heads_kv) as i32)
                .collect(),
        ),
    }
}

/// Deterministic pseudo-random values in [-1, 1).
fn lcg(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Three-pass F32 reference over the F16-rounded K/V the kernels see.
/// Returns `(o[n_heads_q * head_dim], scores[n_heads_q][n_kv])`; hole
/// columns `[0, k_lo)` are 0, exactly what the kernels must write.
fn reference(
    q: &[f32],
    k16: &[u16],
    v16: &[u16],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_kv: usize,
    starts: Option<&[i32]>,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let gqa = n_heads_q / n_heads_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut o = vec![0.0f32; n_heads_q * head_dim];
    let mut scores = vec![vec![0.0f32; n_kv]; n_heads_q];
    for h in 0..n_heads_q {
        let kv_h = h / gqa;
        let k_lo = starts.map_or(0, |s| (s[kv_h] as usize).min(n_kv));
        let qv = &q[h * head_dim..(h + 1) * head_dim];
        let mut logits = vec![f32::NEG_INFINITY; n_kv];
        let mut m = f32::NEG_INFINITY;
        for t in k_lo..n_kv {
            let off = (kv_h * CAPACITY + t) * head_dim;
            let dot: f32 = (0..head_dim)
                .map(|d| qv[d] * half::f16::from_bits(k16[off + d]).to_f32())
                .sum();
            logits[t] = dot * scale;
            m = m.max(logits[t]);
        }
        if m == f32::NEG_INFINITY {
            continue; // empty head: O = 0, scores = 0
        }
        let mut l = 0.0f32;
        for t in k_lo..n_kv {
            let p = (logits[t] - m).exp();
            scores[h][t] = p;
            l += p;
        }
        for t in k_lo..n_kv {
            scores[h][t] /= l;
            let off = (kv_h * CAPACITY + t) * head_dim;
            for d in 0..head_dim {
                o[h * head_dim + d] += scores[h][t] * half::f16::from_bits(v16[off + d]).to_f32();
            }
        }
    }
    (o, scores)
}

struct Gpu {
    arc: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
}

impl Gpu {
    fn ocl(&self) -> &OpenCLBackend {
        self.arc
            .as_any()
            .downcast_ref::<OpenCLBackend>()
            .expect("OpenCLBackend downcast")
    }
    fn upload(
        &self,
        bytes: &[u8],
        dtype: argus_engine::buffer::DType,
        shape: Vec<usize>,
    ) -> argus_engine::tensor::Tensor {
        let buf = self.memory.alloc(bytes.len(), dtype).unwrap();
        let mut t = argus_engine::tensor::Tensor::new(
            argus_engine::shape::Shape::new(shape),
            buf,
            self.arc.clone(),
        );
        self.arc.write_buffer(&mut t, bytes).unwrap();
        t
    }
    fn f32_tensor(&self, data: &[f32], shape: Vec<usize>) -> argus_engine::tensor::Tensor {
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.upload(bytes, argus_engine::buffer::DType::F32, shape)
    }
    fn f16_tensor(&self, data: &[u16], shape: Vec<usize>) -> argus_engine::tensor::Tensor {
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        self.upload(bytes, argus_engine::buffer::DType::F16, shape)
    }
    fn read_f32_tensor(&self, t: &argus_engine::tensor::Tensor, n: usize) -> Vec<f32> {
        let mut raw = vec![0u8; n * 4];
        self.arc.read_buffer(t, &mut raw).unwrap();
        (0..n)
            .map(|i| f32::from_ne_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()))
            .collect()
    }
    fn mem_from_f32(&self, data: &[f32]) -> ocl::core::Mem {
        unsafe {
            ocl::core::create_buffer(
                self.ocl().context.as_core(),
                ocl::core::MEM_READ_WRITE | ocl::core::MEM_COPY_HOST_PTR,
                data.len(),
                Some(data),
            )
        }
        .unwrap()
    }
    fn mem_from_i32(&self, data: &[i32]) -> ocl::core::Mem {
        unsafe {
            ocl::core::create_buffer(
                self.ocl().context.as_core(),
                ocl::core::MEM_READ_ONLY | ocl::core::MEM_COPY_HOST_PTR,
                data.len(),
                Some(data),
            )
        }
        .unwrap()
    }
    fn read_mem_f32(&self, mem: &ocl::core::Mem, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        unsafe {
            ocl::core::enqueue_read_buffer(
                self.ocl().queue.as_core(),
                mem,
                true,
                0,
                &mut out,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        out
    }
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |a, &x| a.max(x.abs()))
}

fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut worst_i = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}] is not finite: {g} (want {w})");
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: max |Δ| = {worst} at [{worst_i}] (got {} want {}) > tol {tol}",
        got[worst_i],
        want[worst_i]
    );
}

#[test]
fn split_kv_decode_matches_q1_and_reference() {
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
        }
    };
    let arc: Arc<dyn Backend> = Arc::new(backend);
    let ocl_ref = arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCLBackend downcast");
    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl_ref.context.clone(),
        ocl_ref.queue.clone(),
        true,
    ));
    let gpu = Gpu { arc, memory };

    let mut cases = 0usize;
    for &(n_heads_q, n_heads_kv, head_dim) in &SHAPES {
        // Q scaled up so the running max moves several times per row.
        let q: Vec<f32> = lcg(n_heads_q * head_dim, 7)
            .into_iter()
            .map(|x| x * 3.0)
            .collect();
        let kv_total = n_heads_kv * CAPACITY * head_dim;
        let k16: Vec<u16> = lcg(kv_total, 11)
            .into_iter()
            .map(|x| half::f16::from_f32(x).to_bits())
            .collect();
        let v16: Vec<u16> = lcg(kv_total, 13)
            .into_iter()
            .map(|x| half::f16::from_f32(x).to_bits())
            .collect();

        let q_t = gpu.f32_tensor(&q, vec![1, 1, n_heads_q, head_dim]);
        let k_t = gpu.f16_tensor(&k16, vec![1, n_heads_kv, CAPACITY, head_dim]);
        let v_t = gpu.f16_tensor(&v16, vec![1, n_heads_kv, CAPACITY, head_dim]);
        let zero_o = vec![0.0f32; n_heads_q * head_dim];

        // Score buffer: one "layer" of [n_heads_q, CAPACITY], offset 0.
        let score_len = n_heads_q * CAPACITY;
        let sentinel_scores = vec![SENTINEL; score_len];

        for &n_kv in &N_KV {
            for ragged in [Ragged::Uniform, Ragged::Small, Ragged::Large] {
                let starts = kv_starts(ragged, n_heads_kv, n_kv);
                let (ref_o, ref_scores) = reference(
                    &q,
                    &k16,
                    &v16,
                    n_heads_q,
                    n_heads_kv,
                    head_dim,
                    n_kv,
                    starts.as_deref(),
                );
                let ref_o_scale = max_abs(&ref_o).max(1e-3);
                let start_mem = starts.as_ref().map(|s| gpu.mem_from_i32(s));

                for write_scores in [false, true] {
                    // --- control arm: original q1 kernel ---
                    let mut o_q1 = gpu.f32_tensor(&zero_o, vec![1, 1, n_heads_q, head_dim]);
                    let s_q1 = gpu.mem_from_f32(&sentinel_scores);
                    let ok = gpu
                        .ocl()
                        .flash_attention_decode_q1_gpu(
                            &q_t,
                            &k_t,
                            &v_t,
                            &mut o_q1,
                            n_heads_q,
                            n_heads_kv,
                            head_dim,
                            n_kv,
                            write_scores.then_some((&s_q1, 0, CAPACITY as i32)),
                            start_mem.as_ref(),
                        )
                        .expect("q1 dispatch");
                    if !ok {
                        eprintln!(
                            "Skipping shape {n_heads_q}/{n_heads_kv}/{head_dim}: q1 kernel unavailable on this host"
                        );
                        return;
                    }
                    gpu.arc.synchronize().unwrap();
                    let got_q1 = gpu.read_f32_tensor(&o_q1, n_heads_q * head_dim);
                    let tag = format!(
                        "shape {n_heads_q}/{n_heads_kv}/{head_dim} n_kv={n_kv} {ragged:?} scores={write_scores}"
                    );
                    assert_close(
                        &format!("q1 O {tag}"),
                        &got_q1,
                        &ref_o,
                        REL_TOL * ref_o_scale,
                    );
                    let scores_q1 = gpu.read_mem_f32(&s_q1, score_len);
                    if write_scores {
                        check_scores(&format!("q1 scores {tag}"), &scores_q1, &ref_scores, n_kv);
                    } else {
                        assert!(
                            scores_q1.iter().all(|&x| x == SENTINEL),
                            "q1 wrote scores with write_scores=0 ({tag})"
                        );
                    }

                    // --- test arm: split pair, every split count ---
                    for &n_splits in &SPLITS {
                        let mut o_sp = gpu.f32_tensor(&zero_o, vec![1, 1, n_heads_q, head_dim]);
                        let s_sp = gpu.mem_from_f32(&sentinel_scores);
                        let ok = gpu
                            .ocl()
                            .flash_attention_decode_split_gpu(
                                &q_t,
                                &k_t,
                                &v_t,
                                &mut o_sp,
                                n_heads_q,
                                n_heads_kv,
                                head_dim,
                                n_kv,
                                write_scores.then_some((&s_sp, 0, CAPACITY as i32)),
                                start_mem.as_ref(),
                                n_splits,
                            )
                            .expect("split dispatch");
                        assert!(
                            ok,
                            "split kernel must dispatch when q1 did ({tag} S={n_splits})"
                        );
                        gpu.arc.synchronize().unwrap();
                        let got_sp = gpu.read_f32_tensor(&o_sp, n_heads_q * head_dim);
                        let stag = format!("{tag} S={n_splits}");
                        assert_close(
                            &format!("split O vs ref {stag}"),
                            &got_sp,
                            &ref_o,
                            REL_TOL * ref_o_scale,
                        );
                        assert_close(
                            &format!("split O vs q1 {stag}"),
                            &got_sp,
                            &got_q1,
                            REL_TOL * ref_o_scale,
                        );
                        let scores_sp = gpu.read_mem_f32(&s_sp, score_len);
                        if write_scores {
                            check_scores(
                                &format!("split scores {stag}"),
                                &scores_sp,
                                &ref_scores,
                                n_kv,
                            );
                            assert_close(
                                &format!("split scores vs q1 {stag}"),
                                &scores_sp,
                                &scores_q1,
                                REL_TOL,
                            );
                        } else {
                            assert!(
                                scores_sp.iter().all(|&x| x == SENTINEL),
                                "split wrote scores with write_scores=0 ({stag})"
                            );
                        }

                        // Determinism: a second dispatch is bit-identical.
                        if n_splits == 16 && write_scores {
                            let mut o_again =
                                gpu.f32_tensor(&zero_o, vec![1, 1, n_heads_q, head_dim]);
                            let s_again = gpu.mem_from_f32(&sentinel_scores);
                            gpu.ocl()
                                .flash_attention_decode_split_gpu(
                                    &q_t,
                                    &k_t,
                                    &v_t,
                                    &mut o_again,
                                    n_heads_q,
                                    n_heads_kv,
                                    head_dim,
                                    n_kv,
                                    Some((&s_again, 0, CAPACITY as i32)),
                                    start_mem.as_ref(),
                                    n_splits,
                                )
                                .unwrap();
                            gpu.arc.synchronize().unwrap();
                            let again = gpu.read_f32_tensor(&o_again, n_heads_q * head_dim);
                            assert!(
                                again
                                    .iter()
                                    .zip(&got_sp)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "split O nondeterministic ({stag})"
                            );
                            let s_again_v = gpu.read_mem_f32(&s_again, score_len);
                            assert!(
                                s_again_v
                                    .iter()
                                    .zip(&scores_sp)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "split scores nondeterministic ({stag})"
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("split-KV parity: {cases} split cases passed");
    assert!(cases > 0);
}

/// Score row check: `[0, n_kv)` equals the reference (holes are 0 there),
/// `[n_kv, CAPACITY)` still holds the sentinel.
fn check_scores(what: &str, got: &[f32], want: &[Vec<f32>], n_kv: usize) {
    for (h, row) in want.iter().enumerate() {
        let g = &got[h * CAPACITY..(h + 1) * CAPACITY];
        assert_close(&format!("{what} head {h}"), &g[..n_kv], row, REL_TOL);
        assert!(
            g[n_kv..].iter().all(|&x| x == SENTINEL),
            "{what} head {h}: wrote beyond n_kv"
        );
    }
}
