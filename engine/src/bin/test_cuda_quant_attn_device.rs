//! CUDA device test — KIVI quantized-KV fused attention plugin (`CudaQuantAttnBackend`).
//!
//! The KIVI twin of `test_cuda_gpu_score_device`. Drives the force-linked `kivi_abi` CUDA backend on
//! the device and asserts numeric equivalence against a host f64 reference:
//!   A. residual-only attention (q2_tokens=0) — exercises the 3-pass softmax + block-wide barrier
//!      reductions + GQA head→kv mapping + F32 residual indexing (the highest-risk shared-memory part).
//!   B. Q2-only attention (res_tokens=0) with exactly-representable Q2 (dequant error = 0) —
//!      exercises `deq_q2` + the per-channel key / per-token value flush-interleaved index math.
//!   C. gather_update round-trip — the flat SeqMajor→head-first remap kernel.
//! Plus score readback (per-head softmax weights sum to 1 and match the reference).
//!
//! Proves the whole CUDA KIVI path on-device: find_cuda_quant_attn → from_raw_context (ManuallyDrop)
//! → nvcc→PTX (two modules) → launch_kernel (16-param attn, shared scratch) → DtoH score readback.
//! Requires a CUDA GPU. Run: `cargo run --no-default-features --features cuda --bin test_cuda_quant_attn_device`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("test_cuda_quant_attn_device requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use anyhow::anyhow;
    use argus_engine::backend::cuda_pc::CudaBackend;
    use argus_engine::buffer::DType;
    use argus_engine::memory::cuda::buffer::CudaDeviceBuffer;

    let be = CudaBackend::new()?;
    let cu_stream = be.context().default_stream().cu_stream() as *mut std::ffi::c_void;

    // Resolve + build the force-linked CUDA KIVI backend from the engine's live context.
    let reg = argus_extension_api::find_cuda_quant_attn("kivi_abi")
        .ok_or_else(|| anyhow!("kivi_abi not registered in CUDA_QUANT_ATTN_REGS"))?;
    let backend = be
        .with_cuda_quant_attn_make_args(|ma| (reg.make)(ma))
        .map_err(|e| anyhow!("kivi cuda make failed: {e}"))?;
    assert!(backend.has_quant_attn_kernel(2), "Q2 kernel present");
    assert!(backend.has_quant_attn_kernel(4), "Q4 kernel present");
    assert!(backend.has_quant_attn_kernel(8), "Q8 kernel present");
    println!("[make] kivi_abi CUDA backend built, Q2/Q4/Q8 kernels present");

    // Geometry: 2 Q heads, 1 KV head (gqa_ratio=2), head_dim=32, res_cap=32.
    let nhq = 2usize;
    let nhkv = 1usize;
    let hd = 32usize;
    let res_cap = 32usize;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let dev = |bytes: usize| -> anyhow::Result<CudaDeviceBuffer> {
        Ok(CudaDeviceBuffer::new(bytes.max(1), DType::F32)?)
    };
    let upload = |buf: &CudaDeviceBuffer, data: &[f32]| -> anyhow::Result<()> {
        buf.copy_from_host(data.as_ptr() as *const u8, std::mem::size_of_val(data))?;
        Ok(())
    };
    let upload_bytes = |buf: &CudaDeviceBuffer, data: &[u8]| -> anyhow::Result<()> {
        buf.copy_from_host(data.as_ptr(), data.len())?;
        Ok(())
    };
    let download = |buf: &CudaDeviceBuffer, out: &mut [f32]| -> anyhow::Result<()> {
        buf.copy_to_host(out.as_mut_ptr() as *mut u8, std::mem::size_of_val(out))?;
        Ok(())
    };

    // Host f64 reference: full softmax attention over `tokens` F32 KV vectors (one KV head).
    // `q` is this head's [hd] query. Returns (O[hd], weights[tokens]).
    let ref_attn = |q: &[f32], k: &[Vec<f32>], v: &[Vec<f32>]| -> (Vec<f32>, Vec<f64>) {
        let n = k.len();
        let mut scores = vec![0.0f64; n];
        let mut mx = f64::NEG_INFINITY;
        for t in 0..n {
            let mut s = 0.0f64;
            for d in 0..hd {
                s += q[d] as f64 * k[t][d] as f64;
            }
            s *= scale as f64;
            scores[t] = s;
            mx = mx.max(s);
        }
        let mut sum = 0.0f64;
        for s in &mut scores {
            *s = (*s - mx).exp();
            sum += *s;
        }
        let mut o = vec![0.0f32; hd];
        let mut w = vec![0.0f64; n];
        for t in 0..n {
            let wt = scores[t] / sum;
            w[t] = wt;
            for d in 0..hd {
                o[d] += (wt * v[t][d] as f64) as f32;
            }
        }
        (o, w)
    };

    // Q for both heads (distinct so a head→kv or head-offset bug shows).
    let mut q = vec![0.0f32; nhq * hd];
    for h in 0..nhq {
        for d in 0..hd {
            q[h * hd + d] = 0.1 * ((h * 7 + d) % 5) as f32 - 0.2;
        }
    }
    let q_buf = dev(nhq * hd * 4)?;
    upload(&q_buf, &q)?;

    // ── Test A: residual-only attention (q2_tokens=0, res_tokens=R) ──────────────────────────────
    {
        let r = 20usize;
        // res_k/res_v: [kv_heads=1, res_cap, hd].
        let mut rk = vec![0.0f32; nhkv * res_cap * hd];
        let mut rv = vec![0.0f32; nhkv * res_cap * hd];
        let mut k_tok: Vec<Vec<f32>> = Vec::new();
        let mut v_tok: Vec<Vec<f32>> = Vec::new();
        for t in 0..r {
            let mut kv = vec![0.0f32; hd];
            let mut vv = vec![0.0f32; hd];
            for d in 0..hd {
                kv[d] = 0.05 * ((t * 3 + d) % 7) as f32 - 0.15;
                vv[d] = 0.03 * ((t + d * 2) % 9) as f32;
                rk[t * hd + d] = kv[d];
                rv[t * hd + d] = vv[d];
            }
            k_tok.push(kv);
            v_tok.push(vv);
        }
        let rk_buf = dev(rk.len() * 4)?;
        let rv_buf = dev(rv.len() * 4)?;
        upload(&rk_buf, &rk)?;
        upload(&rv_buf, &rv)?;
        let dummy = dev(16)?; // q2_k/q2_v unused when q2_tokens=0.
        let o_buf = dev(nhq * hd * 4)?;
        let mut scores_host = vec![0.0f32; nhq * r];

        let args = argus_extension_api::CudaQuantAttnArgs {
            cu_stream,
            q_mem: q_buf.device_ptr(),
            qk_mem: dummy.device_ptr(),
            qv_mem: dummy.device_ptr(),
            res_k_mem: rk_buf.device_ptr(),
            res_v_mem: rv_buf.device_ptr(),
            out_mem: o_buf.device_ptr(),
            scores_out: scores_host.as_mut_ptr(),
            scores_len: scores_host.len(),
            num_heads_q: nhq,
            num_heads_kv: nhkv,
            head_dim: hd,
            q_tokens: 0,
            res_tokens: r,
            res_cap,
            scale,
            bits: 2,
        };
        let rc = backend.attention_gen_quant(&args);
        assert_eq!(rc, 0, "attention_gen_quant residual rc");
        let mut o = vec![0.0f32; nhq * hd];
        download(&o_buf, &mut o)?;

        for h in 0..nhq {
            let (o_ref, w_ref) = ref_attn(&q[h * hd..(h + 1) * hd], &k_tok, &v_tok);
            for d in 0..hd {
                let got = o[h * hd + d];
                let exp = o_ref[d];
                assert!((got - exp).abs() < 1e-4, "A: O[h{h},d{d}]={got} != {exp}");
            }
            let wsum: f64 = (0..r).map(|t| scores_host[h * r + t] as f64).sum();
            assert!(
                (wsum - 1.0).abs() < 1e-4,
                "A: head {h} weights sum {wsum} != 1"
            );
            for t in 0..r {
                assert!(
                    (scores_host[h * r + t] as f64 - w_ref[t]).abs() < 1e-4,
                    "A: weight[h{h},t{t}] mismatch"
                );
            }
        }
        println!("[A] residual-only attention + score readback: O & weights match host f64 ✓");
    }

    // ── Test B: Q2-only attention (res_tokens=0), exactly-representable Q2 (d=0.5, m=1.0) ─────────
    {
        let q2_tokens = 32usize; // exactly one group (groups_per_flush=1, one flush)
        let d_h = 0.5f32;
        let m_h = 1.0f32;
        let d_bits = f32_to_f16_bits(d_h);
        let m_bits = f32_to_f16_bits(m_h);
        // codes: K per (token, channel); V per (token, dim). Distinct so any index transposition shows.
        let code_k = |t: usize, ch: usize| -> u8 { ((t + ch) % 4) as u8 };
        let code_v = |t: usize, d: usize| -> u8 { ((t * 2 + d) % 4) as u8 };
        let deq = |code: u8| -> f32 { code as f32 * d_h + m_h };

        // q2_k: per-channel, flush-interleaved. One flush, one head, gif=0 → block_idx = ch (0..hd).
        // block ch holds the 32 tokens' values of channel ch.
        let mut q2_k = Vec::<u8>::new();
        for ch in 0..hd {
            let codes: Vec<u8> = (0..32).map(|t| code_k(t, ch)).collect();
            q2_k.extend_from_slice(&pack_q2_block(&codes, d_bits, m_bits));
        }
        // q2_v: per-token. One flush, one head, blocks_per_tok_v = hd/32 = 1 → block t holds token t's hd vec.
        let mut q2_v = Vec::<u8>::new();
        for t in 0..q2_tokens {
            let codes: Vec<u8> = (0..32).map(|d| code_v(t, d)).collect(); // hd==32
            q2_v.extend_from_slice(&pack_q2_block(&codes, d_bits, m_bits));
        }

        // Host reference KV (dequantized exactly).
        let mut k_tok: Vec<Vec<f32>> = Vec::new();
        let mut v_tok: Vec<Vec<f32>> = Vec::new();
        for t in 0..q2_tokens {
            k_tok.push((0..hd).map(|ch| deq(code_k(t, ch))).collect());
            v_tok.push((0..hd).map(|d| deq(code_v(t, d))).collect());
        }

        let qk_buf = dev(q2_k.len())?;
        let qv_buf = dev(q2_v.len())?;
        upload_bytes(&qk_buf, &q2_k)?;
        upload_bytes(&qv_buf, &q2_v)?;
        let dummy_res = dev(nhkv * res_cap * hd * 4)?; // res_tokens=0 → unread.
        let o_buf = dev(nhq * hd * 4)?;

        let args = argus_extension_api::CudaQuantAttnArgs {
            cu_stream,
            q_mem: q_buf.device_ptr(),
            qk_mem: qk_buf.device_ptr(),
            qv_mem: qv_buf.device_ptr(),
            res_k_mem: dummy_res.device_ptr(),
            res_v_mem: dummy_res.device_ptr(),
            out_mem: o_buf.device_ptr(),
            scores_out: std::ptr::null_mut(),
            scores_len: 0,
            num_heads_q: nhq,
            num_heads_kv: nhkv,
            head_dim: hd,
            q_tokens: q2_tokens,
            res_tokens: 0,
            res_cap,
            scale,
            bits: 2,
        };
        let rc = backend.attention_gen_quant(&args);
        assert_eq!(rc, 0, "attention_gen_quant q2 rc");
        let mut o = vec![0.0f32; nhq * hd];
        download(&o_buf, &mut o)?;

        for h in 0..nhq {
            let (o_ref, _) = ref_attn(&q[h * hd..(h + 1) * hd], &k_tok, &v_tok);
            for d in 0..hd {
                let got = o[h * hd + d];
                let exp = o_ref[d];
                assert!(
                    (got - exp).abs() < 1e-3,
                    "B: O[h{h},d{d}]={got} != {exp} (Q2 dequant/index)"
                );
            }
        }
        println!("[B] Q2-only attention (exact quant): O matches host f64 (deq_q2 + index math) ✓");
    }

    // ── Test C: gather_update round-trip (SeqMajor input → head-first residual) ───────────────────
    {
        let seq_len = 5usize;
        let res_pos = 3usize;
        // input: [seq_len, kv_heads, hd].
        let mut input = vec![0.0f32; seq_len * nhkv * hd];
        for s in 0..seq_len {
            for d in 0..hd {
                input[s * nhkv * hd + d] = (s * 100 + d) as f32;
            }
        }
        let in_buf = dev(input.len() * 4)?;
        upload(&in_buf, &input)?;
        let res_buf = dev(nhkv * res_cap * hd * 4)?; // zero-init
        let gargs = argus_extension_api::CudaQuantAttnGatherArgs {
            cu_stream,
            input_mem: in_buf.device_ptr(),
            residual_mem: res_buf.device_ptr(),
            kv_heads: nhkv,
            res_cap,
            head_dim: hd,
            seq_len,
            res_pos,
        };
        assert_eq!(backend.gather_update_quant(&gargs), 0, "gather rc");
        let mut res = vec![0.0f32; nhkv * res_cap * hd];
        download(&res_buf, &mut res)?;
        for s in 0..seq_len {
            for d in 0..hd {
                let got = res[(res_pos + s) * hd + d]; // kv_head 0: h*res_cap*hd == 0
                let exp = input[s * nhkv * hd + d];
                assert!(
                    (got - exp).abs() < 1e-6,
                    "C: residual[{}]={got} != {exp}",
                    (res_pos + s) * hd + d
                );
            }
        }
        println!("[C] gather_update round-trip: SeqMajor→head-first remap exact ✓");
    }

    // F16 device buffer + readback helpers (view buffers are F16).
    let dev_f16 = |elems: usize| -> anyhow::Result<CudaDeviceBuffer> {
        Ok(CudaDeviceBuffer::new((elems * 2).max(1), DType::F16)?)
    };
    let download_f16 = |buf: &CudaDeviceBuffer, elems: usize| -> anyhow::Result<Vec<f32>> {
        let mut raw = vec![0u16; elems];
        buf.copy_to_host(raw.as_mut_ptr() as *mut u8, elems * 2)?;
        Ok(raw.into_iter().map(f16_bits_to_f32).collect())
    };

    // ── Test D: scatter_residual (F32 ring → F16 view) — the short-prompt prefill view path ───────
    // Calls the plugin backend directly with raw device ptrs (like A/B/C), isolating the kernel.
    {
        let res_pos = 5usize;
        let tok_base = 0usize;
        let mut residual = vec![0.0f32; nhkv * res_cap * hd];
        for t in 0..res_pos {
            for d in 0..hd {
                residual[t * hd + d] = ((t * 4 + d) % 8) as f32 * 0.5; // {0,0.5,..,3.5} exact in f16
            }
        }
        let res_buf = dev(residual.len() * 4)?;
        upload(&res_buf, &residual)?;
        let view_elems = (tok_base + res_pos) * nhkv * hd;
        let view_buf = dev_f16(view_elems)?;
        let rc = backend.scatter_residual(&argus_extension_api::CudaQuantScatterResidualArgs {
            cu_stream,
            res_mem: res_buf.device_ptr(),
            attn_mem: view_buf.device_ptr(),
            kv_heads: nhkv,
            res_cap,
            head_dim: hd,
            res_pos,
            tok_base,
        });
        assert_eq!(rc, 0, "scatter_residual rc");
        let view = download_f16(&view_buf, view_elems)?;
        for t in 0..res_pos {
            for d in 0..hd {
                let got = view[(tok_base + t) * nhkv * hd + d];
                let exp = residual[t * hd + d];
                assert!(
                    (got - exp).abs() < 1e-3,
                    "D: view[t{t},d{d}]={got} != {exp} (scatter_residual)"
                );
            }
        }
        println!("[D] scatter_residual: F32 ring → F16 SeqMajor view exact ✓");
    }

    // ── Test E: dequant_flush_q2 (per-channel K → F16 view), one group of 32 tokens ───────────────
    {
        let d_h = 0.5f32;
        let m_h = 1.0f32;
        let d_bits = f32_to_f16_bits(d_h);
        let m_bits = f32_to_f16_bits(m_h);
        let code_k = |t: usize, ch: usize| -> u8 { ((t * 3 + ch) % 4) as u8 };
        let deq = |code: u8| -> f32 { code as f32 * d_h + m_h };
        let mut q2_k = Vec::<u8>::new();
        for ch in 0..hd {
            let codes: Vec<u8> = (0..32).map(|t| code_k(t, ch)).collect();
            q2_k.extend_from_slice(&pack_q2_block(&codes, d_bits, m_bits));
        }
        let qk_buf = dev(q2_k.len())?;
        qk_buf.copy_from_host(q2_k.as_ptr(), q2_k.len())?;
        let view_elems = 32 * nhkv * hd;
        let view_buf = dev_f16(view_elems)?;
        let rc = backend.dequant_flush(&argus_extension_api::CudaQuantDequantFlushArgs {
            cu_stream,
            q_blocks_mem: qk_buf.device_ptr(),
            attn_mem: view_buf.device_ptr(),
            kv_heads: nhkv,
            head_dim: hd,
            n_groups_or_tokens: 1, // groups_per_flush
            tok_base: 0,
            block_start: 0,
            bits: 2,
            is_key: true,
        });
        assert_eq!(rc, 0, "dequant_flush rc");
        let view = download_f16(&view_buf, view_elems)?;
        for t in 0..32 {
            for ch in 0..hd {
                let got = view[t * nhkv * hd + ch];
                let exp = deq(code_k(t, ch));
                assert!(
                    (got - exp).abs() < 1e-3,
                    "E: view[t{t},ch{ch}]={got} != {exp} (dequant_flush key)"
                );
            }
        }
        println!("[E] dequant_flush_q2 (per-channel key): Q2 → F16 SeqMajor view exact ✓");
    }

    println!("ALL PASS: CUDA KIVI quant-attn plugin verified on-device (kivi_abi)");
    Ok(())
}

/// IEEE-754 binary16 bits → f32.
#[cfg(feature = "cuda")]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal → normalize
            let mut e = -1i32;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x400 != 0 {
                    break;
                }
            }
            let f_exp = (127 - 15 - e) as u32;
            (sign << 31) | (f_exp << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let f_exp = exp + (127 - 15);
        (sign << 31) | (f_exp << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Round-to-nearest f32 → IEEE-754 binary16 bit pattern (test values are exact powers, so the
/// naive shift is sufficient; handles sign, normals, and zero).
#[cfg(feature = "cuda")]
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if exp <= 0 {
        return sign; // underflow → signed zero (fine for 0.0)
    }
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Pack 32 2-bit codes (+ f16 d/m header) into a 12-byte Q2_0 block: [d:f16][m:f16][qs:u8[8]],
/// 4 codes per byte LSB-first — the exact layout `deq_q2` reads.
#[cfg(feature = "cuda")]
fn pack_q2_block(codes: &[u8], d_bits: u16, m_bits: u16) -> [u8; 12] {
    let mut blk = [0u8; 12];
    blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
    blk[2..4].copy_from_slice(&m_bits.to_le_bytes());
    for i in 0..8 {
        let b = (codes[i * 4] & 3)
            | ((codes[i * 4 + 1] & 3) << 2)
            | ((codes[i * 4 + 2] & 3) << 4)
            | ((codes[i * 4 + 3] & 3) << 6);
        blk[4 + i] = b;
    }
    blk
}
