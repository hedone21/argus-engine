//! Tests for the NEON flash attention **prefill** path (Step 3).
//!
//! Step 3 of the long-context optimization plan introduces a NEON tile +
//! online-softmax prefill kernel (`flash_prefill_forward_f32_neon`) that
//! `layers::attention::flash_attention_forward_strided` dispatches to on
//! aarch64 when:
//!   * `q_len >= PREFILL_FLASH_THRESHOLD` (default 128)
//!   * KV is HeadMajor (`kv_head_stride == capacity * head_dim`)
//!   * `head_dim % 8 == 0`
//!   * `n_heads_q % n_heads_kv == 0`
//!
//! Otherwise the existing scalar rayon path is preserved for correctness
//! (short prefill, odd head_dim, SeqMajor, etc.).
//!
//! The reference used for correctness is an inline F32 3-pass attention
//! (Q·K^T → softmax → P·V) with explicit causal masking and an optional
//! sliding window.  Permitted error: NMSE < 1e-4.
//!
//! The tests compile on all architectures — on non-aarch64 the NEON path
//! is absent and `flash_attention_forward_strided` uses the existing
//! scalar rayon path, still producing correct output.  On aarch64 the
//! same tests exercise the NEON dispatch at `q_len >= THRESHOLD`.

use argus_engine::layers::attention::flash_attention_forward_strided;
use std::sync::Mutex;

// The NEON prefill path (`flash_prefill_forward_f32_neon`) drives the
// process-global SpinPool.  That pool is a single-producer dispatcher —
// concurrent `dispatch` calls from different test threads race each
// other's shared work-ctx state.  Serialise dispatch-heavy flash cases
// behind this mutex so the test runner's default parallel execution
// stays deterministic on aarch64.
static DISPATCH_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random F32 values in [-1, 1).
fn gen_data(n: usize, seed: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        data.push(((s >> 16) as i16 as f32) / 32768.0);
    }
    data
}

/// Reference F32 attention, 3-pass (no tiling), causal + optional sliding
/// window.  KV is HeadMajor `[kv_heads, capacity, head_dim]`.
///
/// Q layout: `[q_len, n_heads_q, head_dim]` (SeqMajor Q rows).
/// Out layout: `[q_len, n_heads_q, head_dim]`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn reference_prefill_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    capacity: usize,
    q_start_pos: usize,
    window_size: Option<usize>,
) {
    let gqa = n_heads_q / n_heads_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_row_stride = n_heads_q * head_dim;
    let out_row_stride = n_heads_q * head_dim;

    for r in 0..q_len {
        let global_r = r + q_start_pos;
        for q_h in 0..n_heads_q {
            let kv_h = q_h / gqa;
            let q_off = r * q_row_stride + q_h * head_dim;
            let q_vec = &q[q_off..q_off + head_dim];

            // Pass 1: raw scores (causal + window masked → NEG_INFINITY).
            let mut scores = vec![f32::NEG_INFINITY; kv_len];
            for t in 0..kv_len {
                let causal_ok = t <= global_r;
                let window_ok = match window_size {
                    Some(ws) => t + ws > global_r,
                    None => true,
                };
                if !causal_ok || !window_ok {
                    continue;
                }
                let k_off = kv_h * capacity * head_dim + t * head_dim;
                let k_vec = &k[k_off..k_off + head_dim];
                let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                let s = dot * scale;
                scores[t] = if s.is_finite() { s } else { f32::NEG_INFINITY };
            }

            // Pass 2: softmax.  All-masked row → zero output (no valid keys).
            let m_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let out_off = r * out_row_stride + q_h * head_dim;
            if m_max == f32::NEG_INFINITY {
                for d in 0..head_dim {
                    out[out_off + d] = 0.0;
                }
                continue;
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                if *s == f32::NEG_INFINITY {
                    *s = 0.0;
                } else {
                    *s = (*s - m_max).exp();
                    sum += *s;
                }
            }
            if sum <= 0.0 || !sum.is_finite() {
                for d in 0..head_dim {
                    out[out_off + d] = 0.0;
                }
                continue;
            }
            let inv = 1.0 / sum;
            for s in scores.iter_mut() {
                *s *= inv;
            }

            // Pass 3: weighted V sum.
            let o = &mut out[out_off..out_off + head_dim];
            for x in o.iter_mut() {
                *x = 0.0;
            }
            for t in 0..kv_len {
                let w = scores[t];
                if w == 0.0 {
                    continue;
                }
                let v_off = kv_h * capacity * head_dim + t * head_dim;
                let v_vec = &v[v_off..v_off + head_dim];
                for d in 0..head_dim {
                    o[d] += w * v_vec[d];
                }
            }
        }
    }
}

fn nmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (*x as f64) - (*y as f64);
        num += diff * diff;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        num as f32
    } else {
        (num / den) as f32
    }
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_case(
    q_len: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    q_start_pos: usize,
    window_size: Option<usize>,
    br: usize,
    bc: usize,
    seed: u32,
) -> f32 {
    let kv_len = q_start_pos + q_len;
    // Allocate KV with capacity = kv_len (no padding) for simplicity.
    let capacity = kv_len.max(1);
    let q = gen_data(q_len * n_heads_q * head_dim, seed.wrapping_add(1));
    let k = gen_data(n_heads_kv * capacity * head_dim, seed.wrapping_add(2));
    let v = gen_data(n_heads_kv * capacity * head_dim, seed.wrapping_add(3));

    let out_len = q_len * n_heads_q * head_dim;
    let mut out_ref = vec![0.0f32; out_len];
    let mut out_flash = vec![0.0f32; out_len];

    reference_prefill_attention(
        &q,
        &k,
        &v,
        &mut out_ref,
        n_heads_q,
        n_heads_kv,
        q_len,
        kv_len,
        head_dim,
        capacity,
        q_start_pos,
        window_size,
    );

    let q_row_stride = n_heads_q * head_dim;
    let out_row_stride = q_row_stride;
    let k_pos_stride = head_dim;
    let v_pos_stride = head_dim;
    let kv_head_stride = capacity * head_dim;

    {
        let _guard = DISPATCH_LOCK.lock().unwrap();
        flash_attention_forward_strided(
            &q,
            &k,
            &v,
            &mut out_flash,
            n_heads_q,
            n_heads_kv,
            q_len,
            kv_len,
            head_dim,
            q_row_stride,
            k_pos_stride,
            v_pos_stride,
            out_row_stride,
            kv_head_stride,
            q_start_pos,
            br,
            bc,
            window_size,
        );
    }

    nmse(&out_flash, &out_ref)
}

// ---------------------------------------------------------------------------
// Baseline: small seq_len exercises scalar fallback (< THRESHOLD).
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_baseline_scalar_small_32() {
    let err = run_prefill_case(32, 4, 2, 64, 0, None, 32, 32, 0x10);
    assert!(err < 1e-5, "scalar small path NMSE {err}");
}

#[test]
fn test_prefill_baseline_scalar_small_64() {
    let err = run_prefill_case(64, 4, 2, 64, 0, None, 32, 32, 0x11);
    assert!(err < 1e-5, "scalar small path NMSE {err}");
}

#[test]
fn test_prefill_baseline_scalar_small_127() {
    // Exactly below THRESHOLD — scalar path.
    let err = run_prefill_case(127, 4, 2, 64, 0, None, 32, 32, 0x12);
    assert!(err < 1e-5, "scalar small path NMSE {err}");
}

// ---------------------------------------------------------------------------
// Flash path: seq_len >= THRESHOLD triggers NEON prefill kernel.
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_flash_128_hd64_gqa2() {
    let err = run_prefill_case(128, 4, 2, 64, 0, None, 32, 32, 0x20);
    assert!(err < 1e-4, "flash NMSE {err}");
}

#[test]
fn test_prefill_flash_129_hd64_gqa2() {
    // Non-tile-multiple seq_len.
    let err = run_prefill_case(129, 4, 2, 64, 0, None, 32, 32, 0x21);
    assert!(err < 1e-4, "flash NMSE {err}");
}

#[test]
fn test_prefill_flash_256_hd64_gqa4() {
    let err = run_prefill_case(256, 8, 2, 64, 0, None, 32, 32, 0x22);
    assert!(err < 1e-4, "flash NMSE {err}");
}

#[test]
fn test_prefill_flash_512_hd64_gqa8() {
    let err = run_prefill_case(512, 32, 8, 64, 0, None, 32, 32, 0x23);
    assert!(err < 1e-4, "flash NMSE {err}");
}

#[test]
fn test_prefill_flash_1024_hd128_gqa6() {
    let err = run_prefill_case(1024, 12, 2, 128, 0, None, 32, 32, 0x24);
    assert!(err < 1e-4, "flash NMSE {err}");
}

#[test]
fn test_prefill_flash_2048_hd128_gqa6() {
    let err = run_prefill_case(2048, 12, 2, 128, 0, None, 32, 32, 0x25);
    assert!(err < 1e-4, "flash NMSE {err}");
}

// Full 4K prefill is the target case for Step 3.  Kept at gqa=6 (Qwen shape).
#[test]
fn test_prefill_flash_4096_hd128_gqa6() {
    let err = run_prefill_case(4096, 12, 2, 128, 0, None, 32, 32, 0x26);
    assert!(err < 1e-4, "flash NMSE {err}");
}

// ---------------------------------------------------------------------------
// Tile boundary cases.
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_flash_tile_boundary_br_plus_1() {
    // br=64 default for head_dim=64, test seq_len = br+1 but above THRESHOLD.
    let err = run_prefill_case(129, 4, 2, 64, 0, None, 32, 32, 0x30);
    assert!(err < 1e-4, "tile-br+1 NMSE {err}");
}

#[test]
fn test_prefill_flash_tile_boundary_2br_minus_1() {
    // seq_len = 2*64-1 = 127 (but under threshold) → use 2*64+65 = 193.
    let err = run_prefill_case(193, 4, 2, 64, 0, None, 32, 32, 0x31);
    assert!(err < 1e-4, "tile-2br-1 NMSE {err}");
}

#[test]
fn test_prefill_flash_tile_boundary_3br_plus_13() {
    let err = run_prefill_case(3 * 64 + 13, 4, 2, 64, 0, None, 32, 32, 0x32);
    assert!(err < 1e-4, "tile-3br+13 NMSE {err}");
}

#[test]
fn test_prefill_flash_tile_boundary_hd128_seq_200() {
    // head_dim=128 (tile is br=64 for hd>=96), seq_len 200 is not a multiple.
    let err = run_prefill_case(200, 12, 2, 128, 0, None, 32, 32, 0x33);
    assert!(err < 1e-4, "hd128 seq 200 NMSE {err}");
}

// ---------------------------------------------------------------------------
// GQA ratio matrix.
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_flash_gqa_1() {
    // n_heads_q == n_heads_kv.
    let err = run_prefill_case(256, 4, 4, 64, 0, None, 32, 32, 0x40);
    assert!(err < 1e-4, "gqa=1 NMSE {err}");
}

#[test]
fn test_prefill_flash_gqa_2() {
    let err = run_prefill_case(256, 4, 2, 64, 0, None, 32, 32, 0x41);
    assert!(err < 1e-4, "gqa=2 NMSE {err}");
}

#[test]
fn test_prefill_flash_gqa_4() {
    let err = run_prefill_case(256, 8, 2, 64, 0, None, 32, 32, 0x42);
    assert!(err < 1e-4, "gqa=4 NMSE {err}");
}

#[test]
fn test_prefill_flash_gqa_6() {
    let err = run_prefill_case(256, 12, 2, 128, 0, None, 32, 32, 0x43);
    assert!(err < 1e-4, "gqa=6 NMSE {err}");
}

#[test]
fn test_prefill_flash_gqa_8() {
    let err = run_prefill_case(256, 32, 4, 64, 0, None, 32, 32, 0x44);
    assert!(err < 1e-4, "gqa=8 NMSE {err}");
}

// ---------------------------------------------------------------------------
// Sliding window (Gemma3 local-attention) combined with causal mask.
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_flash_window_small() {
    let err = run_prefill_case(512, 4, 2, 64, 0, Some(64), 32, 32, 0x50);
    assert!(err < 1e-4, "window NMSE {err}");
}

#[test]
fn test_prefill_flash_window_wider_than_seq() {
    let err = run_prefill_case(256, 4, 2, 64, 0, Some(1024), 32, 32, 0x51);
    assert!(err < 1e-4, "wide window NMSE {err}");
}

// ---------------------------------------------------------------------------
// Non-zero q_start_pos (partial prefill simulating continuation).  This is
// not actively used in the production path (prefill starts at pos 0), but
// the code path must be correct for the shared helper.
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_flash_nonzero_q_start_pos() {
    let err = run_prefill_case(256, 4, 2, 64, 16, None, 32, 32, 0x60);
    assert!(err < 1e-4, "q_start_pos NMSE {err}");
}

// ---------------------------------------------------------------------------
// NaN injection: a single NaN in Q should not propagate beyond its query
// row; every other row is expected to be finite.
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_flash_nan_inject() {
    let q_len = 256;
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let q_start_pos = 0;
    let kv_len = q_start_pos + q_len;
    let capacity = kv_len;

    let mut q = gen_data(q_len * n_heads_q * head_dim, 0x70);
    let k = gen_data(n_heads_kv * capacity * head_dim, 0x71);
    let v = gen_data(n_heads_kv * capacity * head_dim, 0x72);

    // Inject NaN into one K position — the row's scores become non-finite
    // and should be treated as NEG_INFINITY by both the reference and the
    // flash path (Step 2 semantics).
    q[3 * (n_heads_q * head_dim) + 2 * head_dim] = f32::NAN;

    let mut out_flash = vec![0.0f32; q_len * n_heads_q * head_dim];

    {
        let _guard = DISPATCH_LOCK.lock().unwrap();
        flash_attention_forward_strided(
            &q,
            &k,
            &v,
            &mut out_flash,
            n_heads_q,
            n_heads_kv,
            q_len,
            kv_len,
            head_dim,
            n_heads_q * head_dim,
            head_dim,
            head_dim,
            n_heads_q * head_dim,
            capacity * head_dim,
            q_start_pos,
            32,
            32,
            None,
        );
    }

    // All other rows must remain finite.
    let q_row_stride = n_heads_q * head_dim;
    for r in 0..q_len {
        if r == 3 {
            // Skip the row that received the NaN input.
            continue;
        }
        let row = &out_flash[r * q_row_stride..(r + 1) * q_row_stride];
        for (i, &x) in row.iter().enumerate() {
            assert!(x.is_finite(), "row {r} lane {i} not finite (got {x})");
        }
    }
}

// ---------------------------------------------------------------------------
// Causal mask semantics: upper-triangular contribution must be zero.  We
// construct a case where all "future" keys are set so extreme that if they
// were not masked, they would dominate the softmax.  The flash output must
// match the reference (which zeroes them via mask).
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_flash_causal_mask_strict() {
    let q_len = 256;
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let kv_len = q_len;
    let capacity = kv_len;

    let q = gen_data(q_len * n_heads_q * head_dim, 0x80);
    let mut k = gen_data(n_heads_kv * capacity * head_dim, 0x81);
    let v = gen_data(n_heads_kv * capacity * head_dim, 0x82);

    // Poison the "future" keys for position 0 of the query.  If the mask
    // is broken, the tokens at t > 0 will leak into out[0, :, :].
    for kv_h in 0..n_heads_kv {
        for t in 1..kv_len {
            for d in 0..head_dim {
                k[kv_h * capacity * head_dim + t * head_dim + d] = 100.0;
            }
        }
    }

    let mut out_ref = vec![0.0f32; q_len * n_heads_q * head_dim];
    let mut out_flash = vec![0.0f32; q_len * n_heads_q * head_dim];

    reference_prefill_attention(
        &q,
        &k,
        &v,
        &mut out_ref,
        n_heads_q,
        n_heads_kv,
        q_len,
        kv_len,
        head_dim,
        capacity,
        0,
        None,
    );

    {
        let _guard = DISPATCH_LOCK.lock().unwrap();
        flash_attention_forward_strided(
            &q,
            &k,
            &v,
            &mut out_flash,
            n_heads_q,
            n_heads_kv,
            q_len,
            kv_len,
            head_dim,
            n_heads_q * head_dim,
            head_dim,
            head_dim,
            n_heads_q * head_dim,
            capacity * head_dim,
            0,
            32,
            32,
            None,
        );
    }

    let err = nmse(&out_flash, &out_ref);
    assert!(err < 1e-4, "causal strict NMSE {err}");

    // Explicitly check row 0: only key t=0 may contribute, so the output
    // must equal V[:, 0, :] up to scale.
    // Row 0, head 0, kv_h 0.
    let q_stride = n_heads_q * head_dim;
    let out_row0 = &out_flash[..q_stride];
    for q_h in 0..n_heads_q {
        let kv_h = q_h / (n_heads_q / n_heads_kv);
        let o = &out_row0[q_h * head_dim..(q_h + 1) * head_dim];
        let v0 = &v[kv_h * capacity * head_dim..kv_h * capacity * head_dim + head_dim];
        for d in 0..head_dim {
            let diff = (o[d] - v0[d]).abs();
            assert!(
                diff < 1e-3,
                "row0 h{q_h} lane{d}: out {} vs V0 {} (diff {diff})",
                o[d],
                v0[d]
            );
        }
    }
}
