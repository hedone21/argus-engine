use rayon::prelude::*;

/// Naive Standard Attention implementation for verification purposes.
///
/// Computes: Softmax(Q * K^T / sqrt(d)) * V
///
/// Arguments:
/// - `q`, `k`, `v`: [seq_len, head_dim] flat buffers (for a single head)
/// - `out`: [seq_len, head_dim] output buffer
/// - `seq_len`: Sequence length (N)
/// - `head_dim`: Head dimension (D) (e.g. 64 or 128)
pub fn naive_attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Q * K^T -> Scores [N, N]
    let mut scores = vec![0.0; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }

    // Softmax per row
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row_end = row_start + seq_len;
        let row = &mut scores[row_start..row_end];

        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        for val in row.iter_mut() {
            *val /= sum;
        }
    }

    // Scores * V -> Out [N, D]
    for i in 0..seq_len {
        for d in 0..head_dim {
            let mut sum = 0.0;
            for j in 0..seq_len {
                let s = scores[i * seq_len + j];
                let v_val = v[j * head_dim + d];
                sum += s * v_val;
            }
            out[i * head_dim + d] = sum;
        }
    }
}

/// Flash Attention implementation (Forward pass)
///
/// Uses tiling and online softmax to avoid O(N^2) memory usage.
/// Supports causal masking, arbitrary strides, and optional sliding window.
///
/// Arguments:
/// - `q_stride`, `k_stride`, `v_stride`, `out_stride`: Stride (in elements) between consecutive rows (tokens).
/// - `q_start_pos`: The global position of the first query token (for causal masking check `c <= r + q_start_pos`).
/// - `window_size`: Optional sliding window size. When `Some(ws)`, keys at global position
///   `global_c < (global_r + q_start_pos).saturating_sub(ws - 1)` are masked out (`-inf`).
///   `None` = full causal attention (default).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn flash_attention_head(
    q: &[f32],
    q_stride: usize,
    k: &[f32],
    k_stride: usize,
    v: &[f32],
    v_stride: usize,
    out: &mut [f32],
    out_stride: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    q_start_pos: usize,
    br: usize,
    bc: usize,
    window_size: Option<usize>,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let tr = q_len.div_ceil(br);
    let tc = kv_len.div_ceil(bc);

    for i in 0..tr {
        let r_start = i * br;
        let r_end = std::cmp::min(r_start + br, q_len);
        let cur_br = r_end - r_start;

        let mut oi = vec![0.0; cur_br * head_dim];
        let mut li = vec![0.0; cur_br];
        let mut mi = vec![f32::NEG_INFINITY; cur_br];

        for j in 0..tc {
            let c_start = j * bc;
            let c_end = std::cmp::min(c_start + bc, kv_len);
            let cur_bc = c_end - c_start;

            // Optimization: Causal Masking Check
            // c_start is the first key index in this block.
            // r_end - 1 is the last query LOCAL index in this block.
            // Global Q index: (r_end - 1) + q_start_pos.
            // We can skip block if c_start > (r_end - 1) + q_start_pos.
            let max_global_q_in_block = (r_end - 1) + q_start_pos;
            if c_start > max_global_q_in_block {
                continue;
            }
            // Sliding window: skip block if all keys are outside the window.
            // Block keys span [c_start, c_end). Min global_q in block = r_start + q_start_pos.
            // All keys out of window iff c_end + window_size - 1 <= r_start + q_start_pos.
            if let Some(ws) = window_size {
                let min_global_q_in_block = r_start + q_start_pos;
                if c_end + ws <= min_global_q_in_block + 1 {
                    continue;
                }
            }

            for r in 0..cur_br {
                let global_r = r_start + r;
                let mut row_max = f32::NEG_INFINITY;
                let mut sij = vec![f32::NEG_INFINITY; cur_bc];
                let mut any_valid = false;

                let global_q = global_r + q_start_pos;
                for c in 0..cur_bc {
                    let global_c = c_start + c;

                    let causal_ok = global_c <= global_q;
                    let window_ok = match window_size {
                        Some(ws) => global_c + ws > global_q,
                        None => true,
                    };

                    if causal_ok && window_ok {
                        let mut dot = 0.0;
                        let q_ptr = global_r * q_stride;
                        let k_ptr = global_c * k_stride;

                        for d in 0..head_dim {
                            dot += q[q_ptr + d] * k[k_ptr + d];
                        }
                        let val = dot * scale;
                        sij[c] = val;
                        if val > row_max {
                            row_max = val;
                        }
                        any_valid = true;
                    }
                }

                if !any_valid {
                    continue;
                }

                let m_prev = mi[r];
                let m_new = m_prev.max(row_max);

                let mut p_row = vec![0.0; cur_bc];
                let mut l_new_part = 0.0;

                for c in 0..cur_bc {
                    if sij[c] != f32::NEG_INFINITY {
                        p_row[c] = (sij[c] - m_new).exp();
                        l_new_part += p_row[c];
                    }
                }

                let alpha = if m_prev == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m_prev - m_new).exp()
                };

                li[r] = li[r] * alpha + l_new_part;
                mi[r] = m_new;

                for d in 0..head_dim {
                    let mut pv_sum = 0.0;
                    for c in 0..cur_bc {
                        let global_c = c_start + c;
                        let causal_ok = global_c <= global_q;
                        let window_ok = match window_size {
                            Some(ws) => global_c + ws > global_q,
                            None => true,
                        };
                        if causal_ok && window_ok {
                            // v[global_c, d] -> v[global_c * v_stride + d]
                            pv_sum += p_row[c] * v[global_c * v_stride + d];
                        }
                    }
                    oi[r * head_dim + d] = oi[r * head_dim + d] * alpha + pv_sum;
                }
            }
        }

        for r in 0..cur_br {
            let global_r = r_start + r;
            let l_val = li[r];
            let inv_l = if l_val == 0.0 { 0.0 } else { 1.0 / l_val };

            let out_ptr = global_r * out_stride;
            for d in 0..head_dim {
                out[out_ptr + d] = oi[r * head_dim + d] * inv_l;
            }
        }
    }
}

use std::sync::atomic::{AtomicPtr, Ordering};

/// Multi-head attention using Flash Attention with explicit `kv_head_stride`
/// for layout-awareness.
/// - SeqMajor: kv_head_stride = head_dim
/// - HeadMajor: kv_head_stride = capacity * head_dim
///
/// `window_size`: Optional sliding window size. `None` = full causal attention.
#[allow(clippy::too_many_arguments)]
// LAYER-EXEMPT: backend_concrete_downcast — §13.8-L hot-path NEON fallback dispatch
pub fn flash_attention_forward_strided(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    q_stride: usize,
    k_stride: usize,
    v_stride: usize,
    out_stride: usize,
    kv_head_stride: usize,
    q_start_pos: usize,
    br: usize,
    bc: usize,
    window_size: Option<usize>,
) {
    // Step 3: NEON tile + online-softmax prefill path.  Used when the input
    // is long enough to amortise dispatch overhead, head_dim is 8-aligned,
    // the GQA ratio is integer, and KV is HeadMajor (kv_head_stride equals
    // a full KV row × capacity).  `k_stride == head_dim` identifies the
    // HeadMajor case (SeqMajor would set k_stride = n_heads_kv * head_dim).
    //
    // All other combinations fall through to the scalar rayon path below.
    #[cfg(target_arch = "aarch64")]
    {
        use crate::backend::cpu::neon::{PREFILL_FLASH_THRESHOLD, flash_prefill_forward_f32_neon};
        if q_len >= PREFILL_FLASH_THRESHOLD
            && head_dim.is_multiple_of(8)
            && n_heads_q.is_multiple_of(n_heads_kv)
            && k_stride == head_dim
            && v_stride == head_dim
            && q_stride == n_heads_q * head_dim
            && out_stride == n_heads_q * head_dim
            && kv_head_stride.is_multiple_of(head_dim)
        {
            let _ = (br, bc); // advisory — the NEON path picks its own tile.
            flash_prefill_forward_f32_neon(
                q,
                k,
                v,
                out,
                n_heads_q,
                n_heads_kv,
                q_len,
                kv_len,
                head_dim,
                q_stride,
                k_stride,
                v_stride,
                out_stride,
                kv_head_stride,
                q_start_pos,
                window_size,
            );
            return;
        }
    }

    let n_rep = n_heads_q / n_heads_kv;

    // We need to extract pointers BEFORE the closure to avoid capturing `&mut [f32]` which cannot be shared.
    // Use AtomicPtr which is Send + Sync.
    let out_ptr_raw = AtomicPtr::new(out.as_mut_ptr());
    let q_ptr_raw = AtomicPtr::new(q.as_ptr() as *mut f32);
    let k_ptr_raw = AtomicPtr::new(k.as_ptr() as *mut f32);
    let v_ptr_raw = AtomicPtr::new(v.as_ptr() as *mut f32);

    // Re-do with capturing lengths
    let q_total_len = q.len();
    let k_total_len = k.len(); // v len same
    let out_total_len = out.len();

    (0..n_heads_q).into_par_iter().for_each(move |h| {
        let kv_h = h / n_rep;
        let q_head_offset = h * head_dim;
        let k_head_offset = kv_h * kv_head_stride;
        let out_head_offset = h * head_dim;

        let q_ptr = q_ptr_raw.load(Ordering::Relaxed);
        let k_ptr = k_ptr_raw.load(Ordering::Relaxed);
        let v_ptr = v_ptr_raw.load(Ordering::Relaxed);
        let out_ptr = out_ptr_raw.load(Ordering::Relaxed);

        unsafe {
            let q_slice =
                std::slice::from_raw_parts(q_ptr.add(q_head_offset), q_total_len - q_head_offset);
            let k_slice =
                std::slice::from_raw_parts(k_ptr.add(k_head_offset), k_total_len - k_head_offset);
            let v_slice =
                std::slice::from_raw_parts(v_ptr.add(k_head_offset), k_total_len - k_head_offset);
            let out_slice = std::slice::from_raw_parts_mut(
                out_ptr.add(out_head_offset),
                out_total_len - out_head_offset,
            );

            flash_attention_head(
                q_slice,
                q_stride,
                k_slice,
                k_stride,
                v_slice,
                v_stride,
                out_slice,
                out_stride,
                q_len,
                kv_len,
                head_dim,
                q_start_pos,
                br,
                bc,
                window_size,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() > tol {
                panic!(
                    "Mismatch at index {}: {} vs {} (diff {})",
                    i,
                    a[i],
                    b[i],
                    (a[i] - b[i]).abs()
                );
            }
        }
    }

    #[test]
    fn test_naive_attention_sanity() {
        let q = vec![1.0];
        let k = vec![1.0];
        let v = vec![2.0];
        let mut out = vec![0.0];

        naive_attention_head(&q, &k, &v, &mut out, 1, 1);

        assert!((out[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_flash_attention_vs_naive() {
        let mut rng = rand::rng();
        let seq_len = 128; // Small enough for test
        let head_dim = 32;
        let br = 32;
        let bc = 32;

        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|_| rng.random::<f32>())
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|_| rng.random::<f32>())
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|_| rng.random::<f32>())
            .collect();

        let mut out_naive = vec![0.0; seq_len * head_dim];
        let mut out_flash = vec![0.0; seq_len * head_dim];

        // Simple case: Strid = head_dim (Contiguous row major)
        let stride = head_dim;

        #[allow(non_snake_case)]
        fn causal_naive(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], N: usize, D: usize) {
            let scale = 1.0 / (D as f32).sqrt();
            for i in 0..N {
                let mut row_scores = vec![f32::NEG_INFINITY; N];
                for j in 0..N {
                    if j <= i {
                        // Causal Mask
                        let mut dot = 0.0;
                        for d in 0..D {
                            dot += q[i * D + d] * k[j * D + d];
                        }
                        row_scores[j] = dot * scale;
                    }
                }

                let max_val = row_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0;
                for s in row_scores.iter_mut() {
                    if *s != f32::NEG_INFINITY {
                        *s = (*s - max_val).exp();
                        sum_exp += *s;
                    } else {
                        *s = 0.0;
                    }
                }
                for s in row_scores.iter_mut() {
                    *s /= sum_exp;
                }

                for d in 0..D {
                    let mut s_val = 0.0;
                    for j in 0..N {
                        s_val += row_scores[j] * v[j * D + d];
                    }
                    out[i * D + d] = s_val;
                }
            }
        }

        causal_naive(&q, &k, &v, &mut out_naive, seq_len, head_dim);

        flash_attention_head(
            &q,
            stride,
            &k,
            stride,
            &v,
            stride,
            &mut out_flash,
            stride,
            seq_len,
            seq_len,
            head_dim,
            0,
            br,
            bc,
            None,
        );

        // Floating point error accumulates slightly differently due to online softmax vs standard softmax
        // 1e-4 should be safe.
        approx_eq(&out_naive, &out_flash, 1e-4);
    }

    #[test]
    fn test_identity_qk_reproduces_v() {
        // When Q == K (identical), attention weights should be uniform (1/N for each position)
        // For seq_len=1, this means out == V exactly
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![3.0, 7.0, 11.0, 13.0];
        let mut out = vec![0.0; head_dim];

        naive_attention_head(&q, &k, &v, &mut out, 1, head_dim);
        approx_eq(&out, &v, 1e-6);
    }

    #[test]
    fn test_flash_attention_single_token() {
        // Single token: flash attention should produce same result as naive
        let head_dim = 8;
        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..head_dim).map(|i| (i as f32) + 1.0).collect();
        let mut out = vec![0.0; head_dim];

        flash_attention_head(
            &q, head_dim, &k, head_dim, &v, head_dim, &mut out, head_dim, 1, 1, head_dim, 0, 16,
            16, None,
        );

        // With seq_len=1, out should exactly equal V
        approx_eq(&out, &v, 1e-6);
    }

    #[test]
    fn test_flash_attention_decode_causal_mask() {
        // Decode mode: q_len=1, kv_len=4, q_start_pos=3
        // The query at global position 3 can attend to keys at positions 0,1,2,3
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let v = vec![
            10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0,
        ];
        let mut out = vec![0.0; head_dim];

        flash_attention_head(
            &q, head_dim, &k, head_dim, &v, head_dim, &mut out, head_dim, 1, 4, head_dim,
            3, // q_start_pos=3 → can attend to all 4 KV positions
            4, 4, None,
        );

        // With q_start_pos=3 and q_len=1, causal mask allows positions 0..3.
        // Q=[1,0,0,0] dot K[0]=[1,0,0,0] = 1.0, others = 0.0
        // After softmax(scale=0.5): w0 ≈ 0.3547, w1=w2=w3 ≈ 0.2151
        // V values are 10x identity, so output reflects weighted contributions:
        //   out[0] = 0.3547 * 10 ≈ 3.547 (highest weight)
        //   out[d] = 0.2151 * 10 ≈ 2.151 for d=1,2,3
        assert!(
            out.iter().all(|&v| v.is_finite()),
            "All outputs should be finite"
        );
        // out[0] has the highest weight contribution
        assert!(
            out[0] > out[1],
            "out[0]={} should > out[1]={}",
            out[0],
            out[1]
        );

        // Now test restricted causal: q_start_pos=1, only positions 0,1 are visible
        let mut out2 = vec![0.0; head_dim];
        flash_attention_head(
            &q, head_dim, &k, head_dim, &v, head_dim, &mut out2, head_dim, 1, 4, head_dim,
            1, // q_start_pos=1 → can see positions 0,1 only
            4, 4, None,
        );
        // Positions 2,3 are masked. V[2] and V[3] should not contribute.
        assert!(
            out2[2].abs() < 1e-6,
            "Position 2 should be masked: out2[2]={}",
            out2[2]
        );
        assert!(
            out2[3].abs() < 1e-6,
            "Position 3 should be masked: out2[3]={}",
            out2[3]
        );
    }

    /// Sliding window attention: query at global position 5 with window_size=2
    /// should only attend to keys at positions 4 and 5 (not 0..3).
    #[test]
    fn test_flash_attention_sliding_window_mask() {
        // 6 KV positions (0..5), query at global pos 5, window_size=2
        // Allowed: positions 4 and 5 (global_c + ws > global_q → c+2 > 5 → c >= 4)
        let head_dim = 4;
        let kv_len = 6;

        // K: identity-like so each position has a unique signature
        // K[i] = e_i (one-hot-ish): each row has a 1.0 at position i%4
        let mut k = vec![0.0f32; kv_len * head_dim];
        for i in 0..kv_len {
            k[i * head_dim + i % head_dim] = 1.0;
        }

        // V: V[i] = (i+1) * 10 in first dim, 0 elsewhere
        let mut v = vec![0.0f32; kv_len * head_dim];
        for i in 0..kv_len {
            v[i * head_dim] = (i + 1) as f32 * 10.0;
        }

        // Q: aligned with position 5's K pattern to get high attention weight there
        let q = vec![0.0f32, 1.0, 0.0, 0.0]; // matches K[5] (5%4=1)

        let mut out = vec![0.0f32; head_dim];
        flash_attention_head(
            &q,
            head_dim,
            &k,
            head_dim,
            &v,
            head_dim,
            &mut out,
            head_dim,
            1,      // q_len=1
            kv_len, // kv_len=6
            head_dim,
            5, // q_start_pos=5
            8,
            8,
            Some(2), // window_size=2: only positions 4,5 visible
        );

        // out[0] = weighted sum of V[i][0] for i in {4,5} = 50 and 60
        // Positions 0..3 are masked, V[0][0]=10..V[3][0]=40 must NOT contribute
        // Since only positions 4 and 5 are attended to, out[0] must be between 50 and 60
        assert!(
            out[0] >= 50.0 - 1e-4 && out[0] <= 60.0 + 1e-4,
            "out[0]={} should be in [50, 60] (only positions 4,5 visible)",
            out[0]
        );

        // With full causal (no window), out[0] would be dragged toward [10,60] average
        let mut out_full = vec![0.0f32; head_dim];
        flash_attention_head(
            &q,
            head_dim,
            &k,
            head_dim,
            &v,
            head_dim,
            &mut out_full,
            head_dim,
            1,
            kv_len,
            head_dim,
            5,
            8,
            8,
            None,
        );
        // With full causal, positions 0..5 all contribute → out[0] pulled lower
        assert!(
            out_full[0] < out[0],
            "full-causal out[0]={} should be < window-2 out[0]={}",
            out_full[0],
            out[0]
        );
    }

    /// Sliding window with window_size >= seq_len should behave same as full causal.
    #[test]
    fn test_sliding_window_large_window_equals_full() {
        let head_dim = 4;
        let kv_len = 4;
        let mut rng_state = 42u64;
        let mut pseudo_rand = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state & 0xFFFF) as f32) / 65536.0 - 0.5
        };

        let q: Vec<f32> = (0..head_dim).map(|_| pseudo_rand()).collect();
        let k: Vec<f32> = (0..kv_len * head_dim).map(|_| pseudo_rand()).collect();
        let v: Vec<f32> = (0..kv_len * head_dim).map(|_| pseudo_rand()).collect();

        let mut out_full = vec![0.0f32; head_dim];
        flash_attention_head(
            &q,
            head_dim,
            &k,
            head_dim,
            &v,
            head_dim,
            &mut out_full,
            head_dim,
            1,
            kv_len,
            head_dim,
            3,
            4,
            4,
            None,
        );

        let mut out_big_window = vec![0.0f32; head_dim];
        flash_attention_head(
            &q,
            head_dim,
            &k,
            head_dim,
            &v,
            head_dim,
            &mut out_big_window,
            head_dim,
            1,
            kv_len,
            head_dim,
            3,
            4,
            4,
            Some(1000),
        );

        for i in 0..head_dim {
            assert!(
                (out_full[i] - out_big_window[i]).abs() < 1e-5,
                "Mismatch at dim {}: full={} large_window={}",
                i,
                out_full[i],
                out_big_window[i]
            );
        }
    }
}
