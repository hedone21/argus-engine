use super::*;

std::thread_local! {
    static TLS_T_SYNC_DONE: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
}

impl TransformerLayer {
    /// Compute post-softmax attention scores for non-F32 KV cache (Q4_0, F16).
    /// This is a score-only pass — does NOT compute the attention output.
    /// Scores are written to `scores_out` in [n_heads_q, stride] layout.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop, dead_code)]
    pub(super) fn compute_attention_scores(
        q: &Tensor,
        k_cache: &Tensor,
        scores_out: &mut [f32],
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        is_head_major: bool,
        capacity: usize,
        backend: &Arc<dyn Backend>,
    ) -> Result<()> {
        let stride = scores_out.len() / n_heads_q;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_rep = n_heads_q / n_heads_kv;

        // Read Q to CPU (always F32)
        let q_data: Vec<f32> = if backend.is_gpu() {
            let mut buf = vec![0.0f32; q.size() / 4];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 4)
            };
            backend.read_buffer(q, bytes)?;
            buf
        } else {
            q.as_slice::<f32>().to_vec()
        };

        match k_cache.dtype() {
            DType::Q4_0 => {
                use crate::quant::{BlockQ4_0, QK4_0};
                let blocks_per_row = head_dim / QK4_0;

                // Read K cache to CPU
                let k_bytes = if backend.is_gpu() {
                    let mut buf = vec![0u8; k_cache.size()];
                    backend.read_buffer(k_cache, &mut buf)?;
                    buf
                } else {
                    let ptr = k_cache.as_ptr();
                    let len = k_cache.size();
                    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
                };
                let k_blocks = unsafe {
                    std::slice::from_raw_parts(
                        k_bytes.as_ptr() as *const BlockQ4_0,
                        k_bytes.len() / std::mem::size_of::<BlockQ4_0>(),
                    )
                };

                let score_chunks: Vec<&mut [f32]> =
                    scores_out.chunks_mut(stride).take(n_heads_q).collect();

                // Process each Q head
                for (h, scores_h) in score_chunks.into_iter().enumerate() {
                    let kv_h = h / n_rep;
                    let q_off = h * head_dim;
                    let q_vec = &q_data[q_off..q_off + head_dim];
                    let mut kv_f32 = vec![0.0f32; head_dim];

                    for t in 0..cache_seq_len {
                        let block_off = if is_head_major {
                            (kv_h * capacity + t) * blocks_per_row
                        } else {
                            (t * n_heads_kv + kv_h) * blocks_per_row
                        };
                        // Dequantize K row
                        for bi in 0..blocks_per_row {
                            let mut tmp = [0.0f32; QK4_0];
                            k_blocks[block_off + bi].dequantize(&mut tmp);
                            kv_f32[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                        }
                        let score: f32 = q_vec
                            .iter()
                            .zip(kv_f32[..head_dim].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        scores_h[t] = score * scale;
                    }

                    // Softmax
                    let active = &mut scores_h[..cache_seq_len];
                    let max_v = active.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum_e = 0.0f32;
                    for s in active.iter_mut() {
                        *s = (*s - max_v).exp();
                        sum_e += *s;
                    }
                    let inv = 1.0 / sum_e;
                    for s in active.iter_mut() {
                        *s *= inv;
                    }
                }
            }
            DType::F16 => {
                // Read K cache to CPU as raw bytes
                let k_bytes: Vec<u8> = if backend.is_gpu() {
                    let mut buf = vec![0u8; k_cache.size()];
                    backend.read_buffer(k_cache, &mut buf)?;
                    buf
                } else {
                    let ptr = k_cache.as_ptr();
                    let len = k_cache.size();
                    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
                };
                let k_raw = k_bytes.as_ptr() as *const u16;

                let score_chunks: Vec<&mut [f32]> =
                    scores_out.chunks_mut(stride).take(n_heads_q).collect();

                for (h, scores_h) in score_chunks.into_iter().enumerate() {
                    let kv_h = h / n_rep;
                    let q_off = h * head_dim;

                    for t in 0..cache_seq_len {
                        let off = if is_head_major {
                            (kv_h * capacity + t) * head_dim
                        } else {
                            (t * n_heads_kv + kv_h) * head_dim
                        };

                        #[cfg(target_arch = "aarch64")]
                        let score = unsafe {
                            crate::quant::flash_neon::vec_dot_f16_f32(
                                head_dim,
                                q_data.as_ptr().add(q_off),
                                k_raw.add(off),
                            )
                        };
                        #[cfg(not(target_arch = "aarch64"))]
                        let score = {
                            let k_f16 = unsafe {
                                std::slice::from_raw_parts(
                                    k_raw.add(off) as *const half::f16,
                                    head_dim,
                                )
                            };
                            let q_vec = &q_data[q_off..q_off + head_dim];
                            q_vec
                                .iter()
                                .zip(k_f16.iter())
                                .map(|(&a, &b)| a * b.to_f32())
                                .sum::<f32>()
                        };
                        scores_h[t] = score * scale;
                    }

                    let active = &mut scores_h[..cache_seq_len];
                    let max_v = active.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum_e = 0.0f32;
                    for s in active.iter_mut() {
                        *s = (*s - max_v).exp();
                        sum_e += *s;
                    }
                    let inv = 1.0 / sum_e;
                    for s in active.iter_mut() {
                        *s *= inv;
                    }
                }
            }
            _ => {
                // F32 should not reach here (handled by inline attention path)
            }
        }
        Ok(())
    }

    /// Q4_0 KV cache + GPU backend: CPU dequant + attention fallback.
    ///
    /// The OpenCL backend has no Q4_0 dequant-attention kernel, so this path
    /// reads Q4_0 raw bytes from GPU, dequantizes on CPU, computes full
    /// attention (scores + weighted V sum), and writes the result back to GPU.
    // `pub(crate)`: Phase α-K substep (3c) — `StandardFormat::attention_into` 의
    // Q4_0+GPU fallback 흡수가 본 fn 을 재사용한다(중복 0, DRY). forward_gen 의
    // 기존 `Self::attention_q4_gpu_fallback` 호출은 그대로(bit-identical) — 본 변경은
    // visibility 확장뿐이라 codegen 무변.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub(crate) fn attention_q4_gpu_fallback(
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out_attn: &mut Tensor,
        scores_buf: &mut [f32],
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        kv_start_pos: usize,
        layout: crate::kv_cache_ops::KVLayout,
        capacity: usize,
        need_scores: bool,
        // `&dyn Backend` (구 `&Arc<dyn Backend>`): substep (3c) 에서 `StandardFormat::attention_into`
        // (trait method 라 `&dyn Backend` 만 보유)가 본 fn 을 재사용하기 위한 일반화. body 는 backend
        // 의 `&self` 메서드(read_buffer/write_buffer)만 호출하므로 `&dyn` 으로 충분 — Arc 무관.
        backend: &dyn Backend,
    ) -> Result<()> {
        use crate::quant::{BlockQ4_0, QK4_0};

        if cache_seq_len == 0 {
            return Ok(());
        }

        let blocks_per_row = head_dim / QK4_0;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_rep = n_heads_q / n_heads_kv;
        let is_head_major = layout == crate::kv_cache_ops::KVLayout::HeadMajor;

        // 1. Read Q (F32) from GPU
        let mut q_data = vec![0.0f32; q.size() / 4];
        let q_bytes = unsafe {
            std::slice::from_raw_parts_mut(q_data.as_mut_ptr() as *mut u8, q_data.len() * 4)
        };
        backend.read_buffer(q, q_bytes)?;

        // 2. Read K/V (Q4_0 raw bytes) from GPU
        let mut k_raw_bytes = vec![0u8; k_cache.size()];
        let mut v_raw_bytes = vec![0u8; v_cache.size()];
        backend.read_buffer(k_cache, &mut k_raw_bytes)?;
        backend.read_buffer(v_cache, &mut v_raw_bytes)?;

        let k_blocks = unsafe {
            std::slice::from_raw_parts(
                k_raw_bytes.as_ptr() as *const BlockQ4_0,
                k_raw_bytes.len() / std::mem::size_of::<BlockQ4_0>(),
            )
        };
        let v_blocks = unsafe {
            std::slice::from_raw_parts(
                v_raw_bytes.as_ptr() as *const BlockQ4_0,
                v_raw_bytes.len() / std::mem::size_of::<BlockQ4_0>(),
            )
        };

        // 3. CPU dequant + attention (per Q-head)
        let mut out_f32 = vec![0.0f32; n_heads_q * head_dim];
        let stride = scores_buf.len() / n_heads_q;

        out_f32
            .par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(h, out_h)| {
                let kv_h = h / n_rep;
                let q_off = h * head_dim;
                let q_vec = &q_data[q_off..q_off + head_dim];
                let mut kv_f32 = vec![0.0f32; head_dim];
                let mut scores = vec![0.0f32; cache_seq_len];

                // Q * K^T with dequantize
                for t in 0..cache_seq_len {
                    let phys_t = kv_start_pos + t;
                    let block_off = if is_head_major {
                        (kv_h * capacity + phys_t) * blocks_per_row
                    } else {
                        (phys_t * n_heads_kv + kv_h) * blocks_per_row
                    };
                    for bi in 0..blocks_per_row {
                        let mut tmp = [0.0f32; QK4_0];
                        k_blocks[block_off + bi].dequantize(&mut tmp);
                        kv_f32[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                    }
                    let score: f32 = q_vec.iter().zip(kv_f32.iter()).map(|(a, b)| a * b).sum();
                    scores[t] = score * scale;
                }

                // NaN guard
                for s in scores.iter_mut() {
                    if s.is_nan() {
                        *s = f32::NEG_INFINITY;
                    }
                }

                // Softmax
                let max_v = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                if max_v == f32::NEG_INFINITY {
                    let u = 1.0 / cache_seq_len as f32;
                    for s in scores.iter_mut() {
                        *s = u;
                    }
                } else {
                    let mut sum_e = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max_v).exp();
                        sum_e += *s;
                    }
                    if sum_e.is_nan() || sum_e <= 0.0 || sum_e.is_infinite() {
                        let u = 1.0 / cache_seq_len as f32;
                        for s in scores.iter_mut() {
                            *s = u;
                        }
                    } else {
                        let inv = 1.0 / sum_e;
                        for s in scores.iter_mut() {
                            *s *= inv;
                        }
                    }
                }

                // Copy scores to output buffer if needed
                if need_scores {
                    // Safety: each head writes to non-overlapping region in scores_buf
                    unsafe {
                        let len = cache_seq_len.min(stride);
                        let dst = std::slice::from_raw_parts_mut(
                            (scores_buf.as_ptr() as *mut f32).add(h * stride),
                            len,
                        );
                        dst.copy_from_slice(&scores[..len]);
                    }
                }

                // Weighted V sum with dequantize
                for d in out_h.iter_mut() {
                    *d = 0.0;
                }
                for t in 0..cache_seq_len {
                    let phys_t = kv_start_pos + t;
                    let w = scores[t];
                    let block_off = if is_head_major {
                        (kv_h * capacity + phys_t) * blocks_per_row
                    } else {
                        (phys_t * n_heads_kv + kv_h) * blocks_per_row
                    };
                    for bi in 0..blocks_per_row {
                        let mut tmp = [0.0f32; QK4_0];
                        v_blocks[block_off + bi].dequantize(&mut tmp);
                        kv_f32[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                    }
                    for d in 0..head_dim {
                        out_h[d] += w * kv_f32[d];
                    }
                }
            });

        // 4. Write result back to GPU out_attn buffer (in-place, no realloc)
        let out_bytes =
            unsafe { std::slice::from_raw_parts(out_f32.as_ptr() as *const u8, out_f32.len() * 4) };
        backend.write_buffer(out_attn, out_bytes)?;

        Ok(())
    }
}
