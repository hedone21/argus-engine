use crate::backend::Backend;
use crate::buffer::Buffer;
use crate::buffer::DType;
use crate::quant::{BlockQ4_0, BlockQ4_1, QK4_0, QK4_1, QK8_0};
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use rayon::prelude::*;

/// Minimum number of elements before using Rayon parallelism.
/// Below this, serial execution is faster due to thread dispatch overhead.
/// 16384 floats = 64KB ≈ L1 cache size.
const PARALLEL_THRESHOLD: usize = 16384;

pub struct CpuBackendCommon;

impl Default for CpuBackendCommon {
    fn default() -> Self {
        Self
    }
}

impl CpuBackendCommon {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackendCommon {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn name(&self) -> &str {
        "CPU (Scalar)"
    }

    fn device(&self) -> &str {
        "CPU"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let _out_shape = out.shape().dims();

        if a_shape[1] != b_shape[0] {
            return Err(anyhow!(
                "Shape mismatch for matmul: {:?} x {:?}",
                a_shape,
                b_shape
            ));
        }

        let _m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();

        // Naive parallel implementation
        out_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                row[j] = sum;
            }
        });

        Ok(())
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        match b.dtype() {
            DType::F32 => self.matmul_transposed_f32(a, b, out),
            DType::F16 => self.matmul_transposed_f16(a, b, out),
            DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
            DType::Q4_1 => self.matmul_transposed_q4_1(a, b, out),
            // generic floor: dequant via descriptor → f32 matmul. exact, 느릴 뿐.
            _ => {
                let f32_b = crate::format::dequant_to_f32_tensor(b)?;
                self.matmul_transposed_f32(a, &f32_b, out)
            }
        }
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        _rows: usize,
        _cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        // Reuse matmul_transposed logic as implemented in CPU backend
        match b.dtype() {
            DType::F32 => self.matmul_transposed_f32(a, b, out),
            DType::F16 => self.matmul_transposed_f16(a, b, out),
            DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
            DType::Q4_1 => self.matmul_transposed_q4_1(a, b, out),
            // generic floor: dequant via descriptor → f32 matmul. exact, 느릴 뿐.
            _ => {
                let f32_b = crate::format::dequant_to_f32_tensor(b)?;
                self.matmul_transposed_f32(a, &f32_b, out)
            }
        }
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        let a_data = a.as_mut_slice::<f32>();
        let b_data = b.as_slice::<f32>();

        if a_data.len() != b_data.len() {
            return Err(anyhow!("Size mismatch for add_assign"));
        }

        if a_data.len() < PARALLEL_THRESHOLD {
            for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                *x += y;
            }
        } else {
            a_data
                .par_iter_mut()
                .zip(b_data.par_iter())
                .for_each(|(x, y)| {
                    *x += y;
                });
        }
        Ok(())
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        let x_data = x.as_mut_slice::<f32>();
        if x_data.len() < PARALLEL_THRESHOLD {
            for val in x_data.iter_mut() {
                *val *= v;
            }
        } else {
            x_data.par_iter_mut().for_each(|val| *val *= v);
        }
        Ok(())
    }

    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        let a_data = a.as_mut_slice::<f32>();
        let b_data = b.as_slice::<f32>();

        if a_data.len() < PARALLEL_THRESHOLD {
            for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                let silu_x = *x / (1.0 + (-*x).exp());
                *x = silu_x * y;
            }
        } else {
            a_data
                .par_iter_mut()
                .zip(b_data.par_iter())
                .for_each(|(x, y)| {
                    let silu_x = *x / (1.0 + (-*x).exp());
                    *x = silu_x * y;
                });
        }
        Ok(())
    }

    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1]; // Last dim

        let x_data = x.as_mut_slice::<f32>();
        let w_data = w.as_slice::<f32>();

        x_data.par_chunks_mut(dim).for_each(|row| {
            let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
            let rms = (sum_sq / dim as f32 + eps).sqrt();
            let scale = 1.0 / rms;

            for (val, weight) in row.iter_mut().zip(w_data.iter()) {
                let w_eff = if add_unit { 1.0 + weight } else { *weight };
                *val = (*val * scale) * w_eff;
            }
        });

        Ok(())
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];

        let x_data = x.as_mut_slice::<f32>();

        x_data.par_chunks_mut(dim).for_each(|row| {
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
                sum_exp += *val;
            }
            for val in row.iter_mut() {
                *val /= sum_exp;
            }
        });
        Ok(())
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        let dims = x.shape().dims();
        if dims.len() < 3 {
            return Err(anyhow!(
                "RoPE expects at least 3 dims (Seq, Head, Dim) or (Batch, Seq, Head, Dim)"
            ));
        }

        let head_dim = dims[dims.len() - 1];
        let num_heads = dims[dims.len() - 2];
        let seq_len = dims[dims.len() - 3];
        // Flatten batch if exists
        let _batch = if dims.len() == 4 { dims[0] } else { 1 };

        if !head_dim.is_multiple_of(2) {
            return Err(anyhow!("Head dim must be even for RoPE"));
        }

        // Precompute base frequencies: freq[i] = theta^(-2i/head_dim)
        // This eliminates repeated powf() calls in the inner loop.
        let half_dim = head_dim / 2;
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
            .collect();

        let x_data = x.as_mut_slice::<f32>();

        x_data
            .par_chunks_mut(seq_len * num_heads * head_dim)
            .enumerate()
            .for_each(|(_b, batch_chunk)| {
                for t in 0..seq_len {
                    let pos = start_pos + t;
                    for h in 0..num_heads {
                        let offset = (t * num_heads + h) * head_dim;
                        let head_slice = &mut batch_chunk[offset..offset + head_dim];

                        for i in 0..half_dim {
                            let val = pos as f32 * freqs[i];
                            let (sin, cos) = val.sin_cos();

                            let v0 = head_slice[i];
                            let v1 = head_slice[i + half_dim];

                            head_slice[i] = v0 * cos - v1 * sin;
                            head_slice[i + half_dim] = v0 * sin + v1 * cos;
                        }
                    }
                }
            });

        Ok(())
    }

    fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
        use crate::memory::host::shared::SharedBuffer;
        use std::sync::Arc;

        let new_buf = SharedBuffer::new(t.size(), t.dtype());
        // Memcpy
        unsafe {
            std::ptr::copy_nonoverlapping(t.as_ptr(), new_buf.as_mut_ptr(), t.size());
        }

        // Return a Tensor attached to this common backend.
        // User might mix implementations, but tensor belongs to a backend instance.
        Ok(Tensor::new(
            t.shape().clone(),
            Arc::new(new_buf),
            Arc::new(CpuBackendCommon),
        ))
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        match (src.dtype(), dst.dtype()) {
            (DType::F32, DType::F16) => {
                let s = src.as_slice::<f32>();
                let d = dst.as_mut_slice::<half::f16>();
                // Serial for small arrays (avoids Rayon dispatch overhead ~100µs)
                if s.len() < 4096 {
                    for (d_val, s_val) in d.iter_mut().zip(s.iter()) {
                        *d_val = half::f16::from_f32(*s_val);
                    }
                } else {
                    d.par_iter_mut()
                        .zip(s.par_iter())
                        .for_each(|(d_val, s_val)| {
                            *d_val = half::f16::from_f32(*s_val);
                        });
                }
                Ok(())
            }
            (DType::F16, DType::F32) => {
                let s = src.as_slice::<half::f16>();
                let d = dst.as_mut_slice::<f32>();
                if s.len() < 4096 {
                    unsafe {
                        crate::quant::f16_bulk::bulk_f16_to_f32(
                            s.as_ptr() as *const u16,
                            d.as_mut_ptr(),
                            s.len(),
                        );
                    }
                } else {
                    d.par_iter_mut()
                        .zip(s.par_iter())
                        .for_each(|(d_val, s_val)| {
                            *d_val = s_val.to_f32();
                        });
                }
                Ok(())
            }
            (DType::F32, DType::Q4_0) => {
                use crate::quant::{BlockQ4_0, QK4_0};
                let s = src.as_slice::<f32>();
                let n_elements = s.len();
                assert!(
                    n_elements.is_multiple_of(QK4_0),
                    "F32->Q4_0 cast: element count must be multiple of {}",
                    QK4_0
                );
                let n_blocks = n_elements / QK4_0;
                let d = unsafe {
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut BlockQ4_0, n_blocks)
                };
                for bi in 0..n_blocks {
                    let src_block = &s[bi * QK4_0..(bi + 1) * QK4_0];
                    let max_val = src_block
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0f32, |x, y| x.max(y));
                    let scale = max_val / 7.0;
                    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
                    d[bi].d = half::f16::from_f32(scale);
                    for z in 0..16 {
                        let v0 = (src_block[z] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                        let v1 = (src_block[z + 16] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                        d[bi].qs[z] = (v0 + 8) as u8 | (((v1 + 8) as u8) << 4);
                    }
                }
                Ok(())
            }
            _ => Err(anyhow!(
                "Unsupported cast: {:?} -> {:?}",
                src.dtype(),
                dst.dtype()
            )),
        }
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn attention_gen(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        scores_out: Option<&mut [f32]>,
    ) -> Result<()> {
        let q_data = q.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = num_heads_q / num_heads_kv;

        // Wrap raw pointer for Send+Sync so it can be captured by Rayon par_chunks_mut.
        // Safety: each head h writes to non-overlapping region [h*stride .. h*stride+cache_seq_len].
        #[derive(Clone, Copy)]
        struct SendPtr(*mut f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        let scores_ptr = scores_out.as_ref().map(|s| SendPtr(s.as_ptr() as *mut f32));
        let scores_stride = scores_out
            .as_ref()
            .map(|s| s.len() / num_heads_q)
            .unwrap_or(0);

        // Detect layout: HeadMajor [batch, kv_heads, capacity, head_dim]
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };

        match k_cache.dtype() {
            DType::F32 => {
                let k_data = k_cache.as_slice::<f32>();
                let v_data = v_cache.as_slice::<f32>();
                out_data
                    .par_chunks_mut(head_dim)
                    .enumerate()
                    .for_each(|(h, out_h)| {
                        let kv_h = h / gqa_ratio;
                        let q_off = h * head_dim;
                        let q_vec = &q_data[q_off..q_off + head_dim];
                        let mut scores = vec![0.0f32; cache_seq_len];
                        for t in 0..cache_seq_len {
                            let off = if is_head_major {
                                (kv_h * capacity + t) * head_dim
                            } else {
                                (t * num_heads_kv + kv_h) * head_dim
                            };
                            let k_vec = &k_data[off..off + head_dim];
                            let s: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                            scores[t] = s * scale;
                        }
                        // NaN guard: replace NaN logits with -inf before softmax
                        for s in scores.iter_mut() {
                            if s.is_nan() {
                                *s = f32::NEG_INFINITY;
                            }
                        }
                        // softmax
                        let max_v = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        if max_v == f32::NEG_INFINITY {
                            // All logits are -inf: uniform distribution
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
                                for s in scores.iter_mut() {
                                    *s /= sum_e;
                                }
                            }
                        }
                        if let Some(SendPtr(ptr)) = scores_ptr {
                            unsafe {
                                let dst = std::slice::from_raw_parts_mut(
                                    ptr.add(h * scores_stride),
                                    cache_seq_len,
                                );
                                dst.copy_from_slice(&scores[..cache_seq_len]);
                            }
                        }
                        // weighted sum
                        for d in 0..head_dim {
                            out_h[d] = 0.0;
                        }
                        for t in 0..cache_seq_len {
                            let w = scores[t];
                            let off = if is_head_major {
                                (kv_h * capacity + t) * head_dim
                            } else {
                                (t * num_heads_kv + kv_h) * head_dim
                            };
                            let v_vec = &v_data[off..off + head_dim];
                            for d in 0..head_dim {
                                out_h[d] += w * v_vec[d];
                            }
                        }
                    });
            }
            DType::F16 => {
                // With +fp16 target feature, half::f16::to_f32() inlines to scalar fcvt
                // and the compiler auto-vectorizes the conversion loop to fcvtl.
                // This is faster than manual inline asm due to better register allocation.
                let k_data = k_cache.as_slice::<half::f16>();
                let v_data = v_cache.as_slice::<half::f16>();
                out_data
                    .par_chunks_mut(head_dim)
                    .enumerate()
                    .for_each(|(h, out_h)| {
                        let kv_h = h / gqa_ratio;
                        let q_off = h * head_dim;
                        let q_vec = &q_data[q_off..q_off + head_dim];
                        let mut scores = vec![0.0f32; cache_seq_len];
                        let mut kv_f32 = vec![0.0f32; head_dim];

                        // Q * K^T: convert K to F32, then NEON dot
                        for t in 0..cache_seq_len {
                            let off = if is_head_major {
                                (kv_h * capacity + t) * head_dim
                            } else {
                                (t * num_heads_kv + kv_h) * head_dim
                            };
                            let k_row = &k_data[off..off + head_dim];
                            for d in 0..head_dim {
                                kv_f32[d] = k_row[d].to_f32();
                            }

                            #[cfg(target_arch = "aarch64")]
                            let score = unsafe {
                                use std::arch::aarch64::*;
                                let q_ptr = q_vec.as_ptr();
                                let k_ptr = kv_f32.as_ptr();
                                let mut sum_v = vdupq_n_f32(0.0);
                                let mut i = 0;
                                while i + 16 <= head_dim {
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i)),
                                        vld1q_f32(k_ptr.add(i)),
                                    );
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i + 4)),
                                        vld1q_f32(k_ptr.add(i + 4)),
                                    );
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i + 8)),
                                        vld1q_f32(k_ptr.add(i + 8)),
                                    );
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i + 12)),
                                        vld1q_f32(k_ptr.add(i + 12)),
                                    );
                                    i += 16;
                                }
                                while i + 4 <= head_dim {
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i)),
                                        vld1q_f32(k_ptr.add(i)),
                                    );
                                    i += 4;
                                }
                                let mut s = vaddvq_f32(sum_v);
                                while i < head_dim {
                                    s += *q_ptr.add(i) * *k_ptr.add(i);
                                    i += 1;
                                }
                                s
                            };
                            #[cfg(not(target_arch = "aarch64"))]
                            let score: f32 =
                                q_vec.iter().zip(kv_f32.iter()).map(|(a, b)| a * b).sum();

                            scores[t] = score * scale;
                        }

                        // NaN guard: replace NaN logits with -inf before softmax
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
                                for s in scores.iter_mut() {
                                    *s /= sum_e;
                                }
                            }
                        }
                        if let Some(SendPtr(ptr)) = scores_ptr {
                            unsafe {
                                let dst = std::slice::from_raw_parts_mut(
                                    ptr.add(h * scores_stride),
                                    cache_seq_len,
                                );
                                dst.copy_from_slice(&scores[..cache_seq_len]);
                            }
                        }

                        // Weighted V sum
                        for d in 0..head_dim {
                            out_h[d] = 0.0;
                        }
                        for t in 0..cache_seq_len {
                            let w = scores[t];
                            let off = if is_head_major {
                                (kv_h * capacity + t) * head_dim
                            } else {
                                (t * num_heads_kv + kv_h) * head_dim
                            };
                            let v_row = &v_data[off..off + head_dim];
                            for d in 0..head_dim {
                                kv_f32[d] = v_row[d].to_f32();
                            }

                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                use std::arch::aarch64::*;
                                let v_ptr = kv_f32.as_ptr();
                                let o_ptr = out_h.as_mut_ptr();
                                let w_v = vdupq_n_f32(w);
                                let mut i = 0;
                                while i + 16 <= head_dim {
                                    vst1q_f32(
                                        o_ptr.add(i),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i)),
                                        ),
                                    );
                                    vst1q_f32(
                                        o_ptr.add(i + 4),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i + 4)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i + 4)),
                                        ),
                                    );
                                    vst1q_f32(
                                        o_ptr.add(i + 8),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i + 8)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i + 8)),
                                        ),
                                    );
                                    vst1q_f32(
                                        o_ptr.add(i + 12),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i + 12)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i + 12)),
                                        ),
                                    );
                                    i += 16;
                                }
                                while i + 4 <= head_dim {
                                    vst1q_f32(
                                        o_ptr.add(i),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i)),
                                        ),
                                    );
                                    i += 4;
                                }
                                while i < head_dim {
                                    *o_ptr.add(i) += w * *v_ptr.add(i);
                                    i += 1;
                                }
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            for d in 0..head_dim {
                                out_h[d] += w * kv_f32[d];
                            }
                        }
                    });
            }
            DType::Q4_0 => {
                use crate::quant::{BlockQ4_0, QK4_0};
                let k_raw = unsafe {
                    std::slice::from_raw_parts(
                        k_cache.as_ptr() as *const BlockQ4_0,
                        k_cache.size() / std::mem::size_of::<BlockQ4_0>(),
                    )
                };
                let v_raw = unsafe {
                    std::slice::from_raw_parts(
                        v_cache.as_ptr() as *const BlockQ4_0,
                        v_cache.size() / std::mem::size_of::<BlockQ4_0>(),
                    )
                };
                let blocks_per_row = head_dim / QK4_0; // e.g. 64/32 = 2 blocks
                out_data
                    .par_chunks_mut(head_dim)
                    .enumerate()
                    .for_each(|(h, out_h)| {
                        let kv_h = h / gqa_ratio;
                        let q_off = h * head_dim;
                        let q_vec = &q_data[q_off..q_off + head_dim];
                        let mut scores = vec![0.0f32; cache_seq_len];
                        let mut kv_f32 = vec![0.0f32; head_dim];

                        // Q * K^T: dequantize K row, then dot
                        for t in 0..cache_seq_len {
                            let block_off = if is_head_major {
                                (kv_h * capacity + t) * blocks_per_row
                            } else {
                                (t * num_heads_kv + kv_h) * blocks_per_row
                            };
                            // Dequantize K row into kv_f32
                            for bi in 0..blocks_per_row {
                                let mut tmp = [0.0f32; QK4_0];
                                k_raw[block_off + bi].dequantize(&mut tmp);
                                kv_f32[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                            }
                            #[cfg(target_arch = "aarch64")]
                            let score = unsafe {
                                use std::arch::aarch64::*;
                                let q_ptr = q_vec.as_ptr();
                                let k_ptr = kv_f32.as_ptr();
                                let mut sum_v = vdupq_n_f32(0.0);
                                let mut i = 0;
                                while i + 16 <= head_dim {
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i)),
                                        vld1q_f32(k_ptr.add(i)),
                                    );
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i + 4)),
                                        vld1q_f32(k_ptr.add(i + 4)),
                                    );
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i + 8)),
                                        vld1q_f32(k_ptr.add(i + 8)),
                                    );
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i + 12)),
                                        vld1q_f32(k_ptr.add(i + 12)),
                                    );
                                    i += 16;
                                }
                                while i + 4 <= head_dim {
                                    sum_v = vfmaq_f32(
                                        sum_v,
                                        vld1q_f32(q_ptr.add(i)),
                                        vld1q_f32(k_ptr.add(i)),
                                    );
                                    i += 4;
                                }
                                let mut s = vaddvq_f32(sum_v);
                                while i < head_dim {
                                    s += *q_ptr.add(i) * *k_ptr.add(i);
                                    i += 1;
                                }
                                s
                            };
                            #[cfg(not(target_arch = "aarch64"))]
                            let score: f32 =
                                q_vec.iter().zip(kv_f32.iter()).map(|(a, b)| a * b).sum();
                            scores[t] = score * scale;
                        }

                        // NaN guard: replace NaN logits with -inf before softmax
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
                                for s in scores.iter_mut() {
                                    *s /= sum_e;
                                }
                            }
                        }
                        if let Some(SendPtr(ptr)) = scores_ptr {
                            unsafe {
                                let dst = std::slice::from_raw_parts_mut(
                                    ptr.add(h * scores_stride),
                                    cache_seq_len,
                                );
                                dst.copy_from_slice(&scores[..cache_seq_len]);
                            }
                        }

                        // Weighted V sum: dequantize V row, then NEON FMA
                        for d in 0..head_dim {
                            out_h[d] = 0.0;
                        }
                        for t in 0..cache_seq_len {
                            let w = scores[t];
                            let block_off = if is_head_major {
                                (kv_h * capacity + t) * blocks_per_row
                            } else {
                                (t * num_heads_kv + kv_h) * blocks_per_row
                            };
                            for bi in 0..blocks_per_row {
                                let mut tmp = [0.0f32; QK4_0];
                                v_raw[block_off + bi].dequantize(&mut tmp);
                                kv_f32[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                            }
                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                use std::arch::aarch64::*;
                                let v_ptr = kv_f32.as_ptr();
                                let o_ptr = out_h.as_mut_ptr();
                                let w_v = vdupq_n_f32(w);
                                let mut i = 0;
                                while i + 16 <= head_dim {
                                    vst1q_f32(
                                        o_ptr.add(i),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i)),
                                        ),
                                    );
                                    vst1q_f32(
                                        o_ptr.add(i + 4),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i + 4)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i + 4)),
                                        ),
                                    );
                                    vst1q_f32(
                                        o_ptr.add(i + 8),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i + 8)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i + 8)),
                                        ),
                                    );
                                    vst1q_f32(
                                        o_ptr.add(i + 12),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i + 12)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i + 12)),
                                        ),
                                    );
                                    i += 16;
                                }
                                while i + 4 <= head_dim {
                                    vst1q_f32(
                                        o_ptr.add(i),
                                        vfmaq_f32(
                                            vld1q_f32(o_ptr.add(i)),
                                            w_v,
                                            vld1q_f32(v_ptr.add(i)),
                                        ),
                                    );
                                    i += 4;
                                }
                                while i < head_dim {
                                    *o_ptr.add(i) += w * *v_ptr.add(i);
                                    i += 1;
                                }
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            for d in 0..head_dim {
                                out_h[d] += w * kv_f32[d];
                            }
                        }
                    });
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported KV cache dtype for attention: {:?}",
                    k_cache.dtype()
                ));
            }
        }
        Ok(())
    }

    // B-5b Phase 2 Stage 1: CPU companion = self.
    fn cpu_companion(&self) -> &dyn Backend {
        self
    }
}

impl CpuBackendCommon {
    pub fn matmul_transposed_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        if a_rank < 2 || b_rank < 2 {
            return Err(anyhow!("Tensors must be at least 2D for matmul"));
        }

        let k = a_shape[a_rank - 1];
        let k_b = b_shape[b_rank - 1];

        if k != k_b {
            return Err(anyhow!(
                "Shape mismatch for matmul_transposed: K dims {} != {}",
                k,
                k_b
            ));
        }

        let m: usize = a_shape[..a_rank - 1].iter().product();
        let n = b_shape[b_rank - 2];

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();

        if out_data.len() != m * n {
            return Err(anyhow!(
                "Output buffer size mismatch. Expected {}, got {}",
                m * n,
                out_data.len()
            ));
        }

        out_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[j * k + l];
                }
                row[j] = sum;
            }
        });

        Ok(())
    }

    /// F16 weight matmul: A(F32) x B^T(F16) -> Out(F32)
    /// Dequantizes F16 weights to F32 in blocks during dot product.
    pub fn matmul_transposed_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let k_b = b_shape[b_rank - 1];

        if k != k_b {
            return Err(anyhow!(
                "Shape mismatch for matmul_transposed_f16: K dims {} != {}",
                k,
                k_b
            ));
        }

        let n = b_shape[b_rank - 2];

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<half::f16>();
        let out_data = out.as_mut_slice::<f32>();

        out_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let a_offset = i * k;
            let a_row = &a_data[a_offset..a_offset + k];

            for (j, out_val) in row.iter_mut().enumerate().take(n) {
                let b_offset = j * k;
                let b_row = &b_data[b_offset..b_offset + k];

                let mut sum = 0.0f32;
                for z in 0..k {
                    sum += a_row[z] * b_row[z].to_f32();
                }
                *out_val = sum;
            }
        });

        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    pub fn matmul_transposed_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let k_b = b_shape[b_rank - 1];

        if k != k_b {
            return Err(anyhow!("Shape mismatch"));
        }
        if !k.is_multiple_of(QK4_0) {
            return Err(anyhow!("K divisible by black size"));
        }

        let m: usize = a_shape[..a_rank - 1].iter().product();
        let n = b_shape[b_rank - 2];

        let a_data = a.as_slice::<f32>();
        let nb_k = k / QK4_0;
        let total_blocks = n * nb_k;

        let b_blocks =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const BlockQ4_0, total_blocks) };

        let out_data = out.as_mut_slice::<f32>();

        // For small M, use the Q8_0 quantization path (Verification/Optimization)
        // This mirrors the NEON path but scalar.
        if m < 4 {
            let nb_k_q8 = k / QK8_0;
            if k.is_multiple_of(QK8_0) {
                // Ensure divisibility
                let total_q8_blocks = m * nb_k_q8;
                let mut a_q8 = vec![
                    crate::quant::BlockQ8_0 {
                        d: half::f16::from_f32(0.0),
                        qs: [0; crate::quant::QK8_0]
                    };
                    total_q8_blocks
                ];

                for i in 0..m {
                    let a_offset = i * k;
                    let a_row = &a_data[a_offset..a_offset + k];
                    let q8_row = &mut a_q8[i * nb_k_q8..(i + 1) * nb_k_q8];
                    self.quantize_row_q8_0(a_row, q8_row, k);
                }

                out_data.par_iter_mut().enumerate().for_each(|(idx, val)| {
                    let i = idx / n;
                    let j = idx % n;

                    let b_offset = j * nb_k;
                    let b_row_node = b.as_ptr() as *const BlockQ4_0;
                    let b_row_ptr = unsafe { b_row_node.add(b_offset) };

                    let a_row_ptr = unsafe { a_q8.as_ptr().add(i * nb_k_q8) };

                    let mut sum = 0.0;
                    // Since self is &CpuBackendCommon, it is Copy? No. Reference is Copy.
                    // But we can't call methods easily if borrow checker complains.
                    // Let's rely on the method being available.
                    // "self.vec_dot_q4_0_q8_0"

                    unsafe {
                        self.vec_dot_q4_0_q8_0(k, &mut sum, b_row_ptr, a_row_ptr);
                    }
                    *val = sum;
                });
                return Ok(());
            }
        }

        out_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let a_offset = i * k;
            let a_row = &a_data[a_offset..a_offset + k];

            for j in 0..n {
                let b_offset = j * nb_k;
                let b_row_blocks = &b_blocks[b_offset..b_offset + nb_k];

                let mut sum = 0.0;
                for (bi, block) in b_row_blocks.iter().enumerate() {
                    let d = block.d.to_f32();
                    let a_slice = &a_row[bi * QK4_0..(bi + 1) * QK4_0];

                    let mut isum: f32 = 0.0;
                    for z in 0..(QK4_0 / 2) {
                        let b = block.qs[z];
                        let v0 = (b & 0x0F) as i8 - 8;
                        let v1 = (b >> 4) as i8 - 8;

                        isum += v0 as f32 * a_slice[z];
                        isum += v1 as f32 * a_slice[z + QK4_0 / 2];
                    }
                    sum += d * isum;
                }
                row[j] = sum;
            }
        });

        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    pub fn quantize_row_q8_0(&self, x: &[f32], y: &mut [crate::quant::BlockQ8_0], k: usize) {
        use crate::quant::QK8_0;
        assert!(k.is_multiple_of(QK8_0));
        let nb = k / QK8_0;

        for i in 0..nb {
            let src = &x[i * QK8_0..(i + 1) * QK8_0];
            let mut amax = 0.0f32;
            for &v in src {
                amax = amax.max(v.abs());
            }

            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };

            y[i].d = half::f16::from_f32(d);

            for j in 0..QK8_0 {
                let v = src[j] * id;
                y[i].qs[j] = v.round() as i8;
            }
        }
    }

    /// # Safety
    /// `vx` and `vy` must point to valid `BlockQ4_0` / `BlockQ8_0` arrays of length `n / QK8_0`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn vec_dot_q4_0_q8_0(
        &self,
        n: usize,
        s: &mut f32,
        vx: *const BlockQ4_0,
        vy: *const crate::quant::BlockQ8_0,
    ) {
        use crate::quant::QK8_0;
        let nb = n / QK8_0;
        let mut sumf = 0.0;

        for i in 0..nb {
            let x = unsafe { &*vx.add(i) }; // Q4_0 block: 32 values
            let y = unsafe { &*vy.add(i) }; // Q8_0 block: 32 values

            let d = x.d.to_f32() * y.d.to_f32();
            let mut isum = 0;

            for j in 0..(QK4_0 / 2) {
                let v0 = (x.qs[j] & 0x0F) as i8 - 8;
                let v1 = (x.qs[j] >> 4) as i8 - 8;

                isum += (v0 as i32) * (y.qs[j] as i32);
                isum += (v1 as i32) * (y.qs[j + QK4_0 / 2] as i32);
            }
            sumf += d * isum as f32;
        }
        *s = sumf;
    }

    #[allow(clippy::needless_range_loop)]
    pub fn matmul_transposed_q4_1(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let k_b = b_shape[b_rank - 1];

        if k != k_b {
            return Err(anyhow!("Shape mismatch"));
        }

        let _m: usize = a_shape[..a_rank - 1].iter().product();
        let n = b_shape[b_rank - 2];
        let nb_k = k / QK4_1;

        let a_data = a.as_slice::<f32>();

        let b_blocks =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const BlockQ4_1, n * nb_k) };

        let out_data = out.as_mut_slice::<f32>();

        out_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let a_offset = i * k;
            let a_row = &a_data[a_offset..a_offset + k];

            for j in 0..n {
                let b_offset = j * nb_k;
                let b_row_blocks = &b_blocks[b_offset..b_offset + nb_k];

                let mut sum = 0.0;
                for (bi, block) in b_row_blocks.iter().enumerate() {
                    let d = block.d.to_f32();
                    let m = block.m.to_f32();
                    let a_slice = &a_row[bi * QK4_1..(bi + 1) * QK4_1];

                    let mut s0 = 0.0;
                    let mut s1 = 0.0;

                    for z in 0..(QK4_1 / 2) {
                        let b = block.qs[z];
                        let v0 = (b & 0x0F) as f32;
                        let v1 = (b >> 4) as f32;

                        s0 += v0 * a_slice[z] + v1 * a_slice[z + QK4_1 / 2];
                        s1 += a_slice[z] + a_slice[z + QK4_1 / 2];
                    }
                    sum += d * s0 + m * s1;
                }
                row[j] = sum;
            }
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;
    use crate::quant::{BlockQ4_0, BlockQ4_1, QK4_0, QK4_1};
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    // ---- Helpers ----

    fn make_f32_tensor(backend: &Arc<dyn Backend>, shape: Vec<usize>, data: &[f32]) -> Tensor {
        let memory = Galloc::new();
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length must match shape");
        let buf = memory.alloc(numel * 4, DType::F32).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.as_mut_ptr() as *mut f32, numel);
        }
        Tensor::new(Shape::new(shape), buf, backend.clone())
    }

    fn make_f32_tensor_zeros(backend: &Arc<dyn Backend>, shape: Vec<usize>) -> Tensor {
        let memory = Galloc::new();
        let numel: usize = shape.iter().product();
        let buf = memory.alloc(numel * 4, DType::F32).unwrap();
        Tensor::new(Shape::new(shape), buf, backend.clone())
    }

    fn make_f16_tensor_zeros(backend: &Arc<dyn Backend>, shape: Vec<usize>) -> Tensor {
        let memory = Galloc::new();
        let numel: usize = shape.iter().product();
        let buf = memory.alloc(numel * 2, DType::F16).unwrap();
        Tensor::new(Shape::new(shape), buf, backend.clone())
    }

    /// Generate deterministic pseudo-random data
    fn gen_data(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(seed).wrapping_add(17)) % 1000;
                (v as f32 * 0.001) - 0.5
            })
            .collect()
    }

    /// Quantize f32 data to Q4_0 blocks (row-major, row_len must be multiple of 32)
    fn quantize_q4_0(data: &[f32], n_rows: usize, row_len: usize) -> Vec<u8> {
        let nb = row_len / QK4_0;
        let mut blocks = Vec::with_capacity(n_rows * nb);
        for row in 0..n_rows {
            for bi in 0..nb {
                let offset = row * row_len + bi * QK4_0;
                let src = &data[offset..offset + QK4_0];
                let max_val = src.iter().map(|v| v.abs()).fold(0.0f32, |x, y| x.max(y));
                let d = max_val / 7.0;
                let id = if d == 0.0 { 0.0 } else { 1.0 / d };
                let mut block = BlockQ4_0 {
                    d: half::f16::from_f32(d),
                    qs: [0; 16],
                };
                for z in 0..16 {
                    let v0 = (src[z] * id).round().clamp(-8.0, 7.0) as i8;
                    let v1 = (src[z + 16] * id).round().clamp(-8.0, 7.0) as i8;
                    block.qs[z] = (v0 + 8) as u8 | (((v1 + 8) as u8) << 4);
                }
                blocks.push(block);
            }
        }
        // Convert to raw bytes
        let byte_len = blocks.len() * std::mem::size_of::<BlockQ4_0>();
        let mut bytes = vec![0u8; byte_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                blocks.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                byte_len,
            );
        }
        bytes
    }

    /// Quantize f32 data to Q4_1 blocks
    fn quantize_q4_1(data: &[f32], n_rows: usize, row_len: usize) -> Vec<u8> {
        let nb = row_len / QK4_1;
        let mut blocks = Vec::with_capacity(n_rows * nb);
        for row in 0..n_rows {
            for bi in 0..nb {
                let offset = row * row_len + bi * QK4_1;
                let src = &data[offset..offset + QK4_1];
                let min_val = src.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = src.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let d = (max_val - min_val) / 15.0;
                let m_val = min_val;
                let id = if d == 0.0 { 0.0 } else { 1.0 / d };
                let mut block = BlockQ4_1 {
                    d: half::f16::from_f32(d),
                    m: half::f16::from_f32(m_val),
                    qs: [0; 16],
                };
                for z in 0..16 {
                    let v0 = ((src[z] - m_val) * id).round().clamp(0.0, 15.0) as u8;
                    let v1 = ((src[z + 16] - m_val) * id).round().clamp(0.0, 15.0) as u8;
                    block.qs[z] = v0 | (v1 << 4);
                }
                blocks.push(block);
            }
        }
        let byte_len = blocks.len() * std::mem::size_of::<BlockQ4_1>();
        let mut bytes = vec![0u8; byte_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                blocks.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                byte_len,
            );
        }
        bytes
    }

    fn make_q4_0_tensor(backend: &Arc<dyn Backend>, shape: Vec<usize>, f32_data: &[f32]) -> Tensor {
        let memory = Galloc::new();
        let n_rows = shape[0];
        let row_len = shape[1];
        let bytes = quantize_q4_0(f32_data, n_rows, row_len);
        let buf = memory.alloc(bytes.len(), DType::Q4_0).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr(), bytes.len());
        }
        Tensor::new(Shape::new(shape), buf, backend.clone())
    }

    fn make_q4_1_tensor(backend: &Arc<dyn Backend>, shape: Vec<usize>, f32_data: &[f32]) -> Tensor {
        let memory = Galloc::new();
        let n_rows = shape[0];
        let row_len = shape[1];
        let bytes = quantize_q4_1(f32_data, n_rows, row_len);
        let buf = memory.alloc(bytes.len(), DType::Q4_1).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr(), bytes.len());
        }
        Tensor::new(Shape::new(shape), buf, backend.clone())
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{}: length mismatch {} vs {}",
            msg,
            a.len(),
            b.len()
        );
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            assert!(
                diff <= tol,
                "{}: element [{}] differs: {} vs {}, diff={}, tol={}",
                msg,
                i,
                va,
                vb,
                diff,
                tol
            );
        }
    }

    fn scalar_backend() -> Arc<dyn Backend> {
        Arc::new(CpuBackendCommon::new())
    }

    fn auto_backend() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    // ---- T3 Oracle Tests ----

    #[test]
    fn test_matmul_transposed_f32_oracle() {
        let (m, k, n) = (1, 128, 64);
        let a_data = gen_data(m * k, 42);
        let b_data = gen_data(n * k, 137);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let a_s = make_f32_tensor(&scalar, vec![m, k], &a_data);
        let b_s = make_f32_tensor(&scalar, vec![n, k], &b_data);
        let mut out_s = make_f32_tensor_zeros(&scalar, vec![m, n]);

        let a_a = make_f32_tensor(&auto, vec![m, k], &a_data);
        let b_a = make_f32_tensor(&auto, vec![n, k], &b_data);
        let mut out_a = make_f32_tensor_zeros(&auto, vec![m, n]);

        scalar.matmul_transposed(&a_s, &b_s, &mut out_s).unwrap();
        auto.matmul_transposed(&a_a, &b_a, &mut out_a).unwrap();

        assert_close(
            out_s.as_slice::<f32>(),
            out_a.as_slice::<f32>(),
            1e-4,
            "matmul_transposed_f32 [1,128]x[64,128]",
        );
    }

    #[test]
    fn test_matmul_transposed_f32_large_oracle() {
        let (m, k, n) = (8, 256, 128);
        let a_data = gen_data(m * k, 53);
        let b_data = gen_data(n * k, 97);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let a_s = make_f32_tensor(&scalar, vec![m, k], &a_data);
        let b_s = make_f32_tensor(&scalar, vec![n, k], &b_data);
        let mut out_s = make_f32_tensor_zeros(&scalar, vec![m, n]);

        let a_a = make_f32_tensor(&auto, vec![m, k], &a_data);
        let b_a = make_f32_tensor(&auto, vec![n, k], &b_data);
        let mut out_a = make_f32_tensor_zeros(&auto, vec![m, n]);

        scalar.matmul_transposed(&a_s, &b_s, &mut out_s).unwrap();
        auto.matmul_transposed(&a_a, &b_a, &mut out_a).unwrap();

        assert_close(
            out_s.as_slice::<f32>(),
            out_a.as_slice::<f32>(),
            1e-4,
            "matmul_transposed_f32 [8,256]x[128,256]",
        );
    }

    #[test]
    fn test_matmul_transposed_q4_0_oracle() {
        let (m, k, n) = (1, 128, 64);
        let a_data = gen_data(m * k, 42);
        let b_data_f32 = gen_data(n * k, 137);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let a_s = make_f32_tensor(&scalar, vec![m, k], &a_data);
        let b_s = make_q4_0_tensor(&scalar, vec![n, k], &b_data_f32);
        let mut out_s = make_f32_tensor_zeros(&scalar, vec![m, n]);

        let a_a = make_f32_tensor(&auto, vec![m, k], &a_data);
        let b_a = make_q4_0_tensor(&auto, vec![n, k], &b_data_f32);
        let mut out_a = make_f32_tensor_zeros(&auto, vec![m, n]);

        scalar.matmul_transposed(&a_s, &b_s, &mut out_s).unwrap();
        auto.matmul_transposed(&a_a, &b_a, &mut out_a).unwrap();

        assert_close(
            out_s.as_slice::<f32>(),
            out_a.as_slice::<f32>(),
            1e-2,
            "matmul_transposed_q4_0",
        );
    }

    #[test]
    fn test_matmul_transposed_q4_1_oracle() {
        let (m, k, n) = (1, 128, 64);
        let a_data = gen_data(m * k, 42);
        let b_data_f32 = gen_data(n * k, 137);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let a_s = make_f32_tensor(&scalar, vec![m, k], &a_data);
        let b_s = make_q4_1_tensor(&scalar, vec![n, k], &b_data_f32);
        let mut out_s = make_f32_tensor_zeros(&scalar, vec![m, n]);

        let a_a = make_f32_tensor(&auto, vec![m, k], &a_data);
        let b_a = make_q4_1_tensor(&auto, vec![n, k], &b_data_f32);
        let mut out_a = make_f32_tensor_zeros(&auto, vec![m, n]);

        scalar.matmul_transposed(&a_s, &b_s, &mut out_s).unwrap();
        auto.matmul_transposed(&a_a, &b_a, &mut out_a).unwrap();

        assert_close(
            out_s.as_slice::<f32>(),
            out_a.as_slice::<f32>(),
            1e-2,
            "matmul_transposed_q4_1",
        );
    }

    #[test]
    fn test_matmul_slice_f32_oracle() {
        let (m, k, n) = (4, 128, 64);
        let a_data = gen_data(m * k, 31);
        let b_data = gen_data(n * k, 71);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let a_s = make_f32_tensor(&scalar, vec![m, k], &a_data);
        let b_s = make_f32_tensor(&scalar, vec![n, k], &b_data);
        let mut out_s = make_f32_tensor_zeros(&scalar, vec![m, n]);

        let a_a = make_f32_tensor(&auto, vec![m, k], &a_data);
        let b_a = make_f32_tensor(&auto, vec![n, k], &b_data);
        let mut out_a = make_f32_tensor_zeros(&auto, vec![m, n]);

        scalar.matmul_slice(&a_s, &b_s, m, n, &mut out_s).unwrap();
        auto.matmul_slice(&a_a, &b_a, m, n, &mut out_a).unwrap();

        assert_close(
            out_s.as_slice::<f32>(),
            out_a.as_slice::<f32>(),
            1e-4,
            "matmul_slice_f32",
        );
    }

    fn make_f16_tensor(backend: &Arc<dyn Backend>, shape: Vec<usize>, f32_data: &[f32]) -> Tensor {
        let memory = Galloc::new();
        let numel: usize = shape.iter().product();
        assert_eq!(f32_data.len(), numel);
        let buf = memory.alloc(numel * 2, DType::F16).unwrap();
        let f16_data: Vec<half::f16> = f32_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(
                f16_data.as_ptr() as *const u8,
                buf.as_mut_ptr(),
                numel * 2,
            );
        }
        Tensor::new(Shape::new(shape), buf, backend.clone())
    }

    #[test]
    fn test_matmul_transposed_f16_oracle() {
        let (m, k, n) = (1, 128, 64);
        let a_data = gen_data(m * k, 42);
        let b_data = gen_data(n * k, 137);

        let backend = scalar_backend();

        // Reference: F32 x F32
        let a_f32 = make_f32_tensor(&backend, vec![m, k], &a_data);
        let b_f32 = make_f32_tensor(&backend, vec![n, k], &b_data);
        let mut out_ref = make_f32_tensor_zeros(&backend, vec![m, n]);
        backend
            .matmul_transposed(&a_f32, &b_f32, &mut out_ref)
            .unwrap();

        // Test: F32 x F16
        let a_f32_2 = make_f32_tensor(&backend, vec![m, k], &a_data);
        let b_f16 = make_f16_tensor(&backend, vec![n, k], &b_data);
        let mut out_f16 = make_f32_tensor_zeros(&backend, vec![m, n]);
        backend
            .matmul_transposed(&a_f32_2, &b_f16, &mut out_f16)
            .unwrap();

        // F16 precision: ~1e-3 relative error
        assert_close(
            out_ref.as_slice::<f32>(),
            out_f16.as_slice::<f32>(),
            5e-2,
            "matmul_transposed_f16 [1,128]x[64,128]",
        );
    }

    #[test]
    fn test_matmul_transposed_f16_large_oracle() {
        let (m, k, n) = (8, 256, 128);
        let a_data = gen_data(m * k, 53);
        let b_data = gen_data(n * k, 97);

        let backend = scalar_backend();

        // Reference: F32 x F32
        let a_f32 = make_f32_tensor(&backend, vec![m, k], &a_data);
        let b_f32 = make_f32_tensor(&backend, vec![n, k], &b_data);
        let mut out_ref = make_f32_tensor_zeros(&backend, vec![m, n]);
        backend
            .matmul_transposed(&a_f32, &b_f32, &mut out_ref)
            .unwrap();

        // Test: F32 x F16
        let a_f32_2 = make_f32_tensor(&backend, vec![m, k], &a_data);
        let b_f16 = make_f16_tensor(&backend, vec![n, k], &b_data);
        let mut out_f16 = make_f32_tensor_zeros(&backend, vec![m, n]);
        backend
            .matmul_transposed(&a_f32_2, &b_f16, &mut out_f16)
            .unwrap();

        assert_close(
            out_ref.as_slice::<f32>(),
            out_f16.as_slice::<f32>(),
            5e-2,
            "matmul_transposed_f16 [8,256]x[128,256]",
        );
    }

    #[test]
    fn test_rms_norm_oracle() {
        let (rows, dim) = (4, 64);
        let x_data = gen_data(rows * dim, 19);
        let w_data: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 * 0.01)).collect();

        let scalar = scalar_backend();
        let auto = auto_backend();

        let mut x_s = make_f32_tensor(&scalar, vec![rows, dim], &x_data);
        let w_s = make_f32_tensor(&scalar, vec![dim], &w_data);
        let mut x_a = make_f32_tensor(&auto, vec![rows, dim], &x_data);
        let w_a = make_f32_tensor(&auto, vec![dim], &w_data);

        scalar.rms_norm(&mut x_s, &w_s, 1e-5, false).unwrap();
        auto.rms_norm(&mut x_a, &w_a, 1e-5, false).unwrap();

        assert_close(
            x_s.as_slice::<f32>(),
            x_a.as_slice::<f32>(),
            1e-5,
            "rms_norm",
        );
    }

    #[test]
    fn test_rms_norm_add_unit() {
        // input [1.0, 2.0, 3.0, 4.0], weight [0.1, 0.2, 0.3, 0.4]
        // rms = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps)
        // expected[i] = x[i] * (1.0 + w[i]) / rms
        let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let w_data = vec![0.1f32, 0.2, 0.3, 0.4];
        let eps = 1e-6f32;
        let dim = 4usize;

        let backend = scalar_backend();
        let mut x = make_f32_tensor(&backend, vec![1, dim], &x_data);
        let w = make_f32_tensor(&backend, vec![dim], &w_data);
        backend.rms_norm(&mut x, &w, eps, true).unwrap();

        let rms = (7.5_f32 + eps).sqrt();
        let expected: Vec<f32> = x_data
            .iter()
            .zip(w_data.iter())
            .map(|(&xi, &wi)| xi * (1.0 + wi) / rms)
            .collect();

        let result = x.as_slice::<f32>();
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "add_unit rms_norm[{}]: got={} expected={}",
                i,
                got,
                exp
            );
        }

        // Verify it differs from add_unit=false
        let mut x2 = make_f32_tensor(&backend, vec![1, dim], &x_data);
        let w2 = make_f32_tensor(&backend, vec![dim], &w_data);
        backend.rms_norm(&mut x2, &w2, eps, false).unwrap();

        let plain: Vec<f32> = x_data
            .iter()
            .zip(w_data.iter())
            .map(|(&xi, &wi)| xi * wi / rms)
            .collect();
        let result2 = x2.as_slice::<f32>();
        for (i, (&got, &exp)) in result2.iter().zip(plain.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "plain rms_norm[{}]: got={} expected={}",
                i,
                got,
                exp
            );
        }

        // add_unit=true results should differ from add_unit=false
        assert!(
            (result[0] - result2[0]).abs() > 1e-5,
            "add_unit and plain should differ"
        );
    }

    #[test]
    fn test_softmax_oracle() {
        let (rows, dim) = (4, 32);
        let x_data = gen_data(rows * dim, 23);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let mut x_s = make_f32_tensor(&scalar, vec![rows, dim], &x_data);
        let mut x_a = make_f32_tensor(&auto, vec![rows, dim], &x_data);

        scalar.softmax(&mut x_s).unwrap();
        auto.softmax(&mut x_a).unwrap();

        let result_s = x_s.as_slice::<f32>();
        let result_a = x_a.as_slice::<f32>();

        assert_close(result_s, result_a, 1e-5, "softmax");

        // Verify rows sum to 1.0
        for r in 0..rows {
            let sum: f32 = result_a[r * dim..(r + 1) * dim].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax row {} sum = {}, expected 1.0",
                r,
                sum
            );
        }
    }

    #[test]
    fn test_rope_oracle() {
        let (batch, seq, heads, dim) = (1, 4, 2, 64);
        let n = batch * seq * heads * dim;
        let x_data = gen_data(n, 37);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let mut x_s = make_f32_tensor(&scalar, vec![batch, seq, heads, dim], &x_data);
        let mut x_a = make_f32_tensor(&auto, vec![batch, seq, heads, dim], &x_data);

        scalar.rope_inplace(&mut x_s, 0, 10000.0).unwrap();
        auto.rope_inplace(&mut x_a, 0, 10000.0).unwrap();

        assert_close(
            x_s.as_slice::<f32>(),
            x_a.as_slice::<f32>(),
            1e-5,
            "rope_inplace",
        );
    }

    #[test]
    fn test_silu_mul_oracle() {
        let n = 128;
        let a_data = gen_data(n, 11);
        let b_data = gen_data(n, 29);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let mut a_s = make_f32_tensor(&scalar, vec![1, n], &a_data);
        let b_s = make_f32_tensor(&scalar, vec![1, n], &b_data);
        let mut a_a = make_f32_tensor(&auto, vec![1, n], &a_data);
        let b_a = make_f32_tensor(&auto, vec![1, n], &b_data);

        scalar.silu_mul(&mut a_s, &b_s).unwrap();
        auto.silu_mul(&mut a_a, &b_a).unwrap();

        assert_close(
            a_s.as_slice::<f32>(),
            a_a.as_slice::<f32>(),
            1e-5,
            "silu_mul",
        );
    }

    #[test]
    fn test_silu_mul_large_values_oracle() {
        // Test with large values ([-100, 100]) to catch exp() approximation errors.
        // Qwen 2.5 gate values reach ~52, which triggered NaN with buggy v_expf.
        let n = 256;
        let a_data: Vec<f32> = (0..n)
            .map(|i| (i as f32 / n as f32) * 200.0 - 100.0) // [-100, 100]
            .collect();
        let b_data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 1.7 + 3.0) % 100.0) - 50.0) // [-50, 50]
            .collect();

        let scalar = scalar_backend();
        let auto = auto_backend();

        let mut a_s = make_f32_tensor(&scalar, vec![1, n], &a_data);
        let b_s = make_f32_tensor(&scalar, vec![1, n], &b_data);
        let mut a_a = make_f32_tensor(&auto, vec![1, n], &a_data);
        let b_a = make_f32_tensor(&auto, vec![1, n], &b_data);

        scalar.silu_mul(&mut a_s, &b_s).unwrap();
        auto.silu_mul(&mut a_a, &b_a).unwrap();

        let result_s = a_s.as_slice::<f32>();
        let result_a = a_a.as_slice::<f32>();

        // No NaN allowed
        assert!(
            !result_a.iter().any(|v| v.is_nan()),
            "silu_mul produced NaN for large values"
        );

        assert_close(result_s, result_a, 1e-4, "silu_mul_large_values");
    }

    #[test]
    fn test_add_assign_oracle() {
        let n = 64;
        let a_data = gen_data(n, 7);
        let b_data = gen_data(n, 13);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let mut a_s = make_f32_tensor(&scalar, vec![1, n], &a_data);
        let b_s = make_f32_tensor(&scalar, vec![1, n], &b_data);
        let mut a_a = make_f32_tensor(&auto, vec![1, n], &a_data);
        let b_a = make_f32_tensor(&auto, vec![1, n], &b_data);

        scalar.add_assign(&mut a_s, &b_s).unwrap();
        auto.add_assign(&mut a_a, &b_a).unwrap();

        assert_close(
            a_s.as_slice::<f32>(),
            a_a.as_slice::<f32>(),
            1e-7,
            "add_assign",
        );
    }

    #[test]
    fn test_scale_oracle() {
        let n = 64;
        let x_data = gen_data(n, 41);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let mut x_s = make_f32_tensor(&scalar, vec![1, n], &x_data);
        let mut x_a = make_f32_tensor(&auto, vec![1, n], &x_data);

        scalar.scale(&mut x_s, 0.5).unwrap();
        auto.scale(&mut x_a, 0.5).unwrap();

        assert_close(x_s.as_slice::<f32>(), x_a.as_slice::<f32>(), 1e-7, "scale");
    }

    #[test]
    fn test_cast_f32_to_f16_oracle() {
        let n = 64;
        let src_data = gen_data(n, 59);

        let scalar = scalar_backend();
        let auto = auto_backend();

        let src_s = make_f32_tensor(&scalar, vec![n], &src_data);
        let mut dst_s = make_f16_tensor_zeros(&scalar, vec![n]);
        let src_a = make_f32_tensor(&auto, vec![n], &src_data);
        let mut dst_a = make_f16_tensor_zeros(&auto, vec![n]);

        scalar.cast(&src_s, &mut dst_s).unwrap();
        auto.cast(&src_a, &mut dst_a).unwrap();

        // Compare as F16 bytes
        let s_data = dst_s.as_slice::<half::f16>();
        let a_data = dst_a.as_slice::<half::f16>();
        for i in 0..n {
            assert_eq!(
                s_data[i].to_bits(),
                a_data[i].to_bits(),
                "cast F32→F16 element [{}] differs",
                i
            );
        }
    }

    #[test]
    fn test_gather_oracle() {
        let (vocab, dim) = (8, 32);
        let embed_data = gen_data(vocab * dim, 67);
        let indices: Vec<u32> = vec![0, 3, 5, 7];
        let n_idx = indices.len();

        let scalar = scalar_backend();
        let auto = auto_backend();

        let memory = Galloc::new();

        // Embedding table
        let src_s = make_f32_tensor(&scalar, vec![vocab, dim], &embed_data);
        let src_a = make_f32_tensor(&auto, vec![vocab, dim], &embed_data);

        // Indices tensor (stored as U8 buffer holding u32 data)
        let idx_buf_s = memory.alloc(n_idx * 4, DType::U8).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                idx_buf_s.as_mut_ptr(),
                n_idx * 4,
            );
        }
        let idx_s = Tensor::new(Shape::new(vec![n_idx]), idx_buf_s, scalar.clone());

        let idx_buf_a = memory.alloc(n_idx * 4, DType::U8).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                idx_buf_a.as_mut_ptr(),
                n_idx * 4,
            );
        }
        let idx_a = Tensor::new(Shape::new(vec![n_idx]), idx_buf_a, auto.clone());

        // Output
        let mut dst_s = make_f32_tensor_zeros(&scalar, vec![n_idx, dim]);
        let mut dst_a = make_f32_tensor_zeros(&auto, vec![n_idx, dim]);

        scalar.gather(&src_s, &idx_s, &mut dst_s).unwrap();
        auto.gather(&src_a, &idx_a, &mut dst_a).unwrap();

        assert_close(
            dst_s.as_slice::<f32>(),
            dst_a.as_slice::<f32>(),
            1e-7,
            "gather",
        );

        // Also verify correctness: first output row should equal embed row 0
        let out_data = dst_a.as_slice::<f32>();
        for d in 0..dim {
            assert!(
                (out_data[d] - embed_data[d]).abs() < 1e-7,
                "gather correctness: row 0 mismatch at [{}]",
                d
            );
        }
    }

    #[test]
    fn test_copy_from_identity() {
        let n = 64;
        let data = gen_data(n, 73);

        let auto = auto_backend();
        let src = make_f32_tensor(&auto, vec![1, n], &data);

        let copied = auto.copy_from(&src).unwrap();

        assert_close(
            src.as_slice::<f32>(),
            copied.as_slice::<f32>(),
            0.0, // exact match
            "copy_from identity",
        );
        // Ensure different buffers (deep copy)
        assert_ne!(
            src.as_ptr(),
            copied.as_ptr(),
            "copy_from must allocate new buffer"
        );
    }

    // ---- generic floor (Q8_0 via descriptor) ----

    /// Quantize f32 data to Q8_0 blocks (row-major, row_len must be multiple of 32).
    /// Canonical layout per block: [d: f16 LE][qs: 32 × i8].
    fn quantize_q8_0(data: &[f32], n_rows: usize, row_len: usize) -> Vec<u8> {
        use crate::quant::QK8_0;
        let nb = row_len / QK8_0;
        let mut bytes = Vec::with_capacity(n_rows * nb * 34);
        for row in 0..n_rows {
            for bi in 0..nb {
                let offset = row * row_len + bi * QK8_0;
                let src = &data[offset..offset + QK8_0];
                let max_abs = src.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = max_abs / 127.0;
                let id = if d == 0.0 { 0.0 } else { 1.0 / d };
                bytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                for &v in src {
                    let q = (v * id).round().clamp(-128.0, 127.0) as i8;
                    bytes.push(q as u8);
                }
            }
        }
        bytes
    }

    fn make_q8_0_tensor(backend: &Arc<dyn Backend>, shape: Vec<usize>, f32_data: &[f32]) -> Tensor {
        let memory = Galloc::new();
        let n_rows = shape[0];
        let row_len = shape[1];
        let bytes = quantize_q8_0(f32_data, n_rows, row_len);
        let buf = memory.alloc(bytes.len(), DType::Q8_0).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr(), bytes.len());
        }
        Tensor::new(Shape::new(shape), buf, backend.clone())
    }

    /// (b) Q8_0 weight 가 generic floor 를 타고, 결과가 참조(수동 dequant → matmul_transposed_f32)
    /// 와 **bit-identical**(tol=0.0). floor 는 dequant 무손실 + 동일 f32 matmul 이라 exact.
    #[test]
    fn test_matmul_transposed_q8_0_floor_exact() {
        let (m, k, n) = (3, 128, 64);
        let a_data = gen_data(m * k, 42);
        let b_data_f32 = gen_data(n * k, 137);

        let scalar = scalar_backend();

        // Q8_0 weight tensor (floor 경로).
        let a = make_f32_tensor(&scalar, vec![m, k], &a_data);
        let b_q8 = make_q8_0_tensor(&scalar, vec![n, k], &b_data_f32);
        let mut out_floor = make_f32_tensor_zeros(&scalar, vec![m, n]);
        scalar.matmul_transposed(&a, &b_q8, &mut out_floor).unwrap();

        // Reference: 수동 dequant_via_descriptor → matmul_transposed_f32.
        let f32_b_vec = crate::format::dequant_via_descriptor(&b_q8).unwrap();
        let b_ref = make_f32_tensor(&scalar, vec![n, k], &f32_b_vec);
        let mut out_ref = make_f32_tensor_zeros(&scalar, vec![m, n]);
        CpuBackendCommon::new()
            .matmul_transposed_f32(&a, &b_ref, &mut out_ref)
            .unwrap();

        assert_close(
            out_floor.as_slice::<f32>(),
            out_ref.as_slice::<f32>(),
            0.0, // exact
            "Q8_0 floor matmul == manual dequant→f32 matmul",
        );
    }

    /// (c) Q4_0 weight 는 generic floor 가 아니라 **특화 arm**(matmul_transposed_q4_0)을 탄다.
    /// floor 는 q8_0-style block 으로 a 를 재양자화하지 않으므로, 특화 q4_0(a→q8_0 재양자화)와
    /// floor(f32 직접) 의 결과는 미세하게 다르다. 따라서 dispatch 결과 == 특화 함수 직접 호출이면
    /// 특화 arm 을 탄 것이 증명된다(floor 를 안 거침).
    #[test]
    fn test_matmul_transposed_q4_0_uses_specialized_not_floor() {
        let (m, k, n) = (3, 128, 64);
        let a_data = gen_data(m * k, 42);
        let b_data_f32 = gen_data(n * k, 137);

        let scalar = scalar_backend();
        let a = make_f32_tensor(&scalar, vec![m, k], &a_data);
        let b_q4 = make_q4_0_tensor(&scalar, vec![n, k], &b_data_f32);

        // dispatch 경유.
        let mut out_dispatch = make_f32_tensor_zeros(&scalar, vec![m, n]);
        scalar
            .matmul_transposed(&a, &b_q4, &mut out_dispatch)
            .unwrap();

        // 특화 함수 직접 호출.
        let mut out_specialized = make_f32_tensor_zeros(&scalar, vec![m, n]);
        CpuBackendCommon::new()
            .matmul_transposed_q4_0(&a, &b_q4, &mut out_specialized)
            .unwrap();

        assert_close(
            out_dispatch.as_slice::<f32>(),
            out_specialized.as_slice::<f32>(),
            0.0, // exact — same code path
            "Q4_0 dispatch must hit specialized arm (bit-identical to direct call)",
        );
    }
}
