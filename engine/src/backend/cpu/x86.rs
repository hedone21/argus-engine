#![allow(unused_unsafe, unused_variables)]
use crate::backend::Backend;
use crate::backend::cpu::common::CpuBackendCommon;
use crate::buffer::DType;
use crate::quant::{BlockQ4_0, BlockQ8_0, QK4_0, QK8_0};
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Minimum number of elements before using Rayon parallelism.
const PARALLEL_THRESHOLD: usize = 16384;

pub struct CpuBackendAVX2;

impl Default for CpuBackendAVX2 {
    fn default() -> Self {
        Self
    }
}

impl CpuBackendAVX2 {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackendAVX2 {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn name(&self) -> &str {
        "CPU (AVX2)"
    }

    fn device(&self) -> &str {
        "CPU"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        // Fallback or SIMD logic for standard matmul?
        // Let's rely on common for now as matmul is less critical than matmul_transposed for inference
        // Or specific AVX impl if needed.
        CpuBackendCommon::new().matmul(a, b, out)
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
            unsafe { Self::vec_add_f32(a_data, b_data) };
        } else {
            a_data
                .par_chunks_mut(PARALLEL_THRESHOLD)
                .zip(b_data.par_chunks(PARALLEL_THRESHOLD))
                .for_each(|(ac, bc)| unsafe { Self::vec_add_f32(ac, bc) });
        }
        Ok(())
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        let x_data = x.as_mut_slice::<f32>();
        if x_data.len() < PARALLEL_THRESHOLD {
            unsafe { Self::vec_scale_f32(x_data, v) };
        } else {
            x_data
                .par_chunks_mut(PARALLEL_THRESHOLD)
                .for_each(|chunk| unsafe { Self::vec_scale_f32(chunk, v) });
        }
        Ok(())
    }

    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        let a_data = a.as_mut_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        if a_data.len() < PARALLEL_THRESHOLD {
            unsafe { Self::vec_silu_mul_f32(a_data, b_data) };
        } else {
            a_data
                .par_chunks_mut(PARALLEL_THRESHOLD)
                .zip(b_data.par_chunks(PARALLEL_THRESHOLD))
                .for_each(|(ac, bc)| unsafe { Self::vec_silu_mul_f32(ac, bc) });
        }
        Ok(())
    }

    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        if add_unit {
            // add_unit path: fall back to scalar (Gemma3 only, not on hot path)
            return CpuBackendCommon::new().rms_norm(x, w, eps, add_unit);
        }
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let x_data = x.as_mut_slice::<f32>();
        let w_data = w.as_slice::<f32>();
        x_data.par_chunks_mut(dim).for_each(|row| {
            unsafe { Self::rms_norm_row(row, w_data, eps) };
        });
        Ok(())
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let x_data = x.as_mut_slice::<f32>();
        x_data.par_chunks_mut(dim).for_each(|row| {
            unsafe { Self::softmax_row(row) };
        });
        Ok(())
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        CpuBackendCommon::new().rope_inplace(x, start_pos, theta)
    }

    fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
        CpuBackendCommon::new().copy_from(t)
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        match (src.dtype(), dst.dtype()) {
            (DType::F32, DType::F16) => {
                let s = src.as_slice::<f32>();
                let d = dst.as_mut_slice::<half::f16>();
                unsafe { Self::cast_f32_to_f16(s, d) };
                Ok(())
            }
            (DType::F16, DType::F32) => {
                let s = src.as_slice::<half::f16>();
                let d = dst.as_mut_slice::<f32>();
                unsafe { Self::cast_f16_to_f32(s, d) };
                Ok(())
            }
            _ => CpuBackendCommon::new().cast(src, dst),
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

        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };

        #[derive(Clone, Copy)]
        struct SendPtr(*mut f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        match k_cache.dtype() {
            DType::F32 => {
                let k_data = k_cache.as_slice::<f32>();
                let v_data = v_cache.as_slice::<f32>();
                let scores_ptr = scores_out.as_ref().map(|s| SendPtr(s.as_ptr() as *mut f32));
                let scores_stride = scores_out
                    .as_ref()
                    .map(|s| s.len() / num_heads_q)
                    .unwrap_or(0);
                out_data
                    .par_chunks_mut(head_dim)
                    .enumerate()
                    .for_each(|(h, out_h)| {
                        let kv_h = h / gqa_ratio;
                        let q_off = h * head_dim;
                        let q_vec = &q_data[q_off..q_off + head_dim];
                        let mut scores = vec![0.0f32; cache_seq_len];

                        // Q * K^T with AVX2
                        for t in 0..cache_seq_len {
                            let off = if is_head_major {
                                (kv_h * capacity + t) * head_dim
                            } else {
                                (t * num_heads_kv + kv_h) * head_dim
                            };
                            let k_vec = &k_data[off..off + head_dim];
                            scores[t] = unsafe { Self::vec_dot_f32(q_vec, k_vec) * scale };
                        }

                        // Inline softmax with AVX2
                        unsafe { Self::softmax_row(&mut scores) };

                        // Copy softmax scores to scores_out if provided
                        if let Some(SendPtr(ptr)) = scores_ptr {
                            unsafe {
                                let dst = std::slice::from_raw_parts_mut(
                                    ptr.add(h * scores_stride),
                                    cache_seq_len,
                                );
                                dst.copy_from_slice(&scores[..cache_seq_len]);
                            }
                        }

                        // Weighted V sum with AVX2
                        for d in out_h.iter_mut() {
                            *d = 0.0;
                        }
                        for t in 0..cache_seq_len {
                            let w = scores[t];
                            let off = if is_head_major {
                                (kv_h * capacity + t) * head_dim
                            } else {
                                (t * num_heads_kv + kv_h) * head_dim
                            };
                            let v_vec = &v_data[off..off + head_dim];
                            unsafe { Self::vec_fma_f32(out_h, v_vec, w) };
                        }
                    });
            }
            DType::F16 => {
                let k_data = k_cache.as_slice::<half::f16>();
                let v_data = v_cache.as_slice::<half::f16>();
                let scores_ptr = scores_out.as_ref().map(|s| SendPtr(s.as_ptr() as *mut f32));
                let scores_stride = scores_out
                    .as_ref()
                    .map(|s| s.len() / num_heads_q)
                    .unwrap_or(0);
                out_data
                    .par_chunks_mut(head_dim)
                    .enumerate()
                    .for_each(|(h, out_h)| {
                        let kv_h = h / gqa_ratio;
                        let q_off = h * head_dim;
                        let q_vec = &q_data[q_off..q_off + head_dim];
                        let mut scores = vec![0.0f32; cache_seq_len];

                        // Q * K^T: inline F16C convert + FMA dot
                        for t in 0..cache_seq_len {
                            let off = if is_head_major {
                                (kv_h * capacity + t) * head_dim
                            } else {
                                (t * num_heads_kv + kv_h) * head_dim
                            };
                            let k_row = &k_data[off..off + head_dim];
                            scores[t] = unsafe {
                                Self::vec_dot_f16_f32_avx2(
                                    head_dim,
                                    q_vec.as_ptr(),
                                    k_row.as_ptr() as *const u16,
                                ) * scale
                            };
                        }

                        // Inline softmax with AVX2
                        unsafe { Self::softmax_row(&mut scores) };

                        // Copy softmax scores to scores_out if provided
                        if let Some(SendPtr(ptr)) = scores_ptr {
                            unsafe {
                                let dst = std::slice::from_raw_parts_mut(
                                    ptr.add(h * scores_stride),
                                    cache_seq_len,
                                );
                                dst.copy_from_slice(&scores[..cache_seq_len]);
                            }
                        }

                        // Weighted V sum: F16C + FMA
                        for d in out_h.iter_mut() {
                            *d = 0.0;
                        }
                        for t in 0..cache_seq_len {
                            let w = scores[t];
                            let off = if is_head_major {
                                (kv_h * capacity + t) * head_dim
                            } else {
                                (t * num_heads_kv + kv_h) * head_dim
                            };
                            let v_row = &v_data[off..off + head_dim];
                            unsafe {
                                Self::vec_mad_f16_avx2(
                                    head_dim,
                                    out_h.as_mut_ptr(),
                                    v_row.as_ptr() as *const u16,
                                    w,
                                );
                            }
                        }
                    });
            }
            DType::Q4_0 => {
                // Q4_0 attention: fall back to common (dequant path)
                return CpuBackendCommon::new().attention_gen(
                    q,
                    k_cache,
                    v_cache,
                    out,
                    num_heads_q,
                    num_heads_kv,
                    head_dim,
                    cache_seq_len,
                    scores_out,
                );
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

impl CpuBackendAVX2 {
    #[cfg(not(target_arch = "x86_64"))]
    fn matmul_transposed_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().matmul_transposed_f32(a, b, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::needless_range_loop)]
    fn matmul_transposed_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        if !is_x86_feature_detected!("avx2") {
            return CpuBackendCommon::new().matmul_transposed_f32(a, b, out);
        }

        unsafe {
            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();
            let a_rank = a_shape.len();
            let b_rank = b_shape.len();

            let k = a_shape[a_rank - 1];
            // Flatten M dimensions
            let m: usize = a_shape[..a_rank - 1].iter().product();
            let n = b_shape[b_rank - 2];

            let a_data = a.as_slice::<f32>();
            let b_data = b.as_slice::<f32>();
            let out_data = out.as_mut_slice::<f32>();

            // Heuristic: For very small matrices, threading overhead outweighs AVX gains.
            // Fallback to scalar/common implementation which might be serial or lighter weight.
            // Threshold tuned to 100k.
            if (m * n * k) < 100_000 {
                return self.matmul_transposed_f32_serial(a, b, out);
            }

            // Optimization for small M (e.g. generation M=1..8)
            // Parallelize over N instead of M to use all cores.
            // Use small chunks (64) to avoid false sharing while maintaining good load balance (64 tasks for N=4096).
            if m < 8 {
                out_data
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(idx, out_val)| {
                        let i = idx / n;
                        let j = idx % n;

                        let a_ptr = unsafe { a_data.as_ptr().add(i * k) };
                        let b_ptr = unsafe { b_data.as_ptr().add(j * k) };

                        let mut k_idx = 0;

                        unsafe {
                            let mut sum_v = _mm256_setzero_ps();

                            while k_idx + 32 <= k {
                                let va0 = _mm256_loadu_ps(a_ptr.add(k_idx));
                                let vb0 = _mm256_loadu_ps(b_ptr.add(k_idx));
                                sum_v = _mm256_fmadd_ps(va0, vb0, sum_v);

                                let va1 = _mm256_loadu_ps(a_ptr.add(k_idx + 8));
                                let vb1 = _mm256_loadu_ps(b_ptr.add(k_idx + 8));
                                sum_v = _mm256_fmadd_ps(va1, vb1, sum_v);

                                let va2 = _mm256_loadu_ps(a_ptr.add(k_idx + 16));
                                let vb2 = _mm256_loadu_ps(b_ptr.add(k_idx + 16));
                                sum_v = _mm256_fmadd_ps(va2, vb2, sum_v);

                                let va3 = _mm256_loadu_ps(a_ptr.add(k_idx + 24));
                                let vb3 = _mm256_loadu_ps(b_ptr.add(k_idx + 24));
                                sum_v = _mm256_fmadd_ps(va3, vb3, sum_v);

                                k_idx += 32;
                            }

                            while k_idx + 8 <= k {
                                let va = _mm256_loadu_ps(a_ptr.add(k_idx));
                                let vb = _mm256_loadu_ps(b_ptr.add(k_idx));
                                sum_v = _mm256_fmadd_ps(va, vb, sum_v);
                                k_idx += 8;
                            }

                            let mut temp = [0.0f32; 8];
                            _mm256_storeu_ps(temp.as_mut_ptr(), sum_v);
                            let mut sum = temp.iter().sum::<f32>();

                            // Tail
                            while k_idx < k {
                                sum += *a_ptr.add(k_idx) * *b_ptr.add(k_idx);
                                k_idx += 1;
                            }

                            *out_val = sum;
                        }
                    });
                return Ok(());
            }

            // For M in [8, 64), Blocked AVX2 (Block=8) reduces parallelism too much (e.g. M=8 -> 1 task).
            // Scalar (Row-parallel) is better.
            if m < 64 {
                return CpuBackendCommon::new().matmul_transposed_f32(a, b, out);
            }

            // Block Size M=8 (Process 8 rows of A at once to reuse B)
            // This reduces B matrix bandwidth by 8x.
            const BLOCK_M: usize = 8;

            out_data
                .par_chunks_mut(n * BLOCK_M)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_m = chunk_idx * BLOCK_M;
                    let rows_in_chunk = std::cmp::min(BLOCK_M, m - start_m);

                    // Prepare pointers for A rows
                    let mut a_ptrs = [std::ptr::null::<f32>(); BLOCK_M];
                    for r in 0..rows_in_chunk {
                        a_ptrs[r] = a_data.as_ptr().add((start_m + r) * k);
                    }

                    // Iterate over B rows (N)
                    for j in 0..n {
                        let b_ptr = b_data.as_ptr().add(j * k);

                        // Initialize accumulators
                        let mut sums = [_mm256_setzero_ps(); BLOCK_M];

                        let mut k_idx = 0;
                        // Main AVX Loop
                        while k_idx + 8 <= k {
                            let vb = _mm256_loadu_ps(b_ptr.add(k_idx));

                            // FMA for each active row in block
                            // We unroll the loop over rows manually for the compiler or use a loop (since const)
                            for r in 0..rows_in_chunk {
                                let va = _mm256_loadu_ps(a_ptrs[r].add(k_idx));
                                sums[r] = _mm256_fmadd_ps(va, vb, sums[r]);
                            }
                            k_idx += 8;
                        }

                        // Reduce sums and handle tail
                        for r in 0..rows_in_chunk {
                            // Horizontal sum of AVX register
                            let mut temp = [0.0f32; 8];
                            _mm256_storeu_ps(temp.as_mut_ptr(), sums[r]);
                            let mut final_sum = temp.iter().sum::<f32>();

                            // Scalar tail
                            let ar_ptr = a_ptrs[r];
                            let mut tail_k = k_idx;
                            while tail_k < k {
                                final_sum += *ar_ptr.add(tail_k) * *b_ptr.add(tail_k);
                                tail_k += 1;
                            }

                            // Write to output
                            // out_chunk is flattened [BLOCK_M * N]
                            // r-th row of chunk, j-th col
                            out_chunk[r * n + j] = final_sum;
                        }
                    }
                });
            Ok(())
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn matmul_transposed_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().matmul_transposed_f16(a, b, out)
    }

    /// AVX2+F16C F16 matmul: A(F32) x B^T(F16) -> Out(F32)
    /// Uses _mm256_cvtph_ps (F16C) for F16→F32 + _mm256_fmadd_ps (FMA).
    #[cfg(target_arch = "x86_64")]
    fn matmul_transposed_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        if !is_x86_feature_detected!("f16c") || !is_x86_feature_detected!("avx2") {
            return CpuBackendCommon::new().matmul_transposed_f16(a, b, out);
        }

        unsafe {
            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();
            let a_rank = a_shape.len();
            let b_rank = b_shape.len();

            let k = a_shape[a_rank - 1];
            let m: usize = a_shape[..a_rank - 1].iter().product();
            let n = b_shape[b_rank - 2];

            let a_data = a.as_slice::<f32>();
            let b_data = b.as_slice::<half::f16>();
            let out_data = out.as_mut_slice::<f32>();

            if (m * n * k) < 100_000 {
                return self.matmul_transposed_f16_serial(a_data, b_data, out_data, m, n, k);
            }

            // Parallelize over output elements
            out_data
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, out_val)| {
                    let i = idx / n;
                    let j = idx % n;

                    unsafe {
                        let a_ptr = a_data.as_ptr().add(i * k);
                        let b_ptr = (b_data.as_ptr() as *const u16).add(j * k);
                        *out_val = Self::vec_dot_f16_f32_avx2(k, a_ptr, b_ptr);
                    }
                });
            Ok(())
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn matmul_transposed_f16_serial(
        &self,
        a_data: &[f32],
        b_data: &[half::f16],
        out_data: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        for i in 0..m {
            let a_ptr = unsafe { a_data.as_ptr().add(i * k) };
            for j in 0..n {
                let b_ptr = unsafe { (b_data.as_ptr() as *const u16).add(j * k) };
                out_data[i * n + j] = unsafe { Self::vec_dot_f16_f32_avx2(k, a_ptr, b_ptr) };
            }
        }
        Ok(())
    }

    /// AVX2+F16C dot product: A(F32) · B(F16).
    /// Processes 32 elements per iteration (4x unroll of 8-element F16C converts).
    ///
    /// # Safety
    /// `a_ptr` must point to at least `k` f32 values, `b_ptr` to at least `k` u16 (F16) values.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
    pub unsafe fn vec_dot_f16_f32_avx2(k: usize, a_ptr: *const f32, b_ptr: *const u16) -> f32 {
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut k_idx = 0;

            // Main loop: 32 elements per iteration
            while k_idx + 32 <= k {
                let b0 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(k_idx) as *const __m128i));
                let b1 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(k_idx + 8) as *const __m128i));
                let b2 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(k_idx + 16) as *const __m128i));
                let b3 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(k_idx + 24) as *const __m128i));

                let a0 = _mm256_loadu_ps(a_ptr.add(k_idx));
                let a1 = _mm256_loadu_ps(a_ptr.add(k_idx + 8));
                let a2 = _mm256_loadu_ps(a_ptr.add(k_idx + 16));
                let a3 = _mm256_loadu_ps(a_ptr.add(k_idx + 24));

                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                acc3 = _mm256_fmadd_ps(a3, b3, acc3);

                k_idx += 32;
            }

            // 8-element tail
            while k_idx + 8 <= k {
                let b0 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(k_idx) as *const __m128i));
                let a0 = _mm256_loadu_ps(a_ptr.add(k_idx));
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                k_idx += 8;
            }

            // Reduce 4 accumulators
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);

            // Horizontal sum
            let hi = _mm256_extractf128_ps(acc0, 1);
            let lo = _mm256_castps256_ps128(acc0);
            let sum128 = _mm_add_ps(hi, lo);
            let sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
            let mut sum = _mm_cvtss_f32(sum128);

            // Scalar tail
            while k_idx < k {
                sum += *a_ptr.add(k_idx) * half::f16::from_bits(*b_ptr.add(k_idx)).to_f32();
                k_idx += 1;
            }

            sum
        }
    }

    /// AVX2+F16C weighted accumulate: out[i] += weight * v_f16[i]
    /// Fused F16C convert + FMA — operates directly on F16 data.
    ///
    /// # Safety
    /// `out_ptr` must point to at least `k` f32 values (read-write),
    /// `v_ptr` to at least `k` u16 (F16) values.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
    pub unsafe fn vec_mad_f16_avx2(k: usize, out_ptr: *mut f32, v_ptr: *const u16, weight: f32) {
        unsafe {
            let w = _mm256_set1_ps(weight);
            let mut idx = 0;

            // Main loop: 32 elements per iteration
            while idx + 32 <= k {
                let v0 = _mm256_cvtph_ps(_mm_loadu_si128(v_ptr.add(idx) as *const __m128i));
                let v1 = _mm256_cvtph_ps(_mm_loadu_si128(v_ptr.add(idx + 8) as *const __m128i));
                let v2 = _mm256_cvtph_ps(_mm_loadu_si128(v_ptr.add(idx + 16) as *const __m128i));
                let v3 = _mm256_cvtph_ps(_mm_loadu_si128(v_ptr.add(idx + 24) as *const __m128i));

                _mm256_storeu_ps(
                    out_ptr.add(idx),
                    _mm256_fmadd_ps(w, v0, _mm256_loadu_ps(out_ptr.add(idx))),
                );
                _mm256_storeu_ps(
                    out_ptr.add(idx + 8),
                    _mm256_fmadd_ps(w, v1, _mm256_loadu_ps(out_ptr.add(idx + 8))),
                );
                _mm256_storeu_ps(
                    out_ptr.add(idx + 16),
                    _mm256_fmadd_ps(w, v2, _mm256_loadu_ps(out_ptr.add(idx + 16))),
                );
                _mm256_storeu_ps(
                    out_ptr.add(idx + 24),
                    _mm256_fmadd_ps(w, v3, _mm256_loadu_ps(out_ptr.add(idx + 24))),
                );

                idx += 32;
            }

            // 8-element tail
            while idx + 8 <= k {
                let v0 = _mm256_cvtph_ps(_mm_loadu_si128(v_ptr.add(idx) as *const __m128i));
                _mm256_storeu_ps(
                    out_ptr.add(idx),
                    _mm256_fmadd_ps(w, v0, _mm256_loadu_ps(out_ptr.add(idx))),
                );
                idx += 8;
            }

            // Scalar tail
            while idx < k {
                *out_ptr.add(idx) += weight * half::f16::from_bits(*v_ptr.add(idx)).to_f32();
                idx += 1;
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn matmul_transposed_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().matmul_transposed_q4_0(a, b, out)
    }

    #[cfg(target_arch = "x86_64")]
    // -------------------------------------------------------------------------
    // AVX2 Kernel Helpers (Ported from llama.cpp/arch/x86/quants.c)
    // -------------------------------------------------------------------------
    #[target_feature(enable = "avx2")]
    unsafe fn sum_i16_pairs_float(&self, x: __m256i) -> __m256 {
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let summed_pairs = _mm256_madd_epi16(ones, x);
            _mm256_cvtepi32_ps(summed_pairs)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_sum_us8_pairs_float(&self, ax: __m256i, sy: __m256i) -> __m256 {
        unsafe {
            // Perform multiplication and create 16-bit values
            let dot = _mm256_maddubs_epi16(ax, sy);
            self.sum_i16_pairs_float(dot)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_sum_i8_pairs_float(&self, x: __m256i, y: __m256i) -> __m256 {
        unsafe {
            // Get absolute values of x vectors
            let ax = _mm256_sign_epi8(x, x);
            // Sign the values of the y vectors
            let sy = _mm256_sign_epi8(y, x);
            self.mul_sum_us8_pairs_float(ax, sy)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn bytes_from_nibbles_32(&self, rsi: *const u8) -> __m256i {
        unsafe {
            // Load 16 bytes from memory
            let tmp = _mm_loadu_si128(rsi as *const __m128i);
            // Expand to 32 bytes: [low nibbles | high nibbles]
            // _mm256_set_m128i(hi, lo) -> inserts lo at 0, hi at 1.
            // We want (tmp >> 4) and tmp.
            let bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
            let low_mask = _mm256_set1_epi8(0x0F);
            _mm256_and_si256(low_mask, bytes)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum_float_8(&self, x: __m256) -> f32 {
        unsafe {
            let mut res = _mm256_extractf128_ps(x, 1);
            res = _mm_add_ps(res, _mm256_castps256_ps128(x));
            res = _mm_add_ps(res, _mm_movehl_ps(res, res));
            res = _mm_add_ss(res, _mm_movehdup_ps(res));
            _mm_cvtss_f32(res)
        }
    }

    // -------------------------------------------------------------------------
    // Core Kernel
    // -------------------------------------------------------------------------

    /// # Safety
    /// Caller must ensure `vx` and `vy` point to valid blocks and `n` is a multiple of `QK8_0`.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn vec_dot_q4_0_q8_0(
        &self,
        n: usize,
        s: &mut f32,
        vx: *const BlockQ4_0,
        vy: *const BlockQ8_0,
    ) {
        let nb = n / QK8_0;

        unsafe {
            let mut acc = _mm256_setzero_ps();

            // Offset to shift values from [0..15] to [-8..7]
            let off = _mm256_set1_epi8(8);

            for i in 0..nb {
                let x = &*vx.add(i);
                let y = &*vy.add(i);

                // Compute combined scale
                let d_val = x.d.to_f32() * y.d.to_f32();
                let d = _mm256_set1_ps(d_val);

                // Unpack Q4_0 nibbles -> bytes
                let mut qx = self.bytes_from_nibbles_32(x.qs.as_ptr());
                // Shift range
                qx = _mm256_sub_epi8(qx, off);

                // Load Q8_0 bytes
                let qy = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);

                // Dot product (i8*i8 accumulation)
                let q = self.mul_sum_i8_pairs_float(qx, qy);

                // Accumulate with scale
                acc = _mm256_fmadd_ps(d, q, acc);
            }

            *s = self.hsum_float_8(acc);
        }
    }

    // -------------------------------------------------------------------------
    // AVX2 Quantization Kernel
    // -------------------------------------------------------------------------

    /// # Safety
    /// Caller must ensure `k` is a multiple of `QK8_0` and slices are large enough.
    #[allow(clippy::needless_range_loop)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn quantize_row_q8_0(&self, x: &[f32], y: &mut [BlockQ8_0], k: usize) {
        assert!(k.is_multiple_of(QK8_0));
        let nb = k / QK8_0;

        let x_ptr = x.as_ptr();
        unsafe {
            // Offset for permutation to fix byte ordering after packing
            let perm_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            let sign_bit = _mm256_set1_ps(-0.0);

            for i in 0..nb {
                let src = x_ptr.add(i * QK8_0);

                // Load 32 floats (4x AVX registers)
                let mut v0 = _mm256_loadu_ps(src);
                let mut v1 = _mm256_loadu_ps(src.add(8));
                let mut v2 = _mm256_loadu_ps(src.add(16));
                let mut v3 = _mm256_loadu_ps(src.add(24));

                // Compute max(abs(v))
                let mut max_abs = _mm256_andnot_ps(sign_bit, v0);
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));

                // Reduce max_abs (8 floats -> 1 float)
                let max4 = _mm_max_ps(
                    _mm256_extractf128_ps(max_abs, 1),
                    _mm256_castps256_ps128(max_abs),
                );
                let max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
                let max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
                let max_scalar = _mm_cvtss_f32(max4);

                // Scale
                let d = max_scalar / 127.0;
                let id = if max_scalar != 0.0 {
                    127.0 / max_scalar
                } else {
                    0.0
                };

                y[i].d = half::f16::from_f32(d);

                let mul = _mm256_set1_ps(id);

                // Apply multiplier
                v0 = _mm256_mul_ps(v0, mul);
                v1 = _mm256_mul_ps(v1, mul);
                v2 = _mm256_mul_ps(v2, mul);
                v3 = _mm256_mul_ps(v3, mul);

                // Round to nearest integer
                v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST as i32);
                v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST as i32);
                v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST as i32);
                v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST as i32);

                // Convert to i32
                let mut i0 = _mm256_cvtps_epi32(v0);
                let i1 = _mm256_cvtps_epi32(v1);
                let mut i2 = _mm256_cvtps_epi32(v2);
                let i3 = _mm256_cvtps_epi32(v3);

                // Pack i32 -> i16 -> i8
                i0 = _mm256_packs_epi32(i0, i1);
                i2 = _mm256_packs_epi32(i2, i3);
                i0 = _mm256_packs_epi16(i0, i2);

                // Fix order
                i0 = _mm256_permutevar8x32_epi32(i0, perm_idx);

                // Store
                _mm256_storeu_si256(y[i].qs.as_mut_ptr() as *mut __m256i, i0);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn matmul_transposed_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        if !is_x86_feature_detected!("avx2") {
            return CpuBackendCommon::new().matmul_transposed_q4_0(a, b, out);
        }

        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let n = b_shape[b_rank - 2];
        let m: usize = a_shape[..a_rank - 1].iter().product();

        // Heuristic: For very small matrices, threading overhead outweighs AVX gains.
        // Bumped to 100k.
        if (m * n * k) < 100_000 {
            return CpuBackendCommon::new().matmul_transposed_q4_0(a, b, out);
        }

        let a_data = a.as_slice::<f32>();
        let nb_k = k / QK4_0;
        let total_blocks = n * nb_k;

        let out_data = out.as_mut_slice::<f32>();

        // M < 4: Fine-grained (parallelize N) to utilize cores because M is too small.
        // This is the optimized generation path.
        if m < 4 {
            // 1. Quantize A to Q8_0
            let nb_k_q8 = k / QK8_0;
            let total_q8_blocks = m * nb_k_q8;

            // Temp buffer allocation (naive vec for now)
            let mut a_q8 = vec![
                BlockQ8_0 {
                    d: half::f16::from_f32(0.0),
                    qs: [0; QK8_0]
                };
                total_q8_blocks
            ];

            // We can quantize rows in parallel depending on m, but m is small here.
            // Just serial loop over m is fine, or par_iter if m >= 2? M < 4. Serial is fine.
            for i in 0..m {
                let a_offset = i * k;
                let a_row = &a_data[a_offset..a_offset + k];
                let q8_row = &mut a_q8[i * nb_k_q8..(i + 1) * nb_k_q8];
                unsafe {
                    self.quantize_row_q8_0(a_row, q8_row, k);
                }
            }

            out_data
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, out_val)| {
                    let i = idx / n;
                    let j = idx % n;

                    let b_offset = j * nb_k;
                    let b_row_node = unsafe { b.as_ptr() as *const BlockQ4_0 };
                    let b_row_ptr = unsafe { b_row_node.add(b_offset) };

                    let a_row_ptr = unsafe { a_q8.as_ptr().add(i * nb_k_q8) };

                    let mut sum = 0.0;
                    unsafe {
                        self.vec_dot_q4_0_q8_0(k, &mut sum, b_row_ptr, a_row_ptr);
                    }
                    *out_val = sum;
                });
            Ok(())
        } else {
            // M >= 4: AVX2 with row-level parallelism
            // 1. Quantize all A rows to Q8_0
            let nb_k_q8 = k / QK8_0;
            let total_q8_blocks = m * nb_k_q8;

            let mut a_q8 = vec![
                BlockQ8_0 {
                    d: half::f16::from_f32(0.0),
                    qs: [0; QK8_0]
                };
                total_q8_blocks
            ];

            // Parallel quantization for large M
            a_q8.par_chunks_mut(nb_k_q8)
                .enumerate()
                .for_each(|(i, q8_row)| {
                    let a_offset = i * k;
                    let a_row = &a_data[a_offset..a_offset + k];
                    unsafe {
                        self.quantize_row_q8_0(a_row, q8_row, k);
                    }
                });

            // 2. Parallel over M rows: each row computes N dot products
            out_data
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, out_row)| {
                    let a_row_ptr = unsafe { a_q8.as_ptr().add(i * nb_k_q8) };
                    let b_base = unsafe { b.as_ptr() as *const BlockQ4_0 };

                    for (j, out_val) in out_row.iter_mut().enumerate() {
                        let b_row_ptr = unsafe { b_base.add(j * nb_k) };
                        let mut sum = 0.0f32;
                        unsafe {
                            self.vec_dot_q4_0_q8_0(k, &mut sum, b_row_ptr, a_row_ptr);
                        }
                        *out_val = sum;
                    }
                });
            Ok(())
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn matmul_transposed_q4_1(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().matmul_transposed_q4_1(a, b, out)
    }

    // Pure serial implementation for tiny matrices to avoid Rayon overhead
    fn matmul_transposed_f32_serial(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let _a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_rank - 2];
        let m: usize = a_shape[..a_shape.len() - 1].iter().product();

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();

        // Serial loop
        for i in 0..m {
            let a_offset = i * k;
            for j in 0..n {
                let b_offset = j * k;
                let mut sum = 0.0;
                // Auto-vectorization should handle this reasonably well
                for l in 0..k {
                    sum += a_data[a_offset + l] * b_data[b_offset + l];
                }
                out_data[i * n + j] = sum;
            }
        }
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    fn matmul_transposed_q4_1(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        if !is_x86_feature_detected!("avx2") {
            return CpuBackendCommon::new().matmul_transposed_q4_1(a, b, out);
        }
        CpuBackendCommon::new().matmul_transposed_q4_1(a, b, out)
    }

    // =========================================================
    // AVX2 F16↔F32 Bulk Cast
    // =========================================================

    /// AVX2+F16C bulk F32→F16 cast.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "f16c")]
    unsafe fn cast_f32_to_f16(src: &[f32], dst: &mut [half::f16]) {
        unsafe {
            let n = src.len();
            let s_ptr = src.as_ptr();
            let d_ptr = dst.as_mut_ptr() as *mut u16;
            let mut i = 0;
            while i + 8 <= n {
                let v = _mm256_loadu_ps(s_ptr.add(i));
                let h = _mm256_cvtps_ph(v, _MM_ROUND_NEAREST as i32);
                _mm_storeu_si128(d_ptr.add(i) as *mut __m128i, h);
                i += 8;
            }
            while i < n {
                *d_ptr.add(i) = half::f16::from_f32(*s_ptr.add(i)).to_bits();
                i += 1;
            }
        }
    }

    /// AVX2+F16C bulk F16→F32 cast.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "f16c")]
    unsafe fn cast_f16_to_f32(src: &[half::f16], dst: &mut [f32]) {
        unsafe {
            let n = src.len();
            let s_ptr = src.as_ptr() as *const u16;
            let d_ptr = dst.as_mut_ptr();
            let mut i = 0;
            while i + 8 <= n {
                let h = _mm_loadu_si128(s_ptr.add(i) as *const __m128i);
                let v = _mm256_cvtph_ps(h);
                _mm256_storeu_ps(d_ptr.add(i), v);
                i += 8;
            }
            while i < n {
                *d_ptr.add(i) = half::f16::from_bits(*s_ptr.add(i)).to_f32();
                i += 1;
            }
        }
    }

    // =========================================================
    // AVX2 Dot Product / FMA Helpers for Attention
    // =========================================================

    /// AVX2 dot product of two f32 slices.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn vec_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_ptr.add(i)),
                    _mm256_loadu_ps(b_ptr.add(i)),
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_ptr.add(i + 8)),
                    _mm256_loadu_ps(b_ptr.add(i + 8)),
                    acc1,
                );
                i += 16;
            }
            while i + 8 <= n {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_ptr.add(i)),
                    _mm256_loadu_ps(b_ptr.add(i)),
                    acc0,
                );
                i += 8;
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            let mut sum = Self::hsum_f32x8(acc0);
            while i < n {
                sum += *a_ptr.add(i) * *b_ptr.add(i);
                i += 1;
            }
            sum
        }
    }

    /// AVX2 fused multiply-add: out[i] += v[i] * weight
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn vec_fma_f32(out: &mut [f32], v: &[f32], weight: f32) {
        unsafe {
            let n = out.len();
            let o_ptr = out.as_mut_ptr();
            let v_ptr = v.as_ptr();
            let w = _mm256_set1_ps(weight);
            let mut i = 0;
            while i + 16 <= n {
                _mm256_storeu_ps(
                    o_ptr.add(i),
                    _mm256_fmadd_ps(
                        w,
                        _mm256_loadu_ps(v_ptr.add(i)),
                        _mm256_loadu_ps(o_ptr.add(i)),
                    ),
                );
                _mm256_storeu_ps(
                    o_ptr.add(i + 8),
                    _mm256_fmadd_ps(
                        w,
                        _mm256_loadu_ps(v_ptr.add(i + 8)),
                        _mm256_loadu_ps(o_ptr.add(i + 8)),
                    ),
                );
                i += 16;
            }
            while i + 8 <= n {
                _mm256_storeu_ps(
                    o_ptr.add(i),
                    _mm256_fmadd_ps(
                        w,
                        _mm256_loadu_ps(v_ptr.add(i)),
                        _mm256_loadu_ps(o_ptr.add(i)),
                    ),
                );
                i += 8;
            }
            while i < n {
                *o_ptr.add(i) += weight * *v_ptr.add(i);
                i += 1;
            }
        }
    }

    // =========================================================
    // AVX2 Element-wise Helpers
    // =========================================================

    /// Horizontal sum of 8 floats in AVX register (static version).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum_f32x8(x: __m256) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps(x, 1);
            let lo = _mm256_castps256_ps128(x);
            let sum128 = _mm_add_ps(hi, lo);
            let sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
            _mm_cvtss_f32(sum128)
        }
    }

    /// Horizontal max of 8 floats in AVX register.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hmax_f32x8(x: __m256) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps(x, 1);
            let lo = _mm256_castps256_ps128(x);
            let max128 = _mm_max_ps(hi, lo);
            let max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
            let max128 = _mm_max_ss(max128, _mm_movehdup_ps(max128));
            _mm_cvtss_f32(max128)
        }
    }

    /// Fast vectorized exp(x) for 8 floats.
    /// Cephes-based minimax polynomial, max relative error < 1.5e-7.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    #[allow(clippy::excessive_precision)]
    pub(crate) unsafe fn exp_ps(x: __m256) -> __m256 {
        unsafe {
            let exp_hi = _mm256_set1_ps(88.376_26f32);
            let exp_lo = _mm256_set1_ps(-88.376_26f32);
            let log2ef = _mm256_set1_ps(std::f32::consts::LOG2_E);
            let c1 = _mm256_set1_ps(0.693_359_4f32);
            let c2 = _mm256_set1_ps(-2.121_944_4e-4f32);
            let one = _mm256_set1_ps(1.0f32);

            // Minimax polynomial coefficients for (exp(r) - 1 - r) / r²
            let p0 = _mm256_set1_ps(1.987_569_1e-4f32);
            let p1 = _mm256_set1_ps(1.398_200_0e-3f32);
            let p2 = _mm256_set1_ps(8.333_452e-3f32);
            let p3 = _mm256_set1_ps(4.166_580e-2f32);
            let p4 = _mm256_set1_ps(1.666_666_6e-1f32);
            let p5 = _mm256_set1_ps(5.000_000_1e-1f32);

            // Clamp to prevent overflow/underflow
            let x = _mm256_min_ps(x, exp_hi);
            let x = _mm256_max_ps(x, exp_lo);

            // n = round(x * log2(e))
            let fx = _mm256_round_ps(_mm256_mul_ps(x, log2ef), _MM_ROUND_NEAREST as i32);

            // Range reduction: r = x - n * ln(2) using Cahan summation
            let r = _mm256_fnmadd_ps(fx, c1, x);
            let r = _mm256_fnmadd_ps(fx, c2, r);

            let z = _mm256_mul_ps(r, r);

            // Horner polynomial: P(r) such that exp(r) ≈ 1 + r + r²·P(r)
            let mut y = p0;
            y = _mm256_fmadd_ps(y, r, p1);
            y = _mm256_fmadd_ps(y, r, p2);
            y = _mm256_fmadd_ps(y, r, p3);
            y = _mm256_fmadd_ps(y, r, p4);
            y = _mm256_fmadd_ps(y, r, p5);
            y = _mm256_fmadd_ps(y, z, r);
            y = _mm256_add_ps(y, one);

            // 2^n via IEEE754 exponent manipulation
            let n = _mm256_cvtps_epi32(fx);
            let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_add_epi32(n, _mm256_set1_epi32(0x7f)),
                23,
            ));

            _mm256_mul_ps(y, pow2n)
        }
    }

    /// AVX2 vector add: a[i] += b[i]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn vec_add_f32(a: &mut [f32], b: &[f32]) {
        unsafe {
            let n = a.len();
            let a_ptr = a.as_mut_ptr();
            let b_ptr = b.as_ptr();
            let mut i = 0;
            while i + 32 <= n {
                let va0 = _mm256_loadu_ps(a_ptr.add(i));
                let vb0 = _mm256_loadu_ps(b_ptr.add(i));
                _mm256_storeu_ps(a_ptr.add(i), _mm256_add_ps(va0, vb0));
                let va1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                let vb1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                _mm256_storeu_ps(a_ptr.add(i + 8), _mm256_add_ps(va1, vb1));
                let va2 = _mm256_loadu_ps(a_ptr.add(i + 16));
                let vb2 = _mm256_loadu_ps(b_ptr.add(i + 16));
                _mm256_storeu_ps(a_ptr.add(i + 16), _mm256_add_ps(va2, vb2));
                let va3 = _mm256_loadu_ps(a_ptr.add(i + 24));
                let vb3 = _mm256_loadu_ps(b_ptr.add(i + 24));
                _mm256_storeu_ps(a_ptr.add(i + 24), _mm256_add_ps(va3, vb3));
                i += 32;
            }
            while i + 8 <= n {
                let va = _mm256_loadu_ps(a_ptr.add(i));
                let vb = _mm256_loadu_ps(b_ptr.add(i));
                _mm256_storeu_ps(a_ptr.add(i), _mm256_add_ps(va, vb));
                i += 8;
            }
            while i < n {
                *a_ptr.add(i) += *b_ptr.add(i);
                i += 1;
            }
        }
    }

    /// AVX2 vector scale: x[i] *= v
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn vec_scale_f32(x: &mut [f32], v: f32) {
        unsafe {
            let n = x.len();
            let ptr = x.as_mut_ptr();
            let vv = _mm256_set1_ps(v);
            let mut i = 0;
            while i + 32 <= n {
                _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), vv));
                _mm256_storeu_ps(
                    ptr.add(i + 8),
                    _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), vv),
                );
                _mm256_storeu_ps(
                    ptr.add(i + 16),
                    _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 16)), vv),
                );
                _mm256_storeu_ps(
                    ptr.add(i + 24),
                    _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 24)), vv),
                );
                i += 32;
            }
            while i + 8 <= n {
                _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), vv));
                i += 8;
            }
            while i < n {
                *ptr.add(i) *= v;
                i += 1;
            }
        }
    }

    /// AVX2 SiLU gate: a[i] = silu(a[i]) * b[i] = (a[i] / (1 + exp(-a[i]))) * b[i]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn vec_silu_mul_f32(a: &mut [f32], b: &[f32]) {
        unsafe {
            let n = a.len();
            let a_ptr = a.as_mut_ptr();
            let b_ptr = b.as_ptr();
            let one = _mm256_set1_ps(1.0f32);
            let sign_mask = _mm256_set1_ps(-0.0f32);
            let mut i = 0;
            while i + 8 <= n {
                let x = _mm256_loadu_ps(a_ptr.add(i));
                let bv = _mm256_loadu_ps(b_ptr.add(i));
                // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                let neg_x = _mm256_xor_ps(x, sign_mask);
                let exp_neg_x = Self::exp_ps(neg_x);
                let sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));
                let silu = _mm256_mul_ps(x, sigmoid);
                _mm256_storeu_ps(a_ptr.add(i), _mm256_mul_ps(silu, bv));
                i += 8;
            }
            while i < n {
                let x = *a_ptr.add(i);
                let silu_x = x / (1.0 + (-x).exp());
                *a_ptr.add(i) = silu_x * *b_ptr.add(i);
                i += 1;
            }
        }
    }

    /// AVX2 RMS norm for a single row.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn rms_norm_row(row: &mut [f32], w: &[f32], eps: f32) {
        unsafe {
            let dim = row.len();
            let ptr = row.as_mut_ptr();
            let w_ptr = w.as_ptr();

            // Pass 1: sum of squares with 2-accumulator unroll
            let mut sum_v0 = _mm256_setzero_ps();
            let mut sum_v1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= dim {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                sum_v0 = _mm256_fmadd_ps(v0, v0, sum_v0);
                sum_v1 = _mm256_fmadd_ps(v1, v1, sum_v1);
                i += 16;
            }
            while i + 8 <= dim {
                let v = _mm256_loadu_ps(ptr.add(i));
                sum_v0 = _mm256_fmadd_ps(v, v, sum_v0);
                i += 8;
            }
            sum_v0 = _mm256_add_ps(sum_v0, sum_v1);
            let mut sum_sq = Self::hsum_f32x8(sum_v0);
            while i < dim {
                sum_sq += *ptr.add(i) * *ptr.add(i);
                i += 1;
            }

            let scale = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
            let scale_v = _mm256_set1_ps(scale);

            // Pass 2: x[i] = x[i] * scale * w[i]
            i = 0;
            while i + 8 <= dim {
                let v = _mm256_loadu_ps(ptr.add(i));
                let wv = _mm256_loadu_ps(w_ptr.add(i));
                _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(_mm256_mul_ps(v, scale_v), wv));
                i += 8;
            }
            while i < dim {
                *ptr.add(i) = (*ptr.add(i) * scale) * *w_ptr.add(i);
                i += 1;
            }
        }
    }

    /// AVX2 softmax for a single row.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn softmax_row(row: &mut [f32]) {
        unsafe {
            let dim = row.len();
            let ptr = row.as_mut_ptr();

            // Pass 1: find max
            let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut i = 0;
            while i + 8 <= dim {
                max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(ptr.add(i)));
                i += 8;
            }
            let mut max_val = Self::hmax_f32x8(max_v);
            while i < dim {
                max_val = max_val.max(*ptr.add(i));
                i += 1;
            }

            // Pass 2: exp(x - max) and accumulate sum
            let max_v = _mm256_set1_ps(max_val);
            let mut sum_v = _mm256_setzero_ps();
            i = 0;
            while i + 8 <= dim {
                let x = _mm256_loadu_ps(ptr.add(i));
                let e = Self::exp_ps(_mm256_sub_ps(x, max_v));
                _mm256_storeu_ps(ptr.add(i), e);
                sum_v = _mm256_add_ps(sum_v, e);
                i += 8;
            }
            let mut sum_exp = Self::hsum_f32x8(sum_v);
            while i < dim {
                let e = (*ptr.add(i) - max_val).exp();
                *ptr.add(i) = e;
                sum_exp += e;
                i += 1;
            }

            // Pass 3: normalize
            let inv_sum = _mm256_set1_ps(1.0 / sum_exp);
            i = 0;
            while i + 8 <= dim {
                _mm256_storeu_ps(
                    ptr.add(i),
                    _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), inv_sum),
                );
                i += 8;
            }
            let inv = 1.0 / sum_exp;
            while i < dim {
                *ptr.add(i) *= inv;
                i += 1;
            }
        }
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackendCommon;
    use crate::buffer::{Buffer, DType};
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn gen_data(n: usize, seed: u32) -> Vec<f32> {
        let mut data = Vec::with_capacity(n);
        let mut s = seed;
        for _ in 0..n {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            data.push(((s >> 16) as i16 as f32) / 32768.0);
        }
        data
    }

    fn make_f32_tensor(backend: &Arc<dyn Backend>, shape: Vec<usize>, data: &[f32]) -> Tensor {
        let memory = Galloc::new();
        let n = data.len();
        let buf = memory.alloc(n * 4, DType::F32).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.as_mut_ptr(), n * 4);
        }
        Tensor::new(Shape::new(shape), buf, backend.clone())
    }

    fn make_f32_tensor_zeros(backend: &Arc<dyn Backend>, shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        let memory = Galloc::new();
        let buf = memory.alloc(n * 4, DType::F32).unwrap();
        unsafe {
            std::ptr::write_bytes(buf.as_mut_ptr(), 0, n * 4);
        }
        Tensor::new(Shape::new(shape), buf, backend.clone())
    }

    fn scalar_backend() -> Arc<dyn Backend> {
        Arc::new(CpuBackendCommon::new())
    }

    fn avx2_backend() -> Arc<dyn Backend> {
        Arc::new(CpuBackendAVX2::new())
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

    // ---- exp_ps accuracy test ----

    #[test]
    fn test_exp_ps_accuracy() {
        // Test within the clamped range [-88, 88] where exp_ps is precise
        let test_values: Vec<f32> = (-176..=176).map(|i| i as f32 * 0.5).collect();
        for &x in &test_values {
            let expected = x.exp();
            let result = unsafe {
                let v = _mm256_set1_ps(x);
                let r = CpuBackendAVX2::exp_ps(v);
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), r);
                out[0]
            };
            if expected.is_infinite() || expected < 1e-38 || expected == 0.0 {
                continue; // skip denormals and extremes
            }
            let rel_err = ((result - expected) / expected).abs();
            assert!(
                rel_err < 2e-6,
                "exp({}) = {} (expected {}), rel_err = {}",
                x,
                result,
                expected,
                rel_err
            );
        }
    }

    // ---- add_assign ----

    #[test]
    fn test_add_assign_avx2() {
        for &n in &[7, 64, 2048, 20000] {
            let a_data = gen_data(n, 7);
            let b_data = gen_data(n, 13);
            let scalar = scalar_backend();
            let avx2 = avx2_backend();

            let mut a_s = make_f32_tensor(&scalar, vec![n], &a_data);
            let b_s = make_f32_tensor(&scalar, vec![n], &b_data);
            let mut a_a = make_f32_tensor(&avx2, vec![n], &a_data);
            let b_a = make_f32_tensor(&avx2, vec![n], &b_data);

            scalar.add_assign(&mut a_s, &b_s).unwrap();
            avx2.add_assign(&mut a_a, &b_a).unwrap();

            assert_close(
                a_s.as_slice::<f32>(),
                a_a.as_slice::<f32>(),
                1e-7,
                &format!("add_assign n={}", n),
            );
        }
    }

    // ---- scale ----

    #[test]
    fn test_scale_avx2() {
        for &n in &[3, 64, 2048, 20000] {
            let x_data = gen_data(n, 41);
            let scalar = scalar_backend();
            let avx2 = avx2_backend();

            let mut x_s = make_f32_tensor(&scalar, vec![n], &x_data);
            let mut x_a = make_f32_tensor(&avx2, vec![n], &x_data);

            scalar.scale(&mut x_s, 2.5).unwrap();
            avx2.scale(&mut x_a, 2.5).unwrap();

            assert_close(
                x_s.as_slice::<f32>(),
                x_a.as_slice::<f32>(),
                1e-7,
                &format!("scale n={}", n),
            );
        }
    }

    // ---- silu_mul ----

    #[test]
    fn test_silu_mul_avx2() {
        for &n in &[3, 64, 2048, 20000] {
            let a_data = gen_data(n, 11);
            let b_data = gen_data(n, 29);
            let scalar = scalar_backend();
            let avx2 = avx2_backend();

            let mut a_s = make_f32_tensor(&scalar, vec![n], &a_data);
            let b_s = make_f32_tensor(&scalar, vec![n], &b_data);
            let mut a_a = make_f32_tensor(&avx2, vec![n], &a_data);
            let b_a = make_f32_tensor(&avx2, vec![n], &b_data);

            scalar.silu_mul(&mut a_s, &b_s).unwrap();
            avx2.silu_mul(&mut a_a, &b_a).unwrap();

            assert_close(
                a_s.as_slice::<f32>(),
                a_a.as_slice::<f32>(),
                1e-5,
                &format!("silu_mul n={}", n),
            );
        }
    }

    // ---- rms_norm ----

    #[test]
    fn test_rms_norm_avx2() {
        for &(rows, dim) in &[(1, 7), (4, 64), (2, 256)] {
            let x_data = gen_data(rows * dim, 19);
            let w_data: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 * 0.01)).collect();
            let scalar = scalar_backend();
            let avx2 = avx2_backend();

            let mut x_s = make_f32_tensor(&scalar, vec![rows, dim], &x_data);
            let w_s = make_f32_tensor(&scalar, vec![dim], &w_data);
            let mut x_a = make_f32_tensor(&avx2, vec![rows, dim], &x_data);
            let w_a = make_f32_tensor(&avx2, vec![dim], &w_data);

            scalar.rms_norm(&mut x_s, &w_s, 1e-5, false).unwrap();
            avx2.rms_norm(&mut x_a, &w_a, 1e-5, false).unwrap();

            assert_close(
                x_s.as_slice::<f32>(),
                x_a.as_slice::<f32>(),
                1e-5,
                &format!("rms_norm rows={} dim={}", rows, dim),
            );
        }
    }

    #[test]
    fn test_rms_norm_avx2_large() {
        // Large dim requires relaxed tolerance due to FP accumulation order
        let (rows, dim) = (2, 2048);
        let x_data = gen_data(rows * dim, 19);
        let w_data: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 * 0.01)).collect();
        let scalar = scalar_backend();
        let avx2 = avx2_backend();

        let mut x_s = make_f32_tensor(&scalar, vec![rows, dim], &x_data);
        let w_s = make_f32_tensor(&scalar, vec![dim], &w_data);
        let mut x_a = make_f32_tensor(&avx2, vec![rows, dim], &x_data);
        let w_a = make_f32_tensor(&avx2, vec![dim], &w_data);

        scalar.rms_norm(&mut x_s, &w_s, 1e-5, false).unwrap();
        avx2.rms_norm(&mut x_a, &w_a, 1e-5, false).unwrap();

        assert_close(
            x_s.as_slice::<f32>(),
            x_a.as_slice::<f32>(),
            5e-5,
            "rms_norm_large rows=2 dim=2048",
        );
    }

    // ---- softmax ----

    #[test]
    fn test_softmax_avx2() {
        for &(rows, dim) in &[(1, 3), (4, 32), (2, 2048)] {
            let x_data = gen_data(rows * dim, 23);
            let scalar = scalar_backend();
            let avx2 = avx2_backend();

            let mut x_s = make_f32_tensor(&scalar, vec![rows, dim], &x_data);
            let mut x_a = make_f32_tensor(&avx2, vec![rows, dim], &x_data);

            scalar.softmax(&mut x_s).unwrap();
            avx2.softmax(&mut x_a).unwrap();

            assert_close(
                x_s.as_slice::<f32>(),
                x_a.as_slice::<f32>(),
                1e-5,
                &format!("softmax rows={} dim={}", rows, dim),
            );
        }
    }

    // ---- edge cases ----

    #[test]
    fn test_softmax_all_zeros() {
        let dim = 64;
        let x_data = vec![0.0f32; dim];
        let avx2 = avx2_backend();
        let mut x = make_f32_tensor(&avx2, vec![1, dim], &x_data);
        avx2.softmax(&mut x).unwrap();
        let result = x.as_slice::<f32>();
        let expected = 1.0 / dim as f32;
        for &v in result {
            assert!((v - expected).abs() < 1e-6, "uniform softmax: got {}", v);
        }
    }

    // ---- attention_gen F32 ----

    #[test]
    fn test_attention_gen_f32_avx2() {
        let num_heads_q = 4;
        let num_heads_kv = 2;
        let head_dim = 64;
        let cache_seq_len = 16;
        let capacity = 32;

        // HeadMajor layout: [1, kv_heads, capacity, head_dim]
        let q_data = gen_data(num_heads_q * head_dim, 42);
        let k_data = gen_data(num_heads_kv * capacity * head_dim, 137);
        let v_data = gen_data(num_heads_kv * capacity * head_dim, 53);

        let scalar = scalar_backend();
        let avx2 = avx2_backend();

        let q_s = make_f32_tensor(&scalar, vec![num_heads_q, head_dim], &q_data);
        let k_s = make_f32_tensor(&scalar, vec![1, num_heads_kv, capacity, head_dim], &k_data);
        let v_s = make_f32_tensor(&scalar, vec![1, num_heads_kv, capacity, head_dim], &v_data);
        let mut out_s = make_f32_tensor_zeros(&scalar, vec![num_heads_q, head_dim]);

        let q_a = make_f32_tensor(&avx2, vec![num_heads_q, head_dim], &q_data);
        let k_a = make_f32_tensor(&avx2, vec![1, num_heads_kv, capacity, head_dim], &k_data);
        let v_a = make_f32_tensor(&avx2, vec![1, num_heads_kv, capacity, head_dim], &v_data);
        let mut out_a = make_f32_tensor_zeros(&avx2, vec![num_heads_q, head_dim]);

        scalar
            .attention_gen(
                &q_s,
                &k_s,
                &v_s,
                &mut out_s,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
                None,
            )
            .unwrap();
        avx2.attention_gen(
            &q_a,
            &k_a,
            &v_a,
            &mut out_a,
            num_heads_q,
            num_heads_kv,
            head_dim,
            cache_seq_len,
            None,
        )
        .unwrap();

        assert_close(
            out_s.as_slice::<f32>(),
            out_a.as_slice::<f32>(),
            1e-4,
            "attention_gen F32 HeadMajor",
        );
    }

    // ---- attention_gen F16 ----

    #[test]
    fn test_attention_gen_f16_avx2() {
        let num_heads_q = 4;
        let num_heads_kv = 2;
        let head_dim = 64;
        let cache_seq_len = 16;
        let capacity = 32;

        let q_data = gen_data(num_heads_q * head_dim, 42);
        // F16 KV cache
        let k_f32 = gen_data(num_heads_kv * capacity * head_dim, 137);
        let v_f32 = gen_data(num_heads_kv * capacity * head_dim, 53);
        let k_f16: Vec<half::f16> = k_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let v_f16: Vec<half::f16> = v_f32.iter().map(|&v| half::f16::from_f32(v)).collect();

        let scalar = scalar_backend();
        let avx2 = avx2_backend();

        let q_s = make_f32_tensor(&scalar, vec![num_heads_q, head_dim], &q_data);
        let q_a = make_f32_tensor(&avx2, vec![num_heads_q, head_dim], &q_data);

        let kv_shape = vec![1, num_heads_kv, capacity, head_dim];
        let kv_numel = num_heads_kv * capacity * head_dim;

        // Create F16 tensors
        let memory = Galloc::new();
        let k_buf_s = memory.alloc(kv_numel * 2, DType::F16).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                k_f16.as_ptr() as *const u8,
                k_buf_s.as_mut_ptr(),
                kv_numel * 2,
            );
        }
        let k_s = Tensor::new(Shape::new(kv_shape.clone()), k_buf_s, scalar.clone());

        let v_buf_s = memory.alloc(kv_numel * 2, DType::F16).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                v_f16.as_ptr() as *const u8,
                v_buf_s.as_mut_ptr(),
                kv_numel * 2,
            );
        }
        let v_s = Tensor::new(Shape::new(kv_shape.clone()), v_buf_s, scalar.clone());

        let k_buf_a = memory.alloc(kv_numel * 2, DType::F16).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                k_f16.as_ptr() as *const u8,
                k_buf_a.as_mut_ptr(),
                kv_numel * 2,
            );
        }
        let k_a = Tensor::new(Shape::new(kv_shape.clone()), k_buf_a, avx2.clone());

        let v_buf_a = memory.alloc(kv_numel * 2, DType::F16).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                v_f16.as_ptr() as *const u8,
                v_buf_a.as_mut_ptr(),
                kv_numel * 2,
            );
        }
        let v_a = Tensor::new(Shape::new(kv_shape), v_buf_a, avx2.clone());

        let mut out_s = make_f32_tensor_zeros(&scalar, vec![num_heads_q, head_dim]);
        let mut out_a = make_f32_tensor_zeros(&avx2, vec![num_heads_q, head_dim]);

        scalar
            .attention_gen(
                &q_s,
                &k_s,
                &v_s,
                &mut out_s,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
                None,
            )
            .unwrap();
        avx2.attention_gen(
            &q_a,
            &k_a,
            &v_a,
            &mut out_a,
            num_heads_q,
            num_heads_kv,
            head_dim,
            cache_seq_len,
            None,
        )
        .unwrap();

        assert_close(
            out_s.as_slice::<f32>(),
            out_a.as_slice::<f32>(),
            1e-3,
            "attention_gen F16 HeadMajor",
        );
    }

    // ---- cast F16↔F32 ----

    #[test]
    fn test_cast_f32_to_f16_avx2() {
        for &n in &[3, 64, 2048] {
            let src_data = gen_data(n, 59);
            let scalar = scalar_backend();
            let avx2 = avx2_backend();

            let src_s = make_f32_tensor(&scalar, vec![n], &src_data);
            let src_a = make_f32_tensor(&avx2, vec![n], &src_data);

            let memory = Galloc::new();
            let dst_buf_s = memory.alloc(n * 2, DType::F16).unwrap();
            let mut dst_s = Tensor::new(Shape::new(vec![n]), dst_buf_s, scalar.clone());
            let dst_buf_a = memory.alloc(n * 2, DType::F16).unwrap();
            let mut dst_a = Tensor::new(Shape::new(vec![n]), dst_buf_a, avx2.clone());

            scalar.cast(&src_s, &mut dst_s).unwrap();
            avx2.cast(&src_a, &mut dst_a).unwrap();

            let s_data = dst_s.as_slice::<half::f16>();
            let a_data = dst_a.as_slice::<half::f16>();
            for i in 0..n {
                assert_eq!(
                    s_data[i].to_bits(),
                    a_data[i].to_bits(),
                    "cast F32→F16 mismatch at [{}]",
                    i
                );
            }
        }
    }

    #[test]
    fn test_cast_f16_to_f32_avx2() {
        for &n in &[3, 64, 2048] {
            let f32_data = gen_data(n, 71);
            let f16_data: Vec<half::f16> =
                f32_data.iter().map(|&v| half::f16::from_f32(v)).collect();

            let scalar = scalar_backend();
            let avx2 = avx2_backend();

            let memory = Galloc::new();
            // Create F16 source tensors
            let src_buf_s = memory.alloc(n * 2, DType::F16).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f16_data.as_ptr() as *const u8,
                    src_buf_s.as_mut_ptr(),
                    n * 2,
                );
            }
            let src_s = Tensor::new(Shape::new(vec![n]), src_buf_s, scalar.clone());

            let src_buf_a = memory.alloc(n * 2, DType::F16).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f16_data.as_ptr() as *const u8,
                    src_buf_a.as_mut_ptr(),
                    n * 2,
                );
            }
            let src_a = Tensor::new(Shape::new(vec![n]), src_buf_a, avx2.clone());

            let mut dst_s = make_f32_tensor_zeros(&scalar, vec![n]);
            let mut dst_a = make_f32_tensor_zeros(&avx2, vec![n]);

            scalar.cast(&src_s, &mut dst_s).unwrap();
            avx2.cast(&src_a, &mut dst_a).unwrap();

            assert_close(
                dst_s.as_slice::<f32>(),
                dst_a.as_slice::<f32>(),
                0.0,
                &format!("cast F16→F32 n={}", n),
            );
        }
    }

    // ---- edge cases ----

    #[test]
    fn test_rms_norm_all_ones() {
        let dim = 64;
        let x_data = vec![1.0f32; dim];
        let w_data = vec![1.0f32; dim];
        let avx2 = avx2_backend();
        let mut x = make_f32_tensor(&avx2, vec![1, dim], &x_data);
        let w = make_f32_tensor(&avx2, vec![dim], &w_data);
        avx2.rms_norm(&mut x, &w, 1e-5, false).unwrap();
        let result = x.as_slice::<f32>();
        // RMS of all ones = 1.0, so scale ≈ 1.0, result ≈ 1.0
        for &v in result {
            assert!((v - 1.0).abs() < 1e-4, "rms_norm all ones: got {}", v);
        }
    }
}
