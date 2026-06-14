// Argus CUDA custom kernels -- compiled at runtime via NVRTC.
// All kernels use extern "C" for name-mangling-free symbol lookup.
// Targeting SM >= 7.2 (Jetson Xavier+).

#include <cuda_fp16.h>
#include <math_constants.h>

#ifndef INFINITY
#define INFINITY CUDART_INF_F
#endif

#define WARP_SIZE 32

// =================================================================
// Warp/Block reduction infrastructure
// =================================================================

__device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE);
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE));
    return x;
}

// Block-wide reduction using shared memory for inter-warp communication.
// Returns the reduced value to all threads (broadcast from thread 0).
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float brs_warps[32]; // max 32 warps per block (1024 threads)
    __shared__ float brs_result;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) brs_warps[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < (unsigned)num_warps) ? brs_warps[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0) brs_result = val;
    __syncthreads();
    return brs_result;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float brm_warps[32];
    __shared__ float brm_result;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) brm_warps[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < (unsigned)num_warps) ? brm_warps[threadIdx.x] : -INFINITY;
    if (warp_id == 0) val = warp_reduce_max(val);

    if (threadIdx.x == 0) brm_result = val;
    __syncthreads();
    return brm_result;
}

// =================================================================
// 1. RMS Norm (1 block = 1 row)
// =================================================================
// grid=(nrows,1,1), block=(min(ncols,1024),1,1)
// weight multiply included. add_unit support for Gemma3.
extern "C" __global__ void rms_norm_f32(
    float * dst, const float * x, const float * weight,
    int ncols, float eps, int add_unit)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float * row_x = x + (long long)row * ncols;
    float * row_dst = dst + (long long)row * ncols;

    // Parallel reduction for sum of squares across all threads in block
    float partial_sq = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = row_x[i];
        partial_sq += v * v;
    }
    float sum_sq = block_reduce_sum(partial_sq);
    __shared__ float rms_inv_shared;
    if (tid == 0)
        rms_inv_shared = rsqrtf(sum_sq / (float)ncols + eps);
    __syncthreads();

    float rms_inv = rms_inv_shared;

    // Apply: dst = x * rms_inv * weight
    for (int i = tid; i < ncols; i += blockDim.x) {
        float w = add_unit ? (1.0f + weight[i]) : weight[i];
        row_dst[i] = row_x[i] * rms_inv * w;
    }
}

// =================================================================
// Fused residual add + out-of-place RMS norm.
//     x[i] += residual[i]      (in-place residual accumulate)
//     dst[i] = x[i] * rms_inv(x) * weight[i]   (optionally (1 + w) for add_unit)
// Replaces the add_assign → rms_norm_oop pair on the decode hot path.
// One pass through memory for the residual add + sum-sq reduce, one more
// pass for the scaled write. Eliminates the intermediate add_assign launch
// + the add_assign's own DRAM round-trip on x.
// =================================================================
extern "C" __global__ void add_rms_norm_mul_f32(
    float * __restrict__ x,             // in/out: x += residual
    const float * __restrict__ residual,
    float * __restrict__ dst,           // out: rms_norm(x_new) * w
    const float * __restrict__ weight,
    int ncols, float eps, int add_unit)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float * row_x = x + (long long)row * ncols;
    float * row_dst = dst + (long long)row * ncols;
    const float * row_res = residual + (long long)row * ncols;

    // Accumulate residual in-place AND reduce sum of squares over the
    // updated x values.
    float partial_sq = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = row_x[i] + row_res[i];
        row_x[i] = v;
        partial_sq += v * v;
    }
    float sum_sq = block_reduce_sum(partial_sq);
    __shared__ float rms_inv_shared;
    if (tid == 0) {
        rms_inv_shared = rsqrtf(sum_sq / (float)ncols + eps);
    }
    __syncthreads();

    float rms_inv = rms_inv_shared;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float w = add_unit ? (1.0f + weight[i]) : weight[i];
        row_dst[i] = row_x[i] * rms_inv * w;
    }
}

// =================================================================
// 2. RoPE inplace (neox split-half, Llama/Qwen/Gemma3)
// =================================================================
// Supports both decode (seq_len=1) and prefill (seq_len>1).
// grid=(seq_len * n_heads, 1, 1), block=(head_dim/2, 1, 1)
// x layout: [seq_len, n_heads, head_dim] (flattened)
// Pair layout: (x[i], x[i + head_dim/2]) — matches CPU/OpenCL neox style.
extern "C" __global__ void rope_inplace_f32(
    float * x, int head_dim, int n_heads, int seq_len,
    int start_pos, float theta_base)
{
    int block_id = blockIdx.x;          // 0 .. seq_len*n_heads - 1
    int t = block_id / n_heads;         // sequence position within batch
    int h = block_id % n_heads;         // head index
    int i = threadIdx.x;                // dim pair index: 0..head_dim/2
    int half_dim = head_dim / 2;
    if (i >= half_dim) return;

    // freq = theta_base^(-2i/head_dim) = exp(-2i/head_dim * log(theta_base))
    // Using exp+log matches CPU's floating point behavior better than powf.
    float freq = expf(-2.0f * (float)i / (float)head_dim * logf(theta_base));
    float theta = (float)(start_pos + t) * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    int base = (t * n_heads + h) * head_dim;
    float x0 = x[base + i];
    float x1 = x[base + i + half_dim];
    x[base + i]            = x0 * cos_t - x1 * sin_t;
    x[base + i + half_dim] = x0 * sin_t + x1 * cos_t;
}

// =================================================================
// 3. Softmax (1 block = 1 row)
// =================================================================
// grid=(nrows,1,1), block=(min(ncols,1024),1,1)
extern "C" __global__ void softmax_f32(float * x, int ncols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float * row_x = x + row * ncols;

    // Pass 1: find max
    float max_val = -INFINITY;
    for (int i = tid; i < ncols; i += blockDim.x)
        max_val = fmaxf(max_val, row_x[i]);
    max_val = block_reduce_max(max_val);

    // Pass 2: exp(x - max) + sum
    float sum_exp = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float e = expf(row_x[i] - max_val);
        row_x[i] = e;
        sum_exp += e;
    }
    sum_exp = block_reduce_sum(sum_exp);

    // Pass 3: normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < ncols; i += blockDim.x)
        row_x[i] *= inv_sum;
}

// =================================================================
// 4. SiLU-Mul fused (SwiGLU: silu(gate) * up)
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void silu_mul_f32(
    float * gate, const float * up, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    float g = gate[i];
    gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

// =================================================================
// 5. GELU tanh mul (Gemma3 FFN)
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void gelu_tanh_mul_f32(
    float * gate, const float * up, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    float x = gate[i];
    // gelu_tanh(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
    gate[i] = 0.5f * x * (1.0f + t) * up[i];
}

// =================================================================
// 6. Add assign (element-wise)
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void add_assign_f32(float * a, const float * b, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    a[i] += b[i];
}

// =================================================================
// 7. Scale (element-wise)
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void scale_f32(float * x, float v, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    x[i] *= v;
}

// =================================================================
// 8. Cast F32 -> F16
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void cast_f32_to_f16(const float * src, half * dst, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    dst[i] = __float2half(src[i]);
}

// =================================================================
// 9. Cast F16 -> F32
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void cast_f16_to_f32(const half * src, float * dst, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    dst[i] = __half2float(src[i]);
}

// =================================================================
// 10. Add row bias
// =================================================================
// grid=(ceil(nrows*ncols/256),1,1), block=(256,1,1)
extern "C" __global__ void add_row_bias_f32(
    float * x, const float * bias, int ncols, int total)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    x[i] += bias[i % ncols];
}

// =================================================================
// 11. KV scatter F32->F16 (HeadMajor layout)
// =================================================================
// grid=(kv_heads, 1, 1), block=(head_dim, 1, 1)
extern "C" __global__ void kv_scatter_f32_to_f16(
    const float * k_src, const float * v_src,
    half * k_dst, half * v_dst,
    int head_dim, int capacity, int write_pos)
{
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;
    int src_off = h * head_dim + d;
    int dst_off = h * capacity * head_dim + write_pos * head_dim + d;
    k_dst[dst_off] = __float2half(k_src[src_off]);
    v_dst[dst_off] = __float2half(v_src[src_off]);
}

// =================================================================
// 11b. KV scatter F32->F16 BATCH (HeadMajor layout, prefill)
// =================================================================
// grid=(kv_heads, seq_len, 1), block=(head_dim, 1, 1)
// k_src layout: [seq_len, kv_heads, head_dim] (contiguous F32, last-dim = kv_heads*head_dim)
// k_dst layout: HeadMajor [kv_heads, capacity, head_dim] (F16)
extern "C" __global__ void kv_scatter_f32_to_f16_batch(
    const float * k_src, const float * v_src,
    half * k_dst, half * v_dst,
    int kv_heads, int head_dim, int capacity, int write_pos_start, int seq_len)
{
    int h = blockIdx.x;   // kv head index
    int s = blockIdx.y;   // sequence position within batch
    int d = threadIdx.x;  // dimension index
    if (h >= kv_heads || s >= seq_len || d >= head_dim) return;
    // src: contiguous [seq_len, kv_heads * head_dim] -> offset = s * kv_heads * head_dim + h * head_dim + d
    int src_off = (s * kv_heads + h) * head_dim + d;
    // dst: HeadMajor [kv_heads, capacity, head_dim] -> offset = h * capacity * head_dim + (write_pos_start + s) * head_dim + d
    int dst_off = h * capacity * head_dim + (write_pos_start + s) * head_dim + d;
    k_dst[dst_off] = __float2half(k_src[src_off]);
    v_dst[dst_off] = __float2half(v_src[src_off]);
}

// =================================================================
// 12. Gather F16 -> F32 (embedding lookup)
// =================================================================
// grid=(n_tokens, ceil(dim/256), 1), block=(256, 1, 1)
extern "C" __global__ void gather_f16(
    const half * embed, const int * indices, float * dst, int dim)
{
    int token = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= dim) return;
    int row = indices[token];
    dst[token * dim + col] = __half2float(embed[row * dim + col]);
}

// =================================================================
// 13. Attention gen NAIVE (single query decode, F32 KV cache)
// =================================================================
// Retained as fallback. Prefer flash_attn_decode_f32kv.
// grid=(n_heads_q, 1, 1), block=(BLOCK_SIZE, 1, 1)
// HeadMajor layout: k_cache[kv_h, pos, d] = k_cache[kv_h * capacity * head_dim + pos * head_dim + d]
// shared memory: cache_seq_len floats for scores + 33 floats for reduction scratch
//
// Optional post-softmax score export (Phase B: resilience/eviction):
//   scores_out: if non-NULL, layout [n_heads_q, score_stride] float32. Each head h
//               writes `cache_seq_len` normalized weights to scores_out[h*stride..].
//   score_stride: row stride in floats (>= cache_seq_len).
//   When scores_out == NULL (score_stride ignored), the kernel behaves identically
//   to the pre-Phase-B version (no additional writes, no branching in hot loop).
extern "C" __global__ void attention_gen_f32_naive(
    const float * q,
    const float * k_cache,
    const float * v_cache,
    float * out,
    int n_heads_q, int kv_heads, int head_dim,
    int capacity, int cache_seq_len,
    float * __restrict__ scores_out,
    int score_stride)
{
    int h = blockIdx.x;
    int kv_h = h / (n_heads_q / kv_heads);
    int tid = threadIdx.x;
    float scale = rsqrtf((float)head_dim);

    const float * q_vec = q + h * head_dim;
    const float * k_base = k_cache + kv_h * capacity * head_dim;
    const float * v_base = v_cache + kv_h * capacity * head_dim;

    // Dynamic shared memory: scores[cache_seq_len] + scratch[33]
    extern __shared__ float shmem[];
    float * scores = shmem;

    // Phase 1: QK^T scores (each thread handles multiple positions via stride)
    for (int t = tid; t < cache_seq_len; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_vec[d] * k_base[t * head_dim + d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: Softmax over scores
    // 2a: find max
    float max_val = -INFINITY;
    for (int t = tid; t < cache_seq_len; t += blockDim.x)
        max_val = fmaxf(max_val, scores[t]);
    max_val = block_reduce_max(max_val);

    // 2b: exp + sum
    float sum_exp = 0.0f;
    for (int t = tid; t < cache_seq_len; t += blockDim.x) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        sum_exp += e;
    }
    sum_exp = block_reduce_sum(sum_exp);

    // 2c: normalize
    float inv_sum = 1.0f / sum_exp;
    for (int t = tid; t < cache_seq_len; t += blockDim.x)
        scores[t] *= inv_sum;
    __syncthreads();

    // Optional: export post-softmax scores for eviction policies (H2O/D2O/QCF).
    // Branch on NULL is uniform across all threads in the block, so predication
    // has no divergence cost. When disabled, the loop body is never entered.
    if (scores_out != nullptr) {
        float * scores_row = scores_out + (size_t)h * (size_t)score_stride;
        for (int t = tid; t < cache_seq_len; t += blockDim.x) {
            scores_row[t] = scores[t];
        }
    }

    // Phase 3: Weighted sum of V -- each thread handles a subset of head_dim
    float * out_vec = out + h * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < cache_seq_len; t++) {
            acc += scores[t] * v_base[t * head_dim + d];
        }
        out_vec[d] = acc;
    }
}

// =================================================================
// 14. Attention gen NAIVE (single query decode, F16 KV cache)
// =================================================================
// Retained as fallback. Prefer flash_attn_decode_f16kv.
// See attention_gen_f32_naive for scores_out / score_stride semantics.
extern "C" __global__ void attention_gen_f16kv_naive(
    const float * q,
    const half * k_cache,
    const half * v_cache,
    float * out,
    int n_heads_q, int kv_heads, int head_dim,
    int capacity, int cache_seq_len,
    float * __restrict__ scores_out,
    int score_stride)
{
    int h = blockIdx.x;
    int kv_h = h / (n_heads_q / kv_heads);
    int tid = threadIdx.x;
    float scale = rsqrtf((float)head_dim);

    const float * q_vec = q + h * head_dim;
    const half * k_base = k_cache + kv_h * capacity * head_dim;
    const half * v_base = v_cache + kv_h * capacity * head_dim;

    extern __shared__ float shmem[];
    float * scores = shmem;

    // Phase 1: QK^T (F16 K dequantized on-the-fly)
    for (int t = tid; t < cache_seq_len; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_vec[d] * __half2float(k_base[t * head_dim + d]);
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: Softmax
    float max_val = -INFINITY;
    for (int t = tid; t < cache_seq_len; t += blockDim.x)
        max_val = fmaxf(max_val, scores[t]);
    max_val = block_reduce_max(max_val);

    float sum_exp = 0.0f;
    for (int t = tid; t < cache_seq_len; t += blockDim.x) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        sum_exp += e;
    }
    sum_exp = block_reduce_sum(sum_exp);

    float inv_sum = 1.0f / sum_exp;
    for (int t = tid; t < cache_seq_len; t += blockDim.x)
        scores[t] *= inv_sum;
    __syncthreads();

    // Optional: export post-softmax scores (Phase B resilience path). Uniform
    // NULL check across the block; disabled path has no hot-loop overhead.
    if (scores_out != nullptr) {
        float * scores_row = scores_out + (size_t)h * (size_t)score_stride;
        for (int t = tid; t < cache_seq_len; t += blockDim.x) {
            scores_row[t] = scores[t];
        }
    }

    // Phase 3: V weighted sum (F16 V dequantized on-the-fly)
    float * out_vec = out + h * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < cache_seq_len; t++) {
            acc += scores[t] * __half2float(v_base[t * head_dim + d]);
        }
        out_vec[d] = acc;
    }
}

// =================================================================
// 15. Flash Attention Prefill (FlashAttention-2 style online softmax)
// =================================================================
// Macro-generated kernels for head_dim = 64, 128, 256.
// Each thread processes one query row. K/V are loaded in tiles of BLOCK_N
// into shared memory cooperatively. Two KV positions are processed per
// inner iteration to halve rescaling overhead.
//
// Grid:  (ceil(seq_len / BLOCK_M), n_heads_q * batch_size, 1)
// Block: (BLOCK_M, 1, 1)
//
// Memory layout:
//   Q/O: [batch, seq_len, n_heads_q, head_dim]  (contiguous)
//   KV:  HeadMajor [batch, n_heads_kv, capacity, head_dim]
//
// Causal mask: k_pos > (cache_seq_len - seq_len + query_row) => -inf

#define FLASH_PREFILL_KERNEL_F32(SUFFIX, HD, BM, BN) \
extern "C" __global__ void flash_attn_prefill_f32_##SUFFIX( \
    const float* __restrict__ q, \
    const float* __restrict__ k_cache, \
    const float* __restrict__ v_cache, \
    float* __restrict__ out, \
    int n_heads_q, int n_heads_kv, \
    int seq_len, int cache_seq_len, int kv_capacity, \
    int batch_size) \
{ \
    int tile_row = blockIdx.x;  /* which BLOCK_M tile of queries */ \
    int head_batch = blockIdx.y; /* h_q + batch * n_heads_q */ \
    int tid = threadIdx.x;       /* thread within block = query row within tile */ \
    \
    int b = head_batch / n_heads_q; \
    int h_q = head_batch % n_heads_q; \
    int gqa_ratio = n_heads_q / n_heads_kv; \
    int h_kv = h_q / gqa_ratio; \
    \
    int my_query_row = tile_row * BM + tid; \
    int valid = (my_query_row < seq_len); \
    \
    float scale = rsqrtf((float)(HD)); \
    int causal_limit = valid ? (cache_seq_len - seq_len + my_query_row) : -1; \
    \
    /* KV base pointer (same for all threads in block — needed for cooperative load) */ \
    long long kv_base = ((long long)b * n_heads_kv + h_kv) * kv_capacity * (HD); \
    const float* k_base = k_cache + kv_base; \
    const float* v_base = v_cache + kv_base; \
    \
    /* Accumulator in registers */ \
    float o_acc[HD]; \
    for (int d = 0; d < (HD); d++) o_acc[d] = 0.0f; \
    float m_i = -INFINITY; \
    float l_i = 0.0f; \
    \
    /* Load Q into registers (valid threads only — invalid threads have OOB q_offset) */ \
    float q_reg[HD]; \
    if (valid) { \
        long long q_offset = ((long long)b * seq_len * n_heads_q + (long long)my_query_row * n_heads_q + h_q) * (HD); \
        const float* my_q = q + q_offset; \
        for (int d = 0; d < (HD); d++) q_reg[d] = my_q[d]; \
    } \
    \
    /* Shared memory for K and V tiles: BN * HD each */ \
    __shared__ float k_tile[BN * HD]; \
    __shared__ float v_tile[BN * HD]; \
    \
    /* Iterate over KV in tiles of BN */ \
    for (int kv_start = 0; kv_start < cache_seq_len; kv_start += BN) { \
        int kv_end = min(kv_start + BN, cache_seq_len); \
        int tile_len = kv_end - kv_start; \
        \
        /* Cooperative load of K and V tiles into shared memory */ \
        __syncthreads(); \
        for (int i = tid; i < tile_len * (HD); i += BM) { \
            int pos_local = i / (HD); \
            int dim = i % (HD); \
            int pos_global = kv_start + pos_local; \
            k_tile[pos_local * (HD) + dim] = k_base[pos_global * (HD) + dim]; \
            v_tile[pos_local * (HD) + dim] = v_base[pos_global * (HD) + dim]; \
        } \
        __syncthreads(); \
        \
        if (!valid) continue; \
        \
        /* Process pairs of KV positions to reduce rescaling */ \
        int p = 0; \
        for (; p + 1 < tile_len; p += 2) { \
            int kp0 = kv_start + p; \
            int kp1 = kv_start + p + 1; \
            float s0 = (kp0 <= causal_limit) ? 0.0f : -INFINITY; \
            float s1 = (kp1 <= causal_limit) ? 0.0f : -INFINITY; \
            if (kp0 <= causal_limit) { \
                float dot = 0.0f; \
                for (int d = 0; d < (HD); d++) dot += q_reg[d] * k_tile[p * (HD) + d]; \
                s0 = dot * scale; \
            } \
            if (kp1 <= causal_limit) { \
                float dot = 0.0f; \
                for (int d = 0; d < (HD); d++) dot += q_reg[d] * k_tile[(p+1) * (HD) + d]; \
                s1 = dot * scale; \
            } \
            float m_new = fmaxf(m_i, fmaxf(s0, s1)); \
            if (m_new > -INFINITY) { \
                float exp0 = expf(s0 - m_new); \
                float exp1 = expf(s1 - m_new); \
                float rescale = (m_i > -INFINITY) ? expf(m_i - m_new) : 0.0f; \
                l_i = l_i * rescale + exp0 + exp1; \
                for (int d = 0; d < (HD); d++) { \
                    o_acc[d] = o_acc[d] * rescale \
                             + exp0 * v_tile[p * (HD) + d] \
                             + exp1 * v_tile[(p+1) * (HD) + d]; \
                } \
                m_i = m_new; \
            } \
        } \
        /* Handle odd remainder */ \
        if (p < tile_len) { \
            int kp = kv_start + p; \
            float s = -INFINITY; \
            if (kp <= causal_limit) { \
                float dot = 0.0f; \
                for (int d = 0; d < (HD); d++) dot += q_reg[d] * k_tile[p * (HD) + d]; \
                s = dot * scale; \
            } \
            float m_new = fmaxf(m_i, s); \
            if (m_new > -INFINITY) { \
                float exp_s = expf(s - m_new); \
                float rescale = (m_i > -INFINITY) ? expf(m_i - m_new) : 0.0f; \
                l_i = l_i * rescale + exp_s; \
                for (int d = 0; d < (HD); d++) { \
                    o_acc[d] = o_acc[d] * rescale + exp_s * v_tile[p * (HD) + d]; \
                } \
                m_i = m_new; \
            } \
        } \
    } \
    \
    /* Final normalization and write output */ \
    if (valid) { \
        long long out_offset = ((long long)b * seq_len * n_heads_q + (long long)my_query_row * n_heads_q + h_q) * (HD); \
        float* my_out = out + out_offset; \
        if (l_i > 0.0f) { \
            float inv_l = 1.0f / l_i; \
            for (int d = 0; d < (HD); d++) my_out[d] = o_acc[d] * inv_l; \
        } else { \
            for (int d = 0; d < (HD); d++) my_out[d] = 0.0f; \
        } \
    } \
}

#define FLASH_PREFILL_KERNEL_F16KV(SUFFIX, HD, BM, BN) \
extern "C" __global__ void flash_attn_prefill_f16kv_##SUFFIX( \
    const float* __restrict__ q, \
    const half* __restrict__ k_cache, \
    const half* __restrict__ v_cache, \
    float* __restrict__ out, \
    int n_heads_q, int n_heads_kv, \
    int seq_len, int cache_seq_len, int kv_capacity, \
    int batch_size) \
{ \
    int tile_row = blockIdx.x; \
    int head_batch = blockIdx.y; \
    int tid = threadIdx.x; \
    \
    int b = head_batch / n_heads_q; \
    int h_q = head_batch % n_heads_q; \
    int gqa_ratio = n_heads_q / n_heads_kv; \
    int h_kv = h_q / gqa_ratio; \
    \
    int my_query_row = tile_row * BM + tid; \
    int valid = (my_query_row < seq_len); \
    \
    float scale = rsqrtf((float)(HD)); \
    int causal_limit = valid ? (cache_seq_len - seq_len + my_query_row) : -1; \
    \
    /* KV base pointer (same for all threads in block — needed for cooperative load) */ \
    long long kv_base = ((long long)b * n_heads_kv + h_kv) * kv_capacity * (HD); \
    const half* k_base = k_cache + kv_base; \
    const half* v_base = v_cache + kv_base; \
    \
    float o_acc[HD]; \
    for (int d = 0; d < (HD); d++) o_acc[d] = 0.0f; \
    float m_i = -INFINITY; \
    float l_i = 0.0f; \
    \
    /* Load Q into registers (valid threads only — invalid threads have OOB q_offset) */ \
    float q_reg[HD]; \
    if (valid) { \
        long long q_offset = ((long long)b * seq_len * n_heads_q + (long long)my_query_row * n_heads_q + h_q) * (HD); \
        const float* my_q = q + q_offset; \
        for (int d = 0; d < (HD); d++) q_reg[d] = my_q[d]; \
    } \
    \
    /* Shared memory for K and V tiles (stored as F32 after dequant) */ \
    __shared__ float k_tile[BN * HD]; \
    __shared__ float v_tile[BN * HD]; \
    \
    for (int kv_start = 0; kv_start < cache_seq_len; kv_start += BN) { \
        int kv_end = min(kv_start + BN, cache_seq_len); \
        int tile_len = kv_end - kv_start; \
        \
        __syncthreads(); \
        for (int i = tid; i < tile_len * (HD); i += BM) { \
            int pos_local = i / (HD); \
            int dim = i % (HD); \
            int pos_global = kv_start + pos_local; \
            k_tile[pos_local * (HD) + dim] = __half2float(k_base[pos_global * (HD) + dim]); \
            v_tile[pos_local * (HD) + dim] = __half2float(v_base[pos_global * (HD) + dim]); \
        } \
        __syncthreads(); \
        \
        if (!valid) continue; \
        \
        int p = 0; \
        for (; p + 1 < tile_len; p += 2) { \
            int kp0 = kv_start + p; \
            int kp1 = kv_start + p + 1; \
            float s0 = (kp0 <= causal_limit) ? 0.0f : -INFINITY; \
            float s1 = (kp1 <= causal_limit) ? 0.0f : -INFINITY; \
            if (kp0 <= causal_limit) { \
                float dot = 0.0f; \
                for (int d = 0; d < (HD); d++) dot += q_reg[d] * k_tile[p * (HD) + d]; \
                s0 = dot * scale; \
            } \
            if (kp1 <= causal_limit) { \
                float dot = 0.0f; \
                for (int d = 0; d < (HD); d++) dot += q_reg[d] * k_tile[(p+1) * (HD) + d]; \
                s1 = dot * scale; \
            } \
            float m_new = fmaxf(m_i, fmaxf(s0, s1)); \
            if (m_new > -INFINITY) { \
                float exp0 = expf(s0 - m_new); \
                float exp1 = expf(s1 - m_new); \
                float rescale = (m_i > -INFINITY) ? expf(m_i - m_new) : 0.0f; \
                l_i = l_i * rescale + exp0 + exp1; \
                for (int d = 0; d < (HD); d++) { \
                    o_acc[d] = o_acc[d] * rescale \
                             + exp0 * v_tile[p * (HD) + d] \
                             + exp1 * v_tile[(p+1) * (HD) + d]; \
                } \
                m_i = m_new; \
            } \
        } \
        if (p < tile_len) { \
            int kp = kv_start + p; \
            float s = -INFINITY; \
            if (kp <= causal_limit) { \
                float dot = 0.0f; \
                for (int d = 0; d < (HD); d++) dot += q_reg[d] * k_tile[p * (HD) + d]; \
                s = dot * scale; \
            } \
            float m_new = fmaxf(m_i, s); \
            if (m_new > -INFINITY) { \
                float exp_s = expf(s - m_new); \
                float rescale = (m_i > -INFINITY) ? expf(m_i - m_new) : 0.0f; \
                l_i = l_i * rescale + exp_s; \
                for (int d = 0; d < (HD); d++) { \
                    o_acc[d] = o_acc[d] * rescale + exp_s * v_tile[p * (HD) + d]; \
                } \
                m_i = m_new; \
            } \
        } \
    } \
    \
    /* Final normalization and write output */ \
    if (valid) { \
        long long out_offset = ((long long)b * seq_len * n_heads_q + (long long)my_query_row * n_heads_q + h_q) * (HD); \
        float* my_out = out + out_offset; \
        if (l_i > 0.0f) { \
            float inv_l = 1.0f / l_i; \
            for (int d = 0; d < (HD); d++) my_out[d] = o_acc[d] * inv_l; \
        } else { \
            for (int d = 0; d < (HD); d++) my_out[d] = 0.0f; \
        } \
    } \
}

// Generate F32 KV variants
FLASH_PREFILL_KERNEL_F32(dk64,  64,  32, 32)
FLASH_PREFILL_KERNEL_F32(dk128, 128, 32, 16)
FLASH_PREFILL_KERNEL_F32(dk256, 256, 32,  8)

// Generate F16 KV variants
FLASH_PREFILL_KERNEL_F16KV(dk64,  64,  32, 32)
FLASH_PREFILL_KERNEL_F16KV(dk128, 128, 32, 16)
FLASH_PREFILL_KERNEL_F16KV(dk256, 256, 32,  8)

// =================================================================
// 16. F16 GEMV (seq_len=1 decode fast path)
// =================================================================
// Decode-specific GEMV for transposed weight: out[n] = sum_k w[n,k] * x[k].
// Targets Llama 3.2 1B matmul dispatch at M=1 to bypass cuBLAS gemm_ex's
// warp-granularity overhead. Memory-bound by F16 weight streaming.
//
// Structure:
//   - 1 warp (32 threads) per output row (intra-warp __shfl reduce).
//   - N_DST rows per block → block = WARP_SIZE * N_DST threads.
//   - Weight is [N, K] row-major (transposed). Each warp streams its row;
//     the N_DST rows in a block share the same input vector (automatic L1
//     reuse — input is read per warp but hits the same 2K/8K cache lines).
//   - Vectorized load via half2 (2 F16 per transaction) with float2 FMA
//     to amortize global load latency. Scalar tail handles odd K.
//
// Launch:
//   grid = (ceil(N / N_DST), 1, 1), block = (WARP_SIZE * N_DST, 1, 1)
//
// Determinism:
//   Each row reduction is a fixed cascading __shfl_down_sync pattern on a
//   fixed lane ordering. Same input → same bit-exact output across runs.
//
// Alignment requirement:
//   half2 fast path assumes `weight` and `input` are 4-byte aligned and
//   K is even. All Llama 3.2 1B matmul dims (K ∈ {2048, 8192}) satisfy
//   this. For unusual K (odd), the host routes to the scalar kernel.

#define GEMV_N_DST 4
#define GEMV_WARP 32

// --- F16 weight × F16 input → F32 output, K even ---
extern "C" __global__ void gemv_f16_f16_f32(
    const half* __restrict__ weight,   // [N, K] row-major
    const half* __restrict__ input,    // [K]
    float*     __restrict__ output,    // [N]
    int K, int N)
{
    int warp_id = threadIdx.x / GEMV_WARP;     // 0..N_DST-1
    int lane    = threadIdx.x % GEMV_WARP;     // 0..31
    int row     = blockIdx.x * GEMV_N_DST + warp_id;
    if (row >= N) return;

    int k2 = K >> 1;                           // number of half2 elements
    const half2* w2 = reinterpret_cast<const half2*>(weight + (long long)row * K);
    const half2* x2 = reinterpret_cast<const half2*>(input);

    float acc = 0.0f;

    // Main loop: stride GEMV_WARP (32) over half2 pairs.
    for (int i = lane; i < k2; i += GEMV_WARP) {
        float2 wv = __half22float2(w2[i]);
        float2 xv = __half22float2(x2[i]);
        acc += wv.x * xv.x + wv.y * xv.y;
    }

    // Scalar tail (K odd — unlikely for Llama 3.2 1B but kept for safety).
    if ((K & 1) && lane == 0) {
        acc += __half2float(weight[(long long)row * K + (K - 1)])
             * __half2float(input[K - 1]);
    }

    // Deterministic warp reduce: cascading __shfl_down_sync.
    #pragma unroll
    for (int offset = GEMV_WARP / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset, GEMV_WARP);
    }
    if (lane == 0) output[row] = acc;
}

// --- F16 weight × F32 input → F32 output, K even ---
// Skips the F32→F16 cast of the activation vector entirely.
extern "C" __global__ void gemv_f16_f32_f32(
    const half*  __restrict__ weight,  // [N, K] row-major F16
    const float* __restrict__ input,   // [K] F32
    float*       __restrict__ output,  // [N]
    int K, int N)
{
    int warp_id = threadIdx.x / GEMV_WARP;
    int lane    = threadIdx.x % GEMV_WARP;
    int row     = blockIdx.x * GEMV_N_DST + warp_id;
    if (row >= N) return;

    int k2 = K >> 1;
    const half2*  w2 = reinterpret_cast<const half2*>(weight + (long long)row * K);
    const float2* x2 = reinterpret_cast<const float2*>(input);

    float acc = 0.0f;

    for (int i = lane; i < k2; i += GEMV_WARP) {
        float2 wv = __half22float2(w2[i]);
        float2 xv = x2[i];
        acc += wv.x * xv.x + wv.y * xv.y;
    }

    if ((K & 1) && lane == 0) {
        acc += __half2float(weight[(long long)row * K + (K - 1)]) * input[K - 1];
    }

    #pragma unroll
    for (int offset = GEMV_WARP / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset, GEMV_WARP);
    }
    if (lane == 0) output[row] = acc;
}

// --- Q4_0 weight × F32 input → F32 output ---
// Dequant-in-kernel GEMV for BlockQ4_0 layout: { half d; uint8_t qs[16] } = 18
// bytes per 32 elements. Low-nibble → positions 0..15, high → 16..31, with
// dequant value = ((nibble as i8) - 8) * d.
//
// Threading: 1 warp per output row, 32 lanes cooperate on one block at a time
// so lane-strided loads are fully coalesced:
//   - activation reads: 32 contiguous f32 per block (128 B, one coalesced txn).
//   - qs reads:         lanes 0..15 each fetch one byte (16 B coalesced); lanes
//                       16..31 reuse via __shfl_sync.
//   - d bits read:      lane 0 loads 2 bytes, broadcasts via __shfl_sync.
//
// K must be a multiple of 32. Weight pointer is byte-addressed; each row is
// (K/32) * 18 bytes. Block offset 18*b is always 2-aligned (we avoid a wider
// reinterpret_cast on a 2-aligned address).
// Launched with block_dim = (32, Q4_0_NWARPS, 1) and grid_dim = (N, M, 1).
// Each CUDA block handles one (output_row, token) pair; the Q4_0_NWARPS warps
// each process a strided subset of the row's Q4_0 blocks, then tree-reduce via
// SLM. Batching M tokens into the grid Y dimension amortises launch latency
// and removes the per-token host-side loop that dominated prefill on Jetson
// Xavier (~18K launches / prefill for Llama 3.1 8B at seq_len=81).
#ifndef Q4_0_NWARPS
#define Q4_0_NWARPS 4
#endif

extern "C" __global__ void gemv_q4_0_f32_f32(
    const unsigned char* __restrict__ weight,  // [N * (K/32) * 18] bytes
    const float*         __restrict__ input,   // [M * K] F32 (row-major, stride K)
    float*               __restrict__ output,  // [M * N] F32 (row-major, stride N)
    int K, int N)
{
    const int row  = blockIdx.x;
    if (row >= N) return;
    const int tok  = blockIdx.y;               // 0..M-1
    const int lane    = threadIdx.x;           // 0..31
    const int warp_id = threadIdx.y;           // 0..(Q4_0_NWARPS-1)

    const int n_blocks = K >> 5;               // K / 32
    const unsigned char* row_ptr =
        weight + (long long)row * n_blocks * 18;
    const float* input_tok = input + (long long)tok * K;

    float acc = 0.0f;

    // Warp `warp_id` strides over blocks with stride Q4_0_NWARPS so the
    // per-warp accumulators carry disjoint partial sums.
    for (int b = warp_id; b < n_blocks; b += Q4_0_NWARPS) {
        const unsigned char* bp = row_ptr + b * 18;

        unsigned short d_bits =
            (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = __half2float(__ushort_as_half(d_bits));

        int qs_idx = (lane < 16) ? lane : (lane - 16);
        unsigned char q_byte = bp[2 + qs_idx];
        int nib = (lane < 16) ? (int)(q_byte & 0xFu) : (int)(q_byte >> 4);
        int v   = nib - 8;

        float act = input_tok[b * 32 + lane];
        acc += (float)v * d * act;
    }

    // Intra-warp reduce.
    #pragma unroll
    for (int offset = GEMV_WARP / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset, GEMV_WARP);
    }

    // Cross-warp reduce via SLM (Q4_0_NWARPS values; warp 0 finalizes).
    __shared__ float partial[Q4_0_NWARPS];
    if (lane == 0) partial[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0 && lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < Q4_0_NWARPS; ++w) sum += partial[w];
        output[(long long)tok * N + row] = sum;
    }
}

// =================================================================
// Q8_1 activation quantization + Q4_0 × Q8_1 mmvq (dp4a integer dot).
// Ported from llama.cpp's mmvq.cu + quantize.cu (MIT license).
//
// Decode (M=1) fast path:
//   1. Launch `quantize_q8_1_f32` to compress the F32 activation vector
//      into (K/32) block_q8_1 records (36 B each = half2 ds + int8 qs[32]).
//   2. Launch `mul_mat_vec_q4_0_q8_1` which, for every output row, uses
//      dp4a to compute sum((w_q-8) * x_q) on 4-way packed int8 pairs,
//      then applies the per-block scales.
//
// Replaces dequant-in-kernel float GEMV (~5.9 tok/s on Xavier) with the
// llama.cpp-style integer path (~17 tok/s reference).
// =================================================================

#define QK8_1 32
#define QI8_1 8    // = QK8_1/4 (int32s of qs per Q8_1 block)
#define QK4_0 32
#define QI4_0 4    // = QK4_0/8 (int32s of qs per Q4_0 block)
#define VDR_Q4_0_Q8_1_MMVQ 2

// Quantize the activation vector(s) `x[M*K]` into Q8_1 blocks `vy[M*(K/32)*36]`.
// Each block_q8_1 record is 36 bytes: half2 ds (d=amax/127, sum=sum_x) + int8 qs[32].
// Grid: (ceil(K / blockDim.x), M, 1), block: (Q8_1_QUANT_BSIZE, 1, 1) with
// blockDim.x a multiple of QK8_1 so the warp reduction below is well-defined.
// For decode (M=1) this degenerates to the single-vector launch.
extern "C" __global__ void quantize_q8_1_f32(
    const float* __restrict__ x,
    unsigned char* __restrict__ vy,
    int K)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    const int tok = blockIdx.y;

    const int ib  = i >> 5;   // block index within a token (QK8_1 = 32)
    const int iqs = i & 31;   // quant index within block

    const int n_blocks = K >> 5;
    const float xi = x[(long long)tok * K + i];
    float amax = fabsf(xi);
    float sum  = xi;

    // Reduce amax/sum within the 32 lanes that share a Q8_1 block.
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
        sum  +=            __shfl_xor_sync(0xffffffff, sum,  mask, 32);
    }

    const float d = amax / 127.0f;
    const int   q = (amax == 0.0f) ? 0 : __float2int_rn(xi / d);

    // Layout: [half2 ds | int8 qs[32]] — 36 B, naturally 4-byte aligned.
    unsigned char* blk = vy + ((long long)tok * n_blocks + ib) * 36;
    signed char* qs = (signed char*)(blk + 4);
    qs[iqs] = (signed char)q;

    if (iqs == 0) {
        __half2* ds = (__half2*)blk;
        *ds = __floats2half2_rn(d, sum);
    }
}

// Q4_0 × Q8_1 mul_mat_vec (ncols_dst = 1, rows_per_cuda_block = 1).
// Grid: (N, 1, 1), block: (32, MMVQ_NWARPS, 1) with 4 warps per output row.
//
//   weight       : [N * (K/QK4_0) * 18] bytes (block_q4_0 = half d + uint8 qs[16])
//   activation_q : [ (K/QK8_1) * 36 ] bytes (block_q8_1 produced by the kernel above)
//   output       : [N] F32
//
// Per-thread: pick one Q4_0 block (kbx) and one of the 2 dot-product slots
// (kqs ∈ {0,2}), producing half of the block's contribution via two dp4a
// calls. 4 warps * 32 lanes = 128 threads process 64 blocks per iteration
// (blocks_per_iter = vdr * nwarps * warp_size / qi = 2*4*32/4 = 64).
#ifndef MMVQ_NWARPS
#define MMVQ_NWARPS 4
#endif

// Pin the block size so the CUDA compiler can size its register allocation
// against a known upper bound — matches llama.cpp's mmvq.cu launch_bounds
// and avoids the occupancy regression that happens when nvcc assumes
// 1024-thread blocks.
extern "C" __global__ __launch_bounds__(MMVQ_NWARPS * 32, 1) void mul_mat_vec_q4_0_q8_1(
    const unsigned char* __restrict__ weight,        // Q4_0 blocks
    const unsigned char* __restrict__ activation_q,  // Q8_1 blocks [M*(K/32)*36]
    float*               __restrict__ output,        // [M*N] F32
    int K, int N)
{
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tok = blockIdx.y;

    const int tid = 32 * threadIdx.y + threadIdx.x;   // 0..127
    const int blocks_per_row = K >> 5;                 // K / QK4_0
    const int blocks_per_iter = (VDR_Q4_0_Q8_1_MMVQ * MMVQ_NWARPS * 32) / QI4_0; // 64

    // Q4_0 block stride (bytes) and Q8_1 block stride (bytes).
    const int q4_block_bytes = 18;
    const int q8_block_bytes = 36;

    const unsigned char* w_row = weight + (long long)row * blocks_per_row * q4_block_bytes;
    const unsigned char* act_tok = activation_q
        + (long long)tok * blocks_per_row * q8_block_bytes;

    float tmp = 0.0f;

    // Each thread owns one (kbx, kqs) pair per iteration.
    //   kbx = tid / (qi/vdr) = tid / 2
    //   kqs = vdr * (tid % (qi/vdr)) = 2 * (tid % 2)  ∈ {0, 2}
    const int kqs = VDR_Q4_0_Q8_1_MMVQ * (tid & 1);

    for (int kbx = tid >> 1; kbx < blocks_per_row; kbx += blocks_per_iter) {
        const unsigned char* bq4 = w_row + kbx * q4_block_bytes;
        const unsigned char* bq8 = act_tok + kbx * q8_block_bytes;

        // Load Q4_0 scale (half) and the 2 int32 quants this thread owns.
        // bq4 layout: [half d | uint8 qs[16]] — d is 2-byte aligned.
        unsigned short d_bits = (unsigned short)bq4[0] | ((unsigned short)bq4[1] << 8);
        float d4 = __half2float(__ushort_as_half(d_bits));

        // v[0], v[1]: 2 int32s starting at qs[kqs*4].
        // Use byte loads to be robust to the 2-byte alignment of qs.
        int v0, v1;
        {
            const unsigned char* p = bq4 + 2 + kqs * 4;  // kqs * 4 ∈ {0, 8}
            v0 = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
            v1 = (int)p[4] | ((int)p[5] << 8) | ((int)p[6] << 16) | ((int)p[7] << 24);
        }

        // u[4]: 4 int32s from bq8->qs at offsets kqs, kqs+1, kqs+QI4_0, kqs+QI4_0+1.
        // bq8 layout: [half2 ds (4 B) | int8 qs[32] (32 B)] — qs is 4-byte aligned.
        const int* qs8 = (const int*)(bq8 + 4);
        int u00 = qs8[kqs + 0];            // lower 4 values batch
        int u01 = qs8[kqs + 1];
        int u10 = qs8[kqs + QI4_0 + 0];    // upper 4 values batch (offset +4 int32 = +16 bytes)
        int u11 = qs8[kqs + QI4_0 + 1];

        // dp4a over low/high nibbles.
        int sumi = 0;
        {
            int vi_lo = (v0 >> 0) & 0x0F0F0F0F;
            int vi_hi = (v0 >> 4) & 0x0F0F0F0F;
            sumi = __dp4a(vi_lo, u00, sumi);
            sumi = __dp4a(vi_hi, u10, sumi);
        }
        {
            int vi_lo = (v1 >> 0) & 0x0F0F0F0F;
            int vi_hi = (v1 >> 4) & 0x0F0F0F0F;
            sumi = __dp4a(vi_lo, u01, sumi);
            sumi = __dp4a(vi_hi, u11, sumi);
        }

        // Q8_1 scale + sum: half2 at bq8[0..4].
        const __half2 ds8 = *(const __half2*)bq8;
        float2 ds8f = __half22float2(ds8);
        // 8 * vdr / QI4_0 = 8*2/4 = 4 (adjusts the -8 bias for half-block coverage).
        tmp += d4 * (sumi * ds8f.x - 4.0f * ds8f.y);
    }

    // Cross-warp reduce: each of the 32 lanes carries partial sums across
    // MMVQ_NWARPS warps; lane 0 of warp 0 writes the final scalar.
    __shared__ float tmp_shared[MMVQ_NWARPS - 1][32];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    #pragma unroll
    for (int l = 0; l < MMVQ_NWARPS - 1; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, offset, 32);
    }
    if (threadIdx.x == 0) {
        output[(long long)tok * N + row] = tmp;
    }
}

// =================================================================
// Row×Token MMQ-style tiled Q4_0 × Q8_1 kernel.
//
// Each block processes a tile of (ROW_TILE output rows) × (M_TILE tokens),
// so both the weight-row load (shared across M_TILE tokens) and the
// Q8_1 activation load (shared across ROW_TILE rows) are amortised
// over register-held accumulators. The previous "1 row × M_TILE tokens"
// shape topped out at ~32 tok/s for Llama 3.1 8B prefill on Xavier;
// adding row tiling cuts activation BW ~ROW_TILE× and is the main
// lever that closes the gap toward llama.cpp's MMQ throughput.
//
// Grid: (ceil(N / ROW_TILE), ceil(M_total / M_TILE), 1)
// Block: (32, MMVQ_NWARPS, 1) = 128 threads
// Per thread: ROW_TILE * M_TILE f32 accumulators + 4-way dp4a scratch
// =================================================================
// Empirically swept on Jetson AGX Xavier (sm_72, Llama 3.1 8B Q4_0, pp81):
//   (R=1,M=8)=32.4   (R=2,M=2)=25.6   (R=2,M=4)=37.2*   (R=2,M=8)=27.7
//   (R=4,M=2)=spill  (R=4,M=8)=spill  (R=1,M=16)=spill
// R=2,M=4 keeps 8 accumulators/thread (within Xavier's 64-reg/thread budget
// for 128-thread blocks) while amortising weight BW ~4× and activation BW
// ~2×. Going further (R=4 or M=8) spills registers → local memory traffic.
#define ROW_TILE 2
#define M_TILE   4

extern "C" __global__ __launch_bounds__(MMVQ_NWARPS * 32, 1)
void mul_mat_q4_0_q8_1_mtile(
    const unsigned char* __restrict__ weight,        // Q4_0 weight [N * (K/32)*18]
    const unsigned char* __restrict__ activation_q,  // Q8_1 [M_total * (K/32)*36]
    float*               __restrict__ output,        // F32 [M_total * N]
    int K, int N, int M_total)
{
    const int row_base = blockIdx.x * ROW_TILE;
    const int tok_base = blockIdx.y * M_TILE;

    const int tid = 32 * threadIdx.y + threadIdx.x;
    const int blocks_per_row = K >> 5;
    const int blocks_per_iter = (VDR_Q4_0_Q8_1_MMVQ * MMVQ_NWARPS * 32) / QI4_0;

    const int q4_block_bytes = 18;
    const int q8_block_bytes = 36;

    const int kqs = VDR_Q4_0_Q8_1_MMVQ * (tid & 1);

    // ROW_TILE * M_TILE accumulators per thread (register-held).
    float tmp[ROW_TILE][M_TILE];
    #pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) tmp[r][m] = 0.0f;
    }

    for (int kbx = tid >> 1; kbx < blocks_per_row; kbx += blocks_per_iter) {
        // Load ROW_TILE Q4_0 blocks (one per row) once, split into dp4a ops.
        float d4[ROW_TILE];
        int   vlo0[ROW_TILE], vhi0[ROW_TILE], vlo1[ROW_TILE], vhi1[ROW_TILE];
        #pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) {
            const int row = row_base + r;
            if (row >= N) { d4[r] = 0.0f; vlo0[r] = vhi0[r] = vlo1[r] = vhi1[r] = 0; continue; }
            const unsigned char* w_row = weight
                + (long long)row * blocks_per_row * q4_block_bytes;
            const unsigned char* bq4 = w_row + kbx * q4_block_bytes;
            // Block-start is 2-byte aligned (18 B records); keep byte loads
            // for qs and only cache-hint the half2 ds pair (2-byte safe).
            unsigned short d_bits =
                (unsigned short)bq4[0] | ((unsigned short)bq4[1] << 8);
            d4[r] = __half2float(__ushort_as_half(d_bits));
            const unsigned char* p = bq4 + 2 + kqs * 4;
            int v0 = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
            int v1 = (int)p[4] | ((int)p[5] << 8) | ((int)p[6] << 16) | ((int)p[7] << 24);
            vlo0[r] = (v0 >> 0) & 0x0F0F0F0F;
            vhi0[r] = (v0 >> 4) & 0x0F0F0F0F;
            vlo1[r] = (v1 >> 0) & 0x0F0F0F0F;
            vhi1[r] = (v1 >> 4) & 0x0F0F0F0F;
        }

        // Per M token: load Q8_1 once, issue dp4a against every row.
        // __ldg() routes through Xavier's read-only (texture) cache; for
        // the M_TILE=4 reuse pattern this raises hit rate vs the default
        // L1 path (~60% → ~85% in practice for strided Q8_1 access).
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            const int tok = tok_base + m;
            if (tok >= M_total) continue;
            const unsigned char* bq8 = activation_q
                + (long long)tok * blocks_per_row * q8_block_bytes
                + kbx * q8_block_bytes;
            const int* qs8 = (const int*)(bq8 + 4);
            int u00 = __ldg(&qs8[kqs + 0]);
            int u01 = __ldg(&qs8[kqs + 1]);
            int u10 = __ldg(&qs8[kqs + QI4_0 + 0]);
            int u11 = __ldg(&qs8[kqs + QI4_0 + 1]);
            const __half2 ds8 = __ldg((const __half2*)bq8);
            float2 ds8f = __half22float2(ds8);

            #pragma unroll
            for (int r = 0; r < ROW_TILE; ++r) {
                int sumi = 0;
                sumi = __dp4a(vlo0[r], u00, sumi);
                sumi = __dp4a(vhi0[r], u10, sumi);
                sumi = __dp4a(vlo1[r], u01, sumi);
                sumi = __dp4a(vhi1[r], u11, sumi);
                tmp[r][m] += d4[r] * (sumi * ds8f.x - 4.0f * ds8f.y);
            }
        }
    }

    // Cross-warp reduction per (row, tok).
    __shared__ float tmp_shared[ROW_TILE][M_TILE][MMVQ_NWARPS - 1][32];
    #pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (threadIdx.y > 0) {
                tmp_shared[r][m][threadIdx.y - 1][threadIdx.x] = tmp[r][m];
            }
        }
    }
    __syncthreads();

    if (threadIdx.y != 0) return;
    #pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
        const int row = row_base + r;
        if (row >= N) continue;
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            const int tok = tok_base + m;
            if (tok >= M_total) continue;
            float accum = tmp[r][m];
            #pragma unroll
            for (int l = 0; l < MMVQ_NWARPS - 1; ++l) {
                accum += tmp_shared[r][m][l][threadIdx.x];
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                accum += __shfl_down_sync(0xffffffff, accum, offset, 32);
            }
            if (threadIdx.x == 0) {
                output[(long long)tok * N + row] = accum;
            }
        }
    }
}

// =================================================================
// Fused FFN gate + up mmvq + SiLU gating (decode, M=1).
//
// Replaces the sequence
//     gate = w_gate · x    (mmvq)
//     up   = w_up   · x    (mmvq)
//     out  = silu(gate) * up
// with a single kernel. Both gate and up matmul scan the same
// Q8_1-quantised activation, so the read of `activation_q` is shared
// between the two dp4a accumulators (L2 reuse). The silu-multiply
// runs in registers inside the reduction write-back.
//
// Register budget: the non-fused kernel carries one float accumulator
// + a handful of int loads per dp4a; doubling for gate+up keeps us
// well under Xavier's 64-regs-per-thread sweet spot (checked via
// `--ptxas-options=-v` — N regs, no spill).
//
// Grid: (N, 1, 1) where N = FFN hidden dim (14336 for Llama 3.1 8B).
// Block: (32, MMVQ_NWARPS, 1) = 128 threads.
// =================================================================
extern "C" __global__ __launch_bounds__(MMVQ_NWARPS * 32, 1)
void mul_mat_vec_q4_0_q8_1_gate_up_silu(
    const unsigned char* __restrict__ weight_gate,   // Q4_0 blocks
    const unsigned char* __restrict__ weight_up,     // Q4_0 blocks
    const unsigned char* __restrict__ activation_q,  // Q8_1 blocks (shared)
    float*               __restrict__ output,        // [N] — silu(gate)·up
    int K, int N)
{
    const int row = blockIdx.x;
    if (row >= N) return;

    const int tid = 32 * threadIdx.y + threadIdx.x;
    const int blocks_per_row = K >> 5;
    const int blocks_per_iter = (VDR_Q4_0_Q8_1_MMVQ * MMVQ_NWARPS * 32) / QI4_0; // 64

    const int q4_block_bytes = 18;
    const int q8_block_bytes = 36;

    const unsigned char* w_gate_row =
        weight_gate + (long long)row * blocks_per_row * q4_block_bytes;
    const unsigned char* w_up_row =
        weight_up + (long long)row * blocks_per_row * q4_block_bytes;

    float tmp_gate = 0.0f;
    float tmp_up   = 0.0f;

    const int kqs = VDR_Q4_0_Q8_1_MMVQ * (tid & 1);

    for (int kbx = tid >> 1; kbx < blocks_per_row; kbx += blocks_per_iter) {
        const unsigned char* bq4_gate = w_gate_row + kbx * q4_block_bytes;
        const unsigned char* bq4_up   = w_up_row   + kbx * q4_block_bytes;
        const unsigned char* bq8      = activation_q + kbx * q8_block_bytes;

        // Q8_1 reads shared between gate and up.
        const int* qs8 = (const int*)(bq8 + 4);
        const int u00 = qs8[kqs + 0];
        const int u01 = qs8[kqs + 1];
        const int u10 = qs8[kqs + QI4_0 + 0];
        const int u11 = qs8[kqs + QI4_0 + 1];
        const __half2 ds8 = *(const __half2*)bq8;
        const float2 ds8f = __half22float2(ds8);

        // Gate block.
        {
            unsigned short d_bits =
                (unsigned short)bq4_gate[0] | ((unsigned short)bq4_gate[1] << 8);
            float d4 = __half2float(__ushort_as_half(d_bits));
            int v0, v1;
            {
                const unsigned char* p = bq4_gate + 2 + kqs * 4;
                v0 = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
                v1 = (int)p[4] | ((int)p[5] << 8) | ((int)p[6] << 16) | ((int)p[7] << 24);
            }
            int sumi = 0;
            {
                int vi_lo = (v0 >> 0) & 0x0F0F0F0F;
                int vi_hi = (v0 >> 4) & 0x0F0F0F0F;
                sumi = __dp4a(vi_lo, u00, sumi);
                sumi = __dp4a(vi_hi, u10, sumi);
            }
            {
                int vi_lo = (v1 >> 0) & 0x0F0F0F0F;
                int vi_hi = (v1 >> 4) & 0x0F0F0F0F;
                sumi = __dp4a(vi_lo, u01, sumi);
                sumi = __dp4a(vi_hi, u11, sumi);
            }
            tmp_gate += d4 * (sumi * ds8f.x - 4.0f * ds8f.y);
        }

        // Up block.
        {
            unsigned short d_bits =
                (unsigned short)bq4_up[0] | ((unsigned short)bq4_up[1] << 8);
            float d4 = __half2float(__ushort_as_half(d_bits));
            int v0, v1;
            {
                const unsigned char* p = bq4_up + 2 + kqs * 4;
                v0 = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
                v1 = (int)p[4] | ((int)p[5] << 8) | ((int)p[6] << 16) | ((int)p[7] << 24);
            }
            int sumi = 0;
            {
                int vi_lo = (v0 >> 0) & 0x0F0F0F0F;
                int vi_hi = (v0 >> 4) & 0x0F0F0F0F;
                sumi = __dp4a(vi_lo, u00, sumi);
                sumi = __dp4a(vi_hi, u10, sumi);
            }
            {
                int vi_lo = (v1 >> 0) & 0x0F0F0F0F;
                int vi_hi = (v1 >> 4) & 0x0F0F0F0F;
                sumi = __dp4a(vi_lo, u01, sumi);
                sumi = __dp4a(vi_hi, u11, sumi);
            }
            tmp_up += d4 * (sumi * ds8f.x - 4.0f * ds8f.y);
        }
    }

    // Cross-warp reduce for both accumulators.
    __shared__ float tmp_shared_gate[MMVQ_NWARPS - 1][32];
    __shared__ float tmp_shared_up  [MMVQ_NWARPS - 1][32];
    if (threadIdx.y > 0) {
        tmp_shared_gate[threadIdx.y - 1][threadIdx.x] = tmp_gate;
        tmp_shared_up  [threadIdx.y - 1][threadIdx.x] = tmp_up;
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    #pragma unroll
    for (int l = 0; l < MMVQ_NWARPS - 1; ++l) {
        tmp_gate += tmp_shared_gate[l][threadIdx.x];
        tmp_up   += tmp_shared_up  [l][threadIdx.x];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        tmp_gate += __shfl_down_sync(0xffffffff, tmp_gate, offset, 32);
        tmp_up   += __shfl_down_sync(0xffffffff, tmp_up,   offset, 32);
    }
    if (threadIdx.x == 0) {
        // SiLU(gate) = gate / (1 + exp(-gate)).
        float g = tmp_gate;
        float silu = g / (1.0f + __expf(-g));
        output[row] = silu * tmp_up;
    }
}
