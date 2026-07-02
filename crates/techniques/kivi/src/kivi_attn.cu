// CUDA port of kivi_attn.cl — KIVI native fused attention kernels (Q2/Q4/Q8 quantized KV).
//
// One block per Q head (blockDim.x == LOCAL_SIZE == 64). Each thread strides over tokens,
// dequantizing KV on the fly (no intermediate F32 buffer) and accumulating dot products +
// weighted V sums in private registers, with block-wide tree reductions for max / sum / output.
//
// Translation from OpenCL (byte-faithful math):
//   - `local float* scratch` (trailing kernel arg) → `extern __shared__ float scratch[]`
//     ⇒ the kernel arity drops from 17 to 16; the host passes sharedMemBytes = 64*4 = 256.
//   - every `barrier(CLK_LOCAL_MEM_FENCE)` → `__syncthreads()` (ALL of them, incl. the one at
//     the end of each d-iteration in the output reduction).
//   - `get_group_id(0)`→blockIdx.x, `get_local_id(0)`→threadIdx.x, `get_local_size(0)`→blockDim.x.
//   - `vload_half(0,(half*)p)` → `__half2float(*reinterpret_cast<const __half*>(p))`.
//   - `fmax`→fmaxf, `exp`→expf; dequant op order `code*d+m` preserved exactly; scale asymmetry
//     preserved (pass1 `score*=scale` before fmaxf; pass2/2b `exp(score*scale - max_score)`).
//   - blockDim.x MUST be a power of two (64) for the `for(s=size/2; s>0; s>>=1)` reductions.

#include <cuda_fp16.h>

#define GROUP_SIZE 32

// ============================================================
// Q2 block: 12 bytes (d:f16 + m:f16 + qs:u8[8]); 32 values, 4 per byte, 2 bits each
// ============================================================
#define Q2_BLOCK_BYTES 12

__device__ inline float deq_q2(const unsigned char* data, int src_off, int within) {
    float d = __half2float(*reinterpret_cast<const __half*>(data + src_off));
    float m = __half2float(*reinterpret_cast<const __half*>(data + src_off + 2));
    unsigned char byte = data[src_off + 4 + within / 4];
    return (float)((byte >> ((within % 4) * 2)) & 0x03) * d + m;
}

extern "C" __global__ void kernel_attn_gen_kivi_q2(
    float * __restrict__ Q,
    const unsigned char * __restrict__ q2_k,
    const unsigned char * __restrict__ q2_v,
    const float * __restrict__ res_k,
    const float * __restrict__ res_v,
    float * __restrict__ O,
    float * __restrict__ S,
    int num_heads_q,
    int num_heads_kv,
    int head_dim,
    int q2_tokens,
    int res_tokens,
    int res_cap,
    float scale,
    int score_stride,
    int has_scores)
{
    extern __shared__ float scratch[];
    int head_idx = blockIdx.x;
    int lid = threadIdx.x;
    int local_size = blockDim.x;

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;

    float * q_ptr = Q + head_idx * head_dim;
    int total_tokens = q2_tokens + res_tokens;

    int groups_per_flush = res_cap / GROUP_SIZE;
    int blocks_per_tok_v = head_dim / GROUP_SIZE;

    // === PASS 1: max score ===
    float my_max = -INFINITY;
    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q2_tokens) {
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q2_BLOCK_BYTES;
                float kval = deq_q2(q2_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            int rt = t - q2_tokens;
            const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        score *= scale;
        my_max = fmaxf(my_max, score);
    }

    scratch[lid] = my_max;
    __syncthreads();
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmaxf(scratch[lid], scratch[lid + s]);
        __syncthreads();
    }
    float max_score = scratch[0];
    __syncthreads();

    float my_sum = 0.0f;
    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q2_tokens) {
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q2_BLOCK_BYTES;
                float kval = deq_q2(q2_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            int rt = t - q2_tokens;
            const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        my_sum += expf(score * scale - max_score);
    }

    scratch[lid] = my_sum;
    __syncthreads();
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        __syncthreads();
    }
    float total_sum = scratch[0];
    __syncthreads();

    // === PASS 2: weighted V sum ===
    float out_local[256];
    for (int d = 0; d < head_dim; d++) {
        out_local[d] = 0.0f;
    }

    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q2_tokens) {
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q2_BLOCK_BYTES;
                float kval = deq_q2(q2_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            int rt = t - q2_tokens;
            const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        float weight = expf(score * scale - max_score) / total_sum;

        if (has_scores) {
            S[head_idx * score_stride + t] = weight;
        }

        if (t < q2_tokens) {
            int v_flush = t / res_cap;
            int tif = t % res_cap;
            int v_base = v_flush * num_heads_kv * res_cap * blocks_per_tok_v
                       + kv_head * res_cap * blocks_per_tok_v
                       + tif * blocks_per_tok_v;
            for (int d = 0; d < head_dim; d++) {
                int b = d / GROUP_SIZE;
                int within = d % GROUP_SIZE;
                int src_off = (v_base + b) * Q2_BLOCK_BYTES;
                float vval = deq_q2(q2_v, src_off, within);
                out_local[d] += weight * vval;
            }
        } else {
            int rt = t - q2_tokens;
            const float * v_ptr = res_v + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_local[d] += weight * v_ptr[d];
            }
        }
    }

    for (int d = 0; d < head_dim; d++) {
        scratch[lid] = out_local[d];
        __syncthreads();
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            __syncthreads();
        }
        if (lid == 0) {
            O[head_idx * head_dim + d] = scratch[0];
        }
        __syncthreads();
    }
}

// ============================================================
// Q4 block: 20 bytes (d:f16 + m:f16 + qs:u8[16]); 32 values, 2 per byte, 4 bits each
// ============================================================
#define Q4_BLOCK_BYTES 20

__device__ inline float deq_q4(const unsigned char* data, int src_off, int within) {
    float d = __half2float(*reinterpret_cast<const __half*>(data + src_off));
    float m = __half2float(*reinterpret_cast<const __half*>(data + src_off + 2));
    unsigned char byte = data[src_off + 4 + within / 2];
    return (float)((byte >> ((within % 2) * 4)) & 0x0F) * d + m;
}

extern "C" __global__ void kernel_attn_gen_kivi_q4(
    float * __restrict__ Q,
    const unsigned char * __restrict__ q4_k,
    const unsigned char * __restrict__ q4_v,
    const float * __restrict__ res_k,
    const float * __restrict__ res_v,
    float * __restrict__ O,
    float * __restrict__ S,
    int num_heads_q,
    int num_heads_kv,
    int head_dim,
    int q_tokens,
    int res_tokens,
    int res_cap,
    float scale,
    int score_stride,
    int has_scores)
{
    extern __shared__ float scratch[];
    int head_idx = blockIdx.x;
    int lid = threadIdx.x;
    int local_size = blockDim.x;

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;

    float * q_ptr = Q + head_idx * head_dim;
    int total_tokens = q_tokens + res_tokens;

    int groups_per_flush = res_cap / GROUP_SIZE;
    int blocks_per_tok_v = head_dim / GROUP_SIZE;

    float my_max = -INFINITY;
    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q_tokens) {
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q4_BLOCK_BYTES;
                float kval = deq_q4(q4_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            int rt = t - q_tokens;
            const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        score *= scale;
        my_max = fmaxf(my_max, score);
    }

    scratch[lid] = my_max;
    __syncthreads();
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmaxf(scratch[lid], scratch[lid + s]);
        __syncthreads();
    }
    float max_score = scratch[0];
    __syncthreads();

    float my_sum = 0.0f;
    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q_tokens) {
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q4_BLOCK_BYTES;
                float kval = deq_q4(q4_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            int rt = t - q_tokens;
            const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        my_sum += expf(score * scale - max_score);
    }

    scratch[lid] = my_sum;
    __syncthreads();
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        __syncthreads();
    }
    float total_sum = scratch[0];
    __syncthreads();

    float out_local[256];
    for (int d = 0; d < head_dim; d++) {
        out_local[d] = 0.0f;
    }

    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q_tokens) {
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q4_BLOCK_BYTES;
                float kval = deq_q4(q4_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            int rt = t - q_tokens;
            const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        float weight = expf(score * scale - max_score) / total_sum;

        if (has_scores) {
            S[head_idx * score_stride + t] = weight;
        }

        if (t < q_tokens) {
            int v_flush = t / res_cap;
            int tif = t % res_cap;
            int v_base = v_flush * num_heads_kv * res_cap * blocks_per_tok_v
                       + kv_head * res_cap * blocks_per_tok_v
                       + tif * blocks_per_tok_v;
            for (int d = 0; d < head_dim; d++) {
                int b = d / GROUP_SIZE;
                int within = d % GROUP_SIZE;
                int src_off = (v_base + b) * Q4_BLOCK_BYTES;
                float vval = deq_q4(q4_v, src_off, within);
                out_local[d] += weight * vval;
            }
        } else {
            int rt = t - q_tokens;
            const float * v_ptr = res_v + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_local[d] += weight * v_ptr[d];
            }
        }
    }

    for (int d = 0; d < head_dim; d++) {
        scratch[lid] = out_local[d];
        __syncthreads();
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            __syncthreads();
        }
        if (lid == 0) {
            O[head_idx * head_dim + d] = scratch[0];
        }
        __syncthreads();
    }
}

// ============================================================
// Q8 block: 36 bytes (d:f16 + m:f16 + qs:u8[32]); 32 values, 1 per byte, 8 bits each
// ============================================================
#define Q8_BLOCK_BYTES 36

__device__ inline float deq_q8(const unsigned char* data, int src_off, int within) {
    float d = __half2float(*reinterpret_cast<const __half*>(data + src_off));
    float m = __half2float(*reinterpret_cast<const __half*>(data + src_off + 2));
    return (float)(data[src_off + 4 + within]) * d + m;
}

extern "C" __global__ void kernel_attn_gen_kivi_q8(
    float * __restrict__ Q,
    const unsigned char * __restrict__ q8_k,
    const unsigned char * __restrict__ q8_v,
    const float * __restrict__ res_k,
    const float * __restrict__ res_v,
    float * __restrict__ O,
    float * __restrict__ S,
    int num_heads_q,
    int num_heads_kv,
    int head_dim,
    int q_tokens,
    int res_tokens,
    int res_cap,
    float scale,
    int score_stride,
    int has_scores)
{
    extern __shared__ float scratch[];
    int head_idx = blockIdx.x;
    int lid = threadIdx.x;
    int local_size = blockDim.x;

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;

    float * q_ptr = Q + head_idx * head_dim;
    int total_tokens = q_tokens + res_tokens;

    int groups_per_flush = res_cap / GROUP_SIZE;
    int blocks_per_tok_v = head_dim / GROUP_SIZE;

    float my_max = -INFINITY;
    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q_tokens) {
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q8_BLOCK_BYTES;
                float kval = deq_q8(q8_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            int rt = t - q_tokens;
            const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        score *= scale;
        my_max = fmaxf(my_max, score);
    }

    scratch[lid] = my_max;
    __syncthreads();
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmaxf(scratch[lid], scratch[lid + s]);
        __syncthreads();
    }
    float max_score = scratch[0];
    __syncthreads();

    float my_sum = 0.0f;
    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q_tokens) {
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q8_BLOCK_BYTES;
                float kval = deq_q8(q8_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            int rt = t - q_tokens;
            const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        my_sum += expf(score * scale - max_score);
    }

    scratch[lid] = my_sum;
    __syncthreads();
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        __syncthreads();
    }
    float total_sum = scratch[0];
    __syncthreads();

    float out_local[256];
    for (int d = 0; d < head_dim; d++) {
        out_local[d] = 0.0f;
    }

    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q_tokens) {
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q8_BLOCK_BYTES;
                float kval = deq_q8(q8_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            int rt = t - q_tokens;
            const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        float weight = expf(score * scale - max_score) / total_sum;

        if (has_scores) {
            S[head_idx * score_stride + t] = weight;
        }

        if (t < q_tokens) {
            int v_flush = t / res_cap;
            int tif = t % res_cap;
            int v_base = v_flush * num_heads_kv * res_cap * blocks_per_tok_v
                       + kv_head * res_cap * blocks_per_tok_v
                       + tif * blocks_per_tok_v;
            for (int d = 0; d < head_dim; d++) {
                int b = d / GROUP_SIZE;
                int within = d % GROUP_SIZE;
                int src_off = (v_base + b) * Q8_BLOCK_BYTES;
                float vval = deq_q8(q8_v, src_off, within);
                out_local[d] += weight * vval;
            }
        } else {
            int rt = t - q_tokens;
            const float * v_ptr = res_v + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_local[d] += weight * v_ptr[d];
            }
        }
    }

    for (int d = 0; d < head_dim; d++) {
        scratch[lid] = out_local[d];
        __syncthreads();
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            __syncthreads();
        }
        if (lid == 0) {
            O[head_idx * head_dim + d] = scratch[0];
        }
        __syncthreads();
    }
}
