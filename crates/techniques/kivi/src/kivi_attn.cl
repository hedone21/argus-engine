// KIVI native fused attention kernels for quantized KV cache.
//
// Eliminates the intermediate F32 dequant buffer by doing on-the-fly dequant
// inside the attention kernel. Supports Q2, Q4, and Q8 KV cache formats.
//
// Work group: [LOCAL_SIZE, 1, 1], one work group per Q head.
// Each thread processes a subset of tokens (strided), accumulates dot products
// and weighted V sums using private registers.
//
// KV layout (from KiviCache flush_residual — flush-interleaved):
//   Key (per-channel): blocks are pushed per-flush:
//     for each flush, for each head, for each group_in_flush, for each channel
//     block_idx = flush * kv_heads * gpf * head_dim + kv_head * gpf * head_dim + gif * head_dim + ch
//     where gpf = groups_per_flush = res_cap / GROUP_SIZE, gif = group % gpf
//   Value (per-token): blocks are pushed per-flush:
//     for each flush, for each head, for each token_in_flush, for each block
//     block_idx = flush * kv_heads * res_cap * bpt + kv_head * res_cap * bpt + tif * bpt + b
//     where tif = t % res_cap

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LOCAL_SIZE 64
#define GROUP_SIZE 32

// ============================================================
// Q2 block: 12 bytes (d:f16 + m:f16 + qs:u8[8])
// 32 values, 4 per byte, 2 bits each
// ============================================================
#define Q2_BLOCK_BYTES 12

// Dequant single element from Q2 block at byte offset `src_off` in data buffer
// `within` is the element index 0..31 within the block
inline float deq_q2(global const uchar* data, int src_off, int within) {
    float d = vload_half(0, (global const half*)(data + src_off));
    float m = vload_half(0, (global const half*)(data + src_off + 2));
    uchar byte = data[src_off + 4 + within / 4];
    return (float)((byte >> ((within % 4) * 2)) & 0x03) * d + m;
}

kernel void kernel_attn_gen_kivi_q2(
    global float * Q,              // [num_heads_q, head_dim]
    global const uchar * q2_k,     // Q2 key blocks (per-channel)
    global const uchar * q2_v,     // Q2 value blocks (per-token)
    global const float * res_k,    // F32 residual keys [kv_heads, res_cap, head_dim]
    global const float * res_v,    // F32 residual values [kv_heads, res_cap, head_dim]
    global float * O,              // output [num_heads_q, head_dim]
    global float * S,              // scores_out (may be dummy)
    int num_heads_q,
    int num_heads_kv,
    int head_dim,
    int q2_tokens,                 // quantized token count
    int res_tokens,                // residual token count
    int res_cap,                   // residual buffer capacity
    float scale,                   // 1/sqrt(head_dim)
    int score_stride,
    int has_scores,                // 1=write scores, 0=skip
    local float * scratch          // size = LOCAL_SIZE
) {
    int head_idx = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;

    global float * q_ptr = Q + head_idx * head_dim;
    int total_tokens = q2_tokens + res_tokens;

    // Flush-interleaved layout constants.
    // CPU pushes blocks in per-flush order: for each flush, for each head, ...
    // groups_per_flush = res_cap / GROUP_SIZE (tokens flushed per flush)
    int groups_per_flush = res_cap / GROUP_SIZE;
    int blocks_per_tok_v = head_dim / GROUP_SIZE;

    // === PASS 1: max score ===
    float my_max = -INFINITY;
    for (int t = lid; t < total_tokens; t += local_size) {
        float score = 0.0f;
        if (t < q2_tokens) {
            // On-the-fly Q2 key dequant (per-channel, flush-interleaved layout)
            int group = t / GROUP_SIZE;
            int within = t % GROUP_SIZE;
            int flush = group / groups_per_flush;
            int gif = group % groups_per_flush;  // group-in-flush
            int block_idx = flush * num_heads_kv * groups_per_flush * head_dim
                          + kv_head * groups_per_flush * head_dim
                          + gif * head_dim;
            for (int ch = 0; ch < head_dim; ch++) {
                int src_off = (block_idx + ch) * Q2_BLOCK_BYTES;
                float kval = deq_q2(q2_k, src_off, within);
                score += q_ptr[ch] * kval;
            }
        } else {
            // F32 residual key
            int rt = t - q2_tokens;
            global const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        score *= scale;
        my_max = fmax(my_max, score);
    }

    // Reduce max across workgroup
    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute exp sum
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
            global const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        my_sum += exp(score * scale - max_score);
    }

    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // === PASS 2: weighted V sum ===
    float out_local[256];  // max head_dim
    for (int d = 0; d < head_dim; d++) {
        out_local[d] = 0.0f;
    }

    for (int t = lid; t < total_tokens; t += local_size) {
        // Recompute K score
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
            global const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        float weight = exp(score * scale - max_score) / total_sum;

        if (has_scores) {
            S[head_idx * score_stride + t] = weight;
        }

        // Accumulate V contribution (per-token, flush-interleaved layout)
        if (t < q2_tokens) {
            int v_flush = t / res_cap;
            int tif = t % res_cap;  // token-in-flush
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
            global const float * v_ptr = res_v + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_local[d] += weight * v_ptr[d];
            }
        }
    }

    // Reduce output across threads
    for (int d = 0; d < head_dim; d++) {
        scratch[lid] = out_local[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            O[head_idx * head_dim + d] = scratch[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


// ============================================================
// Q4 block: 20 bytes (d:f16 + m:f16 + qs:u8[16])
// 32 values, 2 per byte, 4 bits each
// ============================================================
#define Q4_BLOCK_BYTES 20

inline float deq_q4(global const uchar* data, int src_off, int within) {
    float d = vload_half(0, (global const half*)(data + src_off));
    float m = vload_half(0, (global const half*)(data + src_off + 2));
    uchar byte = data[src_off + 4 + within / 2];
    return (float)((byte >> ((within % 2) * 4)) & 0x0F) * d + m;
}

kernel void kernel_attn_gen_kivi_q4(
    global float * Q,
    global const uchar * q4_k,
    global const uchar * q4_v,
    global const float * res_k,
    global const float * res_v,
    global float * O,
    global float * S,
    int num_heads_q,
    int num_heads_kv,
    int head_dim,
    int q_tokens,
    int res_tokens,
    int res_cap,
    float scale,
    int score_stride,
    int has_scores,
    local float * scratch
) {
    int head_idx = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;

    global float * q_ptr = Q + head_idx * head_dim;
    int total_tokens = q_tokens + res_tokens;

    int groups_per_flush = res_cap / GROUP_SIZE;
    int blocks_per_tok_v = head_dim / GROUP_SIZE;

    // === PASS 1: max score ===
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
            global const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        score *= scale;
        my_max = fmax(my_max, score);
    }

    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

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
            global const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        my_sum += exp(score * scale - max_score);
    }

    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // === PASS 2: weighted V sum ===
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
            global const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        float weight = exp(score * scale - max_score) / total_sum;

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
            global const float * v_ptr = res_v + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_local[d] += weight * v_ptr[d];
            }
        }
    }

    for (int d = 0; d < head_dim; d++) {
        scratch[lid] = out_local[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            O[head_idx * head_dim + d] = scratch[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


// ============================================================
// Q8 block: 36 bytes (d:f16 + m:f16 + qs:u8[32])
// 32 values, 1 per byte, 8 bits each
// ============================================================
#define Q8_BLOCK_BYTES 36

inline float deq_q8(global const uchar* data, int src_off, int within) {
    float d = vload_half(0, (global const half*)(data + src_off));
    float m = vload_half(0, (global const half*)(data + src_off + 2));
    return (float)(data[src_off + 4 + within]) * d + m;
}

kernel void kernel_attn_gen_kivi_q8(
    global float * Q,
    global const uchar * q8_k,
    global const uchar * q8_v,
    global const float * res_k,
    global const float * res_v,
    global float * O,
    global float * S,
    int num_heads_q,
    int num_heads_kv,
    int head_dim,
    int q_tokens,
    int res_tokens,
    int res_cap,
    float scale,
    int score_stride,
    int has_scores,
    local float * scratch
) {
    int head_idx = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;

    global float * q_ptr = Q + head_idx * head_dim;
    int total_tokens = q_tokens + res_tokens;

    int groups_per_flush = res_cap / GROUP_SIZE;
    int blocks_per_tok_v = head_dim / GROUP_SIZE;

    // === PASS 1: max score ===
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
            global const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        score *= scale;
        my_max = fmax(my_max, score);
    }

    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

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
            global const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        my_sum += exp(score * scale - max_score);
    }

    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // === PASS 2: weighted V sum ===
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
            global const float * k_ptr = res_k + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
        }
        float weight = exp(score * scale - max_score) / total_sum;

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
            global const float * v_ptr = res_v + kv_head * res_cap * head_dim + rt * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_local[d] += weight * v_ptr[d];
            }
        }
    }

    for (int d = 0; d < head_dim; d++) {
        scratch[lid] = out_local[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            O[head_idx * head_dim + d] = scratch[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
