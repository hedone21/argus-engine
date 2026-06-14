#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Score-only attention kernel for decode (seq_len=1).
// Computes Q*K^T dot product + softmax, WITHOUT the V*score weighted sum.
// Used alongside flash_attention_decode_gpu() to decouple output computation
// from score extraction: flash attn produces the output vector, this kernel
// produces post-softmax attention scores for eviction policies (H2O/D2O).
//
// Layout: HeadMajor KV cache [kv_heads, capacity, head_dim], K is F16.
// One workgroup per query head. SLM tree-reduce for max and sum.
//
// Args:
//   Q:              [num_heads_q, head_dim] F32
//   K:              HeadMajor [kv_heads, capacity, head_dim] F16
//   S:              [num_heads_q, score_stride] F32 output (post-softmax)
//   head_dim:       dimension per head
//   num_heads_q:    number of query heads
//   num_heads_kv:   number of KV heads (GQA)
//   cache_seq_len:  number of valid positions in cache
//   scale:          1/sqrt(head_dim)
//   kv_pos_stride:  stride between positions (head_dim for HeadMajor)
//   kv_head_stride: stride between heads (capacity*head_dim for HeadMajor)
//   score_stride:   stride between heads in S
//   scratch:        local memory [local_size] for reductions
__kernel void kernel_score_only_half(
    __global const float * Q,
    __global const half  * K,
    __global float       * S,
    int head_dim,
    int num_heads_q,
    int num_heads_kv,
    int cache_seq_len,
    float scale,
    int kv_pos_stride,
    int kv_head_stride,
    int score_stride,
    __local float * scratch
) {
    int head_idx = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;
    int kv_base = kv_head * kv_head_stride;

    __global const float * q_ptr = Q + head_idx * head_dim;

    // === PASS 1: Q*K^T dot products + find max score ===
    float my_max = -INFINITY;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        __global const half * k_ptr = K + kv_base + t * kv_pos_stride;
        float dot = 0.0f;
        // Vectorized F16 load (4 elements at a time)
        int d = 0;
        for (; d + 3 < head_dim; d += 4) {
            half4 kv = vload_half4(0, k_ptr + d);
            float4 kf = convert_float4(kv);
            float4 qf = vload4(0, q_ptr + d);
            dot += qf.x * kf.x + qf.y * kf.y + qf.z * kf.z + qf.w * kf.w;
        }
        for (; d < head_dim; d++) {
            dot += q_ptr[d] * vload_half(d, k_ptr);
        }
        dot *= scale;
        my_max = fmax(my_max, dot);
    }

    // SLM tree-reduce for max
    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // === PASS 2: exp(score - max) sum ===
    float my_sum = 0.0f;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        __global const half * k_ptr = K + kv_base + t * kv_pos_stride;
        float dot = 0.0f;
        int d = 0;
        for (; d + 3 < head_dim; d += 4) {
            half4 kv = vload_half4(0, k_ptr + d);
            float4 kf = convert_float4(kv);
            float4 qf = vload4(0, q_ptr + d);
            dot += qf.x * kf.x + qf.y * kf.y + qf.z * kf.z + qf.w * kf.w;
        }
        for (; d < head_dim; d++) {
            dot += q_ptr[d] * vload_half(d, k_ptr);
        }
        my_sum += exp(dot * scale - max_score);
    }

    // SLM tree-reduce for sum
    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // === PASS 3: Write post-softmax scores ===
    float inv_sum = 1.0f / total_sum;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        __global const half * k_ptr = K + kv_base + t * kv_pos_stride;
        float dot = 0.0f;
        int d = 0;
        for (; d + 3 < head_dim; d += 4) {
            half4 kv = vload_half4(0, k_ptr + d);
            float4 kf = convert_float4(kv);
            float4 qf = vload4(0, q_ptr + d);
            dot += qf.x * kf.x + qf.y * kf.y + qf.z * kf.z + qf.w * kf.w;
        }
        for (; d < head_dim; d++) {
            dot += q_ptr[d] * vload_half(d, k_ptr);
        }
        S[head_idx * score_stride + t] = exp(dot * scale - max_score) * inv_sum;
    }
}
