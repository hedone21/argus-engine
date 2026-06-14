#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Adreno-specific sub-group size hint (A-3 B-1).
// `qcom_reqd_sub_group_size("half")` pins the wavefront to 64 lanes on Adreno
// 6xx/7xx/8xx so we can split a 32-row Q tile across a 64-wide work-group
// without the driver silently falling back to 128-wide waves.
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_KV_ACC4(x) convert_float4(x)
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
// A-3 B-1: DK=128 prefill uses a 2-lanes-per-Q-row subgroup split layout.
// DK=64 (Llama 3.2 1B) keeps the original single-thread-per-Q-row layout
// to avoid any regression on that model.
#if DK == 128
#define FA_SUBGROUP_SPLIT 1
#define LANES_PER_QROW 2
#define DK_VEC_HALF (DK_VEC/2)
#define DV_VEC_HALF (DV_VEC/2)
#define WG_SIZE (LANES_PER_QROW * BLOCK_M)   // 64 threads when BLOCK_M=32
#else
#define FA_SUBGROUP_SPLIT 0
#define WG_SIZE (BLOCK_M)
#endif
#define Q1_WG_SIZE 64

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return pow(base, exph);
}
#if FA_SUBGROUP_SPLIT
// Prefill flash attention (A-3 B-1 reduction layout, DK=128 only).
//
// Lane layout (local_work_size = [2*BLOCK_M, 1, 1] = 64 when BLOCK_M=32):
//   q_row_in_wg = tid % BLOCK_M           // 0..BLOCK_M-1, the Q-row this lane serves
//   half        = tid / BLOCK_M           // 0 => lower DK/DV slice, 1 => upper slice
// Each Q-row is cooperatively handled by a pair of lanes (lane_low, lane_high)
// that own disjoint halves of DK and DV, cutting per-thread q_priv[]/o_acc[]
// footprint by 2x. QK partial sums from each half are combined via one SLM
// exchange per K-row pair. The online softmax state (m_i, l_i) is maintained
// redundantly on both lanes — deterministic because the combined score drives
// identical updates on both sides.
__kernel REQD_SUBGROUP_SIZE_64 void flash_attn_f32_f16(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int q_row_in_wg = tid & (BLOCK_M - 1);
    // NOTE: avoid the bare identifier `half` — it is reserved (fp16 type) by
    // the Adreno OpenCL compiler and rejected as a variable name.
    const int lane_half = tid / BLOCK_M;          // 0 or 1
    const int dk_off = lane_half * DK_VEC_HALF;   // lane's DK slice offset (in float4s)
    const int dv_off = lane_half * DV_VEC_HALF;   // lane's DV slice offset (in float4s)

    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + q_row_in_wg;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    // Per-lane Q slice: DK_VEC/2 float4s (lower or upper half of the DK axis).
    ACC_TYPE4 q_priv[DK_VEC_HALF];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC_HALF; ++i) {
            q_priv[i] = CONVERT_Q_ACC4(q_ptr[dk_off + i]);
        }
    }

    // Per-lane O slice accumulator (DV_VEC/2 float4s).
    ACC_TYPE4 o_acc[DV_VEC_HALF];
    #pragma unroll
    for (int i = 0; i < DV_VEC_HALF; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];
    // Per-Q-row pair-sum exchange buffer: [q_row][half] -> partial dot sum.
    // Each K-row pair writes 2 entries/lane, consumed via a single barrier.
    __local ACC_TYPE l_dot[BLOCK_M][2][2]; // [q_row][k_pair_idx 0|1][half 0|1]

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        // Cooperative K/V tile load. WG_SIZE=2*BLOCK_M lanes share the work.
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_base + k_row_offset))[col];
            } else {
                l_k[row][col] = (KV_DATA_TYPE4)(0);
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
            } else {
                l_v[row][col] = (KV_DATA_TYPE4)(0);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < BLOCK_N; j += 2) {
            // QK partial dot on this lane's half of DK.
            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
            ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
            if (my_query_row < n_q) {
                #pragma unroll
                for (int k = 0; k < DK_VEC_HALF; k++) {
                    dot_acc0 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j    ][dk_off + k]), dot_acc0);
                    dot_acc1 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j + 1][dk_off + k]), dot_acc1);
                }
            }
            ACC_TYPE partial0 = dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3;
            ACC_TYPE partial1 = dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3;

            // Publish this lane's partial sums.
            l_dot[q_row_in_wg][0][lane_half] = partial0;
            l_dot[q_row_in_wg][1][lane_half] = partial1;
            barrier(CLK_LOCAL_MEM_FENCE);

            // Each lane reads both halves to reconstruct the full dot product.
            ACC_TYPE score0 = (l_dot[q_row_in_wg][0][0] + l_dot[q_row_in_wg][0][1]) * scale;
            ACC_TYPE score1 = (l_dot[q_row_in_wg][1][0] + l_dot[q_row_in_wg][1][1]) * scale;
            // Barrier before next write to l_dot (next iter) — see end of loop.

            const int k_row0 = k_start + j;
            const int k_row1 = k_start + j + 1;

            if (is_causal) {
                if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
                if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
            }

            if (k_row0 >= n_kv) score0 = -INFINITY;
            if (k_row1 >= n_kv) score1 = -INFINITY;

            if (mask_base != NULL) {
                const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
            }

            if (logit_softcap > 0.0f) {
                score0 = logit_softcap * tanh(score0 / logit_softcap);
                score1 = logit_softcap * tanh(score1 / logit_softcap);
            }

            const ACC_TYPE m_new = max(m_i, max(score0, score1));
            const ACC_TYPE p0 = exp(score0 - m_new);
            const ACC_TYPE p1 = exp(score1 - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);

            // O update on this lane's half of DV.
            #pragma unroll
            for (int i = 0; i < DV_VEC_HALF; ++i) {
                o_acc[i] = o_acc[i] * scale_prev
                         + p0 * CONVERT_KV_ACC4(l_v[j    ][dv_off + i])
                         + p1 * CONVERT_KV_ACC4(l_v[j + 1][dv_off + i]);
            }
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;

            // Ensure all lanes have consumed l_dot before the next pair overwrites it.
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC_HALF; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC_HALF; ++i) {
                o_row[dv_off + i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC_HALF; ++i) {
                o_row[dv_off + i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
}

#else  // FA_SUBGROUP_SPLIT == 0 (DK=64 original layout, unchanged for Llama 3.2 1B)

__kernel void flash_attn_f32_f16(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
        }
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_base + k_row_offset))[col];
            } else {
                l_k[row][col] = (KV_DATA_TYPE4)(0);
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
            } else {
                l_v[row][col] = (KV_DATA_TYPE4)(0);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) {
            continue;
        }

        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j;
            const int k_row1 = k_start + j + 1;

            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
            ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_acc0 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j][k]), dot_acc0);
                dot_acc1 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j+1][k]), dot_acc1);
            }
            ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
            ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;

            if (is_causal) {
                if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
                if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
            }

            if (k_row0 >= n_kv) score0 = -INFINITY;
            if (k_row1 >= n_kv) score1 = -INFINITY;

            if (mask_base != NULL) {
                const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
            }

            if (logit_softcap > 0.0f) {
                score0 = logit_softcap * tanh(score0 / logit_softcap);
                score1 = logit_softcap * tanh(score1 / logit_softcap);
            }

            const ACC_TYPE m_new = max(m_i, max(score0, score1));
            const ACC_TYPE p0 = exp(score0 - m_new);
            const ACC_TYPE p1 = exp(score1 - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] = o_acc[i] * scale_prev + p0 * CONVERT_KV_ACC4(l_v[j][i]) + p1 * CONVERT_KV_ACC4(l_v[j+1][i]);
            }
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
}

#endif  // FA_SUBGROUP_SPLIT

// Decode Q1 kernel — SLM tree-reduce 패턴 (llama.cpp 유래, 2026-04-14 cross-run
// 검증 후 B-4 sub_group_reduce 버전에서 revert). Adreno 830 측정 결과
// `sub_group_reduce_*` + `REQD_SUBGROUP_SIZE_64` 조합이 SLM tree-reduce보다
// 33-55% 느렸음 (.agent/research/microbench_flash_attn/cross_run_verdict.md).
__kernel void flash_attn_f32_f16_q1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset,
    // Post-softmax score output (arg 40-43). When `write_scores != 0`, the
    // kernel writes per-token attention weights into
    // `S[score_layer_offset + head_idx * score_stride + k_idx]`. `S` must be
    // a valid buffer (use a dummy 1-element buffer when `write_scores == 0`).
    global float * S,
    const int score_layer_offset,
    const int score_stride,
    const int write_scores
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const int score_row_base = score_layer_offset + head_idx * score_stride;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    // SLM tree-reduce for m_i (cross-run verified faster than sub_group_reduce on Adreno).
    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const ACC_TYPE p = exp(score - m_final);
        // Pre-normalize p snapshot — finalized via `/ l_final` below once the
        // SLM tree-reduce produces l_final. Adreno: uniform write_scores
        // across the WG keeps this branch divergence-free.
        if (write_scores) {
            S[score_row_base + k_idx] = (float)p;
        }
        l_i += p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; i++) {
            o_acc[i] = mad(p, CONVERT_KV_ACC4(v_ptr[i]), o_acc[i]);
        }
    }

    // SLM tree-reduce for l_i, plus shared local_o_comp buffer for o_acc reductions.
    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
    ACC_TYPE l_final = local_l[0];

    if (sinks_ptr != NULL) {
        l_final += exp(sinks_ptr[head_idx] - m_final);
    }

    // Post-normalize scores to `exp(score - m_final) / l_final`, matching the
    // softmax weight written by `kernel_attn_gen_half`. Strided across the WG
    // for coalesced global writes.
    if (write_scores && l_final > 0.0f) {
        const float inv_l = 1.0f / (float)l_final;
        for (int t = tid; t < n_kv; t += Q1_WG_SIZE) {
            S[score_row_base + t] *= inv_l;
        }
    }

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
        for (int i = 0; i < DV_VEC; i++) {
            local_o_comp[tid] = o_acc[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = CONVERT_O_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
    }
}

// ---------------------------------------------------------------------------
// `flash_attn_f32_f16_q1_dynkv` — D-D.6 OOP variant.
//
// `n_kv` argument is read from a 1-element INT_32 buffer (`n_kv_buf[0]`)
// instead of a build-time SCALAR. Lets QNN OpPackage backend update
// `n_kv` per `graphExecute` for multi-token decode without re-finalize.
// All other arguments and behavior are identical to `flash_attn_f32_f16_q1`.
__kernel void flash_attn_f32_f16_q1_dynkv(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const global int * n_kv_buf,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset,
    global float * S,
    const int score_layer_offset,
    const int score_stride,
    const int write_scores
) {
    const int n_kv = n_kv_buf[0];

    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const int score_row_base = score_layer_offset + head_idx * score_stride;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const ACC_TYPE p = exp(score - m_final);
        if (write_scores) {
            S[score_row_base + k_idx] = (float)p;
        }
        l_i += p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; i++) {
            o_acc[i] = mad(p, CONVERT_KV_ACC4(v_ptr[i]), o_acc[i]);
        }
    }

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
    ACC_TYPE l_final = local_l[0];

    if (sinks_ptr != NULL) {
        l_final += exp(sinks_ptr[head_idx] - m_final);
    }

    if (write_scores && l_final > 0.0f) {
        const float inv_l = 1.0f / (float)l_final;
        for (int t = tid; t < n_kv; t += Q1_WG_SIZE) {
            S[score_row_base + t] *= inv_l;
        }
    }

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
        for (int i = 0; i < DV_VEC; i++) {
            local_o_comp[tid] = o_acc[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = CONVERT_O_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
    }
}

// ---------------------------------------------------------------------------
// UMA Hybrid CPU-GPU Attention — partial variant of `flash_attn_f32_f16_q1`.
//
// 차이점 (vs `flash_attn_f32_f16_q1`):
//   1. KV loop 범위를 [kv_start, kv_end)로 한정한다. 기존 kernel의 n_kv 전체
//      루프 대신 GPU가 담당한 슬라이스만 처리.
//   2. 결과는 **정규화 전(`o_unnorm`)** 형태로 기록된다. 즉 `o / l_final`
//      나눗셈을 수행하지 않는다. 호스트가 CPU partial과 함께 merge 단계에서
//      최종 정규화를 수행한다.
//   3. score 출력 인자(기존 40~43)는 제거된다. 대신 아래 5개 출력/파라미터가
//      추가된다:
//        - `const int kv_start, const int kv_end`
//        - `global float * partial_ml`    : f32[n_heads_q * 2]
//              layout: partial_ml[q_h*2 + 0] = m_final
//                      partial_ml[q_h*2 + 1] = l_final
//        - `global float * partial_o`     : f32[n_heads_q * head_dim]
//              layout: partial_o[q_h*head_dim .. (q_h+1)*head_dim]
//                      = Σ exp(s_t - m_final) · V[t]    (정규화 전)
//        - `global volatile int * ready_flags` : int[n_heads_q]
//              head별 완료 sigflag. WG가 partial_ml/partial_o 쓰기를 마친 후
//              atomic_xchg(&ready_flags[head_idx], 1)로 release.
//   4. sinks_void가 NULL이 아닐 경우에도 이 partial kernel은 sink를 적용하지
//      않는다. sink는 merge 단계에서 **정규화와 함께** 반영되어야 하며,
//      partial의 (m, l, o_unnorm) 불변식과 호환되지 않는다. Stage A의 CPU
//      partial도 동일하게 sink를 미적용 (tests/integration에서 확인).
//
// WG / GWS 규약 (Stage C의 plan.rs가 맞춰야 함):
//   - gws = [Q1_WG_SIZE=64, n_heads_q, 1], lws = [Q1_WG_SIZE, 1, 1]
//   - 즉 head당 1 WG, head별 독립. WG 내부 tid ∈ [0, 64)가 협력.
//
// Adreno register pressure 참고:
//   - q_priv[DK_VEC] (32 float4 @DK=128), o_acc[DV_VEC] (32 float4 @DV=128)로
//     기존 Q1과 동일한 사용량. 본 partial은 정규화 스텝을 생략하지만 그 외
//     구조는 그대로 유지하여 spill 리스크를 늘리지 않는다.
//   - 새 __local 버퍼는 추가하지 않는다. 기존 local_m/local_l/local_o_comp를
//     재사용.
// ---------------------------------------------------------------------------
__kernel void flash_attn_f32_f16_q1_partial(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,      // unused (호환성용, partial은 o_row를 건드리지 않음)
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset,
    // Partial 전용 인자 (arg 40-44)
    const int kv_start,
    const int kv_end,
    global float * partial_ml,
    global float * partial_o,
    global volatile int * ready_flags
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    // Partial: 범위가 비어 있으면 (m=-inf, l=0, o=0)를 기록하고 ready_flag를 세운 후 종료.
    // CPU `flash_partial_kv_range_f16`의 빈 범위 처리와 bit-exact.
    //
    // kv_start/kv_end는 커널 enqueue 시점에 동일 값이 모든 lane에 바인딩되므로
    // 이 분기는 WG 전체가 균일하게(uniform하게) 취하며, `barrier`를 분기 안에
    // 두어도 OpenCL 1.2 semantics 위반이 아니다.
    if (kv_start >= kv_end) {
        if (tid == 0) {
            partial_ml[head_idx * 2 + 0] = -INFINITY;
            partial_ml[head_idx * 2 + 1] = 0.0f;
            const int o_base_idx = head_idx * DV_VEC; // float4 단위
            global float4 * po4 = (global float4 *)partial_o;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                po4[o_base_idx + i] = (float4)(0.0f);
            }
        }
        // GPU 쓰기 완료 후 sigflag release. barrier는 `if (tid == 0)` 바깥에 둔다.
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (tid == 0) {
            atomic_xchg(&ready_flags[head_idx], 1);
        }
        return;
    }

    // --- Pass 1: per-lane m_i over [kv_start, kv_end) -----------------------
    // 기존 Q1과 달리 sink를 적용하지 않는다 (merge 단계에서 반영).
    ACC_TYPE m_i = -INFINITY;
    for (int k_idx = kv_start + tid; k_idx < kv_end; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    // SLM tree-reduce for m (Adreno 실측 우위, feedback_adreno_subgroup_reduce 참조).
    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    // --- Pass 2: l_i + o_acc (정규화 전) over [kv_start, kv_end) -----------
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = kv_start + tid; k_idx < kv_end; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const ACC_TYPE p = exp(score - m_final);
        l_i += p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; i++) {
            o_acc[i] = mad(p, CONVERT_KV_ACC4(v_ptr[i]), o_acc[i]);
        }
    }

    // SLM tree-reduce for l.
    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE l_final = local_l[0];

    // o_unnorm 쓰기: WG-wide sum reduce per-component, tid==0이 partial_o에 store.
    // 정규화(/l_final) 수행하지 않음 — 이것이 partial의 정의.
    const int o_base_idx = head_idx * DV_VEC; // float4 단위 인덱스
    global float4 * po4 = (global float4 *)partial_o;
    for (int i = 0; i < DV_VEC; i++) {
        local_o_comp[tid] = o_acc[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) {
            po4[o_base_idx + i] = local_o_comp[0];
        }
        // 다음 iter가 local_o_comp를 덮어쓰기 전에 모든 lane이 위 reduce를 완료했는지 보장.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        partial_ml[head_idx * 2 + 0] = (float)m_final;
        partial_ml[head_idx * 2 + 1] = (float)l_final;
    }
    // GPU 쓰기가 글로벌 메모리에 도달한 뒤 sigflag를 release.
    // CPU가 ready_flags[head_idx] == 1을 본 시점에서 partial_ml/partial_o
    // 내용이 캐시 무효화 경로로도 일관되게 보이도록 보장하는 것은 Stage C의
    // map/flush 규약 책임 (host 쪽 atomic load로 pairing).
    //
    // NOTE: barrier는 OpenCL 1.2 명세상 uniform control flow 아래에서만
    // 정의되므로 `if (tid == 0)` 분기 바깥에 둔다.
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid == 0) {
        atomic_xchg(&ready_flags[head_idx], 1);
    }
}
