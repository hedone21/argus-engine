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
            // `vload_half4` already converts to float4 — assigning it to a `half4` is a type
            // error, which is why this program never compiled on any platform (2026-09-03).
            float4 kf = vload_half4(0, k_ptr + d);
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
            // `vload_half4` already converts to float4 — assigning it to a `half4` is a type
            // error, which is why this program never compiled on any platform (2026-09-03).
            float4 kf = vload_half4(0, k_ptr + d);
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
            // `vload_half4` already converts to float4 — assigning it to a `half4` is a type
            // error, which is why this program never compiled on any platform (2026-09-03).
            float4 kf = vload_half4(0, k_ptr + d);
            float4 qf = vload4(0, q_ptr + d);
            dot += qf.x * kf.x + qf.y * kf.y + qf.z * kf.z + qf.w * kf.w;
        }
        for (; d < head_dim; d++) {
            dot += q_ptr[d] * vload_half(d, k_ptr);
        }
        S[head_idx * score_stride + t] = exp(dot * scale - max_score) * inv_sum;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Decision-time observation-window attention (the GPU twin of
// `kv::aperturb_select::window_attention`).
//
// The CPU version materializes `[n_heads_q][rows][current_pos]` logits and then
// SUM-pools a per-row softmax into `[n_heads_q][current_pos]`. Here a workgroup
// owns one query head and walks the ring's rows in blocks of WATT_B, keeping
// only the block's scores in `ZROW`.
//
// Why blocks and not one row at a time: a row-at-a-time sweep re-reads the whole
// K head per row, which at 4096 resident tokens is 805 MB per layer and makes
// the pass memory-bound (measured on an Adreno 830: 1.4x over the CPU, no more).
// A block of WATT_B rows reads K once and computes WATT_B dot products from it,
// cutting that traffic by WATT_B.
//
// Differences from `kernel_score_only_half` above, which is the one-row decode
// twin: a row loop, a causal upper bound, a ragged lower bound, and a `+=`
// accumulate instead of a write.
//
// Every thread reads back only the `ZROW` slots it wrote itself (same strided
// assignment in every pass), so the passes need no global fence — the only
// barriers are the SLM reductions and the query-block load, and every condition
// that skips one is uniform across the workgroup.
//
// Requires `head_dim % 4 == 0` (the host checks) and a power-of-two local size.
//
// Args:
//   Q:            [n_layers][n_heads_q][rows][head_dim] F32, host-packed
//   K:            HeadMajor [kv_heads, capacity, head_dim] F16 — the live cache
//   KV_START:     [n_layers][n_kv_heads] I32 first resident slot per KV head
//   ACC:          [n_heads_q][current_pos] F32 output (the kernel zeroes it first)
//   ZROW:         [n_heads_q][WATT_B][current_pos] F32 scratch
//   q_layer_off:  element offset of this layer's block in Q
//   acc_off:      element offset of this layer's block in ACC
//   start_off:    element offset of this layer's row in KV_START
//   denom:        sqrt(head_dim) — a DIVISION, matching the CPU reference
//   scratch:      __local float[local_size]
//   qblk:         __local float[WATT_B * head_dim]
//   ZOUT:         [n_layers][n_heads_q][export_rows][current_pos] F32, the raw logits of the
//                 window's TRAILING `export_rows` rows — pass A's own z, before pass B turns it
//                 into exp(z - m). `aperturb::decide` scores its metric rows from exactly this
//                 quantity, so exporting it here is what lets the decision skip a second Q·Kᵀ.
//                 Its layer offset is derived, not passed: ACC packs [l][h][p] and ZOUT packs
//                 [l][h][t][p] over the same `h` and `p`, so `acc_off * export_rows` names it.
//   export_rows:  how many trailing rows to export. 0 disables the export entirely (the CPU
//                 fallback path, and any caller that has no metric to feed).

// The host always passes `-DWATT_B`; this default only matters to a direct compile of the file,
// and matches what the host picks (8 — above that an Adreno 830 spills).
#ifndef WATT_B
#define WATT_B 8
#endif

__kernel void kernel_window_attn_sum_half(
    __global const float * Q,
    __global const half  * K,
    __global const int   * KV_START,
    __global float       * ACC,
    __global float       * ZROW,
    int head_dim,
    int n_heads_q,
    int n_kv_heads,
    int current_pos,
    int rows,
    int capacity,
    int q_layer_off,
    int acc_off,
    int start_off,
    float denom,
    __local float * scratch,
    __local float * qblk,
    // Appended AFTER the two `__local` arguments on purpose: the host sets `scratch`/`qblk` at
    // indices 15/16, and moving them would leave those indices unset (CL_INVALID_KERNEL_ARGS).
    __global float * ZOUT,
    int export_rows
) {
    const int h = get_group_id(0);
    const int lid = get_local_id(0);
    const int ls = get_local_size(0);

    const int n_rep = max(n_heads_q / n_kv_heads, 1);
    const int kv_h = h / n_rep;
    const int kbase = kv_h * capacity * head_dim;
    const int qbase = q_layer_off + h * rows * head_dim;
    const int accb = acc_off + h * current_pos;
    const int zb = h * WATT_B * current_pos;

    // A ragged head's holes read 0.0, the same value the CPU leaves there.
    int start = (KV_START != 0) ? KV_START[start_off + kv_h] : 0;
    if (start > current_pos) start = current_pos;

    // Zero on the SAME stride the accumulate below uses, so each thread owns its own columns
    // outright. Splitting at `start` matters: with a ragged head whose start is not a multiple of
    // the group width, `p = lid + k*ls` and `p = start + lid + k*ls` are different column sets, and
    // one thread's zero would race another thread's `+=` (a local-memory barrier would not order it
    // — these are global writes).
    for (int p = lid; p < start; p += ls) {
        ACC[accb + p] = 0.0f;
    }
    for (int p = start + lid; p < current_pos; p += ls) {
        ACC[accb + p] = 0.0f;
    }

    const int first_pos = current_pos - rows;
    // Window row `t0 + i` is metric row `t0 + i - (rows - export_rows)`: the metric reads the
    // window's TAIL, so the mapping is an offset, not the identity and not a whole row block.
    const int first_export = rows - export_rows;

    // The export's holes need the same treatment as ACC's, and for a second reason: a row block
    // whose causal bound falls below `start` never runs pass A at all (`continue` below), so
    // without this its rows would hand the host whatever the previous decision left in the buffer.
    //
    // Only the holes, though — zeroing the whole `export_rows x current_pos` block and then
    // copying `[start, end_max)` back over it doubles this kernel's global write traffic in
    // exactly the production geometry (16 rows x `current_pos`, of which the copy overwrites all
    // but the ragged head). So each exported row zeroes `[0, start)` and `[end_max, current_pos)`
    // for ITS OWN block's `end_max` — the same arithmetic the block loop below does — and a row
    // whose block never runs (`end_max <= start`) zeroes end to end. Every column the host can
    // read is still written exactly once, by exactly one thread: the three ranges are disjoint, so
    // no zero can land on a column the copy owns, whatever the group width.
    const int zob = acc_off * export_rows + h * export_rows * current_pos;
    for (int i = 0; i < export_rows; ++i) {
        const int tw = first_export + i;              // this metric row's window row
        const int bt0 = (tw / WATT_B) * WATT_B;       // and the block that computes it
        const int bnb = min(WATT_B, rows - bt0);
        int bend = first_pos + bt0 + bnb;             // the block's widest causal bound
        if (bend > current_pos) bend = current_pos;
        if (bend < start) bend = start;               // block skipped: nothing is copied in
        const int zor = zob + i * current_pos;
        for (int p = lid; p < start; p += ls) {
            ZOUT[zor + p] = 0.0f;
        }
        for (int p = bend + lid; p < current_pos; p += ls) {
            ZOUT[zor + p] = 0.0f;
        }
    }

    for (int t0 = 0; t0 < rows; t0 += WATT_B) {
        const int nb = min(WATT_B, rows - t0);
        // The block's widest causal bound: row `t0 + nb - 1` sees the most keys.
        int end_max = first_pos + t0 + nb;
        if (end_max > current_pos) end_max = current_pos;
        if (end_max <= start) continue;

        // The block's query rows, read once into SLM and then broadcast to every
        // position this workgroup owns. A short tail block zero-fills the rows it
        // does not have, so the dot-product loop below needs no `i < nb` test in
        // its innermost body.
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < WATT_B * head_dim; i += ls) {
            qblk[i] = (i < nb * head_dim) ? Q[qbase + t0 * head_dim + i] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float mx[WATT_B];
        #pragma unroll
        for (int i = 0; i < WATT_B; ++i) {
            mx[i] = -3.0e38f;
        }

        // ── pass A: one K sweep, WATT_B logits from it ──
        for (int p = start + lid; p < end_max; p += ls) {
            __global const half * kp = K + kbase + p * head_dim;
            float acc[WATT_B];
            #pragma unroll
            for (int i = 0; i < WATT_B; ++i) {
                acc[i] = 0.0f;
            }
            for (int d = 0; d < head_dim; d += 4) {
                const float4 kf = vload_half4(0, kp + d);
                #pragma unroll
                for (int i = 0; i < WATT_B; ++i) {
                    acc[i] += dot(vload4(0, qblk + i * head_dim + d), kf);
                }
            }
            #pragma unroll
            for (int i = 0; i < WATT_B; ++i) {
                if (i < nb) {
                    const float z = acc[i] / denom;
                    ZROW[zb + i * current_pos + p] = z;
                    // Only the rows whose own causal bound reaches `p` see it.
                    if (p < first_pos + t0 + i + 1) {
                        mx[i] = fmax(mx[i], z);
                    }
                }
            }
        }

        // ── A1: copy out the raw z before passes B/C overwrite it with exp(z - m) ──
        // Deliberately the SAME column partition as pass A above: a thread reads back only the
        // ZROW slots it wrote itself, so this needs no global fence between the two — the same
        // property the rest of this kernel is built on.
        for (int i = 0; i < WATT_B; ++i) {
            const int te = t0 + i - first_export;
            if (i >= nb || te < 0) {
                continue;
            }
            const int zor = zob + te * current_pos;
            for (int p = start + lid; p < end_max; p += ls) {
                ZOUT[zor + p] = ZROW[zb + i * current_pos + p];
            }
        }

        // ── passes B and C, one row at a time over what pass A left in ZROW ──
        #pragma unroll
        for (int i = 0; i < WATT_B; ++i) {
            if (i >= nb) {
                continue;
            }
            int end = first_pos + t0 + i + 1;
            if (end > current_pos) {
                end = current_pos;
            }
            if (end <= start) {
                continue;
            }
            scratch[lid] = mx[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int r = ls / 2; r > 0; r >>= 1) {
                if (lid < r) {
                    scratch[lid] = fmax(scratch[lid], scratch[lid + r]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const float m = scratch[0];
            barrier(CLK_LOCAL_MEM_FENCE);

            float my_sum = 0.0f;
            for (int p = start + lid; p < end; p += ls) {
                const float e = exp(ZROW[zb + i * current_pos + p] - m);
                ZROW[zb + i * current_pos + p] = e;
                my_sum += e;
            }
            scratch[lid] = my_sum;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int r = ls / 2; r > 0; r >>= 1) {
                if (lid < r) {
                    scratch[lid] += scratch[lid + r];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const float inv = 1.0f / scratch[0];
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int p = start + lid; p < end; p += ls) {
                ACC[accb + p] += ZROW[zb + i * current_pos + p] * inv;
            }
        }
    }
}
