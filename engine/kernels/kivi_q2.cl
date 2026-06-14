// KIVI Q2 dequantization and data movement kernels for GPU-native KiviCache.
//
// Q2_0 block format (12 bytes):
//   [0..2] d: float16 — scale = (max - min) / 3
//   [2..4] m: float16 — minimum (zero point)
//   [4..12] qs: uint8[8] — 32 × 2-bit packed (4 values per byte, LSB first)
//
// Dequantization: out[i] = ((qs[i/4] >> ((i%4)*2)) & 0x03) * d + m

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define Q2_BLOCK_SIZE  12   // bytes per Q2_0 block
#define Q2_GROUP_SIZE  32   // values per Q2_0 block

// ─── Value dequantization (per-token) ────────────────────────────────────────
//
// Q2 block order for Values (from flush_residual):
//   for h in 0..kv_heads:
//     for t in 0..flush_tokens:
//       for b in 0..blocks_per_token:   (blocks_per_token = head_dim / 32)
//         one Q2 block = 32 values of V[h, t, b*32..(b+1)*32]
//
// Output layout: SeqMajor [total_seq, kv_heads, head_dim]
//   attn_v[(tok_base + t) * kv_heads * head_dim + h * head_dim + b*32 + i]
//
// Each work item dequantizes one Q2 block (32 values).

__kernel void kivi_dequantize_value_q2(
    __global const uchar* q2_data,    // packed Q2 blocks
    __global float* attn_v,           // [max_seq, kv_heads, head_dim]
    const int kv_heads,
    const int head_dim,
    const int flush_tokens,           // tokens in this flush batch
    const int tok_base,               // starting token index in attn buffer
    const int block_offset            // starting block index in q2 storage
) {
    const int bid = get_global_id(0);
    const int blocks_per_token = head_dim / Q2_GROUP_SIZE;
    const int total_blocks = kv_heads * flush_tokens * blocks_per_token;
    if (bid >= total_blocks) return;

    // Decompose block index → (head, token, dim_block)
    const int b = bid % blocks_per_token;
    const int temp = bid / blocks_per_token;
    const int t = temp % flush_tokens;
    const int h = temp / flush_tokens;

    // Read Q2 block header
    const int src_off = (block_offset + bid) * Q2_BLOCK_SIZE;
    const float d = vload_half(0, (__global const half*)(q2_data + src_off));
    const float m = vload_half(0, (__global const half*)(q2_data + src_off + 2));

    // Output position
    const int dst_base = (tok_base + t) * kv_heads * head_dim + h * head_dim + b * Q2_GROUP_SIZE;

    // Dequantize 32 values (8 bytes × 4 values/byte)
    for (int i = 0; i < 8; i++) {
        const uchar byte = q2_data[src_off + 4 + i];
        attn_v[dst_base + i * 4 + 0] = (float)((byte >> 0) & 0x03) * d + m;
        attn_v[dst_base + i * 4 + 1] = (float)((byte >> 2) & 0x03) * d + m;
        attn_v[dst_base + i * 4 + 2] = (float)((byte >> 4) & 0x03) * d + m;
        attn_v[dst_base + i * 4 + 3] = (float)((byte >> 6) & 0x03) * d + m;
    }
}

// ─── Key dequantization (per-channel scatter) ────────────────────────────────
//
// Q2 block order for Keys (from flush_residual):
//   for h in 0..kv_heads:
//     for g in 0..groups_per_flush:    (groups_per_flush = flush_tokens / 32)
//       for ch in 0..head_dim:
//         one Q2 block = 32 values of K[h, g*32..(g+1)*32, ch]
//         (i.e., one channel across 32 consecutive tokens)
//
// Output layout: SeqMajor [total_seq, kv_heads, head_dim]
//   attn_k[(tok_base + g*32 + i) * kv_heads * head_dim + h * head_dim + ch]
//
// Each work item dequantizes one Q2 block and scatters to 32 output positions.

__kernel void kivi_dequantize_key_q2(
    __global const uchar* q2_data,
    __global float* attn_k,           // [max_seq, kv_heads, head_dim]
    const int kv_heads,
    const int head_dim,
    const int groups_per_flush,       // = flush_tokens / 32
    const int tok_base,               // starting token index in attn buffer
    const int block_offset            // starting block index in q2 storage
) {
    const int bid = get_global_id(0);
    const int total_blocks = kv_heads * groups_per_flush * head_dim;
    if (bid >= total_blocks) return;

    // Decompose block index → (head, group, channel)
    const int ch = bid % head_dim;
    const int temp = bid / head_dim;
    const int g = temp % groups_per_flush;
    const int h = temp / groups_per_flush;

    // Read Q2 block header
    const int src_off = (block_offset + bid) * Q2_BLOCK_SIZE;
    const float d = vload_half(0, (__global const half*)(q2_data + src_off));
    const float m = vload_half(0, (__global const half*)(q2_data + src_off + 2));

    // Scatter dequantized values to 32 different token positions
    const int tok_start = tok_base + g * Q2_GROUP_SIZE;
    const int head_offset = h * head_dim + ch;

    for (int i = 0; i < 8; i++) {
        const uchar byte = q2_data[src_off + 4 + i];
        const int base_t = i * 4;

        attn_k[(tok_start + base_t + 0) * kv_heads * head_dim + head_offset] =
            (float)((byte >> 0) & 0x03) * d + m;
        attn_k[(tok_start + base_t + 1) * kv_heads * head_dim + head_offset] =
            (float)((byte >> 2) & 0x03) * d + m;
        attn_k[(tok_start + base_t + 2) * kv_heads * head_dim + head_offset] =
            (float)((byte >> 4) & 0x03) * d + m;
        attn_k[(tok_start + base_t + 3) * kv_heads * head_dim + head_offset] =
            (float)((byte >> 6) & 0x03) * d + m;
    }
}

// ─── Residual scatter: [kv_heads, res_cap, head_dim] → [seq, kv_heads, head_dim] ──
//
// Copies residual F32 data from head-first layout to SeqMajor attention buffer.
// Called every decode step to update the residual portion of the attention view.

__kernel void kivi_scatter_residual(
    __global const float* residual,   // [kv_heads, res_cap, head_dim]
    __global float* attn,             // [max_seq, kv_heads, head_dim]
    const int kv_heads,
    const int res_cap,
    const int head_dim,
    const int res_pos,                // valid tokens in residual (0..res_pos)
    const int tok_base                // q2_tokens offset in attn buffer
) {
    const int tid = get_global_id(0);
    const int total = kv_heads * res_pos * head_dim;
    if (tid >= total) return;

    const int d = tid % head_dim;
    const int tmp = tid / head_dim;
    const int t = tmp % res_pos;
    const int h = tmp / res_pos;

    const int src_idx = h * res_cap * head_dim + t * head_dim + d;
    const int dst_idx = (tok_base + t) * kv_heads * head_dim + h * head_dim + d;
    attn[dst_idx] = residual[src_idx];
}

// ─── Update gather: [seq_len, kv_heads, head_dim] → [kv_heads, res_cap, head_dim] ──
//
// Writes new K/V tokens from SeqMajor input to head-first residual buffer.
// Called during update() to scatter incoming tokens.

// ─── F16-output dequant/scatter kernels ─────────────────────────────────────
//
// Identical logic to the F32 variants above, but write half instead of float.
// Used when gpu_attn_k/gpu_attn_v are allocated as F16 buffers to halve VRAM.

__kernel void kivi_dequantize_value_q2_f16(
    __global const uchar* q2_data,
    __global half* attn_v,            // [max_seq, kv_heads, head_dim] F16
    const int kv_heads,
    const int head_dim,
    const int flush_tokens,
    const int tok_base,
    const int block_offset
) {
    const int bid = get_global_id(0);
    const int blocks_per_token = head_dim / Q2_GROUP_SIZE;
    const int total_blocks = kv_heads * flush_tokens * blocks_per_token;
    if (bid >= total_blocks) return;

    const int b = bid % blocks_per_token;
    const int temp = bid / blocks_per_token;
    const int t = temp % flush_tokens;
    const int h = temp / flush_tokens;

    const int src_off = (block_offset + bid) * Q2_BLOCK_SIZE;
    const float d = vload_half(0, (__global const half*)(q2_data + src_off));
    const float m = vload_half(0, (__global const half*)(q2_data + src_off + 2));

    const int dst_base = (tok_base + t) * kv_heads * head_dim + h * head_dim + b * Q2_GROUP_SIZE;

    for (int i = 0; i < 8; i++) {
        const uchar byte = q2_data[src_off + 4 + i];
        attn_v[dst_base + i * 4 + 0] = convert_half((float)((byte >> 0) & 0x03) * d + m);
        attn_v[dst_base + i * 4 + 1] = convert_half((float)((byte >> 2) & 0x03) * d + m);
        attn_v[dst_base + i * 4 + 2] = convert_half((float)((byte >> 4) & 0x03) * d + m);
        attn_v[dst_base + i * 4 + 3] = convert_half((float)((byte >> 6) & 0x03) * d + m);
    }
}

__kernel void kivi_dequantize_key_q2_f16(
    __global const uchar* q2_data,
    __global half* attn_k,            // [max_seq, kv_heads, head_dim] F16
    const int kv_heads,
    const int head_dim,
    const int groups_per_flush,
    const int tok_base,
    const int block_offset
) {
    const int bid = get_global_id(0);
    const int total_blocks = kv_heads * groups_per_flush * head_dim;
    if (bid >= total_blocks) return;

    const int ch = bid % head_dim;
    const int temp = bid / head_dim;
    const int g = temp % groups_per_flush;
    const int h = temp / groups_per_flush;

    const int src_off = (block_offset + bid) * Q2_BLOCK_SIZE;
    const float d = vload_half(0, (__global const half*)(q2_data + src_off));
    const float m = vload_half(0, (__global const half*)(q2_data + src_off + 2));

    const int tok_start = tok_base + g * Q2_GROUP_SIZE;
    const int head_offset = h * head_dim + ch;

    for (int i = 0; i < 8; i++) {
        const uchar byte = q2_data[src_off + 4 + i];
        const int base_t = i * 4;

        attn_k[(tok_start + base_t + 0) * kv_heads * head_dim + head_offset] =
            convert_half((float)((byte >> 0) & 0x03) * d + m);
        attn_k[(tok_start + base_t + 1) * kv_heads * head_dim + head_offset] =
            convert_half((float)((byte >> 2) & 0x03) * d + m);
        attn_k[(tok_start + base_t + 2) * kv_heads * head_dim + head_offset] =
            convert_half((float)((byte >> 4) & 0x03) * d + m);
        attn_k[(tok_start + base_t + 3) * kv_heads * head_dim + head_offset] =
            convert_half((float)((byte >> 6) & 0x03) * d + m);
    }
}

__kernel void kivi_scatter_residual_f16(
    __global const float* residual,   // [kv_heads, res_cap, head_dim] F32
    __global half* attn,              // [max_seq, kv_heads, head_dim] F16
    const int kv_heads,
    const int res_cap,
    const int head_dim,
    const int res_pos,
    const int tok_base
) {
    const int tid = get_global_id(0);
    const int total = kv_heads * res_pos * head_dim;
    if (tid >= total) return;

    const int d = tid % head_dim;
    const int tmp = tid / head_dim;
    const int t = tmp % res_pos;
    const int h = tmp / res_pos;

    const int src_idx = h * res_cap * head_dim + t * head_dim + d;
    const int dst_idx = (tok_base + t) * kv_heads * head_dim + h * head_dim + d;
    attn[dst_idx] = convert_half(residual[src_idx]);
}

// ─── Update gather: [seq_len, kv_heads, head_dim] → [kv_heads, res_cap, head_dim] ──
//
// Writes new K/V tokens from SeqMajor input to head-first residual buffer.
// Called during update() to scatter incoming tokens.

__kernel void kivi_gather_update(
    __global const float* input,      // [seq_len, kv_heads, head_dim]
    __global float* residual,         // [kv_heads, res_cap, head_dim]
    const int kv_heads,
    const int res_cap,
    const int head_dim,
    const int seq_len,
    const int res_pos                 // write position in residual
) {
    const int tid = get_global_id(0);
    const int total = seq_len * kv_heads * head_dim;
    if (tid >= total) return;

    const int d = tid % head_dim;
    const int tmp = tid / head_dim;
    const int h = tmp % kv_heads;
    const int s = tmp / kv_heads;

    const int src_idx = s * kv_heads * head_dim + h * head_dim + d;
    const int dst_idx = h * res_cap * head_dim + (res_pos + s) * head_dim + d;
    residual[dst_idx] = input[src_idx];
}
