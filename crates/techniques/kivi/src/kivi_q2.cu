// CUDA port of kivi_q2.cl — KIVI Q2 dequant / scatter / gather data-movement kernels.
//
// Byte-for-byte the same index math as the OpenCL kernels (only the CUDA path's F16-output
// variants + the F32 gather are ported — the F32-output dequant variants are unused on the
// CUDA quant-window path). One thread per output element; no shared memory, no atomics
// (every destination address is unique). Compiled at plugin make() time by shelling out to
// system nvcc (parity with the engine's kernels.cu + attn-score's score_reduce.cu).
//
// Q2_0 block format (12 bytes): [0..2] d:f16 scale, [2..4] m:f16 min, [4..12] qs:u8[8]
//   (32 values, 4 per byte, LSB first). dequant: out = ((qs[i/4] >> ((i%4)*2)) & 0x03) * d + m.

#include <cuda_fp16.h>

#define Q2_BLOCK_SIZE 12
#define Q2_GROUP_SIZE 32

// ─── Value dequant (per-token) → F16 SeqMajor [max_seq, kv_heads, head_dim] ───────────────────
extern "C" __global__ void kivi_dequantize_value_q2_f16(
    const unsigned char * __restrict__ q2_data,
    __half * __restrict__ attn_v,
    int kv_heads,
    int head_dim,
    int flush_tokens,
    int tok_base,
    int block_offset)
{
    const int bid = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_token = head_dim / Q2_GROUP_SIZE;
    const int total_blocks = kv_heads * flush_tokens * blocks_per_token;
    if (bid >= total_blocks) return;

    const int b = bid % blocks_per_token;
    const int temp = bid / blocks_per_token;
    const int t = temp % flush_tokens;
    const int h = temp / flush_tokens;

    const int src_off = (block_offset + bid) * Q2_BLOCK_SIZE;
    const float d = __half2float(*reinterpret_cast<const __half*>(q2_data + src_off));
    const float m = __half2float(*reinterpret_cast<const __half*>(q2_data + src_off + 2));

    const int dst_base = (tok_base + t) * kv_heads * head_dim + h * head_dim + b * Q2_GROUP_SIZE;

    for (int i = 0; i < 8; i++) {
        const unsigned char byte = q2_data[src_off + 4 + i];
        attn_v[dst_base + i * 4 + 0] = __float2half((float)((byte >> 0) & 0x03) * d + m);
        attn_v[dst_base + i * 4 + 1] = __float2half((float)((byte >> 2) & 0x03) * d + m);
        attn_v[dst_base + i * 4 + 2] = __float2half((float)((byte >> 4) & 0x03) * d + m);
        attn_v[dst_base + i * 4 + 3] = __float2half((float)((byte >> 6) & 0x03) * d + m);
    }
}

// ─── Key dequant (per-channel scatter) → F16 SeqMajor [max_seq, kv_heads, head_dim] ───────────
extern "C" __global__ void kivi_dequantize_key_q2_f16(
    const unsigned char * __restrict__ q2_data,
    __half * __restrict__ attn_k,
    int kv_heads,
    int head_dim,
    int groups_per_flush,
    int tok_base,
    int block_offset)
{
    const int bid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_blocks = kv_heads * groups_per_flush * head_dim;
    if (bid >= total_blocks) return;

    const int ch = bid % head_dim;
    const int temp = bid / head_dim;
    const int g = temp % groups_per_flush;
    const int h = temp / groups_per_flush;

    const int src_off = (block_offset + bid) * Q2_BLOCK_SIZE;
    const float d = __half2float(*reinterpret_cast<const __half*>(q2_data + src_off));
    const float m = __half2float(*reinterpret_cast<const __half*>(q2_data + src_off + 2));

    const int tok_start = tok_base + g * Q2_GROUP_SIZE;
    const int head_offset = h * head_dim + ch;

    for (int i = 0; i < 8; i++) {
        const unsigned char byte = q2_data[src_off + 4 + i];
        const int base_t = i * 4;
        attn_k[(tok_start + base_t + 0) * kv_heads * head_dim + head_offset] =
            __float2half((float)((byte >> 0) & 0x03) * d + m);
        attn_k[(tok_start + base_t + 1) * kv_heads * head_dim + head_offset] =
            __float2half((float)((byte >> 2) & 0x03) * d + m);
        attn_k[(tok_start + base_t + 2) * kv_heads * head_dim + head_offset] =
            __float2half((float)((byte >> 4) & 0x03) * d + m);
        attn_k[(tok_start + base_t + 3) * kv_heads * head_dim + head_offset] =
            __float2half((float)((byte >> 6) & 0x03) * d + m);
    }
}

// ─── Residual scatter: F32 [kv_heads, res_cap, head_dim] → F16 [max_seq, kv_heads, head_dim] ───
extern "C" __global__ void kivi_scatter_residual_f16(
    const float * __restrict__ residual,
    __half * __restrict__ attn,
    int kv_heads,
    int res_cap,
    int head_dim,
    int res_pos,
    int tok_base)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kv_heads * res_pos * head_dim;
    if (tid >= total) return;

    const int d = tid % head_dim;
    const int tmp = tid / head_dim;
    const int t = tmp % res_pos;
    const int h = tmp / res_pos;

    const int src_idx = h * res_cap * head_dim + t * head_dim + d;
    const int dst_idx = (tok_base + t) * kv_heads * head_dim + h * head_dim + d;
    attn[dst_idx] = __float2half(residual[src_idx]);
}

// ─── Update gather: F32 [seq_len, kv_heads, head_dim] → F32 [kv_heads, res_cap, head_dim] ──────
extern "C" __global__ void kivi_gather_update(
    const float * __restrict__ input,
    float * __restrict__ residual,
    int kv_heads,
    int res_cap,
    int head_dim,
    int seq_len,
    int res_pos)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
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
