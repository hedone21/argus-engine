// GPU-native dequant of opaque q2_0 KV (W-CODEC slice 3).
//
// Decodes the descriptor-floor q2_0 format (asymmetric, ScaleLayout::PerBlockF16WithMin /
// Packing::Quad) into a device F16 HeadMajor KV buffer, so the existing F16 flash-attention
// kernels can run on it without the host CpuBackend floor (host dequant -> CPU attention ->
// write_buffer round-trip).
//
// q2_0 block (12 bytes):   [f16 d][f16 m][8 x uint8 qs]   (32 values, 4 per byte, LSB first)
//   value = (float)((qs[i/4] >> ((i%4)*2)) & 0x03) * d + m
// This is byte-identical to crates/argus-kv-codec BlockQ2_0::dequantize and the host
// `unpack_block_via_descriptor` Quad arm (engine/src/format/dtype_layout.rs).
//
// Source layout (descriptor floor, HeadMajor) for (head h, position pos):
//   bph = (head_dim / 32) * 12 bytes, at byte offset (h*capacity + pos)*bph; block b within at +b*12.
//   (matches apply_weighted_merges_opaque into_off = (h*capacity + pos)*bph and
//    KVCache::opaque_bytes_per_head = (head_dim / block_elems) * block_bytes.)
// Output layout (F16 HeadMajor [kv_heads, capacity, head_dim]):
//   out[(h*capacity + pos)*head_dim + b*32 + i]
//
// Dispatched over kv_heads * seq_len * (head_dim/32) work items — only the valid positions
// [0, seq_len) are decoded; the [seq_len, capacity) tail is left untouched (never read by
// flash attention, which is bounded by cache_seq_len).
//
// BYTE-EXACTNESS: this kernel is compiled WITHOUT -cl-mad-enable / -cl-fast-relaxed-math (its own
// strict program), so `q * d + m` is two correctly-rounded f32 ops (no FMA contraction) and the
// f16 result is bit-identical to the host decode (decode_via_descriptor then f16::from_f32).
// Do NOT introduce mad()/fma()/-cl-fast-relaxed-math here — q=3 can diverge by 1 ULP under
// contraction, breaking the byte-exact dequant invariant.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define Q2_BLOCK_BYTES 12
#define Q2_GROUP       32

__kernel void kernel_dequant_opaque_q2_to_f16(
    __global const uchar* q2_data,   // opaque q2 KV (HeadMajor)
    __global half* out_f16,          // [kv_heads, capacity, head_dim]
    const int kv_heads,
    const int head_dim,
    const int capacity,
    const int seq_len                // valid positions [0, seq_len)
) {
    const int bid = get_global_id(0);
    const int blocks_per_pos = head_dim / Q2_GROUP;
    const int total = kv_heads * seq_len * blocks_per_pos;
    if (bid >= total) return;

    const int b   = bid % blocks_per_pos;
    const int tmp = bid / blocks_per_pos;       // h*seq_len + pos
    const int pos = tmp % seq_len;
    const int h   = tmp / seq_len;

    const int bph = blocks_per_pos * Q2_BLOCK_BYTES;
    const int src_off = (h * capacity + pos) * bph + b * Q2_BLOCK_BYTES;

    const float d = vload_half(0, (__global const half*)(q2_data + src_off));
    const float m = vload_half(0, (__global const half*)(q2_data + src_off + 2));

    const int dst_base = (h * capacity + pos) * head_dim + b * Q2_GROUP;

    for (int i = 0; i < 8; i++) {
        const uchar byte = q2_data[src_off + 4 + i];
        // Explicit non-contracted multiply-then-add (host op order: q*scale then +min).
        const float v0 = (float)((byte >> 0) & 0x03) * d + m;
        const float v1 = (float)((byte >> 2) & 0x03) * d + m;
        const float v2 = (float)((byte >> 4) & 0x03) * d + m;
        const float v3 = (float)((byte >> 6) & 0x03) * d + m;
        out_f16[dst_base + i * 4 + 0] = convert_half(v0);
        out_f16[dst_base + i * 4 + 1] = convert_half(v1);
        out_f16[dst_base + i * 4 + 2] = convert_half(v2);
        out_f16[dst_base + i * 4 + 3] = convert_half(v3);
    }
}
