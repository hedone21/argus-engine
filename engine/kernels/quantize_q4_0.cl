#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK4_0 32

typedef struct {
    half d;
    uchar qs[QK4_0 / 2];
} block_q4_0;

__kernel void kernel_quantize_f32_to_q4_0(__global const float* src, __global block_q4_0* dst, const int num_elements) {
    const int i = get_global_id(0);
    const int num_blocks = num_elements / QK4_0;
    
    if (i >= num_blocks) return;

    __global const float* src_ptr = src + i * QK4_0;
    
    float max_abs = 0.0f;
    for (int j = 0; j < QK4_0; j++) {
        float v = fabs(src_ptr[j]);
        if (v > max_abs) max_abs = v;
    }

    float d = max_abs / 7.0f;
    float id = (d == 0.0f) ? 0.0f : (1.0f / d);

    dst[i].d = (half)d;

    for (int j = 0; j < QK4_0 / 2; j++) {
        float v0 = src_ptr[j] * id;
        float v1 = src_ptr[j + QK4_0 / 2] * id;

        // Clamp and rebase to 0..15 range
        // We want (round(v) + 8) to be in [0, 15]
        // v is in [-7, 7] roughly.
        // If v = -8, +8 = 0.
        // If v = 7, +8 = 15.
        
        uchar b0 = (uchar)(clamp(round(v0), -8.0f, 7.0f) + 8.0f);
        uchar b1 = (uchar)(clamp(round(v1), -8.0f, 7.0f) + 8.0f);

        dst[i].qs[j] = b0 | (b1 << 4);
    }
}
