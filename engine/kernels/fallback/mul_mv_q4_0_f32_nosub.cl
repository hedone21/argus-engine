// Fallback Q4_0 matmul kernel without subgroup operations (e.g., for pocl)
// Same function name and signature as mul_mv_q4_0_f32.cl
// Uses a simple per-row dot product with local memory tree reduction.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK4_0 32

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

struct block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2];
};

kernel void kernel_mul_mat_q4_0_f32(
        global void * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    // Each workgroup computes one output element: dst[r1, r0]
    int r0 = get_group_id(0);  // row of src0 (output row)
    int r1 = get_group_id(1);  // row of src1 (output col)
    int im = get_group_id(2);  // batch index

    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int i12 = im % ne12;
    int i13 = im / ne12;

    int nb = ne00 / QK4_0;  // number of Q4_0 blocks per row

    ulong src0_offset = (ulong)r0 * nb + (i12/r2) * ((ulong)nb * ne01) + (i13/r3) * ((ulong)nb * ne01 * ne02);
    global struct block_q4_0 * x = (global struct block_q4_0 *)src0 + src0_offset;
    global float * y = (global float *)src1 + r1 * ne10 + im * ne00 * ne1;

    // Each work-item dequantizes and accumulates a subset of blocks
    float sumf = 0.0f;
    for (int ib = lid; ib < nb; ib += local_size) {
        float d = (float)x[ib].d;
        global uint8_t * qs = x[ib].qs;

        // Dequantize and dot product with y for this block
        for (int j = 0; j < QK4_0 / 2; j++) {
            uint8_t q = qs[j];
            float v0 = (float)(q & 0x0F) - 8.0f;
            float v1 = (float)((q >> 4) & 0x0F) - 8.0f;

            sumf += d * v0 * y[ib * QK4_0 + j];
            sumf += d * v1 * y[ib * QK4_0 + j + QK4_0 / 2];
        }
    }

    // Tree reduction in local memory
    local float scratch[256];
    scratch[lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        dst[r1 * ne0 + im * ne0 * ne1 + r0] = scratch[0];
    }
}
