// Fallback F32 matmul kernel without subgroup operations (e.g., for pocl)
// Same function name and signature as mul_mv_f32_f32.cl
// Uses local memory tree reduction instead of sub_group_reduce_add.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define N_F32_F32 4

kernel void kernel_mul_mat_f32_f32(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F32_F32;
    int im = get_group_id(2);

    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global float * x = (global float *) (src0 + offset_src0);

    // Local memory for reduction (allocated by host)
    local float scratch[256];

    for (int row = 0; row < N_F32_F32; ++row) {
        int r1 = rb + row;
        if (r1 >= ne11) {
            break;
        }

        ulong offset_src1 = r1*nb11 + (i12)*nb12 + (i13)*nb13;
        global float * y = (global float *) (src1 + offset_src1);

        // Each work-item computes partial dot product
        float sumf = 0.0f;
        for (int i = lid; i < ne00; i += local_size) {
            sumf += x[i] * y[i];
        }

        // Tree reduction in local memory
        scratch[lid] = sumf;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = scratch[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
