// F16 GEMV — dual-mode: subgroup reduce (zero barriers) when available,
// tree reduction fallback otherwise.
// N_DST=4: 4 rows per WG for activation reuse.
// Uses vload_half4/vload_half (core OpenCL, no cl_khr_fp16 needed).
//
// Dispatch: global = [ceil(n/4)*64, m, 1], local = [64, 1, 1]

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Detect subgroup support at compile time
#if defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define HAS_SUBGROUPS 1
#elif defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define HAS_SUBGROUPS 1
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE
#endif

#define N_DST 4
#define WG_SIZE 64

REQD_SUBGROUP_SIZE
kernel void kernel_mul_mat_f16_f32(
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
    global half * x = (global half *)((global char*)src0 + offset0);
    global float * y = (global float *)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = r0 * N_DST;

#ifdef HAS_SUBGROUPS
    int lid = get_sub_group_local_id();
    int stride = get_max_sub_group_size();
#else
    int lid = get_local_id(0);
    int stride = WG_SIZE;
#endif

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong x_base = (ulong)first_row * ne00
                  + (ulong)(i12/r2) * ((ulong)ne00 * ne01)
                  + (ulong)(i13/r3) * ((ulong)ne00 * ne01 * ne02);

    global half * xr0 = x + x_base;
    global half * xr1 = xr0 + ne00;
    global half * xr2 = xr1 + ne00;
    global half * xr3 = xr2 + ne00;
    global float * yr = y + r1 * ne10 + im * ne00 * ne1;

    bool v0 = (first_row     < ne01);
    bool v1 = (first_row + 1 < ne01);
    bool v2 = (first_row + 2 < ne01);
    bool v3 = (first_row + 3 < ne01);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    if (ne00 >= 128) {
        // Vectorized: vload_half4 (core OpenCL, no cl_khr_fp16 needed)
        global float4 * y4 = (global float4 *) yr;
        int ne00_4 = ne00 / 4;

        for (int i = lid; i < ne00_4; i += stride) {
            float4 a = y4[i];
            if (v0) { float4 w = vload_half4(0, xr0 + i*4); s0 += dot(w, a); }
            if (v1) { float4 w = vload_half4(0, xr1 + i*4); s1 += dot(w, a); }
            if (v2) { float4 w = vload_half4(0, xr2 + i*4); s2 += dot(w, a); }
            if (v3) { float4 w = vload_half4(0, xr3 + i*4); s3 += dot(w, a); }
        }
    } else {
        for (int i = lid; i < ne00; i += stride) {
            float yval = yr[i];
            if (v0) s0 += vload_half(i, xr0) * yval;
            if (v1) s1 += vload_half(i, xr1) * yval;
            if (v2) s2 += vload_half(i, xr2) * yval;
            if (v3) s3 += vload_half(i, xr3) * yval;
        }
    }

    // Reduction
#ifdef HAS_SUBGROUPS
    // Zero-barrier subgroup reduce
    s0 = sub_group_reduce_add(s0);
    s1 = sub_group_reduce_add(s1);
    s2 = sub_group_reduce_add(s2);
    s3 = sub_group_reduce_add(s3);

    if (lid == 0) {
        // Scalar tail (only when ne00 not multiple of 4)
        if (ne00 >= 128) {
            for (int i = 4*(ne00/4); i < ne00; i++) {
                float yval = yr[i];
                if (v0) s0 += vload_half(i, xr0) * yval;
                if (v1) s1 += vload_half(i, xr1) * yval;
                if (v2) s2 += vload_half(i, xr2) * yval;
                if (v3) s3 += vload_half(i, xr3) * yval;
            }
        }
        int out_base = r1 * ne0 + im * ne0 * ne1;
        if (v0) dst[out_base + first_row]     = s0;
        if (v1) dst[out_base + first_row + 1] = s1;
        if (v2) dst[out_base + first_row + 2] = s2;
        if (v3) dst[out_base + first_row + 3] = s3;
    }
#else
    // Tree reduction fallback
    local float4 scratch[WG_SIZE];
    scratch[lid] = (float4)(s0, s1, s2, s3);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        int out_base = r1 * ne0 + im * ne0 * ne1;
        float4 result = scratch[0];
        if (v0) dst[out_base + first_row]     = result.s0;
        if (v1) dst[out_base + first_row + 1] = result.s1;
        if (v2) dst[out_base + first_row + 2] = result.s2;
        if (v3) dst[out_base + first_row + 3] = result.s3;
    }
#endif
}
