#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#elif defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE
#endif

// Adreno-optimized F16 GEMV for large-N matmuls (lm_head, FFN gate/up).
//
// N_DST=4 variant of mul_mv_f16_f32.cl: each WG produces 4 output rows instead
// of 2, reusing activation reads across all 4 rows for better bandwidth use.
// Halves the WG count for large vocab/hidden matmuls.
//
// Layout: local_size = [64, 4, 1]  (4 waves × 64 lanes each = 256 work-items)
//   - 64 lanes in one wave cooperate on the SAME 4 rows (K-parallel, stride 256)
//   - 4 waves split K: wave w handles K[w*K/4 .. (w+1)*K/4)
//   - Intra-wave reduction: sub_group_reduce_add (zero barriers)
//   - Inter-wave reduction: local-memory accumulation into wave 0
//
// Dispatch: global = [ceil(n/4)*64, m*4, 1], local = [64, 4, 1]
//   See mul_mv_f16_f32.cl for dispatch-geometry notes.
#define N_DST 4
#define N_SIMDGROUP 4
#define N_SIMDWIDTH 64

__attribute__((reqd_work_group_size(64, 4, 1)))
REQD_SUBGROUP_SIZE
kernel void kernel_mul_mat_f16_f32_l4(
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

    int r0 = get_group_id(0);   // output row-quad group
    int r1 = get_group_id(1);   // batch m
    int im = get_group_id(2);   // multi-head batch

    int tid = get_local_id(0);            // K-parallel lane 0..63
    int wave_id = get_local_id(1);        // K-split wave 0..3

    int first_row = r0 * N_DST;           // 4 rows per WG

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong x_offset = (ulong)first_row * ne00
                    + (ulong)(i12/r2) * ((ulong)ne00 * ne01)
                    + (ulong)(i13/r3) * ((ulong)ne00 * ne01 * ne02);

    global half * xr0 = x + x_offset;
    global half * xr1 = xr0 + ne00;
    global half * xr2 = xr1 + ne00;
    global half * xr3 = xr2 + ne00;

    global float * yr = y + r1 * ne10 + im * ne00 * ne1;

    bool row0_valid = (first_row     < ne01);
    bool row1_valid = (first_row + 1 < ne01);
    bool row2_valid = (first_row + 2 < ne01);
    bool row3_valid = (first_row + 3 < ne01);

    int k_per_wave = ((ne00 + N_SIMDGROUP - 1) / N_SIMDGROUP + 3) & ~3;
    int k_start = wave_id * k_per_wave;
    int k_end = k_start + k_per_wave;
    if (k_end > ne00) k_end = ne00;
    if (k_start > ne00) k_start = ne00;

    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    float sumf2 = 0.0f;
    float sumf3 = 0.0f;

    int vec4_end = k_start + ((k_end - k_start) / (N_SIMDWIDTH * 4)) * (N_SIMDWIDTH * 4);

    for (int i = k_start + tid * 4; i < vec4_end; i += N_SIMDWIDTH * 4) {
        float4 act = vload4(0, yr + i);
        if (row0_valid) sumf0 += dot(vload_half4(0, xr0 + i), act);
        if (row1_valid) sumf1 += dot(vload_half4(0, xr1 + i), act);
        if (row2_valid) sumf2 += dot(vload_half4(0, xr2 + i), act);
        if (row3_valid) sumf3 += dot(vload_half4(0, xr3 + i), act);
    }

    for (int i = vec4_end + tid; i < k_end; i += N_SIMDWIDTH) {
        float yval = yr[i];
        if (row0_valid) sumf0 += vload_half(i, xr0) * yval;
        if (row1_valid) sumf1 += vload_half(i, xr1) * yval;
        if (row2_valid) sumf2 += vload_half(i, xr2) * yval;
        if (row3_valid) sumf3 += vload_half(i, xr3) * yval;
    }

    sumf0 = sub_group_reduce_add(sumf0);
    sumf1 = sub_group_reduce_add(sumf1);
    sumf2 = sub_group_reduce_add(sumf2);
    sumf3 = sub_group_reduce_add(sumf3);

    local float4 wave_partials[N_SIMDGROUP];
    if (tid == 0) {
        wave_partials[wave_id] = (float4)(sumf0, sumf1, sumf2, sumf3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (wave_id == 0 && tid == 0) {
        float4 total = wave_partials[0]
                     + wave_partials[1]
                     + wave_partials[2]
                     + wave_partials[3];

        int out_base = r1 * ne0 + im * ne0 * ne1;
        if (row0_valid) dst[out_base + first_row]     = total.s0;
        if (row1_valid) dst[out_base + first_row + 1] = total.s1;
        if (row2_valid) dst[out_base + first_row + 2] = total.s2;
        if (row3_valid) dst[out_base + first_row + 3] = total.s3;
    }
}
