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

// Adreno-optimized F16 GEMV — 4-wave K-split with per-row cooperation.
//
// Layout: local_size = [64, 4, 1]  (4 waves × 64 lanes each = 256 work-items)
//   - A WG produces N_DST=2 output rows (row = r0*2 .. r0*2+1)
//   - 64 lanes in one wave cooperate on the SAME row-pair, parallelizing the
//     K dimension with stride 256 = 64*4. Each lane loads 4 halves at a time.
//   - 4 waves further split K: wave w handles K[w*K/4 .. (w+1)*K/4)
//   - Intra-wave reduction: sub_group_reduce_add() across 64 lanes (zero barriers)
//   - Inter-wave reduction: local memory accumulation into wave 0
//
// Total K parallelism per row = 64 lanes × 4 waves = 256-way.
//
// Dispatch: global = [ceil(n/2)*64, m*4, 1], local = [64, 4, 1]
//   get_group_id(0) ∈ [0, ceil(n/2))  — output row-pair index
//   get_group_id(1) = batch m         — since global_y/local_y = m
//   get_local_id(0) = lane 0..63      — K-parallel lane within a wave
//   get_local_id(1) = wave 0..3       — K-split partition
//
// Subgroup contract: one subgroup per wave (64 lanes aligned with dim 0).
// On Adreno with `qcom_reqd_sub_group_size("half")` this maps to the 64-lane
// fiber width. The subgroup size is communicated to hardware via the attribute;
// `sub_group_reduce_add` operates on whichever subgroup size is selected.
#define N_DST 2
#define N_SIMDGROUP 4
#define N_SIMDWIDTH 64

__attribute__((reqd_work_group_size(64, 4, 1)))
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

    int r0 = get_group_id(0);   // output row-pair group
    int r1 = get_group_id(1);   // batch m
    int im = get_group_id(2);   // multi-head batch

    int tid = get_local_id(0);            // K-parallel lane 0..63
    int wave_id = get_local_id(1);        // K-split wave 0..3

    int first_row = r0 * N_DST;

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong x_offset = (ulong)first_row * ne00
                    + (ulong)(i12/r2) * ((ulong)ne00 * ne01)
                    + (ulong)(i13/r3) * ((ulong)ne00 * ne01 * ne02);

    global half * xr0 = x + x_offset;
    global half * xr1 = xr0 + ne00;

    global float * yr = y + r1 * ne10 + im * ne00 * ne1;

    bool row0_valid = (first_row     < ne01);
    bool row1_valid = (first_row + 1 < ne01);

    // K-split: each wave handles a contiguous chunk of K, aligned to 4.
    int k_per_wave = ((ne00 + N_SIMDGROUP - 1) / N_SIMDGROUP + 3) & ~3;
    int k_start = wave_id * k_per_wave;
    int k_end = k_start + k_per_wave;
    if (k_end > ne00) k_end = ne00;
    if (k_start > ne00) k_start = ne00;

    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    // Main loop: 64 lanes stride the wave's K chunk by 64*4 = 256 elements.
    int vec4_end = k_start + ((k_end - k_start) / (N_SIMDWIDTH * 4)) * (N_SIMDWIDTH * 4);

    for (int i = k_start + tid * 4; i < vec4_end; i += N_SIMDWIDTH * 4) {
        float4 act = vload4(0, yr + i);
        if (row0_valid) sumf0 += dot(vload_half4(0, xr0 + i), act);
        if (row1_valid) sumf1 += dot(vload_half4(0, xr1 + i), act);
    }

    // Scalar tail: leftover < 256 elements at the end of the wave's K chunk.
    for (int i = vec4_end + tid; i < k_end; i += N_SIMDWIDTH) {
        float yval = yr[i];
        if (row0_valid) sumf0 += vload_half(i, xr0) * yval;
        if (row1_valid) sumf1 += vload_half(i, xr1) * yval;
    }

    // -- Intra-wave reduction: 64 → 1 via subgroup reduce (zero barriers) --
    sumf0 = sub_group_reduce_add(sumf0);
    sumf1 = sub_group_reduce_add(sumf1);

    // -- Inter-wave reduction: wave 0 accumulates partial K-chunk sums --
    // Each wave's lane 0 now holds its full wave partial. Publish to local mem,
    // barrier, and have wave 0 lane 0 sum the 4 waves.
    local float2 wave_partials[N_SIMDGROUP];  // one per wave
    if (tid == 0) {
        wave_partials[wave_id] = (float2)(sumf0, sumf1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (wave_id == 0 && tid == 0) {
        float2 total = wave_partials[0]
                     + wave_partials[1]
                     + wave_partials[2]
                     + wave_partials[3];

        int out_base = r1 * ne0 + im * ne0 * ne1;
        if (row0_valid) dst[out_base + first_row]     = total.s0;
        if (row1_valid) dst[out_base + first_row + 1] = total.s1;
    }
}
