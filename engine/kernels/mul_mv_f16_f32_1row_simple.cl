// SPDX-License-Identifier: MIT
// Portions adapted from llama.cpp / ggml (https://github.com/ggml-org/llama.cpp), MIT licensed.
// See THIRD-PARTY-LICENSES.md.
// F16 GEMV single-token decode variant (N_DST=1, 64 threads/WG).
// Uses the same argument signature as mul_mv_f16_f32.cl so dispatch code
// can swap in this kernel via kernel handle without changing args.
//
// When activation batch m=1 (single-token decode), this kernel produces
// one output row per WG using a single subgroup of 64 lanes. No local
// memory, no inter-wave reduction — pure sub_group_reduce_add.
//
// Dispatch: global = [n * 64, 1, 1], local = [64, 1, 1]
//   get_group_id(0) = output row r0 ∈ [0, n)
//   get_local_id(0) = subgroup lane 0..63
//
// This matches llama.cpp's `kernel_mul_mat_f16_f32_1row` decode path
// (selected when ne11*ne12 < 4), but reuses argus_engine's existing 15-arg
// dispatch signature (no nb strides, assumes row-major contiguous).

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

REQD_SUBGROUP_SIZE
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_mul_mat_f16_f32_1row(
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
    global half  * x = (global half  *)((global char*)src0 + offset0);
    global float * y = (global float *)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);   // output row 0..n-1
    int r1 = get_group_id(1);   // batch m (=0 for single-token decode)
    int im = get_group_id(2);   // multi-head batch (=0)

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong x_offset = (ulong)r0 * ne00
                    + (ulong)(i12/r2) * ((ulong)ne00 * ne01)
                    + (ulong)(i13/r3) * ((ulong)ne00 * ne01 * ne02);

    global half  * xr = x + x_offset;
    global float * yr = y + r1 * ne10 + im * ne00 * ne1;

    int lid = get_sub_group_local_id();
    int stride = get_max_sub_group_size();

    float sumf = 0.0f;

    if (ne00 >= 128) {
        // Vectorized half4 × float4
        // NOTE: on Adreno 830, `vload_half4 + dot()` outperformed
        // `(half4*)` cast + 4 scalar MADs (ablation 2026-04-22, +12 ms
        // regression with scalar MAD). The `dot()` reduction compiles to
        // Adreno's FDP4 single-cycle instruction while scalar MAD chain
        // does not pipeline as efficiently. Do NOT "optimize" to llama.cpp
        // style without re-verifying on hardware.
        int ne00_4 = ne00 / 4;
        for (int i = lid; i < ne00_4; i += stride) {
            float4 a = vload4(0, yr + i * 4);
            float4 w = vload_half4(0, xr + i * 4);
            sumf += dot(w, a);
        }
        float all_sum = sub_group_reduce_add(sumf);
        if (lid == 0) {
            // Scalar tail for non-multiple-of-4 K
            for (int i = 4 * ne00_4; i < ne00; i++) {
                all_sum += (float)xr[i] * yr[i];
            }
            dst[im * ne1 * ne0 + r1 * ne0 + r0] = all_sum;
        }
    } else {
        for (int i = lid; i < ne00; i += stride) {
            sumf += (float)xr[i] * yr[i];
        }
        float all_sum = sub_group_reduce_add(sumf);
        if (lid == 0) {
            dst[im * ne1 * ne0 + r1 * ne0 + r0] = all_sum;
        }
    }
}
