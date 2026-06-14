// Optimized OpenCL kernels for argus_engine using local memory reductions
// Based on llama.cpp kernel patterns

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// Optimized RMS Norm with local memory reduction (following llama.cpp pattern)
// Input x: [rows, dim], weight: [dim], output: x (inplace)
// Each workgroup handles one row, uses local memory for reduction
//------------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_opt(
    global float * x,
    global float * weight,
    int dim,
    float eps,
    int add_unit,           // Gemma3 style: weight = 1 + weight when nonzero
    local float * scratch   // local memory for reduction
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float * row_ptr = x + row * dim;

    // Phase 1: Parallel sum of squares
    float sum_sq = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float val = row_ptr[i];
        sum_sq += val * val;
    }

    // Subgroup reduction first
    sum_sq = sub_group_reduce_add(sum_sq);

    // Store subgroup results to local memory
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Final reduction in first subgroup
    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }

    // Broadcast final result
    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];

    // Phase 2: Apply normalization
    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;

    for (int i = lid; i < dim; i += local_size) {
        float w = add_unit ? (1.0f + weight[i]) : weight[i];
        row_ptr[i] = row_ptr[i] * scale * w;
    }
}

// Vectorized RMSNorm (float4) — in-place. Requires dim % 4 == 0.
// Uses dot() + float4 loads for ~4x less loop iterations. Pattern mirrors
// llama.cpp kernel_rms_norm_mul. Each workgroup handles one row.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_opt_f4(
    global float * x,
    global float * weight,
    int dim,                // must be multiple of 4
    float eps,
    int add_unit,
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float4 * x_v = (global float4 *)(x + row * dim);
    global float4 * w_v = (global float4 *)weight;
    int dim_v = dim >> 2;

    // Phase 1: parallel sum of squares via dot()
    float sum_sq = 0.0f;
    for (int i = lid; i < dim_v; i += local_size) {
        float4 v = x_v[i];
        sum_sq += dot(v, v);
    }

    sum_sq = sub_group_reduce_add(sum_sq);
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }
    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];

    // Phase 2: apply normalization vectorized
    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;
    float4 ones = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

    for (int i = lid; i < dim_v; i += local_size) {
        float4 w = w_v[i];
        if (add_unit) w = w + ones;
        x_v[i] = x_v[i] * scale * w;
    }
}

// Fused add + out-of-place RMSNorm: x += residual; out = norm(x) * weight
// Eliminates separate add_assign kernel before rms_norm_oop.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_add_rms_norm_oop(
    global float * x,
    global float * residual,
    global float * out,
    global float * weight,
    int dim,
    float eps,
    int add_unit,           // Gemma3 style: weight = 1 + weight when nonzero
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float * x_row = x + row * dim;
    global float * res_row = residual + row * dim;
    global float * out_row = out + row * dim;

    // Phase 1: x += residual, compute sum of squares
    float sum_sq = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float val = x_row[i] + res_row[i];
        x_row[i] = val;
        sum_sq += val * val;
    }

    sum_sq = sub_group_reduce_add(sum_sq);
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }

    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];

    // Phase 2: normalize and write output
    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;

    for (int i = lid; i < dim; i += local_size) {
        float w = add_unit ? (1.0f + weight[i]) : weight[i];
        out_row[i] = x_row[i] * scale * w;
    }
}

// Vectorized fused add + RMSNorm (float4). dim must be multiple of 4.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_add_rms_norm_oop_f4(
    global float * x,
    global float * residual,
    global float * out,
    global float * weight,
    int dim,
    float eps,
    int add_unit,
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float4 * x_v = (global float4 *)(x + row * dim);
    global float4 * res_v = (global float4 *)(residual + row * dim);
    global float4 * out_v = (global float4 *)(out + row * dim);
    global float4 * w_v = (global float4 *)weight;
    int dim_v = dim >> 2;

    // Phase 1: x += residual, sum of squares via dot()
    float sum_sq = 0.0f;
    for (int i = lid; i < dim_v; i += local_size) {
        float4 v = x_v[i] + res_v[i];
        x_v[i] = v;
        sum_sq += dot(v, v);
    }

    sum_sq = sub_group_reduce_add(sum_sq);
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }
    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];

    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;
    float4 ones = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

    for (int i = lid; i < dim_v; i += local_size) {
        float4 w = w_v[i];
        if (add_unit) w = w + ones;
        out_v[i] = x_v[i] * scale * w;
    }
}

// Variant of kernel_add_rms_norm_oop_f4 that signals completion by writing
// 1 to a host-visible flag (CL_MEM_ALLOC_HOST_PTR). Used by tensor-partition
// plan path to let the CPU spin-poll instead of blocking on
// `clFinish(queue)` — shaves Adreno driver submission/completion latency
// (~0.3 ms/layer) off the partition critical path.
//
// Correctness requirements:
//   - Single work-group (row count == 1, i.e. decode m=1).
//   - `ready_flag` points to a 4-byte ALLOC_HOST_PTR buffer initialized to 0
//     by the CPU before the kernel is enqueued.
//   - `barrier(CLK_GLOBAL_MEM_FENCE)` + `atomic_xchg` establishes a release
//     edge: the CPU, after observing flag==1 via an acquire fence, is
//     guaranteed to see the preceding stores to `out`.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_add_rms_norm_oop_f4_sigflag(
    global float * x,
    global float * residual,
    global float * out,
    global float * weight,
    int dim,
    float eps,
    int add_unit,
    global volatile int * ready_flag,
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float4 * x_v = (global float4 *)(x + row * dim);
    global float4 * res_v = (global float4 *)(residual + row * dim);
    global float4 * out_v = (global float4 *)(out + row * dim);
    global float4 * w_v = (global float4 *)weight;
    int dim_v = dim >> 2;

    float sum_sq = 0.0f;
    for (int i = lid; i < dim_v; i += local_size) {
        float4 v = x_v[i] + res_v[i];
        x_v[i] = v;
        sum_sq += dot(v, v);
    }

    sum_sq = sub_group_reduce_add(sum_sq);
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }
    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];

    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;
    float4 ones = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

    for (int i = lid; i < dim_v; i += local_size) {
        float4 w = w_v[i];
        if (add_unit) w = w + ones;
        out_v[i] = x_v[i] * scale * w;
    }

    // Release edge: wait for all work-items in the WG to finish writing
    // `out`, then publish the ready flag.
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (lid == 0) {
        atomic_xchg(ready_flag, 1);
    }
}

// Fused 3-input merge + out-of-place RMSNorm for tensor-partition layer boundary.
//
// Replaces the decode-path layer merge (copy_slice gpu->down, copy_slice cpu->staging,
// add_assign(down,staging), residual+=down) + next-layer RMSNorm with a single kernel.
// Computes:
//     sum[i]          = prior_residual[i] + gpu_partial[i] + cpu_staging[i]
//     residual_out[i] = sum[i]                          (updated residual for next layer)
//     out[i]          = (sum[i] / rms(sum)) * w_eff[i]  (pre-attention norm output)
// where w_eff = (1 + norm_weight) when add_unit != 0 (Gemma3 convention) else norm_weight.
//
// Buffer aliasing: `residual_out` MAY alias `prior_residual` (same cl_mem) for an in-place
// residual update — the kernel reads every input once before writing `residual_out`, so the
// overlap is safe. `out` MUST NOT alias any input. The other three inputs MUST be distinct.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_fused_norm_merge(
    global float * prior_residual,
    global float * gpu_partial,
    global float * cpu_staging,
    global float * norm_weight,
    global float * out,
    global float * residual_out,
    int dim,
    float eps,
    int add_unit,           // Gemma3 style: weight = 1 + weight when nonzero
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float * prior_row = prior_residual + row * dim;
    global float * gpu_row = gpu_partial + row * dim;
    global float * cpu_row = cpu_staging + row * dim;
    global float * out_row = out + row * dim;
    global float * res_out_row = residual_out + row * dim;

    // Phase 1: sum = prior + (gpu + cpu); residual_out = sum; accumulate sum_sq
    //
    // Reduction order MUST match the baseline 3-step path so generated
    // tokens stay bit-exact with fused off:
    //   baseline:  ws.down = gpu + cpu;  x = prior + ws.down
    // Evaluating as `prior + gpu + cpu` (left-to-right) would reduce as
    // `(prior + gpu) + cpu` and drift one-ULP per element.
    float sum_sq = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float partial = gpu_row[i] + cpu_row[i];
        float val = prior_row[i] + partial;
        res_out_row[i] = val;
        sum_sq += val * val;
    }

    sum_sq = sub_group_reduce_add(sum_sq);
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }

    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];

    // Phase 2: normalize & apply weight (reads residual_out to support in-place prior_residual aliasing)
    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;

    for (int i = lid; i < dim; i += local_size) {
        float w = add_unit ? (1.0f + norm_weight[i]) : norm_weight[i];
        out_row[i] = res_out_row[i] * scale * w;
    }
}

// Vectorized fused norm-merge (float4). dim must be multiple of 4.
// Same semantics as kernel_fused_norm_merge; see notes above for aliasing rules.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_fused_norm_merge_f4(
    global float * prior_residual,
    global float * gpu_partial,
    global float * cpu_staging,
    global float * norm_weight,
    global float * out,
    global float * residual_out,
    int dim,
    float eps,
    int add_unit,
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float4 * prior_v = (global float4 *)(prior_residual + row * dim);
    global float4 * gpu_v = (global float4 *)(gpu_partial + row * dim);
    global float4 * cpu_v = (global float4 *)(cpu_staging + row * dim);
    global float4 * w_v = (global float4 *)norm_weight;
    global float4 * out_v = (global float4 *)(out + row * dim);
    global float4 * res_out_v = (global float4 *)(residual_out + row * dim);
    int dim_v = dim >> 2;

    // Phase 1 — see scalar variant for reduction-order rationale.
    // Order: v = prior + (gpu + cpu), matching baseline `x += (ws.down = gpu + cpu)`.
    float sum_sq = 0.0f;
    for (int i = lid; i < dim_v; i += local_size) {
        float4 partial = gpu_v[i] + cpu_v[i];
        float4 v = prior_v[i] + partial;
        res_out_v[i] = v;
        sum_sq += dot(v, v);
    }

    sum_sq = sub_group_reduce_add(sum_sq);
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }
    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];

    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;
    float4 ones = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

    for (int i = lid; i < dim_v; i += local_size) {
        float4 w = w_v[i];
        if (add_unit) w = w + ones;
        out_v[i] = res_out_v[i] * scale * w;
    }
}

// Out-of-place variant: reads from x, writes to out (x is preserved).
// Used to eliminate copy_residual in the forward pass.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_oop(
    global float * x,
    global float * out,
    global float * weight,
    int dim,
    float eps,
    int add_unit,           // Gemma3 style: weight = 1 + weight when nonzero
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float * x_row = x + row * dim;
    global float * out_row = out + row * dim;

    float sum_sq = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    sum_sq = sub_group_reduce_add(sum_sq);
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }

    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];

    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;

    for (int i = lid; i < dim; i += local_size) {
        float w = add_unit ? (1.0f + weight[i]) : weight[i];
        out_row[i] = x_row[i] * scale * w;
    }
}

// Vectorized out-of-place RMSNorm (float4). dim must be multiple of 4.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_oop_f4(
    global float * x,
    global float * out,
    global float * weight,
    int dim,
    float eps,
    int add_unit,
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float4 * x_v = (global float4 *)(x + row * dim);
    global float4 * out_v = (global float4 *)(out + row * dim);
    global float4 * w_v = (global float4 *)weight;
    int dim_v = dim >> 2;

    float sum_sq = 0.0f;
    for (int i = lid; i < dim_v; i += local_size) {
        float4 v = x_v[i];
        sum_sq += dot(v, v);
    }

    sum_sq = sub_group_reduce_add(sum_sq);
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }
    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];

    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;
    float4 ones = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

    for (int i = lid; i < dim_v; i += local_size) {
        float4 w = w_v[i];
        if (add_unit) w = w + ones;
        out_v[i] = x_v[i] * scale * w;
    }
}

//------------------------------------------------------------------------------
// Optimized Softmax with local memory reduction
// Input x: [rows, dim], output: x (inplace)
// Each workgroup handles one row
//------------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_softmax_opt(
    global float * x,
    int dim,
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    global float * row_ptr = x + row * dim;
    
    // Phase 1: Find max
    float lmax = -INFINITY;
    for (int i = lid; i < dim; i += local_size) {
        lmax = fmax(lmax, row_ptr[i]);
    }
    
    // Subgroup reduction for max
    lmax = sub_group_reduce_max(lmax);
    
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = lmax;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Final max reduction
    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : -INFINITY;
        lmax = sub_group_reduce_max(val);
    }
    
    if (lid == 0) {
        scratch[0] = lmax;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float max_val = scratch[0];
    
    // Phase 2: Compute exp and sum
    float lsum = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float exp_val = exp(row_ptr[i] - max_val);
        row_ptr[i] = exp_val;
        lsum += exp_val;
    }
    
    // Subgroup reduction for sum
    lsum = sub_group_reduce_add(lsum);
    
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = lsum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        lsum = sub_group_reduce_add(val);
    }
    
    if (lid == 0) {
        scratch[0] = lsum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum_exp = scratch[0];
    
    // Phase 3: Normalize
    for (int i = lid; i < dim; i += local_size) {
        row_ptr[i] /= sum_exp;
    }
}

//------------------------------------------------------------------------------
// Optimized RoPE inplace using neox-style layout
// x: [batch * seq * num_heads * head_dim]
// This kernel processes pairs (x[i], x[i + head_dim/2])
//------------------------------------------------------------------------------
kernel void kernel_rope_opt(
    global float * x,
    int head_dim,
    int num_heads,
    int seq_len,
    int start_pos,
    float theta_base
) {
    int gid = get_global_id(0);
    int half_dim = head_dim / 2;
    
    // Calculate position within the tensor
    int pair_idx = gid % half_dim;            // which pair within head_dim
    int head_idx = (gid / half_dim) % num_heads;
    int seq_idx = (gid / (half_dim * num_heads)) % seq_len;
    int batch_idx = gid / (half_dim * num_heads * seq_len);
    
    int pos = start_pos + seq_idx;
    
    // Calculate frequency for this pair
    float freq = 1.0f / pow(theta_base, (float)(pair_idx * 2) / (float)head_dim);
    float angle = (float)pos * freq;
    
    float cos_val = cos(angle);
    float sin_val = sin(angle);
    
    // Calculate indices into the tensor (neox layout)
    int base_offset = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim;
    
    int i0 = base_offset + pair_idx;
    int i1 = base_offset + pair_idx + half_dim;
    
    float x0 = x[i0];
    float x1 = x[i1];
    
    x[i0] = x0 * cos_val - x1 * sin_val;
    x[i1] = x0 * sin_val + x1 * cos_val;
}

//------------------------------------------------------------------------------
// Scale: x[i] *= scale (vectorized float4)
//------------------------------------------------------------------------------
kernel void kernel_scale_opt(
    global float4 * x,
    float scale,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        x[i] *= scale;
    }
}

//------------------------------------------------------------------------------
// Add assign: x[i] += y[i] (vectorized float4)
//------------------------------------------------------------------------------
kernel void kernel_add_assign_opt(
    global float4 * x,
    global float4 * y,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        x[i] += y[i];
    }
}

//------------------------------------------------------------------------------
// SiLU * mul: x[i] = silu(x[i]) * y[i] (vectorized float4)
//------------------------------------------------------------------------------
kernel void kernel_silu_mul_opt(
    global float4 * x,
    global float4 * y,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        float4 val = x[i];
        float4 sigmoid = 1.0f / (1.0f + exp(-val));
        x[i] = val * sigmoid * y[i];
    }
}

//------------------------------------------------------------------------------
// Simple fallback versions (no subgroup operations)
//------------------------------------------------------------------------------

// Simple RMS Norm - one thread per row (for small dims or fallback)
kernel void kernel_rms_norm_simple(
    global float * x,
    global float * weight,
    global float * output,
    int dim,
    float eps
) {
    int row = get_global_id(0);

    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = x[row * dim + i];
        sum_sq += val * val;
    }

    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;

    for (int i = 0; i < dim; i++) {
        output[row * dim + i] = x[row * dim + i] * scale * weight[i];
    }
}

// D-D.6 RMS norm — OOP single-subgroup variant.
// Production `kernel_rms_norm_opt` (in-place, parallel reduction with __local
// scratch)와 numerical-equivalent reduction order. SDK가 LOCAL arg를 OpPackage
// path에서 reject하므로 (graphFinalize err=0x1786 — see rms_norm op doc),
// workgroup_size를 single subgroup (64 threads)로 강제하여 cross-subgroup
// reduction 제거 → __local scratch 불필요.
//
// Production opt의 default workgroup이 64 + REQD_SUBGROUP_SIZE_64이면 multi-
// subgroup count = 1이라 cross-subgroup phase는 single-value passthrough →
// 본 single-subgroup 변종과 byte-equal numerical 결과.
//
// Workgroup contract (caller responsibility):
//   global = [rows * 64, 1, 1]
//   local  = [64, 1, 1]
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_oop_subgroup(
    global float * x,
    global float * weight,
    global float * output,
    int dim,
    float eps
) {
    int row = get_group_id(0);
    int lid = get_sub_group_local_id();
    int sg_size = get_max_sub_group_size();  // = 64 on Adreno (REQD_SUBGROUP_SIZE_64)

    global float * row_in = x + row * dim;
    global float * row_out = output + row * dim;

    // Phase 1: parallel partial sum-of-squares (sg_size threads).
    float sum_sq = 0.0f;
    for (int i = lid; i < dim; i += sg_size) {
        float val = row_in[i];
        sum_sq += val * val;
    }

    // Phase 2: subgroup reduction (no SLM, hardware intrinsic).
    sum_sq = sub_group_reduce_add(sum_sq);
    // sub_group_reduce_add returns the same value for all subgroup lanes,
    // so no broadcast is needed.

    // Phase 3: scale + weight in parallel.
    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;
    for (int i = lid; i < dim; i += sg_size) {
        row_out[i] = row_in[i] * scale * weight[i];
    }
}

kernel void kernel_softmax_simple(
    global float * x,
    global float * output,
    int dim
) {
    int row = get_global_id(0);
    
    float max_val = -INFINITY;
    for (int i = 0; i < dim; i++) {
        max_val = fmax(max_val, x[row * dim + i]);
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < dim; i++) {
        float exp_val = exp(x[row * dim + i] - max_val);
        output[row * dim + i] = exp_val;
        sum_exp += exp_val;
    }
    
    for (int i = 0; i < dim; i++) {
        output[row * dim + i] /= sum_exp;
    }
}

kernel void kernel_rope_simple(
    global float * x,
    int head_dim,
    int num_heads,
    int seq_len,
    int start_pos,
    float theta
) {
    // x is [batch, seq, num_heads, head_dim]
    // Each work item processes one (i, i+head_dim/2) pair
    int gid = get_global_id(0);

    int half_dim = head_dim / 2;
    int elements_per_seq = num_heads * head_dim;
    int pairs_per_seq = num_heads * half_dim;

    // Decode gid into (seq_idx, head_idx, pair_idx)
    int seq_idx = gid / pairs_per_seq;
    int in_seq = gid % pairs_per_seq;
    int head_idx = in_seq / half_dim;
    int pair_idx = in_seq % half_dim;

    int pos = start_pos + seq_idx;

    // Calculate frequency for this pair
    float freq = 1.0f / pow(theta, (float)(pair_idx * 2) / (float)head_dim);
    float angle = (float)pos * freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Calculate indices into the tensor (neox layout: first half and second half of head)
    int base_offset = seq_idx * elements_per_seq + head_idx * head_dim;
    int i0 = base_offset + pair_idx;
    int i1 = base_offset + pair_idx + half_dim;

    float x0 = x[i0];
    float x1 = x[i1];

    x[i0] = x0 * cos_val - x1 * sin_val;
    x[i1] = x0 * sin_val + x1 * cos_val;
}

// Out-of-place variant of kernel_rope_simple: writes to x_out instead of
// mutating x_in. Each thread (= 1 pair) reads (x_in[i0], x_in[i1]) and writes
// the rotated values to (x_out[i0], x_out[i1]). Layout is identical, so
// every element of x_out is covered by exactly one thread (i0/i1 pair).
//
// Used by the QNN OpPackage `CustomRope` op (M2.H): the SDK's chain-composition
// path requires the graph-level output edge to be a distinct buffer from the
// input. The in-place variant works for stand-alone single-op execution but
// breaks when downstream ops consume the output via a separate allocation.
//
// M3.4 D-D.1: `start_pos` is now read from `pos_buf[0]` instead of being a
// SCALAR op param. Reason: QNN_PARAMTYPE_SCALAR is baked into the graph at
// finalize time, which prevents multi-token decode (every token would use the
// build-time pos value). Reading from a buffer makes pos a runtime input.
kernel void kernel_rope_simple_oop(
    global const float * x_in,
    global float * x_out,
    global const int * pos_buf,
    int head_dim,
    int num_heads,
    int seq_len,
    float theta
) {
    int gid = get_global_id(0);

    int half_dim = head_dim / 2;
    int elements_per_seq = num_heads * head_dim;
    int pairs_per_seq = num_heads * half_dim;

    int seq_idx = gid / pairs_per_seq;
    int in_seq = gid % pairs_per_seq;
    int head_idx = in_seq / half_dim;
    int pair_idx = in_seq % half_dim;

    int start_pos = pos_buf[0];
    int pos = start_pos + seq_idx;

    float freq = 1.0f / pow(theta, (float)(pair_idx * 2) / (float)head_dim);
    float angle = (float)pos * freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    int base_offset = seq_idx * elements_per_seq + head_idx * head_dim;
    int i0 = base_offset + pair_idx;
    int i1 = base_offset + pair_idx + half_dim;

    float x0 = x_in[i0];
    float x1 = x_in[i1];

    x_out[i0] = x0 * cos_val - x1 * sin_val;
    x_out[i1] = x0 * sin_val + x1 * cos_val;
}

kernel void kernel_scale_simple(
    global float * x,
    float scale,
    int size
) {
    int i = get_global_id(0);
    if (i < size) {
        x[i] *= scale;
    }
}

kernel void kernel_add_assign_simple(
    global float4 * x,
    global float4 * y,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        x[i] += y[i];
    }
}

// Tensor-partition fused merge + residual: x += gpu_partial + cpu_partial.
// Replaces 3 sequential kernels (copy_slice + add_assign_staging + post_ffn add_assign)
// with a single dispatch on partition layers. `gpu_partial` = down_partial_gpu
// output, `cpu_partial` = cpu_merge_staging ALLOC_HOST_PTR buffer written by CPU.
// Both are full `dim`-sized vectors (their respective share of rows reproject
// through down_proj back into dim dimensions). gws: [dim/4, 1, 1].
kernel void kernel_partition_fused_merge_residual_f4(
    global float4 * x,
    global const float4 * gpu_partial,
    global const float4 * cpu_partial,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        x[i] += gpu_partial[i] + cpu_partial[i];
    }
}

// Scalar-stride copy of `size` floats from src[src_offset ..] to dst[dst_offset ..].
// Used by the tensor-partition plan merge step (arch A.7.1) — the runtime `copy_slice`
// backend method bypasses the kernel plan (it uses clEnqueueCopyBuffer), so the plan
// path needs a GPU kernel with equivalent semantics to stay fully enqueue-driven.
// gws: [size, 1, 1]; lws: None (any power-of-two subgroup works).
kernel void kernel_copy_slice_simple(
    global const float * src,
    global float * dst,
    int src_offset,
    int dst_offset,
    int size
) {
    int i = get_global_id(0);
    if (i < size) {
        dst[dst_offset + i] = src[src_offset + i];
    }
}

// Broadcast-add a 1D bias to each row of x.
// x: [rows * dim], bias: [dim]. bias is added to each row.
kernel void kernel_add_row_bias(
    global float * x,
    global const float * bias,
    int dim,
    int total_elements
) {
    int gid = get_global_id(0);
    if (gid < total_elements) {
        x[gid] += bias[gid % dim];
    }
}

// D-D.6 Phase B: out-of-place broadcast-add bias to each row of x.
// y[i] = x[i] + bias[i % dim]. graph 14-node chain compatible (OOP buffers).
kernel void kernel_add_row_bias_oop(
    global const float * x,
    global const float * bias,
    global float * y,
    int dim,
    int total_elements
) {
    int gid = get_global_id(0);
    if (gid < total_elements) {
        y[gid] = x[gid] + bias[gid % dim];
    }
}

kernel void kernel_silu_mul_simple(
    global float4 * x,
    global float4 * y,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        float4 val = x[i];
        float4 sigmoid = 1.0f / (1.0f + exp(-val));
        x[i] = val * sigmoid * y[i];
    }
}

// Out-of-place variant of kernel_silu_mul_simple: out[i] = silu(x[i]) * y[i].
// Used by the QNN OpPackage `CustomSiluMul` op (M2.H): the SDK requires the
// graph-level output edge to be a distinct buffer from inputs for chain
// composition.
kernel void kernel_silu_mul_simple_oop(
    global const float4 * x,
    global const float4 * y,
    global float4 * out,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        float4 val = x[i];
        float4 sigmoid = 1.0f / (1.0f + exp(-val));
        out[i] = val * sigmoid * y[i];
    }
}

//------------------------------------------------------------------------------
// GELU tanh approximation * mul: x[i] = gelu_tanh(x[i]) * y[i] (vectorized float4)
// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//------------------------------------------------------------------------------
kernel void kernel_gelu_tanh_mul(
    global float4 * x,
    global float4 * y,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        float4 val = x[i];
        float4 inner = 0.7978845608f * (val + 0.044715f * val * val * val);
        float4 gelu = 0.5f * val * (1.0f + tanh(inner));
        x[i] = gelu * y[i];
    }
}

//------------------------------------------------------------------------------
// Optimized Single-Query Attention Kernel for Generation (GQA-aware)
// Q: [num_heads_q, head_dim] - single query token
// K: [cache_seq_len, num_heads_kv, head_dim] - key cache
// V: [cache_seq_len, num_heads_kv, head_dim] - value cache  
// O: [num_heads_q, head_dim] - output
// Each workgroup processes one query head
// 
// OPTIMIZATION: 2-pass approach
// Pass 1: Compute scores, find max, compute softmax sum (fused)
// Pass 2: Compute weighted V using stored weights
//------------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_attn_gen(
    global float * Q,            // [num_heads_q, head_dim]
    global float * K,            // SeqMajor: [seq, kv_heads, head_dim] or HeadMajor: [kv_heads, cap, head_dim]
    global float * V,            // same layout as K
    global float * O,            // [num_heads_q, head_dim]
    global float * S,            // score output [n_layers, num_heads_q, score_stride]
    int head_dim,
    int num_heads_q,
    int num_heads_kv,
    int cache_seq_len,
    float scale,
    int kv_pos_stride,           // stride between positions (SeqMajor: kv_heads*head_dim, HeadMajor: head_dim)
    int kv_head_stride,          // stride between heads (SeqMajor: head_dim, HeadMajor: cap*head_dim)
    int write_scores,            // 0=skip, 1=write
    int score_stride,            // stride between heads in S
    int score_layer_offset,      // base offset in S for this layer (= layer_idx * num_heads_q * score_stride)
    local float * scratch        // size = local_size
) {
    int head_idx = get_group_id(0);    // which Q head
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    // GQA: map Q head to KV head
    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;
    int kv_base = kv_head * kv_head_stride;

    // Pointers
    global float * q_ptr = Q + head_idx * head_dim;

    // === PASS 1: Compute max score using online approach ===
    float my_max = -INFINITY;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global float * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;
        my_max = fmax(my_max, score);
    }

    // Reduce max across workgroup
    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute exp sum with known max
    float my_sum = 0.0f;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global float * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        my_sum += exp(score * scale - max_score);
    }

    // Reduce sum
    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // === PASS 2: Compute weighted V sum ===
    // Each thread accumulates output for ALL head_dim elements for its subset of tokens
    // Then we reduce across threads

    // Initialize output accumulator per thread
    float out_local[256]; // Max head_dim across supported models (Gemma 3: 256, Qwen2: 128, Llama: 64)
    for (int d = 0; d < head_dim; d++) {
        out_local[d] = 0.0f;
    }

    // Each thread processes its subset of tokens
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global float * k_ptr = K + kv_base + t * kv_pos_stride;

        // Compute weight for this token
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        float weight = exp(score * scale - max_score) / total_sum;

        if (write_scores) {
            S[score_layer_offset + head_idx * score_stride + t] = weight;
        }

        // Accumulate V contribution
        global float * v_ptr = V + kv_base + t * kv_pos_stride;
        for (int d = 0; d < head_dim; d++) {
            out_local[d] += weight * v_ptr[d];
        }
    }

    // Reduce output across threads for each dimension
    // Use scratch for reduction (one dimension at a time)
    for (int d = 0; d < head_dim; d++) {
        scratch[lid] = out_local[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            O[head_idx * head_dim + d] = scratch[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// ============================================================
// F16 KV Cache Support Kernels
// ============================================================

// Convert F32 buffer to F16 buffer element-wise
kernel void kernel_cast_f32_to_f16(
    global float * src,
    global half * dst,
    int count
) {
    int gid = get_global_id(0);
    if (gid < count) {
        dst[gid] = (half)src[gid];
    }
}

// Attention generation kernel with F16 KV cache
// Q is F32, K/V cache is F16, output is F32
kernel void kernel_attn_gen_half(
    global float * Q,            // [num_heads_q, head_dim]
    global half * K,             // SeqMajor: [seq, kv_heads, head_dim] or HeadMajor: [kv_heads, cap, head_dim]
    global half * V,             // same layout as K
    global float * O,            // [num_heads_q, head_dim]
    global float * S,            // score output [n_layers, num_heads_q, score_stride]
    int head_dim,
    int num_heads_q,
    int num_heads_kv,
    int cache_seq_len,
    float scale,
    int kv_pos_stride,           // stride between positions (SeqMajor: kv_heads*head_dim, HeadMajor: head_dim)
    int kv_head_stride,          // stride between heads (SeqMajor: head_dim, HeadMajor: cap*head_dim)
    int write_scores,            // 0=skip, 1=write
    int score_stride,            // stride between heads in S
    int score_layer_offset,      // base offset in S for this layer (= layer_idx * num_heads_q * score_stride)
    local float * scratch        // size = local_size
) {
    int head_idx = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;
    int kv_base = kv_head * kv_head_stride;

    global float * q_ptr = Q + head_idx * head_dim;

    // === PASS 1: Compute max score ===
    float my_max = -INFINITY;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global half * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * vload_half(d, k_ptr);
        }
        score *= scale;
        my_max = fmax(my_max, score);
    }

    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute exp sum
    float my_sum = 0.0f;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global half * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * vload_half(d, k_ptr);
        }
        my_sum += exp(score * scale - max_score);
    }

    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // === PASS 2: Weighted V sum ===
    float out_local[256]; // Max head_dim across supported models (Gemma 3: 256, Qwen2: 128, Llama: 64)
    for (int d = 0; d < head_dim; d++) {
        out_local[d] = 0.0f;
    }

    for (int t = lid; t < cache_seq_len; t += local_size) {
        global half * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * vload_half(d, k_ptr);
        }
        float weight = exp(score * scale - max_score) / total_sum;

        if (write_scores) {
            S[score_layer_offset + head_idx * score_stride + t] = weight;
        }

        global half * v_ptr = V + kv_base + t * kv_pos_stride;
        for (int d = 0; d < head_dim; d++) {
            out_local[d] += weight * vload_half(d, v_ptr);
        }
    }

    for (int d = 0; d < head_dim; d++) {
        scratch[lid] = out_local[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            O[head_idx * head_dim + d] = scratch[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// ============================================================
// Fused F32->F16 Cast + HeadMajor Scatter for KV Cache Update
// ============================================================
// Replaces 2x cast kernel + 16x enqueue_copy_buffer with 1 kernel dispatch.
// Input: F32 [kv_heads, head_dim] (seq-major from rope/matmul)
// Output: F16 KV cache [kv_heads, capacity, head_dim] (head-major)
//
// Work items: kv_heads * head_dim (one per element)
kernel void kernel_kv_scatter_f32_to_f16(
    global const float * k_src,   // [kv_heads * head_dim] F32
    global const float * v_src,   // [kv_heads * head_dim] F32
    global half * k_dst,          // [kv_heads, capacity, head_dim] F16 HeadMajor
    global half * v_dst,          // [kv_heads, capacity, head_dim] F16 HeadMajor
    int head_dim,
    int capacity,
    int write_pos                 // current position in cache to write
) {
    int gid = get_global_id(0);
    int h = gid / head_dim;       // which head
    int d = gid % head_dim;       // which dim

    // Source: seq-major [h * head_dim + d]
    int src_idx = h * head_dim + d;

    // Dest: head-major [h * capacity * head_dim + write_pos * head_dim + d]
    int dst_idx = h * capacity * head_dim + write_pos * head_dim + d;

    k_dst[dst_idx] = (half)k_src[src_idx];
    v_dst[dst_idx] = (half)v_src[src_idx];
}

// ============================================================
// Fused F32->F16 Cast + HeadMajor Scatter — pos-as-buffer variant
// ============================================================
// M3.4 D-D.1: identical to kernel_kv_scatter_f32_to_f16 but reads `write_pos`
// from `pos_buf[0]` instead of taking it as an int scalar. Used by the QNN
// OpPackage `CustomKvScatter` op so multi-token decode can update pos at
// graph-execute time (SCALAR op params are baked at graph finalize).
//
// The original kernel_kv_scatter_f32_to_f16 is preserved for the OpenCL
// backend (engine/src/backend/opencl/{mod,plan}.rs) which sets pos via
// kernel arg per-call — that path doesn't have the QNN graph-finalize issue.
kernel void kernel_kv_scatter_f32_to_f16_oop(
    global const float * k_src,   // [kv_heads * head_dim] F32
    global const float * v_src,   // [kv_heads * head_dim] F32
    global const int * pos_buf,   // [1] int — pos_buf[0] = write_pos
    global half * k_dst,          // [kv_heads, capacity, head_dim] F16 HeadMajor
    global half * v_dst,          // [kv_heads, capacity, head_dim] F16 HeadMajor
    int head_dim,
    int capacity
) {
    int gid = get_global_id(0);
    int h = gid / head_dim;       // which head
    int d = gid % head_dim;       // which dim

    int write_pos = pos_buf[0];

    // Source: seq-major [h * head_dim + d]
    int src_idx = h * head_dim + d;

    // Dest: head-major [h * capacity * head_dim + write_pos * head_dim + d]
    int dst_idx = h * capacity * head_dim + write_pos * head_dim + d;

    k_dst[dst_idx] = (half)k_src[src_idx];
    v_dst[dst_idx] = (half)v_src[src_idx];
}

// ============================================================
// Fused F32->F16 Cast + HeadMajor Scatter (BATCH variant for prefill)
// ============================================================
// Writes `seq_len` positions in one kernel launch, replacing the per-token
// CPU scatter loop used by the F16 HeadMajor `can_direct_copy` path in
// `KVCache::update()`. Target is Qwen/Llama prefill where kv_write dominates
// (~40-53% of per-op time on Adreno 830, measured 2026-04-22).
//
// Input layout  (k_src / v_src): contiguous [seq_len, kv_heads, head_dim] F32
//   src_off = (s * kv_heads + h) * head_dim + d
// Output layout (k_dst / v_dst): HeadMajor [kv_heads, capacity, head_dim] F16
//   dst_off = h * capacity * head_dim + (write_pos_start + s) * head_dim + d
//
// NDRange: 3D [head_dim, seq_len, kv_heads]. No reduction, no SLM, no
// sub_group ops — pure embarrassingly parallel cast+copy. Adreno-friendly:
// no register pressure (scatter only).
kernel void kernel_kv_scatter_f32_to_f16_batch(
    global const float * k_src,
    global const float * v_src,
    global half * k_dst,
    global half * v_dst,
    int kv_heads,
    int head_dim,
    int capacity,
    int write_pos_start,
    int seq_len
) {
    int d = get_global_id(0);
    int s = get_global_id(1);
    int h = get_global_id(2);
    if (d >= head_dim || s >= seq_len || h >= kv_heads) return;

    int src_off = (s * kv_heads + h) * head_dim + d;
    int dst_off = h * capacity * head_dim + (write_pos_start + s) * head_dim + d;

    k_dst[dst_off] = (half)k_src[src_off];
    v_dst[dst_off] = (half)v_src[src_off];
}
