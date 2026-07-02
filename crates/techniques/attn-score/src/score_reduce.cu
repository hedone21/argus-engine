// CUDA translation of score_reduce.cl — the GPU half of the observer/score axis (the H2O score
// policy). Compiled at plugin make() time by shelling out to system nvcc (parity with the engine's
// own kernels.cu), loaded via cudarc's module API, and launched on the engine's lent CUstream.
//
// Both kernels use one thread per token: t = blockIdx.x * blockDim.x + threadIdx.x, guarded by
// t < cache_seq_len. Reduction is entirely in per-thread stack arrays (no shared memory). The math is
// byte-for-byte the same policy as the OpenCL kernels: per-layer MAX aggregation, GQA group averaging
// (kernel A), A2SF decay + cross-step SUM, and a per-layer flat accumulate (kernel B). Time-norm is
// applied CPU-side by the producer, not here.
//
// `n_kv_heads <= 16` (the engine guards this before init) so the fixed stack arrays are safe.

extern "C" __global__ void kernel_score_fused_reduce(
    const float * __restrict__ scores,          // [n_layers, n_heads_q, score_stride]
    float * __restrict__ importance,            // [max_seq_len], updated in place
    float * __restrict__ head_importance,       // [n_kv_heads * max_seq_len], updated in place
    float decay_factor,
    int n_layers,
    int n_heads_q,
    int n_kv_heads,
    int cache_seq_len,
    int score_stride,
    int max_seq_len)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cache_seq_len) {
        return;
    }

    const int n_rep = n_heads_q / n_kv_heads;
    const float inv_rep = 1.0f / (float)n_rep;
    const int layer_stride = n_heads_q * score_stride;

    float step_flat = 0.0f;
    float step_head_local[16];
    for (int kv = 0; kv < n_kv_heads; ++kv) {
        step_head_local[kv] = 0.0f;
    }

    for (int l = 0; l < n_layers; ++l) {
        const int layer_base = l * layer_stride;
        float layer_flat = 0.0f;
        float layer_head[16];
        for (int kv = 0; kv < n_kv_heads; ++kv) {
            layer_head[kv] = 0.0f;
        }
        for (int h = 0; h < n_heads_q; ++h) {
            const float w = scores[layer_base + h * score_stride + t];
            layer_flat += w;
            layer_head[h / n_rep] += w;
        }
        step_flat = fmaxf(step_flat, layer_flat);
        for (int kv = 0; kv < n_kv_heads; ++kv) {
            const float avg = layer_head[kv] * inv_rep;
            step_head_local[kv] = fmaxf(step_head_local[kv], avg);
        }
    }

    importance[t] = importance[t] * decay_factor + step_flat;
    for (int kv = 0; kv < n_kv_heads; ++kv) {
        const int idx = kv * max_seq_len + t;
        head_importance[idx] = head_importance[idx] * decay_factor + step_head_local[kv];
    }
}

extern "C" __global__ void kernel_score_fused_reduce_per_layer(
    const float * __restrict__ scores,          // [n_layers, n_heads_q, score_stride]
    float * __restrict__ layer_flat,            // [n_layers * max_seq_len], updated in place
    float decay_factor,
    int n_layers,
    int n_heads_q,
    int cache_seq_len,
    int score_stride,
    int max_seq_len)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cache_seq_len) {
        return;
    }

    const int layer_stride = n_heads_q * score_stride;
    for (int l = 0; l < n_layers; ++l) {
        const int layer_base = l * layer_stride;
        float layer_sum = 0.0f;
        for (int h = 0; h < n_heads_q; ++h) {
            layer_sum += scores[layer_base + h * score_stride + t];
        }
        const int idx = l * max_seq_len + t;
        layer_flat[idx] = layer_flat[idx] * decay_factor + layer_sum;
    }
}
