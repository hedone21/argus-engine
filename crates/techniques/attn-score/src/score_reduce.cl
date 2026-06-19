// GPU-side attention score reduction — fused single-dispatch variant.
//
// Older design (pre-fused):
//   - kernel_score_reduce    per layer (28x/token on Qwen2.5-1.5B)
//   - kernel_score_end_step  1x/token
//   - kernel_score_clear     2x/token
//   => 31 dispatches/token, ~500-700 us of launch overhead on Adreno.
//
// New design:
//   1. kernel_attn_gen writes scores to a per-layer slice of a
//      [n_layers, n_heads_q, score_stride] score buffer.
//   2. kernel_score_fused_reduce runs ONCE per token, iterating all
//      layers to compute MAX-across-layers of per-token aggregated
//      scores, and directly applies exponential decay + add into
//      cumulative importance buffers.
//
// Readback still happens only at eviction time (1 blocking read per
// eviction event).

// Fused reduction:
//   * Loops across layers for MAX aggregation (commutative, idempotent).
//   * Applies decay to the previous cumulative value once and adds the
//     aggregated step score.
//   * Handles both flat (sum over Q-heads) and per-KV-head (avg Q-heads
//     within GQA group) importance in a single pass so we only read the
//     score buffer once per (layer, t).
//
// scores: [n_layers, n_heads_q, score_stride] - post-softmax attn weights
//   slot(layer, h, t) = layer * (n_heads_q * score_stride) + h * score_stride + t
// importance: [max_seq_len] - cumulative flat importance (in-place update)
// head_importance: [n_kv_heads * max_seq_len] - cumulative per-head importance (in-place)
// decay_factor: (1.0 - decay), applied to cumulative before adding step contribution.
//
// Each work-item processes one token position t.
__kernel void kernel_score_fused_reduce(
    __global const float * scores,
    __global float * importance,
    __global float * head_importance,
    float decay_factor,
    int n_layers,
    int n_heads_q,
    int n_kv_heads,
    int cache_seq_len,
    int score_stride,
    int max_seq_len
) {
    int t = get_global_id(0);
    if (t >= cache_seq_len) return;

    int n_rep = n_heads_q / n_kv_heads;
    float inv_rep = 1.0f / (float)n_rep;
    int layer_stride = n_heads_q * score_stride;

    // MAX across layers of (sum over Q-heads of weights[l, h, t]).
    float step_flat = 0.0f;
    // MAX across layers of (avg over Q-heads within GQA group of weights[l, kv*n_rep+r, t]).
    // Materialised into per-kv local array so the (l, h) loop can update both
    // flat and per-head state in a single read of each score element.
    // n_kv_heads <= 16 in every supported model (Qwen2.5-1.5B = 2, Llama 3.2 1B = 8,
    // Gemma 3 = 4). Using a fixed-size stack array avoids dynamic allocation.
    float step_head_local[16];
    for (int kv = 0; kv < n_kv_heads; kv++) {
        step_head_local[kv] = 0.0f;
    }

    for (int l = 0; l < n_layers; l++) {
        int layer_base = l * layer_stride;

        // Pass 1: flat sum + per-head accumulation for this layer.
        float layer_flat = 0.0f;
        // Per-layer per-kv group sum.
        float layer_head[16];
        for (int kv = 0; kv < n_kv_heads; kv++) {
            layer_head[kv] = 0.0f;
        }

        for (int h = 0; h < n_heads_q; h++) {
            float w = scores[layer_base + h * score_stride + t];
            layer_flat += w;
            int kv = h / n_rep;
            layer_head[kv] += w;
        }

        step_flat = fmax(step_flat, layer_flat);
        for (int kv = 0; kv < n_kv_heads; kv++) {
            float avg = layer_head[kv] * inv_rep;
            step_head_local[kv] = fmax(step_head_local[kv], avg);
        }
    }

    importance[t] = importance[t] * decay_factor + step_flat;
    for (int kv = 0; kv < n_kv_heads; kv++) {
        int idx = kv * max_seq_len + t;
        head_importance[idx] = head_importance[idx] * decay_factor + step_head_local[kv];
    }
}
