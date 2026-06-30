#!/usr/bin/env python3
"""Ground-truth reference for the KVPress PyramidKV per-head token SELECTION.

Pure Python (no torch/numpy) faithful re-implementation of the SnapKV score pipeline +
topk that NVIDIA kvpress PyramidKVPress.compress() runs per kv-head:

  scores = attn_weights.mean(dim=-2)                       # mean over the q-window  (÷window)
  scores = F.avg_pool1d(scores, k, padding=k//2, stride=1) # ÷k, count_include_pad=True
  scores = scores.view(bsz, n_kv, groups, L-w).mean(2)     # GQA group mean
  scores = F.pad(scores, (0, w), value=scores.max()+1)     # window forced-kept
  indices = scores.topk(n_kept, dim=-1).indices            # per kv-head keep set

INPUT = TensorKind::PrefillAttention: attn[q_head][key_pos] = window-summed softmax
attention to each prefix key, per ATTENTION head (pre-GQA). INTEGER-valued (a shared LCG,
mirrored exactly by the Rust test) so the per-position ranking is decided by integer
pooled-sums — identical in f32 (Rust) and f64 (here), with ties broken lower-index-first
in BOTH compile_keep_top_k (STABLE-desc) and topk_indices below. NOTE: lower-index-first is
NOT torch.topk's tie order (torch.topk's is implementation-defined, not lower-index-first);
this oracle is a deterministic parity check of the SCORE pipeline, and every case uses
n_kept >= window_size so the always-kept window is fully resident and no torch-tie residual
arises. The fixture stores only case params + expected per-kv-head keep sets; the Rust test
regenerates the attn via the same LCG.

Regenerate:
    python3 pyramidkv_select_ref.py > ../tests/fixtures/select_fixture.txt
"""

import sys

LCG_A = 1103515245
LCG_C = 12345
LCG_MASK = 0x7FFFFFFF


def synth_attn(n_q_heads, k_len, seed):
    """Deterministic INTEGER attention matrix; LCG runs row-major (head outer, pos inner),
    state continuous across heads. Mirrored bit-for-bit by the Rust test."""
    attn = []
    s = seed
    for _h in range(n_q_heads):
        row = []
        for _p in range(k_len):
            s = (LCG_A * s + LCG_C) & LCG_MASK
            row.append(float(s % 1000))
        attn.append(row)
    return attn


def avg_pool1d(x, kernel_size, padding):
    """F.avg_pool1d(stride=1, count_include_pad=True): zero-pad, divide by kernel_size."""
    n = len(x)
    out = []
    for i in range(n):
        s = 0.0
        for j in range(kernel_size):
            idx = i - padding + j
            if 0 <= idx < n:
                s += x[idx]
        out.append(s / kernel_size)
    return out


def topk_indices(scores, k):
    """Indices of the k largest scores; ties → lower index first (mirrors compile_keep_top_k's
    STABLE-desc, NOT torch.topk — torch.topk's tie order is implementation-defined). Returned
    ascending (the kept SET; cache order is irrelevant to attention)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(order[:k])


def pyramidkv_keep_per_head(attn, n_kv_heads, n_q_heads, window_size, kernel_size, n_kept):
    k_len = len(attn[0])
    groups = n_q_heads // n_kv_heads
    heavy_len = k_len - window_size
    assert n_kept >= window_size, "this reference covers n_kept >= window_size"

    per_qhead = []
    for h in range(n_q_heads):
        scaled = [attn[h][p] / window_size for p in range(heavy_len)]
        per_qhead.append(avg_pool1d(scaled, kernel_size, kernel_size // 2))

    keep_per_kv = []
    for kvh in range(n_kv_heads):
        grp = [per_qhead[kvh * groups + g] for g in range(groups)]
        scores = [sum(grp[g][p] for g in range(groups)) / groups
                  for p in range(heavy_len)]
        max_s = max(scores) if scores else 0.0
        full = scores + [max_s + 1.0] * window_size
        keep_per_kv.append(topk_indices(full, n_kept))
    return keep_per_kv


# (n_kv_heads, n_q_heads, k_len, window, kernel, n_kept, seed)
CASES = [
    (4, 4, 128, 8, 5, 64, 11),      # MHA, groups=1
    (2, 8, 128, 8, 5, 40, 22),      # GQA groups=4
    (8, 8, 256, 32, 5, 100, 33),    # larger, groups=1
    (1, 4, 64, 8, 1, 32, 44),       # kernel=1 (no pooling), single kv head, groups=4
    (4, 8, 200, 16, 3, 64, 55),     # kernel=3, groups=2
    (2, 2, 100, 8, 5, 99, 66),      # n_kept = k_len-1 (keep almost all)
    (3, 6, 150, 8, 5, 50, 77),      # n_kv not power of two, groups=2
]


def main(out):
    for (n_kv, n_q, k_len, w, k, n_kept, seed) in CASES:
        attn = synth_attn(n_q, k_len, seed)
        keep = pyramidkv_keep_per_head(attn, n_kv, n_q, w, k, n_kept)
        out.write(f"CASE {n_kv} {n_q} {k_len} {w} {k} {n_kept} {seed}\n")
        for h in keep:
            out.write("KEEP " + " ".join(str(x) for x in h) + "\n")


if __name__ == "__main__":
    main(sys.stdout)
