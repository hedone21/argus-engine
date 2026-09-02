#!/usr/bin/env python3
"""Ground-truth reference for the AdaKV (Ada-SnapKV) STAGE decision: FFY0/AdaKV `update_kv_gqa`
head-adaptive budget allocation, KV-head unit (`--gqa_support`, `gqa_func=mean`), as the paper's
arm runs it (argus-labs `harness/drive_longgen_code.py::build_selection_adakv`, audited against the
official repo at `analysis/audits/adakv_impl/`, Jaccard 1.0).

Pure Python (no torch/numpy). Pipeline per layer, for a `current`-long cache with observation
window `w`, kernel `k`, floor `alpha`, per-head base budget `base` (heavy hitters per head, the
window excluded) and an optional protected prefix `p` (force-kept in front of every head, additive
— the engine's attention-sink guard; FFY0 has none, so `p = 0` is the faithful setting):

  L      = current - w                                    # candidate prefix (the window is always kept)
  m[qh]  = attn[qh][:L] / w                               # window-MEAN attention per query head
  m_kv   = group mean over the query heads of each KV head   # gqa_func=mean  (BEFORE pooling)
  pooled = max_pool1d(m_kv, k, padding=k//2, stride=1)    # -inf padding
  region = [p, L) per head                                # the protected prefix does not compete
  win    = top-(nkv*base) over the flattened (head, pos) region   -> count[h] winners per head
  cap[h] = round_f32(count[h] * (1 - alpha) + int(base * alpha))  # FFY0 floor blend, float32 like torch
  keep[h] = [0, p) + top-cap[h] of pooled[h] over the region + [L, current)

INPUT = TensorKind::PrefillAttention: attn[q_head][key_pos] = window-SUMMED softmax attention to
each prefix key, per ATTENTION head (pre-GQA), INTEGER-valued via the shared LCG (pyramidkv's
constants, reduced mod 1_000_003 — see `synth_attn`) so f32 (Rust) and f64 (here) agree exactly;
every window and group count below is a power of two, so the means are exact too.

TIES. Max-pooling makes plateaus (up to `k` neighbours share one value), so a cut inside a plateau
is the common case, not the exception — for the real technique too, where `torch.topk`'s order
among equal scores is implementation-defined (the residual pyramidkv/snapkv document). This oracle
and the Rust stage break ties the SAME deterministic way — flattened competition: (score desc,
head asc, pos asc); per-head cut: (score desc, pos asc), the STABLE top-k — so the fixture is
exact for the stage. Each case records how many ties sat at its cuts (`TIES` line) so a reader can
see where torch could legitimately pick a different member of the same plateau.

Regenerate:
    python3 adakv_select_ref.py > ../tests/fixtures/select_fixture.txt
"""

import struct
import sys

_LCG_A, _LCG_C, _LCG_MASK = 1103515245, 12345, 0x7FFFFFFF
WIDE_MOD = 1_000_003  # integers below 2^24: exact in f32 AND f64


def synth_attn(n_q_heads, k_len, seed):
    """The shared LCG (pyramidkv's `synth_attn`, same constants and row-major state), but reduced
    mod `WIDE_MOD` instead of 1000: AdaKV's flattened competition ties far too often on a
    1000-valued alphabet (max-pooling squashes neighbours onto one value). Mirrored bit-for-bit by
    the Rust test."""
    attn = []
    s = seed
    for _h in range(n_q_heads):
        row = []
        for _p in range(k_len):
            s = (_LCG_A * s + _LCG_C) & _LCG_MASK
            row.append(float(s % WIDE_MOD))
        attn.append(row)
    return attn


def f32(x):
    """Round a Python float to float32 (torch does the blend in float32)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def round_half_even(x):
    """Python/torch `round`: half-to-even."""
    return int(round(x))


def max_pool1d(x, kernel):
    n = len(x)
    pad = kernel // 2
    out = []
    for i in range(n):
        lo = max(0, i - pad)
        hi = min(n, i + pad + 1)
        out.append(max(x[lo:hi]))
    return out


def adakv_caps(counts, base, alpha):
    """FFY0 `head_adaptive_capacity`: round(count*(1-alpha) + int(base*alpha)) in float32."""
    floor_cap = f32(float(int(base * alpha)))
    one_minus = f32(1.0 - alpha)
    return [round_half_even(f32(f32(float(c) * one_minus) + floor_cap)) for c in counts]


def adakv_keep_per_head(attn, n_kv, n_q, window, kernel, alpha, base, protected=0):
    """Returns (caps, keep sets per kv-head, ties) — `ties` = (flattened-cut tie?, per-head cut ties)."""
    k_len = len(attn[0])
    groups = n_q // n_kv
    L = k_len - window
    assert L > protected, "nothing to rank"
    # group mean of window-mean attention, then max-pool
    pooled = []
    for h in range(n_kv):
        m_kv = []
        for pos in range(L):
            s = 0.0
            for g in range(groups):
                s += attn[h * groups + g][pos] / window
            m_kv.append(s / groups)
        pooled.append(max_pool1d(m_kv, kernel))
    # flattened competition over the region [protected, L) of every head
    flat = [(pooled[h][pos], h, pos) for h in range(n_kv) for pos in range(protected, L)]
    flat.sort(key=lambda t: (-t[0], t[1], t[2]))
    n_win = n_kv * base
    assert n_win <= len(flat), "budget exceeds the region"
    flat_tie = n_win < len(flat) and flat[n_win - 1][0] == flat[n_win][0]
    counts = [0] * n_kv
    for _, h, _ in flat[:n_win]:
        counts[h] += 1
    caps = adakv_caps(counts, base, alpha)
    keeps = []
    head_ties = 0
    for h in range(n_kv):
        order = sorted(range(protected, L), key=lambda pos: (-pooled[h][pos], pos))
        c = caps[h]
        if 0 < c < len(order) and pooled[h][order[c - 1]] == pooled[h][order[c]]:
            head_ties += 1
        keep = list(range(protected)) + sorted(order[:c]) + list(range(L, k_len))
        keeps.append(keep)
    return caps, keeps, (flat_tie, head_ties)


# (n_kv_heads, n_q_heads, k_len, window, kernel, alpha, base, protected, seed)
CASES = [
    (2, 2, 96, 8, 7, 0.2, 20, 0, 101),   # MHA, 2 heads compete for 40 slots
    (2, 4, 128, 8, 7, 0.2, 24, 0, 202),  # GQA groups=2
    (4, 8, 160, 16, 7, 0.2, 20, 0, 303), # GQA groups=2, 4 heads
    (2, 4, 128, 8, 7, 0.0, 24, 0, 404),  # alpha=0: pure competition, no floor
    (2, 4, 128, 8, 7, 1.0, 24, 0, 505),  # alpha=1: floor only = uniform base (SnapKV-shaped counts)
    (2, 4, 128, 8, 5, 0.2, 24, 0, 606),  # kernel 5
    (2, 4, 128, 8, 7, 0.2, 24, 4, 707),  # protected prefix 4 (the pool's sink guard), additive
    (3, 6, 150, 8, 7, 0.2, 18, 0, 808),  # n_kv not a power of two
]


def main(out):
    for (n_kv, n_q, k_len, w, k, alpha, base, protected, seed) in CASES:
        attn = synth_attn(n_q, k_len, seed)
        caps, keeps, (flat_tie, head_ties) = adakv_keep_per_head(
            attn, n_kv, n_q, w, k, alpha, base, protected
        )
        out.write(f"CASE {n_kv} {n_q} {k_len} {w} {k} {alpha!r} {base} {protected} {seed}\n")
        out.write(f"TIES flat={int(flat_tie)} heads={head_ties}\n")
        out.write("CAPS " + " ".join(str(c) for c in caps) + "\n")
        for h in keeps:
            out.write("KEEP " + " ".join(str(x) for x in h) + "\n")


if __name__ == "__main__":
    main(sys.stdout)
