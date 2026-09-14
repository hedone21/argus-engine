#!/usr/bin/env python3
"""Ground-truth reference for the SnapKV STAGE decision: kvpress budget x SnapKV selection.

There is ONE selection oracle in this repository — pyramidkv's `pyramidkv_select_ref.py` (the
`SnapKVPress.score` + `topk` pipeline, which kvpress `PyramidKVPress` inherits unchanged; the Rust
side is the shared `argus_extension_api::snapkv_per_head_keep`). This file loads it by path and
composes it with the SnapKV budget (`snapkv_budget_ref.py`, `int(k_len * (1 - cr))`), so a case
here is what the whole `snapkv` stage must decide from a given prefill attention.

Every case has `n_kept >= window_size` (the window fully resident) — the sub-window residual is
documented by pyramidkv and inherited. Ties break lower-index-first on both sides (see the
pyramidkv oracle's note on torch.topk). The fixture stores case params + expected per-kv-head
keep sets; the Rust test regenerates the attention via the same LCG.

Regenerate:
    python3 snapkv_select_ref.py > ../tests/fixtures/select_fixture.txt
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYRAMIDKV_REF = os.path.join(_HERE, "..", "..", "pyramidkv", "reference")


def _load(directory, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(directory, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


select = _load(_PYRAMIDKV_REF, "pyramidkv_select_ref")  # the shared SnapKV selection oracle
budget = _load(_HERE, "snapkv_budget_ref")

synth_attn = select.synth_attn
snapkv_keep_per_head = select.pyramidkv_keep_per_head  # SnapKV selection; PyramidKV inherits it


# (n_kv_heads, n_q_heads, k_len, window, kernel, compression_ratio, seed)
# Products with a fractional part >= .5 make the truncation visible (int != round).
CASES = [
    (4, 4, 128, 8, 5, 0.5, 11),       # MHA, n_kept = 64
    (4, 4, 128, 8, 5, 0.55, 12),      # 128*0.45 = 57.6 -> int 57 (round would give 58)
    (2, 8, 128, 8, 5, 0.7, 22),       # GQA groups=4, 128*0.3 = 38.4 -> 38
    (8, 8, 256, 32, 5, 0.6, 33),      # 256*0.4 = 102.4 -> 102
    (1, 4, 64, 8, 1, 0.5, 44),        # kernel=1 (no pooling), single kv head, groups=4
    (4, 8, 200, 16, 3, 0.7, 56),      # kernel=3, groups=2, n_kept = 60 (seed 55 ties at the cut)
    (2, 2, 100, 8, 5, 0.01, 66),      # keep almost all: 100*0.99 -> 99
    (3, 6, 150, 8, 5, 0.66, 77),      # n_kv not a power of two, groups=2, 150*0.34 -> 51
]


def main(out):
    for (n_kv, n_q, k_len, w, k, cr, seed) in CASES:
        n_kept = budget.snapkv_budget(k_len, cr)
        assert n_kept >= w, f"case {seed}: n_kept {n_kept} < window {w}"
        attn = synth_attn(n_q, k_len, seed)
        keep = snapkv_keep_per_head(attn, n_kv, n_q, w, k, n_kept)
        out.write(f"CASE {n_kv} {n_q} {k_len} {w} {k} {cr!r} {seed}\n")
        for h in keep:
            out.write("KEEP " + " ".join(str(x) for x in h) + "\n")


if __name__ == "__main__":
    main(sys.stdout)
