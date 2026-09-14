#!/usr/bin/env python3
"""Ground-truth reference for the NVIDIA kvpress SnapKV budget.

`SnapKVPress` inherits `ScorerPress.compress` unchanged
(https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/scorer_press.py):

    n_kept = int(k_len * (1 - self.compression_ratio))

Python `int()` TRUNCATES toward zero. This is NOT `round`: `PyramidKVPress.get_layer_budget`'s
SnapKV *fallback* is `round(q_len * (1 - compression_ratio))` (half-to-even), so the two presses'
"uniform SnapKV budget" disagree whenever the product's fractional part is >= .5 (q=1283, cr=0.5:
641 here, 642 there). The Rust `snapkv::snapkv_budget` mirrors THIS file; `f64` ops are kept in
the same order so the bits agree, and `as usize` truncates like `int()`.

No torch/numpy. Emits a CSV grid the Rust unit test asserts byte-identically. Numbers like "0.1"
round-trip identically through Python `float()` and Rust `str::parse::<f64>()`.

Regenerate (and, in the kvpress venv, cross-check against the REAL `SnapKVPress.compress`):
    python3 snapkv_budget_ref.py > ../tests/fixtures/budget_grid.csv
    python3 verify_vs_kvpress.py
"""

import csv
import sys


def snapkv_budget(k_len, compression_ratio):
    """VERBATIM `ScorerPress.compress` arithmetic (NVIDIA kvpress)."""
    return int(k_len * (1 - compression_ratio))


# Odd/even lengths, the pool's real prompt lengths (1281/1283), and ratios whose products land on
# .5 (truncation vs rounding), just below an integer (1e-15 noise) and near-zero.
K_LENS = [1, 2, 3, 7, 64, 65, 100, 128, 500, 1024, 1281, 1283, 2048, 4096]
CRATIOS = [0.1, 0.25, 0.333, 0.5, 0.7, 0.9, 0.999999]


def dump(out):
    w = csv.writer(out)
    w.writerow(["k_len", "compression_ratio", "budget"])
    for k_len in K_LENS:
        for cr in CRATIOS:
            w.writerow([k_len, repr(cr), snapkv_budget(k_len, cr)])


if __name__ == "__main__":
    dump(sys.stdout)
