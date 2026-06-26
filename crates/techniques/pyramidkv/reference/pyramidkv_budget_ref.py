#!/usr/bin/env python3
"""Ground-truth reference for the KVPress PyramidKVPress per-layer budget.

`get_layer_budget` is a VERBATIM port of NVIDIA kvpress:
  https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/pyramidkv_press.py
which ports the official PyramidKV authors' arithmetic:
  https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/pyramidkv_utils.py#L197

No torch/numpy: the budget is pure scalar `f64` math. Emits a CSV grid the Rust unit
test (`get_layer_budget` in src/lib.rs) asserts byte-identically against. `round(...)`
is Python round-half-to-EVEN; the Rust side uses `f64::round_ties_even()` to match, and
`f64` ops are kept in the same order so the bits agree. Numbers like "0.1" round-trip
identically through Python `float()` and Rust `str::parse::<f64>()`, so the CSV is a true
cross-language oracle.

Regenerate (and, on a machine with kvpress installed, cross-check against the REAL
`kvpress.PyramidKVPress.get_layer_budget`):
    python3 pyramidkv_budget_ref.py > ../tests/fixtures/budget_grid.csv
"""

import csv
import sys


class _Cfg:
    def __init__(self, num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers


class _Module:
    def __init__(self, num_hidden_layers, layer_idx):
        self.config = _Cfg(num_hidden_layers)
        self.layer_idx = layer_idx


def get_layer_budget(module, q_len, *, compression_ratio, window_size, beta):
    """VERBATIM port of PyramidKVPress.get_layer_budget (NVIDIA kvpress)."""
    assert beta >= 1, "Beta should >= 1"

    max_capacity_prompt = window_size + q_len * (1 - compression_ratio)

    min_num = (max_capacity_prompt - window_size) / beta
    max_num = (max_capacity_prompt - window_size) * 2 - min_num

    if max_num >= q_len - window_size:
        max_num = q_len - window_size
        min_num = (max_capacity_prompt - window_size) * 2 - max_num

    if not (q_len >= max_num >= min_num >= window_size):
        return round(q_len * (1 - compression_ratio))

    steps = (max_num - min_num) / (module.config.num_hidden_layers - 1)
    return round(max_num - module.layer_idx * steps)


# Compact grid — exercises the pyramid branch, the max_num clamp, the SnapKV fallback,
# round-half-to-even boundaries, steep (beta=1) vs gentle (beta=20) pyramids, and all
# layer indices (the max_num/min_num endpoints + the interior).
Q_LENS = [128, 512, 1024, 4096]
CRATIOS = [0.1, 0.25, 0.5, 0.7, 0.9]
WINDOWS = [8, 64]
BETAS = [1, 20]
NUM_LAYERS = [16, 32]


def dump(out):
    w = csv.writer(out)
    w.writerow(["q_len", "compression_ratio", "window_size", "beta",
                "num_layers", "layer_idx", "budget"])
    for q_len in Q_LENS:
        for cr in CRATIOS:
            for ws in WINDOWS:
                for beta in BETAS:
                    for nl in NUM_LAYERS:
                        for li in range(nl):
                            m = _Module(nl, li)
                            b = get_layer_budget(
                                m, q_len, compression_ratio=cr,
                                window_size=ws, beta=beta)
                            w.writerow([q_len, repr(cr), ws, beta, nl, li, b])


if __name__ == "__main__":
    dump(sys.stdout)
