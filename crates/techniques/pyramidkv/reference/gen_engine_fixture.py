#!/usr/bin/env python3
"""Generate the expected per-layer / per-kv-head keep-sets for the engine byte-identical test
(`engine/src/stages/kv/prefill_keepset.rs`), from the verbatim kvpress reference.

Chosen params exercise the PYRAMID across layers (distinct per-layer budgets) + GQA (groups=2) +
the window-forced-keep. Prints Rust array literals to paste into the test (regenerate on change).
"""
import importlib.util
import os

_d = os.path.dirname(__file__)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_d, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


budget = _load("pyramidkv_budget_ref")
select = _load("pyramidkv_select_ref")

# Engine test params.
N_LAYERS = 4
N_KV = 2
N_Q = 4           # groups = 2
K_LEN = 32
WINDOW = 4
KERNEL = 1        # identity pooling — keeps the hand-verifiable path simple
BETA = 2
CR = 0.5
SEED_BASE = 1000  # layer seed = SEED_BASE + layer

print(f"// params: n_layers={N_LAYERS} n_kv={N_KV} n_q={N_Q} k_len={K_LEN} "
      f"window={WINDOW} kernel={KERNEL} beta={BETA} cr={CR} seed_base={SEED_BASE}")
budgets = []
for layer in range(N_LAYERS):
    m = budget._Module(N_LAYERS, layer)
    nk = budget.get_layer_budget(m, K_LEN, compression_ratio=CR, window_size=WINDOW, beta=BETA)
    budgets.append(nk)
print(f"// budgets per layer: {budgets}")

print("let expected: [[&[usize]; %d]; %d] = [" % (N_KV, N_LAYERS))
for layer in range(N_LAYERS):
    attn = select.synth_attn(N_Q, K_LEN, SEED_BASE + layer)
    keep = select.pyramidkv_keep_per_head(attn, N_KV, N_Q, WINDOW, KERNEL, budgets[layer])
    rows = ", ".join("&" + str(h) for h in keep)
    print(f"    [{rows}],  // layer {layer}, n_kept={budgets[layer]}")
print("];")
print("let budgets: [usize; %d] = %s;" % (N_LAYERS, budgets))
