#!/usr/bin/env python3
"""Cross-check of the pure-Python AdaKV oracle (`adakv_select_ref.py`, hence the Rust `adakv`
crate) against a verbatim torch port of the ORIGINAL authors' `FFY0/AdaKV` `update_kv_gqa`
(`adaptive_snapkv/monkeypatch/snapkv_utils.py` @ 04497ab, L335-460), the port argus-labs audited
to Jaccard 1.0 against the official repo (`argus-labs/analysis/audits/adakv_impl/parity_adakv.py`,
`official_adakv_gqa`, reproduced below with its steps numbered as there).

What it checks, for every fixture case:
  [1] CAPS   the per-head budgets `cap_h` are identical (the flattened competition + floor blend);
  [2] KEEP   the per-head keep SETS are identical — or differ ONLY inside a plateau of equal
             pooled scores at a cut, which is the documented tie residual (torch.sort/topk order
             among equal values is implementation-defined; the oracle breaks ties lower-index
             first). Any other difference is a bug.

Setup: any CPU torch (the snapkv/pyramidkv kvpress venv works; no kvpress import needed).
    python verify_vs_official_port.py          # exit 0 iff all checks pass
"""

import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "tests", "fixtures", "select_fixture.txt")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oracle = _load("adakv_select_ref")


def official_adakv_gqa(prefix_mean_qhead, base, floor_alpha, n_kv, groups, kernel, pooling="maxpool"):
    """Verbatim from argus-labs parity_adakv.py::official_adakv_gqa (FFY0 update_kv_gqa port).
    Input `prefix_mean_qhead` = [H, L] window-mean attention (softmax-then-mean over the window).
    (1) gqa reduce (mean) -> [n_kv, L]  (2) pool  (3) sort desc  (4) flatten top-(n_kv*base) ->
    per-head count  (5) round(count*(1-fa) + floor_cap)  (6) per-head top-cap keep set."""
    H, L = prefix_mean_qhead.shape
    assert H == n_kv * groups
    m = prefix_mean_qhead.view(n_kv, groups, L).mean(dim=1)
    pool = {"maxpool": F.max_pool1d, "avgpool": F.avg_pool1d}[pooling]
    attn = pool(m, kernel_size=kernel, padding=kernel // 2, stride=1)  # [n_kv, L]
    sorted_score, sorted_idx = attn.sort(dim=-1, descending=True)
    flat = sorted_score.reshape(1, n_kv * L)
    win = flat.topk(n_kv * base, dim=-1).indices // L
    cap = torch.zeros(1, n_kv, dtype=torch.long)
    cap.scatter_add_(-1, win, torch.ones_like(win))
    assert int(cap.sum()) == n_kv * base
    floor_cap = int(base * floor_alpha)
    cap = torch.round(cap.float() * (1 - floor_alpha) + floor_cap).long()[0]
    keep = [sorted(sorted_idx[h, : int(cap[h])].tolist()) for h in range(n_kv)]
    return cap.tolist(), keep, attn


def parse_fixture():
    cases = []
    for line in open(FIXTURE):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "CASE":
            n_kv, n_q, k_len, w, k = map(int, parts[1:6])
            alpha = float(parts[6])
            base, protected, seed = map(int, parts[7:10])
            cases.append(dict(n_kv=n_kv, n_q=n_q, k_len=k_len, w=w, k=k, alpha=alpha, base=base,
                              protected=protected, seed=seed, caps=None, keeps=[]))
        elif parts[0] == "CAPS":
            cases[-1]["caps"] = list(map(int, parts[1:]))
        elif parts[0] == "KEEP":
            cases[-1]["keeps"].append(list(map(int, parts[1:])))
    return cases


def main():
    bad = 0
    tie_only = 0
    cases = parse_fixture()
    for c in cases:
        if c["protected"]:
            # FFY0 has no protected prefix; the oracle's `protected` shrinks the competition region
            # by construction. Verified through the p=0 cases; skipped here.
            print(f"seed {c['seed']}: protected={c['protected']} — not an FFY0 setting, skipped")
            continue
        attn = oracle.synth_attn(c["n_q"], c["k_len"], c["seed"])
        L = c["k_len"] - c["w"]
        # window-SUM (engine PFA) -> window-MEAN, the port's input
        pm = torch.tensor([[v / c["w"] for v in row[:L]] for row in attn], dtype=torch.float32)
        caps, keeps, pooled = official_adakv_gqa(pm, c["base"], c["alpha"], c["n_kv"],
                                                 c["n_q"] // c["n_kv"], c["k"])
        ok_caps = caps == c["caps"]
        print(f"seed {c['seed']}: caps official={caps} oracle={c['caps']} -> {'OK' if ok_caps else 'MISMATCH'}")
        if not ok_caps:
            bad += 1
            continue
        for h in range(c["n_kv"]):
            want = [p for p in c["keeps"][h] if p < L]  # the oracle's keep minus the forced window
            got = keeps[h]
            if got == want:
                continue
            # Diagnose: a symmetric difference entirely at one pooled value == a plateau tie.
            sd = set(got) ^ set(want)
            vals = {float(pooled[h, p]) for p in sd}
            if len(vals) == 1:
                tie_only += 1
                print(f"  head {h}: differs only inside a plateau at score {vals.pop():.6g} (tie residual, {len(sd)//2} swaps)")
            else:
                bad += 1
                print(f"  head {h}: MISMATCH beyond ties: got {sorted(sd)}")
    print(f"\nRESULT: {'PASS' if bad == 0 else 'FAIL'} (mismatches={bad}, plateau-tie-only heads={tie_only})")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
