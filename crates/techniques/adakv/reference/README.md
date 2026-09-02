# AdaKV — verification against the original authors' allocation

The Rust `adakv` crate is a port of **FFY0/AdaKV** `update_kv_gqa`
([source](https://github.com/FFY0/AdaKV/blob/main/adaptive_snapkv/monkeypatch/snapkv_utils.py),
`--gqa_support`, `gqa_func=mean`, `floor_alpha` blend) — the form the paper's AdaKV arm runs in
argus-labs (`harness/drive_longgen_code.py::build_selection_adakv`, audited against the official
repo at `argus-labs/analysis/audits/adakv_impl/`, Jaccard 1.0). It is NOT NVIDIA kvpress
`AdaKVPress`: that press guards the top `int(n_kept·α)` per head and takes a global bottom-k (a
pure floor, no blend), and then only masks the pruned keys ("does not reduce peak memory"), so it
is neither the paper's arm nor a memory technique.

## Pipeline (FFY0 order)

```
L      = current − window                                  # candidates; the window is always kept
m      = attn[qh][:L] / window                             # window-MEAN attention per query head
m_kv   = mean over the query heads of each KV head         # gqa_func=mean, BEFORE pooling
pooled = max_pool1d(m_kv, kernel=7, padding=3, stride=1)   # -inf padding
win    = top-(n_kv·base) over the flattened (head, pos)    # → count_h winners per head
cap_h  = round_f32(count_h·(1−α) + int(base·α))            # α = floor_alpha = 0.2
keep_h = [start_h, start_h+p) ∪ top-cap_h of pooled[h] ∪ [L, current)
```

`base = target_len − window` per head (the engine's ask, as `snapkv` derives it), `p` = the
engine's protected prefix (additive; FFY0 has none), `start_h` = the head's first resident slot on
a ragged cache (`0` otherwise).

## Verification tiers

| Tier | What it proves | Artifact | Needs |
|---|---|---|---|
| **1. Unit oracle** | Rust == committed fixture (pure-Python verbatim port) | `adakv_select_ref.py` → `../tests/fixtures/select_fixture.txt` asserted by `../src/tests.rs` | nothing (`cargo test -p adakv`) |
| **2. Official-port cross-check** | the fixture (hence the Rust) == a verbatim torch port of FFY0 `update_kv_gqa` | **`verify_vs_official_port.py`** | any CPU torch |

Tier 2 ports the labs `official_adakv_gqa` (itself audited against FFY0) rather than running the
FFY0 repository: that code is a transformers monkeypatch over a full model, and the allocation is
the whole difference between AdaKV and SnapKV.

## Ties — the residual

Max-pooling makes plateaus: up to `kernel` neighbours share one pooled value, so a cut inside a run
of equal scores is the common case. `torch.sort` / `torch.topk` order among equal values is
implementation-defined; the oracle and the Rust break ties the same deterministic way (flattened:
score desc, head asc, pos asc; per head: stable top-k, lower position first). Tier 2 therefore
diagnoses a head whose keep set differs from the torch port **only inside one plateau** as the tie
residual, and anything else as a mismatch. Each fixture case records the ties at its cuts
(`TIES` line).

Last run (2026-09-02, torch 2.10.0+cpu): caps identical on all 7 FFY0-settable cases; keep sets
identical or plateau-tie-only. See the script's output for the per-head diagnosis.

## Regenerating the fixture

```bash
python3 adakv_select_ref.py > ../tests/fixtures/select_fixture.txt
python3 verify_vs_official_port.py
```
