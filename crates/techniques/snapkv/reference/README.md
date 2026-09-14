# SnapKV — byte-by-byte verification against NVIDIA kvpress

The Rust `snapkv` crate is a port of NVIDIA **kvpress** `SnapKVPress`
([source](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/snapkv_press.py)). It shares
its *selection* with the `pyramidkv` crate — kvpress `PyramidKVPress` subclasses `SnapKVPress` and
overrides only the budget — so the selection lives in the extension API
(`argus_extension_api::snapkv_per_head_keep`) and is verified by
[`../../pyramidkv/reference/`](../../pyramidkv/reference/README.md). This directory verifies what is
SnapKV's own: the **budget**, and the **stage decision** that composes budget and selection.

## The one thing that differs from PyramidKV's "SnapKV fallback": `int`, not `round`

`SnapKVPress` inherits `ScorerPress.compress`
([source](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/scorer_press.py)):

```python
n_kept = int(k_len * (1 - self.compression_ratio))   # Python int(): truncation toward zero
```

`PyramidKVPress.get_layer_budget` falls back to `round(q_len * (1 - compression_ratio))`
(half-to-even). The two disagree whenever the product's fractional part is ≥ .5, and also on
plain f64 noise: `1 - 0.9` is `0.09999999999999998`, so `int(80 * (1 - 0.9))` is **7**, where
`round` gives 8. The Rust `snapkv_budget` keeps the f64 operation order and truncates with
`as usize`; the pyramidkv crate's fallback keeps `round_ties_even`. Both are faithful to *their*
press.

## Verification tiers

| Tier | What it proves | Artifact | Needs |
|---|---|---|---|
| **1. Unit oracle** | Rust == committed fixtures (pure-Python verbatim ports) | `snapkv_budget_ref.py`, `snapkv_select_ref.py` → `../tests/fixtures/*` asserted by `../src/tests.rs` | nothing (`cargo test -p snapkv`) |
| **2. Real-library cross-check** | the fixtures (hence the Rust) == the **actual** `kvpress.SnapKVPress.compress` | **`verify_vs_kvpress.py`** | `pip install kvpress` (CPU) |

`snapkv_select_ref.py` loads pyramidkv's selection oracle by path (one oracle for the shared
selection) and composes it with this budget, so `select_fixture.txt` is the whole **stage**
decision — what `snapkv` must return for a given prefill attention — and `src/tests.rs` drives the
real stage (`keep_spec`) against it, not the selection function alone.

## Tier 2 — run the real-library cross-check

Same venv as pyramidkv (kvpress `0.5.4` pins `transformers<5.3`; Python ≥ 3.13 needs the 2-line
`pipes` shim the script injects). No CUDA needed.

```bash
cd crates/techniques/snapkv/reference
python3 -m venv --system-site-packages .venv     # reuse an existing CPU torch
. .venv/bin/activate
pip install "transformers>=4.56,<5.3" "kvpress==0.5.4"
python verify_vs_kvpress.py                       # exit 0 iff all byte-identical checks pass
```

`[2] STAGE` drives the real `compress()` end to end: the keys are tagged with their position, and
the gathered keys reveal which positions kvpress kept — budget AND selection in one call.

**Pinned reference:** `kvpress==0.5.4`, `transformers 5.2.0`, `torch 2.10.0+cpu`. Last run
(2026-09-02):

```
[1] BUDGET      : 91 grid rows (7 skipped: q_len <= window) + 3000 random | mismatches=0 -> PASS (byte-identical)
[2] STAGE       : 8 cases / 28 kv-heads | mismatches=0 (exact-tie residuals=0) -> PASS (byte-identical)
RESULT: ALL BYTE-IDENTICAL CHECKS PASS ✓
```

(The 7 skipped grid rows are `k_len = 1`: kvpress asserts `q_len > window_size`, so the real
library cannot be asked; the oracle still covers them and the Rust test asserts them.)

## Residuals

Inherited verbatim from the shared selection — see pyramidkv's
[README](../../pyramidkv/reference/README.md) "What byte-identical covers — and the three
residuals": exact score ties (the fixture cases are chosen tie-free at the cut, so Tier 2 reports
`exact-tie residuals=0`; a case whose cut ties would differ from `torch.topk` only in tie order and
is diagnosed as such, not as a mismatch), f16-vs-f32, and sub-window budgets (count-faithful only).
Nothing is re-derived here.

## Inside a `--aperturb-select` pool

Two things differ from a bare kvpress run and are documented in the crate:

* the pool hands every candidate a **protected prefix of 4** (`stage_params_for` turns a declared
  `0` into 4, the attention-sink guard), additive to the budget — the Tier 1/2 fixtures are taken at
  `protected_prefix = 0`, the kvpress-faithful value;
* with no explicit `compression_ratio` the budget is the engine's ask (`target_len`) itself, exact —
  not a `cr` round trip, which the truncation would turn into an off-by-one for most asks.

## Regenerating the fixtures

```bash
python snapkv_budget_ref.py  > ../tests/fixtures/budget_grid.csv
python snapkv_select_ref.py  > ../tests/fixtures/select_fixture.txt
```

After any regeneration, re-run **Tier 2** to confirm the new fixtures still match the real library
(and that no case ties at the top-k cut).
