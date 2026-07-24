# PyramidKV — byte-by-byte verification against NVIDIA kvpress

The Rust `pyramidkv` crate is a port of NVIDIA **kvpress** `PyramidKVPress`
([source](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/pyramidkv_press.py), which
itself ports the authors' [KVCache-Factory `pyramidkv_utils.py#L197`](https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/pyramidkv_utils.py#L197)).
This directory holds the machinery that keeps that port **honest against the real library**, at
three tiers.

## Verification tiers

| Tier | What it proves | Artifact | Needs |
|---|---|---|---|
| **1. Unit oracle** | Rust == committed fixtures (pure-Python verbatim ports of the kvpress arithmetic) | `pyramidkv_budget_ref.py`, `pyramidkv_select_ref.py` → `../tests/fixtures/*` asserted by `../src/tests.rs` and `engine/src/stages/kv/prefill_keepset.rs` | nothing (Rust `cargo test`) |
| **2. Real-library cross-check** | the fixtures (hence the Rust) == the **actual** `kvpress.PyramidKVPress` | **`verify_vs_kvpress.py`** | `pip install kvpress` (CPU) |
| **3. End-to-end token parity** | full greedy generation matches under compression | `run_kvpress.py` + `compare_tokens.py` + `gen_engine_fixture.py` | a HF model + the argus engine (GPU recommended) |

Tier 1 runs in CI with no Python. **Tier 2 is the "byte-by-byte vs the actual reference
implementation" check** — run it whenever you bump the pinned kvpress version. Tier 3 is the
optional whole-pipeline sanity (see "End-to-end" below).

## Tier 2 — run the real-library cross-check

kvpress `0.5.4` pins `transformers<5.3`, and Python ≥ 3.13 removed the stdlib `pipes` module that
kvpress's transitive dep `fire` imports (`verify_vs_kvpress.py` injects a 2-line `pipes` shim). No
CUDA needed — everything runs on CPU in float64/float32/float16.

```bash
cd crates/techniques/pyramidkv/reference
python3 -m venv --system-site-packages .venv     # reuse an existing CPU torch (>=2.3.1)
. .venv/bin/activate
pip install "transformers>=4.56,<5.3" "kvpress==0.5.4"
python verify_vs_kvpress.py                       # exit 0 iff all byte-identical checks pass
```

**Pinned reference:** `kvpress==0.5.4` (upstream git `6d965557`), `transformers 5.2.0`,
`torch 2.10.0+cpu`. Last run (2026-07-24):

```
[1] BUDGET      : 3840 grid rows + 60000 random/boundary | mismatches=0 -> PASS (byte-identical)
[2] SELECTION   : 7 cases / 24 kv-heads | mismatches=0 -> PASS (byte-identical)
[3] ENGINE-FIX  : budgets [24, 19, 13, 8] + all keep-sets | mismatches=0 -> PASS (byte-identical)
[4] ALGORITHM   : 1500 continuous-float configs | max|score diff|=5.82e-11 structural flips=0 -> PASS
[5] RESIDUALS   : 344 integer divergences = 344 EXACT ties, 0 non-tie bugs; f16-vs-f32 flip 9%
RESULT: ALL BYTE-IDENTICAL CHECKS PASS ✓
```

## What "byte-identical" covers — and the three residuals

**Byte-identical (proven above):**

* **Per-layer budget** (`get_layer_budget`) — exact over the whole committed grid *and* a 60 000
  random/boundary sweep. The Rust keeps the f64 op-order and uses `round_ties_even` to match
  Python's banker's rounding.
* **Per-head SnapKV selection** in the dominant `n_kept ≥ window` regime under **f32** scores —
  the real `SnapKVPress.score()` (mean-over-window → `avg_pool1d` → GQA group-mean → window-forced
  pad) + `torch.topk` reproduce the fixtures exactly, and on continuous-float attention the score
  vectors agree to ~6e-11 with **zero structural keep-set flips**.

**Residuals (NOT byte-identical — documented in `../src/lib.rs`, quantified by check [5]):**

1. **Exact score ties.** The integer LCG fixtures (values 0..999) manufacture score ties that the
   Rust breaks lower-index-first but `torch.topk`'s implementation-defined order does not — this is
   the *only* source of the fixture divergences (check [5] confirms **all 344 are exact ties, 0 are
   bugs**). Real f32 attention ties with measure zero, so this never fires in practice.
2. **f16 vs f32.** The engine accumulates the prefill attention in f32; kvpress softmaxes in f32
   then casts to the model dtype before pooling. On an **f16** model the topk boundary can flip for
   ~**9 %** of configs (finite precision, not an algorithm difference). For an f32 model the
   selection is exact. → *If a paper table reports PyramidKV on an f16/bf16 model, note that the
   engine's keep-set can differ from kvpress on a small fraction of layers purely from rounding;
   the budget is always exact.*
3. **Sub-window budgets** (`n_kept < window`, only in the degenerate high-compression
   SnapKV-uniform fallback). kvpress keeps `n_kept` of the max-tied window positions in
   `torch.topk`'s arbitrary order; the engine keeps the `n_kept` most-recent. **Only the kept
   COUNT is faithful**, not the set (there is no canonical target — the tie order is
   platform/dtype/version dependent, so recency is the principled choice).

## Tier 3 — real-forward keep-set parity (RUN — 2026-07-24)

Tiers 1–2 prove the *decision* is byte-identical **given the same attention**. Tier 3 closes the
last gap: does the engine, on a REAL forward, compute the same attention and make the same per-head
keep decision as kvpress? We run both on the **same qwen2.5-1.5b-instruct safetensors weights** and
the same 502-token prompt (`realforward_prompt.txt`), dump the engine's actual committed keep-set
(`ARGUS_DUMP_KEEPSET`), and compare it to the real `PyramidKVPress`'s keep-set
(`realforward_parity.py`, which monkeypatches `PyramidKVPress.compress` to capture the topk indices).

**Result (cr=0.5, kvpress f32):** per-(layer, head) Jaccard **mean 0.9990, min 0.9886**, **49/56
(88%) layer-heads exact**, per-layer pyramid budgets **byte-exact** (0 count mismatch). The <1%
residual is the engine's F16 weights + F32 PFA vs kvpress F32, plus Rust-vs-PyTorch kernel /
accumulation-order differences — well above the Gate 1 threshold of 0.98. (Recorded in the
argus-labs ledger: `experiment=pyramidkv-pfa-parity`,
run `2026-07-24T021928Z_pyramidkv-realforward-parity`.)

> ⚠ **Invocation footgun (fixed for eval-ll).** Originally the faithful per-head SnapKV path was
> armed **only when `eviction_policy == "none"`** (`engine/src/session/eval/runner.rs`), so invoking
> **`eviction plugin --name pyramidkv`** fell through to the generic score-fed eviction path — which
> never arms the PFA producer, so pyramidkv silently degraded to the **layer-wide `importance()`
> selection** (per-head 0/28, H2O-style; correct pyramid *budget*, wrong *selection*). This is now
> fixed on the **eval-ll** path: `routes_to_prefill_keepset` sends any registered PFA-reading stage
> to the faithful executor whether named explicitly or via the happy path (budget from
> `--kv-budget-ratio`), so **both** invocations now match kvpress (per-head Jaccard 0.999). The
> **bench/chat** loops still gate the same arming on `cache_manager.is_none()`, so
> `eviction plugin --name pyramidkv` there still runs degraded — those are the throughput/interactive
> paths, not the quality path; **use `--eval-ll` for pyramidkv quality numbers.**

Reproduce (engine side via the labs harness so it lands in the ledger; kvpress side in the venv):

```bash
# 1) engine — FAITHFUL: --eviction-target-ratio sets the keep fraction; NO `eviction plugin`.
#    (--eviction-target-ratio default is 0.75, i.e. cr=0.25 — set it explicitly to match kvpress.)
ARGUS_DUMP_KEEPSET=$RUN_DIR/keepset_engine.json \
  argus-eval --model-path <.../qwen2.5-1.5b-instruct> --eval-ll --eval-batch <one-record.json> \
    --max-seq-len 2048 --kv-budget-ratio 0.5 --eviction-target-ratio 0.5
#   confirm stderr shows:  [prefill-keepset] 'pyramidkv' active — PFA producer arms q_window=64
# 2) kvpress — same weights + same prompt, f32 (f16 overflows at deep layers):
python realforward_parity.py --engine-keepset $RUN_DIR/keepset_engine.json \
    --prompt-file realforward_prompt.txt --dtype float32 --cr 0.5
```

The older **token-parity** scripts (`run_kvpress.py` + `compare_tokens.py`) diff greedy generation
end-to-end; they are a coarser check (cross-stack base-model kernel differences can diverge tokens
independent of pyramidkv) and are superseded by the keep-set parity above. Keep them for a
whole-pipeline smoke test with `--dtype float32` and a short generation.

## Regenerating the fixtures

```bash
python pyramidkv_budget_ref.py  > ../tests/fixtures/budget_grid.csv
python pyramidkv_select_ref.py  > ../tests/fixtures/select_fixture.txt
python gen_engine_fixture.py    # prints Rust array literals for the engine test
```

After any regeneration, re-run **Tier 2** to confirm the new fixtures still match the real library.
