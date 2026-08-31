# QCF decision overhead — measured against the closed-form cost model

Reproduction protocol and results for the cost of one QCF decision on Argus Engine, checked against
the closed form the paper states for it (`03_QCF_Function.tex:105` in the PerCom'27 draft).

Measured 2026-08-29 at `bf7a67d`.

---

## 1. What is being measured, and what is not

QCF scores a pool of KV-cache compression candidates by how far each moves the model's attention
output, and picks the smallest perturbation. In this engine that is `aperturb::decide`, reached
through `argus-eval --dump aperturb`. Per layer it computes one shared score matrix, then for the
baseline and every candidate an attention output and a rank-`r` projection, and aggregates the
per-cell relative change into one number per candidate.

**Two costs are in play and they are reported apart, because only the first is what the closed form
predicts.**

| symbol | what it is | charged to |
|---|---|---|
| `decide` | the algorithm: shared logits, per-candidate attention, projection, readout | QCF |
| `read` | materializing K and V as host `f32` — on a device cache, a round trip first | the engine's cache residency, not QCF |

`decide` is host code with no backend dispatch: there is no GPU implementation of the QCF
arithmetic. A GPU run and a CPU run execute the same instructions on the same data. What a GPU
backend changes is `read`. Section 8 is about that, and it is the one place where the measurement
found a cost the model has no term for.

Also **not** measured here: applying the winning candidate to the cache. `decide` is read-only
(INV-147); eviction cost is a separate axis with its own protocol.

---

## 2. The closed form being checked

For `L` layers, width `d`, `n_q` query heads of dimension `d_h`, cache length `S`, `R` scored
trailing rows, truncation rank `r`, and a pool of `|C|` candidates, the paper gives one decision as

```
|C| · L · ( 4·n_q·R·S·d_h  +  4·R·d·r )
             └ attention ┘    └ projection ┘
```

and states four properties: linear in `R`, linear in `S`, independent of vocabulary size and FFN
width, with hyperparameters `R = 16` and `r = d/128`.

> **The rank has since moved to `r = d/256`.** Everything measured here was run at `d/128`, which is
> what the tables below report; the engine's `APERTURB_WO_FRAC` now says `1/256`. Only the
> projection term scales with `r`, and §6.1 measures it flat and negligible either way, so the shape
> results are unaffected. What halves is the one-time factorization (§9).

The same section argues that the `W_o` matmul is `O(R d²)` — a fixed cost independent of cache
length, which therefore exceeds the `O(R S d)` attention term at short context — and that a rank-`r`
truncated SVD lowers it to `O(R d r)`, the factors being computed once per model online and stored,
so that no overhead is incurred after the first run.

Every one of those claims is checked below.

---

## 3. Method

Timing was added at two levels.

- **Inside the decision.** `aperturb::PhaseTimes`, carried on `Decision`, splits the wall clock into
  `logits` (the shared `Q Kᵀ`), `keypos` (per-candidate admitted-prefix bookkeeping), `attend`
  (softmax and the `V` contraction), `project` (the rank-`r` projection) and `readout` (per-cell
  relative change plus the closing RMS). The split is chosen to map onto the closed form's two
  terms: `logits + attend` is the attention term, `project` is the projection term, and `keypos` is
  work the model does not charge at all.
- **Around cache materialization.** One timer in the dump, over the loop that dequantizes every
  layer's K and V into host `f32`.

Both are emitted as one stderr line per question. The dump's JSONL schema is unchanged — it is a
contract, and a measurement artifact does not belong in it.

`S` is swept by prompt length and `R` by a temporary override of `APERTURB_ROWS`, which is
otherwise a constant (§5.2). Fits are least squares against the **measured** token count `n`, not
the target length.

Performance is measured without `--profile`, per `CLAUDE.md`.

---

## 4. Setup

### 4.1 Hardware and toolchain

| | |
|---|---|
| CPU | Intel Core Ultra 7 265K, 20 threads (the engine reports `Using 20 threads`) |
| GPU | NVIDIA GeForce RTX 3090 Ti, 24 GB, `sm_86` |
| CUDA | toolkit 13.3, driver 610.57.04 |
| OpenCL | NVIDIA ICD on the same device |

### 4.2 Builds

`opencl` is a default feature and the two GPU backends are mutually exclusive, so CUDA needs an
explicit opt-out:

```bash
cargo build --release --bin argus-eval                                        # CPU + OpenCL
cargo build --release --no-default-features --features cuda --bin argus-eval  # CPU + CUDA
```

Three binaries were used (CPU-backend runs were taken on the OpenCL build, and separately on the
CUDA build as a cross-check). That the three agree on `decide` to within a few percent is also the
cleanest available check that the numbers are not an artifact of one build.

**The GPU legs of this experiment are only possible at or after `bf7a67d`**, which lifted the dump's
outright refusal of a GPU backend; at `5a60cfe` the guard is still in place
(`aperturb_dump.rs:352`) and no GPU leg runs there. The refusal could only be lifted because
`5a60cfe` removed what caused it: the prefill flash-attention kernels raced on their local K/V tile,
so F32 KV scores moved between runs of the same binary.

### 4.3 Models

| | `L` | `n_q` | `n_kv` | `d_h` | `d` | `r` |
|---|---|---|---|---|---|---|
| `llama3.2-1b` | 16 | 32 | 8 | 64 | 2048 | 16 |
| `llama3.1-8b` | 32 | 32 | 8 | 128 | 4096 | 32 |

`r = d/128` in both cases — the value in force when these runs were taken — and the dump records
it (`wo_rank`). The canonical fraction is now `d/256`; see the note in §2.

### 4.4 Candidate pool

`build_pool` emits five candidates at three distinct budgets: `keep_all` (identity), `recent_r50`,
`recent_r25`, `sink_recent_r25`, `stride4`. So `|C| = 5`, and the retained fraction per candidate is
`1, 0.5, 0.25, 0.25, 0.25`. `decide` additionally computes one identity baseline, so six attention
passes per layer, not five — §7 accounts for it.

### 4.5 Prompts

Prefixes of a long English text, cut to hit target token counts of 512 … 15,000. **Content is
irrelevant to the timing**: `logits_into` is dot products and `attend_into` does one `exp` per
admitted column, both data-independent, so any sufficiently long text reproduces these numbers. The
runs below used prefixes of `scripts/fixtures/peekkv_needle_15714.txt` (currently on
`feat/peekkv-phase-a`). Each row's actual token count is recorded and the fits use it.

```python
import json
txt = open("long.txt").read()
cpt = len(txt) / 15714.0                       # chars per token, calibrated once
json.dump([{"id": f"s{t}", "prompt": txt[:int(t * cpt)], "choices": [" yes"]}
           for t in (512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 15000)],
          open("sweep.json", "w"))
```

---

## 5. Commands

```bash
argus-eval --model-path models/llama3.2-1b -b cpu --kv-type f32 \
  --max-seq-len 16384 --eval-ll --eval-batch sweep.json \
  --dump aperturb --dump-dir out/
```

`-b opencl` and `-b cuda` for the other legs; `--max-seq-len 8192` and a sweep topping out at 8,000
for `llama3.1-8b`, whose F32 cache is 4× larger per token.

### 5.1 Stop the process once the dump is written

The dump runs before the eval-LL scoring pass. That pass is not part of this measurement and, on
CPU at 15k tokens, runs for tens of minutes on every core. Kill on the `wrote N record(s)` line:

```bash
"$BIN" ... --dump aperturb --dump-dir out/ 2> run.err &
PID=$!
while kill -0 $PID 2>/dev/null; do
  grep -q "record(s), skipped" run.err && { sleep 1; kill $PID; break; }
  sleep 2
done
```

Kill by PID. `pkill -f argus-eval` also matches the shell running the harness.

### 5.2 The `R` sweep

`APERTURB_ROWS` is a `const`. The sweep used a temporary override in `run_aperturb_dump`, reverted
immediately afterwards — it is not in the tree:

```rust
let aperturb_rows: usize = std::env::var("ARGUS_APERTURB_ROWS")
    .ok().and_then(|v| v.parse().ok()).unwrap_or(APERTURB_ROWS);
```

used in place of `APERTURB_ROWS` at the `QRowCapture::new` call and the short-prompt skip. `decide`
already accepts any `rows` in `1..=32`.

### 5.3 The stored basis (§9.1)

```bash
# factor once, write the table
argus-eval --model-path models/llama3.2-1b -b cpu --kv-type f32 --max-seq-len 4096 \
  --eval-ll --eval-batch batch.json --dump aperturb --dump-dir out_compute/ \
  --aperturb-basis-out wo_1b.basis

# every run after that
argus-eval ... --dump-dir out_load/ --aperturb-basis wo_1b.basis

diff out_compute/aperturb.jsonl out_load/aperturb.jsonl   # must be empty
```

Pointing `--aperturb-basis wo_1b.basis` at `models/llama3.2-1b-instruct` is the negative control:
same `L`, same `d`, same rank, and it must still be refused.

---

## 6. Results — shape

### 6.1 Cache length

`llama3.2-1b`, CPU backend. `paper` and `engine` are FLOP counts derived from the formula and from
the code respectively (§7), not estimates.

| `S` | paper GF | engine GF | ratio | decide (s) | GFLOP/s | logits | attend | project |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 493 | 5.34 | 2.26 | 2.37 | 0.026 | 86.8 | 0.006 | 0.013 | **0.007** |
| 1 007 | 10.73 | 4.55 | 2.36 | 0.041 | 110.9 | 0.010 | 0.024 | **0.007** |
| 2 034 | 21.50 | 9.12 | 2.36 | 0.079 | 115.5 | 0.021 | 0.050 | **0.007** |
| 3 060 | 32.25 | 13.70 | 2.35 | 0.098 | 139.8 | 0.027 | 0.063 | **0.007** |
| 4 087 | 43.02 | 18.27 | 2.35 | 0.129 | 141.6 | 0.036 | 0.085 | **0.007** |
| 6 141 | 64.56 | 27.43 | 2.35 | 0.201 | 136.4 | 0.056 | 0.137 | **0.007** |
| 8 195 | 86.10 | 36.58 | 2.35 | 0.265 | 138.0 | 0.071 | 0.185 | **0.007** |
| 12 302 | 129.16 | 54.88 | 2.35 | 0.431 | 127.3 | 0.114 | 0.307 | **0.007** |
| 15 022 | 157.68 | 67.00 | 2.35 | 0.562 | 119.2 | 0.164 | 0.388 | **0.007** |

```
llama3.2-1b, CPU     decide(S)     =  36.079 µs/tok · S −  6.3 ms    R² = 0.99201
                     attention(S)  =  35.869 µs/tok · S − 13.4 ms    R² = 0.99184
                     projection(S) =   0.0000 µs/tok · S +  7.00 ms

llama3.2-1b, CUDA    decide(S)     =  36.639 µs/tok · S −  2.3 ms    R² = 0.98742
llama3.2-1b, OpenCL  decide(S)     =  35.698 µs/tok · S −  2.6 ms    R² = 0.98284
llama3.1-8b, CUDA    decide(S)     = 141.178 µs/tok · S −  2.7 ms    R² = 0.99156
```

**Linear in `S`: holds.** R² between 0.983 and 0.992 across three backends and two model
geometries, over a 30× span of cache length.

**The projection is independent of cache length: holds, exactly.** 7.00 ms at every `S` from 493 to
15,022 tokens; the fitted slope is 0.0000 µs per token. On the 8B geometry it is 31–41 ms, again
with no trend in `S` — the spread there is measurement resolution, not scaling (the stderr line
prints milliseconds).

**Independent of vocabulary size and FFN width: holds by construction.** `decide` reads query rows,
K, V and the `W_o` basis. No path touches the vocabulary or the FFN. There is nothing to measure.

### 6.2 Scored rows

`llama3.2-1b`, CUDA backend, `S` pinned at 8,002.

| `R` | decide (s) | logits | attend | project | read (s) |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.080 | 0.024 | 0.052 | 0.002 | 0.194 |
| 8 | 0.144 | 0.040 | 0.099 | 0.004 | 0.203 |
| 16 | 0.253 | 0.070 | 0.174 | 0.007 | 0.200 |
| 32 | 0.487 | 0.136 | 0.335 | 0.014 | 0.196 |

```
decide(R)     = 14.443 ms/row · R + 24.3 ms    R² = 0.99971
projection(R) =  0.424 ms/row · R +  0.4 ms    R² = 0.99895
```

**Linear in `R`: holds in the slope, with a floor the model does not have.** R² = 0.9997, but
eight times the rows costs 6.09× the time, not 8×, because of a 24.3 ms `R`-independent intercept.
That intercept is cache streaming: `logits_into` reads the whole of K and `attend_into` reads V
whether `R` is 4 or 32, so at small `R` the decision is memory-bound rather than FLOP-bound. The
projection, which touches no cache, is proportional with a 0.4 ms intercept — the pure-arithmetic
term behaving exactly as modelled.

`read` is flat in `R`, as it must be.

---

## 7. Results — magnitude

### 7.1 The engine executes 2.35× fewer FLOPs than the formula charges

The ratio converges to 2.35 and every point measured — both model sizes, every cache length — sits
within 0.8% of it: 2.3654 down to 2.3534 across the 1B sweep, and 2.3356 up to 2.3518 across the 8B
one. Three savings, none accidental.

**(a) The score matrix is computed once per layer, not once per candidate.** `Q Kᵀ` over the full
cache does not depend on which positions a candidate keeps — a candidate selects *which* logits are
admitted, downstream of computing them. `logits_into` is hoisted out of the candidate loop and its
doc comment says so. The formula charges it `|C|` times.

**(b) The value contraction runs over retained columns, not all of them.** `attend_into` walks
`list[..n_adm]`, the admitted prefix. A candidate at a 25% budget contracts a quarter of the cache.
The formula charges every candidate a full-length pass.

**(c) The projection is one matmul, not two.** Both readouts are invariant to the orthonormal `U_r`,
so `OutputBasis` stores `B_r = V_r Σ_r` and `project_into` computes `X · B_r` and never rotates
back — exactly half the formula's `4 R d r`. The type's doc comment states the invariance.

Counting in units of one full-cache pass (`n_q · R · S · d_h` MACs):

| | formula | engine |
|---|---:|---:|
| `Q Kᵀ` | 5 (one per candidate) | **1** (shared) |
| `· V` | 5 (each full length) | **3.25** (baseline 1 + `1, 0.5, 0.25, 0.25, 0.25`) |
| total | 10 | 4.25 |

`10 / 4.25 = 2.35`. Savings (a) accounts for roughly 70% of the gap and (b) for 30%; (c) moves a
term that is 0.2% of the total.

That is the limit at large `S`, and it is what the two sweeps approach from opposite sides. Two
finite-`S` corrections, both `O(1/S)`, explain the sub-percent drift:

- **The projection.** The formula charges it ten half-unit-equivalents and the engine six, so at
  short cache it pulls the ratio *down* toward 10/6. Its weight relative to the attention term is
  exactly `r/S`, which is twice as large at 8B (`r = 32`) as at 1B (`r = 16`) — which is why 8B
  approaches 2.35 from below and 1B does not.
- **The window boundary.** Scored row `t` admits only retained positions at or before `S − R + t`,
  so a short cache admits slightly *under* the nominal budget — `recent_r50` averages 0.49125 of a
  full pass at `S = 493` against 0.49975 at `S = 15,022` — which makes the engine cheaper still and
  pulls the ratio *up*.

The `max(1, ...)` blind-row clamp in `attend_into` plays no part: the tightest admitted prefix
anywhere in this sweep is 108 positions at `S = 493`, and every dumped arm reports `n_cells` at its
full `L · R`, so no scored row was blind in any run.

So the closed form is **sound as an upper bound and conservative as an estimate**. Nothing is
missing from the implementation; it is cheaper than its own specification.

### 7.2 The FLOP count predicts the clock

Effective throughput, from the fitted slope and the engine's FLOP count:

| | GFLOP/s |
|---|---:|
| `llama3.2-1b`, CPU | 123.5 |
| `llama3.2-1b`, CUDA | 121.6 |
| `llama3.2-1b`, OpenCL | 124.8 |
| `llama3.1-8b`, CUDA | 126.3 |

Stable within 4% across a 4× change in model size and three backends, which is what it means for a
FLOP model to explain a runtime. Independent cross-check: the FLOP counts predict the 8B slope
should be `17.83 / 4.46 = 4.00×` the 1B slope; measured, `141.178 / 36.079 = 3.91×`.

### 7.3 The truncation earns its place

Untruncated, the projection would cost `d/r = 128×` more — 0.90 s per decision at 1B, against a
measured 7 ms. That exceeds the entire attention term at every cache length measured here. The
paper's argument for the rank-`r` factorization is therefore confirmed by measurement rather than
assumed.

---

## 8. Results — CPU and GPU

`llama3.2-1b`. `decide` is the algorithm; `read` is getting K and V to where it can be run.

| `S` | decide CPU | decide OpenCL | decide CUDA | read CPU | read OpenCL | read CUDA | read ÷ decide (CUDA) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 493 | 0.026 | 0.037 | 0.040 | 0.010 | 0.167 | 0.347 | **8.67×** |
| 1 007 | 0.041 | 0.053 | 0.049 | 0.011 | 0.166 | 0.147 | 3.00× |
| 2 034 | 0.079 | 0.080 | 0.082 | 0.030 | 0.182 | 0.223 | 2.72× |
| 3 060 | 0.098 | 0.095 | 0.105 | 0.068 | 0.192 | 0.256 | 2.44× |
| 4 087 | 0.129 | 0.133 | 0.130 | 0.058 | 0.195 | 0.290 | 2.23× |
| 6 141 | 0.201 | 0.192 | 0.191 | 0.107 | 0.215 | 0.273 | 1.43× |
| 8 195 | 0.265 | 0.258 | 0.270 | 0.123 | 0.238 | 0.305 | 1.13× |
| 12 302 | 0.431 | 0.423 | 0.463 | 0.206 | 0.281 | 0.346 | 0.75× |
| 15 022 | 0.562 | 0.574 | 0.567 | 0.284 | 0.305 | 0.364 | 0.64× |

**The decision is backend-independent**, as expected of host code: above 2,000 tokens CPU and CUDA
differ by at most 7.4%.

**The cost of reaching the cache is not.** On CPU `read` is a dequantize; on a device cache it is a
per-layer synchronize plus a host copy, and at short context that fixed cost dominates — 0.347 s
against a 0.040 s decision. Only past roughly 10,000 tokens does the decision overtake the transfer.
On `llama3.1-8b` the same term is 0.57–1.08 s per decision.

This term does not appear in the cost model. The model is about arithmetic, and the arithmetic
checks out; but on a GPU backend the real per-decision cost below ~10k tokens is dominated by moving
K and V to the host, not by computing on them. Either the model needs a data-movement term, or the
decision needs to run where the cache already lives.

---

## 9. The one-time factorization, and what it cost before it was stored

The paper says the rank-`r` factors are computed once per model, online, and the values stored, so
that no offline step is needed and no overhead is incurred after the first run.

At the time of these measurements only the first half was implemented. There was no on-disk form and
no caller of a load path, so `OutputBasis::from_weights` ran at every process start:

| model | factorization |
|---|---:|
| `llama3.2-1b` (rank 16 of 2048, 16 layers) | 28.4 s, 29.0 s, 28.9 s over three runs |
| `llama3.1-8b` (rank 32 of 4096, 32 layers) | **717.5 s** |

Twelve minutes is not a rounding error against decisions that cost about a second. As measured,
"once per model" was really "once per process".

### 9.1 Closed since

`aperturb::basis_file` gives the table an on-disk form and two flags carry it:
`--aperturb-basis-out <file>` factors and writes, `--aperturb-basis <file>` loads. Measured on
`llama3.2-1b` at the current `r = d/256` (rank 8), same host, CPU backend:

| | wall |
|---|---:|
| factor (`--aperturb-basis-out`) | 10.6 s |
| load (`--aperturb-basis`) | **0.2 s** |

The two runs' `aperturb.jsonl` are byte-identical, so the load changes the cost and nothing else.
The file is 1 048 624 bytes for this model (48-byte header + `16 × 2048 × 8` f32) and 8 MiB for an
8B model; both are small enough to ship with an application.

The load path is deliberately load-only — a missing file or a header that disagrees is an error, not
a fall back to factoring — because the deployment target is Android, where the numbers above become
minutes to hours and a silent recompute would look like a hang. What the header carries is a digest
of the output projections themselves, not just the shapes: a basis built from `llama3.2-1b` offered
to `llama3.2-1b-instruct` has every dimension right and is refused on the digest.

The load path still pays one read and dequantize of `W_o`, which is what the digest is computed
over. That is the whole of what remains: 0.2 s here against the 10.6 s it removes.

## 10. What this says for the paper

- State the closed form as an **upper bound** and name the two structural savings — the shared
  `Q Kᵀ` and the retained-column contraction. "The implementation runs 2.35× under the stated cost"
  is a stronger and equally defensible claim than "the implementation matches it".
- Saving (a) holds only because every candidate scores the *same* query rows. That is true of QCF as
  specified — worth one sentence, since it is what licenses the hoist.
- The `R`-linearity claim is right in the slope and has a cache-streaming floor. At small `R` the
  decision is memory-bound.
- Either add a data-movement term or scope the model to a host-resident cache; on GPU it is the
  larger cost below ~10k tokens.
- The "no overhead after the first run" claim now holds for this code, but only because the
  factors are stored on disk and reloaded (§9.1); it did not hold for an in-memory-only
  implementation, and the paper is worth one sentence on which of the two it means.

---

## 11. Threats to validity

**One measurement was discarded.** The first CUDA sweep ran while a leftover CPU job held 15.6 of 20
cores — the eval-LL pass of the previous run, which is why §5.1 exists. Its decision times came out
2–17× high — median 6.5×, worst at the short prefixes, where fixed contention dwarfs a small
decision — and non-monotonic in `S`; the factorization read 73.5 s against 28.9 s clean, 2.5× on its
own. Everything
reported here was re-measured on an idle host, gated on load average ≤ 2. A decision time that is
not monotonic in `S` is the signature of that contamination.

**One host.** The GFLOP/s figures are specific to this machine. The ratios, the fitted shapes and
the FLOP counts are not.

**No Adreno leg.** Every number here is x86 host arithmetic. On a phone the same host code runs on
far fewer, slower cores while `read` crosses a unified rather than a discrete memory bus, so both
columns move and they do not move together. Section 8's crossover point in particular should not be
carried over.

**`decide` FLOP counts are derived, not instrumented.** They come from reading the kernels and
reconstructing each candidate's admitted counts, not from a hardware counter. The agreement of the
implied throughput across two model sizes and three backends (§7.2) is the evidence that the
derivation is right.

**The projection column is printed in milliseconds.** At 1B it reads a flat 7 ms; a sub-millisecond
trend in `S` would be invisible. The claim it supports — no scaling with cache length — is safe at
that resolution, since a term proportional to `S` would have grown 30× over this sweep.

---

## Appendix — raw output

```
### llama3.2-1b, CPU
[dump:aperturb] output-projection rank 16 of 2048 for 16 layers in 28.4s (worst residual 2.4e-7, worst deflation 0.9954)
[dump:aperturb] s512   n=493   rows=16 |C|=5 decide=0.026s (logits 0.006 keypos 0.000 attend 0.013 project 0.007 readout 0.000) read=0.010s
[dump:aperturb] s1024  n=1007  rows=16 |C|=5 decide=0.041s (logits 0.010 keypos 0.000 attend 0.024 project 0.007 readout 0.000) read=0.011s
[dump:aperturb] s2048  n=2034  rows=16 |C|=5 decide=0.079s (logits 0.021 keypos 0.000 attend 0.050 project 0.007 readout 0.000) read=0.030s
[dump:aperturb] s3072  n=3060  rows=16 |C|=5 decide=0.098s (logits 0.027 keypos 0.001 attend 0.063 project 0.007 readout 0.000) read=0.068s
[dump:aperturb] s4096  n=4087  rows=16 |C|=5 decide=0.129s (logits 0.036 keypos 0.001 attend 0.085 project 0.007 readout 0.000) read=0.058s
[dump:aperturb] s6144  n=6141  rows=16 |C|=5 decide=0.201s (logits 0.056 keypos 0.001 attend 0.137 project 0.007 readout 0.000) read=0.107s
[dump:aperturb] s8192  n=8195  rows=16 |C|=5 decide=0.265s (logits 0.071 keypos 0.001 attend 0.185 project 0.007 readout 0.000) read=0.123s
[dump:aperturb] s12288 n=12302 rows=16 |C|=5 decide=0.431s (logits 0.114 keypos 0.002 attend 0.307 project 0.007 readout 0.000) read=0.206s
[dump:aperturb] s15000 n=15022 rows=16 |C|=5 decide=0.562s (logits 0.164 keypos 0.002 attend 0.388 project 0.007 readout 0.000) read=0.284s

### llama3.2-1b, OpenCL
[dump:aperturb] output-projection rank 16 of 2048 for 16 layers in 29.0s (worst residual 2.4e-7, worst deflation 0.9954)
[dump:aperturb] s512   n=493   rows=16 |C|=5 decide=0.037s (logits 0.009 keypos 0.000 attend 0.021 project 0.007 readout 0.000) read=0.167s
[dump:aperturb] s1024  n=1007  rows=16 |C|=5 decide=0.053s (logits 0.014 keypos 0.000 attend 0.033 project 0.007 readout 0.000) read=0.166s
[dump:aperturb] s2048  n=2034  rows=16 |C|=5 decide=0.080s (logits 0.022 keypos 0.000 attend 0.051 project 0.007 readout 0.000) read=0.182s
[dump:aperturb] s3072  n=3060  rows=16 |C|=5 decide=0.095s (logits 0.026 keypos 0.001 attend 0.061 project 0.007 readout 0.000) read=0.192s
[dump:aperturb] s4096  n=4087  rows=16 |C|=5 decide=0.133s (logits 0.039 keypos 0.001 attend 0.086 project 0.007 readout 0.000) read=0.195s
[dump:aperturb] s6144  n=6141  rows=16 |C|=5 decide=0.192s (logits 0.054 keypos 0.001 attend 0.129 project 0.007 readout 0.000) read=0.215s
[dump:aperturb] s8192  n=8195  rows=16 |C|=5 decide=0.258s (logits 0.072 keypos 0.001 attend 0.177 project 0.007 readout 0.000) read=0.238s
[dump:aperturb] s12288 n=12302 rows=16 |C|=5 decide=0.423s (logits 0.122 keypos 0.002 attend 0.291 project 0.007 readout 0.000) read=0.281s
[dump:aperturb] s15000 n=15022 rows=16 |C|=5 decide=0.574s (logits 0.166 keypos 0.002 attend 0.398 project 0.008 readout 0.000) read=0.305s

### llama3.2-1b, CUDA
[dump:aperturb] output-projection rank 16 of 2048 for 16 layers in 28.9s (worst residual 2.4e-7, worst deflation 0.9954)
[dump:aperturb] s512   n=493   rows=16 |C|=5 decide=0.040s (logits 0.010 keypos 0.000 attend 0.023 project 0.007 readout 0.000) read=0.347s
[dump:aperturb] s1024  n=1007  rows=16 |C|=5 decide=0.049s (logits 0.013 keypos 0.000 attend 0.029 project 0.006 readout 0.000) read=0.147s
[dump:aperturb] s2048  n=2034  rows=16 |C|=5 decide=0.082s (logits 0.022 keypos 0.000 attend 0.052 project 0.007 readout 0.000) read=0.223s
[dump:aperturb] s3072  n=3060  rows=16 |C|=5 decide=0.105s (logits 0.030 keypos 0.001 attend 0.067 project 0.007 readout 0.000) read=0.256s
[dump:aperturb] s4096  n=4087  rows=16 |C|=5 decide=0.130s (logits 0.037 keypos 0.001 attend 0.085 project 0.007 readout 0.000) read=0.290s
[dump:aperturb] s6144  n=6141  rows=16 |C|=5 decide=0.191s (logits 0.057 keypos 0.001 attend 0.125 project 0.007 readout 0.000) read=0.273s
[dump:aperturb] s8192  n=8195  rows=16 |C|=5 decide=0.270s (logits 0.076 keypos 0.001 attend 0.185 project 0.008 readout 0.000) read=0.305s
[dump:aperturb] s12288 n=12302 rows=16 |C|=5 decide=0.463s (logits 0.127 keypos 0.002 attend 0.327 project 0.007 readout 0.000) read=0.346s
[dump:aperturb] s15000 n=15022 rows=16 |C|=5 decide=0.567s (logits 0.163 keypos 0.002 attend 0.394 project 0.008 readout 0.000) read=0.364s

### llama3.1-8b, CUDA
[dump:aperturb] output-projection rank 32 of 4096 for 32 layers in 717.5s (worst residual 3.6e-7, worst deflation 0.9971)
[dump:aperturb] s512   n=493  rows=16 |C|=5 decide=0.105s (logits 0.027 keypos 0.000 attend 0.040 project 0.038 readout 0.000) read=0.735s
[dump:aperturb] s1024  n=1007 rows=16 |C|=5 decide=0.152s (logits 0.046 keypos 0.000 attend 0.074 project 0.031 readout 0.000) read=0.569s
[dump:aperturb] s2048  n=2034 rows=16 |C|=5 decide=0.258s (logits 0.085 keypos 0.001 attend 0.138 project 0.033 readout 0.000) read=0.578s
[dump:aperturb] s4096  n=4087 rows=16 |C|=5 decide=0.519s (logits 0.170 keypos 0.002 attend 0.306 project 0.041 readout 0.000) read=0.684s
[dump:aperturb] s8000  n=8002 rows=16 |C|=5 decide=1.158s (logits 0.389 keypos 0.003 attend 0.728 project 0.039 readout 0.000) read=1.084s

### llama3.2-1b, CUDA, R sweep at S = 8002
R=4  decide=0.080s (logits 0.024 keypos 0.002 attend 0.052 project 0.002 readout 0.000) read=0.194s
R=8  decide=0.144s (logits 0.040 keypos 0.002 attend 0.099 project 0.004 readout 0.000) read=0.203s
R=16 decide=0.253s (logits 0.070 keypos 0.001 attend 0.174 project 0.007 readout 0.000) read=0.200s
R=32 decide=0.487s (logits 0.136 keypos 0.001 attend 0.335 project 0.014 readout 0.000) read=0.196s
```
