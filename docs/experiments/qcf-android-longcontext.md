# QCF on Adreno — the long-context leg, and what the read-back cost really is

Answers the two items left open by the Android/Adreno port write-up and its follow-up handoff (both
uncommitted working documents under `docs/design/`). Measured 2026-08-31.

The headline is not the Android numbers. It is that the read-back term this repo has been treating
as an intrinsic cost of running QCF on a GPU backend is **proportional to the allocated cache, not
to the retained cache** — which makes `docs/experiments/qcf-overhead.md` §8's "only past roughly
10,000 tokens does the decision overtake the transfer" an artifact of that sweep's
`--max-seq-len 16384`. §4 reproduces the effect on the desktop, where it is 8.6×.

Every number below is the **minimum over three repetitions**; §1.1 says why.

---

## 1. Setup

| | |
|---|---|
| device | Galaxy S25 (SM-S931N), Adreno 830, 11.4 GB RAM, serial `R3CY408S5SB` |
| build | `aarch64-linux-android`, `--no-default-features --features opencl,pyramidkv`, branch `chore/remove-legacy-qcf-v2` |
| model | `models/llama3.2-1b`, BF16 safetensors loaded as F16, `L=16 n_q=32 n_kv=8 d_h=64` |
| basis | `wo_1b.basis`, 1 048 624 B, rank 8 of 2048 (`APERTURB_WO_FRAC = 1/256`), factored on x86, **loaded** on ARM in 0.3–0.8 s |
| KV | F32, HeadMajor, fully pre-allocated at `capacity = max_seq_len` |
| prompts | prefixes of `scripts/fixtures/peekkv_needle_15714.txt`, the same ladder as the desktop sweep (`qcf-overhead.md` §4.5) |

`decide` is host code on every backend; only the forward and the cache location change.

### 1.1 Single-shot timings on this phone are not usable

The first pass — one question per length, ascending — produced a **non-monotone** curve: `decide`
fell from 0.384 s at `S = 2 034` to 0.356 s at `S = 3 060`, and later 1.385 s at `S = 8 195` against
1.136 s at `S = 9 222`. That is DVFS and clock ramp, and the desktop sweep never had to deal with it.

Everything below therefore uses **three interleaved repetitions** — `s512 s1024 … s8192` repeated
three times in one process, so each length meets a mix of thermal states — and reports the minimum
per length. The spread that estimator removes is worth recording on its own:

| `S` | min | median | max | spread |
|---:|---:|---:|---:|---:|
| 493 | 0.093 | 0.101 | 0.123 | 32.3% |
| 1 007 | 0.167 | 0.229 | 0.236 | 41.3% |
| 2 034 | 0.268 | 0.285 | 0.373 | 39.2% |
| 4 087 | 0.482 | 0.509 | 0.511 | 6.0% |
| 6 141 | 0.712 | 0.752 | 0.771 | 8.3% |
| 8 195 | 0.951 | 0.991 | 0.992 | 4.3% |

The spread collapses as `S` grows, which is the signature of a fixed-size scheduling and clock-ramp
disturbance rather than of throttling: at long `S` the measured interval is long enough to average
the governor out. Zone `cpu-0-0-0` went 31.1 °C → 56.5 °C over the session; no point had to be
discarded as throttled. (`devices.toml` names the zone `cpu-0-0-usr` for these units; the actual
type string on this handset is `cpu-0-0-0`.)

---

## 2. Part A — the shape claims hold on Adreno

`llama3.2-1b`, OpenCL backend, `--max-seq-len 8704`, min of three.

| `S` | decide | logits | attend | project | read |
|---:|---:|---:|---:|---:|---:|
| 493 | 0.093 | 0.029 | 0.052 | **0.010** | 0.259 |
| 1 007 | 0.167 | 0.053 | 0.101 | **0.013** | 0.273 |
| 2 034 | 0.268 | 0.094 | 0.162 | **0.012** | 0.325 |
| 4 087 | 0.482 | 0.176 | 0.293 | **0.012** | 0.342 |
| 6 141 | 0.712 | 0.258 | 0.440 | **0.012** | 0.431 |
| 8 195 | 0.951 | 0.337 | 0.601 | **0.012** | 0.429 |

```
Adreno 830   decide(S)  = 109.684 µs/tok · S + 44.1  ms    R² = 0.99931
             attend(S)  =  69.655 µs/tok · S + 19.9  ms    R² = 0.99813
             project(S) =   0.079 µs/tok · S + 11.5  ms    R² = 0.060    (i.e. flat, ~12 ms)
             read(S)    =  23.359 µs/tok · S + 257.7 ms    R² = 0.928    [capacity 8704 throughout — see §4]
```

**Linear in `S`: holds, and more cleanly than on the desktop.** R² = 0.99931 against the desktop's
0.983–0.992. The device is slower, so the fixed costs are a smaller fraction of each point.

**The projection is independent of cache length: holds.** 10–13 ms at every `S` from 493 to 8 195,
with no trend; the fit's R² of 0.06 is the fit describing noise, which is the correct outcome for a
constant. The desktop's constant was 7.00 ms at rank 16; this is rank 8 on a ~3× slower core.

**The FLOP model still predicts the clock, at one new constant.** The engine executes 4.456 MFLOP
per token of `S` at this geometry (`qcf-overhead.md` §7.1; the rank-dependent projection term is
0.2% of it, so rank 8 vs 16 does not move this, and it lands in the intercept, not the slope). The
slope implies:

| | GFLOP/s |
|---|---:|
| desktop CPU | 123.5 |
| desktop OpenCL | 124.8 |
| desktop CUDA | 121.6 |
| **Adreno 830, host CPU** | **40.6** |

One ratio — 3.04× — carries the entire difference between the phone and the desktop. The cost model
does not need a new term to describe the phone; it needs the phone's throughput constant.

### 2.1 The decision is *not* backend-independent on the phone

`qcf-overhead.md` §8 records that CPU and CUDA `decide` differ by at most 7.4% above 2 000 tokens,
"as expected of host code". On the phone, same host code, same `S`:

| `S` | `-b opencl` | `-b cpu` | ratio |
|---:|---:|---:|---:|
| 2 034 (both at `--max-seq-len 2560`) | 0.297 | 0.527 | **1.77×** |
| 493 (`8704` vs `2560`) | 0.093 | 0.150 | 1.61× |

The second row compares across allocations, which is sound only because §4.1 shows `decide` does not
depend on the allocation; the first row is a clean same-allocation comparison.

The CPU-backend run does its prefill on the same cores that then run `decide`, and leaves them hot
and clock-limited. On a desktop the CPU has thermal headroom for both; on a phone it does not. So
"the decision is backend-independent" is a desktop property, not a property of the decision.

### 2.2 The ceiling

Stepped up until the process died, `--max-seq-len` set just above the prompt in each probe.

| prompt | `--max-seq-len` | result |
|---:|---:|---|
| 8 195 | 8 704 | ran |
| **9 222** | **9 472** | **ran; peak RSS 4.26 GB, 200 MB swapped, `MemFree` floor 399 MB** |
| 10 240 | 10 496 | **SIGKILL** during prefill, `MemFree` floor 228 MB, peak RSS 3.04 GB |
| 12 302 | 16 384 | **SIGKILL** during prefill |

**The ceiling on this handset is between 9 222 and 10 240 tokens**, and the kill is the Android
low-memory killer, not an allocation failure — the process takes SIGKILL (rc 137) while `MemFree` is
at 228 MB, where a surviving run exits on the harness's own SIGTERM (rc 143). There is 12.6 GB of
swap and it is barely touched (200 MB at the last surviving point), so swap does not extend the
ceiling; the killer fires on pressure first.

Peak RSS at the surviving point is 4.26 GB against 11.4 GB of RAM, so RSS alone does not explain the
kill: **the OpenCL buffers are not counted in it**. The budget that does explain it, at
`capacity = C` and retained `S`, is 65 536 bytes per token times

- the engine's own F32 KV cache — `C`
- the reference cache `--dump aperturb` allocates beside it — `C`
- the host mirror `read` builds, for all layers at once — `S`

plus 2.5 GB of F16 weights. At `C = 9 472, S = 9 222` that is 2.5 + 0.58 + 0.58 + 0.58 ≈ 4.3 GB in
the process, matching the measured peak, with the device-side copies on top of it and outside RSS.

**8B was not attempted, for a reason that makes the attempt pointless**: F16 weights alone are
~16 GB against 11.4 GB of RAM, so the model never loads and the F32 KV question never arises. A
quantized 8B checkpoint would load, but its `W_o` digests differently and needs its own basis
file (§6).

---

## 3. Part B — the device dump agrees with the paper harness

The `l2_rms` agreement of 2.5e-7 recorded in `qcf-android-opencl-port.md` was a *host* number; the
device dump had only ever been compared against the host dump. Fed to
`~/.cache/argus-qcf/check_engine_dump.py`, which recomputes the metric from the engine's own dumped
`(q, K, V)` using the paper harness's functions:

```
python3 check_engine_dump.py models/llama3.2-1b <device>/aperturb.jsonl <device>/tensors
model: L=16 n_q=32 n_kv=8 hd=64 rank=8
worst relative disagreement:
  l2        2.455e-07   (q1/sink_recent_r25)
  l2_rms    9.595e-07   (q1/recent_r50)
```

Two questions (`n = 186`, `n = 327`), five candidates each, OpenCL backend, `--max-seq-len 1024`.
**PASS** — the same ~1e-6 band the host got, and the ranking of all five candidates is identical to
the reference on both questions.

Two notes on reading that output:

- The harness also prints `dcos` relative errors up to 1.3e+01 for the `keep_all` arm. That arm's
  value is exactly zero by construction — the identity candidate perturbs nothing — and both sides
  report ~1e-8. The ratio is a division by float noise, not a disagreement.
- `rank=8` in the header is the check that matters for the basis work: the device loaded the 1/256
  table, and `check_engine_dump.py`'s `assert rank == max(1, round(WO_FRAC * d))` accepted it.

---

## 4. Part C — the read-back cost is proportional to the *allocated* cache

### 4.1 The control

Same prompt, same everything, only `--max-seq-len` changed. Adreno, `n = 2 034`, min of three:

| `--max-seq-len` | decide | read |
|---:|---:|---:|
| 2 560 | 0.297 | **0.182** |
| 4 096 | 0.288 | **0.202** |
| 8 704 | 0.281 | **0.270** |

`decide` does not move. `read` does: `read(C) = 14.42 µs/tok · C + 144 ms` at fixed `S`, R² = 0.9994.

The desktop reproduces it more sharply, because its allocation ratio can be pushed further. Desktop
OpenCL, `llama3.2-1b`, three repetitions, min (all three values in parentheses):

| `--max-seq-len` | `S` | decide | read |
|---:|---:|---:|---:|
| 2 560 | 493 | 0.029 (0.032 / 0.035 / 0.029) | **0.028** (0.028 / 0.032 / 0.031) |
| 16 384 | 493 | 0.030 (0.031 / 0.030 / 0.032) | **0.240** (0.240 / 0.397 / 0.390) |
| 2 560 | 2 034 | 0.069 (0.074 / 0.070 / 0.069) | **0.057** (0.062 / 0.057 / 0.060) |
| 16 384 | 2 034 | 0.063 (0.077 / 0.063 / 0.073) | **0.266** (0.266 / 0.419 / 0.422) |

**8.6× on `read` at `S = 493` (12.6× on medians), 1.0× on `decide`, from an allocation the
measurement never used.** The 0.167 s that `qcf-overhead.md` §8 records for OpenCL at `S = 493` is
this: that sweep ran at `--max-seq-len 16384` throughout.

### 4.2 Why, in the code

`read` is `KVCache::host_snapshot` (`engine/src/kv/kv_cache.rs:102`) plus a dequantize.
`host_snapshot` calls `read_device_tensor_to_host` (`engine/src/kv/kv_cache.rs:73`) on `k_buffer`
and `v_buffer` whole — it allocates `let bytes = t.size()` of host memory and fills it with one
`read_buffer` — and those buffers are allocated at `capacity = max_seq_len`, never trimmed to
`current_pos` on this path. The dequantize that follows is over `current_pos` rows. So

```
read  =  fixed  +  b · capacity   (the whole-buffer host mirror)  +  c · S   (the dequantize)
```

and the ladder in §2, run at one fixed capacity, folded the middle term into what looked like an
intercept. Splitting it with the two controls gives, on Adreno: **97 ms fixed, 14.4 µs/tok ·
capacity, 23.4 µs/tok · `S`** — `b` and the fixed part from the capacity control, `c` from the
ladder.

That form is not validated by the capacity control, which is where two of its three coefficients
came from. Held out against the ladder it predicts each point 3–17% low. Some of that is the form
being approximate; some of it is that `read` is simply not reproducible between processes — the same
`(C = 8 704, S = 2 034)` point measures 0.270 s in the capacity control and 0.325 s in the ladder, a
20% gap between two runs of the same binary on the same phone. **Treat the three-term split as the
attribution it is, not as a predictive fit.** The attribution is what the next paragraph nails down,
and it does not depend on the coefficients.

**The capacity term is entirely in that mirror, and nowhere else.** A host-resident cache never
enters the branch — `read_layer_kv` takes `is_gpu_buffer()` false and dequantizes in place — so the
CPU backend is the control that isolates it. Desktop CPU backend, min of three:

| `--max-seq-len` | `S = 493` | `S = 2 034` |
|---:|---:|---:|
| 2 560 | 0.011 | 0.059 |
| 16 384 | 0.006 | 0.055 |

Flat in capacity across a 6.4× change (the small drop is noise, and is in the wrong direction for an
allocation-size explanation), and linear in `S`. Against the same runs on the OpenCL backend, where
`read` at `S = 493` goes 0.028 → 0.240, the attribution is unambiguous.

The implied bandwidth of the mirror is consistent across the two very different devices. Desktop, at
`C = 16 384`: 1.07 GB moved in `0.240 − 0.055 = 0.185 s` → **5.8 GB/s**. Adreno, from the fitted
capacity slope: `65 536 B / 14.42 µs` → **4.5 GB/s**. Two machines, one mechanism.

What this control does *not* separate is the allocation from the transfer: `read_device_tensor_to_host`
does both, and both are `O(capacity)`. That distinction does not matter for the conclusion, because
the fix in §4.4 removes both.

The `-b cpu` leg on the phone separates the copy from the dequantize instead. At
`--max-seq-len 2560`, `S = 2 034`: `read` is 0.182 s on OpenCL and 0.084 s on CPU. **The device
round trip is 0.098 s and the dequantize is 0.084 s** — the transfer is not even the larger half
once the allocation is honest.

### 4.3 What the crossover actually is

Measured, not extrapolated. Min of three throughout.

| | `S` | decide | read | decide ÷ read |
|---|---:|---:|---:|---:|
| desktop OpenCL, `C = 16 384` | 493 | 0.030 | 0.240 | 0.13× |
| desktop OpenCL, `C = 2 560` | 493 | 0.029 | 0.028 | **1.04×** |
| desktop OpenCL, `C = 2 560` | 2 034 | 0.069 | 0.057 | 1.21× |
| Adreno, `C = 8 704` | 2 034 | 0.281 | 0.270 | 1.04× |
| Adreno, `C = 2 560` | 2 034 | 0.297 | 0.182 | **1.63×** |
| Adreno, `C = 8 704` | 8 195 | 0.951 | 0.429 | 2.22× |

With the cache sized to the workload the decision overtakes the transfer at roughly **500 tokens on
the desktop and below 2 000 on Adreno** — not at 10 000. Even at the oversized 8 704 allocation the
ladder used, the Adreno fits cross at `S = 2 474`.

### 4.4 Recommendation

**Option 2 of the handoff — scope the claim and state the crossover — and it is now a much smaller
concession than it looked.**

Option 1 (add a data-movement term to the cost model) would be adding a term for an artifact: the
quantity it would model is dominated by `capacity`, which is a deployment knob, not a property of the
method. Option 3 (port `decide` to the device) is not justified by anything measured here — on
Adreno the decision is already the larger half at a proportionate allocation.

What should be said instead is the honest version: on a GPU backend the decision needs K and V on
the host; that costs a device→host copy plus a dequantize; with the cache sized to the workload it is
comparable to the decision itself and is overtaken by it within a few hundred tokens.

There is also a fourth option the handoff did not have, now that the cause is known: **mirror only
the live rows.** In HeadMajor the live rows of each head are contiguous —
`offset(pos, head) = head · capacity · head_dim + pos · head_dim` (`kv_cache.rs:241`) — so
`n_kv_heads` ranged copies per tensor would replace one whole-buffer copy and make the term `O(S)`
outright. It is not free: `Backend::read_buffer(&self, t, dst)` has no offset/length form on any
backend (`opencl.rs:4645`, `cuda_pc.rs:1967`, `cuda_embedded.rs:2787`), so this is an ABI-additive
method plus three implementations. Worth doing if the read-back ever lands on a production path; not
needed to make the paper's claim true.

---

## 5. What changes in `qcf-overhead.md`

§8's closing paragraphs are wrong as written, and its `read` column is capacity-confounded:

- "at short context that fixed cost dominates — 0.347 s against a 0.040 s decision" — 0.347 s is
  0.028–0.030 s at a proportionate allocation.
- "Only past roughly 10,000 tokens does the decision overtake the transfer" — ~500 tokens.
- "Either the model needs a data-movement term, or the decision needs to run where the cache already
  lives" — neither, per §4.4.

The §8 table should either be re-measured at a proportionate `--max-seq-len` or keep its numbers and
state the allocation, with the control above beside it. That edit is **not applied**: this repo's
rule is that `docs/**` is not committed without the maintainer saying so, and §8 is tracked.

---

## 6. Not covered

- **One handset.** `R3CY408S4HN` was not attached. Every prior on-device claim in this repo was made
  on two units; this one is not.
- **One checkpoint, one precision.** `llama3.2-1b`, BF16 safetensors as F16. GGUF/Q4_0 digests `W_o`
  differently and needs its own basis file — still untested.
- **No CUDA control.** The desktop controls are OpenCL and CPU; the CUDA leg of §8 was not re-run
  at a proportionate allocation. Its `read` column has the same shape — 0.347 → 0.364 over a 30×
  span of `S` at a fixed 16 384 allocation, i.e. nearly flat where a genuinely `S`-driven cost would
  not be — so the same explanation should apply, but that is inference, not measurement. The build
  in this worktree is `opencl`-featured and `cuda` is mutually exclusive with it, so this needs a
  second build.
- **`-b cpu` on device was run at two lengths only** (493, 2 034), enough for the §2.1 and §4.2
  splits and not for a fit.
- **`decide` under memory pressure** was not isolated. The 9 222-token point swapped 200 MB; it
  appears in §2.2 as a ceiling probe and is deliberately excluded from the §2 fit.
- **The `read` fit's R² is 0.928**, well below the others, because it mixes a capacity term that is
  constant within the ladder with an `S` term that is not. §4.2's three-term form is the one to use.

---

## Appendix — commands

The device runner (`apsweep.sh`) kills on the `wrote N record(s)` line, as `qcf-overhead.md` §5.1
does, because the eval-LL scoring that follows is not part of this measurement and takes far longer
on a phone than the dump does.

```sh
# on device, /data/local/tmp, LD_LIBRARY_PATH=.
./argus-eval --model-path models/llama3.2-1b -b opencl --kv-type f32 \
  --max-seq-len 8704 --eval-ll --eval-batch sweep_rep.json \
  --dump aperturb --dump-dir out/ --aperturb-basis wo_1b.basis
```

`sweep_rep.json` is the ladder 512/1024/2048/4096/6144/8192 repeated three times, interleaved
(`s512_r0 s1024_r0 … s8192_r0 s512_r1 …`), built from `scripts/fixtures/peekkv_needle_15714.txt` at
3.7603 chars/token exactly as `qcf-overhead.md` §4.5 describes.

Capacity control: the same one-length batch run at three `--max-seq-len` values. Ceiling probe: one
question, `--max-seq-len` just above it, with a 1 Hz sampler on `/proc/<pid>/status` `VmRSS`/`VmSwap`
and `/proc/meminfo` — the sampler is what showed that peak RSS does not explain the kill.

Reference parity (`--max-seq-len 1024`, two short questions — the tensor dump is the whole resident
cache, 14 MB and 22 MB here, so a 4k-token question would be ~250 MB):

```sh
./argus-eval ... --eval-batch eval_batch.json \
  --dump aperturb --dump-dir ap_ref --aperturb-tensor-dir ap_ref_t --aperturb-basis wo_1b.basis
# host
python3 ~/.cache/argus-qcf/check_engine_dump.py \
  models/llama3.2-1b ap_ref/aperturb.jsonl ap_ref_t
```

Thermal zone: the type string on this handset is `cpu-0-0-0` (and `gpuss-0` for the GPU), not the
`cpu-0-0-usr` that `devices.toml` names.
