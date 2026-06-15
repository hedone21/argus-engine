# Argus Engine

[![CI](https://github.com/hedone21/argus-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/hedone21/argus-engine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue.svg)](Cargo.toml)
[![Release](https://img.shields.io/github/v/release/hedone21/argus-engine)](https://github.com/hedone21/argus-engine/releases)

**English** | [한국어](README.ko.md)

**Run quantized Llama / Qwen / Gemma models on a phone or edge ARM64 GPU — and swap
the KV-cache precision format with a flag, no engine rebuild.**

Argus is an on-device LLM inference engine in Rust for Android/Linux ARM64 SoCs: NEON
CPU and Adreno-OpenCL / CUDA GPU backends, a zero-copy UMA memory path, and a plugin
surface for KV-cache and precision research.

> **Status: early.** Adreno / ARM64 is the primary tested path. The shipped `argus-cli`
> does **single-prompt text generation** (prompt in → continuation + a `Decode: X ms/tok`
> line out), can load a **KV-cache precision-format plugin** (`--kv-format`), and runs
> **score-free KV-cache eviction** (Sliding / StreamingLLM + `--load-plugin` stages,
> KV-fill-triggered). Score-based eviction (H2O / D2O), **KIVI** KV quantization, tensor
> partition, and runtime weight swap are implemented and tested, but in v0 they run
> through the `argus-bench` / `argus-eval` binaries rather than `argus-cli`. A multi-turn
> **chat** server (`argus-chat`, OpenAI-compatible HTTP API) ships alongside the CLI
> (`--interactive` REPL streams tokens as they generate); `--profile` is **planned for v1**.
> The [Roadmap](#roadmap) table says exactly where each feature runs today.

## Demo

<!-- TODO: record an asciinema/GIF of an on-device Adreno decode showing the token
     stream + the `Decode: X ms/tok` line, captured WITHOUT --profile (per the repo's
     performance-measurement rule), and embed it here (export to GIF/animated SVG —
     GitHub strips asciinema's <script> tag). -->

_A recorded on-device decode (asciinema / GIF) will go here — token streaming plus the
`Decode: X ms/tok` line that is the engine's headline metric._

## Quickstart — what runs today

```bash
git clone https://github.com/hedone21/argus-engine.git
cd argus-engine
cargo build --release

# 1. Single-prompt generation on CPU (default host backend)
./target/release/argus-cli -m model.gguf --prompt "Hello" -n 50 -b cpu

# 2. Same prompt on the Adreno OpenCL GPU — one flag switches the backend
./target/release/argus-cli -m model.gguf --prompt "Hello" -n 50 -b opencl

# 3. Sampling controls
./target/release/argus-cli -m model.gguf --prompt "Hello" -n 50 \
    --temperature 0.8 --top-p 0.9 --top-k 40 --repetition-penalty 1.1

# 4. Load a KV-cache *precision format* plugin at runtime — no engine rebuild
#    (.so is the Linux/Android name; a macOS host build produces .dylib)
cargo build --release -p example-kv-format --features plugin-cdylib
./target/release/argus-cli -m model.gguf --prompt "Hello" -n 50 \
    --load-plugin target/release/libexample_kv_format.so \
    --kv-format example_kv_format

# 5. Multi-turn chat — OpenAI-compatible HTTP server (POST /v1/chat/completions)
./target/release/argus-chat -m model.gguf --listen 127.0.0.1:8080
#    then, from another shell (streaming SSE shown; drop "stream" for one JSON reply):
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"argus","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Each run prints the generated continuation plus `TTFT`, `Decode: X ms/tok`, and
`Avg TBT` lines. Point it at a `.gguf` and the dtype is auto-detected (no conversion); a
`tokenizer.json` must sit next to the model, or pass `--tokenizer-path`. CUDA, Android
cross-compile, and Safetensors → GGUF / AUF conversion are under
[Install / Build from source](#install--build-from-source).

Step 4 is the precision **format** plugin path — a loaded `.so` reaches the real decode
path on `argus-cli` today. Score-free KV-cache **eviction** stages (`eviction
sliding|streaming`, plus `--load-plugin` stages) also run on `argus-cli`; score-based
H2O / D2O (which need the attention-score accumulator) and **KIVI** precision packing run
through `argus-bench` / `argus-eval` in v0; see the [Roadmap](#roadmap).

## What you can do

**On-device & fast**

- **ARM64-optimized** — NEON + dotprod intrinsics for Android/Linux ARM64 SoCs; AVX2 +
  FMA on x86_64 hosts.
- **Zero-copy UMA memory** — `CL_MEM_ALLOC_HOST_PTR` / DMA-BUF maps GPU buffers to CPU
  pointers on unified-memory SoCs, eliminating CPU↔GPU memcpy.
- **GPU flash attention** — GQA-aware, strided.
- **Quantized weights** — Q4_0 / Q8_0 block quant, F16 / BF16. GGUF loads directly
  (dtype auto-detected); Safetensors F16/BF16 convert on load.

**Memory-adaptive KV cache** *(score-free eviction runs on `argus-cli`; score-based
H2O/D2O + KIVI via `argus-bench` / `argus-eval` in v0 — see [Roadmap](#roadmap))*

- **Eviction stages** — Sliding Window / H2O / H2O+ / D2O (merge compensation) /
  StreamingLLM, as composable `KVCacheStage` plugins. Sliding / StreamingLLM and
  `--load-plugin` stages run on `argus-cli` (single-prompt, KV-fill-triggered).
- **KIVI KV quantization** — dynamic Q4/Q8 KV packing to cut cache memory.
- **Adaptive resilience** *(optional; `resilience` feature + `argus-manager`)* — runtime
  adaptation under memory/thermal pressure (eviction, backend switch, throttle).

**Extensible** *(zero engine-core edits)*

- **Pluggable KV cache & precision** — add a KV-cache stage / format / read-stage as a
  separate crate that self-registers via `linkme`, or load it at runtime as a `.so`.
  Three orthogonal axes — **stage** ⊥ **format** ⊥ **hardware**. The precision **format**
  axis (`--kv-format`) and the **read** axis (`--read-stage`) work from `argus-cli`
  today. See [Extending Argus](#extending-argus).
- **Pluggable backends** — `Backend` trait over CPU (NEON) / OpenCL (Adreno) / CUDA.

## Why Argus / how it relates to llama.cpp

Argus reuses kernels adapted from [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
(see [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES.md)) and is **complementary, not a
replacement**. Where llama.cpp is portable inference across many platforms, Argus is
tuned for **Adreno / ARM64 UMA edge devices** and adds a **zero-compile plugin surface
for KV-cache and precision research**: swap an eviction stage or a KV precision format by
name (**stage** ⊥ **format** ⊥ **hardware**), with no engine recompile. If you want to
prototype KV-cache or quantization techniques on a phone-class GPU, that extension
surface is the reason to reach for Argus.

## Roadmap

Where each capability runs today (v0). The **v1 plan** column marks features slated to
reach `argus-cli` (or ship as a new binary) next.

| Capability | `argus-cli` | `argus-bench` / `argus-eval` | v1 plan |
|---|:---:|:---:|:---:|
| Single-prompt generation (CPU / OpenCL / CUDA) | ✅ | ✅ | |
| Sampling (temperature / top-p / top-k / rep-penalty) | ✅ | ✅ | |
| KV-cache precision-format plugin (`--kv-format`, `--read-stage`) | ✅ | ✅ | |
| Prefix-cache save / restore | ✅ | | |
| KV-cache eviction — score-free (Sliding / StreamingLLM / `--load-plugin` stage) | ✅ | ✅ | |
| KV-cache eviction — score-based (H2O / H2O+ / D2O) | | ✅ | ✅ |
| KIVI KV quantization (`--kv-mode kivi`) | | ✅ | ✅ |
| Tensor partition (FFN split across GPU + CPU) | | ✅ | ✅ |
| Runtime weight swap | | ✅ | ✅ |
| Per-op profiling (`--profile`) | | | ✅ |
| Interactive chat REPL | | | ✅ |

> Multi-turn chat now ships as `argus-chat` — an OpenAI-compatible HTTP server
> (`POST /v1/chat/completions`, streaming + non-streaming) supporting all three KV modes
> (Standard / KIVI / Offload) and manager-integrated resilience. `--interactive` runs a
> local stdin REPL instead.

## Supported models & hardware

### Models

| Family | Architectures | Source format | Quantization |
|--------|---------------|---------------|--------------|
| Llama  | Llama-family (`LlamaForCausalLM`) | GGUF, Safetensors | Q4_0, Q8_0, F16, BF16 |
| Qwen   | Qwen2 / Qwen2.5 | GGUF, Safetensors | Q4_0, Q8_0, F16, BF16 |
| Gemma  | Gemma / Gemma 2 / Gemma 3 (text) | GGUF, Safetensors | Q4_0, Q8_0, F16, BF16 |

GGUF is recommended (dtype auto-detected, no load-time conversion); Safetensors F16/BF16
convert on load.

### Hardware backends

| Backend | Build | Hardware / target |
|---------|-------|-------------------|
| CPU (NEON + dotprod) | default | ARM64 — Android / Linux |
| CPU (AVX2 + FMA) | default | x86_64 — Linux (host / dev) |
| OpenCL | default (`opencl`) | Adreno GPU — Android ARM64 (production path) |
| CUDA | `--no-default-features --features cuda` | NVIDIA discrete GPU / Jetson Orin |
| CUDA (embedded UMA) | `--features cuda-embedded` | Jetson Xavier |

Cross-compile targets: `aarch64-linux-android`, `aarch64-unknown-linux-gnu`,
`aarch64-unknown-linux-musl`, `x86_64-unknown-linux-gnu` (see `.cargo/config.toml`).

## Prerequisites

- **Rust** (stable): `rustup install stable`.
- **OpenCL headers** — the default build enables the `opencl` feature. On Linux:
  `sudo apt-get install ocl-icd-opencl-dev`. macOS ships the OpenCL framework. A GPU
  is *not* required to build (only to run GPU backends).

## Install / Build from source

Argus is distributed as source: it depends on
[`argus-shared`](https://github.com/hedone21/argus-shared) as a git dependency, so it is
not published to crates.io (`cargo install argus-engine` will not work). Build it from a
git checkout — the [Quickstart](#quickstart--what-runs-today) covers the default
`cargo build --release` + CPU/OpenCL run.

```bash
# CUDA (NVIDIA discrete GPU / Jetson) — mutually exclusive with opencl
cargo build --release --no-default-features --features cuda
```

### Model conversion

Argus loads GGUF directly. To produce a GGUF from a HuggingFace Safetensors model — or
to build an AUF (Argus Unified Format) asset — use the tools in [`scripts/`](scripts/):

```bash
pip install -r scripts/requirements.txt

# Safetensors → GGUF (Q4_0 by default)
python scripts/convert_safetensors_to_gguf.py models/qwen2.5-1.5b out.gguf

# Safetensors or GGUF → AUF (one shot; builds the auf_tool binary as needed)
scripts/convert_to_auf.sh --input models/qwen2.5-1.5b/ --output model.auf
```

See [`scripts/README.md`](scripts/README.md) for the full conversion guide.

### Android (cross-compile + deploy)

Build for `aarch64-linux-android` with the Android NDK. The Adreno production path is
`-b opencl`; the device needs a vendor `libOpenCL.so` (not distributed here — pull it
from the device's `/vendor/lib64`).

`scripts/run_device.py` automates build → push → run over adb (and ssh for Jetson),
driven by two local config files (templates are committed; your filled-in copies are
gitignored):

```bash
cp hosts.toml.example hosts.toml        # set your NDK path
# or: python scripts/device_registry.py bootstrap-host   # auto-probe the NDK
cp devices.toml.example devices.toml    # register your device(s)
# or: python scripts/device_registry.py discover         # auto-probe attached adb devices

python scripts/run_device.py --list-devices
python scripts/run_device.py -d android argus-cli \
    --model-path /data/local/tmp/models/model.gguf -b opencl --prompt "Hello"
```

See `.cargo/config.toml` for the raw target flags and [`scripts/README.md`](scripts/README.md)
for the device-runner and evaluation tooling.

## Cargo features

| Feature | Default | Description |
|---------|---------|-------------|
| `opencl` | ✅ | OpenCL GPU backend (Adreno) |
| `profile` | ✅ | Per-op profiling instrumentation |
| `cuda` | | CUDA backend (discrete GPU / Jetson) |
| `cuda-embedded` | | CUDA for embedded UMA (Jetson Xavier) |
| `resilience` | | D-Bus IPC integration with `argus-manager` |
| `caote` | | CAOTE value-aware eviction plugin |
| `rkv` | | R-KV joint eviction measurement prototype (experimental, default-off) |

> `opencl` and `cuda`/`cuda-embedded` are mutually exclusive — select exactly one GPU
> backend. The CUDA build above drops the default `opencl` via `--no-default-features`
> and adds `--features cuda`. Building with **no** GPU backend at all is not currently
> supported.
>
> The `profile` feature compiles the per-op instrumentation, but the shipped v0
> `argus-cli` rejects the `--profile` flag (profiling output is read from `argus-bench`);
> CLI profiling is planned for v1.

## Extending Argus

A KV-cache technique is a **separate crate** that depends only on `technique-api` +
`linkme` and **self-registers** — adding one touches **zero engine-core code**. There are
three orthogonal axes (**stage** ⊥ **format** ⊥ **hardware**): a **stage** adjusts which
tokens stay resident (eviction/merge), a **format** defines the KV byte layout
(precision), and a **read-stage** chooses what to read back.

### Example: a KV-cache precision plugin

This is the plugin loaded in [Quickstart](#quickstart--what-runs-today) step 4 — copy
`crates/techniques/example-kv-format/` as a template. The whole plugin is one small crate:

```
crates/techniques/my-kv-format/
├── Cargo.toml
└── src/lib.rs
```

`Cargo.toml` — two dependency lines plus a `cdylib` so it can build as a loadable `.so`:

```toml
[package]
name = "my-kv-format"
# version / edition / license inherited from the workspace

[lib]
crate-type = ["cdylib", "rlib"]   # cdylib = the .so; rlib = static force-link

[dependencies]
technique-api = { path = "../../technique-api" }
linkme = "0.3"

[features]
plugin-cdylib = []                # gates the .so C-ABI export
```

`src/lib.rs` — implement the trait (here `KVFormat` = a name + a byte-layout descriptor)
and register it in one line:

```rust
use technique_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

struct MyKvFormat;

impl KVFormat for MyKvFormat {
    fn name(&self) -> &str { "my_kv_format" }   // the --kv-format selector
    fn layout(&self) -> KVLayoutDesc {          // q4_0-like: 32-elem blocks, 4-bit nibbles
        KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        }
    }
}

// One line dual-wires it: static (linkme registry) + dynamic (the `.so` C-ABI export).
technique_api::register_kv_format!("my_kv_format", || Box::new(MyKvFormat));
technique_api::export_plugin!();                // emits the .so entry point for --load-plugin
```

Build it as a `.so` and load it by name — no engine rebuild:

```bash
cargo build --release -p my-kv-format --features plugin-cdylib
./target/release/argus-cli -m model.gguf --prompt "Hello" -n 50 \
    --load-plugin target/release/libmy_kv_format.so --kv-format my_kv_format
```

At startup the engine `dlopen`s the `.so`, registers the format, and `--kv-format
my_kv_format` routes the KV cache through your layout (the `[DecodeLoop] kv storage =
OPAQUE` log line confirms it reached the decode path). Drop `--features plugin-cdylib`,
add a one-line path dependency on the crate instead, and the same code is force-linked
statically — identical `--kv-format my_kv_format`, no `.so`.

### The other axes

- **Stage** (eviction/merge) — implement `KVCacheStage::plan(&ctx) -> Option<KVCachePlan>`
  returning which tokens to `keep` / `merge`, register with `register_kv_stage!`, select
  with `eviction plugin --name <name>`. Template: `example-keep-recent`. *Score-free stages
  run on `argus-cli` (`--load-plugin <.so> eviction plugin --name <name>`, KV-fill-triggered);
  score-based stages that need attention scores run via `argus-bench` / `argus-eval`.*
- **Read** (query-aware read) — implement `KVReadStage`, select with `--read-stage <name>`.
  Reference: the `quest` builtin.

A `KVFormat` contributes a byte-layout *descriptor*, not a compute kernel — a precision
with no matching builtin rides a generic dequant→f32 path. See [`CONTEXT.md`](CONTEXT.md)
for the axis vocabulary, **[`docs/plugins.md`](docs/plugins.md)** for the full onboarding
guide (quickstart through shipping a plugin), and
[`crates/technique-api/README.md`](crates/technique-api/README.md) for the terse
registration reference and the `example-*` templates (bundles, multi-format, rollback /
error vehicles).

## Repository map

This repository is the **engine**. It is one of three Argus repositories:

| Repo | Role |
|------|------|
| [`argus-engine`](https://github.com/hedone21/argus-engine) | LLM inference engine (this repo) |
| [`argus-shared`](https://github.com/hedone21/argus-shared) | IPC protocol types (manager ↔ engine) |
| [`argus-manager`](https://github.com/hedone21/argus-manager) | System resource manager service |

## Documentation

- [`docs/plugins.md`](docs/plugins.md) — **Writing an Argus plugin**: the developer
  onboarding guide for adding a KV-cache technique (stage / format / read / backend-cap)
  without forking the engine.
- [`CONTEXT.md`](CONTEXT.md) — domain glossary: the stage ⊥ format ⊥ hardware axes and
  the cache-management vocabulary.
- [`AGENTS.md`](AGENTS.md) — guide for AI coding agents and contributors.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your
option. Unless you state otherwise, contributions are dual licensed as above.

Portions of this engine are adapted from [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
and [jquesnelle/yarn](https://github.com/jquesnelle/yarn) (both MIT). See
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md) for full attribution.
