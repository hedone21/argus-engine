# Argus Engine

[![CI](https://github.com/hedone21/argus-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/hedone21/argus-engine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue.svg)](Cargo.toml)
[![Release](https://img.shields.io/github/v/release/hedone21/argus-engine)](https://github.com/hedone21/argus-engine/releases)

**English** | [한국어](README.ko.md)

**On-device LLM inference engine for ARM64 edge & mobile devices**, written in Rust.

Argus targets Android/Linux ARM64 SoCs with a flexible backend abstraction and a
zero-copy memory architecture. It runs Llama-family and Qwen/Gemma models from
HuggingFace Safetensors and GGUF, with Q4_0/Q8_0 block quantization and OpenCL /
CUDA GPU acceleration.

This repository is the **engine**. It is one of three Argus repositories:

| Repo | Role |
|------|------|
| [`argus-engine`](https://github.com/hedone21/argus-engine) | LLM inference engine (this repo) |
| [`argus-shared`](https://github.com/hedone21/argus-shared) | IPC protocol types (manager ↔ engine) |
| [`argus-manager`](https://github.com/hedone21/argus-manager) | System resource manager service |

## Key features

- **ARM64 optimized** — NEON + dotprod intrinsics for Android/Linux ARM64 SoCs.
- **Zero-copy memory** — `CL_MEM_ALLOC_HOST_PTR` / DMA-BUF maps GPU buffers to CPU
  pointers on UMA SoCs, eliminating CPU↔GPU memcpy.
- **Pluggable backends** — `Backend` trait over CPU (NEON) / OpenCL (Adreno) / CUDA.
- **Quantization** — Q4_0 / Q8_0 block quant, F16/BF16. GGUF pre-quantized models
  load directly; Safetensors F16/BF16 convert on load.
- **KV-cache eviction** — Sliding Window / H2O / H2O+ / D2O (merge compensation) /
  StreamingLLM, as composable `KVCacheStage` plugins.
- **KIVI KV-cache quantization** — dynamic Q4/Q8 KV quantization to cut memory.
- **Flash attention** — GQA-aware GPU flash attention (strided).
- **Tensor partition** — split FFN gate/up matmul across GPU + CPU concurrently.
- **Adaptive resilience** — optional integration with `argus-manager` for runtime
  adaptation under memory/thermal pressure (eviction, backend switch, throttle).
- **Zero-compile extension surface** — add a KV-cache stage / format / read-stage as
  a separate crate that self-registers via `linkme`, with no engine-core edits.
  See [`CONTEXT.md`](CONTEXT.md) for the three orthogonal axes (stage ⊥ format ⊥
  hardware) and [`crates/technique-api/README.md`](crates/technique-api/README.md) for a
  how-to walkthrough.

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
[`argus-shared`](https://github.com/hedone21/argus-shared) as a git dependency, so it
is not published to crates.io (`cargo install argus-engine` will not work). Build it
from a git checkout.

```bash
git clone https://github.com/hedone21/argus-engine.git
cd argus-engine
cargo build --release

# CPU
./target/release/argus_cli -m model.gguf --prompt "Hello" -n 50 -b cpu

# GPU (OpenCL, Adreno production path)
./target/release/argus_cli -m model.gguf --prompt "Hello" -n 50 -b opencl
```

GGUF is the recommended model format (dtype auto-detected, no load-time conversion).
A `tokenizer.json` must sit next to the `.gguf` file, or pass `--tokenizer-path`.

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
python scripts/run_device.py -d android argus_cli \
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

## Documentation

- [`CONTEXT.md`](CONTEXT.md) — domain glossary: the stage ⊥ format ⊥ hardware axes and
  the cache-management vocabulary.
- [`AGENTS.md`](AGENTS.md) — guide for AI coding agents and contributors.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your
option. Unless you state otherwise, contributions are dual licensed as above.

Portions of this engine are adapted from [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
and [jquesnelle/yarn](https://github.com/jquesnelle/yarn) (both MIT). See
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md) for full attribution.
