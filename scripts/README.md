# scripts

Model-preparation tooling for Argus.

## Python setup

```bash
pip install -r scripts/requirements.txt   # numpy, safetensors
```

## `convert_safetensors_to_gguf.py`

Convert a HuggingFace Safetensors model directory to a GGUF file (Q4_0 hybrid or
F16), compatible with both Argus and llama.cpp.

```bash
# Q4_0 (default): 2D weights quantized, embeddings F16, norms F32
python scripts/convert_safetensors_to_gguf.py models/qwen2.5-1.5b out.gguf

# F16
python scripts/convert_safetensors_to_gguf.py --outtype f16 models/qwen2.5-1.5b out-f16.gguf
```

A `tokenizer.json` should sit next to the model so it can be carried along.

## `convert_to_auf.sh`

One-shot build of an **AUF** (Argus Unified Format) asset from Safetensors *or* an
existing GGUF. It runs the two stages for you: Safetensors → GGUF (via the script
above) → AUF (via the `auf_tool` binary, which it builds with
`cargo build --release -p argus-engine --bin auf_tool` if needed).

```bash
# Safetensors → AUF (most common)
scripts/convert_to_auf.sh --input models/llama-3.2-1b/ --output models/llama-3.2-1b.auf

# Already have a GGUF? Stage 1 is skipped.
scripts/convert_to_auf.sh --input models/llama-3.2-1b-q4_0.gguf --output models/llama-3.2-1b.auf

# Multi-dtype AUF (e.g. q4_0 + f16 variants in one file)
scripts/convert_to_auf.sh --input models/llama-3.2-1b/ --output out.auf --dtypes q4_0,f16
```

Run `scripts/convert_to_auf.sh --help` for the full flag list (`--outtype`,
`--variants`, `--include-lm-head`, `--dtypes`, `--default-dtype`, `--tokenizer`,
`--keep-gguf`, `--created-by`, `--quiet`).

The underlying `auf_tool` binary can also be used directly:

```bash
cargo run --release --bin auf_tool -- build \
    --input model.gguf --tokenizer tokenizer.json --output model.auf --variants all
```

## Device runner — `run_device.py` + `device_registry/`

Cross-compile, push, and run engine binaries on a registered device (adb for
Android, ssh for Jetson, or the local host). Two config files drive it; copy the
committed templates and edit them (your copies are gitignored):

```bash
cp hosts.toml.example hosts.toml        # per-host toolchain (NDK path, etc.)
cp devices.toml.example devices.toml    # device registry (connection, target, paths)
```

Or auto-probe:

```bash
python scripts/device_registry.py bootstrap-host   # write a hosts.toml skeleton for this machine
python scripts/device_registry.py discover         # detect attached adb devices
```

Then:

```bash
python scripts/run_device.py --list-devices
python scripts/run_device.py -d android argus-cli \
    --model-path /data/local/tmp/models/model.gguf -b opencl --prompt "Hello"

# Deploy without executing (e.g. extra binaries):
python scripts/run_device.py -d android --skip-exec argus-cli --extra-bin test_backend
```

`hosts.toml` and `devices.toml` are gitignored; only the `.example` templates are
committed.

## Evaluation — `eval_ll_batched.py`

Run `argus-eval --eval-ll` over a batch of questions in fixed-size chunks,
restarting the process between chunks so the GPU/OpenCL driver releases its state
(works around drivers that accumulate deferred allocations over a long run).

```bash
python3 scripts/eval_ll_batched.py \
    --binary ./target/release/argus-eval \
    --model-path models/model.gguf \
    --eval-batch questions.json \
    --output eval_out.json \
    --chunk-size 8 \
    -- --backend opencl --kv-type f32 --greedy
```

Args after `--` are forwarded to `argus-eval` verbatim.
