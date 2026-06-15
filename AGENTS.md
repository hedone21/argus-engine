# Argus Engine — Agent Guide

Guidance for AI coding agents (Claude Code, Codex, Cursor, ...) and contributors
working in this repository.

## Project overview

`argus-engine` is a high-performance on-device LLM inference engine in Rust,
targeting ARM64 Android/edge devices. It loads Llama/Qwen/Gemma models (Safetensors
or GGUF) with Q4_0/Q8_0 quantization and OpenCL/CUDA acceleration.

**Inference flow:** Prefill (batched tokens) → Decode (token by token). Per layer:
RMSNorm → QKV matmul → RoPE → KV-cache update → Attention → FFN. The unified forward
pass is `forward_into()`; KV eviction is the caller's responsibility via the cache
manager.

**Domain vocabulary** lives in [`CONTEXT.md`](CONTEXT.md): three orthogonal axes —
**stage** (resident data: evict/merge/swap) ⊥ **format** (precision/layout: f16/q4_0/
KIVI) ⊥ **hardware** (compute location: CPU/OpenCL/CUDA). "Layer" means a transformer
decoder block only; storage representation is called a **Format**, never a "Layer".

## Working agreement

- **Think before coding.** State assumptions; surface trade-offs; if a simpler
  approach exists, say so. If something is ambiguous, ask.
- **Simplest thing that works.** No speculative abstraction or unrequested
  configurability. Minimal code to solve the problem.
- **Surgical changes.** Touch only what the task requires. Don't reformat or refactor
  unrelated code. Match the surrounding style.
- **Goal-based.** Turn the task into a verifiable goal (a failing test that should
  pass) and iterate until it's met.

## Build & test

```bash
cargo build --release          # default features: opencl + profile
cargo test --workspace         # host unit + integration tests
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

GPU correctness and on-device tests require an actual device/GPU and are run manually,
not in CI. CI builds the OpenCL-inclusive feature matrix on Linux (no GPU needed to
build).

## Conventions

- **Commits:** Conventional Commits — `type(scope): subject`, imperative mood. Types:
  `feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert`. Do **not**
  add `Co-Authored-By` trailers (including AI/agent co-authors) to commit messages.
- **`.cl` kernels:** avoid editing by default; performance work may. Adreno lessons:
  DK=128 flash-attn register spill above 32 float4/thread; `sub_group_reduce_*` is
  slower than SLM tree-reduce on Adreno; always compare engines by wall-clock.
- **Performance measurement:** measure *without* `--profile`. The v0 `argus-cli`
  intentionally rejects `--profile` (and other non-happy-path flags); production TBT is
  read from the `Decode: X ms/tok` log line. (`--profile` adds ~54 ms/token of sync
  overhead, so even when re-enabled it is valid only for *relative* per-op comparison.)

## Extension architecture

Adding a KV-cache stage / format / read-stage does **not** require editing the engine
core. Create a crate under `crates/techniques/`, implement the trait, and self-register
via `linkme`. See the `example-*` crates for templates and `technique-api` for the
registration surface.

## License

Contributions are dual licensed `MIT OR Apache-2.0`.
