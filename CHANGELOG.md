# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This
project is pre-1.0; minor releases may include breaking changes.

## [Unreleased]

## [0.1.0] - 2026-06-14

Initial public release.

### Added

- On-device LLM inference engine for ARM64 edge / mobile devices, written in Rust.
- Model loading: Llama-family and Qwen/Gemma from HuggingFace Safetensors and GGUF, with
  Q4_0/Q8_0 block quantization and F16/BF16.
- Pluggable backends behind a `Backend` trait: CPU (NEON + dotprod), OpenCL (Adreno), and
  CUDA (discrete GPU / Jetson); zero-copy UMA memory via `CL_MEM_ALLOC_HOST_PTR` / DMA-BUF.
- KV-cache eviction stages (Sliding Window / H2O / H2O+ / D2O / StreamingLLM) and KIVI
  KV-cache quantization.
- GQA-aware GPU flash attention; FFN gate/up tensor partition across GPU + CPU.
- Zero-compile extension surface: KV-cache stage / format / read-stage techniques
  self-register via `linkme` from crates under `crates/techniques/` with no engine-core
  edits (see `crates/technique-api`).
- Model conversion tooling (Safetensors → GGUF, and → AUF) under `scripts/`.

[Unreleased]: https://github.com/hedone21/argus-engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hedone21/argus-engine/releases/tag/v0.1.0
