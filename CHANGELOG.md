# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This
project is pre-1.0; minor releases may include breaking changes.

## [Unreleased]

### Changed

- Renamed the extension-API crate `technique-api` → `argus-extension-api` (import
  `argus_extension_api`); the name now states what it is (the public plugin/extension surface).
- Extracted the **StreamingLLM** and **H2O** KV-cache eviction policies out of the engine core
  into self-registering technique crates (`crates/techniques/streaming-llm`,
  `crates/techniques/h2o`), following the `caote`/`quest` precedent. They depend only on
  `argus-extension-api` + `linkme`, implement `KVCacheStage`, and are force-linked so
  `eviction streaming` / `eviction h2o` resolve the out-of-tree plugins with no built-in copy.
  Behaviour is unchanged: the plugin keep-lists are byte-identical to the former built-ins
  (proven by `beta3_eviction_stage_equivalence` across F32/F16/Q4_0) and verified on-device on
  Adreno OpenCL.
- Extracted **D2O** (Dynamic Discriminative Operations) into `crates/techniques/d2o`. Unlike
  StreamingLLM/H2O it produces `WeightedMerge`s (cosine-nearest + Eq.11 weights, EMA threshold);
  the engine executor `apply_weighted_merges` applies them (already proven bit-identical to the
  former in-place scatter-reduce). The in-place `D2OHandler` is gone — production now resolves
  `d2o` through the same `StageBackedPolicy` path as the other techniques. `StageCtx` gained
  `layer_idx`/`n_layers` (so the plugin honours `protected_layers` / last-layer protection) and
  `kv_on_device` (device-only buffers degrade to a keep-only plan, preserving the former GPU
  fallback). Shared dequant helpers (`dequantize_k`/`dequantize_v`/`cosine_similarity`) moved to
  an engine-core `kv::dequant` module. (KIVI extraction to follow.)

### Removed

- Repo-wide dead-code sweep (liveness-first triage + adversarial verification of every
  candidate). Removed only code with no live driver: the never-mounted D2O per-layer
  *layer-alloc* budget machinery (`kv::d2o_layer_alloc`, `VarianceObserver`,
  `OffloadForwardArgs.variance_collector`, `CacheManager::force_evict_with_scores_and_budgets`,
  `HandlerContext.layer_ratios`); dead KIVI `QuantizedBlocks` accessors; write-only/test-only
  helpers (`KVCacheSnapshot.capacities`, `OffloadFormat::reset_session_locked`, a test-only
  `hm_read_v`, `RpcmemLayerRegion.size`); dead backend code (the deprecated OpenCL
  `compute_scores_gpu` + its kernel slot, two write-only `PartitionPlanContext` buffer handles,
  `HybridGpuBuffer.elem_count`, a stale x86 `matmul_transposed_q4_0_serial`); and dead code in
  the `auf_tool`/`test_backend` dev binaries. The `--d2o-layer-alloc` flag is **kept** — its one
  live effect (last-layer protection in the d2o plugin) is independent of the removed budget
  machinery, so behaviour is unchanged. Deliberately-staged scaffolding (KIVI entry points,
  Phase 2-B / Sprint-2c / v2 fields) was kept; stale "unwired" doc-comments on the now-live
  format-axis modules were corrected.

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
  edits (see `crates/argus-extension-api`).
- Model conversion tooling (Safetensors → GGUF, and → AUF) under `scripts/`.

[Unreleased]: https://github.com/hedone21/argus-engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hedone21/argus-engine/releases/tag/v0.1.0
