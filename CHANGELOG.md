# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This
project is pre-1.0; minor releases may include breaking changes.

## [Unreleased]

### Added

- Added a synthetic **`kivi`** backend-capability plugin (`crates/techniques/kivi`, registers
  `kivi_abi`) — the last KV technique to move out, and the first non-example user of the
  backend-capability axis (`register_kivi_attention_plugin!`). It exercises the full dynamic
  C-ABI round-trip (`register_backend_caps_v2` envelope → category bridge → `DynKiviAttentionBackend`
  adapter → make/dispatch/drop) **without GPU math**: `attention_gen_kivi` writes a deterministic
  sentinel so the host gate (`gate_c_kivi_backend_cap`) can verify `KiviAttnArgs` crossed the ABI
  boundary intact. The real GPU KIVI kernels (`backend/opencl.rs`, `kernels/kivi_*.cl`) and the
  in-engine KIVI cache/format/forward path stay in the engine and are untouched — this is an
  ABI-verification artifact, not a runtime KIVI implementation. It is dlopen-only (no engine
  force-link), so `--backend-cap kivi_abi` requires `--load-plugin`; the `_abi` suffix signals it
  must not be selected on the decode hot path (it would replace the live attention backend and
  produce meaningless output). Verified host (unit + dlopen gate) and on-device on Adreno OpenCL
  (cross-built `.so` loads, resolves, and is selected at construction; absent `--load-plugin` it
  fail-fasts with `Unknown --backend-cap`).

### Changed

- **Host default backend is now `cuda` on a CUDA build.** `argus-cli` / `argus-bench` /
  `argus-chat` / `argus-eval` share one `--backend`/`-b` default that is now three-way: an Android
  target still defaults to `opencl`; a host built with `--features cuda` (or `cuda-embedded`) now
  defaults to **`cuda`**; every other host build still defaults to `cpu`. This matches the AUF
  primary-variant default, which already selects `CudaAos` under `cuda`/`cuda-embedded`. To run on
  the CPU from a CUDA build, pass `--backend cpu` explicitly — the CUDA init error now also points
  at that flag. Non-CUDA builds are unaffected.

- Added a plugin-declared argument channel to `argus-extension-api`. `KVCacheStageReg` gained
  `make_with_args(StageParams, StageArgs)` alongside `make`, where `StageArgs = &[PluginArg]` is an
  opaque `key=value` blob the engine routes **without knowing any technique's private params**. The
  `d2o` plugin now parses its own knobs (`target_ratio`/`ema_beta`/`merge_e`/`layer_alloc`/
  `protected_layers`/`merge_axis`) in `D2OConfig::from_args`, and `eval`/`bench`/`chat` resolve d2o
  through the generic `make_stage_with_args("d2o", &params, &blob)` path instead of hard-constructing
  `D2OConfig` — the engine no longer references d2o's config type (the inversion is gone). This fixes
  two divergences: the chat path now forwards `--merge-axis` (previously dropped) and `argus-bench`
  now honors explicit `eviction d2o` flags (previously hard-coded; default runs are unchanged).
  `register_kv_stage!`-based plugins are unchanged (the macro auto-wires an args-ignoring shim), and
  the dynamic `.so` stage C-ABI is unchanged — no out-of-tree stage plugin needs private args yet, so
  that ABI extension is deferred until one does.

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
  an engine-core `kv::dequant` module. (KIVI follows — see the `kivi` plugin under Added.)

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
