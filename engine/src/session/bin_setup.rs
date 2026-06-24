//! 추론 bin 공용 셋업 (argus_cli / argus_bench 공유).
//!
//! `SessionInitCtx::build` → tokenizer resolve/load → prompt encode →
//! KV cache 할당 → resilience adapter 까지 묶어 [`StandardHappyCtx`] 를 만든다.
//! 각 bin 의 `main` 은 reject 가드 + dispatch (run_standard_happy_path /
//! run_experiment_path) 만 담당한다.

use std::sync::Arc;

use anyhow::bail;
use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::capability::quant_attn::QuantAttnBackend;
use crate::hardware::Hardware;
use crate::inference::sampling::SamplingConfig;
use crate::kv::kv_cache::{KVCache, KVLayout};
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::session::cli::Args;
use crate::session::init::SessionInitCtx;
use crate::session::resilience_adapter::ResilienceAdapter;
use crate::session::resilience_init::build_command_executor;
use crate::session::standard_happy::StandardHappyCtx;
use crate::shape::Shape;
use crate::tensor::Tensor;
use argus_extension_api::{
    KVFormatPolicy, KVLayoutDesc, StageCtx, StageParams, TensorHandle, TensorKind,
    find_format_policy,
};

/// `build_inference_ctx` / `build_quant_window_bench_ctx` 공통 prelude 산출물 (AB-2 §5.7.7).
///
/// init(`SessionInitCtx`) → tokenizer resolve/load → prompt encode → token 까지 공통부.
/// Standard 는 이 뒤에 `Vec<KVCache>` 할당을, quant-window 는 caps pull + `Vec<QuantizedRecentWindowCache>` 할당을 한다.
pub struct InferencePrelude {
    pub init: SessionInitCtx,
    pub tokenizer: Tokenizer,
    pub tokens: Vec<u32>,
}

/// `build_inference_ctx` / `build_quant_window_bench_ctx` 공통 prelude 조립 (AB-2 §5.7.7).
///
/// plugin dlopen + fat-LTO self-test + `SessionInitCtx::build` + tokenizer/prompt/token 까지.
/// caps 보존을 위해 `SessionInitCtx` 전체를 [`InferencePrelude`] 로 반환한다(quant-window 는 caps 가
/// `caps.get::<dyn QuantAttnBackend>()` pull 에 필요).
pub fn build_inference_prelude(args: &Args) -> anyhow::Result<InferencePrelude> {
    // GATE-C: --load-plugin 의 `.so` 들을 .so 당 1회 dlopen 해 stage+format 양축
    // capability 를 등록한다(cross-axis open-once dispatcher — 번들/단일축 `.so` 모두 흡수). 이후
    // make_stage(eviction <policy>)/make_format(--kv-format)가 정적(linkme)+동적(여기) 통합 조회로
    // 해소한다. 봉투 abi_version mismatch / 이름 충돌 / capability-0 은 여기서 fail-fast.
    crate::session::plugin_dispatch::register_dynamic_plugins(&args.load_plugin)?;
    // fat-LTO self-test(C3 배선): 내장 KV format 4종 링크 확인 — --gc-sections silent
    // drop 시 --kv-format 미해석 폴백 대신 fail-fast.
    crate::format::ensure_builtin_kv_formats_registered()?;
    // 같은 self-test 의 format-policy 짝: 내장 policy(mixed_precision)가 KV_FORMAT_POLICIES 에
    // 링크됐는지 확인 — drop 시 `--kv-format mixed_precision` 미해석(single-format arm 으로 조용한
    // 폴백) 대신 fail-fast (W-ALLOC per-layer mixed precision 의 등록 가시성 보장).
    crate::format::ensure_builtin_format_policies_registered()?;
    // 같은 self-test 의 KV-mode 짝: 내장 mode 3종(standard/kivi/offload)이 KV_MODES 에
    // 링크됐는지 확인 — drop 시 mode_caps 가 silent None(폴백) 대신 fail-fast.
    crate::session::mode::ensure_builtin_kv_modes_registered()?;

    let init = SessionInitCtx::build(args)?;

    let tokenizer_path = resolve_tokenizer_path(args, &init.model_path, init.is_gguf);
    eprintln!("[Tokenizer] {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Cannot load tokenizer from {}: {}", tokenizer_path, e))?;
    check_vocab_compatibility(&tokenizer, &init.model, &tokenizer_path)?;

    let prompt = if let Some(path) = &args.prompt_file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt file {}: {}", path, e))?
    } else {
        args.prompt.clone()
    };
    eprintln!("Prompt: {}", prompt);
    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("Token Length: {}", tokens.len());

    Ok(InferencePrelude {
        init,
        tokenizer,
        tokens,
    })
}

/// `Args` 를 받아 추론에 필요한 전 컨텍스트를 조립한다.
///
/// `args.enable_resilience` 가 true 면 `build_command_executor` 로 transport 를
/// 연결하고 [`ResilienceAdapter`] 를 만든다 (transport 실패는 Err 전파).
/// false 면 `resilience = None` (NoOp default).
pub fn build_inference_ctx(args: Args) -> anyhow::Result<StandardHappyCtx> {
    let InferencePrelude {
        init,
        tokenizer,
        tokens,
    } = build_inference_prelude(&args)?;
    let backend = init.backend;
    let memory = init.memory;
    let hardware = init.hardware;
    let sampling_config = init.sampling_config;
    let model = init.model;

    let max_seq_len = args.max_seq_len;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;
    let vocab_size = model.config.vocab_size;
    eprintln!(
        "Model config: layers={}, kv_heads={}, head_dim={}, max_seq_len={}",
        num_layers, kv_heads, head_dim, max_seq_len
    );

    let kv_type = match args.kv_type.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "q4" => DType::Q4_0,
        other => bail!("Unsupported KV type: {other}. Use f32, f16, or q4."),
    };
    let kv_layout = KVLayout::from_cli(&args.kv_layout)
        .ok_or_else(|| anyhow::anyhow!("Unsupported --kv-layout: '{}'", args.kv_layout))?;

    let initial_kv_capacity = if args.initial_kv_capacity() > 0 {
        args.initial_kv_capacity().min(max_seq_len)
    } else {
        tokens
            .len()
            .saturating_add(args.num_tokens)
            .next_power_of_two()
            .max(128)
            .min(max_seq_len)
    };

    // dispatch: --kv-format(registry name)이 있으면 우선. 내장(f32/f16/q4_0/q8_0)은
    // typed 저장, 그 외 등록 format(예 synth_q4)은 opaque 저장(DType 없음). 미설정 시 --kv-type 하위호환.
    let kv_caches = match args.kv_format.as_deref().filter(|s| !s.is_empty()) {
        // 이름 기반 typed/opaque 분기. 내장 typed 이름이 아니면 make_format(정적 우선
        // → 동적 .so fallback)로 해소 후, descriptor 가 내장 DType 과 bit-equivalent 면 typed fast
        // path 로(layout_desc_to_builtin_dtype), 아니면 opaque floor 로 라우팅(2026-06-09 결정).
        // W-ALLOC: a registered KVFormatPolicy name routes to per-layer mixed precision (each layer
        // stored in its policy-assigned base format). Guarded before the single-format arm so a
        // policy name is never resolved as a format name. The construction-time consumer of the
        // (previously dormant) KVFormatPolicy producer — "expressible per-layer plan → executable".
        Some(fmt_name) if find_format_policy(fmt_name).is_some() => {
            eprintln!("KV format policy: {fmt_name} (per-layer mixed precision)");
            let pol_reg =
                find_format_policy(fmt_name).expect("guarded by find_format_policy().is_some()");
            let policy = (pol_reg.make)(StageParams::default());
            let per_layer = per_layer_storage_from_policy(
                &*policy,
                num_layers,
                kv_heads,
                head_dim,
                LayerStorage::Typed(kv_type),
            )?;
            alloc_mixed_kv_caches(
                &backend,
                memory.clone(),
                &per_layer,
                initial_kv_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                kv_layout,
            )?
        }
        Some(fmt_name) => match crate::format::builtin_format_dtype(fmt_name) {
            Some(dt) => {
                eprintln!("KV format: {fmt_name} (typed dtype {dt:?})");
                alloc_standard_kv_caches(
                    &backend,
                    memory.clone(),
                    num_layers,
                    initial_kv_capacity,
                    max_seq_len,
                    kv_heads,
                    head_dim,
                    dt,
                    kv_layout,
                )?
            }
            None => {
                // 내장 typed 이름 아님 → make_format(정적 force-link 또는 동적 .so, source-agnostic)로 해소.
                let fmt = crate::format::dynamic_format_registry::make_format(fmt_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Unknown --kv-format '{fmt_name}' (not found in either static KV_FORMATS or dynamic registration — check --load-plugin)"
                        )
                    })?;
                let desc = fmt.layout();
                // descriptor 가 내장 DType 과 bit-equivalent 면 typed fast path 로 라우팅(
                // name-keyed dispatch 를 descriptor-keyed 로 확장, 2026-06-09 결정). opaque generic floor
                // (dequant-whole→F32)는 ARM 에서 typed Q4_0(NEON) 대비 ~1.34x 느림(S25 실측) — descriptor
                // 가 내장과 일치하면 floor 비용이 불필요. 미일치(novel descriptor)는 opaque floor 유지.
                match crate::format::layout_desc_to_builtin_dtype(&desc) {
                    Some(dt) => {
                        eprintln!(
                            "KV format: {fmt_name} → bit-equivalent to builtin {dt:?} → typed fast path (bypassing opaque floor)"
                        );
                        alloc_standard_kv_caches(
                            &backend,
                            memory.clone(),
                            num_layers,
                            initial_kv_capacity,
                            max_seq_len,
                            kv_heads,
                            head_dim,
                            dt,
                            kv_layout,
                        )?
                    }
                    None => {
                        eprintln!("KV format: {fmt_name} (opaque — no DType, descriptor-driven)");
                        alloc_opaque_kv_caches(
                            &backend,
                            memory.clone(),
                            num_layers,
                            initial_kv_capacity,
                            max_seq_len,
                            kv_heads,
                            head_dim,
                            desc,
                            kv_layout,
                        )?
                    }
                }
            }
        },
        None => alloc_standard_kv_caches(
            &backend,
            memory.clone(),
            num_layers,
            initial_kv_capacity,
            max_seq_len,
            kv_heads,
            head_dim,
            kv_type,
            kv_layout,
        )?,
    };

    // ResilienceAdapter 생성. `--no-resilience` (effective enable_resilience=false)
    // 시 None. transport 연결 실패는 Err 전파 — graceful fail, panic 없음.
    let resilience: Option<ResilienceAdapter> = if args.enable_resilience {
        build_command_executor(&args, &model)?.map(|exec| {
            let mut adapter = ResilienceAdapter::new(exec);
            // heartbeat available_actions 가 Capability 와 동일 조건으로 산출되도록
            // 설정된 eviction policy 를 전파한다 (미전파 시 "none" → kv.evict_* 탈락).
            adapter.set_eviction_policy(args.eviction_policy());
            adapter
        })
    } else {
        None
    };

    Ok(StandardHappyCtx {
        args,
        backend,
        memory,
        hardware,
        model,
        tokenizer,
        kv_caches,
        tokens,
        max_seq_len,
        sampling_config,
        vocab_size,
        resilience,
    })
}

/// `--tokenizer-path` 우선, GGUF 면 sibling stem 검색, safetensors 면 dir 안의
/// `tokenizer.json`. legacy `generate` 와 동일한 resolution 순서.
pub fn resolve_tokenizer_path(args: &Args, model_path: &str, is_gguf: bool) -> String {
    if let Some(p) = args.tokenizer_path.as_ref() {
        return p.to_string_lossy().into_owned();
    }
    if is_gguf {
        let path = std::path::Path::new(model_path);
        let parent = path.parent().unwrap_or(std::path::Path::new("."));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        const QUANT_SUFFIXES: &[&str] = &["-f16", "-f32", "-q4_0", "-q4_1", "-q8_0", "-q4_k"];
        let stem_lower = stem.to_ascii_lowercase();
        let stem_stripped: Option<String> = QUANT_SUFFIXES.iter().find_map(|suf| {
            stem_lower
                .strip_suffix(suf)
                .map(|s| stem[..s.len()].to_string())
        });
        let mut candidates: Vec<std::path::PathBuf> = Vec::with_capacity(3);
        candidates.push(parent.join(format!("{stem}.tokenizer.json")));
        if let Some(ref s) = stem_stripped {
            candidates.push(parent.join(format!("{s}.tokenizer.json")));
        }
        candidates.push(parent.join("tokenizer.json"));
        candidates
            .iter()
            .find(|p| p.exists())
            .cloned()
            .unwrap_or_else(|| parent.join("tokenizer.json"))
            .to_string_lossy()
            .into_owned()
    } else {
        format!("{}/tokenizer.json", model_path)
    }
}

/// tokenizer vocab 과 model vocab 불일치 검사 (legacy generate 와 동일 정책).
pub fn check_vocab_compatibility(
    tokenizer: &Tokenizer,
    model: &TransformerModel,
    tokenizer_path: &str,
) -> anyhow::Result<()> {
    let tok_vocab = tokenizer.get_vocab_size(true);
    let model_vocab = model.config.vocab_size;
    let oob_tolerance: usize = 8;
    if tok_vocab > model_vocab + oob_tolerance {
        bail!(
            "Tokenizer vocab ({}) exceeds model vocab ({}) by more than {} — OOB embedding lookup risk. \
             Path: {}. Pass --tokenizer-path with the matching tokenizer.json.",
            tok_vocab,
            model_vocab,
            oob_tolerance,
            tokenizer_path
        );
    } else if tok_vocab > model_vocab {
        eprintln!(
            "[Tokenizer] WARNING: tokenizer vocab ({}) > model vocab ({}) by {} (likely multimodal special tokens).",
            tok_vocab,
            model_vocab,
            tok_vocab - model_vocab
        );
    }
    let pad_tolerance: usize = (model_vocab / 20).max(256);
    if model_vocab > tok_vocab + pad_tolerance {
        bail!(
            "Model vocab ({}) exceeds tokenizer vocab ({}) by more than {} — likely wrong tokenizer for model. \
             Path: {}. Pass --tokenizer-path with the matching tokenizer.json.",
            model_vocab,
            tok_vocab,
            pad_tolerance,
            tokenizer_path
        );
    }
    Ok(())
}

/// Resolve the requested KV layout against the backend. GPU flash decode is
/// HeadMajor-only (SeqMajor would silently fall back to CPU attention per token),
/// so a GPU backend forces HeadMajor and warns when `seq` was requested. CPU
/// honours the request as-is. Pure (takes `is_gpu`, not the backend) so the
/// policy is unit-testable without a GPU backend.
fn resolve_kv_layout(is_gpu: bool, requested: KVLayout) -> KVLayout {
    if is_gpu && requested == KVLayout::SeqMajor {
        eprintln!(
            "[KV layout] WARNING: GPU backend requires HeadMajor (flash decode is HeadMajor-only); \
             ignoring --kv-layout seq, using HeadMajor."
        );
        KVLayout::HeadMajor
    } else {
        requested
    }
}

/// Dynamic grow-on-demand KV cache 를 num_layers 만큼 할당. `layout` 으로 메모리
/// 레이아웃 선택(GPU 는 HeadMajor 강제 — [`resolve_kv_layout`]).
#[allow(clippy::too_many_arguments)]
pub fn alloc_standard_kv_caches(
    backend: &Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    num_layers: usize,
    initial_kv_capacity: usize,
    max_seq_len: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_type: DType,
    layout: KVLayout,
) -> anyhow::Result<Vec<KVCache>> {
    let layout = resolve_kv_layout(backend.is_gpu(), layout);
    let n_values = initial_kv_capacity * kv_heads * head_dim;
    let kv_buf_size = match kv_type {
        DType::Q4_0 => {
            use crate::quant::{BlockQ4_0, QK4_0};
            (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>()
        }
        _ => n_values * kv_type.size(),
    };
    eprintln!(
        "KV cache type: {:?}, layout: {:?} (initial capacity: {} tokens, {}B per layer, max: {})",
        kv_type, layout, initial_kv_capacity, kv_buf_size, max_seq_len
    );
    let mut kv_caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let k_buf = memory.alloc_kv(kv_buf_size, kv_type)?;
        let v_buf = memory.alloc_kv(kv_buf_size, kv_type)?;
        let shape = Shape::new(vec![1, kv_heads, initial_kv_capacity, head_dim]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend.clone());
        kv_caches.push(
            KVCache::new_dynamic(
                k,
                v,
                initial_kv_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                memory.clone(),
            )
            .with_layout(layout),
        );
    }
    Ok(kv_caches)
}

/// HeadMajor opaque(.so block-quant) KV cache 를 num_layers 만큼 할당.
///
/// 각 K/V 버퍼 = `OpaqueBuffer`(inner U8 + sidecar `desc`). byte 크기는 descriptor-keyed
/// (`bytes_for_elems`, G1). grow/attention 은 `KVCache`/`StandardFormat` 의 opaque arm 이 처리.
#[allow(clippy::too_many_arguments)]
pub fn alloc_opaque_kv_caches(
    backend: &Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    num_layers: usize,
    initial_kv_capacity: usize,
    max_seq_len: usize,
    kv_heads: usize,
    head_dim: usize,
    desc: KVLayoutDesc,
    layout: KVLayout,
) -> anyhow::Result<Vec<KVCache>> {
    use crate::buffer::Buffer;
    use crate::buffer::opaque::OpaqueBuffer;

    let layout = resolve_kv_layout(backend.is_gpu(), layout);
    let block_elems = desc.block_elems as usize;
    if block_elems == 0 || !head_dim.is_multiple_of(block_elems) {
        bail!("opaque KV: head_dim {head_dim} is not a multiple of block_elems {block_elems}");
    }
    let n_values = initial_kv_capacity * kv_heads * head_dim;
    let nbytes = desc.bytes_for_elems(n_values).ok_or_else(|| {
        anyhow::anyhow!("opaque KV: bytes_for_elems({n_values}) failed (block-aligned?)")
    })?;
    eprintln!(
        "KV cache: opaque (block_elems={}, bits={}, {}B per layer, {:?}, initial cap: {}, max: {})",
        desc.block_elems, desc.bits, nbytes, layout, initial_kv_capacity, max_seq_len
    );
    let mut kv_caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let shape = Shape::new(vec![1, kv_heads, initial_kv_capacity, head_dim]);
        let mk = || -> anyhow::Result<Tensor> {
            let inner = memory.alloc_kv(nbytes, DType::U8)?;
            let op: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner, desc));
            Ok(Tensor::new(shape.clone(), op, backend.clone()))
        };
        let k = mk()?;
        let v = mk()?;
        kv_caches.push(
            KVCache::new_dynamic(
                k,
                v,
                initial_kv_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                memory.clone(),
            )
            .with_layout(layout),
        );
    }
    Ok(kv_caches)
}

// ════════════════════════════════════════════════════════════════════════════
// W-ALLOC — construction-time per-layer KV precision (N-way mixed precision / KVTuner)
// ════════════════════════════════════════════════════════════════════════════
//
// The alloc functions above store ONE format for every layer, so a `KVFormatPolicy` that assigns a
// different base format per layer was *expressible* (it returns a per-layer `KVFormatPlan`) but not
// *executable*. This section closes that gap for the static per-layer case: the engine queries the
// policy once per layer at allocation time and stores each layer in its assigned base format. No new
// kernels — each `KVCache` self-describes its dtype and the per-cache compute path already dispatches
// f16/q4_0/opaque. Per-region/head/token overrides are the runtime re-encode concern (`format_apply`),
// deliberately out of scope here (alloc consumes only the per-layer base).

/// One layer's resolved KV storage: a builtin typed dtype (fast path) or an opaque descriptor (floor).
#[derive(Clone, Copy, Debug)]
pub enum LayerStorage {
    /// Builtin typed dtype (f32/f16/q4_0) — typed NEON/GPU compute.
    Typed(DType),
    /// Opaque block-quant descriptor (e.g. a `.so` format) — descriptor-driven floor.
    Opaque(KVLayoutDesc),
}

/// Minimal construction-time [`StageCtx`]: exposes only geometry + the layer position. No tokens or
/// runtime signals exist at allocation time, so every tensor/score accessor is empty. A policy that
/// needs runtime signals to choose precision therefore sees nothing here and keeps its base — only
/// layer-index-driven assignment (KVTuner / N-way fixed precision) is honored on this static path.
struct AllocCtx {
    layer: usize,
    n_layers: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl StageCtx for AllocCtx {
    fn current_pos(&self) -> usize {
        0
    }
    fn target_len(&self) -> usize {
        0
    }
    fn layer_idx(&self) -> usize {
        self.layer
    }
    fn importance(&self) -> Option<&[f32]> {
        None
    }
    fn n_kv_heads(&self) -> usize {
        self.kv_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
        None
    }
    fn n_layers(&self) -> usize {
        self.n_layers
    }
}

/// Resolve a registered format **name** to a concrete [`LayerStorage`] (mirror of the single-format
/// dispatch in [`build_inference_ctx`]): builtin typed → `Typed`; else `make_format(name).layout()`,
/// then bit-equivalent-to-builtin → `Typed` (fast path) else `Opaque`. Unknown name → `Err`.
fn resolve_format_id(name: &str) -> anyhow::Result<LayerStorage> {
    if let Some(dt) = crate::format::builtin_format_dtype(name) {
        return Ok(LayerStorage::Typed(dt));
    }
    let fmt = crate::format::dynamic_format_registry::make_format(name).ok_or_else(|| {
        anyhow::anyhow!(
            "per-layer KV format '{name}' is not a builtin and not registered (static KV_FORMATS or --load-plugin)"
        )
    })?;
    let desc = fmt.layout();
    match crate::format::layout_desc_to_builtin_dtype(&desc) {
        Some(dt) => Ok(LayerStorage::Typed(dt)),
        None => Ok(LayerStorage::Opaque(desc)),
    }
}

/// Query `policy` once per layer to build the per-layer storage vector (the W-ALLOC executable seam).
///
/// For each layer the engine builds a construction-time [`AllocCtx`] (geometry + layer index only)
/// and reads `policy.assign(ctx)`; the **base** `FormatId` of the returned plan is that layer's
/// storage format. `None` (no change) falls back to `default`. Per-region/head/token overrides are
/// ignored here — they are the runtime re-encode concern (`format_apply`), not construction storage.
pub fn per_layer_storage_from_policy(
    policy: &dyn KVFormatPolicy,
    num_layers: usize,
    kv_heads: usize,
    head_dim: usize,
    default: LayerStorage,
) -> anyhow::Result<Vec<LayerStorage>> {
    let mut per_layer = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let ctx = AllocCtx {
            layer,
            n_layers: num_layers,
            kv_heads,
            head_dim,
        };
        let storage = match policy.assign(&ctx) {
            Some(plan) => {
                // Honesty (lib.rs KVFormatPlan contract + no-silent-no-op): construction-time
                // per-layer alloc can store only a uniform-per-layer BASE format. A plan carrying
                // per-region/head/token overrides is heterogeneous-within-layer and unholdable here —
                // REJECT it (not silently drop to base), exactly as the runtime executor rejects it.
                if !plan.overrides.is_empty() {
                    bail!(
                        "KVFormatPolicy '{}' returned {} override(s) for layer {layer}; \
                         construction-time per-layer allocation honors only a uniform base format. \
                         Heterogeneous-within-layer precision is unsupported (would need a runtime \
                         re-encode store) — rejected rather than silently dropped.",
                        policy.name(),
                        plan.overrides.len()
                    );
                }
                resolve_format_id(&plan.base.0)?
            }
            None => default,
        };
        per_layer.push(storage);
    }
    Ok(per_layer)
}

/// Allocate one dynamic KV cache per layer, each in its own (possibly different) [`LayerStorage`] —
/// the per-layer twin of [`alloc_standard_kv_caches`] / [`alloc_opaque_kv_caches`]. Each layer routes
/// to the typed or opaque alloc body per its assigned format.
#[allow(clippy::too_many_arguments)]
pub fn alloc_mixed_kv_caches(
    backend: &Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    per_layer: &[LayerStorage],
    initial_kv_capacity: usize,
    max_seq_len: usize,
    kv_heads: usize,
    head_dim: usize,
    layout: KVLayout,
) -> anyhow::Result<Vec<KVCache>> {
    use crate::buffer::Buffer;
    use crate::buffer::opaque::OpaqueBuffer;

    let layout = resolve_kv_layout(backend.is_gpu(), layout);
    let n_values = initial_kv_capacity * kv_heads * head_dim;
    let shape = || Shape::new(vec![1, kv_heads, initial_kv_capacity, head_dim]);
    let mut kv_caches = Vec::with_capacity(per_layer.len());

    for (layer, storage) in per_layer.iter().enumerate() {
        let (k, v) = match storage {
            LayerStorage::Typed(kv_type) => {
                let kv_buf_size = match kv_type {
                    DType::Q4_0 => {
                        use crate::quant::{BlockQ4_0, QK4_0};
                        (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>()
                    }
                    _ => n_values * kv_type.size(),
                };
                let k_buf = memory.alloc_kv(kv_buf_size, *kv_type)?;
                let v_buf = memory.alloc_kv(kv_buf_size, *kv_type)?;
                (
                    Tensor::new(shape(), k_buf, backend.clone()),
                    Tensor::new(shape(), v_buf, backend.clone()),
                )
            }
            LayerStorage::Opaque(desc) => {
                let block_elems = desc.block_elems as usize;
                if block_elems == 0 || !head_dim.is_multiple_of(block_elems) {
                    bail!(
                        "opaque KV (layer {layer}): head_dim {head_dim} not a multiple of block_elems {block_elems}"
                    );
                }
                let nbytes = desc.bytes_for_elems(n_values).ok_or_else(|| {
                    anyhow::anyhow!(
                        "opaque KV (layer {layer}): bytes_for_elems({n_values}) failed (block-aligned?)"
                    )
                })?;
                let mk = || -> anyhow::Result<Tensor> {
                    let inner = memory.alloc_kv(nbytes, DType::U8)?;
                    let op: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner, *desc));
                    Ok(Tensor::new(shape(), op, backend.clone()))
                };
                (mk()?, mk()?)
            }
        };
        kv_caches.push(
            KVCache::new_dynamic(
                k,
                v,
                initial_kv_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                memory.clone(),
            )
            .with_layout(layout),
        );
    }
    eprintln!(
        "KV cache: per-layer mixed precision ({} layers, layout {:?}, initial cap: {}, max: {})",
        per_layer.len(),
        layout,
        initial_kv_capacity,
        max_seq_len
    );
    Ok(kv_caches)
}

/// AB-2 §5.7.7: argus-bench quant-window 분기 컨텍스트.
///
/// Standard [`StandardHappyCtx`] 와 달리 quant-window 는 `Vec<QuantizedRecentWindowCache>`(typed `KVCache` 아님) + caps
/// (`QuantAttnBackend` pull) + initial_bits/residual_size 를 보유한다. `build_bench_quant_window_loop`
/// (assembly) 가 이를 소비해 `QuantWindowForward` + `QuantWindowBitTransitionStage` 배선 `DecodeLoop` 를 조립한다.
pub struct QuantWindowBenchCtx {
    pub args: Args,
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub hardware: Arc<Hardware>,
    pub model: TransformerModel,
    /// quant-window native attention capability (OpenCL backend 면 `Some` 필수 — alloc_quant_window_kv_caches R3).
    pub quant_attn: Option<Arc<dyn QuantAttnBackend>>,
    pub tokenizer: Tokenizer,
    pub tokens: Vec<u32>,
    pub max_seq_len: usize,
    pub sampling_config: SamplingConfig,
    pub vocab_size: usize,
    /// quant-window 진입 시 양자화 bits (`--kv-mode kivi` → `--kivi-bits`, `--kv-dynamic-quant` → 16).
    pub initial_bits: u8,
    /// quant-window residual buffer 길이 (`--kv-mode kivi` → `--kivi-residual-len`,
    /// `--kv-dynamic-quant` → `(max_seq_len/32)*32`).
    pub residual_size: usize,
    pub resilience: Option<ResilienceAdapter>,
}

/// AB-2 §5.7.7: quant-window bench ctx 조립. v1 quant-window 진입 시맨틱(`generate.rs`(d5ed71d2^) L744-760) 재현.
///
/// `--kv-mode kivi` → initial_bits=`effective_quant_window_bits()`, residual=`effective_quant_window_residual_size()`.
/// `--kv-dynamic-quant`(orphan flag 재배선) → initial_bits=16(F16 등가 진입), residual=
/// `(max_seq_len/32)*32`. verify YAML baseline 은 `--kv-dynamic-quant` 로 진입한다.
pub fn build_quant_window_bench_ctx(args: Args) -> anyhow::Result<QuantWindowBenchCtx> {
    // W-ALLOC honesty: the quant-window KV mode (--kv-mode kivi / --kv-dynamic-quant) uses its own
    // KV path and does not consult a KVFormatPolicy. Fail fast on a per-layer mixed-precision policy
    // name instead of silently allocating its own uniform quant-window cache (no-silent-no-op).
    if let Some(fmt) = args.kv_format.as_deref().filter(|s| !s.is_empty())
        && find_format_policy(fmt).is_some()
    {
        bail!(
            "argus-bench: --kv-format '{fmt}' (per-layer mixed precision) is not supported under the \
             quant-window KV mode (--kv-mode kivi / --kv-dynamic-quant), which uses its own KV path. \
             Run mixed precision in the standard KV mode."
        );
    }
    let InferencePrelude {
        init,
        tokenizer,
        tokens,
    } = build_inference_prelude(&args)?;
    let backend = init.backend;
    let memory = init.memory;
    let hardware = init.hardware;
    let sampling_config = init.sampling_config;
    let model = init.model;
    // quant-window native attention capability pull (R3: OpenCL backend 면 Some 필수, init.rs 가 register).
    let quant_attn = init.caps.get::<dyn QuantAttnBackend>();

    let max_seq_len = args.max_seq_len;
    let vocab_size = model.config.vocab_size;
    eprintln!(
        "Model config: layers={}, kv_heads={}, head_dim={}, max_seq_len={}",
        model.config.num_hidden_layers,
        model.config.num_key_value_heads,
        model.config.head_dim,
        max_seq_len
    );

    // v1 census 재현: quantized-KV mode → Q2 진입, --kv-dynamic-quant → bits=16 진입.
    // 경로 게이트는 선언 cap(`ModeCaps.is_quantized_kv`)을 읽는다 — 구체 기술 이름 분기 0
    // (site #6). bits/residual 도출 자체는 quant_attn-specific 진입부라 여기 머문다(dispatch 아님).
    let is_quantized_kv_mode = crate::session::mode::mode_caps(args.effective_kv_mode())
        .is_some_and(|c| c.is_quantized_kv);
    let initial_bits: u8 = if is_quantized_kv_mode {
        args.effective_quant_window_bits()
    } else {
        16
    };
    let residual_size = if initial_bits == 16 {
        // bits=16: 전 토큰이 residual 에 잔류(quant flush 없음). QKKV(32) 배수로 내림.
        (max_seq_len / 32) * 32
    } else {
        args.effective_quant_window_residual_size()
    };

    let resilience: Option<ResilienceAdapter> = if args.enable_resilience {
        build_command_executor(&args, &model)?.map(|exec| {
            let mut adapter = ResilienceAdapter::new(exec);
            // StandardHappyCtx 경로와 동일 — heartbeat available_actions 일관성.
            adapter.set_eviction_policy(args.eviction_policy());
            adapter
        })
    } else {
        None
    };

    Ok(QuantWindowBenchCtx {
        args,
        backend,
        memory,
        hardware,
        model,
        quant_attn,
        tokenizer,
        tokens,
        max_seq_len,
        sampling_config,
        vocab_size,
        initial_bits,
        residual_size,
        resilience,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::galloc::Galloc;

    /// W-ALLOC test policy: assigns a different builtin format per layer purely by layer index.
    struct PerLayerMixPolicy;
    impl KVFormatPolicy for PerLayerMixPolicy {
        fn name(&self) -> &str {
            "test-per-layer-mix"
        }
        fn assign(
            &self,
            ctx: &dyn argus_extension_api::StageCtx,
        ) -> Option<argus_extension_api::KVFormatPlan> {
            // even layers → q4_0, odd layers → f16 (layer-index driven; KVTuner / N-way shape).
            let fmt = if ctx.layer_idx() % 2 == 0 {
                "q4_0"
            } else {
                "f16"
            };
            Some(argus_extension_api::KVFormatPlan {
                base: argus_extension_api::FormatId(fmt.into()),
                overrides: vec![],
            })
        }
    }

    /// The (formerly dormant) KVFormatPolicy producer is consumed per layer → per-layer storage.
    #[test]
    fn per_layer_storage_from_policy_assigns_per_layer_format() {
        let per_layer = per_layer_storage_from_policy(
            &PerLayerMixPolicy,
            4,
            2,
            8,
            LayerStorage::Typed(DType::F16),
        )
        .unwrap();
        assert!(matches!(per_layer[0], LayerStorage::Typed(DType::Q4_0)));
        assert!(matches!(per_layer[1], LayerStorage::Typed(DType::F16)));
        assert!(matches!(per_layer[2], LayerStorage::Typed(DType::Q4_0)));
        assert!(matches!(per_layer[3], LayerStorage::Typed(DType::F16)));
    }

    /// N-way mixed precision EXECUTES: alloc materializes each layer's cache in its own dtype
    /// (the W-ALLOC wall was a single uniform dtype loop). Proof the engine runs the expressible plan.
    #[test]
    fn alloc_mixed_kv_caches_materializes_per_layer_dtype() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
        let per_layer = per_layer_storage_from_policy(
            &PerLayerMixPolicy,
            4,
            2,
            8,
            LayerStorage::Typed(DType::F16),
        )
        .unwrap();
        let caches = alloc_mixed_kv_caches(
            &backend,
            memory,
            &per_layer,
            16,
            64,
            2,
            8,
            KVLayout::HeadMajor,
        )
        .unwrap();
        assert_eq!(caches.len(), 4);
        assert_eq!(caches[0].kv_dtype(), DType::Q4_0);
        assert_eq!(caches[1].kv_dtype(), DType::F16);
        assert_eq!(caches[2].kv_dtype(), DType::Q4_0);
        assert_eq!(caches[3].kv_dtype(), DType::F16);
    }

    /// Test policy that emits a per-region override — must be REJECTED, not silently downgraded.
    struct OverridePolicy;
    impl KVFormatPolicy for OverridePolicy {
        fn name(&self) -> &str {
            "test-override"
        }
        fn assign(
            &self,
            _ctx: &dyn argus_extension_api::StageCtx,
        ) -> Option<argus_extension_api::KVFormatPlan> {
            Some(argus_extension_api::KVFormatPlan {
                base: argus_extension_api::FormatId("f16".into()),
                overrides: vec![argus_extension_api::FormatOverride {
                    region: argus_extension_api::KeepSpec::LayerWide(vec![0]),
                    format: argus_extension_api::FormatId("q4_0".into()),
                    side: argus_extension_api::MergeAxis::Both,
                }],
            })
        }
    }

    /// Honesty: an override-bearing plan is rejected (Err), not silently dropped to base.
    #[test]
    fn override_bearing_plan_is_rejected_not_dropped() {
        let res = per_layer_storage_from_policy(
            &OverridePolicy,
            2,
            2,
            8,
            LayerStorage::Typed(DType::F16),
        );
        assert!(res.is_err(), "override-bearing plan must be rejected");
        let msg = res.unwrap_err().to_string();
        assert!(msg.contains("override"), "error names the override: {msg}");
    }

    /// The shipped builtin `mixed_precision` policy is registered AND consumable through the production
    /// calls (`find_format_policy` → `make` → `per_layer_storage_from_policy`) — i.e. the W-ALLOC
    /// routing arm has a reachable trigger (no longer a dead arm). Env-free: unset `ARGUS_KV_MIXED`
    /// makes the policy a no-op (all layers keep the default), which is deterministic for CI.
    #[test]
    fn registered_mixed_precision_policy_is_consumable() {
        let reg = find_format_policy("mixed_precision")
            .expect("mixed_precision registered in KV_FORMAT_POLICIES (dead-arm fixed)");
        let policy = (reg.make)(StageParams::default());
        let per_layer =
            per_layer_storage_from_policy(&*policy, 6, 2, 8, LayerStorage::Typed(DType::F16))
                .expect("registered policy consumes without error");
        assert_eq!(per_layer.len(), 6);
    }

    #[test]
    fn kv_layout_from_cli_parses_known_values() {
        assert_eq!(KVLayout::from_cli("head"), Some(KVLayout::HeadMajor));
        assert_eq!(KVLayout::from_cli("seq"), Some(KVLayout::SeqMajor));
        assert_eq!(KVLayout::from_cli("bogus"), None);
    }

    #[test]
    fn resolve_kv_layout_cpu_honours_request() {
        // CPU honours the requested layout as-is (both directions).
        assert_eq!(
            resolve_kv_layout(false, KVLayout::SeqMajor),
            KVLayout::SeqMajor
        );
        assert_eq!(
            resolve_kv_layout(false, KVLayout::HeadMajor),
            KVLayout::HeadMajor
        );
    }

    #[test]
    fn resolve_kv_layout_gpu_forces_head_major() {
        // GPU flash decode is HeadMajor-only: `seq` is upgraded to `head`, `head` stays.
        assert_eq!(
            resolve_kv_layout(true, KVLayout::SeqMajor),
            KVLayout::HeadMajor
        );
        assert_eq!(
            resolve_kv_layout(true, KVLayout::HeadMajor),
            KVLayout::HeadMajor
        );
    }

    #[test]
    fn alloc_standard_kv_caches_selects_layout_on_cpu() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
        // Default `head` stays HeadMajor (byte-identical to the previous hardcoded path);
        // `seq` now actually selects SeqMajor on CPU (the dead flag is live).
        for (req, want) in [
            (KVLayout::HeadMajor, KVLayout::HeadMajor),
            (KVLayout::SeqMajor, KVLayout::SeqMajor),
        ] {
            let caches = alloc_standard_kv_caches(
                &backend,
                memory.clone(),
                2,  // num_layers
                64, // initial_kv_capacity
                64, // max_seq_len
                4,  // kv_heads
                8,  // head_dim
                DType::F32,
                req,
            )
            .unwrap();
            assert_eq!(caches.len(), 2);
            assert_eq!(
                caches[0].layout(),
                want,
                "CPU must honour requested layout {req:?}"
            );
        }
    }
}
