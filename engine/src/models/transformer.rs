use anyhow::{Result, anyhow};
use std::sync::Arc;

use crate::backend::Backend;
use crate::buffer::{Buffer, DType};
use crate::inference::attention_scores::AttentionScoreAccumulator;
#[cfg(feature = "opencl")]
use crate::layers::tensor_partition::PartitionContext;
use crate::layers::transformer_layer::TransformerLayer;
use crate::layers::workspace::LayerWorkspace;
use crate::memory::Memory;
use crate::model_config::{ModelArch, ModelConfig};
use crate::models::weights::{LayerSlot, SecondaryMmap};
// §13.8-O: `preload_pool` struct 필드(`OnceLock<Box<dyn PreloadAccess>>`)가 참조하는
// PreloadAccess trait 은 L2 로 격상되었다(`crate::preload_access`) — inference/pressure
// 공유 어휘. offload-path forward(`forward_into_offload`)는 `models/offload_forward.rs`
// 로 분리되어 OffloadKVCache/OffloadFormat/PrefetchController 등 pressure concrete 결합은
// 본 파일 본문에서 제거되었다.
use crate::preload_access::PreloadAccess;
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::weight::compute_quant_noise;

#[cfg(feature = "opencl")]
// LAYER-EXEMPT: backend_concrete_downcast — §13.8-L
use crate::backend::opencl::plan::FullKernelPlan;

/// Returns true if this layer uses local (sliding window) attention.
/// Gemma3 pattern: every `pattern`-th layer (1-indexed) uses global attention;
/// all other layers use local attention.
/// Example: pattern=6 → layers 5,11,17,... (0-indexed) are global.
// pub(crate): offload-path forward(`models/offload_forward.rs`)가 호출하는 공유 헬퍼.
pub(crate) fn is_local_layer(layer_idx: usize, pattern: Option<usize>) -> bool {
    match pattern {
        Some(p) if p > 0 => !(layer_idx + 1).is_multiple_of(p),
        _ => false,
    }
}

/// Map one weight tensor so the CPU backend can read it via `as_ptr()`
/// (host-access mapping for SwitchHw / tensor partition).
///
/// Extracted verbatim from the former `map_one` closure inside
/// `map_weights_for_cpu` (AB-4 §5.5.4) so the slot-only `&self` variant
/// (`map_layer_weights_for_host_access`) and the full `&mut self` path share it.
///
/// Returns `(mapped_tensor, changed)`: `changed` is true when the tensor's
/// backing buffer or backend tag was replaced. `gguf.rs::load_raw` builds
/// weights via `backend.copy_weight_from`, which on OpenCL leaves
/// `Tensor::backend()` pointing at the CPU loader even though the buffer is a
/// `UnifiedBuffer` with valid `cl_mem`. Downstream `weight.backend().copy_from`
/// (e.g. `split_weight_col` for FFN-down partition) then dispatches CPU copy
/// and produces a `SharedBuffer`-backed GPU slice without `cl_mem`, and the
/// next `matmul_f16` aborts with "B is not OpenCL buffer". Retag onto the
/// active GPU backend whenever the loader tag differs, so partition / matmul
/// dispatch see GPU consistently.
#[cfg(feature = "opencl")]
fn map_weight_for_host(
    t: &Tensor,
    be: &Arc<dyn Backend>,
    queue: &ocl::Queue,
) -> Result<(Tensor, bool)> {
    let needs_retag = !Arc::ptr_eq(t.backend(), be);
    let retagged = |t: &Tensor| -> Tensor {
        if needs_retag {
            Tensor::new(t.shape().clone(), t.buffer().clone(), be.clone())
        } else {
            t.clone()
        }
    };
    // Already CPU-accessible (mapped UnifiedBuffer, mmap-backed) → no-op.
    if !t.buffer().as_ptr().is_null() {
        t.buffer().map_for_cpu()?; // no-op if already mapped
        return Ok((retagged(t), needs_retag));
    }
    // Unmapped UnifiedBuffer (post-swap `materialise_tensor` lands one in
    // the LayerWeights snapshot). Map in-place to recover the host
    // pointer without paying the read_buffer + alloc cost below.
    t.buffer().map_for_cpu()?;
    if !t.buffer().as_ptr().is_null() {
        return Ok((retagged(t), needs_retag));
    }
    // Device-only buffer (OpenCLBuffer): read to new UnifiedBuffer.
    let size = t.size();
    let ub = crate::memory::opencl::unified::UnifiedBuffer::new(queue.clone(), size, t.dtype())?;
    ub.map()?;
    // SAFETY: ub.as_mut_ptr() is valid after map(). read_buffer copies from GPU.
    let dst = unsafe { std::slice::from_raw_parts_mut(ub.as_mut_ptr(), size) };
    be.read_buffer(t, dst)?;
    let buf: Arc<dyn Buffer> = Arc::new(ub);
    Ok((Tensor::new(t.shape().clone(), buf, be.clone()), true))
}

/// Map every layer slot's weights for host (CPU) access (AB-4 §5.5.4).
///
/// `map_weights_for_cpu` 의 slot 순회 부분을 slot 슬라이스 + gpu backend 만 받는
/// free function 으로 추출한 것. top-level `norm`/`lm_head`/`gpu_embed_tokens`
/// (SwitchHw 전용, model `&mut self` 필요)는 다루지 않으므로 model handle 이
/// 필요 없다 — `PartitionStage` 가 보유한 `Vec<Arc<LayerSlot>>` 만으로 lazy
/// host-mapping 을 수행하는 진입점이다. slot 갱신은 `store_weights_same_dtype`
/// (ArcSwap, clone-then-install INV-123). 반환값 = 새로 매핑한 weight tensor 개수.
#[cfg(feature = "opencl")]
pub fn map_layer_slots_for_host_access(
    slots: &[Arc<LayerSlot>],
    gpu_backend: &Arc<dyn Backend>,
) -> Result<usize> {
    // COLD-EXT: weight mmap startup (1회).
    let ocl_be = gpu_backend
        .get_extension(crate::backend::EXT_OPENCL_QUEUE)
        .and_then(|a| a.downcast_ref::<crate::backend::opencl::OpenCLBackend>())
        .ok_or_else(|| anyhow!("GPU backend is not OpenCL"))?;
    let queue = ocl_be.queue.clone();

    let mut count = 0;
    // Weight slot migration uses clone-then-install so the ArcSwap sees a
    // single atomic transition per slot (INV-123).
    for slot in slots {
        let old = slot.load_weights();
        let mut layer = (*old).clone();
        macro_rules! map_slot_weight {
            ($t:expr) => {
                let (new, changed) = map_weight_for_host(&$t, gpu_backend, &queue)?;
                if changed {
                    $t = new;
                    count += 1;
                }
            };
        }
        map_slot_weight!(layer.wq);
        map_slot_weight!(layer.wk);
        map_slot_weight!(layer.wv);
        map_slot_weight!(layer.wo);
        map_slot_weight!(layer.w_gate);
        map_slot_weight!(layer.w_up);
        map_slot_weight!(layer.w_down);
        map_slot_weight!(layer.attention_norm);
        map_slot_weight!(layer.ffn_norm);
        if let Some(ref mut bias) = layer.qkv_bias {
            map_slot_weight!(bias.bq);
            map_slot_weight!(bias.bk);
            map_slot_weight!(bias.bv);
        }
        if let Some(ref t) = layer.q_norm {
            let (new, _) = map_weight_for_host(t, gpu_backend, &queue)?;
            layer.q_norm = Some(new);
            count += 1;
        }
        if let Some(ref t) = layer.k_norm {
            let (new, _) = map_weight_for_host(t, gpu_backend, &queue)?;
            layer.k_norm = Some(new);
            count += 1;
        }
        if let Some(ref t) = layer.pre_ffn_norm {
            let (new, _) = map_weight_for_host(t, gpu_backend, &queue)?;
            layer.pre_ffn_norm = Some(new);
            count += 1;
        }
        if let Some(ref t) = layer.post_ffn_norm {
            let (new, _) = map_weight_for_host(t, gpu_backend, &queue)?;
            layer.post_ffn_norm = Some(new);
            count += 1;
        }
        slot.store_weights_same_dtype(Arc::new(layer));
    }
    Ok(count)
}

pub struct TransformerModel {
    pub config: ModelConfig,
    /// Swap-aware decoder layer slots. Each slot wraps its weights behind an
    /// `ArcSwap` so the forward path acquires a lock-free `Arc<TransformerLayer>`
    /// snapshot per layer (INV-123). Mutation (partition install, backend
    /// migration) uses `LayerSlot::rcu_weights` to clone-and-install atomically.
    /// Spec: ENG-DAT-093.
    ///
    /// LISWAP-2 Phase 6.1 (prototype): layers are now wrapped in `Arc` so that
    /// `AsyncSwapDispatcher::submit_commit` can take ownership of a slot handle
    /// across thread boundaries (Phase 6.2 will activate the dispatcher path).
    /// The forward / mutation surface is unchanged thanks to `Arc::deref` on
    /// `LayerSlot`.
    pub layers: Vec<Arc<LayerSlot>>,
    /// Optional secondary GGUF handle retained for the entire model lifetime
    /// (INV-125). `None` means the dynamic-swap path is disabled.
    /// Phase 1 only populates and keeps the handle alive; Phase 2
    /// `SwapExecutor` consumes it.
    pub secondary_mmap: Option<Arc<SecondaryMmap>>,
    /// Global swap generation counter (ENG-DAT-093). Declared in Phase 1,
    /// bumped by Phase 2 `SwapExecutor`. Kept as an `Arc<AtomicU64>` so plan
    /// builders / handlers that need generation-aware invalidation can share
    /// the same counter.
    pub ratio_generation: Arc<std::sync::atomic::AtomicU64>,
    /// embed_tokens is always kept on CPU for model loading.
    /// When the main backend is GPU, `gpu_embed_tokens` holds a device-side copy
    /// so that `gather_embed` can run entirely on the GPU without CPU round-trips.
    pub embed_tokens: Tensor,
    pub norm: Tensor,
    /// lm_head weight tensor.
    /// When `lm_head_on_cpu` is true, this lives on the CPU backend even when the
    /// main backend is GPU (avoids `CL_INVALID_BUFFER_SIZE` for large vocabs).
    pub lm_head: Tensor,
    /// When true, `lm_head` is on CPU and the final matmul must be done via CPU
    /// fallback: read x from GPU → CPU matmul → write logits back.
    /// This happens for models with large tied embeddings (e.g., gemma3-1b:
    /// 262144 × 1152 × 2 = ~604 MB exceeds typical mobile GPU alloc limits).
    pub lm_head_on_cpu: bool,
    /// GPU-side copy of embed_tokens for zero-sync gather on GPU backends.
    /// - Tied weights: shares the same GPU buffer as `lm_head` (zero extra memory).
    /// - Untied weights: a separate GPU copy uploaded once at load time.
    /// - CPU backend: None (unnecessary, gather runs on CPU directly).
    pub(crate) gpu_embed_tokens: Option<Tensor>,
    /// CPU backend used for embed_tokens gather when the main backend is GPU.
    /// None when the main backend is already CPU.
    pub(crate) cpu_backend: Option<Arc<dyn Backend>>,
    /// Persistent thread pool for offload preload operations.
    /// Lazily initialized on first `forward_into_offload` call.
    /// `OnceLock` allows interior mutability without lock contention on subsequent calls
    /// (`forward_into_offload` takes `&self`).
    pub(crate) preload_pool: std::sync::OnceLock<Box<dyn PreloadAccess>>,
    /// Per-layer quantization noise factor table accessor (ENG-DAT-095).
    ///
    /// Trait object so the struct definition does not reference pressure-side
    /// concrete types directly (§13.8-O cross-L3 vocabulary 본질 해소,
    /// 2026-05-27 sprint). The installed impl is `QuantNoiseTable` produced
    /// by `weight::setup_runtime_resources` and later replaced via
    /// `weight::compute_quant_noise` once the secondary mmap is
    /// known (ENG-ALG-216).
    ///
    /// - `secondary_mmap == None`: `QuantNoiseTable::empty()` (len==0).
    /// - Secondary present but all layers failed: `QuantNoiseTable::uniform_ones(n)`.
    /// - Normal: `QuantNoiseTable::new_from_frobenius(...)` with `is_computed=true`.
    pub quant_noise: Arc<dyn crate::runtime_resources_access::QuantNoiseAccess>,
    /// Async primary cl_mem release worker accessor (ENG-ALG-228 / ENG-DAT-100).
    ///
    /// Trait object — the installed impl is `PrimaryReleaseWorker` spawned by
    /// `weight::setup_runtime_resources`. `SwapExecutor` clones the
    /// `Arc<dyn ReleaseWorkerAccess>` to enqueue displaced `LayerWeights` in
    /// Stage (c) and to call `drain` (INV-141) before the next swap batch.
    ///
    /// `Arc` so `SwapExecutor` (which borrows references) can hold a clone
    /// without lifetime coupling to the model. Drop impl on the underlying
    /// `PrimaryReleaseWorker` joins the worker thread, ensuring all destructors
    /// run before process exit.
    pub release_worker: Arc<dyn crate::runtime_resources_access::ReleaseWorkerAccess>,
}

// offload fmt 경로(`forward_into_offload`)는 KV 를 별도 `fmts: &[Arc<OffloadFormat>]`
// 파라미터로 받는다 — 본 구조체는 그 외 forward 입력만 운반한다. (옛 generic `kv_caches:
// &mut [C]` 슬롯은 양쪽 호출부에서 `&mut []` 로 전달돼 본문에서 무시되던 vestigial 잔재라
// 제거했다; 그에 따라 `C` 타입 파라미터도 더는 쓰이지 않아 함께 제거.)
pub struct OffloadForwardArgs<'a> {
    pub input_tokens: &'a Tensor,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub logits_out: &'a mut Tensor,
    pub x_gen: Option<&'a mut Tensor>,
    pub workspace: Option<&'a mut LayerWorkspace>,
    /// Pre-allocated prefill workspace (caller-owned, max_seq_len-sized).
    /// When `Some`, `forward_into` reuses it instead of allocating a fresh one
    /// for each prefill call. Essential on NVIDIA OpenCL where repeated
    /// alloc/free cycles accumulate deferred-release pressure (→ CL_OUT_OF_RESOURCES).
    pub prefill_workspace: Option<&'a mut crate::layers::workspace::PrefillWorkspace>,
    /// Optional attention score accumulator for heavy-hitter-style eviction.
    /// When active, post-softmax scores are captured from tracked layers.
    pub score_accumulator: Option<&'a mut AttentionScoreAccumulator>,
    /// Optional per-op profiler.
    pub profiler: Option<&'a mut dyn crate::instrument::OpInstrument>,
    /// Optional skip configuration for layer skipping.
    pub skip_config: Option<&'a crate::inference::skip_config::SkipConfig>,
    /// Optional importance collector for Layer Skip QCF.
    /// When provided during prefill, captures per-layer cosine similarity.
    /// Uses L2 `ImportanceCollect` trait (§13.8-G + INV-LAYER-003 trait inversion).
    pub importance_collector: Option<&'a mut dyn crate::qcf_collector::ImportanceCollect>,
    /// When true, only compute logits for the last sequence position.
    /// Saves ~3GB GPU memory for long-context prefill (e.g., eval-ll with 5K+ tokens).
    /// logits_out shape should be [1, 1, vocab_size] instead of [1, seq_len, vocab_size].
    pub logits_last_only: bool,
    /// Optional layer boundary hook (LISWAP-4 / ENG-ALG-235).
    ///
    /// When `Some`, `forward_into` calls `hook.on_layer_boundary(idx, seq_len)`
    /// after each layer's compute step (and inside the layer loop, before
    /// `layer.forward`, calls the wait gate via `hook.pending_event_for(idx)`).
    /// When `None`, the hot-path overhead is one `Option::is_some` branch
    /// per layer (INV-147).
    pub layer_boundary_hook: Option<&'a dyn crate::layer_boundary_hook::LayerBoundaryHook>,
    /// S6: KV read stage — decode arm 에서 layer i 완료 후 read_plan 을 산출해
    /// layer i+1 의 prefetch 우선 힌트로 공급한다.
    /// `None`(기본) = 기존 순차 preload 그대로(read_stage=None byte-identical, INV-147).
    pub read_stage: Option<&'a dyn argus_extension_api::KVReadStage>,
}

/// `forward_into` 의 KVCacheFormat trait-object fork 인자 (Phase α-K substep 3c).
///
/// `forward_into` 의 generic `kv_caches: &mut [C]` 입력을
/// `fmts: &[Arc<dyn KVCacheFormat>]`(trait object, `&self` mutation 이라 `&mut` 불요)로 교체.
/// **decode-only**(seq_len=1) — prefill flip 은 (3d). score/profiler/skip/importance/variance/
/// prefill_ws/layer_boundary 는 happy-path decode 가 안 쓰므로(`is_standard_happy_path` 보장) 드롭
/// (CLAUDE.md 추측성 유연성 금지 — 필요 시 후속 substep 에서 추가). eviction score 누적은 (3c-evict).
pub struct TransformerModelForwardArgs<'a> {
    pub input_tokens: &'a Tensor,
    pub start_pos: usize,
    pub fmts: &'a [Arc<dyn crate::format::KVCacheFormat>],
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub logits_out: &'a mut Tensor,
    pub x_gen: Option<&'a mut Tensor>,
    /// decode(seq_len=1) workspace. prefill(seq_len>1)은 forward_into 가 owned
    /// PrefillWorkspace 를 자체 할당하므로 None 이어도 무방.
    pub workspace: Option<&'a mut LayerWorkspace>,
    /// prefill 한정: true 면 마지막 토큰 hidden 만 lm_head (logits_out=[1,1,vocab]).
    /// decode(seq_len=1)에선 무관. forward_into 의 `logits_last_only`(transformer.rs:1960) 미러.
    pub logits_last_only: bool,
    /// heavy-hitter-style eviction score accumulator (Phase α-K ①-c — eval flip). `Some` 이면 decode 마다
    /// post-softmax score 를 누적(`forward_into:1894-1922` 미러). production(ModelForward)은 항상 `None`.
    pub score_accumulator: Option<&'a mut AttentionScoreAccumulator>,
    /// Q running mean/var 누적기 (QueryStats, future-attention 추정용). `Some` 이면 decode
    /// 마다 활성 layer 의 RoPE-적용 Q(`ws.q`)를 GQA 환원해 누적(score 누적 seam 인접 1지점). score
    /// 비활성 happy path 는 `None`(MQ-4 hot-path 게이트 — `Option::is_some` 분기 1회 외 비용 0).
    /// production(ModelForward)은 항상 `None`; score-active 측정 하네스만 `Some`.
    pub query_stats_accumulator:
        Option<&'a mut crate::inference::query_stats::QueryStatsAccumulator>,
    /// layer-skip 설정 (Phase α-K ①-c). production 은 항상 `None`.
    pub skip_config: Option<&'a crate::inference::skip_config::SkipConfig>,
    /// Layer Skip QCF importance collector — prefill 2-pass 전용 (Phase α-K ①-c). production 은 `None`.
    pub importance_collector: Option<&'a mut dyn crate::qcf_collector::ImportanceCollect>,
    /// cache 자가-need(quant-window AWQE) 힌트 (Phase α-K ①-c). base trait 에 `needs_attn_scores` 가 없으므로
    /// (§4.1 R4 ③) caller 가 `caches[0].needs_attn_scores()` 를 산출해 주입한다 — `need_scores` 의 OR
    /// 항(`forward_gen.rs:409` 미러). production 은 `false`.
    pub cache_self_need_scores: bool,
    /// IntraForward/LayerImmediate weight swap 의 layer-boundary hook (§5.9.2 Track B).
    /// `Some` 이면 forward_into 의 layer loop 가 layer `i` 진입 전 wait-gate
    /// (`pending_event_for_dyn(i)`) + layer `i` compute 직후 `on_layer_boundary(i, seq_len)` 를
    /// 호출한다. `None`(production 기본/모든 eval·ppl·warmup 경로)이면 layer 당 `Option::is_some`
    /// branch 1회만 추가 — INV-147 zero-overhead. `OffloadForwardArgs` 의 동명 슬롯(offload 전용)과
    /// 별개로 production decode 경로(`forward_into`)에 신설된 슬롯이다.
    pub layer_boundary_hook: Option<&'a dyn crate::layer_boundary_hook::LayerBoundaryHook>,
    /// KV read stage — "무엇을 읽을지" plugin 표면. `Some` 이면 decode
    /// 의 layer 루프가 attention 직전에 `read_plan(ctx)` 를 layer 당 1회 호출해, 활성 format 이
    /// `SelectiveRead` 를 구현하면 선택적 읽기(`attention_into_selected`)로, 아니면 plan 을 무시하고
    /// full read 폴백(D4). `None`(production 기본 / 빌트인 0개 시작)이면 layer 당 `Option::is_some`
    /// branch 1회만 추가 — INV-147 zero-overhead, α-K frozen byte-identical 게이트 대상.
    pub read_stage: Option<&'a dyn argus_extension_api::KVReadStage>,
    /// R-P1-1 PFA producer target. `Some((buf, q_window))` — `ModelForward` 가 **최종 prefill 청크**
    /// 에서 무장 시에만 설정(chunk gating 은 caller 책임) → 실제 prefill(seq_len>1) layer 루프가 layer
    /// `i` 슬라이스 `buf[i]`(`[n_heads_q * prefix_len]`, caller pre-zeroed)에 trailing q_window 의
    /// per-attention-head attention 확률을 SUM-누적. `None`(decode / 미무장 / production·eval·ppl 기본)
    /// = byte-identical(layer 당 `is_some` branch 1회).
    pub prefill_attn: Option<(&'a mut Vec<Vec<f32>>, usize)>,
    /// IMP-4 per-step PFA target. `Some((bufs, q_window, per_head))` — like [`Self::prefill_attn`]
    /// but each trailing query row is written to its **own step slot** (assign, not SUM-collapsed),
    /// yielding the per-output-step attention trajectory. `bufs[i]` is layer `i`'s flat per-step
    /// buffer (head-mean `[q_window * prefix_len]` or, when `per_head`, full
    /// `[q_window * n_heads_q * prefix_len]`), caller pre-zeroed. `None` everywhere except the
    /// `answer_attention_steps` dump = byte-identical (per-layer `as_mut` branch once, INV-147).
    pub prefill_attn_per_row: Option<(&'a mut Vec<Vec<f32>>, usize, bool)>,
}

impl TransformerModel {
    pub fn load(model_path: &str, backend: Arc<dyn Backend>, _memory: &dyn Memory) -> Result<Self> {
        Self::load_with_dtype(model_path, backend, _memory, DType::F16)
    }

    pub fn load_with_dtype(
        model_path: &str,
        backend: Arc<dyn Backend>,
        _memory: &dyn Memory,
        weight_dtype: DType,
    ) -> Result<Self> {
        use crate::models::loader::safetensors::SafetensorsSource;
        let source = SafetensorsSource::open(model_path, weight_dtype)?;
        crate::models::loader::load_model(&source, backend, _memory, None)
    }

    /// Load a model from a GGUF file.
    ///
    /// The weight dtype is determined from the GGUF file's tensor types.
    /// No conversion is performed -- tensors are loaded in their native format.
    pub fn load_gguf(
        model_path: &str,
        backend: Arc<dyn Backend>,
        memory: &dyn Memory,
    ) -> Result<Self> {
        Self::load_gguf_with_secondary(model_path, None, backend, memory)
    }

    /// Load a model from a `LoadConfig` (ENG-DAT-090).
    ///
    /// Sprint 1 W-AUF-1 C3: 3-way primary format dispatch — GGUF/AUF/Safetensors.
    /// AUF self-secondary 자동 활성은 W-AUF-2에서 본격화 (현 stub: explicit
    /// `secondary_source`는 AUF primary에서 warning 후 무시).
    pub fn load_from_config(
        config: &crate::models::loader::LoadConfig,
        backend: Arc<dyn Backend>,
        memory: &dyn Memory,
    ) -> Result<Self> {
        use crate::models::loader::PrimaryFormat;
        match config.primary_format {
            PrimaryFormat::Gguf => Self::load_gguf_from_config(config, backend, memory),
            PrimaryFormat::Auf => Self::load_auf_from_config(config, backend, memory),
            PrimaryFormat::Safetensors => Err(anyhow!(
                "load_from_config: Safetensors primary는 본 진입점에서 미지원. \
                 `TransformerModel::load_with_dtype`를 직접 호출하라."
            )),
        }
    }

    /// GGUF primary 전용 `load_from_config` 구현 (기존 로직).
    fn load_gguf_from_config(
        config: &crate::models::loader::LoadConfig,
        backend: Arc<dyn Backend>,
        memory: &dyn Memory,
    ) -> Result<Self> {
        use crate::models::loader::TensorSource;
        use crate::models::loader::gguf::GgufSource;
        use crate::models::weights::open_secondary_with_backend;

        let primary_path = config
            .primary_source
            .to_str()
            .ok_or_else(|| anyhow!("primary_source path is not valid UTF-8"))?;

        let source = GgufSource::open(std::path::Path::new(primary_path))?;
        let secondary_mmap = match config.secondary_source.as_deref() {
            None => None,
            Some(p) => {
                let gguf = source.gguf_file();
                let model_config = source.config();
                // LISWAP-6: backend-aware open. When `backend` is qnn_oppkg
                // and the secondary is GGUF, this returns the Rpcmem variant
                // for DMA-BUF alias swap (zero H2D copy). All other backend
                // / format combinations dispatch to the standard path.
                let handle = open_secondary_with_backend(
                    p,
                    model_config,
                    gguf,
                    config.secondary_dtype_choice,
                    config.secondary_layout_choice,
                    &backend,
                )
                .map_err(|e| anyhow!("secondary weight load failed: {e}"))?;
                let arc = Arc::new(handle);
                // LISWAP-6 Phase 1 — install self-Weak so the Rpcmem store
                // can populate the cl_mem alias cache eagerly inside
                // `ensure_layer_loaded`. Other variants ignore the call.
                if let crate::models::weights::SecondaryMmap::Rpcmem(rpc) = arc.as_ref() {
                    rpc.install_self_arc(&arc);
                }
                Some(arc)
            }
        };
        let mut model =
            crate::models::loader::load_model(&source, backend, memory, secondary_mmap)?;

        // ENG-ALG-216: eager ε computation immediately after secondary mmap open.
        model.quant_noise =
            compute_quant_noise(model.layers.as_slice(), model.secondary_mmap.as_ref());

        if std::env::var("LLMRS_MADV_DONTNEED").is_ok() {
            source.madvise_dontneed();
        }

        Ok(model)
    }

    /// AUF primary 전용 `load_from_config` 구현 (W-AUF-1 C3 + W-AUF-2 C3).
    ///
    /// `resolve_secondary` (loader/mod.rs)가 AUF self-secondary 자동 활성을 담당한다.
    /// multi-dtype capability bit이 켜져 있고 `--no-self-secondary`가 꺼져 있으면
    /// 같은 mmap을 secondary로 재포장하여 swap 후보로 노출한다.
    fn load_auf_from_config(
        config: &crate::models::loader::LoadConfig,
        backend: Arc<dyn Backend>,
        memory: &dyn Memory,
    ) -> Result<Self> {
        use crate::models::loader::AufSource;
        let source = AufSource::open(
            &config.primary_source,
            config.primary_variant_choice,
            config.primary_dtype_choice,
            config.primary_eos_override,
        )?;

        // W-AUF-2 C3: resolve_secondary가 AUF self-secondary 자동 활성을 처리.
        // explicit secondary + AUF 조합은 함수 내부에서 명시 에러로 거부된다.
        let secondary_mmap = crate::models::loader::resolve_secondary(config, &source, &backend)?;

        let mut model =
            crate::models::loader::load_model(&source, backend, memory, secondary_mmap)?;
        model.quant_noise =
            compute_quant_noise(model.layers.as_slice(), model.secondary_mmap.as_ref());
        Ok(model)
    }

    /// Load a primary GGUF plus an optional secondary GGUF reserved for
    /// runtime weight swap (Phase 2). The secondary file is validated against
    /// the primary metadata (ENG-DAT-C10) and its mmap handle is retained on
    /// the model (INV-125); it is **not** consulted for the initial weight
    /// install.
    pub fn load_gguf_with_secondary(
        primary_path: &str,
        secondary_path: Option<&std::path::Path>,
        backend: Arc<dyn Backend>,
        memory: &dyn Memory,
    ) -> Result<Self> {
        use crate::models::loader::TensorSource;
        use crate::models::loader::gguf::GgufSource;
        use crate::models::weights::open_secondary;

        let source = GgufSource::open(std::path::Path::new(primary_path))?;
        let secondary_mmap = match secondary_path {
            None => None,
            Some(p) => {
                let gguf = source.gguf_file();
                let config = source.config();
                let handle = open_secondary(p, config, gguf)
                    .map_err(|e| anyhow!("secondary weight load failed: {e}"))?;
                Some(Arc::new(handle))
            }
        };
        let mut model =
            crate::models::loader::load_model(&source, backend, memory, secondary_mmap)?;

        // ENG-ALG-216: eager ε computation immediately after secondary mmap open.
        // `load_model` already stored the `Arc<SecondaryMmap>` on the model.
        model.quant_noise =
            compute_quant_noise(model.layers.as_slice(), model.secondary_mmap.as_ref());

        // LLMRS_MADV_DONTNEED: after all weights are loaded (and GPU-uploaded by
        // load_model), advise the kernel that the file page cache is expendable.
        // MmapBuffer tensors still hold Arc<Mmap> refs — the mapping stays valid —
        // but the OS can reclaim physical pages, reducing RssFile.
        if std::env::var("LLMRS_MADV_DONTNEED").is_ok() {
            source.madvise_dontneed();
        }
        Ok(model)
    }

    /// Migrate all model weight tensors from CPU to GPU zero-copy memory.
    ///
    /// Creates `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR) for each weight tensor,
    /// maps it for CPU access, copies the data from the original CPU buffer,
    /// and keeps the mapping alive. This gives single-VMA dual access:
    /// - `as_ptr()` → mapped host pointer (valid for CPU backend)
    /// - `cl_mem()` → OpenCL handle (valid for GPU kernels)
    ///
    /// Returns the number of tensors migrated.
    #[cfg(feature = "opencl")]
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path GPU migration
    pub fn migrate_weights_to_gpu(
        &mut self,
        _gpu_mem: &dyn Memory,
        gpu_backend: &Arc<dyn Backend>,
    ) -> Result<usize> {
        let mut count = 0;
        #[cfg(feature = "opencl")]
        // COLD-EXT: KV cache GPU buffer init (1회만 호출).
        let ocl_queue = gpu_backend
            .get_extension(crate::backend::EXT_OPENCL_QUEUE)
            .and_then(|a| a.downcast_ref::<crate::backend::opencl::OpenCLBackend>())
            .map(|be| be.queue.clone());
        #[cfg(not(feature = "opencl"))]
        let ocl_queue: Option<()> = None;
        let queue = ocl_queue.ok_or_else(|| anyhow!("GPU backend is not OpenCL"))?;

        let migrate_one = |t: &Tensor| -> Result<Tensor> {
            let size = t.size();
            let src_ptr = t.as_ptr();
            if src_ptr.is_null() {
                return Err(anyhow!("Cannot migrate null-pointer buffer"));
            }
            // Create ALLOC_HOST_PTR buffer, map, copy data from CPU buffer.
            let ub =
                crate::memory::opencl::unified::UnifiedBuffer::new(queue.clone(), size, t.dtype())?;
            ub.map()?; // clEnqueueMapBuffer → host pointer
            // SAFETY: src_ptr is non-null (checked above), ub.as_mut_ptr() is valid after map().
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, ub.as_mut_ptr(), size);
            }
            // Keep mapped: as_ptr()=valid, cl_mem()=valid → dual access.
            let buf: Arc<dyn Buffer> = Arc::new(ub);
            Ok(Tensor::new(t.shape().clone(), buf, gpu_backend.clone()))
        };
        // Layer weights (always small — Q4 ~6MB, F16 ~16MB max per tensor).
        // Each slot's inner `TransformerLayer` is cloned-and-mutated outside
        // the ArcSwap, then installed in one atomic step. No forward can
        // observe a half-migrated layer because the snapshot install is
        // atomic (INV-123).
        for slot in &self.layers {
            let old = slot.load_weights();
            let mut new = (*old).clone();
            new.wq = migrate_one(&new.wq)?;
            count += 1;
            new.wk = migrate_one(&new.wk)?;
            count += 1;
            new.wv = migrate_one(&new.wv)?;
            count += 1;
            new.wo = migrate_one(&new.wo)?;
            count += 1;
            new.w_gate = migrate_one(&new.w_gate)?;
            count += 1;
            new.w_up = migrate_one(&new.w_up)?;
            count += 1;
            new.w_down = migrate_one(&new.w_down)?;
            count += 1;
            new.attention_norm = migrate_one(&new.attention_norm)?;
            count += 1;
            new.ffn_norm = migrate_one(&new.ffn_norm)?;
            count += 1;
            if let Some(ref mut bias) = new.qkv_bias {
                bias.bq = migrate_one(&bias.bq)?;
                count += 1;
                bias.bk = migrate_one(&bias.bk)?;
                count += 1;
                bias.bv = migrate_one(&bias.bv)?;
                count += 1;
            }
            if let Some(ref t) = new.q_norm {
                new.q_norm = Some(migrate_one(t)?);
                count += 1;
            }
            if let Some(ref t) = new.k_norm {
                new.k_norm = Some(migrate_one(t)?);
                count += 1;
            }
            if let Some(ref t) = new.pre_ffn_norm {
                new.pre_ffn_norm = Some(migrate_one(t)?);
                count += 1;
            }
            if let Some(ref t) = new.post_ffn_norm {
                new.post_ffn_norm = Some(migrate_one(t)?);
                count += 1;
            }
            slot.store_weights_same_dtype(Arc::new(new));
        }
        self.norm = migrate_one(&self.norm)?;
        count += 1;
        // lm_head + embed_tokens: may be large (>512MB for big vocab).
        // Migrate if possible, otherwise keep on CPU with fallback paths.
        let max_alloc = gpu_backend.max_single_alloc();
        if !self.lm_head_on_cpu {
            if self.lm_head.size() <= max_alloc {
                self.lm_head = migrate_one(&self.lm_head)?;
                count += 1;
            } else {
                self.lm_head_on_cpu = true;
            }
        }
        if self.embed_tokens.size() <= max_alloc {
            self.gpu_embed_tokens = Some(migrate_one(&self.embed_tokens)?);
            count += 1;
        }
        // Set cpu_backend for gather_embed fallback path
        if self.cpu_backend.is_none() {
            self.cpu_backend = Some(crate::backend::cpu::cpu_singleton());
        }
        Ok(count)
    }

    /// Quantize the F16 lm_head tensor to Q4_0 in place.
    ///
    /// Sprint F (2026-04-26): TBT root-cause fix. F16 GGUF retains lm_head as
    /// F16 (~524 MB on Llama 3.2 1B), while Q4 GGUF derives it from Q4_0
    /// embed_tokens (~64 MB). On Adreno, an F16×F16 lm_head matmul costs
    /// ~8.4 ms/call vs ~3.8 ms/call for F16×Q4_0 — a +4.6 ms/tok gap that
    /// dominates the "ratio=1.0 mixed" TBT regression because lm_head is not
    /// covered by the AUF swap registry (transformer layers only).
    ///
    /// Quantizing lm_head at load time (one-shot, ~hundreds of ms) recovers
    /// the gap with zero per-token overhead. Embed_tokens stays F16 even on
    /// tied-weight models — the gather/lookup path requires a non-quantized
    /// dtype, and after this call lm_head becomes a separate Q4_0 tensor
    /// (the original tied F16 buffer is preserved through `gpu_embed_tokens`
    /// and `embed_tokens`).
    ///
    /// Behaviour:
    /// - F16 lm_head present (CPU or GPU): dequantize to F32, requantize to
    ///   Q4_0 blocks, build a fresh CPU Q4_0 SharedBuffer tensor, and upload
    ///   to GPU via `backend.copy_weight_from()` when the active backend is
    ///   GPU. Preserve `lm_head_on_cpu` semantics.
    /// - lm_head already Q4_0: no-op (return Ok(false)).
    /// - F32 lm_head: same path (skip dequant, quantize directly).
    /// - vocab dim must be a multiple of 32 (Q4_0 block size). Most
    ///   Llama-family vocabs satisfy this (Llama 3.2 1B vocab=128256).
    ///
    /// Returns `Ok(true)` if quantization happened, `Ok(false)` if lm_head
    /// was already Q4_0 (or smaller-than-block dims; unsupported case).
    ///
    /// `runtime_backend` is the backend that owns lm_head's `cl_mem` (i.e. the
    /// backend the model is currently running on). On GPU primary, callers
    /// must pass the OpenCL/CUDA backend here — `Tensor::backend()` may still
    /// point at the CPU loader because `copy_weight_from` preserves the
    /// source tensor's backend reference.
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path lm_head quant
    pub fn quantize_lm_head_to_q4_0(&mut self, runtime_backend: &Arc<dyn Backend>) -> Result<bool> {
        use crate::memory::host::shared::SharedBuffer;
        use crate::models::loader::convert::quantize_q4_0;
        use half::f16;

        // Already quantized — nothing to do.
        if self.lm_head.dtype() == DType::Q4_0 {
            return Ok(false);
        }
        // Q4_0 layout requires the inner dim to be a multiple of 32.
        let dims = self.lm_head.shape().dims();
        if dims.len() < 2 {
            return Err(anyhow!(
                "quantize_lm_head_to_q4_0: lm_head must be at least 2-D (got {:?})",
                dims
            ));
        }
        let cols = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if !cols.is_multiple_of(crate::quant::QK4_0) {
            return Err(anyhow!(
                "quantize_lm_head_to_q4_0: last dim ({}) is not a multiple of {} (Q4_0 block size)",
                cols,
                crate::quant::QK4_0,
            ));
        }

        let backend = runtime_backend.clone();
        // GPU dispatch covers any backend reporting `is_gpu() == true` — at
        // present OpenCL, CUDA (cuda_pc / cuda_embedded), and qnn_oppkg. The
        // legacy `name()` comparison (`"OpenCL" || "cuda-embedded"`) silently
        // skipped GPU upload for qnn_oppkg, leaving lm_head as a CPU
        // SharedBuffer that `OpenCLBackend::matmul_q4_0` would later reject.
        let is_gpu = backend.is_gpu();
        let numel = rows * cols;

        // Step 1: read lm_head data into an F32 host buffer (dequantize as
        // needed). For GPU-resident tensors with `as_ptr() == null` we route
        // through `read_buffer`. For CPU-resident or mapped UnifiedBuffer
        // tensors we read directly via the host pointer.
        let mut f32_data = vec![0.0f32; numel];
        let dtype_in = self.lm_head.dtype();
        let bytes_in = self.lm_head.size();
        let host_ptr = self.lm_head.as_ptr();
        let mut bytes_buf = vec![0u8; bytes_in];
        let src_bytes: &[u8] = if !host_ptr.is_null() {
            // Safety: host_ptr is non-null and the buffer reports `bytes_in` valid bytes.
            unsafe { std::slice::from_raw_parts(host_ptr, bytes_in) }
        } else {
            // Use the runtime backend (not Tensor::backend(), which may still
            // point at the CPU loader for tied-weight uploads).
            backend.read_buffer(&self.lm_head, &mut bytes_buf)?;
            &bytes_buf
        };
        match dtype_in {
            DType::F16 => {
                let src_u16 =
                    unsafe { std::slice::from_raw_parts(src_bytes.as_ptr() as *const u16, numel) };
                for (i, &b) in src_u16.iter().enumerate() {
                    f32_data[i] = f16::from_bits(b).to_f32();
                }
            }
            DType::F32 => {
                let src_f32 =
                    unsafe { std::slice::from_raw_parts(src_bytes.as_ptr() as *const f32, numel) };
                f32_data.copy_from_slice(src_f32);
            }
            DType::BF16 => {
                let src_u16 =
                    unsafe { std::slice::from_raw_parts(src_bytes.as_ptr() as *const u16, numel) };
                for (i, &b) in src_u16.iter().enumerate() {
                    f32_data[i] = half::bf16::from_bits(b).to_f32();
                }
            }
            other => {
                return Err(anyhow!(
                    "quantize_lm_head_to_q4_0: unsupported source dtype {:?}",
                    other
                ));
            }
        }

        // Step 2: quantize to Q4_0 blocks.
        let blocks = quantize_q4_0(&f32_data, rows, cols);
        let block_bytes = std::mem::size_of::<crate::quant::BlockQ4_0>();
        let total_bytes = blocks.len() * block_bytes;
        // Convert Vec<BlockQ4_0> → Vec<u8> without allocating twice.
        let mut bytes_out: Vec<u8> = Vec::with_capacity(total_bytes);
        // Safety: BlockQ4_0 is `#[repr(C)]` with no padding; reading its bytes
        // through `from_raw_parts` is well-defined.
        let block_bytes_view =
            unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, total_bytes) };
        bytes_out.extend_from_slice(block_bytes_view);

        // Step 3: build a CPU SharedBuffer Q4_0 tensor.
        let cpu_buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::from_vec(bytes_out, DType::Q4_0));
        let cpu_shape = self.lm_head.shape().clone();
        // Find a CPU backend reference. Prefer the model's stored cpu_backend,
        // otherwise instantiate a fresh one. SharedBuffer is dtype-only — the
        // backend is only used as a tag here.
        let cpu_backend: Arc<dyn Backend> = if let Some(ref cb) = self.cpu_backend {
            cb.clone()
        } else if !is_gpu {
            backend.clone()
        } else {
            crate::backend::cpu::cpu_singleton()
        };
        let cpu_tensor = Tensor::new(cpu_shape, cpu_buf, cpu_backend);

        // Step 4: install. When the model runs on GPU and lm_head was on GPU,
        // upload the new Q4_0 tensor through `copy_weight_from`. When lm_head
        // was already on CPU (large-vocab fallback), keep it on CPU.
        let new_lm_head = if is_gpu && !self.lm_head_on_cpu {
            backend.copy_weight_from(&cpu_tensor)?
        } else {
            cpu_tensor
        };

        // Step 5: tied-weight handling. If `gpu_embed_tokens` previously
        // shared the lm_head buffer (tied), we must keep an F16 copy alive
        // for embed gathers — don't let it become Q4_0. Detect this by
        // checking whether the stored gpu_embed_tokens still has the same
        // cl_mem as the *old* lm_head; if so, replace it with a fresh F16
        // upload from `embed_tokens` (CPU F16) before swapping lm_head.
        #[cfg(feature = "opencl")]
        if is_gpu {
            let old_lm_ptr = self.lm_head.buffer().as_ptr() as usize;
            let shares_with_embed = self
                .gpu_embed_tokens
                .as_ref()
                .map(|t| t.buffer().as_ptr() as usize == old_lm_ptr)
                .unwrap_or(false);
            if shares_with_embed {
                let fresh_embed = backend.copy_weight_from(&self.embed_tokens)?;
                self.gpu_embed_tokens = Some(fresh_embed);
            }
        }

        self.lm_head = new_lm_head;
        eprintln!(
            "[quantize_lm_head] {:?} → Q4_0 ({} → {} bytes, {} rows × {} cols)",
            dtype_in, bytes_in, total_bytes, rows, cols,
        );
        Ok(true)
    }

    /// Load lm_head Q4_0 weights directly from an AUF payload (Sprint G-1-D).
    ///
    /// # Contract
    ///
    /// `payload.bytes` layout depends on the backend variant:
    ///
    /// - **WEIGHTS_ADRENO_SOA**: bytes are `q_buf(N*16) || d_buf(N*2)` — the
    ///   same packed SOA format produced by `q4_0_aos_to_adreno_soa` and
    ///   consumed by `split_pre_converted_soa`. This function splits the bytes
    ///   and delegates to `Backend::alloc_pre_converted_soa_tensor` (registers
    ///   a `NoshuffleWeightBuffer` + noshuffle SOA registry entry). If the
    ///   backend returns `None` (CPU / driver cvt program missing), falls back
    ///   to the AOS SharedBuffer path with a runtime re-conversion warning.
    ///
    /// - **WEIGHTS_CPU_AOS / WEIGHTS_CUDA_AOS**: bytes are raw Q4_0 AOS
    ///   (18B/block, alignment-padded). Wrapped as `SharedBuffer<Q4_0>` and
    ///   uploaded via `copy_weight_from`.
    ///
    /// Tied-weight handling: same as `quantize_lm_head_to_q4_0` — if
    /// `gpu_embed_tokens` shared the old lm_head cl_mem, we replace it with a
    /// fresh F16 upload before swapping lm_head.
    ///
    /// Returns `Ok(())` on success. On error (e.g. upload failed), the model
    /// state is unchanged (install is the last step).
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path AUF lm_head load
    pub fn load_lm_head_from_auf(
        &mut self,
        payload: &crate::auf::reader::LmHeadPayload<'_>,
        runtime_backend: &Arc<dyn Backend>,
    ) -> Result<()> {
        use crate::memory::host::shared::SharedBuffer;

        let backend = runtime_backend.clone();
        // Use `is_gpu()` trait method — legacy `name()` comparison silently
        // skipped GPU upload for qnn_oppkg, leaving lm_head as a CPU
        // SharedBuffer that the OpenCL fallback's `matmul_q4_0` would reject
        // ("B (weight) is not OpenCL buffer (kind=SharedBuffer)"). Same fix
        // as `quantize_lm_head_to_q4_0` above.
        let is_gpu = backend.is_gpu();

        let dims = self.lm_head.shape().dims();
        if dims.len() < 2 {
            return Err(anyhow!(
                "load_lm_head_from_auf: lm_head must be at least 2-D (got {:?})",
                dims
            ));
        }
        // ne01 = rows (vocab_size), ne00 = cols (hidden_dim).
        let ne00 = dims[dims.len() - 1]; // cols
        let ne01: usize = dims[..dims.len() - 1].iter().product(); // rows

        if !ne00.is_multiple_of(crate::quant::QK4_0) {
            return Err(anyhow!(
                "load_lm_head_from_auf: last dim ({}) is not a multiple of {} (Q4_0 block size)",
                ne00,
                crate::quant::QK4_0,
            ));
        }
        let num_blocks = ne01 * (ne00 / crate::quant::QK4_0);
        let expected_bytes = num_blocks * std::mem::size_of::<crate::quant::BlockQ4_0>();

        // payload size must equal N * 18 in all variants (SOA total size = AOS total size).
        if payload.bytes.len() != expected_bytes {
            return Err(anyhow!(
                "load_lm_head_from_auf: payload size mismatch \
                 (expected {} bytes for {}×{} Q4_0, got {} bytes, variant={})",
                expected_bytes,
                ne01,
                ne00,
                payload.bytes.len(),
                payload.variant_tag,
            ));
        }

        let shape = self.lm_head.shape().clone();

        // ── AOS path (모든 variant 공통, G-1-F fix) ──
        //
        // **G-1-F fix (INV-135 v2)**: lm_head Q4_0 entry는 모든 backend variant에서
        // AOS 18B/block layout으로 동봉된다 (writer 측 `build_variant_payload`의
        // TAG_WEIGHTS_ADRENO_SOA 분기에서 lm_head 예외 처리 참조).
        //
        // 이유: lm_head q_buf size는 vocab×hidden 차원으로 OpenCL
        // `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` 한계를 초과 (Llama 3.2 1B: 32M texels).
        // `image1d_buffer_t` 생성 실패 → q_img=None → forward의 m=1 SOA path가
        // standard GEMV로 fall through → SOA의 d_buf만 노출된 cl_mem을 AOS layout으로
        // 잘못 해석 → silent corruption (Sprint G-1-F 디바이스 측정에서 garbage 토큰
        // "θα364..." 출력으로 확인).
        //
        // 따라서 ADRENO_SOA section 내부에서도 lm_head는 AOS bytes로 동봉되며, reader
        // 도 모든 variant에서 SharedBuffer<Q4_0> + `copy_weight_from` (Sprint F와
        // 동등 path)로 처리한다. lm_head는 image 한계로 어차피 SOA 빠른 path 사용
        // 불가능하므로 성능 손실 없음.
        //
        // payload.bytes = raw Q4_0 AOS (18B/block, alignment-padded). All variants.
        let bytes_owned: Vec<u8> = payload.bytes.to_vec();
        let cpu_buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::from_vec(bytes_owned, DType::Q4_0));
        let cpu_backend: Arc<dyn Backend> = if let Some(ref cb) = self.cpu_backend {
            cb.clone()
        } else if !is_gpu {
            backend.clone()
        } else {
            crate::backend::cpu::cpu_singleton()
        };
        let cpu_tensor = Tensor::new(shape, cpu_buf, cpu_backend);

        // Upload to GPU via copy_weight_from (same path as quantize_lm_head_to_q4_0).
        let new_lm_head = if is_gpu && !self.lm_head_on_cpu {
            backend.copy_weight_from(&cpu_tensor)?
        } else {
            cpu_tensor
        };

        // Tied-weight handling: if gpu_embed_tokens shared the old lm_head
        // cl_mem, replace it with a fresh F16 embed upload.
        #[cfg(feature = "opencl")]
        if is_gpu {
            let old_lm_ptr = self.lm_head.buffer().as_ptr() as usize;
            let shares_with_embed = self
                .gpu_embed_tokens
                .as_ref()
                .map(|t| t.buffer().as_ptr() as usize == old_lm_ptr)
                .unwrap_or(false);
            if shares_with_embed {
                let fresh_embed = backend.copy_weight_from(&self.embed_tokens)?;
                self.gpu_embed_tokens = Some(fresh_embed);
            }
        }

        self.lm_head = new_lm_head;
        eprintln!(
            "[lm_head] loaded from AUF AOS payload ({} MB, {}×{}, variant={})",
            payload.bytes.len() / (1024 * 1024),
            ne01,
            ne00,
            payload.variant_tag,
        );
        Ok(())
    }

    /// Make GPU-primary weight buffers CPU-accessible by mapping UnifiedBuffers.
    ///
    /// When `--backend opencl` with `use_zero_copy=true`, `copy_from()` creates
    /// `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR) for each weight. These start
    /// unmapped (as_ptr()=null). This method maps each buffer so the CPU backend
    /// can read weights via `as_ptr()` for GPU→CPU SwitchHw or tensor partition.
    ///
    /// When `use_zero_copy=false`, weights are in `OpenCLBuffer` (device-only).
    /// This method reads each GPU-only weight into a new `UnifiedBuffer`
    /// (ALLOC_HOST_PTR + mapped), replacing the tensor. Single VMA, no
    /// PSS double-counting.
    ///
    /// Call at startup when resilience/tensor-partition is enabled and backend is GPU.
    /// Backend-agnostic wrapper that dispatches to the active GPU's host-access
    /// remap path. CPU / CUDA UMA / non-OpenCL backends already expose weights
    /// to host pointers; only OpenCL needs explicit `UnifiedBuffer` mapping, so
    /// the wrapper is no-op for everything else. Always defined regardless of
    /// `opencl` feature — callers stay cfg-free.
    pub fn map_weights_for_host_access(&mut self, gpu_backend: &Arc<dyn Backend>) -> Result<usize> {
        #[cfg(feature = "opencl")]
        {
            if gpu_backend
                .get_extension(crate::backend::EXT_OPENCL_QUEUE)
                .is_some()
            {
                return self.map_weights_for_cpu(gpu_backend);
            }
        }
        // CPU / CUDA UMA / QNN: weights already host-pointer-accessible.
        let _ = gpu_backend;
        Ok(0)
    }

    #[cfg(feature = "opencl")]
    pub fn map_weights_for_cpu(&mut self, gpu_backend: &Arc<dyn Backend>) -> Result<usize> {
        // COLD-EXT: weight mmap startup (1회).
        let ocl_be = gpu_backend
            .get_extension(crate::backend::EXT_OPENCL_QUEUE)
            .and_then(|a| a.downcast_ref::<crate::backend::opencl::OpenCLBackend>())
            .ok_or_else(|| anyhow!("GPU backend is not OpenCL"))?;
        let queue = ocl_be.queue.clone();

        // slot 순회(partition 에 필요한 layer weight)는 slot-only `&self` 변형으로 분리
        // (AB-4 §5.5.4 — `PartitionStage` 가 model handle 없이 도달). top-level
        // norm/lm_head/gpu_embed_tokens 는 SwitchHw 전용이라 여기서만 처리한다.
        let mut count = self.map_layer_weights_for_host_access(gpu_backend)?;

        macro_rules! map_weight {
            ($t:expr) => {
                let (new, changed) = map_weight_for_host(&$t, gpu_backend, &queue)?;
                if changed {
                    $t = new;
                    count += 1;
                }
            };
        }
        map_weight!(self.norm);
        if !self.lm_head_on_cpu {
            map_weight!(self.lm_head);
        }
        // embed_tokens: check gpu_embed_tokens first
        if let Some(ref t) = self.gpu_embed_tokens {
            let (new, _) = map_weight_for_host(t, gpu_backend, &queue)?;
            self.gpu_embed_tokens = Some(new);
            count += 1;
        }
        Ok(count)
    }

    /// slot-only `&self` host-access mapping (AB-4 §5.5.4).
    ///
    /// `map_weights_for_cpu` 의 slot 순회 부분만 떼어낸 변형이다. top-level
    /// `norm`/`lm_head`/`gpu_embed_tokens` (SwitchHw 전용, `&mut self` 필요)는
    /// 건드리지 않으므로 순수 `&self`. slot 슬라이스 + gpu backend 만 받는 free
    /// function [`map_layer_slots_for_host_access`] 의 thin wrapper — `PartitionStage`
    /// 는 model handle 없이 그 free function 을 직접 호출한다.
    #[cfg(feature = "opencl")]
    pub fn map_layer_weights_for_host_access(
        &self,
        gpu_backend: &Arc<dyn Backend>,
    ) -> Result<usize> {
        map_layer_slots_for_host_access(&self.layers, gpu_backend)
    }

    /// Prepare tensor partitioning for CPU-GPU cooperative inference.
    ///
    /// Splits FFN gate/up projections row-wise into GPU and CPU partitions:
    ///   - `[0, split_row)` stays on the current (GPU) backend
    ///   - `[split_row, out_dim)` is tagged with `cpu_backend`
    ///
    /// Attention (wq/wk/wv/wo) and FFN down are not partitioned — their
    /// merge overhead exceeds the compute benefit, especially during prefill.
    ///
    /// Each slice is pre-copied into an independent buffer via `Backend::copy_from()`,
    /// ensuring GPU slices have a valid `cl_mem` handle for OpenCL kernel dispatch.
    /// Call after `map_weights_for_cpu()` so weights are CPU-accessible.
    ///
    /// Returns the number of weights actually partitioned.
    pub fn prepare_tensor_partition(
        &mut self,
        gpu_ratio: f32,
        hw: &crate::hardware::Hardware,
    ) -> Result<usize> {
        // Thin wrapper over the shared fan-out (AB-4, §5.5.3). The GPU-only
        // fast path, the `LayerDispatch::Partition` install, and the per-slot
        // gen-counter / RCU store sequencing (INV-120) all live in
        // `apply_partition_dispatch` + `LayerSlot::apply_dispatch`, shared with
        // the runtime `SetPartitionRatio` directive path (`PartitionStage`).
        crate::layers::tensor_partition::apply_partition_dispatch(&self.layers, gpu_ratio, hw)
    }

    /// Convert all Q4_0 weight tensors to noshuffle SOA layout for Adreno-optimized GEMV.
    ///
    /// For each Q4_0 weight tensor on GPU, runs the SOA conversion pipeline
    /// (GPU nibble rearrange + CPU transpose) and registers the result in the
    /// backend's noshuffle SOA registry. After this, `matmul_q4_0()` will
    /// automatically dispatch to the noshuffle GEMV kernel for decode (m==1).
    ///
    /// Additionally, unless the caller keeps the original AOS allocation
    /// (`keep_original = true`, used when CPU-accessible weights are needed
    /// for resilience `SwitchHw`, tensor partitioning, or prefill CPU chunks),
    /// each weight tensor's backing `Arc<dyn Buffer>` is replaced by a
    /// `NoshuffleWeightBuffer` that owns the SOA allocations, letting the
    /// original AOS `cl_mem` drop (≈523 MiB reclaimed on Llama 3.2 1B Q4_0).
    ///
    /// Returns the number of weight tensors converted.
    #[cfg(feature = "opencl")]
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path noshuffle SOA prep
    pub fn prepare_noshuffle_buffers(
        &mut self,
        backend: &Arc<dyn Backend>,
        keep_original: bool,
    ) -> Result<usize> {
        use crate::backend::opencl::{NoshuffleSoaEntry, OpenCLBackend, get_cl_mem};
        use crate::memory::opencl::noshuffle::NoshuffleWeightBuffer;

        // COLD-EXT: noshuffle SOA loader path. `as_any().downcast_ref` 대신
        // trait extension lookup 사용.
        let ocl_be = backend
            .get_extension(crate::backend::EXT_OPENCL_QUEUE)
            .and_then(|a| a.downcast_ref::<OpenCLBackend>())
            .ok_or_else(|| anyhow!("Not OpenCL backend"))?;

        // `LLMRS_KEEP_Q4_ORIGINAL=1` is an escape hatch for diagnostic
        // comparisons — it forces the original AOS cl_mem to stay alive even
        // when the caller would otherwise consent to swapping.
        let env_keep = std::env::var("LLMRS_KEEP_Q4_ORIGINAL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let swap_to_placeholder = !(keep_original || env_keep);

        let mut count = 0usize;
        let mut swapped = 0usize;
        let mut bytes_released: usize = 0;

        // Convert one weight tensor in place. When `allow_swap` is true and
        // the caller's closure returns `Some(bytes)` the tensor's buffer is
        // replaced with a `NoshuffleWeightBuffer` owning the new SOA
        // allocations; otherwise the original AOS cl_mem stays in the
        // backend but the SOA entry is still registered for lookup.
        //
        // `allow_swap=false` for partition sub-slices — swapping a
        // `ClSubBuffer` would also drop the parent full-weight allocation
        // prematurely.
        let mut process_weight = |tensor: &mut Tensor, allow_swap: bool| -> Result<bool> {
            if tensor.dtype() != DType::Q4_0 {
                return Ok(false);
            }
            let cl_mem = get_cl_mem(tensor.buffer().as_ref())
                .map_err(|_| anyhow!("Weight buffer has no cl_mem"))?;
            let dims = tensor.shape().dims();
            // Weight shape: [rows, cols] = [ne01, ne00]
            let (ne01, ne00) = (dims[0], dims[1]);
            let num_blocks = ne01 * ne00 / 32; // QK4_0 = 32
            let original_bytes = tensor.size();

            let (q_buf, d_buf, q_img) =
                ocl_be.convert_q4_0_to_noshuffle(cl_mem, num_blocks, ne00, ne01)?;

            // Key choice: when we swap, the tensor's post-swap cl_mem will
            // return `d_buf`, so the registry key *must* be the d_buf
            // address for `matmul_q4_0`'s `b_buf.as_ptr() as usize` to hit.
            // When we do not swap, key remains the original AOS cl_mem
            // address so lookups on the pre-existing buffer still succeed.
            let can_swap = allow_swap && swap_to_placeholder;
            let key = if can_swap {
                d_buf.as_ptr() as usize
            } else {
                cl_mem.as_ptr() as usize
            };

            if can_swap {
                // Transfer SOA ownership into the tensor buffer. The old AOS
                // `Arc<dyn Buffer>` drops when this closure returns, which in
                // turn releases the original Q4_0 cl_mem.
                let placeholder = Arc::new(NoshuffleWeightBuffer::new(
                    // `ocl::core::Mem` impls `Clone` via `clRetainMemObject`
                    // — the registry entry keeps its own reference below.
                    q_buf.clone(),
                    d_buf.clone(),
                    q_img.as_ref().cloned(),
                    ne00,
                    ne01,
                    original_bytes,
                ));
                let backend_arc = tensor.backend().clone();
                *tensor = Tensor::new(tensor.shape().clone(), placeholder, backend_arc);
                swapped += 1;
                bytes_released = bytes_released.saturating_add(original_bytes);
            }

            ocl_be.register_noshuffle_soa(
                key,
                NoshuffleSoaEntry {
                    q_buf,
                    d_buf,
                    q_img,
                    ne00,
                    ne01,
                },
            );
            Ok(true)
        };

        for slot in &self.layers {
            // RCU pattern: clone the current LayerWeights snapshot, mutate the
            // Tensor buffers in place (noshuffle swap replaces tensor.buffer),
            // then publish the new Arc via store_weights_same_dtype. Readers
            // that already hold a snapshot see the pre-swap tensors; the next
            // load picks up the noshuffle-converted ones.
            let snapshot = slot.load_weights();
            let mut layer = (*snapshot).clone();
            let mut layer_mutated = false;
            for weight in [
                &mut layer.wq,
                &mut layer.wk,
                &mut layer.wv,
                &mut layer.wo,
                &mut layer.w_gate,
                &mut layer.w_up,
                &mut layer.w_down,
            ] {
                if process_weight(weight, true)? {
                    count += 1;
                    layer_mutated = true;
                }
            }
            // Partition slices: when `--tensor-partition <r>` is active the
            // plan-path FFN dispatches onto `partition_ctx.{gate,up,down}.
            // gpu_slice()` (slices[0]). Without SOA these fall back to the AOS
            // GEMV kernel which is measurably slower than noshuffle on Adreno
            // 830 (see build_partitioned_layer_plan). Register the sub-buffer
            // cl_mems here so the plan builder can look them up via the same
            // key scheme used for full weights.
            //
            // Partition slices live on `ClSubBuffer` whose cl_mem references
            // a parent full-weight allocation — we cannot drop the parent
            // without invalidating the sub-buffers, so keep `allow_swap=false`
            // here and rely on the plan path's key matching the sub-buffer
            // cl_mem address rather than the placeholder.
            if let Some(ref mut ctx) = layer.partition_ctx {
                for weight in [
                    ctx.gate.gpu_slice_mut(),
                    ctx.up.gpu_slice_mut(),
                    ctx.down.gpu_slice_mut(),
                ] {
                    if process_weight(weight, false)? {
                        count += 1;
                        layer_mutated = true;
                    }
                }
            }
            // Phase 4-4.8: RCU publish step. Without this the swapped tensors
            // live only on the local `layer` clone and slot readers continue
            // to see the pre-swap AOS buffers, so `lookup_noshuffle_soa` keys
            // (registered against `d_buf`) never match the post-load cl_mem
            // (still the original AOS allocation). `build_plan` then aborts
            // at the "Q4_0 SOA entry missing" guard and the GPU plan path is
            // disabled for the entire session. Publish only when at least one
            // weight changed to avoid pointless ArcSwap churn.
            if layer_mutated {
                slot.store_weights_same_dtype(Arc::new(layer));
            }
        }

        // lm_head (only if on GPU). Same swap rules as layer weights.
        if !self.lm_head_on_cpu && process_weight(&mut self.lm_head, true)? {
            count += 1;
        }

        if swapped > 0 {
            eprintln!(
                "[NoShuffle] Released original Q4_0 weights after SOA conversion \
                 ({} tensors, ≈{:.1} MiB reclaimed)",
                swapped,
                bytes_released as f64 / (1024.0 * 1024.0)
            );
        } else if swap_to_placeholder {
            eprintln!(
                "[NoShuffle] SOA conversion done but no tensors swapped \
                 (non-Q4_0 weights or partition-only run)"
            );
        } else {
            eprintln!(
                "[NoShuffle] Kept original Q4_0 weights alongside SOA \
                 (keep_original={}, LLMRS_KEEP_Q4_ORIGINAL={})",
                keep_original, env_keep
            );
        }
        eprintln!(
            "[Backend] Q4_0 noshuffle SOA prepared: {} weight tensors",
            count
        );
        Ok(count)
    }

    /// Migrate all model weight tensors to CUDA pinned host memory (CudaHostBuffer).
    ///
    /// Uses `Backend::copy_weight_from()` so the backend's weight-specific
    /// allocation policy applies (e.g. `--cuda-weights-device` routes
    /// through a pure `cuMemAlloc` buffer; the default keeps the
    /// `CudaHostBuffer` zero-copy path). Norms and biases still use
    /// `copy_from` since they are loaded as activations.
    ///
    /// On Jetson (UMA) the default (pinned) path gives zero-copy CPU+GPU
    /// access; cuBLAS reads via the device pointer companion to the same
    /// physical allocation.
    ///
    /// Returns the number of tensors migrated.
    #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path CUDA migration
    pub fn migrate_weights_to_cuda(&mut self, gpu_backend: &Arc<dyn Backend>) -> Result<usize> {
        let mut count = 0;

        // Weight tensors (matmul operands): follow backend weight policy.
        let migrate_weight = |t: &Tensor| -> Result<Tensor> { gpu_backend.copy_weight_from(t) };
        // Norms / biases / the dangling activations still use the zero-copy
        // pinned host path. They are touched by CPU from time to time
        // (diagnostics, CPU fallback) and are tiny compared to matmul
        // weights, so keeping them on UMA is strictly better.
        let migrate_norm = |t: &Tensor| -> Result<Tensor> { gpu_backend.copy_from(t) };

        macro_rules! migrate_w {
            ($t:expr) => {
                $t = migrate_weight(&$t)?;
                count += 1;
            };
        }
        macro_rules! migrate_n {
            ($t:expr) => {
                $t = migrate_norm(&$t)?;
                count += 1;
            };
        }

        for slot in &self.layers {
            let old = slot.load_weights();
            let mut layer = (*old).clone();
            layer.wq = migrate_weight(&layer.wq)?;
            count += 1;
            layer.wk = migrate_weight(&layer.wk)?;
            count += 1;
            layer.wv = migrate_weight(&layer.wv)?;
            count += 1;
            layer.wo = migrate_weight(&layer.wo)?;
            count += 1;
            layer.w_gate = migrate_weight(&layer.w_gate)?;
            count += 1;
            layer.w_up = migrate_weight(&layer.w_up)?;
            count += 1;
            layer.w_down = migrate_weight(&layer.w_down)?;
            count += 1;
            layer.attention_norm = migrate_norm(&layer.attention_norm)?;
            count += 1;
            layer.ffn_norm = migrate_norm(&layer.ffn_norm)?;
            count += 1;
            if let Some(ref mut bias) = layer.qkv_bias {
                bias.bq = migrate_norm(&bias.bq)?;
                count += 1;
                bias.bk = migrate_norm(&bias.bk)?;
                count += 1;
                bias.bv = migrate_norm(&bias.bv)?;
                count += 1;
            }
            if let Some(ref t) = layer.q_norm {
                layer.q_norm = Some(migrate_norm(t)?);
                count += 1;
            }
            if let Some(ref t) = layer.k_norm {
                layer.k_norm = Some(migrate_norm(t)?);
                count += 1;
            }
            if let Some(ref t) = layer.pre_ffn_norm {
                layer.pre_ffn_norm = Some(migrate_norm(t)?);
                count += 1;
            }
            if let Some(ref t) = layer.post_ffn_norm {
                layer.post_ffn_norm = Some(migrate_norm(t)?);
                count += 1;
            }
            slot.store_weights_same_dtype(Arc::new(layer));
        }

        migrate_n!(self.norm);

        // lm_head + embed_tokens: weights — follow the weight policy.
        let max_alloc = gpu_backend.max_single_alloc();
        if !self.lm_head_on_cpu {
            if self.lm_head.size() <= max_alloc {
                migrate_w!(self.lm_head);
            } else {
                self.lm_head_on_cpu = true;
            }
        }
        if self.embed_tokens.size() <= max_alloc {
            self.gpu_embed_tokens = Some(migrate_weight(&self.embed_tokens)?);
            count += 1;
        }

        // Set cpu_backend for gather_embed fallback path
        if self.cpu_backend.is_none() {
            self.cpu_backend = Some(crate::backend::cpu::cpu_singleton());
        }

        Ok(count)
    }

    /// `forward_into` 의 KVCacheFormat trait-object fork (Phase α-K substep 3c + ①-b).
    ///
    /// `forward_into` 와 동일한 골격(embedding → layer loop → final norm → lm_head)이되, 각 layer 의
    /// KV write + attention 을 trait object 로 위임한다: **decode(seq_len=1)** = `forward_gen_fmt`
    /// (→ `fmt.write_kv` / `fmt.attention_into` single-query), **prefill(seq_len>1)** = `forward_prefill_fmt`
    /// (→ `fmt.write_kv_batch` / `fmt.attention_into` multi-token causal, ①-b). branch-by-abstraction:
    /// 기존 `forward_into` 와 공존(production 무변), `ModelForward` 가 `LLMRS_KV_FMT` 게이트 ON 시 호출.
    ///
    /// **happy-path 전용**: eviction=none / skip=0 / profile off / partition off 보장 하에
    /// score_accumulator·profiler·skip·partition 이 전부 비활성이므로 골격이 `forward_into` 의 라이브
    /// arm 과 일치한다. score 누적·eviction flip 은 (3c-evict) 후속.
    ///
    /// **bit-identical 범위**: decode 는 F16/Q4_0 KV(default=F16) 및 F32-device-only 에서만 동치
    /// (⚠️ **F32 KV + host-mapped** 는 `forward_gen` 이 inline-NEON attention 을 타는 반면 decode
    /// `attention_into` 는 `backend.attention_gen` 위임 → NOT bit-identical, `forward_gen_fmt.rs` 헤더).
    /// **prefill 은 F16/Q4_0/F32 모두 동치** — decode 와 달리 `forward_prefill` 도 inline-NEON 아닌
    /// flash 경로라 `attention_into` prefill arm 과 누산 순서 일치(`standard_format::prefill_attention`).
    ///
    /// read-plan select 검증: ascending + 범위 내. 위반 시 `None`
    /// 반환(엔진이 full read 폴백) + stderr 1회 경고(silent garbage 금지). `Token` 단위는 select 가
    /// pos(`< current_pos`), `Page` 단위는 page index(`page * page_size < current_pos` 인 page 만 유효)
    /// 라 상한을 granularity 로 분기한다. 빈 select 는 그대로 통과(format 이 빈 캐시처럼 처리, D2).
    // pub(crate): offload-path forward(`models/offload_forward.rs`)가 호출하는 공유 헬퍼.
    pub(crate) fn validate_read_plan(
        plan: &argus_extension_api::KVReadPlan,
        current_pos: usize,
    ) -> Option<&[usize]> {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);

        let upper = match plan.granularity {
            argus_extension_api::ReadGranularity::Token => current_pos,
            // page index 상한: 마지막 유효 토큰을 담는 page 까지. ceil(current_pos / page_size).
            argus_extension_api::ReadGranularity::Page { page_size } => {
                if page_size == 0 {
                    current_pos // page_size=0 은 무의미 — Token 상한으로 강등(아래 violation 검출에 맡김).
                } else {
                    current_pos.div_ceil(page_size as usize)
                }
            }
        };
        let ascending = plan.select.windows(2).all(|w| w[0] < w[1]);
        let in_range = plan.select.iter().all(|&s| s < upper);
        if ascending && in_range {
            Some(&plan.select)
        } else {
            if !WARNED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[read-plan] WARN: invalid select (ascending={ascending}, in_range={in_range}, \
                     upper={upper}) — full read 폴백. 이후 동일 경고는 억제됨."
                );
            }
            None
        }
    }

    pub fn forward_into(&self, args: TransformerModelForwardArgs) -> Result<()> {
        let input_tokens = args.input_tokens;
        let start_pos = args.start_pos;
        let fmts = args.fmts;
        let backend = args.backend;
        let memory = args.memory;
        let logits_out = args.logits_out;
        let x_gen = args.x_gen;
        let mut workspace = args.workspace;
        // Phase α-K ①-c: eval feature threading (production ModelForward 은 전부 None/false).
        let mut score_accumulator = args.score_accumulator;
        // Q running mean/var 누적기 (QueryStats, future-attention 추정용; production 은 None).
        let mut query_stats_accumulator = args.query_stats_accumulator;
        let skip_config = args.skip_config;
        let mut importance_collector = args.importance_collector;
        let cache_self_need_scores = args.cache_self_need_scores;
        // §5.9.2 Track B: IntraForward/LayerImmediate swap 의 layer-boundary hook.
        // production/eval 경로는 None → 아래 분기들이 layer 당 `is_some` branch 1회만(INV-147).
        let layer_boundary_hook = args.layer_boundary_hook;
        // read stage 핸들을 per-step 1회 캡처 → layer loop 가 재사용.
        // None(production 기본)이면 decode arm 이 layer 당 `is_some` branch 1회만(INV-147 동형).
        let read_stage = args.read_stage;
        // R-P1-1 PFA producer target (최종 prefill 청크에서만 Some). None=byte-identical(prefill arm
        // 이 layer 당 `as_mut` branch 1회). prefix_len 은 caller(ModelForward)가 buf 크기로 박음.
        let mut prefill_attn = args.prefill_attn;
        // IMP-4 per-step PFA target (answer_attention_steps dump 에서만 Some). None=byte-identical.
        let mut prefill_attn_per_row = args.prefill_attn_per_row;

        let batch_size = input_tokens.shape().dims()[0];
        let seq_len = input_tokens.shape().dims()[1];
        let hidden_size = self.config.hidden_size;
        let is_decode = seq_len == 1;

        // 1. Embedding — decode(seq_len=1) 는 caller x_gen 재사용, prefill 은 [batch, seq_len, hidden] 할당.
        let mut x = if is_decode {
            if let Some(xb) = x_gen {
                (*xb).clone()
            } else {
                let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
                Tensor::new(
                    Shape::new(vec![batch_size, seq_len, hidden_size]),
                    x_buf,
                    backend.clone(),
                )
            }
        } else {
            let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
            Tensor::new(
                Shape::new(vec![batch_size, seq_len, hidden_size]),
                x_buf,
                backend.clone(),
            )
        };
        self.gather_embed(input_tokens, &mut x, backend)?;
        if let Some(scale) = self.config.embed_scale {
            backend.scale(&mut x, scale)?;
        }

        let is_gemma3 = self.config.arch == ModelArch::Gemma3;

        // prefill owned PrefillWorkspace (forward_into:1583-1607 미러, CPU/GPU 모두 할당 — fmt 경로는
        // forward_prefill_fmt 가 항상 workspace 를 요구). needs_ws_sync = GPU 한정(drop 전 sync).
        // Phase α-K ①-d: ws_cfg 를 함수 스코프로 끌어올려 prefill arm + workspace-None decode
        // fallthrough(발산 A)가 공유. prefill 은 즉시 alloc, fallthrough 는 진입 시 lazy alloc.
        use crate::layers::workspace::{PrefillWorkspace, WorkspaceConfig as WsCfg};
        let ws_cfg = WsCfg {
            batch_size,
            dim: hidden_size,
            q_dim: self.config.num_attention_heads * self.config.head_dim,
            k_dim: self.config.num_key_value_heads * self.config.head_dim,
            v_dim: self.config.num_key_value_heads * self.config.head_dim,
            ffn_hidden: self.config.intermediate_size,
            n_heads: self.config.num_attention_heads,
            max_seq_len: 0,
        };
        let mut owned_prefill_ws: Option<PrefillWorkspace> = None;
        let mut needs_ws_sync = false;
        if !is_decode {
            owned_prefill_ws = Some(PrefillWorkspace::new(
                &ws_cfg,
                seq_len,
                memory,
                backend.clone(),
            )?);
            needs_ws_sync = backend.is_gpu();
        }

        // GPU-side score accumulator active 여부 — active 면 attention_gen 이 on-device 로 score 를
        // 처리하므로 CPU need_scores=false (forward_into:1673 미러).
        let gpu_score_active = backend.gpu_score_acc().is_some_and(|acc| acc.is_active());

        // Per-token weight snapshot (ENG-ALG-214-SNAP / INV-121) — forward_into 와 동일.
        let layer_snapshots: Vec<Arc<TransformerLayer>> =
            self.layers.iter().map(|s| s.load_weights()).collect();
        for (i, layer_arc) in layer_snapshots.iter().enumerate() {
            let layer = &**layer_arc;

            // §5.9.2 Track B wait-gate (ENG-ALG-238 / INV-149): IntraForward swap 이 layer `i` 의
            // weight 를 in-flight 로 commit 중이면, 그 새 weight 가 완전히 쓰인 뒤 forward 가 읽도록
            // (commit-before-read) blocking wait. 위에서 캡처한 layer_snapshots Arc 는 영향 없음 —
            // 다음 forward 의 ordering 만 강제. hook=None 이면 `is_some` branch 1회(INV-147).
            if let Some(hook) = layer_boundary_hook
                && let Some(evt) = hook.pending_event_for_dyn(i)
            {
                backend.wait_event_blocking(&evt)?;
            }

            // layer-skip (forward_into:1715 미러). skip_config None 이면 (false, false).
            let (s_attn, s_mlp) =
                skip_config.map_or((false, false), |sc| (sc.skip_attn(i), sc.skip_mlp(i)));

            let rope_theta = if is_gemma3 && is_local_layer(i, self.config.sliding_window_pattern) {
                self.config
                    .rope_local_theta
                    .unwrap_or(self.config.rope_theta) as f32
            } else {
                self.config.rope_theta as f32
            };
            let is_local = if is_gemma3 {
                Some(is_local_layer(i, self.config.sliding_window_pattern))
            } else {
                None
            };

            // importance snapshot before layer (forward_into:1733 미러) — prefill 2-pass 전용.
            if let Some(ref mut coll) = importance_collector {
                let x_data = x.as_slice::<f32>();
                coll.snapshot_before(x_data, seq_len, hidden_size);
            }

            if is_decode && workspace.is_none() {
                // 발산 A (Phase α-K ①-d): 구 forward_into 의 decode(seq_len==1, workspace=None)는
                // layer.forward 가 forward_prefill 로 fall-through(transformer_layer.rs:261/287,
                // degenerate 1-token). forward_prefill_fmt 는 forward_prefill 과 bit-identical(seq_len=1
                // flash 경로)이라 동일 시맨틱. warmup(기본 warmup_tokens=1)·qcf decode-X 만 진입 —
                // production 호출처(model_forward decode=workspace Some, eval decode=workspace Some,
                // 모든 prefill=seq_len>1)는 미발화(순수 additive).
                if owned_prefill_ws.is_none() {
                    owned_prefill_ws = Some(PrefillWorkspace::new(
                        &ws_cfg,
                        seq_len,
                        memory,
                        backend.clone(),
                    )?);
                    needs_ws_sync = backend.is_gpu();
                }
                let pws = owned_prefill_ws
                    .as_mut()
                    .expect("fallthrough PrefillWorkspace just allocated");
                layer.forward_prefill_fmt(
                    crate::layers::transformer_layer::ForwardPrefillFmtArgs {
                        x: &mut x,
                        fmt: &fmts[i],
                        start_pos,
                        backend,
                        pws,
                        rms_norm_eps: self.config.rms_norm_eps as f32,
                        rope_theta,
                        head_dim: self.config.head_dim,
                        batch_size,
                        seq_len,
                        dim: hidden_size,
                        skip_attn: s_attn,
                        skip_mlp: s_mlp,
                        rms_norm_add_unit: is_gemma3,
                        use_gelu_tanh: is_gemma3,
                        is_local_attn: is_local,
                        local_attn_window: self.config.sliding_window,
                        // R-P1-1: degenerate 1-token fall-through(warmup/qcf decode-X)는 PFA 미산출.
                        pfa_target: None,
                        pfa_per_row_target: None,
                    },
                )?;
            } else if is_decode {
                // need_scores = (GPU acc 미활성 시 score_acc layer-track) || cache 자가-need(quant-window AWQE).
                // forward_into:1707 + forward_gen.rs:409 AWQE OR 항 미러.
                let acc_need = if gpu_score_active {
                    false
                } else {
                    score_accumulator
                        .as_ref()
                        .is_some_and(|acc| acc.should_track_layer(i))
                };
                let need_scores = acc_need || cache_self_need_scores;
                // faithful read-plan seam. 실제 read_plan 호출은 forward_gen_fmt 내부(이 step·layer 의
                // RoPE-적용 현재 Q 를 산출한 직후·KV write 이전, 기존 seam 과 동일 캐시 뷰)로 이동했다 —
                // Quest 정본의 current-Q 선택을 충실히 실행하기 위함(running-mean 근사 제거). 활성 read
                // stage 핸들만 Faithful 라우팅으로 넘긴다. QueryStats running-mean 은 faithful Q 로 대체되어
                // 더는 공급/누적하지 않는다(QueryStats seam dormant — accumulator 미무장). read_stage=None
                // (production)이면 read_routing=None → 분기 1회, full read 직행(INV-147 byte-identical).
                let read_routing =
                    read_stage.map(
                        |rs| crate::layers::transformer_layer::ReadRouting::Faithful { stage: rs },
                    );
                let ws = workspace
                    .as_deref_mut()
                    .expect("forward_into decode requires a LayerWorkspace (seq_len=1)");
                layer.forward_gen_fmt(crate::layers::transformer_layer::ForwardGenFmtArgs {
                    x: &mut x,
                    fmt: &fmts[i],
                    start_pos,
                    backend,
                    ws,
                    rms_norm_eps: self.config.rms_norm_eps as f32,
                    rope_theta,
                    need_scores,
                    head_dim: self.config.head_dim,
                    skip_attn: s_attn,
                    skip_mlp: s_mlp,
                    rms_norm_add_unit: is_gemma3,
                    use_gelu_tanh: is_gemma3,
                    is_local_attn: is_local,
                    local_attn_window: self.config.sliding_window,
                    layer_idx: i,
                    read_routing,
                })?;
            } else {
                let pws = owned_prefill_ws
                    .as_mut()
                    .expect("forward_into prefill requires PrefillWorkspace (seq_len>1)");
                // R-P1-1: 무장 시 이 layer 의 PFA target((buf[i] 슬라이스, q_window)). `prefill_attn` 은
                // 최종 청크에서만 Some(ModelForward 가 chunk-gating) → 여기서 layer 슬라이스 reborrow.
                let pfa_target = prefill_attn
                    .as_mut()
                    .map(|(buf, qw)| (buf[i].as_mut_slice(), *qw));
                // IMP-4: 무장 시 이 layer 의 per-step target((buf[i] 슬라이스, q_window, per_head)).
                let pfa_per_row_target = prefill_attn_per_row
                    .as_mut()
                    .map(|(buf, qw, ph)| (buf[i].as_mut_slice(), *qw, *ph));
                layer.forward_prefill_fmt(
                    crate::layers::transformer_layer::ForwardPrefillFmtArgs {
                        x: &mut x,
                        fmt: &fmts[i],
                        start_pos,
                        backend,
                        pws,
                        rms_norm_eps: self.config.rms_norm_eps as f32,
                        rope_theta,
                        head_dim: self.config.head_dim,
                        batch_size,
                        seq_len,
                        dim: hidden_size,
                        skip_attn: s_attn,
                        skip_mlp: s_mlp,
                        rms_norm_add_unit: is_gemma3,
                        use_gelu_tanh: is_gemma3,
                        is_local_attn: is_local,
                        local_attn_window: self.config.sliding_window,
                        pfa_target,
                        pfa_per_row_target,
                    },
                )?;
            }

            // §5.9.2 Track B (ENG-ALG-235): layer `i` compute 직후 hook 발화 — IntraForward swap 이
            // 다음 토큰부터 dispatch_at 에 도달한 layer 의 secondary→GPU async swap 을 enqueue 한다.
            // decode(seq_len=1)에서만 동작(hook 내부 seq_len>1 가드, intra_forward_swap.rs:383).
            // hook=None 이면 `is_some` branch 1회(INV-147).
            if let Some(hook) = layer_boundary_hook {
                hook.on_layer_boundary(i, seq_len);
            }

            // importance record after layer (forward_into:1882-1891 미러) — prefill 2-pass 전용.
            if let Some(ref mut coll) = importance_collector {
                let x_data = x.as_slice::<f32>();
                coll.record_after(
                    x_data,
                    seq_len,
                    hidden_size,
                    i,
                    crate::qcf_types::SubLayer::Full,
                );
            }

            // CPU attention-score 누적 (forward_into:1893-1922 미러). workspace Some(decode)에서만 —
            // prefill 은 owned_prefill_ws 라 args.workspace=None → 자연 skip. cache_seq_len 은
            // fmts[i].current_pos() 로 취득(base trait, kv_caches[i] 대체).
            if let (Some(acc), Some(ws)) = (&mut score_accumulator, &workspace)
                && acc.should_track_layer(i)
            {
                let cache_seq_len = fmts[i].current_pos();
                let score_offset = ws.score_offset;
                let effective_len = cache_seq_len - score_offset;
                let n_heads_q = self.config.num_attention_heads;
                let stride = ws.scores.len() / n_heads_q;

                // Attribute this accumulate to layer `i` for the optional per-(layer,
                // KV-head) importance dump (IMP-1). No-op unless the dump is armed
                // (the producer only reads it under `dump_layer_head`) → INV-147.
                acc.set_current_layer(i);

                if acc.n_kv_heads() > 0 {
                    let n_kv_heads = self.config.num_key_value_heads;
                    acc.accumulate_layer_gqa(
                        &ws.scores,
                        stride,
                        effective_len,
                        n_heads_q,
                        n_kv_heads,
                        score_offset,
                    );
                } else {
                    acc.accumulate_layer(
                        &ws.scores,
                        stride,
                        effective_len,
                        n_heads_q,
                        score_offset,
                    );
                }
            }

            // QueryStats 캡처 1지점 — score 누적 seam 인접. score-active
            // 폴백 경로(workspace Some = decode)에서만, query_stats_accumulator Some + active 시
            // RoPE-적용 Q(`ws.q`, forward_gen_fmt L135 rope_inplace 직후 보존)를 GQA 환원해 누적한다.
            // hook=None happy path 는 args 슬롯이 None → `Option::is_some` 분기 1회뿐(INV-147 동형,
            // forward 출력 byte-identical). prefill 은 args.workspace=None 이라 미진입(score 누적 동형).
            if let (Some(qacc), Some(ws)) = (&mut query_stats_accumulator, &workspace)
                && qacc.is_active()
            {
                qacc.accumulate_layer(ws.q.as_slice::<f32>(), i);
            }
        }

        // step-local importance 를 cumulative 로 flush (forward_into:1926-1928 미러).
        if let Some(ref mut acc) = score_accumulator {
            acc.end_step();
        }

        // GPU score accumulator step flush (forward_into:1930-1941 미러).
        #[cfg(feature = "opencl")]
        if gpu_score_active
            && let Some(ocl_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            && let Some(gpu_acc) = ocl_be.gpu_score_acc_mut()
        {
            let cache_seq_len = fmts[0].current_pos();
            gpu_acc.end_step(ocl_be.queue.as_core(), cache_seq_len)?;
        }

        // CUDA twin of the GPU score step flush. The reduce runs on the engine's
        // default stream (same stream the attention kernels wrote score_buf on),
        // so stream ordering guarantees it sees every layer's writes.
        #[cfg(feature = "cuda")]
        if gpu_score_active
            && let Some(cuda_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::cuda_pc::CudaBackend>()
            && let Some(gpu_acc) = cuda_be.gpu_score_acc_mut()
        {
            let cache_seq_len = fmts[0].current_pos();
            let cu_stream = cuda_be.context().default_stream().cu_stream() as *mut std::ffi::c_void;
            gpu_acc.end_step(cu_stream, cache_seq_len)?;
        }

        // 3. Final Norm.
        backend.rms_norm(
            &mut x,
            &self.norm,
            self.config.rms_norm_eps as f32,
            is_gemma3,
        )?;

        // 4. Head — prefill+logits_last_only 면 마지막 토큰 hidden 만 (forward_into:1960 미러).
        if args.logits_last_only && seq_len > 1 {
            let last_offset = (seq_len - 1) * hidden_size;
            let last_buf = memory.alloc(hidden_size * 4, DType::F32)?;
            let mut x_last = Tensor::new(
                Shape::new(vec![1, 1, hidden_size]),
                last_buf,
                backend.clone(),
            );
            backend.copy_slice(&x, &mut x_last, last_offset, 0, hidden_size)?;
            if self.lm_head_on_cpu {
                self.lm_head_matmul_cpu(&x_last, logits_out, backend)?;
            } else {
                backend.matmul_transposed(&x_last, &self.lm_head, logits_out)?;
            }
        } else if self.lm_head_on_cpu {
            self.lm_head_matmul_cpu(&x, logits_out, backend)?;
        } else {
            backend.matmul_transposed(&x, &self.lm_head, logits_out)?;
        }

        // prefill owned PrefillWorkspace drop 전 GPU 커널 완료 보장 (forward_into:1989-1994).
        if needs_ws_sync {
            backend.synchronize()?;
        }

        Ok(())
    }

    /// (3p) ④-a — `build_plan` 의 fmt-handle copy-fork (Phase α-K BC Step 3).
    ///
    /// `build_plan` 본문 byte-identical 이되, KV buffer 직접 접근(`kv_caches[i].k_buffer`
    /// 등)을 `StandardFormat` lock guard 경유로 도달하도록 진입부에서 guard 슬라이스를 잡고
    /// `&KVCache` 슬라이스로 재바인딩한다. plan 빌드는 decode 첫 step lazy 1회 — lock 비용
    /// perf 무영향. production 게이트(`LLMRS_KV_FMT`) OFF 시 미발화(byte-불변).
    #[cfg(feature = "opencl")]
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L build-time plan construction; StandardFormat guard seam for cl_mem extraction
    pub fn build_plan(
        &self,
        x: &Tensor,
        logits: &Tensor,
        ws: &LayerWorkspace,
        handles: &[Arc<crate::kv::standard_format::StandardFormat>],
        backend: &Arc<dyn Backend>,
    ) -> Option<FullKernelPlan> {
        // (3p) ④-a: lock every StandardFormat guard up front and bind a
        // `&KVCache` slice so the byte-identical `build_plan` body below can
        // keep indexing `kv_caches[i]`. The cl_mem / host-ptr handles read
        // here are bound into the kernel via `set_kernel_arg` *inside*
        // `build_full_plan`, so the guards only need to outlive that call
        // (FullKernelPlan holds no borrow into the caches).
        let __fmt_guards: Vec<_> = handles.iter().map(|h| h.plan_lock()).collect();
        let kv_caches: Vec<&crate::kv::kv_cache::KVCache> =
            __fmt_guards.iter().map(|g| &g.cache).collect();
        use crate::backend::opencl::get_cl_mem;
        use crate::backend::opencl::plan::*;
        use crate::kv_cache_ops::KVLayout;

        // Phase 4-4.8 diagnostic: env-gated line-by-line trace to identify
        // which None-return path fires when the happy-path ModelForward
        // sticky-locks. Costs 1 syscall per build_plan call when unset.
        let trace = std::env::var_os("LLMRS_BUILD_PLAN_TRACE").is_some();
        macro_rules! trace_none {
            ($tag:expr) => {{
                if trace {
                    eprintln!(
                        "[build_plan-trace] None at {} ({}:{})",
                        $tag,
                        file!(),
                        line!()
                    );
                }
                return None;
            }};
        }

        // Snapshot every layer's weights once and keep the Arcs alive for the
        // duration of plan construction. The cl_mem references captured below
        // rely on these Arcs remaining in scope (INV-123 snapshot semantics).
        let layer_snaps: Vec<Arc<TransformerLayer>> =
            self.layers.iter().map(|s| s.load_weights()).collect();
        if layer_snaps.is_empty() {
            trace_none!("layer_snaps.is_empty");
        }
        // ENG-ALG-219 / weight-swap: per-layer dtype tracking. A weight swap
        // batch may leave the model in a *mixed* state (some layers F16, some
        // Q4_0) — `uniform_target_layers` deliberately excludes layer 0 for
        // ratio ≤ 0.5. Earlier the plan derived a single `is_q4_0` flag from
        // layer 0 alone, which routed every Q4_0 layer through the F16 matmul
        // kernel and produced silent garbage (Phase 3.7b ratio scan: ratios
        // 0.10/0.20/0.25/0.50 stuck in invisible-token loop). The plan now
        // accepts heterogeneous F16 + Q4_0 layers; per-layer noshuffle lookup
        // selects the correct matmul step inside `build_layer_plan`.
        //
        // GPU plan supports F16 weights or Q4_0 with noshuffle SOA conversion.
        // Reject the plan only when at least one layer has neither dtype.
        if layer_snaps.iter().any(|l| {
            let d = l.wq.dtype();
            d != crate::buffer::DType::F16 && d != crate::buffer::DType::Q4_0
        }) {
            trace_none!("wq dtype not F16/Q4_0");
        }
        let any_q4_0 = layer_snaps
            .iter()
            .any(|l| l.wq.dtype() == crate::buffer::DType::Q4_0);

        // kernel_add_row_bias expects F32 bias buffers. If any layer has
        // a QKV bias that isn't F32, fall back to the legacy path.
        if self.config.has_qkv_bias {
            for layer in &layer_snaps {
                if layer.qkv_bias.as_ref().is_some_and(|bias| {
                    bias.bq.dtype() != crate::buffer::DType::F32
                        || bias.bk.dtype() != crate::buffer::DType::F32
                        || bias.bv.dtype() != crate::buffer::DType::F32
                }) {
                    trace_none!("qkv_bias dtype not F32");
                }
            }
        }

        if backend.name() != "OpenCL" || kv_caches.is_empty() {
            trace_none!("backend not OpenCL or kv_caches empty");
        }

        // Helper macro to extract cl_mem from tensor. Diagnostic-aware: emits
        // a trace tag identifying which tensor lookup failed.
        macro_rules! cl {
            ($t:expr) => {
                match get_cl_mem($t.buffer().as_ref()) {
                    Ok(m) => m,
                    Err(_) => {
                        if trace {
                            eprintln!(
                                "[build_plan-trace] None at cl!(...) get_cl_mem failed: \
                                 expr={} ({}:{})",
                                stringify!($t),
                                file!(),
                                line!()
                            );
                        }
                        return None;
                    }
                }
            };
        }

        let dim = self.config.hidden_size;
        let head_dim = self.config.head_dim;
        let n_kv_heads = self.config.num_key_value_heads;
        let capacity = kv_caches[0].capacity();

        let (kv_pos_stride, kv_head_stride) = if kv_caches[0].layout() == KVLayout::HeadMajor {
            (head_dim as i32, (capacity * head_dim) as i32)
        } else {
            ((n_kv_heads * head_dim) as i32, head_dim as i32)
        };

        let ocl_backend = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()?;

        // For each Q4_0 layer, verify the noshuffle SOA entry (with q_img)
        // is available. Mixed-state batches install Q4_0 layers piecewise so
        // the layer 0-only check that lived here previously was insufficient
        // — if layer 0 stays F16 and layer 1 is freshly Q4_0, missing layer 1
        // SOA must still abort the GPU plan and force the legacy path.
        for (li, layer) in layer_snaps.iter().enumerate() {
            if layer.wq.dtype() != crate::buffer::DType::Q4_0 {
                continue;
            }
            let wq_key = cl!(layer.wq).as_ptr() as usize;
            if trace && li == 0 {
                let is_nb = layer
                    .wq
                    .buffer()
                    .as_any()
                    .downcast_ref::<crate::memory::opencl::noshuffle::NoshuffleWeightBuffer>()
                    .is_some();
                eprintln!(
                    "[build_plan-trace] layer0 wq key=0x{:x} is_NoshuffleWeightBuffer={}",
                    wq_key, is_nb
                );
            }
            match ocl_backend.lookup_noshuffle_soa(wq_key) {
                Some(e) if e.q_img.is_some() => {}
                Some(_) => {
                    if trace {
                        eprintln!(
                            "[build_plan-trace] None at Q4_0 SOA q_img missing layer={} \
                             ({}:{})",
                            li,
                            file!(),
                            line!()
                        );
                    }
                    return None;
                }
                None => {
                    if trace {
                        eprintln!(
                            "[build_plan-trace] None at Q4_0 SOA entry missing layer={} \
                             ({}:{})",
                            li,
                            file!(),
                            line!()
                        );
                    }
                    return None;
                }
            }
        }

        // Helper: lookup noshuffle SOA entry and return NoshufflePlanEntry if
        // available. Per-tensor dtype gate — F16 weights legitimately have no
        // SOA entry and must return None so `build_layer_plan` selects the
        // F16 matmul step. Q4_0 weights with a missing entry also return
        // None, but the per-layer abort above means we never reach this with
        // an unregistered Q4_0 weight in production.
        let ns_entry = |tensor: &Tensor| -> Option<NoshufflePlanEntry<'_>> {
            if tensor.dtype() != crate::buffer::DType::Q4_0 {
                return None;
            }
            let key = cl!(tensor).as_ptr() as usize;
            let entry = ocl_backend.lookup_noshuffle_soa(key)?;
            let q_img = entry.q_img.as_ref()?;
            Some(NoshufflePlanEntry {
                q_img,
                d_buf: &entry.d_buf,
                ne00: entry.ne00,
                ne01: entry.ne01,
            })
        };

        // Collect per-layer buffer handles
        let mut layer_bufs = Vec::new();
        let mut kv_bufs_vec = Vec::new();
        for (i, layer) in layer_snaps.iter().map(|a| &**a).enumerate() {
            // Extract optional QKV bias cl_mem handles. Non-bias models
            // short-circuit this block with (None, None, None). For bias
            // models (e.g. Qwen2), the `cl!` macro returns None from the
            // outer function if any buffer lookup fails.
            let (bq, bk, bv) = match layer.qkv_bias.as_ref() {
                Some(bias) => (Some(cl!(bias.bq)), Some(cl!(bias.bk)), Some(cl!(bias.bv))),
                None => (None, None, None),
            };
            let (partition_gate_ns, partition_up_ns, partition_down_ns) =
                match layer.partition_ctx.as_ref() {
                    Some(ctx) => (
                        ns_entry(ctx.gate.gpu_slice()),
                        ns_entry(ctx.up.gpu_slice()),
                        ns_entry(ctx.down.gpu_slice()),
                    ),
                    None => (None, None, None),
                };
            layer_bufs.push(LayerBufs {
                wq: cl!(layer.wq),
                wk: cl!(layer.wk),
                wv: cl!(layer.wv),
                wo: cl!(layer.wo),
                w_gate: cl!(layer.w_gate),
                w_up: cl!(layer.w_up),
                w_down: cl!(layer.w_down),
                attn_norm: cl!(layer.attention_norm),
                ffn_norm: cl!(layer.ffn_norm),
                bq,
                bk,
                bv,
                wq_noshuffle: ns_entry(&layer.wq),
                wk_noshuffle: ns_entry(&layer.wk),
                wv_noshuffle: ns_entry(&layer.wv),
                wo_noshuffle: ns_entry(&layer.wo),
                w_gate_noshuffle: ns_entry(&layer.w_gate),
                w_up_noshuffle: ns_entry(&layer.w_up),
                w_down_noshuffle: ns_entry(&layer.w_down),
                partition_gate_noshuffle: partition_gate_ns,
                partition_up_noshuffle: partition_up_ns,
                partition_down_noshuffle: partition_down_ns,
            });
            kv_bufs_vec.push(KvBufs {
                k_cache: cl!(kv_caches[i].k_buffer),
                v_cache: cl!(kv_caches[i].v_buffer),
            });
        }

        // Build noshuffle GEMV programs whenever *any* layer is Q4_0
        // (compile per unique ne01 dimension). In a heterogeneous F16+Q4_0
        // mixed-state batch the F16 layers contribute no entries to the
        // ne01 set; only the Q4_0 layers do, which is exactly what the
        // noshuffle programs need to cover.
        let noshuffle_programs = if any_q4_0 {
            // Collect unique ne01 values from all noshuffle entries — including
            // partition slice entries whose `ne01` (split_row or hidden_size)
            // typically differs from the full weight's ne01.
            let mut ne01_set: Vec<usize> = layer_bufs
                .iter()
                .flat_map(|lb| {
                    [
                        lb.wq_noshuffle.as_ref(),
                        lb.wk_noshuffle.as_ref(),
                        lb.wv_noshuffle.as_ref(),
                        lb.wo_noshuffle.as_ref(),
                        lb.w_gate_noshuffle.as_ref(),
                        lb.w_up_noshuffle.as_ref(),
                        lb.w_down_noshuffle.as_ref(),
                        lb.partition_gate_noshuffle.as_ref(),
                        lb.partition_up_noshuffle.as_ref(),
                        lb.partition_down_noshuffle.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .map(|e| e.ne01)
                })
                .collect();
            // Also include lm_head ne01 if applicable
            if !self.lm_head_on_cpu
                && let Some(ref e) = ns_entry(&self.lm_head)
            {
                ne01_set.push(e.ne01);
            }
            ne01_set.sort_unstable();
            ne01_set.dedup();
            match build_noshuffle_programs(
                &ocl_backend.device,
                &ocl_backend.context,
                &ocl_backend.cl_opts,
                &ne01_set,
            ) {
                Ok(progs) => Some(progs),
                Err(e) => {
                    log::warn!("Failed to build noshuffle programs for plan: {}", e);
                    if trace {
                        eprintln!(
                            "[build_plan-trace] None at build_noshuffle_programs err={} \
                             ({}:{})",
                            e,
                            file!(),
                            line!()
                        );
                    }
                    return None;
                }
            }
        } else {
            None
        };

        // lm_head noshuffle entry (Q4_0 on GPU only)
        let lm_head_noshuffle = if !self.lm_head_on_cpu {
            ns_entry(&self.lm_head)
        } else {
            None
        };

        // Mirror the runtime gate in `attention_gen` — flash attention has no
        // score output, so an active GPU score accumulator forces the legacy
        // attention path and pre-binds the persistent score buffer into the
        // attention kernel args (arg 4, `write_scores=1`, `score_stride`).
        let plan_needs_scores = ocl_backend
            .gpu_score_acc()
            .is_some_and(|acc| acc.is_active());
        let (gpu_score_buf, gpu_score_stride) = if plan_needs_scores {
            match ocl_backend.gpu_score_acc() {
                Some(acc) => (Some(acc.score_buf_mem()), acc.score_stride() as i32),
                None => (None, 0),
            }
        } else {
            (None, 0)
        };

        let full_config = FullPlanConfig {
            context: &ocl_backend.context,
            f16_program: &ocl_backend.f16_program,
            f16_l4_program: ocl_backend.f16_l4_program.as_ref(),
            simple_ops_program: &ocl_backend.simple_ops_program,
            q4_0_program: &ocl_backend.q4_0_program,
            flash_attn_f32_f16_program_dk64: ocl_backend.flash_attn_f32_f16_program_dk64.as_ref(),
            flash_attn_f32_f16_program_dk128: ocl_backend.flash_attn_f32_f16_program_dk128.as_ref(),
            needs_attention_scores: plan_needs_scores,
            gpu_score_buf,
            gpu_score_stride,
            layer_bufs,
            x_buf: cl!(x),
            q_buf: cl!(ws.q),
            k_buf: cl!(ws.k),
            v_buf: cl!(ws.v),
            out_attn_buf: cl!(ws.out_attn),
            attn_out_buf: cl!(ws.attn_out),
            gate_buf: cl!(ws.gate),
            up_buf: cl!(ws.up),
            down_buf: cl!(ws.down),
            residual_buf: cl!(ws.residual),
            // Permanent-mapped host ptr when residual UnifiedBuffer has
            // been mapped (LLMRS_PARTITION_ZCOPY_RESIDUAL or the
            // partition poll-flag auto-enable). `as_ptr()` returns null
            // when unmapped.
            residual_host_ptr: ws.residual.buffer().as_ptr(),
            kv_bufs: kv_bufs_vec,
            final_norm_buf: cl!(self.norm),
            lm_head_buf: if self.lm_head_on_cpu {
                None
            } else {
                Some(cl!(self.lm_head))
            },
            logits_buf: cl!(logits),
            dim,
            n_heads_q: self.config.num_attention_heads,
            n_kv_heads,
            head_dim,
            ffn_hidden: self.config.intermediate_size,
            vocab_size: self.config.vocab_size,
            rms_norm_eps: self.config.rms_norm_eps as f32,
            rope_theta: self.config.rope_theta as f32,
            kv_capacity: capacity,
            kv_pos_stride,
            kv_head_stride,
            is_nosub: ocl_backend.is_nosub(),
            noshuffle_programs,
            lm_head_noshuffle,
            partition_layers: {
                // Route each layer's optional PartitionContext into the plan
                // builder. When any layer has a partition_ctx, the FFN
                // segment uses `build_partitioned_layer_plan`; otherwise the
                // legacy GPU-only FFN is used per layer. See arch A.6.1.
                let mut any = false;
                let v: Vec<Option<&PartitionContext>> = layer_snaps
                    .iter()
                    .map(|l| {
                        let opt = l.partition_ctx.as_ref();
                        any |= opt.is_some();
                        opt
                    })
                    .collect();
                if any { Some(v) } else { None }
            },
            partition_workspace: ws.partition_ws.clone(),
            partition_cpu_backend: layer_snaps
                .iter()
                .find_map(|l| l.partition_ctx.as_ref().map(|c| c.cpu_backend.clone())),
            partition_use_gelu_tanh: self.config.arch == crate::model_config::ModelArch::Gemma3,
            lm_head_dtype: self.lm_head.dtype(),
            // ENG-ALG-219: pass the global ratio_generation counter so the
            // plan can detect weight swaps at execute() entry (INV-129).
            ratio_generation: self.ratio_generation.clone(),
        };

        match build_full_plan(&full_config) {
            Ok(plan) => {
                // Mixed-state diagnostic: count Q4_0 layers explicitly so a
                // partial swap shows up (e.g. "8/16 Q4_0 noshuffle") rather
                // than collapsing to a single boolean.
                let q4_0_count = layer_snaps
                    .iter()
                    .filter(|l| l.wq.dtype() == crate::buffer::DType::Q4_0)
                    .count();
                log::info!(
                    "GPU kernel plan built ({} layers, capacity={}, q4_noshuffle={}/{})",
                    self.layers.len(),
                    capacity,
                    q4_0_count,
                    self.layers.len(),
                );
                Some(plan)
            }
            Err(e) => {
                // The plan builder bails with a short message whenever an
                // opt-in flag is off (e.g. `LLMRS_PARTITION_PLAN=0`, currently
                // the default on Adreno until the upstream plan-path parity
                // bug is triaged). Demote that expected signal to `info!` so
                // partition runs — which rebuild the plan every token after a
                // fallback — do not spam warnings. Any real kernel-build /
                // cl_mem failure still surfaces via the full context chain.
                let chain = format!("{:#}", e);
                if chain.contains("LLMRS_PARTITION_PLAN=0") {
                    log::info!("GPU kernel plan skipped: {}", chain);
                } else {
                    log::warn!("Failed to build GPU kernel plan: {}", chain);
                }
                if trace {
                    eprintln!(
                        "[build_plan-trace] None at build_full_plan err chain={} ({}:{})",
                        chain,
                        file!(),
                        line!()
                    );
                }
                None
            }
        }
    }

    /// (3p) ④-a — `execute_plan` 의 fmt-handle copy-fork (Phase α-K BC Step 3).
    ///
    /// `execute_plan` 미러 — `plan.execute(...)` → `plan.execute(...)`(StandardFormat
    /// concrete-handle) 만 교체. lm_head / gather_embed 분기 동일. production 게이트
    /// (`LLMRS_KV_FMT`) OFF 시 미발화. acceptance = device 세션(plan GPU-only).
    #[cfg(feature = "opencl")]
    #[allow(clippy::too_many_arguments)]
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L hot-path plan execute (token TBT measured)
    pub fn execute_plan(
        &self,
        plan: &FullKernelPlan,
        input_tokens: &Tensor,
        start_pos: usize,
        x_gen: &mut Tensor,
        handles: &[Arc<crate::kv::standard_format::StandardFormat>],
        logits_out: &mut Tensor,
        backend: &Arc<dyn Backend>,
    ) -> Result<bool> {
        // 1. Embedding lookup: CPU gather + upload to GPU x-buffer
        self.gather_embed(input_tokens, x_gen, backend)?;

        // 2. Execute plan (all layers + final norm + optionally lm_head)
        let ocl_backend = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            .ok_or_else(|| anyhow!("Backend is not OpenCL"))?;

        let result = plan.execute(ocl_backend, start_pos, handles);
        if std::env::var_os("LLMRS_PLAN_TRACE").is_some() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static OK_CNT: AtomicU64 = AtomicU64::new(0);
            static ERR_CNT: AtomicU64 = AtomicU64::new(0);
            match &result {
                Ok(()) => {
                    let n = OK_CNT.fetch_add(1, Ordering::Relaxed) + 1;
                    if n == 1 || n.is_power_of_two() || n.is_multiple_of(32) {
                        eprintln!(
                            "[plan-trace] execute_plan ok={} err={}",
                            n,
                            ERR_CNT.load(Ordering::Relaxed)
                        );
                    }
                }
                Err(_) => {
                    ERR_CNT.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        match result {
            Ok(()) => {
                // lm_head=None gating identical to execute_plan (CPU tied embedding
                // or unsupported GPU dtype → dispatch here instead of CPU fallback).
                if plan.lm_head.is_none() {
                    if self.lm_head_on_cpu {
                        self.lm_head_matmul_cpu(x_gen, logits_out, backend)?;
                    } else {
                        backend.matmul_transposed(x_gen, &self.lm_head, logits_out)?;
                    }
                }
                Ok(true)
            }
            Err(_) => Ok(false), // plan invalidated, caller should rebuild
        }
    }

    /// Run lm_head matmul on CPU when `self.lm_head_on_cpu` is true.
    ///
    /// 1. Synchronize GPU (ensure x is ready).
    /// 2. Read the normalized hidden state `x` from GPU to a CPU buffer.
    /// 3. Perform `matmul_transposed(x_cpu, lm_head_cpu, logits_cpu)` on CPU.
    /// 4. Write the F32 logits back to the GPU `logits_out` buffer.
    ///
    /// This is only called for models with huge tied embeddings (e.g., gemma3-1b's
    /// 604 MB embed_tokens) that exceed the GPU single-buffer allocation limit.
    // pub(crate): offload-path forward(`models/offload_forward.rs`)가 호출하는 공유 헬퍼.
    pub(crate) fn lm_head_matmul_cpu(
        &self,
        x: &Tensor,
        logits_out: &mut Tensor,
        backend: &Arc<dyn Backend>,
    ) -> Result<()> {
        let cpu_be = self
            .cpu_backend
            .as_ref()
            .ok_or_else(|| anyhow!("lm_head_on_cpu requires cpu_backend"))?;

        // 1. Synchronize GPU to ensure x is fully computed.
        backend.synchronize()?;

        // 2. Read x from GPU into a CPU-backed tensor.
        let x_size = x.size(); // F32 bytes
        let x_cpu_buf = Arc::new(crate::memory::host::shared::SharedBuffer::new(
            x_size,
            DType::F32,
        ));
        let x_cpu = Tensor::new(
            x.shape().clone(),
            x_cpu_buf as Arc<dyn Buffer>,
            cpu_be.clone(),
        );
        // SAFETY: SharedBuffer guarantees a valid writable pointer of `x_size` bytes.
        let x_dst = unsafe { std::slice::from_raw_parts_mut(x_cpu.as_mut_ptr(), x_size) };
        backend.read_buffer(x, x_dst)?;

        // 3. Allocate CPU logits buffer and run matmul.
        let logits_size = logits_out.size();
        let logits_cpu_buf = Arc::new(crate::memory::host::shared::SharedBuffer::new(
            logits_size,
            DType::F32,
        ));
        let mut logits_cpu = Tensor::new(
            logits_out.shape().clone(),
            logits_cpu_buf as Arc<dyn Buffer>,
            cpu_be.clone(),
        );

        cpu_be.matmul_transposed(&x_cpu, &self.lm_head, &mut logits_cpu)?;

        // 4. Write CPU logits to GPU buffer.
        let logits_ptr = logits_cpu.as_ptr();
        // SAFETY: logits_cpu is a valid SharedBuffer with logits_size bytes.
        let logits_bytes = unsafe { std::slice::from_raw_parts(logits_ptr, logits_size) };
        backend.write_buffer(logits_out, logits_bytes)?;

        Ok(())
    }

    /// Perform embedding lookup using CPU gather, then upload the result to the
    /// target backend buffer `dst`.
    ///
    /// When `self.cpu_backend` is Some (i.e. main backend is GPU), embed_tokens
    /// lives on CPU, so we:
    ///   1. Copy input token indices to a CPU buffer (if they are on GPU).
    ///   2. Gather into a temporary CPU F32 buffer.
    ///   3. Write the bytes to the pre-allocated GPU `dst` tensor via `write_buffer`.
    ///
    /// When main backend is CPU, `cpu_backend` is None and we call `gather` directly.
    // pub(crate): offload-path forward(`models/offload_forward.rs`)가 호출하는 공유 헬퍼.
    pub(crate) fn gather_embed(
        &self,
        input_tokens: &Tensor,
        dst: &mut Tensor,
        backend: &Arc<dyn Backend>,
    ) -> Result<()> {
        if let Some(ref gpu_embed) = self.gpu_embed_tokens {
            // GPU-direct path: both embed table and indices are on GPU.
            // Single kernel launch, no CPU round-trip or blocking sync.
            backend.gather(gpu_embed, input_tokens, dst)
        } else if let Some(ref cpu_be) = self.cpu_backend {
            // Fallback GPU path (gpu_embed_tokens not available):
            // indices may be on GPU — read them to a CPU buffer first.
            let num_tokens = input_tokens.size() / 4; // u32 elements
            let hidden_size = self.config.hidden_size;
            let galloc = crate::memory::galloc::Galloc::new();

            // Read indices to CPU (even if already CPU this is a cheap memcpy)
            let idx_size = num_tokens * 4;
            let idx_buf = galloc
                .alloc(idx_size, DType::F32) // DType doesn't matter for raw bytes here
                .map_err(|e| anyhow!("gather_embed: idx alloc failed: {e}"))?;
            let cpu_indices = Tensor::new(input_tokens.shape().clone(), idx_buf, cpu_be.clone());
            let idx_bytes_mut =
                unsafe { std::slice::from_raw_parts_mut(cpu_indices.as_mut_ptr(), idx_size) };
            backend.read_buffer(input_tokens, idx_bytes_mut)?;

            // Gather on CPU
            let tmp_size = num_tokens * hidden_size * 4; // F32 bytes
            let tmp_buf = galloc
                .alloc(tmp_size, DType::F32)
                .map_err(|e| anyhow!("gather_embed: tmp alloc failed: {e}"))?;
            let mut tmp = Tensor::new(
                Shape::new(vec![1, num_tokens, hidden_size]),
                tmp_buf,
                cpu_be.clone(),
            );
            cpu_be.gather(&self.embed_tokens, &cpu_indices, &mut tmp)?;

            // Upload F32 result to GPU dst buffer
            let bytes =
                unsafe { std::slice::from_raw_parts(tmp.as_ptr(), num_tokens * hidden_size * 4) };
            backend.write_buffer(dst, bytes)
        } else {
            // CPU path: direct gather (embed_tokens already on CPU backend)
            backend.gather(&self.embed_tokens, input_tokens, dst)
        }
    }
}

// ── ε computation helper ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::memory::galloc::Galloc;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    /// Build a minimal TransformerModel-like object with CPU backend and
    /// a small embed_tokens table for gather_embed testing.
    fn make_cpu_model_with_embed(vocab: usize, dim: usize) -> (TransformerModel, Arc<dyn Backend>) {
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();

        // Build embed table: row i = [i as f32; dim]
        let buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for r in 0..vocab {
                for c in 0..dim {
                    slice[r * dim + c] = r as f32;
                }
            }
        }
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), buf, cpu_be.clone());

        // Build a trivial norm weight (all 1s)
        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        {
            let slice =
                unsafe { std::slice::from_raw_parts_mut(norm_buf.as_mut_ptr() as *mut f32, dim) };
            for v in slice.iter_mut() {
                *v = 1.0;
            }
        }
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, cpu_be.clone());
        let lm_head = norm.clone(); // dummy

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 2,
            arch: crate::model_config::ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        };

        let runtime = crate::weight::setup_runtime_resources(cpu_be.clone());
        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: None, // CPU-only model
            cpu_backend: None,
            preload_pool: std::sync::OnceLock::new(),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            quant_noise: runtime.quant_noise.clone(),
            release_worker: runtime.release_worker.clone(),
        };
        (model, cpu_be)
    }

    #[test]
    fn test_gather_embed_cpu_path() {
        // CPU model: gather_embed should produce correct rows from embed table
        let vocab = 8usize;
        let dim = 4usize;
        let (model, cpu_be) = make_cpu_model_with_embed(vocab, dim);

        let mem = Galloc::new();
        let idx_buf = mem.alloc(2 * 4, DType::F32).unwrap();
        // indices = [3, 5]
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 2) };
        idx_slice[0] = 3;
        idx_slice[1] = 5;
        let indices = Tensor::new(Shape::new(vec![1, 2]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(2 * dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 2, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, 2 * dim) };
        // Row 3 → all 3.0, row 5 → all 5.0
        for v in &result[..dim] {
            assert_eq!(*v, 3.0, "row 3 mismatch");
        }
        for v in &result[dim..] {
            assert_eq!(*v, 5.0, "row 5 mismatch");
        }
    }

    // §5.9.2 Track B: `forward_into` layer loop 의 hook 호출 횟수 == num_layers 검증은 **host 단위
    // 테스트 생략** — 본 모듈의 CPU 모델 fixture 는 전부 `layers: vec![]`(0 layers)이고, 0-layer
    // forward_into 는 lm_head 의 degenerate slicing 에서 subtract-overflow(x86.rs:387) 로 패닉한다
    // (fixture 가 forward 비대응). ≥1-layer 의 attention/RoPE/FFN 을 살리는 CPU forward fixture 는
    // 부재하며 신설은 본 작업 범위 밖(과복잡). hook 호출 *지점*(layer compute 직후 `on_layer_boundary`
    // + 진입 전 wait-gate)은 본 배선이 도입한 `if let Some(hook)` 분기 2개라 코드 리뷰 + 컴파일로
    // 보장하고, 실 count==L·정확성은 S25 device 게이트(IntraForward swap = secondary + GPU async)에서
    // 검증한다. (model_forward 의 cell-read 슬롯 구성 + weight_swap 의 cell 설치 토글은 host 검증됨.)

    #[test]
    fn test_gather_embed_cpu_backend_none_means_direct() {
        // When cpu_backend is None, gather_embed routes through backend.gather directly
        let vocab = 4usize;
        let dim = 2usize;
        let (model, cpu_be) = make_cpu_model_with_embed(vocab, dim);
        assert!(model.cpu_backend.is_none());

        let mem = Galloc::new();
        let idx_buf = mem.alloc(4, DType::F32).unwrap();
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 1) };
        idx_slice[0] = 2;
        let indices = Tensor::new(Shape::new(vec![1, 1]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 1, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, dim) };
        for v in result {
            assert_eq!(*v, 2.0, "row 2 mismatch");
        }
    }

    #[test]
    fn test_is_local_layer() {
        // pattern=None → all global
        assert!(!is_local_layer(0, None));
        assert!(!is_local_layer(5, None));

        // pattern=0 → all global (degenerate)
        assert!(!is_local_layer(0, Some(0)));
        assert!(!is_local_layer(3, Some(0)));

        // Gemma3 1B: pattern=6 → layer indices 5,11,17,23 are global (1-indexed: 6,12,18,24)
        // All other layers are local.
        assert!(is_local_layer(0, Some(6))); // layer 1 → local
        assert!(is_local_layer(1, Some(6))); // layer 2 → local
        assert!(!is_local_layer(5, Some(6))); // layer 6 → global
        assert!(is_local_layer(6, Some(6))); // layer 7 → local
        assert!(!is_local_layer(11, Some(6))); // layer 12 → global
        assert!(is_local_layer(10, Some(6))); // layer 11 → local
    }

    #[test]
    fn test_embed_scale_applied() {
        // Verify that embed_scale is applied after gather_embed.
        // Build a Gemma3-like model config with embed_scale and run gather_embed + scale.
        use crate::backend::cpu::CpuBackend;
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();

        let vocab = 4usize;
        let dim = 2usize;
        let scale = 2.0f32;

        // embed table: row i = [1.0; dim]
        let buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for v in slice.iter_mut() {
                *v = 1.0;
            }
        }
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), buf, cpu_be.clone());

        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        {
            let s =
                unsafe { std::slice::from_raw_parts_mut(norm_buf.as_mut_ptr() as *mut f32, dim) };
            s.iter_mut().for_each(|v| *v = 1.0);
        }
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, cpu_be.clone());

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 1,
            arch: ModelArch::Gemma3,
            rope_local_theta: Some(10000.0),
            sliding_window: Some(512),
            sliding_window_pattern: Some(6),
            query_pre_attn_scalar: None,
            embed_scale: Some(scale),
            weight_prefix: String::new(),
        };

        let lm_head = norm.clone();
        let runtime = crate::weight::setup_runtime_resources(cpu_be.clone());
        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: None,
            cpu_backend: None,
            preload_pool: std::sync::OnceLock::new(),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            quant_noise: runtime.quant_noise.clone(),
            release_worker: runtime.release_worker.clone(),
        };

        // Gather token 0 → should be [1.0, 1.0], then scale → [2.0, 2.0]
        let idx_buf = mem.alloc(4, DType::F32).unwrap();
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 1) };
        idx_slice[0] = 0;
        let indices = Tensor::new(Shape::new(vec![1, 1]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 1, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();
        cpu_be.scale(&mut dst, scale).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, dim) };
        for &v in result {
            assert!(
                (v - scale).abs() < 1e-5,
                "Expected {scale}, got {v} after embed_scale"
            );
        }
    }

    #[test]
    fn test_write_buffer_default_impl() {
        // Verify the default write_buffer impl copies bytes correctly (CPU backend)
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let buf = mem.alloc(8, DType::F32).unwrap();
        let mut t = Tensor::new(Shape::new(vec![2]), buf, cpu_be.clone());

        let src: Vec<u8> = (0u8..8).collect();
        cpu_be.write_buffer(&mut t, &src).unwrap();

        let got = unsafe { std::slice::from_raw_parts(t.as_ptr(), 8) };
        assert_eq!(got, src.as_slice());
    }

    #[test]
    fn test_gather_embed_gpu_direct_path() {
        // When gpu_embed_tokens is Some, gather_embed should use it directly
        // (simulated with CPU backend acting as the "GPU" backend).
        let vocab = 8usize;
        let dim = 4usize;
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();

        // Build embed table on "CPU" side
        let buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for r in 0..vocab {
                for c in 0..dim {
                    slice[r * dim + c] = r as f32;
                }
            }
        }
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), buf, cpu_be.clone());

        // Build "GPU" embed table (same data, simulates GPU copy)
        let gpu_buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let src = unsafe { std::slice::from_raw_parts(embed_tokens.as_ptr(), vocab * dim * 4) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(gpu_buf.as_mut_ptr(), vocab * dim * 4) };
            dst.copy_from_slice(src);
        }
        let gpu_embed = Tensor::new(Shape::new(vec![vocab, dim]), gpu_buf, cpu_be.clone());

        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, cpu_be.clone());
        let lm_head = norm.clone();

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: true,
            eos_token_id: 2,
            arch: crate::model_config::ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        };

        let runtime = crate::weight::setup_runtime_resources(cpu_be.clone());
        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: Some(gpu_embed),
            cpu_backend: None,
            preload_pool: std::sync::OnceLock::new(),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            quant_noise: runtime.quant_noise.clone(),
            release_worker: runtime.release_worker.clone(),
        };

        // Gather tokens [1, 6]
        let idx_buf = mem.alloc(2 * 4, DType::F32).unwrap();
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 2) };
        idx_slice[0] = 1;
        idx_slice[1] = 6;
        let indices = Tensor::new(Shape::new(vec![1, 2]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(2 * dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 2, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, 2 * dim) };
        for v in &result[..dim] {
            assert_eq!(*v, 1.0, "row 1 mismatch");
        }
        for v in &result[dim..] {
            assert_eq!(*v, 6.0, "row 6 mismatch");
        }
    }

    #[test]
    fn test_gather_embed_gpu_path_takes_priority_over_cpu_fallback() {
        // When both gpu_embed_tokens and cpu_backend are set,
        // the GPU-direct path should take priority.
        let vocab = 4usize;
        let dim = 2usize;
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();

        // CPU embed table: row i = [i as f32; dim]
        let buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for r in 0..vocab {
                for c in 0..dim {
                    slice[r * dim + c] = r as f32;
                }
            }
        }
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), buf, cpu_be.clone());

        // "GPU" embed table: row i = [(i + 100) as f32; dim]
        // Different values to prove the GPU path is used, not the CPU fallback.
        let gpu_buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(gpu_buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for r in 0..vocab {
                for c in 0..dim {
                    slice[r * dim + c] = (r + 100) as f32;
                }
            }
        }
        let gpu_embed = Tensor::new(Shape::new(vec![vocab, dim]), gpu_buf, cpu_be.clone());

        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, cpu_be.clone());
        let lm_head = norm.clone();

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: true,
            eos_token_id: 2,
            arch: crate::model_config::ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        };

        let runtime = crate::weight::setup_runtime_resources(cpu_be.clone());
        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: Some(gpu_embed),
            cpu_backend: Some(cpu_be.clone()), // both set
            preload_pool: std::sync::OnceLock::new(),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            quant_noise: runtime.quant_noise.clone(),
            release_worker: runtime.release_worker.clone(),
        };

        let idx_buf = mem.alloc(4, DType::F32).unwrap();
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 1) };
        idx_slice[0] = 2;
        let indices = Tensor::new(Shape::new(vec![1, 1]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 1, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, dim) };
        // Should get GPU values (102.0), not CPU values (2.0)
        for v in result {
            assert_eq!(*v, 102.0, "GPU path should take priority, expected 102.0");
        }
    }

    // ── load_lm_head_from_auf unit tests (Sprint G-1-E) ──────────────────────

    /// Build a minimal TransformerModel with F16 lm_head (VOCAB×HIDDEN),
    /// CPU backend only.
    fn make_cpu_model_lm_head(vocab: usize, hidden: usize) -> (TransformerModel, Arc<dyn Backend>) {
        use crate::memory::galloc::Galloc;
        use crate::memory::host::shared::SharedBuffer;
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();

        // embed_tokens: F16, (vocab, hidden)
        let embed_buf = mem.alloc(vocab * hidden * 2, DType::F16).unwrap();
        let embed = Tensor::new(Shape::new(vec![vocab, hidden]), embed_buf, cpu_be.clone());

        // norm: F32, (hidden,)
        let norm_buf = mem.alloc(hidden * 4, DType::F32).unwrap();
        let norm = Tensor::new(Shape::new(vec![hidden]), norm_buf, cpu_be.clone());

        // lm_head: F16, (vocab, hidden) — will be replaced by load_lm_head_from_auf.
        let lm_head_buf = Arc::new(SharedBuffer::new(vocab * hidden * 2, DType::F16));
        let lm_head = Tensor::new(Shape::new(vec![vocab, hidden]), lm_head_buf, cpu_be.clone());

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: hidden,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: hidden,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 2,
            arch: crate::model_config::ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        };

        let runtime = crate::weight::setup_runtime_resources(cpu_be.clone());
        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens: embed,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: None,
            cpu_backend: None,
            preload_pool: std::sync::OnceLock::new(),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            quant_noise: runtime.quant_noise.clone(),
            release_worker: runtime.release_worker.clone(),
        };
        (model, cpu_be)
    }

    /// G-1-E: load_lm_head_from_auf — CPU_AOS payload → dtype becomes Q4_0.
    #[test]
    fn load_lm_head_from_auf_cpu_aos_sets_q4_0_dtype() {
        use crate::auf::reader::LmHeadPayload;
        use crate::auf::section::TAG_WEIGHTS_CPU_AOS;
        use crate::auf::tensor_index::TensorDType;

        const VOCAB: usize = 64;
        const HIDDEN: usize = 128;
        const NUM_BLOCKS: usize = VOCAB * HIDDEN / 32;
        let aos_bytes: Vec<u8> = (0..NUM_BLOCKS * 18).map(|i| i as u8).collect();

        let (mut model, cpu_be) = make_cpu_model_lm_head(VOCAB, HIDDEN);
        let payload = LmHeadPayload {
            bytes: &aos_bytes,
            shape: [VOCAB, HIDDEN],
            dtype: TensorDType::Q4_0,
            alignment: 65536,
            variant_tag: TAG_WEIGHTS_CPU_AOS,
        };

        model.load_lm_head_from_auf(&payload, &cpu_be).unwrap();
        assert_eq!(
            model.lm_head.dtype(),
            DType::Q4_0,
            "lm_head dtype must be Q4_0 after AUF load"
        );
        assert_eq!(model.lm_head.size(), NUM_BLOCKS * 18);
    }

    /// G-1-E: load_lm_head_from_auf — size mismatch → Err.
    #[test]
    fn load_lm_head_from_auf_size_mismatch_err() {
        use crate::auf::reader::LmHeadPayload;
        use crate::auf::section::TAG_WEIGHTS_CPU_AOS;
        use crate::auf::tensor_index::TensorDType;

        const VOCAB: usize = 64;
        const HIDDEN: usize = 128;
        let wrong_bytes = vec![0u8; 100]; // clearly wrong size

        let (mut model, cpu_be) = make_cpu_model_lm_head(VOCAB, HIDDEN);
        let payload = LmHeadPayload {
            bytes: &wrong_bytes,
            shape: [VOCAB, HIDDEN],
            dtype: TensorDType::Q4_0,
            alignment: 65536,
            variant_tag: TAG_WEIGHTS_CPU_AOS,
        };

        let result = model.load_lm_head_from_auf(&payload, &cpu_be);
        assert!(result.is_err(), "size mismatch must return Err");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("payload size mismatch"),
            "error must say 'payload size mismatch', got: {msg}"
        );
    }

    /// G-1-E: load_lm_head_from_auf — bytes are preserved (CPU path).
    #[test]
    fn load_lm_head_from_auf_cpu_aos_bytes_preserved() {
        use crate::auf::reader::LmHeadPayload;
        use crate::auf::section::TAG_WEIGHTS_CPU_AOS;
        use crate::auf::tensor_index::TensorDType;

        const VOCAB: usize = 32;
        const HIDDEN: usize = 64;
        const NUM_BLOCKS: usize = VOCAB * HIDDEN / 32;
        // Deterministic pattern.
        let aos_bytes: Vec<u8> = (0..NUM_BLOCKS * 18)
            .map(|i| ((i * 7 + 13) % 256) as u8)
            .collect();

        let (mut model, cpu_be) = make_cpu_model_lm_head(VOCAB, HIDDEN);
        let payload = LmHeadPayload {
            bytes: &aos_bytes,
            shape: [VOCAB, HIDDEN],
            dtype: TensorDType::Q4_0,
            alignment: 65536,
            variant_tag: TAG_WEIGHTS_CPU_AOS,
        };

        model.load_lm_head_from_auf(&payload, &cpu_be).unwrap();

        // CPU path: SharedBuffer stores bytes directly.
        let loaded =
            unsafe { std::slice::from_raw_parts(model.lm_head.as_ptr(), model.lm_head.size()) };
        assert_eq!(
            loaded,
            &aos_bytes[..],
            "CPU_AOS: loaded bytes must match original AOS payload"
        );
    }

    /// W-AUF-1 C3: `load_from_config`이 `PrimaryFormat::Safetensors`에 대해
    /// 명시 에러를 반환하는지 확인 (Safetensors는 `load_with_dtype`을 직접 호출).
    #[test]
    fn load_from_config_rejects_safetensors_primary() {
        use crate::backend::cpu::CpuBackend;
        use crate::memory::galloc::Galloc;
        use crate::models::loader::{LoadConfig, PrimaryFormat};

        let cfg = LoadConfig {
            primary_source: std::path::PathBuf::from("/nonexistent.safetensors"),
            primary_format: PrimaryFormat::Safetensors,
            ..Default::default()
        };
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let res = TransformerModel::load_from_config(&cfg, backend, &mem);
        let err = res
            .err()
            .expect("Safetensors via load_from_config must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("Safetensors") && msg.contains("load_with_dtype"),
            "error must steer caller to load_with_dtype, got: {msg}"
        );
    }

    /// W-AUF-2 C3: `resolve_secondary`는 GGUF/Safetensors primary에서 `None`만 반환한다
    /// (GGUF는 자체 dispatch가 `open_secondary_with_backend`를 호출하므로 본 helper는 no-op).
    #[test]
    fn resolve_secondary_non_auf_returns_none() {
        use crate::backend::cpu::CpuBackend;
        use crate::models::loader::{LoadConfig, PrimaryFormat, TensorId, TensorSource};

        struct DummySource;
        impl TensorSource for DummySource {
            fn config(&self) -> &crate::model_config::ModelConfig {
                unimplemented!("not exercised by resolve_secondary non-AUF path")
            }
            fn load_tensor(
                &self,
                _id: &TensorId,
                _is_weight: bool,
                _backend: &Arc<dyn Backend>,
                _memory: &dyn crate::memory::Memory,
            ) -> Result<crate::tensor::Tensor> {
                unimplemented!()
            }
            fn load_tensor_cpu(
                &self,
                _id: &TensorId,
                _is_weight: bool,
                _memory: &dyn crate::memory::Memory,
            ) -> Result<crate::tensor::Tensor> {
                unimplemented!()
            }
            fn has_tensor(&self, _id: &TensorId) -> bool {
                false
            }
            fn weight_dtype(&self) -> DType {
                DType::F16
            }
            fn cpu_backend(&self) -> Arc<dyn Backend> {
                Arc::new(CpuBackend::new()) as Arc<dyn Backend>
            }
        }

        for fmt in [PrimaryFormat::Gguf, PrimaryFormat::Safetensors] {
            let cfg = LoadConfig {
                primary_format: fmt,
                ..Default::default()
            };
            let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
            let src = DummySource;
            let out = crate::models::loader::resolve_secondary(&cfg, &src, &backend).unwrap();
            assert!(
                out.is_none(),
                "non-AUF primary must produce None from resolve_secondary, got Some for {fmt:?}"
            );
        }
    }

    /// W-AUF-2 C3: AUF primary + explicit `--secondary-gguf`는 정책상 금지 (R10).
    #[test]
    fn resolve_secondary_rejects_explicit_for_auf_primary() {
        use crate::backend::cpu::CpuBackend;
        use crate::models::loader::{LoadConfig, PrimaryFormat, TensorId, TensorSource};

        struct DummySource;
        impl TensorSource for DummySource {
            fn config(&self) -> &crate::model_config::ModelConfig {
                unimplemented!()
            }
            fn load_tensor(
                &self,
                _id: &TensorId,
                _is_weight: bool,
                _backend: &Arc<dyn Backend>,
                _memory: &dyn crate::memory::Memory,
            ) -> Result<crate::tensor::Tensor> {
                unimplemented!()
            }
            fn load_tensor_cpu(
                &self,
                _id: &TensorId,
                _is_weight: bool,
                _memory: &dyn crate::memory::Memory,
            ) -> Result<crate::tensor::Tensor> {
                unimplemented!()
            }
            fn has_tensor(&self, _id: &TensorId) -> bool {
                false
            }
            fn weight_dtype(&self) -> DType {
                DType::F16
            }
            fn cpu_backend(&self) -> Arc<dyn Backend> {
                Arc::new(CpuBackend::new()) as Arc<dyn Backend>
            }
        }

        let cfg = LoadConfig {
            primary_format: PrimaryFormat::Auf,
            secondary_source: Some(std::path::PathBuf::from("/tmp/legacy.gguf")),
            ..Default::default()
        };
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let src = DummySource;
        let err = match crate::models::loader::resolve_secondary(&cfg, &src, &backend) {
            Ok(_) => panic!("expected AUF primary + --secondary-gguf to error"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("AUF") && msg.contains("--secondary-gguf"),
            "error must mention AUF primary + --secondary-gguf conflict, got: {msg}"
        );
    }

    // ── S2: validate_read_plan + 미지원 format capability 폴백 ──

    use argus_extension_api::{KVReadPlan, ReadGranularity};

    /// validate_read_plan: ascending + 범위 내 → Some(전체 통과).
    #[test]
    fn validate_read_plan_accepts_valid() {
        let plan = KVReadPlan {
            granularity: ReadGranularity::Token,
            select: vec![0, 2, 5],
        };
        let sel = TransformerModel::validate_read_plan(&plan, 6);
        assert_eq!(sel, Some(&[0usize, 2, 5][..]));
    }

    /// validate_read_plan: 비-ascending → None(full read 폴백).
    #[test]
    fn validate_read_plan_rejects_non_ascending() {
        let plan = KVReadPlan {
            granularity: ReadGranularity::Token,
            select: vec![0, 5, 2],
        };
        assert!(TransformerModel::validate_read_plan(&plan, 6).is_none());
    }

    /// validate_read_plan: 범위 초과(pos >= current_pos) → None.
    #[test]
    fn validate_read_plan_rejects_out_of_range() {
        let plan = KVReadPlan {
            granularity: ReadGranularity::Token,
            select: vec![0, 2, 9],
        };
        assert!(TransformerModel::validate_read_plan(&plan, 6).is_none());
    }

    /// validate_read_plan: 중복(non-strict-ascending) → None.
    #[test]
    fn validate_read_plan_rejects_duplicates() {
        let plan = KVReadPlan {
            granularity: ReadGranularity::Token,
            select: vec![1, 1, 2],
        };
        assert!(TransformerModel::validate_read_plan(&plan, 6).is_none());
    }

    /// validate_read_plan: Page 단위 상한 = ceil(current_pos/page_size). current_pos=6,page_size=2 → 상한 3.
    #[test]
    fn validate_read_plan_page_granularity_upper_bound() {
        let granularity = ReadGranularity::Page { page_size: 2 };
        // page 0,1,2 = 유효 (상한 3), page 3 = 범위 초과.
        let ok = KVReadPlan {
            granularity,
            select: vec![0, 1, 2],
        };
        assert!(TransformerModel::validate_read_plan(&ok, 6).is_some());
        let bad = KVReadPlan {
            granularity,
            select: vec![0, 3],
        };
        assert!(TransformerModel::validate_read_plan(&bad, 6).is_none());
    }

    /// validate_read_plan: 빈 select 는 통과(format 이 빈 캐시처럼 처리, D2).
    #[test]
    fn validate_read_plan_empty_passes() {
        let plan = KVReadPlan {
            granularity: ReadGranularity::Token,
            select: vec![],
        };
        assert_eq!(
            TransformerModel::validate_read_plan(&plan, 6),
            Some(&[][..])
        );
    }

    /// 미지원 format(QuantWindowFormat) 은 `as_selective_read()==None` → 엔진 full read 폴백 경로.
    /// seam 라우팅 match arm 의 guard(`as_selective_read().is_some()`)가 false → full read 직행.
    #[test]
    fn unsupported_format_falls_back_to_full_read() {
        use crate::kv::quant_window_cache::QuantizedRecentWindowCache;
        use crate::kv::quant_window_format::QuantWindowFormat;
        // QuantWindowFormat 은 as_selective_read override 없음 → base trait default None.
        // (head_dim/residual_size 는 QKKV=32 의 배수 제약)
        let fmt = QuantWindowFormat::new(0, QuantizedRecentWindowCache::new(1, 32, 64, 32));
        let cap = (&fmt as &dyn crate::format::KVCacheFormat).as_selective_read();
        assert!(
            cap.is_none(),
            "QuantWindowFormat 은 SelectiveRead 미구현 → as_selective_read()==None → full read 폴백"
        );
    }

    // ── S6: forward_into_offload read_stage → prefetch priority hint 배선 테스트 ──
    //
    // forward_into_offload 는 full e2e offload 환경(GPU + DiskStore) 없이 단위 테스트가 어렵다.
    // 따라서 S6 배선은 두 단계로 검증한다:
    //   (a) PrefetchController priority hint API 단위 테스트(prefetch.rs — 이미 추가)
    //   (b) 배선 연결 계약 테스트: OffloadForwardArgs 에 read_stage 필드가 있고 None 기본값이
    //       compile-time 으로 확인된다. 구조체 구성 가능성만 검증(forward 실행 없음).
    //
    // e2e 실행이 어려운 사유: OffloadFormat/OffloadKVCache 는 GPU backend + DiskStore/RawStore 가
    // 필요하고, forward_into_offload 는 모델 가중치 로드(파일 시스템)까지 요구한다.
    // 대안: prefetch.rs 단위 테스트(set/take + 독립성)가 배선 GREEN 의 핵심 관찰 표면이다.

    /// S6 compile-gate: OffloadForwardArgs 에 read_stage 필드가 존재하고 None 으로 초기화된다.
    ///
    /// 이 테스트는 구조체 생성을 시도해 컴파일 타임에 필드 존재를 보장한다.
    /// forward 실행 없이 타입 레벨만 검증.
    #[test]
    fn s6_offload_forward_args_has_read_stage_field() {
        use crate::backend::cpu::CpuBackend;
        use crate::buffer::DType;
        use crate::memory::galloc::Galloc;
        use crate::shape::Shape;
        use crate::tensor::Tensor;
        use std::sync::Arc;

        let backend: Arc<dyn crate::backend::Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn crate::memory::Memory> = Arc::new(Galloc::new());

        let tok_buf = memory.alloc(4, DType::U8).unwrap();
        let input_tokens = Tensor::new(Shape::new(vec![1, 1]), tok_buf, backend.clone());
        let logits_buf = memory.alloc(4 * 4, DType::F32).unwrap();
        let mut logits_out = Tensor::new(Shape::new(vec![1, 1, 4]), logits_buf, backend.clone());

        // OffloadForwardArgs 생성 — read_stage: None 필드가 컴파일 가능해야 한다.
        let _args = OffloadForwardArgs {
            input_tokens: &input_tokens,
            start_pos: 0,
            backend: &backend,
            memory: memory.as_ref(),
            logits_out: &mut logits_out,
            x_gen: None,
            workspace: None,
            prefill_workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            layer_boundary_hook: None,
            read_stage: None, // S6: 이 필드가 컴파일 통과 = 배선 연결 확인
        };
        // forward 호출 없이 타입 레벨 검증만 수행.
        // read_stage=None 이면 기존 경로와 byte-identical(INV-147).
        assert!(
            _args.read_stage.is_none(),
            "read_stage=None 기본 상태 확인(INV-147 byte-identical)"
        );
    }
}
