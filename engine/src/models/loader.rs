//! Model loader abstraction layer.
//!
//! Defines the `TensorSource` trait for format-agnostic tensor loading,
//! and the `load_model()` function that assembles a `TransformerModel` from
//! any `TensorSource` implementation.
//!
//! Current implementations:
//! - `SafetensorsSource` — HuggingFace safetensors format

pub mod auf;
pub mod convert;
pub mod gguf;
pub mod safetensors;

pub use auf::{AufDtypeChoice, AufSource, AufVariantChoice, PrimaryFormat, detect_primary_format};

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::layers::transformer_layer::{QkvBias, TransformerLayer};
use crate::memory::Memory;
use crate::model_config::ModelConfig;
use crate::models::transformer::TransformerModel;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O inference loader uses pressure-owned runtime resource setup (위계 정합 방향, design doc §7.4)
use crate::tensor::Tensor;
use crate::weight::setup_runtime_resources;

/// Runtime weight-load configuration.
///
/// Sprint 1 (W-AUF-1) 확장: AUF primary 진입을 위한 `primary_format` +
/// `primary_*_choice` 필드 추가. 기존 GGUF/Safetensors 호출처는 `..Default::default()`
/// 만 추가하여 무영향 유지.
///
/// Spec: ENG-DAT-090.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Path to the primary weight file (typically the higher-precision one,
    /// e.g. F16 GGUF). The existing `--model-path` CLI flag binds here.
    pub primary_source: PathBuf,
    /// Primary file format (Sprint 1 W-AUF-1). `Gguf`가 기본값으로 기존 호출처 보존.
    /// `Auf`는 `--model-path foo.auf` 진입 경로.
    pub primary_format: PrimaryFormat,
    /// AUF primary backend variant 선택 (Sprint 1 W-AUF-1). GGUF/Safetensors에서 무시.
    pub primary_variant_choice: AufVariantChoice,
    /// AUF primary dtype 선택 (Sprint 1 W-AUF-1). GGUF/Safetensors에서 무시.
    pub primary_dtype_choice: AufDtypeChoice,
    /// AUF TOKENIZER에 `eos_id`가 비어있을 때 CLI fallback (Sprint 1 F1).
    /// GGUF/Safetensors에서 무시.
    pub primary_eos_override: Option<u32>,
    /// Dtype supplied by the primary file. Loader infers it from the GGUF
    /// header; this becomes the initial `LayerSlot::current_dtype` for every
    /// decoder layer.
    pub default_dtype: DType,
    /// Optional secondary weight file (typically a lower-precision dtype,
    /// e.g. Q4_0 GGUF). `None` disables the swap path: initial loading is
    /// identical to the legacy behaviour and `SwapWeights` actions are
    /// no-ops. See ENG-DAT-C09.
    pub secondary_source: Option<PathBuf>,
    /// AUF self-secondary 자동 활성을 의도적으로 비활성화 (Sprint 1 W-AUF-2 사용 예정).
    /// `--no-self-secondary` CLI flag로 설정. 디버그/벤치마크 용도.
    pub disable_self_secondary: bool,
    /// Dtype selection for AUF-backed secondary files (ENG-ALG-225, Sprint D).
    ///
    /// Controls which dtype is selected from a multi-dtype AUF TENSOR_INDEX.
    /// Ignored for GGUF secondaries. `Auto` is the default and selects
    /// META.default_dtype or the first available candidate.
    pub secondary_dtype_choice: crate::models::weights::SecondaryDtypeChoice,
    /// Layout (backend variant) selection for AUF-backed secondary files.
    ///
    /// `Auto`는 build feature로 결정한 preferred variant를 우선 시도하고,
    /// 그게 AUF에 없으면 `CpuAos`로 폴백한다. `Aos`/`Soa`는 명시 강제.
    /// switch_hw cpu / partition lazy-map과 함께 쓰려면 `Aos`가 필요하다.
    pub secondary_layout_choice: crate::models::weights::SecondaryLayoutChoice,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            primary_source: PathBuf::new(),
            primary_format: PrimaryFormat::Gguf,
            primary_variant_choice: AufVariantChoice::Auto,
            primary_dtype_choice: AufDtypeChoice::Auto,
            primary_eos_override: None,
            default_dtype: DType::F16,
            secondary_source: None,
            disable_self_secondary: false,
            secondary_dtype_choice: crate::models::weights::SecondaryDtypeChoice::Auto,
            secondary_layout_choice: crate::models::weights::SecondaryLayoutChoice::Auto,
        }
    }
}

/// Internal standard tensor identifier (format-agnostic).
#[derive(Debug, Clone, Copy)]
pub enum LayerWeightKind {
    Wq,
    Wk,
    Wv,
    Wo,
    WGate,
    WUp,
    WDown,
    AttentionNorm,
    FfnNorm,
    // Gemma3 extra norms
    PreFfnNorm,
    PostFfnNorm,
    QNorm,
    KNorm,
}

/// Bias tensor kinds (Qwen2).
#[derive(Debug, Clone, Copy)]
pub enum LayerBiasKind {
    Bq,
    Bk,
    Bv,
}

/// Format-agnostic tensor identifier.
#[derive(Debug, Clone)]
pub enum TensorId {
    Embed,
    FinalNorm,
    LmHead,
    LayerWeight { layer: usize, kind: LayerWeightKind },
    LayerBias { layer: usize, kind: LayerBiasKind },
}

/// Format-agnostic tensor source trait.
///
/// Implementations translate `TensorId` into format-specific lookups
/// (e.g., safetensors weight names, GGUF tensor indices).
pub trait TensorSource {
    /// Model configuration.
    fn config(&self) -> &ModelConfig;

    /// Load a tensor onto the given backend.
    fn load_tensor(
        &self,
        id: &TensorId,
        is_weight: bool,
        backend: &Arc<dyn Backend>,
        memory: &dyn Memory,
    ) -> Result<Tensor>;

    /// Load a tensor onto CPU only (for embed_tokens, etc.).
    fn load_tensor_cpu(
        &self,
        id: &TensorId,
        is_weight: bool,
        memory: &dyn Memory,
    ) -> Result<Tensor>;

    /// Check if a tensor exists in the source.
    fn has_tensor(&self, id: &TensorId) -> bool;

    /// The weight dtype for this source.
    fn weight_dtype(&self) -> DType;

    /// CPU backend reference (for gather_embed fallback paths).
    fn cpu_backend(&self) -> Arc<dyn Backend>;

    /// AUF downcast hook. Default `None`; only [`AufSource`] overrides with `Some(self)`.
    ///
    /// `resolve_secondary`가 `&dyn TensorSource`에서 `AufSource` 핸들을 얻어
    /// `view_arc()` / `primary_variant_tag()` / `has_swap_candidate()` 등에 접근하기 위한
    /// trait-level downcast. `Any`/`downcast_ref` 대신 default method를 사용해
    /// LSP를 보존하고 `Box<dyn TensorSource>` 환경에서도 안정 동작한다 (W-AUF-2.2).
    ///
    /// [`AufSource`]: crate::models::loader::auf::AufSource
    fn as_auf(&self) -> Option<&crate::models::loader::auf::AufSource> {
        None
    }
}

/// Resolve the secondary mmap handle for an **AUF primary** source.
///
/// Sprint 1 W-AUF-2 본격 구현. GGUF primary는 `TransformerModel::load_gguf_from_config`
/// 가 `open_secondary_with_backend`를 직접 호출하므로 본 함수는 **AUF primary 경로 전용**.
///
/// 결정 순서:
/// 1. `cfg.secondary_source`가 Some (explicit `--secondary-gguf`)이면서 primary가 AUF인 경우
///    명시 에러 — AUF primary는 self-secondary가 정식 경로이므로 외부 secondary 지정은 충돌.
/// 2. AUF primary + multi-dtype capability + `!disable_self_secondary` + primary_dtype 확정 →
///    [`from_auf_self_secondary`] 자동 활성.
/// 3. 그 외 → `Ok(None)` (swap 비활성).
///
/// GGUF primary는 본 함수를 호출하지 않고 자체 dispatch를 유지한다.
pub fn resolve_secondary(
    cfg: &LoadConfig,
    source: &dyn TensorSource,
    backend: &Arc<dyn Backend>,
) -> Result<Option<Arc<crate::models::weights::SecondaryMmap>>> {
    // 본 함수는 AUF primary 진입에서만 의미가 있다. GGUF/Safetensors는 즉시 None.
    if cfg.primary_format != PrimaryFormat::Auf {
        return Ok(None);
    }

    // R10: AUF primary + explicit --secondary-gguf 조합은 정책상 금지.
    if cfg.secondary_source.is_some() {
        anyhow::bail!(
            "AUF primary와 --secondary-gguf는 함께 쓸 수 없습니다. \
             AUF는 self-secondary가 정식 경로이므로 --secondary-gguf를 빼거나 \
             --no-self-secondary로 swap 자체를 비활성화하세요."
        );
    }

    if cfg.disable_self_secondary {
        eprintln!("[AUF] self-secondary disabled by --no-self-secondary");
        return Ok(None);
    }

    let Some(auf_src) = source.as_auf() else {
        // primary_format == Auf인데 source가 AUF가 아니면 dispatch 버그.
        anyhow::bail!(
            "resolve_secondary: primary_format=Auf but source is not AufSource (dispatch bug)"
        );
    };
    if !auf_src.has_swap_candidate() {
        // multi-dtype capability bit OFF → swap 후보 없음. 정상 정지.
        return Ok(None);
    }

    let primary_dtype = auf_src.primary_dtype_tensor().ok_or_else(|| {
        anyhow::anyhow!(
            "AUF self-secondary: primary_dtype 미해결 (META.default_dtype 없음 + --primary-dtype 미지정)"
        )
    })?;
    let primary_tag = auf_src.primary_variant_tag();
    let view = auf_src.view_arc();

    let mmap = crate::models::loader::auf::from_auf_self_secondary(
        Arc::clone(&view),
        primary_tag,
        primary_dtype,
        source.config(),
    )?;

    // R8: backend가 qnn_oppkg면 RpcMem alias 경로로 promote 시도.
    // 실패 / 다른 backend / 호스트 → 그대로 SecondaryMmap::Auf 유지.
    if crate::models::weights::secondary_mmap::backend_supports_rpcmem_secondary(backend) {
        let crate::models::weights::SecondaryMmap::Auf(auf_sec) = mmap.as_ref() else {
            // from_auf_self_secondary는 항상 SecondaryMmap::Auf를 반환 — defensive.
            return Ok(Some(mmap));
        };
        let layer_index = auf_sec.layer_index.clone();
        let diag_path = auf_sec.source_path.clone();
        if let Some(promoted) =
            crate::models::weights::secondary_mmap::try_promote_auf_self_secondary_to_rpcmem(
                Arc::clone(&auf_sec.view),
                layer_index,
                diag_path,
                backend,
            )
        {
            return Ok(Some(Arc::new(promoted)));
        }
    }

    Ok(Some(mmap))
}

/// Assemble a `TransformerModel` from a `TensorSource`.
///
/// This function contains the layer-building loop and lm_head/embed logic
/// previously in `TransformerModel::load_with_dtype()`.
///
/// `secondary_mmap` is the pre-validated handle to an optional secondary
/// GGUF file (e.g. a Q4_0 companion for a primary F16 model). It is stored
/// on the model for Phase 2 `SwapExecutor` consumption; initial loading
/// always uses `source` (ENG-ALG-210).
pub fn load_model(
    source: &dyn TensorSource,
    backend: Arc<dyn Backend>,
    memory: &dyn Memory,
    secondary_mmap: Option<Arc<crate::models::weights::SecondaryMmap>>,
) -> Result<TransformerModel> {
    use crate::models::weights::LayerSlot;

    let config = source.config();
    let weight_dtype = source.weight_dtype();
    let is_cpu = backend.name().contains("CPU");
    let num_layers = config.num_hidden_layers;

    // 1. Load layers as `LayerSlot`s. Each slot wraps the initial
    //    `TransformerLayer` snapshot behind an `ArcSwap` so Phase 2 can swap
    //    it atomically (ENG-DAT-092, INV-123).
    // LISWAP-2 Phase 6.1: wrap each LayerSlot in Arc so the async swap
    // dispatcher (Phase 6.2) can take ownership of a slot handle across
    // thread boundaries.  Forward path uses Arc::deref transparently.
    let mut layers: Vec<Arc<LayerSlot>> = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let load_weight = |kind: LayerWeightKind| -> Result<Tensor> {
            source.load_tensor(
                &TensorId::LayerWeight { layer: i, kind },
                true,
                &backend,
                memory,
            )
        };
        let load_norm = |kind: LayerWeightKind| -> Result<Tensor> {
            source.load_tensor(
                &TensorId::LayerWeight { layer: i, kind },
                false,
                &backend,
                memory,
            )
        };

        let qkv_bias = if config.has_qkv_bias {
            let has_bq = source.has_tensor(&TensorId::LayerBias {
                layer: i,
                kind: LayerBiasKind::Bq,
            });
            if has_bq {
                Some(QkvBias {
                    bq: source.load_tensor(
                        &TensorId::LayerBias {
                            layer: i,
                            kind: LayerBiasKind::Bq,
                        },
                        false,
                        &backend,
                        memory,
                    )?,
                    bk: source.load_tensor(
                        &TensorId::LayerBias {
                            layer: i,
                            kind: LayerBiasKind::Bk,
                        },
                        false,
                        &backend,
                        memory,
                    )?,
                    bv: source.load_tensor(
                        &TensorId::LayerBias {
                            layer: i,
                            kind: LayerBiasKind::Bv,
                        },
                        false,
                        &backend,
                        memory,
                    )?,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Optional Gemma3 tensors
        let q_norm = if source.has_tensor(&TensorId::LayerWeight {
            layer: i,
            kind: LayerWeightKind::QNorm,
        }) {
            Some(load_norm(LayerWeightKind::QNorm)?)
        } else {
            None
        };
        let k_norm = if source.has_tensor(&TensorId::LayerWeight {
            layer: i,
            kind: LayerWeightKind::KNorm,
        }) {
            Some(load_norm(LayerWeightKind::KNorm)?)
        } else {
            None
        };
        let pre_ffn_norm = if source.has_tensor(&TensorId::LayerWeight {
            layer: i,
            kind: LayerWeightKind::PreFfnNorm,
        }) {
            Some(load_norm(LayerWeightKind::PreFfnNorm)?)
        } else {
            None
        };
        let post_ffn_norm = if source.has_tensor(&TensorId::LayerWeight {
            layer: i,
            kind: LayerWeightKind::PostFfnNorm,
        }) {
            Some(load_norm(LayerWeightKind::PostFfnNorm)?)
        } else {
            None
        };

        let layer = TransformerLayer {
            wq: load_weight(LayerWeightKind::Wq)?,
            wk: load_weight(LayerWeightKind::Wk)?,
            wv: load_weight(LayerWeightKind::Wv)?,
            wo: load_weight(LayerWeightKind::Wo)?,
            w_gate: load_weight(LayerWeightKind::WGate)?,
            w_up: load_weight(LayerWeightKind::WUp)?,
            w_down: load_weight(LayerWeightKind::WDown)?,
            attention_norm: load_norm(LayerWeightKind::AttentionNorm)?,
            ffn_norm: load_norm(LayerWeightKind::FfnNorm)?,
            qkv_bias,
            q_norm,
            k_norm,
            pre_ffn_norm,
            post_ffn_norm,
            partition_ctx: None,
        };
        // ENG-ALG-210: every slot starts at `default_dtype` (== primary
        // weight dtype). Secondary handle is cloned into the slot so the
        // Phase 2 `SwapExecutor` can locate per-layer tensor bytes without
        // touching the root container.
        let layer_idx = layers.len();
        layers.push(Arc::new(LayerSlot::new(
            layer,
            weight_dtype,
            secondary_mmap.clone(),
            layer_idx,
        )));
    }

    // 2. Embed tokens (always CPU)
    let embed_tokens =
        source.load_tensor_cpu(&TensorId::Embed, weight_dtype == DType::F16, memory)?;

    // 3. Final norm
    let norm = source.load_tensor(&TensorId::FinalNorm, false, &backend, memory)?;

    // 4. lm_head
    //
    // LLMRS_SKIP_GPU_EMBED: RSS diagnostic flag (also checked in step 6 below).
    // For tied-weight models (no separate lm_head tensor), the GPU upload happens
    // here via `backend.copy_weight_from(&embed_tokens)`.  When the flag is set
    // we force `lm_head_on_cpu = true` so the tensor stays on CPU — the decode
    // path will fall through to the CPU fallback for the final logit matmul.
    // NOTE: T2 measurement is still valid; T3+ decode may be slower or fail if
    // the CPU fallback for lm_head matmul is not implemented.
    // Labelled "RSS diag — tied-weight skip" for grep-ability.
    let skip_gpu_embed = std::env::var("LLMRS_SKIP_GPU_EMBED").is_ok();
    let has_lm_head = source.has_tensor(&TensorId::LmHead);
    let (lm_head, lm_head_on_cpu) = if has_lm_head {
        (
            source.load_tensor(&TensorId::LmHead, true, &backend, memory)?,
            false,
        )
    } else {
        // Tied weights: build lm_head from embed_tokens
        eprintln!(
            "lm_head not found, deriving from embed_tokens ({:?}) for lm_head...",
            weight_dtype
        );
        if is_cpu {
            (embed_tokens.clone(), false)
        } else if skip_gpu_embed {
            // RSS diag — tied-weight skip: keep lm_head on CPU.
            // gpu_embed_tokens will be None (set in step 6) so the gather_embed
            // fallback path is also exercised.  decode lm_head matmul uses the
            // CPU-resident tensor; correctness depends on the model's lm_head
            // matmul having a CPU fallback (check TransformerModel::forward_into).
            eprintln!(
                "[RSS-diag] LLMRS_SKIP_GPU_EMBED set: tied-weight lm_head kept on CPU \
                 (lm_head_on_cpu=true). Decode correctness requires CPU lm_head matmul path."
            );
            (embed_tokens.clone(), true)
        } else {
            let embed_size = embed_tokens.size();
            let max_alloc = backend.max_single_alloc();
            if embed_size > max_alloc {
                eprintln!(
                    "lm_head too large for GPU ({:.0} MB > {:.0} MB limit), keeping on CPU",
                    embed_size as f64 / (1024.0 * 1024.0),
                    max_alloc as f64 / (1024.0 * 1024.0),
                );
                (embed_tokens.clone(), true)
            } else {
                // Tied lm_head: a real weight — route through the weight
                // path so `--cuda-weights-device` can take effect.
                (backend.copy_weight_from(&embed_tokens)?, false)
            }
        }
    };

    // 5. CPU backend reference (only when main backend is GPU)
    let stored_cpu_backend = if is_cpu {
        None
    } else {
        Some(source.cpu_backend())
    };

    // 6. GPU-side embed_tokens
    //
    // LLMRS_SKIP_GPU_EMBED: RSS diagnostic flag (skip_gpu_embed declared in step 4).
    // When set, skip uploading embed_tokens to GPU (saves one ~30 MB GPU alloc
    // for Llama 3.2 1B Q4_0). gather_embed() falls back to:
    //   cpu_be.gather(embed_tokens) → backend.write_buffer()  (CPU→GPU per call)
    // This is slower at runtime but lets Tester measure the RSS contribution of
    // the GPU embed copy without changing any other logic.
    // Fallback path exists in TransformerModel::gather_embed (lines 2018-2050):
    // if gpu_embed_tokens is None but cpu_backend is Some, it reads indices to CPU,
    // gathers on CPU, and uploads the result — correct but ~3-5x slower per token.
    //
    // For tied-weight models: lm_head GPU upload was already skipped in step 4.
    // gpu_embed_tokens must remain None here too (cannot clone a CPU tensor as GPU).
    if skip_gpu_embed && !is_cpu {
        eprintln!(
            "[RSS-diag] LLMRS_SKIP_GPU_EMBED set: skipping GPU embed_tokens upload \
             (gather will use CPU→GPU fallback path)"
        );
    }
    let gpu_embed_tokens = if is_cpu || skip_gpu_embed {
        None
    } else if !has_lm_head && !lm_head_on_cpu {
        // Tied weights with GPU lm_head: reuse (zero extra memory)
        Some(lm_head.clone())
    } else {
        // embed_tokens is a static weight — use the weight upload path.
        match backend.copy_weight_from(&embed_tokens) {
            Ok(gpu_t) => Some(gpu_t),
            Err(e) => {
                eprintln!(
                    "Warning: failed to upload embed_tokens to GPU ({e}), \
                     gather will use CPU path"
                );
                None
            }
        }
    };

    // Clone config for the model (TensorSource owns the original)
    let model_config = ModelConfig {
        arch: config.arch,
        hidden_size: config.hidden_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        has_qkv_bias: config.has_qkv_bias,
        tie_word_embeddings: config.tie_word_embeddings,
        eos_token_id: config.eos_token_id,
        rope_local_theta: config.rope_local_theta,
        sliding_window: config.sliding_window,
        sliding_window_pattern: config.sliding_window_pattern,
        query_pre_attn_scalar: config.query_pre_attn_scalar,
        embed_scale: config.embed_scale,
        weight_prefix: config.weight_prefix.clone(),
    };

    // ENG-ALG-228 / ENG-DAT-100: spawn the async primary release worker + empty
    // ε table via the pressure-side setup helper (§13.8-O 우선순위 #2). The
    // worker retains a clone of `backend`; `TransformerModel` owns the `Arc`s
    // so they outlive any `SwapExecutor` borrow.
    let runtime = setup_runtime_resources(backend.clone());

    Ok(TransformerModel {
        config: model_config,
        layers,
        secondary_mmap,
        ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        embed_tokens,
        norm,
        lm_head,
        lm_head_on_cpu,
        gpu_embed_tokens,
        cpu_backend: stored_cpu_backend,
        preload_pool: std::sync::OnceLock::new(),
        // ε table is initialized to empty here and populated by
        // `TransformerModel::load_gguf_with_secondary` right after this
        // call, or left as empty when the caller builds the model directly
        // (e.g. tests). ENG-DAT-095, ENG-ALG-216.
        quant_noise: runtime.quant_noise,
        release_worker: runtime.release_worker,
    })
}
