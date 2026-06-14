//! `AufSource` — AUF (Argus Unified Format) primary loader implementing
//! [`TensorSource`].
//!
//! ## 책임
//! - AUF 파일을 mmap 기반 [`AufView`]로 열고 `BackendTag` + `TensorDType` 선택을
//!   생성 시점에 고정 (LSP 안정성).
//! - `META` JSON에서 [`ModelConfig`] 구축 (`from_auf_meta`).
//! - `TensorId` ↔ `TensorEntry` 매핑 (`tensor_id_to_auf`).
//! - `load_tensor` / `load_tensor_cpu`는 [`AufViewBuffer`]로 zero-copy 로드한다.
//!
//! AUF는 빌드 시점에 (1) Q/K unpermute, (2) dtype, (3) backend variant SOA/AOS
//! 가 모두 확정되므로 `load_tensor`는 GGUF 대비 단순하다 (재변환 없음).

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};

use crate::auf::{AufView, BackendTag, LAYER_IDX_CROSS, TensorDType, TensorKind, open as auf_open};
use crate::backend::Backend;
// LAYER-EXEMPT: backend_concrete_downcast — §13.8-L
use crate::backend::cpu::CpuBackend;
use crate::buffer::auf_view_buffer::AufViewBuffer;
use crate::buffer::{Buffer, DType};
use crate::memory::Memory;
use crate::model_config::{ModelArch, ModelConfig};
use crate::models::loader::auf::secondary::auf_dtype_to_engine;
use crate::models::loader::{LayerBiasKind, LayerWeightKind, TensorId, TensorSource};
use crate::shape::Shape;
use crate::tensor::Tensor;

use super::{AufDtypeChoice, AufVariantChoice};

/// AUF primary tensor source.
///
/// Lifetimes:
/// - `view`: `Arc<AufView>` 보유. mmap이 `AufSource` 폐기 시까지 유지된다.
/// - `variant_tag`/`primary_dtype`: 생성 시 결정, 이후 불변 (LSP).
pub struct AufSource {
    view: Arc<AufView>,
    variant_tag: BackendTag,
    /// 명시된 dtype 또는 `META.default_dtype` 기반 첫 후보.
    /// `lookup_tensor`의 `requested_dtype` 인자로 사용된다.
    primary_dtype: Option<TensorDType>,
    config: ModelConfig,
    cpu_backend: Arc<CpuBackend>,
    /// AUF source path (diagnostics).
    source_path: std::path::PathBuf,
}

impl AufSource {
    /// AUF 파일을 열어 primary `TensorSource`를 생성한다.
    ///
    /// Precedence:
    /// - `variant_choice` `Auto` → build feature 기반 default (`AufVariantChoice::default_tag`)
    /// - `dtype_choice` `Auto` → `META.default_dtype` 사용 (lookup 시점 결정, `primary_dtype = None`)
    /// - `eos_override` → `TOKENIZER.eos_id`가 N/A일 때 fallback
    pub fn open(
        path: &Path,
        variant_choice: AufVariantChoice,
        dtype_choice: AufDtypeChoice,
        eos_override: Option<u32>,
    ) -> Result<Self> {
        // Auto일 때만 fallback 후보 chain 시도. 명시 선택은 단발 (실패 시 명시 에러).
        let (view, variant_tag) = if matches!(variant_choice, AufVariantChoice::Auto) {
            open_with_auto_fallback(path)?
        } else {
            let tag = variant_choice.to_backend_tag();
            let v = auf_open(path, tag).map_err(|e| {
                anyhow!(
                    "AUF open failed for '{}' with explicit variant {:?}: {}",
                    path.display(),
                    tag,
                    e
                )
            })?;
            (v, tag)
        };
        let view = Arc::new(view);

        let primary_dtype = dtype_choice.to_tensor_dtype();
        let config = from_auf_meta(&view, eos_override)?;

        eprintln!(
            "[AUF] Loaded: '{}' arch={:?} variant={:?} dtype={:?} layers={} vocab={}",
            path.display(),
            config.arch,
            variant_tag,
            primary_dtype,
            config.num_hidden_layers,
            config.vocab_size,
        );

        Ok(Self {
            view,
            variant_tag,
            primary_dtype,
            config,
            cpu_backend: Arc::new(CpuBackend::new()),
            source_path: path.to_path_buf(),
        })
    }

    /// `Arc<AufView>` 공유 핸들 (W-AUF-2 self-secondary에서 사용 예정).
    pub fn view_arc(&self) -> Arc<AufView> {
        Arc::clone(&self.view)
    }

    /// 본 source가 self-secondary swap 후보를 가지는지 (W-AUF-2에서 사용).
    /// multi-dtype capability bit ON이면 true.
    pub fn has_swap_candidate(&self) -> bool {
        self.view.header.has_multi_dtype()
    }

    /// Primary backend variant tag (W-AUF-2 self-secondary에서 자기 제외용).
    pub fn primary_variant_tag(&self) -> BackendTag {
        self.variant_tag
    }

    /// Primary dtype (W-AUF-2 self-secondary에서 자기 제외용).
    ///
    /// 우선순위:
    /// 1. 생성 시 명시된 `primary_dtype` (`Some` when `--primary-dtype` ≠ Auto)
    /// 2. `META.default_dtype`
    /// 3. `None` (resolve 실패 — self-secondary 진입 차단)
    pub fn primary_dtype_tensor(&self) -> Option<TensorDType> {
        self.primary_dtype.or_else(|| {
            self.view
                .meta
                .default_dtype
                .as_deref()
                .and_then(parse_meta_dtype_str)
        })
    }

    /// 진단용 source path 반환.
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }
}

/// AUF primary `Auto` variant 자동 fallback chain.
///
/// 2026-05-20 정책 — `default_tag()`가 CpuAos이지만 AUF에 CpuAos variant가 없을 수도
/// 있으므로 (예: `qwen2.5-1.5b-q4_0.auf` ADRENO_SOA only) 후보 chain을 순회한다.
/// 순서: `default_tag()` → 그 외 변종 (AdrenoSoa, CudaAos) → 첫 성공 반환. 모두 실패 시
/// 마지막 에러 그대로 반환.
fn open_with_auto_fallback(path: &Path) -> Result<(crate::auf::AufView, BackendTag)> {
    let primary = AufVariantChoice::default_tag();
    let mut chain: Vec<BackendTag> = vec![primary];
    for tag in [
        BackendTag::CpuAos,
        BackendTag::AdrenoSoa,
        BackendTag::CudaAos,
    ] {
        if !chain.contains(&tag) {
            chain.push(tag);
        }
    }

    let mut last_err: Option<crate::auf::AufError> = None;
    for tag in &chain {
        match auf_open(path, *tag) {
            Ok(view) => {
                if *tag != primary {
                    eprintln!(
                        "[AUF] Auto: primary variant {:?} 부재 → {:?}로 fallback (path={})",
                        primary,
                        tag,
                        path.display()
                    );
                }
                return Ok((view, *tag));
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(anyhow!(
        "AUF open failed for '{}' (Auto): tried {:?}, last error: {}",
        path.display(),
        chain,
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "(none)".to_owned())
    ))
}

/// `TensorId` → AUF `(layer_idx, kind)` 매핑.
///
/// `LayerBias`는 Qwen2 qkv bias만 지원 (AUF TensorKind::Attn{Q,K,V}Bias). 그 외 모델/kind는
/// `None`을 반환하여 `has_tensor=false`로 GGUF/Safetensors와의 동작 격차를 막는다.
fn tensor_id_to_auf(id: &TensorId) -> Option<(u32, TensorKind)> {
    match id {
        TensorId::Embed => Some((LAYER_IDX_CROSS, TensorKind::Embedding)),
        TensorId::FinalNorm => Some((LAYER_IDX_CROSS, TensorKind::FinalNorm)),
        TensorId::LmHead => Some((LAYER_IDX_CROSS, TensorKind::LmHead)),
        TensorId::LayerWeight { layer, kind } => {
            let k = layer_weight_kind_to_tensor_kind(*kind)?;
            Some((*layer as u32, k))
        }
        TensorId::LayerBias { layer, kind } => {
            let k = match kind {
                LayerBiasKind::Bq => TensorKind::AttnQBias,
                LayerBiasKind::Bk => TensorKind::AttnKBias,
                LayerBiasKind::Bv => TensorKind::AttnVBias,
            };
            Some((*layer as u32, k))
        }
    }
}

/// `TensorId`가 dtype-strict lookup을 요구하는지 결정한다.
///
/// - **strict (true)** — weight matmul (Wq..WDown). multi-dtype AUF의 핵심 분기로,
///   primary_dtype에 따라 정확한 entry를 골라야 한다.
/// - **lenient (false)** — norm 텐서 (AttentionNorm, FfnNorm 등) + cross-layer
///   (Embed, FinalNorm, LmHead). 보통 F32/F16 단일 dtype으로만 빌드되므로 strict
///   매칭 시 lookup 실패. `META.default_dtype` + first-match fallback 사용.
fn is_dtype_strict_kind(id: &TensorId) -> bool {
    match id {
        TensorId::LayerWeight { kind, .. } => matches!(
            kind,
            LayerWeightKind::Wq
                | LayerWeightKind::Wk
                | LayerWeightKind::Wv
                | LayerWeightKind::Wo
                | LayerWeightKind::WGate
                | LayerWeightKind::WUp
                | LayerWeightKind::WDown
        ),
        // norm 텐서 + cross-layer + bias — dtype lenient.
        _ => false,
    }
}

fn layer_weight_kind_to_tensor_kind(kind: LayerWeightKind) -> Option<TensorKind> {
    Some(match kind {
        LayerWeightKind::Wq => TensorKind::AttnQ,
        LayerWeightKind::Wk => TensorKind::AttnK,
        LayerWeightKind::Wv => TensorKind::AttnV,
        LayerWeightKind::Wo => TensorKind::AttnO,
        LayerWeightKind::WGate => TensorKind::FfnGate,
        LayerWeightKind::WUp => TensorKind::FfnUp,
        LayerWeightKind::WDown => TensorKind::FfnDown,
        LayerWeightKind::AttentionNorm => TensorKind::AttnNorm,
        LayerWeightKind::FfnNorm => TensorKind::FfnNorm,
        // Gemma3 extras — AUF v0.1 TensorKind에 미정의. 본 sprint는 Llama/Qwen2 한정.
        LayerWeightKind::PreFfnNorm
        | LayerWeightKind::PostFfnNorm
        | LayerWeightKind::QNorm
        | LayerWeightKind::KNorm => return None,
    })
}

/// AUF `META` JSON + `TENSOR_INDEX` + `TOKENIZER`에서 `ModelConfig`를 구축한다.
fn from_auf_meta(view: &AufView, eos_override: Option<u32>) -> Result<ModelConfig> {
    let m = &view.meta;
    let arch = parse_arch_str(&m.architecture)?;
    let has_qkv_bias = matches!(arch, ModelArch::Qwen2);
    let tie_word_embeddings = infer_tie_word_embeddings(view);
    let eos_token_id = resolve_eos(view, eos_override);

    Ok(ModelConfig {
        arch,
        hidden_size: m.hidden_dim as usize,
        num_hidden_layers: m.n_layers as usize,
        num_attention_heads: m.n_heads_q as usize,
        num_key_value_heads: m.n_kv_heads as usize,
        head_dim: m.head_dim as usize,
        intermediate_size: m.ffn_dim as usize,
        vocab_size: m.vocab_size as usize,
        rms_norm_eps: m.rms_norm_epsilon,
        rope_theta: m.rope_theta,
        has_qkv_bias,
        tie_word_embeddings,
        eos_token_id,
        weight_prefix: String::new(),
        // Gemma3 fields — 본 sprint 미지원 (F4).
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
    })
}

/// AUF의 `tie_word_embeddings` 추론.
///
/// **배경**: Sprint G-1-B 이후 AUF builder는 tied 모델에서도 `token_embd.weight`를
/// source로 사용해 `LmHead` Q4_0 entry를 별도 precompute한다. 따라서 단순히
/// "LmHead entry 존재 여부"로 tied/separate를 판정할 수 없다 (예전 추론 버그).
///
/// **정확한 추론**: GGUF는 `find_tensor("output.weight").is_none()`을 신호로 사용.
/// AUF에서는 `token_embd.weight`의 shape와 `LmHead` entry의 shape를 비교하여
/// 같으면 tied로 추론한다. tied 모델의 lm_head는 token_embd와 동일한
/// `[vocab_size, hidden_dim]` shape를 갖는다.
///
/// **fallback**: token_embd 또는 LmHead entry가 없으면 tied=false 보수 추론.
fn infer_tie_word_embeddings(view: &AufView) -> bool {
    let lm_head = match view.tensor_index.find_lm_head_entry() {
        Some(e) => e,
        None => return true, // LmHead entry 없으면 항상 tied
    };
    // token_embd.weight entry (kind=Embedding, layer_idx=LAYER_IDX_CROSS).
    let embed = view.tensor_index.entries.iter().find(|e| {
        e.kind == crate::auf::TensorKind::Embedding.as_u32() && e.layer_idx == LAYER_IDX_CROSS
    });
    let Some(embed) = embed else {
        return false; // token_embd 없으면 lm_head를 별도 source로 가정 (separate)
    };
    embed.shape == lm_head.shape
}

fn parse_arch_str(s: &str) -> Result<ModelArch> {
    match s.to_ascii_lowercase().as_str() {
        "llama" => Ok(ModelArch::Llama),
        "qwen2" | "qwen2.5" => Ok(ModelArch::Qwen2),
        "gemma3" => Ok(ModelArch::Gemma3),
        other => Err(anyhow!(
            "AUF: unsupported architecture '{other}' (expected llama / qwen2 / gemma3)"
        )),
    }
}

/// EOS token id 결정 우선순위:
/// 1. AUF TOKENIZER section의 `eos_id` (>= 0).
/// 2. CLI override (`--eos-token-id`).
/// 3. `u32::MAX` fallback + stderr warning (EOS 정지 조건 비활성).
fn resolve_eos(view: &AufView, cli_override: Option<u32>) -> u32 {
    if view.tokenizer.eos_id >= 0 {
        return view.tokenizer.eos_id as u32;
    }
    cli_override.unwrap_or_else(|| {
        eprintln!(
            "[Warning] AUF TOKENIZER에 eos_id가 없습니다. --eos-token-id로 명시하지 않으면 EOS 정지 조건이 동작하지 않습니다."
        );
        u32::MAX
    })
}

// ── TensorSource impl ────────────────────────────────────────────────

impl TensorSource for AufSource {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weight_dtype(&self) -> DType {
        // primary_dtype이 명시되었으면 그것을 사용, 아니면 META.default_dtype 또는 F16 default.
        let tdt = self.primary_dtype.or_else(|| {
            self.view
                .meta
                .default_dtype
                .as_deref()
                .and_then(parse_meta_dtype_str)
        });
        match tdt {
            Some(TensorDType::F32) => DType::F32,
            Some(TensorDType::F16) => DType::F16,
            Some(TensorDType::Q4_0) => DType::Q4_0,
            Some(TensorDType::Q8_0) => DType::Q8_0,
            // BF16/Q4_1/U8은 engine DType에 직접 매핑이 없으므로 F16 fallback.
            _ => DType::F16,
        }
    }

    fn cpu_backend(&self) -> Arc<dyn Backend> {
        self.cpu_backend.clone() as Arc<dyn Backend>
    }

    fn has_tensor(&self, id: &TensorId) -> bool {
        // tied 모델: AUF가 LmHead Q4_0 entry를 별도 precompute(Sprint G-1-B)했어도
        // forward는 GGUF tied path (lm_head = embed_tokens.clone())와 동일하게 동작해야
        // 한다. has_tensor(LmHead) = false를 반환하여 load_model이 tied 분기로 가게 한다.
        // AUF의 LmHead precompute는 향후 별도 wire-up sprint에서 활성화 예정.
        if matches!(id, TensorId::LmHead) && self.config.tie_word_embeddings {
            return false;
        }
        let Some((layer_idx, kind)) = tensor_id_to_auf(id) else {
            return false;
        };
        let candidates = self.view.tensor_index.entries_for(layer_idx, kind.as_u32());
        !candidates.is_empty()
    }

    fn load_tensor(
        &self,
        id: &TensorId,
        is_weight: bool,
        backend: &Arc<dyn Backend>,
        _memory: &dyn Memory,
    ) -> Result<Tensor> {
        let cpu_tensor = self.materialise_cpu_tensor(id)?;
        let is_cpu = backend.name().contains("CPU");
        if is_cpu {
            Ok(cpu_tensor)
        } else if is_weight {
            backend.copy_weight_from(&cpu_tensor)
        } else {
            backend.copy_from(&cpu_tensor)
        }
    }

    fn load_tensor_cpu(
        &self,
        id: &TensorId,
        _is_weight: bool,
        _memory: &dyn Memory,
    ) -> Result<Tensor> {
        self.materialise_cpu_tensor(id)
    }

    fn as_auf(&self) -> Option<&AufSource> {
        Some(self)
    }
}

impl AufSource {
    /// `TensorId` → CPU `Tensor` (zero-copy via [`AufViewBuffer`]).
    ///
    /// AUF는 빌드 시점에 dtype/variant/unpermute가 모두 확정되므로 GGUF처럼
    /// 런타임 변환이 필요하지 않다. lookup → variant slice → AufViewBuffer 한 줄.
    fn materialise_cpu_tensor(&self, id: &TensorId) -> Result<Tensor> {
        let (layer_idx, kind) = tensor_id_to_auf(id).ok_or_else(|| {
            anyhow!(
                "AUF: TensorId {:?} unsupported (e.g. Gemma3 PreFfnNorm/QNorm/KNorm)",
                id
            )
        })?;

        // dtype 우선순위:
        // 1. Weight matmul (Wq..WDown) — primary_dtype을 lookup에 강제 (multi-dtype 분기 핵심).
        // 2. Norm / cross-layer (Embed/FinalNorm/LmHead 등) — primary_dtype을 무시하고
        //    AUF의 META.default_dtype + first-match (lookup_tensor 내부 fallback).
        //    이유: norm 텐서는 보통 F32 단일이라 multi-dtype 강제 시 lookup 실패.
        let lookup_dtype = if is_dtype_strict_kind(id) {
            self.primary_dtype
        } else {
            None
        };
        let entry = self
            .view
            .lookup_tensor(layer_idx, kind.as_u32(), lookup_dtype)
            .map_err(|e| {
                anyhow!(
                    "AUF lookup failed (layer={}, kind={:?}, lookup_dtype={:?}): {}",
                    layer_idx,
                    kind,
                    lookup_dtype,
                    e
                )
            })?;

        let weights_tag = self.variant_tag.weights_section_tag().ok_or_else(|| {
            anyhow!("AUF primary: BackendTag::Any cannot host tensors (variant_tag invariant)")
        })?;
        let var_idx = self
            .view
            .tensor_index
            .variant_index_for_tag(weights_tag)
            .ok_or_else(|| {
                anyhow!(
                    "AUF '{}' lacks WEIGHTS variant '{}'",
                    self.source_path.display(),
                    weights_tag,
                )
            })?;

        let var_offset = entry.variant_offsets.get(var_idx).copied().ok_or_else(|| {
            anyhow!(
                "AUF: variant_offsets[{}] missing for (layer={}, kind={:?})",
                var_idx,
                layer_idx,
                kind,
            )
        })?;
        let var_size = entry.variant_sizes.get(var_idx).copied().ok_or_else(|| {
            anyhow!(
                "AUF: variant_sizes[{}] missing for (layer={}, kind={:?})",
                var_idx,
                layer_idx,
                kind,
            )
        })?;
        if var_offset == u64::MAX || var_size == 0 {
            return Err(anyhow!(
                "AUF: variant '{}' has no payload for (layer={}, kind={:?})",
                weights_tag,
                layer_idx,
                kind,
            ));
        }

        // Shape: AUF stores outermost-first → 그대로 사용 (GGUF의 `.rev()` 미적용).
        let shape = Shape::new(entry.shape.iter().map(|&d| d as usize).collect());

        let dtype = auf_dtype_to_engine(entry.dtype).ok_or_else(|| {
            anyhow!(
                "AUF: unsupported dtype code {} for (layer={}, kind={:?})",
                entry.dtype,
                layer_idx,
                kind,
            )
        })?;

        // abs_offset = WEIGHTS section file offset + section-local var_offset.
        let (weights_section_offset, _) = self
            .view
            .weights_range
            .ok_or_else(|| anyhow!("AUF: weights_range None (BackendTag::Any opened?)"))?;
        let abs_offset = weights_section_offset as usize + var_offset as usize;

        // Safety: abs_offset + var_size ≤ raw_bytes().len() (TensorEntry invariant
        // enforced by AufView::open).
        let buffer: Arc<dyn Buffer> = Arc::new(unsafe {
            AufViewBuffer::new(Arc::clone(&self.view), abs_offset, var_size as usize, dtype)
        });
        Ok(Tensor::new(
            shape,
            buffer,
            self.cpu_backend.clone() as Arc<dyn Backend>,
        ))
    }
}

/// AUF META `default_dtype` 문자열 → `TensorDType` (graceful Option).
fn parse_meta_dtype_str(s: &str) -> Option<TensorDType> {
    match s {
        "F32" => Some(TensorDType::F32),
        "F16" => Some(TensorDType::F16),
        "BF16" => Some(TensorDType::BF16),
        "Q4_0" => Some(TensorDType::Q4_0),
        "Q4_1" => Some(TensorDType::Q4_1),
        "Q8_0" => Some(TensorDType::Q8_0),
        "U8" => Some(TensorDType::U8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_id_layer_weight_round_trip() {
        let id = TensorId::LayerWeight {
            layer: 7,
            kind: LayerWeightKind::Wq,
        };
        let (layer_idx, kind) = tensor_id_to_auf(&id).unwrap();
        assert_eq!(layer_idx, 7);
        assert_eq!(kind, TensorKind::AttnQ);
    }

    #[test]
    fn tensor_id_cross_layer() {
        assert_eq!(
            tensor_id_to_auf(&TensorId::Embed).unwrap(),
            (LAYER_IDX_CROSS, TensorKind::Embedding)
        );
        assert_eq!(
            tensor_id_to_auf(&TensorId::FinalNorm).unwrap(),
            (LAYER_IDX_CROSS, TensorKind::FinalNorm)
        );
        assert_eq!(
            tensor_id_to_auf(&TensorId::LmHead).unwrap(),
            (LAYER_IDX_CROSS, TensorKind::LmHead)
        );
    }

    #[test]
    fn tensor_id_bias_maps_to_qkv_bias() {
        assert_eq!(
            tensor_id_to_auf(&TensorId::LayerBias {
                layer: 0,
                kind: crate::models::loader::LayerBiasKind::Bq,
            })
            .unwrap(),
            (0, TensorKind::AttnQBias)
        );
        assert_eq!(
            tensor_id_to_auf(&TensorId::LayerBias {
                layer: 5,
                kind: crate::models::loader::LayerBiasKind::Bk,
            })
            .unwrap(),
            (5, TensorKind::AttnKBias)
        );
        assert_eq!(
            tensor_id_to_auf(&TensorId::LayerBias {
                layer: 10,
                kind: crate::models::loader::LayerBiasKind::Bv,
            })
            .unwrap(),
            (10, TensorKind::AttnVBias)
        );
    }

    #[test]
    fn parse_arch_handles_known_values() {
        assert_eq!(parse_arch_str("llama").unwrap(), ModelArch::Llama);
        assert_eq!(parse_arch_str("qwen2").unwrap(), ModelArch::Qwen2);
        assert_eq!(parse_arch_str("Qwen2.5").unwrap(), ModelArch::Qwen2);
        assert!(parse_arch_str("unknown_arch").is_err());
    }

    /// `materialise_cpu_tensor` zero-copy round-trip — builds a 1-layer AUF in
    /// memory, opens via `AufSource`, then loads a tensor and verifies bytes
    /// match the original payload.
    #[test]
    fn load_tensor_cpu_zero_copy_round_trip() {
        use crate::auf::reader::open_from_bytes;
        use crate::auf::section::TAG_WEIGHTS_CPU_AOS;
        use crate::auf::tensor_index::{TensorEntry, TensorIndex};
        use crate::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
        use crate::auf::writer::AufWriter;
        use crate::auf::{AufMeta, BackendTag};

        const TENSOR_BYTES: usize = 256;
        // Deterministic byte pattern.
        let payload: Vec<u8> = (0..TENSOR_BYTES).map(|i| (i & 0xFF) as u8).collect();
        let payload_clone = payload.clone();

        let mut tag_buf = [0u8; 24];
        tag_buf[..TAG_WEIGHTS_CPU_AOS.len()].copy_from_slice(TAG_WEIGHTS_CPU_AOS.as_bytes());

        let entries = vec![TensorEntry {
            layer_idx: 0,
            kind: TensorKind::AttnQ.as_u32(),
            dtype: TensorDType::F32.as_u32(),
            shape: vec![1, 64],
            alignment: 64,
            variant_offsets: vec![0],
            variant_sizes: vec![TENSOR_BYTES as u64],
        }];
        let tensor_index = TensorIndex {
            variant_tags: vec![tag_buf],
            entries,
        };

        let meta = AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 1,
            n_heads_q: 1,
            n_kv_heads: 1,
            head_dim: 64,
            hidden_dim: 64,
            ffn_dim: 128,
            vocab_size: 2,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rotary_dim: 64,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        };
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec()],
            merges: vec![],
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            unk_id: 0,
            chat_template: None,
        };

        let auf_bytes = AufWriter::new(meta, tok, [0u8; 32], 0, 0)
            .with_tensor_index(tensor_index)
            .add_weights_section(TAG_WEIGHTS_CPU_AOS, payload_clone)
            .build()
            .unwrap();

        let view = Arc::new(open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap());
        let config = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 64,
            intermediate_size: 128,
            vocab_size: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 2,
            weight_prefix: String::new(),
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
        };

        // 직접 AufSource 구성 (open()은 file path 기반이므로 in-memory bytes 시나리오용 우회).
        let src = AufSource {
            view,
            variant_tag: BackendTag::CpuAos,
            primary_dtype: None,
            config,
            cpu_backend: Arc::new(CpuBackend::new()),
            source_path: std::path::PathBuf::from("/fake/test.auf"),
        };

        // has_tensor 동작 검증.
        let id = TensorId::LayerWeight {
            layer: 0,
            kind: LayerWeightKind::Wq,
        };
        assert!(src.has_tensor(&id));

        // load_tensor_cpu가 zero-copy로 원본 payload를 그대로 노출하는지 검증.
        let mem = crate::memory::galloc::Galloc::new();
        let tensor = src.load_tensor_cpu(&id, true, &mem).unwrap();
        let buf = tensor.buffer();
        assert_eq!(buf.size(), TENSOR_BYTES);
        assert_eq!(buf.dtype(), DType::F32);
        let loaded = unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.size()) };
        assert_eq!(loaded, &payload[..], "loaded bytes must equal payload");
    }

    /// AUF에 bias entry가 없는 경우 (예: Llama, 또는 bias 미포함 Qwen2 빌드)
    /// `has_tensor=false` + `load_tensor_cpu=Err` 가 일관되게 동작해야 한다.
    #[test]
    fn load_tensor_layer_bias_missing_entry_errors_out() {
        use crate::auf::reader::open_from_bytes;
        use crate::auf::section::TAG_WEIGHTS_CPU_AOS;
        use crate::auf::tensor_index::{TensorEntry, TensorIndex};
        use crate::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
        use crate::auf::writer::AufWriter;
        use crate::auf::{AufMeta, BackendTag};

        let mut tag_buf = [0u8; 24];
        tag_buf[..TAG_WEIGHTS_CPU_AOS.len()].copy_from_slice(TAG_WEIGHTS_CPU_AOS.as_bytes());

        let auf_bytes = AufWriter::new(
            AufMeta {
                architecture: "llama".to_owned(),
                n_layers: 1,
                n_heads_q: 1,
                n_kv_heads: 1,
                head_dim: 64,
                hidden_dim: 64,
                ffn_dim: 128,
                vocab_size: 2,
                max_seq_len: 32,
                rope_theta: 10000.0,
                rotary_dim: 64,
                rope_scaling: 1.0,
                rms_norm_epsilon: 1e-5,
                default_dtype: None,
            },
            AufTokenizer {
                kind: TOKENIZER_KIND_BPE,
                tokens: vec![b"a".to_vec()],
                merges: vec![],
                bos_id: 1,
                eos_id: 2,
                pad_id: -1,
                unk_id: 0,
                chat_template: None,
            },
            [0u8; 32],
            0,
            0,
        )
        .with_tensor_index(TensorIndex {
            variant_tags: vec![tag_buf],
            entries: vec![TensorEntry {
                layer_idx: 0,
                kind: TensorKind::AttnQ.as_u32(),
                dtype: TensorDType::F32.as_u32(),
                shape: vec![1, 64],
                alignment: 64,
                variant_offsets: vec![0],
                variant_sizes: vec![256],
            }],
        })
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![0u8; 256])
        .build()
        .unwrap();

        let view = Arc::new(open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap());
        let src = AufSource {
            view,
            variant_tag: BackendTag::CpuAos,
            primary_dtype: None,
            config: ModelConfig {
                arch: ModelArch::Qwen2,
                hidden_size: 64,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 64,
                intermediate_size: 128,
                vocab_size: 2,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                has_qkv_bias: true,
                tie_word_embeddings: false,
                eos_token_id: 2,
                weight_prefix: String::new(),
                rope_local_theta: None,
                sliding_window: None,
                sliding_window_pattern: None,
                query_pre_attn_scalar: None,
                embed_scale: None,
            },
            cpu_backend: Arc::new(CpuBackend::new()),
            source_path: std::path::PathBuf::from("/fake/test.auf"),
        };

        // 위 fixture는 AttnQ entry만 등록 — bias entry는 없음.
        let id = TensorId::LayerBias {
            layer: 0,
            kind: crate::models::loader::LayerBiasKind::Bq,
        };
        assert!(
            !src.has_tensor(&id),
            "bias entry가 없으므로 has_tensor=false 여야 한다"
        );
        let mem = crate::memory::galloc::Galloc::new();
        assert!(
            src.load_tensor_cpu(&id, false, &mem).is_err(),
            "bias entry가 없을 때 load_tensor_cpu는 Err를 반환해야 한다"
        );
    }
}
