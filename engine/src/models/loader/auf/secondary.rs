//! AUF secondary path — moved out of `weights/secondary_mmap.rs` in W-AUF-1 C2.
//!
//! Owns the AUF-specific helpers used by both the primary [`AufSource`] (zero-copy
//! `load_tensor`) and the secondary `open_secondary_*` family. The `SecondaryMmap`
//! enum + `AufSecondaryMmap` struct still live in `weights/secondary_mmap.rs`
//! because they participate in the GGUF/AUF/Rpcmem unified dispatcher.
//!
//! Dependency direction is strictly one-way: `loader/auf/secondary.rs` →
//! `weights/secondary_mmap.rs` (imports `AufSecondaryMmap`, `LayerTensorSlice`,
//! `SecondaryTensorInfo`, `SecondaryMmap`, `SecondaryDtypeChoice`,
//! `SecondaryLayoutChoice`, `LoadError`). The reverse import is permitted
//! through `pub use` re-exports for backward compatibility.
//!
//! [`AufSource`]: super::AufSource

use std::path::Path;

use crate::auf::{
    AufError, AufMeta, AufView, BackendTag,
    section::TAG_WEIGHTS_ADRENO_SOA,
    tensor_index::{LAYER_IDX_CROSS, TensorDType},
};
use crate::buffer::DType;
use crate::model_config::ModelConfig;
use crate::models::weights::secondary_mmap::{
    AufSecondaryMmap, LayerTensorSlice, LoadError, SecondaryDtypeChoice, SecondaryLayoutChoice,
    SecondaryMmap, SecondaryTensorInfo, tensor_kind_to_subname,
};

use super::variant_select::AufVariantChoice;

/// Returns `true` if `path` should be opened as AUF.
///
/// Detection order:
/// 1. Extension `.auf` (case-sensitive).
/// 2. Magic bytes `b"ARGUS_W\0"` in the first 8 bytes of the file
///    (see `crate::auf::header::AUF_MAGIC`).
pub fn is_auf_path(path: &Path) -> bool {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "auf")
        .unwrap_or(false)
    {
        return true;
    }
    if let Ok(mut f) = std::fs::File::open(path) {
        use std::io::Read;
        let mut magic = [0u8; 8];
        if f.read_exact(&mut magic).is_ok() && &magic == crate::auf::header::AUF_MAGIC {
            return true;
        }
    }
    false
}

/// Map `TensorDType` to engine `DType`. Returns `None` for unrecognised codes.
pub fn auf_dtype_to_engine(dtype: u32) -> Option<DType> {
    match TensorDType::from_u32(dtype)? {
        TensorDType::F32 => Some(DType::F32),
        TensorDType::F16 => Some(DType::F16),
        TensorDType::BF16 => Some(DType::BF16),
        TensorDType::Q4_0 => Some(DType::Q4_0),
        TensorDType::Q4_1 => Some(DType::Q4_1),
        TensorDType::Q8_0 => Some(DType::Q8_0),
        TensorDType::U8 => Some(DType::U8),
    }
}

/// Resolve the candidate `BackendTag` list to try in order, given the
/// `SecondaryLayoutChoice`. For `Auto`, falls back to `CpuAos` if the
/// preferred variant is missing from the AUF file.
pub(crate) fn resolve_backend_tag_candidates(layout: SecondaryLayoutChoice) -> Vec<BackendTag> {
    match layout {
        SecondaryLayoutChoice::Auto => {
            let preferred = AufVariantChoice::default_tag();
            if preferred == BackendTag::CpuAos {
                vec![BackendTag::CpuAos]
            } else {
                vec![preferred, BackendTag::CpuAos]
            }
        }
        SecondaryLayoutChoice::Aos => {
            #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
            return vec![BackendTag::CudaAos, BackendTag::CpuAos];
            #[cfg(not(any(feature = "cuda", feature = "cuda-embedded")))]
            vec![BackendTag::CpuAos]
        }
        SecondaryLayoutChoice::Soa => vec![BackendTag::AdrenoSoa],
    }
}

/// Build an F16-only `SecondaryMmap` view from an existing `Arc<AufView>` for
/// recall (Q4_0→F16 restoration, ENG-ALG-240).
///
/// Unlike `open_secondary_auf` / `build_auf_secondary_from_view`, this path:
/// - Forces dtype = F16 unconditionally.
/// - **Skips `ReverseSwapRejected`** — that guard protects the *forward* swap
///   path (production swap: F16→Q4_0 only).  Recall is the intentional reverse
///   direction and must not hit that guard.
/// - **Propagates `AdrenoSoaF16Rejected`** (SOA layout is Q4_0-only, loud no-op b)
///   and **`DtypeNotFound`** (no F16 entry in AUF, loud no-op a).
///
/// Same `Arc<AufView>` / mmap is reused (INV-125 — no second mmap open).
/// The caller is responsible for caching the result (`OnceLock`) to avoid
/// repeated AUF scan on every recall.
///
/// # Errors
/// - `LoadError::AdrenoSoaF16Rejected` — SOA layout (loud no-op b).
/// - `LoadError::DtypeNotFound { dtype: "F16" }` — AUF has no F16 entry (loud no-op a).
/// - `LoadError::AufInvariantViolation` — TENSOR_INDEX inconsistency.
pub fn open_secondary_f16_for_recall(
    view: std::sync::Arc<crate::auf::AufView>,
    backend_tag: crate::auf::BackendTag,
    primary_config: &ModelConfig,
) -> Result<SecondaryMmap, LoadError> {
    use crate::auf::section::TAG_WEIGHTS_ADRENO_SOA;
    use crate::auf::tensor_index::{LAYER_IDX_CROSS, TensorDType};
    use crate::models::weights::secondary_mmap::{AufSecondaryMmap, tensor_kind_to_subname};

    // (b) SOA 경로는 Q4_0 전용 — F16 recall 원천 차단.
    let weights_tag =
        backend_tag
            .weights_section_tag()
            .ok_or_else(|| LoadError::AufInvariantViolation {
                detail: "recall: BackendTag::Any cannot host tensors".to_string(),
            })?;
    if weights_tag == TAG_WEIGHTS_ADRENO_SOA {
        return Err(LoadError::AdrenoSoaF16Rejected);
    }

    // TENSOR_INDEX 내 variant slot 인덱스 조회.
    let variant_idx = view
        .tensor_index
        .variant_index_for_tag(weights_tag)
        .ok_or_else(|| LoadError::AufInvariantViolation {
            detail: format!("recall F16 view: TENSOR_INDEX missing variant '{weights_tag}'"),
        })?;

    // AUF에서 F16 entry가 있는지 확인 (없으면 DtypeNotFound — loud no-op a).
    let has_f16 = view.tensor_index.entries.iter().any(|e| {
        if e.layer_idx == LAYER_IDX_CROSS {
            return false;
        }
        if e.dtype != TensorDType::F16.as_u32() {
            return false;
        }
        let var_offset = e
            .variant_offsets
            .get(variant_idx)
            .copied()
            .unwrap_or(u64::MAX);
        let var_size = e.variant_sizes.get(variant_idx).copied().unwrap_or(0);
        var_offset != u64::MAX && var_size != 0
    });
    if !has_f16 {
        return Err(LoadError::DtypeNotFound {
            dtype: "F16".to_string(),
        });
    }

    // layer_index 구성 — F16 entry만 선택 (ReverseSwapRejected 가드 없음).
    let num_layers = primary_config.num_hidden_layers;
    let mut layer_index: Vec<LayerTensorSlice> = vec![LayerTensorSlice::default(); num_layers];

    for entry in &view.tensor_index.entries {
        if entry.layer_idx == LAYER_IDX_CROSS {
            continue;
        }
        let layer_idx = entry.layer_idx as usize;
        if layer_idx >= num_layers {
            continue;
        }
        if entry.dtype != TensorDType::F16.as_u32() {
            continue;
        }
        let var_offset = entry
            .variant_offsets
            .get(variant_idx)
            .copied()
            .unwrap_or(u64::MAX);
        let var_size = entry.variant_sizes.get(variant_idx).copied().unwrap_or(0);
        if var_offset == u64::MAX || var_size == 0 {
            continue;
        }
        let Some(subname) = tensor_kind_to_subname(entry.kind) else {
            continue;
        };
        let Some(dtype) = auf_dtype_to_engine(entry.dtype) else {
            continue;
        };
        let slice_info = SecondaryTensorInfo {
            offset: var_offset as usize,
            len: var_size as usize,
            dtype,
            dims: entry.shape.iter().rev().copied().collect(),
        };
        layer_index[layer_idx]
            .tensors
            .insert(subname.to_string(), slice_info);
    }

    // is_pre_converted_soa = false (SOA 경로는 위에서 차단됨).
    let source_path = std::path::PathBuf::from("<recall-f16-view>".to_string());
    Ok(SecondaryMmap::Auf(AufSecondaryMmap {
        view,
        layer_index,
        source_path,
        is_pre_converted_soa: false,
    }))
}

/// Open an AUF-format secondary weight file directly (skips GGUF detection).
///
/// Public so integration tests can exercise the layout-fallback path without
/// constructing a `GgufFile` placeholder. Production callers should use
/// `open_secondary_with_options`, which dispatches GGUF vs AUF on extension.
pub fn open_secondary_auf(
    path: &Path,
    primary_config: &ModelConfig,
    secondary_dtype_choice: SecondaryDtypeChoice,
    secondary_layout_choice: SecondaryLayoutChoice,
) -> Result<SecondaryMmap, LoadError> {
    let candidates = resolve_backend_tag_candidates(secondary_layout_choice);
    debug_assert!(
        !candidates.is_empty(),
        "resolve_backend_tag_candidates must return at least one tag"
    );

    let mut last_missing: Option<AufError> = None;
    let backend_tag = {
        let mut chosen: Option<BackendTag> = None;
        for tag in &candidates {
            match crate::auf::reader::open(path, *tag) {
                Ok(_view) => {
                    chosen = Some(*tag);
                    break;
                }
                Err(e @ AufError::WeightsSectionMissing { .. }) => {
                    last_missing = Some(e);
                    continue;
                }
                Err(e) => {
                    return Err(LoadError::AufInvariantViolation {
                        detail: format!("open AUF {}: {e}", path.display()),
                    });
                }
            }
        }
        match chosen {
            Some(t) => t,
            None => {
                let detail = match last_missing {
                    Some(e) => format!(
                        "AUF {} has none of the candidate variants {:?}: {e}",
                        path.display(),
                        candidates,
                    ),
                    None => format!("AUF {} variant probe yielded no result", path.display()),
                };
                return Err(LoadError::AufInvariantViolation { detail });
            }
        }
    };

    let default_tag = AufVariantChoice::default_tag();
    if backend_tag != default_tag && matches!(secondary_layout_choice, SecondaryLayoutChoice::Auto)
    {
        eprintln!(
            "[Secondary] AUF lacks preferred variant {:?}; using {:?} (swap will use AOS path — \
             switch_hw / partition compatible, GPU TBT may regress vs SOA)",
            default_tag, backend_tag,
        );
    }

    let view = crate::auf::reader::open(path, backend_tag).map_err(|e| {
        LoadError::AufInvariantViolation {
            detail: format!("{}  (file: {})", e, path.display()),
        }
    })?;

    check_auf_metadata(primary_config, &view.meta, path)?;

    build_auf_secondary_from_view(
        view,
        primary_config,
        path,
        backend_tag,
        secondary_dtype_choice,
    )
}

/// Core logic for building an `AufSecondaryMmap` from a pre-parsed `AufView`.
///
/// Extracted for testability: callers (e.g. unit tests) can supply an
/// `AufView` constructed directly from bytes via `open_from_bytes`, bypassing
/// file I/O, while still exercising the full tensor-index → layer-index →
/// tensor_bytes round-trip.
///
/// # Dtype 선택 정책 (ENG-ALG-225, Sprint D)
///
/// `secondary_dtype_choice`는 layer-index 구성 시 어느 dtype entry를 사용할지 결정한다:
///
/// 1. `SecondaryDtypeChoice::Auto` — primary와 다른 dtype candidate를 자동 선택.
///    후보가 여럿이면 META.default_dtype을 우선 사용하고, 그래도 모호하면 첫 번째 candidate.
/// 2. 명시 dtype (`F16`, `Q4_0`, `F32`) — 해당 dtype의 entry만 사용.
///
/// # Adreno SOA × F16 reject (Sprint D 함정 3)
///
/// `backend_tag = AdrenoSoa`인데 선택된 dtype이 F16이면 `LoadError::AdrenoSoaF16Rejected`를
/// 반환한다. SOA layout은 Q4_0 전용이므로 F16 secondary는 지원되지 않는다.
///
/// # 단방향 swap 정합성 (Sprint D)
///
/// primary dtype이 Q4_0인데 secondary dtype이 F16이면 `LoadError::ReverseSwapRejected`를
/// 반환한다. weight swap은 F16→Q4_0 단방향만 지원한다.
///
/// # Offset contract
/// `TensorIndex::variant_offsets` entries are **section-local** (relative to the
/// start of the WEIGHTS payload). `AufView::weights_bytes()` already returns a
/// slice that starts at the WEIGHTS section offset, so we store `var_offset`
/// directly into `SecondaryTensorInfo::offset` — we must **not** add
/// `weights_section_offset` again (that would cause a double-base OOB panic).
pub fn build_auf_secondary_from_view(
    view: AufView,
    primary_config: &ModelConfig,
    path: &Path,
    backend_tag: BackendTag,
    secondary_dtype_choice: SecondaryDtypeChoice,
) -> Result<SecondaryMmap, LoadError> {
    let weights_tag = backend_tag
        .weights_section_tag()
        .expect("detect_backend_tag returns a concrete tag");
    let variant_idx = view
        .tensor_index
        .variant_index_for_tag(weights_tag)
        .ok_or_else(|| LoadError::AufInvariantViolation {
            detail: format!(
                "TENSOR_INDEX does not list variant '{}' (file: {})",
                weights_tag,
                path.display()
            ),
        })?;

    let (_weights_section_offset, _weights_section_size) = view
        .weights_range
        .expect("AufView::weights_range must be Some after open() with concrete backend_tag");

    // ── dtype 선택 정책 (ENG-ALG-225) ─────────────────────────────────────
    let available_dtypes: std::collections::BTreeSet<u32> = view
        .tensor_index
        .entries
        .iter()
        .filter(|e| {
            if e.layer_idx == LAYER_IDX_CROSS {
                return false;
            }
            let var_offset = e
                .variant_offsets
                .get(variant_idx)
                .copied()
                .unwrap_or(u64::MAX);
            let var_size = e.variant_sizes.get(variant_idx).copied().unwrap_or(0);
            var_offset != u64::MAX && var_size != 0
        })
        .map(|e| e.dtype)
        .collect();

    let selected_dtype: Option<TensorDType> = match secondary_dtype_choice {
        SecondaryDtypeChoice::Auto => {
            let default_from_meta: Option<TensorDType> = view
                .meta
                .default_dtype
                .as_deref()
                .and_then(dtype_str_to_tensor_dtype_local);

            // weight-dtype 우선순위: Q4_0 → F16 → BF16 → Q4_1 → Q8_0 → F32.
            // F32는 일반적으로 norm 전용 dtype이므로 Auto 선택에서는 마지막으로
            // 떨어뜨려야 한다 (BTreeSet 자연 정렬을 그대로 쓰면 F32가 잡혀 weight가
            // 전부 필터링됨).
            let weight_preference = [
                TensorDType::Q4_0,
                TensorDType::F16,
                TensorDType::BF16,
                TensorDType::Q4_1,
                TensorDType::Q8_0,
                TensorDType::F32,
            ];
            let pick_by_preference = || -> Option<TensorDType> {
                weight_preference
                    .iter()
                    .copied()
                    .find(|d| available_dtypes.contains(&d.as_u32()))
            };

            if let Some(d) = default_from_meta {
                if available_dtypes.contains(&d.as_u32()) {
                    Some(d)
                } else {
                    pick_by_preference()
                }
            } else {
                pick_by_preference()
            }
        }
        SecondaryDtypeChoice::F16 => {
            if available_dtypes.contains(&TensorDType::F16.as_u32()) {
                Some(TensorDType::F16)
            } else {
                return Err(LoadError::DtypeNotFound {
                    dtype: "F16".to_string(),
                });
            }
        }
        SecondaryDtypeChoice::Q4_0 => {
            if available_dtypes.contains(&TensorDType::Q4_0.as_u32()) {
                Some(TensorDType::Q4_0)
            } else {
                return Err(LoadError::DtypeNotFound {
                    dtype: "Q4_0".to_string(),
                });
            }
        }
        SecondaryDtypeChoice::F32 => {
            if available_dtypes.contains(&TensorDType::F32.as_u32()) {
                Some(TensorDType::F32)
            } else {
                return Err(LoadError::DtypeNotFound {
                    dtype: "F32".to_string(),
                });
            }
        }
    };

    if backend_tag == BackendTag::AdrenoSoa && matches!(selected_dtype, Some(TensorDType::F16)) {
        return Err(LoadError::AdrenoSoaF16Rejected);
    }

    if let Some(TensorDType::F16) = selected_dtype {
        let only_q4_0 =
            available_dtypes.len() == 1 && available_dtypes.contains(&TensorDType::Q4_0.as_u32());
        if only_q4_0 {
            return Err(LoadError::ReverseSwapRejected {
                primary_dtype: "Q4_0".to_string(),
                secondary_dtype: "F16".to_string(),
            });
        }
    }

    let num_layers = primary_config.num_hidden_layers;
    let mut layer_index: Vec<LayerTensorSlice> = vec![LayerTensorSlice::default(); num_layers];

    for entry in &view.tensor_index.entries {
        if entry.layer_idx == LAYER_IDX_CROSS {
            continue;
        }
        let layer_idx = entry.layer_idx as usize;
        if layer_idx >= num_layers {
            continue;
        }

        if let Some(sel) = selected_dtype
            && entry.dtype != sel.as_u32()
        {
            continue;
        }

        let var_offset = entry
            .variant_offsets
            .get(variant_idx)
            .copied()
            .unwrap_or(u64::MAX);
        let var_size = entry.variant_sizes.get(variant_idx).copied().unwrap_or(0);
        if var_offset == u64::MAX || var_size == 0 {
            continue;
        }

        let Some(subname) = tensor_kind_to_subname(entry.kind) else {
            continue;
        };
        let Some(dtype) = auf_dtype_to_engine(entry.dtype) else {
            continue;
        };

        // var_offset is section-local (relative to WEIGHTS payload start).
        // weights_bytes() already returns a slice starting at weights_section_offset,
        // so we store the section-local offset directly — do NOT add
        // weights_section_offset again.
        let slice_info = SecondaryTensorInfo {
            offset: var_offset as usize,
            len: var_size as usize,
            dtype,
            // AUF shape is stored in logical order (outermost first).
            // `swap_executor.rs` expects GGUF order (innermost first), so reverse.
            dims: entry.shape.iter().rev().copied().collect(),
        };

        layer_index[layer_idx]
            .tensors
            .insert(subname.to_string(), slice_info);
    }

    let is_pre_converted_soa = weights_tag == TAG_WEIGHTS_ADRENO_SOA;

    Ok(SecondaryMmap::Auf(AufSecondaryMmap {
        view: std::sync::Arc::new(view),
        layer_index,
        source_path: path.to_path_buf(),
        is_pre_converted_soa,
    }))
}

/// Build a `SecondaryMmap::Auf` that **shares the primary `AufView` mmap**.
///
/// W-AUF-2.1 — `--model-path foo.auf`로 primary가 AUF일 때, 같은 mmap을
/// secondary로 재포장하여 swap 후보를 자기 자신에서 추출한다. 두 번째 mmap을
/// 만들지 않으므로 R6 (이중 mmap) 위험이 해결된다.
///
/// 동작:
/// 1. **primary 자기 제외**: `primary_variant_tag` + `primary_dtype` 조합은
///    swap 후보에서 명시 제외. AUF가 multi-dtype이면 자동으로 다른 dtype 후보를
///    선택한다.
/// 2. 나머지는 `build_auf_secondary_from_view`와 동일 — layer_index 구성, AdrenoSoa F16
///    reject, ReverseSwapRejected 등.
///
/// # 에러
/// - `LoadError::AufInvariantViolation` — primary 외 swap 후보가 없을 때 (즉 multi-dtype 아님)
/// - 기타 `build_auf_secondary_from_view`와 동일 게이트
pub fn from_auf_self_secondary(
    view: std::sync::Arc<AufView>,
    primary_variant_tag: BackendTag,
    primary_dtype: TensorDType,
    primary_config: &ModelConfig,
) -> Result<std::sync::Arc<SecondaryMmap>, LoadError> {
    // primary와 같은 variant_tag에 대해 layer_index를 구성하되, dtype은 primary가
    // 아닌 후보로 선택한다. variant slot은 primary와 같지만 entry.dtype 필터만
    // 다르게 적용.
    let weights_tag = primary_variant_tag
        .weights_section_tag()
        .ok_or_else(|| LoadError::AufInvariantViolation {
            detail: "AUF self-secondary: BackendTag::Any cannot host tensors (primary_variant_tag invariant)".to_string(),
        })?;

    let variant_idx = view
        .tensor_index
        .variant_index_for_tag(weights_tag)
        .ok_or_else(|| LoadError::AufInvariantViolation {
            detail: format!(
                "AUF self-secondary: TENSOR_INDEX missing primary variant '{weights_tag}'"
            ),
        })?;

    let _ = view
        .weights_range
        .ok_or_else(|| LoadError::AufInvariantViolation {
            detail: "AUF self-secondary: weights_range None (BackendTag::Any opened?)".to_string(),
        })?;

    // primary 외 dtype 후보 수집.
    let primary_dtype_u32 = primary_dtype.as_u32();
    let available_dtypes: std::collections::BTreeSet<u32> = view
        .tensor_index
        .entries
        .iter()
        .filter(|e| {
            if e.layer_idx == LAYER_IDX_CROSS {
                return false;
            }
            if e.dtype == primary_dtype_u32 {
                return false; // self-exclude
            }
            let var_offset = e
                .variant_offsets
                .get(variant_idx)
                .copied()
                .unwrap_or(u64::MAX);
            let var_size = e.variant_sizes.get(variant_idx).copied().unwrap_or(0);
            var_offset != u64::MAX && var_size != 0
        })
        .map(|e| e.dtype)
        .collect();

    if available_dtypes.is_empty() {
        return Err(LoadError::AufInvariantViolation {
            detail: format!(
                "AUF self-secondary: no swap candidates (primary dtype={primary_dtype:?} \
                 is the only dtype present for variant '{weights_tag}'). \
                 Build multi-dtype AUF with `auf_tool build --dtypes` to enable swap."
            ),
        });
    }

    // 후보 dtype 선택 — weight 우선순위 (Q4_0 → F16 → BF16 → Q4_1 → Q8_0 → F32).
    let weight_preference = [
        TensorDType::Q4_0,
        TensorDType::F16,
        TensorDType::BF16,
        TensorDType::Q4_1,
        TensorDType::Q8_0,
        TensorDType::F32,
    ];
    let selected_dtype = weight_preference
        .iter()
        .copied()
        .find(|d| available_dtypes.contains(&d.as_u32()))
        .ok_or_else(|| LoadError::AufInvariantViolation {
            detail: format!(
                "AUF self-secondary: no recognised dtype among candidates {:?}",
                available_dtypes
            ),
        })?;

    // AdrenoSoa × F16 reject (Sprint D 함정 3 — SOA layout은 Q4_0 전용).
    if primary_variant_tag == BackendTag::AdrenoSoa && selected_dtype == TensorDType::F16 {
        return Err(LoadError::AdrenoSoaF16Rejected);
    }

    // 단방향 swap 정합성: primary=Q4_0 + secondary=F16 reject.
    if primary_dtype == TensorDType::Q4_0 && selected_dtype == TensorDType::F16 {
        return Err(LoadError::ReverseSwapRejected {
            primary_dtype: "Q4_0".to_string(),
            secondary_dtype: "F16".to_string(),
        });
    }

    let num_layers = primary_config.num_hidden_layers;
    let mut layer_index: Vec<LayerTensorSlice> = vec![LayerTensorSlice::default(); num_layers];

    for entry in &view.tensor_index.entries {
        if entry.layer_idx == LAYER_IDX_CROSS {
            continue;
        }
        let layer_idx = entry.layer_idx as usize;
        if layer_idx >= num_layers {
            continue;
        }
        if entry.dtype != selected_dtype.as_u32() {
            continue;
        }
        let var_offset = entry
            .variant_offsets
            .get(variant_idx)
            .copied()
            .unwrap_or(u64::MAX);
        let var_size = entry.variant_sizes.get(variant_idx).copied().unwrap_or(0);
        if var_offset == u64::MAX || var_size == 0 {
            continue;
        }
        let Some(subname) = tensor_kind_to_subname(entry.kind) else {
            continue;
        };
        let Some(dtype) = auf_dtype_to_engine(entry.dtype) else {
            continue;
        };
        let slice_info = SecondaryTensorInfo {
            offset: var_offset as usize,
            len: var_size as usize,
            dtype,
            // AUF outermost-first → swap_executor expects innermost-first.
            dims: entry.shape.iter().rev().copied().collect(),
        };
        layer_index[layer_idx]
            .tensors
            .insert(subname.to_string(), slice_info);
    }

    let is_pre_converted_soa = weights_tag == TAG_WEIGHTS_ADRENO_SOA;
    let source_path = std::path::PathBuf::from(format!(
        "<self-secondary of primary AUF, variant={weights_tag}, dtype={selected_dtype:?}>"
    ));

    eprintln!(
        "[AUF self-secondary] auto-activated: variant={} primary_dtype={:?} secondary_dtype={:?} layers={}",
        weights_tag, primary_dtype, selected_dtype, num_layers,
    );

    Ok(std::sync::Arc::new(SecondaryMmap::Auf(AufSecondaryMmap {
        view,
        layer_index,
        source_path,
        is_pre_converted_soa,
    })))
}

/// META.default_dtype / CLI dtype 문자열을 `TensorDType`으로 변환.
///
/// `reader.rs`의 `dtype_str_to_tensor_dtype`와 동일 로직이지만 cross-crate
/// 공유를 피하기 위해 module-local로 유지한다.
pub(crate) fn dtype_str_to_tensor_dtype_local(s: &str) -> Option<TensorDType> {
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

/// Validate AUF META against primary `ModelConfig`.
pub fn check_auf_metadata(
    primary: &ModelConfig,
    meta: &AufMeta,
    path: &Path,
) -> Result<(), LoadError> {
    macro_rules! check {
        ($primary_field:expr, $auf_field:expr, $name:literal) => {
            if $primary_field as u64 != $auf_field as u64 {
                return Err(LoadError::MetadataMismatch {
                    field: $name,
                    primary: format!("{}", $primary_field),
                    secondary: format!("{} (AUF: {})", $auf_field, path.display()),
                });
            }
        };
    }
    check!(primary.hidden_size, meta.hidden_dim, "hidden_size");
    check!(
        primary.num_hidden_layers,
        meta.n_layers,
        "num_hidden_layers"
    );
    check!(
        primary.num_attention_heads,
        meta.n_heads_q,
        "num_attention_heads"
    );
    check!(
        primary.num_key_value_heads,
        meta.n_kv_heads,
        "num_key_value_heads"
    );
    check!(primary.head_dim, meta.head_dim, "head_dim");
    check!(primary.intermediate_size, meta.ffn_dim, "intermediate_size");
    check!(primary.vocab_size, meta.vocab_size, "vocab_size");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn is_auf_path_by_extension() {
        assert!(is_auf_path(&PathBuf::from("/tmp/model.auf")));
        assert!(!is_auf_path(&PathBuf::from("/tmp/model.gguf")));
        assert!(!is_auf_path(&PathBuf::from("/tmp/model")));
    }

    #[test]
    fn auf_dtype_to_engine_round_trip() {
        assert_eq!(
            auf_dtype_to_engine(TensorDType::Q4_0.as_u32()),
            Some(DType::Q4_0)
        );
        assert_eq!(
            auf_dtype_to_engine(TensorDType::F16.as_u32()),
            Some(DType::F16)
        );
        assert_eq!(
            auf_dtype_to_engine(TensorDType::F32.as_u32()),
            Some(DType::F32)
        );
        assert_eq!(auf_dtype_to_engine(0xFFFF_FFFF), None);
    }

    #[test]
    fn resolve_backend_tag_candidates_auto_returns_nonempty() {
        let v = resolve_backend_tag_candidates(SecondaryLayoutChoice::Auto);
        assert!(!v.is_empty());
        // CpuAos must always be in the Auto fallback chain.
        assert!(v.contains(&BackendTag::CpuAos));
    }

    #[test]
    fn resolve_backend_tag_candidates_soa_is_adreno_only() {
        let v = resolve_backend_tag_candidates(SecondaryLayoutChoice::Soa);
        assert_eq!(v, vec![BackendTag::AdrenoSoa]);
    }

    // ── W-AUF-2 C4: AUF self-secondary 통합 테스트 ───────────────────────────

    /// Helper: in-memory multi-dtype AUF bytes (CPU_AOS variant, Q4_0 + F16).
    ///
    /// 1-layer 1-tensor 구조로 최소화:
    /// - layer 0, AttnQ, Q4_0 (32 bytes payload)
    /// - layer 0, AttnQ, F16 (32 bytes payload)
    ///
    /// section offset 검증을 위해 두 dtype variant 모두 같은 `variant_offsets[0]`
    /// (= CPU_AOS variant slot)를 사용한다.
    fn build_multi_dtype_auf_bytes() -> Vec<u8> {
        use crate::auf::section::TAG_WEIGHTS_CPU_AOS;
        use crate::auf::tensor_index::{TensorEntry, TensorIndex};
        use crate::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
        use crate::auf::writer::AufWriter;
        use crate::auf::{AufMeta, BackendTag, TensorKind};

        let _ = BackendTag::CpuAos; // import keeper

        let payload_q4_0: Vec<u8> = (0..32).map(|i| (i & 0xFF) as u8).collect();
        let payload_f16: Vec<u8> = (0..32).map(|i| ((i + 100) & 0xFF) as u8).collect();
        // CPU_AOS section payload: q4_0 (offset 0) followed by f16 (offset 32).
        let mut weights_payload = payload_q4_0.clone();
        weights_payload.extend_from_slice(&payload_f16);

        let mut tag_buf = [0u8; 24];
        tag_buf[..TAG_WEIGHTS_CPU_AOS.len()].copy_from_slice(TAG_WEIGHTS_CPU_AOS.as_bytes());

        let entries = vec![
            TensorEntry {
                layer_idx: 0,
                kind: TensorKind::AttnQ.as_u32(),
                dtype: TensorDType::Q4_0.as_u32(),
                shape: vec![1, 32],
                alignment: 32,
                variant_offsets: vec![0],
                variant_sizes: vec![32],
            },
            TensorEntry {
                layer_idx: 0,
                kind: TensorKind::AttnQ.as_u32(),
                dtype: TensorDType::F16.as_u32(),
                shape: vec![1, 16],
                alignment: 32,
                variant_offsets: vec![32],
                variant_sizes: vec![32],
            },
        ];
        let tensor_index = TensorIndex {
            variant_tags: vec![tag_buf],
            entries,
        };

        AufWriter::new(
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
                default_dtype: Some("Q4_0".to_owned()),
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
        .with_multi_dtype(true)
        .with_tensor_index(tensor_index)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, weights_payload)
        .build()
        .unwrap()
    }

    fn make_test_config() -> ModelConfig {
        use crate::model_config::ModelArch;
        ModelConfig {
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
        }
    }

    #[test]
    fn from_auf_self_secondary_selects_other_dtype() {
        use crate::auf::reader::open_from_bytes;
        use crate::auf::{BackendTag, TensorDType};

        let bytes = build_multi_dtype_auf_bytes();
        let view = std::sync::Arc::new(open_from_bytes(bytes, BackendTag::CpuAos).unwrap());
        let cfg = make_test_config();

        // primary_dtype = F16 → secondary가 Q4_0을 골라야 한다 (정합 swap 방향 F16→Q4_0).
        let mmap = super::from_auf_self_secondary(
            std::sync::Arc::clone(&view),
            BackendTag::CpuAos,
            TensorDType::F16,
            &cfg,
        )
        .expect("self-secondary should auto-select Q4_0 (other dtype)");

        // layer 0, attn_q.weight가 Q4_0 dtype + 32 bytes로 가용.
        let info = mmap
            .layer_tensor(0, "attn_q.weight")
            .expect("layer_tensor must find attn_q.weight from Q4_0 entry");
        assert_eq!(info.dtype, DType::Q4_0, "self-secondary must pick Q4_0");
        assert_eq!(info.len, 32, "Q4_0 entry size");
        // Q4_0 payload는 weights section offset 0부터 시작 → bytes의 첫 byte는 0.
        let bytes_view = mmap.tensor_bytes(info);
        assert_eq!(bytes_view[0], 0, "Q4_0 payload starts with byte 0");
    }

    #[test]
    fn from_auf_self_secondary_rejects_reverse_swap_q4_to_f16() {
        use crate::auf::reader::open_from_bytes;
        use crate::auf::{BackendTag, TensorDType};

        let bytes = build_multi_dtype_auf_bytes();
        let view = std::sync::Arc::new(open_from_bytes(bytes, BackendTag::CpuAos).unwrap());
        let cfg = make_test_config();

        // primary_dtype = Q4_0 + secondary candidate = F16 → reject (정책: F16→Q4_0 단방향만).
        let err = super::from_auf_self_secondary(view, BackendTag::CpuAos, TensorDType::Q4_0, &cfg)
            .expect_err("Q4_0 primary + F16 secondary must be ReverseSwapRejected");
        assert!(
            matches!(err, LoadError::ReverseSwapRejected { .. }),
            "expected ReverseSwapRejected, got {err:?}"
        );
    }

    #[test]
    fn from_auf_self_secondary_errors_when_only_primary_dtype() {
        use crate::auf::reader::open_from_bytes;
        use crate::auf::section::TAG_WEIGHTS_CPU_AOS;
        use crate::auf::tensor_index::{TensorEntry, TensorIndex};
        use crate::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
        use crate::auf::writer::AufWriter;
        use crate::auf::{AufMeta, BackendTag, TensorDType, TensorKind};

        // single-dtype AUF (Q4_0 only) — multi-dtype bit OFF.
        let payload: Vec<u8> = vec![0u8; 32];
        let mut tag_buf = [0u8; 24];
        tag_buf[..TAG_WEIGHTS_CPU_AOS.len()].copy_from_slice(TAG_WEIGHTS_CPU_AOS.as_bytes());
        let bytes = AufWriter::new(
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
                default_dtype: Some("Q4_0".to_owned()),
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
                dtype: TensorDType::Q4_0.as_u32(),
                shape: vec![1, 32],
                alignment: 32,
                variant_offsets: vec![0],
                variant_sizes: vec![32],
            }],
        })
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, payload)
        .build()
        .unwrap();

        let view = std::sync::Arc::new(open_from_bytes(bytes, BackendTag::CpuAos).unwrap());
        let cfg = make_test_config();
        let err = super::from_auf_self_secondary(view, BackendTag::CpuAos, TensorDType::Q4_0, &cfg)
            .expect_err("single-dtype self-secondary must fail (no swap candidate)");
        let msg = format!("{err}");
        assert!(
            msg.contains("no swap candidates") || msg.contains("auf_tool build --dtypes"),
            "error should mention missing swap candidates, got: {msg}"
        );
    }

    #[test]
    fn from_auf_self_secondary_rejects_adreno_soa_with_only_f16() {
        // primary=AdrenoSoa, only non-primary dtype = F16 → AdrenoSoaF16Rejected.
        use crate::auf::reader::open_from_bytes;
        use crate::auf::section::TAG_WEIGHTS_ADRENO_SOA;
        use crate::auf::tensor_index::{TensorEntry, TensorIndex};
        use crate::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
        use crate::auf::writer::AufWriter;
        use crate::auf::{AufMeta, BackendTag, TensorDType, TensorKind};

        let payload: Vec<u8> = vec![0u8; 64];
        let mut tag_buf = [0u8; 24];
        tag_buf[..TAG_WEIGHTS_ADRENO_SOA.len()].copy_from_slice(TAG_WEIGHTS_ADRENO_SOA.as_bytes());
        let bytes = AufWriter::new(
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
                default_dtype: Some("Q4_0".to_owned()),
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
        .with_multi_dtype(true)
        .with_tensor_index(TensorIndex {
            variant_tags: vec![tag_buf],
            entries: vec![
                TensorEntry {
                    layer_idx: 0,
                    kind: TensorKind::AttnQ.as_u32(),
                    dtype: TensorDType::Q4_0.as_u32(),
                    shape: vec![1, 32],
                    alignment: 32,
                    variant_offsets: vec![0],
                    variant_sizes: vec![32],
                },
                TensorEntry {
                    layer_idx: 0,
                    kind: TensorKind::AttnQ.as_u32(),
                    dtype: TensorDType::F16.as_u32(),
                    shape: vec![1, 16],
                    alignment: 32,
                    variant_offsets: vec![32],
                    variant_sizes: vec![32],
                },
            ],
        })
        .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, payload)
        .build()
        .unwrap();

        let view = std::sync::Arc::new(open_from_bytes(bytes, BackendTag::AdrenoSoa).unwrap());
        let cfg = make_test_config();
        let err =
            super::from_auf_self_secondary(view, BackendTag::AdrenoSoa, TensorDType::Q4_0, &cfg)
                .expect_err("AdrenoSoa primary + F16 secondary must be rejected");
        assert!(
            matches!(err, LoadError::AdrenoSoaF16Rejected),
            "expected AdrenoSoaF16Rejected, got {err:?}"
        );
    }
}
