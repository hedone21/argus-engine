//! `weight-swap` — WeightSwap DECIDER 기법 크레이트 (EPIC 3 B3-3).
//!
//! 엔진 코어에서 추출한 self-registering 기법 크레이트 (h2o/layer-importance/attn-score 선례):
//! `argus-extension-api` + `linkme` 만 의존하고, importance × ε bottom-k 레이어 선택 랭킹
//! (`WeightSwapDecider`) 과 그것을 plan-returning `WeightStage` 표면으로 감싸는 어댑터
//! (`WeightSwapDeciderAsStage`) 를 보유하며 `"swap"` 으로 `#[distributed_slice(WEIGHT_STAGES)]`
//! 에 자가 등록한다. 엔진은 `use weight_swap as _;` 로 force-link 하고 decider 타입을
//! `pub use weight_swap::{...}` 로 되받아(engine/src/qcf.rs 의 layer-importance 거울) off-seam
//! 소비자 경로를 bit-identical 로 보존한다.
//!
//! plan/executor 분리(D1): decider 는 읽고 plan(선택 layer 집합)만 내며, swap EXECUTOR
//! (SecondaryMmap 재물질화 / cl_mem 수명 / qk_permute / ratio_generation)는 엔진 독점이다.
//!
//! Spec: ENG-ALG-215, ENG-ALG-217, INV-127.

use std::collections::HashSet;

use argus_extension_api::{
    LayerDirective, LayerDispatch, TensorDtype, WEIGHT_STAGES, WeightDispatchPlan, WeightStage,
    WeightStageCtx, WeightStageParams, WeightStageReg,
};
use linkme::distributed_slice;

// ── Public types ──────────────────────────────────────────────────────────────

/// Layer-selection algorithm for `WeightSwapDecider` (U5 ablation, EuroSys'27).
///
/// The default `ImportanceAware` matches production ARGUS behavior; the others
/// exist for the U5 "Layer-swap algorithm comparison" table that shows the
/// quality cost of *not* using importance-aware ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SwapAlgorithm {
    /// `importance × ε` ascending bottom-k (current production default).
    #[default]
    ImportanceAware,
    /// Layer index ascending (0 → N-1).
    Sequential,
    /// Layer index descending (N-1 → 0).
    Reverse,
    /// Evenly spaced across candidates (matches the existing fallback path).
    Uniform,
    /// `importance × ε` descending top-k (worst-case baseline; the inverse of
    /// `ImportanceAware`).
    AntiImportance,
}

impl SwapAlgorithm {
    /// Parse a CLI-style identifier (case-insensitive).
    pub fn from_cli(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "imp" | "importance" | "importance-aware" => Some(Self::ImportanceAware),
            "seq" | "sequential" => Some(Self::Sequential),
            "rev" | "reverse" => Some(Self::Reverse),
            "uni" | "uniform" => Some(Self::Uniform),
            "anti" | "anti-importance" | "antiimportance" => Some(Self::AntiImportance),
            _ => None,
        }
    }

    /// Stable short identifier used in dump JSON / logs.
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::ImportanceAware => "imp",
            Self::Sequential => "seq",
            Self::Reverse => "rev",
            Self::Uniform => "uni",
            Self::AntiImportance => "anti",
        }
    }
}

/// Result of `WeightSwapDecider::decide()`.
#[derive(Debug, Clone)]
pub struct SwapDecision {
    /// Decoder layer indices selected for this swap batch.
    pub selected_layers: Vec<usize>,
    /// QCF_swap estimate for the selected set (ENG-ALG-217).
    pub qcf_swap_estimate: f32,
    /// `true` when a uniform fallback was used (importance or noise absent).
    pub fallback_used: bool,
}

/// Stateless decider that converts a swap `budget` (절대 layer 수) to a layer
/// index set.
///
/// Both `importance` and `noise` are optional flat per-layer slices (index =
/// layer id): pass `None` when the table has not been built yet.  When either
/// is absent (or effectively empty / uncomputed), the `ImportanceAware` /
/// `AntiImportance` algorithms fall back to `Uniform`.
///
/// **계약**: `noise = Some(_)` ⟺ 노이즈 ε 가 secondary tensor 에서 실제 계산됨
/// (구 `QuantNoiseAccess::is_computed() == true`). 호출자는 `is_computed()` 가
/// `false` 일 때 `None` 을 넘겨 기존 uniform fallback 게이트를 보존한다.
pub struct WeightSwapDecider<'a> {
    /// Per-layer importance (index = layer id, value = `SubLayer::Full`
    /// importance). None = fallback. 길이는 `n_decoder_layers` 이상이어야 한다.
    pub importance: Option<&'a [f32]>,
    /// Per-layer quantization noise ε (index = layer id). None = fallback /
    /// uncomputed. NaN 원소는 candidate 에서 제외된다 (INV-127).
    pub noise: Option<&'a [f32]>,
    /// Total number of decoder layers in the model.
    pub n_decoder_layers: usize,
    /// Layers that are already at the target dtype — excluded from re-selection.
    pub currently_swapped: &'a [usize],
    /// When `true`, allow layer 0 and the last decoder layer to be selected
    /// for swap. Default (`false`) keeps them protected, matching production
    /// safety semantics. Used for research/ablation experiments (e.g. PPL
    /// teacher-forcing NLL measurement with full-coverage swap).
    pub allow_boundary_layers: bool,
    /// Layer-selection algorithm. Default = `ImportanceAware` (production).
    /// U5 ablation uses the other variants.
    pub algorithm: SwapAlgorithm,
}

impl<'a> WeightSwapDecider<'a> {
    /// Decide which layers to swap for the given `budget` (ENG-ALG-215).
    ///
    /// `budget` 은 이 호출에서 추가로 swap 할 **절대 layer 수**다. ratio→count
    /// 환산 (`floor(ratio*n)` 에서 `currently_swapped` 차감)은 호출자 책임이다 (KV
    /// `target_len` 거울; MW-C).
    ///
    /// Returns a `SwapDecision` containing the layer indices, the computed
    /// QCF_swap estimate, and whether the uniform fallback was used.
    pub fn decide(&self, budget: usize) -> SwapDecision {
        if budget == 0 || self.n_decoder_layers == 0 {
            return SwapDecision {
                selected_layers: Vec::new(),
                qcf_swap_estimate: 0.0,
                fallback_used: false,
            };
        }

        let n = self.n_decoder_layers;
        let already_swapped_set: HashSet<usize> = self.currently_swapped.iter().copied().collect();
        let needed = budget;

        // Protected layers: exclude layer 0 and last decoder layer by default.
        // `allow_boundary_layers` overrides this for research/ablation runs
        // (e.g. PPL teacher-forcing measurement of full-coverage swap NLL).
        let mut protected = HashSet::new();
        if !self.allow_boundary_layers {
            protected.insert(0usize);
            if n > 1 {
                protected.insert(n - 1);
            }
        }

        // Build candidate list: exclude protected, already swapped, NaN-ε layers.
        let candidates: Vec<usize> = (0..n)
            .filter(|i| !protected.contains(i))
            .filter(|i| !already_swapped_set.contains(i))
            .filter(|i| {
                // INV-127: exclude layers with NaN ε when the table is present.
                // noise=None → ε 전체 가용으로 취급 (include all).
                self.noise
                    .map(|s| s.get(*i).map(|v| !v.is_nan()).unwrap_or(false))
                    .unwrap_or(true)
            })
            .collect();

        // `ImportanceAware` and `AntiImportance` require both tables; without
        // them they fall back to `Uniform`. The pure layer-index algorithms
        // (`Sequential`, `Reverse`, `Uniform`) never use the tables.
        // noise=Some(_) ⟺ (구) is_computed()==true (호출자 계약, struct docs 참조).
        let scored_path_available =
            self.importance.map(|s| !s.is_empty()).unwrap_or(false) && self.noise.is_some();
        let effective_algo = match self.algorithm {
            SwapAlgorithm::ImportanceAware | SwapAlgorithm::AntiImportance
                if !scored_path_available =>
            {
                SwapAlgorithm::Uniform
            }
            other => other,
        };
        let use_fallback =
            matches!(self.algorithm, SwapAlgorithm::ImportanceAware) && !scored_path_available;

        let selected: Vec<usize> = match effective_algo {
            SwapAlgorithm::Sequential => candidates.iter().take(needed).copied().collect(),
            SwapAlgorithm::Reverse => candidates.iter().rev().take(needed).copied().collect(),
            SwapAlgorithm::Uniform => uniform_select_by_index(needed, &candidates),
            SwapAlgorithm::ImportanceAware | SwapAlgorithm::AntiImportance => {
                let imp = self.importance.expect("importance checked non-empty");

                // Key = importance[i] (= SubLayer::Full importance, flattened by
                // caller). ε was previously multiplied in but empirical
                // measurement showed Spearman ρ(imp × ε, imp) = 0.998 under Q4_0
                // (ε layer-uniform), so ε contributes no ranking signal. Removed
                // for §4 simplicity. ε is still checked at the candidate-filter
                // stage above for NaN exclusion (INV-127).
                let mut scored: Vec<(usize, f32)> = candidates
                    .iter()
                    .map(|&i| {
                        let imp_val = imp.get(i).copied().unwrap_or(0.0);
                        (i, imp_val)
                    })
                    .collect();

                // `ImportanceAware`: ascending (smallest key first — cheap layers).
                // `AntiImportance`: descending (largest key first — costly layers).
                let ascending = matches!(effective_algo, SwapAlgorithm::ImportanceAware);
                scored.sort_by(|(ia, ka), (ib, kb)| {
                    let primary = if ascending {
                        ka.partial_cmp(kb).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        kb.partial_cmp(ka).unwrap_or(std::cmp::Ordering::Equal)
                    };
                    primary.then(ia.cmp(ib))
                });

                scored.truncate(needed);
                scored.into_iter().map(|(i, _)| i).collect()
            }
        };

        let qcf = compute_qcf_swap_internal(&selected, n, self.importance, self.noise);

        SwapDecision {
            selected_layers: selected,
            qcf_swap_estimate: qcf,
            fallback_used: use_fallback,
        }
    }

    /// Dry-run: compute QCF_swap at the given `budget` without executing a swap.
    ///
    /// Returns `(selected_layers, qcf_swap)`.  Used for `LayerSwapEstimate`
    /// curve sampling in `QcfEstimate` (ENG-ALG-218).
    pub fn decide_dry_run(&self, budget: usize) -> (Vec<usize>, f32) {
        let decision = self.decide(budget);
        (decision.selected_layers, decision.qcf_swap_estimate)
    }
}

// ── QCF_swap computation (ENG-ALG-217) ───────────────────────────────────────

/// Internal QCF_swap implementation used by both the engine-side timed wrapper
/// (`compute_qcf_weight_swap`, which keeps the engine-only `qcf_timer!`) and the
/// decider above. Promoted to `pub` (B3-3) so the engine wrapper can delegate
/// across the crate boundary, keeping a single copy of the arithmetic.
///
/// ```text
/// QCF_swap(S) = Σ_{i ∈ S} importance_i × ε_i
///               ───────────────────────────────
///               Σ_{j ∈ all_valid} importance_j × ε_j
/// ```
///
/// - Layers with NaN ε are excluded from both numerator and denominator.
/// - Missing importance entries (table absent) default to `1.0`.
/// - Returns `0.0` when `swap_set` is empty or denominator ≈ 0.
///
/// `importance`/`noise` 는 layer 인덱스 기준의 평탄 슬라이스다 (MW-C). NaN 원소는
/// numerator/denominator 양쪽에서 제외된다.
pub fn compute_qcf_swap_internal(
    swap_set: &[usize],
    n_decoder_layers: usize,
    importance: Option<&[f32]>,
    noise: Option<&[f32]>,
) -> f32 {
    if swap_set.is_empty() {
        return 0.0;
    }

    // QCF 디폴트 1.0 유지 — ranking 의 0.0 과 비대칭 보존 (MW-C).
    let imp_for = |i: usize| -> f32 { importance.and_then(|s| s.get(i).copied()).unwrap_or(1.0) };

    // ε removed from the QCF_swap formula (see decide() comment). The noise
    // table is still consulted here only to exclude NaN-ε layers from the
    // sum (INV-127 — they were never valid swap candidates).
    let valid_layer = |i: usize| -> bool {
        noise
            .map(|s| s.get(i).map(|v| !v.is_nan()).unwrap_or(false))
            .unwrap_or(true)
    };

    let swap_set_hash: HashSet<usize> = swap_set.iter().copied().collect();

    let numerator: f32 = (0..n_decoder_layers)
        .filter(|i| swap_set_hash.contains(i))
        .filter(|i| valid_layer(*i))
        .map(imp_for)
        .sum();

    let denominator: f32 = (0..n_decoder_layers)
        .filter(|i| valid_layer(*i))
        .map(imp_for)
        .sum();

    if denominator < 1e-8 {
        return 0.0;
    }

    (numerator / denominator).clamp(0.0, 1.0)
}

// ── Uniform fallback helper ───────────────────────────────────────────────────

/// Uniformly spaced index selection from a candidate slice (ENG-ALG-213 fallback).
///
/// `needed` layers are chosen at evenly-spaced positions within `candidates`.
/// Deterministic — no random seed.
fn uniform_select_by_index(needed: usize, candidates: &[usize]) -> Vec<usize> {
    if needed == 0 || candidates.is_empty() {
        return Vec::new();
    }
    if needed >= candidates.len() {
        return candidates.to_vec();
    }
    let stride = candidates.len() as f32 / needed as f32;
    let mut out = Vec::with_capacity(needed);
    for k in 0..needed {
        let idx = (k as f32 * stride).floor() as usize;
        out.push(candidates[idx.min(candidates.len() - 1)]);
    }
    out
}

// ── `WeightStage` 어댑터 + 등록 ─────────────────────────────────────────────────

/// `WeightSwapDecider` 를 `WeightStage` 로 노출하는 빌트인 어댑터 (MW-C).
///
/// 상태가 없는 stateless 어댑터다 — 매 `plan()` 호출마다 ctx 의 읽기 값으로
/// decider 를 즉석 생성한다. 누적 상태가 없어 interior-mutability(D4) 불요.
pub struct WeightSwapDeciderAsStage {
    /// 경계 레이어(0, 마지막)도 swap 대상에 포함할지 (연구/ablation; 기본 false).
    allow_boundary_layers: bool,
    /// 레이어 선택 알고리즘 (기본 `ImportanceAware` = production).
    algorithm: SwapAlgorithm,
}

impl WeightSwapDeciderAsStage {
    /// 등록 팩토리에서 호출 — `WeightStageParams` 로부터 어댑터를 만든다.
    /// algorithm 은 production 기본(`ImportanceAware`)으로 고정한다.
    pub fn new(p: WeightStageParams) -> Self {
        Self {
            allow_boundary_layers: p.allow_boundary_layers,
            algorithm: SwapAlgorithm::ImportanceAware,
        }
    }
}

impl WeightStage for WeightSwapDeciderAsStage {
    fn name(&self) -> &str {
        "swap"
    }

    fn plan(&self, ctx: &dyn WeightStageCtx) -> Option<WeightDispatchPlan> {
        let n = ctx.n_layers();
        let budget = ctx.budget();
        if budget == 0 || n == 0 {
            return None;
        }

        // ctx 가 노출하는 flat per-layer 메트릭. decider 의 `Option<&[f32]>`
        // 필드와 동형 — 그대로 전달한다(noise=Some ⟺ is_computed 계약은 엔진의
        // WeightStageCtx impl(MW-D)이 책임진다).
        let importance = ctx.importance();
        let noise = ctx.quant_noise();

        // 현재 이미 Q4_0(=swap 완료) 인 레이어는 재선택 제외.
        let currently_swapped: Vec<usize> = (0..n)
            .filter(|&i| ctx.current_format(i) == TensorDtype::Q4_0)
            .collect();

        let decider = WeightSwapDecider {
            importance,
            noise,
            n_decoder_layers: n,
            currently_swapped: &currently_swapped,
            allow_boundary_layers: self.allow_boundary_layers,
            algorithm: self.algorithm,
        };
        let decision = decider.decide(budget);

        if decision.selected_layers.is_empty() {
            return None;
        }

        // swap = precision F16→Q4_0, dispatch=Full (precision ⊥ dispatch, R1).
        let per_layer = decision
            .selected_layers
            .iter()
            .map(|&l| LayerDirective {
                layer: l,
                dispatch: LayerDispatch::Full,
                precision: Some(TensorDtype::Q4_0),
            })
            .collect();
        Some(WeightDispatchPlan { per_layer })
    }
}

#[distributed_slice(WEIGHT_STAGES)]
static SWAP_STAGE: WeightStageReg = WeightStageReg {
    name: "swap",
    make: |p| Box::new(WeightSwapDeciderAsStage::new(p)),
};

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{LayerMetricKind, find_weight_stage};

    // ── decider 단위 테스트 helper builders ──
    //
    // MW-C: importance/noise 는 layer 인덱스 기준 평탄 `Vec<f32>` (구
    // ImportanceTable/QuantNoiseTable 대신). expected 결과는 동일하게 유지한다
    // (bit-identical). ratio→budget 환산은 테스트가 직접 `floor(ratio*n) -
    // |currently_swapped|` 로 수행한다 (구 decide(ratio) 가 내부에서 하던 일).

    /// layer 인덱스 기준 평탄 importance. index=layer_id, value=Full importance.
    fn make_importance(entries: Vec<(usize, f32)>) -> Vec<f32> {
        let n = entries.iter().map(|(id, _)| id + 1).max().unwrap_or(0);
        let mut out = vec![0.0f32; n];
        for (id, imp) in entries {
            out[id] = imp;
        }
        out
    }

    /// layer 인덱스 기준 평탄 ε. index=layer_id, value=ε (NaN 허용).
    fn make_noise(vals: Vec<f32>) -> Vec<f32> {
        vals
    }

    // ── Normal-path test (spec example) ──────────────────────────────────────

    /// 4-layer fixture from the spec (post-ε removal):
    /// importance = [0.1, 0.5, 0.3, 0.7], ε = [0.2, 0.1, 0.3, 0.05] (ε no
    /// longer affects ranking; kept here only for the NaN-exclusion path).
    /// key = importance = [0.1, 0.5, 0.3, 0.7]
    /// Layers 0 and 3 are protected; candidates = [1, 2].
    /// budget = floor(0.5 × 4) = 2 → both candidates selected.
    /// qcf_swap = (imp[1] + imp[2]) / Σ imp = (0.5 + 0.3) / 1.6 = 0.5
    #[test]
    fn decide_normal_path_spec_example() {
        let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);

        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };

        // budget = floor(0.5 * 4) - 0 = 2
        let decision = decider.decide(2);

        assert!(!decision.fallback_used, "should use scored path");
        assert_eq!(decision.selected_layers.len(), 2);
        // Both candidates (layers 1 and 2) should be selected
        assert!(decision.selected_layers.contains(&1));
        assert!(decision.selected_layers.contains(&2));

        // qcf_swap = 0.8 / 1.6 = 0.5
        let expected_qcf = 0.5f32;
        assert!(
            (decision.qcf_swap_estimate - expected_qcf).abs() < 1e-4,
            "qcf={:.6}, expected={:.6}",
            decision.qcf_swap_estimate,
            expected_qcf
        );
    }

    /// NaN ε layers must be excluded from candidates (INV-127).
    #[test]
    fn nan_epsilon_excluded_inv_127() {
        let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
        let noise = make_noise(vec![0.2, f32::NAN, 0.3, 0.05]);

        // Layer 1 has NaN ε → must be excluded even though layer 0 and 3 are the protected ones
        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };

        // budget=2, candidates are [1, 2] normally. With layer 1 NaN excluded → [2] only.
        let decision = decider.decide(2);
        assert!(
            !decision.selected_layers.contains(&1),
            "layer 1 with NaN epsilon must be excluded"
        );
        // Layer 2 should be selected (only valid candidate)
        assert!(decision.selected_layers.contains(&2));
    }

    /// When ImportanceTable is empty, uniform fallback is used.
    #[test]
    fn fallback_when_importance_empty() {
        let empty_importance: Vec<f32> = Vec::new();
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);

        let decider = WeightSwapDecider {
            importance: Some(&empty_importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };

        let decision = decider.decide(2);
        assert!(
            decision.fallback_used,
            "should use fallback when importance empty"
        );
        // Uniform fallback still selects some layers
        assert!(!decision.selected_layers.is_empty());
    }

    /// budget = 0 → empty decision, qcf_swap = 0.0.
    #[test]
    fn ratio_zero_returns_empty() {
        let importance = make_importance(vec![(0, 0.5), (1, 0.3), (2, 0.4), (3, 0.6)]);
        let noise = make_noise(vec![0.1, 0.2, 0.3, 0.1]);

        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };

        // budget = floor(0.0 * 4) = 0
        let decision = decider.decide(0);
        assert!(decision.selected_layers.is_empty());
        assert_eq!(decision.qcf_swap_estimate, 0.0);
    }

    /// `currently_swapped` layers must not be re-selected.
    #[test]
    fn currently_swapped_excluded() {
        let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);

        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[2], // layer 2 already swapped
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };

        // budget = floor(0.5 * 4) - |{2}| = 2 - 1 = 1
        let decision = decider.decide(1);
        assert!(
            !decision.selected_layers.contains(&2),
            "already-swapped layer 2 must not be re-selected"
        );
    }

    /// Layer 0 and last layer must never be selected regardless of importance.
    #[test]
    fn protected_layers_never_selected() {
        // Make layer 0 and 3 look very cheap (low key) to verify they are still excluded
        let importance = make_importance(vec![(0, 0.001), (1, 0.9), (2, 0.9), (3, 0.001)]);
        let noise = make_noise(vec![0.001, 0.9, 0.9, 0.001]);

        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };

        // budget = floor(0.9 * 4) = 3
        let decision = decider.decide(3);
        assert!(
            !decision.selected_layers.contains(&0),
            "layer 0 is protected"
        );
        assert!(
            !decision.selected_layers.contains(&3),
            "last decoder layer is protected"
        );
    }

    /// `allow_boundary_layers=true` 면 layer 0 과 마지막 layer 도 후보가 된다
    /// (research/ablation 용; PPL teacher-forcing 전체 coverage 실험).
    #[test]
    fn boundary_layers_included_when_allowed() {
        // 4 layer 모두 동일한 importance×ε → budget=4 이면 4 layer 전부 선정 가능.
        let importance = make_importance(vec![(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]);
        let noise = make_noise(vec![0.5, 0.5, 0.5, 0.5]);

        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
            allow_boundary_layers: true,
            algorithm: SwapAlgorithm::ImportanceAware,
        };

        // budget = floor(1.0 * 4) = 4
        let decision = decider.decide(4);
        assert_eq!(
            decision.selected_layers.len(),
            4,
            "budget=4 with allow_boundary_layers should select all 4 layers"
        );
        assert!(
            decision.selected_layers.contains(&0),
            "layer 0 must be selectable when allow_boundary_layers=true"
        );
        assert!(
            decision.selected_layers.contains(&3),
            "last layer must be selectable when allow_boundary_layers=true"
        );
    }

    /// Fallback path 에서도 boundary 우회가 동작해야 한다 (uniform_select).
    #[test]
    fn boundary_layers_allowed_in_fallback_path() {
        // importance None → fallback uniform_select_by_index 사용.
        let decider = WeightSwapDecider {
            importance: None,
            noise: None,
            n_decoder_layers: 4,
            currently_swapped: &[],
            allow_boundary_layers: true,
            algorithm: SwapAlgorithm::ImportanceAware,
        };

        // budget = floor(1.0 * 4) = 4
        let decision = decider.decide(4);
        assert!(
            decision.fallback_used,
            "absent importance/noise should trigger fallback path"
        );
        assert_eq!(decision.selected_layers.len(), 4);
        assert!(decision.selected_layers.contains(&0));
        assert!(decision.selected_layers.contains(&3));
    }

    // ── dry-run test ──────────────────────────────────────────────────────────

    #[test]
    fn decide_dry_run_matches_decide() {
        let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };
        let (layers_dr, qcf_dr) = decider.decide_dry_run(2);
        let decision = decider.decide(2);
        assert_eq!(layers_dr, decision.selected_layers);
        assert!((qcf_dr - decision.qcf_swap_estimate).abs() < 1e-8);
    }

    // ── 어댑터(`WeightSwapDeciderAsStage`) 테스트 ───────────────────────────────

    /// flat importance/noise/budget/current_format 를 직접 보유하는 최소 mock ctx.
    struct MockWeightCtx {
        n_layers: usize,
        budget: usize,
        importance: Option<Vec<f32>>,
        noise: Option<Vec<f32>>,
        /// Q4_0(swap 완료) 로 간주할 레이어 인덱스.
        swapped: Vec<usize>,
    }

    impl WeightStageCtx for MockWeightCtx {
        fn n_layers(&self) -> usize {
            self.n_layers
        }
        fn budget(&self) -> usize {
            self.budget
        }
        fn pressure(&self) -> u8 {
            0
        }
        fn current_format(&self, layer: usize) -> TensorDtype {
            if self.swapped.contains(&layer) {
                TensorDtype::Q4_0
            } else {
                TensorDtype::F16
            }
        }
        fn layer_metric(&self, kind: LayerMetricKind) -> Option<&[f32]> {
            match kind {
                LayerMetricKind::Importance => self.importance.as_deref(),
                LayerMetricKind::QuantNoise => self.noise.as_deref(),
            }
        }
    }

    /// stage `plan()` 의 선택 layer 집합 == 동일 입력으로 직접 호출한
    /// `decider.decide(budget).selected_layers` (bit-identical).
    #[test]
    fn stage_plan_matches_decider() {
        let importance = vec![0.1f32, 0.5, 0.3, 0.7];
        let noise = vec![0.2f32, 0.1, 0.3, 0.05];
        let ctx = MockWeightCtx {
            n_layers: 4,
            budget: 2,
            importance: Some(importance.clone()),
            noise: Some(noise.clone()),
            swapped: Vec::new(),
        };

        let stage = WeightSwapDeciderAsStage::new(WeightStageParams {
            allow_boundary_layers: false,
        });
        let plan = stage.plan(&ctx).expect("plan should be Some");
        let stage_layers: Vec<usize> = plan.per_layer.iter().map(|d| d.layer).collect();

        // 동일 입력으로 decider 직접 호출.
        let currently_swapped: Vec<usize> = Vec::new();
        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &currently_swapped,
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };
        let direct = decider.decide(2);

        assert_eq!(
            stage_layers, direct.selected_layers,
            "stage plan layer set must equal decider.decide(budget).selected_layers"
        );
        // precision=Q4_0, dispatch=Full (R1 직교).
        for d in &plan.per_layer {
            assert!(matches!(d.dispatch, LayerDispatch::Full));
            assert_eq!(d.precision, Some(TensorDtype::Q4_0));
        }
    }

    /// B3-0 위험(HIGH): live commit 은 `read_allow_boundary_env()` 를 1회 읽어
    /// `WeightStageParams.allow_boundary_layers` 로 주입한다(plan() 내부 env 재독 금지). 어댑터가
    /// 그 flag 를 decider 로 정확히 threading 하는지 true/false 양쪽에서 검증한다 — 동일 flag 의
    /// 직접 decider 와 selected_layers 가 **순서까지** bit-identical(Vec== 는 order-sensitive).
    #[test]
    fn stage_plan_threads_allow_boundary_both_ways() {
        // 경계 레이어(0, 3)가 high importance → allow_boundary flag 가 선택을 가른다.
        let importance = vec![0.9f32, 0.2, 0.3, 0.8];
        let noise = vec![0.1f32, 0.1, 0.1, 0.1];
        for allow_boundary in [false, true] {
            let ctx = MockWeightCtx {
                n_layers: 4,
                budget: 2,
                importance: Some(importance.clone()),
                noise: Some(noise.clone()),
                swapped: Vec::new(),
            };
            let stage = WeightSwapDeciderAsStage::new(WeightStageParams {
                allow_boundary_layers: allow_boundary,
            });
            let stage_layers: Vec<usize> = stage
                .plan(&ctx)
                .map(|p| p.per_layer.iter().map(|d| d.layer).collect())
                .unwrap_or_default();

            let currently_swapped: Vec<usize> = Vec::new();
            let decider = WeightSwapDecider {
                importance: Some(&importance),
                noise: Some(&noise),
                n_decoder_layers: 4,
                currently_swapped: &currently_swapped,
                allow_boundary_layers: allow_boundary,
                algorithm: SwapAlgorithm::ImportanceAware,
            };
            let direct = decider.decide(2);

            assert_eq!(
                stage_layers, direct.selected_layers,
                "allow_boundary={allow_boundary}: stage plan 의 layer 집합(순서 포함)이 동일 \
                 flag 의 decider.decide(budget) 와 bit-identical 이어야 한다"
            );
        }
    }

    /// budget=0 → None (no-op).
    #[test]
    fn stage_plan_zero_budget_is_none() {
        let ctx = MockWeightCtx {
            n_layers: 4,
            budget: 0,
            importance: Some(vec![0.1, 0.5, 0.3, 0.7]),
            noise: Some(vec![0.2, 0.1, 0.3, 0.05]),
            swapped: Vec::new(),
        };
        let stage = WeightSwapDeciderAsStage::new(WeightStageParams {
            allow_boundary_layers: false,
        });
        assert!(stage.plan(&ctx).is_none());
    }

    /// 이미 swap 완료(Q4_0)된 레이어는 currently_swapped 로 제외된다 — decider 와 동형.
    #[test]
    fn stage_plan_excludes_currently_swapped() {
        let importance = vec![0.1f32, 0.5, 0.3, 0.7];
        let noise = vec![0.2f32, 0.1, 0.3, 0.05];
        let ctx = MockWeightCtx {
            n_layers: 4,
            budget: 1,
            importance: Some(importance.clone()),
            noise: Some(noise.clone()),
            swapped: vec![2],
        };
        let stage = WeightSwapDeciderAsStage::new(WeightStageParams {
            allow_boundary_layers: false,
        });
        let plan = stage.plan(&ctx).expect("plan should be Some");
        let stage_layers: Vec<usize> = plan.per_layer.iter().map(|d| d.layer).collect();

        let currently_swapped = vec![2usize];
        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &currently_swapped,
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };
        let direct = decider.decide(1);

        assert_eq!(stage_layers, direct.selected_layers);
        assert!(
            !stage_layers.contains(&2),
            "swapped layer 2 must be excluded"
        );
    }

    /// "swap" 이 `WEIGHT_STAGES` 에 등록돼 있고 팩토리가 동작한다.
    #[test]
    fn swap_registered_in_slice() {
        let reg = find_weight_stage("swap").expect("'swap' 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "swap");
        let stage = (reg.make)(WeightStageParams {
            allow_boundary_layers: false,
        });
        assert_eq!(stage.name(), "swap");
    }
}
