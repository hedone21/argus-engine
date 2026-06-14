//! `WeightSwapDecider` — importance × ε bottom-k layer selection (ENG-ALG-215).
//!
//! Converts a Manager-supplied `ratio_max` into a concrete set of decoder layer
//! indices to swap from F16 to Q4_0.  Selection key is `importance_i × ε_i`
//! ascending: layers with the smallest product are swapped first (they are the
//! cheapest in quality terms).
//!
//! Layer 0 and the last decoder layer are always protected (ENG-ALG-215).
//! Layers already swapped (in `currently_swapped`) are excluded to keep the
//! operation idempotent.  Layers whose ε is NaN (INV-127) are excluded.
//!
//! When either `ImportanceTable` or `QuantNoiseTable` is absent / uncomputed,
//! the decider falls back to uniform index-spaced selection (ENG-ALG-213
//! fallback, absorbed by ENG-ALG-215).
//!
//! Spec: ENG-ALG-215, ENG-ALG-217, INV-127.

use std::collections::HashSet;

use crate::qcf_collector::ImportanceLookup;
use crate::qcf_types::SubLayer;

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

/// `ImportanceLookup` 를 layer 인덱스 기준의 평탄 `Vec<f32>` 로 투영한다 (MW-C).
///
/// 길이 `n` 의 벡터를 반환하며, 각 원소는 해당 layer 의 `SubLayer::Full`
/// importance 값이다. lookup 에 해당 entry 가 없으면 `0.0` 으로 채운다 (decider
/// ranking 의 기존 디폴트 `0.0` 보존). decider 의 flat `importance: Option<&[f32]>`
/// 필드에 넣기 위한 호출자 측 어댑터.
pub fn flatten_importance(lookup: &dyn ImportanceLookup, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for e in lookup.entries() {
        if e.sublayer == SubLayer::Full && e.layer_id < n {
            out[e.layer_id] = e.importance;
        }
    }
    out
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

/// Compute QCF_swap for a given set of swapped layers (ENG-ALG-217).
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
/// `importance`/`noise` 는 layer 인덱스 기준의 평탄 슬라이스다 (MW-C). `noise` 는
/// `&[f32]` (decide() 의 계약대로 ε 가 실제 계산된 경우의 슬라이스); NaN 원소는
/// numerator/denominator 양쪽에서 제외된다.
pub fn compute_qcf_weight_swap(
    swap_set: &[usize],
    noise: &[f32],
    importance: Option<&[f32]>,
    n_decoder_layers: usize,
) -> f32 {
    let _t = crate::qcf_timer!(QCF_WEIGHT_SWAP);
    compute_qcf_swap_internal(swap_set, n_decoder_layers, importance, Some(noise))
}

/// Internal implementation used by both the public function and the decider.
fn compute_qcf_swap_internal(
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ──
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

    // ── compute_qcf_weight_swap tests (ENG-ALG-217) ─────────────────────────────────

    /// Empty swap set → QCF_swap = 0.0.
    #[test]
    fn qcf_swap_empty_set_is_zero() {
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        let result = compute_qcf_weight_swap(&[], &noise, None, 4);
        assert_eq!(result, 0.0);
    }

    /// All-layers swap set (excluding NaN) → QCF_swap ≈ 1.0.
    #[test]
    fn qcf_swap_full_set_approx_one() {
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        // All layers in the "valid" set (no NaN)
        let result = compute_qcf_weight_swap(&[0, 1, 2, 3], &noise, None, 4);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "full set should give QCF_swap ≈ 1.0, got {result}"
        );
    }

    /// Monotonic property: adding a layer to the set must not decrease QCF_swap.
    #[test]
    fn qcf_swap_monotonic() {
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        let q1 = compute_qcf_weight_swap(&[1], &noise, None, 4);
        let q2 = compute_qcf_weight_swap(&[1, 2], &noise, None, 4);
        let q3 = compute_qcf_weight_swap(&[0, 1, 2], &noise, None, 4);
        assert!(q1 <= q2, "monotonic: q({{1}})={q1} <= q({{1,2}})={q2}");
        assert!(q2 <= q3, "monotonic: q({{1,2}})={q2} <= q({{0,1,2}})={q3}");
    }

    /// NaN ε layer contributes 0 to both numerator and denominator.
    #[test]
    fn qcf_swap_nan_layer_excluded_from_both() {
        let noise = make_noise(vec![0.2, f32::NAN, 0.3, 0.05]);
        // Layer 1 has NaN ε — should contribute 0 to numerator and denominator.
        // Including it in swap_set should not change result vs. excluding it.
        let without_nan = compute_qcf_weight_swap(&[0, 2, 3], &noise, None, 4);
        let with_nan = compute_qcf_weight_swap(&[0, 1, 2, 3], &noise, None, 4);
        assert!(
            (without_nan - with_nan).abs() < 1e-6,
            "NaN layer should not affect QCF_swap: without={without_nan}, with={with_nan}"
        );
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
}
