//! D2O (Dynamic Discriminative Operations) technique crate — H2O-style 3-partition eviction with
//! cosine-similarity token-merging compensation (Wan et al., 2024).
//!
//! Extracted from the engine core into a self-registering technique crate (the `caote`/`quest`/`h2o`
//! precedent): depends only on `argus-extension-api` + `linkme`, implements [`KVMutationStage`], and
//! registers under the name `"d2o"` via `register_kv_mutation_stage!`. The engine
//! force-links it (`use d2o as _;`) so `eviction plugin --name d2o` resolves the out-of-tree plugin.
//!
//! Algorithm (ported verbatim from the engine `compute_d2o_plan` + `D2OStage::plan`, proven
//! bit-identical to the former in-place handler by the engine's M4 equivalence tests):
//! 1. **3-partition** `[Protected Prefix] [Heavy Hitters (score-ranked)] [Recent Window]`; the
//!    remainder is marked for eviction.
//! 2. **Layer-wide nearest neighbor** (paper Eq.8): each evicted token is matched to its most
//!    similar retained token by cosine similarity on the head-concatenated K vector (single argmax,
//!    read via [`StageCtx::dequant_k`]).
//! 3. **EMA threshold** τ (paper Eq.10, max-based): only matches with `sim ≥ τ` merge; the rest are
//!    dropped. τ is interior-mutable state accumulated across plan calls (`Mutex<D2OState>`).
//! 4. **Eq.11 weighted merge**: evicted tokens scatter into their retained neighbor with weights
//!    `w_i = exp(u_i)/D`, `w_c = e/D`, `D = Σ exp(u_i) + e` (sum = 1, magnitude preserving). Emitted
//!    as [`WeightedMerge`]s; the engine executor (`apply_weighted_merges`) applies them on the buffer.
//!
//! Unlike streaming/h2o this stage produces merges, so it requires CPU-accessible KV buffers. On a
//! device-only buffer ([`StageCtx::kv_on_device`] is `true`) it degrades to a keep-only plan
//! (H2O-style score eviction), matching the engine's former GPU-only fallback. Per-layer protection
//! (`protected_layers`, and last-layer protection under `use_layer_allocation`) is honored via
//! [`StageCtx::layer_idx`]/[`StageCtx::n_layers`].

use argus_extension_api::{
    CacheHandle, CacheOpError, KVMutationStage, MergeAxis, MutationPhase, StageArgs, StageCaps,
    StageCtx, StageParams, TensorKind, WeightedMerge, register_kv_mutation_stage,
};
use std::sync::Mutex;

/// The merge-producing caps for the v3 registration: D2O ranks
/// by importance (Scores) and, on the off-device merge path, reads cached K (Key) for the Eq.8
/// matching; protects 4 sinks by default.
const D2O_CAPS: StageCaps = StageCaps {
    reads: &[TensorKind::Scores, TensorKind::Key],
    reads_signals: &[],
    default_protected_prefix: 4,
    produces_merge_plan: true,
    whole_model: false,
    prefill_attn_window: None,
};

// ── Configuration ────────────────────────────────────────────────

/// D2O configuration parameters.
pub struct D2OConfig {
    /// Fraction of available budget allocated to heavy hitters (0.0–1.0).
    /// D2O paper recommends N:M = 3:1, i.e. keep_ratio = 0.75.
    pub keep_ratio: f32,
    /// Number of prefix tokens to always protect (attention sinks).
    pub protected_prefix: usize,
    /// Target cache ratio: keep this fraction of current_pos after eviction.
    /// E.g. 0.5 = keep 50% of tokens. (The engine resolves ratio→`target_len`; this field is kept
    /// for parity with the former engine config and for callers that construct it directly.)
    pub target_ratio: f32,
    /// EMA smoothing factor β for the threshold update (paper Eq.10, default 0.7).
    /// τ_t = β · max U_t + (1−β) · τ_{t−1}.
    pub ema_beta: f32,
    /// Constant `e` in Eq.11 normalisation: D_j = Σ exp(u_ij) + e.
    /// Controls retained token's self-weight (w_c = e/D). Paper default 0.1.
    pub merge_e: f32,
    /// Enable per-layer dynamic budget allocation (Phase B). When `true`, the last layer is
    /// protected from eviction (matching the official D2O code intent). The variance-driven budget
    /// itself is engine-side (currently unwired); this flag's live effect is the last-layer guard.
    pub use_layer_allocation: bool,
    /// Layer indices to skip eviction entirely.
    pub protected_layers: Vec<usize>,
    /// Weighted-merge axis (WeightedKV roadmap item 2). `Both` (default) = weighted merge of both
    /// K and V (old behavior). `ValueOnly` = discard K + weighted-merge V only.
    pub merge_axis: MergeAxis,
}

impl Default for D2OConfig {
    fn default() -> Self {
        Self {
            keep_ratio: 0.75,
            protected_prefix: 4,
            target_ratio: 0.5,
            ema_beta: 0.7,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: MergeAxis::Both,
        }
    }
}

impl D2OConfig {
    /// Build the full d2o config from the shared [`StageParams`] (`keep_ratio`/`protected_prefix`)
    /// plus the technique-private [`StageArgs`] blob the engine routes opaquely. Unrecognized keys
    /// are ignored; a malformed value falls back to the field default. This is the single place that
    /// knows d2o's private knobs — the engine no longer constructs `D2OConfig` itself.
    ///
    /// Recognized keys: `target_ratio`, `ema_beta`, `merge_e` (f32); `layer_alloc` (`"true"`/else);
    /// `protected_layers` (comma-separated `usize`); `merge_axis` (`key_only`/`value_only`/else=both).
    pub fn from_args(base: StageParams, args: StageArgs<'_>) -> Self {
        let mut c = D2OConfig {
            keep_ratio: base.keep_ratio,
            protected_prefix: base.protected_prefix,
            ..D2OConfig::default()
        };
        for a in args {
            match a.key {
                "target_ratio" => {
                    if let Ok(v) = a.val.parse() {
                        c.target_ratio = v;
                    }
                }
                "ema_beta" => {
                    if let Ok(v) = a.val.parse() {
                        c.ema_beta = v;
                    }
                }
                "merge_e" => {
                    if let Ok(v) = a.val.parse() {
                        c.merge_e = v;
                    }
                }
                "layer_alloc" => c.use_layer_allocation = a.val == "true",
                "protected_layers" => {
                    c.protected_layers = a
                        .val
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
                "merge_axis" => {
                    c.merge_axis = match a.val {
                        "key_only" => MergeAxis::KeyOnly,
                        "value_only" => MergeAxis::ValueOnly,
                        _ => MergeAxis::Both,
                    };
                }
                _ => {}
            }
        }
        c
    }
}

// ── Mutable state ────────────────────────────────────────────────

/// Per-stage mutable state (wrapped in `Mutex` for interior mutability — `plan(&self, ...)`).
struct D2OState {
    /// EMA similarity threshold τ_t. Tokens merge only if similarity ≥ τ.
    ema_threshold: f32,
    /// Whether the EMA has been initialized (first eviction sets it).
    initialized: bool,
    /// Cumulative merge/delete statistics.
    total_merged: usize,
    total_deleted: usize,
}

impl D2OState {
    fn new() -> Self {
        Self {
            ema_threshold: 0.0,
            initialized: false,
            total_merged: 0,
            total_deleted: 0,
        }
    }
}

// ── Match ────────────────────────────────────────────────────────

/// Layer-wide nearest neighbor matching result for a single evicted token.
///
/// Per the D2O paper, the nearest retained token is determined on the concatenated K vector across
/// all KV heads (single argmax per evicted), not per-head independently. The same retained position
/// then receives merged contributions on every head and on V.
#[derive(Clone, Copy, Debug)]
struct Match {
    /// Position of the nearest retained token in the cache.
    retain_pos: usize,
    /// Layer-wide cosine similarity u_ij (single value, not per-head).
    sim: f32,
}

// ── Pure functions ───────────────────────────────────────────────

/// Cosine similarity between two slices (L2-normalized dot product; zero-norm → 0).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

/// Dequantize the layer-wide K vector at `pos` (concat of all KV heads) into `out`, reading each
/// head via `reader(pos, head, &mut out_head)`. `out` len = `kv_heads * head_dim`.
fn dequantize_k_layer_wide_via(
    reader: &dyn Fn(usize, usize, &mut [f32]),
    pos: usize,
    kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), kv_heads * head_dim);
    for h in 0..kv_heads {
        reader(pos, h, &mut out[h * head_dim..(h + 1) * head_dim]);
    }
}

/// Find the nearest retained token using **layer-wide K** (head-concatenated), reading K via
/// `reader`. Per D2O paper Eq.8: single argmax over cosine on the head-concat vector.
fn find_nearest_layer_wide_via(
    reader: &dyn Fn(usize, usize, &mut [f32]),
    evict_pos: usize,
    retain_set: &[usize],
    kv_heads: usize,
    head_dim: usize,
) -> Match {
    let layer_dim = kv_heads * head_dim;
    let mut evict_buf = vec![0.0f32; layer_dim];
    let mut retain_buf = vec![0.0f32; layer_dim];

    dequantize_k_layer_wide_via(reader, evict_pos, kv_heads, head_dim, &mut evict_buf);

    let mut best_pos = retain_set.first().copied().unwrap_or(evict_pos);
    let mut best_sim = f32::NEG_INFINITY;

    for &retain_pos in retain_set {
        if retain_pos == evict_pos {
            continue;
        }
        dequantize_k_layer_wide_via(reader, retain_pos, kv_heads, head_dim, &mut retain_buf);
        let sim = cosine_similarity(&evict_buf, &retain_buf);
        if sim > best_sim {
            best_sim = sim;
            best_pos = retain_pos;
        }
    }

    if best_sim == f32::NEG_INFINITY {
        // No valid retain target (e.g. retain_set is empty or only contains evict_pos)
        best_sim = 0.0;
    }

    Match {
        retain_pos: best_pos,
        sim: best_sim,
    }
}

/// Group passing evicted tokens by their nearest retained token (Eq.8 m_ij ⇒ groups).
/// Returns a map `retain_pos → Vec<(evict_pos, sim)>`.
fn group_by_retain(
    passing_positions: &[usize],
    matches: &[Match],
) -> std::collections::HashMap<usize, Vec<(usize, f32)>> {
    let mut groups: std::collections::HashMap<usize, Vec<(usize, f32)>> =
        std::collections::HashMap::new();
    for (i, &evict_pos) in passing_positions.iter().enumerate() {
        let m = matches[i];
        groups
            .entry(m.retain_pos)
            .or_default()
            .push((evict_pos, m.sim));
    }
    groups
}

/// Compute Eq.11 weights for one retained token's group.
///
/// Returns `(w_c, weights_per_evicted)` where `D = Σ exp(u_i) + e`, `w_c = e / D` (retained
/// self-weight), `w_i = exp(u_i) / D`. `u_i` is clamped to `[-10, 10]` before exp.
fn compute_eq11_weights(evicted_list: &[(usize, f32)], merge_e: f32) -> (f32, Vec<f32>) {
    let exps: Vec<f32> = evicted_list
        .iter()
        .map(|&(_, sim)| sim.clamp(-10.0, 10.0).exp())
        .collect();
    let sum_exp: f32 = exps.iter().sum();
    let denom = sum_exp + merge_e;
    let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
    let w_c = merge_e * inv_denom;
    let w_e: Vec<f32> = exps.iter().map(|e| e * inv_denom).collect();
    (w_c, w_e)
}

/// D2O evict **plan** computation — no buffer mutation. Returns `(retain_all keep, passing evicts,
/// matches)`. `reader(pos, head, &mut out)` reads K (via the ctx). `None` = no-op (current ≤ keep,
/// or nothing to evict). Ported verbatim from the engine `compute_d2o_plan` (Steps 1–4).
#[allow(clippy::too_many_arguments)]
fn compute_d2o_plan(
    reader: &dyn Fn(usize, usize, &mut [f32]),
    config: &D2OConfig,
    state: &mut D2OState,
    current_pos: usize,
    target_len: usize,
    importance: &[f32],
    kv_heads: usize,
    head_dim: usize,
    merge_enabled: bool,
) -> Option<(Vec<usize>, Vec<usize>, Vec<Match>)> {
    let current = current_pos;
    let prefix = config.protected_prefix.min(current);
    let keep = target_len.max(prefix + 2);
    if current <= keep {
        return None;
    }

    // ── Step 1: H2O-style 3-partition ──
    let available = keep.saturating_sub(prefix);
    let hh_budget = (available as f32 * config.keep_ratio) as usize;
    let recent_budget = available.saturating_sub(hh_budget);
    let recent_start = current.saturating_sub(recent_budget).max(prefix);
    let actual_recent = current - recent_start;
    let actual_hh_budget = available.saturating_sub(actual_recent);

    let mut token_scores: Vec<(usize, f32)> = (prefix..recent_start)
        .map(|pos| (pos, importance.get(pos).copied().unwrap_or(0.0)))
        .collect();
    token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let hh_positions: Vec<usize> = token_scores
        .iter()
        .take(actual_hh_budget)
        .map(|(pos, _)| *pos)
        .collect();
    let evict_positions: Vec<usize> = token_scores
        .iter()
        .skip(actual_hh_budget)
        .map(|(pos, _)| *pos)
        .collect();

    if evict_positions.is_empty() {
        return None;
    }

    let recent_positions: Vec<usize> = (recent_start..current).collect();
    let mut retain_all: Vec<usize> = (0..prefix)
        .chain(hh_positions.iter().copied())
        .chain(recent_positions.iter().copied())
        .collect();
    retain_all.sort();

    if !merge_enabled {
        // GPU-only buffers: skip merge, count all evicted as deleted.
        state.total_deleted += evict_positions.len();
        return Some((retain_all, Vec::new(), Vec::new()));
    }

    // ── Step 2: Layer-wide nearest neighbor (paper Eq.8 m_ij) — reader reads K ──
    let merge_targets: Vec<usize> = retain_all
        .iter()
        .copied()
        .filter(|&p| p >= prefix)
        .collect();
    let all_matches: Vec<Match> = evict_positions
        .iter()
        .map(|&pos| find_nearest_layer_wide_via(reader, pos, &merge_targets, kv_heads, head_dim))
        .collect();

    // ── Step 3: EMA threshold τ_t (paper Eq.10) ──
    if !all_matches.is_empty() {
        if !state.initialized {
            let mean_max =
                all_matches.iter().map(|m| m.sim).sum::<f32>() / all_matches.len() as f32;
            state.ema_threshold = mean_max;
            state.initialized = true;
        } else {
            let global_max = all_matches
                .iter()
                .map(|m| m.sim)
                .fold(f32::NEG_INFINITY, f32::max);
            state.ema_threshold =
                config.ema_beta * global_max + (1.0 - config.ema_beta) * state.ema_threshold;
        }
    }

    // ── Step 4: Filter — per-evicted max sim ≥ τ ──
    let passing_indices: Vec<usize> = (0..evict_positions.len())
        .filter(|&i| all_matches[i].sim >= state.ema_threshold)
        .collect();
    let passing_positions: Vec<usize> = passing_indices
        .iter()
        .map(|&i| evict_positions[i])
        .collect();
    let passing_matches: Vec<Match> = passing_indices.iter().map(|&i| all_matches[i]).collect();

    state.total_merged += passing_positions.len();
    state.total_deleted += evict_positions.len() - passing_positions.len();

    Some((retain_all, passing_positions, passing_matches))
}

// ── D2OStage ─────────────────────────────────────────────────────

/// D2O as an imperative [`KVMutationStage`]. `on_phase(ctx, cache)` runs [`compute_d2o_plan`] (K read
/// via `ctx.dequant_k`) and stages retain_all keep + Eq.11 [`WeightedMerge`]s onto the cache handle.
/// EMA τ is held in `Mutex<D2OState>` and accumulates across calls (per layer / per
/// decode step), matching the former engine handler's single shared state.
pub struct D2OStage {
    config: D2OConfig,
    state: Mutex<D2OState>,
}

impl D2OStage {
    /// Create with the given config. EMA state accumulates across `plan` calls.
    pub fn new(config: D2OConfig) -> Self {
        Self {
            config,
            state: Mutex::new(D2OState::new()),
        }
    }

    /// Whether this layer should be skipped entirely (no eviction). Mirrors the former engine
    /// `D2OHandler::is_protected`: explicit `protected_layers`, plus the last layer when
    /// `use_layer_allocation` is on (official D2O behavior).
    fn is_protected(&self, layer_idx: usize, n_layers: usize) -> bool {
        if self.config.protected_layers.contains(&layer_idx) {
            return true;
        }
        if self.config.use_layer_allocation && n_layers > 0 && layer_idx == n_layers - 1 {
            return true;
        }
        false
    }
}

impl D2OStage {
    /// The (keep, merges) decision (`None` = protected layer / no-op) driving the v3 `on_phase`.
    /// Merges are emitted only off-device (`!kv_on_device()`); a device-resident cache degrades to
    /// keep-only (empty merges).
    #[allow(clippy::type_complexity)]
    fn compute_plan(&self, ctx: &dyn StageCtx) -> Option<(Vec<usize>, Vec<WeightedMerge>)> {
        // Per-layer protection (formerly the handler's per-layer loop skip).
        if self.is_protected(ctx.layer_idx(), ctx.n_layers()) {
            return None;
        }

        let kv_heads = ctx.n_kv_heads();
        let head_dim = ctx.head_dim();
        let importance = ctx.importance().unwrap_or(&[]);
        // Device-only KV buffers cannot be CPU-read/merged → degrade to keep-only (H2O-style),
        // matching the former engine GPU-only fallback (merge compensation disabled).
        let merge_enabled = !ctx.kv_on_device();
        let mut state = self.state.lock().unwrap();

        let (retain_all, passing, matches) = compute_d2o_plan(
            &|p, h, o| ctx.dequant_k(p, h, o),
            &self.config,
            &mut state,
            ctx.current_pos(),
            ctx.target_len(),
            importance,
            kv_heads,
            head_dim,
            merge_enabled,
        )?;

        // passing+matches → per-group Eq.11 weighted WeightedMerge (same grouping the in-place
        // scatter_reduce used).
        let merges: Vec<WeightedMerge> = if passing.is_empty() {
            Vec::new()
        } else {
            group_by_retain(&passing, &matches)
                .iter()
                .map(|(retain, evicted_list)| {
                    let (w_c, w_e) = compute_eq11_weights(evicted_list, self.config.merge_e);
                    WeightedMerge {
                        into: *retain,
                        into_weight: w_c,
                        from: evicted_list
                            .iter()
                            .zip(w_e.iter())
                            .map(|(&(ep, _), &w)| (ep, w))
                            .collect(),
                        apply_to: self.config.merge_axis,
                    }
                })
                .collect()
        };

        Some((retain_all, merges))
    }
}

// ── v3 native (imperative) surface — the production path ──

impl KVMutationStage for D2OStage {
    fn name(&self) -> &str {
        "d2o"
    }

    /// Stage the Eq.11 weighted merges (off-device only) then the retain-all keep, or no-op on a
    /// protected layer. On a device-resident cache `compute_plan` yields no merges, so this stages
    /// keep-only WITHOUT aborting (T-8).
    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        match self.compute_plan(ctx) {
            None => Ok(()),
            Some((keep, merges)) => {
                if !merges.is_empty() {
                    cache.merge(&merges)?;
                }
                cache.keep(&keep)
            }
        }
    }
}

register_kv_mutation_stage!(
    "d2o",
    |p, args| Box::new(D2OStage::new(D2OConfig::from_args(p, args))),
    D2O_CAPS,
    MutationPhase::KvMutate
);

// ── QCF estimator (observer/score axis) ──────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorDtype, TensorHandle, TensorKind, TensorShape};

    // ── cosine_similarity ──

    #[test]
    fn cosine_identical_orthogonal_opposite_zero() {
        assert!((cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0); // zero-norm → 0
    }

    // ── from_args (engine routes an opaque blob; the plugin parses its own knobs) ──

    #[test]
    fn from_args_empty_blob_uses_base_and_defaults() {
        let base = StageParams {
            keep_ratio: 0.8,
            protected_prefix: 7,
            ..StageParams::default()
        };
        let c = D2OConfig::from_args(base, &[]);
        // base supplies keep_ratio/protected_prefix; everything else is the D2OConfig default.
        assert_eq!(c.keep_ratio, 0.8);
        assert_eq!(c.protected_prefix, 7);
        assert_eq!(c.target_ratio, 0.5);
        assert_eq!(c.ema_beta, 0.7);
        assert_eq!(c.merge_e, 0.1);
        assert!(!c.use_layer_allocation);
        assert!(c.protected_layers.is_empty());
        assert_eq!(c.merge_axis, MergeAxis::Both);
    }

    #[test]
    fn from_args_parses_all_keys_ignores_unknown_and_malformed() {
        use argus_extension_api::PluginArg;
        let base = StageParams {
            keep_ratio: 0.8,
            protected_prefix: 7,
            ..StageParams::default()
        };
        let args = [
            PluginArg {
                key: "target_ratio",
                val: "0.3",
            },
            PluginArg {
                key: "ema_beta",
                val: "0.9",
            },
            PluginArg {
                key: "merge_e",
                val: "0.25",
            },
            PluginArg {
                key: "layer_alloc",
                val: "true",
            },
            PluginArg {
                key: "protected_layers",
                val: "0, 1, 27",
            }, // whitespace tolerated
            PluginArg {
                key: "merge_axis",
                val: "value_only",
            },
            PluginArg {
                key: "unknown_key",
                val: "ignored",
            }, // unknown → ignored
            PluginArg {
                key: "merge_e",
                val: "not_a_float",
            }, // malformed → keeps prior 0.25
        ];
        let c = D2OConfig::from_args(base, &args);
        assert_eq!(c.keep_ratio, 0.8); // still from base
        assert_eq!(c.protected_prefix, 7);
        assert_eq!(c.target_ratio, 0.3);
        assert_eq!(c.ema_beta, 0.9);
        assert_eq!(c.merge_e, 0.25); // malformed second merge_e left it at 0.25
        assert!(c.use_layer_allocation);
        assert_eq!(c.protected_layers, vec![0, 1, 27]);
        assert_eq!(c.merge_axis, MergeAxis::ValueOnly);
    }

    // ── compute_eq11_weights (D = Σ exp(u_i) + e; w_c = e/D; w_i = exp(u_i)/D) ──

    #[test]
    fn eq11_weights_sum_to_one() {
        let (w_c, w_e) = compute_eq11_weights(&[(7, 0.5), (9, -0.2)], 0.1);
        let total: f32 = w_c + w_e.iter().sum::<f32>();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "weights must sum to 1, got {total}"
        );
        assert_eq!(w_e.len(), 2);
        // higher sim → larger weight.
        assert!(w_e[0] > w_e[1]);
    }

    #[test]
    fn eq11_handles_extreme_sim_without_overflow() {
        // sim is clamped to [-10, 10] before exp; very large sim must not produce NaN/Inf.
        let (w_c, w_e) = compute_eq11_weights(&[(1, 1e9), (2, -1e9)], 0.1);
        let total: f32 = w_c + w_e.iter().sum::<f32>();
        assert!(total.is_finite() && (total - 1.0).abs() < 1e-4);
    }

    // ── find_nearest_layer_wide_via (single argmax over head-concat cosine) ──

    #[test]
    fn find_nearest_picks_most_similar_retained() {
        // head_dim=2, kv_heads=1. evict=0 ([1,0]); retained {1:[0,1] sim0, 2:[1,0] sim1, 3:[-1,0] sim-1}.
        let reader = |pos: usize, _head: usize, out: &mut [f32]| {
            let v: [f32; 2] = match pos {
                0 => [1.0, 0.0],
                1 => [0.0, 1.0],
                2 => [1.0, 0.0],
                3 => [-1.0, 0.0],
                _ => [0.0, 0.0],
            };
            out[..2].copy_from_slice(&v);
        };
        let m = find_nearest_layer_wide_via(&reader, 0, &[1, 2, 3], 1, 2);
        assert_eq!(m.retain_pos, 2, "nearest is the identical retained token");
        assert!((m.sim - 1.0).abs() < 1e-6);
    }

    // ── compute_d2o_plan: partition / EMA / filter ──

    fn cfg(keep_ratio: f32, protected_prefix: usize) -> D2OConfig {
        D2OConfig {
            keep_ratio,
            protected_prefix,
            ..D2OConfig::default()
        }
    }

    #[test]
    fn plan_noop_within_budget() {
        let mut st = D2OState::new();
        let r = |_: usize, _: usize, _: &mut [f32]| {};
        // current(10) <= keep(max(12, prefix+2)) → None.
        assert!(compute_d2o_plan(&r, &cfg(0.75, 4), &mut st, 10, 12, &[], 1, 4, false).is_none());
    }

    #[test]
    fn plan_partition_keep_list_3_to_1() {
        // current=20, prefix=4, target=12, keep_ratio=0.75 → keep=12, available=8, hh_budget=6,
        // recent_budget=2, recent_start=18. HH = top-6 by importance over [4..18).
        let mut imp = vec![0.0f32; 20];
        for (rank, &p) in [5usize, 7, 9, 11, 13, 15].iter().enumerate() {
            imp[p] = 10.0 - rank as f32; // distinct, highest in [4..18)
        }
        let mut st = D2OState::new();
        let r = |_: usize, _: usize, _: &mut [f32]| {};
        // merge_enabled=false → keep-only (no K reads needed).
        let (retain, passing, matches) =
            compute_d2o_plan(&r, &cfg(0.75, 4), &mut st, 20, 12, &imp, 1, 4, false).unwrap();
        assert_eq!(retain, vec![0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 18, 19]);
        assert!(
            passing.is_empty() && matches.is_empty(),
            "merge disabled → keep-only"
        );
        assert_eq!(st.total_deleted, 8);
    }

    #[test]
    fn plan_ema_init_mean_then_update_global_max_and_filter() {
        // current=6, prefix=1, target=3 → keep=3, available=2, hh_budget(0.5)=1, recent_budget=1,
        // recent_start=5. importance puts HH at pos 1 → evict {2,3,4}, retained merge_targets {1,5}.
        let mut imp = vec![0.0f32; 6];
        imp[1] = 10.0;
        let mut st = D2OState::new();

        // Call 1: every position's K identical → all match sims = 1.0 → τ init = mean = 1.0, all pass.
        let identical = |_: usize, _: usize, out: &mut [f32]| out[..2].copy_from_slice(&[1.0, 0.0]);
        let (_retain, passing1, _m1) =
            compute_d2o_plan(&identical, &cfg(0.5, 1), &mut st, 6, 3, &imp, 1, 2, true).unwrap();
        assert!(st.initialized);
        assert!(
            (st.ema_threshold - 1.0).abs() < 1e-6,
            "init τ = mean sim = 1.0"
        );
        assert_eq!(passing1.len(), 3, "all evicted pass (sim 1.0 ≥ τ 1.0)");

        // Call 2: evicted ⟂ retained → match sims = 0 → τ = 0.7·0 + 0.3·1.0 = 0.3, none pass.
        let ortho = |pos: usize, _: usize, out: &mut [f32]| {
            let v = if (2..=4).contains(&pos) {
                [0.0, 1.0]
            } else {
                [1.0, 0.0]
            };
            out[..2].copy_from_slice(&v);
        };
        let (_r2, passing2, _m2) =
            compute_d2o_plan(&ortho, &cfg(0.5, 1), &mut st, 6, 3, &imp, 1, 2, true).unwrap();
        assert!(
            (st.ema_threshold - 0.3).abs() < 1e-6,
            "update τ = β·max + (1-β)·prev = 0.3"
        );
        assert!(passing2.is_empty(), "sim 0 < τ 0.3 → nothing merges");
    }

    // ── D2OStage / on_phase (kv_on_device / merge structure / registration) ──

    /// Minimal K-providing tensor handle: `k[(row*n_kv_heads + kv_head)*head_dim + d]`.
    struct KeyData {
        k: Vec<f32>,
        n_kv_heads: usize,
        head_dim: usize,
    }
    impl TensorHandle for KeyData {
        fn shape(&self) -> TensorShape {
            TensorShape {
                rows: self.k.len() / (self.n_kv_heads * self.head_dim),
                cols: self.head_dim,
                per_head: true,
            }
        }
        fn dtype(&self) -> TensorDtype {
            TensorDtype::F32
        }
        fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
            let base = (row * self.n_kv_heads + kv_head) * self.head_dim;
            out[..self.head_dim].copy_from_slice(&self.k[base..base + self.head_dim]);
        }
    }

    struct Ctx {
        current_pos: usize,
        target_len: usize,
        importance: Option<Vec<f32>>,
        layer_idx: usize,
        n_layers: usize,
        on_device: bool,
        n_kv_heads: usize,
        head_dim: usize,
        key: Option<KeyData>,
    }
    impl StageCtx for Ctx {
        fn current_pos(&self) -> usize {
            self.current_pos
        }
        fn target_len(&self) -> usize {
            self.target_len
        }
        fn layer_idx(&self) -> usize {
            self.layer_idx
        }
        fn n_layers(&self) -> usize {
            self.n_layers
        }
        fn kv_on_device(&self) -> bool {
            self.on_device
        }
        fn importance(&self) -> Option<&[f32]> {
            self.importance.as_deref()
        }
        fn n_kv_heads(&self) -> usize {
            self.n_kv_heads
        }
        fn head_dim(&self) -> usize {
            self.head_dim
        }
        fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
            match kind {
                TensorKind::Key => self.key.as_ref().map(|k| k as &dyn TensorHandle),
                _ => None,
            }
        }
    }

    fn base_ctx() -> Ctx {
        Ctx {
            current_pos: 20,
            target_len: 12,
            importance: Some(vec![1.0f32; 20]),
            layer_idx: 0,
            n_layers: 4,
            on_device: false,
            n_kv_heads: 1,
            head_dim: 2,
            // distinct K per position so cosine-nearest is well-defined.
            key: Some(KeyData {
                k: (0..20).flat_map(|p| [1.0, p as f32]).collect(),
                n_kv_heads: 1,
                head_dim: 2,
            }),
        }
    }

    /// A mock [`CacheHandle`] capturing the staged merges + keep.
    #[derive(Default)]
    struct CaptureHandle {
        merged: Option<Vec<WeightedMerge>>,
        kept: Option<Vec<usize>>,
    }
    impl CacheHandle for CaptureHandle {
        fn current_pos(&self) -> usize {
            20
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            2
        }
        fn kv_on_device(&self) -> bool {
            false
        }
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
        fn keep(&mut self, keep: &[usize]) -> Result<(), CacheOpError> {
            self.kept = Some(keep.to_vec());
            Ok(())
        }
        fn keep_per_head(&mut self, _keep: &[&[usize]]) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn merge(&mut self, merges: &[WeightedMerge]) -> Result<(), CacheOpError> {
            self.merged = Some(merges.to_vec());
            Ok(())
        }
        fn reencode(&mut self, _target: argus_extension_api::FormatId) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn transition_quant_bits(&mut self, _bits: u8) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn offload(&mut self, _prefix_len: usize) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn recall(&mut self) -> Result<(), CacheOpError> {
            Ok(())
        }
    }

    /// v3 native registration + DECISION coverage: the v3 `on_phase` stages exactly the merges + keep
    /// the shared `compute_plan` decides, and on a device-resident cache it stages keep-only without a
    /// merge (T-8). Fresh stage instances so the EMA state doesn't carry.
    #[test]
    fn v3_native_stages_shared_decision() {
        use argus_extension_api::find_mutation_stage;
        let reg = find_mutation_stage("d2o").expect("d2o in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "d2o");
        assert_eq!(reg.caps, D2O_CAPS);
        assert!(reg.caps.produces_merge_plan);
        assert_eq!((reg.make)(StageParams::default(), &[]).name(), "d2o");

        // host path: merges + keep (the shared decision the v3 on_phase stages).
        let (expected_keep, expected_merges) = D2OStage::new(D2OConfig::default())
            .compute_plan(&base_ctx())
            .expect("compute_plan Some");
        assert!(
            !expected_merges.is_empty(),
            "host base_ctx must produce merges"
        );
        let mut h = CaptureHandle::default();
        <D2OStage as KVMutationStage>::on_phase(
            &D2OStage::new(D2OConfig::default()),
            &base_ctx(),
            &mut h,
        )
        .unwrap();
        // The merge GROUPING is HashMap-backed, so the emitted Vec order is not deterministic across
        // two calls (group_by_retain). Compare order-independently by sorting on the `into` target —
        // both calls produce the SAME set of weighted merges, just possibly in a different order.
        let sort_by_into = |mut m: Vec<WeightedMerge>| {
            m.sort_by_key(|w| w.into);
            m
        };
        assert_eq!(
            h.merged.map(sort_by_into),
            Some(sort_by_into(expected_merges)),
            "v3 stages the same set of merges as the shared decision"
        );
        assert_eq!(h.kept, Some(expected_keep));

        // device path: keep-only, no merge call.
        let mut dctx = base_ctx();
        dctx.on_device = true;
        let mut h2 = CaptureHandle::default();
        <D2OStage as KVMutationStage>::on_phase(
            &D2OStage::new(D2OConfig::default()),
            &dctx,
            &mut h2,
        )
        .unwrap();
        assert_eq!(h2.merged, None, "device → no merge staged (T-8 degrade)");
        assert!(h2.kept.is_some(), "device still stages the keep");
    }

    #[test]
    fn stage_produces_normalized_weighted_merges() {
        let stage = D2OStage::new(D2OConfig::default());
        let (keep, merges) = stage.compute_plan(&base_ctx()).expect("compute_plan Some");
        assert!(!merges.is_empty(), "merge enabled → some merges expected");
        for m in &merges {
            assert!(
                keep.contains(&m.into),
                "merge target must be a retained token"
            );
            for &(from, _) in &m.from {
                assert!(
                    !keep.contains(&from),
                    "merged-from must be an evicted token"
                );
            }
            let total: f32 = m.into_weight + m.from.iter().map(|(_, w)| *w).sum::<f32>();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "Eq.11 weights sum to 1, got {total}"
            );
        }
    }
}
