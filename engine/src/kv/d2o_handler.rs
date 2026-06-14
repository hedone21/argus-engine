//! D2O (Dynamic Discriminative Operations) handler — eviction with compensation merging.
//!
//! Implements the D2O paper (Wan et al., 2024): H2O-style 3-partition eviction
//! with token merging compensation. Evicted tokens are matched to their nearest
//! retained neighbor by **layer-wide K cosine similarity** (head-concatenated);
//! merge weights follow paper Eq.11 (softmax-style with `e` constant) so that
//! retained + merged contributions sum to 1.
//!
//! Phase 1: Layer-wide nearest + Eq.11 merge weight + `e` hyperparameter.
//! Phase 2 (deferred): EMA threshold (max-based, single β).
//! Phase B (deferred): Layer-level dynamic allocation.
//! Supports F32, F16, and Q4_0 KV cache dtypes.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use crate::buffer::DType;
use crate::kv::kv_cache::{KVCache, max_cache_pos};
use crate::quant::{BlockQ4_0, QK4_0};
use anyhow::Result;
use half::f16;
use std::sync::Mutex;
use technique_api::{KVCachePlan, KVCacheStage, KeepSpec, StageCtx, WeightedMerge};

// ── Configuration ────────────────────────────────────────────────

/// D2O configuration parameters.
pub struct D2OConfig {
    /// Fraction of available budget allocated to heavy hitters (0.0–1.0).
    /// D2O paper recommends N:M = 3:1, i.e. keep_ratio = 0.75.
    pub keep_ratio: f32,
    /// Number of prefix tokens to always protect (attention sinks).
    pub protected_prefix: usize,
    /// Target cache ratio: keep this fraction of current_pos after eviction.
    /// E.g. 0.5 = keep 50% of tokens.
    pub target_ratio: f32,
    /// EMA smoothing factor β for the threshold update (paper Eq.10, default 0.7).
    /// τ_t = β · max U_t + (1−β) · τ_{t−1}.
    pub ema_beta: f32,
    /// Constant `e` in Eq.11 normalisation: D_j = Σ exp(u_ij) + e.
    /// Controls retained token's self-weight (w_c = e/D). Paper default 0.1.
    pub merge_e: f32,
    /// Enable per-layer dynamic budget allocation (Phase B).
    pub use_layer_allocation: bool,
    /// Layer indices to skip eviction entirely.
    pub protected_layers: Vec<usize>,
    /// Weighted-merge axis (WeightedKV, KV 로드맵 항목 2). `Both`(기본) = K·V 양쪽 가중 merge(구 동작).
    /// `ValueOnly` = K discard + V 만 가중 merge. ablation 측정용 런타임 선택.
    pub merge_axis: technique_api::MergeAxis,
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
            merge_axis: technique_api::MergeAxis::Both,
        }
    }
}

// ── Mutable state ────────────────────────────────────────────────

/// Per-invocation mutable state (wrapped in Mutex for interior mutability).
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
/// Per the D2O paper, the nearest retained token is determined on the
/// concatenated K vector across all KV heads (single argmax per evicted),
/// not per-head independently. The same retained position then receives
/// merged contributions on every head and on V.
#[derive(Clone, Copy, Debug)]
struct Match {
    /// Position of the nearest retained token in the cache.
    retain_pos: usize,
    /// Layer-wide cosine similarity u_ij (single value, not per-head).
    sim: f32,
}

// ── D2OHandler ───────────────────────────────────────────────────

/// D2O cache pressure handler: H2O eviction + cosine similarity merge.
pub struct D2OHandler {
    config: D2OConfig,
    state: Mutex<D2OState>,
}

impl D2OHandler {
    pub fn new(config: D2OConfig) -> Self {
        Self {
            config,
            state: Mutex::new(D2OState::new()),
        }
    }

    /// Check if a layer should be protected from eviction.
    fn is_protected(&self, layer_idx: usize, num_layers: usize) -> bool {
        // Protected layers from config
        if self.config.protected_layers.contains(&layer_idx) {
            return true;
        }
        // When layer allocation is active, always protect the last layer
        // (matching official D2O code behavior)
        if self.config.use_layer_allocation && layer_idx == num_layers - 1 {
            return true;
        }
        false
    }

    /// Create with default configuration.
    pub fn with_defaults(target_ratio: f32) -> Self {
        Self::new(D2OConfig {
            target_ratio: target_ratio.clamp(0.1, 0.99),
            ..D2OConfig::default()
        })
    }

    /// Process a single layer's cache: partition → similarity → merge/delete → compact.
    ///
    /// When `merge_enabled` is false, the similarity matching and scatter-reduce
    /// merge steps are skipped, reducing behavior to H2O-style score-based eviction.
    /// This is used when KV buffers are GPU-only (discrete GPU, no zero-copy)
    /// and cannot be safely accessed from the CPU.
    fn evict_and_merge(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: &[f32],
        state: &mut D2OState,
        merge_enabled: bool,
    ) -> Result<usize> {
        let current = cache.current_pos;
        let kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();

        // Step 1~5 (mutation 없는 계획 산출)을 D2OStage 와 공유하는 compute_d2o_plan 으로 위임.
        // reader = dequantize_k(cache) — D2OStage 는 동일 함수에 ctx.dequant_k reader 를 넘긴다.
        let plan = {
            let c: &KVCache = cache;
            compute_d2o_plan(
                &|p, h, o| dequantize_k(c, p, h, head_dim, o),
                &self.config,
                state,
                current,
                target_len,
                importance,
                kv_heads,
                head_dim,
                merge_enabled,
            )
        };
        let (retain_all, passing_positions, passing_matches) = match plan {
            Some(p) => p,
            None => return Ok(0),
        };

        // ── Step 5(apply): Scatter-reduce merge (Eq.11 weights, layer-wide nearest) ──
        if !passing_positions.is_empty() {
            scatter_reduce_merge_layer_wide(
                cache,
                &passing_positions,
                &passing_matches,
                kv_heads,
                head_dim,
                self.config.merge_e,
            );
        }

        // ── Step 6: Compact cache ──
        cache.compact_keep_positions(&retain_all, 0)?;
        cache.current_pos = retain_all.len();

        Ok(current - cache.current_pos)
    }
}

/// (M4-c) d2o evict **계획** 계산 — buffer mutation 없이 `(retain_all keep, passing evicts, matches)`
/// 를 산출한다. `reader(pos, head, &mut out)` = K 읽기(cache 또는 ctx). D2OHandler::evict_and_merge
/// (cache reader)와 D2OStage::plan(ctx reader)가 **이 함수를 공유**해 동일 계획을 낸다 — bit-identity
/// by construction. `None` = no-op(current ≤ keep 또는 evict 대상 없음). step1~4 는 기존
/// evict_and_merge 와 byte-identical(검증: 기존 d2o 테스트 + D2OStage 동등성 테스트).
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

    // ── Step 2: Layer-wide nearest neighbor (paper Eq.8 m_ij) — reader 로 K 읽기 ──
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

impl CachePressureHandler for D2OHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
        if ctx.caches.is_empty() {
            return Ok(ActionResult::NoOp);
        }

        // Validate layout consistency
        debug_assert!(
            ctx.caches
                .windows(2)
                .all(|w| w[0].layout() == w[1].layout()),
            "D2OHandler: all caches must use the same KVLayout"
        );

        let importance = match ctx.importance {
            Some(imp) => imp,
            None => return Ok(ActionResult::NoOp),
        };

        let num_layers = ctx.caches.len();
        let current_pos = max_cache_pos(ctx.caches);
        let effective_ratio = ctx.target_ratio.unwrap_or(self.config.target_ratio);

        // Default uniform target_len (used when layer_ratios is None)
        let uniform_target = ((current_pos as f32) * effective_ratio) as usize;
        let uniform_target = uniform_target.max(1);

        if current_pos <= uniform_target && ctx.layer_ratios.is_none() {
            return Ok(ActionResult::NoOp);
        }

        // Check if KV buffers are CPU-accessible (zero-copy or CPU backend).
        // Discrete GPUs (e.g. NVIDIA) allocate device-only buffers where as_ptr()
        // returns null — CPU access would SIGSEGV. In that case, skip the merge
        // step and fall back to score-based eviction only (same behavior as H2O).
        let merge_enabled = !ctx.caches[0].k_buffer.as_ptr().is_null();
        if !merge_enabled {
            log::warn!(
                "[D2O] KV buffers are GPU-only (as_ptr is null); \
                 merge compensation disabled, falling back to eviction-only mode"
            );
        }

        let mut state = self.state.lock().unwrap();
        let mut total_removed = 0;
        let mut min_new_pos = usize::MAX;

        for (layer_idx, cache) in ctx.caches.iter_mut().enumerate() {
            // Layer protection
            if self.is_protected(layer_idx, num_layers) {
                min_new_pos = min_new_pos.min(cache.current_pos);
                continue;
            }

            // Per-layer target from layer_ratios or uniform fallback
            let layer_target = if let Some(ratios) = ctx.layer_ratios {
                if layer_idx < ratios.len() {
                    let (hh_r, rec_r) = ratios[layer_idx];
                    let ratio = (hh_r + rec_r).clamp(0.01, 1.0);
                    ((cache.current_pos as f32 * ratio) as usize).max(1)
                } else {
                    uniform_target
                }
            } else {
                uniform_target
            };

            if cache.current_pos <= layer_target {
                min_new_pos = min_new_pos.min(cache.current_pos);
                continue;
            }

            let removed =
                self.evict_and_merge(cache, layer_target, importance, &mut state, merge_enabled)?;
            total_removed += removed;
            min_new_pos = min_new_pos.min(cache.current_pos);
        }

        if total_removed > 0 {
            let active_layers = (0..num_layers)
                .filter(|&l| !self.is_protected(l, num_layers))
                .count()
                .max(1);
            log::info!(
                "[D2O] Evicted {} tokens across {} layers (merged={}, deleted={}), new_pos={}",
                total_removed,
                active_layers,
                state.total_merged,
                state.total_deleted,
                min_new_pos,
            );
            Ok(ActionResult::Evicted {
                tokens_removed: total_removed / active_layers,
                new_pos: min_new_pos,
            })
        } else {
            Ok(ActionResult::NoOp)
        }
    }

    fn name(&self) -> &str {
        "d2o"
    }
}

// ── D2OStage: d2o 를 KVCacheStage(plan-returning) 표면으로 노출 (M4-c) ──────

/// d2o 를 [`KVCacheStage`] 로 재구현한 표면. `plan(ctx)` 가 [`compute_d2o_plan`](D2OHandler 와
/// 공유)으로 retain_all keep + 가중 [`WeightedMerge`](Eq.11)를 산출하고, EMA τ 는 impl `Mutex`
/// 가 보유한다. 버퍼 변형은 엔진 executor(`apply_weighted_merges`+compact)가 실행.
///
/// **동등성**: 동일 config·cache·call 시퀀스(공유 EMA)에서 `D2OHandler::evict_and_merge` 와
/// bit-identical — `compute_d2o_plan` 공유 + `apply_weighted_merges`≡`scatter_reduce`(M4-b 증명).
/// 단 **non-layer-alloc config 한정**(StageBackedPolicy/run_policy_eviction 경로는 per-cache uniform
/// target 만 주므로 layer-alloc/protected-layer 는 미지원 — production d2o 는 if-branch=D2OHandler
/// 유지, M4-c 구조적 결정). raw K 는 `ctx.dequant_k`(= `dequantize_k` 위임)로 읽는다.
pub struct D2OStage {
    config: D2OConfig,
    state: Mutex<D2OState>,
}

impl D2OStage {
    /// 주어진 config 로 생성. EMA 상태는 호출 간 누적(impl Mutex).
    pub fn new(config: D2OConfig) -> Self {
        Self {
            config,
            state: Mutex::new(D2OState::new()),
        }
    }
}

impl KVCacheStage for D2OStage {
    fn name(&self) -> &str {
        "d2o"
    }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let kv_heads = ctx.n_kv_heads();
        let head_dim = ctx.head_dim();
        let importance = ctx.importance().unwrap_or(&[]);
        let mut state = self.state.lock().unwrap();

        // D2OHandler::evict_and_merge 와 공유하는 계획 계산 — K 는 ctx.dequant_k reader.
        // merge_enabled=true: ctx 가 dequant 가능한 CPU-accessible 경로(non-alloc). compute_d2o_plan
        // None → no-op → plan None.
        let (retain_all, passing, matches) = compute_d2o_plan(
            &|p, h, o| ctx.dequant_k(p, h, o),
            &self.config,
            &mut state,
            ctx.current_pos(),
            ctx.target_len(),
            importance,
            kv_heads,
            head_dim,
            true,
        )?;

        // passing+matches → group 별 Eq.11 가중 WeightedMerge (scatter_reduce 내부 grouping 과 동일).
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

        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(retain_all),
            merges,
        })
    }
}

// ── Pure functions ───────────────────────────────────────────────

/// Cosine similarity between two slices.
///
/// pub(crate): R-KV 측정 프로토타입(`rkv_stage.rs`, feature `rkv`)이 N×N pairwise
/// redundancy row-mean 의 빌딩블록으로 재사용한다(arch/kv_roadmap_item0_measurement.md §4.1).
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// Dequantize a K vector at (pos, head) into the output buffer.
/// Works for F32, F16, and Q4_0 dtypes.
///
/// pub(crate): `StageCtx::dequant_k`(stage_registry.rs `KVStageCtx`)가 이 정본을 위임 재사용해
/// d2o-stage(M4-c)의 raw-K 읽기를 D2OHandler 와 bit-identical 하게 한다.
pub(crate) fn dequantize_k(
    cache: &KVCache,
    pos: usize,
    head: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    match cache.k_buffer.dtype() {
        DType::F32 => {
            let k = cache.k_buffer.as_slice::<f32>();
            let off = cache.offset(pos, head);
            out[..head_dim].copy_from_slice(&k[off..off + head_dim]);
        }
        DType::F16 => {
            let k = cache.k_buffer.as_slice::<f16>();
            let off = cache.offset(pos, head);
            for d in 0..head_dim {
                out[d] = k[off + d].to_f32();
            }
        }
        DType::Q4_0 => {
            let k = cache.k_buffer.as_slice::<BlockQ4_0>();
            let blocks_per_pos = head_dim / QK4_0;
            let block_off = cache.q4_block_offset(pos, head, blocks_per_pos);
            for bi in 0..blocks_per_pos {
                let mut tmp = [0.0f32; QK4_0];
                k[block_off + bi].dequantize(&mut tmp);
                let dst = bi * QK4_0;
                out[dst..dst + QK4_0].copy_from_slice(&tmp);
            }
        }
        _ => {}
    }
}

/// Dequantize a V vector at (pos, head) into the output buffer.
/// Works for F32, F16, and Q4_0 dtypes — exact mirror of [`dequantize_k`] on `v_buffer`.
///
/// V uses the IDENTICAL `[1, kv_heads, capacity, head_dim]` layout and `offset`/`q4_block_offset`
/// as K (confirmed by `apply_weighted_merges` which dispatches K/V independently over the same
/// offsets), so this is byte-faithful to K dequant. pub(crate): `StageCtx::tensor(Value)`
/// (stage_registry.rs `KVStageCtx`)가 위임 재사용 — CAOTE 의 v_i 읽기.
pub(crate) fn dequantize_v(
    cache: &KVCache,
    pos: usize,
    head: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    match cache.v_buffer.dtype() {
        DType::F32 => {
            let v = cache.v_buffer.as_slice::<f32>();
            let off = cache.offset(pos, head);
            out[..head_dim].copy_from_slice(&v[off..off + head_dim]);
        }
        DType::F16 => {
            let v = cache.v_buffer.as_slice::<f16>();
            let off = cache.offset(pos, head);
            for d in 0..head_dim {
                out[d] = v[off + d].to_f32();
            }
        }
        DType::Q4_0 => {
            let v = cache.v_buffer.as_slice::<BlockQ4_0>();
            let blocks_per_pos = head_dim / QK4_0;
            let block_off = cache.q4_block_offset(pos, head, blocks_per_pos);
            for bi in 0..blocks_per_pos {
                let mut tmp = [0.0f32; QK4_0];
                v[block_off + bi].dequantize(&mut tmp);
                let dst = bi * QK4_0;
                out[dst..dst + QK4_0].copy_from_slice(&tmp);
            }
        }
        _ => {}
    }
}

/// Dequantize the layer-wide K vector at `pos` (concat of all KV heads) into `out`,
/// reading each head via `reader(pos, head, &mut out_head)`. `out` len = `kv_heads * head_dim`.
///
/// (M4-c) reader 추상 — D2OHandler 는 `dequantize_k`(cache) reader, D2OStage 는 `StageCtx::dequant_k`
/// (ctx) reader 를 넘긴다. 둘 다 동일 dequant 정본을 위임하므로 layer-wide K 가 bit-identical.
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

/// cache 기반 wrapper — reader = `dequantize_k`(cache). production 경로(evict_and_merge)는 이제
/// compute_d2o_plan 이 `find_nearest_layer_wide_via` 를 직접 쓰므로, 이 wrapper 는 기존 테스트가
/// cache 로 직접 호출하기 위한 **test 전용** 편의 함수다(lib 빌드 미사용 → `#[cfg(test)]`).
#[cfg(test)]
fn find_nearest_layer_wide(
    cache: &KVCache,
    evict_pos: usize,
    retain_set: &[usize],
    kv_heads: usize,
    head_dim: usize,
) -> Match {
    find_nearest_layer_wide_via(
        &|p, h, o| dequantize_k(cache, p, h, head_dim, o),
        evict_pos,
        retain_set,
        kv_heads,
        head_dim,
    )
}

/// Group passing evicted tokens by their nearest retained token (Eq.8 m_ij ⇒ groups).
///
/// Returns a map `retain_pos → Vec<(evict_pos, sim)>`, deterministically ordered
/// by retain_pos for reproducibility (BTreeMap-style sort applied at use sites
/// where order matters; HashMap iteration order is fine for arithmetic).
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
/// Returns `(w_c, weights_per_evicted)` where:
///   D = Σ exp(u_i) + e
///   w_c = e / D   (retained self-weight)
///   w_i = exp(u_i) / D   (each evicted's contribution weight)
///
/// `u_i` is clamped to `[-10, 10]` before exp to prevent overflow / underflow.
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

/// Scatter-reduce merge using paper Eq.11 weights with layer-wide nearest mapping.
///
/// For each retained token c_j with group {e_i}:
///   D_j = Σ_i exp(u_ij) + e
///   w_cj = e / D_j;   w_ei = exp(u_ij) / D_j
///   k_cj ← w_cj · k_cj + Σ_i w_ei · k_ei      (per head)
///   v_cj ← w_cj · v_cj + Σ_i w_ei · v_ei      (per head)
///
/// Weights sum to 1 by construction (Σ w_ei + w_cj = 1), preserving K/V magnitude.
fn scatter_reduce_merge_layer_wide(
    cache: &mut KVCache,
    passing_positions: &[usize],
    matches: &[Match],
    kv_heads: usize,
    head_dim: usize,
    merge_e: f32,
) {
    let dtype = cache.k_buffer.dtype();
    match dtype {
        DType::F32 => {
            scatter_reduce_f32(
                cache,
                passing_positions,
                matches,
                kv_heads,
                head_dim,
                merge_e,
            );
        }
        DType::F16 => {
            scatter_reduce_f16(
                cache,
                passing_positions,
                matches,
                kv_heads,
                head_dim,
                merge_e,
            );
        }
        DType::Q4_0 => {
            scatter_reduce_q4(
                cache,
                passing_positions,
                matches,
                kv_heads,
                head_dim,
                merge_e,
            );
        }
        _ => {}
    }
}

#[allow(clippy::needless_range_loop)]
fn scatter_reduce_f32(
    cache: &mut KVCache,
    passing_positions: &[usize],
    matches: &[Match],
    kv_heads: usize,
    head_dim: usize,
    merge_e: f32,
) {
    let groups = group_by_retain(passing_positions, matches);

    for (retain_pos, evicted_list) in &groups {
        // Eq.11 weights are layer-wide constants (sim is layer-wide), shared across heads.
        let (w_c, w_e) = compute_eq11_weights(evicted_list, merge_e);

        for h in 0..kv_heads {
            let retain_off = cache.offset(*retain_pos, h);
            let evict_offs: Vec<usize> = evicted_list
                .iter()
                .map(|&(ep, _)| cache.offset(ep, h))
                .collect();

            // K
            {
                let k = cache.k_buffer.as_mut_slice::<f32>();
                for d in 0..head_dim {
                    let mut acc = w_c * k[retain_off + d];
                    for (idx, &evict_off) in evict_offs.iter().enumerate() {
                        acc += w_e[idx] * k[evict_off + d];
                    }
                    k[retain_off + d] = acc;
                }
            }
            // V
            {
                let v = cache.v_buffer.as_mut_slice::<f32>();
                for d in 0..head_dim {
                    let mut acc = w_c * v[retain_off + d];
                    for (idx, &evict_off) in evict_offs.iter().enumerate() {
                        acc += w_e[idx] * v[evict_off + d];
                    }
                    v[retain_off + d] = acc;
                }
            }
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn scatter_reduce_f16(
    cache: &mut KVCache,
    passing_positions: &[usize],
    matches: &[Match],
    kv_heads: usize,
    head_dim: usize,
    merge_e: f32,
) {
    let groups = group_by_retain(passing_positions, matches);

    for (retain_pos, evicted_list) in &groups {
        let (w_c, w_e) = compute_eq11_weights(evicted_list, merge_e);

        for h in 0..kv_heads {
            let retain_off = cache.offset(*retain_pos, h);
            let evict_offs: Vec<usize> = evicted_list
                .iter()
                .map(|&(ep, _)| cache.offset(ep, h))
                .collect();

            // K
            {
                let k = cache.k_buffer.as_mut_slice::<f16>();
                for d in 0..head_dim {
                    let mut acc = w_c * k[retain_off + d].to_f32();
                    for (idx, &evict_off) in evict_offs.iter().enumerate() {
                        acc += w_e[idx] * k[evict_off + d].to_f32();
                    }
                    k[retain_off + d] = f16::from_f32(acc);
                }
            }
            // V
            {
                let v = cache.v_buffer.as_mut_slice::<f16>();
                for d in 0..head_dim {
                    let mut acc = w_c * v[retain_off + d].to_f32();
                    for (idx, &evict_off) in evict_offs.iter().enumerate() {
                        acc += w_e[idx] * v[evict_off + d].to_f32();
                    }
                    v[retain_off + d] = f16::from_f32(acc);
                }
            }
        }
    }
}

#[allow(clippy::needless_range_loop)]
// LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path D2O Q4 scatter
fn scatter_reduce_q4(
    cache: &mut KVCache,
    passing_positions: &[usize],
    matches: &[Match],
    kv_heads: usize,
    head_dim: usize,
    merge_e: f32,
) {
    let blocks_per_pos = head_dim / QK4_0;
    let groups = group_by_retain(passing_positions, matches);

    for (retain_pos, evicted_list) in &groups {
        let (w_c, w_e) = compute_eq11_weights(evicted_list, merge_e);

        for h in 0..kv_heads {
            // Dequantize evict K and V upfront (each evicted token used by both
            // K and V merge — single dequant per token saves Q4 unpack cost).
            let evict_k_f32s: Vec<Vec<f32>> = evicted_list
                .iter()
                .map(|&(ep, _)| {
                    let mut buf = vec![0.0f32; head_dim];
                    dequantize_k(cache, ep, h, head_dim, &mut buf);
                    buf
                })
                .collect();

            let evict_v_f32s: Vec<Vec<f32>> = evicted_list
                .iter()
                .map(|&(ep, _)| {
                    let mut buf = vec![0.0f32; head_dim];
                    match cache.v_buffer.dtype() {
                        DType::Q4_0 => {
                            let v = cache.v_buffer.as_slice::<BlockQ4_0>();
                            let ep_block_off = cache.q4_block_offset(ep, h, blocks_per_pos);
                            for bi in 0..blocks_per_pos {
                                let mut tmp = [0.0f32; QK4_0];
                                v[ep_block_off + bi].dequantize(&mut tmp);
                                let dst = bi * QK4_0;
                                buf[dst..dst + QK4_0].copy_from_slice(&tmp);
                            }
                        }
                        DType::F32 => {
                            let v_off = cache.offset(ep, h);
                            let v = cache.v_buffer.as_slice::<f32>();
                            buf.copy_from_slice(&v[v_off..v_off + head_dim]);
                        }
                        DType::F16 => {
                            let v_off = cache.offset(ep, h);
                            let v = cache.v_buffer.as_slice::<f16>();
                            for d in 0..head_dim {
                                buf[d] = v[v_off + d].to_f32();
                            }
                        }
                        _ => {}
                    }
                    buf
                })
                .collect();

            // K merge (Q4_0)
            let retain_block_off = cache.q4_block_offset(*retain_pos, h, blocks_per_pos);
            {
                let k = cache.k_buffer.as_mut_slice::<BlockQ4_0>();
                for bi in 0..blocks_per_pos {
                    let mut r_f32 = [0.0f32; QK4_0];
                    k[retain_block_off + bi].dequantize(&mut r_f32);
                    for i in 0..QK4_0 {
                        r_f32[i] *= w_c;
                    }
                    for (idx, e_buf) in evict_k_f32s.iter().enumerate() {
                        let base = bi * QK4_0;
                        for i in 0..QK4_0 {
                            r_f32[i] += w_e[idx] * e_buf[base + i];
                        }
                    }
                    k[retain_block_off + bi] = BlockQ4_0::quantize(&r_f32);
                }
            }

            // V merge — dispatch on V dtype
            match cache.v_buffer.dtype() {
                DType::Q4_0 => {
                    let retain_v_block_off = cache.q4_block_offset(*retain_pos, h, blocks_per_pos);
                    let v = cache.v_buffer.as_mut_slice::<BlockQ4_0>();
                    for bi in 0..blocks_per_pos {
                        let mut r_f32 = [0.0f32; QK4_0];
                        v[retain_v_block_off + bi].dequantize(&mut r_f32);
                        for i in 0..QK4_0 {
                            r_f32[i] *= w_c;
                        }
                        for (idx, e_buf) in evict_v_f32s.iter().enumerate() {
                            let base = bi * QK4_0;
                            for i in 0..QK4_0 {
                                r_f32[i] += w_e[idx] * e_buf[base + i];
                            }
                        }
                        v[retain_v_block_off + bi] = BlockQ4_0::quantize(&r_f32);
                    }
                }
                DType::F32 => {
                    let retain_v_off = cache.offset(*retain_pos, h);
                    let v = cache.v_buffer.as_mut_slice::<f32>();
                    for d in 0..head_dim {
                        let mut acc = w_c * v[retain_v_off + d];
                        for (idx, e_buf) in evict_v_f32s.iter().enumerate() {
                            acc += w_e[idx] * e_buf[d];
                        }
                        v[retain_v_off + d] = acc;
                    }
                }
                DType::F16 => {
                    let retain_v_off = cache.offset(*retain_pos, h);
                    let v = cache.v_buffer.as_mut_slice::<f16>();
                    for d in 0..head_dim {
                        let mut acc = w_c * v[retain_v_off + d].to_f32();
                        for (idx, e_buf) in evict_v_f32s.iter().enumerate() {
                            acc += w_e[idx] * e_buf[d];
                        }
                        v[retain_v_off + d] = f16::from_f32(acc);
                    }
                }
                _ => {}
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::useless_vec)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::PressureLevel;
    use crate::kv_cache_ops::KVLayout;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    // ── Test helpers ──

    /// Create multiple KVCaches all at the same position (for multi-layer tests).
    fn make_caches(n_layers: usize, pos: usize) -> Vec<KVCache> {
        (0..n_layers).map(|_| make_cache(100, 1, 4, pos)).collect()
    }

    /// Create a KVCache with known K/V values for testing.
    /// Layout: SeqMajor [1, max_seq, kv_heads, head_dim].
    fn make_cache(max_seq: usize, kv_heads: usize, head_dim: usize, pos: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * kv_heads * head_dim * 4;
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend,
        );
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = pos;
        cache
    }

    fn make_cache_head_major(
        max_seq: usize,
        kv_heads: usize,
        head_dim: usize,
        pos: usize,
    ) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * kv_heads * head_dim * 4;
        // Pass shape as [1, max_seq, kv_heads, head_dim] so KVCache::new
        // correctly reads kv_heads from shape[2] and head_dim from shape[3].
        // The HeadMajor layout flag controls how offset() computes addresses.
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend,
        );
        let mut cache = KVCache::new(k, v, max_seq).with_layout(KVLayout::HeadMajor);
        cache.current_pos = pos;
        cache
    }

    /// Write a known F32 pattern to K[pos, head, dim].
    fn write_k(cache: &mut KVCache, pos: usize, head: usize, values: &[f32]) {
        let off = cache.offset(pos, head);
        let k_data = cache.k_buffer.as_mut_slice::<f32>();
        k_data[off..off + values.len()].copy_from_slice(values);
    }

    /// Write a known F32 pattern to V[pos, head, dim].
    fn write_v(cache: &mut KVCache, pos: usize, head: usize, values: &[f32]) {
        let off = cache.offset(pos, head);
        let v_data = cache.v_buffer.as_mut_slice::<f32>();
        v_data[off..off + values.len()].copy_from_slice(values);
    }

    /// Read K[pos, head] as a Vec<f32>.
    fn read_k(cache: &KVCache, pos: usize, head: usize, dim: usize) -> Vec<f32> {
        let off = cache.offset(pos, head);
        let k_data = cache.k_buffer.as_slice::<f32>();
        k_data[off..off + dim].to_vec()
    }

    fn read_v(cache: &KVCache, pos: usize, head: usize, dim: usize) -> Vec<f32> {
        let off = cache.offset(pos, head);
        let v_data = cache.v_buffer.as_slice::<f32>();
        v_data[off..off + dim].to_vec()
    }

    // ── Cosine similarity tests ──

    #[test]
    fn test_cosine_identical_vectors() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors: sim={sim}");
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors: sim={sim}");
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = [1.0, 2.0, 3.0];
        let b = [-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "opposite vectors: sim={sim}");
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [1.0, 2.0, 3.0];
        let b = [0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "zero vector: sim={sim}");
    }

    // ── find_nearest_layer_wide tests ──

    #[test]
    fn test_find_nearest_layer_wide_seq_major() {
        // Single head, identical to per-head case.
        let mut cache = make_cache(10, 1, 4, 5);

        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]); // pos 0
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]); // pos 1
        write_k(&mut cache, 2, 0, &[0.9, 0.1, 0.0, 0.0]); // pos 2 — similar to pos 0
        write_k(&mut cache, 3, 0, &[0.0, 0.0, 1.0, 0.0]); // pos 3
        write_k(&mut cache, 4, 0, &[0.0, 0.0, 0.0, 1.0]); // pos 4

        let retain = vec![0, 1, 3, 4];
        let m = find_nearest_layer_wide(&cache, 2, &retain, 1, 4);

        assert_eq!(m.retain_pos, 0, "pos 2 should be nearest to pos 0");
        assert!(m.sim > 0.9, "similarity should be high, got {}", m.sim);
    }

    #[test]
    fn test_find_nearest_layer_wide_multi_head() {
        // 2 heads. Layer-wide concat means per-head ambiguity is averaged out.
        let mut cache = make_cache(10, 2, 4, 4);

        // Concat([1,0,0,0],[1,0,0,0]) vs ([0,1,0,0],[0,1,0,0]) vs evict([0.95,0.05,0,0],[0.95,0.05,0,0])
        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]);
        write_k(&mut cache, 0, 1, &[1.0, 0.0, 0.0, 0.0]);
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]);
        write_k(&mut cache, 1, 1, &[0.0, 1.0, 0.0, 0.0]);
        write_k(&mut cache, 2, 0, &[0.95, 0.05, 0.0, 0.0]);
        write_k(&mut cache, 2, 1, &[0.95, 0.05, 0.0, 0.0]);

        let retain = vec![0, 1];
        let m = find_nearest_layer_wide(&cache, 2, &retain, 2, 4);
        assert_eq!(m.retain_pos, 0, "layer-wide concat → nearest = pos 0");
        assert!(m.sim > 0.9);
    }

    #[test]
    fn test_find_nearest_layer_wide_layout_equivalence() {
        // SeqMajor and HeadMajor must agree.
        let mut cache_seq = make_cache(10, 2, 4, 4);
        let mut cache_hm = make_cache_head_major(10, 2, 4, 4);

        for cache in [&mut cache_seq, &mut cache_hm] {
            write_k(cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]);
            write_k(cache, 0, 1, &[0.5, 0.5, 0.0, 0.0]);
            write_k(cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]);
            write_k(cache, 1, 1, &[0.0, 0.0, 1.0, 0.0]);
            write_k(cache, 2, 0, &[0.9, 0.1, 0.0, 0.0]);
            write_k(cache, 2, 1, &[0.45, 0.45, 0.0, 0.0]);
        }

        let retain = vec![0, 1];
        let m_seq = find_nearest_layer_wide(&cache_seq, 2, &retain, 2, 4);
        let m_hm = find_nearest_layer_wide(&cache_hm, 2, &retain, 2, 4);
        assert_eq!(
            m_seq.retain_pos, m_hm.retain_pos,
            "layout-equivalent target"
        );
        assert!(
            (m_seq.sim - m_hm.sim).abs() < 1e-6,
            "layout-equivalent sim: seq={} hm={}",
            m_seq.sim,
            m_hm.sim
        );
    }

    // ── scatter_reduce_merge_layer_wide tests ──

    #[test]
    fn test_scatter_reduce_1to1() {
        // 1 evicted → 1 retained, paper Eq.11:
        //   D = exp(sim) + e
        //   w_c = e / D, w_e = exp(sim) / D
        //   K[c] ← w_c·K[c] + w_e·K[e]
        let mut cache = make_cache(10, 1, 4, 3);

        write_k(&mut cache, 0, 0, &[1.0, 2.0, 3.0, 4.0]); // retain
        write_k(&mut cache, 1, 0, &[5.0, 6.0, 7.0, 8.0]); // evict
        write_v(&mut cache, 0, 0, &[10.0, 20.0, 30.0, 40.0]);
        write_v(&mut cache, 1, 0, &[50.0, 60.0, 70.0, 80.0]);

        let sim = 0.8f32;
        let e = 0.1f32;
        let match0 = Match { retain_pos: 0, sim };
        let passing = vec![1usize];
        let matches = vec![match0];

        scatter_reduce_merge_layer_wide(&mut cache, &passing, &matches, 1, 4, e);

        let k = read_k(&cache, 0, 0, 4);
        let v = read_v(&cache, 0, 0, 4);

        let exp_sim = sim.exp();
        let denom = exp_sim + e;
        let w_c = e / denom;
        let w_e = exp_sim / denom;

        let expected_k: Vec<f32> = (0..4)
            .map(|d| w_c * (d as f32 + 1.0) + w_e * (d as f32 + 5.0))
            .collect();
        let expected_v: Vec<f32> = (0..4)
            .map(|d| w_c * (10.0 * (d as f32 + 1.0)) + w_e * (10.0 * (d as f32 + 5.0)))
            .collect();

        for d in 0..4 {
            assert!(
                (k[d] - expected_k[d]).abs() < 1e-5,
                "K[{d}]: got {}, expected {}",
                k[d],
                expected_k[d]
            );
            assert!(
                (v[d] - expected_v[d]).abs() < 1e-4,
                "V[{d}]: got {}, expected {}",
                v[d],
                expected_v[d]
            );
        }
    }

    #[test]
    fn test_scatter_reduce_nto1() {
        // 2 evicted → same retained:
        //   D = exp(s1) + exp(s2) + e
        //   K[c] ← (e·K[c] + exp(s1)·K[e1] + exp(s2)·K[e2]) / D
        let mut cache = make_cache(10, 1, 4, 4);

        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]); // retain
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]); // evict 1
        write_k(&mut cache, 2, 0, &[0.0, 0.0, 1.0, 0.0]); // evict 2

        let s1 = 0.6f32;
        let s2 = 0.4f32;
        let e = 0.1f32;

        let matches = vec![
            Match {
                retain_pos: 0,
                sim: s1,
            },
            Match {
                retain_pos: 0,
                sim: s2,
            },
        ];
        let passing = vec![1usize, 2usize];

        scatter_reduce_merge_layer_wide(&mut cache, &passing, &matches, 1, 4, e);

        let k = read_k(&cache, 0, 0, 4);

        let e1 = s1.exp();
        let e2 = s2.exp();
        let denom = e1 + e2 + e;
        let w_c = e / denom;
        let w_1 = e1 / denom;
        let w_2 = e2 / denom;

        let expected_k = vec![w_c * 1.0, w_1 * 1.0, w_2 * 1.0, 0.0];

        for d in 0..4 {
            assert!(
                (k[d] - expected_k[d]).abs() < 1e-5,
                "K[{d}]: got {}, expected {}",
                k[d],
                expected_k[d]
            );
        }
    }

    #[test]
    fn test_scatter_reduce_no_mapping_unchanged() {
        // Retain pos 1 is not a merge target — must stay untouched.
        let mut cache = make_cache(10, 1, 4, 3);

        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]); // retain (target)
        write_k(&mut cache, 1, 0, &[9.9, 9.9, 9.9, 9.9]); // retain (NOT a target)
        write_k(&mut cache, 2, 0, &[0.9, 0.1, 0.0, 0.0]); // evict → pos 0

        let matches = vec![Match {
            retain_pos: 0,
            sim: 0.99,
        }];
        let passing = vec![2usize];

        scatter_reduce_merge_layer_wide(&mut cache, &passing, &matches, 1, 4, 0.1);

        let k1 = read_k(&cache, 1, 0, 4);
        for d in 0..4 {
            assert!(
                (k1[d] - 9.9).abs() < 1e-5,
                "unmapped retain pos 1 K[{d}] should be unchanged: got {}",
                k1[d]
            );
        }
    }

    #[test]
    fn test_merge_weight_normalization() {
        // For any group, w_c + Σ w_e must equal 1.0.
        let cases: Vec<(Vec<f32>, f32)> = vec![
            (vec![0.8], 0.1),
            (vec![0.6, 0.4], 0.1),
            (vec![1.0, 0.5, 0.0, -0.5], 0.1),
            (vec![0.0], 0.5),
            (vec![5.0, -5.0, 0.0], 0.1),
        ];
        for (sims, e) in cases {
            let evicted_list: Vec<(usize, f32)> =
                sims.iter().enumerate().map(|(i, &s)| (i, s)).collect();
            let (w_c, w_e) = compute_eq11_weights(&evicted_list, e);
            let sum: f32 = w_c + w_e.iter().sum::<f32>();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "weights must sum to 1: w_c={w_c}, w_e={w_e:?}, sum={sum}, sims={sims:?}",
            );
        }
    }

    #[test]
    fn test_merge_weight_handles_negative_sim() {
        // sim < 0 ⇒ exp(sim) < 1 (small positive), never negative weight.
        // K contribution is reduced but never subtracted.
        let mut cache = make_cache(10, 1, 4, 3);
        write_k(&mut cache, 0, 0, &[10.0, 0.0, 0.0, 0.0]); // retain (large positive)
        write_k(&mut cache, 1, 0, &[1.0, 0.0, 0.0, 0.0]); // evict, sim=-1 (anti-correlated context)

        let matches = vec![Match {
            retain_pos: 0,
            sim: -1.0,
        }];
        let passing = vec![1usize];
        let e = 0.1f32;

        scatter_reduce_merge_layer_wide(&mut cache, &passing, &matches, 1, 4, e);

        let k = read_k(&cache, 0, 0, 4);
        // K[0,0] starts at 10.0. After merge:
        //   exp(-1) ≈ 0.3679, D = 0.3679 + 0.1 = 0.4679
        //   w_c = 0.1/0.4679 ≈ 0.2137, w_e ≈ 0.7862
        //   k = 0.2137 * 10.0 + 0.7862 * 1.0 ≈ 2.9234
        let exp_neg = (-1.0f32).exp();
        let denom = exp_neg + e;
        let w_c = e / denom;
        let w_e = exp_neg / denom;
        let expected = w_c * 10.0 + w_e * 1.0;
        assert!(
            (k[0] - expected).abs() < 1e-4,
            "negative-sim merge: got {}, expected {}",
            k[0],
            expected
        );
        // Crucially: weight is positive, so contribution is additive (not subtracted)
        assert!(w_e > 0.0, "evict weight must be positive even for sim<0");
        // And the merged value is between K_evict (1.0) and K_retain (10.0)
        assert!(k[0] > 1.0 && k[0] < 10.0, "merged value should lie between");
    }

    // ── EMA threshold tests ──

    #[test]
    fn test_ema_initialization() {
        // Paper Eq.10 first-call init: τ_0 = mean over evicted of max_j u_ij.
        // Layer-wide nearest gives one sim per evicted ⇒ τ_0 = mean(match.sim).
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_beta: 0.7,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });

        let mut cache = make_cache(20, 1, 4, 10);
        // Identical vectors → all cosine sims = 1.0 ⇒ mean_max = 1.0
        for pos in 0..10 {
            write_k(&mut cache, pos, 0, &[1.0, 0.0, 0.0, 0.0]);
        }

        let importance: Vec<f32> = (0..20).map(|i| 10.0 - i as f32).collect();

        let state = handler.state.lock().unwrap();
        assert!(!state.initialized, "should start uninitialized");
        drop(state);

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        handler.handle(&mut ctx).unwrap();

        let state = handler.state.lock().unwrap();
        assert!(
            state.initialized,
            "should be initialized after first eviction"
        );
        assert!(
            (state.ema_threshold - 1.0).abs() < 1e-4,
            "τ_0 should equal mean of max_sim ≈ 1.0, got {}",
            state.ema_threshold
        );
    }

    #[test]
    fn test_ema_init_no_double_update() {
        // First call sets τ_0 = mean(max_sim). The β-weighted update only fires
        // on the second call, not stacked on top of the init within one call.
        let handler = D2OHandler::new(D2OConfig::default());
        let mut state = D2OState::new();
        let mut cache = make_cache(20, 1, 4, 12);

        // Write identical vectors → cosine sim = 1.0 for all pairs
        for pos in 0..12 {
            write_k(&mut cache, pos, 0, &[1.0, 0.0, 0.0, 0.0]);
        }
        let importance: Vec<f32> = (0..12).map(|i| 12.0 - i as f32).collect();

        handler
            .evict_and_merge(&mut cache, 6, &importance, &mut state, true)
            .unwrap();

        // All sims = 1.0 → τ_0 = 1.0 (init only, no β update).
        assert!(state.initialized, "should be initialized");
        assert!(
            (state.ema_threshold - 1.0).abs() < 1e-4,
            "τ_0 should equal mean(max_sim)=1.0 with no β update, got {}",
            state.ema_threshold
        );
    }

    #[test]
    fn test_ema_update() {
        // Pure-arithmetic check of paper Eq.10:
        //   τ_t = β · max U_t + (1−β) · τ_{t−1}.
        let beta: f32 = 0.7;
        let mut threshold: f32 = 0.5; // initial τ_0 (already initialised)

        // Second call: global max sim = 0.8
        threshold = beta * 0.8 + (1.0 - beta) * threshold;
        let expected = 0.7 * 0.8 + 0.3 * 0.5; // = 0.71
        assert!(
            (threshold - expected).abs() < 1e-6,
            "threshold={threshold}, expected={expected}"
        );

        // Third call: global max sim = 0.3
        threshold = beta * 0.3 + (1.0 - beta) * threshold;
        let expected2 = 0.7 * 0.3 + 0.3 * expected;
        assert!(
            (threshold - expected2).abs() < 1e-6,
            "threshold={threshold}, expected={expected2}"
        );
    }

    #[test]
    fn test_ema_uses_global_max_after_init() {
        // After init, the threshold update must use the *max* sim across the
        // current eviction batch, not its mean. We verify by driving two calls
        // with hand-controlled cosine similarities.
        let beta = 0.7f32;
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.5,
            protected_prefix: 0,
            target_ratio: 0.5,
            ema_beta: beta,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });
        let mut state = D2OState::new();
        let mut cache = make_cache(20, 1, 4, 4);

        // First call: 2 evicted (pos 0, 1), 2 retained (pos 2, 3).
        // Make sim(evict_0, retain_2)=1.0 and sim(evict_1, retain_3)=1.0.
        // ⇒ τ_0 = mean(1.0, 1.0) = 1.0.
        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]);
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]);
        write_k(&mut cache, 2, 0, &[1.0, 0.0, 0.0, 0.0]);
        write_k(&mut cache, 3, 0, &[0.0, 1.0, 0.0, 0.0]);

        // Importance: low for pos 0, 1 (evicted), high for pos 2, 3 (kept as HH).
        let importance = vec![0.1, 0.1, 10.0, 9.0];
        handler
            .evict_and_merge(&mut cache, 2, &importance, &mut state, true)
            .unwrap();
        assert!(state.initialized, "first call must initialize");
        assert!(
            (state.ema_threshold - 1.0).abs() < 1e-4,
            "τ_0 = mean(max_sim) = 1.0, got {}",
            state.ema_threshold
        );

        // Second call on a fresh, mixed-similarity layout:
        //   evict pos 0: orthogonal to retain pos 2 (sim ≈ 0.0).
        //   evict pos 1: identical to retain pos 3 (sim = 1.0).
        // Global max = 1.0; mean would be 0.5 — must distinguish.
        let mut cache2 = make_cache(20, 1, 4, 4);
        write_k(&mut cache2, 0, 0, &[1.0, 0.0, 0.0, 0.0]);
        write_k(&mut cache2, 1, 0, &[0.0, 1.0, 0.0, 0.0]);
        write_k(&mut cache2, 2, 0, &[0.0, 0.0, 1.0, 0.0]); // orthogonal → sim≈0
        write_k(&mut cache2, 3, 0, &[0.0, 1.0, 0.0, 0.0]); // identical → sim=1
        handler
            .evict_and_merge(&mut cache2, 2, &importance, &mut state, true)
            .unwrap();

        // Expected: τ_1 = β · max + (1−β) · τ_0 = 0.7·1.0 + 0.3·1.0 = 1.0.
        // Distinguish from mean-based: 0.7·0.5 + 0.3·1.0 = 0.65 ≠ 1.0.
        let expected_max_based = beta * 1.0 + (1.0 - beta) * 1.0;
        let mean_based_alt = beta * 0.5 + (1.0 - beta) * 1.0;
        assert!(
            (state.ema_threshold - expected_max_based).abs() < 1e-3,
            "τ should follow max-based update: got {}, expected {} (mean-based={})",
            state.ema_threshold,
            expected_max_based,
            mean_based_alt
        );
    }

    #[test]
    fn test_filter_uses_max_sim() {
        // Verify that the merge filter compares each evicted token's sim
        // against τ: matches with sim < τ are deleted, sim ≥ τ are merged.
        // We construct a scenario where τ lands between two evicted sims.
        //
        // Layout (4 tokens, prefix=0, target=2, keep_ratio=0.5):
        //   available = 2, hh_budget = 1, recent_budget = 1
        //   recent_start = 3 ⇒ pos 3 is the recent window
        //   pos 0..3 are ranked by importance: pos 2 (10.0) → HH; pos 0,1 → evict
        // Retain set after partition: {2 (HH), 3 (recent)}.
        //
        //   K[0] = e1 → nearest in {2,3} via cosine ⇒ pos 2 (sim = 1.0)
        //   K[1] = e3 → nearest in {2,3} ⇒ both orthogonal (sim = 0.0)
        //   τ_0 = mean(1.0, 0.0) = 0.5
        //   evict 0: sim 1.0 ≥ τ → merged
        //   evict 1: sim 0.0 < τ → deleted
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.5,
            protected_prefix: 0,
            target_ratio: 0.5,
            ema_beta: 0.7,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });
        let mut state = D2OState::new();

        let mut cache = make_cache(20, 1, 4, 4);
        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]); // e1
        write_k(&mut cache, 1, 0, &[0.0, 0.0, 1.0, 0.0]); // e3 (orthogonal to retain set)
        write_k(&mut cache, 2, 0, &[1.0, 0.0, 0.0, 0.0]); // HH; matches pos 0
        write_k(&mut cache, 3, 0, &[0.0, 1.0, 0.0, 0.0]); // recent; orthogonal to pos 1

        let importance = vec![0.1, 0.1, 10.0, 9.0];
        handler
            .evict_and_merge(&mut cache, 2, &importance, &mut state, true)
            .unwrap();

        assert!(
            (state.ema_threshold - 0.5).abs() < 1e-4,
            "τ_0 expected 0.5, got {}",
            state.ema_threshold
        );
        assert_eq!(
            state.total_merged, 1,
            "exactly one evicted token should pass the filter"
        );
        assert_eq!(
            state.total_deleted, 1,
            "exactly one evicted token should be deleted (sim<τ)"
        );
    }

    // ── Handler integration tests ──

    #[test]
    fn test_handler_noop_no_importance() {
        let handler = D2OHandler::with_defaults(0.5);
        let mut caches = vec![make_cache(20, 1, 4, 10)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None, // no scores
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_handler_noop_below_target() {
        let handler = D2OHandler::with_defaults(0.9); // keep 90%
        let cache = make_cache(20, 1, 4, 5); // only 5 tokens
        let importance = vec![1.0; 20];

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        // target_len = 5 * 0.9 = 4; 5 > 4 so eviction should happen
        // But with prefix=4, keep=max(4, 4+2)=6, 5 <= 6 → NoOp
        assert!(!result.is_action());
    }

    #[test]
    fn test_handler_evicts_tokens() {
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_beta: 0.5,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });

        let mut cache = make_cache(50, 1, 4, 30);
        // Write distinct K vectors for meaningful similarity
        for pos in 0..30 {
            let angle = pos as f32 * 0.2;
            write_k(&mut cache, pos, 0, &[angle.cos(), angle.sin(), 0.0, 0.0]);
            write_v(&mut cache, pos, 0, &[pos as f32, 0.0, 0.0, 0.0]);
        }

        // High importance for first tokens, low for middle
        let mut importance = vec![0.1; 50];
        for i in 0..5 {
            importance[i] = 10.0;
        }
        for i in 25..30 {
            importance[i] = 5.0; // recent window
        }

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted {
                tokens_removed,
                new_pos,
            } => {
                assert!(tokens_removed > 0, "should have removed tokens");
                assert!(new_pos < 30, "new_pos should be less than original");
                assert!(new_pos >= 2, "should keep at least prefix");
                // target = 30 * 0.5 = 15
                assert_eq!(new_pos, 15, "should reduce to target");
            }
            other => panic!("Expected Evicted, got {:?}", other),
        }
    }

    #[test]
    fn test_handler_empty_caches() {
        let handler = D2OHandler::with_defaults(0.5);
        let importance = vec![1.0; 10];
        let mut caches: Vec<KVCache> = vec![];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_handler_multi_layer() {
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_beta: 0.5,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });

        let mut caches: Vec<KVCache> = (0..4)
            .map(|_| {
                let mut c = make_cache(50, 1, 4, 20);
                for pos in 0..20 {
                    let angle = pos as f32 * 0.3;
                    write_k(&mut c, pos, 0, &[angle.cos(), angle.sin(), 0.0, 0.0]);
                    write_v(&mut c, pos, 0, &[pos as f32, 0.0, 0.0, 0.0]);
                }
                c
            })
            .collect();

        let mut importance = vec![0.1; 50];
        for i in 0..5 {
            importance[i] = 10.0;
        }

        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(result.is_action());

        // All layers should be at the same position
        let pos = ctx.caches[0].current_pos;
        for cache in ctx.caches.iter() {
            assert_eq!(cache.current_pos, pos, "all layers should have same pos");
        }
    }

    #[test]
    fn test_handler_name() {
        let handler = D2OHandler::with_defaults(0.5);
        assert_eq!(handler.name(), "d2o");
    }

    #[test]
    fn test_prefix_always_protected() {
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 4,
            target_ratio: 0.3, // aggressive
            ema_beta: 0.5,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });

        let mut cache = make_cache(50, 1, 4, 20);
        // Write unique K vectors for prefix tokens
        for pos in 0..4 {
            let val = 100.0 + pos as f32;
            write_k(&mut cache, pos, 0, &[val, val, val, val]);
        }
        for pos in 4..20 {
            write_k(&mut cache, pos, 0, &[0.1, 0.1, 0.1, 0.1]);
        }

        let importance = vec![0.01; 50]; // low importance everywhere
        // Prefix still protected regardless of importance

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        handler.handle(&mut ctx).unwrap();

        // Check prefix tokens are preserved
        for pos in 0..4 {
            let k = read_k(&ctx.caches[0], pos, 0, 4);
            let expected = 100.0 + pos as f32;
            assert!(
                (k[0] - expected).abs() < 1e-4,
                "prefix pos {pos} should be preserved: k={k:?}",
            );
        }
    }

    #[test]
    fn test_keep_ratio_3_to_1() {
        // Verify that D2O's 3:1 HH:Recent ratio gives more budget to heavy hitters
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_beta: 0.5,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });

        let mut cache = make_cache(50, 1, 4, 40);
        for pos in 0..40 {
            write_k(&mut cache, pos, 0, &[pos as f32, 0.0, 0.0, 0.0]);
        }
        // Assign importance: tokens 5, 10, 15 are most important
        let mut importance = vec![0.01; 50];
        importance[5] = 10.0;
        importance[10] = 9.0;
        importance[15] = 8.0;
        importance[6] = 7.0;
        importance[7] = 6.0;
        importance[8] = 5.0;
        importance[9] = 4.0;
        importance[11] = 3.0;
        importance[12] = 2.0;
        importance[13] = 1.5;
        importance[14] = 1.0;

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        handler.handle(&mut ctx).unwrap();

        let new_pos = ctx.caches[0].current_pos;
        // target = 40 * 0.5 = 20; keep = max(20, 2+2) = 20
        // available = 20 - 2 = 18; hh_budget = 18 * 0.75 = 13; recent = 5
        assert_eq!(new_pos, 20);
    }

    // ── Q4_0 tests ──

    /// Create a Q4_0 KV cache for testing.
    fn make_cache_q4(max_seq: usize, kv_heads: usize, head_dim: usize, pos: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let blocks_per_pos = head_dim / QK4_0;
        let total_blocks = max_seq * kv_heads * blocks_per_pos;
        let buf_size = total_blocks * std::mem::size_of::<BlockQ4_0>();
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::Q4_0)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::Q4_0)),
            backend,
        );
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = pos;
        cache
    }

    /// Write an f32 vector to Q4_0 K cache at (pos, head) via quantize.
    fn write_k_q4(cache: &mut KVCache, pos: usize, head: usize, values: &[f32]) {
        let head_dim = values.len();
        let blocks_per_pos = head_dim / QK4_0;
        let block_off = cache.q4_block_offset(pos, head, blocks_per_pos);
        let k = cache.k_buffer.as_mut_slice::<BlockQ4_0>();
        for bi in 0..blocks_per_pos {
            let start = bi * QK4_0;
            let mut src = [0.0f32; QK4_0];
            src.copy_from_slice(&values[start..start + QK4_0]);
            k[block_off + bi] = BlockQ4_0::quantize(&src);
        }
    }

    /// Read K cache at (pos, head) from Q4_0 as dequantized f32.
    fn read_k_q4(cache: &KVCache, pos: usize, head: usize, head_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; head_dim];
        dequantize_k(cache, pos, head, head_dim, &mut out);
        out
    }

    /// Write an f32 vector to Q4_0 V cache at (pos, head) via quantize.
    fn write_v_q4(cache: &mut KVCache, pos: usize, head: usize, values: &[f32]) {
        let head_dim = values.len();
        let blocks_per_pos = head_dim / QK4_0;
        let block_off = cache.q4_block_offset(pos, head, blocks_per_pos);
        let v = cache.v_buffer.as_mut_slice::<BlockQ4_0>();
        for bi in 0..blocks_per_pos {
            let start = bi * QK4_0;
            let mut src = [0.0f32; QK4_0];
            src.copy_from_slice(&values[start..start + QK4_0]);
            v[block_off + bi] = BlockQ4_0::quantize(&src);
        }
    }

    #[test]
    fn dequantize_v_equals_dequantize_k_bit_identical() {
        // (M-C) dequantize_v 는 dequantize_k 의 v_buffer 미러 — 동일 레이아웃/offset/dequant 로직.
        // K·V 에 동일 바이트를 쓰면 출력이 bitwise 일치해야 한다(F32/F16/Q4_0).
        let hd = QK4_0 * 2; // 64
        let vals: Vec<f32> = (0..hd).map(|d| (d as f32) * 0.013 - 0.4).collect();
        let mut ok = vec![0.0f32; hd];
        let mut ov = vec![0.0f32; hd];

        // F32
        let mut c = make_cache(8, 2, hd, 6);
        write_k(&mut c, 5, 1, &vals);
        write_v(&mut c, 5, 1, &vals);
        dequantize_k(&c, 5, 1, hd, &mut ok);
        dequantize_v(&c, 5, 1, hd, &mut ov);
        assert_eq!(ok, ov, "F32 dequant_v != dequant_k");

        // Q4_0
        let mut cq = make_cache_q4(8, 2, hd, 6);
        write_k_q4(&mut cq, 5, 1, &vals);
        write_v_q4(&mut cq, 5, 1, &vals);
        dequantize_k(&cq, 5, 1, hd, &mut ok);
        dequantize_v(&cq, 5, 1, hd, &mut ov);
        assert_eq!(ok, ov, "Q4_0 dequant_v != dequant_k");

        // F16
        let backend = Arc::new(CpuBackend::new());
        let buf = 8 * 2 * hd * std::mem::size_of::<f16>();
        let kt = Tensor::new(
            Shape::new(vec![1, 8, 2, hd]),
            Arc::new(SharedBuffer::new(buf, DType::F16)),
            backend.clone(),
        );
        let vt = Tensor::new(
            Shape::new(vec![1, 8, 2, hd]),
            Arc::new(SharedBuffer::new(buf, DType::F16)),
            backend,
        );
        let mut cf = KVCache::new(kt, vt, 8);
        cf.current_pos = 6;
        let off = cf.offset(5, 1);
        for d in 0..hd {
            cf.k_buffer.as_mut_slice::<f16>()[off + d] = f16::from_f32(vals[d]);
            cf.v_buffer.as_mut_slice::<f16>()[off + d] = f16::from_f32(vals[d]);
        }
        dequantize_k(&cf, 5, 1, hd, &mut ok);
        dequantize_v(&cf, 5, 1, hd, &mut ov);
        assert_eq!(ok, ov, "F16 dequant_v != dequant_k");
    }

    // ── M4-c: D2OStage(plan→executor) ≡ D2OHandler(evict_and_merge) — non-alloc config ──
    // compute_d2o_plan 공유 + apply_weighted_merges≡scatter_reduce(M4-b) → buffer bit-identical.
    // 같은 stage/state 인스턴스로 layer 루프를 돌아 EMA τ 누적 순서까지 일치시킨다.

    fn mk_d2o_config() -> D2OConfig {
        D2OConfig {
            keep_ratio: 0.5,
            protected_prefix: 4,
            target_ratio: 0.5,
            ema_beta: 0.7,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        }
    }

    /// A = D2OHandler.evict_and_merge per layer(공유 state_a). B = D2OStage.plan→execute_kv_plan
    /// per layer(공유 stage = 공유 EMA). 동일 target/importance/merge_enabled=true.
    fn run_d2o_parity(
        caches_a: &mut [KVCache],
        caches_b: &mut [KVCache],
        target: usize,
        imp: &[f32],
    ) {
        use crate::kv::eviction::stage_registry::{KVStageCtx, execute_kv_plan};
        let handler = D2OHandler::new(mk_d2o_config());
        let mut state_a = D2OState::new();
        for ca in caches_a.iter_mut() {
            handler
                .evict_and_merge(ca, target, imp, &mut state_a, true)
                .unwrap();
        }
        let stage = D2OStage::new(mk_d2o_config());
        for cb in caches_b.iter_mut() {
            let plan = {
                let ctx = KVStageCtx::new(cb, target, Some(imp), None, None, None);
                stage.plan(&ctx)
            };
            if let Some(p) = plan {
                execute_kv_plan(cb, &p).unwrap();
            }
        }
    }

    fn vbytes(c: &KVCache, bytes_per_pos: usize) -> (Vec<u8>, Vec<u8>) {
        let nb = c.current_pos * bytes_per_pos;
        unsafe {
            (
                std::slice::from_raw_parts(c.k_buffer.as_mut_ptr() as *const u8, nb).to_vec(),
                std::slice::from_raw_parts(c.v_buffer.as_mut_ptr() as *const u8, nb).to_vec(),
            )
        }
    }

    #[test]
    fn d2o_stage_eq_handler_f32() {
        let (kvh, hd, ms, pos, nl, tgt) = (2usize, 8usize, 64usize, 40usize, 3usize, 20usize);
        let imp: Vec<f32> = (0..pos).map(|i| (pos - i) as f32 + 1.0).collect();
        let build = || {
            (0..nl)
                .map(|_| {
                    let mut c = make_cache(ms, kvh, hd, pos);
                    for p in 0..pos {
                        for h in 0..kvh {
                            let base = ((p * kvh + h) * hd) as f32;
                            let k: Vec<f32> =
                                (0..hd).map(|d| (base + d as f32) * 0.05 + 0.3).collect();
                            let v: Vec<f32> =
                                (0..hd).map(|d| (base + d as f32) * -0.03 + 0.7).collect();
                            write_k(&mut c, p, h, &k);
                            write_v(&mut c, p, h, &v);
                        }
                    }
                    c
                })
                .collect::<Vec<_>>()
        };
        let mut a = build();
        let mut b = build();
        run_d2o_parity(&mut a, &mut b, tgt, &imp);
        for (ca, cb) in a.iter().zip(b.iter()) {
            assert_eq!(ca.current_pos, cb.current_pos, "f32 current_pos");
            assert_eq!(
                vbytes(ca, kvh * hd * 4),
                vbytes(cb, kvh * hd * 4),
                "f32 buffer"
            );
        }
    }

    #[test]
    fn d2o_stage_eq_handler_f16() {
        let (kvh, hd, ms, pos, nl, tgt) = (2usize, 8usize, 64usize, 40usize, 3usize, 20usize);
        let imp: Vec<f32> = (0..pos).map(|i| (pos - i) as f32 + 1.0).collect();
        let build = || {
            (0..nl)
                .map(|_| {
                    let backend = Arc::new(CpuBackend::new());
                    let buf = ms * kvh * hd * 2;
                    let k = Tensor::new(
                        Shape::new(vec![1, ms, kvh, hd]),
                        Arc::new(SharedBuffer::new(buf, DType::F16)),
                        backend.clone(),
                    );
                    let v = Tensor::new(
                        Shape::new(vec![1, ms, kvh, hd]),
                        Arc::new(SharedBuffer::new(buf, DType::F16)),
                        backend,
                    );
                    let mut c = KVCache::new(k, v, ms);
                    c.current_pos = pos;
                    for p in 0..pos {
                        for h in 0..kvh {
                            let off = c.offset(p, h);
                            let base = ((p * kvh + h) * hd) as f32;
                            let kb = c.k_buffer.as_mut_slice::<f16>();
                            for d in 0..hd {
                                kb[off + d] = f16::from_f32((base + d as f32) * 0.05 + 0.3);
                            }
                            let vb = c.v_buffer.as_mut_slice::<f16>();
                            for d in 0..hd {
                                vb[off + d] = f16::from_f32((base + d as f32) * -0.03 + 0.7);
                            }
                        }
                    }
                    c
                })
                .collect::<Vec<_>>()
        };
        let mut a = build();
        let mut b = build();
        run_d2o_parity(&mut a, &mut b, tgt, &imp);
        for (ca, cb) in a.iter().zip(b.iter()) {
            assert_eq!(ca.current_pos, cb.current_pos, "f16 current_pos");
            assert_eq!(
                vbytes(ca, kvh * hd * 2),
                vbytes(cb, kvh * hd * 2),
                "f16 buffer"
            );
        }
    }

    #[test]
    fn d2o_stage_eq_handler_q4() {
        let (kvh, hd, ms, pos, nl, tgt) = (2usize, QK4_0, 64usize, 40usize, 3usize, 20usize);
        let imp: Vec<f32> = (0..pos).map(|i| (pos - i) as f32 + 1.0).collect();
        let build = || {
            (0..nl)
                .map(|_| {
                    let mut c = make_cache_q4(ms, kvh, hd, pos);
                    for p in 0..pos {
                        for h in 0..kvh {
                            let base = ((p * kvh + h) * hd) as f32;
                            let k: Vec<f32> =
                                (0..hd).map(|d| (base + d as f32) * 0.05 + 0.3).collect();
                            let v: Vec<f32> =
                                (0..hd).map(|d| (base + d as f32) * -0.03 + 0.7).collect();
                            write_k_q4(&mut c, p, h, &k);
                            write_v_q4(&mut c, p, h, &v);
                        }
                    }
                    c
                })
                .collect::<Vec<_>>()
        };
        let bpp = (kvh * hd / QK4_0) * std::mem::size_of::<BlockQ4_0>();
        let mut a = build();
        let mut b = build();
        run_d2o_parity(&mut a, &mut b, tgt, &imp);
        for (ca, cb) in a.iter().zip(b.iter()) {
            assert_eq!(ca.current_pos, cb.current_pos, "q4 current_pos");
            assert_eq!(vbytes(ca, bpp), vbytes(cb, bpp), "q4 buffer");
        }
    }

    // ── M4-b: apply_weighted_merges(standard_format) ≡ scatter_reduce_merge_layer_wide(d2o) ──
    // 동일 weights(group_by_retain + compute_eq11_weights)에서 buffer bit-identical 임을 증명한다.
    // d2o-stage(M4-c)가 plan→executor(apply_weighted_merges)로 가도 기존 D2OHandler 와 무회귀임의 근거.

    /// scatter_reduce 가 내부에서 쓰는 grouping+weights 와 동일하게 WeightedMerge 를 구성.
    fn merges_from(
        passing: &[usize],
        matches: &[Match],
        merge_e: f32,
    ) -> Vec<technique_api::WeightedMerge> {
        let groups = group_by_retain(passing, matches);
        groups
            .iter()
            .map(|(retain, evicted_list)| {
                let (w_c, w_e) = compute_eq11_weights(evicted_list, merge_e);
                technique_api::WeightedMerge {
                    into: *retain,
                    into_weight: w_c,
                    from: evicted_list
                        .iter()
                        .zip(w_e.iter())
                        .map(|(&(ep, _), &w)| (ep, w))
                        .collect(),
                    apply_to: technique_api::MergeAxis::Both,
                }
            })
            .collect()
    }

    /// 유효 영역 [0..current_pos) 의 K/V raw byte (K=V 동일 dtype 전제).
    fn valid_bytes(cache: &KVCache, kv_heads: usize, head_dim: usize) -> (Vec<u8>, Vec<u8>) {
        let bpp = match cache.k_buffer.dtype() {
            DType::F32 => kv_heads * head_dim * 4,
            DType::F16 => kv_heads * head_dim * 2,
            DType::Q4_0 => (kv_heads * head_dim / QK4_0) * std::mem::size_of::<BlockQ4_0>(),
            other => panic!("unsupported dtype {other:?}"),
        };
        let nb = cache.current_pos * bpp;
        unsafe {
            (
                std::slice::from_raw_parts(cache.k_buffer.as_mut_ptr() as *const u8, nb).to_vec(),
                std::slice::from_raw_parts(cache.v_buffer.as_mut_ptr() as *const u8, nb).to_vec(),
            )
        }
    }

    fn parity_passing_matches() -> (Vec<usize>, Vec<Match>, f32) {
        // evict 5,8 → retain 3; evict 11 → retain 12 (retain∩evict=∅, pos<16).
        let passing = vec![5usize, 8, 11];
        let matches = vec![
            Match {
                retain_pos: 3,
                sim: 0.8,
            },
            Match {
                retain_pos: 3,
                sim: 0.5,
            },
            Match {
                retain_pos: 12,
                sim: 0.9,
            },
        ];
        (passing, matches, 0.1)
    }

    #[test]
    fn apply_weighted_merges_eq_scatter_f32() {
        let (kvh, hd, ms, pos) = (2usize, 8usize, 64usize, 16usize);
        let fill = |c: &mut KVCache| {
            let k = c.k_buffer.as_mut_slice::<f32>();
            for (i, x) in k.iter_mut().enumerate() {
                *x = (i as f32 + 1.0) * 0.123;
            }
            let v = c.v_buffer.as_mut_slice::<f32>();
            for (i, x) in v.iter_mut().enumerate() {
                *x = (i as f32 + 1.0) * -0.077 + 3.0;
            }
        };
        let mut a = make_cache(ms, kvh, hd, pos);
        let mut b = make_cache(ms, kvh, hd, pos);
        fill(&mut a);
        fill(&mut b);
        let (passing, matches, merge_e) = parity_passing_matches();
        scatter_reduce_merge_layer_wide(&mut a, &passing, &matches, kvh, hd, merge_e);
        crate::kv::standard_format::apply_weighted_merges(
            &mut b,
            &merges_from(&passing, &matches, merge_e),
        );
        let (ka, va) = valid_bytes(&a, kvh, hd);
        let (kb, vb) = valid_bytes(&b, kvh, hd);
        assert_eq!(ka, kb, "F32 K bit-identical");
        assert_eq!(va, vb, "F32 V bit-identical");
    }

    #[test]
    fn apply_weighted_merges_eq_scatter_f16() {
        use half::f16;
        let (kvh, hd, ms, pos) = (2usize, 8usize, 64usize, 16usize);
        let backend = Arc::new(CpuBackend::new());
        let buf_size = ms * kvh * hd * 2;
        let mk = || {
            let k = Tensor::new(
                Shape::new(vec![1, ms, kvh, hd]),
                Arc::new(SharedBuffer::new(buf_size, DType::F16)),
                backend.clone(),
            );
            let v = Tensor::new(
                Shape::new(vec![1, ms, kvh, hd]),
                Arc::new(SharedBuffer::new(buf_size, DType::F16)),
                backend.clone(),
            );
            let mut c = KVCache::new(k, v, ms);
            c.current_pos = pos;
            c
        };
        let fill = |c: &mut KVCache| {
            let k = c.k_buffer.as_mut_slice::<f16>();
            for (i, x) in k.iter_mut().enumerate() {
                *x = f16::from_f32((i as f32 + 1.0) * 0.123);
            }
            let v = c.v_buffer.as_mut_slice::<f16>();
            for (i, x) in v.iter_mut().enumerate() {
                *x = f16::from_f32((i as f32 + 1.0) * -0.077 + 3.0);
            }
        };
        let mut a = mk();
        let mut b = mk();
        fill(&mut a);
        fill(&mut b);
        let (passing, matches, merge_e) = parity_passing_matches();
        scatter_reduce_merge_layer_wide(&mut a, &passing, &matches, kvh, hd, merge_e);
        crate::kv::standard_format::apply_weighted_merges(
            &mut b,
            &merges_from(&passing, &matches, merge_e),
        );
        let (ka, va) = valid_bytes(&a, kvh, hd);
        let (kb, vb) = valid_bytes(&b, kvh, hd);
        assert_eq!(ka, kb, "F16 K bit-identical");
        assert_eq!(va, vb, "F16 V bit-identical");
    }

    #[test]
    fn apply_weighted_merges_eq_scatter_q4() {
        let (kvh, hd, ms, pos) = (2usize, QK4_0, 64usize, 16usize); // head_dim=QK4_0 → 1 block/pos
        let fill = |c: &mut KVCache| {
            for p in 0..pos {
                for h in 0..kvh {
                    let kvals: Vec<f32> = (0..hd)
                        .map(|d| ((p * kvh + h) * hd + d) as f32 * 0.05 - 1.0)
                        .collect();
                    let vvals: Vec<f32> = (0..hd)
                        .map(|d| ((p * kvh + h) * hd + d) as f32 * -0.03 + 2.0)
                        .collect();
                    write_k_q4(c, p, h, &kvals);
                    write_v_q4(c, p, h, &vvals);
                }
            }
        };
        let mut a = make_cache_q4(ms, kvh, hd, pos);
        let mut b = make_cache_q4(ms, kvh, hd, pos);
        fill(&mut a);
        fill(&mut b);
        let (passing, matches, merge_e) = parity_passing_matches();
        scatter_reduce_merge_layer_wide(&mut a, &passing, &matches, kvh, hd, merge_e);
        crate::kv::standard_format::apply_weighted_merges(
            &mut b,
            &merges_from(&passing, &matches, merge_e),
        );
        let (ka, va) = valid_bytes(&a, kvh, hd);
        let (kb, vb) = valid_bytes(&b, kvh, hd);
        assert_eq!(ka, kb, "Q4_0 K bit-identical");
        assert_eq!(va, vb, "Q4_0 V bit-identical");
    }

    #[test]
    fn test_quantize_round_trip() {
        let src: [f32; QK4_0] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.5);
        let block = BlockQ4_0::quantize(&src);
        let mut dst = [0.0f32; QK4_0];
        block.dequantize(&mut dst);
        // Q4_0 has limited precision; check within tolerance
        for i in 0..QK4_0 {
            assert!(
                (src[i] - dst[i]).abs() < 1.5,
                "round-trip drift too large at {i}: src={}, dst={}",
                src[i],
                dst[i]
            );
        }
    }

    #[test]
    fn test_q4_dequantize_k() {
        let mut cache = make_cache_q4(10, 1, 64, 3);
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        write_k_q4(&mut cache, 1, 0, &values);
        let read = read_k_q4(&cache, 1, 0, 64);
        // Verify dequantize produces values close to originals
        for i in 0..64 {
            assert!(
                (values[i] - read[i]).abs() < 0.5,
                "mismatch at {i}: wrote={}, read={}",
                values[i],
                read[i]
            );
        }
    }

    #[test]
    fn test_q4_find_nearest_layer_wide() {
        // head_dim must be multiple of QK4_0=32, use 32
        let mut cache = make_cache_q4(10, 1, 32, 3);
        let v0: Vec<f32> = (0..32).map(|i| 1.0 + i as f32 * 0.1).collect();
        let v1: Vec<f32> = (0..32).map(|i| -1.0 - i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..32).map(|i| 1.1 + i as f32 * 0.1).collect();
        write_k_q4(&mut cache, 0, 0, &v0);
        write_k_q4(&mut cache, 1, 0, &v1);
        write_k_q4(&mut cache, 2, 0, &v2);

        let retain = vec![0, 1];
        let m = find_nearest_layer_wide(&cache, 2, &retain, 1, 32);
        assert_eq!(m.retain_pos, 0, "pos 2 should be nearest to pos 0");
        assert!(m.sim > 0.9, "similarity should be high, got {}", m.sim);
    }

    #[test]
    fn test_q4_scatter_reduce() {
        let mut cache = make_cache_q4(10, 1, 32, 3);
        let v_retain: Vec<f32> = (0..32).map(|i| 2.0 + i as f32 * 0.1).collect();
        let v_evict: Vec<f32> = (0..32).map(|i| 4.0 + i as f32 * 0.1).collect();
        write_k_q4(&mut cache, 0, 0, &v_retain);
        write_k_q4(&mut cache, 1, 0, &v_evict);

        // sim=1.0, e=0.1 → exp(1) ≈ 2.7183, D = 2.8183, w_c ≈ 0.0355, w_e ≈ 0.9645
        // → merged ≈ heavy weight on evict
        let sim = 1.0f32;
        let e = 0.1f32;
        let matches = vec![Match { retain_pos: 0, sim }];
        let passing = vec![1usize];

        scatter_reduce_merge_layer_wide(&mut cache, &passing, &matches, 1, 32, e);

        let merged = read_k_q4(&cache, 0, 0, 32);

        let exp_sim = sim.exp();
        let denom = exp_sim + e;
        let w_c = e / denom;
        let w_e = exp_sim / denom;
        // Compute analytic expected per-element mean
        let expected_avg: f32 = (0..32)
            .map(|i| {
                let r = 2.0 + i as f32 * 0.1;
                let ev = 4.0 + i as f32 * 0.1;
                w_c * r + w_e * ev
            })
            .sum::<f32>()
            / 32.0;
        let actual_avg: f32 = merged.iter().sum::<f32>() / 32.0;
        // Q4_0 quant noise tolerance: well within ±1.0 even after merge.
        assert!(
            (expected_avg - actual_avg).abs() < 1.0,
            "Q4_0 merge average drift too large: expected={expected_avg}, actual={actual_avg}"
        );
        // Sanity: w_e ≫ w_c when sim=1.0, so merged ≈ evict (≈ 4 + i*0.1 region)
        assert!(w_e > 0.9, "w_e should dominate at sim=1.0, got {w_e}");
    }

    #[test]
    fn test_handler_q4_evicts() {
        // Full handler integration with Q4_0 cache
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_beta: 0.5,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });

        let mut cache = make_cache_q4(50, 1, 32, 20);
        // Write distinct K vectors so similarity is meaningful
        for pos in 0..20 {
            let v: Vec<f32> = (0..32).map(|d| ((pos * 32 + d) as f32) * 0.01).collect();
            write_k_q4(&mut cache, pos, 0, &v);
        }

        let importance: Vec<f32> = (0..50).map(|i| 20.0 - i as f32).collect();
        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        handler.handle(&mut ctx).unwrap();

        let new_pos = ctx.caches[0].current_pos;
        // target = 20 * 0.5 = 10
        assert!(
            new_pos <= 10,
            "Q4_0 eviction should reduce pos, got {new_pos}"
        );
        assert!(new_pos >= 2, "prefix should be preserved");
    }

    // ── Layout coverage tests ──

    #[test]
    fn test_handler_evicts_head_major() {
        // Full handler integration with HeadMajor layout.
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_beta: 0.5,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });

        let mut cache = make_cache_head_major(50, 2, 4, 30);
        for pos in 0..30 {
            let angle = pos as f32 * 0.2;
            for h in 0..2 {
                write_k(
                    &mut cache,
                    pos,
                    h,
                    &[angle.cos(), angle.sin(), h as f32 * 0.1, 0.0],
                );
                write_v(&mut cache, pos, h, &[pos as f32, h as f32, 0.0, 0.0]);
            }
        }

        let mut importance = vec![0.1; 50];
        importance[..5].fill(10.0);
        importance[25..30].fill(5.0);

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted {
                tokens_removed,
                new_pos,
            } => {
                assert!(tokens_removed > 0, "should have removed tokens");
                assert!(new_pos < 30, "new_pos should be less than original");
                assert!(new_pos >= 2, "should keep at least prefix");
                assert_eq!(new_pos, 15, "should reduce to target (30*0.5)");
            }
            other => panic!("Expected Evicted, got {:?}", other),
        }

        // Verify prefix tokens preserved for both heads
        // pos=0: angle = 0.0, so K[0] = cos(0.0) = 1.0
        for h in 0..2 {
            let k0 = read_k(&ctx.caches[0], 0, h, 4);
            assert!(
                (k0[0] - 1.0).abs() < 1e-4,
                "head {h} prefix pos 0 K[0] should be cos(0)=1.0, got {}",
                k0[0]
            );
        }
    }

    #[test]
    fn test_handler_layout_equivalence() {
        // SeqMajor and HeadMajor should produce identical eviction results
        // (same tokens_removed, same new_pos) given the same logical data.
        let make_config = || D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_beta: 0.5,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        };

        let mut importance = vec![0.1; 50];
        importance[..5].fill(10.0);
        importance[15..20].fill(5.0);

        // Populate identical logical content into both layouts
        let populate = |cache: &mut KVCache| {
            for pos in 0..20 {
                let angle = pos as f32 * 0.3;
                write_k(cache, pos, 0, &[angle.cos(), angle.sin(), 0.0, 0.0]);
                write_v(cache, pos, 0, &[pos as f32, 0.0, 0.0, 0.0]);
            }
        };

        // --- SeqMajor ---
        let handler_seq = D2OHandler::new(make_config());
        let mut cache_seq = make_cache(50, 1, 4, 20);
        populate(&mut cache_seq);
        let mut caches_seq = vec![cache_seq];
        let mut ctx_seq = HandlerContext {
            caches: &mut caches_seq,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };
        let result_seq = handler_seq.handle(&mut ctx_seq).unwrap();

        // --- HeadMajor ---
        let handler_hm = D2OHandler::new(make_config());
        let mut cache_hm = make_cache_head_major(50, 1, 4, 20);
        populate(&mut cache_hm);
        let mut caches_hm = vec![cache_hm];
        let mut ctx_hm = HandlerContext {
            caches: &mut caches_hm,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };
        let result_hm = handler_hm.handle(&mut ctx_hm).unwrap();

        // Both should produce Evicted with matching stats
        match (&result_seq, &result_hm) {
            (
                ActionResult::Evicted {
                    tokens_removed: rem_s,
                    new_pos: pos_s,
                },
                ActionResult::Evicted {
                    tokens_removed: rem_h,
                    new_pos: pos_h,
                },
            ) => {
                assert_eq!(rem_s, rem_h, "tokens_removed must match across layouts");
                assert_eq!(pos_s, pos_h, "new_pos must match across layouts");
            }
            _ => panic!(
                "Expected Evicted for both, got seq={:?} hm={:?}",
                result_seq, result_hm
            ),
        }

        // Verify the compacted K values match logically
        let final_pos = ctx_seq.caches[0].current_pos;
        for pos in 0..final_pos {
            let k_seq = read_k(&ctx_seq.caches[0], pos, 0, 4);
            let k_hm = read_k(&ctx_hm.caches[0], pos, 0, 4);
            for d in 0..4 {
                assert!(
                    (k_seq[d] - k_hm[d]).abs() < 1e-4,
                    "K mismatch at pos={pos} dim={d}: seq={} hm={}",
                    k_seq[d],
                    k_hm[d],
                );
            }
        }
    }

    // ── Layer protection tests ──

    #[test]
    fn test_layer_protection() {
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_beta: 0.5,
            merge_e: 0.1,
            use_layer_allocation: true,
            protected_layers: vec![0, 1],
            merge_axis: technique_api::MergeAxis::Both,
        });

        // 4 layers, all at pos 20
        let mut caches = make_caches(4, 20);
        let importance: Vec<f32> = (0..20).map(|i| i as f32).collect();

        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: Some(0.5),
            qcf_sink: None,
            layer_ratios: None,
        };

        handler.handle(&mut ctx).unwrap();

        // Layers 0, 1 are protected (from config)
        // Layer 3 is protected (last layer, use_layer_allocation=true)
        // Only layer 2 should be evicted
        assert_eq!(ctx.caches[0].current_pos, 20, "Layer 0 should be protected");
        assert_eq!(ctx.caches[1].current_pos, 20, "Layer 1 should be protected");
        assert!(ctx.caches[2].current_pos < 20, "Layer 2 should be evicted");
        assert_eq!(
            ctx.caches[3].current_pos, 20,
            "Layer 3 (last) should be protected"
        );
    }

    #[test]
    fn test_per_layer_different_budgets() {
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_beta: 0.5,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: vec![],
            merge_axis: technique_api::MergeAxis::Both,
        });

        // 3 layers, all at pos 40
        let mut caches = make_caches(3, 40);
        // Write distinct K vectors so similarity calculations work
        for cache in caches.iter_mut() {
            for pos in 0..40 {
                let angle = pos as f32 * 0.2;
                let off = cache.offset(pos, 0);
                let k = cache.k_buffer.as_mut_slice::<f32>();
                k[off] = angle.cos();
                k[off + 1] = angle.sin();
                k[off + 2] = 0.0;
                k[off + 3] = 0.0;
            }
        }
        let importance: Vec<f32> = (0..40).map(|i| i as f32).collect();

        // Different ratios per layer: layer 0 keeps 80%, layer 1 keeps 50%, layer 2 keeps 20%
        let layer_ratios = vec![(0.6, 0.2), (0.375, 0.125), (0.15, 0.05)];

        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: Some(0.5),
            qcf_sink: None,
            layer_ratios: Some(&layer_ratios),
        };

        handler.handle(&mut ctx).unwrap();

        // Layer 0 should keep more tokens than layer 2
        assert!(
            ctx.caches[0].current_pos > ctx.caches[2].current_pos,
            "Layer 0 (high budget) should keep more tokens than layer 2 (low budget): {} vs {}",
            ctx.caches[0].current_pos,
            ctx.caches[2].current_pos,
        );
    }

    // ── GPU-only buffer fallback tests ──

    #[test]
    fn test_merge_disabled_evicts_without_crash() {
        // When merge_enabled=false, evict_and_merge should perform H2O-style
        // score-based eviction without attempting to read buffer contents for
        // cosine similarity computation.
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.5,
            protected_prefix: 2,
            target_ratio: 0.5,
            ..D2OConfig::default()
        });
        let mut state = D2OState::new();
        let mut cache = make_cache(100, 1, 4, 20);

        // Write some K values (would be used by merge if enabled)
        for pos in 0..20 {
            write_k(&mut cache, pos, 0, &[pos as f32, 0.0, 0.0, 0.0]);
        }

        let importance: Vec<f32> = (0..20).map(|i| 20.0 - i as f32).collect();

        let removed = handler
            .evict_and_merge(&mut cache, 10, &importance, &mut state, false)
            .unwrap();

        // Should have evicted tokens
        assert!(
            removed > 0,
            "should evict some tokens, got removed={removed}"
        );
        assert_eq!(cache.current_pos, 10, "cache should be compacted to target");

        // With merge disabled, all evicted tokens counted as deleted (not merged)
        assert_eq!(state.total_merged, 0, "no tokens should be merged");
        assert_eq!(
            state.total_deleted, removed,
            "all evicted tokens should be deleted"
        );

        // EMA state should NOT be initialized (merge path was skipped)
        assert!(
            !state.initialized,
            "EMA should not be initialized when merge is disabled"
        );
    }

    #[test]
    fn test_merge_disabled_via_handle_with_cpu_buffers() {
        // CPU buffers have non-null as_ptr, so merge_enabled should be true.
        // This verifies the detection logic in handle() — CPU buffers should
        // still get full D2O behavior (merge + evict).
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.5,
            protected_prefix: 2,
            target_ratio: 0.5,
            ..D2OConfig::default()
        });

        let mut caches = make_caches(2, 20);
        for cache in caches.iter_mut() {
            for pos in 0..20 {
                write_k(cache, pos, 0, &[1.0, 0.0, 0.0, 0.0]);
            }
        }

        let importance: Vec<f32> = (0..20).map(|i| 20.0 - i as f32).collect();

        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: Some(0.5),
            qcf_sink: None,
            layer_ratios: None,
        };

        let result = handler.handle(&mut ctx).unwrap();

        // CPU buffers → merge_enabled=true → full D2O behavior
        match result {
            ActionResult::Evicted { tokens_removed, .. } => {
                assert!(tokens_removed > 0, "should evict tokens");
            }
            _ => panic!("expected Evicted result"),
        }

        // With CPU buffers and identical vectors (sim=1.0), merge should have happened
        let state = handler.state.lock().unwrap();
        assert!(
            state.total_merged > 0,
            "CPU buffers should allow merge: merged={}, deleted={}",
            state.total_merged,
            state.total_deleted,
        );
    }
}
