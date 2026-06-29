//! EvictionHook: StepHook implementation for budget-based KV cache eviction.
//!
//! Encapsulates the eviction logic previously embedded in `run_eval_ll` (generate.rs).
//! Supports both score-based and position-based eviction policies,
//! and collects QCF/value-aware metrics at each eviction event.

use super::hook::{CacheSnapshot, StepHook};
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::kv::cache_manager::CacheManager;
use crate::kv::kv_cache::{KVCache, max_cache_pos};
use crate::qcf::{QcfKvParams, VDataSource, compute_c1, compute_d7, compute_qcf_kv};
use crate::qcf_types::{AggregationMode, QcfConfig, aggregate_heads};
use argus_extension_api::{StageParams, find_qcf_estimator};

/// QCF result from the single post-prefill eviction event (eval-ll mode).
#[derive(Debug, Clone)]
pub struct EvictionQcfResult {
    pub tokens_evicted: usize,
    pub eviction_ratio: f32,
    pub qcf_value_aware: f32,
}

/// QCF record schema v3 payload — cross-family unified (Eviction + quant-window).
///
/// Per-layer worst-head and mean-head series of `‖ΔO_h‖₂ / ‖O_h‖₂`, plus
/// binary pre-computed record-level scalars and D7 / C1 dispersion metrics
/// used by EuroSys'27 §3.
#[derive(Debug, Clone, Default)]
pub struct ExpQcfV3 {
    /// Per-layer worst-head value: `max_h (qcf^h_l)`.
    pub layer_worst_head: Vec<f32>,
    /// Per-layer mean-head value: `mean_h (qcf^h_l)`.
    pub layer_mean_head: Vec<f32>,
    /// `max_l layer_worst_head`.
    pub record_worst_head_max: f32,
    /// `mean_l layer_worst_head`.
    pub record_worst_head_mean: f32,
    /// `max_l layer_mean_head`.
    pub record_mean_head_max: f32,
    /// `mean_l layer_mean_head`.
    pub record_mean_head_mean: f32,
    /// D7 dispersion ratio computed on `layer_worst_head`.
    pub d7_worst_head: f32,
    /// D7 dispersion ratio computed on `layer_mean_head`.
    pub d7_mean_head: f32,
    /// C1 = D7 + population std, computed on `layer_worst_head`.
    pub c1_worst_head: f32,
    /// C1 = D7 + population std, computed on `layer_mean_head`.
    pub c1_mean_head: f32,
}

/// KV cache snapshot for save/restore between multi-token choice scoring.
///
/// Stores raw byte copies of K and V buffers for each layer, along with
/// their `current_pos` counters. Supports both CPU and GPU (OpenCL) buffers.
pub struct KVCacheSnapshot {
    /// Per-layer raw bytes: K buffer followed immediately by V buffer.
    data: Vec<Vec<u8>>,
    /// Per-layer K buffer size at snapshot time (used for K/V split in restore).
    k_sizes: Vec<usize>,
    /// Backend reference for GPU read/write operations.
    backend: std::sync::Arc<dyn crate::backend::Backend>,
    /// Per-layer `current_pos` values.
    positions: Vec<usize>,
}

impl CacheSnapshot<KVCache> for KVCacheSnapshot {
    fn restore_to(&self, caches: &mut [KVCache]) {
        for (i, cache) in caches.iter_mut().enumerate() {
            // Use snapshot-time buffer sizes, not current sizes.
            // Cache may have grown/shrunk between snapshot and restore.
            let snap_k_size = self.k_sizes[i];
            let snap_v_size = self.data[i].len() - snap_k_size;

            // If cache grew since snapshot, the current buffer is larger — write only
            // snapshot-sized data (snap_k_size bytes). Extra bytes are harmless garbage.
            // If cache shrunk since snapshot (shouldn't happen in eval-ll flow), skip
            // write to avoid buffer overrun — the cache will be reset next question anyway.

            let cur_k_size = cache.k_buffer.buffer().size();
            let cur_v_size = cache.v_buffer.buffer().size();
            let k_ptr = cache.k_buffer.buffer().as_mut_ptr();

            if !k_ptr.is_null() {
                // CPU path: direct memcpy (copy min of snapshot and current sizes)
                let k_copy = snap_k_size.min(cur_k_size);
                let v_copy = snap_v_size.min(cur_v_size);
                unsafe {
                    std::ptr::copy_nonoverlapping(self.data[i].as_ptr(), k_ptr, k_copy);
                    std::ptr::copy_nonoverlapping(
                        self.data[i].as_ptr().add(snap_k_size),
                        cache.v_buffer.buffer().as_mut_ptr(),
                        v_copy,
                    );
                }
            } else {
                // GPU path: write_buffer requires exact size match.
                // If cache grew since snapshot, pad with zeros to match current buffer size.
                if snap_k_size == cur_k_size {
                    let _ = self
                        .backend
                        .write_buffer(&mut cache.k_buffer, &self.data[i][..snap_k_size]);
                } else {
                    let mut padded = vec![0u8; cur_k_size];
                    let copy_len = snap_k_size.min(cur_k_size);
                    padded[..copy_len].copy_from_slice(&self.data[i][..copy_len]);
                    let _ = self.backend.write_buffer(&mut cache.k_buffer, &padded);
                }
                if snap_v_size == cur_v_size {
                    let _ = self
                        .backend
                        .write_buffer(&mut cache.v_buffer, &self.data[i][snap_k_size..]);
                } else {
                    let mut padded = vec![0u8; cur_v_size];
                    let copy_len = snap_v_size.min(cur_v_size);
                    padded[..copy_len]
                        .copy_from_slice(&self.data[i][snap_k_size..snap_k_size + copy_len]);
                    let _ = self.backend.write_buffer(&mut cache.v_buffer, &padded);
                }
            }
            cache.current_pos = self.positions[i];
            cache.high_water_pos = self.positions[i];
        }
    }
}

/// StepHook for budget-based eviction (eviction eval-ll mode).
///
/// After each decode step, checks whether `kv_caches[0].current_pos > effective_budget`.
/// When over budget:
/// - heavy-hitter / score-based: calls `force_evict_with_scores` with identified evicted tokens,
///   computes `eviction_attn` (and optionally `eviction_caote`) QCF metrics.
/// - Sliding / position-based: calls `force_evict`, computes `sliding_attn`
///   (and optionally `sliding_caote`) QCF metrics.
///
/// After eviction, the score accumulator is reset.
pub struct EvictionHook {
    /// KV cache manager (wraps the eviction policy).
    pub cache_manager: CacheManager,
    /// Attention score accumulator for heavy-hitter scoring (Some iff score-based).
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    /// QCF metric collection config.
    pub qcf_config: QcfConfig,
    /// Maximum KV cache tokens before eviction triggers.
    pub effective_budget: usize,
    /// Number of prefix tokens protected from eviction.
    pub protected_prefix: usize,
    /// Whether to use score-based eviction (vs. positional sliding).
    pub score_based_eviction: bool,
    /// Keep ratio (fraction of non-prefix tokens kept as heavy hitters).
    pub h2o_keep_ratio: f32,
    /// Whether to use weighted-merge merge compensation (vs. plain heavy-hitter eviction) for QCF-value-aware.
    pub produces_merge_plan: bool,
    /// KV cache dtype string for QCF gating (only "f32" collects QCF).
    pub kv_type: String,
    /// Backend reference for GPU buffer read/write in snapshot/restore.
    pub backend: std::sync::Arc<dyn crate::backend::Backend>,
    /// Whether to compute and dump experimental QCF metrics (ARGUS).
    pub experimental_enabled: bool,
    /// Sample layer indices for multi-layer QCF (ARGUS #1).
    /// Empty → use [0] for backward compat.
    pub qcf_sample_layers: Vec<usize>,

    /// Whether to capture the IMP-1 `evict_importance` dump snapshot at eviction.
    /// Off by default → no capture, eviction path byte-identical (`INV-147`).
    dump_evict_importance: bool,

    /// `--evict-timing prefill_streaming` (variant b): cap the resident cache at
    /// `effective_budget` and evict per-overflow during token-by-token prefill (via
    /// [`on_prefill_step`](StepHook::on_prefill_step)), instead of one cut at
    /// `post_prefill`. False (the default) leaves `post_prefill` as the sole trigger,
    /// keeping the other two timings byte-identical (`INV-147`).
    streaming_overflow: bool,

    // -- Statistics (reset per question) --
    /// Number of eviction events this question.
    eviction_count: usize,
    /// Total tokens evicted this question.
    evicted_total: usize,
    /// QCF result from the single post-prefill eviction event (eval-ll mode).
    eviction_qcf: Option<EvictionQcfResult>,
    /// Experimental QCF payload (Some when experimental_enabled and prefill happened).
    experimental_qcf: Option<ExpQcfV3>,
    /// IMP-1 dump snapshot captured at the most recent eviction (drained by the loop).
    last_evict_dump: Option<crate::session::eval::dump::EvictImportanceSnapshot>,

    /// Streaming (variant b) original-index map: original prompt token index of each
    /// resident cache slot, in slot order (`len == resident cache_pos`). Rebuilt across
    /// eviction-driven compactions so the multi-event dump stays in original token
    /// space. Maintained only while `streaming_overflow && dump_evict_importance`.
    resident_orig: Vec<usize>,
    /// Streaming per-event dump snapshots, drained by the loop after prefill.
    streaming_dumps: Vec<crate::session::eval::dump::EvictImportanceSnapshot>,
}

/// Streaming low-water mark: on overflow, evict down to `floor(budget * KEEP)` so
/// eviction fires in batches (every `~budget * (1 - KEEP)` tokens) instead of
/// re-triggering on every subsequent token — mirroring how an h2o decode eviction
/// drops a block. Tunable here as a single constant (the request leaves the exact
/// low-water as an implementation choice; `cache_pos` stays provably bounded by
/// `budget` (+ one step's slack) for any value in `(0, 1)`).
const STREAMING_LOW_WATER_KEEP: f32 = 0.9;

/// Resident low-water target for a streaming overflow eviction: `floor(budget * KEEP)`,
/// floored at `protected_prefix` and `1`, capped at `budget`. Pure, so the
/// bounded-residency property is testable without a model.
fn streaming_low_water_target(budget: usize, protected_prefix: usize) -> usize {
    let t = ((budget as f32) * STREAMING_LOW_WATER_KEEP).floor() as usize;
    t.max(protected_prefix).max(1).min(budget)
}

/// Map a keep-set (slot indices into the pre-eviction resident cache) back to **original**
/// token indices, returning `(kept_original, evicted_original, compacted_map)`.
///
/// `resident_positions[s]` is the original prompt token index currently at slot `s`.
/// `kept_slots` is the policy's keep-set in ascending slot order (as the engine compacts
/// it). The compacted map equals `kept_original` — the new resident slots hold exactly the
/// kept tokens in ascending slot order, so their original indices stay ascending. Pure, so
/// the original-index translation the streaming dump relies on (and the cross-event map
/// carry-over) is testable without a cache or model.
fn map_keepset_to_original(
    resident_positions: &[usize],
    kept_slots: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = resident_positions.len();
    let mut kept_flag = vec![false; n];
    for &s in kept_slots {
        if s < n {
            kept_flag[s] = true;
        }
    }
    let kept: Vec<usize> = kept_slots
        .iter()
        .filter_map(|&s| resident_positions.get(s).copied())
        .collect();
    let evicted: Vec<usize> = (0..n)
        .filter(|&s| !kept_flag[s])
        .map(|s| resident_positions[s])
        .collect();
    let compacted = kept.clone();
    (kept, evicted, compacted)
}

impl EvictionHook {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cache_manager: CacheManager,
        score_accumulator: Option<AttentionScoreAccumulator>,
        qcf_config: QcfConfig,
        effective_budget: usize,
        protected_prefix: usize,
        score_based_eviction: bool,
        h2o_keep_ratio: f32,
        produces_merge_plan: bool,
        kv_type: String,
        backend: std::sync::Arc<dyn crate::backend::Backend>,
        experimental_enabled: bool,
        qcf_sample_layers: Vec<usize>,
        dump_evict_importance: bool,
        streaming_overflow: bool,
    ) -> Self {
        Self {
            cache_manager,
            score_accumulator,
            qcf_config,
            effective_budget,
            protected_prefix,
            score_based_eviction,
            h2o_keep_ratio,
            produces_merge_plan,
            kv_type,
            backend,
            experimental_enabled,
            qcf_sample_layers,
            dump_evict_importance,
            streaming_overflow,
            eviction_count: 0,
            evicted_total: 0,
            eviction_qcf: None,
            experimental_qcf: None,
            last_evict_dump: None,
            resident_orig: Vec::new(),
            streaming_dumps: Vec::new(),
        }
    }

    /// Assemble the IMP-1 `evict_importance` snapshot from the accumulator's
    /// (now non-collapsed) importance + the technique-agnostic captured keep-set.
    /// Must be called AFTER `force_evict` (so the keep-set is captured) and BEFORE
    /// `acc.reset()` (which wipes the importance buffers). Returns `None` if the
    /// per-layer-head buffer or the keep-set is unavailable.
    fn assemble_evict_importance(
        &self,
        before_len: usize,
        n_layers: usize,
    ) -> Option<crate::session::eval::dump::EvictImportanceSnapshot> {
        use crate::session::eval::dump;

        let captured = crate::kv::eviction::keepset_dump::drain_capture();
        let acc = self.score_accumulator.as_ref()?;
        let layer_head = acc.layer_head_importance()?; // None unless GQA + dump enabled
        let max_seq_len = acc.importance_scores().len();
        let n_kv_heads = acc.n_kv_heads();
        let prompt_len = before_len.min(max_seq_len);

        // Keep-set is technique-agnostic and (for the LayerWide eval path) uniform
        // across layers/heads — take the first captured layer's first head's list.
        let kept_positions: Vec<usize> = captured
            .first()
            .and_then(|c| c.keep.first())
            .cloned()
            .unwrap_or_default();
        let evicted_positions = dump::complement_positions(&kept_positions, prompt_len);

        let importance_flat = acc.importance_scores()[..prompt_len].to_vec();
        let importance_by_layer_head =
            dump::reshape_layer_head(layer_head, n_layers, n_kv_heads, max_seq_len, prompt_len);

        Some(dump::EvictImportanceSnapshot {
            prompt_len,
            budget: self.effective_budget,
            keep_ratio: self.h2o_keep_ratio,
            // R1: the dump groups by technique, so record the stable policy
            // identity ("h2o"), not the pressure-level-decorated log descriptor
            // (`policy_name()` → "h2o@Warning").
            technique: self.cache_manager.policy_id(),
            kept_positions,
            evicted_positions,
            importance_flat,
            importance_by_layer_head,
            // Single-shot record (schema v1): no per-event metadata.
            event: None,
        })
    }

    /// Sync GPU-accumulated attention scores into the CPU accumulator before an
    /// eviction reads them. No-op on CPU backends / without the `opencl` feature, so
    /// the host path is unchanged. Shared by `post_prefill` and the streaming path.
    fn sync_gpu_scores_to_cpu(&mut self) {
        // Shared score-fed body (disjoint field borrows: `score_accumulator` mut, `backend` shared).
        if let Some(acc) = self.score_accumulator.as_mut() {
            crate::kv::eviction::score_fed::sync_gpu_scores_to_cpu(acc, self.backend.as_ref());
        }
    }

    /// Reset the GPU per-`(layer, token)` FLAT cumulative buffer at an eviction boundary, in lockstep
    /// with the CPU `acc.reset()` (faithful-H2O `(b)`). The GPU per-layer reduce accumulates on-device
    /// and is otherwise never reset, so without this a 2nd eviction would rank GPU per-layer importance
    /// monotonically accumulated since prefill (misaligned with the compacted cache) while the CPU twin
    /// starts fresh — they would diverge. No-op on CPU backends / when per-layer is not armed; touches
    /// ONLY the per-layer buffer (the collapsed GPU buffers keep their existing behavior).
    fn reset_gpu_layer_flat(&self) {
        if let Some(acc) = self.score_accumulator.as_ref() {
            crate::kv::eviction::score_fed::reset_gpu_layer_flat(acc, self.backend.as_ref());
        }
    }

    /// Streaming (variant b) per-overflow eviction. Evict the resident cache down to
    /// the low-water mark, reusing the **same** decode-path machinery
    /// (`force_evict[_with_scores]`) the single-shot `post_prefill` uses, and (when the
    /// dump is enabled) record one per-event snapshot in original token-index space.
    /// `cache_pos_before` is the resident length at the moment of overflow; `prefill_pos`
    /// is the number of original tokens ingested so far.
    fn streaming_evict(
        &mut self,
        caches: &mut [KVCache],
        cache_pos_before: usize,
        prefill_pos: usize,
    ) {
        let n_layers = caches.len();
        let target = streaming_low_water_target(self.effective_budget, self.protected_prefix);
        // Keep `target` of the `cache_pos_before` resident tokens (force_evict reads a
        // keep-ratio); float floor keeps the survivor count <= target <= budget.
        let ratio = (target as f32 / cache_pos_before as f32).clamp(0.0, 1.0);

        // Arm the technique-agnostic keep-set capture for this event (drained below).
        if self.dump_evict_importance {
            crate::kv::eviction::keepset_dump::arm_capture();
        }
        // GPU score sync before a score-based eviction reads importance (no-op on CPU).
        self.sync_gpu_scores_to_cpu();

        // Shared score-fed body: extract (only when score-based + active) → route (force, ratio).
        use crate::kv::eviction::score_fed;
        let extracted = if self.score_based_eviction {
            self.score_accumulator
                .as_ref()
                .and_then(score_fed::extract_scores)
        } else {
            None
        };
        let result = {
            let (scores, last_attn, per_layer) = extracted
                .as_ref()
                .map(|e| e.as_args())
                .unwrap_or((None, None, None));
            score_fed::route_evict(
                &self.cache_manager,
                caches,
                scores,
                last_attn,
                per_layer,
                true,
                ratio,
            )
        };

        if let Ok(evict_result) = result
            && evict_result.evicted
        {
            self.eviction_count += 1;
            self.evicted_total += evict_result.tokens_removed;
            let cache_pos_after = max_cache_pos(caches);

            if self.dump_evict_importance {
                // Maps the keep-set back to original indices, compacts `resident_orig`,
                // and (when the per-layer-head buffer is available) builds the record.
                if let Some(snap) = self.assemble_streaming_dump(
                    cache_pos_before,
                    cache_pos_after,
                    n_layers,
                    prefill_pos,
                ) {
                    self.streaming_dumps.push(snap);
                }
            }

            // Reset the accumulator so the next event ranks on a fresh, slot-aligned
            // window of importance over the compacted cache (mirrors the single-shot
            // hook's post-evict reset; avoids stale pre-compaction slot importance).
            if let Some(acc) = self.score_accumulator.as_mut() {
                acc.reset();
            }
            // Faithful-H2O (b): reset the GPU per-layer buffer in lockstep (CPU acc.reset above zeroed
            // the CPU twin; the GPU reduce accumulates on-device and is otherwise never reset).
            self.reset_gpu_layer_flat();
        } else if self.dump_evict_importance {
            // No eviction fired — drop the armed capture so it can't leak forward.
            crate::kv::eviction::keepset_dump::disarm_capture();
        }
    }

    /// Map the just-captured keep-set back to original token indices, compact the
    /// `resident_orig` map for subsequent events, and assemble the streaming dump
    /// snapshot. Must run AFTER `force_evict` (keep-set captured) and BEFORE
    /// `acc.reset()`. Returns `None` (and still compacts `resident_orig`) when the
    /// importance / per-(layer,head) buffer is unavailable.
    fn assemble_streaming_dump(
        &mut self,
        cache_pos_before: usize,
        cache_pos_after: usize,
        n_layers: usize,
        prefill_pos: usize,
    ) -> Option<crate::session::eval::dump::EvictImportanceSnapshot> {
        use crate::session::eval::dump;

        let captured = crate::kv::eviction::keepset_dump::drain_capture();
        // LayerWide eval eviction → take layer 0 / head 0's keep-set as the canonical
        // resident set (uniform across layers/heads), mirroring the single-shot path.
        let kept_slots: Vec<usize> = captured
            .first()
            .and_then(|c| c.keep.first())
            .cloned()
            .unwrap_or_default();

        // Pre-compaction slot→original map (length cache_pos_before).
        let resident_positions = std::mem::take(&mut self.resident_orig);

        // Map the keep-set (slot space) back to original token indices.
        let (kept_positions, evicted_positions, compacted) =
            map_keepset_to_original(&resident_positions, &kept_slots);

        // Compact the live map for subsequent events (new resident slots hold the kept
        // tokens in ascending slot order = ascending original indices).
        self.resident_orig = compacted;

        // Importance payload (slot-indexed). Absent per-(layer,head) buffer → no record
        // (matches the single-shot path); the map above is still compacted so later
        // events stay consistent.
        let acc = self.score_accumulator.as_ref()?;
        let layer_head = acc.layer_head_importance()?;
        let max_seq_len = acc.importance_scores().len();
        let n_kv_heads = acc.n_kv_heads();
        let prompt_len = cache_pos_before.min(max_seq_len);
        let importance_flat = acc.importance_scores()[..prompt_len].to_vec();
        let importance_by_layer_head =
            dump::reshape_layer_head(layer_head, n_layers, n_kv_heads, max_seq_len, prompt_len);

        Some(dump::EvictImportanceSnapshot {
            prompt_len,
            budget: self.effective_budget,
            keep_ratio: self.h2o_keep_ratio,
            technique: self.cache_manager.policy_id(),
            kept_positions,
            evicted_positions,
            importance_flat,
            importance_by_layer_head,
            event: Some(dump::EvictEventMeta {
                eviction_event: self.eviction_count,
                prefill_pos,
                cache_pos_before,
                cache_pos_after,
                resident_positions,
            }),
        })
    }
}

impl StepHook<KVCache> for EvictionHook {
    fn post_prefill(&mut self, caches: &mut [KVCache]) {
        // After full batch prefill, evict if cache exceeds budget.
        // This replaces the old chunked-prefill approach that decoded overflow
        // tokens one-by-one (causing 2-3.3x slowdown).
        //
        // Skip when effective_budget == 0 (full-prefill / no-budget mode):
        // ratio = 0/before_len would ask the pipeline for full eviction, and
        // the resulting release_unused_pages → shrink_to_fit reallocation
        // breaks the next question's GPU reads on NVIDIA OpenCL.
        if caches.is_empty()
            || self.effective_budget == 0
            || max_cache_pos(caches) <= self.effective_budget
        {
            return;
        }

        let before_len = max_cache_pos(caches);
        let ratio = self.effective_budget as f32 / before_len as f32;
        let eviction_ratio = 1.0 - ratio;

        // V buffer readback for QCF computation (GPU backends only — CPU buffers are
        // always accessible via as_ptr() and do not need a readback).
        let v_cpu_bytes: Option<Vec<u8>> =
            if !caches.is_empty() && caches[0].v_buffer.buffer().as_ptr().is_null() {
                let size = caches[0].v_buffer.buffer().size();
                let mut buf = vec![0u8; size];
                match self.backend.read_buffer(&caches[0].v_buffer, &mut buf) {
                    Ok(()) => Some(buf),
                    Err(_) => None,
                }
            } else {
                None
            };
        // GPU score sync before QCF computation (eval-ll path).
        // On GPU backends, forward_into() accumulates scores entirely on the device.
        // The CPU accumulator's importance and last_layer_head_attn are empty.
        // We sync both: (1) cumulative importance via import_gpu_scores, and
        // (2) head importance as proxy for last_layer_head_attn (the GPU path
        // doesn't have raw per-step attention weights, but cumulative head
        // importance is proportional and sufficient for QCF computation).
        self.sync_gpu_scores_to_cpu();

        // can_compute_qcf: true when V data is CPU-accessible (CPU backend) or
        // successfully read back (GPU backend). Supports F32, F16, and Q4_0 dtypes.
        let can_compute_qcf =
            v_cpu_bytes.is_some() || !caches[0].v_buffer.buffer().as_ptr().is_null();

        // QCF (unified output-error formula). Action picks the simulated retention.
        let qcf_value_aware = if can_compute_qcf {
            let cache = &caches[0];
            let v_source = VDataSource::from_buffer(&cache.v_buffer, v_cpu_bytes.as_deref())
                .unwrap_or_else(|| {
                    // fallback: treat as F32 (may be incorrect for unknown dtypes)
                    VDataSource::F32(cache.v_buffer.as_slice::<f32>())
                });
            let target_len = ((before_len as f32) * ratio) as usize;
            // Resolve the estimator by name (d2o/h2o when score-based, else sliding); the engine no
            // longer enumerates techniques here.
            let (est_name, est_params) = if self.score_based_eviction {
                let sp = StageParams {
                    keep_ratio: self.h2o_keep_ratio,
                    protected_prefix: self.protected_prefix,
                    ..Default::default()
                };
                if self.produces_merge_plan {
                    ("d2o", sp)
                } else {
                    ("h2o", sp)
                }
            } else {
                ("sliding", StageParams::default())
            };
            let estimator = (find_qcf_estimator(est_name)
                .expect("eviction QCF estimator registered")
                .make)(est_params, &[]);
            let attention_scores: Vec<f32> = self
                .score_accumulator
                .as_ref()
                .filter(|acc| acc.is_active())
                .map(|acc| acc.importance_scores().to_vec())
                .unwrap_or_default();
            let head_attn_opt = self
                .score_accumulator
                .as_ref()
                .and_then(|acc| acc.last_step_head_attn());
            // The merge simulator (cosine-nearest matching) needs K for nearest-neighbour matching; other techniques
            // ignore `k_source` (their estimators never call read_k).
            let k_source = if self.produces_merge_plan {
                VDataSource::from_buffer(&cache.k_buffer, None)
            } else {
                None
            };
            let params = QcfKvParams {
                estimator: &*estimator,
                target_len,
                v_source,
                k_source,
                attention_scores: &attention_scores,
                head_attn: head_attn_opt,
                n_kv_heads: cache.kv_heads(),
                head_dim: cache.head_dim(),
                current_pos: before_len,
                capacity: cache.capacity(),
                layout: cache.layout(),
                aggregation: AggregationMode::Mean,
                beta: 1.0,
            };
            let (qcf, per_head) = compute_qcf_kv(&params);

            if self.experimental_enabled {
                // Schema v3: per-layer worst-head + mean-head over the sample layers.
                // Layer 0 reuses the `per_head` already computed above.
                let sample_layers: Vec<usize> = if self.qcf_sample_layers.is_empty() {
                    vec![0]
                } else {
                    self.qcf_sample_layers.clone()
                };

                let mut layer_worst_head: Vec<f32> = Vec::with_capacity(sample_layers.len());
                let mut layer_mean_head: Vec<f32> = Vec::with_capacity(sample_layers.len());

                for &layer_idx in &sample_layers {
                    if layer_idx >= caches.len() {
                        continue;
                    }
                    // Layer 0: reuse `per_head` from the scalar call above (no extra readback).
                    let per_head_l: Vec<f32> = if layer_idx == 0 {
                        per_head.clone()
                    } else {
                        // Per-layer V readback (GPU only — CPU buffers accessible via as_ptr).
                        let cache_l = &caches[layer_idx];
                        let v_cpu_bytes_l: Option<Vec<u8>> =
                            if cache_l.v_buffer.buffer().as_ptr().is_null() {
                                let size = cache_l.v_buffer.buffer().size();
                                let mut buf = vec![0u8; size];
                                match self.backend.read_buffer(&cache_l.v_buffer, &mut buf) {
                                    Ok(()) => Some(buf),
                                    Err(_) => None,
                                }
                            } else {
                                None
                            };

                        let can_compute_l = v_cpu_bytes_l.is_some()
                            || !cache_l.v_buffer.buffer().as_ptr().is_null();
                        if !can_compute_l {
                            continue;
                        }

                        let v_source_l = match VDataSource::from_buffer(
                            &cache_l.v_buffer,
                            v_cpu_bytes_l.as_deref(),
                        ) {
                            Some(vs) => vs,
                            None => VDataSource::F32(cache_l.v_buffer.as_slice::<f32>()),
                        };
                        let k_source_l = if self.produces_merge_plan {
                            VDataSource::from_buffer(&cache_l.k_buffer, None)
                        } else {
                            None
                        };
                        let target_len_l = ((cache_l.current_pos as f32) * ratio) as usize;
                        // Same technique as the scalar call above; only target_len differs per layer.
                        let estimator_l = (find_qcf_estimator(est_name)
                            .expect("eviction QCF estimator registered")
                            .make)(est_params, &[]);
                        let params_l = QcfKvParams {
                            estimator: &*estimator_l,
                            target_len: target_len_l,
                            v_source: v_source_l,
                            k_source: k_source_l,
                            attention_scores: &attention_scores,
                            head_attn: head_attn_opt,
                            n_kv_heads: cache_l.kv_heads(),
                            head_dim: cache_l.head_dim(),
                            current_pos: before_len,
                            capacity: cache_l.capacity(),
                            layout: cache_l.layout(),
                            aggregation: AggregationMode::Mean,
                            beta: 1.0,
                        };
                        let (_qcf_l, ph_l) = compute_qcf_kv(&params_l);
                        ph_l
                    };

                    let worst = aggregate_heads(&per_head_l, &AggregationMode::Max);
                    let mean = aggregate_heads(&per_head_l, &AggregationMode::Mean);
                    layer_worst_head.push(worst);
                    layer_mean_head.push(mean);
                }

                // Record-level scalars.
                let max_or_zero = |s: &[f32]| -> f32 {
                    s.iter().copied().fold(f32::NEG_INFINITY, f32::max).max(0.0)
                };
                let mean_or_zero = |s: &[f32]| -> f32 {
                    if s.is_empty() {
                        0.0
                    } else {
                        s.iter().sum::<f32>() / s.len() as f32
                    }
                };
                let record_worst_head_max = if layer_worst_head.is_empty() {
                    0.0
                } else {
                    max_or_zero(&layer_worst_head)
                };
                let record_worst_head_mean = mean_or_zero(&layer_worst_head);
                let record_mean_head_max = if layer_mean_head.is_empty() {
                    0.0
                } else {
                    max_or_zero(&layer_mean_head)
                };
                let record_mean_head_mean = mean_or_zero(&layer_mean_head);

                let payload = ExpQcfV3 {
                    d7_worst_head: compute_d7(&layer_worst_head),
                    d7_mean_head: compute_d7(&layer_mean_head),
                    c1_worst_head: compute_c1(&layer_worst_head),
                    c1_mean_head: compute_c1(&layer_mean_head),
                    layer_worst_head,
                    layer_mean_head,
                    record_worst_head_max,
                    record_worst_head_mean,
                    record_mean_head_max,
                    record_mean_head_mean,
                };
                self.experimental_qcf = Some(payload);
            }

            qcf
        } else {
            0.0
        };

        // IMP-1 evict_importance dump: arm the technique-agnostic keep-set capture
        // around this eviction (drained in `assemble_evict_importance` below).
        let n_layers = caches.len();
        if self.dump_evict_importance {
            crate::kv::eviction::keepset_dump::arm_capture();
        }

        // Perform eviction — shared score-fed body: extract (score-based + active) → route.
        use crate::kv::eviction::score_fed;
        let extracted = if self.score_based_eviction {
            self.score_accumulator
                .as_ref()
                .and_then(score_fed::extract_scores)
        } else {
            None
        };
        let result = {
            let (scores, last_attn, per_layer) = extracted
                .as_ref()
                .map(|e| e.as_args())
                .unwrap_or((None, None, None));
            score_fed::route_evict(
                &self.cache_manager,
                caches,
                scores,
                last_attn,
                per_layer,
                true,
                ratio,
            )
        };

        if let Ok(evict_result) = result
            && evict_result.evicted
        {
            self.eviction_count += 1;
            self.evicted_total += evict_result.tokens_removed;

            // IMP-1: snapshot importance + per-(layer, KV-head) buffer + captured
            // keep-set BEFORE acc.reset() wipes the accumulator.
            if self.dump_evict_importance {
                self.last_evict_dump = self.assemble_evict_importance(before_len, n_layers);
            }

            if let Some(acc) = self.score_accumulator.as_mut() {
                acc.reset();
            }
            // Faithful-H2O (b): reset the GPU per-layer buffer in lockstep with the CPU reset above.
            self.reset_gpu_layer_flat();

            // Store QCF result for extra_question_fields
            self.eviction_qcf = Some(EvictionQcfResult {
                tokens_evicted: evict_result.tokens_removed,
                eviction_ratio,
                qcf_value_aware,
            });
        } else if self.dump_evict_importance {
            // No eviction fired — drop the armed capture so it can't leak into a
            // later event.
            crate::kv::eviction::keepset_dump::disarm_capture();
        }
    }

    fn reset_caches(&mut self, caches: &mut [KVCache]) {
        for cache in caches.iter_mut() {
            cache.current_pos = 0;
            cache.high_water_pos = 0;
        }
        if let Some(acc) = self.score_accumulator.as_mut() {
            acc.reset();
        }
        // Faithful-H2O (b): reset the GPU per-layer buffer in lockstep with the CPU reset above.
        self.reset_gpu_layer_flat();
        self.eviction_count = 0;
        self.evicted_total = 0;
        self.eviction_qcf = None;
        self.experimental_qcf = None;
        self.last_evict_dump = None;
        self.resident_orig.clear();
        self.streaming_dumps.clear();
    }

    fn on_prefill_step(&mut self, caches: &mut [KVCache], orig_token_idx: usize) {
        // Variant b only. The other timings (and the quant-window hook) leave this a
        // no-op, so token-by-token prefill stays byte-identical for them (`INV-147`).
        if !self.streaming_overflow || self.effective_budget == 0 {
            return;
        }
        let cache_pos = max_cache_pos(caches);
        // Track the original prompt index of this just-ingested token at its new slot.
        // Maintained only when the dump needs the original-index map; the eviction
        // decision itself never reads it.
        if self.dump_evict_importance {
            self.resident_orig.push(orig_token_idx);
            debug_assert_eq!(
                self.resident_orig.len(),
                cache_pos,
                "resident_orig must track cache occupancy slot-for-slot"
            );
        }
        if cache_pos <= self.effective_budget {
            return; // within budget — no overflow this step.
        }
        // Overflow: evict down to the low-water mark. `cache_pos` is bounded by
        // `effective_budget + 1` (we check after every single-token ingest).
        self.streaming_evict(caches, cache_pos, orig_token_idx + 1);
    }

    fn snapshot(&self, caches: &[KVCache]) -> Box<dyn CacheSnapshot<KVCache>> {
        let mut data = Vec::with_capacity(caches.len());
        let mut k_sizes = Vec::with_capacity(caches.len());
        let mut positions = Vec::with_capacity(caches.len());
        for cache in caches {
            let k_size = cache.k_buffer.buffer().size();
            let v_size = cache.v_buffer.buffer().size();
            let mut buf = vec![0u8; k_size + v_size];
            let k_ptr = cache.k_buffer.buffer().as_ptr();
            if !k_ptr.is_null() {
                // CPU path: direct memcpy
                unsafe {
                    std::ptr::copy_nonoverlapping(k_ptr, buf.as_mut_ptr(), k_size);
                    std::ptr::copy_nonoverlapping(
                        cache.v_buffer.buffer().as_ptr(),
                        buf.as_mut_ptr().add(k_size),
                        v_size,
                    );
                }
            } else {
                // GPU path: read via OpenCL
                let _ = self
                    .backend
                    .read_buffer(&cache.k_buffer, &mut buf[..k_size]);
                let _ = self
                    .backend
                    .read_buffer(&cache.v_buffer, &mut buf[k_size..]);
            }
            data.push(buf);
            k_sizes.push(k_size);
            positions.push(cache.current_pos);
        }
        Box::new(KVCacheSnapshot {
            data,
            k_sizes,
            positions,
            backend: self.backend.clone(),
        })
    }

    fn set_effective_budget(&mut self, budget: usize) {
        self.effective_budget = budget;
    }

    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator> {
        self.score_accumulator.as_mut()
    }

    fn needs_score_probe(&self, caches: &[KVCache]) -> bool {
        // Probe is needed when cache exceeds budget (eviction will happen).
        // The probe step populates score_accumulator for heavy-hitter decisions and
        // captures last_step_head_attn for QCF-ATTN measurement.
        !caches.is_empty() && max_cache_pos(caches) > self.effective_budget
    }

    fn ranks_on_scores(&self) -> bool {
        // Heavy-hitter / value-aware policies rank on accumulated attention; sliding /
        // positional policies do not. Only the former benefit from a token-by-token
        // prefill pass under `--evict-timing prefill_end`.
        self.score_based_eviction
    }

    fn extra_question_fields(&self, _caches: &[KVCache]) -> serde_json::Value {
        let mut obj = serde_json::json!({
            "effective_budget": self.effective_budget,
            "eviction_count": self.eviction_count,
            "evicted_tokens": self.evicted_total,
        });
        if let Some(ref qcf) = self.eviction_qcf {
            obj["qcf"] = serde_json::json!(qcf.qcf_value_aware);
            obj["tokens_evicted"] = serde_json::json!(qcf.tokens_evicted);
            obj["eviction_ratio"] = serde_json::json!(qcf.eviction_ratio);
        }
        if let Some(ref exp) = self.experimental_qcf {
            obj["schema_version"] = serde_json::json!(3);
            obj["action_family"] = serde_json::json!("eviction");
            obj["n_layers"] = serde_json::json!(exp.layer_worst_head.len());
            obj["qcf_layer_worst_head"] = serde_json::json!(exp.layer_worst_head);
            obj["qcf_layer_mean_head"] = serde_json::json!(exp.layer_mean_head);
            obj["qcf_record_worst_head_max"] = serde_json::json!(exp.record_worst_head_max);
            obj["qcf_record_worst_head_mean"] = serde_json::json!(exp.record_worst_head_mean);
            obj["qcf_record_mean_head_max"] = serde_json::json!(exp.record_mean_head_max);
            obj["qcf_record_mean_head_mean"] = serde_json::json!(exp.record_mean_head_mean);
            obj["qcf_d7_worst_head"] = serde_json::json!(exp.d7_worst_head);
            obj["qcf_d7_mean_head"] = serde_json::json!(exp.d7_mean_head);
            obj["qcf_c1_worst_head"] = serde_json::json!(exp.c1_worst_head);
            obj["qcf_c1_mean_head"] = serde_json::json!(exp.c1_mean_head);
        }
        obj
    }

    fn extra_config_fields(&self) -> serde_json::Value {
        serde_json::json!({
            "effective_budget": self.effective_budget,
            "protected_prefix": self.protected_prefix,
            "score_based_eviction": self.score_based_eviction,
            "h2o_keep_ratio": self.h2o_keep_ratio,
            "is_d2o": self.produces_merge_plan,
            "kv_type": self.kv_type,
            "experimental_enabled": self.experimental_enabled,
        })
    }

    fn take_evict_importance_dump(
        &mut self,
    ) -> Option<crate::session::eval::dump::EvictImportanceSnapshot> {
        self.last_evict_dump.take()
    }

    fn take_streaming_evict_dumps(
        &mut self,
    ) -> Vec<crate::session::eval::dump::EvictImportanceSnapshot> {
        std::mem::take(&mut self.streaming_dumps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::cache_manager::CacheManager;
    use crate::kv::eviction::stage_registry::none_backed_policy;
    use crate::qcf_types::{QcfConfig, QcfMode};
    use crate::resilience::sys_monitor::{MemoryStats, SystemMonitor};
    use anyhow::Result as AResult;

    struct AlwaysOkMonitor;
    impl SystemMonitor for AlwaysOkMonitor {
        fn mem_stats(&self) -> AResult<MemoryStats> {
            Ok(MemoryStats {
                total: usize::MAX,
                available: usize::MAX,
                free: usize::MAX,
            })
        }
    }

    fn make_hook(budget: usize, score_based: bool) -> EvictionHook {
        make_hook_with_d2o(budget, score_based, false)
    }

    fn make_hook_with_d2o(
        budget: usize,
        score_based: bool,
        produces_merge_plan: bool,
    ) -> EvictionHook {
        let policy = none_backed_policy();
        let monitor = Box::new(AlwaysOkMonitor);
        let manager = CacheManager::new(policy, monitor, 0, 1.0);
        let config = QcfConfig {
            mode: QcfMode::Attn,
            ..Default::default()
        };
        EvictionHook::new(
            manager,
            None,
            config,
            budget,
            0,
            score_based,
            0.5,
            produces_merge_plan,
            "f32".to_string(),
            crate::backend::cpu::cpu_singleton(),
            false,
            vec![], // qcf_sample_layers: empty → internal fallback to [0]
            false,  // dump_evict_importance
            false,  // streaming_overflow
        )
    }

    #[test]
    fn dump_technique_is_clean_policy_id_not_log_name() {
        // R1: assemble_evict_importance records `cache_manager.policy_id()` in the
        // dump's `technique`, which must be the bare policy name — NOT the
        // logging descriptor `policy_name()` that folds in the pressure level
        // ("none@Warning"). A real (force-linked "none") policy is fed here, so
        // this exercises the actual plugin-backed name, unlike the hard-coded
        // "h2o" in the dump.rs record test.
        let hook = make_hook(8, true);
        let log_name = hook.cache_manager.policy_name();
        let technique = hook.cache_manager.policy_id();
        assert!(
            log_name.contains('@'),
            "log descriptor keeps the level tag: {log_name}"
        );
        assert_eq!(
            technique, "none",
            "dump technique is the clean id: {technique}"
        );
        assert!(
            !technique.contains('@') && !technique.contains('→'),
            "dump technique must carry no status/level/stage-join decoration"
        );
    }

    #[test]
    fn test_extra_question_fields_initial() {
        let hook = make_hook(512, false);
        let fields = hook.extra_question_fields(&[]);
        assert_eq!(fields["effective_budget"], 512);
        assert_eq!(fields["eviction_count"], 0);
        assert_eq!(fields["evicted_tokens"], 0);
    }

    #[test]
    fn test_extra_config_fields() {
        let hook = make_hook(256, true);
        let fields = hook.extra_config_fields();
        assert_eq!(fields["effective_budget"], 256);
        assert_eq!(fields["score_based_eviction"], true);
        assert_eq!(fields["is_d2o"], false);
        assert_eq!(fields["kv_type"], "f32");
    }

    #[test]
    fn test_extra_config_fields_d2o() {
        let hook = make_hook_with_d2o(256, true, true);
        let fields = hook.extra_config_fields();
        assert_eq!(fields["is_d2o"], true);
        assert_eq!(fields["score_based_eviction"], true);
    }

    #[test]
    fn test_snapshot_empty() {
        let hook = make_hook(512, false);
        let snapshot = hook.snapshot(&[]);
        // Restoring an empty snapshot on empty caches should not panic.
        snapshot.restore_to(&mut []);
    }

    #[test]
    fn test_make_hook_with_experimental_off() {
        // Verifies that experimental_enabled=false is accepted and stored correctly.
        let hook = make_hook_with_d2o(512, false, false);
        assert!(!hook.experimental_enabled);
    }

    #[test]
    fn test_extra_question_fields_no_experimental() {
        // When experimental_qcf is None, extra_question_fields must not contain
        // new experimental keys.
        let hook = make_hook(512, false);
        let fields = hook.extra_question_fields(&[]);
        assert!(
            fields.get("qcf_value_aware_max").is_none(),
            "qcf_value_aware_max should be absent when experimental_qcf is None"
        );
        assert!(
            fields.get("qcf_per_head").is_none(),
            "qcf_per_head should be absent when experimental_qcf is None"
        );
    }

    #[test]
    fn test_extra_config_fields_experimental_enabled() {
        // experimental_enabled field should appear in extra_config_fields.
        let hook = make_hook(256, false);
        let fields = hook.extra_config_fields();
        assert_eq!(
            fields["experimental_enabled"], false,
            "experimental_enabled should be false for default hook"
        );
    }

    #[test]
    fn test_qcf_sample_layers_default_fallback() {
        // Empty qcf_sample_layers → stored as empty vec.
        // Internal fallback to [0] occurs at runtime in post_prefill.
        // Here we verify the field is stored as-is and the hook is created successfully.
        let hook = make_hook_with_d2o(512, false, false);
        assert!(
            hook.qcf_sample_layers.is_empty(),
            "make_hook_with_d2o passes vec![] → qcf_sample_layers should be empty"
        );
    }

    #[test]
    fn test_qcf_sample_layers_explicit() {
        // When explicit layers are provided, they should be stored unchanged.
        let policy = none_backed_policy();
        let monitor = Box::new(AlwaysOkMonitor);
        let manager = CacheManager::new(policy, monitor, 0, 1.0);
        let config = QcfConfig {
            mode: QcfMode::Attn,
            ..Default::default()
        };
        let hook = EvictionHook::new(
            manager,
            None,
            config,
            512,
            0,
            false,
            0.5,
            false,
            "f32".to_string(),
            crate::backend::cpu::cpu_singleton(),
            false,
            vec![0, 4, 8, 12, 15],
            false, // dump_evict_importance
            false, // streaming_overflow
        );
        assert_eq!(hook.qcf_sample_layers, vec![0, 4, 8, 12, 15]);
    }

    // ── variant b: streaming overflow eviction (mechanism, model-free) ──

    #[test]
    fn low_water_target_is_below_budget_and_respects_floors() {
        // floor(budget * 0.9), so each event drops a block instead of re-triggering.
        assert_eq!(streaming_low_water_target(16, 0), 14); // floor(14.4)
        assert_eq!(streaming_low_water_target(256, 0), 230); // floor(230.4)
        assert_eq!(streaming_low_water_target(2, 0), 1); // floor(1.8)
        // protected prefix raises the floor…
        assert_eq!(streaming_low_water_target(16, 15), 15);
        // …but the target never exceeds the budget.
        assert!(streaming_low_water_target(16, 100) <= 16);
        // …and is always at least one resident token.
        assert_eq!(streaming_low_water_target(1, 0), 1);
    }

    /// Acceptance #2 (bounded residency): simulate the token-by-token streaming loop —
    /// ingest one token (cache_pos += 1), and on `cache_pos > B` evict to the low-water
    /// mark exactly as `streaming_evict` does (force_evict keeps `floor(before*ratio)`).
    /// Occupancy must never exceed `B + 1` and a long prompt must overflow more than once.
    #[test]
    fn streaming_residency_is_bounded_by_budget_plus_one_slack() {
        for &budget in &[8usize, 16, 31, 256] {
            let protected = 0;
            let mut cache_pos = 0usize;
            let mut max_seen = 0usize;
            let mut events = 0usize;
            for _ in 0..(budget * 8 + 5) {
                cache_pos += 1; // ingest one token
                max_seen = max_seen.max(cache_pos);
                if cache_pos > budget {
                    let before = cache_pos;
                    let target = streaming_low_water_target(budget, protected);
                    let ratio = target as f32 / before as f32;
                    cache_pos = ((before as f32) * ratio).floor() as usize;
                    events += 1;
                    assert!(
                        cache_pos <= budget,
                        "budget={budget}: post-evict occupancy {cache_pos} must be <= budget"
                    );
                }
            }
            assert_eq!(
                max_seen,
                budget + 1,
                "budget={budget}: resident peaks at exactly one step's slack over budget"
            );
            assert!(
                events >= 2,
                "budget={budget}: a prompt many times the budget overflows repeatedly"
            );
        }
    }

    /// Acceptance #3 (positions in original index space): after a prior compaction the
    /// cache's slot space is reindexed; the keep-set (slot space) must map back to the
    /// original prompt indices, with kept/evicted disjoint and unioning to the resident
    /// set, and the compacted map carried to the next event.
    #[test]
    fn keepset_maps_back_to_original_index_space() {
        // Slots {0,1,2,3} currently hold original tokens {3,7,9,11} (post-compaction).
        let resident = [3usize, 7, 9, 11];
        // Policy keeps slots {0,2,3} (drops slot 1).
        let (kept, evicted, compacted) = map_keepset_to_original(&resident, &[0, 2, 3]);
        assert_eq!(kept, vec![3, 9, 11], "kept slots → original indices");
        assert_eq!(evicted, vec![7], "evicted slot → original index");
        assert_eq!(compacted, vec![3, 9, 11], "next-event map = kept originals");
        // Disjoint, and their union is exactly the resident set.
        assert!(kept.iter().all(|k| !evicted.contains(k)));
        let mut all: Vec<usize> = kept.iter().chain(&evicted).copied().collect();
        all.sort_unstable();
        assert_eq!(all, vec![3, 7, 9, 11]);
        // Original indices stay ascending across the carry-over (join-friendly).
        assert!(compacted.windows(2).all(|w| w[0] < w[1]));
    }

    /// Two compaction rounds: the map composes, so event 2's `kept_positions` are still
    /// in the original prompt space even though it ranks over a twice-reindexed cache.
    #[test]
    fn original_map_composes_across_events() {
        // Event 1 over original [0..6): keep slots {0,2,4,5} → originals {0,2,4,5}.
        let resident0: Vec<usize> = (0..6).collect();
        let (_k1, _e1, after1) = map_keepset_to_original(&resident0, &[0, 2, 4, 5]);
        assert_eq!(after1, vec![0, 2, 4, 5]);
        // Two more original tokens (6, 7) ingested → resident = [0,2,4,5,6,7].
        let mut resident1 = after1;
        resident1.extend([6, 7]);
        // Event 2: keep slots {0,3,4,5} → originals {0,5,6,7}.
        let (k2, e2, _after2) = map_keepset_to_original(&resident1, &[0, 3, 4, 5]);
        assert_eq!(k2, vec![0, 5, 6, 7], "still original prompt indices");
        assert_eq!(e2, vec![2, 4], "evicted-at-event-2 originals");
    }
}
