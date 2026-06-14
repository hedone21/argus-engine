//! EvictionHook: StepHook implementation for budget-based KV cache eviction.
//!
//! Encapsulates the eviction logic previously embedded in `run_eval_ll` (generate.rs).
//! Supports both H2O (score-based) and Sliding (position-based) eviction policies,
//! and collects QCF/CAOTE metrics at each eviction event.

use super::hook::{CacheSnapshot, StepHook};
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::kv::cache_manager::CacheManager;
use crate::kv::kv_cache::{KVCache, max_cache_pos};
use crate::qcf::{
    QcfActionType, QcfKvParams, VDataSource, compute_c1, compute_d7, compute_qcf_kv,
    identify_retained_for_action,
};
use crate::qcf_types::{AggregationMode, QcfConfig, aggregate_heads};

/// QCF result from the single post-prefill eviction event (eval-ll mode).
#[derive(Debug, Clone)]
pub struct EvictionQcfResult {
    pub tokens_evicted: usize,
    pub eviction_ratio: f32,
    pub qcf_caote: f32,
}

/// QCF record schema v3 payload — cross-family unified (Eviction + KIVI).
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
    /// Per-layer capacity at snapshot time.
    #[allow(dead_code)]
    capacities: Vec<usize>,
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
/// - H2O / score-based: calls `force_evict_with_scores` with identified evicted tokens,
///   computes `eviction_attn` (and optionally `eviction_caote`) QCF metrics.
/// - Sliding / position-based: calls `force_evict`, computes `sliding_attn`
///   (and optionally `sliding_caote`) QCF metrics.
///
/// After eviction, the score accumulator is reset.
pub struct EvictionHook {
    /// KV cache manager (wraps the eviction policy).
    pub cache_manager: CacheManager,
    /// Attention score accumulator for H2O scoring (Some iff score-based).
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    /// QCF metric collection config.
    pub qcf_config: QcfConfig,
    /// Maximum KV cache tokens before eviction triggers.
    pub effective_budget: usize,
    /// Number of prefix tokens protected from eviction.
    pub protected_prefix: usize,
    /// Whether to use H2O/D2O score-based eviction (vs. positional sliding).
    pub score_based_eviction: bool,
    /// H2O keep ratio (fraction of non-prefix tokens kept as heavy hitters).
    pub h2o_keep_ratio: f32,
    /// Whether to use D2O merge compensation (vs. plain H2O eviction) for QCF-CAOTE.
    pub is_d2o: bool,
    /// KV cache dtype string for QCF gating (only "f32" collects QCF).
    pub kv_type: String,
    /// Backend reference for GPU buffer read/write in snapshot/restore.
    pub backend: std::sync::Arc<dyn crate::backend::Backend>,
    /// Whether to compute and dump experimental QCF metrics (ARGUS).
    pub experimental_enabled: bool,
    /// Sample layer indices for multi-layer QCF (ARGUS #1).
    /// Empty → use [0] for backward compat.
    pub qcf_sample_layers: Vec<usize>,

    // -- Statistics (reset per question) --
    /// Number of eviction events this question.
    eviction_count: usize,
    /// Total tokens evicted this question.
    evicted_total: usize,
    /// QCF result from the single post-prefill eviction event (eval-ll mode).
    eviction_qcf: Option<EvictionQcfResult>,
    /// Experimental QCF payload (Some when experimental_enabled and prefill happened).
    experimental_qcf: Option<ExpQcfV3>,
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
        is_d2o: bool,
        kv_type: String,
        backend: std::sync::Arc<dyn crate::backend::Backend>,
        experimental_enabled: bool,
        qcf_sample_layers: Vec<usize>,
    ) -> Self {
        Self {
            cache_manager,
            score_accumulator,
            qcf_config,
            effective_budget,
            protected_prefix,
            score_based_eviction,
            h2o_keep_ratio,
            is_d2o,
            kv_type,
            backend,
            experimental_enabled,
            qcf_sample_layers,
            eviction_count: 0,
            evicted_total: 0,
            eviction_qcf: None,
            experimental_qcf: None,
        }
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
        #[cfg(feature = "opencl")]
        if let Some(ref mut acc) = self.score_accumulator
            && acc.is_active()
            && let Some(ocl_be) = self
                .backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            && let Some(gpu_acc) = ocl_be.gpu_score_acc()
            && gpu_acc.is_active()
            && let Ok((flat, head)) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())
        {
            acc.import_gpu_scores(&flat, &head);
        }

        // can_compute_qcf: true when V data is CPU-accessible (CPU backend) or
        // successfully read back (GPU backend). Supports F32, F16, and Q4_0 dtypes.
        let can_compute_qcf =
            v_cpu_bytes.is_some() || !caches[0].v_buffer.buffer().as_ptr().is_null();

        // QCF (unified output-error formula). Action picks the simulated retention.
        let qcf_caote = if can_compute_qcf {
            let cache = &caches[0];
            let v_source = VDataSource::from_buffer(&cache.v_buffer, v_cpu_bytes.as_deref())
                .unwrap_or_else(|| {
                    // fallback: treat as F32 (may be incorrect for unknown dtypes)
                    VDataSource::F32(cache.v_buffer.as_slice::<f32>())
                });
            let target_len = ((before_len as f32) * ratio) as usize;
            let action = if self.score_based_eviction {
                if self.is_d2o {
                    QcfActionType::MergeD2o {
                        target_len,
                        keep_ratio: self.h2o_keep_ratio,
                        protected_prefix: self.protected_prefix,
                    }
                } else {
                    QcfActionType::EvictH2o {
                        target_len,
                        keep_ratio: self.h2o_keep_ratio,
                        protected_prefix: self.protected_prefix,
                    }
                }
            } else {
                QcfActionType::EvictSliding { target_len }
            };
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
            // D2O simulator (paper Eq.8) needs K for nearest-neighbour
            // matching; other actions ignore `k_source`.
            let k_source = if matches!(action, QcfActionType::MergeD2o { .. }) {
                VDataSource::from_buffer(&cache.k_buffer, None)
            } else {
                None
            };
            // Compute retained set before moving `action` into params (experimental path).
            // Clone action for re-use in β-amplified measurements.
            let action_for_beta = if self.experimental_enabled {
                Some(action.clone())
            } else {
                None
            };
            let retained_for_topk = if self.experimental_enabled {
                Some(identify_retained_for_action(
                    &action,
                    &attention_scores,
                    before_len,
                ))
            } else {
                None
            };
            let params = QcfKvParams {
                action,
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
                // _ = action_for_beta; _ = retained_for_topk;   // (kept names for diff clarity)
                let _ = action_for_beta;
                let _ = retained_for_topk;

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
                        let k_source_l = if self.is_d2o {
                            VDataSource::from_buffer(&cache_l.k_buffer, None)
                        } else {
                            None
                        };
                        let target_len_l = ((cache_l.current_pos as f32) * ratio) as usize;
                        let action_l = if self.score_based_eviction {
                            if self.is_d2o {
                                QcfActionType::MergeD2o {
                                    target_len: target_len_l,
                                    keep_ratio: self.h2o_keep_ratio,
                                    protected_prefix: self.protected_prefix,
                                }
                            } else {
                                QcfActionType::EvictH2o {
                                    target_len: target_len_l,
                                    keep_ratio: self.h2o_keep_ratio,
                                    protected_prefix: self.protected_prefix,
                                }
                            }
                        } else {
                            QcfActionType::EvictSliding {
                                target_len: target_len_l,
                            }
                        };
                        let params_l = QcfKvParams {
                            action: action_l,
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

        // Perform eviction
        let result = if self.score_based_eviction {
            let active = self
                .score_accumulator
                .as_ref()
                .is_some_and(|acc| acc.is_active());
            if active {
                let scores = self
                    .score_accumulator
                    .as_ref()
                    .unwrap()
                    .importance_scores()
                    .to_vec();
                self.cache_manager
                    .force_evict_with_scores(caches, ratio, &scores)
            } else {
                self.cache_manager.force_evict(caches, ratio)
            }
        } else {
            self.cache_manager.force_evict(caches, ratio)
        };

        if let Ok(evict_result) = result
            && evict_result.evicted
        {
            self.eviction_count += 1;
            self.evicted_total += evict_result.tokens_removed;
            if let Some(acc) = self.score_accumulator.as_mut() {
                acc.reset();
            }

            // Store QCF result for extra_question_fields
            self.eviction_qcf = Some(EvictionQcfResult {
                tokens_evicted: evict_result.tokens_removed,
                eviction_ratio,
                qcf_caote,
            });
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
        self.eviction_count = 0;
        self.evicted_total = 0;
        self.eviction_qcf = None;
        self.experimental_qcf = None;
    }

    fn snapshot(&self, caches: &[KVCache]) -> Box<dyn CacheSnapshot<KVCache>> {
        let mut data = Vec::with_capacity(caches.len());
        let mut k_sizes = Vec::with_capacity(caches.len());
        let mut positions = Vec::with_capacity(caches.len());
        let mut capacities = Vec::with_capacity(caches.len());
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
            capacities.push(cache.capacity());
        }
        Box::new(KVCacheSnapshot {
            data,
            k_sizes,
            positions,
            capacities,
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
        // The probe step populates score_accumulator for H2O decisions and
        // captures last_step_head_attn for QCF-ATTN measurement.
        !caches.is_empty() && max_cache_pos(caches) > self.effective_budget
    }

    fn extra_question_fields(&self, _caches: &[KVCache]) -> serde_json::Value {
        let mut obj = serde_json::json!({
            "effective_budget": self.effective_budget,
            "eviction_count": self.eviction_count,
            "evicted_tokens": self.evicted_total,
        });
        if let Some(ref qcf) = self.eviction_qcf {
            obj["qcf"] = serde_json::json!(qcf.qcf_caote);
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
            "is_d2o": self.is_d2o,
            "kv_type": self.kv_type,
            "experimental_enabled": self.experimental_enabled,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::cache_manager::CacheManager;
    use crate::kv::eviction::no_eviction::NoEvictionPolicy;
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

    fn make_hook_with_d2o(budget: usize, score_based: bool, is_d2o: bool) -> EvictionHook {
        let policy = Box::new(NoEvictionPolicy::new());
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
            is_d2o,
            "f32".to_string(),
            crate::backend::cpu::cpu_singleton(),
            false,
            vec![], // qcf_sample_layers: empty → internal fallback to [0]
        )
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
            fields.get("qcf_caote_max").is_none(),
            "qcf_caote_max should be absent when experimental_qcf is None"
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
        let policy = Box::new(NoEvictionPolicy::new());
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
        );
        assert_eq!(hook.qcf_sample_layers, vec![0, 4, 8, 12, 15]);
    }
}
