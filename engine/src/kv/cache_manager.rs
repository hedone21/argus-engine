use std::sync::Arc;

use anyhow::Result;

use crate::kv::eviction::EvictionPolicy;
use crate::kv::kv_cache::{KVCache, max_cache_pos};
use crate::kv::{
    ActionResult, CachePressurePipeline, EvictionHandler, HandlerContext, MIN_EVICT_TOKENS,
    PressureLevel, PressureStageConfig, SwapHandler,
};
// LAYER-EXEMPT: cross_cutting_trait_usage — §13.8-N SystemMonitor trait
use crate::resilience::sys_monitor::SystemMonitor;
use std::path::PathBuf;

/// Result of an eviction attempt.
#[derive(Debug, Clone)]
pub struct EvictionResult {
    /// Whether eviction was actually performed.
    pub evicted: bool,
    /// Number of tokens removed per cache.
    pub tokens_removed: usize,
    /// New position after eviction.
    pub new_pos: usize,
}

/// Score context variants for the unified dispatch path.
pub enum ScoreContext<'a> {
    /// No importance scores available.
    None,
    /// Flat per-token importance scores, plus an optional last-layer last-step
    /// per-(kv_head,pos) attention slice (`[n_kv_heads * max_seq_len]`, row-major) for
    /// value-aware techniques (the `a_i` slice). `None` when no AttnWeights producer is active —
    /// the stage then falls back to flat `importance`.
    Flat {
        importance: &'a [f32],
        last_attn: Option<&'a [f32]>,
    },
    /// Per-KV-head importance scores (GQA-aware).
    PerHead {
        flat: &'a [f32],
        head: &'a [f32],
        n_kv_heads: usize,
    },
}

/// Orchestrates KV cache management based on memory pressure and policy decisions.
///
/// Internally, CacheManager always operates through a `CachePressurePipeline`.
/// When created with `new()` (legacy API), the `EvictionPolicy` is wrapped in
/// an `EvictionHandler` adapter automatically. This eliminates routing duplication
/// while preserving full backward compatibility.
///
/// CacheManager follows the Dependency Inversion principle:
/// - Depends on `dyn EvictionPolicy` / `CachePressureHandler` (abstractions)
/// - Depends on `dyn SystemMonitor` (abstraction), not OS-specific implementations
pub struct CacheManager {
    pipeline: CachePressurePipeline,
    monitor: Box<dyn SystemMonitor>,
    /// Eviction triggers when available memory drops below this threshold (bytes).
    threshold_bytes: usize,
    /// Optional disk-backed swap handler for `KvOffload` directives.
    /// Not part of the pressure pipeline — invoked directly by `offload()` /
    /// `recall()` so it only runs when the Manager explicitly asks for it.
    swap_handler: Option<Arc<SwapHandler>>,
}

impl CacheManager {
    /// Create a CacheManager in legacy mode (single eviction policy).
    ///
    /// The policy is wrapped in an `EvictionHandler` and placed in a single-stage
    /// pipeline at `Warning` level, preserving the original behavior.
    pub fn new(
        policy: Box<dyn EvictionPolicy>,
        monitor: Box<dyn SystemMonitor>,
        threshold_bytes: usize,
        target_ratio: f32,
    ) -> Self {
        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(policy, target_ratio)),
        }]);
        Self {
            pipeline,
            monitor,
            threshold_bytes,
            swap_handler: None,
        }
    }

    /// Create a CacheManager in pipeline mode (multi-handler pressure pipeline).
    ///
    /// The pipeline dispatches different handlers based on `PressureLevel`,
    /// which is determined from available memory relative to `threshold_bytes`.
    pub fn with_pipeline(
        pipeline: CachePressurePipeline,
        monitor: Box<dyn SystemMonitor>,
        threshold_bytes: usize,
    ) -> Self {
        Self {
            pipeline,
            monitor,
            threshold_bytes,
            swap_handler: None,
        }
    }

    /// Enable disk-backed KV swap. The resulting `SwapHandler` is stored on the
    /// manager but *not* registered in the pressure pipeline — it fires only
    /// when the engine explicitly calls `offload()` / `recall()` in response
    /// to a `KvOffload` / `RestoreDefaults` directive.
    pub fn enable_swap(&mut self, swap_dir: PathBuf) {
        self.swap_handler = Some(Arc::new(SwapHandler::with_disk(0.5, swap_dir)));
    }

    /// Offload `ratio` fraction (LRU prefix) of each layer's KV cache to disk.
    /// No-op + warning when swap is not enabled. Returns the number of tokens
    /// offloaded (summed across layers).
    pub fn offload(&mut self, caches: &mut [KVCache], ratio: f32) -> Result<usize> {
        let Some(handler_arc) = self.swap_handler.as_mut() else {
            eprintln!("[CacheManager] KvOffload ignored: swap not enabled (missing --swap-dir)");
            return Ok(0);
        };
        // Update ratio on the shared handler.
        if let Some(h) = Arc::get_mut(handler_arc) {
            h.set_ratio(ratio);
        } else {
            // Shared reference outlives us; build a new handler preserving the dir + state.
            let new_handler = SwapHandler {
                offload_ratio: ratio.clamp(0.0, 1.0),
                swap_dir: handler_arc.swap_dir.clone(),
                state: handler_arc.state.clone(),
            };
            *handler_arc = Arc::new(new_handler);
        }
        handler_arc.offload_caches(caches)
    }

    /// Recall any previously offloaded tokens for each cache layer.
    /// Returns the number of tokens restored. No-op when swap is not enabled.
    pub fn recall(&mut self, caches: &mut [KVCache]) -> Result<usize> {
        let Some(handler) = self.swap_handler.as_ref() else {
            return Ok(0);
        };
        handler.recall_caches(caches)
    }

    /// Determine pressure level from available memory.
    ///
    /// - `>= threshold`: Normal
    /// - `>= threshold / 2`: Warning
    /// - `>= threshold / 4`: Critical
    /// - `< threshold / 4`: Emergency
    ///
    /// β-5: 계단 산식은 `Pressure::from_mem_available` 으로 일원화되었다 (cutoff 의 단일
    /// 거처). 본 메서드는 `Pressure → band()` 강등으로 동일 결과를 낸다 (behavior-preserving).
    fn determine_pressure_level(&self, mem_available: usize) -> PressureLevel {
        crate::pipeline::Pressure::from_mem_available(mem_available, self.threshold_bytes).band()
    }

    /// Convert pipeline `ActionResult`s into a legacy `EvictionResult`.
    fn pipeline_results_to_eviction_result(
        results: &[ActionResult],
        caches: &[KVCache],
    ) -> EvictionResult {
        let mut total_removed = 0usize;
        let mut any_action = false;
        let mut last_new_pos = max_cache_pos(caches);

        for r in results {
            match r {
                ActionResult::Evicted {
                    tokens_removed,
                    new_pos,
                } => {
                    total_removed += tokens_removed;
                    last_new_pos = *new_pos;
                    any_action = true;
                }
                ActionResult::NoOp => {}
                _ => {
                    any_action = true;
                }
            }
        }

        EvictionResult {
            evicted: any_action,
            tokens_removed: total_removed,
            new_pos: last_new_pos,
        }
    }

    /// Unified dispatch: query memory, determine pressure, build context, execute pipeline.
    ///
    /// When `force` is true, bypasses memory checks and runs at `Emergency` level.
    fn execute_dispatch(
        &self,
        caches: &mut [KVCache],
        scores: ScoreContext,
        force: bool,
        force_target_ratio: Option<f32>,
    ) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        let (pressure, mem_available) = if force {
            // Budget-driven forced eviction: use Emergency to ensure all pipeline
            // handlers run regardless of their min_level. Read actual mem_available
            // for accurate logging (previously hardcoded to 0, which was misleading).
            let mem = self
                .monitor
                .mem_stats()
                .map(|s| s.available)
                .unwrap_or(usize::MAX);
            (PressureLevel::Emergency, mem)
        } else {
            let mem_available = match self.monitor.mem_stats() {
                Ok(stats) => stats.available,
                Err(e) => {
                    log::warn!("Failed to read memory stats: {}, skipping eviction", e);
                    return Ok(EvictionResult {
                        evicted: false,
                        tokens_removed: 0,
                        new_pos: max_cache_pos(caches),
                    });
                }
            };
            let pressure = self.determine_pressure_level(mem_available);
            if pressure == PressureLevel::Normal {
                return Ok(EvictionResult {
                    evicted: false,
                    tokens_removed: 0,
                    new_pos: max_cache_pos(caches),
                });
            }
            (pressure, mem_available)
        };

        if force {
            log::info!("[CacheEvent] Budget eviction (forced)");
            log::info!(
                "[CacheManager] budget eviction (forced), executing '{}'",
                self.pipeline.name(),
            );
        } else {
            log::info!(
                "[CacheEvent] Pressure {:?}, mem_available={} MB",
                pressure,
                mem_available / (1024 * 1024),
            );
            log::info!(
                "[CacheManager] pressure={:?}, executing '{}'",
                pressure,
                self.pipeline.name(),
            );
        }

        let (importance, head_importance, n_kv_heads, last_attn) = match scores {
            ScoreContext::None => (None, None, 0, None),
            ScoreContext::Flat {
                importance,
                last_attn,
            } => (Some(importance), None, 0, last_attn),
            ScoreContext::PerHead {
                flat,
                head,
                n_kv_heads,
            } => (Some(flat), Some(head), n_kv_heads, None),
        };

        let mut ctx = HandlerContext {
            caches,
            importance,
            head_importance,
            n_kv_heads,
            last_attn,
            pressure_level: pressure,
            mem_available,
            target_ratio: force_target_ratio,
            qcf_sink: None,
        };
        let results = self.pipeline.execute(&mut ctx)?;
        let eviction_result = Self::pipeline_results_to_eviction_result(&results, ctx.caches);

        if eviction_result.evicted {
            // Release physical pages for unused KV buffer regions (madvise MADV_DONTNEED)
            let mut bytes_released = 0usize;
            for cache in ctx.caches.iter_mut() {
                bytes_released += cache.release_unused_pages();
            }
            log::info!(
                "[CacheEvent] Eviction completed: policy='{}', removed={}, new_pos={}",
                self.pipeline.name(),
                eviction_result.tokens_removed,
                eviction_result.new_pos,
            );
            if bytes_released > 0 {
                log::info!(
                    "[CacheManager] released {} MB of physical pages after eviction",
                    bytes_released / (1024 * 1024),
                );
            }
        }

        Ok(eviction_result)
    }

    // ── Public API (all signatures preserved) ───────────────────────

    /// Check memory pressure and evict from all caches if needed.
    ///
    /// Called after each generation step in the inference loop.
    pub fn maybe_evict(&self, caches: &mut [KVCache]) -> Result<EvictionResult> {
        self.execute_dispatch(caches, ScoreContext::None, false, None)
    }

    /// Check memory pressure and evict using importance scores.
    ///
    /// Same logic as `maybe_evict()`, but passes importance scores to the handler.
    /// Used when `AttentionScoreAccumulator` is active.
    pub fn maybe_evict_with_scores(
        &self,
        caches: &mut [KVCache],
        importance: &[f32],
        last_attn: Option<&[f32]>,
    ) -> Result<EvictionResult> {
        self.execute_dispatch(
            caches,
            ScoreContext::Flat {
                importance,
                last_attn,
            },
            false,
            None,
        )
    }

    /// Check memory pressure and evict using per-KV-head importance scores.
    ///
    /// GQA-aware version of `maybe_evict_with_scores()`.
    pub fn maybe_evict_with_head_scores(
        &self,
        caches: &mut [KVCache],
        flat_importance: &[f32],
        head_importance: &[f32],
        n_kv_heads: usize,
    ) -> Result<EvictionResult> {
        self.execute_dispatch(
            caches,
            ScoreContext::PerHead {
                flat: flat_importance,
                head: head_importance,
                n_kv_heads,
            },
            false,
            None,
        )
    }

    /// Force eviction without scores, bypassing should_evict() and memory checks.
    ///
    /// Used when eviction is triggered externally (e.g., by resilience signals).
    /// Runs at `Emergency` pressure level.
    pub fn force_evict(&self, caches: &mut [KVCache], target_ratio: f32) -> Result<EvictionResult> {
        self.execute_dispatch(caches, ScoreContext::None, true, Some(target_ratio))
    }

    /// Force eviction with importance scores, bypassing should_evict() and memory checks.
    ///
    /// Used when eviction is triggered externally for score-aware policies like heavy-hitter.
    /// Runs at `Emergency` pressure level with scores.
    pub fn force_evict_with_scores(
        &self,
        caches: &mut [KVCache],
        target_ratio: f32,
        importance: &[f32],
        last_attn: Option<&[f32]>,
    ) -> Result<EvictionResult> {
        self.execute_dispatch(
            caches,
            ScoreContext::Flat {
                importance,
                last_attn,
            },
            true,
            Some(target_ratio),
        )
    }

    /// Force eviction with per-KV-head importance scores.
    ///
    /// Used when heavy-hitter+ (GQA-aware) policy needs per-head eviction.
    /// Runs at `Emergency` pressure level with head scores.
    pub fn force_evict_with_head_scores(
        &self,
        caches: &mut [KVCache],
        target_ratio: f32,
        flat_importance: &[f32],
        head_importance: &[f32],
        n_kv_heads: usize,
    ) -> Result<EvictionResult> {
        self.execute_dispatch(
            caches,
            ScoreContext::PerHead {
                flat: flat_importance,
                head: head_importance,
                n_kv_heads,
            },
            true,
            Some(target_ratio),
        )
    }

    /// Returns the name of the active policy or pipeline.
    pub fn policy_name(&self) -> String {
        self.pipeline.name()
    }

    /// Shared eviction core: guard on `target_len`, dispatch to policy methods,
    /// assemble result. `target_len == 0` means "policy decides" (guards skipped).
    ///
    /// pressure 파이프라인의 `EvictionHandler::handle` 가 호출하는 **단일 eviction
    /// 알고리즘** (α-K 2b 통합 — 구 EvictionHandler 인라인 복제 제거). 호출자는 자기
    /// 규칙으로 `target_len` 을 미리 해소해 전달한다 (EHH=`max(1)`).
    // LAYER-EXEMPT: backend_concrete_downcast — §13.8-L cold-path eviction dispatch
    pub(crate) fn run_policy_eviction(
        policy: &dyn EvictionPolicy,
        caches: &mut [KVCache],
        target_len: usize,
        scores: ScoreContext,
        per_layer_target_len: Option<&[usize]>,
    ) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        let current_pos = max_cache_pos(caches);

        // Global guards apply only to the uniform (scalar `target_len`) path. With per-layer
        // budgets (R-P1-6) each layer self-guards inside the loop, since layers have different
        // targets, so the global early-returns are skipped. When `per_layer_target_len` is
        // `None` this is byte-identical to the previous behavior (Gate-0).
        if per_layer_target_len.is_none() {
            if target_len > 0 && current_pos <= target_len {
                return Ok(EvictionResult {
                    evicted: false,
                    tokens_removed: 0,
                    new_pos: current_pos,
                });
            }

            if target_len > 0 {
                let tokens_to_remove = current_pos - target_len;
                if tokens_to_remove < MIN_EVICT_TOKENS {
                    log::debug!(
                        "[CacheManager] skip: policy='{}', tokens_to_remove={} < MIN_EVICT_TOKENS={}",
                        policy.name(),
                        tokens_to_remove,
                        MIN_EVICT_TOKENS,
                    );
                    return Ok(EvictionResult {
                        evicted: false,
                        tokens_removed: 0,
                        new_pos: current_pos,
                    });
                }
            }
        }

        log::debug!(
            "[CacheManager] policy='{}': {} → {} tokens",
            policy.name(),
            current_pos,
            target_len,
        );

        let (importance, head_importance, n_kv_heads, last_attn) = match &scores {
            ScoreContext::None => (None, None, 0, None),
            ScoreContext::Flat {
                importance,
                last_attn,
            } => (Some(*importance), None, 0, *last_attn),
            ScoreContext::PerHead {
                flat,
                head,
                n_kv_heads,
            } => (Some(*flat), Some(*head), *n_kv_heads, None),
        };

        let n_layers = caches.len();
        for (layer_idx, cache) in caches.iter_mut().enumerate() {
            // Per-layer KV budget (R-P1-6): with a per-layer target vector each layer uses its
            // own target_len (an out-of-range index falls back to the scalar); with `None` every
            // layer uses the shared scalar `target_len` (byte-identical to the prior behavior).
            let layer_target_len = match per_layer_target_len {
                Some(v) => v.get(layer_idx).copied().unwrap_or(target_len),
                None => target_len,
            };

            // Per-layer self-guard (only on the per-layer path; the uniform path was guarded above).
            if per_layer_target_len.is_some() && layer_target_len > 0 {
                let layer_pos = cache.current_pos;
                if layer_pos <= layer_target_len || layer_pos - layer_target_len < MIN_EVICT_TOKENS
                {
                    continue;
                }
            }

            if let (Some(flat), Some(head_imp)) = (importance, head_importance) {
                if n_kv_heads > 0 {
                    // Per-head (h2o_plus) — not a layer-aware stage, but the real (layer_idx,
                    // n_layers) is threaded through so the keep-set dump (R-P0-2) keys by layer.
                    policy.evict_with_head_scores(
                        cache,
                        layer_target_len,
                        flat,
                        head_imp,
                        n_kv_heads,
                        layer_idx,
                        n_layers,
                    )?;
                } else {
                    // last_attn does not vary by layer (last-layer last-step approximation),
                    // so the same slice is threaded into every layer's plan.
                    policy.evict_layer(
                        cache,
                        layer_target_len,
                        Some(flat),
                        last_attn,
                        layer_idx,
                        n_layers,
                    )?;
                }
            } else {
                // flat-score or score-free: route through the per-layer entry so a layer-aware
                // adapter (StageBackedPolicy → d2o) sees the real (layer_idx, n_layers).
                policy.evict_layer(
                    cache,
                    layer_target_len,
                    importance,
                    last_attn,
                    layer_idx,
                    n_layers,
                )?;
            }
        }

        let new_pos = max_cache_pos(caches);
        let tokens_removed = current_pos - new_pos;

        // The undershoot warning compares against the single scalar target, so it is meaningful
        // only on the uniform path (per-layer budgets have no single "expected" total). Gating it
        // keeps the `None` path byte-identical.
        if per_layer_target_len.is_none() {
            let expected_removed = current_pos - target_len;
            // Warn when eviction achieved significantly less than requested.
            // This catches silent clamping by policies (e.g. protected_prefix > target_len).
            if expected_removed > 0 && tokens_removed < expected_removed / 2 {
                log::warn!(
                    "[CacheManager] policy='{}': eviction undershot — removed {} tokens but target was {} ({}% of request). \
                     Check protected_prefix or policy constraints.",
                    policy.name(),
                    tokens_removed,
                    expected_removed,
                    tokens_removed * 100 / expected_removed,
                );
            }
        }

        Ok(EvictionResult {
            evicted: true,
            tokens_removed,
            new_pos,
        })
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::eviction::stage_registry::none_backed_policy;
    use crate::kv::eviction::stage_registry::sliding_backed_policy;
    use crate::memory::host::shared::SharedBuffer;
    use crate::resilience::sys_monitor::MemoryStats;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    /// Mock SystemMonitor for testing
    struct MockMonitor {
        available: usize,
    }

    impl SystemMonitor for MockMonitor {
        fn mem_stats(&self) -> Result<MemoryStats> {
            Ok(MemoryStats {
                total: 4 * 1024 * 1024 * 1024,
                available: self.available,
                free: self.available / 2,
            })
        }
    }

    fn make_caches(n_layers: usize, pos: usize) -> Vec<KVCache> {
        let max_seq = 100;
        let backend = Arc::new(CpuBackend::new());
        (0..n_layers)
            .map(|_| {
                let buf_size = max_seq * 4 * 4;
                let k = Tensor::new(
                    Shape::new(vec![1, max_seq, 1, 4]),
                    Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                    backend.clone(),
                );
                let v = Tensor::new(
                    Shape::new(vec![1, max_seq, 1, 4]),
                    Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                    backend.clone(),
                );
                let mut cache = KVCache::new(k, v, max_seq);
                cache.current_pos = pos;
                cache
            })
            .collect()
    }

    #[test]
    fn test_no_eviction_with_plenty_memory() {
        let cm = CacheManager::new(
            none_backed_policy(),
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024,
            }), // 1GB
            256 * 1024 * 1024, // 256MB threshold
            0.75,
        );
        let mut caches = make_caches(4, 50);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
        assert_eq!(caches[0].current_pos, 50);
    }

    #[test]
    fn test_sliding_window_with_memory_pressure() {
        // target_ratio=0.3 → target_len=30, tokens_to_remove=70 >= MIN_EVICT_TOKENS(64) → guard passes.
        let cm = CacheManager::new(
            sliding_backed_policy(30, 0),
            Box::new(MockMonitor {
                available: 100 * 1024 * 1024,
            }), // 100MB (below threshold)
            256 * 1024 * 1024, // 256MB threshold
            0.3,
        );
        let mut caches = make_caches(4, 100);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(result.evicted);
        for cache in &caches {
            assert!(cache.current_pos < 100);
        }
    }

    #[test]
    fn test_eviction_across_all_layers() {
        // pos=100, target_ratio=0.3 → target_len=30, tokens_to_remove=70 >= MIN_EVICT_TOKENS(64).
        let cm = CacheManager::new(
            sliding_backed_policy(20, 0),
            Box::new(MockMonitor {
                available: 10 * 1024 * 1024,
            }), // Very low
            256 * 1024 * 1024,
            0.3,
        );
        let mut caches = make_caches(16, 100);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(result.evicted);
        // All 16 layers should have the same position
        let pos = caches[0].current_pos;
        for cache in &caches {
            assert_eq!(cache.current_pos, pos);
        }
    }

    /// R-P1-6 end-to-end: SqueezeAttention budgets (`compute_squeeze_budgets`) flow through the
    /// real `run_policy_eviction` per-layer path and produce *per-layer-independent* decisions —
    /// low-budget (important-prefix-light) layers evict while higher-budget layers are preserved by
    /// the per-layer self-guard. Exercises the budget math + the eviction mechanism together.
    #[test]
    fn per_layer_budget_evicts_layers_independently() {
        use crate::kv::squeeze_budget::compute_squeeze_budgets;

        let start = 100usize;
        // Increasing per-layer importance → non-decreasing tier budgets. With pos=100 and
        // MIN_EVICT_TOKENS=64, only budgets ≤ 36 trigger eviction; larger budgets are skipped by
        // the per-layer self-guard. tiers low/low/mid/mid/high/high → weights 1,1,2,2,3,3 (wsum=12),
        // total 360 → [30,30,60,60,90,90].
        let importance = [0.05f32, 0.05, 0.5, 0.5, 0.95, 0.95];
        let budgets = compute_squeeze_budgets(&importance, 360, 1);
        assert_eq!(budgets, vec![30, 30, 60, 60, 90, 90], "tier allocation");

        let policy = sliding_backed_policy(0, 0);
        let mut caches = make_caches(importance.len(), start);
        let res = CacheManager::run_policy_eviction(
            policy.as_ref(),
            &mut caches,
            0, // scalar fallback unused — every index is in range
            ScoreContext::None,
            Some(&budgets),
        )
        .unwrap();
        assert!(res.evicted);

        // budget 30: 100-30=70 ≥ 64 → evicted.
        assert!(caches[0].current_pos < start, "low-budget layer evicts");
        assert!(caches[1].current_pos < start);
        // budget 60: 100-60=40 < 64 → per-layer self-guard skips → preserved.
        assert_eq!(caches[2].current_pos, start, "mid-budget layer preserved");
        assert_eq!(caches[3].current_pos, start);
        // budget 90: 100-90=10 < 64 → preserved.
        assert_eq!(caches[4].current_pos, start, "high-budget layer preserved");
        assert_eq!(caches[5].current_pos, start);
    }

    /// R-P1-6 Gate-0: with `per_layer_target_len = None` every layer uses the scalar target_len
    /// uniformly — the pre-R-P1-6 behavior (no per-layer differentiation).
    #[test]
    fn per_layer_budget_none_is_uniform() {
        let policy = sliding_backed_policy(0, 0);
        let mut caches = make_caches(4, 100);
        let res = CacheManager::run_policy_eviction(
            policy.as_ref(),
            &mut caches,
            30,
            ScoreContext::None,
            None,
        )
        .unwrap();
        assert!(res.evicted);
        let p = caches[0].current_pos;
        assert!(p < 100);
        for c in &caches {
            assert_eq!(c.current_pos, p, "None path must be uniform across layers");
        }
    }

    #[test]
    fn test_empty_caches() {
        let cm = CacheManager::new(
            none_backed_policy(),
            Box::new(MockMonitor { available: 0 }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches: Vec<KVCache> = Vec::new();
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
    }

    #[test]
    fn test_policy_name() {
        let cm = CacheManager::new(
            sliding_backed_policy(10, 0),
            Box::new(MockMonitor { available: 0 }),
            0,
            0.75,
        );
        // Legacy mode wraps policy in EvictionHandler at Warning level
        assert!(cm.policy_name().contains("sliding"));
    }

    /// Mock monitor that always returns an error
    struct ErrorMonitor;
    impl SystemMonitor for ErrorMonitor {
        fn mem_stats(&self) -> Result<MemoryStats> {
            Err(anyhow::anyhow!("simulated monitor failure"))
        }
    }

    #[test]
    fn test_monitor_error_skips_eviction() {
        let cm = CacheManager::new(
            sliding_backed_policy(10, 0),
            Box::new(ErrorMonitor),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches = make_caches(4, 50);
        let result = cm.maybe_evict(&mut caches).unwrap();
        // Should not evict when monitor fails
        assert!(!result.evicted);
        assert_eq!(result.new_pos, 50);
    }

    #[test]
    fn test_maybe_evict_with_scores_triggers() {
        use crate::kv::eviction::stage_registry::h2o_backed_policy;

        // pos=100, target_ratio=0.3 → target_len=30, tokens_to_remove=70 >= MIN_EVICT_TOKENS(64).
        let cm = CacheManager::new(
            h2o_backed_policy(0.3, 0), // prefix=4, keep_ratio=0.3
            Box::new(MockMonitor {
                available: 10 * 1024 * 1024,
            }),
            256 * 1024 * 1024,
            0.3,
        );
        let mut caches = make_caches(4, 100);

        let mut importance = vec![0.0f32; 100];
        // Give some tokens high importance
        importance[10] = 10.0;
        importance[20] = 9.0;
        importance[30] = 8.0;

        let result = cm
            .maybe_evict_with_scores(&mut caches, &importance, None)
            .unwrap();
        assert!(result.evicted);
        // All layers should have the same position
        let pos = caches[0].current_pos;
        for cache in &caches {
            assert_eq!(cache.current_pos, pos);
        }
    }

    #[test]
    fn test_maybe_evict_with_scores_no_eviction_needed() {
        let cm = CacheManager::new(
            none_backed_policy(),
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024,
            }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches = make_caches(4, 50);
        let importance = vec![1.0f32; 100];

        let result = cm
            .maybe_evict_with_scores(&mut caches, &importance, None)
            .unwrap();
        assert!(!result.evicted);
        assert_eq!(caches[0].current_pos, 50);
    }

    // ── force_evict tests (signal-driven) ──

    #[test]
    fn test_force_evict_bypasses_should_evict() {
        // heavy-hitter's should_evict() always returns false, but force_evict must still work.
        // pos=100, target_ratio=0.3 → tokens_to_remove=70 >= MIN_EVICT_TOKENS(64) → guard passes.
        use crate::kv::eviction::stage_registry::h2o_backed_policy;

        let cm = CacheManager::new(
            h2o_backed_policy(0.3, 0),
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024, // plenty of memory
            }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches = make_caches(4, 100);

        // maybe_evict should NOT trigger (memory OK → Normal pressure)
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
        assert_eq!(caches[0].current_pos, 100);

        // force_evict MUST trigger regardless (Emergency level)
        let result = cm.force_evict(&mut caches, 0.3).unwrap();
        assert!(result.evicted);
        assert!(caches[0].current_pos < 100);
    }

    #[test]
    fn test_force_evict_with_scores_bypasses_checks() {
        // pos=100, target_ratio=0.3 → tokens_to_remove=70 >= MIN_EVICT_TOKENS(64) → guard passes.
        use crate::kv::eviction::stage_registry::h2o_backed_policy;

        let cm = CacheManager::new(
            h2o_backed_policy(0.3, 0),
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024,
            }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches = make_caches(4, 100);

        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[20] = 9.0;
        importance[30] = 8.0;

        let result = cm
            .force_evict_with_scores(&mut caches, 0.3, &importance, None)
            .unwrap();
        assert!(result.evicted);
        let pos = caches[0].current_pos;
        for cache in &caches {
            assert_eq!(cache.current_pos, pos);
        }
    }

    #[test]
    fn test_force_evict_empty_caches() {
        let cm = CacheManager::new(
            none_backed_policy(),
            Box::new(MockMonitor { available: 0 }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches: Vec<KVCache> = Vec::new();
        let result = cm.force_evict(&mut caches, 0.5).unwrap();
        assert!(!result.evicted);
    }

    #[test]
    fn test_force_evict_ratio_clamping() {
        // target_ratio=0.0 clamps to 0.1 inside EvictionHandler.
        // pos=100, target_len=100*0.1=10, tokens_to_remove=90 >= MIN_EVICT_TOKENS(64) → guard passes.
        use crate::kv::eviction::stage_registry::h2o_backed_policy;

        let cm = CacheManager::new(
            h2o_backed_policy(0.5, 0),
            Box::new(MockMonitor { available: 0 }),
            0,
            0.75,
        );
        let mut caches = make_caches(1, 100);

        // target_ratio=0.0 should clamp to 0.1 (inside EvictionHandler), tokens_to_remove=90
        let result = cm.force_evict(&mut caches, 0.0).unwrap();
        assert!(result.evicted);
        assert!(caches[0].current_pos > 0);
    }

    #[test]
    fn test_target_ratio_clamping() {
        // target_ratio below 0.1 should be clamped to 0.1.
        // pos=100, clamped target_len=10, tokens_to_remove=90 >= MIN_EVICT_TOKENS(64) → guard passes.
        let cm = CacheManager::new(
            sliding_backed_policy(10, 0),
            Box::new(MockMonitor { available: 10 }),
            256 * 1024 * 1024,
            0.01, // should clamp to 0.1
        );
        let mut caches = make_caches(1, 100);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(result.evicted);
        assert!(caches[0].current_pos > 0);

        // target_ratio above 0.99 should be clamped to 0.99.
        // pos=100, clamped target_len=99, tokens_to_remove=1 < MIN_EVICT_TOKENS(64).
        // Guard fires → NoOp (eviction skipped to avoid useless compaction).
        let cm2 = CacheManager::new(
            sliding_backed_policy(10, 0),
            Box::new(MockMonitor { available: 10 }),
            256 * 1024 * 1024,
            5.0, // should clamp to 0.99
        );
        let mut caches2 = make_caches(1, 100);
        let result2 = cm2.maybe_evict(&mut caches2).unwrap();
        // Guard fires because tokens_to_remove=1 < MIN_EVICT_TOKENS(64).
        assert!(!result2.evicted);
        assert_eq!(caches2[0].current_pos, 100);
    }

    // ── Pipeline-backed CacheManager tests ──

    #[test]
    fn test_pipeline_manager_evicts_at_pressure() {
        // pos=100, target_ratio=0.3 → tokens_to_remove=70 >= MIN_EVICT_TOKENS(64) → guard passes.
        use crate::kv::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(sliding_backed_policy(10, 0), 0.3)),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 100 * 1024 * 1024, // 100MB
            }),
            256 * 1024 * 1024, // 256MB threshold → Warning level
                               // (100MB >= 128MB=threshold/2 → Warning)
        );

        let mut caches = make_caches(4, 100);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(result.evicted);
        for cache in &caches {
            assert!(cache.current_pos < 100);
        }
    }

    #[test]
    fn test_pipeline_manager_no_action_at_normal() {
        use crate::kv::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(sliding_backed_policy(10, 0), 0.5)),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 512 * 1024 * 1024, // 512MB — above 256MB threshold → Normal
            }),
            256 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 40);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
        assert_eq!(caches[0].current_pos, 40);
    }

    #[test]
    fn test_pipeline_manager_force_evict() {
        // pos=100, target_ratio=0.3 → tokens_to_remove=70 >= MIN_EVICT_TOKENS(64) → guard passes.
        use crate::kv::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Emergency,
            handler: Box::new(EvictionHandler::new(sliding_backed_policy(10, 0), 0.3)),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024, // plenty of memory
            }),
            256 * 1024 * 1024,
        );

        // maybe_evict should NOT trigger (Normal pressure)
        let mut caches = make_caches(4, 100);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);

        // force_evict MUST trigger (Emergency level)
        let result = cm.force_evict(&mut caches, 0.3).unwrap();
        assert!(result.evicted);
        assert!(caches[0].current_pos < 100);
    }

    #[test]
    fn test_pipeline_manager_force_evict_with_scores() {
        // pos=100, target_ratio=0.3 → tokens_to_remove=70 >= MIN_EVICT_TOKENS(64) → guard passes.
        use crate::kv::eviction::stage_registry::h2o_backed_policy;
        use crate::kv::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Emergency,
            handler: Box::new(EvictionHandler::new(h2o_backed_policy(0.5, 0), 0.3)),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024,
            }),
            256 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 100);
        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[20] = 9.0;

        let result = cm
            .force_evict_with_scores(&mut caches, 0.3, &importance, None)
            .unwrap();
        assert!(result.evicted);
        assert!(caches[0].current_pos < 100);
    }

    #[test]
    fn test_pipeline_manager_with_scores() {
        // pos=100, target_ratio=0.3 → tokens_to_remove=70 >= MIN_EVICT_TOKENS(64) → guard passes.
        use crate::kv::eviction::stage_registry::h2o_backed_policy;
        use crate::kv::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(h2o_backed_policy(0.5, 0), 0.3)),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 100 * 1024 * 1024, // Warning level
            }),
            256 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 100);
        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[20] = 9.0;
        for i in 4..100 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        let result = cm
            .maybe_evict_with_scores(&mut caches, &importance, None)
            .unwrap();
        assert!(result.evicted);
        let pos = caches[0].current_pos;
        for cache in &caches {
            assert_eq!(cache.current_pos, pos);
        }
    }

    #[test]
    fn test_pipeline_manager_policy_name() {
        use crate::kv::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(EvictionHandler::new(sliding_backed_policy(10, 0), 0.8)),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(EvictionHandler::new(sliding_backed_policy(10, 0), 0.5)),
            },
        ]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor { available: 0 }),
            256 * 1024 * 1024,
        );

        let name = cm.policy_name();
        assert!(name.contains("sliding"));
        assert!(name.contains("Warning"));
        assert!(name.contains("Critical"));
    }

    #[test]
    fn test_pipeline_manager_multi_level_graduated_response() {
        // Use pos=100 and ratios that produce tokens_to_remove >= MIN_EVICT_TOKENS(64).
        // Warning: ratio=0.3 → remove 70. Critical: additional ratio=0.1 → further removal.
        use crate::kv::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        // Two eviction stages: mild at Warning, aggressive at Critical
        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(EvictionHandler::new(
                    sliding_backed_policy(50, 0),
                    0.3, // keep 30% → tokens_to_remove=70 on pos=100
                )),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(EvictionHandler::new(
                    sliding_backed_policy(10, 0),
                    0.1, // keep 10%
                )),
            },
        ]);

        // At Warning level: only the first stage should run
        let cm_warning = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 200 * 1024 * 1024, // 200MB, threshold=400MB → Warning
            }),
            400 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 100);
        let result = cm_warning.maybe_evict(&mut caches).unwrap();
        assert!(result.evicted);
        let pos_after_warning = caches[0].current_pos;

        // At Critical level: both stages should run (more aggressive)
        let pipeline2 = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(EvictionHandler::new(sliding_backed_policy(50, 0), 0.3)),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(EvictionHandler::new(sliding_backed_policy(10, 0), 0.1)),
            },
        ]);

        let cm_critical = CacheManager::with_pipeline(
            pipeline2,
            Box::new(MockMonitor {
                available: 50 * 1024 * 1024, // 50MB, threshold=400MB → Critical
            }),
            400 * 1024 * 1024,
        );

        let mut caches2 = make_caches(4, 100);
        let result2 = cm_critical.maybe_evict(&mut caches2).unwrap();
        assert!(result2.evicted);
        let pos_after_critical = caches2[0].current_pos;

        // Critical should be more aggressive than Warning
        assert!(
            pos_after_critical <= pos_after_warning,
            "Critical ({}) should evict at least as much as Warning ({})",
            pos_after_critical,
            pos_after_warning,
        );
    }

    #[test]
    fn test_pipeline_manager_empty_pipeline() {
        use crate::kv::CachePressurePipeline;

        let pipeline = CachePressurePipeline::new(vec![]);
        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor { available: 0 }),
            256 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 40);
        // Emergency level but empty pipeline → no action
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
    }

    #[test]
    fn test_pipeline_manager_monitor_error_skips() {
        use crate::kv::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(sliding_backed_policy(10, 0), 0.5)),
        }]);

        let cm = CacheManager::with_pipeline(pipeline, Box::new(ErrorMonitor), 256 * 1024 * 1024);

        let mut caches = make_caches(4, 40);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
        assert_eq!(result.new_pos, 40);
    }
}
