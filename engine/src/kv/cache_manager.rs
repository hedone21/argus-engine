use std::sync::Arc;

use anyhow::Result;

use crate::kv::eviction::EvictionPolicy;
use crate::kv::kv_cache::{KVCache, max_cache_pos};
use crate::kv::{
    ActionResult, CachePressurePipeline, EvictionHandler, HandlerContext, MIN_EVICT_TOKENS,
    PressureLevel, PressureStageConfig, SwapHandler,
};
use crate::stages::kv::mutation::drive_cross_layer;
use argus_extension_api::KVMutationStage;
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
    /// Per-`(layer, token)` FLAT importance (faithful-H2O `H2OKVCache_LayerWise`, divergence `(b)`):
    /// `layer_flat` is `[n_layers * max_seq]`, row-major `layer * max_seq + pos`. The per-layer
    /// eviction loop slices each layer's own `max_seq` window and ranks that layer's heavy hitters
    /// independently — NO cross-layer MAX collapse. `last_attn` carries the value-aware `a_i` as in
    /// `Flat`. Opt-in (faithful-H2O only); every other path uses the collapsed variants above.
    PerLayerFlat {
        layer_flat: &'a [f32],
        max_seq: usize,
        last_attn: Option<&'a [f32]>,
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

        let (importance, head_importance, n_kv_heads, last_attn, per_layer_flat) = match scores {
            ScoreContext::None => (None, None, 0, None, None),
            ScoreContext::Flat {
                importance,
                last_attn,
            } => (Some(importance), None, 0, last_attn, None),
            ScoreContext::PerHead {
                flat,
                head,
                n_kv_heads,
            } => (Some(flat), Some(head), n_kv_heads, None, None),
            ScoreContext::PerLayerFlat {
                layer_flat,
                max_seq,
                last_attn,
            } => (None, None, 0, last_attn, Some((layer_flat, max_seq))),
        };

        let mut ctx = HandlerContext {
            caches,
            importance,
            head_importance,
            n_kv_heads,
            last_attn,
            per_layer_flat,
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

    /// Force eviction with per-`(layer, token)` FLAT importance (faithful-H2O `H2OKVCache_LayerWise`,
    /// divergence `(b)`). `layer_flat` is `[n_layers * max_seq]`, row-major `layer * max_seq + pos`;
    /// each layer ranks its own heavy hitters on its `max_seq` window with no cross-layer MAX. Opt-in
    /// (faithful-H2O only); other paths use the collapsed `force_evict_with_scores`.
    pub fn force_evict_with_per_layer_scores(
        &self,
        caches: &mut [KVCache],
        target_ratio: f32,
        layer_flat: &[f32],
        max_seq: usize,
        last_attn: Option<&[f32]>,
    ) -> Result<EvictionResult> {
        self.execute_dispatch(
            caches,
            ScoreContext::PerLayerFlat {
                layer_flat,
                max_seq,
                last_attn,
            },
            true,
            Some(target_ratio),
        )
    }

    /// Pressure-checked variant of [`force_evict_with_per_layer_scores`](Self::force_evict_with_per_layer_scores)
    /// (mirror of [`maybe_evict_with_scores`](Self::maybe_evict_with_scores)) — evicts only if memory
    /// pressure warrants it, ranking each layer on its own FLAT importance window.
    pub fn maybe_evict_with_per_layer_scores(
        &self,
        caches: &mut [KVCache],
        layer_flat: &[f32],
        max_seq: usize,
        last_attn: Option<&[f32]>,
    ) -> Result<EvictionResult> {
        self.execute_dispatch(
            caches,
            ScoreContext::PerLayerFlat {
                layer_flat,
                max_seq,
                last_attn,
            },
            false,
            None,
        )
    }

    /// Returns the name of the active policy or pipeline.
    ///
    /// This is the **logging** descriptor — each stage is decorated with the
    /// pressure level that arms it (`h2o@Warning`). For a stable identity to
    /// group diagnostic records by, use [`policy_id`](Self::policy_id).
    pub fn policy_name(&self) -> String {
        self.pipeline.name()
    }

    /// Returns the stable policy identity (no pressure-level decoration).
    ///
    /// Where [`policy_name`](Self::policy_name) yields the log descriptor
    /// `h2o@Warning`, this yields the bare `h2o` — what diagnostic dumps record
    /// in their `technique` field so the lab can join/group by technique.
    pub fn policy_id(&self) -> String {
        self.pipeline.policy_id()
    }

    /// WHOLE-MODEL cross-layer keepset eviction (TriAttention's global mode) — the cross-layer sibling
    /// of [`run_policy_eviction`]. BYPASSES the per-layer `EvictionPolicy`/`StageBackedPolicy` loop:
    /// it asserts uniform geometry across layers, then drives the stage's
    /// [`KVMutationStage::on_whole_model`] ONCE over all caches (via
    /// [`drive_cross_layer`](crate::stages::kv::mutation::drive_cross_layer) — an owned host-mirrored
    /// `CrossLayerStageCtx` + an `EngineModelCacheHandle` whose keep fans out to every layer). One
    /// keep-set, applied identically to all layers.
    ///
    /// `positions` are the engine's absolute positions of the resident slots (len == current_pos;
    /// identity `0..current` before any eviction, survivors' original positions after — the source of
    /// `round_start` in TriAttention). The caller supplies them (e.g. the eval loop's `saved_positions`),
    /// or `0..current` for the single-prefill / first-eviction frame.
    pub(crate) fn run_cross_layer_keepset_eviction(
        stage: &dyn KVMutationStage,
        caches: &mut [KVCache],
        target_len: usize,
        positions: &[usize],
    ) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }
        // Uniform-geometry precondition: every layer shares the same resident length (the whole-model
        // keep-set is applied identically to all layers). Bail cleanly rather than mis-evict on a
        // ragged (per-layer-budgeted) cache.
        let current_pos = caches[0].current_pos();
        if !caches.iter().all(|c| c.current_pos() == current_pos) {
            anyhow::bail!(
                "cross-layer keepset eviction requires a uniform resident length across layers \
                 (found mixed current_pos); whole-model stages are incompatible with per-layer budgets"
            );
        }
        // Global guard (the uniform path): nothing to remove.
        if target_len > 0 && current_pos <= target_len {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: current_pos,
            });
        }
        let mutated = drive_cross_layer(stage, caches, target_len, positions)?;
        let new_pos = max_cache_pos(caches);
        Ok(EvictionResult {
            evicted: mutated && new_pos < current_pos,
            tokens_removed: current_pos.saturating_sub(new_pos),
            new_pos,
        })
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
        whole_model_positions: Option<&[usize]>,
    ) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        // WHOLE-MODEL routing: a stage that decides over all layers at once (caps `whole_model`, e.g.
        // TriAttention's global mode) bypasses the per-layer loop below. Per-layer policies (the common
        // case) return `None` here → the loop runs unchanged (byte-identical). `whole_model_positions`
        // are the engine's absolute positions of the resident slots: the survivors' original positions +
        // new tokens after a prior eviction (multi-round, e.g. the eval loop's `saved_positions`), or
        // `None` → the single-prefill identity frame `0..current` (round-1, where slot index == absolute
        // position — and identity IS the round-1 `saved_positions`). The position frame drives
        // TriAttention's `round_start`, so threading the real positions keeps multi-round scoring
        // faithful instead of mis-dating survivors as a fresh `0..current`.
        if let Some((stage, _caps)) = policy.as_whole_model_stage() {
            let current = max_cache_pos(caches);
            let identity: Vec<usize> = (0..current).collect();
            let positions = whole_model_positions.unwrap_or(&identity);
            return Self::run_cross_layer_keepset_eviction(stage, caches, target_len, positions);
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

        let (importance, head_importance, n_kv_heads, last_attn, per_layer_flat) = match &scores {
            ScoreContext::None => (None, None, 0, None, None),
            ScoreContext::Flat {
                importance,
                last_attn,
            } => (Some(*importance), None, 0, *last_attn, None),
            ScoreContext::PerHead {
                flat,
                head,
                n_kv_heads,
            } => (Some(*flat), Some(*head), *n_kv_heads, None, None),
            ScoreContext::PerLayerFlat {
                layer_flat,
                max_seq,
                last_attn,
            } => (None, None, 0, *last_attn, Some((*layer_flat, *max_seq))),
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

            // Faithful-H2O `(b)` — per-`(layer, token)` FLAT importance: rank THIS layer's heavy
            // hitters on its own `max_seq` window (`&layer_flat[layer * max_seq ..][.. max_seq]`),
            // with no cross-layer MAX. Takes priority over the collapsed paths. On a malformed buffer
            // (slice out of range) degrade to the score-free recency fallback (`None`) — NOT the whole
            // multi-layer buffer, which would silently mis-rank this layer on the wrong window. The
            // invariant `layer_flat.len() == n_layers * max_seq` holds by construction (the producer
            // sizes it `total_layers * max_seq_len`), so the fallback is defensive, not a live path.
            if let Some((layer_flat, max_seq)) = per_layer_flat {
                let base = layer_idx * max_seq;
                let layer_imp = layer_flat.get(base..base + max_seq);
                debug_assert!(
                    layer_imp.is_some(),
                    "per-layer FLAT slice out of range: layer={layer_idx} max_seq={max_seq} len={}",
                    layer_flat.len()
                );
                policy.evict_layer(
                    cache,
                    layer_target_len,
                    layer_imp,
                    last_attn,
                    layer_idx,
                    n_layers,
                )?;
            } else if let (Some(flat), Some(head_imp)) = (importance, head_importance) {
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
            None,
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
            h2o_backed_policy(15, 15, 0), // keep hh=15 + recent=15 = 30 (faithful absolute budget)
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

    /// Build `n_layers` F32 caches of `max_seq` capacity with `pos` resident tokens where every
    /// element of position `p` equals `p` (so a survivor's ORIGINAL position is read directly off
    /// `k_buffer` after compaction). kv_heads=1, head_dim=4.
    fn make_pos_caches(n_layers: usize, pos: usize, max_seq: usize) -> Vec<KVCache> {
        let backend = Arc::new(CpuBackend::new());
        (0..n_layers)
            .map(|_| {
                let buf = max_seq * 4 * 4;
                let k = Tensor::new(
                    Shape::new(vec![1, max_seq, 1, 4]),
                    Arc::new(SharedBuffer::new(buf, DType::F32)),
                    backend.clone(),
                );
                let v = Tensor::new(
                    Shape::new(vec![1, max_seq, 1, 4]),
                    Arc::new(SharedBuffer::new(buf, DType::F32)),
                    backend.clone(),
                );
                let mut cache = KVCache::new(k, v, max_seq);
                cache.current_pos = pos;
                for p in 0..pos {
                    let off = cache.offset(p, 0);
                    let kb = cache.k_buffer.as_mut_slice::<f32>();
                    for d in 0..4 {
                        kb[off + d] = p as f32;
                    }
                }
                cache
            })
            .collect()
    }

    /// Faithful-H2O `(b)` — per-`(layer, token)` FLAT importance makes each layer evict on its OWN
    /// heavy hitters (no cross-layer MAX), so layer 0 and layer 1 keep DIFFERENT tokens. A token that
    /// is the heavy hitter only in layer 0 survives in layer 0's cache but is evicted from layer 1's
    /// cache, and vice-versa. The collapsed (MAX-combined) path gives every layer the SAME keep-set —
    /// the mutation-proof contrast asserted at the end.
    #[test]
    fn faithful_h2o_per_layer_evicts_divergent_keepsets() {
        use crate::kv::eviction::stage_registry::h2o_backed_policy;

        // hh_size=1 + recent_size=2 + prefix=0 → keep exactly 3 tokens: the single heavy hitter over
        // the evictable middle [0, pos-recent) plus the 2 recent. pos=100 so removed=97 ≥ MIN_EVICT(64).
        let (n_layers, pos, max_seq) = (2usize, 100usize, 200usize);
        let make_cm = || {
            CacheManager::new(
                h2o_backed_policy(1, 2, 0),
                Box::new(MockMonitor { available: 0 }),
                256 * 1024 * 1024,
                0.03, // target_len = (100*0.03).max(1) = 3 (h2o ignores it; abs budget = 3)
            )
        };

        // Per-(layer, token) FLAT importance: layer 0 ranks token 30 highest, layer 1 ranks token 60
        // highest — divergent heavy hitters (both in the evictable middle [0, 98)).
        let mut layer_flat = vec![0.0f32; n_layers * max_seq];
        layer_flat[0 * max_seq + 30] = 10.0; // layer 0 heavy hitter
        layer_flat[1 * max_seq + 60] = 20.0; // layer 1 heavy hitter

        // ── Faithful per-layer path ──
        let mut caches = make_pos_caches(n_layers, pos, max_seq);
        let res = make_cm()
            .force_evict_with_per_layer_scores(&mut caches, 0.03, &layer_flat, max_seq, None)
            .unwrap();
        assert!(res.evicted, "per-layer eviction must fire");

        // Read each layer's survivor original positions (K[p]==p, compacted to the front, h2o keeps
        // an ascending list so order is prefix∪heavy∪recent sorted).
        let survivors = |c: &KVCache| -> Vec<usize> {
            let n = c.current_pos;
            let kb = c.k_buffer.as_slice::<f32>();
            (0..n).map(|s| kb[c.offset(s, 0)] as usize).collect()
        };
        let l0 = survivors(&caches[0]);
        let l1 = survivors(&caches[1]);

        assert_eq!(
            l0,
            vec![30, 98, 99],
            "layer 0 keeps its own heavy hitter 30"
        );
        assert_eq!(
            l1,
            vec![60, 98, 99],
            "layer 1 keeps its own heavy hitter 60"
        );
        // The defining property: the two layers' kept-sets DIFFER (token 30 survives only in layer 0,
        // token 60 only in layer 1) — impossible under a single cross-layer-MAX-collapsed importance.
        assert_ne!(l0, l1, "faithful (b): per-layer kept-sets must diverge");
        assert!(
            l0.contains(&30) && !l1.contains(&30),
            "30 kept only in layer 0"
        );
        assert!(
            l1.contains(&60) && !l0.contains(&60),
            "60 kept only in layer 1"
        );

        // ── Mutation-proof: the collapsed (MAX-combined) flat gives EVERY layer the SAME keep-set. ──
        // Element-wise MAX over layers → token 60 (20.0) outranks token 30 (10.0) everywhere, so both
        // layers keep {60, 98, 99} and token 30 (layer-0-critical) is lost. This is exactly what (b)
        // fixes; if the per-layer routing silently collapsed, the first block would equal this one.
        let mut collapsed = vec![0.0f32; max_seq];
        for t in 0..max_seq {
            collapsed[t] = layer_flat[t].max(layer_flat[max_seq + t]);
        }
        let mut caches_c = make_pos_caches(n_layers, pos, max_seq);
        make_cm()
            .force_evict_with_scores(&mut caches_c, 0.03, &collapsed, None)
            .unwrap();
        let c0 = survivors(&caches_c[0]);
        let c1 = survivors(&caches_c[1]);
        assert_eq!(c0, c1, "collapsed path: all layers share one keep-set");
        assert_eq!(
            c0,
            vec![60, 98, 99],
            "collapsed keeps the global MAX hitter 60"
        );
        assert!(
            !c0.contains(&30),
            "collapsed loses layer-0-critical token 30"
        );
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
            h2o_backed_policy(15, 15, 0),
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
            h2o_backed_policy(15, 15, 0),
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
            h2o_backed_policy(15, 15, 0),
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
            handler: Box::new(EvictionHandler::new(h2o_backed_policy(15, 15, 0), 0.3)),
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
            handler: Box::new(EvictionHandler::new(h2o_backed_policy(15, 15, 0), 0.3)),
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

#[cfg(all(test, feature = "triattention"))]
mod cross_layer_parity_tests {
    //! LIVE-ENGINE parity for the cross-layer (whole-model) keepset seam. Drives the real engine
    //! [`CacheManager::run_cross_layer_keepset_eviction`] over KVCaches populated from the TriAttention
    //! reference oracle dump, and asserts the resulting keep-set is byte-identical
    //! (`symmetric_diff == 0`) to the reference Path-1 oracle — round 1 (identity positions) AND round
    //! 2 (non-identity survivors + new tokens). The decision flows through the ENGINE path
    //! (`EngineCrossLayerStageCtx` + `EngineModelCacheHandle` + `TriAttention::on_whole_model`), NOT the
    //! plugin's `compute_keepset_global` harness. Env-gated on `TRIATTN_FIXTURE_DIR` (no fixture → SKIP,
    //! so the default `cargo test` stays green); the fixture is the same one the plugin's `parity.rs`
    //! consumes.

    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::eviction::keepset_dump::{arm_capture, capture_test_lock, drain_capture};
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::collections::{BTreeMap, HashSet};
    use std::path::Path;
    use std::sync::Arc;
    use triattention::{Calib, TriAttention};

    fn parse_params(path: &Path) -> BTreeMap<String, String> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect()
    }
    fn read_ints(path: &Path) -> Vec<usize> {
        std::fs::read_to_string(path)
            .unwrap()
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect()
    }
    fn read_keys_flat(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }
    fn symmetric_diff(a: &[usize], b: &[usize]) -> usize {
        let sa: HashSet<usize> = a.iter().copied().collect();
        let sb: HashSet<usize> = b.iter().copied().collect();
        sa.symmetric_difference(&sb).count()
    }

    /// Build one layer's F32 SeqMajor KVCache, populating K so `dequant_snapshot` reproduces the
    /// fixture `[kv_head][slot][head_dim]` (`layer_keys[(kv*l_total + slot)*head_dim ..]`). V is left
    /// zeroed (TriAttention reads only Key).
    fn build_layer_cache(
        layer_keys: &[f32],
        n_kv: usize,
        l_total: usize,
        head_dim: usize,
    ) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let bytes = l_total * n_kv * head_dim * std::mem::size_of::<f32>();
        let shape = Shape::new(vec![1, l_total, n_kv, head_dim]);
        let mut c = KVCache::new(
            Tensor::new(
                shape.clone(),
                Arc::new(SharedBuffer::new(bytes, DType::F32)),
                backend.clone(),
            ),
            Tensor::new(
                shape,
                Arc::new(SharedBuffer::new(bytes, DType::F32)),
                backend,
            ),
            l_total,
        );
        c.set_current_pos(l_total);
        for kv in 0..n_kv {
            for slot in 0..l_total {
                let src = ((kv * l_total) + slot) * head_dim;
                let off = c.offset(slot, kv);
                let k = c.k_buffer.as_mut_slice::<f32>();
                k[off..off + head_dim].copy_from_slice(&layer_keys[src..src + head_dim]);
            }
        }
        c
    }

    fn run_round(
        dir: &Path,
        round: &str,
        params_file: &str,
        keys_file: &str,
        keepset_file: &str,
        positions_file: Option<&str>,
    ) {
        let p = parse_params(&dir.join(params_file));
        let g = |k: &str| p[k].parse::<usize>().unwrap();
        let (l_total, prefix_length, budget) = (g("L"), g("prefix_length"), g("budget"));
        let (num_layers, num_kv_heads, head_dim) =
            (g("num_layers"), g("num_kv_heads"), g("head_dim"));
        let theta: f32 = p["rope_theta"].parse().unwrap();
        let offset_max_length = g("offset_max_length");
        let normalize = g("normalize_scores") != 0;

        let calib = Calib::from_path(dir.join("qwen25_calib.bin").to_str().unwrap()).unwrap();
        let keys = read_keys_flat(&dir.join(keys_file));
        assert_eq!(
            keys.len(),
            num_layers * num_kv_heads * l_total * head_dim,
            "[{round}] keys size mismatch"
        );
        let oracle = read_ints(&dir.join(keepset_file));
        let positions: Vec<usize> = match positions_file {
            Some(f) => read_ints(&dir.join(f)),
            None => (0..l_total).collect(),
        };
        assert_eq!(positions.len(), l_total, "[{round}] positions length == L");

        // One KVCache per transformer layer, populated from the fixture's post-RoPE keys.
        let layer_stride = num_kv_heads * l_total * head_dim;
        let mut caches: Vec<KVCache> = (0..num_layers)
            .map(|layer| {
                build_layer_cache(
                    &keys[layer * layer_stride..(layer + 1) * layer_stride],
                    num_kv_heads,
                    l_total,
                    head_dim,
                )
            })
            .collect();

        let stage =
            TriAttention::with_calib(calib, prefix_length, offset_max_length, normalize, theta);

        // Drive the LIVE engine whole-model path and capture the applied keep-set (the model handle
        // records into the keepset dump exactly as the per-layer path does).
        let _guard = capture_test_lock();
        arm_capture();
        let result =
            CacheManager::run_cross_layer_keepset_eviction(&stage, &mut caches, budget, &positions)
                .unwrap();
        let captured = drain_capture();

        // The committed keep-set (LayerWide → per-head replication; take head 0 of a layer recorded at
        // this round's pre-eviction fingerprint seq_len == L).
        let mine: Vec<_> = captured.iter().filter(|c| c.seq_len == l_total).collect();
        assert!(
            !mine.is_empty(),
            "[{round}] whole-model commit recorded no keep-set at seq_len={l_total}"
        );
        let engine_keep: Vec<usize> = mine[0].keep[0].clone();

        let sd = symmetric_diff(&engine_keep, &oracle);
        println!(
            "[live-parity {round}] oracle_keep={} engine_keep={} symmetric_diff={} \
             (new_pos={}, removed={})",
            oracle.len(),
            engine_keep.len(),
            sd,
            result.new_pos,
            result.tokens_removed
        );
        assert_eq!(
            sd, 0,
            "[{round}] live engine whole-model keep-set must match the reference Path-1 oracle exactly"
        );
        assert_eq!(
            result.new_pos, budget,
            "[{round}] resident count after eviction == budget"
        );
        assert_eq!(
            caches[0].current_pos(),
            budget,
            "[{round}] layer 0 compacted to budget"
        );
    }

    fn fixture_dir() -> Option<String> {
        match std::env::var("TRIATTN_FIXTURE_DIR") {
            Ok(d) => Some(d),
            Err(_) => {
                eprintln!("[live-parity] TRIATTN_FIXTURE_DIR unset → SKIP");
                None
            }
        }
    }

    /// Round 1: identity coordinates (single-prefill faithful frame), via the live engine path.
    #[test]
    fn live_engine_round1_keepset_parity() {
        let Some(dir) = fixture_dir() else { return };
        run_round(
            Path::new(&dir),
            "round1",
            "oracle_params.txt",
            "oracle_keys.f32",
            "oracle_keepset.txt",
            None,
        );
    }

    /// Round 2: NON-IDENTITY coordinates (survivors at original positions + new decode tokens), via the
    /// live engine path — `round_start` is derived from the engine's absolute positions.
    #[test]
    fn live_engine_round2_keepset_parity() {
        let Some(dir) = fixture_dir() else { return };
        let dir = Path::new(&dir);
        if !dir.join("oracle_params_r2.txt").exists() {
            eprintln!("[live-parity round2] no round-2 fixture → SKIP");
            return;
        }
        run_round(
            dir,
            "round2",
            "oracle_params_r2.txt",
            "oracle_keys_r2.f32",
            "oracle_keepset_r2.txt",
            Some("oracle_positions_r2.txt"),
        );
    }
}

/// P1: the PRODUCTION eval eviction path fires TriAttention's whole-model keep-set end-to-end.
///
/// Proves the full composition the eval loop uses — with NO bespoke harness call:
/// 1. `make_stage_backed_policy("triattention", .., calib_path)` — the EXACT constructor
///    `eval_setup.rs` resolves any `--eviction-policy <name> --set calib_path=…` through (generic by
///    name; the calib rides the opaque `StageArgs` blob). The `whole_model = true` cap travels with it.
/// 2. `CacheManager::force_evict_with_scores` — the entry the eval hook actually uses. TriAttention IS
///    score-based (`caps.reads ∋ Key` → `stage_is_score_based` is true), so `post_prefill` extracts
///    scores and `score_fed::route_evict(scores=Some(..), .., true, ratio)` dispatches the
///    `ScoreContext::Flat` variant. Those scores are then DISCARDED for a whole-model stage:
///    `run_policy_eviction` consults `as_whole_model_stage()` and short-circuits to the cross-layer path
///    BEFORE any `ScoreContext` is read — so `force_evict` (None) and `force_evict_with_scores` (Flat)
///    yield the IDENTICAL keep-set (the test pins that invariant too).
/// 3. inside, `run_policy_eviction` → `as_whole_model_stage()` → `Some` → routes to
///    `run_cross_layer_keepset_eviction`, whose ONE union keep-set fans out to every layer.
///
/// Self-contained: a deterministic synthetic calib written to a temp file (no external fixture), so it
/// runs under `cargo test --workspace --features triattention`. Asserts the cache shrinks to budget on
/// EVERY layer and that the keep-set (recovered from cache content) is identical across layers (the
/// whole-model invariant). Mutation-proof: if the whole-model routing regressed to the per-layer loop,
/// the keep-sets would differ per layer (each layer ranks its own keys), breaking the cross-layer assert.
#[cfg(all(test, feature = "triattention"))]
mod cross_layer_production_tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::eviction::stage_registry::make_stage_backed_policy;
    use crate::memory::host::shared::SharedBuffer;
    use crate::resilience::sys_monitor::NoOpMonitor;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    const N_LAYERS: usize = 3;
    const N_KV: usize = 2;
    const HD: usize = 64; // freq_count = HD/2 = 32
    const CALIB_HEADS: usize = 2; // attention heads (== N_KV → num_kv_groups = 1)
    const RESIDENT: usize = 20;
    const MAX_SEQ: usize = 32;
    const PREFIX: usize = 4;

    /// Serialize a deterministic synthetic calib (`TACALIB1`, the format `Calib::from_bytes` parses) for
    /// `N_LAYERS × CALIB_HEADS × (HD/2)`. Values are non-degenerate so the score ordering is well-defined.
    fn write_synthetic_calib(path: &std::path::Path) {
        let fc = HD / 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"TACALIB1");
        bytes.extend_from_slice(&(N_LAYERS as u32).to_le_bytes());
        bytes.extend_from_slice(&(CALIB_HEADS as u32).to_le_bytes());
        bytes.extend_from_slice(&(fc as u32).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved
        for l in 0..N_LAYERS {
            for h in 0..CALIB_HEADS {
                for which in 0..3 {
                    for f in 0..fc {
                        let v = ((l * 7 + h * 3 + which * 2 + f) % 13) as f32 * 0.1 - 0.6;
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        std::fs::write(path, &bytes).unwrap();
    }

    /// An F32 SeqMajor cache with `RESIDENT` tokens; K = a deterministic UNIT-NORM per-(slot,head) key
    /// whose per-channel content is strongly position-dependent (the spatial frequency across channels
    /// scales with `pos`). Unit-norm so the TriAttention score is PHASE-driven, not magnitude-dominated —
    /// which is what lets the position frame (`round_start`) reorder the union top-k (so P2's
    /// mutation-proof can observe that `saved_positions` actually changes the keep-set). V left zero
    /// (TriAttention reads only Key).
    fn build_cache(layer: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let total = MAX_SEQ * N_KV * HD;
        let shape = Shape::new(vec![1, MAX_SEQ, N_KV, HD]);
        let mut c = KVCache::new(
            Tensor::new(
                shape.clone(),
                Arc::new(SharedBuffer::new(total * 4, DType::F32)),
                backend.clone(),
            ),
            Tensor::new(
                shape,
                Arc::new(SharedBuffer::new(total * 4, DType::F32)),
                backend,
            ),
            MAX_SEQ,
        );
        c.set_current_pos(RESIDENT);
        for pos in 0..RESIDENT {
            for head in 0..N_KV {
                // Per-channel pattern with a position-scaled spatial frequency, then L2-normalized.
                let mut v = vec![0.0f32; HD];
                let freq = 0.07 * (pos as f32 + 1.0) + 0.013 * (head as f32 + 1.0);
                for (d, vd) in v.iter_mut().enumerate() {
                    *vd = (freq * d as f32 + 0.3 * layer as f32).sin();
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                let off = c.offset(pos, head);
                let kb = c.k_buffer.as_mut_slice::<f32>();
                for d in 0..HD {
                    kb[off + d] = v[d] / norm;
                }
            }
        }
        c
    }

    /// Head-0 key vector of cache slot `slot` (slot-unique by construction — see [`build_cache`]).
    fn read_k_head0(cache: &KVCache, slot: usize) -> Vec<f32> {
        let off = cache.offset(slot, 0);
        cache.k_buffer.as_slice::<f32>()[off..off + HD].to_vec()
    }

    /// After eviction, recover the ascending list of ORIGINAL slot indices that survived in `cache`
    /// (layer `layer`), by exact-matching each compacted slot's head-0 K against the deterministic
    /// originals. CpuBackend F32 is exact and eviction never re-rotates K, so a survivor's bytes equal
    /// its pre-eviction bytes — making this a fully LOCAL observation of the keep-set (no global
    /// keepset-dump, so it is immune to cross-test capture interference under `cargo test`'s parallelism).
    fn recovered_kept_slots(cache: &KVCache, layer: usize) -> Vec<usize> {
        let orig = build_cache(layer);
        (0..cache.current_pos())
            .map(|i| {
                let now = read_k_head0(cache, i);
                (0..RESIDENT)
                    .find(|&s| read_k_head0(&orig, s) == now)
                    .expect("each survivor matches exactly one original slot")
            })
            .collect()
    }

    #[test]
    fn eval_production_path_fires_whole_model_keepset() {
        let dir = std::env::temp_dir().join(format!("triattn_p1_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let calib_path = dir.join("synthetic_calib.bin");
        write_synthetic_calib(&calib_path);

        // (1) Production policy resolution by NAME + calib_path — exactly eval_setup.rs:187-211.
        let params = argus_extension_api::StageParams {
            eviction_window: 0,
            protected_prefix: PREFIX,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        };
        let extra = vec![argus_extension_api::PluginArg {
            key: "calib_path",
            val: calib_path.to_str().unwrap(),
        }];
        let policy = make_stage_backed_policy("triattention", &params, &extra).expect(
            "triattention resolves via the production registry (force-linked under feature)",
        );
        assert_eq!(policy.name(), "triattention");

        let cm = CacheManager::new(policy, Box::new(NoOpMonitor), 0, 0.5);
        let mut caches: Vec<KVCache> = (0..N_LAYERS).map(build_cache).collect();
        assert_eq!(max_cache_pos(&caches), RESIDENT);

        // (2) force_evict_with_scores = the entry the eval hook uses for TriAttention. It IS score-based
        //     (caps.reads ∋ Key), so post_prefill extracts scores and route_evict dispatches the
        //     ScoreContext::Flat variant. The whole-model branch in run_policy_eviction short-circuits on
        //     as_whole_model_stage() BEFORE the ScoreContext is read, so the scores are discarded.
        //     ratio 0.5 → target_len 10; decode_partition(20,10,prefix=4) → 4+6 = 10.
        let importance = vec![0.0f32; RESIDENT]; // discarded by the whole-model short-circuit
        let res = cm
            .force_evict_with_scores(&mut caches, 0.5, &importance, None)
            .expect("whole-model eviction succeeds via the score-based production entry");

        // (3) the whole-model keep-set fired and shrank every layer to budget.
        assert!(res.evicted, "whole-model keep-set must fire (evicted)");
        assert_eq!(
            res.new_pos, 10,
            "cache must shrink to budget (4 prefix + 6 decode)"
        );
        for (l, c) in caches.iter().enumerate() {
            assert_eq!(
                c.current_pos(),
                10,
                "layer {l} must shrink to budget — the single keep-set fans out to every layer"
            );
        }

        // The keep-set is IDENTICAL across layers (the whole-model invariant — a per-layer regression
        // would rank each layer's own keys and diverge). Recovered from cache content (local, no global
        // dump), so robust under parallel `cargo test`.
        let kept0 = recovered_kept_slots(&caches[0], 0);
        assert_eq!(kept0.len(), 10, "exactly budget slots kept");
        assert_eq!(&kept0[..PREFIX], &[0, 1, 2, 3], "prefill prefix pinned");
        for l in 1..N_LAYERS {
            assert_eq!(
                recovered_kept_slots(&caches[l], l),
                kept0,
                "whole-model keep-set must be identical across layers (layer {l})"
            );
        }

        // (4) the whole-model branch discards the ScoreContext: the score-FREE entry (force_evict,
        //     ScoreContext::None) produces the IDENTICAL keep-set. A regression that consumed scores
        //     before the as_whole_model_stage() short-circuit would diverge here.
        let mut caches_noscore: Vec<KVCache> = (0..N_LAYERS).map(build_cache).collect();
        cm.force_evict(&mut caches_noscore, 0.5)
            .expect("score-free entry also fires the whole-model keep-set");
        for l in 0..N_LAYERS {
            assert_eq!(
                recovered_kept_slots(&caches_noscore[l], l),
                recovered_kept_slots(&caches[l], l),
                "force_evict (None) and force_evict_with_scores (Flat) must yield the identical \
                 whole-model keep-set — scores are discarded by the branch (layer {l})"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P2: `run_policy_eviction`'s whole-model branch threads the caller's `saved_positions` (multi-round
    /// frame) into `run_cross_layer_keepset_eviction`, instead of always synthesizing the identity
    /// `0..current` frame. Proven by routing the SAME non-identity round-2 positions through two paths:
    /// - the DIRECT `run_cross_layer_keepset_eviction(stage, .., &positions)` — the exact call the
    ///   fixture round-2 parity test (`live_parity_round2`) trusts against the reference oracle;
    /// - the PRODUCTION `run_policy_eviction(policy, .., Some(&positions))` branch.
    /// The two keep-sets must be byte-identical (symmetric_diff = 0) → the production branch is round-2
    /// faithful (transitively, vs the oracle the direct path already matches). Mutation-proof: running
    /// the production branch with `None` (identity) yields a DIFFERENT keep-set, proving the branch
    /// actually consumes `saved_positions` rather than ignoring them (a frame that mis-dates survivors as
    /// `0..current` shifts every decode token's `round_start − pos`, reordering the union top-k).
    #[test]
    fn whole_model_branch_threads_saved_positions_round2() {
        let dir = std::env::temp_dir().join(format!("triattn_p2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let calib_path = dir.join("synthetic_calib.bin");
        write_synthetic_calib(&calib_path);
        let calib_str = calib_path.to_str().unwrap().to_string();
        let target = 10usize;

        // Round-2 NON-IDENTITY positions: survivors of a prior eviction (gaps) + recent decode tokens,
        // so the resident frame is NOT 0..current. round_start = max+1 = 208 ≫ current(20).
        let positions: Vec<usize> = vec![
            0, 1, 2, 3, 9, 17, 33, 64, 96, 128, 160, 192, 200, 201, 202, 203, 204, 205, 206, 207,
        ];
        assert_eq!(positions.len(), RESIDENT);

        // DIRECT cross-layer path (the oracle-trusted one) with these positions.
        let calib_direct = triattention::Calib::from_path(&calib_str).unwrap();
        let stage =
            triattention::TriAttention::with_calib(calib_direct, PREFIX, 65536, false, 1_000_000.0);
        let mut caches_direct: Vec<KVCache> = (0..N_LAYERS).map(build_cache).collect();
        let res_direct = CacheManager::run_cross_layer_keepset_eviction(
            &stage,
            &mut caches_direct,
            target,
            &positions,
        )
        .unwrap();

        // PRODUCTION branch: run_policy_eviction with Some(positions) — same StageBackedPolicy the eval
        // path resolves, routed through the production kernel.
        let params = argus_extension_api::StageParams {
            eviction_window: 0,
            protected_prefix: PREFIX,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        };
        let extra = vec![argus_extension_api::PluginArg {
            key: "calib_path",
            val: &calib_str,
        }];
        let policy = make_stage_backed_policy("triattention", &params, &extra).unwrap();
        let mut caches_prod: Vec<KVCache> = (0..N_LAYERS).map(build_cache).collect();
        let res_prod = CacheManager::run_policy_eviction(
            policy.as_ref(),
            &mut caches_prod,
            target,
            ScoreContext::None,
            None,
            Some(&positions),
        )
        .unwrap();

        // Mutation-proof setup: the SAME production branch with identity (None) frame.
        let mut caches_id: Vec<KVCache> = (0..N_LAYERS).map(build_cache).collect();
        let res_id = CacheManager::run_policy_eviction(
            policy.as_ref(),
            &mut caches_id,
            target,
            ScoreContext::None,
            None,
            None,
        )
        .unwrap();

        assert!(
            res_direct.evicted && res_prod.evicted && res_id.evicted,
            "all evict"
        );
        assert_eq!(res_prod.new_pos, res_direct.new_pos, "same shrink");

        // Keep-sets recovered from cache content (local — no global dump, robust under parallelism).
        let kept_direct = recovered_kept_slots(&caches_direct[0], 0);
        let kept_prod = recovered_kept_slots(&caches_prod[0], 0);
        let kept_id = recovered_kept_slots(&caches_id[0], 0);

        // The production branch, given saved_positions, reproduces the DIRECT path's keep-set exactly
        // (symmetric_diff = 0) — round-2 faithful (the direct path matches the reference oracle in the
        // fixture round-2 parity test).
        assert_eq!(
            kept_prod, kept_direct,
            "production branch keep-set == direct path keep-set for the SAME round-2 positions"
        );
        // Mutation-proof: the identity (None) frame yields a DIFFERENT keep-set — the branch genuinely
        // consumes `saved_positions` (mis-dating survivors as 0..current reorders the union top-k).
        assert_ne!(
            kept_id, kept_prod,
            "identity-frame keep-set must differ from the saved_positions keep-set (positions are used)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
