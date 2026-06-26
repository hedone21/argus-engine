//! KV cache pressure management pipeline.
//!
//! Generalizes cache management beyond eviction: swap, quantize, merge,
//! compress, sparse attention, and future techniques share a common
//! `CachePressureHandler` trait and are orchestrated by
//! `CachePressurePipeline` according to memory pressure level.
//!
//! Step 4-D: L3-pressure 도메인 모듈 — `core/`에서 promote.
//! KV 캐시 상태(state) + eviction 정책 + 압력 핸들러를 한 디렉토리에 응집.

// State + manager + I/O
pub mod cache_manager;
pub mod eviction;
pub mod squeeze_budget;
// Engine-owned GPU buffer set lent to the KV compute capability (FORMAT Phase 2, Stage B).
pub(crate) mod gpu_kv_buffers;
pub mod kv_cache;
pub mod kv_migrate;
pub mod offload;
pub mod offload_format;
pub mod quant_window_cache;
// EPIC 4 PR3 de-risk: first in-tree QuantCacheBackend over the real cache (host-only harness).
pub mod quant_window_cache_abi_shim;
pub mod quant_window_format;
// D3 — engine-side executor for the format/precision axis (KVFormatPlan). Honest reject of
// heterogeneous-within-layer plans (no current container holds per-head/per-token precision).
pub mod format_apply;
// read 축(KVReadStage) 빌트인 등록 + force-link self-test (Quest).
pub mod read;
pub mod standard_format;

// Shared KV dequantization + similarity helpers (formerly in d2o_handler; weighted-merge was extracted to
// the out-of-tree `d2o` plugin crate). Consumed by the StageCtx Key/Value handles + R-KV stage.
pub(crate) mod dequant;
// Pressure pipeline handlers (구 core/pressure/ 내용 flat 병합)
pub mod eviction_handler;
pub mod quantize_handler;
// R-KV (rkv_stage) was extracted to the out-of-tree `rkv` technique crate (feature `rkv`,
// force-linked in eviction/stage_registry.rs) — like streaming/h2o/d2o/sliding/no-eviction.
pub mod swap_handler;
use crate::kv::kv_cache::KVCache;
use anyhow::Result;

pub use eviction_handler::{EvictionHandler, MIN_EVICT_TOKENS};
pub use quantize_handler::target_bits_for_pressure;
pub use swap_handler::SwapHandler;

// ── Pressure level ─────────────────────────────────────────────────

/// Memory pressure severity, reused from the shared signal type.
///
/// `Normal < Warning < Critical < Emergency` (derives `Ord`).
pub type PressureLevel = argus_shared::Level;

// ── Handler context ────────────────────────────────────────────────

/// Data passed to each handler during pipeline execution.
pub struct HandlerContext<'a> {
    /// KV caches (one per transformer layer).
    pub caches: &'a mut [KVCache],
    /// Optional per-token importance scores (from AttentionScoreAccumulator).
    pub importance: Option<&'a [f32]>,
    /// Optional per-KV-head importance scores for GQA-aware eviction.
    /// Layout: `[n_kv_heads * max_seq_len]`, row-major.
    pub head_importance: Option<&'a [f32]>,
    /// Number of KV heads (0 = GQA mode disabled).
    pub n_kv_heads: usize,
    /// Optional last-layer last-step per-(kv_head,pos) attention slice (value-aware's `a_i`).
    /// Layout: `[n_kv_heads * max_seq_len]`, row-major. `None` when no AttnWeights producer
    /// is active; the value-aware stage then falls back to flat `importance`.
    pub last_attn: Option<&'a [f32]>,
    /// Current pressure level determined by the pipeline.
    pub pressure_level: PressureLevel,
    /// Available system memory in bytes.
    pub mem_available: usize,
    /// Optional target ratio override from external signal (e.g., resilience Evict action).
    /// When set, handlers should use this instead of their internal config target_ratio.
    pub target_ratio: Option<f32>,
    /// Optional sink for proxy metrics collected during handler execution.
    /// When `Some`, handlers push `QcfMetric` values for degradation estimation.
    pub qcf_sink: Option<&'a mut Vec<crate::qcf_types::QcfMetric>>,
}

// ── Action result ──────────────────────────────────────────────────

pub use crate::action_result::ActionResult;

// ── Handler trait ──────────────────────────────────────────────────

/// A cache pressure handler — the generalized replacement for EvictionPolicy.
///
/// Each handler implements one cache management technique (eviction, swap,
/// quantize, merge, etc.). Handlers are composed into a pipeline via
/// `CachePressurePipeline` and executed in order based on pressure level.
pub trait CachePressureHandler: Send + Sync {
    /// Execute this handler's action.
    ///
    /// The handler should inspect `ctx.pressure_level` and cache state
    /// to decide whether to act. Return `ActionResult::NoOp` if no
    /// action is needed.
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult>;

    /// Handler name for logging and debugging.
    fn name(&self) -> &str;
}

// ── Stage config ───────────────────────────────────────────────────

/// Binds a handler to a minimum pressure level.
///
/// The pipeline activates this stage only when the current pressure level
/// is **at or above** `min_level`.
///
/// # Flexible configuration
///
/// Same handler with different coefficients per level:
/// ```text
/// stages = [
///     PressureStageConfig { min_level: Warning,   handler: Evict(ratio=0.8) },
///     PressureStageConfig { min_level: Emergency,  handler: Evict(ratio=0.5) },
/// ]
/// ```
///
/// Different handlers per level:
/// ```text
/// stages = [
///     PressureStageConfig { min_level: Warning,   handler: Quantize },
///     PressureStageConfig { min_level: Critical,  handler: Evict(ratio=0.7) },
/// ]
/// ```
pub struct PressureStageConfig {
    /// Minimum pressure level that activates this stage.
    pub min_level: PressureLevel,
    /// Handler to execute when this stage is active.
    pub handler: Box<dyn CachePressureHandler>,
}

// ── Pipeline ───────────────────────────────────────────────────────

/// Orchestrates multiple handlers in priority order based on pressure level.
///
/// Stages are sorted by `min_level` ascending (least aggressive first).
/// When executed at a given pressure level, all stages whose `min_level`
/// is at or below the current level run sequentially.
pub struct CachePressurePipeline {
    stages: Vec<PressureStageConfig>,
}

impl CachePressurePipeline {
    /// Create a pipeline from a list of stage configs.
    ///
    /// Stages are sorted by `min_level` ascending internally.
    pub fn new(mut stages: Vec<PressureStageConfig>) -> Self {
        stages.sort_by_key(|s| s.min_level);
        Self { stages }
    }

    /// Execute all matching stages for the given context.
    ///
    /// A stage matches if `stage.min_level <= ctx.pressure_level`.
    /// Stages run sequentially in ascending `min_level` order.
    /// Each handler sees the cache state left by the previous handler.
    pub fn execute(&self, ctx: &mut HandlerContext) -> Result<Vec<ActionResult>> {
        let mut results = Vec::new();
        for stage in &self.stages {
            if ctx.pressure_level >= stage.min_level {
                let result = stage.handler.handle(ctx)?;
                results.push(result);
            }
        }
        Ok(results)
    }

    /// Descriptive name for logging (concatenation of stage names).
    pub fn name(&self) -> String {
        if self.stages.is_empty() {
            return "empty_pipeline".to_string();
        }
        self.stages
            .iter()
            .map(|s| format!("{}@{:?}", s.handler.name(), s.min_level))
            .collect::<Vec<_>>()
            .join(" → ")
    }

    /// Stable policy identity for grouping/joining diagnostic records.
    ///
    /// Unlike [`name`](Self::name) — a logging descriptor that decorates each
    /// stage with the pressure level that arms it (`h2o@Warning`) — this returns
    /// just the handler identities (`h2o`), so a consumer can group records by
    /// technique without parsing out the `@<level>` tag. Multi-stage pipelines
    /// join the bare names with ` → `.
    pub fn policy_id(&self) -> String {
        if self.stages.is_empty() {
            return "empty_pipeline".to_string();
        }
        self.stages
            .iter()
            .map(|s| s.handler.name())
            .collect::<Vec<_>>()
            .join(" → ")
    }

    /// Number of stages in the pipeline.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Whether the pipeline has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ── Helpers ──

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

    /// Mock handler that records how many times it was called.
    struct CountingHandler {
        call_count: Arc<AtomicUsize>,
        handler_name: &'static str,
    }

    impl CountingHandler {
        fn new(name: &'static str) -> (Self, Arc<AtomicUsize>) {
            let count = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    call_count: count.clone(),
                    handler_name: name,
                },
                count,
            )
        }
    }

    impl CachePressureHandler for CountingHandler {
        fn handle(&self, _ctx: &mut HandlerContext) -> Result<ActionResult> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(ActionResult::NoOp)
        }
        fn name(&self) -> &str {
            self.handler_name
        }
    }

    /// Mock handler that performs eviction by halving current_pos.
    struct HalvingHandler;

    impl CachePressureHandler for HalvingHandler {
        fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
            if ctx.caches.is_empty() {
                return Ok(ActionResult::NoOp);
            }
            let before = ctx.caches[0].current_pos;
            let new_pos = before / 2;
            for cache in ctx.caches.iter_mut() {
                cache.current_pos = new_pos;
            }
            Ok(ActionResult::Evicted {
                tokens_removed: before - new_pos,
                new_pos,
            })
        }
        fn name(&self) -> &str {
            "halving"
        }
    }

    // ── Pipeline tests ──

    #[test]
    fn test_pipeline_executes_matching_stages() {
        let (h_warn, c_warn) = CountingHandler::new("warn_handler");
        let (h_crit, c_crit) = CountingHandler::new("crit_handler");
        let (h_emerg, c_emerg) = CountingHandler::new("emerg_handler");

        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(h_warn),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(h_crit),
            },
            PressureStageConfig {
                min_level: PressureLevel::Emergency,
                handler: Box::new(h_emerg),
            },
        ]);

        let mut caches = make_caches(2, 50);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
        };

        let results = pipeline.execute(&mut ctx).unwrap();

        // Warning and Critical should run, Emergency should not
        assert_eq!(c_warn.load(Ordering::SeqCst), 1);
        assert_eq!(c_crit.load(Ordering::SeqCst), 1);
        assert_eq!(c_emerg.load(Ordering::SeqCst), 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_pipeline_skips_all_at_normal() {
        let (h1, c1) = CountingHandler::new("h1");
        let (h2, c2) = CountingHandler::new("h2");

        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(h1),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(h2),
            },
        ]);

        let mut caches = make_caches(1, 30);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Normal,
            mem_available: 1024 * 1024 * 1024,
            target_ratio: None,
            qcf_sink: None,
        };

        let results = pipeline.execute(&mut ctx).unwrap();
        assert_eq!(c1.load(Ordering::SeqCst), 0);
        assert_eq!(c2.load(Ordering::SeqCst), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn policy_id_drops_pressure_level_decoration() {
        // The logging `name()` decorates each stage with the level that arms it;
        // `policy_id()` keeps only the bare handler identity so diagnostic records
        // can be grouped by technique (R1: dump `technique` was "h2o@Warning").
        let (h_warn, _c) = CountingHandler::new("h2o");
        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(h_warn),
        }]);
        assert_eq!(pipeline.name(), "h2o@Warning");
        assert_eq!(pipeline.policy_id(), "h2o");
        assert!(
            !pipeline.policy_id().contains('@'),
            "policy_id must not carry a status/level tag"
        );
        // Empty pipeline keeps a stable sentinel id.
        assert_eq!(
            CachePressurePipeline::new(vec![]).policy_id(),
            "empty_pipeline"
        );
    }

    #[test]
    fn test_pipeline_ordering_sorts_by_level() {
        // Add stages out of order — pipeline should sort them
        let (h_emerg, c_emerg) = CountingHandler::new("emerg");
        let (h_warn, c_warn) = CountingHandler::new("warn");

        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Emergency,
                handler: Box::new(h_emerg),
            },
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(h_warn),
            },
        ]);

        // Verify internal ordering via name()
        assert!(pipeline.name().starts_with("warn@Warning"));

        let mut caches = make_caches(1, 20);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
        };

        let results = pipeline.execute(&mut ctx).unwrap();
        assert_eq!(c_warn.load(Ordering::SeqCst), 1);
        assert_eq!(c_emerg.load(Ordering::SeqCst), 1);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = CachePressurePipeline::new(vec![]);

        let mut caches = make_caches(1, 20);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
        };

        let results = pipeline.execute(&mut ctx).unwrap();
        assert!(results.is_empty());
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.name(), "empty_pipeline");
    }

    #[test]
    fn test_context_updated_after_eviction() {
        // First handler halves the cache, second handler observes the reduced state
        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(HalvingHandler),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(HalvingHandler),
            },
        ]);

        let mut caches = make_caches(2, 40);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
        };

        let results = pipeline.execute(&mut ctx).unwrap();

        // First halving: 40 → 20, second halving: 20 → 10
        assert_eq!(results.len(), 2);
        match &results[0] {
            ActionResult::Evicted { new_pos, .. } => assert_eq!(*new_pos, 20),
            _ => panic!("Expected Evicted"),
        }
        match &results[1] {
            ActionResult::Evicted { new_pos, .. } => assert_eq!(*new_pos, 10),
            _ => panic!("Expected Evicted"),
        }

        // All caches should be at 10
        for cache in ctx.caches.iter() {
            assert_eq!(cache.current_pos, 10);
        }
    }

    #[test]
    fn test_action_result_is_action() {
        assert!(!ActionResult::NoOp.is_action());
        assert!(
            ActionResult::Evicted {
                tokens_removed: 5,
                new_pos: 10
            }
            .is_action()
        );
        assert!(ActionResult::Swapped { tokens_swapped: 0 }.is_action());
        assert!(
            ActionResult::WeightSwapped {
                layers_changed: 1,
                freed_bytes: 0,
                duration_ms: 0.0,
            }
            .is_action()
        );
    }

    #[test]
    fn test_pipeline_len() {
        let (h1, _) = CountingHandler::new("h1");
        let (h2, _) = CountingHandler::new("h2");

        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(h1),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(h2),
            },
        ]);

        assert_eq!(pipeline.len(), 2);
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn test_same_level_multiple_handlers() {
        // Two handlers at the same level — both should run
        let (h1, c1) = CountingHandler::new("first");
        let (h2, c2) = CountingHandler::new("second");

        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(h1),
            },
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(h2),
            },
        ]);

        let mut caches = make_caches(1, 20);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            last_attn: None,
            pressure_level: PressureLevel::Warning,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
        };

        let results = pipeline.execute(&mut ctx).unwrap();
        assert_eq!(c1.load(Ordering::SeqCst), 1);
        assert_eq!(c2.load(Ordering::SeqCst), 1);
        assert_eq!(results.len(), 2);
    }
}
