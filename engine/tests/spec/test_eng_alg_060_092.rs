//! ENG-ALG-060 ~ ENG-ALG-092: DegradationEstimator + CachePressurePipeline
//!
//! PiecewiseLinear 함수, DegradationEstimator d_max clamp / EMA correction,
//! CachePressurePipeline 실행 로직, target_bits_for_pressure / SwapHandler / EvictionHandler.

// `max_seq * 1 * 4 * 4` / `100 * 1 * 4 * 4` 계산식은 코드 가독성을 위해
// kv_heads=1, head_dim=4, bytes_per_elem=4 항목을 명시적으로 표기함.
#![allow(clippy::identity_op)]

use argus_engine::kv::quantize_handler::target_bits_for_pressure;
use argus_engine::qcf::QcfMetric;
use argus_engine::qcf::estimator::{DegradationEstimator, PiecewiseLinear};
use argus_shared::Level as PressureLevel;
use std::collections::HashMap;

// ══════════════════════════════════════════════════════════════
// ENG-ALG-060: PiecewiseLinear 함수 평가
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_060_piecewise_linear_below_breakpoint() {
    let pw = PiecewiseLinear::new(0.3, 2.0, 8.0);
    assert!((pw.evaluate(0.1) - 0.2).abs() < 1e-6);
    assert!((pw.evaluate(0.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_eng_alg_060_piecewise_linear_above_breakpoint() {
    let pw = PiecewiseLinear::new(0.3, 2.0, 8.0);
    // f(0.5) = 2.0*0.3 + 8.0*(0.5-0.3) = 0.6 + 1.6 = 2.2
    assert!((pw.evaluate(0.5) - 2.2).abs() < 1e-5);
}

#[test]
fn test_eng_alg_060_piecewise_linear_at_breakpoint() {
    let pw = PiecewiseLinear::new(0.3, 2.0, 8.0);
    assert!((pw.evaluate(0.3) - 0.6).abs() < 1e-6);
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-060: DegradationEstimator — d_max clamp
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_060_estimator_d_max_clamp() {
    let est = DegradationEstimator::with_defaults(2.0);
    let metric = QcfMetric {
        action: "kv.evict_h2o".to_string(),
        raw_value: 5.0,
        normalized_value: 5.0,
        per_head: None,
        tokens_affected: 10,
    };
    let d = est.estimate(&metric);
    assert!((d - 2.0).abs() < 1e-6, "d_max=2.0으로 클램프되어야 함");
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-060: DegradationEstimator — unknown action fallback
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_060_estimator_unknown_action() {
    let est = DegradationEstimator::with_defaults(5.0);
    let metric = QcfMetric {
        action: "unknown_action".to_string(),
        raw_value: 0.5,
        normalized_value: 0.5,
        per_head: None,
        tokens_affected: 1,
    };
    let d = est.estimate(&metric);
    // fallback → linear slope=1.0
    assert!((d - 0.5).abs() < 1e-6);
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-060: EMA correction
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_060_ema_correction() {
    let mut est = DegradationEstimator::new(
        {
            let mut m = HashMap::new();
            m.insert("kv.evict_h2o".to_string(), PiecewiseLinear::linear(2.0));
            m
        },
        10.0,
        0.5, // 공격적 EMA
    );

    // Predicted: 2.0 * 0.3 = 0.6, Actual: 1.2 → ratio = 2.0
    est.update_ema("kv.evict_h2o", 0.3, 1.2);
    let correction = est.ema_correction("kv.evict_h2o");
    // EMA: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
    assert!((correction - 1.5).abs() < 1e-5);

    // correction 적용 후 estimate
    let metric = QcfMetric {
        action: "kv.evict_h2o".to_string(),
        raw_value: 0.3,
        normalized_value: 0.3,
        per_head: None,
        tokens_affected: 5,
    };
    let d = est.estimate(&metric);
    // 2.0 * 0.3 * 1.5 = 0.9
    assert!((d - 0.9).abs() < 1e-5);
}

#[test]
fn test_eng_alg_060_ema_no_update_when_alpha_zero() {
    let mut est = DegradationEstimator::new(
        {
            let mut m = HashMap::new();
            m.insert("kv.evict_h2o".to_string(), PiecewiseLinear::linear(1.0));
            m
        },
        5.0,
        0.0, // no EMA
    );

    est.update_ema("kv.evict_h2o", 0.3, 1.2);
    assert_eq!(est.ema_correction("kv.evict_h2o"), 1.0);
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-091: CachePressurePipeline — 매칭 스테이지 실행
// (파이프라인 생성에 KVCache가 필요하므로 CpuBackend/SharedBuffer 사용)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_091_pipeline_executes_matching_stages() {
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::buffer::DType;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::kv::{
        ActionResult, CachePressureHandler, CachePressurePipeline, HandlerContext,
        PressureStageConfig,
    };
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingHandler {
        count: Arc<AtomicUsize>,
    }
    impl CachePressureHandler for CountingHandler {
        fn handle(&self, _ctx: &mut HandlerContext) -> anyhow::Result<ActionResult> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(ActionResult::NoOp)
        }
        fn name(&self) -> &str {
            "counter"
        }
    }

    let c_warn = Arc::new(AtomicUsize::new(0));
    let c_crit = Arc::new(AtomicUsize::new(0));
    let c_emerg = Arc::new(AtomicUsize::new(0));

    let pipeline = CachePressurePipeline::new(vec![
        PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(CountingHandler {
                count: c_warn.clone(),
            }),
        },
        PressureStageConfig {
            min_level: PressureLevel::Critical,
            handler: Box::new(CountingHandler {
                count: c_crit.clone(),
            }),
        },
        PressureStageConfig {
            min_level: PressureLevel::Emergency,
            handler: Box::new(CountingHandler {
                count: c_emerg.clone(),
            }),
        },
    ]);

    let backend = Arc::new(CpuBackend::new());
    let make_cache = |pos: usize| -> KVCache {
        let max_seq = 100;
        let buf_size = max_seq * 1 * 4 * 4;
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
        let mut c = KVCache::new(k, v, max_seq);
        c.current_pos = pos;
        c
    };

    let mut caches = vec![make_cache(50), make_cache(50)];
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        last_attn: None,
        per_layer_flat: None,
        pressure_level: PressureLevel::Critical,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
    };

    let results = pipeline.execute(&mut ctx).unwrap();
    assert_eq!(c_warn.load(Ordering::SeqCst), 1);
    assert_eq!(c_crit.load(Ordering::SeqCst), 1);
    assert_eq!(c_emerg.load(Ordering::SeqCst), 0);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_eng_alg_091_pipeline_skips_all_at_normal() {
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::buffer::DType;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::kv::{
        ActionResult, CachePressureHandler, CachePressurePipeline, HandlerContext,
        PressureStageConfig,
    };
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingHandler {
        count: Arc<AtomicUsize>,
    }
    impl CachePressureHandler for CountingHandler {
        fn handle(&self, _ctx: &mut HandlerContext) -> anyhow::Result<ActionResult> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(ActionResult::NoOp)
        }
        fn name(&self) -> &str {
            "counter"
        }
    }

    let c1 = Arc::new(AtomicUsize::new(0));
    let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
        min_level: PressureLevel::Warning,
        handler: Box::new(CountingHandler { count: c1.clone() }),
    }]);

    let backend = Arc::new(CpuBackend::new());
    let buf_size = 100 * 1 * 4 * 4;
    let k = Tensor::new(
        Shape::new(vec![1, 100, 1, 4]),
        Arc::new(SharedBuffer::new(buf_size, DType::F32)),
        backend.clone(),
    );
    let v = Tensor::new(
        Shape::new(vec![1, 100, 1, 4]),
        Arc::new(SharedBuffer::new(buf_size, DType::F32)),
        backend,
    );
    let mut cache = KVCache::new(k, v, 100);
    cache.current_pos = 30;
    let mut caches = vec![cache];

    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        last_attn: None,
        per_layer_flat: None,
        pressure_level: PressureLevel::Normal,
        mem_available: 1024 * 1024 * 1024,
        target_ratio: None,
        qcf_sink: None,
    };

    let results = pipeline.execute(&mut ctx).unwrap();
    assert_eq!(c1.load(Ordering::SeqCst), 0);
    assert!(results.is_empty());
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-091/C08: Pipeline ordering — 스테이지 정렬 검증
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_091_c08_pipeline_ordering_sorts_by_level() {
    use argus_engine::kv::{
        ActionResult, CachePressureHandler, CachePressurePipeline, HandlerContext,
        PressureStageConfig,
    };

    struct DummyHandler(&'static str);
    impl CachePressureHandler for DummyHandler {
        fn handle(&self, _ctx: &mut HandlerContext) -> anyhow::Result<ActionResult> {
            Ok(ActionResult::NoOp)
        }
        fn name(&self) -> &str {
            self.0
        }
    }

    // 역순으로 추가해도 정렬되어야 함
    let pipeline = CachePressurePipeline::new(vec![
        PressureStageConfig {
            min_level: PressureLevel::Emergency,
            handler: Box::new(DummyHandler("emerg")),
        },
        PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(DummyHandler("warn")),
        },
    ]);

    assert!(pipeline.name().starts_with("warn@Warning"));
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-092: target_bits_for_pressure — pressure level → target bits 매핑
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_092_target_bits_normal() {
    assert_eq!(target_bits_for_pressure(PressureLevel::Normal), None);
}

#[test]
fn test_eng_alg_092_target_bits_warning() {
    assert_eq!(target_bits_for_pressure(PressureLevel::Warning), Some(8));
}

#[test]
fn test_eng_alg_092_target_bits_critical() {
    assert_eq!(target_bits_for_pressure(PressureLevel::Critical), Some(4));
}

#[test]
fn test_eng_alg_092_target_bits_emergency() {
    assert_eq!(target_bits_for_pressure(PressureLevel::Emergency), Some(2));
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-092: SwapHandler — pressure level별 동작
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_092_swap_warning_offloads() {
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::buffer::DType;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::kv::{CachePressureHandler, HandlerContext, SwapHandler};
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;
    use std::sync::Arc;

    let backend = Arc::new(CpuBackend::new());
    let buf_size = 100 * 1 * 4 * 4;
    let k = Tensor::new(
        Shape::new(vec![1, 100, 1, 4]),
        Arc::new(SharedBuffer::new(buf_size, DType::F32)),
        backend.clone(),
    );
    let v = Tensor::new(
        Shape::new(vec![1, 100, 1, 4]),
        Arc::new(SharedBuffer::new(buf_size, DType::F32)),
        backend,
    );
    let mut cache = KVCache::new(k, v, 100);
    cache.current_pos = 50;
    let mut caches = vec![cache];

    let handler = SwapHandler::new(0.5);
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        last_attn: None,
        per_layer_flat: None,
        pressure_level: PressureLevel::Warning,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
    };

    let result = handler.handle(&mut ctx).unwrap();
    assert!(result.is_action());
    assert_eq!(ctx.caches[0].current_pos, 25);
}

#[test]
fn test_eng_alg_092_swap_emergency_offloads() {
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::buffer::DType;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::kv::{CachePressureHandler, HandlerContext, SwapHandler};
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;
    use std::sync::Arc;

    let backend = Arc::new(CpuBackend::new());
    let buf_size = 100 * 1 * 4 * 4;
    let k = Tensor::new(
        Shape::new(vec![1, 100, 1, 4]),
        Arc::new(SharedBuffer::new(buf_size, DType::F32)),
        backend.clone(),
    );
    let v = Tensor::new(
        Shape::new(vec![1, 100, 1, 4]),
        Arc::new(SharedBuffer::new(buf_size, DType::F32)),
        backend,
    );
    let mut cache = KVCache::new(k, v, 100);
    cache.current_pos = 40;
    let mut caches = vec![cache];

    let handler = SwapHandler::new(0.75);
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        last_attn: None,
        per_layer_flat: None,
        pressure_level: PressureLevel::Emergency,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
    };

    let result = handler.handle(&mut ctx).unwrap();
    assert!(result.is_action());
    assert_eq!(ctx.caches[0].current_pos, 10);
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-092: EvictionHandler — SlidingWindow/H2O 래핑
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_092_eviction_handler_wraps_sliding_window() {
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::buffer::DType;
    use argus_engine::kv::eviction::stage_registry::sliding_backed_policy;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::kv::{ActionResult, CachePressureHandler, EvictionHandler, HandlerContext};
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;
    use std::sync::Arc;

    let backend = Arc::new(CpuBackend::new());
    let make_cache = |pos: usize| -> KVCache {
        let max_seq = 100;
        let buf_size = max_seq * 1 * 4 * 4;
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
        let mut c = KVCache::new(k, v, max_seq);
        c.current_pos = pos;
        c
    };

    // pos=100, ratio=0.3 → tokens_to_remove=70 >= MIN_EVICT_TOKENS(64) → guard passes.
    let handler = EvictionHandler::new(sliding_backed_policy(10, 0), 0.3);

    let mut caches: Vec<KVCache> = (0..4).map(|_| make_cache(100)).collect();
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        last_attn: None,
        per_layer_flat: None,
        pressure_level: PressureLevel::Critical,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
    };

    let result = handler.handle(&mut ctx).unwrap();
    match result {
        ActionResult::Evicted {
            tokens_removed,
            new_pos,
        } => {
            assert!(tokens_removed > 0);
            assert!(new_pos < 100);
            // SlidingWindow may clamp; verify significant reduction.
            assert!(tokens_removed >= 64);
        }
        _ => panic!("Expected Evicted"),
    }
}

#[test]
fn test_eng_alg_092_eviction_handler_wraps_h2o() {
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::buffer::DType;
    use argus_engine::kv::eviction::stage_registry::make_stage_backed_policy;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::kv::{ActionResult, CachePressureHandler, EvictionHandler, HandlerContext};
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;
    use argus_extension_api::StageParams;
    use std::sync::Arc;

    let backend = Arc::new(CpuBackend::new());
    let make_cache = |pos: usize| -> KVCache {
        let max_seq = 100;
        let buf_size = max_seq * 1 * 4 * 4;
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
        let mut c = KVCache::new(k, v, max_seq);
        c.current_pos = pos;
        c
    };

    // pos=100, ratio=0.3 → tokens_to_remove=70 >= MIN_EVICT_TOKENS(64) → guard passes.
    // Faithful H2O absolute budget: hh(15)+recent(15) = 30 = the old target (100 * 0.3).
    let h2o_policy = make_stage_backed_policy(
        "h2o",
        &StageParams {
            protected_prefix: 0,
            ..Default::default()
        },
        &[
            argus_extension_api::PluginArg {
                key: "hh_size",
                val: "15",
            },
            argus_extension_api::PluginArg {
                key: "recent_size",
                val: "15",
            },
        ],
    )
    .expect("h2o v3 stage registered");
    let handler = EvictionHandler::new(h2o_policy, 0.3);
    assert_eq!(handler.name(), "h2o");

    let mut caches: Vec<KVCache> = (0..4).map(|_| make_cache(100)).collect();
    let mut importance = vec![0.01f32; 100];
    importance[10] = 10.0;
    importance[20] = 9.0;
    importance[30] = 8.0;

    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: Some(&importance),
        head_importance: None,
        n_kv_heads: 0,
        last_attn: None,
        per_layer_flat: None,
        pressure_level: PressureLevel::Critical,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
    };

    let result = handler.handle(&mut ctx).unwrap();
    match result {
        ActionResult::Evicted {
            tokens_removed,
            new_pos,
        } => {
            assert!(tokens_removed > 0);
            assert_eq!(new_pos, 30); // target = 100 * 0.3 = 30
        }
        _ => panic!("Expected Evicted"),
    }
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-091/C08: context_updated_after_eviction (파이프라인 체이닝)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_091_c08_context_updated_after_eviction() {
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::buffer::DType;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::kv::{
        ActionResult, CachePressureHandler, CachePressurePipeline, HandlerContext,
        PressureStageConfig,
    };
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;
    use std::sync::Arc;

    // HalvingHandler: current_pos를 절반으로 줄임
    struct HalvingHandler;
    impl CachePressureHandler for HalvingHandler {
        fn handle(&self, ctx: &mut HandlerContext) -> anyhow::Result<ActionResult> {
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

    let backend = Arc::new(CpuBackend::new());
    let make_cache = |pos: usize| -> KVCache {
        let max_seq = 100;
        let buf_size = max_seq * 1 * 4 * 4;
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
        let mut c = KVCache::new(k, v, max_seq);
        c.current_pos = pos;
        c
    };

    let mut caches = vec![make_cache(40), make_cache(40)];
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        last_attn: None,
        per_layer_flat: None,
        pressure_level: PressureLevel::Critical,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
    };

    let results = pipeline.execute(&mut ctx).unwrap();
    assert_eq!(results.len(), 2);

    // 첫 halving: 40→20, 두 번째: 20→10
    match &results[0] {
        ActionResult::Evicted { new_pos, .. } => assert_eq!(*new_pos, 20),
        _ => panic!("Expected Evicted"),
    }
    match &results[1] {
        ActionResult::Evicted { new_pos, .. } => assert_eq!(*new_pos, 10),
        _ => panic!("Expected Evicted"),
    }

    for cache in ctx.caches.iter() {
        assert_eq!(cache.current_pos, 10);
    }
}
