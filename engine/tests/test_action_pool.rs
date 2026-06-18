//! Action Pool integration tests.
//!
//! Tests cross-action compatibility, mutual exclusion, and combined behavior
//! for the 8 action pool algorithms (W1-W3, C1, C4-C6, C8).

use argus_engine::inference::skip_config::SkipConfig;
use argus_engine::kv::kv_cache::{KVCache, KVLayout};
use argus_engine::kv::offload::store::OffloadStore;
use argus_engine::kv::quant_window_cache::QuantizedRecentWindowCache;
use argus_engine::kv::quantize_handler::target_bits_for_pressure;
use argus_engine::kv::{CachePressureHandler, HandlerContext, PressureLevel, SwapHandler};
use argus_engine::quant::{BlockKVQ4, BlockKVQ8, QKKV};

use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::{Buffer, DType};
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use std::sync::Arc;

// ── Test helpers ─────────────────────────────────────────────────────────────

fn make_seqmajor_cache(num_tokens: usize, kv_heads: usize, head_dim: usize) -> KVCache {
    let max_seq = 256;
    let backend = Arc::new(CpuBackend::new());
    let buf_size = max_seq * kv_heads * head_dim * 4;
    let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
    let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

    unsafe {
        let ptr = k_buf.as_mut_ptr() as *mut f32;
        for i in 0..num_tokens * kv_heads * head_dim {
            *ptr.add(i) = i as f32 * 0.01;
        }
    }

    let k = Tensor::new(
        Shape::new(vec![1, max_seq, kv_heads, head_dim]),
        k_buf,
        backend.clone(),
    );
    let v = Tensor::new(
        Shape::new(vec![1, max_seq, kv_heads, head_dim]),
        v_buf,
        backend,
    );
    let mut cache = KVCache::new(k, v, max_seq);
    cache.current_pos = num_tokens;
    cache
}

#[allow(dead_code)]
fn make_headmajor_cache(num_tokens: usize, kv_heads: usize, head_dim: usize) -> KVCache {
    let max_seq = 256;
    let backend = Arc::new(CpuBackend::new());
    let buf_size = max_seq * kv_heads * head_dim * 4;
    let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
    let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

    let k = Tensor::new(
        Shape::new(vec![1, kv_heads, max_seq, head_dim]),
        k_buf,
        backend.clone(),
    );
    let v = Tensor::new(
        Shape::new(vec![1, kv_heads, max_seq, head_dim]),
        v_buf,
        backend,
    );
    let mut cache = KVCache::new(k, v, max_seq).with_layout(KVLayout::HeadMajor);
    cache.current_pos = num_tokens;
    cache
}

// ── AP-4-1: Unit test coverage ───────────────────────────────────────────────

// `test_streaming_alias_parameters` was removed when SlidingWindow was extracted to the
// `sliding-window` plugin: it asserted `should_evict()` thresholds, but the stage no longer owns
// the WHEN decision (StageBackedPolicy delegates the trigger to the engine MIN_EVICT/target guard).
// The sink+window / recent-window keep-list specs are pinned by the `streaming-llm` and
// `sliding-window` crates' own unit tests.

#[test]
fn test_kivi_q4_q8_roundtrip_error_bounds() {
    let src: [f32; QKKV] = std::array::from_fn(|i| i as f32 * 0.1);

    // Q4: 16 levels → error ≈ range/30
    let q4 = BlockKVQ4::quantize(&src);
    let mut dst4 = [0.0f32; QKKV];
    q4.dequantize(&mut dst4);
    let max_err_4 = src
        .iter()
        .zip(dst4.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let range = 3.1f32;
    assert!(
        max_err_4 < range / 15.0 + 0.05,
        "Q4 max error {max_err_4} too high"
    );

    // Q8: 256 levels → error ≈ range/510
    let q8 = BlockKVQ8::quantize(&src);
    let mut dst8 = [0.0f32; QKKV];
    q8.dequantize(&mut dst8);
    let max_err_8 = src
        .iter()
        .zip(dst8.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err_8 < range / 255.0 + 0.05,
        "Q8 max error {max_err_8} too high"
    );

    // Q8 should have strictly lower error than Q4
    assert!(max_err_8 < max_err_4);
}

#[test]
fn test_kivi_transition_8_4_2_bounded_error() {
    let kv_heads = 2;
    let head_dim = 64;
    let mut cache = QuantizedRecentWindowCache::new_with_bits(kv_heads, head_dim, 256, 32, 8);

    // Fill 65 tokens → 2 flushes + 1 residual
    for i in 0..65 {
        let data = make_kivi_input(1, kv_heads, head_dim, i as f32 * 0.03);
        cache.update(&data.0, &data.1).unwrap();
    }
    assert_eq!(cache.bits(), 8);

    cache.transition_bits(4).unwrap();
    assert_eq!(cache.bits(), 4);

    cache.transition_bits(2).unwrap();
    assert_eq!(cache.bits(), 2);

    // Cache should still be usable
    let (k, _v) = cache.get_view();
    assert_eq!(k.shape().dims()[1], 65);
    // (QuantizedRecentWindowCache::get_view — inherent 0-arg, Phase α-K BC 5-E)
}

fn make_kivi_input(
    seq_len: usize,
    kv_heads: usize,
    head_dim: usize,
    base: f32,
) -> (Tensor, Tensor) {
    let n = seq_len * kv_heads * head_dim;
    let backend: Arc<dyn argus_engine::backend::Backend> = Arc::new(CpuBackend::new());

    let k_buf = Arc::new(SharedBuffer::new(n * 4, DType::F32));
    let v_buf = Arc::new(SharedBuffer::new(n * 4, DType::F32));
    unsafe {
        let kp = k_buf.as_mut_ptr() as *mut f32;
        let vp = v_buf.as_mut_ptr() as *mut f32;
        for i in 0..n {
            *kp.add(i) = base + i as f32 * 0.01;
            *vp.add(i) = base + i as f32 * 0.01 + 100.0;
        }
    }

    let shape = Shape::new(vec![1, seq_len, kv_heads, head_dim]);
    (
        Tensor::new(shape.clone(), k_buf, backend.clone()),
        Tensor::new(shape, v_buf, backend),
    )
}

#[test]
fn test_quantize_handler_pressure_mapping() {
    assert_eq!(target_bits_for_pressure(PressureLevel::Normal), None);
    assert_eq!(target_bits_for_pressure(PressureLevel::Warning), Some(8));
    assert_eq!(target_bits_for_pressure(PressureLevel::Critical), Some(4));
    assert_eq!(target_bits_for_pressure(PressureLevel::Emergency), Some(2));
}

// ── AP-4-2: Cross-action integration tests ──────────────────────────────────

#[test]
fn test_throttle_plus_eviction_independence() {
    // W3 (throttle) + C4 (eviction) are independent actions.
    // Throttle only inserts delay (tested in resilience tests).
    // Eviction modifies cache. They don't interfere.

    let mut cache = make_seqmajor_cache(50, 1, 4);
    use argus_engine::kv::eviction::stage_registry::sliding_backed_policy;

    let sliding = sliding_backed_policy(10, 4);
    // Evict while "throttled" (simulated — no actual delay needed for correctness test)
    sliding.evict(&mut cache, 14).unwrap();
    assert_eq!(cache.current_pos, 14);
    // Output correctness unaffected by delay presence
}

#[test]
fn test_swap_then_continue() {
    // W2 offload: SwapHandler prunes cache, then inference can continue
    let cache = make_seqmajor_cache(80, 1, 4);
    let handler = SwapHandler::new(0.5);
    let mut caches = vec![cache];
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        pressure_level: PressureLevel::Warning,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
    };
    let result = handler.handle(&mut ctx).unwrap();
    assert!(result.is_action());
    assert_eq!(ctx.caches[0].current_pos, 40);
    // Cache is still usable (not corrupted)
    assert!(ctx.caches[0].current_pos < ctx.caches[0].capacity());
}

#[test]
fn test_skip_config_validate_boundary() {
    let config = SkipConfig::uniform_init(16, 0.5);
    assert!(config.validate(16));
    assert!(!config.skip_attn(0));
    assert!(!config.skip_attn(15));
    assert!(!config.skip_mlp(0));
    assert!(!config.skip_mlp(15));
    assert!(config.total_skips() > 0);
}

#[test]
fn test_all_actions_data_flow() {
    // Smoke test: verify each action's data structures can be created,
    // configured, and invoked without crash.

    // C6: StreamingLLM (SlidingWindowPolicy with sink)
    use argus_engine::kv::eviction::stage_registry::sliding_backed_policy;
    let _streaming = sliding_backed_policy(2000, 4);

    // C8: KIVI multi-bit
    let cache = QuantizedRecentWindowCache::new_with_bits(8, 64, 2048, 32, 4);
    assert_eq!(cache.bits(), 4);

    // W2: DiskStore
    use argus_engine::kv::offload::disk_store::DiskStore;
    let dir = std::env::temp_dir().join("argus_engine_test_all_actions");
    let _ = std::fs::remove_dir_all(&dir);
    let mut store = DiskStore::new(dir.clone(), 0, 64).unwrap();
    store.store(&[0u8; 64], &[0u8; 64], 1).unwrap();
    assert_eq!(store.stored_tokens(), 1);
    let _ = std::fs::remove_dir_all(&dir);

    // W2: SwapHandler
    let _swap = SwapHandler::new(0.5);

    // C1: SkipConfig
    let skip = SkipConfig::uniform_init(16, 0.3);
    assert!(skip.validate(16));

    // C8: target_bits_for_pressure pressure mapping
    assert_eq!(target_bits_for_pressure(PressureLevel::Emergency), Some(2));
}

#[test]
fn test_disk_store_integration() {
    use argus_engine::kv::offload::disk_store::DiskStore;
    use argus_engine::kv::offload::store::OffloadStore;

    let dir = std::env::temp_dir().join("argus_engine_test_disk_integration");
    let _ = std::fs::remove_dir_all(&dir);
    let bpt = 32;

    let mut store = DiskStore::new(dir.clone(), 0, bpt).unwrap();

    // Simulate layer offload: 10 tokens
    let k_data: Vec<u8> = (0..10 * bpt).map(|i| (i % 256) as u8).collect();
    let v_data: Vec<u8> = (0..10 * bpt).map(|i| ((i + 128) % 256) as u8).collect();
    store.store(&k_data, &v_data, 10).unwrap();
    assert_eq!(store.stored_tokens(), 10);

    // Recall
    let mut k_buf = vec![0u8; 10 * bpt];
    let mut v_buf = vec![0u8; 10 * bpt];
    let n = store.load_into(&mut k_buf, &mut v_buf).unwrap();
    assert_eq!(n, 10);
    assert_eq!(k_buf, k_data);
    assert_eq!(v_buf, v_data);

    let _ = std::fs::remove_dir_all(&dir);
}
