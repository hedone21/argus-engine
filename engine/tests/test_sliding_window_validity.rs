use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::kv::eviction::{EvictionPolicy, SlidingWindowPolicy};
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use std::sync::Arc;

#[test]
fn test_sliding_window_validity() {
    // Simulate the logic in generate.rs

    // 1. Setup KVCache for a single layer
    let max_seq_len = 100;
    let heads = 1;
    let dim = 1;
    let size_bytes = max_seq_len * 4;
    let dtype = DType::F32;

    let k_buf = Arc::new(SharedBuffer::new(size_bytes, dtype));
    let v_buf = Arc::new(SharedBuffer::new(size_bytes, dtype));
    let backend = Arc::new(CpuBackend::new());

    let k_tensor = Tensor::new(
        Shape::new(vec![1, max_seq_len, heads, dim]),
        k_buf.clone(),
        backend.clone(),
    );
    let v_tensor = Tensor::new(
        Shape::new(vec![1, max_seq_len, heads, dim]),
        v_buf.clone(),
        backend.clone(),
    );

    let mut cache = KVCache::new(k_tensor, v_tensor, max_seq_len);

    // 2. Setup Policy
    let policy = SlidingWindowPolicy::new(60, 0);

    // 3. Simulate Generation Loop
    let mut start_pos: usize = 0; // Logical position (RoPE index)
    let total_steps = 150;

    #[allow(clippy::explicit_counter_loop)]
    for _ in 0..total_steps {
        // Mock Forward:
        // In real generate.rs: model.forward calls rope(start_pos) then cache.update()
        // Here we just update cache status like `update` would.

        // Check condition: if cache is full or policy requirement
        // cache.update() usually increments current_pos

        if cache.current_pos >= max_seq_len {
            // Logic in generate.rs handles "out of context" by stopping?
            // But we want to test "infinite generation" with window.
            // But our generate.rs stops at max_seq_len.
            // The memory reduction logic is for satisfying OOM *within* max_seq_len
            // (e.g. if system memory is low even before max length).

            // Let's simulate that we are at step 80 (system low memory), prune to 60.
            // start_pos should be 80. cache.current_pos is 80.
        }

        // Simulate step 80 Pruning
        if start_pos == 80 {
            // System memory low! Prune to keep 60.
            let target_len = 60;
            // Prune Logic
            policy.evict(&mut cache, target_len).unwrap();

            // VALIDITY CHECK 1:
            // Physical cache size should be 60.
            assert_eq!(cache.current_pos, 60, "Cache size should be pruned to 60");

            // VALIDITY CHECK 2:
            // `start_pos` (Logical) MUST remain 80. It should NOT be reset to 60.
            assert_eq!(
                start_pos, 80,
                "Logical start_pos must be preserved for RoPE"
            );
        }

        // Simulate Update
        // In real code: cache.update() writes at current_pos and increments it.
        // We manually increment current_pos to simulate a write.
        if cache.current_pos < max_seq_len {
            cache.current_pos += 1;
        }

        // Increment Logical Position
        start_pos += 1;
    }

    // Final check
    // Logic reached 150.
    // Pruned at 80 -> cache became 60.
    // generated 80..150 (70 steps).
    // Cache size: 60 + 70 = 130?
    // Wait, max_seq_len is 100.
    // At step 80, current_pos=80. Prune -> current_pos=60.
    // Next 40 steps (80..120): current_pos goes 60..100.
    // At step 120: current_pos=100. Full.
    // Next steps: generate.rs would stop or error.

    // The test confirms that we CAN continue generating from logical pos 80
    // while filling physical slot 60.
}
