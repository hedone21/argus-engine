//! Integration test: KV cache eviction → memory reduction
//!
//! Tests the eviction pipeline directly:
//!   apply_eviction(caches, ratio) → CacheManager → prune_prefix
//!
//! MemoryStrategy was removed in α-W-3. Memory pressure no longer produces
//! discrete Evict commands. This test exercises the eviction helper directly
//! using the ratios that MemoryStrategy previously mapped:
//!   Warning→0.85, Critical→0.50, Emergency→0.25.
//!
//! Outputs JSON data to results/data/eviction_memory_test.json for visualization.

#[cfg(feature = "resilience")]
mod eviction_memory_test {
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::buffer::{Buffer, DType};
    use argus_engine::kv::cache_manager::CacheManager;
    use argus_engine::kv::eviction::stage_registry::sliding_backed_policy;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;

    use std::sync::Arc;

    /// Llama 3.2 1B-like config
    const NUM_LAYERS: usize = 16;
    const NUM_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 64;
    const MAX_SEQ_LEN: usize = 2048;

    /// Mock SystemMonitor that reports zero available memory (always triggers eviction threshold)
    struct MockMonitor;

    impl argus_engine::resilience::sys_monitor::SystemMonitor for MockMonitor {
        fn mem_stats(&self) -> anyhow::Result<argus_engine::resilience::sys_monitor::MemoryStats> {
            Ok(argus_engine::resilience::sys_monitor::MemoryStats {
                total: 4 * 1024 * 1024 * 1024,
                available: 0,
                free: 0,
            })
        }
    }

    /// Create multi-layer KV caches (Llama 3.2 1B-like)
    fn make_caches(num_layers: usize) -> Vec<KVCache> {
        let backend = Arc::new(CpuBackend::new());
        (0..num_layers)
            .map(|_| {
                let buf_size = MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * DType::F32.size();
                let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
                let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

                let k = Tensor::new(
                    Shape::new(vec![1, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]),
                    k_buf,
                    backend.clone(),
                );
                let v = Tensor::new(
                    Shape::new(vec![1, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]),
                    v_buf,
                    backend.clone(),
                );
                KVCache::new(k, v, MAX_SEQ_LEN)
            })
            .collect()
    }

    /// Fill one token into each layer cache by writing recognizable data
    fn fill_token(caches: &mut [KVCache], backend: &Arc<CpuBackend>) {
        let elements = NUM_KV_HEADS * HEAD_DIM;
        let buf_size = elements * DType::F32.size();
        for cache in caches.iter_mut() {
            let pos = cache.current_pos;
            let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
            let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
            // Write pos-based pattern for verification
            unsafe {
                let k_ptr = k_buf.as_mut_ptr() as *mut f32;
                let v_ptr = v_buf.as_mut_ptr() as *mut f32;
                for i in 0..elements {
                    *k_ptr.add(i) = (pos + 1) as f32;
                    *v_ptr.add(i) = (pos + 1) as f32 * 10.0;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, NUM_KV_HEADS, HEAD_DIM]),
                k_buf,
                backend.clone(),
            );
            let v = Tensor::new(
                Shape::new(vec![1, 1, NUM_KV_HEADS, HEAD_DIM]),
                v_buf,
                backend.clone(),
            );
            cache.update(&k, &v).unwrap();
        }
    }

    /// Total memory across all layer caches
    fn total_memory_bytes(caches: &[KVCache]) -> usize {
        caches.iter().map(|c| c.memory_usage_bytes()).sum()
    }

    /// Apply eviction via CacheManager with the given target_ratio
    fn apply_eviction(caches: &mut [KVCache], target_ratio: f32) {
        // Use sliding window policy with large window (so CacheManager target_ratio controls eviction)
        let window_size = MAX_SEQ_LEN;
        let policy = sliding_backed_policy(window_size, 0);

        let monitor = MockMonitor;

        let cm = CacheManager::new(
            policy,
            Box::new(monitor),
            usize::MAX, // threshold: always triggers
            target_ratio,
        );

        let result = cm.maybe_evict(caches).unwrap();
        if result.evicted {
            println!(
                "  Evicted: {} tokens removed, new_pos={}",
                result.tokens_removed, result.new_pos
            );
        }
    }

    /// Data point for JSON output
    #[derive(serde::Serialize)]
    struct DataPoint {
        step: usize,
        event: String,
        tokens: usize,
        memory_kb: f64,
        memory_mb: f64,
        level: String,
    }

    /// Full scenario result
    #[derive(serde::Serialize)]
    struct ScenarioResult {
        name: String,
        description: String,
        config: ScenarioConfig,
        data: Vec<DataPoint>,
    }

    #[derive(serde::Serialize)]
    struct ScenarioConfig {
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        per_token_bytes: usize,
    }

    #[test]
    fn test_eviction_reduces_memory_on_pressure() {
        let backend = Arc::new(CpuBackend::new());
        let mut caches = make_caches(NUM_LAYERS);
        let mut data_points: Vec<DataPoint> = Vec::new();

        // Per-token memory: NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * 4bytes(F32) * 2(K+V)
        let per_token_bytes = NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * 4 * 2;

        // ── Phase 1: Fill 200 tokens (growth phase) ──
        let mut current_level = "normal".to_string();
        for i in 0..200 {
            fill_token(&mut caches, &backend);
            // Record every 10 tokens
            if (i + 1) % 10 == 0 {
                let mem = total_memory_bytes(&caches);
                data_points.push(DataPoint {
                    step: data_points.len(),
                    event: "fill".to_string(),
                    tokens: caches[0].current_pos,
                    memory_kb: mem as f64 / 1024.0,
                    memory_mb: mem as f64 / (1024.0 * 1024.0),
                    level: current_level.clone(),
                });
            }
        }
        assert_eq!(caches[0].current_pos, 200);

        // ── Phase 2: Warning → Evict 85% directly ──
        current_level = "warning".to_string();

        // Record pre-eviction
        let mem_before = total_memory_bytes(&caches);
        data_points.push(DataPoint {
            step: data_points.len(),
            event: "pre_eviction_warning".to_string(),
            tokens: caches[0].current_pos,
            memory_kb: mem_before as f64 / 1024.0,
            memory_mb: mem_before as f64 / (1024.0 * 1024.0),
            level: current_level.clone(),
        });

        // Warning level → ratio 0.85 (matches former MemoryStrategy::Warning mapping)
        apply_eviction(&mut caches, 0.85);

        let mem_after = total_memory_bytes(&caches);
        let pos_after_warning = caches[0].current_pos;
        data_points.push(DataPoint {
            step: data_points.len(),
            event: "post_eviction_warning".to_string(),
            tokens: pos_after_warning,
            memory_kb: mem_after as f64 / 1024.0,
            memory_mb: mem_after as f64 / (1024.0 * 1024.0),
            level: current_level.clone(),
        });

        assert!(
            mem_after < mem_before,
            "Memory should decrease after warning eviction: {} >= {}",
            mem_after,
            mem_before
        );
        assert!(
            pos_after_warning < 200,
            "Tokens should be reduced from 200, got {}",
            pos_after_warning
        );

        // ── Phase 3: Continue filling 100 more tokens ──
        for i in 0..100 {
            fill_token(&mut caches, &backend);
            if (i + 1) % 10 == 0 {
                let mem = total_memory_bytes(&caches);
                data_points.push(DataPoint {
                    step: data_points.len(),
                    event: "fill".to_string(),
                    tokens: caches[0].current_pos,
                    memory_kb: mem as f64 / 1024.0,
                    memory_mb: mem as f64 / (1024.0 * 1024.0),
                    level: current_level.clone(),
                });
            }
        }

        // ── Phase 4: Critical → Evict 50% directly ──
        current_level = "critical".to_string();

        let mem_before_crit = total_memory_bytes(&caches);
        let pos_before_crit = caches[0].current_pos;
        data_points.push(DataPoint {
            step: data_points.len(),
            event: "pre_eviction_critical".to_string(),
            tokens: pos_before_crit,
            memory_kb: mem_before_crit as f64 / 1024.0,
            memory_mb: mem_before_crit as f64 / (1024.0 * 1024.0),
            level: current_level.clone(),
        });

        // Critical level → ratio 0.50
        apply_eviction(&mut caches, 0.50);

        let mem_after_crit = total_memory_bytes(&caches);
        let pos_after_crit = caches[0].current_pos;
        data_points.push(DataPoint {
            step: data_points.len(),
            event: "post_eviction_critical".to_string(),
            tokens: pos_after_crit,
            memory_kb: mem_after_crit as f64 / 1024.0,
            memory_mb: mem_after_crit as f64 / (1024.0 * 1024.0),
            level: current_level.clone(),
        });

        assert!(
            mem_after_crit < mem_before_crit,
            "Memory should decrease after critical eviction"
        );

        // ── Phase 5: Fill more, then Emergency → Evict 25% directly ──
        for i in 0..80 {
            fill_token(&mut caches, &backend);
            if (i + 1) % 10 == 0 {
                let mem = total_memory_bytes(&caches);
                data_points.push(DataPoint {
                    step: data_points.len(),
                    event: "fill".to_string(),
                    tokens: caches[0].current_pos,
                    memory_kb: mem as f64 / 1024.0,
                    memory_mb: mem as f64 / (1024.0 * 1024.0),
                    level: current_level.clone(),
                });
            }
        }

        current_level = "emergency".to_string();

        let mem_before_emerg = total_memory_bytes(&caches);
        let pos_before_emerg = caches[0].current_pos;
        data_points.push(DataPoint {
            step: data_points.len(),
            event: "pre_eviction_emergency".to_string(),
            tokens: pos_before_emerg,
            memory_kb: mem_before_emerg as f64 / 1024.0,
            memory_mb: mem_before_emerg as f64 / (1024.0 * 1024.0),
            level: current_level.clone(),
        });

        // Emergency level → ratio 0.25
        apply_eviction(&mut caches, 0.25);

        let mem_after_emerg = total_memory_bytes(&caches);
        let pos_after_emerg = caches[0].current_pos;
        data_points.push(DataPoint {
            step: data_points.len(),
            event: "post_eviction_emergency".to_string(),
            tokens: pos_after_emerg,
            memory_kb: mem_after_emerg as f64 / 1024.0,
            memory_mb: mem_after_emerg as f64 / (1024.0 * 1024.0),
            level: current_level.clone(),
        });

        assert!(
            mem_after_emerg < mem_before_emerg,
            "Memory should decrease after emergency eviction"
        );

        // ── Phase 6: Recovery — Normal level produces no commands. Cache unchanged. ──
        // (MemoryStrategy is removed; Normal signal no longer triggers commands.
        //  We just verify that not calling apply_eviction leaves the cache intact.)
        current_level = "normal".to_string();
        let pos_before_recovery = caches[0].current_pos;

        // Continue filling to show recovery
        for i in 0..60 {
            fill_token(&mut caches, &backend);
            if (i + 1) % 10 == 0 {
                let mem = total_memory_bytes(&caches);
                data_points.push(DataPoint {
                    step: data_points.len(),
                    event: "fill".to_string(),
                    tokens: caches[0].current_pos,
                    memory_kb: mem as f64 / 1024.0,
                    memory_mb: mem as f64 / (1024.0 * 1024.0),
                    level: current_level.clone(),
                });
            }
        }

        // Verify recovery: cache grew past recovery start point
        assert!(
            caches[0].current_pos > pos_before_recovery,
            "Cache should grow during recovery phase"
        );

        // ── Verify data integrity after evictions ──
        // After all evictions, the remaining tokens should still have valid data
        let k_data = caches[0].k_buffer.as_slice::<f32>();
        // pos 0 should have a positive float value (not garbage/zero from uninitialized memory)
        assert!(
            k_data[0] > 0.0,
            "Data at pos 0 should be valid after eviction"
        );

        // ── Verify all layers consistent ──
        let pos = caches[0].current_pos;
        for (i, cache) in caches.iter().enumerate() {
            assert_eq!(
                cache.current_pos, pos,
                "Layer {} has inconsistent pos: {} vs {}",
                i, cache.current_pos, pos
            );
        }

        // ── Output JSON ──
        let result = ScenarioResult {
            name: "memory_pressure_eviction".to_string(),
            description: "KV cache memory under progressive memory pressure signals".to_string(),
            config: ScenarioConfig {
                num_layers: NUM_LAYERS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
                max_seq_len: MAX_SEQ_LEN,
                per_token_bytes,
            },
            data: data_points,
        };

        let json = serde_json::to_string_pretty(&result).unwrap();
        let out_dir = std::path::Path::new("results/data");
        std::fs::create_dir_all(out_dir).unwrap();
        std::fs::write(out_dir.join("eviction_memory_test.json"), &json).unwrap();

        // ── Print summary ──
        println!("\n=== Eviction Memory Test Summary ===");
        println!(
            "Config: {} layers × {} kv_heads × {} dim = {} bytes/token",
            NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, per_token_bytes
        );
        println!(
            "Phase 1: Filled 200 tokens → {:.2} MB",
            (200 * per_token_bytes) as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Phase 2 (Warning 0.85): {} → {} tokens, {:.2} → {:.2} MB",
            200,
            pos_after_warning,
            mem_before as f64 / (1024.0 * 1024.0),
            mem_after as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Phase 4 (Critical 0.50): {} → {} tokens, {:.2} → {:.2} MB",
            pos_before_crit,
            pos_after_crit,
            mem_before_crit as f64 / (1024.0 * 1024.0),
            mem_after_crit as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Phase 5 (Emergency 0.25): {} → {} tokens, {:.2} → {:.2} MB",
            pos_before_emerg,
            pos_after_emerg,
            mem_before_emerg as f64 / (1024.0 * 1024.0),
            mem_after_emerg as f64 / (1024.0 * 1024.0)
        );
        println!("Phase 6: Recovery to Normal, continued filling");
        println!("JSON output: results/data/eviction_memory_test.json");
    }

    /// Verify exact memory reduction ratios match strategy expectations
    #[test]
    fn test_eviction_ratios_are_accurate() {
        let backend = Arc::new(CpuBackend::new());
        let mut caches = make_caches(NUM_LAYERS);

        // Fill exactly 100 tokens
        for _ in 0..100 {
            fill_token(&mut caches, &backend);
        }
        assert_eq!(caches[0].current_pos, 100);

        let mem_100 = total_memory_bytes(&caches);

        // Evict with ratio 0.50 → should keep ~50 tokens
        apply_eviction(&mut caches, 0.50);
        let pos_after = caches[0].current_pos;
        let mem_after = total_memory_bytes(&caches);

        assert_eq!(
            pos_after, 50,
            "50% eviction from 100 should leave 50 tokens"
        );
        assert_eq!(mem_after, mem_100 / 2, "Memory should be exactly halved");

        // Evict with ratio 0.50 again → should keep ~25 tokens
        apply_eviction(&mut caches, 0.50);
        let pos_after2 = caches[0].current_pos;
        let mem_after2 = total_memory_bytes(&caches);

        assert_eq!(
            pos_after2, 25,
            "50% eviction from 50 should leave 25 tokens"
        );
        assert_eq!(
            mem_after2,
            mem_100 / 4,
            "Memory should be exactly quartered from original"
        );
    }

    /// Verify all layers are evicted consistently
    #[test]
    fn test_all_layers_evicted_consistently() {
        let backend = Arc::new(CpuBackend::new());
        let mut caches = make_caches(NUM_LAYERS);

        for _ in 0..200 {
            fill_token(&mut caches, &backend);
        }

        apply_eviction(&mut caches, 0.50);

        let expected_pos = 100;
        for (i, cache) in caches.iter().enumerate() {
            assert_eq!(
                cache.current_pos, expected_pos,
                "Layer {} should have pos={}, got {}",
                i, expected_pos, cache.current_pos
            );

            // Verify data: pos 0 is in the protected prefix (min 4), so still has original value 1.0
            // SlidingWindowPolicy protects first 4 tokens, evicts from middle, keeps tail
            let k_data = cache.k_buffer.as_slice::<f32>();
            assert!(
                k_data[0] > 0.0,
                "Layer {} pos 0 should have valid data, got {}",
                i,
                k_data[0]
            );
        }
    }
}
