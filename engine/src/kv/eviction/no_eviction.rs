use super::EvictionPolicy;
use crate::kv::kv_cache::KVCache;
use anyhow::Result;

/// Default policy that never evicts. Maintains backward compatibility
/// with existing behavior — cache simply fills up and overflows produce an error.
pub struct NoEvictionPolicy;

impl NoEvictionPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoEvictionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for NoEvictionPolicy {
    fn should_evict(&self, _cache: &KVCache, _mem_available: usize) -> bool {
        false
    }

    fn evict(&self, _cache: &mut KVCache, _target_len: usize) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "none"
    }

    /// (3c-evict) keep-list. `evict()` 는 no-op 이므로 전체 보존 `[0..current)` — `compact` 도
    /// src==dst 단일 batch no-op 으로 버퍼/`current_pos` 무변(등가).
    fn plan_keep(
        &self,
        current_pos: usize,
        _target_len: usize,
        _importance: Option<&[f32]>,
    ) -> Option<(Vec<usize>, Vec<crate::format::Merge>)> {
        Some(((0..current_pos).collect(), Vec::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache(pos: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let buf_size = 100 * 4 * 4;
        let k = Tensor::new(
            Shape::new(vec![1, 100, 1, 4]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, 100, 1, 4]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let mut cache = KVCache::new(k, v, 100);
        cache.current_pos = pos;
        cache
    }

    #[test]
    fn test_no_eviction_never_evicts() {
        let policy = NoEvictionPolicy::new();
        let cache = make_cache(99);
        assert!(!policy.should_evict(&cache, 0)); // Even with 0 memory available
        assert!(!policy.should_evict(&cache, usize::MAX));
    }

    #[test]
    fn test_no_eviction_evict_is_noop() {
        let policy = NoEvictionPolicy::new();
        let mut cache = make_cache(50);
        policy.evict(&mut cache, 30).unwrap();
        assert_eq!(cache.current_pos, 50); // Unchanged
    }

    #[test]
    fn test_no_eviction_name() {
        assert_eq!(NoEvictionPolicy::new().name(), "none");
    }
}
