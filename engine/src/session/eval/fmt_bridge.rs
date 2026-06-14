//! Phase α-K BC ①-c: eval transient fmt-wrap bridge.
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §9.1-BC1'(line 794) + cut
//! `.agent/todos/design_alpha_k_1c_cut_2026_06_04.md`(workflow `wdrcgtqwz`, Strategy A).
//!
//! eval 은 concrete `Vec<KVCache>`(EvictionHook) / `Vec<KiviCache>`(KiviHook) 를 **계속 소유**하되,
//! forward 1회 동안만 `Arc<dyn KVCacheFormat>` 로 wrap 하여 `forward_into` 에 위임한다(round-trip).
//! hook/snapshot/eviction 은 forward 와 interleave 하지 않으므로(forward → post_prefill/snapshot/
//! restore 시퀀셜) round-trip 후 복귀한 concrete slice 를 그대로 받는다 → hook impl 무수정.
//!
//! `KVCacheOps` 바운드 제거: `run_eval_ll_generic<C: KVCacheOps>` → `<C: EvalCacheKind>`. EvalCacheKind 는
//! `KVCacheOps` 를 상속하지 않는다(★반증3 회피). 단 `EvalCacheKind for KiviCache` 의 `cur_pos`/
//! `needs_scores` 위임은 `KiviCache::current_pos`(=`total_tokens()` private fn)·`needs_attn_scores`
//! (`awqe_enabled` private 필드)를 호출해야 해 `KVCacheOps` 경유가 불가피하다 — **①-c 수용 잔여**
//! (Step 5 에서 KVCacheOps→inherent 이전 시 정리). KVCache 측은 `current_pos`/`high_water_pos` 가
//! pub 필드라 trait 불요.

use std::sync::Arc;

use anyhow::Result;

use crate::format::KVCacheFormat;
use crate::kv::kivi_cache::KiviCache;
use crate::kv::kivi_format::KIVIFormat;
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;

/// eval 의 cache 다형성을 `KVCacheOps` 바운드 없이 추상화 (Phase α-K ①-c).
///
/// `forward_fmt_roundtrip` 가 fmt-wrap → `forward_into` → unwrap 을 캡슐화하고, 나머지 3 메서드는
/// eval_loop 의 직접 cache 접근(`current_pos`/`set_current_pos`/`needs_attn_scores`)을 노출한다.
pub trait EvalCacheKind: Sized + Send {
    /// `caches` 전체를 forward 1회 동안 `Arc<dyn KVCacheFormat>` 로 wrap 하여 `run` 에 넘기고, 종료 후
    /// concrete 로 복귀시킨다. **panic-safe**: `run` 이 `Err`/`?` early-return 해도 복귀 후 그 결과를 반환
    /// (run 내부 panic 은 caches 손실 = process abort 라 실해 없음).
    fn forward_fmt_roundtrip(
        caches: &mut Vec<Self>,
        run: impl FnOnce(&[Arc<dyn KVCacheFormat>]) -> Result<()>,
    ) -> Result<()>;

    /// 현재 유효 토큰 수 (구 `KVCacheOps::current_pos`).
    fn cur_pos(&self) -> usize;

    /// 위치 카운터 덮어쓰기 (구 `KVCacheOps::set_current_pos`).
    fn set_cur_pos(&mut self, pos: usize);

    /// 이 cache 가 decode 마다 attention score 를 요구하는가 (구 `KVCacheOps::needs_attn_scores`).
    fn needs_scores(&self) -> bool;
}

impl EvalCacheKind for KVCache {
    fn forward_fmt_roundtrip(
        caches: &mut Vec<Self>,
        run: impl FnOnce(&[Arc<dyn KVCacheFormat>]) -> Result<()>,
    ) -> Result<()> {
        let taken = std::mem::take(caches);
        let sfs: Vec<Arc<StandardFormat>> = taken
            .into_iter()
            .enumerate()
            .map(|(i, c)| Arc::new(StandardFormat::new(i, c)))
            .collect();
        let dyn_slice: Vec<Arc<dyn KVCacheFormat>> = sfs
            .iter()
            .map(|a| a.clone() as Arc<dyn KVCacheFormat>)
            .collect();
        let r = run(&dyn_slice);
        drop(dyn_slice); // transient dyn refcount → 1
        *caches = sfs
            .into_iter()
            .map(|a| {
                Arc::try_unwrap(a)
                    .ok()
                    .expect("transient StandardFormat has external Arc clone")
                    .into_inner()
            })
            .collect();
        r
    }

    fn cur_pos(&self) -> usize {
        // KVCache.current_pos 는 pub 필드 — KVCacheOps trait 불요.
        self.current_pos
    }

    fn set_cur_pos(&mut self, pos: usize) {
        // KVCache::set_current_pos(kv_cache.rs:1005) 미러 — pos==0 일 때만 high_water 도 reset.
        self.current_pos = pos;
        if pos == 0 {
            self.high_water_pos = 0;
        }
    }

    fn needs_scores(&self) -> bool {
        false
    }
}

impl EvalCacheKind for KiviCache {
    fn forward_fmt_roundtrip(
        caches: &mut Vec<Self>,
        run: impl FnOnce(&[Arc<dyn KVCacheFormat>]) -> Result<()>,
    ) -> Result<()> {
        let taken = std::mem::take(caches);
        let kfs: Vec<Arc<KIVIFormat>> = taken
            .into_iter()
            .enumerate()
            .map(|(i, c)| Arc::new(KIVIFormat::new(i, c)))
            .collect();
        let dyn_slice: Vec<Arc<dyn KVCacheFormat>> = kfs
            .iter()
            .map(|a| a.clone() as Arc<dyn KVCacheFormat>)
            .collect();
        let r = run(&dyn_slice);
        drop(dyn_slice);
        *caches = kfs
            .into_iter()
            .map(|a| {
                Arc::try_unwrap(a)
                    .ok()
                    .expect("transient KIVIFormat has external Arc clone")
                    .into_inner()
            })
            .collect();
        r
    }

    fn cur_pos(&self) -> usize {
        // Phase α-K BC 5-E: KiviCache inherent `current_pos`(=total_tokens()) 직접 호출 (①-c 수용 잔여 해소).
        self.current_pos()
    }

    fn set_cur_pos(&mut self, _pos: usize) {
        // KiviCache position 은 q2_tokens + res_pos 에서 파생 — no-op (kivi_cache.rs:2242).
    }

    fn needs_scores(&self) -> bool {
        // Phase α-K BC 5-E: KiviCache inherent `is_awqe_enabled`(=awqe_enabled) 직접 호출 (①-c 수용 잔여 해소).
        self.is_awqe_enabled()
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

    fn f32_tensor(dims: Vec<usize>, data: &[f32]) -> Tensor {
        let buf = Arc::new(SharedBuffer::new(data.len() * 4, DType::F32));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    fn make_cache(max_seq: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        let total = max_seq * kv_heads * head_dim;
        let k = f32_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0; total]);
        let v = f32_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0; total]);
        KVCache::new(k, v, max_seq)
    }

    #[test]
    fn test_roundtrip_preserves_current_pos_and_len() {
        // 2-layer KVCache 에 current_pos 를 세팅 → round-trip 후 동일 보존 + fmt 가 올바른 pos 노출.
        let mut caches: Vec<KVCache> = vec![make_cache(16, 2, 4), make_cache(16, 2, 4)];
        caches[0].current_pos = 5;
        caches[1].current_pos = 5;

        let mut seen_len = 0usize;
        let mut seen_pos = 0usize;
        KVCache::forward_fmt_roundtrip(&mut caches, |fmts| {
            seen_len = fmts.len();
            seen_pos = fmts[0].current_pos();
            Ok(())
        })
        .unwrap();

        // 클로저는 wrap 된 2개 fmt 와 current_pos=5 를 본다.
        assert_eq!(seen_len, 2);
        assert_eq!(seen_pos, 5);
        // 복귀: concrete Vec 길이 + current_pos 보존.
        assert_eq!(caches.len(), 2);
        assert_eq!(caches[0].current_pos, 5);
        assert_eq!(caches[1].current_pos, 5);
    }

    #[test]
    fn test_roundtrip_restores_on_err() {
        // run 이 Err 를 반환해도 caches 가 복귀되고 그 Err 가 전파된다.
        let mut caches: Vec<KVCache> = vec![make_cache(16, 2, 4)];
        caches[0].current_pos = 3;

        let r =
            KVCache::forward_fmt_roundtrip(&mut caches, |_fmts| anyhow::bail!("forward failed"));

        assert!(r.is_err());
        // Err 경로에서도 복귀.
        assert_eq!(caches.len(), 1);
        assert_eq!(caches[0].current_pos, 3);
    }

    #[test]
    fn test_eval_cache_kind_kvcache_accessors() {
        let mut c = make_cache(16, 2, 4);
        c.current_pos = 7;
        c.high_water_pos = 7;
        assert_eq!(c.cur_pos(), 7);
        assert!(!c.needs_scores());
        // set_cur_pos(nonzero): current_pos 만 변경, high_water 보존.
        c.set_cur_pos(4);
        assert_eq!(c.current_pos, 4);
        assert_eq!(c.high_water_pos, 7);
        // set_cur_pos(0): high_water 도 reset (KVCache::set_current_pos 미러).
        c.set_cur_pos(0);
        assert_eq!(c.current_pos, 0);
        assert_eq!(c.high_water_pos, 0);
    }
}
