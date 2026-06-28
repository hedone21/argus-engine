//! `PrefillKeepSetStage` — R-P1-1 PFA consumer harness (`PrefillEnd` phase).
//!
//! prefill 종료 시 1회 발화한다. `ModelForward` 가 채운 per-layer PFA(`[n_heads_q * prefix_len]`,
//! SUM-pooled trailing-q_window attention 확률)를 공유 cell 에서 읽어, 등록된 plugin
//! (`KVCacheStage`, `caps.reads ∋ PrefillAttention`)이 layer 별 keep-set plan 을 산출하고, 엔진이
//! [`execute_kv_plan`] 으로 적용한다.
//!
//! [`EvictionStage`](super::eviction::EvictionStage) 의 take_inner/put_inner UER 를 미러하되,
//! CacheManager force_evict 대신 **plugin plan + `execute_kv_plan`** 경로를 쓴다(`KVStageCtx` 에 PFA
//! 슬라이스 주입 → `plan(&ctx)` → `execute_kv_plan`). pos 환류는 driver(decode_loop) 책임
//! (`reconcile_kv_pos_after_eviction`).
//!
//! **PR1 = LayerWide keep only.** `KeepSpec::PerHead`(HeadMajor 경로)는 R-P1-2. 미무장(cell `None`)
//! 이면 즉시 `Consumed`(no-op).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use argus_extension_api::KVMutationStage;

use crate::kv::cache_handle::EngineCacheHandle;
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};
use crate::stages::kv::mutation::SnapshotStageCtx;

/// `PrefillEnd` phase 에서 PFA 기반 keep-set 을 1회 적용하는 OneShot Stage.
pub struct PrefillKeepSetStage {
    /// register 시점 보유 핸들 — enumerate 순서 == layer idx(EvictionStage 와 동일 W1 불변식).
    handles: Vec<Arc<StandardFormat>>,
    /// keep-set 생산 plugin (`caps.reads ∋ PrefillAttention`) — v3 imperative [`KVMutationStage`].
    stage: Box<dyn KVMutationStage>,
    /// §5.1 producer(`ModelForward`)와 공유하는 PFA cell. `PrefillEnd` 에서 read.
    prefill_attn_cell: Arc<Mutex<Option<Vec<Vec<f32>>>>>,
    /// attention head 수(pre-GQA) — PFA handle shape `rows`.
    n_heads_q: usize,
    /// keep budget ratio (`target_len = prefix_len * ratio`).
    target_ratio: f32,
}

impl PrefillKeepSetStage {
    /// 무장된 PFA producer cell + keep-set plugin 으로 stage 를 만든다(assembly 가 PFA-reading stage
    /// 발견 시 submit). `handles` enumerate 순서는 layer idx 와 일치해야 한다.
    pub fn new(
        handles: Vec<Arc<StandardFormat>>,
        stage: Box<dyn KVMutationStage>,
        prefill_attn_cell: Arc<Mutex<Option<Vec<Vec<f32>>>>>,
        n_heads_q: usize,
        target_ratio: f32,
    ) -> Self {
        Self {
            handles,
            stage,
            prefill_attn_cell,
            n_heads_q,
            target_ratio,
        }
    }
}

impl PipelineStage for PrefillKeepSetStage {
    fn name(&self) -> &str {
        "kv.prefill_keepset"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::OneShot
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter: PrefillEnd 외 phase 는 무시.
        if *phase != LifecyclePhase::PrefillEnd {
            return Ok(StageOutcome::Continue);
        }
        // PFA cell read. 미무장/미산출(None)이면 no-op Consumed.
        let pfa_guard = self
            .prefill_attn_cell
            .lock()
            .expect("PrefillKeepSetStage PFA cell Mutex poisoned");
        let pfa = match pfa_guard.as_ref() {
            Some(v) => v,
            None => return Ok(StageOutcome::Consumed),
        };

        // UER (Unwrap-Evict-Rewrap, EvictionStage 미러): take_inner → per-layer (stage.on_phase via the
        // v3 handle) → put_inner. `?` 전파는 rewrap 이후로 미룬다(placeholder 폐기 보장).
        let mut temp: Vec<KVCache> = self.handles.iter().map(|f| f.take_inner()).collect();
        let n_layers = temp.len();
        let result = (|| -> anyhow::Result<()> {
            for (layer_idx, cache) in temp.iter_mut().enumerate() {
                let prefix_len = cache.current_pos();
                if prefix_len == 0 || layer_idx >= pfa.len() {
                    continue;
                }
                let target_len = ((prefix_len as f32) * self.target_ratio) as usize;
                // Read ctx with the per-layer PFA slice injected. `for_prefill_attn` copies the cache
                // scalars and borrows ONLY `pfa` (outside the cache), so it coexists with the `&mut`
                // handle below — the read view and the write view never alias.
                let sctx = SnapshotStageCtx::for_prefill_attn(
                    cache,
                    target_len,
                    layer_idx,
                    n_layers,
                    &pfa[layer_idx],
                    self.n_heads_q,
                );
                let handle = EngineCacheHandle::new(cache, layer_idx, n_layers);
                // catch_unwind (mirror of `KVMutationDriverStage`, P0-5a): a panic in an untrusted
                // plugin's on_phase must not unwind past the `put_inner` rewrap below (which would leave
                // the StandardFormat handles empty). Convert it to an `anyhow::Err`.
                let stage = self.stage.as_ref();
                let driven = catch_unwind(AssertUnwindSafe(move || -> anyhow::Result<()> {
                    let mut handle = handle;
                    stage.on_phase(&sctx, &mut handle).map_err(|e| {
                        anyhow::anyhow!("prefill keep-set stage '{}' failed: {e}", stage.name())
                    })?;
                    handle.commit()?;
                    Ok(())
                }));
                match driven {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Err(e),
                    Err(_) => {
                        return Err(anyhow::anyhow!(
                            "prefill keep-set stage '{}' panicked during on_phase/commit",
                            self.stage.name()
                        ));
                    }
                }
            }
            Ok(())
        })();
        for (f, c) in self.handles.iter().zip(temp) {
            f.put_inner(c);
        }
        result?;
        drop(pfa_guard);
        Ok(StageOutcome::Consumed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Force-link the out-of-tree `pyramidkv` stage into the TEST binary only (dev-dependency), so
    // `make_stage_with_args("pyramidkv", ...)` resolves it from the linkme slice. Not linked into
    // the production engine — no default-behavior change.
    use pyramidkv as _;

    use argus_extension_api::{CacheHandle, CacheOpError, StageCtx, TensorKind};

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::format::KVCacheFormat;
    use crate::memory::host::shared::SharedBuffer;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    const HEAD_DIM: usize = 2;
    const KV_HEADS: usize = 1;
    const MAX_SEQ: usize = 32;

    /// SeqMajor F32 cache, current_pos=n, pos p 의 모든 원소 = (p+1)(잘못된 keep 은 값 비교로 검출).
    fn make_cache(n: usize) -> KVCache {
        let total = MAX_SEQ * KV_HEADS * HEAD_DIM;
        let kb = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let vb = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        // SAFETY: kb 는 방금 할당된 total*4 바이트 F32 버퍼; n <= MAX_SEQ 범위 내 write.
        unsafe {
            let kp = kb.as_mut_ptr() as *mut f32;
            for p in 0..n {
                for d in 0..(KV_HEADS * HEAD_DIM) {
                    *kp.add(p * KV_HEADS * HEAD_DIM + d) = (p + 1) as f32;
                }
            }
        }
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
        let mut c = KVCache::new(
            Tensor::new(sh.clone(), kb, be.clone()),
            Tensor::new(sh, vb, be),
            MAX_SEQ,
        );
        c.current_pos = n;
        c
    }

    /// 테스트용 keep-set plugin: PFA 를 GQA-reduce(전 head 합)해 per-token importance 산출 → top
    /// `target_len` 위치 LayerWide keep(ascending). R-P1-2 SnapKV 의 LayerWide 축소판.
    struct TopKPfaStage;
    impl TopKPfaStage {
        fn keep_list(&self, ctx: &dyn StageCtx) -> Option<Vec<usize>> {
            let prefix_len = ctx.current_pos();
            let target = ctx.target_len().min(prefix_len);
            let h = ctx.tensor(TensorKind::PrefillAttention)?;
            let shape = h.shape();
            let (n_heads, cols) = (shape.rows, shape.cols);
            let mut imp = vec![0.0f32; cols];
            let mut row = vec![0.0f32; cols];
            for hh in 0..n_heads {
                h.read_row(hh, 0, &mut row);
                for (acc, &x) in imp.iter_mut().zip(row.iter()) {
                    *acc += x;
                }
            }
            // top-`target` positions(중요도 내림차순), 이후 compaction 위해 ascending.
            let mut idx: Vec<usize> = (0..prefix_len).collect();
            idx.sort_by(|&a, &b| imp[b].partial_cmp(&imp[a]).unwrap());
            let mut keep: Vec<usize> = idx.into_iter().take(target).collect();
            keep.sort_unstable();
            Some(keep)
        }
    }
    impl KVMutationStage for TopKPfaStage {
        fn name(&self) -> &str {
            "test.topk_pfa"
        }
        fn on_phase(
            &self,
            ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            match self.keep_list(ctx) {
                Some(keep) => cache.keep(&keep),
                None => Ok(()),
            }
        }
    }

    fn make_ctx(profiler: &mut OpProfiler) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(0),
                prev_token: 0,
            },
            profiler,
        }
    }

    /// PFA cell 에 odd 위치를 선호하는 점수를 채운다(n_heads_q 행 × prefix_len 열, 1 layer).
    fn odd_favoring_pfa(n_heads_q: usize, prefix_len: usize) -> Arc<Mutex<Option<Vec<Vec<f32>>>>> {
        let mut layer = vec![0.0f32; n_heads_q * prefix_len];
        for h in 0..n_heads_q {
            for kp in 0..prefix_len {
                layer[h * prefix_len + kp] = if kp % 2 == 1 { 1.0 } else { 0.01 };
            }
        }
        Arc::new(Mutex::new(Some(vec![layer])))
    }

    #[test]
    fn prefill_end_applies_layerwide_keepset() {
        // LayerWide smoke(§6.3): PrefillEnd → take_inner/plan/execute_kv_plan/put_inner. prefix_len=8,
        // target_ratio=0.5 → keep 4. odd 위치 선호 → keep {1,3,5,7}. 적용 후 current_pos==4 + 값 검증.
        let prefix_len = 8;
        let n_heads_q = 2; // GQA 2:1 (cache kv_heads=1).
        let handle = Arc::new(StandardFormat::new(0, make_cache(prefix_len)));
        let cell = odd_favoring_pfa(n_heads_q, prefix_len);
        let stage = PrefillKeepSetStage::new(
            vec![handle.clone()],
            Box::new(TopKPfaStage),
            cell,
            n_heads_q,
            0.5,
        );

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::PrefillEnd, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        assert_eq!(handle.current_pos(), 4, "keep target = 8 * 0.5");

        // 압축된 cache pos i 의 값 = 보존된 원본 위치(p)의 (p+1). keep {1,3,5,7} → {2,4,6,8}.
        let inner = handle.take_inner();
        let k = inner.k_buffer.as_slice::<f32>();
        for (i, &orig) in [1usize, 3, 5, 7].iter().enumerate() {
            assert_eq!(
                k[i * KV_HEADS * HEAD_DIM],
                (orig + 1) as f32,
                "compacted pos {i} should hold original pos {orig}"
            );
        }
    }

    #[test]
    fn non_prefill_end_phase_is_noop() {
        // self-filter: PrefillEnd 외 phase → Continue + cache 불변.
        let handle = Arc::new(StandardFormat::new(0, make_cache(8)));
        let cell = odd_favoring_pfa(2, 8);
        let stage =
            PrefillKeepSetStage::new(vec![handle.clone()], Box::new(TopKPfaStage), cell, 2, 0.5);
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::DecodeStart, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Continue));
        assert_eq!(handle.current_pos(), 8, "no prune off PrefillEnd");
    }

    #[test]
    fn unarmed_cell_is_noop_consumed() {
        // PFA cell None(미무장/미산출) → Consumed + cache 불변.
        let handle = Arc::new(StandardFormat::new(0, make_cache(8)));
        let cell: Arc<Mutex<Option<Vec<Vec<f32>>>>> = Arc::new(Mutex::new(None));
        let stage =
            PrefillKeepSetStage::new(vec![handle.clone()], Box::new(TopKPfaStage), cell, 2, 0.5);
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::PrefillEnd, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        assert_eq!(handle.current_pos(), 8, "unarmed → no prune");
    }

    // ───────────────── R-P1-2 per-head (HeadMajor) mechanism ─────────────────

    /// HeadMajor F32 cache, current_pos=n. head h pos p 의 모든 원소 = `h*1000 + p + 1`
    /// (per-head compaction 후 어느 (head,pos)가 살았는지 값으로 디코드). HeadMajor offset =
    /// `(h*capacity + p)*head_dim` (kv_cache.rs:186).
    fn make_head_major_cache(n_kv_heads: usize, n: usize) -> KVCache {
        use crate::kv_cache_ops::KVLayout;
        let total = MAX_SEQ * n_kv_heads * HEAD_DIM;
        let kb = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let vb = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        // SAFETY: kb total*4 바이트 F32; HeadMajor offset 은 capacity(=MAX_SEQ) 내.
        unsafe {
            let kp = kb.as_mut_ptr() as *mut f32;
            for h in 0..n_kv_heads {
                for p in 0..n {
                    let base = (h * MAX_SEQ + p) * HEAD_DIM;
                    for d in 0..HEAD_DIM {
                        *kp.add(base + d) = (h * 1000 + p + 1) as f32;
                    }
                }
            }
        }
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, MAX_SEQ, n_kv_heads, HEAD_DIM]);
        let mut c = KVCache::new(
            Tensor::new(sh.clone(), kb, be.clone()),
            Tensor::new(sh, vb, be),
            MAX_SEQ,
        )
        .with_layout(KVLayout::HeadMajor);
        c.current_pos = n;
        c
    }

    /// 테스트용 per-head keep-set plugin: PFA(n_heads_q 행)를 kv-head 로 GQA-reduce(group 합) →
    /// kv-head 별 top-`target` 위치 → `KeepSpec::PerHead`(동일 길이, ascending). R-P1-2 SnapKV per-head
    /// 의 축소판(executor 의 per-head 분기 = HeadMajor 전제를 검증).
    struct PerHeadTopKStage;
    impl PerHeadTopKStage {
        fn per_head_keep(&self, ctx: &dyn StageCtx) -> Option<Vec<Vec<usize>>> {
            let prefix_len = ctx.current_pos();
            let n_kv_heads = ctx.n_kv_heads();
            let target = ctx.target_len().min(prefix_len);
            let h = ctx.tensor(TensorKind::PrefillAttention)?;
            let n_heads_q = h.shape().rows;
            let cols = h.shape().cols; // prefix_len
            let gqa = n_heads_q / n_kv_heads;
            let mut heads: Vec<Vec<usize>> = Vec::with_capacity(n_kv_heads);
            let mut row = vec![0.0f32; cols];
            for kv in 0..n_kv_heads {
                // GQA-reduce: sum the gqa attention-head rows mapping to this kv-head.
                let mut imp = vec![0.0f32; cols];
                for g in 0..gqa {
                    h.read_row(kv * gqa + g, kv, &mut row);
                    for (acc, &x) in imp.iter_mut().zip(row.iter()) {
                        *acc += x;
                    }
                }
                let mut idx: Vec<usize> = (0..prefix_len).collect();
                idx.sort_by(|&a, &b| imp[b].partial_cmp(&imp[a]).unwrap());
                let mut keep: Vec<usize> = idx.into_iter().take(target).collect();
                keep.sort_unstable();
                heads.push(keep);
            }
            Some(heads)
        }
    }
    impl KVMutationStage for PerHeadTopKStage {
        fn name(&self) -> &str {
            "test.perhead_topk_pfa"
        }
        fn on_phase(
            &self,
            ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            match self.per_head_keep(ctx) {
                Some(heads) => {
                    let refs: Vec<&[usize]> = heads.iter().map(|h| h.as_slice()).collect();
                    cache.keep_per_head(&refs)
                }
                None => Ok(()),
            }
        }
    }

    /// kv-head 별 divergent PFA: kv_head0(attn 0,1)→even 선호, kv_head1(attn 2,3)→odd 선호.
    fn divergent_pfa(n_heads_q: usize, n_kv_heads: usize, prefix_len: usize) -> Vec<f32> {
        let gqa = n_heads_q / n_kv_heads;
        let mut layer = vec![0.0f32; n_heads_q * prefix_len];
        for h in 0..n_heads_q {
            let kv = h / gqa;
            for kp in 0..prefix_len {
                // kv 0 → even high, kv 1 → odd high.
                let favored = if kv == 0 { kp % 2 == 0 } else { kp % 2 == 1 };
                layer[h * prefix_len + kp] = if favored { 1.0 } else { 0.01 };
            }
        }
        layer
    }

    #[test]
    fn prefill_end_applies_perhead_keepset_headmajor() {
        // R-P1-2 mechanism: HeadMajor cache + per-head PFA plugin → KeepSpec::PerHead → executor
        // per-head 분기(stage_registry.rs:113) → compact_keep_positions_for_head. kv-head 별로 다른
        // 위치가 살아남음(divergence) + HeadMajor 압축 byte-exact. (이 경로는 이전엔 dead — 첫 e2e 검증.)
        let n_kv_heads = 2;
        let n_heads_q = 4; // GQA 2:1.
        let prefix_len = 8;
        let target = 4; // ratio 0.5.
        let handle = Arc::new(StandardFormat::new(
            0,
            make_head_major_cache(n_kv_heads, prefix_len),
        ));
        let cell = Arc::new(Mutex::new(Some(vec![divergent_pfa(
            n_heads_q, n_kv_heads, prefix_len,
        )])));
        let stage = PrefillKeepSetStage::new(
            vec![handle.clone()],
            Box::new(PerHeadTopKStage),
            cell,
            n_heads_q,
            0.5,
        );

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::PrefillEnd, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        assert_eq!(handle.current_pos(), target, "per-head keep len");

        // 압축 후 head h pos i 값 = (h*1000 + kept_h[i] + 1). kv0 keeps even {0,2,4,6}, kv1 odd {1,3,5,7}.
        let inner = handle.take_inner();
        let k = inner.k_buffer.as_slice::<f32>();
        let expected: [[usize; 4]; 2] = [[0, 2, 4, 6], [1, 3, 5, 7]];
        for (kv, kept) in expected.iter().enumerate() {
            for (i, &orig) in kept.iter().enumerate() {
                let off = (kv * MAX_SEQ + i) * HEAD_DIM; // HeadMajor offset.
                assert_eq!(
                    k[off],
                    (kv * 1000 + orig + 1) as f32,
                    "kv-head {kv} compacted pos {i} should hold original pos {orig}"
                );
            }
        }
    }

    /// Byte-identical engine verification of the REAL `pyramidkv` stage driven through the full
    /// PrefillEnd path: PFA → `pyramidkv::plan()` → `execute_kv_plan` (PerHead) → HeadMajor
    /// per-head compaction. The expected per-layer/per-kv-head keep-sets + pyramid budgets are the
    /// kvpress oracle from `crates/techniques/pyramidkv/reference/gen_engine_fixture.py`
    /// (4 layers, GQA 2:1, k_len=32, window=4, kernel=1, beta=2, compression_ratio=0.5). This is
    /// the engine-execution half of the byte-by-byte chain: pyramidkv's decision == kvpress (crate
    /// unit suite) AND the engine compacts the buffers to exactly that decision (here).
    #[test]
    fn prefill_end_real_pyramidkv_byte_identical_vs_kvpress() {
        use crate::kv_cache_ops::KVLayout;
        const N_LAYERS: usize = 4;
        const N_KV: usize = 2;
        const N_Q: usize = 4; // GQA 2:1
        const K_LEN: usize = 32; // == MAX_SEQ

        // Same LCG as reference/pyramidkv_select_ref.py (head outer, pos inner, continuous state).
        fn synth_attn(n_q: usize, k_len: usize, seed: i64) -> Vec<f32> {
            let mut data = Vec::with_capacity(n_q * k_len);
            let mut s = seed;
            for _ in 0..n_q {
                for _ in 0..k_len {
                    s = (1_103_515_245_i64 * s + 12_345) & 0x7FFF_FFFF;
                    data.push((s % 1000) as f32);
                }
            }
            data
        }

        // kvpress-reference oracle (reference/gen_engine_fixture.py). Pyramid budgets per layer:
        let budgets: [usize; N_LAYERS] = [24, 19, 13, 8];
        #[rustfmt::skip]
        let expected: [[&[usize]; N_KV]; N_LAYERS] = [
            [&[0,1,3,5,7,8,9,11,12,13,15,16,17,18,19,20,22,25,26,27,28,29,30,31],
             &[0,1,2,4,5,7,8,10,11,14,15,16,18,19,21,23,24,25,26,27,28,29,30,31]],
            [&[1,3,4,5,7,8,9,10,11,18,19,21,22,24,25,28,29,30,31],
             &[2,5,6,7,9,10,11,12,14,16,19,21,23,24,26,28,29,30,31]],
            [&[0,5,7,8,11,17,19,21,25,28,29,30,31],
             &[0,4,10,12,14,16,19,20,27,28,29,30,31]],
            [&[3,5,16,20,28,29,30,31],
             &[0,1,9,20,28,29,30,31]],
        ];

        let handles: Vec<Arc<StandardFormat>> = (0..N_LAYERS)
            .map(|l| Arc::new(StandardFormat::new(l, make_head_major_cache(N_KV, K_LEN))))
            .collect();
        let pfa: Vec<Vec<f32>> = (0..N_LAYERS)
            .map(|l| synth_attn(N_Q, K_LEN, 1000 + l as i64))
            .collect();
        let cell = Arc::new(Mutex::new(Some(pfa)));

        // pyramidkv with explicit knobs via the StageArgs blob (compression_ratio drives the budget,
        // so PrefillKeepSetStage's target_ratio is irrelevant here — set to 1.0).
        let blob = [
            argus_extension_api::PluginArg {
                key: "compression_ratio",
                val: "0.5",
            },
            argus_extension_api::PluginArg {
                key: "window_size",
                val: "4",
            },
            argus_extension_api::PluginArg {
                key: "kernel_size",
                val: "1",
            },
            argus_extension_api::PluginArg {
                key: "beta",
                val: "2",
            },
        ];
        let reg = argus_extension_api::find_mutation_stage("pyramidkv")
            .expect("pyramidkv v3 stage registered (engine dev-dep force-link)");
        let plugin = (reg.make)(argus_extension_api::StageParams::default(), &blob);

        let stage = PrefillKeepSetStage::new(handles.clone(), plugin, cell, N_Q, 1.0);
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::PrefillEnd, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));

        for (layer, handle) in handles.iter().enumerate() {
            assert_eq!(handle.current_pos(), budgets[layer], "layer {layer} budget");
            let inner = handle.take_inner();
            assert_eq!(inner.layout(), KVLayout::HeadMajor);
            let k = inner.k_buffer.as_slice::<f32>();
            for (kv, kept) in expected[layer].iter().enumerate() {
                assert_eq!(
                    kept.len(),
                    budgets[layer],
                    "layer {layer} head {kv} keep len"
                );
                for (i, &orig) in kept.iter().enumerate() {
                    let off = (kv * MAX_SEQ + i) * HEAD_DIM; // HeadMajor offset.
                    assert_eq!(
                        k[off],
                        (kv * 1000 + orig + 1) as f32,
                        "layer {layer} kv-head {kv} compacted pos {i} != kvpress original pos {orig}"
                    );
                }
            }
        }
    }
}
