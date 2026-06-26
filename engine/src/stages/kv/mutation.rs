//! `KVMutationDriverStage` — the engine driver for imperative [`KVMutationStage`] callbacks.
//!
//! The mutation twin of [`FormatReencodeStage`](super::format_reencode::FormatReencodeStage): it
//! mirrors that stage's take_inner / per-layer / put_inner UER, but drives the imperative
//! [`CacheHandle`](argus_extension_api::CacheHandle) surface — a [`KVMutationStage`] callback stages
//! mutation ops on a per-layer [`EngineCacheHandle`], and the engine commits the transaction once per
//! layer ([`EngineCacheHandle::commit`]).
//!
//! Because every handle op routes to the SAME executor `execute_kv_plan` uses
//! (`compact_keep_positions` / `apply_weighted_merges` / `apply_format_plan`), a keep applied through
//! this driver is **byte-identical** to the same keep applied through the plan executor (the s1 gate).
//!
//! Read/mutate aliasing: a mutation stage reads its cache state through an owned-scalar
//! [`ScalarStageCtx`] (current_pos / target_len / geometry, copied before the handle is built) and
//! mutates through the `&mut` handle. The ctx owns copies rather than borrowing the cache, so the
//! `&dyn StageCtx` read view and the `&mut dyn CacheHandle` write view never alias — and both observe
//! the entry frame (the cache is untouched until `commit`).
//!
//! The driver is NOT registered into the pipeline in s1 (the `KV_MUTATION_STAGES` slice / production
//! wiring is a follow-up); it is constructed directly and exercised by the byte-identical gate.

use std::sync::Arc;

use argus_extension_api::{
    CacheHandle, CacheOpError, KVCacheStage, KVMutationStage, KeepSpec, MutationPhase, StageCtx,
    TensorHandle, TensorKind,
};

use crate::kv::cache_handle::EngineCacheHandle;
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};

/// Maps the additive [`MutationPhase`] to the engine [`LifecyclePhase`] the driver fires at.
fn lifecycle_of(phase: MutationPhase) -> LifecyclePhase {
    match phase {
        MutationPhase::PrefillEnd => LifecyclePhase::PrefillEnd,
        MutationPhase::KvMutate => LifecyclePhase::KvMutate,
    }
}

/// An owned-scalar [`StageCtx`] that does NOT borrow the cache — so it can coexist with a `&mut`
/// [`CacheHandle`] over the same cache. Carries the entry-frame scalars a score-free stage needs
/// (current_pos / target_len / geometry); external signals (importance / scores) are `None`
/// (production signal plumbing is a follow-up — score-based stages route through the s2 keep_top_k
/// compiler, not this driver).
pub struct ScalarStageCtx {
    current_pos: usize,
    target_len: usize,
    layer_idx: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    on_device: bool,
}

impl ScalarStageCtx {
    /// Snapshot the entry-frame scalars of `cache` (copies, no borrow held).
    pub fn from_cache(cache: &KVCache, target_len: usize, layer_idx: usize, n_layers: usize) -> Self {
        Self {
            current_pos: cache.current_pos(),
            target_len,
            layer_idx,
            n_layers,
            n_kv_heads: cache.kv_heads(),
            head_dim: cache.head_dim(),
            on_device: cache.k_buffer.buffer().is_gpu_buffer(),
        }
    }
}

impl StageCtx for ScalarStageCtx {
    fn current_pos(&self) -> usize {
        self.current_pos
    }
    fn target_len(&self) -> usize {
        self.target_len
    }
    fn layer_idx(&self) -> usize {
        self.layer_idx
    }
    fn n_layers(&self) -> usize {
        self.n_layers
    }
    fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn kv_on_device(&self) -> bool {
        self.on_device
    }
    fn importance(&self) -> Option<&[f32]> {
        None
    }
    fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
        None
    }
}

/// Adapts a plan-returning [`KVCacheStage`] to the imperative [`KVMutationStage`] by applying its
/// plan through the transactional [`CacheHandle`]. This is the bridge that makes the byte-identical
/// gate faithful: the same plugin's plan drives both `execute_kv_plan` and the handle.
pub struct PlanStageAdapter {
    inner: Box<dyn KVCacheStage>,
    phase: MutationPhase,
}

impl PlanStageAdapter {
    /// Wrap a plan-returning stage, firing at `phase`.
    pub fn new(inner: Box<dyn KVCacheStage>, phase: MutationPhase) -> Self {
        Self { inner, phase }
    }
}

impl KVMutationStage for PlanStageAdapter {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn phase(&self) -> MutationPhase {
        self.phase
    }
    fn on_phase(&self, ctx: &dyn StageCtx, cache: &mut dyn CacheHandle) -> Result<(), CacheOpError> {
        let Some(plan) = self.inner.plan(ctx) else {
            return Ok(()); // no-op plan
        };
        if plan.channels.is_some() {
            // Channel-axis selection is a dormant geometry wall (common to the plan executor, which
            // rejects `KVCachePlan.channels` for the same reason).
            return Err(CacheOpError::GeometryImmutable);
        }
        if !plan.merges.is_empty() {
            cache.merge(&plan.merges)?;
        }
        match &plan.keep {
            KeepSpec::LayerWide(k) => cache.keep(k)?,
            KeepSpec::PerHead(heads) => {
                let refs: Vec<&[usize]> = heads.iter().map(|h| h.as_slice()).collect();
                cache.keep_per_head(&refs)?;
            }
        }
        Ok(())
    }
}

/// A pipeline stage that runs a [`KVMutationStage`] callback over a set of per-layer caches, mirroring
/// [`FormatReencodeStage`](super::format_reencode::FormatReencodeStage). Constructed directly in s1
/// (not registered into the pipeline; production wiring is a follow-up).
pub struct KVMutationDriverStage {
    /// register-time handles — enumerate order == layer idx (same W1 invariant as `FormatReencodeStage`).
    handles: Vec<Arc<StandardFormat>>,
    /// The imperative mutation callback this driver runs.
    stage: Box<dyn KVMutationStage>,
}

impl KVMutationDriverStage {
    /// `handles` enumerate order must equal layer idx. `stage` is the mutation callback.
    pub fn new(handles: Vec<Arc<StandardFormat>>, stage: Box<dyn KVMutationStage>) -> Self {
        Self { handles, stage }
    }
}

impl PipelineStage for KVMutationDriverStage {
    fn name(&self) -> &str {
        "kv.mutation_driver"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::Persistent
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter: only the stage's declared phase.
        if *phase != lifecycle_of(self.stage.phase()) {
            return Ok(StageOutcome::Continue);
        }

        // UER (mirroring FormatReencodeStage): take_inner -> per-layer drive+commit -> put_inner.
        let mut temp: Vec<KVCache> = self.handles.iter().map(|f| f.take_inner()).collect();
        let n_layers = temp.len();
        let result = (|| -> anyhow::Result<()> {
            for (layer_idx, cache) in temp.iter_mut().enumerate() {
                if cache.current_pos() == 0 {
                    continue;
                }
                // Owned-scalar read ctx (no cache borrow) — coexists with the &mut handle below.
                // target_len = 0 in s1 (the eviction budget plumbing is production/follow-up).
                let sctx = ScalarStageCtx::from_cache(cache, 0, layer_idx, n_layers);
                let mut handle = EngineCacheHandle::new(cache, layer_idx, n_layers);
                self.stage
                    .on_phase(&sctx, &mut handle)
                    .map_err(|e| anyhow::anyhow!("mutation stage '{}' failed: {e}", self.stage.name()))?;
                handle.commit()?;
            }
            Ok(())
        })();
        for (f, c) in self.handles.iter().zip(temp) {
            f.put_inner(c);
        }
        result?;
        Ok(StageOutcome::Consumed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use argus_extension_api::{KVCachePlan, StageParams, find_stage};

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::eviction::stage_registry::execute_kv_plan;
    use crate::memory::host::shared::SharedBuffer;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::quant::{BlockQ4_0, QK4_0};
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use half::f16;

    const HD: usize = 64; // q4_0-valid: a multiple of QK4_0 (=32).
    const KV_HEADS: usize = 2;
    const MAX_SEQ: usize = 32;
    const RESIDENT: usize = 20;
    const SINK: usize = 4;
    const WINDOW: usize = 6;

    /// Build a SeqMajor cache of `dtype` with a known per-(pos,head,d) pattern in the resident region
    /// of both K and V (V uses a distinct salt). The pattern is written as f32 then encoded to `dtype`,
    /// so the two paths under test start from byte-identical buffers.
    fn make_cache(dtype: DType) -> KVCache {
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HD]);
        let total = MAX_SEQ * KV_HEADS * HD;
        let bytes = match dtype {
            DType::F32 => total * 4,
            DType::F16 => total * 2,
            DType::Q4_0 => (total / QK4_0) * std::mem::size_of::<BlockQ4_0>(),
            _ => unreachable!(),
        };
        let mut c = KVCache::new(
            Tensor::new(sh.clone(), Arc::new(SharedBuffer::new(bytes, dtype)), be.clone()),
            Tensor::new(sh, Arc::new(SharedBuffer::new(bytes, dtype)), be),
            MAX_SEQ,
        );
        c.set_current_pos(RESIDENT);
        let pat = |pos: usize, head: usize, d: usize, salt: f32| {
            salt + pos as f32 * 0.11 + head as f32 * 0.27 + (d as f32 - HD as f32 / 2.0) * 0.04
        };
        for pos in 0..RESIDENT {
            for head in 0..KV_HEADS {
                let off = c.offset(pos, head);
                let mut row_k = [0.0f32; HD];
                let mut row_v = [0.0f32; HD];
                for d in 0..HD {
                    row_k[d] = pat(pos, head, d, 0.5);
                    row_v[d] = pat(pos, head, d, -1.3);
                }
                write_row(&mut c, off, dtype, &row_k, true);
                write_row(&mut c, off, dtype, &row_v, false);
            }
        }
        c
    }

    fn write_row(c: &mut KVCache, off: usize, dtype: DType, row: &[f32], is_k: bool) {
        match dtype {
            DType::F32 => {
                let buf = if is_k { c.k_buffer.as_mut_slice::<f32>() } else { c.v_buffer.as_mut_slice::<f32>() };
                buf[off..off + HD].copy_from_slice(row);
            }
            DType::F16 => {
                let buf = if is_k { c.k_buffer.as_mut_slice::<f16>() } else { c.v_buffer.as_mut_slice::<f16>() };
                for d in 0..HD {
                    buf[off + d] = f16::from_f32(row[d]);
                }
            }
            DType::Q4_0 => {
                let bpp = HD / QK4_0;
                let bo = off / QK4_0;
                let buf = if is_k { c.k_buffer.as_mut_slice::<BlockQ4_0>() } else { c.v_buffer.as_mut_slice::<BlockQ4_0>() };
                for bi in 0..bpp {
                    let mut blk = [0.0f32; QK4_0];
                    blk.copy_from_slice(&row[bi * QK4_0..(bi + 1) * QK4_0]);
                    buf[bo + bi] = BlockQ4_0::quantize(&blk);
                }
            }
            _ => unreachable!(),
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

    /// The s1 gate: a StreamingLLM eviction applied through the handle driver is byte-identical to the
    /// same plan applied through `execute_kv_plan`, for f32 / f16 / q4_0.
    ///
    /// Mutation-proof / non-tautological: both caches start byte-identical; only the application path
    /// differs. The keep-set comes from the REAL `find_stage("streaming")` plugin (one plan instance
    /// drives `execute_kv_plan`; a second identical instance drives the handle), so if the handle
    /// commit diverged from the plan executor in ANY byte, the final-buffer comparison would fail.
    fn streaming_handle_byte_identical(dtype: DType) {
        let params = StageParams {
            sink_size: SINK,
            streaming_window: WINDOW,
            ..Default::default()
        };
        let reg = find_stage("streaming").expect("streaming force-linked");

        // Reference cache: streaming plan -> execute_kv_plan (v2).
        let mut cache_v2 = make_cache(dtype);
        let sctx = ScalarStageCtx::from_cache(&cache_v2, 0, 0, 1);
        let plan: KVCachePlan = (reg.make)(params)
            .plan(&sctx)
            .expect("streaming evicts at resident > sink+window");
        execute_kv_plan(&mut cache_v2, &plan, 0, 1).unwrap();

        // Handle cache: same streaming plugin driven through KVMutationDriverStage.
        let cache_h = make_cache(dtype);
        let handle = Arc::new(StandardFormat::new(0, cache_h));
        let driver = KVMutationDriverStage::new(
            vec![handle.clone()],
            Box::new(PlanStageAdapter::new((reg.make)(params), MutationPhase::KvMutate)),
        );
        let mut profiler = OpProfiler::new();
        let mut pctx = make_ctx(&mut profiler);
        let outcome = driver.on_phase(&LifecyclePhase::KvMutate, &mut pctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        let cache_h = handle.take_inner();

        // Same surviving token count, and byte-identical K and V buffers over the resident region.
        assert_eq!(cache_v2.current_pos(), cache_h.current_pos());
        let new_pos = cache_v2.current_pos();
        assert!(new_pos < RESIDENT, "streaming must have evicted something");
        for pos in 0..new_pos {
            for head in 0..KV_HEADS {
                let off = cache_v2.offset(pos, head);
                assert_eq!(off, cache_h.offset(pos, head));
                match dtype {
                    DType::F32 => {
                        let a = cache_v2.k_buffer.as_slice::<f32>();
                        let b = cache_h.k_buffer.as_slice::<f32>();
                        assert_eq!(a[off..off + HD], b[off..off + HD], "K pos {pos} head {head}");
                        let a = cache_v2.v_buffer.as_slice::<f32>();
                        let b = cache_h.v_buffer.as_slice::<f32>();
                        assert_eq!(a[off..off + HD], b[off..off + HD], "V pos {pos} head {head}");
                    }
                    DType::F16 => {
                        let a = cache_v2.k_buffer.as_slice::<f16>();
                        let b = cache_h.k_buffer.as_slice::<f16>();
                        assert_eq!(a[off..off + HD], b[off..off + HD], "K pos {pos} head {head}");
                        let a = cache_v2.v_buffer.as_slice::<f16>();
                        let b = cache_h.v_buffer.as_slice::<f16>();
                        assert_eq!(a[off..off + HD], b[off..off + HD], "V pos {pos} head {head}");
                    }
                    DType::Q4_0 => {
                        let bpp = HD / QK4_0;
                        let bo = off / QK4_0;
                        let a = cache_v2.k_buffer.as_slice::<BlockQ4_0>();
                        let b = cache_h.k_buffer.as_slice::<BlockQ4_0>();
                        for bi in 0..bpp {
                            assert_eq!(a[bo + bi].d, b[bo + bi].d, "K q4 d pos {pos} head {head} blk {bi}");
                            assert_eq!(a[bo + bi].qs, b[bo + bi].qs, "K q4 qs pos {pos} head {head} blk {bi}");
                        }
                        let a = cache_v2.v_buffer.as_slice::<BlockQ4_0>();
                        let b = cache_h.v_buffer.as_slice::<BlockQ4_0>();
                        for bi in 0..bpp {
                            assert_eq!(a[bo + bi].d, b[bo + bi].d, "V q4 d pos {pos} head {head} blk {bi}");
                            assert_eq!(a[bo + bi].qs, b[bo + bi].qs, "V q4 qs pos {pos} head {head} blk {bi}");
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn streaming_via_handle_byte_identical_f32() {
        streaming_handle_byte_identical(DType::F32);
    }

    #[test]
    fn streaming_via_handle_byte_identical_f16() {
        streaming_handle_byte_identical(DType::F16);
    }

    #[test]
    fn streaming_via_handle_byte_identical_q4_0() {
        streaming_handle_byte_identical(DType::Q4_0);
    }
}
