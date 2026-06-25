//! `FormatReencodeStage` — L1-runtime per-layer KV re-encode trigger (D3 format axis).
//!
//! The format twin of [`PrefillKeepSetStage`](super::prefill_keepset::PrefillKeepSetStage): it
//! mirrors that stage's take_inner/plan/execute/put_inner UER, but drives the FORMAT axis — a
//! registered [`KVFormatPolicy`] produces a per-layer [`KVFormatPlan`](argus_extension_api::KVFormatPlan)
//! and the engine applies it with [`apply_format_plan`] (a re-encode), instead of an eviction
//! keep-set. This is the first **production caller** of `apply_format_plan` (until now dormant).
//!
//! Fires once at [`LifecyclePhase::PrefillEnd`]: after the prompt is prefilled, each layer's KV may be
//! re-encoded to the policy's assigned format (the canonical L1-runtime use: downgrade the prefilled
//! KV to a cheaper precision before decode). A policy returning `None` (no change), or a Gate-0 plan
//! (`base` == the layer's current stored format), is a no-op — so when the caches were already
//! allocated in the policy's per-layer format at construction time
//! (`per_layer_storage_from_policy`), this re-encode pass is a byte-identical no-op.
//!
//! GPU note: `PrefillEnd` precedes the first decode step, and the fused decode plan is built lazily on
//! that first step (`decode_loop.rs` — "첫 decode plan 은 lazy"). So a re-encode here is observed by
//! the *initial* plan build (it reads the already-re-encoded caches) — no plan invalidation is needed
//! for this timing. The invalidation guard for a *post-plan-build* (mid-decode) re-encode is
//! `ModelForward::on_kv_reencode` (separate concern).

use std::sync::Arc;

use argus_extension_api::KVFormatPolicy;

use crate::kv::eviction::stage_registry::KVStageCtx;
use crate::kv::format_apply::apply_format_plan;
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};

/// A `PrefillEnd` OneShot stage that applies a [`KVFormatPolicy`]'s per-layer format assignment via
/// [`apply_format_plan`] (re-encode).
pub struct FormatReencodeStage {
    /// register-time handles — enumerate order == layer idx (same W1 invariant as `EvictionStage` /
    /// `PrefillKeepSetStage`).
    handles: Vec<Arc<StandardFormat>>,
    /// per-layer format-assignment producer (`--kv-format <policy>` resolved via `find_format_policy`).
    policy: Box<dyn KVFormatPolicy>,
}

impl FormatReencodeStage {
    /// `handles` enumerate order must equal layer idx. `policy` is the resolved format policy.
    pub fn new(handles: Vec<Arc<StandardFormat>>, policy: Box<dyn KVFormatPolicy>) -> Self {
        Self { handles, policy }
    }
}

impl PipelineStage for FormatReencodeStage {
    fn name(&self) -> &str {
        "kv.format_reencode"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::OneShot
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter: only PrefillEnd.
        if *phase != LifecyclePhase::PrefillEnd {
            return Ok(StageOutcome::Continue);
        }

        // UER (Unwrap-Evict-Rewrap, mirroring PrefillKeepSetStage): take_inner → per-layer
        // assign+apply → put_inner. Defer `?` past the rewrap so the placeholder is never left behind.
        let mut temp: Vec<KVCache> = self.handles.iter().map(|f| f.take_inner()).collect();
        let n_layers = temp.len();
        let result = (|| -> anyhow::Result<()> {
            for (layer_idx, cache) in temp.iter_mut().enumerate() {
                if cache.current_pos() == 0 {
                    continue;
                }
                // plan production: the ctx borrows `&cache` immutably and is dropped before the
                // `&mut cache` re-encode below (no borrow conflict).
                let plan = {
                    let ctx = KVStageCtx::new(cache, cache.current_pos(), None, None, None, None)
                        .with_layer(layer_idx, n_layers);
                    self.policy.assign(&ctx)
                };
                if let Some(plan) = plan {
                    apply_format_plan(cache, &plan, layer_idx, n_layers)?;
                }
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

    use argus_extension_api::{FormatId, KVFormatPlan, StageCtx};

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::dequant::dequantize_k;
    use crate::memory::host::shared::SharedBuffer;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::quant::QK4_0;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use half::f16;

    const HD: usize = 64; // q4_0-valid: a multiple of QK4_0 (=32).
    const KV_HEADS: usize = 2;
    const MAX_SEQ: usize = 16;

    /// SeqMajor F16 cache, current_pos=n, with a known per-(pos,head,d) pattern written into the
    /// resident region. Returns the cache plus the recorded f16-rounded originals (flat [pos][head][d]).
    fn make_f16_cache(n: usize) -> (KVCache, Vec<f32>) {
        let total = MAX_SEQ * KV_HEADS * HD;
        let kb = Arc::new(SharedBuffer::new(total * 2, DType::F16));
        let vb = Arc::new(SharedBuffer::new(total * 2, DType::F16));
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HD]);
        let mut c = KVCache::new(
            Tensor::new(sh.clone(), kb, be.clone()),
            Tensor::new(sh, vb, be),
            MAX_SEQ,
        );
        c.set_current_pos(n);
        let mut orig = vec![0.0f32; n * KV_HEADS * HD];
        {
            let ks = c.k_buffer.as_mut_slice::<f16>();
            for pos in 0..n {
                for head in 0..KV_HEADS {
                    let off = pos * KV_HEADS * HD + head * HD; // SeqMajor
                    for d in 0..HD {
                        let x = f16::from_f32(
                            0.5 + pos as f32 * 0.11
                                + head as f32 * 0.27
                                + (d as f32 - HD as f32 / 2.0) * 0.04,
                        );
                        ks[off + d] = x;
                        orig[(pos * KV_HEADS + head) * HD + d] = x.to_f32();
                    }
                }
            }
        }
        (c, orig)
    }

    /// A format policy that assigns one fixed format to every layer (forces a real re-encode when the
    /// cache is in a different format). Mirror of the registered `mixed_precision` producer shape.
    struct ForceFormatPolicy(&'static str);
    impl KVFormatPolicy for ForceFormatPolicy {
        fn name(&self) -> &str {
            "test.force_format"
        }
        fn assign(&self, _ctx: &dyn StageCtx) -> Option<KVFormatPlan> {
            Some(KVFormatPlan {
                base: FormatId(self.0.into()),
                overrides: Vec::new(),
            })
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

    #[test]
    fn prefill_end_reencodes_layer_to_target_format() {
        // Live trigger: PrefillEnd → take_inner / policy.assign / apply_format_plan / put_inner. An
        // F16 cache forced to q4_0 must (a) flip its stored dtype and (b) read back faithfully (the
        // values the decode forward will consume). A silent no-op would leave dtype F16 → fail.
        let (cache, orig) = make_f16_cache(8);
        let resident = cache.current_pos();
        let handle = Arc::new(StandardFormat::new(0, cache));
        let stage =
            FormatReencodeStage::new(vec![handle.clone()], Box::new(ForceFormatPolicy("q4_0")));

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::PrefillEnd, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));

        let inner = handle.take_inner();
        assert_eq!(inner.kv_dtype(), DType::Q4_0, "layer re-encoded to q4_0");
        assert_eq!(inner.current_pos(), resident, "current_pos preserved");

        // Forward-read correctness: dequant of the re-encoded cache is within q4_0 tolerance of the
        // original f16 values (what attention reads).
        let mut got = vec![0.0f32; HD];
        for pos in 0..resident {
            for head in 0..KV_HEADS {
                dequantize_k(&inner, pos, head, HD, &mut got);
                let idx = (pos * KV_HEADS + head) * HD;
                for bi in 0..(HD / QK4_0) {
                    let slice = &orig[idx + bi * QK4_0..idx + (bi + 1) * QK4_0];
                    let max_abs = slice.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let tol = max_abs / 7.0 + 1e-3;
                    for j in 0..QK4_0 {
                        let o = orig[idx + bi * QK4_0 + j];
                        let g = got[bi * QK4_0 + j];
                        assert!(
                            (g - o).abs() <= tol,
                            "pos {pos} head {head}: |{g}-{o}| > {tol}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn non_prefill_end_phase_is_noop() {
        // self-filter: off PrefillEnd → Continue + cache unchanged (still F16).
        let (cache, _) = make_f16_cache(8);
        let handle = Arc::new(StandardFormat::new(0, cache));
        let stage =
            FormatReencodeStage::new(vec![handle.clone()], Box::new(ForceFormatPolicy("q4_0")));
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::DecodeStart, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Continue));
        assert_eq!(
            handle.take_inner().kv_dtype(),
            DType::F16,
            "no re-encode off PrefillEnd"
        );
    }

    #[test]
    fn gate0_same_format_is_byte_identical_noop() {
        // Policy assigns the layer's CURRENT format (f16) → Gate-0 no-op: dtype + bytes unchanged.
        let (cache, orig) = make_f16_cache(8);
        let handle = Arc::new(StandardFormat::new(0, cache));
        let stage =
            FormatReencodeStage::new(vec![handle.clone()], Box::new(ForceFormatPolicy("f16")));
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::PrefillEnd, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        let inner = handle.take_inner();
        assert_eq!(inner.kv_dtype(), DType::F16, "f16→f16 is a no-op");
        // bytes intact: re-read the f16 values, compare exactly to the originals.
        let ks = inner.k_buffer.as_slice::<f16>();
        for pos in 0..8 {
            for head in 0..KV_HEADS {
                let off = pos * KV_HEADS * HD + head * HD;
                let idx = (pos * KV_HEADS + head) * HD;
                for d in 0..HD {
                    assert_eq!(ks[off + d].to_f32(), orig[idx + d]);
                }
            }
        }
    }
}
