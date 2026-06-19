//! `LayerSlot` — atomic weight container for one decoder layer.
//!
//! Encapsulates one decoder block's weights behind an `ArcSwap` for lock-free
//! snapshot reads. Phase 1 only exercises the initial-install path; runtime
//! swap (`SwapExecutor::swap_layer`) lands in Phase 2.
//!
//! Spec: ENG-DAT-092 (`LayerSlot`), INV-123 (atomic snapshot), INV-124
//! (current_dtype <-> weights dtype consistency).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use arc_swap::ArcSwap;

use crate::buffer::DType;
use crate::format::weight_format::{LayerDispatch, WeightFormat};
use crate::hardware::Hardware;
use crate::layers::tensor_partition::{PartitionContext, split_weight, split_weight_col};
use crate::layers::transformer_layer::TransformerLayer;

use super::secondary_mmap::SecondaryMmap;

/// Phase 1 alias: one layer's weight bundle is the existing `TransformerLayer`
/// struct. Phase 2 may evolve this to a dedicated weight-only struct; for now
/// we reuse the proven layout to minimise the forward-path diff.
pub type LayerWeights = TransformerLayer;

/// Encode a `DType` into the `AtomicU8` discriminant stored by `LayerSlot`.
#[inline]
fn dtype_to_u8(dt: DType) -> u8 {
    match dt {
        DType::Q4_0 => 0,
        DType::Q4_1 => 1,
        DType::Q8_0 => 2,
        DType::F16 => 3,
        DType::BF16 => 4,
        DType::F32 => 5,
        DType::U8 => 6,
    }
}

#[inline]
fn u8_to_dtype(v: u8) -> DType {
    match v {
        0 => DType::Q4_0,
        1 => DType::Q4_1,
        2 => DType::Q8_0,
        3 => DType::F16,
        4 => DType::BF16,
        5 => DType::F32,
        6 => DType::U8,
        _ => panic!("LayerSlot: corrupt current_dtype discriminant {v}"),
    }
}

/// One decoder layer's swappable slot.
///
/// Readers (forward pass): `slot.load_weights()` → `Arc<LayerWeights>`
/// snapshot, usable for the remainder of the layer without further atomics.
///
/// Writers (`SwapExecutor`, Phase 2): `slot.swap_weights(new_arc, new_dtype)`
/// atomically installs a new snapshot and bumps `generation`.
pub struct LayerSlot {
    /// Current weight dtype. Updated in the same logical step as `weights` on
    /// swap (INV-124). Encoded as `AtomicU8` because `DType` is not atomic.
    current_dtype: AtomicU8,
    /// Wait-free snapshot of the layer weights.
    weights: ArcSwap<LayerWeights>,
    /// Optional secondary mmap handle shared with `TransformerWeights`. When
    /// `None`, this slot cannot be swapped. INV-125 requires that this handle
    /// remain alive for the slot's entire lifetime.
    secondary_mmap_handle: Option<Arc<SecondaryMmap>>,
    /// Per-slot generation counter. Incremented on each successful swap. Used
    /// by downstream plan invalidation analogous to INV-120.
    generation: AtomicU64,
    /// Decoder layer index this slot represents. Surfaced via
    /// `WeightFormat::idx()` so the Stage layer can address a specific layer
    /// (plan.rs already threads `layer_idx` separately; LayerSlot ownership is
    /// forward-looking for the α-W dispatch wiring).
    layer_idx: usize,
}

impl LayerSlot {
    /// Build a new slot from an initial weight snapshot.
    pub fn new(
        weights: LayerWeights,
        dtype: DType,
        secondary_mmap_handle: Option<Arc<SecondaryMmap>>,
        layer_idx: usize,
    ) -> Self {
        Self {
            current_dtype: AtomicU8::new(dtype_to_u8(dtype)),
            weights: ArcSwap::from(Arc::new(weights)),
            secondary_mmap_handle,
            generation: AtomicU64::new(0),
            layer_idx,
        }
    }

    /// Acquire a full Arc snapshot of the current weights. O(1) atomic load
    /// plus one Arc clone; the returned handle is safe to use for the rest of
    /// the current layer's forward pass even if a swap lands concurrently.
    #[inline]
    pub fn load_weights(&self) -> Arc<LayerWeights> {
        self.weights.load_full()
    }

    /// Current dtype (INV-124).
    #[inline]
    pub fn current_dtype(&self) -> DType {
        u8_to_dtype(self.current_dtype.load(Ordering::Acquire))
    }

    /// Current generation counter value.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Reference to the secondary mmap handle (None if swap disabled for this
    /// slot).
    #[inline]
    pub fn secondary_mmap_handle(&self) -> Option<&Arc<SecondaryMmap>> {
        self.secondary_mmap_handle.as_ref()
    }

    /// Phase 2 seam — atomic weight swap.
    ///
    /// Replaces the weight snapshot, updates `current_dtype`, and bumps
    /// `generation`. INV-124 requires the dtype update and snapshot install to
    /// occur within one logical step; we use Release ordering so any reader
    /// observing the new generation sees both the new Arc and the new dtype.
    ///
    /// Internally uses `ArcSwap::swap()` rather than `store()`: this returns
    /// the previous `Arc<LayerWeights>` after `wait_for_readers` so the caller
    /// can drop it deterministically — including the path that explicitly
    /// reclaims the primary cl_mem (WSWAP-5-PRIMARY-DROP). The returned Arc
    /// is the only outstanding reference held by the slot; in steady state
    /// the caller observes `Arc::strong_count == 1` and can `try_unwrap`
    /// to release primary buffers without waiting for the destructor chain.
    pub fn swap_weights(
        &self,
        new_weights: Arc<LayerWeights>,
        new_dtype: DType,
    ) -> Arc<LayerWeights> {
        self.current_dtype
            .store(dtype_to_u8(new_dtype), Ordering::Release);
        let old = self.weights.swap(new_weights);
        self.generation.fetch_add(1, Ordering::Release);
        old
    }

    /// Install fresh weights while preserving the current generation counter.
    /// Used for in-place rewrap flows (e.g. `rewrap_weights_for_dual_access`)
    /// that need the same dtype but a new backing buffer.
    pub fn store_weights_same_dtype(&self, new_weights: Arc<LayerWeights>) {
        self.weights.store(new_weights);
    }

    /// Perform an in-place closure update of the weight snapshot under RCU
    /// semantics. Used during partition (re)configuration: the closure
    /// receives a reference to the current `Arc<LayerWeights>`, clones the
    /// inner struct, mutates it, and the ArcSwap installs the new Arc
    /// atomically.
    ///
    /// On contention the closure may be invoked more than once, so it must be
    /// idempotent / side-effect free.
    pub fn rcu_weights<F>(&self, mut f: F)
    where
        F: FnMut(&LayerWeights) -> LayerWeights,
    {
        self.weights.rcu(|current| Arc::new(f(current.as_ref())));
    }
}

/// `WeightFormat` dispatch wiring (Phase α-W-5).
///
/// `apply_dispatch` generalizes the former `transformer.rs::prepare_tensor_partition`
/// per-slot loop body. The gen-counter / RCU store sequencing is transplanted
/// verbatim from there to preserve INV-120 (stale-plan detection) bit-for-bit.
/// The companion CPU backend is resolved from `Hardware` instead of being a
/// baked-in parameter (§4.2 (2)); the cached `PartitionContext.cpu_backend`
/// field is unchanged (forward hot-path avoids per-token resolve).
///
/// `rcu_weights` is intentionally **not** used: the closure may re-run under
/// contention and the generation `fetch_add` is a side effect, so we use the
/// explicit `load_weights` + clone + `store_weights_same_dtype` pattern.
impl WeightFormat for LayerSlot {
    fn idx(&self) -> usize {
        self.layer_idx
    }

    fn apply_dispatch(&self, d: LayerDispatch, hw: &Hardware) -> anyhow::Result<()> {
        match d {
            // GPU-only fast path. Bump any surviving generation counter before
            // clearing the context so a plan built against the prior ratio sees
            // `PlanInvalidated`; a `None` slot has no plan to invalidate, so it
            // must **not** bump. (transformer.rs:1011-1018 verbatim)
            LayerDispatch::Full => {
                let old = self.load_weights();
                if let Some(ref ctx) = old.partition_ctx {
                    ctx.ratio_generation.fetch_add(1, Ordering::Release);
                }
                let mut new = (*old).clone();
                new.partition_ctx = None;
                self.store_weights_same_dtype(Arc::new(new));
                Ok(())
            }
            // Partition install. (transformer.rs:1023-1068 verbatim; only the
            // companion backend source changes: cpu_backend → hw.resolve(Cpu).)
            LayerDispatch::Partition(shares) => {
                let gpu_spec = &shares[0]; // GPU
                let cpu_spec = &shares[1]; // CPU companion
                let gpu_ratio = gpu_spec.share;
                // per-slice format 은 weight dtype 에서 파생(split byte layout 원천,
                // `bytes_per_row`) — `PartitionShare` 표면에 없으므로 변환/assert 불요
                // (format = executor-내부).
                let cpu_backend = hw
                    .resolve(cpu_spec.hardware.into())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "{:?} backend not available (partition companion)",
                            cpu_spec.hardware
                        )
                    })?
                    .0
                    .clone();
                let old = self.load_weights();
                // Reuse the existing Arc<AtomicU64> if a partition_ctx is already
                // installed so that plans built against the prior generation see
                // the bump. A fresh install starts at 0.
                let prev_gen = old
                    .partition_ctx
                    .as_ref()
                    .map(|c| c.ratio_generation.clone());

                // Strategy B: whole-FFN slice. gate/up split_row is on the
                // ffn_hidden (out_dim) axis. down split_col is on the ffn_hidden
                // (in_dim) axis — same logical dimension, so we reuse gate's
                // split_row.
                let gate = split_weight(&old.w_gate, gpu_ratio, &cpu_backend)?;
                let up = split_weight(&old.w_up, gpu_ratio, &cpu_backend)?;
                debug_assert_eq!(
                    gate.split_row, up.split_row,
                    "gate/up split_row must match (same ffn_hidden, same gpu_ratio)",
                );
                let down = split_weight_col(&old.w_down, gate.split_row, &cpu_backend)?;

                // Bump the shared counter if this is a re-split; allocate a
                // fresh counter at 0 otherwise. Release ordering pairs with the
                // Acquire load in `PartitionStep::run`.
                let gen_arc = match prev_gen {
                    Some(g) => {
                        g.fetch_add(1, Ordering::Release);
                        g
                    }
                    None => Arc::new(AtomicU64::new(0)),
                };

                let mut new = (*old).clone();
                new.partition_ctx = Some(PartitionContext {
                    gpu_ratio,
                    cpu_backend,
                    gate,
                    up,
                    down,
                    ratio_generation: gen_arc,
                });
                self.store_weights_same_dtype(Arc::new(new));
                Ok(())
            }
            LayerDispatch::Skip => {
                anyhow::bail!(
                    "layer skip dispatch not wired (Phase β layer-skip; arch §4.2 (6) mode preserved)"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::format::weight_format::PartitionShare;
    use crate::memory::Memory;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    #[test]
    fn dtype_roundtrip_covers_all_variants() {
        for dt in [
            DType::Q4_0,
            DType::Q4_1,
            DType::Q8_0,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::U8,
        ] {
            assert_eq!(u8_to_dtype(dtype_to_u8(dt)), dt);
        }
    }

    // ── INV-120 regression guard (B7) ───────────────────────────────────────
    //
    // The S25 `--tensor-partition` device gate only exercises a single install
    // (no ratio change), so it cannot catch a regression in the re-split
    // generation bump. This focused unit test drives `apply_dispatch(Partition)`
    // twice on one slot and asserts the shared `ratio_generation` counter is
    // (1) reused across re-splits and (2) bumped by exactly one per re-split —
    // the INV-120 invariant transplanted from `prepare_tensor_partition`.

    fn cpu_be() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    fn f32_weight(be: &Arc<dyn Backend>, out_dim: usize, in_dim: usize) -> Tensor {
        let numel = out_dim * in_dim;
        let buf: Arc<dyn crate::buffer::Buffer> =
            Arc::new(SharedBuffer::new(numel * 4, DType::F32));
        Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, be.clone())
    }

    fn ffn_layer(be: &Arc<dyn Backend>) -> LayerWeights {
        // FFN weights are 2D and large enough to satisfy split_weight's
        // out_dim >= 256 / 128-alignment requirements. Attention weights are
        // never partitioned, so a small placeholder shape is fine.
        let small = f32_weight(be, 1, 1);
        TransformerLayer {
            wq: small.clone(),
            wk: small.clone(),
            wv: small.clone(),
            wo: small.clone(),
            w_gate: f32_weight(be, 512, 256),
            w_up: f32_weight(be, 512, 256),
            w_down: f32_weight(be, 256, 512),
            attention_norm: small.clone(),
            ffn_norm: small,
            qkv_bias: None,
            q_norm: None,
            k_norm: None,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            partition_ctx: None,
        }
    }

    fn cpu_only_hardware(be: &Arc<dyn Backend>) -> Hardware {
        let host: Arc<dyn Memory> = Arc::new(crate::memory::galloc::Galloc::new());
        Hardware::new(be.clone(), None, None, host, None)
    }

    fn partition_specs() -> LayerDispatch {
        LayerDispatch::Partition(vec![
            PartitionShare {
                share: 0.5,
                hardware: argus_extension_api::DeviceTarget::Gpu,
            },
            PartitionShare {
                share: 0.5,
                hardware: argus_extension_api::DeviceTarget::Cpu,
            },
        ])
    }

    #[test]
    fn apply_dispatch_partition_reuses_and_bumps_generation() {
        let be = cpu_be();
        let hw = cpu_only_hardware(&be);
        let slot = LayerSlot::new(ffn_layer(&be), DType::F32, None, 3);

        assert_eq!(slot.idx(), 3, "idx() returns the stored layer_idx");

        // First install: fresh partition_ctx, generation starts at 0.
        slot.apply_dispatch(partition_specs(), &hw).unwrap();
        let gen_arc = slot
            .load_weights()
            .partition_ctx
            .as_ref()
            .expect("partition_ctx installed")
            .ratio_generation
            .clone();
        assert_eq!(
            gen_arc.load(Ordering::Acquire),
            0,
            "fresh install: generation == 0"
        );

        // Re-split: same shared counter, bumped to 1.
        slot.apply_dispatch(partition_specs(), &hw).unwrap();
        let gen_after = slot
            .load_weights()
            .partition_ctx
            .as_ref()
            .expect("partition_ctx still installed")
            .ratio_generation
            .clone();
        assert!(
            Arc::ptr_eq(&gen_arc, &gen_after),
            "re-split must reuse the same Arc<AtomicU64> so old plans observe the bump"
        );
        assert_eq!(
            gen_after.load(Ordering::Acquire),
            1,
            "re-split: generation bumped by exactly one"
        );

        // Second re-split: monotonic bump to 2.
        slot.apply_dispatch(partition_specs(), &hw).unwrap();
        assert_eq!(
            gen_after.load(Ordering::Acquire),
            2,
            "second re-split: generation == 2"
        );
    }

    #[test]
    fn apply_dispatch_full_bumps_only_when_ctx_present() {
        let be = cpu_be();
        let hw = cpu_only_hardware(&be);
        let slot = LayerSlot::new(ffn_layer(&be), DType::F32, None, 0);

        // Full on a slot with no partition_ctx: no counter to bump, no-op clear.
        slot.apply_dispatch(LayerDispatch::Full, &hw).unwrap();
        assert!(
            slot.load_weights().partition_ctx.is_none(),
            "Full leaves partition_ctx == None"
        );

        // Install a partition, then Full must bump the surviving counter before
        // clearing the context (so a stale plan sees PlanInvalidated).
        slot.apply_dispatch(partition_specs(), &hw).unwrap();
        let gen_arc = slot
            .load_weights()
            .partition_ctx
            .as_ref()
            .unwrap()
            .ratio_generation
            .clone();
        assert_eq!(gen_arc.load(Ordering::Acquire), 0);

        slot.apply_dispatch(LayerDispatch::Full, &hw).unwrap();
        assert!(
            slot.load_weights().partition_ctx.is_none(),
            "Full clears partition_ctx"
        );
        assert_eq!(
            gen_arc.load(Ordering::Acquire),
            1,
            "Full bumps the surviving counter exactly once before clearing"
        );
    }
}
