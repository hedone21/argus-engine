//! HYBRID v3 — engine-side transaction engine for the imperative [`CacheHandle`] ABI.
//!
//! [`EngineCacheHandle`] is the engine's implementation of the additive
//! [`argus_extension_api::CacheHandle`] surface (the M4 callback class, the imperative surface a
//! [`KVMutationStage`] mutates through). It is the *transaction* layer that makes imperative mutation
//! sound:
//!
//! - **T-1 deferred-commit / single-renumber**: position-mutating ops (`keep`/`evict`/`keep_per_head`)
//!   only *stage* an intent; the engine applies it once in [`EngineCacheHandle::commit`].
//! - **T-2 at-most-one compaction**: a second position-mutating op returns
//!   [`CacheOpError::MultipleCompactions`].
//! - **T-3 entry-frame reads**: reads observe the pre-callback `current_pos` (nothing mutates the
//!   cache until `commit`, so reads naturally see the entry frame).
//! - **T-5 position-preserving exemption**: `reencode` is exempt from the at-most-one rule.
//! - **T-8 Err leaves bytes untouched**: every op validates *eagerly* at call time, so a rejected op
//!   never stages and never mutates; `commit` then applies only pre-validated intents.
//! - **T-9 cl_mem never exposed**: no op hands back a raw device buffer.
//! - **T-10 keep hardening**: keep-lists are validated ascending + unique + in-range *before* any
//!   mutation — closing the historical leaky-keep hole in the v2 plan executor.
//!
//! Every op routes to an EXISTING executor (`compact_keep_positions` / `compact_keep_positions_for_head`
//! / `apply_weighted_merges` / `apply_format_plan`), so a handle-driven keep is byte-identical to the
//! same keep applied through the v2 plan executor (the s1 gate).

use crate::kv::kv_cache::KVCache;
use crate::kv_cache_ops::KVLayout;
use anyhow::Result;
use argus_extension_api::{
    CacheHandle, CacheOpError, FormatId, KVFormatPlan, KeepSpec, TensorHandle, TensorKind,
    WeightedMerge,
};

/// The cache's current stored-format name, mirroring `format_apply::current_format_name` (the
/// `register_kv_format!` floor names). Used to skip a re-encode to the already-current format.
fn current_format_name(cache: &KVCache) -> &'static str {
    match cache.kv_dtype() {
        crate::buffer::DType::F32 => "f32",
        crate::buffer::DType::F16 => "f16",
        crate::buffer::DType::Q4_0 => "q4_0",
        _ => "unknown",
    }
}

/// The single staged position-mutating compaction slot (T-2: at most one per callback).
enum Compaction {
    /// LayerWide keep — all heads keep the same ascending positions.
    Keep(Vec<usize>),
    /// Per-head keep — `[n_kv_heads][keep]`, each ascending, all equal length (HeadMajor only).
    KeepPerHead(Vec<Vec<usize>>),
    /// Offload the LRU prefix of `n` tokens through the swap handler (residency axis).
    Offload(usize),
    /// Recall this layer's outstanding offloaded prefix through the swap handler.
    Recall,
}

/// Engine-side [`CacheHandle`] over **one layer's** `&mut KVCache`. Stages position-mutating intents
/// and applies them once in [`commit`](Self::commit); position-preserving ops (`reencode`) take effect
/// at commit too. See the module docs for the transaction invariants.
pub struct EngineCacheHandle<'a> {
    cache: &'a mut KVCache,
    layer_idx: usize,
    n_layers: usize,
    /// The swap handler backing `offload` / `recall` (residency axis). `None` ⇒ those ops are
    /// unconfigured and reject (the common keep/merge/reencode path needs no swap).
    swap: Option<&'a crate::kv::swap_handler::SwapHandler>,
    /// Entry-frame token count (T-3): the frame every read + every keep validation observes.
    entry_pos: usize,
    /// Staged weighted merges (applied in the pre-compaction frame at commit).
    merges: Option<Vec<WeightedMerge>>,
    /// The single staged compaction (T-2).
    compaction: Option<Compaction>,
    /// Staged re-encode target (position-preserving; last write wins, T-5).
    reencode: Option<FormatId>,
    /// Whether `commit` mutated any bytes. Intended for the production driver's coalesced plan
    /// invalidation (T-6) — a mid-decode KV mutation must invalidate the fused decode plan, mirroring
    /// `FormatReencodeStage::reencode_fired`. NOTE: the s1 `KVMutationDriverStage` does NOT yet consult
    /// it (it discards `commit`'s return); routing this into plan invalidation is part of the
    /// production driver-wiring follow-up.
    mutated: bool,
}

impl<'a> EngineCacheHandle<'a> {
    /// Build a handle over one layer's cache for a mutation callback (no swap backend — `offload` /
    /// `recall` reject until built via [`with_swap`](Self::with_swap)).
    pub fn new(cache: &'a mut KVCache, layer_idx: usize, n_layers: usize) -> Self {
        let entry_pos = cache.current_pos();
        Self {
            cache,
            layer_idx,
            n_layers,
            swap: None,
            entry_pos,
            merges: None,
            compaction: None,
            reencode: None,
            mutated: false,
        }
    }

    /// Build a handle with a swap backend so `offload` / `recall` are live (residency axis).
    pub fn with_swap(
        cache: &'a mut KVCache,
        layer_idx: usize,
        n_layers: usize,
        swap: &'a crate::kv::swap_handler::SwapHandler,
    ) -> Self {
        let mut h = Self::new(cache, layer_idx, n_layers);
        h.swap = Some(swap);
        h
    }

    /// Validate a keep-list against the entry frame: ascending, unique (strictly increasing), and
    /// every index in `[0, entry_pos)` (T-10). Returns [`CacheOpError::InvalidKeep`] otherwise.
    fn validate_keep(&self, keep: &[usize]) -> Result<(), CacheOpError> {
        let mut prev: Option<usize> = None;
        for &k in keep {
            if k >= self.entry_pos {
                return Err(CacheOpError::InvalidKeep);
            }
            if let Some(p) = prev
                && k <= p
            {
                return Err(CacheOpError::InvalidKeep);
            }
            prev = Some(k);
        }
        Ok(())
    }

    /// Stage a compaction, rejecting a second one (T-2).
    fn set_compaction(&mut self, c: Compaction) -> Result<(), CacheOpError> {
        if self.compaction.is_some() {
            return Err(CacheOpError::MultipleCompactions);
        }
        self.compaction = Some(c);
        Ok(())
    }

    /// Apply the staged transaction in canonical order and return whether any bytes changed.
    ///
    /// Order: re-encode (position-preserving, commutes with row permutation) → merges (pre-compaction
    /// frame) → compaction (the single renumber, T-1).
    ///
    /// Atomicity contract (T-8, precise): every op is validated *eagerly* at call time, so a rejected
    /// op (`CacheOpError`) never stages and never mutates — that path is fully all-or-nothing. At
    /// commit, the staged ops apply sequentially with no buffer snapshot; the only residual failure
    /// modes are an executor I/O / backend error (e.g. `offload` disk write, `compact` GPU
    /// `buffer_shift`) AFTER an earlier byte-mutating step has run. In that case commit returns `Err`
    /// with the earlier step applied — multi-op commits are NOT rolled back. Single-op commits, and
    /// any eager rejection, remain strictly all-or-nothing. (Eager validation — merge positions,
    /// reencode floor/head_dim, offload/recall backend presence — removes every *reachable*
    /// commit-time failure for the in-tree executors; buffer-snapshot rollback for the I/O residual is
    /// a deliberate non-goal until a production multi-op caller needs it.)
    pub fn commit(mut self) -> Result<bool> {
        if let Some(target) = self.reencode.take() {
            let plan = KVFormatPlan {
                base: target,
                overrides: vec![],
            };
            crate::kv::format_apply::apply_format_plan(
                self.cache,
                &plan,
                self.layer_idx,
                self.n_layers,
            )
            .map_err(|e| anyhow::anyhow!("re-encode commit failed: {e}"))?;
            self.mutated = true;
        }
        if let Some(merges) = self.merges.take()
            && !merges.is_empty()
        {
            crate::kv::standard_format::apply_weighted_merges(self.cache, &merges);
            self.mutated = true;
        }
        match self.compaction.take() {
            Some(Compaction::Keep(keep)) => {
                // P0-2: record the FINAL committed keep-set BEFORE compaction, so the dump's
                // positions are absolute indices into the pre-eviction `[0, current_pos)` frame
                // (reencode/merges above are position-preserving, so current_pos == entry_pos here).
                // Gated on `is_active()` so the default disabled path neither allocates the KeepSpec
                // nor calls record — keeping the commit byte-identical AND allocation-free (the v2
                // the v2 plan executor borrows `&plan.keep` for free; here the keep is a staged `Vec`, so
                // wrapping it in a KeepSpec would otherwise clone even with the dump off). When active,
                // a handle-driven eviction appears in keepset dumps identically to the v2 path.
                if crate::kv::eviction::keepset_dump::is_active() {
                    crate::kv::eviction::keepset_dump::record(
                        self.cache,
                        &KeepSpec::LayerWide(keep.clone()),
                        self.layer_idx,
                        self.n_layers,
                    );
                }
                self.cache.compact_keep_positions(&keep, 0)?;
                self.cache.set_current_pos(keep.len());
                self.mutated = true;
            }
            Some(Compaction::KeepPerHead(heads)) => {
                if crate::kv::eviction::keepset_dump::is_active() {
                    crate::kv::eviction::keepset_dump::record(
                        self.cache,
                        &KeepSpec::PerHead(heads.clone()),
                        self.layer_idx,
                        self.n_layers,
                    );
                }
                let new_pos = heads.first().map_or(0, |h| h.len());
                for (kv_head, keep) in heads.iter().enumerate() {
                    self.cache
                        .compact_keep_positions_for_head(kv_head, keep, 0)?;
                }
                self.cache.set_current_pos(new_pos);
                self.mutated = true;
            }
            Some(Compaction::Offload(prefix)) => {
                let swap = self
                    .swap
                    .ok_or_else(|| anyhow::anyhow!("offload requires a swap handler"))?;
                let n = swap.offload_prefix(self.layer_idx, self.cache, prefix)?;
                self.mutated |= n > 0;
            }
            Some(Compaction::Recall) => {
                let swap = self
                    .swap
                    .ok_or_else(|| anyhow::anyhow!("recall requires a swap handler"))?;
                let n = swap.recall_layer(self.layer_idx, self.cache)?;
                self.mutated |= n > 0;
            }
            None => {}
        }
        Ok(self.mutated)
    }
}

impl CacheHandle for EngineCacheHandle<'_> {
    fn current_pos(&self) -> usize {
        self.entry_pos
    }
    fn n_kv_heads(&self) -> usize {
        self.cache.kv_heads()
    }
    fn head_dim(&self) -> usize {
        self.cache.head_dim()
    }
    fn kv_on_device(&self) -> bool {
        self.cache.k_buffer.buffer().is_gpu_buffer()
    }
    fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
        // In s1 the handle is a pure mutation surface: a mutation stage reads through its companion
        // `StageCtx` (the `on_phase(ctx, cache)` ctx argument), which observes the same entry frame
        // (nothing mutates the cache until `commit`). Mid-transaction handle reads are a follow-up;
        // until then every kind is unavailable via this surface.
        None
    }

    fn keep(&mut self, keep: &[usize]) -> Result<(), CacheOpError> {
        self.validate_keep(keep)?;
        self.set_compaction(Compaction::Keep(keep.to_vec()))
    }

    fn keep_per_head(&mut self, keep: &[&[usize]]) -> Result<(), CacheOpError> {
        // Per-head compaction requires HeadMajor layout (so a head's tokens are contiguous and
        // shiftable in isolation); reject on SeqMajor rather than silently corrupting.
        if self.cache.layout() != KVLayout::HeadMajor {
            return Err(CacheOpError::WrongContainer);
        }
        if keep.len() != self.cache.kv_heads() {
            return Err(CacheOpError::InvalidKeep);
        }
        // All heads keep the same NUMBER of tokens (the single shared current_pos invariant), and each
        // list is ascending + unique + in-range. Validate ALL before staging any (T-8 atomicity).
        let new_pos = keep.first().map_or(0, |h| h.len());
        for head in keep {
            if head.len() != new_pos {
                return Err(CacheOpError::InvalidKeep);
            }
            self.validate_keep(head)?;
        }
        self.set_compaction(Compaction::KeepPerHead(
            keep.iter().map(|h| h.to_vec()).collect(),
        ))
    }

    fn merge(&mut self, merges: &[WeightedMerge]) -> Result<(), CacheOpError> {
        // The CPU merge executor reads via `as_mut_slice`; a device-only buffer would null-deref.
        if self.kv_on_device() {
            return Err(CacheOpError::NotOnHost);
        }
        if self.merges.is_some() {
            return Err(CacheOpError::MergeAlreadyStaged);
        }
        // Eager position validation (the merge twin of T-10): every into/from must be in the entry
        // frame. The CPU executor indexes `offset(pos, head)` with no bounds guard, so an out-of-range
        // position would panic (>= capacity) or silently sum non-resident tail (>= current_pos);
        // reject before staging so a malformed merge never mutates (T-8).
        for m in merges {
            if m.into >= self.entry_pos || m.from.iter().any(|&(p, _)| p >= self.entry_pos) {
                return Err(CacheOpError::InvalidMerge);
            }
        }
        self.merges = Some(merges.to_vec());
        Ok(())
    }

    fn reencode(&mut self, target: FormatId) -> Result<(), CacheOpError> {
        // Eager feasibility (T-8): the host re-encode path cannot read a device-resident buffer, and
        // it can only materialize typed-floor targets (f32/f16/q4_0; opaque codecs route elsewhere).
        if self.kv_on_device() {
            return Err(CacheOpError::NotOnHost);
        }
        // Eager SOURCE-floor check: apply_format_plan's host re-encoder can only READ a typed-floor
        // source. An opaque/codec-stored cache (e.g. a q2_0-descriptor U8 buffer) would otherwise stage
        // Ok here and fail deep inside commit — surface it now as a CacheOpError at the call site so the
        // documented eager-validation contract holds for opaque sources too.
        if !matches!(
            self.cache.kv_dtype(),
            crate::buffer::DType::F32 | crate::buffer::DType::F16 | crate::buffer::DType::Q4_0
        ) {
            return Err(CacheOpError::UnsupportedFormat(
                current_format_name(self.cache).to_string(),
            ));
        }
        if !matches!(target.0.as_str(), "f32" | "f16" | "q4_0") {
            return Err(CacheOpError::UnsupportedFormat(target.0));
        }
        // A re-encode to the cache's CURRENT stored format is a byte-identical no-op (apply_format_plan
        // Gate-0). CLEAR any previously staged re-encode (last-write-wins, T-5): "re-encode back to the
        // current format" is the caller's last word and must CANCEL an earlier staged target, not be
        // silently overridden by it at commit. (Early-returning Ok without clearing would leave the
        // earlier target staged, so commit would apply it — the bug this guards against.)
        if target.0 == current_format_name(self.cache) {
            self.reencode = None;
            return Ok(());
        }
        // q4_0 tiles a head's `head_dim` values into QK4_0-sized blocks; a non-multiple head_dim
        // cannot be re-encoded — check eagerly so commit's apply_format_plan never fails on it.
        if target.0 == "q4_0" && !self.cache.head_dim().is_multiple_of(crate::quant::QK4_0) {
            return Err(CacheOpError::UnsupportedFormat(target.0));
        }
        self.reencode = Some(target);
        Ok(())
    }

    fn transition_quant_bits(&mut self, _bits: u8) -> Result<(), CacheOpError> {
        // A StandardFormat cache has no runtime-transitionable bit-width; this is a quant-window
        // container op (follow-up). Honest reject rather than silent no-op.
        Err(CacheOpError::WrongContainer)
    }

    fn offload(&mut self, prefix_len: usize) -> Result<(), CacheOpError> {
        // Residency compaction-slot op (T-2). Reject eagerly when no swap backend is configured, so it
        // never stages alongside a byte-mutating reencode/merge that a commit-time failure would orphan
        // (T-8). With a backend, stage it; commit routes to swap_handler::offload_prefix.
        if self.swap.is_none() {
            return Err(CacheOpError::NoResidencyBackend);
        }
        self.set_compaction(Compaction::Offload(prefix_len))
    }
    fn recall(&mut self) -> Result<(), CacheOpError> {
        if self.swap.is_none() {
            return Err(CacheOpError::NoResidencyBackend);
        }
        self.set_compaction(Compaction::Recall)
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

    const MAX_SEQ: usize = 32;
    const HD: usize = 4;
    const N_KV: usize = 2;

    /// An F32 SeqMajor KVCache with `resident` tokens, each (pos,head,d) filled with a distinct value.
    fn cache_f32(resident: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let buf = || {
            Arc::new(SharedBuffer::new(
                N_KV * MAX_SEQ * HD * std::mem::size_of::<f32>(),
                DType::F32,
            ))
        };
        let shape = Shape::new(vec![1, MAX_SEQ, N_KV, HD]);
        let mut c = KVCache::new(
            Tensor::new(shape.clone(), buf(), backend.clone()),
            Tensor::new(shape, buf(), backend),
            MAX_SEQ,
        );
        for pos in 0..resident {
            for head in 0..N_KV {
                let off = c.offset(pos, head);
                let k = c.k_buffer.as_mut_slice::<f32>();
                for d in 0..HD {
                    k[off + d] = (pos * 100 + head * 10 + d) as f32;
                }
                let v = c.v_buffer.as_mut_slice::<f32>();
                for d in 0..HD {
                    v[off + d] = (pos * 100 + head * 10 + d) as f32 + 0.5;
                }
            }
        }
        c.set_current_pos(resident);
        c
    }

    /// T-2: a second position-mutating compaction in one callback is `MultipleCompactions`, and the
    /// cache is left untouched (the first intent was only STAGED, never committed). Mutation-proof:
    /// dropping the `set_compaction` guard makes the second `keep` return `Ok`, failing the assert.
    #[test]
    fn test_t2_second_compaction_rejected() {
        let mut c = cache_f32(8);
        let before = c.k_buffer.as_slice::<f32>().to_vec();
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert_eq!(h.keep(&[0, 1, 2, 3]), Ok(()));
        assert_eq!(h.keep(&[0, 1]), Err(CacheOpError::MultipleCompactions));
        // dropped without commit → bytes untouched.
        drop(h);
        assert_eq!(c.k_buffer.as_slice::<f32>(), &before[..]);
        assert_eq!(c.current_pos(), 8);
    }

    /// T-10: a non-ascending / out-of-range / duplicate keep is `InvalidKeep` (the leaky-keep hole is
    /// closed before any mutation). Mutation-proof: removing `validate_keep` lets these stage as `Ok`.
    #[test]
    fn test_t10_invalid_keep_rejected() {
        let mut c = cache_f32(8);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert_eq!(h.keep(&[2, 1, 0]), Err(CacheOpError::InvalidKeep)); // descending
        assert_eq!(h.keep(&[0, 0, 1]), Err(CacheOpError::InvalidKeep)); // duplicate
        assert_eq!(h.keep(&[0, 1, 8]), Err(CacheOpError::InvalidKeep)); // out of range (resident=8)
    }

    /// T-8 + dormant geometry walls: `prune_channels` / `set_head_dim` / `project_rank` always reject
    /// with `GeometryImmutable`, and leave the cache untouched (they are pure `Err`, no staging).
    #[test]
    fn test_wall_stubs_geometry_immutable() {
        let mut c = cache_f32(8);
        let before = c.k_buffer.as_slice::<f32>().to_vec();
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert_eq!(
            h.prune_channels(&[0, 1]),
            Err(CacheOpError::GeometryImmutable)
        );
        assert_eq!(h.set_head_dim(2), Err(CacheOpError::GeometryImmutable));
        assert_eq!(h.project_rank(2), Err(CacheOpError::GeometryImmutable));
        drop(h);
        assert_eq!(c.k_buffer.as_slice::<f32>(), &before[..]);
    }

    /// c4: a HeadMajor F32 cache with a deterministic pattern, for the offload/recall round-trip.
    fn cache_hm_f32(resident: usize, heads: usize, dim: usize) -> KVCache {
        use crate::kv_cache_ops::KVLayout;
        let backend = Arc::new(CpuBackend::new());
        let cap = 32;
        let bytes = cap * heads * dim * std::mem::size_of::<f32>();
        let shape = Shape::new(vec![1, cap, heads, dim]);
        let mut c = KVCache::new(
            Tensor::new(
                shape.clone(),
                Arc::new(SharedBuffer::new(bytes, DType::F32)),
                backend.clone(),
            ),
            Tensor::new(
                shape,
                Arc::new(SharedBuffer::new(bytes, DType::F32)),
                backend,
            ),
            cap,
        )
        .with_layout(KVLayout::HeadMajor);
        c.set_current_pos(resident);
        for (i, x) in c.k_buffer.as_mut_slice::<f32>().iter_mut().enumerate() {
            *x = i as f32;
        }
        for (i, x) in c.v_buffer.as_mut_slice::<f32>().iter_mut().enumerate() {
            *x = -(i as f32);
        }
        c
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "argus_cache_handle_test_{}_{}_{}",
            tag,
            std::process::id(),
            nanos
        ));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    /// c4: `offload` then `recall` through the handle route to swap_handler::offload_prefix /
    /// recall_layer. Offload shrinks current_pos by the prefix; recall restores it AND the original
    /// prefix bytes. Mutation-proof: a broken offload/recall wiring leaves current_pos or the
    /// restored bytes wrong. Offload/recall are compaction-slot ops (a 2nd in one callback would be
    /// MultipleCompactions — covered by test_t2).
    #[test]
    fn test_offload_recall_round_trip_via_handle() {
        use crate::kv::swap_handler::SwapHandler;
        let (heads, dim, resident, prefix) = (2usize, 4usize, 10usize, 5usize);
        let dir = tmp_dir("offload_recall");
        let swap = SwapHandler::with_disk(0.5, dir.clone());

        let mut c = cache_hm_f32(resident, heads, dim);
        let cap = c.capacity();
        // Snapshot the original per-head prefix (first `prefix` positions of each head).
        let mut orig = Vec::<f32>::new();
        {
            let k = c.k_buffer.as_slice::<f32>();
            for h in 0..heads {
                let base = h * cap * dim;
                for pos in 0..prefix {
                    let off = base + pos * dim;
                    orig.extend_from_slice(&k[off..off + dim]);
                }
            }
        }

        // offload the prefix through the handle.
        let mut h = EngineCacheHandle::with_swap(&mut c, 0, 1, &swap);
        assert_eq!(h.offload(prefix), Ok(()));
        assert_eq!(h.commit().unwrap(), true);
        assert_eq!(
            c.current_pos(),
            resident - prefix,
            "offload pruned the prefix"
        );
        assert_eq!(swap.state.lock().unwrap().records.len(), 1);

        // recall it back through the handle.
        let mut h = EngineCacheHandle::with_swap(&mut c, 0, 1, &swap);
        assert_eq!(h.recall(), Ok(()));
        assert_eq!(h.commit().unwrap(), true);
        assert_eq!(
            c.current_pos(),
            resident,
            "recall restored the prefix length"
        );
        // restored prefix bytes match the original.
        let k = c.k_buffer.as_slice::<f32>();
        let mut got = Vec::<f32>::new();
        for head in 0..heads {
            let base = head * cap * dim;
            for pos in 0..prefix {
                let off = base + pos * dim;
                got.extend_from_slice(&k[off..off + dim]);
            }
        }
        assert_eq!(
            got, orig,
            "recalled prefix bytes match the offloaded original"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A staged keep, when committed, applies via `compact_keep_positions` + `set_current_pos` — the
    /// same executor the v2 plan executor uses. (The full streaming byte-identical gate lands with the
    /// driver in c3; this pins the commit path itself.)
    #[test]
    fn test_keep_commit_compacts() {
        let mut c = cache_f32(8);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert_eq!(h.keep(&[1, 3, 5, 7]), Ok(()));
        assert_eq!(h.commit().unwrap(), true);
        assert_eq!(c.current_pos(), 4);
        // kept rows are now contiguous at the front, in order.
        for (new_pos, &src) in [1usize, 3, 5, 7].iter().enumerate() {
            for head in 0..N_KV {
                let off = c.offset(new_pos, head);
                let k = c.k_buffer.as_slice::<f32>();
                for d in 0..HD {
                    assert_eq!(k[off + d], (src * 100 + head * 10 + d) as f32);
                }
            }
        }
    }

    /// P0-2: a keep committed through the handle records the FINAL keep-set into the keepset dump
    /// (in-memory capture), so a handle-driven eviction appears in dumps exactly as the v2
    /// the v2 plan executor path does. Mutation-proof: removing the `keepset_dump::record` call from
    /// `commit`'s Keep arm captures nothing for our fingerprint, failing the assert. Uses a unique
    /// `seq_len` (13) fingerprint + the capture serialization lock for determinism.
    #[test]
    fn test_keep_commit_records_keepset_dump() {
        use crate::kv::eviction::keepset_dump::{arm_capture, capture_test_lock, drain_capture};
        let _guard = capture_test_lock();

        let mut c = cache_f32(13); // unique pre-eviction seq_len fingerprint
        let keep = [1usize, 4, 9];
        // LayerWide capture replicates the one keep-list across all kv heads.
        let mine = |x: &crate::kv::eviction::keepset_dump::CapturedKeepSet| {
            x.seq_len == 13 && x.keep == vec![keep.to_vec(); N_KV]
        };

        arm_capture();
        {
            let mut h = EngineCacheHandle::new(&mut c, 0, 1);
            assert_eq!(h.keep(&keep), Ok(()));
            assert_eq!(h.commit().unwrap(), true);
        }
        let captured = drain_capture();
        assert_eq!(
            captured.iter().filter(|x| mine(x)).count(),
            1,
            "handle commit must record the keep-set into the dump capture"
        );
        // current_pos shrank to the keep count (the keep actually committed).
        assert_eq!(c.current_pos(), 3);
    }

    /// c10: the high-level `keep_top_k` op compiles the 3-partition set (T1) and stages it through the
    /// validated low-level `keep`, then commits to the expected surviving positions (h2o shape).
    /// Mutation-proof: a wrong compile or a skipped commit leaves current_pos / the survivor rows off.
    #[test]
    fn test_keep_top_k_high_level_op() {
        use argus_extension_api::KeepTopK;
        // current=8, prefix=2, recent=2 (recent_start=6), heavy=2 over [2..6); scores pick 3 and 5.
        let mut c = cache_f32(8);
        let scores = [0.0f32, 0.0, 1.0, 9.0, 2.0, 8.0, 0.0, 0.0];
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert_eq!(
            h.keep_top_k(
                KeepTopK {
                    current: 8,
                    prefix: 2,
                    recent: 2,
                    heavy: 2,
                },
                &|p| scores[p],
            ),
            Ok(())
        );
        assert_eq!(h.commit().unwrap(), true);
        // keep = [0,1] ++ {3,5} ++ [6,7]
        assert_eq!(c.current_pos(), 6);
        for (new_pos, &src) in [0usize, 1, 3, 5, 6, 7].iter().enumerate() {
            for head in 0..N_KV {
                let off = c.offset(new_pos, head);
                let k = c.k_buffer.as_slice::<f32>();
                for d in 0..HD {
                    assert_eq!(k[off + d], (src * 100 + head * 10 + d) as f32);
                }
            }
        }
    }

    /// c10: the high-level `keep_intersect_of` / `keep_union_of` combinators compose component
    /// keep-sets and commit to the intersection / union token count.
    #[test]
    fn test_keep_combinator_high_level_ops() {
        let mut c = cache_f32(8);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        // intersect([0,1,2,3,5], [1,3,5,7]) = [1,3,5]
        assert_eq!(
            h.keep_intersect_of(&[&[0, 1, 2, 3, 5], &[1, 3, 5, 7]]),
            Ok(())
        );
        assert_eq!(h.commit().unwrap(), true);
        assert_eq!(c.current_pos(), 3);
        // Pass2-TR3: verify WHICH positions survived (not just the count) — the survivors are the
        // intersection's source positions [1,3,5], compacted to the front in order. A wrong-but-equal-
        // cardinality intersect (e.g. [0,1,2]) would pass the count check but fail here.
        for (new_pos, &src) in [1usize, 3, 5].iter().enumerate() {
            for head in 0..N_KV {
                let off = c.offset(new_pos, head);
                let k = c.k_buffer.as_slice::<f32>();
                for d in 0..HD {
                    assert_eq!(k[off + d], (src * 100 + head * 10 + d) as f32);
                }
            }
        }

        let mut c = cache_f32(8);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        // union([0,2], [2,5,7]) = [0,2,5,7]
        assert_eq!(h.keep_union_of(&[&[0, 2], &[2, 5, 7]]), Ok(()));
        assert_eq!(h.commit().unwrap(), true);
        assert_eq!(c.current_pos(), 4);
    }

    /// F1: merge() eager-validates positions (the merge twin of T-10). An into/from outside
    /// [0, current_pos) is InvalidMerge BEFORE staging — closing the OOB panic / silent-corruption the
    /// CPU executor would otherwise hit. Mutation-proof: bytes + pos untouched after the reject.
    #[test]
    fn test_merge_out_of_range_rejected() {
        use argus_extension_api::{MergeAxis, WeightedMerge};
        let mut c = cache_f32(8);
        let before = c.k_buffer.as_slice::<f32>().to_vec();
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        // into == entry_pos (8) is out of range.
        let bad_into = [WeightedMerge {
            into: 8,
            into_weight: 0.5,
            from: vec![(1, 0.5)],
            apply_to: MergeAxis::Both,
        }];
        assert_eq!(h.merge(&bad_into), Err(CacheOpError::InvalidMerge));
        // from position out of range.
        let bad_from = [WeightedMerge {
            into: 0,
            into_weight: 0.5,
            from: vec![(40, 0.5)],
            apply_to: MergeAxis::Both,
        }];
        assert_eq!(h.merge(&bad_from), Err(CacheOpError::InvalidMerge));
        drop(h);
        assert_eq!(c.k_buffer.as_slice::<f32>(), &before[..]);
        assert_eq!(c.current_pos(), 8);
    }

    /// F1: offload()/recall() on a handle with NO swap backend reject eagerly (NoResidencyBackend)
    /// rather than staging and failing at commit — so a residency op never orphans an earlier
    /// byte-mutating op. Bytes + pos untouched.
    #[test]
    fn test_residency_without_backend_rejected() {
        let mut c = cache_f32(8);
        let before = c.k_buffer.as_slice::<f32>().to_vec();
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert_eq!(h.offload(2), Err(CacheOpError::NoResidencyBackend));
        assert_eq!(h.recall(), Err(CacheOpError::NoResidencyBackend));
        drop(h);
        assert_eq!(c.k_buffer.as_slice::<f32>(), &before[..]);
        assert_eq!(c.current_pos(), 8);
    }

    /// F1: a multi-op callback (merge + keep) commits BOTH in canonical order (merge in the
    /// pre-compaction frame, then compaction). Exercises the merge-commit arm + composition.
    #[test]
    fn test_merge_then_keep_commits_both() {
        use argus_extension_api::{MergeAxis, WeightedMerge};
        let mut c = cache_f32(8);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        // row 0 = 0.5*row0 + 0.5*row1 (both K and V); then drop position 1.
        assert_eq!(
            h.merge(&[WeightedMerge {
                into: 0,
                into_weight: 0.5,
                from: vec![(1, 0.5)],
                apply_to: MergeAxis::Both,
            }]),
            Ok(())
        );
        assert_eq!(h.keep(&[0, 2, 3, 4, 5, 6, 7]), Ok(()));
        assert_eq!(h.commit().unwrap(), true);
        assert_eq!(c.current_pos(), 7);
        // new pos 0 (the merged target): K = 0.5*(head*10+d) + 0.5*(100+head*10+d) = head*10+d+50.
        for head in 0..N_KV {
            let off = c.offset(0, head);
            let k = c.k_buffer.as_slice::<f32>();
            for d in 0..HD {
                assert_eq!(k[off + d], (head * 10 + d) as f32 + 50.0);
            }
        }
        // new pos 1 == original position 2 (unmerged, shifted down by the dropped pos 1).
        for head in 0..N_KV {
            let off = c.offset(1, head);
            let k = c.k_buffer.as_slice::<f32>();
            for d in 0..HD {
                assert_eq!(k[off + d], (2 * 100 + head * 10 + d) as f32);
            }
        }
    }

    /// F1: a re-encode to the cache's CURRENT format is a no-op that does not stage (commit reports
    /// no mutation); a real re-encode commits and flips the dtype; an infeasible q4_0 target (head_dim
    /// not a QK4_0 multiple) is eager-rejected.
    #[test]
    fn test_reencode_noop_and_feasibility() {
        use crate::buffer::DType;
        // no-op: f32 cache reencoded to "f32" → does not stage → commit returns mutated=false.
        let mut c = cache_f32(8);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert_eq!(h.reencode(FormatId("f32".into())), Ok(()));
        assert_eq!(h.commit().unwrap(), false);
        assert_eq!(c.kv_dtype(), DType::F32);

        // infeasible: HD=4 is not a multiple of QK4_0 (32) → q4_0 reencode eager-rejected.
        let mut c = cache_f32(8);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert!(matches!(
            h.reencode(FormatId("q4_0".into())),
            Err(CacheOpError::UnsupportedFormat(_))
        ));

        // real: f32 → f16 commits and flips the stored dtype.
        let mut c = cache_f32(8);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert_eq!(h.reencode(FormatId("f16".into())), Ok(()));
        assert_eq!(h.commit().unwrap(), true);
        assert_eq!(c.kv_dtype(), DType::F16);
    }

    /// Pass2-TX1: "re-encode to the CURRENT format" is the caller's LAST word — it must CANCEL an
    /// earlier staged re-encode (last-write-wins, T-5), not be silently overridden by it at commit.
    /// Mutation-proof: without the `self.reencode = None` on the no-op-to-current path, commit applies
    /// the stale f16 and the cache ends F16 (failing the F32 assert).
    #[test]
    fn test_reencode_last_write_to_current_cancels_staged() {
        let mut c = cache_f32(8);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        // Stage a real f32 -> f16, then "reset" by re-encoding to the current (f32) format.
        assert_eq!(h.reencode(FormatId("f16".into())), Ok(()));
        assert_eq!(h.reencode(FormatId("f32".into())), Ok(())); // no-op-to-current: cancels the f16.
        assert_eq!(h.commit().unwrap(), false); // nothing left staged -> no mutation.
        assert_eq!(c.kv_dtype(), DType::F32); // stayed f32 (the last write), not f16.
    }

    /// Pass2-TX2: a re-encode of an OPAQUE (non-typed-floor, e.g. U8) source is eager-rejected at the
    /// call site (UnsupportedFormat), not deferred to a commit-time failure — honoring the documented
    /// eager-validation contract for opaque sources. Mutation-proof: dropping the source-floor check
    /// lets it stage Ok, deferring the failure to commit.
    #[test]
    fn test_reencode_opaque_source_eager_rejected() {
        // A minimal U8 (opaque-floor) host cache: the reject fires before any buffer op, so the cache
        // only needs to report kv_dtype() == U8.
        let backend = Arc::new(CpuBackend::new());
        let bytes = N_KV * MAX_SEQ * HD; // 1 byte/elem for U8.
        let shape = Shape::new(vec![1, MAX_SEQ, N_KV, HD]);
        let mut c = KVCache::new(
            Tensor::new(
                shape.clone(),
                Arc::new(SharedBuffer::new(bytes, DType::U8)),
                backend.clone(),
            ),
            Tensor::new(
                shape,
                Arc::new(SharedBuffer::new(bytes, DType::U8)),
                backend,
            ),
            MAX_SEQ,
        );
        c.set_current_pos(4);
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert!(matches!(
            h.reencode(FormatId("f16".into())),
            Err(CacheOpError::UnsupportedFormat(_))
        ));
    }
}
