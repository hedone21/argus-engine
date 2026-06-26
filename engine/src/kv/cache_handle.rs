//! HYBRID v3 — engine-side transaction engine for the imperative [`CacheHandle`] ABI.
//!
//! [`EngineCacheHandle`] is the engine's implementation of the additive
//! [`argus_extension_api::CacheHandle`] surface (the M4 callback class, the imperative sibling of the
//! plan-returning `KVCacheStage`). It is the *transaction* layer that makes imperative mutation sound:
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
//!   mutation — closing the historical leaky-keep hole in `execute_kv_plan`.
//!
//! Every op routes to an EXISTING executor (`compact_keep_positions` / `compact_keep_positions_for_head`
//! / `apply_weighted_merges` / `apply_format_plan`), so a handle-driven keep is byte-identical to the
//! same keep applied through `execute_kv_plan` (the s1 gate).

use crate::kv::kv_cache::KVCache;
use crate::kv_cache_ops::KVLayout;
use anyhow::Result;
use argus_extension_api::{
    CacheHandle, CacheOpError, FormatId, KVFormatPlan, TensorHandle, TensorKind, WeightedMerge,
};

/// The single staged position-mutating compaction slot (T-2: at most one per callback).
enum Compaction {
    /// LayerWide keep — all heads keep the same ascending positions.
    Keep(Vec<usize>),
    /// Per-head keep — `[n_kv_heads][keep]`, each ascending, all equal length (HeadMajor only).
    KeepPerHead(Vec<Vec<usize>>),
}

/// Engine-side [`CacheHandle`] over **one layer's** `&mut KVCache`. Stages position-mutating intents
/// and applies them once in [`commit`](Self::commit); position-preserving ops (`reencode`) take effect
/// at commit too. See the module docs for the transaction invariants.
pub struct EngineCacheHandle<'a> {
    cache: &'a mut KVCache,
    layer_idx: usize,
    n_layers: usize,
    /// Entry-frame token count (T-3): the frame every read + every keep validation observes.
    entry_pos: usize,
    /// Staged weighted merges (applied in the pre-compaction frame at commit).
    merges: Option<Vec<WeightedMerge>>,
    /// The single staged compaction (T-2).
    compaction: Option<Compaction>,
    /// Staged re-encode target (position-preserving; last write wins, T-5).
    reencode: Option<FormatId>,
    /// Whether `commit` mutated any bytes (the driver uses this for coalesced plan invalidation, T-6).
    mutated: bool,
}

impl<'a> EngineCacheHandle<'a> {
    /// Build a handle over one layer's cache for a mutation callback.
    pub fn new(cache: &'a mut KVCache, layer_idx: usize, n_layers: usize) -> Self {
        let entry_pos = cache.current_pos();
        Self {
            cache,
            layer_idx,
            n_layers,
            entry_pos,
            merges: None,
            compaction: None,
            reencode: None,
            mutated: false,
        }
    }

    /// Validate a keep-list against the entry frame: ascending, unique (strictly increasing), and
    /// every index in `[0, entry_pos)` (T-10). Returns [`CacheOpError::InvalidKeep`] otherwise.
    fn validate_keep(&self, keep: &[usize]) -> Result<(), CacheOpError> {
        let mut prev: Option<usize> = None;
        for &k in keep {
            if k >= self.entry_pos {
                return Err(CacheOpError::InvalidKeep);
            }
            if let Some(p) = prev {
                if k <= p {
                    return Err(CacheOpError::InvalidKeep);
                }
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
    /// frame) → compaction (the single renumber, T-1). Every staged op was validated eagerly, so this
    /// does not reject; a defensive executor `Err` is surfaced as `anyhow`.
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
        if let Some(merges) = self.merges.take() {
            if !merges.is_empty() {
                crate::kv::standard_format::apply_weighted_merges(self.cache, &merges);
                self.mutated = true;
            }
        }
        match self.compaction.take() {
            Some(Compaction::Keep(keep)) => {
                self.cache.compact_keep_positions(&keep, 0)?;
                self.cache.set_current_pos(keep.len());
                self.mutated = true;
            }
            Some(Compaction::KeepPerHead(heads)) => {
                let new_pos = heads.first().map_or(0, |h| h.len());
                for (kv_head, keep) in heads.iter().enumerate() {
                    self.cache.compact_keep_positions_for_head(kv_head, keep, 0)?;
                }
                self.cache.set_current_pos(new_pos);
                self.mutated = true;
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
            return Err(CacheOpError::MultipleCompactions);
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
        if !matches!(target.0.as_str(), "f32" | "f16" | "q4_0") {
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

    fn offload(&mut self, _prefix_len: usize) -> Result<(), CacheOpError> {
        // Residency ops are wired through swap_handler in the next commit (s1 c4).
        Err(CacheOpError::WrongContainer)
    }
    fn recall(&mut self) -> Result<(), CacheOpError> {
        Err(CacheOpError::WrongContainer)
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
        assert_eq!(h.prune_channels(&[0, 1]), Err(CacheOpError::GeometryImmutable));
        assert_eq!(h.set_head_dim(2), Err(CacheOpError::GeometryImmutable));
        assert_eq!(h.project_rank(2), Err(CacheOpError::GeometryImmutable));
        drop(h);
        assert_eq!(c.k_buffer.as_slice::<f32>(), &before[..]);
    }

    /// A staged keep, when committed, applies via `compact_keep_positions` + `set_current_pos` — the
    /// same executor `execute_kv_plan` uses. (The full streaming byte-identical gate lands with the
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
}
