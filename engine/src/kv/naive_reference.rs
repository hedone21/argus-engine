//! R-1/R-2 oracle: a handle-INDEPENDENT naive reference for KV mutations.
//!
//! Given a SOURCE cache (entry frame) + a keep-set ([`KeepSpec`]) + weighted merges, the reference
//! computes the expected post-op K/V VALUES by gathering / recomputing into a FRESH model the obvious
//! way — sharing NO code with `compact_keep_positions` / `compact_keep_positions_for_head` /
//! `apply_weighted_merges` / [`EngineCacheHandle`](crate::kv::cache_handle::EngineCacheHandle) (the
//! mutation logic under test). That independence is what makes the byte/value-identity gates
//! non-tautological once the v2 `execute_kv_plan` reference is deleted (Phase 2).
//!
//! Comparison is at the VALUE level (dequantized f32). Reading the SOURCE (and the committed result)
//! through the dequant codec is fine — the codec is not the mutation logic. The properties that hold:
//! a keep is a pure byte-preserving gather (exact for EVERY dtype, including q4_0 — the kept rows are
//! byte-copied, never re-quantized); a single non-chained merge recomputes a weighted sum then
//! re-quantizes (exact for f32, within a dtype tolerance for f16/q4_0); a widening reencode (f16→f32)
//! preserves values exactly.

use argus_extension_api::{KeepSpec, MergeAxis, WeightedMerge};

use crate::kv::dequant::{dequantize_k, dequantize_v};
use crate::kv::kv_cache::KVCache;

/// A pure f32 model of a cache's resident K/V, `[pos][kv_head][d]`.
pub(crate) struct NaiveModel {
    /// `k[pos][head][d]`.
    pub k: Vec<Vec<Vec<f32>>>,
    /// `v[pos][head][d]`.
    pub v: Vec<Vec<Vec<f32>>>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl NaiveModel {
    /// Capture the resident `[0, current_pos)` region of `cache` as f32 (via the dequant codec — this
    /// reads the INPUT, independent of the mutation logic under test).
    pub(crate) fn capture(cache: &KVCache) -> Self {
        let n = cache.current_pos();
        let n_kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();
        let mut k = vec![vec![vec![0.0f32; head_dim]; n_kv_heads]; n];
        let mut v = vec![vec![vec![0.0f32; head_dim]; n_kv_heads]; n];
        for (pos, (kp, vp)) in k.iter_mut().zip(v.iter_mut()).enumerate() {
            for (head, (kh, vh)) in kp.iter_mut().zip(vp.iter_mut()).enumerate() {
                dequantize_k(cache, pos, head, head_dim, kh);
                dequantize_v(cache, pos, head, head_dim, vh);
            }
        }
        Self {
            k,
            v,
            n_kv_heads,
            head_dim,
        }
    }

    /// Independently recompute weighted merges in the PRE-compaction frame, on a SNAPSHOT of the
    /// pre-merge rows (`from` rows are read from the snapshot, not from already-merged state). This
    /// matches `apply_weighted_merges` for the non-chained merges the gates use (no `into` of one
    /// merge is a `from`/`into` of another). Per the `apply_to` axis.
    pub(crate) fn apply_merges(&mut self, merges: &[WeightedMerge]) {
        let src_k = self.k.clone();
        let src_v = self.v.clone();
        for m in merges {
            let do_k = matches!(m.apply_to, MergeAxis::Both | MergeAxis::KeyOnly);
            let do_v = matches!(m.apply_to, MergeAxis::Both | MergeAxis::ValueOnly);
            for head in 0..self.n_kv_heads {
                for d in 0..self.head_dim {
                    if do_k {
                        let mut acc = m.into_weight * src_k[m.into][head][d];
                        for &(from, w) in &m.from {
                            acc += w * src_k[from][head][d];
                        }
                        self.k[m.into][head][d] = acc;
                    }
                    if do_v {
                        let mut acc = m.into_weight * src_v[m.into][head][d];
                        for &(from, w) in &m.from {
                            acc += w * src_v[from][head][d];
                        }
                        self.v[m.into][head][d] = acc;
                    }
                }
            }
        }
    }

    /// Gather the kept positions to the front (the independent compaction). Returns the expected
    /// post-compaction model. For [`KeepSpec::LayerWide`] every head gathers the same list; for
    /// [`KeepSpec::PerHead`] each head gathers its own (all equal length, the engine's single
    /// `current_pos` invariant).
    pub(crate) fn gather(&self, keep: &KeepSpec) -> NaiveModel {
        let per_head: Vec<Vec<usize>> = match keep {
            KeepSpec::LayerWide(list) => vec![list.clone(); self.n_kv_heads],
            KeepSpec::PerHead(heads) => heads.clone(),
        };
        let new_pos = per_head.first().map_or(0, |h| h.len());
        let mut k = vec![vec![vec![0.0f32; self.head_dim]; self.n_kv_heads]; new_pos];
        let mut v = vec![vec![vec![0.0f32; self.head_dim]; self.n_kv_heads]; new_pos];
        for (head, keeps) in per_head.iter().enumerate() {
            for (new_i, &src) in keeps.iter().enumerate() {
                k[new_i][head].copy_from_slice(&self.k[src][head]);
                v[new_i][head].copy_from_slice(&self.v[src][head]);
            }
        }
        NaiveModel {
            k,
            v,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
        }
    }

    /// The expected post-op model for `merges` (pre-compaction) then `keep` — the order the engine
    /// commits in (merge in the entry frame, then the single compaction).
    pub(crate) fn expected_after(&self, merges: &[WeightedMerge], keep: &KeepSpec) -> NaiveModel {
        let mut merged = NaiveModel {
            k: self.k.clone(),
            v: self.v.clone(),
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
        };
        merged.apply_merges(merges);
        merged.gather(keep)
    }
}

/// Assert that `cache`'s resident region holds the values of `expected` (dequantized, `[pos][head][d]`)
/// within `tol` (use `0.0` for the byte-preserving keep / exact-f32 cases; a small tolerance for a
/// lossy-dtype merge). Independent of the mutation logic: it re-reads `cache` via the dequant codec
/// and compares to the independently-computed `expected`.
pub(crate) fn assert_cache_matches(cache: &KVCache, expected: &NaiveModel, tol: f32) {
    assert_eq!(
        cache.current_pos(),
        expected.k.len(),
        "naive reference: current_pos mismatch (expected {} survivors)",
        expected.k.len()
    );
    let head_dim = cache.head_dim();
    let mut row = vec![0.0f32; head_dim];
    for pos in 0..expected.k.len() {
        for head in 0..expected.n_kv_heads {
            dequantize_k(cache, pos, head, head_dim, &mut row);
            for d in 0..head_dim {
                let (got, want) = (row[d], expected.k[pos][head][d]);
                assert!(
                    (got - want).abs() <= tol,
                    "K mismatch at pos {pos} head {head} d {d}: got {got}, want {want} (tol {tol})"
                );
            }
            dequantize_v(cache, pos, head, head_dim, &mut row);
            for d in 0..head_dim {
                let (got, want) = (row[d], expected.v[pos][head][d]);
                assert!(
                    (got - want).abs() <= tol,
                    "V mismatch at pos {pos} head {head} d {d}: got {got}, want {want} (tol {tol})"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::kv::cache_handle::EngineCacheHandle;
    use crate::kv_cache_ops::KVLayout;
    use crate::memory::host::shared::SharedBuffer;
    use crate::quant::{BlockQ4_0, QK4_0};
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use argus_extension_api::{CacheHandle, FormatId};
    use half::f16;
    use std::sync::Arc;

    const HD: usize = QK4_0; // 32 — one q4_0 block per head.
    const KV: usize = 2;
    const MAXS: usize = 16;
    const RESIDENT: usize = 8;

    /// Small, q4-friendly analytic pattern so a keep (byte-copy) is exact for every dtype and a merge
    /// recompute stays within a tight q4 tolerance.
    fn pat(pos: usize, head: usize, d: usize, salt: f32) -> f32 {
        salt + pos as f32 * 0.05 + head as f32 * 0.5 + d as f32 * 0.01
    }

    fn build(dtype: DType, layout: KVLayout) -> KVCache {
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, MAXS, KV, HD]);
        let total = MAXS * KV * HD;
        let bytes = match dtype {
            DType::F32 => total * 4,
            DType::F16 => total * 2,
            DType::Q4_0 => (total / QK4_0) * std::mem::size_of::<BlockQ4_0>(),
            _ => unreachable!(),
        };
        let mut c = KVCache::new(
            Tensor::new(
                sh.clone(),
                Arc::new(SharedBuffer::new(bytes, dtype)),
                be.clone(),
            ),
            Tensor::new(sh, Arc::new(SharedBuffer::new(bytes, dtype)), be),
            MAXS,
        )
        .with_layout(layout);
        c.set_current_pos(RESIDENT);
        for pos in 0..RESIDENT {
            for head in 0..KV {
                let off = c.offset(pos, head);
                let mut rk = [0.0f32; HD];
                let mut rv = [0.0f32; HD];
                for d in 0..HD {
                    rk[d] = pat(pos, head, d, 0.5);
                    rv[d] = pat(pos, head, d, 1.3);
                }
                write_row(&mut c, off, dtype, &rk, true);
                write_row(&mut c, off, dtype, &rv, false);
            }
        }
        c
    }

    fn write_row(c: &mut KVCache, off: usize, dtype: DType, row: &[f32], is_k: bool) {
        match dtype {
            DType::F32 => {
                let b = if is_k {
                    c.k_buffer.as_mut_slice::<f32>()
                } else {
                    c.v_buffer.as_mut_slice::<f32>()
                };
                b[off..off + HD].copy_from_slice(row);
            }
            DType::F16 => {
                let b = if is_k {
                    c.k_buffer.as_mut_slice::<f16>()
                } else {
                    c.v_buffer.as_mut_slice::<f16>()
                };
                for d in 0..HD {
                    b[off + d] = f16::from_f32(row[d]);
                }
            }
            DType::Q4_0 => {
                let bo = off / QK4_0;
                let b = if is_k {
                    c.k_buffer.as_mut_slice::<BlockQ4_0>()
                } else {
                    c.v_buffer.as_mut_slice::<BlockQ4_0>()
                };
                let mut blk = [0.0f32; QK4_0];
                blk.copy_from_slice(&row[..QK4_0]);
                b[bo] = BlockQ4_0::quantize(&blk);
            }
            _ => unreachable!(),
        }
    }

    /// R-2: a LayerWide keep through the handle equals the naive gather, VALUE-exact for f32/f16/q4_0
    /// (a keep is a byte-preserving gather — no requantization — so it is exact even on q4_0).
    #[test]
    fn keep_matches_naive_all_dtypes() {
        for dtype in [DType::F32, DType::F16, DType::Q4_0] {
            let mut cache = build(dtype, KVLayout::SeqMajor);
            let src = NaiveModel::capture(&cache);
            let keep = KeepSpec::LayerWide(vec![1usize, 3, 4, 6]);
            let expected = src.expected_after(&[], &keep);
            {
                let mut h = EngineCacheHandle::new(&mut cache, 0, 1);
                let KeepSpec::LayerWide(ref k) = keep else {
                    unreachable!()
                };
                h.keep(k).unwrap();
                assert!(h.commit().unwrap());
            }
            assert_cache_matches(&cache, &expected, 0.0);
        }
    }

    /// R-2: a per-head keep on a HeadMajor cache equals the naive per-head gather (byte-copy, exact).
    #[test]
    fn per_head_keep_matches_naive() {
        for dtype in [DType::F32, DType::F16, DType::Q4_0] {
            let mut cache = build(dtype, KVLayout::HeadMajor);
            let src = NaiveModel::capture(&cache);
            let heads = vec![vec![0usize, 2, 4], vec![1usize, 3, 5]];
            let keep = KeepSpec::PerHead(heads.clone());
            let expected = src.expected_after(&[], &keep);
            {
                let mut h = EngineCacheHandle::new(&mut cache, 0, 1);
                let refs: Vec<&[usize]> = heads.iter().map(|x| x.as_slice()).collect();
                h.keep_per_head(&refs).unwrap();
                assert!(h.commit().unwrap());
            }
            assert_cache_matches(&cache, &expected, 0.0);
        }
    }

    /// R-2: a merge + keep through the handle equals the naive weighted-sum-then-gather. Exact for
    /// f32; within a dtype tolerance for the requantized lossy dtypes (only the merged `into` row is
    /// re-encoded — kept rows stay byte-exact).
    #[test]
    fn merge_then_keep_matches_naive() {
        use argus_extension_api::{MergeAxis, WeightedMerge};
        // f32 is exact (tol 0) — that pins the merge MATH. f16/q4_0 differ only by the requantization
        // of the merged `into` block (kept rows stay byte-exact), so the tolerance is the dtype's
        // quant step over the merged block range (~7.5% of ~2.0 for 4-bit q4_0).
        for (dtype, tol) in [
            (DType::F32, 0.0f32),
            (DType::F16, 0.01),
            (DType::Q4_0, 0.25),
        ] {
            let mut cache = build(dtype, KVLayout::SeqMajor);
            let src = NaiveModel::capture(&cache);
            let merges = vec![WeightedMerge {
                into: 0,
                into_weight: 0.5,
                from: vec![(5, 0.5)],
                apply_to: MergeAxis::Both,
            }];
            let keep = KeepSpec::LayerWide((0..RESIDENT).filter(|&p| p != 5).collect());
            let expected = src.expected_after(&merges, &keep);
            {
                let mut h = EngineCacheHandle::new(&mut cache, 0, 1);
                h.merge(&merges).unwrap();
                let KeepSpec::LayerWide(ref k) = keep else {
                    unreachable!()
                };
                h.keep(k).unwrap();
                assert!(h.commit().unwrap());
            }
            assert_cache_matches(&cache, &expected, tol);
        }
    }

    /// R-2: a widening reencode (f16 -> f32) preserves values exactly (the naive model is unchanged;
    /// only the stored dtype flips). Confirms the value-level oracle covers the reencode op.
    #[test]
    fn reencode_f16_to_f32_matches_naive() {
        let mut cache = build(DType::F16, KVLayout::SeqMajor);
        let expected = NaiveModel::capture(&cache); // values pre-reencode == values post (widening)
        {
            let mut h = EngineCacheHandle::new(&mut cache, 0, 1);
            h.reencode(FormatId("f32".into())).unwrap();
            assert!(h.commit().unwrap());
        }
        assert_eq!(
            cache.kv_dtype(),
            DType::F32,
            "reencode flipped the stored dtype"
        );
        assert_cache_matches(&cache, &expected, 0.0);
    }
}
