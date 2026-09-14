//! The ragged (per-head) KV geometry, pinned end to end: a per-head keep whose heads keep
//! different counts right-aligns every head on the longest one (`KVCache::head_start`), attention
//! reads each head from its own first resident slot, and a keep that reaches into a hole is
//! refused rather than trimmed. Every expectation here is stated against the handle-independent
//! naive oracle or a scalar re-derivation, never against the code under test.

use std::sync::Arc;

use argus_extension_api::{CacheHandle, CacheOpError, KeepSpec};
use half::f16;

use crate::aperturb::KeepSets;
use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::format::{AttnDims, KVCacheFormat};
use crate::kv::cache_handle::EngineCacheHandle;
use crate::kv::kv_cache::KVCache;
use crate::kv::naive_reference::{NaiveModel, assert_cache_matches};
use crate::kv::standard_format::StandardFormat;
use crate::kv_cache_ops::KVLayout;
use crate::layers::attention::{flash_attention_head, flash_attention_head_from};
use crate::memory::host::shared::SharedBuffer;
use crate::shape::Shape;
use crate::tensor::Tensor;

const MAX_SEQ: usize = 32;
const HD: usize = 4;
const N_KV: usize = 2;

/// `(pos, head, d)` → a value every dtype here stores exactly (integers below 2048 in f16).
fn k_val(pos: usize, head: usize, d: usize) -> f32 {
    (pos * 32 + head * 8 + d) as f32
}
fn v_val(pos: usize, head: usize, d: usize) -> f32 {
    k_val(pos, head, d) + 0.5
}

/// A HeadMajor cache with `resident` distinct tokens, f32 or f16.
fn head_major_cache(dtype: DType, resident: usize) -> KVCache {
    let backend = Arc::new(CpuBackend::new());
    let buf = || Arc::new(SharedBuffer::new(N_KV * MAX_SEQ * HD * dtype.size(), dtype));
    let shape = Shape::new(vec![1, N_KV, MAX_SEQ, HD]);
    let mut c = KVCache::new_with_geometry(
        Tensor::new(shape.clone(), buf(), backend.clone()),
        Tensor::new(shape, buf(), backend),
        MAX_SEQ,
        N_KV,
        HD,
        KVLayout::HeadMajor,
    );
    for pos in 0..resident {
        for head in 0..N_KV {
            let off = c.offset(pos, head);
            for d in 0..HD {
                match dtype {
                    DType::F32 => {
                        c.k_buffer.as_mut_slice::<f32>()[off + d] = k_val(pos, head, d);
                        c.v_buffer.as_mut_slice::<f32>()[off + d] = v_val(pos, head, d);
                    }
                    DType::F16 => {
                        c.k_buffer.as_mut_slice::<f16>()[off + d] =
                            f16::from_f32(k_val(pos, head, d));
                        c.v_buffer.as_mut_slice::<f16>()[off + d] =
                            f16::from_f32(v_val(pos, head, d));
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    c.set_current_pos(resident);
    c
}

fn read_k(c: &KVCache, pos: usize, head: usize, d: usize) -> f32 {
    let off = c.offset(pos, head) + d;
    match c.kv_dtype() {
        DType::F32 => c.k_buffer.as_slice::<f32>()[off],
        DType::F16 => c.k_buffer.as_slice::<f16>()[off].to_f32(),
        _ => unreachable!(),
    }
}

fn commit_per_head(c: &mut KVCache, heads: &[Vec<usize>]) {
    let borrowed: Vec<&[usize]> = heads.iter().map(|h| h.as_slice()).collect();
    let mut h = EngineCacheHandle::new(c, 0, 1);
    h.keep_per_head(&borrowed).expect("staged");
    assert!(h.commit().expect("committed"));
}

/// A per-head keep with unequal counts right-aligns every head on the longest one, and the bytes
/// land where the independent oracle says they do (f32 and f16). Mutation-proof: a commit that
/// left `write_start` at 0 (the old left-aligned per-head path) puts head 1's tokens at slots
/// `[0, 3)` and fails the value comparison at slot 2; forgetting `set_head_starts` fails the
/// geometry assertion.
#[test]
fn unequal_per_head_keep_right_aligns_and_matches_the_naive_oracle() {
    for dtype in [DType::F32, DType::F16] {
        let mut c = head_major_cache(dtype, 12);
        let keep = vec![vec![0, 1, 5, 9, 11], vec![0, 3, 4]];
        let expected =
            NaiveModel::capture(&c).expected_after(&[], &KeepSpec::PerHead(keep.clone()));
        assert_eq!(expected.head_start, vec![0, 2]);
        commit_per_head(&mut c, &keep);
        assert_cache_matches(&c, &expected, 0.0);
        assert!(c.is_ragged());
        assert_eq!(c.current_pos(), 5);
        assert_eq!(c.head_starts(), vec![0, 2]);
        assert_eq!(c.head_len(0), 5);
        assert_eq!(c.head_len(1), 3);
        assert_eq!(c.resident_positions(), 8);
        assert_eq!(c.resident_tokens(), 4);
        assert_eq!(c.memory_usage_bytes(), 8 * HD * dtype.size() * 2);
        // Head 1's three survivors sit at the END of its run: slots 2, 3, 4 hold tokens 0, 3, 4.
        assert_eq!(read_k(&c, 2, 1, 0), k_val(0, 1, 0));
        assert_eq!(read_k(&c, 3, 1, 0), k_val(3, 1, 0));
        assert_eq!(read_k(&c, 4, 1, 0), k_val(4, 1, 0));
    }
}

/// Equal counts are the uniform commit this always was: no head start, and byte-for-byte what a
/// layer-wide keep of the same list produces.
#[test]
fn equal_per_head_keep_stays_uniform_and_matches_the_layer_wide_keep() {
    let list = vec![0usize, 2, 5, 6, 9];
    let mut per_head = head_major_cache(DType::F32, 12);
    commit_per_head(&mut per_head, &[list.clone(), list.clone()]);
    assert!(!per_head.is_ragged());
    assert_eq!(per_head.head_starts(), vec![0, 0]);

    let mut layer_wide = head_major_cache(DType::F32, 12);
    let mut h = EngineCacheHandle::new(&mut layer_wide, 0, 1);
    h.keep(&list).expect("staged");
    h.commit().expect("committed");
    assert_eq!(per_head.current_pos(), layer_wide.current_pos());
    assert_eq!(
        per_head.k_buffer.as_slice::<f32>(),
        layer_wide.k_buffer.as_slice::<f32>()
    );
    assert_eq!(
        per_head.v_buffer.as_slice::<f32>(),
        layer_wide.v_buffer.as_slice::<f32>()
    );
}

/// The right-aligning gather never reads a run after it has been overwritten. The three shapes
/// that matter — every run moving up, every run moving down, a mix — plus a random sweep, each
/// checked against a gather done on a copy. Mutation-proof: flushing the up-moving prefix
/// first-to-last (the old order) clobbers `keep = [0, 1, 7]` at `write_start = 7`.
#[test]
fn right_aligning_compaction_flushes_runs_in_a_clobber_free_order() {
    fn check(keep: &[usize], write_start: usize) {
        let mut c = head_major_cache(DType::F32, 12);
        let before: Vec<Vec<f32>> = (0..12)
            .map(|p| (0..HD).map(|d| read_k(&c, p, 0, d)).collect())
            .collect();
        c.compact_keep_positions_for_head(0, keep, write_start)
            .expect("compacted");
        for (i, &src) in keep.iter().enumerate() {
            for d in 0..HD {
                assert_eq!(
                    read_k(&c, write_start + i, 0, d),
                    before[src][d],
                    "keep {keep:?} write_start {write_start}: slot {} should hold token {src}",
                    write_start + i
                );
            }
        }
    }
    check(&[0, 1, 7], 7); // all up
    check(&[2, 8], 1); // all down
    check(&[0, 9], 5); // up then down
    check(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2); // one run, up
    check(&[3, 4, 5, 6], 2); // one run, down
    let mut seed = 0x9E37u32;
    for _ in 0..200 {
        let mut keep = Vec::new();
        for p in 0..12 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            if (seed >> 16) & 1 == 1 {
                keep.push(p);
            }
        }
        if keep.is_empty() {
            continue;
        }
        check(&keep, 12 - keep.len());
    }
}

/// A keep that names a hole is `NotResident`, layer-wide or per-head, and stages nothing.
/// Mutation-proof: dropping the floor check lets `keep(&[0, 1, 2])` stage as `Ok`.
#[test]
fn keeps_that_reach_into_a_hole_are_refused() {
    let mut c = head_major_cache(DType::F32, 12);
    commit_per_head(&mut c, &[vec![0, 1, 5, 9, 11], vec![0, 3, 4]]);
    assert_eq!(c.head_starts(), vec![0, 2]);
    let before = c.k_buffer.as_slice::<f32>().to_vec();
    {
        let mut h = EngineCacheHandle::new(&mut c, 0, 1);
        assert_eq!(h.head_start(0), 0);
        assert_eq!(h.head_start(1), 2);
        assert_eq!(h.keep(&[0, 1, 2]), Err(CacheOpError::NotResident));
        assert_eq!(h.keep(&[1, 3]), Err(CacheOpError::NotResident));
        assert_eq!(
            h.keep_per_head(&[&[0, 1], &[1, 3]]),
            Err(CacheOpError::NotResident)
        );
        assert_eq!(h.keep_per_head(&[&[0, 1], &[2, 3]]), Ok(()));
        drop(h);
    }
    assert_eq!(c.k_buffer.as_slice::<f32>(), &before[..]);
    assert_eq!(c.current_pos(), 5);
    // A layer-wide keep inside every head's range restores the uniform shape.
    let expected = NaiveModel::capture(&c).expected_after(&[], &KeepSpec::LayerWide(vec![2, 4]));
    let mut h = EngineCacheHandle::new(&mut c, 0, 1);
    h.keep(&[2, 4]).expect("inside every head");
    h.commit().expect("committed");
    assert_cache_matches(&c, &expected, 0.0);
    assert!(!c.is_ragged());
    assert_eq!(c.current_pos(), 2);
}

/// An unequal keep on a store that cannot be ragged is `WrongContainer` before anything stages.
#[test]
fn unequal_keep_needs_a_head_major_typed_store() {
    let backend = Arc::new(CpuBackend::new());
    let buf = || Arc::new(SharedBuffer::new(N_KV * MAX_SEQ * HD * 4, DType::F32));
    let shape = Shape::new(vec![1, MAX_SEQ, N_KV, HD]);
    let mut seq_major = KVCache::new(
        Tensor::new(shape.clone(), buf(), backend.clone()),
        Tensor::new(shape, buf(), backend),
        MAX_SEQ,
    );
    seq_major.set_current_pos(8);
    assert!(!seq_major.supports_ragged());
    let mut h = EngineCacheHandle::new(&mut seq_major, 0, 1);
    assert_eq!(
        h.keep_per_head(&[&[0, 1, 2], &[0, 1]]),
        Err(CacheOpError::WrongContainer)
    );
    assert!(head_major_cache(DType::F16, 4).supports_ragged());
}

/// Scalar reference: softmax attention of query `q` over slots `[start, n)` of `kv_head`.
fn reference_row(
    c: &KVCache,
    q: &[f32],
    kv_head: usize,
    start: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (HD as f32).sqrt();
    let logits: Vec<f32> = (start..n)
        .map(|p| {
            (0..HD)
                .map(|d| q[d] * read_k(c, p, kv_head, d))
                .sum::<f32>()
                * scale
        })
        .collect();
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let z: f32 = e.iter().sum();
    let w: Vec<f32> = e.iter().map(|x| x / z).collect();
    let mut out = vec![0.0f32; HD];
    for (i, p) in (start..n).enumerate() {
        for d in 0..HD {
            let v = match c.kv_dtype() {
                DType::F32 => c.v_buffer.as_slice::<f32>()[c.offset(p, kv_head) + d],
                DType::F16 => c.v_buffer.as_slice::<f16>()[c.offset(p, kv_head) + d].to_f32(),
                _ => unreachable!(),
            };
            out[d] += w[i] * v;
        }
    }
    (out, w)
}

fn q_tensor(n_heads_q: usize) -> (Tensor, Vec<f32>) {
    let backend = Arc::new(CpuBackend::new());
    let mut q = vec![0.0f32; n_heads_q * HD];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i * 7 % 11) as f32 - 5.0) * 0.05;
    }
    let buf = Arc::new(SharedBuffer::new(q.len() * 4, DType::F32));
    let mut t = Tensor::new(Shape::new(vec![1, 1, n_heads_q, HD]), buf, backend);
    t.as_mut_slice::<f32>().copy_from_slice(&q);
    (t, q)
}

fn out_tensor(n_heads_q: usize) -> Tensor {
    let backend = Arc::new(CpuBackend::new());
    let buf = Arc::new(SharedBuffer::new(n_heads_q * HD * 4, DType::F32));
    Tensor::new(Shape::new(vec![1, 1, n_heads_q, HD]), buf, backend)
}

/// Decode attention over a ragged cache attends to `[head_start, n)` per KV head, writes `0.0`
/// into the hole columns of the score row, and equals the plain kernel when nothing is ragged.
/// GQA: query heads 0,1 → KV head 0 (start 0); 2,3 → KV head 1 (start 2). Mutation-proof: a kernel
/// that starts every head at 0 attends to head 1's holes (stale token bytes) and misses the
/// reference by far more than f32 rounding.
#[test]
fn ragged_decode_attention_skips_the_holes() {
    for dtype in [DType::F32, DType::F16] {
        let mut c = head_major_cache(dtype, 12);
        commit_per_head(&mut c, &[vec![0, 1, 5, 9, 11], vec![0, 3, 4]]);
        let n = c.current_pos();
        let n_heads_q = 4;
        let (q, q_data) = q_tensor(n_heads_q);
        let mut out = out_tensor(n_heads_q);
        let mut scores = vec![f32::NAN; n_heads_q * MAX_SEQ];
        let backend = CpuBackend::new();
        let starts = c.head_starts();
        let (k, v) = c.view();
        backend
            .attention_gen_ragged(
                &q,
                &k,
                &v,
                &mut out,
                n_heads_q,
                N_KV,
                HD,
                &starts,
                None,
                0,
                n,
                Some(&mut scores),
            )
            .expect("ragged attention");
        for h in 0..n_heads_q {
            let kv_h = h / 2;
            let (want, w) = reference_row(&c, &q_data[h * HD..(h + 1) * HD], kv_h, starts[kv_h], n);
            let got = &out.as_slice::<f32>()[h * HD..(h + 1) * HD];
            for d in 0..HD {
                assert!(
                    (got[d] - want[d]).abs() < 1e-4,
                    "{dtype:?} head {h} d {d}: {} vs {}",
                    got[d],
                    want[d]
                );
            }
            let row = &scores[h * MAX_SEQ..h * MAX_SEQ + n];
            for (p, s) in row.iter().enumerate() {
                let want = if p < starts[kv_h] {
                    0.0
                } else {
                    w[p - starts[kv_h]]
                };
                assert!(
                    (s - want).abs() < 1e-6,
                    "{dtype:?} head {h} slot {p}: score {s} vs {want}"
                );
            }
        }

        // Sliding window of 2 composes: head 0 sees [3, 5), head 1 sees [3, 5) too; column base 3.
        let mut out_w = out_tensor(n_heads_q);
        let mut scores_w = vec![f32::NAN; n_heads_q * MAX_SEQ];
        let win_start = n - 2;
        let clamped: Vec<usize> = starts.iter().map(|&s| s.max(win_start)).collect();
        backend
            .attention_gen_ragged(
                &q,
                &k,
                &v,
                &mut out_w,
                n_heads_q,
                N_KV,
                HD,
                &clamped,
                None,
                win_start,
                n,
                Some(&mut scores_w),
            )
            .expect("ragged attention with window");
        for h in 0..n_heads_q {
            let (want, w) = reference_row(&c, &q_data[h * HD..(h + 1) * HD], h / 2, win_start, n);
            let got = &out_w.as_slice::<f32>()[h * HD..(h + 1) * HD];
            for d in 0..HD {
                assert!((got[d] - want[d]).abs() < 1e-4);
            }
            assert!((scores_w[h * MAX_SEQ] - w[0]).abs() < 1e-6);
            assert!((scores_w[h * MAX_SEQ + 1] - w[1]).abs() < 1e-6);
        }
    }

    // Uniform cache: the ragged kernel is the plain kernel.
    let mut c = head_major_cache(DType::F32, 9);
    let n_heads_q = 4;
    let (q, _) = q_tensor(n_heads_q);
    let (mut a, mut b) = (out_tensor(n_heads_q), out_tensor(n_heads_q));
    let backend = CpuBackend::new();
    let (k, v) = c.view();
    backend
        .attention_gen(&q, &k, &v, &mut a, n_heads_q, N_KV, HD, 9, None)
        .expect("plain");
    backend
        .attention_gen_ragged(
            &q,
            &k,
            &v,
            &mut b,
            n_heads_q,
            N_KV,
            HD,
            &[0, 0],
            None,
            0,
            9,
            None,
        )
        .expect("ragged");
    for (x, y) in a.as_slice::<f32>().iter().zip(b.as_slice::<f32>()) {
        assert!((x - y).abs() < 1e-5);
    }
}

/// `StandardFormat::attention_into` routes a ragged cache to the ragged kernel — the seam every
/// CPU decode step goes through. Mutation-proof: dropping the `is_ragged` dispatch hands the
/// holes to the plain kernel and head 1's output moves.
#[test]
fn standard_format_decode_dispatches_a_ragged_cache_to_the_ragged_kernel() {
    let mut c = head_major_cache(DType::F32, 12);
    commit_per_head(&mut c, &[vec![0, 1, 5, 9, 11], vec![0, 3, 4]]);
    let starts = c.head_starts();
    let n = c.current_pos();
    let n_heads_q = 4;
    let (q, q_data) = q_tensor(n_heads_q);
    let fmt = StandardFormat::new(0, c);
    assert_eq!(fmt.resident_tokens(), 4);
    assert_eq!(fmt.current_pos(), 5);
    let mut out = out_tensor(n_heads_q);
    let backend = CpuBackend::new();
    fmt.attention_into(
        &q,
        &backend,
        &mut out,
        AttnDims {
            n_heads_q,
            window: None,
        },
        None,
        None,
    )
    .expect("attention_into");
    let c = head_major_cache(DType::F32, 12);
    // Rebuild the reference values from the ORIGINAL tokens the survivors came from.
    let survivors = [vec![0usize, 1, 5, 9, 11], vec![0usize, 3, 4]];
    for h in 0..n_heads_q {
        let kv_h = h / 2;
        let qv = &q_data[h * HD..(h + 1) * HD];
        let scale = 1.0 / (HD as f32).sqrt();
        let logits: Vec<f32> = survivors[kv_h]
            .iter()
            .map(|&p| (0..HD).map(|d| qv[d] * k_val(p, kv_h, d)).sum::<f32>() * scale)
            .collect();
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
        let z: f32 = e.iter().sum();
        let got = &out.as_slice::<f32>()[h * HD..(h + 1) * HD];
        for d in 0..HD {
            let want: f32 = survivors[kv_h]
                .iter()
                .zip(&e)
                .map(|(&p, w)| w / z * v_val(p, kv_h, d))
                .sum();
            assert!(
                (got[d] - want).abs() < 1e-4,
                "head {h} d {d}: {} vs {want}",
                got[d]
            );
        }
    }
    let _ = (starts, n, c);
}

/// The prefill kernel over a head resident from `kv_start` equals the plain kernel over the same
/// keys with the hole cut off — the definition of a hole, stated as an identity.
#[test]
fn ragged_prefill_head_equals_the_plain_kernel_on_the_resident_run() {
    let kv_len = 14;
    let q_len = 3;
    let kv_start = 5;
    let q_start_pos = kv_len - q_len;
    let mut k = vec![0.0f32; kv_len * HD];
    let mut v = vec![0.0f32; kv_len * HD];
    for (i, (kk, vv)) in k.iter_mut().zip(v.iter_mut()).enumerate() {
        *kk = ((i * 13 % 17) as f32 - 8.0) * 0.07;
        *vv = ((i * 5 % 13) as f32 - 6.0) * 0.09;
    }
    let q: Vec<f32> = (0..q_len * HD)
        .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.1)
        .collect();
    let mut ragged = vec![0.0f32; q_len * HD];
    flash_attention_head_from(
        &q,
        HD,
        &k,
        HD,
        &v,
        HD,
        &mut ragged,
        HD,
        q_len,
        kv_len,
        HD,
        q_start_pos,
        2,
        4,
        None,
        kv_start,
    );
    let mut plain = vec![0.0f32; q_len * HD];
    flash_attention_head(
        &q,
        HD,
        &k[kv_start * HD..],
        HD,
        &v[kv_start * HD..],
        HD,
        &mut plain,
        HD,
        q_len,
        kv_len - kv_start,
        HD,
        q_start_pos - kv_start,
        2,
        4,
        None,
    );
    for (a, b) in ragged.iter().zip(&plain) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
    // And it is NOT the plain kernel over the full run (the hole carries attention mass).
    let mut full = vec![0.0f32; q_len * HD];
    flash_attention_head(
        &q,
        HD,
        &k,
        HD,
        &v,
        HD,
        &mut full,
        HD,
        q_len,
        kv_len,
        HD,
        q_start_pos,
        2,
        4,
        None,
    );
    assert!(ragged.iter().zip(&full).any(|(a, b)| (a - b).abs() > 1e-3));
}

/// The QCF reference on a ragged cache is what is resident, and a candidate below a head's first
/// resident slot is a `Hole` error rather than a score over stale bytes.
#[test]
fn aperturb_reference_is_the_resident_set_and_holes_are_rejected() {
    let starts = [[0usize, 2], [1usize, 0]];
    let base = KeepSets::resident(2, 2, 5, |l, h| starts[l][h]);
    assert_eq!(base.head(0, 0), &[0, 1, 2, 3, 4]);
    assert_eq!(base.head(0, 1), &[2, 3, 4]);
    assert_eq!(base.head(1, 0), &[1, 2, 3, 4]);
    assert_eq!(base.head(1, 1), &[0, 1, 2, 3, 4]);
    assert!(base.is_ragged());
    assert_eq!(
        KeepSets::resident(2, 2, 5, |_, _| 0),
        KeepSets::identity(2, 2, 5)
    );

    let mut ok = KeepSets::with_capacity(2, 2, 0);
    ok.push(0, 0, &[0, 4]).unwrap();
    ok.push(0, 1, &[2, 4]).unwrap();
    ok.push(1, 0, &[1, 4]).unwrap();
    ok.push(1, 1, &[4]).unwrap();
    assert!(ok.validate_within(&base).is_ok());

    let mut bad = KeepSets::with_capacity(2, 2, 0);
    bad.push(0, 0, &[0, 4]).unwrap();
    bad.push(0, 1, &[1, 4]).unwrap(); // slot 1 is head 1's hole in layer 0
    bad.push(1, 0, &[1, 4]).unwrap();
    bad.push(1, 1, &[4]).unwrap();
    let err = bad.validate_within(&base).unwrap_err();
    assert!(
        matches!(
            err,
            crate::aperturb::KeepError::Hole {
                layer: 0,
                kv_head: 1,
                pos: 1,
                start: 2
            }
        ),
        "{err}"
    );
}

/// A per-head keep applied on a caller-chosen cursor (the selector's model-wide one) right-aligns
/// every head on that cursor: the bytes are the oracle's gather shifted up by the extra holes, the
/// head starts grow by the same amount, and a cursor below the longest head is refused.
#[test]
fn a_shared_cursor_right_aligns_every_head_on_it() {
    let keep = vec![vec![0usize, 1, 5, 9, 11], vec![0usize, 3, 4]];
    let mut c = head_major_cache(DType::F32, 12);
    crate::kv::cache_handle::apply_per_head_keep_at(&mut c, 0, 1, &keep, 8).expect("applied");
    assert_eq!(c.current_pos(), 8);
    assert_eq!(c.head_starts(), vec![3, 5]);
    assert_eq!(c.resident_positions(), 8);
    for (h, list) in keep.iter().enumerate() {
        let start = c.head_start(h);
        for (i, &src) in list.iter().enumerate() {
            for d in 0..HD {
                assert_eq!(read_k(&c, start + i, h, d), k_val(src, h, d));
            }
        }
    }
    // Below the longest head, or above the frame: refused, nothing moved.
    let mut c = head_major_cache(DType::F32, 12);
    let before = c.k_buffer.as_slice::<f32>().to_vec();
    assert!(crate::kv::cache_handle::apply_per_head_keep_at(&mut c, 0, 1, &keep, 4).is_err());
    assert!(crate::kv::cache_handle::apply_per_head_keep_at(&mut c, 0, 1, &keep, 13).is_err());
    assert_eq!(c.k_buffer.as_slice::<f32>(), &before[..]);
    assert_eq!(c.current_pos(), 12);
}
