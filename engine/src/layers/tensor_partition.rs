use crate::backend::Backend;
use crate::buffer::DType;
use crate::shape::Shape;
use crate::tensor::Tensor;
use anyhow::{Result, ensure};
use std::sync::Arc;

/// GPU-only fast-path threshold for `gpu_ratio`.
///
/// When `gpu_ratio >= GPU_ONLY_THRESHOLD`, the partition activation path is
/// skipped entirely: `partition_ctx` stays `None`, and the forward pass uses
/// the dense full-weight matmul on GPU. This avoids the ratio-independent
/// constant overhead of the partition path (host staging `read_buffer`,
/// CPU matmul kick-off for a clamped 128-row slice, and GPU<->host merge).
///
/// The rationale: the split is clamped to leave at least one quantum (128 FFN
/// rows, one Q head) on the CPU. A caller passing `ratio=0.999` would still end
/// up with that CPU share, forcing the partition dispatch path every token even
/// though the split is effectively GPU-only. This threshold makes that corner
/// case behave as the user intends.
///
/// 0.995 is chosen empirically:
///   - For Llama 3.2 1B `ffn_hidden=8192`, 0.995 → CPU gets at most 128 rows
///     (1.6% of output), which is already the clamp minimum. Anything above
///     0.995 is structurally equivalent to "clamp min CPU rows" for any sane
///     FFN size, so skipping the partition path is information-preserving.
///   - 1.0 is considered GPU-only by design (`generate.rs` CLI gate is
///     `tensor_partition > 0.0 && tensor_partition < 1.0`).
pub const GPU_ONLY_THRESHOLD: f32 = 0.995;

/// Returns true when the given ratio should take the GPU-only fast path
/// (no partition context installed, no per-token host staging).
pub fn is_gpu_only_ratio(gpu_ratio: f32) -> bool {
    gpu_ratio >= GPU_ONLY_THRESHOLD
}

/// Fan a partition dispatch mode out over every layer slot (AB-4, §5.5.3).
///
/// Extracted from `TransformerModel::prepare_tensor_partition` so the static
/// CLI path (`session/init.rs`) and the runtime `SetPartitionRatio` directive
/// path (`PartitionStage`) share a single fan-out implementation — INV-120
/// generation bookkeeping then has a single source of truth.
///
/// - `gpu_ratio >= GPU_ONLY_THRESHOLD` → `LayerDispatch::Full` fan-out, returns 0
///   (GPU-only fast path: `partition_ctx` stays cleared, dense GPU matmul).
/// - otherwise → `LayerDispatch::Partition([gpu, cpu])` fan-out, returns
///   `slots.len() * 3` (gate/up/down split per layer — in place, no copy).
///
/// `LayerSlot::apply_dispatch(&self)` performs the ArcSwap RCU install and the
/// `ratio_generation` bump (INV-120); this function only chooses the dispatch
/// mode and iterates.
pub fn apply_partition_dispatch(
    slots: &[Arc<crate::models::weights::LayerSlot>],
    gpu_ratio: f32,
    hw: &crate::hardware::Hardware,
) -> Result<usize> {
    use crate::format::weight_format::{LayerDispatch, PartitionShare, WeightFormat};
    use argus_extension_api::DeviceTarget;

    if is_gpu_only_ratio(gpu_ratio) {
        for slot in slots {
            slot.apply_dispatch(LayerDispatch::Full, hw)?;
        }
        return Ok(0);
    }

    let specs = vec![
        PartitionShare {
            share: gpu_ratio,
            hardware: DeviceTarget::Gpu,
        },
        PartitionShare {
            share: 1.0 - gpu_ratio,
            hardware: DeviceTarget::Cpu,
        },
    ];
    let mut count = 0;
    for slot in slots {
        slot.apply_dispatch(LayerDispatch::Partition(specs.clone()), hw)?;
        count += 3;
    }
    Ok(count)
}

/// Per-layer tensor-partition context (ticket 021).
///
/// The weights are never copied or re-sliced: the GPU reads its share of the original
/// buffers through kernel offsets / row strides, the CPU through host pointers into the same
/// (ALLOC_HOST_PTR, host-mapped) memory. What this context carries is only the split itself.
/// The per-token split moves under the adaptive controller (`layers::tp_controller`), which
/// starts from `gpu_ratio`.
///
/// Two segments per layer, each merged once:
/// - ATTN: Q heads `[0, h_g)` + Wo columns `[0, h_g·head_dim)` on the GPU, the rest on the CPU.
/// - FFN: gate/up rows + down columns `[0, ffn_split)` on the GPU, the rest on the CPU.
#[derive(Clone)]
pub struct PartitionContext {
    pub gpu_ratio: f32,
    pub cpu_backend: Arc<dyn Backend>,
    /// Initial FFN split (rows of gate/up, columns of down) on the GPU; 128-aligned.
    pub ffn_split: usize,
    /// INV-120: monotonic counter bumped every time the partition ratio changes. Plan builds
    /// capture this value at build time; a plan whose captured value no longer matches returns
    /// `PlanInvalidated` and is rebuilt.
    ///
    /// Shared via `Arc` so all layers observe the same monotonic ordering.
    pub ratio_generation: Arc<std::sync::atomic::AtomicU64>,
}

/// GPU range and CPU range of one split axis: `[0, split)` and `[split, total)`.
pub fn split_ranges(
    split: usize,
    total: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    (0..split, split..total)
}

/// ATTN element ranges on the Q / Wo-column axis for `h_g` GPU heads.
pub fn attn_ranges(
    h_g: usize,
    n_heads_q: usize,
    head_dim: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    split_ranges(h_g * head_dim, n_heads_q * head_dim)
}

/// Column-slice alignment of the CPU/GPU GEMV shares (GPU work-group and CPU chunk alignment).
pub const COLSLICE_ALIGN: usize = 128;

/// F16 GEMVs sharing one activation: for each `(w, ld, out, n)`,
/// `out[j] = Σ_{i<k} w[j·ld + i] · x[i]` for `j < n`.
///
/// A row slice `W[r_lo..r_hi, :]` of a `[N, K]` matrix is `w = W + r_lo·K, ld = K`; a column
/// slice `W[:, k_lo..k_hi]` is `w = W + k_lo, ld = K` with `x` starting at `k_lo`. NEON through
/// the CPU kernel set on aarch64 (one F32→F16 activation conversion, one SpinPool dispatch);
/// scalar elsewhere.
///
/// # Safety
/// `x` must have `k` readable floats; every weight row `j < n` must have `k` readable halves at
/// `w + j·ld`; every `out` must have `n` writable floats; at most 3 GEMVs.
pub unsafe fn gemv_f16_strided(
    kernels: Option<&'static crate::cpu_kernels::CpuKernelSet>,
    x: *const f32,
    k: usize,
    mats: &[(*const u16, usize, *mut f32, usize)],
) {
    #[cfg(target_arch = "aarch64")]
    if let Some(ks) = kernels {
        unsafe { (ks.fused_matmul_f16_ld)(x, k, mats) };
        return;
    }
    let _ = kernels;
    let x = unsafe { std::slice::from_raw_parts(x, k) };
    for &(w, ld, out, n) in mats {
        for j in 0..n {
            let row = unsafe { std::slice::from_raw_parts(w.add(j * ld), k) };
            let acc: f32 = row
                .iter()
                .zip(x)
                .map(|(&h, &a)| half::f16::from_bits(h).to_f32() * a)
                .sum();
            unsafe { *out.add(j) = acc };
        }
    }
}

/// Column-slice GEMV `out = W[:, k_lo..k_hi] · x[k_lo..k_hi]` of a row-major F16 `W[n, ld]`.
///
/// Both bounds must sit on the 128-column grid (`k_hi` may also be `ld`), mirroring the GPU
/// `kernel_mul_mat_f16_f32_ld` share, which always takes the leading columns.
#[allow(clippy::too_many_arguments)]
pub fn matmul_f16_colslice(
    kernels: Option<&'static crate::cpu_kernels::CpuKernelSet>,
    x: &[f32],
    w: &[u16],
    n: usize,
    ld: usize,
    k_lo: usize,
    k_hi: usize,
    out: &mut [f32],
) -> Result<()> {
    ensure!(
        k_lo < k_hi && k_hi <= ld,
        "column slice [{k_lo}, {k_hi}) outside [0, {ld})"
    );
    ensure!(
        k_lo.is_multiple_of(COLSLICE_ALIGN) && (k_hi.is_multiple_of(COLSLICE_ALIGN) || k_hi == ld),
        "column slice [{k_lo}, {k_hi}) is off the {COLSLICE_ALIGN}-column grid"
    );
    ensure!(
        x.len() >= k_hi && w.len() >= n * ld && out.len() >= n,
        "column slice buffers too small"
    );
    // SAFETY: bounds checked above.
    unsafe {
        gemv_f16_strided(
            kernels,
            x.as_ptr().add(k_lo),
            k_hi - k_lo,
            &[(w.as_ptr().add(k_lo), ld, out.as_mut_ptr(), n)],
        );
    }
    Ok(())
}

/// Indices the CPU attention share uses for the token at RoPE position `start_pos` that is
/// written at cache slot `write_pos` — `(rope_pos, slot, attn_len, new_len)`. The two differ
/// after a compaction: the RoPE clock keeps counting while the cache renumbers its slots down.
pub fn cpu_share_indices(start_pos: usize, write_pos: usize) -> (usize, usize, usize, usize) {
    (start_pos, write_pos, write_pos + 1, write_pos + 1)
}

/// Single-query attention for Q heads `heads` only, one KV group at a time.
///
/// `q` / `out` hold all `n_heads_q` heads (`[.., n_heads_q·head_dim]` F32); only the rows of
/// `heads` are read / written. `k_cache` / `v_cache` are HeadMajor `[1, n_kv, capacity,
/// head_dim]`. The backend's `attention_gen` maps Q head `h` to KV head `h / (n_q / n_kv)`, so a
/// range that straddles a group boundary must be called per group — each call sees one KV head
/// and only that group's Q heads.
///
/// `kv_start` (one entry per KV head) marks a ragged cache: KV head `h` is resident over
/// `[kv_start[h], cache_seq_len)` (see `Backend::attention_gen_ragged`). `scores_out` holds
/// `n_heads_q` rows of `scores_out.len() / n_heads_q` floats; the post-softmax row of each head in
/// `heads` is written to its columns `[0, cache_seq_len)` (hole columns 0).
#[allow(clippy::too_many_arguments)]
pub fn attention_head_range(
    backend: &dyn Backend,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    out: &mut Tensor,
    heads: std::ops::Range<usize>,
    n_heads_q: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cache_seq_len: usize,
    kv_start: Option<&[usize]>,
    mut scores_out: Option<&mut [f32]>,
) -> Result<()> {
    let dims = k_cache.shape().dims();
    ensure!(
        dims.len() == 4 && dims[1] == n_kv_heads && dims[3] == head_dim,
        "attention_head_range expects HeadMajor [1, n_kv, cap, head_dim], got {dims:?}"
    );
    ensure!(
        kv_start.is_none_or(|s| s.len() == n_kv_heads),
        "attention_head_range: kv_start needs one entry per KV head"
    );
    let score_stride = scores_out.as_ref().map_or(0, |s| s.len() / n_heads_q);
    ensure!(
        scores_out.is_none() || score_stride >= cache_seq_len,
        "attention_head_range: score rows shorter than the cache ({score_stride} < {cache_seq_len})"
    );
    let capacity = dims[2];
    let group = n_heads_q / n_kv_heads;
    let kv_dtype = k_cache.dtype();
    let kv_elem = kv_dtype.size();
    let head_bytes = capacity * head_dim * kv_elem;
    let row_bytes = head_dim * 4;
    let view = |t: &Tensor, off: usize, len: usize, shape: Vec<usize>, dtype| -> Result<Tensor> {
        let buf = crate::buffer::slice::SliceBuffer::new(t.buffer().clone(), off, len, dtype)?;
        Ok(Tensor::new(
            Shape::new(shape),
            Arc::new(buf),
            t.backend().clone(),
        ))
    };
    let mut h = heads.start;
    while h < heads.end {
        let kv_h = h / group;
        let h_end = heads.end.min((kv_h + 1) * group);
        let nh = h_end - h;
        let q_v = view(
            q,
            h * row_bytes,
            nh * row_bytes,
            vec![1, 1, nh, head_dim],
            DType::F32,
        )?;
        let mut o_v = view(
            out,
            h * row_bytes,
            nh * row_bytes,
            vec![1, 1, nh, head_dim],
            DType::F32,
        )?;
        let kv_shape = vec![1, 1, capacity, head_dim];
        let k_v = view(
            k_cache,
            kv_h * head_bytes,
            head_bytes,
            kv_shape.clone(),
            kv_dtype,
        )?;
        let v_v = view(v_cache, kv_h * head_bytes, head_bytes, kv_shape, kv_dtype)?;
        let scores = scores_out
            .as_deref_mut()
            .map(|s| &mut s[h * score_stride..h_end * score_stride]);
        match kv_start {
            Some(starts) => backend.attention_gen_ragged(
                &q_v,
                &k_v,
                &v_v,
                &mut o_v,
                nh,
                1,
                head_dim,
                &starts[kv_h..kv_h + 1],
                None,
                0,
                cache_seq_len,
                scores,
            )?,
            None => backend.attention_gen(
                &q_v,
                &k_v,
                &v_v,
                &mut o_v,
                nh,
                1,
                head_dim,
                cache_seq_len,
                scores,
            )?,
        }
        h = h_end;
    }
    Ok(())
}

/// Merge 2D partial results from GPU and CPU partitions (prefill path).
///
/// GPU partial: `[batch, seq_len, split_row]` (F32, on GPU)
/// CPU partial: `[batch, seq_len, cpu_rows]` (F32, on CPU)
/// Output:      `[batch, seq_len, out_dim]`  (F32, on GPU) where out_dim = split_row + cpu_rows
///
/// Strategy (approach C): read GPU partial to CPU temp, interleave rows, write to output.
/// Prefill is bandwidth-bound, so the extra memcpy overhead is acceptable.
pub fn merge_partials_2d(
    backend: &dyn Backend,
    gpu_partial: &Tensor,
    cpu_partial: &Tensor,
    output: &mut Tensor,
    total_rows: usize, // batch_size * seq_len
    split_row: usize,
    cpu_rows: usize,
) -> Result<()> {
    let out_dim = split_row + cpu_rows;

    // 1. Read GPU partial to CPU temp buffer
    let gpu_bytes = total_rows * split_row * 4;
    let mut gpu_temp = vec![0u8; gpu_bytes];
    backend.read_buffer(gpu_partial, &mut gpu_temp)?;

    // 2. Interleave: build merged [total_rows, out_dim] on CPU
    let out_bytes = total_rows * out_dim * 4;
    let mut merged = vec![0u8; out_bytes];

    // Safety: cpu_partial is a CPU tensor with valid host pointer.
    let cpu_data =
        unsafe { std::slice::from_raw_parts(cpu_partial.as_ptr(), total_rows * cpu_rows * 4) };

    for s in 0..total_rows {
        let gpu_row_start = s * split_row * 4;
        let cpu_row_start = s * cpu_rows * 4;
        let out_row_start = s * out_dim * 4;

        // GPU columns [0..split_row)
        merged[out_row_start..out_row_start + split_row * 4]
            .copy_from_slice(&gpu_temp[gpu_row_start..gpu_row_start + split_row * 4]);

        // CPU columns [split_row..out_dim)
        merged[out_row_start + split_row * 4..out_row_start + out_dim * 4]
            .copy_from_slice(&cpu_data[cpu_row_start..cpu_row_start + cpu_rows * 4]);
    }

    // 3. Write merged result to GPU output
    backend.write_buffer(output, &merged)?;

    Ok(())
}

/// Variant of `merge_partials_2d` that uses caller-provided scratch buffers
/// instead of allocating fresh `Vec<u8>`s on each call. Designed for the
/// per-layer prefill partition path where the same buffers are reused 28+
/// times per prefill — eliminating the alloc churn (`~56` Vec allocations
/// per token for a 28-layer FFN partition) is the primary D-min win.
///
/// `gpu_temp_scratch.len()` must be >= `total_rows * split_row * 4`.
/// `merged_scratch.len()` must be >= `total_rows * (split_row + cpu_rows) * 4`.
/// Both scratches may be larger (sized for the max of gate/up); only the
/// prefix used by the current call is touched.
#[allow(clippy::too_many_arguments)]
pub fn merge_partials_2d_into(
    backend: &dyn Backend,
    gpu_partial: &Tensor,
    cpu_partial: &Tensor,
    output: &mut Tensor,
    total_rows: usize,
    split_row: usize,
    cpu_rows: usize,
    gpu_temp_scratch: &mut [u8],
    merged_scratch: &mut [u8],
) -> Result<()> {
    let out_dim = split_row + cpu_rows;
    let gpu_bytes = total_rows * split_row * 4;
    let out_bytes = total_rows * out_dim * 4;
    debug_assert!(
        gpu_temp_scratch.len() >= gpu_bytes,
        "gpu_temp_scratch too small: {} < {}",
        gpu_temp_scratch.len(),
        gpu_bytes
    );
    debug_assert!(
        merged_scratch.len() >= out_bytes,
        "merged_scratch too small: {} < {}",
        merged_scratch.len(),
        out_bytes
    );

    // 1. Read GPU partial into the prefix of the scratch buffer.
    let gpu_slice = &mut gpu_temp_scratch[..gpu_bytes];
    backend.read_buffer(gpu_partial, gpu_slice)?;

    // 2. Interleave: build merged [total_rows, out_dim] in scratch.
    let merged = &mut merged_scratch[..out_bytes];
    // Safety: cpu_partial is a CPU tensor with valid host pointer.
    let cpu_data =
        unsafe { std::slice::from_raw_parts(cpu_partial.as_ptr(), total_rows * cpu_rows * 4) };

    let split_bytes = split_row * 4;
    let cpu_bytes_per_row = cpu_rows * 4;
    let out_bytes_per_row = out_dim * 4;
    for s in 0..total_rows {
        let gpu_row_start = s * split_bytes;
        let cpu_row_start = s * cpu_bytes_per_row;
        let out_row_start = s * out_bytes_per_row;
        merged[out_row_start..out_row_start + split_bytes]
            .copy_from_slice(&gpu_slice[gpu_row_start..gpu_row_start + split_bytes]);
        merged[out_row_start + split_bytes..out_row_start + out_bytes_per_row]
            .copy_from_slice(&cpu_data[cpu_row_start..cpu_row_start + cpu_bytes_per_row]);
    }

    // 3. Write merged result to GPU output.
    backend.write_buffer(output, merged)?;
    Ok(())
}

/// When `LLMRS_PARTITION_FUSED_MERGE=1`, the tensor-partition decode path
/// folds the per-layer merge 3-step + residual add + next layer's
/// attention-norm into a single `fused_norm_merge` kernel call at the next
/// layer's entry. Reduces 5 inter-kernel barriers to 1 on the OpenCL
/// in-order queue. Default: off.
pub fn partition_fused_merge_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("LLMRS_PARTITION_FUSED_MERGE").is_ok_and(|v| v == "1"))
}

/// Gate for the tensor-partition plan path (`backend/opencl/tp_plan.rs`).
/// `LLMRS_PARTITION_PLAN=0` makes the partitioned plan build fail; the partition arm then stops
/// with `[tp] FATAL plan path unavailable` (exit 3) instead of silently running GPU-only — the
/// negative control of ticket 021 completion criterion 3. Default: enabled.
///
/// The decision is cached at first read via `OnceLock` because plan builds
/// and hot dispatch must observe a single stable value across a generation.
pub fn partition_plan_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LLMRS_PARTITION_PLAN")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;

    /// Helper: create a CPU backend Arc.
    fn cpu_backend() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    /// Helper: allocate an F32 weight tensor [out_dim, in_dim] with sequential values.
    fn make_f32_weight(out_dim: usize, in_dim: usize) -> Tensor {
        let memory = Galloc::new();
        let size = out_dim * in_dim * 4;
        let buf = memory.alloc(size, DType::F32).unwrap();
        let mut tensor = Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, cpu_backend());
        let data = tensor.as_mut_slice::<f32>();
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f32 * 0.001;
        }
        tensor
    }

    // PA-T1-FASTPATH: GPU-only fast-path threshold classifier is correct.
    // Ratios at or above `GPU_ONLY_THRESHOLD` must take the fast path
    // (partition_ctx = None). Ratios strictly below must NOT (real split).
    #[test]
    fn test_gpu_only_fast_path_classifier() {
        // At threshold: fast path.
        assert!(is_gpu_only_ratio(GPU_ONLY_THRESHOLD));
        // Above: fast path.
        assert!(is_gpu_only_ratio(0.999));
        assert!(is_gpu_only_ratio(1.0));
        // Below: partition path.
        assert!(!is_gpu_only_ratio(0.99));
        assert!(!is_gpu_only_ratio(0.75));
        assert!(!is_gpu_only_ratio(0.5));
        assert!(!is_gpu_only_ratio(0.001));
        assert!(!is_gpu_only_ratio(0.0));
    }

    // ── ticket 021: zero-copy split helpers ──

    /// Deterministic pseudo-random values in `[-1, 1)` (xorshift; no rand dependency).
    fn noise(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed.max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    fn tp_colslice_gemv_matches_dense() {
        let (n, k) = (1536usize, 8960usize);
        let w_f32 = noise(7, n * k);
        let w: Vec<u16> = w_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let x = noise(11, k);
        // Dense reference in f32 over the f16-rounded weights.
        let dense: Vec<f32> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|i| half::f16::from_bits(w[j * k + i]).to_f32() * x[i])
                    .sum()
            })
            .collect();
        let y_inf = dense.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for split in [128usize, 1280, 4480, 7680, 8832] {
            let mut lo = vec![0.0f32; n];
            let mut hi = vec![0.0f32; n];
            matmul_f16_colslice(None, &x, &w, n, k, 0, split, &mut lo).unwrap();
            matmul_f16_colslice(None, &x, &w, n, k, split, k, &mut hi).unwrap();
            let err = (0..n)
                .map(|j| (lo[j] + hi[j] - dense[j]).abs())
                .fold(0.0f32, f32::max);
            assert!(
                err <= 1e-3 * y_inf,
                "split={split}: max err {err} > 1e-3·{y_inf}"
            );
        }
        let mut out = vec![0.0f32; n];
        assert!(
            matmul_f16_colslice(None, &x, &w, n, k, 0, 100, &mut out).is_err(),
            "an off-grid column bound must be refused"
        );
        assert!(matmul_f16_colslice(None, &x, &w, n, k, 100, k, &mut out).is_err());
    }

    #[test]
    fn tp_rowslice_offsets_cover_dim() {
        let check = |(g, c): (std::ops::Range<usize>, std::ops::Range<usize>), total: usize| {
            assert_eq!(g.start, 0);
            assert_eq!(g.end, c.start, "no gap, no overlap");
            assert_eq!(c.end, total);
            assert!(!g.is_empty() && !c.is_empty(), "both devices keep a share");
        };
        for h_g in 1..=11 {
            check(attn_ranges(h_g, 12, 128), 1536);
            check(split_ranges(h_g, 12), 12);
        }
        for s in (128..=8832).step_by(128) {
            check(split_ranges(s, 8960), 8960);
        }
    }

    #[test]
    fn tp_attn_group_split_matches_full() {
        let be = cpu_backend();
        let (n_q, n_kv, hd, cap) = (12usize, 2usize, 128usize, 512usize);
        let alloc = |shape: Vec<usize>, dtype: DType| {
            let elems: usize = shape.iter().product();
            let buf = Galloc::new().alloc(elems * dtype.size(), dtype).unwrap();
            Tensor::new(Shape::new(shape), buf, be.clone())
        };
        let mut q = alloc(vec![1, 1, n_q, hd], DType::F32);
        q.as_mut_slice::<f32>().copy_from_slice(&noise(3, n_q * hd));
        let mut k = alloc(vec![1, n_kv, cap, hd], DType::F16);
        let mut v = alloc(vec![1, n_kv, cap, hd], DType::F16);
        for (t, seed) in [(&mut k, 5u64), (&mut v, 9)] {
            let vals = noise(seed, n_kv * cap * hd);
            for (d, s) in t.as_mut_slice::<half::f16>().iter_mut().zip(vals) {
                *d = half::f16::from_f32(s);
            }
        }
        for seq in [1usize, 37, 512] {
            let mut full = alloc(vec![1, 1, n_q, hd], DType::F32);
            be.attention_gen(&q, &k, &v, &mut full, n_q, n_kv, hd, seq, None)
                .unwrap();
            for h_g in [1usize, 5, 6, 7, 11] {
                let mut split = alloc(vec![1, 1, n_q, hd], DType::F32);
                for heads in [0..h_g, h_g..n_q] {
                    attention_head_range(
                        be.as_ref(),
                        &q,
                        &k,
                        &v,
                        &mut split,
                        heads,
                        n_q,
                        n_kv,
                        hd,
                        seq,
                        None,
                        None,
                    )
                    .unwrap();
                }
                let err = full
                    .as_slice::<f32>()
                    .iter()
                    .zip(split.as_slice::<f32>())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(err <= 1e-5, "seq={seq} h_g={h_g}: max err {err}");
            }
        }
    }

    /// Ticket 023 E1: after a compaction the RoPE position runs ahead of the cache slot.
    #[test]
    fn tp_cpu_share_indices_after_compaction() {
        for (start_pos, write_pos) in [(10usize, 10usize), (1500, 882), (2324, 807)] {
            let (rope_pos, slot, attn_len, new_len) = cpu_share_indices(start_pos, write_pos);
            assert_eq!(rope_pos, start_pos);
            assert_eq!(slot, write_pos);
            assert_eq!(attn_len, write_pos + 1);
            assert_eq!(new_len, write_pos + 1);
        }
    }

    /// Q 12 · KV 2 · hd 128 inputs over a KV capacity of 64 (> 1: x86 reads a capacity-1 view as
    /// SeqMajor).
    fn head_range_inputs(be: &Arc<dyn Backend>) -> (Tensor, Tensor, Tensor) {
        let (n_q, n_kv, hd, cap) = (12usize, 2usize, 128usize, 64usize);
        let alloc = |shape: Vec<usize>, dtype: DType| {
            let elems: usize = shape.iter().product();
            let buf = Galloc::new().alloc(elems * dtype.size(), dtype).unwrap();
            Tensor::new(Shape::new(shape), buf, be.clone())
        };
        let mut q = alloc(vec![1, 1, n_q, hd], DType::F32);
        q.as_mut_slice::<f32>().copy_from_slice(&noise(3, n_q * hd));
        let mut k = alloc(vec![1, n_kv, cap, hd], DType::F16);
        let mut v = alloc(vec![1, n_kv, cap, hd], DType::F16);
        for (t, seed) in [(&mut k, 5u64), (&mut v, 9)] {
            let vals = noise(seed, n_kv * cap * hd);
            for (d, s) in t.as_mut_slice::<half::f16>().iter_mut().zip(vals) {
                *d = half::f16::from_f32(s);
            }
        }
        (q, k, v)
    }

    fn out_tensor(be: &Arc<dyn Backend>) -> Tensor {
        let buf = Galloc::new().alloc(12 * 128 * 4, DType::F32).unwrap();
        Tensor::new(Shape::new(vec![1, 1, 12, 128]), buf, be.clone())
    }

    fn max_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Ticket 023 E2: the CPU heads' probability rows equal the full attention's rows.
    #[test]
    fn tp_attn_head_range_scores_match_full() {
        let be = cpu_backend();
        let (n_q, n_kv, hd) = (12usize, 2usize, 128usize);
        let (q, k, v) = head_range_inputs(&be);
        for seq in [1usize, 37, 64] {
            let mut full = out_tensor(&be);
            let mut full_s = vec![0.0f32; n_q * seq];
            be.attention_gen(&q, &k, &v, &mut full, n_q, n_kv, hd, seq, Some(&mut full_s))
                .unwrap();
            for h_g in [1usize, 5, 6, 7, 11] {
                let mut part = out_tensor(&be);
                let mut s = vec![f32::NAN; n_q * seq];
                attention_head_range(
                    be.as_ref(),
                    &q,
                    &k,
                    &v,
                    &mut part,
                    h_g..n_q,
                    n_q,
                    n_kv,
                    hd,
                    seq,
                    None,
                    Some(&mut s),
                )
                .unwrap();
                let rows = h_g * seq..n_q * seq;
                let err = max_err(&full_s[rows.clone()], &s[rows]);
                assert!(err <= 1e-5, "seq={seq} h_g={h_g}: score err {err}");
                for h in h_g..n_q {
                    let sum: f32 = s[h * seq..(h + 1) * seq].iter().sum();
                    assert!((sum - 1.0).abs() <= 1e-4, "seq={seq} h={h}: row sum {sum}");
                }
                let err = max_err(
                    &full.as_slice::<f32>()[h_g * hd..],
                    &part.as_slice::<f32>()[h_g * hd..],
                );
                assert!(err <= 1e-5, "seq={seq} h_g={h_g}: output err {err}");
            }
        }
    }

    /// Ticket 023 E4: on a ragged cache the head range equals the full ragged attention.
    /// (`attention_head_range` calls `attention_gen_ragged` itself, so this checks the per-group
    /// slicing of Q / KV / start / score rows, not the ragged kernel.)
    #[test]
    fn tp_attn_head_range_ragged_matches_full() {
        let be = cpu_backend();
        let (n_q, n_kv, hd, seq) = (12usize, 2usize, 128usize, 64usize);
        let (q, k, v) = head_range_inputs(&be);
        for starts in [[0usize, 0], [0, 17], [30, 5]] {
            let mut full = out_tensor(&be);
            let mut full_s = vec![0.0f32; n_q * seq];
            be.attention_gen_ragged(
                &q,
                &k,
                &v,
                &mut full,
                n_q,
                n_kv,
                hd,
                &starts,
                None,
                0,
                seq,
                Some(&mut full_s),
            )
            .unwrap();
            for h_g in [1usize, 5, 6, 7, 11] {
                let mut part = out_tensor(&be);
                let mut s = vec![f32::NAN; n_q * seq];
                attention_head_range(
                    be.as_ref(),
                    &q,
                    &k,
                    &v,
                    &mut part,
                    h_g..n_q,
                    n_q,
                    n_kv,
                    hd,
                    seq,
                    Some(&starts),
                    Some(&mut s),
                )
                .unwrap();
                let rows = h_g * seq..n_q * seq;
                let err = max_err(&full_s[rows.clone()], &s[rows]);
                assert!(err <= 1e-5, "starts={starts:?} h_g={h_g}: score err {err}");
                let err = max_err(
                    &full.as_slice::<f32>()[h_g * hd..],
                    &part.as_slice::<f32>()[h_g * hd..],
                );
                assert!(err <= 1e-5, "starts={starts:?} h_g={h_g}: output err {err}");
                for h in h_g..n_q {
                    let start = starts[h / (n_q / n_kv)];
                    assert!(
                        s[h * seq..h * seq + start].iter().all(|&x| x == 0.0),
                        "starts={starts:?} h={h}: hole column not 0"
                    );
                }
            }
        }
    }

    // ── merge_partials_2d tests ──

    /// Verify that merge_partials_2d correctly interleaves GPU and CPU partials.
    #[test]
    fn test_merge_partials_2d_basic() {
        let backend = cpu_backend();
        let memory = Galloc::new();

        let seq_len = 4;
        let split_row = 3;
        let cpu_rows = 2;
        let out_dim = split_row + cpu_rows;

        // GPU partial: [seq_len, split_row], values 100+
        let gpu_buf = memory.alloc(seq_len * split_row * 4, DType::F32).unwrap();
        let mut gpu_partial = Tensor::new(
            Shape::new(vec![seq_len, split_row]),
            gpu_buf,
            backend.clone(),
        );
        let gpu_data = gpu_partial.as_mut_slice::<f32>();
        for (i, v) in gpu_data.iter_mut().enumerate() {
            *v = 100.0 + i as f32;
        }

        // CPU partial: [seq_len, cpu_rows], values 200+
        let cpu_buf = memory.alloc(seq_len * cpu_rows * 4, DType::F32).unwrap();
        let mut cpu_partial = Tensor::new(
            Shape::new(vec![seq_len, cpu_rows]),
            cpu_buf,
            backend.clone(),
        );
        let cpu_data_w = cpu_partial.as_mut_slice::<f32>();
        for (i, v) in cpu_data_w.iter_mut().enumerate() {
            *v = 200.0 + i as f32;
        }

        // Output: [seq_len, out_dim]
        let out_buf = memory.alloc(seq_len * out_dim * 4, DType::F32).unwrap();
        let mut output = Tensor::new(Shape::new(vec![seq_len, out_dim]), out_buf, backend.clone());

        super::merge_partials_2d(
            backend.as_ref(),
            &gpu_partial,
            &cpu_partial,
            &mut output,
            seq_len,
            split_row,
            cpu_rows,
        )
        .unwrap();

        let result = output.as_slice::<f32>();
        for s in 0..seq_len {
            // GPU part
            for c in 0..split_row {
                let expected = 100.0 + (s * split_row + c) as f32;
                let actual = result[s * out_dim + c];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "row {} col {}: expected {}, got {}",
                    s,
                    c,
                    expected,
                    actual,
                );
            }
            // CPU part
            for c in 0..cpu_rows {
                let expected = 200.0 + (s * cpu_rows + c) as f32;
                let actual = result[s * out_dim + split_row + c];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "row {} col {}: expected {}, got {}",
                    s,
                    split_row + c,
                    expected,
                    actual,
                );
            }
        }
    }

    /// Verify merge_partials_2d with single-row input (degenerate case matching decode).
    #[test]
    fn test_merge_partials_2d_single_row() {
        let backend = cpu_backend();
        let memory = Galloc::new();

        let split_row = 4;
        let cpu_rows = 3;
        let out_dim = split_row + cpu_rows;

        let gpu_buf = memory.alloc(split_row * 4, DType::F32).unwrap();
        let mut gpu_partial = Tensor::new(Shape::new(vec![1, split_row]), gpu_buf, backend.clone());
        gpu_partial
            .as_mut_slice::<f32>()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let cpu_buf = memory.alloc(cpu_rows * 4, DType::F32).unwrap();
        let mut cpu_partial = Tensor::new(Shape::new(vec![1, cpu_rows]), cpu_buf, backend.clone());
        cpu_partial
            .as_mut_slice::<f32>()
            .copy_from_slice(&[5.0, 6.0, 7.0]);

        let out_buf = memory.alloc(out_dim * 4, DType::F32).unwrap();
        let mut output = Tensor::new(Shape::new(vec![1, out_dim]), out_buf, backend.clone());

        super::merge_partials_2d(
            backend.as_ref(),
            &gpu_partial,
            &cpu_partial,
            &mut output,
            1,
            split_row,
            cpu_rows,
        )
        .unwrap();

        let result = output.as_slice::<f32>();
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    /// `merge_partials_2d_into` (scratch-aware variant) must produce identical
    /// output to `merge_partials_2d` for the same inputs. Covers the prefill
    /// path that reuses caller-owned scratch buffers across all layers.
    #[test]
    fn test_merge_partials_2d_into_matches_legacy() {
        let backend = cpu_backend();
        let memory = Galloc::new();

        let total_rows = 7; // batch*seq_len, deliberately not aligned
        let split_row = 5;
        let cpu_rows = 3;
        let out_dim = split_row + cpu_rows;

        // Build the same GPU+CPU partials twice, run both functions, expect equality.
        let make_input = || {
            let gpu_buf = memory
                .alloc(total_rows * split_row * 4, DType::F32)
                .unwrap();
            let mut gpu_partial = Tensor::new(
                Shape::new(vec![total_rows, split_row]),
                gpu_buf,
                backend.clone(),
            );
            for (i, v) in gpu_partial.as_mut_slice::<f32>().iter_mut().enumerate() {
                *v = (i as f32) * 1.5 - 0.25;
            }
            let cpu_buf = memory.alloc(total_rows * cpu_rows * 4, DType::F32).unwrap();
            let mut cpu_partial = Tensor::new(
                Shape::new(vec![total_rows, cpu_rows]),
                cpu_buf,
                backend.clone(),
            );
            for (i, v) in cpu_partial.as_mut_slice::<f32>().iter_mut().enumerate() {
                *v = -(i as f32) * 2.0 + 7.0;
            }
            (gpu_partial, cpu_partial)
        };

        // Reference (legacy).
        let (gpu_a, cpu_a) = make_input();
        let mut out_a = Tensor::new(
            Shape::new(vec![total_rows, out_dim]),
            memory.alloc(total_rows * out_dim * 4, DType::F32).unwrap(),
            backend.clone(),
        );
        super::merge_partials_2d(
            backend.as_ref(),
            &gpu_a,
            &cpu_a,
            &mut out_a,
            total_rows,
            split_row,
            cpu_rows,
        )
        .unwrap();
        let ref_vals: Vec<f32> = out_a.as_slice::<f32>().to_vec();

        // New scratch-aware path with oversized scratches (mimics prefill scratch
        // sized for the larger of gate/up).
        let (gpu_b, cpu_b) = make_input();
        let mut out_b = Tensor::new(
            Shape::new(vec![total_rows, out_dim]),
            memory.alloc(total_rows * out_dim * 4, DType::F32).unwrap(),
            backend.clone(),
        );
        let mut gpu_temp = vec![0u8; total_rows * (split_row + 4) * 4];
        let mut merged = vec![0u8; total_rows * (out_dim + 7) * 4];
        super::merge_partials_2d_into(
            backend.as_ref(),
            &gpu_b,
            &cpu_b,
            &mut out_b,
            total_rows,
            split_row,
            cpu_rows,
            &mut gpu_temp,
            &mut merged,
        )
        .unwrap();
        assert_eq!(out_b.as_slice::<f32>(), ref_vals.as_slice());
    }

    // ── AB-4 (C): apply_partition_dispatch fan-out 등가 ──

    use crate::models::weights::LayerSlot;

    fn ffn_layer_slot(be: &Arc<dyn Backend>, idx: usize) -> Arc<LayerSlot> {
        use crate::layers::transformer_layer::TransformerLayer;
        // FFN weights with ffn_hidden >= 256 (two 128-row quanta); the attention
        // placeholders are never read by apply_dispatch.
        let small = make_f32_weight(1, 1);
        let layer = TransformerLayer {
            wq: small.clone(),
            wk: small.clone(),
            wv: small.clone(),
            wo: small.clone(),
            w_gate: make_f32_weight(512, 256),
            w_up: make_f32_weight(512, 256),
            w_down: make_f32_weight(256, 512),
            attention_norm: small.clone(),
            ffn_norm: small,
            qkv_bias: None,
            q_norm: None,
            k_norm: None,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            partition_ctx: None,
        };
        let _ = be;
        Arc::new(LayerSlot::new(layer, DType::F32, None, idx))
    }

    fn cpu_only_hardware(be: &Arc<dyn Backend>) -> crate::hardware::Hardware {
        let host: Arc<dyn crate::memory::Memory> = Arc::new(Galloc::new());
        crate::hardware::Hardware::new(be.clone(), None, None, host, None)
    }

    /// GPU-only fast path: ratio >= threshold → Full fan-out, returns 0 and
    /// leaves partition_ctx cleared.
    #[test]
    fn apply_partition_dispatch_gpu_only_returns_zero() {
        let be = cpu_backend();
        let hw = cpu_only_hardware(&be);
        let slots: Vec<_> = (0..3).map(|i| ffn_layer_slot(&be, i)).collect();

        let n = apply_partition_dispatch(&slots, 0.999, &hw).unwrap();
        assert_eq!(n, 0, "GPU-only fast path fans out Full, returns 0");
        for slot in &slots {
            assert!(
                slot.load_weights().partition_ctx.is_none(),
                "Full leaves partition_ctx cleared"
            );
        }
    }

    /// Partition fan-out: ratio below threshold → Partition fan-out, returns
    /// slots.len() * 3 and installs partition_ctx on every slot.
    #[test]
    fn apply_partition_dispatch_partition_returns_three_per_slot() {
        let be = cpu_backend();
        let hw = cpu_only_hardware(&be);
        let slots: Vec<_> = (0..3).map(|i| ffn_layer_slot(&be, i)).collect();

        let n = apply_partition_dispatch(&slots, 0.5, &hw).unwrap();
        assert_eq!(
            n,
            slots.len() * 3,
            "Partition fan-out returns slots.len()*3"
        );
        for slot in &slots {
            assert!(
                slot.load_weights().partition_ctx.is_some(),
                "Partition installs partition_ctx on every slot"
            );
        }
    }
}
