use crate::backend::Backend;
use crate::buffer::DType;
use crate::buffer::slice::SliceBuffer;
use crate::shape::Shape;
use crate::tensor::Tensor;
use anyhow::{Result, ensure};
use log::debug;
use std::sync::Arc;

/// Row alignment granularity for weight splits.
/// 128 rows ensures alignment with GPU workgroup sizes and Q4_0 block boundaries.
const ROW_ALIGNMENT: usize = 128;

/// GPU-only fast-path threshold for `gpu_ratio`.
///
/// When `gpu_ratio >= GPU_ONLY_THRESHOLD`, the partition activation path is
/// skipped entirely: `partition_ctx` stays `None`, and the forward pass uses
/// the dense full-weight matmul on GPU. This avoids the ratio-independent
/// constant overhead of the partition path (host staging `read_buffer`,
/// CPU matmul kick-off for a clamped 128-row slice, and GPU<->host merge).
///
/// The rationale: `split_weight` clamps `split_row` to `[128, out_dim - 128]`
/// (Q4_0 alignment requirement). A caller passing `ratio=0.999` still ends up
/// with 128 CPU rows per weight, forcing the partition dispatch path every
/// token even though the split is effectively GPU-only. This threshold makes
/// that corner case behave as the user intends.
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
///   `slots.len() * 3` (gate/up/down re-sliced per layer).
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
    use technique_api::DeviceTarget;

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

/// A single slice of a partitioned weight: the sub-tensor, the backend that
/// computes it, and the number of elements along the split axis.
///
/// `rows` = number of elements along the **split axis**: for a row-split
/// (`split_weight`, out_dim axis) it is the number of output rows; for a
/// col-split (`split_weight_col`, in_dim axis) it is the number of inner
/// columns. This may differ from `tensor.shape().dims()[0]` (e.g. col-split
/// keeps `out_dim` rows while `rows` carries the column count).
///
/// N-capable container element (current production: always 2 = `[gpu, cpu]`).
#[derive(Clone)]
pub struct WeightSlice {
    pub tensor: Tensor,
    pub backend: Arc<dyn Backend>,
    pub rows: usize,
}

/// A weight tensor split row-wise into GPU and CPU partitions.
///
/// For a weight `W [out_dim, in_dim]`, the split produces:
///   - `slices[0]` (GPU): `W[0..split_row, :]`   (processed by GPU)
///   - `slices[1]` (CPU): `W[split_row.., :]`     (processed by CPU)
///
/// Each slice owns an independent buffer (pre-copied from the original weight).
/// GPU slices have a valid `cl_mem` handle; CPU slices have valid host pointers.
///
/// `slices` is an N-capable composite; current production always holds exactly
/// two slices in fixed order `[gpu, cpu]`. Use `gpu_slice()` / `cpu_slice()` to
/// read the underlying tensors by that contract.
#[derive(Clone)]
pub struct PartitionedWeight {
    /// N-slice composite. Current production = `[gpu, cpu]` (fixed order).
    pub slices: Vec<WeightSlice>,
    pub split_row: usize,
}

impl PartitionedWeight {
    #[inline]
    pub fn gpu_slice(&self) -> &Tensor {
        &self.slices[0].tensor
    }
    #[inline]
    pub fn cpu_slice(&self) -> &Tensor {
        &self.slices[1].tensor
    }
    #[inline]
    pub fn gpu_slice_mut(&mut self) -> &mut Tensor {
        &mut self.slices[0].tensor
    }
}

/// Per-layer partition context holding CPU backend and partitioned weights.
///
/// Strategy B (whole-FFN slice): gate/up are split row-wise on `out_dim`, and
/// `down` is split column-wise on `in_dim` at the **same split point**
/// (`gate.split_row == down.split_col`). That way the CPU can compute the full
/// FFN chain — gate → up → silu_mul → down — on its slice independently, and
/// only the final `[hidden]` partial result needs to be summed with the GPU's
/// partial at the very end of the layer (instead of merging after gate/up).
#[derive(Clone)]
pub struct PartitionContext {
    pub gpu_ratio: f32,
    pub cpu_backend: Arc<dyn Backend>,
    // FFN gate/up projections, row-split on out_dim (ffn_hidden).
    pub gate: PartitionedWeight,
    pub up: PartitionedWeight,
    // FFN down projection, col-split on in_dim (ffn_hidden).
    // `down.split_row` is reused to mean `split_col` — i.e. the number of
    // columns handled on GPU. Its value matches `gate.split_row`.
    pub down: PartitionedWeight,
    /// INV-120: monotonic counter bumped every time the partition ratio /
    /// slice geometry changes. Plan builds (see `build_partitioned_layer_plan`
    /// in `backend/opencl/plan.rs`) capture this value at build time; the
    /// matching `PartitionStep::run` path checks it on entry and returns
    /// `PlanInvalidated` when the captured value no longer matches. This keeps
    /// stale plans from dispatching against freshly re-sliced `cl_mem`
    /// handles (the cause of R-PP2 in `arch/plan_partition_integration.md`).
    ///
    /// The counter is shared via `Arc` so all layers observe the same
    /// monotonic ordering — bumping any single layer's counter (e.g. from a
    /// dynamic `prepare_tensor_partition` call that rebuilds every layer)
    /// would still be observed as a miss by every layer's captured generation.
    pub ratio_generation: Arc<std::sync::atomic::AtomicU64>,
}

/// Compute the number of bytes per row for a given dtype and inner dimension.
///
/// For Q4_0: each row of `in_dim` elements = `in_dim / 32` blocks, each 18 bytes.
/// For Q4_1: each row of `in_dim` elements = `in_dim / 32` blocks, each 20 bytes.
/// For F16/BF16: each row = `in_dim * 2` bytes.
/// For F32: each row = `in_dim * 4` bytes.
fn bytes_per_row(dtype: DType, in_dim: usize) -> Result<usize> {
    match dtype {
        DType::Q4_0 => {
            ensure!(
                in_dim.is_multiple_of(32),
                "Q4_0 requires in_dim divisible by 32, got {}",
                in_dim
            );
            Ok(in_dim / 32 * 18)
        }
        DType::Q4_1 => {
            ensure!(
                in_dim.is_multiple_of(32),
                "Q4_1 requires in_dim divisible by 32, got {}",
                in_dim
            );
            Ok(in_dim / 32 * 20)
        }
        DType::Q8_0 => {
            ensure!(
                in_dim.is_multiple_of(32),
                "Q8_0 requires in_dim divisible by 32, got {}",
                in_dim
            );
            Ok(in_dim / 32 * 34)
        }
        DType::F16 | DType::BF16 => Ok(in_dim * 2),
        DType::F32 => Ok(in_dim * 4),
        DType::U8 => Ok(in_dim),
    }
}

/// Round `value` down to the nearest multiple of `align`.
fn align_down(value: usize, align: usize) -> usize {
    (value / align) * align
}

/// Split a weight tensor row-wise for CPU-GPU cooperative inference.
///
/// Given a weight `W [out_dim, in_dim]` and a `gpu_ratio` (0.0 to 1.0),
/// produces a `PartitionedWeight` with:
///   - `gpu_slice`: rows `[0, split_row)` — independent GPU buffer (copied from original)
///   - `cpu_slice`: rows `[split_row, out_dim)` — independent CPU buffer (copied)
///
/// `split_row` is aligned to `ROW_ALIGNMENT` (128), clamped to `[128, out_dim - 128]`.
///
/// Each slice owns its own buffer via `Backend::copy_from()`, ensuring that
/// GPU slices have a valid `cl_mem` handle and CPU slices have a valid host pointer.
/// This avoids `SliceBuffer`'s inability to provide `cl_mem` for sub-regions.
pub fn split_weight(
    weight: &Tensor,
    gpu_ratio: f32,
    cpu_backend: &Arc<dyn Backend>,
) -> Result<PartitionedWeight> {
    let shape = weight.shape();
    let dims = shape.dims();
    ensure!(
        dims.len() == 2,
        "split_weight expects 2D weight [out_dim, in_dim], got {:?}",
        dims
    );

    let out_dim = dims[0];
    let in_dim = dims[1];
    let dtype = weight.dtype();
    let bpr = bytes_per_row(dtype, in_dim)?;

    ensure!(
        out_dim >= ROW_ALIGNMENT * 2,
        "out_dim ({}) must be >= {} for partitioning",
        out_dim,
        ROW_ALIGNMENT * 2
    );

    // Compute split_row with alignment and clamping.
    let raw_split = (out_dim as f32 * gpu_ratio) as usize;
    let split_row = align_down(raw_split, ROW_ALIGNMENT)
        .max(ROW_ALIGNMENT)
        .min(out_dim - ROW_ALIGNMENT);

    let gpu_bytes = split_row * bpr;
    let cpu_bytes = (out_dim - split_row) * bpr;

    // Verify total matches weight buffer size.
    ensure!(
        gpu_bytes + cpu_bytes == weight.size(),
        "Byte split mismatch: gpu({}) + cpu({}) = {} != weight.size({})",
        gpu_bytes,
        cpu_bytes,
        gpu_bytes + cpu_bytes,
        weight.size()
    );

    let parent_buf = weight.buffer().clone();

    // GPU slice: zero-copy sub-buffer if parent has cl_mem, else fallback to copy.
    #[cfg(feature = "opencl")]
    let gpu_tensor = {
        use crate::memory::opencl::sub::ClSubBuffer;
        match ClSubBuffer::new(parent_buf.clone(), 0, gpu_bytes, dtype) {
            Ok(sub_buf) => Tensor::new(
                Shape::new(vec![split_row, in_dim]),
                Arc::new(sub_buf),
                weight.backend().clone(),
            ),
            Err(_) => {
                // Fallback: parent has no cl_mem (CPU-only backend). Copy as before.
                let tmp_gpu_buf =
                    Arc::new(SliceBuffer::new(parent_buf.clone(), 0, gpu_bytes, dtype)?);
                let tmp_gpu_tensor = Tensor::new(
                    Shape::new(vec![split_row, in_dim]),
                    tmp_gpu_buf,
                    weight.backend().clone(),
                );
                let copied = weight.backend().copy_from(&tmp_gpu_tensor)?;
                Tensor::new(
                    Shape::new(vec![split_row, in_dim]),
                    copied.buffer().clone(),
                    weight.backend().clone(),
                )
            }
        }
    };
    #[cfg(not(feature = "opencl"))]
    let gpu_tensor = {
        let tmp_gpu_buf = Arc::new(SliceBuffer::new(parent_buf.clone(), 0, gpu_bytes, dtype)?);
        let tmp_gpu_tensor = Tensor::new(
            Shape::new(vec![split_row, in_dim]),
            tmp_gpu_buf,
            weight.backend().clone(),
        );
        let copied = weight.backend().copy_from(&tmp_gpu_tensor)?;
        Tensor::new(
            Shape::new(vec![split_row, in_dim]),
            copied.buffer().clone(),
            weight.backend().clone(),
        )
    };

    // CPU slice: zero-copy SliceBuffer (cl_mem not needed for CPU matmul).
    let cpu_tensor = Tensor::new(
        Shape::new(vec![out_dim - split_row, in_dim]),
        Arc::new(SliceBuffer::new(parent_buf, gpu_bytes, cpu_bytes, dtype)?),
        cpu_backend.clone(),
    );

    debug!(
        "split_weight: [{}x{}] split_row={}, gpu_bytes={}, cpu_bytes={} (pre-copy)",
        out_dim, in_dim, split_row, gpu_bytes, cpu_bytes,
    );

    Ok(PartitionedWeight {
        // slices[0]=GPU, slices[1]=CPU 순서 고정 (gpu_slice()/cpu_slice() 접근자 계약)
        slices: vec![
            WeightSlice {
                tensor: gpu_tensor,
                backend: weight.backend().clone(),
                rows: split_row,
            },
            WeightSlice {
                tensor: cpu_tensor,
                backend: cpu_backend.clone(),
                rows: out_dim - split_row,
            },
        ],
        split_row,
    })
}

/// Split a weight tensor **column-wise** (along the `in_dim` axis) for the
/// Strategy-B whole-FFN slice path.
///
/// Given a weight `W [out_dim, in_dim]` and a `split_col` (number of inner
/// columns handled on GPU, must match the caller's gate/up `split_row`):
///   - `gpu_slice`: `W[:, 0..split_col]`       (processed by GPU)
///   - `cpu_slice`: `W[:, split_col..in_dim]`  (processed by CPU)
///
/// Unlike `split_weight` (row-split), this requires **per-row copying**
/// because each row's bytes are interleaved in memory. We issue one
/// backend-level `memcpy` per row into a fresh contiguous buffer.
///
/// `split_col` must be a multiple of 128 (enforces `ROW_ALIGNMENT` and, for
/// Q4_0, block-boundary alignment at 32-elem granularity).
///
/// Returns a `PartitionedWeight` whose `split_row` field holds the
/// **column split** (repurposed) so downstream consumers can read it with
/// the same accessor.
pub fn split_weight_col(
    weight: &Tensor,
    split_col: usize,
    cpu_backend: &Arc<dyn Backend>,
) -> Result<PartitionedWeight> {
    let shape = weight.shape();
    let dims = shape.dims();
    ensure!(
        dims.len() == 2,
        "split_weight_col expects 2D weight [out_dim, in_dim], got {:?}",
        dims
    );

    let out_dim = dims[0];
    let in_dim = dims[1];
    let dtype = weight.dtype();

    ensure!(
        split_col > 0 && split_col < in_dim,
        "split_col ({}) must be in (0, in_dim={})",
        split_col,
        in_dim
    );
    ensure!(
        split_col.is_multiple_of(ROW_ALIGNMENT),
        "split_col ({}) must be a multiple of {} (Q4_0 block + workgroup alignment)",
        split_col,
        ROW_ALIGNMENT
    );

    let full_row_bytes = bytes_per_row(dtype, in_dim)?;
    let gpu_row_bytes = bytes_per_row(dtype, split_col)?;
    let cpu_row_bytes = bytes_per_row(dtype, in_dim - split_col)?;
    ensure!(
        gpu_row_bytes + cpu_row_bytes == full_row_bytes,
        "Row byte split mismatch: gpu({}) + cpu({}) = {} != full({})",
        gpu_row_bytes,
        cpu_row_bytes,
        gpu_row_bytes + cpu_row_bytes,
        full_row_bytes,
    );

    let total_bytes = weight.size();
    ensure!(
        total_bytes == out_dim * full_row_bytes,
        "Weight buffer size ({}) != out_dim*row_bytes ({})",
        total_bytes,
        out_dim * full_row_bytes,
    );

    // Build the GPU slice and CPU slice as fresh host buffers, one row at a
    // time. This is O(out_dim * row_bytes) memcpy on the host side, executed
    // once at model setup — decode path amortizes it over every token.
    let gpu_bytes = out_dim * gpu_row_bytes;
    let cpu_bytes = out_dim * cpu_row_bytes;

    // Allocate fresh CPU-owned host buffers via cpu_backend's memory arena.
    // We go through `Galloc` (host RAM) so `as_ptr()` works for the CPU path
    // and the GPU backend can later adopt these bytes via `backend.copy_from`.
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;

    let galloc = Galloc::new();
    let gpu_host_buf = galloc.alloc(gpu_bytes, dtype)?;
    let cpu_host_buf = galloc.alloc(cpu_bytes, dtype)?;

    // Safety: weight has a valid host pointer (callers pass CPU-accessible
    // tensors); gpu_host_buf / cpu_host_buf were just allocated as host RAM.
    unsafe {
        let src = weight.as_ptr();
        let gpu_dst = gpu_host_buf.as_mut_ptr();
        let cpu_dst = cpu_host_buf.as_mut_ptr();
        ensure!(
            !src.is_null() && !gpu_dst.is_null() && !cpu_dst.is_null(),
            "split_weight_col: null pointer (weight must be CPU-mapped)",
        );
        for r in 0..out_dim {
            let src_row = src.add(r * full_row_bytes);
            let gpu_row_dst = gpu_dst.add(r * gpu_row_bytes);
            let cpu_row_dst = cpu_dst.add(r * cpu_row_bytes);
            std::ptr::copy_nonoverlapping(src_row, gpu_row_dst, gpu_row_bytes);
            std::ptr::copy_nonoverlapping(src_row.add(gpu_row_bytes), cpu_row_dst, cpu_row_bytes);
        }
    }

    // CPU slice tensor: host buffer directly tagged with the CPU backend.
    let cpu_tensor = Tensor::new(
        Shape::new(vec![out_dim, in_dim - split_col]),
        cpu_host_buf,
        cpu_backend.clone(),
    );

    // GPU slice tensor: adopt the host buffer via backend.copy_from so GPU
    // backends upload it to a device-resident buffer. CPU backends will see
    // this as a no-op clone.
    // `copy_from` may tag the returned tensor with the source backend (see
    // OpenCLBackend::copy_from: uses src.backend().clone()), so we explicitly
    // retag with the weight's backend to restore the GPU ownership invariant.
    let gpu_host_tensor = Tensor::new(
        Shape::new(vec![out_dim, split_col]),
        gpu_host_buf,
        cpu_backend.clone(),
    );
    let copied = weight.backend().copy_from(&gpu_host_tensor)?;
    let gpu_tensor = Tensor::new(
        Shape::new(vec![out_dim, split_col]),
        copied.buffer().clone(),
        weight.backend().clone(),
    );

    debug!(
        "split_weight_col: [{}x{}] split_col={}, gpu_bytes={}, cpu_bytes={} (per-row copy)",
        out_dim, in_dim, split_col, gpu_bytes, cpu_bytes,
    );

    Ok(PartitionedWeight {
        // slices[0]=GPU, slices[1]=CPU 순서 고정 (gpu_slice()/cpu_slice() 접근자 계약)
        slices: vec![
            WeightSlice {
                tensor: gpu_tensor,
                backend: weight.backend().clone(),
                rows: split_col,
            },
            WeightSlice {
                tensor: cpu_tensor,
                backend: cpu_backend.clone(),
                rows: in_dim - split_col,
            },
        ],
        split_row: split_col, // field repurposed to carry the col-split
    })
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

// ─────────────────────────────────────────────────────────────────────────────
// Partition trace (env-gated)
//
// Set `LLMRS_PARTITION_TRACE=1` to enable per-layer timing of the partition
// decode path. Adds one extra `synchronize()` per layer (measurement cost
// ~variable), so use for diagnosis only, not production benchmarks.
//
// Prints a summary to stderr every 280 layer calls (≈ 10 tokens at 28 layers).
// ─────────────────────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};

static PART_SYNC_NS: AtomicU64 = AtomicU64::new(0);
static PART_READ_NS: AtomicU64 = AtomicU64::new(0);
static PART_CPU_NS: AtomicU64 = AtomicU64::new(0);
static PART_GPU_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static PART_MERGE_NS: AtomicU64 = AtomicU64::new(0);
static PART_LAYER_COUNT: AtomicU64 = AtomicU64::new(0);
const TRACE_SUMMARY_PERIOD: u64 = 28; // every decode token (28 layers)

// Path distribution counters.
static PART_ZCOPY_COUNT: AtomicU64 = AtomicU64::new(0);
static PART_ASYNC_COUNT: AtomicU64 = AtomicU64::new(0);
static PART_SYNC_PATH_COUNT: AtomicU64 = AtomicU64::new(0);

// Per-path sync_ns aggregates:
//   zcopy     — synchronize() with zero-copy residual (no DMA read).
//   non_zcopy — async_read or sync+read_buffer paths (both use DMA after sync).
static PART_SYNC_NS_ZCOPY: AtomicU64 = AtomicU64::new(0);
static PART_SYNC_NS_NONZCOPY: AtomicU64 = AtomicU64::new(0);

/// Which residual-transfer path was taken for a given partition layer call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionPath {
    /// Zero-copy path: residual already CPU-visible via mapped host pointer.
    Zcopy,
    /// Async DMA read path: non-blocking `enqueue_read_buffer_async` + deferred `wait_event`.
    AsyncRead,
    /// Synchronous path: blocking `synchronize()` followed by `read_buffer`.
    SyncRead,
}

pub fn partition_trace_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let enabled = std::env::var_os("LLMRS_PARTITION_TRACE").is_some();
        if enabled {
            println!("[partition-trace-init] enabled=true");
        }
        enabled
    })
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

/// Gate for routing tensor partition through the OpenCL plan path
/// (`backend/opencl/plan.rs::build_partitioned_layer_plan`). When disabled,
/// `build_plan` returns an error for layers with a `PartitionContext` and the
/// caller falls back to `forward_gen` — kept as a bit-exact comparison
/// backstop for the new plan path.
///
/// Default: **enabled** (2026-04-21 re-enable after lm_head dtype fix). The
/// plan parity regression that prompted a default-off flip in `49c0d91` was
/// rooted in the plan building an F16 GEMV step for an F32 `lm_head` (tied
/// tokens inherited from `token_embd.weight`, which GGUF Llama 3.2 1B stores
/// as F32). Fixed by gating the plan's lm_head matmul on the weight dtype and
/// falling back to `backend.matmul_transposed` when no GPU GEMV variant is
/// available. Set `LLMRS_PARTITION_PLAN=0` to force the legacy `forward_gen`
/// partition path (bit-exact comparison backstop).
///
/// The decision is cached at first read via `OnceLock` because plan builds
/// and hot dispatch must observe a single stable value across a generation
/// (otherwise a mid-loop flip would desynchronize builder and executor).
pub fn partition_plan_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LLMRS_PARTITION_PLAN")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// When enabled, `PartitionStep::run` skips the explicit `clFinish(queue)`
/// wait for `add_rms_norm_oop` and instead spin-polls an ALLOC_HOST_PTR
/// flag written by the `kernel_add_rms_norm_oop_f4_sigflag` variant. This
/// elides Adreno driver submission/completion latency (~0.29 ms/layer on
/// S25) at the cost of a busy-wait CPU loop scoped to that window.
///
/// Default: **enabled**. Disable with `LLMRS_PARTITION_POLL_FLAG=0` to fall
/// back to the `clFinish` drain on hardware where the sigflag path is slower
/// or ALLOC_HOST_PTR coherency is unreliable.
pub fn partition_poll_flag_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LLMRS_PARTITION_POLL_FLAG")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Diagnostic gate for the plan-path partition executor. When enabled, each
/// sub-step inside `PartitionStep::run` is followed by a blocking `clFinish`
/// and a stderr line with the sub-step label. Intended for debugging sync
/// ordering or residual propagation bugs when bit-exactness tests fail —
/// pairs with `PLAN_DEBUG` on the non-partition path.
///
/// Default: **disabled**. `LLMRS_PLAN_PARTITION_DEBUG=1` turns it on. Not
/// cached because the expectation is one-shot investigation.
pub fn partition_plan_debug_enabled() -> bool {
    std::env::var_os("LLMRS_PLAN_PARTITION_DEBUG").is_some()
}

pub fn record_partition_timing(
    sync_ns: u64,
    read_ns: u64,
    cpu_ns: u64,
    gpu_wait_ns: u64,
    merge_ns: u64,
    path: PartitionPath,
) {
    PART_SYNC_NS.fetch_add(sync_ns, Ordering::Relaxed);
    PART_READ_NS.fetch_add(read_ns, Ordering::Relaxed);
    PART_CPU_NS.fetch_add(cpu_ns, Ordering::Relaxed);
    PART_GPU_WAIT_NS.fetch_add(gpu_wait_ns, Ordering::Relaxed);
    PART_MERGE_NS.fetch_add(merge_ns, Ordering::Relaxed);

    // Update path distribution counters + per-path sync_ns aggregates.
    match path {
        PartitionPath::Zcopy => {
            PART_ZCOPY_COUNT.fetch_add(1, Ordering::Relaxed);
            PART_SYNC_NS_ZCOPY.fetch_add(sync_ns, Ordering::Relaxed);
        }
        PartitionPath::AsyncRead => {
            PART_ASYNC_COUNT.fetch_add(1, Ordering::Relaxed);
            PART_SYNC_NS_NONZCOPY.fetch_add(sync_ns, Ordering::Relaxed);
        }
        PartitionPath::SyncRead => {
            PART_SYNC_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            PART_SYNC_NS_NONZCOPY.fetch_add(sync_ns, Ordering::Relaxed);
        }
    }

    let count = PART_LAYER_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if count.is_multiple_of(TRACE_SUMMARY_PERIOD) {
        print_partition_trace_summary(count);
    }
}

pub fn print_partition_trace_summary(count: u64) {
    if count == 0 {
        return;
    }
    let sync = PART_SYNC_NS.load(Ordering::Relaxed) as f64 / count as f64 / 1e6;
    let read = PART_READ_NS.load(Ordering::Relaxed) as f64 / count as f64 / 1e6;
    let cpu = PART_CPU_NS.load(Ordering::Relaxed) as f64 / count as f64 / 1e6;
    let gpu_wait = PART_GPU_WAIT_NS.load(Ordering::Relaxed) as f64 / count as f64 / 1e6;
    let merge = PART_MERGE_NS.load(Ordering::Relaxed) as f64 / count as f64 / 1e6;
    // parallel efficiency: if gpu_wait ≈ 0, CPU was the longer phase (GPU done waiting = idle)
    // if gpu_wait >> 0, GPU was still running when CPU done (GPU-bottleneck)
    let bottleneck = if gpu_wait < 0.05 {
        "CPU (GPU already idle when CPU finished)"
    } else if gpu_wait < cpu * 0.2 {
        "balanced"
    } else {
        "GPU (CPU finished well before GPU)"
    };
    println!(
        "[partition-trace] layers={} avg/layer: sync_drain={:.2}ms dma_read={:.2}ms cpu_matmul={:.2}ms gpu_wait={:.2}ms merge={:.2}ms — bottleneck: {}",
        count, sync, read, cpu, gpu_wait, merge, bottleneck
    );

    // Path distribution + per-path sync_ns averages.
    let zcopy_n = PART_ZCOPY_COUNT.load(Ordering::Relaxed);
    let async_n = PART_ASYNC_COUNT.load(Ordering::Relaxed);
    let sync_read_n = PART_SYNC_PATH_COUNT.load(Ordering::Relaxed);
    let zcopy_avg = if zcopy_n > 0 {
        format!(
            "{:.2}ms",
            PART_SYNC_NS_ZCOPY.load(Ordering::Relaxed) as f64 / zcopy_n as f64 / 1e6
        )
    } else {
        "n/a".to_string()
    };
    let nonzcopy_n = async_n + sync_read_n;
    let nonzcopy_avg = if nonzcopy_n > 0 {
        format!(
            "{:.2}ms",
            PART_SYNC_NS_NONZCOPY.load(Ordering::Relaxed) as f64 / nonzcopy_n as f64 / 1e6
        )
    } else {
        "n/a".to_string()
    };
    println!(
        "[partition-trace] path dist: zcopy={} async={} sync_read={} — sync_ns zcopy={} non_zcopy={}",
        zcopy_n, async_n, sync_read_n, zcopy_avg, nonzcopy_avg
    );

    use std::io::Write;
    let _ = std::io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;
    use crate::quant::{BlockQ4_0, QK4_0};

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

    /// Helper: allocate a Q4_0 weight tensor [out_dim, in_dim].
    /// Fills with quantized data from sequential float values.
    fn make_q4_weight(out_dim: usize, in_dim: usize) -> Tensor {
        assert!(
            in_dim.is_multiple_of(QK4_0),
            "in_dim must be divisible by 32 for Q4_0"
        );
        let blocks_per_row = in_dim / QK4_0;
        let total_blocks = out_dim * blocks_per_row;
        let total_bytes = total_blocks * std::mem::size_of::<BlockQ4_0>();

        let memory = Galloc::new();
        let buf = memory.alloc(total_bytes, DType::Q4_0).unwrap();
        let tensor = Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, cpu_backend());

        // Fill with quantized data
        let blocks = unsafe {
            std::slice::from_raw_parts_mut(tensor.as_mut_ptr() as *mut BlockQ4_0, total_blocks)
        };
        for (bi, block) in blocks.iter_mut().enumerate() {
            let mut src = [0.0f32; QK4_0];
            for (j, v) in src.iter_mut().enumerate() {
                *v = ((bi * QK4_0 + j) as f32) * 0.01;
            }
            *block = BlockQ4_0::quantize(&src);
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

    // PA-T1-06: Q4_0 alignment — split_row is a multiple of 128.
    #[test]
    fn test_q4_alignment() {
        let w = make_q4_weight(5504, 1536);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.5, &cpu).unwrap();
        assert_eq!(pw.split_row % ROW_ALIGNMENT, 0);
        assert!(pw.split_row >= ROW_ALIGNMENT);
        assert!(pw.split_row <= 5504 - ROW_ALIGNMENT);
    }

    // PA-T1-07: F16 split total — gpu + cpu rows == out_dim.
    // We use F32 here because F16 allocation is identical in shape logic.
    #[test]
    fn test_split_total_rows() {
        let w = make_f32_weight(5504, 1536);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.7, &cpu).unwrap();
        let gpu_rows = pw.gpu_slice().shape().dims()[0];
        let cpu_rows = pw.cpu_slice().shape().dims()[0];
        assert_eq!(gpu_rows + cpu_rows, 5504);
        assert_eq!(gpu_rows, pw.split_row);
    }

    // PA-T1-08: split -> concat roundtrip for Q4_0.
    // Verifies that gpu_slice bytes + cpu_slice bytes == original weight bytes.
    #[test]
    fn test_q4_split_concat_roundtrip() {
        let w = make_q4_weight(5504, 1536);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.6, &cpu).unwrap();

        let gpu_size = pw.gpu_slice().size();
        let cpu_size = pw.cpu_slice().size();
        assert_eq!(gpu_size + cpu_size, w.size());

        // Verify data content matches original weight bytes (pre-copy creates independent buffers).
        let orig = unsafe { std::slice::from_raw_parts(w.as_ptr(), w.size()) };
        let gpu_data = unsafe { std::slice::from_raw_parts(pw.gpu_slice().as_ptr(), gpu_size) };
        let cpu_data = unsafe { std::slice::from_raw_parts(pw.cpu_slice().as_ptr(), cpu_size) };
        assert_eq!(gpu_data, &orig[..gpu_size]);
        assert_eq!(cpu_data, &orig[gpu_size..]);
    }

    // PA-T1-09: split -> concat roundtrip for F32.
    #[test]
    fn test_f32_split_concat_roundtrip() {
        let w = make_f32_weight(1024, 512);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.5, &cpu).unwrap();

        assert_eq!(pw.gpu_slice().size() + pw.cpu_slice().size(), w.size());

        // Verify data integrity: read a value from the cpu_slice region
        let split_row = pw.split_row;
        let in_dim = 512;
        let expected_idx = split_row * in_dim; // first element of cpu region
        let expected_val = expected_idx as f32 * 0.001;
        let cpu_data = pw.cpu_slice().as_slice::<f32>();
        assert!(
            (cpu_data[0] - expected_val).abs() < 1e-6,
            "expected {}, got {}",
            expected_val,
            cpu_data[0]
        );
    }

    // PA-T1-10: partitioned matmul F32 accuracy (CPU-only).
    // Compare full matmul vs partitioned (gpu_part + cpu_part) concat.
    #[test]
    fn test_partitioned_matmul_f32() {
        let backend = cpu_backend();
        let memory = Galloc::new();

        let out_dim = 512;
        let in_dim = 256;

        // Weight [out_dim, in_dim]
        let w = make_f32_weight(out_dim, in_dim);

        // Input x [1, 1, in_dim]
        let x_buf = memory.alloc(in_dim * 4, DType::F32).unwrap();
        let mut x = Tensor::new(Shape::new(vec![1, 1, in_dim]), x_buf, backend.clone());
        let x_data = x.as_mut_slice::<f32>();
        for (i, v) in x_data.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }

        // Full matmul: out_full = x * W^T => [1, 1, out_dim]
        let out_buf = memory.alloc(out_dim * 4, DType::F32).unwrap();
        let mut out_full = Tensor::new(Shape::new(vec![1, 1, out_dim]), out_buf, backend.clone());
        backend.matmul_transposed(&x, &w, &mut out_full).unwrap();

        // Partitioned matmul
        let cpu2 = cpu_backend(); // second "cpu" acting as cpu_backend
        let pw = split_weight(&w, 0.5, &cpu2).unwrap();
        let split_row = pw.split_row;

        // GPU part: out_gpu = x * W_gpu^T => [1, 1, split_row]
        let gpu_buf = memory.alloc(split_row * 4, DType::F32).unwrap();
        let mut out_gpu = Tensor::new(Shape::new(vec![1, 1, split_row]), gpu_buf, backend.clone());
        backend
            .matmul_transposed(&x, pw.gpu_slice(), &mut out_gpu)
            .unwrap();

        // CPU part: out_cpu = x * W_cpu^T => [1, 1, out_dim - split_row]
        let cpu_rows = out_dim - split_row;
        let cpu_buf = memory.alloc(cpu_rows * 4, DType::F32).unwrap();
        let mut out_cpu = Tensor::new(Shape::new(vec![1, 1, cpu_rows]), cpu_buf, cpu2.clone());
        cpu2.matmul_transposed(&x, pw.cpu_slice(), &mut out_cpu)
            .unwrap();

        // Compare: concat(out_gpu, out_cpu) == out_full
        let full_data = out_full.as_slice::<f32>();
        let gpu_data = out_gpu.as_slice::<f32>();
        let cpu_data = out_cpu.as_slice::<f32>();

        // Tolerance: relative error < 1e-4 or absolute error < 0.01.
        // F32 matmul with parallel summation can produce different rounding order.
        for i in 0..split_row {
            let diff = (full_data[i] - gpu_data[i]).abs();
            let rel = if full_data[i].abs() > 1e-6 {
                diff / full_data[i].abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-4 || diff < 0.01,
                "GPU mismatch at [{}]: full={}, gpu={}, diff={}, rel={}",
                i,
                full_data[i],
                gpu_data[i],
                diff,
                rel,
            );
        }
        for i in 0..cpu_rows {
            let diff = (full_data[split_row + i] - cpu_data[i]).abs();
            let rel = if full_data[split_row + i].abs() > 1e-6 {
                diff / full_data[split_row + i].abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-4 || diff < 0.01,
                "CPU mismatch at [{}]: full={}, cpu={}, diff={}, rel={}",
                i,
                full_data[split_row + i],
                cpu_data[i],
                diff,
                rel,
            );
        }
    }

    // PA-T1-11: partitioned matmul Q4_0 accuracy (CPU-only).
    #[test]
    fn test_partitioned_matmul_q4_0() {
        let backend = cpu_backend();
        let memory = Galloc::new();

        let out_dim = 512;
        let in_dim = 256; // divisible by 32

        let w = make_q4_weight(out_dim, in_dim);

        // Input x [1, 1, in_dim] F32
        let x_buf = memory.alloc(in_dim * 4, DType::F32).unwrap();
        let mut x = Tensor::new(Shape::new(vec![1, 1, in_dim]), x_buf, backend.clone());
        let x_data = x.as_mut_slice::<f32>();
        for (i, v) in x_data.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }

        // Full matmul
        let out_buf = memory.alloc(out_dim * 4, DType::F32).unwrap();
        let mut out_full = Tensor::new(Shape::new(vec![1, 1, out_dim]), out_buf, backend.clone());
        backend.matmul_transposed(&x, &w, &mut out_full).unwrap();

        // Partitioned
        let cpu2 = cpu_backend();
        let pw = split_weight(&w, 0.5, &cpu2).unwrap();
        let split_row = pw.split_row;
        let cpu_rows = out_dim - split_row;

        let gpu_buf = memory.alloc(split_row * 4, DType::F32).unwrap();
        let mut out_gpu = Tensor::new(Shape::new(vec![1, 1, split_row]), gpu_buf, backend.clone());
        backend
            .matmul_transposed(&x, pw.gpu_slice(), &mut out_gpu)
            .unwrap();

        let cpu_buf = memory.alloc(cpu_rows * 4, DType::F32).unwrap();
        let mut out_cpu = Tensor::new(Shape::new(vec![1, 1, cpu_rows]), cpu_buf, cpu2.clone());
        cpu2.matmul_transposed(&x, pw.cpu_slice(), &mut out_cpu)
            .unwrap();

        // Compare
        let full_data = out_full.as_slice::<f32>();
        let gpu_data = out_gpu.as_slice::<f32>();
        let cpu_data = out_cpu.as_slice::<f32>();

        // Q4_0 accumulates more rounding error from quantized dot products.
        // Use relative tolerance of 1e-3 or absolute tolerance of 1.0.
        for i in 0..split_row {
            let diff = (full_data[i] - gpu_data[i]).abs();
            let rel = if full_data[i].abs() > 1e-6 {
                diff / full_data[i].abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-3 || diff < 1.0,
                "Q4_0 GPU mismatch at [{}]: full={}, gpu={}, diff={}, rel={}",
                i,
                full_data[i],
                gpu_data[i],
                diff,
                rel,
            );
        }
        for i in 0..cpu_rows {
            let diff = (full_data[split_row + i] - cpu_data[i]).abs();
            let rel = if full_data[split_row + i].abs() > 1e-6 {
                diff / full_data[split_row + i].abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-3 || diff < 1.0,
                "Q4_0 CPU mismatch at [{}]: full={}, cpu={}, diff={}, rel={}",
                i,
                full_data[split_row + i],
                cpu_data[i],
                diff,
                rel,
            );
        }
    }

    // PA-T1-12: workspace size calculation for partitioned output.
    #[test]
    fn test_workspace_sizes() {
        let w = make_f32_weight(5504, 1536);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.6, &cpu).unwrap();

        let split_row = pw.split_row;
        let cpu_rows = 5504 - split_row;

        // GPU output workspace: split_row * sizeof(f32)
        let gpu_ws_bytes = split_row * 4;
        // CPU output workspace: cpu_rows * sizeof(f32)
        let cpu_ws_bytes = cpu_rows * 4;
        // Total should equal full output size
        assert_eq!(gpu_ws_bytes + cpu_ws_bytes, 5504 * 4);
    }

    // PA-T1-13: extreme ratios (very low and very high).
    #[test]
    fn test_extreme_ratios() {
        let w = make_f32_weight(1024, 256);
        let cpu = cpu_backend();

        // Very low ratio: should clamp split_row to ROW_ALIGNMENT (128)
        let pw_low = split_weight(&w, 0.01, &cpu).unwrap();
        assert_eq!(pw_low.split_row, ROW_ALIGNMENT);
        assert_eq!(
            pw_low.gpu_slice().shape().dims()[0] + pw_low.cpu_slice().shape().dims()[0],
            1024
        );

        // Very high ratio: should clamp split_row to out_dim - ROW_ALIGNMENT
        let pw_high = split_weight(&w, 0.99, &cpu).unwrap();
        assert_eq!(pw_high.split_row, 1024 - ROW_ALIGNMENT);
        assert_eq!(
            pw_high.gpu_slice().shape().dims()[0] + pw_high.cpu_slice().shape().dims()[0],
            1024
        );
    }

    // Edge case: out_dim exactly 256 (minimum viable).
    #[test]
    fn test_minimum_out_dim() {
        let w = make_f32_weight(256, 128);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.5, &cpu).unwrap();
        assert_eq!(pw.split_row, 128);
        assert_eq!(pw.gpu_slice().shape().dims(), &[128, 128]);
        assert_eq!(pw.cpu_slice().shape().dims(), &[128, 128]);
    }

    // Error case: out_dim too small for partitioning.
    #[test]
    fn test_too_small_out_dim() {
        let memory = Galloc::new();
        let buf = memory.alloc(128 * 64 * 4, DType::F32).unwrap();
        let w = Tensor::new(Shape::new(vec![128, 64]), buf, cpu_backend());
        let cpu = cpu_backend();
        let result = split_weight(&w, 0.5, &cpu);
        assert!(result.is_err());
    }

    // Error case: 1D tensor.
    #[test]
    fn test_1d_tensor_error() {
        let memory = Galloc::new();
        let buf = memory.alloc(1024 * 4, DType::F32).unwrap();
        let w = Tensor::new(Shape::new(vec![1024]), buf, cpu_backend());
        let cpu = cpu_backend();
        let result = split_weight(&w, 0.5, &cpu);
        assert!(result.is_err());
    }

    // ── split_weight_col tests (Strategy B: whole-FFN slice) ──

    // Byte layout: each row is contiguous in memory. col-split must preserve
    // each row's first `split_col` elements in the GPU slice and the rest in
    // the CPU slice, for every row.
    #[test]
    fn test_split_weight_col_f32_roundtrip() {
        let out_dim = 512;
        let in_dim = 256;
        let split_col = 128;
        let w = make_f32_weight(out_dim, in_dim);
        let cpu = cpu_backend();

        let pw = split_weight_col(&w, split_col, &cpu).unwrap();
        assert_eq!(pw.split_row, split_col); // repurposed field
        assert_eq!(pw.gpu_slice().shape().dims(), &[out_dim, split_col]);
        assert_eq!(
            pw.cpu_slice().shape().dims(),
            &[out_dim, in_dim - split_col]
        );

        // Verify byte-exact data for every row.
        let orig = w.as_slice::<f32>();
        let gpu_data = pw.gpu_slice().as_slice::<f32>();
        let cpu_data = pw.cpu_slice().as_slice::<f32>();
        let cpu_cols = in_dim - split_col;
        for r in 0..out_dim {
            for c in 0..split_col {
                let expected = orig[r * in_dim + c];
                let actual = gpu_data[r * split_col + c];
                assert_eq!(actual, expected, "GPU row {} col {}", r, c);
            }
            for c in 0..cpu_cols {
                let expected = orig[r * in_dim + split_col + c];
                let actual = cpu_data[r * cpu_cols + c];
                assert_eq!(actual, expected, "CPU row {} col {}", r, c);
            }
        }
    }

    // Q4_0 col-split: split_col must be a multiple of 32 (block size) to keep
    // whole blocks intact per row. 128 satisfies both the 32-block and 128
    // workgroup alignment.
    #[test]
    fn test_split_weight_col_q4_0_roundtrip() {
        let out_dim = 256;
        let in_dim = 512; // 16 Q4_0 blocks per row
        let split_col = 128; // 4 blocks GPU-side, 12 blocks CPU-side
        let w = make_q4_weight(out_dim, in_dim);
        let cpu = cpu_backend();

        let pw = split_weight_col(&w, split_col, &cpu).unwrap();
        let gpu_data =
            unsafe { std::slice::from_raw_parts(pw.gpu_slice().as_ptr(), pw.gpu_slice().size()) };
        let cpu_data =
            unsafe { std::slice::from_raw_parts(pw.cpu_slice().as_ptr(), pw.cpu_slice().size()) };
        let orig = unsafe { std::slice::from_raw_parts(w.as_ptr(), w.size()) };

        let row_bytes = in_dim / 32 * 18;
        let gpu_row_bytes = split_col / 32 * 18;
        let cpu_row_bytes = (in_dim - split_col) / 32 * 18;
        assert_eq!(pw.gpu_slice().size(), out_dim * gpu_row_bytes);
        assert_eq!(pw.cpu_slice().size(), out_dim * cpu_row_bytes);

        for r in 0..out_dim {
            let orig_row = &orig[r * row_bytes..(r + 1) * row_bytes];
            let gpu_row = &gpu_data[r * gpu_row_bytes..(r + 1) * gpu_row_bytes];
            let cpu_row = &cpu_data[r * cpu_row_bytes..(r + 1) * cpu_row_bytes];
            assert_eq!(gpu_row, &orig_row[..gpu_row_bytes], "GPU row {}", r);
            assert_eq!(cpu_row, &orig_row[gpu_row_bytes..], "CPU row {}", r);
        }
    }

    // End-to-end: col-split partitioned matmul must match the full matmul.
    // Input must be col-split alongside the weight so CPU and GPU dot products
    // both sum over their own slice of the inner dimension.
    #[test]
    fn test_split_weight_col_matmul_f32() {
        let backend = cpu_backend();
        let memory = Galloc::new();
        let out_dim = 512;
        let in_dim = 256;
        let split_col = 128;

        let w = make_f32_weight(out_dim, in_dim);
        let x_buf = memory.alloc(in_dim * 4, DType::F32).unwrap();
        let mut x = Tensor::new(Shape::new(vec![1, 1, in_dim]), x_buf, backend.clone());
        for (i, v) in x.as_mut_slice::<f32>().iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }

        // Reference: full matmul.
        let out_buf = memory.alloc(out_dim * 4, DType::F32).unwrap();
        let mut out_full = Tensor::new(Shape::new(vec![1, 1, out_dim]), out_buf, backend.clone());
        backend.matmul_transposed(&x, &w, &mut out_full).unwrap();

        // Partition: gpu handles input[:split_col] * w[:, :split_col],
        // cpu handles input[split_col:] * w[:, split_col:], then sum.
        let cpu2 = cpu_backend();
        let pw = split_weight_col(&w, split_col, &cpu2).unwrap();

        // x_gpu: fresh [1, 1, split_col] copy of x[0..split_col].
        let xg_buf = memory.alloc(split_col * 4, DType::F32).unwrap();
        let mut x_gpu = Tensor::new(Shape::new(vec![1, 1, split_col]), xg_buf, backend.clone());
        x_gpu
            .as_mut_slice::<f32>()
            .copy_from_slice(&x.as_slice::<f32>()[..split_col]);

        // x_cpu: fresh [1, 1, in_dim-split_col] copy of x[split_col..].
        let cpu_cols = in_dim - split_col;
        let xc_buf = memory.alloc(cpu_cols * 4, DType::F32).unwrap();
        let mut x_cpu = Tensor::new(Shape::new(vec![1, 1, cpu_cols]), xc_buf, cpu2.clone());
        x_cpu
            .as_mut_slice::<f32>()
            .copy_from_slice(&x.as_slice::<f32>()[split_col..]);

        let gpu_out_buf = memory.alloc(out_dim * 4, DType::F32).unwrap();
        let mut out_gpu = Tensor::new(
            Shape::new(vec![1, 1, out_dim]),
            gpu_out_buf,
            backend.clone(),
        );
        backend
            .matmul_transposed(&x_gpu, pw.gpu_slice(), &mut out_gpu)
            .unwrap();

        let cpu_out_buf = memory.alloc(out_dim * 4, DType::F32).unwrap();
        let mut out_cpu = Tensor::new(Shape::new(vec![1, 1, out_dim]), cpu_out_buf, cpu2.clone());
        cpu2.matmul_transposed(&x_cpu, pw.cpu_slice(), &mut out_cpu)
            .unwrap();

        // Sum: full ≈ gpu_partial + cpu_partial (elementwise on out_dim).
        let full = out_full.as_slice::<f32>();
        let g = out_gpu.as_slice::<f32>();
        let c = out_cpu.as_slice::<f32>();
        for i in 0..out_dim {
            let combined = g[i] + c[i];
            let diff = (full[i] - combined).abs();
            let rel = if full[i].abs() > 1e-6 {
                diff / full[i].abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-4 || diff < 0.01,
                "mismatch at [{}]: full={}, combined={}, diff={}, rel={}",
                i,
                full[i],
                combined,
                diff,
                rel,
            );
        }
    }

    #[test]
    fn test_split_weight_col_invalid_alignment() {
        let w = make_f32_weight(256, 256);
        let cpu = cpu_backend();
        // 64 is not a multiple of ROW_ALIGNMENT (128).
        assert!(split_weight_col(&w, 64, &cpu).is_err());
    }

    #[test]
    fn test_split_weight_col_bounds() {
        let w = make_f32_weight(256, 256);
        let cpu = cpu_backend();
        assert!(split_weight_col(&w, 0, &cpu).is_err());
        assert!(split_weight_col(&w, 256, &cpu).is_err());
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

    // PA-T1-PATHCNT: record_partition_timing increments the correct path counter
    // and accumulates sync_ns into the correct per-path aggregate.
    #[test]
    fn test_record_partition_timing_path_counters() {
        // Snapshot counters before the test calls.
        let zcopy_before = PART_ZCOPY_COUNT.load(Ordering::Relaxed);
        let async_before = PART_ASYNC_COUNT.load(Ordering::Relaxed);
        let sync_before = PART_SYNC_PATH_COUNT.load(Ordering::Relaxed);
        let zcopy_ns_before = PART_SYNC_NS_ZCOPY.load(Ordering::Relaxed);
        let nonzcopy_ns_before = PART_SYNC_NS_NONZCOPY.load(Ordering::Relaxed);
        let layer_before = PART_LAYER_COUNT.load(Ordering::Relaxed);

        // Call with Zcopy path: sync_ns=1000, rest 0.
        record_partition_timing(1_000, 0, 0, 0, 0, PartitionPath::Zcopy);
        assert_eq!(
            PART_ZCOPY_COUNT.load(Ordering::Relaxed),
            zcopy_before + 1,
            "Zcopy counter should increment by 1"
        );
        assert_eq!(
            PART_ASYNC_COUNT.load(Ordering::Relaxed),
            async_before,
            "AsyncRead counter must not change for Zcopy call"
        );
        assert_eq!(
            PART_SYNC_PATH_COUNT.load(Ordering::Relaxed),
            sync_before,
            "SyncRead counter must not change for Zcopy call"
        );
        assert_eq!(
            PART_SYNC_NS_ZCOPY.load(Ordering::Relaxed),
            zcopy_ns_before + 1_000,
            "zcopy sync_ns aggregate should increase by 1000"
        );
        assert_eq!(
            PART_SYNC_NS_NONZCOPY.load(Ordering::Relaxed),
            nonzcopy_ns_before,
            "non_zcopy sync_ns aggregate must not change for Zcopy call"
        );

        // Snapshot again before AsyncRead call.
        let zcopy_before2 = PART_ZCOPY_COUNT.load(Ordering::Relaxed);
        let async_before2 = PART_ASYNC_COUNT.load(Ordering::Relaxed);
        let nonzcopy_ns_before2 = PART_SYNC_NS_NONZCOPY.load(Ordering::Relaxed);

        // Call with AsyncRead path: sync_ns=500.
        record_partition_timing(500, 200, 0, 0, 0, PartitionPath::AsyncRead);
        assert_eq!(
            PART_ASYNC_COUNT.load(Ordering::Relaxed),
            async_before2 + 1,
            "AsyncRead counter should increment by 1"
        );
        assert_eq!(
            PART_ZCOPY_COUNT.load(Ordering::Relaxed),
            zcopy_before2,
            "Zcopy counter must not change for AsyncRead call"
        );
        assert_eq!(
            PART_SYNC_NS_NONZCOPY.load(Ordering::Relaxed),
            nonzcopy_ns_before2 + 500,
            "non_zcopy sync_ns aggregate should increase by 500 for AsyncRead"
        );

        // Snapshot again before SyncRead call.
        let sync_before3 = PART_SYNC_PATH_COUNT.load(Ordering::Relaxed);
        let nonzcopy_ns_before3 = PART_SYNC_NS_NONZCOPY.load(Ordering::Relaxed);
        let layer_before3 = PART_LAYER_COUNT.load(Ordering::Relaxed);

        // Call with SyncRead path: sync_ns=2000.
        record_partition_timing(2_000, 300, 100, 50, 10, PartitionPath::SyncRead);
        assert_eq!(
            PART_SYNC_PATH_COUNT.load(Ordering::Relaxed),
            sync_before3 + 1,
            "SyncRead counter should increment by 1"
        );
        assert_eq!(
            PART_SYNC_NS_NONZCOPY.load(Ordering::Relaxed),
            nonzcopy_ns_before3 + 2_000,
            "non_zcopy sync_ns aggregate should increase by 2000 for SyncRead"
        );
        assert_eq!(
            PART_LAYER_COUNT.load(Ordering::Relaxed),
            layer_before3 + 1,
            "layer count should increment by 1"
        );

        // Verify the overall layer count increased by 3 across all three calls.
        assert_eq!(
            PART_LAYER_COUNT.load(Ordering::Relaxed),
            layer_before + 3,
            "total layer count should have increased by 3"
        );
    }

    // ── AB-4 (C): apply_partition_dispatch fan-out 등가 ──

    use crate::models::weights::LayerSlot;

    fn ffn_layer_slot(be: &Arc<dyn Backend>, idx: usize) -> Arc<LayerSlot> {
        use crate::layers::transformer_layer::TransformerLayer;
        // FFN weights large enough for split_weight (out_dim >= 256, 128-aligned);
        // attention weights are never partitioned, so small placeholders suffice.
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
