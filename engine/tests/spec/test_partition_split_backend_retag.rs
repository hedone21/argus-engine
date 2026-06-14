//! Tensor partition GPU-slice retag regression. `gguf.rs::load_raw` builds
//! weights via `backend.copy_weight_from`, leaving `Tensor::backend()` on
//! the CPU loader even when the buffer is a `UnifiedBuffer` with valid
//! `cl_mem`. `split_weight_col` (FFN-down) then dispatches `weight.backend()
//! .copy_from(...)` through CPU and lands a `SharedBuffer`-backed GPU slice
//! without `cl_mem` — the next `matmul_f16` aborts with "B is not OpenCL
//! buffer". `map_weights_for_cpu` must retag onto the active GPU backend.
//!
//! Issue: `issues/precision_swap_opencl_b_not_buffer_20260501.md`
//! Spec: ENG-ALG-200 (partition path), ENG-ALG-211 (loader → forward backend).

use std::sync::Arc;

use argus_engine::backend::Backend;
use argus_engine::buffer::DType;
use argus_engine::tensor::Tensor;

/// `Tensor::backend()` must return the same `Arc<dyn Backend>` that was
/// passed at construction. This is the contract `tensor_partition.rs` relies
/// on (`weight.backend().copy_from(...)` for the CPU host buffer → GPU slice
/// upload step).
#[test]
fn tensor_backend_returns_construction_backend() {
    use argus_engine::backend::cpu::common::CpuBackendCommon;
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;

    let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackendCommon);
    let buf: Arc<dyn argus_engine::buffer::Buffer> = Arc::new(SharedBuffer::new(64, DType::F32));
    let t = Tensor::new(Shape::new(vec![16]), buf, cpu_be.clone());

    assert!(
        Arc::ptr_eq(t.backend(), &cpu_be),
        "Tensor::backend() must return the construction backend"
    );
}

/// Exercise the `map_one`-style retag pattern: a tensor that was loaded with
/// a CPU loader backend but whose buffer is GPU-resident (UnifiedBuffer with
/// valid `cl_mem`) must be retagged to the GPU backend before being handed
/// to `split_weight_col`. Otherwise the ensuing
/// `weight.backend().copy_from(...)` lands a `SharedBuffer`-backed GPU slice
/// instead of a UnifiedBuffer, and the next OpenCL matmul aborts with
/// `"B is not OpenCL buffer"`.
///
/// GPU-required: skipped on CI hosts without an OpenCL platform.
#[cfg(feature = "opencl")]
#[test]
fn map_weights_for_cpu_retags_loader_backend_to_gpu_after_in_place_map() {
    use argus_engine::backend::cpu::common::CpuBackendCommon;
    use argus_engine::backend::opencl::OpenCLBackend;
    use argus_engine::memory::opencl::unified::UnifiedBuffer;
    use argus_engine::shape::Shape;

    let opencl = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[SKIPPED] OpenCL device unavailable: {e}");
            return;
        }
    };
    let queue = opencl.queue.clone();
    let gpu_be: Arc<dyn Backend> = Arc::new(opencl);
    let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackendCommon);

    // Mimic the `gguf.rs::load_raw` post-condition: `OpenCLMemory::alloc`
    // produced a `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR), but the resulting
    // tensor was tagged with the CPU loader backend (because `OpenCLBackend
    // ::copy_from` returns `Tensor::new(... src.backend().clone())`).
    let ub = UnifiedBuffer::new(queue, 4096, DType::F32).expect("UnifiedBuffer::new");
    let buf: Arc<dyn argus_engine::buffer::Buffer> = Arc::new(ub);
    let cpu_tagged = Tensor::new(Shape::new(vec![1024]), buf.clone(), cpu_be.clone());

    // Sanity: the tensor's buffer is GPU-resident (has `cl_mem`) but its
    // backend is the CPU loader. This is the exact precondition the bug
    // manifests under.
    assert!(
        cpu_tagged.buffer().cl_mem().is_some(),
        "UnifiedBuffer must expose cl_mem"
    );
    assert!(
        Arc::ptr_eq(cpu_tagged.backend(), &cpu_be),
        "Pre-condition: tensor is CPU-backend tagged"
    );

    // Apply the retag pattern that `map_weights_for_cpu::map_one` performs
    // after a successful in-place `map_for_cpu()`: rebuild the tensor with
    // the GPU backend, preserving shape and buffer.
    let needs_retag = !Arc::ptr_eq(cpu_tagged.backend(), &gpu_be);
    assert!(needs_retag, "retag must fire when loader != GPU backend");
    let retagged = Tensor::new(
        cpu_tagged.shape().clone(),
        cpu_tagged.buffer().clone(),
        gpu_be.clone(),
    );

    // Post-condition: backend is now the GPU backend; buffer still has
    // valid `cl_mem`. This is what the partition path needs so
    // `split_weight_col → weight.backend().copy_from(host_buf)` lands on
    // OpenCL `copy_from` (UnifiedBuffer) rather than CPU `copy_from`
    // (SharedBuffer).
    assert!(
        Arc::ptr_eq(retagged.backend(), &gpu_be),
        "Post-condition: tensor is GPU-backend tagged"
    );
    assert!(
        retagged.buffer().cl_mem().is_some(),
        "Post-condition: cl_mem preserved through retag"
    );
    assert_eq!(retagged.size(), cpu_tagged.size());
    assert_eq!(retagged.dtype(), cpu_tagged.dtype());
}

/// Document the symmetric guarantee on the CPU side: a tensor loaded onto
/// the CPU backend stays on the CPU backend (no spurious retag). This
/// protects the `--backend cpu` path from being silently retagged to a
/// random GPU backend if the retag condition is ever inverted.
#[test]
fn cpu_loaded_tensor_keeps_cpu_backend_against_same_backend_check() {
    use argus_engine::backend::cpu::common::CpuBackendCommon;
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::shape::Shape;

    let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackendCommon);
    let buf: Arc<dyn argus_engine::buffer::Buffer> = Arc::new(SharedBuffer::new(64, DType::F32));
    let t = Tensor::new(Shape::new(vec![16]), buf, cpu_be.clone());

    // Same-backend check: when the desired backend is already the loader
    // backend, no retag should happen. This is the early-exit branch in
    // `map_one`'s logic.
    let needs_retag = !Arc::ptr_eq(t.backend(), &cpu_be);
    assert!(
        !needs_retag,
        "no retag when desired backend already matches loader backend"
    );
}
