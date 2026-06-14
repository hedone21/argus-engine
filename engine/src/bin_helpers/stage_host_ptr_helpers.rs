//! Shared helpers for Stage 1/2 Direction A microbenches.
//!
//! These helpers wrap the raw OpenCL `clCreateBuffer` flag combinations that
//! Direction A explores (USE_HOST_PTR, ALLOC_HOST_PTR, staging baseline) and
//! the canonical map/memcpy/unmap fill cycle for host-pinned cl_mem.
//!
//! Imported by `engine/src/bin/stage1_host_ptr_microbench.rs` and
//! `engine/src/bin/stage2_pool_stability.rs` via
//! `#[path = "../bin_helpers/stage_host_ptr_helpers.rs"] mod ...;` to satisfy
//! the "no duplicate helper definitions" constraint of the Stage 2 plan.
//!
//! Located outside `src/bin/` so that Cargo's binary auto-discovery does not
//! attempt to compile this file as a standalone executable.
//!
//! Plan ref: compiled-chasing-hopper Direction A track (Stage 1b → Stage 2).
//!
//! Callers must guard the `mod ...` declaration with
//! `#[cfg(feature = "opencl")]` (this file does not have an inner cfg of its
//! own to avoid clippy's `duplicated_attributes` lint).

use anyhow::Result;

/// Build a Q4_0 GEMV kernel (`kernel_mul_mat_q4_0_f32`) reusing the backend's
/// already-built `q4_0_program`. A fresh kernel object is created so that
/// argument-set state cannot bleed into the engine's cached one.
pub fn build_q4_0_kernel(
    backend: &argus_engine::backend::opencl::OpenCLBackend,
) -> Result<ocl::core::Kernel> {
    let kernel = ocl::core::create_kernel(&backend.q4_0_program, "kernel_mul_mat_q4_0_f32")?;
    Ok(kernel)
}

/// Path 0 (baseline) — `clCreateBuffer(MEM_READ_ONLY) + clEnqueueWriteBuffer`.
/// Mirrors the production `copy_weight_from` flow.
pub fn build_buffer_staging(
    backend: &argus_engine::backend::opencl::OpenCLBackend,
    weights: &[u8],
    offset: usize,
    size: usize,
) -> Result<ocl::core::Mem> {
    let mem = unsafe {
        ocl::core::create_buffer::<_, u8>(
            backend.context.as_core(),
            ocl::core::MEM_READ_ONLY,
            size,
            None,
        )?
    };
    unsafe {
        ocl::core::enqueue_write_buffer(
            &backend.queue,
            &mem,
            false, // non-blocking; matches engine's write path
            0,
            &weights[offset..offset + size],
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    Ok(mem)
}

/// Path 4 alloc step — `clCreateBuffer(ALLOC_HOST_PTR | READ_ONLY)` with a
/// null host_ptr. Driver picks a host-accessible memory location (typically
/// pinned, possibly write-combined). Caller must fill via
/// `fill_alloc_host_ptr_via_map`.
pub fn build_buffer_alloc_host_ptr_empty(
    backend: &argus_engine::backend::opencl::OpenCLBackend,
    size: usize,
) -> Result<ocl::core::Mem> {
    use ocl::core::ClContextPtr;

    const CL_MEM_READ_ONLY: u64 = 1 << 2;
    const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;

    let flags = (CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR) as ocl::ffi::cl_bitfield;
    let ctx_ptr = <&ocl::Context as ClContextPtr>::as_ptr(&&backend.context);
    let mut errcode: ocl::ffi::cl_int = 0;
    let raw = unsafe {
        ocl::ffi::clCreateBuffer(ctx_ptr, flags, size, std::ptr::null_mut(), &mut errcode)
    };
    if errcode != 0 || raw.is_null() {
        anyhow::bail!("clCreateBuffer(ALLOC_HOST_PTR) failed: errcode={}", errcode);
    }
    let mem = unsafe { ocl::core::Mem::from_raw_create_ptr(raw) };
    Ok(mem)
}

/// Path 4 fill step — map a `CL_MEM_ALLOC_HOST_PTR` cl_mem with MAP_WRITE,
/// memcpy `src..src+size` into it, then unmap. Driver flushes CPU cache +
/// makes the contents GPU-visible on unmap.
///
/// # Safety
/// `src` must be valid (readable) for `size` bytes for the duration of the
/// memcpy. `mem` must be a host-accessible buffer (created with
/// `CL_MEM_ALLOC_HOST_PTR` or `CL_MEM_USE_HOST_PTR`).
pub unsafe fn fill_alloc_host_ptr_via_map(
    backend: &argus_engine::backend::opencl::OpenCLBackend,
    mem: &ocl::core::Mem,
    src: *const u8,
    size: usize,
) -> Result<()> {
    use ocl::ffi;
    const CL_TRUE: ffi::cl_bool = 1;
    const CL_MAP_WRITE: ffi::cl_map_flags = 1 << 1;

    let q_ref: &ocl::core::CommandQueue = &backend.queue;
    let q_ptr: ffi::cl_command_queue = q_ref.as_ptr();

    let mut errcode: ffi::cl_int = 0;
    let mapped_ptr = unsafe {
        ffi::clEnqueueMapBuffer(
            q_ptr,
            mem.as_ptr(),
            CL_TRUE,
            CL_MAP_WRITE,
            0,
            size,
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            &mut errcode,
        )
    };
    if errcode != 0 || mapped_ptr.is_null() {
        anyhow::bail!(
            "clEnqueueMapBuffer(MAP_WRITE) failed: errcode={} ptr_null={}",
            errcode,
            mapped_ptr.is_null()
        );
    }

    unsafe { std::ptr::copy_nonoverlapping(src, mapped_ptr as *mut u8, size) };

    let rc = unsafe {
        ffi::clEnqueueUnmapMemObject(
            q_ptr,
            mem.as_ptr(),
            mapped_ptr,
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    if rc != 0 {
        anyhow::bail!("clEnqueueUnmapMemObject failed: rc={}", rc);
    }
    backend.queue.finish()?;
    Ok(())
}

/// Standard Q4_0 AOS GEMV (`m=1` GEMV) launch. `weight_mem` must hold the
/// AOS Q4_0 layout (18B/block: 2B fp16 scale + 16B nibbles). `input_mem`
/// holds k f32 inputs, `output_mem` receives n f32 outputs.
pub fn run_matmul_q4_0(
    backend: &argus_engine::backend::opencl::OpenCLBackend,
    kernel: &ocl::core::Kernel,
    input_mem: &ocl::core::Mem,
    weight_mem: &ocl::core::Mem,
    output_mem: &ocl::core::Mem,
    n: usize,
    k: usize,
) -> Result<()> {
    use ocl::core::ArgVal;

    let m: usize = 1;
    let ne00 = k as i32;
    let ne01 = n as i32;
    let ne02: i32 = 1;
    let ne10 = k as i32;
    let ne12 = (k * m) as i32;
    let ne0 = n as i32;
    let ne1 = (n * m) as i32;
    let r2: i32 = 1;
    let r3: i32 = 1;

    unsafe {
        ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(weight_mem))?;
        ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&0u64))?;
        ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(input_mem))?;
        ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&0u64))?;
        ocl::core::set_kernel_arg(kernel, 4, ArgVal::mem(output_mem))?;
        ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&0u64))?;
        ocl::core::set_kernel_arg(kernel, 6, ArgVal::scalar(&ne00))?;
        ocl::core::set_kernel_arg(kernel, 7, ArgVal::scalar(&ne01))?;
        ocl::core::set_kernel_arg(kernel, 8, ArgVal::scalar(&ne02))?;
        ocl::core::set_kernel_arg(kernel, 9, ArgVal::scalar(&ne10))?;
        ocl::core::set_kernel_arg(kernel, 10, ArgVal::scalar(&ne12))?;
        ocl::core::set_kernel_arg(kernel, 11, ArgVal::scalar(&ne0))?;
        ocl::core::set_kernel_arg(kernel, 12, ArgVal::scalar(&ne1))?;
        ocl::core::set_kernel_arg(kernel, 13, ArgVal::scalar(&r2))?;
        ocl::core::set_kernel_arg(kernel, 14, ArgVal::scalar(&r3))?;

        let local_work_size: [usize; 3] = [64, 1, 1];
        let group_size_0 = n.div_ceil(4);
        let global_work_size: [usize; 3] = [group_size_0 * local_work_size[0], m, 1];

        ocl::core::enqueue_kernel(
            &backend.queue,
            kernel,
            3,
            None,
            &global_work_size,
            Some(local_work_size),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    Ok(())
}
