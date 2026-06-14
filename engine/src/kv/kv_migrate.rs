//! KV cache migration between backends (CPU↔GPU).
//!
//! On UMA devices (Adreno, Jetson) with zero-copy buffers, GPU and CPU share
//! the same physical memory. Migration is a zero-copy re-tag of the backend —
//! no data is copied. On discrete GPUs, a full copy via intermediate CPU
//! buffers is performed.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::kv::kv_cache::KVCache;
use crate::memory::Memory;
use crate::tensor::Tensor;

/// Migrate KV caches from one backend to another.
///
/// On UMA (zero-copy): re-tags existing buffers with the destination backend.
/// No memory allocation or copy occurs — the same host_data backing the
/// CL_MEM_USE_HOST_PTR buffer is directly accessible from both CPU and GPU.
///
/// On discrete GPU: reads KV data via `src_backend`, creates intermediate CPU
/// tensors, then optionally copies to `dst_backend`.
///
/// # Arguments
/// * `kv_caches` - Mutable KV caches to migrate in-place
/// * `src_backend` - Backend to read current KV data from
/// * `dst_backend` - Backend to write migrated KV data to
/// * `cpu_backend` - CPU backend for intermediate tensors
/// * `cpu_memory` - CPU memory allocator for intermediate buffers
/// * `dst_memory` - Memory allocator for the destination backend
/// * `kv_heads` - Number of KV attention heads
/// * `head_dim` - Dimension per attention head
/// * `max_seq_len` - Maximum sequence length (for new KVCache allocation)
/// * `copy_to_dst` - If true, copy CPU tensors to dst_backend; if false, keep on CPU
#[allow(clippy::too_many_arguments)]
pub fn migrate_kv_caches(
    kv_caches: &mut [KVCache],
    src_backend: &Arc<dyn Backend>,
    dst_backend: &Arc<dyn Backend>,
    cpu_backend: &Arc<dyn Backend>,
    cpu_memory: &Arc<dyn Memory>,
    dst_memory: &Arc<dyn Memory>,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    copy_to_dst: bool,
) -> Result<()> {
    // UMA detection: check EITHER backend for GPU capability (handles both directions).
    let is_uma = (src_backend.is_gpu() || dst_backend.is_gpu())
        && !src_backend.is_discrete_gpu()
        && !dst_backend.is_discrete_gpu();

    // Check if buffers have cl_mem (UnifiedBuffer or other cl_mem-bearing types).
    let has_cl_mem = is_uma
        && kv_caches
            .first()
            .is_some_and(|kv| kv.k_buffer.buffer().cl_mem().is_some());

    // UMA path: map UnifiedBuffers for CPU access and flush GPU caches.
    // UnifiedBuffer (ALLOC_HOST_PTR) starts unmapped — map_for_cpu() calls
    // clEnqueueMapBuffer which flushes GPU caches and returns a coherent pointer.
    // USE_HOST_PTR buffers are always mapped — map_for_cpu() is a no-op.
    if has_cl_mem {
        src_backend.synchronize()?;
        for kv in kv_caches.iter_mut() {
            // Map if not already mapped (UnifiedBuffer case).
            // After map, as_ptr() returns a valid coherent pointer.
            kv.k_buffer.buffer().map_for_cpu()?;
            kv.v_buffer.buffer().map_for_cpu()?;
        }
    }

    let host_accessible = has_cl_mem
        && kv_caches
            .first()
            .is_some_and(|kv| !kv.k_buffer.as_ptr().is_null());

    for kv in kv_caches.iter_mut() {
        let current_capacity = kv.capacity();
        let saved_pos = kv.current_pos;

        let (k_final, v_final) = if host_accessible {
            // UMA zero-copy path: reuse existing buffer, just swap backend tag.
            // GPU cache was flushed above via map_for_cpu(), so as_ptr() is coherent.
            let k = Tensor::new(
                kv.k_buffer.shape().clone(),
                kv.k_buffer.buffer().clone(),
                dst_backend.clone(),
            );
            let v = Tensor::new(
                kv.v_buffer.shape().clone(),
                kv.v_buffer.buffer().clone(),
                dst_backend.clone(),
            );
            (k, v)
        } else {
            // Discrete GPU path: full copy through CPU intermediate buffers.
            let kv_dtype = kv.k_buffer.dtype();
            let k_size = kv.k_buffer.size();
            let v_size = kv.v_buffer.size();

            let mut k_data = vec![0u8; k_size];
            let mut v_data = vec![0u8; v_size];
            src_backend.read_buffer(&kv.k_buffer, &mut k_data)?;
            src_backend.read_buffer(&kv.v_buffer, &mut v_data)?;

            let k_cpu_buf = cpu_memory.alloc(k_size, kv_dtype)?;
            unsafe {
                std::ptr::copy_nonoverlapping(k_data.as_ptr(), k_cpu_buf.as_mut_ptr(), k_size);
            }
            let k_cpu_tensor =
                Tensor::new(kv.k_buffer.shape().clone(), k_cpu_buf, cpu_backend.clone());

            let v_cpu_buf = cpu_memory.alloc(v_size, kv_dtype)?;
            unsafe {
                std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_cpu_buf.as_mut_ptr(), v_size);
            }
            let v_cpu_tensor =
                Tensor::new(kv.v_buffer.shape().clone(), v_cpu_buf, cpu_backend.clone());

            if copy_to_dst {
                let kf = dst_backend.copy_from(&k_cpu_tensor)?;
                let vf = dst_backend.copy_from(&v_cpu_tensor)?;
                (
                    Tensor::new(kf.shape().clone(), kf.buffer().clone(), dst_backend.clone()),
                    Tensor::new(vf.shape().clone(), vf.buffer().clone(), dst_backend.clone()),
                )
            } else {
                (k_cpu_tensor, v_cpu_tensor)
            }
        };

        let saved_layout = kv.layout();
        let mut new_kv = KVCache::new_dynamic(
            k_final,
            v_final,
            current_capacity,
            max_seq_len,
            kv_heads,
            head_dim,
            dst_memory.clone(),
        )
        .with_layout(saved_layout);
        new_kv.current_pos = saved_pos;
        *kv = new_kv;
    }

    eprintln!(
        "[KV Migrate] {} layers migrated ({})",
        kv_caches.len(),
        if host_accessible {
            "UMA zero-copy re-tag"
        } else {
            "GPU→CPU copy"
        }
    );
    Ok(())
}
