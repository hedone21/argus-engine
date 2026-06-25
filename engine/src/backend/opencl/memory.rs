use crate::buffer::{Buffer, DType};
use crate::memory::Memory;
use crate::memory::opencl::device::OpenCLBuffer;
use crate::memory::opencl::unified::UnifiedBuffer;
use crate::memory::rpcmem::allocator::RpcmemAllocator;
use anyhow::Result;
use ocl::flags::MemFlags;
use ocl::{Context, Queue};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

static KV_ALLOC_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Returns `true` if `LLMRS_LOG_WEIGHT_VMA` is set.
/// Cached after first call — no per-alloc env lookup.
fn log_weight_vma_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLMRS_LOG_WEIGHT_VMA").is_ok())
}

pub struct OpenCLMemory {
    #[allow(dead_code)]
    context: Context,
    queue: Queue,
    /// Live bytes allocated through this `OpenCLMemory` (telemetry only — no
    /// production reader; see `used_memory`). Shared with every buffer built by
    /// `alloc`/`alloc_kv` so the buffer's `Drop` can subtract its size,
    /// preventing the monotonic inflation the plain `+= size` counter had.
    used_memory: Arc<AtomicUsize>,
    /// If true, use UnifiedBuffer (zero-copy shared memory)
    /// If false, use OpenCLBuffer (device-only, faster)
    use_zero_copy: bool,
    /// Sprint 2a Phase 2 (ENG-RPCMEM-022): when `Some` and `alloc_kv` is
    /// invoked, allocate via rpcmem heap + `CL_MEM_USE_HOST_PTR` alias instead
    /// of the standard `UnifiedBuffer` path. Note: `alloc` (activation
    /// tensors) ignores this field — rpcmem is KV/secondary only
    /// (INV-RPCMEM-007).
    rpcmem_allocator: Option<Arc<RpcmemAllocator>>,
}

impl OpenCLMemory {
    pub fn new(context: Context, queue: Queue, use_zero_copy: bool) -> Self {
        Self::new_with_rpcmem(context, queue, use_zero_copy, None)
    }

    /// Sprint 2a Phase 2 constructor — wires the backend-agnostic
    /// `RpcmemAllocator`. `None` matches the legacy behaviour of `new`. When
    /// `Some`, `alloc_kv` switches to the rpcmem + USE_HOST_PTR alias path.
    pub fn new_with_rpcmem(
        context: Context,
        queue: Queue,
        use_zero_copy: bool,
        rpcmem_allocator: Option<Arc<RpcmemAllocator>>,
    ) -> Self {
        Self {
            context,
            queue,
            used_memory: Arc::new(AtomicUsize::new(0)),
            use_zero_copy,
            rpcmem_allocator,
        }
    }

    /// rpcmem-backed KV alloc — wraps `RpcmemAllocator::alloc` +
    /// `CL_MEM_USE_HOST_PTR` alias creation into a `RpcmemKvBuffer`.
    /// Returns `Err` so the caller can apply per-buffer fallback
    /// (INV-RPCMEM-003).
    fn alloc_kv_rpcmem(
        &self,
        size: usize,
        dtype: DType,
        allocator: &Arc<RpcmemAllocator>,
    ) -> Result<Arc<dyn Buffer>> {
        use crate::memory::rpcmem::kv_buffer::RpcmemKvBuffer;
        use ocl::core::MEM_USE_HOST_PTR;

        // SAFETY: rpcmem allocator validated at construction; size > 0 enforced.
        let (host_ptr, fd) = unsafe { allocator.alloc(size)? };

        // CL_MEM_USE_HOST_PTR alias — zero-copy. ocl::core::create_buffer
        // requires a `Some(&mut [T])` for the host pointer; we materialise the
        // slice over the rpcmem region just-in-time.
        //
        // SAFETY: host_ptr is a freshly allocated rpcmem region of `size` bytes
        // owned by `allocator`; the alias is dropped before host_ptr in
        // `RpcmemKvBuffer::drop` (field declaration order).
        let cl_mem_slice = unsafe { std::slice::from_raw_parts_mut(host_ptr, size) };
        let cl_mem = unsafe {
            match ocl::core::create_buffer::<_, u8>(
                self.context.as_core(),
                MEM_USE_HOST_PTR,
                size,
                Some(cl_mem_slice),
            ) {
                Ok(m) => m,
                Err(e) => {
                    // Roll back rpcmem alloc on alias creation failure so the
                    // heap region doesn't leak.
                    allocator.free(host_ptr);
                    return Err(anyhow::anyhow!("rpcmem KV alias create_buffer failed: {e}"));
                }
            }
        };

        let mut buffer =
            RpcmemKvBuffer::new(host_ptr, fd, cl_mem, size, dtype, Arc::clone(allocator));
        buffer.attach_mem_accounting(Arc::clone(&self.used_memory));
        Ok(Arc::new(buffer))
    }
}

impl Memory for OpenCLMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        // INV-RPCMEM-007: activation alloc path does NOT consult
        // `rpcmem_allocator`. rpcmem heap is KV/secondary only.
        let buffer: Arc<dyn Buffer> = if self.use_zero_copy {
            // Zero-copy shared memory (CPU-GPU accessible, but slower GPU kernels)
            let mut ub = UnifiedBuffer::new(self.queue.clone(), size, dtype)?;
            if log_weight_vma_enabled() {
                // cl_mem pointer: obtained via cl_mem().as_ptr() on the Mem wrapper.
                // host_ptr: map briefly to get the VMA address, then unmap immediately.
                // The map/unmap here is diagnostic-only; it does not affect GPU-read state
                // because the buffer is freshly allocated (no GPU writes in flight yet).
                let cl_hex = ub
                    .cl_mem()
                    .map(|m| format!("{:#x}", m.as_ptr() as usize))
                    .unwrap_or_else(|| "null".to_string());
                let host_hex = match ub.map() {
                    Ok(p) => {
                        let s = format!("{p:p}");
                        let _ = ub.unmap();
                        s
                    }
                    Err(_) => "null".to_string(),
                };
                eprintln!(
                    "[VMA] alloc path=zero_copy size={size} dtype={dtype:?} \
                     cl_mem={cl_hex} host_ptr={host_hex}"
                );
            }
            ub.attach_mem_accounting(Arc::clone(&self.used_memory));
            Arc::new(ub)
        } else {
            // Device-only memory (faster GPU kernels, requires explicit copies)
            let ocl_buffer = ocl::Buffer::<u8>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(size)
                .build()?;
            let mut ob = OpenCLBuffer::new(self.queue.clone(), ocl_buffer, size, dtype)?;
            if log_weight_vma_enabled() {
                let cl_hex = ob
                    .cl_mem()
                    .map(|m| format!("{:#x}", m.as_ptr() as usize))
                    .unwrap_or_else(|| "null".to_string());
                eprintln!(
                    "[VMA] alloc path=device_only size={size} dtype={dtype:?} \
                     cl_mem={cl_hex} host_ptr=null"
                );
            }
            ob.attach_mem_accounting(Arc::clone(&self.used_memory));
            Arc::new(ob)
        };

        Ok(buffer)
    }

    fn alloc_kv(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        // Sprint 2a Phase 2 (ENG-RPCMEM-022): rpcmem heap KV branch.
        if let Some(allocator) = self.rpcmem_allocator.as_ref() {
            match self.alloc_kv_rpcmem(size, dtype, allocator) {
                Ok(buf) => {
                    // Accounting is attached inside `alloc_kv_rpcmem` (the
                    // `RpcmemKvBuffer` carries the counter and decrements on Drop).
                    return Ok(buf);
                }
                Err(e) => {
                    // Per-buffer fallback (INV-RPCMEM-003) — log once per
                    // failure and continue with the standard path.
                    eprintln!(
                        "[OpenCL] alloc_kv rpcmem 실패 (size={size}): {e} — UnifiedBuffer fallback"
                    );
                    // fall through to default branch below
                }
            }
        }

        if self.use_zero_copy {
            // KV cache: CL_MEM_ALLOC_HOST_PTR (driver-managed, single VMA).
            // Starts UNMAPPED — GPU writes via cl_mem during forward pass.
            // Map for CPU access only during SwitchHw (via kv_migrate).
            // Note: OpenCL spec says writing to a mapped buffer via cl_mem is UB.
            let mut buffer = UnifiedBuffer::new(self.queue.clone(), size, dtype)?;

            if std::env::var_os("LLMRS_LOG_KV_VMA").is_some() {
                let idx = KV_ALLOC_COUNTER.fetch_add(1, Ordering::Relaxed);
                match buffer.map() {
                    Ok(host_ptr) => {
                        eprintln!(
                            "[KV_VMA] idx={:3} host_ptr={:p} size={}",
                            idx, host_ptr, size
                        );
                        let _ = buffer.unmap();
                    }
                    Err(e) => {
                        eprintln!("[KV_VMA] idx={} map failed: {}", idx, e);
                    }
                }
            }

            buffer.attach_mem_accounting(Arc::clone(&self.used_memory));
            let buf: Arc<dyn Buffer> = Arc::new(buffer);
            Ok(buf)
        } else {
            self.alloc(size, dtype)
        }
    }

    fn used_memory(&self) -> usize {
        self.used_memory.load(Ordering::Relaxed)
    }
}

// `used_memory` decrement-on-drop regression coverage. Requires a working
// OpenCL device (GPU or POCL CPU). Skips gracefully when none is present, the
// same contract as `unified.rs`'s buffer tests — so it is a no-op on GPU-less
// CI but exercises the real alloc→drop path on a developer host.
#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    use super::*;
    use crate::buffer::DType;
    use ocl::{Context, Device, Platform, Queue};
    use std::panic;

    /// Build an `OpenCLMemory` over a real device, or `None` if no OpenCL
    /// platform/device is available (test then skips). GPU platforms are
    /// preferred over a POCL CPU platform, matching `unified.rs`.
    fn test_memory(use_zero_copy: bool) -> Option<OpenCLMemory> {
        let platform_list = panic::catch_unwind(|| Platform::list()).ok()?;
        let platform = platform_list
            .iter()
            .copied()
            .find(|p| {
                Device::list(*p, Some(ocl::flags::DEVICE_TYPE_GPU))
                    .map(|d| !d.is_empty())
                    .unwrap_or(false)
            })
            .or_else(|| platform_list.into_iter().next())?;
        let device = match Device::list(platform, Some(ocl::flags::DEVICE_TYPE_GPU)) {
            Ok(list) if !list.is_empty() => list[0],
            _ => Device::first(platform).ok()?,
        };
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .ok()?;
        let queue = Queue::new(&context, device, None).ok()?;
        Some(OpenCLMemory::new(context, queue, use_zero_copy))
    }

    #[test]
    fn used_memory_returns_to_zero_after_drop_device() {
        // Device-only path → `OpenCLBuffer`. Covers `alloc` and the
        // non-zero-copy `alloc_kv` (which delegates to `alloc`).
        let Some(mem) = test_memory(false) else {
            eprintln!("[SKIPPED] No OpenCL device");
            return;
        };
        assert_eq!(mem.used_memory(), 0);

        let a = mem.alloc(1024, DType::F32).unwrap();
        assert_eq!(mem.used_memory(), 1024);

        // non-zero-copy alloc_kv → alloc → OpenCLBuffer
        let b = mem.alloc_kv(512, DType::F16).unwrap();
        assert_eq!(mem.used_memory(), 1024 + 512);

        // Partial drop must subtract only `a`'s size (proves per-buffer size
        // tracking, not a flag/zero reset).
        drop(a);
        assert_eq!(mem.used_memory(), 512);

        drop(b);
        assert_eq!(mem.used_memory(), 0);
    }

    #[test]
    fn used_memory_returns_to_zero_after_drop_unified() {
        // Zero-copy path → `UnifiedBuffer`, covering both `alloc` and the
        // zero-copy `alloc_kv` branch.
        let Some(mem) = test_memory(true) else {
            eprintln!("[SKIPPED] No OpenCL device");
            return;
        };
        assert_eq!(mem.used_memory(), 0);

        let a = mem.alloc(2048, DType::F32).unwrap();
        assert_eq!(mem.used_memory(), 2048);

        // zero-copy alloc_kv → UnifiedBuffer
        let b = mem.alloc_kv(256, DType::F16).unwrap();
        assert_eq!(mem.used_memory(), 2048 + 256);

        drop(b);
        assert_eq!(mem.used_memory(), 2048);

        drop(a);
        assert_eq!(mem.used_memory(), 0);
    }
}
