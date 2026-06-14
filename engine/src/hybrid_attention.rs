//! UMA Hybrid CPU-GPU Attention — shared setup & thread-local scope.
//!
//! Stage A: `HybridAttnSetup` + thread-local install/clear discipline + env
//! bootstrap.
//!
//! Stage C (this file): extends `HybridAttnSetup` with per-decode scratch
//! buffers and host-mapped pointers required by the KV-split flash-decoding
//! plan path. All buffers are allocated as `CL_MEM_ALLOC_HOST_PTR` and
//! permanent-mapped for the lifetime of the decode — the setup is constructed
//! right before the decode loop and dropped when it exits, so the map cost is
//! paid once (not per token).
//!
//! Buffer layout (decode seq_len=1 only):
//!   * `partial_ml_gpu` — `f32[n_heads_q * 2]` — GPU-written `(m, l)` pair
//!     per Q-head, filled by `flash_attn_f32_f16_q1_partial`.
//!   * `partial_o_gpu`  — `f32[n_heads_q * head_dim]` — GPU-written
//!     un-normalised output rows.
//!   * `ready_flags_gpu` — `i32[n_heads_q]` — per-head sigflag (Stage D uses
//!     spin-poll; Stage C relies on blocking `clFinish` and ignores the flag
//!     at the host side, though the kernel still sets it).
//!   * `partial_ml_cpu` / `partial_o_cpu` — pure host scratch of matching
//!     shape, written by the NEON partial helper.
//!
//! All three GPU buffers are shared across layers (no concurrent layer
//! execution) and allocated once per decode. The K/V cache buffers stay in
//! the `KVCache` struct and are mapped separately by the caller (see
//! `generate.rs` decode entry hook) because their lifetime is the full decode
//! loop, not just the setup.

use std::cell::RefCell;
use std::sync::Arc;

#[cfg(feature = "opencl")]
use ocl::Queue;
#[cfg(feature = "opencl")]
use ocl::core::Mem;

/// Minimum meaningful KV split fraction. Below this, the GPU tail is too
/// small to amortise the dispatch + merge cost.
const HYBRID_KV_FRAC_MIN: f32 = 0.05;
/// Maximum meaningful KV split fraction. Above this, the CPU head is too
/// small to benefit from the split.
const HYBRID_KV_FRAC_MAX: f32 = 0.9;
/// Environment variable for KV-split fraction override.
pub const ENV_KV_FRAC: &str = "LLMRS_ATTN_HYBRID_KV_FRAC";

/// Shared configuration for UMA hybrid CPU-GPU attention.
///
/// Stage A carried only `kv_frac`. Stage C extends it with per-decode scratch
/// buffers (GPU + CPU) and their permanent-mapped host pointers, which is
/// what the plan dispatch path and the NEON merge helper actually consume.
///
/// All buffers are shared across layers (no concurrent layer execution in
/// decode). Per-layer partitioning would be over-engineering for the v1.
///
/// Instances are Arc-shared through `install`/`current`. The Drop impl takes
/// care of unmapping the GPU buffers and releasing the CL memory objects
/// when the last clone goes away (end of decode).
pub struct HybridAttnSetup {
    /// Fraction of the KV range routed to the CPU partial. The GPU handles
    /// the complementary prefix `[0, kv_end)` where
    /// `kv_end = kv_len - round(kv_len * kv_frac)`. Clamped into
    /// `[HYBRID_KV_FRAC_MIN, HYBRID_KV_FRAC_MAX]` at construction time.
    pub kv_frac: f32,
    /// Number of Q heads (for sizing partial buffers).
    pub n_heads_q: usize,
    /// Head dimension (64 or 128 — validated at construction).
    pub head_dim: usize,

    // --- GPU scratch (Stage C) ---
    /// GPU-written partial `(m, l)` pair per Q-head.
    /// Size: `n_heads_q * 2 * sizeof(f32)`.
    #[cfg(feature = "opencl")]
    pub partial_ml_gpu: HybridGpuBuffer,
    /// GPU-written un-normalised output rows.
    /// Size: `n_heads_q * head_dim * sizeof(f32)`.
    #[cfg(feature = "opencl")]
    pub partial_o_gpu: HybridGpuBuffer,
    /// GPU-written per-head sigflag array. Stage C does not read it (uses
    /// blocking `clFinish` instead); Stage D switches to spin-poll here.
    /// Size: `n_heads_q * sizeof(i32)`.
    #[cfg(feature = "opencl")]
    pub ready_flags_gpu: HybridGpuBuffer,

    // --- CPU scratch (Stage A/C) ---
    /// Host-side partial `(m, l)` buffer produced by `flash_partial_kv_range_f16`.
    /// Kept inside the setup so the Mutex-protected allocation only happens
    /// once per decode — same shape as `partial_ml_gpu`. Mutex is required
    /// because HybridAttnSetup is Arc-shared across dispatches, and the
    /// plan dispatch mutates this slice per layer.
    pub partial_ml_cpu: std::sync::Mutex<Vec<f32>>,
    /// Host-side un-normalised output rows. Same shape as `partial_o_gpu`.
    pub partial_o_cpu: std::sync::Mutex<Vec<f32>>,
}

/// Wrapper around a permanent-mapped host_ptr GPU buffer. Created via
/// `HybridGpuBuffer::new_host_visible`. The map is taken in the constructor
/// and released in Drop; the raw host pointer is exposed via `host_ptr()`.
///
/// Raw pointer fields require `unsafe impl Send/Sync` because the plan
/// dispatch thread (which today is the same as the generate main thread, but
/// Rayon worker fanning is possible under `flash_partial_kv_range_f16`)
/// dereferences them. The OpenCL buffer itself is thread-safe (driver-side)
/// and the host_ptr points into driver-owned memory for the lifetime of the
/// map.
#[cfg(feature = "opencl")]
pub struct HybridGpuBuffer {
    /// Owned ocl::Buffer<u8> — kept alive for the duration of the map.
    buffer: ocl::Buffer<u8>,
    /// Permanent host map pointer. Valid between `new_host_visible` and Drop.
    host_ptr: *mut u8,
    /// Number of elements (not bytes). Element dtype is implicit in the
    /// caller's cast.
    #[allow(dead_code)]
    elem_count: usize,
    /// Queue used for map/unmap (kept alive for Drop).
    queue: Queue,
}

#[cfg(feature = "opencl")]
unsafe impl Send for HybridGpuBuffer {}
#[cfg(feature = "opencl")]
unsafe impl Sync for HybridGpuBuffer {}

#[cfg(feature = "opencl")]
impl HybridGpuBuffer {
    /// Allocate a host-visible buffer (ALLOC_HOST_PTR) sized `byte_size` and
    /// permanent-map it. The returned handle keeps the cl_mem and the host
    /// map alive until dropped.
    pub fn new_host_visible(
        queue: &Queue,
        byte_size: usize,
        elem_count: usize,
    ) -> anyhow::Result<Self> {
        use anyhow::anyhow;
        use ocl::flags;

        let buffer = ocl::Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR)
            .len(byte_size)
            .build()
            .map_err(|e| anyhow!("HybridGpuBuffer build failed: {}", e))?;

        // 영구 매핑. 포인터는 Drop까지 유효.
        let mapped = unsafe {
            buffer
                .map()
                .read()
                .write()
                .len(byte_size)
                .enq()
                .map_err(|e| anyhow!("HybridGpuBuffer map failed: {}", e))?
        };
        let host_ptr = mapped.as_ptr() as *mut u8;
        // MemMap 해제 시 자동 unmap이 호출되므로 leak으로 살려둔다. Drop에서 수동 finish.
        std::mem::forget(mapped);

        Ok(Self {
            buffer,
            host_ptr,
            elem_count,
            queue: queue.clone(),
        })
    }

    /// Raw cl_mem handle (for kernel arg binding).
    pub fn cl_mem(&self) -> &Mem {
        self.buffer.as_core()
    }

    /// Raw host pointer (cast by caller to the element type).
    pub fn host_ptr(&self) -> *mut u8 {
        self.host_ptr
    }
}

#[cfg(feature = "opencl")]
impl Drop for HybridGpuBuffer {
    fn drop(&mut self) {
        // 큐 드레인 후 버퍼를 자연 해제. std::mem::forget으로 누락된 MemMap은
        // cl_mem 객체와 함께 driver가 해제한다.
        let _ = self.queue.finish();
    }
}

impl HybridAttnSetup {
    /// Construct a setup with the given KV-split fraction (host-only, no GPU
    /// buffers attached). Used by Stage A unit tests and whenever only
    /// `kv_frac` is needed.
    ///
    /// Returns `None` if `kv_frac` is outside the meaningful range
    /// `[HYBRID_KV_FRAC_MIN, HYBRID_KV_FRAC_MAX]` or is not finite.
    #[cfg(test)]
    pub fn new(kv_frac: f32) -> Option<Self> {
        if !Self::is_kv_frac_valid(kv_frac) {
            return None;
        }
        // 테스트 전용 인스턴스: 버퍼 필드는 opencl feature off에서 생략.
        #[cfg(feature = "opencl")]
        {
            None // opencl 빌드에서는 new_for_decode를 써야 한다.
        }
        #[cfg(not(feature = "opencl"))]
        Some(Self {
            kv_frac,
            n_heads_q: 0,
            head_dim: 0,
            partial_ml_cpu: std::sync::Mutex::new(Vec::new()),
            partial_o_cpu: std::sync::Mutex::new(Vec::new()),
        })
    }

    fn is_kv_frac_valid(kv_frac: f32) -> bool {
        kv_frac.is_finite() && (HYBRID_KV_FRAC_MIN..=HYBRID_KV_FRAC_MAX).contains(&kv_frac)
    }

    /// Parse `LLMRS_ATTN_HYBRID_KV_FRAC` from the process environment.
    ///
    /// Returns `Some(kv_frac)` only if the variable is set AND parses to a
    /// value in the meaningful range. Unset / invalid / out-of-range values
    /// all map to `None` (hybrid path disabled).
    pub fn from_env() -> Option<f32> {
        let raw = std::env::var(ENV_KV_FRAC).ok()?;
        let val: f32 = raw.trim().parse().ok()?;
        if Self::is_kv_frac_valid(val) {
            Some(val)
        } else {
            None
        }
    }

    /// Construct a full setup ready for decode: allocates the three shared
    /// GPU scratch buffers (partial_ml, partial_o, ready_flags) via
    /// `CL_MEM_ALLOC_HOST_PTR` and pre-maps them. Also allocates matching
    /// CPU scratch.
    ///
    /// Caller is responsible for verifying gating (backend type, kv dtype,
    /// head_dim, GQA, etc.) before calling this — unlike `from_env`, this
    /// does NOT re-check the environment variable.
    #[cfg(feature = "opencl")]
    pub fn new_for_decode(
        queue: &Queue,
        kv_frac: f32,
        n_heads_q: usize,
        head_dim: usize,
    ) -> anyhow::Result<Self> {
        use anyhow::ensure;
        ensure!(
            Self::is_kv_frac_valid(kv_frac),
            "hybrid kv_frac {} out of range",
            kv_frac
        );
        ensure!(
            head_dim == 64 || head_dim == 128,
            "hybrid head_dim must be 64 or 128, got {}",
            head_dim
        );
        ensure!(n_heads_q > 0, "hybrid n_heads_q must be > 0");

        let ml_elems = n_heads_q * 2;
        let o_elems = n_heads_q * head_dim;
        let flag_elems = n_heads_q;

        let partial_ml_gpu = HybridGpuBuffer::new_host_visible(queue, ml_elems * 4, ml_elems)?;
        let partial_o_gpu = HybridGpuBuffer::new_host_visible(queue, o_elems * 4, o_elems)?;
        let ready_flags_gpu = HybridGpuBuffer::new_host_visible(queue, flag_elems * 4, flag_elems)?;

        // Stage D: ready_flags 를 0으로 초기 설정. Plan dispatch 진입 시에도
        // 매 layer 리셋하지만, 첫 layer가 폴링을 곧바로 시작하기 때문에 할당
        // 직후 한 번 0으로 클리어해 둔다. UMA 직접 쓰기 + Release fence로
        // GPU 커널이 관측하기 전에 가시화.
        unsafe {
            let ptr = ready_flags_gpu.host_ptr() as *mut i32;
            std::ptr::write_bytes(ptr, 0, flag_elems);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

        Ok(Self {
            kv_frac,
            n_heads_q,
            head_dim,
            partial_ml_gpu,
            partial_o_gpu,
            ready_flags_gpu,
            partial_ml_cpu: std::sync::Mutex::new(vec![0.0; ml_elems]),
            partial_o_cpu: std::sync::Mutex::new(vec![0.0; o_elems]),
        })
    }
}

/// Compute the KV split point for a given `kv_len`.
///
/// Returns `(kv_end_gpu, kv_end_cpu)` where:
///   * `kv_end_gpu` = exclusive end of the GPU prefix `[0, kv_end_gpu)`
///   * `kv_end_cpu` = `kv_len` (the CPU handles `[kv_end_gpu, kv_len)`)
///
/// Guarantees `kv_end_gpu <= kv_len`. When `kv_frac > 0` and `kv_len > 0`,
/// the CPU tail has at least 1 token. Returns `(kv_len, kv_len)` (CPU no-op)
/// for `kv_len == 0`.
pub fn compute_kv_split(kv_len: usize, kv_frac: f32) -> (usize, usize) {
    if kv_len == 0 {
        return (0, 0);
    }
    let cpu_tail = (kv_len as f32 * kv_frac).round() as usize;
    // 최소 1 토큰은 CPU에 할당 (kv_frac > 0일 때).
    let cpu_tail = cpu_tail.max(1).min(kv_len);
    let kv_end_gpu = kv_len - cpu_tail;
    (kv_end_gpu, kv_len)
}

thread_local! {
    /// Thread-local installed setup. Written only by `install`/`HybridScope::drop`.
    static CURRENT: RefCell<Option<Arc<HybridAttnSetup>>> = const { RefCell::new(None) };
}

/// RAII guard returned by [`install`]. On drop, restores whatever setup was
/// installed before the matching `install` call (typically `None`). This
/// pattern guarantees that nested installs (should they ever occur) unwind
/// cleanly and that the setup never survives past the scope where it was
/// intended to be active.
#[must_use = "HybridScope must be held alive for the duration of the decode; \
              dropping it immediately clears the setup"]
pub struct HybridScope {
    /// Setup that was installed when `install` was called. Re-installed on
    /// drop. `None` means "there was nothing installed before".
    previous: Option<Arc<HybridAttnSetup>>,
}

impl Drop for HybridScope {
    fn drop(&mut self) {
        // 이전 설정으로 원상 복구. 현재 셀 값을 previous로 덮어쓴다.
        let prev = self.previous.take();
        CURRENT.with(|cell| {
            *cell.borrow_mut() = prev;
        });
    }
}

/// Install `setup` as the active hybrid attention configuration for the
/// current thread, returning a scope guard that restores the previous
/// configuration on drop.
pub fn install(setup: Arc<HybridAttnSetup>) -> HybridScope {
    let previous = CURRENT.with(|cell| cell.borrow_mut().replace(setup));
    HybridScope { previous }
}

/// Return the currently-installed hybrid attention setup for this thread,
/// or `None` if no setup is active.
pub fn current() -> Option<Arc<HybridAttnSetup>> {
    CURRENT.with(|cell| cell.borrow().clone())
}

#[cfg(test)]
#[cfg(not(feature = "opencl"))]
mod tests {
    use super::*;

    #[test]
    fn is_kv_frac_valid_range() {
        assert!(HybridAttnSetup::is_kv_frac_valid(0.05));
        assert!(HybridAttnSetup::is_kv_frac_valid(0.5));
        assert!(HybridAttnSetup::is_kv_frac_valid(0.9));
        assert!(!HybridAttnSetup::is_kv_frac_valid(0.0));
        assert!(!HybridAttnSetup::is_kv_frac_valid(0.04));
        assert!(!HybridAttnSetup::is_kv_frac_valid(0.91));
        assert!(!HybridAttnSetup::is_kv_frac_valid(1.0));
        assert!(!HybridAttnSetup::is_kv_frac_valid(-0.1));
        assert!(!HybridAttnSetup::is_kv_frac_valid(f32::NAN));
        assert!(!HybridAttnSetup::is_kv_frac_valid(f32::INFINITY));
    }

    #[test]
    fn install_and_current_roundtrip() {
        // 처음엔 아무 것도 설치되지 않아야 한다.
        assert!(current().is_none());

        let setup = Arc::new(HybridAttnSetup::new(0.5).unwrap());
        let scope = install(Arc::clone(&setup));

        let got = current().expect("setup should be installed");
        assert!((got.kv_frac - 0.5).abs() < 1e-6);
        // Arc 동일 인스턴스인지 확인.
        assert!(Arc::ptr_eq(&got, &setup));

        drop(scope);
        // 스코프 종료 후에는 원상복구되어야 한다.
        assert!(current().is_none());
    }

    #[test]
    fn nested_install_restores_previous_on_drop() {
        let outer = Arc::new(HybridAttnSetup::new(0.3).unwrap());
        let inner = Arc::new(HybridAttnSetup::new(0.7).unwrap());

        let outer_scope = install(Arc::clone(&outer));
        assert!((current().unwrap().kv_frac - 0.3).abs() < 1e-6);

        {
            let _inner_scope = install(Arc::clone(&inner));
            assert!((current().unwrap().kv_frac - 0.7).abs() < 1e-6);
        }
        // 내부 스코프가 drop되면 바깥 설정으로 복귀.
        assert!((current().unwrap().kv_frac - 0.3).abs() < 1e-6);

        drop(outer_scope);
        assert!(current().is_none());
    }
}

#[cfg(test)]
mod from_env_tests {
    use super::*;

    #[test]
    fn from_env_rejects_out_of_range() {
        // 테스트 환경 오염 방지를 위해 set/unset을 한 스레드 안에서 수행.
        // SAFETY: set_var / remove_var는 멀티스레드 환경에서 unsafe (Rust 1.74+).
        //         단일 테스트 스레드 내에서만 접근하도록 직렬화는 테스트 런너의 책임.
        unsafe {
            std::env::set_var(ENV_KV_FRAC, "1.5");
        }
        assert!(HybridAttnSetup::from_env().is_none());

        unsafe {
            std::env::set_var(ENV_KV_FRAC, "not-a-number");
        }
        assert!(HybridAttnSetup::from_env().is_none());

        unsafe {
            std::env::set_var(ENV_KV_FRAC, "0.5");
        }
        let v = HybridAttnSetup::from_env().expect("0.5 is in range");
        assert!((v - 0.5).abs() < 1e-6);

        unsafe {
            std::env::remove_var(ENV_KV_FRAC);
        }
        assert!(HybridAttnSetup::from_env().is_none());
    }

    #[test]
    fn compute_kv_split_basic() {
        // kv_len=100, kv_frac=0.25 → CPU 25, GPU 75
        let (g, c) = compute_kv_split(100, 0.25);
        assert_eq!(g, 75);
        assert_eq!(c, 100);

        // kv_len=0 → no-op
        let (g, c) = compute_kv_split(0, 0.5);
        assert_eq!(g, 0);
        assert_eq!(c, 0);

        // kv_len=1, kv_frac=0.5 → CPU 최소 1 보장 → GPU 0
        let (g, c) = compute_kv_split(1, 0.5);
        assert_eq!(g, 0);
        assert_eq!(c, 1);
    }
}
