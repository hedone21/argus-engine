//! LISWAP-3 prototype unit tests — `HostPtrPool` slot lifecycle.
//!
//! Plan: `compiled-chasing-hopper.md` Direction A track, Stage 3.
//! Prototype-grade — no spec ID. Tests run against an actual OpenCL backend
//! and skip cleanly when OpenCL initialisation fails (CI / sandboxed dev
//! shells without a usable driver).
//!
//! Coverage:
//! - **A**: pool with `n_slots = 4` builds successfully.
//! - **B**: `acquire(size)` succeeds up to `n_slots`, returns `None` on
//!   exhaustion.
//! - **C**: dropping a guard releases its slot back to the pool.
//! - **D**: `acquire(size)` rejects requests `> max_tensor_size` with
//!   `None` (not a hard error — caller falls back to staging).
//! - **E**: `host_ptr_pool_or_init` is gated by the env var and is
//!   `None` when the env is unset.
//! - **F**: `host_ptr_pool_or_init` returns the same `Arc` on subsequent
//!   calls (lazy-init contract).
//! - **G**: a pool slot can be filled via `fill_host_ptr_buffer` and the
//!   bytes are GPU-visible (read back via `clEnqueueReadBuffer`).
//! - **H** (Stage 4 fix): `get_cl_mem` accepts a `HostPtrPoolBuffer` and
//!   returns `Ok` — fixes the "Unknown" dispatch error on zerocopy decode.
//! - **I** (Stage 4 fix): `buffer_kind_label` returns `"HostPtrPool"` for
//!   a `HostPtrPoolBuffer`.

#![cfg(feature = "opencl")]

use std::sync::Arc;

use argus_engine::backend::opencl::OpenCLBackend;
use argus_engine::backend::opencl::host_ptr_pool::{
    HostPtrPool, HostPtrPoolConfig, host_ptr_pool_env_enabled,
};
use argus_engine::backend::opencl::host_ptr_pool_buffer::HostPtrPoolBuffer;
use argus_engine::backend::opencl::{buffer_kind_label, get_cl_mem};
use argus_engine::buffer::DType;

/// Try to bring up an OpenCL backend, returning `None` if the driver is
/// unavailable. Mirrors the pattern in `core::backend::tests`.
fn try_init_backend() -> Option<OpenCLBackend> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(OpenCLBackend::new)) {
        Ok(Ok(b)) => Some(b),
        Ok(Err(e)) => {
            eprintln!("[test] skipping host_ptr_pool: OpenCL backend unavailable ({e})");
            None
        }
        Err(_) => {
            eprintln!("[test] skipping host_ptr_pool: OpenCL backend init panicked (no devices?)");
            None
        }
    }
}

#[test]
fn test_a_pool_construction_succeeds() {
    let Some(backend) = try_init_backend() else {
        return;
    };
    let cfg = HostPtrPoolConfig {
        n_slots: 4,
        max_tensor_size: 4096,
    };
    let pool = HostPtrPool::new(&backend, cfg).expect("pool construction must succeed");
    assert_eq!(pool.n_slots(), 4);
    assert_eq!(pool.max_tensor_size(), 4096);
    assert_eq!(pool.in_use_count(), 0);
}

#[test]
fn test_b_acquire_until_exhaustion_returns_none() {
    let Some(backend) = try_init_backend() else {
        return;
    };
    let cfg = HostPtrPoolConfig {
        n_slots: 3,
        max_tensor_size: 4096,
    };
    let pool = Arc::new(HostPtrPool::new(&backend, cfg).expect("pool construction"));

    let g0 = pool.acquire(1024).expect("slot 0 should acquire");
    let g1 = pool.acquire(1024).expect("slot 1 should acquire");
    let g2 = pool.acquire(1024).expect("slot 2 should acquire");
    assert_eq!(pool.in_use_count(), 3);

    let g3 = pool.acquire(1024);
    assert!(
        g3.is_none(),
        "4th acquire on a 3-slot pool must return None (caller falls back to staging)"
    );

    // Sanity: the three guards are all distinct slots.
    assert_ne!(g0.slot_idx(), g1.slot_idx());
    assert_ne!(g1.slot_idx(), g2.slot_idx());
    assert_ne!(g0.slot_idx(), g2.slot_idx());
}

#[test]
fn test_c_guard_drop_releases_slot() {
    let Some(backend) = try_init_backend() else {
        return;
    };
    let cfg = HostPtrPoolConfig {
        n_slots: 1,
        max_tensor_size: 4096,
    };
    let pool = Arc::new(HostPtrPool::new(&backend, cfg).expect("pool construction"));

    {
        let g = pool.acquire(1024).expect("first acquire");
        assert_eq!(pool.in_use_count(), 1);
        assert_eq!(g.slot_idx(), 0);
        // g goes out of scope here.
    }
    assert_eq!(
        pool.in_use_count(),
        0,
        "slot must be released back to the pool on Drop"
    );

    // Re-acquire should succeed since the slot is free again.
    let g_again = pool
        .acquire(1024)
        .expect("re-acquire must succeed after drop");
    assert_eq!(g_again.slot_idx(), 0, "single-slot pool reuses slot 0");
}

#[test]
fn test_d_acquire_rejects_oversized_request() {
    let Some(backend) = try_init_backend() else {
        return;
    };
    let cfg = HostPtrPoolConfig {
        n_slots: 2,
        max_tensor_size: 4096,
    };
    let pool = Arc::new(HostPtrPool::new(&backend, cfg).expect("pool construction"));

    // Within capacity → acquires.
    let _g_ok = pool.acquire(4096).expect("acquire at exactly capacity");
    assert_eq!(pool.in_use_count(), 1);

    // Beyond capacity → returns None (graceful fallback signal).
    let g_too_big = pool.acquire(4097);
    assert!(
        g_too_big.is_none(),
        "size > max_tensor_size must return None, not a hard error"
    );
    assert_eq!(
        pool.in_use_count(),
        1,
        "rejected acquire must not consume a slot"
    );
}

#[test]
fn test_e_env_gate_default_off() {
    // The env-gate is read-once via OnceLock. Snapshot the cached value;
    // we cannot reliably mutate the env in a multi-test process without
    // disturbing other tests, so this just validates the documented default.
    //
    // When the gate is OFF (default) `host_ptr_pool_or_init` must return
    // `None` even with a valid OpenCL backend. When ON, the test below
    // (test_g) exercises the success path.
    let Some(backend) = try_init_backend() else {
        return;
    };
    let cfg = HostPtrPoolConfig {
        n_slots: 1,
        max_tensor_size: 4096,
    };
    if host_ptr_pool_env_enabled() {
        // Env was set before the test process started — the default-off
        // assertion does not apply. Just exercise the lazy-init path.
        let pool = backend.host_ptr_pool_or_init(cfg);
        assert!(pool.is_some(), "env=ON: pool must initialise");
        let pool2 = backend.host_ptr_pool_or_init(cfg);
        assert!(
            Arc::ptr_eq(&pool.unwrap(), &pool2.unwrap()),
            "host_ptr_pool_or_init must be idempotent"
        );
    } else {
        let pool = backend.host_ptr_pool_or_init(cfg);
        assert!(
            pool.is_none(),
            "env=OFF: host_ptr_pool_or_init must return None"
        );
    }
}

#[test]
fn test_f_alloc_host_ptr_buffer_empty_succeeds() {
    let Some(backend) = try_init_backend() else {
        return;
    };
    // 4 KiB minimum granularity to make sure the driver doesn't reject
    // anything subtle on size 0.
    let mem = backend
        .alloc_host_ptr_buffer_empty(4096)
        .expect("alloc_host_ptr_buffer_empty must succeed");
    // The Mem handle drops at end of scope; no leaks expected.
    drop(mem);
}

#[test]
fn test_g_fill_host_ptr_buffer_byte_equal_readback() {
    let Some(backend) = try_init_backend() else {
        return;
    };
    // ALLOC_HOST_PTR fill→readback 왕복은 UMA(host unified memory) 전제 — LISWAP-3 의
    // 설계 타겟(ARM SoC zero-copy). discrete GPU(NVIDIA 실측 2026-06-12)는
    // READ_ONLY|ALLOC_HOST_PTR 버퍼의 `clEnqueueReadBuffer` 를 CL_INVALID_OPERATION 으로
    // 거부한다. `memory/opencl/unified.rs` 의 non-UMA skip 가드와 동일 패턴.
    let unified = matches!(
        backend
            .queue
            .device()
            .info(ocl::core::DeviceInfo::HostUnifiedMemory),
        Ok(ocl::core::DeviceInfoResult::HostUnifiedMemory(true))
    );
    if !unified {
        eprintln!("[SKIPPED] non-UMA device (host_unified_memory=false)");
        return;
    }
    const N: usize = 256;
    let src: Vec<u8> = (0..N).map(|i| (i as u8).wrapping_mul(7)).collect();

    let mem = backend
        .alloc_host_ptr_buffer_empty(N)
        .expect("alloc must succeed");
    // SAFETY: src lives for the duration of the call; mem is
    // ALLOC_HOST_PTR-allocated of N bytes.
    unsafe {
        backend
            .fill_host_ptr_buffer(&mem, src.as_ptr(), N)
            .expect("fill must succeed");
    }

    // Read back via clEnqueueReadBuffer — the same pattern Stage 1b used.
    let mut dst = vec![0u8; N];
    unsafe {
        ocl::core::enqueue_read_buffer(
            &backend.queue,
            &mem,
            true,
            0,
            &mut dst,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )
        .expect("read_buffer must succeed");
    }
    assert_eq!(dst, src, "ALLOC_HOST_PTR fill must be byte-equal to source");
}

/// Stage 4 fix — H: `get_cl_mem` must recognise a `HostPtrPoolBuffer` and
/// return `Ok(&Mem)` instead of the `"Unknown"` error that caused all
/// zerocopy decode steps to fail before this fix.
#[test]
fn test_get_cl_mem_for_host_ptr_pool_buffer() {
    let Some(backend) = try_init_backend() else {
        return;
    };
    let cfg = HostPtrPoolConfig {
        n_slots: 1,
        max_tensor_size: 4096,
    };
    let pool = Arc::new(HostPtrPool::new(&backend, cfg).expect("pool construction"));
    let guard = pool.acquire(1024).expect("acquire slot");
    let buf = HostPtrPoolBuffer::new(guard, 1024, DType::F32, None);

    let result = get_cl_mem(&buf);
    assert!(
        result.is_ok(),
        "get_cl_mem must accept HostPtrPoolBuffer; got: {:?}",
        result.err()
    );
}

/// Stage 4 fix — I: `buffer_kind_label` must return `"HostPtrPool"` for a
/// `HostPtrPoolBuffer` so that diagnostic messages clearly identify the
/// pool-backed path rather than falling through to `"Unknown"`.
#[test]
fn test_buffer_kind_label_for_host_ptr_pool_buffer() {
    let Some(backend) = try_init_backend() else {
        return;
    };
    let cfg = HostPtrPoolConfig {
        n_slots: 1,
        max_tensor_size: 4096,
    };
    let pool = Arc::new(HostPtrPool::new(&backend, cfg).expect("pool construction"));
    let guard = pool.acquire(1024).expect("acquire slot");
    let buf = HostPtrPoolBuffer::new(guard, 1024, DType::F32, None);

    let label = buffer_kind_label(&buf);
    assert_eq!(
        label, "HostPtrPool",
        "buffer_kind_label must return \"HostPtrPool\" for HostPtrPoolBuffer"
    );
}
