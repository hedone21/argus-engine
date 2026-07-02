//! `InTreeQuantWindowShim` — EPIC 4 PR3 de-risk: the FIRST in-tree [`QuantCacheBackend`]
//! over the REAL [`QuantizedRecentWindowCache`] (vs the synthetic `SmokeCache` fake in
//! `capability/dynamic_backend_registry.rs`). It validates that the existing CACHE-category
//! ABI surface (`crates/argus-extension-api/src/lib.rs`, trait `QuantCacheBackend`) can drive
//! the real cache, BEFORE PR4 commits the ~40-file container move.
//!
//! The PR3 spike (`docs/design/epic4-pr3-shim-spike.md`) concluded `FUSE_PR3_PR4`: a standalone
//! PR3 that *reroutes* `new_gpu` through `resolve_quant_cache_backend` is infeasible, because
//! construction needs engine `Arc<dyn Backend>`/`Arc<dyn Memory>` a POD `QuantCacheMakeArgs`
//! cannot carry, and the residual-flush counter writes interleave with the ring-shift in one
//! `&mut self` body. This shim is the *honest* residue of that finding: an ENGINE-constructed
//! pass-through (NOT registered in `QUANT_CACHE_REGS`, whose `make` is POD-only) that proves the
//! ABI READ surface fits the real cache, and turns each spike finding into compilable code.
//!
//! ## Validated here (host, CI-green)
//! - The trait surface maps onto the real cache's accessors (it compiles ⇒ the surface fits).
//! - The READ marshalling (`assemble_view`/`get_raw_buffers` via `get_cl_mem`) reuses the proven
//!   `QuantWindowFormat::attention_native` extraction (`quant_window_format.rs`) — PR4 inherits it.
//! - The scalar/control surface forwards faithfully (CPU-fallback cache, host unit tests below).
//! - The on-device GPU READ round-trip (`#[ignore]` test, run on Adreno `R3CY408S4HN` with the
//!   cross-built `kivi` plugin via `KIVI_PLUGIN_SO`) = the maintainer's one device cycle: valid
//!   non-null `cl_mem` handles + correct scalars across ≥2 flush boundaries, the assembled-view
//!   layout faithfully propagated, and the forward `transition_bits(→16)` dequant, for bits 2/4/8.
//!   The test surfaced one PRE-EXISTING cache bug (NOT an ABI fault): the reverse `16→Q` requant
//!   over-read the GPU residual buffer (`transition_bits(16)` grew only the CPU residual Vec, not
//!   `slab.res_k`/`res_v`) → CL_INVALID_VALUE. FIXED as bug 42 (`new_gpu` now sizes the GPU residual
//!   to `res_cap`, and the Q→16 branch regrows + uploads it); the device test now asserts the
//!   back-transition returns `RC_OK`.
//!
//! ## Deferred to PR4 — each surfaced concretely in code, not hand-waved
//! 1. **`update` write marshalling.** The cache's `update` takes `&Tensor` even on its GPU path
//!    (`update_gpu`); the ABI hands raw `cl_mem`. Re-wrapping a *borrowed* `cl_mem` into a
//!    `Tensor` needs a `clRetainMemObject` to avoid the `Mem::from_raw_create_ptr` double-free
//!    (ABI C5 "borrow-for-call"). That is refcount-sensitive, device-only code → PR4 owns it with
//!    on-device verification. Here [`QuantCacheBackend::update`] returns [`RC_UPDATE_DEFERRED`] so
//!    the gap is explicit; writes in the de-risk go through the engine-side [`InTreeQuantWindowShim::feed`].
//! 2. **`flush_if_full` has no cache counterpart.** Flush is auto-triggered *inside* `update`
//!    (when `res_pos >= res_cap`). The ABI models flush as a separate step → here it is a no-op
//!    (the engine-side write already flushed). PR4 decides whether to expose an explicit flush.
//! 3. **`reset()` + AWQE `set_attn_scores` are NOT on the trait.** A live reroute that forgets
//!    them loses turn-boundary reset and silently DROPS AWQE score absorption. Both must be added
//!    to the ABI in PR4. `reset` is offered here only as the engine-side [`InTreeQuantWindowShim::reset`].
//! 4. **Construction needs engine handles.** `new_gpu` consumes `Arc<dyn Backend>`/`Arc<dyn Memory>`
//!    that POD `QuantCacheMakeArgs` cannot carry ⇒ this shim wraps an already-constructed cache via
//!    [`InTreeQuantWindowShim::wrap`]; it does not (and cannot) register a POD `make`.

use std::sync::Mutex;

use anyhow::Result;
use argus_extension_api::{
    QuantCacheBackend, QuantCacheRawBuffersOut, QuantCacheUpdateArgs, QuantCacheViewOut,
};
// Used by the opencl-gated `layout_tag` and by the `cfg(test)` assertions; keep it under either.
#[cfg(any(feature = "opencl", test))]
use argus_extension_api::ViewLayoutTag;

use crate::kv::quant_window_cache::QuantizedRecentWindowCache;
// Only consumed by `layout_tag`, which is reached solely from the opencl-gated `assemble_view`.
#[cfg(feature = "opencl")]
use crate::kv_cache_ops::KVLayout;
use crate::tensor::Tensor;

/// Work-fn success (ABI contract: 0 = OK, negative = error).
const RC_OK: i32 = 0;
/// `assemble_view` on a non-GPU cache: the assembled view tensors are host buffers with no
/// `cl_mem` to marshal across the ABI. The CPU path uses the F32 view directly, not this seam.
const RC_NO_CL_MEM: i32 = -1;
/// `transition_bits` underlying `Result` was `Err`.
const RC_TRANSITION_ERR: i32 = -2;
/// `update` is intentionally not wired in the de-risk — the cl_mem→Tensor write marshalling is
/// refcount-sensitive, device-only, and lands in PR4 (see module docs, deferred item 1).
pub const RC_UPDATE_DEFERRED: i32 = -100;

/// Engine-constructed pass-through wrapping the real [`QuantizedRecentWindowCache`] behind the
/// CACHE-category ABI. `&self` trait methods reach the cache's `&mut self` methods through the
/// `Mutex` (exactly as `QuantWindowFormat` wraps the same cache).
pub struct InTreeQuantWindowShim {
    inner: Mutex<QuantizedRecentWindowCache>,
}

impl InTreeQuantWindowShim {
    /// Wrap an already-constructed cache. The cache must be built engine-side (`new_gpu`/`new`)
    /// because its construction needs `Backend`/`Memory` handles a POD `make` cannot carry
    /// (module docs, deferred item 4).
    pub fn wrap(cache: QuantizedRecentWindowCache) -> Self {
        Self {
            inner: Mutex::new(cache),
        }
    }

    /// Engine-side write seam. Writes stay engine-side in the de-risk (the trait's cl_mem
    /// [`QuantCacheBackend::update`] is PR4, deferred item 1); the harness/decode path feeds
    /// `&Tensor` K/V exactly as production does today.
    pub fn feed(&self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        self.inner.lock().unwrap().update(new_k, new_v)
    }

    /// Engine-side reset seam. `reset` is not on the `QuantCacheBackend` trait (deferred item 3);
    /// PR4 must add it to the ABI for turn-boundary eviction.
    pub fn reset(&self) {
        self.inner.lock().unwrap().reset();
    }
}

/// `KVLayout` (engine) → `ViewLayoutTag` (ABI closed vocabulary). Keeps the ABI from naming the
/// engine's `KVLayout`. Only used by the opencl-gated `assemble_view` cl_mem path.
#[cfg(feature = "opencl")]
fn layout_tag(layout: KVLayout) -> ViewLayoutTag {
    match layout {
        KVLayout::SeqMajor => ViewLayoutTag::SeqMajor,
        KVLayout::HeadMajor => ViewLayoutTag::HeadMajor,
    }
}

impl QuantCacheBackend for InTreeQuantWindowShim {
    fn current_pos(&self) -> usize {
        self.inner.lock().unwrap().current_pos()
    }

    fn capacity(&self) -> usize {
        self.inner.lock().unwrap().capacity()
    }

    fn current_bits(&self) -> u8 {
        self.inner.lock().unwrap().bits()
    }

    fn update(&self, _args: &QuantCacheUpdateArgs) -> i32 {
        // Deferred item 1: the cl_mem→Tensor borrow-retain write marshalling is device-only and
        // lands in PR4. Engine-side writes go through `feed`. Returning a sentinel keeps the gap
        // explicit rather than silently dropping the write.
        RC_UPDATE_DEFERRED
    }

    fn flush_if_full(&self) -> i32 {
        // Deferred item 2: the cache auto-flushes inside `update`; there is no separate
        // flush-if-full step to forward to. No-op by construction.
        RC_OK
    }

    fn assemble_view(&self, out: &mut QuantCacheViewOut) -> i32 {
        // The assembled-view seam marshals `cl_mem` handles and is OpenCL-only. On a non-opencl
        // (CUDA/CPU) build there is no cl_mem, so return the no-cl-mem sentinel — exactly what the
        // opencl path returns for host buffers (callers then use the F32 view). The CUDA analog
        // lands with the KIVI-CUDA cap.
        #[cfg(not(feature = "opencl"))]
        {
            let _ = out;
            RC_NO_CL_MEM
        }
        #[cfg(feature = "opencl")]
        {
            use crate::backend::opencl::get_cl_mem;
            let mut cache = self.inner.lock().unwrap();
            let layout = cache.layout();
            let tokens = cache.current_pos();
            let (k_view, v_view) = cache.get_view();
            // GPU mode: the assembled view buffers are `cl_mem`. CPU mode: host buffers → no cl_mem,
            // and attention uses the F32 view directly, not this seam (RC_NO_CL_MEM).
            let (Ok(k_mem), Ok(v_mem)) = (
                get_cl_mem(k_view.buffer().as_ref()),
                get_cl_mem(v_view.buffer().as_ref()),
            ) else {
                return RC_NO_CL_MEM;
            };
            out.k_mem = k_mem.as_ptr();
            out.v_mem = v_mem.as_ptr();
            out.tokens = tokens;
            out.layout = layout_tag(layout) as u32;
            RC_OK
        }
    }

    fn get_raw_buffers(&self, out: &mut QuantCacheRawBuffersOut) -> bool {
        // Raw `cl_mem` buffer export is OpenCL-only; a non-opencl build has no cl_mem to hand out.
        #[cfg(not(feature = "opencl"))]
        {
            let _ = out;
            false
        }
        #[cfg(feature = "opencl")]
        {
            use crate::backend::opencl::get_cl_mem;
            let cache = self.inner.lock().unwrap();
            // None on CPU / bits=16 / empty — exactly the native-attention gate
            // (`QuantWindowFormat::attention_into`).
            let Some(raw) = cache.get_quant_window_raw_buffers() else {
                return false;
            };
            let (Ok(qk), Ok(qv), Ok(rk), Ok(rv)) = (
                get_cl_mem(raw.qk_buf.buffer().as_ref()),
                get_cl_mem(raw.qv_buf.buffer().as_ref()),
                get_cl_mem(raw.res_k.buffer().as_ref()),
                get_cl_mem(raw.res_v.buffer().as_ref()),
            ) else {
                return false;
            };
            out.qk_mem = qk.as_ptr();
            out.qv_mem = qv.as_ptr();
            out.res_k_mem = rk.as_ptr();
            out.res_v_mem = rv.as_ptr();
            out.q_tokens = raw.q_tokens;
            out.res_tokens = raw.res_tokens;
            out.res_cap = raw.res_cap;
            out.bits = raw.bits;
            true
        }
    }

    fn transition_bits(&self, target_bits: u8) -> i32 {
        match self.inner.lock().unwrap().transition_bits(target_bits) {
            Ok(()) => RC_OK,
            Err(_) => RC_TRANSITION_ERR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use std::sync::Arc;

    // QuantizedRecentWindowCache requires residual_size and head_dim to be multiples of QKKV(=32).
    const HD: usize = 32;
    const RES: usize = 32;
    const MAXSEQ: usize = 256;

    fn f32_tensor(dims: Vec<usize>, val: f32) -> Tensor {
        let n: usize = dims.iter().product();
        let buf = Arc::new(SharedBuffer::new(n * 4, DType::F32));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        t.as_mut_slice::<f32>().fill(val);
        t
    }

    fn null_view_out() -> QuantCacheViewOut {
        QuantCacheViewOut {
            k_mem: std::ptr::null_mut(),
            v_mem: std::ptr::null_mut(),
            tokens: 0,
            layout: 0,
        }
    }

    fn null_raw_out() -> QuantCacheRawBuffersOut {
        QuantCacheRawBuffersOut {
            qk_mem: std::ptr::null_mut(),
            qv_mem: std::ptr::null_mut(),
            res_k_mem: std::ptr::null_mut(),
            res_v_mem: std::ptr::null_mut(),
            q_tokens: 0,
            res_tokens: 0,
            res_cap: 0,
            bits: 0,
        }
    }

    #[test]
    fn scalars_forward_to_real_cache() {
        let shim = InTreeQuantWindowShim::wrap(QuantizedRecentWindowCache::new(2, HD, MAXSEQ, RES));
        assert_eq!(shim.current_pos(), 0);
        assert_eq!(shim.capacity(), MAXSEQ); // CPU mode capacity == max_seq_len
        assert_eq!(shim.current_bits(), 2); // new() default

        // engine-side write seam advances the real cache.
        let k = f32_tensor(vec![1, 1, 2, HD], 1.0);
        let v = f32_tensor(vec![1, 1, 2, HD], 1.0);
        shim.feed(&k, &v).unwrap();
        assert_eq!(shim.current_pos(), 1);

        shim.reset();
        assert_eq!(shim.current_pos(), 0);
    }

    #[test]
    fn update_is_explicitly_deferred() {
        // The trait's cl_mem write path returns the sentinel (deferred item 1), not a silent drop.
        let shim = InTreeQuantWindowShim::wrap(QuantizedRecentWindowCache::new(1, HD, MAXSEQ, RES));
        let args = QuantCacheUpdateArgs {
            cl_queue: std::ptr::null_mut(),
            k_in_mem: std::ptr::null_mut(),
            v_in_mem: std::ptr::null_mut(),
            seq_len: 1,
        };
        assert_eq!(shim.update(&args), RC_UPDATE_DEFERRED);
    }

    #[test]
    fn flush_if_full_is_noop() {
        let shim = InTreeQuantWindowShim::wrap(QuantizedRecentWindowCache::new(1, HD, MAXSEQ, RES));
        assert_eq!(shim.flush_if_full(), RC_OK);
    }

    #[test]
    fn cpu_read_surface_has_no_cl_mem() {
        // CPU-fallback cache: get_raw_buffers → false (gate: !is_gpu), assemble_view → RC_NO_CL_MEM
        // (host view buffers carry no cl_mem). This is the faithful CPU mapping, not a failure.
        let shim = InTreeQuantWindowShim::wrap(QuantizedRecentWindowCache::new(1, HD, MAXSEQ, RES));
        let k = f32_tensor(vec![1, 1, 1, HD], 1.0);
        let v = f32_tensor(vec![1, 1, 1, HD], 1.0);
        shim.feed(&k, &v).unwrap();

        let mut raw = null_raw_out();
        assert!(!shim.get_raw_buffers(&mut raw));

        let mut view = null_view_out();
        assert_eq!(shim.assemble_view(&mut view), RC_NO_CL_MEM);
    }

    #[test]
    fn transition_same_bits_is_ok() {
        let shim = InTreeQuantWindowShim::wrap(QuantizedRecentWindowCache::new(1, HD, MAXSEQ, RES));
        assert_eq!(shim.transition_bits(2), RC_OK); // 2 → 2: no-op
        assert_eq!(shim.current_bits(), 2);
    }

    /// On-device GPU READ round-trip — the maintainer's one Adreno (`R3CY408S4HN`) cycle.
    /// Validates the cl_mem marshalling yields valid non-null handles + correct scalars across
    /// ≥2 residual-flush boundaries and `transition_bits` both directions, for bits 2/4/8.
    /// `#[ignore]` because host CI has no GPU (and the KIVI Q2 kernels compile on Adreno only);
    /// run on the device with the cross-built kivi plugin in scope:
    ///   `KIVI_PLUGIN_SO=/data/local/tmp/libkivi.so \`
    ///   `  ./<test-bin> --ignored --test-threads=1 cache_abi_shim_gpu_read_round_trip`.
    #[test]
    #[ignore = "device test required: Adreno R3CY408S4HN (KIVI Q2 kernels are Adreno-only)"]
    fn cache_abi_shim_gpu_read_round_trip() {
        use crate::backend::Backend;
        use crate::backend::opencl::OpenCLBackend;
        use crate::backend::opencl::memory::OpenCLMemory;
        use crate::buffer::DType;
        use crate::capability::dynamic_backend_registry::{
            dynamic_registered_backend_cap_names, resolve_quant_attn_capability,
        };
        use crate::session::plugin_dispatch::register_dynamic_plugins;

        let Ok(be) = OpenCLBackend::new() else {
            eprintln!("[skip] no OpenCL device");
            return;
        };
        // The cache's GPU path quantizes tokens through the `kivi` QuantAttn capability, which the
        // engine names nowhere (FORMAT Phase 2 Stage E). Load it the production way — dlopen the
        // cross-built `.so` named by KIVI_PLUGIN_SO — so this stays a faithful end-to-end exercise
        // with zero static kivi reference in the engine.
        let Ok(so_path) = std::env::var("KIVI_PLUGIN_SO") else {
            eprintln!("[skip] KIVI_PLUGIN_SO unset (need cross-built libkivi.so on device)");
            return;
        };
        let so = std::path::PathBuf::from(so_path);
        register_dynamic_plugins(std::slice::from_ref(&so)).expect("kivi .so registration failed");
        assert!(
            dynamic_registered_backend_cap_names()
                .iter()
                .any(|n| n == "kivi_abi"),
            "kivi_abi not registered after dlopen"
        );

        let be = Arc::new(be);
        let backend: Arc<dyn Backend> = be.clone();
        let kv_heads = 2;

        // Real GPU memory (cl_mem allocator). `Galloc` would hand back host `SharedBuffer`s, so both
        // the cache's internal buffers and the fed tensors must come from the backend's own OpenCL
        // context for the round-trip to exercise actual cl_mem.
        let gpu_mem: Arc<dyn crate::memory::Memory> = Arc::new(OpenCLMemory::new(
            be.context.clone(),
            be.queue.clone(),
            false,
        ));

        // Upload a constant-filled f32 tensor onto the device (mirrors the engine's QKV-output
        // tensors that feed the cache during decode).
        let upload = |dims: Vec<usize>, val: f32| -> Tensor {
            let n: usize = dims.iter().product();
            let buf = gpu_mem.alloc(n * 4, DType::F32).unwrap();
            let mut t = Tensor::new(Shape::new(dims), buf, backend.clone());
            let host = vec![val; n];
            // SAFETY: reinterpret the f32 slice as its byte view for the device upload.
            let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, n * 4) };
            backend.write_buffer(&mut t, bytes).unwrap();
            t
        };

        for bits in [2u8, 4, 8] {
            // Build the kivi handle from the *live* GPU context (same path as engine init), so it
            // shares the backend's ocl instance.
            let quant_attn = be
                .with_quant_attn_make_args(|ma| resolve_quant_attn_capability("kivi_abi", ma))
                .expect("resolve kivi_abi handle from live GPU context");
            let cache = QuantizedRecentWindowCache::new_gpu(
                kv_heads,
                HD,
                MAXSEQ,
                RES,
                bits,
                backend.clone(),
                Some(quant_attn),
                None, // OpenCL host-harness — no CUDA cap.
                gpu_mem.clone(),
            );
            assert!(cache.is_gpu(), "real OpenCL backend must enter GPU mode");
            let shim = InTreeQuantWindowShim::wrap(cache);

            // Feed > 2*res_cap tokens so the residual ring flushes at least twice.
            let fed = RES * 2 + RES / 2;
            for i in 0..fed {
                let k = upload(vec![1, 1, kv_heads, HD], i as f32 * 0.01);
                let v = upload(vec![1, 1, kv_heads, HD], i as f32 * 0.01 + 1.0);
                shim.feed(&k, &v).unwrap();
            }
            assert_eq!(shim.current_pos(), fed);
            assert_eq!(shim.current_bits(), bits);

            // Raw-buffer marshalling: real GPU handles, non-null, counts consistent.
            let mut raw = null_raw_out();
            assert!(
                shim.get_raw_buffers(&mut raw),
                "bits={bits}: GPU cache with quantized tokens must expose raw buffers"
            );
            assert!(!raw.qk_mem.is_null() && !raw.qv_mem.is_null());
            assert!(!raw.res_k_mem.is_null() && !raw.res_v_mem.is_null());
            assert_eq!(raw.q_tokens + raw.res_tokens, fed);
            assert_eq!(raw.bits, bits);
            assert_eq!(raw.res_cap, RES);

            // Assembled-view marshalling: non-null handles, full tokens, and the layout the cache
            // reports faithfully propagated. For Q2/Q4/Q8 GPU mode the assembled attn view is
            // SeqMajor [total, kv_heads, head_dim] (HeadMajor is the bits=16 residual layout).
            let mut view = null_view_out();
            assert_eq!(shim.assemble_view(&mut view), RC_OK, "bits={bits}");
            assert!(!view.k_mem.is_null() && !view.v_mem.is_null());
            assert_eq!(view.tokens, fed);
            assert_eq!(view.layout, ViewLayoutTag::SeqMajor as u32, "bits={bits}");

            // Forward transition through the trait — the production-relevant direction (quantized →
            // full-precision F16 window). Drives a real GPU dequant and round-trips the ABI slot.
            assert_eq!(shim.transition_bits(16), RC_OK, "bits={bits} → 16");
            assert_eq!(shim.current_bits(), 16);

            // The reverse (16 → Q) requant. This was bug 42: `transition_bits(16)` grew only the
            // CPU residual Vec (`self.res_k.resize`) + set `res_cap = max_seq_len`, but left the
            // GPU `slab.res_k`/`res_v` at their residual_size capacity, so the back-transition's
            // "read GPU F32 residual → CPU" step (`res_bytes = kv_heads * res_cap * head_dim * 4`)
            // over-read the smaller GPU buffer → CL_INVALID_VALUE. FIXED: new_gpu now sizes the GPU
            // residual to res_cap, and the Q→16 branch regrows + uploads it, so the readback is
            // in-bounds. The back-transition must now succeed.
            let back = shim.transition_bits(bits);
            assert_eq!(
                back, RC_OK,
                "16 → {bits}: requant must succeed post-bug-42-fix"
            );
            assert_eq!(
                shim.current_bits(),
                bits,
                "bits={bits}: back at target width"
            );
        }
    }
}
