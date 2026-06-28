//! kivi — KIVI (Q2/Q4/Q8 quantized KV + recent-window F32 residual) GPU attention/flush
//! kernels as a backend-capability (ATTENTION category) dlopen plugin.
//!
//! FORMAT Phase 2 **Stage E**: the engine core holds ZERO KIVI GPU-kernel source/compile/
//! dispatch. This plugin owns `kivi_q2.cl` + `kivi_attn.cl` (moved verbatim out of
//! `engine/kernels/`), compiles them **from the host's borrowed `cl_context`** in `make()`
//! — the same runtime compile the engine used to do — and enqueues every kernel on the
//! host's live `cl_command_queue` (lent per call via `QuantAttn*Args.cl_queue`, populated by
//! the engine's call sites in Stage E). It is the first plugin to compile real OpenCL from a
//! borrowed cdylib context (all other backend-cap plugins are synthetic and ignore `cl_ctx`).
//!
//! **Default-path policy (Stage E option b):** the engine no longer registers a built-in
//! QuantAttn cap, and it contains no `kivi`-specific branch. To run GPU KIVI you must load
//! this plugin and select it: `--load-plugin <kivi>.so --backend-cap kivi_abi`. Without a
//! cap, the (engine-resident) `QuantizedRecentWindowCache` errors from its own GPU flush
//! path — the requirement lives in the KIVI domain, not in the engine core.
//!
//! **Borrow/lifetime (C5/C7/D4):** `make` reconstructs the context via
//! `Context::from_raw_copied_ptr` (which `clRetainContext`s, so the clone outlives the
//! `FnOnce`) and balances it with the `ocl::core::Context` `Drop` (`clReleaseContext`) invoked
//! once through the vtable `drop`. Per-call cl_mem are wrapped borrow-only (`ManuallyDrop`, no
//! retain); the per-call queue is reconstructed (retain) and released on drop. **panic=abort:**
//! every fallible step uses `.ok()`/`.map_err` and null handles are guarded — no `unwrap`/
//! `expect` is reachable except `from_raw_copied_ptr`'s internal `retain_context().unwrap()`,
//! which the non-null `cl_ctx`/`cl_queue` guards make unreachable for any valid handle.

use std::ffi::{CStr, CString};

use argus_extension_api::{
    QuantAttnArgs, QuantAttnBackend, QuantAttnGatherArgs, QuantAttnMakeArgs, QuantDequantFlushArgs,
    QuantScatterResidualArgs,
};
use ocl::core::{
    self, ArgVal, CommandQueue, Context as ContextCore, DeviceId, Event, Kernel, Mem, Program,
};

/// KIVI Q2 dequant/scatter/gather kernels (moved from `engine/kernels/kivi_q2.cl`).
const KIVI_Q2_SRC: &str = include_str!("kivi_q2.cl");
/// KIVI fused native attention kernels (moved from `engine/kernels/kivi_attn.cl`).
const KIVI_ATTN_SRC: &str = include_str!("kivi_attn.cl");

/// Live GPU resources reconstructed from the host's borrowed context in `make()`.
struct KiviGpuInner {
    /// Retained copy of the host context; kept alive for the programs/kernels and used to
    /// allocate the per-call score buffer. `Drop` releases it (balances the make-time retain).
    ctx: ContextCore,
    /// The two compiled programs are kept alive for the lifetime of the kernels created from
    /// them (mirrors the engine holding `kivi_q2_program`/`kivi_attn_program`).
    #[allow(dead_code)]
    prog_q2: Program,
    #[allow(dead_code)]
    prog_attn: Program,
    /// 1-element placeholder bound to the attention kernel's score arg when no scores are
    /// requested (mirrors `OpenCLBackend::dummy_score_buf`).
    dummy_score_buf: Mem,
    /// Mirrors the engine's `f16_is_nosub` default-path determination (`!LLMRS_F16_GEMV_USE_4WAVE`).
    /// Advisory only: the engine reads the nosub property off the backend, not this cap.
    is_nosub: bool,
    // The 7 live kernels (None if `create_kernel` failed → the corresponding op returns -1 and
    // `has_quant_attn_kernel` reports false, so the engine gate falls back to the F32 path).
    deq_value_q2_f16: Option<Kernel>,
    deq_key_q2_f16: Option<Kernel>,
    scatter_residual_f16: Option<Kernel>,
    gather_update: Option<Kernel>,
    attn_q2: Option<Kernel>,
    attn_q4: Option<Kernel>,
    attn_q8: Option<Kernel>,
}

/// The capability handle. `inner == None` is the degraded state (null context, program build
/// failure, or buffer alloc failure) — every op then returns -1 and `has_quant_attn_kernel`
/// reports false, matching the engine's graceful "KIVI kernels disabled" behavior.
struct KiviGpuBackend {
    inner: Option<KiviGpuInner>,
}

// SAFETY: single-threaded inference (the same assumption the engine's `OpenCLBackend` makes
// for its `UnsafeCell<KernelCache>`). OpenCL retain/release is thread-safe; kernel arg-setting
// is serialized by the single decode loop. The raw handles belong to the host context.
unsafe impl Send for KiviGpuBackend {}
unsafe impl Sync for KiviGpuBackend {}

/// Wrap a borrowed raw `cl_mem` (host-owned) for kernel-arg use WITHOUT taking ownership
/// (C5 borrow-only). `Mem` has a releasing `Drop`, so it is returned in `ManuallyDrop` and
/// must never be dropped; `Deref` yields `&Mem` for `ArgVal::mem`. Mirrors
/// `OpenCLBackend::borrow_cl_mem`.
///
/// # Safety
/// `ptr` must be a valid, non-null `cl_mem` of the plugin's context, live for the call.
#[inline]
unsafe fn borrow_mem(ptr: *mut std::ffi::c_void) -> std::mem::ManuallyDrop<Mem> {
    std::mem::ManuallyDrop::new(unsafe { Mem::from_raw_create_ptr(ptr as ocl::ffi::cl_mem) })
}

/// Reconstruct the host's live command queue from the lent raw handle (retain; the returned
/// `CommandQueue`'s `Drop` releases). Returns `None` on null (panic=abort guard, since
/// `from_raw_copied_ptr` asserts non-null + unwraps the retain).
#[inline]
fn reconstruct_queue(ptr: *mut std::ffi::c_void) -> Option<CommandQueue> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the host lends its live `cl_command_queue` (Stage E populated the slot);
    // from_raw_copied_ptr retains it so this clone is valid for the call and released on drop.
    Some(unsafe { CommandQueue::from_raw_copied_ptr(ptr as ocl::ffi::cl_command_queue) })
}

/// Compile one `.cl` source into a program for `device`, using the host's verbatim build
/// options (Adreno fast-math consistency — never recompute them). Reproduces the engine's
/// `Program::builder().devices(device).cmplr_opt(build_opts).src(src).build(&ctx)`.
fn build_program_src(
    ctx: &ContextCore,
    device: DeviceId,
    src: &str,
    opts: &CString,
) -> Option<Program> {
    let src_c = CString::new(src).ok()?;
    let prog = core::create_program_with_source(ctx, &[src_c]).ok()?;
    core::build_program(&prog, Some(&[device]), opts, None, None).ok()?;
    Some(prog)
}

/// Reconstruct the GPU resources from the borrowed make-args. Any failure → `None` (degraded).
fn build_inner(args: &QuantAttnMakeArgs) -> Option<KiviGpuInner> {
    // panic=abort guard: from_raw_copied_ptr asserts non-null + unwraps the retain.
    if args.cl_ctx.is_null() || args.device.is_null() {
        return None;
    }
    // SAFETY (C7/D4): borrow-for-make; from_raw_copied_ptr clRetainContexts so the clone
    // outlives the FnOnce, balanced by the ContextCore Drop via the vtable drop.
    let ctx = unsafe { ContextCore::from_raw_copied_ptr(args.cl_ctx as ocl::ffi::cl_context) };
    // SAFETY: cl_device_id is not refcounted; from_raw just wraps it. Non-null checked above.
    let device = unsafe { DeviceId::from_raw(args.device as ocl::ffi::cl_device_id) };

    // Use the host's exact build_opts (== its build_cl_opts(device)); null → empty.
    let opts = if args.build_opts.is_null() {
        CString::new("").ok()?
    } else {
        // SAFETY: the host passes a NUL-terminated options string owned for the make call.
        unsafe { CStr::from_ptr(args.build_opts) }.to_owned()
    };

    let prog_q2 = build_program_src(&ctx, device, KIVI_Q2_SRC, &opts)?;
    let prog_attn = build_program_src(&ctx, device, KIVI_ATTN_SRC, &opts)?;

    // 1-element dummy score buffer (mirrors OpenCLBackend::dummy_score_buf).
    // SAFETY: ctx is a valid retained context.
    let dummy_score_buf =
        unsafe { core::create_buffer::<_, f32>(&ctx, core::MEM_READ_WRITE, 1, None).ok()? };

    // Mirror the engine's f16_is_nosub default-path value (= !use_4wave); advisory only.
    let use_4wave = std::env::var("LLMRS_F16_GEMV_USE_4WAVE")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);

    Some(KiviGpuInner {
        deq_value_q2_f16: core::create_kernel(&prog_q2, "kivi_dequantize_value_q2_f16").ok(),
        deq_key_q2_f16: core::create_kernel(&prog_q2, "kivi_dequantize_key_q2_f16").ok(),
        scatter_residual_f16: core::create_kernel(&prog_q2, "kivi_scatter_residual_f16").ok(),
        gather_update: core::create_kernel(&prog_q2, "kivi_gather_update").ok(),
        attn_q2: core::create_kernel(&prog_attn, "kernel_attn_gen_kivi_q2").ok(),
        attn_q4: core::create_kernel(&prog_attn, "kernel_attn_gen_kivi_q4").ok(),
        attn_q8: core::create_kernel(&prog_attn, "kernel_attn_gen_kivi_q8").ok(),
        prog_q2,
        prog_attn,
        dummy_score_buf,
        is_nosub: !use_4wave,
        ctx,
    })
}

impl KiviGpuInner {
    /// Port of `OpenCLBackend::kivi_gather_update` — enqueue on the lent queue.
    fn gather_update(&self, args: &QuantAttnGatherArgs) -> Result<(), String> {
        let kernel = self
            .gather_update
            .as_ref()
            .ok_or("gather_update kernel N/A")?;
        let queue = reconstruct_queue(args.cl_queue).ok_or("null cl_queue")?;
        // SAFETY: borrow-only cl_mem (host retains ownership).
        let input_mem = unsafe { borrow_mem(args.input_mem) };
        let residual_mem = unsafe { borrow_mem(args.residual_mem) };
        let total = args.seq_len * args.kv_heads * args.head_dim;
        let (kv_heads_i, res_cap_i, head_dim_i, seq_len_i, res_pos_i) = (
            args.kv_heads as i32,
            args.res_cap as i32,
            args.head_dim as i32,
            args.seq_len as i32,
            args.res_pos as i32,
        );
        set_args(
            kernel,
            [
                ArgVal::mem(&input_mem),
                ArgVal::mem(&residual_mem),
                ArgVal::scalar(&kv_heads_i),
                ArgVal::scalar(&res_cap_i),
                ArgVal::scalar(&head_dim_i),
                ArgVal::scalar(&seq_len_i),
                ArgVal::scalar(&res_pos_i),
            ],
        )?;
        // SAFETY: validated handles; single-threaded; mems live until the call returns.
        unsafe { enqueue(&queue, kernel, 3, &[total, 1, 1], None) }?;
        Ok(())
    }

    /// Port of `OpenCLBackend::kivi_dequantize_{value,key}_q2_f16_core` (selected by `is_key`).
    fn dequant_flush(&self, args: &QuantDequantFlushArgs) -> Result<(), String> {
        if args.bits != 2 {
            // Only Q2 has a GPU dequant kernel (Q4/Q8 flush is CPU in the engine, unchanged).
            return Err(format!("dequant_flush unsupported for bits={}", args.bits));
        }
        let (kernel, total) = if args.is_key {
            (
                self.deq_key_q2_f16
                    .as_ref()
                    .ok_or("deq_key_q2_f16 kernel N/A")?,
                // K: per-channel — kv_heads * groups_per_flush * head_dim
                args.kv_heads * args.n_groups_or_tokens * args.head_dim,
            )
        } else {
            (
                self.deq_value_q2_f16
                    .as_ref()
                    .ok_or("deq_value_q2_f16 kernel N/A")?,
                // V: per-token — kv_heads * flush_tokens * (head_dim / 32)
                args.kv_heads * args.n_groups_or_tokens * (args.head_dim / 32),
            )
        };
        let queue = reconstruct_queue(args.cl_queue).ok_or("null cl_queue")?;
        // SAFETY: borrow-only cl_mem.
        let q2_mem = unsafe { borrow_mem(args.q_blocks_mem) };
        let attn_mem = unsafe { borrow_mem(args.attn_mem) };
        let (kv_heads_i, head_dim_i, n_i, tok_base_i, block_offset_i) = (
            args.kv_heads as i32,
            args.head_dim as i32,
            args.n_groups_or_tokens as i32,
            args.tok_base as i32,
            args.block_start as i32,
        );
        set_args(
            kernel,
            [
                ArgVal::mem(&q2_mem),
                ArgVal::mem(&attn_mem),
                ArgVal::scalar(&kv_heads_i),
                ArgVal::scalar(&head_dim_i),
                ArgVal::scalar(&n_i),
                ArgVal::scalar(&tok_base_i),
                ArgVal::scalar(&block_offset_i),
            ],
        )?;
        // SAFETY: validated handles; single-threaded; mems live until the call returns.
        unsafe { enqueue(&queue, kernel, 3, &[total, 1, 1], None) }?;
        Ok(())
    }

    /// Port of `OpenCLBackend::kivi_scatter_residual_f16_core`.
    fn scatter_residual(&self, args: &QuantScatterResidualArgs) -> Result<(), String> {
        let kernel = self
            .scatter_residual_f16
            .as_ref()
            .ok_or("scatter_residual_f16 kernel N/A")?;
        let queue = reconstruct_queue(args.cl_queue).ok_or("null cl_queue")?;
        // SAFETY: borrow-only cl_mem.
        let residual_mem = unsafe { borrow_mem(args.res_mem) };
        let attn_mem = unsafe { borrow_mem(args.attn_mem) };
        let total = args.kv_heads * args.res_pos * args.head_dim;
        let (kv_heads_i, res_cap_i, head_dim_i, res_pos_i, tok_base_i) = (
            args.kv_heads as i32,
            args.res_cap as i32,
            args.head_dim as i32,
            args.res_pos as i32,
            args.tok_base as i32,
        );
        set_args(
            kernel,
            [
                ArgVal::mem(&residual_mem),
                ArgVal::mem(&attn_mem),
                ArgVal::scalar(&kv_heads_i),
                ArgVal::scalar(&res_cap_i),
                ArgVal::scalar(&head_dim_i),
                ArgVal::scalar(&res_pos_i),
                ArgVal::scalar(&tok_base_i),
            ],
        )?;
        // SAFETY: validated handles; single-threaded; mems live until the call returns.
        unsafe { enqueue(&queue, kernel, 3, &[total, 1, 1], None) }?;
        Ok(())
    }

    /// Port of `OpenCLBackend::attention_gen_kivi` — fused Q2/Q4/Q8 native attention with
    /// optional post-softmax score readback (feeds AWQE/QCF). Enqueue + blocking readback on
    /// the SAME lent queue (ordering-preserving).
    fn attention_gen(&self, args: &QuantAttnArgs) -> Result<(), String> {
        let kernel = match args.bits {
            2 => self.attn_q2.as_ref().ok_or("Q2 attention kernel N/A")?,
            4 => self.attn_q4.as_ref().ok_or("Q4 attention kernel N/A")?,
            8 => self.attn_q8.as_ref().ok_or("Q8 attention kernel N/A")?,
            b => return Err(format!("unsupported KIVI bits: {b}")),
        };
        let queue = reconstruct_queue(args.cl_queue).ok_or("null cl_queue")?;

        // SAFETY: borrow-only cl_mem (host retains ownership).
        let q_buf = unsafe { borrow_mem(args.q_mem) };
        let qk_mem = unsafe { borrow_mem(args.qk_mem) };
        let qv_mem = unsafe { borrow_mem(args.qv_mem) };
        let res_k_mem = unsafe { borrow_mem(args.res_k_mem) };
        let res_v_mem = unsafe { borrow_mem(args.res_v_mem) };
        let o_buf = unsafe { borrow_mem(args.out_mem) };

        let want_scores = !args.scores_out.is_null() && args.scores_len > 0;
        let has_scores = want_scores as i32;
        let total_tokens = args.q_tokens + args.res_tokens;
        let score_stride_val = if want_scores {
            (args.scores_len / args.num_heads_q) as i32
        } else {
            total_tokens as i32
        };

        // Allocate a GPU score buffer when scores are requested, else bind the dummy.
        // SAFETY: ctx is valid.
        let score_buf = if want_scores {
            Some(
                unsafe {
                    core::create_buffer::<_, f32>(
                        &self.ctx,
                        core::MEM_READ_WRITE | core::MEM_ALLOC_HOST_PTR,
                        args.num_heads_q * score_stride_val as usize,
                        None,
                    )
                }
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let s_buf = score_buf.as_ref().unwrap_or(&self.dummy_score_buf);

        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        let (nhq_i, nhkv_i, hd_i, qt_i, rt_i, rc_i) = (
            args.num_heads_q as i32,
            args.num_heads_kv as i32,
            args.head_dim as i32,
            args.q_tokens as i32,
            args.res_tokens as i32,
            args.res_cap as i32,
        );
        set_args(
            kernel,
            [
                ArgVal::mem(&q_buf),
                ArgVal::mem(&qk_mem),
                ArgVal::mem(&qv_mem),
                ArgVal::mem(&res_k_mem),
                ArgVal::mem(&res_v_mem),
                ArgVal::mem(&o_buf),
                ArgVal::mem(s_buf),
                ArgVal::scalar(&nhq_i),
                ArgVal::scalar(&nhkv_i),
                ArgVal::scalar(&hd_i),
                ArgVal::scalar(&qt_i),
                ArgVal::scalar(&rt_i),
                ArgVal::scalar(&rc_i),
                ArgVal::scalar(&args.scale),
                ArgVal::scalar(&score_stride_val),
                ArgVal::scalar(&has_scores),
                ArgVal::local::<f32>(&local_mem_size),
            ],
        )?;
        // SAFETY: validated handles; single-threaded; mems live until the call returns.
        unsafe {
            enqueue(
                &queue,
                kernel,
                1,
                &[args.num_heads_q * local_size, 1, 1],
                Some([local_size, 1, 1]),
            )
        }?;

        // Read back scores to the host slice if requested, on the SAME queue (blocking).
        // `score_buf` is `Some` iff `want_scores`, so this single check covers both.
        if let Some(buf) = &score_buf {
            // SAFETY: scores_out/scores_len form a valid host slice (C5 borrow-for-call).
            let scores =
                unsafe { std::slice::from_raw_parts_mut(args.scores_out, args.scores_len) };
            // SAFETY: queue + buf valid; blocking read.
            unsafe {
                core::enqueue_read_buffer(
                    &queue,
                    buf,
                    true,
                    0,
                    scores,
                    None::<Event>,
                    None::<&mut Event>,
                )
            }
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Set kernel args by index (0..). Helper to keep the dispatch bodies terse. `set_kernel_arg`
/// is a safe ocl-core call; the args are consumed (they borrow the caller's scalar locals,
/// which outlive the call).
fn set_args<const N: usize>(kernel: &Kernel, args: [ArgVal; N]) -> Result<(), String> {
    for (i, a) in args.into_iter().enumerate() {
        core::set_kernel_arg(kernel, i as u32, a).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Enqueue a kernel on `queue` (mirrors the non-profile branch of
/// `OpenCLBackend::enqueue_kernel_labeled`; profiling is host-only and behavior-neutral).
///
/// # Safety
/// `kernel`/`queue` valid; `gws`/`lws` describe a valid launch for the kernel.
unsafe fn enqueue(
    queue: &CommandQueue,
    kernel: &Kernel,
    work_dim: u32,
    gws: &[usize; 3],
    lws: Option<[usize; 3]>,
) -> Result<(), String> {
    unsafe {
        core::enqueue_kernel(
            queue,
            kernel,
            work_dim,
            None,
            gws,
            lws,
            None::<&Event>,
            None::<&mut Event>,
        )
    }
    .map_err(|e| e.to_string())
}

impl QuantAttnBackend for KiviGpuBackend {
    fn has_quant_attn_kernel(&self, bits: u8) -> bool {
        match (&self.inner, bits) {
            (Some(i), 2) => i.attn_q2.is_some(),
            (Some(i), 4) => i.attn_q4.is_some(),
            (Some(i), 8) => i.attn_q8.is_some(),
            _ => false,
        }
    }

    fn is_nosub_device(&self) -> bool {
        // Advisory: the engine reads the nosub property off the backend, not this cap.
        self.inner.as_ref().map(|i| i.is_nosub).unwrap_or(false)
    }

    fn attention_gen_quant(&self, args: &QuantAttnArgs) -> i32 {
        match self.inner.as_ref() {
            Some(i) => match i.attention_gen(args) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("[KIVI plugin] attention_gen_quant failed: {e}");
                    -1
                }
            },
            None => -1,
        }
    }

    fn gather_update_quant(&self, args: &QuantAttnGatherArgs) -> i32 {
        match self.inner.as_ref() {
            Some(i) => match i.gather_update(args) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("[KIVI plugin] gather_update_quant failed: {e}");
                    -1
                }
            },
            None => -1,
        }
    }

    fn dequant_flush(&self, args: &QuantDequantFlushArgs) -> i32 {
        match self.inner.as_ref() {
            Some(i) => match i.dequant_flush(args) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("[KIVI plugin] dequant_flush failed: {e}");
                    -1
                }
            },
            None => -1,
        }
    }

    fn scatter_residual(&self, args: &QuantScatterResidualArgs) -> i32 {
        match self.inner.as_ref() {
            Some(i) => match i.scatter_residual(args) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("[KIVI plugin] scatter_residual failed: {e}");
                    -1
                }
            },
            None => -1,
        }
    }
}

/// make factory — reconstructs the GPU resources from the borrowed host context (Stage E).
/// On any failure returns a degraded handle (`inner == None`) so the host's
/// `has_quant_attn_kernel` gate disables the native path (never panics across the C-ABI).
fn make_kivi(args: &QuantAttnMakeArgs) -> Box<dyn QuantAttnBackend> {
    Box::new(KiviGpuBackend {
        inner: build_inner(args),
    })
}

// Static (linkme name survival) + dynamic (cdylib C-ABI vtable) registration. KEEP the
// `"kivi_abi"` registry key byte-exact (the `--backend-cap kivi_abi` selector).
argus_extension_api::register_quant_attn_plugin!("kivi_abi", make_kivi);
// One per `.so` — emits the register_kv_formats_v2 / _backend_caps_v2 entries (stage axis is static-linkme only).
argus_extension_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;

    /// A null-context make degrades gracefully (no panic across the abort boundary) and
    /// reports no kernels — the host gate then falls back to the F32 path.
    #[test]
    fn null_context_degrades() {
        let args = QuantAttnMakeArgs {
            cl_ctx: std::ptr::null_mut(),
            device: std::ptr::null_mut(),
            build_opts: std::ptr::null(),
        };
        let be = make_kivi(&args);
        assert!(!be.has_quant_attn_kernel(2));
        assert!(!be.has_quant_attn_kernel(4));
        assert!(!be.has_quant_attn_kernel(8));
        assert!(!be.is_nosub_device());
    }

    /// In the degraded state every GPU op returns the error sentinel (-1) rather than
    /// dispatching (host-runnable: no GPU). Real kernel correctness is proven only on-device.
    #[test]
    fn degraded_ops_return_error_code() {
        let be = KiviGpuBackend { inner: None };
        let null = std::ptr::null_mut();
        let attn = QuantAttnArgs {
            cl_queue: null,
            q_mem: null,
            qk_mem: null,
            qv_mem: null,
            res_k_mem: null,
            res_v_mem: null,
            out_mem: null,
            scores_out: std::ptr::null_mut(),
            scores_len: 0,
            num_heads_q: 1,
            num_heads_kv: 1,
            head_dim: 64,
            q_tokens: 1,
            res_tokens: 0,
            res_cap: 32,
            scale: 0.125,
            bits: 2,
        };
        assert_eq!(be.attention_gen_quant(&attn), -1);
        let gather = QuantAttnGatherArgs {
            cl_queue: null,
            input_mem: null,
            residual_mem: null,
            kv_heads: 1,
            res_cap: 32,
            head_dim: 64,
            seq_len: 1,
            res_pos: 0,
        };
        assert_eq!(be.gather_update_quant(&gather), -1);
        let flush = QuantDequantFlushArgs {
            cl_queue: null,
            q_blocks_mem: null,
            attn_mem: null,
            kv_heads: 1,
            head_dim: 64,
            n_groups_or_tokens: 1,
            tok_base: 0,
            block_start: 0,
            bits: 2,
            is_key: true,
        };
        assert_eq!(be.dequant_flush(&flush), -1);
        let scatter = QuantScatterResidualArgs {
            cl_queue: null,
            res_mem: null,
            attn_mem: null,
            kv_heads: 1,
            res_cap: 32,
            head_dim: 64,
            res_pos: 0,
            tok_base: 0,
        };
        assert_eq!(be.scatter_residual(&scatter), -1);
    }
}
