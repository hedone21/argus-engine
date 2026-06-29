//! GPU half of the observer/score axis — the [`ScoreReduceBackend`] that owns `score_reduce.cl`.
//!
//! The engine's `GpuScoreAccumulator` resolves this reducer by name (`"attn_score"`) at
//! `init_gpu_score_acc`, which compiles the fused reduce kernel from the host's borrowed
//! `cl_context` (the FORMAT Phase 2 Stage E precedent) and dispatches it on the lent
//! `cl_command_queue`. The engine retains ownership of the three score buffers and lends them
//! borrow-for-call. This is the H2O-family score POLICY (per-layer MAX + GQA group averaging + A2SF
//! exponential decay) the engine core no longer holds — the GPU twin of
//! [`AttnScoreProducer`](crate::AttnScoreProducer).
//!
//! Static-linkme only (no cdylib C-ABI): the reducer is the in-tree default GPU scoring path, so a
//! `.so` loader would be a speculative abstraction with no out-of-tree consumer. The args structs
//! are `repr(C)` POD nonetheless, so a future cdylib path needs no ABI re-break.

use std::ffi::{CStr, CString};

use argus_extension_api::{
    SCORE_REDUCERS, ScoreReduceArgs, ScoreReduceBackend, ScoreReduceMakeArgs, ScoreReduceReg,
};
use linkme::distributed_slice;
use ocl::core::{
    self, ArgVal, CommandQueue, Context as ContextCore, DeviceId, Event, Kernel, Mem, Program,
};

/// The fused per-token reduce kernel — moved verbatim from `engine/kernels/score_reduce.cl`.
const SCORE_REDUCE_SRC: &str = include_str!("score_reduce.cl");

/// Round `n` up to a multiple of `multiple` (global-size rounding for the WG=64 launch; mirrors the
/// engine's prior `GpuScoreAccumulator::round_up`).
#[inline]
fn round_up(n: usize, multiple: usize) -> usize {
    n.div_ceil(multiple) * multiple
}

/// Borrow a host-owned `cl_mem` for kernel-arg use WITHOUT taking ownership. `Mem` has a releasing
/// `Drop`, so it is wrapped in `ManuallyDrop` and must never be dropped.
///
/// # Safety
/// `ptr` must be a valid, non-null `cl_mem` of the reducer's context, live for the call.
#[inline]
unsafe fn borrow_mem(ptr: *mut std::ffi::c_void) -> std::mem::ManuallyDrop<Mem> {
    std::mem::ManuallyDrop::new(unsafe { Mem::from_raw_create_ptr(ptr as ocl::ffi::cl_mem) })
}

/// Reconstruct the host's live command queue from the lent raw handle (retain; the returned
/// `CommandQueue`'s `Drop` releases). `None` on null.
#[inline]
fn reconstruct_queue(ptr: *mut std::ffi::c_void) -> Option<CommandQueue> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the host lends its live `cl_command_queue`; from_raw_copied_ptr retains it so this
    // clone is valid for the call and released on drop.
    Some(unsafe { CommandQueue::from_raw_copied_ptr(ptr as ocl::ffi::cl_command_queue) })
}

/// The compiled reduce kernel, reconstructed from the host's borrowed context in `make`.
struct AttnScoreReducer {
    /// Retained copy of the host context, kept alive for the program/kernel. `Drop` releases it
    /// (balances the make-time `clRetainContext`).
    _ctx: ContextCore,
    /// Owns the compiled program (kept alive for `kernel`).
    _program: Program,
    /// `kernel_score_fused_reduce` — re-dispatched once per decode step.
    kernel: Kernel,
    /// `kernel_score_fused_reduce_per_layer` — the faithful-H2O `(b)` per-`(layer, token)` FLAT reduce
    /// (no cross-layer MAX). Dispatched in addition to `kernel` only when per-layer is armed.
    kernel_per_layer: Kernel,
}

// SAFETY: the reducer is only used from the single inference thread, exactly like the engine's
// `GpuScoreAccumulator` (which held the equivalent kernel handle and is itself `unsafe Send+Sync`).
unsafe impl Send for AttnScoreReducer {}
unsafe impl Sync for AttnScoreReducer {}

impl ScoreReduceBackend for AttnScoreReducer {
    fn name(&self) -> &str {
        "attn_score"
    }

    fn reduce(&self, args: &ScoreReduceArgs) -> i32 {
        match self.reduce_inner(args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[attn_score reducer] reduce failed: {e}");
                -1
            }
        }
    }

    fn reduce_per_layer(&self, args: &ScoreReduceArgs) -> i32 {
        match self.reduce_per_layer_inner(args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[attn_score reducer] per-layer reduce failed: {e}");
                -1
            }
        }
    }
}

impl AttnScoreReducer {
    fn reduce_inner(&self, args: &ScoreReduceArgs) -> Result<(), String> {
        let queue = reconstruct_queue(args.cl_queue).ok_or("null cl_queue")?;
        // SAFETY: borrow-only cl_mem (host retains ownership; never dropped).
        let score_buf = unsafe { borrow_mem(args.score_buf) };
        let importance = unsafe { borrow_mem(args.importance) };
        let head_importance = unsafe { borrow_mem(args.head_importance) };

        let decay_factor = args.decay_factor;
        let (n_layers, n_heads_q, n_kv_heads, cache_seq_len, score_stride, max_seq_len) = (
            args.n_layers as i32,
            args.n_heads_q as i32,
            args.n_kv_heads as i32,
            args.cache_seq_len as i32,
            args.score_stride as i32,
            args.max_seq_len as i32,
        );

        // Arg order mirrors `kernel_score_fused_reduce(scores, importance, head_importance,
        // decay_factor, n_layers, n_heads_q, n_kv_heads, cache_seq_len, score_stride, max_seq_len)`.
        for (i, a) in [
            ArgVal::mem(&score_buf),
            ArgVal::mem(&importance),
            ArgVal::mem(&head_importance),
            ArgVal::scalar(&decay_factor),
            ArgVal::scalar(&n_layers),
            ArgVal::scalar(&n_heads_q),
            ArgVal::scalar(&n_kv_heads),
            ArgVal::scalar(&cache_seq_len),
            ArgVal::scalar(&score_stride),
            ArgVal::scalar(&max_seq_len),
        ]
        .into_iter()
        .enumerate()
        {
            core::set_kernel_arg(&self.kernel, i as u32, a).map_err(|e| e.to_string())?;
        }

        let gws = [round_up(args.cache_seq_len, 64), 1, 1];
        let lws = [64usize, 1, 1];
        // SAFETY: validated handles; single-threaded; borrowed mems live until the call returns.
        unsafe {
            core::enqueue_kernel(
                &queue,
                &self.kernel,
                1,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                None::<&mut Event>,
            )
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Faithful-H2O `(b)` per-`(layer, token)` FLAT reduce — dispatches
    /// `kernel_score_fused_reduce_per_layer` over `args.layer_flat_importance` (no cross-layer MAX).
    fn reduce_per_layer_inner(&self, args: &ScoreReduceArgs) -> Result<(), String> {
        if args.layer_flat_importance.is_null() {
            return Err("null layer_flat_importance".to_string());
        }
        let queue = reconstruct_queue(args.cl_queue).ok_or("null cl_queue")?;
        // SAFETY: borrow-only cl_mem (host retains ownership; never dropped).
        let score_buf = unsafe { borrow_mem(args.score_buf) };
        let layer_flat = unsafe { borrow_mem(args.layer_flat_importance) };

        let decay_factor = args.decay_factor;
        let (n_layers, n_heads_q, cache_seq_len, score_stride, max_seq_len) = (
            args.n_layers as i32,
            args.n_heads_q as i32,
            args.cache_seq_len as i32,
            args.score_stride as i32,
            args.max_seq_len as i32,
        );

        // Arg order mirrors `kernel_score_fused_reduce_per_layer(scores, layer_flat, decay_factor,
        // n_layers, n_heads_q, cache_seq_len, score_stride, max_seq_len)`.
        for (i, a) in [
            ArgVal::mem(&score_buf),
            ArgVal::mem(&layer_flat),
            ArgVal::scalar(&decay_factor),
            ArgVal::scalar(&n_layers),
            ArgVal::scalar(&n_heads_q),
            ArgVal::scalar(&cache_seq_len),
            ArgVal::scalar(&score_stride),
            ArgVal::scalar(&max_seq_len),
        ]
        .into_iter()
        .enumerate()
        {
            core::set_kernel_arg(&self.kernel_per_layer, i as u32, a).map_err(|e| e.to_string())?;
        }

        let gws = [round_up(args.cache_seq_len, 64), 1, 1];
        let lws = [64usize, 1, 1];
        // SAFETY: validated handles; single-threaded; borrowed mems live until the call returns.
        unsafe {
            core::enqueue_kernel(
                &queue,
                &self.kernel_per_layer,
                1,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                None::<&mut Event>,
            )
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// make factory — compiles `score_reduce.cl` from the borrowed host context using the host's
/// verbatim build options (Adreno fast-math consistency — never recompute them). Returns an error
/// string on any failure (the engine then falls back to the per-token CPU readback path).
fn make_attn_score_reducer(
    args: &ScoreReduceMakeArgs,
) -> Result<Box<dyn ScoreReduceBackend>, String> {
    // panic=abort guard: from_raw_copied_ptr asserts non-null + unwraps the retain.
    if args.cl_context.is_null() || args.cl_device.is_null() {
        return Err("null cl_context/cl_device".to_string());
    }
    // SAFETY: borrow-for-make; from_raw_copied_ptr clRetainContexts so the clone outlives make,
    // balanced by the ContextCore `Drop` when the reducer is dropped.
    let ctx = unsafe { ContextCore::from_raw_copied_ptr(args.cl_context as ocl::ffi::cl_context) };
    // SAFETY: cl_device_id is not refcounted; from_raw just wraps it. Non-null checked above.
    let device = unsafe { DeviceId::from_raw(args.cl_device as ocl::ffi::cl_device_id) };

    // Host's exact build_cl_opts(device); null → empty.
    let opts = if args.build_opts.is_null() {
        CString::new("").map_err(|e| e.to_string())?
    } else {
        // SAFETY: the host passes a NUL-terminated options string owned for the make call.
        unsafe { CStr::from_ptr(args.build_opts) }.to_owned()
    };

    let src = CString::new(SCORE_REDUCE_SRC).map_err(|e| e.to_string())?;
    let program = core::create_program_with_source(&ctx, &[src]).map_err(|e| e.to_string())?;
    core::build_program(&program, Some(&[device]), &opts, None, None)
        .map_err(|e| format!("score_reduce.cl build failed: {e}"))?;
    let kernel =
        core::create_kernel(&program, "kernel_score_fused_reduce").map_err(|e| e.to_string())?;
    let kernel_per_layer = core::create_kernel(&program, "kernel_score_fused_reduce_per_layer")
        .map_err(|e| e.to_string())?;

    Ok(Box::new(AttnScoreReducer {
        _ctx: ctx,
        _program: program,
        kernel,
        kernel_per_layer,
    }))
}

/// Static registration — the engine resolves `"attn_score"` via `find_score_reducer`. Force-linked
/// through the same `use attn_score as _;` that anchors the CPU producer.
#[distributed_slice(SCORE_REDUCERS)]
static ATTN_SCORE_REDUCER: ScoreReduceReg = ScoreReduceReg {
    name: "attn_score",
    make: make_attn_score_reducer,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// A null-context make degrades gracefully (no panic) and reports an error so the engine falls
    /// back to the CPU readback path.
    #[test]
    fn null_context_make_errors() {
        let args = ScoreReduceMakeArgs {
            cl_context: std::ptr::null_mut(),
            cl_device: std::ptr::null_mut(),
            build_opts: std::ptr::null(),
        };
        assert!(make_attn_score_reducer(&args).is_err());
    }

    /// The reducer is registered under `"attn_score"` (matched to the CPU producer name).
    #[test]
    fn reducer_registered() {
        assert!(argus_extension_api::find_score_reducer("attn_score").is_some());
    }
}
