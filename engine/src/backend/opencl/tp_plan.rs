//! Two-segment CPU–GPU tensor partition inside the decode plan (ticket 021).
//!
//! Per layer, both the attention block and the FFN are split between the GPU (plan kernels over
//! the leading share of each weight, read in place) and the CPU (NEON GEMVs over the rest of the
//! same host-mapped weights). One merge per segment:
//!
//! ```text
//! ATTN  GPU: entry norm (flag A) · Q[0,h_g) K V · RoPE · KV update · attention(h_g heads)
//!            · Wo[:, 0,h_g·hd) → attn_out · done A'
//!       CPU: (after A) Q[h_g,n) K V · RoPE · host KV append · attention(n−h_g heads)
//!            · Wo[:, h_g·hd,n·hd) → staging_attn
//! FFN   GPU: attn_out += staging_attn · entry norm (flag F) · gate/up[0,s) · silu
//!            · down[:, 0,s) → down · done F'
//!       CPU: (after F) gate/up[s,F) · silu · down[:, s,F) → staging_ffn
//!       GPU: x += down + staging_ffn
//! ```
//!
//! The CPU keeps its own host KV cache (all KV heads, recomputed K/V), so it never reads the GPU
//! cache — except once, to catch up with what prefill wrote.
//!
//! GPU → CPU visibility: a flag the GPU raises is seen by the host before the GPU's other stores
//! to host-mapped memory are (measured on Adreno 830: the in-kernel `_sigflag` release, and a
//! separate flag kernel right after, both let the CPU read a stale `residual` within a few
//! tokens). What orders them is a submission boundary: the entry norm is flushed on its own and
//! the input flag is raised by a kernel of the next submission.
//!
//! Observation (§D3): the CPU looks at the segment's done-flag when its own share is finished.
//! Still down → the GPU time is measured by waiting for it (`waited`); already up → only
//! `t_gpu <= t_cpu` is known. The wait is not a pipeline stall: the next commands are enqueued
//! first, and the flag is polled while the host spins on the next input flag.

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use ocl::core::{ArgVal, Kernel as CoreKernel, Mem};

use super::plan::{
    AttentionVariant, FullKernelPlan, GemvKind, KernelStep, LayerKernelPlan, LayerPlanConfig,
    OpTag, PlanInvalidated, Q1_MERGE_KV_START_ARG, Q1_SPLIT_MAIN_KV_START_ARG, QRowPlanCopy,
    check_partition_generation,
};
use crate::backend::Backend;
use crate::layers::tensor_partition::{
    PartitionContext, attention_head_range, cpu_share_indices, gemv_f16_strided,
    partition_plan_enabled,
};
use crate::layers::tp_controller::{Obs, SegGeom, TpController};
use crate::partition_workspace::{HostKv, PartitionWorkspace, PartitionWsCell, PendingObs};
use crate::tensor::Tensor;

/// Weights of one layer the CPU share reads through host pointers. Holding the tensors keeps
/// the mappings alive for the plan's lifetime.
pub struct TpHostWeights {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
    pub bq: Option<Tensor>,
    pub bk: Option<Tensor>,
    pub bv: Option<Tensor>,
}

impl TpHostWeights {
    fn check(&self) -> Result<()> {
        use crate::buffer::DType;
        for (name, t) in [
            ("wq", &self.wq),
            ("wk", &self.wk),
            ("wv", &self.wv),
            ("wo", &self.wo),
            ("w_gate", &self.w_gate),
            ("w_up", &self.w_up),
            ("w_down", &self.w_down),
        ] {
            ensure!(
                t.dtype() == DType::F16,
                "tensor partition needs F16 weights ({name} is {:?})",
                t.dtype()
            );
            ensure!(!t.as_ptr().is_null(), "{name} is not host-mapped");
        }
        for b in [&self.bq, &self.bk, &self.bv].into_iter().flatten() {
            ensure!(
                b.dtype() == DType::F32 && !b.as_ptr().is_null(),
                "QKV bias must be host-mapped F32"
            );
        }
        Ok(())
    }
}

/// Spin cap for a flag wait: seconds on a 3 GHz core. A flag that never rises means a lost
/// kernel; the plan then reports `PlanInvalidated` and the partition arm stops with FATAL.
const MAX_SPINS: u64 = 50_000_000;

#[inline]
fn flag_ptr(t: &Tensor) -> *mut i32 {
    t.buffer().as_mut_ptr() as *mut i32
}

#[inline]
fn flag_up(p: *mut i32) -> bool {
    // SAFETY: `p` is a host-mapped 4-byte flag owned by the partition workspace.
    unsafe { std::ptr::read_volatile(p) != 0 }
}

#[inline]
fn flag_reset(p: *mut i32) {
    // SAFETY: as above; the GPU writes this flag again only in a later command.
    unsafe { std::ptr::write_volatile(p, 0) }
}

/// Resolve every pending observation whose done-flag has risen.
fn poll_pending(rt: &mut crate::partition_workspace::TpRuntime) {
    if rt.pending.is_empty() {
        return;
    }
    let now = Instant::now();
    let TpRuntimeParts { pending, ctl } = split_rt(rt);
    pending.retain(|p| {
        if !flag_up(p.flag) {
            return true;
        }
        flag_reset(p.flag);
        std::sync::atomic::fence(Ordering::Acquire);
        if let Some(ctl) = ctl.as_mut() {
            let t_gpu = now.duration_since(p.t0).as_secs_f32() * 1e3;
            ctl.observe(
                p.layer,
                p.seg,
                Obs {
                    waited: true,
                    t_cpu: p.t_cpu,
                    t_gpu: Some(t_gpu),
                },
            );
        }
        false
    });
}

struct TpRuntimeParts<'a> {
    pending: &'a mut Vec<PendingObs>,
    ctl: &'a mut Option<TpController>,
}

fn split_rt(rt: &mut crate::partition_workspace::TpRuntime) -> TpRuntimeParts<'_> {
    TpRuntimeParts {
        pending: &mut rt.pending,
        ctl: &mut rt.ctl,
    }
}

/// Spin until `flag` rises (then reset it), resolving pending observations meanwhile. Returns
/// the instant the flag was seen.
fn spin_flag(
    flag: *mut i32,
    rt: &mut crate::partition_workspace::TpRuntime,
) -> std::result::Result<Instant, PlanInvalidated> {
    let mut spins = 0u64;
    loop {
        if flag_up(flag) {
            let t = Instant::now();
            std::sync::atomic::fence(Ordering::Acquire);
            flag_reset(flag);
            poll_pending(rt);
            return Ok(t);
        }
        poll_pending(rt);
        std::hint::spin_loop();
        spins += 1;
        if spins == MAX_SPINS {
            log::error!("tensor partition: flag wait timed out");
            return Err(PlanInvalidated);
        }
    }
}

fn flush(backend: &super::OpenCLBackend) -> std::result::Result<(), PlanInvalidated> {
    ocl::core::flush(backend.queue.as_core()).map_err(|e| {
        log::error!("tensor partition: flush failed: {e}");
        PlanInvalidated
    })
}

unsafe fn set_i32(k: &CoreKernel, idx: u32, v: i32) {
    if let Err(e) = unsafe { ocl::core::set_kernel_arg(k, idx, ArgVal::scalar(&v)) } {
        log::error!("tensor partition: set arg {idx} failed: {e}");
    }
}

/// A kernel with fixed args from `simple_ops`.
fn simple_step(
    program: &ocl::Program,
    name: &str,
    args: &[ArgVal],
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    op_tag: OpTag,
) -> Result<KernelStep> {
    let kernel =
        ocl::core::create_kernel(program, name).with_context(|| format!("create {name}"))?;
    for (i, a) in args.iter().enumerate() {
        unsafe { ocl::core::set_kernel_arg(&kernel, i as u32, a.clone())? };
    }
    Ok(KernelStep {
        kernel,
        ndim: 1,
        global_work_size: gws,
        local_work_size: lws,
        dynamic_args: vec![],
        op_tag,
        retained_bufs: vec![],
        noshuffle_act_rebuild: None,
    })
}

/// `x += residual_in; out = rmsnorm(x)·w` — a segment's entry. Its input flag is raised by a
/// separate `kernel_signal_flag` in the next submission (see the module doc).
fn entry_norm_step(
    config: &LayerPlanConfig,
    residual_in: &Mem,
    norm_w: &Mem,
) -> Result<KernelStep> {
    let local = 64usize;
    let dim = config.dim as i32;
    simple_step(
        config.simple_ops_program,
        "kernel_add_rms_norm_oop_f4",
        &[
            ArgVal::mem(config.x_buf),
            ArgVal::mem(residual_in),
            ArgVal::mem(config.residual_buf),
            ArgVal::mem(norm_w),
            ArgVal::scalar(&dim),
            ArgVal::scalar(&config.rms_norm_eps),
            ArgVal::scalar(&0i32),
            ArgVal::local::<f32>(&(local * 4)),
        ],
        [local, 1, 1],
        Some([local, 1, 1]),
        OpTag::AddRmsNorm,
    )
}

fn signal_step(config: &LayerPlanConfig, flag: &Mem) -> Result<KernelStep> {
    simple_step(
        config.simple_ops_program,
        "kernel_signal_flag",
        &[ArgVal::mem(flag)],
        [1, 1, 1],
        Some([1, 1, 1]),
        OpTag::AddAssign,
    )
}

/// Row-strided F16 GEMV (`kernel_mul_mat_f16_f32_ld`): `dst[j] = Σ_{i<k} w[j·ld + i]·src[i]`,
/// `j < n`. Arg 6/9 = `k` (patched per token), arg 15 = `ld`.
#[allow(clippy::too_many_arguments)]
fn gemv_ld_step(
    config: &LayerPlanConfig,
    src: &Mem,
    weight: &Mem,
    dst: &Mem,
    n: usize,
    k: usize,
    ld: usize,
    op_tag: OpTag,
) -> Result<(KernelStep, GemvKind)> {
    let kind = GemvKind::select(n, false, config.is_nosub);
    let kernel = ocl::core::create_kernel(config.f16_program, "kernel_mul_mat_f16_f32_ld")
        .context("create kernel_mul_mat_f16_f32_ld")?;
    let (k_i, n_i, ld_i) = (k as i32, n as i32, ld as i32);
    unsafe {
        use ocl::core::set_kernel_arg as set;
        set(&kernel, 0, ArgVal::mem(weight))?;
        set(&kernel, 1, ArgVal::scalar(&0u64))?;
        set(&kernel, 2, ArgVal::mem(src))?;
        set(&kernel, 3, ArgVal::scalar(&0u64))?;
        set(&kernel, 4, ArgVal::mem(dst))?;
        set(&kernel, 5, ArgVal::scalar(&0u64))?;
        set(&kernel, 6, ArgVal::scalar(&k_i))?;
        set(&kernel, 7, ArgVal::scalar(&n_i))?;
        set(&kernel, 8, ArgVal::scalar(&1i32))?;
        set(&kernel, 9, ArgVal::scalar(&k_i))?;
        set(&kernel, 10, ArgVal::scalar(&1i32))?;
        set(&kernel, 11, ArgVal::scalar(&n_i))?;
        set(&kernel, 12, ArgVal::scalar(&1i32))?;
        set(&kernel, 13, ArgVal::scalar(&1i32))?;
        set(&kernel, 14, ArgVal::scalar(&1i32))?;
        set(&kernel, 15, ArgVal::scalar(&ld_i))?;
    }
    let (gws, lws) = kind.work_size(n);
    Ok((
        KernelStep {
            kernel,
            ndim: 3,
            global_work_size: gws,
            local_work_size: Some(lws),
            dynamic_args: vec![],
            op_tag,
            retained_bufs: vec![],
            noshuffle_act_rebuild: None,
        },
        kind,
    ))
}

/// RoPE frequencies of `kernel_rope_simple`: `1/θ^(2i/hd)`, then the llama3 scaling.
fn rope_freqs(config: &LayerPlanConfig) -> Vec<f32> {
    let hd = config.head_dim;
    (0..hd / 2)
        .map(|i| {
            let f = 1.0 / config.rope_theta.powf((2 * i) as f32 / hd as f32);
            config.rope_freq_scaling.scale_freq(f)
        })
        .collect()
}

/// NeoX-style RoPE (pairs `(i, i + hd/2)`) of `heads` rows of `x` at position `pos`.
fn rope_rows(x: &mut [f32], heads: std::ops::Range<usize>, hd: usize, pos: usize, freqs: &[f32]) {
    let half = hd / 2;
    let sc: Vec<(f32, f32)> = freqs.iter().map(|f| (pos as f32 * f).sin_cos()).collect();
    for h in heads {
        let row = &mut x[h * hd..(h + 1) * hd];
        for (i, &(sin, cos)) in sc.iter().enumerate() {
            let (v0, v1) = (row[i], row[i + half]);
            row[i] = v0 * cos - v1 * sin;
            row[i + half] = v0 * sin + v1 * cos;
        }
    }
}

#[inline]
fn f32_mut(t: &Tensor) -> *mut f32 {
    t.buffer().as_mut_ptr() as *mut f32
}

#[inline]
fn f32_ptr(t: &Tensor) -> *const f32 {
    t.buffer().as_ptr() as *const f32
}

#[inline]
fn u16_ptr(t: &Tensor) -> *const u16 {
    t.buffer().as_ptr() as *const u16
}

/// Copy the CPU partial into a host-mapped staging buffer the GPU reads next.
fn deliver(src: &Tensor, staging: &Tensor, n: usize) {
    // SAFETY: both are `n`-float host buffers (staging is permanently mapped ALLOC_HOST_PTR).
    unsafe {
        std::ptr::copy_nonoverlapping(f32_ptr(src), f32_mut(staging), n);
    }
    // Commit the stores before the kernel that reads them is enqueued.
    std::sync::atomic::fence(Ordering::Release);
}

/// Shared plumbing of the two segment steps.
struct SegCommon {
    layer: usize,
    ws: Arc<PartitionWsCell>,
    kernels: Option<&'static crate::cpu_kernels::CpuKernelSet>,
    /// Host view of the permanently mapped `ws.residual` (each segment's normed input).
    residual_host: *const f32,
    ratio_generation_at_build: u64,
    ratio_generation: Arc<AtomicU64>,
    build_thread: std::thread::ThreadId,
}

impl SegCommon {
    #[allow(clippy::mut_from_ref)]
    fn ws(&self) -> &mut PartitionWorkspace {
        debug_assert_eq!(std::thread::current().id(), self.build_thread);
        // SAFETY: single-threaded plan dispatch (see `PartitionWsCell`).
        unsafe { &mut *self.ws.get() }
    }

    fn check(&self) -> std::result::Result<(), PlanInvalidated> {
        check_partition_generation(self.ratio_generation_at_build, &self.ratio_generation)
    }

    /// Quantum index this segment runs at this token.
    fn applied(&self, seg: usize) -> usize {
        self.ws()
            .tp
            .ctl
            .as_ref()
            .expect("controller created before the partitioned plan")
            .applied(self.layer, seg)
    }

    /// Host pointers of this layer's `[A, A', F, F']` flags.
    fn flags(&self) -> [*mut i32; 4] {
        let f = &self.ws().flags[self.layer];
        [
            flag_ptr(&f[0]),
            flag_ptr(&f[1]),
            flag_ptr(&f[2]),
            flag_ptr(&f[3]),
        ]
    }

    /// Wait for a flag, resolving pending observations meanwhile.
    fn spin(&self, flag: *mut i32) -> std::result::Result<Instant, PlanInvalidated> {
        spin_flag(flag, &mut self.ws().tp)
    }

    /// Copy the segment input (`ws.residual`, final once the input flag rose) for the CPU share.
    fn load_input(&self, dim: usize) {
        // SAFETY: `residual_host` maps the `dim`-float residual buffer; `residual_cpu` is `dim`.
        unsafe {
            std::ptr::copy_nonoverlapping(self.residual_host, f32_mut(&self.ws().residual_cpu), dim)
        };
    }

    fn serial(&self, seg: usize) -> bool {
        self.ws()
            .tp
            .ctl
            .as_ref()
            .is_some_and(|c| c.serial(self.layer, seg))
    }

    /// After the CPU share: record what the done-flag says (§D3 observation model).
    fn observe_after_cpu(
        &self,
        seg: usize,
        done: *mut i32,
        t0: Instant,
        t_cpu_end: Instant,
        serial_t_gpu: Option<f32>,
    ) {
        let rt = &mut self.ws().tp;
        let t_cpu = t_cpu_end.duration_since(t0).as_secs_f32() * 1e3;
        if let Some(t_gpu) = serial_t_gpu {
            if let Some(ctl) = rt.ctl.as_mut() {
                ctl.observe(
                    self.layer,
                    seg,
                    Obs {
                        waited: true,
                        t_cpu,
                        t_gpu: Some(t_gpu),
                    },
                );
            }
            return;
        }
        if flag_up(done) {
            flag_reset(done);
            if let Some(ctl) = rt.ctl.as_mut() {
                ctl.observe(
                    self.layer,
                    seg,
                    Obs {
                        waited: false,
                        t_cpu,
                        t_gpu: None,
                    },
                );
            }
        } else {
            rt.pending.push(PendingObs {
                layer: self.layer,
                seg,
                flag: done,
                t0,
                t_cpu,
            });
        }
    }
}

// SAFETY: raw kernel handles + the workspace cell, all used only from the dispatch thread
// (`SegCommon::ws` debug-asserts it) — the `KernelStep` safety model.
unsafe impl Send for AttnPartitionStep {}
unsafe impl Sync for AttnPartitionStep {}
unsafe impl Send for FfnPartitionStep {}
unsafe impl Sync for FfnPartitionStep {}

/// ATTN segment of one partitioned layer.
pub struct AttnPartitionStep {
    c: SegCommon,
    entry: KernelStep,
    /// Raises flag A (input ready) — first command after the entry norm's submission.
    entry_flag: KernelStep,
    q: KernelStep,
    q_kind: GemvKind,
    q_bias: Option<KernelStep>,
    /// K, (bias K), V, (bias V) — full KV heads, fixed.
    kv_proj: Vec<KernelStep>,
    rope_q: KernelStep,
    rope_k: KernelStep,
    kv_update: KernelStep,
    attention: AttentionVariant,
    wo: KernelStep,
    done: Option<KernelStep>,
    /// Q heads currently bound on the GPU.
    bound: Cell<usize>,
    host: Arc<TpHostWeights>,
    n_heads_q: usize,
    n_kv: usize,
    head_dim: usize,
    dim: usize,
    freqs: Vec<f32>,
    k_cache: Mem,
    v_cache: Mem,
    kv_capacity: usize,
    cpu_backend: Arc<dyn Backend>,
    /// Set when the plan writes GPU scores: the CPU heads' rows go here too (ticket 023 E2).
    scores: Option<ScoreRows>,
}

/// This layer's slice of the GPU score buffer (`[n_heads_q][stride]` f32 at `layer_offset`).
struct ScoreRows {
    buf: Mem,
    layer_offset: usize,
    stride: usize,
}

/// Per-token inputs of [`AttnPartitionStep::run`] beyond the cache positions.
pub struct AttnRunCtx<'a> {
    /// Device mirror of the ragged cache's per-KV-head first slots (`None` = uniform).
    pub head_start: Option<&'a Mem>,
    /// Host copy of the same starts (empty when `head_start` is `None`).
    pub head_starts: &'a [usize],
    /// The query-row ring copy, when armed.
    pub q_rows: Option<&'a QRowPlanCopy>,
}

static RAGGED_BOUND_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// FFN segment of one partitioned layer.
pub struct FfnPartitionStep {
    c: SegCommon,
    merge_attn: KernelStep,
    entry: KernelStep,
    /// Raises flag F (input ready).
    entry_flag: KernelStep,
    gate: KernelStep,
    up: KernelStep,
    kind: GemvKind,
    silu: KernelStep,
    down: KernelStep,
    done: Option<KernelStep>,
    /// `x += down + staging_ffn`.
    pub merge: super::plan::PartitionMerge,
    /// gate/up rows currently bound on the GPU.
    bound: Cell<usize>,
    host: Arc<TpHostWeights>,
    dim: usize,
    ffn_hidden: usize,
}

impl AttnPartitionStep {
    /// Bind `h_g` GPU heads into the kernels whose shape depends on it.
    fn bind(&self, h_g: usize) {
        if self.bound.get() == h_g {
            return;
        }
        let n = (h_g * self.head_dim) as i32;
        unsafe {
            set_i32(&self.q.kernel, 7, n);
            set_i32(&self.q.kernel, 11, n);
            if let Some(b) = &self.q_bias {
                set_i32(&b.kernel, 2, n);
                set_i32(&b.kernel, 3, n);
            }
            set_i32(&self.rope_q.kernel, 2, h_g as i32);
            set_i32(&self.wo.kernel, 6, n);
            set_i32(&self.wo.kernel, 9, n);
        }
        self.bound.set(h_g);
    }

    /// Bring the host KV cache up to `pos` from the GPU cache (after prefill / a rebuild).
    fn catch_up(
        &self,
        backend: &super::OpenCLBackend,
        kv: &mut HostKv,
        pos: usize,
    ) -> std::result::Result<(), PlanInvalidated> {
        let lo = if kv.len > pos { 0 } else { kv.len };
        if lo == pos {
            kv.len = pos;
            return Ok(());
        }
        let row = self.head_dim * 2;
        let queue = backend.queue.as_core();
        for (mem, host) in [(&self.k_cache, &kv.k), (&self.v_cache, &kv.v)] {
            for h in 0..self.n_kv {
                let off = (h * self.kv_capacity + lo) * row;
                let len = (pos - lo) * row;
                // SAFETY: the host cache has the GPU cache's geometry.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(host.buffer().as_mut_ptr().add(off), len)
                };
                if let Err(e) = unsafe {
                    ocl::core::enqueue_read_buffer(
                        queue,
                        mem,
                        true,
                        off,
                        dst,
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )
                } {
                    log::error!("tensor partition: KV catch-up read failed: {e}");
                    return Err(PlanInvalidated);
                }
            }
        }
        kv.len = pos;
        Ok(())
    }

    /// CPU share of the attention block for heads `[h_g, n_heads_q)` for the token at RoPE
    /// position `start_pos`, written at cache slot `write_pos`. `kv_start` marks a ragged cache;
    /// `scores` receives the heads' post-softmax rows (`n_heads_q` rows of the attention length).
    fn cpu_share(
        &self,
        h_g: usize,
        start_pos: usize,
        write_pos: usize,
        kv_start: Option<&[usize]>,
        scores: Option<&mut [f32]>,
    ) -> Result<()> {
        let (rope_pos, slot, attn_len, new_len) = cpu_share_indices(start_pos, write_pos);
        ensure!(
            slot < self.kv_capacity,
            "host KV slot {slot} outside capacity {}",
            self.kv_capacity
        );
        let ws = self.c.ws();
        let hd = self.head_dim;
        let (nq, nkv, dim) = (self.n_heads_q, self.n_kv, self.dim);
        let q_lo = h_g * hd;
        let q_rows = (nq - h_g) * hd;
        let kv_rows = nkv * hd;
        let h = &self.host;
        // SAFETY: host-mapped weights of this layer (kept alive by `host`), workspace buffers
        // sized for the full model geometry.
        unsafe {
            gemv_f16_strided(
                self.c.kernels,
                f32_ptr(&ws.residual_cpu),
                dim,
                &[
                    (
                        u16_ptr(&h.wq).add(q_lo * dim),
                        dim,
                        f32_mut(&ws.q_cpu).add(q_lo),
                        q_rows,
                    ),
                    (u16_ptr(&h.wk), dim, f32_mut(&ws.k_cpu), kv_rows),
                    (u16_ptr(&h.wv), dim, f32_mut(&ws.v_cpu), kv_rows),
                ],
            );
            let q = std::slice::from_raw_parts_mut(f32_mut(&ws.q_cpu), nq * hd);
            let k = std::slice::from_raw_parts_mut(f32_mut(&ws.k_cpu), kv_rows);
            let v = std::slice::from_raw_parts_mut(f32_mut(&ws.v_cpu), kv_rows);
            let add_bias = |x: &mut [f32], b: &Option<Tensor>, off: usize| {
                if let Some(b) = b {
                    let b = std::slice::from_raw_parts(f32_ptr(b).add(off), x.len());
                    x.iter_mut().zip(b).for_each(|(x, b)| *x += b);
                }
            };
            add_bias(&mut q[q_lo..], &h.bq, q_lo);
            add_bias(k, &h.bk, 0);
            add_bias(v, &h.bv, 0);
            rope_rows(q, h_g..nq, hd, rope_pos, &self.freqs);
            rope_rows(k, 0..nkv, hd, rope_pos, &self.freqs);

            let kv = &mut ws.tp.kv[self.c.layer];
            let kc = kv.k.buffer().as_mut_ptr() as *mut half::f16;
            let vc = kv.v.buffer().as_mut_ptr() as *mut half::f16;
            for head in 0..nkv {
                let base = (head * self.kv_capacity + slot) * hd;
                for d in 0..hd {
                    *kc.add(base + d) = half::f16::from_f32(k[head * hd + d]);
                    *vc.add(base + d) = half::f16::from_f32(v[head * hd + d]);
                }
            }
            kv.len = new_len;
        }
        let kv = &ws.tp.kv[self.c.layer];
        attention_head_range(
            self.cpu_backend.as_ref(),
            &ws.q_cpu,
            &kv.k,
            &kv.v,
            &mut ws.attn_out_cpu,
            h_g..nq,
            nq,
            nkv,
            hd,
            attn_len,
            kv_start,
            scores,
        )?;
        unsafe {
            gemv_f16_strided(
                self.c.kernels,
                f32_ptr(&ws.attn_out_cpu).add(q_lo),
                q_rows,
                &[(
                    u16_ptr(&h.wo).add(q_lo),
                    nq * hd,
                    f32_mut(&ws.wo_partial_cpu),
                    dim,
                )],
            );
        }
        Ok(())
    }

    /// Run the ATTN segment of layer `self.c.layer`.
    pub fn run(
        &self,
        backend: &super::OpenCLBackend,
        start_pos: usize,
        write_pos: usize,
        kv_cap: i32,
        rc: &AttnRunCtx,
    ) -> std::result::Result<(), PlanInvalidated> {
        self.c.check()?;
        let layer = self.c.layer;
        {
            let rt = &mut self.c.ws().tp;
            rt.catch_up.pos = start_pos;
            let kv = &mut rt.kv[layer];
            if kv.len != write_pos {
                let full = kv.len > write_pos || kv.len == 0;
                self.catch_up(backend, kv, write_pos)?;
                let n = &mut rt.catch_up;
                if full {
                    n.full += 1;
                    n.full_this_token += 1;
                } else {
                    n.tail += 1;
                }
            }
        }
        let h_g = SegGeom::attn(self.n_heads_q).split(self.c.applied(0));
        self.bind(h_g);
        let serial = self.c.serial(0);

        let (sp, cs, wp) = (start_pos as i32, write_pos as i32, write_pos as i32);
        let d = |step: &KernelStep, gws: &[usize; 3], seq: i32| {
            FullKernelPlan::dispatch_step_gws(backend, step, gws, sp, seq, wp, kv_cap, 0, 0, 0);
        };
        d(&self.entry, &self.entry.global_work_size, cs);
        flush(backend)?;
        d(&self.entry_flag, &self.entry_flag.global_work_size, cs);
        let (q_gws, _) = self.q_kind.work_size(h_g * self.head_dim);
        d(&self.q, &q_gws, cs);
        if let Some(b) = &self.q_bias {
            d(b, &[(h_g * self.head_dim).div_ceil(64) * 64, 1, 1], cs);
        }
        for s in &self.kv_proj {
            d(s, &s.global_work_size, cs);
        }
        d(&self.rope_q, &[h_g * self.head_dim / 2, 1, 1], cs);
        d(&self.rope_k, &self.rope_k.global_work_size, cs);
        // The GPU heads' rotated query rows: `ws.q` holds only `[0, h_g)` valid (ticket 023 E3).
        let q_ring_off = rc
            .q_rows
            .filter(|c| layer < c.n_layers)
            .map(|c| (c, (layer * c.rows + start_pos % c.rows) * c.row_bytes));
        if let Some((c, off)) = q_ring_off
            && let Err(e) = unsafe {
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    backend.queue.as_core(),
                    &c.src,
                    &c.ring,
                    0,
                    off,
                    h_g * self.head_dim * 4,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )
            }
        {
            log::error!("tensor partition: q-row copy failed: layer={layer}: {e}");
            c.failed.store(true, Ordering::Relaxed);
        }
        d(&self.kv_update, &self.kv_update.global_work_size, cs);
        let attn_seq = cs + 1;
        // A ragged cache: the GPU heads read from their own first slot (ticket 023 E4).
        if let Some(m) = rc.head_start {
            let bind = |k: &CoreKernel, idx: u32| {
                if let Err(e) = unsafe { ocl::core::set_kernel_arg(k, idx, ArgVal::mem(m)) } {
                    log::error!("tensor partition: set kv_start arg {idx} failed: {e}");
                }
            };
            match &self.attention {
                AttentionVariant::StandardFlash(s) => bind(&s.kernel, 44),
                AttentionVariant::SplitFlash { main, merge } => {
                    bind(&main.kernel, Q1_SPLIT_MAIN_KV_START_ARG);
                    bind(&merge.kernel, Q1_MERGE_KV_START_ARG);
                }
                AttentionVariant::Standard(_) => unreachable!("rejected at build"),
            }
            if !RAGGED_BOUND_LOGGED.swap(true, Ordering::Relaxed) {
                eprintln!("[tp] ragged kv_start bound");
            }
        }
        match &self.attention {
            AttentionVariant::StandardFlash(s) => d(s, &[s.global_work_size[0], h_g, 1], attn_seq),
            AttentionVariant::SplitFlash { main, merge } => {
                let g = main.global_work_size;
                d(main, &[g[0], h_g, g[2]], attn_seq);
                if g[2] > 1 {
                    d(merge, &[merge.global_work_size[0], h_g, 1], attn_seq);
                }
            }
            AttentionVariant::Standard(_) => unreachable!("rejected at build"),
        }
        d(&self.wo, &self.wo.global_work_size, cs);
        if let Some(done) = &self.done {
            d(done, &done.global_work_size, cs);
        }
        flush(backend)?;

        let flags = self.c.flags();
        let t0 = self.c.spin(flags[0])?;
        self.c.load_input(self.dim);
        let serial_t_gpu = match (&self.done, serial) {
            (Some(_), true) => Some(self.c.spin(flags[1])?.duration_since(t0).as_secs_f32() * 1e3),
            _ => None,
        };
        let t_cpu0 = if serial { Instant::now() } else { t0 };
        let ragged = rc.head_starts.iter().any(|&s| s > 0);
        let kv_start = ragged.then_some(rc.head_starts);
        let (_, _, attn_len, _) = cpu_share_indices(start_pos, write_pos);
        let nq = self.n_heads_q;
        let mut score_stage = self.scores.as_ref().map(|_| {
            let stage = &mut self.c.ws().tp.score_stage[layer];
            stage.resize(nq * attn_len, 0.0);
            stage
        });
        let res = self.cpu_share(
            h_g,
            start_pos,
            write_pos,
            kv_start,
            score_stage.as_deref_mut().map(|v| v.as_mut_slice()),
        );
        if let Err(e) = res {
            log::error!("tensor partition: CPU attention failed: layer={layer} err={e:#}");
            return Err(PlanInvalidated);
        }
        let t_cpu_end = Instant::now();
        // Hand the CPU heads' score rows and query rows to the device. Non-blocking: each layer
        // has its own staging, and the token's blocking logits read ends before a layer's staging
        // is written again. Enqueued before `end_step`'s reduce, which reads the score rows.
        let queue = backend.queue.as_core();
        if let (Some(sr), Some(stage)) = (&self.scores, score_stage) {
            for h in h_g..nq {
                if let Err(e) = unsafe {
                    ocl::core::enqueue_write_buffer(
                        queue,
                        &sr.buf,
                        false,
                        sr.layer_offset + h * sr.stride,
                        &stage[h * attn_len..(h + 1) * attn_len],
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )
                } {
                    log::error!("tensor partition: score row write failed: layer={layer}: {e}");
                    return Err(PlanInvalidated);
                }
            }
        }
        if let Some((c, off)) = q_ring_off {
            let ws = self.c.ws();
            let q_lo = h_g * self.head_dim;
            let q_all = nq * self.head_dim;
            let stage = &mut ws.tp.q_stage[layer];
            stage.resize(q_all, 0.0);
            // SAFETY: `q_cpu` holds `n_heads_q * head_dim` floats.
            let q = unsafe { std::slice::from_raw_parts(f32_ptr(&ws.q_cpu), q_all) };
            stage[q_lo..].copy_from_slice(&q[q_lo..]);
            if let Err(e) = unsafe {
                ocl::core::enqueue_write_buffer(
                    queue,
                    &c.ring,
                    false,
                    off / 4 + q_lo,
                    &stage[q_lo..],
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )
            } {
                log::error!("tensor partition: q-row write failed: layer={layer}: {e}");
                c.failed.store(true, Ordering::Relaxed);
            }
        }
        {
            let ws = self.c.ws();
            deliver(&ws.wo_partial_cpu, &ws.staging_attn, self.dim);
        }
        if self.done.is_some() {
            self.c
                .observe_after_cpu(0, flags[1], t_cpu0, t_cpu_end, serial_t_gpu);
        }
        Ok(())
    }
}

impl FfnPartitionStep {
    fn bind(&self, s: usize) {
        if self.bound.get() == s {
            return;
        }
        let n = s as i32;
        unsafe {
            for k in [&self.gate.kernel, &self.up.kernel] {
                set_i32(k, 7, n);
                set_i32(k, 11, n);
            }
            set_i32(&self.silu.kernel, 2, n / 4);
            set_i32(&self.down.kernel, 6, n);
            set_i32(&self.down.kernel, 9, n);
        }
        self.bound.set(s);
    }

    fn cpu_share(&self, s: usize) {
        let ws = self.c.ws();
        let (dim, f) = (self.dim, self.ffn_hidden);
        let rows = f - s;
        let h = &self.host;
        // SAFETY: host-mapped weights (kept alive by `host`), workspace sized for the model.
        unsafe {
            gemv_f16_strided(
                self.c.kernels,
                f32_ptr(&ws.residual_cpu),
                dim,
                &[
                    (
                        u16_ptr(&h.w_gate).add(s * dim),
                        dim,
                        f32_mut(&ws.gate_cpu),
                        rows,
                    ),
                    (
                        u16_ptr(&h.w_up).add(s * dim),
                        dim,
                        f32_mut(&ws.up_cpu),
                        rows,
                    ),
                ],
            );
            let g = std::slice::from_raw_parts_mut(f32_mut(&ws.gate_cpu), rows);
            let u = std::slice::from_raw_parts(f32_ptr(&ws.up_cpu), rows);
            for (g, &u) in g.iter_mut().zip(u) {
                *g = *g / (1.0 + (-*g).exp()) * u;
            }
            gemv_f16_strided(
                self.c.kernels,
                f32_ptr(&ws.gate_cpu),
                rows,
                &[(
                    u16_ptr(&h.w_down).add(s),
                    f,
                    f32_mut(&ws.down_partial_cpu),
                    dim,
                )],
            );
        }
    }

    /// Run the FFN segment (and the ATTN merge in front of it).
    pub fn run(&self, backend: &super::OpenCLBackend) -> std::result::Result<(), PlanInvalidated> {
        self.c.check()?;
        let s = SegGeom::ffn(self.ffn_hidden).split(self.c.applied(1));
        self.bind(s);
        let serial = self.c.serial(1);

        let d = |step: &KernelStep, gws: &[usize; 3]| {
            FullKernelPlan::dispatch_step_gws(backend, step, gws, 0, 0, 0, 0, 0, 0, 0);
            // Catch the previous segment's done-flag as early as possible.
            poll_pending(&mut self.c.ws().tp);
        };
        d(&self.merge_attn, &self.merge_attn.global_work_size);
        d(&self.entry, &self.entry.global_work_size);
        flush(backend)?;
        d(&self.entry_flag, &self.entry_flag.global_work_size);
        let (gws, _) = self.kind.work_size(s);
        d(&self.gate, &gws);
        d(&self.up, &gws);
        d(&self.silu, &[s / 4, 1, 1]);
        d(&self.down, &self.down.global_work_size);
        if let Some(done) = &self.done {
            d(done, &done.global_work_size);
        }
        flush(backend)?;

        let flags = self.c.flags();
        let t0 = self.c.spin(flags[2])?;
        self.c.load_input(self.dim);
        let serial_t_gpu = match (&self.done, serial) {
            (Some(_), true) => Some(self.c.spin(flags[3])?.duration_since(t0).as_secs_f32() * 1e3),
            _ => None,
        };
        let t_cpu0 = if serial { Instant::now() } else { t0 };
        self.cpu_share(s);
        let t_cpu_end = Instant::now();
        {
            let ws = self.c.ws();
            deliver(&ws.down_partial_cpu, &ws.staging_ffn, self.dim);
        }
        if self.done.is_some() {
            self.c
                .observe_after_cpu(1, flags[3], t_cpu0, t_cpu_end, serial_t_gpu);
        }
        let super::plan::PartitionMerge::Fused { fused_step } = &self.merge;
        FullKernelPlan::dispatch_step_gws(
            backend,
            fused_step,
            &fused_step.global_work_size,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        Ok(())
    }
}

/// Close a token: wait out the pending done-flags, let the controller step, apply a thread
/// change, publish the per-token stats.
pub(crate) fn end_token(ws: &PartitionWsCell) -> std::result::Result<(), PlanInvalidated> {
    // SAFETY: dispatch thread, after the last layer ran.
    let pw = unsafe { &mut *ws.get() };
    let rt = &mut pw.tp;
    let mut spins = 0u64;
    while !rt.pending.is_empty() {
        poll_pending(rt);
        std::hint::spin_loop();
        spins += 1;
        if spins == MAX_SPINS {
            log::error!("tensor partition: done-flag drain timed out");
            return Err(PlanInvalidated);
        }
    }
    let n = &mut rt.catch_up;
    if n.full_this_token > 0 {
        eprintln!(
            "[tp] recopy step={} pos={} full_layers={}",
            n.step, n.pos, n.full_this_token
        );
        n.full_this_token = 0;
    }
    n.step += 1;
    if let Some(ctl) = rt.ctl.as_mut() {
        if let Some(threads) = ctl.end_token() {
            crate::thread_pool::get_pool().set_active_workers(threads.saturating_sub(1));
        }
        crate::layers::tp_controller::telemetry_push(ctl.stats());
    }
    Ok(())
}

/// Everything `build_tp_layer` needs besides the shared layer config.
pub struct TpLayerInputs<'a> {
    pub ctx: &'a PartitionContext,
    pub host: Arc<TpHostWeights>,
    pub ws: &'a Arc<PartitionWsCell>,
    pub layer: usize,
    pub kernels: Option<&'static crate::cpu_kernels::CpuKernelSet>,
}

/// Build one partitioned layer: the GPU-only layer plan's QKV/RoPE/KV/attention steps are
/// reused as they are; the steps whose shape depends on the split are rebuilt around the whole
/// weights (row offsets / row strides — no copy).
pub fn build_tp_layer(config: &LayerPlanConfig, inp: TpLayerInputs) -> Result<LayerKernelPlan> {
    if !partition_plan_enabled() {
        anyhow::bail!("LLMRS_PARTITION_PLAN=0 — partition plan path disabled");
    }
    inp.host.check()?;
    ensure!(
        config.dim.is_multiple_of(4) && config.head_dim.is_multiple_of(2),
        "tensor partition needs dim % 4 == 0"
    );
    ensure!(
        SegGeom::attn(config.n_heads_q).splittable()
            && SegGeom::ffn(config.ffn_hidden).splittable(),
        "model too small to split"
    );
    let layer = inp.layer;
    let ws_ref: &mut PartitionWorkspace = unsafe { &mut *inp.ws.get() };
    let cl = |t: &Tensor| super::get_cl_mem(t.buffer().as_ref()).cloned();

    let base = super::plan::build_layer_plan(config)?;
    let super::plan::AttnVariant::GpuOnly(gpu) = base.attn else {
        unreachable!("build_layer_plan builds GPU-only layers")
    };
    ensure!(
        matches!(
            gpu.attention,
            AttentionVariant::StandardFlash(_) | AttentionVariant::SplitFlash { .. }
        ),
        "tensor partition needs the flash decode attention (no score readback)"
    );
    // steps_pre_kv = [rms_norm, Q, (bQ), K, (bK), V, (bV), rope Q, rope K]
    let mut pre = gpu.steps_pre_kv;
    let has_bias = config.bq_buf.is_some();
    ensure!(
        pre.len() == if has_bias { 9 } else { 6 },
        "unexpected GPU pre-KV step layout ({} steps)",
        pre.len()
    );
    let rope_k = pre.pop().unwrap();
    let rope_q = pre.pop().unwrap();
    ensure!(rope_q.op_tag == OpTag::Rope && rope_k.op_tag == OpTag::Rope);
    let mut qkv = pre.split_off(1); // drop the stand-alone attention norm
    let q = qkv.remove(0);
    let q_bias = if has_bias { Some(qkv.remove(0)) } else { None };
    let kv_proj = qkv;
    let super::plan::KvUpdateVariant::Standard(kv_update) = gpu.kv_update;

    let flags: Vec<Mem> = ws_ref.flags[layer].iter().map(cl).collect::<Result<_>>()?;
    let staging_attn = cl(&ws_ref.staging_attn)?;
    let staging_ffn = cl(&ws_ref.staging_ffn)?;
    let zero = cl(&ws_ref.zero_dim)?;
    let with_flags = ws_ref.tp.opts.flags;

    let h_g0 = SegGeom::attn(config.n_heads_q)
        .quantize(inp.ctx.gpu_ratio)
        .context("attention split")?;
    let q_kind = GemvKind::select(config.n_q, false, config.is_nosub);

    let entry = entry_norm_step(config, &zero, config.attn_norm_buf)?;
    let entry_flag = signal_step(config, &flags[0])?;
    let (wo, _) = gemv_ld_step(
        config,
        config.out_attn_buf,
        config.wo_buf,
        config.attn_out_buf,
        config.dim,
        h_g0 * config.head_dim,
        config.n_q,
        OpTag::MatmulWo,
    )?;
    let done_a = if with_flags {
        Some(signal_step(config, &flags[1])?)
    } else {
        None
    };
    let dim4 = (config.dim / 4) as i32;
    let merge_attn = simple_step(
        config.simple_ops_program,
        "kernel_add_assign_simple",
        &[
            ArgVal::mem(config.attn_out_buf),
            ArgVal::mem(&staging_attn),
            ArgVal::scalar(&dim4),
        ],
        [config.dim / 4, 1, 1],
        None,
        OpTag::AddAssign,
    )?;
    let entry_f = entry_norm_step(config, config.attn_out_buf, config.ffn_norm_buf)?;
    let entry_flag_f = signal_step(config, &flags[2])?;

    let s0 = inp.ctx.ffn_split;
    let kind = GemvKind::select(
        config.ffn_hidden,
        config.f16_l4_program.is_some(),
        config.is_nosub,
    );
    let gate = super::plan::make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.w_gate_buf,
        config.gate_buf,
        config.ffn_hidden,
        config.dim,
        OpTag::MatmulGateUp,
        config.f16_l4_program,
        config.is_nosub,
    )?;
    let up = super::plan::make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.w_up_buf,
        config.up_buf,
        config.ffn_hidden,
        config.dim,
        OpTag::MatmulGateUp,
        config.f16_l4_program,
        config.is_nosub,
    )?;
    let silu = simple_step(
        config.simple_ops_program,
        "kernel_silu_mul_simple",
        &[
            ArgVal::mem(config.gate_buf),
            ArgVal::mem(config.up_buf),
            ArgVal::scalar(&((s0 / 4) as i32)),
        ],
        [s0 / 4, 1, 1],
        None,
        OpTag::SiluMul,
    )?;
    let (down, _) = gemv_ld_step(
        config,
        config.gate_buf,
        config.w_down_buf,
        config.down_buf,
        config.dim,
        s0,
        config.ffn_hidden,
        OpTag::MatmulDown,
    )?;
    let done_f = if with_flags {
        Some(signal_step(config, &flags[3])?)
    } else {
        None
    };
    let fused_step = simple_step(
        config.simple_ops_program,
        "kernel_partition_fused_merge_residual_f4",
        &[
            ArgVal::mem(config.x_buf),
            ArgVal::mem(config.down_buf),
            ArgVal::mem(&staging_ffn),
            ArgVal::scalar(&dim4),
        ],
        [config.dim / 4, 1, 1],
        None,
        OpTag::AddAssign,
    )?;

    ensure!(
        ws_ref
            .tp
            .kv
            .get(layer)
            .is_some_and(|kv| kv.k.shape().dims()[2] == config.kv_capacity),
        "host KV cache of layer {layer} not prepared"
    );

    let generation = inp.ctx.ratio_generation.clone();
    let residual_host = config.residual_host_ptr as *const f32;
    ensure!(
        !residual_host.is_null(),
        "tensor partition needs the host-mapped residual buffer"
    );
    let common = |layer| SegCommon {
        layer,
        ws: inp.ws.clone(),
        kernels: inp.kernels,
        residual_host,
        ratio_generation_at_build: generation.load(Ordering::Acquire),
        ratio_generation: generation.clone(),
        build_thread: std::thread::current().id(),
    };
    let attn = AttnPartitionStep {
        c: common(layer),
        entry,
        entry_flag,
        q,
        q_kind,
        q_bias,
        kv_proj,
        rope_q,
        rope_k,
        kv_update,
        attention: gpu.attention,
        wo,
        done: done_a,
        bound: Cell::new(usize::MAX),
        host: inp.host.clone(),
        n_heads_q: config.n_heads_q,
        n_kv: config.n_kv_heads,
        head_dim: config.head_dim,
        dim: config.dim,
        freqs: rope_freqs(config),
        k_cache: config.k_cache_buf.clone(),
        v_cache: config.v_cache_buf.clone(),
        kv_capacity: config.kv_capacity,
        cpu_backend: inp.ctx.cpu_backend.clone(),
        scores: config.gpu_score_buf.map(|buf| ScoreRows {
            buf: buf.clone(),
            layer_offset: config.gpu_score_layer_offset as usize,
            stride: config.gpu_score_stride as usize,
        }),
    };
    let ffn = FfnPartitionStep {
        c: common(layer),
        merge_attn,
        entry: entry_f,
        entry_flag: entry_flag_f,
        gate,
        up,
        kind,
        silu,
        down,
        done: done_f,
        merge: super::plan::PartitionMerge::Fused { fused_step },
        bound: Cell::new(usize::MAX),
        host: inp.host,
        dim: config.dim,
        ffn_hidden: config.ffn_hidden,
    };
    Ok(LayerKernelPlan {
        attn: super::plan::AttnVariant::Partitioned(Box::new(attn)),
        ffn: super::plan::FfnVariant::Partitioned(Box::new(ffn)),
        steps_post_ffn: vec![],
        flush_after: false,
    })
}

/// Geometry `prepare_runtime` sizes the controller and the host KV caches with.
pub struct TpRuntimeGeom {
    pub n_layers: usize,
    pub n_heads_q: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub kv_capacity: usize,
}

/// Before the partitioned layers are built: create the controller on the first build (it
/// outlives plan rebuilds), and (re)allocate the host KV caches when the capacity changed —
/// a fresh cache catches up from the GPU cache on its first token.
pub fn prepare_runtime(
    ws: &PartitionWsCell,
    g: &TpRuntimeGeom,
    r0: f32,
    cpu_backend: &Arc<dyn Backend>,
) -> Result<()> {
    use crate::memory::Memory;
    // SAFETY: plan build runs on the dispatch thread.
    let pw = unsafe { &mut *ws.get() };
    if pw.tp.ctl.is_none() {
        let threads = crate::thread_pool::get_pool().n_workers() + 1;
        pw.tp.ctl = Some(TpController::new(
            g.n_layers,
            g.n_heads_q,
            g.ffn_hidden,
            r0,
            threads,
            pw.tp.opts.adaptive && pw.tp.opts.flags,
            pw.tp.opts.cfg,
        ));
    }
    if pw.tp.score_stage.len() != g.n_layers {
        pw.tp.score_stage = vec![Vec::new(); g.n_layers];
        pw.tp.q_stage = vec![Vec::new(); g.n_layers];
    }
    let fits = pw.tp.kv.len() == g.n_layers
        && pw
            .tp
            .kv
            .iter()
            .all(|kv| kv.k.shape().dims()[2] == g.kv_capacity);
    if !fits {
        let shape = vec![1, g.n_kv_heads, g.kv_capacity, g.head_dim];
        let bytes = g.n_kv_heads * g.kv_capacity * g.head_dim * 2;
        let alloc = || -> Result<Tensor> {
            let buf =
                crate::memory::galloc::Galloc::new().alloc(bytes, crate::buffer::DType::F16)?;
            Ok(Tensor::new(
                crate::shape::Shape::new(shape.clone()),
                buf,
                cpu_backend.clone(),
            ))
        };
        pw.tp.kv = (0..g.n_layers)
            .map(|_| {
                Ok(HostKv {
                    k: alloc()?,
                    v: alloc()?,
                    len: 0,
                })
            })
            .collect::<Result<_>>()?;
    }
    Ok(())
}
