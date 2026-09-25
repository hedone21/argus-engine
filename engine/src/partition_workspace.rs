//! Partition workspace types — UMA hybrid CPU-GPU tensor partition data carriers.
//!
//! §13.8-G shared identifier promotion (B-5a sprint):
//! `PartitionWsCell` + `PartitionWorkspace`는 `layers/workspace.rs`에 정의되어
//! `backend/opencl/plan.rs`(L1 → L3 import 위반)에서 직접 import되었다. 두 struct는
//! 사실상 forward path의 *workspace data carrier*이며, 어느 도메인의 *상태 owner*가
//! 아니다 (UnsafeCell wrapper + per-decode lifetime). L3 inference 도메인의
//! identifier로 명시하여 backend가 inference를 import하는 *데이터 의존 방향*을
//! INV-LAYER-001 정상 경로로 정리한다.

use std::cell::UnsafeCell;
use std::sync::Arc;

use anyhow::Result;

use crate::buffer::{Buffer, DType};
use crate::layers::tp_controller::{TpConfig, TpController};
use crate::memory::Memory;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Thin wrapper around `UnsafeCell<PartitionWorkspace>` that makes the
/// interior `Sync` for the restricted single-threaded-dispatch safety model
/// enforced by the OpenCL plan (the partition steps' `run`). Do not use outside
/// of that invariant.
pub struct PartitionWsCell(pub UnsafeCell<PartitionWorkspace>);

impl PartitionWsCell {
    pub fn new(ws: PartitionWorkspace) -> Self {
        Self(UnsafeCell::new(ws))
    }

    #[inline]
    pub fn get(&self) -> *mut PartitionWorkspace {
        self.0.get()
    }
}

// SAFETY: The LayerWorkspace single-threaded-dispatch contract (see
// `LayerWorkspace::partition_ws` doc) guarantees no aliased mutable access
// across threads. Every partition step runs on the plan's dispatch thread.
unsafe impl Send for PartitionWsCell {}
unsafe impl Sync for PartitionWsCell {}

/// Session options of the tensor-partition arm (argus-bench `--tp-*`, ticket 021).
#[derive(Clone, Copy, Debug)]
pub struct TpOptions {
    /// `--tp-adaptive`: the controller moves the split. Off = static split at `--tensor-partition`.
    pub adaptive: bool,
    /// Done-flag kernels + observation. `--tp-no-flags` turns them off (static split only), the
    /// control arm that prices the measurement.
    pub flags: bool,
    pub cfg: TpConfig,
    /// `--tp-start-off`: prepared, but every layer starts on the GPU until `gpu.offload`.
    pub start_off: bool,
}

impl Default for TpOptions {
    fn default() -> Self {
        Self {
            adaptive: false,
            flags: true,
            cfg: TpConfig::default(),
            start_off: false,
        }
    }
}

/// A GPU share whose done-flag had not risen when its CPU share finished: the wait that follows
/// gives the exact GPU time (ticket 021 §D3 observation model).
#[derive(Clone, Copy, Debug)]
pub struct PendingObs {
    pub layer: usize,
    /// 0 = ATTN, 1 = FFN.
    pub seg: usize,
    /// Host pointer of the done-flag.
    pub flag: *mut i32,
    /// When the segment's input flag rose (the GPU share started).
    pub t0: std::time::Instant,
    pub t_cpu: f32,
}

/// Host KV cache of one layer for the CPU attention share: `[1, n_kv, capacity, head_dim]` F16
/// HeadMajor, all KV heads (the CPU recomputes K/V itself — it never reads the GPU cache
/// except once, to catch up after prefill).
pub struct HostKv {
    pub k: Tensor,
    pub v: Tensor,
    /// Positions filled.
    pub len: usize,
}

/// Adaptive-split runtime shared by every partition step of a plan (and kept across plan
/// rebuilds, so the controller's state survives an invalidation).
pub struct TpRuntime {
    pub opts: TpOptions,
    /// Created by the first plan build (needs the model geometry).
    pub ctl: Option<TpController>,
    pub pending: Vec<PendingObs>,
    pub kv: Vec<HostKv>,
    /// Host-KV catch-up counts (ticket 023 T6): layers copied from slot 0 / from their tail.
    pub catch_up: CatchUpCount,
    /// Per layer: the CPU heads' score rows (`n_heads_q` rows of the attention length) and
    /// rotated query row, staged for a non-blocking device write. One buffer per layer because
    /// the next layer's CPU share runs before the write has read its source.
    pub score_stage: Vec<Vec<f32>>,
    pub q_stage: Vec<Vec<f32>>,
    /// `gpu.offload` turned the partition on before the controller existed: the plan build that
    /// creates it marks it as turned on (see [`TpController::on_tokens`]).
    pub on_pending: bool,
}

/// What turning the partition off released (ticket 024), for the `[tp] offload off` line.
pub struct OffloadOff {
    /// Active CPU pool workers before → after.
    pub workers: (usize, usize),
    /// Host KV bytes allocated.
    pub host_kv_freed: usize,
    /// Host KV bytes written (`2 · n_kv · len · head_dim · 2` per layer): what PSS can return.
    pub touched: usize,
    /// `Pss` of `/proc/self/smaps_rollup` right before and after the release, in kB.
    pub pss_before: Option<u64>,
    pub pss_after: Option<u64>,
}

impl TpRuntime {
    /// `gpu.offload` turned the partition off: give the CPU pool back its workers (the contention
    /// rule may have parked some) and free the host KV, so the partition costs no memory while it
    /// is off. Turning it on again allocates a fresh host KV, which the next plan build fills from
    /// slot 0.
    pub fn release_for_off(&mut self) -> OffloadOff {
        let pool = crate::thread_pool::get_pool();
        let before = pool.active_workers();
        pool.set_active_workers(pool.n_workers());
        let workers = (before, pool.active_workers());
        let mut host_kv_freed = 0;
        let mut touched = 0;
        for kv in &self.kv {
            let d = kv.k.shape().dims();
            host_kv_freed += kv.k.size() + kv.v.size();
            touched += 2 * d[1] * kv.len * d[3] * 2;
        }
        let pss_before = pss_kb();
        self.kv = Vec::new();
        let pss_after = pss_kb();
        OffloadOff {
            workers,
            host_kv_freed,
            touched,
            pss_before,
            pss_after,
        }
    }

    /// `gpu.offload` turned the partition on: the controller starts over
    /// ([`TpController::reset_for_on`]).
    pub fn reset_for_on(&mut self) {
        match self.ctl.as_mut() {
            Some(ctl) => ctl.reset_for_on(),
            None => self.on_pending = true,
        }
    }
}

/// `Pss` of this process from `/proc/self/smaps_rollup`, in kB (`None` where it cannot be read).
fn pss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    s.lines()
        .find_map(|l| l.strip_prefix("Pss:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Host-KV catch-up counter. A full copy from slot 0 happens after prefill and after every
/// compaction; one on any other token means the host cache lost track of the GPU cache.
#[derive(Default)]
pub struct CatchUpCount {
    pub full: u64,
    pub tail: u64,
    /// Layers fully re-copied in the token being run.
    pub full_this_token: usize,
    /// RoPE position of the token being run.
    pub pos: usize,
    /// Decode step of the token being run (set by the forward each step).
    pub step: u64,
}

/// Host-mapped (`CL_MEM_ALLOC_HOST_PTR`) buffer that stays mapped for its lifetime, so the CPU
/// reads/writes it through `as_mut_ptr()` while GPU kernels use the same memory.
fn alloc_mapped(
    gpu_alloc: &dyn Fn(usize, DType) -> Result<Arc<dyn Buffer>>,
    bytes: usize,
    shape: Vec<usize>,
    gpu_backend: &Arc<dyn crate::backend::Backend>,
) -> Result<Tensor> {
    let buf = gpu_alloc(bytes, DType::F32)?;
    buf.map_for_cpu()?;
    anyhow::ensure!(
        !buf.as_ptr().is_null(),
        "partition buffer has no host mapping (needs a zero-copy UMA device)"
    );
    // SAFETY: freshly mapped, `bytes` long.
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, bytes) };
    Ok(Tensor::new(Shape::new(shape), buf, gpu_backend.clone()))
}

/// Buffers of the two-segment tensor partition (ticket 021 §D1), one set shared by all layers
/// (plus one set of flags per layer).
///
/// CPU scratch is plain host memory. Everything the GPU also touches is host-mapped
/// ALLOC_HOST_PTR memory: the stagings the CPU writes its partial sums into, the zero vector
/// layer 0's entry norm adds, and the flags.
pub struct PartitionWorkspace {
    // --- CPU scratch (F32) ---
    /// Copy of the normed segment input (`ws.residual`) the CPU share reads: `[dim]`.
    pub residual_cpu: Tensor,
    /// Q / attention output of all heads (`[n_q·head_dim]`); the CPU uses its heads' rows.
    pub q_cpu: Tensor,
    pub attn_out_cpu: Tensor,
    /// K / V of the current token, all KV heads: `[n_kv·head_dim]`.
    pub k_cpu: Tensor,
    pub v_cpu: Tensor,
    /// CPU share of gate / up (`[ffn_hidden]`, the CPU rows packed from 0).
    pub gate_cpu: Tensor,
    pub up_cpu: Tensor,
    /// CPU partial sums before they go to the stagings: `[dim]`.
    pub wo_partial_cpu: Tensor,
    pub down_partial_cpu: Tensor,

    // --- GPU-visible, host-mapped ---
    /// CPU Wo partial, added to `ws.attn_out` before the FFN entry norm: `[dim]`.
    pub staging_attn: Tensor,
    /// CPU down partial, added to `ws.down` (or straight to `x` on the last layer): `[dim]`.
    pub staging_ffn: Tensor,
    /// Zeros: the residual input of layer 0's entry norm (later layers add the previous
    /// layer's FFN output there): `[dim]`.
    pub zero_dim: Tensor,
    /// Per layer `[attn input ready, attn GPU done, ffn input ready, ffn GPU done]`, one
    /// 4-byte flag each — per layer so a late flag of layer `l` never aliases layer `l+1`'s.
    pub flags: Vec<[Tensor; 4]>,

    pub tp: TpRuntime,
}

/// Geometry for [`PartitionWorkspace::new`].
#[derive(Clone, Copy, Debug)]
pub struct PartitionWsGeom {
    pub n_layers: usize,
    pub dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub ffn_hidden: usize,
}

impl PartitionWorkspace {
    /// `gpu_alloc` must return host-mappable (`CL_MEM_ALLOC_HOST_PTR`) buffers.
    pub fn new(
        g: PartitionWsGeom,
        gpu_alloc: &dyn Fn(usize, DType) -> Result<Arc<dyn Buffer>>,
        gpu_backend: Arc<dyn crate::backend::Backend>,
        cpu_backend: Arc<dyn crate::backend::Backend>,
    ) -> Result<Self> {
        use crate::memory::galloc::Galloc;
        let host = Galloc::new();
        let cpu = |n: usize| -> Result<Tensor> {
            let buf = host.alloc(n * 4, DType::F32)?;
            Ok(Tensor::new(
                Shape::new(vec![1, 1, n]),
                buf,
                cpu_backend.clone(),
            ))
        };
        let mapped = |n: usize| alloc_mapped(gpu_alloc, n * 4, vec![1, 1, n], &gpu_backend);
        let flags = (0..g.n_layers)
            .map(|_| -> Result<[Tensor; 4]> {
                Ok([mapped(1)?, mapped(1)?, mapped(1)?, mapped(1)?])
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            residual_cpu: cpu(g.dim)?,
            q_cpu: cpu(g.q_dim)?,
            attn_out_cpu: cpu(g.q_dim)?,
            k_cpu: cpu(g.kv_dim)?,
            v_cpu: cpu(g.kv_dim)?,
            gate_cpu: cpu(g.ffn_hidden)?,
            up_cpu: cpu(g.ffn_hidden)?,
            wo_partial_cpu: cpu(g.dim)?,
            down_partial_cpu: cpu(g.dim)?,
            staging_attn: mapped(g.dim)?,
            staging_ffn: mapped(g.dim)?,
            zero_dim: mapped(g.dim)?,
            flags,
            tp: TpRuntime {
                opts: TpOptions::default(),
                ctl: None,
                pending: Vec::new(),
                kv: Vec::new(),
                catch_up: CatchUpCount::default(),
                score_stage: Vec::new(),
                q_stage: Vec::new(),
                on_pending: false,
            },
        })
    }

    /// Every buffer, for `LayerWorkspace::take_buffers` (keep-alive across a backend switch).
    pub fn buffers(&self) -> Vec<Arc<dyn Buffer>> {
        let mut v: Vec<Arc<dyn Buffer>> = [
            &self.residual_cpu,
            &self.q_cpu,
            &self.attn_out_cpu,
            &self.k_cpu,
            &self.v_cpu,
            &self.gate_cpu,
            &self.up_cpu,
            &self.wo_partial_cpu,
            &self.down_partial_cpu,
            &self.staging_attn,
            &self.staging_ffn,
            &self.zero_dim,
        ]
        .iter()
        .map(|t| t.buffer().clone())
        .collect();
        v.extend(self.flags.iter().flatten().map(|t| t.buffer().clone()));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::galloc::Galloc;

    /// Tests that change the global pool's active worker count.
    static POOL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn host_kv(n_kv: usize, cap: usize, hd: usize, len: usize) -> HostKv {
        let be: Arc<dyn crate::backend::Backend> = Arc::new(CpuBackend::new());
        let t = || {
            let buf = Galloc::new()
                .alloc(n_kv * cap * hd * 2, DType::F16)
                .unwrap();
            Tensor::new(Shape::new(vec![1, n_kv, cap, hd]), buf, be.clone())
        };
        HostKv {
            k: t(),
            v: t(),
            len,
        }
    }

    /// ticket 024: turning the partition off restores every pool worker and frees the host KV.
    #[test]
    fn release_for_off_restores_workers_and_frees_host_kv() {
        let _g = POOL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pool = crate::thread_pool::get_pool();
        let mut rt = TpRuntime {
            opts: TpOptions::default(),
            ctl: None,
            pending: Vec::new(),
            kv: vec![host_kv(2, 64, 8, 10), host_kv(2, 64, 8, 64)],
            catch_up: CatchUpCount::default(),
            score_stage: Vec::new(),
            q_stage: Vec::new(),
            on_pending: false,
        };
        pool.set_active_workers(pool.n_workers().saturating_sub(1));
        let r = rt.release_for_off();
        assert_eq!(
            r.workers,
            (pool.n_workers().saturating_sub(1), pool.n_workers())
        );
        assert_eq!(pool.active_workers(), pool.n_workers());
        assert_eq!(r.host_kv_freed, 2 * 2 * (2 * 64 * 8 * 2));
        assert_eq!(r.touched, 2 * 2 * (10 + 64) * 8 * 2);
        assert!(rt.kv.is_empty());

        rt.reset_for_on();
        assert!(
            rt.on_pending,
            "no controller yet: the next plan build marks it"
        );
    }
}
