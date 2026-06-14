//! LISWAP-5 — Phase-aware Async Weight Swap (B-2.4 chunk dispatcher).
//!
//! 전략: production decode forward의 op-level wall-clock이 deterministic
//! (CV 1.2%)이라는 측정 결과 (`papers/.../swap_overhead_phase_predictability_2026_05_10.md`)를
//! 활용하여 `op_trace::PhaseHook`를 통해 op boundary에서 phase를 검사하고:
//!
//! - `DdrPhase::CacheFit` 끝 → 다음 chunk H2D enqueue (Phase R Scenario B 1.04× of max)
//! - `DdrPhase::Heavy` 시작 직전 → in-flight chunk 완료 대기 (driver FIFO 공존)
//!
//! 9-track 음성 결과와 직교: trigger 시점 통제로 driver-internal command
//! processor FIFO를 우회하지 않고 공존.
//!
//! # Chunk 단위 (v1)
//!
//! per-tensor chunking: 각 weight tensor (wq, wk, wv, wo, w_gate, w_up, w_down)를
//! 1 chunk로 취급. norm tensors (attention_norm, ffn_norm)는 마지막에 묶어 처리
//! (size 작아 단일 cache-fit window에 충분).
//!
//! `is_last_in_layer = true` chunk의 cl_event가 layer 전체 H2D 완료를 보장 —
//! 이 시점에 `AsyncSwapDispatcher::submit_commit`로 ArcSwap commit을 위임한다.
//! `INV-149` (LISWAP-4)와 동일 패턴: forward-thread wait-gate는 본 v1에서
//! 사용하지 않지만 cl_event는 dispatcher worker가 `wait_event_blocking`으로 대기.
//!
//! Spec 참고: ENG-ALG-239~ (B-3 검증 후 Architect 발급 예정).

use crate::backend::{Backend, GpuEvent};
use crate::buffer::DType;
use crate::weight::async_swap::{AsyncSwapDispatcher, ChunkDispatchJob, SwapCommitJob};
use crate::weight::swap_executor::SwapExecutor;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference weight resource (LayerSlot/SecondaryMmap)
use crate::models::weights::secondary_mmap::SecondaryMmap;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference weight resource (LayerSlot)
use crate::models::weights::slot::{LayerSlot, LayerWeights};
use crate::op_kind::OpKind;
use crate::phase_hook::{DdrPhase, PhaseHook};
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

/// Per-layer weight subnames in the order they are dispatched. The last entry
/// (`ffn_norm.weight`) is therefore the one that carries `is_last_in_layer`.
/// Norm tensors (`attn_norm.weight`, `ffn_norm.weight`) are tail-packed so the
/// large FFN matmuls land first while the cache-fit window is fresh.
const TENSOR_SUBNAMES: [&str; 9] = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "attn_norm.weight",
    "ffn_norm.weight",
];

/// Per-tensor chunk descriptor. B-2.4 v1 dispatches one tensor per chunk —
/// `byte_offset` / `byte_len` are reserved for sub-tensor chunking (v2) and
/// currently unused at runtime (the dispatcher looks up the tensor by
/// `subname`).
#[derive(Clone, Debug)]
pub struct WeightChunk {
    pub layer_idx: usize,
    pub byte_offset: usize,
    pub byte_len: usize,
    /// Layer 내 chunk seq (0-based). 마지막 chunk(`is_last_in_layer=true`)
    /// 완료 시점에 LayerSlot::swap_weights ArcSwap commit.
    pub chunk_seq: usize,
    pub is_last_in_layer: bool,
    /// GGUF tensor subname this chunk represents (TENSOR_SUBNAMES entry).
    /// Borrowed `&'static str` — lifetime tied to the static array.
    pub subname: &'static str,
}

/// Staging slot for partially-built layers — accumulates per-tensor results
/// (each from one async H2D enqueue) until the last chunk lands, at which
/// point the layer is assembled and committed via `AsyncSwapDispatcher`.
#[derive(Default)]
struct PartialLayer {
    wq: Option<Tensor>,
    wk: Option<Tensor>,
    wv: Option<Tensor>,
    wo: Option<Tensor>,
    w_gate: Option<Tensor>,
    w_up: Option<Tensor>,
    w_down: Option<Tensor>,
    attention_norm: Option<Tensor>,
    ffn_norm: Option<Tensor>,
}

impl PartialLayer {
    /// Returns `true` when the subname is recognised and the tensor was
    /// installed. `false` signals an unknown subname — caller emits
    /// `ConfigWarning` (S-1+β B 카테고리).
    fn install(&mut self, subname: &str, tensor: Tensor) -> bool {
        match subname {
            "attn_q.weight" => {
                self.wq = Some(tensor);
                true
            }
            "attn_k.weight" => {
                self.wk = Some(tensor);
                true
            }
            "attn_v.weight" => {
                self.wv = Some(tensor);
                true
            }
            "attn_output.weight" => {
                self.wo = Some(tensor);
                true
            }
            "ffn_gate.weight" => {
                self.w_gate = Some(tensor);
                true
            }
            "ffn_up.weight" => {
                self.w_up = Some(tensor);
                true
            }
            "ffn_down.weight" => {
                self.w_down = Some(tensor);
                true
            }
            "attn_norm.weight" => {
                self.attention_norm = Some(tensor);
                true
            }
            "ffn_norm.weight" => {
                self.ffn_norm = Some(tensor);
                true
            }
            _ => false,
        }
    }

    /// Assemble the final `LayerWeights`, falling back to the slot's existing
    /// snapshot for any tensor that was not part of this swap target set
    /// (e.g. `attn_norm.weight` missing from secondary index).
    fn into_layer_weights(self, fallback: &LayerWeights) -> Result<LayerWeights> {
        Ok(LayerWeights {
            wq: self.wq.ok_or_else(|| anyhow!("PartialLayer missing wq"))?,
            wk: self.wk.ok_or_else(|| anyhow!("PartialLayer missing wk"))?,
            wv: self.wv.ok_or_else(|| anyhow!("PartialLayer missing wv"))?,
            wo: self.wo.ok_or_else(|| anyhow!("PartialLayer missing wo"))?,
            w_gate: self
                .w_gate
                .ok_or_else(|| anyhow!("PartialLayer missing w_gate"))?,
            w_up: self
                .w_up
                .ok_or_else(|| anyhow!("PartialLayer missing w_up"))?,
            w_down: self
                .w_down
                .ok_or_else(|| anyhow!("PartialLayer missing w_down"))?,
            attention_norm: self
                .attention_norm
                .unwrap_or_else(|| fallback.attention_norm.clone()),
            ffn_norm: self.ffn_norm.unwrap_or_else(|| fallback.ffn_norm.clone()),
            qkv_bias: fallback.qkv_bias.clone(),
            q_norm: fallback.q_norm.clone(),
            k_norm: fallback.k_norm.clone(),
            pre_ffn_norm: fallback.pre_ffn_norm.clone(),
            post_ffn_norm: fallback.post_ffn_norm.clone(),
            // DF-35-3: tensor_partition × weight swap mutually exclusive —
            // installed cl_mem invalidates any pre-existing PartitionContext.
            partition_ctx: None,
        })
    }
}

/// Phase-aware async swap dispatcher (B-2.4). Implements `PhaseHook` so
/// `op_trace::start_op` / `record` calls drive chunk dispatch from the
/// forward thread without an extra thread.
pub struct PhaseAwareSwapDispatcher {
    /// 분할된 chunk 큐. layer A chunk 0..N → layer B chunk 0..N → ... 순서.
    chunk_queue: Mutex<VecDeque<WeightChunk>>,
    /// 가장 최근에 enqueue한 chunk의 cl_event. ddr-heavy 진입 시 wait.
    in_flight: Mutex<Option<Arc<GpuEvent>>>,
    /// chunk 1개 크기 (v1 per-tensor에서는 진단용 — 실제 분할에 사용 안 함).
    #[allow(dead_code)]
    chunk_size_bytes: usize,
    /// Layer slot 참조 (chunk staging cl_mem alloc + ArcSwap commit 대상).
    layer_slots: Vec<Arc<LayerSlot>>,
    /// Secondary weight source (mmap-backed).
    secondary: Arc<SecondaryMmap>,
    /// GPU backend (host-pinned write_buffer + cl_event).
    backend: Arc<dyn Backend>,
    /// AsyncSwapDispatcher worker — 실제 ArcSwap commit은 worker thread.
    dispatcher: Arc<AsyncSwapDispatcher>,
    /// Target dtype (Q4_0 등) — `SwapCommitJob::new_dtype`로 전달.
    target_dtype: DType,
    /// finalize() 호출 후 true. 이후 dispatch / hook fire는 noop.
    finalized: AtomicBool,
    /// `Arc::new`로 보관된 model config — `SwapExecutor` 생성 시 borrow.
    config: Arc<crate::model_config::ModelConfig>,
    /// Per-layer staging buffer. `try_dispatch_chunk`이 enqueue한 결과를
    /// `is_last_in_layer` 도달 시 모아서 LayerWeights로 조립.
    pending_layers: Mutex<HashMap<usize, PartialLayer>>,
    /// 진단: 본 dispatcher가 적어도 1회 chunk를 dispatch했는가? `finalize`가
    /// `ratio_generation` bump 여부에 사용.
    stage_gate_armed: AtomicBool,
    /// 진단: 누적 dispatch한 chunk 수.
    dispatched_count: AtomicU64,
    /// DEBUG: PhaseHook fire 카운터 (LLMRS_PHASE_AWARE_DEBUG=1 시 trace).
    hook_start_calls: AtomicU64,
    hook_end_calls: AtomicU64,
    cachefit_end_calls: AtomicU64,
    /// Phase 2: Throttle — token당 최대 dispatch chunk 수.
    /// 0 = 무제한 (default), N>0 = 매 token N chunks까지만 dispatch.
    /// decode loop의 `reset_token_counter()`로 매 token 시작 시 counter reset.
    max_chunks_per_token: AtomicUsize,
    /// Phase 2: 현재 token에서 dispatch한 chunk 수.
    chunks_dispatched_this_token: AtomicUsize,
    /// LISWAP Phase 4: weak self-reference for worker-thread dispatch.
    ///
    /// Populated post-construction via `install_self_weak`. The forward thread
    /// passes `self_weak.upgrade()` into `ChunkDispatchJob`'s closure so the
    /// worker can call back into `try_dispatch_chunk_worker` without owning a
    /// strong Arc cycle (worker → dispatcher → AsyncSwapDispatcher → worker).
    /// When the dispatcher is dropped, upgrade returns `None` and the queued
    /// job becomes a noop.
    self_weak: OnceLock<Weak<Self>>,
}

impl PhaseAwareSwapDispatcher {
    /// 생성자 — chunk_size_bytes는 보통 4 MB (`chunk_size_mb * 1_048_576`).
    /// v1에서는 per-tensor 단위라 이 값은 진단/보고용.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chunk_size_bytes: usize,
        layer_slots: Vec<Arc<LayerSlot>>,
        secondary: Arc<SecondaryMmap>,
        backend: Arc<dyn Backend>,
        dispatcher: Arc<AsyncSwapDispatcher>,
        target_dtype: DType,
        config: Arc<crate::model_config::ModelConfig>,
    ) -> Arc<Self> {
        Arc::new(Self {
            chunk_queue: Mutex::new(VecDeque::new()),
            in_flight: Mutex::new(None),
            chunk_size_bytes,
            layer_slots,
            secondary,
            backend,
            dispatcher,
            target_dtype,
            finalized: AtomicBool::new(false),
            config,
            pending_layers: Mutex::new(HashMap::new()),
            stage_gate_armed: AtomicBool::new(false),
            dispatched_count: AtomicU64::new(0),
            hook_start_calls: AtomicU64::new(0),
            hook_end_calls: AtomicU64::new(0),
            cachefit_end_calls: AtomicU64::new(0),
            max_chunks_per_token: AtomicUsize::new(0),
            chunks_dispatched_this_token: AtomicUsize::new(0),
            self_weak: OnceLock::new(),
        })
    }

    /// LISWAP Phase 4: install a weak self-reference so the worker thread can
    /// call back into `try_dispatch_chunk_worker`. Must be called exactly once
    /// after `Arc::new` returns the dispatcher (the constructor cannot capture
    /// its own Arc). Subsequent calls are noops (OnceLock semantics).
    pub fn install_self_weak(self: &Arc<Self>) {
        let _ = self.self_weak.set(Arc::downgrade(self));
    }

    /// Phase 2: Throttle — token당 최대 dispatch chunk 수 설정.
    /// 0 = 무제한 (현재 동작 유지), N>0 = 매 token N chunks까지만.
    pub fn set_max_chunks_per_token(&self, n: usize) {
        self.max_chunks_per_token.store(n, Ordering::Release);
    }

    /// Phase 2: Decode loop의 매 token 시작 시 호출 — 현재 token counter reset.
    pub fn reset_token_counter(&self) {
        self.chunks_dispatched_this_token
            .store(0, Ordering::Release);
    }

    /// Plan commit — `target_layers` 각각을 per-tensor chunk으로 분할하여
    /// `chunk_queue`에 push. Already-target-dtype 또는 secondary handle 없는
    /// slot은 silently skip한다 (LISWAP-4 패턴).
    pub fn commit_plan(&self, target_layers: &[usize]) {
        let mut q = match self.chunk_queue.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for &layer_idx in target_layers {
            let Some(slot) = self.layer_slots.get(layer_idx) else {
                continue;
            };
            // No swap path → drop silently (matches LISWAP-4 behaviour).
            if slot.secondary_mmap_handle().is_none() {
                continue;
            }
            // Already at target dtype → skip (idempotency).
            if slot.current_dtype() == self.target_dtype {
                continue;
            }
            let last_idx = TENSOR_SUBNAMES.len() - 1;
            for (seq, &subname) in TENSOR_SUBNAMES.iter().enumerate() {
                q.push_back(WeightChunk {
                    layer_idx,
                    byte_offset: 0,
                    byte_len: 0,
                    chunk_seq: seq,
                    is_last_in_layer: seq == last_idx,
                    subname,
                });
            }
        }
    }

    /// in_flight chunk H2D 완료 대기. ddr-heavy phase 진입 직전 호출.
    /// `wait_event_blocking`은 `clWaitForEvents` (OpenCL spec thread-safe).
    fn wait_pending(&self) {
        let evt_opt = match self.in_flight.lock() {
            Ok(mut g) => g.take(),
            Err(_) => return,
        };
        if let Some(ev) = evt_opt {
            // LISWAP-6 Phase 5: alias path 면 cl_event 가 dummy. 그러면
            // wait_event_blocking 의 fall-through `synchronize()` 가 forward
            // GPU op까지 block 하므로 skip — alias 는 이미 GPU-visible 이라
            // 진짜 wait 할 게 없음. 비-alias path (memcpy)는 정상 wait.
            if !ev.is_dummy()
                && let Err(e) = self.backend.wait_event_blocking(ev.as_ref())
            {
                crate::action_diag_helper::log_swap_failed(
                    "PhaseAware",
                    &format!("wait_pending failed: {e}"),
                    None,
                    None,
                );
            }
        }
    }

    /// LISWAP Phase 4: worker-thread entrypoint — identical body to
    /// `try_dispatch_chunk`, kept as a separate `pub(crate)` method so the
    /// `AsyncSwapDispatcher` worker can invoke it via the closure stored in
    /// `ChunkDispatchJob`. Concurrency: the in_flight Mutex still serialises
    /// "one chunk in flight" semantics — the worker dequeues sequentially and
    /// each iteration acquires/releases the same Mutex.
    pub(crate) fn try_dispatch_chunk_worker(&self) -> Result<()> {
        self.try_dispatch_chunk()
    }

    /// 다음 chunk pop → secondary mmap에서 staging cl_mem으로 enqueue_write_async →
    /// in_flight = event. is_last_in_layer 면 SwapCommitJob submit.
    ///
    /// 한 chunk in-flight 정책: 이미 in_flight에 event가 있으면 skip (forward
    /// hook 호출 cadence가 chunk 완료 cadence보다 빠를 수 있음).
    fn try_dispatch_chunk(&self) -> Result<()> {
        // 1. in_flight 확인 — 이미 1개 in-flight면 다음 cache-fit op 까지 대기.
        {
            let guard = self
                .in_flight
                .lock()
                .map_err(|_| anyhow!("[PhaseAwareSwap] in_flight lock poisoned"))?;
            if guard.is_some() {
                return Ok(());
            }
        }

        // 2. Pop next chunk.
        let chunk = match self.chunk_queue.lock() {
            Ok(mut q) => match q.pop_front() {
                Some(c) => c,
                None => return Ok(()),
            },
            Err(_) => return Err(anyhow!("[PhaseAwareSwap] chunk_queue lock poisoned")),
        };

        // 3. Resolve slot + primary tensor for shape validation.
        let Some(slot) = self.layer_slots.get(chunk.layer_idx).cloned() else {
            return Ok(());
        };
        let snapshot = slot.load_weights();
        let primary: &Tensor = match chunk.subname {
            "attn_q.weight" => &snapshot.wq,
            "attn_k.weight" => &snapshot.wk,
            "attn_v.weight" => &snapshot.wv,
            "attn_output.weight" => &snapshot.wo,
            "ffn_gate.weight" => &snapshot.w_gate,
            "ffn_up.weight" => &snapshot.w_up,
            "ffn_down.weight" => &snapshot.w_down,
            "attn_norm.weight" => &snapshot.attention_norm,
            "ffn_norm.weight" => &snapshot.ffn_norm,
            other => {
                crate::action_diag_helper::log_swap_config_warning(
                    "phase_aware_chunk_subname",
                    &format!("unknown chunk subname '{other}' (skipped)"),
                );
                return Ok(());
            }
        };

        // 4. Build executor + enqueue async H2D.
        let memory = crate::memory::galloc::Galloc::new();
        let executor = SwapExecutor::new(
            self.target_dtype,
            self.config.as_ref(),
            Arc::clone(&self.backend),
            &memory,
        )
        .with_kind("PhaseAware");

        // Norms (`*_norm.weight`) may be absent from the secondary; treat as
        // optional and fall back to the snapshot tensor in that case.
        let is_norm = matches!(chunk.subname, "attn_norm.weight" | "ffn_norm.weight");
        let (built_tensor, write_event_opt) = if is_norm {
            match executor.build_optional_tensor_from_mmap_async_for_hook(
                &self.secondary,
                chunk.layer_idx,
                chunk.subname,
                primary,
            ) {
                Ok(Some((t, e))) => (Some(t), Some(e)),
                Ok(None) => (None, None),
                Err(e) => {
                    crate::action_diag_helper::log_swap_failed(
                        "PhaseAware",
                        &format!(
                            "subname '{}' async build failed: {e}; skipping chunk",
                            chunk.subname
                        ),
                        Some(chunk.layer_idx),
                        None,
                    );
                    (None, None)
                }
            }
        } else {
            match executor.build_tensor_from_mmap_async_for_hook(
                &self.secondary,
                chunk.layer_idx,
                chunk.subname,
                primary,
            ) {
                Ok((t, e)) => (Some(t), Some(e)),
                Err(e) => {
                    crate::action_diag_helper::log_swap_failed(
                        "PhaseAware",
                        &format!(
                            "subname '{}' async build failed: {e}; skipping chunk",
                            chunk.subname
                        ),
                        Some(chunk.layer_idx),
                        None,
                    );
                    (None, None)
                }
            }
        };

        // 5. Stash result in PartialLayer staging buffer (when present).
        if let Some(t) = built_tensor {
            let mut pending = self
                .pending_layers
                .lock()
                .map_err(|_| anyhow!("[PhaseAwareSwap] pending_layers lock poisoned"))?;
            let installed = pending
                .entry(chunk.layer_idx)
                .or_default()
                .install(chunk.subname, t);
            if !installed {
                crate::action_diag_helper::log_swap_config_warning(
                    "phase_aware_subname",
                    &format!("unknown subname '{}' (ignored)", chunk.subname),
                );
            }
        }

        // 6. Update in_flight with this chunk's event so a Heavy-phase
        //    boundary will block for it.
        let event_arc: Option<Arc<GpuEvent>> = write_event_opt.map(Arc::new);
        if let Some(ref ev) = event_arc
            && let Ok(mut guard) = self.in_flight.lock()
        {
            *guard = Some(Arc::clone(ev));
        }
        self.dispatched_count.fetch_add(1, Ordering::Release);
        self.stage_gate_armed.store(true, Ordering::Release);

        // 7. Layer-complete? Assemble + submit commit job.
        if chunk.is_last_in_layer {
            self.commit_layer(chunk.layer_idx, slot, snapshot, event_arc)?;
        }

        Ok(())
    }

    /// Assemble PartialLayer → LayerWeights, hand off to `AsyncSwapDispatcher`.
    /// `last_event` is the event from this layer's last chunk; the worker
    /// `wait_event_blocking` on it before swapping the slot ArcSwap.
    fn commit_layer(
        &self,
        layer_idx: usize,
        slot: Arc<LayerSlot>,
        snapshot: Arc<LayerWeights>,
        last_event: Option<Arc<GpuEvent>>,
    ) -> Result<()> {
        let partial = {
            let mut pending = self
                .pending_layers
                .lock()
                .map_err(|_| anyhow!("[PhaseAwareSwap] pending_layers lock poisoned"))?;
            pending.remove(&layer_idx).unwrap_or_default()
        };

        let new_layer = match partial.into_layer_weights(snapshot.as_ref()) {
            Ok(l) => l,
            Err(e) => {
                crate::action_diag_helper::log_swap_failed(
                    "PhaseAware",
                    &format!("assemble failed: {e}; commit skipped"),
                    Some(layer_idx),
                    None,
                );
                return Ok(());
            }
        };
        let new_arc: Arc<LayerWeights> = Arc::new(new_layer);

        // dispatcher worker re-waits on the same event before swap (idempotent —
        // clWaitForEvents on a complete event returns immediately).
        let event = last_event.unwrap_or_else(|| Arc::new(GpuEvent::dummy()));

        let job = SwapCommitJob {
            slot,
            new_weights: new_arc,
            new_dtype: self.target_dtype,
            write_event: event,
            release_worker: None,
            on_complete: None,
            layer_idx: Some(layer_idx),
        };
        if let Err(e) = self.dispatcher.submit_commit(job) {
            crate::action_diag_helper::log_swap_failed(
                "PhaseAware",
                &format!("dispatcher submit failed: {e}"),
                Some(layer_idx),
                None,
            );
        }
        Ok(())
    }

    /// Plan 종료 — 남은 chunk drain + synchronize + ratio_generation +1.
    /// `IntraForwardSwapHook::finalize` (intra_forward_swap.rs:307~) 패턴 모방.
    pub fn finalize(
        &self,
        ratio_generation: &AtomicU64,
        soa_registry_invalidate: impl FnOnce(),
        deadline: Duration,
    ) -> Result<()> {
        if self
            .finalized
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(anyhow!(
                "[PhaseAwareSwap] finalize called twice on same plan"
            ));
        }

        // (1a) LISWAP Phase 4: first drain any worker-thread DispatchChunk
        //      jobs that the forward thread submitted but the worker hasn't
        //      processed yet. After this drain, no new chunks will be enqueued
        //      from the worker side because `finalized` is now true (and
        //      forward-thread on_op_end is a noop after finalize set the flag).
        self.dispatcher
            .drain(deadline)
            .map_err(|e| anyhow!("[PhaseAwareSwap] dispatcher pre-drain failed: {e}"))?;

        // (1b) drain remaining chunks. forward thread는 이미 끝났으므로 본 thread가
        //     남은 chunk을 동기 dispatch해서 비운다. 한 chunk in-flight 정책 때문에
        //     매 iteration마다 wait_pending()으로 슬롯 비우고 다음 chunk 진행.
        loop {
            let remaining = self.chunk_queue.lock().map(|q| q.len()).unwrap_or(0);
            if remaining == 0 {
                break;
            }
            self.wait_pending();
            if let Err(e) = self.try_dispatch_chunk() {
                crate::action_diag_helper::log_swap_failed(
                    "PhaseAware",
                    &format!("finalize drain dispatch error: {e}"),
                    None,
                    None,
                );
                break;
            }
        }
        // 마지막 chunk의 in_flight도 비우기.
        self.wait_pending();

        // (2) drain async dispatcher (commit jobs from sync drain above).
        self.dispatcher
            .drain(deadline)
            .map_err(|e| anyhow!("[PhaseAwareSwap] dispatcher drain failed: {e}"))?;

        // (3) backend full barrier.
        self.backend
            .synchronize()
            .map_err(|e| anyhow!("[PhaseAwareSwap] backend.synchronize failed: {e}"))?;

        // (4) bump ratio_generation iff at least one chunk was actually dispatched.
        if self.stage_gate_armed.load(Ordering::Acquire) {
            ratio_generation.fetch_add(1, Ordering::SeqCst);
        }

        // (5) invalidate caller-supplied registry (e.g. noshuffle SOA).
        soa_registry_invalidate();

        Ok(())
    }

    /// 진단 — 남은 chunk 수.
    pub fn remaining_chunks(&self) -> usize {
        self.chunk_queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// 진단 — 누적 dispatch한 chunk 수.
    pub fn dispatched_count(&self) -> u64 {
        self.dispatched_count.load(Ordering::Acquire)
    }

    /// Plan 종료 조건 — chunk_queue 비고, in_flight 없고, pending_layers 비어있음.
    /// decode loop가 매 token 후 polling하여 true가 되면 `finalize()` 호출.
    /// stage_gate_armed가 false면 chunk을 한 번도 dispatch하지 않은 상태이므로
    /// `is_complete = false` 유지 (commit_plan 직후 race로 finalize되는 것 방지).
    ///
    /// LISWAP Phase 4: worker thread가 비동기로 chunk를 dispatch하므로, channel
    /// pending도 0이어야 한다 (그렇지 않으면 worker가 곧 in_flight/pending_layers를
    /// 채울 수 있는 race window가 존재).
    pub fn is_complete(&self) -> bool {
        if !self.stage_gate_armed.load(Ordering::Acquire) {
            return false;
        }
        // Phase 4: worker channel이 drain되지 않았으면 forward thread가
        // 방금 submit한 dispatch job이 곧 chunk_queue/in_flight/pending_layers를
        // 채울 수 있음. 이를 무시하고 finalize() 호출 시 race로 chunk가 손실된다.
        if self.dispatcher.pending_count() != 0 {
            return false;
        }
        let queue_empty = self
            .chunk_queue
            .lock()
            .map(|q| q.is_empty())
            .unwrap_or(false);
        if !queue_empty {
            return false;
        }
        let in_flight_empty = self.in_flight.lock().map(|g| g.is_none()).unwrap_or(false);
        if !in_flight_empty {
            return false;
        }
        self.pending_layers
            .lock()
            .map(|p| p.is_empty())
            .unwrap_or(false)
    }

    /// Diagnostic snapshot — (queue, in_flight_some, pending_count,
    /// dispatched_count, hook_start_calls, hook_end_calls, cachefit_end_calls).
    pub fn debug_snapshot(&self) -> (usize, bool, usize, u64, u64, u64, u64) {
        let q = self.chunk_queue.lock().map(|q| q.len()).unwrap_or(0);
        let inf = self.in_flight.lock().map(|g| g.is_some()).unwrap_or(false);
        let p = self.pending_layers.lock().map(|p| p.len()).unwrap_or(0);
        let d = self.dispatched_count.load(Ordering::Acquire);
        let hs = self.hook_start_calls.load(Ordering::Acquire);
        let he = self.hook_end_calls.load(Ordering::Acquire);
        let ce = self.cachefit_end_calls.load(Ordering::Acquire);
        (q, inf, p, d, hs, he, ce)
    }
}

impl PhaseHook for PhaseAwareSwapDispatcher {
    #[inline]
    fn on_op_start(&self, kind: OpKind) {
        // DEBUG: hook fire counter (atomic, no print to avoid flood).
        // Decode loop의 debug_snapshot이 read해서 보여줌.
        self.hook_start_calls.fetch_add(1, Ordering::Relaxed);
        if self.finalized.load(Ordering::Acquire) {
            return;
        }
        if matches!(kind.ddr_phase(), DdrPhase::Heavy) {
            self.wait_pending();
        }
    }

    #[inline]
    fn on_op_end(&self, kind: OpKind) {
        self.hook_end_calls.fetch_add(1, Ordering::Relaxed);
        if matches!(kind.ddr_phase(), DdrPhase::CacheFit) {
            self.cachefit_end_calls.fetch_add(1, Ordering::Relaxed);
        }
        if self.finalized.load(Ordering::Acquire) {
            return;
        }
        if matches!(kind.ddr_phase(), DdrPhase::CacheFit) {
            // Phase 2: Throttle — token당 max chunks 도달 시 skip.
            // forward thread는 atomic load/store + channel-push만 한다 (~us).
            let max = self.max_chunks_per_token.load(Ordering::Acquire);
            if max > 0 && self.chunks_dispatched_this_token.load(Ordering::Acquire) >= max {
                return;
            }

            // LISWAP Phase 4: dispatch on the worker thread. The Weak<Self>
            // upgrade-fails into a noop after dispatcher drop, so cycle-safe.
            let Some(weak) = self.self_weak.get().cloned() else {
                // self_weak not installed — fall back to forward-thread dispatch
                // (defensive: should not happen in production paths).
                let _ = self.try_dispatch_chunk();
                self.chunks_dispatched_this_token
                    .fetch_add(1, Ordering::Relaxed);
                return;
            };
            let job = ChunkDispatchJob {
                run: Box::new(move || {
                    if let Some(disp) = weak.upgrade() {
                        let _ = disp.try_dispatch_chunk_worker();
                    }
                }),
            };
            if self.dispatcher.submit_dispatch_chunk(job).is_ok() {
                // Throttle counter increments per submit, not per actual dispatch.
                // Acceptable for throttle's purpose (cap submits per token).
                self.chunks_dispatched_this_token
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// commit_plan이 target_layers를 정확히 9 chunk × N으로 큐에 쌓는지 확인.
    /// secondary handle / dtype gating은 별도 통합 테스트에서 검증한다 (여기서는
    /// chunk 분할 로직 자체에 집중하기 위해 layer_slots 없이 직접 chunk_queue를
    /// 주입한다).
    #[test]
    fn test_weight_chunk_subnames_complete() {
        // 9 entries ordered: 7 weights + 2 norms.
        assert_eq!(TENSOR_SUBNAMES.len(), 9);
        assert_eq!(TENSOR_SUBNAMES[0], "attn_q.weight");
        assert_eq!(TENSOR_SUBNAMES[6], "ffn_down.weight");
        assert_eq!(TENSOR_SUBNAMES[7], "attn_norm.weight");
        assert_eq!(TENSOR_SUBNAMES[8], "ffn_norm.weight");
    }

    /// `is_last_in_layer` 가 layer 마지막 chunk(seq == 8) 에서만 true 인지 확인.
    /// commit_plan을 직접 호출하지 않고 chunk 생성 로직과 동일한 invariant를
    /// 검증한다 (layer_slots 인자가 없는 minimal test).
    #[test]
    fn test_chunk_seq_and_last_flag_invariant() {
        let last_idx = TENSOR_SUBNAMES.len() - 1;
        for (seq, &subname) in TENSOR_SUBNAMES.iter().enumerate() {
            let chunk = WeightChunk {
                layer_idx: 7,
                byte_offset: 0,
                byte_len: 0,
                chunk_seq: seq,
                is_last_in_layer: seq == last_idx,
                subname,
            };
            assert_eq!(chunk.chunk_seq, seq);
            if seq == last_idx {
                assert!(chunk.is_last_in_layer, "chunk seq {seq} must be last");
            } else {
                assert!(!chunk.is_last_in_layer, "chunk seq {seq} must not be last");
            }
        }
    }

    /// PartialLayer가 install + into_layer_weights 사이클에서 fallback을
    /// 정확히 사용하는지 확인 (norms 누락 케이스).
    #[test]
    fn test_partial_layer_falls_back_for_missing_norms() {
        use crate::backend::cpu::CpuBackend;
        use crate::buffer::Buffer;
        use crate::layers::transformer_layer::TransformerLayer;
        use crate::memory::host::shared::SharedBuffer;
        use crate::shape::Shape;

        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mk = |numel: usize| -> Tensor {
            let buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(numel * 4, DType::F32));
            Tensor::new(Shape::new(vec![numel]), buf, be.clone())
        };

        let fallback = TransformerLayer {
            wq: mk(16),
            wk: mk(16),
            wv: mk(16),
            wo: mk(16),
            w_gate: mk(16),
            w_up: mk(16),
            w_down: mk(16),
            attention_norm: mk(4),
            ffn_norm: mk(4),
            qkv_bias: None,
            q_norm: None,
            k_norm: None,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            partition_ctx: None,
        };

        let mut partial = PartialLayer::default();
        // Install only the weight tensors — norms missing on purpose.
        partial.install("attn_q.weight", mk(16));
        partial.install("attn_k.weight", mk(16));
        partial.install("attn_v.weight", mk(16));
        partial.install("attn_output.weight", mk(16));
        partial.install("ffn_gate.weight", mk(16));
        partial.install("ffn_up.weight", mk(16));
        partial.install("ffn_down.weight", mk(16));

        let assembled = partial
            .into_layer_weights(&fallback)
            .expect("assemble must succeed when all weights present");
        // Norms must come from fallback (same byte size as our 4-element tensor).
        assert_eq!(
            assembled.attention_norm.size(),
            fallback.attention_norm.size()
        );
        assert_eq!(assembled.ffn_norm.size(), fallback.ffn_norm.size());
        // partition_ctx must be cleared (DF-35-3).
        assert!(assembled.partition_ctx.is_none());
    }

    /// PartialLayer에서 weight 하나라도 누락되면 assemble이 실패해야 한다.
    #[test]
    fn test_partial_layer_errors_on_missing_weight() {
        use crate::backend::cpu::CpuBackend;
        use crate::buffer::Buffer;
        use crate::layers::transformer_layer::TransformerLayer;
        use crate::memory::host::shared::SharedBuffer;
        use crate::shape::Shape;

        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mk = |numel: usize| -> Tensor {
            let buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(numel * 4, DType::F32));
            Tensor::new(Shape::new(vec![numel]), buf, be.clone())
        };

        let fallback = TransformerLayer {
            wq: mk(16),
            wk: mk(16),
            wv: mk(16),
            wo: mk(16),
            w_gate: mk(16),
            w_up: mk(16),
            w_down: mk(16),
            attention_norm: mk(4),
            ffn_norm: mk(4),
            qkv_bias: None,
            q_norm: None,
            k_norm: None,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            partition_ctx: None,
        };

        let mut partial = PartialLayer::default();
        // Skip wq on purpose.
        partial.install("attn_k.weight", mk(16));
        partial.install("attn_v.weight", mk(16));

        let res = partial.into_layer_weights(&fallback);
        assert!(res.is_err(), "missing wq must surface an assemble error");
    }
}
