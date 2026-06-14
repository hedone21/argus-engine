//! `IntraForwardSwapHook` — Intra-forward Layer-aligned Swap (LISWAP-4).
//!
//! Spec: ENG-ALG-235~238, ENG-DAT-101, ENG-DAT-C18, INV-147~150.
//! Arch:  `arch/weight_swap.md` §10.
//!
//! Forward `layer i` 종료 직후 hook이 호출되면 plan에 등록된 layer의 swap을
//! 별도 transfer queue로 비동기 dispatch하고, ArcSwap commit은
//! `AsyncSwapDispatcher` worker thread가 cl_event 완료 후 수행한다.
//!
//! 주요 구성:
//! - `LayerBoundaryHook` trait — forward layer 경계 dispatch hook (ENG-ALG-235).
//! - `IntraForwardSwapPlan` — dispatch 대상 layer 집합 + idempotent mark
//!   (ENG-ALG-236, INV-148).
//! - `IntraForwardSwapHook` — 본체. plan + dispatcher + secondary +
//!   pending_events registry (ENG-ALG-237, ENG-DAT-101).
//! - `pending_event_for(idx)` — wait gate read (ENG-ALG-238, INV-149).
//! - `finalize(...)` — drain → synchronize → ratio_generation +1 → invalidate
//!   (INV-150).
//!
//! `GpuEvent` 공유 (LISWAP-4 v2): `enqueue_write_async`가 반환한 진짜 cl_event를
//! `Arc<GpuEvent>`로 wrap하여 (1) `pending_events[idx]`에 저장 (forward thread
//! wait gate가 read), (2) `SwapCommitJob::write_event`에도 동일 Arc clone 전달
//! (dispatcher worker가 read). 두 thread는 같은 cl_event에 `wait_event_blocking`
//! (= `clWaitForEvents`, OpenCL spec상 thread-safe). v1의 dummy event 사용은
//! `wait_event_blocking`이 `inner_cl=None` fall-through로 `clFinish` 풀 배리어를
//!호출하던 버그를 유발했음 — v2는 진짜 event로 fast no-op 보장.

use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use anyhow::{Result, anyhow};
use arc_swap::ArcSwapOption;

use crate::backend::{Backend, GpuEvent};
use crate::buffer::DType;
use crate::layer_boundary_hook::LayerBoundaryHook;
use crate::runtime_resources_access::ReleaseWorkerAccess;
use crate::weight::async_swap::{AsyncSwapDispatcher, SwapCommitJob};
use crate::weight::swap_executor::SwapExecutor;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference weight resource (LayerSlot/SecondaryMmap)
use crate::models::weights::secondary_mmap::SecondaryMmap;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference weight resource (LayerSlot)
use crate::models::weights::slot::{LayerSlot, LayerWeights};

// ── IntraForwardSwapPlan ──────────────────────────────────────────────────

/// Per-plan dispatch state (ENG-ALG-236).
///
/// `dispatch_at`: 해당 layer index 도달 시 swap dispatch.
/// `dispatched`: 이미 dispatch한 layer 집합 (idempotency, INV-148).
pub struct IntraForwardSwapPlan {
    dispatch_at: BTreeSet<usize>,
    dispatched: BTreeSet<usize>,
    started_at_token: usize,
    layer_order: Vec<usize>,
}

impl IntraForwardSwapPlan {
    /// Caller가 `layers.is_empty()`면 plan을 생성하지 않는 책임.
    pub fn new(layers: Vec<usize>, token: usize) -> Self {
        let dispatch_at: BTreeSet<usize> = layers.iter().copied().collect();
        Self {
            dispatch_at,
            dispatched: BTreeSet::new(),
            started_at_token: token,
            layer_order: layers,
        }
    }

    /// `idx`가 dispatch 대상이고 아직 dispatch되지 않았는가? (side-effect 없음)
    #[inline]
    pub fn should_dispatch(&self, idx: usize) -> bool {
        self.dispatch_at.contains(&idx) && !self.dispatched.contains(&idx)
    }

    /// `idx`를 dispatch 완료 표시. 이후 `should_dispatch(idx)`는 false.
    #[inline]
    pub fn mark_dispatched(&mut self, idx: usize) {
        self.dispatched.insert(idx);
    }

    /// 모든 dispatch 대상이 mark되었는가? (빈 plan은 즉시 true)
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.dispatched.is_superset(&self.dispatch_at)
    }

    /// 진단용: 아직 dispatch되지 않은 layer set.
    pub fn pending_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.dispatch_at.difference(&self.dispatched).copied()
    }

    /// 진단용: started token.
    #[inline]
    pub fn started_at_token(&self) -> usize {
        self.started_at_token
    }

    /// 진단용: 외부 입력 순서 그대로의 layer 리스트.
    pub fn layer_order(&self) -> &[usize] {
        &self.layer_order
    }
}

// ── IntraForwardSwapHook ──────────────────────────────────────────────────

/// `LayerBoundaryHook` 구현체 — forward 중간 layer 경계에서 swap을 dispatch
/// (ENG-ALG-237, ENG-DAT-101).
///
/// 동작:
/// 1. `on_layer_boundary(idx)` — plan 확인 → secondary tensor build →
///    `enqueue_write_async` → arm pending_events[idx] → `submit_commit` →
///    mark plan.
/// 2. dispatcher worker가 cl_event 대기 후 ArcSwap commit + clear pending.
/// 3. `pending_event_for(idx)` — forward thread가 layer K 진입 직전 wait
///    gate에서 호출 (ENG-ALG-238, INV-149).
/// 4. `finalize` — plan complete 후 decode loop가 호출. drain → synchronize
///    → ratio_generation+1 → invalidate (INV-150).
///
/// hook이 `Arc<Self>`로 wrap되는 이유: dispatcher worker thread가
/// `clear_pending` callback을 통해 공유 참조를 보유해야 하므로.
pub struct IntraForwardSwapHook {
    plan: Mutex<IntraForwardSwapPlan>,
    dispatcher: Arc<AsyncSwapDispatcher>,
    /// Secondary weight source. `None` only in unit tests that exercise the
    /// trait-level wait-gate paths without going through `on_layer_boundary`'s
    /// dispatch branch.
    secondary: Option<Arc<SecondaryMmap>>,
    layer_slots: Vec<Arc<LayerSlot>>,
    backend: Arc<dyn Backend>,
    release_worker: Option<Arc<dyn ReleaseWorkerAccess>>,
    target_dtype: DType,
    /// Per-slot pending event registry (ENG-DAT-101). Indexed by layer_idx.
    /// `None` = no in-flight swap. Lock-free read by forward thread.
    /// 값은 sentinel `GpuEvent::dummy()` (in-flight signaling용); 진짜
    /// cl_event는 dispatcher worker가 보유.
    pending_events: Vec<ArcSwapOption<GpuEvent>>,
    /// Stage gate flag. `on_layer_boundary`가 적어도 1회 dispatch에 성공하면
    /// `true`. `finalize`가 `ratio_generation` bump 여부 결정에 사용.
    stage_gate_armed: AtomicBool,
    /// Set when `finalize` succeeds. Subsequent `finalize` calls return `Err`
    /// so the caller cannot accidentally bump ratio_generation twice on the
    /// same plan (INV-150).
    finalized: AtomicBool,
    /// Model config — needed by `SwapExecutor::build_layer_from_mmap_async`
    /// for Q/K permutation gating.
    config: Arc<crate::model_config::ModelConfig>,
    /// Self-weak reference for the dispatcher worker callback path. Set
    /// after construction by `Arc::new_cyclic`. Used so the trait-impl
    /// `on_layer_boundary` can hand a typed `Arc<Self>` to the callback
    /// closure (trait-internal `&self` cannot upgrade to `Arc<Self>`).
    self_weak: Weak<Self>,
    _phantom: PhantomData<()>,
}

impl IntraForwardSwapHook {
    /// Production constructor.
    ///
    /// `layer_slots.len()` is the total layer count (used for `pending_events`
    /// size). `target_dtype` is the dtype the swap converts to.
    /// `config` is borrowed by `SwapExecutor` during dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layers: Vec<usize>,
        token: usize,
        dispatcher: Arc<AsyncSwapDispatcher>,
        secondary: Arc<SecondaryMmap>,
        layer_slots: Vec<Arc<LayerSlot>>,
        backend: Arc<dyn Backend>,
        release_worker: Option<Arc<dyn ReleaseWorkerAccess>>,
        target_dtype: DType,
        config: Arc<crate::model_config::ModelConfig>,
    ) -> Arc<Self> {
        Self::new_internal(
            layers,
            token,
            dispatcher,
            Some(secondary),
            layer_slots,
            backend,
            release_worker,
            target_dtype,
            config,
        )
    }

    /// Test-only constructor that omits the secondary mmap. Hook constructed
    /// this way will not dispatch on `on_layer_boundary` (no slot has a
    /// secondary handle), but `arm_pending_for_test` / `clear_pending_for_test`
    /// / `pending_event_for` work as usual for INV-149 ordering tests.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn new_for_test(
        layers: Vec<usize>,
        token: usize,
        dispatcher: Arc<AsyncSwapDispatcher>,
        layer_slots: Vec<Arc<LayerSlot>>,
        backend: Arc<dyn Backend>,
        target_dtype: DType,
        config: Arc<crate::model_config::ModelConfig>,
    ) -> Arc<Self> {
        Self::new_internal(
            layers,
            token,
            dispatcher,
            None,
            layer_slots,
            backend,
            None,
            target_dtype,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_internal(
        layers: Vec<usize>,
        token: usize,
        dispatcher: Arc<AsyncSwapDispatcher>,
        secondary: Option<Arc<SecondaryMmap>>,
        layer_slots: Vec<Arc<LayerSlot>>,
        backend: Arc<dyn Backend>,
        release_worker: Option<Arc<dyn ReleaseWorkerAccess>>,
        target_dtype: DType,
        config: Arc<crate::model_config::ModelConfig>,
    ) -> Arc<Self> {
        let num_layers = layer_slots.len();
        let pending_events: Vec<ArcSwapOption<GpuEvent>> =
            (0..num_layers).map(|_| ArcSwapOption::from(None)).collect();
        Arc::new_cyclic(|w| Self {
            plan: Mutex::new(IntraForwardSwapPlan::new(layers, token)),
            dispatcher,
            secondary,
            layer_slots,
            backend,
            release_worker,
            target_dtype,
            pending_events,
            stage_gate_armed: AtomicBool::new(false),
            finalized: AtomicBool::new(false),
            config,
            self_weak: w.clone(),
            _phantom: PhantomData,
        })
    }

    /// Lock-free snapshot of pending event for layer `idx`. Returns clone of
    /// the inner `Arc<GpuEvent>` if a swap is in flight, else `None`.
    ///
    /// Used by forward thread wait gate (ENG-ALG-238 / INV-149).
    #[inline]
    pub fn pending_event_for(&self, idx: usize) -> Option<Arc<GpuEvent>> {
        self.pending_events.get(idx).and_then(|s| s.load_full())
    }

    /// Plan progress probe. Used by decode loop after `forward_into` to
    /// decide whether to call `finalize`.
    pub fn plan_is_complete(&self) -> bool {
        self.plan.lock().map(|p| p.is_complete()).unwrap_or(false)
    }

    /// Number of layers still pending (diagnostics).
    pub fn pending_layer_count(&self) -> usize {
        self.plan
            .lock()
            .map(|p| p.pending_layers().count())
            .unwrap_or(0)
    }

    /// Plan run-to-completion finalize (INV-150).
    ///
    /// Order:
    /// 1. `dispatcher.drain(deadline)` — wait for all in-flight commits.
    /// 2. `backend.synchronize()` — full barrier.
    /// 3. `ratio_generation.fetch_add(1, SeqCst)` — bump if `stage_gate_armed`.
    /// 4. `soa_registry_invalidate()` — clear stale registry.
    ///
    /// Returns `Err` on drain timeout, backend sync error, or if already
    /// finalized (idempotency guard).
    pub fn finalize(
        self: &Arc<Self>,
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
                "[IntraForwardSwap] finalize called twice on same plan (INV-150 guard)"
            ));
        }

        // (1) drain dispatcher
        self.dispatcher
            .drain(deadline)
            .map_err(|e| anyhow!("[IntraForwardSwap] dispatcher drain failed: {e}"))?;

        // (2) backend full barrier — required so the next plan sees a fully
        // committed cl_mem state before noshuffle SOA registry rebuild.
        self.backend
            .synchronize()
            .map_err(|e| anyhow!("[IntraForwardSwap] backend.synchronize failed: {e}"))?;

        // (3) bump ratio_generation exactly once per plan, only if at least
        // one layer was actually dispatched.
        if self.stage_gate_armed.load(Ordering::Acquire) {
            ratio_generation.fetch_add(1, Ordering::SeqCst);
        }

        // (4) invalidate caller-supplied registry (e.g. noshuffle SOA).
        soa_registry_invalidate();

        Ok(())
    }

    /// Internal: store the real cl_event into `pending_events[idx]`. Called
    /// from `on_layer_boundary` before `submit_commit`. The same `Arc<GpuEvent>`
    /// is also passed to `SwapCommitJob::write_event` so the dispatcher worker
    /// and forward thread wait gate share one underlying cl_event.
    fn arm_pending(&self, idx: usize, event: Arc<GpuEvent>) {
        if let Some(slot) = self.pending_events.get(idx) {
            slot.store(Some(event));
        }
    }

    /// Internal: clear pending event for `idx`. Called by dispatcher worker
    /// thread via `SwapCommitJob::on_complete` after `slot.swap_weights`.
    fn clear_pending(&self, idx: usize) {
        if let Some(slot) = self.pending_events.get(idx) {
            slot.store(None);
        }
    }

    /// Test-only: arm with a caller-supplied sentinel for INV-149 ordering
    /// verification.
    #[doc(hidden)]
    pub fn arm_pending_for_test(&self, idx: usize, event: Arc<GpuEvent>) {
        if let Some(slot) = self.pending_events.get(idx) {
            slot.store(Some(event));
        }
    }

    /// Test-only: clear pending without going through dispatcher.
    #[doc(hidden)]
    pub fn clear_pending_for_test(&self, idx: usize) {
        self.clear_pending(idx);
    }

    /// Test-only: mark plan dispatched without going through real dispatch.
    #[doc(hidden)]
    pub fn mark_dispatched_for_test(&self, idx: usize) {
        if let Ok(mut p) = self.plan.lock() {
            p.mark_dispatched(idx);
        }
    }

    /// Test-only: arm stage gate so finalize bumps ratio_generation.
    #[doc(hidden)]
    pub fn arm_stage_gate_for_test(&self) {
        self.stage_gate_armed.store(true, Ordering::Release);
    }

    /// Dispatcher accessor (decode loop diagnostics).
    pub fn dispatcher(&self) -> &Arc<AsyncSwapDispatcher> {
        &self.dispatcher
    }
}

impl LayerBoundaryHook for IntraForwardSwapHook {
    #[inline]
    fn pending_event_for_dyn(&self, idx: usize) -> Option<Arc<GpuEvent>> {
        self.pending_event_for(idx)
    }

    fn on_layer_boundary(&self, idx: usize, seq_len: usize) {
        // Prefill swap is forbidden — cl_mem 폭증 risk + plan timing 불일치.
        if seq_len > 1 {
            return;
        }

        // Locked plan check (cheap when idx not in dispatch_at).
        let mut plan = match self.plan.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if !plan.should_dispatch(idx) {
            return;
        }

        let Some(slot) = self.layer_slots.get(idx) else {
            return;
        };
        if slot.secondary_mmap_handle().is_none() {
            // No swap path for this slot — drop silently.
            plan.mark_dispatched(idx);
            return;
        }
        // Test-only constructor (`new_for_test`) leaves `secondary` as None;
        // dispatch is silently disabled in that path.
        let Some(secondary) = self.secondary.as_ref() else {
            plan.mark_dispatched(idx);
            return;
        };

        // Reuse SwapExecutor::build_layer_from_mmap_async via the public
        // hook helper to materialise + enqueue_write_async.
        let memory = crate::memory::galloc::Galloc::new();
        let executor = match self.release_worker.as_ref() {
            Some(rw) => SwapExecutor::new_with_worker(
                self.target_dtype,
                self.config.as_ref(),
                Arc::clone(&self.backend),
                &memory,
                Arc::clone(rw),
            ),
            None => SwapExecutor::new(
                self.target_dtype,
                self.config.as_ref(),
                Arc::clone(&self.backend),
                &memory,
            ),
        };

        let async_build =
            executor.build_layer_from_mmap_async_for_hook(secondary, slot.as_ref(), idx);
        let (new_layer, write_event) = match async_build {
            Ok(p) => p,
            Err(e) => {
                crate::action_diag_helper::log_swap_failed(
                    "IntraForward",
                    &format!(
                        "async build failed: {e}; dropping (plan continues without this layer)"
                    ),
                    Some(idx),
                    None,
                );
                plan.mark_dispatched(idx);
                return;
            }
        };
        let new_arc: Arc<LayerWeights> = Arc::new(new_layer);

        // Wrap the real cl_event in Arc and share between the forward-thread
        // wait gate (pending_events) and the dispatcher worker
        // (SwapCommitJob.write_event). Both threads call
        // Backend::wait_event_blocking on the same underlying cl_event;
        // clWaitForEvents is thread-safe per OpenCL spec.
        let write_event_arc: Arc<GpuEvent> = Arc::new(write_event);

        // INV-149: arm pending_events BEFORE submit_commit so forward thread
        // cannot observe a stale `None` between submission and clear.
        self.arm_pending(idx, Arc::clone(&write_event_arc));

        // dispatcher worker callback: clear pending after commit. Hand the
        // worker a Weak<Self> so it does not extend the hook's lifetime past
        // finalize/retire.
        let weak_self: Weak<IntraForwardSwapHook> = self.self_weak.clone();
        let on_complete: Arc<dyn Fn(usize) + Send + Sync> = {
            let w = weak_self;
            Arc::new(move |layer_idx: usize| {
                if let Some(h) = w.upgrade() {
                    h.clear_pending(layer_idx);
                }
            })
        };

        let job = SwapCommitJob {
            slot: Arc::clone(slot),
            new_weights: new_arc,
            new_dtype: self.target_dtype,
            write_event: write_event_arc,
            release_worker: self.release_worker.clone(),
            on_complete: Some(on_complete),
            layer_idx: Some(idx),
        };

        if let Err(e) = self.dispatcher.submit_commit(job) {
            crate::action_diag_helper::log_swap_failed(
                "IntraForward",
                &format!("dispatcher submit failed: {e}; clearing pending and skipping"),
                Some(idx),
                None,
            );
            self.clear_pending(idx);
            plan.mark_dispatched(idx);
            return;
        }

        plan.mark_dispatched(idx);
        self.stage_gate_armed.store(true, Ordering::Release);
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // INV-148: plan dispatch idempotency.
    #[test]
    fn test_plan_should_dispatch_then_mark() {
        let mut plan = IntraForwardSwapPlan::new(vec![3, 5, 7], 0);
        assert!(plan.should_dispatch(3));
        plan.mark_dispatched(3);
        assert!(!plan.should_dispatch(3));
        assert!(plan.should_dispatch(5));
    }

    #[test]
    fn test_plan_double_mark_safe() {
        let mut plan = IntraForwardSwapPlan::new(vec![3], 0);
        plan.mark_dispatched(3);
        plan.mark_dispatched(3);
        assert!(!plan.should_dispatch(3));
        assert!(plan.is_complete());
    }

    #[test]
    fn test_plan_out_of_set() {
        let plan = IntraForwardSwapPlan::new(vec![3], 0);
        assert!(!plan.should_dispatch(0));
        assert!(!plan.should_dispatch(99));
    }

    #[test]
    fn test_plan_complete_after_all_marked() {
        let mut plan = IntraForwardSwapPlan::new(vec![3, 5, 7], 0);
        for i in [3, 5, 7] {
            plan.mark_dispatched(i);
        }
        assert!(plan.is_complete());
        assert_eq!(plan.pending_layers().count(), 0);
    }

    #[test]
    fn test_plan_empty_is_complete() {
        let plan = IntraForwardSwapPlan::new(vec![], 0);
        assert!(plan.is_complete());
    }

    #[test]
    fn test_plan_pending_layers_diff() {
        let mut plan = IntraForwardSwapPlan::new(vec![1, 2, 3], 0);
        plan.mark_dispatched(2);
        let pending: Vec<usize> = plan.pending_layers().collect();
        assert_eq!(pending, vec![1, 3]);
    }
}
