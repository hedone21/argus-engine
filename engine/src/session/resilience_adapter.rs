//! [`ResilienceAdapter`] — connects [`CommandExecutor`] to the session's command seam.
//!
//! The session pipeline takes a [`CommandSource`]; the executor offers `poll`-shaped
//! drain plus heartbeat emission, and this adapter is the join. Per-token ticks come from
//! `TickStage` (PostSample, `stages/system/tick.rs`) calling [`ResilienceAdapter::tick`]
//! through a shared `Arc`.
//!
//! `DecodeLoopBuilder::with_resilience` wraps this in `Arc<Mutex<_>>` and injects the
//! newtype wrapper into the command-source slot.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use argus_shared::{CommandResult, EngineCommand, EngineMessage, EngineState, Phase};

use crate::format::KVCacheFormat;
use crate::resilience::{CommandExecutor, KVSnapshot};
use crate::session::command_dispatcher::CommandSource;
use crate::session::forward::PrefillProgress;

/// Narrow neutral seam for a KV format's live resident byte count.
///
/// The base [`KVCacheFormat`] surface deliberately has no byte accessor
/// (`INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC` — no base-trait downcast), so the concrete
/// format implements this instead and the assembly injects it. Implementations delegate
/// to the caches' own `memory_usage_bytes()`, which is dtype-aware: it sizes Q4_0 by
/// block, asks an opaque descriptor for `bytes_for_elems`, and otherwise multiplies by
/// the buffer's real dtype size.
///
/// That is what makes the heartbeat's `kv_cache_bytes` an actual byte count rather than a
/// token count wearing a constant — the distinction the KV budget rests on, since a
/// figure derived from fixed geometry would cancel against the geometry-derived
/// denominator and leave a byte ratio numerically identical to a token ratio.
pub trait KvBytesHandle: Send + Sync {
    /// Bytes this layer's K and V currently occupy for the resident tokens.
    fn resident_bytes(&self) -> u64;
}

/// Adapts [`CommandExecutor`] to the session's [`CommandSource`] slot.
///
/// `poll` is pure — it returns the drained [`EngineCommand`]s — but heartbeat emission
/// lives inside it, because reaching it is what tells the adapter a decode step is
/// happening. The heartbeat payload is built from handles injected at register time
/// rather than passed down through the step context.
pub struct ResilienceAdapter {
    executor: CommandExecutor,
    /// Layer-0 handle, for the resident token count and the capacity the budget
    /// denominator is computed from. `None` leaves the snapshot empty.
    kv_handle: Option<Arc<dyn KVCacheFormat>>,
    /// Per-layer resident-byte probes, summed for `kv_cache_bytes`. Empty reports 0.
    kv_byte_handles: Vec<Arc<dyn KvBytesHandle>>,
    /// Bytes one token occupies across all decoder layers **uncompressed**. Times the
    /// cache capacity this is `kv_cache_budget_bytes`, the denominator a `KvCompress`
    /// budget is a fraction of. Geometry is the right basis here and only here: the
    /// question it answers is what the cache *would* cost without compression.
    uncompressed_bytes_per_token: usize,
}

impl ResilienceAdapter {
    pub fn new(executor: CommandExecutor) -> Self {
        Self {
            executor,
            kv_handle: None,
            kv_byte_handles: Vec::new(),
            uncompressed_bytes_per_token: 0,
        }
    }

    /// Layer-0 handle for the heartbeat's token count and capacity.
    pub fn set_kv_handle(&mut self, handle: Arc<dyn KVCacheFormat>) {
        self.kv_handle = Some(handle);
    }

    /// Per-layer resident-byte probes. Pass every decoder layer — the heartbeat reports
    /// the whole-model figure.
    pub fn set_kv_byte_handles(&mut self, handles: Vec<Arc<dyn KvBytesHandle>>) {
        self.kv_byte_handles = handles;
    }

    /// Whole-model uncompressed bytes per token, for the budget denominator. See
    /// [`crate::session::resilience_init::uncompressed_kv_bytes_per_token`].
    pub fn set_uncompressed_bytes_per_token(&mut self, bytes: usize) {
        self.uncompressed_bytes_per_token = bytes;
    }

    /// Direct access for callers that configure the executor itself.
    pub fn executor_mut(&mut self) -> &mut CommandExecutor {
        &mut self.executor
    }

    /// Clone the engine to manager channel.
    pub fn report_sender(&self) -> std::sync::mpsc::Sender<EngineMessage> {
        self.executor.report_sender()
    }

    /// Per-token tick, called by `TickStage` (PostSample) for every sampled token. Feeds
    /// the heartbeat's smoothed time-between-tokens.
    pub fn tick(&mut self) {
        self.executor.on_token_generated();
    }

    /// Whole-model resident KV bytes, summed over the injected per-layer probes.
    fn resident_kv_bytes(&self) -> u64 {
        self.kv_byte_handles
            .iter()
            .map(|h| h.resident_bytes())
            .sum()
    }

    /// Build the heartbeat's KV payload from the held handles.
    ///
    /// `total_bytes` is what the cache actually occupies, at its real dtype;
    /// `budget_bytes` is what it would occupy full and uncompressed. The two are
    /// deliberately computed differently — a ratio of two geometry figures would cancel
    /// to a token ratio and tell the Manager nothing about compression.
    fn build_kv_snapshot(&self) -> KVSnapshot {
        match &self.kv_handle {
            Some(h) => KVSnapshot {
                total_tokens: h.current_pos(),
                total_bytes: self.resident_kv_bytes(),
                budget_bytes: (h.capacity() as u64)
                    .saturating_mul(self.uncompressed_bytes_per_token as u64),
            },
            None => KVSnapshot::default(),
        }
    }

    /// Prefill has begun: stamp the phase and report it immediately.
    ///
    /// Forced rather than interval-gated because the transition is itself the news. A
    /// Manager that knows the engine entered prefill can read the quiet that follows as
    /// work rather than as a stall — which is the whole reason [`Phase`] is on the wire,
    /// and something no amount of utilization tells it.
    pub fn enter_prefill(&mut self) {
        self.executor
            .set_phase(Phase::Prefill, EngineState::Running);
        let kv_snap = self.build_kv_snapshot();
        self.executor.send_heartbeat_now(&kv_snap);
    }

    /// One prefill chunk landed. Interval-gated: a prompt short enough to finish inside
    /// one heartbeat period should not turn into a burst, and the boundaries around this
    /// are reported unconditionally anyway.
    pub fn prefill_chunk(&mut self) {
        let kv_snap = self.build_kv_snapshot();
        self.executor.send_heartbeat_if_due(&kv_snap);
    }

    /// Prefill finished. Forced, for the same reason as [`Self::enter_prefill`] and one
    /// more: the cache just grew by the entire prompt, the largest single change in a
    /// run, and the next report would otherwise wait on a decode step.
    ///
    /// The phase deliberately stays `Prefill` — this heartbeat describes the instant
    /// prefill ended, and the first decode poll stamps `Decode` a moment later.
    pub fn leave_prefill(&mut self) {
        let kv_snap = self.build_kv_snapshot();
        self.executor.send_heartbeat_now(&kv_snap);
    }
}

/// Lets a chunked forward report prefill progress into the adapter it shares with the
/// decode loop. The lock is uncontended here: the driver takes it inside a stage dispatch
/// or a poll, never across `Forward::prefill`.
impl PrefillProgress for Mutex<ResilienceAdapter> {
    fn on_prefill_chunk(&self) {
        self.lock()
            .expect("resilience mutex poisoned")
            .prefill_chunk();
    }
}

impl CommandSource for ResilienceAdapter {
    fn poll(&mut self) -> Result<Vec<EngineCommand>> {
        // Reaching this IS the engine being in decode: it is called once per decode step
        // and nowhere else, which is why the phase stamp lives here rather than in a
        // separate hook. Prefill stamps itself the same way, from `PrefillPhaseStage`.
        self.executor.set_phase(Phase::Decode, EngineState::Running);
        let kv_snap = self.build_kv_snapshot();
        self.executor.send_heartbeat_if_due(&kv_snap);
        Ok(self.executor.drain_commands())
    }

    fn report_results(&mut self, results: Vec<CommandResult>) {
        self.executor.report_results(results);
    }
}

/// Exposes `Arc<Mutex<ResilienceAdapter>>` as a [`CommandSource`].
///
/// The builder wraps a single adapter so the command-source slot and `TickStage` can
/// share it. Per-step contention is two short locks (one poll, one tick).
pub(crate) struct CmdSrcWrapper(pub Arc<Mutex<ResilienceAdapter>>);

impl CommandSource for CmdSrcWrapper {
    fn poll(&mut self) -> Result<Vec<EngineCommand>> {
        self.0.lock().expect("resilience mutex poisoned").poll()
    }

    fn report_results(&mut self, results: Vec<CommandResult>) {
        self.0
            .lock()
            .expect("resilience mutex poisoned")
            .report_results(results);
    }
}
