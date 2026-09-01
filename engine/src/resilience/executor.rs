use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use argus_shared::{
    CommandResponse, CommandResult, EngineCommand, EngineMessage, EngineState, EngineStatus,
    ManagerMessage, Phase,
};

// ── Public types ────────────────────────────────────────────

/// KV cache state for the heartbeat, gathered by the caller from the handles it holds.
#[derive(Debug, Clone, Default)]
pub struct KVSnapshot {
    /// Bytes the cache actually occupies right now, at its real dtype.
    pub total_bytes: u64,
    /// Bytes the cache would occupy at full capacity, uncompressed — the denominator of
    /// a `KvCompress` budget.
    pub budget_bytes: u64,
    /// Resident tokens.
    pub total_tokens: usize,
}

// ── CommandExecutor ─────────────────────────────────────────

/// Receives `ManagerMessage`s and drains them into `EngineCommand`s for the inference
/// loop, and emits the heartbeat. No policy: applying a command is the
/// `CommandDispatcher`'s job.
pub struct CommandExecutor {
    cmd_rx: mpsc::Receiver<ManagerMessage>,
    resp_tx: mpsc::Sender<EngineMessage>,

    engine_state: EngineState,
    phase: Phase,

    /// Smoothed time between tokens, in ms. Held in the unit the heartbeat reports so
    /// there is no reciprocal to take at the edge where no token has been produced yet.
    tbt_ms_ema: f32,
    last_token_time: Option<Instant>,

    last_heartbeat: Instant,
    heartbeat_interval: Duration,

    /// Directives drained but not yet answered, as `(seq_id, command_count)` in arrival
    /// order. `drain_commands` pushes; `report_results` pops and emits one `Response` per
    /// entry. The queue exists because the result of a command is only known after the
    /// dispatcher has applied it and the stage it submitted has run, which is two call
    /// sites later than the drain.
    pending_responses: VecDeque<(u64, usize)>,
}

impl CommandExecutor {
    pub fn new(
        cmd_rx: mpsc::Receiver<ManagerMessage>,
        resp_tx: mpsc::Sender<EngineMessage>,
        heartbeat_interval: Duration,
    ) -> Self {
        Self {
            cmd_rx,
            resp_tx,
            engine_state: EngineState::Idle,
            phase: Phase::Idle,
            tbt_ms_ema: 0.0,
            last_token_time: None,
            last_heartbeat: Instant::now(),
            heartbeat_interval,
            pending_responses: VecDeque::new(),
        }
    }

    /// Clone the engine to manager channel. Held by anything that reports outside the
    /// command path.
    pub fn report_sender(&self) -> mpsc::Sender<EngineMessage> {
        self.resp_tx.clone()
    }

    /// Record a generated token, updating the smoothed time between tokens.
    pub fn on_token_generated(&mut self) {
        let now = Instant::now();
        if let Some(last) = self.last_token_time {
            let elapsed_ms = now.duration_since(last).as_secs_f32() * 1000.0;
            if elapsed_ms > 0.0 {
                const ALPHA: f32 = 0.1;
                self.tbt_ms_ema = if self.tbt_ms_ema == 0.0 {
                    elapsed_ms
                } else {
                    ALPHA * elapsed_ms + (1.0 - ALPHA) * self.tbt_ms_ema
                };
            }
        }
        self.last_token_time = Some(now);
    }

    /// Set the inference phase and operational state the heartbeat carries.
    ///
    /// Stamped from two places, each of which *is* the engine being in that phase:
    /// `PrefillPhaseStage` on the driver's prefill boundaries, and the command poll on
    /// every decode step. Nothing schedules a phase in advance, so the value a heartbeat
    /// carries is always one the engine has actually reached.
    pub fn set_phase(&mut self, phase: Phase, state: EngineState) {
        self.phase = phase;
        self.engine_state = state;
    }

    /// Emit one heartbeat if the interval has elapsed. The command source calls this
    /// right before `drain_commands` so emission stays on the live poll path; `kv_snap`
    /// is built by the source from the handles it holds.
    pub fn send_heartbeat_if_due(&mut self, kv_snap: &KVSnapshot) {
        if self.last_heartbeat.elapsed() >= self.heartbeat_interval {
            self.send_heartbeat(kv_snap);
            self.last_heartbeat = Instant::now();
        }
    }

    /// Emit a heartbeat now, whatever the interval says, and restart the interval.
    ///
    /// For a change worth reporting on its own rather than on a clock: entering prefill,
    /// and leaving it having grown the cache by an entire prompt. Restarting the interval
    /// is what keeps a forced heartbeat from being chased by a due one a millisecond
    /// later — the caller gets one report per event, not two.
    pub fn send_heartbeat_now(&mut self, kv_snap: &KVSnapshot) {
        self.send_heartbeat(kv_snap);
        self.last_heartbeat = Instant::now();
    }

    /// Drain arrived manager commands and return them flattened, in arrival order.
    ///
    /// **The response is not sent here.** Whether a command was applied is only
    /// known after `CommandDispatcher` has run, one call site later, so each
    /// directive is recorded in `pending_responses` and answered by
    /// [`Self::report_results`]. Emitting `Ok` here — as this did until the
    /// `Rejected` semantics landed — reported success for commands the dispatcher
    /// silently ignored (an unconfigured cache manager, a removed subsystem),
    /// which is exactly the signal the Manager needs to learn the engine's action
    /// set. Heartbeat emission is separate ([`Self::send_heartbeat_if_due`]).
    pub fn drain_commands(&mut self) -> Vec<EngineCommand> {
        let mut commands = Vec::new();
        while let Ok(msg) = self.cmd_rx.try_recv() {
            match msg {
                ManagerMessage::Directive(d) => {
                    let seq_id = d.seq_id;
                    for cmd in &d.commands {
                        eprintln!("[Resilience] Directive seq={}: {:?}", seq_id, cmd);
                    }
                    self.pending_responses.push_back((seq_id, d.commands.len()));
                    commands.extend(d.commands);
                }
            }
        }
        commands
    }

    /// Answer every directive drained since the last call, splitting `results`
    /// back across the directives it came from.
    ///
    /// `results` must be the per-command outcomes of the commands the matching
    /// [`Self::drain_commands`] returned, in the same order. The queue is always
    /// emptied: a caller that supplies too few results has, by definition, not
    /// applied the remainder, so those commands are answered `Rejected` rather
    /// than left unanswered — the contract is one `Response` per `Directive`.
    pub fn report_results(&mut self, results: Vec<CommandResult>) {
        let mut it = results.into_iter();
        while let Some((seq_id, n)) = self.pending_responses.pop_front() {
            let mut batch = Vec::with_capacity(n);
            for _ in 0..n {
                batch.push(it.next().unwrap_or_else(|| CommandResult::Rejected {
                    reason: "engine did not apply the directive".to_string(),
                }));
            }
            let _ = self.resp_tx.send(EngineMessage::Response(CommandResponse {
                seq_id,
                results: batch,
            }));
        }
    }

    fn send_heartbeat(&mut self, kv_snap: &KVSnapshot) {
        let status = EngineStatus {
            kv_cache_bytes: kv_snap.total_bytes,
            kv_cache_budget_bytes: kv_snap.budget_bytes,
            kv_cache_tokens: kv_snap.total_tokens,
            tbt_ms: self.tbt_ms_ema,
            phase: self.phase,
            state: self.engine_state,
        };
        let _ = self.resp_tx.send(EngineMessage::Heartbeat(status));
    }
}
