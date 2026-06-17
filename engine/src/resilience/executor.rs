use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use argus_shared::{
    CommandResponse, CommandResult, EngineCapability, EngineCommand, EngineMessage, EngineState,
    EngineStatus, ManagerMessage, QcfEstimate, ResourceLevel,
};

use crate::resilience::gpu_self_meter::GpuSelfMeter;
use crate::resilience::proc_self_meter::ProcSelfMeter;

// ── Public types ────────────────────────────────────────────

/// Snapshot of KV cache state for status reporting.
#[derive(Debug, Clone, Default)]
pub struct KVSnapshot {
    pub total_bytes: u64,
    pub total_tokens: usize,
    pub capacity: usize,
    pub protected_prefix: usize,
    /// Current KV dtype name for heartbeat reporting ("f16", "q8", "q4").
    pub kv_dtype: String,
    /// Current eviction policy name for heartbeat reporting.
    pub eviction_policy: String,
    /// Current layer skip ratio for heartbeat reporting.
    pub skip_ratio: f32,
}

// ── CommandExecutor ─────────────────────────────────────────

/// Receives ManagerMessages and drains them into `EngineCommand`s for the
/// inference loop. No strategy logic — the `CommandDispatcher` applies them.
pub struct CommandExecutor {
    cmd_rx: mpsc::Receiver<ManagerMessage>,
    resp_tx: mpsc::Sender<EngineMessage>,

    // Current state (deprecated fields kept for EngineStatus backward compat)
    compute_level: ResourceLevel,
    memory_level: ResourceLevel,
    engine_state: EngineState,
    active_device: String,

    // Currently active action names (e.g. "kv.evict_h2o", "throttle")
    active_actions: Vec<String>,

    // Tensor partition ratio for heartbeat reporting
    partition_ratio: f32,

    // Prefill progress reporting
    phase: String,
    prefill_pos: usize,
    prefill_total: usize,

    // Throughput tracking
    throughput_ema: f32,
    last_token_time: Option<Instant>,
    tokens_generated: usize,

    // Heartbeat
    last_heartbeat: Instant,
    heartbeat_interval: Duration,

    // Engine self-util (MSG-067): /proc/self/stat 기반 자가 CPU 사용률 measurer.
    proc_meter: ProcSelfMeter,

    // Engine self-util (MSG-068, Phase 2): OpenCL profiling 기반 자가 GPU
    // 사용률 measurer. None이면 Phase 1 호환 동작 (self_gpu_pct=0.0).
    gpu_meter: Option<Arc<dyn GpuSelfMeter>>,
    // 직전 heartbeat 송출 시각. gpu_meter의 wall_elapsed 계산에 사용한다.
    // 첫 샘플은 new() 시각을 기준으로 하여 warm-up 구간을 자연스럽게 흡수.
    last_heartbeat_at: Instant,

    // secondary GGUF/AUF 파일 존재 여부. true이면 swap_weights 액션이
    // available_actions에 포함된다 (ENG-ST-032).
    has_secondary: bool,
}

impl CommandExecutor {
    pub fn new(
        cmd_rx: mpsc::Receiver<ManagerMessage>,
        resp_tx: mpsc::Sender<EngineMessage>,
        active_device: String,
        heartbeat_interval: Duration,
    ) -> Self {
        Self::with_gpu_meter(cmd_rx, resp_tx, active_device, heartbeat_interval, None)
    }

    /// MSG-068 / MGR-DAT-076 Phase 2: GPU self-utilization meter를 주입할 수
    /// 있는 확장 생성자. `gpu_meter`가 `Some`이면 heartbeat 송출 시 meter를
    /// 샘플링하여 `self_gpu_pct`에 실어 보낸다. `None`이면 Phase 1 호환
    /// (항상 0.0).
    pub fn with_gpu_meter(
        cmd_rx: mpsc::Receiver<ManagerMessage>,
        resp_tx: mpsc::Sender<EngineMessage>,
        active_device: String,
        heartbeat_interval: Duration,
        gpu_meter: Option<Arc<dyn GpuSelfMeter>>,
    ) -> Self {
        let now = Instant::now();
        Self {
            cmd_rx,
            resp_tx,
            compute_level: ResourceLevel::Normal,
            memory_level: ResourceLevel::Normal,
            engine_state: EngineState::Idle,
            active_device,
            active_actions: Vec::new(),
            partition_ratio: 0.0,
            phase: String::new(),
            prefill_pos: 0,
            prefill_total: 0,
            throughput_ema: 0.0,
            last_token_time: None,
            tokens_generated: 0,
            last_heartbeat: now,
            heartbeat_interval,
            proc_meter: ProcSelfMeter::new(),
            gpu_meter,
            last_heartbeat_at: now,
            has_secondary: false,
        }
    }

    /// secondary GGUF/AUF 경로 존재 여부를 설정한다.
    ///
    /// `true`이면 이후 Heartbeat의 `available_actions`에 `"swap_weights"`가 포함된다.
    /// generate.rs에서 모델 로드 직후, Capability 송출 전에 호출한다 (ENG-ST-032).
    pub fn set_has_secondary(&mut self, has_secondary: bool) {
        self.has_secondary = has_secondary;
    }

    /// Send initial capability report to Manager.
    pub fn send_capability(&self, cap: EngineCapability) {
        let _ = self.resp_tx.send(EngineMessage::Capability(cap));
    }

    /// Send QCF estimate to Manager (SEQ-096).
    pub fn send_qcf_estimate(&self, qcf: QcfEstimate) {
        let _ = self.resp_tx.send(EngineMessage::QcfEstimate(qcf));
    }

    /// Send weight swap completion report to Manager (MSG-089).
    ///
    /// Called by the generate.rs dispatch handler after a successful
    /// `EngineCommand::SwapWeights` execution (ENG-ALG-214-ROUTE).
    pub fn send_weight_swap_report(&self, report: argus_shared::WeightSwapReport) {
        let _ = self.resp_tx.send(EngineMessage::WeightSwapReport(report));
    }

    /// AB-6 §5.6.6: clone the engine→manager response channel so a
    /// `WeightSwapStage`(`&self`) can send `WeightSwapReport` directly at commit
    /// time without holding a `&mut` adapter. `EngineSwapRuntime::report_tx` 가
    /// 이 clone 을 보유한다.
    pub fn report_sender(&self) -> std::sync::mpsc::Sender<EngineMessage> {
        self.resp_tx.clone()
    }

    /// Record a generated token for throughput tracking.
    pub fn on_token_generated(&mut self) {
        let now = Instant::now();
        self.tokens_generated += 1;

        if let Some(last) = self.last_token_time {
            let elapsed = now.duration_since(last).as_secs_f32();
            if elapsed > 0.0 {
                let instant_tps = 1.0 / elapsed;
                const ALPHA: f32 = 0.1;
                if self.throughput_ema == 0.0 {
                    self.throughput_ema = instant_tps;
                } else {
                    self.throughput_ema = ALPHA * instant_tps + (1.0 - ALPHA) * self.throughput_ema;
                }
            }
        }
        self.last_token_time = Some(now);
    }

    /// Emit one heartbeat if the interval has elapsed (interval check +
    /// `send_heartbeat` + `last_heartbeat` 갱신). The command source calls this
    /// right before `drain_commands` so heartbeat emission stays on the live
    /// poll path. `kv_snap` is built by the source from its held handle.
    pub fn send_heartbeat_if_due(&mut self, kv_snap: &KVSnapshot) {
        if self.last_heartbeat.elapsed() >= self.heartbeat_interval {
            self.send_heartbeat(kv_snap);
            self.last_heartbeat = Instant::now();
        }
    }

    /// Drain arrived manager commands and return them (pure production).
    ///
    /// Flattens each directive's commands in order and, for every command,
    /// immediately sends a `CommandResult::Ok` response (the executor never
    /// rejects). Command application (eviction etc.) is the `CommandDispatcher`'s
    /// job; heartbeat emission is separate ([`Self::send_heartbeat_if_due`]).
    pub fn drain_commands(&mut self) -> Vec<EngineCommand> {
        let mut commands = Vec::new();
        while let Ok(msg) = self.cmd_rx.try_recv() {
            match msg {
                ManagerMessage::Directive(d) => {
                    let seq_id = d.seq_id;
                    let mut results = Vec::with_capacity(d.commands.len());
                    for cmd in &d.commands {
                        eprintln!("[Resilience] Directive seq={}: {:?}", seq_id, cmd);
                        results.push(CommandResult::Ok);
                    }
                    let _ = self
                        .resp_tx
                        .send(EngineMessage::Response(CommandResponse { seq_id, results }));
                    commands.extend(d.commands);
                }
            }
        }
        commands
    }

    fn send_heartbeat(&mut self, kv_snap: &KVSnapshot) {
        let utilization = if kv_snap.capacity > 0 {
            kv_snap.total_tokens as f32 / kv_snap.capacity as f32
        } else {
            0.0
        };

        let memory_lossy_min = if kv_snap.total_tokens > 0 {
            (kv_snap.protected_prefix as f32 / kv_snap.total_tokens as f32).max(0.01)
        } else {
            0.01
        };

        let eviction_policy = if kv_snap.eviction_policy.is_empty() {
            "none".to_string()
        } else {
            kv_snap.eviction_policy.clone()
        };

        let kv_dtype = if kv_snap.kv_dtype.is_empty() {
            "f16".to_string()
        } else {
            kv_snap.kv_dtype.clone()
        };

        let status = EngineStatus {
            active_device: self.active_device.clone(),
            compute_level: self.compute_level,
            actual_throughput: self.throughput_ema,
            memory_level: self.memory_level,
            kv_cache_bytes: kv_snap.total_bytes,
            kv_cache_tokens: kv_snap.total_tokens,
            kv_cache_utilization: utilization,
            memory_lossless_min: 1.0, // 현재 무손실 축소 불가
            memory_lossy_min,
            state: self.engine_state,
            tokens_generated: self.tokens_generated,
            available_actions: Self::compute_available_actions(
                &eviction_policy,
                &kv_dtype,
                self.has_secondary,
            ),
            active_actions: self.active_actions.clone(),
            eviction_policy,
            kv_dtype,
            skip_ratio: kv_snap.skip_ratio,
            phase: self.phase.clone(),
            prefill_pos: self.prefill_pos,
            prefill_total: self.prefill_total,
            partition_ratio: self.partition_ratio,
            // MSG-067: /proc/self/stat 기반 자가 CPU 사용률. 측정 실패 시 0.0 (INV-092).
            self_cpu_pct: self.proc_meter.sample(),
            // MSG-068 / MGR-DAT-076 Phase 2: OpenCL profiling 기반 자가 GPU
            // 사용률. meter 미주입 시(기본값) 0.0을 유지하여 Phase 1 호환
            // (INV-092). INV-091 clamp는 meter 구현 내에서 보장된다.
            self_gpu_pct: self
                .gpu_meter
                .as_ref()
                .map(|m| {
                    let elapsed = self.last_heartbeat_at.elapsed();
                    m.sample(elapsed).clamp(0.0, 1.0)
                })
                .unwrap_or(0.0),
        };

        // heartbeat 송출 직후 기준점 갱신. GPU meter의 다음 wall_elapsed 창을
        // 여기서 시작하여 heartbeat 간격과 정확히 정렬한다.
        self.last_heartbeat_at = Instant::now();

        let _ = self.resp_tx.send(EngineMessage::Heartbeat(status));
    }

    /// Compute available actions based on engine capabilities.
    pub(crate) fn compute_available_actions(
        eviction_policy: &str,
        kv_dtype: &str,
        has_secondary: bool,
    ) -> Vec<String> {
        let mut actions = vec![
            "throttle".to_string(),
            "switch_hw".to_string(),
            "weight.skip".to_string(),
        ];
        // Eviction actions: only if an eviction policy is configured
        if eviction_policy != "none" {
            actions.push("kv.evict_h2o".to_string());
            actions.push("kv.evict_sliding".to_string());
            actions.push("kv.evict_streaming".to_string());
            actions.push("kv.merge_d2o".to_string());
        }
        // KV quantization: only available with KIVI cache (q2/q4/q8)
        if kv_dtype.starts_with('q') {
            actions.push("kv.quant_dynamic".to_string());
        }
        // Weight swap: only available when a secondary GGUF/AUF is loaded (ENG-ST-032).
        if has_secondary {
            actions.push("swap_weights".to_string());
        }
        actions
    }

    /// Update tensor partition ratio for heartbeat reporting.
    pub fn set_partition_ratio(&mut self, ratio: f32) {
        self.partition_ratio = ratio;
    }
}
