use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use argus_shared::{
    CommandResponse, CommandResult, EngineCapability, EngineCommand, EngineMessage, EngineState,
    EngineStatus, ManagerMessage, QcfEstimate, ResourceLevel,
};

// LAYER-EXEMPT: cross_cutting_trait_usage — §13.8-N §F enum-as-data identifier (V-10)
use crate::kv::eviction::EvictMethod;
use crate::resilience::gpu_self_meter::GpuSelfMeter;
use crate::resilience::proc_self_meter::ProcSelfMeter;

// ── Public types ────────────────────────────────────────────

/// Plan produced by CommandExecutor::poll() for the inference loop to execute.
#[derive(Debug, Default)]
pub struct ExecutionPlan {
    /// KV cache eviction plan, if any.
    pub evict: Option<EvictPlan>,
    /// Device to switch to (e.g., "cpu", "gpu").
    pub switch_device: Option<String>,
    /// Device to pre-warm.
    pub prepare_device: Option<String>,
    /// Throttle delay between tokens (ms). 0 = no throttle.
    pub throttle_delay_ms: u64,
    /// Target TBT in ms. Engine sleeps max(0, target - actual_tbt) per token.
    /// 0 = disabled. Set via SetTargetTbt command.
    pub target_tbt_ms: u64,
    /// True iff `target_tbt_ms` was explicitly set by a SetTargetTbt directive
    /// received at least once. Lets the engine distinguish "manager explicitly
    /// said 0 (disable pacing)" from "manager never sent a directive, use CLI
    /// default". Cleared by RestoreDefaults.
    pub target_tbt_set: bool,
    /// Whether inference should be suspended.
    pub suspended: bool,
    /// Whether inference should resume from suspension.
    pub resumed: bool,
    /// Layer skip ratio: 0.0 = no skip, from LayerSkip command.
    pub layer_skip: Option<f32>,
    /// KV quantization bits, from KvQuantDynamic command.
    pub kv_quant_bits: Option<u8>,
    /// KV cache offload ratio, from KvOffload command (0.0~1.0).
    /// Consumed by the decode loop: when `Some(r)`, the engine calls
    /// `CacheManager::offload(r)` and clears this field.
    pub offload_ratio: Option<f32>,
    /// True when a RestoreDefaults directive arrives and offloaded KV data
    /// should be recalled from swap back into the cache.
    pub recall_offload: bool,
    /// Whether to restore all action-induced state to defaults.
    pub restore_defaults: bool,
    /// Whether Engine should compute and send QCF estimates.
    pub request_qcf: bool,
    /// Tensor partition ratio (0.0~1.0), set via SetPartitionRatio command.
    pub partition_ratio: Option<f32>,
    /// Prefill policy update (partial — None fields mean "keep current").
    pub prefill_chunk_size: Option<usize>,
    pub prefill_yield_ms: Option<u32>,
    pub prefill_cpu_chunk_size: Option<usize>,
    /// Weight swap request from a SwapWeights command (ENG-ALG-214-ROUTE).
    /// `Some((ratio, target_dtype))` means generate.rs should call
    /// `WeightSwapDecider` + `SwapExecutor`.  `None` = no pending swap.
    pub swap_weights: Option<(f32, argus_shared::DtypeTag)>,
}

/// StreamingLLM eviction parameters (sink + window).
#[derive(Debug, Clone)]
pub struct StreamingParams {
    pub sink_size: usize,
    pub window_size: usize,
}

/// Eviction plan parameters.
#[derive(Debug, Clone)]
pub struct EvictPlan {
    /// Target ratio of cache to keep (0.0–1.0).
    pub target_ratio: f32,
    /// Resource level: Warning = lossless only, Critical = lossy OK.
    pub level: ResourceLevel,
    /// Which eviction algorithm to use.
    pub method: EvictMethod,
    /// StreamingLLM-specific parameters (only set when method == Streaming).
    pub streaming_params: Option<StreamingParams>,
}

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

/// Receives ManagerMessages, builds ExecutionPlans for the inference loop.
/// No strategy logic — just translates commands to plans.
pub struct CommandExecutor {
    cmd_rx: mpsc::Receiver<ManagerMessage>,
    resp_tx: mpsc::Sender<EngineMessage>,

    // Current state (deprecated fields kept for EngineStatus backward compat)
    compute_level: ResourceLevel,
    memory_level: ResourceLevel,
    engine_state: EngineState,
    active_device: String,
    throttle_delay_ms: u64,
    target_tbt_ms: u64,
    target_tbt_set: bool,

    // Currently active action names (e.g. "kv.evict_h2o", "throttle")
    active_actions: Vec<String>,

    // Sticky eviction plan: retained until RestoreDefaults
    evict_plan: Option<EvictPlan>,
    // Sticky KV quantization bits: retained until RestoreDefaults
    kv_quant_bits: Option<u8>,
    // Sticky tensor partition ratio: retained until RestoreDefaults.
    // Ensures SetPartitionRatio directives received during prefill are
    // visible in subsequent decode-loop polls (ISSUE-5 fix).
    partition_ratio_sticky: Option<f32>,

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

    // Sticky pending SwapWeights directive (LISWAP-6 manager routing fix).
    //
    // `apply_command::SwapWeights` 가 transient `ExecutionPlan.swap_weights`
    // 에만 기록하면, prefill loop 의 `executor.poll` 이 plan 을 받고 drop
    // 하는 순간 directive 가 소실되어 decode loop 에서 dispatch 가 일어나지
    // 않는다. directive 가 도착하면 여기에 저장하여 decode loop 의
    // `take_pending_swap_weights()` 가 consume 할 때까지 유지한다.
    pending_swap_weights: Option<(f32, argus_shared::DtypeTag)>,
    // Sticky pending RecallWeights directive (ENG-ALG-240 — 동형 패턴).
    pending_recall_weights: Option<f32>,
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
            throttle_delay_ms: 0,
            target_tbt_ms: 0,
            target_tbt_set: false,
            active_actions: Vec::new(),
            evict_plan: None,
            kv_quant_bits: None,
            partition_ratio_sticky: None,
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
            pending_swap_weights: None,
            pending_recall_weights: None,
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

    /// Consume pending SwapWeights directive, if any (LISWAP-6 manager routing).
    ///
    /// `apply_command::SwapWeights` stashes the directive in
    /// `pending_swap_weights` so it survives `executor.poll` calls made by
    /// prefill loops (which drop the `ExecutionPlan` without consuming
    /// `swap_weights`). The decode loop calls this each tick before
    /// dispatching incremental swap.
    pub fn take_pending_swap_weights(&mut self) -> Option<(f32, argus_shared::DtypeTag)> {
        self.pending_swap_weights.take()
    }

    /// Consume pending RecallWeights directive, if any (ENG-ALG-240 — swap-routing 동형).
    pub fn take_pending_recall_weights(&mut self) -> Option<f32> {
        self.pending_recall_weights.take()
    }

    /// Notify executor that inference has started.
    pub fn set_running(&mut self) {
        self.engine_state = EngineState::Running;
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

    /// β-4 (매핑 문서 4부 채택안 (가)): heartbeat interval 이 경과했으면 1회 송출한다.
    ///
    /// 구 [`Self::poll`] 의 heartbeat 분기(이 메서드로 분리)와 byte-identical — interval 체크 +
    /// `send_heartbeat` + `last_heartbeat` 갱신. `ManagerCommandSource` 가 pure `drain_commands`
    /// 직전에 호출하여 heartbeat 송출 등가를 보존한다(`CommandSource::poll` pure 화 후에도 송출
    /// 경로 무단절). `kv_snap` 은 source 가 held-handle 에서 query 해 구성한 값.
    pub fn send_heartbeat_if_due(&mut self, kv_snap: &KVSnapshot) {
        if self.last_heartbeat.elapsed() >= self.heartbeat_interval {
            self.send_heartbeat(kv_snap);
            self.last_heartbeat = Instant::now();
        }
    }

    /// β-4 (v2 §5.4 A-1): 도착한 manager command 를 drain 하여 반환한다 (pure 생산).
    ///
    /// 구 [`Self::poll`] 의 mpsc drain + per-directive 응답 송출 부분만 분리한다 —
    /// `apply_command`/`ExecutionPlan` 번역은 하지 않는다(그건 `CommandDispatcher` 책임). drain 한
    /// command 를 `EngineDirective` 순서대로 평탄화해 반환하며, 각 directive 의 command 수만큼
    /// `CommandResult::Ok` 응답을 즉시 송출한다(v1 응답 송출 등가 — apply_command 가 항상 Ok 였으므로
    /// 등가). heartbeat 송출은 별도([`Self::send_heartbeat_if_due`]).
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

    /// Poll for pending commands and build an ExecutionPlan.
    /// Called once per token in the inference loop.
    pub fn poll(&mut self, kv_snap: &KVSnapshot) -> ExecutionPlan {
        let mut plan = ExecutionPlan::default();

        // 1. Heartbeat check
        if self.last_heartbeat.elapsed() >= self.heartbeat_interval {
            self.send_heartbeat(kv_snap);
            self.last_heartbeat = Instant::now();
        }

        // 2. Drain all pending messages
        let mut directives = Vec::new();
        while let Ok(msg) = self.cmd_rx.try_recv() {
            match msg {
                ManagerMessage::Directive(d) => directives.push(d),
            }
        }

        if directives.is_empty() {
            // Maintain existing throttle, target TBT, and sticky evict/quant state
            plan.throttle_delay_ms = self.throttle_delay_ms;
            plan.target_tbt_ms = self.target_tbt_ms;
            plan.target_tbt_set = self.target_tbt_set;
            plan.evict = self.evict_plan.clone();
            plan.kv_quant_bits = self.kv_quant_bits;
            // Carry forward sticky partition ratio so prefill-time directives
            // remain visible in subsequent decode-loop polls (ISSUE-5).
            plan.partition_ratio = self.partition_ratio_sticky;
            return plan;
        }

        // 3. Process each directive, building plan and responses
        for directive in &directives {
            let seq_id = directive.seq_id;
            let mut results = Vec::with_capacity(directive.commands.len());

            for cmd in &directive.commands {
                eprintln!("[Resilience] Directive seq={}: {:?}", seq_id, cmd);
                let result = self.apply_command(cmd, &mut plan);
                results.push(result);
            }

            // Send response
            let _ = self
                .resp_tx
                .send(EngineMessage::Response(CommandResponse { seq_id, results }));
        }

        // 4. Carry forward sticky state if not overridden by this directive batch
        if plan.evict.is_none() {
            plan.evict = self.evict_plan.clone();
        }
        if plan.kv_quant_bits.is_none() {
            plan.kv_quant_bits = self.kv_quant_bits;
        }

        // 5. Suspend overrides everything
        if plan.suspended {
            plan.evict = None;
            plan.switch_device = None;
            plan.prepare_device = None;
            plan.throttle_delay_ms = 0;
            plan.resumed = false;
            self.engine_state = EngineState::Suspended;
        }

        // Update internal throttle state
        self.throttle_delay_ms = plan.throttle_delay_ms;

        plan
    }

    fn apply_command(&mut self, cmd: &EngineCommand, plan: &mut ExecutionPlan) -> CommandResult {
        match cmd {
            EngineCommand::Throttle { delay_ms } => {
                plan.throttle_delay_ms = *delay_ms;
                if *delay_ms > 0 {
                    if !self.active_actions.contains(&"throttle".to_string()) {
                        self.active_actions.push("throttle".to_string());
                    }
                } else {
                    self.active_actions.retain(|a| a != "throttle");
                }
                CommandResult::Ok
            }
            EngineCommand::SetTargetTbt { target_ms } => {
                plan.target_tbt_ms = *target_ms;
                plan.target_tbt_set = true;
                self.target_tbt_ms = *target_ms;
                self.target_tbt_set = true;
                if *target_ms > 0 {
                    // DecodeLoop::run 의 pacing 발동을 알리는 transition 로그.
                    eprintln!("[Resilience] SetTargetTbt: pacing to {}ms", target_ms);
                    if !self.active_actions.contains(&"target_tbt".to_string()) {
                        self.active_actions.push("target_tbt".to_string());
                    }
                } else {
                    eprintln!("[Resilience] SetTargetTbt: released (-> 0ms)");
                    self.active_actions.retain(|a| a != "target_tbt");
                }
                CommandResult::Ok
            }
            EngineCommand::LayerSkip { skip_ratio } => {
                plan.layer_skip = Some(*skip_ratio);
                if !self.active_actions.contains(&"weight.skip".to_string()) {
                    self.active_actions.push("weight.skip".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvEvictH2o { keep_ratio } => {
                let evict = EvictPlan {
                    target_ratio: *keep_ratio,
                    level: ResourceLevel::Critical,
                    method: EvictMethod::H2o,
                    streaming_params: None,
                };
                self.evict_plan = Some(evict.clone());
                plan.evict = Some(evict);
                if !self.active_actions.contains(&"kv.evict_h2o".to_string()) {
                    self.active_actions.push("kv.evict_h2o".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvEvictSliding { keep_ratio } => {
                let evict = EvictPlan {
                    target_ratio: *keep_ratio,
                    level: ResourceLevel::Critical,
                    method: EvictMethod::Sliding,
                    streaming_params: None,
                };
                self.evict_plan = Some(evict.clone());
                plan.evict = Some(evict);
                if !self
                    .active_actions
                    .contains(&"kv.evict_sliding".to_string())
                {
                    self.active_actions.push("kv.evict_sliding".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvStreaming {
                sink_size,
                window_size,
            } => {
                let evict = EvictPlan {
                    target_ratio: 0.0, // streaming stage ignores target_len (uses sink+window)
                    level: ResourceLevel::Critical,
                    method: EvictMethod::Streaming,
                    streaming_params: Some(StreamingParams {
                        sink_size: *sink_size,
                        window_size: *window_size,
                    }),
                };
                self.evict_plan = Some(evict.clone());
                plan.evict = Some(evict);
                if !self
                    .active_actions
                    .contains(&"kv.evict_streaming".to_string())
                {
                    self.active_actions.push("kv.evict_streaming".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvMergeD2o { keep_ratio } => {
                let evict = EvictPlan {
                    target_ratio: *keep_ratio,
                    level: ResourceLevel::Critical,
                    method: EvictMethod::D2o,
                    streaming_params: None,
                };
                self.evict_plan = Some(evict.clone());
                plan.evict = Some(evict);
                if !self.active_actions.contains(&"kv.merge_d2o".to_string()) {
                    self.active_actions.push("kv.merge_d2o".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvQuantDynamic { target_bits } => {
                self.kv_quant_bits = Some(*target_bits);
                plan.kv_quant_bits = Some(*target_bits);
                if !self
                    .active_actions
                    .contains(&"kv.quant_dynamic".to_string())
                {
                    self.active_actions.push("kv.quant_dynamic".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvOffload { ratio } => {
                let r = ratio.clamp(0.0, 1.0);
                plan.offload_ratio = Some(r);
                if !self.active_actions.contains(&"kv_offload".to_string()) {
                    self.active_actions.push("kv_offload".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::RestoreDefaults => {
                plan.restore_defaults = true;
                plan.recall_offload = true;
                plan.throttle_delay_ms = 0;
                plan.target_tbt_ms = 0;
                // Intentionally leave plan.target_tbt_set = false so the engine
                // takes the `restore_defaults` branch (CLI fallback) rather
                // than treating this as an explicit "set to 0".
                self.throttle_delay_ms = 0;
                self.target_tbt_ms = 0;
                self.target_tbt_set = false;
                self.evict_plan = None;
                self.kv_quant_bits = None;
                self.partition_ratio_sticky = None;
                self.compute_level = ResourceLevel::Normal;
                self.memory_level = ResourceLevel::Normal;
                self.active_actions.clear();
                CommandResult::Ok
            }
            EngineCommand::SwitchHw { device } => {
                // 배치 내 이전 스위치를 덮어쓴다
                plan.switch_device = Some(device.clone());
                CommandResult::Ok
            }
            EngineCommand::PrepareComputeUnit { device } => {
                plan.prepare_device = Some(device.clone());
                CommandResult::Ok
            }
            EngineCommand::Suspend => {
                plan.suspended = true;
                self.engine_state = EngineState::Suspended;
                CommandResult::Ok
            }
            EngineCommand::Resume => {
                plan.resumed = true;
                self.engine_state = EngineState::Running;
                // Resume 시 레벨과 스로틀을 Normal로 초기화
                self.compute_level = ResourceLevel::Normal;
                self.memory_level = ResourceLevel::Normal;
                self.throttle_delay_ms = 0;
                plan.throttle_delay_ms = 0;
                CommandResult::Ok
            }
            EngineCommand::RequestQcf => {
                plan.request_qcf = true;
                CommandResult::Ok
            }
            EngineCommand::SetPartitionRatio { ratio } => {
                self.partition_ratio_sticky = Some(*ratio);
                plan.partition_ratio = Some(*ratio);
                CommandResult::Ok
            }
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                if let Some(v) = chunk_size {
                    plan.prefill_chunk_size = Some(*v);
                }
                if let Some(v) = yield_ms {
                    plan.prefill_yield_ms = Some(*v);
                }
                if let Some(v) = cpu_chunk_size {
                    plan.prefill_cpu_chunk_size = Some(*v);
                }
                CommandResult::Ok
            }
            // SwapWeights is dispatched directly in generate.rs (ENG-ALG-214-ROUTE).
            // Manager 가 prefill 도중 directive 를 보낼 수 있으므로 sticky
            // `pending_swap_weights` 에 저장한다. prefill loop 의 `executor.poll`
            // 은 plan 을 받고 drop 하지만, decode loop 가
            // `take_pending_swap_weights()` 로 consume 할 때까지 directive 는
            // executor 내부에 살아남는다 (LISWAP-6 manager routing fix).
            //
            // `plan.swap_weights` 는 backward-compat 으로 유지 — 동일 directive
            // 가 decode 시점 poll 에서 곧바로 들어오는 경우(예: directive 가 한
            // 번도 흘려보내지지 않은 idle 상태) 에도 정상 동작.
            EngineCommand::SwapWeights {
                ratio,
                target_dtype,
            } => {
                self.pending_swap_weights = Some((*ratio, *target_dtype));
                plan.swap_weights = Some((*ratio, *target_dtype));
                CommandResult::Ok
            }
            // RecallWeights is dispatched directly in CommandDispatcher (ENG-ALG-240).
            // The resilience executor (legacy path) stores it in pending_recall_weights
            // so the decode loop can consume it via take_pending_recall_weights().
            EngineCommand::RecallWeights { ratio } => {
                self.pending_recall_weights = Some(*ratio);
                CommandResult::Ok
            }
        }
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

    /// Current engine state.
    pub fn state(&self) -> EngineState {
        self.engine_state
    }

    /// Current compute level (deprecated, always Normal unless set by legacy path).
    pub fn compute_level(&self) -> ResourceLevel {
        self.compute_level
    }

    /// Current memory level (deprecated, always Normal unless set by legacy path).
    pub fn memory_level(&self) -> ResourceLevel {
        self.memory_level
    }

    /// Current throttle delay.
    pub fn throttle_delay_ms(&self) -> u64 {
        self.throttle_delay_ms
    }

    /// Update tensor partition ratio for heartbeat reporting.
    pub fn set_partition_ratio(&mut self, ratio: f32) {
        self.partition_ratio = ratio;
    }

    /// Seed the sticky throttle delay (e.g. from CLI `--throttle-delay-ms`).
    /// Future Manager `Throttle` directives still override this value.
    pub fn set_throttle_delay_ms(&mut self, delay_ms: u64) {
        self.throttle_delay_ms = delay_ms;
    }

    /// Update prefill progress for heartbeat reporting.
    pub fn set_prefill_state(&mut self, phase: &str, pos: usize, total: usize) {
        self.phase = phase.to_string();
        self.prefill_pos = pos;
        self.prefill_total = total;
    }

    /// Currently active action names.
    pub fn active_actions(&self) -> &[String] {
        &self.active_actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_shared::EngineDirective;

    fn make_executor() -> (
        CommandExecutor,
        mpsc::Sender<ManagerMessage>,
        mpsc::Receiver<EngineMessage>,
    ) {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (resp_tx, resp_rx) = mpsc::channel();
        let executor = CommandExecutor::new(
            cmd_rx,
            resp_tx,
            "cpu".to_string(),
            Duration::from_secs(10), // 테스트에서 하트비트 노이즈 방지용 긴 간격
        );
        (executor, cmd_tx, resp_rx)
    }

    fn empty_snap() -> KVSnapshot {
        KVSnapshot::default()
    }

    fn snap_with_tokens(tokens: usize, capacity: usize, prefix: usize) -> KVSnapshot {
        KVSnapshot {
            total_bytes: (tokens * 256) as u64,
            total_tokens: tokens,
            capacity,
            protected_prefix: prefix,
            kv_dtype: "f16".to_string(),
            eviction_policy: "none".to_string(),
            skip_ratio: 0.0,
        }
    }

    #[test]
    fn test_executor_poll_empty() {
        let (mut executor, _tx, _rx) = make_executor();
        let plan = executor.poll(&empty_snap());
        assert!(plan.evict.is_none());
        assert!(plan.switch_device.is_none());
        assert!(!plan.suspended);
        assert!(!plan.resumed);
        assert_eq!(plan.throttle_delay_ms, 0);
    }

    #[test]
    fn test_executor_throttle_direct() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Throttle { delay_ms: 50 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.throttle_delay_ms, 50);
        assert!(executor.active_actions().contains(&"throttle".to_string()));

        // 응답 확인
        let resp = rx.recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 1);
                assert!(matches!(r.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_executor_kv_evict_h2o() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(plan.evict.is_some());
        let evict = plan.evict.unwrap();
        assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(evict.level, ResourceLevel::Critical);
        assert_eq!(evict.method, EvictMethod::H2o);
        assert!(
            executor
                .active_actions()
                .contains(&"kv.evict_h2o".to_string())
        );

        let resp = rx.recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 1);
                assert!(matches!(r.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_executor_kv_evict_sliding() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.6 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(plan.evict.is_some());
        let evict = plan.evict.unwrap();
        assert!((evict.target_ratio - 0.6).abs() < f32::EPSILON);
        assert_eq!(evict.method, EvictMethod::Sliding);
        assert!(
            executor
                .active_actions()
                .contains(&"kv.evict_sliding".to_string())
        );
    }

    #[test]
    fn test_executor_kv_streaming_ok() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvStreaming {
                sink_size: 4,
                window_size: 256,
            }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(
            plan.evict.is_some(),
            "KvStreaming은 evict plan을 생성해야 함"
        );
        let evict = plan.evict.unwrap();
        assert_eq!(evict.method, EvictMethod::Streaming);
        assert!((evict.target_ratio - 0.0).abs() < f32::EPSILON);
        assert_eq!(evict.level, ResourceLevel::Critical);
        let params = evict
            .streaming_params
            .expect("streaming_params가 있어야 함");
        assert_eq!(params.sink_size, 4);
        assert_eq!(params.window_size, 256);
        assert!(
            executor
                .active_actions()
                .contains(&"kv.evict_streaming".to_string())
        );

        let resp = rx.recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 1);
                assert!(matches!(r.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_executor_kv_quant_dynamic() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvQuantDynamic { target_bits: 4 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.kv_quant_bits, Some(4));
        assert!(
            executor
                .active_actions()
                .contains(&"kv.quant_dynamic".to_string())
        );
    }

    #[test]
    fn test_kv_offload_sets_plan_field() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvOffload { ratio: 0.5 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.offload_ratio, Some(0.5));
        assert!(!plan.recall_offload);
        assert!(
            executor
                .active_actions()
                .contains(&"kv_offload".to_string())
        );

        // RestoreDefaults should set recall_offload and clear the sticky action
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::RestoreDefaults],
        }))
        .unwrap();
        let plan2 = executor.poll(&empty_snap());
        assert!(plan2.restore_defaults);
        assert!(plan2.recall_offload);
        assert!(executor.active_actions().is_empty());
    }

    #[test]
    fn test_executor_layer_skip() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::LayerSkip { skip_ratio: 0.25 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.layer_skip, Some(0.25));
        assert!(
            executor
                .active_actions()
                .contains(&"weight.skip".to_string())
        );
    }

    #[test]
    fn test_executor_restore_defaults() {
        let (mut executor, tx, _rx) = make_executor();

        // 먼저 스로틀과 액션 설정
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::Throttle { delay_ms: 50 },
                EngineCommand::LayerSkip { skip_ratio: 0.3 },
            ],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert!(!executor.active_actions().is_empty());

        // RestoreDefaults로 초기화
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::RestoreDefaults],
        }))
        .unwrap();
        let plan = executor.poll(&empty_snap());
        assert!(plan.restore_defaults);
        assert_eq!(plan.throttle_delay_ms, 0);
        assert_eq!(executor.throttle_delay_ms(), 0);
        assert!(executor.active_actions().is_empty());
    }

    #[test]
    fn test_executor_switch_hw() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SwitchHw {
                device: "opencl".to_string(),
            }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.switch_device.as_deref(), Some("opencl"));
    }

    #[test]
    fn test_executor_suspend_override() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
                EngineCommand::SwitchHw {
                    device: "cpu".to_string(),
                },
                EngineCommand::Suspend,
            ],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(plan.suspended);
        // Suspend는 다른 plan 필드를 초기화해야 함
        assert!(plan.evict.is_none());
        assert!(plan.switch_device.is_none());
        assert_eq!(executor.state(), EngineState::Suspended);
    }

    #[test]
    fn test_executor_superseding_evict() {
        let (mut executor, tx, rx) = make_executor();

        // 두 개의 evict 명령 — 두 번째가 첫 번째를 덮어씀
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvEvictH2o { keep_ratio: 0.8 }],
        }))
        .unwrap();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        // 두 번째 명령이 승리
        let evict = plan.evict.unwrap();
        assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(evict.method, EvictMethod::Sliding);

        // 두 응답 모두 전송돼야 함
        let r1 = rx.recv().unwrap();
        let r2 = rx.recv().unwrap();
        match (r1, r2) {
            (EngineMessage::Response(r1), EngineMessage::Response(r2)) => {
                assert_eq!(r1.seq_id, 1);
                assert_eq!(r2.seq_id, 2);
            }
            _ => panic!("Expected two Responses"),
        }
    }

    #[test]
    fn test_executor_heartbeat() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (resp_tx, resp_rx) = mpsc::channel();
        let mut executor = CommandExecutor::new(
            cmd_rx,
            resp_tx,
            "cpu".to_string(),
            Duration::from_millis(10), // 매우 짧은 간격으로 하트비트 유도
        );
        executor.set_running();

        // 하트비트 간격이 지나길 기다림
        std::thread::sleep(Duration::from_millis(20));

        let snap = snap_with_tokens(100, 2048, 4);
        let _plan = executor.poll(&snap);

        // 하트비트가 전송돼야 함
        let msg = resp_rx.recv().unwrap();
        match msg {
            EngineMessage::Heartbeat(status) => {
                assert_eq!(status.active_device, "cpu");
                assert_eq!(status.kv_cache_tokens, 100);
                assert_eq!(status.state, EngineState::Running);
                assert!((status.kv_cache_utilization - 100.0 / 2048.0).abs() < 0.01);
                assert!((status.memory_lossy_min - 4.0 / 100.0).abs() < 0.01);
                assert_eq!(status.memory_lossless_min, 1.0);
                assert_eq!(status.eviction_policy, "none");
                assert_eq!(status.kv_dtype, "f16");
            }
            _ => panic!("Expected Heartbeat"),
        }

        drop(cmd_tx); // 미사용 경고 억제
    }

    #[test]
    fn test_executor_heartbeat_active_actions() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (resp_tx, resp_rx) = mpsc::channel();
        let mut executor = CommandExecutor::new(
            cmd_rx,
            resp_tx,
            "cpu".to_string(),
            Duration::from_millis(10),
        );
        executor.set_running();

        // 스로틀 액션 활성화
        cmd_tx
            .send(ManagerMessage::Directive(EngineDirective {
                seq_id: 1,
                commands: vec![EngineCommand::Throttle { delay_ms: 30 }],
            }))
            .unwrap();
        executor.poll(&empty_snap());

        std::thread::sleep(Duration::from_millis(20));
        let _plan = executor.poll(&empty_snap());

        // 하트비트에 active_actions가 포함돼야 함
        // Response와 Heartbeat가 순서대로 올 수 있음
        let mut found_heartbeat = false;
        for _ in 0..3 {
            if let Ok(EngineMessage::Heartbeat(status)) = resp_rx.try_recv() {
                assert!(status.active_actions.contains(&"throttle".to_string()));
                found_heartbeat = true;
                break;
            }
        }
        assert!(
            found_heartbeat,
            "Should have received heartbeat with active_actions"
        );
    }

    #[test]
    fn test_executor_throughput_ema() {
        let (mut executor, _tx, _rx) = make_executor();

        // ~10 tok/s 속도로 토큰 생성 시뮬레이션
        executor.on_token_generated();
        std::thread::sleep(Duration::from_millis(100));
        executor.on_token_generated();

        // EMA가 약 10 tok/s 범위에 있어야 함
        assert!(executor.throughput_ema > 5.0);
        assert!(executor.throughput_ema < 20.0);
    }

    #[test]
    fn test_executor_command_response() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 42,
            commands: vec![
                EngineCommand::Throttle { delay_ms: 0 },
                EngineCommand::PrepareComputeUnit {
                    device: "gpu".to_string(),
                },
            ],
        }))
        .unwrap();

        let _plan = executor.poll(&empty_snap());

        let resp = rx.recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 42);
                assert_eq!(r.results.len(), 2);
                assert!(matches!(r.results[0], CommandResult::Ok));
                assert!(matches!(r.results[1], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_executor_resume_resets_state() {
        let (mut executor, tx, _rx) = make_executor();

        // 스로틀 설정
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Throttle { delay_ms: 100 }],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert_eq!(executor.throttle_delay_ms(), 100);

        // 중단
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::Suspend],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert_eq!(executor.state(), EngineState::Suspended);

        // Resume은 스로틀을 Normal로 초기화해야 함
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 3,
            commands: vec![EngineCommand::Resume],
        }))
        .unwrap();
        let plan = executor.poll(&empty_snap());
        assert!(plan.resumed);
        assert_eq!(executor.state(), EngineState::Running);
        assert_eq!(executor.compute_level(), ResourceLevel::Normal);
        assert_eq!(executor.memory_level(), ResourceLevel::Normal);
        assert_eq!(executor.throttle_delay_ms(), 0);
    }

    #[test]
    fn test_executor_switch_and_prepare() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::PrepareComputeUnit {
                    device: "gpu".to_string(),
                },
                EngineCommand::SwitchHw {
                    device: "gpu".to_string(),
                },
            ],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.prepare_device.as_deref(), Some("gpu"));
        assert_eq!(plan.switch_device.as_deref(), Some("gpu"));
    }

    #[test]
    fn test_executor_send_capability() {
        let (_executor, _tx, rx) = make_executor();

        _executor.send_capability(EngineCapability {
            available_devices: vec!["cpu".into(), "opencl".into()],
            active_device: "cpu".into(),
            max_kv_tokens: 2048,
            bytes_per_kv_token: 256,
            num_layers: 16,
            ..Default::default()
        });

        let msg = rx.recv().unwrap();
        match msg {
            EngineMessage::Capability(cap) => {
                assert_eq!(cap.available_devices.len(), 2);
                assert_eq!(cap.num_layers, 16);
            }
            _ => panic!("Expected Capability"),
        }
    }

    #[test]
    fn test_set_target_tbt_dirty_flag_lifecycle() {
        let (mut executor, tx, _rx) = make_executor();

        // No directive → plan.target_tbt_set must stay false (CLI default path).
        let plan0 = executor.poll(&empty_snap());
        assert!(!plan0.target_tbt_set);
        assert_eq!(plan0.target_tbt_ms, 0);

        // SetTargetTbt { 250 } → set=true, ms=250.
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SetTargetTbt { target_ms: 250 }],
        }))
        .unwrap();
        let plan1 = executor.poll(&empty_snap());
        assert!(plan1.target_tbt_set);
        assert_eq!(plan1.target_tbt_ms, 250);

        // No directive → sticky (set=true, ms=250).
        let plan2 = executor.poll(&empty_snap());
        assert!(plan2.target_tbt_set);
        assert_eq!(plan2.target_tbt_ms, 250);

        // SetTargetTbt { 0 } (explicit pacing disable) → set=true, ms=0.
        // This is the ISSUE-3 fix: the engine must see set=true to honor 0.
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::SetTargetTbt { target_ms: 0 }],
        }))
        .unwrap();
        let plan3 = executor.poll(&empty_snap());
        assert!(plan3.target_tbt_set);
        assert_eq!(plan3.target_tbt_ms, 0);

        // RestoreDefaults → set=false (CLI fallback branch), restore flag on.
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 3,
            commands: vec![EngineCommand::RestoreDefaults],
        }))
        .unwrap();
        let plan4 = executor.poll(&empty_snap());
        assert!(!plan4.target_tbt_set);
        assert_eq!(plan4.target_tbt_ms, 0);
        assert!(plan4.restore_defaults);
    }

    #[test]
    fn test_executor_restore_clears_active_actions() {
        let (mut executor, tx, _rx) = make_executor();

        // 여러 액션 활성화
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::Throttle { delay_ms: 50 },
                EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
                EngineCommand::LayerSkip { skip_ratio: 0.2 },
            ],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert_eq!(executor.active_actions().len(), 3);

        // RestoreDefaults로 모두 초기화
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::RestoreDefaults],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert!(executor.active_actions().is_empty());
        assert_eq!(executor.throttle_delay_ms(), 0);
    }

    #[test]
    fn test_request_qcf_sets_plan_flag() {
        let (mut executor, tx, rx) = make_executor();
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 10,
            commands: vec![EngineCommand::RequestQcf],
        }))
        .unwrap();
        let plan = executor.poll(&empty_snap());
        assert!(plan.request_qcf);

        // Should receive Response with Ok
        let resp = rx.try_recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 10);
                assert_eq!(r.results.len(), 1);
                assert!(matches!(r.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_request_qcf_with_other_commands() {
        let (mut executor, tx, rx) = make_executor();
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 11,
            commands: vec![
                EngineCommand::Throttle { delay_ms: 30 },
                EngineCommand::RequestQcf,
            ],
        }))
        .unwrap();
        let plan = executor.poll(&empty_snap());
        assert!(plan.request_qcf);
        assert_eq!(plan.throttle_delay_ms, 30);

        let resp = rx.try_recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.results.len(), 2);
                assert!(matches!(r.results[0], CommandResult::Ok));
                assert!(matches!(r.results[1], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_request_qcf_default_false() {
        let (mut executor, _tx, _rx) = make_executor();
        let plan = executor.poll(&empty_snap());
        assert!(!plan.request_qcf);
    }

    #[test]
    fn test_evict_plan_is_sticky() {
        let (mut executor, tx, _rx) = make_executor();

        // KvEvictSliding 커맨드를 한 번만 보냄
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.7 }],
        }))
        .unwrap();
        let plan1 = executor.poll(&empty_snap());
        assert!(plan1.evict.is_some());
        assert_eq!(plan1.evict.as_ref().unwrap().method, EvictMethod::Sliding);

        // 다음 poll()에서는 커맨드 없이도 evict_plan이 유지돼야 함
        let plan2 = executor.poll(&empty_snap());
        assert!(
            plan2.evict.is_some(),
            "evict_plan should be sticky across polls"
        );
        let evict = plan2.evict.unwrap();
        assert!((evict.target_ratio - 0.7).abs() < f32::EPSILON);
        assert_eq!(evict.method, EvictMethod::Sliding);
    }

    #[test]
    fn test_kv_quant_bits_is_sticky() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvQuantDynamic { target_bits: 4 }],
        }))
        .unwrap();
        let plan1 = executor.poll(&empty_snap());
        assert_eq!(plan1.kv_quant_bits, Some(4));

        // 다음 poll()에서도 kv_quant_bits가 유지돼야 함
        let plan2 = executor.poll(&empty_snap());
        assert_eq!(
            plan2.kv_quant_bits,
            Some(4),
            "kv_quant_bits should be sticky across polls"
        );
    }

    #[test]
    fn test_restore_defaults_clears_sticky_evict_and_quant() {
        let (mut executor, tx, _rx) = make_executor();

        // evict와 quant 설정
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
                EngineCommand::KvQuantDynamic { target_bits: 8 },
            ],
        }))
        .unwrap();
        executor.poll(&empty_snap());

        // sticky 확인
        let plan_mid = executor.poll(&empty_snap());
        assert!(plan_mid.evict.is_some());
        assert_eq!(plan_mid.kv_quant_bits, Some(8));

        // RestoreDefaults로 초기화
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::RestoreDefaults],
        }))
        .unwrap();
        executor.poll(&empty_snap());

        // 이후 poll에서는 evict와 quant가 사라져야 함
        let plan_after = executor.poll(&empty_snap());
        assert!(
            plan_after.evict.is_none(),
            "evict_plan should be cleared after RestoreDefaults"
        );
        assert_eq!(
            plan_after.kv_quant_bits, None,
            "kv_quant_bits should be cleared after RestoreDefaults"
        );
    }

    #[test]
    fn test_new_evict_command_overrides_sticky() {
        let (mut executor, tx, _rx) = make_executor();

        // H2O로 설정
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }],
        }))
        .unwrap();
        executor.poll(&empty_snap());

        // Sliding으로 교체
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.8 }],
        }))
        .unwrap();
        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.evict.as_ref().unwrap().method, EvictMethod::Sliding);
        assert!((plan.evict.as_ref().unwrap().target_ratio - 0.8).abs() < f32::EPSILON);

        // 이후 poll에서도 Sliding이 유지돼야 함
        let plan_next = executor.poll(&empty_snap());
        assert_eq!(
            plan_next.evict.as_ref().unwrap().method,
            EvictMethod::Sliding
        );
    }

    #[test]
    fn test_send_qcf_estimate() {
        let (executor, _tx, rx) = make_executor();
        let mut estimates = std::collections::HashMap::new();
        estimates.insert("kv.evict_sliding".to_string(), 0.5);
        estimates.insert("kv.evict_h2o".to_string(), 0.1);
        executor.send_qcf_estimate(QcfEstimate {
            estimates,
            layer_swap: None,
        });

        let msg = rx.try_recv().unwrap();
        match msg {
            EngineMessage::QcfEstimate(qcf) => {
                assert_eq!(qcf.estimates.len(), 2);
                assert!((qcf.estimates["kv.evict_sliding"] - 0.5).abs() < f32::EPSILON);
                assert!((qcf.estimates["kv.evict_h2o"] - 0.1).abs() < f32::EPSILON);
            }
            _ => panic!("Expected QcfEstimate"),
        }
    }

    // ISSUE-5: SetPartitionRatio는 sticky 상태로 유지되어 prefill 중 수신된
    // directive가 이후 decode-loop poll에서도 반영돼야 한다.
    #[test]
    fn test_set_partition_ratio_sticky_across_polls() {
        let (mut executor, tx, _rx) = make_executor();

        // 1. SetPartitionRatio directive 송신
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }],
        }))
        .unwrap();

        // 2. 첫 poll → plan.partition_ratio == Some(0.3)
        let plan1 = executor.poll(&empty_snap());
        assert_eq!(
            plan1.partition_ratio,
            Some(0.3),
            "첫 poll에서 SetPartitionRatio가 반영돼야 함"
        );

        // 3. 두 번째 poll (directive 없음) → plan.partition_ratio == Some(0.3) (sticky)
        let plan2 = executor.poll(&empty_snap());
        assert_eq!(
            plan2.partition_ratio,
            Some(0.3),
            "directive 없는 poll에서도 partition_ratio_sticky가 유지돼야 함 (ISSUE-5)"
        );

        // 4. 세 번째 poll (여전히 directive 없음) → 여전히 sticky
        let plan3 = executor.poll(&empty_snap());
        assert_eq!(
            plan3.partition_ratio,
            Some(0.3),
            "복수의 directive 없는 poll에서도 sticky 유지"
        );

        // 5. RestoreDefaults directive 송신
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::RestoreDefaults],
        }))
        .unwrap();
        executor.poll(&empty_snap());

        // 6. RestoreDefaults 이후 poll → partition_ratio가 None으로 초기화
        let plan_after = executor.poll(&empty_snap());
        assert_eq!(
            plan_after.partition_ratio, None,
            "RestoreDefaults 이후 partition_ratio_sticky가 초기화돼야 함"
        );
    }

    /// LISWAP-6 manager routing fix: SwapWeights directive 가 prefill loop 의
    /// `poll`(`plan.swap_weights` drop) 을 거친 뒤에도 decode loop 가 sticky
    /// `take_pending_swap_weights()` 로 consume 할 수 있어야 한다.
    #[test]
    fn test_swap_weights_pending_survives_intermediate_polls() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SwapWeights {
                ratio: 0.9,
                target_dtype: argus_shared::DtypeTag::Q4_0,
            }],
        }))
        .unwrap();

        // 1차 poll: prefill loop 가 호출했다고 가정. plan.swap_weights 는 채워지지만
        // prefill 은 이를 consume 하지 않고 drop.
        let plan1 = executor.poll(&empty_snap());
        assert_eq!(
            plan1.swap_weights,
            Some((0.9, argus_shared::DtypeTag::Q4_0))
        );
        drop(plan1);

        // 2차 poll: directive 없음 → plan.swap_weights 는 None 이지만, sticky
        // pending_swap_weights 는 살아 있어야 한다 (decode loop 의 take).
        let plan2 = executor.poll(&empty_snap());
        assert!(
            plan2.swap_weights.is_none(),
            "transient plan.swap_weights 는 두 번째 poll 에서 비어 있어야 함"
        );
        let taken = executor.take_pending_swap_weights();
        assert_eq!(
            taken,
            Some((0.9, argus_shared::DtypeTag::Q4_0)),
            "pending_swap_weights 는 prefill poll 후에도 sticky 로 유지돼야 함"
        );

        // take 이후에는 비어 있어야 한다 (one-shot consume).
        assert!(
            executor.take_pending_swap_weights().is_none(),
            "take 이후 pending_swap_weights 는 None"
        );
    }

    #[test]
    fn test_swap_weights_same_tick_consume_via_plan() {
        // Decode loop 가 sticky 와 plan 양쪽을 참조하므로, 같은 tick 에 directive
        // 가 도착한 경우에도 plan.swap_weights 가 (fallback 으로) 정확히 set 됨.
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SwapWeights {
                ratio: 0.5,
                target_dtype: argus_shared::DtypeTag::F16,
            }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.swap_weights, Some((0.5, argus_shared::DtypeTag::F16)));
        // sticky 도 동일 directive 로 set.
        assert_eq!(
            executor.take_pending_swap_weights(),
            Some((0.5, argus_shared::DtypeTag::F16))
        );
    }
}
