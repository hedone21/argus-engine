//! Phase β-4 host 게이트 — EngineCommand → 신 채널(CommandDispatcher/LoopControl + OneShot Stage)
//! 의 구==신 등가 + heartbeat 송출 연속성.
//!
//! 설계 SSOT: `arch/beta4_command_channel_mapping.md` 5부 (host 게이트 명세 5종).
//!
//! 본 파일이 커버하는 게이트:
//! 1. **매핑표 행별 등가** — mock directive 시퀀스로 v1 `ExecutionPlan` 산출 ↔ 신 `LoopControl`
//!    산출 동치 (control + 과도기 필드). (dispatcher unit `src/session/command_dispatcher.rs` 가
//!    sticky/method-drop/exhaustive 를 커버하므로, 본 파일은 **v1 executor anchor 대조**에 집중.)
//! 4. **heartbeat 연속성** — `ResilienceAdapter::poll`(pure) 전환 후에도 heartbeat 가 interval
//!    마다 송출되고, payload(kv_cache_tokens == held-handle.current_pos())가 v1 등가.
//!
//! 게이트 2(exhaustive match)/3(sticky 전이)/5(method-drop)은 dispatcher unit 에서 컴파일·검증.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use argus_shared::{EngineCommand, EngineDirective, EngineMessage, EngineState, ManagerMessage};

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::format::KVCacheFormat;
use argus_engine::kv::cache_manager::CacheManager;
use argus_engine::kv::eviction::sliding_window::SlidingWindowPolicy;
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::kv::standard_format::StandardFormat;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::resilience::sys_monitor::NoOpMonitor;
use argus_engine::resilience::{CommandExecutor, KVSnapshot};
use argus_engine::session::CommandSource;
use argus_engine::session::command_dispatcher::CommandDispatcher;
use argus_engine::session::pipeline_registry::PipelineRegistry;
use argus_engine::session::resilience_adapter::ResilienceAdapter;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;

const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 32;
const MAX_SEQ: usize = 128;

fn make_handle(n_tokens: usize) -> Arc<StandardFormat> {
    let total = MAX_SEQ * KV_HEADS * HEAD_DIM;
    let k_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
    let v_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
    let k = Tensor::new(shape.clone(), k_buf, backend.clone());
    let v = Tensor::new(shape, v_buf, backend);
    let mut cache = KVCache::new(k, v, MAX_SEQ);
    cache.current_pos = n_tokens;
    Arc::new(StandardFormat::new(0, cache))
}

fn make_cm() -> Arc<Mutex<CacheManager>> {
    let policy = Box::new(SlidingWindowPolicy::new(10, 4));
    Arc::new(Mutex::new(CacheManager::new(
        policy,
        Box::new(NoOpMonitor),
        usize::MAX,
        0.3,
    )))
}

/// v1 `CommandExecutor` + mpsc 채널 헬퍼.
fn make_executor() -> (
    CommandExecutor,
    mpsc::Sender<ManagerMessage>,
    mpsc::Receiver<EngineMessage>,
) {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let exec = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_secs(3600), // heartbeat 노이즈 차단(등가 테스트는 plan 산출만)
    );
    (exec, cmd_tx, resp_rx)
}

fn send(tx: &mpsc::Sender<ManagerMessage>, seq_id: u64, cmds: Vec<EngineCommand>) {
    tx.send(ManagerMessage::Directive(EngineDirective {
        seq_id,
        commands: cmds,
    }))
    .unwrap();
}

// ── 게이트 1: 매핑표 행별 등가 (v1 ExecutionPlan ↔ 신 LoopControl) ──
//
// 같은 directive 시퀀스를 (A) v1 executor.poll → ExecutionPlan, (B) 신 dispatcher.dispatch →
// LoopControl 로 돌려 control + 과도기 필드의 산출이 동치임을 단언한다. anchor = v1 executor.

/// throttle/tbt/suspend control 3종 + RestoreDefaults reset 등가.
#[test]
fn control_fields_equivalent_to_v1() {
    let (mut exec, tx, _rx) = make_executor();
    let registry = Arc::new(PipelineRegistry::new());
    let mut disp = CommandDispatcher::new(
        registry,
        vec![make_handle(120)],
        Some(make_cm()),
        Vec::new(),
        None,
        None,
        None,
        None,
        Vec::new(),
        None,                       // report_tx: AB-5 — 단위테스트는 미배선
        Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
        Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
    );

    // step 1: Throttle{50} + SetTargetTbt{200}.
    let cmds1 = vec![
        EngineCommand::Throttle { delay_ms: 50 },
        EngineCommand::SetTargetTbt { target_ms: 200 },
    ];
    send(&tx, 1, cmds1.clone());
    let v1 = exec.poll(&KVSnapshot::default());
    let v2 = disp.dispatch(cmds1);
    assert_eq!(v1.throttle_delay_ms, v2.throttle_delay_ms, "throttle 등가");
    assert_eq!(v1.target_tbt_ms, v2.target_tbt_ms, "tbt 등가");
    assert_eq!(v1.target_tbt_set, v2.target_tbt_set, "tbt_set 등가");
    assert!(!v1.suspended && !v2.suspended);

    // step 2: 빈 batch → sticky carry 등가 (throttle/tbt 유지).
    let v1b = exec.poll(&KVSnapshot::default());
    let v2b = disp.dispatch(vec![]);
    assert_eq!(
        v1b.throttle_delay_ms, v2b.throttle_delay_ms,
        "sticky throttle"
    );
    assert_eq!(v1b.target_tbt_ms, v2b.target_tbt_ms, "sticky tbt");
    assert_eq!(v1b.target_tbt_set, v2b.target_tbt_set);

    // step 3: RestoreDefaults → reset 묶음 등가.
    send(&tx, 2, vec![EngineCommand::RestoreDefaults]);
    let v1c = exec.poll(&KVSnapshot::default());
    let v2c = disp.dispatch(vec![EngineCommand::RestoreDefaults]);
    assert_eq!(
        v1c.throttle_delay_ms, v2c.throttle_delay_ms,
        "restore throttle=0"
    );
    assert_eq!(v1c.target_tbt_ms, v2c.target_tbt_ms, "restore tbt=0");
    assert_eq!(
        v1c.target_tbt_set, v2c.target_tbt_set,
        "restore tbt_set=false"
    );
    assert_eq!(
        v1c.restore_defaults, v2c.restore_defaults,
        "restore_defaults flag"
    );
}

/// suspend override 등가 — suspend 시 device seam 무효 + throttle 0.
#[test]
fn suspend_override_equivalent_to_v1() {
    let (mut exec, tx, _rx) = make_executor();
    let registry = Arc::new(PipelineRegistry::new());
    let mut disp = CommandDispatcher::new(
        registry,
        vec![make_handle(120)],
        Some(make_cm()),
        Vec::new(),
        None,
        None,
        None,
        None,
        Vec::new(),
        None,                       // report_tx: AB-5
        Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
        Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
    );

    let cmds = vec![
        EngineCommand::SwitchHw {
            device: "cpu".to_string(),
        },
        EngineCommand::Suspend,
    ];
    send(&tx, 1, cmds.clone());
    let v1 = exec.poll(&KVSnapshot::default());
    let v2 = disp.dispatch(cmds);
    assert_eq!(v1.suspended, v2.suspended, "suspended 등가");
    assert!(v1.suspended);
    assert_eq!(
        v1.switch_device.is_none(),
        v2.switch_device.is_none(),
        "suspend override device seam 등가"
    );
}

/// 과도기 1종(layer_skip) live 소비 경로 보유분 구==신 등가.
///
/// AB-3 완료: KvOffload 는 OneShot OffloadStage submit 으로 이전됨(offload_ratio/recall_offload 필드
/// 삭제). 잔존 과도기는 layer_skip 만. offload submit 거동은 dispatcher unit
/// `offload_submits_one_shot_each_directive` / `restore_defaults_submits_recall_when_offloaded` 가 검증.
#[test]
fn transitional_fields_equivalent_to_v1() {
    let (mut exec, tx, _rx) = make_executor();
    let registry = Arc::new(PipelineRegistry::new());
    let mut disp = CommandDispatcher::new(
        registry,
        vec![make_handle(120)],
        Some(make_cm()),
        Vec::new(),
        None,
        None,
        None,
        None,
        Vec::new(),
        None,                       // report_tx: AB-5
        Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
        Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
    );

    let cmds = vec![EngineCommand::LayerSkip { skip_ratio: 0.25 }];
    send(&tx, 1, cmds.clone());
    let v1 = exec.poll(&KVSnapshot::default());
    let v2 = disp.dispatch(cmds);
    assert_eq!(v1.layer_skip, v2.layer_skip, "layer_skip 등가");
}

/// AB-4: SetPartitionRatio directive → OneShot PartitionStage submit 거동 검증.
///
/// v1 의 `partition_ratio` 필드 carry(LoopControl 값 비교)를 대체하는 submit-시맨틱 게이트:
/// dispatch 후 registry.len() 증가 1 / 같은 값 반복 미증가 / 값 변경 재증가. (구==신 비교 차원이
/// "LoopControl 필드 값" → "registry submit 횟수" 로 바뀜 — §5.5.2 last-applied 게이트.)
#[test]
fn partition_directive_submits_one_shot_stage() {
    use argus_engine::hardware::Hardware;
    use argus_engine::layers::transformer_layer::TransformerLayer;
    use argus_engine::memory::Memory;
    use argus_engine::memory::galloc::Galloc;
    use argus_engine::models::weights::LayerSlot;

    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let f32_weight = |out_dim: usize, in_dim: usize| -> Tensor {
        let buf: Arc<dyn argus_engine::buffer::Buffer> =
            Arc::new(SharedBuffer::new(out_dim * in_dim * 4, DType::F32));
        Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, be.clone())
    };
    let small = f32_weight(1, 1);
    let layer = TransformerLayer {
        wq: small.clone(),
        wk: small.clone(),
        wv: small.clone(),
        wo: small.clone(),
        w_gate: f32_weight(512, 256),
        w_up: f32_weight(512, 256),
        w_down: f32_weight(256, 512),
        attention_norm: small.clone(),
        ffn_norm: small,
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    };
    let slots: Vec<Arc<LayerSlot>> = vec![Arc::new(LayerSlot::new(layer, DType::F32, None, 0))];
    let host: Arc<dyn Memory> = Arc::new(Galloc::new());
    let hw = Arc::new(Hardware::new(be.clone(), None, None, host, None));

    let registry = Arc::new(PipelineRegistry::new());
    let mut disp = CommandDispatcher::new(
        Arc::clone(&registry),
        Vec::new(),
        None,
        slots,
        Some(hw),
        None,
        None,
        None,
        Vec::new(),
        None,                       // report_tx: AB-5
        Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
        Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
    );

    // 새 ratio → submit 1.
    disp.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }]);
    assert_eq!(registry.len(), 1, "첫 partition directive → OneShot submit");
    // 같은 값 반복 → 미증가.
    disp.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.3 }]);
    assert_eq!(registry.len(), 1, "같은 ratio 반복 — 재submit 없음");
    // 값 변경 → 재증가.
    disp.dispatch(vec![EngineCommand::SetPartitionRatio { ratio: 0.5 }]);
    assert_eq!(registry.len(), 2, "ratio 변경 → 새 OneShot submit");
}

/// AB-2: KvQuantDynamic directive → OneShot KiviQuantStage submit 거동 검증.
///
/// v1 의 `kv_quant_bits` 필드 carry(LoopControl 값 비교)를 대체하는 submit-시맨틱 게이트 (§5.7.9
/// 과도기 등가 테스트 승계). 검증 포인트(§5.7): kivi_handles 비면 inert(directive 무시) /
/// sticky last-applied(같은 bits 재directive 미증가, 값 변경 재증가) / RestoreDefaults 는
/// `last_quant_bits` clear 만(16bit 복원 submit 없음 — partition `submit_partition_full` 과 비대칭).
#[test]
fn quant_directive_submits_one_shot_stage() {
    use argus_engine::kv::kivi_cache::KiviCache;
    use argus_engine::kv::kivi_format::KIVIFormat;

    // (A) inert: kivi_handles 비면 KvQuantDynamic 무시 (non-KIVI: Standard/Offload 경로).
    let registry_inert = Arc::new(PipelineRegistry::new());
    let mut disp_inert = CommandDispatcher::new(
        Arc::clone(&registry_inert),
        Vec::new(),
        None,
        Vec::new(),
        None,
        None,
        None,
        None,
        Vec::new(),                 // 빈 kivi_handles → inert
        None,                       // report_tx: AB-5
        Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
        Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
    );
    disp_inert.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 4 }]);
    assert_eq!(
        registry_inert.len(),
        0,
        "kivi_handles 비면 inert — directive 무시"
    );

    // (B) configured: CPU KiviCache(bits=16 initial -- --kv-dynamic-quant 진입 동형).
    // head_dim/residual = QKKV 배수.
    let kivi_handles: Vec<Arc<KIVIFormat>> = (0..2)
        .map(|i| {
            Arc::new(KIVIFormat::new(
                i,
                KiviCache::new_with_bits(1, 32, 128, 32, 16),
            ))
        })
        .collect();
    let registry = Arc::new(PipelineRegistry::new());
    let mut disp = CommandDispatcher::new(
        Arc::clone(&registry),
        Vec::new(),
        None,
        Vec::new(),
        None,
        None,
        None,
        None,
        kivi_handles,
        None,                       // report_tx: AB-5
        Arc::new(Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
        Arc::new(Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
    );

    // 새 bits → submit 1.
    disp.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 4 }]);
    assert_eq!(registry.len(), 1, "첫 quant directive → OneShot submit");
    // 같은 bits 반복 → 미증가 (sticky last-applied).
    disp.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 4 }]);
    assert_eq!(registry.len(), 1, "같은 bits 반복 — 재submit 없음");
    // 값 변경 → 재증가.
    disp.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 8 }]);
    assert_eq!(registry.len(), 2, "bits 변경 → 새 OneShot submit");
    // RestoreDefaults → last_quant_bits clear 만 (16bit 복원 submit 없음 — registry 불변).
    disp.dispatch(vec![EngineCommand::RestoreDefaults]);
    assert_eq!(
        registry.len(),
        2,
        "RestoreDefaults → 16bit 복원 submit 없음 (partition 과 비대칭)"
    );
    // 재무장: 같은 bits(8)도 last=None reset 후라 재적용된다.
    disp.dispatch(vec![EngineCommand::KvQuantDynamic { target_bits: 8 }]);
    assert_eq!(
        registry.len(),
        3,
        "RestoreDefaults 후 재무장 → 어떤 bits 든 재적용"
    );
}

// ── 게이트 4: heartbeat 연속성 (pure poll 전환 후 송출·payload 등가) ──

/// `ResilienceAdapter::poll`(pure) 가 호출될 때마다 interval 경과 시 heartbeat 를 송출하고,
/// payload 의 kv_cache_tokens == held-handle.current_pos() 임을 검증한다 (매핑 문서 4.4).
#[test]
fn heartbeat_continuity_via_held_handle() {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let mut exec = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(10), // 짧은 interval 로 heartbeat 유도
    );
    exec.set_running();
    // throughput EMA 적재 (actual_throughput != 0 검증용).
    exec.on_token_generated();
    std::thread::sleep(Duration::from_millis(15));
    exec.on_token_generated();

    let mut adapter = ResilienceAdapter::new(exec);
    // held-handle 주입 — heartbeat snapshot 의 kv_cache_tokens 출처.
    let handle = make_handle(100);
    let h: Arc<dyn KVCacheFormat> = handle.clone();
    adapter.set_kv_handle(h);

    // interval 경과 후 pure poll → heartbeat 송출.
    std::thread::sleep(Duration::from_millis(15));
    let cmds = adapter.poll().unwrap();
    assert!(cmds.is_empty(), "directive 없음 → 빈 command vec");

    // heartbeat 수신 + payload 검증.
    let mut hb = None;
    while let Ok(msg) = resp_rx.try_recv() {
        if let EngineMessage::Heartbeat(status) = msg {
            hb = Some(status);
        }
    }
    let status = hb.expect("interval 경과 후 heartbeat 송출되어야 함");
    assert_eq!(status.active_device, "cpu");
    assert_eq!(status.state, EngineState::Running);
    // (3) kv_cache_tokens == held-handle.current_pos() — held-handle query 전환 핵심 가드.
    assert_eq!(
        status.kv_cache_tokens,
        handle.current_pos(),
        "heartbeat kv_cache_tokens == held-handle.current_pos()"
    );
    assert_eq!(status.kv_cache_tokens, 100);
    // (2) actual_throughput != 0 (EMA 적재 확인).
    assert!(
        status.actual_throughput > 0.0,
        "throughput EMA 적재 — actual_throughput != 0"
    );

    drop(cmd_tx); // 미사용 경고 억제
}

/// directive drain 등가 — pure poll 이 도착한 command 를 그대로 반환하고, 각 directive 에 Ok
/// 응답을 송출한다 (v1 apply_command 가 항상 Ok 였으므로 등가).
#[test]
fn pure_poll_drains_commands_and_acks() {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let exec = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_secs(3600),
    );
    let mut adapter = ResilienceAdapter::new(exec);

    cmd_tx
        .send(ManagerMessage::Directive(EngineDirective {
            seq_id: 7,
            commands: vec![
                EngineCommand::Throttle { delay_ms: 30 },
                EngineCommand::RequestQcf,
            ],
        }))
        .unwrap();

    let cmds = adapter.poll().unwrap();
    assert_eq!(cmds.len(), 2, "drain 한 command 2건 반환");
    assert!(matches!(cmds[0], EngineCommand::Throttle { delay_ms: 30 }));
    assert!(matches!(cmds[1], EngineCommand::RequestQcf));

    // Ok 응답 송출 (seq_id 7, 2 results).
    let resp = resp_rx.recv().unwrap();
    match resp {
        EngineMessage::Response(r) => {
            assert_eq!(r.seq_id, 7);
            assert_eq!(r.results.len(), 2);
        }
        _ => panic!("Expected Response"),
    }
}

// ── β-6 commit C: TickStage 경유 heartbeat token count 등가 ──

/// `TickStage`(PostSample) 가 N회 발화하면, v1 `on_token_generated` N회 직접 호출과 동일하게
/// executor throughput EMA 가 적재되어 heartbeat `actual_throughput`(token count 채널)이 채워진다.
/// 구 `TokenTickSink.on_token_generated` 호출 == 신 TickStage PostSample 발화 등가 (mock 없이
/// 실 ResilienceAdapter + executor 채널로 검증).
#[test]
fn tick_stage_drives_heartbeat_throughput_via_post_sample() {
    use argus_engine::pipeline::{LifecyclePhase, PipelineStage, Pressure, StageContext, StepInfo};
    use argus_engine::stages::system::tick::TickStage;

    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let mut exec = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(10),
    );
    exec.set_running();
    let adapter = Arc::new(Mutex::new(ResilienceAdapter::new(exec)));

    // TickStage 가 공유 adapter 로 per-token tick. PostSample 2회 발화 → throughput EMA 적재.
    let stage = TickStage::new(Arc::clone(&adapter));
    let mut profiler = argus_engine::observability::profile::OpProfiler::new();
    let step = StepInfo {
        pos: 0,
        decode_step: 0,
        pressure: Pressure::new(0),
        prev_token: 0,
    };

    std::thread::sleep(Duration::from_millis(2));
    {
        let mut ctx = StageContext {
            step,
            profiler: &mut profiler,
        };
        stage
            .on_phase(&LifecyclePhase::PostSample, &mut ctx)
            .unwrap();
    }
    std::thread::sleep(Duration::from_millis(2));
    {
        let mut ctx = StageContext {
            step,
            profiler: &mut profiler,
        };
        stage
            .on_phase(&LifecyclePhase::PostSample, &mut ctx)
            .unwrap();
    }

    // interval 경과 후 heartbeat 송출 → token count 채널(actual_throughput) 검증.
    std::thread::sleep(Duration::from_millis(15));
    adapter
        .lock()
        .unwrap()
        .executor_mut()
        .send_heartbeat_if_due(&KVSnapshot::default());

    let mut throughput = 0.0;
    while let Ok(EngineMessage::Heartbeat(status)) = resp_rx.try_recv() {
        throughput = status.actual_throughput;
    }
    assert!(
        throughput > 0.0,
        "TickStage PostSample 2회 발화로 heartbeat throughput 채널 적재 (v1 tick 등가)"
    );

    drop(cmd_tx);
}
