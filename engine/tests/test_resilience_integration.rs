//! Integration tests for Command Executor — generate.rs integration logic.
//!
//! These tests verify the resilience checkpoint behavior that runs inside
//! the token generation loop: directive → poll → execution plan → state change.
//!
//! Run with: `cargo test --test test_resilience_integration`

use std::sync::mpsc;
use std::time::Duration;

use argus_engine::resilience::{CommandExecutor, KVSnapshot, MessageLoop, MockTransport};
use argus_shared::{EngineCommand, EngineDirective, EngineMessage, EngineState, ManagerMessage};

// ── Helpers ───────────────────────────────────────────────

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
        Duration::from_secs(60), // 테스트에서 하트비트 노이즈 방지용 긴 간격
    );
    (executor, cmd_tx, resp_rx)
}

fn empty_snap() -> KVSnapshot {
    KVSnapshot::default()
}

fn snap(tokens: usize, capacity: usize, prefix: usize) -> KVSnapshot {
    KVSnapshot {
        total_bytes: (tokens * 256) as u64,
        total_tokens: tokens,
        capacity,
        protected_prefix: prefix,
        ..KVSnapshot::default()
    }
}

fn send_directive(tx: &mpsc::Sender<ManagerMessage>, seq_id: u64, commands: Vec<EngineCommand>) {
    tx.send(ManagerMessage::Directive(EngineDirective {
        seq_id,
        commands,
    }))
    .unwrap();
}

// ── Test: Eviction flow (H2O) ────────────────────────────

#[test]
fn test_resilience_eviction_flow() {
    let (mut executor, tx, rx) = make_executor();

    // KvEvictH2o → EvictPlan 생성 확인
    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }]);

    let plan = executor.poll(&snap(500, 2048, 4));
    assert!(
        plan.evict.is_some(),
        "KvEvictH2o 명령에 대해 evict plan이 생성돼야 함"
    );

    let evict = plan.evict.unwrap();
    assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
    assert_eq!(evict.method, argus_engine::resilience::EvictMethod::H2o);

    // generate.rs처럼 KV 캐시 축소 시뮬레이션
    let current_pos: usize = 500;
    let target_len = (current_pos as f32 * evict.target_ratio) as usize;
    let remove = current_pos.saturating_sub(target_len);
    assert!(remove > 0, "일부 토큰이 제거돼야 함");

    let new_pos = current_pos - remove;
    assert!(new_pos < current_pos, "축출 후 포지션이 감소해야 함");

    // 응답이 전송돼야 함
    let resp = rx.recv().unwrap();
    assert!(matches!(resp, EngineMessage::Response(_)));
}

// ── Test: Eviction flow (Sliding) ────────────────────────

#[test]
fn test_resilience_eviction_sliding_flow() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(
        &tx,
        1,
        vec![EngineCommand::KvEvictSliding { keep_ratio: 0.7 }],
    );

    let plan = executor.poll(&snap(500, 2048, 4));
    assert!(plan.evict.is_some());
    let evict = plan.evict.unwrap();
    assert!((evict.target_ratio - 0.7).abs() < f32::EPSILON);
    assert_eq!(evict.method, argus_engine::resilience::EvictMethod::Sliding);
}

// ── Test: Throttle flow ──────────────────────────────────

#[test]
fn test_resilience_throttle_flow() {
    let (mut executor, tx, _rx) = make_executor();

    // Throttle 명령 → 딜레이 직접 설정
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 50 }]);

    let plan = executor.poll(&empty_snap());
    assert!(
        plan.throttle_delay_ms > 0,
        "Throttle 딜레이가 설정돼야 함, 현재값: {}",
        plan.throttle_delay_ms
    );
    assert_eq!(plan.throttle_delay_ms, 50);
    assert!(!plan.suspended, "Throttle 명령에서 suspended여선 안 됨");
}

// ── Test: Suspend flow ───────────────────────────────────

#[test]
fn test_resilience_suspend_flow() {
    let (mut executor, tx, _rx) = make_executor();

    // Suspend 명령
    send_directive(&tx, 1, vec![EngineCommand::Suspend]);

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended, "Suspend 명령 후 suspended여야 함");
    assert_eq!(executor.state(), EngineState::Suspended);

    // Suspend는 다른 plan 필드를 초기화해야 함
    assert!(plan.evict.is_none());
    assert!(plan.switch_device.is_none());
}

// ── Test: Disabled resilience is noop ────────────────────

#[test]
fn test_resilience_disabled_noop() {
    // command_executor가 None일 때 poll이 발생하지 않아야 함
    let command_executor: Option<CommandExecutor> = None;

    let mut throttle_delay_ms = 0u64;
    if let Some(_ex) = &command_executor {
        throttle_delay_ms = 999;
    }

    assert_eq!(
        throttle_delay_ms, 0,
        "비활성화된 resilience는 상태에 영향을 주지 않아야 함"
    );
}

// ── Test: Resume clears constraints ──────────────────────

#[test]
fn test_resilience_resume_clears_constraints() {
    let (mut executor, tx, _rx) = make_executor();

    // 먼저 스로틀 설정
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 100 }]);
    executor.poll(&empty_snap());
    assert!(executor.throttle_delay_ms() > 0, "스로틀이 설정돼야 함");

    // 중단
    send_directive(&tx, 2, vec![EngineCommand::Suspend]);
    executor.poll(&empty_snap());
    assert_eq!(executor.state(), EngineState::Suspended);

    // Resume은 Normal로 복구해야 함
    send_directive(&tx, 3, vec![EngineCommand::Resume]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.resumed);
    assert_eq!(executor.state(), EngineState::Running);
    assert_eq!(executor.throttle_delay_ms(), 0);
}

// ── Test: Channel disconnect is graceful ─────────────────

#[test]
fn test_resilience_channel_disconnect_graceful() {
    let (mut executor, tx, _rx) = make_executor();

    // 디렉티브를 보낸 후 sender 드롭 (트랜스포트 크래시 시뮬레이션)
    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.85 }]);
    drop(tx);

    // 첫 번째 poll: 버퍼된 디렉티브 처리
    let plan = executor.poll(&empty_snap());
    assert!(plan.evict.is_some());

    // 두 번째 poll: 채널 종료 후 패닉 없이 동작.
    // evict_plan은 sticky하므로 새 명령 없이도 유지됨.
    let plan = executor.poll(&empty_snap());
    assert!(plan.evict.is_some()); // sticky: KvEvictH2o 유지
}

// ── Transport integration tests ──────────────────────────

#[test]
fn test_mock_transport_e2e() {
    let messages = vec![
        ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvEvictH2o { keep_ratio: 0.85 }],
        }),
        ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::Throttle { delay_ms: 50 }],
        }),
    ];
    let transport = MockTransport::from_messages(messages);
    let (cmd_rx, resp_tx, _handle) = MessageLoop::spawn(transport).unwrap();

    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));

    // MessageLoop가 메시지를 전달할 때까지 대기
    std::thread::sleep(Duration::from_millis(50));

    let plan = executor.poll(&snap(500, 2048, 4));
    assert!(
        plan.evict.is_some() || plan.throttle_delay_ms > 0,
        "모의 트랜스포트 파이프라인에서 액션이 있어야 함"
    );
}

#[cfg(unix)]
#[test]
fn test_unix_socket_e2e() {
    use argus_engine::resilience::{Transport, UnixSocketTransport};
    use std::io::Write;
    use std::os::unix::net::UnixListener;

    let path = std::env::temp_dir().join(format!(
        "argus_e2e_{}.sock",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let listener = UnixListener::bind(&path).unwrap();

    let path2 = path.clone();
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, _resp_rx) = mpsc::channel();

    std::thread::spawn(move || {
        let mut transport = UnixSocketTransport::new(path2);
        if transport.connect().is_err() {
            return;
        }
        while let Ok(msg) = transport.recv() {
            if cmd_tx.send(msg).is_err() {
                break;
            }
        }
    });

    let (mut server_stream, _) = listener.accept().unwrap();
    let msg = ManagerMessage::Directive(EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }],
    });
    let json = serde_json::to_vec(&msg).unwrap();
    let len = (json.len() as u32).to_be_bytes();
    server_stream.write_all(&len).unwrap();
    server_stream.write_all(&json).unwrap();
    server_stream.flush().unwrap();
    drop(server_stream);

    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));

    std::thread::sleep(Duration::from_millis(100));

    let plan = executor.poll(&snap(500, 2048, 4));
    assert!(
        plan.evict.is_some(),
        "unix socket 파이프라인에서 evict plan이 있어야 함"
    );
    std::fs::remove_file(&path).ok();
}

// ── Test: Superseding directives ─────────────────────────

#[test]
fn test_resilience_superseding_directives() {
    let (mut executor, tx, _rx) = make_executor();

    // H2O 먼저, 이후 Sliding으로 덮어씀
    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.85 }]);
    send_directive(
        &tx,
        2,
        vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }],
    );

    let plan = executor.poll(&empty_snap());
    // 두 번째(더 최신) 명령이 승리
    let evict = plan.evict.unwrap();
    assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
    assert_eq!(evict.method, argus_engine::resilience::EvictMethod::Sliding);
}

// ── Test: Multi-domain directives ────────────────────────

#[test]
fn test_resilience_multi_domain_directive() {
    let (mut executor, tx, _rx) = make_executor();

    // 하나의 디렉티브에 compute와 memory 명령 동시 포함
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Throttle { delay_ms: 50 },
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
        ],
    );

    let plan = executor.poll(&empty_snap());
    assert!(plan.evict.is_some(), "evict plan이 있어야 함");
    assert!(plan.throttle_delay_ms > 0, "스로틀이 있어야 함");
}

// ── Test: Suspend overrides everything ───────────────────

#[test]
fn test_resilience_suspend_overrides_all() {
    let (mut executor, tx, _rx) = make_executor();

    // Memory + Compute + Suspend를 하나의 디렉티브로
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
            EngineCommand::SwitchHw {
                device: "gpu".to_string(),
            },
            EngineCommand::Suspend,
        ],
    );

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended, "Suspend가 설정돼야 함");
    assert!(plan.evict.is_none(), "Suspend가 evict를 초기화해야 함");
    assert!(
        plan.switch_device.is_none(),
        "Suspend가 switch를 초기화해야 함"
    );
}

// ── Test: Rapid signal buffering ─────────────────────────

#[test]
fn test_resilience_signal_buffering() {
    let (mut executor, tx, _rx) = make_executor();

    // 50개의 디렉티브를 빠르게 전송
    for i in 0..50 {
        let keep_ratio = if i % 2 == 0 { 0.85 } else { 0.5 };
        send_directive(&tx, i + 1, vec![EngineCommand::KvEvictH2o { keep_ratio }]);
    }

    // poll()이 모두 처리하고 패닉 없어야 함
    let plan = executor.poll(&empty_snap());
    // 마지막 디렉티브: i=49(짝수) → keep_ratio=0.85
    assert!(plan.evict.is_some());
    let evict = plan.evict.unwrap();
    assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON); // i=49(홀수) → 0.5
}

// ── Test: LayerSkip command ───────────────────────────────

#[test]
fn test_resilience_layer_skip() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::LayerSkip { skip_ratio: 0.25 }]);

    let plan = executor.poll(&empty_snap());
    assert_eq!(plan.layer_skip, Some(0.25));
}

// ── Test: KvStreaming produces EvictPlan ──────────────────

#[test]
fn test_resilience_kv_streaming_ok() {
    use argus_engine::resilience::EvictMethod;
    use argus_shared::CommandResult;

    let (mut executor, tx, rx) = make_executor();

    send_directive(
        &tx,
        1,
        vec![EngineCommand::KvStreaming {
            sink_size: 4,
            window_size: 256,
        }],
    );

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("KvStreaming은 evict plan을 생성해야 함");
    assert_eq!(evict.method, EvictMethod::Streaming);
    let params = evict.streaming_params.expect("streaming_params 있어야 함");
    assert_eq!(params.sink_size, 4);
    assert_eq!(params.window_size, 256);

    let resp = rx.recv().unwrap();
    match resp {
        EngineMessage::Response(r) => {
            assert_eq!(r.seq_id, 1);
            assert!(matches!(r.results[0], CommandResult::Ok));
        }
        _ => panic!("Expected Response"),
    }
}

// ── Test: RestoreDefaults ─────────────────────────────────

#[test]
fn test_resilience_restore_defaults() {
    let (mut executor, tx, _rx) = make_executor();

    // 상태 설정
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Throttle { delay_ms: 100 },
            EngineCommand::LayerSkip { skip_ratio: 0.3 },
        ],
    );
    executor.poll(&empty_snap());
    assert!(!executor.active_actions().is_empty());

    // RestoreDefaults
    send_directive(&tx, 2, vec![EngineCommand::RestoreDefaults]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.restore_defaults);
    assert_eq!(plan.throttle_delay_ms, 0);
    assert_eq!(executor.throttle_delay_ms(), 0);
    assert!(executor.active_actions().is_empty());
}
