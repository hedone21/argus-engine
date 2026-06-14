//! SEQ-011 ~ SEQ-035: 핸드셰이크 및 정상 운영 시퀀스 (Engine-side)
//!
//! 커버: SEQ-020 (Handshake), SEQ-030 (Steady-State)
//! CommandExecutor의 capability 전송, heartbeat 발생, 빈 inbox 처리,
//! EOF disconnect 내성, MessageLoop E2E 파이프라인을 검증한다.

use std::sync::mpsc;
use std::time::Duration;

use argus_engine::resilience::{CommandExecutor, KVSnapshot, MessageLoop, MockTransport};
use argus_shared::{
    EngineCapability, EngineCommand, EngineDirective, EngineMessage, EngineState, ManagerMessage,
};

use super::helpers::{empty_snap, make_executor};

// ═══════════════════════════════════════════════════════════════
// SEQ-022: Capability 전송
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_022_send_capability() {
    let (executor, _tx, resp_rx) = make_executor();

    // send_capability() 호출
    executor.send_capability(EngineCapability {
        available_devices: vec!["cpu".into(), "opencl".into()],
        active_device: "cpu".into(),
        max_kv_tokens: 2048,
        bytes_per_kv_token: 256,
        num_layers: 16,
        ..Default::default()
    });

    // resp_rx에서 EngineMessage::Capability 수신 확인
    let msg = resp_rx.recv().unwrap();
    match msg {
        EngineMessage::Capability(cap) => {
            assert_eq!(cap.available_devices.len(), 2);
            assert_eq!(cap.active_device, "cpu");
            assert_eq!(cap.max_kv_tokens, 2048);
            assert_eq!(cap.bytes_per_kv_token, 256);
            assert_eq!(cap.num_layers, 16);
        }
        other => panic!("Capability 메시지를 기대했으나 {:?} 수신", other),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-024: 첫 Heartbeat
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_024_first_heartbeat() {
    // heartbeat_interval=10ms로 짧게 설정하여 heartbeat 유도
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(10),
    );
    executor.set_running();

    // heartbeat 간격이 지나도록 대기
    std::thread::sleep(Duration::from_millis(20));

    let snap = KVSnapshot {
        total_bytes: 25600,
        total_tokens: 100,
        capacity: 2048,
        protected_prefix: 4,
        kv_dtype: "f16".to_string(),
        eviction_policy: "none".to_string(),
        skip_ratio: 0.0,
    };
    let _plan = executor.poll(&snap);

    // Heartbeat 수신 확인
    let msg = resp_rx.recv().unwrap();
    match msg {
        EngineMessage::Heartbeat(status) => {
            // EngineStatus 필드 존재 확인
            assert_eq!(status.state, EngineState::Running);
            assert_eq!(status.active_device, "cpu");
            assert_eq!(status.kv_cache_tokens, 100);
            assert!(status.kv_cache_bytes > 0);
        }
        other => panic!("Heartbeat 메시지를 기대했으나 {:?} 수신", other),
    }

    drop(cmd_tx); // 미사용 경고 억제
}

// ═══════════════════════════════════════════════════════════════
// SEQ-030: Heartbeat 16 필드
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_030_heartbeat_16_fields() {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(10),
    );
    executor.set_running();

    std::thread::sleep(Duration::from_millis(20));

    let snap = KVSnapshot {
        total_bytes: 51200,
        total_tokens: 200,
        capacity: 2048,
        protected_prefix: 8,
        kv_dtype: "f16".to_string(),
        eviction_policy: "h2o".to_string(),
        skip_ratio: 0.1,
    };
    let _plan = executor.poll(&snap);

    let msg = resp_rx.recv().unwrap();
    match msg {
        EngineMessage::Heartbeat(status) => {
            // EngineStatus 16 필드 모두 접근 가능 확인
            let _: &str = &status.active_device; // 1
            let _: argus_shared::ResourceLevel = status.compute_level; // 2
            let _: f32 = status.actual_throughput; // 3
            let _: argus_shared::ResourceLevel = status.memory_level; // 4
            let _: u64 = status.kv_cache_bytes; // 5
            let _: usize = status.kv_cache_tokens; // 6
            let _: f32 = status.kv_cache_utilization; // 7
            let _: f32 = status.memory_lossless_min; // 8
            let _: f32 = status.memory_lossy_min; // 9
            let _: EngineState = status.state; // 10
            let _: usize = status.tokens_generated; // 11
            let _: &Vec<String> = &status.available_actions; // 12
            let _: &Vec<String> = &status.active_actions; // 13
            let _: &str = &status.eviction_policy; // 14
            let _: &str = &status.kv_dtype; // 15
            let _: f32 = status.skip_ratio; // 16

            // 값 정합성 확인
            assert_eq!(status.kv_cache_tokens, 200);
            assert_eq!(status.kv_cache_bytes, 51200);
            assert_eq!(status.eviction_policy, "h2o");
            assert_eq!(status.kv_dtype, "f16");
            assert!((status.skip_ratio - 0.1).abs() < f32::EPSILON);
            // eviction_policy가 "none"이 아니면 kv_evict 액션이 available에 포함
            assert!(
                status
                    .available_actions
                    .contains(&"kv.evict_h2o".to_string())
            );
        }
        other => panic!("Heartbeat 메시지를 기대했으나 {:?} 수신", other),
    }

    drop(cmd_tx);
}

// ═══════════════════════════════════════════════════════════════
// SEQ-034: poll() 호출 타이밍에 따른 Heartbeat 발생
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_034_heartbeat_timing() {
    let (_cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(100),
    );
    executor.set_running();

    // 즉시 poll() 호출 -> heartbeat 없어야 함 (interval 미도달)
    let _plan = executor.poll(&empty_snap());
    assert!(
        resp_rx.try_recv().is_err(),
        "interval 미도달 시 heartbeat가 발생하면 안 됨"
    );

    // 100ms 이상 대기 후 poll() -> heartbeat 발생해야 함
    std::thread::sleep(Duration::from_millis(110));
    let _plan = executor.poll(&empty_snap());
    let msg = resp_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(
        matches!(msg, EngineMessage::Heartbeat(_)),
        "interval 도달 후 heartbeat가 발생해야 함"
    );
}

// ═══════════════════════════════════════════════════════════════
// SEQ-011: Directive 없이 정상 운영
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_011_no_directive_default_plan() {
    let (mut executor, _tx, _rx) = make_executor();

    // Directive를 보내지 않고 poll() 호출
    let plan = executor.poll(&empty_snap());

    // default plan 확인: evict=None, throttle=0, suspended=false
    assert!(plan.evict.is_none(), "evict는 None이어야 함");
    assert_eq!(plan.throttle_delay_ms, 0, "throttle은 0이어야 함");
    assert!(!plan.suspended, "suspended=false이어야 함");
    assert!(!plan.resumed, "resumed=false이어야 함");
    assert!(
        plan.switch_device.is_none(),
        "switch_device는 None이어야 함"
    );
    assert!(!plan.restore_defaults, "restore_defaults=false이어야 함");
}

// ═══════════════════════════════════════════════════════════════
// SEQ-013: EOF disconnect
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_013_eof_disconnect_graceful() {
    let (mut executor, tx, _rx) = make_executor();

    // sender drop으로 EOF 시뮬레이션
    drop(tx);

    // poll() 호출 시 패닉 없음, default plan 반환
    let plan = executor.poll(&empty_snap());
    assert!(plan.evict.is_none());
    assert_eq!(plan.throttle_delay_ms, 0);
    assert!(!plan.suspended);

    // 반복 호출도 안전해야 함
    let plan2 = executor.poll(&empty_snap());
    assert!(!plan2.suspended);
}

// ═══════════════════════════════════════════════════════════════
// SEQ-021: MessageLoop E2E
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_021_message_loop_e2e() {
    // MockTransport에 2개 Directive를 미리 로드
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

    // poll()로 plan에 반영 확인
    let plan = executor.poll(&empty_snap());
    assert!(
        plan.evict.is_some() || plan.throttle_delay_ms > 0,
        "MockTransport E2E 파이프라인에서 액션이 있어야 함"
    );
}

// ═══════════════════════════════════════════════════════════════
// SEQ-035: 빈 inbox -> default plan
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_035_empty_inbox_default_plan() {
    let (mut executor, _tx, _rx) = make_executor();

    // 메시지 없이 여러 번 poll() -> 매번 default plan
    for _ in 0..5 {
        let plan = executor.poll(&empty_snap());
        assert!(plan.evict.is_none());
        assert!(!plan.suspended);
        assert!(!plan.resumed);
        assert!(plan.switch_device.is_none());
    }
}
