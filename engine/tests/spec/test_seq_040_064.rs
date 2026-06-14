//! SEQ-040 ~ SEQ-064: 에스컬레이션 및 디에스컬레이션 시퀀스 (Engine-side)
//!
//! Directive 수신 -> plan 반영, command-result 매핑, seq_id 일치,
//! RestoreDefaults 초기화, Directive drain, Suspend/Resume 사이클,
//! Rejected command 처리를 검증한다.

use std::time::Duration;

use argus_shared::{CommandResult, EngineCommand, EngineMessage, EngineState};

use super::helpers::{empty_snap, make_executor, send_directive};

// ═══════════════════════════════════════════════════════════════
// SEQ-047: Directive 수신 -> plan 반영
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_047_directive_to_plan() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }]);

    let plan = executor.poll(&empty_snap());
    assert!(plan.evict.is_some(), "KvEvictH2o -> evict plan 생성 확인");
    let evict = plan.evict.unwrap();
    assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
}

// ═══════════════════════════════════════════════════════════════
// SEQ-048: 각 command -> 각 result
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_048_each_command_each_result() {
    let (mut executor, tx, resp_rx) = make_executor();

    // 3개 commands를 하나의 Directive에 포함
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Throttle { delay_ms: 30 },
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
            EngineCommand::LayerSkip { skip_ratio: 0.2 },
        ],
    );

    let _plan = executor.poll(&empty_snap());

    let msg = resp_rx.recv().unwrap();
    match msg {
        EngineMessage::Response(r) => {
            assert_eq!(r.seq_id, 1);
            assert_eq!(
                r.results.len(),
                3,
                "3개 command에 대해 3개 result가 있어야 함"
            );
        }
        other => panic!("Response를 기대했으나 {:?} 수신", other),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-049: 1 Directive -> 1 Response, seq_id 일치
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_049_one_directive_one_response_seq_id_match() {
    let (mut executor, tx, resp_rx) = make_executor();

    // 3개 Directive (seq_id 1,2,3) 전송
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 10 }]);
    send_directive(&tx, 2, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.7 }]);
    send_directive(&tx, 3, vec![EngineCommand::LayerSkip { skip_ratio: 0.1 }]);

    let _plan = executor.poll(&empty_snap());

    // 3개 Response 수신 확인
    let mut received_seq_ids = Vec::new();
    for _ in 0..3 {
        let msg = resp_rx.recv().unwrap();
        match msg {
            EngineMessage::Response(r) => {
                // 각 results.len() == commands.len() (각 1개)
                assert_eq!(r.results.len(), 1);
                received_seq_ids.push(r.seq_id);
            }
            other => panic!("Response를 기대했으나 {:?} 수신", other),
        }
    }

    // seq_id 순서대로 1, 2, 3
    assert_eq!(received_seq_ids, vec![1, 2, 3]);
}

// ═══════════════════════════════════════════════════════════════
// SEQ-063: RestoreDefaults
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_063_restore_defaults() {
    let (mut executor, tx, _rx) = make_executor();

    // 먼저 액션 활성화
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Throttle { delay_ms: 50 },
            EngineCommand::LayerSkip { skip_ratio: 0.3 },
        ],
    );
    executor.poll(&empty_snap());
    assert!(
        !executor.active_actions().is_empty(),
        "RestoreDefaults 전 active_actions가 비어있지 않아야 함"
    );

    // RestoreDefaults -> 전부 초기화
    send_directive(&tx, 2, vec![EngineCommand::RestoreDefaults]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.restore_defaults, "restore_defaults=true이어야 함");
    assert_eq!(plan.throttle_delay_ms, 0, "throttle=0으로 초기화");
    assert!(
        executor.active_actions().is_empty(),
        "active_actions가 비어야 함"
    );
}

// ═══════════════════════════════════════════════════════════════
// SEQ-063: RestoreDefaults after multi-actions
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_063_restore_defaults_after_multi_actions() {
    let (mut executor, tx, _rx) = make_executor();

    // 여러 도메인 액션 활성화
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
            EngineCommand::Throttle { delay_ms: 100 },
            EngineCommand::SwitchHw {
                device: "opencl".to_string(),
            },
        ],
    );
    executor.poll(&empty_snap());

    // RestoreDefaults -> 전부 초기화
    send_directive(&tx, 2, vec![EngineCommand::RestoreDefaults]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.restore_defaults);
    assert_eq!(plan.throttle_delay_ms, 0);
    assert_eq!(executor.throttle_delay_ms(), 0);
    assert!(executor.active_actions().is_empty());
}

// ═══════════════════════════════════════════════════════════════
// SEQ-092: Directive drain (50개)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_092_directive_drain_50() {
    let (mut executor, tx, resp_rx) = make_executor();

    // 50개 Directive 전송
    for i in 1..=50u64 {
        send_directive(&tx, i, vec![EngineCommand::Throttle { delay_ms: i }]);
    }

    // poll() 1회로 모두 처리
    let _plan = executor.poll(&empty_snap());

    // resp_rx에서 50개 Response 수신
    let mut received_seq_ids = Vec::new();
    for _ in 0..50 {
        let msg = resp_rx
            .recv_timeout(Duration::from_millis(500))
            .expect("50개 Response 중 하나를 수신하지 못함");
        match msg {
            EngineMessage::Response(r) => {
                received_seq_ids.push(r.seq_id);
            }
            other => panic!("Response를 기대했으나 {:?} 수신", other),
        }
    }

    // 각 seq_id가 1..=50 범위 내 확인
    assert_eq!(received_seq_ids.len(), 50);
    for id in &received_seq_ids {
        assert!(
            (1..=50).contains(id),
            "seq_id {} 가 1..=50 범위를 벗어남",
            id
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Suspend/Resume 전체 사이클
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_suspend_resume_full_cycle() {
    let (mut executor, tx, _rx) = make_executor();

    // 1단계: Throttle 설정
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 100 }]);
    let plan = executor.poll(&empty_snap());
    assert_eq!(plan.throttle_delay_ms, 100);
    assert_eq!(executor.state(), EngineState::Idle);

    // 2단계: Suspend
    send_directive(&tx, 2, vec![EngineCommand::Suspend]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended, "suspended=true이어야 함");
    assert!(plan.evict.is_none(), "Suspend는 evict를 None으로 만듦");
    assert_eq!(executor.state(), EngineState::Suspended);

    // 3단계: Resume
    send_directive(&tx, 3, vec![EngineCommand::Resume]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.resumed, "resumed=true이어야 함");
    assert_eq!(plan.throttle_delay_ms, 0, "Resume 후 throttle=0");
    assert_eq!(executor.state(), EngineState::Running);
}

// ═══════════════════════════════════════════════════════════════
// SEQ-048: KvStreaming command → Ok + EvictPlan
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_048_kv_streaming_ok() {
    use argus_engine::resilience::EvictMethod;

    let (mut executor, tx, resp_rx) = make_executor();

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
    let params = evict
        .streaming_params
        .expect("streaming_params가 있어야 함");
    assert_eq!(params.sink_size, 4);
    assert_eq!(params.window_size, 256);

    let msg = resp_rx.recv().unwrap();
    match msg {
        EngineMessage::Response(r) => {
            assert_eq!(r.seq_id, 1);
            assert!(
                matches!(r.results[0], CommandResult::Ok),
                "KvStreaming은 Ok여야 함"
            );
        }
        other => panic!("Response를 기대했으나 {:?} 수신", other),
    }
}
