//! ENG-DAT-C05: KvStreaming protocol path spec tests.
//!
//! KvStreaming EngineCommand -> EvictPlan 생성, streaming_params 전달,
//! active_actions / available_actions 포함 검증.

use argus_engine::resilience::{EvictMethod, KVSnapshot};
use argus_shared::{CommandResult, EngineCommand, EngineMessage, ResourceLevel};

use super::helpers::{empty_snap, make_executor, send_directive};

// ── ENG-DAT-C05-01: KvStreaming produces EvictPlan with Streaming method ──

#[test]
fn test_eng_dat_c05_01_kv_streaming_produces_evict_plan() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(
        &tx,
        1,
        vec![EngineCommand::KvStreaming {
            sink_size: 4,
            window_size: 256,
        }],
    );

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("KvStreaming must produce an evict plan");
    assert_eq!(evict.method, EvictMethod::Streaming);
    assert_eq!(evict.level, ResourceLevel::Critical);
    assert!((evict.target_ratio - 0.0).abs() < f32::EPSILON);
}

// ── ENG-DAT-C05-02: streaming_params correctly forwarded ──

#[test]
fn test_eng_dat_c05_02_streaming_params_forwarded() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(
        &tx,
        1,
        vec![EngineCommand::KvStreaming {
            sink_size: 8,
            window_size: 512,
        }],
    );

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("evict plan");
    let params = evict
        .streaming_params
        .expect("streaming_params must be Some for Streaming method");
    assert_eq!(params.sink_size, 8);
    assert_eq!(params.window_size, 512);
}

// ── ENG-DAT-C05-03: KvStreaming returns CommandResult::Ok ──

#[test]
fn test_eng_dat_c05_03_kv_streaming_returns_ok() {
    let (mut executor, tx, rx) = make_executor();

    send_directive(
        &tx,
        42,
        vec![EngineCommand::KvStreaming {
            sink_size: 4,
            window_size: 256,
        }],
    );

    executor.poll(&empty_snap());

    let msg = rx.recv().unwrap();
    match msg {
        EngineMessage::Response(r) => {
            assert_eq!(r.seq_id, 42);
            assert_eq!(r.results.len(), 1);
            assert!(
                matches!(r.results[0], CommandResult::Ok),
                "KvStreaming must return Ok, got {:?}",
                r.results[0]
            );
        }
        other => panic!("Expected Response, got {:?}", other),
    }
}

// ── ENG-DAT-C05-04: active_actions includes kv_evict_streaming ──

#[test]
fn test_eng_dat_c05_04_active_actions_includes_streaming() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(
        &tx,
        1,
        vec![EngineCommand::KvStreaming {
            sink_size: 4,
            window_size: 256,
        }],
    );

    executor.poll(&empty_snap());

    assert!(
        executor
            .active_actions()
            .contains(&"kv.evict_streaming".to_string()),
        "active_actions must include kv_evict_streaming after KvStreaming command"
    );
}

// ── ENG-DAT-C05-05: available_actions includes kv_evict_streaming when policy != none ──

#[test]
fn test_eng_dat_c05_05_available_actions_includes_streaming() {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
    let (resp_tx, resp_rx) = std::sync::mpsc::channel();
    let mut executor = argus_engine::resilience::CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        std::time::Duration::from_millis(10),
    );
    executor.set_running();

    // Wait for heartbeat interval
    std::thread::sleep(std::time::Duration::from_millis(20));

    let snap = KVSnapshot {
        total_bytes: 1024,
        total_tokens: 100,
        capacity: 2048,
        protected_prefix: 4,
        kv_dtype: "f16".to_string(),
        eviction_policy: "streaming".to_string(),
        skip_ratio: 0.0,
    };
    executor.poll(&snap);

    // Find heartbeat message
    let mut found = false;
    for _ in 0..5 {
        if let Ok(EngineMessage::Heartbeat(status)) = resp_rx.try_recv() {
            assert!(
                status
                    .available_actions
                    .contains(&"kv.evict_streaming".to_string()),
                "Heartbeat available_actions must include kv_evict_streaming, got: {:?}",
                status.available_actions
            );
            found = true;
            break;
        }
    }
    assert!(
        found,
        "Should have received heartbeat with available_actions"
    );

    drop(cmd_tx);
}

// ── ENG-DAT-C05-06: KvStreaming supersedes prior eviction ──

#[test]
fn test_eng_dat_c05_06_streaming_supersedes_prior_eviction() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }]);
    send_directive(
        &tx,
        2,
        vec![EngineCommand::KvStreaming {
            sink_size: 4,
            window_size: 256,
        }],
    );

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("evict plan");
    assert_eq!(
        evict.method,
        EvictMethod::Streaming,
        "Later KvStreaming must supersede earlier KvEvictH2o"
    );
}

// ── ENG-DAT-C05-07: H2O/Sliding have streaming_params = None ──

#[test]
fn test_eng_dat_c05_07_non_streaming_has_no_streaming_params() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }]);

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("evict plan");
    assert!(
        evict.streaming_params.is_none(),
        "H2o evict plan must have streaming_params = None"
    );
}
