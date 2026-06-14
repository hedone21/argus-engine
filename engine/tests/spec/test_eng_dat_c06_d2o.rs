//! ENG-DAT-C06: KvMergeD2o protocol path spec tests.
//!
//! KvMergeD2o EngineCommand -> EvictPlan { D2o, keep_ratio } 생성,
//! active_actions / available_actions 포함 검증.

use argus_engine::resilience::{EvictMethod, KVSnapshot};
use argus_shared::{CommandResult, EngineCommand, EngineMessage, ResourceLevel};

use super::helpers::{empty_snap, make_executor, send_directive};

// ── ENG-DAT-C06-01: KvMergeD2o produces EvictPlan with D2o method ──

#[test]
fn test_eng_dat_c06_01_kv_merge_d2o_produces_evict_plan() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::KvMergeD2o { keep_ratio: 0.75 }]);

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("KvMergeD2o must produce an evict plan");
    assert_eq!(evict.method, EvictMethod::D2o);
    assert_eq!(evict.level, ResourceLevel::Critical);
    assert!((evict.target_ratio - 0.75).abs() < f32::EPSILON);
}

// ── ENG-DAT-C06-02: KvMergeD2o has streaming_params = None ──

#[test]
fn test_eng_dat_c06_02_d2o_has_no_streaming_params() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::KvMergeD2o { keep_ratio: 0.6 }]);

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("evict plan");
    assert!(
        evict.streaming_params.is_none(),
        "D2o evict plan must have streaming_params = None"
    );
}

// ── ENG-DAT-C06-03: KvMergeD2o returns CommandResult::Ok ──

#[test]
fn test_eng_dat_c06_03_kv_merge_d2o_returns_ok() {
    let (mut executor, tx, rx) = make_executor();

    send_directive(
        &tx,
        42,
        vec![EngineCommand::KvMergeD2o { keep_ratio: 0.75 }],
    );

    executor.poll(&empty_snap());

    let msg = rx.recv().unwrap();
    match msg {
        EngineMessage::Response(r) => {
            assert_eq!(r.seq_id, 42);
            assert_eq!(r.results.len(), 1);
            assert!(
                matches!(r.results[0], CommandResult::Ok),
                "KvMergeD2o must return Ok, got {:?}",
                r.results[0]
            );
        }
        other => panic!("Expected Response, got {:?}", other),
    }
}

// ── ENG-DAT-C06-04: active_actions includes kv_merge_d2o ──

#[test]
fn test_eng_dat_c06_04_active_actions_includes_d2o() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::KvMergeD2o { keep_ratio: 0.75 }]);

    executor.poll(&empty_snap());

    assert!(
        executor
            .active_actions()
            .contains(&"kv.merge_d2o".to_string()),
        "active_actions must include kv_merge_d2o after KvMergeD2o command"
    );
}

// ── ENG-DAT-C06-05: available_actions includes kv_merge_d2o when policy != none ──

#[test]
fn test_eng_dat_c06_05_available_actions_includes_d2o() {
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
        eviction_policy: "d2o".to_string(),
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
                    .contains(&"kv.merge_d2o".to_string()),
                "Heartbeat available_actions must include kv_merge_d2o, got: {:?}",
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

// ── ENG-DAT-C06-06: KvMergeD2o supersedes prior eviction ──

#[test]
fn test_eng_dat_c06_06_d2o_supersedes_prior_eviction() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }]);
    send_directive(&tx, 2, vec![EngineCommand::KvMergeD2o { keep_ratio: 0.75 }]);

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("evict plan");
    assert_eq!(
        evict.method,
        EvictMethod::D2o,
        "Later KvMergeD2o must supersede earlier KvEvictH2o"
    );
    assert!((evict.target_ratio - 0.75).abs() < f32::EPSILON);
}

// ── ENG-DAT-C06-07: keep_ratio correctly forwarded ──

#[test]
fn test_eng_dat_c06_07_keep_ratio_forwarded() {
    let (mut executor, tx, _rx) = make_executor();

    send_directive(&tx, 1, vec![EngineCommand::KvMergeD2o { keep_ratio: 0.42 }]);

    let plan = executor.poll(&empty_snap());
    let evict = plan.evict.expect("evict plan");
    assert_eq!(evict.method, EvictMethod::D2o);
    assert!(
        (evict.target_ratio - 0.42).abs() < f32::EPSILON,
        "target_ratio must match keep_ratio from directive"
    );
}
