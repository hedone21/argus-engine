//! SEQ-095 ~ SEQ-098: QCF Request/Estimate sequence (Engine side)
//!
//! SEQ-095: Manager sends Directive([RequestQcf])
//! SEQ-096: Engine responds Ok, then sends separate QcfEstimate
//! SEQ-097: (Manager side — tested in manager/tests/spec/)
//! SEQ-098: (Manager side — timeout fallback)

use std::collections::HashMap;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use argus_shared::{CommandResult, EngineCommand, EngineMessage, QcfEstimate};

use super::helpers::{empty_snap, make_executor, send_directive};

// ═══════════════════════════════════════════════════════════════
// SEQ-095: RequestQcf Directive 수신 및 처리
// (AB-5 §5.8.6 승계: plan.request_qcf 단언 → dispatcher 송출 단언으로 교체)
// ═══════════════════════════════════════════════════════════════

/// AB-5 §5.8.6 gate 1: dispatcher 에 report_tx 주입 → RequestQcf dispatch 시 QcfEstimate 1회 송출.
#[test]
fn test_seq_095_dispatcher_sends_qcf_estimate_on_request_qcf() {
    use argus_engine::backend::Backend;
    use argus_engine::backend::cpu::CpuBackend;
    use argus_engine::buffer::DType;
    use argus_engine::kv::cache_manager::CacheManager;
    use argus_engine::kv::eviction::stage_registry::sliding_backed_policy;
    use argus_engine::kv::kv_cache::KVCache;
    use argus_engine::kv::standard_format::StandardFormat;
    use argus_engine::memory::host::shared::SharedBuffer;
    use argus_engine::resilience::sys_monitor::NoOpMonitor;
    use argus_engine::session::command_dispatcher::CommandDispatcher;
    use argus_engine::session::pipeline_registry::PipelineRegistry;
    use argus_engine::shape::Shape;
    use argus_engine::tensor::Tensor;
    use std::sync::Mutex;

    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 32;
    const MAX_SEQ: usize = 128;

    let total = MAX_SEQ * KV_HEADS * HEAD_DIM;
    let k_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
    let v_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
    let k = Tensor::new(shape.clone(), k_buf, backend.clone());
    let v = Tensor::new(shape, v_buf, backend);
    let mut cache = KVCache::new(k, v, MAX_SEQ);
    cache.current_pos = 60; // 충분히 채워서 compute 가능.
    let handle = Arc::new(StandardFormat::new(0, cache));

    let policy = sliding_backed_policy(10, 4);
    let cm = Arc::new(Mutex::new(CacheManager::new(
        policy,
        Box::new(NoOpMonitor),
        usize::MAX,
        0.3,
    )));

    let registry = Arc::new(PipelineRegistry::new());
    let (report_tx, report_rx) = mpsc::channel::<EngineMessage>();

    let mut disp = CommandDispatcher::new(
        Arc::clone(&registry),
        vec![handle],
        Some(cm),
        Vec::new(),
        None,
        None,
        None,
        None,
        Vec::new(),
        Some(report_tx), // AB-5: report_tx 주입 → RequestQcf 시 QcfEstimate 송출.
        Arc::new(std::sync::Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
        Arc::new(std::sync::Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
    );

    // dispatch RequestQcf → compute_and_send_qcf 경유 QcfEstimate 1회 송출.
    disp.dispatch(vec![EngineCommand::RequestQcf]);

    let msg = report_rx
        .recv_timeout(Duration::from_millis(200))
        .expect("QcfEstimate 1회 송출되어야 함");
    assert!(
        matches!(msg, EngineMessage::QcfEstimate(_)),
        "RequestQcf dispatch → QcfEstimate 수신: {:?}",
        msg
    );
}

/// AB-5 §5.8.6 gate 1 (None inert): report_tx=None 이면 RequestQcf dispatch 시 무송출.
#[test]
fn test_seq_095_dispatcher_inert_without_report_tx() {
    use argus_engine::session::command_dispatcher::CommandDispatcher;
    use argus_engine::session::pipeline_registry::PipelineRegistry;

    let registry = Arc::new(PipelineRegistry::new());
    let (report_tx, report_rx) = mpsc::channel::<EngineMessage>();

    // report_tx=None → inert.
    let mut disp = CommandDispatcher::new(
        Arc::clone(&registry),
        Vec::new(),
        None,
        Vec::new(),
        None,
        None,
        None,
        None,
        Vec::new(),
        None,                                  // None → inert
        Arc::new(std::sync::Mutex::new(None)), // hook_cell: §5.9.2 (테스트 더미)
        Arc::new(std::sync::Mutex::new(None)), // score_cell: §5.9.1 (테스트 더미)
    );

    disp.dispatch(vec![EngineCommand::RequestQcf]);

    // Nothing should be sent.
    assert!(
        report_rx.try_recv().is_err(),
        "report_tx=None → RequestQcf 무송출"
    );
    drop(report_tx); // suppress unused warning
}

#[test]
fn test_seq_095_request_qcf_returns_ok() {
    // SEQ-095: Engine receives Directive([RequestQcf]) and returns Ok (executor 표면 유지)
    let (mut executor, tx, rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::RequestQcf]);
    let _plan = executor.poll(&empty_snap());

    // Verify Response
    let msg = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    match msg {
        EngineMessage::Response(resp) => {
            assert_eq!(resp.seq_id, 1);
            assert_eq!(resp.results.len(), 1);
            assert!(matches!(resp.results[0], CommandResult::Ok));
        }
        _ => panic!("Expected Response, got {:?}", msg),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-096: Response(Ok) 후 별도 QcfEstimate 전송
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_096_send_qcf_estimate_after_response() {
    // SEQ-096: Engine sends Response first, then QcfEstimate as separate message
    let (mut executor, tx, rx) = make_executor();
    send_directive(&tx, 5, vec![EngineCommand::RequestQcf]);
    let _plan = executor.poll(&empty_snap());

    // Message 1: Response(Ok)
    let msg1 = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(
        matches!(msg1, EngineMessage::Response(_)),
        "First message should be Response"
    );

    // Simulate QCF computation and send (as generate.rs would do)
    let mut estimates = HashMap::new();
    estimates.insert("kv.evict_sliding".to_string(), 0.15);
    estimates.insert("kv.evict_h2o".to_string(), 0.22);
    executor.send_qcf_estimate(QcfEstimate {
        estimates: estimates.clone(),
        layer_swap: None,
    });

    // Message 2: QcfEstimate
    let msg2 = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    match msg2 {
        EngineMessage::QcfEstimate(qcf) => {
            assert_eq!(qcf.estimates.len(), 2);
            assert!((qcf.estimates["kv.evict_sliding"] - 0.15).abs() < f32::EPSILON);
            assert!((qcf.estimates["kv.evict_h2o"] - 0.22).abs() < f32::EPSILON);
        }
        _ => panic!("Expected QcfEstimate, got {:?}", msg2),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-096: QcfEstimate 빈 map (캐시 비어있을 때)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_096_empty_qcf_estimate_when_cache_empty() {
    // MSG-086: Engine sends empty estimates when no action computable
    let (executor, _tx, rx) = make_executor();

    executor.send_qcf_estimate(QcfEstimate {
        estimates: HashMap::new(),
        layer_swap: None,
    });

    let msg = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    match msg {
        EngineMessage::QcfEstimate(qcf) => {
            assert!(qcf.estimates.is_empty());
        }
        _ => panic!("Expected QcfEstimate"),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-095: RequestQcf와 다른 커맨드 혼합
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_095_request_qcf_mixed_with_other_commands() {
    // RequestQcf can be included with other commands in same Directive
    let (mut executor, tx, rx) = make_executor();
    send_directive(
        &tx,
        2,
        vec![
            EngineCommand::Throttle { delay_ms: 50 },
            EngineCommand::RequestQcf,
        ],
    );
    let plan = executor.poll(&empty_snap());

    assert!(plan.request_qcf);
    assert_eq!(plan.throttle_delay_ms, 50);

    let msg = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    match msg {
        EngineMessage::Response(resp) => {
            assert_eq!(resp.seq_id, 2);
            assert_eq!(resp.results.len(), 2);
            assert!(matches!(resp.results[0], CommandResult::Ok));
            assert!(matches!(resp.results[1], CommandResult::Ok));
        }
        _ => panic!("Expected Response"),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-096: Response와 QcfEstimate 순서 보장
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_096_message_ordering_response_before_estimate() {
    // SEQ-096: Response MUST be sent before QcfEstimate
    let (mut executor, tx, rx) = make_executor();
    send_directive(&tx, 7, vec![EngineCommand::RequestQcf]);
    let _plan = executor.poll(&empty_snap());

    // Send QcfEstimate (simulating generate.rs behavior after poll)
    let mut estimates = HashMap::new();
    estimates.insert("kv.evict_sliding".to_string(), 0.1);
    executor.send_qcf_estimate(QcfEstimate {
        estimates,
        layer_swap: None,
    });

    // Verify order: Response first, then QcfEstimate
    let msg1 = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(
        matches!(msg1, EngineMessage::Response(ref r) if r.seq_id == 7),
        "First message must be Response"
    );

    let msg2 = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(
        matches!(msg2, EngineMessage::QcfEstimate(_)),
        "Second message must be QcfEstimate"
    );
}

// ═══════════════════════════════════════════════════════════════
// SEQ-095: request_qcf 기본값은 false
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_095_request_qcf_default_false() {
    // Non-RequestQcf commands should not set request_qcf flag
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 10 }]);
    let plan = executor.poll(&empty_snap());
    assert!(!plan.request_qcf);
}
