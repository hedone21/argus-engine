//! Live resilience command-channel host gate.
//!
//! The legacy v1 `CommandExecutor::poll`/`ExecutionPlan` surface (and its v1↔v2
//! equivalence anchors) was removed; command application now lives in
//! `CommandDispatcher` (covered by `src/session/command_dispatcher.rs` unit tests).
//! What remains here is the LIVE `ResilienceAdapter::poll` path that has no other
//! coverage:
//! - **heartbeat continuity** — `ResilienceAdapter::poll` emits a heartbeat each
//!   interval whose `kv_cache_tokens == held-handle.current_pos()` (held-handle
//!   query), with the throughput EMA loaded via `on_token_generated`.
//! - **command drain + ack** — pure `poll` returns arrived commands verbatim and
//!   acks each directive with `CommandResult::Ok`.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use argus_shared::{EngineCommand, EngineDirective, EngineMessage, EngineState, ManagerMessage};

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::format::KVCacheFormat;
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::kv::standard_format::StandardFormat;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::resilience::CommandExecutor;
use argus_engine::session::CommandSource;
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

// ── heartbeat 연속성 (pure poll 송출·payload) ──

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
    // 레거시 set_running() 경로 제거 후 engine_state 는 기본값 Idle 로 보고된다.
    assert_eq!(status.state, EngineState::Idle);
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
