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
//! - **command drain + response** — pure `poll` returns arrived commands verbatim,
//!   and `report_results` (called by the driver *after* the dispatcher has run)
//!   answers each directive with the outcomes it actually produced.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use argus_shared::{
    CommandResult, EngineCommand, EngineDirective, EngineMessage, EngineState, ManagerMessage,
    Phase,
};

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::format::KVCacheFormat;
use argus_engine::kv::kv_cache::KVCache;
use argus_engine::kv::standard_format::StandardFormat;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::resilience::CommandExecutor;
use argus_engine::session::CommandSource;
use argus_engine::session::resilience_adapter::{KvBytesHandle, ResilienceAdapter};
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
    adapter.set_kv_byte_handles(vec![handle.clone() as Arc<dyn KvBytesHandle>]);

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
    // `poll` 은 decode loop 에서만 호출되므로 heartbeat 는 언제나 decode 중 상태를 싣는다.
    // (이전에는 두 필드가 dead placeholder 라 `Idle`/`""` 로 고정 보고됐다.)
    assert_eq!(status.state, EngineState::Running);
    assert_eq!(status.phase, Phase::Decode);
    // (3) kv_cache_tokens == held-handle.current_pos() — held-handle query 전환 핵심 가드.
    assert_eq!(
        status.kv_cache_tokens,
        handle.current_pos(),
        "heartbeat kv_cache_tokens == held-handle.current_pos()"
    );
    assert_eq!(status.kv_cache_tokens, 100);
    // (4) kv_cache_bytes 는 캐시의 **실제 dtype** 회계다. 이 fixture 는 F32 (SharedBuffer::new(_, F32),
    // kv_heads=1, head_dim=32, pos=100) 이므로 100*1*32*4*2 = 25600 바이트다. 같은 토큰 수를 고정
    // F16 기하로 환산하면 12800 이 되므로, 이 단언은 "토큰 수에 상수를 곱한 값"과 진짜 바이트를
    // 구별한다 — KV 예산이 토큰 비율이 아니라 바이트 비율이려면 반드시 성립해야 하는 성질이다.
    assert_eq!(
        status.kv_cache_bytes, 25_600,
        "kv_cache_bytes 는 버퍼 dtype(F32)을 반영해야 한다 — F16 기하값 12800 이 아니다"
    );
    // (2) actual_throughput != 0 (EMA 적재 확인).
    assert!(
        status.tbt_ms > 0.0,
        "tick 2회로 time-between-tokens EMA 가 적재된다"
    );

    drop(cmd_tx); // 미사용 경고 억제
}

/// pure poll 이 도착한 command 를 그대로 반환하되 **응답은 아직 보내지 않고**,
/// `report_results` 가 드라이버가 실제로 만든 결과로 directive 에 답한다.
#[test]
fn poll_defers_response_until_results_are_reported() {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let exec = CommandExecutor::new(cmd_rx, resp_tx, Duration::from_secs(3600));
    let mut adapter = ResilienceAdapter::new(exec);

    cmd_tx
        .send(ManagerMessage::Directive(EngineDirective {
            seq_id: 7,
            commands: vec![EngineCommand::Suspend, EngineCommand::RestoreDefaults],
        }))
        .unwrap();

    let cmds = adapter.poll().unwrap();
    assert_eq!(cmds.len(), 2, "drain 한 command 2건 반환");
    assert!(matches!(cmds[0], EngineCommand::Suspend));
    assert!(matches!(cmds[1], EngineCommand::RestoreDefaults));

    // poll 만으로는 응답이 나가지 않는다 — 명령의 결과는 dispatcher 가 돈 뒤에야 정해진다.
    assert!(
        !resp_rx
            .try_iter()
            .any(|m| matches!(m, EngineMessage::Response(_))),
        "poll 단독으로 Response 를 보내면 적용 전에 성공을 보고하는 셈이다"
    );

    // 드라이버가 결과를 되돌려주면 그때 directive 1건에 Response 1건.
    adapter.report_results(vec![
        CommandResult::Ok,
        CommandResult::Rejected {
            reason: "qcf estimates are no longer produced by this engine".to_string(),
        },
    ]);
    let resp = resp_rx.recv().unwrap();
    match resp {
        EngineMessage::Response(r) => {
            assert_eq!(r.seq_id, 7);
            assert_eq!(r.results.len(), 2);
            assert!(matches!(r.results[0], CommandResult::Ok));
            assert!(
                matches!(r.results[1], CommandResult::Rejected { .. }),
                "미구현 명령은 Rejected 로 나가야 한다: {:?}",
                r.results[1]
            );
        }
        _ => panic!("Expected Response"),
    }
}

/// 여러 directive 가 한 poll 에 드레인되면, 결과는 온 순서대로 각 directive 로 되갈린다.
/// 결과가 모자라면 남은 명령은 미적용이 사실이므로 `Rejected` 로 답한다 —
/// directive 1건 = Response 1건 불변식은 어떤 경우에도 유지된다.
#[test]
fn results_are_split_back_across_directives() {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let exec = CommandExecutor::new(cmd_rx, resp_tx, Duration::from_secs(3600));
    let mut adapter = ResilienceAdapter::new(exec);

    for (seq, n) in [(1u64, 2usize), (2, 1)] {
        cmd_tx
            .send(ManagerMessage::Directive(EngineDirective {
                seq_id: seq,
                commands: vec![EngineCommand::Suspend; n],
            }))
            .unwrap();
    }
    assert_eq!(adapter.poll().unwrap().len(), 3);

    // 3건 중 2건 분량만 보고 → 나머지 1건은 Rejected 로 채워진다.
    adapter.report_results(vec![CommandResult::Ok, CommandResult::Ok]);

    let responses: Vec<_> = resp_rx
        .try_iter()
        .filter_map(|m| match m {
            EngineMessage::Response(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(responses.len(), 2, "directive 2건 → Response 2건");
    assert_eq!(responses[0].seq_id, 1);
    assert_eq!(responses[0].results.len(), 2);
    assert!(
        responses[0]
            .results
            .iter()
            .all(|r| matches!(r, CommandResult::Ok))
    );
    assert_eq!(responses[1].seq_id, 2);
    assert_eq!(responses[1].results.len(), 1);
    assert!(
        matches!(responses[1].results[0], CommandResult::Rejected { .. }),
        "보고되지 않은 명령은 미적용이다: {:?}",
        responses[1].results[0]
    );
}
