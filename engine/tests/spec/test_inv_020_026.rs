/// INV-020 ~ INV-026 (+ INV-014, INV-025): Protocol seq_id 및 Directive/Response 불변식 테스트
///
/// 추가 커버: PROTO-074 (seq_id 생성), PROTO-075 (Directive-Response 1:1)
/// - INV-014: seq_id 단조 증가 (01-architecture SYS-094, INV-020 재확인)
/// - INV-020: seq_id 단조 증가 — seq_id(N+1) > seq_id(N)
/// - INV-021: seq_id 재사용 금지
/// - INV-022: 모든 Directive는 정확히 1개 Response를 유발
/// - INV-023: Response.seq_id == Directive.seq_id
/// - INV-024: len(results) == len(commands)
/// - INV-025: INV-024 재확인 (11-protocol-messages.md)
/// - INV-026: 수신 seq_id에만 Response 전송
use std::collections::HashSet;

use argus_shared::{CommandResult, EngineCommand, EngineMessage};

use crate::helpers;

// ── INV-020: seq_id 단조 증가 ──────────────────────────────────

/// seq_id가 단조 증가하는 Directive를 보내면 Response도 동일 순서로 단조 증가해야 한다.
#[test]
fn test_inv_020_seq_id_monotonic_increasing() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    let seq_ids: Vec<u64> = vec![1, 2, 3, 5, 10, 100];
    for &seq_id in &seq_ids {
        helpers::send_directive(&tx, seq_id, vec![EngineCommand::RestoreDefaults]);
    }

    executor.poll(&snap);

    let mut response_seq_ids = Vec::new();
    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(resp) = msg {
            response_seq_ids.push(resp.seq_id);
        }
    }

    assert_eq!(
        response_seq_ids.len(),
        seq_ids.len(),
        "모든 Directive에 대해 Response가 있어야 한다"
    );

    // 단조 증가 확인
    for i in 1..response_seq_ids.len() {
        assert!(
            response_seq_ids[i] > response_seq_ids[i - 1],
            "seq_id는 단조 증가해야 한다: seq_ids[{}]={} <= seq_ids[{}]={}",
            i,
            response_seq_ids[i],
            i - 1,
            response_seq_ids[i - 1]
        );
    }
}

/// INV-014 재확인: 단조 증가 검증 (INV-020과 동일 속성)
#[test]
fn test_inv_014_seq_id_monotonic_architecture_level() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    // 단일 poll 내에서 여러 Directive 처리
    for seq_id in 1..=20 {
        helpers::send_directive(&tx, seq_id, vec![EngineCommand::Throttle { delay_ms: 0 }]);
    }

    executor.poll(&snap);

    let mut prev_seq_id = 0u64;
    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(resp) = msg {
            assert!(
                resp.seq_id > prev_seq_id,
                "INV-014: seq_id({}) must be > previous seq_id({})",
                resp.seq_id,
                prev_seq_id
            );
            prev_seq_id = resp.seq_id;
        }
    }
    assert!(prev_seq_id > 0, "최소 1개 이상의 Response가 있어야 한다");
}

// ── INV-021: seq_id 재사용 금지 ────────────────────────────────

/// N개 Directive 전송 후 모든 Response seq_id가 유일해야 한다.
#[test]
fn test_inv_021_seq_id_no_reuse() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    let n = 50;
    for seq_id in 1..=n {
        helpers::send_directive(&tx, seq_id, vec![EngineCommand::Throttle { delay_ms: 0 }]);
    }

    executor.poll(&snap);

    let mut seen = HashSet::new();
    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(resp) = msg {
            assert!(
                seen.insert(resp.seq_id),
                "INV-021: seq_id {} 가 중복되었다 (재사용 금지)",
                resp.seq_id
            );
        }
    }
    assert_eq!(
        seen.len(),
        n as usize,
        "모든 Directive에 대해 유일한 seq_id Response가 있어야 한다"
    );
}

/// 다중 poll 호출에 걸쳐서도 seq_id가 재사용되지 않아야 한다.
#[test]
fn test_inv_021_seq_id_no_reuse_across_polls() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    let mut seen = HashSet::new();

    for batch in 0..5 {
        let base = batch * 10 + 1;
        for seq_id in base..base + 10 {
            helpers::send_directive(&tx, seq_id, vec![EngineCommand::Throttle { delay_ms: 0 }]);
        }
        executor.poll(&snap);

        while let Ok(msg) = rx.try_recv() {
            if let EngineMessage::Response(resp) = msg {
                assert!(
                    seen.insert(resp.seq_id),
                    "INV-021: seq_id {} 가 다른 poll 배치에서 중복되었다",
                    resp.seq_id
                );
            }
        }
    }
    assert_eq!(seen.len(), 50, "총 50개의 유일한 Response가 있어야 한다");
}

// ── INV-022: Directive → 정확히 1 Response ─────────────────────

/// N개 Directive를 전송하면 정확히 N개 Response를 수신해야 한다.
#[test]
fn test_inv_022_exactly_one_response_per_directive() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    let n = 10;
    for seq_id in 1..=n {
        helpers::send_directive(
            &tx,
            seq_id,
            vec![EngineCommand::Throttle {
                delay_ms: seq_id * 5,
            }],
        );
    }

    executor.poll(&snap);

    let mut response_count = 0;
    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(_) = msg {
            response_count += 1;
        }
    }

    assert_eq!(
        response_count, n as usize,
        "INV-022: {} Directive에 대해 정확히 {} Response가 있어야 한다 (실제: {})",
        n, n, response_count
    );
}

/// 단일 Directive에 대해서도 정확히 1개 Response.
#[test]
fn test_inv_022_single_directive_single_response() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    helpers::send_directive(&tx, 1, vec![EngineCommand::Suspend]);
    executor.poll(&snap);

    let mut response_count = 0;
    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(_) = msg {
            response_count += 1;
        }
    }
    assert_eq!(
        response_count, 1,
        "INV-022: 단일 Directive에 대해 정확히 1개 Response"
    );
}

/// Directive가 0개이면 Response도 0개.
#[test]
fn test_inv_022_no_directive_no_response() {
    let (mut executor, _tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    executor.poll(&snap);

    let response_count = rx
        .try_recv()
        .map(|msg| matches!(msg, EngineMessage::Response(_)))
        .unwrap_or(false);
    assert!(
        !response_count,
        "INV-022: Directive가 없으면 Response도 없어야 한다"
    );
}

// ── INV-023: Response.seq_id == Directive.seq_id ───────────────

/// 각 Directive의 seq_id와 해당 Response의 seq_id가 정확히 일치해야 한다.
#[test]
fn test_inv_023_response_seq_id_matches_directive() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    let expected_seq_ids: Vec<u64> = vec![7, 42, 100, 999];
    for &seq_id in &expected_seq_ids {
        helpers::send_directive(&tx, seq_id, vec![EngineCommand::Throttle { delay_ms: 10 }]);
    }

    executor.poll(&snap);

    let mut actual_seq_ids = Vec::new();
    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(resp) = msg {
            actual_seq_ids.push(resp.seq_id);
        }
    }

    assert_eq!(
        actual_seq_ids, expected_seq_ids,
        "INV-023: Response seq_id가 Directive seq_id와 순서대로 일치해야 한다"
    );
}

/// 여러 명령을 포함한 Directive에서도 seq_id가 일치해야 한다.
#[test]
fn test_inv_023_response_seq_id_multi_command_directive() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    helpers::send_directive(
        &tx,
        55,
        vec![
            EngineCommand::Throttle { delay_ms: 20 },
            EngineCommand::LayerSkip { skip_ratio: 0.1 },
            EngineCommand::PrepareComputeUnit {
                device: "gpu".to_string(),
            },
        ],
    );

    executor.poll(&snap);

    let msg = rx.recv().unwrap();
    match msg {
        EngineMessage::Response(resp) => {
            assert_eq!(
                resp.seq_id, 55,
                "INV-023: Response.seq_id({}) == Directive.seq_id(55)",
                resp.seq_id
            );
        }
        _ => panic!("Expected Response message"),
    }
}

// ── INV-024 / INV-025: len(results) == len(commands) ──────────

/// CommandResponse의 results 길이가 Directive의 commands 길이와 같아야 한다.
#[test]
fn test_inv_024_results_len_equals_commands_len() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    let test_cases: Vec<(u64, Vec<EngineCommand>)> = vec![
        (1, vec![EngineCommand::Throttle { delay_ms: 10 }]),
        (
            2,
            vec![
                EngineCommand::Throttle { delay_ms: 20 },
                EngineCommand::LayerSkip { skip_ratio: 0.2 },
            ],
        ),
        (
            3,
            vec![
                EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
                EngineCommand::PrepareComputeUnit {
                    device: "gpu".into(),
                },
                EngineCommand::SwitchHw {
                    device: "cpu".into(),
                },
            ],
        ),
        (
            4,
            vec![
                EngineCommand::Throttle { delay_ms: 0 },
                EngineCommand::LayerSkip { skip_ratio: 0.0 },
                EngineCommand::KvEvictSliding { keep_ratio: 0.8 },
                EngineCommand::RestoreDefaults,
            ],
        ),
    ];

    let expected_lens: Vec<usize> = test_cases.iter().map(|(_, cmds)| cmds.len()).collect();

    for (seq_id, cmds) in test_cases {
        helpers::send_directive(&tx, seq_id, cmds);
    }

    executor.poll(&snap);

    let mut actual_lens = Vec::new();
    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(resp) = msg {
            actual_lens.push(resp.results.len());
        }
    }

    assert_eq!(
        actual_lens, expected_lens,
        "INV-024/025: 각 Response의 results 길이가 commands 길이와 같아야 한다"
    );
}

/// 빈 commands 배열에 대해서도 results가 빈 배열이어야 한다.
#[test]
fn test_inv_024_empty_commands_empty_results() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    helpers::send_directive(&tx, 1, vec![]);
    executor.poll(&snap);

    let msg = rx.recv().unwrap();
    match msg {
        EngineMessage::Response(resp) => {
            assert_eq!(
                resp.results.len(),
                0,
                "INV-024: 빈 commands에 대해 빈 results"
            );
        }
        _ => panic!("Expected Response"),
    }
}

/// INV-025 재확인: 다양한 크기의 commands에 대해 results 길이가 일치.
#[test]
fn test_inv_025_results_len_reconfirm() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    // 5개 명령을 포함하는 Directive
    helpers::send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Throttle { delay_ms: 10 },
            EngineCommand::LayerSkip { skip_ratio: 0.1 },
            EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
            EngineCommand::PrepareComputeUnit {
                device: "gpu".into(),
            },
            EngineCommand::Suspend,
        ],
    );

    executor.poll(&snap);

    let msg = rx.recv().unwrap();
    match msg {
        EngineMessage::Response(resp) => {
            assert_eq!(
                resp.results.len(),
                5,
                "INV-025: 5개 commands에 대해 5개 results"
            );
        }
        _ => panic!("Expected Response"),
    }
}

/// 다양한 명령을 혼합해도 results 길이는 commands 길이와 같아야 한다.
#[test]
fn test_inv_024_all_commands_counted() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    helpers::send_directive(
        &tx,
        1,
        vec![
            EngineCommand::Throttle { delay_ms: 10 },
            EngineCommand::KvStreaming {
                sink_size: 4,
                window_size: 256,
            },
            EngineCommand::Resume,
        ],
    );

    executor.poll(&snap);

    let msg = rx.recv().unwrap();
    match msg {
        EngineMessage::Response(resp) => {
            assert_eq!(
                resp.results.len(),
                3,
                "INV-024: results.len() == commands.len()"
            );
            // 모든 명령이 Ok
            assert!(matches!(resp.results[0], CommandResult::Ok));
            assert!(matches!(resp.results[1], CommandResult::Ok));
            assert!(matches!(resp.results[2], CommandResult::Ok));
        }
        _ => panic!("Expected Response"),
    }
}

// ── INV-026: 수신 seq_id에만 Response ──────────────────────────

/// 전송한 seq_id 집합에 포함되지 않은 seq_id의 Response가 없어야 한다.
#[test]
fn test_inv_026_response_only_for_received_seq_ids() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    let sent_seq_ids: HashSet<u64> = vec![3, 7, 15, 42].into_iter().collect();

    for &seq_id in &sent_seq_ids {
        helpers::send_directive(&tx, seq_id, vec![EngineCommand::Throttle { delay_ms: 0 }]);
    }

    executor.poll(&snap);

    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(resp) = msg {
            assert!(
                sent_seq_ids.contains(&resp.seq_id),
                "INV-026: Response seq_id({})가 전송한 seq_id 집합에 없다",
                resp.seq_id
            );
        }
    }
}

/// 빈 Directive 시나리오 — 아무 Response도 없어야 한다.
#[test]
fn test_inv_026_no_unsolicited_response_on_empty_poll() {
    let (mut executor, _tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    // Directive 없이 여러 번 poll
    for _ in 0..5 {
        executor.poll(&snap);
    }

    // Response가 없어야 한다 (Heartbeat는 60초 간격이므로 나오지 않는다)
    let unsolicited = rx.try_recv();
    assert!(
        unsolicited.is_err(),
        "INV-026: Directive 없이 Response가 나와선 안 된다"
    );
}

/// 다중 poll 배치에서 각 배치의 Response가 해당 배치의 seq_id에만 속해야 한다.
#[test]
fn test_inv_026_response_strictly_matches_sent_ids_across_batches() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    // 배치 1: seq_id 1, 2, 3
    let batch1: HashSet<u64> = vec![1, 2, 3].into_iter().collect();
    for &seq_id in &batch1 {
        helpers::send_directive(&tx, seq_id, vec![EngineCommand::Throttle { delay_ms: 0 }]);
    }
    executor.poll(&snap);

    let mut batch1_responses = HashSet::new();
    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(resp) = msg {
            batch1_responses.insert(resp.seq_id);
        }
    }
    assert_eq!(
        batch1_responses, batch1,
        "INV-026: 배치 1 Response seq_id 집합이 전송한 것과 정확히 일치해야 한다"
    );

    // 배치 2: seq_id 10, 20, 30
    let batch2: HashSet<u64> = vec![10, 20, 30].into_iter().collect();
    for &seq_id in &batch2 {
        helpers::send_directive(&tx, seq_id, vec![EngineCommand::RestoreDefaults]);
    }
    executor.poll(&snap);

    let mut batch2_responses = HashSet::new();
    while let Ok(msg) = rx.try_recv() {
        if let EngineMessage::Response(resp) = msg {
            batch2_responses.insert(resp.seq_id);
        }
    }
    assert_eq!(
        batch2_responses, batch2,
        "INV-026: 배치 2 Response seq_id 집합이 전송한 것과 정확히 일치해야 한다"
    );
}
