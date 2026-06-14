/// INV-015: Capability는 세션당 정확히 1회 전송.
///
/// 원본: 01-architecture SYS-094, 10-protocol PROTO-044
/// 검증: CommandExecutor.send_capability()는 EngineMessage::Capability를 정확히 1개 전송하며,
///       poll() 과정에서 추가 Capability가 전송되지 않아야 한다.
use argus_shared::{EngineCapability, EngineMessage};

use crate::helpers;

/// 테스트용 EngineCapability 생성
fn test_capability() -> EngineCapability {
    EngineCapability {
        available_devices: vec!["cpu".to_string(), "opencl".to_string()],
        active_device: "cpu".to_string(),
        max_kv_tokens: 2048,
        bytes_per_kv_token: 256,
        num_layers: 16,
        ..Default::default()
    }
}

/// send_capability 호출 시 정확히 1개의 Capability 메시지가 전송되어야 한다.
#[test]
fn test_inv_015_capability_sent_once() {
    let (executor, _tx, rx) = helpers::make_executor();

    executor.send_capability(test_capability());

    // 정확히 1개의 Capability 수신
    let msg = rx.recv().unwrap();
    assert!(
        matches!(msg, EngineMessage::Capability(_)),
        "INV-015: send_capability 후 Capability 메시지가 와야 한다"
    );

    // 추가 Capability가 없어야 한다
    let extra = rx.try_recv();
    assert!(
        extra.is_err(),
        "INV-015: send_capability 후 추가 메시지가 없어야 한다"
    );
}

/// send_capability 후 poll()을 호출해도 추가 Capability가 전송되지 않아야 한다.
#[test]
fn test_inv_015_no_extra_capability_after_poll() {
    let (mut executor, _tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    executor.send_capability(test_capability());

    // Capability 수신
    let msg = rx.recv().unwrap();
    assert!(matches!(msg, EngineMessage::Capability(_)));

    // 여러 번 poll 해도 추가 Capability가 나오지 않아야 한다
    for _ in 0..10 {
        executor.poll(&snap);
    }

    let mut capability_count = 0;
    while let Ok(msg) = rx.try_recv() {
        if matches!(msg, EngineMessage::Capability(_)) {
            capability_count += 1;
        }
    }

    assert_eq!(
        capability_count, 0,
        "INV-015: poll() 이후 추가 Capability가 전송되면 안 된다"
    );
}

/// send_capability 후 Directive 처리 과정에서도 추가 Capability가 나오지 않아야 한다.
#[test]
fn test_inv_015_no_capability_during_directive_processing() {
    let (mut executor, tx, rx) = helpers::make_executor();
    let snap = helpers::empty_snap();

    executor.send_capability(test_capability());

    // Capability 수신 확인
    let msg = rx.recv().unwrap();
    assert!(matches!(msg, EngineMessage::Capability(_)));

    // Directive 전송 및 처리
    for seq_id in 1..=5 {
        helpers::send_directive(
            &tx,
            seq_id,
            vec![argus_shared::EngineCommand::Throttle { delay_ms: 10 }],
        );
    }
    executor.poll(&snap);

    // 수신된 메시지 중 Capability가 없어야 한다
    let mut capability_count = 0;
    let mut response_count = 0;
    while let Ok(msg) = rx.try_recv() {
        match msg {
            EngineMessage::Capability(_) => capability_count += 1,
            EngineMessage::Response(_) => response_count += 1,
            _ => {}
        }
    }

    assert_eq!(
        capability_count, 0,
        "INV-015: Directive 처리 중 추가 Capability가 전송되면 안 된다"
    );
    assert_eq!(
        response_count, 5,
        "5개 Directive에 대해 5개 Response가 있어야 한다"
    );
}

/// Capability 메시지의 내용이 전달한 것과 일치해야 한다.
#[test]
fn test_inv_015_capability_content_correct() {
    let (executor, _tx, rx) = helpers::make_executor();

    let cap = test_capability();
    executor.send_capability(cap.clone());

    let msg = rx.recv().unwrap();
    match msg {
        EngineMessage::Capability(received) => {
            assert_eq!(received.available_devices, vec!["cpu", "opencl"]);
            assert_eq!(received.active_device, "cpu");
            assert_eq!(received.max_kv_tokens, 2048);
            assert_eq!(received.bytes_per_kv_token, 256);
            assert_eq!(received.num_layers, 16);
        }
        _ => panic!("Expected Capability message"),
    }
}
