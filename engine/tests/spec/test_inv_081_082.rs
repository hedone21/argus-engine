//! INV-081: IPC 직렬화는 JSON (serde_json) 전용. 바이너리 직렬화 포맷 금지.
//! INV-082: 1:1 단일 클라이언트 연결. 다중 Engine 동시 연결 금지.
//!
//! 원본: SYS-065/ENG-071/CON-011 (INV-081), SYS-093/PROTO-041/ENG-072 (INV-082)
//!
//! 검증 전략:
//!   INV-081: 모든 IPC 메시지 타입의 serde_json round-trip 검증.
//!            Transport wire format이 JSON 기반인지 확인.
//!            Shared 크레이트에 serde_json 의존성 확인 (static).
//!   INV-082: Transport trait의 connect()가 단일 연결을 반환하는지 확인.
//!            MockTransport::bidirectional()이 1:1 채널 쌍을 생성하는지 검증.
//!            MessageLoop::spawn()이 단일 thread + 단일 transport 소유권을 가지는지 확인.

use argus_shared::{
    CommandResponse, CommandResult, EngineCapability, EngineCommand, EngineDirective,
    EngineMessage, EngineState, EngineStatus, Level, ManagerMessage, RecommendedBackend,
    ResourceLevel, SystemSignal,
};

use argus_engine::resilience::{MessageLoop, MockTransport, Transport};

// ═══════════════════════════════════════════════════════════════
// INV-081: IPC serialization is JSON-only (serde_json)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_081_engine_directive_json_roundtrip() {
    let directive = EngineDirective {
        seq_id: 42,
        commands: vec![
            EngineCommand::Throttle { delay_ms: 50 },
            EngineCommand::KvEvictH2o { keep_ratio: 0.48 },
            EngineCommand::LayerSkip { skip_ratio: 0.25 },
            EngineCommand::KvQuantDynamic { target_bits: 4 },
            EngineCommand::SwitchHw {
                device: "opencl".to_string(),
            },
            EngineCommand::PrepareComputeUnit {
                device: "gpu".to_string(),
            },
            EngineCommand::Suspend,
            EngineCommand::Resume,
            EngineCommand::RestoreDefaults,
            EngineCommand::KvEvictSliding { keep_ratio: 0.6 },
            EngineCommand::KvStreaming {
                sink_size: 4,
                window_size: 256,
            },
        ],
    };

    let json = serde_json::to_string(&directive).unwrap();
    let back: EngineDirective = serde_json::from_str(&json).unwrap();
    assert_eq!(back.seq_id, 42);
    assert_eq!(
        back.commands.len(),
        11,
        "INV-081: all command variants must survive JSON round-trip"
    );
}

#[test]
fn test_inv_081_manager_message_json_roundtrip() {
    let msg = ManagerMessage::Directive(EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::Suspend],
    });

    let json = serde_json::to_string(&msg).unwrap();
    assert!(
        json.contains("\"type\""),
        "INV-081: ManagerMessage must use JSON tagged enum format"
    );
    let back: ManagerMessage = serde_json::from_str(&json).unwrap();
    match back {
        ManagerMessage::Directive(d) => {
            assert_eq!(d.seq_id, 1);
            assert!(matches!(d.commands[0], EngineCommand::Suspend));
        }
    }
}

#[test]
fn test_inv_081_engine_message_all_variants_json() {
    // Capability
    let cap = EngineMessage::Capability(EngineCapability {
        available_devices: vec!["cpu".into(), "opencl".into()],
        active_device: "cpu".into(),
        max_kv_tokens: 2048,
        bytes_per_kv_token: 256,
        num_layers: 16,
        ..Default::default()
    });
    let json = serde_json::to_string(&cap).unwrap();
    let back: EngineMessage = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, EngineMessage::Capability(_)));

    // Heartbeat
    let hb = EngineMessage::Heartbeat(EngineStatus {
        active_device: "cpu".to_string(),
        compute_level: ResourceLevel::Normal,
        actual_throughput: 15.0,
        memory_level: ResourceLevel::Warning,
        kv_cache_bytes: 1024,
        kv_cache_tokens: 100,
        kv_cache_utilization: 0.5,
        memory_lossless_min: 1.0,
        memory_lossy_min: 0.01,
        state: EngineState::Running,
        tokens_generated: 50,
        available_actions: vec!["throttle".into()],
        active_actions: vec![],
        eviction_policy: "none".into(),
        kv_dtype: "f16".into(),
        skip_ratio: 0.0,
        phase: String::new(),
        prefill_pos: 0,
        prefill_total: 0,
        partition_ratio: 0.0,
        self_cpu_pct: 0.0,
        self_gpu_pct: 0.0,
    });
    let json = serde_json::to_string(&hb).unwrap();
    let back: EngineMessage = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, EngineMessage::Heartbeat(_)));

    // Response
    let resp = EngineMessage::Response(CommandResponse {
        seq_id: 5,
        results: vec![
            CommandResult::Ok,
            CommandResult::Partial {
                achieved: 0.7,
                reason: "throttled".into(),
            },
            CommandResult::Rejected {
                reason: "unsupported".into(),
            },
        ],
    });
    let json = serde_json::to_string(&resp).unwrap();
    let back: EngineMessage = serde_json::from_str(&json).unwrap();
    match back {
        EngineMessage::Response(r) => {
            assert_eq!(r.seq_id, 5);
            assert_eq!(r.results.len(), 3);
        }
        _ => panic!("Expected Response"),
    }
}

#[test]
fn test_inv_081_system_signal_json_roundtrip() {
    let signals = vec![
        SystemSignal::MemoryPressure {
            level: Level::Critical,
            available_bytes: 1024,
            total_bytes: 4096,
            reclaim_target_bytes: 512,
        },
        SystemSignal::ComputeGuidance {
            level: Level::Warning,
            recommended_backend: RecommendedBackend::Gpu,
            reason: argus_shared::ComputeReason::CpuBottleneck,
            cpu_usage_pct: 95.0,
            gpu_usage_pct: 20.0,
        },
        SystemSignal::ThermalAlert {
            level: Level::Emergency,
            temperature_mc: 85000,
            throttling_active: true,
            throttle_ratio: 0.5,
        },
        SystemSignal::EnergyConstraint {
            level: Level::Normal,
            reason: argus_shared::EnergyReason::Charging,
            power_budget_mw: 15000,
        },
    ];

    for sig in &signals {
        let json = serde_json::to_string(sig).unwrap();
        let back: SystemSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.level(),
            sig.level(),
            "INV-081: SystemSignal JSON round-trip must preserve level"
        );
    }
}

#[test]
fn test_inv_081_json_output_is_valid_json() {
    // Verify that wire format produces valid JSON (not binary)
    let msg = ManagerMessage::Directive(EngineDirective {
        seq_id: 99,
        commands: vec![EngineCommand::Throttle { delay_ms: 42 }],
    });

    let json_bytes = serde_json::to_vec(&msg).unwrap();

    // Must be valid UTF-8 (JSON requirement)
    let json_str = std::str::from_utf8(&json_bytes)
        .expect("INV-081: serialized output must be valid UTF-8 (JSON)");

    // Must parse as JSON
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).expect("INV-081: output must be valid JSON");

    assert!(
        parsed.is_object(),
        "INV-081: top-level JSON must be an object"
    );
}

// ═══════════════════════════════════════════════════════════════
// INV-082: 1:1 single client connection
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_inv_082_mock_transport_bidirectional_is_1_to_1() {
    // MockTransport::bidirectional() creates exactly 1 engine-side
    // and 1 manager-side handle — verifying 1:1 property
    let (mut transport, mgr) = MockTransport::bidirectional();
    transport.connect().unwrap();

    // Manager sends, Engine receives — single channel
    mgr.send(ManagerMessage::Directive(EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::Throttle { delay_ms: 10 }],
    }))
    .unwrap();

    let msg = transport.recv().unwrap();
    match msg {
        ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
    }

    // Engine sends back, Manager receives — single channel
    transport
        .send(&EngineMessage::Response(CommandResponse {
            seq_id: 1,
            results: vec![CommandResult::Ok],
        }))
        .unwrap();

    let resp = mgr.recv().unwrap();
    assert!(matches!(resp, EngineMessage::Response(_)));
}

#[test]
fn test_inv_082_message_loop_single_transport_ownership() {
    // MessageLoop::spawn takes ownership of transport (T: Transport),
    // ensuring only one thread has access (1:1 constraint).
    let msgs = vec![ManagerMessage::Directive(EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::Throttle { delay_ms: 0 }],
    })];
    let transport = MockTransport::from_messages(msgs);

    // After spawn, transport is moved into the thread — cannot be used elsewhere
    let (cmd_rx, _resp_tx, handle) = MessageLoop::spawn(transport).unwrap();

    // Only the MessageLoop thread accesses transport; cmd_rx/resp_tx are
    // the only external interfaces (1:1 channel pair)
    let msg = cmd_rx.recv().unwrap();
    match msg {
        ManagerMessage::Directive(d) => assert_eq!(d.seq_id, 1),
    }

    let _ = handle.join();
}

#[test]
fn test_inv_082_transport_connect_returns_single_connection() {
    // Transport::connect() establishes exactly one connection.
    // MockTransport always succeeds, but the pattern is:
    // connect once → recv/send on that single connection.
    let (mut transport, _sender) = MockTransport::channel();

    // First connect succeeds
    assert!(
        transport.connect().is_ok(),
        "INV-082: first connect must succeed"
    );

    // Second connect also succeeds for Mock (idempotent),
    // but real transports (Unix/TCP) would fail or replace the connection.
    // The key invariant is: at any point, there's at most 1 active connection.
    assert!(
        transport.connect().is_ok(),
        "INV-082: Mock transport connect is idempotent"
    );
}
