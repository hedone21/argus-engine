//! ENG-ST-032: available_actions 계산 테스트
//!
//! CommandExecutor::compute_available_actions()는 engine 상태에 따라
//! 실행 가능한 액션 목록을 동적으로 결정한다.
//!
//! - throttle, switch_hw, layer_skip: 항상 사용 가능
//! - kv_evict_h2o, kv_evict_sliding: eviction_policy != "none"일 때만
//! - kv_quant_dynamic: kv_dtype가 'q'로 시작할 때만 (KIVI cache)
//! - swap_weights: secondary GGUF/AUF가 로드된 경우에만 (ENG-ST-032)

use argus_engine::resilience::{CommandExecutor, KVSnapshot};
use argus_shared::{EngineCommand, EngineDirective, EngineMessage, ManagerMessage};
use std::sync::mpsc;
use std::time::Duration;

/// Heartbeat를 유도하여 available_actions를 확인하는 헬퍼.
/// `has_secondary`가 true이면 secondary GGUF/AUF 존재 시나리오를 시뮬레이션한다.
fn get_available_actions_with_secondary(
    eviction_policy: &str,
    kv_dtype: &str,
    has_secondary: bool,
) -> Vec<String> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_millis(1), // 매우 짧은 heartbeat 간격
    );
    executor.set_has_secondary(has_secondary);
    executor.set_running();

    // heartbeat 간격 경과 대기
    std::thread::sleep(Duration::from_millis(5));

    let snap = KVSnapshot {
        total_bytes: 1000,
        total_tokens: 100,
        capacity: 2048,
        protected_prefix: 4,
        kv_dtype: kv_dtype.to_string(),
        eviction_policy: eviction_policy.to_string(),
        skip_ratio: 0.0,
    };

    let _ = executor.poll(&snap);

    // Heartbeat에서 available_actions 추출
    while let Ok(msg) = resp_rx.try_recv() {
        if let EngineMessage::Heartbeat(status) = msg {
            drop(cmd_tx);
            return status.available_actions;
        }
    }
    drop(cmd_tx);
    panic!("No heartbeat received");
}

/// 편의 헬퍼: has_secondary=false (기존 동작 유지)
fn get_available_actions(eviction_policy: &str, kv_dtype: &str) -> Vec<String> {
    get_available_actions_with_secondary(eviction_policy, kv_dtype, false)
}

/// 기본 상태 (eviction_policy="none", kv_dtype="f16"):
/// throttle, switch_hw, layer_skip만 포함
#[test]
fn test_eng_st_032_available_actions_initial() {
    let actions = get_available_actions("none", "f16");

    assert!(actions.contains(&"throttle".to_string()));
    assert!(actions.contains(&"switch_hw".to_string()));
    assert!(actions.contains(&"weight.skip".to_string()));

    // eviction 없으면 evict 액션 비포함
    assert!(
        !actions.contains(&"kv.evict_h2o".to_string()),
        "kv_evict_h2o should not be available without eviction policy"
    );
    assert!(
        !actions.contains(&"kv.evict_sliding".to_string()),
        "kv_evict_sliding should not be available without eviction policy"
    );

    // f16이면 quant 비포함
    assert!(
        !actions.contains(&"kv.quant_dynamic".to_string()),
        "kv_quant_dynamic should not be available with f16"
    );
}

/// active_actions에 이미 있는 액션은 available_actions에서 제외되지 않음
/// (available은 "사용 가능한 액션", active와 독립)
#[test]
fn test_eng_st_032_available_excludes_active() {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerMessage>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_millis(1));
    executor.set_running();

    // throttle 액션 활성화
    cmd_tx
        .send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Throttle { delay_ms: 50 }],
        }))
        .unwrap();
    let snap = KVSnapshot {
        total_bytes: 1000,
        total_tokens: 100,
        capacity: 2048,
        protected_prefix: 4,
        kv_dtype: "f16".to_string(),
        eviction_policy: "none".to_string(),
        skip_ratio: 0.0,
    };
    executor.poll(&snap);

    // throttle이 active
    assert!(executor.active_actions().contains(&"throttle".to_string()));

    // heartbeat 유도
    std::thread::sleep(Duration::from_millis(5));
    let _ = executor.poll(&snap);

    // available_actions는 여전히 throttle 포함
    // (active 여부와 무관하게 capability 기반)
    let mut found = false;
    while let Ok(msg) = resp_rx.try_recv() {
        if let EngineMessage::Heartbeat(status) = msg {
            assert!(
                status.available_actions.contains(&"throttle".to_string()),
                "throttle should still be in available even when active"
            );
            // active_actions에도 포함
            assert!(
                status.active_actions.contains(&"throttle".to_string()),
                "throttle should be in active_actions"
            );
            found = true;
            break;
        }
    }
    assert!(found, "Should have received heartbeat");
    drop(cmd_tx);
}

/// 디바이스 종속: eviction_policy="h2o"이면 evict 액션 추가,
/// kv_dtype="q8"이면 kv_quant_dynamic 추가
#[test]
fn test_eng_st_032_available_device_dependent() {
    // eviction policy가 설정되면 evict 액션 가용
    let actions = get_available_actions("h2o", "f16");
    assert!(
        actions.contains(&"kv.evict_h2o".to_string()),
        "kv_evict_h2o should be available with h2o policy"
    );
    assert!(
        actions.contains(&"kv.evict_sliding".to_string()),
        "kv_evict_sliding should be available with eviction policy"
    );

    // KIVI cache (q8)이면 kv_quant_dynamic 가용
    let actions = get_available_actions("none", "q8");
    assert!(
        actions.contains(&"kv.quant_dynamic".to_string()),
        "kv_quant_dynamic should be available with KIVI cache"
    );

    // 둘 다 설정
    let actions = get_available_actions("sliding", "q4");
    assert!(actions.contains(&"kv.evict_h2o".to_string()));
    assert!(actions.contains(&"kv.quant_dynamic".to_string()));
}

/// secondary 비활성: swap_weights가 available_actions에 포함되지 않아야 한다.
#[test]
fn test_eng_st_032_swap_weights_absent_without_secondary() {
    let actions = get_available_actions_with_secondary("none", "f16", false);
    assert!(
        !actions.contains(&"swap_weights".to_string()),
        "swap_weights should not be available when no secondary is loaded"
    );
}

/// secondary 활성: swap_weights가 available_actions에 포함되어야 한다.
#[test]
fn test_eng_st_032_swap_weights_present_with_secondary() {
    let actions = get_available_actions_with_secondary("none", "f16", true);
    assert!(
        actions.contains(&"swap_weights".to_string()),
        "swap_weights should be available when secondary GGUF/AUF is loaded"
    );
    // 기본 액션들은 여전히 포함
    assert!(actions.contains(&"throttle".to_string()));
    assert!(actions.contains(&"switch_hw".to_string()));
    assert!(actions.contains(&"weight.skip".to_string()));
}

/// secondary 활성 + eviction + KIVI 조합: 모든 액션이 동시에 등록되어야 한다.
#[test]
fn test_eng_st_032_swap_weights_combined_with_eviction_and_quant() {
    let actions = get_available_actions_with_secondary("h2o", "q4", true);
    assert!(actions.contains(&"swap_weights".to_string()));
    assert!(actions.contains(&"kv.evict_h2o".to_string()));
    assert!(actions.contains(&"kv.quant_dynamic".to_string()));
    assert!(actions.contains(&"throttle".to_string()));
}
