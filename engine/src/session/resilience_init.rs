//! CommandExecutor 생성 헬퍼 (P3.2).
//!
//! `build_command_executor`는 legacy `generate.rs` L596~700의 CommandExecutor
//! 생성 블록을 외과적으로 이식한 함수다. argus-cli 경유로만 호출된다.
//!
//! - experiment_schedule 분기 제외 (argus-cli v0 reject)
//! - `args.enable_resilience` 가 false이면 `Ok(None)` 반환
//! - **graceful fallback**: transport 연결 실패 / unknown transport / feature
//!   off 시 `warn!` 로그 + `Ok(None)` 반환 (Manager 없이 NoOp 추론 진행).
//!   v1-1 default-on 정책에서 일반 사용자가 Manager 안 띄워도 추론이
//!   깨지지 않도록 한다.

use std::time::Duration;

use anyhow::Result;

use crate::models::transformer::TransformerModel;
use crate::resilience::{CommandExecutor, MessageLoop, TcpTransport};
use crate::session::cli::Args;

/// Args + TransformerModel 메타에서 CommandExecutor를 생성한다.
///
/// `args.enable_resilience` 가 false이면 `Ok(None)`.
/// transport 연결 실패(connection refused 등)는 `Err` 로 전파된다.
pub fn build_command_executor(
    args: &Args,
    model: &TransformerModel,
) -> Result<Option<CommandExecutor>> {
    if !args.enable_resilience {
        return Ok(None);
    }

    let heartbeat_interval = Duration::from_millis(1000);

    // transport 분기. spawn 실패 / unknown transport / feature off 모두 graceful
    // fallback (warn + Ok(None)) — default-on 정책 회귀 차단.
    let spawn_result: Result<_> = match args.resilience_transport.as_str() {
        #[cfg(feature = "resilience")]
        "dbus" => {
            use crate::resilience::DbusTransport;
            MessageLoop::spawn(DbusTransport::new()).map_err(anyhow::Error::from)
        }
        #[cfg(unix)]
        s if s.starts_with("unix:") => {
            use crate::resilience::UnixSocketTransport;
            let path = std::path::PathBuf::from(&s[5..]);
            MessageLoop::spawn(UnixSocketTransport::new(path)).map_err(anyhow::Error::from)
        }
        s if s.starts_with("tcp:") => {
            let addr = s[4..].to_string();
            MessageLoop::spawn(TcpTransport::new(addr)).map_err(anyhow::Error::from)
        }
        other => Err(anyhow::anyhow!(
            "Unknown transport '{}' (use dbus / unix:<path> / tcp:<addr>)",
            other
        )),
    };

    let (cmd_rx, resp_tx, _handle) = match spawn_result {
        Ok(triple) => triple,
        Err(e) => {
            eprintln!(
                "[Resilience] Manager unreachable ({}), running without resilience.",
                e
            );
            return Ok(None);
        }
    };

    eprintln!(
        "[Resilience] Executor enabled — transport: {}",
        args.resilience_transport
    );

    let executor = CommandExecutor::new(cmd_rx, resp_tx, heartbeat_interval);

    // No capability report: the contract has none. A command the engine cannot execute is
    // answered `Rejected`, which is both per-command and always current — a capability
    // list could not describe an action that comes and goes with configuration, and was a
    // second description of the engine to keep in sync by hand.
    let _ = model;
    Ok(Some(executor))
}

/// Bytes one token occupies in the KV cache across **all** decoder layers, uncompressed.
///
/// Times the cache capacity this is the heartbeat's `kv_cache_budget_bytes` — the
/// denominator a `KvCompress` budget is a fraction of. Deriving it from geometry is
/// correct precisely because the question is hypothetical: what the cache *would* cost
/// with no compression applied. The numerator it is compared against is measured, not
/// derived, which is what keeps their ratio meaningful.
pub fn uncompressed_kv_bytes_per_token(cfg: &crate::model_config::ModelConfig) -> usize {
    cfg.num_key_value_heads
        * cfg.head_dim
        * 2  // K + V
        * 2  // f16
        * cfg.num_hidden_layers
}
