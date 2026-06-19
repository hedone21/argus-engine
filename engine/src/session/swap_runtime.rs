//! Engine-internal swap dispatch resources.
//!
//! `EngineSwapRuntime` — `main()` 진입 시 1회 생성하는 swap 자원 묶음(swap_backend / dispatcher /
//! config / release_worker / default_mode / phase 설정 / 공유 in-flight 마커 / report sender).
//! `WeightSwapStage`(stages/weight/weight_swap.rs) 가 `Arc` 공유로 보유해 commit 시 in-flight
//! 가드(§5.6.4 재submit 차단)와 manager `WeightSwapReport` 송출(§5.6.6)에 쓴다.
//!
//! Manager → Engine wire format은 `shared::EngineCommand::SwapWeights { ratio, target_dtype }`
//! (WHAT); swap mode (Incremental / IntraForward / PhaseAware / LayerImmediate) 는 wire 에 노출되지
//! 않고 engine default mode(`--swap` normalize)로 결정된다(arch/weight_swap.md §2.8.1).
//!
//! **AB-6 (arch/pipeline_stage_design_v2.md §5.6)**: 옛 `handle_swap_weights` trigger 정본은
//! `WeightSwapStage::commit` 으로 흡수됐고(레이어 선택은 다시 "swap" `WeightStage` seam 으로
//! 라우팅, EPIC 3 B3-0), 그 orphan 사본과 `SwapCommitSlot` 은 B3-0 에서 삭제됐다. Stage 간 공유
//! in-flight 마커(`in_flight`)와 report sender(`report_tx`)는 `EngineSwapRuntime`(Arc 공유)이 보유.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use argus_shared::EngineMessage;

use crate::backend::Backend;
use crate::runtime_resources_access::ReleaseWorkerAccess;
use crate::session::cli::SwapMode;
use crate::weight::AsyncSwapDispatcher;

/// Engine-wide swap dispatch 자원 묶음.
///
/// `main()` 진입 시 1회 생성. Manager 또는 CLI force-swap 신호 수신 시
/// `handle_*` method 가 self 의 자원을 사용하여 commit slot 에 mode-specific
/// 객체를 commit. 본 sprint (α) 는 Manager 경로만 통합 — CLI 강제 경로
/// (`dispatch_force_swap!` 매크로) 는 기존 그대로 유지.
pub struct EngineSwapRuntime {
    swap_backend: Arc<dyn Backend>,
    dispatcher: Arc<AsyncSwapDispatcher>,
    config: Arc<crate::model_config::ModelConfig>,
    release_worker: Arc<dyn ReleaseWorkerAccess>,
    /// CLI `--swap` flag normalize 결과. Manager-driven swap 시 이 mode로 commit.
    default_mode: SwapMode,
    /// PhaseAware mode 전용: `--swap-phase-aware-chunk-mb` * 1 MB.
    phase_chunk_size_bytes: usize,
    /// PhaseAware mode 전용: `--swap-phase-aware-max-chunks-per-token`.
    phase_max_chunks_per_token: usize,
    /// AB-6 §5.6.3: Stage 간 공유 in-flight 마커. Incremental plan 이 미완 drain
    /// 이면 `true` — 새 `WeightSwapStage` 의 commit §2 가드가 `is_idle()` 로 이를 보고
    /// reject 한다 (R-1 동시 활성화 차단). swap_runtime 이 Arc 공유 되므로 새 Stage
    /// 인스턴스도 같은 마커를 본다.
    in_flight: Arc<AtomicBool>,
    /// AB-6 §5.6.6: manager `WeightSwapReport` 송출 채널 (`CommandExecutor::resp_tx`
    /// clone). `None` 이면 미배선(resilience-off / host 단위테스트) — report drop.
    report_tx: Option<std::sync::mpsc::Sender<EngineMessage>>,
}

impl EngineSwapRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        swap_backend: Arc<dyn Backend>,
        dispatcher: Arc<AsyncSwapDispatcher>,
        config: Arc<crate::model_config::ModelConfig>,
        release_worker: Arc<dyn ReleaseWorkerAccess>,
        default_mode: SwapMode,
        phase_chunk_size_bytes: usize,
        phase_max_chunks_per_token: usize,
        report_tx: Option<std::sync::mpsc::Sender<EngineMessage>>,
    ) -> Self {
        Self {
            swap_backend,
            dispatcher,
            config,
            release_worker,
            default_mode,
            phase_chunk_size_bytes,
            phase_max_chunks_per_token,
            in_flight: Arc::new(AtomicBool::new(false)),
            report_tx,
        }
    }

    pub fn default_mode(&self) -> SwapMode {
        self.default_mode
    }

    pub fn swap_backend(&self) -> &Arc<dyn Backend> {
        &self.swap_backend
    }

    pub fn dispatcher(&self) -> &Arc<AsyncSwapDispatcher> {
        &self.dispatcher
    }

    pub fn config(&self) -> &Arc<crate::model_config::ModelConfig> {
        &self.config
    }

    pub fn release_worker(&self) -> &Arc<dyn ReleaseWorkerAccess> {
        &self.release_worker
    }

    pub fn phase_chunk_size_bytes(&self) -> usize {
        self.phase_chunk_size_bytes
    }

    pub fn phase_max_chunks_per_token(&self) -> usize {
        self.phase_max_chunks_per_token
    }

    /// AB-6 §5.6.2 §2: 현재 in-flight swap 이 없으면 `true`. `WeightSwapStage` commit
    /// §2 가드가 호출해 R-1 동시 활성화를 차단한다.
    pub fn is_idle(&self) -> bool {
        !self.in_flight.load(Ordering::Acquire)
    }

    /// AB-6 §5.6.3: Incremental plan 설치/hook 설치 시 in-flight 진입. drain 완료
    /// (`is_done()`) 또는 hook 설치 직후(non-Incremental 은 즉시 Stage 떠남) 시 해제.
    pub fn mark_in_flight(&self, active: bool) {
        self.in_flight.store(active, Ordering::Release);
    }

    /// AB-6 §5.6.6: manager 로 `WeightSwapReport` 송출 (`&self` — resp_tx clone).
    /// 미배선(`None`) 이면 no-op.
    pub fn send_swap_report(&self, report: argus_shared::WeightSwapReport) {
        if let Some(tx) = &self.report_tx {
            let _ = tx.send(EngineMessage::WeightSwapReport(report));
        }
    }
}
