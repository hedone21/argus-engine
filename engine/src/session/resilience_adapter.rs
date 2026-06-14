//! [`ResilienceAdapter`] — [`CommandExecutor`]를 session 인터페이스에 연결하는 어댑터.
//!
//! `CommandExecutor`는 `poll` / `send_capability` / `on_token_generated` 메서드를 각각 갖는다.
//! session pipeline 은 이를 [`CommandSource`](poll) / [`EngineReport`](send_*) slot 으로
//! 받는다. **β-6 commit C**: per-token tick(`on_token_generated`)은 더 이상 `TokenTickSink`
//! slot 이 아니라 `TickStage`(PostSample, stages/system/tick.rs)가 공유 Arc 로 호출한다
//! ([`ResilienceAdapter::tick`]).
//!
//! `DecodeLoopBuilder::with_resilience` 내에서 `Arc<Mutex<Self>>`로 감싸 cmd_source/report slot
//! 에 newtype wrapper 를 주입하고, tick 은 build() 에서 TickStage 로 registry submit 한다.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use argus_shared::{EngineCapability, EngineCommand, EngineMessage, QcfEstimate, WeightSwapReport};

use crate::format::KVCacheFormat;
use crate::kv::kivi_format::KIVIFormat;
use crate::resilience::{CommandExecutor, KVSnapshot};
use crate::session::command_dispatcher::{CommandSource, EngineReport};

/// [`CommandExecutor`]를 session 3-trait으로 연결하는 어댑터 (β-4: ManagerCommandSource 역할).
///
/// **β-4 (매핑 문서 4부 채택안 (가))**: `CommandSource::poll` 이 pure(`Vec<EngineCommand>` 반환)로
/// retarget 되면서, heartbeat 송출은 이 source 내부에 잔존한다 — poll 직전에
/// `send_heartbeat_if_due` 를 호출한다. heartbeat payload(`KVSnapshot`)의 `kv_snap` 운반은 poll
/// 인자 대신 **held-handle**(register 시점 주입받은 layer-0 `Arc<dyn KVCacheFormat>`)에서
/// `current_pos`/`capacity` 를 query 해 자체 구성한다(§5.2.1 (가) god-ctx 회피 정신).
pub struct ResilienceAdapter {
    executor: CommandExecutor,
    /// β-4: heartbeat payload 의 kv_cache_tokens/capacity 를 query 할 held-handle.
    /// `None` 이면 partial snapshot(pos=0) — set_kv_handle 미주입 경로.
    kv_handle: Option<Arc<dyn KVCacheFormat>>,
    /// AB-2 §5.7.6: heartbeat kv_dtype 를 query 할 KIVI concrete handle (layer-0). `None` 이면
    /// KIVI 미배선(Standard/Offload) → `kv_dtype` 는 default `""` 유지(기존 동작 불변 — 회귀 금지).
    kivi_handle: Option<Arc<KIVIFormat>>,
    /// 설정된 eviction policy 의 canonical 이름 (예: "h2o"). 빈 문자열이면 heartbeat 가
    /// "none" 으로 보고해 `compute_available_actions` 에서 kv.evict_* 가 빠진다 —
    /// Capability(12 액션) 를 manager merge 가 heartbeat 의 non-empty 3-액션 리스트로
    /// 덮어써 DPP 후보에서 kv eviction 이 전멸하는 결함의 원인 (2026-06-12 S25 적발).
    eviction_policy: String,
}

impl ResilienceAdapter {
    pub fn new(executor: CommandExecutor) -> Self {
        Self {
            executor,
            kv_handle: None,
            kivi_handle: None,
            eviction_policy: String::new(),
        }
    }

    /// 설정된 eviction policy 이름을 heartbeat `KVSnapshot` 에 전파한다.
    /// Capability 산출(`resilience_init.rs`)과 동일 소스(`args.eviction_policy()`)를 써야
    /// heartbeat available_actions 가 Capability 와 일관된다.
    pub fn set_eviction_policy(&mut self, policy: &str) {
        self.eviction_policy = policy.to_string();
    }

    /// β-4: heartbeat snapshot query 용 held-handle 주입(§5.2.1 (가) — kv_pos_handle 과 동일 패턴).
    pub fn set_kv_handle(&mut self, handle: Arc<dyn KVCacheFormat>) {
        self.kv_handle = Some(handle);
    }

    /// AB-2 §5.7.6: KIVI bench 경로에서 heartbeat kv_dtype query 용 KIVI concrete handle 주입.
    ///
    /// base `KVCacheFormat` 표면에 bit-width query 가 없어 `KIVIFormat::current_bits()` concrete
    /// 접근이 필요하다(base trait downcast 미추가 — INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC). 같은
    /// layer-0 handle 을 pos/capacity query 용 `kv_handle`(base coerce)에도 설치한다. Standard 경로는
    /// 이 seam 미주입 → `kv_dtype` default `""` 유지(기존 동작 불변 — 회귀 금지).
    pub fn set_kivi_handle(&mut self, handle: Arc<KIVIFormat>) {
        self.kv_handle = Some(handle.clone() as Arc<dyn KVCacheFormat>);
        self.kivi_handle = Some(handle);
    }

    /// 직접 접근이 필요한 caller (capability send 전 `set_has_secondary` 등)를 위해
    /// mutable ref를 노출한다.
    pub fn executor_mut(&mut self) -> &mut CommandExecutor {
        &mut self.executor
    }

    /// AB-6 §5.6.6: `EngineSwapRuntime` 구성 시 swap report 송출 채널 clone 을 노출한다
    /// (`WeightSwapStage` 가 `&self` 로 commit 시점 송신).
    pub fn report_sender(&self) -> std::sync::mpsc::Sender<EngineMessage> {
        self.executor.report_sender()
    }

    /// β-6 commit C: per-token tick. [`TickStage`](crate::stages::system::tick::TickStage)(PostSample
    /// 구독)가 매 sampled 토큰마다 호출한다. v1 `TickWrapper.on_token_generated` 와 동일 효과
    /// (executor throughput EMA 적재 + heartbeat token count 채널). `StepCtx` 불필요(stage 는
    /// `StepInfo` 만 본다).
    pub fn tick(&mut self) {
        self.executor.on_token_generated();
    }

    /// held-handle 에서 heartbeat payload 용 `KVSnapshot` 을 구성한다 (§5.2.1 (가) query).
    ///
    /// v1 `build_kv_snapshot`(decode_loop.rs)과 동일 partial — `current_pos`/`capacity` 만 채우고
    /// total_bytes 등은 placeholder(v1 도 0). handle 미주입 시 default(pos=0).
    fn build_kv_snapshot(&self) -> KVSnapshot {
        match &self.kv_handle {
            Some(h) => KVSnapshot {
                total_tokens: h.current_pos(),
                capacity: h.capacity(),
                // AB-2 §5.7.6: KIVI 경로면 layer-0 KIVIFormat 의 현재 bits 를 dtype 문자열로 매핑
                // (v1 census `generate.rs`(d5ed71d2^) L4352 동형). Standard 경로(kivi_handle=None)는
                // default `""` 유지(회귀 금지).
                kv_dtype: self
                    .kivi_handle
                    .as_ref()
                    .map(|k| bits_to_kv_dtype(k.current_bits()))
                    .unwrap_or_default(),
                eviction_policy: self.eviction_policy.clone(),
                ..KVSnapshot::default()
            },
            // handle 미주입이어도 eviction policy 는 config 차원 사실 — heartbeat
            // available_actions 가 kv.evict_* 를 잃지 않도록 항상 전파한다.
            None => KVSnapshot {
                eviction_policy: self.eviction_policy.clone(),
                ..KVSnapshot::default()
            },
        }
    }
}

/// AB-2 §5.7.6: KIVI bit-width → heartbeat `kv_dtype` 문자열 (v1 census `generate.rs`(d5ed71d2^)
/// L4352 동형). verify YAML(`direct_cmd_kvquant_to_q4.yaml:27-30`)이 `q4` transition 을 검사한다.
fn bits_to_kv_dtype(bits: u8) -> String {
    match bits {
        16 => "f16".to_string(),
        8 => "q8".to_string(),
        4 => "q4".to_string(),
        2 => "q2".to_string(),
        other => format!("q{other}"),
    }
}

impl CommandSource for ResilienceAdapter {
    fn poll(&mut self) -> Result<Vec<EngineCommand>> {
        // β-4: pure 화 후에도 heartbeat 송출 잔존 (매핑 문서 4부 채택안 (가)).
        // drain 직전 interval 체크 + 송출. kv_snap 은 held-handle query 로 자체 구성.
        let kv_snap = self.build_kv_snapshot();
        self.executor.send_heartbeat_if_due(&kv_snap);
        Ok(self.executor.drain_commands())
    }
}

impl EngineReport for ResilienceAdapter {
    fn send_capability(&mut self, cap: EngineCapability) {
        self.executor.send_capability(cap);
    }
    fn send_qcf_estimate(&mut self, qcf: QcfEstimate) {
        self.executor.send_qcf_estimate(qcf);
    }
    fn send_swap_report(&mut self, report: WeightSwapReport) {
        self.executor.send_weight_swap_report(report);
    }
}

// ── Arc<Mutex<ResilienceAdapter>> 기반 newtype wrapper ──
//
// `DecodeLoopBuilder::with_resilience`가 단일 ResilienceAdapter 인스턴스를 Arc<Mutex<…>>로
// 감싸 cmd_source 슬롯에 wrapper 를 주입한다. per-token tick 은 `TickStage`(PostSample,
// stages/system/tick.rs)가 공유 Arc 로 직접 호출한다 — `ResilienceAdapter::tick`.
//
// β-7: v1 report 슬롯 제거로 `ReportWrapper` 는 삭제됐다 — IPC 송출(capability/qcf/
// swap_report)은 `EngineReport` trait 슬롯을 경유한 적이 없다(resilience_init 가
// `CommandExecutor` 직접 호출). per-token 호출 빈도: poll 1회 + tick 1회 → contention 무시.

/// `Arc<Mutex<ResilienceAdapter>>` 를 `CommandSource` 로 노출하는 newtype.
pub(crate) struct CmdSrcWrapper(pub Arc<Mutex<ResilienceAdapter>>);

impl CommandSource for CmdSrcWrapper {
    fn poll(&mut self) -> Result<Vec<EngineCommand>> {
        self.0.lock().expect("resilience mutex poisoned").poll()
    }
}
