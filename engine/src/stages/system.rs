//! system Stage 거주지 (§2.1, 확정 거주자 1개).
//!
//! 예정 입주자: `SwitchStage`(hardware 축 동작 — `Arc<Hardware>` 를 register 시점 보관하고
//! `resolve(target)` 으로 대상 backend/memory 를 푼다, §5.1). switch 는 KV migrate 를 *유발*
//! 하지만 주 의도가 device 변경이므로 `system/`. resilience-제어 Stage / observe·보고 Stage 의
//! 분해·명명은 downstream(resilience→stage 매핑 별도 설계).
//!
//! **입주자 1호(Phase β-6 commit C)**: [`tick::TickStage`] — `PostSample` phase 에서
//! `ResilienceAdapter` 의 per-token tick(throughput EMA + heartbeat token count)을 발화한다.
//! v1 `TokenTickSink`(`TickWrapper`) 의 stage 화. `SwitchStage` 등 나머지는 후속.
//!
//! **입주자 2호**: [`prefill_phase::PrefillPhaseStage`] — `PrefillStart`/`PrefillEnd` 에서
//! heartbeat 의 `Phase` 를 stamp 하고 강제 송출한다. 그 전까지 매니저가 관측하는 phase 는
//! decode 상수였고, 긴 prompt 의 prefill 은 hang 과 구분 불가한 무음 구간이었다.

pub mod prefill_phase;
pub mod tick;
