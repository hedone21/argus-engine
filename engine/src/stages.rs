//! L3 cross-cutting state-mutation Stage 거주지 (§2.1 규칙 B — subdir = 그 Stage 가 *주로*
//! 바꾸는 state 도메인 `kv`/`weight`/`system`).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §2.1 / §5.
//!
//! 각 Stage 는 `crate::pipeline::PipelineStage` 를 impl 하고 자기 Format handle 을 register 시점
//! 보관한다(§3.4 3종 handle, §3.6 wiring 표준). Stage 는 **얇은 trigger** 만 담고, 알고리즘
//! (d2o merge·`offload_one`/`recall_one` 등)은 도메인 dir(`kv/`·`weight/`)에 둔다(G3-reconcile Q3
//! 함수 단위 cut).
//!
//! **Phase α-W 신설 골격**이다 — 모듈 선언 + 예정 입주자 명세까지만. 실제 Stage impl 은 현
//! `pressure/*_handler.rs` 의 `handle()` 트리거 부분을 추출하는 **Phase α-K** 에서 입주한다.
//! 따라서 현재 서브모듈은 비어 있고(빈 trait/struct 미생성, deletion-test 정신), 어떤 live 경로에도
//! 배선되어 있지 않다(byte-identical).

pub mod kv;
pub mod system;
pub mod weight;
