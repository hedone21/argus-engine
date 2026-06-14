//! Stateful [`super::traits::TokenSampler`] 구현 모음 (Phase 4-4.7+).
//!
//! [`super::defaults::GreedySampler`]가 raw argmax 무상태 sampler인 반면,
//! 본 모듈의 samplers는 token history 등 내부 상태를 유지한다. caller가
//! 별도 history vec를 누적할 필요 없이, sampler 자체가 `observe_token` hook
//! 으로 ring buffer를 갱신.

pub mod repetition_penalty;

pub use repetition_penalty::RepetitionPenaltySampler;
