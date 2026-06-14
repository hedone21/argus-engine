// Step 4-C: L3-inference 도메인 모듈 — `core/`에서 promote.
// 각 모듈은 추론 path에서 직접 사용되는 연산/구성 단위로,
// L3-inference layer에 속한다 (LAYER_RULES `("inference", "L3-inference")`).
pub mod attention_scores;
pub mod query_stats;
pub mod sampling;
pub mod skip_config;
pub mod speculative;
