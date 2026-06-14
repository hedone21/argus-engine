//! Eviction method identifier — engine-internal policy dispatch key.
//!
//! definitional owner = pressure domain (variants ↔ EvictionPolicy / CachePressureHandler 1:1).
//! resilience/executor.rs는 EvictPlan.method 필드로 이 enum을 참조 (cross-cutting → L3,
//! §13.8-F enum-as-data identifier 예외).

/// Eviction method identifier (engine-internal, not in shared protocol).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictMethod {
    H2o,
    Sliding,
    Streaming,
    D2o,
}
