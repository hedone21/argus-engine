//! QCF (Quality Cost Function) based degradation estimation for lossy actions.
//!
//! Each lossy action (H2O eviction, KIVI quantization, SWIFT layer skip)
//! produces a `QcfMetric` as a side effect.
//! A `DegradationEstimator` converts QCF values to estimated PPL increase
//! via offline-calibrated piecewise-linear coefficients.
//!
//! **Layered structure (§13.8-G)**: data identifier 집합(`QcfMetric`,
//! `QcfConfig`, `QcfMode`, `AggregationMode`, `KiviFlushParams`,
//! `FlushAttentionParams`, `SubLayer`, `ImportanceFormula`)은 L2의
//! `crate::qcf_types`에 위치하며, 본 모듈에서는 편의를 위해 re-export 한다.
//! 측정 로직(compute_flush_*, ImportanceCollector 등)은 본 L3-qcf 도메인에
//! 유지된다.

pub mod entropy;
pub mod estimator;
pub mod layer_aggregation;
pub mod layer_importance;
pub mod qcf_kv;
pub mod quant_qcf;
pub mod skip_qcf;
pub mod topk_retention;

// L2 data identifier re-export (§13.8-G shared identifier promotion).
pub use crate::qcf_types::{
    AggregationMode, FlushAttentionParams, ImportanceFormula, KiviFlushParams, QcfConfig,
    QcfMetric, QcfMode, SubLayer, aggregate_heads,
};

pub use entropy::{EntropyResult, compute_normalized_entropy};
pub use estimator::DegradationEstimator;
pub use layer_aggregation::{
    LayerAggregationMode, aggregate_layers, compute_auto_sample_layers, compute_c1, compute_d7,
};
pub use layer_importance::{ImportanceCollector, ImportanceTable};
pub use qcf_kv::{
    QcfActionType, QcfKvParams, VDataSource, compute_qcf_kv, identify_retained_for_action,
    identify_retained_h2o, identify_retained_sliding,
};
pub use quant_qcf::{
    KiviQcfComputer, compute_flush_aw_vopr, compute_flush_awqe, compute_flush_nmse,
    compute_flush_opr,
};
pub use skip_qcf::SkipQcfTracker;
pub use topk_retention::{TopKRetentionResult, compute_topk_retention};
