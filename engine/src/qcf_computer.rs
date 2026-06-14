//! QcfComputer trait — KIVI flush proxy 메트릭 계산 추상화 (L2).
//!
//! L3-pressure (`pressure/kivi_cache.rs`) 가 L3-qcf 도메인의 자유 함수
//! (`compute_flush_nmse` / `_opr` / `_awqe` / `_aw_vopr`) 를 직접 호출하는
//! 패턴을 trait inversion으로 해소한다 (S-3b-4 / §G + INV-LAYER-003 정합).
//!
//! 구현체는 `engine/src/qcf/quant_qcf.rs` 의 `KiviQcfComputer` (struct unit
//! 또는 stateless impl). caller (pressure)는 `&dyn QcfComputer` 또는
//! `&impl QcfComputer` 로 trait dispatch.

use crate::qcf_types::{FlushAttentionParams, KiviFlushParams, QcfConfig, QcfMetric};

/// KIVI flush proxy 계산 trait.
///
/// 4 메서드는 quant_qcf.rs 의 자유 함수 4개와 동등 signature.
/// KiviCache 가 flush event 시 trait method 를 호출하여 QcfMetric 생성.
pub trait QcfComputer: Send + Sync {
    /// NMSE 기반 V/K residual quantization error.
    fn flush_nmse(&self, params: &KiviFlushParams, config: &QcfConfig) -> QcfMetric;

    /// OPR (output perturbation ratio) variant.
    fn flush_opr(&self, params: &KiviFlushParams, config: &QcfConfig) -> QcfMetric;

    /// AWQE (Attention-Weighted Quantization Error) scalar.
    fn flush_awqe(&self, params: &FlushAttentionParams, config: &QcfConfig) -> QcfMetric;

    /// AW-VOPR (attention-weighted vector OPR).
    fn flush_aw_vopr(&self, params: &FlushAttentionParams, config: &QcfConfig) -> QcfMetric;

    /// Dyn-compatible clone — `Box<dyn QcfComputer>` 컨테이너의 Clone 구현
    /// (KiviCache::Clone 등 owner type derive 지원).
    fn clone_box(&self) -> Box<dyn QcfComputer>;
}

impl Clone for Box<dyn QcfComputer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
