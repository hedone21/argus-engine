//! KIVI fused attention capability (§3.3).
//!
//! D8(2026-06-10, single-trait) 채택으로 canonical 정의가 `argus-extension-api` 로
//! 이동했다. `&Tensor` 시그니처를 ABI struct(cl_mem) 시그니처로 바꿔 plugin 이
//! 엔진 타입을 비참조하고 컴파일되게 한다. 본 모듈은 argus-extension-api 의 정의를
//! re-export 하여 기존 `crate::capability::quant_attn::*` 및
//! `crate::backend::QuantAttnBackend` import path 를 보존한다(re-export =
//! 동일 TypeId, CapabilityRegistry register/get 일관).

pub use argus_extension_api::{
    QuantAttnArgs, QuantAttnBackend, QuantAttnGatherArgs, QuantAttnMakeArgs, QuantDequantFlushArgs,
    QuantScatterResidualArgs,
};
