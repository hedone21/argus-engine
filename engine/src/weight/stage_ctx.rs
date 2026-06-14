//! `WeightStageModelCtx` — weight stage 의 엔진 측 `WeightStageCtx` 구현.
//!
//! KV 거울(`pressure::eviction::stage_registry` 의 KV stage ctx)의 weight 판.
//! `&TransformerModel` 의 읽기 표면을 technique-api `WeightStageCtx` plugin 표면으로
//! 투영한다 — plugin 은 읽고 plan 만 내며 변형은 엔진 executor 독점.
//!
//! 투영은 기존 호출자(`session::swap_runtime::handle_swap_weights` §3–4)가 이미
//! 수행하던 것과 **bit-identical**: budget = floor(ratio·n) − |currently_swapped|,
//! importance = `flatten_importance`(SubLayer::Full), noise = `is_computed()` 일 때만 Some.
//!
//! production 호출부(decode loop) 배선은 Seam B(Phase β) 의 일이라, 본 모듈은 ctx 구현
//! + 투영 + bit-identical 단위테스트까지만 둔다(호출부 배선 금지).

use technique_api::{LayerMetricKind, TensorDtype, WeightStageCtx};

use crate::buffer::DType;
use crate::models::transformer::TransformerModel;
use crate::qcf_collector::ImportanceLookup;
use crate::runtime_resources_access::QuantNoiseAccess;
use crate::weight::decider::flatten_importance;

/// 엔진 `&TransformerModel` 위 `WeightStageCtx` 구현 (MW-D).
///
/// 읽기 값은 생성 시점에 투영해 **owned** 로 보관한다 — `WeightStageCtx` 가
/// `Option<&[f32]>` 반환을 요구(dyn-safe)하고, importance 는 `flatten_importance`
/// 가 본질적으로 새 `Vec` 를 만드는 투영이라 빌릴 수 없기 때문. noise 도 동일한
/// is_computed 계약을 owned 사본으로 보존한다.
pub struct WeightStageModelCtx {
    n_layers: usize,
    budget: usize,
    pressure: u8,
    /// per-layer 현재 저장 dtype (`current_dtype` → `TensorDtype` 투영).
    layer_formats: Vec<TensorDtype>,
    /// per-layer importance (`SubLayer::Full` 투영). 미가용 시 `None`.
    importance_flat: Option<Vec<f32>>,
    /// per-layer quant noise ε. `is_computed() == false` 면 `None`
    /// (decider 의 uniform fallback 게이트 보존).
    noise_flat: Option<Vec<f32>>,
}

impl WeightStageModelCtx {
    /// `&TransformerModel` 로부터 ctx 를 투영한다(엔진 진입점, Seam B 가 호출 예정).
    ///
    /// - `ratio` = command-driven swap 목표 비율 (`EngineCommand::SwapWeights{ratio}`).
    /// - `importance` = warmup 산출 `ImportanceTable` lookup (없으면 `None` → uniform).
    /// - `pressure` = graded 메모리 압력 0–100 (command path 는 보통 0).
    pub fn from_model(
        model: &TransformerModel,
        ratio: f32,
        importance: Option<&dyn ImportanceLookup>,
        pressure: u8,
    ) -> Self {
        let layer_formats: Vec<TensorDtype> = model
            .layers
            .iter()
            .map(|slot| dtype_to_tensor_dtype(slot.current_dtype()))
            .collect();
        Self::from_parts(
            layer_formats,
            &*model.quant_noise,
            importance,
            ratio,
            pressure,
        )
    }

    /// 투영 코어 — `from_model` 이 위임. `TransformerModel` 비의존(테스트가 직접 구성 가능).
    ///
    /// `swap_runtime::handle_swap_weights` 의 호출자 측 투영(§3–4)과 1:1:
    /// currently_swapped = Q4_0 레이어 수, budget = floor(ratio·n) − swapped,
    /// importance = `flatten_importance`, noise = `is_computed()` 일 때만 `Some`.
    pub fn from_parts(
        layer_formats: Vec<TensorDtype>,
        quant_noise: &dyn QuantNoiseAccess,
        importance: Option<&dyn ImportanceLookup>,
        ratio: f32,
        pressure: u8,
    ) -> Self {
        let n = layer_formats.len();
        let currently_swapped = layer_formats
            .iter()
            .filter(|&&d| d == TensorDtype::Q4_0)
            .count();
        let target_count = (ratio * n as f32).floor() as usize;
        let budget = target_count.saturating_sub(currently_swapped);

        let importance_flat = importance.map(|imp| flatten_importance(imp, n));
        let noise_flat = if quant_noise.is_computed() {
            Some(quant_noise.as_slice().to_vec())
        } else {
            None
        };

        Self {
            n_layers: n,
            budget,
            pressure,
            layer_formats,
            importance_flat,
            noise_flat,
        }
    }
}

impl WeightStageCtx for WeightStageModelCtx {
    fn n_layers(&self) -> usize {
        self.n_layers
    }
    fn budget(&self) -> usize {
        self.budget
    }
    fn pressure(&self) -> u8 {
        self.pressure
    }
    fn current_format(&self, layer: usize) -> TensorDtype {
        self.layer_formats[layer]
    }
    fn layer_metric(&self, kind: LayerMetricKind) -> Option<&[f32]> {
        match kind {
            LayerMetricKind::Importance => self.importance_flat.as_deref(),
            LayerMetricKind::QuantNoise => self.noise_flat.as_deref(),
        }
    }
}

/// 엔진 `DType`(7종) → plugin `TensorDtype`(3종) 투영.
///
/// R6 landmine "dtype 어휘 천장": swap-eligible weight 는 F16(primary)/Q4_0(secondary)
/// 뿐이고 `current_format` 의 유일 소비자(swap 어댑터)는 `== Q4_0` 만 본다. Q4_1/Q8_0/
/// BF16/U8 은 swap 컨텍스트에 등장하지 않으며, 도달 시 "비-Q4_0" 의미(swap 제외)만
/// 보존하도록 F16 으로 환원한다.
fn dtype_to_tensor_dtype(d: DType) -> TensorDtype {
    match d {
        DType::Q4_0 => TensorDtype::Q4_0,
        DType::F16 => TensorDtype::F16,
        DType::F32 => TensorDtype::F32,
        DType::Q4_1 | DType::Q8_0 | DType::BF16 | DType::U8 => TensorDtype::F16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qcf_types::{ImportanceEntry, SubLayer};
    use crate::weight::stage_registry::WeightSwapDeciderAsStage;
    use crate::weight::{SwapAlgorithm, WeightSwapDecider};
    use technique_api::{LayerDispatch, WeightStage, WeightStageParams};

    /// `SubLayer::Full` 만 담는 최소 ImportanceLookup mock.
    struct MockImp(Vec<ImportanceEntry>);
    impl ImportanceLookup for MockImp {
        fn is_empty(&self) -> bool {
            self.0.is_empty()
        }
        fn entries(&self) -> &[ImportanceEntry] {
            &self.0
        }
    }
    fn full_entries(vals: &[f32]) -> MockImp {
        MockImp(
            vals.iter()
                .enumerate()
                .map(|(i, &v)| ImportanceEntry {
                    layer_id: i,
                    sublayer: SubLayer::Full,
                    importance: v,
                    opr: 0.0,
                    importance_mean_pool: None,
                    importance_shortgpt_bi: None,
                })
                .collect(),
        )
    }

    /// is_computed / as_slice 를 제어하는 최소 QuantNoiseAccess mock.
    struct MockNoise {
        eps: Vec<f32>,
        computed: bool,
    }
    impl QuantNoiseAccess for MockNoise {
        fn epsilon(&self, i: usize) -> Option<f32> {
            self.eps.get(i).copied()
        }
        fn len(&self) -> usize {
            self.eps.len()
        }
        fn is_computed(&self) -> bool {
            self.computed
        }
        fn as_slice(&self) -> &[f32] {
            &self.eps
        }
    }

    /// ctx 투영이 swap_runtime 호출자 경로와 bit-identical 이고, 그 ctx 로 빌트인
    /// swap stage 가 낸 plan 의 layer 집합 == 동일 입력 `decider.decide(budget)`.
    #[test]
    fn ctx_projection_and_plan_bit_identical_with_swap_runtime() {
        let layer_formats = vec![
            TensorDtype::F16,
            TensorDtype::F16,
            TensorDtype::Q4_0,
            TensorDtype::F16,
        ];
        let n = layer_formats.len();
        let imp = full_entries(&[0.1, 0.5, 0.3, 0.7]);
        let noise = MockNoise {
            eps: vec![0.2, 0.1, 0.3, 0.05],
            computed: true,
        };
        let ratio = 0.5f32;

        let ctx =
            WeightStageModelCtx::from_parts(layer_formats.clone(), &noise, Some(&imp), ratio, 0);

        // ── 1. ctx accessor == swap_runtime 투영 (handle_swap_weights §3–4) ──
        let currently_swapped: Vec<usize> = (0..n)
            .filter(|&i| layer_formats[i] == TensorDtype::Q4_0)
            .collect();
        let importance_flat = flatten_importance(&imp, n);
        let noise_flat: Option<&[f32]> = if noise.is_computed() {
            Some(noise.as_slice())
        } else {
            None
        };
        let target_count = (ratio * n as f32).floor() as usize;
        let budget = target_count.saturating_sub(currently_swapped.len());

        assert_eq!(ctx.n_layers(), n);
        assert_eq!(ctx.budget(), budget);
        assert_eq!(ctx.importance(), Some(importance_flat.as_slice()));
        assert_eq!(ctx.quant_noise(), noise_flat);
        for i in 0..n {
            assert_eq!(ctx.current_format(i), layer_formats[i]);
        }

        // ── 2. stage.plan(&ctx).layers == decider.decide(budget).selected_layers ──
        let stage = WeightSwapDeciderAsStage::new(WeightStageParams {
            allow_boundary_layers: false,
        });
        let plan = stage.plan(&ctx).expect("plan should be Some");
        let stage_layers: Vec<usize> = plan.per_layer.iter().map(|d| d.layer).collect();

        let decider = WeightSwapDecider {
            importance: Some(&importance_flat),
            noise: noise_flat,
            n_decoder_layers: n,
            currently_swapped: &currently_swapped,
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };
        let direct = decider.decide(budget);

        assert_eq!(
            stage_layers, direct.selected_layers,
            "ctx-driven plan layer set must equal swap_runtime decider.decide(budget)"
        );
        // swap = precision F16→Q4_0, dispatch=Full (precision ⊥ dispatch, R1).
        for d in &plan.per_layer {
            assert!(matches!(d.dispatch, LayerDispatch::Full));
            assert_eq!(d.precision, Some(TensorDtype::Q4_0));
        }
    }

    /// noise 미산출(is_computed=false) → `quant_noise() == None`
    /// (decider uniform fallback 게이트 보존). importance 미제공 → `None`.
    #[test]
    fn ctx_noise_none_when_not_computed() {
        let layer_formats = vec![TensorDtype::F16; 4];
        let noise = MockNoise {
            eps: vec![1.0; 4],
            computed: false,
        };
        let ctx = WeightStageModelCtx::from_parts(layer_formats, &noise, None, 0.5, 0);
        assert_eq!(ctx.quant_noise(), None);
        assert_eq!(ctx.importance(), None);
    }

    /// 이미 Q4_0 인 레이어는 budget 산출에서 차감된다(swap_runtime 동형).
    /// 6 레이어, 2개 이미 Q4_0, ratio=0.5 → target=floor(3.0)=3, budget=3−2=1.
    #[test]
    fn ctx_budget_subtracts_currently_swapped() {
        let layer_formats = vec![
            TensorDtype::Q4_0,
            TensorDtype::F16,
            TensorDtype::Q4_0,
            TensorDtype::F16,
            TensorDtype::F16,
            TensorDtype::F16,
        ];
        let noise = MockNoise {
            eps: vec![0.1; 6],
            computed: true,
        };
        let ctx = WeightStageModelCtx::from_parts(layer_formats, &noise, None, 0.5, 0);
        assert_eq!(ctx.budget(), 1);
    }
}
