//! `WeightSwapDeciderAsStage` — weight 축 빌트인 어댑터.
//!
//! KV 거울(`pressure::eviction::stage_registry`)의 weight 판: 엔진 내부
//! `WeightSwapDecider` 를 argus-extension-api 의 plan-returning `WeightStage` 표면으로
//! 감싸 `WEIGHT_STAGES` 슬라이스에 등록한다. 엔진 내부 등록이라 KV 빌트인과
//! 동일하게 force-link 불요(dep 선언만으로 링크).
//!
//! swap = precision F16→Q4_0 (`LayerDirective.precision`); dispatch 는 항상
//! `Full` 이다 — precision(format 축) ⊥ dispatch(stage/hardware 축) 직교(R1).
//!
//! production 호출부 배선은 EPIC 3 B3-0 에서 완료됐다 — `WeightSwapStage::commit` 이
//! `find_weight_stage("swap")` 로 레이어 선택을 라우팅하고, `ensure_builtin_weight_stages_registered`
//! 는 build_bench_loop 의 has_secondary(swap 구성) 경로에서 live 호출된다. 본 모듈은 어댑터 +
//! 등록 + self-test fn(= seam vs 옛 inline decider 등가 anchor) + 단위테스트.

use argus_extension_api::{
    LayerDirective, LayerDispatch, TensorDtype, WEIGHT_STAGES, WeightDispatchPlan, WeightStage,
    WeightStageCtx, WeightStageParams, WeightStageReg,
};
use linkme::distributed_slice;

use crate::weight::{SwapAlgorithm, WeightSwapDecider};

/// `WeightSwapDecider` 를 `WeightStage` 로 노출하는 빌트인 어댑터 (MW-C).
///
/// 상태가 없는 stateless 어댑터다 — 매 `plan()` 호출마다 ctx 의 읽기 값으로
/// decider 를 즉석 생성한다. 누적 상태가 없어 interior-mutability(D4) 불요.
pub struct WeightSwapDeciderAsStage {
    /// 경계 레이어(0, 마지막)도 swap 대상에 포함할지 (연구/ablation; 기본 false).
    allow_boundary_layers: bool,
    /// 레이어 선택 알고리즘 (기본 `ImportanceAware` = production).
    algorithm: SwapAlgorithm,
}

impl WeightSwapDeciderAsStage {
    /// 등록 팩토리에서 호출 — `WeightStageParams` 로부터 어댑터를 만든다.
    /// algorithm 은 production 기본(`ImportanceAware`)으로 고정한다.
    pub fn new(p: WeightStageParams) -> Self {
        Self {
            allow_boundary_layers: p.allow_boundary_layers,
            algorithm: SwapAlgorithm::ImportanceAware,
        }
    }
}

impl WeightStage for WeightSwapDeciderAsStage {
    fn name(&self) -> &str {
        "swap"
    }

    fn plan(&self, ctx: &dyn WeightStageCtx) -> Option<WeightDispatchPlan> {
        let n = ctx.n_layers();
        let budget = ctx.budget();
        if budget == 0 || n == 0 {
            return None;
        }

        // ctx 가 노출하는 flat per-layer 메트릭. decider 의 `Option<&[f32]>`
        // 필드와 동형 — 그대로 전달한다(noise=Some ⟺ is_computed 계약은 엔진의
        // WeightStageCtx impl(MW-D)이 책임진다).
        let importance = ctx.importance();
        let noise = ctx.quant_noise();

        // 현재 이미 Q4_0(=swap 완료) 인 레이어는 재선택 제외.
        let currently_swapped: Vec<usize> = (0..n)
            .filter(|&i| ctx.current_format(i) == TensorDtype::Q4_0)
            .collect();

        let decider = WeightSwapDecider {
            importance,
            noise,
            n_decoder_layers: n,
            currently_swapped: &currently_swapped,
            allow_boundary_layers: self.allow_boundary_layers,
            algorithm: self.algorithm,
        };
        let decision = decider.decide(budget);

        if decision.selected_layers.is_empty() {
            return None;
        }

        // swap = precision F16→Q4_0, dispatch=Full (precision ⊥ dispatch, R1).
        let per_layer = decision
            .selected_layers
            .iter()
            .map(|&l| LayerDirective {
                layer: l,
                dispatch: LayerDispatch::Full,
                precision: Some(TensorDtype::Q4_0),
            })
            .collect();
        Some(WeightDispatchPlan { per_layer })
    }
}

/// 빌트인 weight stage("swap")가 `WEIGHT_STAGES` 에 등록됐는지 단언한다 —
/// weight stage 구성 진입 시 1회 호출(KV `ensure_builtin_stages_registered`
/// 거울). fat-LTO `--gc-sections` 가 linkme 등록을 silent drop 하면 `Err` 로
/// fail-fast 한다.
///
/// EPIC 3 B3-0: build_bench_loop 의 has_secondary(swap 구성) 경로에서 live 호출된다
/// (KV 거울이 build_resilience_cache_manager 에서 호출되는 것과 동형) — `WeightSwapStage::commit`
/// 의 `find_weight_stage("swap").expect(..)` 가 decode-time 패닉이 되기 전에 fail-fast 한다.
pub fn ensure_builtin_weight_stages_registered() -> anyhow::Result<()> {
    for name in ["swap"] {
        if argus_extension_api::find_weight_stage(name).is_none() {
            anyhow::bail!(
                "내장 WeightStage '{name}' 미등록 — linkme fat-LTO --gc-sections silent drop 의심\
                 . weights/stage_registry 의 #[distributed_slice] 등록이 링크되지 않음."
            );
        }
    }
    Ok(())
}

#[distributed_slice(WEIGHT_STAGES)]
static SWAP_STAGE: WeightStageReg = WeightStageReg {
    name: "swap",
    make: |p| Box::new(WeightSwapDeciderAsStage::new(p)),
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{LayerMetricKind, find_weight_stage};

    /// flat importance/noise/budget/current_format 를 직접 보유하는 최소 mock ctx.
    struct MockWeightCtx {
        n_layers: usize,
        budget: usize,
        importance: Option<Vec<f32>>,
        noise: Option<Vec<f32>>,
        /// Q4_0(swap 완료) 로 간주할 레이어 인덱스.
        swapped: Vec<usize>,
    }

    impl WeightStageCtx for MockWeightCtx {
        fn n_layers(&self) -> usize {
            self.n_layers
        }
        fn budget(&self) -> usize {
            self.budget
        }
        fn pressure(&self) -> u8 {
            0
        }
        fn current_format(&self, layer: usize) -> TensorDtype {
            if self.swapped.contains(&layer) {
                TensorDtype::Q4_0
            } else {
                TensorDtype::F16
            }
        }
        fn layer_metric(&self, kind: LayerMetricKind) -> Option<&[f32]> {
            match kind {
                LayerMetricKind::Importance => self.importance.as_deref(),
                LayerMetricKind::QuantNoise => self.noise.as_deref(),
            }
        }
    }

    /// stage `plan()` 의 선택 layer 집합 == 동일 입력으로 직접 호출한
    /// `decider.decide(budget).selected_layers` (bit-identical).
    #[test]
    fn stage_plan_matches_decider() {
        let importance = vec![0.1f32, 0.5, 0.3, 0.7];
        let noise = vec![0.2f32, 0.1, 0.3, 0.05];
        let ctx = MockWeightCtx {
            n_layers: 4,
            budget: 2,
            importance: Some(importance.clone()),
            noise: Some(noise.clone()),
            swapped: Vec::new(),
        };

        let stage = WeightSwapDeciderAsStage::new(WeightStageParams {
            allow_boundary_layers: false,
        });
        let plan = stage.plan(&ctx).expect("plan should be Some");
        let stage_layers: Vec<usize> = plan.per_layer.iter().map(|d| d.layer).collect();

        // 동일 입력으로 decider 직접 호출.
        let currently_swapped: Vec<usize> = Vec::new();
        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &currently_swapped,
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };
        let direct = decider.decide(2);

        assert_eq!(
            stage_layers, direct.selected_layers,
            "stage plan layer set must equal decider.decide(budget).selected_layers"
        );
        // precision=Q4_0, dispatch=Full (R1 직교).
        for d in &plan.per_layer {
            assert!(matches!(d.dispatch, LayerDispatch::Full));
            assert_eq!(d.precision, Some(TensorDtype::Q4_0));
        }
    }

    /// B3-0 위험(HIGH): live commit 은 `read_allow_boundary_env()` 를 1회 읽어
    /// `WeightStageParams.allow_boundary_layers` 로 주입한다(plan() 내부 env 재독 금지). 어댑터가
    /// 그 flag 를 decider 로 정확히 threading 하는지 true/false 양쪽에서 검증한다 — 동일 flag 의
    /// 직접 decider 와 selected_layers 가 **순서까지** bit-identical(Vec== 는 order-sensitive).
    #[test]
    fn stage_plan_threads_allow_boundary_both_ways() {
        // 경계 레이어(0, 3)가 high importance → allow_boundary flag 가 선택을 가른다.
        let importance = vec![0.9f32, 0.2, 0.3, 0.8];
        let noise = vec![0.1f32, 0.1, 0.1, 0.1];
        for allow_boundary in [false, true] {
            let ctx = MockWeightCtx {
                n_layers: 4,
                budget: 2,
                importance: Some(importance.clone()),
                noise: Some(noise.clone()),
                swapped: Vec::new(),
            };
            let stage = WeightSwapDeciderAsStage::new(WeightStageParams {
                allow_boundary_layers: allow_boundary,
            });
            let stage_layers: Vec<usize> = stage
                .plan(&ctx)
                .map(|p| p.per_layer.iter().map(|d| d.layer).collect())
                .unwrap_or_default();

            let currently_swapped: Vec<usize> = Vec::new();
            let decider = WeightSwapDecider {
                importance: Some(&importance),
                noise: Some(&noise),
                n_decoder_layers: 4,
                currently_swapped: &currently_swapped,
                allow_boundary_layers: allow_boundary,
                algorithm: SwapAlgorithm::ImportanceAware,
            };
            let direct = decider.decide(2);

            assert_eq!(
                stage_layers, direct.selected_layers,
                "allow_boundary={allow_boundary}: stage plan 의 layer 집합(순서 포함)이 동일 \
                 flag 의 decider.decide(budget) 와 bit-identical 이어야 한다"
            );
        }
    }

    /// budget=0 → None (no-op).
    #[test]
    fn stage_plan_zero_budget_is_none() {
        let ctx = MockWeightCtx {
            n_layers: 4,
            budget: 0,
            importance: Some(vec![0.1, 0.5, 0.3, 0.7]),
            noise: Some(vec![0.2, 0.1, 0.3, 0.05]),
            swapped: Vec::new(),
        };
        let stage = WeightSwapDeciderAsStage::new(WeightStageParams {
            allow_boundary_layers: false,
        });
        assert!(stage.plan(&ctx).is_none());
    }

    /// 이미 swap 완료(Q4_0)된 레이어는 currently_swapped 로 제외된다 — decider 와 동형.
    #[test]
    fn stage_plan_excludes_currently_swapped() {
        let importance = vec![0.1f32, 0.5, 0.3, 0.7];
        let noise = vec![0.2f32, 0.1, 0.3, 0.05];
        let ctx = MockWeightCtx {
            n_layers: 4,
            budget: 1,
            importance: Some(importance.clone()),
            noise: Some(noise.clone()),
            swapped: vec![2],
        };
        let stage = WeightSwapDeciderAsStage::new(WeightStageParams {
            allow_boundary_layers: false,
        });
        let plan = stage.plan(&ctx).expect("plan should be Some");
        let stage_layers: Vec<usize> = plan.per_layer.iter().map(|d| d.layer).collect();

        let currently_swapped = vec![2usize];
        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &currently_swapped,
            allow_boundary_layers: false,
            algorithm: SwapAlgorithm::ImportanceAware,
        };
        let direct = decider.decide(1);

        assert_eq!(stage_layers, direct.selected_layers);
        assert!(
            !stage_layers.contains(&2),
            "swapped layer 2 must be excluded"
        );
    }

    /// "swap" 이 `WEIGHT_STAGES` 에 등록돼 있고 팩토리가 동작한다.
    #[test]
    fn swap_registered_in_slice() {
        let reg = find_weight_stage("swap").expect("'swap' 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "swap");
        let stage = (reg.make)(WeightStageParams {
            allow_boundary_layers: false,
        });
        assert_eq!(stage.name(), "swap");
    }

    /// self-test fn 이 swap 등록을 통과시킨다.
    #[test]
    fn ensure_builtin_weight_stages_ok() {
        assert!(ensure_builtin_weight_stages_registered().is_ok());
    }
}
