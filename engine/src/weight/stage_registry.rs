//! weight 축 빌트인 등록 shim (EPIC 3 B3-3).
//!
//! `WeightSwapDeciderAsStage` 어댑터와 `"swap"` `#[distributed_slice(WEIGHT_STAGES)]`
//! 등록은 `weight-swap` 기법 크레이트로 이전됐다. 본 모듈은 엔진 측에 남는 세 가지를
//! 보유한다:
//!
//! - `use weight_swap as _;` — fat-LTO `--gc-sections` 가 외부 rlib 의 linkme 등록을
//!   silent drop 하지 않도록 하는 force-link 앵커 (KV `kv/eviction/stage_registry.rs`
//!   의 `use ::h2o as _;` 거울). **타입 re-export(weight.rs / decider.rs)와는 독립** —
//!   타입 참조만으로는 distributed_slice static 이 링크되지 않는다.
//! - [`WeightSwapDeciderAsStage`] 재export — `stage_ctx.rs` 의 bit-identical 테스트가
//!   `crate::weight::stage_registry::WeightSwapDeciderAsStage` 경로로 참조한다.
//! - [`ensure_builtin_weight_stages_registered`] — startup fail-fast 가드. build_bench_loop
//!   의 has_secondary(swap 구성) 경로에서 live 호출되며(EPIC 3 B3-0), 이 fully-qualified
//!   호출이 본 모듈을 링크-도달 가능하게 유지해 위 force-link 앵커가 DCE 되지 않게 한다.

// force-link 앵커: weight-swap 플러그인의 `#[distributed_slice(WEIGHT_STAGES)]` 등록이
// fat-LTO 에서 살아남도록 한다 (타입 re-export 와 독립적으로 필요).
use weight_swap as _;

pub use weight_swap::WeightSwapDeciderAsStage;

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
                 . weight-swap 플러그인의 #[distributed_slice] 등록이 링크되지 않음."
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// self-test fn 이 swap 등록을 통과시킨다 (엔진 바이너리에서 force-link 앵커가
    /// 플러그인 등록을 링크-도달 가능하게 유지하는지 검증).
    #[test]
    fn ensure_builtin_weight_stages_ok() {
        assert!(ensure_builtin_weight_stages_registered().is_ok());
    }
}
