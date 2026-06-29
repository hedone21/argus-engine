//! Built-in eviction-technique force-link anchors + the [`StageBackedPolicy`] reverse-adapter that
//! exposes a v3 [`KVMutationStage`] as the legacy [`EvictionPolicy`] surface (the eviction
//! `CacheManager` path drives the v3 handle internally via [`drive_mutation_layer`]).
//!
//! Each built-in technique (sliding/streaming/h2o/h2o_plus/d2o/none, + caote/rkv/pyramidkv under
//! features) lives in its own `crates/techniques/*` crate and self-registers into
//! [`KV_MUTATION_STAGES`](argus_extension_api::KV_MUTATION_STAGES) via `#[distributed_slice]`. The
//! `use X as _;` force-links below make those registrations survive fat-LTO `--gc-sections`;
//! [`ensure_builtin_stages_registered`] fail-fasts at CacheManager build if any were dropped.
//!
//! The [`stage_is_score_based`] / [`stage_default_protected_prefix`] / [`stage_produces_merge_plan`]
//! lookups read each technique's declared [`StageCaps`] through
//! [`mutation_stage_caps`](argus_extension_api::mutation_stage_caps) so the CLI/chat/eval/bench paths
//! never name a plugin.

use anyhow::Result;
use argus_extension_api::{
    KVMutationStage, StageCaps, StageParams, TensorKind, find_mutation_stage, find_qcf_estimator,
    mutation_stage_caps,
};

use super::EvictionPolicy;
use crate::kv::kv_cache::KVCache;
use crate::stages::kv::mutation::drive_mutation_layer;

// value-aware production 활성화. feature `caote` ON 시 caote crate 를 force-link 한다 —
// dep 선언만으로는 미참조 rlib 이 링크 제외돼 `#[distributed_slice]` 등록이 누락되기 때문이다.
// 이 1줄이 production 바이너리에서 `find_mutation_stage("caote")` 를 가시화한다(session score_based
// 경유 value-aware 동작). feature OFF = 미링크 + `eviction caote` 서브커맨드 부재(clap reject).
#[cfg(feature = "caote")]
use caote as _;

// StreamingLLM production force-link. Extracted from the engine core into the `streaming-llm`
// technique crate; the dep declaration alone leaves the unreferenced rlib out of the link, so
// this one line makes `find_mutation_stage("streaming")` visible (the `#[distributed_slice]` registration).
use streaming_llm as _;

// heavy-hitter production force-link. Extracted from the engine core into the `h2o` technique crate;
// makes `find_mutation_stage("h2o")` visible (same force-link rationale as streaming above).
use ::h2o as _;

// weighted-merge production force-link. Extracted from the engine core into the `d2o` technique crate
// (registers "d2o", a WeightedMerge-producing stage); makes `find_mutation_stage("d2o")` visible.
// Production resolves it via `make_stage_backed_policy("d2o", &params, &blob)`
// (eval_setup/build_bench_loop/chat), with the d2o-private knobs in the StageArgs blob; the
// registration must survive fat-LTO.
use d2o as _;

// Sliding-window + no-eviction production force-link. Extracted from the engine core into the
// `sliding-window` (registers "sliding") and `no-eviction` (registers "none") technique crates; the
// dep declaration alone leaves the unreferenced rlib out of the link, so these lines make
// `find_mutation_stage("sliding")` / `find_mutation_stage("none")` visible (same rationale as streaming/h2o above).
use no_eviction as _;
use sliding_window as _;

// Layer-importance scorers force-link (observer/score axis, EPIC 2 Stage B). The per-layer
// importance formulas (mean_pool / shortgpt_bi) were extracted into the `layer-importance` crate;
// this one line makes their `#[distributed_slice(LAYER_SCORERS)]` registration visible to
// `find_layer_scorer` (same fat-LTO --gc-sections rationale as the stages above). `mean_pool` is the
// default `--importance-formula`, so the crate is non-optional.
use layer_importance as _;

// Attention-score producer force-link (observer/score axis, EPIC 2 Stage C). The forward-time
// score-accumulation policy (per-layer MAX / GQA averaging / value-aware overwrite / forgetting-factor decay / SUM /
// time-norm) was extracted into the `attn-score` crate; this one line makes its
// `#[distributed_slice(SCORE_PRODUCERS)]` registration visible to `find_score_producer` (same
// fat-LTO --gc-sections rationale as above). `attn_score` is the default scoring path → non-optional.
use attn_score as _;

// R-KV measurement force-link (feature `rkv`). Extracted from the engine core into the `rkv`
// technique crate (registers "rkv"); feature OFF = unlinked + `eviction rkv` subcommand absent.
#[cfg(feature = "rkv")]
use rkv as _;

// PyramidKV force-link (feature `pyramidkv`). Registers "pyramidkv" with `caps.reads ∋
// PrefillAttention`, which arms the prefill-keepset producer (`build_standard_loop` /
// `find_prefill_attn_stage_name`) so PyramidKV runs at prefill end. Feature OFF = unlinked = the
// PFA path stays dormant (byte-identical to before).
#[cfg(feature = "pyramidkv")]
use pyramidkv as _;

/// (dormant) Pre-make compatibility check: reject a stage whose declared capabilities the current
/// container cannot execute, BEFORE instantiation — surfacing the expressible-vs-executable boundary
/// at make time instead of as a runtime executor reject (the channel-axis honest-reject precedent).
/// Today it checks the one pre-make-knowable constraint: a merge-producing stage
/// ([`StageCaps::produces_merge_plan`]) needs CPU-resident KV (the merge executor is host-only).
///
/// DORMANT: not yet wired into the production make path (zero behavior change); the CacheHandle /
/// mutation driver consult it as the constraint surface matures. Exercised by its unit test; the
/// `allow(dead_code)` marks the deliberate dormancy (mirroring the ChannelKeep dormant-surface
/// precedent) rather than hiding an oversight.
///
/// ⚠ Wiring it as a HARD make-time reject is NOT behavior-neutral: a self-degrading stage (e.g. d2o,
/// which sets `produces_merge_plan: true` statically but at runtime emits a keep-only plan on
/// device-resident KV via `merge_enabled = !kv_on_device()`) currently WORKS on device. Feeding its
/// caps here with `supports_merge=false` returns `Err`, regressing it from works→fail. Before wiring,
/// treat this as advisory (degrade-to-keep-only), or drive it off an effective/runtime capability
/// rather than the static `produces_merge_plan`.
#[allow(dead_code)]
pub(crate) fn validate_stage_constraints(
    caps: &argus_extension_api::StageCaps,
    supports_merge: bool,
) -> Result<(), String> {
    if caps.produces_merge_plan && !supports_merge {
        return Err(
            "stage produces weighted merges but the cache is device-resident (the merge executor is \
             CPU-only); degrade to a keep-only plan or place KV on host"
                .to_string(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod constraint_validator_tests {
    use super::validate_stage_constraints;
    use argus_extension_api::{StageCaps, TensorKind};

    /// A merge-producing stage is rejected on device-only KV, accepted on host KV; a drop-only stage
    /// is always accepted. Mutation-proof: dropping the `produces_merge_plan && !supports_merge` guard
    /// makes the device-merge case return Ok, failing the first assert.
    #[test]
    fn merge_stage_requires_host_residency() {
        let merge_caps = StageCaps {
            reads: &[TensorKind::Scores],
            default_protected_prefix: 4,
            produces_merge_plan: true,
        };
        assert!(validate_stage_constraints(&merge_caps, false).is_err()); // device → reject
        assert!(validate_stage_constraints(&merge_caps, true).is_ok()); // host → ok
        assert!(validate_stage_constraints(&StageCaps::SCORE_FREE, false).is_ok()); // drop-only → ok
    }
}

/// Exposes a v3 [`KVMutationStage`] as the legacy [`EvictionPolicy`] surface (the reverse-adapter).
///
/// The production eviction path (`run_policy_eviction` → `evict*`) keeps its structure; internally it
/// drives the stage's imperative `on_phase` through the transactional [`CacheHandle`] via
/// [`drive_mutation_layer`]. Byte-identical to the prior in-place `evict*` (the Phase-1 gate).
pub struct StageBackedPolicy {
    stage: Box<dyn KVMutationStage>,
    /// The stage's declared capabilities (from its `MutationStageReg`) — gates the per-layer raw K/V
    /// dequant snapshot in [`drive_mutation_layer`] (e.g. caote's `Value` read).
    caps: StageCaps,
}

impl StageBackedPolicy {
    /// 주어진 v3 [`KVMutationStage`] + 그 caps 를 `EvictionPolicy` 표면으로 감싼다.
    pub fn new(stage: Box<dyn KVMutationStage>, caps: StageCaps) -> Self {
        Self { stage, caps }
    }

    /// Drive the stage's imperative `on_phase` through the v3 [`CacheHandle`]
    /// ([`drive_mutation_layer`]) — byte-identical to the prior in-place eviction it replaced
    /// (the Phase-1 decision-equivalence gate). `layer_idx`/`n_layers` 는 per-layer 기법(d2o
    /// protected_layers/last-layer protect)용 — 비-layer 인지 호출자(직접 evict)는 `(0, 1)` 단일-layer
    /// 뷰를 쓴다. `last_attn`(AttnWeights, value-aware a_i)는 score accumulator 의
    /// last_step_head_attn 을 공급할 때 Some — value-aware 기법(caote)이 읽는다.
    fn run(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: Option<&[f32]>,
        last_attn: Option<&[f32]>,
        layer_idx: usize,
        n_layers: usize,
    ) -> Result<()> {
        drive_mutation_layer(
            self.stage.as_ref(),
            &self.caps,
            cache,
            layer_idx,
            n_layers,
            target_len,
            importance,
            None, // flat-score / value-aware path: no per-head scores (see evict_with_head_scores)
            last_attn,
        )
        .map(|_mutated| ())
    }
}

impl EvictionPolicy for StageBackedPolicy {
    fn should_evict(&self, _cache: &KVCache, _mem_available: usize) -> bool {
        // WHEN(트리거)은 엔진 소유 — `run_policy_eviction` 의 target_len/MIN_EVICT
        // 가드가 결정한다. 프로덕션 미호출(should_evict 의미는 구체 정책 테스트에서 검증). 엔진 위임.
        true
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        self.run(cache, target_len, None, None, 0, 1)
    }

    fn evict_with_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: &[f32],
    ) -> Result<()> {
        self.run(cache, target_len, Some(importance), None, 0, 1)
    }

    /// Per-layer eviction: thread the real `(layer_idx, n_layers)` + the optional `last_attn`
    /// (value-aware's `a_i`) into the stage ctx so per-layer / value-aware techniques (d2o
    /// `protected_layers` / last-layer protection, caote attention-weighted criticality) see them.
    fn evict_layer(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: Option<&[f32]>,
        last_attn: Option<&[f32]>,
        layer_idx: usize,
        n_layers: usize,
    ) -> Result<()> {
        self.run(
            cache, target_len, importance, last_attn, layer_idx, n_layers,
        )
    }

    /// Per-KV-head eviction (stage ⑤ / F5): route the per-head accumulated importance
    /// (`[n_kv_heads * max_seq]`, row-major) into the stage ctx as `tensor(Scores)` so a per-head
    /// stage (h2o_plus) sees `ctx.head_score(kv_head, pos)` and stages a per-head keep; the handle
    /// then compacts each head independently. `flat_importance` remains available via
    /// `ctx.importance()` for the stage's score-free / flat fallback.
    fn evict_with_head_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        flat_importance: &[f32],
        head_importance: &[f32],
        _n_kv_heads: usize,
        layer_idx: usize,
        n_layers: usize,
    ) -> Result<()> {
        // Per-head scores reach the stage as `tensor(Scores)` (`head_importance`), the flat fallback as
        // `importance()` (`flat_importance`); a per-head stage (h2o_plus) emits a per-head keep applied
        // head-independently by the handle. Byte-identical to the prior in-place per-head eviction.
        drive_mutation_layer(
            self.stage.as_ref(),
            &self.caps,
            cache,
            layer_idx,
            n_layers,
            target_len,
            Some(flat_importance),
            Some(head_importance),
            None,
        )
        .map(|_mutated| ())
    }

    fn name(&self) -> &str {
        self.stage.name()
    }
}

/// Resolve a v3 [`KVMutationStage`] by name (static linkme + dynamic `--load-plugin`) and wrap it,
/// with its declared `caps`, as a legacy [`EvictionPolicy`] via [`StageBackedPolicy`] — the production
/// constructor for the eviction `CacheManager`'s policy after the v2 plan path was retired (the
/// eviction kernel `run_policy_eviction` then drives the v3 handle internally). `None` for an unknown /
/// non-v3 name.
pub fn make_stage_backed_policy(
    name: &str,
    params: &StageParams,
    args: argus_extension_api::StageArgs<'_>,
) -> Option<Box<dyn EvictionPolicy>> {
    let reg = find_mutation_stage(name)?;
    Some(Box::new(StageBackedPolicy::new(
        (reg.make)(*params, args),
        reg.caps,
    )))
}

/// Test helper: build the out-of-tree `h2o` stage wrapped as a legacy [`EvictionPolicy`]
/// (`StageBackedPolicy`). Used by CacheManager / EvictionHandler tests after heavy-hitter was extracted
/// to the `h2o` plugin crate — production resolves "h2o" the same way (`make_stage_backed_policy`).
#[cfg(test)]
pub(crate) fn h2o_backed_policy(
    hh_size: usize,
    recent_size: usize,
    protected_prefix: usize,
) -> Box<dyn EvictionPolicy> {
    // Faithful H2O takes ABSOLUTE budgets via the `--set` blob (hh_size/recent_size), not a ratio.
    // Test-only: leak the formatted values for the 'static StageArgs lifetime.
    let p = StageParams {
        protected_prefix,
        ..Default::default()
    };
    let hh: &'static str = Box::leak(hh_size.to_string().into_boxed_str());
    let recent: &'static str = Box::leak(recent_size.to_string().into_boxed_str());
    let args = [
        argus_extension_api::PluginArg {
            key: "hh_size",
            val: hh,
        },
        argus_extension_api::PluginArg {
            key: "recent_size",
            val: recent,
        },
    ];
    make_stage_backed_policy("h2o", &p, &args)
        .expect("h2o v3 stage registered (force-linked h2o plugin)")
}

/// Build the out-of-tree `sliding` stage wrapped as a legacy [`EvictionPolicy`]
/// (`StageBackedPolicy`). Convenience constructor used by CacheManager / EvictionHandler tests (and
/// engine integration tests) after SlidingWindowPolicy was extracted to the `sliding-window` plugin
/// crate — production resolves "sliding" the same way (`make_stage_backed_policy`).
pub fn sliding_backed_policy(window: usize, protected_prefix: usize) -> Box<dyn EvictionPolicy> {
    let p = StageParams {
        eviction_window: window,
        protected_prefix,
        ..Default::default()
    };
    make_stage_backed_policy("sliding", &p, &[])
        .expect("sliding v3 stage registered (force-linked sliding-window plugin)")
}

/// Build the out-of-tree `none` stage wrapped as a legacy [`EvictionPolicy`] (`StageBackedPolicy`)
/// — a no-op policy. Convenience constructor used by tests after NoEvictionPolicy was extracted to
/// the `no-eviction` plugin crate; production resolves "none" the same way.
pub fn none_backed_policy() -> Box<dyn EvictionPolicy> {
    make_stage_backed_policy("none", &StageParams::default(), &[])
        .expect("none v3 stage registered (force-linked no-eviction plugin)")
}

/// Whether the named stage is score-based (consumes importance) — the generic capability lookup the
/// CLI/chat/eval/bench paths use instead of `matches!(name, "h2o" | "d2o" | "caote" | "rkv" | ...)`.
/// Reads the plugin's declared [`StageCaps`](argus_extension_api::stage_caps). Unknown /
/// unregistered (incl. dynamic `.so` stages whose caps don't cross the ABI yet) → `false`.
pub fn stage_is_score_based(name: &str) -> bool {
    mutation_stage_caps(name)
        .map(|c| !c.reads.is_empty())
        .unwrap_or(false)
}

/// The default `--protected-prefix` the named stage declares (`4` for score-based, `0` = "engine
/// picks its own fallback"). The generic lookup that replaces the `match name { ... => 4 }` prefix
/// tables. Reads the plugin's declared [`StageCaps`]. Unknown → `0`.
pub fn stage_default_protected_prefix(name: &str) -> usize {
    mutation_stage_caps(name)
        .map(|c| c.default_protected_prefix)
        .unwrap_or(0)
}

/// Whether the named stage's `plan()` may emit a weighted-merge plan (à la weighted-merge). The generic lookup
/// the eval/QCF path uses instead of the `eviction_policy() == "d2o"` name match — it selects a
/// merge-compensation estimator + K readback. Reads the plugin's declared [`StageCaps`]. Unknown →
/// `false` (pure-drop).
pub fn stage_produces_merge_plan(name: &str) -> bool {
    mutation_stage_caps(name)
        .map(|c| c.produces_merge_plan)
        .unwrap_or(false)
}

/// Asserts every force-linked built-in technique crate registered into `KV_MUTATION_STAGES` —
/// called once at eviction CacheManager build. If fat-LTO `--gc-sections` silently drops a
/// force-linked crate's linkme registration, `mutation_stage_caps` stops resolving its name and we
/// `Err` fail-fast (no silent policy-name fallthrough in release). The caps semantics
/// (is_score_based/protected_prefix/produces_merge_plan) are owned solely by each plugin, so this
/// only verifies registration existence (resolution), never re-declares them.
pub fn ensure_builtin_stages_registered() -> Result<()> {
    // The force-linked built-in technique crate names (the `use X as _;` block above). This list is
    // the fail-fast ANCHOR: it can't be derived from the registry, because the registry is exactly
    // what we verify — if fat-LTO `--gc-sections` drops a crate, `mutation_stage_caps` stops resolving
    // its name and we bail. It does NOT re-declare any plugin's caps (those are read from the registry
    // by `stage_is_score_based` / `stage_default_protected_prefix` / `stage_produces_merge_plan` and
    // owned solely by the plugin). Mirrors `ensure_score_producers_registered` /
    // `ensure_layer_scorers_registered`, which likewise keep a hardcoded name list + assert only
    // resolution.
    for name in ["sliding", "streaming", "h2o", "d2o"] {
        if mutation_stage_caps(name).is_none() {
            anyhow::bail!(
                "built-in KV mutation stage '{name}' not registered — suspect linkme fat-LTO \
                 --gc-sections silent drop of its force-linked crate (the #[distributed_slice] \
                 registration in the stage crate was not linked; see the `use X as _;` force-links)."
            );
        }
    }

    // QCF estimators (observer/score axis, EPIC 2): each eviction technique crate also registers a
    // QcfEstimator into QCF_ESTIMATORS via the same force-link as the stages above. A missing entry
    // is the same fat-LTO --gc-sections silent-drop risk — fail fast, checking the declared curve key.
    for (name, want_curve_key) in [
        ("sliding", "kv.evict_sliding"),
        ("streaming", "kv.evict_streaming"),
        ("h2o", "kv.evict_h2o"),
        ("d2o", "kv.merge_d2o"),
    ] {
        let Some(reg) = find_qcf_estimator(name) else {
            anyhow::bail!(
                "QCF estimator '{name}' not registered — suspect linkme fat-LTO --gc-sections \
                 silent drop of its QCF_ESTIMATORS registration."
            );
        };
        if reg.curve_key != want_curve_key {
            anyhow::bail!(
                "QCF estimator '{name}' declares curve_key='{}' but the engine expects \
                 '{want_curve_key}'.",
                reg.curve_key
            );
        }
    }

    Ok(())
}

/// R-P1-1: the first registered KV mutation stage that reads `TensorKind::PrefillAttention` (else
/// `None`) — the `build_standard_loop` PFA-producer arming gate, caps-driven (plugin-name agnostic,
/// the twin of `wants_query_stats`). With no such built-in it is `None` (arming dormant →
/// byte-identical); a per-head keep-set plugin registered with `caps.reads ∋ PrefillAttention`
/// (pyramidkv) activates it.
pub fn find_prefill_attn_stage_name() -> Option<String> {
    argus_extension_api::KV_MUTATION_STAGES
        .iter()
        .find(|r| r.caps.reads.contains(&TensorKind::PrefillAttention))
        .map(|r| r.name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::kv::cache_handle::EngineCacheHandle;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use argus_extension_api::{CacheHandle, find_mutation_stage, registered_mutation_names};
    use std::sync::Arc;

    #[test]
    fn builtins_registered() {
        // linkme 가 엔진의 v3 등록을 슬라이스로 모으는지 (fat-LTO 생존은 release self-test).
        let names = registered_mutation_names();
        for n in ["sliding", "streaming", "h2o"] {
            assert!(
                names.contains(&n),
                "'{n}' 등록 누락 (linkme distributed_slice)"
            );
        }
    }

    #[test]
    fn d2o_stage_registered() {
        // D2OStage 가 "d2o" 로 KV_MUTATION_STAGES 에 등록됐는지 — find_mutation_stage 해석 + make 가능
        // (release fat-LTO 에서도 생존해야). make 로 D2OStage 인스턴스 생성 가능 확인.
        let reg = find_mutation_stage("d2o").expect("d2o stage 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "d2o");
        let params = StageParams {
            eviction_window: 0,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 0,
            streaming_window: 0,
        };
        let stage = (reg.make)(params, &[]);
        assert_eq!(stage.name(), "d2o");
    }

    // cross-crate linkme 실증 결과(M3): **dev-dep 선언만으로는 부족**하다. Rust 는 미참조
    // 의존 rlib 을 링크에서 제외하므로 `#[distributed_slice]` 등록이 누락된다(실측 — forcing 없으면
    // find_mutation_stage None). 따라서 technique crate 의 등록을 활성화하려면 의존 1줄에 더해 **force-link
    // 참조 1줄**(`use <crate> as _;`)이 designated 지점에 필요하다. 즉 확장 비용 = dep 1줄 + force-link
    // 1줄(둘 다 기계적, 기존 로직 수정 0 → OCP 유지). 상세: (M3 정정).
    use example_keep_recent as _;
    // value-aware 의 force-link 는 production(module-level `#[cfg(feature = "caote")] use caote as _`)
    // 가 담당한다 — `--features caote` 테스트 시 그 cfg 가 활성이라 별도 test-only force-link 불필요.

    #[test]
    fn example_technique_crate_visible_to_engine() {
        // force-link(위 `use ... as _`) 가 걸린 상태에서 별도 technique crate 의 등록이 엔진 뷰의
        // KV_MUTATION_STAGES 에 나타나는가 — "폴더 추가 + dep 1줄 + force-link 1줄 = 기법 추가" 검증.
        assert!(
            find_mutation_stage("example_keep_recent").is_some(),
            "force-link 후 예제 technique crate 등록이 엔진에서 보여야 한다"
        );
    }

    /// W-REROTATE invariant (Finch / KeyRerotation는 RESTRICTED): production eviction
    /// executor 는 생존 키를 압축 슬롯으로 옮길 때 **순수 byte memmove** 만 하고 새 위치로
    /// **재회전(re-rotate)하지 않는다**. 따라서 생존 키는 *원래 절대 write 위치*에 baked-in 된
    /// RoPE phase 를 그대로 보존하고, 다음 토큰의 RoPE 는 압축된 `current_pos` 가 아니라 진짜
    /// 시퀀스 위치에서 이어지므로(session/eval/eval_loop.rs:299-303,
    /// session/ppl/runner.rs:1160-1166) 학습된 상대거리
    /// (query_pos − key_orig_pos)가 그대로 유지된다 = NLL-correct. 재회전(생존자를 0..keep.len()
    /// 로 renumber + 그 위치로 RoPE 재적용)은 the v2 plan(keep/merges/channels)에 표현할 verb 가
    /// 없고 엔진은 어디서도 수행하지 않는다. 이 테스트는 그 부재를 핀한다 — eviction 시 재회전을
    /// 도입하는 어떤 경로가 생기는 즉시 실패한다.
    #[test]
    fn eviction_keeps_survivor_rope_phase_does_not_rerotate() {
        use crate::backend::Backend;
        let be = Arc::new(CpuBackend::new());
        let theta = 10_000.0f32;

        // 고정 base 키를 절대 위치 `p` 로 RoPE 회전 — forward 경로(forward_gen_fmt.rs:160-161
        // `rope_inplace(&mut k_rope, start_pos, ..)`)와 동일. 결정적이라 같은 p 는 항상 byte-identical.
        let base: Vec<f32> = (0..PHD).map(|d| (d as f32) * 0.1 + 1.0).collect();
        let rotated_at = |p: usize| -> Vec<f32> {
            let buf = Arc::new(SharedBuffer::new(PHD * 4, DType::F32));
            let mut t = Tensor::new(Shape::new(vec![1, 1, 1, PHD]), buf, be.clone());
            t.as_mut_slice::<f32>().copy_from_slice(&base);
            be.rope_inplace(&mut t, p, theta).unwrap();
            t.as_slice::<f32>().to_vec()
        };

        // 8 개 키, 각자 자기 절대 위치 0..8 로 회전해 적재.
        let n = 8usize;
        let mut c = mk(DType::F32, n);
        for p in 0..n {
            let off = c.offset(p, 0);
            let rk = rotated_at(p);
            c.k_buffer.as_mut_slice::<f32>()[off..off + PHD].copy_from_slice(&rk);
        }

        // v3 핸들 keep 으로 eviction — 생존자가 물리적으로 이동(slot ≠ 원위치)하도록 흩어진
        // keep-set 선택.
        let keep = vec![0usize, 3, 5, 7];
        {
            let mut h = EngineCacheHandle::new(&mut c, 0, 1);
            h.keep(&keep).unwrap();
            assert!(h.commit().unwrap());
        }
        assert_eq!(c.current_pos(), keep.len(), "keep.len() 로 compact");

        let k = c.k_buffer.as_slice::<f32>().to_vec();
        for (new_slot, &orig_pos) in keep.iter().enumerate() {
            let off = c.offset(new_slot, 0);
            let stored = &k[off..off + PHD];

            // (1) 생존자는 *원래 위치* 의 RoPE phase 를 그대로 유지 — memmove 가 byte-for-byte,
            //     재회전 없음.
            assert_eq!(
                stored,
                rotated_at(orig_pos).as_slice(),
                "slot {new_slot} 생존자는 원위치 {orig_pos} 의 RoPE phase 를 보존해야 함"
            );

            // (2) ...그리고 새(압축) 슬롯으로 재회전됐을 때의 phase 와는 **다르다**. 이동한
            //     생존자(slot ≠ 원위치)에선 두 phase 가 달라 이게 결정적인 "재회전 없음" 증거다.
            if new_slot != orig_pos {
                assert_ne!(
                    stored,
                    rotated_at(new_slot).as_slice(),
                    "slot {new_slot} 생존자가 압축 위치로 재회전되면 안 됨"
                );
            }
        }
    }

    // The rkv visibility/execute test moved to the `rkv` technique crate (it owns RkvStage now); the
    // adapter-vs-plan_keep and World-A↔B sliding parity tests were removed when SlidingWindowPolicy
    // was extracted to the `sliding-window` plugin crate — the plugin is plan-only (no in-place
    // evict/plan_keep), so there is no World-A path left to compare. The plugin's keep-list spec is
    // pinned by its own unit tests, and beta3_eviction_stage_equivalence.rs proves the World-B
    // application end-to-end. (Streaming/h2o were retired the same way.)

    const PHD: usize = 32; // head_dim = QK4_0 → Q4_0 위치당 1 block
    const PMAX: usize = 128;

    fn pbytes(dt: DType) -> usize {
        match dt {
            DType::F32 => PHD * 4,
            DType::F16 => PHD * 2,
            DType::Q4_0 => {
                (PHD / crate::quant::QK4_0) * std::mem::size_of::<crate::quant::BlockQ4_0>()
            }
            o => panic!("unsupported dtype {o:?}"),
        }
    }

    /// 위치 p 의 모든 byte = (p+1) (K), +128 (V) — distinct 라 잘못된 keep 은 byte 비교로 잡힘.
    fn mk(dt: DType, n: usize) -> KVCache {
        let bpp = pbytes(dt);
        let kb = Arc::new(SharedBuffer::new(PMAX * bpp, dt));
        let vb = Arc::new(SharedBuffer::new(PMAX * bpp, dt));
        unsafe {
            let (kp, vp) = (kb.as_mut_ptr(), vb.as_mut_ptr());
            for p in 0..n {
                let byte = (p + 1) as u8;
                for b in 0..bpp {
                    *kp.add(p * bpp + b) = byte;
                    *vp.add(p * bpp + b) = byte.wrapping_add(128);
                }
            }
        }
        let be = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, PMAX, 1, PHD]);
        let mut c = KVCache::new(
            Tensor::new(sh.clone(), kb, be.clone()),
            Tensor::new(sh, vb, be),
            PMAX,
        );
        c.current_pos = n;
        c
    }
}

#[cfg(test)]
mod kv_on_device_unification_tests {
    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::kv::cache_handle::EngineCacheHandle;
    use crate::kv::kv_cache::KVCache;
    use crate::shape::Shape;
    use crate::stages::kv::mutation::SnapshotStageCtx;
    use crate::tensor::Tensor;
    use argus_extension_api::{CacheHandle, StageCtx};
    use std::any::Any;
    use std::sync::Arc;

    /// A UMA-shaped mock buffer: a GPU buffer (`is_gpu_buffer()==true`) that is ALSO host-mapped
    /// (`as_ptr()` non-null) — exactly the Adreno `UnifiedBuffer` case where the old
    /// `as_ptr().is_null()` predicate and `is_gpu_buffer()` disagree. Backed by a real `Vec` so the
    /// pointer is valid + non-null.
    struct UmaMockBuffer {
        data: Vec<u8>,
    }
    impl Buffer for UmaMockBuffer {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn size(&self) -> usize {
            self.data.len()
        }
        fn as_ptr(&self) -> *const u8 {
            self.data.as_ptr() // non-null: host-mapped, like a mapped UMA buffer
        }
        fn as_mut_ptr(&self) -> *mut u8 {
            self.data.as_ptr() as *mut u8
        }
        #[cfg(feature = "opencl")]
        fn cl_mem(&self) -> Option<&ocl::core::Mem> {
            None
        }
        #[cfg(not(feature = "opencl"))]
        fn cl_mem(&self) -> Option<()> {
            None
        }
        fn sync_device(&self) -> anyhow::Result<()> {
            Ok(())
        }
        fn is_gpu_buffer(&self) -> bool {
            true // UMA buffers ARE GPU buffers (the discriminator vs. as_ptr().is_null())
        }
    }

    fn uma_cache() -> KVCache {
        let (kv_heads, head_dim, max_seq) = (2usize, 4usize, 8usize);
        let bytes = max_seq * kv_heads * head_dim * 4;
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, max_seq, kv_heads, head_dim]);
        let mk = || {
            Tensor::new(
                sh.clone(),
                Arc::new(UmaMockBuffer {
                    data: vec![0u8; bytes],
                }) as Arc<dyn Buffer>,
                be.clone(),
            )
        };
        let mut c = KVCache::new(mk(), mk(), max_seq);
        c.set_current_pos(4);
        c
    }

    /// P0-4: a UMA-shaped buffer (`is_gpu_buffer()==true`, `as_ptr()` non-null) reports
    /// `kv_on_device()==true` on both v3 surfaces — the `SnapshotStageCtx` read view and the
    /// `EngineCacheHandle` transaction — so a value-aware / merge stage degrades to keep-only on
    /// Adreno UMA instead of running an unsafe host merge over a GPU buffer. Mutation-proof: reverting
    /// the predicate to `as_ptr().is_null()` makes both asserts fail (non-null ptr → false).
    #[test]
    fn uma_buffer_is_on_device_across_v3_surfaces() {
        let mut cache = uma_cache();
        // v3 read ctx (the mutation driver's read view).
        let snap = SnapshotStageCtx::from_cache(&cache, 0, 0, 1);
        assert!(
            snap.kv_on_device(),
            "v3 SnapshotStageCtx: UMA must read as on-device"
        );
        // v3 transactional handle.
        let h = EngineCacheHandle::new(&mut cache, 0, 1);
        assert!(
            CacheHandle::kv_on_device(&h),
            "v3 EngineCacheHandle: UMA must read as on-device"
        );
    }
}
