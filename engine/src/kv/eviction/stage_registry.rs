//! 빌트인 eviction 정책을 argus-extension-api `KVCacheStage` 표면으로 노출하는 어댑터 + linkme 등록.
//!
//! stage 축 레지스트리([`KV_CACHE_STAGES`])에 빌트인 LayerWide 정책 3종
//! (sliding/streaming/h2o)을 등록한다. 각 정책은 기존 [`EvictionPolicy::plan_keep`]
//! (`compact_parity` 가 in-place `evict*` 와 bit-identical 임을 증명)을 [`KVCacheStage::plan`]
//! 으로 위임하는 [`EvictionPolicyAsStage`] 어댑터로 감싼다.
//!
//! 본 단계(②a)는 **등록만** — 프로덕션 소비(match arm 교체 + plan executor)는 ②b. 그래서 등록은
//! 되어 있으나 아직 `find_stage` 로 구동되지 않는다(unwired). 등록 누락(linkme fat-LTO `--gc-sections`
//! silent drop)은 ②b 의 startup self-test 가 fail-fast 로 잡는다.
//!
//! **제외**: h2o_plus(per-head, `plan_keep`→`None`)는 head_score source(F5) 미완으로 단계 ⑤ deferred,
//! d2o(`EvictionPolicy` 아님, 가중 merge)는 M4, no_eviction("none")은 happy-path 라 match 밖.

use anyhow::{Context, Result};
use argus_extension_api::{
    KV_CACHE_STAGES, KV_PLAN_NOOP, KV_PLAN_OK, KV_STAGE_ABI_VERSION, KVCachePlan, KVCacheStage,
    KVCacheStageReg, KeepSpec, PlanAbi, PluginVTableAbi, StageCtx, StageCtxAbi, StageExportAbi,
    StageParams, TensorDtype, TensorHandle, TensorKind, TensorShape, WeightedMerge,
};
use core::ffi::c_void;
use linkme::distributed_slice;
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use super::{EvictionPolicy, SlidingWindowPolicy};
use crate::buffer::DType;
use crate::kv::dequant::{dequantize_k, dequantize_v};
use crate::kv::kv_cache::KVCache;

// CAOTE production 활성화. feature `caote` ON 시 caote crate 를 force-link 한다 —
// dep 선언만으로는 미참조 rlib 이 링크 제외돼 `#[distributed_slice]` 등록이 누락되기 때문이다.
// 이 1줄이 production 바이너리에서 `find_stage("caote")` 를 가시화한다(session score_based
// 경유 value-aware 동작). feature OFF = 미링크 + `eviction caote` 서브커맨드 부재(clap reject).
#[cfg(feature = "caote")]
use caote as _;

// StreamingLLM production force-link. Extracted from the engine core into the `streaming-llm`
// technique crate; the dep declaration alone leaves the unreferenced rlib out of the link, so
// this one line makes `find_stage("streaming")` visible (the `#[distributed_slice]` registration).
use streaming_llm as _;

// H2O production force-link. Extracted from the engine core into the `h2o` technique crate;
// makes `find_stage("h2o")` visible (same force-link rationale as streaming above).
use ::h2o as _;

// D2O production force-link. Extracted from the engine core into the `d2o` technique crate
// (registers "d2o", a WeightedMerge-producing stage); makes `find_stage("d2o")` visible. Production
// resolves it via `make_stage_with_args("d2o", &params, &blob)` (eval_setup/build_bench_loop/chat),
// with the d2o-private knobs in the StageArgs blob; the registration must survive fat-LTO.
use d2o as _;

/// 기존 [`EvictionPolicy`](in-place `evict*` + `plan_keep`)를 plan-returning [`KVCacheStage`] 로 노출.
///
/// [`KVCacheStage::plan`] 은 [`EvictionPolicy::plan_keep`](layer-wide keep + 균등 merge)을
/// [`KVCachePlan`](`KeepSpec::LayerWide` + [`WeightedMerge`])으로 매핑한다. `plan_keep` 이 `None`
/// (per-head 등 단일 layer-wide keep 으로 표현 불가)이면 `None` 을 전파한다. 버퍼 변형은 엔진
/// executor 가 실행한다.
pub struct EvictionPolicyAsStage {
    inner: Box<dyn EvictionPolicy>,
}

impl EvictionPolicyAsStage {
    /// 주어진 정책을 stage 표면으로 감싼다.
    pub fn new(inner: Box<dyn EvictionPolicy>) -> Self {
        Self { inner }
    }
}

impl KVCacheStage for EvictionPolicyAsStage {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let (keep, merges) =
            self.inner
                .plan_keep(ctx.current_pos(), ctx.target_len(), ctx.importance())?;
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
            merges: merges.into_iter().map(uniform_to_weighted).collect(),
        })
    }
}

/// 균등 `format::Merge` → [`WeightedMerge`] 매핑. 현 빌트인 3정책은 모두 빈 merge 를 내므로 실질적으로
/// 빈 Vec→빈 Vec 이나, 균등 가중치 의미(`into` 포함 N개 동일 가중, Σ=1)를 보존한다. d2o 의 Eq.11 가중
/// merge 는 이 경로가 아니라 M4 에서 직접 산출한다.
/// `pub(crate)`: `compact_parity` 가 Path B retarget에서 재사용한다.
pub(crate) fn uniform_to_weighted(m: crate::format::Merge) -> WeightedMerge {
    let n = (1 + m.from.len()) as f32; // into + from 토큰 수
    let w = 1.0 / n;
    WeightedMerge {
        into: m.into,
        into_weight: w,
        from: m.from.into_iter().map(|p| (p, w)).collect(),
        apply_to: argus_extension_api::MergeAxis::Both,
    }
}

// ── ②b: KVCachePlan executor + StageBackedPolicy 역어댑터 (World B) ──────────────

/// [`KVCacheStage`] 가 산출한 [`KVCachePlan`] 을 `&mut KVCache` 에 적용한다(변형은
/// 엔진 독점). `StandardFormat::compact` 의 빈-merge 경로와 동일: `compact_keep_positions(keep, 0)` +
/// `set_current_pos(keep.len())`. compact_parity 가 이 경로 ≡ in-place `evict*` 를 4정책×3dtype 에서
/// 증명하므로, plan keep 이 `plan_keep` keep 과 같으면(②a 어댑터 faithful) 버퍼 bit-identical 무회귀.
///
/// pub(crate): the `d2o` plugin's WeightedMerge plan flows through here (apply_weighted_merges +
/// compact) — the production merge-application path.
pub(crate) fn execute_kv_plan(cache: &mut KVCache, plan: &KVCachePlan) -> Result<()> {
    match &plan.keep {
        KeepSpec::LayerWide(keep) => {
            if !plan.merges.is_empty() {
                // (M4-b) 가중 merge 를 compact 이전 좌표계에서 in-place 적용(scatter_reduce 와
                // bit-identical, F32/F16/Q4_0). (M4 정정) — Q4_0 merge 활성.
                crate::kv::standard_format::apply_weighted_merges(cache, &plan.merges);
            }
            cache.compact_keep_positions(keep, 0)?;
            cache.set_current_pos(keep.len());
            Ok(())
        }
        KeepSpec::PerHead(_) => {
            // per-head executor = 단계 ⑤(h2o_plus, head_score source 미완) deferred.
            // 현 빌트인 3정책은 PerHead 미생산이라 도달 불가.
            anyhow::bail!("per-head executor not implemented (stage ⑤ deferred)")
        }
    }
}

/// 엔진 `DType` → argus-extension-api `TensorDtype` 매핑(핸들 진단용; 읽기 산출은 항상 f32).
fn map_dtype(dt: DType) -> TensorDtype {
    match dt {
        DType::F16 => TensorDtype::F16,
        DType::Q4_0 => TensorDtype::Q4_0,
        _ => TensorDtype::F32,
    }
}

/// `tensor(Key)` 핸들 — raw K 를 `dequantize_k` 정본으로 읽는다(D2OHandler 와 bit-identical).
struct KeyHandle<'a> {
    cache: &'a KVCache,
}
impl TensorHandle for KeyHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.cache.current_pos(),
            cols: self.cache.head_dim(),
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        map_dtype(self.cache.k_buffer.dtype())
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        dequantize_k(self.cache, row, kv_head, self.cache.head_dim(), out);
    }
}

/// `tensor(Value)` 핸들 — raw V 를 `dequantize_v` 정본으로 읽는다(CAOTE 의 v_i).
struct ValueHandle<'a> {
    cache: &'a KVCache,
}
impl TensorHandle for ValueHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.cache.current_pos(),
            cols: self.cache.head_dim(),
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        map_dtype(self.cache.v_buffer.dtype())
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        dequantize_v(self.cache, row, kv_head, self.cache.head_dim(), out);
    }
}

/// `tensor(Scores)`/`tensor(AttnWeights)` 핸들 — per-(kv_head,pos) f32 스칼라.
/// 원천 레이아웃 `[n_kv_heads * max_seq]` row-major(accumulator stride=max_seq).
struct ScalarHandle<'a> {
    data: &'a [f32],
    rows: usize,
    max_seq: usize,
}
impl TensorHandle for ScalarHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.rows,
            cols: 1,
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        out[0] = self
            .data
            .get(kv_head * self.max_seq + row)
            .copied()
            .unwrap_or(0.0);
    }
}

/// `tensor(QueryStats)` 핸들 — per-(kv_head) Q running mean/var.
/// 공급원 = `QueryStatsAccumulator::layer_stats(layer)` 의 단일-layer 슬라이스(MQ-4 (c)).
/// 레이아웃 `[n_kv_heads * 2 * head_dim]`: `data[kv_head*2*head_dim + stat_row*head_dim + d]`,
/// `stat_row 0 = mean / 1 = var`. `shape = {rows:2, cols:head_dim, per_head:true}`,
/// `read_row(row, kv_head, out)` = `data[base .. base+head_dim]` copy.
struct QueryStatsHandle<'a> {
    data: &'a [f32],
    head_dim: usize,
}
impl TensorHandle for QueryStatsHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: 2, // row0 = mean, row1 = var.
            cols: self.head_dim,
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        // base = kv_head * 2 * head_dim + row * head_dim (row 0=mean / 1=var).
        let base = kv_head * 2 * self.head_dim + row * self.head_dim;
        let hd = self.head_dim.min(out.len());
        if base + hd <= self.data.len() {
            out[..hd].copy_from_slice(&self.data[base..base + hd]);
        } else {
            out[..hd].fill(0.0);
        }
    }
}

/// `&KVCache`(+ budget + scores) 위로 구현한 [`StageCtx`].
///
/// 모든 텐서/스코어 읽기는 [`StageCtx::tensor`] 단일 경로로 흐른다: Key/Value 핸들은 항상,
/// Scores/AttnWeights 는 `new()` 에 슬라이스가 공급될 때만 `Some`. flat `importance()` 만 zero-copy 직접
/// 노출(D1 예외). builtin LayerWide(sliding/streaming/h2o) + d2o(tensor(Key))는 production 에서 구동,
/// Scores/AttnWeights 공급은 현재 host 테스트(CAOTE) 경로 — production eviction-hook threading 은 CLI
/// 배선(D-3 deferred)과 함께 후속.
pub(crate) struct KVStageCtx<'a> {
    cache: &'a KVCache,
    target_len: usize,
    importance: Option<&'a [f32]>,
    /// Which layer this single-cache view represents + the total layer count, for per-layer
    /// techniques (d2o `protected_layers` / last-layer protection). Default `(0, 1)`; the engine
    /// eviction loop sets the real values via [`with_layer`](Self::with_layer).
    layer_idx: usize,
    n_layers: usize,
    /// Whether the KV buffers are device-only (no CPU pointer) → no raw read / no merge.
    on_device: bool,
    key_handle: KeyHandle<'a>,
    value_handle: ValueHandle<'a>,
    scores_handle: Option<ScalarHandle<'a>>,
    attn_handle: Option<ScalarHandle<'a>>,
    query_stats_handle: Option<QueryStatsHandle<'a>>,
}

impl<'a> KVStageCtx<'a> {
    /// 엔진 eviction 경로(+ d2o 동등성/CAOTE host 테스트)가 `&KVCache` 위로 ctx 를 만든다.
    /// `head_scores`/`last_attn`: per-(kv_head,pos) `[n_kv_heads*max_seq]`. `None`=미공급(`tensor()`→None).
    /// `query_stats`: 단일-layer Q running mean/var `[n_kv_heads*2*head_dim]`.
    /// `None`=미공급(`tensor(QueryStats)`→None) — production builtins 는 None(score-active e2e seam 한정).
    pub(crate) fn new(
        cache: &'a KVCache,
        target_len: usize,
        importance: Option<&'a [f32]>,
        head_scores: Option<&'a [f32]>,
        last_attn: Option<&'a [f32]>,
        query_stats: Option<&'a [f32]>,
    ) -> Self {
        let rows = cache.current_pos();
        let max_seq = cache.max_seq_len;
        let head_dim = cache.head_dim();
        Self {
            cache,
            target_len,
            importance,
            layer_idx: 0,
            n_layers: 1,
            // Device-only KV (discrete GPU) returns a null host pointer → CPU read/merge unsafe.
            on_device: cache.k_buffer.as_ptr().is_null(),
            key_handle: KeyHandle { cache },
            value_handle: ValueHandle { cache },
            scores_handle: head_scores.map(|data| ScalarHandle {
                data,
                rows,
                max_seq,
            }),
            attn_handle: last_attn.map(|data| ScalarHandle {
                data,
                rows,
                max_seq,
            }),
            query_stats_handle: query_stats.map(|data| QueryStatsHandle { data, head_dim }),
        }
    }

    /// Set the real layer index + total layer count (the engine eviction loop injects these while
    /// iterating caches). Enables per-layer techniques (d2o protected_layers / last-layer protect).
    pub(crate) fn with_layer(mut self, layer_idx: usize, n_layers: usize) -> Self {
        self.layer_idx = layer_idx;
        self.n_layers = n_layers;
        self
    }
}

impl StageCtx for KVStageCtx<'_> {
    fn current_pos(&self) -> usize {
        self.cache.current_pos
    }
    fn target_len(&self) -> usize {
        self.target_len
    }
    fn layer_idx(&self) -> usize {
        self.layer_idx
    }
    fn n_layers(&self) -> usize {
        self.n_layers
    }
    fn kv_on_device(&self) -> bool {
        self.on_device
    }
    fn importance(&self) -> Option<&[f32]> {
        self.importance
    }
    fn n_kv_heads(&self) -> usize {
        self.cache.kv_heads()
    }
    fn head_dim(&self) -> usize {
        self.cache.head_dim()
    }
    /// 단일 텐서 접근 — Key/Value 항상, Scores/AttnWeights 는 공급 시. dequant_k/v·head_score·
    /// attn_weight 등 sugar 는 argus-extension-api default 가 이 위에 얹힌다.
    fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
        match kind {
            TensorKind::Key => Some(&self.key_handle),
            TensorKind::Value => Some(&self.value_handle),
            TensorKind::Scores => self.scores_handle.as_ref().map(|h| h as &dyn TensorHandle),
            TensorKind::AttnWeights => self.attn_handle.as_ref().map(|h| h as &dyn TensorHandle),
            TensorKind::QueryStats => self
                .query_stats_handle
                .as_ref()
                .map(|h| h as &dyn TensorHandle),
        }
    }
}

/// [`KVCacheStage`](plan-returning)를 레거시 [`EvictionPolicy`](in-place)로 노출하는 역어댑터.
///
/// 프로덕션 eviction 경로(`run_policy_eviction` → `evict*`)는 구조 불변으로 두되, 내부에서 stage 의
/// plan 을 [`execute_kv_plan`] 으로 실행한다 — 즉 sliding/streaming/h2o 의 evict 가 in-place(World A)
/// 에서 plan→compact(World B)로 바뀐다. compact_parity 가 등가성을 보장(무회귀).
pub struct StageBackedPolicy {
    stage: Box<dyn KVCacheStage>,
}

impl StageBackedPolicy {
    /// 주어진 stage 를 `EvictionPolicy` 표면으로 감싼다.
    pub fn new(stage: Box<dyn KVCacheStage>) -> Self {
        Self { stage }
    }

    /// 읽기 ctx 로 plan 산출(immutable borrow) → borrow 종료 후 executor 가 `&mut` 로 실행.
    /// `layer_idx`/`n_layers` 는 per-layer 기법(d2o protected_layers/last-layer protect)용 — 비-layer
    /// 인지 호출자(직접 evict)는 `(0, 1)` 단일-layer 뷰를 쓴다.
    fn run(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: Option<&[f32]>,
        layer_idx: usize,
        n_layers: usize,
    ) -> Result<()> {
        let plan = {
            // QueryStats(MQ-4 e2e seam)는 production eviction 경로에서 미공급(None) — score-active
            // 측정 하네스가 별도로 공급한다(dump_importance.rs).
            let ctx = KVStageCtx::new(cache, target_len, importance, None, None, None)
                .with_layer(layer_idx, n_layers);
            self.stage.plan(&ctx)
        };
        if let Some(plan) = plan {
            execute_kv_plan(cache, &plan)?;
        }
        Ok(())
    }
}

impl EvictionPolicy for StageBackedPolicy {
    fn should_evict(&self, _cache: &KVCache, _mem_available: usize) -> bool {
        // WHEN(트리거)은 엔진 소유 — `run_policy_eviction` 의 target_len/MIN_EVICT
        // 가드가 결정한다. 프로덕션 미호출(should_evict 의미는 구체 정책 테스트에서 검증). 엔진 위임.
        true
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        self.run(cache, target_len, None, 0, 1)
    }

    fn evict_with_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: &[f32],
    ) -> Result<()> {
        self.run(cache, target_len, Some(importance), 0, 1)
    }

    /// Per-layer eviction: thread the real `(layer_idx, n_layers)` into the stage ctx so per-layer
    /// techniques (d2o `protected_layers` / last-layer protection) see which layer they handle.
    fn evict_layer(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: Option<&[f32]>,
        layer_idx: usize,
        n_layers: usize,
    ) -> Result<()> {
        self.run(cache, target_len, importance, layer_idx, n_layers)
    }

    fn name(&self) -> &str {
        self.stage.name()
    }
}

/// Test helper: build the out-of-tree `h2o` stage wrapped as a legacy [`EvictionPolicy`]
/// (`StageBackedPolicy`). Used by CacheManager / EvictionHandler tests after H2O was extracted
/// to the `h2o` plugin crate — production resolves "h2o" the same way (make_stage → plugin).
#[cfg(test)]
pub(crate) fn h2o_backed_policy(
    keep_ratio: f32,
    protected_prefix: usize,
) -> Box<dyn EvictionPolicy> {
    let p = StageParams {
        keep_ratio,
        protected_prefix,
        ..Default::default()
    };
    Box::new(StageBackedPolicy::new(
        make_stage("h2o", &p).expect("h2o stage registered (force-linked h2o plugin)"),
    ))
}

/// 빌트인 LayerWide 기법(sliding/streaming/h2o)이 `KV_CACHE_STAGES` 에 등록됐는지 단언한다 — eviction
/// CacheManager build 진입 시 1회 호출. fat-LTO `--gc-sections` 가 linkme 등록을 silent
/// drop 하면 누락 기법에 대해 `Err` 로 fail-fast 한다(release 에서 정책 이름 미해석 → 조용한 폴백 방지).
pub fn ensure_builtin_stages_registered() -> Result<()> {
    for name in ["sliding", "streaming", "h2o", "d2o"] {
        if argus_extension_api::find_stage(name).is_none() {
            anyhow::bail!(
                "built-in KVCacheStage '{name}' not registered — suspect linkme fat-LTO --gc-sections silent drop\
                 stage_registry's #[distributed_slice] registration was not linked."
            );
        }
    }
    Ok(())
}

#[distributed_slice(KV_CACHE_STAGES)]
static SLIDING_STAGE: KVCacheStageReg = KVCacheStageReg {
    name: "sliding",
    make: |p: StageParams| {
        Box::new(EvictionPolicyAsStage::new(Box::new(
            SlidingWindowPolicy::new(p.eviction_window, p.protected_prefix),
        )))
    },
    // sliding takes no technique-private args — drop the blob, build from StageParams.
    make_with_args: |p: StageParams, _args| {
        Box::new(EvictionPolicyAsStage::new(Box::new(
            SlidingWindowPolicy::new(p.eviction_window, p.protected_prefix),
        )))
    },
};

// `streaming` is registered by the out-of-tree `streaming-llm` technique crate (force-linked
// above via `use streaming_llm as _;`), not here — extracted from the engine core.

// `h2o` is registered by the out-of-tree `h2o` technique crate (force-linked above via
// `use ::h2o as _;`), not here — extracted from the engine core.

// `d2o` is registered by the out-of-tree `d2o` technique crate (force-linked above via
// `use d2o as _;`), not here — extracted from the engine core. It emits WeightedMerge (Eq.11);
// the engine executor `apply_weighted_merges` applies them. Production resolves it via
// `make_stage_with_args("d2o", &params, &blob)`: StageParams carries keep_ratio/protected_prefix,
// and the d2o-private knobs ride the StageArgs blob (parsed by `d2o::D2OConfig::from_args`).

/// R-KV(KV roadmap 항목 0 측정, P2a) — `RkvStage`(cosine redundancy + importance joint eviction).
/// `StageParams` 에 R-KV 전용 필드(λ)가 없어 기본 `RkvConfig`(λ=0.1, α=8, τ=0.5)로 등록한다 — CLI
/// λ override 는 측정 schedule 의 stage 직접 생성 경로(d2o 의 if-branch 와 동형). feature `rkv` OFF =
/// 이 등록 미컴파일 → `find_stage("rkv")` None → production 정책 카탈로그 불변(arch §6 Spec Triage).
#[cfg(feature = "rkv")]
#[distributed_slice(KV_CACHE_STAGES)]
static RKV_STAGE: KVCacheStageReg = KVCacheStageReg {
    name: "rkv",
    make: |_p: StageParams| Box::new(crate::kv::rkv_stage::RkvStage::new(Default::default())),
    // rkv's λ override rides the eval stage-direct path, not StageParams — no private blob here.
    make_with_args: |_p: StageParams, _args| {
        Box::new(crate::kv::rkv_stage::RkvStage::new(Default::default()))
    },
};

// ════════════════════════════════════════════════════════════════════════════
// GATE-C — 런타임 `.so` dlopen 레지스트리
// ════════════════════════════════════════════════════════════════════════════
//
// 정적 `KV_CACHE_STAGES`(linkme)는 그대로 두고(D3 가산), dlopen 된 plugin 을 별도
// `DYN_REGISTRY` 에 모은다. `make_stage(name, params)` 가 정적 우선 → 동적 fallback 으로
// source-agnostic `Box<dyn KVCacheStage>` 를 돌려준다. `.so` 는 init-once 로 frozen 이고
// `Arc<Library>` 를 영구 보관(leak-and-keep)해 vtable/handle 이 살아 있게 한다.

/// dlopen 된 한 stage plugin 의 등록 항목. vtable 은 plugin `.so` 의 immutable static 을 가리킨다.
struct RuntimeStageReg {
    name: String,
    vtable: *const PluginVTableAbi,
    /// `.so` 를 프로세스 수명 동안 유지(vtable/handle dangling 방지). drop 안 함.
    _lib: Arc<libloading::Library>,
}

// SAFETY: vtable 은 `.so` 의 immutable static 을 가리키고 `_lib`(Arc) 가 `.so` 를 살려 둔다.
// 읽기 전용 공유이므로 스레드 간 안전 — `DYN_REGISTRY`(static) 에 담기 위해 필요.
unsafe impl Send for RuntimeStageReg {}
unsafe impl Sync for RuntimeStageReg {}

/// 동적 등록 레지스트리 — init 시 append, construction 시 read. 정적 슬라이스와 **병합하지 않는다**(D3).
static DYN_REGISTRY: OnceLock<RwLock<Vec<RuntimeStageReg>>> = OnceLock::new();

/// 이미 dlopen 된 `.so`(Arc) 에서 stage capability 를 [`DYN_REGISTRY`] 에 등록하는 per-`.so` 코어.
///
/// `register_kv_stages_v2` 봉투 entry 를 dlsym 한다 — **없으면 `Ok(0)`**(이 `.so` 는 stage 미보유, format 전용
/// 일 수 있음). 있으면 봉투 `abi_version` 검사 → `count` 개 vtable 순회. **2-pass 원자성**: ① 전 이름을
/// 빌트인 충돌·봉투 내부 중복 검사(통과 전 push 0) → ② write-lock 1회로 동적 중복 검사 + 일괄 push(부분
/// 등록 롤백 회피). 반환 = 등록한 stage 개수. cross-axis dispatcher([`register_dynamic_plugins`](crate::session::plugin_dispatch::register_dynamic_plugins))
/// 가 `.so` 1회 dlopen 후 호출(Arc 공유), batch 래퍼([`register_dynamic_stages`])도 사용.
pub(crate) fn try_register_stage(lib: &Arc<libloading::Library>, path: &Path) -> Result<usize> {
    // SAFETY: register_kv_stages_v2 dlsym. 부재 = 이 .so 가 stage 축 미보유 → Ok(0)(에러 아님).
    let reg_fn: libloading::Symbol<unsafe extern "C" fn() -> StageExportAbi> =
        match unsafe { lib.get(b"register_kv_stages_v2\0") } {
            Ok(f) => f,
            Err(_) => return Ok(0),
        };
    // SAFETY: 봉투 by-value 반환(sret). vtables 는 `.so` static 배열 base, abi_version 은 .so 단위 게이트.
    let export = unsafe { reg_fn() };
    if export.abi_version != KV_STAGE_ABI_VERSION {
        anyhow::bail!(
            "plugin {}: stage abi_version {} != expected {} (rebuild required)",
            path.display(),
            export.abi_version,
            KV_STAGE_ABI_VERSION
        );
    }
    if export.count == 0 {
        return Ok(0);
    }
    if export.vtables.is_null() {
        anyhow::bail!(
            "plugin {}: register_kv_stages_v2 has count {} but null vtables",
            path.display(),
            export.count
        );
    }
    let registry = DYN_REGISTRY.get_or_init(|| RwLock::new(Vec::new()));
    // ── pass 1: 이름 추출 + 빌트인 충돌 / 봉투 내부 중복 검사 (lock 불요). ──
    let mut pending: Vec<(String, *const PluginVTableAbi)> = Vec::with_capacity(export.count);
    for i in 0..export.count {
        // SAFETY: vtables 는 `.so` static 배열 base, i < count. 봉투 스택과 무관(원소는 .so 수명).
        let vtable_ptr = unsafe { export.vtables.add(i) };
        let vtable = unsafe { &*vtable_ptr };
        let name = unsafe { CStr::from_ptr(vtable.name) }
            .to_str()
            .with_context(|| {
                format!(
                    "plugin {}: stage name[{i}] is not valid UTF-8",
                    path.display()
                )
            })?
            .to_owned();
        // 빌트인 우선 — silent override 차단(Known Bug #1/#2 류 재발 방지).
        if argus_extension_api::find_stage(&name).is_some() {
            anyhow::bail!(
                "plugin {}: stage name '{}' conflicts with a built-in (built-in takes precedence, dynamic registration rejected)",
                path.display(),
                name
            );
        }
        if pending.iter().any(|(n, _)| *n == name) {
            anyhow::bail!(
                "plugin {}: stage name '{}' is duplicated within the envelope",
                path.display(),
                name
            );
        }
        pending.push((name, vtable_ptr));
    }
    // ── pass 2: 동적 registry 중복 검사 + 일괄 push (write-lock 1회 = per-.so 원자). ──
    let mut w = registry.write().expect("DYN_REGISTRY RwLock poisoned");
    for (name, _) in &pending {
        if w.iter().any(|r| r.name == *name) {
            anyhow::bail!(
                "plugin {}: stage name '{}' is already dynamically registered (duplicate)",
                path.display(),
                name
            );
        }
    }
    let n = pending.len();
    for (name, vtable_ptr) in pending {
        w.push(RuntimeStageReg {
            name,
            vtable: vtable_ptr,
            _lib: Arc::clone(lib),
        });
    }
    Ok(n)
}

/// `--load-plugin` 의 `.so` 들을 dlopen 해 stage 만 등록하는 **strict batch 래퍼**(gate 테스트·축-격리 진단용).
/// 각 `.so` 가 stage 0개면 "심볼 부재" bail(기존 계약 유지). production 혼합 로드는
/// [`register_dynamic_plugins`](crate::session::plugin_dispatch::register_dynamic_plugins) 사용.
pub fn register_dynamic_stages(paths: &[PathBuf]) -> Result<()> {
    for path in paths {
        // SAFETY: dlopen — 신뢰된 plugin 경로(사용자 명시 --load-plugin). RTLD_NOW 즉시 바인딩.
        let lib = Arc::new(
            unsafe { libloading::Library::new(path) }
                .with_context(|| format!("plugin dlopen failed: {}", path.display()))?,
        );
        if try_register_stage(&lib, path)? == 0 {
            anyhow::bail!(
                "plugin {}: register_kv_stages_v2 symbol missing (or 0 stages)",
                path.display()
            );
        }
    }
    Ok(())
}

/// 동적으로 등록된 stage 이름들(self-test / 진단용 — 정적 `registered_names()` 의 동적 짝).
pub fn dynamic_registered_stage_names() -> Vec<String> {
    DYN_REGISTRY
        .get()
        .map(|r| {
            r.read()
                .expect("DYN_REGISTRY RwLock poisoned")
                .iter()
                .map(|reg| reg.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// 이름으로 stage 인스턴스를 만든다 — **정적 우선 → 동적 fallback**(D3). 호출부는 source 를 모른다.
/// technique-private 인자(d2o 의 ema_beta/merge_axis 등)는 `args` blob([`StageArgs`])으로 전달되고
/// plugin 이 직접 파싱한다(`make_with_args`) — 엔진은 plugin 의 private 파라미터를 모른다. built-in/
/// args-무시 기법은 빈 blob 으로 동작한다. 정적/동적 모두 miss 면 `None`(graceful unknown).
///
/// **동적(dlopen) plugin 은 현재 private 인자를 받지 않는다**(소비자 0 — d2o 는 force-link 정적 등록;
/// vtable C-ABI(`PluginVTableAbi.make`)는 `StageParams` 만 운반). 필요해지면 그때 `.so` ABI 를 확장한다.
pub fn make_stage_with_args(
    name: &str,
    params: &StageParams,
    args: argus_extension_api::StageArgs<'_>,
) -> Option<Box<dyn KVCacheStage>> {
    // 1) 정적(linkme) 우선 — make_with_args 로 technique-private blob 전달.
    if let Some(reg) = argus_extension_api::find_stage(name) {
        return Some((reg.make_with_args)(*params, args));
    }
    // 2) 동적(dlopen) fallback — vtable.make 는 StageParams 만 받으므로 blob 은 무시(소비자 0).
    let registry = DYN_REGISTRY.get()?;
    let (vtable, lib) = {
        let guard = registry.read().expect("DYN_REGISTRY RwLock poisoned");
        let reg = guard.iter().find(|r| r.name == name)?;
        (reg.vtable, Arc::clone(&reg._lib))
    };
    // SAFETY: vtable 는 `.so` static (lib 가 살려 둠). make 가 opaque plugin 핸들 반환.
    let handle = unsafe { ((*vtable).make)(params as *const StageParams) };
    if handle.is_null() {
        eprintln!("[make_stage] plugin '{name}' make returned a null handle");
        return None;
    }
    Some(Box::new(DynStage {
        handle,
        vtable,
        _lib: lib,
    }))
}

/// [`make_stage_with_args`] 의 빈-blob shim — technique-private 인자 없이 stage 를 만든다.
pub fn make_stage(name: &str, params: &StageParams) -> Option<Box<dyn KVCacheStage>> {
    make_stage_with_args(name, params, &[])
}

/// 동적 plugin stage 의 host 측 어댑터 — vtable 마샬링으로 [`KVCacheStage`] 를 구현(D2).
struct DynStage {
    handle: *mut c_void,
    vtable: *const PluginVTableAbi,
    _lib: Arc<libloading::Library>,
}

// SAFETY: 핸들은 plugin 의 `KVCacheStage`(trait 계약상 Send+Sync) 인스턴스, vtable 불변, lib Arc 유지.
unsafe impl Send for DynStage {}
unsafe impl Sync for DynStage {}

impl Drop for DynStage {
    fn drop(&mut self) {
        // SAFETY: handle 은 make 가 만든 plugin 인스턴스, 정확히 1회 해제.
        unsafe { ((*self.vtable).drop)(self.handle) };
    }
}

impl KVCacheStage for DynStage {
    fn name(&self) -> &str {
        // SAFETY: vtable.name 은 plugin `.so` 의 'static null-종단 str (lib 가 살려 둠).
        unsafe { CStr::from_ptr((*self.vtable).name) }
            .to_str()
            .unwrap_or("<plugin>")
    }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        // host concrete ctx(fat ref)를 thin ptr 로 평탄화 — shim 들이 deref 해 메서드 호출.
        let ctx_ref: &dyn StageCtx = ctx;
        let abi = StageCtxAbi {
            ctx: (&ctx_ref) as *const &dyn StageCtx as *const c_void,
            current_pos: shim_current_pos,
            target_len: shim_target_len,
            layer_idx: shim_layer_idx,
            n_kv_heads: shim_n_kv_heads,
            head_dim: shim_head_dim,
            importance: shim_importance,
            tensor_read_row: shim_tensor_read_row,
            tensor_shape: shim_tensor_shape,
        };
        let mut plan_abi = PlanAbi::zeroed();
        // SAFETY: handle/vtable 유효. abi 는 plan 호출 동안만 산다(ctx_ref 가 그 scope 에 유효).
        let code = unsafe { ((*self.vtable).plan)(self.handle, &abi, &mut plan_abi) };
        match code {
            KV_PLAN_NOOP => None,
            KV_PLAN_OK => {
                // SAFETY: plan 이 KV_PLAN_OK 면 plan_abi 가 plugin-arena 를 가리킨다.
                let result = unsafe { planabi_to_plan(&plan_abi) };
                // 복사 직후 plugin arena 회수 (각자 자기 것 free).
                unsafe { ((*self.vtable).plan_free)(plan_abi.owner) };
                match result {
                    Ok(p) => Some(p),
                    Err(e) => {
                        eprintln!("[DynStage:{}] plan marshalling rejected: {e}", self.name());
                        None
                    }
                }
            }
            other => {
                eprintln!(
                    "[DynStage:{}] plugin plan error code {other} — treated as no-op",
                    self.name()
                );
                None
            }
        }
    }
}

/// [`PlanAbi`](plugin-arena flat)를 host `KVCachePlan` 으로 복사 재구성(D5). v1 은 LayerWide 만 —
/// PerHead(`keep_kind==1`)는 promotion-trigger 전까지 명시적 bail(silent garbage 방지).
///
/// # Safety
/// `abi` 는 plugin 의 `plan` 이 `KV_PLAN_OK` 와 함께 채운 유효 PlanAbi 여야 한다.
unsafe fn planabi_to_plan(abi: &PlanAbi) -> Result<KVCachePlan> {
    if abi.keep_kind == 1 {
        anyhow::bail!(
            "GATE-C v1: plugin produced PerHead keep — unsupported (before promotion-trigger)"
        );
    }
    let keep: Vec<usize> = if abi.keep_len == 0 || abi.keep_ptr.is_null() {
        Vec::new()
    } else {
        // SAFETY: keep_ptr/len 은 plugin-arena 의 유효 슬라이스(plan_free 전).
        unsafe { core::slice::from_raw_parts(abi.keep_ptr, abi.keep_len) }.to_vec()
    };
    let mut merges = Vec::with_capacity(abi.merges_len);
    if abi.merges_len > 0 && !abi.merges_ptr.is_null() {
        // SAFETY: merges_ptr/len 유효.
        let m_slice = unsafe { core::slice::from_raw_parts(abi.merges_ptr, abi.merges_len) };
        for m in m_slice {
            let from: Vec<(usize, f32)> = if m.from_len == 0 || m.from_ptr.is_null() {
                Vec::new()
            } else {
                // SAFETY: from_ptr/len 유효(plugin-arena).
                unsafe { core::slice::from_raw_parts(m.from_ptr, m.from_len) }
                    .iter()
                    .map(|p| (p.pos, p.weight))
                    .collect()
            };
            merges.push(WeightedMerge {
                into: m.into,
                into_weight: m.into_weight,
                from,
                apply_to: argus_extension_api::MergeAxis::from_u32(m.apply_to),
            });
        }
    }
    Ok(KVCachePlan {
        keep: KeepSpec::LayerWide(keep),
        merges,
    })
}

/// u32 discriminant → [`TensorKind`] (StageCtxAbi C-ABI 의 kind 인자 역매핑). repr(u32) 순서 고정.
fn tensor_kind_from_u32(k: u32) -> Option<TensorKind> {
    match k {
        0 => Some(TensorKind::Key),
        1 => Some(TensorKind::Value),
        2 => Some(TensorKind::AttnWeights),
        3 => Some(TensorKind::Scores),
        4 => Some(TensorKind::QueryStats),
        _ => None,
    }
}

// ── StageCtxAbi shim 들 (host concrete `&dyn StageCtx` 위 extern "C" 브리지) ──
// 모두 `c` 를 `*const &dyn StageCtx`(thin→fat) 로 deref. host 가 plan 동안 ctx 유효 보장.

unsafe extern "C" fn shim_current_pos(c: *const c_void) -> usize {
    let ctx = unsafe { *(c as *const &dyn StageCtx) };
    ctx.current_pos()
}
unsafe extern "C" fn shim_target_len(c: *const c_void) -> usize {
    let ctx = unsafe { *(c as *const &dyn StageCtx) };
    ctx.target_len()
}
unsafe extern "C" fn shim_layer_idx(c: *const c_void) -> usize {
    let ctx = unsafe { *(c as *const &dyn StageCtx) };
    ctx.layer_idx()
}
unsafe extern "C" fn shim_n_kv_heads(c: *const c_void) -> usize {
    let ctx = unsafe { *(c as *const &dyn StageCtx) };
    ctx.n_kv_heads()
}
unsafe extern "C" fn shim_head_dim(c: *const c_void) -> usize {
    let ctx = unsafe { *(c as *const &dyn StageCtx) };
    ctx.head_dim()
}
unsafe extern "C" fn shim_importance(
    c: *const c_void,
    out_ptr: *mut *const f32,
    out_len: *mut usize,
) -> bool {
    let ctx = unsafe { *(c as *const &dyn StageCtx) };
    match ctx.importance() {
        Some(s) => {
            unsafe {
                *out_ptr = s.as_ptr();
                *out_len = s.len();
            }
            true
        }
        None => false,
    }
}
unsafe extern "C" fn shim_tensor_shape(c: *const c_void, kind: u32, out: *mut TensorShape) -> bool {
    let ctx = unsafe { *(c as *const &dyn StageCtx) };
    let Some(k) = tensor_kind_from_u32(kind) else {
        return false;
    };
    match ctx.tensor(k) {
        Some(h) => {
            unsafe { *out = h.shape() };
            true
        }
        None => false,
    }
}
unsafe extern "C" fn shim_tensor_read_row(
    c: *const c_void,
    kind: u32,
    row: usize,
    kv_head: usize,
    out: *mut f32,
    out_len: usize,
) -> bool {
    let ctx = unsafe { *(c as *const &dyn StageCtx) };
    let Some(k) = tensor_kind_from_u32(kind) else {
        return false;
    };
    match ctx.tensor(k) {
        Some(h) => {
            let cols = h.shape().cols;
            // out_len 계약(== cols) 검증 — plugin 이 작은 버퍼를 줘도 OOB write 차단.
            if out_len < cols {
                return false;
            }
            // SAFETY: out 은 plugin 이 준 out_len(≥cols) 버퍼. cols 만 쓴다.
            let out_slice = unsafe { core::slice::from_raw_parts_mut(out, cols) };
            h.read_row(row, kv_head, out_slice);
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use argus_extension_api::{find_stage, registered_names};
    use std::sync::Arc;

    /// 최소 StageCtx 스텁 — LayerWide 정책이 읽는 current_pos/target_len/importance 만 의미가 있고
    /// per-head/dequant accessor 는 trivial(이 단계 미사용).
    struct TestCtx {
        current_pos: usize,
        target_len: usize,
        importance: Option<Vec<f32>>,
    }
    impl StageCtx for TestCtx {
        fn current_pos(&self) -> usize {
            self.current_pos
        }
        fn target_len(&self) -> usize {
            self.target_len
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn importance(&self) -> Option<&[f32]> {
            self.importance.as_deref()
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            1
        }
        // LayerWide 정책만 구동 → 텐서 미공급(None). head_score/dequant_* 는 default sugar(None→trivial).
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
    }

    #[test]
    fn builtins_registered() {
        // linkme 가 엔진의 등록을 슬라이스로 모으는지 (fat-LTO 생존은 ②b release self-test).
        let names = registered_names();
        for n in ["sliding", "streaming", "h2o"] {
            assert!(
                names.contains(&n),
                "'{n}' 등록 누락 (linkme distributed_slice)"
            );
        }
    }

    #[test]
    fn d2o_stage_registered() {
        // (M4-c) D2OStage 가 "d2o" 로 KV_CACHE_STAGES 에 등록됐는지 — find_stage 해석 + make 가능.
        // production 은 if-branch(D2OHandler) 가 가로채므로 이 등록은 proven-equivalent available
        // 표면(release fat-LTO 에서도 생존해야). make 로 D2OStage 인스턴스 생성 가능 확인.
        let reg = find_stage("d2o").expect("d2o stage 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "d2o");
        let params = StageParams {
            eviction_window: 0,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 0,
            streaming_window: 0,
        };
        let stage = (reg.make)(params);
        assert_eq!(stage.name(), "d2o");
    }

    // cross-crate linkme 실증 결과(M3): **dev-dep 선언만으로는 부족**하다. Rust 는 미참조
    // 의존 rlib 을 링크에서 제외하므로 `#[distributed_slice]` 등록이 누락된다(실측 — forcing 없으면
    // find_stage None). 따라서 technique crate 의 등록을 활성화하려면 의존 1줄에 더해 **force-link
    // 참조 1줄**(`use <crate> as _;`)이 designated 지점에 필요하다. 즉 확장 비용 = dep 1줄 + force-link
    // 1줄(둘 다 기계적, 기존 로직 수정 0 → OCP 유지). 상세: (M3 정정).
    use example_keep_recent as _;
    // CAOTE 의 force-link 는 production(module-level `#[cfg(feature = "caote")] use caote as _`)
    // 가 담당한다 — `--features caote` 테스트 시 그 cfg 가 활성이라 별도 test-only force-link 불필요.

    #[test]
    fn example_technique_crate_visible_to_engine() {
        // force-link(위 `use ... as _`) 가 걸린 상태에서 별도 technique crate 의 등록이 엔진 뷰의
        // KV_CACHE_STAGES 에 나타나는가 — "폴더 추가 + dep 1줄 + force-link 1줄 = 기법 추가" 검증.
        assert!(
            find_stage("example_keep_recent").is_some(),
            "force-link 후 예제 technique crate 등록이 엔진에서 보여야 한다"
        );
    }

    #[cfg(feature = "caote")]
    #[test]
    fn caote_stage_visible_and_value_aware_executes() {
        // (M-F) CAOTE crate 의 cross-crate 등록 + KVStageCtx(V 공급)로 value-aware plan 산출 →
        // execute_kv_plan 실행. mk() 가 토큰별 distinct V 를 채우므로 criticality(‖v_i−o_h‖)가 V 에
        // 의존 → 기법이 [`StageCtx::tensor`]`(Value)` 로 V 를 직접 읽어 자체 metric 을 계산함을 증명.
        let reg = find_stage("caote").expect("caote 등록이 엔진에서 보여야 한다");
        let stage = (reg.make)(StageParams {
            eviction_window: 0,
            protected_prefix: 0,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        });
        let mut c = mk(DType::F32, 8); // kv_heads=1, head_dim=PHD, V distinct per pos, current_pos=8
        let imp = vec![1.0f32; 8]; // 균일 가중 → criticality 는 V 가 결정
        let plan = {
            let ctx = KVStageCtx::new(&c, 4, Some(&imp), None, None, None);
            assert!(
                ctx.tensor(TensorKind::Value).is_some(),
                "KVStageCtx 는 Value 핸들을 항상 공급"
            );
            stage.plan(&ctx).expect("plan Some")
        };
        match &plan.keep {
            KeepSpec::LayerWide(k) => {
                assert_eq!(k.len(), 4, "target_len=4 만큼 유지");
                assert!(k.windows(2).all(|w| w[0] < w[1]), "ascending keep");
                assert!(k.iter().all(|&p| p < 8), "유효 위치");
            }
            KeepSpec::PerHead(_) => panic!("v1 CAOTE 는 LayerWide"),
        }
        assert!(plan.merges.is_empty());
        execute_kv_plan(&mut c, &plan).unwrap();
        assert_eq!(c.current_pos(), 4, "executor 가 keep.len() 로 compact");
    }

    /// R-KV 측정 프로토타입(P2a, feature `rkv`) — 등록 가시성 + sliding 하네스(KVStageCtx→plan→
    /// execute_kv_plan) 실행 + redundant fraction 덤프(last_stats). 설계서 §4.1 완료 게이트 (3)(4).
    #[cfg(feature = "rkv")]
    #[test]
    fn rkv_stage_visible_executes_and_dumps_redundancy() {
        use crate::kv::rkv_stage::RkvStage;

        // (게이트 4) find_stage("rkv") 가시 + sliding/h2o 와 동일 registry 표면.
        let reg = find_stage("rkv").expect("feature rkv ON 시 rkv 등록이 보여야 한다");
        assert_eq!(reg.name, "rkv");

        // (게이트 3) 동일 하네스 실행: KVStageCtx(Key 핸들 공급) → plan → execute_kv_plan.
        // 직접 생성한 RkvStage 로 last_stats(1단 덤프)도 검증한다(registry make 는 last_stats 미노출).
        let mut c = mk(DType::F32, 8); // kv_heads=1, head_dim=PHD, K distinct per pos
        let imp = vec![1.0f32; 8];
        let stage = RkvStage::new(Default::default());
        let plan = {
            let ctx = KVStageCtx::new(&c, 4, Some(&imp), None, None, None);
            assert!(
                ctx.tensor(TensorKind::Key).is_some(),
                "KVStageCtx 는 Key 핸들 항상 공급(redundancy 입력)"
            );
            stage.plan(&ctx).expect("rkv plan Some (target<n)")
        };
        match &plan.keep {
            KeepSpec::LayerWide(k) => {
                assert_eq!(k.len(), 4, "target_len=4 만큼 보존");
                assert!(k.windows(2).all(|w| w[0] < w[1]), "ascending keep");
                assert!(k.iter().all(|&p| p < 8), "유효 위치");
            }
            KeepSpec::PerHead(_) => panic!("R-KV 프로토타입은 LayerWide"),
        }

        // (게이트 4) 1단 측정 hook: per-kv_head redundancy stats 덤프(MPC, redundant fraction).
        let stats = stage.last_stats();
        assert_eq!(stats.len(), 1, "kv_heads=1 → stats 1개");
        assert!(
            (0.0..=1.0).contains(&stats[0].redundant_fraction),
            "redundant_fraction 은 [0,1]: {}",
            stats[0].redundant_fraction
        );
        assert!(stats[0].mpc.is_finite(), "MPC 유한: {}", stats[0].mpc);

        execute_kv_plan(&mut c, &plan).unwrap();
        assert_eq!(c.current_pos(), 4, "executor 가 keep.len() 로 compact");
    }

    #[test]
    fn adapter_plan_matches_plan_keep_sliding() {
        // 어댑터 plan() 의 LayerWide keep 이 원본 plan_keep keep 과 동일한지 (faithful, score-free).
        let params = StageParams {
            eviction_window: 8,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 4,
            streaming_window: 8,
        };
        let reg = find_stage("sliding").expect("sliding 등록");
        let stage = (reg.make)(params);
        let ctx = TestCtx {
            current_pos: 200,
            target_len: 100,
            importance: None,
        };
        let plan = stage.plan(&ctx).expect("sliding plan Some");
        let direct = SlidingWindowPolicy::new(8, 4)
            .plan_keep(200, 100, None)
            .expect("direct plan_keep Some");
        match plan.keep {
            KeepSpec::LayerWide(keep) => {
                assert_eq!(keep, direct.0, "어댑터 keep == plan_keep keep")
            }
            KeepSpec::PerHead(_) => panic!("sliding 은 LayerWide 여야 한다"),
        }
        assert!(plan.merges.is_empty(), "sliding 은 merge 없음");
    }

    // The h2o adapter-vs-plan_keep test was removed when H2O was extracted to the `h2o` plugin
    // crate: "h2o" now resolves to the plugin, whose own unit tests pin the keep-list spec
    // (score-free + score-based). The sliding adapter test above still covers the in-tree path.

    // ── StageBackedPolicy parity: World B(plan→compact) ≡ in-place evict(World A) ──
    // ②a 어댑터 faithful + compact_parity(plan_keep→compact ≡ in-place) 의 합성을, 프로덕션
    // 메커니즘 전체(find_stage→make→StageBackedPolicy→KVStageCtx→plan→execute_kv_plan)로 직접 확인.
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

    fn region(c: &KVCache) -> (Vec<u8>, Vec<u8>) {
        let nb = c.current_pos * pbytes(c.k_buffer.dtype());
        unsafe {
            (
                std::slice::from_raw_parts(c.k_buffer.as_mut_ptr() as *const u8, nb).to_vec(),
                std::slice::from_raw_parts(c.v_buffer.as_mut_ptr() as *const u8, nb).to_vec(),
            )
        }
    }

    fn sb_params() -> StageParams {
        StageParams {
            eviction_window: 10,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 4,
            streaming_window: 6,
        }
    }

    #[test]
    fn stage_backed_evict_parity_sliding() {
        for dt in [DType::F32, DType::F16, DType::Q4_0] {
            let mut a = mk(dt, 40);
            SlidingWindowPolicy::new(10, 4).evict(&mut a, 20).unwrap();
            let mut b = mk(dt, 40);
            let stage = (find_stage("sliding").unwrap().make)(sb_params());
            StageBackedPolicy::new(stage).evict(&mut b, 20).unwrap();
            assert_eq!(a.current_pos, b.current_pos, "sliding[{dt:?}] current_pos");
            assert_eq!(region(&a), region(&b), "sliding[{dt:?}] valid-region byte");
        }
    }

    // The h2o World-A↔World-B parity test was removed with the H2O extraction: the `h2o` plugin
    // is plan-only (no in-place evict_with_scores), so there is no World-A path to compare. The
    // sliding parity test above still exercises the StageBackedPolicy → execute_kv_plan mechanism,
    // and beta3_eviction_stage_equivalence.rs covers h2o end-to-end across F32/F16/Q4_0.

    #[test]
    fn kvstagectx_dequant_k_reads_f32() {
        // (M-D) dequant_k sugar(→ tensor(Key) → KeyHandle → kv::dequant::dequantize_k)로 raw K(F32) 읽기.
        // 완전 통합 후에도 기존 dequant_k 시그니처·결과가 보존됨을 확인.
        let mut c = mk(DType::F32, 8);
        let off = c.offset(5, 0);
        {
            let k = c.k_buffer.as_mut_slice::<f32>();
            for d in 0..PHD {
                k[off + d] = (d as f32) * 0.5 + 1.0;
            }
        }
        let ctx = KVStageCtx::new(&c, 0, None, None, None, None);
        let mut out = vec![0.0f32; PHD];
        ctx.dequant_k(5, 0, &mut out);
        for d in 0..PHD {
            assert_eq!(out[d], (d as f32) * 0.5 + 1.0, "dequant_k F32 d={d}");
        }
        // tensor(Key) 핸들 shape/dtype 계약.
        let kh = ctx.tensor(TensorKind::Key).expect("Key handle 항상 존재");
        assert_eq!(kh.shape().cols, PHD);
        assert!(kh.shape().per_head);
        assert_eq!(kh.dtype(), TensorDtype::F32);
    }

    #[test]
    fn kvstagectx_dequant_v_reads_f32() {
        // (M-C/M-D) dequant_v sugar(→ tensor(Value) → ValueHandle → dequantize_v)로 raw V(F32) 읽기.
        let mut c = mk(DType::F32, 8);
        let off = c.offset(5, 0);
        {
            let v = c.v_buffer.as_mut_slice::<f32>();
            for d in 0..PHD {
                v[off + d] = (d as f32) * 0.25 - 2.0;
            }
        }
        let ctx = KVStageCtx::new(&c, 0, None, None, None, None);
        let mut out = vec![0.0f32; PHD];
        ctx.dequant_v(5, 0, &mut out);
        for d in 0..PHD {
            assert_eq!(out[d], (d as f32) * 0.25 - 2.0, "dequant_v F32 d={d}");
        }
    }

    #[test]
    fn kvstagectx_scores_and_attn_handles() {
        // (M-D) Scores/AttnWeights 핸들 — 공급 시 per-(kv_head,pos) 스칼라 읽기, 미공급 시 None.
        let c = mk(DType::F32, 4); // kv_heads=1
        let max_seq = c.max_seq_len;
        let scores: Vec<f32> = (0..max_seq).map(|p| p as f32 + 0.5).collect();
        let attn: Vec<f32> = (0..max_seq).map(|p| p as f32 * 10.0).collect();
        let ctx = KVStageCtx::new(&c, 0, None, Some(&scores), Some(&attn), None);
        assert!(ctx.has_head_scores());
        assert!(ctx.has_attn_weights());
        assert_eq!(ctx.head_score(0, 3), 3.5);
        assert_eq!(ctx.attn_weight(0, 2), 20.0);
        // 미공급 ctx → None / trivial.
        let bare = KVStageCtx::new(&c, 0, None, None, None, None);
        assert!(!bare.has_head_scores());
        assert!(!bare.has_attn_weights());
        assert_eq!(bare.head_score(0, 3), 0.0);
        assert!(bare.tensor(TensorKind::Scores).is_none());
        // QueryStats 미공급 → None.
        assert!(bare.tensor(TensorKind::QueryStats).is_none());
    }

    /// TQS-7/8: `QueryStatsHandle` shape={2,head_dim,true} + read_row(0)=mean/(1)=var + 공급 시
    /// Some/미공급 None + 기존 0~3 kind 무영향.
    #[test]
    fn kvstagectx_query_stats_handle() {
        let c = mk(DType::F32, 4); // kv_heads=1, head_dim=PHD
        let head_dim = c.head_dim();
        assert_eq!(head_dim, PHD);
        // 단일-layer QueryStats 슬라이스 [n_kv_heads(1) * 2 * head_dim]:
        // row0(mean)[d] = d + 0.5, row1(var)[d] = d * 2.0.
        let mut qs = vec![0.0f32; 2 * head_dim];
        for d in 0..head_dim {
            qs[d] = d as f32 + 0.5; // mean
            qs[head_dim + d] = d as f32 * 2.0; // var
        }
        let ctx = KVStageCtx::new(&c, 0, None, None, None, Some(&qs));
        let h = ctx
            .tensor(TensorKind::QueryStats)
            .expect("QueryStats 공급 시 Some");
        // shape 계약.
        let sh = h.shape();
        assert_eq!(sh.rows, 2, "rows=2 (mean/var)");
        assert_eq!(sh.cols, head_dim, "cols=head_dim");
        assert!(sh.per_head);
        assert_eq!(h.dtype(), TensorDtype::F32);
        // read_row(0)=mean / (1)=var.
        let mut mean = vec![0.0f32; head_dim];
        let mut var = vec![0.0f32; head_dim];
        h.read_row(0, 0, &mut mean);
        h.read_row(1, 0, &mut var);
        for d in 0..head_dim {
            assert_eq!(mean[d], d as f32 + 0.5, "mean d={d}");
            assert_eq!(var[d], d as f32 * 2.0, "var d={d}");
        }
        // 기존 0~3 kind 무영향 (Key/Value 항상 공급, Scores/AttnWeights 미공급 None).
        assert!(ctx.tensor(TensorKind::Key).is_some());
        assert!(ctx.tensor(TensorKind::Value).is_some());
        assert!(ctx.tensor(TensorKind::Scores).is_none());
        assert!(ctx.tensor(TensorKind::AttnWeights).is_none());
        // 미공급 ctx → QueryStats None.
        let bare = KVStageCtx::new(&c, 0, None, None, None, None);
        assert!(bare.tensor(TensorKind::QueryStats).is_none());
    }

    /// TQS-9: `tensor_kind_from_u32(4)==Some(QueryStats)`, `(5)==None` + 0~3 불변 (MQ-5 역매핑).
    #[test]
    fn tensor_kind_from_u32_query_stats() {
        assert_eq!(tensor_kind_from_u32(0), Some(TensorKind::Key));
        assert_eq!(tensor_kind_from_u32(1), Some(TensorKind::Value));
        assert_eq!(tensor_kind_from_u32(2), Some(TensorKind::AttnWeights));
        assert_eq!(tensor_kind_from_u32(3), Some(TensorKind::Scores));
        assert_eq!(tensor_kind_from_u32(4), Some(TensorKind::QueryStats));
        assert_eq!(tensor_kind_from_u32(5), None);
    }

    #[test]
    fn d2o_stage_executes_full_mechanism_all_dtypes() {
        // End-to-end production path for the extracted `d2o` plugin: real KVStageCtx (raw K via the
        // KeyHandle → dequantize_k) → D2OStage::plan (cosine-nearest + Eq.11 WeightedMerges) →
        // execute_kv_plan (apply_weighted_merges + compact). Proves the plan flows through the
        // engine merge executor + compaction on real F32/F16/Q4_0 buffers without panic, and
        // compacts to the partition keep size (prefix + HH + recent = target_len).
        for dt in [DType::F32, DType::F16, DType::Q4_0] {
            let mut c = mk(dt, 20); // kv_heads=1, head_dim=PHD, current_pos=20, distinct per pos.
            let imp: Vec<f32> = (0..20).map(|i| i as f32).collect();
            let stage = d2o::D2OStage::new(d2o::D2OConfig {
                keep_ratio: 0.75,
                protected_prefix: 4,
                ..d2o::D2OConfig::default()
            });
            let plan = {
                // target_len=12 → keep = 4(prefix) + 6(HH) + 2(recent) = 12.
                let ctx = KVStageCtx::new(&c, 12, Some(&imp), None, None, None);
                assert!(!ctx.kv_on_device(), "CPU buffers → merge enabled");
                stage
                    .plan(&ctx)
                    .expect("d2o plan Some (current 20 > keep 12)")
            };
            match &plan.keep {
                KeepSpec::LayerWide(k) => {
                    assert_eq!(k.len(), 12, "d2o[{dt:?}] keeps prefix+HH+recent = target");
                    assert!(k.windows(2).all(|w| w[0] < w[1]), "ascending keep");
                }
                KeepSpec::PerHead(_) => panic!("d2o is layer-wide"),
            }
            assert!(
                !plan.merges.is_empty(),
                "d2o[{dt:?}] merge enabled → WeightedMerges present"
            );
            execute_kv_plan(&mut c, &plan).unwrap();
            assert_eq!(
                c.current_pos(),
                12,
                "d2o[{dt:?}] executor compacts to keep.len()"
            );
        }
    }
}
