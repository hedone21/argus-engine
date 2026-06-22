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
    KV_PLAN_NOOP, KV_PLAN_OK, KV_STAGE_ABI_VERSION, KVCachePlan, KVCacheStage, KeepSpec, PlanAbi,
    PluginVTableAbi, StageCtx, StageCtxAbi, StageExportAbi, StageParams, TensorDtype, TensorHandle,
    TensorKind, TensorShape, WeightedMerge, find_qcf_estimator,
};
use core::ffi::c_void;
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use super::EvictionPolicy;
use crate::buffer::DType;
use crate::kv::dequant::{dequantize_k, dequantize_v};
use crate::kv::kv_cache::KVCache;

// value-aware production 활성화. feature `caote` ON 시 caote crate 를 force-link 한다 —
// dep 선언만으로는 미참조 rlib 이 링크 제외돼 `#[distributed_slice]` 등록이 누락되기 때문이다.
// 이 1줄이 production 바이너리에서 `find_stage("caote")` 를 가시화한다(session score_based
// 경유 value-aware 동작). feature OFF = 미링크 + `eviction caote` 서브커맨드 부재(clap reject).
#[cfg(feature = "caote")]
use caote as _;

// StreamingLLM production force-link. Extracted from the engine core into the `streaming-llm`
// technique crate; the dep declaration alone leaves the unreferenced rlib out of the link, so
// this one line makes `find_stage("streaming")` visible (the `#[distributed_slice]` registration).
use streaming_llm as _;

// heavy-hitter production force-link. Extracted from the engine core into the `h2o` technique crate;
// makes `find_stage("h2o")` visible (same force-link rationale as streaming above).
use ::h2o as _;

// heavy-hitter+ production force-link. Extracted from the engine core into the `h2o-plus` technique crate
// (the first PerHead-plan stage); makes `find_stage("h2o_plus")` visible.
use h2o_plus as _;

// weighted-merge production force-link. Extracted from the engine core into the `d2o` technique crate
// (registers "d2o", a WeightedMerge-producing stage); makes `find_stage("d2o")` visible. Production
// resolves it via `make_stage_with_args("d2o", &params, &blob)` (eval_setup/build_bench_loop/chat),
// with the d2o-private knobs in the StageArgs blob; the registration must survive fat-LTO.
use d2o as _;

// Sliding-window + no-eviction production force-link. Extracted from the engine core into the
// `sliding-window` (registers "sliding") and `no-eviction` (registers "none") technique crates; the
// dep declaration alone leaves the unreferenced rlib out of the link, so these lines make
// `find_stage("sliding")` / `find_stage("none")` visible (same rationale as streaming/h2o above).
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

// ── KVCachePlan executor + StageBackedPolicy 역어댑터 (World B) ──────────────

/// [`KVCacheStage`] 가 산출한 [`KVCachePlan`] 을 `&mut KVCache` 에 적용한다(변형은
/// 엔진 독점). `StandardFormat::compact` 의 빈-merge 경로와 동일: `compact_keep_positions(keep, 0)` +
/// `set_current_pos(keep.len())`. compact_parity 가 이 경로 ≡ in-place `evict*` 를 4정책×3dtype 에서
/// 증명하므로, plan keep 이 `plan_keep` keep 과 같으면(②a 어댑터 faithful) 버퍼 bit-identical 무회귀.
///
/// pub(crate): the `d2o` plugin's WeightedMerge plan flows through here (apply_weighted_merges +
/// compact) — the production merge-application path.
pub(crate) fn execute_kv_plan(
    cache: &mut KVCache,
    plan: &KVCachePlan,
    layer_idx: usize,
    n_layers: usize,
) -> Result<()> {
    // R-P0-2: optional keep-set dump (no-op unless `ARGUS_DUMP_KEEPSET` is set).
    // Recorded before any compaction so the kept positions are absolute indices
    // into the pre-eviction `[0, current_pos)` range.
    super::keepset_dump::record(cache, plan, layer_idx, n_layers);
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
        KeepSpec::PerHead(heads) => {
            // Per-head executor (stage ⑤): each KV head keeps a different token set (h2o_plus). The
            // plugin emits prefix-inclusive ascending keep-lists of equal length per head; the engine
            // compacts each head independently and sets the single shared current_pos.
            //
            // Per-head compaction requires HeadMajor layout (so a head's tokens are contiguous and
            // shiftable in isolation); on the default SeqMajor cache it is undefined, so bail cleanly
            // rather than panic. Production never reaches this arm — only the score-free/flat fallback
            // (LayerWide) fires there — so this gate is for the head-score-driven (HeadMajor) path.
            if cache.layout() != crate::kv_cache_ops::KVLayout::HeadMajor {
                anyhow::bail!(
                    "per-head (KeepSpec::PerHead) eviction requires HeadMajor KV layout, got {:?}",
                    cache.layout()
                );
            }
            // Per-head + weighted merge has no producer; reject rather than silently drop merges.
            if !plan.merges.is_empty() {
                anyhow::bail!("per-head plan with weighted merges is unsupported");
            }
            // All heads keep the same NUMBER of tokens (the engine's single-current_pos invariant);
            // the new position is that shared length.
            let new_pos = heads.first().map_or(0, |h| h.len());
            for (kv_head, keep) in heads.iter().enumerate() {
                debug_assert_eq!(
                    keep.len(),
                    new_pos,
                    "PerHead keep-lists must be equal length (head {kv_head})"
                );
                cache.compact_keep_positions_for_head(kv_head, keep, 0)?;
            }
            cache.set_current_pos(new_pos);
            Ok(())
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

/// `tensor(Value)` 핸들 — raw V 를 `dequantize_v` 정본으로 읽는다(value-aware 의 v_i).
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

/// `tensor(PrefillAttention)` 핸들 — per-ATTENTION-head(pre-GQA) prefill attention 확률, q_window SUM-pooled.
/// 레이아웃 `[n_heads x prefix_len]` row-major: `data[row*cols + key_pos]`,
/// row=attention head(NOT kv_head), cols=prefix_len. 이 핸들만 per_head:false → kv_head 인자 무시,
/// head 정체성은 `row`. (Key/Value/Scores/AttnWeights/QueryStats 는 kv_head 인덱싱 + per_head:true.)
struct PrefillAttnHandle<'a> {
    data: &'a [f32],
    rows: usize, // = n_heads_q (attention heads, pre-GQA)
    cols: usize, // = prefix_len
}
impl TensorHandle for PrefillAttnHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.rows,
            cols: self.cols,
            per_head: false,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, _kv_head: usize, out: &mut [f32]) {
        debug_assert_eq!(
            out.len(),
            self.cols,
            "PrefillAttention read_row: out.len must == cols(prefix_len)"
        );
        let base = row * self.cols;
        let n = self.cols.min(out.len());
        if base + n <= self.data.len() {
            out[..n].copy_from_slice(&self.data[base..base + n]);
        } else {
            out[..n].fill(0.0);
        }
    }
}

/// `&KVCache`(+ budget + scores) 위로 구현한 [`StageCtx`].
///
/// 모든 텐서/스코어 읽기는 [`StageCtx::tensor`] 단일 경로로 흐른다: Key/Value 핸들은 항상,
/// Scores/AttnWeights 는 `new()` 에 슬라이스가 공급될 때만 `Some`. flat `importance()` 만 zero-copy 직접
/// 노출(D1 예외). builtin LayerWide(sliding/streaming/h2o) + d2o(tensor(Key))는 production 에서 구동,
/// Scores/AttnWeights 공급은 현재 host 테스트(value-aware) 경로 — production eviction-hook threading 은 CLI
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
    /// R-P1-1: prefill-end producer 가 채운 `[n_heads_q x prefix_len]` SUM-pooled PFA 슬라이스.
    /// `None`=미공급(`tensor(PrefillAttention)`→None, byte-identical disabled path).
    prefill_attn_handle: Option<PrefillAttnHandle<'a>>,
}

impl<'a> KVStageCtx<'a> {
    /// 엔진 eviction 경로(+ d2o 동등성/value-aware host 테스트)가 `&KVCache` 위로 ctx 를 만든다.
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
            // R-P1-1: producer 가 채울 때만 `with_prefill_attn` 으로 Some (signature 불변 → 기존
            // caller 전부 무수정 + disabled path 자동 byte-identical).
            prefill_attn_handle: None,
        }
    }

    /// Set the real layer index + total layer count (the engine eviction loop injects these while
    /// iterating caches). Enables per-layer techniques (d2o protected_layers / last-layer protect).
    pub(crate) fn with_layer(mut self, layer_idx: usize, n_layers: usize) -> Self {
        self.layer_idx = layer_idx;
        self.n_layers = n_layers;
        self
    }

    /// R-P1-1: prefill-end producer 가 채운 `[n_heads x prefix_len]` SUM-pooled PFA 슬라이스 주입.
    /// 미호출 시 `tensor(PrefillAttention)==None` (byte-identical disabled path).
    pub(crate) fn with_prefill_attn(
        mut self,
        data: &'a [f32],
        n_heads: usize,
        prefix_len: usize,
    ) -> Self {
        self.prefill_attn_handle = Some(PrefillAttnHandle {
            data,
            rows: n_heads,
            cols: prefix_len,
        });
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
            TensorKind::PrefillAttention => self
                .prefill_attn_handle
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
        last_attn: Option<&[f32]>,
        layer_idx: usize,
        n_layers: usize,
    ) -> Result<()> {
        let plan = {
            // last_attn(AttnWeights, value-aware a_i): production eviction 경로가 score accumulator 의
            // last_step_head_attn 을 공급할 때 Some — value-aware 기법(caote)이 ctx.attn_weight 로 읽는다.
            // QueryStats(MQ-4 e2e seam)는 production eviction 경로에서 미공급(None) — score-active
            // 측정 하네스가 별도로 공급한다(dump_importance.rs).
            let ctx = KVStageCtx::new(cache, target_len, importance, None, last_attn, None)
                .with_layer(layer_idx, n_layers);
            self.stage.plan(&ctx)
        };
        if let Some(plan) = plan {
            execute_kv_plan(cache, &plan, layer_idx, n_layers)?;
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
    /// stage (h2o_plus) sees `ctx.head_score(kv_head, pos)` and emits a [`KeepSpec::PerHead`] plan;
    /// the engine then compacts each head independently in [`execute_kv_plan`]. `flat_importance`
    /// remains available via `ctx.importance()` for the stage's score-free / flat fallback.
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
        let plan = {
            let ctx = KVStageCtx::new(
                cache,
                target_len,
                Some(flat_importance),
                Some(head_importance),
                None,
                None,
            );
            self.stage.plan(&ctx)
        };
        if let Some(plan) = plan {
            execute_kv_plan(cache, &plan, layer_idx, n_layers)?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        self.stage.name()
    }
}

/// Test helper: build the out-of-tree `h2o` stage wrapped as a legacy [`EvictionPolicy`]
/// (`StageBackedPolicy`). Used by CacheManager / EvictionHandler tests after heavy-hitter was extracted
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

/// Build the out-of-tree `sliding` stage wrapped as a legacy [`EvictionPolicy`]
/// (`StageBackedPolicy`). Convenience constructor used by CacheManager / EvictionHandler tests (and
/// engine integration tests) after SlidingWindowPolicy was extracted to the `sliding-window` plugin
/// crate — production resolves "sliding" the same way (`make_stage` → plugin → StageBackedPolicy).
pub fn sliding_backed_policy(window: usize, protected_prefix: usize) -> Box<dyn EvictionPolicy> {
    let p = StageParams {
        eviction_window: window,
        protected_prefix,
        ..Default::default()
    };
    Box::new(StageBackedPolicy::new(make_stage("sliding", &p).expect(
        "sliding stage registered (force-linked sliding-window plugin)",
    )))
}

/// Build the out-of-tree `none` stage wrapped as a legacy [`EvictionPolicy`] (`StageBackedPolicy`)
/// — a no-op policy. Convenience constructor used by tests after NoEvictionPolicy was extracted to
/// the `no-eviction` plugin crate; production resolves "none" the same way.
pub fn none_backed_policy() -> Box<dyn EvictionPolicy> {
    Box::new(StageBackedPolicy::new(
        make_stage("none", &StageParams::default())
            .expect("none stage registered (force-linked no-eviction plugin)"),
    ))
}

/// Whether the named stage is score-based (consumes importance) — the generic capability lookup the
/// CLI/chat/eval/bench paths use instead of `matches!(name, "h2o" | "d2o" | "caote" | "rkv" | ...)`.
/// Reads the plugin's declared [`StageCaps`](argus_extension_api::stage_caps). Unknown /
/// unregistered (incl. dynamic `.so` stages whose caps don't cross the ABI yet) → `false`.
pub fn stage_is_score_based(name: &str) -> bool {
    argus_extension_api::stage_caps(name)
        .map(|c| !c.reads.is_empty())
        .unwrap_or(false)
}

/// The default `--protected-prefix` the named stage declares (`4` for score-based, `0` = "engine
/// picks its own fallback"). The generic lookup that replaces the `match name { ... => 4 }` prefix
/// tables. Reads the plugin's declared [`StageCaps`]. Unknown → `0`.
pub fn stage_default_protected_prefix(name: &str) -> usize {
    argus_extension_api::stage_caps(name)
        .map(|c| c.default_protected_prefix)
        .unwrap_or(0)
}

/// Whether the named stage's `plan()` may emit a weighted-merge plan (à la weighted-merge). The generic lookup
/// the eval/QCF path uses instead of the `eviction_policy() == "d2o"` name match — it selects a
/// merge-compensation estimator + K readback. Reads the plugin's declared [`StageCaps`]. Unknown →
/// `false` (pure-drop).
pub fn stage_produces_merge_plan(name: &str) -> bool {
    argus_extension_api::stage_caps(name)
        .map(|c| c.produces_merge_plan)
        .unwrap_or(false)
}

/// 모든 force-link 된 빌트인 stage 크레이트가 `KV_CACHE_STAGES` 에 등록됐는지 단언한다 — eviction
/// CacheManager build 진입 시 1회 호출. fat-LTO `--gc-sections` 가 force-link 된 크레이트의 linkme
/// 등록을 silent drop 하면 `stage_caps` 가 그 이름을 더 못 풀어 `Err` 로 fail-fast 한다(release 에서
/// 정책 이름 미해석 → 조용한 폴백 방지). caps 의 의미(is_score_based/protected_prefix/produces_merge_plan)
/// 는 plugin 이 단독 소유하므로 여기서 재선언하지 않는다 — 등록 존재(resolution)만 검증한다.
pub fn ensure_builtin_stages_registered() -> Result<()> {
    // The force-linked built-in stage crate names (the `use X as _;` block above). This list is the
    // fail-fast ANCHOR: it can't be derived from the registry, because the registry is exactly what
    // we verify — if fat-LTO `--gc-sections` drops a crate, `stage_caps` stops resolving its name
    // and we bail. It does NOT re-declare any plugin's caps (those are read from the registry by
    // `stage_is_score_based` / `stage_default_protected_prefix` / `stage_produces_merge_plan` and
    // owned solely by the plugin). Mirrors `ensure_score_producers_registered` /
    // `ensure_layer_scorers_registered`, which likewise keep a hardcoded name list + assert only
    // resolution.
    for name in ["sliding", "streaming", "h2o", "h2o_plus", "d2o"] {
        if argus_extension_api::stage_caps(name).is_none() {
            anyhow::bail!(
                "built-in KVCacheStage '{name}' not registered — suspect linkme fat-LTO \
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

// All built-in LayerWide/score stages are now out-of-tree technique crates, force-linked above:
//   "sliding"   → `sliding-window`   (use sliding_window as _)
//   "none"      → `no-eviction`      (use no_eviction as _)
//   "streaming" → `streaming-llm`    (use streaming_llm as _)
//   "h2o"       → `h2o`              (use ::h2o as _)
//   "d2o"       → `d2o`              (use d2o as _; WeightedMerge via apply_weighted_merges)
//   "rkv"       → `rkv`              (#[cfg(feature = "rkv")] use rkv as _; λ rides the StageArgs blob)
//   "caote"     → `caote`           (#[cfg(feature = "caote")] use caote as _)
// The engine names none of them here — each registers itself into KV_CACHE_STAGES via linkme, and
// production resolves them by name through `make_stage(_with_args)`. The force-link references above
// are the only place the engine spells a stage crate (so its #[distributed_slice] survives fat-LTO).

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

/// R-P1-1: 등록된 KV stage 중 `TensorKind::PrefillAttention` 을 읽는 첫 stage 이름(없으면 `None`).
/// `build_standard_loop` 의 PFA producer arming gate — **caps-driven**(plugin-name 무지, `wants_query_stats`
/// 와 동형). PR1 은 그런 builtin 이 0개라 항상 `None`(arming dormant → byte-identical). R-P1-2 의
/// per-head keep-set plugin 이 `caps.reads ∋ PrefillAttention` 으로 등록되면 그때 활성화된다.
pub fn find_prefill_attn_stage_name() -> Option<String> {
    // 정적(linkme) 빌트인.
    if let Some(reg) = argus_extension_api::KV_CACHE_STAGES
        .iter()
        .find(|r| r.caps.reads.contains(&TensorKind::PrefillAttention))
    {
        return Some(reg.name.to_string());
    }
    // 동적(dlopen) plugin — caps 는 stage_caps(name) 로 해석.
    dynamic_registered_stage_names().into_iter().find(|name| {
        argus_extension_api::stage_caps(name)
            .is_some_and(|c| c.reads.contains(&TensorKind::PrefillAttention))
    })
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
        5 => Some(TensorKind::PrefillAttention),
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
    // value-aware 의 force-link 는 production(module-level `#[cfg(feature = "caote")] use caote as _`)
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
        // (M-F) value-aware crate 의 cross-crate 등록 + KVStageCtx(V 공급)로 value-aware plan 산출 →
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
            KeepSpec::PerHead(_) => panic!("v1 value-aware 는 LayerWide"),
        }
        assert!(plan.merges.is_empty());
        execute_kv_plan(&mut c, &plan, 0, 1).unwrap();
        assert_eq!(c.current_pos(), 4, "executor 가 keep.len() 로 compact");
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

    // The World-A↔World-B sliding parity test (and its `region`/`sb_params` helpers) were removed
    // when SlidingWindowPolicy was extracted to the `sliding-window` plugin crate: the plugin is
    // plan-only, so there is no in-place World-A path left to compare. The plugin's keep-list spec is
    // pinned by its own unit tests; beta3_eviction_stage_equivalence.rs proves the World-B path
    // end-to-end across F32/F16/Q4_0; and d2o_stage_executes_full_mechanism_all_dtypes below still
    // exercises the full find_stage → make → StageBackedPolicy → KVStageCtx → execute_kv_plan chain.

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
    fn kvstagectx_prefill_attn_handle_round_trip() {
        // R-P1-1 seam: with_prefill_attn → tensor(PrefillAttention) Some, shape
        // {rows:n_heads_q, cols:prefix_len, per_head:false}, read_row(row, _kv_head) = data[row*cols..].
        // 미공급 ctx → None(distinctness: decode/unarmed 에서 wrong-tensor 아닌 loud None).
        let c = mk(DType::F32, 4);
        let n_heads_q = 2;
        let prefix_len = c.current_pos(); // 4
        // [n_heads_q x prefix_len] row-major: data[row*prefix_len + key_pos].
        let pfa: Vec<f32> = (0..n_heads_q * prefix_len)
            .map(|i| i as f32 + 0.5)
            .collect();
        let ctx = KVStageCtx::new(&c, 0, None, None, None, None)
            .with_prefill_attn(&pfa, n_heads_q, prefix_len);
        let h = ctx
            .tensor(TensorKind::PrefillAttention)
            .expect("PrefillAttention handle Some when supplied");
        let shape = h.shape();
        assert_eq!(shape.rows, n_heads_q);
        assert_eq!(shape.cols, prefix_len);
        assert!(
            !shape.per_head,
            "PFA per_head must be false (per-attention-head)"
        );
        assert_eq!(h.dtype(), TensorDtype::F32);
        // read_row: per-attention-head row, kv_head 인자 무시(per_head=false).
        let mut out = vec![0.0f32; prefix_len];
        h.read_row(1, 999 /* kv_head ignored */, &mut out);
        for kp in 0..prefix_len {
            assert_eq!(out[kp], (prefix_len + kp) as f32 + 0.5, "row 1 key {kp}");
        }
        // 미공급 → None (decode/unarmed distinctness).
        let bare = KVStageCtx::new(&c, 0, None, None, None, None);
        assert!(bare.tensor(TensorKind::PrefillAttention).is_none());
        // u32 round-trip 역매핑(disc 5).
        assert_eq!(tensor_kind_from_u32(5), Some(TensorKind::PrefillAttention));
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

    /// TQS-9: `(4)==Some(QueryStats)`, `(5)==Some(PrefillAttention)`(R-P1-1), `(6)==None` + 0~3 불변.
    #[test]
    fn tensor_kind_from_u32_query_stats() {
        assert_eq!(tensor_kind_from_u32(0), Some(TensorKind::Key));
        assert_eq!(tensor_kind_from_u32(1), Some(TensorKind::Value));
        assert_eq!(tensor_kind_from_u32(2), Some(TensorKind::AttnWeights));
        assert_eq!(tensor_kind_from_u32(3), Some(TensorKind::Scores));
        assert_eq!(tensor_kind_from_u32(4), Some(TensorKind::QueryStats));
        assert_eq!(tensor_kind_from_u32(5), Some(TensorKind::PrefillAttention));
        assert_eq!(tensor_kind_from_u32(6), None);
    }

    #[test]
    fn d2o_stage_executes_full_mechanism_all_dtypes() {
        // End-to-end production path for the extracted `d2o` plugin: real KVStageCtx (raw K via the
        // KeyHandle → dequantize_k) → D2OStage::plan (cosine-nearest + WeightedMerges) →
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
            execute_kv_plan(&mut c, &plan, 0, 1).unwrap();
            assert_eq!(
                c.current_pos(),
                12,
                "d2o[{dt:?}] executor compacts to keep.len()"
            );
        }
    }

    /// (stage ⑤) End-to-end per-head path: the out-of-tree `h2o_plus` plugin emits a
    /// `KeepSpec::PerHead` plan from per-(kv_head, pos) scores, and the engine's per-head executor
    /// compacts each KV head independently — **without bailing**. Proves the F5 score source
    /// (`StageBackedPolicy::evict_with_head_scores` → `tensor(Scores)`) + the per-head executor work
    /// end to end, and that heads diverge (head 0 keeps different heavy hitters than head 1).
    #[test]
    fn h2o_plus_per_head_executor_runs_without_bail() {
        use crate::kv_cache_ops::KVLayout;

        const MAX_SEQ: usize = 64;
        const HD: usize = 4;
        let n_kv_heads = 2;

        // HeadMajor cache (per-head compaction requires it) with a distinct f32 marker per (head, pos).
        let backend = Arc::new(CpuBackend::new());
        let buf = || {
            Arc::new(SharedBuffer::new(
                n_kv_heads * MAX_SEQ * HD * std::mem::size_of::<f32>(),
                DType::F32,
            ))
        };
        let shape = Shape::new(vec![1, MAX_SEQ, n_kv_heads, HD]);
        let mut c = KVCache::new(
            Tensor::new(shape.clone(), buf(), backend.clone()),
            Tensor::new(shape, buf(), backend),
            MAX_SEQ,
        )
        .with_layout(KVLayout::HeadMajor);
        c.current_pos = 20;
        // marker(head, pos) = (pos+1) + head*1000 — distinct so a wrong keep shows up immediately.
        let marker = |head: usize, pos: usize| (pos + 1) as f32 + head as f32 * 1000.0;
        for head in 0..n_kv_heads {
            for pos in 0..20 {
                let off = c.offset(pos, head);
                let k = c.k_buffer.as_mut_slice::<f32>();
                for d in 0..HD {
                    k[off + d] = marker(head, pos);
                }
            }
        }

        // Per-(kv_head, pos) importance, stride = MAX_SEQ: head 0 prefers tokens 5,6,7; head 1 → 10,11,12.
        let mut head_imp = vec![0.0f32; n_kv_heads * MAX_SEQ];
        for (rank, &pos) in [5usize, 6, 7].iter().enumerate() {
            head_imp[pos] = 10.0 - rank as f32;
        }
        for (rank, &pos) in [10usize, 11, 12].iter().enumerate() {
            head_imp[MAX_SEQ + pos] = 10.0 - rank as f32;
        }
        let flat = vec![1.0f32; MAX_SEQ];

        // Resolve h2o_plus through the registry (force-linked plugin) and run the per-head path.
        let stage = make_stage(
            "h2o_plus",
            &StageParams {
                keep_ratio: 0.5,
                protected_prefix: 4,
                ..Default::default()
            },
        )
        .expect("h2o_plus stage registered (force-linked h2o-plus plugin)");
        let policy = StageBackedPolicy::new(stage);
        // target=10, prefix=4, keep_ratio=0.5 → keep=10, hh_budget=3, recent_start=17.
        policy
            .evict_with_head_scores(&mut c, 10, &flat, &head_imp, n_kv_heads, 0, 1)
            .expect("per-head executor runs without bail");

        // All heads compacted to the same count (prefix 4 + HH 3 + recent 3 = 10).
        assert_eq!(c.current_pos(), 10, "uniform per-head current_pos");

        // Head 0 kept its own HH (5,6,7); head 1 kept (10,11,12). After compaction the slots after
        // the 4-token prefix hold each head's heavy hitters, then the recent window (17,18,19).
        let at = |head: usize, slot: usize| c.k_buffer.as_slice::<f32>()[c.offset(slot, head)];
        // prefix preserved for both heads.
        for head in 0..n_kv_heads {
            for slot in 0..4 {
                assert_eq!(
                    at(head, slot),
                    marker(head, slot),
                    "head {head} prefix slot {slot}"
                );
            }
        }
        // head 0: slots 4,5,6 = tokens 5,6,7.
        assert_eq!(at(0, 4), marker(0, 5));
        assert_eq!(at(0, 5), marker(0, 6));
        assert_eq!(at(0, 6), marker(0, 7));
        // head 1: slots 4,5,6 = tokens 10,11,12 (DIFFERENT tokens than head 0 → per-head divergence).
        assert_eq!(at(1, 4), marker(1, 10));
        assert_eq!(at(1, 5), marker(1, 11));
        assert_eq!(at(1, 6), marker(1, 12));
        // recent window (17,18,19) tail, same positions for both heads.
        for head in 0..n_kv_heads {
            for (i, &pos) in [17usize, 18, 19].iter().enumerate() {
                assert_eq!(
                    at(head, 7 + i),
                    marker(head, pos),
                    "head {head} recent {pos}"
                );
            }
        }
    }

    // ── B2-2 handshake precondition: declared `StageCaps.reads` ⊇ what `plan()` reads ──

    use std::cell::RefCell;
    use std::collections::HashSet;

    /// One [`TensorHandle`] backing any kind — `head_dim` cols for Key/Value/QueryStats, 1 col for
    /// Scores/AttnWeights, filled with distinct nonzero values so no plan degenerates.
    struct AnyHandle {
        cols: usize,
        rows: usize,
    }
    impl TensorHandle for AnyHandle {
        fn shape(&self) -> TensorShape {
            TensorShape {
                rows: self.rows,
                cols: self.cols,
                per_head: true,
            }
        }
        fn dtype(&self) -> TensorDtype {
            TensorDtype::F32
        }
        fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
            for (d, o) in out.iter_mut().enumerate() {
                *o = 1.0 + (row + kv_head + d) as f32 * 0.5;
            }
        }
    }

    /// A [`StageCtx`] that supplies ALL five [`TensorKind`]s with valid f32 data (so every `plan()`
    /// takes its maximal-read path and nothing early-returns on a missing tensor) and records every
    /// distinct kind the stage reads. `importance()` records [`TensorKind::Scores`] (the flat
    /// zero-copy form of the per-token score, the D1 exception). It deliberately does NOT reuse
    /// `KVStageCtx`, which hardwires Key/Value-always and only supplies Scores/AttnWeights/QueryStats
    /// when slices are passed — that under-supplies and would let plans early-return, masking leaks.
    struct RecordingCtx {
        cur: usize,
        tgt: usize,
        n_kv_heads: usize,
        head_dim: usize,
        imp: Vec<f32>,
        seen: RefCell<HashSet<TensorKind>>,
        h_key: AnyHandle,
        h_value: AnyHandle,
        h_scores: AnyHandle,
        h_attn: AnyHandle,
        h_qstats: AnyHandle,
    }

    impl RecordingCtx {
        fn new(n_kv_heads: usize, head_dim: usize, cur: usize, tgt: usize) -> Self {
            Self {
                cur,
                tgt,
                n_kv_heads,
                head_dim,
                imp: (0..cur).map(|i| i as f32 + 1.0).collect(),
                seen: RefCell::new(HashSet::new()),
                h_key: AnyHandle {
                    cols: head_dim,
                    rows: cur,
                },
                h_value: AnyHandle {
                    cols: head_dim,
                    rows: cur,
                },
                h_scores: AnyHandle { cols: 1, rows: cur },
                h_attn: AnyHandle { cols: 1, rows: cur },
                h_qstats: AnyHandle {
                    cols: head_dim,
                    rows: 2,
                },
            }
        }
    }

    impl StageCtx for RecordingCtx {
        fn current_pos(&self) -> usize {
            self.cur
        }
        fn target_len(&self) -> usize {
            self.tgt
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn n_layers(&self) -> usize {
            2 // non-last → d2o is_protected(0, 2) == false, so its merge/K-read path fires.
        }
        fn kv_on_device(&self) -> bool {
            false // CPU-resident → caote/d2o raw-read + merge paths run.
        }
        fn n_kv_heads(&self) -> usize {
            self.n_kv_heads
        }
        fn head_dim(&self) -> usize {
            self.head_dim
        }
        fn importance(&self) -> Option<&[f32]> {
            self.seen.borrow_mut().insert(TensorKind::Scores);
            Some(&self.imp)
        }
        fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
            self.seen.borrow_mut().insert(kind);
            Some(match kind {
                TensorKind::Key => &self.h_key,
                TensorKind::Value => &self.h_value,
                TensorKind::Scores => &self.h_scores,
                TensorKind::AttnWeights => &self.h_attn,
                TensorKind::QueryStats => &self.h_qstats,
                TensorKind::PrefillAttention => return None,
            })
        }
    }

    /// Handshake (B2-2) precondition guard: for EVERY registered score-based stage, the set of
    /// [`TensorKind`]s its `plan()` actually reads must be a subset of its declared
    /// `StageCaps.reads`. An undeclared read means the future buffer-allocation handshake would fail
    /// to wire a tensor the stage silently consumes. caote/d2o/rkv `reads` were widened to satisfy
    /// this (see their `KVCacheStageReg`s).
    #[test]
    fn plan_reads_are_subset_of_declared_caps() {
        let mut checked = 0usize;
        for name in registered_names() {
            let Some(caps) = argus_extension_api::stage_caps(name) else {
                continue; // dynamic `.so` stage whose caps don't cross the ABI — nothing to check.
            };
            if caps.reads.is_empty() {
                continue; // score-free (sliding/streaming/none/example) — reads nothing.
            }
            let declared: HashSet<TensorKind> = caps.reads.iter().copied().collect();
            let stage = make_stage(
                name,
                &StageParams {
                    keep_ratio: 0.5,
                    protected_prefix: 4,
                    ..Default::default()
                },
            )
            .expect("score-based stage builds via make_stage");
            // cur > tgt + every kind supplied + non-empty importance → maximal-read path.
            let ctx = RecordingCtx::new(2, PHD, 16, 8);
            let _ = stage.plan(&ctx);
            let recorded = ctx.seen.into_inner();
            let undeclared: Vec<_> = recorded.difference(&declared).collect();
            assert!(
                undeclared.is_empty(),
                "stage '{name}' plan() reads {recorded:?} but declares StageCaps.reads = \
                 {declared:?}; undeclared kinds {undeclared:?} would be unwired by the B2-2 \
                 handshake — widen its reads.",
            );
            checked += 1;
        }
        assert!(
            checked > 0,
            "no score-based stages registered — force-link / feature regression?"
        );
    }
}
