//! `KVCacheFormat` base trait + `Merge`/`AttnDims` (§4.1, C2).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §4.1 (R4 연혁 + ④ KIVI creep 제거).
//!
//! KV 캐시의 **state 책임**(geometry · mutation · attention)을 **storage-format-agnostic**
//! 하게 제공하는 base trait — geometry 3(`idx`/`current_pos`/`capacity`) + mutation 2
//! (`write_kv`/`write_kv_batch`) + attention 1(`attention_into`) = **6 method**.
//! `compact` 는 에서 폐기 — production 정본은 `execute_kv_plan`(stage_registry.rs).
//! base-trait-handle 을 든 Stage 는 geometry·mutation 만 알고, forward 는 `attention_into` 로
//! q→out 만 보므로 양쪽 다 dtype/codebook/rotation/sparse pattern 을 모른다(impl 이 캡슐화,
//! `INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`).
//!
//! **guard rail: impl 은 여기 금지** (§2.1) — `StandardFormat`/`QuantWindowFormat` impl 은 `pressure/`
//! (그들이 감싸는 `KVCache`/`QuantizedRecentWindowCache` 옆)에 둔다.
//!
//! **Phase α-K substep (1)** 신설 — purely additive, host-only. 기존 `KVCacheOps`
//! 경로(`kv_cache_ops.rs`)와 공존하며, 이제 표준/KIVI/offload forward 가 이 trait 을
//! trait-object(`Arc<dyn KVCacheFormat>`)로 live 사용한다.

use anyhow::Result;

use crate::backend::Backend;
use crate::tensor::Tensor;

/// per-call attention 파라미터 — cache 가 `self` 로 알 수 없는 외부 값만 (§4.1, R4).
///
/// `n_heads_kv`/`head_dim`/`capacity`/`current_pos`/`scale`(=1/√head_dim) 는 전부 format 내부.
#[derive(Clone, Copy)]
pub struct AttnDims {
    /// GQA 쿼리 헤드 (q 속성 — KV 캐시는 `kv_heads` 만 앎).
    pub n_heads_q: usize,
    /// Gemma3 local SWA window (global 이면 `None`).
    pub window: Option<usize>,
}

/// D2O merge 의미 기반의 compact merge 명세 (§4.1 `compact(keep, merges)`).
///
/// "evicted 토큰들(`from`)을 retained(kept) 토큰(`into`)에 가중 병합" — `d2o` 플러그인의 cosine-nearest
/// 와 Eq.11 가중 merge(엔진 executor `apply_weighted_merges` 가 적용)의 의미를 trait 표면으로 표현한다.
/// 가중치(Eq.11 `w_c`/`w_e`, EMA threshold)는 impl 의 config
/// 책임이라 여기 담지 않는다 — `from` 의 코사인 유사도·필터링·정규화는 D2O impl 이 자체 수행한다.
///
/// `into`/`from` 은 **compact 적용 직전(pre-compact)의 논리 위치**다 — compact 가 keep 을 앞으로
/// 당기기 전 좌표계에서 해석한다(`scatter_reduce_merge_layer_wide` 가 `cache.offset(pos, h)` 로
/// 원위치를 읽는 것과 동일). impl 은 merge 를 buffer 에 in-place 적용한 뒤 `keep` compaction 을
/// 수행한다(D2O Step 5 → Step 6 순서).
#[derive(Clone, Debug)]
pub struct Merge {
    /// 병합 대상 retained 토큰의 위치 (가중 합이 누적될 자리).
    pub into: usize,
    /// 병합될 evicted 토큰들의 위치.
    pub from: Vec<usize>,
}

/// KV 캐시의 state 책임을 storage-format-agnostic 하게 제공하는 base trait (§4.1).
///
/// impl: `StandardFormat`(F32/F16/Q4_0) / `QuantWindowFormat`(Q2 + residual). 새 Format = 새 impl +
/// paired attention kernel (`INV-KVCACHELAYER-PAIRED-KERNEL`), base trait·forward 변경 0.
pub trait KVCacheFormat: Send + Sync {
    // ── geometry (3) ──

    /// transformer layer 인덱스 (이 format 이 어느 decoder layer 의 KV 인가).
    fn idx(&self) -> usize;

    /// 현재 캐시에 유효한 토큰 수.
    fn current_pos(&self) -> usize;

    /// 물리 버퍼 용량 (토큰 단위).
    fn capacity(&self) -> usize;

    // ── mutation (2) ──

    /// 단일 토큰 write (decode). 입력 shape: `[batch, 1, kv_heads, head_dim]`.
    ///
    /// `backend` = KV write 가 hardware 를 알아야 하는 두 경우의 핸들 (§9.1 (3a) form A):
    /// (1) producer tensor 가 host-mapped GPU 메모리(CUDA pinned / OpenCL UMA)면 device 커널
    /// 완료 전 stale 읽기 방지를 위한 `synchronize` (구 `update_kv_cache` 의 readback 계약),
    /// (2) GPU decode fast-path 의 fused cast+scatter 커널 dispatch (`kv_scatter_*`). `backend` 는
    /// execution-owned 범용 핸들이라 format⊥hardware (`attention_into` 가 이미 받는 것과 동일 패턴)
    /// — base trait 표면에 dtype/codebook 이 새지 않는다(`INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`).
    fn write_kv(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()>;

    /// prefill multi-token write. 입력 shape: `[batch, seq_len, kv_heads, head_dim]`.
    ///
    /// `backend` = `write_kv` 와 동일 계약. (GPU prefill batch scatter 흡수는 후속 substep.)
    fn write_kv_batch(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()>;

    // ── attention (1) ──

    /// q→out attention. impl 이 paired kernel dispatch (NVIDIA fused / Adreno dequant / CPU).
    ///
    /// `scores` 가 `Some` 이면 raw post-softmax score 를 기록한다(생산 seam — 누적·소비는 밖).
    /// KIVI AWQE 자기-need 는 impl 내부에서 자가 흡수한다(base trait 에 `needs_attn_scores` 없음,
    /// §4.1 R4 ③).
    fn attention_into(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        dims: AttnDims,
        scores: Option<&mut [f32]>,
    ) -> Result<()>;

    // ── capability-handle (default) ──

    /// 선택적 읽기 capability 핸들 (INV-STAGE-LAYER-HANDLE capability-handle).
    ///
    /// `Some` 이면 이 format 은 read-plan 기반 선택적 읽기([`crate::format::SelectiveRead`])를 제공한다.
    /// **default = `None`**(미지원) — base trait 6-method 의미 불변, default 제공이라 기존 impl 무영향이고
    /// downcast/`as_any` 금지 제약과 양립한다. `StandardFormat` 만 override. read stage 가 plan 을 내도
    /// 이 핸들이 `None` 인 format 은 엔진이 plan 을 무시하고 `attention_into`(full read) 로 폴백한다(D4).
    fn as_selective_read(&self) -> Option<&dyn crate::format::SelectiveRead> {
        None
    }
    // as_any() 없음 — downcast 의도적 차단.
    // dtype() / KVCacheView 없음 — Stage 가 어느 Format 인지 모름.
    // needs_attn_scores() 없음 — KIVI AWQE 자기-need 는 impl 내부 흡수 (§4.1 R4 ②③).
}
