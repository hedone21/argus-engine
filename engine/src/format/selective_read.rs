//! `SelectiveRead` capability — `KVCacheFormat` 의 선택적 읽기 opt-in 표면.
//!
//! 설계 SSOT: `docs/adr/0011-kv-read-plan-surface.md` D4 / Amendment A1.
//!
//! **capability opt-in**: `KVCacheFormat` base trait(6-method)에 추가하지 않는다 — ISP 보존.
//! 미지원 format 은 full read 폴백(정확). `SelectiveRead` 를 구현한 format 만 선택적 읽기를 제공.
//!
//! 첫 구현체: `StandardFormat` (`engine/src/kv/standard_format.rs`).
//! 미구현: KIVI/opaque → 엔진 폴백 = full read.

use anyhow::Result;
use argus_extension_api::{KVReadPlan, KVReadStage, ReadGranularity};

use crate::backend::Backend;
use crate::format::AttnDims;
use crate::tensor::Tensor;

/// KV 캐시의 **선택적 읽기** capability.
///
/// `attention_into_selected` 는 `select` 된 KV 위치/페이지만 읽어 attention 을 수행한다.
/// **정확성 계약 아님** — `select` 가 전체면 `attention_into` 와 bit-identical(Tier 1 게이트),
/// 부분 select 는 *근사 답*(softmax 분모가 부분집합으로 정규화됨 — Quest 의 의도된 근사).
///
/// `KVCacheFormat` base trait 무변경: 이 capability 를 구현 안 한 format 은
/// 엔진이 plan 을 무시하고 `attention_into`(full read) 를 호출한다.
pub trait SelectiveRead {
    /// `select` 된 KV 위치만 읽는 attention.
    ///
    /// - `select`: ascending pos 목록(Token 단위) 또는 page index 목록(Page 단위).
    /// - `granularity`: `ReadGranularity::Token` 이면 `select` 를 pos 그대로 사용,
    ///   `Page { page_size }` 이면 각 page index 를 `[page*page_size .. (page+1)*page_size)` pos 범위로 전개.
    /// - `scores`: `Some` 이면 선택된 토큰에 대한 post-softmax score 를 기록.
    /// - 기타 계약은 `KVCacheFormat::attention_into` 와 동일.
    #[allow(clippy::too_many_arguments)]
    fn attention_into_selected(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        dims: AttnDims,
        select: &[usize],
        granularity: ReadGranularity,
        scores: Option<&mut [f32]>,
    ) -> Result<()>;

    /// layer `i` attention 직전에 read stage 의 `read_plan` 을 자기 캐시 위에서 호출
    /// A1.1d/A1.3).
    ///
    /// **ctx 구성 캡슐화**: read stage 가 읽는 [`argus_extension_api::StageCtx`](`tensor(Key)`/`tensor(QueryStats)`/
    /// geometry)는 concrete 캐시(예: `StandardFormat` 의 `Mutex<KVCache>`) 위에서만 만들 수 있다. 이를
    /// format 내부에 위임해 (1) 캐시 borrow/lock 이 format 안에 갇히고, (2) `attention_into_selected`(다시
    /// lock)와 충돌하지 않으며(plan 산출 후 owned `KVReadPlan` 반환 → borrow 종료), (3) `SelectiveRead`
    /// 미구현 format 은 `KVCacheFormat::as_selective_read()==None` 이라 엔진이 자동 full read 폴백한다(D4).
    ///
    /// `None` = full read(read stage 가 `None` 반환). 반환 plan 은 *근사 힌트*(정확성 계약 아님, D3).
    fn read_plan(&self, rs: &dyn KVReadStage, layer_idx: usize) -> Option<KVReadPlan>;
}
