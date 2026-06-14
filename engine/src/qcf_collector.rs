//! Layer importance collect/lookup trait (L2).
//!
//! L3-inference (`models/transformer.rs` + `models/weights/decider.rs`) 가
//! L3-qcf의 `ImportanceCollector` / `ImportanceTable` concrete struct 에
//! 직접 의존하는 패턴을 trait inversion으로 해소한다 (S-3b-4 γ-2 적용).
//!
//! 구현체는 `engine/src/qcf/layer_importance.rs` 의 `ImportanceCollector`
//! (mut state + 측정 로직) 와 `ImportanceTable` (산출물 컨테이너).
//! caller (models)는 `&mut dyn ImportanceCollect` / `&dyn ImportanceLookup` 로
//! trait dispatch.

use crate::qcf_types::{ImportanceEntry, SubLayer};

/// Layer importance 측정 수집 trait.
///
/// transformer forward path 가 매 layer 종료 시점에 `record_after` 를
/// 호출하여 hidden state 의 cosine-similarity 기반 importance 를 누적한다.
///
/// `Send` 만 요구 — collector 는 단일 thread 보유 mut state.
pub trait ImportanceCollect: Send {
    /// Layer forward 진입 직전에 hidden state 입력을 snapshot.
    ///
    /// MeanPool / DirectAttn / ShortGptBi 등 공식이 이 snapshot 을
    /// `record_after` 에서 cosine-similarity 비교의 base 로 사용한다.
    fn snapshot_before(&mut self, x: &[f32], seq_len: usize, dim: usize);

    /// 한 layer (또는 sub-layer) 의 hidden state 출력을 수집한다.
    ///
    /// 호출 형태:
    /// - `x`: layer output slice (flattened `[seq_len * hidden_size]` F32)
    /// - `seq_len`: 토큰 길이
    /// - `hidden_size`: hidden dimension
    /// - `layer`: layer index (0-based)
    /// - `sublayer`: Full / Attention / Mlp
    fn record_after(
        &mut self,
        x: &[f32],
        seq_len: usize,
        hidden_size: usize,
        layer: usize,
        sublayer: SubLayer,
    );
}

/// Layer importance 산출물 read-only lookup trait.
///
/// `ImportanceTable` 의 read-only API 를 노출한다. decider/eval 등이 layer
/// importance 를 *읽고 정렬/필터링* 하는 용도로 사용.
pub trait ImportanceLookup: Send + Sync {
    /// 측정 결과가 비어있는지 (warmup 미수행 또는 비활성화).
    fn is_empty(&self) -> bool;

    /// 모든 importance entry 의 read-only slice.
    /// decider 등이 layer/sublayer 기준으로 filter/sort 한다.
    fn entries(&self) -> &[ImportanceEntry];
}

/// D2O layer-level variance 수집 trait (S-C3).
///
/// L3-inference (transformer forward) 가 매 layer attention 종료 시점에
/// per-token attention column-sums 를 수집하여 layer-level keep ratio 할당
/// (compute_budgets) 에 사용한다. caller(transformer/forward)는
/// `Option<&mut dyn VarianceObserver>` 로 trait dispatch.
///
/// 구현체는 `engine/src/pressure/d2o_layer_alloc.rs::D2OVarianceCollector`.
pub trait VarianceObserver: Send {
    /// Per-layer attention column-sums 수집. hot path 호출 (per-layer).
    #[allow(clippy::too_many_arguments)]
    fn collect_layer(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        cache_seq_len: usize,
        q_stride: usize,
        k_stride: usize,
        kv_head_stride: usize,
        start_pos: usize,
    );
}
