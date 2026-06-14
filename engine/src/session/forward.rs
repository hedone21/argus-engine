//! [`Forward`] trait + concrete implementations (Phase 4-3+).
//!
//! Each submodule provides one `Forward` variant that the `DecodeLoopBuilder`
//! can be wired with. The first is [`model_forward::ModelForward`] — the
//! standard `KVCache`-backed path that wraps
//! [`crate::models::transformer::TransformerModel::forward_into`].
//!
//! Phase 4-5-a adds [`kivi_forward::KiviForward`] and
//! [`offload_forward::OffloadForward`] for KIVI-quantized and token-streaming
//! offload KV cache paths respectively.
//!
//! **Phase β-7**: the [`Forward`] trait itself moved here from the deleted
//! `session::traits` (slim internal seam, G2-(ii)). [`StepCtx`] lives in
//! [`crate::inference::sampling`] (shared with [`TokenSampler`]).

pub mod kivi_forward;
pub mod model_forward;
pub mod offload_forward;

pub use kivi_forward::{KiviForward, alloc_kivi_kv_caches};
pub use model_forward::{ModelForward, alloc_standard_kv_caches};
pub use offload_forward::{OffloadForward, alloc_offload_kv_caches};

use crate::inference::sampling::StepCtx;

/// Required forward pass. Provides KV-bearing model semantics.
///
/// `finalize` and `on_kv_prune` are default no-ops so a minimal Forward
/// implementation needs only `prefill` + `step`.
pub trait Forward {
    /// Run prefill over `tokens` starting at KV position `start_pos`.
    /// Returns logits for the last token.
    ///
    /// `start_pos`는 chat multi-turn에서 turn 사이 KV 누적 보존을 위해 필수.
    /// non-chat 모드에선 caller가 0 전달 (단일 prefill).
    fn prefill(&mut self, tokens: &[u32], start_pos: usize) -> anyhow::Result<Vec<f32>>;

    /// Decode 1 step. After return, `pos` is conceptually advanced by 1.
    fn step(&mut self, ctx: &StepCtx, token: u32) -> anyhow::Result<Vec<f32>>;

    /// Called once after [`crate::session::DecodeLoop::run`] exits.
    fn finalize(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Notified after an eviction stage pruned KV state.
    fn on_kv_prune(&mut self, _new_pos: usize) {}

    /// Phase 4-5-d: chat `/reset` 처리용. KV cache를 초기 상태로 reset한다.
    ///
    /// Default no-op — generate 모드는 호출하지 않는다. chat 모드의 각 Forward
    /// 구현체가 override하여 KV-type별 reset 로직을 수행한다.
    fn reset_kv(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Phase 4-5-e: chat `ensure_capacity` / `on_turn_end` 에서 eviction 실행.
    ///
    /// ModelForward만 override한다. KiviForward / OffloadForward는 default no-op
    /// 유지 (eviction 미지원).
    ///
    /// - `cache_manager`: eviction 정책을 보유한 CacheManager.
    /// - `scores`: score-based policy (H2O/D2O)용 importance 점수. `None`이면
    ///   score-free force/maybe evict.
    /// - `force`: true이면 `force_evict*`, false이면 `maybe_evict*`.
    /// - `target_ratio`: `force=true` 시 eviction 목표 비율.
    ///
    /// Returns (removed_count, new_pos). removed_count == 0이면 eviction 미발생.
    fn try_evict(
        &mut self,
        cache_manager: &crate::kv::cache_manager::CacheManager,
        scores: Option<&[f32]>,
        force: bool,
        target_ratio: f32,
    ) -> anyhow::Result<(usize, usize)> {
        let _ = (cache_manager, scores, force, target_ratio);
        Ok((0, 0))
    }

    /// prefix cache save (ENG-085, INV-189). prefill 완료 직후 호출.
    ///
    /// `token_ids` = prompt 토큰 전체 (snapshot 대상). path = 저장 경로.
    /// `last_logits` = prefill이 산출한 마지막 토큰 logits (f32×vocab). full restore 시 재사용.
    /// ModelForward(StandardFormat) 만 override. 비지원 Forward 는 default no-op.
    ///
    /// Save 실패(권한/disk full 등)는 경고 후 무시(run 계속).
    fn save_kv_prefix(
        &self,
        path: &std::path::Path,
        model_hash: &[u8; 32],
        tokenizer_hash: &[u8; 32],
        token_ids: &[u32],
        last_logits: &[f32],
        backend: &dyn crate::backend::Backend,
    ) -> anyhow::Result<()> {
        let _ = (
            path,
            model_hash,
            tokenizer_hash,
            token_ids,
            last_logits,
            backend,
        );
        Ok(()) // default: no-op
    }
}
