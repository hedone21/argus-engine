//! Phase 4-4.7: production `sampling::sample` 호출과 결과 동치를 보장하는
//! stateful [`TokenSampler`].
//!
//! [`observe_token`]을 통해 최근 토큰을 [`VecDeque`] ring buffer로 보유하고,
//! [`sample`] 호출 시 logits을 scratch 버퍼에 복사한 뒤
//! [`crate::inference::sampling::sample`]에 위임한다. logits in-place 수정으로부터
//! 외부 호출자(`DecodeLoop`)가 받은 슬라이스를 보호하기 위해 clone 필수.
//!
//! **호출 규약**: caller는 prefill 직후 prompt history seed (또는 첫 토큰
//! `observe_token`)를 통해 ring buffer를 채워야 production fallback
//! (generate.rs `sampling::sample(&mut logits, &tokens, ...)`)과 paradigm
//! equivalence를 가진다. [`crate::session::DecodeLoop::prefill`]에서 prompt
//! 전체를 자동 seed.
//!
//! `repetition_window` 입력이 0이면 1로 클램프 (방어).

use std::collections::VecDeque;

use crate::inference::sampling::{SamplingConfig, StepCtx, TokenSampler, sample};

/// Production `sampling::sample` 호출과 동치 결과를 내는 stateful sampler.
///
/// `recent` ring buffer는 최대 `config.repetition_window` 개 토큰을 보유.
/// `scratch_logits` / `scratch_indices`는 매 step 재사용 (steady state에서
/// heap allocation 0).
pub struct RepetitionPenaltySampler {
    config: SamplingConfig,
    vocab_size: usize,
    recent: VecDeque<u32>,
    scratch_logits: Vec<f32>,
    scratch_indices: Vec<usize>,
}

impl RepetitionPenaltySampler {
    pub fn new(config: SamplingConfig, vocab_size: usize) -> Self {
        let window = config.repetition_window.max(1);
        Self {
            recent: VecDeque::with_capacity(window),
            scratch_logits: Vec::with_capacity(vocab_size),
            scratch_indices: Vec::with_capacity(vocab_size),
            vocab_size,
            config,
        }
    }

    /// 외부 시드용 헬퍼 (테스트 + 명시적 caller 공용). `tokens` 전체를 push —
    /// `sample()` 내부에서 window slice가 자동 적용된다.
    pub fn seed_history(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.observe_token(t);
        }
    }

    /// 현재 ring buffer 길이 (테스트/검증용).
    pub fn history_len(&self) -> usize {
        self.recent.len()
    }
}

impl TokenSampler for RepetitionPenaltySampler {
    fn sample(&mut self, _ctx: &StepCtx, logits: &[f32]) -> u32 {
        self.scratch_logits.clear();
        self.scratch_logits.extend_from_slice(logits);
        let recent_slice = self.recent.make_contiguous();
        sample(
            &mut self.scratch_logits,
            recent_slice,
            self.vocab_size,
            &self.config,
            Some(&mut self.scratch_indices),
        )
    }

    fn observe_token(&mut self, token: u32) {
        let window = self.config.repetition_window.max(1);
        if self.recent.len() == window {
            self.recent.pop_front();
        }
        self.recent.push_back(token);
    }
}
