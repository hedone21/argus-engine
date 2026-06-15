//! Per-request configurable token sampler for the chat server.
//!
//! Unlike [`RepetitionPenaltySampler`](crate::session::RepetitionPenaltySampler),
//! whose [`SamplingConfig`] is fixed at construction, [`ChatSampler`] reads its
//! config from a shared `Arc<Mutex<SamplingConfig>>` on every `sample()` call.
//! The chat server owns the other end of that handle and overwrites it with the
//! per-request `temperature` / `top_p` before each generation
//! ([`ChatSession::set_sampling`](crate::session::chat::session::ChatSession::set_sampling)).
//! This lets one persistent decode loop honor a different sampling config per
//! OpenAI request without rebuilding the model forward.
//!
//! `reset()` clears the repetition ring buffer — the stateless chat server calls
//! it (via `ChatSession::reset`) so a request does not inherit the previous
//! request's token history.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::inference::sampling::{SamplingConfig, StepCtx, TokenSampler, sample};

/// Shared, per-request-mutable sampling config handle.
pub type SharedSamplingConfig = Arc<Mutex<SamplingConfig>>;

/// Stateful sampler that reads a shared [`SamplingConfig`] each step. Mirrors
/// [`RepetitionPenaltySampler`](crate::session::RepetitionPenaltySampler) but
/// with a runtime-swappable config + a `reset()` that drops the ring buffer.
pub struct ChatSampler {
    config: SharedSamplingConfig,
    vocab_size: usize,
    recent: VecDeque<u32>,
    scratch_logits: Vec<f32>,
    scratch_indices: Vec<usize>,
}

impl ChatSampler {
    pub fn new(config: SharedSamplingConfig, vocab_size: usize) -> Self {
        Self {
            config,
            vocab_size,
            recent: VecDeque::new(),
            scratch_logits: Vec::with_capacity(vocab_size),
            scratch_indices: Vec::with_capacity(vocab_size),
        }
    }

    fn window(&self) -> usize {
        self.config
            .lock()
            .map(|c| c.repetition_window)
            .unwrap_or(0)
            .max(1)
    }
}

impl TokenSampler for ChatSampler {
    fn sample(&mut self, _ctx: &StepCtx, logits: &[f32]) -> u32 {
        // Snapshot the shared config so we hold the lock only briefly.
        let cfg = self
            .config
            .lock()
            .expect("ChatSampler config mutex poisoned")
            .clone();
        self.scratch_logits.clear();
        self.scratch_logits.extend_from_slice(logits);
        let recent_slice = self.recent.make_contiguous();
        sample(
            &mut self.scratch_logits,
            recent_slice,
            self.vocab_size,
            &cfg,
            Some(&mut self.scratch_indices),
        )
    }

    fn observe_token(&mut self, token: u32) {
        let window = self.window();
        while self.recent.len() >= window {
            self.recent.pop_front();
        }
        self.recent.push_back(token);
    }

    fn reset(&mut self) {
        self.recent.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn cfg(temp: f32) -> SamplingConfig {
        SamplingConfig {
            temperature: temp,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_window: 4,
        }
    }

    fn ctx() -> (StepCtx<'static>, &'static AtomicBool) {
        // leak a stop flag for the 'static StepCtx in tests.
        let flag: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
        (
            StepCtx {
                pos: 0,
                prev_token: 0,
                kv_capacity: 0,
                decode_step: 0,
                stop_requested: flag,
            },
            flag,
        )
    }

    #[test]
    fn greedy_when_temperature_zero() {
        let shared = Arc::new(Mutex::new(cfg(0.0)));
        let mut s = ChatSampler::new(Arc::clone(&shared), 4);
        let (c, _) = ctx();
        // logits[2] is max → greedy picks 2.
        assert_eq!(s.sample(&c, &[0.1, 0.2, 0.9, 0.3]), 2);
    }

    #[test]
    fn reset_clears_ring() {
        let shared = Arc::new(Mutex::new(cfg(0.0)));
        let mut s = ChatSampler::new(shared, 4);
        s.observe_token(1);
        s.observe_token(2);
        s.reset();
        assert!(s.recent.is_empty());
    }

    #[test]
    fn config_swap_is_observed() {
        let shared = Arc::new(Mutex::new(cfg(0.0)));
        let mut s = ChatSampler::new(Arc::clone(&shared), 4);
        let (c, _) = ctx();
        // temperature 0 → deterministic greedy.
        assert_eq!(s.sample(&c, &[0.1, 0.9, 0.2, 0.3]), 1);
        // swap to a high temperature: still returns a valid token id < vocab.
        *shared.lock().unwrap() = cfg(1.0);
        let t = s.sample(&c, &[0.1, 0.9, 0.2, 0.3]);
        assert!(t < 4);
    }
}
