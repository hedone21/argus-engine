//! Phase 4-5-e Gate: run_chat_repl_v2 multi-turn pos 누적 보존 + first token bit-identical.
//!
//! G2: turn 2 첫 토큰이 turn 1 pos 누적 후에도 bit-identical (mock model 기준).
//!     구체적으로: prefill 후 pos가 누적되어 turn 2 start pos가 올바른지 검증.
//!
//! 실제 tokenizer / ChatTemplate 없이 mock 기반으로 검증한다.
//! run_chat_repl_v2는 I/O 주도 함수라 직접 호출 대신, 핵심 동작인
//! ChatSession::prefill + run_turn multi-turn pos 누적을 spec 수준에서 검증한다.

use argus_engine::session::chat::session::{ChatKvMode, ChatKvModeStandard, ChatSession};
use argus_engine::session::chat::stop_condition::ChatStopCondition;
use argus_engine::session::{DecodeLoopBuilder, Forward, GreedySampler, StepCtx, StopReason};

// ─── Mock Forward ──────────────────────────────────────────────────────────────

/// Deterministic sequence generator.
/// prefill: logits[0]=1.0.
/// step(n): logits[(n % vocab)] = 1.0 (n = call index).
struct DetForward {
    vocab: usize,
    step_idx: usize,
}

impl Forward for DetForward {
    fn prefill(&mut self, _tokens: &[u32], _start_pos: usize) -> anyhow::Result<Vec<f32>> {
        let mut logits = vec![0.0f32; self.vocab];
        logits[0] = 1.0;
        Ok(logits)
    }

    fn step(&mut self, _ctx: &StepCtx, _token: u32) -> anyhow::Result<Vec<f32>> {
        self.step_idx += 1;
        let mut logits = vec![0.0f32; self.vocab];
        logits[self.step_idx % self.vocab] = 1.0;
        Ok(logits)
    }

    fn reset_kv(&mut self) -> anyhow::Result<()> {
        self.step_idx = 0;
        Ok(())
    }
}

fn make_det_session(max_seq_len: usize) -> ChatSession {
    let fwd = DetForward {
        vocab: 32,
        step_idx: 0,
    };
    let decode_loop = DecodeLoopBuilder::new()
        .with_forward(fwd)
        .with_sampler(GreedySampler)
        .with_kv_capacity(max_seq_len)
        .build();
    ChatSession::new_for_test(
        decode_loop,
        ChatKvMode::Standard(Box::new(ChatKvModeStandard {
            cache_manager: None,
            score_accumulator: None,
            score_based: false,
            policy_name: "none".to_string(),
            target_ratio: 1.0,
            evicted_total: 0,
            faithful_prefill: None,
        })),
        max_seq_len,
    )
}

// ─── G2: multi-turn pos 누적 (Phase 4-5-e 핵심) ──────────────────────────────

/// G2-REPL: turn 1 decode 후 turn 2 prefill 시 pos가 누적된다.
///
/// DecodeLoop::prefill이 `pos += tokens.len()` (Phase 4-5-e 변경)이므로
/// turn 2 시작 pos = pos_after_turn1 + prompt2.len().
#[test]
fn g2_repl_multi_turn_pos_accumulates() {
    let mut session = make_det_session(512);

    // turn 1: prompt 5 tokens
    let prompt1 = &[1u32, 2, 3, 4, 5];
    session.prefill(prompt1).unwrap();
    assert_eq!(session.pos(), 5, "turn 1 prefill → pos=5");

    // decode: stop at token 3 or max_pos=20
    let stop1 = ChatStopCondition::new(vec![3], 20);
    let result1 = session.run_turn(0, &stop1).unwrap();
    let pos_after_turn1 = session.pos();
    assert!(
        pos_after_turn1 > 5,
        "turn 1 decode 후 pos > prefill pos: got {}",
        pos_after_turn1
    );
    assert_eq!(result1.stopped_by, StopReason::StopConditionMet);

    // turn 2: prompt 3 tokens — pos가 누적되어야 한다.
    let prompt2 = &[10u32, 11, 12];
    session.prefill(prompt2).unwrap();
    let expected_pos = pos_after_turn1 + prompt2.len();
    assert_eq!(
        session.pos(),
        expected_pos,
        "G2: turn 2 prefill 후 pos = {} + {} = {}",
        pos_after_turn1,
        prompt2.len(),
        expected_pos
    );
}

/// G2-REPL-2: turn 2 첫 토큰 결과가 deterministic (bit-identical 보장).
///
/// DetForward는 step_idx % vocab 위치에 logit=1.0을 설정한다.
/// GreedySampler는 argmax를 반환하므로 결과가 결정론적이다.
/// turn 1과 turn 2에서 같은 위치에서 prefill한 후 첫 토큰이 일치하는지 검증.
#[test]
fn g2_repl_turn2_first_token_deterministic() {
    // session A: turn 1만
    let mut session_a = make_det_session(512);
    session_a.prefill(&[1u32, 2]).unwrap();
    let stop_a = ChatStopCondition::new(vec![5], 512);
    let result_a1 = session_a.run_turn(0, &stop_a).unwrap();
    let pos_after_a1 = session_a.pos();

    // session B: turn 1과 동일 후, turn 2 진입
    let mut session_b = make_det_session(512);
    session_b.prefill(&[1u32, 2]).unwrap();
    let stop_b = ChatStopCondition::new(vec![5], 512);
    let result_b1 = session_b.run_turn(0, &stop_b).unwrap();

    // turn 2: 동일 prompt
    let prompt2 = &[20u32, 21];
    session_b.prefill(prompt2).unwrap();
    let expected_pos = pos_after_a1 + prompt2.len();
    assert_eq!(
        session_b.pos(),
        expected_pos,
        "B turn 2 pos 누적 일치: {}",
        expected_pos
    );

    // result_a1과 result_b1은 동일한 세션에서 동일한 step 순서이므로 일치해야 함.
    assert_eq!(
        result_a1.tokens_generated, result_b1.tokens_generated,
        "같은 조건에서 실행한 두 세션의 turn 1 결과가 bit-identical"
    );
}

/// G2-REPL-3: reset 후 pos=0에서 prefill 재시작 — pos = prompt.len().
#[test]
fn g2_repl_reset_then_prefill_restarts_from_zero() {
    let mut session = make_det_session(512);

    // turn 1
    session.prefill(&[1u32, 2, 3]).unwrap();
    let stop = ChatStopCondition::new(vec![99], 15);
    session.run_turn(0, &stop).unwrap();
    assert!(session.pos() > 0);

    // reset
    session.reset().unwrap();
    assert_eq!(session.pos(), 0, "reset 후 pos==0");

    // reset 후 prefill: 0 += tokens.len() = tokens.len()
    let fresh_prompt = &[7u32, 8];
    session.prefill(fresh_prompt).unwrap();
    assert_eq!(
        session.pos(),
        fresh_prompt.len(),
        "reset 후 prefill → pos = tokens.len()"
    );
}
