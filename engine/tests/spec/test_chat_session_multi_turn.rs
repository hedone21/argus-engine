//! Phase 4-5-d Gate G2/G3/G4 enforce.
//!
//! [`ChatSession`] multi-turn KV 보존 + /reset + ensure_capacity 동작 검증.
//!
//! G2: multi-turn 2nd-turn 시작 시 pos가 1st turn 직후 값에서 누적된다.
//!     (DecodeLoop를 turn마다 build/drop하면 pos가 초기화되어 실패한다.)
//! G3: `/reset` 후 pos == 0 + score_accumulator.evicted_total == 0.
//!     Forward::reset_kv가 호출되어 mock step_count가 0으로 복귀한다.
//! G4: `ensure_capacity`가 Standard(no-cache-manager)/QuantWindow/Offload 각 모드에서
//!     overflow 시 Err를 반환하고, 여유 있으면 Ok를 반환한다.
//!
//! D5/G1: stats_line 포맷이 generate.rs 원본과 byte-identical한지 검증한다.
//!
//! 실제 모델 없이 mock Forward로 모든 게이트를 검증한다.

use argus_engine::session::chat::session::{ChatKvMode, ChatKvModeStandard, ChatSession};
use argus_engine::session::chat::stop_condition::StopCondition;
use argus_engine::session::{DecodeLoopBuilder, Forward, GreedySampler, StepCtx, StopReason};

// ─── Mock Forward ──────────────────────────────────────────────────────────────

/// Simple sequence generator for testing.
/// prefill → logits[0]=1.0, step → logits[step_count % vocab]=1.0.
/// reset_kv → step_count = 0 (호출 여부 추적).
struct MockSeqForward {
    vocab: usize,
    step_count: usize,
    pub reset_kv_calls: usize,
}

impl Forward for MockSeqForward {
    fn prefill(&mut self, _tokens: &[u32], _start_pos: usize) -> anyhow::Result<Vec<f32>> {
        let mut logits = vec![0.0f32; self.vocab];
        logits[0] = 1.0;
        Ok(logits)
    }

    fn step(&mut self, _ctx: &StepCtx, _token: u32) -> anyhow::Result<Vec<f32>> {
        self.step_count += 1;
        let mut logits = vec![0.0f32; self.vocab];
        let target = self.step_count % self.vocab;
        logits[target] = 1.0;
        Ok(logits)
    }

    fn reset_kv(&mut self) -> anyhow::Result<()> {
        self.reset_kv_calls += 1;
        self.step_count = 0;
        Ok(())
    }
}

// ─── Mock StopCondition ────────────────────────────────────────────────────────

struct TokenStop {
    stop_id: u32,
    max_pos: usize,
}

impl StopCondition for TokenStop {
    fn should_stop(&self, sampled: u32, pos: usize) -> bool {
        sampled == self.stop_id || pos >= self.max_pos
    }
}

// ─── ChatSession factory ───────────────────────────────────────────────────────

fn make_standard_session(max_seq_len: usize) -> ChatSession {
    let fwd = MockSeqForward {
        vocab: 16,
        step_count: 0,
        reset_kv_calls: 0,
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

fn make_kivi_session(max_seq_len: usize) -> ChatSession {
    let fwd = MockSeqForward {
        vocab: 8,
        step_count: 0,
        reset_kv_calls: 0,
    };
    let decode_loop = DecodeLoopBuilder::new()
        .with_forward(fwd)
        .with_kv_capacity(max_seq_len)
        .build();
    ChatSession::new_for_test(
        decode_loop,
        ChatKvMode::NonEvicting {
            stats_fragment: "mode=kivi bits=4 residual=32".to_string(),
        },
        max_seq_len,
    )
}

fn make_offload_session(max_seq_len: usize) -> ChatSession {
    let fwd = MockSeqForward {
        vocab: 8,
        step_count: 0,
        reset_kv_calls: 0,
    };
    let decode_loop = DecodeLoopBuilder::new()
        .with_forward(fwd)
        .with_kv_capacity(max_seq_len)
        .build();
    ChatSession::new_for_test(
        decode_loop,
        ChatKvMode::NonEvicting {
            stats_fragment: "mode=offload store=raw prefetch_depth=4".to_string(),
        },
        max_seq_len,
    )
}

// ─── G2: multi-turn KV pos 누적 보존 ─────────────────────────────────────────

/// G2-A: turn 1 prefill 후 pos == prompt.len().
#[test]
fn g2a_pos_equals_prompt_len_after_prefill() {
    let mut session = make_standard_session(2048);
    let prompt = &[1u32, 2, 3, 4, 5];
    session.prefill(prompt).unwrap();
    assert_eq!(session.pos(), prompt.len());
}

/// G2-B: turn 1 decode 후 pos > turn 1 prefill pos.
#[test]
fn g2b_pos_increases_after_decode() {
    let mut session = make_standard_session(2048);
    let prompt = &[1u32, 2];
    session.prefill(prompt).unwrap();
    let pos_after_prefill = session.pos();

    // stop_id=3이면 3 step 후 종료 (step_count=1→tok1, 2→tok2, 3→tok3=stop)
    let stop = TokenStop {
        stop_id: 3,
        max_pos: 2048,
    };
    session.run_turn(0, &stop).unwrap();
    assert!(
        session.pos() > pos_after_prefill,
        "decode 후 pos > prefill pos"
    );
}

/// G2-C: turn 1 decode 후 pos가 > 0이고 ChatSession이 살아있는 동안 유지된다.
///
/// Phase 4-5-e: DecodeLoop::prefill은 `pos += tokens.len()` (누적).
/// turn 2 prefill 후 pos = pos_after_turn1 + prompt2.len().
///
/// R1 보존 검증: turn 1 decode 완료 후 session이 여전히 유효하여 turn 2 prefill을
/// 받을 수 있는지 확인한다 (panic 없이 진행되어야 함).
#[test]
fn g2c_session_survives_multi_turn() {
    let mut session = make_standard_session(2048);
    let prompt1 = &[1u32, 2, 3];
    session.prefill(prompt1).unwrap();
    let stop = TokenStop {
        stop_id: 3,
        max_pos: 2048,
    };
    session.run_turn(0, &stop).unwrap();
    let pos_after_turn1 = session.pos();
    assert!(pos_after_turn1 > 0, "turn 1 decode 후 pos > 0");

    // R1 검증: ChatSession이 drop되지 않고 turn 2 prefill을 받을 수 있다.
    let prompt2 = &[10u32, 11];
    let result = session.prefill(prompt2);
    assert!(
        result.is_ok(),
        "R1: ChatSession이 turn 사이 살아있어 turn 2 prefill 가능"
    );
    // prefill 후 pos = pos_after_turn1 + prompt2.len() (DecodeLoop::prefill 누적).
    let expected = pos_after_turn1 + prompt2.len();
    assert_eq!(
        session.pos(),
        expected,
        "DecodeLoop::prefill accumulates pos (multi-turn): expected {}",
        expected
    );
}

/// G2-D: run_turn이 StopConditionMet로 종료된다 (finalize 미호출 보장 확인).
#[test]
fn g2d_run_turn_stops_on_condition() {
    let mut session = make_standard_session(2048);
    session.prefill(&[0u32]).unwrap();
    let stop = TokenStop {
        stop_id: 2,
        max_pos: 2048,
    };
    let result = session.run_turn(0, &stop).unwrap();
    assert_eq!(
        result.stopped_by,
        StopReason::StopConditionMet,
        "run_turn은 StopConditionMet으로 종료"
    );
}

// ─── G3: /reset 동작 ──────────────────────────────────────────────────────────

/// G3-A: reset 후 pos == 0.
#[test]
fn g3a_reset_clears_pos() {
    let mut session = make_standard_session(2048);
    session.prefill(&[1u32, 2, 3]).unwrap();
    let stop = TokenStop {
        stop_id: 99,
        max_pos: 8,
    };
    session.run_turn(0, &stop).unwrap();
    assert!(session.pos() > 0);

    session.reset().unwrap();
    assert_eq!(session.pos(), 0, "reset 후 pos == 0");
}

/// G3-B: Standard 모드에서 reset 후 evicted_total == 0.
#[test]
fn g3b_reset_clears_evicted_total() {
    let mut session = make_standard_session(2048);

    // evicted_total 수동 설정
    if let ChatKvMode::Standard(s) = &mut session.kv_mode {
        s.evicted_total = 99;
    }

    session.reset().unwrap();

    if let ChatKvMode::Standard(s) = &session.kv_mode {
        assert_eq!(s.evicted_total, 0, "reset 후 evicted_total == 0");
    } else {
        panic!("expected Standard mode");
    }
}

/// G3-C: reset 후 decode 재개 시 pos가 0에서 다시 증가한다.
#[test]
fn g3c_decode_restarts_after_reset() {
    let mut session = make_standard_session(2048);
    session.prefill(&[1u32, 2]).unwrap();
    let stop = TokenStop {
        stop_id: 99,
        max_pos: 5,
    };
    session.run_turn(0, &stop).unwrap();

    session.reset().unwrap();
    assert_eq!(session.pos(), 0);

    // reset 후 새 세션처럼 prefill 가능
    session.prefill(&[5u32, 6]).unwrap();
    assert_eq!(session.pos(), 2, "reset 후 prefill → pos == prompt.len()");
}

// ─── G4: ensure_capacity 분기 ─────────────────────────────────────────────────

/// G4-A: Standard(no cache_manager), pos+additional <= max_seq_len → Ok.
#[test]
fn g4a_standard_ok_when_capacity_sufficient() {
    let mut session = make_standard_session(10);
    // 5 토큰 prefill → 점유=5 (Standard capacity 는 누적 pos 가 아니라 점유로 회계).
    session.prefill(&[1u32, 2, 3, 4, 5]).unwrap();
    assert!(
        session.ensure_capacity(3).is_ok(),
        "점유=5, add=3, max=10 → Ok"
    );
}

/// G4-B: Standard(no cache_manager), pos+additional > max_seq_len → Err.
#[test]
fn g4b_standard_bails_on_overflow_without_cache_manager() {
    let mut session = make_standard_session(10);
    // 9 토큰 prefill → 점유=9 (Standard capacity 는 점유로 회계).
    session.prefill(&[1u32, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
    assert!(
        session.ensure_capacity(2).is_err(),
        "점유=9, add=2, max=10 → Err"
    );
}

/// G4-C: QuantWindow, pos+additional <= max_seq_len → Ok.
#[test]
fn g4c_kivi_ok_when_sufficient() {
    let mut session = make_kivi_session(10);
    session.pos = 5;
    assert!(
        session.ensure_capacity(4).is_ok(),
        "pos=5, add=4, max=10 → Ok"
    );
}

/// G4-D: QuantWindow, pos+additional > max_seq_len → Err.
#[test]
fn g4d_kivi_bails_on_overflow() {
    let mut session = make_kivi_session(10);
    session.pos = 9;
    assert!(session.ensure_capacity(2).is_err(), "Kivi overflow → Err");
}

/// G4-E: Offload, pos+additional > max_seq_len → Err.
#[test]
fn g4e_offload_bails_on_overflow() {
    let mut session = make_offload_session(10);
    session.pos = 9;
    assert!(
        session.ensure_capacity(2).is_err(),
        "Offload overflow → Err"
    );
}

/// G4-F: Offload, pos+additional == max_seq_len → Err (경계값: > 비교이므로 exact는 Ok).
#[test]
fn g4f_offload_exact_boundary_is_ok() {
    let mut session = make_offload_session(10);
    session.pos = 8;
    // 8 + 2 = 10, 10 > 10은 false → Ok
    assert!(
        session.ensure_capacity(2).is_ok(),
        "pos+additional == max_seq_len → Ok (> 아님)"
    );
}

// ─── D5/G1: stats_line 포맷 보존 ─────────────────────────────────────────────

/// G1-A: Standard 모드 stats_line 포맷 byte-identical.
#[test]
fn g1a_standard_stats_line_format() {
    let mut session = make_standard_session(2048);
    session.pos = 42;
    if let ChatKvMode::Standard(s) = &mut session.kv_mode {
        s.evicted_total = 10;
        s.policy_name = "sliding".to_string();
    }
    assert_eq!(
        session.stats_line(),
        "kv_pos=42/2048 policy=sliding evicted_total=10",
        "Standard stats_line 포맷 불일치"
    );
}

/// G1-B: QuantWindow 모드 stats_line 포맷 byte-identical.
#[test]
fn g1b_kivi_stats_line_format() {
    let mut session = make_kivi_session(512);
    session.pos = 100;
    assert_eq!(
        session.stats_line(),
        "kv_pos=100/512 mode=kivi bits=4 residual=32",
        "Kivi stats_line 포맷 불일치"
    );
}

/// G1-C: Offload 모드 stats_line 포맷 byte-identical.
#[test]
fn g1c_offload_stats_line_format() {
    let mut session = make_offload_session(512);
    session.pos = 77;
    assert_eq!(
        session.stats_line(),
        "kv_pos=77/512 mode=offload store=raw prefetch_depth=4",
        "Offload stats_line 포맷 불일치"
    );
}

/// G1-D: Standard eviction=none stats_line (기본값 확인).
#[test]
fn g1d_standard_none_policy_stats_line() {
    let session = make_standard_session(1024);
    assert_eq!(
        session.stats_line(),
        "kv_pos=0/1024 policy=none evicted_total=0",
    );
}
