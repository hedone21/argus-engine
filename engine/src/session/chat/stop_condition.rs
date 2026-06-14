//! Chat REPL의 inner decode loop 종료 조건 trait + helper.
//! Phase 4-5-c. chat REPL은 4-5-e에서 ChatStopCondition을 owned 보유하여
//! DecodeLoop::run_until_stop에 전달.
//!
//! **β-6**: stop 판정은 [`ChatStopStage`] (L2 `PipelineStage`) 로 수렴한다 — `DecodeEnd` phase
//! 에서 슬롯의 stop condition 으로 `should_stop` 을 평가해 `Stop(StopConditionMet)` 를 반환한다.

use std::sync::{Arc, Mutex};

use crate::pipeline::{
    LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome,
    StopReason as StageStopReason,
};
use crate::session::chat_template::ChatTemplate;
use tokenizers::Tokenizer;

/// Decode loop 종료 조건. `should_stop`이 true를 반환하면 loop를 즉시 종료한다.
///
/// Stop token 체크 위치: sampled 직후 호출. true이면 즉시 break하므로
/// stop token은 KV에 baking되지 않는다. chat REPL에서 EOT는 별도 decode_step으로
/// baking(4-5-e에서 처리).
pub trait StopCondition {
    /// `sampled` token이 방금 emit되었고 KV pos가 `pos`인 시점에 호출.
    /// true 반환 시 decode loop 종료. sampled token은 KV에 baking되지 않았음.
    fn should_stop(&self, sampled: u32, pos: usize) -> bool;
}

/// Chat 모드 전용 stop condition.
/// - stop_ids 중 하나가 sampled되면 true
/// - pos가 max_pos 이상이면 true (overflow 안전망)
pub struct ChatStopCondition {
    stop_ids: Vec<u32>,
    max_pos: usize,
}

impl ChatStopCondition {
    /// `stop_ids`는 내부에서 sort + dedup하므로 순서 무관.
    pub fn new(stop_ids: Vec<u32>, max_pos: usize) -> Self {
        let mut ids = stop_ids;
        ids.sort_unstable();
        ids.dedup();
        Self {
            stop_ids: ids,
            max_pos,
        }
    }
}

impl StopCondition for ChatStopCondition {
    fn should_stop(&self, sampled: u32, pos: usize) -> bool {
        if pos >= self.max_pos {
            return true;
        }
        self.stop_ids.binary_search(&sampled).is_ok()
    }
}

// ─── β-6: ChatStopStage (L2 PipelineStage 수렴) ─────────────────────────────────

/// turn별 동적 stop condition 을 [`ChatStopStage`] 에 전달하기 위한 공유 슬롯.
///
/// `ChatSession::run_turn` 은 매 turn 다른 stop condition(예: `max_pos = pos + max_new_tokens`)을
/// `&dyn StopCondition` 으로 받는다 — 빌드 시점 고정이 불가하다. `ChatStopStage` 는 영구
/// (Persistent) 등록되고, 슬롯에 들어 있는 현재 turn 의 stop condition 으로 `DecodeEnd` phase
/// 마다 판정한다. `ChatSession` 과 stage 가 동일 `Arc<ChatStopSlot>` 을 공유한다.
///
/// 슬롯은 **borrowed** stop pointer 를 보유한다(`*const dyn StopCondition`). `run_turn` 이
/// 동기적으로 driver 를 실행하는 동안에만 set 되며(RAII [`ChatStopGuard`]), driver 가 단일
/// 스레드라 stop 의 수명이 run 구간을 덮음이 보장된다(`INV-018` 단일 스레드 추론).
#[derive(Default)]
pub struct ChatStopSlot {
    /// 현재 turn 의 stop condition raw pointer. `None` 이면 stop 미판정(항상 진행).
    cond: Mutex<Option<*const (dyn StopCondition + 'static)>>,
}

// SAFETY: `cond` 의 raw pointer 는 `ChatStopGuard` 수명(= `run_turn` 의 동기 driver 실행 구간)
// 동안에만 set 되고, 그 사이 단일 스레드(INV-018)만 접근한다. pointee(`dyn StopCondition`)는
// `should_stop(&self)` 만 호출하는 `Sync` 한 read-only trait object 다. Mutex 가 내부 가변성을
// 직렬화하므로 `ChatStopSlot` 을 `Send + Sync` 로 노출해도 안전하다.
unsafe impl Send for ChatStopSlot {}
unsafe impl Sync for ChatStopSlot {}

impl ChatStopSlot {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// turn 시작 시 stop condition 을 슬롯에 set 하고 RAII guard 를 반환한다. guard drop 시
    /// 슬롯이 자동 clear 되어 dangling pointer 가 남지 않는다.
    ///
    /// # Safety 계약
    /// `stop` 은 반환된 guard 가 drop 될 때까지 살아 있어야 한다(`run_turn` 의 stack-local 이
    /// driver 동기 실행을 덮음). guard 는 `run_until_stop` 반환 직후 drop 된다.
    pub fn arm<'g>(self: &'g Arc<Self>, stop: &'g dyn StopCondition) -> ChatStopGuard<'g> {
        // lifetime erasure: `&'g dyn StopCondition` 의 pointee lifetime `'g` 를 `'static` 로
        // 지운다. SAFETY: raw pointer 는 반환된 `ChatStopGuard<'g>` 가 drop 될 때(= `run_turn` 의
        // 동기 driver 실행 종료) 슬롯에서 제거되므로, dereference 는 `'g` 가 유효한 구간 안에서만
        // 일어난다 — `'static` erasure 가 실제 수명을 넘어선 접근을 만들지 않는다.
        let ptr: *const (dyn StopCondition + 'static) =
            unsafe { std::mem::transmute::<*const dyn StopCondition, _>(stop) };
        *self.cond.lock().expect("ChatStopSlot mutex poisoned") = Some(ptr);
        ChatStopGuard { slot: self }
    }

    /// 슬롯의 현재 stop condition 으로 판정. 슬롯 비어 있으면 `false`(미종료).
    fn should_stop(&self, sampled: u32, pos: usize) -> bool {
        let guard = self.cond.lock().expect("ChatStopSlot mutex poisoned");
        match *guard {
            // SAFETY: ptr 은 `arm` 의 guard 수명 동안에만 set 되며, 그 사이 호출되므로 유효하다.
            Some(ptr) => unsafe { (*ptr).should_stop(sampled, pos) },
            None => false,
        }
    }
}

/// RAII guard — drop 시 [`ChatStopSlot`] 의 stop pointer 를 clear 한다.
pub struct ChatStopGuard<'g> {
    slot: &'g Arc<ChatStopSlot>,
}

impl Drop for ChatStopGuard<'_> {
    fn drop(&mut self) {
        *self.slot.cond.lock().expect("ChatStopSlot mutex poisoned") = None;
    }
}

/// chat stop 판정 `PipelineStage`. `DecodeEnd` phase 에서 슬롯의 stop condition 으로
/// `should_stop(prev_token, pos)` 를 평가한다(충족 시 `Stop(StopConditionMet)`).
///
/// **DecodeEnd 구독 (roadmap 정정)**: roadmap 본문은 "PostSample 구독" 이라 썼으나
/// 오케스트레이터 census 로 **DecodeEnd 로 정정**한다. v1 `run_until_stop` 은 stop 토큰에도
/// (f2) tick / (g) observer 를 발화하고 pos 를 증가시킨 **뒤** `should_stop(sampled, self.pos)`
/// 를 평가한다. driver 의 DecodeEnd 는 bookkeeping(`prev_token=sampled; pos+=1; decode_step+=1`)
/// **후** 발화하므로 `ctx.step.prev_token == 방금 sampled`, `ctx.step.pos == 증가된 pos` 가
/// v1 의 `should_stop(sampled, self.pos)` 인자와 정확히 일치한다. PostSample 구독이면 stop 토큰의
/// tick/obs/pos++ 가 누락돼 chat spec(final_pos 단언)이 깨진다.
pub struct ChatStopStage {
    slot: Arc<ChatStopSlot>,
}

impl ChatStopStage {
    pub fn new(slot: Arc<ChatStopSlot>) -> Self {
        Self { slot }
    }
}

impl PipelineStage for ChatStopStage {
    fn name(&self) -> &str {
        "chat.stop"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::Persistent
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.3): DecodeEnd 외 phase 는 무시.
        if *phase != LifecyclePhase::DecodeEnd {
            return Ok(StageOutcome::Continue);
        }
        // DecodeEnd: prev_token == 방금 sampled, pos == 증가된 pos (v1 should_stop 인자 등가).
        if self.slot.should_stop(ctx.step.prev_token, ctx.step.pos) {
            Ok(StageOutcome::Stop(StageStopReason::StopConditionMet))
        } else {
            Ok(StageOutcome::Continue)
        }
    }
}

/// 토크나이저로 literal token들을 토큰 ID로 변환.
///
/// `required=true`인데 literal에 대한 ID를 찾지 못하면 Err.
/// `required=false`인 경우 없는 literal은 조용히 스킵.
///
/// generate.rs의 `resolve_token_ids` (l.9805~9824)와 동일 시그니처 — 4-5-f에서
/// generate.rs의 사본은 삭제하고 이 버전을 사용하게 된다.
pub fn resolve_token_ids(
    tokenizer: &Tokenizer,
    literals: &[&'static str],
    required: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut out = Vec::with_capacity(literals.len());
    for lit in literals {
        match tokenizer.token_to_id(lit) {
            Some(id) => out.push(id),
            None if required => {
                anyhow::bail!(
                    "tokenizer is missing required special token `{}`. \
                     Make sure tokenizer.json has it registered as an added_token.",
                    lit
                );
            }
            None => {}
        }
    }
    Ok(out)
}

/// Chat stop_ids 빌드 헬퍼.
///
/// `template.stop_token_literals()` + `eos_token_id`를 합산하여
/// sorted dedup Vec<u32>를 반환한다.
///
/// generate.rs:9870~9881의 stop_ids 빌드 로직을 이관.
pub fn build_chat_stop_ids(
    template: &ChatTemplate,
    tokenizer: &Tokenizer,
    eos_token_id: u32,
) -> anyhow::Result<Vec<u32>> {
    let lits = template.stop_token_literals();
    if lits.is_empty() {
        anyhow::bail!("chat template has no stop token literals");
    }
    let mut ids = resolve_token_ids(tokenizer, &lits[..1], true)?;
    ids.extend(resolve_token_ids(tokenizer, &lits[1..], false)?);
    ids.push(eos_token_id);
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_stop_returns_true_on_stop_id_match() {
        let cond = ChatStopCondition::new(vec![128009, 128001], 2048);
        assert!(cond.should_stop(128009, 10));
        assert!(cond.should_stop(128001, 10));
        assert!(!cond.should_stop(42, 10));
    }

    #[test]
    fn should_stop_returns_true_on_max_pos_overflow() {
        let cond = ChatStopCondition::new(vec![128009], 100);
        // pos < max_pos, non-stop token → false
        assert!(!cond.should_stop(1, 99));
        // pos == max_pos → true (안전망)
        assert!(cond.should_stop(1, 100));
        // pos > max_pos → true
        assert!(cond.should_stop(1, 200));
    }

    #[test]
    fn resolve_token_ids_errors_when_required_but_empty() {
        // tokenizers::Tokenizer::from_pretrained은 네트워크 없이 사용 불가.
        // 더미 vocab JSON으로 Tokenizer를 직접 생성하여 검증.
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": {"hello": 0, "world": 1},
                "merges": [],
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false
            }
        }"#;
        use std::str::FromStr;
        let tokenizer = Tokenizer::from_str(json).expect("tokenizer parse");

        // required=true이고 토큰이 없으면 Err
        let result = resolve_token_ids(&tokenizer, &["<|eot_id|>"], true);
        assert!(result.is_err(), "expected Err for missing required token");

        // required=false이면 스킵하고 Ok
        let result = resolve_token_ids(&tokenizer, &["<|eot_id|>"], false);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
