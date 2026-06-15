//! Chat REPL v2 — [`ChatSession`] 기반 multi-turn loop (Phase 4-5-e).
//!
//! [`run_chat_repl_v2`]는 generate.rs::run_chat_repl (l.9855~10053)의 로직을
//! [`ChatSession`]을 사용하여 재작성한 버전이다.
//!
//! 주요 차이점:
//! - `ChatTurnExec` trait 대신 `ChatSession` API (prefill/run_turn/ensure_capacity)
//! - decode inner loop는 `DecodeLoop::run_until_stop`에 위임
//! - turn 출력은 서버 SSE 경로와 **동일하게** per-token streaming — `session.stream_slot()`
//!   을 stdout/소켓 콜백으로 arm 하고 [`IncDetok`] 로 증분 detok 한다(byte 경계 안전).
//! - eviction은 ChatSession::ensure_capacity + on_turn_end에 내장

use std::collections::VecDeque;
use std::io::Write as _;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::inference::sampling::{self, SamplingConfig};
use crate::model_config::ModelArch;
use crate::session::chat::session::ChatSession;
use crate::session::chat::stop_condition::{ChatStopCondition, build_chat_stop_ids};
use crate::session::chat::stream_stage::IncDetok;
use crate::session::chat_ipc::{
    ChatInput, finish_reply_stream, spawn_chat_input_sources, write_reply_bytes,
};
use crate::session::chat_template::ChatTemplate;

/// [`run_chat_repl_v2`]에 전달하는 인자 struct.
///
/// generate.rs::run_chat_repl의 개별 파라미터들을 한 struct로 묶었다.
pub struct ChatReplArgs<'a> {
    pub model_arch: ModelArch,
    pub tokenizer: &'a Tokenizer,
    pub eos_token_id: u32,
    pub vocab_size: usize,
    pub sampling_config: &'a SamplingConfig,
    pub max_seq_len: usize,
    pub system_prompt: Option<&'a str>,
    /// `--prompt` 값. 첫 번째 user turn으로 사용.
    pub initial_user_prompt: Option<&'a str>,
    /// Unix domain socket 경로 (generate.rs `--chat-socket`).
    pub chat_socket: Option<&'a str>,
    /// TCP 주소 (generate.rs `--chat-tcp`).
    pub chat_tcp: Option<&'a str>,
    /// sampling용 repetition window 크기.
    pub repetition_window: usize,
    /// 턴당 최대 생성 토큰 수.
    pub max_new_tokens: usize,
}

/// rep-penalty 용 recent ring 갱신 — `window` 초과 시 oldest pop.
/// (스트림 콜백과 first_tok 경로가 공유하므로 free fn 으로 분리 — 동시 `&mut recent` 차용 회피.)
fn push_recent(recent: &mut VecDeque<u32>, tok: u32, window: usize) {
    recent.push_back(tok);
    if recent.len() > window {
        recent.pop_front();
    }
}

/// ChatSession 기반 chat REPL 루프.
///
/// generate.rs::run_chat_repl (l.9855~10053)의 동치 구현.
/// `session`은 caller가 미리 build해서 전달한다 (R1: turn 사이 drop 금지).
pub fn run_chat_repl_v2(args: &ChatReplArgs<'_>, session: &mut ChatSession) -> Result<()> {
    let template = ChatTemplate::new(args.model_arch)?;
    let stop_ids = build_chat_stop_ids(&template, args.tokenizer, args.eos_token_id)?;
    let assistant_eot_ids: Vec<u32> = args
        .tokenizer
        .encode(template.assistant_eot(), false)
        .map_err(|e| anyhow::anyhow!("encode EOT: {}", e))?
        .get_ids()
        .to_vec();
    let bos_id = if template.bos_needed_on_first_prefill() {
        template
            .bos_literal()
            .and_then(|lit| args.tokenizer.token_to_id(lit))
    } else {
        None
    };

    // ── system prompt prefill (1회, KV에 영구 기록) ───────────────────────
    if let Some(sys) = args.system_prompt {
        let rendered = template.render_system(sys);
        let mut ids = args
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode system: {}", e))?
            .get_ids()
            .to_vec();
        if let Some(b) = bos_id {
            ids.insert(0, b);
        }
        if ids.len() > args.max_seq_len {
            anyhow::bail!(
                "system prompt produces {} tokens, exceeds max_seq_len={}",
                ids.len(),
                args.max_seq_len
            );
        }
        let _ = session.prefill(&ids)?;
    }

    // ── input source ───────────────────────────────────────────────────────
    let input_rx = spawn_chat_input_sources(args.chat_socket, args.chat_tcp)?;
    let mut first_user: Option<String> = args
        .initial_user_prompt
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string());
    let mut recent: VecDeque<u32> = VecDeque::new();

    eprintln!(
        "[Chat] Ready. Arch={:?}, max_seq_len={}. Commands: /exit /reset /stats /help",
        args.model_arch, args.max_seq_len
    );
    let mut stdout_lock = std::io::stdout();

    'outer: loop {
        print!("> ");
        stdout_lock.flush().ok();

        let (user_line_raw, reply_writer) = if let Some(line) = first_user.take() {
            (line, None)
        } else {
            match input_rx.recv() {
                Ok(ChatInput::Line(s, w)) => (s, w),
                Ok(ChatInput::Eof) | Err(_) => {
                    eprintln!();
                    break 'outer;
                }
            }
        };
        let user_line = user_line_raw
            .trim_end_matches(&['\n', '\r'][..])
            .to_string();
        let trimmed = user_line.trim();

        match trimmed {
            "" => continue,
            "/exit" | "/quit" => break 'outer,
            "/help" => {
                println!("(commands: /exit /quit /reset /stats /help; empty line ignored)");
                continue;
            }
            "/stats" => {
                println!("{}", session.stats_line());
                continue;
            }
            "/reset" => {
                session.reset()?;
                recent.clear();
                println!("(session reset)");
                continue;
            }
            _ => {}
        }

        // ── user turn tokenize ─────────────────────────────────────────────
        let rendered = template.render_user_and_assistant_header(trimmed);
        let mut turn_ids: Vec<u32> = args
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode user turn: {}", e))?
            .get_ids()
            .to_vec();

        // BOS: system prompt가 없고 첫 user turn이면 prepend.
        if session.pos() == 0
            && let Some(b) = bos_id
        {
            turn_ids.insert(0, b);
        }

        // ── capacity check + prefill ───────────────────────────────────────
        if let Err(e) = session.ensure_capacity(turn_ids.len() + args.max_new_tokens) {
            let msg = format!("error: {}", e);
            eprintln!("{}", msg);
            write_reply_bytes(reply_writer.as_ref(), msg.as_bytes());
            finish_reply_stream(reply_writer.as_ref());
            anyhow::bail!("context overflow: {}", e);
        }

        let mut prefill_logits = session.prefill(&turn_ids)?;

        // ── first token sampling ───────────────────────────────────────────
        let mut indices_buf: Vec<usize> = Vec::with_capacity(args.vocab_size);
        let recent_slice: Vec<u32> = recent.iter().copied().collect();
        let first_tok = sampling::sample(
            &mut prefill_logits,
            &recent_slice,
            args.vocab_size,
            args.sampling_config,
            Some(&mut indices_buf),
        );

        // ── inner decode via run_turn — per-token streaming ────────────────
        // stop pos 상한: 현재 pos + max_new_tokens (overflow 안전망).
        let stop_max_pos = session.pos() + args.max_new_tokens;
        let stop_cond = ChatStopCondition::new(stop_ids.clone(), stop_max_pos);

        // 서버 SSE 경로(`server::stream_response`)와 동일: turn 직전 `stream_slot` 을
        // stdout/소켓 콜백으로 arm 하고 `IncDetok` 로 증분 detok 한다(multi-byte/BPE 경계
        // 안전). first_tok 은 run_turn 진입 전에 샘플되므로 콜백과 동일 경로로 한 번 직접
        // 흘린다. is_first_stop(첫 토큰이 stop_id)이면 출력은 빈 채로 두되 run_turn 은
        // 그대로 호출해 KV 상태 진행을 보존한다(슬롯 미arm → 생성 토큰 미스트리밍 = 기존
        // 일괄 경로의 `all_tokens=[]` 와 동치).
        let is_first_stop = stop_ids.contains(&first_tok);
        let rep_window = args.repetition_window.max(1);
        let tokenizer = args.tokenizer;
        let reply = reply_writer.as_ref();
        let mut detok = IncDetok::new();

        if is_first_stop {
            let _ = session.run_turn(first_tok, &stop_cond)?;
        } else {
            // first_tok 을 콜백과 동일 경로로 출력 + recent(rep penalty) 갱신.
            let piece = detok.push(first_tok, tokenizer);
            if !piece.is_empty() {
                write!(stdout_lock, "{}", piece).ok();
                stdout_lock.flush().ok();
                write_reply_bytes(reply, piece.as_bytes());
            }
            push_recent(&mut recent, first_tok, rep_window);

            {
                // Per-token 콜백: 증분 detok → stdout + 소켓. `ChatStreamStage` 가 kept
                // 토큰마다 발화(stop 토큰은 `ChatStopStage` 가 먼저 break → 미발화이므로
                // 일괄 `decode(all_tokens)` 와 동일 최종 문자열).
                let detok_ref = &mut detok;
                let recent_ref = &mut recent;
                let stdout_ref = &mut stdout_lock;
                let mut cb = |tok: u32| {
                    let piece = detok_ref.push(tok, tokenizer);
                    if !piece.is_empty() {
                        write!(stdout_ref, "{}", piece).ok();
                        stdout_ref.flush().ok();
                        write_reply_bytes(reply, piece.as_bytes());
                    }
                    push_recent(recent_ref, tok, rep_window);
                };
                let slot = session.stream_slot();
                let _guard = slot.arm(&mut cb);
                let _ = session.run_turn(first_tok, &stop_cond)?;
            }

            // multi-byte 경계로 보류된 잔여 바이트 flush (turn 끝 1회 — 서버 `detok.flush`).
            let tail = detok.flush(tokenizer);
            if !tail.is_empty() {
                write!(stdout_lock, "{}", tail).ok();
                stdout_lock.flush().ok();
                write_reply_bytes(reply, tail.as_bytes());
            }
        }

        // ── assistant EOT baking ───────────────────────────────────────────
        // stop token이 EOS/EOT이면 EOT를 한 번 더 baking하지 않는다 (중복 방지).
        // assistant_eot_ids가 빈 배열(Gemma3 등)이면 skip.
        if !assistant_eot_ids.is_empty()
            && session.pos() + assistant_eot_ids.len() <= args.max_seq_len
        {
            let _ = session.prefill(&assistant_eot_ids)?;
        }

        // ── on_turn_end (opportunistic eviction) ──────────────────────────
        session.on_turn_end()?;

        println!();
        stdout_lock.flush().ok();
        finish_reply_stream(reply_writer.as_ref());
    }

    Ok(())
}
