//! Chat REPL v2 — [`ChatSession`] 기반 multi-turn loop (Phase 4-5-e).
//!
//! [`run_chat_repl_v2`]는 generate.rs::run_chat_repl (l.9855~10053)의 로직을
//! [`ChatSession`]을 사용하여 재작성한 버전이다.
//!
//! 주요 차이점:
//! - `ChatTurnExec` trait 대신 `ChatSession` API (prefill/run_turn/ensure_capacity)
//! - decode inner loop는 `DecodeLoop::run_until_stop`에 위임 (streaming 대신 일괄 출력)
//! - eviction은 ChatSession::ensure_capacity + on_turn_end에 내장

use std::collections::VecDeque;
use std::io::Write as _;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::inference::sampling::{self, SamplingConfig};
use crate::model_config::ModelArch;
use crate::session::chat::session::ChatSession;
use crate::session::chat::stop_condition::{ChatStopCondition, build_chat_stop_ids};
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

        // ── inner decode via run_turn ──────────────────────────────────────
        // stop pos 상한: 현재 pos + max_new_tokens (overflow 안전망).
        let stop_max_pos = session.pos() + args.max_new_tokens;
        let stop_cond = ChatStopCondition::new(stop_ids.clone(), stop_max_pos);

        let decode_result = session.run_turn(first_tok, &stop_cond)?;

        // ── 출력 (first_tok + generated) ──────────────────────────────────
        // first_tok가 stop_id이면 빈 출력. 아니면 first_tok + tokens_generated.
        let is_first_stop = stop_ids.contains(&first_tok);
        let all_tokens: Vec<u32> = if is_first_stop {
            vec![]
        } else {
            let mut v = vec![first_tok];
            v.extend_from_slice(&decode_result.tokens_generated);
            v
        };

        if !all_tokens.is_empty() {
            let text = args.tokenizer.decode(&all_tokens, true).unwrap_or_default();
            print!("{}", text);
            stdout_lock.flush().ok();
            write_reply_bytes(reply_writer.as_ref(), text.as_bytes());

            // recent 갱신 (rep penalty용)
            for &t in &all_tokens {
                recent.push_back(t);
                if recent.len() > args.repetition_window.max(1) {
                    recent.pop_front();
                }
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
