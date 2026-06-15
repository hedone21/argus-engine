//! OpenAI-compatible HTTP chat server for `argus-chat`.
//!
//! Blocking, single-threaded: the engine decode loop is not `Send` and is
//! serialized on one thread anyway, so [`tiny_http`]'s accept loop runs on the
//! calling thread and handles one request at a time. Routes:
//!
//! - `POST /v1/chat/completions` — OpenAI chat completions (stream + non-stream)
//! - `GET  /v1/models`           — list the single loaded model
//! - `GET  /health`              — liveness
//!
//! Statelessness: each request carries the whole conversation in `messages[]`,
//! so the handler [`ChatSession::reset`]s, renders the full history, prefills,
//! and generates. Per-request `temperature`/`top_p` are applied via
//! [`ChatSession::set_sampling`]. Streaming writes Server-Sent Events directly to
//! the raw connection ([`tiny_http::Request::into_writer`]).

use std::io::Write;
use std::sync::atomic::Ordering;

use anyhow::{Result, anyhow};
use tiny_http::{Header, Method, Request, Response, Server};
use tokenizers::Tokenizer;

use crate::inference::sampling::{self, SamplingConfig};
use crate::model_config::ModelArch;
use crate::session::chat::openai::{
    ChatCompletion, ChatCompletionChunk, Choice, ChunkChoice, Delta, OpenAiChatRequest,
    RespMessage, Usage, effective_sampling, extra_stop_ids, render_messages, unix_secs,
};
use crate::session::chat::session::ChatSession;
use crate::session::chat::stop_condition::{ChatStopCondition, build_chat_stop_ids};
use crate::session::chat::stream_stage::IncDetok;
use crate::session::chat_template::ChatTemplate;
use crate::session::cli::Args;
use crate::session::decode_loop::StopReason;

/// Default per-request generation cap when `max_tokens` is omitted.
const DEFAULT_MAX_TOKENS: usize = 512;

/// Immutable per-server context shared across requests.
struct ServerCtx {
    tokenizer: Tokenizer,
    template: ChatTemplate,
    /// Template EOT/EOS stop ids (request `stop[]` are unioned per request).
    base_stop_ids: Vec<u32>,
    base_sampling: SamplingConfig,
    vocab_size: usize,
    model_id: String,
}

/// Run the OpenAI-compatible HTTP server (blocking). Owns `session` for the
/// process lifetime; serves requests serially on the calling thread.
pub fn serve(
    args: &Args,
    mut session: ChatSession,
    tokenizer: Tokenizer,
    arch: ModelArch,
    eos_token_id: u32,
    vocab_size: usize,
    base_sampling: SamplingConfig,
) -> Result<()> {
    let template = ChatTemplate::new(arch)?;
    let base_stop_ids = build_chat_stop_ids(&template, &tokenizer, eos_token_id)?;
    let model_id = std::path::Path::new(&args.model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("argus")
        .to_string();

    let addr = args.listen_addr();
    let server = Server::http(addr.as_str())
        .map_err(|e| anyhow!("failed to bind HTTP server at {addr}: {e}"))?;
    eprintln!("[argus-chat] OpenAI-compatible server listening on http://{addr}");
    eprintln!("[argus-chat] endpoints: POST /v1/chat/completions, GET /v1/models, GET /health");

    let ctx = ServerCtx {
        tokenizer,
        template,
        base_stop_ids,
        base_sampling,
        vocab_size,
        model_id,
    };

    for request in server.incoming_requests() {
        if let Err(e) = route(request, &ctx, &mut session) {
            eprintln!("[argus-chat] request error: {e}");
        }
    }
    Ok(())
}

fn route(request: Request, ctx: &ServerCtx, session: &mut ChatSession) -> Result<()> {
    let method = request.method().clone();
    let path = request.url().split('?').next().unwrap_or("").to_string();

    match (&method, path.as_str()) {
        (Method::Options, _) => respond_cors_preflight(request),
        (Method::Get, "/health") => respond_text(request, 200, "ok"),
        (Method::Get, "/v1/models") => {
            let body = serde_json::json!({
                "object": "list",
                "data": [{
                    "id": ctx.model_id,
                    "object": "model",
                    "created": unix_secs(),
                    "owned_by": "argus-engine",
                }],
            })
            .to_string();
            respond_json(request, 200, body)
        }
        (Method::Post, "/v1/chat/completions") => handle_chat(request, ctx, session),
        _ => respond_error(request, 404, &format!("no route for {method} {path}")),
    }
}

fn handle_chat(mut request: Request, ctx: &ServerCtx, session: &mut ChatSession) -> Result<()> {
    // 1. read + parse body.
    let mut body = String::new();
    if let Err(e) = request.as_reader().read_to_string(&mut body) {
        return respond_error(request, 400, &format!("failed to read body: {e}"));
    }
    let req: OpenAiChatRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(e) => return respond_error(request, 400, &format!("invalid JSON request: {e}")),
    };

    // 2. render the full conversation → prompt.
    let rendered = match render_messages(&ctx.template, &req.messages) {
        Ok(s) => s,
        Err(e) => return respond_error(request, 400, &e.to_string()),
    };

    // 3. stateless: reset, then encode (+BOS for Llama on the fresh sequence).
    session.reset()?;
    let mut ids: Vec<u32> = match ctx.tokenizer.encode(rendered.as_str(), false) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => return respond_error(request, 400, &format!("tokenize error: {e}")),
    };
    if ctx.template.bos_needed_on_first_prefill()
        && let Some(b) = ctx
            .template
            .bos_literal()
            .and_then(|lit| ctx.tokenizer.token_to_id(lit))
    {
        ids.insert(0, b);
    }

    // 4. budget + capacity.
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS).max(1);
    if let Err(e) = session.ensure_capacity(ids.len() + max_tokens) {
        return respond_error(request, 400, &e.to_string());
    }

    // 5. per-request sampling + prefill + first token.
    let cfg = effective_sampling(&ctx.base_sampling, &req);
    session.set_sampling(cfg.clone());
    let prompt_tokens = ids.len();
    let mut logits = session.prefill(&ids)?;
    let first = sampling::sample(&mut logits, &ids, ctx.vocab_size, &cfg, None);

    // 6. stop set: template EOT/EOS + request stop strings (single-token only).
    let mut stop_ids = ctx.base_stop_ids.clone();
    stop_ids.extend(extra_stop_ids(&req, &ctx.tokenizer));
    let stop = ChatStopCondition::new(stop_ids.clone(), session.pos() + max_tokens);
    let first_is_stop = stop_ids.contains(&first);

    if req.is_stream() {
        stream_response(request, ctx, session, first, first_is_stop, &stop)
    } else {
        whole_response(
            request,
            ctx,
            session,
            first,
            first_is_stop,
            &stop,
            prompt_tokens,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn whole_response(
    request: Request,
    ctx: &ServerCtx,
    session: &mut ChatSession,
    first: u32,
    first_is_stop: bool,
    stop: &ChatStopCondition,
    prompt_tokens: usize,
) -> Result<()> {
    let (out_ids, finish) = if first_is_stop {
        (Vec::new(), "stop")
    } else {
        let result = session.run_turn(first, stop)?;
        let mut v = Vec::with_capacity(1 + result.tokens_generated.len());
        v.push(first);
        v.extend_from_slice(&result.tokens_generated);
        (v, finish_reason(&result.stopped_by))
    };
    let completion_tokens = out_ids.len();
    let text = ctx.tokenizer.decode(&out_ids, true).unwrap_or_default();

    let resp = ChatCompletion {
        id: completion_id(),
        object: "chat.completion",
        created: unix_secs(),
        model: ctx.model_id.clone(),
        choices: vec![Choice {
            index: 0,
            message: RespMessage {
                role: "assistant",
                content: text,
            },
            finish_reason: finish.to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };
    let json = serde_json::to_string(&resp)?;
    respond_json(request, 200, json)
}

fn stream_response(
    request: Request,
    ctx: &ServerCtx,
    session: &mut ChatSession,
    first: u32,
    first_is_stop: bool,
    stop: &ChatStopCondition,
) -> Result<()> {
    let id = completion_id();
    let created = unix_secs();
    let model = ctx.model_id.clone();

    let mut w = request.into_writer();
    w.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\n\
          Connection: close\r\n\
          Access-Control-Allow-Origin: *\r\n\r\n",
    )?;
    // initial role chunk (OpenAI sends delta.role="assistant" first).
    sse_chunk(&mut w, &id, created, &model, Some("assistant"), None, None)?;

    let mut detok = IncDetok::new();
    let mut stopped_by = StopReason::StopConditionMet;

    if !first_is_stop {
        let d0 = detok.push(first, &ctx.tokenizer);
        if !d0.is_empty() {
            sse_chunk(&mut w, &id, created, &model, None, Some(d0), None)?;
        }
        let stop_flag = session.stop_flag();
        {
            // Per-token callback: incremental detok → SSE delta. On a write error
            // (client gone) flip the loop stop flag to cut generation short.
            let mut cb = |tok: u32| {
                let piece = detok.push(tok, &ctx.tokenizer);
                if !piece.is_empty()
                    && sse_chunk(&mut w, &id, created, &model, None, Some(piece), None).is_err()
                {
                    stop_flag.store(true, Ordering::Release);
                }
            };
            let slot = session.stream_slot();
            let _guard = slot.arm(&mut cb);
            stopped_by = session.run_turn(first, stop)?.stopped_by;
        }
        let tail = detok.flush(&ctx.tokenizer);
        if !tail.is_empty() {
            sse_chunk(&mut w, &id, created, &model, None, Some(tail), None)?;
        }
    }

    let finish = finish_reason(&stopped_by);
    sse_chunk(&mut w, &id, created, &model, None, None, Some(finish))?;
    w.write_all(b"data: [DONE]\n\n")?;
    w.flush()?;
    Ok(())
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn finish_reason(stopped_by: &StopReason) -> &'static str {
    match stopped_by {
        StopReason::StopConditionMet | StopReason::EosToken => "stop",
        _ => "length",
    }
}

fn completion_id() -> String {
    format!("chatcmpl-{}", unix_secs())
}

fn sse_chunk(
    w: &mut dyn Write,
    id: &str,
    created: u64,
    model: &str,
    role: Option<&'static str>,
    content: Option<String>,
    finish: Option<&str>,
) -> std::io::Result<()> {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta { role, content },
            finish_reason: finish.map(|s| s.to_string()),
        }],
    };
    let json = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
    w.write_all(b"data: ")?;
    w.write_all(json.as_bytes())?;
    w.write_all(b"\n\n")?;
    w.flush()
}

fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
        .expect("static header is valid")
}

fn cors_header() -> Header {
    Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..])
        .expect("static header is valid")
}

fn respond_json(request: Request, status: u16, json: String) -> Result<()> {
    let resp = Response::from_string(json)
        .with_status_code(status)
        .with_header(json_header())
        .with_header(cors_header());
    request.respond(resp).map_err(|e| anyhow!("respond: {e}"))
}

fn respond_text(request: Request, status: u16, text: &str) -> Result<()> {
    let resp = Response::from_string(text.to_string())
        .with_status_code(status)
        .with_header(cors_header());
    request.respond(resp).map_err(|e| anyhow!("respond: {e}"))
}

fn respond_error(request: Request, status: u16, msg: &str) -> Result<()> {
    let body = serde_json::json!({
        "error": { "message": msg, "type": "invalid_request_error" }
    })
    .to_string();
    respond_json(request, status, body)
}

fn respond_cors_preflight(request: Request) -> Result<()> {
    let resp = Response::empty(204)
        .with_header(cors_header())
        .with_header(
            Header::from_bytes(
                &b"Access-Control-Allow-Methods"[..],
                &b"GET, POST, OPTIONS"[..],
            )
            .expect("static header is valid"),
        )
        .with_header(
            Header::from_bytes(
                &b"Access-Control-Allow-Headers"[..],
                &b"Content-Type, Authorization"[..],
            )
            .expect("static header is valid"),
        );
    request.respond(resp).map_err(|e| anyhow!("respond: {e}"))
}
