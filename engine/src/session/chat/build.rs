//! Assemble a [`ChatSession`] from parsed [`Args`] + a built [`SessionInitCtx`].
//!
//! Resolves [`Args::effective_kv_mode`] through the engine KV-mode registry
//! ([`resolve_kv_mode`]) and calls the mode's `build` fn-ptr to construct the
//! whole-pipeline `Box<dyn Forward>` (+ resilience handles / `ChatKvMode` payload).
//! The decode-loop assembly (sampler / registry / resilience wiring) is
//! mode-agnostic and lives here — it no longer matches on a concrete KV technique
//! identity (FORMAT-axis mode/knob declaration, design §4.3).

use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::session::chat::sampler::{ChatSampler, SharedSamplingConfig};
use crate::session::chat::session::{ChatSession, finish_chat_loop, make_chat_registry};
use crate::session::cli::Args;
use crate::session::init::SessionInitCtx;
use crate::session::mode::ChatModeBuild;
use crate::session::mode::ModeBuildCtx;
use crate::session::resilience_adapter::ResilienceAdapter;
use crate::session::resilience_init::build_command_executor;
use crate::session::{DecodeLoopBuilder, HasForward};

/// Build a [`ChatSession`] for the requested KV mode. Consumes `init` (owns the
/// loaded model). The resilience adapter is created here (manager IPC) when
/// `args.enable_resilience` is set; the mode's `build` closure surfaces the
/// resilience heartbeat handles, which are wired in here.
pub fn build_chat_session(init: SessionInitCtx, args: &Args) -> Result<ChatSession> {
    // Resilience adapter (before the model is moved). Graceful: transport failure
    // returns None inside build_command_executor; the eviction policy is set below.
    let mut resilience: Option<ResilienceAdapter> = if args.enable_resilience {
        build_command_executor(args, &init.model)?.map(ResilienceAdapter::new)
    } else {
        None
    };

    let SessionInitCtx {
        backend,
        memory,
        hardware,
        caps,
        model,
        sampling_config,
        ..
    } = init;
    let model = Arc::new(model);

    let max_seq_len = args.max_seq_len;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;
    let vocab_size = model.config.vocab_size;

    let cpu_backend = hardware
        .resolve(crate::hardware::DeviceTarget::Cpu)
        .expect("Cpu device always resolves")
        .0
        .clone();

    // Resolve the mode name through the engine KV-mode registry via the shared
    // checked funnel (fail-fast on an unknown name, listing the registered names —
    // the same funnel guards bench/eval, so the reject can't drift between bins).
    let mode_name = args.effective_kv_mode();
    let reg = crate::session::mode::resolve_kv_mode_checked(mode_name)?;

    // Mode's `build` fn-ptr constructs the whole-pipeline forward + handles. No
    // name-match here — the registry is the factory.
    let built = (reg.build)(ModeBuildCtx {
        args,
        backend,
        memory,
        cpu_backend,
        model,
        caps: &caps,
        kv_heads,
        head_dim,
        num_layers,
        max_seq_len,
    })?;
    let ChatModeBuild {
        forward,
        kv_handles,
        kv_handle,
        quant_handle,
        eviction_policy,
        kv_mode,
    } = built;

    // Mode-agnostic resilience handle wiring (§4.5: pos/capacity via base handle,
    // bit-width via the neutral QuantStageHandle).
    if let Some(adapter) = resilience.as_mut() {
        adapter.set_eviction_policy(&eviction_policy);
        if let Some(h) = kv_handle {
            adapter.set_kv_handle(h);
        }
        if let Some(q) = quant_handle {
            adapter.set_quant_handle(q);
        }
    }

    // Per-request sampling: the in-loop ChatSampler reads this shared config each
    // step; the server overwrites it per OpenAI request.
    let sampling: SharedSamplingConfig = Arc::new(Mutex::new(sampling_config));
    let (registry, stop_slot, stream_slot) = make_chat_registry();
    let builder: DecodeLoopBuilder<HasForward> = DecodeLoopBuilder::new()
        .with_forward_boxed(forward)
        .with_kv_capacity(max_seq_len)
        .with_sampler(ChatSampler::new(Arc::clone(&sampling), vocab_size));
    let decode_loop = finish_chat_loop(builder, registry, resilience, kv_handles);

    Ok(ChatSession::from_parts(
        decode_loop,
        kv_mode,
        max_seq_len,
        stop_slot,
        stream_slot,
        Some(sampling),
    ))
}
