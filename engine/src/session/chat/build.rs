//! Assemble a [`ChatSession`] from parsed [`Args`] + a built [`SessionInitCtx`].
//!
//! Mirrors [`build_inference_ctx`](crate::session::bin_setup::build_inference_ctx)
//! for the chat path: dispatches on [`Args::effective_kv_mode`] to the right
//! `build_chat_*` builder, allocating KV caches and wiring the resilience adapter
//! (manager IPC). The actual decode-loop assembly + resilience wiring lives in the
//! builders ([`build_chat_standard`] etc.).

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::buffer::DType;
use crate::capability::kivi_attention::KiviAttentionBackend;
use crate::hardware::DeviceTarget;
use crate::session::bin_setup::alloc_standard_kv_caches;
use crate::session::chat::session::{
    ChatKiviArgs, ChatOffloadArgs, ChatSession, ChatStandardArgs, build_chat_kivi,
    build_chat_offload, build_chat_standard,
};
use crate::session::cli::{Args, KvMode};
use crate::session::init::SessionInitCtx;
use crate::session::resilience_adapter::ResilienceAdapter;
use crate::session::resilience_init::build_command_executor;

/// Build a [`ChatSession`] for the requested KV mode. Consumes `init` (owns the
/// loaded model). The resilience adapter is created here (manager IPC) when
/// `args.enable_resilience` is set, and the per-mode builder wires it into the
/// decode loop.
pub fn build_chat_session(init: SessionInitCtx, args: &Args) -> Result<ChatSession> {
    // Resilience adapter (before the model is moved). Graceful: transport failure
    // returns None inside build_command_executor; the eviction policy is set by
    // the per-mode builder.
    let resilience: Option<ResilienceAdapter> = if args.enable_resilience {
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

    match args.effective_kv_mode() {
        KvMode::Standard => {
            let kv_dtype = parse_kv_type(&args.kv_type)?;
            let initial = {
                let r = args.initial_kv_capacity();
                if r > 0 {
                    r.min(max_seq_len)
                } else {
                    256.min(max_seq_len)
                }
            };
            let cpu_backend = hardware
                .resolve(DeviceTarget::Cpu)
                .expect("Cpu device always resolves")
                .0
                .clone();
            let kv_caches = alloc_standard_kv_caches(
                &backend,
                memory.clone(),
                num_layers,
                initial,
                max_seq_len,
                kv_heads,
                head_dim,
                kv_dtype,
            )?;
            build_chat_standard(ChatStandardArgs {
                backend,
                memory,
                cpu_backend,
                model,
                kv_caches,
                initial_kv_capacity: initial,
                max_seq_len,
                kv_dtype,
                eviction_policy: args.eviction_policy().to_string(),
                eviction_target_ratio: args.eviction_target_ratio(),
                eviction_window: args.eviction_window(),
                protected_prefix: args.protected_prefix(),
                sink_size: args.sink_size(),
                streaming_window: args.streaming_window(),
                kv_budget: args.kv_budget(),
                h2o_keep_ratio: args.h2o_keep_ratio(),
                h2o_tracked_layers: args.h2o_tracked_layers(),
                h2o_decay: args.h2o_decay(),
                h2o_raw_scores: args.h2o_raw_scores(),
                d2o_keep_ratio: args.d2o_keep_ratio(),
                d2o_ema_beta: args.d2o_ema_beta(),
                d2o_merge_e: args.d2o_merge_e(),
                d2o_layer_alloc: args.d2o_layer_alloc(),
                d2o_protected_layers: args.d2o_protected_layers().unwrap_or_default(),
                memory_threshold_mb: args.memory_threshold_mb() as u64,
                sampling_config,
                resilience,
            })
        }
        KvMode::Kivi => {
            let kivi = caps.get::<dyn KiviAttentionBackend>();
            build_chat_kivi(ChatKiviArgs {
                backend,
                kivi,
                memory,
                model,
                kv_heads,
                head_dim,
                num_layers,
                max_seq_len,
                bits: args.effective_kivi_bits(),
                residual_size: args.effective_kivi_residual_size(),
                sampling_config,
                resilience,
            })
        }
        KvMode::Offload => {
            let kv_dtype = parse_kv_type(&args.kv_type)?;
            build_chat_offload(ChatOffloadArgs {
                backend,
                memory,
                model,
                kv_heads,
                head_dim,
                num_layers,
                max_seq_len,
                kv_dtype,
                offload_mode: args.effective_kv_offload_storage(),
                disk_dir: args.swap_dir.clone(),
                max_prefetch_depth: args.kv_mode_args.kv_max_prefetch_depth,
                sampling_config,
                resilience,
            })
        }
    }
}

fn parse_kv_type(s: &str) -> Result<DType> {
    match s {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "q4" => Ok(DType::Q4_0),
        other => bail!("Unsupported KV type: {other}. Use f32, f16, or q4."),
    }
}
