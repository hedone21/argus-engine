//! argus-bench AB-2: KIVI dynamic-quant 지원 [`DecodeLoop`] 조립자.
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.7.7/§5.7.8.
//!
//! [`build_bench_loop`](super::build_bench_loop) 의 KIVI 형제다 — Standard `ModelForward`
//! (`Vec<KVCache>`) 대신 [`KiviForward`](crate::session::forward::KiviForward)(`Vec<KiviCache>`)
//! 를 조립하고, dispatcher 에 `kivi_handles` 를 주입해 `KvQuantDynamic` directive 가 OneShot
//! [`KiviQuantStage`](crate::stages::kv::kivi_quant::KiviQuantStage) 로 submit 되게 한다.
//!
//! **stage 배선 범위 (§5.7.7)**: control 디렉티브(Throttle/SetTargetTbt/Suspend/Resume/
//! RestoreDefaults) + **KvQuantDynamic** 만 활성. eviction/partition/swap stage 는 KIVI 경로에서
//! **미배선**(빈 handle/None — inert). v1 등가: KIVI 경로는 eviction 미지원(`on_kv_prune` no-op, D4).

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::capability::kivi_attention::KiviAttentionBackend;
use crate::inference::sampling::SamplingConfig;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::session::command_dispatcher::CommandDispatcher;
use crate::session::forward::KiviForward;
use crate::session::forward::kivi_forward::alloc_kivi_kv_caches;
use crate::session::pipeline_registry::PipelineRegistry;
use crate::session::resilience_adapter::ResilienceAdapter;
use crate::session::{DecodeLoop, DecodeLoopBuilder, GreedySampler, RepetitionPenaltySampler};

/// KIVI bench `DecodeLoop` 조립 (AB-2 §5.7.8).
///
/// `build_chat_kivi`(chat/session.rs:491)의 KiviForward 생성 recipe 에 dispatcher/registry/
/// resilience 배선을 더한다(`build_bench_loop` 의 Standard 배선과 동형이되 KiviQuantStage 만 활성).
#[allow(clippy::too_many_arguments)]
pub fn build_bench_kivi_loop(
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    model: TransformerModel,
    kivi: &Option<Arc<dyn KiviAttentionBackend>>,
    initial_bits: u8,
    residual_size: usize,
    max_seq_len: usize,
    sampling_config: SamplingConfig,
    resilience: Option<ResilienceAdapter>,
) -> Result<DecodeLoop> {
    let vocab_size = model.config.vocab_size;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;

    eprintln!(
        "[DecodeLoop/KIVI] bits={}, residual_size={}, layers={}, max_seq_len={}",
        initial_bits, residual_size, num_layers, max_seq_len
    );

    // KIVI cache alloc (R3: OpenCL backend 면 `kivi` 가 Some 필수, init.rs 가 register).
    let kv_caches = alloc_kivi_kv_caches(
        num_layers,
        kv_heads,
        head_dim,
        max_seq_len,
        residual_size,
        initial_bits,
        &backend,
        kivi,
        &memory,
    );

    let fwd = KiviForward::new(
        backend,
        memory,
        Arc::new(model),
        kv_caches,
        initial_bits,
        residual_size,
        max_seq_len,
    )?;

    // §5.7.8: KiviQuantStage 가 transition 할 persistent KIVI handle (register 시점 보유).
    let kivi_handles = fwd.kivi_caches().to_vec();

    // §5.7.6: heartbeat kv_dtype query 용 layer-0 KIVIFormat concrete handle (resilience adapter
    // 에 주입 — bits query 는 base trait 표면에 없어 KIVI concrete 필요).
    let kivi_handle = fwd.kivi_caches().first().cloned();

    let registry = Arc::new(PipelineRegistry::new());

    // §5.7.6: resilience adapter 에 KIVI handle 주입 → heartbeat kv_dtype 를 현재 bits 에서 query.
    let resilience = match (resilience, kivi_handle) {
        (Some(mut adapter), Some(h)) => {
            adapter.set_kivi_handle(h);
            Some(adapter)
        }
        (other, _) => other,
    };

    // AB-5 §5.8.4: report_tx = resilience.as_ref().map(|a| a.report_sender()) — Standard 경로
    // build_bench_loop 와 동일 source(같은 report_sender() clone). resilience-off 면 None → inert.
    let report_tx_for_dispatcher = resilience.as_ref().map(|a| a.report_sender());

    // §5.7.7: dispatcher — control + KvQuantDynamic 만 활성. eviction(kv_handles 빈 + CM=None)·
    // partition(layer_slots 빈 + hardware None)·swap(model/swap_runtime None) 전부 inert.
    // resilience-on 일 때만 구성(control 디렉티브 소비 + KvQuantDynamic).
    let dispatcher = resilience.is_some().then(|| {
        CommandDispatcher::new(
            Arc::clone(&registry),
            Vec::new(), // kv_handles: KIVI 경로 eviction 미지원 (StandardFormat 부재).
            None,       // cache_manager: eviction inert.
            Vec::new(), // layer_slots: partition 미배선.
            None,       // hardware: partition inert.
            None,       // model: swap 미배선.
            None,       // swap_runtime: swap inert.
            None,       // importance.
            kivi_handles,
            // AB-5: QcfEstimate 송출 채널. resilience-on 이면 Some, off 이면 None(inert).
            report_tx_for_dispatcher,
            // §5.9.2 Track B: KIVI 경로는 swap 미배선(model None) → swap directive inert.
            // 더미 cell (KiviForward 는 ModelForward 와 무관 — hook 미소비).
            Arc::new(std::sync::Mutex::new(None)),
            // §5.9.1 Track A: KIVI 경로는 score-based eviction 미지원 → 더미 None cell.
            Arc::new(std::sync::Mutex::new(None)),
        )
    });

    let use_stateful =
        sampling_config.repetition_penalty != 1.0 || sampling_config.temperature != 0.0;
    let builder = DecodeLoopBuilder::new()
        .with_forward(fwd)
        .with_kv_capacity(max_seq_len)
        .with_pipeline(Arc::clone(&registry));
    let builder = if use_stateful {
        builder.with_sampler(RepetitionPenaltySampler::new(sampling_config, vocab_size))
    } else {
        builder.with_sampler(GreedySampler)
    };
    let builder = match resilience {
        Some(adapter) => builder.with_resilience(adapter),
        None => builder,
    };
    let builder = match dispatcher {
        Some(d) => builder.with_command_dispatcher(d),
        None => builder,
    };
    Ok(builder.build())
}
