//! W-SIGNAL-H (SqueezeAttention) production arming — the warmup-prefill *capture* half.
//!
//! SqueezeAttention assigns different per-layer KV budgets based on each decoder layer's
//! importance, measured as `1 − cos(h_in, h_out)` (how much the layer transforms the residual
//! stream). The scoring arithmetic lives in the `layer-importance` plugin (`mean_pool` /
//! `shortgpt_bi`) and the budget allocation in [`crate::kv::squeeze_budget`]; this module only
//! *captures* the per-layer importance by running one warmup prefill over the prompt with an
//! [`ImportanceCollector`] attached (the exact pass `--dump-importance` / qcf-warmup use), then
//! returns the flat per-layer importance vector for the eviction layer to turn into budgets.
//!
//! The hidden-state read inside `forward_into` is device-safe (`read_buffer`, see
//! `transformer::read_hidden_for_importance`), so the warmup runs correctly on Adreno GPU as well
//! as host CPU. It is opt-in (`--kv-squeeze`): production decode never constructs a collector and
//! is byte-identical.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::kv::kv_cache::KVCache;
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::qcf::ImportanceCollector;
use crate::session::eval::EvalCacheKind;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Parse `--kv-squeeze-formula` into an activation-only `(ImportanceFormula, three_way)`.
///
/// Only `mean_pool` / `shortgpt_bi` are valid — both are `PerLayerStreaming` 1 − cos signals. The
/// weight-perturbation formulas (`dpllm_*` / `direct_attn`) are rejected here: they are
/// `OneShotPostWarmup` scorers whose real value is computed post-warmup, so in this single
/// streaming pass they would yield a constant placeholder (no layer differentiation).
fn parse_squeeze_formula(name: &str) -> Result<(crate::qcf_types::ImportanceFormula, bool)> {
    use crate::qcf_types::ImportanceFormula;
    match name {
        "mean_pool" => Ok((ImportanceFormula::MeanPool, false)),
        "shortgpt_bi" => Ok((ImportanceFormula::PerTokenCosineBi, true)),
        other => anyhow::bail!(
            "--kv-squeeze-formula: unknown/unsupported value '{other}'. \
             Valid: mean_pool, shortgpt_bi (activation-only 1 − cos signals)"
        ),
    }
}

/// Run one warmup prefill over `tokens` with an [`ImportanceCollector`] and return the flat
/// per-layer importance (`out[i]` = layer `i`'s `SubLayer::Full` importance, length
/// `num_hidden_layers`). Empty vec when there is nothing to score.
///
/// `kv_caches` is used for the warmup forward and **reset to empty** afterwards (`current_pos` /
/// `high_water_pos` → 0) so a subsequent fresh / prefix-restore prefill starts from a clean cache.
/// Call this on a clean cache (before any prefix-cache restore), since the warmup overwrites
/// positions `0..prompt_len`.
pub fn capture_layer_importance_via_warmup(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut Vec<KVCache>,
    tokens: &[u32],
    vocab_size: usize,
    formula_name: &str,
) -> Result<Vec<f32>> {
    let n_layers = model.config.num_hidden_layers;
    if tokens.is_empty() || n_layers == 0 || kv_caches.is_empty() {
        return Ok(Vec::new());
    }

    // Force-link the layer-importance scorers (fat-LTO --gc-sections safety; same guard the qcf
    // warmup uses) so `mean_pool` / `shortgpt_bi` resolve.
    crate::qcf::layer_importance::ensure_layer_scorers_registered()?;

    let (formula, three_way) = parse_squeeze_formula(formula_name)?;
    let mut collector = ImportanceCollector::new_with_formula(formula, three_way);

    // Build the prompt input tensor (CPU u32 → backend) and a logits sink, mirroring
    // dump_importance.rs (the same warmup-prefill shape).
    let prompt_len = tokens.len();
    let cpu_buf = Galloc::new().alloc(prompt_len * 4, DType::U8)?;
    // SAFETY: cpu_buf is prompt_len * 4 bytes = prompt_len u32 slots; we copy exactly prompt_len.
    unsafe {
        let ptr = cpu_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, prompt_len);
    }
    let cpu_input = Tensor::new(
        Shape::new(vec![1, prompt_len]),
        cpu_buf,
        Arc::new(CpuBackend::new()),
    );
    let input_tensor = backend.copy_from(&cpu_input)?;

    let logits_buf = memory.alloc(prompt_len * vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, prompt_len, vocab_size]),
        logits_buf,
        backend.clone(),
    );

    KVCache::forward_fmt_roundtrip(kv_caches, |fmts| {
        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            fmts,
            backend,
            memory,
            logits_out: &mut logits,
            x_gen: None,
            workspace: None,
            logits_last_only: false,
            score_accumulator: None,
            query_stats_accumulator: None,
            skip_config: None,
            importance_collector: Some(&mut collector),
            cache_self_need_scores: false,
            layer_boundary_hook: None,
            read_stage: None,
            prefill_attn: None,
        })
    })?;

    let table = collector.build();
    let importance = crate::weight::decider::flatten_importance(&table, n_layers);

    // Reset the warmup KV so the real prefill starts from an empty cache.
    for cache in kv_caches.iter_mut() {
        cache.current_pos = 0;
        cache.high_water_pos = 0;
    }

    Ok(importance)
}
