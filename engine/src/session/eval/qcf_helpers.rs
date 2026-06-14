//! Layer-swap QCF dump infrastructure for the `--qcf-dump` workflow.
//!
//! Collected by `generate.rs` and serialized to a JSON file consumed by
//! the external harness (`pact2026/experiments/scripts/`) to correlate
//! `qcf_swap_predicted` with actual NLL/quality metrics.

use super::output::EvalOutput;
use crate::qcf::layer_importance::ImportanceTable;
use crate::runtime_resources_access::QuantNoiseAccess;

/// Context for one layer-swap QCF measurement run.
pub struct QcfSwapDumpContext<'a> {
    pub model_arch: &'a str,
    pub model_path: &'a str,
    pub secondary_path: Option<&'a str>,
    pub primary_dtype: &'a str,
    pub secondary_dtype: &'a str,
    pub num_layers: usize,
    pub force_swap_ratio: Option<f32>,
    /// Layer-selection algorithm short name used for this run (U5 ablation).
    /// One of `"imp" | "seq" | "rev" | "uni" | "anti"`. `None` when no swap was
    /// performed (e.g. baseline ratio=0).
    pub swap_algorithm: Option<&'a str>,
    /// Decoder layer indices that were swapped (empty if ratio=0 or no secondary).
    pub swap_set: &'a [usize],
    /// QCF_swap predicted value from `WeightSwapDecider` (ENG-ALG-217).
    pub qcf_swap_predicted: f32,
    /// `true` when uniform fallback was used (importance/noise absent).
    pub fallback_used: bool,
    /// Full importance table built from warmup prefill (optional).
    pub importance_table: Option<&'a ImportanceTable>,
    /// Quantization noise table built from secondary mmap (optional). Held as
    /// `dyn` so callers can pass `model.quant_noise.as_ref()` directly.
    pub noise_table: Option<&'a dyn QuantNoiseAccess>,
    /// Perplexity result from `run_ppl()` (None in generation mode).
    pub ppl: Option<f64>,
    /// Average negative log-likelihood (None in generation mode).
    pub avg_nll: Option<f64>,
    /// Number of tokens evaluated in the main measurement.
    pub n_eval_tokens: usize,
    /// Total wall-clock time in seconds (warmup + swap + measurement).
    pub wall_time_s: f64,
    /// Number of warmup prefill tokens used to build the importance table.
    pub warmup_tokens: usize,
    /// Backend string (e.g. "cpu", "opencl", "cuda").
    pub backend: &'a str,
    /// KV cache dtype string (e.g. "f16", "f32", "q4_0").
    pub kv_type: &'a str,
    /// Path to the PPL reference corpus file (None in generation mode).
    pub ppl_corpus: Option<&'a str>,
    /// EvalOutput from `--eval-ll` mode (per-question NLL summary).
    ///
    /// Set when `--eval-ll` and `--qcf-dump` are both active. None in PPL/generation mode.
    /// The full `EvalOutput` is serialized as `eval_ll_output` in the JSON dump so the
    /// external harness can compute `qcf_swap_predicted ↔ ΔNLL` Spearman ρ directly.
    pub eval_ll_output: Option<&'a EvalOutput>,
    /// Per-step trajectory for `--qcf-trajectory` mode (U5 mid-swap quality study).
    ///
    /// When set, `eval_ll_output` is ignored and the JSON dump contains a
    /// `trajectory` array with one entry per swap step: step index, the
    /// cumulative set of layers swapped *before* the eval-ll measurement at
    /// this step, the layer added at this step (`null` for step 0 baseline),
    /// and the full `EvalOutput` from the eval-ll run.
    pub trajectory: Option<&'a [TrajectoryStep<'a>]>,
    /// Per-layer DP-LLM proxy ε (single-tensor relative) from
    /// `--importance-formula compare` mode. Length = `num_layers`.
    pub dpllm_epsilon: Option<&'a [f32]>,
    /// §4 candidate A: per-layer multi-tensor sum of input-aware relative ε.
    pub dpllm_epsilon_multi: Option<&'a [f32]>,
    /// §4 candidate D: per-layer absolute L2 of the activation difference.
    pub dpllm_epsilon_abs: Option<&'a [f32]>,
    /// §4 candidate E: per-layer QCF-style multiplicative `ε_v × ε_o`.
    pub dpllm_epsilon_qcf: Option<&'a [f32]>,
    /// §4.2 F4: per-layer cascade-aware single output projection perturbation.
    /// `‖(W_o^F16 − W_o^Q4) · V_out^F16‖_F / ‖W_o^F16 · V_out^F16‖_F`.
    pub direct_attn_f4: Option<&'a [f32]>,
    /// §4.2 F5: per-layer direct attention output relative L2 perturbation.
    /// `‖O^F16 − O^Q4‖_F / ‖O^F16‖_F`.
    pub direct_attn_f5: Option<&'a [f32]>,
    /// §4.2 decode-only F5: per-layer F5 evaluated with X = the N decode-step
    /// raws (T = N) only. `Some` only when `--decode-x-steps > 0` and a
    /// secondary GGUF was loaded.
    pub direct_attn_f5_decode_only: Option<&'a [f32]>,
    /// §4.2 prefill+decode F5: per-layer F5 evaluated with X =
    /// concat(prefill raws, decode raws) (T = 256 + N). `Some` only when
    /// `--decode-x-steps > 0` and a secondary GGUF was loaded.
    pub direct_attn_f5_prefill_decode: Option<&'a [f32]>,
}

/// One step of a `--qcf-trajectory` measurement.
pub struct TrajectoryStep<'a> {
    /// Step index, starting at 0 (= no swap baseline) and going up to K.
    pub step: usize,
    /// Layers swapped *before* this step's eval-ll measurement.
    /// At step 0 this is empty; at step t > 0 this is `selected_layers[..t]`.
    pub swapped_layers: Vec<usize>,
    /// Layer added at this step (`None` for step 0 baseline).
    pub layer_added: Option<usize>,
    /// eval-ll output measured *after* the swap at this step.
    pub eval_ll_output: &'a EvalOutput,
}

/// Serialize a `QcfSwapDumpContext` to a JSON file.
///
/// `schema_version` is `1` for the single-eval (non-trajectory) layout and `2`
/// when `ctx.trajectory.is_some()` (the `trajectory` field carries a per-step
/// array of `EvalOutput`s).
///
/// The JSON schema matches the external harness expectation exactly:
/// - All fields are always present (`null` when absent, NOT omitted).
/// - `importance_table` and `noise_table` include all collected entries.
/// - `noise_table` excludes NaN/non-finite ε entries.
/// - `swap_count` is derived as `swap_set.len()`.
pub fn dump_qcf_swap_json(
    path: &std::path::Path,
    ctx: &QcfSwapDumpContext<'_>,
) -> anyhow::Result<()> {
    use serde_json::{Value, json};

    // importance_table entries
    let importance_arr: Value = match ctx.importance_table {
        Some(table) => {
            let entries: Vec<Value> = table
                .entries()
                .iter()
                .map(|e| {
                    json!({
                        "layer": e.layer_id,
                        "sublayer": format!("{:?}", e.sublayer),
                        "importance": e.importance,
                        "opr": e.opr,
                    })
                })
                .collect();
            Value::Array(entries)
        }
        None => Value::Null,
    };

    // noise_table entries: only finite ε values
    let noise_arr: Value = match ctx.noise_table {
        Some(table) => {
            let entries: Vec<Value> = (0..ctx.num_layers)
                .filter_map(|i| {
                    table.epsilon(i).map(|eps| {
                        json!({
                            "layer": i,
                            "epsilon": eps,
                        })
                    })
                })
                .collect();
            Value::Array(entries)
        }
        None => Value::Null,
    };

    // eval_ll_output: serialize as JSON object when present, null otherwise.
    // When trajectory is set, eval_ll_output is implied null (trajectory carries
    // the per-step EvalOutputs instead).
    let eval_ll_output_val: Value = if ctx.trajectory.is_some() {
        Value::Null
    } else {
        match ctx.eval_ll_output {
            Some(output) => {
                let json_str = output.to_json().unwrap_or_else(|_| "null".to_string());
                serde_json::from_str(&json_str).unwrap_or(Value::Null)
            }
            None => Value::Null,
        }
    };

    // trajectory: serialize as an array when present.
    let trajectory_val: Value = match ctx.trajectory {
        Some(steps) => {
            let arr: Vec<Value> = steps
                .iter()
                .map(|s| {
                    let eval_json = s
                        .eval_ll_output
                        .to_json()
                        .unwrap_or_else(|_| "null".to_string());
                    let eval_val: Value = serde_json::from_str(&eval_json).unwrap_or(Value::Null);
                    json!({
                        "step": s.step,
                        "swapped_layers": s.swapped_layers,
                        "layer_added": s.layer_added,
                        "eval_ll_output": eval_val,
                    })
                })
                .collect();
            Value::Array(arr)
        }
        None => Value::Null,
    };

    // per_layer_3way: only present when `--importance-formula compare` was used.
    // Each entry merges side-by-side importance (mean_pool + shortgpt_bi) with
    // DP-LLM ε and Frobenius ε, one row per layer where mean_pool/shortgpt_bi
    // are both populated. Layers with `None` measurements (i.e. non-comparison
    // mode entries) are skipped.
    let three_way_val: Value = match (ctx.importance_table, ctx.dpllm_epsilon) {
        (Some(table), Some(eps)) => {
            let opt_to_json =
                |v: f32| -> Value { if v.is_finite() { json!(v) } else { Value::Null } };
            let entries: Vec<Value> = table
                .entries()
                .iter()
                .filter(|e| e.importance_mean_pool.is_some() || e.importance_shortgpt_bi.is_some())
                .map(|e| {
                    let dpllm = eps.get(e.layer_id).copied().unwrap_or(f32::NAN);
                    let dpllm_multi = ctx
                        .dpllm_epsilon_multi
                        .and_then(|v| v.get(e.layer_id).copied())
                        .unwrap_or(f32::NAN);
                    let dpllm_abs = ctx
                        .dpllm_epsilon_abs
                        .and_then(|v| v.get(e.layer_id).copied())
                        .unwrap_or(f32::NAN);
                    let dpllm_qcf = ctx
                        .dpllm_epsilon_qcf
                        .and_then(|v| v.get(e.layer_id).copied())
                        .unwrap_or(f32::NAN);
                    let direct_attn_f4 = ctx
                        .direct_attn_f4
                        .and_then(|v| v.get(e.layer_id).copied())
                        .unwrap_or(f32::NAN);
                    let direct_attn_f5 = ctx
                        .direct_attn_f5
                        .and_then(|v| v.get(e.layer_id).copied())
                        .unwrap_or(f32::NAN);
                    let direct_attn_f5_decode_only = ctx
                        .direct_attn_f5_decode_only
                        .and_then(|v| v.get(e.layer_id).copied())
                        .unwrap_or(f32::NAN);
                    let direct_attn_f5_prefill_decode = ctx
                        .direct_attn_f5_prefill_decode
                        .and_then(|v| v.get(e.layer_id).copied())
                        .unwrap_or(f32::NAN);
                    let frob_json: Value = ctx
                        .noise_table
                        .and_then(|t| t.epsilon(e.layer_id))
                        .map(|v| json!(v))
                        .unwrap_or(Value::Null);
                    json!({
                        "layer": e.layer_id,
                        "sublayer": format!("{:?}", e.sublayer),
                        "importance_mean_pool": e.importance_mean_pool,
                        "importance_shortgpt_bi": e.importance_shortgpt_bi,
                        "dpllm_epsilon": opt_to_json(dpllm),
                        "dpllm_epsilon_multi": opt_to_json(dpllm_multi),
                        "dpllm_epsilon_abs": opt_to_json(dpllm_abs),
                        "dpllm_epsilon_qcf": opt_to_json(dpllm_qcf),
                        "direct_attn_f4": opt_to_json(direct_attn_f4),
                        "direct_attn_f5": opt_to_json(direct_attn_f5),
                        "direct_attn_f5_decode_only": opt_to_json(direct_attn_f5_decode_only),
                        "direct_attn_f5_prefill_decode": opt_to_json(direct_attn_f5_prefill_decode),
                        "epsilon_frobenius": frob_json,
                    })
                })
                .collect();
            Value::Array(entries)
        }
        _ => Value::Null,
    };

    let schema_version = if ctx.trajectory.is_some() { 2 } else { 1 };
    let doc = json!({
        "schema_version": schema_version,
        "model_arch": ctx.model_arch,
        "model_path": ctx.model_path,
        "secondary_path": ctx.secondary_path,
        "primary_dtype": ctx.primary_dtype,
        "secondary_dtype": ctx.secondary_dtype,
        "num_layers": ctx.num_layers,
        "force_swap_ratio": ctx.force_swap_ratio,
        "swap_algorithm": ctx.swap_algorithm,
        "swap_set": ctx.swap_set,
        "swap_count": ctx.swap_set.len(),
        "qcf_swap_predicted": ctx.qcf_swap_predicted,
        "fallback_used": ctx.fallback_used,
        "importance_table": importance_arr,
        "noise_table": noise_arr,
        "ppl": ctx.ppl,
        "avg_nll": ctx.avg_nll,
        "n_eval_tokens": ctx.n_eval_tokens,
        "wall_time_s": ctx.wall_time_s,
        "warmup_tokens": ctx.warmup_tokens,
        "backend": ctx.backend,
        "kv_type": ctx.kv_type,
        "ppl_corpus": ctx.ppl_corpus,
        "eval_ll_output": eval_ll_output_val,
        "trajectory": trajectory_val,
        "per_layer_3way": three_way_val,
    });

    let json_str = serde_json::to_string_pretty(&doc)?;
    std::fs::write(path, json_str)
        .map_err(|e| anyhow::anyhow!("Failed to write QCF dump to {}: {}", path.display(), e))?;

    Ok(())
}
