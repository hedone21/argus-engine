//! IMP-4 — per-output-step gold-answer attention trajectory (`--dump answer_attention_steps`).
//!
//! This is [`answer_attention`](super::answer_attention_dump) (IMP-2) **un-summed over the
//! step axis**. The v1 dump runs one reference forward over `prompt ++ gold` and SUM-collapses
//! the trailing `q_window` gold-answer query rows into a single `[layer][head][token]` snapshot;
//! the lab's H2 hypothesis ("attention moves across the generation") needs the *trajectory* —
//! the distribution at output step `t` separately from step `t+1`. So this dump captures each
//! trailing query row in its **own step slot** and emits one JSONL record per `(question, step)`.
//!
//! ## Substrate (identical to v1, one knob different)
//!
//! Same standalone pass on a fresh, uncompressed F32 [`StandardFormat`](crate::kv::standard_format)
//! reference cache, decoupled from the scoring loop (so NLL/MC are byte-identical whether it runs
//! — `INV-147`). The only difference is which PFA target is armed: v1 arms `prefill_attn`
//! (SUM-over-steps); this arms `prefill_attn_per_row` (per-step assign). The producer reads the
//! same post-softmax `scratch/denom`, so summing the per-head step slots reproduces v1 exactly.
//!
//! ## Size / head reduction
//!
//! Full `[step][layer][head][token]` is `q_window ×` the v1 dump (≈ hundreds of MB / question),
//! so the **default emits the head-mean** `attn_by_layer[step][layer][token]` (≈ `q_window/n_heads ×`
//! v1, ~tens of MB), which is exactly the lab's 3-D view (`score = layer·head mean`). The producer
//! accumulates the head-mean directly, so the in-RAM per-layer buffer is `[q_window * prefix_len]`,
//! not the per-head buffer. `--answer-attention-steps-per-head` opts into the full per-head dump
//! (`attn_by_layer_head`), with the size cost logged at startup.
//!
//! ## Backend
//!
//! PFA is materialized on both CPU and GPU. On GPU the flash prefill kernel computes the attention
//! output, then the host-mirror re-runs the same scalar `prefill_attention_scores` (its per-row arm)
//! over K read back from VRAM (see `standard_format.rs` `attention_into`), so the per-step PFA buffer
//! this dump reads is filled regardless of backend. GPU vs CPU differ only by the RoPE/KV-write
//! floating-point rounding folded into K, not by a zero-vs-nonzero gap.
//!
//! ## Row semantics (decode scope, schema 3)
//!
//! Each decode record is **self-describing** so the "which row is this?" ambiguity that mislabeled
//! a head in the lab (a duplicate-token head read as an induction head) can't recur. Two roles over
//! the same `[layer][head][token]` shape:
//! - `query_role: "gold_token_row"` (`step >= 0`) — the row IS gold token `step` being processed;
//!   its logits predict `step + 1`. Here `query_pos == gold_token_position == prompt_len + step`.
//! - `query_role: "predicting_row"` (`step == -1`, opt-in `--answer-attention-steps-predict-row`) —
//!   `query_pos == prompt_len - 1`, the row whose logits DECIDE the first gold token
//!   (`gold_token_position == prompt_len`). The producer prepends exactly this one row: the
//!   predicting row of gold `s + 1` already equals the gold-token row of `s`, so `prompt_len - 1`
//!   is the only row missing from the default window.
//!
//! `keys_span == [0, prompt_len)` (context only; rows need not sum to 1 — the self/gold keys are
//! cut). A consumer that sums the per-step rows to cross-check the v1 `answer_attention` sum must
//! include only `step >= 0` (the v1 sum never covered `prompt_len - 1`).
//!
//! `schema_version` is **3**, not 2: the `full` scope already occupies 2. It is a shape
//! discriminator (1 = original decode, 2 = full, 3 = self-describing decode), not an ordering.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use serde::Serialize;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::kv::kv_cache::{KVCache, KVLayout};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::shape::Shape;
use crate::tensor::Tensor;

use super::dump::{DUMP_ANSWER_ATTENTION_STEPS, JsonlDumpWriter};
use super::fmt_bridge::EvalCacheKind;
use super::output::EvalQuestion;

/// Schema version of the decode-default `answer_attention_steps` record (trailing gold rows over
/// the context columns), now self-describing (`query_pos`/`query_role`/`keys_span`).
///
/// **3, not 2** — `2` is already [`ANSWER_ATTENTION_STEPS_SCHEMA_VERSION_FULL`] (the `full`-scope
/// record, which landed after the row-semantics request was drafted). `1` = the original decode
/// record without the self-describing fields. The number is a shape discriminator, not an ordering:
/// consumers key row interpretation on `query_role` and use the version to reject pre-enrichment
/// dumps.
pub const ANSWER_ATTENTION_STEPS_SCHEMA_VERSION: u32 = 3;

/// Schema version of the `full`-scope record (every forward row over the full key axis).
pub const ANSWER_ATTENTION_STEPS_SCHEMA_VERSION_FULL: u32 = 2;

/// One JSONL record (per `(question, step)`) for the head-mean default.
///
/// `attn_by_layer[l][t]` = `(1/n_heads_q) Σ_h` post-softmax mass this step's gold-answer query
/// row places on context token `t` at layer `l`, over the context only (`[0, prompt_len)`).
#[derive(Debug, Serialize)]
struct AnswerAttentionStepRecord<'a> {
    kind: &'static str,
    schema_version: u32,
    question_id: &'a str,
    /// Context length = the gold continuation's `cont_start` (tokenizer-merge safe).
    prompt_len: usize,
    /// Gold continuation length (number of output steps captured).
    q_window: usize,
    gold_index: usize,
    /// Which output step. `0..q_window` for gold-token rows; `-1` for the opt-in predicting row
    /// (`--answer-attention-steps-predict-row`), which precedes step 0.
    step: i64,
    /// Absolute row index whose distribution this record holds (`prompt_len + step` for gold rows;
    /// `prompt_len - 1` for the predicting row).
    query_pos: usize,
    /// `"gold_token_row"` — the row IS gold token `step` being processed (its logits predict
    /// `step + 1`); or `"predicting_row"` — the row whose logits DECIDE the first gold token.
    query_role: &'static str,
    /// Absolute index of the gold token this row concerns. Gold rows: the token AT this row
    /// (`prompt_len + step`, so `== query_pos`). Predicting row: the token it decides (`prompt_len`).
    gold_token_position: usize,
    /// Half-open key span kept in `attn_by_layer` (`[0, prompt_len)`); rows need not sum to 1.
    keys_span: [usize; 2],
    /// `[num_hidden_layers][prompt_len]` — head-mean post-softmax over context.
    attn_by_layer: Vec<Vec<f32>>,
}

/// One JSONL record (per `(question, step)`) for `--answer-attention-steps-per-head`.
#[derive(Debug, Serialize)]
struct AnswerAttentionStepRecordPerHead<'a> {
    kind: &'static str,
    schema_version: u32,
    question_id: &'a str,
    prompt_len: usize,
    q_window: usize,
    gold_index: usize,
    /// `0..q_window` for gold-token rows; `-1` for the opt-in predicting row.
    step: i64,
    /// Absolute row index this record holds (see [`AnswerAttentionStepRecord::query_pos`]).
    query_pos: usize,
    /// `"gold_token_row"` or `"predicting_row"` (see [`AnswerAttentionStepRecord::query_role`]).
    query_role: &'static str,
    gold_token_position: usize,
    /// Half-open key span kept (`[0, prompt_len)`); rows need not sum to 1.
    keys_span: [usize; 2],
    /// `[num_hidden_layers][num_attention_heads][prompt_len]` — per-head post-softmax over context.
    attn_by_layer_head: Vec<Vec<Vec<f32>>>,
}

/// One JSONL record (per `(question, row)`) for the `full` scope, head-mean.
///
/// `row` is the forward query position (`0..seq_len`): rows `< prompt_len` are prefill, the rest
/// are decode. `attn_by_layer[l][k]` is the head-mean post-softmax mass row `row` places on key
/// `k` over the FULL key axis `[0, seq_len)`; columns `k > row` are `0` (causal), so
/// `n_valid_keys == row + 1` marks the causal fill boundary.
#[derive(Debug, Serialize)]
struct AnswerAttentionStepFullRecord<'a> {
    kind: &'static str,
    schema_version: u32,
    question_id: &'a str,
    /// Prompt/context length = the gold continuation's `cont_start` (prefill→decode boundary).
    prompt_len: usize,
    /// Full forward length (`prompt + gold`).
    seq_len: usize,
    gold_index: usize,
    /// Forward step = query position (`0..seq_len`).
    row: usize,
    /// `"prefill"` if `row < prompt_len`, else `"decode"`.
    phase: &'static str,
    /// Causal fill boundary: keys `[0, n_valid_keys)` are attended, `>=` are `0` (`= row + 1`).
    n_valid_keys: usize,
    /// `[num_hidden_layers][seq_len]` — head-mean post-softmax over the full key axis.
    attn_by_layer: Vec<Vec<f32>>,
}

/// One JSONL record (per `(question, row)`) for the `full` scope, `--answer-attention-steps-per-head`.
#[derive(Debug, Serialize)]
struct AnswerAttentionStepFullRecordPerHead<'a> {
    kind: &'static str,
    schema_version: u32,
    question_id: &'a str,
    prompt_len: usize,
    seq_len: usize,
    gold_index: usize,
    row: usize,
    phase: &'static str,
    n_valid_keys: usize,
    /// `[num_hidden_layers][num_attention_heads][seq_len]` — per-head over the full key axis.
    attn_by_layer_head: Vec<Vec<Vec<f32>>>,
}

/// The gold-choice index to forward for a question (mirrors v1).
///
/// Uses the host-supplied `gold_index` when present; for a single-choice question the gold is
/// unambiguous so default to `0`; multi-choice with no gold is ambiguous → `None` (skip loudly).
fn effective_gold_index(gold_index: Option<usize>, n_choices: usize) -> Option<usize> {
    match gold_index {
        Some(g) => Some(g),
        None if n_choices == 1 => Some(0),
        None => None,
    }
}

/// Slice step `step`'s head-mean per-layer buffer into `[layer][key]`, keeping the first
/// `keep_cols` key columns (`keep_cols <= prefix_len`).
///
/// `per_row_buf[l]` is laid out `[step * prefix_len + key_pos]`. The decode dump keeps the context
/// prefix (`keep_cols = prompt_len`); the `full` dump keeps the whole key axis
/// (`keep_cols = prefix_len = seq_len`, with causal zeros above the diagonal).
fn build_step_attn_by_layer(
    per_row_buf: &[Vec<f32>],
    step: usize,
    prefix_len: usize,
    keep_cols: usize,
) -> Vec<Vec<f32>> {
    debug_assert!(keep_cols <= prefix_len);
    per_row_buf
        .iter()
        .map(|layer| {
            let base = step * prefix_len;
            layer[base..base + keep_cols].to_vec()
        })
        .collect()
}

/// Slice step `step`'s per-head per-layer buffer into `[layer][head][key]`, keeping the first
/// `keep_cols` key columns.
///
/// `per_row_buf[l]` is laid out `[step * (n_heads_q * prefix_len) + h * prefix_len + key_pos]`.
fn build_step_attn_by_layer_head(
    per_row_buf: &[Vec<f32>],
    step: usize,
    n_heads_q: usize,
    prefix_len: usize,
    keep_cols: usize,
) -> Vec<Vec<Vec<f32>>> {
    debug_assert!(keep_cols <= prefix_len);
    per_row_buf
        .iter()
        .map(|layer| {
            (0..n_heads_q)
                .map(|h| {
                    let base = step * (n_heads_q * prefix_len) + h * prefix_len;
                    layer[base..base + keep_cols].to_vec()
                })
                .collect()
        })
        .collect()
}

/// Semantic identity of a decode-scope buffer slot — which query row it holds and how a consumer
/// should read it.
struct DecodeRowMeta {
    step: i64,
    query_pos: usize,
    query_role: &'static str,
    gold_token_position: usize,
}

/// Map a decode-scope buffer slot to its record identity.
///
/// Without the predicting row the buffer holds the `q_window` trailing gold rows: slot `s` is gold
/// token `s` at row `prompt_len + s` (`"gold_token_row"`). With `emit_predict` the producer prepends
/// row `prompt_len - 1`, so slot 0 is the predicting row (`"predicting_row"`, `step -1`, deciding
/// gold token `prompt_len`) and slots `1..` are the same gold rows shifted up by one.
fn decode_row_meta(slot: usize, emit_predict: bool, prompt_len: usize) -> DecodeRowMeta {
    if emit_predict && slot == 0 {
        DecodeRowMeta {
            step: -1,
            query_pos: prompt_len - 1,
            query_role: "predicting_row",
            gold_token_position: prompt_len,
        }
    } else {
        let step = if emit_predict { slot - 1 } else { slot };
        let pos = prompt_len + step;
        DecodeRowMeta {
            step: step as i64,
            query_pos: pos,
            query_role: "gold_token_row",
            gold_token_position: pos,
        }
    }
}

/// Run the `answer_attention_steps` (IMP-4) dump over `questions`, writing one JSONL record per
/// captured row whose gold choice is known. `per_head` = full per-head dump (default head-mean).
///
/// `full` selects the capture scope:
/// - `false` (decode default, schema 1): the trailing gold-answer rows over the context columns
///   `[0, prompt_len)` — one record per `(question, step in 0..q_window)`.
/// - `true` (full, schema 2): EVERY forward row (prefill `0..prompt_len` then decode
///   `prompt_len..seq_len`) over the FULL key axis `[0, seq_len)` (causal → zeros above the
///   diagonal) — one record per `(question, row in 0..seq_len)`. Quadratic in `seq_len`.
///
/// `predict_row` (decode scope only) prepends one extra record per question for the row that
/// DECIDES the first gold token (`query_pos = prompt_len - 1`, `query_role = "predicting_row"`,
/// `step = -1`); no-op under `full` (which already dumps every row) or when `prompt_len == 0`.
///
/// The skip rules (ambiguous gold / out-of-range / empty continuation / too long) mirror v1.
/// `out_path` is created (with parent dirs); `vocab_size` sizes the throwaway logits buffer.
#[allow(clippy::too_many_arguments)]
pub fn run_answer_attention_steps_dump(
    model: &TransformerModel,
    tokenizer: &tokenizers::Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    questions: &[EvalQuestion],
    max_seq_len: usize,
    vocab_size: usize,
    out_path: &Path,
    per_head: bool,
    full: bool,
    predict_row: bool,
) -> Result<()> {
    let n_layers = model.config.num_hidden_layers;
    let n_heads_q = model.config.num_attention_heads;
    let n_kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;

    if per_head || full {
        // per-head is ≈ n_heads_q × head-mean; `full` is quadratic in seq_len (every row × full
        // key axis). Call the upper-bound size out up front so a many-GB run is a conscious choice.
        // (Upper bound: decode rows ≤ max_seq_len, full rows = seq_len ≤ max_seq_len.)
        let heads = if per_head { n_heads_q as u128 } else { 1 };
        let approx_bytes_per_q =
            (max_seq_len as u128) * (n_layers as u128) * heads * (max_seq_len as u128) * 4;
        eprintln!(
            "[dump:answer_attention_steps] scope={}, {} — up to ~{} MB per question at \
             max_seq_len={}{}",
            if full { "full" } else { "decode" },
            if per_head { "per-head" } else { "head-mean" },
            approx_bytes_per_q / (1024 * 1024),
            max_seq_len,
            if full {
                " (quadratic in seq_len; intended for short diagnostic benches)"
            } else {
                ""
            },
        );
    }

    if predict_row && full {
        eprintln!(
            "[dump:answer_attention_steps] --answer-attention-steps-predict-row is a no-op under \
             scope=full (every forward row, incl. prompt_len-1, is already dumped)"
        );
    }

    // One fresh, uncompressed (F32) reference cache set, reused across questions (reset to pos 0
    // each time). Independent of the eval mode's caches.
    let mut ref_caches: Vec<KVCache> = crate::session::bin_setup::alloc_standard_kv_caches(
        backend,
        memory.clone(),
        n_layers,
        max_seq_len,
        max_seq_len,
        n_kv_heads,
        head_dim,
        DType::F32,
        KVLayout::SeqMajor,
    )?;

    let mut writer = JsonlDumpWriter::create(out_path)?;
    let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());

    let mut skipped = 0usize;

    for question in questions {
        let Some(gold_index) = effective_gold_index(question.gold_index, question.choices.len())
        else {
            eprintln!(
                "[dump:answer_attention_steps] {}: no `gold_index` and {} choices (gold ambiguous), \
                 skipping",
                question.id,
                question.choices.len()
            );
            skipped += 1;
            continue;
        };
        if gold_index >= question.choices.len() {
            eprintln!(
                "[dump:answer_attention_steps] {}: gold_index {} out of range ({} choices), skipping",
                question.id,
                gold_index,
                question.choices.len()
            );
            skipped += 1;
            continue;
        }

        // Tokenize prompt and prompt+gold using the SAME convention as the scoring loop
        // (full_text = prompt + choice, cont_start = min) for tokenizer-merge safety.
        let prompt_ids: Vec<u32> = tokenizer
            .encode(question.prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .to_vec();
        let full_text = format!("{}{}", question.prompt, question.choices[gold_index]);
        let full_ids: Vec<u32> = tokenizer
            .encode(full_text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .to_vec();

        let cont_start = prompt_ids.len().min(full_ids.len());
        let prompt_len = cont_start; // context boundary for the record
        let q_window = full_ids.len() - cont_start; // gold continuation length = #steps
        let seq_len = full_ids.len();

        if q_window == 0 {
            eprintln!(
                "[dump:answer_attention_steps] {}: empty gold continuation, skipping",
                question.id
            );
            skipped += 1;
            continue;
        }
        if seq_len > max_seq_len {
            eprintln!(
                "[dump:answer_attention_steps] {}: prompt+gold too long ({} > {}), skipping",
                question.id, seq_len, max_seq_len
            );
            skipped += 1;
            continue;
        }

        // Fresh reference cache for this question.
        for c in &mut ref_caches {
            c.current_pos = 0;
            c.high_water_pos = 0;
        }

        // Input tensor: full sequence (prompt ++ gold continuation), positions 0..seq_len.
        let cpu_buf = Galloc::new().alloc(seq_len * 4, DType::U8)?;
        // SAFETY: allocated exactly seq_len u32 words above.
        unsafe {
            let ptr = cpu_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(full_ids.as_ptr(), ptr, seq_len);
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, seq_len]), cpu_buf, cpu_backend.clone());
        let input_tensor = backend.copy_from(&cpu_input)?;

        // Throwaway logits buffer (last position only — logits are unused).
        let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
        let mut logits = Tensor::new(
            Shape::new(vec![1, 1, vocab_size]),
            logits_buf,
            backend.clone(),
        );

        // Per-step PFA target: per-layer flat buffer, prefix_len == seq_len (start_pos 0).
        // head-mean accumulates over heads → pre-zero; per-head assigns each (row,h) slot but
        // masked key positions stay zero → pre-zero too.
        let prefix_len = seq_len;
        // Prepend the predicting row (`prompt_len - 1`) in decode scope when armed: guard
        // `prompt_len >= 1` (no row before an empty context) and `!full` (full already dumps
        // every row). When set, the producer's row window widens by one at the front simply by
        // passing `cap = q_window + 1` (loop start `seq_len - cap == prompt_len - 1`).
        let emit_predict = predict_row && !full && prompt_len >= 1;
        // Capture window: `full` = every forward row (0..seq_len); decode = trailing gold rows
        // (plus the one predicting row when armed).
        let cap = if full {
            seq_len
        } else {
            q_window + emit_predict as usize
        };
        // Kept key columns: `full` = whole key axis [0, seq_len); decode = context [0, prompt_len).
        let keep_cols = if full { prefix_len } else { prompt_len };
        let per_layer_len = if per_head {
            cap * n_heads_q * prefix_len
        } else {
            cap * prefix_len
        };
        let mut per_row_buf: Vec<Vec<f32>> =
            (0..n_layers).map(|_| vec![0.0f32; per_layer_len]).collect();

        KVCache::forward_fmt_roundtrip(&mut ref_caches, |fmts| {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: 0,
                fmts,
                backend,
                memory: memory.as_ref(),
                logits_out: &mut logits,
                x_gen: None,
                workspace: None,
                logits_last_only: true,
                score_accumulator: None,
                query_stats_accumulator: None,
                skip_config: None,
                importance_collector: None,
                cache_self_need_scores: false,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: Some((&mut per_row_buf, cap, per_head)),
            })
        })?;

        // Stream one record per captured row so peak per-record RAM is one row's slice.
        for row in 0..cap {
            if full {
                // `full`: every forward row over the full key axis (schema 2).
                let phase = if row < prompt_len {
                    "prefill"
                } else {
                    "decode"
                };
                let n_valid_keys = row + 1; // causal: keys [0, row] attended, > row are 0.
                if per_head {
                    let attn_by_layer_head = build_step_attn_by_layer_head(
                        &per_row_buf,
                        row,
                        n_heads_q,
                        prefix_len,
                        keep_cols,
                    );
                    writer.write_record(&AnswerAttentionStepFullRecordPerHead {
                        kind: DUMP_ANSWER_ATTENTION_STEPS,
                        schema_version: ANSWER_ATTENTION_STEPS_SCHEMA_VERSION_FULL,
                        question_id: &question.id,
                        prompt_len,
                        seq_len,
                        gold_index,
                        row,
                        phase,
                        n_valid_keys,
                        attn_by_layer_head,
                    })?;
                } else {
                    let attn_by_layer =
                        build_step_attn_by_layer(&per_row_buf, row, prefix_len, keep_cols);
                    writer.write_record(&AnswerAttentionStepFullRecord {
                        kind: DUMP_ANSWER_ATTENTION_STEPS,
                        schema_version: ANSWER_ATTENTION_STEPS_SCHEMA_VERSION_FULL,
                        question_id: &question.id,
                        prompt_len,
                        seq_len,
                        gold_index,
                        row,
                        phase,
                        n_valid_keys,
                        attn_by_layer,
                    })?;
                }
            } else {
                // decode (schema 3): trailing gold rows over context columns, self-describing.
                // `row` is the buffer slot; `decode_row_meta` maps it to the record identity —
                // slot 0 is the predicting row when armed, otherwise gold rows are slot==step.
                let meta = decode_row_meta(row, emit_predict, prompt_len);
                let keys_span = [0, keep_cols]; // = [0, prompt_len) in decode scope.
                if per_head {
                    let attn_by_layer_head = build_step_attn_by_layer_head(
                        &per_row_buf,
                        row,
                        n_heads_q,
                        prefix_len,
                        keep_cols,
                    );
                    writer.write_record(&AnswerAttentionStepRecordPerHead {
                        kind: DUMP_ANSWER_ATTENTION_STEPS,
                        schema_version: ANSWER_ATTENTION_STEPS_SCHEMA_VERSION,
                        question_id: &question.id,
                        prompt_len,
                        q_window,
                        gold_index,
                        step: meta.step,
                        query_pos: meta.query_pos,
                        query_role: meta.query_role,
                        gold_token_position: meta.gold_token_position,
                        keys_span,
                        attn_by_layer_head,
                    })?;
                } else {
                    let attn_by_layer =
                        build_step_attn_by_layer(&per_row_buf, row, prefix_len, keep_cols);
                    writer.write_record(&AnswerAttentionStepRecord {
                        kind: DUMP_ANSWER_ATTENTION_STEPS,
                        schema_version: ANSWER_ATTENTION_STEPS_SCHEMA_VERSION,
                        question_id: &question.id,
                        prompt_len,
                        q_window,
                        gold_index,
                        step: meta.step,
                        query_pos: meta.query_pos,
                        query_role: meta.query_role,
                        gold_token_position: meta.gold_token_position,
                        keys_span,
                        attn_by_layer,
                    })?;
                }
            }
        }
    }

    let n = writer.finish()?;
    eprintln!(
        "[dump:answer_attention_steps] wrote {} record(s) ({} skipped) → {}",
        n,
        skipped,
        out_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_gold_index_defaults_only_for_single_choice() {
        assert_eq!(effective_gold_index(Some(1), 3), Some(1));
        assert_eq!(effective_gold_index(None, 1), Some(0));
        assert_eq!(effective_gold_index(None, 3), None);
        assert_eq!(effective_gold_index(None, 0), None);
    }

    #[test]
    fn decode_row_meta_maps_slots_to_roles() {
        let prompt_len = 5;

        // No predicting row: slot s is gold row s (query_pos == gold_token_position == prompt_len+s).
        let g0 = decode_row_meta(0, false, prompt_len);
        assert_eq!(
            (g0.step, g0.query_role, g0.query_pos, g0.gold_token_position),
            (0, "gold_token_row", 5, 5)
        );
        let g2 = decode_row_meta(2, false, prompt_len);
        assert_eq!((g2.step, g2.query_pos, g2.gold_token_position), (2, 7, 7));

        // Predicting row armed: slot 0 is the predicting row (prompt_len - 1, step -1, deciding
        // gold token prompt_len); slots 1.. are the gold rows shifted up by one.
        let p = decode_row_meta(0, true, prompt_len);
        assert_eq!(
            (p.step, p.query_role, p.query_pos, p.gold_token_position),
            (-1, "predicting_row", 4, 5)
        );
        let g0b = decode_row_meta(1, true, prompt_len);
        assert_eq!(
            (g0b.step, g0b.query_role, g0b.query_pos),
            (0, "gold_token_row", 5)
        );

        // Invariant: gold row `s` reports the SAME identity with or without the flag — the flag only
        // shifts its BUFFER slot (s → s+1), never the numbers a consumer reads.
        for s in 0..4usize {
            let without = decode_row_meta(s, false, prompt_len);
            let with = decode_row_meta(s + 1, true, prompt_len);
            assert_eq!(
                (
                    with.step,
                    with.query_pos,
                    with.query_role,
                    with.gold_token_position
                ),
                (
                    without.step,
                    without.query_pos,
                    without.query_role,
                    without.gold_token_position
                ),
            );
        }
    }

    #[test]
    fn head_mean_step_slice_picks_right_step_and_context() {
        // 2 layers, prefix_len 4, q_window 3, keep context prompt_len 2.
        // per_row_buf[l][step*prefix_len + key] = l*1000 + step*100 + key.
        let n_layers = 2;
        let prefix_len = 4;
        let q_window = 3;
        let prompt_len = 2;
        let per_row_buf: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                let mut row = vec![0.0f32; q_window * prefix_len];
                for step in 0..q_window {
                    for k in 0..prefix_len {
                        row[step * prefix_len + k] = (l * 1000 + step * 100 + k) as f32;
                    }
                }
                row
            })
            .collect();

        // Step 2, layer 1, context positions 0,1 → 1*1000 + 2*100 + {0,1}.
        let out = build_step_attn_by_layer(&per_row_buf, 2, prefix_len, prompt_len);
        assert_eq!(out.len(), n_layers, "layer dim");
        assert_eq!(out[0].len(), prompt_len, "context-token dim");
        assert_eq!(out[1], vec![1200.0, 1201.0]);
        // Context slice excludes positions >= prompt_len (the answer tokens).
        assert_eq!(out[0], vec![200.0, 201.0]);
    }

    #[test]
    fn full_axis_slice_keeps_all_columns() {
        // `full` mode slices keep_cols = prefix_len → the WHOLE key axis (incl. columns the decode
        // dump drops). Same buffer/layout as the decode test; only keep_cols differs.
        let n_layers = 2;
        let prefix_len = 4; // = seq_len in full mode
        let cap = 3; // = seq_len rows in full mode
        let per_row_buf: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                let mut row = vec![0.0f32; cap * prefix_len];
                for step in 0..cap {
                    for k in 0..prefix_len {
                        row[step * prefix_len + k] = (l * 1000 + step * 100 + k) as f32;
                    }
                }
                row
            })
            .collect();

        // Row 1, full key axis (keep_cols = prefix_len) → all 4 columns kept.
        let out = build_step_attn_by_layer(&per_row_buf, 1, prefix_len, prefix_len);
        assert_eq!(out[0].len(), prefix_len, "full key-axis width");
        // Layer 0, row 1 → 0*1000 + 1*100 + {0,1,2,3}.
        assert_eq!(out[0], vec![100.0, 101.0, 102.0, 103.0]);
        // Per-head full axis keeps all columns too.
        let n_heads_q = 1;
        let ph: Vec<Vec<f32>> = vec![{
            let mut row = vec![0.0f32; cap * n_heads_q * prefix_len];
            for k in 0..prefix_len {
                row[2 * prefix_len + k] = (200 + k) as f32; // row 2
            }
            row
        }];
        let outh = build_step_attn_by_layer_head(&ph, 2, n_heads_q, prefix_len, prefix_len);
        assert_eq!(outh[0][0], vec![200.0, 201.0, 202.0, 203.0]);
    }

    #[test]
    fn per_head_step_slice_picks_right_step_head_and_context() {
        // 2 layers, 3 heads, prefix_len 4, q_window 2, prompt_len 2.
        // buf[l][step*(H*pl) + h*pl + key] = l*1000 + step*100 + h*10 + key.
        let n_layers = 2;
        let n_heads_q = 3;
        let prefix_len = 4;
        let q_window = 2;
        let prompt_len = 2;
        let per_row_buf: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                let mut row = vec![0.0f32; q_window * n_heads_q * prefix_len];
                for step in 0..q_window {
                    for h in 0..n_heads_q {
                        for k in 0..prefix_len {
                            row[step * (n_heads_q * prefix_len) + h * prefix_len + k] =
                                (l * 1000 + step * 100 + h * 10 + k) as f32;
                        }
                    }
                }
                row
            })
            .collect();

        // Step 1, layer 1, head 2, context positions 0,1 → 1*1000 + 1*100 + 2*10 + {0,1}.
        let out = build_step_attn_by_layer_head(&per_row_buf, 1, n_heads_q, prefix_len, prompt_len);
        assert_eq!(out.len(), n_layers);
        assert_eq!(out[0].len(), n_heads_q);
        assert_eq!(out[0][0].len(), prompt_len);
        assert_eq!(out[1][2], vec![1120.0, 1121.0]);
        // head 0, step 1, layer 0 → 0 + 100 + 0 + {0,1}.
        assert_eq!(out[0][0], vec![100.0, 101.0]);
    }
}
