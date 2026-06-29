//! IMP-2 — full-cache gold-answer attention dump (`--dump answer_attention`).
//!
//! For each eval question with a known gold choice, this runs ONE extra reference
//! forward over `prompt_ids ++ gold_continuation_ids` on a **fresh, uncompressed
//! `StandardFormat` KV cache** (F32, no eviction, no quant) and captures the
//! post-softmax attention the gold answer's query rows place on every context
//! position, per `(layer, attention-head pre-GQA)`. The result is the
//! technique-independent ground truth the lab correlates against a technique's
//! compression-time importance (IMP-1).
//!
//! This is a **standalone pass**, completely decoupled from the scoring loop
//! (`run_eval_ll_generic`): it owns a separate reference cache and never touches
//! the caches used to compute NLL / MC. Hence scoring is byte-identical whether
//! or not the dump runs (`INV-147`) — by construction, not by a flag branch in
//! the hot path.
//!
//! ## Substrate
//!
//! The prefill-attention (PFA) side channel already exists on the transformer
//! (`TransformerModelForwardArgs::prefill_attn`): when armed it SUM-accumulates,
//! per pre-GQA head, the post-softmax probabilities of the trailing `q_window`
//! query rows over the full prefix, into a per-layer `[n_heads_q * prefix_len]`
//! buffer. The eval loop calls `model.forward_into` directly (not the
//! `ModelForward` driver), so we arm PFA simply by passing `Some((&mut buf, qw))`
//! — no registered consumer stage needed.
//!
//! ## CPU-only
//!
//! PFA is computed only on the CPU attention path (the GPU flash kernel
//! short-circuits before it). The caller (`argus-eval`) fail-fasts on a non-CPU
//! backend; this module additionally asserts it defensively.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, bail};
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

use super::dump::{DUMP_ANSWER_ATTENTION, JsonlDumpWriter};
use super::fmt_bridge::EvalCacheKind;
use super::output::EvalQuestion;

/// One JSONL record (per question) for the `answer_attention` dump.
///
/// `answer_attn_by_layer_head[l][h][t]` = summed post-softmax mass the `q_window`
/// gold-answer rows place on context token `t` at layer `l`, pre-GQA head `h`.
/// Heads are `num_attention_heads` (pre-GQA); `t` ranges over the context only
/// (`[0, prompt_len)`), so per `(l, h)` the sum is `<= q_window` (the remaining
/// mass falls on the answer tokens themselves, which are excluded here).
#[derive(Debug, Serialize)]
pub struct AnswerAttentionRecord {
    pub kind: &'static str,
    pub schema_version: u32,
    pub question_id: String,
    /// Context length = the gold continuation's `cont_start` (tokenizer-merge safe).
    pub prompt_len: usize,
    /// Number of trailing gold-answer query rows summed (gold continuation length).
    pub q_window: usize,
    pub gold_index: usize,
    /// `[num_hidden_layers][num_attention_heads][prompt_len]`.
    pub answer_attn_by_layer_head: Vec<Vec<Vec<f32>>>,
}

/// Schema version of the `answer_attention` record (bump on breaking changes).
pub const ANSWER_ATTENTION_SCHEMA_VERSION: u32 = 1;

/// Slice the flat per-layer PFA buffer into `[layer][head][context-token]`.
///
/// `pfa_buf[l]` is laid out `[h * prefix_len + key_pos]` over `n_heads_q` pre-GQA
/// heads and `prefix_len` key positions; we keep only the context prefix
/// `[0, prompt_len)` per head (`prompt_len <= prefix_len`).
fn build_answer_attn_by_layer_head(
    pfa_buf: &[Vec<f32>],
    n_heads_q: usize,
    prefix_len: usize,
    prompt_len: usize,
) -> Vec<Vec<Vec<f32>>> {
    debug_assert!(prompt_len <= prefix_len);
    pfa_buf
        .iter()
        .map(|layer| {
            (0..n_heads_q)
                .map(|h| {
                    let base = h * prefix_len;
                    layer[base..base + prompt_len].to_vec()
                })
                .collect()
        })
        .collect()
}

/// The gold-choice index to forward for a question's answer-attention record.
///
/// Uses the host-supplied `gold_index` when present. When it is absent but the
/// question has exactly one choice, the gold is unambiguous, so default to `0`
/// rather than skip — this keeps single-choice (e.g. continuation) batches from
/// silently producing zero records. Multi-choice questions with no `gold_index`
/// are genuinely ambiguous and return `None` (the caller skips them, loudly).
fn effective_gold_index(gold_index: Option<usize>, n_choices: usize) -> Option<usize> {
    match gold_index {
        Some(g) => Some(g),
        None if n_choices == 1 => Some(0),
        None => None,
    }
}

/// Run the `answer_attention` (IMP-2) dump over `questions`, writing one JSONL
/// record per question whose gold choice is known. The gold is the supplied
/// `gold_index`, or `0` for a single-choice question (unambiguous). Questions
/// that remain ambiguous (no `gold_index` and >1 choice), or have an empty gold
/// continuation, or are longer than `max_seq_len`, are skipped (with a named
/// warning) so a partially-annotated batch still produces a usable dump.
///
/// `out_path` is created (with parent dirs); `vocab_size` sizes the throwaway
/// logits buffer (logits are not used — only the PFA side channel is read).
#[allow(clippy::too_many_arguments)]
pub fn run_answer_attention_dump(
    model: &TransformerModel,
    tokenizer: &tokenizers::Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    questions: &[EvalQuestion],
    max_seq_len: usize,
    vocab_size: usize,
    out_path: &Path,
) -> Result<()> {
    if backend.is_gpu() {
        // PFA is CPU-only (the GPU flash kernel short-circuits before it). The
        // caller already fail-fasts; assert defensively so a buffer of zeros is
        // never silently emitted as "ground truth".
        bail!(
            "answer_attention dump requires a CPU backend (prefill-attention capture is CPU-only)"
        );
    }

    let n_layers = model.config.num_hidden_layers;
    let n_heads_q = model.config.num_attention_heads;
    let n_kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;

    // One fresh, uncompressed (F32) reference cache set, reused across questions
    // (reset to position 0 each time). Independent of the eval mode's caches.
    let mut ref_caches: Vec<KVCache> = crate::session::bin_setup::alloc_standard_kv_caches(
        backend,
        memory.clone(),
        n_layers,
        max_seq_len, // initial capacity = full preallocation
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
        // Single-choice questions need no `gold_index` (the gold is unambiguous);
        // a multi-choice question without one is genuinely ambiguous → skip with a
        // named warning rather than silently inflating the "skipped" count (R3).
        let Some(gold_index) = effective_gold_index(question.gold_index, question.choices.len())
        else {
            eprintln!(
                "[dump:answer_attention] {}: no `gold_index` and {} choices (gold ambiguous), \
                 skipping",
                question.id,
                question.choices.len()
            );
            skipped += 1;
            continue;
        };
        if gold_index >= question.choices.len() {
            eprintln!(
                "[dump:answer_attention] {}: gold_index {} out of range ({} choices), skipping",
                question.id,
                gold_index,
                question.choices.len()
            );
            skipped += 1;
            continue;
        }

        // Tokenize prompt and prompt+gold using the SAME convention as the scoring
        // loop (full_text = prompt + choice, cont_start = min) for tokenizer-merge
        // safety — do NOT re-encode the gold string independently.
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
        let q_window = full_ids.len() - cont_start; // gold continuation length
        let seq_len = full_ids.len();

        if q_window == 0 {
            eprintln!(
                "[dump:answer_attention] {}: empty gold continuation, skipping",
                question.id
            );
            skipped += 1;
            continue;
        }
        if seq_len > max_seq_len {
            eprintln!(
                "[dump:answer_attention] {}: prompt+gold too long ({} > {}), skipping",
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

        // PFA target: per-layer [n_heads_q * prefix_len], prefix_len == seq_len
        // (start_pos 0). The producer SUM-accumulates over the trailing q_window
        // rows, so it must be pre-zeroed.
        let prefix_len = seq_len;
        let mut pfa_buf: Vec<Vec<f32>> = (0..n_layers)
            .map(|_| vec![0.0f32; n_heads_q * prefix_len])
            .collect();

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
                prefill_attn: Some((&mut pfa_buf, q_window)),
                prefill_attn_per_row: None,
            })
        })?;

        let answer_attn_by_layer_head =
            build_answer_attn_by_layer_head(&pfa_buf, n_heads_q, prefix_len, prompt_len);

        writer.write_record(&AnswerAttentionRecord {
            kind: DUMP_ANSWER_ATTENTION,
            schema_version: ANSWER_ATTENTION_SCHEMA_VERSION,
            question_id: question.id.clone(),
            prompt_len,
            q_window,
            gold_index,
            answer_attn_by_layer_head,
        })?;
    }

    let n = writer.finish()?;
    eprintln!(
        "[dump:answer_attention] wrote {} record(s) ({} skipped) → {}",
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
        // Explicit gold is always honored.
        assert_eq!(effective_gold_index(Some(1), 3), Some(1));
        // Single choice → gold unambiguous, default to 0 instead of skipping.
        assert_eq!(effective_gold_index(None, 1), Some(0));
        // Multi-choice with no gold → ambiguous → skip (None).
        assert_eq!(effective_gold_index(None, 3), None);
        // Degenerate zero-choice question is not single-choice → still skip.
        assert_eq!(effective_gold_index(None, 0), None);
    }

    #[test]
    fn record_builder_shape_and_context_slice() {
        // 2 layers, 3 pre-GQA heads, prefix_len 4, keep context prompt_len 2.
        let n_heads_q = 3;
        let prefix_len = 4;
        let prompt_len = 2;
        // Build a buffer where value encodes (layer, head, key_pos) so we can
        // verify the slice picks the right elements.
        let pfa_buf: Vec<Vec<f32>> = (0..2)
            .map(|l| {
                let mut row = vec![0.0f32; n_heads_q * prefix_len];
                for h in 0..n_heads_q {
                    for k in 0..prefix_len {
                        row[h * prefix_len + k] = (l * 100 + h * 10 + k) as f32;
                    }
                }
                row
            })
            .collect();

        let out = build_answer_attn_by_layer_head(&pfa_buf, n_heads_q, prefix_len, prompt_len);
        assert_eq!(out.len(), 2, "layer dim");
        assert_eq!(out[0].len(), n_heads_q, "head dim");
        assert_eq!(out[0][0].len(), prompt_len, "context-token dim");
        // Layer 1, head 2, context positions 0,1 → 1*100 + 2*10 + {0,1}.
        assert_eq!(out[1][2], vec![120.0, 121.0]);
        // Context slice excludes positions >= prompt_len (the answer tokens).
        assert_eq!(out[0][1], vec![10.0, 11.0]);
    }

    #[test]
    fn record_builder_context_sum_is_at_most_q_window() {
        // Synthesize a head whose FULL prefix row sums to q_window (post-softmax
        // property: q_window rows each summing to 1). The emitted context slice
        // must sum to <= q_window — never more.
        let n_heads_q = 1;
        let prefix_len = 5;
        let prompt_len = 3;
        let q_window = 2.0f32;
        // Distribute q_window mass across all 5 key positions.
        let full_row: Vec<f32> = vec![0.5, 0.5, 0.4, 0.3, 0.3]; // sum = 2.0 = q_window
        let pfa_buf = vec![full_row.clone()];
        let out = build_answer_attn_by_layer_head(&pfa_buf, n_heads_q, prefix_len, prompt_len);
        let context_sum: f32 = out[0][0].iter().sum();
        let full_sum: f32 = full_row.iter().sum();
        assert!((full_sum - q_window).abs() < 1e-6, "fixture invariant");
        assert!(
            context_sum <= q_window + 1e-6,
            "context sum {context_sum} must be <= q_window {q_window}"
        );
        assert!(
            context_sum < full_sum,
            "context slice drops answer-token mass"
        );
    }
}
