//! Phase 4-B: eval-ll question loader + warmup text builder.
//!
//! `bin/generate.rs`에서 이동. lift-and-shift: 본문 변경 없음.
//! `argus_engine::X` → `crate::X` 만 적용. pub 노출하여 main()의 quant-window eval-ll
//! path (l.246)에서도 호출 가능.

use crate::session::cli::Args;

/// Load and normalize eval questions from `--eval-batch` or `--eval-continuation`.
///
/// Produces a `Vec<EvalQuestion>` in grouped format (prompt + choices).
pub fn load_eval_questions(
    args: &Args,
    default_prompt: &str,
) -> anyhow::Result<Vec<crate::session::eval::EvalQuestion>> {
    let raw_tasks: Vec<serde_json::Value> = if let Some(ref path) = args.eval_batch {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to open eval batch {}: {}", path, e))?;
        serde_json::from_reader(file)?
    } else {
        let cont = args.eval_continuation.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--eval-ll requires --eval-continuation or --eval-batch")
        })?;
        vec![serde_json::json!({
            "id": "single",
            "prompt": default_prompt,
            "choices": [cont],
        })]
    };

    let mut questions: Vec<crate::session::eval::EvalQuestion> = Vec::new();
    for task in &raw_tasks {
        if let Some(choices) = task["choices"].as_array() {
            questions.push(crate::session::eval::EvalQuestion {
                id: task["id"].as_str().unwrap_or("unknown").to_string(),
                prompt: task["prompt"]
                    .as_str()
                    .unwrap_or(default_prompt)
                    .to_string(),
                choices: choices
                    .iter()
                    .filter_map(|c| c.as_str().map(|s| s.to_string()))
                    .collect(),
            });
        } else if let Some(cont) = task["continuation"].as_str() {
            questions.push(crate::session::eval::EvalQuestion {
                id: task["id"].as_str().unwrap_or("unknown").to_string(),
                prompt: task["prompt"]
                    .as_str()
                    .unwrap_or(default_prompt)
                    .to_string(),
                choices: vec![cont.to_string()],
            });
        }
    }
    Ok(questions)
}

/// Build a warmup token sequence from the eval-ll question set.
///
/// Concatenates the `prompt` fields of the questions (separated by `"\n\n"`),
/// tokenizes the result, and returns at most `max_tokens` token IDs.
/// If fewer tokens are produced than requested, a warning is emitted but the
/// function succeeds — the caller handles the reduced warmup gracefully.
///
/// Returns an empty Vec when tokenization fails entirely (non-fatal).
pub fn build_eval_ll_warmup_text(
    questions: &[crate::session::eval::EvalQuestion],
    max_tokens: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<u32> {
    // Join question prompts.
    let combined: String = questions
        .iter()
        .map(|q| q.prompt.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    if combined.is_empty() {
        eprintln!("[QCF-dump] WARNING: all eval questions have empty prompts; warmup skipped");
        return Vec::new();
    }

    let enc = match tokenizer.encode(combined.as_str(), true) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "[QCF-dump] WARNING: warmup tokenize error: {}; warmup skipped",
                e
            );
            return Vec::new();
        }
    };

    let ids: Vec<u32> = enc.get_ids().iter().take(max_tokens).copied().collect();

    if ids.len() < max_tokens {
        eprintln!(
            "[QCF-dump] WARNING: only {} warmup tokens available (requested {}); \
             using all available tokens",
            ids.len(),
            max_tokens
        );
    }

    ids
}
