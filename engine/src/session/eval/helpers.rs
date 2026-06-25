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
        // Optional diagnostic metadata (consumed only by `--dump`; absent keys → None,
        // so scoring is unaffected — INV-147).
        let gold_index = task["gold_index"].as_u64().map(|v| v as usize);
        let gold_token_positions = parse_usize_array(task, "gold_token_positions");
        let needle_token_positions = parse_usize_array(task, "needle_token_positions");

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
                gold_index,
                gold_token_positions,
                needle_token_positions,
            });
        } else if let Some(cont) = task["continuation"].as_str() {
            questions.push(crate::session::eval::EvalQuestion {
                id: task["id"].as_str().unwrap_or("unknown").to_string(),
                prompt: task["prompt"]
                    .as_str()
                    .unwrap_or(default_prompt)
                    .to_string(),
                choices: vec![cont.to_string()],
                gold_index,
                gold_token_positions,
                needle_token_positions,
            });
        }
    }
    Ok(questions)
}

/// Parse an optional `[usize, ...]` JSON array field. Returns `None` when the key
/// is absent or not an array; non-integer / negative entries are dropped.
fn parse_usize_array(task: &serde_json::Value, key: &str) -> Option<Vec<usize>> {
    task[key].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn loader_parses_optional_gold_and_needle_metadata() {
        let dir = std::env::temp_dir().join(format!("argus-loader-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("batch.json");
        std::fs::write(
            &path,
            r#"[
                {"id":"q1","prompt":"P","choices":[" A"," B"],"gold_index":1,
                 "gold_token_positions":[3,4],"needle_token_positions":[0]},
                {"id":"q2","prompt":"P2","choices":[" X"]}
            ]"#,
        )
        .unwrap();

        let args = Args::try_parse_from([
            "argus-eval",
            "--model-path",
            "/tmp/m.gguf",
            "--eval-ll",
            "--eval-batch",
            path.to_str().unwrap(),
        ])
        .unwrap();

        let qs = load_eval_questions(&args, "default").unwrap();
        assert_eq!(qs.len(), 2);
        // q1 carries all optional metadata.
        assert_eq!(qs[0].gold_index, Some(1));
        assert_eq!(qs[0].gold_token_positions, Some(vec![3, 4]));
        assert_eq!(qs[0].needle_token_positions, Some(vec![0]));
        // q2 omits the optional keys → None (backward compatible; INV-147).
        assert_eq!(qs[1].gold_index, None);
        assert_eq!(qs[1].gold_token_positions, None);
        assert_eq!(qs[1].needle_token_positions, None);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
