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
        // Model-agnostic alternative to the raw token-index arrays: a string the
        // engine later locates in its own canonical tokenization (see
        // `resolve_token_spans_from_text`). `needle` is the NIAH generator's key;
        // `needle_text` / `gold_text` are the explicit forms.
        let gold_text = task["gold_text"].as_str().map(str::to_string);
        let needle_text = task["needle_text"]
            .as_str()
            .or_else(|| task["needle"].as_str())
            .map(str::to_string);

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
                gold_text,
                needle_text,
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
                gold_text,
                needle_text,
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

/// Token indices whose byte span overlaps `[byte_start, byte_end)`.
///
/// `offsets[i] = (s, e)` is the byte range token `i` covers in the source string
/// (from the tokenizer's own offset mapping). A token is included when its range
/// overlaps the target span and is non-empty — so special tokens (BOS etc., which
/// carry the empty `(0, 0)` range) are skipped while every content token that
/// touches the span, including one straddling a boundary, is kept. Pure: no
/// tokenizer needed, so the boundary logic is unit-tested directly.
fn token_positions_overlapping(
    offsets: &[(usize, usize)],
    byte_start: usize,
    byte_end: usize,
) -> Vec<usize> {
    offsets
        .iter()
        .enumerate()
        .filter(|&(_, &(s, e))| e > s && s < byte_end && e > byte_start)
        .map(|(i, _)| i)
        .collect()
}

/// Locate `text` inside `prompt`'s canonical tokenization and return the token
/// indices it spans, or `None` if it can't be located.
///
/// The string is found as a byte substring of `prompt` (first occurrence), then
/// that byte range is mapped to token indices via the tokenizer's offset mapping
/// — we never re-encode `text` on its own, which would risk BOS / prefix-space /
/// merge mismatches at the span boundaries. `prompt` is tokenized with special
/// tokens (`encode(prompt, true)`), matching the scorer's own prompt tokenization
/// so the indices line up with the importance buffer's positions.
fn resolve_span_from_text(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    text: &str,
) -> Option<Vec<usize>> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let byte_start = prompt.find(text)?;
    let byte_end = byte_start + text.len();
    let enc = tokenizer.encode(prompt, true).ok()?;
    let positions = token_positions_overlapping(enc.get_offsets(), byte_start, byte_end);
    (!positions.is_empty()).then_some(positions)
}

/// Fill `gold_token_positions` / `needle_token_positions` from the string forms
/// (`gold_text` / `needle_text`) when the host supplied a span as text instead of
/// raw token indices.
///
/// Resolving against each question's own canonical tokenization keeps fixtures
/// model-agnostic: token indices depend on the tokenizer, but a string does not,
/// so one NIAH JSON can drive any model. Raw token positions, when present, are
/// an explicit override and are left untouched. A string that can't be located is
/// left as `None` (no positions) rather than guessed.
pub fn resolve_token_spans_from_text(
    questions: &mut [crate::session::eval::EvalQuestion],
    tokenizer: &tokenizers::Tokenizer,
) {
    for q in questions.iter_mut() {
        if q.gold_token_positions.is_none()
            && let Some(text) = q.gold_text.clone()
        {
            q.gold_token_positions = resolve_span_from_text(tokenizer, &q.prompt, &text);
        }
        if q.needle_token_positions.is_none()
            && let Some(text) = q.needle_text.clone()
        {
            q.needle_token_positions = resolve_span_from_text(tokenizer, &q.prompt, &text);
        }
    }
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

    #[test]
    fn loader_reads_needle_and_gold_text_keys() {
        let dir =
            std::env::temp_dir().join(format!("argus-loader-text-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("batch.json");
        // `needle` is the NIAH generator's key; `gold_text` the explicit form.
        std::fs::write(
            &path,
            r#"[
                {"id":"q1","prompt":"P","choices":[" A"],"needle":"the code is 42",
                 "gold_text":"42"}
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
        assert_eq!(qs[0].needle_text.as_deref(), Some("the code is 42"));
        assert_eq!(qs[0].gold_text.as_deref(), Some("42"));
        // Strings alone do not populate raw positions (that needs a tokenizer pass).
        assert_eq!(qs[0].needle_token_positions, None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn overlap_keeps_content_tokens_and_skips_specials() {
        // Tokenization of "ab XX cd" as: BOS | "ab" | " XX" | " cd", with BOS
        // carrying the empty (0,0) range that special tokens use.
        let offsets = [(0usize, 0usize), (0, 2), (2, 5), (5, 8)];
        // Needle "XX" spans bytes [3, 5) → only token 2 (" XX") overlaps.
        assert_eq!(token_positions_overlapping(&offsets, 3, 5), vec![2]);
        // A span touching two tokens keeps both (boundary-straddle is included).
        assert_eq!(token_positions_overlapping(&offsets, 1, 6), vec![1, 2, 3]);
        // The empty-range BOS is never selected, even for a whole-string span.
        assert!(!token_positions_overlapping(&offsets, 0, 8).contains(&0));
    }
}
