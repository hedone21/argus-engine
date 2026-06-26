//! Generic eval diagnostic-dump scheme (`--dump-dir` + `--dump <kinds>`).
//!
//! A single repeatable/CSV flag (`--dump <kind>[,<kind>...]`) selects which
//! read-only diagnostic dumps to emit; each enabled kind writes one JSONL file
//! `<dump-dir>/<kind>.jsonl`, one JSON record per question. This replaces the
//! per-dump flag-per-diagnostic design so the shared CLI surface (which all four
//! ARGUS binaries pay for) does not balloon as new diagnostics are added — a new
//! dump kind registers a *name* in [`KNOWN_DUMP_KINDS`], not a new clap field.
//!
//! Kind names are validated at startup against [`KNOWN_DUMP_KINDS`] (a runtime
//! registry miss, not a clap closed-set), so adding a kind never edits the CLI.
//! The kind string is the single source of truth: it is both the `--dump` token
//! and the record's `"kind"` field, so producers and the lab-side consumer agree
//! by construction.
//!
//! All dumps are read-only with respect to scoring (NLL / MC) — `INV-147`: with
//! no `--dump` flag the production path is untouched.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

/// IMP-2 — full-cache gold-answer attention (technique-independent ground truth).
pub const DUMP_ANSWER_ATTENTION: &str = "answer_attention";

/// IMP-1 — compression-time per-(token × layer × KV-head) importance (technique-side).
pub const DUMP_EVICT_IMPORTANCE: &str = "evict_importance";

/// Dump kinds this build knows how to produce. Validated at runtime (NOT a clap
/// closed value-set) so registering a new kind never touches the CLI surface.
pub const KNOWN_DUMP_KINDS: &[&str] = &[DUMP_ANSWER_ATTENTION, DUMP_EVICT_IMPORTANCE];

/// True if `kind` is a dump this build can produce.
pub fn is_known_dump_kind(kind: &str) -> bool {
    KNOWN_DUMP_KINDS.contains(&kind)
}

/// Startup warning for the `evict_importance` + no-KV-budget footgun.
///
/// `evict_importance` only emits on a real eviction event, which in eval-ll fires
/// only when a global KV budget (`--kv-budget` / `--kv-budget-ratio`) is set — the
/// policy's own `keep_ratio` does **not** set a budget. With no budget the run is
/// full-prefill, nothing is evicted, and the dump is silently empty. Returns the
/// warning to print when that combination is requested, else `None`.
pub fn evict_importance_empty_dump_warning(
    evict_dump_enabled: bool,
    has_kv_budget: bool,
) -> Option<&'static str> {
    (evict_dump_enabled && !has_kv_budget).then_some(
        "[dump:evict_importance] WARNING: no KV budget set \
         (--kv-budget / --kv-budget-ratio) → eviction will not fire → \
         the evict_importance dump will be empty",
    )
}

/// Validate a set of requested `--dump` kinds, erroring on the first unknown one.
pub fn validate_dump_kinds(kinds: &[String]) -> Result<()> {
    for k in kinds {
        if !is_known_dump_kind(k) {
            anyhow::bail!(
                "unknown --dump kind '{}'. known kinds: {}",
                k,
                KNOWN_DUMP_KINDS.join(", ")
            );
        }
    }
    Ok(())
}

/// Append-only JSONL writer: one compact JSON record per line. Creates the parent
/// directory on open. Shared by every dump kind so the on-disk format is uniform
/// (one record per question, joinable across kinds by `question_id`).
pub struct JsonlDumpWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    count: usize,
}

impl JsonlDumpWriter {
    /// Open `path` for writing, truncating any existing file and creating parent
    /// directories as needed.
    pub fn create(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dump directory {}", parent.display()))?;
        }
        let file =
            File::create(&path).with_context(|| format!("create dump file {}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
            path,
            count: 0,
        })
    }

    /// Serialize `rec` as one compact JSON line.
    pub fn write_record<T: Serialize>(&mut self, rec: &T) -> Result<()> {
        serde_json::to_writer(&mut self.writer, rec)
            .with_context(|| format!("serialize dump record to {}", self.path.display()))?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        Ok(())
    }

    /// Number of records written so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Output path (for log messages).
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flush and return the record count.
    pub fn finish(mut self) -> Result<usize> {
        self.writer.flush()?;
        Ok(self.count)
    }
}

// ── IMP-1: evict_importance (technique-side per-(layer, KV-head) importance) ──

/// Per-eviction-event metadata for the `prefill_streaming` (variant b) multi-event
/// dump. Present (`Some`) only for streaming records (`schema_version: 2`); `None` for
/// the single-shot `post_prefill_probe` / `prefill_end` records (`schema_version: 1`,
/// byte-identical to before).
///
/// Because streaming compacts the live cache on each eviction, the cache's slot space
/// is reindexed; these fields keep the dump in the **original** prompt token-index
/// space so it joins against `answer_attention` (which is in original positions).
#[derive(Debug, Clone)]
pub struct EvictEventMeta {
    /// 1-based eviction event index within the question (strictly increasing).
    pub eviction_event: usize,
    /// Tokens ingested so far (original prompt position reached) when this eviction
    /// fired. `>= cache_pos_before` once a prior event has compacted the cache.
    pub prefill_pos: usize,
    /// Resident cache length immediately *before* this eviction (`= prompt_len`).
    pub cache_pos_before: usize,
    /// Resident cache length immediately *after* this eviction (`<= budget`).
    pub cache_pos_after: usize,
    /// Original token index of each resident slot *before* this eviction, in slot
    /// order (`len == cache_pos_before`). `importance_flat[i]` and the per-(layer,head)
    /// rows are indexed by slot `i`, so `resident_positions[i]` is that slot's original
    /// prompt position — the join key from slot-space importance to original-space
    /// `kept_positions` / `evicted_positions` / `answer_attention`.
    pub resident_positions: Vec<usize>,
}

/// Everything the eviction hook can produce for one `evict_importance` record.
/// The eval loop adds the per-question metadata (`question_id`, gold/needle token
/// positions) and writes the record — the hook has no question context or writer.
#[derive(Debug, Clone)]
pub struct EvictImportanceSnapshot {
    /// Pre-eviction cache length (the context length the policy ranked over).
    pub prompt_len: usize,
    /// Effective KV budget at eviction.
    pub budget: usize,
    /// Keep-ratio parameter of the policy.
    pub keep_ratio: f32,
    /// Eviction policy name (the actual registered policy, e.g. `"h2o"`).
    pub technique: String,
    /// Kept context positions (ascending, technique-agnostic — from the policy's plan).
    /// **Original** token-index space (mapped back through the resident map for
    /// streaming; identity for the single-shot modes that never reindex).
    pub kept_positions: Vec<usize>,
    /// Evicted context positions = resident positions minus `kept_positions`, in
    /// **original** token-index space.
    pub evicted_positions: Vec<usize>,
    /// The flat per-token importance the policy actually ranked on `[prompt_len]`
    /// (slot-indexed; see [`EvictEventMeta::resident_positions`] for the streaming map).
    pub importance_flat: Vec<f32>,
    /// Non-collapsed per-(layer, KV-head, token) importance `[L][Hkv][prompt_len]`.
    pub importance_by_layer_head: Vec<Vec<Vec<f32>>>,
    /// `Some` for `prefill_streaming` per-event records (`schema_version: 2`); `None`
    /// for the single-shot modes (`schema_version: 1`, unchanged).
    pub event: Option<EvictEventMeta>,
}

/// Evicted positions = `[0, seq_len)` minus the kept set (ascending).
pub fn complement_positions(kept: &[usize], seq_len: usize) -> Vec<usize> {
    let mut keep_flags = vec![false; seq_len];
    for &p in kept {
        if p < seq_len {
            keep_flags[p] = true;
        }
    }
    (0..seq_len).filter(|&p| !keep_flags[p]).collect()
}

/// Reshape the producer's flat `[total_layers * n_kv_heads * max_seq_len]` per-layer-head
/// buffer into `[n_layers][n_kv_heads][prompt_len]`, keeping only the context prefix
/// (`prompt_len <= max_seq_len`) per `(layer, KV-head)` row.
pub fn reshape_layer_head(
    buf: &[f32],
    n_layers: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    prompt_len: usize,
) -> Vec<Vec<Vec<f32>>> {
    debug_assert!(prompt_len <= max_seq_len);
    (0..n_layers)
        .map(|l| {
            (0..n_kv_heads)
                .map(|h| {
                    let base = (l * n_kv_heads + h) * max_seq_len;
                    let end = (base + prompt_len).min(buf.len());
                    if base <= end && base < buf.len() {
                        buf[base..end].to_vec()
                    } else {
                        vec![0.0; prompt_len]
                    }
                })
                .collect()
        })
        .collect()
}

/// Write one `evict_importance` JSONL record, joining the hook snapshot with the
/// per-question metadata (`question_id`, gold/needle token positions).
///
/// A snapshot with no [`event`](EvictImportanceSnapshot::event) is a single-shot record
/// (`post_prefill_probe` / `prefill_end`) and is serialized **byte-identically** to
/// before (`schema_version: 1`, `INV-147`). A snapshot with an event is a
/// `prefill_streaming` per-event record (`schema_version: 2`) carrying the extra
/// eviction-event fields.
pub fn write_evict_importance_record(
    writer: &mut JsonlDumpWriter,
    snapshot: &EvictImportanceSnapshot,
    question_id: &str,
    gold_token_positions: Option<&[usize]>,
    needle_token_positions: Option<&[usize]>,
) -> Result<()> {
    match &snapshot.event {
        None => write_evict_importance_record_v1(
            writer,
            snapshot,
            question_id,
            gold_token_positions,
            needle_token_positions,
        ),
        Some(event) => write_evict_importance_record_v2(
            writer,
            snapshot,
            event,
            question_id,
            gold_token_positions,
            needle_token_positions,
        ),
    }
}

/// Single-shot record (`schema_version: 1`) — unchanged on-disk format.
fn write_evict_importance_record_v1(
    writer: &mut JsonlDumpWriter,
    snapshot: &EvictImportanceSnapshot,
    question_id: &str,
    gold_token_positions: Option<&[usize]>,
    needle_token_positions: Option<&[usize]>,
) -> Result<()> {
    #[derive(Serialize)]
    struct Record<'a> {
        kind: &'static str,
        schema_version: u32,
        question_id: &'a str,
        prompt_len: usize,
        budget: usize,
        keep_ratio: f32,
        technique: &'a str,
        evicted_positions: &'a [usize],
        kept_positions: &'a [usize],
        gold_token_positions: Option<&'a [usize]>,
        needle_token_positions: Option<&'a [usize]>,
        importance_flat: &'a [f32],
        importance_by_layer_head: &'a [Vec<Vec<f32>>],
    }
    writer.write_record(&Record {
        kind: DUMP_EVICT_IMPORTANCE,
        schema_version: EVICT_IMPORTANCE_SCHEMA_VERSION,
        question_id,
        prompt_len: snapshot.prompt_len,
        budget: snapshot.budget,
        keep_ratio: snapshot.keep_ratio,
        technique: &snapshot.technique,
        evicted_positions: &snapshot.evicted_positions,
        kept_positions: &snapshot.kept_positions,
        gold_token_positions,
        needle_token_positions,
        importance_flat: &snapshot.importance_flat,
        importance_by_layer_head: &snapshot.importance_by_layer_head,
    })
}

/// Streaming per-event record (`schema_version: 2`). One per (question, eviction
/// event); positions are in original token-index space and `keep_ratio` is `null`
/// (the absolute budget `B`, not a ratio, is the trigger).
fn write_evict_importance_record_v2(
    writer: &mut JsonlDumpWriter,
    snapshot: &EvictImportanceSnapshot,
    event: &EvictEventMeta,
    question_id: &str,
    gold_token_positions: Option<&[usize]>,
    needle_token_positions: Option<&[usize]>,
) -> Result<()> {
    #[derive(Serialize)]
    struct Record<'a> {
        kind: &'static str,
        schema_version: u32,
        question_id: &'a str,
        eviction_event: usize,
        prefill_pos: usize,
        cache_pos_before: usize,
        cache_pos_after: usize,
        budget: usize,
        /// Always `null`: streaming evicts on an absolute budget, not a keep-ratio.
        keep_ratio: Option<f32>,
        technique: &'a str,
        evicted_positions: &'a [usize],
        kept_positions: &'a [usize],
        resident_positions: &'a [usize],
        gold_token_positions: Option<&'a [usize]>,
        needle_token_positions: Option<&'a [usize]>,
        importance_flat: &'a [f32],
        importance_by_layer_head: &'a [Vec<Vec<f32>>],
    }
    writer.write_record(&Record {
        kind: DUMP_EVICT_IMPORTANCE,
        schema_version: EVICT_IMPORTANCE_SCHEMA_VERSION_STREAMING,
        question_id,
        eviction_event: event.eviction_event,
        prefill_pos: event.prefill_pos,
        cache_pos_before: event.cache_pos_before,
        cache_pos_after: event.cache_pos_after,
        budget: snapshot.budget,
        keep_ratio: None,
        technique: &snapshot.technique,
        evicted_positions: &snapshot.evicted_positions,
        kept_positions: &snapshot.kept_positions,
        resident_positions: &event.resident_positions,
        gold_token_positions,
        needle_token_positions,
        importance_flat: &snapshot.importance_flat,
        importance_by_layer_head: &snapshot.importance_by_layer_head,
    })
}

/// Schema version of the single-shot `evict_importance` record
/// (`post_prefill_probe` / `prefill_end`).
pub const EVICT_IMPORTANCE_SCHEMA_VERSION: u32 = 1;

/// Schema version of the `prefill_streaming` multi-event `evict_importance` record.
pub const EVICT_IMPORTANCE_SCHEMA_VERSION_STREAMING: u32 = 2;

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct Rec {
        id: u32,
        v: Vec<f32>,
    }

    #[test]
    fn known_kinds_contains_answer_attention() {
        assert!(is_known_dump_kind(DUMP_ANSWER_ATTENTION));
        assert!(is_known_dump_kind("answer_attention"));
    }

    #[test]
    fn unknown_kind_is_rejected() {
        assert!(!is_known_dump_kind("nope"));
        let err = validate_dump_kinds(&["answer_attention".into(), "nope".into()]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nope"), "msg: {msg}");
        assert!(msg.contains("answer_attention"), "known list shown: {msg}");
    }

    #[test]
    fn empty_kinds_validate_ok() {
        assert!(validate_dump_kinds(&[]).is_ok());
    }

    #[test]
    fn evict_importance_warns_only_without_budget() {
        // Requested + no budget → warn (the dump would be silently empty).
        assert!(evict_importance_empty_dump_warning(true, false).is_some());
        // Requested + a budget set → fine, no warning.
        assert!(evict_importance_empty_dump_warning(true, true).is_none());
        // Not requested → never warn, budget or not.
        assert!(evict_importance_empty_dump_warning(false, false).is_none());
        assert!(evict_importance_empty_dump_warning(false, true).is_none());
    }

    #[test]
    fn writer_emits_one_line_per_record_and_creates_dirs() {
        let dir = std::env::temp_dir().join(format!("argus-dump-test-{}", std::process::id()));
        let path = dir.join("nested").join("answer_attention.jsonl");
        // Parent dirs do not exist yet — create() must make them.
        let mut w = JsonlDumpWriter::create(&path).expect("create writer");
        w.write_record(&Rec {
            id: 1,
            v: vec![0.5, 0.25],
        })
        .unwrap();
        w.write_record(&Rec { id: 2, v: vec![] }).unwrap();
        assert_eq!(w.count(), 2);
        let n = w.finish().unwrap();
        assert_eq!(n, 2);

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "one JSON record per line");
        // Each line must be standalone valid JSON (line-delimited, not pretty).
        let r0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(r0["id"], 1);
        assert_eq!(r0["v"][0], 0.5);
        let r1: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(r1["id"], 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn known_kinds_contains_evict_importance() {
        assert!(is_known_dump_kind(DUMP_EVICT_IMPORTANCE));
        assert!(is_known_dump_kind("evict_importance"));
    }

    #[test]
    fn complement_positions_excludes_kept() {
        assert_eq!(complement_positions(&[0, 2, 4], 6), vec![1, 3, 5]);
        // Out-of-range kept entries are ignored.
        assert_eq!(complement_positions(&[1, 9], 4), vec![0, 2, 3]);
        // Empty kept → everything evicted.
        assert_eq!(complement_positions(&[], 3), vec![0, 1, 2]);
        // Kept ∩ evicted = ∅, and their union is [0, seq_len).
        let kept = [0, 3];
        let ev = complement_positions(&kept, 5);
        assert!(kept.iter().all(|k| !ev.contains(k)));
        assert_eq!(kept.len() + ev.len(), 5);
    }

    #[test]
    fn reshape_layer_head_slices_context_per_layer_head() {
        // 2 layers, 2 KV-heads, max_seq_len 4, keep prompt_len 2.
        let n_layers = 2;
        let n_kv_heads = 2;
        let max_seq_len = 4;
        let prompt_len = 2;
        // buf[(l*kvh + h)*max_seq + pos] = l*100 + h*10 + pos
        let mut buf = vec![0.0f32; n_layers * n_kv_heads * max_seq_len];
        for l in 0..n_layers {
            for h in 0..n_kv_heads {
                for p in 0..max_seq_len {
                    buf[(l * n_kv_heads + h) * max_seq_len + p] = (l * 100 + h * 10 + p) as f32;
                }
            }
        }
        let out = reshape_layer_head(&buf, n_layers, n_kv_heads, max_seq_len, prompt_len);
        assert_eq!(out.len(), n_layers);
        assert_eq!(out[0].len(), n_kv_heads);
        assert_eq!(out[0][0].len(), prompt_len);
        // Layer 1, KV-head 1, context positions 0,1 → 1*100 + 1*10 + {0,1}.
        assert_eq!(out[1][1], vec![110.0, 111.0]);
        // Context slice drops positions >= prompt_len.
        assert_eq!(out[0][0], vec![0.0, 1.0]);
    }

    #[test]
    fn evict_importance_record_has_join_key_and_dims() {
        let dir = std::env::temp_dir().join(format!("argus-evict-test-{}", std::process::id()));
        let path = dir.join("evict_importance.jsonl");
        let mut w = JsonlDumpWriter::create(&path).unwrap();
        let snap = EvictImportanceSnapshot {
            prompt_len: 3,
            budget: 2,
            keep_ratio: 0.5,
            technique: "h2o".into(),
            kept_positions: vec![0, 2],
            evicted_positions: vec![1],
            importance_flat: vec![0.9, 0.1, 0.7],
            importance_by_layer_head: vec![vec![vec![0.9, 0.1, 0.7], vec![0.8, 0.2, 0.6]]],
            event: None,
        };
        write_evict_importance_record(&mut w, &snap, "q1", Some(&[2]), None).unwrap();
        w.finish().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let r: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
        assert_eq!(r["kind"], "evict_importance");
        assert_eq!(r["schema_version"], 1);
        assert_eq!(r["question_id"], "q1"); // join key
        assert_eq!(r["prompt_len"], 3);
        assert_eq!(r["technique"], "h2o");
        assert_eq!(r["kept_positions"], serde_json::json!([0, 2]));
        assert_eq!(r["evicted_positions"], serde_json::json!([1]));
        assert_eq!(r["gold_token_positions"], serde_json::json!([2]));
        assert!(r["needle_token_positions"].is_null());
        assert_eq!(r["importance_flat"], serde_json::json!([0.9, 0.1, 0.7]));
        // [L][Hkv][prompt_len] dims.
        assert_eq!(r["importance_by_layer_head"].as_array().unwrap().len(), 1);
        assert_eq!(
            r["importance_by_layer_head"][0].as_array().unwrap().len(),
            2
        );
        assert_eq!(
            r["importance_by_layer_head"][0][0]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        // v1 record carries NO streaming-only fields.
        assert!(r.get("eviction_event").is_none());
        assert!(r.get("cache_pos_before").is_none());
        assert!(r.get("resident_positions").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn streaming_record_is_schema_v2_with_event_fields_in_original_space() {
        let dir = std::env::temp_dir().join(format!("argus-evict-v2-test-{}", std::process::id()));
        let path = dir.join("evict_importance.jsonl");
        let mut w = JsonlDumpWriter::create(&path).unwrap();
        // Second eviction event: the cache already compacted once, so slot space is
        // reindexed. `resident_positions` maps slot i → original prompt position.
        // Slots {0,1,2} hold original tokens {3,7,9}; the policy keeps slots {0,2}.
        let snap = EvictImportanceSnapshot {
            prompt_len: 3, // == cache_pos_before
            budget: 2,
            keep_ratio: 0.9, // ignored by v2 writer (emits null)
            technique: "h2o".into(),
            kept_positions: vec![3, 9], // ORIGINAL indices of kept slots {0,2}
            evicted_positions: vec![7], // ORIGINAL index of evicted slot {1}
            importance_flat: vec![0.9, 0.1, 0.7],
            importance_by_layer_head: vec![vec![vec![0.9, 0.1, 0.7]]],
            event: Some(EvictEventMeta {
                eviction_event: 2,
                prefill_pos: 12,
                cache_pos_before: 3,
                cache_pos_after: 2,
                resident_positions: vec![3, 7, 9],
            }),
        };
        write_evict_importance_record(&mut w, &snap, "q1", Some(&[9]), Some(&[3])).unwrap();
        w.finish().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let r: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
        assert_eq!(r["kind"], "evict_importance");
        assert_eq!(r["schema_version"], 2);
        assert_eq!(r["question_id"], "q1");
        assert_eq!(r["eviction_event"], 2);
        assert_eq!(r["prefill_pos"], 12);
        assert_eq!(r["cache_pos_before"], 3);
        assert_eq!(r["cache_pos_after"], 2);
        assert_eq!(r["budget"], 2);
        assert!(r["keep_ratio"].is_null(), "v2 keep_ratio is null");
        // Positions are ORIGINAL token indices, kept/evicted disjoint, both ⊆ resident.
        assert_eq!(r["kept_positions"], serde_json::json!([3, 9]));
        assert_eq!(r["evicted_positions"], serde_json::json!([7]));
        assert_eq!(r["resident_positions"], serde_json::json!([3, 7, 9]));
        assert_eq!(r["gold_token_positions"], serde_json::json!([9]));
        assert_eq!(r["needle_token_positions"], serde_json::json!([3]));
        // importance dims [L][Hkv][cache_pos_before].
        assert_eq!(r["importance_flat"].as_array().unwrap().len(), 3);
        assert_eq!(
            r["importance_by_layer_head"][0][0]
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }
}
