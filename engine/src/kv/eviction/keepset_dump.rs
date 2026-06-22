//! R-P0-2: optional KV-eviction keep-set dump (harness Gate 1, set-level).
//!
//! The set-level mechanism-correctness gate compares a plugin's eviction keep-set
//! against a kvpress oracle (Jaccard). The engine otherwise never exposes the
//! keep-set, so this module appends it to a file when — and only when — the
//! `ARGUS_DUMP_KEEPSET=<path>` env var is set.
//!
//! ## Hot-path cost
//!
//! [`record`] is called from [`execute_kv_plan`](super::stage_registry::execute_kv_plan).
//! When the env var is unset (the default), it does a single cached `OnceLock` read
//! and returns — no allocation, no lock, no I/O — so the eviction path stays
//! byte-identical (acceptance #1). When set, eviction is a warm path (fires at
//! pressure / turn boundaries, not per decoded token), so a `Mutex` + file rewrite
//! is acceptable.
//!
//! ## Schema (exactly what the harness reads)
//!
//! ```json
//! { "n_layers": L, "n_kv_heads": H, "seq_len": N,
//!   "keep": { "<layer>": { "<kv_head>": [pos, ...ascending] } } }
//! ```
//!
//! - [`KeepSpec::LayerWide`]`(v)` → every head shares the same list `v`.
//! - [`KeepSpec::PerHead`]`(vv)` → one list per head.
//!
//! `seq_len` is the pre-eviction cache length (`current_pos` at plan-apply time),
//! so the kept positions are absolute indices into `[0, seq_len)`. The file holds
//! one cross-layer snapshot: each eviction event overwrites its layer's entry, so a
//! deterministic input reproduces the same file (acceptance #3). Keys are kept in
//! a `BTreeMap` for stable ordering.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use argus_extension_api::{KVCachePlan, KeepSpec};

use crate::kv::kv_cache::KVCache;

/// Lazily-read `ARGUS_DUMP_KEEPSET` destination. `None` = disabled (default).
fn dump_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var("ARGUS_DUMP_KEEPSET")
            .ok()
            .filter(|s| !s.is_empty())
    })
    .as_deref()
}

#[derive(Default)]
struct DumpState {
    n_layers: usize,
    n_kv_heads: usize,
    seq_len: usize,
    /// layer → kv_head → kept positions (ascending).
    keep: BTreeMap<usize, BTreeMap<usize, Vec<usize>>>,
}

fn state() -> &'static Mutex<DumpState> {
    static S: OnceLock<Mutex<DumpState>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(DumpState::default()))
}

/// Record one layer's keep-set, then rewrite the dump file. No-op (cached env read)
/// when `ARGUS_DUMP_KEEPSET` is unset.
pub(crate) fn record(cache: &KVCache, plan: &KVCachePlan, layer_idx: usize, n_layers: usize) {
    let Some(path) = dump_path() else {
        return; // disabled: byte-identical eviction path.
    };

    let n_kv_heads = cache.kv_heads();
    let seq_len = cache.current_pos();

    let mut per_head: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    match &plan.keep {
        KeepSpec::LayerWide(keep) => {
            for h in 0..n_kv_heads.max(1) {
                per_head.insert(h, keep.clone());
            }
        }
        KeepSpec::PerHead(heads) => {
            for (h, keep) in heads.iter().enumerate() {
                per_head.insert(h, keep.clone());
            }
        }
    }

    let mut st = state().lock().unwrap_or_else(|e| e.into_inner());
    st.n_layers = n_layers;
    st.n_kv_heads = n_kv_heads;
    st.seq_len = seq_len;
    st.keep.insert(layer_idx, per_head);
    write_file(path, &st);
}

/// Build the harness-facing JSON object from accumulated state. Pure (no I/O / env /
/// global), so the schema contract is unit-testable.
fn build_json(st: &DumpState) -> serde_json::Value {
    let keep_obj: serde_json::Map<String, serde_json::Value> = st
        .keep
        .iter()
        .map(|(layer, heads)| {
            let head_obj: serde_json::Map<String, serde_json::Value> = heads
                .iter()
                .map(|(h, positions)| (h.to_string(), serde_json::json!(positions)))
                .collect();
            (layer.to_string(), serde_json::Value::Object(head_obj))
        })
        .collect();
    serde_json::json!({
        "n_layers": st.n_layers,
        "n_kv_heads": st.n_kv_heads,
        "seq_len": st.seq_len,
        "keep": keep_obj,
    })
}

fn write_file(path: &str, st: &DumpState) {
    if let Ok(s) = serde_json::to_string(&build_json(st)) {
        let _ = std::fs::write(path, s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LayerWide replicates one keep-list to every head; PerHead keeps per-head lists.
    /// Asserts the exact `{n_layers, n_kv_heads, seq_len, keep:{layer:{head:[...]}}}` shape.
    #[test]
    fn build_json_matches_harness_schema() {
        let mut st = DumpState {
            n_layers: 2,
            n_kv_heads: 2,
            seq_len: 5,
            keep: BTreeMap::new(),
        };
        // layer 0: LayerWide-style — both heads share [0, 1, 4].
        let mut l0 = BTreeMap::new();
        l0.insert(0, vec![0, 1, 4]);
        l0.insert(1, vec![0, 1, 4]);
        st.keep.insert(0, l0);
        // layer 1: PerHead-style — head 0 keeps [0,2], head 1 keeps [0,3,4].
        let mut l1 = BTreeMap::new();
        l1.insert(0, vec![0, 2]);
        l1.insert(1, vec![0, 3, 4]);
        st.keep.insert(1, l1);

        let v = build_json(&st);
        assert_eq!(v["n_layers"], 2);
        assert_eq!(v["n_kv_heads"], 2);
        assert_eq!(v["seq_len"], 5);
        assert_eq!(v["keep"]["0"]["0"], serde_json::json!([0, 1, 4]));
        assert_eq!(v["keep"]["0"]["1"], serde_json::json!([0, 1, 4]));
        assert_eq!(v["keep"]["1"]["0"], serde_json::json!([0, 2]));
        assert_eq!(v["keep"]["1"]["1"], serde_json::json!([0, 3, 4]));

        // Deterministic: BTreeMap ordering → byte-identical serialization across builds.
        assert_eq!(
            serde_json::to_string(&build_json(&st)).unwrap(),
            serde_json::to_string(&v).unwrap()
        );
    }
}
