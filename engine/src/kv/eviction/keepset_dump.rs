//! R-P0-2: optional KV-eviction keep-set dump (harness Gate 1, set-level).
//!
//! The set-level mechanism-correctness gate compares a plugin's eviction keep-set
//! against a kvpress oracle (Jaccard). The engine otherwise never exposes the
//! keep-set, so this module appends it to a file when — and only when — the
//! `ARGUS_DUMP_KEEPSET=<path>` env var is set.
//!
//! ## Hot-path cost
//!
//! [`record`] is called from the eviction commit path: the v2 `execute_kv_plan`
//! (`super::stage_registry`) and the v3 `EngineCacheHandle::commit`
//! (`crate::kv::cache_handle`). Both pass the FINAL committed keep-set (the same
//! position-set the engine compacts to), so the dump is path-independent.
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use argus_extension_api::KeepSpec;

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

// ── In-memory keep-set capture (IMP-1 `evict_importance` dump) ──────────────
//
// Independent of the env-gated file dump above: the eval eviction hook arms this
// around a single eviction event (synchronous, single-threaded), then drains the
// captured per-layer keep-sets to co-locate them with the importance dump. The
// fast-path cost when disarmed (the default) is one relaxed atomic load → the
// eviction decision is unchanged (`INV-147`); it never allocates or locks.

static CAPTURE_ARMED: AtomicBool = AtomicBool::new(false);

/// One layer's captured keep-set, in pre-eviction absolute positions. `layer_idx` /
/// `seq_len` / `n_kv_heads` are captured diagnostic metadata (the consumer only needs
/// `keep`, but they make the capture self-describing and are asserted by tests).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct CapturedKeepSet {
    pub layer_idx: usize,
    /// Pre-eviction cache length (`current_pos` at plan-apply time).
    pub seq_len: usize,
    pub n_kv_heads: usize,
    /// `[n_kv_heads][kept positions ascending]`. For `LayerWide`, every head shares
    /// the same list (replicated); for `PerHead`, one list per head.
    pub keep: Vec<Vec<usize>>,
}

fn capture_buf() -> &'static Mutex<Vec<CapturedKeepSet>> {
    static C: OnceLock<Mutex<Vec<CapturedKeepSet>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(Vec::new()))
}

/// Arm in-memory keep-set capture (clearing any prior events).
pub(crate) fn arm_capture() {
    capture_buf()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    CAPTURE_ARMED.store(true, Ordering::Relaxed);
}

/// Disarm capture and discard any captured events (e.g. when no eviction fired).
pub(crate) fn disarm_capture() {
    CAPTURE_ARMED.store(false, Ordering::Relaxed);
    capture_buf()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}

/// Disarm and return the captured per-layer keep-sets (in record order).
pub(crate) fn drain_capture() -> Vec<CapturedKeepSet> {
    CAPTURE_ARMED.store(false, Ordering::Relaxed);
    std::mem::take(&mut *capture_buf().lock().unwrap_or_else(|e| e.into_inner()))
}

/// Serializes tests that use the global arm/record/drain capture buffer, so a concurrent test's
/// `drain_capture` (which takes ALL events) cannot steal another's captured keep-sets. Test-only.
#[cfg(test)]
pub(crate) fn capture_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Push one layer's keep-set into the capture buffer when armed. Called before any
/// compaction, so positions are absolute indices into `[0, seq_len)`.
fn capture_if_armed(cache: &KVCache, keep_spec: &KeepSpec, layer_idx: usize) {
    if !CAPTURE_ARMED.load(Ordering::Relaxed) {
        return; // disarmed (default): single atomic load, no behaviour change.
    }
    let n_kv_heads = cache.kv_heads();
    let seq_len = cache.current_pos();
    let keep: Vec<Vec<usize>> = match keep_spec {
        KeepSpec::LayerWide(k) => vec![k.clone(); n_kv_heads.max(1)],
        KeepSpec::PerHead(heads) => heads.clone(),
    };
    capture_buf()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(CapturedKeepSet {
            layer_idx,
            seq_len,
            n_kv_heads,
            keep,
        });
}

/// Record one layer's FINAL committed keep-set, then rewrite the dump file. No-op
/// (cached env read) when `ARGUS_DUMP_KEEPSET` is unset. Call BEFORE compaction so
/// `cache.current_pos()` is still the pre-eviction `seq_len` and the positions are
/// absolute indices into `[0, seq_len)`.
pub(crate) fn record(cache: &KVCache, keep_spec: &KeepSpec, layer_idx: usize, n_layers: usize) {
    // In-memory capture (IMP-1) is independent of the env-gated file dump below.
    capture_if_armed(cache, keep_spec, layer_idx);

    let Some(path) = dump_path() else {
        return; // disabled: byte-identical eviction path.
    };

    let n_kv_heads = cache.kv_heads();
    let seq_len = cache.current_pos();

    let mut per_head: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    match keep_spec {
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

    /// In-memory capture is gated on `arm_capture`: armed → `record` pushes the
    /// keep-set (LayerWide replicated per head); disarmed → nothing captured.
    /// Filters by a unique fingerprint so it tolerates any concurrent eviction in
    /// other tests writing to the shared capture buffer.
    #[test]
    fn in_memory_capture_is_arm_gated() {
        // Serialize with other capture-using tests so a concurrent drain can't steal our events.
        let _guard = capture_test_lock();
        use crate::backend::cpu::CpuBackend;
        use crate::buffer::DType;
        use crate::memory::host::shared::SharedBuffer;
        use crate::shape::Shape;
        use crate::tensor::Tensor;
        use std::sync::Arc;

        let mk = |dims: Vec<usize>| {
            let n: usize = dims.iter().product();
            let buf = Arc::new(SharedBuffer::new(n * 4, DType::F32));
            Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()))
        };
        let mut cache = KVCache::new(mk(vec![1, 8, 2, 4]), mk(vec![1, 8, 2, 4]), 8);
        cache.current_pos = 7; // unique fingerprint seq_len
        let keep = KeepSpec::LayerWide(vec![0, 1, 6]);
        let mine =
            |c: &CapturedKeepSet| c.seq_len == 7 && c.keep == vec![vec![0, 1, 6], vec![0, 1, 6]];

        // Disarmed: record captures nothing of ours.
        disarm_capture();
        record(&cache, &keep, 0, 2);
        assert_eq!(drain_capture().iter().filter(|c| mine(c)).count(), 0);

        // Armed: record captures the (replicated) keep-set per layer.
        arm_capture();
        record(&cache, &keep, 0, 2);
        record(&cache, &keep, 1, 2);
        let captured = drain_capture();
        let ours: Vec<_> = captured.into_iter().filter(mine).collect();
        assert_eq!(ours.len(), 2, "one capture per layer while armed");
        assert_eq!(ours[0].n_kv_heads, 2);
        assert_eq!(ours[0].keep, vec![vec![0, 1, 6], vec![0, 1, 6]]);

        // drain disarmed: a further record is not captured.
        record(&cache, &keep, 0, 2);
        assert_eq!(drain_capture().iter().filter(|c| mine(c)).count(), 0);
    }
}
