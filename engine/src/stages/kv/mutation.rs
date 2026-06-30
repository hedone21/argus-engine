//! `KVMutationDriverStage` — the engine driver for imperative [`KVMutationStage`] callbacks.
//!
//! The mutation twin of [`FormatReencodeStage`](super::format_reencode::FormatReencodeStage): it
//! mirrors that stage's take_inner / per-layer / put_inner UER, but drives the imperative
//! [`CacheHandle`](argus_extension_api::CacheHandle) surface — a [`KVMutationStage`] callback stages
//! mutation ops on a per-layer [`EngineCacheHandle`], and the engine commits the transaction once per
//! layer ([`EngineCacheHandle::commit`]).
//!
//! Because every handle op routes to the SAME low-level executors
//! (`compact_keep_positions` / `apply_weighted_merges` / `apply_format_plan`), a keep applied through
//! this driver is **byte-identical** to the same keep applied directly (the byte-identity oracle).
//! The handle commit path also emits the `keepset_dump` observability side-channel (P0-2), so a
//! handle-driven eviction appears in `ARGUS_DUMP_KEEPSET` / the in-memory capture identically to a
//! direct executor call.
//!
//! Read/mutate aliasing: a mutation stage reads its cache state through a [`SnapshotStageCtx`] backed
//! by OWNED snapshots captured in the entry frame BEFORE the handle is built — scalars + budget
//! (`target_len`, P0-3a), flat importance / per-head scores (P0-3b), and raw K/V dequant snapshots
//! (P0-3c) — and mutates through the `&mut` handle. The ctx borrows those loop-local snapshots rather
//! than the cache, so the `&dyn StageCtx` read view and the `&mut dyn CacheHandle` write view never
//! alias — and both observe the entry frame (the cache is untouched until `commit`).
//!
//! Production wiring (P0-5c/P0-6): `build_standard_loop` resolves the chosen `eviction <policy>` to a
//! v3 stage via `resolve_mutation_driver` (`find_mutation_stage`) and, when one exists, registers this
//! driver in [`with_pressure_gate`](KVMutationDriverStage::with_pressure_gate) mode as a faithful
//! drop-in for the v2 `EvictionStage` (MUTUALLY EXCLUSIVE at KvMutate). The bare `new()` path stays the
//! direct-construction surface the gates below exercise.

use std::cell::OnceCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use argus_extension_api::{
    CrossLayerStageCtx, KVMutationStage, MutationPhase, StageCaps, StageCtx, TensorDtype,
    TensorHandle, TensorKind, TensorShape,
};
use argus_shared::Level;

use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::kv::cache_handle::{EngineCacheHandle, EngineModelCacheHandle};
use crate::kv::dequant::{dequantize_k, dequantize_v};
use crate::kv::eviction_handler::MIN_EVICT_TOKENS;
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};

/// Maps the additive [`MutationPhase`] to the engine [`LifecyclePhase`] the driver fires at.
fn lifecycle_of(phase: MutationPhase) -> LifecyclePhase {
    match phase {
        MutationPhase::PrefillEnd => LifecyclePhase::PrefillEnd,
        MutationPhase::KvMutate => LifecyclePhase::KvMutate,
    }
}

/// A per-(kv_head, pos) scalar snapshot handle (`tensor(Scores)` / `tensor(AttnWeights)`), reading
/// from a driver-owned slice in the `[n_kv_heads * max_seq]` row-major layout the score accumulator
/// produces (stride `max_seq`). The mirror of `stage_registry::ScalarHandle`, but over an owned
/// snapshot instead of the live cache, so it never aliases the `&mut` handle.
struct ScalarSnapHandle<'a> {
    data: &'a [f32],
    rows: usize,
    max_seq: usize,
}
impl TensorHandle for ScalarSnapHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.rows,
            cols: 1,
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        out[0] = self
            .data
            .get(kv_head * self.max_seq + row)
            .copied()
            .unwrap_or(0.0);
    }
}

/// A per-(kv_head, pos) dequantized K/V snapshot handle (`tensor(Key)` / `tensor(Value)`), reading
/// from a driver-owned `[n_kv_heads * rows * head_dim]` f32 snapshot (idx `(kv_head*rows + row)*hd`).
/// The mirror of `stage_registry::{KeyHandle,ValueHandle}`, but over an owned dequant snapshot
/// captured BEFORE the handle is built (entry frame, T-3) — so a value-aware stage (caote) or a
/// merge-similarity stage (d2o) can read raw K/V while the `&mut` [`CacheHandle`] holds the cache.
struct KvSnapHandle<'a> {
    data: &'a [f32],
    rows: usize,
    head_dim: usize,
}
impl TensorHandle for KvSnapHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.rows,
            cols: self.head_dim,
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        let base = (kv_head * self.rows + row) * self.head_dim;
        let n = self.head_dim.min(out.len());
        if base + n <= self.data.len() {
            out[..n].copy_from_slice(&self.data[base..base + n]);
        } else {
            out[..n].fill(0.0);
        }
    }
}

/// The owning twin of [`KvSnapHandle`] (`Vec` instead of `&[f32]`), for a snapshot materialized lazily
/// inside the ctx (read seam: `tensor(Value)` on demand, see [`LazyValueSrc`]). Same `(kv_head*rows +
/// row)*head_dim` layout and `read_row` semantics; it owns its buffer because there is no caller-frame
/// slice to borrow — the snapshot is built on first access and cached in the ctx.
struct OwnedKvSnapHandle {
    data: Vec<f32>,
    rows: usize,
    head_dim: usize,
}
impl TensorHandle for OwnedKvSnapHandle {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.rows,
            cols: self.head_dim,
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        let base = (kv_head * self.rows + row) * self.head_dim;
        let n = self.head_dim.min(out.len());
        if base + n <= self.data.len() {
            out[..n].copy_from_slice(&self.data[base..base + n]);
        } else {
            out[..n].fill(0.0);
        }
    }
}

/// A deferred source for a `tensor(Value)` dequant snapshot (read seam only — [`SnapshotStageCtx::
/// for_read`]). Holds a host-resident cache ref + geometry; the V snapshot is materialized into the
/// ctx's `value_cell` only on the first `tensor(Value)` access and skipped entirely when the read
/// stage never reads Value (e.g. Quest reads only Key/Query). The materialized bytes are identical to
/// an eager `dequant_snapshot(.., is_k = false)` — this only defers the work, it does not change it.
struct LazyValueSrc<'a> {
    cache: &'a KVCache,
    rows: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

/// A per-q-head prefill-attention snapshot handle (`tensor(PrefillAttention)`), reading from an owned
/// `[n_heads_q * cols]` f32 slice (row-major, head outer / pos inner; `per_head: false` — the `kv_head`
/// arg is ignored, the mirror of `stage_registry`'s PFA handle). The slice lives outside the cache (the
/// shared PFA producer cell), so it never aliases the `&mut` [`CacheHandle`].
struct PfaSnapHandle<'a> {
    data: &'a [f32],
    rows: usize, // n_heads_q (pre-GQA)
    cols: usize, // prefix_len
}
impl TensorHandle for PfaSnapHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.rows,
            cols: self.cols,
            per_head: false,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, _kv_head: usize, out: &mut [f32]) {
        let base = row * self.cols;
        let n = self.cols.min(out.len());
        if base + n <= self.data.len() {
            out[..n].copy_from_slice(&self.data[base..base + n]);
        } else {
            out[..n].fill(0.0);
        }
    }
}

/// A current-step Q snapshot handle (`tensor(Query)`), reading a borrowed `[n_kv_heads * head_dim]`
/// f32 slice (`data[kv_head*head_dim + d]`, `shape = {rows:1, cols:head_dim, per_head:true}`). The
/// read-stage's faithful current-Q (Quest); the mirror of the deleted `stage_registry::QueryHandle`.
struct QuerySnapHandle<'a> {
    data: &'a [f32],
    head_dim: usize,
}
impl TensorHandle for QuerySnapHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: 1,
            cols: self.head_dim,
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, _row: usize, kv_head: usize, out: &mut [f32]) {
        let base = kv_head * self.head_dim;
        let n = self.head_dim.min(out.len());
        if base + n <= self.data.len() {
            out[..n].copy_from_slice(&self.data[base..base + n]);
        } else {
            out[..n].fill(0.0);
        }
    }
}

/// A per-kv-head Q running mean/var snapshot handle (`tensor(QueryStats)`), reading a borrowed
/// `[n_kv_heads * 2 * head_dim]` f32 slice (`data[kv_head*2*head_dim + stat_row*head_dim + d]`,
/// `stat_row 0 = mean / 1 = var`; `shape = {rows:2, cols:head_dim, per_head:true}`). Dormant fallback
/// (no production producer); the mirror of the deleted `stage_registry::QueryStatsHandle`.
struct QueryStatsSnapHandle<'a> {
    data: &'a [f32],
    head_dim: usize,
}
impl TensorHandle for QueryStatsSnapHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: 2, // row0 = mean, row1 = var.
            cols: self.head_dim,
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        let base = kv_head * 2 * self.head_dim + row * self.head_dim;
        let n = self.head_dim.min(out.len());
        if base + n <= self.data.len() {
            out[..n].copy_from_slice(&self.data[base..base + n]);
        } else {
            out[..n].fill(0.0);
        }
    }
}

/// Dequantize the resident region of `cache`'s K (or V) into an owned `[n_kv_heads * rows * head_dim]`
/// f32 snapshot (layout matching [`KvSnapHandle`]). Host-only — the caller must gate on
/// `!kv_on_device()` (a device buffer has no host pointer to dequantize from).
pub(crate) fn dequant_snapshot(
    cache: &KVCache,
    rows: usize,
    n_kv_heads: usize,
    head_dim: usize,
    is_k: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_kv_heads * rows * head_dim];
    let mut tmp = vec![0.0f32; head_dim];
    for kv_head in 0..n_kv_heads {
        for row in 0..rows {
            if is_k {
                dequantize_k(cache, row, kv_head, head_dim, &mut tmp);
            } else {
                dequantize_v(cache, row, kv_head, head_dim, &mut tmp);
            }
            let base = (kv_head * rows + row) * head_dim;
            out[base..base + head_dim].copy_from_slice(&tmp);
        }
    }
    out
}

/// A read [`StageCtx`] backed by OWNED/borrowed snapshots — never the live cache on the MUTATION path,
/// so it coexists with a `&mut` [`CacheHandle`] over the same cache (a v3 callback holds both at once).
/// The scalars are copied from the entry frame; the signal slices are borrowed from driver-loop locals
/// captured in that same pre-callback frame (T-3): `importance` (flat per-token) and the per-head
/// `Scores` / `AttnWeights` from the score accumulator, plus raw `Key`/`Value` dequant snapshots for
/// value-aware (caote) / merge (d2o) stages. A score-free stage gets all-`None` signals and reads only
/// the scalars + budget.
///
/// The READ seam ([`Self::for_read`]) is the one exception to "never the live cache": it holds a
/// host-resident cache ref in `value_lazy` to dequantize `tensor(Value)` on demand (no `&mut` handle
/// exists on that path, so there is no aliasing to avoid). `value_cell` caches that lazy snapshot.
pub struct SnapshotStageCtx<'a> {
    current_pos: usize,
    target_len: usize,
    layer_idx: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    on_device: bool,
    importance: Option<&'a [f32]>,
    score_handle: Option<ScalarSnapHandle<'a>>,
    attn_handle: Option<ScalarSnapHandle<'a>>,
    key_handle: Option<KvSnapHandle<'a>>,
    value_handle: Option<KvSnapHandle<'a>>,
    prefill_attn_handle: Option<PfaSnapHandle<'a>>,
    /// (read seam) faithful current-step Q (`tensor(Query)`) — supplied only by [`Self::for_read`].
    query_handle: Option<QuerySnapHandle<'a>>,
    /// (read seam) dormant Q running mean/var (`tensor(QueryStats)`) — supplied only by [`Self::for_read`].
    query_stats_handle: Option<QueryStatsSnapHandle<'a>>,
    /// (read seam) deferred `tensor(Value)` source — supplied only by [`Self::for_read`], `None` on the
    /// mutation path (which uses the eager `value_handle`). Materialized lazily into `value_cell`.
    value_lazy: Option<LazyValueSrc<'a>>,
    /// Cache for the lazily-materialized `tensor(Value)` snapshot (see `value_lazy`). Built at most once.
    value_cell: OnceCell<OwnedKvSnapHandle>,
}

impl<'a> SnapshotStageCtx<'a> {
    /// A scalars-only ctx (no signals) over `cache`'s entry frame — copies, no borrow of `cache` held.
    /// Used by score-free stages and as the no-signal reference in tests.
    pub fn from_cache(
        cache: &KVCache,
        target_len: usize,
        layer_idx: usize,
        n_layers: usize,
    ) -> Self {
        Self {
            current_pos: cache.current_pos(),
            target_len,
            layer_idx,
            n_layers,
            n_kv_heads: cache.kv_heads(),
            head_dim: cache.head_dim(),
            on_device: cache.k_buffer.buffer().is_gpu_buffer(),
            importance: None,
            score_handle: None,
            attn_handle: None,
            key_handle: None,
            value_handle: None,
            prefill_attn_handle: None,
            query_handle: None,
            query_stats_handle: None,
            value_lazy: None,
            value_cell: OnceCell::new(),
        }
    }

    /// A ctx exposing `tensor(PrefillAttention)` over a per-layer PFA slice (`[n_heads_q * prefix_len]`,
    /// the shared producer cell) — used by [`PrefillKeepSetStage`](super::prefill_keepset). Copies the
    /// cache scalars (no `cache` borrow held) and borrows ONLY `pfa`, so the returned ctx coexists with
    /// the `&mut` [`EngineCacheHandle`] built from the same cache (`pfa` lives outside the cache).
    pub fn for_prefill_attn(
        cache: &KVCache,
        target_len: usize,
        layer_idx: usize,
        n_layers: usize,
        pfa: &'a [f32],
        n_heads_q: usize,
    ) -> Self {
        Self {
            prefill_attn_handle: Some(PfaSnapHandle {
                data: pfa,
                rows: n_heads_q,
                cols: cache.current_pos(),
            }),
            ..Self::from_cache(cache, target_len, layer_idx, n_layers)
        }
    }

    /// A read-stage ctx exposing `tensor(Key)`/`tensor(Value)` plus the optional faithful current-Q
    /// (`tensor(Query)`) and dormant Q running-stats (`tensor(QueryStats)`). Used by
    /// `StandardFormat::read_plan` (the W-DEVKV / Quest read seam). `cache` must be host-resident.
    /// `key_snap` is an owned `[n_kv_heads * rows * head_dim]` K dequant snapshot the caller built via
    /// [`dequant_snapshot`] (same layout as [`KvSnapHandle`]). The V snapshot is **deferred**: it is
    /// dequantized from `cache` only if the read stage actually reads `tensor(Value)` (Quest does not),
    /// then cached — byte-identical to an eager `dequant_snapshot(cache, .., is_k = false)`, just lazy.
    pub fn for_read(
        cache: &'a KVCache,
        key_snap: &'a [f32],
        query: Option<&'a [f32]>,
        query_stats: Option<&'a [f32]>,
    ) -> Self {
        let rows = cache.current_pos();
        let n_kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();
        Self {
            key_handle: Some(KvSnapHandle {
                data: key_snap,
                rows,
                head_dim,
            }),
            value_lazy: Some(LazyValueSrc {
                cache,
                rows,
                n_kv_heads,
                head_dim,
            }),
            query_handle: query.map(|data| QuerySnapHandle { data, head_dim }),
            query_stats_handle: query_stats.map(|data| QueryStatsSnapHandle { data, head_dim }),
            ..Self::from_cache(cache, 0, 0, 1)
        }
    }
}

impl StageCtx for SnapshotStageCtx<'_> {
    fn current_pos(&self) -> usize {
        self.current_pos
    }
    fn target_len(&self) -> usize {
        self.target_len
    }
    fn layer_idx(&self) -> usize {
        self.layer_idx
    }
    fn n_layers(&self) -> usize {
        self.n_layers
    }
    fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn kv_on_device(&self) -> bool {
        self.on_device
    }
    fn importance(&self) -> Option<&[f32]> {
        self.importance
    }
    fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
        match kind {
            TensorKind::Scores => self.score_handle.as_ref().map(|h| h as &dyn TensorHandle),
            TensorKind::AttnWeights => self.attn_handle.as_ref().map(|h| h as &dyn TensorHandle),
            TensorKind::Key => self.key_handle.as_ref().map(|h| h as &dyn TensorHandle),
            TensorKind::Value => {
                // Mutation path: eager `value_handle`. Read path: materialize the deferred V snapshot
                // from `value_lazy` on first access (skipped entirely when the stage never reads Value)
                // and cache it in `value_cell` — byte-identical to an eager dequant, just lazy.
                if let Some(h) = self.value_handle.as_ref() {
                    Some(h as &dyn TensorHandle)
                } else if let Some(src) = self.value_lazy.as_ref() {
                    let h = self.value_cell.get_or_init(|| OwnedKvSnapHandle {
                        data: dequant_snapshot(
                            src.cache,
                            src.rows,
                            src.n_kv_heads,
                            src.head_dim,
                            false,
                        ),
                        rows: src.rows,
                        head_dim: src.head_dim,
                    });
                    Some(h as &dyn TensorHandle)
                } else {
                    None
                }
            }
            TensorKind::PrefillAttention => self
                .prefill_attn_handle
                .as_ref()
                .map(|h| h as &dyn TensorHandle),
            TensorKind::Query => self.query_handle.as_ref().map(|h| h as &dyn TensorHandle),
            TensorKind::QueryStats => self
                .query_stats_handle
                .as_ref()
                .map(|h| h as &dyn TensorHandle),
        }
    }
}

/// Drive a single layer's [`KVMutationStage`] callback through an [`EngineCacheHandle`] and commit it,
/// returning whether the commit mutated any bytes. The shared per-layer core of the v3 mutation path,
/// used by [`KVMutationDriverStage`] (the pipeline driver) AND the score-aware eviction adapter
/// (`StageBackedPolicy`) so both build the read ctx + handle identically.
///
/// Builds an owned-snapshot [`SnapshotStageCtx`] in the entry frame (T-3): the score signals
/// (`importance` flat / `head_scores` per-head / `last_attn`) are borrowed from caller-owned slices,
/// and raw K/V dequant snapshots are built locally ONLY when `caps.reads` declares the kind AND the
/// cache is host-resident — so the read view never aliases the `&mut` handle. A panic in the
/// (untrusted) stage is caught (mirror of P0-5a) so the caller's UER rewrap always runs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_mutation_layer(
    stage: &dyn KVMutationStage,
    caps: &StageCaps,
    cache: &mut KVCache,
    layer_idx: usize,
    n_layers: usize,
    target_len: usize,
    importance: Option<&[f32]>,
    head_scores: Option<&[f32]>,
    last_attn: Option<&[f32]>,
) -> anyhow::Result<bool> {
    let current_pos = cache.current_pos();
    if current_pos == 0 {
        return Ok(false);
    }
    let max_seq = cache.max_seq_len;
    let n_kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();
    let on_device = cache.k_buffer.buffer().is_gpu_buffer();
    // Per-layer raw K/V dequant snapshots (P0-3c) — only when the stage declares the read AND the cache
    // is host-resident (a device buffer has no host pointer). Captured BEFORE the handle (entry frame).
    let want_key = !on_device && caps.reads.contains(&TensorKind::Key);
    let want_value = !on_device && caps.reads.contains(&TensorKind::Value);
    let key_snap: Option<Vec<f32>> =
        want_key.then(|| dequant_snapshot(cache, current_pos, n_kv_heads, head_dim, true));
    let value_snap: Option<Vec<f32>> =
        want_value.then(|| dequant_snapshot(cache, current_pos, n_kv_heads, head_dim, false));
    let sctx = SnapshotStageCtx {
        current_pos,
        target_len,
        layer_idx,
        n_layers,
        n_kv_heads,
        head_dim,
        on_device,
        importance,
        score_handle: head_scores.map(|data| ScalarSnapHandle {
            data,
            rows: current_pos,
            max_seq,
        }),
        attn_handle: last_attn.map(|data| ScalarSnapHandle {
            data,
            rows: current_pos,
            max_seq,
        }),
        key_handle: key_snap.as_deref().map(|data| KvSnapHandle {
            data,
            rows: current_pos,
            head_dim,
        }),
        value_handle: value_snap.as_deref().map(|data| KvSnapHandle {
            data,
            rows: current_pos,
            head_dim,
        }),
        prefill_attn_handle: None,
        query_handle: None,
        query_stats_handle: None,
        value_lazy: None,
        value_cell: OnceCell::new(),
    };
    let handle = EngineCacheHandle::new(cache, layer_idx, n_layers);
    let driven = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        move || -> anyhow::Result<bool> {
            let mut handle = handle;
            stage
                .on_phase(&sctx, &mut handle)
                .map_err(|e| anyhow::anyhow!("mutation stage '{}' failed: {e}", stage.name()))?;
            handle.commit()
        },
    ));
    match driven {
        Ok(r) => r,
        Err(_) => Err(anyhow::anyhow!(
            "mutation stage '{}' panicked during on_phase/commit",
            stage.name()
        )),
    }
}

/// A whole-model read view ([`CrossLayerStageCtx`]) over ALL layers' caches. Holds an owned,
/// host-mirrored Key snapshot per layer (built via [`dequant_snapshot`] — the same bytes the per-layer
/// [`SnapshotStageCtx`] exposes through [`KvSnapHandle`]), plus the engine's absolute positions. The
/// snapshots are OWNED (no cache borrow retained), so this read ctx coexists with the `&mut`
/// [`EngineModelCacheHandle`] over the same caches (the whole-model twin of the per-layer
/// read/mutate-aliasing split). Built in the entry frame (T-3).
pub struct EngineCrossLayerStageCtx<'a> {
    n_layers: usize,
    current_pos: usize,
    target_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// The engine's absolute position of each resident slot (len == current_pos; identity before any
    /// eviction, survivors' original positions after). The plugin reads it via `abs_position`.
    positions: &'a [usize],
    /// One host-mirrored Key snapshot per layer (index == layer), `[kv_head][slot][head_dim]` layout
    /// (the [`OwnedKvSnapHandle`] convention) — `layer_tensor(layer, Key)` returns `&key_handles[layer]`.
    key_handles: Vec<OwnedKvSnapHandle>,
}

impl<'a> EngineCrossLayerStageCtx<'a> {
    /// Build the whole-model read ctx from all layers' (host-resident) caches + the engine's absolute
    /// `positions`. Each layer's Key is dequantized into an owned host-mirror snapshot here, so the
    /// `on_whole_model` closure can stream `(layer, kv_head, slot)` keys without holding a cache borrow.
    /// `caches` must be NON-EMPTY and uniform (every layer the same `current_pos`/geometry — the
    /// caller's precondition); the geometry is read from the first layer.
    ///
    /// HOST-ONLY: [`dequant_snapshot`] reads through the host pointer, so on a device-resident cache the
    /// caller must host-mirror (read the device buffer back) before constructing this ctx — mirroring
    /// the `!kv_on_device()` gate in [`drive_mutation_layer`].
    pub fn new(caches: &[KVCache], target_len: usize, positions: &'a [usize]) -> Self {
        let c0 = &caches[0];
        let current_pos = c0.current_pos();
        let n_kv_heads = c0.kv_heads();
        let head_dim = c0.head_dim();
        let key_handles = caches
            .iter()
            .map(|cache| OwnedKvSnapHandle {
                data: dequant_snapshot(cache, current_pos, n_kv_heads, head_dim, true),
                rows: current_pos,
                head_dim,
            })
            .collect();
        Self {
            n_layers: caches.len(),
            current_pos,
            target_len,
            n_kv_heads,
            head_dim,
            positions,
            key_handles,
        }
    }
}

impl CrossLayerStageCtx for EngineCrossLayerStageCtx<'_> {
    fn n_layers(&self) -> usize {
        self.n_layers
    }
    fn current_pos(&self) -> usize {
        self.current_pos
    }
    fn target_len(&self) -> usize {
        self.target_len
    }
    fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn abs_position(&self, slot: usize) -> usize {
        // Identity fallback for an out-of-range slot (defensive; positions.len() == current_pos holds
        // by construction).
        self.positions.get(slot).copied().unwrap_or(slot)
    }
    fn layer_tensor(&self, layer: usize, kind: TensorKind) -> Option<&dyn TensorHandle> {
        match kind {
            TensorKind::Key => self.key_handles.get(layer).map(|h| h as &dyn TensorHandle),
            _ => None,
        }
    }
}

/// Drive a WHOLE-MODEL [`KVMutationStage::on_whole_model`] callback over all layers' caches through an
/// [`EngineModelCacheHandle`] and commit it, returning whether the commit mutated any bytes. The
/// cross-layer sibling of [`drive_mutation_layer`]: it builds an owned, host-mirrored
/// [`EngineCrossLayerStageCtx`] (so the read view never aliases the `&mut` model handle), runs the
/// stage once over all layers, and commits the fanned-out keep-set. `positions` are the engine's
/// absolute positions (len == current_pos; identity before any eviction). `caches` must be non-empty +
/// uniform (the caller asserts uniform geometry). A panic in the (untrusted) stage is caught (mirror of
/// `drive_mutation_layer`).
pub(crate) fn drive_cross_layer(
    stage: &dyn KVMutationStage,
    caches: &mut [KVCache],
    target_len: usize,
    positions: &[usize],
) -> anyhow::Result<bool> {
    if caches.is_empty() || caches[0].current_pos() == 0 {
        return Ok(false);
    }
    // Entry-frame host-mirrored read ctx (owned snapshots — the immutable borrow of `caches` is
    // released here, before the `&mut` model handle below).
    let sctx = EngineCrossLayerStageCtx::new(caches, target_len, positions);
    let handle = EngineModelCacheHandle::new(caches);
    let driven = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        move || -> anyhow::Result<bool> {
            let mut handle = handle;
            stage
                .on_whole_model(&sctx, &mut handle)
                .map_err(|e| anyhow::anyhow!("whole-model stage '{}' failed: {e}", stage.name()))?;
            handle.commit()
        },
    ));
    match driven {
        Ok(r) => r,
        Err(_) => Err(anyhow::anyhow!(
            "whole-model stage '{}' panicked during on_whole_model/commit",
            stage.name()
        )),
    }
}

/// A pipeline stage that runs a [`KVMutationStage`] callback over a set of per-layer caches, mirroring
/// [`FormatReencodeStage`](super::format_reencode::FormatReencodeStage). Constructed directly in s1
/// (not registered into the pipeline; production wiring is a follow-up).
pub struct KVMutationDriverStage {
    /// register-time handles — enumerate order == layer idx (same W1 invariant as `FormatReencodeStage`).
    handles: Vec<Arc<StandardFormat>>,
    /// The imperative mutation callback this driver runs.
    stage: Box<dyn KVMutationStage>,
    /// The lifecycle phase this driver fires its stage at — taken from the stage's registration
    /// (`MutationStageReg.phase`, the single source of truth). The trait has no `phase()` method, so
    /// the registered placement phase and the runtime firing phase are the same value by construction.
    phase: MutationPhase,
    /// Per-layer eviction budget ratio. The read ctx exposes `target_len = (current_pos *
    /// target_ratio).max(1)` (mirror of [`EvictionHandler`](crate::kv::EvictionHandler)), so a
    /// score-based stage receives a real keep budget instead of the s1 stub's `0` (which made a
    /// budget-driven stage stage an empty keep = a full-cache wipe). `1.0` ⇒ keep all (no eviction by
    /// budget); a score-free positional stage that ignores `target_len` is unaffected.
    target_ratio: f32,
    /// Optional attention-score source for score-based stages (mirror of
    /// [`EvictionStage`](super::eviction::EvictionStage)'s `score_cell`). When `Some` and the
    /// accumulator is active, the driver extracts (flat importance, per-head importance, last-step
    /// head attn) ONCE per fire into owned snapshots fed to the read ctx, then `reset()`s the
    /// accumulator after a successful commit (stale-score guard: the KV geometry just changed). `None`
    /// = score-free (sliding/streaming/no-eviction): the ctx exposes no signals.
    score_cell: Option<Arc<Mutex<Option<AttentionScoreAccumulator>>>>,
    /// Shared plan-invalidation cell (T-6). Set `true` when a commit mutated any layer so the decode
    /// loop invalidates the fused decode plan. `None` until wired (P0-5c) — the test-driven driver and
    /// score-free positional stages that only shrink keeps are already covered by the pos-shrink reflux.
    mutation_fired: Option<Arc<AtomicBool>>,
    /// The stage's declared capabilities — used to gate the per-layer raw K/V dequant snapshot: a
    /// snapshot is built only when `caps.reads` contains the kind (and the cache is host-resident), so
    /// a score-free / importance-only stage pays nothing. Defaults to [`StageCaps::SCORE_FREE`] (no
    /// K/V reads); set via [`with_caps`](Self::with_caps) for value-aware (caote) / merge (d2o) stages.
    caps: StageCaps,
    /// Pressure band gate for the **standard-loop eviction mode** (mirror of
    /// [`EvictionStage::persistent`](super::eviction::EvictionStage)'s `min_band`). `Some(min)` makes
    /// the driver a faithful drop-in for the pressure-driven `EvictionStage`: it fires only when
    /// `ctx.step.pressure.band() >= min` (episode edge-triggered via [`armed`](Self::armed)) AND
    /// applies the full CacheManager bookkeeping fold — the MIN_EVICT_TOKENS budget guard, per-cache
    /// `release_unused_pages` (madvise), and `[CacheEvent]` logging. `None` (the bare [`new`](Self::new)
    /// default, used by the eval/test path) fires every matching phase step with none of that — the
    /// raw per-layer driver.
    min_band: Option<Level>,
    /// Episode edge-trigger for the pressure gate (mirror of `EvictionStage`'s `armed`): set on a
    /// band-met fire, re-armed when the band falls back below `min_band`, so a sustained high-pressure
    /// episode prunes once rather than spiralling the cache to the floor (per-step madvise/CacheEvent
    /// churn). Unused when `min_band` is `None`.
    armed: AtomicBool,
}

impl KVMutationDriverStage {
    /// `handles` enumerate order must equal layer idx. `stage` is the mutation callback; `phase` is its
    /// registered firing phase (`MutationStageReg.phase`); `target_ratio` sets the per-layer keep
    /// budget the read ctx exposes as `target_len`.
    pub fn new(
        handles: Vec<Arc<StandardFormat>>,
        stage: Box<dyn KVMutationStage>,
        phase: MutationPhase,
        target_ratio: f32,
    ) -> Self {
        Self {
            handles,
            stage,
            phase,
            target_ratio,
            score_cell: None,
            mutation_fired: None,
            caps: StageCaps::SCORE_FREE,
            min_band: None,
            armed: AtomicBool::new(true),
        }
    }

    /// Enable the **standard-loop eviction mode** — the driver becomes a faithful drop-in for the
    /// pressure-driven [`EvictionStage::persistent`](super::eviction::EvictionStage), differing only in
    /// that the keep-set is applied through the v3 [`CacheHandle`] (byte-identical to the prior in-place
    /// eviction it replaced, the Phase-1 gate) instead of `force_evict`. Sets the firing gate to
    /// `ctx.step.pressure.band() >= min_band` (episode edge-triggered) and folds in the CacheManager
    /// bookkeeping the v2 path owned: the `MIN_EVICT_TOKENS` budget guard, per-cache
    /// `release_unused_pages` (madvise), and `[CacheEvent]` logging. Without this builder the driver
    /// fires every matching phase step with no gate / guard / bookkeeping (the eval/test path).
    pub fn with_pressure_gate(mut self, min_band: Level) -> Self {
        self.min_band = Some(min_band);
        self
    }

    /// Attach a score accumulator cell so score-based stages receive importance / per-head scores /
    /// last-step attention through the read ctx (mirror of `EvictionStage::one_shot_scored`).
    pub fn with_score_cell(
        mut self,
        score_cell: Arc<Mutex<Option<AttentionScoreAccumulator>>>,
    ) -> Self {
        self.score_cell = Some(score_cell);
        self
    }

    /// Declare the stage's capabilities so the driver builds the per-layer raw K/V dequant snapshot
    /// only for the kinds the stage actually reads (`caps.reads` ∋ Key/Value). Without this the driver
    /// assumes [`StageCaps::SCORE_FREE`] (no K/V snapshot).
    pub fn with_caps(mut self, caps: StageCaps) -> Self {
        self.caps = caps;
        self
    }

    /// Attach the shared plan-invalidation cell (T-6, mirror of `FormatReencodeStage::reencode_fired`
    /// / `CommandDispatcher::reencode_fired_cell`). The driver sets it `true` when a commit mutated any
    /// layer, so the decode loop swap-checks it after the KvMutate dispatch and invalidates the fused
    /// decode plan (`on_kv_reencode`). A position-shrinking keep is ALSO covered by the loop's
    /// pos-shrink reflux, but a position-PRESERVING mutation (reencode) needs this cell.
    pub fn with_mutation_fired(mut self, cell: Arc<AtomicBool>) -> Self {
        self.mutation_fired = Some(cell);
        self
    }

    /// Extract owned snapshots of (flat importance, per-head importance, last-step head attn) from the
    /// score cell ONCE per fire (mirror of `EvictionStage::run_eviction`). Empty when there is no cell
    /// or the accumulator is inactive. The slices are cloned so the read ctx can borrow them without
    /// holding the cell lock across the per-layer callbacks.
    #[allow(clippy::type_complexity)]
    fn extract_scores(&self) -> (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>) {
        match self.score_cell.as_ref() {
            Some(cell) => {
                let guard = cell.lock().unwrap_or_else(|e| e.into_inner());
                match guard.as_ref().filter(|acc| acc.is_active()) {
                    Some(acc) => (
                        Some(acc.importance_scores().to_vec()),
                        acc.head_importance_scores().map(|s| s.to_vec()),
                        acc.last_step_head_attn().map(|s| s.to_vec()),
                    ),
                    None => (None, None, None),
                }
            }
            None => (None, None, None),
        }
    }

    /// Reset the score accumulator after a successful eviction (KV geometry changed → prior scores are
    /// stale). Mirror of `EvictionStage::run_eviction`'s post-eviction reset.
    fn reset_scores(&self) {
        if let Some(cell) = self.score_cell.as_ref() {
            let mut guard = cell.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(acc) = guard.as_mut() {
                acc.reset();
            }
        }
    }

    /// The per-layer keep budget for a cache at `current_pos`, mirroring
    /// [`EvictionHandler`](crate::kv::EvictionHandler): `(current_pos * target_ratio).max(1)`. The
    /// `.max(1)` floor is the wipe guard — a budget can never round down to `0` and make a
    /// budget-driven stage retain nothing.
    fn budget_for(&self, current_pos: usize) -> usize {
        (((current_pos as f32) * self.target_ratio) as usize).max(1)
    }
}

impl PipelineStage for KVMutationDriverStage {
    fn name(&self) -> &str {
        "kv.mutation_driver"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::Persistent
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter: only the stage's registered phase (the single placement source of truth).
        if *phase != lifecycle_of(self.phase) {
            return Ok(StageOutcome::Continue);
        }

        // Pressure gate (standard-loop eviction mode, mirror of `EvictionStage::on_phase`): when a
        // `min_band` is set, fire only on a pressure episode's rising edge. Below the band → no-op and
        // re-arm; at/above but already fired this episode → no-op. `None` (eval/test) → always fire.
        if let Some(min) = self.min_band {
            if ctx.step.pressure.band() < min {
                self.armed.store(true, Ordering::Relaxed);
                return Ok(StageOutcome::Continue);
            }
            if !self.armed.swap(false, Ordering::Relaxed) {
                return Ok(StageOutcome::Continue);
            }
        }

        // MIN_EVICT_TOKENS budget guard (standard-loop mode, mirror of the v2 uniform path in
        // `CacheManager::run_policy_eviction`): skip the whole eviction when the global budget delta is
        // below the floor, so a near-full cache is not compacted for a handful of tokens (matches the
        // v2 gate that ran BEFORE the per-layer plan). The per-layer `current_pos == 0` skip below still
        // applies in every mode. Note the band edge-trigger above is consumed even when this skips —
        // identical to v2, where `EvictionStage` disarms on band-met regardless of the inner guard.
        if self.min_band.is_some() {
            let current_pos = self
                .handles
                .iter()
                .map(|f| f.with_cache_mut(|c| c.current_pos()))
                .max()
                .unwrap_or(0);
            let target_len = self.budget_for(current_pos);
            if current_pos <= target_len || current_pos - target_len < MIN_EVICT_TOKENS {
                log::debug!(
                    "[CacheManager] skip: stage='{}', current_pos={current_pos}, target_len={target_len} (< MIN_EVICT_TOKENS={MIN_EVICT_TOKENS})",
                    self.stage.name(),
                );
                return Ok(StageOutcome::Continue);
            }
        }

        // Extract score snapshots ONCE per fire (owned, so the per-layer read ctx borrows them without
        // holding the cell lock across callbacks). All `None` for a score-free stage.
        let (importance, head_scores, last_attn) = self.extract_scores();

        // UER (mirroring FormatReencodeStage): take_inner -> per-layer drive+commit -> put_inner.
        let mut temp: Vec<KVCache> = self.handles.iter().map(|f| f.take_inner()).collect();
        let n_layers = temp.len();
        // T-6 (P0-5b): tracks whether any layer's commit mutated bytes, to fire plan invalidation.
        let mut any_mutated = false;
        let result = (|| -> anyhow::Result<()> {
            for (layer_idx, cache) in temp.iter_mut().enumerate() {
                // Real per-layer keep budget (P0-3a): a budget-driven stage gets a non-zero target_len
                // (the `.max(1)` floor in budget_for is the wipe guard). The shared per-layer core
                // builds the read ctx + handle and drives the stage (catch_unwind, P0-5a). A
                // `current_pos == 0` layer is a no-op inside it.
                let target_len = self.budget_for(cache.current_pos());
                any_mutated |= drive_mutation_layer(
                    self.stage.as_ref(),
                    &self.caps,
                    cache,
                    layer_idx,
                    n_layers,
                    target_len,
                    importance.as_deref(),
                    head_scores.as_deref(),
                    last_attn.as_deref(),
                )?;
            }
            Ok(())
        })();
        for (f, c) in self.handles.iter().zip(temp) {
            f.put_inner(c);
        }
        result?;
        // Reset scores after a successful eviction (KV geometry changed → prior scores stale).
        self.reset_scores();
        // CacheManager bookkeeping fold (standard-loop mode, mirror of `CacheManager::execute_dispatch`'s
        // post-eviction tail): on an actual mutation, advise the OS to reclaim the now-unused KV pages
        // (madvise MADV_DONTNEED, a host-buffer no-op on GPU/UMA) and emit the grep-able `[CacheEvent]`
        // line. Runs AFTER the per-layer `commit` shrank `current_pos` (release_unused_pages keys off it)
        // and only when `any_mutated`, so an idle/no-op fire does no madvise/log churn.
        if self.min_band.is_some() && any_mutated {
            let mut bytes_released = 0usize;
            let mut new_pos = 0usize;
            for f in self.handles.iter() {
                f.with_cache_mut(|c| {
                    bytes_released += c.release_unused_pages();
                    new_pos = new_pos.max(c.current_pos());
                });
            }
            log::info!(
                "[CacheEvent] kv.mutation_driver eviction: stage='{}', new_pos={new_pos}, bytes_released={bytes_released}",
                self.stage.name(),
            );
        }
        // T-6: a committed mutation invalidates the fused decode plan. Fire the shared cell so the
        // decode loop rebuilds it (covers position-PRESERVING reencode; a shrinking keep is also
        // covered by the loop's pos-shrink reflux).
        if any_mutated && let Some(cell) = self.mutation_fired.as_ref() {
            cell.store(true, Ordering::Relaxed);
        }
        // Persistent lifecycle → return `Continue` (mirror of `EvictionStage::persistent`): the driver
        // stays resident and re-fires on the next pressure episode (or the next PrefillEnd). Returning
        // `Consumed` would make `PipelineRegistry::dispatch` GC the stage after its first fire (and
        // `debug_assert` a Persistent/Consumed INV-DECODE-STAGE-007 violation) — disabling eviction for
        // the rest of the session. The pressure gate's `armed` edge-trigger is what bounds re-firing.
        Ok(StageOutcome::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{CacheHandle, CacheOpError};

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::memory::host::shared::SharedBuffer;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::quant::{BlockQ4_0, QK4_0};
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use half::f16;

    const HD: usize = 64; // q4_0-valid: a multiple of QK4_0 (=32).
    const KV_HEADS: usize = 2;
    const MAX_SEQ: usize = 32;
    const RESIDENT: usize = 20;

    /// Build a SeqMajor cache of `dtype` with a known per-(pos,head,d) pattern in the resident region
    /// of both K and V (V uses a distinct salt). The pattern is written as f32 then encoded to `dtype`,
    /// so the two paths under test start from byte-identical buffers.
    fn make_cache(dtype: DType) -> KVCache {
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HD]);
        let total = MAX_SEQ * KV_HEADS * HD;
        let bytes = match dtype {
            DType::F32 => total * 4,
            DType::F16 => total * 2,
            DType::Q4_0 => (total / QK4_0) * std::mem::size_of::<BlockQ4_0>(),
            _ => unreachable!(),
        };
        let mut c = KVCache::new(
            Tensor::new(
                sh.clone(),
                Arc::new(SharedBuffer::new(bytes, dtype)),
                be.clone(),
            ),
            Tensor::new(sh, Arc::new(SharedBuffer::new(bytes, dtype)), be),
            MAX_SEQ,
        );
        c.set_current_pos(RESIDENT);
        let pat = |pos: usize, head: usize, d: usize, salt: f32| {
            salt + pos as f32 * 0.11 + head as f32 * 0.27 + (d as f32 - HD as f32 / 2.0) * 0.04
        };
        for pos in 0..RESIDENT {
            for head in 0..KV_HEADS {
                let off = c.offset(pos, head);
                let mut row_k = [0.0f32; HD];
                let mut row_v = [0.0f32; HD];
                for d in 0..HD {
                    row_k[d] = pat(pos, head, d, 0.5);
                    row_v[d] = pat(pos, head, d, -1.3);
                }
                write_row(&mut c, off, dtype, &row_k, true);
                write_row(&mut c, off, dtype, &row_v, false);
            }
        }
        c
    }

    fn write_row(c: &mut KVCache, off: usize, dtype: DType, row: &[f32], is_k: bool) {
        match dtype {
            DType::F32 => {
                let buf = if is_k {
                    c.k_buffer.as_mut_slice::<f32>()
                } else {
                    c.v_buffer.as_mut_slice::<f32>()
                };
                buf[off..off + HD].copy_from_slice(row);
            }
            DType::F16 => {
                let buf = if is_k {
                    c.k_buffer.as_mut_slice::<f16>()
                } else {
                    c.v_buffer.as_mut_slice::<f16>()
                };
                for d in 0..HD {
                    buf[off + d] = f16::from_f32(row[d]);
                }
            }
            DType::Q4_0 => {
                let bpp = HD / QK4_0;
                let bo = off / QK4_0;
                let buf = if is_k {
                    c.k_buffer.as_mut_slice::<BlockQ4_0>()
                } else {
                    c.v_buffer.as_mut_slice::<BlockQ4_0>()
                };
                for bi in 0..bpp {
                    let mut blk = [0.0f32; QK4_0];
                    blk.copy_from_slice(&row[bi * QK4_0..(bi + 1) * QK4_0]);
                    buf[bo + bi] = BlockQ4_0::quantize(&blk);
                }
            }
            _ => unreachable!(),
        }
    }

    fn make_ctx(profiler: &mut OpProfiler) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(0),
                prev_token: 0,
            },
            profiler,
        }
    }

    /// A budget-driven mutation stage: keeps `[0..target_len)`. Stands in for the score-based class
    /// (caote/h2o) whose keep size is the engine-supplied budget — used to prove the P0-3 budget +
    /// wipe guard (the s1 stub's `target_len=0` would make this keep nothing = a full-cache wipe).
    struct BudgetKeepStage;
    impl KVMutationStage for BudgetKeepStage {
        fn name(&self) -> &str {
            "test.budget_keep"
        }
        fn on_phase(
            &self,
            ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            let keep: Vec<usize> = (0..ctx.target_len().min(ctx.current_pos())).collect();
            cache.keep(&keep)
        }
    }

    fn drive_budget_keep(resident: usize, target_ratio: f32) -> usize {
        let handle = Arc::new(StandardFormat::new(0, make_cache_f32_resident(resident)));
        let driver = KVMutationDriverStage::new(
            vec![handle.clone()],
            Box::new(BudgetKeepStage),
            MutationPhase::KvMutate,
            target_ratio,
        );
        let mut profiler = OpProfiler::new();
        let mut pctx = make_ctx(&mut profiler);
        driver
            .on_phase(&LifecyclePhase::KvMutate, &mut pctx)
            .unwrap();
        handle.take_inner().current_pos()
    }

    /// A minimal F32 SeqMajor cache with `resident` tokens (the byte pattern is irrelevant here — only
    /// `current_pos` drives the budget). Reuses the `make_cache` geometry constants.
    fn make_cache_f32_resident(resident: usize) -> KVCache {
        let mut c = make_cache(DType::F32);
        c.set_current_pos(resident);
        c
    }

    /// An F32 SeqMajor cache with `resident` tokens where every element of position `p` is `p`
    /// (so a survivor's original position is read directly off `k_buffer` after compaction).
    fn make_int_cache(resident: usize) -> KVCache {
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HD]);
        let total = MAX_SEQ * KV_HEADS * HD;
        let mut c = KVCache::new(
            Tensor::new(
                sh.clone(),
                Arc::new(SharedBuffer::new(total * 4, DType::F32)),
                be.clone(),
            ),
            Tensor::new(sh, Arc::new(SharedBuffer::new(total * 4, DType::F32)), be),
            MAX_SEQ,
        );
        c.set_current_pos(resident);
        for pos in 0..resident {
            for head in 0..KV_HEADS {
                let off = c.offset(pos, head);
                let kb = c.k_buffer.as_mut_slice::<f32>();
                for d in 0..HD {
                    kb[off + d] = pos as f32;
                }
                let vb = c.v_buffer.as_mut_slice::<f32>();
                for d in 0..HD {
                    vb[off + d] = pos as f32;
                }
            }
        }
        c
    }

    /// Task ② (read seam): `for_read` must DEFER the `tensor(Value)` dequant — Quest reads only
    /// Key/Query, so on the production read path V must never be dequantized. Proves: (a) the V cell is
    /// empty right after construction (no eager dequant), (b) reading `tensor(Key)` does not materialize
    /// it, and (c) the first `tensor(Value)` access materializes a snapshot byte-identical to an eager
    /// `dequant_snapshot(.., is_k = false)`. Mutation-proof: making `for_read` eager (or skipping the
    /// cell) fails (a)/(b); reading K instead of V in the lazy arm fails (c) (V carries a distinct salt).
    #[test]
    fn for_read_defers_value_dequant_until_first_access() {
        let mut cache = make_int_cache(RESIDENT);
        // Give V a pattern distinct from K (which is `pos`) so a lazy arm that mistakenly dequantized K
        // would be caught by the byte-identity check below.
        for pos in 0..RESIDENT {
            for head in 0..KV_HEADS {
                let off = cache.offset(pos, head);
                let vb = cache.v_buffer.as_mut_slice::<f32>();
                for d in 0..HD {
                    vb[off + d] = pos as f32 + 100.0;
                }
            }
        }
        let key_snap = dequant_snapshot(&cache, RESIDENT, KV_HEADS, HD, true);
        let ctx = SnapshotStageCtx::for_read(&cache, &key_snap, None, None);

        // (a) V dequant is deferred — nothing materialized at construction.
        assert!(
            ctx.value_cell.get().is_none(),
            "for_read must NOT dequantize V eagerly"
        );
        // (b) Reading Key (exactly what Quest does) must not trigger the V dequant.
        assert!(ctx.tensor(TensorKind::Key).is_some());
        assert!(
            ctx.value_cell.get().is_none(),
            "reading tensor(Key) must not materialize V"
        );

        // (c) First tensor(Value) access materializes the snapshot, byte-identical to an eager dequant.
        assert!(ctx.tensor(TensorKind::Value).is_some());
        let materialized = ctx
            .value_cell
            .get()
            .expect("tensor(Value) materializes the cell");
        let eager_v = dequant_snapshot(&cache, RESIDENT, KV_HEADS, HD, false);
        assert_eq!(
            materialized.data, eager_v,
            "lazy V snapshot must equal the eager dequant"
        );
    }

    /// A score-based stage that keeps the top-`target_len` positions by `importance()` (h2o shape).
    /// Reads scores through the ctx — the score-free fallback (no importance) would keep `[0..target)`.
    struct ImportanceTopKStage;
    impl KVMutationStage for ImportanceTopKStage {
        fn name(&self) -> &str {
            "test.importance_topk"
        }
        fn on_phase(
            &self,
            ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            let pos = ctx.current_pos();
            let target = ctx.target_len().min(pos);
            let imp = ctx.importance().unwrap_or(&[]);
            let mut idx: Vec<usize> = (0..pos).collect();
            idx.sort_by(|&a, &b| {
                imp.get(b)
                    .unwrap_or(&0.0)
                    .partial_cmp(imp.get(a).unwrap_or(&0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut keep: Vec<usize> = idx.into_iter().take(target).collect();
            keep.sort_unstable();
            cache.keep(&keep)
        }
    }

    /// P0-3b: a score-based stage receives flat `importance()` through the driver's score_cell, and the
    /// accumulator is reset after eviction. Scores favor positions {2,5}; budget 0.25*8=2 → exactly
    /// those survive (the score-free fallback would keep {0,1}), proving importance reached the plugin.
    /// Mutation-proof: dropping the score_cell wiring leaves importance None → survivors become {0,1}.
    #[test]
    fn importance_reaches_stage_and_resets() {
        let handle = Arc::new(StandardFormat::new(0, make_int_cache(8)));
        let mut acc = AttentionScoreAccumulator::new(MAX_SEQ, 1, 1, 1, 1.0);
        acc.set_active(true);
        let mut flat = vec![0.0f32; MAX_SEQ];
        flat[2] = 10.0;
        flat[5] = 9.0;
        acc.import_gpu_scores(&flat, &[]);
        let cell = Arc::new(Mutex::new(Some(acc)));

        let driver = KVMutationDriverStage::new(
            vec![handle.clone()],
            Box::new(ImportanceTopKStage),
            MutationPhase::KvMutate,
            0.25,
        )
        .with_score_cell(cell.clone());
        let mut profiler = OpProfiler::new();
        let mut pctx = make_ctx(&mut profiler);
        driver
            .on_phase(&LifecyclePhase::KvMutate, &mut pctx)
            .unwrap();

        let inner = handle.take_inner();
        assert_eq!(inner.current_pos(), 2, "budget 0.25 * 8 = 2");
        let k = inner.k_buffer.as_slice::<f32>();
        // Survivors compacted to the front: new pos 0 == original 2, new pos 1 == original 5.
        assert_eq!(k[inner.offset(0, 0)], 2.0, "highest-importance pos 2 kept");
        assert_eq!(k[inner.offset(1, 0)], 5.0, "second-importance pos 5 kept");

        // reset-on-eviction: the accumulator's importance is cleared after a successful eviction.
        let g = cell.lock().unwrap();
        assert!(
            g.as_ref()
                .unwrap()
                .importance_scores()
                .iter()
                .all(|&x| x == 0.0),
            "scores reset after eviction (stale-score guard)"
        );
    }

    /// A value-aware stage (caote shape): reads raw V via `tensor(Value)` and keeps positions whose
    /// V value is even. With V[p]=p this keeps {0,2,4,6}. Asserts internally that tensor(Value) is
    /// present (the snapshot was built) — so a missing snapshot fails loudly instead of silently
    /// keeping nothing.
    struct ValueReadStage;
    impl KVMutationStage for ValueReadStage {
        fn name(&self) -> &str {
            "test.value_read"
        }
        fn on_phase(
            &self,
            ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            assert!(
                ctx.tensor(TensorKind::Value).is_some(),
                "value snapshot must be present when caps declare Value"
            );
            let pos = ctx.current_pos();
            let hd = ctx.head_dim();
            let mut row = vec![0.0f32; hd];
            let mut keep = Vec::new();
            for p in 0..pos {
                ctx.dequant_v(p, 0, &mut row); // reads tensor(Value)
                if (row[0] as usize).is_multiple_of(2) {
                    keep.push(p);
                }
            }
            cache.keep(&keep)
        }
    }

    /// P0-3c: a value-aware stage reads raw V through the driver's per-layer dequant snapshot (caote's
    /// `v_i`), and a stage that does NOT declare the read sees `tensor(Value)==None` (no snapshot
    /// built — the score-free fast path). Mutation-proof: removing the caps gate would build a snapshot
    /// even for the no-caps stage; removing the dequant_snapshot wiring makes the value read all-zero
    /// → every position "even" → keeps all 8 (failing the ==4 assert).
    #[test]
    fn value_snapshot_reaches_value_aware_stage() {
        // caps declare Value → snapshot built → stage keeps evens {0,2,4,6}.
        let handle = Arc::new(StandardFormat::new(0, make_int_cache(8)));
        let caps = StageCaps {
            reads: &[TensorKind::Value],
            default_protected_prefix: 0,
            produces_merge_plan: false,
            whole_model: false,
            prefill_attn_window: None,
        };
        let driver = KVMutationDriverStage::new(
            vec![handle.clone()],
            Box::new(ValueReadStage),
            MutationPhase::KvMutate,
            1.0, // budget irrelevant — the stage keeps by value, not target_len.
        )
        .with_caps(caps);
        let mut profiler = OpProfiler::new();
        let mut pctx = make_ctx(&mut profiler);
        driver
            .on_phase(&LifecyclePhase::KvMutate, &mut pctx)
            .unwrap();
        let inner = handle.take_inner();
        assert_eq!(inner.current_pos(), 4, "kept the 4 even-valued positions");
        let k = inner.k_buffer.as_slice::<f32>();
        for (i, &orig) in [0usize, 2, 4, 6].iter().enumerate() {
            assert_eq!(
                k[inner.offset(i, 0)],
                orig as f32,
                "survivor {i} == orig {orig}"
            );
        }

        // No caps (SCORE_FREE default) → no snapshot → tensor(Value) is None.
        struct ProbeNoValue;
        impl KVMutationStage for ProbeNoValue {
            fn name(&self) -> &str {
                "test.probe_no_value"
            }
            fn on_phase(
                &self,
                ctx: &dyn StageCtx,
                _cache: &mut dyn CacheHandle,
            ) -> Result<(), CacheOpError> {
                assert!(
                    ctx.tensor(TensorKind::Value).is_none(),
                    "no Value snapshot without caps"
                );
                Ok(())
            }
        }
        let handle2 = Arc::new(StandardFormat::new(0, make_int_cache(8)));
        let driver2 = KVMutationDriverStage::new(
            vec![handle2.clone()],
            Box::new(ProbeNoValue),
            MutationPhase::KvMutate,
            1.0,
        );
        let mut profiler2 = OpProfiler::new();
        let mut pctx2 = make_ctx(&mut profiler2);
        driver2
            .on_phase(&LifecyclePhase::KvMutate, &mut pctx2)
            .unwrap();
        assert_eq!(
            handle2.take_inner().current_pos(),
            8,
            "probe kept all (no-op)"
        );
    }

    /// A stage that panics in on_phase — stands in for an untrusted native plugin that faults.
    struct PanicStage;
    impl KVMutationStage for PanicStage {
        fn name(&self) -> &str {
            "test.panic"
        }
        fn on_phase(
            &self,
            _ctx: &dyn StageCtx,
            _cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            panic!("intentional plugin panic");
        }
    }

    /// P0-5a catch_unwind: a panic in a plugin's on_phase is caught and converted to an Err — it does
    /// NOT unwind past the driver's put_inner rewrap, so the StandardFormat handle is restored intact
    /// (take_inner succeeds afterward, cache untouched). Mutation-proof: removing the catch_unwind
    /// makes on_phase panic-unwind, the handle is left empty, and the post-run take_inner would observe
    /// a placeholder (current_pos 0) instead of the original 8.
    #[test]
    fn panic_in_stage_is_caught_and_handles_restored() {
        let handle = Arc::new(StandardFormat::new(0, make_int_cache(8)));
        let driver = KVMutationDriverStage::new(
            vec![handle.clone()],
            Box::new(PanicStage),
            MutationPhase::KvMutate,
            1.0,
        );
        let mut profiler = OpProfiler::new();
        let mut pctx = make_ctx(&mut profiler);
        // Suppress the default panic hook's stderr noise for this intentional panic.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            driver.on_phase(&LifecyclePhase::KvMutate, &mut pctx)
        }));
        std::panic::set_hook(prev);
        // The driver converted the plugin panic into an Err (it did NOT propagate a panic).
        match r {
            Ok(Ok(_)) => panic!("driver should have returned Err on a plugin panic"),
            Ok(Err(_)) => {} // expected: caught panic -> Err
            Err(_) => panic!("panic escaped the driver (catch_unwind missing)"),
        }
        // The handle was restored by put_inner: take_inner works and the cache is intact.
        assert_eq!(
            handle.take_inner().current_pos(),
            8,
            "cache untouched + handle restored after a caught panic"
        );
    }

    /// A position-PRESERVING mutation stage: re-encodes f32 -> f16 (no compaction). Exercises the T-6
    /// path that the pos-shrink reflux does NOT cover.
    struct ReencodeStage;
    impl KVMutationStage for ReencodeStage {
        fn name(&self) -> &str {
            "test.reencode"
        }
        fn on_phase(
            &self,
            _ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            cache.reencode(argus_extension_api::FormatId("f16".into()))
        }
    }

    /// A no-op stage (stages nothing) — commit reports no mutation.
    struct NoopMutStage;
    impl KVMutationStage for NoopMutStage {
        fn name(&self) -> &str {
            "test.noop_mut"
        }
        fn on_phase(
            &self,
            _ctx: &dyn StageCtx,
            _cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            Ok(())
        }
    }

    /// P0-5b T-6: a committed mutation fires the shared plan-invalidation cell (so the decode loop
    /// rebuilds the fused plan), and a no-op commit leaves it unset. Mutation-proof: dropping the
    /// `cell.store(true)` leaves the cell false after a real reencode → the fused plan would not be
    /// invalidated and decode would read stale geometry.
    #[test]
    fn mutation_fires_plan_invalidation_cell() {
        // A real reencode (f32 -> f16) sets the cell.
        let handle = Arc::new(StandardFormat::new(0, make_int_cache(8)));
        let cell = Arc::new(AtomicBool::new(false));
        let driver = KVMutationDriverStage::new(
            vec![handle.clone()],
            Box::new(ReencodeStage),
            MutationPhase::KvMutate,
            1.0,
        )
        .with_mutation_fired(cell.clone());
        let mut profiler = OpProfiler::new();
        let mut pctx = make_ctx(&mut profiler);
        driver
            .on_phase(&LifecyclePhase::KvMutate, &mut pctx)
            .unwrap();
        assert!(
            cell.load(Ordering::Relaxed),
            "a committed reencode must fire plan invalidation (T-6)"
        );

        // A no-op stage leaves the cell unset.
        let handle2 = Arc::new(StandardFormat::new(0, make_int_cache(8)));
        let cell2 = Arc::new(AtomicBool::new(false));
        let driver2 = KVMutationDriverStage::new(
            vec![handle2.clone()],
            Box::new(NoopMutStage),
            MutationPhase::KvMutate,
            1.0,
        )
        .with_mutation_fired(cell2.clone());
        let mut profiler2 = OpProfiler::new();
        let mut pctx2 = make_ctx(&mut profiler2);
        driver2
            .on_phase(&LifecyclePhase::KvMutate, &mut pctx2)
            .unwrap();
        assert!(
            !cell2.load(Ordering::Relaxed),
            "a no-op commit must NOT fire plan invalidation"
        );
    }

    /// P0-3 wipe guard: a budget-driven stage keeps `target_len = (pos * ratio).max(1)` tokens, NOT 0.
    /// With the s1 stub (`target_len=0`) this stage would `keep(&[])` and wipe the whole cache; the real
    /// budget makes it retain the budgeted count. The `.max(1)` floor guarantees a tiny ratio still
    /// keeps ≥1 token (never a wipe). Mutation-proof: reverting budget_for to `0` makes both asserts
    /// fail (current_pos would be 0).
    #[test]
    fn budget_drives_keep_and_never_wipes() {
        // ratio 0.5 on 8 resident → target_len 4 → keeps [0,1,2,3].
        assert_eq!(drive_budget_keep(8, 0.5), 4, "budget = (8*0.5).max(1) = 4");
        // tiny ratio: (8 * 0.01) rounds to 0 → .max(1) floor → keep [0] → current_pos 1, NOT a wipe.
        assert_eq!(
            drive_budget_keep(8, 0.01),
            1,
            "wipe guard: budget floored to 1, never 0"
        );
        // ratio 1.0 → keep all 8 (no budget-driven eviction).
        assert_eq!(drive_budget_keep(8, 1.0), 8, "ratio 1.0 keeps all");
    }

    /// Assert K and V byte-identical over the resident region of two caches.
    fn assert_kv_byte_identical(a: &KVCache, b: &KVCache) {
        assert_eq!(a.current_pos(), b.current_pos());
        let n = a.current_pos();
        for pos in 0..n {
            for head in 0..KV_HEADS {
                let off = a.offset(pos, head);
                assert_eq!(off, b.offset(pos, head));
                assert_eq!(
                    &a.k_buffer.as_slice::<f32>()[off..off + HD],
                    &b.k_buffer.as_slice::<f32>()[off..off + HD],
                    "K pos {pos} head {head}"
                );
                assert_eq!(
                    &a.v_buffer.as_slice::<f32>()[off..off + HD],
                    &b.v_buffer.as_slice::<f32>()[off..off + HD],
                    "V pos {pos} head {head}"
                );
            }
        }
    }

    /// F4: a re-encode (f16 -> f32) through the handle commit is byte-identical to a direct
    /// apply_format_plan — covering the reencode-commit arm (zero handle reencode call-sites before).
    #[test]
    fn reencode_via_handle_byte_identical() {
        use crate::kv::format_apply::apply_format_plan;
        use argus_extension_api::{FormatId, KVFormatPlan};
        let mut cv2 = make_cache(DType::F16);
        apply_format_plan(
            &mut cv2,
            &KVFormatPlan {
                base: FormatId("f32".into()),
                overrides: vec![],
            },
            0,
            1,
        )
        .unwrap();

        let mut ch = make_cache(DType::F16);
        {
            let mut h = EngineCacheHandle::new(&mut ch, 0, 1);
            h.reencode(FormatId("f32".into())).unwrap();
            assert_eq!(h.commit().unwrap(), true);
        }
        assert_eq!(cv2.kv_dtype(), DType::F32);
        assert_eq!(ch.kv_dtype(), DType::F32);
        assert_kv_byte_identical(&cv2, &ch);
    }

    // ── P0-5c: standard-loop eviction mode (pressure gate + MIN_EVICT guard + bookkeeping) ──

    const BIG_MAX: usize = 256;

    /// A large all-zero SeqMajor F32 cache — large enough that `current_pos - target_len` can clear the
    /// MIN_EVICT_TOKENS(64) floor (the small `make_cache` can't). Byte content is irrelevant here; the
    /// tests assert firing/positions, not buffer bytes (those are covered by the byte-identity gates).
    fn make_big_cache(resident: usize) -> KVCache {
        let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, BIG_MAX, KV_HEADS, HD]);
        let total = BIG_MAX * KV_HEADS * HD;
        let mut c = KVCache::new(
            Tensor::new(
                sh.clone(),
                Arc::new(SharedBuffer::new(total * 4, DType::F32)),
                be.clone(),
            ),
            Tensor::new(sh, Arc::new(SharedBuffer::new(total * 4, DType::F32)), be),
            BIG_MAX,
        );
        c.set_current_pos(resident);
        c
    }

    fn make_ctx_pressure(profiler: &mut OpProfiler, pressure: u8) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(pressure),
                prev_token: 0,
            },
            profiler,
        }
    }

    /// A minimal score-free mutation stage that evicts to the budget: keep the first `target_len`
    /// positions. Always mutates when `current_pos > target_len`, so it deterministically drives the
    /// gate/guard/bookkeeping under test.
    struct KeepBudget;
    impl KVMutationStage for KeepBudget {
        fn name(&self) -> &str {
            "keep_budget_test"
        }
        fn on_phase(
            &self,
            ctx: &dyn StageCtx,
            cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            let keep: Vec<usize> = (0..ctx.target_len().min(ctx.current_pos())).collect();
            if keep.is_empty() {
                return Ok(());
            }
            cache.keep(&keep)
        }
    }

    fn pressure_gated(handle: Arc<StandardFormat>, ratio: f32) -> KVMutationDriverStage {
        KVMutationDriverStage::new(
            vec![handle],
            Box::new(KeepBudget),
            MutationPhase::KvMutate,
            ratio,
        )
        .with_pressure_gate(Level::Warning)
    }

    /// Band met (Warning) + budget delta ≥ MIN_EVICT → fires, evicting to the budget.
    #[test]
    fn pressure_gate_fires_on_band_met() {
        let handle = Arc::new(StandardFormat::new(0, make_big_cache(200)));
        // pos=200, ratio=0.3 → target_len=60, tokens_to_remove=140 ≥ 64.
        let driver = pressure_gated(handle.clone(), 0.3);
        let mut prof = OpProfiler::new();
        let mut ctx = make_ctx_pressure(&mut prof, 50); // band() = Warning
        let outcome = driver
            .on_phase(&LifecyclePhase::KvMutate, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Continue)); // Persistent → stays resident
        assert_eq!(handle.take_inner().current_pos(), 60, "evicted to budget");
    }

    /// Band below `min_band` (Normal) → no-op, cache untouched.
    #[test]
    fn pressure_gate_skips_below_band() {
        let handle = Arc::new(StandardFormat::new(0, make_big_cache(200)));
        let driver = pressure_gated(handle.clone(), 0.3);
        let mut prof = OpProfiler::new();
        let mut ctx = make_ctx_pressure(&mut prof, 10); // band() = Normal < Warning
        driver
            .on_phase(&LifecyclePhase::KvMutate, &mut ctx)
            .unwrap();
        assert_eq!(handle.take_inner().current_pos(), 200, "below band → no-op");
    }

    /// Episode edge-trigger: fires once per pressure episode; re-arms after the band falls back.
    /// Mirror of `EvictionStage::persistent_edge_trigger_once_per_episode`.
    #[test]
    fn pressure_gate_edge_trigger_once_per_episode() {
        let handle = Arc::new(StandardFormat::new(0, make_big_cache(200)));
        let driver = pressure_gated(handle.clone(), 0.3);
        let mut prof = OpProfiler::new();

        // episode 1: rising edge → fire.
        let mut c = make_ctx_pressure(&mut prof, 50);
        driver.on_phase(&LifecyclePhase::KvMutate, &mut c).unwrap();
        assert_eq!(
            handle.with_cache_mut(|c| c.current_pos()),
            60,
            "episode 1 fires"
        );

        // sustained pressure, cache regrown → same episode → no re-fire (armed consumed).
        handle.with_cache_mut(|c| c.set_current_pos(200));
        let mut c = make_ctx_pressure(&mut prof, 60);
        driver.on_phase(&LifecyclePhase::KvMutate, &mut c).unwrap();
        assert_eq!(
            handle.with_cache_mut(|c| c.current_pos()),
            200,
            "same episode → no re-fire"
        );

        // band drops below min → re-arm (no-op).
        let mut c = make_ctx_pressure(&mut prof, 10);
        driver.on_phase(&LifecyclePhase::KvMutate, &mut c).unwrap();
        assert_eq!(
            handle.with_cache_mut(|c| c.current_pos()),
            200,
            "Normal → no-op"
        );

        // episode 2: rising edge again → re-fire.
        let mut c = make_ctx_pressure(&mut prof, 80);
        driver.on_phase(&LifecyclePhase::KvMutate, &mut c).unwrap();
        assert_eq!(
            handle.with_cache_mut(|c| c.current_pos()),
            60,
            "re-armed → fires again"
        );
    }

    /// MIN_EVICT_TOKENS budget guard: band met but the budget delta is below the floor → skip (the
    /// near-full cache is not compacted for a handful of tokens), mirror of the v2 uniform-path guard.
    #[test]
    fn min_evict_guard_skips_small_delta() {
        let handle = Arc::new(StandardFormat::new(0, make_big_cache(80)));
        // pos=80, ratio=0.75 → target_len=60, tokens_to_remove=20 < 64 → skip.
        let driver = pressure_gated(handle.clone(), 0.75);
        let mut prof = OpProfiler::new();
        let mut ctx = make_ctx_pressure(&mut prof, 50); // Warning — band passes, guard skips.
        driver
            .on_phase(&LifecyclePhase::KvMutate, &mut ctx)
            .unwrap();
        assert_eq!(
            handle.take_inner().current_pos(),
            80,
            "delta < MIN_EVICT → skip"
        );
    }

    /// INV-DECODE-STAGE-007 regression: a FIRING driver must return `Continue` (not `Consumed`) under
    /// its `Persistent` lifecycle, else `PipelineRegistry::dispatch` GCs it after the first eviction
    /// (release) / `debug_assert`s a Persistent/Consumed violation (debug) — silently disabling
    /// eviction for the rest of the session. The registry side of this invariant is pinned by
    /// `tests/spec/test_inv_decode_stage_004_005_006_007.rs`; this pins the driver's side.
    #[test]
    fn firing_driver_returns_continue_under_persistent_lifecycle() {
        let handle = Arc::new(StandardFormat::new(0, make_big_cache(200)));
        let driver = pressure_gated(handle.clone(), 0.3);
        assert_eq!(driver.lifecycle(), StageLifecycle::Persistent);
        let mut prof = OpProfiler::new();
        let mut ctx = make_ctx_pressure(&mut prof, 50); // band met → actually fires + evicts.
        let outcome = driver
            .on_phase(&LifecyclePhase::KvMutate, &mut ctx)
            .unwrap();
        assert!(
            matches!(outcome, StageOutcome::Continue),
            "a Persistent stage that fired must return Continue (INV-DECODE-STAGE-007), not Consumed"
        );
        assert!(
            handle.with_cache_mut(|c| c.current_pos()) < 200,
            "non-vacuous: the fire actually evicted"
        );
    }

    /// The bare `new()` (eval/test) mode has NO pressure gate and NO MIN_EVICT guard: it fires every
    /// matching phase step regardless of pressure, and evicts even a sub-MIN_EVICT delta. This pins that
    /// `with_pressure_gate` is strictly additive — the prior every-step behavior is unchanged.
    #[test]
    fn plain_mode_ignores_pressure_and_min_evict() {
        let handle = Arc::new(StandardFormat::new(0, make_big_cache(80)));
        // Same pos/ratio as the MIN_EVICT skip test (delta=20<64), but plain mode → still evicts.
        let driver = KVMutationDriverStage::new(
            vec![handle.clone()],
            Box::new(KeepBudget),
            MutationPhase::KvMutate,
            0.75,
        );
        let mut prof = OpProfiler::new();
        let mut ctx = make_ctx_pressure(&mut prof, 0); // Normal — plain mode ignores it.
        driver
            .on_phase(&LifecyclePhase::KvMutate, &mut ctx)
            .unwrap();
        assert_eq!(
            handle.take_inner().current_pos(),
            60,
            "plain mode fires every step, no MIN_EVICT guard"
        );
    }
}
