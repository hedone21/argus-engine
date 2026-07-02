//! argus-extension-api — the additive surface where extension techniques (stage axis) register themselves with **zero engine-core modifications**.
//!
//! extension mechanism = statically linked technique crate + linkme auto-registration. Each technique, in its own crate
//! (`crates/techniques/<name>/`), depends only on this crate, implements [`KVMutationStage`], and
//! submits itself to the [`KV_MUTATION_STAGES`] slice via `#[distributed_slice]`. At construction
//! time the engine reads that slice to pick a technique (removing closed match arms → OCP).
//!
//! stage-axis extension techniques (eviction/merge) are unified under a **single plan-returning trait [`KVMutationStage`]**
//! (a sibling of the engine-side storage-representation trait `KVCacheFormat`). A technique *reads* [`StageCtx`] and stages mutations on a transactional [`CacheHandle`]
//! (retained tokens + weighted merges); it never mutates buffers directly — the engine owns the commit
//! via `compact` (D1). State (d2o EMA, etc.) is held by the plugin struct itself via `&self` + interior mutability
//! (D4); it is not threaded through the ctx.
//!
//! Dependency direction: `engine → argus-extension-api ← technique crate` (one-way, no cycles). Hence this crate
//! **does not reference** engine types (`KVCache`/`Backend`) — the cache state a technique needs to read is exposed through the read-only abstraction [`StageCtx`] that this crate
//! defines, with the engine implementing it over `&KVCache` (D5). In the static
//! stage this is a borrow; in a future `.so` C-ABI stage the same abstraction is swapped for C accessors / a flat snapshot — forward-compatible.

use core::ffi::{c_char, c_void};

/// Re-exports linkme's proc-macro so the `register_kv_mutation_stage!` macro can reference the `distributed_slice` attribute by path
/// from a plugin crate (so the plugin need not depend on linkme directly).
/// This crate's own internal registration (`#[distributed_slice]`) also uses this import. (The macro itself, not the crate,
/// must be re-exported directly so the proc-macro attribute path resolves.)
pub use linkme::distributed_slice;

/// The named cache tensors the engine exposes. Mutation (retain/merge) happens only via a plan; reads are unified through this enum.
/// **OCP**: a future input (Query/PageBounds, etc.) is one added variant + one engine impl site — no new `StageCtx`
/// method required. Read-dispatch cost is on par with additive accessors (PoC: host/ARM ±0–1%).
/// `#[repr(u32)]`: in a future `.so` C-ABI the fieldless enum is passed across as a u32 discriminant as-is (ADR §7).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorKind {
    /// raw K. row=(pos,head), cols=head_dim. dtype branching (F32/F16/Q4_0) is absorbed inside the handle.
    Key,
    /// raw V (the per-(pos,head) value vector value-aware policies read as v_i). Same (pos,head) coordinate system as Key.
    Value,
    /// per-(kv_head,pos) attention weight from the previous decode step (last layer) — the a_i read by value-aware policies. cols=1, per_head.
    /// Source: `AttentionScoreAccumulator::last_step_head_attn` (CPU overwrite / GPU = head_importance proxy).
    /// **Note**: a last-layer, last-step approximation — not a windowed/per-layer exact value (`has_attn_weights` gate).
    AttnWeights,
    /// per-(kv_head,pos) accumulated head importance (h2o_plus). cols=1, per_head.
    /// flat per-token importance is exposed zero-copy directly via [`StageCtx::importance`] rather than this handle (a D1 exception).
    Scores,
    /// per-(layer,kv_head) Q (query) running statistics — the input to a closed-form future-attention
    /// estimate. `shape = {rows:2, cols:head_dim, per_head:true}`
    /// (MQ-1): `read_row(0, kv_head, out)` = that kv_head's Q running **mean[head_dim]**,
    /// `read_row(1, kv_head, out)` = running **var[head_dim]**. Reduced to kv_head coordinates by the element-wise
    /// mean of the Q-head statistics within a GQA group (MQ-2 — the same kv_head coordinates as the GQA reduction of `Scores`/`AttnWeights`, so they are cross-
    /// usable). `Some` only on the score-active path (decode-step RoPE-applied Q capture); `None` otherwise
    /// (MQ-3/MQ-4 hot-path gate). discriminant 4 — existing 0–3 unchanged (C-ABI additive, MQ-5).
    QueryStats,
    /// Per-ATTENTION-head (PRE-GQA, NOT n_kv_heads) attention probabilities computed at prefill from
    /// a trailing query window (`q_window`) to all prefix keys, SUM-aggregated over the window.
    /// shape = `{ rows: n_heads_q, cols: prefix_len (== cache_seq_len), per_head: false }`.
    /// per_head=false ⇒ read_row(row, _kv_head, out) IGNORES kv_head and addresses by `row`
    /// (= attention head). Row→kv_head GQA grouping is `kv_head = row / (n_heads_q / n_kv_heads)`;
    /// GQA/key-pooling/`q_window`/mean-vs-sum reduction are PLUGIN policy. SUM is the neutral engine
    /// choice (mean = sum/q_window recoverable; MAX-over-window is NOT). Distinct from
    /// [`TensorKind::Scores`] (disc 3: decode-time, `[n_kv_heads x max_seq]`, cols=1, per_head:true).
    /// `Some` only on the prefill-end PFA-active path; `None` at KvMutate/decode and when unarmed.
    /// discriminant 5 — 0–4 unchanged (C-ABI additive, fieldless repr(u32)).
    PrefillAttention,
    /// (D2) raw CURRENT-step Q (query) vector, RoPE-applied. `shape = {rows:1, cols:head_dim,
    /// per_head:true}`: `read_row(0, kv_head, out)` = that kv_head's current Q (GQA-reduced to kv_head
    /// coordinates, like [`TensorKind::QueryStats`]). This is the exact per-channel `q_d` faithful
    /// Quest needs for `Σ_d max(q_d·min_d, q_d·max_d)` — [`TensorKind::QueryStats`] only carries a
    /// retrospective running mean+var, never the live query. `Some` when the forward seam fed the
    /// current Q on the faithful read-plan path; `None` otherwise (production decode feeds no Query).
    /// discriminant 6 — 0–5 unchanged (C-ABI additive, fieldless repr(u32)).
    Query,
}

/// Discriminant pins — the `#[repr(u32)]` values are the wire format (`kind as u32`), so any
/// reorder/insertion silently renumbers the C-ABI. These compile-time asserts freeze 0–5.
const _: () = assert!(TensorKind::Key as u32 == 0);
const _: () = assert!(TensorKind::Value as u32 == 1);
const _: () = assert!(TensorKind::AttnWeights as u32 == 2);
const _: () = assert!(TensorKind::Scores as u32 == 3);
const _: () = assert!(TensorKind::QueryStats as u32 == 4);
const _: () = assert!(TensorKind::PrefillAttention as u32 == 5);
const _: () = assert!(TensorKind::Query as u32 == 6);

/// dtype-agnostic tensor shape (POD). Only flat fields that can cross a future FFI boundary as-is (`#[repr(C)]`-able).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// number of valid rows (usually `current_pos`; QueryStats=2 = mean row + var row, MQ-1;
    /// PrefillAttention=n_heads_q = attention heads pre-GQA).
    pub rows: usize,
    /// number of f32 elements per row (Key/Value=head_dim, AttnWeights/Scores=1, QueryStats=head_dim;
    /// PrefillAttention=prefix_len).
    pub cols: usize,
    /// whether rows are split per-kv-head (true for all kinds **except** PrefillAttention, which is
    /// per-attention-head pre-GQA and addressed by `row`; layer-wide flat goes through the separate
    /// `importance()` path).
    pub per_head: bool,
}

/// the handle's storage dtype (for diagnostics / buffer sizing). Read output is always f32.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDtype {
    F32,
    F16,
    Q4_0,
}

/// A **read-only** handle for one named tensor. dyn-safe: no generics / `Self` by value /
/// associated types / `impl Trait` arguments. Output is always a dtype-agnostic f32 out-param (no slice returns —
/// inheriting the `dequant_k` out-param convention). In a future `.so` stage this reduces to (opaque ptr + read function pointer + POD shape).
pub trait TensorHandle {
    /// POD shape. In a future FFI stage the same struct crosses over as-is.
    fn shape(&self) -> TensorShape;
    /// storage dtype (independent of the f32 read output; for diagnostics).
    fn dtype(&self) -> TensorDtype;
    /// Fills the `(row, kv_head)` row into `out` as f32. Contract: `out.len() == shape().cols`.
    /// `per_head=false` tensors ignore `kv_head`. dtype branching is absorbed inside the impl.
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]);
}

/// The abstraction through which a technique reads the cache. The engine implements it over `&KVCache` (+ scores/budget).
///
/// **dyn-safe is mandatory**: since `plan(&self, ctx: &dyn StageCtx)` takes ctx as a trait object, every
/// method must be free of generic parameters / `Self` by value / associated types / `impl Trait` arguments. Therefore
/// even dtype-branching accesses like raw K reads are exposed as an `out: &mut [f32]` out-param rather than a slice return.
///
/// State accumulated across calls (d2o EMA, etc.) is not threaded through here — the plugin struct holds it (D4).
///
/// **Read unification**: all tensor/score reads flow through the single [`StageCtx::tensor`] mechanism.
/// `dequant_k`/`dequant_v`/`head_score`/`has_head_scores`/`attn_weight`/`has_attn_weights` are
/// default sugar on top of `tensor()` — the engine only needs to implement `tensor()`. Only flat `importance()` is exposed zero-copy
/// directly (an exception, since routing a scalar through per-element read_row would be a net loss for the heavy-hitter ranking path).
pub trait StageCtx {
    /// Current number of valid tokens. Every technique reads this as the starting point for computing its keep/prune budget.
    /// Engine impl source: `KVCache::current_pos()`.
    fn current_pos(&self) -> usize;

    /// The resolved budget — the absolute number of tokens to retain. ratio→len conversion is the engine's responsibility (`EvictionHandler`), so
    /// the plugin reads only the converted value. score-free or head-relative budget techniques (no_eviction/h2o_plus) may
    /// not call it at all.
    fn target_len(&self) -> usize;

    /// The layer index this plan call handles (for d2o per-layer budget/protect decisions). The engine injects it while iterating layers,
    /// so the ctx maintains a single-layer view.
    fn layer_idx(&self) -> usize;

    /// flat per-token importance score. `Some` → score-based (heavy-hitter eviction, token-rank merge),
    /// `None` → score-free (sliding/streaming). For positional indexed access only (`imp.get(pos)`). The returned slice's
    /// borrow is bound to the ctx lifetime, keeping it dyn-safe.
    fn importance(&self) -> Option<&[f32]>;

    /// Number of KV heads. The upper bound of the h2o_plus per-head loop + the outer Vec length of [`KeepSpec::PerHead`], and d2o's
    /// `layer_dim = n_kv_heads * head_dim` computation. Engine impl source: `KVCache::kv_heads()`.
    fn n_kv_heads(&self) -> usize;

    /// Dimension per head. Determines d2o's K vector length / cosine dimension / dequant buffer size.
    /// Engine impl source: `KVCache::head_dim()`.
    fn head_dim(&self) -> usize;

    /// ★ **The single tensor-access mechanism** (D1 unification). Returns a handle if the given `kind` is available for this call, otherwise `None`
    /// (a score-free policy gives `tensor(Scores)==None`; a value-unaware/attn-unaware engine gives `None` for that kind).
    /// The returned handle's borrow is bound to the ctx lifetime, keeping it dyn-safe. All the sugar below sits on top of this.
    fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle>;

    // ── default sugar below (all delegate to `tensor()`). The engine need not override ──

    /// Fills raw K(`pos`,`head`) into `out` as f32 (for d2o cosine-nearest). Sugar over `tensor(Key)`.
    /// Contract: `out.len() == head_dim`. no-op if the kind is unavailable (out unchanged).
    fn dequant_k(&self, pos: usize, head: usize, out: &mut [f32]) {
        if let Some(h) = self.tensor(TensorKind::Key) {
            h.read_row(pos, head, out);
        }
    }

    /// Fills raw V(`pos`,`head`) into `out` as f32 (value-aware policies' v_i). Sugar over `tensor(Value)`.
    /// Contract: `out.len() == head_dim`. no-op if the kind is unavailable.
    fn dequant_v(&self, pos: usize, head: usize, out: &mut [f32]) {
        if let Some(h) = self.tensor(TensorKind::Value) {
            h.read_row(pos, head, out);
        }
    }

    /// per-head accumulated importance (h2o_plus). A flat `(kv_head, pos) → f32`. Sugar over `tensor(Scores)`.
    fn head_score(&self, kv_head: usize, pos: usize) -> f32 {
        match self.tensor(TensorKind::Scores) {
            Some(h) => {
                let mut o = [0.0f32];
                h.read_row(pos, kv_head, &mut o);
                o[0]
            }
            None => 0.0,
        }
    }

    /// Whether per-head scores exist. If `false`, h2o_plus degenerates to [`KeepSpec::LayerWide`].
    fn has_head_scores(&self) -> bool {
        self.tensor(TensorKind::Scores).is_some()
    }

    /// The previous decode step's per-head attention weight at `(kv_head, pos)` (value-aware policies' a_i). Sugar over `tensor(AttnWeights)`.
    /// If `has_attn_weights()==false` it is meaningless (0.0) — value-aware policies are advised to fall back to `importance()`.
    fn attn_weight(&self, kv_head: usize, pos: usize) -> f32 {
        match self.tensor(TensorKind::AttnWeights) {
            Some(h) => {
                let mut o = [0.0f32];
                h.read_row(pos, kv_head, &mut o);
                o[0]
            }
            None => 0.0,
        }
    }

    /// Whether attn_weight is populated (whether previous-step last-layer per-head attn is being tracked). Fall back if `false`.
    fn has_attn_weights(&self) -> bool {
        self.tensor(TensorKind::AttnWeights).is_some()
    }

    /// Total number of transformer layers (caches) in this eviction pass. Paired with [`layer_idx`](Self::layer_idx)
    /// for techniques that protect specific layers (e.g. d2o's last-layer protection: `layer_idx == n_layers - 1`).
    /// Default `0` — a single-layer view where no last-layer reasoning applies; the engine overrides it while iterating.
    fn n_layers(&self) -> usize {
        0
    }

    /// Whether the KV buffers live device-only (no CPU-accessible pointer), e.g. a discrete GPU.
    /// When `true`, a technique MUST NOT read raw K/V (`dequant_k`/`dequant_v` would fault) or emit
    /// [`WeightedMerge`]s (the engine merge executor is CPU-only); it should degrade to a keep-only plan.
    /// Default `false` — CPU-accessible (zero-copy / CPU backend), the common on-device case.
    fn kv_on_device(&self) -> bool {
        false
    }

    // ── constraint advertisement (default methods; the engine overrides, StageCtxAbi unchanged) ──
    //
    // A technique reads these BEFORE producing a plan to avoid emitting one the current container
    // cannot execute (the "expressible != executable" honesty boundary, surfaced up-front instead of
    // as a runtime executor reject). All are default methods so the ~13 existing `StageCtx` impls and
    // the C-ABI `StageCtxAbi` (a fixed fn-ptr table) compile unchanged.
    //
    // C-ABI CAVEAT: these are NOT flattened into `StageCtxAbi`, so a `.so` plugin reaching the host
    // through `AbiStageCtx` sees the DEFAULTS here (cache_dtype=F32, supports_per_head=false,
    // keep_granularity=1), not the real cache state — e.g. a Q4_0 cache reports keep_granularity=1 over
    // the C-ABI. There is no in-tree consumer yet; a future `.so` consumer must append a `cache_dtype`
    // fn-ptr (bumping `KV_STAGE_ABI_VERSION`) and override these on `AbiStageCtx`.

    /// The stored KV dtype of this layer's cache (for diagnostics / granularity reasoning).
    /// Default [`TensorDtype::F32`].
    fn cache_dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }

    /// Whether per-head compaction ([`KeepSpec::PerHead`]) is supported — `true` only on a HeadMajor
    /// cache (where a head's tokens are contiguous and shiftable in isolation). A per-head policy
    /// degrades to [`KeepSpec::LayerWide`] when this is `false`. Default `false`.
    fn supports_per_head(&self) -> bool {
        false
    }

    /// Whether a weighted merge ([`WeightedMerge`]) can be applied — the merge executor is CPU-only,
    /// so `false` on device-only KV. Default `= !kv_on_device()`.
    fn supports_merge(&self) -> bool {
        !self.kv_on_device()
    }

    /// The keep/merge position granularity: `1` for typed-float caches; the quant block length for
    /// block-quantized caches (a keep-set off this granularity forces a re-encode). Default `1`.
    fn keep_granularity(&self) -> usize {
        1
    }
}

/// A weighted merge instruction. Sums the evicted tokens (`from`) with weights into a single retained token's slot (`into`).
/// `Σ from.1 + into_weight ≈ 1` (magnitude preservation, the merge weights).
///
/// The `into`/`from` positions are logical coordinates just before compact is applied (pre-compact). The weights are baked into the plan,
/// and the engine executor (`apply_merges`) uses them as-is (replacing the current uniform merge). A merge-free policy uses an empty Vec.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMerge {
    /// The position of the retained token being merged into (the slot where the weighted sum accumulates).
    pub into: usize,
    /// The weight of `into` itself (the center weight `w_c` of the merge).
    pub into_weight: f32,
    /// The `(position, weight)` of the evicted tokens to be merged.
    pub from: Vec<(usize, f32)>,
    /// Which axis, K or V, the weighted merge applies to (WeightedKV, KV roadmap item 2). `Both` (default) =
    /// bit-identical to the old uniform-merge behavior. `ValueOnly` = discard K (excluded from merge) + weighted-merge V only.
    pub apply_to: MergeAxis,
}

/// The axis a weighted merge applies to (WeightedKV, KV roadmap item 2). `Both` = merge both K and V (bit-identical
/// to the old uniform behavior). `KeyOnly`/`ValueOnly` = merge only one buffer (the other is simply evicted).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergeAxis {
    /// Apply the weighted merge to both K and V (old behavior).
    #[default]
    Both,
    /// Merge the K buffer only; evict V.
    KeyOnly,
    /// Merge the V buffer only; evict K (WeightedKV).
    ValueOnly,
}

impl MergeAxis {
    /// Restores [`MergeAbi::apply_to`] u32 → enum. Unknown values (including old-plugin zero-init) fall back to `Both`.
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => MergeAxis::KeyOnly,
            2 => MergeAxis::ValueOnly,
            _ => MergeAxis::Both,
        }
    }
}

/// The shape of retained tokens — a mutually exclusive enum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeepSpec {
    /// sliding/h2o/streaming/no_eviction/d2o. **ascending**, prefix included.
    LayerWide(Vec<usize>),
    /// h2o_plus. `[n_kv_heads][keep]`, each ascending and of equal length (the engine asserts this).
    PerHead(Vec<Vec<usize>>),
}

/// The common parameters needed to create a technique instance. The engine maps CLI args into this struct and passes it along
/// (carrying only flat values so argus-extension-api does not depend on the engine's args type).
///
/// NOTE: technique-private parameters (e.g. d2o's `ema_beta`/`merge_e`/`merge_axis`/`protected_layers`)
/// are deliberately **not carried here** — rather than bloat this shared struct, they ride the opaque
/// [`StageArgs`] blob into [`MutationStageReg::make_with_args`], where the plugin parses its own params
/// (see `d2o::D2OConfig::from_args`). This keeps the engine from knowing any plugin's private knobs.
/// The 5 fields below are the common params shared by the built-ins (sliding/streaming/h2o/no_eviction).
#[repr(C)] // GATE-C: the `.so` C-ABI passes it by value as a POD (the make-thunk argument).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageParams {
    /// sliding window size (number of recent tokens to keep).
    pub eviction_window: usize,
    /// the prefix length to protect at the front (BOS / system prompt, etc.).
    pub protected_prefix: usize,
    /// heavy-hitter keep ratio (score-based eviction family).
    pub keep_ratio: f32,
    /// streaming sink (attention sink) size.
    pub sink_size: usize,
    /// streaming window size (if 0, the engine derives a default).
    pub streaming_window: usize,
}

/// One engine-supplied plugin argument: a `key=value` pair carrying a technique-private parameter
/// that does not fit the shared [`StageParams`] POD (e.g. d2o's `ema_beta`, `merge_axis`,
/// `protected_layers`). The plugin owns parsing, range-checks, and defaults for every key it
/// recognizes, and ignores keys it does not. This inverts the old coupling — the engine routes an
/// opaque blob, the plugin declares/receives its own params. `key`/`val` borrow from the caller for
/// the duration of the `make_with_args` call.
pub struct PluginArg<'a> {
    /// The parameter name (e.g. `"ema_beta"`).
    pub key: &'a str,
    /// The unparsed parameter value (e.g. `"0.7"`, `"value_only"`, `"0,1,27"`).
    pub val: &'a str,
}

/// The technique-private argument blob passed to [`MutationStageReg::make_with_args`]. Empty (`&[]`)
/// for built-ins served entirely by [`StageParams`].
pub type StageArgs<'a> = &'a [PluginArg<'a>];

/// Plugin-declared capabilities the engine reads **before** instantiating a stage (off the
/// [`MutationStageReg`], not via a trait method — the decision precedes `make`). This is the surface
/// that lets the engine CLI/chat/eval/bench paths stay free of any plugin-name knowledge: instead of
/// `matches!(name, "h2o" | "d2o" | ...)` capability lists and `match name { ... => 4 }` prefix
/// tables, each consumer reads these caps generically through [`stage_caps`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageCaps {
    /// The score tensors `plan()` consumes ([`TensorKind`]). A **non-empty** list ⟺ the stage is
    /// score-based: the engine wires an attention-score accumulator and routes per-token (and, for
    /// per-head stages, per-head) scores into the stage. **Empty** (`&[]`) = score-free
    /// (sliding/streaming/no-eviction). Replaces the former `is_score_based: bool` (read via
    /// [`stage_caps`] as `!reads.is_empty()`) — the per-kind list lets the future buffer-allocation
    /// handshake match a stage's `reads` against a [`ScoreProducer::produces`], instead of a single
    /// flag. It also replaces the scattered `matches!(name, "h2o" | "d2o" | ...)` capability checks.
    pub reads: &'static [TensorKind],
    /// The default `--protected-prefix` to apply when the user omits it. Score-based stages use `4`
    /// (attention sinks — protecting the whole prompt would defeat heavy-hitter selection); `0` means
    /// "no stage-declared default — the engine applies its own fallback" (sliding/streaming/none let
    /// the engine pick the recency/prompt-length default). Replaces the `match name { ... => 4 }`
    /// prefix tables.
    pub default_protected_prefix: usize,
    /// `true` ⟺ `plan()` may emit a non-empty `merges` vector (a weighted KV merge).
    /// The eval/QCF path reads this to pick a merge-compensation estimator + K readback
    /// instead of a pure-drop estimator. Replaces the engine-side `eviction_policy() == "d2o"` name
    /// match (the last STAGE-axis technique-name leak in eval).
    pub produces_merge_plan: bool,
    /// `true` ⟺ this stage decides over the WHOLE model at once: the engine invokes
    /// [`KVMutationStage::on_whole_model`] ONCE with a [`CrossLayerStageCtx`] spanning every layer and
    /// a model-scoped [`CacheHandle`] (verbs fan out to all layers), instead of the per-layer
    /// `on_phase` loop. A cross-layer global keep-set technique (TriAttention's default mode, which
    /// aggregates all layers' resident keys into ONE keep-set) sets this. Mirrors
    /// [`produces_merge_plan`](Self::produces_merge_plan): a per-stage declaration the engine reads
    /// pre-`make`, `false` for all but the relevant stage (default `false` → the per-layer path,
    /// byte-identical to before).
    pub whole_model: bool,
    /// The trailing query window the prefill-attention (PFA) producer should SUM over for this stage,
    /// or `None` (no preference — the engine arms its own default). A stage that reads
    /// [`TensorKind::PrefillAttention`] and scores like SnapKV (PyramidKV) declares its scoring
    /// `window_size` here so the engine observes EXACTLY that many trailing queries (a different
    /// window sums a different, differently-scaled query set and ranks different heavy hitters — the
    /// D1 divergence). A **declaration** the engine reads pre-`make` (off the [`MutationStageReg`]),
    /// not a runtime behavior — so it lives on the caps, not on a trait method (the engine no longer
    /// instantiates a throwaway default stage just to read this constant). `None` for the ~11
    /// score-free / non-PFA stages.
    pub prefill_attn_window: Option<usize>,
}

impl StageCaps {
    /// Score-free defaults — no reads, no stage-declared prefix, drop-only, per-layer, no PFA window
    /// (`{ &[], 0, false, false, None }`). Used by the `register_kv_mutation_stage!` macro so
    /// macro-registered (and example) plugins compile unchanged: a score-free drop-only LayerWide
    /// per-layer technique is the common case, and any stage that needs scores / emits merges / decides
    /// whole-model / declares a PFA window declares it via a direct-literal [`MutationStageReg`].
    pub const SCORE_FREE: StageCaps = StageCaps {
        reads: &[],
        default_protected_prefix: 0,
        produces_merge_plan: false,
        whole_model: false,
        prefill_attn_window: None,
    };
}

// ════════════════════════════════════════════════════════════════════════════
// HYBRID v3 — imperative CacheHandle mutation surface (M4 callback class)
// ════════════════════════════════════════════════════════════════════════════
//
// The plan-returning [`KVMutationStage`] above expresses one declarative shape per call (a keep-set +
// merges). HYBRID v3 adds an imperative sibling for the composite / stateful / escape-hatch class: a
// [`KVMutationStage`] callback that drives a transactional [`CacheHandle`]. The two coexist (this is
// purely additive); a technique picks whichever surface fits.
//
// The whole surface is rule-respecting and transactional — every guarantee below is enforced by the
// engine's `CacheHandle` impl, never by the plugin author:
//   T-1  position-mutating ops are STAGED, then committed once at callback end (single renumber).
//   T-2  at most ONE position-mutating compaction per callback; the 2nd is `MultipleCompactions`.
//   T-3  reads observe the pre-callback coordinate frame (no read-after-mutate).
//   T-5  position-PRESERVING ops (reencode/transition) are exempt from T-2 (free composition).
//   T-8  an `Err` leaves the cache bytes untouched (all-or-nothing).
//   T-9  a raw device buffer (cl_mem) is NEVER exposed through this surface.
//  T-10  keep-lists are validated ascending + unique + in-range before any mutation.

/// The lifecycle phase at which a [`KVMutationStage`] fires. A minimal, additive enum: `PrefillEnd`
/// (after prefill batch, before the decode loop) and `KvMutate` (mid-decode, the eviction / format
/// re-encode slot). A `PreAttention` phase is intentionally omitted — it has no engine lifecycle
/// mapping today. `#[repr(u32)]` so a future `.so` C-ABI passes the discriminant directly.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MutationPhase {
    /// After the prefill batch is materialized, before the first decode step.
    PrefillEnd,
    /// Mid-decode, the per-step KV mutation slot (eviction / merge / re-encode / offload).
    KvMutate,
}

/// Why a [`CacheHandle`] mutation op was rejected. Every variant is a clean logical error, not a
/// panic, and (transaction invariant T-8) leaves the cache bytes untouched — an `Err` is an
/// all-or-nothing abort, never a partial mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheOpError {
    /// More than one position-mutating compaction was requested in a single callback (T-2). At most
    /// one compaction (keep / evict / keep_per_head / offload / recall) may run per transaction so
    /// the single `current_pos` renumber stays unambiguous; the second op is rejected rather than
    /// composed with an implicit ordering.
    MultipleCompactions,
    /// A keep / evict list was not ascending + unique + in-range (T-10). This closes the historical
    /// leaky-keep hole where an unsorted/out-of-range keep silently corrupted the compaction.
    InvalidKeep,
    /// A geometry-narrowing op (`prune_channels` / `set_head_dim` / `project_rank`) was requested.
    /// head_dim and rank are a single per-cache scalar woven through every offset / alloc / kernel
    /// call, so no current container can store a narrowed or ragged geometry. Faithful channel /
    /// rank pruning needs an engine-core base-split (common to the declarative model too).
    GeometryImmutable,
    /// The op requires a container kind the current cache is not (e.g. `transition_quant_bits` on a
    /// `StandardFormat` cache, whose bit-width is not a runtime-transitionable property).
    WrongContainer,
    /// A host-only op was requested on a device-resident (GPU) cache that has no CPU pointer.
    NotOnHost,
    /// The op names a storage format the current backend / host path cannot materialize (e.g. an
    /// opaque `.so` codec on the typed-floor host re-encode path).
    UnsupportedFormat(String),
    /// The requested mutation would produce heterogeneous-within-layer state (per-head or per-token
    /// precision) that no current single-precision-per-layer container can hold.
    HeterogeneousUnsupported,
    /// A weighted merge named an `into` or `from` position outside `[0, current_pos)` (eager-rejected
    /// before any mutation — the merge twin of [`InvalidKeep`](Self::InvalidKeep)). Closes the
    /// out-of-range merge that would otherwise panic / silently corrupt in the CPU merge executor.
    InvalidMerge,
    /// A second weighted-merge batch was staged in one callback. Merge is position-preserving but the
    /// engine accepts at most one merge batch per transaction (distinct from
    /// [`MultipleCompactions`](Self::MultipleCompactions), which is about position-renumbering ops).
    MergeAlreadyStaged,
    /// `offload` / `recall` was requested on a handle with no residency (swap) backend configured.
    /// Rejected eagerly so the op never stages alongside a byte-mutating op that would then be
    /// orphaned by a commit-time failure.
    NoResidencyBackend,
}

impl core::fmt::Display for CacheOpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CacheOpError::MultipleCompactions => write!(
                f,
                "more than one position-mutating compaction in a single callback (T-2): at most one \
                 keep/evict/keep_per_head/offload/recall may run per transaction"
            ),
            CacheOpError::InvalidKeep => write!(
                f,
                "keep/evict list is not ascending + unique + in-range (T-10)"
            ),
            CacheOpError::GeometryImmutable => write!(
                f,
                "head_dim / rank geometry is immutable: no current container can store a narrowed or \
                 ragged geometry (needs an engine-core base-split)"
            ),
            CacheOpError::WrongContainer => {
                write!(
                    f,
                    "op requires a different KV container kind than the current cache"
                )
            }
            CacheOpError::NotOnHost => {
                write!(f, "host-only op requested on a device-resident (GPU) cache")
            }
            CacheOpError::UnsupportedFormat(name) => {
                write!(
                    f,
                    "op names a storage format this path cannot materialize: {name}"
                )
            }
            CacheOpError::HeterogeneousUnsupported => write!(
                f,
                "mutation would produce heterogeneous-within-layer precision no container can hold"
            ),
            CacheOpError::InvalidMerge => write!(
                f,
                "weighted merge names an into/from position outside [0, current_pos)"
            ),
            CacheOpError::MergeAlreadyStaged => {
                write!(
                    f,
                    "a second weighted-merge batch was staged in one callback"
                )
            }
            CacheOpError::NoResidencyBackend => write!(
                f,
                "offload/recall requested but this handle has no residency (swap) backend"
            ),
        }
    }
}

/// A rule-respecting, transactional mutation surface over **the scope handed to the stage** — one
/// layer's KV cache in the per-layer [`on_phase`](KVMutationStage::on_phase) path, ALL layers' caches
/// in the whole-model [`on_whole_model`](KVMutationStage::on_whole_model) path (where each verb fans
/// out to every layer). Handed to a [`KVMutationStage`] callback as `&mut dyn CacheHandle`. The verbs
/// are identical across both scopes; only the breadth they apply to differs (the engine supplies a
/// one-layer `EngineCacheHandle` or a model-scoped `EngineModelCacheHandle`).
///
/// **dyn-safe is mandatory** (the callback takes a `&mut dyn CacheHandle`): every method is free of
/// generic parameters / `Self` by value / associated types / `impl Trait` arguments. Like
/// [`StageCtx`], dtype-branching reads are exposed via the [`TensorHandle`] out-param convention.
///
/// Transaction model: reads observe the pre-callback frame (T-3); position-mutating ops are staged
/// and committed once by the engine at callback end (T-1/T-2); position-preserving ops are exempt
/// from the at-most-one rule (T-5); any `Err` leaves the bytes untouched (T-8); a raw device buffer
/// is never exposed (T-9). The dormant geometry walls (`prune_channels` / `set_head_dim` /
/// `project_rank`) return [`CacheOpError::GeometryImmutable`] via default methods — a future
/// narrowed-geometry container would override them.
pub trait CacheHandle {
    // ── reads (pre-callback frame, T-3) ──

    /// Current number of valid tokens (the compaction starting point). Pre-callback frame.
    fn current_pos(&self) -> usize;
    /// Number of KV heads.
    fn n_kv_heads(&self) -> usize;
    /// Dimension per head.
    fn head_dim(&self) -> usize;
    /// Whether the KV buffers live device-only (no CPU pointer). Host-only ops bail with
    /// [`CacheOpError::NotOnHost`] when this is `true`.
    fn kv_on_device(&self) -> bool;
    /// The single tensor-access mechanism (mirror of [`StageCtx::tensor`]). Reads observe the
    /// pre-callback frame. Returns `None` when the kind is unavailable for this call.
    fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle>;

    // ── token axis (S / L / N) — at most one position-mutating compaction per callback (T-2) ──

    /// Stage a keep-set of token positions (ascending + unique + in-range, validated eagerly per
    /// T-10). LayerWide (all heads keep the same positions). The engine compacts once at commit.
    fn keep(&mut self, keep: &[usize]) -> Result<(), CacheOpError>;

    /// Stage an evict-set; sugar for `keep(complement)`. The complement is taken over
    /// `[0, current_pos)` so the resulting keep-set is ascending + unique + in-range by construction.
    fn evict(&mut self, drop: &[usize]) -> Result<(), CacheOpError> {
        let n = self.current_pos();
        let keep: Vec<usize> = (0..n).filter(|p| !drop.contains(p)).collect();
        self.keep(&keep)
    }

    /// Stage a per-head keep-set (`[n_kv_heads][keep]`, each ascending + unique + in-range, all heads
    /// equal length — the engine's single shared `current_pos` invariant). Requires HeadMajor layout.
    fn keep_per_head(&mut self, keep: &[&[usize]]) -> Result<(), CacheOpError>;

    /// Stage weighted merges (summed in the pre-compaction frame, then compacted by a paired
    /// `keep`). Position-preserving in itself; composes with one compaction in the same callback.
    fn merge(&mut self, merges: &[WeightedMerge]) -> Result<(), CacheOpError>;

    // ── high-level rule-respecting ops (T1 compiler + combinators, worn as imperative methods) ──
    //
    // The dominant score-based eviction class becomes a single op call: the author supplies budgets +
    // a score fn (or component keep-sets), and the engine owns the top-k / recency / ascending re-sort
    // / composition — then stages the result through the validated low-level `keep` (T-10 + T-2). These
    // are default methods (`&dyn Fn` keeps the trait dyn-safe); a custom container needs only `keep`.

    /// Keep the canonical 3-partition top-k set (T1): compiles via [`compile_keep_top_k`] and stages
    /// it. `score(pos)` is the per-position ranking key over the heavy-hitter middle.
    fn keep_top_k(
        &mut self,
        spec: KeepTopK,
        score: &dyn Fn(usize) -> f32,
    ) -> Result<(), CacheOpError> {
        let keep = compile_keep_top_k(spec, score);
        self.keep(&keep)
    }

    /// Keep the intersection of `sets` (CAOTE meta-intersection) — stages [`keep_intersect`].
    fn keep_intersect_of(&mut self, sets: &[&[usize]]) -> Result<(), CacheOpError> {
        self.keep(&keep_intersect(sets))
    }

    /// Keep the union of `sets` (NaCl) — stages [`keep_union`].
    fn keep_union_of(&mut self, sets: &[&[usize]]) -> Result<(), CacheOpError> {
        self.keep(&keep_union(sets))
    }

    // ── format / precision axis (position-preserving, T-5: exempt from the at-most-one rule) ──

    /// Re-encode the resident tokens to `target` (typed floor f32/f16/q4_0). An opaque codec or a
    /// device-resident buffer bails with [`CacheOpError::UnsupportedFormat`] / `NotOnHost`.
    fn reencode(&mut self, target: FormatId) -> Result<(), CacheOpError>;

    /// Transition the per-layer quantization bit-width (quant-window container only). On a
    /// `StandardFormat` cache this is [`CacheOpError::WrongContainer`].
    fn transition_quant_bits(&mut self, bits: u8) -> Result<(), CacheOpError>;

    // ── residency (hardware) axis — compaction-slot ops (T-2) ──

    /// Offload the LRU prefix (`prefix_len` tokens) to the backing store. Requires a residency
    /// backend (else [`CacheOpError::NoResidencyBackend`], eager). NOTE: only F32/F16 host-resident KV
    /// is persisted to the store; a Q4_0 / non-persistable cache (or an unset store) degrades to a
    /// **lossy prune** of the prefix — those tokens are dropped, not recoverable by `recall`. An
    /// out-of-range `prefix_len` (0, or `>= current_pos` — "offload nothing / everything") is a silent
    /// no-op at commit but still CONSUMES the single per-callback compaction slot (T-2), so a later
    /// keep / recall / offload in the same callback would be rejected with `MultipleCompactions`.
    fn offload(&mut self, prefix_len: usize) -> Result<(), CacheOpError>;
    /// Recall this layer's most-recent outstanding offloaded prefix. Requires a residency backend
    /// (else [`CacheOpError::NoResidencyBackend`], eager). Recalls ONE record per call (the engine
    /// stages at most one residency op per callback, T-2); a layer with multiple outstanding offloads
    /// needs one `recall` per offload.
    fn recall(&mut self) -> Result<(), CacheOpError>;

    // ── dormant geometry walls (always Err today; common to the declarative model) ──

    /// Prune KEY head_dim channels (ThinK). Dormant: [`CacheOpError::GeometryImmutable`].
    fn prune_channels(&mut self, _keep: &[usize]) -> Result<(), CacheOpError> {
        Err(CacheOpError::GeometryImmutable)
    }
    /// Narrow head_dim. Dormant: [`CacheOpError::GeometryImmutable`].
    fn set_head_dim(&mut self, _head_dim: usize) -> Result<(), CacheOpError> {
        Err(CacheOpError::GeometryImmutable)
    }
    /// Project K/V to a lower rank (ShadowKV). Dormant: [`CacheOpError::GeometryImmutable`].
    fn project_rank(&mut self, _rank: usize) -> Result<(), CacheOpError> {
        Err(CacheOpError::GeometryImmutable)
    }
}

/// The whole-model read view handed to [`KVMutationStage::on_whole_model`] — the cross-layer sibling
/// of [`StageCtx`]. Where `StageCtx` is a single-layer view, this spans EVERY layer at once, so a
/// technique that must decide over all layers' resident keys together (TriAttention's global mode)
/// can read them through one ctx. dyn-safe (the callback takes `&dyn CrossLayerStageCtx`): every
/// method is free of generic params / `Self` by value / associated types / `impl Trait` args, and
/// tensor reads use the same [`TensorHandle`] out-param convention as `StageCtx`.
///
/// **Uniform-geometry precondition.** The whole-model path is defined only when every layer shares the
/// same resident length and the same per-slot absolute positions (the common single-prefill / uniform
/// decode case). The engine asserts this before building the ctx; `current_pos` / `abs_position` are
/// therefore model-wide, not per-layer.
pub trait CrossLayerStageCtx {
    /// Total number of transformer layers (caches) spanned.
    fn n_layers(&self) -> usize;
    /// The uniform resident token count (every layer equal — the precondition above).
    fn current_pos(&self) -> usize;
    /// The resolved budget — the absolute number of tokens to retain (the engine's ratio→len
    /// conversion already applied; the mirror of [`StageCtx::target_len`]).
    fn target_len(&self) -> usize;
    /// Number of KV heads (uniform across layers).
    fn n_kv_heads(&self) -> usize;
    /// Dimension per head.
    fn head_dim(&self) -> usize;
    /// The absolute RoPE position of cache slot `slot` — the engine source-of-truth (derived from its
    /// RoPE-continuation `saved_positions`), so the plugin's position frame never drifts from the
    /// engine's. Identity (`slot`) before any eviction; survivors carry their original positions after.
    fn abs_position(&self, slot: usize) -> usize;
    /// The `kind` tensor handle for `layer` (`StageCtx::tensor` with a layer argument; `TensorKind`
    /// reused). `None` when the kind is unavailable for this call (e.g. a device-only cache with no
    /// host mirror, or a kind the engine did not snapshot).
    fn layer_tensor(&self, layer: usize, kind: TensorKind) -> Option<&dyn TensorHandle>;

    // ── default sugar (delegates to `layer_tensor`; the engine need not override) ──

    /// Fills the post-RoPE key at `(layer, kv_head, slot)` into `out` as f32. Sugar over
    /// `layer_tensor(layer, Key)` (the `StageCtx::dequant_k` pattern). `out.len() == head_dim`;
    /// no-op if the Key kind is unavailable for `layer`.
    fn read_key(&self, layer: usize, kv_head: usize, slot: usize, out: &mut [f32]) {
        if let Some(h) = self.layer_tensor(layer, TensorKind::Key) {
            h.read_row(slot, kv_head, out);
        }
    }
}

/// The imperative sibling of [`KVMutationStage`] — a callback that drives a transactional
/// [`CacheHandle`] at a [`MutationPhase`]. Additive: a technique implements EITHER this or the
/// plan-returning [`KVMutationStage`], whichever its mutation shape fits.
///
/// The stage's lifecycle [`MutationPhase`] is NOT a trait method — it is declared once, at
/// registration, on [`MutationStageReg::phase`] (the single source of truth the engine reads
/// pre-`make` to place the stage). Carrying it on the trait too would let the registered phase and a
/// `phase()` return value disagree — the engine placing the stage at one phase while a driver
/// self-filters on the other, firing it at no phase. Keeping phase on the registration alone makes
/// that mismatch unrepresentable.
pub trait KVMutationStage: Send + Sync {
    /// The technique name (unique within the mutation-stage slice; CLI selector / logging).
    fn name(&self) -> &str;
    /// Drive the cache through the transactional handle. Reads observe the pre-callback frame; staged
    /// position-mutating ops are committed once when this returns `Ok`. An `Err` aborts the whole
    /// transaction (T-8: bytes untouched).
    fn on_phase(&self, ctx: &dyn StageCtx, cache: &mut dyn CacheHandle)
    -> Result<(), CacheOpError>;

    /// Drive a WHOLE-MODEL mutation: read every layer's resident KV through [`CrossLayerStageCtx`] and
    /// mutate through a model-scoped [`CacheHandle`] whose verbs fan out to all layers. The SAME shape
    /// as [`on_phase`](Self::on_phase) — only the READ scope differs (all layers vs one); the write
    /// generality is identical (the full [`CacheHandle`] verb set, not keep-only). The engine invokes
    /// this ONCE per eviction round for a stage that declares [`StageCaps::whole_model`], instead of
    /// the per-layer `on_phase` loop. The cross-layer global keep-set class (TriAttention's default
    /// mode — score all layers' keys, aggregate, apply ONE keep-set to every layer) lives here.
    ///
    /// Default no-op: the ~11 per-layer stages do not implement it (they declare `whole_model = false`
    /// and the engine never calls this on them). Reads observe the pre-callback frame and staged
    /// mutations commit once on `Ok`, exactly like `on_phase` (the same transaction invariants T-1..T-10).
    fn on_whole_model(
        &self,
        _ctx: &dyn CrossLayerStageCtx,
        _cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        Ok(())
    }
}

/// The canonical 3-partition keep-set shape (T1): `[0..prefix)` (protected) + the top-`heavy`
/// scorers over `[prefix..recent_start)` (re-sorted ascending) + `[recent_start..current)` (recent
/// window), where `recent_start = current.saturating_sub(recent).max(prefix)`.
///
/// This is the SINGLE shape the dominant score-based eviction class (H2O / StreamingLLM / sliding /
/// H2O+ / SnapKV / PyramidKV / …) reduces to. A policy supplies only the budgets (and a per-position
/// score function); [`compile_keep_top_k`] owns the recency window, the STABLE top-k selection, and
/// the ascending re-sort — so the policy is "score → budgets", not "hand-roll a keep-list".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeepTopK {
    /// Resident token count (the entry frame).
    pub current: usize,
    /// Protected prefix length (always kept; attention sinks / system prompt).
    pub prefix: usize,
    /// Recent-window token count (the trailing tokens always kept).
    pub recent: usize,
    /// Number of heavy hitters to keep from the middle `[prefix..recent_start)` by score.
    pub heavy: usize,
}

/// Compile the [`KeepTopK`] shape into an ascending, prefix-inclusive keep-list. `score(pos)` is the
/// per-position ranking key over the heavy-hitter range (use `|_| 0.0` for the score-free case, where
/// `heavy` is typically 0). The top-k uses a STABLE descending sort (ties keep input/position order),
/// matching the verbatim heavy-hitter selection the built-in eviction plugins ship — so routing a
/// plugin's keep-list assembly through this is byte-identical.
pub fn compile_keep_top_k(spec: KeepTopK, score: impl Fn(usize) -> f32) -> Vec<usize> {
    // Clamp the protected prefix to the resident count. When current < prefix (few tokens resident —
    // e.g. early in decode, or a prefix configured above the current occupancy) the correct keep-set is
    // the whole resident range, NOT a list containing indices >= current that the T-10 keep validator
    // would later reject as InvalidKeep. With prefix <= current this clamp is a no-op (byte-identical).
    let prefix = spec.prefix.min(spec.current);
    let recent_start = spec.current.saturating_sub(spec.recent).max(prefix);
    // (pos, score) over the evictable middle, STABLE sort desc, take top-`heavy`, re-sort ascending.
    let mut token_scores: Vec<(usize, f32)> = (prefix..recent_start)
        .map(|pos| (pos, score(pos)))
        .collect();
    token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    let mut heavy: Vec<usize> = token_scores
        .iter()
        .take(spec.heavy)
        .map(|(pos, _)| *pos)
        .collect();
    heavy.sort_unstable();
    let mut keep: Vec<usize> = (0..prefix).collect();
    keep.extend_from_slice(&heavy);
    keep.extend(recent_start..spec.current);
    keep
}

/// Intersect keep-sets into a single ascending, deduplicated keep-set (positions kept by ALL inputs).
/// Explicit composition for meta-intersection policies (CAOTE): the combination is visible at the
/// call site instead of riding an implicit op ordering. Empty input ⇒ empty.
pub fn keep_intersect(sets: &[&[usize]]) -> Vec<usize> {
    match sets.split_first() {
        None => Vec::new(),
        Some((first, rest)) => {
            let mut out: Vec<usize> = first.to_vec();
            out.sort_unstable();
            out.dedup();
            out.retain(|p| rest.iter().all(|s| s.contains(p)));
            out
        }
    }
}

/// Union keep-sets into a single ascending, deduplicated keep-set (positions kept by ANY input).
/// Explicit composition for union policies (NaCl). Empty input ⇒ empty.
pub fn keep_union(sets: &[&[usize]]) -> Vec<usize> {
    let mut out: Vec<usize> = sets.iter().flat_map(|s| s.iter().copied()).collect();
    out.sort_unstable();
    out.dedup();
    out
}

// ════════════════════════════════════════════════════════════════════════════
// v3 native registry — KV_MUTATION_STAGES (static-linkme only)
// ════════════════════════════════════════════════════════════════════════════
//
// The imperative [`KVMutationStage`] sibling of the plan-returning [`KVMutationStage`] needs its own
// registration slice so the engine can resolve a technique by name, read its [`StageCaps`] and
// [`MutationPhase`] BEFORE instantiation, and drive it through the transactional [`CacheHandle`].
// Mirrors [`MutationStageReg`] / [`KV_MUTATION_STAGES`], but static-linkme only: no `.so` C-ABI is built
// for the imperative path (a `CacheHandle` is a `&mut dyn` transaction engine, not a flat fn-ptr
// table that a stable C-ABI could carry). Caps + phase are carried in the registration so the engine
// reads them pre-`make`.

/// The registration entry for one imperative mutation-stage technique. A technique crate submits it
/// via `#[distributed_slice(KV_MUTATION_STAGES)] static FOO: MutationStageReg = ...` (or the
/// [`register_kv_mutation_stage!`] macro). The sibling of [`MutationStageReg`] for the v3 imperative
/// [`KVMutationStage`] surface.
pub struct MutationStageReg {
    /// The CLI selector name (`eviction plugin --name <name>`, or a built-in policy). Unique within
    /// the slice.
    pub name: &'static str,
    /// The factory that builds a technique instance from the common parameters + the technique-private
    /// [`StageArgs`] blob. (Unlike [`MutationStageReg`] there is no separate args-free `make`: the
    /// imperative path always routes args, and the macro wires an args-ignoring shim for techniques
    /// that take none.)
    pub make: fn(StageParams, StageArgs<'_>) -> Box<dyn KVMutationStage>,
    /// Capabilities the engine reads pre-`make` ([`StageCaps`]) — score reads, default protected
    /// prefix, merge-emitting. Read via [`mutation_stage_caps`] so consumers never name a plugin.
    pub caps: StageCaps,
    /// The lifecycle phase this stage fires at ([`MutationPhase`]) — the SINGLE source of truth for
    /// placement. The engine reads it pre-`make` to place the stage (PrefillEnd vs KvMutate); the
    /// driver fires the made stage at the SAME phase. [`KVMutationStage`] deliberately has no `phase()`
    /// method, so the registered phase and the runtime phase can never disagree.
    pub phase: MutationPhase,
}

/// The global mutation-stage registration slice — gathered at link time from all linked technique
/// crates (mirror of [`KV_MUTATION_STAGES`]). fat-LTO + `--gc-sections` may drop unreferenced sections,
/// so the engine force-links every technique crate; the startup self-test that asserts every expected
/// technique is registered lands with the production driver wiring (the mirror of
/// `ensure_builtin_stages_registered`).
#[distributed_slice]
pub static KV_MUTATION_STAGES: [MutationStageReg] = [..];

/// Finds a registered mutation-stage technique by name (used at engine construction).
///
/// During the v2→v3 migration window both [`KV_MUTATION_STAGES`] (plan path) and [`KV_MUTATION_STAGES`]
/// (this slice) may carry the SAME technique name. The selector namespace
/// (`eviction plugin --name <name>`) is shared, so the engine resolver MUST prefer this v3 slice (the
/// migration target) over the v2 one when a name resolves in both — `find_mutation_stage` first, then
/// `find_stage` as the legacy fallback. (The precedence is enforced engine-side at the selection seam;
/// these two lookups stay independent.)
pub fn find_mutation_stage(name: &str) -> Option<&'static MutationStageReg> {
    KV_MUTATION_STAGES.iter().find(|r| r.name == name)
}

/// All registered mutation-stage technique names (for self-test / diagnostics).
pub fn registered_mutation_names() -> Vec<&'static str> {
    KV_MUTATION_STAGES.iter().map(|r| r.name).collect()
}

/// The [`StageCaps`] of a statically registered mutation-stage technique, by name. `None` if the name
/// is not a statically linked mutation stage. The lookup the engine CLI/chat/eval/bench paths use to
/// read a stage's score-based-ness and default protected prefix without naming any plugin.
pub fn mutation_stage_caps(name: &str) -> Option<StageCaps> {
    find_mutation_stage(name).map(|r| r.caps)
}

/// Static registration macro for an imperative [`KVMutationStage`] technique (static-linkme only — no
/// `.so` C-ABI for the imperative path). The v3 counterpart of [`register_kv_mutation_stage!`].
///
/// Two forms — **`$phase` is always explicit** (there is no defaulting form: phase is the placement
/// source the engine reads pre-`make`, so a silent default would mis-place a `PrefillEnd` technique):
/// - 3-arg `($name, $make, $phase)` — a score-free, drop-only stage; `$make` is
///   `fn(StageParams) -> Box<dyn KVMutationStage>` (closures allowed) and the macro wires an
///   args-ignoring shim. Caps default to [`StageCaps::SCORE_FREE`].
/// - 4-arg `($name, $make, $caps, $phase)` — declare explicit [`StageCaps`]; `$make` is
///   `fn(StageParams, StageArgs) -> Box<dyn KVMutationStage>` (receives technique-private args).
///
/// **Callable multiple times** within a single crate: every contributed static is isolated in an
/// anonymous `const _: () = {}` scope so invocations do not collide (linkme does not rename a static
/// element's ident, so scope isolation is the only workaround).
///
/// ```ignore
/// argus_extension_api::register_kv_mutation_stage!(
///     "no_eviction", |_p| Box::new(NoEviction), MutationPhase::KvMutate);
/// ```
#[macro_export]
macro_rules! register_kv_mutation_stage {
    // ── score-free form: args-free make + explicit phase (caps default to SCORE_FREE) ──
    ($name:literal, $make:expr, $phase:expr) => {
        $crate::register_kv_mutation_stage!(
            $name,
            {
                // args-ignoring shim: the score-free form takes no technique-private args, so the
                // make-with-args shape drops the blob and delegates to the args-free `$make`.
                fn __mwa(
                    p: $crate::StageParams,
                    _args: $crate::StageArgs<'_>,
                ) -> ::std::boxed::Box<dyn $crate::KVMutationStage> {
                    let f: fn(
                        $crate::StageParams,
                    ) -> ::std::boxed::Box<dyn $crate::KVMutationStage> = $make;
                    f(p)
                }
                __mwa
            },
            $crate::StageCaps::SCORE_FREE,
            $phase
        );
    };
    // ── full form: explicit args-aware make + caps + phase ──
    ($name:literal, $make:expr, $caps:expr, $phase:expr) => {
        const _: () = {
            #[$crate::distributed_slice($crate::KV_MUTATION_STAGES)]
            static __MREG: $crate::MutationStageReg = $crate::MutationStageReg {
                name: $name,
                make: $make,
                caps: $caps,
                phase: $phase,
            };
        };
    };
}

// ── weight-axis dispatch types ──
//
// Surface types that express a weight stage plugin's dispatch decisions, isomorphic to KV's plan-returning.
// (`WeightStage`/`WeightDispatchPlan`/`WeightStageCtx` proper are introduced in MW-B; this stage only adds the dispatch-mode types.)

/// Plugin-surface mirror of the compute-location axis (hardware). 1:1 with the engine's `hardware::DeviceTarget` (the engine side
/// has bidirectional `From` + a drift gate). Defined separately so the api crate does not depend on the engine.
/// `#[repr(u32)]` lets a future `.so` C-ABI pass the discriminant directly.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTarget {
    Cpu,
    Gpu,
    Npu,
}

/// Dispatch mode for a weight layer. This is the **stage/hardware-axis mode** decided by the plugin.
/// Precision (the format axis) is kept separate via `LayerDirective.precision` (MW-B) rather than this enum (R1 orthogonality).
#[derive(Debug, Clone)]
pub enum LayerDispatch {
    /// 1-slice dense fast-path (bypasses the slice machinery).
    Full,
    /// 0-slice (layer skip; execution wiring is Phase β).
    Skip,
    /// N-slice composite, shares summing to ≈ 1.0.
    Partition(Vec<PartitionShare>),
}

/// **Plugin-decided coordinates** of a single partition slice = (share, hardware).
///
/// The per-slice storage format (precision) is **not a plugin decision but derived by the executor from the weight dtype**,
/// so it is excluded from this surface.
/// Rationale: a split's byte layout comes from the weight tensor's actual dtype (the engine's `bytes_per_row`), and the old
/// `SliceSpec.format` was merely an equality assert against it. Narrowing the surface to the limited `TensorDtype` (3 variants) would regress
/// Q4_1/Q8_0/BF16/U8 among the current 7-dtype partition, so the format stays executor-internal (the full `DType`).
#[derive(Debug, Clone)]
pub struct PartitionShare {
    /// What fraction of the weight this slice covers (along the out_dim axis).
    pub share: f32,
    /// The hardware location where this slice is resolved.
    pub hardware: DeviceTarget,
}

// ── weight stage plugin (isomorphic to KVMutationStage) ──

/// Kinds of per-layer metric a weight stage reads. The kind argument of `WeightStageCtx::layer_metric`
/// (mirror of KV's `TensorKind`). `#[repr(u32)]` is for passing the discriminant directly across a future `.so` C-ABI.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerMetricKind {
    /// per-layer importance (one axis of the swap-ranking key). The engine impl flattens the ImportanceTable's
    /// `SubLayer::Full` projection to provide it (reducing the sublayer dimension of entries()).
    Importance,
    /// per-layer quantization noise ε (the ε axis of the swap-ranking key).
    QuantNoise,
}

/// Read-only context that a weight stage plugin reads (mirror of KV's `StageCtx`, dyn-safe).
/// The engine implements it over `&TransformerModel` (MW-D). No mutation rights — the plugin only reads and returns a plan (D1/D3).
pub trait WeightStageCtx {
    /// Total number of decoder layers.
    fn n_layers(&self) -> usize;
    /// Swap budget resolved by the engine = an **absolute layer count** (the engine handles ratio→count + subtracting currently_swapped +
    /// boundary protection; mirror of KV's `target_len`).
    fn budget(&self) -> usize;
    /// Graded memory pressure 0–100 (for pressure-driven stages).
    fn pressure(&self) -> u8;
    /// The current storage dtype of the given layer.
    fn current_format(&self, layer: usize) -> TensorDtype;
    /// ★ Single accessor for per-layer metrics (mirror of KV's `tensor(kind)`, OCP). `None` when the kind is unavailable.
    /// The returned slice's borrow is tied to the ctx lifetime (dyn-safe). Length = `n_layers()`.
    fn layer_metric(&self, kind: LayerMetricKind) -> Option<&[f32]>;

    // ── default sugar (all delegate to `layer_metric`). The engine need not override ──

    /// per-layer importance. Sugar over `layer_metric(Importance)`.
    fn importance(&self) -> Option<&[f32]> {
        self.layer_metric(LayerMetricKind::Importance)
    }
    /// per-layer quantization noise. Sugar over `layer_metric(QuantNoise)`.
    fn quant_noise(&self) -> Option<&[f32]> {
        self.layer_metric(LayerMetricKind::QuantNoise)
    }
}

/// Dispatch directive for a single layer (D2). dispatch (stage/hardware axis) ⊥ precision (format axis, R1).
#[derive(Debug, Clone)]
pub struct LayerDirective {
    /// Index of the target decoder layer.
    pub layer: usize,
    /// Dispatch mode (Full / Skip / Partition).
    pub dispatch: LayerDispatch,
    /// Target dtype for a precision swap. `None` = keep the current dtype. Orthogonal to dispatch (R1).
    pub precision: Option<TensorDtype>,
}

/// A weight stage's plan output (mirror of the KV stage axis). Rust-native data holding decisions only
/// (step/boundary-tier, no repr(C) needed). Mutation is performed by the engine executor (D3).
#[derive(Debug, Clone, Default)]
pub struct WeightDispatchPlan {
    /// Per-layer directives. Empty means no-op.
    pub per_layer: Vec<LayerDirective>,
}

/// Plan-returning technique trait for the weight axis (mirror of KV's `KVMutationStage`).
pub trait WeightStage: Send + Sync {
    /// Technique name (canonical stage name; unique within the slice).
    fn name(&self) -> &str;
    /// Reads the ctx and returns a dispatch plan. `None` = no-op (not applied).
    fn plan(&self, ctx: &dyn WeightStageCtx) -> Option<WeightDispatchPlan>;
}

/// CLI-derived static configuration for a weight stage (mirror of KV's `StageParams`).
///
/// Currently holds only the swap builtin's static knobs. The runtime value (swap ratio) comes not from params but
/// from `WeightStageCtx::budget` (command-driven). Per-builtin extra fields are extended when MW-C is wired up
/// (isomorphic to KV's `StageParams` 4-field-vs-opaque open question).
#[derive(Debug, Clone, Copy)]
pub struct WeightStageParams {
    /// Include boundary layers (0, last) as swap targets too (research/ablation; production default false).
    pub allow_boundary_layers: bool,
}

/// Registration entry for a single weight stage technique (mirror of KV's `MutationStageReg`).
pub struct WeightStageReg {
    /// canonical stage name (matches the resilience `EngineCommand` → name normalization table, Seam C).
    pub name: &'static str,
    /// Factory that builds a technique instance from the parameters.
    pub make: fn(WeightStageParams) -> Box<dyn WeightStage>,
}

/// Global weight stage registration slice — the **4th parallel registry** of the stage axis.
/// linkme gathers them at link time; the startup self-test guarding against fat-LTO `--gc-sections` lives on the engine side (MW-C).
#[distributed_slice]
pub static WEIGHT_STAGES: [WeightStageReg] = [..];

/// Finds a registered weight stage by name (used during engine construction).
pub fn find_weight_stage(name: &str) -> Option<&'static WeightStageReg> {
    WEIGHT_STAGES.iter().find(|r| r.name == name)
}

/// Names of all registered weight stages (for self-test / diagnostics).
pub fn registered_weight_names() -> Vec<&'static str> {
    WEIGHT_STAGES.iter().map(|r| r.name).collect()
}

// ── Format-axis plugin registry (isomorphic to KVMutationStage) ──

/// How a quantization block stores its scale (block-quant family vocabulary).
///
/// `#[repr(u32)]`: a future `.so` C-ABI passes the fieldless discriminant through as-is (L1).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleLayout {
    /// No scale (f32/f16 raw).
    None,
    /// A single f16 scale per block (q4_0/q8_0).
    PerBlockF16,
    /// An f16 scale + f16 min per block (q4_1).
    PerBlockF16WithMin,
}

/// How a quantization block packs its bits (block-quant family vocabulary).
///
/// `#[repr(u32)]`: a future `.so` C-ABI passes the fieldless discriminant through as-is (L1). New
/// variants are **appended** (existing discriminants unchanged) so the wire layout stays additive.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    /// Contiguous raw (f32/f16).
    Dense,
    /// nibble (4-bit) packing (q4_0/q4_1).
    Nibble,
    /// byte (8-bit) packing (q8_0).
    Byte,
    /// quad (2-bit, 4 elems/byte) packing (q2_0).
    Quad,
}

/// layer-tier boundary POD — the format plugin's **actual contribution**.
///
/// Holds only block-quant family vocabulary (`block_elems`/`bits`/`scale_layout`/`packing`):
/// q4_0/q4_1/q8_0/q5 etc. are driven through this descriptor by the generic floor (dequant→f32 matmul, M-F3).
/// mxfp4 shared-exponent / codebook / sparse fall outside the floor → backend-specific opt-in escape (D5).
///
/// `#[repr(C)]`: a flat POD that crosses a future `.so` C-ABI boundary as-is (L1 gate — kept repr(C) now
/// to avoid a forced reshape at `.so` time).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KVLayoutDesc {
    /// Number of elements in one quantization block (q4_0/q8_0 = 32). raw (f32/f16) formats are 1.
    pub block_elems: u32,
    /// Bits per element (q4_0 = 4, q8_0 = 8, f16 = 16, f32 = 32).
    pub bits: u8,
    /// Scale storage layout.
    pub scale_layout: ScaleLayout,
    /// Bit packing layout.
    pub packing: Packing,
}

impl KVLayoutDesc {
    /// Raw bytes of one block-quant block (scale + packed quants). raw (`Dense`) has no block concept,
    /// so it returns `None` (per-element accounting is in [`Self::bytes_for_elems`]).
    ///
    /// single source of truth for byte accounting — the engine's `dequant_via_descriptor` inline formula
    /// (formerly `dtype_layout.rs`) and `OpaqueBuffer` alloc share this method.
    pub fn block_bytes(&self) -> Option<usize> {
        let quant_bytes = match self.packing {
            Packing::Dense => return None,
            Packing::Nibble => self.block_elems as usize / 2,
            Packing::Byte => self.block_elems as usize,
            Packing::Quad => self.block_elems as usize / 4,
        };
        let scale_bytes = match self.scale_layout {
            ScaleLayout::None => 0,
            ScaleLayout::PerBlockF16 => 2,
            ScaleLayout::PerBlockF16WithMin => 4,
        };
        Some(scale_bytes + quant_bytes)
    }

    /// Total bytes to store `numel` elements in this layout.
    ///
    /// raw (`Dense`) = `numel * (bits/8)` (f32=4, f16/bf16=2), block-quant =
    /// `(numel / block_elems) * block_bytes`. For block-quant, if `numel` is not a multiple of `block_elems`,
    /// returns `None` (partial blocks are not allowed).
    pub fn bytes_for_elems(&self, numel: usize) -> Option<usize> {
        match self.block_bytes() {
            None => Some(numel * (self.bits as usize / 8)),
            Some(block_bytes) => {
                let be = self.block_elems as usize;
                if be == 0 || !numel.is_multiple_of(be) {
                    return None;
                }
                Some((numel / be) * block_bytes)
            }
        }
    }
}

/// Format-axis plugin trait — describes the storage layout.
///
/// layer-tier COMPUTE (`write_kv`/`attention_into`) is not in this trait — that is owned by the hardware axis's
/// M×N kernel cell, owned by the backend (D4). A format plugin is a pure descriptor (2 methods: name+layout).
///
/// NOTE(phasing, S4-2 2026-06-07): step-tier `compact` is **not added** to this trait —
/// superseded. The keep/merge decision belongs to the stage axis ([`KVMutationStage`]),
/// while mutation is owned exclusively by the engine (decision=plugin, mutation=engine). Pulling compact into the format axis
/// would blur the stage⊥format orthogonality at the decision layer and leak `Merge` (engine) into the api. Therefore
/// `KVFormat`'s layer-tier contribution is only the `layout()` descriptor read (M-F2, the L1 repr(C) boundary).
pub trait KVFormat: Send + Sync {
    /// canonical format name (e.g. "q4_0"/"f16"/"f32"). Unique within the slice.
    fn name(&self) -> &str;

    /// This format's storage layout descriptor (read by the engine's generic reader on the hot path, D3).
    fn layout(&self) -> KVLayoutDesc;
}

/// Registration entry for one format technique (mirror of KV `MutationStageReg`).
pub struct KVFormatReg {
    /// Canonical format name. Unique within the slice.
    pub name: &'static str,
    /// Format instance factory.
    pub make: fn() -> Box<dyn KVFormat>,
}

/// Global format registration slice — one of the three parallel per-axis registries.
///
/// The fat-LTO `--gc-sections` silent-drop risk is gated by an engine startup self-test at the point
/// the first builtin registration appears (M-F3), isomorphic.
#[distributed_slice]
pub static KV_FORMATS: [KVFormatReg] = [..];

/// Looks up a registered format by name (used during engine construction).
pub fn find_kv_format(name: &str) -> Option<&'static KVFormatReg> {
    KV_FORMATS.iter().find(|r| r.name == name)
}

/// All registered format names (for self-test / diagnostics).
pub fn registered_kv_format_names() -> Vec<&'static str> {
    KV_FORMATS.iter().map(|r| r.name).collect()
}

// ════════════════════════════════════════════════════════════════════════════
// D3 — KVFormatPlan: the dual of the KV residency decision (format/precision assignment)
// ════════════════════════════════════════════════════════════════════════════
//
// The stage axis answers "which tokens stay" (residency); KVFormatPlan answers "in what format is each
// region stored" (precision/layout) — a near-uniform total function over layer x head x token x {K,V}.
// Static-linkme only (no C-ABI yet; a repr(C) projection is a separate tail-append PR when a real
// `.so` codec author appears). The executor (`apply_format_plan`) and its `FormatApplyError` live
// engine-side (applying a format touches engine container types); this api surface is plan-only.
//
// HONESTY (verified): the value types below are fully constructible, but on current containers only a
// uniform-per-layer assignment is even a candidate for execution — per-head / per-token heterogeneous
// precision is structurally unholdable (one dtype per `KVCache` layer buffer; one bit-width per
// quant-window layer), so the engine executor REJECTS such plans rather than silently mis-storing.
// "Expressible (a well-formed plan value) != executable (the engine can re-materialize it)".

/// A storage-format identity = a registry **name** only — not an enum, not a `{name,bits}` pair.
/// Every precision detail (bit-width, scale layout, codebook/LUT, per-channel scale, pre-RoPE) lives
/// BEHIND the name (inside the codec), which keeps the format set open to novel codecs (e.g. a
/// backend-cap GPU decoder) without engine-core edits. Bit variants are distinct names
/// (`"q2"`/`"q4_0"`/`"f16"`), exactly as the floor already distinguishes them via [`KVFormatReg::name`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatId(pub String);

/// Pins one region to one registered storage `format` on one K/V `side`. Reuses [`KeepSpec`] for the
/// token region and [`MergeAxis`] for the side — zero new selector/side vocabulary.
#[derive(Clone, Debug, PartialEq)]
pub struct FormatOverride {
    /// The token region this override covers (the same selector the eviction axis uses).
    pub region: KeepSpec,
    /// The registered storage format applied to `region`.
    pub format: FormatId,
    /// Which of K / V / both this override applies to.
    pub side: MergeAxis,
}

/// The dual of the KV residency decision: `base` + an ordered, last-wins `overrides` list encodes a
/// near-uniform total function over layer x head x token x {K,V}. Empty `overrides` <=> uniform
/// `base` <=> today's behavior (Gate-0: byte-identical when `base` equals the current stored format).
#[derive(Clone, Debug, PartialEq)]
pub struct KVFormatPlan {
    /// The default format applied where no override matches.
    pub base: FormatId,
    /// Ordered overrides; later entries win (see [`KVFormatPlan::format_of`]).
    pub overrides: Vec<FormatOverride>,
}

impl KVFormatPlan {
    /// Resolver: the last override matching `(head, token, side)`, else `base`. Total + deterministic
    /// — this is the semantics of the base+override encoding, and it is what makes the Gate-0 identity
    /// hold (empty overrides => `base` everywhere).
    pub fn format_of(&self, head: usize, token: usize, side: MergeAxis) -> &FormatId {
        self.overrides
            .iter()
            .rev()
            .find(|o| side_matches(o.side, side) && region_contains(&o.region, head, token))
            .map(|o| &o.format)
            .unwrap_or(&self.base)
    }
}

/// `Both` matches any queried side; otherwise the override side must equal the queried side.
fn side_matches(over: MergeAxis, queried: MergeAxis) -> bool {
    matches!(over, MergeAxis::Both) || over == queried
}

/// Whether a [`KeepSpec`] region contains `token` for `head` (token-position membership).
fn region_contains(region: &KeepSpec, head: usize, token: usize) -> bool {
    match region {
        KeepSpec::LayerWide(positions) => positions.contains(&token),
        KeepSpec::PerHead(per_head) => per_head.get(head).is_some_and(|p| p.contains(&token)),
    }
}

/// The dynamic format-assignment producer — the third sibling of [`KVMutationStage::on_phase`] and
/// `WeightStage::plan`, reusing the SAME [`StageCtx`] read seam. `assign` returns `None` for "no
/// change" (uniform base kept), the safe default.
pub trait KVFormatPolicy: Send + Sync {
    /// Policy name (unique within the slice; CLI selector / logging).
    fn name(&self) -> &str;

    /// Computes the format assignment from read-only ctx. `None` = no change (base kept).
    fn assign(&self, ctx: &dyn StageCtx) -> Option<KVFormatPlan>;

    /// (g2) The formats the current backend can decode; the engine rejects a plan that names an
    /// unsupported format rather than silently mis-storing. Default = empty = unconstrained (the
    /// engine validates against the registries).
    fn supported_formats(&self, _ctx: &dyn StageCtx) -> Vec<FormatId> {
        Vec::new()
    }
}

/// Registration entry for one format policy — the 4th per-axis producer registry (mirror of
/// [`MutationStageReg`] / [`KVFormatReg`]). Its register symbol is never unified with the others (D6).
pub struct KVFormatPolicyReg {
    /// Policy name. Unique within the slice.
    pub name: &'static str,
    /// Policy factory from the common params.
    pub make: fn(StageParams) -> Box<dyn KVFormatPolicy>,
    /// The tensor kinds this policy reads (producer<->consumer handshake; mirror of `StageCaps.reads`).
    pub reads: &'static [TensorKind],
}

/// Global format-policy registration slice — one of the parallel per-axis producer registries.
#[distributed_slice]
pub static KV_FORMAT_POLICIES: [KVFormatPolicyReg] = [..];

/// Looks up a registered format policy by name.
pub fn find_format_policy(name: &str) -> Option<&'static KVFormatPolicyReg> {
    KV_FORMAT_POLICIES.iter().find(|r| r.name == name)
}

/// All registered format-policy names (for self-test / diagnostics).
pub fn registered_format_policy_names() -> Vec<&'static str> {
    KV_FORMAT_POLICIES.iter().map(|r| r.name).collect()
}

// ════════════════════════════════════════════════════════════════════════════
// GATE-C v2 — Format-axis `.so` cdylib dlopen plugin C-ABI
// ════════════════════════════════════════════════════════════════════════════
//
// Isomorphic to the stage axis (GATE-C above) but **overwhelmingly simpler**: [`KVFormat`] has zero callbacks (no ctx,
// no plan — a pure descriptor, `name`+`layout`), so the stage axis ctx/plan marshalling is all
// unnecessary. The vtable carries only `make` (opaque handle) + `layout` ([`KVLayoutDesc`] POD by-value) + `drop`.
// `KVLayoutDesc`/`ScaleLayout`/`Packing` are already `#[repr(C)]`/`#[repr(u32)]`, so they pass through by value
// as-is on the fn-ptr return (zero reshape). A plugin exports only the single `register_kv_format_v1() -> *const FormatVTableAbi`
// (D6 landmine: the stage/format/backend register symbols are kept separate — never unify them).

/// ABI version of the `register_kv_formats_v2` envelope ([`FormatExportAbi`]). The host refuses to load on mismatch.
pub const KV_FORMAT_ABI_VERSION: u32 = 2;

/// C-ABI flattening of [`KVFormat`] (D4). The single vtable a plugin exports. `name` is the static identifier
/// (`'static`) used for registry matching; `layout` returns the handle instance's [`KVLayoutDesc`] as POD by-value.
///
/// There is no counterpart to the stage's `plan`/`plan_free` (arena marshalling) — the descriptor is a stack POD, so
/// no cross-allocator boundary arises ("each side frees its own" applies only to the handle lifecycle, `make`/`drop`).
///
#[repr(C)]
pub struct FormatVTableAbi {
    /// Null-terminated canonical name (`--kv-format`/registry matching). A `'static` str in the plugin `.so`.
    /// (ABI gating is handled by the envelope [`FormatExportAbi::abi_version`] — no per-vtable version field.)
    pub name: *const c_char,
    /// Creates a format instance → opaque handle. Called by the host on `make_format`.
    pub make: unsafe extern "C" fn() -> *mut c_void,
    /// Handle → [`KVLayoutDesc`] (POD by-value). The very descriptor the engine generic floor reads (D3).
    pub layout: unsafe extern "C" fn(*mut c_void) -> KVLayoutDesc,
    /// Releases the format instance handle (called by the host when the format is dropped).
    pub drop: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: the vtable is immutable and `name` points to a `'static` str in the plugin `.so`. fn-ptrs are inherently
// Send+Sync. Required for the plugin's distributed_slice element static declaration.
unsafe impl Sync for FormatVTableAbi {}

/// Format-axis envelope — declares all of a plugin `.so`'s format vtables at once. `register_kv_formats_v2()`
/// returns it **by-value** (sret >16B; `count`/`vtables` are derived from the slice at runtime, so a const static is impossible).
/// `vtables` is the [`PLUGIN_KV_FORMAT_VTABLES`] base (`.so` static) → valid for the `.so`'s lifetime; `count==0` is allowed (empty axis).
#[repr(C)]
pub struct FormatExportAbi {
    /// [`KV_FORMAT_ABI_VERSION`]. The host rejects the `.so` on mismatch (one ABI per `.so`).
    pub abi_version: u32,
    /// Length of the contiguous array `vtables` points to.
    pub count: usize,
    /// A contiguous array of `count` [`FormatVTableAbi`] (`.so` static). The loader accesses elements via `vtables.add(i)`.
    pub vtables: *const FormatVTableAbi,
}

/// Slice that accumulates format vtables inside a plugin `.so`. **Declared in exactly one place, argus-extension-api**
/// (the linkme section name is determined by the declaring static's name — a plugin-side declaration would break cross-crate contributions). `register_kv_format!`
/// contributes a const-block-isolated static under the plugin-cdylib gate (multiple calls = multiple formats). In a static build it is left empty
/// and harmless (the engine reads only `KV_FORMATS`).
#[distributed_slice]
pub static PLUGIN_KV_FORMAT_VTABLES: [FormatVTableAbi] = [..];

/// Dual-wiring macro that registers a format plugin both statically (rlib→linkme) and dynamically (cdylib→C-ABI) (D4).
///
/// `$make` is the same `fn() -> Box<dyn KVFormat>` as the existing [`KVFormatReg::make`] (a closure is allowed). The dynamic
/// C-ABI export (`register_kv_format_v1`) is gated on the `plugin-cdylib` feature so that, under static force-link,
/// `#[no_mangle]` symbol collisions are ruled out at the source (only the `.so` build uses `--features plugin-cdylib`). The format-axis
/// counterpart of [`register_kv_mutation_stage!`].
///
/// **May be called multiple times** within one plugin crate (`.so`) (multiple formats = a quant family). All contributed
/// statics are isolated in anonymous `const _: () = {}` scopes. The `.so` entry (`register_kv_formats_v2`) is emitted separately,
/// once per `.so`, by [`export_plugin!`]. The format-axis counterpart of [`register_kv_mutation_stage!`].
///
/// ```ignore
/// argus_extension_api::register_kv_format!("nf4",  || Box::new(Nf4));
/// argus_extension_api::register_kv_format!("awq4", || Box::new(Awq4));   // multiple formats in one .so
/// argus_extension_api::export_plugin!();   // once per .so
/// ```
#[macro_export]
macro_rules! register_kv_format {
    ($name:literal, $make:expr) => {
        // ── Static path (rlib → linkme distributed_slice). const-block isolation = multiple calls allowed (E2). ──
        const _: () = {
            #[$crate::distributed_slice($crate::KV_FORMATS)]
            static __REG: $crate::KVFormatReg = $crate::KVFormatReg {
                name: $name,
                make: $make,
            };
        };

        // ── Dynamic path (cdylib → contributes to PLUGIN_KV_FORMAT_VTABLES). Gated on plugin-cdylib, so not emitted in a static build. ──
        // The entry (register_kv_formats_v2) is emitted by export_plugin!; here we only contribute the vtable to the slice (E2).
        #[cfg(feature = "plugin-cdylib")]
        const _: () = {
            // Handle = Box<Box<dyn KVFormat>> (thin ptr). make/layout/drop share this representation.
            type __Handle = ::std::boxed::Box<dyn $crate::KVFormat>;

            unsafe extern "C" fn __make() -> *mut ::core::ffi::c_void {
                // $make (a Rust-ABI fn) is for internal calls here only — never cast it directly to extern "C".
                let make_fn: fn() -> __Handle = $make;
                let fmt: __Handle = make_fn();
                ::std::boxed::Box::into_raw(::std::boxed::Box::new(fmt)) as *mut ::core::ffi::c_void
            }

            unsafe extern "C" fn __layout(h: *mut ::core::ffi::c_void) -> $crate::KVLayoutDesc {
                // SAFETY: h is the Box<Box<dyn>> created by __make. layout() returns a POD (no arena needed).
                let fmt: &dyn $crate::KVFormat = unsafe { &**(h as *const __Handle) };
                fmt.layout()
            }

            unsafe extern "C" fn __drop(h: *mut ::core::ffi::c_void) {
                // SAFETY: h is the Box<Box<dyn>> created by __make; the host calls this exactly once.
                drop(unsafe { ::std::boxed::Box::from_raw(h as *mut __Handle) });
            }

            // Contribute the vtable to PLUGIN_KV_FORMAT_VTABLES (const-block isolation = accumulates across multiple calls). Not an entry.
            #[$crate::distributed_slice($crate::PLUGIN_KV_FORMAT_VTABLES)]
            static __VTABLE: $crate::FormatVTableAbi = $crate::FormatVTableAbi {
                name: ::core::concat!($name, "\0").as_ptr() as *const ::core::ffi::c_char,
                make: __make,
                layout: __layout,
                drop: __drop,
            };
        };
    };
}

/// Called **once** per `.so` — emits this plugin's per-axis entry symbols. plugin-cdylib gate:
/// not emitted in a static force-link build (prevents entry collisions across multiple force-linked plugins). Returns the
/// [`PLUGIN_KV_FORMAT_VTABLES`] / [`PLUGIN_BACKEND_CAP_VTABLES`] slices accumulated by the format/backend register macros as by-value envelopes.
///
/// **Two-axis separate-symbol invariant** (the stage axis is static-linkme only): `register_kv_formats_v2` ⊥
/// `register_backend_caps_v2` — separate entries + separate slices, not a unified symbol/registry (they are merely emitted
/// together for the author's convenience). The backend axis (the third) was added by the D8 implementation.
///
/// An axis with zero contributions yields a `count==0` envelope (an empty distributed_slice — ELF `__start==__stop`, safe).
///
/// ```ignore
/// argus_extension_api::register_kv_format!("nf4", || Box::new(Nf4));
/// argus_extension_api::export_plugin!();
/// ```
#[macro_export]
macro_rules! export_plugin {
    () => {
        #[cfg(feature = "plugin-cdylib")]
        const _: () = {
            // The KV stage axis is static-linkme only (no `.so` C-ABI); a `.so` exports just the
            // format and backend-cap axes. `.len()`/`.as_ptr()` are evaluated at runtime via linkme's
            // Deref (static_slice) — each envelope is computed at call time.

            /// Format envelope entry — returns `PLUGIN_KV_FORMAT_VTABLES` by-value (sret).
            #[unsafe(no_mangle)] // Rust 2024: no_mangle is an unsafe attribute.
            pub extern "C" fn register_kv_formats_v2() -> $crate::FormatExportAbi {
                $crate::FormatExportAbi {
                    abi_version: $crate::KV_FORMAT_ABI_VERSION,
                    count: $crate::PLUGIN_KV_FORMAT_VTABLES.len(),
                    vtables: $crate::PLUGIN_KV_FORMAT_VTABLES.as_ptr(),
                }
            }

            /// Backend-cap envelope entry (third axis, D8) — returns `PLUGIN_BACKEND_CAP_VTABLES` by-value (sret).
            #[unsafe(no_mangle)]
            pub extern "C" fn register_backend_caps_v2() -> $crate::BackendCapExportAbi {
                $crate::BackendCapExportAbi {
                    abi_version: $crate::BACKEND_CAP_ABI_VERSION,
                    count: $crate::PLUGIN_BACKEND_CAP_VTABLES.len(),
                    vtables: $crate::PLUGIN_BACKEND_CAP_VTABLES.as_ptr(),
                }
            }
        };
    };
}

// ── Backend-capability axis plugin registry ──

/// Backend capability plugin trait — a specialized opt-in capability layered on backend-owned kernels.
///
/// Skeleton only (D6): the first instance, such as GpuFold (step5, beyond this crate's stage), will finalize the methods. The backend
/// always provides the generic floor (descriptor-driven dequant→f32) and specializes only the hot path via this capability.
pub trait BackendCapability: Send + Sync {
    /// Canonical capability name (e.g. "gpu_fold"). Unique within the slice.
    fn name(&self) -> &str;
}

/// Registration entry for one backend capability (mirror of KV `MutationStageReg`).
pub struct BackendCapReg {
    /// Canonical capability name. Unique within the slice.
    pub name: &'static str,
    /// Capability instance factory.
    pub make: fn() -> Box<dyn BackendCapability>,
}

/// Global backend-capability registration slice — one of the three parallel per-axis registries.
#[distributed_slice]
pub static BACKEND_CAPABILITIES: [BackendCapReg] = [..];

/// Looks up a registered capability by name.
pub fn find_backend_capability(name: &str) -> Option<&'static BackendCapReg> {
    BACKEND_CAPABILITIES.iter().find(|r| r.name == name)
}

/// All registered capability names (for self-test / diagnostics).
pub fn registered_backend_capability_names() -> Vec<&'static str> {
    BACKEND_CAPABILITIES.iter().map(|r| r.name).collect()
}

// ── Backend-capability axis — ATTENTION (quantized fused attention, e.g. quant-window) category dynamic C-ABI ──
//
// D8 (single-trait): argus-extension-api owns the canonical [`QuantAttnBackend`]. The engine's static OpenCL
// impl, the host dlopen adapter, and the plugin `.so` **all implement this one trait** (isomorphic to the stage `KVMutationStage`). The signatures take
// ABI structs (`QuantAttnArgs`/`QuantAttnGatherArgs`, cl_mem `*mut c_void`) rather than `&Tensor`, so the plugin does not reference engine
// types (independent). The static `BACKEND_CAPABILITIES` above (keyed by name) stays as-is — for the fat-LTO name-survival smoke test.

/// ABI version of the `register_backend_caps_v2` envelope ([`BackendCapExportAbi`]). The host rejects the `.so` on mismatch.
///
/// v2 (FORMAT Phase 2, Stage A): [`QuantAttnVTable`] gained `dequant_flush`/`scatter_residual`
/// fn-ptrs so the residual-flush GPU kernels cross the cap trait (closing the engine's
/// concrete-`OpenCLBackend` downcast on the live FORMAT-flush path). A v1 `.so` is rejected.
pub const BACKEND_CAP_ABI_VERSION: u32 = 2;

/// Capability category tag — ATTENTION (quantized fused dequant+attention, e.g. quant-window). [`BackendCapVTableAbi::category`].
/// The host's category bridge (`match`) uses this value to cast the `vtable` pointer to its per-category table ([`QuantAttnVTable`]) (D7).
pub const BACKEND_CAP_CATEGORY_ATTENTION: u32 = 1;

/// Arguments for creating a quantized fused-attention capability instance (D4). Using the GPU context/device borrowed from the host plus build options, the plugin
/// builds its kernels **once** and produces an opaque handle. Bare-C handles only (C4) — no `ocl` wrapper types.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantAttnMakeArgs {
    /// `cl_context` raw handle (owned by the host backend, borrow-for-make).
    pub cl_ctx: *mut c_void,
    /// `cl_device_id` raw handle.
    pub device: *mut c_void,
    /// Null-terminated OpenCL build options (the result of the host's `build_cl_opts(device)` — Adreno consistency, C7). May be null.
    pub build_opts: *const c_char,
}

/// Arguments for a quantized fused dequant+attention call (D6). All GPU resources are **borrow-for-call** (C5: no retain).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantAttnArgs {
    /// `cl_command_queue` raw handle (lent by the host).
    pub cl_queue: *mut c_void,
    pub q_mem: *mut c_void,
    pub qk_mem: *mut c_void,
    pub qv_mem: *mut c_void,
    pub res_k_mem: *mut c_void,
    pub res_v_mem: *mut c_void,
    pub out_mem: *mut c_void,
    /// CPU score readback buffer (optional; null = no scores). `scores_len` f32 values.
    pub scores_out: *mut f32,
    pub scores_len: usize,
    pub num_heads_q: usize,
    pub num_heads_kv: usize,
    pub head_dim: usize,
    pub q_tokens: usize,
    pub res_tokens: usize,
    pub res_cap: usize,
    pub scale: f32,
    pub bits: u8,
}

/// Arguments for a quantized-attention residual gather-update call (D6). Two mem (input/residual) + five scalars.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantAttnGatherArgs {
    pub cl_queue: *mut c_void,
    pub input_mem: *mut c_void,
    pub residual_mem: *mut c_void,
    pub kv_heads: usize,
    pub res_cap: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub res_pos: usize,
}

/// Arguments for a residual-flush dequant call (FORMAT Phase 2, Stage A). The GPU half of a
/// KVCache residual-ring flush: dequantize a freshly-written block range into the F16
/// attention buffer. One struct serves both the per-channel **key** and per-token **value**
/// kernels (identical arity); `is_key` selects which, and `n_groups_or_tokens` carries the K
/// `groups_per_flush` or the V `flush_tokens`. cl_mem is **borrow-for-call** (C5).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantDequantFlushArgs {
    /// `cl_command_queue` raw handle. The engine's static impl uses its own queue (mirrors
    /// [`QuantAttnArgs::cl_queue`]); a plugin uses the queue it built in `make`. May be null.
    pub cl_queue: *mut c_void,
    /// Quantized block buffer (source).
    pub q_blocks_mem: *mut c_void,
    /// F16 attention buffer (destination).
    pub attn_mem: *mut c_void,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// K: `groups_per_flush` (per-channel); V: `flush_tokens` (per-token).
    pub n_groups_or_tokens: usize,
    pub tok_base: usize,
    /// Block index the freshly-uploaded blocks start at (the kernel's `block_offset`).
    pub block_start: usize,
    pub bits: u8,
    /// true → per-channel key dequant; false → per-token value dequant.
    pub is_key: bool,
}

/// Arguments for scattering the F32 residual ring into the F16 attention buffer (Stage A).
/// cl_mem is **borrow-for-call** (C5).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantScatterResidualArgs {
    /// `cl_command_queue` raw handle (see [`QuantDequantFlushArgs::cl_queue`]). May be null.
    pub cl_queue: *mut c_void,
    /// F32 residual ring buffer (source).
    pub res_mem: *mut c_void,
    /// F16 attention buffer (destination).
    pub attn_mem: *mut c_void,
    pub kv_heads: usize,
    pub res_cap: usize,
    pub head_dim: usize,
    pub res_pos: usize,
    /// Token base in the attention buffer (the assembled-view's quantized-token count).
    pub tok_base: usize,
}

/// Canonical capability trait for the ATTENTION category (D8 single-trait). Owned by argus-extension-api → the engine's static impl,
/// the host dlopen adapter, and the plugin `.so` **all implement this one trait**. It takes an ABI struct instead of `&Tensor` so the plugin is
/// independent (does not reference engine types). Returns `i32` (C3 panic=abort: 0=OK, negative=err — vtable fn-ptrs must not panic).
pub trait QuantAttnBackend: Send + Sync {
    /// Whether a fused quantized-attention kernel for `bits` (2/4/8) is available.
    fn has_quant_attn_kernel(&self, bits: u8) -> bool;
    /// Whether the device lacks sub-group support (Adreno nosub) — used to select the kernel variant.
    fn is_nosub_device(&self) -> bool;
    /// Fused dequant+attention. cl_mem lives inside [`QuantAttnArgs`], borrow-for-call (C5).
    fn attention_gen_quant(&self, args: &QuantAttnArgs) -> i32;
    /// Residual ring gather-update (just before K/V quantization).
    fn gather_update_quant(&self, args: &QuantAttnGatherArgs) -> i32;
    /// Residual-flush dequant (per-channel key / per-token value) — the GPU half of a KVCache
    /// flush. Default `-1` (unsupported) so existing impls compile; the host's OpenCL backend
    /// and the dlopen adapter override it (FORMAT Phase 2, Stage A).
    fn dequant_flush(&self, _args: &QuantDequantFlushArgs) -> i32 {
        -1
    }
    /// Scatter the F32 residual ring into the F16 attention buffer. Default `-1` (unsupported).
    fn scatter_residual(&self, _args: &QuantScatterResidualArgs) -> i32 {
        -1
    }
}

/// Static (force-link) quantized-attention capability registration entry — the backend-axis counterpart of the stage [`MutationStageReg`] (D8).
/// `make` is called only when the host has a GPU context (`QuantAttnMakeArgs`); the fat-LTO survival smoke test checks the name only.
pub struct QuantAttnReg {
    /// Canonical capability name. Unique within the slice.
    pub name: &'static str,
    /// Capability instance factory (builds the kernels once using the host GPU context, D4).
    pub make: fn(&QuantAttnMakeArgs) -> Box<dyn QuantAttnBackend>,
}

/// Global static registration slice for quantized-attention capabilities (linkme). Contributed to by `register_quant_attn_plugin!`.
/// Separate from the dynamic dlopen path ([`PLUGIN_BACKEND_CAP_VTABLES`]) — the host merges them for source-agnostic lookup (mirror of D3).
#[distributed_slice]
pub static QUANT_ATTN_REGS: [QuantAttnReg] = [..];

/// Looks up a statically registered quantized-attention capability by name.
pub fn find_quant_attn(name: &str) -> Option<&'static QuantAttnReg> {
    QUANT_ATTN_REGS.iter().find(|r| r.name == name)
}

/// All statically registered quantized-attention capability names (for the fat-LTO survival smoke test / diagnostics).
pub fn registered_quant_attn_names() -> Vec<&'static str> {
    QUANT_ATTN_REGS.iter().map(|r| r.name).collect()
}

/// ATTENTION category C-ABI vtable (D7) — the table [`BackendCapVTableAbi::vtable`] points to when category==ATTENTION.
/// make/drop live here too (make's arguments are the per-category [`QuantAttnMakeArgs`], so they cannot go in the common header).
#[repr(C)]
pub struct QuantAttnVTable {
    /// [`QuantAttnMakeArgs`] → opaque plugin handle (one-time kernel build, D4). Called on host `make`.
    pub make: unsafe extern "C" fn(*const QuantAttnMakeArgs) -> *mut c_void,
    /// handle + bits → bool for whether the kernel is present.
    pub has_quant_attn_kernel: unsafe extern "C" fn(*mut c_void, u8) -> bool,
    /// handle → nosub-device bool.
    pub is_nosub_device: unsafe extern "C" fn(*mut c_void) -> bool,
    /// handle + [`QuantAttnArgs`] → i32 (0=OK, negative=err). Per-token hot path.
    pub attention_gen_quant: unsafe extern "C" fn(*mut c_void, *const QuantAttnArgs) -> i32,
    /// handle + [`QuantAttnGatherArgs`] → i32. residual gather-update.
    pub gather_update_quant: unsafe extern "C" fn(*mut c_void, *const QuantAttnGatherArgs) -> i32,
    /// handle + [`QuantDequantFlushArgs`] → i32. residual-flush dequant (ABI v2).
    pub dequant_flush: unsafe extern "C" fn(*mut c_void, *const QuantDequantFlushArgs) -> i32,
    /// handle + [`QuantScatterResidualArgs`] → i32. residual scatter (ABI v2).
    pub scatter_residual: unsafe extern "C" fn(*mut c_void, *const QuantScatterResidualArgs) -> i32,
    /// Release the handle (called once by the host when the capability is dropped).
    pub drop: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: the vtable is immutable and fn-ptrs are inherently Send+Sync. Required to declare a distributed_slice element static.
unsafe impl Sync for QuantAttnVTable {}

/// backend-cap axis entry (D7 tagged pointer) — a thin `{name, category, vtable}`. The actual functions live in
/// a per-category table (e.g. [`QuantAttnVTable`]). The host casts `vtable` using `category` (the category bridge).
#[repr(C)]
pub struct BackendCapVTableAbi {
    /// null-terminated canonical name (for registry matching). A `'static` str in the plugin `.so`.
    /// (ABI gating is handled by the envelope's [`BackendCapExportAbi::abi_version`] — there is no per-vtable version field.)
    pub name: *const c_char,
    /// Category tag (e.g. [`BACKEND_CAP_CATEGORY_ATTENTION`]). The host's `match` key.
    pub category: u32,
    /// Pointer to the per-category `#[repr(C)]` table (e.g. `*const QuantAttnVTable`). The host casts it using `category`.
    pub vtable: *const c_void,
}

// SAFETY: immutable, and name/vtable are `'static` in the `.so`. Required for a distributed_slice element static.
unsafe impl Sync for BackendCapVTableAbi {}

/// backend-cap axis envelope — declares all of one `.so`'s capability vtables at once. `register_backend_caps_v2()`
/// returns it **by-value** (sret). `vtables` points at the [`PLUGIN_BACKEND_CAP_VTABLES`] base → valid for the `.so`'s lifetime; `count==0` is allowed.
#[repr(C)]
pub struct BackendCapExportAbi {
    /// [`BACKEND_CAP_ABI_VERSION`]. On a host mismatch the `.so` is rejected (one ABI per `.so`).
    pub abi_version: u32,
    /// Length of the contiguous `vtables` array.
    pub count: usize,
    /// A contiguous array of `count` [`BackendCapVTableAbi`] entries (`.so` static). The loader uses `vtables.add(i)`.
    pub vtables: *const BackendCapVTableAbi,
}

/// Slice that accumulates backend-cap vtables inside a plugin `.so`. **Declared in exactly one place: argus-extension-api.**
/// `register_quant_attn_plugin!` contributes to it under the plugin-cdylib gate. Harmlessly empty in static builds.
#[distributed_slice]
pub static PLUGIN_BACKEND_CAP_VTABLES: [BackendCapVTableAbi] = [..];

/// dual-wiring macro (D8) that registers a quantized-attention capability plugin both statically (rlib → linkme name survival)
/// and dynamically (cdylib → C-ABI vtable). `$make` = `fn(&QuantAttnMakeArgs) -> Box<dyn QuantAttnBackend>` (a closure is allowed).
///
/// **Static path**: contributes `$make` to the [`QUANT_ATTN_REGS`] slice (name survives under force-link — for the fat-LTO survival
/// smoke test, where `registered_quant_attn_names()` checks the name). **Dynamic path** (plugin-cdylib): wraps `$make` and the trait methods
/// in C thunks and contributes a [`QuantAttnVTable`] + envelope entry to [`PLUGIN_BACKEND_CAP_VTABLES`]. The `.so` entry point
/// (`register_backend_caps_v2`) is emitted by [`export_plugin!`]. May be invoked multiple times within one `.so` (multiple capabilities).
#[macro_export]
macro_rules! register_quant_attn_plugin {
    ($name:literal, $make:expr) => {
        // ── Static path (rlib → linkme QUANT_ATTN_REGS, name survives under force-link). Ungated (common to both builds). ──
        // Store `$make` in a live distributed_slice static → the static-lookup infrastructure, and even feature-OFF builds,
        // keep `$make`/its associated types reachable (isomorphic to the Stage `register_kv_mutation_stage!`, no unused warnings).
        const _: () = {
            #[$crate::distributed_slice($crate::QUANT_ATTN_REGS)]
            static __REG: $crate::QuantAttnReg = $crate::QuantAttnReg {
                name: $name,
                make: $make,
            };
        };

        // ── Dynamic path (cdylib → QuantAttnVTable + envelope entry). Gated by plugin-cdylib, so not emitted in static builds. ──
        #[cfg(feature = "plugin-cdylib")]
        const _: () = {
            // handle = Box<Box<dyn QuantAttnBackend>> (thin ptr). All thunks share this representation.
            type __Handle = ::std::boxed::Box<dyn $crate::QuantAttnBackend>;

            unsafe extern "C" fn __make(
                p: *const $crate::QuantAttnMakeArgs,
            ) -> *mut ::core::ffi::c_void {
                // SAFETY: the host passes a valid QuantAttnMakeArgs pointer (D4). QuantAttnMakeArgs is a Copy POD.
                let args: &$crate::QuantAttnMakeArgs = unsafe { &*p };
                let make_fn: fn(&$crate::QuantAttnMakeArgs) -> __Handle = $make;
                let be: __Handle = make_fn(args);
                ::std::boxed::Box::into_raw(::std::boxed::Box::new(be)) as *mut ::core::ffi::c_void
            }

            unsafe extern "C" fn __has(h: *mut ::core::ffi::c_void, bits: u8) -> bool {
                // SAFETY: h is the Box<Box<dyn>> created by __make.
                let be: &dyn $crate::QuantAttnBackend = unsafe { &**(h as *const __Handle) };
                be.has_quant_attn_kernel(bits)
            }

            unsafe extern "C" fn __nosub(h: *mut ::core::ffi::c_void) -> bool {
                // SAFETY: same as above.
                let be: &dyn $crate::QuantAttnBackend = unsafe { &**(h as *const __Handle) };
                be.is_nosub_device()
            }

            unsafe extern "C" fn __attn(
                h: *mut ::core::ffi::c_void,
                a: *const $crate::QuantAttnArgs,
            ) -> i32 {
                // SAFETY: h is the Box<Box<dyn>> created by __make; a is a valid QuantAttnArgs filled in by the host (C5).
                let be: &dyn $crate::QuantAttnBackend = unsafe { &**(h as *const __Handle) };
                be.attention_gen_quant(unsafe { &*a })
            }

            unsafe extern "C" fn __gather(
                h: *mut ::core::ffi::c_void,
                a: *const $crate::QuantAttnGatherArgs,
            ) -> i32 {
                // SAFETY: same as above.
                let be: &dyn $crate::QuantAttnBackend = unsafe { &**(h as *const __Handle) };
                be.gather_update_quant(unsafe { &*a })
            }

            unsafe extern "C" fn __dequant_flush(
                h: *mut ::core::ffi::c_void,
                a: *const $crate::QuantDequantFlushArgs,
            ) -> i32 {
                // SAFETY: h is the Box<Box<dyn>> created by __make; a is a valid QuantDequantFlushArgs (C5).
                let be: &dyn $crate::QuantAttnBackend = unsafe { &**(h as *const __Handle) };
                be.dequant_flush(unsafe { &*a })
            }

            unsafe extern "C" fn __scatter_residual(
                h: *mut ::core::ffi::c_void,
                a: *const $crate::QuantScatterResidualArgs,
            ) -> i32 {
                // SAFETY: same as above.
                let be: &dyn $crate::QuantAttnBackend = unsafe { &**(h as *const __Handle) };
                be.scatter_residual(unsafe { &*a })
            }

            unsafe extern "C" fn __drop(h: *mut ::core::ffi::c_void) {
                // SAFETY: h is the Box<Box<dyn>> created by __make; the host calls this exactly once.
                drop(unsafe { ::std::boxed::Box::from_raw(h as *mut __Handle) });
            }

            static __VTABLE: $crate::QuantAttnVTable = $crate::QuantAttnVTable {
                make: __make,
                has_quant_attn_kernel: __has,
                is_nosub_device: __nosub,
                attention_gen_quant: __attn,
                gather_update_quant: __gather,
                dequant_flush: __dequant_flush,
                scatter_residual: __scatter_residual,
                drop: __drop,
            };

            // Contributes the envelope entry to PLUGIN_BACKEND_CAP_VTABLES (const-block isolation = accumulation across multiple invocations). Not the entry point.
            #[$crate::distributed_slice($crate::PLUGIN_BACKEND_CAP_VTABLES)]
            static __ENTRY: $crate::BackendCapVTableAbi = $crate::BackendCapVTableAbi {
                name: ::core::concat!($name, "\0").as_ptr() as *const ::core::ffi::c_char,
                category: $crate::BACKEND_CAP_CATEGORY_ATTENTION,
                vtable: &__VTABLE as *const $crate::QuantAttnVTable as *const ::core::ffi::c_void,
            };
        };
    };
}

// ════════════════════════════════════════════════════════════════════════════
// CACHE category — stateful quantized-KV cache construction (FORMAT Phase 2, Stage C).
// A second backend-cap category alongside ATTENTION. It rides the SAME envelope
// (`BackendCapVTableAbi {name, category, vtable}` / `register_backend_caps_v2` /
// `BACKEND_CAP_ABI_VERSION`) — only the `category` tag is new. The CACHE vtable owns
// the cache *construction + lifecycle* (a stateful per-layer quantized cache: residual
// ring + cold quantized store), distinct from ATTENTION which is the stateless fused
// dequant+attention compute. Allocation invariant (mirror of QuantAttn): the engine
// pre-allocates every GPU buffer at `max_seq_len` upfront and lends `cl_mem`; the plugin
// owns layout policy + kernels, never bytes. There is no allocator callback into the
// engine (an engine alloc after a cached OpenCL Plan caches `cl_mem` would stale the
// Plan → garbage/crash), so this ABI carries only borrowed `cl_mem` and POD scalars.
// ════════════════════════════════════════════════════════════════════════════

/// Capability category tag — CACHE (stateful quantized-KV cache construction, e.g. quant-window).
/// [`BackendCapVTableAbi::category`]. The host's category bridge casts the `vtable` pointer
/// to [`QuantCacheVTable`] when `category == BACKEND_CAP_CATEGORY_CACHE` (D7). A fresh value
/// (ATTENTION=1 is taken) — a new category is a new host `match` arm = host recompile (C1).
pub const BACKEND_CAP_CATEGORY_CACHE: u32 = 2;

/// Closed layout vocabulary for the assembled K/V view the engine wraps in a tensor.
/// Keeps this ABI from naming the engine's `KVLayout`. `HeadMajor` = `[tokens, kv_heads,
/// head_dim]` (GPU assembled), `SeqMajor` = `[kv_heads, tokens, head_dim]` (CPU/residual).
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewLayoutTag {
    /// `[tokens, kv_heads, head_dim]`.
    HeadMajor = 0,
    /// `[kv_heads, tokens, head_dim]`.
    SeqMajor = 1,
}

/// Arguments for creating a quantized-KV cache instance (one per transformer layer). Bare-C
/// GPU handles borrowed-for-make (the plugin builds its kernels once, mirror of
/// [`QuantAttnMakeArgs`]) plus the cache geometry the engine passes typed today. The engine
/// pre-allocates at `max_seq_len` (Plan-safe); the plugin allocates nothing. No engine type.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantCacheMakeArgs {
    /// `cl_context` raw handle (owned by the host backend, borrow-for-make). Null in CPU mode.
    pub cl_ctx: *mut c_void,
    /// `cl_device_id` raw handle. Null in CPU mode.
    pub device: *mut c_void,
    /// `cl_command_queue` raw handle (lent). Null in CPU mode.
    pub cl_queue: *mut c_void,
    /// Null-terminated OpenCL build options (host `build_cl_opts(device)`, Adreno C7). May be null.
    pub build_opts: *const c_char,
    /// Number of KV heads (GQA).
    pub kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Maximum sequence length — the engine pre-allocs every buffer at this upfront (Plan-safe).
    pub max_seq_len: usize,
    /// FP residual-ring capacity (tokens kept unquantized before a flush).
    pub residual_size: usize,
    /// Quantization bit-width (2/4/8, or 16 = unquantized FP fallback).
    pub bits: u8,
}

/// Arguments for writing a new K/V step into the cache (D6). All GPU resources are
/// **borrow-for-call** (C5: no retain). One handle is one layer, so no layer index.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantCacheUpdateArgs {
    /// `cl_command_queue` raw handle (lent). May be null (impl uses its own queue).
    pub cl_queue: *mut c_void,
    /// New-token key `cl_mem` (engine-marshalled from its tensor). Null in CPU mode.
    pub k_in_mem: *mut c_void,
    /// New-token value `cl_mem`.
    pub v_in_mem: *mut c_void,
    /// Number of new tokens written (1 = decode, >1 = prefill batch).
    pub seq_len: usize,
}

/// Out-parameter the plugin fills with the `cl_mem` of its assembled K/V view; the engine
/// wraps each in a tensor (the underlying buffer is engine-owned → zero-alloc reuse) and
/// runs standard attention on it. Mirrors `QuantizedRecentWindowCache::get_view` for the GPU regime.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantCacheViewOut {
    /// Assembled key view `cl_mem` (engine-owned buffer the plugin wrote into).
    pub k_mem: *mut c_void,
    /// Assembled value view `cl_mem`.
    pub v_mem: *mut c_void,
    /// Valid token count in the view.
    pub tokens: usize,
    /// View layout ([`ViewLayoutTag`] as `u32`).
    pub layout: u32,
}

/// Out-parameter the plugin fills with the raw quantized + residual `cl_mem` set the ATTENTION
/// cap (`attention_gen_quant`) consumes directly on the fused-native path — four separate
/// buffers, NOT one assembled view. Mirrors the cache's raw-buffer accessor. The vtable fn
/// returns `false` (and leaves this untouched) when there is no native-consumable set (CPU /
/// bits=16 / empty).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantCacheRawBuffersOut {
    /// Quantized key blocks `cl_mem`.
    pub qk_mem: *mut c_void,
    /// Quantized value blocks `cl_mem`.
    pub qv_mem: *mut c_void,
    /// FP residual key ring `cl_mem`.
    pub res_k_mem: *mut c_void,
    /// FP residual value ring `cl_mem`.
    pub res_v_mem: *mut c_void,
    /// Quantized-token count.
    pub q_tokens: usize,
    /// Residual (unquantized) token count.
    pub res_tokens: usize,
    /// Residual-ring capacity.
    pub res_cap: usize,
    /// Quantization bit-width of the blocks.
    pub bits: u8,
}

/// Canonical capability trait for the CACHE category (D8 single-trait, mirror of
/// [`QuantAttnBackend`]). Owned by argus-extension-api → the host dlopen adapter and the
/// plugin `.so` both implement it. Takes ABI POD (`cl_mem` as `*mut c_void`, never `&Tensor`)
/// so the plugin references no engine type. Work fns return `i32` (0=OK, negative=err;
/// panic=abort discipline). `make`/`drop` live in [`QuantCacheVTable`] (lifecycle, not trait).
pub trait QuantCacheBackend: Send + Sync {
    /// Number of valid tokens currently in the cache.
    fn current_pos(&self) -> usize;
    /// Physical buffer capacity in tokens.
    fn capacity(&self) -> usize;
    /// Current quantization bit-width (2/4/8/16).
    fn current_bits(&self) -> u8;
    /// Write a new K/V step. `cl_mem` lives in [`QuantCacheUpdateArgs`], borrow-for-call (C5).
    fn update(&self, args: &QuantCacheUpdateArgs) -> i32;
    /// Flush the residual ring into the quantized store if it is full (idempotent no-op if not).
    fn flush_if_full(&self) -> i32;
    /// Fill `out` with the assembled K/V view `cl_mem`; the engine wraps it in tensors.
    fn assemble_view(&self, out: &mut QuantCacheViewOut) -> i32;
    /// Fill `out` with the raw quantized + residual `cl_mem` set for the fused-native attention
    /// path; return `false` (leaving `out` untouched) when no native set exists (CPU/16/empty).
    fn get_raw_buffers(&self, out: &mut QuantCacheRawBuffersOut) -> bool;
    /// Transition the quantization bit-width at runtime (F16↔Q2/Q4/Q8). Same-bits = no-op.
    fn transition_bits(&self, target_bits: u8) -> i32;
}

/// Static (force-link) quantized-KV cache registration entry — the CACHE-category counterpart
/// of [`QuantAttnReg`] (D8). `make` builds the kernels once from the host GPU context.
/// The built-in engine quant-window does **not** register here (its construction needs engine
/// `Backend`/`Memory` handles a POD `make` cannot carry — that lives engine-side); this slice
/// is populated by out-of-tree cache plugins.
pub struct QuantCacheReg {
    /// Canonical capability name. Unique within the slice.
    pub name: &'static str,
    /// Cache instance factory (builds the kernels once using the host GPU context, D4).
    pub make: fn(&QuantCacheMakeArgs) -> Box<dyn QuantCacheBackend>,
}

/// Global static registration slice for quantized-KV cache capabilities (linkme). Contributed
/// to by `register_quant_cache_plugin!`. Separate from the dynamic dlopen path
/// ([`PLUGIN_BACKEND_CAP_VTABLES`]) — the host merges them for source-agnostic lookup.
#[distributed_slice]
pub static QUANT_CACHE_REGS: [QuantCacheReg] = [..];

/// Looks up a statically registered quantized-KV cache capability by name.
pub fn find_quant_cache(name: &str) -> Option<&'static QuantCacheReg> {
    QUANT_CACHE_REGS.iter().find(|r| r.name == name)
}

/// All statically registered quantized-KV cache capability names (diagnostics / smoke test).
pub fn registered_quant_cache_names() -> Vec<&'static str> {
    QUANT_CACHE_REGS.iter().map(|r| r.name).collect()
}

/// CACHE category C-ABI vtable (D7) — the table [`BackendCapVTableAbi::vtable`] points to when
/// `category == BACKEND_CAP_CATEGORY_CACHE`. `make`/`drop` live here (make's args are the
/// per-category [`QuantCacheMakeArgs`], so they cannot go in the common header).
#[repr(C)]
pub struct QuantCacheVTable {
    /// [`QuantCacheMakeArgs`] → opaque plugin handle (one-time kernel build, D4).
    pub make: unsafe extern "C" fn(*const QuantCacheMakeArgs) -> *mut c_void,
    /// handle → valid token count.
    pub current_pos: unsafe extern "C" fn(*mut c_void) -> usize,
    /// handle → physical capacity (tokens).
    pub capacity: unsafe extern "C" fn(*mut c_void) -> usize,
    /// handle → current bit-width.
    pub current_bits: unsafe extern "C" fn(*mut c_void) -> u8,
    /// handle + [`QuantCacheUpdateArgs`] → i32 (0=OK, negative=err). Per-token hot path.
    pub update: unsafe extern "C" fn(*mut c_void, *const QuantCacheUpdateArgs) -> i32,
    /// handle → i32. Flush the residual ring if full.
    pub flush_if_full: unsafe extern "C" fn(*mut c_void) -> i32,
    /// handle + [`QuantCacheViewOut`] out-param → i32. Assemble the dequantized view.
    pub assemble_view: unsafe extern "C" fn(*mut c_void, *mut QuantCacheViewOut) -> i32,
    /// handle + [`QuantCacheRawBuffersOut`] out-param → bool. Raw set for the native path.
    pub get_raw_buffers: unsafe extern "C" fn(*mut c_void, *mut QuantCacheRawBuffersOut) -> bool,
    /// handle + target bits → i32. Runtime bit-width transition.
    pub transition_bits: unsafe extern "C" fn(*mut c_void, u8) -> i32,
    /// Release the handle (called once by the host when the capability is dropped).
    pub drop: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: the vtable is immutable and fn-ptrs are inherently Send+Sync. Required to declare a distributed_slice element static.
unsafe impl Sync for QuantCacheVTable {}

/// dual-wiring macro (D8) that registers a quantized-KV cache capability plugin both statically
/// (rlib → linkme [`QUANT_CACHE_REGS`] name survival) and dynamically (cdylib → C-ABI
/// [`QuantCacheVTable`] + envelope entry tagged [`BACKEND_CAP_CATEGORY_CACHE`]). Exact shape of
/// [`register_quant_attn_plugin!`]. `$make` = `fn(&QuantCacheMakeArgs) -> Box<dyn QuantCacheBackend>`.
#[macro_export]
macro_rules! register_quant_cache_plugin {
    ($name:literal, $make:expr) => {
        // ── Static path (rlib → linkme QUANT_CACHE_REGS, name survives under force-link). Ungated. ──
        const _: () = {
            #[$crate::distributed_slice($crate::QUANT_CACHE_REGS)]
            static __CACHE_REG: $crate::QuantCacheReg = $crate::QuantCacheReg {
                name: $name,
                make: $make,
            };
        };

        // ── Dynamic path (cdylib → QuantCacheVTable + envelope entry). Gated by plugin-cdylib. ──
        #[cfg(feature = "plugin-cdylib")]
        const _: () = {
            // handle = Box<Box<dyn QuantCacheBackend>> (thin ptr). All thunks share this representation.
            type __Handle = ::std::boxed::Box<dyn $crate::QuantCacheBackend>;

            unsafe extern "C" fn __make(
                p: *const $crate::QuantCacheMakeArgs,
            ) -> *mut ::core::ffi::c_void {
                // SAFETY: the host passes a valid QuantCacheMakeArgs pointer (D4). It is a Copy POD.
                let args: &$crate::QuantCacheMakeArgs = unsafe { &*p };
                let make_fn: fn(&$crate::QuantCacheMakeArgs) -> __Handle = $make;
                let be: __Handle = make_fn(args);
                ::std::boxed::Box::into_raw(::std::boxed::Box::new(be)) as *mut ::core::ffi::c_void
            }

            unsafe extern "C" fn __current_pos(h: *mut ::core::ffi::c_void) -> usize {
                // SAFETY: h is the Box<Box<dyn>> created by __make.
                let be: &dyn $crate::QuantCacheBackend = unsafe { &**(h as *const __Handle) };
                be.current_pos()
            }

            unsafe extern "C" fn __capacity(h: *mut ::core::ffi::c_void) -> usize {
                // SAFETY: same as above.
                let be: &dyn $crate::QuantCacheBackend = unsafe { &**(h as *const __Handle) };
                be.capacity()
            }

            unsafe extern "C" fn __current_bits(h: *mut ::core::ffi::c_void) -> u8 {
                // SAFETY: same as above.
                let be: &dyn $crate::QuantCacheBackend = unsafe { &**(h as *const __Handle) };
                be.current_bits()
            }

            unsafe extern "C" fn __update(
                h: *mut ::core::ffi::c_void,
                a: *const $crate::QuantCacheUpdateArgs,
            ) -> i32 {
                // SAFETY: h is the Box<Box<dyn>> created by __make; a is a valid QuantCacheUpdateArgs (C5).
                let be: &dyn $crate::QuantCacheBackend = unsafe { &**(h as *const __Handle) };
                be.update(unsafe { &*a })
            }

            unsafe extern "C" fn __flush_if_full(h: *mut ::core::ffi::c_void) -> i32 {
                // SAFETY: same as above.
                let be: &dyn $crate::QuantCacheBackend = unsafe { &**(h as *const __Handle) };
                be.flush_if_full()
            }

            unsafe extern "C" fn __assemble_view(
                h: *mut ::core::ffi::c_void,
                o: *mut $crate::QuantCacheViewOut,
            ) -> i32 {
                // SAFETY: h is the Box<Box<dyn>> created by __make; o is a valid out-param the host owns.
                let be: &dyn $crate::QuantCacheBackend = unsafe { &**(h as *const __Handle) };
                be.assemble_view(unsafe { &mut *o })
            }

            unsafe extern "C" fn __get_raw_buffers(
                h: *mut ::core::ffi::c_void,
                o: *mut $crate::QuantCacheRawBuffersOut,
            ) -> bool {
                // SAFETY: same as above; o is a valid out-param the host owns.
                let be: &dyn $crate::QuantCacheBackend = unsafe { &**(h as *const __Handle) };
                be.get_raw_buffers(unsafe { &mut *o })
            }

            unsafe extern "C" fn __transition_bits(h: *mut ::core::ffi::c_void, bits: u8) -> i32 {
                // SAFETY: same as above.
                let be: &dyn $crate::QuantCacheBackend = unsafe { &**(h as *const __Handle) };
                be.transition_bits(bits)
            }

            unsafe extern "C" fn __drop(h: *mut ::core::ffi::c_void) {
                // SAFETY: h is the Box<Box<dyn>> created by __make; the host calls this exactly once.
                drop(unsafe { ::std::boxed::Box::from_raw(h as *mut __Handle) });
            }

            static __VTABLE: $crate::QuantCacheVTable = $crate::QuantCacheVTable {
                make: __make,
                current_pos: __current_pos,
                capacity: __capacity,
                current_bits: __current_bits,
                update: __update,
                flush_if_full: __flush_if_full,
                assemble_view: __assemble_view,
                get_raw_buffers: __get_raw_buffers,
                transition_bits: __transition_bits,
                drop: __drop,
            };

            // Contributes the envelope entry to PLUGIN_BACKEND_CAP_VTABLES (const-block isolation = accumulation). Not the entry point.
            #[$crate::distributed_slice($crate::PLUGIN_BACKEND_CAP_VTABLES)]
            static __ENTRY: $crate::BackendCapVTableAbi = $crate::BackendCapVTableAbi {
                name: ::core::concat!($name, "\0").as_ptr() as *const ::core::ffi::c_char,
                category: $crate::BACKEND_CAP_CATEGORY_CACHE,
                vtable: &__VTABLE as *const $crate::QuantCacheVTable as *const ::core::ffi::c_void,
            };
        };
    };
}

// ════════════════════════════════════════════════════════════════════════════
// KV read-plan surface — the 4th plan-returning plugin surface, deciding "what to read".
// A parallel mirror copy of KVMutationStage (eviction) / WeightStage (dispatch) / KVFormat.
// ════════════════════════════════════════════════════════════════════════════

/// The **granularity abstraction** for KV-cache reads.
///
/// `Token` = `select` is a subset of KV token positions (pos).
/// `Page { page_size }` = `select` is a subset of page indices, and each page groups `page_size` tokens.
///
/// NOTE: `Page { page_size }` is a variant with a field, so `#[repr(u32)]` cannot be applied directly. The C-ABI flattening
/// the future read-plan C-ABI is to be defined with separate `granularity: u32 + page_size: u32` fields.
/// For now it stays a Rust-native enum (no .so conversion needed at the implementation stage — phased rollout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadGranularity {
    /// Token-granularity (pos-level) selection.
    Token,
    /// Page-granularity selection. `page_size` tokens make up one page.
    Page { page_size: u32 },
}

/// The read plan produced by a read stage.
///
/// `granularity` sets the unit, and `select` holds an ascending subset (Token=pos, Page=page index).
/// **No `new_pos`** — a read plan does not mutate the cache (the decisive difference from an eviction plan, D2).
#[derive(Clone, Debug, PartialEq)]
pub struct KVReadPlan {
    /// The read granularity.
    pub granularity: ReadGranularity,
    /// List of token positions / page indices to read (ascending). When empty it means a full read (handled by the engine).
    pub select: Vec<usize>,
}

/// The plan-returning trait for a KV read stage — it decides "what to read".
///
/// `None` = full read for this layer (the current behavior, zero happy-path cost). `ctx` reuses the existing `StageCtx`
/// (no need to introduce a read-stage-specific ctx). The plugin incrementally updates page
/// metadata via `tensor(Key)`/`tensor(QueryStats)` (D5). A plan is an *approximate hint*, not a correctness
/// contract (fallback = full read = exact; approximate acceleration = opt-in).
///
/// **INV-HOTPATH-DISPATCH**: fired at the layer tier (just before attention, once per layer). The happy path (absent stage)
/// short-circuits with a single `Option::is_none()`.
pub trait KVReadStage: Send + Sync {
    /// Technique name (matched against CLI `--read-stage <name>`, used for logging). Must be unique within the slice.
    fn name(&self) -> &str;

    /// Called just before layer i's attention — produces the plan for "which KV to read at layer i".
    /// `None` = full read (preserving current behavior). The plugin incrementally updates page metadata via ctx.
    fn read_plan(&self, ctx: &dyn StageCtx) -> Option<KVReadPlan>;
}

/// The CLI-derived static configuration for a read stage (a mirror of the KV `StageParams`). The static knobs of the first built-in read stage.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReadStageParams {
    /// Page size (number of tokens grouped per page). Default 16.
    pub page_size: u32,
    /// Fraction of pages to select (1/`top_k_ratio_denom` of all pages). Default 4 (= 1/4 of the total).
    pub top_k_ratio_denom: u32,
}

impl Default for ReadStageParams {
    fn default() -> Self {
        Self {
            page_size: 16,
            top_k_ratio_denom: 4,
        }
    }
}

/// The registration entry for one read stage technique (a mirror of `MutationStageReg`).
pub struct KVReadStageReg {
    /// CLI `--read-stage` name. Must be unique within the slice.
    pub name: &'static str,
    /// Factory that builds a technique instance from the parameters.
    pub make: fn(ReadStageParams) -> Box<dyn KVReadStage>,
    /// Whether this stage's `read_plan` consumes [`TensorKind::QueryStats`] (per-(layer,kv_head)
    /// running Q mean/var). When `true`, the engine builds + activates a `QueryStatsAccumulator`
    /// and threads its per-layer stats into the read ctx; when `false` it skips that per-decode-step
    /// cost entirely. The engine reads this off the registration **before** instantiating the stage,
    /// so it never needs to name-match a specific read stage. The read-axis analogue of `StageCaps.reads`.
    pub wants_query_stats: bool,
}

/// Global read-stage registration slice — the **4th parallel linkme registry** on the stage axis.
///
/// **Starts with zero built-ins** — when no read stage is present the engine always does a full read (100% of current behavior preserved, D5).
/// The first built-in read stage is registered out-of-tree (S4/S5).
#[distributed_slice]
pub static KV_READ_STAGES: [KVReadStageReg] = [..];

/// Find a registered read stage by name (used during engine construction / CLI parsing).
pub fn find_read_stage(name: &str) -> Option<&'static KVReadStageReg> {
    KV_READ_STAGES.iter().find(|r| r.name == name)
}

/// Names of all registered read stages (for self-test / diagnostics).
pub fn registered_read_names() -> Vec<&'static str> {
    KV_READ_STAGES.iter().map(|r| r.name).collect()
}

// ════════════════════════════════════════════════════════════════════════════
// QCF estimator axis (observer/score axis, EPIC 2 Stage A)
// ════════════════════════════════════════════════════════════════════════════
//
// Per-technique degradation simulation. The engine's QCF harness used to duplicate every eviction
// plugin's arithmetic (identify-retained + redistribute / d2o merge) a second time, purely to compute
// the post-eviction attention output O_after. This axis lets each technique supply its own O_after, so
// the engine core holds no concrete technique arithmetic — only the generic metric and the lent
// read primitives.
//
// Static-linkme only (no cdylib C-ABI): the only consumers (the engine QCF runtime / eval / ppl
// paths) are all in-engine, so there is no out-of-tree `.so` consumer to serve. A future cdylib path
// stays open — every estimator output is POD (`o_after` writes `&mut [f32]`), so it would not break the
// surface.

/// Read-only per-eviction context the engine lends to a [`QcfEstimator`] so it can build its own
/// O_after **without touching the production cache**. The engine implements this over a host (D2H)
/// readback of the KV buffers — GPU caches are cache-incoherent through `as_ptr`, so a host copy is
/// required — plus the per-head redistribution weights it already computes for O_before. Every vector
/// is host `f32`.
pub trait EstimatorCtx {
    /// Tokens currently resident in the cache (the simulated set is `[0, current_pos)`).
    fn current_pos(&self) -> usize;
    /// Post-eviction token budget the estimate simulates (engine-derived, e.g. `current_pos / 2`).
    fn target_len(&self) -> usize;
    /// Number of KV heads.
    fn n_kv_heads(&self) -> usize;
    /// Per-head dimension — the length of every `read_v` / `read_k` / O_after buffer.
    fn head_dim(&self) -> usize;
    /// β exponent for redistributed-attention amplification (`1.0` = no amplification / legacy).
    fn beta(&self) -> f32;
    /// Per-head redistribution weights α_h[t] (`out.len() == current_pos`) — the same weights the
    /// engine uses for O_before (per-head attention with a flat-score fallback). Estimators rank and
    /// redistribute with these so O_after shares O_before's softmax space.
    fn alpha_h(&self, kv_head: usize, out: &mut [f32]);
    /// Read V[kv_head][pos] as host `f32` (`out.len() == head_dim`); out-of-range reads zero-fill.
    fn read_v(&self, kv_head: usize, pos: usize, out: &mut [f32]);
    /// Read K[kv_head][pos] as host `f32` (`out.len() == head_dim`). Returns `false` when K is
    /// unavailable, so the estimator can fall back to V (e.g. d2o's V-based nearest matching).
    fn read_k(&self, kv_head: usize, pos: usize, out: &mut [f32]) -> bool;
}

/// A per-technique degradation simulator: given the [`EstimatorCtx`], it builds the post-eviction
/// attention output O_after for one KV head. The engine's QCF harness owns the metric
/// (`‖O_before − O_after‖ / ‖O_before‖`, O_before, β, aggregation); the estimator owns only the
/// technique-specific O_after — the arithmetic the engine used to duplicate per technique.
pub trait QcfEstimator: Send + Sync {
    /// Technique name (== the actuator stage name, e.g. `"h2o"`).
    fn name(&self) -> &str;
    /// The estimate-map / `DegradationEstimator` curve key (e.g. `"kv.evict_h2o"`).
    fn curve_key(&self) -> &'static str;
    /// Build per-head O_after into `out` (`out.len() == ctx.head_dim()`). Returns `false` when the
    /// technique evicts nothing for the current budget (within-budget no-op); the caller then treats
    /// O_after == O_before (QCF contribution `0` for that head).
    fn o_after(&self, ctx: &dyn EstimatorCtx, kv_head: usize, out: &mut [f32]) -> bool;
}

/// Registration entry for one QCF estimator — a mirror of [`MutationStageReg`], static-linkme only.
/// `make` parses the same [`StageParams`] / [`StageArgs`] as the actuator stage so technique config
/// (keep_ratio, merge_e, ...) has a single source.
pub struct QcfEstimatorReg {
    /// Technique name (== the actuator stage name). Unique within the slice.
    pub name: &'static str,
    /// The estimate-map / curve key (e.g. `"kv.evict_h2o"`).
    pub curve_key: &'static str,
    /// Factory from the shared estimate parameters plus the technique-private blob.
    pub make: fn(StageParams, StageArgs<'_>) -> Box<dyn QcfEstimator>,
    /// Whether the estimate needs per-token attention scores (h2o / d2o). When `true` and the engine
    /// has no scores, the driver skips this estimator (matching the legacy `requires_scores` gate).
    pub requires_scores: bool,
    /// Whether the estimate needs an engine-supplied streaming `(sink, window)` config. When `true`
    /// and the engine has none, the driver skips this estimator — streaming cannot be dry-run blind.
    pub requires_streaming_config: bool,
}

/// Global QCF-estimator registration slice — the producer half of the observer/score axis. Each
/// eviction technique crate contributes one entry via `#[distributed_slice(QCF_ESTIMATORS)]`.
#[distributed_slice]
pub static QCF_ESTIMATORS: [QcfEstimatorReg] = [..];

/// Find a registered QCF estimator by name.
pub fn find_qcf_estimator(name: &str) -> Option<&'static QcfEstimatorReg> {
    QCF_ESTIMATORS.iter().find(|r| r.name == name)
}

/// Names of all registered QCF estimators (self-test / diagnostics).
pub fn registered_qcf_estimator_names() -> Vec<&'static str> {
    QCF_ESTIMATORS.iter().map(|r| r.name).collect()
}

/// Engine-lent primitive: `O_after = Σ_{t∈retained} (α_t^β / Σ_s α_s^β) · V[kv_head][t]`
/// (β = 1 → the legacy `α_t / Σα` form). Zero-fills `out` first; an all-zero α-sum leaves O_after at
/// zero. Eviction-only estimators (sliding / h2o / streaming) build their retained set then call this;
/// d2o redistributes its own merged V instead.
pub fn redistribute_value(
    ctx: &dyn EstimatorCtx,
    kv_head: usize,
    alpha: &[f32],
    retained: &[usize],
    beta: f32,
    out: &mut [f32],
) {
    for x in out.iter_mut() {
        *x = 0.0;
    }
    let head_dim = out.len();
    let mut v_t = vec![0.0f32; head_dim];
    if beta == 1.0 {
        let alpha_sum: f32 = retained.iter().map(|&t| alpha[t]).sum();
        if alpha_sum <= 0.0 {
            return;
        }
        for &t in retained {
            ctx.read_v(kv_head, t, &mut v_t);
            let w = alpha[t] / alpha_sum;
            for d in 0..head_dim {
                out[d] += w * v_t[d];
            }
        }
    } else {
        let alpha_pow_sum: f32 = retained.iter().map(|&t| alpha[t].max(0.0).powf(beta)).sum();
        if alpha_pow_sum <= 0.0 {
            return;
        }
        for &t in retained {
            ctx.read_v(kv_head, t, &mut v_t);
            let w = alpha[t].max(0.0).powf(beta) / alpha_pow_sum;
            for d in 0..head_dim {
                out[d] += w * v_t[d];
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Layer-scorer axis (observer/score axis, EPIC 2 Stage B)
// ════════════════════════════════════════════════════════════════════════════
//
// Per-layer importance scoring. The engine's ImportanceCollector used to hold every concrete
// importance formula inline (mean-pool cosine, per-token-cosine block-influence), duplicating the
// technique arithmetic in the engine core. This axis lets each formula live in a technique crate and
// self-register, so the engine keeps only the streaming harness, the OPR telemetry, and the generic
// per-layer ctx — no concrete scoring arithmetic.
//
// Static-linkme only (no cdylib C-ABI), exactly like the QcfEstimator axis above: every consumer (the
// QCF warmup workflow / weight-swap decider feed) is in-engine, so there is no out-of-tree `.so` to
// serve. A future cdylib path stays open — the scorer output is POD (`score` returns `f32` by value),
// so it would not break the surface.
//
// Two lifecycles share the axis. PerLayerStreaming scorers (mean-pool / per-token-cosine) run inside the
// prefill layer loop, one call per (layer, sublayer), over the current layer's activations.
// OneShotPostWarmup scorers (the weight-perturbation / DirectAttn weight-perturbation formulas, not yet migrated)
// run once after warmup over all layers, reading cached per-layer means and weight subtensors. The
// `LayerScorerCtx` carries accessors for both; an implementation populates only the half matching its
// phase (a streaming ctx returns None/0 for the post-warmup accessors, and vice versa).

/// When a [`LayerScorer`] is invoked — decides which [`LayerScorerCtx`] accessors it may rely on.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerScorerPhase {
    /// Called once per (layer, sublayer) inside the prefill layer loop, over the current layer's
    /// activations (pooled + raw in/out). Mean-pool and per-token-cosine scorers are of this kind.
    PerLayerStreaming,
    /// Called once after warmup over all layers, reading cached per-layer means and weight subtensors
    /// (weight-perturbation / DirectAttn). Reserved for a later stage; no built-in scorer uses it yet.
    OneShotPostWarmup,
}

/// Read-only per-layer context the engine lends to a [`LayerScorer`]. Every slice is host `f32`.
///
/// The accessors split by [`LayerScorerPhase`]:
/// - **PerLayerStreaming** reads the current layer's activations: `dim`/`seq_len`, the mean-pooled
///   `pooled_in`/`pooled_out` (`[dim]`), and the raw `raw_in`/`raw_out` (`[seq_len × dim]`).
/// - **OneShotPostWarmup** reads `n_layers`, the cached `x_mean(layer)`, the F16/quantized weight
///   `primary_subtensor`/`secondary_subtensor(layer, name)`, and `gqa()`.
///
/// A given implementation serves one phase; the other phase's accessors return `None`/`0`/empty.
pub trait LayerScorerCtx {
    /// Total decoder layers (OneShotPostWarmup scorers index `[0, n_layers)`). `0` for a streaming ctx.
    fn n_layers(&self) -> usize;
    /// Hidden dimension of the current layer's activations.
    fn dim(&self) -> usize;
    /// Sequence length of the current pass.
    fn seq_len(&self) -> usize;

    // ── PerLayerStreaming activations (current layer) ──
    /// Mean-pooled input `[dim]` before the current layer. Always present for a streaming ctx.
    fn pooled_in(&self) -> &[f32];
    /// Mean-pooled output `[dim]` after the current layer. Always present for a streaming ctx.
    fn pooled_out(&self) -> &[f32];
    /// Raw input `[seq_len × dim]` before the current layer; `None` when not cached (single mode).
    fn raw_in(&self) -> Option<&[f32]>;
    /// Raw output `[seq_len × dim]` after the current layer. Always present for a streaming ctx.
    fn raw_out(&self) -> &[f32];
    /// `(before_seq_len, before_dim)` describing `raw_in`'s shape (may differ from `seq_len`/`dim`).
    fn raw_in_dims(&self) -> (usize, usize);

    // ── OneShotPostWarmup inputs (weight-perturbation / DirectAttn) — None/0/empty for a streaming ctx ──
    /// Cached mean-pooled input `[dim]` for `layer` (post-warmup). `None` for a streaming ctx.
    fn x_mean(&self, layer: usize) -> Option<&[f32]>;
    /// F16 weight subtensor `name` for `layer` as `(data, rows, cols)`. `None` if unavailable.
    fn primary_subtensor(&self, layer: usize, name: &str) -> Option<(&[f32], usize, usize)>;
    /// Quantized weight subtensor `name` for `layer` (dequantized to f32). `None` if unavailable.
    fn secondary_subtensor(&self, layer: usize, name: &str) -> Option<(&[f32], usize, usize)>;
    /// `(n_q_heads, n_kv_heads, head_dim)` GQA geometry for DirectAttn. `(0, 0, 0)` if unknown.
    fn gqa(&self) -> (usize, usize, usize);
}

/// A per-layer importance scorer: given the [`LayerScorerCtx`], it produces one layer's importance as
/// `f32` (higher = more important = costlier to skip). The engine owns the streaming harness, the
/// per-layer ctx, OPR telemetry, and the `Σ importance` QCF metric; the scorer owns only the concrete
/// formula — the arithmetic the engine core used to hold inline.
pub trait LayerScorer: Send + Sync {
    /// Scorer name (== the `ImportanceFormula::as_str()` selector key, e.g. `"mean_pool"`).
    fn name(&self) -> &str;
    /// Which lifecycle this scorer runs in (decides which ctx accessors it may read).
    fn phase(&self) -> LayerScorerPhase;
    /// Named weight subtensors the scorer reads (OneShotPostWarmup only) so the engine can pre-resolve
    /// them to f32 before `score`. Empty for activation-only scorers (mean-pool / per-token-cosine).
    fn reads_subtensors(&self) -> &'static [&'static str];
    /// Score `layer` using `ctx`. PerLayerStreaming scorers ignore `layer` (the ctx already holds the
    /// current layer's activations); OneShotPostWarmup scorers index `ctx` by `layer`.
    fn score(&self, layer: usize, ctx: &dyn LayerScorerCtx) -> f32;
}

/// Registration entry for one layer scorer — a mirror of [`QcfEstimatorReg`], static-linkme only.
/// `make` takes the same [`StageParams`] / [`StageArgs`] as the other technique factories (the
/// built-in scorers ignore both).
pub struct LayerScorerReg {
    /// Scorer name (== the `ImportanceFormula::as_str()` selector key). Unique within the slice.
    pub name: &'static str,
    /// The lifecycle the scorer runs in.
    pub phase: LayerScorerPhase,
    /// Factory from the shared params plus the technique-private blob.
    pub make: fn(StageParams, StageArgs<'_>) -> Box<dyn LayerScorer>,
    /// Named weight subtensors the scorer reads (mirrors [`LayerScorer::reads_subtensors`]); the engine
    /// reads this off the registration before `make` to plan subtensor pre-resolution.
    pub reads_subtensors: &'static [&'static str],
}

/// Global layer-scorer registration slice — the producer half of the observer/score axis (Stage B).
/// Each technique crate contributes its scorers via `#[distributed_slice(LAYER_SCORERS)]`.
#[distributed_slice]
pub static LAYER_SCORERS: [LayerScorerReg] = [..];

/// Find a registered layer scorer by name.
pub fn find_layer_scorer(name: &str) -> Option<&'static LayerScorerReg> {
    LAYER_SCORERS.iter().find(|r| r.name == name)
}

/// Names of all registered layer scorers (self-test / diagnostics).
pub fn registered_layer_scorer_names() -> Vec<&'static str> {
    LAYER_SCORERS.iter().map(|r| r.name).collect()
}

// ════════════════════════════════════════════════════════════════════════════
// Score-producer axis (observer/score axis, EPIC 2 Stage C)
// ════════════════════════════════════════════════════════════════════════════
//
// Forward-time attention-score production. The engine's `AttentionScoreAccumulator` used to hold the
// per-layer score-accumulation policy inline (per-layer MAX, GQA group averaging, the value-aware
// last-layer overwrite, forgetting-factor decay, cross-step SUM, time-normalization). This axis hosts that policy
// in a plugin, so the engine core holds no concrete scoring arithmetic — only a delegating shell that
// preserves the typed `&mut AttentionScoreAccumulator` forward param and forwards each call.
//
// Static-linkme only (no cdylib C-ABI): the only consumer (the in-engine decode driver) is in-tree,
// and every input/output is host `f32` (`scores: &[f32]`, `&[f32]` accessors), so a future cdylib
// path stays open without breaking the surface. The GPU half of this axis is `ScoreReduceBackend`
// below (Stage E) — repr(C) POD args (so a cdylib path stays open) but the same static-linkme
// registration, since its sole consumer is the in-tree default GPU scoring path.

/// Construction geometry for a [`ScoreProducer`], mirroring the engine accumulator's `new` / `new_gqa`
/// signatures (the only POD a producer needs at build time). `n_kv_heads == 0` selects the flat
/// (non-GQA) mode — per-token importance only; `n_kv_heads > 0` additionally tracks per-KV-head
/// importance and the value-aware last-layer attention. `time_normalize` is configured after construction
/// via [`ScoreProducer::set_time_normalize`] (mirroring the engine's prior two-step setup).
#[derive(Clone, Copy, Debug)]
pub struct ScoreProducerParams {
    /// Maximum sequence length (cache capacity); the length of the flat importance buffers.
    pub max_seq_len: usize,
    /// Total query heads (kept for parity with the engine's `new`; the built-in producer derives
    /// per-step head counts from the `accumulate_layer*` arguments and does not read this).
    pub n_heads: usize,
    /// KV heads for GQA grouping. `0` = flat mode (no per-head / value-aware buffers).
    pub n_kv_heads: usize,
    /// Total decoder layers (for the `last_n_layers` tracked-layer window).
    pub total_layers: usize,
    /// Track only the last N layers (`0` or `>= total_layers` = track all).
    pub last_n_layers: usize,
    /// Exponential (forgetting-factor) decay factor per step (`0.0` = no decay; clamped to `[0, 1]`).
    pub decay: f32,
}

/// A forward-time attention-score producer: it owns the per-token (and, in GQA mode, per-KV-head)
/// importance accumulation across a decode step and across steps. The engine drives it with the
/// per-layer post-softmax scores via [`ScoreProducer::accumulate_layer`] /
/// [`ScoreProducer::accumulate_layer_gqa`] between [`ScoreProducer::begin_step`] and
/// [`ScoreProducer::end_step`]; eviction stages then read the accumulated importance. The engine's
/// `AttentionScoreAccumulator` is a thin delegating shell over this trait — the arithmetic the engine
/// core used to hold inline.
pub trait ScoreProducer: Send + Sync {
    /// Producer name (the registry key, e.g. `"attn_score"`).
    fn name(&self) -> &str;
    /// Which score tensors this producer makes available to consumers (the producer half of the
    /// [`StageCaps::reads`] handshake). The built-in attention-score producer yields
    /// [`TensorKind::Scores`].
    fn produces(&self) -> &'static [TensorKind];

    // ── lifecycle / config ──
    /// Activate or deactivate accumulation. When inactive, `begin_step` / `end_step` are no-ops.
    fn set_active(&mut self, active: bool);
    /// Whether accumulation is currently active.
    fn is_active(&self) -> bool;
    /// Whether `layer` is within the tracked-layer window (and accumulation is active).
    fn should_track_layer(&self, layer: usize) -> bool;
    /// Enable time-normalized importance (`importance[t] / step_count[t]`).
    fn set_time_normalize(&mut self, enable: bool);

    // ── per-step driver ──
    /// Begin a decode step: apply decay to cumulative importance and clear the step-local buffers.
    fn begin_step(&mut self);
    /// Accumulate one layer's flat per-token scores (non-GQA). `scores` is `[n_heads_q * stride]`;
    /// per-token scores are summed across heads, then combined across layers with MAX. `score_offset`
    /// is the cache position of `scores[t = 0]` (`0` global, `kv_start_pos` for sliding-window layers).
    fn accumulate_layer(
        &mut self,
        scores: &[f32],
        stride: usize,
        cache_seq_len: usize,
        n_heads_q: usize,
        score_offset: usize,
    );
    /// GQA-aware accumulation: additionally averages Q-head scores within each KV group (per-KV-head
    /// importance, MAX across layers) and overwrites the last tracked layer's per-KV-head attention
    /// (value-aware policy input).
    fn accumulate_layer_gqa(
        &mut self,
        scores: &[f32],
        stride: usize,
        cache_seq_len: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
        score_offset: usize,
    );
    /// End a decode step: flush the step-local per-layer-MAX importance into cumulative importance
    /// (SUM across steps), update step counts, and recompute time-normalized scores if enabled.
    fn end_step(&mut self);

    // ── GPU bridge ──
    /// Overwrite cumulative importance with GPU-accumulated scores (after a GPU score sync). `flat` is
    /// `[max_seq_len]`; `head` is `[n_kv_heads * max_seq_len]` (used only in GQA mode).
    fn import_gpu_scores(&mut self, flat: &[f32], head: &[f32]);
    /// Reset all accumulated state (e.g. after eviction).
    fn reset(&mut self);

    // ── accessors ──
    /// Per-token importance (time-normalized if enabled, else raw cumulative). `[max_seq_len]`.
    fn importance_scores(&self) -> &[f32];
    /// Raw cumulative per-token importance regardless of normalization. `[max_seq_len]`.
    fn raw_importance_scores(&self) -> &[f32];
    /// Per-KV-head cumulative importance `[n_kv_heads * max_seq_len]`, or `None` in flat mode.
    fn head_importance_scores(&self) -> Option<&[f32]>;
    /// Last tracked layer's per-KV-head attention from the most recent step (value-aware policy input)
    /// `[n_kv_heads * max_seq_len]`, or `None` in flat mode.
    fn last_step_head_attn(&self) -> Option<&[f32]>;
    /// Number of KV heads (`0` = flat mode).
    fn n_kv_heads(&self) -> usize;

    // ── per-(layer, KV-head) importance dump (IMP-1 diagnostics; opt-in) ──
    //
    // These default to no-ops / `None`, so the dump is purely additive: a producer that does
    // not implement them is unaffected, and production scoring is byte-identical unless
    // `enable_layer_head_dump` is explicitly called (`INV-147`).

    /// Set the layer index that the *next* `accumulate_layer*` call belongs to, so a producer
    /// capturing per-layer state can attribute scores to a layer (the per-step driver loses the
    /// layer index otherwise). Default no-op. Diagnostics-only — never affects scoring.
    fn set_current_layer(&mut self, _layer: usize) {}

    /// Enable capture of a non-collapsed per-`(layer, KV-head, token)` importance buffer for
    /// diagnostics (IMP-1). Default no-op; the built-in producer lazily allocates an
    /// `[n_layers * n_kv_heads * max_seq_len]` buffer (GQA mode only). Off by default so the
    /// production decode path keeps its MAX-collapsed, memory-light footprint.
    fn enable_layer_head_dump(&mut self) {}

    /// The non-collapsed per-`(layer, KV-head, token)` importance from the most recent step, if
    /// [`Self::enable_layer_head_dump`] was called and GQA mode is active. Layout is row-major
    /// `[n_layers * n_kv_heads * max_seq_len]`, indexed `(layer * n_kv_heads + kv_head) * max_seq_len + pos`.
    /// Default `None`.
    fn layer_head_importance(&self) -> Option<&[f32]> {
        None
    }

    // ── per-(layer, token) FLAT importance (faithful-H2O LayerWise, divergence `(b)`; opt-in) ──
    //
    // The eviction-decision twin of the per-head dump above: when enabled, the producer keeps each
    // layer's OWN head-summed cumulative attention with NO cross-layer MAX, so a per-layer eviction
    // path can rank each layer's heavy hitters independently (faithful `H2OKVCache_LayerWise`).
    // Default no-op / `None` → purely additive, and the collapsed `importance` stays byte-identical
    // unless `enable_per_layer_flat` is explicitly called (the `INV-147` precedent for the flat axis).

    /// Enable capture of a per-`(layer, token)` FLAT importance buffer (no cross-layer MAX). Default
    /// no-op; the built-in producer lazily allocates an `[n_layers * max_seq_len]` buffer. Works in
    /// both flat and GQA mode (the flat layer score exists in both accumulate paths).
    fn enable_per_layer_flat(&mut self) {}

    /// The per-`(layer, token)` FLAT cumulative importance, if [`Self::enable_per_layer_flat`] was
    /// called. Row-major `[n_layers * max_seq_len]`, indexed `layer * max_seq_len + pos`. Each layer's
    /// own accumulated attention (no cross-layer MAX) — the faithful LayerWise score. Default `None`.
    fn layer_flat_importance(&self) -> Option<&[f32]> {
        None
    }

    /// Overwrite the per-`(layer, token)` FLAT cumulative with GPU-computed values `[n_layers *
    /// max_seq_len]` (the GPU per-layer reduce accumulates on-device; this is its eviction-time
    /// readback, the per-layer twin of [`Self::import_gpu_scores`]). Default no-op.
    fn import_gpu_layer_flat(&mut self, _layer_flat: &[f32]) {}
}

/// Registration entry for one score producer — a mirror of [`LayerScorerReg`], static-linkme only.
pub struct ScoreProducerReg {
    /// Producer name (the registry key). Unique within the slice.
    pub name: &'static str,
    /// The score tensors the producer makes available (mirrors [`ScoreProducer::produces`]); read off
    /// the registration so the engine can plan the [`StageCaps::reads`] handshake before `make`.
    pub produces: &'static [TensorKind],
    /// Factory from the construction geometry.
    pub make: fn(ScoreProducerParams) -> Box<dyn ScoreProducer>,
}

/// Global score-producer registration slice — the forward-time producer half of the observer/score
/// axis (Stage C). The built-in attention-score producer registers via
/// `#[distributed_slice(SCORE_PRODUCERS)]`.
#[distributed_slice]
pub static SCORE_PRODUCERS: [ScoreProducerReg] = [..];

/// Find a registered score producer by name.
pub fn find_score_producer(name: &str) -> Option<&'static ScoreProducerReg> {
    SCORE_PRODUCERS.iter().find(|r| r.name == name)
}

/// Names of all registered score producers (self-test / diagnostics).
pub fn registered_score_producer_names() -> Vec<&'static str> {
    SCORE_PRODUCERS.iter().map(|r| r.name).collect()
}

// ════════════════════════════════════════════════════════════════════════════
// OBSERVER/SCORE axis — (b) GPU half: ScoreReduceBackend (EPIC 2 Stage E)
// ════════════════════════════════════════════════════════════════════════════
//
// The GPU twin of the forward-time score policy. The attention decode kernel writes raw,
// policy-neutral post-softmax weights into a `[n_layers, n_heads_q, score_stride]` score buffer
// (that WRITE stays welded into the engine's flash/legacy attention kernels — it cannot be
// intercepted mid-kernel). The per-token REDUCE that folds those weights into cumulative importance
// — per-layer MAX, GQA group averaging, exponential (forgetting-factor) decay — is the score-reduce POLICY, and
// this trait hosts it in the technique plugin so the engine core holds no GPU scoring policy (the CPU
// twin already moved to `ScoreProducer` in Stage C).
//
// repr(C) POD args (GPU handles as `*mut c_void`) keep a future cdylib path open with no ABI
// re-break, but registration is static-linkme only: the sole consumer (the in-engine OpenCL
// `GpuScoreAccumulator`) is in-tree and the default scoring path, so a cdylib loader (a SCORE
// backend-cap category) would be a speculative abstraction with no out-of-tree consumer — the same
// static-only rule the CPU score/estimator/scorer halves follow. The plugin compiles its own reduce
// kernel from the host's borrowed `cl_context` (the FORMAT Phase 2 Stage E precedent) and dispatches
// on the lent `cl_command_queue`; the engine retains buffer ownership and lends the three score
// buffers borrow-for-call.

/// GPU context handles lent to a [`ScoreReduceBackend`] factory at construction (repr(C) POD). All
/// pointers are borrowed for the `make` call only; the backend must build its kernel from a retained
/// copy of the context and must not retain `build_opts`.
#[repr(C)]
pub struct ScoreReduceMakeArgs {
    /// `cl_context` the engine's score buffers live in. Borrowed for the call.
    pub cl_context: *mut c_void,
    /// `cl_device_id` to build the reduce program for. Borrowed for the call.
    pub cl_device: *mut c_void,
    /// Host build options (the engine's exact `build_cl_opts(device)`), or null for empty. The
    /// backend MUST use these verbatim so the compiled kernel matches the engine's numerics.
    pub build_opts: *const c_char,
}

/// Per-decode-step reduce dispatch arguments (repr(C) POD), mirroring the engine's score buffers and
/// the reduce kernel's scalar signature. All `cl_mem` / `cl_command_queue` handles are engine-owned
/// and borrowed for the call only (the backend must NOT release them).
#[repr(C)]
pub struct ScoreReduceArgs {
    /// `cl_command_queue` to enqueue the reduce on (the engine's inference queue). Borrowed.
    pub cl_queue: *mut c_void,
    /// `cl_mem` of the per-token score buffer `[n_layers, n_heads_q, score_stride]` the attention
    /// kernel wrote this step. Read-only input. Borrowed.
    pub score_buf: *mut c_void,
    /// `cl_mem` of the cumulative flat importance `[max_seq_len]`, updated in place. Borrowed.
    pub importance: *mut c_void,
    /// `cl_mem` of the cumulative per-KV-head importance `[n_kv_heads * max_seq_len]`, updated in
    /// place. Borrowed.
    pub head_importance: *mut c_void,
    /// Cumulative decay applied before adding this step's contribution (`1.0 - decay`, pre-clamped to
    /// `[0, 1]` by the engine — the policy's decay knob, opaque to the kernel dispatch).
    pub decay_factor: f32,
    /// Decoder layer count (the partition count of `score_buf`).
    pub n_layers: usize,
    /// Query heads per layer.
    pub n_heads_q: usize,
    /// KV heads (GQA groups); `<= 16` for the fused kernel's stack array.
    pub n_kv_heads: usize,
    /// Valid cache length this step (work-items `0..cache_seq_len`).
    pub cache_seq_len: usize,
    /// Row stride of `score_buf` (`== max_seq_len`).
    pub score_stride: usize,
    /// Capacity stride of the cumulative buffers (`== max_seq_len`).
    pub max_seq_len: usize,
    /// `cl_mem` of the per-`(layer, token)` FLAT cumulative buffer `[n_layers * max_seq_len]`
    /// (faithful-H2O `H2OKVCache_LayerWise`, divergence `(b)`), updated in place by
    /// [`reduce_per_layer`](ScoreReduceBackend::reduce_per_layer). `null` on the collapsed-only path
    /// (the `reduce` method never reads it). Appended last so the prior field layout is unchanged.
    pub layer_flat_importance: *mut c_void,
}

/// A GPU attention-score reducer: owns a compiled reduce kernel and folds one decode step's per-layer
/// post-softmax weights (in `score_buf`) into the cumulative `importance` / `head_importance` buffers
/// entirely on the device. The engine drives one [`reduce`](ScoreReduceBackend::reduce) per decode
/// token at end-of-step and reads the cumulative buffers back only at eviction time. This is the GPU
/// half of the observer/score axis — the score POLICY (MAX / GQA / decay) the engine core no longer
/// holds.
pub trait ScoreReduceBackend: Send + Sync {
    /// Reducer name (the registry key, e.g. `"attn_score"` — matched to the [`ScoreProducer`]).
    fn name(&self) -> &str;
    /// Dispatch the per-token fused reduce on `args.cl_queue`. Returns `0` on success, a negative code
    /// on failure (the engine logs and continues — cumulative importance is unchanged for the step).
    /// Must not panic across the call (panic = abort discipline).
    fn reduce(&self, args: &ScoreReduceArgs) -> i32;

    /// Dispatch the per-`(layer, token)` FLAT reduce (faithful-H2O `H2OKVCache_LayerWise`, divergence
    /// `(b)`) into `args.layer_flat_importance` — each layer's head-sum accumulated into its own slot
    /// with NO cross-layer MAX. Same return contract as [`reduce`](Self::reduce). Default `-1`
    /// (unsupported): a reducer without a per-layer kernel leaves the GPU per-layer buffer untouched,
    /// and the engine logs + falls back. Additive — existing reducers compile unchanged.
    fn reduce_per_layer(&self, _args: &ScoreReduceArgs) -> i32 {
        -1
    }
}

/// Registration entry for one GPU score reducer — a mirror of [`ScoreProducerReg`], static-linkme
/// only. `make` compiles the reduce kernel from the lent context and returns the reducer, or an error
/// string on compile/setup failure (the engine then falls back to the per-token CPU readback).
pub struct ScoreReduceReg {
    /// Reducer name (the registry key). Unique within the slice; matched to the producer name.
    pub name: &'static str,
    /// Factory from the lent GPU context. Borrows `args` for the call only.
    pub make: fn(&ScoreReduceMakeArgs) -> Result<Box<dyn ScoreReduceBackend>, String>,
}

/// Global GPU score-reducer registration slice — the GPU producer half of the observer/score axis
/// (Stage E). The built-in attention-score reducer registers via `#[distributed_slice(SCORE_REDUCERS)]`
/// under the `opencl` feature.
#[distributed_slice]
pub static SCORE_REDUCERS: [ScoreReduceReg] = [..];

/// Find a registered GPU score reducer by name.
pub fn find_score_reducer(name: &str) -> Option<&'static ScoreReduceReg> {
    SCORE_REDUCERS.iter().find(|r| r.name == name)
}

/// Names of all registered GPU score reducers (self-test / diagnostics).
pub fn registered_score_reducer_names() -> Vec<&'static str> {
    SCORE_REDUCERS.iter().map(|r| r.name).collect()
}

// ── OBSERVER/SCORE axis — (b) GPU half: the CUDA twin of ScoreReduceBackend ─────────────────────
//
// The OpenCL score-reduce seam above carries `cl_context`/`cl_command_queue`/`cl_mem` as `*mut c_void`.
// CUDA device pointers are `u64` (CUdeviceptr), not pointers, so a parallel POD set keeps the typing
// honest and lets a plugin register BOTH an OpenCL and a CUDA reducer under the same name in DISJOINT
// slices (no collision). Registration is static-linkme only (no live cdylib), so these `repr(C)` structs
// are a future-proofing gesture, not a frozen C-ABI. All scalar fields mirror `ScoreReduceArgs`.

/// GPU context handles lent to a [`CudaScoreReduceBackend`] factory at construction (repr(C) POD).
/// The CUDA analog of [`ScoreReduceMakeArgs`]; `build_opts` is replaced by the compute capability
/// (the plugin derives nvcc `-arch=sm_{major}{minor}` from it).
#[repr(C)]
pub struct CudaScoreReduceMakeArgs {
    /// Raw `CUcontext` (from `CudaContext::cu_ctx()`) the engine's score buffers live in. Borrowed.
    pub cu_context: *mut c_void,
    /// `CUdevice` ordinal (also the ordinal for `CudaContext::from_raw_context`). Borrowed.
    pub cu_device: i32,
    /// Compute capability major (for `nvcc -arch=sm_{major}{minor}`).
    pub cc_major: i32,
    /// Compute capability minor.
    pub cc_minor: i32,
}

/// Per-dispatch args for a CUDA score reduce (repr(C) POD). The CUDA analog of [`ScoreReduceArgs`]:
/// `cl_queue` → `cu_stream` (raw `CUstream`), and every `cl_mem` → a raw `CUdeviceptr` (`u64`). All
/// buffers are engine-owned and lent for the call; the scalar fields are identical to the OpenCL args.
#[repr(C)]
pub struct CudaScoreReduceArgs {
    /// Raw `CUstream` (from `CudaStream::cu_stream()`) to launch on. Borrowed.
    pub cu_stream: *mut c_void,
    /// `CUdeviceptr` of `[n_layers, n_heads_q, score_stride]` post-softmax scores, read-only. Borrowed.
    pub score_buf: u64,
    /// `CUdeviceptr` of `[max_seq_len]` cumulative flat importance, updated in place. Borrowed.
    pub importance: u64,
    /// `CUdeviceptr` of `[n_kv_heads * max_seq_len]` cumulative per-head importance, in place. Borrowed.
    pub head_importance: u64,
    /// `1.0 - decay`, pre-clamped to `[0,1]` by the engine.
    pub decay_factor: f32,
    pub n_layers: usize,
    pub n_heads_q: usize,
    /// `<= 16` for the fused kernel's stack array.
    pub n_kv_heads: usize,
    /// Work-items `0..cache_seq_len`.
    pub cache_seq_len: usize,
    /// Row stride of `score_buf` (== `max_seq_len`).
    pub score_stride: usize,
    /// Capacity stride of the cumulative buffers (== `max_seq_len`).
    pub max_seq_len: usize,
    /// `CUdeviceptr` of `[n_layers * max_seq_len]` per-layer flat importance; `0` on the collapsed-only
    /// path (mirrors [`ScoreReduceArgs::layer_flat_importance`]).
    pub layer_flat_importance: u64,
}

/// A CUDA score reducer — the device-side arithmetic of the observer/score axis. The CUDA twin of
/// [`ScoreReduceBackend`]; the engine drives one [`reduce`](CudaScoreReduceBackend::reduce) per decode
/// step on the lent stream.
pub trait CudaScoreReduceBackend: Send + Sync {
    /// Registry key (matched to the score producer name, e.g. `"attn_score"`).
    fn name(&self) -> &str;
    /// Dispatch the per-token fused reduce on `args.cu_stream`. Returns `0` on success, a negative code
    /// on failure. Must not panic.
    fn reduce(&self, args: &CudaScoreReduceArgs) -> i32;
    /// Per-layer flat reduce (the `layer_flat_importance` path). Default `-1` (unsupported).
    fn reduce_per_layer(&self, _args: &CudaScoreReduceArgs) -> i32 {
        -1
    }
}

/// Registration entry for one CUDA score reducer — the CUDA mirror of [`ScoreReduceReg`], static-linkme
/// only.
pub struct CudaScoreReduceReg {
    /// Reducer name (registry key). Unique within the slice; matched to the producer name.
    pub name: &'static str,
    /// Factory from the lent CUDA context. Borrows `args` for the call only.
    pub make: fn(&CudaScoreReduceMakeArgs) -> Result<Box<dyn CudaScoreReduceBackend>, String>,
}

/// Global CUDA score-reducer registration slice — the CUDA twin of [`SCORE_REDUCERS`]. The built-in
/// attention-score reducer registers here via `#[distributed_slice(CUDA_SCORE_REDUCERS)]` under the
/// `cuda` feature, disjoint from (and coexisting with) the OpenCL `SCORE_REDUCERS` registration.
#[distributed_slice]
pub static CUDA_SCORE_REDUCERS: [CudaScoreReduceReg] = [..];

/// Find a registered CUDA score reducer by name.
pub fn find_cuda_score_reducer(name: &str) -> Option<&'static CudaScoreReduceReg> {
    CUDA_SCORE_REDUCERS.iter().find(|r| r.name == name)
}

// ─── CUDA QuantAttn (ATTENTION backend-cap) axis ────────────────────────────────────────────────
//
// The CUDA twin of the OpenCL [`QuantAttnBackend`] family (KIVI quantized-KV fused attention). Same
// handle-translation rule as the CUDA score axis: `cl_context`→`cu_context:*mut c_void`;
// `cl_device_id`+`build_opts`→`cu_device:i32`+`cc_major/cc_minor:i32` (nvcc `-arch=sm_XY`);
// `cl_command_queue`→`cu_stream:*mut c_void`; every `cl_mem`(`*mut c_void`)→`u64` CUdeviceptr;
// `scores_out` stays `*mut f32` (host readback pointer, NOT a device buffer). Registered in a DISJOINT
// static-linkme slice (`CUDA_QUANT_ATTN_REGS`) — no cdylib `QuantAttnVTable` twin, no new
// `BACKEND_CAP_CATEGORY_*` (the CUDA axis is static-linkme only, like the score axis). All scalar
// fields mirror the OpenCL args 1:1; `#[repr(C)]` is future-proofing, not a frozen C-ABI.

/// GPU context handles lent to a [`CudaQuantAttnBackend`] factory at construction (repr(C) POD).
/// CUDA analog of [`QuantAttnMakeArgs`]; `build_opts` is replaced by the compute capability.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaQuantAttnMakeArgs {
    /// Raw `CUcontext` (from `CudaContext::cu_ctx()`) the engine's KV buffers live in. Borrowed.
    pub cu_context: *mut c_void,
    /// `CUdevice` ordinal (also the ordinal for `CudaContext::from_raw_context`). Borrowed.
    pub cu_device: i32,
    /// Compute capability major (for `nvcc -arch=sm_{major}{minor}`).
    pub cc_major: i32,
    /// Compute capability minor.
    pub cc_minor: i32,
}

/// Per-call args for CUDA fused dequant+attention (repr(C) POD). CUDA analog of [`QuantAttnArgs`]:
/// `cl_queue`→`cu_stream`, six `cl_mem`→`u64` CUdeviceptr, `scores_out` stays `*mut f32` (host).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaQuantAttnArgs {
    /// Raw `CUstream` to launch on (may be null == default stream). Borrowed.
    pub cu_stream: *mut c_void,
    /// `CUdeviceptr` of Q `[num_heads_q, head_dim]` F32. Borrowed.
    pub q_mem: u64,
    /// `CUdeviceptr` of quantized key blocks (per-channel, flush-interleaved). Borrowed.
    pub qk_mem: u64,
    /// `CUdeviceptr` of quantized value blocks (per-token, flush-interleaved). Borrowed.
    pub qv_mem: u64,
    /// `CUdeviceptr` of F32 residual keys `[kv_heads, res_cap, head_dim]`. Borrowed.
    pub res_k_mem: u64,
    /// `CUdeviceptr` of F32 residual values `[kv_heads, res_cap, head_dim]`. Borrowed.
    pub res_v_mem: u64,
    /// `CUdeviceptr` of output `[num_heads_q, head_dim]` F32, written. Borrowed.
    pub out_mem: u64,
    /// Host pointer for post-softmax score readback (may be null == skip). NOT a device buffer.
    pub scores_out: *mut f32,
    pub scores_len: usize,
    pub num_heads_q: usize,
    pub num_heads_kv: usize,
    pub head_dim: usize,
    pub q_tokens: usize,
    pub res_tokens: usize,
    pub res_cap: usize,
    pub scale: f32,
    pub bits: u8,
}

/// Per-call args for the CUDA residual gather-update (repr(C) POD). CUDA analog of [`QuantAttnGatherArgs`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaQuantAttnGatherArgs {
    /// Raw `CUstream` (may be null == default stream). Borrowed.
    pub cu_stream: *mut c_void,
    /// `CUdeviceptr` of input `[seq_len, kv_heads, head_dim]` F32. Borrowed.
    pub input_mem: u64,
    /// `CUdeviceptr` of residual `[kv_heads, res_cap, head_dim]` F32, written. Borrowed.
    pub residual_mem: u64,
    pub kv_heads: usize,
    pub res_cap: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub res_pos: usize,
}

/// Per-call args for the CUDA Q2 dequant-flush (repr(C) POD). CUDA analog of [`QuantDequantFlushArgs`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaQuantDequantFlushArgs {
    /// Raw `CUstream` (may be null == default stream). Borrowed.
    pub cu_stream: *mut c_void,
    /// `CUdeviceptr` of source quantized blocks. Borrowed.
    pub q_blocks_mem: u64,
    /// `CUdeviceptr` of destination F16 attention buffer. Borrowed.
    pub attn_mem: u64,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// K: groups_per_flush; V: flush_tokens.
    pub n_groups_or_tokens: usize,
    pub tok_base: usize,
    pub block_start: usize,
    pub bits: u8,
    pub is_key: bool,
}

/// Per-call args for the CUDA residual-scatter into the F16 view (repr(C) POD). CUDA analog of
/// [`QuantScatterResidualArgs`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaQuantScatterResidualArgs {
    /// Raw `CUstream` (may be null == default stream). Borrowed.
    pub cu_stream: *mut c_void,
    /// `CUdeviceptr` of F32 residual ring source. Borrowed.
    pub res_mem: u64,
    /// `CUdeviceptr` of F16 attention view destination. Borrowed.
    pub attn_mem: u64,
    pub kv_heads: usize,
    pub res_cap: usize,
    pub head_dim: usize,
    pub res_pos: usize,
    pub tok_base: usize,
}

/// A CUDA quantized-KV attention backend — the CUDA twin of [`QuantAttnBackend`]. Like the OpenCL
/// trait it has NO `name` method (the name lives on [`CudaQuantAttnReg`]). Returns `0` on success, a
/// negative code on failure; must not panic (C3 panic=abort across the C boundary in the OpenCL twin —
/// kept here for symmetry even though the CUDA axis is static-linkme).
pub trait CudaQuantAttnBackend: Send + Sync {
    /// Whether a native fused attention kernel exists for `bits` (2/4/8).
    fn has_quant_attn_kernel(&self, bits: u8) -> bool;
    /// Advisory nosub flag (mirrors the OpenCL twin; the engine reads the real flag off the backend).
    fn is_nosub_device(&self) -> bool;
    /// Fused dequant+attention on `args.cu_stream`.
    fn attention_gen_quant(&self, args: &CudaQuantAttnArgs) -> i32;
    /// Residual gather-update.
    fn gather_update_quant(&self, args: &CudaQuantAttnGatherArgs) -> i32;
    /// Q2 dequant-flush into the F16 view. Default `-1` (unsupported).
    fn dequant_flush(&self, _args: &CudaQuantDequantFlushArgs) -> i32 {
        -1
    }
    /// Residual scatter into the F16 view. Default `-1` (unsupported).
    fn scatter_residual(&self, _args: &CudaQuantScatterResidualArgs) -> i32 {
        -1
    }
}

/// Registration entry for one CUDA quant-attn backend — the CUDA mirror of [`QuantAttnReg`],
/// static-linkme only. `make` returns `Result` (nvcc/PTX compile can fail at runtime), unlike the
/// OpenCL `QuantAttnReg` whose `make` is infallible.
pub struct CudaQuantAttnReg {
    /// Registry key (e.g. `"kivi_abi"`, matched to the OpenCL `--backend-cap` selector).
    pub name: &'static str,
    /// Factory from the lent CUDA context. Borrows `args` for the call only.
    pub make: fn(&CudaQuantAttnMakeArgs) -> Result<Box<dyn CudaQuantAttnBackend>, String>,
}

/// Global CUDA quant-attn registration slice — the CUDA twin of [`QUANT_ATTN_REGS`]. Disjoint from
/// the OpenCL slice and the dynamic `PLUGIN_BACKEND_CAP_VTABLES` path; the KIVI plugin registers here
/// under the `cuda` feature.
#[distributed_slice]
pub static CUDA_QUANT_ATTN_REGS: [CudaQuantAttnReg] = [..];

/// Find a registered CUDA quant-attn backend by name.
pub fn find_cuda_quant_attn(name: &str) -> Option<&'static CudaQuantAttnReg> {
    CUDA_QUANT_ATTN_REGS.iter().find(|r| r.name == name)
}

/// Names of all statically-registered CUDA quant-attn backends (fat-LTO name-survival diagnostics).
pub fn registered_cuda_quant_attn_names() -> Vec<&'static str> {
    CUDA_QUANT_ATTN_REGS.iter().map(|r| r.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (T1) `compile_keep_top_k` produces the 3-partition keep-list the built-in eviction plugins
    /// hand-roll. Pins: prefix-inclusive, heavy hitters by STABLE desc score re-sorted ascending,
    /// trailing recent window; the whole list is ascending. score-free (`heavy = 0`) degenerates to
    /// prefix + recent.
    #[test]
    fn compile_keep_top_k_three_partition_shape() {
        // current=20, prefix=4, recent=4 (recent_start = 16), heavy=4 over [4..16).
        let mut scores = [0.0f32; 20];
        scores[6] = 10.0;
        scores[9] = 9.0;
        scores[12] = 8.0;
        scores[14] = 7.0;
        let keep = compile_keep_top_k(
            KeepTopK {
                current: 20,
                prefix: 4,
                recent: 4,
                heavy: 4,
            },
            |pos| scores[pos],
        );
        assert_eq!(keep, vec![0, 1, 2, 3, 6, 9, 12, 14, 16, 17, 18, 19]);

        // score-free: heavy=0 => prefix + recent only.
        let keep = compile_keep_top_k(
            KeepTopK {
                current: 20,
                prefix: 4,
                recent: 6,
                heavy: 0,
            },
            |_| 0.0,
        );
        assert_eq!(keep, vec![0, 1, 2, 3, 14, 15, 16, 17, 18, 19]);
    }

    /// The STABLE descending sort is load-bearing for the byte-identical-to-old-plugins refactor: on a
    /// score tie at the heavy-budget cut, the lower input/position index must win. positions 3 and 4
    /// tie (heavy=1) → keep 3, not 4. A switch to sort_unstable_by would flip this and is caught here.
    #[test]
    fn compile_keep_top_k_stable_tie_breaks_by_position() {
        let keep = compile_keep_top_k(
            KeepTopK {
                current: 8,
                prefix: 2,
                recent: 2,
                heavy: 1,
            },
            |p| if p == 3 || p == 4 { 5.0 } else { 0.0 },
        );
        assert_eq!(keep, vec![0, 1, 3, 6, 7]);
    }

    /// Pass2-ABI1: when current < prefix (cache smaller than the protected prefix), the compiler clamps
    /// the prefix to current and yields the whole resident range — ascending + in-range — NOT a list
    /// with indices >= current that the T-10 keep validator would reject as InvalidKeep. Mutation-proof:
    /// without the `spec.prefix.min(spec.current)` clamp, current=2/prefix=4 emits [0,1,2,3] (2,3 out of
    /// range).
    #[test]
    fn compile_keep_top_k_clamps_prefix_to_current() {
        // current=2, prefix=4 (> current): keep everything in range, no index >= current.
        let keep = compile_keep_top_k(
            KeepTopK {
                current: 2,
                prefix: 4,
                recent: 2,
                heavy: 0,
            },
            |_| 0.0,
        );
        assert_eq!(keep, vec![0, 1]);
        assert!(keep.iter().all(|&p| p < 2), "no out-of-range index");

        // current=0, prefix=4: empty resident -> empty keep (no panic, no out-of-range).
        let keep = compile_keep_top_k(
            KeepTopK {
                current: 0,
                prefix: 4,
                recent: 2,
                heavy: 0,
            },
            |_| 0.0,
        );
        assert!(keep.is_empty());
    }

    /// keep_intersect / keep_union compose keep-sets into ascending, deduped results.
    #[test]
    fn keep_combinators_intersect_and_union() {
        let a = [0usize, 1, 3, 5, 7];
        let b = [1usize, 2, 3, 5, 9];
        let c = [3usize, 5, 7, 9];
        assert_eq!(keep_intersect(&[&a, &b, &c]), vec![3, 5]);
        assert_eq!(keep_union(&[&a, &b, &c]), vec![0, 1, 2, 3, 5, 7, 9]);
        assert_eq!(keep_intersect(&[]), Vec::<usize>::new());
        assert_eq!(keep_union(&[]), Vec::<usize>::new());
    }

    /// (D3 proof) `KVFormatPlan` values are constructible and `format_of` resolves last-wins over
    /// layer/token/head/side, with the Gate-0 identity (empty overrides => base everywhere). This is
    /// the interface-expressibility evidence for per-layer/head/token/importance quantization; the
    /// engine-side execution/rejection is proven separately in `kv::format_apply`.
    #[test]
    fn kv_format_plan_format_of_resolves_last_wins_and_gate0() {
        // Gate-0: empty overrides => base everywhere (the byte-identical no-op anchor).
        let base_only = KVFormatPlan {
            base: FormatId("q4_0".into()),
            overrides: vec![],
        };
        assert_eq!(base_only.format_of(0, 5, MergeAxis::Both).0, "q4_0");
        assert_eq!(base_only.format_of(3, 99, MergeAxis::KeyOnly).0, "q4_0");

        // Per-token two-tier (e.g. importance-top tokens kept f16, rest q2).
        let two_tier = KVFormatPlan {
            base: FormatId("q2".into()),
            overrides: vec![FormatOverride {
                region: KeepSpec::LayerWide(vec![5, 6]),
                format: FormatId("f16".into()),
                side: MergeAxis::Both,
            }],
        };
        assert_eq!(two_tier.format_of(0, 5, MergeAxis::ValueOnly).0, "f16");
        assert_eq!(two_tier.format_of(0, 7, MergeAxis::ValueOnly).0, "q2");

        // Side-asymmetric (KIVI-style: key f16 on token 5, value stays base).
        let sided = KVFormatPlan {
            base: FormatId("q2".into()),
            overrides: vec![FormatOverride {
                region: KeepSpec::LayerWide(vec![5]),
                format: FormatId("f16".into()),
                side: MergeAxis::KeyOnly,
            }],
        };
        assert_eq!(sided.format_of(0, 5, MergeAxis::KeyOnly).0, "f16");
        assert_eq!(sided.format_of(0, 5, MergeAxis::ValueOnly).0, "q2");

        // Per-head (head 1 gets f16 at token 2, head 0 does not) — last-wins over an earlier override.
        let per_head = KVFormatPlan {
            base: FormatId("q4_0".into()),
            overrides: vec![
                FormatOverride {
                    region: KeepSpec::LayerWide(vec![2]),
                    format: FormatId("q8_0".into()),
                    side: MergeAxis::Both,
                },
                FormatOverride {
                    region: KeepSpec::PerHead(vec![vec![], vec![2]]),
                    format: FormatId("f16".into()),
                    side: MergeAxis::Both,
                },
            ],
        };
        assert_eq!(per_head.format_of(1, 2, MergeAxis::Both).0, "f16"); // later override wins
        assert_eq!(per_head.format_of(0, 2, MergeAxis::Both).0, "q8_0"); // head 0 falls to earlier
    }

    #[test]
    fn tensor_kind_discriminants_are_additive() {
        assert_eq!(TensorKind::Key as u32, 0);
        assert_eq!(TensorKind::Value as u32, 1);
        assert_eq!(TensorKind::AttnWeights as u32, 2);
        assert_eq!(TensorKind::Scores as u32, 3);
        assert_eq!(TensorKind::QueryStats as u32, 4);
        assert_eq!(TensorKind::PrefillAttention as u32, 5);
        assert_eq!(TensorKind::Query as u32, 6); // additive — never renumbers 0–5
    }

    /// `KVLayoutDesc` byte accounting matches the engine block struct sizes.
    /// Zero engine dependency (argus-extension-api is isolated), so verified against literals — the engine side
    /// cross-checks `size_of::<Block*>()` (dtype_layout.rs).
    #[test]
    fn kv_layout_desc_byte_accounting() {
        let q4_0 = KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        };
        assert_eq!(q4_0.block_bytes(), Some(18)); // == size_of::<BlockQ4_0>()
        assert_eq!(q4_0.bytes_for_elems(32), Some(18));
        assert_eq!(q4_0.bytes_for_elems(64), Some(36));
        assert_eq!(q4_0.bytes_for_elems(31), None); // partial block not allowed

        let q4_1 = KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16WithMin,
            packing: Packing::Nibble,
        };
        assert_eq!(q4_1.block_bytes(), Some(20)); // == size_of::<BlockQ4_1>()
        assert_eq!(q4_1.bytes_for_elems(32), Some(20));

        let q8_0 = KVLayoutDesc {
            block_elems: 32,
            bits: 8,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Byte,
        };
        assert_eq!(q8_0.block_bytes(), Some(34)); // == size_of::<BlockQ8_0>()
        assert_eq!(q8_0.bytes_for_elems(32), Some(34));

        // q2_0 (asymmetric 2-bit, Quad): scale(2) + min(2) + 32/4 quad bytes(8) = 12.
        let q2_0 = KVLayoutDesc {
            block_elems: 32,
            bits: 2,
            scale_layout: ScaleLayout::PerBlockF16WithMin,
            packing: Packing::Quad,
        };
        assert_eq!(q2_0.block_bytes(), Some(12)); // == size_of::<BlockQ2_0>()
        assert_eq!(q2_0.bytes_for_elems(32), Some(12));
        assert_eq!(q2_0.bytes_for_elems(64), Some(24));
        assert_eq!(q2_0.bytes_for_elems(31), None); // partial block not allowed

        // raw (Dense): no block concept, bits/8 per element.
        let f32 = KVLayoutDesc {
            block_elems: 1,
            bits: 32,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        };
        assert_eq!(f32.block_bytes(), None);
        assert_eq!(f32.bytes_for_elems(10), Some(40));
        let f16 = KVLayoutDesc {
            block_elems: 1,
            bits: 16,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        };
        assert_eq!(f16.bytes_for_elems(10), Some(20));
    }

    // ── v3 native registry (KV_MUTATION_STAGES) round-trip ──

    /// No-op imperative technique for verifying the mutation-stage registration/lookup round-trip.
    /// Phase is declared at registration (no trait `phase()`), so the stage itself carries none.
    struct DummyMut;
    impl KVMutationStage for DummyMut {
        fn name(&self) -> &str {
            "dummy_mut"
        }
        fn on_phase(
            &self,
            _ctx: &dyn StageCtx,
            _cache: &mut dyn CacheHandle,
        ) -> Result<(), CacheOpError> {
            Ok(())
        }
    }

    // Registered via the full (4-arg) macro form to exercise caps + phase carriage.
    register_kv_mutation_stage!(
        "dummy_mut",
        |_p, _args| Box::new(DummyMut),
        StageCaps::SCORE_FREE,
        MutationPhase::PrefillEnd
    );

    // Registered via the score-free (3-arg) macro form to compile-cover that arm + its args-ignoring
    // shim + closure→fn coercion (the common case every migrated drop-only plugin uses).
    register_kv_mutation_stage!(
        "dummy_mut_sf",
        |_p| Box::new(DummyMut),
        MutationPhase::KvMutate
    );

    #[test]
    fn dummy_mutation_registers_into_slice() {
        let reg = find_mutation_stage("dummy_mut")
            .expect("dummy_mut must be registered in the mutation slice");
        assert_eq!(reg.name, "dummy_mut");
        // Phase is read from the registration (the SSOT) — the trait has no phase() method.
        assert_eq!(reg.phase, MutationPhase::PrefillEnd);
        assert_eq!(reg.caps, StageCaps::SCORE_FREE);
        let stage = (reg.make)(StageParams::default(), &[]);
        assert_eq!(stage.name(), "dummy_mut");
    }

    #[test]
    fn dummy_mutation_score_free_3arg_form() {
        // The 3-arg form: SCORE_FREE caps defaulted, explicit phase carried, args-ignoring shim built.
        let reg = find_mutation_stage("dummy_mut_sf")
            .expect("dummy_mut_sf (3-arg form) must be registered");
        assert_eq!(reg.caps, StageCaps::SCORE_FREE);
        assert_eq!(reg.phase, MutationPhase::KvMutate);
        assert_eq!((reg.make)(StageParams::default(), &[]).name(), "dummy_mut");
    }

    #[test]
    fn mutation_registry_lookups() {
        assert!(registered_mutation_names().contains(&"dummy_mut"));
        assert_eq!(
            mutation_stage_caps("dummy_mut"),
            Some(StageCaps::SCORE_FREE)
        );
        assert!(mutation_stage_caps("not_a_stage").is_none());
        assert!(find_mutation_stage("not_a_stage").is_none());
    }

    // ── GATE-C v2 Format C-ABI round-trip (in-process verification without a `.so`) ──

    /// Test format — a q4_0-like descriptor. It exposes a handle-lifecycle
    /// (make/layout/drop) thunk isomorphic to what the macro emits, verifying lossless by-value passing of the [`KVLayoutDesc`] POD.
    struct RtFormat;
    impl KVFormat for RtFormat {
        fn name(&self) -> &str {
            "rt_format"
        }
        fn layout(&self) -> KVLayoutDesc {
            KVLayoutDesc {
                block_elems: 32,
                bits: 4,
                scale_layout: ScaleLayout::PerBlockF16,
                packing: Packing::Nibble,
            }
        }
    }

    type RtHandle = Box<dyn KVFormat>;
    unsafe extern "C" fn rt_make() -> *mut c_void {
        Box::into_raw(Box::new(Box::new(RtFormat) as RtHandle)) as *mut c_void
    }
    unsafe extern "C" fn rt_layout(h: *mut c_void) -> KVLayoutDesc {
        // SAFETY: h is the Box<Box<dyn KVFormat>> created by rt_make.
        let fmt: &dyn KVFormat = unsafe { &**(h as *const RtHandle) };
        fmt.layout()
    }
    unsafe extern "C" fn rt_drop(h: *mut c_void) {
        // SAFETY: h is the Box<Box<dyn>> created by rt_make; called exactly once.
        drop(unsafe { Box::from_raw(h as *mut RtHandle) });
    }

    #[test]
    fn format_vtable_layout_pod_round_trip() {
        // Manually construct a vtable isomorphic to the macro's cdylib path (plugin-cdylib) to verify the ABI path without the feature.
        let vtable = FormatVTableAbi {
            name: b"rt_format\0".as_ptr() as *const c_char,
            make: rt_make,
            layout: rt_layout,
            drop: rt_drop,
        };
        // make → layout → drop: KVLayoutDesc round-trips losslessly across the extern "C" boundary by value.
        let handle = unsafe { (vtable.make)() };
        assert!(!handle.is_null(), "make returns a non-null handle");
        let desc = unsafe { (vtable.layout)(handle) };
        assert_eq!(
            desc,
            RtFormat.layout(),
            "KVLayoutDesc passes losslessly by-value across extern \"C\""
        );
        // Recover name (null-terminated 'static).
        let name = unsafe { core::ffi::CStr::from_ptr(vtable.name) }
            .to_str()
            .unwrap();
        assert_eq!(name, "rt_format");
        unsafe { (vtable.drop)(handle) };
    }

    // ── envelope (ExportAbi) by-value sret round-trip + multi-call accumulation ──

    static FMT_EXPORT_VTS: [FormatVTableAbi; 2] = [
        FormatVTableAbi {
            name: b"rt_env_a\0".as_ptr() as *const c_char,
            make: rt_make,
            layout: rt_layout,
            drop: rt_drop,
        },
        FormatVTableAbi {
            name: b"rt_env_b\0".as_ptr() as *const c_char,
            make: rt_make,
            layout: rt_layout,
            drop: rt_drop,
        },
    ];

    // Isomorphic to export_plugin!'s register_kv_formats_v2 — returns the envelope by-value (sret >16B).
    extern "C" fn mk_format_export() -> FormatExportAbi {
        FormatExportAbi {
            abi_version: KV_FORMAT_ABI_VERSION,
            count: FMT_EXPORT_VTS.len(),
            vtables: FMT_EXPORT_VTS.as_ptr(),
        }
    }

    /// Verifies that FormatExportAbi round-trips losslessly via extern "C" by-value (sret) and that the loader, via `vtables.add(i)`,
    /// correctly accesses the .so static array elements (even after the envelope stack frame is discarded).
    #[test]
    fn format_export_abi_by_value_sret_round_trip() {
        let env = mk_format_export();
        assert_eq!(env.abi_version, KV_FORMAT_ABI_VERSION);
        assert_eq!(env.count, 2);
        for (i, expect) in ["rt_env_a", "rt_env_b"].iter().enumerate() {
            // SAFETY: vtables is the base of FMT_EXPORT_VTS (a 'static array), i < count.
            let vt = unsafe { &*env.vtables.add(i) };
            let name = unsafe { core::ffi::CStr::from_ptr(vt.name) }
                .to_str()
                .unwrap();
            assert_eq!(&name, expect);
        }
    }

    /// FORMAT Phase 2, Stage A — backend-cap ABI v2: the residual-flush surface exists, the
    /// arg structs have a stable `repr(C)` layout, and the trait defaults are `-1` (unsupported)
    /// so pre-v2 impls compile unchanged.
    #[test]
    fn backend_cap_abi_v2_flush_surface() {
        // Envelope ABI bumped to 2 (a v1 `.so` is rejected by the loader).
        assert_eq!(BACKEND_CAP_ABI_VERSION, 2);

        // repr(C) layout stability — guards against silent field drift across the `.so` boundary.
        assert_eq!(core::mem::size_of::<QuantDequantFlushArgs>(), 72);
        assert_eq!(core::mem::align_of::<QuantDequantFlushArgs>(), 8);
        assert_eq!(core::mem::size_of::<QuantScatterResidualArgs>(), 64);
        assert_eq!(core::mem::align_of::<QuantScatterResidualArgs>(), 8);

        // The two new methods default to -1 so impls predating v2 (e.g. the synthetic plugins)
        // keep compiling and report "unsupported" rather than dispatching a missing kernel.
        struct NoFlush;
        impl QuantAttnBackend for NoFlush {
            fn has_quant_attn_kernel(&self, _bits: u8) -> bool {
                false
            }
            fn is_nosub_device(&self) -> bool {
                false
            }
            fn attention_gen_quant(&self, _args: &QuantAttnArgs) -> i32 {
                0
            }
            fn gather_update_quant(&self, _args: &QuantAttnGatherArgs) -> i32 {
                0
            }
        }
        let be = NoFlush;
        let dq = QuantDequantFlushArgs {
            cl_queue: core::ptr::null_mut(),
            q_blocks_mem: core::ptr::null_mut(),
            attn_mem: core::ptr::null_mut(),
            kv_heads: 1,
            head_dim: 64,
            n_groups_or_tokens: 1,
            tok_base: 0,
            block_start: 0,
            bits: 2,
            is_key: true,
        };
        let sc = QuantScatterResidualArgs {
            cl_queue: core::ptr::null_mut(),
            res_mem: core::ptr::null_mut(),
            attn_mem: core::ptr::null_mut(),
            kv_heads: 1,
            res_cap: 128,
            head_dim: 64,
            res_pos: 0,
            tok_base: 0,
        };
        assert_eq!(be.dequant_flush(&dq), -1);
        assert_eq!(be.scatter_residual(&sc), -1);
    }

    /// FORMAT Phase 2, Stage C — CACHE backend-cap category: a fresh category tag, stable
    /// `repr(C)` layouts for the four POD arg/out structs, and a `QuantCacheBackend` trait
    /// round-trip (no GPU). The CACHE vtable rides the SAME `BACKEND_CAP_ABI_VERSION=2` envelope.
    #[test]
    fn backend_cap_cache_category_surface() {
        // A distinct category tag — the host's category bridge keys on this.
        assert_eq!(BACKEND_CAP_CATEGORY_CACHE, 2);
        assert_ne!(BACKEND_CAP_CATEGORY_CACHE, BACKEND_CAP_CATEGORY_ATTENTION);

        // repr(C) layout stability — guards against silent field drift across the `.so` boundary.
        assert_eq!(core::mem::size_of::<QuantCacheMakeArgs>(), 72);
        assert_eq!(core::mem::align_of::<QuantCacheMakeArgs>(), 8);
        assert_eq!(core::mem::size_of::<QuantCacheUpdateArgs>(), 32);
        assert_eq!(core::mem::align_of::<QuantCacheUpdateArgs>(), 8);
        assert_eq!(core::mem::size_of::<QuantCacheViewOut>(), 32);
        assert_eq!(core::mem::align_of::<QuantCacheViewOut>(), 8);
        assert_eq!(core::mem::size_of::<QuantCacheRawBuffersOut>(), 64);
        assert_eq!(core::mem::align_of::<QuantCacheRawBuffersOut>(), 8);

        // Closed layout vocab — stable discriminants across the boundary.
        assert_eq!(ViewLayoutTag::HeadMajor as u32, 0);
        assert_eq!(ViewLayoutTag::SeqMajor as u32, 1);

        // Trait round-trip with a synthetic in-memory cache (proves the surface is usable without a GPU).
        struct FakeCache {
            pos: usize,
        }
        impl QuantCacheBackend for FakeCache {
            fn current_pos(&self) -> usize {
                self.pos
            }
            fn capacity(&self) -> usize {
                256
            }
            fn current_bits(&self) -> u8 {
                2
            }
            fn update(&self, args: &QuantCacheUpdateArgs) -> i32 {
                args.seq_len as i32
            }
            fn flush_if_full(&self) -> i32 {
                0
            }
            fn assemble_view(&self, out: &mut QuantCacheViewOut) -> i32 {
                out.tokens = self.pos;
                out.layout = ViewLayoutTag::HeadMajor as u32;
                0
            }
            fn get_raw_buffers(&self, _out: &mut QuantCacheRawBuffersOut) -> bool {
                false
            }
            fn transition_bits(&self, _target_bits: u8) -> i32 {
                0
            }
        }
        let c = FakeCache { pos: 7 };
        assert_eq!(c.current_pos(), 7);
        assert_eq!(c.capacity(), 256);
        assert_eq!(c.current_bits(), 2);
        let upd = QuantCacheUpdateArgs {
            cl_queue: core::ptr::null_mut(),
            k_in_mem: core::ptr::null_mut(),
            v_in_mem: core::ptr::null_mut(),
            seq_len: 3,
        };
        assert_eq!(c.update(&upd), 3);
        assert_eq!(c.flush_if_full(), 0);
        let mut view = QuantCacheViewOut {
            k_mem: core::ptr::null_mut(),
            v_mem: core::ptr::null_mut(),
            tokens: 0,
            layout: ViewLayoutTag::SeqMajor as u32,
        };
        assert_eq!(c.assemble_view(&mut view), 0);
        assert_eq!(view.tokens, 7);
        assert_eq!(view.layout, ViewLayoutTag::HeadMajor as u32);
        let mut raw = QuantCacheRawBuffersOut {
            qk_mem: core::ptr::null_mut(),
            qv_mem: core::ptr::null_mut(),
            res_k_mem: core::ptr::null_mut(),
            res_v_mem: core::ptr::null_mut(),
            q_tokens: 0,
            res_tokens: 0,
            res_cap: 0,
            bits: 0,
        };
        assert!(!c.get_raw_buffers(&mut raw));
        assert_eq!(c.transition_bits(4), 0);

        // No built-in registers in QUANT_CACHE_REGS at Stage C (the engine quant-window is engine-typed).
        assert!(find_quant_cache("nonexistent-cache").is_none());
    }

    // Multiple invocations — register_quant_cache_plugin! twice in one crate (const-block isolation).
    crate::register_quant_cache_plugin!("rt_qc_a", |_a| Box::new(StaticFakeCache));
    crate::register_quant_cache_plugin!("rt_qc_b", |_a| Box::new(StaticFakeCache));

    /// No-op cache for the static-registration round-trip.
    struct StaticFakeCache;
    impl QuantCacheBackend for StaticFakeCache {
        fn current_pos(&self) -> usize {
            0
        }
        fn capacity(&self) -> usize {
            0
        }
        fn current_bits(&self) -> u8 {
            16
        }
        fn update(&self, _args: &QuantCacheUpdateArgs) -> i32 {
            0
        }
        fn flush_if_full(&self) -> i32 {
            0
        }
        fn assemble_view(&self, _out: &mut QuantCacheViewOut) -> i32 {
            0
        }
        fn get_raw_buffers(&self, _out: &mut QuantCacheRawBuffersOut) -> bool {
            false
        }
        fn transition_bits(&self, _target_bits: u8) -> i32 {
            0
        }
    }

    /// Static path: both `register_quant_cache_plugin!` invocations land in QUANT_CACHE_REGS.
    #[test]
    fn register_quant_cache_multicall_static() {
        assert!(find_quant_cache("rt_qc_a").is_some(), "rt_qc_a static reg");
        assert!(find_quant_cache("rt_qc_b").is_some(), "rt_qc_b static reg");
        let names = registered_quant_cache_names();
        assert!(names.contains(&"rt_qc_a") && names.contains(&"rt_qc_b"));
    }

    /// Dynamic path (plugin-cdylib): both invocations accumulate a CACHE-tagged envelope entry
    /// into the shared PLUGIN_BACKEND_CAP_VTABLES (the same slice ATTENTION uses, keyed by category).
    #[cfg(feature = "plugin-cdylib")]
    #[test]
    fn register_quant_cache_multicall_dynamic() {
        let cache_names: Vec<&str> = PLUGIN_BACKEND_CAP_VTABLES
            .iter()
            .filter(|vt| vt.category == BACKEND_CAP_CATEGORY_CACHE)
            .map(|vt| {
                unsafe { core::ffi::CStr::from_ptr(vt.name) }
                    .to_str()
                    .unwrap()
            })
            .collect();
        assert!(
            cache_names.contains(&"rt_qc_a"),
            "rt_qc_a CACHE envelope entry: {cache_names:?}"
        );
        assert!(
            cache_names.contains(&"rt_qc_b"),
            "rt_qc_b CACHE envelope entry: {cache_names:?}"
        );
    }

    // E0428 name clash). Now possible thanks to const-block isolation.
    crate::register_kv_format!("v1_mc_a", || Box::new(DummyFormat));
    crate::register_kv_format!("v1_mc_b", || Box::new(DummyFormat));

    /// Static path: both invocations register into KV_FORMATS (visible via find_kv_format).
    #[test]
    fn register_kv_format_multicall_static() {
        assert!(
            find_kv_format("v1_mc_a").is_some(),
            "v1_mc_a static registration"
        );
        assert!(
            find_kv_format("v1_mc_b").is_some(),
            "v1_mc_b static registration"
        );
    }

    /// Dynamic path (plugin-cdylib): both invocations accumulate into PLUGIN_KV_FORMAT_VTABLES.
    #[cfg(feature = "plugin-cdylib")]
    #[test]
    fn register_kv_format_multicall_dynamic() {
        let names: Vec<&str> = PLUGIN_KV_FORMAT_VTABLES
            .iter()
            .map(|vt| {
                unsafe { core::ffi::CStr::from_ptr(vt.name) }
                    .to_str()
                    .unwrap()
            })
            .collect();
        assert!(
            names.contains(&"v1_mc_a"),
            "v1_mc_a dynamic accumulation: {names:?}"
        );
        assert!(
            names.contains(&"v1_mc_b"),
            "v1_mc_b dynamic accumulation: {names:?}"
        );
    }

    // ── weight stage (MW-B) ──

    /// No-op technique for verifying the weight-axis register/lookup round-trip.
    struct DummyWeight;
    impl WeightStage for DummyWeight {
        fn name(&self) -> &str {
            "dummy_weight"
        }
        fn plan(&self, _ctx: &dyn WeightStageCtx) -> Option<WeightDispatchPlan> {
            None
        }
    }

    /// Minimal weight ctx stub that closes the `plan` path. Only `layer_metric` is implemented — importance/quant_noise are
    /// satisfied by the default sugar (all None → trivial).
    struct DummyWeightCtx;
    impl WeightStageCtx for DummyWeightCtx {
        fn n_layers(&self) -> usize {
            0
        }
        fn budget(&self) -> usize {
            0
        }
        fn pressure(&self) -> u8 {
            0
        }
        fn current_format(&self, _layer: usize) -> TensorDtype {
            TensorDtype::F32
        }
        fn layer_metric(&self, _kind: LayerMetricKind) -> Option<&[f32]> {
            None
        }
    }

    #[distributed_slice(WEIGHT_STAGES)]
    static DUMMY_WEIGHT_REG: WeightStageReg = WeightStageReg {
        name: "dummy_weight",
        make: |_params| Box::new(DummyWeight),
    };

    #[test]
    fn dummy_weight_registers_into_slice() {
        let reg = find_weight_stage("dummy_weight")
            .expect("dummy_weight must be registered in the slice");
        assert_eq!(reg.name, "dummy_weight");
        let stage = (reg.make)(WeightStageParams {
            allow_boundary_layers: false,
        });
        assert_eq!(stage.name(), "dummy_weight");
        assert!(stage.plan(&DummyWeightCtx).is_none());
    }

    #[test]
    fn registered_weight_names_contains_dummy() {
        assert!(registered_weight_names().contains(&"dummy_weight"));
    }

    /// `WeightStageCtx` default sugar (importance/quant_noise) operates on top of `layer_metric`.
    #[test]
    fn weight_ctx_sugar_delegates_to_layer_metric() {
        let ctx = DummyWeightCtx;
        assert!(ctx.importance().is_none());
        assert!(ctx.quant_noise().is_none());
    }

    // ── Format-axis registry ──

    /// No-op format for verifying the format register/lookup round-trip (raw f16 descriptor).
    struct DummyFormat;
    impl KVFormat for DummyFormat {
        fn name(&self) -> &str {
            "dummy_format"
        }
        fn layout(&self) -> KVLayoutDesc {
            KVLayoutDesc {
                block_elems: 1,
                bits: 16,
                scale_layout: ScaleLayout::None,
                packing: Packing::Dense,
            }
        }
    }

    #[distributed_slice(KV_FORMATS)]
    static DUMMY_FORMAT_REG: KVFormatReg = KVFormatReg {
        name: "dummy_format",
        make: || Box::new(DummyFormat),
    };

    #[test]
    fn dummy_format_registers_into_slice() {
        let reg =
            find_kv_format("dummy_format").expect("dummy_format must be registered in the slice");
        assert_eq!(reg.name, "dummy_format");
        let f = (reg.make)();
        assert_eq!(f.name(), "dummy_format");
        // M-F2: layer-tier descriptor read (repr(C) KVLayoutDesc).
        let d = f.layout();
        assert_eq!(d.block_elems, 1);
        assert_eq!(d.bits, 16);
        assert_eq!(d.scale_layout, ScaleLayout::None);
        assert_eq!(d.packing, Packing::Dense);
    }

    #[test]
    fn registered_kv_format_names_contains_dummy() {
        assert!(registered_kv_format_names().contains(&"dummy_format"));
    }

    // ── Backend-capability-axis registry ──

    /// No-op capability for verifying the capability register/lookup round-trip.
    struct DummyCap;
    impl BackendCapability for DummyCap {
        fn name(&self) -> &str {
            "dummy_cap"
        }
    }

    #[distributed_slice(BACKEND_CAPABILITIES)]
    static DUMMY_CAP_REG: BackendCapReg = BackendCapReg {
        name: "dummy_cap",
        make: || Box::new(DummyCap),
    };

    #[test]
    fn dummy_cap_registers_into_slice() {
        let reg = find_backend_capability("dummy_cap")
            .expect("dummy_cap must be registered in the slice");
        assert_eq!(reg.name, "dummy_cap");
        assert_eq!((reg.make)().name(), "dummy_cap");
    }

    #[test]
    fn registered_backend_capability_names_contains_dummy() {
        assert!(registered_backend_capability_names().contains(&"dummy_cap"));
    }

    // ── KV read stage registry ──

    /// Starts with zero built-ins — `registered_read_names()` is an empty list, `find_read_stage("nonexistent")` = None.
    #[test]
    fn read_stage_registry_starts_empty() {
        // Zero-built-ins gate.
        assert!(
            !registered_read_names().contains(&"nonexistent"),
            "nonexistent must not be registered"
        );
        assert!(
            find_read_stage("nonexistent").is_none(),
            "find_read_stage(\"nonexistent\") must be None"
        );
    }

    /// Checks the basic behavior of the `ReadGranularity` variants.
    #[test]
    fn read_granularity_variants() {
        let t = ReadGranularity::Token;
        let p = ReadGranularity::Page { page_size: 16 };
        assert_eq!(t, ReadGranularity::Token);
        assert_ne!(t, p);
        if let ReadGranularity::Page { page_size } = p {
            assert_eq!(page_size, 16);
        } else {
            panic!("not the Page variant");
        }
    }

    /// Constructs a `KVReadPlan` and checks its fields.
    #[test]
    fn kv_read_plan_fields() {
        let plan = KVReadPlan {
            granularity: ReadGranularity::Token,
            select: vec![0, 3, 7],
        };
        assert_eq!(plan.granularity, ReadGranularity::Token);
        assert_eq!(plan.select, vec![0usize, 3, 7]);
    }
}
