//! argus-extension-api — the additive surface where extension techniques (stage axis) register themselves with **zero engine-core modifications**.
//!
//! extension mechanism = statically linked technique crate + linkme auto-registration. Each technique, in its own crate
//! (`crates/techniques/<name>/`), depends only on this crate, implements [`KVCacheStage`], and
//! submits itself to the [`KV_CACHE_STAGES`] slice via `#[distributed_slice]`. At construction
//! time the engine reads that slice to pick a technique (removing closed match arms → OCP).
//!
//! stage-axis extension techniques (eviction/merge) are unified under a **single plan-returning trait [`KVCacheStage`]**
//! (a sibling of the engine-side storage-representation trait `KVCacheFormat`). A technique merely *reads* [`StageCtx`] and returns a [`KVCachePlan`]
//! (retained tokens + weighted merge plan); it never mutates buffers directly — mutation is the engine's exclusive job, executing the plan
//! via `compact` (D1). State (d2o EMA, etc.) is held by the plugin struct itself via `&self` + interior mutability
//! (D4); it is not threaded through the ctx.
//!
//! Dependency direction: `engine → argus-extension-api ← technique crate` (one-way, no cycles). Hence this crate
//! **does not reference** engine types (`KVCache`/`Backend`) — the cache state a technique needs to read is exposed through the read-only abstraction [`StageCtx`] that this crate
//! defines, with the engine implementing it over `&KVCache` (D5). In the static
//! stage this is a borrow; in a future `.so` C-ABI stage the same abstraction is swapped for C accessors / a flat snapshot — forward-compatible.

use core::ffi::{c_char, c_void};

/// Re-exports linkme's proc-macro so the `register_kv_stage!` macro can reference the `distributed_slice` attribute by path
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
    /// raw V. The v_i of CAOTE/VATP. Same (pos,head) coordinate system as Key.
    Value,
    /// per-(kv_head,pos) attention weight from the previous decode step (last layer). The a_i of CAOTE. cols=1, per_head.
    /// Source: `AttentionScoreAccumulator::last_step_head_attn` (CPU overwrite / GPU = head_importance proxy).
    /// **Note**: a last-layer, last-step approximation — not a windowed/per-layer exact value (`has_attn_weights` gate).
    AttnWeights,
    /// per-(kv_head,pos) accumulated head importance (h2o_plus). cols=1, per_head.
    /// flat per-token importance is exposed zero-copy directly via [`StageCtx::importance`] rather than this handle (a D1 exception).
    Scores,
    /// per-(layer,kv_head) Q (query) running statistics — the input to the closed-form future-attention
    /// estimate of Expected Attention (arXiv 2510.00636). `shape = {rows:2, cols:head_dim, per_head:true}`
    /// (MQ-1): `read_row(0, kv_head, out)` = that kv_head's Q running **mean[head_dim]**,
    /// `read_row(1, kv_head, out)` = running **var[head_dim]**. Reduced to kv_head coordinates by the element-wise
    /// mean of the Q-head statistics within a GQA group (MQ-2 — the same kv_head coordinates as the GQA reduction of `Scores`/`AttnWeights`, so they are cross-
    /// usable). `Some` only on the score-active path (decode-step RoPE-applied Q capture); `None` otherwise
    /// (MQ-3/MQ-4 hot-path gate). discriminant 4 — existing 0–3 unchanged (C-ABI additive, MQ-5).
    QueryStats,
}

/// dtype-agnostic tensor shape (POD). Only flat fields that can cross a future FFI boundary as-is (`#[repr(C)]`-able).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// number of valid rows (usually `current_pos`; QueryStats=2 = mean row + var row, MQ-1).
    pub rows: usize,
    /// number of f32 elements per row (Key/Value=head_dim, AttnWeights/Scores=1, QueryStats=head_dim).
    pub cols: usize,
    /// whether rows are split per-kv-head (true for all 5 current kinds; layer-wide flat goes through the separate `importance()` path).
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
/// directly (an exception, since routing a scalar through per-element read_row would be a net loss for the H2O ranking path).
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

    /// flat per-token importance score. `Some` → score-based (h2o heavy-hitter, d2o token rank),
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

    /// Fills raw V(`pos`,`head`) into `out` as f32 (the v_i of CAOTE/VATP). Sugar over `tensor(Value)`.
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

    /// The previous decode step's per-head attention weight at `(kv_head, pos)` (the a_i of CAOTE). Sugar over `tensor(AttnWeights)`.
    /// If `has_attn_weights()==false` it is meaningless (0.0) — CAOTE is advised to fall back to `importance()`.
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
}

/// A weighted merge instruction. Sums the evicted tokens (`from`) with weights into a single retained token's slot (`into`).
/// `Σ from.1 + into_weight ≈ 1` (magnitude preservation, d2o Eq.11 weights).
///
/// The `into`/`from` positions are logical coordinates just before compact is applied (pre-compact). The weights are baked into the plan,
/// and the engine executor (`apply_merges`) uses them as-is (replacing the current uniform merge). A merge-free policy uses an empty Vec.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMerge {
    /// The position of the retained token being merged into (the slot where the weighted sum accumulates).
    pub into: usize,
    /// The weight of `into` itself (the `w_c` of d2o Eq.11).
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

/// The plan a technique produces. `keep` (exclusive) ⊥ `merges` (orthogonal). `new_pos` is not carried — the engine
/// derives it from `keep.len()` (assuming all heads of [`KeepSpec::PerHead`] are of equal length).
#[derive(Clone, Debug, PartialEq)]
pub struct KVCachePlan {
    /// The shape of retained tokens.
    pub keep: KeepSpec,
    /// Weighted merge instructions (empty Vec if none).
    pub merges: Vec<WeightedMerge>,
}

/// The stage-axis extension technique surface — it adjusts resident tokens (a sibling of the engine-side storage-representation trait `KVCacheFormat`).
//
///
/// plan-returning, not self-mutating (D1): rather than touching buffers directly, it returns a [`KVCachePlan`] which the engine
/// executes via `KVCacheFormat::compact`. Hence this trait does not take engine types like `&mut KVCache`,
/// only the [`StageCtx`] read abstraction — consistent with a C-ABI future (`cdylib` promotion), and keeping technique crates from coupling to
/// engine internals.
pub trait KVCacheStage: Send + Sync {
    /// The technique name (matched against the CLI `eviction plugin --name <name>` selector, also for logging). Must be unique within the slice.
    fn name(&self) -> &str;

    /// Computes the retain/merge plan. `None` = not applied (no-op). Computed from ctx reads + impl state (Mutex).
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan>;
}

/// The common parameters needed to create a technique instance. The engine maps CLI args into this struct and passes it along
/// (carrying only flat values so argus-extension-api does not depend on the engine's args type).
///
/// NOTE: technique-private parameters (e.g. d2o's `ema_beta`/`merge_e`/`merge_axis`/`protected_layers`)
/// are deliberately **not carried here** — rather than bloat this shared struct, they ride the opaque
/// [`StageArgs`] blob into [`KVCacheStageReg::make_with_args`], where the plugin parses its own params
/// (see `d2o::D2OConfig::from_args`). This keeps the engine from knowing any plugin's private knobs.
/// The 5 fields below are the common params shared by the built-ins (sliding/streaming/h2o/no_eviction).
#[repr(C)] // GATE-C: the `.so` C-ABI passes it by value as a POD (the make-thunk argument).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageParams {
    /// sliding window size (number of recent tokens to keep).
    pub eviction_window: usize,
    /// the prefix length to protect at the front (BOS / system prompt, etc.).
    pub protected_prefix: usize,
    /// heavy-hitter keep ratio (H2O family).
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

/// The technique-private argument blob passed to [`KVCacheStageReg::make_with_args`]. Empty (`&[]`)
/// for built-ins served entirely by [`StageParams`].
pub type StageArgs<'a> = &'a [PluginArg<'a>];

/// Plugin-declared capabilities the engine reads **before** instantiating a stage (off the
/// [`KVCacheStageReg`], not via a trait method — the decision precedes `make`). This is the surface
/// that lets the engine CLI/chat/eval/bench paths stay free of any plugin-name knowledge: instead of
/// `matches!(name, "h2o" | "d2o" | ...)` capability lists and `match name { ... => 4 }` prefix
/// tables, each consumer reads these caps generically through [`stage_caps`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageCaps {
    /// Whether `plan()` consumes [`StageCtx::importance`]. When `true` the engine wires an attention
    /// score accumulator and routes per-token (and, for per-head stages, per-head) scores into the
    /// stage; when `false` the stage is score-free (sliding/streaming/no-eviction). Replaces the
    /// scattered `matches!(name, "h2o" | "h2o_plus" | "d2o" | "caote" | "rkv")` capability checks.
    pub is_score_based: bool,
    /// The default `--protected-prefix` to apply when the user omits it. Score-based stages use `4`
    /// (attention sinks — protecting the whole prompt would defeat heavy-hitter selection); `0` means
    /// "no stage-declared default — the engine applies its own fallback" (sliding/streaming/none let
    /// the engine pick the recency/prompt-length default). Replaces the `match name { ... => 4 }`
    /// prefix tables.
    pub default_protected_prefix: usize,
}

impl StageCaps {
    /// Score-free defaults — no importance, no stage-declared prefix (`{ false, 0 }`). Used by the
    /// `register_kv_stage!` macro so macro-registered (and example) plugins compile unchanged: a
    /// score-free LayerWide technique is the common case, and any stage that needs scores declares
    /// `is_score_based: true` via a direct-literal [`KVCacheStageReg`].
    pub const SCORE_FREE: StageCaps = StageCaps {
        is_score_based: false,
        default_protected_prefix: 0,
    };
}

/// The registration entry for one stage technique. A technique crate submits it via
/// `#[distributed_slice(KV_CACHE_STAGES)] static FOO: KVCacheStageReg = ...`.
pub struct KVCacheStageReg {
    /// The CLI selector name (`eviction plugin --name <name>`, or a built-in `eviction <policy>`). Must be unique within the slice.
    pub name: &'static str,
    /// The factory that builds a technique instance from the common parameters (no private args).
    pub make: fn(StageParams) -> Box<dyn KVCacheStage>,
    /// Like [`KVCacheStageReg::make`] but also receives the technique-private [`StageArgs`] blob —
    /// CLI knobs that do not fit [`StageParams`]. Techniques that take no private args set this to a
    /// shim that drops the blob and delegates to `make` (`register_kv_stage!` wires that shim
    /// automatically). The engine calls this via `make_stage_with_args`; `make_stage` passes `&[]`.
    pub make_with_args: fn(StageParams, StageArgs<'_>) -> Box<dyn KVCacheStage>,
    /// Capabilities the engine reads pre-`make` ([`StageCaps`]) — whether the stage is score-based and
    /// its default protected prefix. Read via [`stage_caps`] so consumers never name a plugin.
    pub caps: StageCaps,
}

/// The global registration slice — the registrations of all linked technique crates are gathered at **link time**.
///
/// fat-LTO + `--gc-sections` may silently drop unreferenced sections — in release
/// builds the engine asserts via a startup self-test that all expected techniques are registered, failing fast.
#[distributed_slice]
pub static KV_CACHE_STAGES: [KVCacheStageReg] = [..];

/// Finds a registered technique by name (used at engine construction).
pub fn find_stage(name: &str) -> Option<&'static KVCacheStageReg> {
    KV_CACHE_STAGES.iter().find(|r| r.name == name)
}

/// All registered technique names (for self-test / diagnostics).
pub fn registered_names() -> Vec<&'static str> {
    KV_CACHE_STAGES.iter().map(|r| r.name).collect()
}

/// The [`StageCaps`] of a statically registered technique, by name. `None` if the name is not a
/// statically linked stage. This is the lookup the engine CLI/chat/eval/bench paths use to read a
/// stage's score-based-ness and default protected prefix **without naming any plugin** — the one
/// site that makes the name-match collapse possible.
pub fn stage_caps(name: &str) -> Option<StageCaps> {
    find_stage(name).map(|r| r.caps)
}

// ════════════════════════════════════════════════════════════════════════════
// GATE-C — Stage-axis `.so` cdylib dlopen plugin C-ABI
// ════════════════════════════════════════════════════════════════════════════
//
// Static registration (`KV_CACHE_STAGES` + `find_stage`) is left in place (D3 additive), and a `.so` plugin
// adds a surface that exposes the same `KVCacheStage` over a C-ABI. trait objects (`&dyn StageCtx`,
// `Box<dyn KVCacheStage>`) are C-ABI-unstable, so they are flattened into a fn-ptr table ([`StageCtxAbi`]) + an opaque
// handle (D2). The [`AbiStageCtx`] adapter re-implements `StageCtx` on top of `StageCtxAbi`,
// so plugin authors write the same `impl KVCacheStage` code whether static or dynamic.

/// The ABI version of the `register_kv_stages_v2` envelope ([`StageExportAbi`]). The host refuses to load on mismatch.
pub const KV_STAGE_ABI_VERSION: u32 = 2;

/// [`PluginVTableAbi::plan`] return code: success (`out_plan` filled).
pub const KV_PLAN_OK: i32 = 0;
/// [`PluginVTableAbi::plan`] return code: no-op (`None` — eviction not applied).
pub const KV_PLAN_NOOP: i32 = 1;
// negative = a clean logical error from the plugin (the host logs it and treats it as no-op; not a panic).

/// The C-ABI flattening of [`StageCtx`] (D2). The host fills the fn-ptrs over a concrete ctx (`ctx`) and passes them to the plugin,
/// and the plugin calls only the fn-ptrs. Every fn-ptr dereferences a raw pointer, hence `unsafe`.
///
/// Constructed stack-local (per-plan-call) and passed by ptr — not `static` → no `Sync` required.
#[repr(C)]
pub struct StageCtxAbi {
    /// An opaque thin pointer to the host's concrete `&dyn StageCtx` implementation (the first argument of the fn-ptrs below).
    pub ctx: *const c_void,
    /// [`StageCtx::current_pos`].
    pub current_pos: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::target_len`].
    pub target_len: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::layer_idx`].
    pub layer_idx: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::n_kv_heads`].
    pub n_kv_heads: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::head_dim`].
    pub head_dim: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::importance`]. `Some` → `true` + fills `out_ptr`/`out_len` (the borrow lives for the ctx lifetime),
    /// `None` → `false`.
    pub importance:
        unsafe extern "C" fn(*const c_void, out_ptr: *mut *const f32, out_len: *mut usize) -> bool,
    /// Flattening of [`TensorHandle::read_row`]. `kind` ([`TensorKind`] u32) available + read → `true` (fills `out`,
    /// `out_len == shape().cols` contract — verified by the host), unavailable → `false`.
    pub tensor_read_row: unsafe extern "C" fn(
        *const c_void,
        kind: u32,
        row: usize,
        kv_head: usize,
        out: *mut f32,
        out_len: usize,
    ) -> bool,
    /// Flattening of [`TensorHandle::shape`]. `kind` available → `true` (fills `out`), unavailable → `false`.
    pub tensor_shape: unsafe extern "C" fn(*const c_void, kind: u32, out: *mut TensorShape) -> bool,
}

/// C-ABI flattening of [`KVCachePlan`] (D5). Exposes the stable buffer owned by the plugin-arena as (ptr+len);
/// after the host copies it immediately, the plugin reclaims its own arena via [`PluginVTableAbi::plan_free`]`(owner)`
/// ("each side frees its own" — blocks cross-allocator UB).
#[repr(C)]
pub struct PlanAbi {
    /// `0` = [`KeepSpec::LayerWide`], `1` = [`KeepSpec::PerHead`] (reserved for v1 — the host bails).
    pub keep_kind: u32,
    /// Kept positions, ascending. LayerWide = all heads, PerHead = concatenation of all heads.
    pub keep_ptr: *const usize,
    /// Length of `keep_ptr`.
    pub keep_len: usize,
    /// PerHead only: per-head keep length (LayerWide = null).
    pub keep_outer_lens: *const usize,
    /// Length of `keep_outer_lens` (= n_kv_heads, LayerWide = 0).
    pub keep_outer_count: usize,
    /// Array of weighted merges (len 0 if none).
    pub merges_ptr: *const MergeAbi,
    /// Length of `merges_ptr`.
    pub merges_len: usize,
    /// Plugin-arena-owned handle → `plan_free(owner)`. The host must never free it directly.
    pub owner: *mut c_void,
}

impl PlanAbi {
    /// Initial out-param value (all null/0). The host passes it to the plugin by `&mut`.
    pub fn zeroed() -> Self {
        PlanAbi {
            keep_kind: 0,
            keep_ptr: core::ptr::null(),
            keep_len: 0,
            keep_outer_lens: core::ptr::null(),
            keep_outer_count: 0,
            merges_ptr: core::ptr::null(),
            merges_len: 0,
            owner: core::ptr::null_mut(),
        }
    }
}

/// C-ABI flattening of [`WeightedMerge`] (D5). `from` becomes a separate [`FromPairAbi`] array.
#[repr(C)]
pub struct MergeAbi {
    /// [`WeightedMerge::into`].
    pub into: usize,
    /// [`WeightedMerge::into_weight`].
    pub into_weight: f32,
    /// `(pos, weight)` array (plugin-arena-owned).
    pub from_ptr: *const FromPairAbi,
    /// Length of `from_ptr`.
    pub from_len: usize,
    /// Flattening of [`WeightedMerge::apply_to`] (Both=0 / KeyOnly=1 / ValueOnly=2). **Appended at the end** of the fields —
    /// existing field offsets are unchanged; the region not filled by an old plugin `.so` is zero-init'd by the host to 0 (= Both).
    pub apply_to: u32,
}

/// C-ABI POD for a `(pos, weight)` pair (D5).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FromPairAbi {
    /// Position of the evicted token.
    pub pos: usize,
    /// Merge weight.
    pub weight: f32,
}

/// C-ABI vtable for a single stage (D2). In v2 the plugin accumulates these vtables into the [`PLUGIN_KV_STAGE_VTABLES`]
/// slice, and `register_kv_stages_v2()` exposes them wrapped in a [`StageExportAbi`] envelope (one `.so` may host multiple
/// stages). The vtable is a `static` in the plugin `.so`, so it is valid for the entire process lifetime.
#[repr(C)]
pub struct PluginVTableAbi {
    /// Null-terminated canonical name (matched against the CLI `eviction plugin --name`). A `'static` str in the plugin `.so`.
    /// (ABI gating is handled by the envelope's [`StageExportAbi::abi_version`] — no per-vtable version field.)
    pub name: *const c_char,
    /// `StageParams` → opaque plugin instance handle.
    pub make: unsafe extern "C" fn(*const StageParams) -> *mut c_void,
    /// handle + ctx → plan. Returns [`KV_PLAN_OK`]/[`KV_PLAN_NOOP`]/negative (err). Fills `out_plan`.
    pub plan: unsafe extern "C" fn(*mut c_void, *const StageCtxAbi, *mut PlanAbi) -> i32,
    /// Reclaims the [`PlanAbi::owner`] that `plan` filled (called by the host right after copying the plan).
    pub plan_free: unsafe extern "C" fn(owner: *mut c_void),
    /// Releases the plugin instance handle (called by the host when the stage is dropped).
    pub drop: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: the vtable is immutable and `name` points at a `'static` str in the plugin `.so`. fn-ptrs are inherently
// Send+Sync. Therefore it is safe to share across threads — required for declaring the plugin's distributed_slice element static.
unsafe impl Sync for PluginVTableAbi {}

/// stage-axis envelope — declares all of a plugin `.so`'s stage vtables at once. `register_kv_stages_v2()`
/// returns it **by value** (sret >16B; `count`/`vtables` are derived from the slice at runtime, so a const static is not possible).
/// `vtables` is the [`PLUGIN_KV_STAGE_VTABLES`] base (a `.so` static) → valid for the `.so` lifetime; `count==0` is possible (empty axis).
#[repr(C)]
pub struct StageExportAbi {
    /// [`KV_STAGE_ABI_VERSION`]. The host rejects the `.so` on mismatch (one ABI per `.so`).
    pub abi_version: u32,
    /// Length of the contiguous array pointed to by `vtables`.
    pub count: usize,
    /// A contiguous array of `count` [`PluginVTableAbi`] (a `.so` static). The loader accesses elements via `vtables.add(i)`.
    pub vtables: *const PluginVTableAbi,
}

/// The slice that accumulates stage vtables inside the plugin `.so`. **Declared in exactly one place, argus-extension-api**
/// (the linkme section name is determined by the declaring static's name — declaring it on the plugin side breaks cross-crate contribution). `register_kv_stage!`
/// contributes via a const-block-isolated static under the plugin-cdylib gate (multiple calls = multiple stages). In a static build it stays empty and
/// harmless (the engine reads only `KV_CACHE_STAGES`).
#[distributed_slice]
pub static PLUGIN_KV_STAGE_VTABLES: [PluginVTableAbi] = [..];

/// Adapter that exposes one [`TensorKind`] of a [`StageCtxAbi`] as a [`TensorHandle`] (internal to [`AbiStageCtx`]).
/// `abi` borrow is tied to the adapter's lifetime.
pub struct AbiTensorHandle {
    abi: *const StageCtxAbi,
    kind: u32,
    shape: TensorShape,
}

impl TensorHandle for AbiTensorHandle {
    fn shape(&self) -> TensorShape {
        self.shape
    }
    fn dtype(&self) -> TensorDtype {
        // ABI reads always yield f32 (read_row outputs f32). The stored dtype is not carried by the v1 C-ABI
        // (a diagnostic field — add a tensor_dtype fn-ptr in abi_version 2 if needed).
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        // SAFETY: `abi` is valid for the AbiStageCtx lifetime (contract at construction). The out_len contract (== cols) is the caller's responsibility.
        unsafe {
            let a = &*self.abi;
            (a.tensor_read_row)(a.ctx, self.kind, row, kv_head, out.as_mut_ptr(), out.len());
        }
    }
}

/// Adapter that re-implements [`StageCtx`] on top of [`StageCtxAbi`] (a C fn-ptr table) (D2 "write-once,
/// link-either-way"). The plugin's plan thunk wraps the host's `StageCtxAbi` in this to call the same `impl
/// KVCacheStage::plan(&dyn StageCtx)`.
pub struct AbiStageCtx {
    abi: *const StageCtxAbi,
    // TensorKind (repr u32: Key=0/Value=1/AttnWeights=2/Scores=3/QueryStats=4) index. shape is probed
    // at construction. Adding QueryStats (disc 4) is a C-ABI addition (MQ-5) — fn-ptr signatures unchanged, no effect on existing .so files.
    handles: [Option<AbiTensorHandle>; 5],
}

impl AbiStageCtx {
    /// # Safety
    /// `abi` must point at a valid [`StageCtxAbi`], and its `ctx` + all fn-ptrs must stay alive for this adapter's
    /// lifetime (guaranteed by the host for the duration of the plan call).
    pub unsafe fn new(abi: *const StageCtxAbi) -> Self {
        let a = unsafe { &*abi };
        let kinds = [
            TensorKind::Key,
            TensorKind::Value,
            TensorKind::AttnWeights,
            TensorKind::Scores,
            TensorKind::QueryStats,
        ];
        let mut handles: [Option<AbiTensorHandle>; 5] = [None, None, None, None, None];
        for kind in kinds {
            let mut shape = TensorShape {
                rows: 0,
                cols: 0,
                per_head: false,
            };
            // SAFETY: per new's contract, the fn-ptrs and ctx are valid.
            let ok = unsafe { (a.tensor_shape)(a.ctx, kind as u32, &mut shape) };
            if ok {
                handles[kind as u32 as usize] = Some(AbiTensorHandle {
                    abi,
                    kind: kind as u32,
                    shape,
                });
            }
        }
        AbiStageCtx { abi, handles }
    }
}

impl StageCtx for AbiStageCtx {
    fn current_pos(&self) -> usize {
        // SAFETY: new's contract.
        unsafe {
            let a = &*self.abi;
            (a.current_pos)(a.ctx)
        }
    }
    fn target_len(&self) -> usize {
        unsafe {
            let a = &*self.abi;
            (a.target_len)(a.ctx)
        }
    }
    fn layer_idx(&self) -> usize {
        unsafe {
            let a = &*self.abi;
            (a.layer_idx)(a.ctx)
        }
    }
    fn n_kv_heads(&self) -> usize {
        unsafe {
            let a = &*self.abi;
            (a.n_kv_heads)(a.ctx)
        }
    }
    fn head_dim(&self) -> usize {
        unsafe {
            let a = &*self.abi;
            (a.head_dim)(a.ctx)
        }
    }
    fn importance(&self) -> Option<&[f32]> {
        // SAFETY: new's contract. The returned slice's borrow is tied to &self and valid within the ctx lifetime the host guarantees.
        unsafe {
            let a = &*self.abi;
            let mut ptr: *const f32 = core::ptr::null();
            let mut len: usize = 0;
            if (a.importance)(a.ctx, &mut ptr, &mut len) && !ptr.is_null() {
                Some(core::slice::from_raw_parts(ptr, len))
            } else {
                None
            }
        }
    }
    fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
        self.handles[kind as u32 as usize]
            .as_ref()
            .map(|h| h as &dyn TensorHandle)
    }
}

/// plugin-arena: the plan thunk flattens [`KVCachePlan`] and stores it in a stable buffer (D5). [`Self::into_abi`]
/// leaks a Box that the host reclaims via [`Self::free`]. It is self-referential ([`MergeAbi::from_ptr`] points at
/// `from_storage`), but it is safe because the Vec heap buffers never move.
pub struct PlanArena {
    keep: Vec<usize>,
    keep_outer_lens: Vec<usize>,
    keep_kind: u32,
    // The backing buffer that `merges[i].from_ptr` points at. Since it is referenced only via a raw ptr (no direct reads),
    // the compiler cannot see it, but the arena must own it so the ptr does not dangle on drop.
    #[allow(dead_code)]
    from_storage: Vec<Vec<FromPairAbi>>,
    merges: Vec<MergeAbi>,
}

impl PlanArena {
    fn from_plan(plan: KVCachePlan) -> Self {
        let (keep_kind, keep, keep_outer_lens) = match plan.keep {
            KeepSpec::LayerWide(k) => (0u32, k, Vec::new()),
            KeepSpec::PerHead(heads) => {
                let lens: Vec<usize> = heads.iter().map(|h| h.len()).collect();
                let flat: Vec<usize> = heads.into_iter().flatten().collect();
                (1u32, flat, lens)
            }
        };
        // Fill from_storage first to secure stable addresses, then merges point at it.
        let mut from_storage: Vec<Vec<FromPairAbi>> = Vec::with_capacity(plan.merges.len());
        for m in &plan.merges {
            from_storage.push(
                m.from
                    .iter()
                    .map(|&(pos, weight)| FromPairAbi { pos, weight })
                    .collect(),
            );
        }
        let merges: Vec<MergeAbi> = plan
            .merges
            .iter()
            .enumerate()
            .map(|(i, m)| MergeAbi {
                into: m.into,
                into_weight: m.into_weight,
                from_ptr: from_storage[i].as_ptr(),
                from_len: from_storage[i].len(),
                apply_to: m.apply_to as u32,
            })
            .collect();
        PlanArena {
            keep,
            keep_outer_lens,
            keep_kind,
            from_storage,
            merges,
        }
    }

    /// Flattens and leaks the plan, then builds a [`PlanAbi`] pointing at it. Valid until the host releases it
    /// via [`Self::free`]`(owner)`.
    pub fn into_abi(plan: KVCachePlan) -> PlanAbi {
        let arena = Box::new(Self::from_plan(plan));
        let raw = Box::into_raw(arena);
        // SAFETY: a valid pointer just leaked.
        let a = unsafe { &*raw };
        PlanAbi {
            keep_kind: a.keep_kind,
            keep_ptr: a.keep.as_ptr(),
            keep_len: a.keep.len(),
            keep_outer_lens: if a.keep_outer_lens.is_empty() {
                core::ptr::null()
            } else {
                a.keep_outer_lens.as_ptr()
            },
            keep_outer_count: a.keep_outer_lens.len(),
            merges_ptr: a.merges.as_ptr(),
            merges_len: a.merges.len(),
            owner: raw as *mut c_void,
        }
    }

    /// # Safety
    /// `owner` must be a `PlanArena` pointer created by [`Self::into_abi`], and this must be called exactly once.
    pub unsafe fn free(owner: *mut c_void) {
        if !owner.is_null() {
            drop(unsafe { Box::from_raw(owner as *mut PlanArena) });
        }
    }
}

/// Dual-wiring macro (D2) that registers a stage plugin on both the static (rlib→linkme) and dynamic (cdylib→C-ABI) paths.
///
/// `$make` is the same `fn(StageParams) -> Box<dyn KVCacheStage>` as the existing [`KVCacheStageReg::make`]
/// (closures allowed). The dynamic C-ABI export (`register_kv_stage_v1`) is gated behind the `plugin-cdylib` feature,
/// which eliminates `#[no_mangle]` symbol collisions under static force-link (only `.so` builds use `--features plugin-cdylib`).
///
/// **Callable multiple times** within a single plugin crate (`.so`) for multiple stages. Every contributed static is
/// isolated in an anonymous `const _: () = {}` scope so invocations do not collide (linkme does not rename a static element's ident,
/// so scope isolation is the only workaround). The `.so` entry point (`register_kv_stages_v2`) is emitted once per `.so` by a separate [`export_plugin!`].
///
///
/// ```ignore
/// argus_extension_api::register_kv_stage!("example_keep_recent", |_p| Box::new(KeepRecent));
/// argus_extension_api::export_plugin!();   // once per .so
/// ```
#[macro_export]
macro_rules! register_kv_stage {
    ($name:literal, $make:expr) => {
        // ── Static path (rlib → linkme distributed_slice). const-block isolation = multiple invocations allowed (E2). ──
        const _: () = {
            #[$crate::distributed_slice($crate::KV_CACHE_STAGES)]
            static __REG: $crate::KVCacheStageReg = $crate::KVCacheStageReg {
                name: $name,
                make: $make,
                // args-ignoring shim: macro-registered stages take no technique-private args, so
                // make_with_args drops the blob and delegates to `make` (keeps the macro surface unchanged).
                make_with_args: {
                    fn __mwa(
                        p: $crate::StageParams,
                        _args: $crate::StageArgs<'_>,
                    ) -> ::std::boxed::Box<dyn $crate::KVCacheStage> {
                        let f: fn(
                            $crate::StageParams,
                        ) -> ::std::boxed::Box<dyn $crate::KVCacheStage> = $make;
                        f(p)
                    }
                    __mwa
                },
                // Macro-registered stages declare score-free defaults; a stage that needs scores or a
                // non-default prefix registers via a direct-literal `KVCacheStageReg` with its own caps.
                caps: $crate::StageCaps::SCORE_FREE,
            };
        };

        // ── Dynamic path (cdylib → contributes to PLUGIN_KV_STAGE_VTABLES). plugin-cdylib gate keeps it un-emitted on static builds. ──
        // The entry point (register_kv_stages_v2) is emitted by export_plugin!; here we only contribute the vtable to the slice (E2).
        #[cfg(feature = "plugin-cdylib")]
        const _: () = {
            // Handle = Box<Box<dyn KVCacheStage>> (thin ptr). make/plan/drop share this representation.
            type __Handle = ::std::boxed::Box<dyn $crate::KVCacheStage>;

            unsafe extern "C" fn __make(p: *const $crate::StageParams) -> *mut ::core::ffi::c_void {
                // SAFETY: the host passes a valid StageParams pointer (D2). StageParams is a Copy POD.
                // $make (a Rust-ABI fn) is for internal calls here only — do not cast it directly to extern "C".
                let params = unsafe { *p };
                let make_fn: fn($crate::StageParams) -> __Handle = $make;
                let stage: __Handle = make_fn(params);
                ::std::boxed::Box::into_raw(::std::boxed::Box::new(stage))
                    as *mut ::core::ffi::c_void
            }

            unsafe extern "C" fn __plan(
                h: *mut ::core::ffi::c_void,
                ctx: *const $crate::StageCtxAbi,
                out: *mut $crate::PlanAbi,
            ) -> i32 {
                // SAFETY: h is the Box<Box<dyn>> created by __make, and ctx is a valid StageCtxAbi filled in by the host (D2).
                let stage: &dyn $crate::KVCacheStage = unsafe { &**(h as *const __Handle) };
                let abi_ctx = unsafe { $crate::AbiStageCtx::new(ctx) };
                match stage.plan(&abi_ctx) {
                    ::core::option::Option::None => $crate::KV_PLAN_NOOP,
                    ::core::option::Option::Some(plan) => {
                        let abi = $crate::PlanArena::into_abi(plan);
                        // SAFETY: out is a valid &mut PlanAbi provided by the host.
                        unsafe {
                            *out = abi;
                        }
                        $crate::KV_PLAN_OK
                    }
                }
            }

            unsafe extern "C" fn __plan_free(owner: *mut ::core::ffi::c_void) {
                // SAFETY: owner is the PlanArena created by into_abi, and the host calls this exactly once (D5).
                unsafe { $crate::PlanArena::free(owner) };
            }

            unsafe extern "C" fn __drop(h: *mut ::core::ffi::c_void) {
                // SAFETY: h is the Box<Box<dyn>> created by __make, and the host calls this exactly once.
                drop(unsafe { ::std::boxed::Box::from_raw(h as *mut __Handle) });
            }

            // Contributes the vtable to PLUGIN_KV_STAGE_VTABLES (const-block isolation = accumulates across multiple invocations). Not an entry point.
            #[$crate::distributed_slice($crate::PLUGIN_KV_STAGE_VTABLES)]
            static __VTABLE: $crate::PluginVTableAbi = $crate::PluginVTableAbi {
                name: ::core::concat!($name, "\0").as_ptr() as *const ::core::ffi::c_char,
                make: __make,
                plan: __plan,
                plan_free: __plan_free,
                drop: __drop,
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

// ── weight stage plugin (isomorphic to KVCacheStage) ──

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

/// A weight stage's plan output (mirror of KV's `KVCachePlan`). Rust-native data holding decisions only
/// (step/boundary-tier, no repr(C) needed). Mutation is performed by the engine executor (D3).
#[derive(Debug, Clone, Default)]
pub struct WeightDispatchPlan {
    /// Per-layer directives. Empty means no-op.
    pub per_layer: Vec<LayerDirective>,
}

/// Plan-returning technique trait for the weight axis (mirror of KV's `KVCacheStage`).
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

/// Registration entry for a single weight stage technique (mirror of KV's `KVCacheStageReg`).
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

// ── Format-axis plugin registry (isomorphic to KVCacheStage) ──

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
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    /// Contiguous raw (f32/f16).
    Dense,
    /// nibble (4-bit) packing (q4_0/q4_1).
    Nibble,
    /// byte (8-bit) packing (q8_0).
    Byte,
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
/// superseded. The keep/merge decision belongs to the stage axis (`KVCacheStage::plan → KVCachePlan`),
/// while mutation is owned exclusively by the engine executor `execute_kv_plan` (decision=plugin, mutation=engine). Pulling compact into the format axis
/// would blur the stage⊥format orthogonality at the decision layer and leak `Merge` (engine) into the api. Therefore
/// `KVFormat`'s layer-tier contribution is only the `layout()` descriptor read (M-F2, the L1 repr(C) boundary).
pub trait KVFormat: Send + Sync {
    /// canonical format name (e.g. "q4_0"/"f16"/"f32"). Unique within the slice.
    fn name(&self) -> &str;

    /// This format's storage layout descriptor (read by the engine's generic reader on the hot path, D3).
    fn layout(&self) -> KVLayoutDesc;
}

/// Registration entry for one format technique (mirror of KV `KVCacheStageReg`).
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
// GATE-C v2 — Format-axis `.so` cdylib dlopen plugin C-ABI
// ════════════════════════════════════════════════════════════════════════════
//
// Isomorphic to the stage axis (GATE-C above) but **overwhelmingly simpler**: [`KVFormat`] has zero callbacks (no ctx,
// no plan — a pure descriptor, `name`+`layout`), so [`StageCtxAbi`]/[`PlanAbi`]/[`PlanArena`] are all
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
/// counterpart of [`register_kv_stage!`].
///
/// **May be called multiple times** within one plugin crate (`.so`) (multiple formats = a quant family). All contributed
/// statics are isolated in anonymous `const _: () = {}` scopes. The `.so` entry (`register_kv_formats_v2`) is emitted separately,
/// once per `.so`, by [`export_plugin!`]. The format-axis counterpart of [`register_kv_stage!`].
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
/// [`PLUGIN_KV_STAGE_VTABLES`]/[`PLUGIN_KV_FORMAT_VTABLES`] slices accumulated by `register_kv_*!` as by-value envelopes.
///
/// **Three-axis separate-symbol invariant**: `register_kv_stages_v2` ⊥ `register_kv_formats_v2` ⊥
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
            /// Stage envelope entry — returns `PLUGIN_KV_STAGE_VTABLES` by-value (sret).
            #[unsafe(no_mangle)] // Rust 2024: no_mangle is an unsafe attribute.
            pub extern "C" fn register_kv_stages_v2() -> $crate::StageExportAbi {
                // .len()/.as_ptr() are evaluated at runtime via linkme's Deref (static_slice) — the envelope is computed at call time.
                $crate::StageExportAbi {
                    abi_version: $crate::KV_STAGE_ABI_VERSION,
                    count: $crate::PLUGIN_KV_STAGE_VTABLES.len(),
                    vtables: $crate::PLUGIN_KV_STAGE_VTABLES.as_ptr(),
                }
            }

            /// Format envelope entry — returns `PLUGIN_KV_FORMAT_VTABLES` by-value (sret).
            #[unsafe(no_mangle)]
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

/// Registration entry for one backend capability (mirror of KV `KVCacheStageReg`).
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

// ── Backend-capability axis — ATTENTION (quantized fused attention, e.g. KIVI) category dynamic C-ABI ──
//
// D8 (single-trait): argus-extension-api owns the canonical [`QuantAttnBackend`]. The engine's static OpenCL
// impl, the host dlopen adapter, and the plugin `.so` **all implement this one trait** (isomorphic to the stage `KVCacheStage`). The signatures take
// ABI structs (`QuantAttnArgs`/`QuantAttnGatherArgs`, cl_mem `*mut c_void`) rather than `&Tensor`, so the plugin does not reference engine
// types (independent). The static `BACKEND_CAPABILITIES` above (keyed by name) stays as-is — for the fat-LTO name-survival smoke test.

/// ABI version of the `register_backend_caps_v2` envelope ([`BackendCapExportAbi`]). The host rejects the `.so` on mismatch.
///
/// v2 (FORMAT Phase 2, Stage A): [`QuantAttnVTable`] gained `dequant_flush`/`scatter_residual`
/// fn-ptrs so the residual-flush GPU kernels cross the cap trait (closing the engine's
/// concrete-`OpenCLBackend` downcast on the live FORMAT-flush path). A v1 `.so` is rejected.
pub const BACKEND_CAP_ABI_VERSION: u32 = 2;

/// Capability category tag — ATTENTION (quantized fused dequant+attention, e.g. KIVI). [`BackendCapVTableAbi::category`].
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

/// Static (force-link) quantized-attention capability registration entry — the backend-axis counterpart of the stage [`KVCacheStageReg`] (D8).
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
        // keep `$make`/its associated types reachable (isomorphic to the Stage `register_kv_stage!`, no unused warnings).
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
// KV read-plan surface — the 4th plan-returning plugin surface, deciding "what to read".
// A parallel mirror copy of KVCacheStage (eviction) / WeightStage (dispatch) / KVFormat.
// ════════════════════════════════════════════════════════════════════════════

/// The **granularity abstraction** for KV-cache reads.
///
/// `Token` = `select` is a subset of KV token positions (pos).
/// `Page { page_size }` = `select` is a subset of page indices, and each page groups `page_size` tokens.
///
/// NOTE: `Page { page_size }` is a variant with a field, so `#[repr(u32)]` cannot be applied directly. The C-ABI flattening
/// (`KVReadPlanAbi`) is to be defined with separate `granularity: u32 + page_size: u32` fields.
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

/// The CLI-derived static configuration for a read stage (a mirror of the KV `StageParams`). The static knobs of Quest (the first built-in).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReadStageParams {
    /// Page size (number of tokens grouped per page). Quest default 16.
    pub page_size: u32,
    /// Fraction of pages to select (1/`top_k_ratio_denom` of all pages). Quest default 4 (= 1/4 of the total).
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

/// The registration entry for one read stage technique (a mirror of `KVCacheStageReg`).
pub struct KVReadStageReg {
    /// CLI `--read-stage` name. Must be unique within the slice.
    pub name: &'static str,
    /// Factory that builds a technique instance from the parameters.
    pub make: fn(ReadStageParams) -> Box<dyn KVReadStage>,
}

/// Global read-stage registration slice — the **4th parallel linkme registry** on the stage axis.
///
/// **Starts with zero built-ins** — when no read stage is present the engine always does a full read (100% of current behavior preserved, D5).
/// The first built-in will be Quest (to be registered in S4/S5).
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// No-op technique for verifying the registration/lookup round-trip.
    struct Dummy;
    impl KVCacheStage for Dummy {
        fn name(&self) -> &str {
            "dummy"
        }
        fn plan(&self, _ctx: &dyn StageCtx) -> Option<KVCachePlan> {
            None
        }
    }

    /// Minimal ctx stub to close the `plan` call path (all accessors trivial).
    struct DummyCtx;
    impl StageCtx for DummyCtx {
        fn current_pos(&self) -> usize {
            0
        }
        fn target_len(&self) -> usize {
            0
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn importance(&self) -> Option<&[f32]> {
            None
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            1
        }
        // Only tensor() is implemented — head_score/has_head_scores/dequant_k/dequant_v/attn_weight/
        // has_attn_weights are satisfied by the default sugar (all None → trivial).
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
    }

    #[distributed_slice(KV_CACHE_STAGES)]
    static DUMMY_REG: KVCacheStageReg = KVCacheStageReg {
        name: "dummy",
        make: |_params| Box::new(Dummy),
        make_with_args: |_params, _args| Box::new(Dummy),
        caps: StageCaps::SCORE_FREE,
    };

    #[test]
    fn dummy_registers_into_slice() {
        // Verify that linkme gathers registrations from the same crate into the slice.
        let reg = find_stage("dummy").expect("dummy must be registered in the slice");
        assert_eq!(reg.name, "dummy");
        let params = StageParams {
            eviction_window: 8,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 4,
            streaming_window: 0,
        };
        let stage = (reg.make)(params);
        assert_eq!(stage.name(), "dummy");
        assert!(stage.plan(&DummyCtx).is_none());
    }

    #[test]
    fn registered_names_contains_dummy() {
        assert!(registered_names().contains(&"dummy"));
    }

    // ── GATE-C C-ABI round-trip (in-process verification without a `.so`) ──

    /// Mock host concrete ctx — the StageCtxAbi fn-ptrs operate on top of this.
    struct HostCtx {
        cur: usize,
        tgt: usize,
        imp: Vec<f32>,
        key: Vec<f32>, // when non-empty, tensor(Key) is Some(1 row × len); when empty, None
        // For QueryStats (disc 4) probing — when non-empty, tensor(QueryStats) is Some(2 rows × qstats_cols); when empty, None.
        // Layout [2 * qstats_cols]: row0=mean, row1=var.
        qstats: Vec<f32>,
        qstats_cols: usize,
    }

    unsafe extern "C" fn h_current_pos(c: *const c_void) -> usize {
        unsafe { (*(c as *const HostCtx)).cur }
    }
    unsafe extern "C" fn h_target_len(c: *const c_void) -> usize {
        unsafe { (*(c as *const HostCtx)).tgt }
    }
    unsafe extern "C" fn h_layer_idx(_c: *const c_void) -> usize {
        0
    }
    unsafe extern "C" fn h_n_kv_heads(_c: *const c_void) -> usize {
        1
    }
    unsafe extern "C" fn h_head_dim(c: *const c_void) -> usize {
        unsafe { (*(c as *const HostCtx)).key.len().max(1) }
    }
    unsafe extern "C" fn h_importance(
        c: *const c_void,
        out_ptr: *mut *const f32,
        out_len: *mut usize,
    ) -> bool {
        let h = unsafe { &*(c as *const HostCtx) };
        if h.imp.is_empty() {
            return false;
        }
        unsafe {
            *out_ptr = h.imp.as_ptr();
            *out_len = h.imp.len();
        }
        true
    }
    unsafe extern "C" fn h_tensor_shape(
        c: *const c_void,
        kind: u32,
        out: *mut TensorShape,
    ) -> bool {
        let h = unsafe { &*(c as *const HostCtx) };
        if kind == TensorKind::Key as u32 && !h.key.is_empty() {
            unsafe {
                *out = TensorShape {
                    rows: 1,
                    cols: h.key.len(),
                    per_head: true,
                };
            }
            true
        } else if kind == TensorKind::QueryStats as u32 && !h.qstats.is_empty() {
            unsafe {
                *out = TensorShape {
                    rows: 2,
                    cols: h.qstats_cols,
                    per_head: true,
                };
            }
            true
        } else {
            false
        }
    }
    unsafe extern "C" fn h_tensor_read_row(
        c: *const c_void,
        kind: u32,
        _row: usize,
        _kv_head: usize,
        out: *mut f32,
        out_len: usize,
    ) -> bool {
        let h = unsafe { &*(c as *const HostCtx) };
        if kind == TensorKind::Key as u32 && out_len == h.key.len() {
            unsafe { core::ptr::copy_nonoverlapping(h.key.as_ptr(), out, out_len) };
            true
        } else if kind == TensorKind::QueryStats as u32
            && !h.qstats.is_empty()
            && out_len == h.qstats_cols
            && _row < 2
        {
            let base = _row * h.qstats_cols;
            unsafe { core::ptr::copy_nonoverlapping(h.qstats.as_ptr().add(base), out, out_len) };
            true
        } else {
            false
        }
    }

    fn make_abi(host: &HostCtx) -> StageCtxAbi {
        StageCtxAbi {
            ctx: host as *const HostCtx as *const c_void,
            current_pos: h_current_pos,
            target_len: h_target_len,
            layer_idx: h_layer_idx,
            n_kv_heads: h_n_kv_heads,
            head_dim: h_head_dim,
            importance: h_importance,
            tensor_read_row: h_tensor_read_row,
            tensor_shape: h_tensor_shape,
        }
    }

    #[test]
    fn abi_stage_ctx_reproduces_scalars_and_importance() {
        let host = HostCtx {
            cur: 100,
            tgt: 30,
            imp: vec![1.0, 2.0, 3.0],
            key: vec![],
            qstats: vec![],
            qstats_cols: 0,
        };
        let abi = make_abi(&host);
        // SAFETY: abi (and host) outlive the ctx.
        let ctx = unsafe { AbiStageCtx::new(&abi) };
        assert_eq!(ctx.current_pos(), 100);
        assert_eq!(ctx.target_len(), 30);
        assert_eq!(ctx.layer_idx(), 0);
        assert_eq!(ctx.n_kv_heads(), 1);
        assert_eq!(ctx.importance(), Some(&[1.0f32, 2.0, 3.0][..]));
        // key/qstats empty → tensor(Key)/Value/Scores/QueryStats all None (the default sugar is trivial too).
        assert!(ctx.tensor(TensorKind::Key).is_none());
        assert!(ctx.tensor(TensorKind::QueryStats).is_none());
        assert!(!ctx.has_head_scores());
    }

    /// MQ-5: AbiStageCtx 5-kind probing — when the host supplies QueryStats (disc 4), the adapter builds a handle
    /// (shape={2,cols,true}), and read_row(0)=mean / read_row(1)=var round-trip losslessly across the C-ABI fn-ptr.
    /// Confirms the existing kinds 0~3 are unaffected (the C-ABI is additive).
    #[test]
    fn abi_stage_ctx_probes_query_stats_5kind() {
        // head_dim=3: mean=[1,2,3], var=[0.5, 0.25, 0.125].
        let host = HostCtx {
            cur: 16,
            tgt: 8,
            imp: vec![],
            key: vec![],
            qstats: vec![1.0, 2.0, 3.0, 0.5, 0.25, 0.125],
            qstats_cols: 3,
        };
        let abi = make_abi(&host);
        let ctx = unsafe { AbiStageCtx::new(&abi) };
        let h = ctx
            .tensor(TensorKind::QueryStats)
            .expect("QueryStats available (disc 4 probing)");
        let sh = h.shape();
        assert_eq!(sh.rows, 2, "rows=2 (mean/var)");
        assert_eq!(sh.cols, 3, "cols=head_dim");
        assert!(sh.per_head);
        let mut mean = [0.0f32; 3];
        let mut var = [0.0f32; 3];
        h.read_row(0, 0, &mut mean);
        h.read_row(1, 0, &mut var);
        assert_eq!(mean, [1.0, 2.0, 3.0], "read_row(0)=mean");
        assert_eq!(var, [0.5, 0.25, 0.125], "read_row(1)=var");
        // Existing kinds unaffected.
        assert!(ctx.tensor(TensorKind::Key).is_none());
        assert!(ctx.tensor(TensorKind::Scores).is_none());
    }

    #[test]
    fn abi_stage_ctx_reproduces_tensor_read() {
        let host = HostCtx {
            cur: 10,
            tgt: 5,
            imp: vec![],
            key: vec![1.5, 2.5, 3.5, 4.5],
            qstats: vec![],
            qstats_cols: 0,
        };
        let abi = make_abi(&host);
        let ctx = unsafe { AbiStageCtx::new(&abi) };
        assert!(ctx.importance().is_none());
        let kh = ctx.tensor(TensorKind::Key).expect("Key available");
        assert_eq!(kh.shape().cols, 4);
        let mut out = [0.0f32; 4];
        // dequant_k is the default sugar over tensor(Key) — it fills the host key across the fn-ptr.
        ctx.dequant_k(0, 0, &mut out);
        assert_eq!(out, [1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn plan_arena_layerwide_round_trip() {
        let plan = KVCachePlan {
            keep: KeepSpec::LayerWide(vec![70, 71, 72]),
            merges: Vec::new(),
        };
        let abi = PlanArena::into_abi(plan.clone());
        assert_eq!(abi.keep_kind, 0);
        // SAFETY: read the valid arena leaked by into_abi until it is freed.
        let keep = unsafe { core::slice::from_raw_parts(abi.keep_ptr, abi.keep_len) };
        assert_eq!(keep, &[70usize, 71, 72]);
        assert_eq!(abi.merges_len, 0);
        assert!(abi.keep_outer_lens.is_null());
        unsafe { PlanArena::free(abi.owner) };
    }

    #[test]
    fn plan_arena_with_merges_round_trip() {
        let plan = KVCachePlan {
            keep: KeepSpec::LayerWide(vec![0, 1]),
            merges: vec![WeightedMerge {
                into: 0,
                into_weight: 0.6,
                from: vec![(5, 0.2), (6, 0.2)],
                apply_to: MergeAxis::ValueOnly,
            }],
        };
        let abi = PlanArena::into_abi(plan.clone());
        // Host-side reconstruct mirror (C2 performs the identical logic).
        let keep = unsafe { core::slice::from_raw_parts(abi.keep_ptr, abi.keep_len) }.to_vec();
        let merges_abi = unsafe { core::slice::from_raw_parts(abi.merges_ptr, abi.merges_len) };
        let merges: Vec<WeightedMerge> = merges_abi
            .iter()
            .map(|m| {
                let from = unsafe { core::slice::from_raw_parts(m.from_ptr, m.from_len) };
                WeightedMerge {
                    into: m.into,
                    into_weight: m.into_weight,
                    from: from.iter().map(|p| (p.pos, p.weight)).collect(),
                    apply_to: MergeAxis::from_u32(m.apply_to),
                }
            })
            .collect();
        let reconstructed = KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
            merges,
        };
        assert_eq!(reconstructed, plan);
        unsafe { PlanArena::free(abi.owner) };
    }

    #[test]
    fn plan_arena_perhead_flattens_with_outer_lens() {
        let plan = KVCachePlan {
            keep: KeepSpec::PerHead(vec![vec![1, 2], vec![3, 4, 5]]),
            merges: Vec::new(),
        };
        let abi = PlanArena::into_abi(plan);
        assert_eq!(abi.keep_kind, 1);
        let keep = unsafe { core::slice::from_raw_parts(abi.keep_ptr, abi.keep_len) };
        assert_eq!(keep, &[1usize, 2, 3, 4, 5]); // all heads concatenated
        let lens =
            unsafe { core::slice::from_raw_parts(abi.keep_outer_lens, abi.keep_outer_count) };
        assert_eq!(lens, &[2usize, 3]);
        unsafe { PlanArena::free(abi.owner) };
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

    // Stub vtable for the stage envelope round-trip (never called — only marshalling is verified).
    unsafe extern "C" fn st_make(_p: *const StageParams) -> *mut c_void {
        ::core::ptr::null_mut()
    }
    unsafe extern "C" fn st_plan(_h: *mut c_void, _c: *const StageCtxAbi, _o: *mut PlanAbi) -> i32 {
        KV_PLAN_NOOP
    }
    unsafe extern "C" fn st_plan_free(_o: *mut c_void) {}
    unsafe extern "C" fn st_drop(_h: *mut c_void) {}

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

    static STAGE_EXPORT_VTS: [PluginVTableAbi; 2] = [
        PluginVTableAbi {
            name: b"rt_st_a\0".as_ptr() as *const c_char,
            make: st_make,
            plan: st_plan,
            plan_free: st_plan_free,
            drop: st_drop,
        },
        PluginVTableAbi {
            name: b"rt_st_b\0".as_ptr() as *const c_char,
            make: st_make,
            plan: st_plan,
            plan_free: st_plan_free,
            drop: st_drop,
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
    extern "C" fn mk_stage_export() -> StageExportAbi {
        StageExportAbi {
            abi_version: KV_STAGE_ABI_VERSION,
            count: STAGE_EXPORT_VTS.len(),
            vtables: STAGE_EXPORT_VTS.as_ptr(),
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

    /// StageExportAbi isomorphic round-trip (symmetric across both axes).
    #[test]
    fn stage_export_abi_by_value_sret_round_trip() {
        let env = mk_stage_export();
        assert_eq!(env.abi_version, KV_STAGE_ABI_VERSION);
        assert_eq!(env.count, 2);
        for (i, expect) in ["rt_st_a", "rt_st_b"].iter().enumerate() {
            // SAFETY: vtables is the base of STAGE_EXPORT_VTS (a 'static array), i < count.
            let vt = unsafe { &*env.vtables.add(i) };
            let name = unsafe { core::ffi::CStr::from_ptr(vt.name) }
                .to_str()
                .unwrap();
            assert_eq!(&name, expect);
        }
    }

    // Multiple invocations — register_kv_format! twice in one crate (impossible under the v1 single-symbol ABI due to the __REGISTER_KV_FORMAT_REG
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
