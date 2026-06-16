//! Engine-internal eviction policies that are **not yet** out-of-tree plugin crates, plus their
//! capability table — the non-plugin analogue of the registry's `make_stage` / `stage_caps`.
//!
//! Currently the only resident is **h2o_plus** (per-head GQA eviction). It cannot be expressed as a
//! `KVCacheStage` plugin until stage ⑤ lands (the per-head plan executor + the per-head score
//! source wired through `StageBackedPolicy`). Until then it stays an in-engine `EvictionPolicy`, and
//! this module is the **one place** that knows the string `"h2o_plus"` — the CLI/chat/eval/bench
//! build sites call [`engine_internal_policy`] / consult [`engine_internal_caps`] generically, so
//! they stay free of any policy-name knowledge.
//!
//! **This whole module is deleted once h2o_plus extracts to `crates/techniques/h2o-plus`** (stage ⑤):
//! it then registers a `StageCaps` of its own and resolves through `make_stage` like every other
//! technique, and the engine names it nowhere.

use super::{EvictionPolicy, H2OPlusPolicy};
use argus_extension_api::StageCaps;

/// [`StageCaps`] for an engine-internal policy, by name — the `stage_caps` analogue for policies
/// that have no `KVCacheStageReg` yet. `None` if `name` is not engine-internal (the caller then
/// consults the plugin registry's [`argus_extension_api::stage_caps`]). Single-sourced with
/// [`engine_internal_policy`] so the two never disagree.
pub(crate) fn engine_internal_caps(name: &str) -> Option<StageCaps> {
    match name {
        // h2o_plus is score-based (per-head heavy hitters) and protects 4 attention sinks — the same
        // caps the `h2o`/`d2o` plugins declare in their registry entries.
        "h2o_plus" => Some(StageCaps {
            is_score_based: true,
            default_protected_prefix: 4,
        }),
        _ => None,
    }
}

/// Build an engine-internal [`EvictionPolicy`] by name — the `make_stage` analogue for policies that
/// have no plugin crate yet. `None` if `name` is not engine-internal (the caller falls back to
/// `make_stage_with_args`). Keeps the build sites from naming `"h2o_plus"` themselves.
pub(crate) fn engine_internal_policy(
    name: &str,
    keep_ratio: f32,
    protected_prefix: usize,
) -> Option<Box<dyn EvictionPolicy>> {
    match name {
        "h2o_plus" => Some(Box::new(H2OPlusPolicy::new(keep_ratio, protected_prefix))),
        _ => None,
    }
}
