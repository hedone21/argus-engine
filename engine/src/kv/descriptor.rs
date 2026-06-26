//! HYBRID v3 — the declarative coordinate map for KV techniques.
//!
//! The imperative `CacheHandle` / `KVCacheStage` / producer traits carry a technique's *behavior*;
//! this module carries its *coordinates*. A [`KvTechniqueDescriptor`] is pure data: the axis cell a
//! technique occupies (stage ⊥ format ⊥ hardware, plus the observer score/read producer axes), the
//! phase it acts at, and the signal edges it reads / produces. Together [`KV_TECHNIQUE_DESCRIPTORS`]
//! is a static, CI-validatable map of "what technique sits where, consuming/feeding which signals".
//!
//! It is a CENTRAL engine table (not a per-crate `linkme` slice) so the built-in set is exactly the
//! same regardless of which feature-gated crates (caote / rkv) are force-linked — the coordinate map
//! is a fixed matrix, not a build-dependent one. The OCCUPANCY invariant ([`validate_occupancy`])
//! checks that no technique reads a signal nothing produces (no orphan read), where the engine forward
//! pass produces the [`ENGINE_INTRINSIC`] signals.

use argus_extension_api::{MutationPhase, TensorKind};

/// A set of signal kinds (a thin alias for the coordinate map's read/produce edges).
pub type SignalSet = &'static [TensorKind];

/// Which orthogonal axis a technique occupies (the coordinate-map cell). Mirrors CONTEXT.md's
/// stage ⊥ format ⊥ hardware, plus the observer score / read producer axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvAxis {
    /// Resident-token adjustment (eviction / merge): sliding / streaming / h2o / h2o_plus / d2o.
    Stage,
    /// Storage precision / layout (the format axis).
    Format,
    /// Compute / residency location (the hardware axis).
    Hardware,
    /// Forward-time attention-score producer (observer/score axis): attn_score.
    Score,
    /// Read-plan producer (sparse/selective read seam): quest.
    Read,
}

/// Declarative coordinate-map entry for one KV technique. Pure data — the behavior lives in the
/// technique's `KVCacheStage` / `KVMutationStage` / score-or-read producer impl.
#[derive(Clone, Copy, Debug)]
pub struct KvTechniqueDescriptor {
    /// Technique name (matches the registry name; unique within the map).
    pub name: &'static str,
    /// The axis cell this technique occupies.
    pub axis: KvAxis,
    /// The phase its effect is consumed at (a coordinate, not behavior; score/read producers feed the
    /// `KvMutate` consumers).
    pub phase: MutationPhase,
    /// Signals this technique reads.
    pub reads: SignalSet,
    /// Signals this technique produces for others to read.
    pub produces: SignalSet,
}

/// Signals the engine forward pass produces intrinsically (not by any technique): raw K/V, the
/// previous-step attention weights, the Q running statistics, the prefill attention slice, and the
/// raw current-Q. The OCCUPANCY invariant treats these as always-available producers.
pub const ENGINE_INTRINSIC: &[TensorKind] = &[
    TensorKind::Key,
    TensorKind::Value,
    TensorKind::AttnWeights,
    TensorKind::QueryStats,
    TensorKind::PrefillAttention,
    TensorKind::Query,
];

/// The central coordinate map — the built-in KV techniques. `phase` is the consumption phase
/// (`KvMutate`, the per-step KV-mutation slot). `reads` mirror each technique's `StageCaps.reads`
/// (the SSOT); `produces` is empty for stage techniques (they mutate, they do not feed a signal).
pub static KV_TECHNIQUE_DESCRIPTORS: &[KvTechniqueDescriptor] = &[
    // ── stage axis (resident-token adjustment) ──
    KvTechniqueDescriptor {
        name: "sliding",
        axis: KvAxis::Stage,
        phase: MutationPhase::KvMutate,
        reads: &[],
        produces: &[],
    },
    KvTechniqueDescriptor {
        name: "streaming",
        axis: KvAxis::Stage,
        phase: MutationPhase::KvMutate,
        reads: &[],
        produces: &[],
    },
    KvTechniqueDescriptor {
        name: "h2o",
        axis: KvAxis::Stage,
        phase: MutationPhase::KvMutate,
        reads: &[TensorKind::Scores],
        produces: &[],
    },
    KvTechniqueDescriptor {
        name: "h2o_plus",
        axis: KvAxis::Stage,
        phase: MutationPhase::KvMutate,
        reads: &[TensorKind::Scores],
        produces: &[],
    },
    KvTechniqueDescriptor {
        name: "d2o",
        axis: KvAxis::Stage,
        phase: MutationPhase::KvMutate,
        reads: &[TensorKind::Scores, TensorKind::Key],
        produces: &[],
    },
    // ── observer/score + read producer axes ──
    // attn_score accumulates the forward-internal attention into the per-token Scores signal that the
    // score-based stage techniques (h2o / h2o_plus / d2o) read — the producer that closes their reads.
    KvTechniqueDescriptor {
        name: "attn_score",
        axis: KvAxis::Score,
        phase: MutationPhase::KvMutate,
        reads: &[],
        produces: &[TensorKind::Scores],
    },
    // quest produces a sparse read plan from query criticality; it reads the raw K plus the current /
    // running query (all engine-intrinsic signals).
    KvTechniqueDescriptor {
        name: "quest",
        axis: KvAxis::Read,
        phase: MutationPhase::KvMutate,
        reads: &[TensorKind::Key, TensorKind::Query, TensorKind::QueryStats],
        produces: &[],
    },
];

/// All descriptor names (for the matrix-invariance self-test / diagnostics).
pub fn descriptor_names() -> Vec<&'static str> {
    KV_TECHNIQUE_DESCRIPTORS.iter().map(|d| d.name).collect()
}

/// OCCUPANCY invariant: every descriptor's `reads` ⊆ (∪ all `produces`) ∪ `intrinsic` — no technique
/// reads a signal that nothing produces and that the engine does not supply intrinsically (no orphan
/// read). Returns the first orphan as an `Err` (the technique + the unsatisfied signal).
pub fn validate_occupancy(
    descriptors: &[KvTechniqueDescriptor],
    intrinsic: &[TensorKind],
) -> Result<(), String> {
    let mut available: Vec<TensorKind> = intrinsic.to_vec();
    for d in descriptors {
        for &p in d.produces {
            if !available.contains(&p) {
                available.push(p);
            }
        }
    }
    for d in descriptors {
        for &r in d.reads {
            if !available.contains(&r) {
                return Err(format!(
                    "technique '{}' reads {r:?} which no descriptor produces and is not \
                     engine-intrinsic (orphan read)",
                    d.name
                ));
            }
        }
    }
    Ok(())
}

/// CARDINALITY invariant: the built-in coordinate map is exactly 7 techniques, regardless of which
/// feature-gated crates are linked (a fixed matrix). A compile-time assert — adding/removing a
/// descriptor without updating this fails the build.
const _: () = assert!(KV_TECHNIQUE_DESCRIPTORS.len() == 7);

/// Boot self-test: the live coordinate map satisfies OCCUPANCY over the engine-intrinsic producer
/// set. Panics with the offending orphan on violation. Callable at engine startup; exercised by the
/// invariant tests.
pub fn descriptor_self_test() {
    if let Err(e) = validate_occupancy(KV_TECHNIQUE_DESCRIPTORS, ENGINE_INTRINSIC) {
        panic!("KV technique coordinate-map OCCUPANCY violation: {e}");
    }
}
