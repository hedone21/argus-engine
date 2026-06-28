//! HYBRID v3 — the declarative coordinate map for KV techniques.
//!
//! The imperative `CacheHandle` / `KVMutationStage` / producer traits carry a technique's *behavior*;
//! this module carries its *coordinates*. A [`KvTechniqueDescriptor`] is pure data: the axis cell a
//! technique occupies (stage ⊥ format ⊥ hardware, plus the observer score/read producer axes), the
//! phase it acts at, and the signal edges it reads / produces. Together [`KV_TECHNIQUE_DESCRIPTORS`]
//! is a static, CI-validatable map of "what technique sits where, consuming/feeding which signals".
//!
//! It is a CENTRAL engine table (not a per-crate `linkme` slice) so the built-in set is exactly the
//! same regardless of which feature-gated crates (caote / rkv) are force-linked — the coordinate map
//! is a fixed matrix, not a build-dependent one.
//!
//! What is VALIDATED today: CARDINALITY (exactly 7, const-assert), OCCUPANCY ([`validate_occupancy`] —
//! no technique reads a signal nothing produces, where the engine supplies [`ENGINE_INTRINSIC`]), and
//! name↔registry resolution (the tests). `axis` and `phase` are DESCRIPTIVE coordinates beyond the
//! name-resolution cross-check — not otherwise validated. Only the stage / score / read cells are
//! populated; no built-in occupies the format / hardware axes (those producers live in their own
//! registries — `KV_FORMAT_POLICIES`, the offload store — and folding them into this map is a
//! follow-up).

use argus_extension_api::{MutationPhase, TensorKind};

/// A set of signal kinds (a thin alias for the coordinate map's read/produce edges).
pub type SignalSet = &'static [TensorKind];

/// Which orthogonal axis a technique occupies (the coordinate-map cell). Mirrors CONTEXT.md's
/// stage ⊥ format ⊥ hardware, plus the observer score / read producer axes. The full taxonomy is
/// listed for completeness; only `Stage` / `Score` / `Read` are occupied by a built-in descriptor
/// today (`Format` / `Hardware` have no built-in entry — their producers live in separate registries).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvAxis {
    /// Resident-token adjustment (eviction / merge): sliding / streaming / h2o / h2o_plus / d2o.
    Stage,
    /// Storage precision / layout (the format axis). No built-in descriptor yet (see `KV_FORMAT_POLICIES`).
    Format,
    /// Compute / residency location (the hardware axis). No built-in descriptor yet (see the offload store).
    Hardware,
    /// Forward-time attention-score producer (observer/score axis): attn_score.
    Score,
    /// Read-plan producer (sparse/selective read seam): quest.
    Read,
}

/// Declarative coordinate-map entry for one KV technique. Pure data — the behavior lives in the
/// technique's `KVMutationStage` / score-or-read producer impl.
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

/// Signals the engine MAY supply directly (not via a technique). `Key`/`Value` are always resident;
/// `AttnWeights` / `QueryStats` / `PrefillAttention` / `Query` are armed on demand by the forward pass
/// (a score-active step, a faithful-read step, prefill-end, …). The OCCUPANCY invariant treats these
/// as available producers, so a built-in must list one in `reads` only when a built-in actually arms
/// it — otherwise the invariant would pass vacuously (see `quest`, which omits the never-armed
/// `QueryStats`).
pub const ENGINE_INTRINSIC: &[TensorKind] = &[
    TensorKind::Key,
    TensorKind::Value,
    TensorKind::AttnWeights,
    TensorKind::QueryStats,
    TensorKind::PrefillAttention,
    TensorKind::Query,
];

/// The central coordinate map — the built-in KV techniques. `phase` is the consumption phase
/// (`KvMutate`, the per-step KV-mutation slot). The `reads`/`produces` SSOT differs by axis: the 5
/// Stage techniques mirror their `MutationStageReg.caps.reads`; `attn_score` mirrors
/// `ScoreProducerReg.produces`; `quest` mirrors `KVReadStageReg` (its runtime-consumed query signals,
/// not `StageCaps` — a read stage has none). `produces` is empty for the stage techniques (they
/// mutate, they do not feed a signal).
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
    // quest produces a sparse read plan from query criticality; it reads the raw K plus the current
    // query (both engine-intrinsic). QueryStats is NOT listed: quest registers wants_query_stats=false,
    // so no built-in arms the QueryStats accumulator — listing it would let OCCUPANCY pass vacuously on
    // a signal nothing produces. (quest falls back to a K-magnitude proxy when Query is absent.)
    KvTechniqueDescriptor {
        name: "quest",
        axis: KvAxis::Read,
        phase: MutationPhase::KvMutate,
        reads: &[TensorKind::Key, TensorKind::Query],
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The coordinate map is exactly the 7-technique matrix, regardless of feature-gated crates.
    /// Mutation-proof: adding/removing a descriptor changes this set (and breaks the CARDINALITY
    /// const-assert).
    #[test]
    fn descriptor_names_match_the_seven_technique_matrix() {
        let mut got = descriptor_names();
        got.sort_unstable();
        let mut want = vec![
            "attn_score",
            "d2o",
            "h2o",
            "h2o_plus",
            "quest",
            "sliding",
            "streaming",
        ];
        want.sort_unstable();
        assert_eq!(got, want);
    }

    /// Every descriptor name resolves to a LIVE registration in the registry for its axis (Stage →
    /// find_mutation_stage, Score → find_score_producer, Read → find_read_stage). This closes the
    /// descriptor↔registry loop: a registry rename or a descriptor typo (e.g. "h2o-plus" vs the
    /// registered "h2o_plus") fails here instead of silently passing the hardcoded-name test above.
    /// Mutation-proof: misspelling any descriptor name flips the matching assert to None.
    #[test]
    fn descriptor_names_resolve_to_live_registrations() {
        use argus_extension_api::{find_mutation_stage, find_read_stage, find_score_producer};
        for d in KV_TECHNIQUE_DESCRIPTORS {
            let resolved = match d.axis {
                KvAxis::Stage => find_mutation_stage(d.name).is_some(),
                KvAxis::Score => find_score_producer(d.name).is_some(),
                KvAxis::Read => find_read_stage(d.name).is_some(),
                // No name-keyed registry for the format/hardware axes (no built-in occupies them).
                KvAxis::Format | KvAxis::Hardware => true,
            };
            assert!(
                resolved,
                "descriptor '{}' (axis {:?}) does not resolve to a registered technique",
                d.name, d.axis
            );
        }
    }

    /// The live map satisfies OCCUPANCY (every read produced or engine-intrinsic) and the boot
    /// self-test does not panic.
    #[test]
    fn live_map_satisfies_occupancy() {
        assert_eq!(
            validate_occupancy(KV_TECHNIQUE_DESCRIPTORS, ENGINE_INTRINSIC),
            Ok(())
        );
        descriptor_self_test();
    }

    /// OCCUPANCY rejects an orphan read (a signal neither produced nor intrinsic), and accepts it once
    /// a producer is present. Mutation-proof / non-tautological: a validator that skipped the read
    /// check would make the first assert `Ok` (failing it), and one that ignored `produces` would make
    /// the second `Err` (failing that).
    #[test]
    fn orphan_read_is_rejected_then_satisfied_by_a_producer() {
        // `ghost` reads Scores; the intrinsic set here is {Key} only and nothing produces Scores.
        let orphaned = [KvTechniqueDescriptor {
            name: "ghost",
            axis: KvAxis::Stage,
            phase: MutationPhase::KvMutate,
            reads: &[TensorKind::Scores],
            produces: &[],
        }];
        assert!(validate_occupancy(&orphaned, &[TensorKind::Key]).is_err());

        // Add a producer of Scores → the same read is now satisfied.
        let with_producer = [
            KvTechniqueDescriptor {
                name: "ghost",
                axis: KvAxis::Stage,
                phase: MutationPhase::KvMutate,
                reads: &[TensorKind::Scores],
                produces: &[],
            },
            KvTechniqueDescriptor {
                name: "source",
                axis: KvAxis::Score,
                phase: MutationPhase::KvMutate,
                reads: &[],
                produces: &[TensorKind::Scores],
            },
        ];
        assert_eq!(
            validate_occupancy(&with_producer, &[TensorKind::Key]),
            Ok(())
        );
    }

    /// Pass2-TR1/DM1: each descriptor's reads/produces actually MATCHES the live registry it claims to
    /// mirror (the SSOT prose at the top of this module), so a future registry caps edit cannot silently
    /// desync the coordinate map while every other descriptor test stays green. Stage axis ->
    /// `MutationStageReg.caps.reads`; Score axis -> `ScoreProducerReg.produces`; the Read axis (quest,
    /// whose `KVReadStageReg` carries no reads field) is pinned directly to `[Key, Query]` — guarding
    /// the F3 accuracy fix (quest reads `Query`, NOT the never-armed `QueryStats`). Mutation-proof:
    /// reverting any descriptor read/produce edge — or a registry `caps.reads` — fails this assert.
    #[test]
    fn descriptor_reads_produces_match_live_registry() {
        use argus_extension_api::{find_mutation_stage, find_score_producer};
        for d in KV_TECHNIQUE_DESCRIPTORS {
            match d.axis {
                KvAxis::Stage => {
                    let reg = find_mutation_stage(d.name).expect("stage registered");
                    assert_eq!(
                        reg.caps.reads, d.reads,
                        "descriptor '{}' reads drifted from registry caps.reads",
                        d.name
                    );
                }
                KvAxis::Score => {
                    let reg = find_score_producer(d.name).expect("score producer registered");
                    assert_eq!(
                        reg.produces, d.produces,
                        "descriptor '{}' produces drifted from registry produces",
                        d.name
                    );
                }
                KvAxis::Read => {
                    assert_eq!(
                        d.reads,
                        [TensorKind::Key, TensorKind::Query].as_slice(),
                        "quest descriptor reads drifted (F3: must be [Key, Query], not QueryStats)"
                    );
                }
                // No name-keyed registry for the format/hardware axes (no built-in occupies them).
                KvAxis::Format | KvAxis::Hardware => {}
            }
        }
    }
}
