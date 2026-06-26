//! `EvictTiming` — *when* eval-LL's KV eviction fires and *which* importance
//! signal drives it.
//!
//! This is the one axis the IMP-1/IMP-2 dumps cannot otherwise vary: whether the
//! eviction decision is made **with** the query in hand (today's post-question
//! probe) or **without** it (importance accumulated over the context during
//! prefill). It is deliberately orthogonal to the eviction *policy*: the policy
//! (h2o / d2o / sliding …) always ranks on whatever importance is in the
//! accumulator; this enum only decides where that importance comes from and when
//! the eviction fires. The eval loop owns timing; the hook owns policy.
//!
//! Rather than hard-code three named modes as scattered special cases, each mode
//! is expressed as a composition of two independent booleans —
//! [`runs_query_probe`](EvictTiming::runs_query_probe) and
//! [`accumulates_context_scores`](EvictTiming::accumulates_context_scores) — so a
//! future timing (e.g. per-chunk *streaming* eviction) is one enum variant plus
//! its two booleans, not a loop rewrite.
//!
//! `INV-147`: the default ([`PostPrefillProbe`](EvictTiming::PostPrefillProbe)) is
//! byte-for-byte today's behavior — full batch prefill, post-question probe, one
//! eviction at `post_prefill`.

/// When eval-LL eviction fires and which importance drives it. CLI: `--evict-timing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictTiming {
    /// **Today's behavior (default).** Full batch prefill accumulates no per-token
    /// scores, so a probe re-feeds the *last prompt token* as a decode step to
    /// populate the accumulator, then a single eviction fires at `post_prefill`.
    /// That last token sits *after* the question in the standard layout, so the
    /// importance the policy ranks on is **query-informed**.
    #[default]
    PostPrefillProbe,

    /// **Query-agnostic, end-of-prefill.** Drive prefill token-by-token so per-step
    /// importance accumulates **over the context**, suppress the post-question
    /// probe, then evict once at `post_prefill` on that context-only importance.
    /// One eviction event, decided without query knowledge.
    PrefillEnd,
}

impl EvictTiming {
    /// CLI token for [`PostPrefillProbe`](Self::PostPrefillProbe).
    pub const POST_PREFILL_PROBE: &'static str = "post_prefill_probe";
    /// CLI token for [`PrefillEnd`](Self::PrefillEnd).
    pub const PREFILL_END: &'static str = "prefill_end";

    /// All accepted CLI tokens, in declaration order. Single source of truth shared
    /// by the clap `value_parser` and [`from_cli`](Self::from_cli); the CLI never
    /// drifts from the parser.
    pub const CLI_VALUES: [&'static str; 2] = [Self::POST_PREFILL_PROBE, Self::PREFILL_END];

    /// Parse a CLI token, `None` on an unknown value (clap rejects those upstream;
    /// this stays total so the accessor can `expect` on already-validated input).
    pub fn from_cli(s: &str) -> Option<Self> {
        match s {
            Self::POST_PREFILL_PROBE => Some(Self::PostPrefillProbe),
            Self::PREFILL_END => Some(Self::PrefillEnd),
            _ => None,
        }
    }

    /// The CLI token for this mode (inverse of [`from_cli`](Self::from_cli)).
    pub fn as_cli(self) -> &'static str {
        match self {
            Self::PostPrefillProbe => Self::POST_PREFILL_PROBE,
            Self::PrefillEnd => Self::PREFILL_END,
        }
    }

    /// Whether the loop runs the post-question score probe. True only for
    /// [`PostPrefillProbe`](Self::PostPrefillProbe): the probe re-feeds the last
    /// prompt token, making the importance query-informed. Query-agnostic modes
    /// suppress it.
    pub fn runs_query_probe(self) -> bool {
        matches!(self, Self::PostPrefillProbe)
    }

    /// Whether prefill must run token-by-token to accumulate per-step **context**
    /// importance (so the eviction can rank query-agnostically). True only for
    /// [`PrefillEnd`](Self::PrefillEnd); the default leaves prefill batched.
    pub fn accumulates_context_scores(self) -> bool {
        matches!(self, Self::PrefillEnd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_post_prefill_probe() {
        // INV-147: the absent flag must resolve to today's behavior.
        assert_eq!(EvictTiming::default(), EvictTiming::PostPrefillProbe);
    }

    #[test]
    fn cli_round_trips_for_every_value() {
        for token in EvictTiming::CLI_VALUES {
            let parsed = EvictTiming::from_cli(token).expect("known token parses");
            assert_eq!(parsed.as_cli(), token, "as_cli is the inverse of from_cli");
        }
        // The default's token is the documented INV-147 default.
        assert_eq!(
            EvictTiming::default().as_cli(),
            EvictTiming::POST_PREFILL_PROBE
        );
    }

    #[test]
    fn unknown_token_is_rejected() {
        assert_eq!(EvictTiming::from_cli("prefill_streaming"), None);
        assert_eq!(EvictTiming::from_cli(""), None);
        assert_eq!(EvictTiming::from_cli("PostPrefillProbe"), None);
    }

    #[test]
    fn the_two_behaviors_are_orthogonal_and_mode_specific() {
        // The default is the only mode that probes; it never accumulates context.
        assert!(EvictTiming::PostPrefillProbe.runs_query_probe());
        assert!(!EvictTiming::PostPrefillProbe.accumulates_context_scores());

        // prefill_end is the mirror image: no probe, accumulate context instead.
        assert!(!EvictTiming::PrefillEnd.runs_query_probe());
        assert!(EvictTiming::PrefillEnd.accumulates_context_scores());

        // Exactly one of the two importance sources is active per mode — never both,
        // never neither — which is what keeps every mode well-defined.
        for m in [EvictTiming::PostPrefillProbe, EvictTiming::PrefillEnd] {
            assert_ne!(
                m.runs_query_probe(),
                m.accumulates_context_scores(),
                "{m:?}: probe and context-accumulation are mutually exclusive"
            );
        }
    }
}
