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
//! Rather than hard-code named modes as scattered special cases, each mode is
//! expressed as a composition of independent booleans —
//! [`runs_query_probe`](EvictTiming::runs_query_probe),
//! [`accumulates_context_scores`](EvictTiming::accumulates_context_scores) and
//! [`evicts_on_overflow`](EvictTiming::evicts_on_overflow) — so a new timing is one
//! enum variant plus its booleans, not a loop rewrite. `prefill_streaming` is the
//! third mode the original two-boolean design anticipated ("one variant plus two
//! booleans"): it reuses `prefill_end`'s token-by-token context accumulation but
//! differs in *when/how often* it evicts — every overflow instead of once at the end.
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

    /// **Query-agnostic, evict-on-overflow (memory-bound faithful).** Drive prefill
    /// token-by-token like [`PrefillEnd`](Self::PrefillEnd), but cap the resident
    /// cache at a fixed budget `B`: whenever ingesting a token would push occupancy
    /// past `B`, evict down to a low-water mark on the importance accumulated so far
    /// (causal — only tokens up to now), then keep ingesting. The cache is never
    /// resident above `B` (+ at most one step's slack), so a far/early needle is
    /// dropped *before* later context and the question are even seen — the regime an
    /// on-device, memory-bound deployment actually runs. Multiple eviction events per
    /// question (vs. the single cut of the other two modes).
    PrefillStreaming,
}

impl EvictTiming {
    /// CLI token for [`PostPrefillProbe`](Self::PostPrefillProbe).
    pub const POST_PREFILL_PROBE: &'static str = "post_prefill_probe";
    /// CLI token for [`PrefillEnd`](Self::PrefillEnd).
    pub const PREFILL_END: &'static str = "prefill_end";
    /// CLI token for [`PrefillStreaming`](Self::PrefillStreaming).
    pub const PREFILL_STREAMING: &'static str = "prefill_streaming";

    /// All accepted CLI tokens, in declaration order. Single source of truth shared
    /// by the clap `value_parser` and [`from_cli`](Self::from_cli); the CLI never
    /// drifts from the parser.
    pub const CLI_VALUES: [&'static str; 3] = [
        Self::POST_PREFILL_PROBE,
        Self::PREFILL_END,
        Self::PREFILL_STREAMING,
    ];

    /// Parse a CLI token, `None` on an unknown value (clap rejects those upstream;
    /// this stays total so the accessor can `expect` on already-validated input).
    pub fn from_cli(s: &str) -> Option<Self> {
        match s {
            Self::POST_PREFILL_PROBE => Some(Self::PostPrefillProbe),
            Self::PREFILL_END => Some(Self::PrefillEnd),
            Self::PREFILL_STREAMING => Some(Self::PrefillStreaming),
            _ => None,
        }
    }

    /// The CLI token for this mode (inverse of [`from_cli`](Self::from_cli)).
    pub fn as_cli(self) -> &'static str {
        match self {
            Self::PostPrefillProbe => Self::POST_PREFILL_PROBE,
            Self::PrefillEnd => Self::PREFILL_END,
            Self::PrefillStreaming => Self::PREFILL_STREAMING,
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
    /// importance (so the eviction can rank query-agnostically). True for the
    /// query-agnostic modes [`PrefillEnd`](Self::PrefillEnd) and
    /// [`PrefillStreaming`](Self::PrefillStreaming); the default leaves prefill batched.
    pub fn accumulates_context_scores(self) -> bool {
        matches!(self, Self::PrefillEnd | Self::PrefillStreaming)
    }

    /// Whether eviction fires **per overflow during prefill** (cap the resident cache
    /// at the budget `B`) instead of once at `post_prefill`. True only for
    /// [`PrefillStreaming`](Self::PrefillStreaming). This is the axis that makes the
    /// resident set bounded and produces one dump record per eviction event; the other
    /// two modes evict at most once per question.
    pub fn evicts_on_overflow(self) -> bool {
        matches!(self, Self::PrefillStreaming)
    }

    /// Validate the KV-budget unit for this timing. Pure, so the budget-unit guard is
    /// testable without the eval runner.
    ///
    /// `prefill_streaming` requires an **absolute** budget `B` (`kv_budget > 0`,
    /// `ratio_mode == false`): a prompt-length ratio is ill-defined mid-prefill (the
    /// prompt length isn't known while ingesting), and the fixed-`B` memory-bound
    /// regime is the whole point. Returns the error message to bail with, or `None`
    /// when the combination is valid. Non-streaming timings never error here (their
    /// budget handling is unchanged — `INV-147`).
    pub fn budget_unit_error(self, kv_budget: usize, ratio_mode: bool) -> Option<&'static str> {
        if !self.evicts_on_overflow() {
            return None;
        }
        if ratio_mode {
            return Some(
                "--evict-timing prefill_streaming requires an absolute --kv-budget <N>; \
                 --kv-budget-ratio is ill-defined mid-prefill (prompt length is unknown while \
                 streaming) — drop --kv-budget-ratio and pass --kv-budget <N>",
            );
        }
        if kv_budget == 0 {
            return Some(
                "--evict-timing prefill_streaming requires an absolute --kv-budget <N> \
                 (the fixed resident cap B); none was set",
            );
        }
        None
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
    fn streaming_requires_absolute_budget() {
        use EvictTiming::*;
        // Non-streaming timings never error here — budget unit is unchanged (INV-147).
        for t in [PostPrefillProbe, PrefillEnd] {
            assert!(t.budget_unit_error(0, false).is_none());
            assert!(t.budget_unit_error(0, true).is_none());
            assert!(t.budget_unit_error(256, true).is_none());
        }
        // Streaming: ratio mode is rejected (ill-defined mid-prefill).
        assert!(PrefillStreaming.budget_unit_error(256, true).is_some());
        // Streaming: no absolute budget is rejected.
        assert!(PrefillStreaming.budget_unit_error(0, false).is_some());
        // Streaming: an absolute budget is accepted.
        assert!(PrefillStreaming.budget_unit_error(256, false).is_none());
    }

    #[test]
    fn prefill_streaming_now_parses() {
        // Variant b is no longer rejected — it round-trips like the other modes.
        let s = EvictTiming::from_cli("prefill_streaming").expect("now a known token");
        assert_eq!(s, EvictTiming::PrefillStreaming);
        assert_eq!(s.as_cli(), "prefill_streaming");
    }

    #[test]
    fn unknown_token_is_rejected() {
        assert_eq!(EvictTiming::from_cli(""), None);
        assert_eq!(EvictTiming::from_cli("PostPrefillProbe"), None);
        assert_eq!(EvictTiming::from_cli("streaming"), None);
    }

    #[test]
    fn the_behaviors_are_orthogonal_and_mode_specific() {
        // The default is the only mode that probes; it never accumulates context and
        // evicts at most once at post_prefill.
        assert!(EvictTiming::PostPrefillProbe.runs_query_probe());
        assert!(!EvictTiming::PostPrefillProbe.accumulates_context_scores());
        assert!(!EvictTiming::PostPrefillProbe.evicts_on_overflow());

        // prefill_end: no probe, accumulate context instead, still a single end cut.
        assert!(!EvictTiming::PrefillEnd.runs_query_probe());
        assert!(EvictTiming::PrefillEnd.accumulates_context_scores());
        assert!(!EvictTiming::PrefillEnd.evicts_on_overflow());

        // prefill_streaming: like prefill_end on probe/accumulation, but evicts on
        // every overflow — the one axis that differs.
        assert!(!EvictTiming::PrefillStreaming.runs_query_probe());
        assert!(EvictTiming::PrefillStreaming.accumulates_context_scores());
        assert!(EvictTiming::PrefillStreaming.evicts_on_overflow());

        // Exactly one importance source is active per mode (probe XOR context
        // accumulation) — never both, never neither — across every mode.
        for m in [
            EvictTiming::PostPrefillProbe,
            EvictTiming::PrefillEnd,
            EvictTiming::PrefillStreaming,
        ] {
            assert_ne!(
                m.runs_query_probe(),
                m.accumulates_context_scores(),
                "{m:?}: probe and context-accumulation are mutually exclusive"
            );
            // Overflow eviction only ever pairs with context accumulation (it is a
            // refinement of the prefill_end timing, never of the probe path).
            assert!(
                !m.evicts_on_overflow() || m.accumulates_context_scores(),
                "{m:?}: overflow eviction implies context accumulation"
            );
        }
    }
}
