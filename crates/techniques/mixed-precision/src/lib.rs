//! Per-layer mixed-precision KV format policy — the W-ALLOC executable (N-way mixed precision).
//!
//! A self-registering [`KVFormatPolicy`] (the `layer-importance` / `quest` / `h2o` precedent):
//! depends only on `argus-extension-api` + `linkme`. Registers as `"mixed_precision"` via
//! `#[distributed_slice(KV_FORMAT_POLICIES)]`; the engine force-links it (`use mixed_precision as _;`)
//! and resolves it through `find_format_policy`. `--kv-format mixed_precision` then makes the engine
//! allocate each transformer layer's KV cache in the per-layer format this policy assigns.
//!
//! Per-layer assignment is read from the `ARGUS_KV_MIXED` environment variable — the v0 happy-path
//! CLI deliberately takes no new flags, so configuration rides an env var. The spec is a
//! comma-separated list of `format:count` segments applied front-to-back:
//!
//! ```text
//! ARGUS_KV_MIXED="f16:8,q4_0:24"   # layers 0..8 stored f16, layers 8..32 stored q4_0
//! ```
//!
//! Layers beyond the spec keep the engine default ([`KVFormatPolicy::assign`] returns `None`). An
//! unset/empty/unparseable spec yields no segments → every layer keeps the default (a safe no-op).
//! `format` names are the engine's registered KV formats (`f16` / `q4_0` / `f32` / a `.so` format);
//! the canonical mixed-precision pair is `f16` (precision-sensitive layers) + `q4_0` (tolerant ones).

use argus_extension_api::{
    FormatId, KV_FORMAT_POLICIES, KVFormatPlan, KVFormatPolicy, KVFormatPolicyReg, StageCtx,
    StageParams,
};
use linkme::distributed_slice;

/// One `format:count` segment of the `ARGUS_KV_MIXED` spec.
struct Segment {
    format: String,
    count: usize,
}

/// Per-layer mixed-precision policy. Holds the parsed `ARGUS_KV_MIXED` spec; [`Self::format_for`]
/// maps a layer index to its segment's format (cumulative, front-to-back).
struct MixedPrecisionPolicy {
    segments: Vec<Segment>,
}

impl MixedPrecisionPolicy {
    /// Parse `ARGUS_KV_MIXED` (`"fmt:count,fmt:count,..."`). Any empty/malformed/zero-count segment
    /// is skipped; an unset var yields no segments (the policy then no-ops, keeping the default).
    fn from_env() -> Self {
        let spec = std::env::var("ARGUS_KV_MIXED").unwrap_or_default();
        Self {
            segments: Self::parse(&spec),
        }
    }

    /// Pure parser (testable without the environment).
    fn parse(spec: &str) -> Vec<Segment> {
        spec.split(',')
            .filter_map(|seg| {
                let (fmt, count) = seg.trim().split_once(':')?;
                let count: usize = count.trim().parse().ok()?;
                let fmt = fmt.trim();
                if fmt.is_empty() || count == 0 {
                    return None;
                }
                Some(Segment {
                    format: fmt.to_string(),
                    count,
                })
            })
            .collect()
    }

    /// The format assigned to `layer`, walking the cumulative segment counts. `None` once `layer`
    /// runs past the last segment (engine default kept for the tail).
    fn format_for(&self, layer: usize) -> Option<&str> {
        let mut acc = 0usize;
        for seg in &self.segments {
            acc += seg.count;
            if layer < acc {
                return Some(&seg.format);
            }
        }
        None
    }
}

impl KVFormatPolicy for MixedPrecisionPolicy {
    fn name(&self) -> &str {
        "mixed_precision"
    }

    /// `Some(plan{base})` with the layer's assigned format, or `None` (keep the engine default).
    /// Emits no `overrides` — this is a uniform-per-layer base assignment (the only thing the
    /// construction-time allocator can honor; the engine rejects override-bearing plans).
    fn assign(&self, ctx: &dyn StageCtx) -> Option<KVFormatPlan> {
        let fmt = self.format_for(ctx.layer_idx())?;
        Some(KVFormatPlan {
            base: FormatId(fmt.to_string()),
            overrides: Vec::new(),
        })
    }
}

/// Registration — the engine resolves this via `find_format_policy("mixed_precision")` and force-links
/// the crate (`use mixed_precision as _;`). Reads no per-token signals (layer-index driven), so
/// `reads` is empty.
#[distributed_slice(KV_FORMAT_POLICIES)]
static MIXED_PRECISION: KVFormatPolicyReg = KVFormatPolicyReg {
    name: "mixed_precision",
    make: |_p: StageParams| Box::new(MixedPrecisionPolicy::from_env()),
    reads: &[],
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{find_format_policy, registered_format_policy_names};

    #[test]
    fn parses_segments_and_skips_malformed() {
        let segs = MixedPrecisionPolicy::parse("f16:8, q4_0:24 ,,bad,zero:0,nocolon");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].format, "f16");
        assert_eq!(segs[0].count, 8);
        assert_eq!(segs[1].format, "q4_0");
        assert_eq!(segs[1].count, 24);
    }

    #[test]
    fn format_for_walks_cumulative_segments() {
        let p = MixedPrecisionPolicy {
            segments: MixedPrecisionPolicy::parse("f16:8,q4_0:24"),
        };
        assert_eq!(p.format_for(0), Some("f16"));
        assert_eq!(p.format_for(7), Some("f16"));
        assert_eq!(p.format_for(8), Some("q4_0"));
        assert_eq!(p.format_for(31), Some("q4_0"));
        // beyond the 32-layer spec → None (engine default kept)
        assert_eq!(p.format_for(32), None);
    }

    #[test]
    fn empty_spec_is_noop() {
        let p = MixedPrecisionPolicy {
            segments: MixedPrecisionPolicy::parse(""),
        };
        assert_eq!(p.format_for(0), None);
    }

    /// Minimal `StageCtx` exposing only the layer index (mirrors the engine's construction-time ctx).
    struct LayerCtx(usize);
    impl StageCtx for LayerCtx {
        fn current_pos(&self) -> usize {
            0
        }
        fn target_len(&self) -> usize {
            0
        }
        fn layer_idx(&self) -> usize {
            self.0
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
        fn tensor(
            &self,
            _kind: argus_extension_api::TensorKind,
        ) -> Option<&dyn argus_extension_api::TensorHandle> {
            None
        }
    }

    #[test]
    fn assign_returns_per_layer_base_format() {
        let p = MixedPrecisionPolicy {
            segments: MixedPrecisionPolicy::parse("f16:2,q4_0:2"),
        };
        assert_eq!(p.assign(&LayerCtx(0)).unwrap().base, FormatId("f16".into()));
        assert_eq!(
            p.assign(&LayerCtx(2)).unwrap().base,
            FormatId("q4_0".into())
        );
        assert!(p.assign(&LayerCtx(4)).is_none());
        // never emits overrides (uniform-per-layer base only)
        assert!(p.assign(&LayerCtx(0)).unwrap().overrides.is_empty());
    }

    #[test]
    fn registered_in_slice() {
        assert!(find_format_policy("mixed_precision").is_some());
        assert!(registered_format_policy_names().contains(&"mixed_precision"));
    }
}
