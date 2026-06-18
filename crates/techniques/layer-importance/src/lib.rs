//! Layer-importance scorer technique crate — the per-layer importance formulas.
//!
//! Extracted from the engine core (the `sliding-window`/`h2o`/`d2o` precedent): depends only on
//! `argus-extension-api` + `linkme`, implements [`LayerScorer`], and registers under the names
//! `"mean_pool"` / `"shortgpt_bi"` via `#[distributed_slice(LAYER_SCORERS)]`. The engine force-links
//! it with a one-line `use layer_importance as _;` and resolves it via `find_layer_scorer(...)`.
//!
//! Both scorers are [`LayerScorerPhase::PerLayerStreaming`] and activation-only (they read only the
//! current layer's pooled / raw hidden states from [`LayerScorerCtx`], never weight subtensors).
//! The arithmetic is ported verbatim from the engine's former `ImportanceCollector::record_after`
//! inline math (the `cosine_similarity` helper + the two formulas), so the values it produces are
//! bit-identical to the pre-extraction engine.
//!
//! - **mean_pool**: `1 − cos(mean_pool(h_in), mean_pool(h_out))`, clamped to `≥ 0`. ARGUS baseline.
//! - **shortgpt_bi**: `1 − (1/T) Σ_t cos(h_in,t, h_out,t)`, clamped to `≥ 0` (ShortGPT BI,
//!   Men et al. 2024) — per-token cosine averaged over tokens with non-negligible input magnitude.

use argus_extension_api::{
    LAYER_SCORERS, LayerScorer, LayerScorerCtx, LayerScorerPhase, LayerScorerReg, StageArgs,
    StageParams,
};
use linkme::distributed_slice;

/// Cosine similarity between two float slices. Returns `0.0` if either vector has zero magnitude.
///
/// Verbatim port of the engine's former `qcf::layer_importance::cosine_similarity` — the single fused
/// dot/norm loop and the `1e-12` zero-norm guard are load-bearing for bit-identity with the old engine.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Mean-pool importance: `1 − cos(pooled_in, pooled_out)`, clamped to `≥ 0`. The ARGUS baseline.
struct MeanPoolScorer;

impl LayerScorer for MeanPoolScorer {
    fn name(&self) -> &str {
        "mean_pool"
    }
    fn phase(&self) -> LayerScorerPhase {
        LayerScorerPhase::PerLayerStreaming
    }
    fn reads_subtensors(&self) -> &'static [&'static str] {
        &[]
    }
    fn score(&self, _layer: usize, ctx: &dyn LayerScorerCtx) -> f32 {
        // Verbatim port of the engine's `imp_mean_pool` (the engine pools in/out for OPR and lends
        // the pooled vectors here, so this re-uses the exact same `[dim]` inputs → bit-identical).
        (1.0 - cosine_similarity(ctx.pooled_in(), ctx.pooled_out())).max(0.0)
    }
}

/// ShortGPT-BI importance: `1 − (1/T) Σ_t cos(h_in,t, h_out,t)`, clamped to `≥ 0`.
struct ShortGptBiScorer;

impl LayerScorer for ShortGptBiScorer {
    fn name(&self) -> &str {
        "shortgpt_bi"
    }
    fn phase(&self) -> LayerScorerPhase {
        LayerScorerPhase::PerLayerStreaming
    }
    fn reads_subtensors(&self) -> &'static [&'static str] {
        &[]
    }
    fn score(&self, _layer: usize, ctx: &dyn LayerScorerCtx) -> f32 {
        // Verbatim port of the engine's `imp_shortgpt_bi` Some(..) branch. The engine only invokes
        // this scorer when the raw before-snapshot is cached (3-way mode); when it is absent the
        // engine's selection falls back to mean-pool, so a missing `raw_in` here is defensive (0.0).
        let Some(raw_in) = ctx.raw_in() else {
            return 0.0;
        };
        let raw_out = ctx.raw_out();
        let (before_seq_len, before_dim) = ctx.raw_in_dims();
        let t = before_seq_len.min(ctx.seq_len());
        let d = before_dim.min(ctx.dim());
        if t == 0 || d == 0 {
            return 0.0;
        }
        let mut sum_cos = 0.0f32;
        let mut valid_t: u32 = 0;
        for pos in 0..t {
            let off = pos * d;
            let be = off + d;
            if be > raw_in.len() || be > raw_out.len() {
                break;
            }
            let before_tok = &raw_in[off..be];
            let after_tok = &raw_out[off..be];
            let mut bm = 0.0f32;
            for &b in before_tok.iter().take(d) {
                bm += b * b;
            }
            if bm > 1e-24 {
                sum_cos += cosine_similarity(before_tok, after_tok);
                valid_t += 1;
            }
        }
        if valid_t > 0 {
            (1.0 - sum_cos / valid_t as f32).max(0.0)
        } else {
            0.0
        }
    }
}

/// Registration — the engine resolves this via `find_layer_scorer("mean_pool")`. The default ARGUS
/// formula, so the engine force-links this crate non-optionally. Takes no params.
#[distributed_slice(LAYER_SCORERS)]
static MEAN_POOL: LayerScorerReg = LayerScorerReg {
    name: "mean_pool",
    phase: LayerScorerPhase::PerLayerStreaming,
    make: |_p: StageParams, _args: StageArgs<'_>| Box::new(MeanPoolScorer),
    reads_subtensors: &[],
};

/// Registration — the engine resolves this via `find_layer_scorer("shortgpt_bi")`. Takes no params.
#[distributed_slice(LAYER_SCORERS)]
static SHORTGPT_BI: LayerScorerReg = LayerScorerReg {
    name: "shortgpt_bi",
    phase: LayerScorerPhase::PerLayerStreaming,
    make: |_p: StageParams, _args: StageArgs<'_>| Box::new(ShortGptBiScorer),
    reads_subtensors: &[],
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{find_layer_scorer, registered_layer_scorer_names};

    // ── cosine_similarity (ported from the engine, with its tests) ──

    #[test]
    fn cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[0.0, 0.0]), 0.0);
    }

    // ── scorer ctx mock ──

    struct MockCtx {
        dim: usize,
        seq_len: usize,
        pooled_in: Vec<f32>,
        pooled_out: Vec<f32>,
        raw_in: Option<Vec<f32>>,
        raw_out: Vec<f32>,
        raw_in_dims: (usize, usize),
    }

    impl LayerScorerCtx for MockCtx {
        fn n_layers(&self) -> usize {
            0
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn seq_len(&self) -> usize {
            self.seq_len
        }
        fn pooled_in(&self) -> &[f32] {
            &self.pooled_in
        }
        fn pooled_out(&self) -> &[f32] {
            &self.pooled_out
        }
        fn raw_in(&self) -> Option<&[f32]> {
            self.raw_in.as_deref()
        }
        fn raw_out(&self) -> &[f32] {
            &self.raw_out
        }
        fn raw_in_dims(&self) -> (usize, usize) {
            self.raw_in_dims
        }
        fn x_mean(&self, _layer: usize) -> Option<&[f32]> {
            None
        }
        fn primary_subtensor(&self, _layer: usize, _name: &str) -> Option<(&[f32], usize, usize)> {
            None
        }
        fn secondary_subtensor(
            &self,
            _layer: usize,
            _name: &str,
        ) -> Option<(&[f32], usize, usize)> {
            None
        }
        fn gqa(&self) -> (usize, usize, usize) {
            (0, 0, 0)
        }
    }

    // ── mean_pool ──

    #[test]
    fn mean_pool_orthogonal_is_one() {
        let ctx = MockCtx {
            dim: 4,
            seq_len: 1,
            pooled_in: vec![1.0, 0.0, 0.0, 0.0],
            pooled_out: vec![0.0, 1.0, 0.0, 0.0],
            raw_in: None,
            raw_out: vec![],
            raw_in_dims: (0, 0),
        };
        assert!((MeanPoolScorer.score(0, &ctx) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_identical_is_zero() {
        let ctx = MockCtx {
            dim: 4,
            seq_len: 1,
            pooled_in: vec![1.0, 2.0, 3.0, 4.0],
            pooled_out: vec![1.0, 2.0, 3.0, 4.0],
            raw_in: None,
            raw_out: vec![],
            raw_in_dims: (0, 0),
        };
        assert!(MeanPoolScorer.score(0, &ctx) < 1e-6);
    }

    // ── shortgpt_bi ──

    #[test]
    fn shortgpt_bi_matches_engine_example() {
        // before = [[1,0],[1,0]]  after = [[0,1],[1,0]]
        //  → per-token cos = (0, 1) → mean 0.5 → importance = 1 − 0.5 = 0.5
        // (mirrors the engine's test_collector_three_way_both_formulas_populated).
        let ctx = MockCtx {
            dim: 2,
            seq_len: 2,
            pooled_in: vec![],
            pooled_out: vec![],
            raw_in: Some(vec![1.0, 0.0, 1.0, 0.0]),
            raw_out: vec![0.0, 1.0, 1.0, 0.0],
            raw_in_dims: (2, 2),
        };
        assert!((ShortGptBiScorer.score(0, &ctx) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn shortgpt_bi_no_raw_is_zero() {
        let ctx = MockCtx {
            dim: 2,
            seq_len: 2,
            pooled_in: vec![],
            pooled_out: vec![],
            raw_in: None,
            raw_out: vec![0.0, 1.0, 1.0, 0.0],
            raw_in_dims: (0, 0),
        };
        assert_eq!(ShortGptBiScorer.score(0, &ctx), 0.0);
    }

    // ── registration ──

    #[test]
    fn registers_into_slice() {
        for name in ["mean_pool", "shortgpt_bi"] {
            let reg = find_layer_scorer(name).expect("scorer registered in LAYER_SCORERS");
            assert_eq!(reg.name, name);
            assert_eq!(reg.phase, LayerScorerPhase::PerLayerStreaming);
            assert!(reg.reads_subtensors.is_empty());
            let scorer = (reg.make)(StageParams::default(), &[]);
            assert_eq!(scorer.name(), name);
        }
        let names = registered_layer_scorer_names();
        assert!(names.contains(&"mean_pool"));
        assert!(names.contains(&"shortgpt_bi"));
    }
}
