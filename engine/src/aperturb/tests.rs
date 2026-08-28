//! Parity against the reference harness.
//!
//! `fixtures/parity_v1.json` is produced by driving the paper harness's own functions — its
//! attention recompute, its causal-boundary derivation, its factored projection, its aggregation —
//! over a small deterministic geometry. Inputs are not stored: both sides generate them from the
//! same 64-bit LCG, so the fixture carries only the output-projection factors and the expected
//! numbers, and there is no way for the two sides to drift onto different tensors without the
//! comparison failing loudly.
//!
//! The geometry is small but not degenerate. `n_heads_q = 6` over `n_kv_heads = 2` gives
//! `n_rep = 3`, so a query head mapped by `h % n_kv` instead of `h / n_rep` produces a different
//! answer and is caught. The candidate set covers each shape the operator has to distinguish:
//! everything retained (must read exactly zero), one list shared by all heads, different lists of
//! equal length, different lists of *different* length, a candidate that blinds a leading row, and
//! a near-lossless one that lands in the regime where `1 − cos` loses its significant digits.

use serde_json::Value;

use super::keep::KeepSets;
use super::kernel::Geom;
use super::readout::{CellAgg, CellValue, Readout};
use super::{Config, LayerSource, OutputBasis, decide};

const FIXTURE: &str = include_str!("fixtures/parity_v1.json");

/// The fixture's generator, reproduced exactly: `s = s·MULT + INC (mod 2^64)`, value
/// `(s >> 40) / 2^24 − 0.5`. The shift leaves 24 bits, so both the quotient and the shift are exact
/// in f32 and the two sides agree bit for bit.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / 16777216.0 - 0.5
    }
    fn take(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

struct Fixture {
    g: Geom,
    wo_frac: f64,
    wo_rank: usize,
    /// `[d_out][d_in]` per layer — the output projection itself, for the SVD-free arm.
    wo: Vec<Vec<f32>>,
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    doc: Value,
}

impl LayerSource for Fixture {
    fn query_rows(&self, layer: usize) -> &[f32] {
        &self.q[layer]
    }
    fn keys(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }
    fn values(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }
}

fn load() -> Fixture {
    let doc: Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    let go = &doc["geom"];
    let u = |k: &str| go[k].as_u64().expect("geom field") as usize;
    let g = Geom {
        n_layers: u("n_layers"),
        n_heads_q: u("n_q"),
        n_kv_heads: u("n_kv"),
        head_dim: u("head_dim"),
        current_pos: u("n"),
        rows: u("rows"),
    };
    let (q_scale, k_scale) = (
        go["q_scale"].as_f64().unwrap() as f32,
        go["k_scale"].as_f64().unwrap() as f32,
    );
    let d = g.q_dim();
    assert_eq!(d, u("q_dim"), "the fixture's q_dim must be n_q * head_dim");

    // One continuous stream: wo | q | k | v, layer-major within each block.
    let mut lcg = Lcg::new(doc["seed"].as_u64().expect("seed"));
    let wo: Vec<Vec<f32>> = (0..g.n_layers).map(|_| lcg.take(d * d)).collect();
    let per_q = g.n_heads_q * g.rows * g.head_dim;
    let per_kv = g.n_kv_heads * g.current_pos * g.head_dim;
    let scale = |v: Vec<f32>, s: f32| v.into_iter().map(|x| x * s).collect::<Vec<f32>>();
    let q = (0..g.n_layers)
        .map(|_| scale(lcg.take(per_q), q_scale))
        .collect();
    let k = (0..g.n_layers)
        .map(|_| scale(lcg.take(per_kv), k_scale))
        .collect();
    let v = (0..g.n_layers).map(|_| lcg.take(per_kv)).collect();

    Fixture {
        g,
        wo_frac: go["wo_frac"].as_f64().unwrap(),
        wo_rank: u("wo_rank"),
        wo,
        q,
        k,
        v,
        doc,
    }
}

fn basis(f: &Fixture) -> OutputBasis {
    let vr: Vec<Vec<f32>> = f.doc["wo_v_r"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            a.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();
    let sg: Vec<Vec<f32>> = f.doc["wo_sigma"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            a.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();
    OutputBasis::from_factors(&vr, &sg, f.g.q_dim(), f.wo_rank, f.wo_frac).expect("basis")
}

fn pool(f: &Fixture) -> Vec<(String, KeepSets)> {
    f.doc["arms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            let name = a["name"].as_str().unwrap().to_string();
            let mut ks = KeepSets::with_capacity(f.g.n_layers, f.g.n_kv_heads, 0);
            for (l, layer) in a["keep"].as_array().unwrap().iter().enumerate() {
                for (h, head) in layer.as_array().unwrap().iter().enumerate() {
                    let idx: Vec<u32> = head
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as u32)
                        .collect();
                    ks.push(l, h, &idx).expect("in order");
                }
            }
            (name, ks)
        })
        .collect()
}

/// Tolerances.
///
/// The relative bound is f32's mantissa carried through a `head_dim`-term dot, a softmax over up to
/// `N` terms and a `d`-term projection — the two sides reduce in different orders, so bit equality
/// is not on offer, but the disagreement is bounded by the arithmetic and not by the algorithm.
///
/// The absolute floor is only there for `1 − cos`, which cancels: an identical row pair produces a
/// cosine that is `1` to within one ulp on both sides, and the difference of those two ulps has no
/// meaningful relative size. Nothing else needs it.
const REL_L2: f32 = 1e-5;
const REL_DCOS: f32 = 1e-4;
const ABS_DCOS: f32 = 1e-6;

/// The worst `(relative, absolute)` disagreement seen, and where.
#[derive(Default)]
struct Worst {
    rel: f32,
    abs: f32,
    at: String,
}

impl Worst {
    fn see(&mut self, got: f32, want: f32, at: impl FnOnce() -> String) {
        let a = (got - want).abs();
        let r = if want == 0.0 { 0.0 } else { a / want.abs() };
        if a > self.abs {
            self.abs = a;
        }
        if r > self.rel {
            self.rel = r;
            self.at = at();
        }
    }
}

#[test]
fn cells_and_scores_match_the_reference_harness() {
    let f = load();
    let b = basis(&f);
    let p = pool(&f);
    let cfg = Config {
        readout: Readout::default(),
        keep_cells: true,
    };
    let dec = decide(&f, &b, &p, f.g, &cfg).expect("decision");

    let mut w_l2 = Worst::default();
    let mut w_dcos = Worst::default();

    for (s, want) in dec.scored.iter().zip(f.doc["arms"].as_array().unwrap()) {
        assert_eq!(s.name, want["name"].as_str().unwrap());

        let n_vis = want["vmask"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|x| x.as_bool().unwrap())
            .count();
        assert_eq!(
            s.visible_rows.count_ones() as usize,
            n_vis,
            "{}: visible-row count",
            s.name
        );
        assert_eq!(
            s.scores.n_cells,
            want["ncell"].as_u64().unwrap() as usize,
            "{}: aggregated cell count",
            s.name
        );

        let cells = s.cells.as_ref().expect("kept");
        for (key, got) in [("tr_l2_lt", &cells.l2), ("tr_dcos_lt", &cells.dcos)] {
            let dcos = key.contains("dcos");
            let acc = if dcos { &mut w_dcos } else { &mut w_l2 };
            for (l, row) in want[key].as_array().unwrap().iter().enumerate() {
                for (t, x) in row.as_array().unwrap().iter().enumerate() {
                    let w = x.as_f64().unwrap() as f32;
                    let g = got[l * f.g.rows + t];
                    let name = s.name.clone();
                    acc.see(g, w, || format!("{name} {key}[{l}][{t}] {g:e} vs {w:e}"));
                }
            }
        }

        for (key, r) in [
            (
                "tr_l2",
                Readout {
                    value: CellValue::L2,
                    agg: CellAgg::Mean,
                },
            ),
            (
                "tr_l2_rms",
                Readout {
                    value: CellValue::L2,
                    agg: CellAgg::Rms,
                },
            ),
            (
                "tr_dcos",
                Readout {
                    value: CellValue::Dcos,
                    agg: CellAgg::Mean,
                },
            ),
            (
                "tr_dcos_rms",
                Readout {
                    value: CellValue::Dcos,
                    agg: CellAgg::Rms,
                },
            ),
        ] {
            let w = want[key].as_f64().unwrap() as f32;
            let g = s.scores.get(r);
            let dcos = key.contains("dcos");
            let acc = if dcos { &mut w_dcos } else { &mut w_l2 };
            let name = s.name.clone();
            acc.see(g, w, || format!("{name} {key} {g:e} vs {w:e}"));
        }
    }

    // Reported, not just asserted: the margin is the evidence that the two implementations agree
    // to arithmetic and not merely to a tolerance someone chose.
    eprintln!(
        "[aperturb] l2   worst rel {:e} abs {:e}  ({})",
        w_l2.rel, w_l2.abs, w_l2.at
    );
    eprintln!(
        "[aperturb] dcos worst rel {:e} abs {:e}  ({})",
        w_dcos.rel, w_dcos.abs, w_dcos.at
    );

    assert!(
        w_l2.rel <= REL_L2,
        "l2 disagreement {:e} at {}",
        w_l2.rel,
        w_l2.at
    );
    assert!(
        w_dcos.rel <= REL_DCOS || w_dcos.abs <= ABS_DCOS,
        "dcos disagreement rel {:e} abs {:e} at {}",
        w_dcos.rel,
        w_dcos.abs,
        w_dcos.at
    );
}

#[test]
fn retaining_everything_scores_exactly_zero() {
    let f = load();
    let b = basis(&f);
    let p = pool(&f);
    let cfg = Config {
        readout: Readout::default(),
        keep_cells: true,
    };
    let dec = decide(&f, &b, &p, f.g, &cfg).expect("decision");
    let noop = dec
        .scored
        .iter()
        .find(|s| s.name == "noop")
        .expect("the fixture carries an identity candidate");

    // The reference and the candidate travel the same operator with the same rounding, so the
    // displacement is not merely small — it is the same float subtracted from itself.
    assert_eq!(
        noop.scores.get(Readout {
            value: CellValue::L2,
            agg: CellAgg::Rms
        }),
        0.0,
        "an identity candidate must displace nothing"
    );
    assert!(noop.cells.as_ref().unwrap().l2.iter().all(|&x| x == 0.0));
    // `1 - cos` cannot reach exactly zero in f32 even on identical vectors; it is pure noise here,
    // which is the whole reason `l2` is the default readout.
    assert!(
        noop.scores.get(Readout {
            value: CellValue::Dcos,
            agg: CellAgg::Rms
        }) < 1e-5
    );
}

#[test]
fn the_blind_candidate_loses_a_row_from_the_denominator() {
    let f = load();
    let b = basis(&f);
    let p = pool(&f);
    let dec = decide(&f, &b, &p, f.g, &Config::default()).expect("decision");
    let blind = dec
        .scored
        .iter()
        .find(|s| s.name == "blind")
        .expect("present");
    assert!(
        (blind.visible_rows.count_ones() as usize) < f.g.rows,
        "the blind candidate must actually blind a row, or it tests nothing"
    );
    assert_eq!(
        blind.scores.n_cells,
        f.g.n_layers * blind.visible_rows.count_ones() as usize
    );
}

#[test]
fn the_ragged_candidate_is_measured_and_flagged() {
    let f = load();
    let b = basis(&f);
    let p = pool(&f);
    let dec = decide(&f, &b, &p, f.g, &Config::default()).expect("decision");
    let ragged = dec
        .scored
        .iter()
        .find(|s| s.name == "ragged")
        .expect("present");
    assert!(
        ragged.ragged,
        "per-head lengths differ, so this cannot be committed even though it scores"
    );
    assert!(dec.scored.iter().filter(|s| s.ragged).count() == 1);
}

#[test]
fn the_winner_is_the_smallest_score_under_the_configured_readout() {
    let f = load();
    let b = basis(&f);
    let p = pool(&f);
    for r in Readout::ALL {
        let cfg = Config {
            readout: r,
            keep_cells: false,
        };
        let dec = decide(&f, &b, &p, f.g, &cfg).expect("decision");
        let best = dec.winner().scores.get(r);
        assert!(
            dec.scored.iter().all(|s| s.scores.get(r) >= best),
            "readout {r}: winner is not the minimum"
        );
        // The identity candidate displaces nothing, so it wins every readout.
        assert_eq!(dec.winner().name, "noop", "readout {r}");
    }
}

#[test]
fn mean_never_exceeds_rms_on_any_candidate() {
    let f = load();
    let b = basis(&f);
    let p = pool(&f);
    let dec = decide(&f, &b, &p, f.g, &Config::default()).expect("decision");
    for s in &dec.scored {
        for v in [CellValue::L2, CellValue::Dcos] {
            let mean = s.scores.get(Readout {
                value: v,
                agg: CellAgg::Mean,
            });
            let rms = s.scores.get(Readout {
                value: v,
                agg: CellAgg::Rms,
            });
            assert!(
                mean <= rms + 1e-6,
                "{}: mean {mean} exceeds rms {rms} — the two aggregations are reading different cells",
                s.name
            );
        }
    }
}

/// The control arm: no decomposition anywhere, so a disagreement here is the recompute's, and a
/// disagreement that appears only in the truncated arm is the factor table's.
///
/// It also exercises the claim that the exact projection and the low-rank one are the same code
/// with a different table — `OutputBasis::untruncated` is just `W_oᵀ` at full rank.
#[test]
fn the_untruncated_arm_matches_without_any_decomposition() {
    let f = load();
    let d = f.g.q_dim();
    let b = OutputBasis::untruncated(&f.wo, d, d).expect("basis");
    assert_eq!(b.frac(), None);
    assert_eq!(
        b.metric_key(16, Readout::default()),
        "aperturb_prev16_l2_rms",
        "the untruncated arm carries no rank token"
    );

    let p = pool(&f);
    let cfg = Config {
        readout: Readout::default(),
        keep_cells: true,
    };
    let dec = decide(&f, &b, &p, f.g, &cfg).expect("decision");

    let mut worst = Worst::default();
    for (s, want) in dec.scored.iter().zip(f.doc["arms"].as_array().unwrap()) {
        let cells = s.cells.as_ref().expect("kept");
        for (l, row) in want["full_l2_lt"].as_array().unwrap().iter().enumerate() {
            for (t, x) in row.as_array().unwrap().iter().enumerate() {
                let w = x.as_f64().unwrap() as f32;
                let g = cells.l2[l * f.g.rows + t];
                let name = s.name.clone();
                worst.see(g, w, || {
                    format!("{name} full_l2_lt[{l}][{t}] {g:e} vs {w:e}")
                });
            }
        }
        for (key, r) in [
            (
                "full_l2",
                Readout {
                    value: CellValue::L2,
                    agg: CellAgg::Mean,
                },
            ),
            (
                "full_l2_rms",
                Readout {
                    value: CellValue::L2,
                    agg: CellAgg::Rms,
                },
            ),
        ] {
            let w = want[key].as_f64().unwrap() as f32;
            let g = s.scores.get(r);
            let name = s.name.clone();
            worst.see(g, w, || format!("{name} {key} {g:e} vs {w:e}"));
        }
    }
    eprintln!(
        "[aperturb] full-W_o worst rel {:e} abs {:e}  ({})",
        worst.rel, worst.abs, worst.at
    );
    assert!(
        worst.rel <= REL_L2,
        "untruncated l2 disagreement {:e} at {}",
        worst.rel,
        worst.at
    );
}

/// The combined arm: the engine factors the projection itself, then scores with its own basis.
///
/// This is what a deployed run does, and it is the only test that carries both error sources at
/// once. Running it beside `cells_and_scores_match_the_reference_harness` — which uses the
/// reference's own factors — separates them: whatever the difference between the two margins is,
/// that is the decomposition's contribution and nothing else's.
#[test]
fn engine_computed_factors_reproduce_the_reference_scores() {
    let f = load();
    let d = f.g.q_dim();
    let cfg_f = super::subspace::FactorConfig::default();
    let (b, reports) =
        OutputBasis::from_weights(&f.wo, d, d, f.wo_frac, &cfg_f).expect("factorization");
    assert_eq!(
        b.rank(),
        f.wo_rank,
        "the rank rule must agree with the fixture"
    );
    for (l, rep) in reports.iter().enumerate() {
        assert!(rep.residual <= cfg_f.residual_limit, "layer {l}: {rep:?}");
        assert!(rep.orthogonality < 1e-10, "layer {l}: {rep:?}");
        assert!(
            rep.deflation_ratio <= 1.0 + 1e-3,
            "layer {l}: converged to a subspace that is not the top one: {rep:?}"
        );
    }

    let p = pool(&f);
    let dec = decide(&f, &b, &p, f.g, &Config::default()).expect("decision");
    let mut worst = Worst::default();
    for (s, want) in dec.scored.iter().zip(f.doc["arms"].as_array().unwrap()) {
        for (key, r) in [
            (
                "tr_l2",
                Readout {
                    value: CellValue::L2,
                    agg: CellAgg::Mean,
                },
            ),
            (
                "tr_l2_rms",
                Readout {
                    value: CellValue::L2,
                    agg: CellAgg::Rms,
                },
            ),
        ] {
            let w = want[key].as_f64().unwrap() as f32;
            let g = s.scores.get(r);
            let name = s.name.clone();
            worst.see(g, w, || format!("{name} {key} {g:e} vs {w:e}"));
        }
    }
    eprintln!(
        "[aperturb] own-factors worst rel {:e} abs {:e}  ({})  iters {:?}",
        worst.rel,
        worst.abs,
        worst.at,
        reports.iter().map(|r| r.iters).collect::<Vec<_>>()
    );
    assert!(
        worst.rel <= REL_L2,
        "own-factor l2 disagreement {:e} at {}",
        worst.rel,
        worst.at
    );
}

#[test]
fn the_rank_rule_reproduces_the_reference_integers() {
    // The five swept fractions against the six canonical widths.
    assert_eq!(OutputBasis::rank_for(0.0078125, 1536), 12);
    assert_eq!(OutputBasis::rank_for(0.0078125, 2048), 16);
    assert_eq!(OutputBasis::rank_for(0.0078125, 3072), 24);
    assert_eq!(OutputBasis::rank_for(0.0078125, 3584), 28);
    assert_eq!(OutputBasis::rank_for(0.0078125, 4096), 32);
    assert_eq!(OutputBasis::rank_for(0.0625, 4096), 256);
    assert_eq!(OutputBasis::rank_for(0.00390625, 4096), 16);
    // Never zero, however small the fraction.
    assert_eq!(OutputBasis::rank_for(1e-9, 128), 1);
}
