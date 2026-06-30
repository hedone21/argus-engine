//! Keep-set parity harness: the Rust faithful core vs the reference Path-1 oracle.
//!
//! Reads a fixture directory (env `TRIATTN_FIXTURE_DIR`) produced by the oracle dumps:
//! - round 1 (`P0`): `oracle_params.txt`, `qwen25_calib.bin`, `oracle_keys.f32`,
//!   `oracle_keepset.txt` — identity coordinates (`positions = 0..L`).
//! - round 2 (`P4`): `oracle_params_r2.txt`, `oracle_keys_r2.f32`, `oracle_keepset_r2.txt`,
//!   `oracle_positions_r2.txt` — NON-IDENTITY coordinates (survivors at their original absolute
//!   positions + new decode tokens, `round_start` past all positions).
//!
//! Each round asserts the Rust `compute_keepset_global` keep-set is byte-identical
//! (`symmetric_diff == 0`) to the reference on identical input keys, then runs an adversarial
//! NON-TAUTOLOGY check (zeroing one kept decode key must change the keep-set). With no fixture env
//! the tests no-op (so the default `cargo test -p triattention` stays green).

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use triattention::{Calib, TriParams, compute_keepset_global};

fn parse_params(path: &Path) -> BTreeMap<String, String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

fn read_ints(path: &Path) -> Vec<usize> {
    std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

fn symmetric_diff(a: &[usize], b: &[usize]) -> usize {
    let sa: HashSet<usize> = a.iter().copied().collect();
    let sb: HashSet<usize> = b.iter().copied().collect();
    sa.symmetric_difference(&sb).count()
}

/// Load a raw f32 key dump into `[layer][kv_head][slot][head_dim]`.
fn load_keys(
    path: &Path,
    num_layers: usize,
    num_kv: usize,
    l_total: usize,
    head_dim: usize,
) -> Vec<Vec<Vec<Vec<f32>>>> {
    let raw = std::fs::read(path).unwrap();
    assert_eq!(
        raw.len(),
        num_layers * num_kv * l_total * head_dim * 4,
        "keys size mismatch"
    );
    let f =
        |i: usize| f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
    let mut idx = 0usize;
    let mut keys = vec![vec![vec![vec![0.0f32; head_dim]; l_total]; num_kv]; num_layers];
    for layer in keys.iter_mut() {
        for kv in layer.iter_mut() {
            for slot in kv.iter_mut() {
                for d in slot.iter_mut() {
                    *d = f(idx);
                    idx += 1;
                }
            }
        }
    }
    keys
}

/// Run one round's parity + adversarial check. `params`/`keys`/`keepset` name the fixture files;
/// `positions_file` is `Some` for the non-identity (round-2) frame, `None` for identity (`0..L`).
fn run_round(
    dir: &Path,
    round: &str,
    params_file: &str,
    keys_file: &str,
    keepset_file: &str,
    positions_file: Option<&str>,
) {
    let p = parse_params(&dir.join(params_file));
    let g = |k: &str| p[k].parse::<usize>().unwrap();
    let (l_total, prefix_length, budget, round_start) =
        (g("L"), g("prefix_length"), g("budget"), g("round_start"));
    let (num_layers, num_kv_heads, head_dim) = (g("num_layers"), g("num_kv_heads"), g("head_dim"));
    let num_attention_heads = g("num_attention_heads");

    let calib = Calib::from_path(dir.join("qwen25_calib.bin").to_str().unwrap()).unwrap();
    let keys = load_keys(
        &dir.join(keys_file),
        num_layers,
        num_kv_heads,
        l_total,
        head_dim,
    );
    let oracle = read_ints(&dir.join(keepset_file));
    let positions: Vec<usize> = match positions_file {
        Some(f) => read_ints(&dir.join(f)),
        None => (0..l_total).collect(),
    };
    assert_eq!(positions.len(), l_total, "positions length must equal L");

    let params = TriParams {
        budget,
        prefix_length,
        round_start,
        head_dim,
        num_kv_heads,
        num_kv_groups: num_attention_heads / num_kv_heads,
        theta: p["rope_theta"].parse::<f32>().unwrap(),
        offset_max_length: g("offset_max_length"),
        normalize_scores: g("normalize_scores") != 0,
    };

    let rust_keep = compute_keepset_global(&keys, &calib, &params, &positions);
    let sd = symmetric_diff(&rust_keep, &oracle);
    println!(
        "[parity {round}] oracle_keep={} rust_keep={} symmetric_diff={}",
        oracle.len(),
        rust_keep.len(),
        sd
    );
    assert_eq!(
        sd, 0,
        "[{round}] Rust keep-set must match the reference Path-1 oracle exactly"
    );

    // ── adversarial NON-TAUTOLOGY: zero one KEPT decode key → keep-set MUST change ──
    let victim = *oracle
        .iter()
        .find(|&&x| x >= prefix_length)
        .expect("at least one decode token kept");
    let mut perturbed = keys.clone();
    for layer in perturbed.iter_mut() {
        for kv in layer.iter_mut() {
            for d in kv[victim].iter_mut() {
                *d = 0.0; // |k_f| = 0 → score collapses → this token should drop out
            }
        }
    }
    let perturbed_keep = compute_keepset_global(&perturbed, &calib, &params, &positions);
    let sd_perturbed = symmetric_diff(&perturbed_keep, &oracle);
    println!(
        "[parity {round}] adversarial: zeroed kept slot {victim} → symmetric_diff={sd_perturbed} (must be > 0)"
    );
    assert!(
        sd_perturbed > 0,
        "[{round}] perturbing a kept key must change the keep-set (non-tautology)"
    );
    assert!(
        !perturbed_keep.contains(&victim),
        "[{round}] the score-collapsed token must be evicted"
    );
}

fn fixture_dir() -> Option<String> {
    match std::env::var("TRIATTN_FIXTURE_DIR") {
        Ok(d) => Some(d),
        Err(_) => {
            eprintln!("[parity] TRIATTN_FIXTURE_DIR unset → SKIP");
            None
        }
    }
}

/// Round 1: identity coordinates (single-prefill faithful frame).
#[test]
fn round1_keepset_parity_vs_oracle() {
    let Some(dir) = fixture_dir() else { return };
    run_round(
        Path::new(&dir),
        "round1",
        "oracle_params.txt",
        "oracle_keys.f32",
        "oracle_keepset.txt",
        None,
    );
}

/// Round 2: NON-IDENTITY coordinates (post-eviction survivors at original positions + new tokens).
#[test]
fn round2_keepset_parity_vs_oracle() {
    let Some(dir) = fixture_dir() else { return };
    let dir = Path::new(&dir);
    if !dir.join("oracle_params_r2.txt").exists() {
        eprintln!("[parity round2] no round-2 fixture → SKIP");
        return;
    }
    run_round(
        dir,
        "round2",
        "oracle_params_r2.txt",
        "oracle_keys_r2.f32",
        "oracle_keepset_r2.txt",
        Some("oracle_positions_r2.txt"),
    );
}
