//! TriAttention (Mao et al., 2026, arXiv:2604.04921) — trigonometric frequency-domain KV eviction.
//!
//! Faithful reproduction of the reference **Path-1** (`WeianMao/triattention`,
//! `methods/pruning_utils.py` + `methods/triattention.py`), DEFAULT config:
//! - `normalize_scores = false` (per-head z-norm OFF),
//! - `seed = None` (no tie-break noise → deterministic),
//! - `per_head_pruning = false` (global union over all sampled heads),
//! - `score_aggregation = "mean"` (average over the geometric offset grid),
//! - `allow_prefill_compression = false` (the WHOLE prefill prefix is pinned; only decode tokens
//!   are scored / evicted).
//!
//! **Route A — no engine forward-core edit.** The engine stores POST-RoPE keys; the reference also
//! stores post-RoPE keys and inverts RoPE at score time (`invert_rope`). So this crate reads
//! [`TensorKind::Key`] (post-RoPE, already exposed), inverts RoPE in-plugin to recover the base-space
//! complex key, and scores it against the calibrated query centers — exactly as the reference does.
//!
//! ## Score (per sampled head, per key `k` at decode position `p`, round start `R`)
//!
//! ```text
//! S(k) = mean over geometric offsets δ ∈ {1,2,4,…,2^16} of
//!          [ Σ_f |E[q_f]|·|k_f| · cos((R - p + δ)·ω_f + φ_f) ]
//!        + Σ_f (E[|q_f|] - |E[q_f]|)·|k_f|
//! ```
//! with `φ_f = atan2(Im, Re)` of `E[q_f]·conj(k_f)`, `ω_f = inv_freq[f] = θ^(-2f/head_dim)`, and
//! `freq_scale_sq ≡ 1` (standard RoPE, `attention_scaling = 1`). `|k_f|` and `arg(k_f)` come from the
//! in-plugin RoPE inversion of the stored post-RoPE key (`to_complex_pairs`, "half" split).
//!
//! ## Global keep-set
//!
//! `head_matrix` = the per-head score rows over ALL sampled heads of ALL layers → `combined` = max
//! over heads → union-based top-k over the decode tokens, with the prefill prefix always pinned. This
//! cross-layer aggregation is computed by [`compute_keepset_global`] (the parity target). The live
//! per-layer [`KVMutationStage`] dispatch sees one layer at a time (see [`TriAttention::on_phase`]).

use argus_extension_api::{
    CacheHandle, CacheOpError, CrossLayerStageCtx, KVMutationStage, MutationPhase, StageArgs,
    StageCaps, StageCtx, StageParams, TensorKind, register_kv_mutation_stage,
};

/// TriAttention reads POST-RoPE `Key` (inverted in-plugin) and pins the whole prefill via the
/// run-supplied `prefix_length` (not a small attention-sink), so the cap-level default prefix is 0.
const TRIATTENTION_CAPS: StageCaps = StageCaps {
    reads: &[TensorKind::Key],
    default_protected_prefix: 0,
    produces_merge_plan: false,
    // TriAttention's DEFAULT mode is a cross-layer GLOBAL keep-set: score every layer's resident keys,
    // aggregate across all sampled heads of all layers, and apply ONE keep-set to every layer. The
    // engine drives that through `on_whole_model` (a `CrossLayerStageCtx` over all caches) instead of
    // the per-layer `on_phase` loop. (`on_phase` stays implemented as the reference per-layer-independent
    // variant — see the struct doc — so the same crate naturally exposes both modes.)
    whole_model: true,
    prefill_attn_window: None,
};

// ───────────────────────────── calibration stats ─────────────────────────────

/// The flat f32 calibration binary the plugin loads (produced from the reference `.pt` by the P1
/// converter). Layout (little-endian):
/// ```text
/// magic  : 8 bytes  b"TACALIB1"
/// u32    : num_layers
/// u32    : num_heads      (attention heads per layer, pre-GQA)
/// u32    : freq_count     (head_dim / 2)
/// u32    : reserved (0)
/// data   : [layer][head] → f32[freq_count] q_mean_real, f32[freq_count] q_mean_imag,
///                          f32[freq_count] q_abs_mean
/// ```
pub struct Calib {
    pub num_layers: usize,
    pub num_heads: usize,
    pub freq_count: usize,
    data: Vec<f32>,
}

impl Calib {
    /// Parse the flat calibration binary (bit-preserving: f32 stays f32).
    pub fn from_bytes(raw: &[u8]) -> Result<Self, String> {
        if raw.len() < 24 || &raw[0..8] != b"TACALIB1" {
            return Err("calib: bad magic / too short".to_string());
        }
        let rd =
            |o: usize| u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]) as usize;
        let (num_layers, num_heads, freq_count) = (rd(8), rd(12), rd(16));
        let want = num_layers * num_heads * 3 * freq_count;
        let body = &raw[24..];
        if body.len() < want * 4 {
            return Err(format!(
                "calib: body {} f32 < expected {} ({}×{}×3×{})",
                body.len() / 4,
                want,
                num_layers,
                num_heads,
                freq_count
            ));
        }
        let mut data = Vec::with_capacity(want);
        for i in 0..want {
            let o = i * 4;
            data.push(f32::from_le_bytes([
                body[o],
                body[o + 1],
                body[o + 2],
                body[o + 3],
            ]));
        }
        Ok(Self {
            num_layers,
            num_heads,
            freq_count,
            data,
        })
    }

    /// Load the calibration binary from a filesystem path.
    pub fn from_path(path: &str) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("calib: read {path}: {e}"))?;
        Self::from_bytes(&raw)
    }

    #[inline]
    fn field(&self, layer: usize, head: usize, which: usize) -> &[f32] {
        let base = (((layer * self.num_heads + head) * 3) + which) * self.freq_count;
        &self.data[base..base + self.freq_count]
    }
    /// `E[Re q_f]` for `(layer, head)`.
    pub fn q_mean_real(&self, layer: usize, head: usize) -> &[f32] {
        self.field(layer, head, 0)
    }
    /// `E[Im q_f]` for `(layer, head)`.
    pub fn q_mean_imag(&self, layer: usize, head: usize) -> &[f32] {
        self.field(layer, head, 1)
    }
    /// `E[|q_f|]` for `(layer, head)`.
    pub fn q_abs_mean(&self, layer: usize, head: usize) -> &[f32] {
        self.field(layer, head, 2)
    }
    /// The three calibration vectors for `(layer, head)` bundled (the scorer's per-head input).
    pub fn head_stat(&self, layer: usize, head: usize) -> HeadStat<'_> {
        HeadStat {
            q_mean_real: self.field(layer, head, 0),
            q_mean_imag: self.field(layer, head, 1),
            q_abs_mean: self.field(layer, head, 2),
        }
    }
}

/// The three per-head calibration vectors (`E[Re q_f]`, `E[Im q_f]`, `E[|q_f|]`), each `freq_count`
/// long — the per-head input to [`score_head`].
#[derive(Clone, Copy)]
pub struct HeadStat<'a> {
    pub q_mean_real: &'a [f32],
    pub q_mean_imag: &'a [f32],
    pub q_abs_mean: &'a [f32],
}

// ───────────────────────────── RoPE / offsets ─────────────────────────────

/// `inv_freq[f] = 1 / θ^(2f/head_dim)` for `f ∈ [0, head_dim/2)` — the RoPE base frequencies, which
/// double as the score's `ω_f`. Matches the reference (HF rotary) `1.0 / (base ** (arange(0,d,2)/d))`.
pub fn inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    let fc = head_dim / 2;
    (0..fc)
        .map(|f| 1.0f32 / theta.powf((2 * f) as f32 / head_dim as f32))
        .collect()
}

/// Geometric offset grid `{1, 2, 4, …, 2^k ≤ max_length}` (the reference `build_geometric_offsets`).
pub fn build_geometric_offsets(max_length: usize) -> Vec<f32> {
    let mut out = Vec::new();
    let mut v: u64 = 1;
    while v <= max_length as u64 {
        out.push(v as f32);
        v *= 2;
    }
    out
}

// ───────────────────────────── scoring ─────────────────────────────

/// Invert RoPE on a post-RoPE key and pair it into complex (`half` split), writing into `real`/`imag`
/// (each length `inv_freq.len()`). The single source of truth for the in-plugin RoPE inversion
/// (`invert_rope` with `scale = 1`) + `to_complex_pairs`:
/// ```text
/// real[f] = k_unrot[f]    = key[f]·cos + key[f+fc]·sin
/// imag[f] = k_unrot[f+fc] = key[f+fc]·cos − key[f]·sin    where cos = cos(pos·ω_f)
/// ```
pub fn invert_rope_into(
    key: &[f32],
    pos: usize,
    inv_freq: &[f32],
    real: &mut [f32],
    imag: &mut [f32],
) {
    let fc = inv_freq.len();
    let p = pos as f32;
    for f in 0..fc {
        let (sin_f, cos_f) = (p * inv_freq[f]).sin_cos();
        real[f] = key[f] * cos_f + key[f + fc] * sin_f;
        imag[f] = key[f + fc] * cos_f - key[f] * sin_f;
    }
}

/// Allocating wrapper over [`invert_rope_into`] (test/diagnostic convenience). Returns
/// `(real, imag)` = the base-space complex key.
pub fn invert_rope_complex(key: &[f32], pos: usize, inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let fc = inv_freq.len();
    let mut real = vec![0.0f32; fc];
    let mut imag = vec![0.0f32; fc];
    invert_rope_into(key, pos, inv_freq, &mut real, &mut imag);
    (real, imag)
}

/// Score one sampled head over its decode-candidate keys (the reference per-head score row).
///
/// `keys[i]` = the post-RoPE key vector (`head_dim` f32) at absolute position `positions[i]`. The
/// RoPE inversion + complex pairing happen via [`invert_rope_into`]. Accumulation order mirrors the
/// reference exactly: per offset `Σ_f` first, then mean over offsets, then the position-independent
/// `additive` term.
pub fn score_head(
    keys: &[Vec<f32>],
    positions: &[usize],
    round_start: usize,
    stat: HeadStat<'_>,
    inv_freq: &[f32],
    offsets: &[f32],
) -> Vec<f32> {
    let HeadStat {
        q_mean_real,
        q_mean_imag,
        q_abs_mean,
    } = stat;
    let fc = inv_freq.len();
    // |E[q_f]| precomputed once per head.
    let q_mag: Vec<f32> = (0..fc)
        .map(|f| (q_mean_real[f] * q_mean_real[f] + q_mean_imag[f] * q_mean_imag[f]).sqrt())
        .collect();

    let mut real = vec![0.0f32; fc];
    let mut imag = vec![0.0f32; fc];
    let mut amp = vec![0.0f32; fc];
    let mut phi = vec![0.0f32; fc];
    let mut out = vec![0.0f32; keys.len()];

    for (i, key) in keys.iter().enumerate() {
        let base_delta = round_start as f32 - positions[i] as f32;
        invert_rope_into(key, positions[i], inv_freq, &mut real, &mut imag);
        let mut additive = 0.0f32;
        for f in 0..fc {
            let (kr, ki) = (real[f], imag[f]);
            let k_abs = (kr * kr + ki * ki).sqrt();
            amp[f] = q_mag[f] * k_abs;
            // φ = atan2(Im, Re) of  E[q]·conj(k) = (qr+i·qi)(kr−i·ki)
            let re = q_mean_real[f] * kr + q_mean_imag[f] * ki;
            let im = q_mean_imag[f] * kr - q_mean_real[f] * ki;
            phi[f] = im.atan2(re);
            additive += (q_abs_mean[f] - q_mag[f]) * k_abs; // freq_scale_sq ≡ 1
        }
        let mut acc = 0.0f32;
        for &d in offsets {
            let delta = base_delta + d;
            let mut s = 0.0f32;
            for f in 0..fc {
                s += amp[f] * (delta * inv_freq[f] + phi[f]).cos();
            }
            acc += s;
        }
        out[i] = acc / offsets.len() as f32 + additive;
    }
    out
}

/// All sampled `(layer, head)` pairs the calibration covers (the reference `sampled_heads` =
/// every layer × every attention head).
pub fn sampled_heads(calib: &Calib) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(calib.num_layers * calib.num_heads);
    for l in 0..calib.num_layers {
        for h in 0..calib.num_heads {
            v.push((l, h));
        }
    }
    v
}

/// The TriAttention round parameters (the reference `TriAttentionConfig` knobs that bear on the
/// keep-set, default mode).
#[derive(Clone, Debug)]
pub struct TriParams {
    /// Absolute KV budget `B`.
    pub budget: usize,
    /// The pinned prefill length (whole prefill; `allow_prefill_compression = false`).
    pub prefix_length: usize,
    /// `round_start` = the reference `absolute_position` at the compression round.
    pub round_start: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    /// `num_attention_heads / num_key_value_heads` (GQA group size).
    pub num_kv_groups: usize,
    pub theta: f32,
    /// Geometric offset cap (the reference `offset_max_length`, default 65536).
    pub offset_max_length: usize,
    /// Per-head z-norm (default `false`).
    pub normalize_scores: bool,
}

/// Build the `head_matrix` (one score row per sampled head) using a key accessor closure
/// `key_at(layer, kv_head, slot) -> Vec<f32>` (the post-RoPE key vector). Generic over key storage so
/// the same scorer serves the dumped-fixture parity harness and the live `ctx`-backed stage.
pub fn build_head_matrix<F>(
    sampled: &[(usize, usize)],
    calib: &Calib,
    params: &TriParams,
    decode_positions: &[usize],
    decode_start: usize,
    key_at: F,
) -> Vec<Vec<f32>>
where
    F: Fn(usize, usize, usize) -> Vec<f32>,
{
    let inv_freq = inv_freq(params.head_dim, params.theta);
    let offsets = build_geometric_offsets(params.offset_max_length);
    let mut rows = Vec::with_capacity(sampled.len());
    for &(layer, head) in sampled {
        let kv_head = (head / params.num_kv_groups.max(1)).min(params.num_kv_heads - 1);
        let keys: Vec<Vec<f32>> = (0..decode_positions.len())
            .map(|j| key_at(layer, kv_head, decode_start + j))
            .collect();
        rows.push(score_head(
            &keys,
            decode_positions,
            params.round_start,
            calib.head_stat(layer, head),
            &inv_freq,
            &offsets,
        ));
    }
    rows
}

/// Per-row z-normalization (`mean` / population `std` with `clamp_min(1e-6)`), applied in place. Only
/// when `normalize_scores` is set (default off). Mirrors the reference
/// `(head_matrix - mean) / std.clamp_min(1e-6)`.
pub fn normalize_rows(rows: &mut [Vec<f32>]) {
    for row in rows.iter_mut() {
        let n = row.len() as f32;
        if n == 0.0 {
            continue;
        }
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let std = var.sqrt().max(1e-6);
        for x in row.iter_mut() {
            *x = (*x - mean) / std;
        }
    }
}

/// Indices of the `k` largest scores, STABLE (ties broken by ascending index) — the deterministic
/// stand-in for `torch.topk(largest=True)` on the (tie-free, seed=None) faithful path.
fn topk_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// Union-based selection over the decode candidates (the reference `_select_union_based`): each head
/// marks its top-`keep_count`, the union is taken, and the top-`keep_count` of the union by `combined`
/// score is returned (ascending positions, relative to decode start). Fills from the residual when the
/// union is smaller than `keep_count`.
pub fn select_union_based(rows: &[Vec<f32>], combined: &[f32], keep_count: usize) -> Vec<usize> {
    let candidate_count = combined.len();
    if candidate_count <= keep_count {
        return (0..candidate_count).collect();
    }
    let per_head_quota = keep_count.min(candidate_count);
    let mut union_mask = vec![false; candidate_count];
    for row in rows {
        let head_k = per_head_quota.min(row.len());
        for idx in topk_indices(row, head_k) {
            union_mask[idx] = true;
        }
    }
    let union_indices: Vec<usize> = (0..candidate_count).filter(|&i| union_mask[i]).collect();

    if union_indices.len() >= keep_count {
        let sub_scores: Vec<f32> = union_indices.iter().map(|&i| combined[i]).collect();
        let mut sel: Vec<usize> = topk_indices(&sub_scores, keep_count)
            .iter()
            .map(|&t| union_indices[t])
            .collect();
        sel.sort_unstable();
        sel
    } else {
        let mut out = union_indices.clone();
        let remaining = keep_count - union_indices.len();
        let available = candidate_count - union_indices.len();
        if remaining > 0 && available > 0 {
            let mut residual = combined.to_vec();
            for &i in &union_indices {
                residual[i] = f32::NEG_INFINITY;
            }
            out.extend(topk_indices(&residual, remaining.min(available)));
        }
        out.sort_unstable();
        out
    }
}

/// `combined = max over heads` (the reference `head_matrix.max(dim=0).values`).
fn combine_max(rows: &[Vec<f32>], decode_count: usize) -> Vec<f32> {
    (0..decode_count)
        .map(|j| rows.iter().map(|r| r[j]).fold(f32::NEG_INFINITY, f32::max))
        .collect()
}

/// Assemble the final keep-set from the decode rows: pin the prefill prefix, select the decode
/// budget via [`select_union_based`], and return ascending absolute positions. Encapsulates the
/// reference `compute_keep_indices` tail shared by the global core and the per-layer stage.
fn assemble_keepset(
    rows: &[Vec<f32>],
    decode_start: usize,
    decode_count: usize,
    decode_budget: usize,
    normalize_scores: bool,
) -> Vec<usize> {
    let mut rows_owned: Vec<Vec<f32>>;
    let rows_ref: &[Vec<f32>] = if normalize_scores {
        rows_owned = rows.to_vec();
        normalize_rows(&mut rows_owned);
        &rows_owned
    } else {
        rows
    };
    let combined = combine_max(rows_ref, decode_count);
    let keep_count = decode_budget.min(decode_count);
    let decode_keep = select_union_based(rows_ref, &combined, keep_count);

    let mut keep: Vec<usize> = (0..decode_start).collect();
    keep.extend(decode_keep.iter().map(|&r| r + decode_start));
    keep.sort_unstable();
    keep
}

/// The decode partition (the reference `compute_keep_indices` guards). `Ok(keep)` is an early
/// trivial keep-set (no scoring needed); `Err((decode_start, decode_count, decode_budget))` means
/// "score the decode range".
fn decode_partition(
    l_total: usize,
    budget: usize,
    prefix_length: usize,
) -> Result<Vec<usize>, (usize, usize, usize)> {
    if l_total <= budget {
        return Ok((0..l_total).collect());
    }
    let decode_start = prefix_length.min(l_total);
    let decode_count = l_total - decode_start;
    if decode_count == 0 {
        return Ok((0..budget.min(l_total)).collect());
    }
    let decode_budget = budget.saturating_sub(decode_start);
    if decode_budget == 0 {
        return Ok((0..budget.min(decode_start)).collect());
    }
    Err((decode_start, decode_count, decode_budget))
}

/// **The parity target.** Compute the GLOBAL TriAttention keep-set from all layers' post-RoPE keys.
///
/// `keys[layer][kv_head][slot]` = the post-RoPE key vector (`head_dim` f32); `positions[slot]` = the
/// absolute position of cache slot `slot` (identity `0..L` for round 1). Returns the ascending kept
/// absolute positions (prefill prefix + selected decode tokens), faithful to the reference default
/// mode (`head_matrix` over all sampled heads → max over heads → union top-k, prefill pinned).
pub fn compute_keepset_global(
    keys: &[Vec<Vec<Vec<f32>>>],
    calib: &Calib,
    params: &TriParams,
    positions: &[usize],
) -> Vec<usize> {
    let l_total = positions.len();
    let (decode_start, decode_count, decode_budget) =
        match decode_partition(l_total, params.budget, params.prefix_length) {
            Ok(keep) => return keep,
            Err(parts) => parts,
        };

    let decode_positions: Vec<usize> = positions[decode_start..l_total].to_vec();
    let sampled = sampled_heads(calib);

    let rows = build_head_matrix(
        &sampled,
        calib,
        params,
        &decode_positions,
        decode_start,
        |layer, kv_head, slot| keys[layer][kv_head][slot].clone(),
    );

    assemble_keepset(
        &rows,
        decode_start,
        decode_count,
        decode_budget,
        params.normalize_scores,
    )
}

// ───────────────────────────── live stage ─────────────────────────────

/// The TriAttention KV mutation stage (live engine path).
///
/// **Scope note.** The stage exposes BOTH modes. [`on_phase`](TriAttention::on_phase) is the
/// per-layer dispatch ([`StageBackedPolicy::evict_layer`] → `drive_mutation_layer`): it sees ONE
/// layer's cache and reproduces the reference scoring/selection over THIS layer's sampled heads (=
/// the reference GLOBAL default for a single-layer model, the per-layer-independent variant
/// otherwise). [`on_whole_model`](TriAttention::on_whole_model) is the cross-layer GLOBAL default:
/// the engine drives it once (caps `whole_model = true`) over all layers via a
/// [`CrossLayerStageCtx`], faithfully reproducing [`compute_keepset_global`] on the live cache (the
/// same aggregation the harness pins). So the cross-layer default is now reachable in the live engine,
/// not only the parity harness.
pub struct TriAttention {
    calib: Option<Calib>,
    calib_err: Option<String>,
    prefix_length: usize,
    offset_max_length: usize,
    normalize_scores: bool,
    theta: f32,
}

impl TriAttention {
    /// Build from the shared params + the technique-private `--set` blob. Recognized keys:
    /// `calib_path` (required for scoring), `prefix_length`, `offset_max_length`, `normalize_scores`,
    /// `rope_theta`. A missing/unreadable `calib_path` is recorded and surfaced at `on_phase` (the
    /// `make` ABI is infallible).
    pub fn from_args(p: StageParams, args: StageArgs<'_>) -> Self {
        let mut calib_path: Option<&str> = None;
        let mut prefix_length = p.protected_prefix;
        let mut offset_max_length = 65536usize;
        let mut normalize_scores = false;
        let mut theta = 1_000_000.0f32;
        for a in args {
            match a.key {
                "calib_path" => calib_path = Some(a.val),
                "prefix_length" => {
                    if let Ok(v) = a.val.parse() {
                        prefix_length = v;
                    }
                }
                "offset_max_length" => {
                    if let Ok(v) = a.val.parse() {
                        offset_max_length = v;
                    }
                }
                "normalize_scores" => {
                    normalize_scores = matches!(a.val, "1" | "true" | "yes");
                }
                "rope_theta" => {
                    if let Ok(v) = a.val.parse() {
                        theta = v;
                    }
                }
                _ => {}
            }
        }
        let (calib, calib_err) = match calib_path {
            Some(path) => match Calib::from_path(path) {
                Ok(c) => (Some(c), None),
                Err(e) => (None, Some(e)),
            },
            None => (
                None,
                Some("triattention: missing --set calib_path".to_string()),
            ),
        };
        Self {
            calib,
            calib_err,
            prefix_length,
            offset_max_length,
            normalize_scores,
            theta,
        }
    }

    /// Construct directly from an in-memory [`Calib`] (tests / harness — bypasses `--set calib_path`).
    pub fn with_calib(
        calib: Calib,
        prefix_length: usize,
        offset_max_length: usize,
        normalize_scores: bool,
        theta: f32,
    ) -> Self {
        Self {
            calib: Some(calib),
            calib_err: None,
            prefix_length,
            offset_max_length,
            normalize_scores,
            theta,
        }
    }
}

impl KVMutationStage for TriAttention {
    fn name(&self) -> &str {
        "triattention"
    }

    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        let Some(calib) = self.calib.as_ref() else {
            // No calibration → cannot score; surface the configuration error (the only
            // String-carrying CacheOpError variant — a config fault, not a byte mutation).
            return Err(CacheOpError::UnsupportedFormat(
                self.calib_err
                    .clone()
                    .unwrap_or_else(|| "triattention: no calib".to_string()),
            ));
        };

        let current = ctx.current_pos();
        let budget = ctx.target_len();
        let (decode_start, decode_count, decode_budget) =
            match decode_partition(current, budget, self.prefix_length.min(current)) {
                Ok(keep) => return cache.keep(&keep),
                Err(parts) => parts,
            };

        let head_dim = ctx.head_dim();
        let n_kv = ctx.n_kv_heads().max(1);
        let layer = ctx.layer_idx();
        // Round-1 coordinate frame: slot index == absolute position, round_start == current_pos.
        let round_start = current;
        let decode_positions: Vec<usize> = (decode_start..current).collect();

        let params = TriParams {
            budget,
            prefix_length: self.prefix_length,
            round_start,
            head_dim,
            num_kv_heads: n_kv,
            num_kv_groups: (calib.num_heads / n_kv).max(1),
            theta: self.theta,
            offset_max_length: self.offset_max_length,
            normalize_scores: self.normalize_scores,
        };

        // This layer's sampled heads only (per-layer dispatch scope — see the struct doc).
        let sampled: Vec<(usize, usize)> = (0..calib.num_heads).map(|h| (layer, h)).collect();

        let rows = build_head_matrix(
            &sampled,
            calib,
            &params,
            &decode_positions,
            decode_start,
            |_layer, kv_head, slot| {
                let mut b = vec![0.0f32; head_dim];
                if let Some(h) = ctx.tensor(TensorKind::Key) {
                    h.read_row(slot, kv_head, &mut b);
                }
                b
            },
        );

        let keep = assemble_keepset(
            &rows,
            decode_start,
            decode_count,
            decode_budget,
            self.normalize_scores,
        );
        cache.keep(&keep)
    }

    /// The cross-layer GLOBAL default mode (the reference Path-1 default). The engine invokes this once
    /// (caps `whole_model = true`) with a [`CrossLayerStageCtx`] spanning EVERY layer; it reproduces
    /// [`compute_keepset_global`] over the live cache — score all layers' sampled heads, max over
    /// heads, union top-k, prefill pinned — and applies the ONE keep-set to every layer via the
    /// model-scoped [`CacheHandle`] (`keep` fans out). Faithful to the parity target; the only
    /// difference from the harness is that keys + positions come from the engine ctx, and `round_start`
    /// is derived from the engine's absolute positions instead of being passed in.
    fn on_whole_model(
        &self,
        ctx: &dyn CrossLayerStageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        let Some(calib) = self.calib.as_ref() else {
            return Err(CacheOpError::UnsupportedFormat(
                self.calib_err
                    .clone()
                    .unwrap_or_else(|| "triattention: no calib".to_string()),
            ));
        };

        let current = ctx.current_pos();
        let budget = ctx.target_len();
        let (decode_start, decode_count, decode_budget) =
            match decode_partition(current, budget, self.prefix_length) {
                Ok(keep) => return cache.keep(&keep),
                Err(parts) => parts,
            };

        let head_dim = ctx.head_dim();
        let n_kv = ctx.n_kv_heads().max(1);
        // Absolute positions = the engine source-of-truth (RoPE-continuation `saved_positions`), so the
        // plugin frame never drifts. `round_start` = the next absolute position to be generated = the
        // largest resident position + 1 (the most-recent token is always resident), which equals the
        // reference `absolute_position` at the compression round (identity frame: max=current−1 → current).
        let positions: Vec<usize> = (0..current).map(|s| ctx.abs_position(s)).collect();
        let round_start = positions.iter().copied().max().map_or(current, |m| m + 1);
        let decode_positions: Vec<usize> = positions[decode_start..current].to_vec();

        let params = TriParams {
            budget,
            prefix_length: self.prefix_length,
            round_start,
            head_dim,
            num_kv_heads: n_kv,
            num_kv_groups: (calib.num_heads / n_kv).max(1),
            theta: self.theta,
            offset_max_length: self.offset_max_length,
            normalize_scores: self.normalize_scores,
        };

        // ALL layers × all sampled heads (the cross-layer aggregation — `head_matrix` over the whole
        // model). The closure streams each (layer, kv_head, slot) key from the engine ctx, so no
        // N-layer key buffer is materialized in the plugin (per-layer-head fold).
        let sampled = sampled_heads(calib);
        let rows = build_head_matrix(
            &sampled,
            calib,
            &params,
            &decode_positions,
            decode_start,
            |layer, kv_head, slot| {
                let mut b = vec![0.0f32; head_dim];
                ctx.read_key(layer, kv_head, slot, &mut b);
                b
            },
        );

        let keep = assemble_keepset(
            &rows,
            decode_start,
            decode_count,
            decode_budget,
            self.normalize_scores,
        );
        cache.keep(&keep)
    }
}

register_kv_mutation_stage!(
    "triattention",
    |p, args| Box::new(TriAttention::from_args(p, args)),
    TRIATTENTION_CAPS,
    MutationPhase::KvMutate
);

#[cfg(test)]
mod tests;
