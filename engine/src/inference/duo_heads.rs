//! DuoAttention streaming-head ablation — output-faithful, compute-only.
//!
//! DuoAttention (Xiao et al. 2024, arXiv:2410.10819) statically partitions each layer's KV heads
//! into **retrieval heads** (full dense attention over all tokens) and **streaming heads** (attend
//! only to a constant-length Λ-window = the first `sink_size` tokens ∪ the last `recent_size`
//! tokens). The classification is offline, per-`(layer, KV-head)`, content-independent — loaded here
//! from a gate/label file.
//!
//! **Scope — this is the compute-only variant (engine "Tier 1").** The engine keeps ONE KV buffer
//! per layer with a single `current_pos` shared by all heads, so it cannot realize DuoAttention's
//! per-head *ragged* storage (retrieval heads full-N, streaming heads sink+recent) that produces the
//! paper's memory saving — that is an engine-core change (a per-`(KV-)head` valid-length container +
//! kernel), not a plugin. What this module *does* deliver is **output fidelity**: streaming heads'
//! attention output is recomputed over exactly the sink∪recent subset (softmax renormalized over
//! that subset), byte-for-byte the streaming primitive DuoAttention uses, so a researcher can study
//! which heads are retrieval vs streaming and how the masking moves the logits. It keeps the full KV
//! allocated → **zero memory saving, and it ADDS compute** (a second, windowed attention on top of
//! the full one). It is a probe, not DuoAttention's system benefit.
//!
//! **Faithfulness rests on original-position RoPE.** The engine stores RoPE'd keys at their original
//! positions and never re-rotates on eviction, so recomputing attention over the full cache
//! *restricted* to sink∪recent reproduces exactly what evicting-the-middle-then-attending would
//! yield. The recompute matches the engine's own host attention numerics (scale `1/sqrt(head_dim)`,
//! max-subtracted softmax, NaN→−∞ guard; see `backend::cpu::common::attention_gen`).
//!
//! **Hook point.** Mirrors [`crate::inference::head_mask`]: after `attention_into` (which fills all
//! heads with full attention) and before the `wo` projection, in BOTH prefill and decode, the
//! streaming heads' `head_dim`-wide slices of `ws.out_attn` are overwritten with their windowed
//! output; retrieval heads keep their full-attention output untouched. Masking both paths is
//! required so a streaming head's contribution is windowed throughout the residual stream.
//!
//! **GQA.** The label is per KV-head; every query head in that KV group `[kv_h·n_rep, (kv_h+1)·n_rep)`
//! shares the group's streaming/retrieval policy. `per_layer` stores the expanded *query*-head set.
//!
//! **Format support.** Requires a [`SelectiveRead`](crate::format::selective_read::SelectiveRead)
//! format (`StandardFormat`); on an unsupported format [`DuoHeads::apply`] bails loudly rather than
//! silently doing nothing. Backends: `cpu`, `cuda`, `opencl` (the GPU path round-trips through host,
//! mirroring the head-mask / W-DEVKV recipe). No flag → `None` → byte-identical.

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::backend::Backend;
use crate::format::KVCacheFormat;
use crate::tensor::Tensor;

/// A resolved DuoAttention head classification. Built once at startup from a gate/label file and
/// read-only during the forward pass — the run-constant analogue of [`crate::inference::head_mask`].
#[derive(Clone, Debug)]
pub struct DuoHeads {
    /// `per_layer[l]` = the **query**-head indices in layer `l` that are STREAMING (windowed).
    /// Empty for a layer whose heads are all retrieval → [`Self::apply`] short-circuits.
    per_layer: Vec<Vec<usize>>,
    /// Attention-sink prefix length (first-N tokens a streaming head always attends to).
    sink_size: usize,
    /// Recent sliding-window length (last-N tokens a streaming head attends to).
    recent_size: usize,
    n_heads_q: usize,
}

impl DuoHeads {
    /// Resolve a DuoAttention classification from a gate/label file, or `Ok(None)` when
    /// `duo_heads_file` is `None` (byte-identical run).
    ///
    /// The file is DuoAttention's `full_attention_heads` artifact: `n_layers` whitespace/tab
    /// separated rows, each with `n_heads_kv` floats (the per-`(layer, KV-head)` retrieval gate).
    /// A head is **retrieval** iff its gate `>= threshold`, else **streaming**. (Pre-binarized 0/1
    /// files work with the default threshold 0.5; continuous gates are thresholded here.)
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        duo_heads_file: Option<&str>,
        threshold: f32,
        sink_size: usize,
        recent_size: usize,
        n_layers: usize,
        n_heads_q: usize,
        n_heads_kv: usize,
    ) -> Result<Option<DuoHeads>> {
        let Some(path) = duo_heads_file else {
            return Ok(None);
        };
        if n_layers == 0 || n_heads_q == 0 || n_heads_kv == 0 {
            bail!("--duo-heads: model reports 0 layers / query heads / kv heads");
        }
        if !n_heads_q.is_multiple_of(n_heads_kv) {
            bail!(
                "--duo-heads: n_heads_q ({n_heads_q}) is not a multiple of n_heads_kv ({n_heads_kv})"
            );
        }
        if sink_size == 0 && recent_size == 0 {
            bail!("--duo-heads: --duo-sink-size and --duo-recent-size cannot both be 0");
        }

        let gates = load_gate_file(path, n_layers, n_heads_kv)?;
        let n_rep = n_heads_q / n_heads_kv;
        let mut per_layer: Vec<Vec<usize>> = vec![Vec::new(); n_layers];
        let mut streaming_kv = 0usize;
        for (l, row) in gates.iter().enumerate() {
            for (kv_h, &g) in row.iter().enumerate() {
                if g < threshold {
                    // Streaming KV head → expand to its GQA query-head group.
                    streaming_kv += 1;
                    for r in 0..n_rep {
                        per_layer[l].push(kv_h * n_rep + r);
                    }
                }
            }
        }
        if streaming_kv == 0 {
            bail!(
                "--duo-heads: no streaming heads at threshold {threshold} (every gate >= threshold \
                 → all-retrieval = full attention; lower --duo-threshold or check the file)"
            );
        }

        Ok(Some(DuoHeads {
            per_layer,
            sink_size,
            recent_size,
            n_heads_q,
        }))
    }

    /// Number of streaming `(layer, query-head)` units (for logging).
    pub fn streaming_head_count(&self) -> usize {
        self.per_layer.iter().map(Vec::len).sum()
    }

    /// `sink_size + recent_size` (the constant streaming budget), for logging.
    pub fn window_desc(&self) -> String {
        format!("sink={}, recent={}", self.sink_size, self.recent_size)
    }

    /// Recompute this layer's streaming query heads over their Λ-window and overwrite their slices
    /// in `out` (the concatenated per-query-head attention output). No-op (byte-identical) when the
    /// layer has no streaming heads. Used at the same seam in both prefill (`seq_len > 1`) and decode
    /// (`seq_len == 1`).
    ///
    /// `q` is the RoPE-applied query `[batch, seq_len, n_heads_q, head_dim]`; `start_pos` is the
    /// absolute position of the first query row; `fmt` supplies the KV cache (via `SelectiveRead`).
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        layer_idx: usize,
        q: &Tensor,
        fmt: &Arc<dyn KVCacheFormat>,
        out: &mut Tensor,
        backend: &dyn Backend,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<()> {
        let heads = match self.per_layer.get(layer_idx) {
            Some(h) if !h.is_empty() => h,
            // Fast path: no streaming head in this layer → out keeps full attention, untouched.
            _ => return Ok(()),
        };
        let sr = fmt.as_selective_read().ok_or_else(|| {
            anyhow::anyhow!(
                "--duo-heads requires a SelectiveRead KV format (StandardFormat); the active format \
                 does not support the streaming-head recompute"
            )
        })?;
        sr.attention_into_streaming(
            q,
            backend,
            out,
            self.n_heads_q,
            heads,
            self.sink_size,
            self.recent_size,
            start_pos,
            seq_len,
        )
    }
}

/// Load `n_layers × n_heads_kv` retrieval gates from a whitespace/tab-separated float file.
fn load_gate_file(path: &str, n_layers: usize, n_heads_kv: usize) -> Result<Vec<Vec<f32>>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("--duo-heads: cannot read '{path}'"))?;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vals: Vec<f32> = line
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<f32>()
                    .map_err(|_| anyhow::anyhow!("--duo-heads: '{path}' line {i}: bad float '{s}'"))
            })
            .collect::<Result<_>>()?;
        if vals.len() != n_heads_kv {
            bail!(
                "--duo-heads: '{path}' line {i} has {} values, expected n_heads_kv={n_heads_kv}",
                vals.len()
            );
        }
        rows.push(vals);
    }
    if rows.len() != n_layers {
        bail!(
            "--duo-heads: '{path}' has {} data rows, expected n_layers={n_layers}",
            rows.len()
        );
    }
    Ok(rows)
}

/// Pure host Λ-mask streaming attention — the numerical core, overwriting ONLY `streaming_heads`'
/// slices in `out` (retrieval heads are left as the caller's full-attention output).
///
/// Layouts (all F32, row-major):
/// - `q_host` / `out`: `[n_rows, n_heads_q, head_dim]`, row `r = batch*seq_len + step`.
/// - `k_all` / `v_all`: HeadMajor over ALL `n` cache positions — element `(kv_head, pos, d)` at
///   `(kv_head * n + pos) * head_dim + d` (the layout `gather_selected_kv` produces for
///   `select = 0..n`).
///
/// For query row at absolute position `qpos = start_pos + step`, a streaming head attends to
/// `sink = [0, sink_size)` ∪ `recent = [qpos+1-recent_size, qpos]`, intersected with the causal
/// prefix `[0, qpos]`. Softmax is renormalized over exactly that subset. Numerics mirror
/// `backend::cpu::common::attention_gen` (scale `1/sqrt(head_dim)`, max-subtracted softmax,
/// NaN→−∞, all-−∞ → uniform), so when the window covers the whole causal prefix the output equals
/// full attention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_streaming_attention(
    q_host: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    out: &mut [f32],
    n_rows: usize,
    n_heads_q: usize,
    kv_heads: usize,
    head_dim: usize,
    n: usize,
    streaming_heads: &[usize],
    sink_size: usize,
    recent_size: usize,
    start_pos: usize,
    seq_len: usize,
) {
    let n_rep = (n_heads_q / kv_heads).max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let row = n_heads_q * head_dim;
    let seq_len = seq_len.max(1);

    for r in 0..n_rows {
        let step = r % seq_len;
        let qpos = start_pos + step;
        let hi = qpos.min(n.saturating_sub(1)); // last causal key
        // Λ-window key positions: sink prefix ∪ recent tail, both within [0, hi].
        let sink_end = sink_size.min(hi + 1);
        let recent_start = (hi + 1).saturating_sub(recent_size).max(sink_end);
        let mut allowed: Vec<usize> = Vec::with_capacity(sink_end + (hi + 1 - recent_start));
        allowed.extend(0..sink_end);
        allowed.extend(recent_start..=hi);
        if allowed.is_empty() {
            continue; // degenerate (guarded at resolve): leave full-attention value in place.
        }

        for &h in streaming_heads {
            if h >= n_heads_q {
                continue;
            }
            let kv_h = h / n_rep;
            let q_off = r * row + h * head_dim;
            let q_vec = &q_host[q_off..q_off + head_dim];

            // logits over the allowed subset (+ max for stability).
            let mut logits = vec![0.0f32; allowed.len()];
            let mut max_v = f32::NEG_INFINITY;
            for (i, &p) in allowed.iter().enumerate() {
                let k_off = (kv_h * n + p) * head_dim;
                let k_vec = &k_all[k_off..k_off + head_dim];
                let mut s: f32 = q_vec.iter().zip(k_vec).map(|(a, b)| a * b).sum();
                s *= scale;
                if s.is_nan() {
                    s = f32::NEG_INFINITY;
                }
                logits[i] = s;
                if s > max_v {
                    max_v = s;
                }
            }

            let out_off = r * row + h * head_dim;
            for d in 0..head_dim {
                out[out_off + d] = 0.0;
            }
            if max_v == f32::NEG_INFINITY {
                // All logits −∞ → uniform (matches attention_gen fallback).
                let u = 1.0 / allowed.len() as f32;
                for &p in &allowed {
                    let v_off = (kv_h * n + p) * head_dim;
                    for d in 0..head_dim {
                        out[out_off + d] += u * v_all[v_off + d];
                    }
                }
                continue;
            }
            let mut denom = 0.0f32;
            for l in logits.iter_mut() {
                *l = (*l - max_v).exp();
                denom += *l;
            }
            let inv = 1.0 / denom;
            for (i, &p) in allowed.iter().enumerate() {
                let w = logits[i] * inv;
                let v_off = (kv_h * n + p) * head_dim;
                for d in 0..head_dim {
                    out[out_off + d] += w * v_all[v_off + d];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference full causal attention for one (row, head) over `[0, qpos]`, host f32.
    #[allow(clippy::too_many_arguments)]
    fn ref_full_attention(
        q_host: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        r: usize,
        h: usize,
        kv_h: usize,
        n: usize,
        n_heads_q: usize,
        head_dim: usize,
        qpos: usize,
    ) -> Vec<f32> {
        let row = n_heads_q * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_off = r * row + h * head_dim;
        let q_vec = &q_host[q_off..q_off + head_dim];
        let mut logits = vec![0.0f32; qpos + 1];
        let mut max_v = f32::NEG_INFINITY;
        for (p, lg) in logits.iter_mut().enumerate() {
            let k_off = (kv_h * n + p) * head_dim;
            let s: f32 = q_vec
                .iter()
                .zip(&k_all[k_off..k_off + head_dim])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale;
            *lg = s;
            if s > max_v {
                max_v = s;
            }
        }
        let mut denom = 0.0f32;
        for l in logits.iter_mut() {
            *l = (*l - max_v).exp();
            denom += *l;
        }
        let mut out = vec![0.0f32; head_dim];
        for (p, &lg) in logits.iter().enumerate() {
            let w = lg / denom;
            let v_off = (kv_h * n + p) * head_dim;
            for d in 0..head_dim {
                out[d] += w * v_all[v_off + d];
            }
        }
        out
    }

    /// Build deterministic q/k/v for `n` positions, MHA (n_heads_q == kv_heads).
    fn make_qkv(
        n_heads_q: usize,
        kv_heads: usize,
        head_dim: usize,
        n: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // q: one row (decode), [1, n_heads_q, head_dim].
        let mut q = vec![0.0f32; n_heads_q * head_dim];
        for (i, v) in q.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.1;
        }
        let mut k = vec![0.0f32; kv_heads * n * head_dim];
        let mut v = vec![0.0f32; kv_heads * n * head_dim];
        for i in 0..k.len() {
            k[i] = ((i % 11) as f32 - 5.0) * 0.05;
            v[i] = ((i % 13) as f32 - 6.0) * 0.03;
        }
        (q, k, v)
    }

    #[test]
    fn streaming_window_covering_all_equals_full_attention() {
        // sink=0, recent>=n → window is the whole causal prefix → must equal full attention.
        let (nq, kv, hd, n) = (2usize, 2usize, 4usize, 6usize);
        let (q, k, v) = make_qkv(nq, kv, hd, n);
        let mut out = vec![999.0f32; nq * hd]; // sentinel; will be overwritten for streaming heads
        let qpos = n - 1;
        compute_streaming_attention(
            &q,
            &k,
            &v,
            &mut out,
            1,
            nq,
            kv,
            hd,
            n,
            &[0, 1],
            0,
            n,
            qpos,
            1,
        );
        for h in 0..nq {
            let expect = ref_full_attention(&q, &k, &v, 0, h, h, n, nq, hd, qpos);
            for d in 0..hd {
                assert!(
                    (out[h * hd + d] - expect[d]).abs() < 1e-6,
                    "head {h} dim {d}: streaming-full {} != full {}",
                    out[h * hd + d],
                    expect[d]
                );
            }
        }
    }

    #[test]
    fn streaming_only_touches_streaming_heads() {
        // 4 heads, only head 2 streaming → heads 0,1,3 keep their sentinel (untouched).
        let (nq, kv, hd, n) = (4usize, 4usize, 3usize, 5usize);
        let (q, k, v) = make_qkv(nq, kv, hd, n);
        let mut out = vec![42.0f32; nq * hd];
        compute_streaming_attention(&q, &k, &v, &mut out, 1, nq, kv, hd, n, &[2], 1, 1, n - 1, 1);
        for h in [0usize, 1, 3] {
            for d in 0..hd {
                assert_eq!(out[h * hd + d], 42.0, "head {h} must be untouched");
            }
        }
        // head 2 was overwritten (no longer the sentinel).
        assert!(
            (0..hd).any(|d| out[2 * hd + d] != 42.0),
            "head 2 must be recomputed"
        );
    }

    #[test]
    fn streaming_window_restricts_to_sink_and_recent() {
        // sink=1, recent=1, qpos=4 → allowed = {0, 4}. Compare against a hand-built softmax over {0,4}.
        let (nq, kv, hd, n) = (1usize, 1usize, 4usize, 5usize);
        let (q, k, v) = make_qkv(nq, kv, hd, n);
        let mut out = vec![0.0f32; nq * hd];
        compute_streaming_attention(&q, &k, &v, &mut out, 1, nq, kv, hd, n, &[0], 1, 1, n - 1, 1);
        // Reference: softmax over positions {0, 4} only.
        let scale = 1.0 / (hd as f32).sqrt();
        let allowed = [0usize, 4];
        let mut logits: Vec<f32> = allowed
            .iter()
            .map(|&p| (0..hd).map(|d| q[d] * k[p * hd + d]).sum::<f32>() * scale)
            .collect();
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0;
        for l in logits.iter_mut() {
            *l = (*l - m).exp();
            denom += *l;
        }
        let mut expect = vec![0.0f32; hd];
        for (i, &p) in allowed.iter().enumerate() {
            let w = logits[i] / denom;
            for d in 0..hd {
                expect[d] += w * v[p * hd + d];
            }
        }
        for d in 0..hd {
            assert!(
                (out[d] - expect[d]).abs() < 1e-6,
                "dim {d}: {} != {}",
                out[d],
                expect[d]
            );
        }
    }

    #[test]
    fn gqa_expands_kv_head_to_query_group() {
        // n_heads_q=4, n_heads_kv=2 → kv head 1 streaming expands to query heads {2,3}.
        // gate file: layer0 = [1.0, 0.0] → kv0 retrieval, kv1 streaming (threshold 0.5).
        let dir = std::env::temp_dir();
        let path = dir.join(format!("duo_gqa_{}.tsv", std::process::id()));
        std::fs::write(&path, "1.0\t0.0\n").unwrap();
        let duo = DuoHeads::resolve(Some(path.to_str().unwrap()), 0.5, 4, 16, 1, 4, 2)
            .unwrap()
            .expect("resolves");
        assert_eq!(
            duo.per_layer[0],
            vec![2, 3],
            "kv1 streaming → query heads 2,3"
        );
        assert_eq!(duo.streaming_head_count(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn resolve_none_without_file() {
        assert!(
            DuoHeads::resolve(None, 0.5, 4, 16, 24, 8, 8)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn resolve_rejects_all_retrieval() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("duo_allret_{}.tsv", std::process::id()));
        std::fs::write(&path, "1.0 1.0\n1.0 1.0\n").unwrap();
        // every gate >= threshold → no streaming head → error (would be plain full attention).
        assert!(DuoHeads::resolve(Some(path.to_str().unwrap()), 0.5, 4, 16, 2, 4, 2).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn resolve_rejects_zero_window() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("duo_zerowin_{}.tsv", std::process::id()));
        std::fs::write(&path, "0.0\n").unwrap();
        assert!(DuoHeads::resolve(Some(path.to_str().unwrap()), 0.5, 0, 0, 1, 2, 1).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn resolve_rejects_wrong_row_width() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("duo_badwidth_{}.tsv", std::process::id()));
        std::fs::write(&path, "0.0 1.0 0.0\n").unwrap(); // 3 vals, expected 2
        assert!(DuoHeads::resolve(Some(path.to_str().unwrap()), 0.5, 4, 16, 1, 4, 2).is_err());
        std::fs::remove_file(&path).ok();
    }
}
