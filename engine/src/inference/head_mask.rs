//! Head-masking ablation — a causal test for candidate "recall heads".
//!
//! Zeros (or mean-substitutes) named `(layer, head)` attention-output contributions during
//! **real generation**, so the lab can causally test whether candidate recall heads are
//! *necessary* for needle-recall (not merely correlated with it). This mirrors the Wu et al.
//! 2024 retrieval-head masking protocol; the mean-substitution mode addresses the IOI
//! (Wang et al. 2022 §4.2) off-distribution critique of a hard zero.
//!
//! **Hook point.** A masked head's `head_dim`-wide slice of `ws.out_attn` (the concatenated
//! per-query-head attention output) is overwritten *after* `attention_into` and *before* the
//! `wo` projection, in **both** prefill and decode ([`forward_prefill_fmt`] /
//! [`forward_gen_fmt`]). Masking both paths is required: masking only decode would leave the
//! needle's influence intact in the residual stream built during prefill (§2 of the spec).
//!
//! **Scope.** `StandardFormat` / `eviction none` only, on the `cpu`, `cuda`, or `opencl` (Adreno)
//! backend (the buffer exists identically on all three; a GPU backend round-trips `ws.out_attn`
//! through host, mirroring the W-DEVKV attention-out recipe in `kv/standard_format.rs`). Empty mask
//! set (no CLI flag) is byte-identical to current behavior — the per-layer fast path returns before
//! touching the buffer.
//!
//! [`forward_prefill_fmt`]: crate::layers::transformer_layer
//! [`forward_gen_fmt`]: crate::layers::transformer_layer

use crate::backend::Backend;
use crate::tensor::Tensor;
use anyhow::{Result, bail};

/// How a masked head's attention-output slice is substituted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskMode {
    /// Overwrite with zeros (Wu et al. protocol, v1 default).
    Zero,
    /// Overwrite with a per-head mean vector supplied lab-side (IOI off-distribution control).
    Mean,
}

/// One masked query head, plus its substitution vector in [`MaskMode::Mean`].
#[derive(Clone, Debug)]
struct MaskedHead {
    /// Query-head index within the layer (`0..n_heads_q`).
    head: usize,
    /// `Some(head_dim floats)` iff mode is [`MaskMode::Mean`]; `None` in [`MaskMode::Zero`].
    mean: Option<Vec<f32>>,
}

/// A resolved head-mask set. Built once at startup from CLI flags and read-only during the
/// forward pass — the run-constant analogue of a `read_stage` (both CLI-derived, resolved once,
/// threaded as an `Option<&_>` forward arg).
#[derive(Clone, Debug)]
pub struct HeadMask {
    /// `per_layer[l]` = the query heads masked in transformer layer `l`. Almost always empty
    /// for most layers, so [`Self::apply`] short-circuits per layer.
    per_layer: Vec<Vec<MaskedHead>>,
    mode: MaskMode,
    n_heads_q: usize,
    head_dim: usize,
}

impl HeadMask {
    /// Resolve a head-mask set from CLI-derived inputs, or `Ok(None)` when no masking is
    /// requested (both `mask_heads` and `mask_heads_random` unset → byte-identical run).
    ///
    /// `n_layers` / `n_heads_q` / `head_dim` come from the loaded model config; they bound the
    /// random draw and validate named pairs. The two selection flags are mutually exclusive.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        mask_heads: Option<&str>,
        mask_heads_random: Option<usize>,
        mask_seed: u64,
        mode_str: &str,
        means_file: Option<&str>,
        n_layers: usize,
        n_heads_q: usize,
        head_dim: usize,
    ) -> Result<Option<HeadMask>> {
        // No masking requested → byte-identical (mode/means are ignored in this case).
        if mask_heads.is_none() && mask_heads_random.is_none() {
            return Ok(None);
        }
        if mask_heads.is_some() && mask_heads_random.is_some() {
            bail!(
                "--mask-heads and --mask-heads-random are mutually exclusive (pass one: a named \
                 list OR N random pairs)"
            );
        }
        if n_layers == 0 || n_heads_q == 0 {
            bail!("head-masking: model reports 0 layers or 0 query heads");
        }

        let mode = match mode_str {
            "zero" => MaskMode::Zero,
            "mean" => MaskMode::Mean,
            other => bail!("--mask-heads-mode must be 'zero' or 'mean' (got '{other}')"),
        };

        // 1. Resolve the (layer, head) pair set (named list OR seeded random draw).
        let pairs: Vec<(usize, usize)> = if let Some(spec) = mask_heads {
            parse_named_pairs(spec, n_layers, n_heads_q)?
        } else {
            let n = mask_heads_random.expect("checked: exactly one selection flag is set");
            draw_random_pairs(n, mask_seed, n_layers, n_heads_q)?
        };
        if pairs.is_empty() {
            bail!("head-masking: resolved an empty mask set (no valid (layer,head) pairs)");
        }

        // 2. In mean mode, load the per-head substitution vectors and require full coverage.
        let means = match mode {
            MaskMode::Zero => {
                if means_file.is_some() {
                    eprintln!(
                        "[mask-heads] --mask-heads-means ignored in 'zero' mode (only used with \
                         --mask-heads-mode mean)"
                    );
                }
                None
            }
            MaskMode::Mean => {
                let path = means_file.ok_or_else(|| {
                    anyhow::anyhow!(
                        "--mask-heads-mode mean requires --mask-heads-means <file> (per-head mean \
                         vectors)"
                    )
                })?;
                Some(load_means_file(path, head_dim)?)
            }
        };

        // 3. Group into per-layer masked-head lists, attaching mean vectors in mean mode.
        let mut per_layer: Vec<Vec<MaskedHead>> = vec![Vec::new(); n_layers];
        for (l, h) in pairs {
            let mean = match &means {
                None => None,
                Some(map) => Some(map.get(&(l, h)).cloned().ok_or_else(|| {
                    anyhow::anyhow!(
                        "--mask-heads-means is missing a mean vector for masked head \
                                 {l}:{h} (mean mode requires one per masked head)"
                    )
                })?),
            };
            per_layer[l].push(MaskedHead { head: h, mean });
        }

        Ok(Some(HeadMask {
            per_layer,
            mode,
            n_heads_q,
            head_dim,
        }))
    }

    /// The masking substitution mode.
    pub fn mode(&self) -> MaskMode {
        self.mode
    }

    /// Number of masked `(layer, head)` units (for logging).
    pub fn masked_head_count(&self) -> usize {
        self.per_layer.iter().map(Vec::len).sum()
    }

    /// A compact `"l:h,l:h,..."` description of the resolved mask set (for run provenance).
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        for (l, heads) in self.per_layer.iter().enumerate() {
            for mh in heads {
                parts.push(format!("{l}:{}", mh.head));
            }
        }
        parts.join(",")
    }

    /// Overwrite this layer's masked query-head slices in `out_attn` — the concatenated
    /// per-head attention output (`n_rows * n_heads_q * head_dim` f32, one row per
    /// `(batch, step)`) — with zeros or the per-head mean vector, in place.
    ///
    /// No-op (byte-identical) when `layer_idx` has no masked heads. On the CPU backend the host
    /// buffer is mutated directly; on a GPU backend (CUDA or OpenCL/Adreno) the device buffer is
    /// round-tripped through host (`read_buffer` → modify → `write_buffer`) so the subsequent `wo`
    /// matmul, which reads the device buffer, sees the change (mirrors the W-DEVKV attention-out
    /// recipe in `kv/standard_format.rs`; both backends enqueue the read/write blocking on their
    /// in-order compute queue/stream).
    pub fn apply(
        &self,
        layer_idx: usize,
        out_attn: &mut Tensor,
        backend: &dyn Backend,
    ) -> Result<()> {
        let heads = match self.per_layer.get(layer_idx) {
            Some(h) if !h.is_empty() => h,
            // Fast path: unmasked layer (the overwhelmingly common case) → no buffer touch.
            _ => return Ok(()),
        };

        let row = self.n_heads_q * self.head_dim;
        let numel = out_attn.numel();
        if row == 0 || !numel.is_multiple_of(row) {
            bail!(
                "head-masking: out_attn numel {numel} not a multiple of row width {row} \
                 (n_heads_q={}, head_dim={})",
                self.n_heads_q,
                self.head_dim
            );
        }
        let n_rows = numel / row;

        if backend.is_gpu() {
            // GPU (CUDA / OpenCL-Adreno): device-resident buffer. read_buffer syncs + copies D2H
            // (cuda_pc synchronize+D2H; OpenCL blocking enqueue_read on the compute queue); the
            // matching write_buffer pushes the whole buffer H2D on the same stream/queue, so the
            // next `wo` kernel (enqueued after) sees the masked data. Full-buffer write (asserts
            // src.len()==size()); a device-only Adreno buffer (null host ptr) is handled by the
            // OpenCL write_buffer's cl_mem enqueue path, not a host memcpy (INV-191).
            let mut bytes = vec![0u8; out_attn.size()];
            backend.read_buffer(out_attn, &mut bytes)?;
            {
                // SAFETY: `bytes` is a freshly-allocated, 4-byte-aligned Vec<u8> holding the f32
                // buffer image; reinterpreting it as &mut [f32] of len bytes/4 is in-bounds and
                // the &mut [u8] is not aliased for the scope of this block.
                let floats = unsafe {
                    std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, bytes.len() / 4)
                };
                overwrite_rows(floats, n_rows, row, heads, self.head_dim);
            }
            backend.write_buffer(out_attn, &bytes)?;
        } else {
            overwrite_rows(
                out_attn.as_mut_slice::<f32>(),
                n_rows,
                row,
                heads,
                self.head_dim,
            );
        }
        Ok(())
    }
}

/// Overwrite the masked head slices in every row of a host-resident attention-output buffer.
fn overwrite_rows(
    buf: &mut [f32],
    n_rows: usize,
    row: usize,
    heads: &[MaskedHead],
    head_dim: usize,
) {
    for r in 0..n_rows {
        let base = r * row;
        for mh in heads {
            let off = base + mh.head * head_dim;
            let slice = &mut buf[off..off + head_dim];
            match &mh.mean {
                None => slice.fill(0.0),
                Some(mean) => slice.copy_from_slice(mean),
            }
        }
    }
}

/// Parse `"14:3,19:3"` into validated, deduplicated `(layer, head)` pairs.
fn parse_named_pairs(spec: &str, n_layers: usize, n_heads_q: usize) -> Result<Vec<(usize, usize)>> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let (l_str, h_str) = tok.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("--mask-heads entry '{tok}' is not 'layer:head' (e.g. \"14:3\")")
        })?;
        let l: usize = l_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("--mask-heads: bad layer index in '{tok}'"))?;
        let h: usize = h_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("--mask-heads: bad head index in '{tok}'"))?;
        if l >= n_layers {
            bail!("--mask-heads: layer {l} out of range (model has {n_layers} layers)");
        }
        if h >= n_heads_q {
            bail!("--mask-heads: head {h} out of range (model has {n_heads_q} query heads)");
        }
        if !out.contains(&(l, h)) {
            out.push((l, h));
        }
    }
    Ok(out)
}

/// Draw `n` distinct `(layer, head)` pairs from `0..n_layers × 0..n_heads_q`, reproducibly seeded.
///
/// Uses an inline SplitMix64 Fisher–Yates over the flattened pair index space (`l*n_heads_q + h`),
/// so the draw is deterministic from `seed` without pulling `rand` into the hot path. Re-running a
/// larger `n` with the same seed *nests* the smaller draw (the first `min` swaps are identical);
/// the spec does not rely on this, but it is the simplest behavior.
fn draw_random_pairs(
    n: usize,
    seed: u64,
    n_layers: usize,
    n_heads_q: usize,
) -> Result<Vec<(usize, usize)>> {
    let total = n_layers * n_heads_q;
    if n == 0 {
        bail!("--mask-heads-random must be >= 1");
    }
    if n > total {
        bail!("--mask-heads-random {n} exceeds the {total} available (layer,head) pairs");
    }
    let mut idx: Vec<usize> = (0..total).collect();
    let mut state = seed;
    for i in 0..n {
        let j = i + (splitmix64(&mut state) as usize) % (total - i);
        idx.swap(i, j);
    }
    Ok(idx[..n]
        .iter()
        .map(|&k| (k / n_heads_q, k % n_heads_q))
        .collect())
}

/// SplitMix64 step (Vigna). Deterministic, cross-platform — reproducible from a fixed seed.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Load per-head mean vectors from a JSON file `{"14:3": [head_dim floats], ...}`.
/// Every vector must have exactly `head_dim` entries; keys are `"layer:head"`.
fn load_means_file(
    path: &str,
    head_dim: usize,
) -> Result<std::collections::HashMap<(usize, usize), Vec<f32>>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("--mask-heads-means: cannot read '{path}': {e}"))?;
    let raw: std::collections::BTreeMap<String, Vec<f32>> = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("--mask-heads-means: '{path}' is not valid JSON: {e}"))?;
    let mut out = std::collections::HashMap::with_capacity(raw.len());
    for (key, vec) in raw {
        let (l_str, h_str) = key.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("--mask-heads-means: key '{key}' is not 'layer:head'")
        })?;
        let l: usize = l_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("--mask-heads-means: bad layer in key '{key}'"))?;
        let h: usize = h_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("--mask-heads-means: bad head in key '{key}'"))?;
        if vec.len() != head_dim {
            bail!(
                "--mask-heads-means: head {key} has {} floats, expected head_dim={head_dim}",
                vec.len()
            );
        }
        out.insert((l, h), vec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use std::sync::Arc;

    fn cpu() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    /// Build a `[rows, n_heads_q, head_dim]` f32 out_attn filled with a nonzero marker (flat index
    /// + 1) so a test can tell which slices were overwritten.
    fn make_out_attn(
        rows: usize,
        n_heads_q: usize,
        head_dim: usize,
        backend: &Arc<dyn Backend>,
    ) -> Tensor {
        let numel = rows * n_heads_q * head_dim;
        let buf = Arc::new(SharedBuffer::from_vec(vec![0u8; numel * 4], DType::F32));
        let mut t = Tensor::new(
            Shape::new(vec![rows, n_heads_q, head_dim]),
            buf,
            backend.clone(),
        );
        let s = t.as_mut_slice::<f32>();
        for (i, v) in s.iter_mut().enumerate() {
            *v = (i + 1) as f32; // nonzero marker: value = flat index + 1
        }
        t
    }

    #[test]
    fn resolve_none_when_no_flags() {
        let hm = HeadMask::resolve(None, None, 0, "zero", None, 24, 8, 64).unwrap();
        assert!(hm.is_none(), "no flags → no masking (byte-identical)");
    }

    #[test]
    fn resolve_named_pairs_dedup_and_validate() {
        let hm = HeadMask::resolve(Some("14:3, 19:3 ,14:3"), None, 0, "zero", None, 24, 8, 64)
            .unwrap()
            .expect("named mask resolves");
        assert_eq!(hm.masked_head_count(), 2, "duplicate 14:3 deduped");
        assert_eq!(hm.describe(), "14:3,19:3");
        assert_eq!(hm.mode(), MaskMode::Zero);
    }

    #[test]
    fn resolve_rejects_out_of_range() {
        assert!(HeadMask::resolve(Some("99:0"), None, 0, "zero", None, 24, 8, 64).is_err());
        assert!(HeadMask::resolve(Some("0:99"), None, 0, "zero", None, 24, 8, 64).is_err());
        assert!(HeadMask::resolve(Some("14-3"), None, 0, "zero", None, 24, 8, 64).is_err());
    }

    #[test]
    fn resolve_rejects_mutual_exclusion() {
        assert!(HeadMask::resolve(Some("1:1"), Some(3), 0, "zero", None, 24, 8, 64).is_err());
    }

    #[test]
    fn resolve_random_reproducible_and_nesting() {
        let a = HeadMask::resolve(None, Some(5), 42, "zero", None, 24, 8, 64)
            .unwrap()
            .unwrap();
        let b = HeadMask::resolve(None, Some(5), 42, "zero", None, 24, 8, 64)
            .unwrap()
            .unwrap();
        assert_eq!(a.describe(), b.describe(), "same seed → identical draw");
        assert_eq!(a.masked_head_count(), 5);

        let c = HeadMask::resolve(None, Some(5), 43, "zero", None, 24, 8, 64)
            .unwrap()
            .unwrap();
        assert_ne!(
            a.describe(),
            c.describe(),
            "different seed → different draw"
        );

        // Nesting: the 5-draw is a SUBSET of the 10-draw (Fisher-Yates prefix property — swaps
        // 5..10 only touch positions >= 5). `describe()` lists layer-sorted, so the prefix reorders;
        // the drawn *set* still nests.
        let ten = HeadMask::resolve(None, Some(10), 42, "zero", None, 24, 8, 64)
            .unwrap()
            .unwrap();
        let five_set: std::collections::HashSet<String> =
            a.describe().split(',').map(String::from).collect();
        let ten_set: std::collections::HashSet<String> =
            ten.describe().split(',').map(String::from).collect();
        assert!(
            five_set.is_subset(&ten_set),
            "same seed → the 5-draw {five_set:?} nests inside the 10-draw {ten_set:?}"
        );
    }

    #[test]
    fn resolve_random_rejects_too_many() {
        // 2 layers * 3 heads = 6 pairs; asking for 7 must fail.
        assert!(HeadMask::resolve(None, Some(7), 0, "zero", None, 2, 3, 64).is_err());
    }

    #[test]
    fn resolve_mean_requires_means_file() {
        assert!(HeadMask::resolve(Some("1:1"), None, 0, "mean", None, 24, 8, 64).is_err());
    }

    #[test]
    fn apply_zero_masks_only_named_heads() {
        let backend = cpu();
        // 2 rows (batch or steps) × 4 heads × 3 dims.
        let (n_heads_q, head_dim) = (4usize, 3usize);
        let mut out = make_out_attn(2, n_heads_q, head_dim, &backend);
        // Mask head 1 in layer 5 only.
        let hm = HeadMask::resolve(Some("5:1"), None, 0, "zero", None, 8, n_heads_q, head_dim)
            .unwrap()
            .unwrap();

        // Unmasked layer 0 → no change.
        hm.apply(0, &mut out, backend.as_ref()).unwrap();
        assert!(
            out.as_slice::<f32>().iter().all(|&v| v != 0.0),
            "layer 0 unmasked"
        );

        // Masked layer 5 → head 1's slice zeroed in EVERY row, others untouched.
        hm.apply(5, &mut out, backend.as_ref()).unwrap();
        let s = out.as_slice::<f32>();
        let row = n_heads_q * head_dim;
        for r in 0..2 {
            for h in 0..n_heads_q {
                for d in 0..head_dim {
                    let v = s[r * row + h * head_dim + d];
                    if h == 1 {
                        assert_eq!(v, 0.0, "row {r} head {h} dim {d} must be zeroed");
                    } else {
                        assert_ne!(v, 0.0, "row {r} head {h} dim {d} must be untouched");
                    }
                }
            }
        }
    }

    #[test]
    fn apply_mean_substitutes_named_heads() {
        let backend = cpu();
        let (n_heads_q, head_dim) = (4usize, 3usize);
        let mut out = make_out_attn(2, n_heads_q, head_dim, &backend);
        // Hand-build a mean-mode HeadMask (skip the JSON file path).
        let hm = HeadMask {
            per_layer: {
                let mut v = vec![Vec::new(); 8];
                v[5].push(MaskedHead {
                    head: 2,
                    mean: Some(vec![7.0, 8.0, 9.0]),
                });
                v
            },
            mode: MaskMode::Mean,
            n_heads_q,
            head_dim,
        };
        hm.apply(5, &mut out, backend.as_ref()).unwrap();
        let s = out.as_slice::<f32>();
        let row = n_heads_q * head_dim;
        for r in 0..2 {
            let off = r * row + 2 * head_dim;
            assert_eq!(
                &s[off..off + head_dim],
                &[7.0, 8.0, 9.0],
                "row {r} head 2 = mean"
            );
        }
    }

    #[test]
    fn apply_empty_layer_is_noop() {
        let backend = cpu();
        let mut out = make_out_attn(1, 4, 3, &backend);
        let before: Vec<f32> = out.as_slice::<f32>().to_vec();
        let hm = HeadMask::resolve(Some("5:1"), None, 0, "zero", None, 8, 4, 3)
            .unwrap()
            .unwrap();
        hm.apply(3, &mut out, backend.as_ref()).unwrap(); // layer 3 has no masked head
        assert_eq!(
            out.as_slice::<f32>(),
            before.as_slice(),
            "empty layer = byte-identical"
        );
    }
}
