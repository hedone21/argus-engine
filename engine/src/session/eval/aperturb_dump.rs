//! Output-perturbation candidate scoring (`--dump aperturb`).
//!
//! For each eval question this prefills the prompt onto a fresh, uncompressed F32 cache with the
//! trailing query-row capture armed, then scores a pool of compression candidates with
//! [`crate::aperturb`] and writes one JSONL record per question. It is a standalone pass over its
//! own cache, so scoring (NLL / MC) is byte-identical whether it runs or not (`INV-147`) — by
//! construction, not by a branch in the hot path.
//!
//! The decision point is immediately after prefill, with nothing generated yet. That is the
//! reference protocol's canonical setting and also its hardest one: it is the moment with the least
//! information about what the model is about to do, so it tests the metric's discrimination rather
//! than flattering it.
//!
//! ## What the pool is, and what it is not
//!
//! The candidates here are **shape rules, not the published techniques**: keep everything, keep a
//! recent window, keep a sink plus a recent window, keep every fourth position. They are
//! content-independent, so an external check can reproduce the exact same retained sets from the
//! prompt length alone and compare the metric — which is precisely the scope this dump serves.
//! Scoring the engine's own eviction plugins is a different question (their retained sets are known
//! to diverge from the reference implementations, and one of the reference arms cannot be
//! represented by this cache at all), and it is deliberately not answered here.
//!
//! ## Tensors
//!
//! With `--aperturb-tensor-dir` the pass also writes, per question, the exact `(Q rows, K, V)` it
//! measured. That is what lets an external check separate two questions the scores alone cannot:
//! whether the metric agrees given the same tensors, and whether the engine's forward produces the
//! same tensors in the first place.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::aperturb::{self, Config, Geom, KeepSets, LayerSource, OutputBasis, Readout};
use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::{Buffer, DType};
use crate::kv::kv_cache::{KVCache, KVLayout};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::shape::Shape;
use crate::tensor::Tensor;

use super::dump::{DUMP_APERTURB, JsonlDumpWriter};
use super::fmt_bridge::EvalCacheKind;
use super::output::EvalQuestion;

/// Trailing query rows scored. The reference's canonical value; doubling it doubles the cost and
/// measurably changes nothing.
pub const APERTURB_ROWS: usize = 16;

/// Rank of the output-projection truncation, as a fraction of the projection's input width.
pub const APERTURB_WO_FRAC: f64 = 1.0 / 128.0;

/// Schema version of the `aperturb` record.
pub const APERTURB_SCHEMA_VERSION: u32 = 1;

/// One candidate's scores.
#[derive(Debug, Serialize)]
pub struct ArmRecord {
    pub name: String,
    /// Retained positions summed over every layer and KV head — the byte proxy the budget is
    /// defined on.
    pub kept_total: usize,
    /// Query rows that saw a retained key in every layer and head. Two scores are comparable only
    /// when this matches.
    pub n_cells: usize,
    /// `true` if per-head lengths differ within a layer: measurable, but this cache cannot hold it.
    pub ragged: bool,
    /// Keyed by the reference's own column suffix (`l2`, `l2_rms`, `dcos`, `dcos_rms`).
    pub scores: std::collections::BTreeMap<String, f32>,
}

/// One JSONL record, per question.
#[derive(Debug, Serialize)]
pub struct AperturbRecord {
    pub kind: &'static str,
    pub schema_version: u32,
    pub question_id: String,
    pub prompt_len: usize,
    pub rows: usize,
    /// Absolute position of the first scored query row.
    pub first_row_pos: usize,
    pub wo_frac: f64,
    pub wo_rank: usize,
    /// Worst eigenproblem residual over the layers — the evidence the truncation converged.
    pub wo_residual_max: f64,
    /// The readout the winner was chosen by.
    pub readout: String,
    pub winner: String,
    pub arms: Vec<ArmRecord>,
}

/// Host-resident tensors for one question's decision point.
struct Snapshot {
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl LayerSource for Snapshot {
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

/// The candidate pool for a context of `n` tokens.
///
/// Shape rules only — see the module header. Each returns positions in ascending order, and every
/// rule keeps the last position, without which the trailing query rows would have nothing to attend
/// to in their own layer.
fn build_pool(n_layers: usize, n_kv_heads: usize, n: usize) -> Vec<(String, KeepSets)> {
    let mut pool = vec![(
        "keep_all".to_string(),
        KeepSets::identity(n_layers, n_kv_heads, n),
    )];
    for pct in [50usize, 25] {
        let take = ((n * pct) / 100).max(1);
        let keep: Vec<u32> = ((n - take) as u32..n as u32).collect();
        pool.push((
            format!("recent_r{pct}"),
            KeepSets::uniform(n_layers, n_kv_heads, &keep),
        ));
    }
    {
        // A sink prefix plus a recent window at the same budget as `recent_r25` — the two differ
        // only in where the retained tokens sit, which is what the metric is being asked to see.
        let take = ((n * 25) / 100).max(1);
        let sink = 4usize.min(take.saturating_sub(1));
        let mut keep: Vec<u32> = (0..sink as u32).collect();
        keep.extend((n - (take - sink)) as u32..n as u32);
        keep.dedup();
        pool.push((
            "sink_recent_r25".to_string(),
            KeepSets::uniform(n_layers, n_kv_heads, &keep),
        ));
    }
    {
        // Every fourth position, plus the last four. Same budget again, spread over the whole
        // context instead of concentrated at one end.
        let mut keep: Vec<u32> = (0..n as u32).step_by(4).collect();
        keep.extend((n.saturating_sub(4)) as u32..n as u32);
        keep.sort_unstable();
        keep.dedup();
        pool.push((
            "stride4".to_string(),
            KeepSets::uniform(n_layers, n_kv_heads, &keep),
        ));
    }
    pool
}

/// Dequantize one layer's resident K and V to host f32, `[n_kv_heads][rows][head_dim]`.
///
/// A device-resident cache is mirrored once and both sides read from that mirror; doing it per side
/// would move the same bytes twice.
fn read_layer_kv(
    cache: &KVCache,
    rows: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::stages::kv::mutation::dequant_snapshot;
    if cache.k_buffer.buffer().is_gpu_buffer() {
        cache.k_buffer.backend().synchronize()?;
        let host = cache.host_snapshot()?;
        Ok((
            dequant_snapshot(&host, rows, n_kv_heads, head_dim, true),
            dequant_snapshot(&host, rows, n_kv_heads, head_dim, false),
        ))
    } else {
        Ok((
            dequant_snapshot(cache, rows, n_kv_heads, head_dim, true),
            dequant_snapshot(cache, rows, n_kv_heads, head_dim, false),
        ))
    }
}

/// `[d_out][d_in]` f32 copies of every layer's output projection.
///
/// A device-resident weight has no host pointer, and a quantized one goes through the shared
/// dequantize floor so the factors see exactly the values the projection itself would apply.
fn read_output_projections(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
) -> Result<(Vec<Vec<f32>>, usize, usize)> {
    let mut out = Vec::with_capacity(model.layers.len());
    let (mut d_out, mut d_in) = (0usize, 0usize);
    for (l, slot) in model.layers.iter().enumerate() {
        let weights = slot.load_weights();
        let wo = &weights.wo;
        let dims = wo.shape().dims();
        let (o, i) = (dims[dims.len() - 2], dims[dims.len() - 1]);
        if l == 0 {
            (d_out, d_in) = (o, i);
        } else if (o, i) != (d_out, d_in) {
            anyhow::bail!("output projection layer {l} is {o}x{i}, layer 0 was {d_out}x{d_in}");
        }
        // A weight's own `Tensor::backend()` is not the session's. On OpenCL the weight carries a
        // `cl_mem` whose host pointer is null, and dispatching the readback through the tensor lands
        // on `Backend::read_buffer`'s default host memcpy, which bails on that null — the OpenCL
        // override that would have done the device→host copy is never reached. The session backend
        // knows how to reach its own memory; `dump_layer_weights_to_dir` reads weights the same way
        // for the same reason. On CPU this is the memcpy the host pointer would have given anyway,
        // including for the zero-copy `ALLOC_HOST_PTR` buffer a CPU-primary run with a GPU secondary
        // holds — which a `is_gpu_buffer` test would have wrongly sent to the device and read as
        // all zeros.
        let nbytes = wo.size();
        let host_buf = crate::memory::host::shared::SharedBuffer::new(nbytes, wo.dtype());
        // SAFETY: `host_buf` was just allocated with exactly `nbytes` bytes; `read_buffer` writes
        // exactly that many and does not retain the pointer past the call.
        let dst = unsafe { std::slice::from_raw_parts_mut(host_buf.as_mut_ptr(), nbytes) };
        backend
            .read_buffer(wo, dst)
            .with_context(|| format!("output projection layer {l}: read to host"))?;
        let host = Tensor::new(
            wo.shape().clone(),
            Arc::new(host_buf),
            Arc::new(CpuBackend::new()),
        );
        let host = &host;
        let owned;
        let f32_t = if host.dtype() == DType::F32 {
            host
        } else {
            owned = crate::format::dequant_to_f32_tensor(host)
                .with_context(|| format!("output projection layer {l}: dequantize"))?;
            &owned
        };
        let v = f32_t.as_slice::<f32>().to_vec();
        // An all-zero projection is not a degenerate model — it is an unreadable weight. Left
        // alone it produces a ranking in which every candidate scores zero and the tie-break picks
        // the first, which looks like a result.
        anyhow::ensure!(
            v.iter().any(|x| *x != 0.0),
            "output projection for layer {l} reads as all zeros ({:?}) — the weight could not be \
             reached, so any score built from it would be meaningless",
            wo.dtype()
        );
        out.push(v);
    }
    Ok((out, d_out, d_in))
}

/// Write one question's measured tensors beside the JSONL, little-endian f32.
fn write_tensors(dir: &Path, id: &str, g: &Geom, s: &Snapshot) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let safe: String = id
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    let path = dir.join(format!("{safe}.bin"));
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
    f.write_all(b"ARGUSAPT")?;
    for x in [
        1u32,
        g.n_layers as u32,
        g.n_heads_q as u32,
        g.n_kv_heads as u32,
        g.head_dim as u32,
        g.current_pos as u32,
        g.rows as u32,
    ] {
        f.write_all(&x.to_le_bytes())?;
    }
    for blocks in [&s.q, &s.k, &s.v] {
        for b in blocks {
            for x in b {
                f.write_all(&x.to_le_bytes())?;
            }
        }
    }
    f.flush()?;
    Ok(())
}

/// Run the `aperturb` dump over `questions`.
#[allow(clippy::too_many_arguments)]
pub fn run_aperturb_dump(
    model: &TransformerModel,
    tokenizer: &tokenizers::Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    questions: &[EvalQuestion],
    max_seq_len: usize,
    vocab_size: usize,
    out_path: &Path,
    tensor_dir: Option<&Path>,
) -> Result<()> {
    let n_layers = model.config.num_hidden_layers;
    let n_heads_q = model.config.num_attention_heads;
    let n_kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let q_dim = n_heads_q * head_dim;

    // One decomposition for the model, before the first question: the factors are a model constant,
    // and building them lazily would charge the whole cost to whichever question happened to be
    // first and report it as that question's measurement.
    let t0 = std::time::Instant::now();
    let (wo, d_out, d_in) = read_output_projections(model, backend)?;
    anyhow::ensure!(
        d_in == q_dim,
        "the output projection takes {d_in} inputs but the query rows are {q_dim} wide"
    );
    let fac_cfg = aperturb::subspace::FactorConfig::default();
    let (basis, reports) = OutputBasis::from_weights(&wo, d_in, d_out, APERTURB_WO_FRAC, &fac_cfg)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    drop(wo);
    let residual_max = reports.iter().map(|r| r.residual).fold(0.0f64, f64::max);
    eprintln!(
        "[dump:aperturb] output-projection rank {} of {d_in} for {n_layers} layers in {:.1}s \
         (worst residual {:.1e}, worst deflation {:.4})",
        basis.rank(),
        t0.elapsed().as_secs_f64(),
        residual_max,
        reports
            .iter()
            .map(|r| r.deflation_ratio)
            .fold(0.0f64, f64::max)
    );

    let mut ref_caches: Vec<KVCache> = crate::session::bin_setup::alloc_standard_kv_caches(
        backend,
        memory.clone(),
        n_layers,
        max_seq_len,
        max_seq_len,
        n_kv_heads,
        head_dim,
        DType::F32,
        KVLayout::SeqMajor,
    )?;

    // The metric is defined on an uncompressed F32 cache, so this pass allocates one. That path was
    // unsound on OpenCL until the prefill flash-attention kernels stopped racing on their local K/V
    // tile: scores came out a few percent wrong and moved between runs of the same binary. Both GPU
    // backends were re-measured on that fix (llama3.2-1b, ten prompts). Each reproduces its own dump
    // byte for byte, and each agrees with the CPU winner and with the CPU candidate ranking on every
    // prompt and all four readouts. Absolute scores land within 2.4e-6 of CPU on OpenCL and 1.2e-3
    // on CUDA — the latter is ordinary CUDA precision, not a KV defect, since its F16 forward parts
    // from CPU by the same amount.

    let mut writer = JsonlDumpWriter::create(out_path)?;
    let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let cfg = Config {
        readout: Readout::default(),
        keep_cells: false,
    };
    let mut skipped = 0usize;

    for question in questions {
        let prompt_ids: Vec<u32> = tokenizer
            .encode(question.prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .to_vec();
        let n = prompt_ids.len();
        if n > max_seq_len {
            eprintln!(
                "[dump:aperturb] {}: prompt too long ({n} > {max_seq_len}), skipping",
                question.id
            );
            skipped += 1;
            continue;
        }
        if n <= APERTURB_ROWS {
            eprintln!(
                "[dump:aperturb] {}: prompt of {n} tokens is not longer than the {APERTURB_ROWS} \
                 scored rows, skipping",
                question.id
            );
            skipped += 1;
            continue;
        }

        for c in &mut ref_caches {
            c.current_pos = 0;
            c.high_water_pos = 0;
        }

        let cpu_buf = Galloc::new().alloc(n * 4, DType::U8)?;
        // SAFETY: allocated exactly `n` u32 words above.
        unsafe {
            let ptr = cpu_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, n);
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, n]), cpu_buf, cpu_backend.clone());
        let input_tensor = backend.copy_from(&cpu_input)?;
        let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
        let mut logits = Tensor::new(
            Shape::new(vec![1, 1, vocab_size]),
            logits_buf,
            backend.clone(),
        );

        let mut capture = crate::inference::q_rows::QRowCapture::new(
            backend.clone(),
            memory.as_ref(),
            n_layers,
            APERTURB_ROWS,
            q_dim,
        )?;

        KVCache::forward_fmt_roundtrip(&mut ref_caches, |fmts| {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: 0,
                fmts,
                backend,
                memory: memory.as_ref(),
                logits_out: &mut logits,
                x_gen: None,
                workspace: None,
                logits_last_only: true,
                score_accumulator: None,
                query_stats_accumulator: None,
                skip_config: None,
                importance_collector: None,
                cache_self_need_scores: false,
                layer_boundary_hook: None,
                read_stage: None,
                prefill_attn: None,
                prefill_attn_per_row: None,
                head_mask: None,
                duo_heads: None,
                q_rows: Some(&mut capture),
            })
        })?;

        let q_snap = capture.snapshot(n)?;
        let mut snap = Snapshot {
            q: Vec::with_capacity(n_layers),
            k: Vec::with_capacity(n_layers),
            v: Vec::with_capacity(n_layers),
        };
        for (l, cache) in ref_caches.iter().enumerate() {
            snap.q.push(q_snap.layer_head_major(l, head_dim));
            let (k, v) = read_layer_kv(cache, n, n_kv_heads, head_dim)?;
            snap.k.push(k);
            snap.v.push(v);
        }

        let g = Geom {
            n_layers,
            n_heads_q,
            n_kv_heads,
            head_dim,
            current_pos: n,
            rows: q_snap.rows,
        };
        if let Some(dir) = tensor_dir {
            write_tensors(dir, &question.id, &g, &snap)?;
        }

        let pool = build_pool(n_layers, n_kv_heads, n);
        let dec = aperturb::decide(&snap, &basis, &pool, g, &cfg)
            .map_err(|e| anyhow::anyhow!("{}: {e}", question.id))?;

        let arms = dec
            .scored
            .iter()
            .zip(&pool)
            .map(|(s, (_, keep))| ArmRecord {
                name: s.name.clone(),
                kept_total: keep.total(),
                n_cells: s.scores.n_cells,
                ragged: s.ragged,
                scores: Readout::ALL
                    .into_iter()
                    .map(|r| (r.suffix().to_string(), s.scores.get(r)))
                    .collect(),
            })
            .collect();

        writer.write_record(&AperturbRecord {
            kind: DUMP_APERTURB,
            schema_version: APERTURB_SCHEMA_VERSION,
            question_id: question.id.clone(),
            prompt_len: n,
            rows: q_snap.rows,
            first_row_pos: q_snap.first_pos,
            wo_frac: APERTURB_WO_FRAC,
            wo_rank: basis.rank(),
            wo_residual_max: residual_max,
            readout: cfg.readout.suffix().to_string(),
            winner: dec.winner().name.clone(),
            arms,
        })?;
    }

    let n = writer.finish()?;
    eprintln!("[dump:aperturb] wrote {n} record(s), skipped {skipped}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rule_keeps_the_last_position_and_stays_ascending() {
        for n in [20usize, 41, 400, 4096] {
            for (name, keep) in build_pool(2, 3, n) {
                keep.validate(n)
                    .unwrap_or_else(|e| panic!("{name} at n={n}: {e}"));
                let h = keep.head(1, 2);
                assert_eq!(
                    *h.last().unwrap() as usize,
                    n - 1,
                    "{name} at n={n} drops the last position, so the newest query row would have \
                     nothing of its own to attend to"
                );
            }
        }
    }

    #[test]
    fn the_budgeted_rules_land_on_the_budget() {
        let n = 400;
        let pool = build_pool(1, 1, n);
        let by = |want: &str| {
            pool.iter()
                .find(|(nm, _)| nm == want)
                .map(|(_, k)| k.head(0, 0).len())
                .unwrap()
        };
        assert_eq!(by("keep_all"), n);
        assert_eq!(by("recent_r50"), 200);
        assert_eq!(by("recent_r25"), 100);
        // The sink variant spends the same budget, just differently placed.
        assert_eq!(by("sink_recent_r25"), 100);
        assert_eq!(by("stride4"), 100 + 3); // every 4th, plus the last four (one already present)
    }

    #[test]
    fn a_context_barely_longer_than_the_window_still_yields_a_pool() {
        let pool = build_pool(1, 1, 17);
        for (name, keep) in &pool {
            keep.validate(17).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
