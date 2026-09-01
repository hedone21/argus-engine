//! The output-projection basis the perturbation metric projects through — built once, or loaded.
//!
//! `V_r Σ_r` per layer is a model constant: it depends on `W_o` alone, not on the cache, the
//! prompt, or the budget. Building it is a subspace iteration per layer — 28 s at 1B and twelve
//! minutes at 8B on twenty desktop threads — so a process that scores anything wants to load it,
//! and only a process that is producing one wants to factor it.
//!
//! Shared by the eval dump (`--dump aperturb`) and the engine's own compression choice
//! ([`crate::kv::aperturb_select`]) so both project through the same table, built the same way,
//! digest-checked against the same weights.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::aperturb::{self, OutputBasis};
use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::{Buffer, DType};
use crate::models::transformer::TransformerModel;
use crate::tensor::Tensor;

/// Trailing query rows scored. The reference's canonical value; doubling it doubles the cost and
/// measurably changes nothing.
pub const APERTURB_ROWS: usize = 16;

/// Rank of the output-projection truncation, as a fraction of the projection's input width.
///
/// The reference's current value. It names the metric key the scores are published under
/// (`aperturb_prev16_wo0p390625_l2_rms`), so a reference sweep to compare against has to have been
/// run at this same fraction — at the previous `1/128` it produces a differently named column and
/// the join finds nothing rather than disagreeing.
pub const APERTURB_WO_FRAC: f64 = 1.0 / 256.0;

/// `[d_out][d_in]` f32 copy of one layer's output projection.
///
/// A device-resident weight has no host pointer, and a quantized one goes through the shared
/// dequantize floor so the factors see exactly the values the projection itself would apply.
fn read_output_projection(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    l: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    let weights = model.layers[l].load_weights();
    let wo = &weights.wo;
    let dims = wo.shape().dims();
    let (d_out, d_in) = (dims[dims.len() - 2], dims[dims.len() - 1]);
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
    Ok((v, d_out, d_in))
}

/// Walk every layer's output projection in order, handing each to `f`, and return `(d_out, d_in)`.
///
/// Every layer is read whichever path the basis takes: factoring needs the values, and loading
/// needs their digest, which is what tells a stored basis apart from one built for a different
/// checkpoint of the same shape. What differs is only whether the caller keeps them.
fn walk_output_projections(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    mut f: impl FnMut(Vec<f32>),
) -> Result<(usize, usize)> {
    let (mut d_out, mut d_in) = (0usize, 0usize);
    for l in 0..model.layers.len() {
        let (v, o, i) = read_output_projection(model, backend, l)?;
        if l == 0 {
            (d_out, d_in) = (o, i);
        } else if (o, i) != (d_out, d_in) {
            anyhow::bail!("output projection layer {l} is {o}x{i}, layer 0 was {d_out}x{d_in}");
        }
        f(v);
    }
    Ok((d_out, d_in))
}

/// The basis this model's scores are measured through, and the worst eigenproblem residual behind
/// it — the evidence the truncation converged.
///
/// `basis_in` loads a stored table (and reads `W_o` anyway, to digest it: a table built for another
/// checkpoint of the same shape is refused rather than silently measured against). `basis_out`
/// writes the freshly factored one. Both at once is refused — it would write back what was just
/// read.
pub fn load_or_factor(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    basis_in: Option<&std::path::Path>,
    basis_out: Option<&std::path::Path>,
) -> Result<(OutputBasis, f64)> {
    let n_layers = model.config.num_hidden_layers;
    let q_dim = model.config.num_attention_heads * model.config.head_dim;
    anyhow::ensure!(
        !(basis_in.is_some() && basis_out.is_some()),
        "a basis path to load and a basis path to write were both given; asking for both \
         would write back what was just read"
    );

    // One decomposition for the model, before the first question: the factors are a model constant,
    // and building them lazily would charge the whole cost to whichever question happened to be
    // first and report it as that question's measurement.
    //
    // A model constant that nothing stored, until `--aperturb-basis-out`: every process paid the
    // subspace iteration again, which is 28 s at 1B and 12 min at 8B on twenty desktop threads. The
    // load path pays a read of `W_o` instead — to digest it, so a table from another checkpoint is
    // refused rather than silently measured against — and no arithmetic beyond that.
    let t0 = std::time::Instant::now();
    if let Some(path) = basis_in {
        let mut digest = aperturb::basis_file::Digest::new(n_layers);
        let (_, d_in) = walk_output_projections(model, backend, |v| digest.layer(&v))?;
        anyhow::ensure!(
            d_in == q_dim,
            "the output projection takes {d_in} inputs but the query rows are {q_dim} wide"
        );
        let expect = aperturb::basis_file::Expect {
            n_layers,
            d: d_in,
            rank: OutputBasis::rank_for(APERTURB_WO_FRAC, d_in),
            frac: APERTURB_WO_FRAC,
            wo_digest: digest.finish(),
        };
        let (basis, residual_max) =
            aperturb::basis_file::read(path, &expect).map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!(
            "[aperturb] output-projection rank {} of {d_in} for {n_layers} layers loaded from \
             {} in {:.1}s (worst residual {:.1e})",
            basis.rank(),
            path.display(),
            t0.elapsed().as_secs_f64(),
            residual_max,
        );
        Ok((basis, residual_max))
    } else {
        let mut wo: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        let (d_out, d_in) = walk_output_projections(model, backend, |v| wo.push(v))?;
        anyhow::ensure!(
            d_in == q_dim,
            "the output projection takes {d_in} inputs but the query rows are {q_dim} wide"
        );
        let wo_digest = aperturb::basis_file::digest(&wo);
        let fac_cfg = aperturb::subspace::FactorConfig::default();
        let (basis, reports) =
            OutputBasis::from_weights(&wo, d_in, d_out, APERTURB_WO_FRAC, &fac_cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        drop(wo);
        let residual_max = reports.iter().map(|r| r.residual).fold(0.0f64, f64::max);
        eprintln!(
            "[aperturb] output-projection rank {} of {d_in} for {n_layers} layers in {:.1}s \
             (worst residual {:.1e}, worst deflation {:.4})",
            basis.rank(),
            t0.elapsed().as_secs_f64(),
            residual_max,
            reports
                .iter()
                .map(|r| r.deflation_ratio)
                .fold(0.0f64, f64::max)
        );
        if let Some(path) = basis_out {
            aperturb::basis_file::write(path, &basis, residual_max, wo_digest)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            eprintln!(
                "[aperturb] wrote the basis to {} ({} bytes) — pass it back with \
                 --aperturb-basis to skip the decomposition",
                path.display(),
                aperturb::basis_file::HEADER_BYTES + n_layers * d_in * basis.rank() * 4,
            );
        }
        Ok((basis, residual_max))
    }
}
