//! `QuantNoiseTable` — per-layer quantization noise factor table (ENG-DAT-095).
//!
//! Holds the per-decoder-layer ε_i values computed once at engine init from the
//! primary vs. secondary weight comparison (ENG-ALG-216).  After construction
//! the table is **read-only**; share it with `Arc<QuantNoiseTable>`.
//!
//! `WeightSwapDecider` (Stage B) is the primary consumer.
//!
//! Spec: ENG-DAT-095, ENG-ALG-216, INV-127.

use std::sync::Arc;
use std::time::Instant;

// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference weight resource (LayerSlot/SecondaryMmap)
use crate::models::weights::secondary_mmap::SecondaryMmap;
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O pressure orchestrator → inference weight resource (LayerSlot)
use crate::models::weights::slot::LayerSlot;
use crate::quant::{BlockQ4_0, QK4_0};
use crate::runtime_resources_access::QuantNoiseAccess;

// ── QuantNoiseTable ───────────────────────────────────────────────────────────

/// Per-decoder-layer quantization noise factor table (ENG-DAT-095).
///
/// Each entry `per_layer[i] = ε_i` represents the mean relative Frobenius
/// squared error between the primary (F16/F32) and the secondary (Q4_0)
/// weights for decoder layer `i`.  A larger ε_i means the secondary weights
/// diverge more from the primary — that layer is a more "expensive" swap
/// candidate.
///
/// `f32::NAN` entries signal a computation failure for that layer; `epsilon(i)`
/// returns `None` for them, and `WeightSwapDecider` must exclude such layers
/// from the candidate set (INV-127).
pub struct QuantNoiseTable {
    /// Per-decoder-layer ε.  Index = layer_id (decoder layers only).
    per_layer: Vec<f32>,
    /// `true` when built via `new_from_frobenius` (eager path).
    /// `false` for the fallback path (`uniform_ones`).
    computed_at_init: bool,
}

impl QuantNoiseTable {
    /// Construct an **empty** table — used when no secondary is present.
    ///
    /// `len() == 0`, `is_computed() == false`.  The dispatch layer should
    /// reject `SwapWeights` before the decider is invoked in this state.
    pub fn empty() -> Self {
        Self {
            per_layer: Vec::new(),
            computed_at_init: false,
        }
    }

    /// Uniform-ones fallback (ENG-ALG-216 "전체 실패" path).
    ///
    /// All `ε_i = 1.0`, `computed_at_init = false`.  The decider falls back to
    /// importance-only ordering (importance × 1.0 = importance) for all layers.
    pub fn uniform_ones(num_layers: usize) -> Self {
        Self {
            per_layer: vec![1.0f32; num_layers],
            computed_at_init: false,
        }
    }

    /// Compute the table via Frobenius relative error (ENG-ALG-216).
    ///
    /// Iterates all decoder layers present in `secondary`, computes the
    /// mean relative Frobenius squared error across the 7 weight tensors
    /// {Q, K, V, O, gate, up, down} and stores the result in `per_layer[i]`.
    ///
    /// Individual tensor failures (shape mismatch, dequant error) set
    /// `ε_t = 1.0` for that tensor but do not abort the layer.  Layers where
    /// **all** tensors fail get `ε_i = f32::NAN` (INV-127).
    ///
    /// If the entire computation fails (e.g. `secondary` has no layers at
    /// all), the caller should fall back to `QuantNoiseTable::uniform_ones`.
    pub fn new_from_frobenius(
        primary_slots: &[Arc<LayerSlot>],
        secondary: &Arc<SecondaryMmap>,
    ) -> Self {
        let n = primary_slots.len();
        if n == 0 {
            return Self::empty();
        }

        let start = Instant::now();
        let mut per_layer = vec![f32::NAN; n];

        // Tensor sub-names for decoder layers (norm tensors excluded —
        // they are F32 and not quantized in secondary, see ENG-ALG-216).
        const TENSOR_SUBNAMES: &[&str] = &[
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];

        for (layer_idx, slot) in primary_slots.iter().enumerate() {
            let weights = slot.load_weights();
            let primary_tensors = [
                &weights.wq,
                &weights.wk,
                &weights.wv,
                &weights.wo,
                &weights.w_gate,
                &weights.w_up,
                &weights.w_down,
            ];

            let mut layer_sum = 0.0f64;
            let mut valid_tensors = 0usize;

            for (&subname, primary) in TENSOR_SUBNAMES.iter().zip(primary_tensors.iter()) {
                let epsilon_t =
                    compute_tensor_epsilon(layer_idx, subname, primary, secondary.as_ref());
                match epsilon_t {
                    Some(e) => {
                        layer_sum += e as f64;
                        valid_tensors += 1;
                    }
                    None => {
                        // Fallback: treat as ε_t = 1.0 for the layer sum.
                        // (ENG-ALG-216: "dequantize 실패 시 ε_t = 1.0, warn log")
                        layer_sum += 1.0;
                        valid_tensors += 1;
                    }
                }
            }

            if valid_tensors == 0 {
                // All tensors failed — mark layer as NaN (INV-127).
                per_layer[layer_idx] = f32::NAN;
            } else {
                per_layer[layer_idx] = (layer_sum / valid_tensors as f64) as f32;
            }

            // Progress log (SHOULD per ENG-ALG-216).
            if layer_idx % 4 == 0 || layer_idx + 1 == n {
                let avg: f32 = per_layer[..=layer_idx]
                    .iter()
                    .filter(|v| v.is_finite())
                    .copied()
                    .sum::<f32>()
                    / (layer_idx + 1) as f32;
                log::info!("ε calc: layer {}/{} done, avg={:.6}", layer_idx + 1, n, avg);
            }
        }

        let elapsed_ms = start.elapsed().as_millis();
        let any_fallback = per_layer.iter().any(|v| !v.is_finite());
        log::info!("ε calc done in {}ms, fallback={}", elapsed_ms, any_fallback);

        Self {
            per_layer,
            computed_at_init: true,
        }
    }

    /// Return `Some(ε_i)` if the entry is a valid finite value, else `None`.
    ///
    /// `None` means the layer's ε was not computed (NaN) and the decider must
    /// exclude it from the candidate set (INV-127).
    pub fn epsilon(&self, layer_id: usize) -> Option<f32> {
        let v = *self.per_layer.get(layer_id)?;
        if v.is_finite() { Some(v) } else { None }
    }

    /// Number of decoder layers in the table.
    pub fn len(&self) -> usize {
        self.per_layer.len()
    }

    /// `true` when the table was built via `new_from_frobenius` (eager path).
    pub fn is_computed(&self) -> bool {
        self.computed_at_init
    }

    /// Raw slice of ε values (including possible NaN entries).
    pub fn as_slice(&self) -> &[f32] {
        &self.per_layer
    }

    /// `true` when the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.per_layer.is_empty()
    }

    /// Test-only constructor: build a table with arbitrary ε values and
    /// `computed_at_init = true` so unit tests can exercise the scored path
    /// without a real secondary mmap.
    #[cfg(test)]
    pub fn new_test(values: Vec<f32>) -> Self {
        Self {
            per_layer: values,
            computed_at_init: true,
        }
    }

    /// Integration-test constructor: build a table with explicit ε values and
    /// `computed_at_init = true`.
    ///
    /// Identical to `new_test` but accessible from integration test binaries
    /// (outside the crate).  Prefer `uniform_ones` for production fallback paths.
    #[doc(hidden)]
    pub fn from_values(values: Vec<f32>) -> Self {
        Self {
            per_layer: values,
            computed_at_init: true,
        }
    }
}

/// Cross-cutting trait impl: lets inference owner / pressure consumers interact
/// with the table via `dyn QuantNoiseAccess` (§13.8-O 본질 해소).
impl QuantNoiseAccess for QuantNoiseTable {
    fn epsilon(&self, layer_id: usize) -> Option<f32> {
        QuantNoiseTable::epsilon(self, layer_id)
    }

    fn len(&self) -> usize {
        QuantNoiseTable::len(self)
    }

    fn is_computed(&self) -> bool {
        QuantNoiseTable::is_computed(self)
    }

    fn as_slice(&self) -> &[f32] {
        QuantNoiseTable::as_slice(self)
    }

    fn is_empty(&self) -> bool {
        QuantNoiseTable::is_empty(self)
    }
}

/// Compute the per-layer quantization noise table for a freshly loaded model.
///
/// Pressure-side relocation of `compute_quant_noise_for_model` (formerly
/// inference-side, §13.8-O 우선순위 #2 — design doc `arch/weights_pressure_split.md
/// §7.4`).  The inference loader / test helpers call this *after* the
/// `TransformerModel` ctor has installed the empty placeholder from
/// [`super::setup::setup_runtime_resources`].
///
/// - `primary_slots` — the model's decoder `LayerSlot`s (typically
///   `model.layers.as_slice()`).
/// - `secondary` — when `None`, returns the empty table (no log).
///
/// On failure (no decoder layers / all-NaN per-layer ε), falls back to
/// `QuantNoiseTable::uniform_ones(n)` with a warning log (ENG-ALG-216).
pub fn compute_quant_noise(
    primary_slots: &[Arc<LayerSlot>],
    secondary: Option<&Arc<SecondaryMmap>>,
) -> Arc<dyn QuantNoiseAccess> {
    let secondary = match secondary {
        Some(s) => s,
        None => return Arc::new(QuantNoiseTable::empty()),
    };

    let n = primary_slots.len();
    if n == 0 {
        log::warn!("ε calc: no decoder layers — returning empty QuantNoiseTable");
        return Arc::new(QuantNoiseTable::empty());
    }

    let table = QuantNoiseTable::new_from_frobenius(primary_slots, secondary);

    if !table.is_computed() {
        // new_from_frobenius returned with computed_at_init=false only
        // when n==0, which we already handled above.
        log::warn!("ε calc: new_from_frobenius returned fallback — using uniform_ones");
        return Arc::new(QuantNoiseTable::uniform_ones(n));
    }

    // Check if all layers failed (all NaN) — ENG-ALG-216 "전체 실패" path.
    let all_nan = table.as_slice().iter().all(|v| !v.is_finite());
    if all_nan {
        log::warn!(
            "ε calc: all {} layers failed — falling back to uniform_ones",
            n
        );
        Arc::new(QuantNoiseTable::uniform_ones(n))
    } else {
        Arc::new(table)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute ε_t for a single tensor in a single layer.
///
/// Returns `Some(ε_t)` on success, `None` when the secondary tensor is
/// absent, has a shape mismatch, or the dequantization produced invalid data.
fn compute_tensor_epsilon(
    layer_idx: usize,
    subname: &str,
    primary: &crate::tensor::Tensor,
    secondary: &SecondaryMmap,
) -> Option<f32> {
    let info = secondary.layer_tensor(layer_idx, subname)?;

    // Verify that the secondary dtype is Q4_0 before dequantizing.
    use crate::buffer::DType;
    if info.dtype != DType::Q4_0 {
        log::warn!(
            "ε calc: layer {} '{}' secondary dtype is {:?}, not Q4_0 — skipping",
            layer_idx,
            subname,
            info.dtype
        );
        return None;
    }

    // Number of f32 elements in the primary tensor.
    let numel = primary.shape().numel();
    if numel == 0 {
        return Some(0.0);
    }

    // Number of Q4_0 blocks expected.
    let n_blocks = numel / QK4_0;
    if !numel.is_multiple_of(QK4_0) || n_blocks == 0 {
        log::warn!(
            "ε calc: layer {} '{}' numel {} is not divisible by QK4_0={}",
            layer_idx,
            subname,
            numel,
            QK4_0
        );
        return None;
    }

    let expected_bytes = n_blocks * std::mem::size_of::<BlockQ4_0>();
    let raw = secondary.tensor_bytes(info);
    if raw.len() != expected_bytes {
        log::warn!(
            "ε calc: layer {} '{}' expected {} bytes, got {}",
            layer_idx,
            subname,
            expected_bytes,
            raw.len()
        );
        return None;
    }

    // Dequantize secondary Q4_0 blocks to f32 (extracted helper for reuse).
    let secondary_f32 = dequantize_q4_0_blocks(raw, numel);

    // Read primary weights as f32.
    // The primary tensor may be F16 or F32.  Use the buffer's `as_f32_slice`
    // abstraction if available; otherwise fall through to the f16→f32 path.
    let primary_f32: Vec<f32> = primary_tensor_to_f32(primary)?;
    if primary_f32.len() != numel {
        log::warn!(
            "ε calc: layer {} '{}' primary f32 len {} != numel {}",
            layer_idx,
            subname,
            primary_f32.len(),
            numel
        );
        return None;
    }

    // Frobenius relative squared error:
    //   num   = || W_primary - W_secondary_fp ||_F²
    //   denom = || W_primary ||_F²
    //   ε_t   = num / max(denom, EPS)
    const EPS: f64 = 1e-12;
    let (mut num, mut denom) = (0.0f64, 0.0f64);
    for (p, s) in primary_f32.iter().zip(secondary_f32.iter()) {
        let diff = (*p as f64) - (*s as f64);
        num += diff * diff;
        denom += (*p as f64) * (*p as f64);
    }
    let epsilon_t = (num / denom.max(EPS)) as f32;
    Some(epsilon_t)
}

/// Dequantize a contiguous slice of Q4_0 blocks to a fresh `Vec<f32>`.
///
/// `numel` must equal `n_blocks * QK4_0` where `n_blocks = raw.len() / sizeof(BlockQ4_0)`.
/// Caller is responsible for validating that `raw.len()` matches expectation.
fn dequantize_q4_0_blocks(raw: &[u8], numel: usize) -> Vec<f32> {
    let n_blocks = numel / QK4_0;
    debug_assert_eq!(raw.len(), n_blocks * std::mem::size_of::<BlockQ4_0>());

    // SAFETY: BlockQ4_0 has repr(C), size=18 B. `raw` is a contiguous byte
    // slice from the mmap whose length is exactly `n_blocks * 18`. GGUF
    // tensors are 32-byte aligned by spec; BlockQ4_0 requires 2-byte
    // alignment (the f16 field).
    let blocks: &[BlockQ4_0] =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const BlockQ4_0, n_blocks) };

    let mut out = vec![0.0f32; numel];
    for (bi, block) in blocks.iter().enumerate() {
        let start = bi * QK4_0;
        // SAFETY: out was allocated with exactly n_blocks * QK4_0 elements.
        let chunk: &mut [f32; QK4_0] =
            unsafe { &mut *(out[start..start + QK4_0].as_mut_ptr() as *mut [f32; QK4_0]) };
        block.dequantize(chunk);
    }
    out
}

/// Layer-wise input-aware quantization perturbation (DP-LLM proxy, NeurIPS 2025 inspired).
///
/// For each layer `i`, compute
///
///   ε_dpllm[i] = ‖(W_F16 − W_Q4) · x_mean[i]‖₂ / max(‖W_F16 · x_mean[i]‖₂, EPS)
///
/// using the layer's `attn_output.weight` (`wo`) tensor.  We pick `wo` because
/// it is hidden_size × hidden_size, so the layer-entry mean-pooled hidden state
/// `x_mean[i]` (length = hidden_size) is directly compatible.  This is a
/// simplified single-tensor proxy; alternatives (ffn_down) would require
/// caching the actual MLP input vector and add ~6× the cost.
///
/// Returns a vector of length `primary_slots.len()`.  Entries are `f32::NAN`
/// for layers where the secondary tensor is missing, the shape is unexpected,
/// or `x_mean` length disagrees with `wo`'s input dim.  When
/// `x_means.len() != primary_slots.len()`, every entry is NaN.
pub fn compute_input_aware_epsilon(
    primary_slots: &[Arc<LayerSlot>],
    secondary: &Arc<SecondaryMmap>,
    x_means: &[Vec<f32>],
) -> Vec<f32> {
    use crate::buffer::DType;
    let n = primary_slots.len();
    let mut result = vec![f32::NAN; n];
    if x_means.len() != n {
        log::warn!(
            "compute_input_aware_epsilon: x_means len {} != slots len {}",
            x_means.len(),
            n
        );
        return result;
    }

    const SUBNAME: &str = "attn_output.weight";
    const EPS: f64 = 1e-12;

    for (i, (slot, x_mean)) in primary_slots.iter().zip(x_means.iter()).enumerate() {
        let weights = slot.load_weights();
        let primary_tensor = &weights.wo;
        let dims = primary_tensor.shape().dims();
        if dims.len() != 2 {
            log::warn!(
                "compute_input_aware_epsilon: layer {} wo is not 2D (dims={:?})",
                i,
                dims
            );
            continue;
        }
        let (out_dim, in_dim) = (dims[0], dims[1]);
        if x_mean.len() != in_dim {
            log::warn!(
                "compute_input_aware_epsilon: layer {} x_mean len {} != in_dim {}",
                i,
                x_mean.len(),
                in_dim
            );
            continue;
        }

        let info = match secondary.layer_tensor(i, SUBNAME) {
            Some(info) => info,
            None => continue,
        };
        if info.dtype != DType::Q4_0 {
            continue;
        }
        let numel = primary_tensor.shape().numel();
        if numel != out_dim * in_dim || !numel.is_multiple_of(QK4_0) {
            continue;
        }
        let n_blocks = numel / QK4_0;
        let expected_bytes = n_blocks * std::mem::size_of::<BlockQ4_0>();
        let raw = secondary.tensor_bytes(info);
        if raw.len() != expected_bytes {
            continue;
        }
        let secondary_f32 = dequantize_q4_0_blocks(raw, numel);

        let primary_f32 = match primary_tensor_to_f32(primary_tensor) {
            Some(v) => v,
            None => continue,
        };
        if primary_f32.len() != numel {
            continue;
        }

        // Row-major GGUF layout: row r → primary_f32[r*in_dim .. (r+1)*in_dim].
        // y_p[r] = Σ_c W_p[r,c] · x[c],   diff_y[r] = Σ_c (W_p[r,c] − W_q[r,c]) · x[c]
        let mut diff_sq = 0.0f64;
        let mut prim_sq = 0.0f64;
        for r in 0..out_dim {
            let row_start = r * in_dim;
            let mut y_p = 0.0f32;
            let mut diff_y = 0.0f32;
            for c in 0..in_dim {
                let w_p = primary_f32[row_start + c];
                let w_s = secondary_f32[row_start + c];
                let x = x_mean[c];
                y_p += w_p * x;
                diff_y += (w_p - w_s) * x;
            }
            diff_sq += (diff_y as f64).powi(2);
            prim_sq += (y_p as f64).powi(2);
        }
        let eps_val = (diff_sq.sqrt() / prim_sq.max(EPS).sqrt()) as f32;
        result[i] = eps_val;
    }

    result
}

/// Layer-wise input-aware perturbation, **single-tensor absolute** variant.
///
/// Identical to `compute_input_aware_epsilon` but drops the `‖W_p · x‖`
/// normalisation in the denominator, yielding the raw absolute L2 norm of
/// the activation difference:
///
///   ε_abs[i] = ‖(W_F16 − W_Q4) · x_mean[i]‖₂
///
/// This is the §4 candidate "D" — signal-scaled rather than relative.
pub fn compute_input_aware_epsilon_absolute(
    primary_slots: &[Arc<LayerSlot>],
    secondary: &Arc<SecondaryMmap>,
    x_means: &[Vec<f32>],
) -> Vec<f32> {
    use crate::buffer::DType;
    let n = primary_slots.len();
    let mut result = vec![f32::NAN; n];
    if x_means.len() != n {
        log::warn!(
            "compute_input_aware_epsilon_absolute: x_means len {} != slots len {}",
            x_means.len(),
            n
        );
        return result;
    }

    const SUBNAME: &str = "attn_output.weight";

    for (i, (slot, x_mean)) in primary_slots.iter().zip(x_means.iter()).enumerate() {
        let weights = slot.load_weights();
        let primary_tensor = &weights.wo;
        let dims = primary_tensor.shape().dims();
        if dims.len() != 2 {
            continue;
        }
        let (out_dim, in_dim) = (dims[0], dims[1]);
        if x_mean.len() != in_dim {
            continue;
        }

        let info = match secondary.layer_tensor(i, SUBNAME) {
            Some(info) => info,
            None => continue,
        };
        if info.dtype != DType::Q4_0 {
            continue;
        }
        let numel = primary_tensor.shape().numel();
        if numel != out_dim * in_dim || !numel.is_multiple_of(QK4_0) {
            continue;
        }
        let n_blocks = numel / QK4_0;
        let expected_bytes = n_blocks * std::mem::size_of::<BlockQ4_0>();
        let raw = secondary.tensor_bytes(info);
        if raw.len() != expected_bytes {
            continue;
        }
        let secondary_f32 = dequantize_q4_0_blocks(raw, numel);
        let primary_f32 = match primary_tensor_to_f32(primary_tensor) {
            Some(v) => v,
            None => continue,
        };
        if primary_f32.len() != numel {
            continue;
        }

        let mut diff_sq = 0.0f64;
        for r in 0..out_dim {
            let row_start = r * in_dim;
            let mut diff_y = 0.0f32;
            for c in 0..in_dim {
                let w_p = primary_f32[row_start + c];
                let w_s = secondary_f32[row_start + c];
                diff_y += (w_p - w_s) * x_mean[c];
            }
            diff_sq += (diff_y as f64).powi(2);
        }
        result[i] = diff_sq.sqrt() as f32;
    }
    result
}

/// Layer-wise input-aware perturbation, **QCF-style multiplicative** variant.
///
/// For each layer `i`, compose the relative L2 perturbations of two
/// attention-block weights:
///
///   ε_qcf[i] = ε_rel(W_v_i, x_mean[i]) × ε_rel(W_o_i, x_mean[i])
///
/// where  `ε_rel(W, x) = ‖(W_F16 − W_Q4) · x‖ / ‖W_F16 · x‖`.
///
/// Decomposes the runtime QCF/caote attention-output perturbation
/// `‖ΔO‖ / ‖O‖` into two static weight-space factors.  Layers with
/// either factor unavailable yield `NaN`.
pub fn compute_input_aware_epsilon_qcf(
    primary_slots: &[Arc<LayerSlot>],
    secondary: &Arc<SecondaryMmap>,
    x_means: &[Vec<f32>],
) -> Vec<f32> {
    let n = primary_slots.len();
    let mut result = vec![f32::NAN; n];
    if x_means.len() != n {
        log::warn!(
            "compute_input_aware_epsilon_qcf: x_means len {} != slots len {}",
            x_means.len(),
            n
        );
        return result;
    }

    for (i, (slot, x_mean)) in primary_slots.iter().zip(x_means.iter()).enumerate() {
        let weights = slot.load_weights();
        let eps_v = match input_aware_relative_epsilon(
            i,
            "attn_v.weight",
            &weights.wv,
            secondary,
            x_mean,
        ) {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        let eps_o = match input_aware_relative_epsilon(
            i,
            "attn_output.weight",
            &weights.wo,
            secondary,
            x_mean,
        ) {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        result[i] = eps_v * eps_o;
    }
    result
}

/// Layer-wise input-aware perturbation, **multi-tensor relative** variant.
///
/// For each layer `i`, sum the relative L2 error across six weight tensors
/// whose input dim matches the layer-entry hidden state `x_mean[i]`:
///
///   q, k, v, o (attention), gate, up (MLP first stage)
///
/// `w_down` (input dim = intermediate_size) is skipped — its true input is
/// the MLP mid-state `act(W_gate·x) ⊙ (W_up·x)` and reusing `x_mean` would
/// be a shape mismatch.  The sum is over min(6, valid) finite per-tensor ε
/// contributions; layers with zero valid tensors yield `NaN`.
pub fn compute_input_aware_epsilon_multitensor(
    primary_slots: &[Arc<LayerSlot>],
    secondary: &Arc<SecondaryMmap>,
    x_means: &[Vec<f32>],
) -> Vec<f32> {
    let n = primary_slots.len();
    let mut result = vec![f32::NAN; n];
    if x_means.len() != n {
        log::warn!(
            "compute_input_aware_epsilon_multitensor: x_means len {} != slots len {}",
            x_means.len(),
            n
        );
        return result;
    }

    const SUBNAMES: [&str; 6] = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
    ];

    for (i, (slot, x_mean)) in primary_slots.iter().zip(x_means.iter()).enumerate() {
        let weights = slot.load_weights();
        let tensors: [&crate::tensor::Tensor; 6] = [
            &weights.wq,
            &weights.wk,
            &weights.wv,
            &weights.wo,
            &weights.w_gate,
            &weights.w_up,
        ];

        let mut acc = 0.0f64;
        let mut valid = 0usize;
        for (subname, primary_tensor) in SUBNAMES.iter().zip(tensors.iter()) {
            if let Some(eps_rel) =
                input_aware_relative_epsilon(i, subname, primary_tensor, secondary, x_mean)
                && eps_rel.is_finite()
            {
                acc += eps_rel as f64;
                valid += 1;
            }
        }

        if valid > 0 {
            result[i] = acc as f32;
        }
    }
    result
}

/// Helper: compute relative input-aware ε for one (layer, tensor) pair.
///
/// Returns `None` on dim/dtype/shape mismatch or missing secondary entry.
fn input_aware_relative_epsilon(
    layer_idx: usize,
    subname: &str,
    primary_tensor: &crate::tensor::Tensor,
    secondary: &Arc<SecondaryMmap>,
    x_mean: &[f32],
) -> Option<f32> {
    use crate::buffer::DType;
    const EPS: f64 = 1e-12;

    let dims = primary_tensor.shape().dims();
    if dims.len() != 2 {
        return None;
    }
    let (out_dim, in_dim) = (dims[0], dims[1]);
    if x_mean.len() != in_dim {
        return None;
    }

    let info = secondary.layer_tensor(layer_idx, subname)?;
    if info.dtype != DType::Q4_0 {
        return None;
    }
    let numel = primary_tensor.shape().numel();
    if numel != out_dim * in_dim || !numel.is_multiple_of(QK4_0) {
        return None;
    }
    let n_blocks = numel / QK4_0;
    let expected_bytes = n_blocks * std::mem::size_of::<BlockQ4_0>();
    let raw = secondary.tensor_bytes(info);
    if raw.len() != expected_bytes {
        return None;
    }
    let secondary_f32 = dequantize_q4_0_blocks(raw, numel);
    let primary_f32 = primary_tensor_to_f32(primary_tensor)?;
    if primary_f32.len() != numel {
        return None;
    }

    let mut diff_sq = 0.0f64;
    let mut prim_sq = 0.0f64;
    for r in 0..out_dim {
        let row_start = r * in_dim;
        let mut y_p = 0.0f32;
        let mut diff_y = 0.0f32;
        for c in 0..in_dim {
            let w_p = primary_f32[row_start + c];
            let w_s = secondary_f32[row_start + c];
            y_p += w_p * x_mean[c];
            diff_y += (w_p - w_s) * x_mean[c];
        }
        diff_sq += (diff_y as f64).powi(2);
        prim_sq += (y_p as f64).powi(2);
    }
    Some((diff_sq.sqrt() / prim_sq.max(EPS).sqrt()) as f32)
}

/// Convert a primary weight tensor to a `Vec<f32>`.
///
/// Supports F32 and F16 dtype buffers.  Returns `None` for Q4_0 primaries
/// (should not occur for the ε calculation path) or on access failure.
fn primary_tensor_to_f32(tensor: &crate::tensor::Tensor) -> Option<Vec<f32>> {
    use crate::buffer::DType;
    use half::f16;

    let numel = tensor.shape().numel();
    match tensor.dtype() {
        DType::F32 => {
            let ptr = tensor.as_ptr();
            if ptr.is_null() {
                return None;
            }
            // SAFETY: ptr points to at least `numel * 4` bytes of F32 data
            // for the lifetime of `tensor`.
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, numel) };
            Some(slice.to_vec())
        }
        DType::F16 => {
            let ptr = tensor.as_ptr();
            if ptr.is_null() {
                return None;
            }
            // SAFETY: ptr points to at least `numel * 2` bytes of F16 data.
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const f16, numel) };
            Some(slice.iter().map(|v| v.to_f32()).collect())
        }
        _ => {
            log::warn!(
                "primary_tensor_to_f32: unsupported dtype {:?}",
                tensor.dtype()
            );
            None
        }
    }
}

/// Per-layer cascade attention perturbation (§4.2 F4 + F5 ablation).
///
/// For each layer `i`, run the attention head forward twice over the cached
/// raw hidden state `X ∈ R^{T × d}` — once with F16 primary weights, once with
/// the Q4 secondary — and report two scalar perturbation measures:
///
/// - **F4** (cascade-aware single output projection):
///   `ε_o(V_out^F16) = ‖(W_o^F16 − W_o^Q4) · V_out^F16‖_F / ‖W_o^F16 · V_out^F16‖_F`
///   where `V_out^F16 = softmax(QK^T / √d_h) · V` is computed with F16 weights.
///
/// - **F5** (direct attention output):
///   `‖O^F16 − O^Q4‖_F / ‖O^F16‖_F`
///   where `O = W_o · softmax(QK^T / √d_h) · V` is the full attention head
///   output, evaluated independently with F16 and Q4 weights.
///
/// Returns one `(f4, f5)` tuple per layer. Entries are `(NaN, NaN)` for layers
/// with missing secondary weights, shape mismatch, or zero-magnitude denominator.
///
/// `raws[i] = (X_row_major, T, d)` is the per-layer hidden state cached by
/// `ImportanceCollector::raws_per_layer()`. `(n_heads, n_kv_heads, d_head)`
/// describes the GQA layout. Causal masking is applied; RoPE is *not* applied
/// (sec4 ablation — relative perturbation is invariant to RoPE under both
/// precisions in expectation).
pub fn compute_cascade_attn_perturbation(
    primary_slots: &[Arc<LayerSlot>],
    secondary: &Arc<SecondaryMmap>,
    raws: &[(Vec<f32>, usize, usize)],
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
) -> Vec<(f32, f32)> {
    let n = primary_slots.len();
    let mut out = vec![(f32::NAN, f32::NAN); n];
    if raws.len() != n {
        log::warn!(
            "compute_cascade_attn_perturbation: raws len {} != slots len {}",
            raws.len(),
            n
        );
        return out;
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        log::warn!(
            "compute_cascade_attn_perturbation: n_heads ({}) not divisible by n_kv_heads ({})",
            n_heads,
            n_kv_heads
        );
        return out;
    }

    let q_dim = n_heads * d_head;
    let kv_dim = n_kv_heads * d_head;

    for (i, slot) in primary_slots.iter().enumerate() {
        let (x_data, t, d) = &raws[i];
        let (t, d) = (*t, *d);
        if t == 0 || d == 0 {
            continue;
        }
        if x_data.len() < t * d {
            continue;
        }
        let weights = slot.load_weights();

        // Dequant Q4 from secondary and convert F16 primary, for each of
        // attn_q, attn_k, attn_v, attn_output.
        let (wq_f16, wq_q4) = match (
            primary_tensor_to_f32(&weights.wq),
            load_q4_dequant(i, "attn_q.weight", secondary, q_dim, d),
        ) {
            (Some(p), Some(s)) => (p, s),
            _ => continue,
        };
        let (wk_f16, wk_q4) = match (
            primary_tensor_to_f32(&weights.wk),
            load_q4_dequant(i, "attn_k.weight", secondary, kv_dim, d),
        ) {
            (Some(p), Some(s)) => (p, s),
            _ => continue,
        };
        let (wv_f16, wv_q4) = match (
            primary_tensor_to_f32(&weights.wv),
            load_q4_dequant(i, "attn_v.weight", secondary, kv_dim, d),
        ) {
            (Some(p), Some(s)) => (p, s),
            _ => continue,
        };
        let (wo_f16, wo_q4) = match (
            primary_tensor_to_f32(&weights.wo),
            load_q4_dequant(i, "attn_output.weight", secondary, d, q_dim),
        ) {
            (Some(p), Some(s)) => (p, s),
            _ => continue,
        };

        // Forward F16: Q = X @ W_q^T, K = X @ W_k^T, V = X @ W_v^T
        let q_f16 = matmul_x_wt(&x_data[..t * d], &wq_f16, t, d, q_dim);
        let k_f16 = matmul_x_wt(&x_data[..t * d], &wk_f16, t, d, kv_dim);
        let v_f16 = matmul_x_wt(&x_data[..t * d], &wv_f16, t, d, kv_dim);
        let v_out_f16 =
            attention_mix_causal(&q_f16, &k_f16, &v_f16, t, n_heads, n_kv_heads, d_head);
        let o_f16 = matmul_x_wt(&v_out_f16, &wo_f16, t, q_dim, d);

        // Forward Q4
        let q_q4 = matmul_x_wt(&x_data[..t * d], &wq_q4, t, d, q_dim);
        let k_q4 = matmul_x_wt(&x_data[..t * d], &wk_q4, t, d, kv_dim);
        let v_q4 = matmul_x_wt(&x_data[..t * d], &wv_q4, t, d, kv_dim);
        let v_out_q4 = attention_mix_causal(&q_q4, &k_q4, &v_q4, t, n_heads, n_kv_heads, d_head);
        let o_q4 = matmul_x_wt(&v_out_q4, &wo_q4, t, q_dim, d);

        // F4: alternative O using F16 V_out + Q4 W_o
        let o_f4_alt = matmul_x_wt(&v_out_f16, &wo_q4, t, q_dim, d);
        let f4 = frob_relative(&o_f16, &o_f4_alt);

        // F5: full F16 vs full Q4
        let f5 = frob_relative(&o_f16, &o_q4);

        out[i] = (f4, f5);
    }

    out
}

/// Load and dequantize a Q4_0 weight tensor from the secondary mmap.
/// Returns `None` on dtype/shape mismatch or missing entry.
fn load_q4_dequant(
    layer_idx: usize,
    subname: &str,
    secondary: &Arc<SecondaryMmap>,
    out_dim: usize,
    in_dim: usize,
) -> Option<Vec<f32>> {
    use crate::buffer::DType;
    let info = secondary.layer_tensor(layer_idx, subname)?;
    if info.dtype != DType::Q4_0 {
        return None;
    }
    let numel = out_dim * in_dim;
    if !numel.is_multiple_of(QK4_0) {
        return None;
    }
    let n_blocks = numel / QK4_0;
    let expected_bytes = n_blocks * std::mem::size_of::<BlockQ4_0>();
    let raw = secondary.tensor_bytes(info);
    if raw.len() != expected_bytes {
        return None;
    }
    Some(dequantize_q4_0_blocks(raw, numel))
}

/// Compute `Y = X · W^T` for row-major `X ∈ R^{T × in}`, `W ∈ R^{out × in}`.
/// Output is row-major `Y ∈ R^{T × out}`.
fn matmul_x_wt(x: &[f32], w: &[f32], t: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; t * out_dim];
    for ti in 0..t {
        let x_row = &x[ti * in_dim..(ti + 1) * in_dim];
        let y_row = &mut y[ti * out_dim..(ti + 1) * out_dim];
        for o in 0..out_dim {
            let w_row = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for k in 0..in_dim {
                acc += x_row[k] * w_row[k];
            }
            y_row[o] = acc;
        }
    }
    y
}

/// Causal-masked multi-head attention with GQA layout.
///
/// `q ∈ R^{T × (n_heads · d_head)}`, `k,v ∈ R^{T × (n_kv_heads · d_head)}`
/// (row-major). Each query head `h` reads from KV head `h / (n_heads/n_kv_heads)`.
/// Softmax is applied row-wise over the causal-masked logits scaled by
/// `1/√d_head`. Returns `V_out ∈ R^{T × (n_heads · d_head)}` row-major.
#[allow(clippy::needless_range_loop)]
fn attention_mix_causal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
) -> Vec<f32> {
    let q_dim = n_heads * d_head;
    let kv_dim = n_kv_heads * d_head;
    let scale = 1.0 / (d_head as f32).sqrt();
    let kv_ratio = n_heads / n_kv_heads;
    let mut v_out = vec![0.0f32; t * q_dim];

    for h in 0..n_heads {
        let kv_h = h / kv_ratio;
        for ti in 0..t {
            // Compute logits over causal range [0, ti]
            let mut logits = vec![f32::NEG_INFINITY; t];
            let mut max_l = f32::NEG_INFINITY;
            for tj in 0..=ti {
                let mut dot = 0.0f32;
                for d in 0..d_head {
                    let qi = ti * q_dim + h * d_head + d;
                    let kj = tj * kv_dim + kv_h * d_head + d;
                    dot += q[qi] * k[kj];
                }
                let l = dot * scale;
                logits[tj] = l;
                if l > max_l {
                    max_l = l;
                }
            }
            // softmax
            let mut exp_sum = 0.0f32;
            for tj in 0..=ti {
                let e = (logits[tj] - max_l).exp();
                logits[tj] = e;
                exp_sum += e;
            }
            if exp_sum < 1e-24 {
                continue;
            }
            for tj in 0..=ti {
                logits[tj] /= exp_sum;
            }
            // Weighted sum: v_out_row[h*d_head:..] = Σ_j logits[j] · v[j, kv_h*d_head:..]
            for tj in 0..=ti {
                let w = logits[tj];
                for d in 0..d_head {
                    let vi = tj * kv_dim + kv_h * d_head + d;
                    let oi = ti * q_dim + h * d_head + d;
                    v_out[oi] += w * v[vi];
                }
            }
        }
    }
    v_out
}

/// Relative Frobenius perturbation `‖A − B‖_F / ‖A‖_F`.
/// Returns `NaN` if `‖A‖_F` is below the EPS guard.
fn frob_relative(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut diff_sq = 0.0f64;
    let mut a_sq = 0.0f64;
    for i in 0..n {
        let d = (a[i] - b[i]) as f64;
        diff_sq += d * d;
        a_sq += (a[i] as f64) * (a[i] as f64);
    }
    if a_sq < 1e-24 {
        return f32::NAN;
    }
    (diff_sq.sqrt() / a_sq.sqrt()) as f32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_table() {
        let t = QuantNoiseTable::empty();
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert!(!t.is_computed());
        assert!(t.epsilon(0).is_none());
        assert_eq!(t.as_slice().len(), 0);
    }

    #[test]
    fn test_uniform_ones() {
        let t = QuantNoiseTable::uniform_ones(16);
        assert_eq!(t.len(), 16);
        assert!(!t.is_computed());
        for i in 0..16 {
            assert_eq!(t.epsilon(i), Some(1.0));
        }
        assert!(t.epsilon(16).is_none());
    }

    #[test]
    fn test_nan_layer_returns_none() {
        let mut table = QuantNoiseTable::uniform_ones(4);
        // Manually inject a NaN to simulate a failed layer.
        table.per_layer[2] = f32::NAN;
        assert_eq!(table.epsilon(0), Some(1.0));
        assert_eq!(table.epsilon(1), Some(1.0));
        // INV-127: NaN → None
        assert!(table.epsilon(2).is_none());
        assert_eq!(table.epsilon(3), Some(1.0));
    }

    #[test]
    fn test_out_of_range_returns_none() {
        let t = QuantNoiseTable::uniform_ones(4);
        assert!(t.epsilon(4).is_none());
        assert!(t.epsilon(100).is_none());
    }

    #[test]
    fn test_is_computed_flag() {
        let empty = QuantNoiseTable::empty();
        assert!(!empty.is_computed());
        let ones = QuantNoiseTable::uniform_ones(4);
        assert!(!ones.is_computed());
    }

    #[test]
    fn test_as_slice_contents() {
        let t = QuantNoiseTable::uniform_ones(3);
        let s = t.as_slice();
        assert_eq!(s.len(), 3);
        assert!(s.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_finite_epsilon_value() {
        // Inject a known finite value and verify epsilon() returns it.
        let mut t = QuantNoiseTable::uniform_ones(4);
        t.per_layer[1] = 0.042;
        assert!((t.epsilon(1).unwrap() - 0.042).abs() < 1e-6);
    }
}
