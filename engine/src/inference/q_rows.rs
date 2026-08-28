//! The trailing `R` post-RoPE query rows of every layer, kept for the output-perturbation metric.
//!
//! [`crate::aperturb`] scores a compression candidate by recomputing those rows' attention against
//! the cache the candidate would leave behind. The rows themselves are not recomputed — they are
//! what the forward already produced, and reusing them is the whole reason the metric costs no
//! forward pass. But the engine overwrites `ws.q` layer to layer and frees the prefill workspace at
//! exit, so they have to be copied out as they go past.
//!
//! **Hook point.** Immediately after `backend.rope_inplace(&mut q_rope, ..)`, in BOTH forks, before
//! the KV write and before anything can touch `ws.q` again. `None` (the production case) is one
//! `is_some` branch per layer and nothing else — the same gating [`crate::inference::head_mask`] and
//! [`crate::inference::duo_heads`] use, for the same reason.
//!
//! **The ring has no cursor.** Absolute position `p` lives in slot `p % R`, and each chunk or step
//! contributes only its trailing `min(R, len)` rows. That makes the capture correct for any chunk
//! size without the caller knowing one: it survives a prefill split at 512, a chunk of 1, and a
//! decode tail of any length, because the writes arrive in non-decreasing `p` and `p ↦ p % R` is a
//! bijection on any `R` consecutive positions. `slot_pos` records what each slot actually holds, so
//! a read that would return the wrong position fails loudly instead.
//!
//! **Device residency.** The copy is `Backend::copy_slice` — a device-to-device memcpy on CUDA and
//! OpenCL — so no query row ever crosses to the host during the forward. Only the drain at the
//! decision point reads back, once, for the whole ring. That matters most on Adreno, where `ws.q`
//! may be device-only with a null host pointer and `as_slice` would be a null dereference (INV-191).

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::backend::Backend;
use crate::buffer::DType;
use crate::memory::Memory;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// A per-layer ring of the last `R` post-RoPE query rows.
pub struct QRowCapture {
    n_layers: usize,
    rows: usize,
    /// `n_heads_q * head_dim`. **Not** `hidden_size`: on a model whose head dimension is not
    /// `hidden_size / n_heads_q` the two differ, and the query rows are the former.
    q_dim: usize,
    /// `[n_layers][rows][q_dim]` f32, allocated from the same memory as the workspaces so it is
    /// device-resident wherever they are.
    ring: Tensor,
    /// What absolute position each slot currently holds; `usize::MAX` for never written. Host-side
    /// bookkeeping only — it is what turns "the ring was not armed at that step" from a plausible
    /// wrong answer into an error.
    slot_pos: Vec<usize>,
}

impl QRowCapture {
    pub fn new(
        backend: Arc<dyn Backend>,
        memory: &dyn Memory,
        n_layers: usize,
        rows: usize,
        q_dim: usize,
    ) -> Result<Self> {
        if n_layers == 0 || rows == 0 || q_dim == 0 {
            bail!("q-rows: degenerate geometry ({n_layers} layers, {rows} rows, q_dim {q_dim})");
        }
        let n = n_layers * rows * q_dim;
        let buf = memory.alloc(n * 4, DType::F32)?;
        Ok(Self {
            n_layers,
            rows,
            q_dim,
            ring: Tensor::new(Shape::new(vec![n_layers, rows, q_dim]), buf, backend),
            slot_pos: vec![usize::MAX; rows],
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn q_dim(&self) -> usize {
        self.q_dim
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Copy this chunk's (or step's) trailing query rows into `layer`'s ring slice.
    ///
    /// `q_rope` is `[1, seq_len, n_heads_q, head_dim]` over the workspace buffer, already rotated.
    /// `start_pos` is the absolute position of its first row.
    pub fn capture(
        &mut self,
        layer: usize,
        q_rope: &Tensor,
        backend: &dyn Backend,
        start_pos: usize,
        seq_len: usize,
        q_dim: usize,
    ) -> Result<()> {
        if layer >= self.n_layers {
            return Ok(());
        }
        if q_dim != self.q_dim {
            bail!(
                "q-rows: layer q_dim {q_dim} but the ring was armed for {}",
                self.q_dim
            );
        }
        if q_rope.size() / 4 != seq_len * self.q_dim {
            bail!(
                "q-rows: expected a single batch of {seq_len} x {} rows, got {} floats",
                self.q_dim,
                q_rope.size() / 4
            );
        }
        let take = self.rows.min(seq_len);
        let first_in_chunk = seq_len - take;
        let first_pos = start_pos + first_in_chunk;
        let base = layer * self.rows;
        let mut done = 0usize;
        while done < take {
            let slot = (first_pos + done) % self.rows;
            // Longest run that stays inside the ring without wrapping — one copy per run, so a
            // whole prefill chunk is at most two calls.
            let run = (self.rows - slot).min(take - done);
            let mut dst = self.ring.clone();
            backend.copy_slice(
                q_rope,
                &mut dst,
                (first_in_chunk + done) * self.q_dim,
                (base + slot) * self.q_dim,
                run * self.q_dim,
            )?;
            done += run;
        }
        if layer == 0 {
            for j in 0..take {
                let p = first_pos + j;
                self.slot_pos[p % self.rows] = p;
            }
        }
        Ok(())
    }

    /// Read the ring back in chronological order for the `min(rows, n_resident)` positions ending
    /// at `n_resident - 1`.
    ///
    /// One device-to-host transfer for the whole ring, at the decision point — not per step.
    pub fn snapshot(&self, n_resident: usize) -> Result<QRowSnapshot> {
        let r_eff = self.rows.min(n_resident);
        if r_eff == 0 {
            bail!("q-rows: nothing resident to snapshot");
        }
        let first_pos = n_resident - r_eff;
        for j in 0..r_eff {
            let p = first_pos + j;
            if self.slot_pos[p % self.rows] != p {
                bail!(
                    "q-rows: slot {} holds position {} but position {p} was asked for — the \
                     capture was not armed when that token went past",
                    p % self.rows,
                    self.slot_pos[p % self.rows] as i64
                );
            }
        }
        let backend = self.ring.backend().clone();
        backend.synchronize()?;
        let mut flat = vec![0.0f32; self.n_layers * self.rows * self.q_dim];
        {
            // SAFETY: `flat` is a freshly allocated f32 slice; `read_buffer` writes f32 bytes into
            // it and keeps no pointer. The same reinterpretation the faithful read seam does.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(flat.as_mut_ptr() as *mut u8, flat.len() * 4)
            };
            backend.read_buffer(&self.ring, bytes)?;
        }
        let mut data = vec![0.0f32; self.n_layers * r_eff * self.q_dim];
        for l in 0..self.n_layers {
            for j in 0..r_eff {
                let s = ((l * self.rows) + ((first_pos + j) % self.rows)) * self.q_dim;
                let d = ((l * r_eff) + j) * self.q_dim;
                data[d..d + self.q_dim].copy_from_slice(&flat[s..s + self.q_dim]);
            }
        }
        Ok(QRowSnapshot {
            n_layers: self.n_layers,
            rows: r_eff,
            q_dim: self.q_dim,
            first_pos,
            data,
        })
    }
}

/// Host-resident query rows in chronological order.
///
/// `data` is `[n_layers][rows][q_dim]`, and within a row the layout is `[n_heads_q][head_dim]` —
/// the workspace's own. The metric wants `[n_heads_q][rows][head_dim]` per layer, which is one
/// transpose, done once per decision by [`Self::layer_head_major`].
pub struct QRowSnapshot {
    pub n_layers: usize,
    pub rows: usize,
    pub q_dim: usize,
    /// Absolute position of row 0.
    pub first_pos: usize,
    pub data: Vec<f32>,
}

impl QRowSnapshot {
    /// One layer's rows as `[n_heads_q][rows][head_dim]`.
    pub fn layer_head_major(&self, layer: usize, head_dim: usize) -> Vec<f32> {
        let n_q = self.q_dim / head_dim;
        let mut out = vec![0.0f32; n_q * self.rows * head_dim];
        let src = layer * self.rows * self.q_dim;
        for t in 0..self.rows {
            for h in 0..n_q {
                let s = src + t * self.q_dim + h * head_dim;
                let d = (h * self.rows + t) * head_dim;
                out[d..d + head_dim].copy_from_slice(&self.data[s..s + head_dim]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::galloc::Galloc;

    fn cap(n_layers: usize, rows: usize, q_dim: usize) -> QRowCapture {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        QRowCapture::new(backend, &mem, n_layers, rows, q_dim).expect("ring")
    }

    /// Feed a token stream through an arbitrary chunking and check the ring holds the trailing
    /// positions, in order. Value of position `p` in layer `l` is `p * 100 + l`.
    fn drive(n_layers: usize, rows: usize, q_dim: usize, chunks: &[usize]) -> QRowSnapshot {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let mut c = cap(n_layers, rows, q_dim);
        let mut pos = 0usize;
        for &len in chunks {
            for l in 0..n_layers {
                let buf = mem.alloc(len * q_dim * 4, DType::F32).unwrap();
                let mut t = Tensor::new(Shape::new(vec![1, len, q_dim]), buf, backend.clone());
                {
                    let s = t.as_mut_slice::<f32>();
                    for j in 0..len {
                        for e in 0..q_dim {
                            s[j * q_dim + e] = ((pos + j) * 100 + l) as f32 + e as f32 / 1000.0;
                        }
                    }
                }
                c.capture(l, &t, backend.as_ref(), pos, len, q_dim).unwrap();
            }
            pos += len;
        }
        c.snapshot(pos).expect("snapshot")
    }

    fn check(snap: &QRowSnapshot, n: usize, q_dim: usize) {
        for l in 0..snap.n_layers {
            for j in 0..snap.rows {
                let p = snap.first_pos + j;
                let got = snap.data[(l * snap.rows + j) * q_dim];
                assert_eq!(
                    got,
                    (p * 100 + l) as f32,
                    "layer {l} row {j} (position {p})"
                );
            }
        }
        assert_eq!(snap.first_pos + snap.rows, n);
    }

    #[test]
    fn a_single_prefill_chunk_leaves_the_trailing_rows() {
        let snap = drive(3, 4, 2, &[10]);
        check(&snap, 10, 2);
    }

    #[test]
    fn the_ring_survives_a_chunk_boundary_inside_the_trailing_window() {
        // The case a cursor-free ring exists for: the last R rows straddle two chunks.
        let snap = drive(2, 4, 2, &[6, 1]);
        check(&snap, 7, 2);
        let snap = drive(2, 16, 3, &[512, 1]);
        check(&snap, 513, 3);
    }

    #[test]
    fn prefill_then_decode_steps_keep_the_window_moving() {
        let mut chunks = vec![9usize];
        chunks.extend(std::iter::repeat_n(1usize, 7));
        let snap = drive(2, 5, 2, &chunks);
        check(&snap, 16, 2);
    }

    #[test]
    fn every_chunking_of_a_stream_gives_the_same_window() {
        for &rows in &[1usize, 2, 3, 5, 16] {
            for &chunk in &[1usize, 2, 3, 7, 32] {
                let n = 40usize;
                let mut chunks = Vec::new();
                let mut left = n;
                while left > 0 {
                    let c = chunk.min(left);
                    chunks.push(c);
                    left -= c;
                }
                let snap = drive(2, rows, 2, &chunks);
                check(&snap, n, 2);
            }
        }
    }

    #[test]
    fn a_shorter_context_than_the_window_is_served_in_full() {
        let snap = drive(2, 8, 2, &[3]);
        assert_eq!(snap.rows, 3);
        assert_eq!(snap.first_pos, 0);
        check(&snap, 3, 2);
    }

    #[test]
    fn a_position_the_ring_never_saw_is_refused() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let mut c = cap(1, 4, 2);
        let buf = mem.alloc(4 * 2 * 4, DType::F32).unwrap();
        let t = Tensor::new(Shape::new(vec![1, 4, 2]), buf, backend.clone());
        c.capture(0, &t, backend.as_ref(), 0, 4, 2).unwrap();
        // Claiming ten tokens are resident when only four went past must not silently return
        // whatever happens to sit in the ring.
        assert!(c.snapshot(10).is_err());
    }

    #[test]
    fn a_geometry_mismatch_is_refused_rather_than_reinterpreted() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let mut c = cap(1, 4, 6);
        let buf = mem.alloc(4 * 6 * 4, DType::F32).unwrap();
        let t = Tensor::new(Shape::new(vec![1, 4, 6]), buf, backend.clone());
        assert!(c.capture(0, &t, backend.as_ref(), 0, 4, 5).is_err());
        assert!(c.capture(0, &t, backend.as_ref(), 0, 8, 6).is_err());
    }

    #[test]
    fn the_head_major_view_transposes_a_row_into_the_metric_layout() {
        let snap = drive(1, 2, 6, &[2]); // q_dim 6 = 3 heads x head_dim 2
        let hm = snap.layer_head_major(0, 2);
        for t in 0..2 {
            for h in 0..3 {
                for e in 0..2 {
                    assert_eq!(
                        hm[(h * 2 + t) * 2 + e],
                        snap.data[t * 6 + h * 2 + e],
                        "row {t} head {h} dim {e}"
                    );
                }
            }
        }
    }
}
