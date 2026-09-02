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
    /// How far the ring's clock has run ahead of the cache's, in positions.
    ///
    /// The ring stamps RoPE positions, which by design keep counting across a compaction
    /// (`decode_loop`: "Do NOT reset start_pos to current_pos after eviction ... severe NLL
    /// degradation"). The cache's `current_pos` is renumbered DOWN by the same event. So the two
    /// clocks separate by exactly the number of evicted positions, permanently and cumulatively,
    /// and a reader that asks in `current_pos` terms is asking about positions the ring stopped
    /// holding — from the first compaction onward, forever. [`Self::set_drift`] carries the gap over
    /// so the window can be addressed in the ring's coordinate without losing the check that
    /// catches a capture which simply stopped.
    drift: usize,
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
            drift: 0,
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

    /// The `min(rows, n_resident)` most recently captured positions, as `(first_pos, count)`, or
    /// `None` when the ring cannot serve that many.
    ///
    /// **The window is taken in the ring's own coordinate, ending at [`Self::last_pos`].** The
    /// caller passes `n_resident` (the cache's `current_pos`) only to bound how many rows it is
    /// entitled to ask for — never as a position. Addressing the window as
    /// `[n_resident - r, n_resident)` is what used to break here: a compaction renumbers the cache
    /// down by the number of evicted positions while the capture keeps stamping RoPE positions that
    /// do not, so from the first compaction onward the asked-for window sat entirely below the
    /// stamped one and every later read was refused. The rows themselves were always fine — they
    /// are the trailing queries, which is exactly what the metric probes with; only the labels were
    /// being compared across two different clocks.
    ///
    /// `slot_pos` is still checked position by position, so "the capture was not armed at that
    /// step" remains a loud failure rather than a plausible wrong answer.
    fn trailing_window(&self, n_resident: usize) -> Option<(usize, usize)> {
        let r_eff = self.rows.min(n_resident);
        if r_eff == 0 {
            return None;
        }
        // `n_resident` counts cache slots; the ring counts RoPE positions. `drift` is the gap.
        let end = n_resident + self.drift;
        let first = end.checked_sub(r_eff)?;
        if (first..end).any(|p| self.slot_pos[p % self.rows] != p) {
            return None;
        }
        Some((first, r_eff))
    }

    /// Set how far the ring's clock leads the cache's: `rope_pos - resident`.
    ///
    /// Stated absolutely, from the two clocks themselves, rather than accumulated per prune. The
    /// accumulating form has to be handed the resident count from *immediately before* the
    /// compaction, and the decode loop's shrink detector only holds the count from the previous
    /// step — one token older. That one position is not a rounding error here: the ring is exactly
    /// `rows` long, so a window shifted down by one asks for a position the newest capture has
    /// already overwritten, and every read is refused just as if nothing had been reported
    /// (measured on an S25, 2026-09-02). Reading both clocks at the same instant cannot drift.
    pub fn set_drift(&mut self, gap: usize) {
        self.drift = gap;
    }

    /// Whether the ring holds the whole window [`Self::snapshot`] would read — the cheap predicate
    /// a caller checks before deciding it has something to measure.
    pub fn covers(&self, n_resident: usize) -> bool {
        self.trailing_window(n_resident).is_some()
    }

    /// Read the ring back in chronological order for the `min(rows, n_resident)` positions ending
    /// at `n_resident - 1`.
    ///
    /// One device-to-host transfer for the whole ring, at the decision point — not per step.
    pub fn snapshot(&self, n_resident: usize) -> Result<QRowSnapshot> {
        let Some((first_pos, r_eff)) = self.trailing_window(n_resident) else {
            bail!(
                "q-rows: cannot serve the {} trailing row(s) for {n_resident} resident token(s) \
                 (ring is {} position(s) ahead of the cache) — the capture was not armed when \
                 those tokens went past",
                self.rows.min(n_resident),
                self.drift
            );
        };
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
        // The same condition, asked cheaply and without an error to unwrap — what a caller checks
        // when it has a choice about whether to read at all.
        assert!(!c.covers(10));
        assert!(c.covers(4));
    }

    /// `covers` is false once the cache is renumbered under the ring, which is what a compaction
    /// between the capture and the read does: the ring is indexed by absolute position, so slots
    /// that held positions 8..12 do not answer for positions 2..6.
    #[test]
    fn a_renumbered_cache_is_no_longer_covered_by_the_ring() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let mut c = cap(1, 4, 2);
        let buf = mem.alloc(12 * 2 * 4, DType::F32).unwrap();
        let t = Tensor::new(Shape::new(vec![1, 12, 2]), buf, backend.clone());
        c.capture(0, &t, backend.as_ref(), 0, 12, 2).unwrap();
        assert!(c.covers(12), "the positions it really saw");
        // A compaction left six tokens resident, renumbered 0..6. The ring still holds 8..12.
        assert!(!c.covers(6));
    }

    /// The bug this ring shipped with: a compaction renumbers the cache down while the capture
    /// keeps stamping RoPE positions, so every read after the FIRST compaction was refused for the
    /// rest of the session. Telling the ring about the prune closes the gap.
    #[test]
    fn a_pruned_cache_is_covered_again_once_the_ring_is_told() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let mut c = cap(1, 4, 2);
        let buf = mem.alloc(12 * 2 * 4, DType::F32).unwrap();
        let t = Tensor::new(Shape::new(vec![1, 12, 2]), buf, backend.clone());
        c.capture(0, &t, backend.as_ref(), 0, 12, 2).unwrap();
        assert!(c.covers(12), "the positions it really saw");
        // A compaction left six resident, renumbered 0..6. Untold, the ring refuses.
        assert!(!c.covers(6));
        c.set_drift(6);
        assert!(
            c.covers(6),
            "after the prune is accounted for, the trailing rows are readable again"
        );
        assert!(c.snapshot(6).is_ok());
    }

    /// The gap grows at every compaction, and a run does many. Reporting it absolutely means the
    /// second compaction's report already includes the first one's.
    #[test]
    fn drift_accumulates_across_repeated_prunes() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let mut c = cap(1, 4, 2);
        let buf = mem.alloc(20 * 2 * 4, DType::F32).unwrap();
        let t = Tensor::new(Shape::new(vec![1, 20, 2]), buf, backend.clone());
        c.capture(0, &t, backend.as_ref(), 0, 20, 2).unwrap();
        // Two compactions: 20 -> 14 (gap 6), then 14 -> 9 (cumulative gap 11).
        c.set_drift(6);
        c.set_drift(11);
        assert!(
            c.covers(9),
            "9 resident + 11 drift = the 20 positions captured"
        );
        assert!(!c.covers(20), "20 + 11 = 31 was never captured");
    }

    /// The gap must be the one measured at the SAME instant on both clocks. Reporting it one
    /// position stale — which is what accumulating from the previous step's occupancy does — shifts
    /// the window down by one, onto a position the newest capture has already overwritten, because
    /// the ring is exactly `rows` long. The symptom is indistinguishable from never reporting at
    /// all: every read refused.
    #[test]
    fn a_gap_reported_one_position_stale_still_refuses() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let mut c = cap(1, 4, 2);
        let buf = mem.alloc(10 * 2 * 4, DType::F32).unwrap();
        let t = Tensor::new(Shape::new(vec![1, 10, 2]), buf, backend.clone());
        // Positions 0..9 went past; the ring (4 slots) holds 6..9.
        c.capture(0, &t, backend.as_ref(), 0, 10, 2).unwrap();
        // The cache was compacted to 6 resident while the RoPE clock stayed at 10: gap 4.
        c.set_drift(4);
        assert!(c.covers(6), "the exact gap reads the window the ring holds");
        // One stale: the window slides to 5..9, and slot 5 % 4 = 1 now holds 9.
        c.set_drift(3);
        assert!(
            !c.covers(6),
            "a stale gap asks for a position already overwritten"
        );
    }

    /// The guarantee the prune accounting must NOT trade away: a capture that simply stopped still
    /// has to be refused. Drift explains a renumbering, not a gap in the recording.
    #[test]
    fn a_capture_that_stopped_is_still_refused_after_a_prune() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let mut c = cap(1, 4, 2);
        let buf = mem.alloc(12 * 2 * 4, DType::F32).unwrap();
        let t = Tensor::new(Shape::new(vec![1, 12, 2]), buf, backend.clone());
        c.capture(0, &t, backend.as_ref(), 0, 12, 2).unwrap();
        c.set_drift(6);
        // Decode went on for four more tokens with the capture disarmed: the cache says 10
        // resident, but the ring never saw positions 12..16.
        assert!(!c.covers(10));
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
