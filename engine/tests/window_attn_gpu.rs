//! GPU/CPU parity for the decision-time observation-window attention.
//!
//! `kv::aperturb_select::window_attention` is the input every prefill-end candidate ranks from, so
//! the property that matters is not the float gap but whether the two paths agree on the ordering
//! a candidate's top-k reads. These tests assert both, over a uniform and a ragged cache.
//!
//! **A1 (ticket 010)** added a second axis: the kernel now exports the raw logits of the window's
//! trailing `metric_rows` rows so `aperturb::decide` can skip `kernel::logits_into`. The cases
//! below are split into two families — `metric_rows == rows`, which is what this file tested
//! before A1 and which does NOT exercise the tail seam at all, and `metric_rows < rows`, which is
//! the only place the seam is executed. Ticket 010 완료 기준 1 counts BOTH families.
//!
//! 🔴 FROZEN by the planning agent (`tickets/010-evidence/a1_window_attn_gpu_test.rs`).
//! Copy verbatim. If it does not compile, the implementation deviated from the contract the ticket
//! pinned — fix the implementation, not this file. Extending it with extra cases is allowed;
//! weakening an assertion or dropping a case is not.
//!
//! Skips cleanly on a host with no OpenCL driver.

#![cfg(feature = "opencl")]

use std::sync::Arc;

use argus_engine::backend::Backend;
use argus_engine::backend::opencl::OpenCLBackend;
use argus_engine::backend::opencl::memory::OpenCLMemory;
use argus_engine::kv::aperturb_select::{WindowSelfcheck, window_attention_selfcheck};
use argus_engine::memory::Memory;

fn run(
    ragged: bool,
    current_pos: usize,
    rows: usize,
    metric_rows: usize,
) -> Option<WindowSelfcheck> {
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return None;
        }
    };
    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        backend.context.clone(),
        backend.queue.clone(),
        true,
    ));
    let backend: Arc<dyn Backend> = Arc::new(backend);
    // Qwen2.5-1.5B's head geometry (GQA 6:1) at a capacity above `current_pos`, so the head stride
    // and the column count differ — the mistake a kernel makes when it copies the host packing.
    let got = window_attention_selfcheck(
        &backend,
        memory.as_ref(),
        /* n_layers */ 3,
        /* n_heads_q */ 12,
        /* n_kv_heads */ 2,
        /* head_dim */ 128,
        /* capacity */ current_pos + 37,
        current_pos,
        rows,
        metric_rows,
        ragged,
    )
    .expect("window selfcheck");
    Some(got)
}

/// The window-attention gate, unchanged from before A1.
fn assert_agrees(got: WindowSelfcheck, what: &str) {
    eprintln!("[{what}] {got:?}");
    assert!(
        got.gpu_ran,
        "{what}: the device path declined, so this compared the CPU against itself"
    );
    assert_eq!(
        got.argmax_mismatch, 0,
        "{what}: the two paths disagree on the top-scoring column ({got:?})"
    );
    assert!(
        got.max_rel < 1e-3,
        "{what}: relative gap {} exceeds 1e-3 ({got:?})",
        got.max_rel
    );
}

/// A1's gate on top of it: the exported raw z.
///
/// `z_gpu_ran` and `z_rows_compared` exist for the reason `gpu_ran` does — without them a run that
/// exported nothing, or compared nothing, reports a perfect score.
///
/// 🔴 The two gated quantities are `z_argmax_unexplained` and `z_max_abs`. `z_argmax_mismatch` and
/// `z_max_rel` are RECORDED, never gated, and the reason is measured rather than assumed
/// (`tickets/010-evidence/z_tie_diagnostic_2026-09-11.log`):
///
/// - A raw `argmax` disagreement can be a 2-ULP separation that the GPU's summation order collapses
///   to a single float. On Adreno, 1 of 21504 triples does exactly that. Demanding zero would be
///   demanding bit-identity, which this ticket knowingly gave up. `z_argmax_unexplained` counts only
///   the flips the two columns' own observed CPU-vs-GPU deviations cannot account for — no constant.
/// - A relative gap is unbounded here: raw logits pass through zero, so a 1-ULP absolute error on a
///   near-zero logit is an enormous relative one. Measured `z_max_rel` runs 8.7e-3 to 3.8e-2 on a
///   CORRECT implementation. The absolute gap is bounded: 1.49e-8 measured, against O(0.1) for an
///   export that carries the wrong rows or leaves columns unwritten. Hence `z_max_abs < 1e-3`.
fn assert_z_agrees(got: WindowSelfcheck, what: &str, want_compared: usize) {
    assert!(
        got.z_gpu_ran,
        "{what}: no z block came back from the device — the export is not wired ({got:?})"
    );
    assert!(
        got.z_rows_compared > 0,
        "{what}: z_rows_compared is 0, so nothing was actually checked ({got:?})"
    );
    assert_eq!(
        got.z_rows_compared, want_compared,
        "{what}: the comparison visited a different set of (layer, head, metric row) triples ({got:?})"
    );
    assert_eq!(
        got.z_argmax_unexplained, 0,
        "{what}: a column outranked the CPU's top by more than the observed deviation allows ({got:?})"
    );
    assert!(
        got.z_max_abs < 1e-3,
        "{what}: absolute z gap {} is five orders above the ~1.5e-7 a correct export shows ({got:?})",
        got.z_max_abs
    );
    // Provenance, and the only check that has it. A z the host recomputed for itself would be
    // bit-identical to the reference, because `kernel::logits_into` is deterministic — so an
    // exported block that never came off the device reads exactly 0.0 here. The kernel sums
    // `dot()` over float4s while the CPU blocks TB=8/VW=4, so across the >20M elements compared
    // they cannot agree everywhere. Measured floor: 8.94e-8 on this host, 1.19e-7 on Adreno.
    assert!(
        got.z_max_abs > 0.0,
        "{what}: the exported z is bit-identical to the CPU reference, so it was not computed on \
         the device ({got:?})"
    );
}

/// How many `(layer, query head, metric row)` triples the z comparison must visit.
///
/// A row is visited when its own causal end reaches past its KV head's ragged start — the same
/// `end <= start` skip `window_attention` makes on the CPU. Restated here rather than imported so
/// that a drift in the selfcheck's ragged geometry fails THIS file loudly instead of quietly
/// shrinking the coverage a passing run reports.
fn expected_z_rows(current_pos: usize, rows: usize, metric_rows: usize, ragged: bool) -> usize {
    const LAYERS: usize = 3;
    const HQ: usize = 12;
    const HKV: usize = 2;
    let n_rep = HQ / HKV;
    let m = metric_rows.min(rows).max(1);
    let mut n = 0usize;
    for h in 0..HQ {
        let kv = h / n_rep;
        // `window_attention_selfcheck` puts KV head `k`'s first resident slot at
        // `k * current_pos / (2 * n_kv_heads)`.
        let start = if ragged {
            kv * current_pos / (2 * HKV)
        } else {
            0
        }
        .min(current_pos);
        for t in 0..m {
            // `Geom::row_pos(t) + 1`, clamped — the metric geom's rows are `m`, not `rows`.
            let end = (current_pos - m + t + 1).min(current_pos);
            if end > start {
                n += 1;
            }
        }
    }
    n * LAYERS
}

// ── family 1: `metric_rows == rows`. These do NOT exercise A1's tail seam. ──

#[test]
fn window_attention_gpu_matches_cpu_on_a_uniform_cache() {
    let Some(got) = run(false, 256, 64, 64) else {
        return;
    };
    assert_agrees(got, "uniform");
    assert_z_agrees(got, "uniform", expected_z_rows(256, 64, 64, false));
}

#[test]
fn window_attention_gpu_matches_cpu_on_a_ragged_cache() {
    let Some(got) = run(true, 256, 64, 64) else {
        return;
    };
    assert_agrees(got, "ragged");
    assert_z_agrees(got, "ragged", expected_z_rows(256, 64, 64, true));
}

/// `current_pos` not a multiple of the work-group size, and a ragged start (here 49) that is not
/// a multiple of it either — the shape `flash_attn_f32.cl`'s tile race hid in, and the shape that
/// catches a kernel whose zero-fill and whose `+=` disagree about which columns a thread owns.
#[test]
fn window_attention_gpu_matches_cpu_on_an_unaligned_cache() {
    let Some(got) = run(true, 199, 64, 64) else {
        return;
    };
    assert_agrees(got, "unaligned");
    assert_z_agrees(got, "unaligned", expected_z_rows(199, 64, 64, true));
}

/// A window whose rows reach below what some heads hold: with a ragged start above a row's own
/// position the CPU skips that row entirely, and the kernel must skip the same ones.
///
/// The geometry is load-bearing and narrow. `window_attention_selfcheck` puts head `h`'s start at
/// `h * current_pos / (2 * n_kv_heads)`, so with two KV heads that is `current_pos / 4`, and the CPU
/// skips row `t` only when `current_pos - rows + t + 1 <= current_pos / 4`. At `rows = 64` that
/// needs `current_pos <= 84`: at 96 — where this test used to sit — the earliest row already ends at
/// 33 against a start of 24 and NOTHING is blinded, which left the boundary the test is named for
/// untouched on both sides. At 80 the start is 20 and rows 0..=3 end at 17..=20, so KV head 1 really
/// does blind them.
#[test]
fn window_attention_gpu_matches_cpu_when_rows_are_blinded() {
    const POS: usize = 80;
    const ROWS: usize = 64;
    // Restate the two formulas so the geometry cannot drift back to one that blinds nothing
    // without this failing: the selfcheck's ragged start for KV head 1 of 2, and the CPU's
    // `end <= start` skip.
    let start = POS / 4;
    let blinded = (0..ROWS)
        .filter(|t| (POS - ROWS + t + 1).min(POS) <= start)
        .count();
    assert!(
        blinded > 0,
        "the point of this case is a blinded row; at pos {POS} / rows {ROWS} none is"
    );
    let Some(got) = run(true, POS, ROWS, ROWS) else {
        return;
    };
    assert_agrees(got, "blinded");
    // This is the ONE case where the z comparison legitimately visits fewer than every triple:
    // KV head 1's ragged start blinds the window's first four rows, and the metric window here IS
    // the whole window. Pin that it really is fewer, so the case cannot drift into a geometry
    // where nothing is skipped and the assertion below stops meaning anything.
    let want = expected_z_rows(POS, ROWS, ROWS, true);
    assert!(
        want < 3 * 12 * ROWS,
        "the point of this case is a skipped row; expected_z_rows says none is skipped"
    );
    assert_z_agrees(got, "blinded", want);
}

// ── family 2: `metric_rows < rows` — the ONLY cases that execute A1's tail seam. ──
//
// Production is always here: the window is `q_snap.rows` (64 for the 4-arm pool) while the metric
// is `APERTURB_ROWS` = 16. A `metric_rows == rows` case cannot tell a correct tail extraction from
// one that exports the head of the window, because there the mapping is the identity.

/// The production shape: 16 of 64.
#[test]
fn the_exported_z_is_the_window_tail_on_a_uniform_cache() {
    let Some(got) = run(false, 256, 64, 16) else {
        return;
    };
    assert_agrees(got, "tail-uniform");
    assert_z_agrees(got, "tail-uniform", expected_z_rows(256, 64, 16, false));
}

#[test]
fn the_exported_z_is_the_window_tail_on_a_ragged_cache() {
    let Some(got) = run(true, 256, 64, 16) else {
        return;
    };
    assert_agrees(got, "tail-ragged");
    assert_z_agrees(got, "tail-ragged", expected_z_rows(256, 64, 16, true));
}

/// `metric_rows` that is NOT a multiple of the kernel's row block.
///
/// This is the case that kills "export the last two `WATT_B` blocks". At `WATT_B = 8` the trailing
/// 12 rows are 52..=63, which starts in the MIDDLE of the block at `t0 = 48`; an implementation
/// that copies whole blocks writes 16 rows into a 12-row window and every row is off by four.
#[test]
fn the_export_follows_metric_rows_not_the_kernel_row_block() {
    let Some(got) = run(true, 256, 64, 12) else {
        return;
    };
    assert_agrees(got, "tail-odd");
    assert_z_agrees(got, "tail-odd", expected_z_rows(256, 64, 12, true));
}

/// A metric window that fits inside ONE row block, over an unaligned `current_pos`.
///
/// `HostLayers::read` clamps the metric to `q_snap.rows.min(metric_rows).max(1)`, so a short
/// capture really does land here in production.
#[test]
fn the_export_handles_a_metric_window_inside_one_row_block() {
    let Some(got) = run(true, 199, 64, 5) else {
        return;
    };
    assert_agrees(got, "tail-tiny");
    assert_z_agrees(got, "tail-tiny", expected_z_rows(199, 64, 5, true));
}

/// The tail seam over a window that has blinded rows in its head.
///
/// The blinded rows are 0..=3 of the window and the metric takes 48..=63, so the metric's own rows
/// are all sighted — the point is that the export must survive a window pass that `continue`d out
/// of whole blocks before it reached the tail.
#[test]
fn the_export_survives_a_window_whose_head_rows_were_blinded() {
    let Some(got) = run(true, 80, 64, 16) else {
        return;
    };
    assert_agrees(got, "tail-blinded");
    assert_z_agrees(got, "tail-blinded", expected_z_rows(80, 64, 16, true));
}
