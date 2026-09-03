//! GPU/CPU parity for the decision-time observation-window attention.
//!
//! `kv::aperturb_select::window_attention` is the input every prefill-end candidate ranks from, so
//! the property that matters is not the float gap but whether the two paths agree on the ordering
//! a candidate's top-k reads. These tests assert both, over a uniform and a ragged cache.
//!
//! Skips cleanly on a host with no OpenCL driver.

#![cfg(feature = "opencl")]

use std::sync::Arc;

use argus_engine::backend::Backend;
use argus_engine::backend::opencl::OpenCLBackend;
use argus_engine::backend::opencl::memory::OpenCLMemory;
use argus_engine::kv::aperturb_select::{WindowSelfcheck, window_attention_selfcheck};
use argus_engine::memory::Memory;

fn run(ragged: bool, current_pos: usize, rows: usize) -> Option<WindowSelfcheck> {
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
        ragged,
    )
    .expect("window selfcheck");
    Some(got)
}

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

#[test]
fn window_attention_gpu_matches_cpu_on_a_uniform_cache() {
    let Some(got) = run(false, 256, 64) else {
        return;
    };
    assert_agrees(got, "uniform");
}

#[test]
fn window_attention_gpu_matches_cpu_on_a_ragged_cache() {
    let Some(got) = run(true, 256, 64) else {
        return;
    };
    assert_agrees(got, "ragged");
}

/// `current_pos` not a multiple of the work-group size, and a ragged start (here 49) that is not
/// a multiple of it either — the shape `flash_attn_f32.cl`'s tile race hid in, and the shape that
/// catches a kernel whose zero-fill and whose `+=` disagree about which columns a thread owns.
#[test]
fn window_attention_gpu_matches_cpu_on_an_unaligned_cache() {
    let Some(got) = run(true, 199, 64) else {
        return;
    };
    assert_agrees(got, "unaligned");
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
    let Some(got) = run(true, POS, ROWS) else {
        return;
    };
    assert_agrees(got, "blinded");
}
