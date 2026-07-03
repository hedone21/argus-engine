//! `SignalRuntime` — the single coherence conduit for the attention-score signal (P1 of the
//! signal-axis inversion, `docs/design/signal-axis-inversion.md` §4.4/§4.5).
//!
//! Today the GPU↔CPU coherence of the attention-score accumulator is a caller-side discipline:
//! every reader calls [`score_fed::sync_gpu_scores_to_cpu`] before reading, every eviction calls
//! `acc.reset()` + [`score_fed::reset_gpu_scores`] after, and the prefill path calls
//! [`seed_prefill_importance_dual`]. That discipline is replicated across five driver loops
//! (cli/chat/bench/eval-ll/eval-ppl); a missed sync/reset is a silent-stale-score bug that has
//! recurred. `SignalRuntime` makes those three operations methods of a single owner so the
//! coherence contract holds by construction and the three free functions have zero out-of-runtime
//! callers (the P1 grep gate).
//!
//! **P1 is transitional.** The runtime wraps the one concrete producer (the
//! [`AttentionScoreAccumulator`]); P2 replaces `acc: Option<_>` with `Vec<Box<dyn SignalProducer>>`
//! and `synced_watermark: Option<u64>` with a per-producer `Vec<u64>`, keeping this method contract.
//! No new signal vocabulary is introduced here.
//!
//! **Ownership is single-window, not single-owner** (§4.4): bench shares one instance as
//! `Arc<Mutex<Option<SignalRuntime>>>` (ModelForward holds the lock across `forward_into`, the
//! eviction/mutation/dispatcher stages read it after); chat/eval/ppl each own an `Option<SignalRuntime>`
//! by value, exactly as they owned the raw accumulator. The device half's real owner is the backend
//! (`gpu_score_acc`); the runtime is a conduit that reaches it through `&dyn Backend`.

use crate::backend::Backend;
use crate::inference::attention_scores::{AttentionScoreAccumulator, seed_prefill_importance_dual};
use crate::kv::eviction::score_fed::{self, ExtractedScores};
use anyhow::Result;

/// The single coherence conduit for the attention-score signal. See the module docs.
pub struct SignalRuntime {
    /// The single P1 producer. `None` = the score-free path (the old dummy-`None` score cell /
    /// by-value `None` accumulator): every accessor degrades to `None`/no-op, byte-identical to the
    /// score-free regime. Forward mutates it via [`SignalRuntime::acc_mut`]; consumers read it via
    /// [`SignalRuntime::view`]/[`SignalRuntime::extract`] **after** [`SignalRuntime::ensure_coherent`].
    acc: Option<AttentionScoreAccumulator>,

    /// Last-synced GPU watermark: the `steps_accumulated` value at the most recent
    /// [`SignalRuntime::ensure_coherent`] pull. `None` = nothing synced this epoch (post-`new`,
    /// post-`reset`, post-`reset_host_only`) → the next read is forced to sync. P1 has a single
    /// producer so this is one value; P2 makes it per-producer.
    synced_watermark: Option<u64>,

    /// The prefill seed writes the GPU cumulative via `seed_cumulative`, which does **not** bump the
    /// device `steps_accumulated` counter — so a watermark comparison alone cannot see it. `seed`
    /// sets this so the next `ensure_coherent` pulls unconditionally (matching today's unconditional
    /// pre-read sync). This is runtime-internal (the runtime that performed the seed knows), NOT a
    /// caller-push mark-dirty (§4.5(2)).
    seed_dirty: bool,

    /// Recorded at [`SignalRuntime::arm`] for the CPU-authoritative-online-fold reject (§4.5(3)):
    /// faithful-H2O token-by-token prefill folds importance on the CPU per step, which an armed GPU
    /// device-authoritative pull would destroy. The combo is rejected at arm; this field documents
    /// the invariant for a `debug_assert` in `ensure_coherent`.
    online_cpu_fold: bool,
}

/// Pure dirty decision for [`SignalRuntime::ensure_coherent`], factored out so the watermark logic
/// is unit-testable without a backend. `gpu_wm` is the current device `steps_accumulated`
/// ([`score_fed::gpu_score_watermark`]); `synced` is the last-synced watermark.
///
/// Byte-identity with today's unconditional pre-read sync: because `import_gpu_scores` is an
/// idempotent OVERWRITE, "sync only when the device advanced" yields the same CPU bytes as "always
/// sync" (a skipped sync re-imports identical bytes). A fresh epoch (`synced == None` with an active
/// device) and any seed both force a sync.
#[inline]
fn signal_is_dirty(seed_dirty: bool, gpu_wm: Option<u64>, synced: Option<u64>) -> bool {
    if seed_dirty {
        return true;
    }
    match (gpu_wm, synced) {
        (Some(cur), Some(prev)) => cur > prev, // device reduce advanced since last sync
        (Some(_), None) => true,               // first read this epoch
        (None, _) => false,                    // CPU-only / unarmed → CPU acc already authoritative
    }
}

impl SignalRuntime {
    /// Wrap an already-constructed accumulator (or `None` for the score-free path). The accumulator
    /// is configured (`new_gqa` / `set_time_normalize` / `enable_per_layer_flat` / …) by the caller
    /// exactly as before; the runtime only takes ownership — no arithmetic, so construction stays
    /// byte-identical to the pre-runtime setup sites.
    pub fn new(acc: Option<AttentionScoreAccumulator>) -> Self {
        Self {
            acc,
            synced_watermark: None,
            seed_dirty: false,
            online_cpu_fold: false,
        }
    }

    /// Arm the GPU-side score accumulator (the device half) and record the authority regime. Folds
    /// [`score_fed::arm_gpu_score_acc`] and contractualizes the §4.5(3) reject: an armed device
    /// accumulator combined with CPU-authoritative online fold (faithful-H2O token-by-token prefill)
    /// is rejected here rather than after the first token, since the device pull would destroy the
    /// CPU fold. Returns `Ok(true)` when a GPU accumulator was armed, `Ok(false)` on a CPU
    /// build/backend (no-op), `Err` on a failed device init or the rejected combo.
    #[allow(clippy::too_many_arguments)]
    pub fn arm(
        &mut self,
        backend: &dyn Backend,
        n_layers: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        decay: f32,
        online_cpu_fold: bool,
    ) -> Result<bool> {
        let armed_gpu = score_fed::arm_gpu_score_acc(
            backend,
            n_layers,
            n_heads_q,
            n_kv_heads,
            max_seq_len,
            decay,
        )?;
        self.online_cpu_fold = online_cpu_fold;
        if online_cpu_fold && armed_gpu {
            anyhow::bail!(
                "faithful-H2O token-by-token prefill on an armed GPU score path is unsupported: \
                 the online CPU fold is correct but the GPU per-(layer, token) reduce is only seeded \
                 by the batched path, so a device-authoritative pull would destroy the CPU fold \
                 (§4.5(3)). Use the batched prefill probe or a CPU backend."
            );
        }
        Ok(armed_gpu)
    }

    /// The single `&mut` coherence point (§4.5(1)): pull the device-accumulated scores into the CPU
    /// accumulator iff the device advanced since the last pull (or a seed marked it dirty). Call once
    /// just before assembling the read ctx / extracting; afterwards all access is `&self`
    /// ([`SignalRuntime::view`]/[`SignalRuntime::extract`]). No-op on CPU builds/backends and when
    /// the accumulator is inactive, so the host path is byte-identical.
    pub fn ensure_coherent(&mut self, backend: &dyn Backend) {
        let Some(acc) = self.acc.as_mut() else {
            return;
        };
        if !acc.is_active() {
            return;
        }
        let gpu_wm = score_fed::gpu_score_watermark(backend);
        debug_assert!(
            !(self.online_cpu_fold && gpu_wm.is_some()),
            "arm() rejects CPU-authoritative fold + armed GPU device pull (§4.5(3))"
        );
        if signal_is_dirty(self.seed_dirty, gpu_wm, self.synced_watermark) {
            score_fed::sync_gpu_scores_to_cpu(acc, backend);
            self.synced_watermark = gpu_wm;
            self.seed_dirty = false;
        }
    }

    /// Forward-time mutation handle. Returns `Some(&mut acc)` only when a producer exists; the caller
    /// (ModelForward `step` / the eval/ppl forward feeds) calls `begin_step()` and lends this into the
    /// decode forward args. **This performs no sync** — coherence is a forward-*boundary* operation
    /// (`ensure_coherent`), never per-forward — so the hot decode path is unchanged.
    #[inline]
    pub fn acc_mut(&mut self) -> Option<&mut AttentionScoreAccumulator> {
        self.acc.as_mut()
    }

    /// Post-coherent read handle for dumps that read raw accumulator buffers
    /// (`importance_scores`/`head_importance_scores`/`last_step_head_attn`). Valid only after
    /// [`SignalRuntime::ensure_coherent`] in the same critical section.
    #[inline]
    pub fn view(&self) -> Option<&AttentionScoreAccumulator> {
        self.acc.as_ref()
    }

    /// Whether a producer exists and is active — a cheap `&self` check for the plan-bypass gate,
    /// which runs before the forward lock and so cannot take `acc_mut`.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.acc.as_ref().is_some_and(|a| a.is_active())
    }

    /// Extract the `(collapsed, value-aware, per-layer FLAT)` eviction score triple. Wraps
    /// [`score_fed::extract_scores`]; `None` when there is no active producer. The caller routes the
    /// result via [`score_fed::route_evict`] (which owns the `&mut [KVCache]`) with the lock released.
    /// Valid only after [`SignalRuntime::ensure_coherent`].
    pub fn extract(&self) -> Option<ExtractedScores> {
        self.acc.as_ref().and_then(score_fed::extract_scores)
    }

    /// Reset at an eviction boundary (KV geometry changed → prior scores are stale): the CPU
    /// `acc.reset()` in lockstep with [`score_fed::reset_gpu_scores`] (which zeroes the device
    /// cumulative + `steps_accumulated`). Clears the watermark so the next `ensure_coherent` starts a
    /// fresh epoch. No-op on CPU backends / when inactive.
    pub fn reset(&mut self, backend: &dyn Backend) {
        if let Some(acc) = self.acc.as_mut() {
            acc.reset();
            score_fed::reset_gpu_scores(acc, backend);
        }
        self.synced_watermark = None;
        self.seed_dirty = false;
    }

    /// CPU-only reset — mirror of today's `KVMutationDriverStage::reset_scores` (which resets the CPU
    /// accumulator with no backend and leaves the device buffers untouched). Distinct from
    /// [`SignalRuntime::reset`]: adding a device reset here would be a behavior change. The watermark
    /// is cleared so the next `ensure_coherent` re-pulls the (untouched, still-advancing) device
    /// buffer — byte-identical to today's unconditional pre-read sync overwriting the CPU reset.
    pub fn reset_host_only(&mut self) {
        if let Some(acc) = self.acc.as_mut() {
            acc.reset();
        }
        self.synced_watermark = None;
    }

    /// Seed prefill importance into the CPU accumulator and mirror it onto the GPU cumulative. Wraps
    /// [`seed_prefill_importance_dual`] and marks the runtime seed-dirty so the next `ensure_coherent`
    /// pulls unconditionally (the seed bypasses `steps_accumulated`, so the watermark cannot see it).
    pub fn seed(
        &mut self,
        backend: &dyn Backend,
        pfa: &[Vec<f32>],
        prefix_len: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
    ) {
        if let Some(acc) = self.acc.as_mut() {
            seed_prefill_importance_dual(acc, backend, pfa, prefix_len, n_heads_q, n_kv_heads);
            self.seed_dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── watermark dirty logic (pure, no backend) ──

    #[test]
    fn dirty_false_when_device_has_not_advanced() {
        // Same watermark, no seed → clean → skip the redundant sync (idempotent OVERWRITE would
        // re-import identical bytes; skipping is byte-identical to always-sync).
        assert!(!signal_is_dirty(false, Some(5), Some(5)));
    }

    #[test]
    fn dirty_true_when_device_advanced() {
        assert!(signal_is_dirty(false, Some(6), Some(5)));
    }

    #[test]
    fn dirty_true_on_first_read_this_epoch() {
        // Active device, nothing synced yet (post-new / post-reset) → must sync.
        assert!(signal_is_dirty(false, Some(3), None));
    }

    #[test]
    fn dirty_true_when_seeded_even_without_device_advance() {
        // Seed writes the GPU cumulative via `seed_cumulative` without bumping `steps_accumulated`,
        // so the watermark alone would read clean — `seed_dirty` forces the pull.
        assert!(signal_is_dirty(true, Some(5), Some(5)));
        assert!(signal_is_dirty(true, None, None));
    }

    #[test]
    fn dirty_false_on_cpu_only_or_unarmed() {
        // No device watermark → the CPU accumulator is already authoritative → never sync.
        assert!(!signal_is_dirty(false, None, None));
        assert!(!signal_is_dirty(false, None, Some(9)));
    }

    // ── score-free (None) runtime: every accessor degrades, no panic ──

    #[test]
    fn none_runtime_accessors_degrade() {
        let mut rt = SignalRuntime::new(None);
        assert!(!rt.is_active());
        assert!(rt.acc_mut().is_none());
        assert!(rt.view().is_none());
        assert!(rt.extract().is_none());
        // ensure_coherent / reset / reset_host_only / seed on a None runtime are no-ops. A CPU
        // backend makes the device helpers no-op too; we only assert no panic on the None arm here.
        rt.reset_host_only();
        assert_eq!(rt.synced_watermark, None);
    }

    // ── reset_host_only clears the watermark but is CPU-only ──

    #[test]
    fn reset_host_only_clears_watermark() {
        let mut rt = SignalRuntime::new(None);
        rt.synced_watermark = Some(7);
        rt.reset_host_only();
        // Cleared → next ensure_coherent starts a fresh epoch (re-pulls the untouched device buffer),
        // matching today's unconditional pre-read sync overwriting the CPU-only reset.
        assert_eq!(rt.synced_watermark, None);
    }

    // ── arm on a CPU backend is a no-op and never rejects ──

    #[test]
    fn arm_on_cpu_backend_is_noop_and_never_rejects() {
        use crate::backend::cpu::CpuBackend;
        let be = CpuBackend::new();
        let mut rt = SignalRuntime::new(None);
        // No GPU accumulator on a CPU backend → Ok(false). The §4.5(3) CPU-fold reject never fires
        // here because `armed_gpu` is false (the reject only guards an ARMED device path).
        assert!(!rt.arm(&be, 4, 8, 2, 128, 0.0, false).unwrap());
        assert!(!rt.arm(&be, 4, 8, 2, 128, 0.0, true).unwrap());
    }

    // ── the P1 grep gate: the three coherence primitives are called ONLY through this conduit ──

    #[test]
    fn gate_coherence_fns_called_only_through_conduit() {
        use std::fs;
        use std::path::{Path, PathBuf};

        // rustc already enforces this via restricted visibility (seed = `pub(in crate::inference)`,
        // sync/reset = `pub(crate)`); this is the human-readable tripwire that also catches a
        // direct call reintroduced alongside a visibility widening.
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let gated = [
            "sync_gpu_scores_to_cpu(",
            "reset_gpu_scores(",
            "seed_prefill_importance_dual(",
        ];
        // The conduit itself + the two definition sites are the only files allowed to name them.
        let allowed = ["signal_runtime.rs", "score_fed.rs", "attention_scores.rs"];

        fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
            for e in fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    collect(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        collect(&src, &mut files);

        let mut violations = Vec::new();
        for f in &files {
            let name = f.file_name().unwrap().to_str().unwrap().to_string();
            if allowed.contains(&name.as_str()) {
                continue;
            }
            let text = fs::read_to_string(f).unwrap();
            for (i, line) in text.lines().enumerate() {
                // Comment lines merely mentioning a name (always in backticks, no `(`) are fine.
                if line.trim_start().starts_with("//") {
                    continue;
                }
                for g in &gated {
                    if line.contains(g) {
                        violations.push(format!("{name}:{} `{g}`", i + 1));
                    }
                }
            }
        }
        assert!(
            violations.is_empty(),
            "gated coherence primitives called outside the SignalRuntime conduit — route through \
             SignalRuntime::{{ensure_coherent, reset, seed}} instead: {violations:?}"
        );
    }
}
