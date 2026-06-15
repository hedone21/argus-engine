//! Per-token streaming hook for the chat server (SSE `stream=true`).
//!
//! [`run_until_stop`](crate::session::DecodeLoop::run_until_stop) batches its
//! output — the kept tokens are returned only when the turn finishes. For
//! OpenAI-style streaming the server needs each token as it is produced. This
//! module provides a [`ChatStreamStage`] (`DecodeEnd` subscriber) that invokes a
//! per-turn callback with the just-generated token, plus [`IncDetok`], an
//! incremental detokenizer that turns the growing token vec into UTF-8-safe text
//! deltas.
//!
//! The stage is registered **after** [`ChatStopStage`](super::stop_condition::ChatStopStage)
//! in the same registry, so a stop token (for which `ChatStopStage` returns
//! `Stop`) is never streamed — matching the non-streaming output semantics.
//! When the slot is empty (non-streaming requests, the interactive REPL) the
//! stage is a cheap no-op.

use std::sync::Mutex;

use tokenizers::Tokenizer;

use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};

/// Shared slot carrying the current turn's per-token callback.
///
/// Mirrors [`ChatStopSlot`](super::stop_condition::ChatStopSlot): the callback is
/// stored as a borrowed raw pointer that is only set for the synchronous
/// `run_turn` window (via [`ChatStreamSlot::arm`]'s RAII guard) and only touched
/// by the single decode thread (`INV-018`).
#[derive(Default)]
pub struct ChatStreamSlot {
    cb: Mutex<Option<*mut (dyn FnMut(u32) + 'static)>>,
}

// SAFETY: identical contract to `ChatStopSlot` — the raw pointer is live only for
// the `ChatStreamGuard` lifetime (= the synchronous driver run), accessed by a
// single thread, and serialized by the Mutex.
unsafe impl Send for ChatStreamSlot {}
unsafe impl Sync for ChatStreamSlot {}

impl ChatStreamSlot {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    /// Arm the slot with this turn's callback. The returned guard clears the slot
    /// on drop. `cb` must outlive the guard (it is a stack-local in the server
    /// handler that spans the synchronous `run_turn`).
    pub fn arm<'g>(
        self: &'g std::sync::Arc<Self>,
        cb: &'g mut dyn FnMut(u32),
    ) -> ChatStreamGuard<'g> {
        // lifetime erasure to 'static; the guard removes the pointer before `cb`
        // can be dropped (same reasoning as ChatStopSlot::arm).
        let ptr: *mut (dyn FnMut(u32) + 'static) =
            unsafe { std::mem::transmute::<*mut dyn FnMut(u32), _>(cb as *mut dyn FnMut(u32)) };
        *self.cb.lock().expect("ChatStreamSlot mutex poisoned") = Some(ptr);
        ChatStreamGuard { slot: self }
    }

    fn emit(&self, token: u32) {
        let guard = self.cb.lock().expect("ChatStreamSlot mutex poisoned");
        if let Some(ptr) = *guard {
            // SAFETY: ptr is valid for the arm guard's lifetime; single-threaded.
            unsafe { (*ptr)(token) };
        }
    }
}

/// RAII guard that clears the [`ChatStreamSlot`] on drop.
pub struct ChatStreamGuard<'g> {
    slot: &'g std::sync::Arc<ChatStreamSlot>,
}

impl Drop for ChatStreamGuard<'_> {
    fn drop(&mut self) {
        *self.slot.cb.lock().expect("ChatStreamSlot mutex poisoned") = None;
    }
}

/// `DecodeEnd` stage that streams each kept token to the armed callback.
pub struct ChatStreamStage {
    slot: std::sync::Arc<ChatStreamSlot>,
}

impl ChatStreamStage {
    pub fn new(slot: std::sync::Arc<ChatStreamSlot>) -> Self {
        Self { slot }
    }
}

impl PipelineStage for ChatStreamStage {
    fn name(&self) -> &str {
        "chat.stream"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::Persistent
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // Fires only at DecodeEnd, after ChatStopStage. prev_token == the token
        // that is about to be pushed to `generated` (kept token).
        if *phase == LifecyclePhase::DecodeEnd {
            self.slot.emit(ctx.step.prev_token);
        }
        Ok(StageOutcome::Continue)
    }
}

/// Incremental detokenizer producing UTF-8-safe text deltas.
///
/// Decoding token-by-token is unsafe across BPE / multi-byte boundaries (a piece
/// may be a fragment of a code point, byte-level BPE emits raw bytes). Instead we
/// decode the *growing* token vec each step and emit only the newly-completed
/// valid-UTF-8 suffix, holding back any trailing partial bytes until the next
/// token completes them.
pub struct IncDetok {
    ids: Vec<u32>,
    emitted_bytes: usize,
}

impl Default for IncDetok {
    fn default() -> Self {
        Self::new()
    }
}

impl IncDetok {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            emitted_bytes: 0,
        }
    }

    /// Append `tok` and return the newly-completed UTF-8 text (may be empty if the
    /// token only extends a partial code point or decodes to nothing, e.g. a
    /// special token under `skip_special_tokens`).
    pub fn push(&mut self, tok: u32, tk: &Tokenizer) -> String {
        self.ids.push(tok);
        self.new_suffix(tk)
    }

    /// Flush any remaining bytes (call once at end of turn).
    pub fn flush(&mut self, tk: &Tokenizer) -> String {
        self.new_suffix(tk)
    }

    fn new_suffix(&mut self, tk: &Tokenizer) -> String {
        let full = tk.decode(&self.ids, true).unwrap_or_default();
        let bytes = full.as_bytes();
        if bytes.len() <= self.emitted_bytes {
            return String::new();
        }
        let new = &bytes[self.emitted_bytes..];
        // Emit only the longest valid-UTF-8 prefix; hold back partial trailing bytes.
        let valid = match std::str::from_utf8(new) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid == 0 {
            return String::new();
        }
        // SAFETY of unwrap: `new[..valid]` is valid UTF-8 by construction above.
        let out = std::str::from_utf8(&new[..valid]).unwrap().to_string();
        self.emitted_bytes += valid;
        out
    }
}
