// ── Action result ──────────────────────────────────────────────────

/// Outcome of a handler's action.
#[derive(Debug, Clone)]
pub enum ActionResult {
    /// No action was taken.
    NoOp,
    /// Tokens were evicted from the cache.
    Evicted {
        tokens_removed: usize,
        new_pos: usize,
    },
    /// KV data was swapped to secondary storage (disk offload).
    Swapped { tokens_swapped: usize },
    /// Decoder layer weights were swapped to a lower-precision dtype (weight swap).
    WeightSwapped {
        /// Number of layers whose weights were atomically replaced.
        layers_changed: usize,
        /// Estimated bytes freed (primary weight pages released via madvise).
        freed_bytes: usize,
        /// Wall-clock time for the full swap batch, in milliseconds.
        duration_ms: f64,
    },
}

impl ActionResult {
    /// Whether this result represents an actual action (not NoOp).
    pub fn is_action(&self) -> bool {
        !matches!(self, ActionResult::NoOp)
    }
}
