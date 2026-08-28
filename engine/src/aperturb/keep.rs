//! What a candidate leaves behind, and the causal boundary that follows from it.
//!
//! A candidate is described by the set of cache positions it retains — never by a mutated cache.
//! Scoring therefore works for shapes the container cannot hold: per-head budgets that differ by a
//! factor of ten are a legal [`KeepSets`], even though `CacheHandle::keep_per_head` rejects them
//! because `KVCache::current_pos` is one scalar shared by every head of a layer. Measuring a
//! candidate and committing it are separate questions, and only the second is constrained.

use std::fmt;

/// Retained cache positions for one whole model, as a flat CSR table.
///
/// `idx` holds every `(layer, kv_head)` list back to back, ascending within a list; `off` is the
/// prefix sum, so head `h` of layer `l` occupies `idx[off[l*n_kv + h] .. off[l*n_kv + h + 1]]`.
/// One allocation for the model instead of `L * n_kv` vectors, and `u32` rather than `usize`
/// because a 4-byte index halves what a device upload has to move.
///
/// The reference encodes the same thing as a rectangular `[n_kv, L_max]` tensor padded with the
/// sentinel `N` plus a companion `valid` mask, because torch needs a rectangle. The padding is
/// always a trailing run, so a per-head length is an exact — and simpler — encoding of it: there is
/// no masked column to skip in the middle, hence no `valid` mask anywhere below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeepSets {
    n_layers: usize,
    n_kv_heads: usize,
    idx: Vec<u32>,
    off: Vec<u32>,
}

/// Why a [`KeepSets`] could not be built or trusted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeepError {
    /// `push` was called out of order, or for a `(layer, head)` that does not exist.
    Order { layer: usize, kv_head: usize },
    /// A list is not strictly ascending. The whole causal-boundary derivation is a prefix count
    /// over this list, so it is exact only under this invariant — check it once, here, rather than
    /// per row.
    NotAscending { layer: usize, kv_head: usize },
    /// A position is at or beyond the resident count.
    OutOfRange {
        layer: usize,
        kv_head: usize,
        pos: u32,
        current_pos: usize,
    },
    /// A `(layer, head)` retains nothing. Every query row would then be blind and the score would
    /// have no cells to average, so this is rejected rather than scored as zero.
    Empty { layer: usize, kv_head: usize },
}

impl fmt::Display for KeepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Order { layer, kv_head } => {
                write!(
                    f,
                    "keep-set pushed out of order at (layer {layer}, kv_head {kv_head})"
                )
            }
            Self::NotAscending { layer, kv_head } => write!(
                f,
                "keep-set at (layer {layer}, kv_head {kv_head}) is not strictly ascending — the \
                 causal boundary is a prefix count and would be wrong"
            ),
            Self::OutOfRange {
                layer,
                kv_head,
                pos,
                current_pos,
            } => write!(
                f,
                "keep-set at (layer {layer}, kv_head {kv_head}) retains position {pos} but only \
                 {current_pos} tokens are resident"
            ),
            Self::Empty { layer, kv_head } => {
                write!(
                    f,
                    "keep-set at (layer {layer}, kv_head {kv_head}) retains nothing"
                )
            }
        }
    }
}

impl std::error::Error for KeepError {}

impl KeepSets {
    /// An empty table to be filled by `push` in `(layer, kv_head)` order.
    pub fn with_capacity(n_layers: usize, n_kv_heads: usize, total_idx: usize) -> Self {
        let mut off = Vec::with_capacity(n_layers * n_kv_heads + 1);
        off.push(0);
        Self {
            n_layers,
            n_kv_heads,
            idx: Vec::with_capacity(total_idx),
            off,
        }
    }

    /// Retain everything — the identity candidate. Scoring it must produce exactly zero
    /// perturbation, which is the cheapest gate on the whole measurement operator.
    pub fn identity(n_layers: usize, n_kv_heads: usize, current_pos: usize) -> Self {
        let mut s = Self::with_capacity(n_layers, n_kv_heads, n_layers * n_kv_heads * current_pos);
        for _ in 0..n_layers * n_kv_heads {
            s.idx.extend(0..current_pos as u32);
            s.off.push(s.idx.len() as u32);
        }
        s
    }

    /// One ascending list shared by every layer and head — the sliding / streaming shape.
    pub fn uniform(n_layers: usize, n_kv_heads: usize, keep: &[u32]) -> Self {
        let mut s = Self::with_capacity(n_layers, n_kv_heads, n_layers * n_kv_heads * keep.len());
        for _ in 0..n_layers * n_kv_heads {
            s.idx.extend_from_slice(keep);
            s.off.push(s.idx.len() as u32);
        }
        s
    }

    /// Append the next `(layer, kv_head)` list. Must be called exactly `n_layers * n_kv_heads`
    /// times, in row-major order.
    pub fn push(
        &mut self,
        layer: usize,
        kv_head: usize,
        ascending: &[u32],
    ) -> Result<(), KeepError> {
        let want = self.off.len() - 1;
        if layer >= self.n_layers
            || kv_head >= self.n_kv_heads
            || want != layer * self.n_kv_heads + kv_head
        {
            return Err(KeepError::Order { layer, kv_head });
        }
        self.idx.extend_from_slice(ascending);
        self.off.push(self.idx.len() as u32);
        Ok(())
    }

    /// `true` once every `(layer, kv_head)` list has been pushed.
    pub fn is_complete(&self) -> bool {
        self.off.len() == self.n_layers * self.n_kv_heads + 1
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// The retained positions of one `(layer, kv_head)`, ascending.
    #[inline]
    pub fn head(&self, layer: usize, kv_head: usize) -> &[u32] {
        let i = layer * self.n_kv_heads + kv_head;
        &self.idx[self.off[i] as usize..self.off[i + 1] as usize]
    }

    /// Total retained positions across the model — the byte proxy the budget band is defined on.
    pub fn total(&self) -> usize {
        self.idx.len()
    }

    /// Ascending, in range, non-empty, and fully populated. Checked once so the hot loop can index
    /// without guards.
    pub fn validate(&self, current_pos: usize) -> Result<(), KeepError> {
        if !self.is_complete() {
            let want = self.off.len() - 1;
            return Err(KeepError::Order {
                layer: want / self.n_kv_heads,
                kv_head: want % self.n_kv_heads,
            });
        }
        for layer in 0..self.n_layers {
            for kv_head in 0..self.n_kv_heads {
                let h = self.head(layer, kv_head);
                if h.is_empty() {
                    return Err(KeepError::Empty { layer, kv_head });
                }
                if *h.last().expect("non-empty") as usize >= current_pos {
                    return Err(KeepError::OutOfRange {
                        layer,
                        kv_head,
                        pos: *h.last().expect("non-empty"),
                        current_pos,
                    });
                }
                if h.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(KeepError::NotAscending { layer, kv_head });
                }
            }
        }
        Ok(())
    }

    /// `true` when some head of some layer retains a different number of positions than another.
    ///
    /// Scoring does not care; committing does. The container gives every KV head of a layer the
    /// same length, so a ragged winner is measurable but not applicable, and the caller should be
    /// told at selection time rather than by a failed mutation later.
    pub fn is_ragged(&self) -> bool {
        (0..self.n_layers).any(|l| {
            let first = self.head(l, 0).len();
            (1..self.n_kv_heads).any(|h| self.head(l, h).len() != first)
        })
    }
}

/// The causal boundary of one layer: for query row `t` and KV head `h`, the index of the last
/// retained position the row is allowed to see.
///
/// Row `t` sits at absolute position `current_pos - rows + t` and may attend to retained positions
/// at or below its own. Because the retained list is ascending, "at or below" is a prefix, so the
/// boundary is one integer: `|{ j : keep[j] <= pos }| - 1`. It is `-1` when the row sees nothing —
/// possible when a candidate keeps only a short recent tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyPos {
    kp: Vec<i32>,
    n_kv_heads: usize,
    rows: usize,
}

impl KeyPos {
    /// Derive the boundary for every `(kv_head, row)` of one layer.
    ///
    /// Both sequences are ascending, so this is a single merge per head rather than a binary search
    /// per row.
    pub fn for_layer(keep: &KeepSets, layer: usize, current_pos: usize, rows: usize) -> Self {
        let n_kv_heads = keep.n_kv_heads();
        let first = current_pos.saturating_sub(rows);
        let mut kp = vec![-1i32; n_kv_heads * rows];
        for h in 0..n_kv_heads {
            let list = keep.head(layer, h);
            let mut j = 0usize;
            for t in 0..rows {
                let pos = (first + t) as u32;
                while j < list.len() && list[j] <= pos {
                    j += 1;
                }
                kp[h * rows + t] = j as i32 - 1;
            }
        }
        Self {
            kp,
            n_kv_heads,
            rows,
        }
    }

    /// The boundary for `(kv_head, row)`; `-1` means the row is blind in this layer.
    #[inline]
    pub fn get(&self, kv_head: usize, row: usize) -> i32 {
        self.kp[kv_head * self.rows + row]
    }

    /// The number of retained positions row `t` of head `h` may attend to.
    #[inline]
    pub fn admitted(&self, kv_head: usize, row: usize) -> usize {
        (self.get(kv_head, row) + 1).max(0) as usize
    }

    /// Rows that see at least one retained position in **every** head of this layer, as a bitmask
    /// over `t < rows`.
    ///
    /// The reference AND-reduces this across layers and heads and then drops the failing rows from
    /// the aggregation — it still computes them, with the boundary clamped, so the trailing-row
    /// slice stays aligned. `rows <= 32` by construction (the reference sweeps R up to 32), so a
    /// `u32` carries the whole mask.
    pub fn visible_rows(&self) -> u32 {
        let mut m = 0u32;
        for t in 0..self.rows.min(32) {
            if (0..self.n_kv_heads).all(|h| self.get(h, t) >= 0) {
                m |= 1 << t;
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_boundary_is_the_row_position() {
        let k = KeepSets::identity(2, 3, 40);
        k.validate(40).expect("identity is valid");
        assert!(!k.is_ragged());
        let kp = KeyPos::for_layer(&k, 1, 40, 6);
        for h in 0..3 {
            for t in 0..6 {
                // row t is absolute position 34+t, and the retained list is 0..40, so the last
                // admissible index is the position itself.
                assert_eq!(kp.get(h, t), 34 + t as i32);
            }
        }
        assert_eq!(kp.visible_rows(), 0b111111);
    }

    #[test]
    fn boundary_is_a_prefix_count() {
        let mut k = KeepSets::with_capacity(1, 1, 4);
        k.push(0, 0, &[0, 5, 36, 39]).unwrap();
        k.validate(40).unwrap();
        let kp = KeyPos::for_layer(&k, 0, 40, 6); // rows at 34..39
        assert_eq!(kp.get(0, 0), 1); // pos 34: {0,5}
        assert_eq!(kp.get(0, 2), 2); // pos 36: {0,5,36}
        assert_eq!(kp.get(0, 5), 3); // pos 39: all four
        assert_eq!(kp.admitted(0, 0), 2);
    }

    #[test]
    fn blind_rows_report_minus_one_and_drop_out_of_the_mask() {
        let mut k = KeepSets::with_capacity(1, 1, 2);
        k.push(0, 0, &[38, 39]).unwrap();
        let kp = KeyPos::for_layer(&k, 0, 40, 6);
        assert_eq!(kp.get(0, 0), -1); // pos 34 sees nothing
        assert_eq!(kp.admitted(0, 0), 0);
        assert_eq!(kp.visible_rows(), 0b110000); // only rows 4,5 (pos 38,39)
    }

    #[test]
    fn a_row_is_visible_only_when_every_head_sees_something() {
        let mut k = KeepSets::with_capacity(1, 2, 8);
        k.push(0, 0, &[0, 39]).unwrap();
        k.push(0, 1, &[38, 39]).unwrap();
        let kp = KeyPos::for_layer(&k, 0, 40, 6);
        assert_eq!(kp.visible_rows(), 0b110000);
    }

    #[test]
    fn validate_rejects_the_shapes_that_would_silently_misscore() {
        let mut k = KeepSets::with_capacity(1, 1, 3);
        k.push(0, 0, &[5, 5, 7]).unwrap();
        assert!(matches!(
            k.validate(40),
            Err(KeepError::NotAscending { .. })
        ));

        let mut k = KeepSets::with_capacity(1, 1, 2);
        k.push(0, 0, &[5, 40]).unwrap();
        assert!(matches!(k.validate(40), Err(KeepError::OutOfRange { .. })));

        let mut k = KeepSets::with_capacity(1, 1, 0);
        k.push(0, 0, &[]).unwrap();
        assert!(matches!(k.validate(40), Err(KeepError::Empty { .. })));

        let mut k = KeepSets::with_capacity(2, 1, 1);
        k.push(0, 0, &[1]).unwrap();
        assert!(matches!(k.validate(40), Err(KeepError::Order { .. })));
    }

    #[test]
    fn raggedness_is_detected_per_layer() {
        let mut k = KeepSets::with_capacity(1, 2, 5);
        k.push(0, 0, &[1, 2]).unwrap();
        k.push(0, 1, &[1, 2, 3]).unwrap();
        assert!(k.is_ragged());
    }
}
