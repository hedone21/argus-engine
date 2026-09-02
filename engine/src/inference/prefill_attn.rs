//! The prompt attention the forward captured, and the cache frame it describes.
//!
//! A prefill-end technique (SnapKV, PyramidKV) ranks keys by how much the prompt's own queries
//! attended to them. The forward measures that once, on the last prefill chunk, and
//! [`crate::session::forward::ModelForward`] publishes it here for whoever asks. What makes it
//! more than a plain buffer is that its columns are **cache positions**: row `q`, column `p` is
//! query head `q`'s attention to whatever key sits in slot `p`. A compaction renumbers those
//! slots, and a capture read afterwards names different keys than the ones it measured.
//!
//! The buffer alone cannot say whether that happened, and the consequence of guessing is not a
//! refusal but a wrong answer: the candidate ranks the surviving keys by another key's score and
//! the engine applies the result. So the capture carries the frame with it. Every compaction the
//! decode loop observes invalidates it, except the one it was **gathered** through —
//! [`gather`](Self::gather) rebuilds the columns in the new numbering, which is exact rather than
//! approximate, because a key's pooled prompt attention is a property of the prompt and does not
//! change when some other key leaves the cache.

/// Per-layer prompt attention, `[layer][n_heads_q * prefix_len]`, SUM-pooled over the trailing
/// observation window, plus the frame those columns are numbered in.
#[derive(Clone, PartialEq)]
pub struct PrefillAttn {
    rows: Vec<Vec<f32>>,
    /// The occupancy a compaction left behind, when [`Self::gather`] has already carried `rows`
    /// through that compaction and the decode loop has not yet observed it.
    ///
    /// `None` means the rows describe the frame as it stands, so the next shrink — whoever caused
    /// it — makes them stale. The claim is consumed when it is checked: a second shrink is a
    /// second compaction, and only one of them was gathered through.
    gathered_at: Option<usize>,
}

impl PrefillAttn {
    /// A freshly captured prompt attention, describing the cache as prefill left it.
    pub fn captured(rows: Vec<Vec<f32>>) -> Self {
        Self {
            rows,
            gathered_at: None,
        }
    }

    /// The per-layer rows, in the frame this capture describes.
    pub fn rows(&self) -> &[Vec<f32>] {
        &self.rows
    }

    /// Carry the capture through a compaction, given the positions each layer kept.
    ///
    /// `keep(layer, kv_head)` is that head's retained positions in the **pre-compaction**
    /// numbering — the same ascending, unique list the engine applied
    /// ([`CacheHandle::keep_per_head`](argus_extension_api::CacheHandle) validates both). New
    /// position `j` holds what old position `keep[j]` held, so new column `j` is old column
    /// `keep[j]`; the retained positions at or past `prefix_len` are the decode tail this capture
    /// never covered, and they drop off the end where they already were.
    ///
    /// `None` when the capture cannot be carried, and the caller must then drop it rather than
    /// keep a buffer whose columns no longer name the keys they measured:
    ///
    /// - a layer whose heads disagree on how many prompt positions survived. The rows are laid out
    ///   at one `prefix_len` per layer, so per-head prefixes of different lengths have no
    ///   representation here — the [`KeepSets`](crate::aperturb::KeepSets) raggedness wall in the
    ///   capture's own terms.
    /// - a layer that kept no prompt position at all, or one the caller had no keep-set for. There
    ///   is nothing left to rank, so the capture has stopped describing anything.
    pub fn gather<'k>(
        &self,
        occupancy: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
        keep: impl Fn(usize, usize) -> Option<&'k [usize]>,
    ) -> Option<Self> {
        // Query heads outnumber KV heads under GQA; head `q` reads the KV head that owns it, so it
        // is renumbered by that head's keep-set.
        let group = n_heads_q.checked_div(n_kv_heads).filter(|g| *g > 0)?;
        let mut out = Vec::with_capacity(self.rows.len());
        for (layer, row) in self.rows.iter().enumerate() {
            let prefix_len = row.len() / n_heads_q.max(1);
            if prefix_len == 0 {
                return None;
            }
            // Resolve every head's surviving prefix first: a disagreement is a property of the
            // layer, and finding it half-way through would leave rows already written at a width
            // the rest cannot match. `partition_point` is exact because the list is ascending.
            let mut kept: Vec<&[usize]> = Vec::with_capacity(n_kv_heads);
            for h in 0..n_kv_heads {
                let full = keep(layer, h)?;
                kept.push(&full[..full.partition_point(|&p| p < prefix_len)]);
            }
            let new_prefix = kept[0].len();
            if new_prefix == 0 || kept.iter().any(|k| k.len() != new_prefix) {
                return None;
            }
            let mut next = vec![0.0f32; n_heads_q * new_prefix];
            for q in 0..n_heads_q {
                let src = &row[q * prefix_len..(q + 1) * prefix_len];
                let dst = &mut next[q * new_prefix..(q + 1) * new_prefix];
                for (slot, &p) in dst.iter_mut().zip(kept[q / group]) {
                    *slot = src[p];
                }
            }
            out.push(next);
        }
        Some(Self {
            rows: out,
            gathered_at: Some(occupancy),
        })
    }

    /// Whether a compaction that left `occupancy` resident is the one this capture was gathered
    /// through. Consumes the claim, so the next shrink is judged on its own.
    pub fn survives_shrink_to(&mut self, occupancy: usize) -> bool {
        self.gathered_at.take() == Some(occupancy)
    }
}

impl std::fmt::Debug for PrefillAttn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillAttn")
            .field("layers", &self.rows.len())
            .field("cols", &self.rows.first().map_or(0, Vec::len))
            .field("gathered_at", &self.gathered_at)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[n_heads_q * prefix_len]` where head `q`'s column `p` is `q * 100 + p`, so a gathered row
    /// says both which head it came from and which old column.
    fn stamped(n_heads_q: usize, prefix_len: usize) -> Vec<f32> {
        (0..n_heads_q)
            .flat_map(|q| (0..prefix_len).map(move |p| (q * 100 + p) as f32))
            .collect()
    }

    /// The columns follow the keys. A layer-wide keep-set that drops positions 1 and 3 of a 5-long
    /// prompt leaves a 3-long capture holding exactly the columns of 0, 2 and 4 — the tail
    /// positions the keep-set also retained are past the prompt and simply do not appear.
    ///
    /// Mutation-proof: gathering by `j` instead of `keep[j]` gives `[0, 1, 2]`.
    #[test]
    fn the_surviving_keys_keep_their_own_prompt_attention() {
        let pfa = PrefillAttn::captured(vec![stamped(2, 5)]);
        let keep = [0usize, 2, 4, 5];
        let g = pfa
            .gather(4, 2, 1, |_, _| Some(&keep[..]))
            .expect("a layer-wide keep-set carries");
        assert_eq!(g.rows()[0], vec![0.0, 2.0, 4.0, 100.0, 102.0, 104.0]);
    }

    /// Under GQA each query head is renumbered by the KV head that owns it, not by head 0's list.
    ///
    /// Mutation-proof: indexing `kept[0]` for every `q` makes head 2 report `[200, 202]`.
    #[test]
    fn a_query_head_follows_its_own_kv_head() {
        let pfa = PrefillAttn::captured(vec![stamped(4, 4)]);
        let (even, odd) = ([0usize, 2], [1usize, 3]);
        let g = pfa
            .gather(2, 4, 2, |_, h| {
                Some(if h == 0 { &even[..] } else { &odd[..] })
            })
            .expect("a per-head keep-set of equal length carries");
        assert_eq!(
            g.rows()[0],
            vec![0.0, 2.0, 100.0, 102.0, 201.0, 203.0, 301.0, 303.0]
        );
    }

    /// Heads that keep different NUMBERS of prompt positions have no representation: the rows are
    /// one `prefix_len` per layer. Dropping the capture is the honest answer.
    #[test]
    fn heads_that_disagree_on_the_prompt_cannot_be_carried() {
        let pfa = PrefillAttn::captured(vec![stamped(2, 6)]);
        // Both heads keep 3 positions, but head 1 spends one of them on the decode tail, so only
        // 2 of its columns are prompt. Equal keep lengths, unequal prompt prefixes.
        let (all_prompt, one_tail) = ([0usize, 1, 2], [0usize, 1, 6]);
        assert!(
            pfa.gather(3, 2, 2, |_, h| Some(if h == 0 {
                &all_prompt[..]
            } else {
                &one_tail[..]
            }))
            .is_none()
        );
    }

    /// A keep-set that retains only decode tail leaves the capture describing nothing.
    #[test]
    fn a_capture_with_no_prompt_left_is_dropped() {
        let pfa = PrefillAttn::captured(vec![stamped(2, 3)]);
        let tail_only = [3usize, 4];
        assert!(pfa.gather(2, 2, 1, |_, _| Some(&tail_only[..])).is_none());
    }

    /// Every layer must be accounted for; a missing keep-set is a capture that cannot be trusted
    /// layer-wide, not one to carry partially.
    #[test]
    fn a_layer_without_a_keep_set_drops_the_whole_capture() {
        let pfa = PrefillAttn::captured(vec![stamped(2, 4), stamped(2, 4)]);
        let keep = [0usize, 1];
        assert!(
            pfa.gather(2, 2, 1, |l, _| (l == 0).then_some(&keep[..]))
                .is_none()
        );
    }

    /// A fresh capture claims nothing: the first shrink it meets invalidates it.
    ///
    /// Mutation-proof: initialising `gathered_at` to `Some(occupancy)` anywhere makes this pass a
    /// stale capture through.
    #[test]
    fn a_capture_that_was_never_gathered_does_not_survive_a_shrink() {
        let mut pfa = PrefillAttn::captured(vec![stamped(2, 4)]);
        assert!(!pfa.survives_shrink_to(2));
    }

    /// The gathered capture survives exactly the compaction it was gathered through — and only
    /// that one. A second shrink is a second compaction, which nothing carried it through.
    ///
    /// Mutation-proof: leaving `gathered_at` set (peek instead of take) lets the second shrink pass.
    #[test]
    fn a_gathered_capture_survives_its_own_compaction_once() {
        let keep = [0usize, 2, 4, 5];
        let mut g = PrefillAttn::captured(vec![stamped(2, 5)])
            .gather(4, 2, 1, |_, _| Some(&keep[..]))
            .expect("carries");
        assert!(g.survives_shrink_to(4));
        assert!(!g.survives_shrink_to(4));
    }

    /// A shrink to a different occupancy than the gather reported is a different compaction.
    #[test]
    fn a_gathered_capture_does_not_excuse_someone_elses_shrink() {
        let keep = [0usize, 2, 4, 5];
        let mut g = PrefillAttn::captured(vec![stamped(2, 5)])
            .gather(4, 2, 1, |_, _| Some(&keep[..]))
            .expect("carries");
        assert!(!g.survives_shrink_to(3));
    }
}
