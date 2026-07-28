//! Position-encoding frequency rescaling — the `rope_scaling` half of RoPE.
//!
//! Plain RoPE turns position into rotation: dimension pair `i` of every Q/K head is rotated by
//! `position * freq[i]`, where `freq[i] = theta^(-2i/head_dim)`. Low `i` rotates fast (short
//! wavelength, resolves nearby tokens); high `i` rotates slowly (long wavelength, carries
//! long-range structure).
//!
//! **Why a rescaling exists.** Llama 3 was trained at 8192 tokens. Its slowest dimensions have
//! wavelengths LONGER than that, so during training they never completed a full turn — pushing them
//! past 8192 at inference feeds the model angles it has never seen. Llama 3.1 fixes this by
//! stretching exactly those dimensions back inside the trained range, leaving the fast ones alone:
//!
//! ```text
//! wavelen = 2*pi / freq
//! wavelen > orig_max/low_freq_factor   ->  freq /= factor        (slow dims: stretched)
//! wavelen < orig_max/high_freq_factor  ->  freq                  (fast dims: untouched)
//! otherwise                            ->  smooth blend of the two
//! ```
//!
//! **This is NOT a switch that turns on past 8192.** `freq[i]` is a property of the DIMENSION, not
//! of the position, so the rescaling applies from position 1. Ignoring it does not merely degrade
//! long context — it runs a different model everywhere, with an error that grows linearly with
//! position (measured against HuggingFace: 17.6% divergence in per-head attention peaks at 615
//! tokens, 54.6% at 15.7 K).
//!
//! Every Llama 3.1/3.2 checkpoint declares `"rope_type": "llama3"`. For `head_dim = 64`,
//! `theta = 5e5` that puts 14 of 32 dimension pairs (44%) in the fully-rescaled band.

/// Per-dimension frequency rescaling parameters, as `config.json` states them.
///
/// [`Self::NONE`] is the exact identity — `factor == 1.0` makes every branch of [`Self::scale_freq`]
/// return its input unchanged — so a backend kernel can apply the rule UNCONDITIONALLY instead of
/// branching on "is scaling configured". A model without `rope_scaling` and a model with a no-op
/// `rope_scaling` then travel the same code path, which is one fewer way for the two to disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeFreqScaling {
    /// How much the slowest dimensions are stretched (Llama 3.1 8B: 8, Llama 3.2 1B/3B: 32).
    pub factor: f32,
    /// Wavelengths longer than `original_max_position_embeddings / low_freq_factor` are stretched.
    pub low_freq_factor: f32,
    /// Wavelengths shorter than `original_max_position_embeddings / high_freq_factor` are kept.
    pub high_freq_factor: f32,
    /// The context length the model was originally trained at (8192 for Llama 3).
    pub original_max_position_embeddings: f32,
}

impl RopeFreqScaling {
    /// Identity. The `low`/`high`/`orig` values are the Llama 3 defaults purely so the medium-band
    /// denominator is non-degenerate; with `factor == 1.0` none of them can change a frequency.
    pub const NONE: Self = Self {
        factor: 1.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192.0,
    };

    /// Whether this leaves every frequency untouched.
    pub fn is_identity(&self) -> bool {
        self.factor == 1.0
    }

    /// Apply the `llama3` rule to one base frequency.
    ///
    /// Mirrors HuggingFace `transformers`' `_compute_llama3_parameters`. The CUDA and OpenCL RoPE
    /// kernels reimplement these same five lines; [`tests::matches_huggingface_llama3_parameters`]
    /// pins this copy against values taken from the reference implementation, and the cross-backend
    /// RoPE equality tests pin the others against this one.
    #[inline]
    pub fn scale_freq(&self, freq: f32) -> f32 {
        // Cheap exit that is also the exact identity — see `NONE`.
        if self.factor == 1.0 {
            return freq;
        }
        let wavelen = std::f32::consts::TAU / freq;
        let low_wavelen = self.original_max_position_embeddings / self.low_freq_factor;
        let high_wavelen = self.original_max_position_embeddings / self.high_freq_factor;
        if wavelen > low_wavelen {
            freq / self.factor
        } else if wavelen < high_wavelen {
            freq
        } else {
            let span = self.high_freq_factor - self.low_freq_factor;
            if span == 0.0 {
                // Degenerate config: no band to interpolate across. Leave the frequency alone
                // rather than dividing by zero.
                return freq;
            }
            let smooth =
                (self.original_max_position_embeddings / wavelen - self.low_freq_factor) / span;
            (1.0 - smooth) * freq / self.factor + smooth * freq
        }
    }
}

impl Default for RopeFreqScaling {
    fn default() -> Self {
        Self::NONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_freq(i: usize, head_dim: usize, theta: f32) -> f32 {
        theta.powf(-2.0 * (i as f32) / (head_dim as f32))
    }

    /// Values produced by HuggingFace's `_compute_llama3_parameters` for the two shipped geometries.
    /// Checked as a RATIO (`scaled / base`) so the assertion does not depend on `powf` reproducing
    /// the base frequency bit-for-bit.
    #[test]
    fn matches_huggingface_llama3_parameters() {
        // Llama 3.1 8B: head_dim 128, theta 5e5, factor 8.
        let s8 = RopeFreqScaling {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        };
        for (i, expect) in [(0usize, 1.0f32), (32, 0.371122), (63, 0.125)] {
            let f = base_freq(i, 128, 500_000.0);
            let ratio = s8.scale_freq(f) / f;
            assert!(
                (ratio - expect).abs() < 1e-5,
                "8B dim {i}: ratio {ratio} != {expect}"
            );
        }

        // Llama 3.2 1B/3B: head_dim 64, theta 5e5, factor 32.
        let s32 = RopeFreqScaling { factor: 32.0, ..s8 };
        for (i, expect) in [(0usize, 1.0f32), (16, 0.303743), (31, 0.031_25)] {
            let f = base_freq(i, 64, 500_000.0);
            let ratio = s32.scale_freq(f) / f;
            assert!(
                (ratio - expect).abs() < 1e-5,
                "1B dim {i}: ratio {ratio} != {expect}"
            );
        }
    }

    /// The three bands must actually be reachable — a rule that only ever hits one branch would pass
    /// the value checks above by accident.
    #[test]
    fn all_three_bands_are_exercised_by_a_real_geometry() {
        let s = RopeFreqScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        };
        let (mut kept, mut blended, mut stretched) = (0, 0, 0);
        for i in 0..32 {
            let f = base_freq(i, 64, 500_000.0);
            let r = s.scale_freq(f) / f;
            if (r - 1.0).abs() < 1e-6 {
                kept += 1;
            } else if (r - 1.0 / 32.0).abs() < 1e-6 {
                stretched += 1;
            } else {
                blended += 1;
            }
        }
        assert_eq!(
            (kept, blended, stretched),
            (15, 3, 14),
            "band split changed — the llama3 rule or the geometry moved"
        );
    }

    #[test]
    fn none_is_the_exact_identity() {
        for i in 0..64 {
            let f = base_freq(i, 128, 500_000.0);
            assert_eq!(RopeFreqScaling::NONE.scale_freq(f), f);
        }
    }
}
