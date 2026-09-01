//! On-disk form of the output-projection basis, so the decomposition is paid once per model.
//!
//! [`OutputBasis::from_weights`](super::OutputBasis::from_weights) is the expensive part of the
//! measurement and the only part that does not depend on the prompt: one subspace iteration per
//! layer, 28 s for a 1B model and 12 min for an 8B one on a 20-thread desktop. Nothing consumed it
//! across processes, so every run paid it again. This module is that missing half — write the table
//! once on a host, load it everywhere else.
//!
//! ## Why a shipped artifact and not a cache
//!
//! The deployment target is Android, and the numbers above do not survive the move: a phone has
//! fewer and slower cores and thermally throttles, so factoring on device is minutes-to-hours, not
//! seconds. So the device never factors. The file is produced on a host, travels with the
//! application, and the device path is load-only — which is also why a header that does not match
//! is an error here rather than a quiet fall back to computing it.
//!
//! There is no default location either: Android has no `~/.cache`, and the writable directory is
//! whatever the application hands the engine. The path therefore comes in from the caller.
//!
//! ## Format
//!
//! Little-endian throughout, matching the tensor dump beside it, so a file written on the host
//! loads on the device.
//!
//! ```text
//! offset  size  field
//!      0     8  magic "ARGUSWOB"
//!      8     4  version (u32)
//!     12     4  n_layers (u32)
//!     16     4  d (u32)              — the projection's input width, what the rows live in
//!     20     4  rank (u32)
//!     24     8  frac (f64)           — the fraction the rank came from; it names the metric key
//!     32     8  residual_max (f64)   — worst eigenproblem residual over the layers
//!     40     8  wo_digest (u64)      — identity of the weights the basis was built from
//!     48     …  n_layers × d × rank  f32, layer-major, row-major within a layer
//! ```
//!
//! `residual_max` rides along because the dump publishes it as the evidence that the truncation
//! converged; recomputing it would mean redoing the decomposition, which is the whole point.
//!
//! `wo_digest` is what makes a wrong file loud. The shape fields alone would accept a basis built
//! from a different checkpoint of the same architecture — a base model's table used to score an
//! instruct model's cache — and the scores would look entirely ordinary. The reader digests the
//! projections it is actually going to measure and refuses anything else. That costs one pass over
//! `W_o` (a read and a dequantize, no arithmetic beyond it), which is the price of knowing the
//! table belongs to this model.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::{AperturbError, OutputBasis};

/// File magic. Distinct from the tensor dump's `ARGUSAPT`.
pub const MAGIC: &[u8; 8] = b"ARGUSWOB";

/// Format version.
pub const VERSION: u32 = 1;

/// Bytes before the first f32.
pub const HEADER_BYTES: usize = 48;

/// What the file must say about itself before its numbers are used.
///
/// Every field is checked, and a mismatch in any of them is an error rather than a warning: there
/// is nothing downstream that could notice a basis built for a different model, so a score computed
/// from one is wrong by an unknown amount and looks fine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Expect {
    pub n_layers: usize,
    pub d: usize,
    pub rank: usize,
    pub frac: f64,
    pub wo_digest: u64,
}

/// Why a basis file could not be used.
#[derive(Debug)]
pub enum BasisFileError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// Not a basis file at all.
    Magic { path: PathBuf },
    /// A basis file, but from a format this build does not read.
    Version { path: PathBuf, got: u32 },
    /// A header field disagrees with the model being measured.
    Mismatch {
        path: PathBuf,
        field: &'static str,
        got: String,
        want: String,
    },
    /// The payload is not exactly `n_layers × d × rank` floats.
    Payload {
        path: PathBuf,
        got: usize,
        want: usize,
    },
    /// The untruncated arm carries no rank fraction, so it has no key and nothing to name a file by.
    Untruncated,
    /// The bytes read back are shaped wrong — cannot happen through this module's own writer.
    Basis(AperturbError),
}

impl std::fmt::Display for BasisFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "aperturb basis {}: {source}", path.display()),
            Self::Magic { path } => write!(
                f,
                "aperturb basis {}: not a basis file (bad magic)",
                path.display()
            ),
            Self::Version { path, got } => write!(
                f,
                "aperturb basis {}: format version {got}, this build reads {VERSION}",
                path.display()
            ),
            Self::Mismatch {
                path,
                field,
                got,
                want,
            } => write!(
                f,
                "aperturb basis {}: {field} is {got}, but this model needs {want} — the file was \
                 built for a different model or a different rank, and its scores would be wrong \
                 without being visibly wrong",
                path.display()
            ),
            Self::Payload { path, got, want } => write!(
                f,
                "aperturb basis {}: {got} floats after the header, expected {want} (truncated or \
                 trailing bytes)",
                path.display()
            ),
            Self::Untruncated => write!(
                f,
                "aperturb basis: the untruncated arm has no rank fraction and is not stored — it is \
                 the projection itself, which the model already carries"
            ),
            Self::Basis(e) => write!(f, "aperturb basis: {e}"),
        }
    }
}

impl std::error::Error for BasisFileError {}

/// Identity of the weights a basis was built from: FNV-1a over each layer's f32 bit patterns.
///
/// A word at a time rather than a byte at a time — 2 GB of `W_o` at 8B would otherwise cost more
/// than it is worth. Not a cryptographic digest: it is here to catch the wrong file, not a forged
/// one. The layer index and length are folded in so reordering or resizing the layers changes it
/// even when the floats do not.
///
/// Layer at a time so the load path can digest a projection and drop it. Holding all of `W_o` to
/// check a 1 MB table would cost more memory than factoring it saved.
#[derive(Clone, Copy, Debug)]
pub struct Digest {
    h: u64,
    next: usize,
}

impl Digest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    /// Start a digest over `n_layers` projections.
    pub fn new(n_layers: usize) -> Self {
        let mut d = Self {
            h: Self::OFFSET,
            next: 0,
        };
        d.eat(n_layers as u64);
        d
    }

    #[inline]
    fn eat(&mut self, x: u64) {
        self.h ^= x;
        self.h = self.h.wrapping_mul(Self::PRIME);
    }

    /// Fold in the next layer. Layers must arrive in order; the index is part of the digest.
    pub fn layer(&mut self, w: &[f32]) {
        let l = self.next;
        self.next += 1;
        self.eat(l as u64);
        self.eat(w.len() as u64);
        for x in w {
            self.eat(x.to_bits() as u64);
        }
    }

    pub fn finish(self) -> u64 {
        self.h
    }
}

/// [`Digest`] over projections that are all in hand at once.
pub fn digest(wo: &[Vec<f32>]) -> u64 {
    let mut d = Digest::new(wo.len());
    for w in wo {
        d.layer(w);
    }
    d.finish()
}

/// Write `basis` and the decomposition's worst residual to `path`.
pub fn write(
    path: &Path,
    basis: &OutputBasis,
    residual_max: f64,
    wo_digest: u64,
) -> Result<(), BasisFileError> {
    let frac = basis.frac().ok_or(BasisFileError::Untruncated)?;
    let io = |source| BasisFileError::Io {
        path: path.to_path_buf(),
        source,
    };
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    let mut f = BufWriter::new(File::create(path).map_err(io)?);
    f.write_all(MAGIC).map_err(io)?;
    for x in [
        VERSION,
        basis.n_layers() as u32,
        basis.d() as u32,
        basis.rank() as u32,
    ] {
        f.write_all(&x.to_le_bytes()).map_err(io)?;
    }
    f.write_all(&frac.to_le_bytes()).map_err(io)?;
    f.write_all(&residual_max.to_le_bytes()).map_err(io)?;
    f.write_all(&wo_digest.to_le_bytes()).map_err(io)?;
    for l in 0..basis.n_layers() {
        for x in basis.layer(l) {
            f.write_all(&x.to_le_bytes()).map_err(io)?;
        }
    }
    f.flush().map_err(io)?;
    Ok(())
}

/// Load the basis at `path`, or fail if it does not belong to the model described by `expect`.
///
/// Returns the basis and the residual the decomposition reported when it was written.
pub fn read(path: &Path, expect: &Expect) -> Result<(OutputBasis, f64), BasisFileError> {
    let io = |source| BasisFileError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(io)?
        .read_to_end(&mut bytes)
        .map_err(io)?;
    if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
        return Err(BasisFileError::Magic {
            path: path.to_path_buf(),
        });
    }
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let version = u32_at(8);
    if version != VERSION {
        return Err(BasisFileError::Version {
            path: path.to_path_buf(),
            got: version,
        });
    }
    let n_layers = u32_at(12) as usize;
    let d = u32_at(16) as usize;
    let rank = u32_at(20) as usize;
    let frac = f64::from_bits(u64_at(24));
    let residual_max = f64::from_bits(u64_at(32));
    let wo_digest = u64_at(40);

    let mismatch = |field: &'static str, got: String, want: String| BasisFileError::Mismatch {
        path: path.to_path_buf(),
        field,
        got,
        want,
    };
    if n_layers != expect.n_layers {
        return Err(mismatch(
            "layer count",
            n_layers.to_string(),
            expect.n_layers.to_string(),
        ));
    }
    if d != expect.d {
        return Err(mismatch(
            "projection width",
            d.to_string(),
            expect.d.to_string(),
        ));
    }
    if rank != expect.rank {
        return Err(mismatch("rank", rank.to_string(), expect.rank.to_string()));
    }
    // Bit equality, not a tolerance: the fraction is a literal that names the metric key, so two
    // values that differ at all are two different columns.
    if frac.to_bits() != expect.frac.to_bits() {
        return Err(mismatch(
            "rank fraction",
            format!("{frac}"),
            format!("{}", expect.frac),
        ));
    }
    if wo_digest != expect.wo_digest {
        return Err(mismatch(
            "output-projection digest",
            format!("{wo_digest:#018x}"),
            format!("{:#018x}", expect.wo_digest),
        ));
    }

    let want = n_layers * d * rank;
    let payload = &bytes[HEADER_BYTES..];
    if payload.len() != want * 4 {
        return Err(BasisFileError::Payload {
            path: path.to_path_buf(),
            got: payload.len() / 4,
            want,
        });
    }
    let per = d * rank;
    let layers: Vec<Vec<f32>> = (0..n_layers)
        .map(|l| {
            payload[l * per * 4..(l + 1) * per * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        })
        .collect();
    let basis =
        OutputBasis::from_layers(layers, d, rank, Some(frac)).map_err(BasisFileError::Basis)?;
    Ok((basis, residual_max))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis(n_layers: usize, d: usize, rank: usize, frac: f64) -> OutputBasis {
        let layers = (0..n_layers)
            .map(|l| {
                (0..d * rank)
                    .map(|i| (l * 7 + i) as f32 * 0.125 - 3.0)
                    .collect()
            })
            .collect();
        OutputBasis::from_layers(layers, d, rank, Some(frac)).unwrap()
    }

    fn expect(b: &OutputBasis, n_layers: usize, digest: u64) -> Expect {
        Expect {
            n_layers,
            d: b.d(),
            rank: b.rank(),
            frac: b.frac().unwrap(),
            wo_digest: digest,
        }
    }

    #[test]
    fn a_written_basis_reads_back_bit_for_bit() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("wo.basis");
        let b = basis(4, 6, 2, 1.0 / 256.0);
        write(&p, &b, 3.25e-8, 0xdead_beef).unwrap();
        let (got, residual) = read(&p, &expect(&b, 4, 0xdead_beef)).unwrap();
        assert_eq!(residual, 3.25e-8);
        assert_eq!(got.rank(), b.rank());
        assert_eq!(got.d(), b.d());
        assert_eq!(got.frac(), b.frac());
        assert_eq!(got.n_layers(), 4);
        for l in 0..4 {
            assert_eq!(
                got.layer(l),
                b.layer(l),
                "layer {l} did not survive the round trip"
            );
        }
        // The key the scores publish under has to survive too, or the file silently renames them.
        assert_eq!(
            got.metric_key(16, crate::aperturb::Readout::default()),
            b.metric_key(16, crate::aperturb::Readout::default())
        );
    }

    #[test]
    fn the_header_is_exactly_as_long_as_it_claims() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("wo.basis");
        let b = basis(3, 5, 2, 1.0 / 128.0);
        write(&p, &b, 0.0, 1).unwrap();
        let len = std::fs::metadata(&p).unwrap().len() as usize;
        assert_eq!(len, HEADER_BYTES + 3 * 5 * 2 * 4);
    }

    #[test]
    fn every_header_field_that_disagrees_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("wo.basis");
        let b = basis(4, 6, 2, 1.0 / 256.0);
        write(&p, &b, 1e-8, 0xabc).unwrap();
        let good = expect(&b, 4, 0xabc);

        let cases: Vec<(&str, Expect)> = vec![
            (
                "layer count",
                Expect {
                    n_layers: 5,
                    ..good
                },
            ),
            ("projection width", Expect { d: 7, ..good }),
            ("rank", Expect { rank: 3, ..good }),
            (
                "rank fraction",
                Expect {
                    frac: 1.0 / 128.0,
                    ..good
                },
            ),
            (
                "output-projection digest",
                Expect {
                    wo_digest: 0xabd,
                    ..good
                },
            ),
        ];
        for (field, e) in cases {
            match read(&p, &e).map(|_| ()) {
                Err(BasisFileError::Mismatch { field: f, .. }) => assert_eq!(f, field),
                other => panic!("{field} was accepted: {other:?}"),
            }
        }
        // And the unmodified expectation still passes, so the cases above are testing the check and
        // not a file that never loaded.
        assert!(read(&p, &good).is_ok());
    }

    #[test]
    fn a_rank_fraction_that_differs_in_the_last_bit_is_a_different_column() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("wo.basis");
        let frac = 1.0 / 256.0;
        let b = basis(2, 4, 1, frac);
        write(&p, &b, 0.0, 9).unwrap();
        let nudged = f64::from_bits(frac.to_bits() + 1);
        assert_ne!(nudged, frac);
        assert!(matches!(
            read(
                &p,
                &Expect {
                    frac: nudged,
                    ..expect(&b, 2, 9)
                }
            ),
            Err(BasisFileError::Mismatch { .. })
        ));
    }

    #[test]
    fn a_truncated_or_padded_payload_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("wo.basis");
        let b = basis(3, 4, 2, 1.0 / 256.0);
        write(&p, &b, 0.0, 5).unwrap();
        let e = expect(&b, 3, 5);
        let full = std::fs::read(&p).unwrap();

        std::fs::write(&p, &full[..full.len() - 4]).unwrap();
        assert!(matches!(read(&p, &e), Err(BasisFileError::Payload { .. })));

        let mut padded = full.clone();
        padded.extend_from_slice(&[0u8; 4]);
        std::fs::write(&p, &padded).unwrap();
        assert!(matches!(read(&p, &e), Err(BasisFileError::Payload { .. })));

        // A header-only file is short of the payload, not of the header, so it must not be mistaken
        // for a foreign file.
        std::fs::write(&p, &full[..HEADER_BYTES]).unwrap();
        assert!(matches!(read(&p, &e), Err(BasisFileError::Payload { .. })));
    }

    #[test]
    fn a_foreign_or_future_file_is_refused_before_its_numbers_are_read() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("wo.basis");
        let b = basis(2, 4, 1, 1.0 / 256.0);
        write(&p, &b, 0.0, 1).unwrap();
        let e = expect(&b, 2, 1);
        let full = std::fs::read(&p).unwrap();

        let mut foreign = full.clone();
        foreign[..8].copy_from_slice(b"ARGUSAPT");
        std::fs::write(&p, &foreign).unwrap();
        assert!(matches!(read(&p, &e), Err(BasisFileError::Magic { .. })));

        std::fs::write(&p, [0u8; 4]).unwrap();
        assert!(matches!(read(&p, &e), Err(BasisFileError::Magic { .. })));

        let mut future = full.clone();
        future[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        std::fs::write(&p, &future).unwrap();
        assert!(matches!(read(&p, &e), Err(BasisFileError::Version { .. })));
    }

    #[test]
    fn a_missing_file_is_an_error_and_not_an_empty_basis() {
        let dir = tempfile::tempdir().unwrap();
        let b = basis(2, 4, 1, 1.0 / 256.0);
        let e = expect(&b, 2, 1);
        assert!(matches!(
            read(&dir.path().join("absent.basis"), &e),
            Err(BasisFileError::Io { .. })
        ));
    }

    #[test]
    fn the_untruncated_arm_is_not_storable() {
        let dir = tempfile::tempdir().unwrap();
        let wo = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let b = OutputBasis::untruncated(&wo, 2, 2).unwrap();
        assert!(matches!(
            write(&dir.path().join("x.basis"), &b, 0.0, 0),
            Err(BasisFileError::Untruncated)
        ));
    }

    #[test]
    fn the_digest_separates_checkpoints_that_share_a_shape() {
        let a = vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let mut b = a.clone();
        assert_eq!(digest(&a), digest(&b));
        // One weight moved by a single ulp — the shape fields cannot see this, and it is exactly
        // the difference between a base checkpoint and its instruct sibling.
        b[1][3] = f32::from_bits(b[1][3].to_bits() + 1);
        assert_ne!(digest(&a), digest(&b));
        // Nor can it see the layers swapped.
        let swapped = vec![a[1].clone(), a[0].clone()];
        assert_ne!(digest(&a), digest(&swapped));
        // Signed zero is a different bit pattern and is treated as one; the check is on bytes.
        let neg = vec![vec![-0.0f32, 2.0, 3.0, 4.0], a[1].clone()];
        let pos = vec![vec![0.0f32, 2.0, 3.0, 4.0], a[1].clone()];
        assert_ne!(digest(&neg), digest(&pos));
    }

    #[test]
    fn the_digest_is_a_fixed_function_and_not_just_a_stable_one() {
        // Pinned so a future change to the mixing is a test failure and not a silent invalidation
        // of every file already shipped with an application.
        assert_eq!(digest(&[]), 0xaf63_bd4c_8601_b7df);
        assert_eq!(digest(&[vec![1.0f32]]), 0x2d2e_305e_a41b_3a8d);
    }
}
