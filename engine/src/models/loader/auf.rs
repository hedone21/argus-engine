//! AUF (Argus Unified Format) primary loader.
//!
//! `--model-path foo.auf`를 직접 받아 `TensorSource`로 로드하는 진입점.
//! 같은 AUF 파일이 multi-dtype 또는 multi-variant capability를 가지면
//! self-secondary 자동 활성 후보가 된다 (W-AUF-2에서 본격화).
//!
//! ## 구성
//! - [`source`] — `AufSource` (impl TensorSource)
//! - [`variant_select`] — backend → BackendTag 매핑 + dtype 후보 선택
//!
//! ## Primary 포맷 식별
//!
//! [`detect_primary_format`]은 확장자 + AUF magic byte로 결정한다.

pub mod secondary;
pub mod source;
pub mod variant_select;

pub use secondary::{
    auf_dtype_to_engine, build_auf_secondary_from_view, check_auf_metadata,
    from_auf_self_secondary, is_auf_path, open_secondary_auf,
};
pub use source::AufSource;
pub use variant_select::{AufDtypeChoice, AufVariantChoice};

use anyhow::{Result, anyhow};
use std::path::Path;

/// `--model-path` 파일의 포맷 식별 결과.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryFormat {
    /// AUF (Argus Unified Format) — `.auf` 확장자 + magic 검증 통과.
    Auf,
    /// GGUF — `.gguf` 확장자.
    Gguf,
    /// Safetensors — 위 둘 모두 아님 (디렉토리 또는 다른 확장자).
    Safetensors,
}

/// `--model-path` 경로의 포맷을 식별한다.
///
/// 결정 순서:
/// 1. 확장자가 `.auf` → magic byte (`crate::auf::header::AUF_MAGIC`)로 추가 검증.
///    실패 시 명시적 에러.
/// 2. 확장자가 `.gguf` → `PrimaryFormat::Gguf`.
/// 3. 그 외 → `PrimaryFormat::Safetensors` (디렉토리, `.safetensors` 등).
pub fn detect_primary_format(path: &Path) -> Result<PrimaryFormat> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());

    match ext.as_deref() {
        Some("auf") => {
            if has_auf_magic(path)? {
                Ok(PrimaryFormat::Auf)
            } else {
                Err(anyhow!(
                    "primary loader: '{}' has .auf extension but AUF magic header is missing",
                    path.display()
                ))
            }
        }
        Some("gguf") => Ok(PrimaryFormat::Gguf),
        _ => Ok(PrimaryFormat::Safetensors),
    }
}

/// AUF magic byte 검증 (첫 8 바이트 = `crate::auf::header::AUF_MAGIC`).
fn has_auf_magic(path: &Path) -> Result<bool> {
    use std::fs::File;
    use std::io::Read;

    let mut f = File::open(path)
        .map_err(|e| anyhow!("primary loader: cannot open '{}': {}", path.display(), e))?;
    let mut buf = [0u8; 8];
    match f.read_exact(&mut buf) {
        Ok(()) => Ok(&buf == crate::auf::header::AUF_MAGIC),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn non_auf_extension_skips_magic_check() {
        let p = PathBuf::from("/tmp/model.gguf");
        let fmt = detect_primary_format(&p).unwrap();
        assert_eq!(fmt, PrimaryFormat::Gguf);
    }

    #[test]
    fn unknown_extension_returns_safetensors() {
        let p = PathBuf::from("/tmp/model_dir");
        let fmt = detect_primary_format(&p).unwrap();
        assert_eq!(fmt, PrimaryFormat::Safetensors);
    }
}
