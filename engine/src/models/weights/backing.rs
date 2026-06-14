//! `WeightSectionView` — secondary weight 백킹 추상화 (W-AUF-2.3, R8).
//!
//! `RpcmemSecondaryStore` (LISWAP-6, qnn_oppkg RpcMem zero-copy alias)가
//! GGUF/AUF 어느 backing이든 받을 수 있도록 weights section 슬라이스 제공자만
//! trait로 추상화한다. 본격적인 trait 분리 (layer_index/prefault/SOA 등)는
//! backlog (`SecondaryMmap` enum의 OCP 위배 해결)로 미룬다.
//!
//! ## 계약
//!
//! - `weights_bytes()`는 `SecondaryTensorInfo::offset`이 base로 삼는 슬라이스를
//!   반환한다. GGUF는 전체 mmap (offset = absolute), AUF는 weights section
//!   슬라이스 (offset = section-local) — 두 경우 모두 같은 인덱싱 식 `bytes[info.offset..]`
//!   로 통일된다.
//! - 구현체는 mmap을 alive로 유지하는 `Arc<...>` 핸들을 보유해야 한다.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::auf::AufView;
use crate::models::loader::gguf::GgufFile;

/// Trait — weights 바이트 슬라이스 + 진단 path 제공.
pub trait WeightSectionView: Send + Sync + std::fmt::Debug {
    /// `SecondaryTensorInfo::offset`이 base로 삼는 슬라이스.
    fn weights_bytes(&self) -> &[u8];
    /// 진단용 source path.
    fn source_path(&self) -> &Path;
}

/// GGUF mmap 백킹. `SecondaryTensorInfo::offset`은 전체 mmap에 대한 absolute offset.
pub struct GgufBacking {
    pub gguf: Arc<GgufFile>,
    pub source_path: PathBuf,
}

impl std::fmt::Debug for GgufBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufBacking")
            .field("source_path", &self.source_path)
            .finish()
    }
}

impl WeightSectionView for GgufBacking {
    fn weights_bytes(&self) -> &[u8] {
        self.gguf.mmap_data()
    }
    fn source_path(&self) -> &Path {
        &self.source_path
    }
}

/// AUF self-secondary 백킹. `SecondaryTensorInfo::offset`은 weights section-local.
pub struct AufBacking {
    pub view: Arc<AufView>,
    pub source_path: PathBuf,
}

impl std::fmt::Debug for AufBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AufBacking")
            .field("source_path", &self.source_path)
            .finish()
    }
}

impl WeightSectionView for AufBacking {
    fn weights_bytes(&self) -> &[u8] {
        // `weights_range`가 Some이라는 invariant는 `AufView::open(_, concrete_tag)` 단계에서 검증된다.
        self.view.weights_bytes().expect(
            "AufBacking: weights_bytes() must be Some (view opened with concrete BackendTag)",
        )
    }
    fn source_path(&self) -> &Path {
        &self.source_path
    }
}
