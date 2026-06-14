// ── PreloadAccess trait (L2 격상) ────────────────────────────────────────────
//!
//! KV cache preload thread-pool 의 공유 인터페이스 (§13.8-O INV-LAYER-003).
//!
//! 위치 근거: L2 공유 어휘 — inference(`models/transformer.rs` struct 필드
//! `preload_pool: OnceLock<Box<dyn PreloadAccess>>`) 와 pressure(구현체
//! `kv::offload::preload_pool::PreloadPool`) 양 도메인이 공유하는 인터페이스
//! 정의. `LayerBoundaryHook`/`KVCacheOps` L2 격상(§13.8-G)과 동일 동기 —
//! trait 정의를 top-level 형제 `*.rs` 로 분리해 inference→pressure cross-L3
//! 어휘 결합을 끊는다. 구현체(`PreloadPool`)는 pressure 도메인에 잔존한다.

use std::sync::mpsc;
use std::time::Duration;

use anyhow::Result;

/// Result of a completed preload task.
pub struct PreloadResult {
    pub result: Result<()>,
    pub duration: Duration,
}

/// Trait abstracting a persistent preload thread pool for KV cache operations.
///
/// Implementors: `kv::offload::preload_pool::PreloadPool`.
///
/// Separating this trait from the concrete struct allows inference-layer code
/// (`models/transformer.rs`) to hold `Box<dyn PreloadAccess>` without importing
/// the concrete `PreloadPool` type, breaking the cross-L3 dependency
/// (§13.8-O INV-LAYER-003).
pub trait PreloadAccess: Send + Sync {
    /// Number of worker threads.
    fn size(&self) -> usize;

    /// Submit a type-erased preload task.
    ///
    /// # Safety
    /// - `cache_ptr` must point to a valid, properly aligned object.
    /// - The object must remain valid until the returned receiver is collected.
    /// - No concurrent task may access the same `cache_ptr`.
    /// - `preload_fn` must correctly cast `*mut ()` back to the original type.
    unsafe fn submit_raw(
        &self,
        cache_ptr: *mut (),
        preload_fn: unsafe fn(*mut ()) -> anyhow::Result<()>,
    ) -> mpsc::Receiver<PreloadResult>;
}
