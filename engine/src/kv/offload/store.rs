//! OffloadStore trait: abstraction for KV cache layer storage backends.

use anyhow::Result;

/// Backend-agnostic KV cache storage for a single layer — offload 의 **단일 내부
/// byte-residency seam**.
///
/// offload 의 유일한 변동축은 *KV 바이트가 어디에 사는가*(RAM / 디스크 / …)이고, 그 축은
/// 전적으로 이 trait 뒤에 격리돼 있다: `&[u8]`(+ token 카운트)만 경계를 넘고
/// `Backend`/`Tensor`/`Memory` 는 결코 넘지 않는 POD-shaped 표면이라 어느 format 이든
/// residency 와 직교로 조합된다. 새 저장 매체(mmap/tmpfs/압축-RAM/…)는 새 impl + 생성자
/// arm 하나로 추가되며 ABI/format 변경이 없다 (`alloc_offload_kv_caches` 가 생성자 dispatch).
/// out-of-tree 저장 backend 가 dlopen 으로 붙어야 할 때 비로소 `repr(C)` VTable 승격이
/// 정당해진다 — 지금은 소비자가 in-engine 뿐이라 in-engine trait 으로 둔다.
///
/// 구현: `RawStore`(in-memory, 무압축) · `DiskStore`(레이어별 파일). 각 store 인스턴스는
/// 한 transformer 레이어(K + V)의 데이터를 보유한다.
pub trait OffloadStore: Send {
    /// Write full KV data to storage (used during migration).
    fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()>;

    /// Load KV data from storage into pre-allocated buffers.
    /// Returns the number of tokens loaded.
    fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize>;

    /// Append a single token's K/V data (used during decode).
    fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()>;

    /// Current storage size in bytes.
    fn storage_size(&self) -> usize;

    /// Number of tokens currently stored.
    fn stored_tokens(&self) -> usize;

    /// Reset storage to empty state.
    fn clear(&mut self);
}
