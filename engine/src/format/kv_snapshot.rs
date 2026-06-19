//! `SnapshotRestore` capability trait — KV prefix cache snapshot/restore opt-in.
//!
//! 설계 SSOT: `docs/adr/0012-session-prefix-cache-snapshot.md` D1.
//! 명세: `spec/30-engine.md` §3.7 (ENG-080). 불변식: `spec/41-invariants.md` §3.29 (INV-189~191).
//!
//! `KVCacheFormat` base trait(6-method)을 변경하지 않는다(ISP — snapshot-aware format만 비용
//! 지불). `StandardFormat`(F32/F16/Q4_0)이 구현하며, capability 미구현 format(quant-window/opaque)은
//! no-cache 폴백(정확성 안전, 단지 가속 없음).
//!
//! **as_any() 추가 금지**: downcast 경로 대신 상위 API(`save_prefix`/`try_restore_prefix`)가
//! `&dyn SnapshotRestore`로 직접 받도록 설계했다(arch/30-engine.md §19).

use anyhow::Result;

use crate::backend::Backend;

/// KV format이 prefix snapshot/restore 를 지원하는 경우 구현하는 capability trait.
///
/// # Contract
///
/// - `snapshot_prefix`: `current_pos == token_count` (INV-189), eviction 미발생 전제.
///   device 버퍼는 `backend.read_buffer()` 경유 (INV-191, ARM UMA stale 방지).
///   반환 bytes = capacity 패딩 제거 packed-form.
///
/// - `restore_prefix`: `current_pos == 0` (빈 캐시) 전제. 복원 후 `current_pos ==
///   token_count`, byte-identical to fresh prefill (INV-191).
///   device 버퍼는 `backend.write_buffer()` 경유.
///
/// - `snapshot_format_id`: 헤더 무효화 케이스 ② (ENG-083) — dtype 고정값.
pub trait SnapshotRestore: Send + Sync {
    /// `[0..token_count)` K+V를 capacity 패딩 제거 packed bytes로 직렬화한다.
    ///
    /// pre: `current_pos == token_count`, eviction 미발생 (INV-189).
    /// device 버퍼는 `backend.read_buffer()` 경유 (INV-191 — as_ptr 금지).
    ///
    /// 반환 layout (HeadMajor 전제):
    /// - K: `kv_heads × token_count × head_dim` packed (capacity head_stride 패딩 제거)
    /// - V: 동일
    /// - 총 bytes = `2 × kv_heads × token_count × head_dim × elem_size`
    ///   (Q4_0: elem 단위 아닌 block 단위 계산)
    fn snapshot_prefix(&self, token_count: usize, backend: &dyn Backend) -> Result<Vec<u8>>;

    /// `bytes`로부터 KV를 복원한다.
    ///
    /// pre: `current_pos == 0` (빈 캐시), `bytes` = 동일 format packed-form.
    /// post: `current_pos == token_count`, KV byte-identical to fresh prefill (INV-191).
    /// device 버퍼는 `backend.write_buffer()` 경유 (INV-191 — as_ptr 금지).
    fn restore_prefix(&self, bytes: &[u8], token_count: usize, backend: &dyn Backend)
    -> Result<()>;

    /// 직렬화 형식 식별자 (헤더 무효화 케이스 — ENG-083).
    ///
    /// F32=1, F16=2, Q4_0=3. opaque/KIVI는 미구현이므로 고려 불요.
    fn snapshot_format_id(&self) -> u32;
}
