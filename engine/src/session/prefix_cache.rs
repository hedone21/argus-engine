//! Session prefix KV cache persistence (Tier 1).
//!
//! 설계 SSOT: `docs/adr/0012-session-prefix-cache-snapshot.md`.
//! 명세: `spec/30-engine.md` §3.7 (ENG-080~085), `spec/33-engine-data.md` §3.25 (ENG-DAT-110).
//! 불변식: `spec/41-invariants.md` §3.29 (INV-189~191).
//!
//! ## 파일 형식 (ENG-DAT-110)
//!
//! ```text
//! [u8; 8]     magic = "ARGUSKV1"
//! u32         version = 2
//! u32         header_len
//! [u8; 32]    model_hash
//! u32         format_id
//! [u8; 32]    tokenizer_hash
//! u32         kv_heads
//! u32         head_dim
//! u32         n_layers
//! u32         token_count
//! [u32; N]    token_ids  (N = token_count)
//! u32         logits_len
//! [f32; L]    last_logits  (L = logits_len)
//! --- payload (layer-major) ---
//! layer 0 K bytes, layer 0 V bytes, layer 1 K bytes, ...
//! ```
//!
//! version 1 파일(logits 섹션 없음)은 무효화(Ok(None)) 처리.
//!
//! ## 무효화 우선순위 (ENG-083)
//!
//! magic → version → geometry → model_hash → format_id → tokenizer_hash → token_ids

use anyhow::Result;

use crate::backend::Backend;
use crate::format::SnapshotRestore;

const MAGIC: &[u8; 8] = b"ARGUSKV1";
/// version 2: logits 섹션 추가 (token_ids 뒤, payload 앞).
/// version 1 파일은 무효화(Ok(None)) 처리 — 재생성 필요.
const VERSION: u32 = 2;

/// 헤더 고정 크기 계산 (token_ids 앞까지).
/// magic(8) + version(4) + header_len(4) + model_hash(32) + format_id(4) +
/// tokenizer_hash(32) + kv_heads(4) + head_dim(4) + n_layers(4) + token_count(4) = 100
const FIXED_HEADER_BYTES: usize = 100;

// ── Serialization helpers ────────────────────────────────────────────────────

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(src: &[u8], off: &mut usize) -> Option<u32> {
    if *off + 4 > src.len() {
        return None;
    }
    let v = u32::from_le_bytes(src[*off..*off + 4].try_into().ok()?);
    *off += 4;
    Some(v)
}

fn read_bytes<const N: usize>(src: &[u8], off: &mut usize) -> Option<[u8; N]> {
    if *off + N > src.len() {
        return None;
    }
    let mut arr = [0u8; N];
    arr.copy_from_slice(&src[*off..*off + N]);
    *off += N;
    Some(arr)
}

// ── Public API ───────────────────────────────────────────────────────────────

/// `try_restore_prefix` 성공 시 반환되는 복원 결과.
///
/// - `token_count`: 복원된 토큰 수.
///   - `== prompt.len()`: full restore — prefill 완전 skip 가능.
///   - `< prompt.len()`: partial restore — `prompt[token_count..]` 잔여 prefill 필요.
/// - `last_logits`: full restore 시 prefill이 산출했던 마지막 토큰 logits (f32×vocab).
///   partial restore 경로에서는 빈 Vec이므로 호출자가 무시해야 한다.
#[derive(Debug, PartialEq)]
pub struct RestoredPrefix {
    pub token_count: usize,
    pub last_logits: Vec<f32>,
}

/// prefill 직후 KV prefix를 파일에 저장한다 (INV-189: eviction 전, current_pos==token_count).
///
/// - `path`: 저장 경로 (`<path>.tmp` → `rename` atomic write로 손상 방지)
/// - `model_hash`: `auf::source_hash::compute_source_hash(model_path)` 결과
/// - `tokenizer_hash`: tokenizer 파일 hash
/// - `token_ids`: prompt 토큰열 전체 (= snapshot 대상 prefix)
/// - `last_logits`: prefill이 산출한 마지막 토큰 logits (f32×vocab). full restore 시 재사용.
/// - `formats`: layer별 `&dyn SnapshotRestore` slice (n_layers 길이)
/// - `backend`: read_buffer 경유 device coherent readback (INV-191)
///
/// save 실패(권한, 디스크 풀 등)는 에러를 반환하며, 호출자는 경고 후 무시해도 된다.
#[allow(clippy::too_many_arguments)]
pub fn save_prefix(
    path: &std::path::Path,
    model_hash: &[u8; 32],
    tokenizer_hash: &[u8; 32],
    token_ids: &[u32],
    last_logits: &[f32],
    format_id: u32,
    formats: &[&dyn SnapshotRestore],
    kv_heads: u32,
    head_dim: u32,
    backend: &dyn Backend,
) -> Result<()> {
    let token_count = token_ids.len() as u32;
    let n_layers = formats.len() as u32;
    let logits_len = last_logits.len() as u32;

    // header_len = fixed fields + token_ids
    let header_len = (FIXED_HEADER_BYTES - 8 - 4 - 4) as u32 // magic/version/header_len 제외
        + token_count * 4;

    // ── 헤더 직렬화 ──
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    write_u32(&mut buf, VERSION);
    write_u32(&mut buf, header_len);
    buf.extend_from_slice(model_hash);
    write_u32(&mut buf, format_id);
    buf.extend_from_slice(tokenizer_hash);
    write_u32(&mut buf, kv_heads);
    write_u32(&mut buf, head_dim);
    write_u32(&mut buf, n_layers);
    write_u32(&mut buf, token_count);
    for &id in token_ids {
        write_u32(&mut buf, id);
    }

    // ── logits 섹션 (token_ids 뒤, payload 앞) ──
    write_u32(&mut buf, logits_len);
    for &v in last_logits {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    // ── payload (layer-major) ──
    for fmt in formats {
        let layer_bytes = fmt.snapshot_prefix(token_ids.len(), backend)?;
        buf.extend_from_slice(&layer_bytes);
    }

    // ── atomic write ──
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, &buf)?;
    std::fs::rename(&tmp_path, path)?;

    Ok(())
}

/// 파일에서 KV prefix를 복원 시도한다.
///
/// 반환:
/// - `Ok(Some(RestoredPrefix { token_count, last_logits }))`: 복원 성공.
///   - `token_count == prompt.len()`: full restore — prefill 완전 skip. `last_logits`에 저장된
///     logits가 채워져 있어 호출자가 re-forward 없이 decode를 시작할 수 있다.
///   - `token_count < prompt.len()`: partial restore — `prompt[token_count..]` 잔여 prefill 필요.
///     `last_logits`는 빈 Vec (partial restore에서는 저장 logits를 사용하지 않음).
/// - `Ok(None)`: 파일 없음 또는 무효화 → fresh prefill. **에러 아님** (INV-190).
/// - `Err(...)`: I/O 오류 또는 restore 실패 (캐시 오염 가능 — 호출자가 처리).
///
/// **무효화 우선순위** (ENG-083, 빠른 reject 순):
/// magic → version → geometry → model_hash → format_id → tokenizer_hash → token_ids
///
/// `logits_len != vocab_size` (geometry 불일치): full restore 경로에서 vocab 크기로
/// 검증하므로, 호출자는 복원 후 `last_logits.len()` 이 기대 vocab과 일치하는지 확인하거나
/// `vocab_size` 인자로 검증을 위임할 수 있다. 현행 구현은 logits_len > 0 이면 그대로 반환하고
/// 호출자(`standard_happy.rs`)가 vocab 크기를 별도 검증한다.
#[allow(clippy::too_many_arguments)]
pub fn try_restore_prefix(
    path: &std::path::Path,
    model_hash: &[u8; 32],
    tokenizer_hash: &[u8; 32],
    current_format_id: u32,
    prompt: &[u32],
    formats: &[&dyn SnapshotRestore],
    kv_heads: u32,
    head_dim: u32,
    backend: &dyn Backend,
) -> Result<Option<RestoredPrefix>> {
    // 파일 없음 = miss (에러 아님)
    let raw = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };

    let mut off = 0usize;

    // ① magic 검증
    let magic = match read_bytes::<8>(&raw, &mut off) {
        Some(m) => m,
        None => return Ok(None),
    };
    if &magic != MAGIC {
        return Ok(None);
    }

    // ② version 검증
    let version = match read_u32(&raw, &mut off) {
        Some(v) => v,
        None => return Ok(None),
    };
    if version != VERSION {
        return Ok(None);
    }

    // header_len (payload offset 계산에 사용)
    let _header_len = match read_u32(&raw, &mut off) {
        Some(v) => v,
        None => return Ok(None),
    };

    // model_hash
    let saved_model_hash = match read_bytes::<32>(&raw, &mut off) {
        Some(h) => h,
        None => return Ok(None),
    };
    // format_id (model_hash 먼저 읽은 후 검증은 아래)
    let saved_format_id = match read_u32(&raw, &mut off) {
        Some(v) => v,
        None => return Ok(None),
    };
    // tokenizer_hash
    let saved_tokenizer_hash = match read_bytes::<32>(&raw, &mut off) {
        Some(h) => h,
        None => return Ok(None),
    };
    // geometry
    let saved_kv_heads = match read_u32(&raw, &mut off) {
        Some(v) => v,
        None => return Ok(None),
    };
    let saved_head_dim = match read_u32(&raw, &mut off) {
        Some(v) => v,
        None => return Ok(None),
    };
    let saved_n_layers = match read_u32(&raw, &mut off) {
        Some(v) => v,
        None => return Ok(None),
    };
    let saved_token_count = match read_u32(&raw, &mut off) {
        Some(v) => v,
        None => return Ok(None),
    };

    // ③ geometry 검증 (payload 길이 mismatch 방어)
    if saved_kv_heads != kv_heads
        || saved_head_dim != head_dim
        || saved_n_layers != formats.len() as u32
    {
        return Ok(None);
    }

    // ④ model_hash 검증
    if &saved_model_hash != model_hash {
        return Ok(None);
    }

    // ⑤ format_id 검증
    if saved_format_id != current_format_id {
        return Ok(None);
    }

    // ⑥ tokenizer_hash 검증
    if &saved_tokenizer_hash != tokenizer_hash {
        return Ok(None);
    }

    // token_ids 읽기
    let token_count = saved_token_count as usize;
    if off + token_count * 4 > raw.len() {
        return Ok(None);
    }
    let mut saved_token_ids = Vec::with_capacity(token_count);
    for _ in 0..token_count {
        match read_u32(&raw, &mut off) {
            Some(id) => saved_token_ids.push(id),
            None => return Ok(None),
        }
    }

    // ⑦ token_ids 접두 일치 검사 (ENG-084)
    // token_count > prompt.len() 이면 miss
    if token_count > prompt.len() {
        return Ok(None);
    }
    if saved_token_ids != prompt[..token_count] {
        return Ok(None);
    }

    // ── logits 섹션 읽기 (version 2 신규) ──
    let logits_len = match read_u32(&raw, &mut off) {
        Some(v) => v as usize,
        None => return Ok(None),
    };
    if off + logits_len * 4 > raw.len() {
        return Ok(None);
    }
    let mut last_logits = Vec::with_capacity(logits_len);
    for _ in 0..logits_len {
        if off + 4 > raw.len() {
            return Ok(None);
        }
        let bytes: [u8; 4] = raw[off..off + 4].try_into().unwrap();
        last_logits.push(f32::from_le_bytes(bytes));
        off += 4;
    }

    // ── 복원 수행 ──
    let payload = &raw[off..];

    // partial restore: logits는 반환하지 않는다 (잔여 prefill이 logits를 산출).
    if token_count < prompt.len() {
        if formats.is_empty() {
            return Ok(Some(RestoredPrefix {
                token_count,
                last_logits: Vec::new(),
            }));
        }
        let per_layer = payload.len() / formats.len();
        if per_layer * formats.len() != payload.len() {
            return Ok(None);
        }
        for (i, fmt) in formats.iter().enumerate() {
            let layer_bytes = &payload[i * per_layer..(i + 1) * per_layer];
            fmt.restore_prefix(layer_bytes, token_count, backend)?;
        }
        return Ok(Some(RestoredPrefix {
            token_count,
            last_logits: Vec::new(),
        }));
    }

    // full restore (token_count == prompt.len())
    // payload를 layer별로 분배 (layer-major: K bytes || V bytes per layer)
    // 각 layer payload 크기는 formats[i].snapshot_prefix로 알 수 없으므로
    // 총 payload를 n_layers로 균등 분배한다 (all layers same geometry).
    if formats.is_empty() {
        return Ok(Some(RestoredPrefix {
            token_count,
            last_logits,
        }));
    }
    let per_layer = payload.len() / formats.len();
    if per_layer * formats.len() != payload.len() {
        // 불일치 = 손상
        return Ok(None);
    }

    for (i, fmt) in formats.iter().enumerate() {
        let layer_bytes = &payload[i * per_layer..(i + 1) * per_layer];
        fmt.restore_prefix(layer_bytes, token_count, backend)?;
    }

    Ok(Some(RestoredPrefix {
        token_count,
        last_logits,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::format::{KVCacheFormat, SnapshotRestore};
    use crate::kv::kv_cache::KVCache;
    use crate::kv::standard_format::StandardFormat;
    use crate::kv_cache_ops::KVLayout;
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn make_f32_headmajor_cache(capacity: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        let total = kv_heads * capacity * head_dim;
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        let backend = Arc::new(CpuBackend::new());
        let buf_k = mem.alloc_kv(total * 4, DType::F32).unwrap();
        let buf_v = mem.alloc_kv(total * 4, DType::F32).unwrap();
        let k = Tensor::new(
            Shape::new(vec![1, kv_heads, capacity, head_dim]),
            buf_k,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, kv_heads, capacity, head_dim]),
            buf_v,
            backend.clone(),
        );
        KVCache::new_dynamic(k, v, capacity, capacity * 4, kv_heads, head_dim, mem)
            .with_layout(KVLayout::HeadMajor)
    }

    fn fill_cache_f32(fmt: &StandardFormat, n_tokens: usize, kv_heads: usize, head_dim: usize) {
        let backend = CpuBackend::new();
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        let be = Arc::new(CpuBackend::new());
        for pos in 0..n_tokens {
            let total = kv_heads * head_dim;
            let mut k_data = vec![0.0f32; total];
            let mut v_data = vec![0.0f32; total];
            for h in 0..kv_heads {
                for d in 0..head_dim {
                    k_data[h * head_dim + d] = (pos * 100 + h * 10 + d) as f32 * 0.01;
                    v_data[h * head_dim + d] = (pos * 100 + h * 10 + d) as f32 * 0.02;
                }
            }
            let buf_k = mem.alloc_kv(total * 4, DType::F32).unwrap();
            let buf_v = mem.alloc_kv(total * 4, DType::F32).unwrap();
            let mut tk = Tensor::new(
                Shape::new(vec![1, 1, kv_heads, head_dim]),
                buf_k,
                be.clone(),
            );
            let mut tv = Tensor::new(
                Shape::new(vec![1, 1, kv_heads, head_dim]),
                buf_v,
                be.clone(),
            );
            tk.as_mut_slice::<f32>().copy_from_slice(&k_data);
            tv.as_mut_slice::<f32>().copy_from_slice(&v_data);
            fmt.write_kv(&tk, &tv, &backend).unwrap();
        }
    }

    #[test]
    fn roundtrip_save_restore_f32() {
        let kv_heads = 2usize;
        let head_dim = 4usize;
        let n_tokens = 3usize;

        // save 측: capacity=8
        let cache_a = make_f32_headmajor_cache(8, kv_heads, head_dim);
        let fmt_a = StandardFormat::new(0, cache_a);
        fill_cache_f32(&fmt_a, n_tokens, kv_heads, head_dim);

        let backend = CpuBackend::new();
        let token_ids: Vec<u32> = (0..n_tokens as u32).collect();
        let model_hash = [1u8; 32];
        let tok_hash = [2u8; 32];
        let format_id = fmt_a.snapshot_format_id();

        // save 전 snapshot bytes 추출 (byte-identical 검증용)
        let bytes_a = fmt_a.snapshot_prefix(n_tokens, &backend).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kv.cache");

        let dummy_logits: Vec<f32> = (0..8u32).map(|i| i as f32 * 0.1).collect();

        save_prefix(
            &path,
            &model_hash,
            &tok_hash,
            &token_ids,
            &dummy_logits,
            format_id,
            &[&fmt_a as &dyn SnapshotRestore],
            kv_heads as u32,
            head_dim as u32,
            &backend,
        )
        .unwrap();

        // restore 측: capacity=4 (다른 capacity, ensure_capacity로 자동 확장)
        let cache_b = make_f32_headmajor_cache(4, kv_heads, head_dim);
        let fmt_b = StandardFormat::new(0, cache_b);

        let prompt: Vec<u32> = (0..n_tokens as u32).collect();
        let result = try_restore_prefix(
            &path,
            &model_hash,
            &tok_hash,
            format_id,
            &prompt,
            &[&fmt_b as &dyn SnapshotRestore],
            kv_heads as u32,
            head_dim as u32,
            &backend,
        )
        .unwrap();

        let restored = result.expect("roundtrip must succeed");
        assert_eq!(restored.token_count, n_tokens);
        // logits round-trip: saved dummy_logits와 동일해야 한다
        assert_eq!(
            restored.last_logits, dummy_logits,
            "logits round-trip must be bit-identical"
        );

        // byte-identical 검증: packed snapshot bytes가 동일해야 한다
        let bytes_b = fmt_b.snapshot_prefix(n_tokens, &backend).unwrap();
        assert_eq!(bytes_a, bytes_b, "roundtrip must be byte-identical");
    }

    #[test]
    fn miss_on_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.cache");
        let backend = CpuBackend::new();
        let cache = make_f32_headmajor_cache(4, 2, 4);
        let fmt = StandardFormat::new(0, cache);

        let result = try_restore_prefix(
            &path,
            &[0u8; 32],
            &[0u8; 32],
            1,
            &[0u32, 1, 2],
            &[&fmt as &dyn SnapshotRestore],
            2,
            4,
            &backend,
        )
        .unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn miss_on_model_hash_mismatch() {
        let kv_heads = 2u32;
        let head_dim = 4u32;
        let n_tokens = 2usize;

        let cache_a = make_f32_headmajor_cache(8, kv_heads as usize, head_dim as usize);
        let fmt_a = StandardFormat::new(0, cache_a);
        fill_cache_f32(&fmt_a, n_tokens, kv_heads as usize, head_dim as usize);

        let backend = CpuBackend::new();
        let token_ids: Vec<u32> = (0..n_tokens as u32).collect();
        let model_hash_a = [1u8; 32];
        let tok_hash = [2u8; 32];
        let format_id = fmt_a.snapshot_format_id();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kv.cache");
        save_prefix(
            &path,
            &model_hash_a,
            &tok_hash,
            &token_ids,
            &[],
            format_id,
            &[&fmt_a as &dyn SnapshotRestore],
            kv_heads,
            head_dim,
            &backend,
        )
        .unwrap();

        // 다른 model_hash로 restore 시도
        let cache_b = make_f32_headmajor_cache(8, kv_heads as usize, head_dim as usize);
        let fmt_b = StandardFormat::new(0, cache_b);
        let model_hash_b = [99u8; 32];

        let result = try_restore_prefix(
            &path,
            &model_hash_b,
            &tok_hash,
            format_id,
            &token_ids,
            &[&fmt_b as &dyn SnapshotRestore],
            kv_heads,
            head_dim,
            &backend,
        )
        .unwrap();
        assert_eq!(result, None, "model_hash mismatch must be cache miss");
    }
}
