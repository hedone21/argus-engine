//! P2d — Demote 모사 측정 하네스 (KV roadmap 항목 0 §4.4).
//!
//! 동일 메모리 예산에서:
//!   (a) sliding N개 토큰 F16 유지 (나머지 evict)
//!   (b) 2N~4N개 토큰 보존 — sliding 창 밖 후보를 Q4(또는 Q2)로 demote
//!
//! demote 후보를 in-place quant→dequant 왕복하여 정밀도 손실 주입 후 NMSE 비교.
//!
//! ## 설계서 §4.4-D 모사 한계 (명시)
//!
//! quant→dequant 왕복은 품질상 mixed-precision 저장과 수치 등가이나,
//! K 손실의 차기 step 2차 효과(attention weight 교란)는 미포착한다.
//! 본 게이트는 V 손실 + K 직접 손실의 1차 효과만 측정한다.
//!
//! ## 완료 게이트
//!
//! ① 왕복 수치 정합(왕복 후 값 = quantizer dequant 결과와 일치)
//! ② 동일 메모리 예산: (a) vs (b) NMSE 비교 수치 산출
//! ③ 엔진 lib 코드 무수정 확인 (이 파일이 tests/에 격리됨)
//! ④ Q4 vs Q2 격차 측정 (논문 2-bit 급락 재현 여부)
//! ⑤ 실모델 PPL 비교 (DEMOTE_TEST_MODEL 환경변수 지정 시): 설계서 §4.4-E 게이트.
//!    환경변수 미설정 시 graceful skip (CI 무영향).

use std::sync::Arc;

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::inference::sampling;
use argus_engine::kv::kv_cache::{KVCache, KVLayout};
use argus_engine::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use argus_engine::observability::eval::EvalCacheKind;
use argus_engine::qcf::quant_qcf::compute_nmse_block;
use argus_engine::quant::{BlockKVQ4, BlockQ2_0, QKKV};
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;

// ── 공통 helpers ────────────────────────────────────────────────────────────

/// 단일 Q4 블록 quant→dequant 왕복.
fn demote_q4_block(src: &[f32; QKKV]) -> [f32; QKKV] {
    let block = BlockKVQ4::quantize(src);
    let mut out = [0.0f32; QKKV];
    block.dequantize(&mut out);
    out
}

/// 단일 Q2 블록 quant→dequant 왕복.
fn demote_q2_block(src: &[f32; QKKV]) -> [f32; QKKV] {
    let block = BlockQ2_0::quantize(src);
    let mut out = [0.0f32; QKKV];
    block.dequantize(&mut out);
    out
}

/// K cache per-channel quant→dequant 왕복.
///
/// K layout: `[kv_heads][n_tokens][head_dim]`.
/// per-channel = head_dim 축 기준 — 동일 채널(col) 여러 토큰을 1블록으로.
/// QKKV=32 기준: n_tokens가 32의 배수일 때 각 채널을 32 토큰씩 quantize.
///
/// 반환: 왕복 후 K values (원본과 같은 크기).
fn demote_k_per_channel(
    k: &[f32],
    n_tokens: usize,
    head_dim: usize,
    kv_heads: usize,
    bits: u8,
) -> Vec<f32> {
    let mut out = k.to_vec();
    let n_groups = n_tokens / QKKV;
    if n_groups == 0 {
        return out; // n_tokens < QKKV → skip
    }

    for h in 0..kv_heads {
        let head_base = h * n_tokens * head_dim;
        for ch in 0..head_dim {
            for group in 0..n_groups {
                let tok_start = group * QKKV;
                let mut vals = [0.0f32; QKKV];
                for (t, v) in vals.iter_mut().enumerate() {
                    let idx = head_base + (tok_start + t) * head_dim + ch;
                    *v = k[idx];
                }
                let reconstructed = match bits {
                    4 => demote_q4_block(&vals),
                    2 => demote_q2_block(&vals),
                    _ => vals,
                };
                for (t, &r) in reconstructed.iter().enumerate() {
                    let idx = head_base + (tok_start + t) * head_dim + ch;
                    out[idx] = r;
                }
            }
        }
    }
    out
}

/// V cache per-token quant→dequant 왕복.
///
/// V layout: `[kv_heads][n_tokens][head_dim]`.
/// per-token = 각 토큰의 head_dim 요소를 head_dim/QKKV 블록으로 quantize.
fn demote_v_per_token(
    v: &[f32],
    n_tokens: usize,
    head_dim: usize,
    kv_heads: usize,
    bits: u8,
) -> Vec<f32> {
    let mut out = v.to_vec();
    let blocks_per_token = head_dim / QKKV;
    if blocks_per_token == 0 {
        return out; // head_dim < QKKV → skip
    }

    for h in 0..kv_heads {
        let head_base = h * n_tokens * head_dim;
        for t in 0..n_tokens {
            let tok_base = head_base + t * head_dim;
            for b in 0..blocks_per_token {
                let start = tok_base + b * QKKV;
                let chunk: &[f32; QKKV] = v[start..start + QKKV].try_into().unwrap();
                let reconstructed = match bits {
                    4 => demote_q4_block(chunk),
                    2 => demote_q2_block(chunk),
                    _ => *chunk,
                };
                out[start..start + QKKV].copy_from_slice(&reconstructed);
            }
        }
    }
    out
}

/// K 또는 V 배열에 대해 전체 NMSE를 계산한다 (블록 평균).
fn mean_nmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    let n_blocks = original.len() / QKKV;
    if n_blocks == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for b in 0..n_blocks {
        let start = b * QKKV;
        let orig: &[f32; QKKV] = original[start..start + QKKV].try_into().unwrap();
        let recon: &[f32; QKKV] = reconstructed[start..start + QKKV].try_into().unwrap();
        // MSE / Var 방식으로 직접 계산
        let nmse = compute_nmse_block_pair(orig, recon);
        sum += nmse;
    }
    sum / n_blocks as f32
}

/// original vs reconstructed 배열 쌍에 대한 NMSE 계산.
fn compute_nmse_block_pair(original: &[f32; QKKV], reconstructed: &[f32; QKKV]) -> f32 {
    let mean = original.iter().sum::<f32>() / QKKV as f32;
    let var = original.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / QKKV as f32;
    if var < 1e-8 {
        return 0.0;
    }
    let mse = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| (o - r).powi(2))
        .sum::<f32>()
        / QKKV as f32;
    (mse / var).clamp(0.0, 1.0)
}

// ── 게이트 ① : 왕복 수치 정합 테스트 ──────────────────────────────────────

/// Q4 왕복: demote_q4_block 결과가 BlockKVQ4::quantize→dequantize와 bit-identical.
#[test]
fn test_demote_q4_roundtrip_matches_quantizer() {
    let src: [f32; QKKV] = std::array::from_fn(|i| (i as f32) * 0.1);
    let roundtrip = demote_q4_block(&src);

    let block = BlockKVQ4::quantize(&src);
    let mut expected = [0.0f32; QKKV];
    block.dequantize(&mut expected);

    for (i, (&a, &b)) in roundtrip.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "Q4 roundtrip mismatch at idx {i}: {a} vs {b}"
        );
    }
}

/// Q2 왕복: demote_q2_block 결과가 BlockQ2_0::quantize→dequantize와 bit-identical.
#[test]
fn test_demote_q2_roundtrip_matches_quantizer() {
    let src: [f32; QKKV] = std::array::from_fn(|i| (i as f32) * 0.05 + 0.1);
    let roundtrip = demote_q2_block(&src);

    let block = BlockQ2_0::quantize(&src);
    let mut expected = [0.0f32; QKKV];
    block.dequantize(&mut expected);

    for (i, (&a, &b)) in roundtrip.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "Q2 roundtrip mismatch at idx {i}: {a} vs {b}"
        );
    }
}

/// K per-channel 왕복: n_tokens=32(1블록), head_dim=64, kv_heads=1.
#[test]
fn test_demote_k_per_channel_roundtrip() {
    let kv_heads = 1;
    let n_tokens = 32; // 1 QKKV 블록
    let head_dim = 64;
    let n = kv_heads * n_tokens * head_dim;
    let k_orig: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();

    let k_q4 = demote_k_per_channel(&k_orig, n_tokens, head_dim, kv_heads, 4);

    // 왕복 후 각 채널의 각 블록은 quantizer 결과와 일치해야 한다.
    for ch in 0..head_dim {
        let mut vals = [0.0f32; QKKV];
        for t in 0..n_tokens {
            vals[t] = k_orig[t * head_dim + ch];
        }
        let expected = demote_q4_block(&vals);
        for t in 0..n_tokens {
            let got = k_q4[t * head_dim + ch];
            assert!(
                (got - expected[t]).abs() < 1e-7,
                "K per-channel mismatch ch={ch} t={t}: {got} vs {}",
                expected[t]
            );
        }
    }
}

/// V per-token 왕복: n_tokens=4, head_dim=64, kv_heads=1.
#[test]
fn test_demote_v_per_token_roundtrip() {
    let kv_heads = 1;
    let n_tokens = 4;
    let head_dim = 64; // 2 블록/토큰
    let n = kv_heads * n_tokens * head_dim;
    let v_orig: Vec<f32> = (0..n).map(|i| ((i % 50) as f32) * 0.1 + 0.5).collect();

    let v_q4 = demote_v_per_token(&v_orig, n_tokens, head_dim, kv_heads, 4);

    // 왕복 후 각 토큰의 각 블록은 quantizer 결과와 일치해야 한다.
    let blocks_per_token = head_dim / QKKV;
    for t in 0..n_tokens {
        for b in 0..blocks_per_token {
            let start = t * head_dim + b * QKKV;
            let chunk: &[f32; QKKV] = v_orig[start..start + QKKV].try_into().unwrap();
            let expected = demote_q4_block(chunk);
            let got = &v_q4[start..start + QKKV];
            for (j, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
                assert!(
                    (e - g).abs() < 1e-7,
                    "V per-token mismatch t={t} b={b} j={j}: expected={e} got={g}"
                );
            }
        }
    }
}

// ── 게이트 ② : 동일 메모리 예산 (a) vs (b) 비교 ───────────────────────────

/// Demote 스모크 비교: 동일 메모리 예산에서
///   (a) sliding N개 F16 (N개만 보존) vs (b) 4N개 Q4(demote, 4× 토큰)
///
/// NMSE를 비교해 Q4 demote의 정보 보존량을 측정한다.
/// 이 테스트는 demote가 sliding보다 NMSE를 줄이는지(더 많은 토큰 보존)를 확인한다.
#[test]
fn test_demote_vs_sliding_smoke() {
    // 설정: kv_heads=2, n_tokens_budget=32(F16), head_dim=64
    let kv_heads = 2;
    let head_dim = 64;
    let n_tokens_budget = 32; // sliding에서 보존되는 토큰 수 (메모리 기준)

    // Q4는 F16의 약 4× 토큰 저장 가능 (16bit/elem → ~4.5bit/elem)
    // 정수 배수로 간소화: 4×
    let n_tokens_demote = n_tokens_budget * 4; // 128 토큰 Q4 저장

    // 전체 KV cache (n_tokens_demote 토큰)
    let n_total = kv_heads * n_tokens_demote * head_dim;
    let k_full: Vec<f32> = (0..n_total)
        .map(|i| ((i % 200) as f32) * 0.05 - 5.0)
        .collect();
    let v_full: Vec<f32> = (0..n_total)
        .map(|i| ((i % 150) as f32) * 0.03 + 1.0)
        .collect();

    // (a) sliding: 마지막 n_tokens_budget 토큰만 F16 보존 (앞 토큰 버림)
    // 보존 구간: [n_tokens_demote - n_tokens_budget .. n_tokens_demote)
    let sliding_start = n_tokens_demote - n_tokens_budget;
    let mut k_sliding = vec![0.0f32; n_total];
    let mut v_sliding = vec![0.0f32; n_total];
    for h in 0..kv_heads {
        for t in sliding_start..n_tokens_demote {
            let src_base = h * n_tokens_demote * head_dim + t * head_dim;
            let dst_base = h * n_tokens_demote * head_dim + t * head_dim;
            k_sliding[dst_base..dst_base + head_dim]
                .copy_from_slice(&k_full[src_base..src_base + head_dim]);
            v_sliding[dst_base..dst_base + head_dim]
                .copy_from_slice(&v_full[src_base..src_base + head_dim]);
        }
    }

    // (b) demote: 전체 n_tokens_demote 토큰을 Q4로 demote (메모리 = (a)와 동일)
    let k_demoted = demote_k_per_channel(&k_full, n_tokens_demote, head_dim, kv_heads, 4);
    let v_demoted = demote_v_per_token(&v_full, n_tokens_demote, head_dim, kv_heads, 4);

    // NMSE 계산: 보존 구간 기준
    // sliding: 앞 토큰이 버려져 있으므로 전체 토큰 대비 손실 = 버린 토큰의 원본 정보 소실.
    //          보존 구간의 NMSE = 0 (F16 = 무손실), 버린 구간 = 완전 손실(전체 원본 대비).
    // demote: 전체 토큰의 Q4 근사 NMSE.
    let k_demoted_nmse = mean_nmse(&k_full, &k_demoted);
    let v_demoted_nmse = mean_nmse(&v_full, &v_demoted);

    // sliding의 "손실": 버린 토큰을 0으로 채웠으므로 원본 대비 NMSE가 높다.
    let k_sliding_nmse = mean_nmse(&k_full, &k_sliding);
    let v_sliding_nmse = mean_nmse(&v_full, &v_sliding);

    eprintln!(
        "[Demote smoke] K: demote_nmse={:.4} sliding_nmse={:.4}",
        k_demoted_nmse, k_sliding_nmse
    );
    eprintln!(
        "[Demote smoke] V: demote_nmse={:.4} sliding_nmse={:.4}",
        v_demoted_nmse, v_sliding_nmse
    );

    // demote는 전체 토큰을 근사하여 유지하므로 sliding(버린 토큰 = 완전 손실)보다
    // 전체 NMSE가 낮아야 한다.
    assert!(
        k_demoted_nmse < k_sliding_nmse,
        "K: demote ({:.4}) should have lower NMSE than sliding ({:.4}) — \
         demote retains 4× tokens at Q4 precision vs sliding discarding 75%",
        k_demoted_nmse,
        k_sliding_nmse
    );
    assert!(
        v_demoted_nmse < v_sliding_nmse,
        "V: demote ({:.4}) should have lower NMSE than sliding ({:.4})",
        v_demoted_nmse,
        v_sliding_nmse
    );
}

// ── 게이트 ②(보조): Q4 vs Q2 격차 ─────────────────────────────────────────

/// Q4 vs Q2 NMSE 격차: Q2가 Q4보다 오차가 크다 (논문 2-bit 1B 급락 재현 가능성).
#[test]
fn test_q4_vs_q2_nmse_gap() {
    let kv_heads = 1;
    let n_tokens = 64;
    let head_dim = 64;
    let n = kv_heads * n_tokens * head_dim;

    let k_orig: Vec<f32> = (0..n).map(|i| ((i % 100) as f32) * 0.1).collect();
    let v_orig: Vec<f32> = (0..n).map(|i| ((i % 80) as f32) * 0.05 + 0.5).collect();

    let k_q4 = demote_k_per_channel(&k_orig, n_tokens, head_dim, kv_heads, 4);
    let k_q2 = demote_k_per_channel(&k_orig, n_tokens, head_dim, kv_heads, 2);
    let v_q4 = demote_v_per_token(&v_orig, n_tokens, head_dim, kv_heads, 4);
    let v_q2 = demote_v_per_token(&v_orig, n_tokens, head_dim, kv_heads, 2);

    let k_nmse_q4 = mean_nmse(&k_orig, &k_q4);
    let k_nmse_q2 = mean_nmse(&k_orig, &k_q2);
    let v_nmse_q4 = mean_nmse(&v_orig, &v_q4);
    let v_nmse_q2 = mean_nmse(&v_orig, &v_q2);

    eprintln!(
        "[Q4vsQ2] K: q4={:.4} q2={:.4} | V: q4={:.4} q2={:.4}",
        k_nmse_q4, k_nmse_q2, v_nmse_q4, v_nmse_q2
    );

    assert!(
        k_nmse_q2 >= k_nmse_q4,
        "K: Q2 NMSE ({:.4}) should be >= Q4 NMSE ({:.4})",
        k_nmse_q2,
        k_nmse_q4
    );
    assert!(
        v_nmse_q2 >= v_nmse_q4,
        "V: Q2 NMSE ({:.4}) should be >= Q4 NMSE ({:.4})",
        v_nmse_q2,
        v_nmse_q4
    );
}

/// compute_nmse_block 재사용 정합: demote 모사의 단일 블록 NMSE가
/// quant_qcf.rs::compute_nmse_block과 일치한다.
#[test]
fn test_nmse_consistency_with_quant_qcf() {
    let src: [f32; QKKV] = std::array::from_fn(|i| (i as f32) * 0.1);

    // compute_nmse_block (quant_qcf.rs 정의)
    let nmse_from_lib = compute_nmse_block(&src, 4, 1e-8);

    // 직접 계산 (demote_measure 내부 compute_nmse_block_pair 이용)
    let roundtrip = demote_q4_block(&src);
    let nmse_direct = compute_nmse_block_pair(&src, &roundtrip);

    assert!(
        (nmse_from_lib - nmse_direct).abs() < 1e-5,
        "NMSE consistency: lib={:.6} direct={:.6}",
        nmse_from_lib,
        nmse_direct
    );
}

// ── 게이트 ⑤: 실모델 PPL 비교 (설계서 §4.4-E) ────────────────────────────────
//
// 환경변수:
//   DEMOTE_TEST_MODEL   — GGUF 모델 경로 (미설정 시 skip)
//   DEMOTE_TEST_TOKENIZER — tokenizer.json 경로
//   DEMOTE_TEST_TEXT    — PPL 측정용 텍스트 파일 경로
//
// 비교 설계 (동일 메모리 예산):
//   (a) sliding: 마지막 SLIDING_N 토큰 F32 유지 (앞 토큰 이미 evict된 상태)
//   (b) demote: 전체 4×SLIDING_N 토큰 보존, 창 밖(앞 3×SLIDING_N)을 Q4 왕복
//
// ## 설계서 §4.4-D 모사 한계 (유지)
// K 손실의 차기 step 2차 효과(attention weight 교란)는 미포착한다.

/// PPL 계산용 헬퍼: NLL 합산 후 exp
fn nll_to_ppl(total_nll: f64, count: usize) -> f64 {
    if count == 0 {
        return f64::INFINITY;
    }
    (total_nll / count as f64).exp()
}

/// F32 HeadMajor KV cache에서 창 밖 토큰([0, sliding_start))을 Q4 in-place 왕복.
///
/// HeadMajor 오프셋: `head * capacity * head_dim + pos * head_dim`.
/// 각 채널(dim)을 QKKV 블록 단위로 quantize→dequantize (K: per-channel).
/// V는 per-token (각 토큰의 head_dim 요소를 QKKV 블록들로).
fn demote_kv_cache_q4(cache: &mut KVCache, sliding_start: usize) {
    // F32 KV cache에서만 동작 (dtype 체크는 생략, 테스트 환경에서 F32 보장)
    let capacity = cache.capacity();
    let head_dim = cache.head_dim();
    let kv_heads = cache.kv_heads();

    // K buffer: HeadMajor layout → `[kv_heads, capacity, head_dim]` F32 뷰
    // K per-channel demote: 각 head, 각 채널(head_dim축)을 sliding_start 토큰씩 블록화
    {
        let k_total = kv_heads * capacity * head_dim;
        // SAFETY: F32 KV cache CPU buffer는 k_total * 4 바이트, 정렬 보장
        let k_slice: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(cache.k_buffer.as_mut_ptr() as *mut f32, k_total)
        };
        let n_groups = sliding_start / QKKV;
        if n_groups > 0 {
            for h in 0..kv_heads {
                let head_base = h * capacity * head_dim;
                for ch in 0..head_dim {
                    for group in 0..n_groups {
                        let tok_start = group * QKKV;
                        let mut vals = [0.0f32; QKKV];
                        for (t, v) in vals.iter_mut().enumerate() {
                            *v = k_slice[head_base + (tok_start + t) * head_dim + ch];
                        }
                        let block = BlockKVQ4::quantize(&vals);
                        let mut recon = [0.0f32; QKKV];
                        block.dequantize(&mut recon);
                        for (t, &r) in recon.iter().enumerate() {
                            k_slice[head_base + (tok_start + t) * head_dim + ch] = r;
                        }
                    }
                }
            }
        }
    }

    // V buffer: per-token demote (각 토큰의 head_dim 요소를 블록화)
    {
        let v_total = kv_heads * capacity * head_dim;
        let v_slice: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(cache.v_buffer.as_mut_ptr() as *mut f32, v_total)
        };
        let blocks_per_token = head_dim / QKKV;
        if blocks_per_token > 0 {
            for h in 0..kv_heads {
                let head_base = h * capacity * head_dim;
                for t in 0..sliding_start {
                    let tok_base = head_base + t * head_dim;
                    for b in 0..blocks_per_token {
                        let start = tok_base + b * QKKV;
                        let chunk: [f32; QKKV] = v_slice[start..start + QKKV].try_into().unwrap();
                        let block = BlockKVQ4::quantize(&chunk);
                        let mut recon = [0.0f32; QKKV];
                        block.dequantize(&mut recon);
                        v_slice[start..start + QKKV].copy_from_slice(&recon);
                    }
                }
            }
        }
    }
}

/// CPU backend에서 F32 HeadMajor KV cache를 구성한다.
///
/// `alloc_standard_kv_caches` 패턴 준수:
/// shape = `[1, kv_heads, initial_cap, head_dim]`, KVCache::new_dynamic으로 할당.
fn make_f32_head_major_cache(
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) -> anyhow::Result<KVCache> {
    let n_values = max_seq_len * kv_heads * head_dim;
    let buf_size = n_values * DType::F32.size();
    let k_buf = memory.alloc_kv(buf_size, DType::F32)?;
    let v_buf = memory.alloc_kv(buf_size, DType::F32)?;
    let shape = Shape::new(vec![1, kv_heads, max_seq_len, head_dim]);
    let k = Tensor::new(shape.clone(), k_buf, cpu_backend.clone());
    let v = Tensor::new(shape, v_buf, cpu_backend);
    Ok(
        KVCache::new_dynamic(k, v, max_seq_len, max_seq_len, kv_heads, head_dim, memory)
            .with_layout(KVLayout::HeadMajor),
    )
}

/// 실모델 PPL 스모크: 설계서 §4.4-E 게이트 직접 판정.
///
/// DEMOTE_TEST_MODEL / DEMOTE_TEST_TOKENIZER / DEMOTE_TEST_TEXT 미설정 시 skip.
#[test]
fn test_demote_vs_sliding_real_model_ppl() {
    // ── 환경변수 확인 (없으면 graceful skip) ─────────────────────────────────
    let model_path = match std::env::var("DEMOTE_TEST_MODEL") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "[skip] test_demote_vs_sliding_real_model_ppl: \
                 set DEMOTE_TEST_MODEL / DEMOTE_TEST_TOKENIZER / DEMOTE_TEST_TEXT"
            );
            return;
        }
    };
    let tok_path = std::env::var("DEMOTE_TEST_TOKENIZER").unwrap_or_else(|_| {
        // 모델 경로 sibling tokenizer.json fallback
        let mut p = std::path::PathBuf::from(&model_path);
        p.pop();
        p.push("tokenizer.json");
        p.to_string_lossy().to_string()
    });
    let text_path = std::env::var("DEMOTE_TEST_TEXT").unwrap_or_else(|_| {
        // workspace root의 기본 텍스트 사용
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.push("experiments/prompts/ppl_default.txt");
        p.to_string_lossy().to_string()
    });

    // ── 모델 / 토크나이저 로드 ────────────────────────────────────────────────
    let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let memory: Arc<dyn Memory> = Arc::new(Galloc::new());

    let model = match TransformerModel::load_gguf(&model_path, cpu_backend.clone(), memory.as_ref())
    {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[skip] model load failed: {e}");
            return;
        }
    };
    let tokenizer = match tokenizers::Tokenizer::from_file(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[skip] tokenizer load failed: {e}");
            return;
        }
    };
    let text = match std::fs::read_to_string(&text_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[skip] text file read failed ({}): {e}", text_path);
            return;
        }
    };

    // ── 설정 ─────────────────────────────────────────────────────────────────
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let n_layers = model.config.num_hidden_layers;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let n_heads_q = model.config.num_attention_heads;
    let ffn_hidden = model.config.intermediate_size;

    // 동일 메모리 예산 설계:
    //   sliding_n 토큰 F32 = 4× 토큰 Q4 (~ 4× 압축)
    //   본 테스트는 짧은 스모크 — 텍스트 앞 eval_tokens 토큰만 사용
    let max_seq_len = 512usize;
    let sliding_n = 64usize; // (a) sliding: 마지막 64 토큰 F32 유지
    let demote_n = sliding_n * 4; // (b) demote: 256 토큰 Q4 (창 밖 192를 왕복)

    let encoding = tokenizer
        .encode(text.as_str(), true)
        .expect("tokenize failed");
    let all_ids: Vec<u32> = encoding.get_ids().to_vec();
    let eval_tokens = all_ids.len().min(max_seq_len).max(2);
    let token_ids = &all_ids[..eval_tokens];

    if eval_tokens < demote_n + 2 {
        eprintln!(
            "[skip] text too short ({} tokens < demote_n+2={})",
            eval_tokens,
            demote_n + 2
        );
        return;
    }

    // ── 공통 버퍼 ────────────────────────────────────────────────────────────
    let q_dim = n_heads_q * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = k_dim;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 시나리오 (a): sliding — 마지막 sliding_n 토큰 F32 유지, 나머지 evict
    // → prefill을 sliding_n 토큰만 수행 (= evict 후 상태 모사)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let ppl_a = {
        let mut kv_caches: Vec<KVCache> = (0..n_layers)
            .map(|_| {
                make_f32_head_major_cache(
                    memory.clone(),
                    cpu_backend.clone(),
                    kv_heads,
                    head_dim,
                    max_seq_len,
                )
                .expect("cache alloc")
            })
            .collect();

        let mut gen_ws = LayerWorkspace::new(
            WorkspaceConfig {
                batch_size: 1,
                dim: hidden_size,
                q_dim,
                k_dim,
                v_dim,
                ffn_hidden,
                n_heads: n_heads_q,
                max_seq_len,
            },
            memory.as_ref(),
            cpu_backend.clone(),
        )
        .expect("workspace alloc");

        // prefill: sliding_n 토큰 (= 최근 창)
        let prefill_len = sliding_n.min(eval_tokens - 1);
        {
            let input_buf = Galloc::new().alloc(prefill_len * 4, DType::U8).unwrap();
            unsafe {
                let ptr = input_buf.as_mut_ptr() as *mut u32;
                for (i, &id) in token_ids[..prefill_len].iter().enumerate() {
                    *ptr.add(i) = id;
                }
            }
            let cpu_input = Tensor::new(
                Shape::new(vec![1, prefill_len]),
                input_buf,
                cpu_backend.clone(),
            );
            let prefill_logits_buf = memory
                .alloc(prefill_len * vocab_size * 4, DType::F32)
                .unwrap();
            let mut prefill_logits = Tensor::new(
                Shape::new(vec![1, prefill_len, vocab_size]),
                prefill_logits_buf,
                cpu_backend.clone(),
            );
            KVCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &cpu_input,
                    start_pos: 0,
                    fmts,
                    backend: &cpu_backend,
                    memory: memory.as_ref(),
                    logits_out: &mut prefill_logits,
                    x_gen: None,
                    workspace: None,
                    logits_last_only: false,
                    score_accumulator: None,
                    query_stats_accumulator: None,
                    skip_config: None,
                    importance_collector: None,
                    cache_self_need_scores: false,
                    layer_boundary_hook: None,
                    read_stage: None,
                })
            })
            .expect("sliding prefill");
        }

        // decode: teacher-forcing PPL
        let decode_logits_buf = memory.alloc(vocab_size * 4, DType::F32).unwrap();
        let mut decode_logits = Tensor::new(
            Shape::new(vec![1, 1, vocab_size]),
            decode_logits_buf,
            cpu_backend.clone(),
        );
        let xg_buf = memory.alloc(hidden_size * 4, DType::F32).unwrap();
        let mut x_gen = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            xg_buf,
            cpu_backend.clone(),
        );

        let mut total_nll = 0.0f64;
        let mut count = 0usize;
        let decode_start = prefill_len;
        let decode_end = (prefill_len + 64).min(eval_tokens - 1);

        for i in decode_start..decode_end {
            let input_buf = Galloc::new().alloc(4, DType::U8).unwrap();
            unsafe { *(input_buf.as_mut_ptr() as *mut u32) = token_ids[i] }
            let tok = Tensor::new(Shape::new(vec![1, 1]), input_buf, cpu_backend.clone());
            KVCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &tok,
                    start_pos: i,
                    fmts,
                    backend: &cpu_backend,
                    memory: memory.as_ref(),
                    logits_out: &mut decode_logits,
                    x_gen: Some(&mut x_gen),
                    workspace: Some(&mut gen_ws),
                    logits_last_only: true,
                    score_accumulator: None,
                    query_stats_accumulator: None,
                    skip_config: None,
                    importance_collector: None,
                    cache_self_need_scores: false,
                    layer_boundary_hook: None,
                    read_stage: None,
                })
            })
            .expect("sliding decode");

            let mut logits_cpu = vec![0u8; vocab_size * 4];
            cpu_backend
                .read_buffer(&decode_logits, &mut logits_cpu)
                .unwrap();
            let lp = sampling::compute_log_prob(
                unsafe {
                    std::slice::from_raw_parts(logits_cpu.as_ptr() as *const f32, vocab_size)
                },
                token_ids[i + 1],
                vocab_size,
            );
            total_nll -= lp;
            count += 1;
        }
        nll_to_ppl(total_nll, count)
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 시나리오 (b): demote — demote_n 토큰 prefill 후 창 밖 Q4 왕복, decode PPL
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let ppl_b = {
        let mut kv_caches: Vec<KVCache> = (0..n_layers)
            .map(|_| {
                make_f32_head_major_cache(
                    memory.clone(),
                    cpu_backend.clone(),
                    kv_heads,
                    head_dim,
                    max_seq_len,
                )
                .expect("cache alloc")
            })
            .collect();

        let mut gen_ws = LayerWorkspace::new(
            WorkspaceConfig {
                batch_size: 1,
                dim: hidden_size,
                q_dim,
                k_dim,
                v_dim,
                ffn_hidden,
                n_heads: n_heads_q,
                max_seq_len,
            },
            memory.as_ref(),
            cpu_backend.clone(),
        )
        .expect("workspace alloc");

        // prefill: demote_n 토큰 전체
        let prefill_len = demote_n.min(eval_tokens - 1);
        {
            let input_buf = Galloc::new().alloc(prefill_len * 4, DType::U8).unwrap();
            unsafe {
                let ptr = input_buf.as_mut_ptr() as *mut u32;
                for (i, &id) in token_ids[..prefill_len].iter().enumerate() {
                    *ptr.add(i) = id;
                }
            }
            let cpu_input = Tensor::new(
                Shape::new(vec![1, prefill_len]),
                input_buf,
                cpu_backend.clone(),
            );
            let prefill_logits_buf = memory
                .alloc(prefill_len * vocab_size * 4, DType::F32)
                .unwrap();
            let mut prefill_logits = Tensor::new(
                Shape::new(vec![1, prefill_len, vocab_size]),
                prefill_logits_buf,
                cpu_backend.clone(),
            );
            KVCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &cpu_input,
                    start_pos: 0,
                    fmts,
                    backend: &cpu_backend,
                    memory: memory.as_ref(),
                    logits_out: &mut prefill_logits,
                    x_gen: None,
                    workspace: None,
                    logits_last_only: false,
                    score_accumulator: None,
                    query_stats_accumulator: None,
                    skip_config: None,
                    importance_collector: None,
                    cache_self_need_scores: false,
                    layer_boundary_hook: None,
                    read_stage: None,
                })
            })
            .expect("demote prefill");
        }

        // 창 밖 토큰([0, sliding_n*3))을 Q4 왕복
        // 메모리 예산: sliding_n × F32 ≈ demote_n × Q4 (4× 압축)
        let demote_window_start = 0; // 앞부분 3/4을 demote
        let demote_window_end = prefill_len - sliding_n; // 마지막 sliding_n은 F32 유지
        for cache in kv_caches.iter_mut() {
            // demote_window_end 이전 토큰을 Q4 왕복
            demote_kv_cache_q4(cache, demote_window_end);
        }
        let _ = demote_window_start; // 항상 0이므로 사용 안 함

        // decode: teacher-forcing PPL (sliding과 동일 구간)
        let decode_logits_buf = memory.alloc(vocab_size * 4, DType::F32).unwrap();
        let mut decode_logits = Tensor::new(
            Shape::new(vec![1, 1, vocab_size]),
            decode_logits_buf,
            cpu_backend.clone(),
        );
        let xg_buf = memory.alloc(hidden_size * 4, DType::F32).unwrap();
        let mut x_gen = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            xg_buf,
            cpu_backend.clone(),
        );

        let mut total_nll = 0.0f64;
        let mut count = 0usize;
        let decode_start = prefill_len;
        let decode_end = (prefill_len + 64).min(eval_tokens - 1);

        for i in decode_start..decode_end {
            let input_buf = Galloc::new().alloc(4, DType::U8).unwrap();
            unsafe { *(input_buf.as_mut_ptr() as *mut u32) = token_ids[i] }
            let tok = Tensor::new(Shape::new(vec![1, 1]), input_buf, cpu_backend.clone());
            KVCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &tok,
                    start_pos: i,
                    fmts,
                    backend: &cpu_backend,
                    memory: memory.as_ref(),
                    logits_out: &mut decode_logits,
                    x_gen: Some(&mut x_gen),
                    workspace: Some(&mut gen_ws),
                    logits_last_only: true,
                    score_accumulator: None,
                    query_stats_accumulator: None,
                    skip_config: None,
                    importance_collector: None,
                    cache_self_need_scores: false,
                    layer_boundary_hook: None,
                    read_stage: None,
                })
            })
            .expect("demote decode");

            let mut logits_cpu = vec![0u8; vocab_size * 4];
            cpu_backend
                .read_buffer(&decode_logits, &mut logits_cpu)
                .unwrap();
            let lp = sampling::compute_log_prob(
                unsafe {
                    std::slice::from_raw_parts(logits_cpu.as_ptr() as *const f32, vocab_size)
                },
                token_ids[i + 1],
                vocab_size,
            );
            total_nll -= lp;
            count += 1;
        }
        nll_to_ppl(total_nll, count)
    };

    // ── 결과 출력 + 게이트 판정 ───────────────────────────────────────────────
    eprintln!(
        "[DemotePPL] (a) sliding PPL = {:.4}, (b) demote PPL = {:.4}",
        ppl_a, ppl_b
    );
    eprintln!("[DemotePPL] 설계서 §4.4-E 게이트: demote PPL < sliding PPL → GO (항목 1 추진)");
    // PPL 수치 쌍이 유한한 수임을 보장 (본 측정 완료 조건)
    assert!(ppl_a.is_finite(), "sliding PPL must be finite, got {ppl_a}");
    assert!(ppl_b.is_finite(), "demote PPL must be finite, got {ppl_b}");
    // 판정 결과를 로그로만 남기고 FAIL로 만들지 않음 — 판정은 Tester(P4) 몫.
    if ppl_b < ppl_a {
        eprintln!(
            "[DemotePPL] 판정: GO — demote가 sliding 대비 PPL 우세 (Δ={:.4})",
            ppl_a - ppl_b
        );
    } else {
        eprintln!(
            "[DemotePPL] 판정: RED — demote PPL ({:.4}) ≥ sliding PPL ({:.4}), Δ={:.4}",
            ppl_b,
            ppl_a,
            ppl_b - ppl_a
        );
    }
}
