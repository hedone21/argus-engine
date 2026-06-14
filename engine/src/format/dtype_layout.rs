//! DType → `KVLayoutDesc` 브리지 + descriptor-구동 generic block-quant unpacker.
//!
//! 설계 SSOT: `docs/adr/0005-format-backend-capability-plugin-unification.md` D5(generic floor).
//!
//! D5 의 generic floor 는 hot format(Q4_0 등) 특화 arm 밖의 dtype 를 **dequant→f32 matmul**
//! 로 처리한다. 그 dequant 는 per-dtype if-else 의 단순 relocation 이 아니라, descriptor 어휘
//! (`block_elems`/`bits`/`scale_layout`/`packing`)로 구동되는 **family-generic block-quant
//! unpacker** 다(D5). 단 **byte-exact** 가 최우선 — 결과는 `quant.rs` 의 per-block
//! `dequantize()` 와 bit-identical 해야 한다(아래 테스트로 강제).
//!
//! 어휘 밖(mxfp4 shared-exponent·codebook·sparse)은 floor 밖 escape — 여기 None 을 반환하고
//! 호출부(backend dispatch)가 loud-fail 한다.

use crate::buffer::DType;
use crate::buffer::opaque::OpaqueBuffer;
use crate::memory::host::shared::SharedBuffer;
use crate::shape::Shape;
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use half::f16;
use std::sync::Arc;
use technique_api::{KVLayoutDesc, Packing, ScaleLayout};

/// `DType` → block-quant family descriptor 도출.
///
/// block-quant family(q4_0/q4_1/q8_0) 와 raw(f32/f16/bf16)만 표현 가능.
/// matmul 부적합 dtype(U8 등)은 `None` — 호출부가 loud-fail 한다(floor 밖 escape).
pub fn dtype_to_layout_desc(d: DType) -> Option<KVLayoutDesc> {
    Some(match d {
        DType::Q8_0 => KVLayoutDesc {
            block_elems: 32,
            bits: 8,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Byte,
        },
        DType::Q4_0 => KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        },
        DType::Q4_1 => KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16WithMin,
            packing: Packing::Nibble,
        },
        DType::F16 => KVLayoutDesc {
            block_elems: 1,
            bits: 16,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        },
        DType::F32 => KVLayoutDesc {
            block_elems: 1,
            bits: 32,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        },
        DType::BF16 => KVLayoutDesc {
            block_elems: 1,
            bits: 16,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        },
        // matmul 부적합 dtype → floor 밖 escape.
        DType::U8 => return None,
    })
}

/// `KVLayoutDesc` → 내장 `DType` ([`dtype_to_layout_desc`] 의 부분역).
///
/// 동적 `.so` format 의 descriptor 가 내장 DType 과 **bit-equivalent** 면 opaque generic floor 대신
/// **typed fast path**(특화 NEON dequant-attention 등)로 라우팅하기 위한 인식(
/// name-keyed dispatch 를 descriptor-keyed 로 확장, 2026-06-09 결정). opaque floor 는
/// dequant-whole→F32 라 ARM 에서 typed Q4_0(NEON) 대비 ~1.34x 느림(Galaxy S25 실측) — descriptor 가
/// 내장과 일치하면 floor 비용이 불필요하다.
///
/// **인식 집합 = {F32, F16, Q4_0}** = `--kv-type` 이 받는 typed-attention 완전지원 집합. Q8_0/Q4_1
/// 등은 typed attention 이 보장되지 않아 제외(opaque floor 유지가 안전). 어휘 밖 novel descriptor 도
/// `None` → 호출부가 opaque floor 로 처리.
pub fn layout_desc_to_builtin_dtype(desc: &KVLayoutDesc) -> Option<DType> {
    [DType::F32, DType::F16, DType::Q4_0]
        .into_iter()
        .find(|&dt| dtype_to_layout_desc(dt).as_ref() == Some(desc))
}

/// 한 block-quant 블록을 descriptor 어휘로 구동해 f32 로 unpack(canonical layout, llama.cpp 호환).
///
/// canonical block layout = `[f16 scale][f16 min?][packed quants]`:
/// - `PerBlockF16`        → `[scale][quants]`           (q4_0/q8_0)
/// - `PerBlockF16WithMin` → `[scale][min][quants]`      (q4_1)
///
/// nibble(4-bit) interleave 규약(llama.cpp): low nibble → `out[i]`, high nibble → `out[i + n/2]`.
/// nibble signed/unsigned 는 scale_layout 로 갈린다:
/// - min 없음(`PerBlockF16`)        → signed, zero-point = 2^(bits-1)  (q4_0: nibble−8)
/// - min 있음(`PerBlockF16WithMin`) → unsigned, dequant = quant·scale + min  (q4_1)
///
/// byte(8-bit) packing 은 signed i8 직저장(q8_0).
///
/// `block` 은 정확히 한 블록의 raw 바이트(scale/min/quants 포함). `out` 길이 = `block_elems`.
fn unpack_block_via_descriptor(desc: &KVLayoutDesc, block: &[u8], out: &mut [f32]) {
    let n = desc.block_elems as usize;
    debug_assert_eq!(out.len(), n);

    // scale (f16, little-endian) at offset 0.
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

    let (min, quants) = match desc.scale_layout {
        ScaleLayout::PerBlockF16WithMin => {
            let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
            (m, &block[4..])
        }
        ScaleLayout::PerBlockF16 => (0.0f32, &block[2..]),
        // raw(Dense) 포맷은 이 함수 경로로 오지 않는다(아래 dequant_via_descriptor 가 직접 처리).
        ScaleLayout::None => (0.0f32, &block[2..]),
    };

    match desc.packing {
        Packing::Nibble => {
            // half = n/2 packed bytes; low nibble → out[i], high nibble → out[i + half].
            let half = n / 2;
            let signed = matches!(desc.scale_layout, ScaleLayout::PerBlockF16);
            for i in 0..half {
                let b = quants[i];
                let lo = (b & 0x0F) as i32;
                let hi = (b >> 4) as i32;
                if signed {
                    // q4_0: zero-point = 2^(bits-1) = 8.
                    let zp = 1i32 << (desc.bits - 1);
                    out[i] = (lo - zp) as f32 * scale;
                    out[i + half] = (hi - zp) as f32 * scale;
                } else {
                    // q4_1: unsigned quant · scale + min.
                    out[i] = lo as f32 * scale + min;
                    out[i + half] = hi as f32 * scale + min;
                }
            }
        }
        Packing::Byte => {
            // q8_0: signed i8 · scale.
            for (i, o) in out.iter_mut().enumerate().take(n) {
                let q = quants[i] as i8 as i32;
                *o = q as f32 * scale;
            }
        }
        // Dense 는 이 함수로 오지 않는다(raw 경로는 dequant_via_descriptor 가 직접 처리).
        Packing::Dense => {
            debug_assert!(
                false,
                "Dense packing must not reach unpack_block_via_descriptor"
            );
        }
    }
}

/// descriptor-구동 generic dequant: weight tensor `b` 를 f32 `Vec<f32>` 로 unpack(D5 floor).
///
/// `b.dtype()` 에서 descriptor 를 도출하고, block-quant family 면 canonical layout 으로
/// block 단위 unpack 한다(`unpack_block_via_descriptor`). raw(f32/f16/bf16)는 직접 변환.
/// descriptor 도출 불가(`None`)면 loud-fail `Err`(floor 밖 escape — 기존 "Unsupported" 메시지 보존).
///
/// 출력은 row-major f32 (b 와 동일 element 순서, 총 `b.numel()` 개).
pub fn dequant_via_descriptor(b: &Tensor) -> Result<Vec<f32>> {
    let dtype = b.dtype();
    // opaque 버퍼(dtype=U8)는 closed `DType` 가 못 담는 format 을 sidecar
    // `KVLayoutDesc` 로 운반한다 → descriptor 를 우선 sidecar 에서, 없으면 dtype 에서 도출.
    let desc = b
        .buffer()
        .as_any()
        .downcast_ref::<OpaqueBuffer>()
        .map(|op| op.descriptor())
        .or_else(|| dtype_to_layout_desc(dtype))
        .ok_or_else(|| anyhow!("Unsupported dtype for matmul (no layout descriptor): {dtype:?}"))?;

    let numel = b.numel();

    // raw(Dense) 포맷 — 직접 변환(블록 unpack 불필요).
    if matches!(desc.packing, Packing::Dense) {
        return match dtype {
            DType::F32 => Ok(b.as_slice::<f32>()[..numel].to_vec()),
            DType::F16 => Ok(b.as_slice::<f16>()[..numel]
                .iter()
                .map(|x| x.to_f32())
                .collect()),
            DType::BF16 => Ok(b.as_slice::<half::bf16>()[..numel]
                .iter()
                .map(|x| x.to_f32())
                .collect()),
            // opaque-Dense(raw 를 opaque 로 운반)는 floor 밖 — typed read 금지.
            other => Err(anyhow!(
                "Dense descriptor with non-raw dtype {other:?} (opaque-Dense not supported by floor)"
            )),
        };
    }

    // block-quant family — block 단위 unpack.
    let block_elems = desc.block_elems as usize;
    if !numel.is_multiple_of(block_elems) {
        return Err(anyhow!(
            "matmul weight numel {numel} not a multiple of block_elems {block_elems} for {dtype:?}"
        ));
    }
    let n_blocks = numel / block_elems;

    // 블록 raw 바이트 크기: scale + quants. 단일원천 — Dense 는 위에서 이미
    // 처리·반환됐으므로 여기선 block-quant 라 항상 Some.
    let block_bytes = desc
        .block_bytes()
        .expect("block-quant family has block_bytes (Dense returned above)");

    // b 의 raw 바이트(packed block 연속).
    let raw = b.as_slice::<u8>();
    if raw.len() < n_blocks * block_bytes {
        return Err(anyhow!(
            "matmul weight raw bytes {} < expected {} ({} blocks × {} bytes) for {dtype:?}",
            raw.len(),
            n_blocks * block_bytes,
            n_blocks,
            block_bytes
        ));
    }

    let mut out = vec![0.0f32; numel];
    for bi in 0..n_blocks {
        let byte_off = bi * block_bytes;
        let elem_off = bi * block_elems;
        unpack_block_via_descriptor(
            &desc,
            &raw[byte_off..byte_off + block_bytes],
            &mut out[elem_off..elem_off + block_elems],
        );
    }
    Ok(out)
}

/// generic floor 의 임시 f32 weight tensor 생성.
///
/// `b` 를 [`dequant_via_descriptor`] 로 f32 로 푼 뒤, **b 와 동일 shape**·dtype=F32 인 임시
/// Tensor 를 만든다. backend 의 floor arm 이 이걸 `matmul_transposed_f32(a, &f32_b, out)` 에
/// 넘긴다 — dequant 무손실 + f32 matmul 동일이라 **exact**(느릴 뿐).
///
/// backend 는 인자로 받지 않고 CPU scalar backend 를 붙인다(matmul_transposed_f32 는 backend
/// 인스턴스를 쓰지 않고 raw f32 slice 만 읽으므로 무해).
pub fn dequant_to_f32_tensor(b: &Tensor) -> Result<Tensor> {
    let f32_data = dequant_via_descriptor(b)?;
    // Vec<f32> → Vec<u8> (canonical little-endian, host-native; matmul 은 같은 호스트에서 읽음).
    let mut bytes = Vec::<u8>::with_capacity(f32_data.len() * 4);
    for v in &f32_data {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    let buf = SharedBuffer::from_vec(bytes, DType::F32);
    Ok(Tensor::new(
        Shape::new(b.shape().dims().to_vec()),
        Arc::new(buf),
        b.backend().clone(),
    ))
}

/// descriptor-구동 block-quant encoder ([`unpack_block_via_descriptor`] 의 역).
///
/// `src`(row-major f32, len = `n_blocks * block_elems`)를 canonical block layout 으로 양자화해
/// `dst`(len ≥ `n_blocks * block_bytes`)에 packed bytes 로 쓴다. **batch** 시그니처(
/// 향후 `.so` vtable indirect call 빈도를 block 단위로 낮춤, panic-free Result).
///
/// 현재 지원 = **PerBlockF16 + Nibble**(q4_0 canonical symmetric). 그 외 family(Byte/WithMin/
/// Dense)는 범위 한정 `Err`(format-bound encoder 가 비-canonical 정책을 코드로
/// 공급할 때 확장). 출력은 `quant.rs::BlockQ4_0::quantize` 와 **byte-exact**(아래 테스트로 강제).
/// `encode_via_descriptor` 의 역연산 — block-quant raw bytes(`src` = n_blocks·block_bytes)를
/// f32(`dst` = n_blocks·block_elems)로 unpack. opaque KV merge(D2O, Stage 3)의 block-level
/// dequant 에 쓴다. `dst.len()` 은 `block_elems` 의 배수여야 한다.
pub fn decode_via_descriptor(desc: &KVLayoutDesc, src: &[u8], dst: &mut [f32]) {
    let block_elems = desc.block_elems as usize;
    let block_bytes = desc
        .block_bytes()
        .expect("decode_via_descriptor: block-quant descriptor (Dense not allowed)");
    let n_blocks = dst.len() / block_elems;
    for bi in 0..n_blocks {
        unpack_block_via_descriptor(
            desc,
            &src[bi * block_bytes..(bi + 1) * block_bytes],
            &mut dst[bi * block_elems..(bi + 1) * block_elems],
        );
    }
}

pub fn encode_via_descriptor(desc: &KVLayoutDesc, src: &[f32], dst: &mut [u8]) -> Result<()> {
    if desc.scale_layout != ScaleLayout::PerBlockF16 || desc.packing != Packing::Nibble {
        return Err(anyhow!(
            "encode_via_descriptor: only PerBlockF16/Nibble(q4_0 canonical) supported, got {:?}/{:?}",
            desc.scale_layout,
            desc.packing
        ));
    }
    let block_elems = desc.block_elems as usize;
    let block_bytes = desc.block_bytes().expect("Nibble packing has block_bytes");
    if block_elems == 0 || !src.len().is_multiple_of(block_elems) {
        return Err(anyhow!(
            "encode_via_descriptor: src len {} not a multiple of block_elems {block_elems}",
            src.len()
        ));
    }
    let n_blocks = src.len() / block_elems;
    if dst.len() < n_blocks * block_bytes {
        return Err(anyhow!(
            "encode_via_descriptor: dst {} < {} ({} blocks × {} bytes)",
            dst.len(),
            n_blocks * block_bytes,
            n_blocks,
            block_bytes
        ));
    }
    for bi in 0..n_blocks {
        encode_block_perblockf16_nibble(
            desc.bits as u32,
            block_elems,
            &src[bi * block_elems..(bi + 1) * block_elems],
            &mut dst[bi * block_bytes..(bi + 1) * block_bytes],
        );
    }
    Ok(())
}

/// q4_0-style symmetric 단일 블록 encode(PerBlockF16 + Nibble). `quant.rs::BlockQ4_0::quantize`
/// 의 descriptor-generic 미러(byte-exact): `d = max|x| / qmax`, `round().clamp(qmin, qmax)`,
/// zero-point `2^(bits-1)`. `dst` = `[f16 scale][n/2 packed nibbles]`(low→i, high→i+n/2).
fn encode_block_perblockf16_nibble(bits: u32, n: usize, src: &[f32], dst: &mut [u8]) {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32; // bits=4 → 7
    let qmin = -((1i32 << (bits - 1)) as f32); // bits=4 → -8
    let zp = 1i32 << (bits - 1); // bits=4 → 8
    let half = n / 2;

    let max_abs = src.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let d = max_abs / qmax;
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };

    dst[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    for i in 0..half {
        let v0 = (src[i] * id).round().clamp(qmin, qmax) as i32;
        let v1 = (src[i + half] * id).round().clamp(qmin, qmax) as i32;
        dst[2 + i] = ((v0 + zp) as u8) | (((v1 + zp) as u8) << 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{BlockQ4_0, BlockQ8_0, QK4_0, QK8_0};

    // ── (a0) KVLayoutDesc byte-회계 == engine block 구조체 크기 ──

    /// `bytes_for_elems(block_elems)` 가 `size_of::<Block*>()` 와 일치 — technique-api
    /// literal 검증(lib.rs)의 engine 측 cross-check(단일원천 drift 가드).
    #[test]
    fn bytes_for_elems_matches_block_struct_size() {
        use crate::quant::{BlockQ4_1, QK4_1};
        assert_eq!(
            dtype_to_layout_desc(DType::Q4_0)
                .unwrap()
                .bytes_for_elems(QK4_0),
            Some(std::mem::size_of::<BlockQ4_0>())
        );
        assert_eq!(
            dtype_to_layout_desc(DType::Q4_1)
                .unwrap()
                .bytes_for_elems(QK4_1),
            Some(std::mem::size_of::<BlockQ4_1>())
        );
        assert_eq!(
            dtype_to_layout_desc(DType::Q8_0)
                .unwrap()
                .bytes_for_elems(QK8_0),
            Some(std::mem::size_of::<BlockQ8_0>())
        );
        // raw: F32 = numel*4, F16 = numel*2.
        assert_eq!(
            dtype_to_layout_desc(DType::F32)
                .unwrap()
                .bytes_for_elems(10),
            Some(40)
        );
        assert_eq!(
            dtype_to_layout_desc(DType::F16)
                .unwrap()
                .bytes_for_elems(10),
            Some(20)
        );
    }

    /// descriptor-keyed typed fast path 인식(2026-06-09): 동적 format 의 descriptor 가 내장 DType 과
    /// bit-equivalent 면 `Some(dt)`(→ typed alloc), 아니면 `None`(→ opaque floor).
    #[test]
    fn layout_desc_to_builtin_dtype_recognizes_q4_0_canonical_only() {
        // q4_0-canonical(synth_q4/bundle_fmt/example_kv 공유) → Q4_0 인식.
        let q4 = KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        };
        assert_eq!(layout_desc_to_builtin_dtype(&q4), Some(DType::Q4_0));
        // F32/F16 raw 도 인식(typed attention 완전지원).
        assert_eq!(
            layout_desc_to_builtin_dtype(&dtype_to_layout_desc(DType::F32).unwrap()),
            Some(DType::F32)
        );
        assert_eq!(
            layout_desc_to_builtin_dtype(&dtype_to_layout_desc(DType::F16).unwrap()),
            Some(DType::F16)
        );
        // Q8_0-canonical(mf_q8 등)은 typed attention 미보장 → 인식 제외(opaque floor 유지).
        assert_eq!(
            layout_desc_to_builtin_dtype(&dtype_to_layout_desc(DType::Q8_0).unwrap()),
            None
        );
        // novel descriptor(어휘 밖 3-bit 가정) → None.
        let novel = KVLayoutDesc {
            block_elems: 32,
            bits: 3,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        };
        assert_eq!(layout_desc_to_builtin_dtype(&novel), None);
    }

    /// opaque 버퍼(U8 tag + q4_0 sidecar desc)의 dequant 이 같은 바이트를
    /// Q4_0 dtype 으로 푼 결과와 bit-identical — floor 가 sidecar descriptor 를 dtype 과
    /// 동등하게 인식(opaque KV attention 의 read 토대).
    #[test]
    fn dequant_via_descriptor_recognizes_opaque_sidecar() {
        use crate::backend::Backend;
        use crate::backend::cpu::CpuBackend;
        use crate::buffer::Buffer;
        use crate::buffer::opaque::OpaqueBuffer;
        use crate::memory::host::shared::SharedBuffer;
        use crate::shape::Shape;
        use std::sync::Arc;

        // 임의 q4_0 블록 1개의 raw 바이트.
        let blk = BlockQ4_0 {
            d: f16::from_f32(-1.25),
            qs: std::array::from_fn(|i| ((i * 13 + 1) % 256) as u8),
        };
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                (&blk as *const BlockQ4_0) as *const u8,
                std::mem::size_of::<BlockQ4_0>(),
            )
        }
        .to_vec();
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let shape = Shape::new(vec![QK4_0]); // numel = 32 logical elements, buffer = 18 packed bytes

        // baseline: Q4_0 dtype tensor.
        let q4_buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::from_vec(bytes.clone(), DType::Q4_0));
        let q4_tensor = Tensor::new(shape.clone(), q4_buf, backend.clone());
        let want = dequant_via_descriptor(&q4_tensor).unwrap();

        // opaque: U8 tag + q4_0 sidecar desc, 같은 바이트.
        let q4_0_desc = dtype_to_layout_desc(DType::Q4_0).unwrap();
        let inner: Arc<dyn Buffer> = Arc::new(SharedBuffer::from_vec(bytes, DType::U8));
        let opaque: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner, q4_0_desc));
        let opaque_tensor = Tensor::new(shape, opaque, backend);
        let got = dequant_via_descriptor(&opaque_tensor).unwrap();

        assert_eq!(got.len(), QK4_0);
        assert_eq!(got, want, "opaque sidecar dequant == Q4_0 dtype dequant");
    }

    /// encode_via_descriptor(q4_0) 가 BlockQ4_0::quantize 와 **byte-exact**.
    /// write 거울 게이트 — read 쪽 test_generic_unpack_q4_0_bit_identical 의 짝.
    #[test]
    fn encode_via_descriptor_q4_0_byte_exact_vs_blockq4_0_quantize() {
        let desc = dtype_to_layout_desc(DType::Q4_0).unwrap();
        let block_bytes = desc.block_bytes().unwrap(); // 18

        // 0 / 양·음 혼합 / 경계값 / 미세값 블록들.
        let cases: Vec<[f32; QK4_0]> = vec![
            [0.0; QK4_0],
            std::array::from_fn(|i| (i as f32 - 16.0) * 0.1),
            std::array::from_fn(|i| ((i * 37 % 23) as f32 - 11.0) * 0.37),
            std::array::from_fn(|i| if i % 2 == 0 { 3.14 } else { -2.71 }),
            std::array::from_fn(|i| (i as f32) * 1.0e-4),
        ];
        let mut src = Vec::<f32>::new();
        for c in &cases {
            src.extend_from_slice(c);
        }
        let mut dst = vec![0u8; cases.len() * block_bytes];
        encode_via_descriptor(&desc, &src, &mut dst).unwrap();

        for (bi, c) in cases.iter().enumerate() {
            let blk = BlockQ4_0::quantize(c);
            let want: &[u8] = unsafe {
                std::slice::from_raw_parts((&blk as *const BlockQ4_0) as *const u8, block_bytes)
            };
            let got = &dst[bi * block_bytes..(bi + 1) * block_bytes];
            assert_eq!(
                got, want,
                "block {bi}: encode_via_descriptor != BlockQ4_0::quantize"
            );
        }

        // round-trip: encode → unpack == dequantize(quantize) (byte-exact 가 함의하나 명시 검증).
        let mut roundtrip = vec![0.0f32; QK4_0];
        unpack_block_via_descriptor(&desc, &dst[..block_bytes], &mut roundtrip);
        let mut want_rt = [0.0f32; QK4_0];
        BlockQ4_0::quantize(&cases[0]).dequantize(&mut want_rt);
        assert_eq!(roundtrip.as_slice(), want_rt.as_slice());

        // 비-canonical family 는 범위 한정 Err.
        let q4_1 = dtype_to_layout_desc(DType::Q4_1).unwrap();
        assert!(encode_via_descriptor(&q4_1, &src[..QK4_0], &mut dst[..20]).is_err());
    }

    // ── (a) generic unpacker bit-identical vs quant.rs dequantize ──

    /// generic unpacker 의 Q8_0 unpack == BlockQ8_0::dequantize(), 요소별 동일.
    #[test]
    fn test_generic_unpack_q8_0_bit_identical() {
        let desc = dtype_to_layout_desc(DType::Q8_0).unwrap();
        // 고정 + 의사난수 블록 여러 개.
        let cases: Vec<BlockQ8_0> = vec![
            BlockQ8_0 {
                d: f16::from_f32(0.0),
                qs: [0; QK8_0],
            },
            BlockQ8_0 {
                d: f16::from_f32(0.125),
                qs: std::array::from_fn(|i| (i as i32 - 16) as i8),
            },
            BlockQ8_0 {
                d: f16::from_f32(-3.5),
                qs: std::array::from_fn(|i| ((i as i32 * 7 + 3) % 256 - 128) as i8),
            },
            BlockQ8_0 {
                d: f16::from_f32(2.0e-3),
                qs: [127; QK8_0],
            },
            BlockQ8_0 {
                d: f16::from_f32(64.0),
                qs: [-128; QK8_0],
            },
        ];
        for blk in &cases {
            // reference
            let mut want = [0.0f32; QK8_0];
            blk.dequantize(&mut want);
            // generic
            let raw: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (blk as *const BlockQ8_0) as *const u8,
                    std::mem::size_of::<BlockQ8_0>(),
                )
            };
            let mut got = [0.0f32; QK8_0];
            unpack_block_via_descriptor(&desc, raw, &mut got);
            assert_eq!(got, want, "Q8_0 generic unpack mismatch");
        }
    }

    /// generic unpacker 의 Q4_0 unpack == BlockQ4_0::dequantize(), 요소별 동일.
    #[test]
    fn test_generic_unpack_q4_0_bit_identical() {
        let desc = dtype_to_layout_desc(DType::Q4_0).unwrap();
        let cases: Vec<BlockQ4_0> = vec![
            BlockQ4_0 {
                d: f16::from_f32(0.0),
                qs: [0; QK4_0 / 2],
            },
            BlockQ4_0 {
                d: f16::from_f32(0.5),
                qs: [0x55; QK4_0 / 2],
            },
            BlockQ4_0 {
                d: f16::from_f32(-1.25),
                qs: std::array::from_fn(|i| ((i * 13 + 1) % 256) as u8),
            },
            BlockQ4_0 {
                d: f16::from_f32(7.0),
                qs: [0xFF; QK4_0 / 2],
            },
            BlockQ4_0 {
                d: f16::from_f32(0.03125),
                qs: std::array::from_fn(|i| (i * 17) as u8),
            },
        ];
        for blk in &cases {
            let mut want = [0.0f32; QK4_0];
            blk.dequantize(&mut want);
            let raw: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (blk as *const BlockQ4_0) as *const u8,
                    std::mem::size_of::<BlockQ4_0>(),
                )
            };
            let mut got = [0.0f32; QK4_0];
            unpack_block_via_descriptor(&desc, raw, &mut got);
            assert_eq!(got, want, "Q4_0 generic unpack mismatch");
        }
    }

    // ── descriptor 어휘 sanity ──

    #[test]
    fn test_dtype_layout_desc_vocabulary() {
        assert_eq!(
            dtype_to_layout_desc(DType::Q8_0).unwrap(),
            KVLayoutDesc {
                block_elems: 32,
                bits: 8,
                scale_layout: ScaleLayout::PerBlockF16,
                packing: Packing::Byte,
            }
        );
        assert_eq!(
            dtype_to_layout_desc(DType::Q4_1).unwrap(),
            KVLayoutDesc {
                block_elems: 32,
                bits: 4,
                scale_layout: ScaleLayout::PerBlockF16WithMin,
                packing: Packing::Nibble,
            }
        );
        // matmul 부적합 dtype → floor 밖 escape.
        assert!(dtype_to_layout_desc(DType::U8).is_none());
    }
}
