//! KV format `q2_0` — 비대칭 2-bit block-quant(writable sub-Q4 floor, W-CODEC slice 1).
//!
//! descriptor = `block_elems:32 / bits:2 / PerBlockF16WithMin / Quad`(4 elems/byte) = 12 bytes/block,
//! `argus_kv_codec::BlockQ2_0`(d=(max−min)/3, m=min) 와 **byte-identical**. `synth_q4` 처럼 plugin 은
//! descriptor(name + layout)만 기여하고 compute 는 엔진의 generic descriptor floor 가 소유한다
//! (`unpack_block_via_descriptor` 의 Quad arm + `encode_via_descriptor` 의 Quad 인코더). 대응 `DType`
//! variant(`Q2_0`)가 **없으므로** 엔진은 opaque(`OpaqueBuffer` + dequant→f32 floor)로 저장한다 —
//! closed `DType` enum 을 우회한 format 확장으로, [`example-keep-recent`](../example-keep-recent)
//! (stage 축)의 format 축 짝이다(엔진 타입 0 참조, descriptor 데이터만 기여).

use argus_extension_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

/// `q2_0` format — 비대칭 2-bit descriptor 만 제공(name + layout, 2-method).
struct Q2Format;

impl KVFormat for Q2Format {
    fn name(&self) -> &str {
        "q2_0"
    }

    fn layout(&self) -> KVLayoutDesc {
        KVLayoutDesc {
            block_elems: 32,
            bits: 2,
            scale_layout: ScaleLayout::PerBlockF16WithMin,
            packing: Packing::Quad,
        }
    }
}

// 등록(dual-wiring) — 정적: linkme `KV_FORMATS`(엔진이 `find_kv_format("q2_0")` 로 발견, DType
// variant 불요). 동적(`--features plugin-cdylib`): `register_kv_format_v1` C-ABI export(host dlopen).
argus_extension_api::register_kv_format!("q2_0", || Box::new(Q2Format));
// `.so` 엔트리(register_kv_formats_v2) emit. plugin-cdylib 게이트 — 엔진 force-link(feature OFF)
// 빌드엔 미emit(심볼 충돌 차단). q2_0 는 엔진 force-link 정적 등록이라 dlopen 시 builtin-collision
// reject 대상.
argus_extension_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::find_kv_format;

    #[test]
    fn q2_0_registers_into_kv_formats() {
        let reg = find_kv_format("q2_0").expect("q2_0 등록이 KV_FORMATS 에 있어야 한다");
        assert_eq!(reg.name, "q2_0");
        let fmt = (reg.make)();
        assert_eq!(fmt.name(), "q2_0");
        // 비대칭 2-bit layout. 대응 DType variant 는 없다(opaque 경로 강제).
        let l = fmt.layout();
        assert_eq!(l.block_elems, 32);
        assert_eq!(l.bits, 2);
        assert_eq!(l.scale_layout, ScaleLayout::PerBlockF16WithMin);
        assert_eq!(l.packing, Packing::Quad);
        // byte-회계 = size_of::<BlockQ2_0>() (scale 2 + min 2 + 8 quad bytes = 12 / 32 elems).
        assert_eq!(l.bytes_for_elems(32), Some(12));
    }
}
