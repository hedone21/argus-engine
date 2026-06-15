//! 합성 KV format `synth_q4` — DType variant 없이 외부 crate(`argus-extension-api` 만 의존)가
//! [`KV_FORMATS`] 에 format 을 더하는 GATE-B 증거(`.so` 대역, format 축).
//!
//! layout 은 q4_0 와 동일(block_elems:32 / bits:4 / Nibble / PerBlockF16)하나, 엔진에 대응
//! `DType` variant 가 **없다**(`dtype_to_layout_desc("synth_q4")` 같은 경로 부재). 따라서 엔진은
//! 이 format 을 opaque(`OpaqueBuffer` + `dequant_via_descriptor` floor + `encode_via_descriptor`)
//! 로만 처리한다 — closed `DType` enum 을 우회한 format 확장의 실증.
//!
//! 본 crate 는 [`example-keep-recent`](../example-keep-recent)(stage 축)의 format 축 짝이다:
//! 엔진 타입(`KVCache`/`Backend`/`Buffer`)을 일절 참조하지 않고 descriptor(데이터)만 기여한다
//! (compute 는 엔진이 floor 로 소유).

use argus_extension_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

/// `synth_q4` format — q4_0 layout 의 descriptor 만 제공(name + layout, 2-method).
struct SynthQ4;

impl KVFormat for SynthQ4 {
    fn name(&self) -> &str {
        "synth_q4"
    }

    fn layout(&self) -> KVLayoutDesc {
        KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        }
    }
}

// 등록(dual-wiring) — 정적: linkme `KV_FORMATS`(엔진이 `find_kv_format("synth_q4")` 로
// 발견, DType variant 불요). 동적(`--features plugin-cdylib`): `register_kv_format_v1` C-ABI export
// (host 가 dlopen). 한 줄로 양쪽.
argus_extension_api::register_kv_format!("synth_q4", || Box::new(SynthQ4));
// GATE-C v2: `.so` 엔트리(register_kv_formats_v2) emit. plugin-cdylib 게이트 — 엔진
// force-link(feature OFF) 빌드엔 미emit(심볼 충돌 차단). synth_q4 는 엔진 force-link 정적 등록이라
// dlopen 시 builtin-collision reject 대상(게이트 vehicle).
argus_extension_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::find_kv_format;

    #[test]
    fn synth_q4_registers_into_kv_formats() {
        let reg = find_kv_format("synth_q4").expect("synth_q4 등록이 KV_FORMATS 에 있어야 한다");
        assert_eq!(reg.name, "synth_q4");
        let fmt = (reg.make)();
        assert_eq!(fmt.name(), "synth_q4");
        // q4_0 와 동일 layout 이나 대응 DType variant 는 없다(opaque 경로 강제).
        let l = fmt.layout();
        assert_eq!(l.block_elems, 32);
        assert_eq!(l.bits, 4);
        assert_eq!(l.scale_layout, ScaleLayout::PerBlockF16);
        assert_eq!(l.packing, Packing::Nibble);
        // byte-회계도 q4_0 와 동일(18 bytes / 32 elems).
        assert_eq!(l.bytes_for_elems(32), Some(18));
    }
}
