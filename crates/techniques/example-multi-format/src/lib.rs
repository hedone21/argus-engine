//! 멀티-format 예제 plugin — 한 `.so` 에 **서로 다른 descriptor** 의 format 2종(quant 패밀리).
//!
//! `register_kv_format!` 를 2회 호출(const-block 격리로 누적) + `export_plugin!()` 1회 → `.so` 의
//! `register_kv_formats_v2` 봉투가 vtable 2개를 신고한다. 두 format 의 descriptor 가 **다르므로**
//! (mf_q4 = Nibble/4-bit, mf_q8 = Byte/8-bit) 인덱스-swap·이름↔vtable 미스바인딩을 게이트가 검출할 수
//! 있다(동일 desc N개는 위양성 통과).
//!
//! 엔진(argus_engine)·다른 축 crate 에 의존하지 않고 descriptor(데이터)만 기여. 엔진 force-link
//! 안 함 → 동적 등록 성공 경로 vehicle.

use argus_extension_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

/// q4_0-like(Nibble, 4-bit) — encode_via_descriptor 가 지원하는 q4_0 canonical(floor round-trip 가능).
struct MfQ4;
impl KVFormat for MfQ4 {
    fn name(&self) -> &str {
        "mf_q4"
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

/// q8_0-like(Byte, 8-bit) — mf_q4 와 **다른 descriptor**(packing/bits 상이). descriptor-identity 전용.
struct MfQ8;
impl KVFormat for MfQ8 {
    fn name(&self) -> &str {
        "mf_q8"
    }
    fn layout(&self) -> KVLayoutDesc {
        KVLayoutDesc {
            block_elems: 32,
            bits: 8,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Byte,
        }
    }
}

// 한 crate(=한 `.so`)에 format 2종 — const-block 격리로 다회 호출 누적.
argus_extension_api::register_kv_format!("mf_q4", || Box::new(MfQ4));
argus_extension_api::register_kv_format!("mf_q8", || Box::new(MfQ8));
argus_extension_api::export_plugin!();

#[cfg(test)]
mod tests {
    use argus_extension_api::{Packing, find_kv_format};

    #[test]
    fn both_formats_register_with_distinct_descriptors() {
        let q4 = (find_kv_format("mf_q4").expect("mf_q4 등록").make)().layout();
        let q8 = (find_kv_format("mf_q8").expect("mf_q8 등록").make)().layout();
        assert_eq!(q4.bits, 4);
        assert_eq!(q4.packing, Packing::Nibble);
        assert_eq!(q8.bits, 8);
        assert_eq!(q8.packing, Packing::Byte);
        assert_ne!(
            q4, q8,
            "두 format 의 descriptor 는 서로 달라야 한다(미스바인딩 검출)"
        );
    }
}
