//! capability-0 vehicle — `register_kv_format!` 는 호출하되 [`export_plugin!`](technique_api::export_plugin)
//! 를 **고의로 누락**한다. plugin-cdylib 빌드 시 `PLUGIN_KV_FORMAT_VTABLES` 슬라이스
//! 기여는 있으나 `register_kv_formats_v2`/`register_kv_stages_v2` entry 심볼이 없다 → host
//! `register_dynamic_plugins` 가 두 축 dlsym 모두 실패 → `n==0` → **capability-0 reject**.
//!
//! 작성자가 `export_plugin!()` 를 잊은 현실적 실수를 fail-fast 로 잡는지 검증하는 vehicle. 엔진
//! force-link 안 함.

use technique_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

struct NoExportFmt;
impl KVFormat for NoExportFmt {
    fn name(&self) -> &str {
        "no_export_fmt"
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

// register_kv_format! 만 호출 — export_plugin!() 는 의도적으로 없음(capability-0 유발).
technique_api::register_kv_format!("no_export_fmt", || Box::new(NoExportFmt));

#[cfg(test)]
mod tests {
    use technique_api::find_kv_format;

    #[test]
    fn registers_statically_but_has_no_v2_entry() {
        // 정적 등록은 됨(KV_FORMATS) — 동적 entry(register_kv_formats_v2) 부재는 .so dlsym 으로만 확인 가능.
        assert!(find_kv_format("no_export_fmt").is_some());
    }
}
