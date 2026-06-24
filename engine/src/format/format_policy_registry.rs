//! format-policy 축(`KVFormatPolicy`) 빌트인 등록 + force-link self-test.
//!
//! read 축의 `kv/read/read_stage_registry.rs::ensure_builtin_read_stages_registered` /
//! format 축의 `format/builtin_kv_formats.rs::ensure_builtin_kv_formats_registered` 거울. 이 축은
//! per-layer KV 정밀도(N-way mixed precision)를 구성-시점에 산출하는 생산자다 — 엔진은
//! `find_format_policy(name)` 로 해소하고, `bin_setup` 의 `--kv-format <policy>` arm 이 소비한다.

use anyhow::Result;

// mixed_precision production 빌트인 force-link. dep 선언만으로는 미참조 rlib 이 링크 제외돼
// `#[distributed_slice(KV_FORMAT_POLICIES)]` 등록이 누락된다. 이 1줄이 production 바이너리에서
// `find_format_policy("mixed_precision")` 를 가시화한다(quest/synth_q4_format 패턴).
use mixed_precision as _;

/// 빌트인 format policy(mixed_precision)가 `KV_FORMAT_POLICIES` 에 등록됐는지 단언한다 — KV cache
/// 구성 진입 시 1회 호출. fat-LTO `--gc-sections` 가 linkme 등록을 silent drop 하면 `Err` 로
/// fail-fast 한다(release 에서 `--kv-format mixed_precision` 미해석 → single-format arm 으로 조용한
/// 폴백 방지).
pub fn ensure_builtin_format_policies_registered() -> Result<()> {
    for name in ["mixed_precision"] {
        if argus_extension_api::find_format_policy(name).is_none() {
            anyhow::bail!(
                "내장 KVFormatPolicy '{name}' 미등록 — linkme fat-LTO --gc-sections silent drop 의심\
                 . mixed-precision crate 의 #[distributed_slice(KV_FORMAT_POLICIES)] 등록이 \
                 링크되지 않음."
            );
        }
    }
    // policy-shadows-format precedence 경계화: `bin_setup` dispatch 는 format 이름보다 policy 이름을
    // 먼저 조회하므로(`find_format_policy` first), policy 이름이 내장/등록 format 이름과 충돌하면 그
    // format 을 silent shadow 한다. 시작 시 disjointness 를 단언해 그 위험을 차단한다(현재는 안전 —
    // "mixed_precision" 은 f32/f16/q4_0/q8_0/synth_q4 와 disjoint).
    for name in argus_extension_api::registered_format_policy_names() {
        if crate::format::builtin_format_dtype(name).is_some()
            || argus_extension_api::find_kv_format(name).is_some()
        {
            anyhow::bail!(
                "KVFormatPolicy '{name}' 가 등록된 KV format 이름과 충돌 — --kv-format dispatch 에서 \
                 policy 가 format 을 가린다(bin_setup). policy 이름을 바꿀 것."
            );
        }
    }
    Ok(())
}

/// `name` 이 등록된 [`KVFormatPolicy`](argus_extension_api::KVFormatPolicy)(per-layer mixed precision)
/// 로 해소되는지 여부. `--kv-format` policy 를 honor 하지 않는 bin(eval/chat)이 uniform 캐시를 조용히
/// 할당하는 대신 fail-fast 하도록 가드하는 데 쓴다.
pub fn is_registered_kv_format_policy(name: &str) -> bool {
    argus_extension_api::find_format_policy(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_builtin_format_policies_ok() {
        ensure_builtin_format_policies_registered()
            .expect("빌트인 format policy(mixed_precision) 등록되어야 함");
    }

    #[test]
    fn mixed_precision_resolvable_by_name() {
        let reg = argus_extension_api::find_format_policy("mixed_precision")
            .expect("mixed_precision 등록 검색 가능");
        assert_eq!(reg.name, "mixed_precision");
    }
}
