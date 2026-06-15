//! read 축(`KVReadStage`) 빌트인 등록 + force-link self-test.
//!
//! eviction 축의 `kv/eviction/stage_registry.rs::ensure_builtin_stages_registered` /
//! format 축의 `format/builtin_kv_formats.rs::ensure_builtin_kv_formats_registered` 거울.

use anyhow::Result;

// Quest production 빌트인 force-link. dep 선언만으로는 미참조 rlib 이 링크 제외돼
// `#[distributed_slice(KV_READ_STAGES)]` 등록이 누락된다. 이 1줄이 production
// 바이너리에서 `find_read_stage("quest")` 를 가시화한다. caote(feature opt-in)와 달리 Quest 는
// read 축 첫 빌트인이라 비-optional force-link(synth_q4_format 패턴).
use quest as _;

/// 빌트인 read stage(Quest)가 `KV_READ_STAGES` 에 등록됐는지 단언한다 — read stage 구성/CLI 파싱
/// 진입 시 1회 호출. fat-LTO `--gc-sections` 가 linkme 등록을 silent drop 하면
/// `Err` 로 fail-fast 한다(release 에서 `--read-stage quest` 미해석 → 조용한 폴백 방지).
pub fn ensure_builtin_read_stages_registered() -> Result<()> {
    for name in ["quest"] {
        if argus_extension_api::find_read_stage(name).is_none() {
            anyhow::bail!(
                "내장 KVReadStage '{name}' 미등록 — linkme fat-LTO --gc-sections silent drop 의심\
                 . quest crate 의 #[distributed_slice(KV_READ_STAGES)] 등록이 \
                 링크되지 않음."
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_builtin_read_stages_ok() {
        ensure_builtin_read_stages_registered().expect("빌트인 read stage(quest) 등록되어야 함");
    }

    #[test]
    fn quest_resolvable_by_name() {
        let reg = argus_extension_api::find_read_stage("quest").expect("quest 등록 검색 가능");
        assert_eq!(reg.name, "quest");
    }
}
