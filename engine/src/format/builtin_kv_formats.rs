//! 내장 KV format descriptor plugin 등록.
//!
//! 설계 SSOT: `docs/adr/0005-format-backend-capability-plugin-unification.md` D3(format
//! plugin = 순수 descriptor) / D6(3축 평행 linkme registry). 거울 = `pressure/eviction/
//! stage_registry.rs`(KV_MUTATION_STAGES 빌트인 등록 + `ensure_builtin_stages_registered`).
//!
//! format 축의 plugin 표면은 `argus_extension_api::KVFormat`(name + layout, 버퍼 0)이다. 여기서
//! 그 표면의 **내장 멤버**(f32/f16/q4_0/q8_0)를 `#[distributed_slice(KV_FORMATS)]` 로 등록한다.
//! descriptor 는 [`dtype_to_layout_desc`] 에서 도출해 단일 진실원천을 공유한다(drift 0).
//!
//! **purely additive (S4-1), now wired** — force-link 호출 `ensure_builtin_kv_formats_registered`
//! 이 startup 에 배선됨(`session/bin_setup`/`eval_setup`); `builtin_format_dtype` 는 `--kv-format`
//! typed 라우터(`session/bin_setup`)가 사용한다. (compute→backend descriptor dispatch 소비는
//! 후속 substep.)

use anyhow::Result;
use argus_extension_api::{KV_FORMATS, KVFormat, KVFormatReg, KVLayoutDesc};
use linkme::distributed_slice;

use crate::buffer::DType;
use crate::format::dtype_layout::dtype_to_layout_desc;

// force-link: 외부 `synth-q4-format` crate(opaque .so format `synth_q4`)가
// `KV_FORMATS` 에 등록되도록 강제 링크한다. dep 선언만으론 dead-crate elision 으로 미링크
// (M3/G5 교훈). `--kv-format synth_q4` 가 `find_kv_format` 으로 이 등록을 찾는다.
use synth_q4_format as _;
// force-link: 외부 `q2-format` crate(opaque format `q2_0`, 비대칭 2-bit)가 `KV_FORMATS` 에
// 등록되도록 강제 링크한다(같은 dead-crate elision 이유). `--kv-format q2_0` 가 이 등록을 찾는다.
use q2_format as _;

/// Stateless descriptor-only `KVFormat` plugin (plugin 은 버퍼 0, descriptor 만).
///
/// 엔진의 버퍼 보유/compute 는 `StandardFormat`(`pressure/standard_format.rs`)이 계속 쥔다
/// (bridge: S4-3). 본 타입은 `KVFormatReg.make` 가 만드는 순수 layout 기술자다.
struct StandardKvFormat {
    name: &'static str,
    desc: KVLayoutDesc,
}

impl KVFormat for StandardKvFormat {
    fn name(&self) -> &str {
        self.name
    }
    fn layout(&self) -> KVLayoutDesc {
        self.desc
    }
}

/// 내장 format 의 descriptor 도출 — [`dtype_to_layout_desc`] 단일 원천(drift 방지).
///
/// 내장 4종(f32/f16/q4_0/q8_0)은 전부 block-quant family/raw 어휘에 속해 항상 `Some`.
fn std_desc(d: DType) -> KVLayoutDesc {
    dtype_to_layout_desc(d).expect("builtin KVFormat dtype always has a descriptor")
}

#[distributed_slice(KV_FORMATS)]
static F32_KV_FORMAT: KVFormatReg = KVFormatReg {
    name: "f32",
    make: || {
        Box::new(StandardKvFormat {
            name: "f32",
            desc: std_desc(DType::F32),
        })
    },
};

#[distributed_slice(KV_FORMATS)]
static F16_KV_FORMAT: KVFormatReg = KVFormatReg {
    name: "f16",
    make: || {
        Box::new(StandardKvFormat {
            name: "f16",
            desc: std_desc(DType::F16),
        })
    },
};

#[distributed_slice(KV_FORMATS)]
static Q4_0_KV_FORMAT: KVFormatReg = KVFormatReg {
    name: "q4_0",
    make: || {
        Box::new(StandardKvFormat {
            name: "q4_0",
            desc: std_desc(DType::Q4_0),
        })
    },
};

#[distributed_slice(KV_FORMATS)]
static Q8_0_KV_FORMAT: KVFormatReg = KVFormatReg {
    name: "q8_0",
    make: || {
        Box::new(StandardKvFormat {
            name: "q8_0",
            desc: std_desc(DType::Q8_0),
        })
    },
};

/// 내장 KV format(f32/f16/q4_0/q8_0)이 `KV_FORMATS` 에 등록됐는지 단언한다 — format 축 소비자
/// 진입 시 1회 호출(`ensure_builtin_stages_registered` 거울).
///
/// fat-LTO `--gc-sections` 가 linkme 등록을 silent drop 하면 누락 format 에 대해 `Err` 로
/// fail-fast 한다(release 에서 format 이름 미해석 → 조용한 폴백 방지). startup 에서 호출됨
/// (`session/bin_setup`/`eval_setup`).
pub fn ensure_builtin_kv_formats_registered() -> Result<()> {
    for name in ["f32", "f16", "q4_0", "q8_0"] {
        if argus_extension_api::find_kv_format(name).is_none() {
            anyhow::bail!(
                "builtin KVFormat '{name}' not registered — suspected linkme fat-LTO --gc-sections silent drop\
                 . the #[distributed_slice] registration in builtin_kv_formats is \
                 not linked."
            );
        }
    }
    Ok(())
}

/// 내장 format 이름 → 대응 `DType`. 내장 4종(f32/f16/q4_0/q8_0)만 `Some`;
/// 그 외 등록 format(.so/opaque, 예 `synth_q4`)은 `None` → opaque 저장 경로.
///
/// **layout 이 아니라 이름(identity)으로 가른다** — `synth_q4` 는 `q4_0` 와 동일 layout 이지만
/// 대응 `DType` 가 없어 opaque 로 처리된다. 미래 `.so` plugin format 도 이 "name→no-DType→opaque"
/// 경로를 친다(GATE-C).
pub fn builtin_format_dtype(name: &str) -> Option<DType> {
    match name {
        "f32" => Some(DType::F32),
        "f16" => Some(DType::F16),
        "q4_0" => Some(DType::Q4_0),
        "q8_0" => Some(DType::Q8_0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{find_kv_format, registered_kv_format_names};

    /// 4종 등록 + 각 등록 descriptor 가 `dtype_to_layout_desc` 와 일치(단일 원천 round-trip).
    #[test]
    fn builtin_kv_formats_registered_and_descriptor_matches_dtype_layout() {
        // self-test fn (force-link 게이트) 통과.
        ensure_builtin_kv_formats_registered().expect("내장 KV format 4종 등록되어야 함");

        let cases = [
            ("f32", DType::F32),
            ("f16", DType::F16),
            ("q4_0", DType::Q4_0),
            ("q8_0", DType::Q8_0),
        ];
        for (name, dt) in cases {
            let reg = find_kv_format(name).unwrap_or_else(|| panic!("{name} 등록되어야 함"));
            let fmt = (reg.make)();
            assert_eq!(fmt.name(), name, "make() 가 만든 format 이름 == 등록 이름");
            assert_eq!(
                fmt.layout(),
                dtype_to_layout_desc(dt).unwrap(),
                "{name} descriptor 는 dtype_to_layout_desc 와 일치해야 함(단일 원천)"
            );
        }

        // 등록 이름 목록에 4종 전부 존재.
        let names = registered_kv_format_names();
        for name in ["f32", "f16", "q4_0", "q8_0"] {
            assert!(
                names.contains(&name),
                "registered_kv_format_names 에 {name} 존재"
            );
        }
    }
}
