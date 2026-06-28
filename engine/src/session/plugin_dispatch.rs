//! GATE-C 멀티-vtable bundle dispatcher — `--load-plugin` 의 `.so` 들을 **`.so` 당 1회
//! dlopen** 하고 present 한 **모든 축** capability 를 등록한다(open-once + `Arc<Library>` 공유).
//!
//! 한 plugin `.so` 는 format(`register_kv_formats_v2`) ⊥ backend-cap(`register_backend_caps_v2`)
//! 엔트리를 **부분집합**으로 export 할 수 있다(번들). 본 dispatcher 는 두 축 `try_register_*` 를 모두
//! 호출해, 단일축 `.so` 든 번들 `.so` 든 한 번의 dlopen 으로 흡수한다. 같은 `.so` 의 format-reg·
//! backend-reg 는 동일 `Arc<Library>` 를 공유(정직한 단일 핸들). (stage 축은 static-linkme 전용 —
//! `.so` 로 KV mutation stage 를 동적 등록하는 경로는 없다.)
//!
//! **registry 병합 없음**: format 은 `DYN_FORMAT_REGISTRY`, backend-cap 은 별도 registry 로 분리
//! 적재된다. dispatcher 는 라우팅만 한다. **wrong-type 은 reject 아님**: 단일축 `.so` 는
//! 번들의 부분집합으로 정당 — 어느 축에도 기여 0 인 `.so` 만 capability-0 로 bail.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

/// `--load-plugin` 의 `.so` 들을 dlopen 1회 후 present 한 모든 축에 라우팅 등록한다(production 단일 진입).
///
/// 각 `.so`: `Library::new`(RTLD_NOW) → `try_register_format` + `try_register_backend_cap`(둘 다
/// `Arc<Library>` 공유) → 등록 합이 0 이면 capability-0 bail(`export_plugin!` 누락 또는 빈 plugin). 각
/// `try_register_*` 의 봉투 abi_version·이름 충돌·중복은 그 안에서 fail-fast(2-pass 원자).
pub fn register_dynamic_plugins(paths: &[PathBuf]) -> Result<()> {
    for path in paths {
        // SAFETY: dlopen — 사용자 명시 신뢰 경로. RTLD_NOW 즉시 바인딩. 핸들은 try_register_* 가
        // Arc::clone 으로 각 registry 에 보관해 `.so` 를 프로세스 수명 동안 유지.
        let lib = Arc::new(
            unsafe { libloading::Library::new(path) }
                .with_context(|| format!("plugin dlopen failed: {}", path.display()))?,
        );
        let formats = crate::format::dynamic_format_registry::try_register_format(&lib, path)?;
        let backends =
            crate::capability::dynamic_backend_registry::try_register_backend_cap(&lib, path)?;
        if formats + backends == 0 {
            anyhow::bail!(
                "plugin {}: 0 capabilities registered (register_kv_formats_v2·register_backend_caps_v2 absent — export_plugin! missing?)",
                path.display()
            );
        }
    }
    Ok(())
}
