//! GATE-C v2 — 런타임 `.so` dlopen format 레지스트리 (CF2).
//!
//! 설계 SSOT: `docs/adr/0009-gate-c-stage-dlopen-plugin.md`(D4 Format 축). 거울 =
//! `pressure/eviction/stage_registry.rs`(GATE-C v1 stage 축: `DYN_REGISTRY`/`register_dynamic_stages`/
//! `make_stage`/`DynStage`).
//!
//! 정적 `KV_FORMATS`(linkme, `builtin_kv_formats.rs` + 외부 crate)는 그대로 두고(D3 가산), dlopen 된
//! format plugin 을 별도 [`struct@DYN_FORMAT_REGISTRY`] 에 모은다. [`make_format`] 이 정적 우선 →
//! 동적 fallback 으로 source-agnostic `Box<dyn KVFormat>` 를 돌려준다. Stage 와 달리 `KVFormat` 은
//! 콜백 0(순수 descriptor)이라 host 측 어댑터([`DynFormat`])는 `layout` fn-ptr 호출 + `drop` 뿐이다.
//!
//! **purely additive (CF2), now wired** — 등록/조회 표면. production 소비자가 배선됨:
//! `make_format`(`--kv-format` 해석, `session/bin_setup`), `try_register_format`(`--load-plugin`,
//! `session/plugin_dispatch`). (`register_dynamic_formats`/`dynamic_registered_format_names` 는
//! 여전히 gate-test 전용.) dlopen 메커니즘은
//! `engine/tests/gate_c_format_dlopen_equivalence.rs`(CF4) 격리 테스트가 descriptor-identity 로 증명.

use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{Context, Result};
use argus_extension_api::{
    FormatExportAbi, FormatVTableAbi, KV_FORMAT_ABI_VERSION, KVFormat, KVLayoutDesc,
};
use core::ffi::c_void;

/// dlopen 된 한 format plugin 의 등록 항목. vtable 은 plugin `.so` 의 immutable static 을 가리킨다.
struct RuntimeFormatReg {
    name: String,
    vtable: *const FormatVTableAbi,
    /// `.so` 를 프로세스 수명 동안 유지(vtable/handle dangling 방지). drop 안 함.
    _lib: Arc<libloading::Library>,
}

// SAFETY: vtable 은 `.so` 의 immutable static 을 가리키고 `_lib`(Arc) 가 `.so` 를 살려 둔다.
// 읽기 전용 공유이므로 스레드 간 안전 — `DYN_FORMAT_REGISTRY`(static) 에 담기 위해 필요.
unsafe impl Send for RuntimeFormatReg {}
unsafe impl Sync for RuntimeFormatReg {}

/// 동적 format 등록 레지스트리 — init 시 append, construction 시 read. 정적 슬라이스와 **병합하지 않는다**(D3).
static DYN_FORMAT_REGISTRY: OnceLock<RwLock<Vec<RuntimeFormatReg>>> = OnceLock::new();

/// `--load-plugin` 으로 지정된 `.so` 들을 dlopen 해 [`struct@DYN_FORMAT_REGISTRY`] 에 등록한다(D4).
///
/// 각 `.so`: `Library::new`(RTLD_NOW) → `register_kv_format_v1` dlsym → `abi_version` 검사 →
/// 이름 충돌 fail-fast(빌트인 우선 + 동적 중복 금지). [`register_dynamic_stages`](crate::kv::eviction::stage_registry::register_dynamic_stages)
/// 의 format 축 짝(동형 strict 로더 — 심볼 부재 시 거부).
/// 이미 dlopen 된 `.so`(Arc) 에서 format capability 를 [`struct@DYN_FORMAT_REGISTRY`] 에 등록하는
/// per-`.so` 코어([`try_register_stage`](crate::kv::eviction::stage_registry::try_register_stage)
/// 의 format 축 짝). `register_kv_formats_v2` 봉투 entry 를 dlsym — **없으면 `Ok(0)`**(이 `.so` 는 format
/// 미보유). 있으면 봉투 `abi_version` 검사 → `count` 개 vtable **2-pass 원자 등록**(① 빌트인 충돌·봉투 내부
/// 중복 검사 → ② write-lock 1회 동적 중복 + 일괄 push). 반환 = 등록한 format 개수.
pub(crate) fn try_register_format(lib: &Arc<libloading::Library>, path: &Path) -> Result<usize> {
    // SAFETY: register_kv_formats_v2 dlsym. 부재 = 이 .so 가 format 축 미보유 → Ok(0)(에러 아님).
    let reg_fn: libloading::Symbol<unsafe extern "C" fn() -> FormatExportAbi> =
        match unsafe { lib.get(b"register_kv_formats_v2\0") } {
            Ok(f) => f,
            Err(_) => return Ok(0),
        };
    // SAFETY: 봉투 by-value 반환(sret). vtables 는 `.so` static 배열 base, abi_version 은 .so 단위 게이트.
    let export = unsafe { reg_fn() };
    if export.abi_version != KV_FORMAT_ABI_VERSION {
        anyhow::bail!(
            "plugin {}: format abi_version {} != expected {} (rebuild required)",
            path.display(),
            export.abi_version,
            KV_FORMAT_ABI_VERSION
        );
    }
    if export.count == 0 {
        return Ok(0);
    }
    if export.vtables.is_null() {
        anyhow::bail!(
            "plugin {}: register_kv_formats_v2 reports count {} but null vtables",
            path.display(),
            export.count
        );
    }
    let registry = DYN_FORMAT_REGISTRY.get_or_init(|| RwLock::new(Vec::new()));
    // ── pass 1: 이름 추출 + 빌트인 충돌(force-link synth_q4 포함) / 봉투 내부 중복 검사. ──
    let mut pending: Vec<(String, *const FormatVTableAbi)> = Vec::with_capacity(export.count);
    for i in 0..export.count {
        // SAFETY: vtables 는 `.so` static 배열 base, i < count. 봉투 스택과 무관(원소는 .so 수명).
        let vtable_ptr = unsafe { export.vtables.add(i) };
        let vtable = unsafe { &*vtable_ptr };
        let name = unsafe { CStr::from_ptr(vtable.name) }
            .to_str()
            .with_context(|| {
                format!(
                    "plugin {}: format name[{i}] is not valid UTF-8",
                    path.display()
                )
            })?
            .to_owned();
        if argus_extension_api::find_kv_format(&name).is_some() {
            anyhow::bail!(
                "plugin {}: format name '{}' conflicts with a builtin (builtin takes precedence, dynamic registration rejected)",
                path.display(),
                name
            );
        }
        if pending.iter().any(|(n, _)| *n == name) {
            anyhow::bail!(
                "plugin {}: format name '{}' is duplicated within the envelope",
                path.display(),
                name
            );
        }
        pending.push((name, vtable_ptr));
    }
    // ── pass 2: 동적 registry 중복 검사 + 일괄 push (write-lock 1회 = per-.so 원자). ──
    let mut w = registry
        .write()
        .expect("DYN_FORMAT_REGISTRY RwLock poisoned");
    for (name, _) in &pending {
        if w.iter().any(|r| r.name == *name) {
            anyhow::bail!(
                "plugin {}: format name '{}' is already dynamically registered (duplicate)",
                path.display(),
                name
            );
        }
    }
    let n = pending.len();
    for (name, vtable_ptr) in pending {
        w.push(RuntimeFormatReg {
            name,
            vtable: vtable_ptr,
            _lib: Arc::clone(lib),
        });
    }
    Ok(n)
}

/// `--load-plugin` 의 `.so` 들을 dlopen 해 format 만 등록하는 **strict batch 래퍼**(gate 테스트·축-격리 진단용).
/// 각 `.so` 가 format 0개면 "심볼 부재" bail(기존 계약 유지). production 혼합 로드는
/// [`register_dynamic_plugins`](crate::session::plugin_dispatch::register_dynamic_plugins) 사용.
pub fn register_dynamic_formats(paths: &[PathBuf]) -> Result<()> {
    for path in paths {
        // SAFETY: dlopen — 신뢰된 plugin 경로(사용자 명시 --load-plugin). RTLD_NOW 즉시 바인딩.
        let lib = Arc::new(
            unsafe { libloading::Library::new(path) }
                .with_context(|| format!("plugin dlopen failed: {}", path.display()))?,
        );
        if try_register_format(&lib, path)? == 0 {
            anyhow::bail!(
                "plugin {}: register_kv_formats_v2 symbol missing (or 0 formats)",
                path.display()
            );
        }
    }
    Ok(())
}

/// 동적으로 등록된 format 이름들(self-test / 진단용 — 정적 `registered_kv_format_names()` 의 동적 짝).
pub fn dynamic_registered_format_names() -> Vec<String> {
    DYN_FORMAT_REGISTRY
        .get()
        .map(|r| {
            r.read()
                .expect("DYN_FORMAT_REGISTRY RwLock poisoned")
                .iter()
                .map(|reg| reg.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// 이름으로 format 인스턴스를 만든다 — **정적 우선 → 동적 fallback**(D3). 호출부는 source 를 모른다.
/// 정적/동적 모두 miss 면 `None`(graceful unknown). `find_kv_format` 의 source-agnostic 짝.
pub fn make_format(name: &str) -> Option<Box<dyn KVFormat>> {
    // 1) 정적(linkme) 우선.
    if let Some(reg) = argus_extension_api::find_kv_format(name) {
        return Some((reg.make)());
    }
    // 2) 동적(dlopen) fallback.
    let registry = DYN_FORMAT_REGISTRY.get()?;
    let (vtable, lib) = {
        let guard = registry
            .read()
            .expect("DYN_FORMAT_REGISTRY RwLock poisoned");
        let reg = guard.iter().find(|r| r.name == name)?;
        (reg.vtable, Arc::clone(&reg._lib))
    };
    // SAFETY: vtable 는 `.so` static (lib 가 살려 둠). make 가 opaque plugin 핸들 반환.
    let handle = unsafe { ((*vtable).make)() };
    if handle.is_null() {
        eprintln!("[make_format] plugin '{name}' make returned a null handle");
        return None;
    }
    Some(Box::new(DynFormat {
        handle,
        vtable,
        _lib: lib,
    }))
}

/// 동적 plugin format 의 host 측 어댑터 — vtable 마샬링으로 [`KVFormat`] 을 구현(D4). Stage 의
/// `DynStage` 거울이나 콜백 0이라 `layout`(POD 반환) + `drop` 뿐.
struct DynFormat {
    handle: *mut c_void,
    vtable: *const FormatVTableAbi,
    _lib: Arc<libloading::Library>,
}

// SAFETY: 핸들은 plugin 의 `KVFormat`(trait 계약상 Send+Sync) 인스턴스, vtable 불변, lib Arc 유지.
unsafe impl Send for DynFormat {}
unsafe impl Sync for DynFormat {}

impl Drop for DynFormat {
    fn drop(&mut self) {
        // SAFETY: handle 은 make 가 만든 plugin 인스턴스, 정확히 1회 해제.
        unsafe { ((*self.vtable).drop)(self.handle) };
    }
}

impl KVFormat for DynFormat {
    fn name(&self) -> &str {
        // SAFETY: vtable.name 은 plugin `.so` 의 'static null-종단 str (lib 가 살려 둠).
        unsafe { CStr::from_ptr((*self.vtable).name) }
            .to_str()
            .unwrap_or("<plugin>")
    }

    fn layout(&self) -> KVLayoutDesc {
        // SAFETY: handle = make 가 만든 인스턴스. layout 은 KVLayoutDesc POD 를 값으로 반환(arena 불요).
        unsafe { ((*self.vtable).layout)(self.handle) }
    }
}
