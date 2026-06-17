//! GATE-C v3 — 런타임 `.so` dlopen backend-capability 레지스트리 (design D2/D7/D8, 봉투 / CB2).
//!
//! 거울 = `format/dynamic_format_registry.rs`(Format 축 CF2). 정적 `QUANT_ATTN_REGS`(linkme,
//! `register_quant_attn_plugin!` 기여)는 그대로 두고(D3 가산), dlopen 된 backend-cap plugin 을
//! 별도 [`struct@DYN_BACKEND_REGISTRY`] 에 모은다. [`resolve_quant_attn_capability`] 가 정적 우선 → 동적
//! fallback 으로 source-agnostic `Arc<dyn QuantAttnBackend>` 를 만든다.
//!
//! **category 다리(D7)**: 동적 엔트리는 얇은 [`BackendCapVTableAbi`]`{name, category, vtable}` 태그드
//! 포인터. host 가 `category` 로 `vtable` 를 카테고리별 테이블([`QuantAttnVTable`])로 캐스팅한다. 알려진
//! 카테고리 1개당 `match` arm 1개 — 새 카테고리는 arm 추가 = host 재컴파일(C1).
//!
//! **단일 trait(D8) 결과**: [`QuantAttnBackend`] 가 이미 ABI-shaped(cl_mem [`QuantAttnArgs`])라 host
//! 어댑터 [`struct@DynQuantAttnBackend`] 는 args 를 vtable fn-ptr 로 그대로 전달할 뿐 — `&Tensor` 다리는
//! 소비자(`kivi_format`/`kivi_cache`)가 한 번 수행한다(static·dynamic 공유).

use std::ffi::CStr;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{Context, Result};
use argus_extension_api::{
    BACKEND_CAP_ABI_VERSION, BACKEND_CAP_CATEGORY_ATTENTION, BackendCapExportAbi, QuantAttnArgs,
    QuantAttnBackend, QuantAttnGatherArgs, QuantAttnMakeArgs, QuantAttnVTable,
};
use core::ffi::c_void;

/// dlopen 된 한 backend-cap 의 등록 항목. 태그드 엔트리는 plugin `.so` 의 immutable static 을 가리킨다.
struct RuntimeBackendCapReg {
    name: String,
    /// 카테고리 태그([`BACKEND_CAP_CATEGORY_ATTENTION`] 등) — 다리 `match` 키.
    category: u32,
    /// category 별 테이블 포인터(`BackendCapVTableAbi.vtable`). 예: `*const QuantAttnVTable`.
    cat_vtable: *const c_void,
    /// `.so` 를 프로세스 수명 동안 유지(vtable/handle dangling 방지). drop 안 함.
    _lib: Arc<libloading::Library>,
}

// SAFETY: 태그드 엔트리/테이블은 `.so` 의 immutable static 을 가리키고 `_lib`(Arc)가 `.so` 를 살려 둔다.
// 읽기 전용 공유라 스레드 간 안전 — `DYN_BACKEND_REGISTRY`(static) 에 담기 위해 필요.
unsafe impl Send for RuntimeBackendCapReg {}
unsafe impl Sync for RuntimeBackendCapReg {}

/// 동적 backend-cap 등록 레지스트리 — init 시 append, construction 시 read. 정적 슬라이스와 **병합 없음**(D3/D6).
static DYN_BACKEND_REGISTRY: OnceLock<RwLock<Vec<RuntimeBackendCapReg>>> = OnceLock::new();

/// 이미 dlopen 된 `.so`(Arc)에서 backend-cap 을 [`struct@DYN_BACKEND_REGISTRY`] 에 등록하는 per-`.so` 코어
/// (`try_register_format` 의 backend 축 짝). `register_backend_caps_v2` 봉투 entry 를 dlsym — **없으면 `Ok(0)`**
/// (이 `.so` 는 backend-cap 미보유). 있으면 봉투 `abi_version` 검사 → `count` 개 엔트리 **2-pass 원자 등록**
/// (① category·이름 추출 + 빌트인 충돌·봉투 내부 중복 → ② write-lock 1회 동적 중복 + 일괄 push). 반환 = 등록 개수.
pub(crate) fn try_register_backend_cap(
    lib: &Arc<libloading::Library>,
    path: &Path,
) -> Result<usize> {
    // SAFETY: register_backend_caps_v2 dlsym. 부재 = 이 .so 가 backend-cap 축 미보유 → Ok(0)(에러 아님).
    let reg_fn: libloading::Symbol<unsafe extern "C" fn() -> BackendCapExportAbi> =
        match unsafe { lib.get(b"register_backend_caps_v2\0") } {
            Ok(f) => f,
            Err(_) => return Ok(0),
        };
    // SAFETY: 봉투 by-value 반환(sret). vtables 는 `.so` static 배열 base, abi_version 은 .so 단위 게이트.
    let export = unsafe { reg_fn() };
    if export.abi_version != BACKEND_CAP_ABI_VERSION {
        anyhow::bail!(
            "plugin {}: backend-cap abi_version {} != expected {} (rebuild required)",
            path.display(),
            export.abi_version,
            BACKEND_CAP_ABI_VERSION
        );
    }
    if export.count == 0 {
        return Ok(0);
    }
    if export.vtables.is_null() {
        anyhow::bail!(
            "plugin {}: register_backend_caps_v2 reports count {} but null vtables",
            path.display(),
            export.count
        );
    }
    let registry = DYN_BACKEND_REGISTRY.get_or_init(|| RwLock::new(Vec::new()));
    // ── pass 1: 이름 추출 + category 검증(알려진 것만) + 빌트인(정적 QUANT_ATTN_REGS) 충돌 / 봉투 내부 중복. ──
    let mut pending: Vec<(String, u32, *const c_void)> = Vec::with_capacity(export.count);
    for i in 0..export.count {
        // SAFETY: vtables 는 `.so` static 배열 base, i < count.
        let entry_ptr = unsafe { export.vtables.add(i) };
        let entry = unsafe { &*entry_ptr };
        let name = unsafe { CStr::from_ptr(entry.name) }
            .to_str()
            .with_context(|| {
                format!(
                    "plugin {}: backend-cap name[{i}] is not valid UTF-8",
                    path.display()
                )
            })?
            .to_owned();
        // 미지의 category = host 가 다리 arm 을 모름 → 거부(C1: 새 카테고리는 host 재컴파일).
        if entry.category != BACKEND_CAP_CATEGORY_ATTENTION {
            anyhow::bail!(
                "plugin {}: backend-cap '{}' has unsupported category {} (not a category known to host — C1)",
                path.display(),
                name,
                entry.category
            );
        }
        if entry.vtable.is_null() {
            anyhow::bail!(
                "plugin {}: backend-cap '{}' has a null category vtable",
                path.display(),
                name
            );
        }
        if argus_extension_api::find_quant_attn(&name).is_some() {
            anyhow::bail!(
                "plugin {}: backend-cap name '{}' collides with a built-in (built-in takes priority, dynamic registration rejected)",
                path.display(),
                name
            );
        }
        if pending.iter().any(|(n, _, _)| *n == name) {
            anyhow::bail!(
                "plugin {}: backend-cap name '{}' is duplicated within the envelope",
                path.display(),
                name
            );
        }
        pending.push((name, entry.category, entry.vtable));
    }
    // ── pass 2: 동적 registry 중복 검사 + 일괄 push (write-lock 1회 = per-.so 원자). ──
    let mut w = registry
        .write()
        .expect("DYN_BACKEND_REGISTRY RwLock poisoned");
    for (name, _, _) in &pending {
        if w.iter().any(|r| r.name == *name) {
            anyhow::bail!(
                "plugin {}: backend-cap name '{}' is already dynamically registered (duplicate)",
                path.display(),
                name
            );
        }
    }
    let n = pending.len();
    for (name, category, cat_vtable) in pending {
        w.push(RuntimeBackendCapReg {
            name,
            category,
            cat_vtable,
            _lib: Arc::clone(lib),
        });
    }
    Ok(n)
}

/// 동적으로 등록된 backend-cap 이름들(self-test / 진단용).
pub fn dynamic_registered_backend_cap_names() -> Vec<String> {
    DYN_BACKEND_REGISTRY
        .get()
        .map(|r| {
            r.read()
                .expect("DYN_BACKEND_REGISTRY RwLock poisoned")
                .iter()
                .map(|reg| reg.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// 이름으로 KIVI ATTENTION capability 인스턴스를 만든다 — **정적 우선 → 동적 fallback**(D3, category 다리 D7).
/// `make_args`(host GPU context)로 커널을 1회 빌드한다(D4). 정적/동적 모두 miss 면 `None`(graceful unknown).
/// host 의 `--kivi-impl <name>` 데이터 선언 바인딩(D1) 해석 진입점.
pub fn resolve_quant_attn_capability(
    name: &str,
    make_args: &QuantAttnMakeArgs,
) -> Option<Arc<dyn QuantAttnBackend>> {
    // 1) 정적(linkme) 우선.
    if let Some(reg) = argus_extension_api::find_quant_attn(name) {
        return Some(Arc::from((reg.make)(make_args)));
    }
    // 2) 동적(dlopen) fallback — category 다리.
    let registry = DYN_BACKEND_REGISTRY.get()?;
    let (category, cat_vtable, lib) = {
        let guard = registry
            .read()
            .expect("DYN_BACKEND_REGISTRY RwLock poisoned");
        let reg = guard.iter().find(|r| r.name == name)?;
        (reg.category, reg.cat_vtable, Arc::clone(&reg._lib))
    };
    // category 다리(D7): 알려진 카테고리 1개당 arm 1개. 새 카테고리 = arm 추가 = 재컴파일(C1).
    match category {
        BACKEND_CAP_CATEGORY_ATTENTION => {
            // SAFETY: try_register 가 category==ATTENTION 일 때만 등록 → cat_vtable 은 *const QuantAttnVTable.
            let vtable = cat_vtable as *const QuantAttnVTable;
            let handle = unsafe { ((*vtable).make)(make_args as *const QuantAttnMakeArgs) };
            if handle.is_null() {
                eprintln!(
                    "[resolve_quant_attn_capability] plugin '{name}' make returned a null handle"
                );
                return None;
            }
            Some(Arc::new(DynQuantAttnBackend {
                handle,
                vtable,
                _lib: lib,
            }))
        }
        other => {
            eprintln!("[resolve_quant_attn_capability] '{name}' category {other} unsupported");
            None
        }
    }
}

/// 동적 plugin KIVI ATTENTION capability 의 host 측 어댑터 — C [`QuantAttnVTable`] 마샬링으로
/// [`QuantAttnBackend`] 를 구현(D8). 단일 trait 이 이미 ABI-shaped(cl_mem args)라 args 를 vtable
/// fn-ptr 로 그대로 전달(Format `DynFormat` 거울이나 work-fn 2개 추가).
struct DynQuantAttnBackend {
    handle: *mut c_void,
    vtable: *const QuantAttnVTable,
    _lib: Arc<libloading::Library>,
}

// SAFETY: 핸들은 plugin 의 `QuantAttnBackend`(trait 계약상 Send+Sync) 인스턴스, vtable 불변, lib Arc 유지.
unsafe impl Send for DynQuantAttnBackend {}
unsafe impl Sync for DynQuantAttnBackend {}

impl Drop for DynQuantAttnBackend {
    fn drop(&mut self) {
        // SAFETY: handle 은 make 가 만든 plugin 인스턴스, 정확히 1회 해제.
        unsafe { ((*self.vtable).drop)(self.handle) };
    }
}

impl QuantAttnBackend for DynQuantAttnBackend {
    fn has_quant_attn_kernel(&self, bits: u8) -> bool {
        // SAFETY: handle/vtable 유효(lib 가 살려 둠).
        unsafe { ((*self.vtable).has_quant_attn_kernel)(self.handle, bits) }
    }

    fn is_nosub_device(&self) -> bool {
        // SAFETY: 위와 동일.
        unsafe { ((*self.vtable).is_nosub_device)(self.handle) }
    }

    fn attention_gen_quant(&self, args: &QuantAttnArgs) -> i32 {
        // SAFETY: args 는 host 가 채운 유효 QuantAttnArgs(C5 borrow-for-call). vtable fn-ptr 로 전달.
        unsafe { ((*self.vtable).attention_gen_quant)(self.handle, args as *const QuantAttnArgs) }
    }

    fn gather_update_quant(&self, args: &QuantAttnGatherArgs) -> i32 {
        // SAFETY: 위와 동일.
        unsafe {
            ((*self.vtable).gather_update_quant)(self.handle, args as *const QuantAttnGatherArgs)
        }
    }
}
