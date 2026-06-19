//! Engine-tier KV-mode registry — the FORMAT-axis analog of the eviction STAGE
//! registry (`KV_CACHE_STAGES` / `find_stage`).
//!
//! 설계 SSOT: `docs/design/format-axis-mode-knob-declaration.md` (A′ — Minimal
//! Mirror + build-fn-ptr seam).
//!
//! 엔진 코어는 더 이상 `KvMode::QuantWindow` 같은 **구체 기술 정체성에 match 하지 않는다**.
//! `--kv-mode <name>` 는 런타임 문자열이 되어 이 레지스트리 [`KV_MODES`] 에 대해 resolve
//! 된다(STAGE 축의 `find_stage`/`KV_CACHE_STAGES` 와 동형). 6개 dispatch 지점은 이름 대신
//! [`mode_caps`] 의 선언 bool([`ModeCaps`])을 읽고, chat 빌더는 [`KvModeReg::build`] fn-ptr
//! 로 whole-pipeline `Box<dyn Forward>` 를 조립한다.
//!
//! **배치(C3 벽 유지):** 이 레지스트리는 **엔진 crate 에 산다** — `build` fn-ptr 이
//! `Box<dyn Forward>` 를 반환하고 [`ModeBuildCtx`] 가 `Arc<dyn Backend>`/`Arc<dyn Memory>` 를
//! 운반하므로, 전부 API crate(`argus-extension-api`)가 봐선 안 되는 엔진 타입이다. 따라서
//! 1단계에서 API crate 표면은 전혀 변하지 않는다.

use std::sync::Arc;

use anyhow::Result;
use linkme::distributed_slice;

use crate::backend::Backend;
use crate::capability::CapabilityRegistry;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::session::cli::Args;
use crate::session::forward::Forward;
use crate::session::resilience_adapter::QuantStageHandle;

/// Plugin/builtin-declared capabilities the engine reads **before** instantiating a
/// mode (off the [`KvModeReg`], not via a trait method — the decision precedes
/// `build`). The FORMAT-axis analog of `StageCaps`. This is the surface that lets
/// the engine CLI/chat/eval/bench paths stay free of any concrete KV-technique
/// name: instead of `matches!(name, KvMode::QuantWindow | KvMode::Offload)` they read
/// these caps generically through [`mode_caps`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModeCaps {
    /// Whether this mode runs a quantized-KV pipeline (bench/eval pick the
    /// quantized runner, bin_setup gates the bits/residual derivation). Replaces
    /// `matches!(.., KvMode::QuantWindow)` at the bench-select + bin_setup sites.
    pub is_quantized_kv: bool,
    /// Whether this mode owns an offload cache container. Replaces
    /// `matches!(.., KvMode::Offload)` at the init.rs conflict check.
    pub supports_offload: bool,
    /// Whether this mode supports KV eviction. Standard = `true`; quantized/offload
    /// modes = `false` (eviction unsupported in v1). Replaces the `--chat` eviction
    /// conflict `matches!`.
    pub supports_eviction: bool,
    /// Whether this mode pulls the `QuantAttnBackend` capability when building
    /// (generalizes the `caps.get::<dyn QuantAttnBackend>()` pull that used to fire
    /// only on the quant-window arm). The quant_attn build closure consumes it; other modes ignore.
    pub needs_quant_attn: bool,
}

/// Inputs a [`KvModeReg::build`] closure consumes to assemble a chat-path
/// `Box<dyn Forward>` (+ the handles the mode-agnostic `ChatSession` assembly must
/// surface). The bundle the existing per-mode chat builders already received.
pub struct ModeBuildCtx<'a> {
    pub args: &'a Args,
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    /// CPU backend (Standard path needs it for `ModelForward`). Resolved by caller.
    pub cpu_backend: Arc<dyn Backend>,
    pub model: Arc<TransformerModel>,
    /// Backend capability registry — the quant_attn build closure pulls
    /// `caps.get::<dyn QuantAttnBackend>()` from here (gated by `needs_quant_attn`).
    pub caps: &'a CapabilityRegistry,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
}

/// The pipeline a [`KvModeReg::build`] closure produces for the chat path — the
/// `Box<dyn Forward>` plus the mode-specific handles the mode-agnostic
/// `ChatSession` assembly threads (resilience heartbeat handle, eviction policy
/// name, and the `ChatKvMode` stats-line payload).
pub struct ChatModeBuild {
    /// The whole-pipeline forward (`ModelForward` / `QuantWindowForward` / `OffloadForward`).
    pub forward: Box<dyn Forward>,
    /// Standard-format KV handles for the resilience `CommandDispatcher` (eviction
    /// inert in chat — out-of-loop CacheManager handles it). Empty for quantized/
    /// offload modes.
    pub kv_handles: Vec<Arc<crate::kv::standard_format::StandardFormat>>,
    /// The base pos/capacity heartbeat handle (layer-0). `Some` when the mode
    /// exposes a `KVCacheFormat` for snapshot queries.
    pub kv_handle: Option<Arc<dyn crate::format::KVCacheFormat>>,
    /// §4.5 quant bit-width handle for the resilience heartbeat (`Some` only for a
    /// quantized-KV mode). Points at the same layer-0 handle as `kv_handle`.
    pub quant_handle: Option<Arc<dyn QuantStageHandle>>,
    /// The eviction policy name to set on the resilience adapter (`""` for
    /// quantized/offload, which have no in-loop eviction).
    pub eviction_policy: String,
    /// The `ChatKvMode` stats-line payload (Standard owns its CacheManager; the
    /// quantized/offload variants carry their display knobs).
    pub kv_mode: crate::session::chat::session::ChatKvMode,
}

/// The registration entry for one KV mode. Builtins register via
/// `#[distributed_slice(KV_MODES)] static FOO: KvModeReg = ...`. The `build`
/// fn-ptr makes the registry a *factory*, not a label (it actually constructs the
/// pipeline — mirroring `find_stage(name).make`).
pub struct KvModeReg {
    /// The `--kv-mode <name>` selector. Unique within the slice.
    pub name: &'static str,
    /// Capabilities read pre-`build` ([`ModeCaps`]) — read via [`mode_caps`] so the
    /// 6 dispatch sites never name a concrete KV technique.
    pub caps: ModeCaps,
    /// Chat-path factory: builds the whole-pipeline `Box<dyn Forward>` + the handles
    /// the ChatSession assembly threads. Engine-tier (returns engine types).
    pub build: fn(ModeBuildCtx<'_>) -> Result<ChatModeBuild>,
}

/// The global KV-mode registration slice (gathered at link time, like
/// `KV_CACHE_STAGES`). Builtins (`standard`/`kivi`/`offload`) register below.
#[distributed_slice]
pub static KV_MODES: [KvModeReg];

/// Resolve a registered KV mode by name. `None` = unknown name. Mirrors
/// [`find_stage`](argus_extension_api::find_stage).
pub fn resolve_kv_mode(name: &str) -> Option<&'static KvModeReg> {
    KV_MODES.iter().find(|r| r.name == name)
}

/// The [`ModeCaps`] of a registered KV mode, by name. `None` if unknown. The lookup
/// the 6 dispatch sites read to stay free of concrete-technique `matches!`. Mirrors
/// [`stage_caps`](argus_extension_api::stage_caps).
pub fn mode_caps(name: &str) -> Option<ModeCaps> {
    resolve_kv_mode(name).map(|r| r.caps)
}

/// All registered KV-mode names (for `--help` / fail-fast diagnostics). Mirrors
/// [`registered_names`](argus_extension_api::registered_names).
pub fn mode_names() -> Vec<&'static str> {
    KV_MODES.iter().map(|r| r.name).collect()
}

/// Assert the 3 builtin KV modes are registered — call before resolving a
/// `--kv-mode` name. Fat-LTO `--gc-sections` may silently drop the
/// `#[distributed_slice]` registration; this fail-fasts instead of letting a valid
/// name mis-resolve. Mirrors `ensure_builtin_kv_formats_registered`.
pub fn ensure_builtin_kv_modes_registered() -> Result<()> {
    for name in ["standard", "kivi", "offload"] {
        if resolve_kv_mode(name).is_none() {
            anyhow::bail!(
                "built-in KV mode '{name}' not registered — suspect linkme fat-LTO \
                 --gc-sections silent drop of the KV_MODES #[distributed_slice]."
            );
        }
    }
    Ok(())
}

/// The single fail-fast funnel for a user-supplied `--kv-mode` name, shared by the
/// chat / bench / eval entry points. First runs the builtin self-test
/// ([`ensure_builtin_kv_modes_registered`]) — so a fat-LTO `--gc-sections` slice
/// drop cannot silently misclassify a known mode — then resolves the name, bailing
/// with the registered names if it is unknown.
///
/// `--kv-mode` is a free `String` (clap cannot enumerate runtime-registered modes
/// at compile time), so the parse-time reject the old closed `ValueEnum` gave for
/// free is restored HERE. Every binary that selects a mode MUST route through this
/// (not a bare [`mode_caps`] read), or a typo silently degrades to the standard
/// pipeline — the exact regression the FORMAT-axis Phase 1 review caught.
pub fn resolve_kv_mode_checked(name: &str) -> Result<&'static KvModeReg> {
    ensure_builtin_kv_modes_registered()?;
    resolve_kv_mode(name).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown --kv-mode '{name}'. Registered modes: {}.",
            mode_names().join(", ")
        )
    })
}

// ════════════════════════════════════════════════════════════════════════════
// 빌트인 3종 등록 (standard / kivi / offload)
// ════════════════════════════════════════════════════════════════════════════
//
// 각 build 클로저는 기존 chat 빌더(`build_chat_standard`/`build_chat_kivi`/
// `build_chat_offload`)의 forward-구성 본문을 *그대로* 들고 와 `Box<dyn Forward>` (+
// 핸들/라벨)를 반환한다. mode-agnostic 한 ChatSession 조립(registry/sampler/finish_chat_loop)
// 은 `chat/build.rs` 가 이 build 결과 *둘레*에서 수행한다. quant-window-private 구성
// (alloc_quant_window_kv_caches / QuantWindowForward::new / bits·residual / caps.get::<QuantAttnBackend>())
// 은 전부 "kivi" 클로저 *안*으로 이동했다 — dispatch 지점이 더는 "kivi" 를 NAME 하지 않는다.

#[distributed_slice(KV_MODES)]
static STANDARD_MODE: KvModeReg = KvModeReg {
    name: "standard",
    caps: ModeCaps {
        is_quantized_kv: false,
        supports_offload: false,
        supports_eviction: true,
        needs_quant_attn: false,
    },
    build: build_standard_forward,
};

#[distributed_slice(KV_MODES)]
static QUANT_WINDOW_MODE: KvModeReg = KvModeReg {
    name: "kivi",
    caps: ModeCaps {
        is_quantized_kv: true,
        supports_offload: false,
        supports_eviction: false,
        needs_quant_attn: true,
    },
    build: build_quant_window_forward,
};

#[distributed_slice(KV_MODES)]
static OFFLOAD_MODE: KvModeReg = KvModeReg {
    name: "offload",
    caps: ModeCaps {
        is_quantized_kv: false,
        supports_offload: true,
        supports_eviction: false,
        needs_quant_attn: false,
    },
    build: build_offload_forward,
};

/// "standard" build closure — `ModelForward` + CacheManager/score-accumulator
/// (eviction is mode-agnostically owned here since Standard is the only eviction
/// mode). `chat/build.rs` 의 옛 `KvMode::Standard` arm verbatim.
fn build_standard_forward(ctx: ModeBuildCtx<'_>) -> Result<ChatModeBuild> {
    crate::session::chat::session::build_chat_standard_forward(ctx)
}

/// "kivi" build closure — quant-window-private 구성 일체(alloc_quant_window_kv_caches / QuantWindowForward::new /
/// bits·residual / caps.get::<QuantAttnBackend>())가 여기로 이동했다.
fn build_quant_window_forward(ctx: ModeBuildCtx<'_>) -> Result<ChatModeBuild> {
    crate::session::chat::session::build_chat_quant_window_forward(ctx)
}

/// "offload" build closure — `OffloadForward` + offload cache container.
fn build_offload_forward(ctx: ModeBuildCtx<'_>) -> Result<ChatModeBuild> {
    crate::session::chat::session::build_chat_offload_forward(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_modes_registered() {
        assert!(resolve_kv_mode("standard").is_some());
        assert!(resolve_kv_mode("kivi").is_some());
        assert!(resolve_kv_mode("offload").is_some());
        assert!(resolve_kv_mode("nonexistent").is_none());
        ensure_builtin_kv_modes_registered().expect("3 builtin modes registered");
    }

    #[test]
    fn mode_caps_declared_correctly() {
        // Standard: eviction-capable, not quantized, no offload.
        let s = mode_caps("standard").unwrap();
        assert!(s.supports_eviction && !s.is_quantized_kv && !s.supports_offload);
        // quant-window: quantized, needs quant-attn, no eviction.
        let k = mode_caps("kivi").unwrap();
        assert!(k.is_quantized_kv && k.needs_quant_attn && !k.supports_eviction);
        // Offload: offload container, no eviction, not quantized.
        let o = mode_caps("offload").unwrap();
        assert!(o.supports_offload && !o.supports_eviction && !o.is_quantized_kv);
    }

    #[test]
    fn mode_names_lists_builtins() {
        let names = mode_names();
        for n in ["standard", "kivi", "offload"] {
            assert!(names.contains(&n), "'{n}' missing from mode_names()");
        }
    }

    #[test]
    fn resolve_kv_mode_checked_accepts_builtins_and_rejects_unknown() {
        // the shared fail-fast funnel (chat/bench/eval): known names resolve...
        for n in ["standard", "kivi", "offload"] {
            assert!(resolve_kv_mode_checked(n).is_ok(), "'{n}' must resolve");
        }
        // ...an unknown/typo'd name errors (restores the lost clap ValueEnum reject;
        // the message lists the registered names). Match rather than `unwrap_err()`
        // since the Ok type `&KvModeReg` (holds a fn-ptr) does not derive Debug.
        let err = match resolve_kv_mode_checked("kvi") {
            Ok(_) => panic!("'kvi' must not resolve"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("Unknown --kv-mode 'kvi'"), "got: {err}");
        assert!(
            err.contains("standard"),
            "error should list registered names: {err}"
        );
    }
}
