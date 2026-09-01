use clap::Parser;

pub mod eviction;
pub mod kv_mode;

pub use eviction::{EvictionCmd, EvictionCommonArgs, PluginArgs, TopLevelCmd};
pub use kv_mode::KvModeArgs;

/// `--secondary-dtype` CLI 인수 값 (D-3, ENG-ALG-225).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryDtypeArg {
    Auto,
    F16,
    Q4_0,
    F32,
}

/// `--secondary-dtype` value_parser.
pub fn parse_secondary_dtype(s: &str) -> Result<SecondaryDtypeArg, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(SecondaryDtypeArg::Auto),
        "f16" => Ok(SecondaryDtypeArg::F16),
        "q4_0" | "q4" => Ok(SecondaryDtypeArg::Q4_0),
        "f32" => Ok(SecondaryDtypeArg::F32),
        other => Err(format!(
            "unknown secondary-dtype '{other}'. Valid values: auto, f16, q4_0, f32"
        )),
    }
}

impl From<SecondaryDtypeArg> for crate::models::weights::SecondaryDtypeChoice {
    fn from(arg: SecondaryDtypeArg) -> Self {
        match arg {
            SecondaryDtypeArg::Auto => Self::Auto,
            SecondaryDtypeArg::F16 => Self::F16,
            SecondaryDtypeArg::Q4_0 => Self::Q4_0,
            SecondaryDtypeArg::F32 => Self::F32,
        }
    }
}

/// `--secondary-layout` CLI 인수 값.
///
/// AUF의 어떤 weights variant로 swap 후 텐서를 만들지 결정한다. 기본은
/// `auto`로, 빌드 환경의 preferred variant(OpenCL→AdrenoSoa) 우선 + AUF에
/// 그게 없으면 CpuAos로 폴백한다. `aos`는 강제로 CpuAos / CudaAos 사용해
/// host pointer를 살려두므로 swap 후 `switch_hw cpu` / partition 호환이
/// 가능하지만 GPU TBT가 SOA 대비 떨어진다 (Adreno 830 실측 33–55%).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryLayoutArg {
    Auto,
    Aos,
    Soa,
}

pub fn parse_secondary_layout(s: &str) -> Result<SecondaryLayoutArg, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(SecondaryLayoutArg::Auto),
        "aos" | "cpu_aos" | "cuda_aos" => Ok(SecondaryLayoutArg::Aos),
        "soa" | "adreno_soa" => Ok(SecondaryLayoutArg::Soa),
        other => Err(format!(
            "unknown secondary-layout '{other}'. Valid values: auto, aos, soa"
        )),
    }
}

impl From<SecondaryLayoutArg> for crate::models::weights::SecondaryLayoutChoice {
    fn from(arg: SecondaryLayoutArg) -> Self {
        match arg {
            SecondaryLayoutArg::Auto => Self::Auto,
            SecondaryLayoutArg::Aos => Self::Aos,
            SecondaryLayoutArg::Soa => Self::Soa,
        }
    }
}

/// `--primary-variant` CLI 인수 값 (W-AUF-1 C4). AUF primary backend variant 선택.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryVariantArg {
    Auto,
    AdrenoSoa,
    CpuAos,
    CudaAos,
}

pub fn parse_primary_variant(s: &str) -> Result<PrimaryVariantArg, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(PrimaryVariantArg::Auto),
        "adreno-soa" | "adreno_soa" => Ok(PrimaryVariantArg::AdrenoSoa),
        "cpu-aos" | "cpu_aos" => Ok(PrimaryVariantArg::CpuAos),
        "cuda-aos" | "cuda_aos" => Ok(PrimaryVariantArg::CudaAos),
        other => Err(format!(
            "unknown primary-variant '{other}'. Valid: auto, adreno-soa, cpu-aos, cuda-aos"
        )),
    }
}

impl From<PrimaryVariantArg> for crate::models::loader::AufVariantChoice {
    fn from(arg: PrimaryVariantArg) -> Self {
        match arg {
            PrimaryVariantArg::Auto => Self::Auto,
            PrimaryVariantArg::AdrenoSoa => Self::AdrenoSoa,
            PrimaryVariantArg::CpuAos => Self::CpuAos,
            PrimaryVariantArg::CudaAos => Self::CudaAos,
        }
    }
}

/// `--primary-dtype` CLI 인수 값 (W-AUF-1 C4). AUF primary dtype 선택.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryDtypeArg {
    Auto,
    F16,
    Q4_0,
    Q8_0,
    Bf16,
    F32,
    Q4_1,
}

pub fn parse_primary_dtype(s: &str) -> Result<PrimaryDtypeArg, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(PrimaryDtypeArg::Auto),
        "f16" => Ok(PrimaryDtypeArg::F16),
        "q4_0" | "q4" => Ok(PrimaryDtypeArg::Q4_0),
        "q8_0" | "q8" => Ok(PrimaryDtypeArg::Q8_0),
        "bf16" => Ok(PrimaryDtypeArg::Bf16),
        "f32" => Ok(PrimaryDtypeArg::F32),
        "q4_1" => Ok(PrimaryDtypeArg::Q4_1),
        other => Err(format!(
            "unknown primary-dtype '{other}'. Valid: auto, f16, q4_0, q8_0, bf16, f32, q4_1"
        )),
    }
}

impl From<PrimaryDtypeArg> for crate::models::loader::AufDtypeChoice {
    fn from(arg: PrimaryDtypeArg) -> Self {
        match arg {
            PrimaryDtypeArg::Auto => Self::Auto,
            PrimaryDtypeArg::F16 => Self::F16,
            PrimaryDtypeArg::Q4_0 => Self::Q4_0,
            PrimaryDtypeArg::Q8_0 => Self::Q8_0,
            PrimaryDtypeArg::Bf16 => Self::BF16,
            PrimaryDtypeArg::F32 => Self::F32,
            PrimaryDtypeArg::Q4_1 => Self::Q4_1,
        }
    }
}

/// REQ-3: surface a one-time setup warning when an engine-level KV budget disagrees with faithful
/// h2o's own ABSOLUTE `hh_size + recent_size (+ protected_prefix)`.
///
/// Faithful h2o sizes its resident cache purely from its absolute budget — `H2o::partition` reads
/// neither `--kv-budget` nor the streaming `target_len`. So a `--kv-budget B` that differs from
/// `hh + recent + prefix` is silently non-authoritative: the cache settles near `hh + recent + prefix`,
/// not `B`, with no per-eviction-event diagnostic. This returns a human-readable warning (or `None`
/// when there is nothing to reconcile) so the caller emits it once at setup. Pure — unit testable
/// without a model or a parsed `Args`.
pub fn h2o_budget_mismatch_warning(
    policy: &str,
    hh_size: Option<usize>,
    recent_size: Option<usize>,
    protected_prefix: usize,
    kv_budget: usize,
    kv_budget_ratio: f32,
) -> Option<String> {
    if policy != "h2o" {
        return None;
    }
    // Missing budgets are a hard error elsewhere (`require_h2o_budgets`); nothing to reconcile here.
    let (hh, recent) = (hh_size?, recent_size?);
    let h2o_keep = protected_prefix + hh + recent;

    if kv_budget > 0 {
        // Absolute streaming/overflow cap. Allow a one-token slack (float-floor rounding of the
        // streaming low-water) before warning.
        if (h2o_keep as i64 - kv_budget as i64).abs() > 1 {
            return Some(format!(
                "[h2o-budget] WARNING: --kv-budget {kv_budget} disagrees with faithful h2o's absolute \
                 budget hh_size + recent_size + protected_prefix = {hh} + {recent} + {protected_prefix} \
                 = {h2o_keep}. h2o sizes its cache from the absolute budget and ignores --kv-budget for \
                 the keep decision, so the resident cache settles near {h2o_keep} tokens, NOT {kv_budget}. \
                 Set --kv-budget = {h2o_keep} (or drop it) to make the streaming cap authoritative."
            ));
        }
        return None;
    }
    if kv_budget_ratio > 0.0 {
        return Some(format!(
            "[h2o-budget] WARNING: --kv-budget-ratio {kv_budget_ratio} does not size faithful h2o's \
             cache — h2o keeps an ABSOLUTE hh_size + recent_size + protected_prefix = {hh} + {recent} \
             + {protected_prefix} = {h2o_keep} tokens regardless of the ratio. The ratio only gates \
             whether the per-question overflow pass runs."
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── REQ-2: require_h2o_budgets contract (the guard argus-eval now wires in) ──

    #[test]
    fn require_h2o_budgets_rejects_missing_and_accepts_present() {
        // h2o with NO budgets → hard error (argus-eval used to silently zero both → empty cache).
        let missing =
            Args::try_parse_from(["test", "eviction", "plugin", "--name", "h2o"]).unwrap();
        assert_eq!(missing.eviction_policy(), "h2o");
        assert!(
            missing.require_h2o_budgets().is_err(),
            "h2o without --set hh_size/recent_size must be rejected"
        );
        // Only one of the two present → still rejected.
        let half = Args::try_parse_from([
            "test",
            "eviction",
            "plugin",
            "--name",
            "h2o",
            "--set",
            "hh_size=64",
        ])
        .unwrap();
        assert!(
            half.require_h2o_budgets().is_err(),
            "partial budgets rejected"
        );
        // Both present → accepted.
        let ok = Args::try_parse_from([
            "test",
            "eviction",
            "plugin",
            "--name",
            "h2o",
            "--set",
            "hh_size=64",
            "--set",
            "recent_size=64",
        ])
        .unwrap();
        assert!(
            ok.require_h2o_budgets().is_ok(),
            "explicit budgets accepted"
        );
        // Non-h2o policy → no-op regardless.
        let sliding =
            Args::try_parse_from(["test", "eviction", "plugin", "--name", "sliding"]).unwrap();
        assert!(sliding.require_h2o_budgets().is_ok());
    }

    // ── REQ-3: h2o_budget_mismatch_warning (pure) ──

    #[test]
    fn h2o_budget_warn_noop_for_non_h2o() {
        assert!(h2o_budget_mismatch_warning("sliding", Some(64), Some(64), 0, 128, 0.0).is_none());
        assert!(h2o_budget_mismatch_warning("d2o", None, None, 0, 128, 0.0).is_none());
    }

    #[test]
    fn h2o_budget_warn_missing_budgets_is_noop() {
        // require_h2o_budgets owns the hard error; this reconciler stays quiet.
        assert!(h2o_budget_mismatch_warning("h2o", None, Some(64), 0, 128, 0.0).is_none());
        assert!(h2o_budget_mismatch_warning("h2o", Some(64), None, 0, 128, 0.0).is_none());
    }

    #[test]
    fn h2o_budget_warn_absolute_agree_is_noop() {
        // hh + recent + prefix == kv_budget → authoritative, no warning. (±1 slack tolerated.)
        assert!(h2o_budget_mismatch_warning("h2o", Some(100), Some(28), 0, 128, 0.0).is_none());
        assert!(h2o_budget_mismatch_warning("h2o", Some(100), Some(100), 0, 200, 0.0).is_none());
        assert!(h2o_budget_mismatch_warning("h2o", Some(64), Some(64), 0, 129, 0.0).is_none()); // +1 slack
    }

    #[test]
    fn h2o_budget_warn_absolute_mismatch_fires() {
        // Spec §2(b) worked example: --kv-budget 128 with hh=100+recent=100 → h2o_keep=200 ≠ 128.
        let w = h2o_budget_mismatch_warning("h2o", Some(100), Some(100), 0, 128, 0.0)
            .expect("mismatch must warn");
        assert!(w.contains("--kv-budget 128"));
        assert!(
            w.contains("200"),
            "warning states the real settling point: {w}"
        );
    }

    #[test]
    fn h2o_budget_warn_prefix_folds_into_total() {
        // 60 + 60 + 8 = 128 == --kv-budget 128 → agrees (prefix counted).
        assert!(h2o_budget_mismatch_warning("h2o", Some(60), Some(60), 8, 128, 0.0).is_none());
        // 60 + 60 + 0 = 120 vs 128 → 8 apart → warns.
        assert!(h2o_budget_mismatch_warning("h2o", Some(60), Some(60), 0, 128, 0.0).is_some());
    }

    #[test]
    fn h2o_budget_warn_ratio_mode_notes_absolute_sizing() {
        // Ratio does not size h2o's cache; warn informationally.
        let w = h2o_budget_mismatch_warning("h2o", Some(50), Some(50), 0, 0, 0.5)
            .expect("ratio + h2o must note absolute sizing");
        assert!(w.contains("--kv-budget-ratio"));
        assert!(w.contains("100"));
        // No engine budget at all → nothing to reconcile.
        assert!(h2o_budget_mismatch_warning("h2o", Some(50), Some(50), 0, 0, 0.0).is_none());
    }

    // ── offload storage backend CLI 검증 (B5-2a: mmap-default 버그 수정) ──

    /// `--kv-offload-storage` 는 배선된 backend(`raw`/`disk`)만 받고 그 외(`mmap`/`tmpfs`/…)는
    /// clap 단에서 거부한다. 기본값은 `raw` — 과거 기본값 `mmap` 은 미배선이라 store 생성자에서
    /// bail 해 offload 가 out-of-box 로 깨졌었다(`alloc_offload_kv_caches`).
    #[test]
    fn kv_offload_storage_rejects_unwired_backends() {
        // 기본값 = raw(배선됨).
        let a = Args::try_parse_from(["test"]).unwrap();
        assert_eq!(
            a.kv_mode_args.kv_offload_storage, "raw",
            "기본 offload storage = raw"
        );

        // 배선된 값은 통과.
        for ok in ["raw", "disk"] {
            assert!(
                Args::try_parse_from(["test", "--kv-offload-storage", ok]).is_ok(),
                "배선된 backend '{ok}' 는 허용되어야 한다"
            );
        }

        // 미배선 값은 런타임 bail 이 아니라 clap 파싱 단에서 거부.
        for bad in ["mmap", "tmpfs", "bogus"] {
            assert!(
                Args::try_parse_from(["test", "--kv-offload-storage", bad]).is_err(),
                "미배선 backend '{bad}' 는 clap 이 거부해야 한다"
            );
        }
    }

    // ── score decay(forgetting-factor) 주입 — CLI 배선 게이트 (KV roadmap 항목 0 §4.2) ──

    /// `--score-decay` 미지정(기본 0.0) → `h2o_decay()` 가 정책 자체 값을 그대로 반환.
    /// flag 도입 전 경로 bit-identical: eviction 미지정 시 0.0, heavy-hitter `--decay 0.3` 시 0.3.
    #[test]
    fn score_decay_default_preserves_policy_decay() {
        // eviction 미지정 → h2o_decay() = 0.0 (기존 동작).
        let a = Args::try_parse_from(["test"]).unwrap();
        assert_eq!(a.eviction_common.score_decay, 0.0, "기본 0.0");
        assert_eq!(
            a.h2o_decay(),
            0.0,
            "미주입 + 정책 없음 → 0.0 (bit-identical)"
        );

        // h2o --set decay=0.3, --score-decay 미지정 → 정책 값 0.3 유지.
        let b = Args::try_parse_from([
            "test",
            "eviction",
            "plugin",
            "--name",
            "h2o",
            "--set",
            "decay=0.3",
        ])
        .unwrap();
        assert!(
            (b.h2o_decay() - 0.3).abs() < 1e-6,
            "score-decay 미주입 → --set decay=0.3 그대로"
        );
    }

    /// `--score-decay 0.8` (> 0.0) → 정책 무관하게 0.8 을 우선 주입.
    /// 정책 자체 decay(heavy-hitter --decay 0.3)보다 측정 flag 가 우선.
    #[test]
    fn score_decay_overrides_when_positive() {
        // eviction 미지정이어도 --score-decay 가 주입된다.
        let a = Args::try_parse_from(["test", "--score-decay", "0.8"]).unwrap();
        assert!((a.h2o_decay() - 0.8).abs() < 1e-6, "정책 무관 0.8 주입");

        // --set decay=0.3 위에 --score-decay 0.9 → measurement flag 우선(0.9).
        let b = Args::try_parse_from([
            "test",
            "--score-decay",
            "0.9",
            "eviction",
            "plugin",
            "--name",
            "h2o",
            "--set",
            "decay=0.3",
        ])
        .unwrap();
        assert!(
            (b.h2o_decay() - 0.9).abs() < 1e-6,
            "score-decay 0.9 가 --set decay=0.3 보다 우선"
        );
    }

    /// B1-3 feature-equivalence: `eviction plugin --name h2o --set k=v` reconstructs the same
    /// downstream config the former typed `eviction h2o --keep-ratio ...` produced.
    #[test]
    fn plugin_set_form_matches_old_typed_h2o() {
        let a = Args::try_parse_from([
            "test",
            "eviction",
            "plugin",
            "--name",
            "h2o",
            "--set",
            "keep_ratio=0.3",
            "--set",
            "tracked_layers=8",
            "--set",
            "decay=0.1",
            "--set",
            "raw_scores=true",
        ])
        .unwrap();
        assert_eq!(a.eviction_policy(), "h2o");
        assert!((a.keep_ratio() - 0.3).abs() < 1e-6);
        assert_eq!(a.h2o_tracked_layers(), 8);
        assert!((a.h2o_decay() - 0.1).abs() < 1e-6);
        assert!(a.h2o_raw_scores());
        // POD knobs default cleanly for h2o (no sliding/streaming keys in the blob).
        assert_eq!(a.eviction_window(), 1024);
        assert_eq!(a.sink_size(), 4);
    }

    /// `--evict-timing` defaults to today's behavior, parses the query-agnostic
    /// modes, and rejects unknown modes (no silent fallback).
    #[test]
    fn evict_timing_flag_parses_and_defaults() {
        use crate::session::eval::EvictTiming;

        // Absent flag → INV-147 default (today's post-question probe path).
        let d = Args::try_parse_from(["test"]).unwrap();
        assert_eq!(d.evict_timing(), EvictTiming::PostPrefillProbe);
        assert!(d.evict_timing().runs_query_probe());

        // Explicit default is identical.
        let p = Args::try_parse_from(["test", "--evict-timing", "post_prefill_probe"]).unwrap();
        assert_eq!(p.evict_timing(), EvictTiming::PostPrefillProbe);

        // Query-agnostic end-of-prefill mode parses and flips both behavior bits.
        let e = Args::try_parse_from(["test", "--evict-timing", "prefill_end"]).unwrap();
        assert_eq!(e.evict_timing(), EvictTiming::PrefillEnd);
        assert!(!e.evict_timing().runs_query_probe());
        assert!(e.evict_timing().accumulates_context_scores());
        assert!(!e.evict_timing().evicts_on_overflow());

        // Variant b (evict-on-overflow streaming) now parses too: same query-agnostic
        // accumulation, plus the overflow-eviction axis.
        let s = Args::try_parse_from(["test", "--evict-timing", "prefill_streaming"]).unwrap();
        assert_eq!(s.evict_timing(), EvictTiming::PrefillStreaming);
        assert!(!s.evict_timing().runs_query_probe());
        assert!(s.evict_timing().accumulates_context_scores());
        assert!(s.evict_timing().evicts_on_overflow());

        // An unknown mode is rejected at parse time, not silently coerced to default.
        assert!(Args::try_parse_from(["test", "--evict-timing", "bogus"]).is_err());
    }

    /// B1-3: sliding/streaming POD knobs are read from `--set`, not a typed variant.
    #[test]
    fn plugin_set_form_sliding_streaming_pod_knobs() {
        let s = Args::try_parse_from([
            "test",
            "eviction",
            "plugin",
            "--name",
            "sliding",
            "--set",
            "window=2048",
        ])
        .unwrap();
        assert_eq!(s.eviction_policy(), "sliding");
        assert_eq!(s.eviction_window(), 2048);

        let st = Args::try_parse_from([
            "test",
            "eviction",
            "plugin",
            "--name",
            "streaming",
            "--set",
            "sink=8",
            "--set",
            "recent_window=64",
        ])
        .unwrap();
        assert_eq!(st.eviction_policy(), "streaming");
        assert_eq!(st.sink_size(), 8);
        assert_eq!(st.streaming_window(), 64);
    }

    /// B1-3: d2o's technique-private knobs round-trip through `stage_args()` (the opaque blob).
    #[test]
    fn plugin_set_form_d2o_blob_roundtrips() {
        let a = Args::try_parse_from([
            "test",
            "eviction",
            "plugin",
            "--name",
            "d2o",
            "--set",
            "keep_ratio=0.75",
            "--set",
            "ema_beta=0.6",
            "--set",
            "merge_axis=value_only",
            "--set",
            "protected_layers=0,1,2",
        ])
        .unwrap();
        assert_eq!(a.eviction_policy(), "d2o");
        // d2o must pass keep_ratio explicitly now (the typed 0.75 default is gone — B1-3).
        assert!((a.keep_ratio() - 0.75).abs() < 1e-6);
        let blob = a.stage_args();
        assert!(blob.contains(&("merge_axis".to_string(), "value_only".to_string())));
        assert!(blob.contains(&("protected_layers".to_string(), "0,1,2".to_string())));
        assert!(blob.contains(&("ema_beta".to_string(), "0.6".to_string())));
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value = "models/llama3.2-1b")]
    pub model_path: String,

    /// Path to a file containing the prompt. Overrides --prompt if set.
    #[arg(long)]
    pub prompt_file: Option<String>,

    #[arg(short, long, default_value = "Hello, world! I am a")]
    pub prompt: String,

    #[arg(short, long, default_value_t = 20)]
    pub num_tokens: usize,

    /// Backend to use: "cpu", "opencl", or "cuda" (build with --features cuda).
    /// Default: Android target → "opencl" (Adreno production path); a host built with a
    /// CUDA feature → "cuda"; otherwise → "cpu". On a CUDA build, pass `--backend cpu`
    /// to force the CPU path. (Mirrors the AUF primary-variant default, which already
    /// selects `CudaAos` under `cuda`/`cuda-embedded`.)
    #[cfg(target_os = "android")]
    #[arg(short, long, default_value = "opencl")]
    pub backend: String,

    /// Backend to use: "cpu", "opencl", or "cuda". Host CUDA build → default "cuda"
    /// (pass `--backend cpu` to force CPU).
    #[cfg(all(
        not(target_os = "android"),
        any(feature = "cuda", feature = "cuda-embedded")
    ))]
    #[arg(short, long, default_value = "cuda")]
    pub backend: String,

    /// Backend to use: "cpu", "opencl", or "cuda" (build with --features cuda).
    /// Host build without a CUDA feature → default "cpu".
    #[cfg(all(
        not(target_os = "android"),
        not(any(feature = "cuda", feature = "cuda-embedded"))
    ))]
    #[arg(short, long, default_value = "cpu")]
    pub backend: String,

    /// Disable zero-copy shared memory (CL_MEM_ALLOC_HOST_PTR).
    ///
    /// Zero-copy is enabled by default on ARM SoC to remove CPU↔GPU memcpy.
    /// Set this flag to fall back to device-only allocations.
    ///
    /// Other features force-enable zero-copy regardless of this flag:
    /// `--resilience-prealloc-switch`, `--tensor-partition > 0`,
    /// `--prefill-cpu-chunk-size > 0`, `--enable-resilience`.
    #[arg(long, default_value_t = false)]
    pub no_zero_copy: bool,

    /// Sprint 2a Phase 2 (ENG-RPCMEM-040): enable rpcmem DMA-BUF zero-copy
    /// allocation for KV cache and precision swap secondary store.
    ///
    /// Adreno Android only — host builds receive a warning and silently
    /// demote. Requires `--backend opencl`.
    ///
    /// When active, OpenCL backend eagerly dlopens `libcdsprpc.so` and shares
    /// a single `Arc<RpcmemAllocator>` between `OpenCLMemory::alloc_kv`
    /// (KV path) and `RpcmemSecondaryStore` (precision swap secondary).
    #[arg(long, default_value_t = false)]
    pub opencl_rpcmem: bool,

    #[arg(long, default_value_t = 2048)]
    pub max_seq_len: usize,

    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,

    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,

    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    #[arg(long, default_value_t = 1.1)]
    pub repetition_penalty: f32,

    #[arg(long, default_value_t = 64)]
    pub repetition_window: usize,

    // ── head-masking ablation (causal recall-head test; argus-cli free-generation only) ──
    /// Zero out named `layer:head` attention-output contributions during real generation, to
    /// causally test whether candidate "recall heads" are necessary for needle-recall. Comma-
    /// separated, e.g. `--mask-heads 14:3,19:3`. Applied in BOTH prefill and decode. Mutually
    /// exclusive with `--mask-heads-random`. Backend: `cpu`, `cuda`, or `opencl` (Adreno).
    #[arg(long)]
    pub mask_heads: Option<String>,

    /// Control variant for `--mask-heads`: mask `N` randomly chosen `(layer,head)` pairs instead of
    /// a named list (seeded by `--mask-seed` for reproducibility). Mutually exclusive with
    /// `--mask-heads`.
    #[arg(long)]
    pub mask_heads_random: Option<usize>,

    /// Seed for `--mask-heads-random`'s draw (only affects which pairs are chosen; sampling is
    /// unaffected). Re-running a larger `N` with the same seed nests the smaller draw.
    #[arg(long, default_value_t = 0)]
    pub mask_seed: u64,

    /// Head-masking substitution mode: `zero` (default; Wu et al. hard zero) or `mean` (substitute
    /// each masked head's slice with a supplied per-head mean vector — an off-distribution-bias
    /// control per the IOI critique). `mean` requires `--mask-heads-means`.
    #[arg(long, default_value = "zero")]
    pub mask_heads_mode: String,

    /// Required iff `--mask-heads-mode mean`: a JSON file of per-head mean vectors,
    /// `{"14:3": [head_dim floats], ...}`, computed lab-side over a reference batch. The engine only
    /// substitutes the slice — it never computes means itself.
    #[arg(long)]
    pub mask_heads_means: Option<String>,

    // ── DuoAttention streaming-head ablation (output-fidelity probe; argus-cli free-generation only) ──
    /// Path to a DuoAttention `full_attention_heads` gate/label file (`n_layers` rows × `n_heads_kv`
    /// whitespace/comma-separated floats). A KV head is *retrieval* (full attention) iff its gate
    /// `>= --duo-threshold`, else *streaming* (attends only to the first `--duo-sink-size` ∪ last
    /// `--duo-recent-size` tokens). Streaming heads' attention output is recomputed over that
    /// Λ-window in BOTH prefill and decode. NOTE: output-fidelity probe only — full KV stays
    /// allocated (ZERO memory saving) and it ADDS compute; it reproduces DuoAttention's streaming
    /// LOGITS, not its system benefit. Backend: cpu, cuda, or opencl. argus-eval rejects it.
    #[arg(long)]
    pub duo_heads: Option<String>,

    /// `--duo-heads` retrieval/streaming threshold (gate `>= threshold` → retrieval head).
    #[arg(long, default_value_t = 0.5)]
    pub duo_threshold: f32,

    /// `--duo-heads` streaming attention-sink prefix length (first-N tokens always attended).
    #[arg(long, default_value_t = 64)]
    pub duo_sink_size: usize,

    /// `--duo-heads` streaming recent sliding-window length (last-N tokens attended).
    #[arg(long, default_value_t = 256)]
    pub duo_recent_size: usize,

    /// Disable GPU kernel plan for decode (fallback to forward_into every token)
    #[arg(long, default_value_t = false)]
    pub no_gpu_plan: bool,

    /// GPU ratio for tensor partition — fraction of FFN gate/up rows assigned to GPU.
    /// Range (0.0, 1.0): 0.0 = disabled (no split), 1.0 = disabled (no split).
    /// 0.1 = 10% GPU + 90% CPU, 0.9 = 90% GPU + 10% CPU.
    /// NOTE: split_row is clamped to [128, out_dim-128], so extreme values like 0.001
    /// still leave 128 rows on GPU and the rest (CPU-heavy) on CPU — not "almost all GPU".
    /// Use 1.0 or omit the flag for GPU-only execution.
    #[arg(long, default_value_t = 0.0)]
    pub tensor_partition: f32,

    /// Chunked prefill: split long prompts into chunks to limit peak memory.
    /// 0 = auto (default): GPU backend derives a safe size from max_single_alloc()
    ///     to avoid CL_INVALID_BUFFER_SIZE; CPU backend processes entire prompt as one batch.
    #[arg(long, default_value_t = 0)]
    pub prefill_chunk_size: usize,

    /// Inter-chunk yield delay in milliseconds during prefill.
    /// After each prefill chunk, engine calls synchronize() + sleep(yield_ms).
    /// 0 = no yield. Dynamically adjustable via SetPrefillPolicy.
    #[arg(long, default_value_t = 0)]
    pub prefill_yield_ms: u32,

    /// CPU chunk size for GPU-CPU prefill interleaving.
    /// 0 = disabled. After each GPU chunk, CPU processes this many tokens.
    /// Requires --zero-copy or --resilience-prealloc-switch for weight access.
    #[arg(long, default_value_t = 0)]
    pub prefill_cpu_chunk_size: usize,

    /// Enable profiling (per-op timing, latency, score snapshots).
    ///
    /// Legacy mode: inserts two `clFinish()` calls per op on GPU, which
    /// inflates decode ms/tok by ~54 ms on Adreno. Useful for **relative**
    /// per-op ranking only. For apples-to-apples comparison with llama.cpp
    /// per-op GPU timing, use `--profile-events` instead.
    #[arg(long, default_value_t = false)]
    pub profile: bool,

    /// Enable OpenCL event-based per-op profiling.
    ///
    /// Creates the command queue with `CL_QUEUE_PROFILING_ENABLE` and
    /// captures a profiling event per kernel dispatch. At decode-step
    /// boundaries the `End-Start` nanoseconds are aggregated per logical
    /// op label. Unlike `--profile`, this adds no `clFinish()` calls and
    /// closely matches absolute GPU time (same mechanism as
    /// `GGML_OPENCL_PROFILING` in llama.cpp).
    ///
    /// Mutually exclusive with `--profile`.
    #[arg(long, default_value_t = false)]
    pub profile_events: bool,

    /// Enable GPU self-utilization measurement in Heartbeat (MSG-068 Phase 2).
    ///
    /// Turns on OpenCL queue profiling (same mechanism as `--profile-events`)
    /// and feeds the accumulated GPU busy ns into `EngineStatus.self_gpu_pct`
    /// so the Manager / LuaPolicy `ctx.engine.gpu_pct` reflects real usage
    /// instead of the Phase 1 hardcoded 0.0.
    ///
    /// **Overhead**: on Adreno, queue profiling adds ~54 ms/token. Keep OFF
    /// for production TBT measurements. OFF is the default — heartbeat
    /// `self_gpu_pct` stays at 0.0 (INV-092 fallback).
    ///
    /// If `--profile-events` is already set this flag is redundant; both
    /// share the same backend profiling infrastructure.
    #[arg(long, default_value_t = false)]
    pub heartbeat_gpu_profile: bool,

    /// Output directory for profiling data.
    #[arg(long, default_value = "results/profile")]
    pub profile_dir: String,

    /// Score snapshot interval (1 = every step, 10 = every 10th step).
    #[arg(long, default_value_t = 1)]
    pub profile_interval: usize,

    /// Comma-separated list of probes: ops,latency,scores,entropy,cache.
    #[arg(long, default_value = "ops,latency,scores")]
    pub profile_probes: String,

    /// Enable per-KV-head score tracking (for heavy-hitter+ analysis).
    #[arg(long, default_value_t = false)]
    pub profile_per_head: bool,

    /// Enable per-op CUDA event profiler (cuda-embedded backend only).
    ///
    /// Wraps each GPU kernel launch in a `cuEventRecord` pair and
    /// aggregates elapsed ms per op label at end-of-run. Label matrix
    /// matches OpenCL's `--profile-events` (matmul_qkv, matmul_wo,
    /// matmul_ffn, rms_norm, rope, attention, kv_update, silu_mul,
    /// lm_head) for apples-to-apples Adreno vs Jetson comparison.
    ///
    /// Independent of `--profile` and `--profile-events`. Writes
    /// `results/profile/cuda_embedded_decode_<timestamp>.json`.
    #[arg(long, default_value_t = false)]
    pub cuda_profile: bool,

    /// Per-category sync policy for the cuda-embedded backend. Lets us
    /// bisect which per-op `cuStreamSynchronize()` calls are load-bearing
    /// for correctness on Jetson UMA versus which are pure overhead.
    ///
    /// Values:
    /// - `all` (default): every launch-site sync stays on (pre-bisect
    ///   behaviour, ~28 tok/s on Xavier).
    /// - `none`: every per-op sync suppressed (equivalent to
    ///   `--cuda-defer-sync`; garbage output).
    /// - `llamacpp`: only the CPU-fallback guard stays on (garbage
    ///   output on Jetson UMA — residual `add_assign` loses cache
    ///   coherency without an intra-layer sync).
    /// - `minimal`: bisection-validated minimal correct set
    ///   (`elem_add` + `fallback`; ~34.8 tok/s on Xavier, +6.4 tok/s
    ///   vs `all`).
    /// - `custom:A,B`: comma-separated category names. Recognised
    ///   categories: `elementwise` (expands to `elem_add` +
    ///   `elem_act` + `elem_misc`), `elem_add`, `elem_act`,
    ///   `elem_misc`, `rmsnorm`, `rope`, `matmul`, `kv_scatter`,
    ///   `attention`, `gather`, `fallback`. Only the listed ones
    ///   keep syncing; everything else is deferred.
    ///
    /// `--cuda-defer-sync` still takes precedence when enabled.
    #[arg(long, default_value = "minimal")]
    pub cuda_sync_policy: String,

    /// Allocate weight tensors in device-only memory (`cuMemAlloc`) instead
    /// of UMA pinned host memory (`cuMemHostAlloc`) on Jetson.
    ///
    /// Jetson integrated GPUs expose the CPU DRAM to CUDA kernels through
    /// a pinned host-mapped alias, which gives zero-copy but weak L2 cache
    /// coherency when kernels read and the CPU writes (see llama.cpp
    /// `ggml-cuda.cu:241`, issue #15034). Weights are written once at load
    /// time and then read from every kernel for the rest of the run, so
    /// moving them off the UMA alias is the strongest lever for cache
    /// ordering without losing zero-copy on per-token activations /
    /// KV cache. No-op on discrete GPUs (managed memory already migrates
    /// weights to VRAM on first touch) and on non-CUDA backends.
    #[arg(long, default_value_t = false)]
    pub cuda_weights_device: bool,

    /// Experimental: bundle each decode step's kernel launches into a
    /// single CUDA Graph (captured and replayed once per token).
    ///
    /// Removes per-kernel driver launch overhead (~5 µs × ~400 launches
    /// = ~2 ms/tok on Jetson Xavier). Pays a per-step graph
    /// instantiate cost (~0.3-1 ms on Xavier) — net win is sensitive
    /// to the instantiate overhead actually measured.
    ///
    /// Currently a per-step re-capture baseline. Incompatible with
    /// `--cuda-profile`, `--profile`, and tensor partition; the
    /// inner decode path must not call `synchronize()`, `read_buffer`,
    /// or any CPU fallback while capture is active.
    #[arg(long, default_value_t = false)]
    pub cuda_graph: bool,

    /// Model weight data type (f16 or q4). f16 = no quantization, q4 = Q4_0 quantization at load time.
    #[arg(long, default_value = "f16")]
    pub weight_dtype: String,

    /// One-shot lm_head quantization at load time (`auto` | `none` | `q4_0`).
    ///
    /// Sprint F (2026-04-26): Recovers the +4.6 ms/tok Adreno gap that
    /// dominates "ratio=1.0 mixed" weight-swap regressions. F16 GGUFs ship
    /// lm_head as F16 (~524 MB), while Q4 GGUFs derive it from Q4_0
    /// embed_tokens. Quantizing lm_head once at load time matches the Q4
    /// baseline cost (~3.8 ms/call) without touching the AUF format.
    /// Embed_tokens stays untouched even on tied-weight models. No-op if
    /// lm_head is already Q4_0.
    ///
    /// `auto` (default): quantize when `--secondary-gguf` is set AND lm_head
    /// is currently F16/F32 (production-safe — pure win, no regression on
    /// Q4 baseline because lm_head is already Q4_0 there).
    /// `q4_0`: force quantize regardless of secondary-gguf presence.
    /// `none`: never quantize (legacy/diagnostic behaviour).
    #[arg(long, default_value = "auto")]
    pub quantize_lm_head: String,

    /// KV cache data type (f32, f16, or q4)
    #[arg(long, default_value = "f16")]
    pub kv_type: String,

    /// KV cache format by registry name (KV_FORMATS). 설정 시 `--kv-type` 보다 우선.
    /// 내장(f32/f16/q4_0/q8_0)은 typed 저장. 그 외 등록 format 은 descriptor 가 내장 DType 과
    /// bit-equivalent 면 typed fast path, 아니면 opaque 저장(descriptor-keyed 2026-06-09).
    #[arg(long)]
    pub kv_format: Option<String>,

    /// GATE-C: runtime plugin `.so` paths (repeatable). dlopen'd once at startup and routed
    /// to the dynamic stage/format/backend-cap registries via the `register_kv_formats_v2` /
    /// `register_kv_formats_v2` / `register_backend_caps_v2` entry symbols. Select a loaded
    /// technique by name: `--kv-format <name>` (format), `eviction plugin --name <name>`
    /// (stage), `--backend-cap <name>` (backend capability). Built-in name collisions
    /// fail fast (built-in wins).
    #[arg(long = "load-plugin")]
    pub load_plugin: Vec<std::path::PathBuf>,

    /// Select a backend-capability implementation by registry name — e.g. a quant-window fused
    /// dequant+attention backend — static (linkme `QUANT_ATTN_REGS`) or `--load-plugin`
    /// dlopen'd. The backend-capability axis's analogue of `--kv-format`. OpenCL-only;
    /// unset = the engine's built-in OpenCL implementation.
    #[arg(long = "backend-cap")]
    pub backend_cap: Option<String>,

    // ── Eviction (S-subcmd C2): policy/h2o/d2o/sink/streaming + common
    // 7 params (kv_budget, protected_prefix, memory_threshold_mb,
    // eviction_target_ratio, initial_kv_capacity, min_kv_cache,
    // kv_budget_ratio) moved to EvictionCmd subcommand + EvictionCommonArgs.
    // Existing call sites continue to read via shim accessors on `Args`
    // (see `impl Args` below). ──
    #[clap(flatten)]
    pub eviction_common: EvictionCommonArgs,

    /// Enable resilience manager for adaptive inference.
    /// Legacy generate 기준 flag. argus-cli v1+ 는 default-on 정책이며,
    /// 비활성화는 [`Self::no_resilience`] (`--no-resilience`) 를 사용한다.
    #[arg(long, default_value_t = false)]
    pub enable_resilience: bool,

    /// Disable resilience manager (argus-cli v1+ opt-out).
    /// argus-cli v1 에서는 resilience 가 default-on 이므로 비활성화하려면
    /// 이 flag 를 명시. legacy `generate` binary 는 이 flag 를 무시
    /// (default-off 정책 유지) — argus-cli main 에서만 [`Self::enable_resilience`]
    /// 를 effective 결정한다.
    #[arg(long, default_value_t = false)]
    pub no_resilience: bool,

    /// Pre-allocate dual CPU/GPU buffers for zero-alloc SwitchHw.
    /// Without this flag, only throttle/suspend directives work (no backend switch).
    /// Enables: zero-copy KV memory + weight dual-access rewrap (increases RSS by ~model size).
    #[arg(long, default_value_t = false)]
    pub resilience_prealloc_switch: bool,

    /// Resilience signal transport: "dbus" or "unix:<path>"
    #[arg(long, default_value = "dbus")]
    pub resilience_transport: String,

    // ── Experiment mode ──────────────────────────────
    /// Experiment schedule JSON file (enables experiment mode)
    #[arg(long)]
    pub experiment_schedule: Option<String>,

    /// Experiment output JSONL file path
    #[arg(long)]
    pub experiment_output: Option<String>,

    /// argus-bench: emit a single-line metrics JSON to stdout
    /// (`decode_ms_per_tok`, `prefill_ms`, `peak_kv_mb`, `tokens_per_sec`, …) and route the
    /// human-readable text + metric lines to stderr, so the validation harness can fold stdout
    /// straight into metrics.json. No effect on other bins.
    #[arg(long)]
    pub bench_json: bool,

    /// Number of top-K logits to record per token in experiment mode
    #[arg(long, default_value_t = 10)]
    pub experiment_logits_topk: usize,

    /// System metric sampling interval (N tokens, 0=disabled)
    #[arg(long, default_value_t = 1)]
    pub experiment_sample_interval: usize,

    /// Force greedy sampling (temperature=0) for reproducibility
    #[arg(long, default_value_t = false)]
    pub greedy: bool,

    /// Ignore EOS token and continue generating (for long-running experiments)
    #[arg(long, default_value_t = false)]
    pub ignore_eos: bool,

    /// Target TBT in milliseconds for pacing (0=disabled).
    /// After each decode step, sleeps to maintain the target TBT.
    /// Used for fair resource comparison across different actions at the same QoS.
    #[arg(long, default_value_t = 0.0)]
    pub target_tbt: f64,

    /// Fixed per-token throttle delay in milliseconds (0=disabled).
    /// Unconditional sleep after each decode step — useful for co-execution
    /// simulations without running a Manager. Manager `Throttle` directives
    /// override this value when resilience is enabled.
    #[arg(long, default_value_t = 0)]
    pub throttle_delay_ms: u64,

    /// OpenCL command-queue priority hint (`cl_khr_priority_hints`).
    /// "low" yields GPU scheduling to foreground apps (e.g. games) during
    /// co-execution. Falls back to normal priority with a warning if the
    /// driver does not advertise the extension. Also settable via env var
    /// `OCL_QUEUE_PRIORITY`.
    #[arg(long, value_parser = ["low", "medium", "normal", "high"], default_value = "normal")]
    pub gpu_priority: String,

    /// Path to write per-token TBT JSONL log.
    /// Each line: {"token_idx":N,"tbt_ms":X,"forward_ms":Y,"cache_pos":Z,"pacing_ms":W}
    #[arg(long)]
    pub tbt_log: Option<String>,

    /// KV cache memory layout: "head" (head-major, GPU flash-decode default) or
    /// "seq" (seq-major, CPU). GPU forces "head" (flash decode is HeadMajor-only).
    #[arg(long, value_parser = ["head", "seq"], default_value = "head")]
    pub kv_layout: String,

    /// Override eviction target_ratio from resilience signals (experiment mode).
    /// When set, all Evict actions will use this ratio instead of the strategy default.
    #[arg(long)]
    pub experiment_eviction_ratio: Option<f32>,

    // ── Eval-LL mode (log-likelihood evaluation) ──
    /// Enable log-likelihood evaluation mode (downstream task accuracy)
    #[arg(long, default_value_t = false)]
    pub eval_ll: bool,

    /// Continuation text to evaluate log-likelihood (single task mode)
    #[arg(long)]
    pub eval_continuation: Option<String>,

    /// Path to evaluation batch JSON file: [{"id","prompt","continuation"}, ...]
    #[arg(long)]
    pub eval_batch: Option<String>,

    /// Enable dynamic KV cache quantization for resilience.
    /// Starts with bits=16 (F16-equivalent QuantizedRecentWindowCache) and allows runtime
    /// transition to Q2/Q4/Q8 via kv_quant_dynamic resilience command.
    #[arg(long, default_value_t = false)]
    pub kv_dynamic_quant: bool,

    /// Number of threads for parallel computation.
    /// Default: auto-detect CPU core count.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// Path to reference text file for perplexity evaluation (teacher-forcing).
    /// Measures PPL and collects proxy metrics during eviction.
    #[arg(long)]
    pub ppl: Option<String>,

    /// PPL per-token NLL CSV 출력 경로. 미지정 시 dump 안 함.
    /// CSV columns: phase, token_idx, token_id, nll, swap_state, layers_swapped.
    #[arg(long)]
    pub ppl_nll_csv: Option<std::path::PathBuf>,

    /// PPL prefill 토큰 수 강제 설정 (1..=eval_tokens). 미지정 시 기존 로직
    /// (kv_budget / sliding window / eval_tokens) 그대로. swap 측정 시 decode
    /// loop 을 충분히 길게 돌려야 하므로 이 옵션으로 prefill 을 짧게 만든다.
    /// 예: 1072 token reference 에서 `--ppl-prefill-tokens 32` → prefill 32 +
    /// decode 1040 step.
    #[arg(long)]
    pub ppl_prefill_tokens: Option<usize>,

    /// Comma-separated layer indices to skip (both attn+mlp).
    /// Example: --skip-layers 1,3,5,7
    #[arg(long, value_delimiter = ',')]
    pub skip_layers: Option<Vec<usize>>,

    /// Skip ratio (0.0-1.0). Uses SkipConfig::uniform_init() to select layers.
    #[arg(long)]
    pub skip_ratio: Option<f32>,

    /// forgetting-factor 게이트 지표 덤프 경로(측정 전용, KV roadmap 항목 0 §4.2).
    /// 지정 시 PPL run 의 eviction 직전 스냅샷 + run 종료 시점에 score accumulator importance 에서
    /// BOS/non-BOS ratio + HH(top-k) 집합을 JSON 으로 이 경로에 쓴다. score accumulator 무수정(읽기
    /// 전용). 미지정 시 호출되지 않음(production 무영향). `--score-decay` 와 함께 사용해 forgetting
    /// factor 효과를 비교한다.
    #[arg(long)]
    pub dump_a2sf: Option<std::path::PathBuf>,

    /// Directory for generic diagnostic dumps. Each kind selected by `--dump`
    /// writes `<dir>/<kind>.jsonl`, one JSON record per question. Required when
    /// `--dump` is set. Read-only diagnostics — no effect on scoring (INV-147).
    #[arg(long)]
    pub dump_dir: Option<std::path::PathBuf>,

    /// Comma-separated diagnostic dumps to emit (e.g. `--dump answer_attention`,
    /// or `--dump a,b`). Each kind writes `<dump-dir>/<kind>.jsonl`. Generic by
    /// design: a new dump kind registers a name, not a new CLI flag. Validated at
    /// startup against the known kinds (`session::eval::dump::KNOWN_DUMP_KINDS`).
    #[arg(long, value_delimiter = ',')]
    pub dump: Vec<String>,

    /// `--dump answer_attention_steps`: emit the full per-head trajectory
    /// `[step][layer][head][token]` instead of the head-mean `[step][layer][token]` default.
    /// Large (≈ `num_attention_heads ×` the head-mean dump) — the dump logs the size at startup.
    /// No effect unless `answer_attention_steps` is requested. Composes with `--..-scope`.
    #[arg(long)]
    pub answer_attention_steps_per_head: bool,

    /// `--dump answer_attention_steps` capture scope. `decode` (default) keeps the trailing
    /// gold-answer rows over the context columns `[0, prompt_len)` (schema 1). `full` keeps EVERY
    /// forward row (prefill then decode) over the full key axis `[0, seq_len)` — the whole
    /// lower-triangular causal matrix (schema 2; per-record `row` / `phase` / `n_valid_keys`).
    /// `full` is quadratic in seq_len (the dump logs the size at startup), intended for short
    /// diagnostic benches. No effect unless `answer_attention_steps` is requested; composes with
    /// `--answer-attention-steps-per-head`.
    #[arg(
        long,
        default_value = "decode",
        value_parser = clap::builder::PossibleValuesParser::new(["decode", "full"])
    )]
    pub answer_attention_steps_scope: String,

    /// `--dump answer_attention_steps` (decode scope): also emit the *predicting* row — the query
    /// at `prompt_len - 1` whose logits DECIDE the first gold token — as one extra record per
    /// question (`step: -1`, `query_role: "predicting_row"`). The default decode window starts at
    /// `prompt_len` (the gold token's OWN row), so this row is otherwise never dumped. No-op under
    /// `--answer-attention-steps-scope full` (every row already dumped) or when `prompt_len == 0`.
    /// Default off → the existing records are unchanged. No effect unless `answer_attention_steps`
    /// is requested.
    #[arg(long)]
    pub answer_attention_steps_predict_row: bool,

    /// `--dump aperturb`: also write, per question, the exact `(query rows, K, V)` the metric
    /// measured, as one little-endian f32 file per question in this directory. Large (the whole
    /// resident cache), and the only way to tell "the metric disagrees" from "the forward produced
    /// different tensors" when checking against an external implementation. No effect unless
    /// `aperturb` is requested.
    #[arg(long)]
    pub aperturb_tensor_dir: Option<std::path::PathBuf>,

    /// Load the output-projection basis from this file instead of factoring the weights, and fail if
    /// the file does not belong to this model. Written by `--aperturb-basis-out`. The decomposition
    /// is a model constant that costs 28 s at 1B and 12 min at 8B, and nothing stored it, so every
    /// run paid it again; on a phone, where factoring is not practical at all, this is the only way
    /// the metric runs. Load-only by design: a missing file or a header that disagrees is an error,
    /// never a quiet fall back to computing it. Read by both `--dump aperturb` and
    /// `--aperturb-select`; not combinable with `--aperturb-basis-out`.
    #[arg(long)]
    pub aperturb_basis: Option<std::path::PathBuf>,

    /// Factor the output projection as usual, then write the basis to this file for
    /// `--aperturb-basis` to load. Little-endian, 1 MB for a 1B model and 8 MB for an 8B one at the
    /// default rank, and portable — produce it on a host and ship it with the application. Honored
    /// by both `--dump aperturb` and `--aperturb-select`.
    #[arg(long)]
    pub aperturb_basis_out: Option<std::path::PathBuf>,

    /// Let the engine choose its own KV compression: a comma-separated pool of registered technique
    /// names (e.g. `h2o,sliding,streaming`). A `KvCompress { budget }` from the Manager then asks each of
    /// them what it would retain at that budget, scores those retained sets by how far each moves
    /// the model's own attention output, and applies the one that moves it least — instead of
    /// applying the single `eviction <policy>` the CLI configured.
    ///
    /// A candidate must be a technique that acts where the budget arrives — mid-decode, at
    /// `KvMutate` — and must read only what the planner can supply there. A prefill-end stage
    /// (PyramidKV/SnapKV) is refused rather than ranked on its degraded fallback.
    ///
    /// Needs `--aperturb-basis` unless a startup decomposition is acceptable (28 s at 1B, 12 min at
    /// 8B). Off by default: the forward then never captures query rows and the compression path is
    /// byte-identical to before.
    #[arg(long, value_delimiter = ',')]
    pub aperturb_select: Vec<String>,

    /// eval-LL KV eviction timing (when eviction fires and which importance drives
    /// it). `post_prefill_probe` (default) = today's behavior: full prefill, a
    /// post-question probe, one eviction — the importance is query-informed.
    /// `prefill_end` = accumulate per-step context importance during a token-by-token
    /// prefill, suppress the probe, then evict query-agnostically. See
    /// [`crate::session::eval::EvictTiming`]. Use the parsed accessor
    /// [`Args::evict_timing`].
    #[arg(
        long,
        default_value = crate::session::eval::EvictTiming::POST_PREFILL_PROBE,
        value_parser = clap::builder::PossibleValuesParser::new(
            crate::session::eval::EvictTiming::CLI_VALUES
        )
    )]
    pub evict_timing: String,

    /// Start an interactive multi-turn chat REPL (Llama 3.2 Instruct / Qwen2).
    /// Uses standard (non-quant-window, non-offload) forward path.
    #[arg(long, default_value_t = false)]
    pub chat: bool,

    /// Optional system prompt injected as the first turn when --chat is set.
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Optional Unix domain socket path. When set, chat mode also accepts
    /// newline-delimited user messages from this socket in addition to stdin,
    /// and streams assistant replies back (terminated by 0x04).
    #[arg(long)]
    pub chat_socket: Option<String>,

    /// Optional TCP listen address (e.g. "127.0.0.1:7878"). Same protocol
    /// as --chat-socket: newline-delimited input, assistant reply bytes
    /// streamed back, 0x04 EOT delimiter per turn. Can be combined with
    /// --chat-socket; both listeners feed the same chat loop.
    #[arg(long)]
    pub chat_tcp: Option<String>,

    /// argus-chat: HTTP listen address for the OpenAI-compatible API
    /// (`POST /v1/chat/completions`). Defaults to `127.0.0.1:8080` when running
    /// the server. Ignored by other binaries.
    #[arg(long)]
    pub listen: Option<String>,

    /// argus-chat: run an interactive stdin chat REPL instead of the HTTP server.
    #[arg(long, default_value_t = false)]
    pub interactive: bool,

    /// Directory used by `KvOffload` directives to write out the LRU prefix
    /// of the KV cache. When set, `CacheManager::enable_swap()` registers a
    /// disk-backed `SwapHandler`; without it the `KvOffload` directive is
    /// a warn-only no-op. `RestoreDefaults` triggers recall of offloaded data.
    #[arg(long)]
    pub swap_dir: Option<std::path::PathBuf>,

    /// Optional secondary GGUF path for runtime weight swap (Phase 2).
    /// When specified together with `--force-swap-ratio`, the engine swaps
    /// decoder layer weights from the primary dtype to the secondary dtype
    /// immediately before generation starts.
    /// When omitted, the weight swap path is disabled (ENG-DAT-C09).
    #[arg(long)]
    pub secondary_gguf: Option<std::path::PathBuf>,

    /// AUF primary backend variant 선택 (W-AUF-1 C4).
    /// AUF primary가 아닐 때 무시됨. default: auto.
    #[arg(long, default_value = "auto", value_parser = parse_primary_variant)]
    pub primary_variant: PrimaryVariantArg,

    /// AUF primary dtype 선택 (W-AUF-1 C4).
    /// AUF primary가 아닐 때 무시됨. default: auto (META.default_dtype 우선).
    #[arg(long, default_value = "auto", value_parser = parse_primary_dtype)]
    pub primary_dtype: PrimaryDtypeArg,

    /// AUF TOKENIZER에 eos_id가 비어있을 때 fallback override (W-AUF-1 C5).
    #[arg(long)]
    pub eos_token_id: Option<u32>,

    /// AUF TOKENIZER에 bos_id가 비어있을 때 fallback override (W-AUF-1 C5).
    #[arg(long)]
    pub bos_token_id: Option<u32>,

    /// AUF self-secondary 자동 활성 비활성 (W-AUF-2). 디버그/벤치마크용.
    #[arg(long, default_value_t = false)]
    pub no_self_secondary: bool,

    /// Secondary dtype selection for AUF-backed weight swap (ENG-ALG-225, Sprint D).
    ///
    /// Controls which dtype entry is selected from a multi-dtype AUF file:
    ///   auto  — automatically select the best candidate dtype (default).
    ///           If META.default_dtype is set, that is used; otherwise the first
    ///           available candidate is picked.
    ///   q4_0  — explicitly select Q4_0 entries.
    ///   f16   — explicitly select F16 entries.
    ///   f32   — explicitly select F32 entries.
    ///
    /// Ignored for GGUF-backed secondaries (GGUF files carry a single dtype).
    /// Adreno SOA backend rejects f16 (SOA layout is Q4_0-only).
    #[arg(long, default_value = "auto", value_parser = parse_secondary_dtype)]
    pub secondary_dtype: SecondaryDtypeArg,

    /// AUF weights variant 선택 ("auto" | "aos" | "soa").
    ///
    /// `auto` (기본): feature flag 기반 preferred variant 우선 + AUF에 없으면
    /// CpuAos 자동 폴백. OpenCL build에선 AdrenoSoa 우선.
    ///
    /// `aos`: 강제 AOS (`WEIGHTS_CPU_AOS` / `WEIGHTS_CUDA_AOS`). swap 후
    /// `switch_hw cpu` / partition lazy-map / CPU forward가 정상 동작.
    /// GPU TBT는 SOA 대비 30~50% 저하 (Adreno 830 실측).
    ///
    /// `soa`: 강제 SOA (`WEIGHTS_ADRENO_SOA`, OpenCL 전용). 가장 빠르지만
    /// swap 후 host-pointer 부재로 switch_hw cpu / partition 호환 불가.
    ///
    /// GGUF secondary에선 무시됨.
    #[arg(long, default_value = "auto", value_parser = parse_secondary_layout)]
    pub secondary_layout: SecondaryLayoutArg,

    /// Explicit path to tokenizer.json. When omitted, the tokenizer is
    /// resolved automatically via the GGUF basename (e.g.
    /// `<dir>/<stem>.tokenizer.json`, then `<dir>/<stem-without-quant>.tokenizer.json`,
    /// then the legacy `<dir>/tokenizer.json` fallback). Required when
    /// multiple models share the same directory (e.g. `/data/local/tmp/`)
    /// because the legacy fallback can pick up a sibling model's tokenizer
    /// and silently produce garbage outputs.
    #[arg(long)]
    pub tokenizer_path: Option<std::path::PathBuf>,

    /// §4.2 decode-X experiment (EuroSys'27). When > 0, the QCF-dump warmup
    /// workflow runs `N` greedy-generation decode steps after the regular
    /// prefill and caches the per-layer hidden state at each decode step in
    /// a fresh collector. Two extra F5 vectors land in the dump JSON:
    /// - `direct_attn_f5_decode_only`: X = decode-only raws (T = N).
    /// - `direct_attn_f5_prefill_decode`: X = concat(prefill raws, decode raws) (T = 256 + N).
    ///
    /// The regular `direct_attn_f5` (prefill X, T = 256) is always written.
    /// Decode token 0 = argmax of prefill's final logits; subsequent decode
    /// tokens = argmax of each previous decode-step logits (greedy).
    /// Only meaningful when a secondary GGUF (Q4) is loaded.
    #[arg(long, default_value_t = 0)]
    pub decode_x_steps: usize,

    /// Eagerly prefault the secondary weight file at model load to remove
    /// per-swap prefault stage cost. Memory commit ≈ AUF size (e.g. 1.2 GB
    /// for Qwen2.5-1.5B Q4_0). Default off; set when --secondary-gguf is
    /// present and on-device app has memory headroom.
    ///
    /// When enabled: immediately after model weights are loaded, the full
    /// secondary weight region is touched (madvise WILLNEED + explicit
    /// page-touch). Subsequent swap invocations find all pages already in
    /// the page cache, eliminating the ~328 ms cold-fault stage measured on
    /// Galaxy S25 (§3.1, swap_overhead_s25.md).
    ///
    /// When `--secondary-gguf` is absent this flag is silently ignored.
    #[arg(long, default_value_t = false)]
    pub eager_prefault_secondary: bool,

    /// Top-level subcommand wrapper.
    ///
    /// `eviction <policy>` form is the only currently registered
    /// subcommand. Omitting the subcommand ≡ `EvictionCmd::None`
    /// (no eviction). See [`crate::session::cli::TopLevelCmd`] and
    /// [`crate::session::cli::EvictionCmd`].
    #[command(subcommand)]
    pub eviction: Option<TopLevelCmd>,

    // ── KV mode subcommand (S-subcmd C4) ─────────────────────────────────
    #[clap(flatten)]
    pub kv_mode_args: KvModeArgs,

    // ── Session prefix KV cache (ENG-080~085) ──────────────────
    /// Save KV prefix cache to this path after prefill (ENG-085).
    ///
    /// Snapshot is taken immediately after prefill and before any eviction
    /// (INV-189). Supported formats: F32/F16/Q4_0 (StandardFormat).
    /// quant-window/opaque caches are silently skipped (no error, no-cache fallback).
    ///
    /// Uses atomic write (`<path>.tmp` → rename) to prevent corruption.
    /// Failure (permissions, disk full) is reported as a warning; the run
    /// continues normally.
    #[arg(long)]
    pub save_prefix_cache: Option<String>,

    /// Restore KV prefix cache from this path at session start (ENG-085).
    ///
    /// Attempts to restore the cached KV state. If the file is missing,
    /// has an incompatible model/format/tokenizer hash, or the token_ids
    /// do not match the current prompt prefix — falls back to fresh
    /// prefill silently (Ok(None), no panic, no error — INV-190).
    ///
    /// On hit: if `token_count == prompt.len()` prefill is completely
    /// skipped; otherwise `prompt[token_count..]` is prefilled from
    /// `start_pos = token_count`.
    #[arg(long)]
    pub prefix_cache: Option<String>,
}

/// Shim accessors for the eviction subcommand + flatten common args.
///
/// Existing 175+ call sites (`args.eviction_policy`, `args.h2o_keep_ratio`,
/// `args.kv_budget`, ...) read through these methods so the C2 commit
/// changes only `cli/mod.rs`. Call sites migrate to direct enum match in C3.
impl Args {
    /// Diagnostic dump kinds requested via `--dump <kind>[,<kind>...]` (in CLI
    /// order). Empty when no dump is selected. Validated against the known kinds
    /// at startup; see [`crate::session::eval::dump`].
    pub fn dump_kinds(&self) -> &[String] {
        &self.dump
    }

    /// True if `kind` was requested via `--dump`.
    pub fn dump_enabled(&self, kind: &str) -> bool {
        self.dump.iter().any(|k| k == kind)
    }

    /// Parsed `--evict-timing` mode. The raw string is validated by clap's
    /// `value_parser`, so the parse is infallible here.
    pub fn evict_timing(&self) -> crate::session::eval::EvictTiming {
        crate::session::eval::EvictTiming::from_cli(&self.evict_timing)
            .expect("--evict-timing validated by clap value_parser")
    }

    /// Output path for a dump `kind` = `<dump-dir>/<kind>.jsonl`. `None` when
    /// `--dump-dir` is unset (the eval guard requires it whenever `--dump` is set).
    pub fn dump_path(&self, kind: &str) -> Option<std::path::PathBuf> {
        self.dump_dir
            .as_ref()
            .map(|dir| dir.join(format!("{kind}.jsonl")))
    }

    /// Whether `--dump answer_attention_steps` should capture the full lower-triangular matrix
    /// (`--answer-attention-steps-scope full`) instead of the decode-only default. Centralizes
    /// the `"full"` scope string (clap validates the value set).
    pub fn answer_attention_steps_full(&self) -> bool {
        self.answer_attention_steps_scope == "full"
    }

    /// KV mode name (runtime string — resolved against the engine KV-mode registry
    /// `KV_MODES` at the build funnel, not a closed clap enum). Mirrors
    /// `eviction_policy() -> &str`.
    pub fn effective_kv_mode(&self) -> &str {
        &self.kv_mode_args.kv_mode
    }

    /// 선택적 KV read stage 이름. 미지정 = None(full read).
    pub fn effective_read_stage(&self) -> Option<&str> {
        self.kv_mode_args.read_stage.as_deref()
    }

    /// quant-window quantization bits.
    pub fn effective_quant_window_bits(&self) -> u8 {
        self.kv_mode_args.quant_window_bits
    }

    /// quant-window residual buffer size.
    pub fn effective_quant_window_residual_size(&self) -> usize {
        self.kv_mode_args.quant_window_residual_len
    }

    /// Offload storage backend. Returns `""` unless the active mode owns an offload
    /// cache container (`ModeCaps.supports_offload`) — reads the declared cap instead
    /// of matching a concrete `KvMode::Offload`.
    pub fn effective_kv_offload_storage(&self) -> String {
        if crate::session::mode::mode_caps(&self.kv_mode_args.kv_mode)
            .is_some_and(|c| c.supports_offload)
        {
            self.kv_mode_args.kv_offload_storage.clone()
        } else {
            String::new()
        }
    }

    /// ENG-RPCMEM-041 / INV-RPCMEM-006: effective `--opencl-rpcmem` 값.
    ///
    /// Sprint 2b: qnn_oppkg backend 제거됨. `--backend qnn_oppkg | qnngpu` 는
    /// 실제 backend init 에서 unknown backend 로 bail 하므로 이 분기는 production
    /// 경로에서 unreachable 하다. INV-RPCMEM-006 spec test 호환을 위해 보존.
    pub fn effective_opencl_rpcmem(&self) -> bool {
        if self.backend == "qnn_oppkg" || self.backend == "qnngpu" {
            false
        } else {
            self.opencl_rpcmem
        }
    }

    /// Returns the nested `EvictionCmd` policy, unwrapping the
    /// `TopLevelCmd::Eviction` wrapper. `None` if no subcommand given.
    fn current_policy(&self) -> Option<&EvictionCmd> {
        match &self.eviction {
            Some(TopLevelCmd::Eviction { policy }) => Some(policy),
            None => None,
        }
    }

    pub fn eviction_policy(&self) -> &str {
        self.current_policy()
            .map(|e| e.policy_name())
            .unwrap_or("none")
    }

    /// Look up a `--set key=value` from the active `Plugin` policy's blob. `None` for
    /// `EvictionCmd::None` or a missing key. The engine-owned POD keys (window/sink/recent_window/
    /// keep_ratio) + the h2o accumulator keys (tracked_layers/decay/raw_scores) are parsed off this
    /// by the accessors below; technique-private keys stay opaque and flow through [`stage_args`].
    fn plugin_set(&self, key: &str) -> Option<&str> {
        match self.current_policy() {
            Some(EvictionCmd::Plugin(p)) => p
                .sets
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str()),
            _ => None,
        }
    }

    pub fn eviction_window(&self) -> usize {
        self.plugin_set("window")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
    }

    pub fn sink_size(&self) -> usize {
        self.plugin_set("sink")
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    }

    pub fn streaming_window(&self) -> usize {
        self.plugin_set("recent_window")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }

    /// The active stage's keep-ratio (heavy-hitter fraction), read from `--set keep_ratio=`;
    /// defaults to 0.5. NOTE (B1-3): the former typed `eviction d2o` defaulted this to 0.75
    /// (paper 3:1) — under the generic CLI, d2o users must pass `--set keep_ratio=0.75` explicitly.
    pub fn keep_ratio(&self) -> f32 {
        self.plugin_set("keep_ratio")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5)
    }

    pub fn h2o_tracked_layers(&self) -> usize {
        self.plugin_set("tracked_layers")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }

    /// Score accumulator decay factor (= forgetting-factor α 의 1 − α). accumulator 생성자(`new`/`new_gqa`)의
    /// 5번째 인자로 흐른다.
    ///
    /// score-decay 측정(arch/kv_roadmap_item0_measurement.md §4.2): `--score-decay` 측정 flag 가 > 0.0 이면
    /// 정책 무관하게 그 값을 우선한다(forgetting factor 주입). 0.0(기본) 이면 정책 자체 decay(heavy-hitter
    /// `--decay`)를 그대로 반환 → flag 도입 전 경로 **bit-identical**(누적 로직 무수정, 주입만 추가).
    pub fn h2o_decay(&self) -> f32 {
        let score_decay = self.eviction_common.score_decay;
        if score_decay > 0.0 {
            return score_decay.clamp(0.0, 1.0);
        }
        self.plugin_set("decay")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    }

    pub fn h2o_raw_scores(&self) -> bool {
        matches!(self.plugin_set("raw_scores"), Some("true") | Some("1"))
    }

    /// The absolute heavy-hitter budget for faithful H2O (`--set hh_size=`). `None` if unset.
    pub fn h2o_hh_size(&self) -> Option<usize> {
        self.plugin_set("hh_size").and_then(|v| v.parse().ok())
    }

    /// The absolute recency budget for faithful H2O (`--set recent_size=`). `None` if unset.
    pub fn h2o_recent_size(&self) -> Option<usize> {
        self.plugin_set("recent_size").and_then(|v| v.parse().ok())
    }

    /// EXPLICIT-REQUIRED guard for faithful H2O: both `hh_size` and `recent_size` must be supplied via
    /// `--set` (the budget IS the policy — there is no default). Returns a clean error instead of a
    /// deep panic at stage construction. No-op for any non-`h2o` policy. Call once at session setup.
    pub fn require_h2o_budgets(&self) -> anyhow::Result<()> {
        // A candidate in the `--aperturb-select` pool is a technique this run may apply, so it
        // needs its budgets on the same terms as the configured policy — otherwise h2o would be
        // ranked, and possibly chosen, on budgets nobody set.
        let h2o_in_play = self.eviction_policy() == "h2o"
            || self.aperturb_select.iter().any(|n| n.trim() == "h2o");
        if h2o_in_play && (self.h2o_hh_size().is_none() || self.h2o_recent_size().is_none()) {
            anyhow::bail!(
                "'h2o' requires explicit budgets: pass \
                 `--set hh_size=<N> --set recent_size=<M>` (faithful H2O keeps hh_size heavy hitters \
                 + recent_size recent tokens; there is no default)."
            );
        }
        Ok(())
    }

    /// heavy-hitter verbose debug output — moved to env var `LLMRS_H2O_DEBUG`
    /// (no longer a CLI flag).
    pub fn h2o_debug(&self) -> bool {
        std::env::var("LLMRS_H2O_DEBUG").is_ok()
    }

    /// The active stage's technique-private parameters as an opaque `(key, val)` blob for
    /// `make_stage_with_args(name, …)` — the raw `--set key=value` pairs of `eviction plugin
    /// --name X --set k=v`. The engine routes them without knowing any plugin's private knobs (each
    /// plugin parses its own keys in `from_args`, ignoring the rest), e.g. d2o reads
    /// `ema_beta`/`merge_e`/`layer_alloc`/`protected_layers`/`merge_axis`, rkv reads `lambda`.
    /// `EvictionCmd::None` → empty. (B1-3: the former per-technique typed mirrors are gone — d2o's
    /// `target_ratio` is no longer auto-injected from `--eviction-target-ratio`; d2o budget now
    /// follows `--set keep_ratio=` / its own defaults.)
    pub fn stage_args(&self) -> Vec<(String, String)> {
        match self.current_policy() {
            Some(EvictionCmd::Plugin(p)) => p.sets.clone(),
            _ => Vec::new(),
        }
    }

    // ── EvictionCommonArgs shim (flatten field 호출처 호환) ──
    pub fn kv_budget(&self) -> usize {
        self.eviction_common.kv_budget
    }
    pub fn kv_budget_ratio(&self) -> f32 {
        self.eviction_common.kv_budget_ratio
    }
    pub fn protected_prefix(&self) -> Option<usize> {
        self.eviction_common.protected_prefix
    }
    pub fn memory_threshold_mb(&self) -> usize {
        self.eviction_common.memory_threshold_mb
    }
    pub fn eviction_target_ratio(&self) -> f32 {
        self.eviction_common.eviction_target_ratio
    }
    pub fn initial_kv_capacity(&self) -> usize {
        self.eviction_common.initial_kv_capacity
    }
    pub fn min_kv_cache(&self) -> usize {
        self.eviction_common.min_kv_cache
    }

    /// argus-chat HTTP listen address (`--listen`), defaulting to `127.0.0.1:8080`.
    pub fn listen_addr(&self) -> String {
        self.listen
            .clone()
            .unwrap_or_else(|| "127.0.0.1:8080".to_string())
    }
}
