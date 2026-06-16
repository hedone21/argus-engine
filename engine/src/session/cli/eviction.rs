//! Eviction policy CLI subcommand (S-subcmd C1, 2026-05-19).
//!
//! Replaces the polic-specific flag flat namespace (`--eviction-policy h2o
//! --h2o-keep-ratio 0.5 --h2o-decay 0.0 ...`) with a clap subcommand
//! enum. Each variant exposes only its own parameters, so policy
//! additions (SnapKV, KVSwap, ...) no longer balloon the global CLI
//! surface.
//!
//! Variant common parameters (kv_budget / protected_prefix /
//! memory_threshold_mb / eviction_target_ratio / initial_kv_capacity /
//! min_kv_cache / kv_budget_ratio) live in [`EvictionCommonArgs`] and
//! are `#[clap(flatten)]`'d at the binary's top-level `Args`.
//!
//! Wired in by C2 (cli/mod.rs Args integration).

use clap::{Args, Subcommand};

/// Top-level wrapper exposing `eviction <policy>` as a single clap
/// subcommand group. clap derive registers each [`EvictionCmd`] variant
/// directly as a subcommand, so without this wrapper the CLI would be
/// `generate ... h2o --keep-ratio 0.5`. The wrapper produces the
/// `generate ... eviction h2o --keep-ratio 0.5` form documented in
/// `docs/USAGE.md` and `docs/35_experiment_runner_guide.md`.
#[derive(Subcommand, Debug, Clone)]
pub enum TopLevelCmd {
    /// KV cache eviction policy (variant chosen via nested subcommand).
    Eviction {
        #[command(subcommand)]
        policy: EvictionCmd,
    },
}

/// Eviction policy selection (nested under [`TopLevelCmd::Eviction`]).
///
/// CLI usage:
/// ```text
/// generate -m model.gguf eviction sliding --window 1024
/// generate -m model.gguf eviction h2o --keep-ratio 0.5 --tracked-layers 0
/// generate -m model.gguf eviction d2o --keep-ratio 0.75 --ema-beta 0.7
/// ```
///
/// Omitting the subcommand is equivalent to [`EvictionCmd::None`] —
/// no eviction; the KV cache grows up to `--max-seq-len`.
#[derive(Subcommand, Debug, Clone)]
pub enum EvictionCmd {
    /// No eviction (default). KV cache grows up to --max-seq-len.
    None,

    /// Sliding window — retain the most recent N tokens.
    Sliding(SlidingArgs),

    /// StreamingLLM — keep `sink` initial tokens plus a recent window.
    Streaming(StreamingArgs),

    /// H2O — heavy hitter selection (Round 14 default for Llama 3.2 1B).
    H2o(H2oArgs),

    /// H2O+ — per-head GQA-aware H2O variant.
    H2oPlus(H2oArgs),

    /// D2O — Dynamic Discriminative Operations (arXiv 2406.13035).
    D2o(D2oArgs),

    /// CAOTE — value-aware criticality `a_i·‖v_i − o_h‖`.
    ///
    /// feature `caote` 플러그인 install 시에만 노출된다(미설치 = subcommand 부재).
    /// 튜닝 파라미터 없음 — V 는 ctx.tensor(Value)로, 가중치 a_i 는 importance 로 자동.
    #[cfg(feature = "caote")]
    Caote,

    /// R-KV — cosine redundancy + importance joint eviction (arXiv 2505.24133).
    ///
    /// KV roadmap 항목 0 측정 프로토타입(arch/kv_roadmap_item0_measurement.md §4.1).
    /// feature `rkv` 측정 게이트 install 시에만 노출된다(미설치 = subcommand 부재 = production
    /// 표면 불변). λ(redundancy/importance fusion)만 측정 조정 가능(기본 0.1).
    #[cfg(feature = "rkv")]
    Rkv(RkvArgs),

    /// Plugin-supplied eviction stage, selected by registry name.
    ///
    /// The stage-axis analogue of `--kv-format <name>`: any technique crate registered
    /// statically (linkme `KV_CACHE_STAGES`) or dynamically (`--load-plugin`) is selectable
    /// by name with no engine edit. CLI form: `eviction plugin --name <stage>`.
    Plugin(PluginArgs),
}

impl EvictionCmd {
    /// Canonical policy name — the stage registry key (a built-in like "h2o", or a plugin
    /// name from `eviction plugin --name <name>`). Also used by manager IPC, the lua policy
    /// DSL, and JSON dumps, so downstream code can keep matching on the policy name.
    pub fn policy_name(&self) -> &str {
        match self {
            EvictionCmd::None => "none",
            EvictionCmd::Sliding(_) => "sliding",
            EvictionCmd::Streaming(_) => "streaming",
            EvictionCmd::H2o(_) => "h2o",
            EvictionCmd::H2oPlus(_) => "h2o_plus",
            EvictionCmd::D2o(_) => "d2o",
            #[cfg(feature = "caote")]
            EvictionCmd::Caote => "caote",
            #[cfg(feature = "rkv")]
            EvictionCmd::Rkv(_) => "rkv",
            // Plugin name is a runtime String (borrowed for &self's lifetime — this is why
            // policy_name/eviction_policy return &str rather than &'static str).
            EvictionCmd::Plugin(a) => a.name.as_str(),
        }
    }
}

/// Plugin eviction-stage selector (the stage-axis analogue of `--kv-format <name>`).
/// `--name <stage>` is resolved against the static `KV_CACHE_STAGES` + dynamic
/// (`--load-plugin`) registry via `make_stage`.
#[derive(Args, Debug, Clone)]
pub struct PluginArgs {
    /// Registry name of the eviction stage to select.
    #[arg(long = "name")]
    pub name: String,

    /// Technique-private parameter, repeatable: `--set key=value`. The engine routes these opaquely
    /// into the selected stage's `make_with_args` blob — the plugin parses/validates/defaults its own
    /// keys and ignores the rest, so the engine knows none of any plugin's private knobs. Example:
    /// `eviction plugin --name d2o --set ema_beta=0.7 --set merge_axis=value_only`.
    #[arg(long = "set", value_parser = parse_kv)]
    pub sets: Vec<(String, String)>,
}

/// clap value-parser for `--set key=value`. Splits on the FIRST `=` (so values may contain `=`).
fn parse_kv(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((k, v)) if !k.is_empty() => Ok((k.to_string(), v.to_string())),
        _ => Err(format!("expected key=value, got '{s}'")),
    }
}

/// R-KV 측정 프로토타입 파라미터(feature `rkv`). λ 만 노출 — α/τ 는 측정 상수(rkv_stage.rs).
#[cfg(feature = "rkv")]
#[derive(Args, Debug, Clone)]
pub struct RkvArgs {
    /// fusion 가중치 λ (Z = λ·I − (1−λ)·R). 0.1 = redundancy 지배(논문 기본).
    #[arg(long, default_value_t = 0.1)]
    pub lambda: f32,
}

#[derive(Args, Debug, Clone)]
pub struct SlidingArgs {
    /// Tokens to retain in the sliding window.
    #[arg(long, default_value_t = 1024)]
    pub window: usize,
}

#[derive(Args, Debug, Clone)]
pub struct StreamingArgs {
    /// Attention sink tokens to preserve at the start.
    #[arg(long, default_value_t = 4)]
    pub sink: usize,
    /// Recent window size. 0 = auto (`kv_budget - sink`).
    #[arg(long, default_value_t = 0)]
    pub recent_window: usize,
}

#[derive(Args, Debug, Clone)]
pub struct H2oArgs {
    /// Fraction of tokens kept as heavy hitters (0.0–1.0).
    /// 0.0 → equivalent to Sliding; Round 14 confirmed this as the
    /// optimal H2O regime for Llama 3.2 1B.
    #[arg(long, default_value_t = 0.5)]
    pub keep_ratio: f32,

    /// Number of final transformer layers to track for importance scores.
    /// 0 = all layers.
    #[arg(long, default_value_t = 0)]
    pub tracked_layers: usize,

    /// EMA decay factor for cumulative importance scores per step
    /// (0.0 = no decay).
    #[arg(long, default_value_t = 0.0)]
    pub decay: f32,

    /// Disable time-normalized scoring (use raw cumulative SUM).
    /// By default H2O / H2O+ time-normalize to remove cumulative bias.
    #[arg(long, default_value_t = false)]
    pub raw_scores: bool,
}

#[derive(Args, Debug, Clone)]
pub struct D2oArgs {
    /// Heavy-hitter keep ratio (paper default 0.75 = 3:1).
    #[arg(long, default_value_t = 0.75)]
    pub keep_ratio: f32,

    /// EMA smoothing factor β for threshold update
    /// (paper Eq.10, default 0.7).
    #[arg(long, default_value_t = 0.7)]
    pub ema_beta: f32,

    /// Eq.11 normalisation constant `e` controlling retained token's
    /// self-weight (paper default 0.1).
    #[arg(long, default_value_t = 0.1)]
    pub merge_e: f32,

    /// Enable D2O layer-level dynamic allocation
    /// (per-layer attention variance from prefill).
    #[arg(long, default_value_t = false)]
    pub layer_alloc: bool,

    /// Protected layer indices for D2O layer allocation (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub protected_layers: Option<Vec<usize>>,

    /// Weighted-merge axis (WeightedKV ablation, KV 로드맵 항목 2).
    /// `both` (default) = uniform K·V merge (구 동작 bit-identical).
    /// `value_only` = WeightedKV (K discard + V-only weighted merge).
    #[arg(long, default_value = "both", value_parser = ["both", "key_only", "value_only"])]
    pub merge_axis: String,
}

/// Variant-independent eviction parameters.
///
/// Flattened into the binary's top-level `Args` because every policy
/// (and the manager IPC path) reads these regardless of which variant
/// is active.
#[derive(Args, Debug, Clone)]
pub struct EvictionCommonArgs {
    /// Maximum KV cache budget in tokens. Evicts when cache_pos exceeds
    /// this. 0 = no budget limit (default).
    #[arg(long, default_value_t = 0)]
    pub kv_budget: usize,

    /// KV cache budget as ratio of prompt length (0.0–1.0).
    /// When > 0, overrides --kv-budget per question. Matches H2O paper
    /// evaluation methodology.
    #[arg(long, default_value_t = 0.0)]
    pub kv_budget_ratio: f32,

    /// Number of prefix tokens protected from eviction.
    /// Defaults to 4 for score-based policies (h2o, h2o_plus, d2o)
    /// and prompt length for sliding.
    #[arg(long)]
    pub protected_prefix: Option<usize>,

    /// Memory threshold in MB below which eviction triggers.
    #[arg(long, default_value_t = 256)]
    pub memory_threshold_mb: usize,

    /// Target ratio of cache to keep when evicting (0.1–0.99).
    #[arg(long, default_value_t = 0.75)]
    pub eviction_target_ratio: f32,

    /// Initial KV cache capacity in tokens.
    /// 0 = auto (prompt length rounded up to power of 2, min 128).
    #[arg(long, default_value_t = 0)]
    pub initial_kv_capacity: usize,

    /// Minimum KV cache size in tokens. Eviction will not reduce cache
    /// below this.
    #[arg(long, default_value_t = 256)]
    pub min_kv_cache: usize,

    /// A2SF forgetting factor 측정용 score decay (arXiv 2407.20485, A2SF α = 1 − decay).
    /// KV roadmap 항목 0 측정(arch/kv_roadmap_item0_measurement.md §4.2) 전용 modifier.
    ///
    /// **기본 0.0 = 미주입** → accumulator 생성자 decay 인자가 정책 자체 값(H2O `--decay`)으로
    /// 결정 → flag 도입 전 경로 bit-identical. > 0.0 일 때만 정책 무관하게 score accumulator 에
    /// decay 를 주입한다(`begin_step()` 누적 로직 무수정 — 주입 경로만). production 무관.
    #[arg(long, default_value_t = 0.0)]
    pub score_decay: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Wrapper struct so we can invoke clap's parser on `EvictionCmd`.
    #[derive(Parser, Debug)]
    struct Wrap {
        #[command(subcommand)]
        ev: Option<EvictionCmd>,
    }

    fn parse(args: &[&str]) -> Wrap {
        let mut full = vec!["test"];
        full.extend_from_slice(args);
        Wrap::try_parse_from(full).expect("parse")
    }

    #[test]
    fn parses_no_subcommand_as_none() {
        let w = parse(&[]);
        assert!(w.ev.is_none(), "absence of subcommand ≡ EvictionCmd::None");
    }

    #[test]
    fn parses_explicit_none() {
        let w = parse(&["none"]);
        assert!(matches!(w.ev, Some(EvictionCmd::None)));
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "none");
    }

    #[test]
    fn parses_sliding_with_window() {
        let w = parse(&["sliding", "--window", "2048"]);
        match w.ev {
            Some(EvictionCmd::Sliding(s)) => assert_eq!(s.window, 2048),
            _ => panic!("expected Sliding"),
        }
    }

    #[test]
    fn parses_streaming_defaults() {
        let w = parse(&["streaming"]);
        match w.ev {
            Some(EvictionCmd::Streaming(s)) => {
                assert_eq!(s.sink, 4);
                assert_eq!(s.recent_window, 0);
            }
            _ => panic!("expected Streaming"),
        }
    }

    #[test]
    fn parses_h2o_full() {
        let w = parse(&[
            "h2o",
            "--keep-ratio",
            "0.3",
            "--tracked-layers",
            "8",
            "--decay",
            "0.1",
            "--raw-scores",
        ]);
        match w.ev {
            Some(EvictionCmd::H2o(h)) => {
                assert!((h.keep_ratio - 0.3).abs() < 1e-6);
                assert_eq!(h.tracked_layers, 8);
                assert!((h.decay - 0.1).abs() < 1e-6);
                assert!(h.raw_scores);
            }
            _ => panic!("expected H2o"),
        }
    }

    #[test]
    fn parses_h2o_plus_uses_same_args() {
        let w = parse(&["h2o-plus", "--keep-ratio", "0.4"]);
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "h2o_plus");
        match w.ev {
            Some(EvictionCmd::H2oPlus(h)) => {
                assert!((h.keep_ratio - 0.4).abs() < 1e-6);
            }
            _ => panic!("expected H2oPlus"),
        }
    }

    #[test]
    fn parses_d2o_with_protected_layers() {
        let w = parse(&[
            "d2o",
            "--keep-ratio",
            "0.8",
            "--ema-beta",
            "0.6",
            "--merge-e",
            "0.15",
            "--layer-alloc",
            "--protected-layers",
            "0,1,2",
        ]);
        match w.ev {
            Some(EvictionCmd::D2o(d)) => {
                assert!((d.keep_ratio - 0.8).abs() < 1e-6);
                assert!((d.ema_beta - 0.6).abs() < 1e-6);
                assert!((d.merge_e - 0.15).abs() < 1e-6);
                assert!(d.layer_alloc);
                assert_eq!(d.protected_layers, Some(vec![0, 1, 2]));
            }
            _ => panic!("expected D2o"),
        }
    }

    #[test]
    fn rejects_unknown_eviction_arg() {
        // H2o has no --window flag (Sliding does) — clap must reject.
        let r = Wrap::try_parse_from(["test", "h2o", "--window", "256"]);
        assert!(r.is_err(), "h2o subcommand must reject Sliding's --window");
    }

    /// feature `caote` install 시 `eviction caote` 가 parse 되고 policy_name 이
    /// "caote" — session/build_bench 의 `find_stage(name)` seam 으로 흘러 플러그인을 선택한다.
    #[cfg(feature = "caote")]
    #[test]
    fn parses_caote_unit_subcommand() {
        let w = parse(&["caote"]);
        assert!(matches!(w.ev, Some(EvictionCmd::Caote)));
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "caote");
    }

    /// feature OFF(plugin 미설치)에서는 caote subcommand 가 존재하지 않아 clap 이 거부한다.
    #[cfg(not(feature = "caote"))]
    #[test]
    fn rejects_caote_when_plugin_absent() {
        let r = Wrap::try_parse_from(["test", "caote"]);
        assert!(
            r.is_err(),
            "feature OFF 시 caote subcommand 미존재 → clap reject"
        );
    }

    /// feature `rkv` install 시 `eviction rkv [--lambda L]` 가 parse 되고 policy_name="rkv"
    /// — find_stage("rkv") seam 으로 흘러 측정 stage 를 선택한다(KV roadmap 항목 0 §4.1).
    #[cfg(feature = "rkv")]
    #[test]
    fn parses_rkv_subcommand() {
        let w = parse(&["rkv", "--lambda", "0.2"]);
        match w.ev {
            Some(EvictionCmd::Rkv(ref a)) => assert!((a.lambda - 0.2).abs() < 1e-6),
            _ => panic!("expected Rkv"),
        }
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "rkv");
    }

    /// feature OFF(측정 게이트 미설치)에서는 rkv subcommand 가 존재하지 않아 clap 이 거부한다 —
    /// production 표면 불변(§6 Spec Triage, 빌드 타임 격리)의 직접 단언.
    #[cfg(not(feature = "rkv"))]
    #[test]
    fn rejects_rkv_when_feature_absent() {
        let r = Wrap::try_parse_from(["test", "rkv"]);
        assert!(
            r.is_err(),
            "feature OFF 시 rkv subcommand 미존재 → clap reject (production 표면 불변)"
        );
    }

    #[test]
    fn common_args_parse_independently() {
        // Separate parser exercises EvictionCommonArgs alone.
        #[derive(Parser, Debug)]
        struct C {
            #[clap(flatten)]
            common: EvictionCommonArgs,
        }
        let c = C::try_parse_from([
            "test",
            "--kv-budget",
            "1024",
            "--protected-prefix",
            "4",
            "--memory-threshold-mb",
            "512",
        ])
        .unwrap();
        assert_eq!(c.common.kv_budget, 1024);
        assert_eq!(c.common.protected_prefix, Some(4));
        assert_eq!(c.common.memory_threshold_mb, 512);
        // Other defaults preserved.
        assert_eq!(c.common.min_kv_cache, 256);
        assert!((c.common.eviction_target_ratio - 0.75).abs() < 1e-6);
    }

    /// `eviction plugin --name <stage>` parses and policy_name() returns the runtime name —
    /// the free-string stage selector (the stage-axis analogue of `--kv-format`). This is the
    /// seam that lets a `--load-plugin` stage be selected with no engine edit.
    #[test]
    fn parses_plugin_subcommand_by_name() {
        let w = parse(&["plugin", "--name", "my_stage"]);
        match w.ev {
            Some(EvictionCmd::Plugin(ref a)) => assert_eq!(a.name, "my_stage"),
            _ => panic!("expected Plugin"),
        }
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "my_stage");
    }

    /// `eviction plugin` without `--name` is a clap error (the name is required).
    #[test]
    fn plugin_requires_name() {
        let r = Wrap::try_parse_from(["test", "plugin"]);
        assert!(r.is_err(), "plugin subcommand requires --name");
    }
}
