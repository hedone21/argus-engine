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
/// `generate ... plugin --name h2o`. The wrapper produces the
/// `generate ... eviction plugin --name h2o --set keep_ratio=0.5` form
/// documented in `docs/USAGE.md` and `docs/35_experiment_runner_guide.md`.
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
/// CLI usage — every stage (built-in or plugin) is selected generically by registry name + `--set`:
/// ```text
/// generate -m model.gguf eviction plugin --name sliding --set window=1024
/// generate -m model.gguf eviction plugin --name h2o --set keep_ratio=0.5 --set tracked_layers=0
/// generate -m model.gguf eviction plugin --name d2o --set keep_ratio=0.75 --set ema_beta=0.7
/// ```
///
/// Omitting the subcommand is equivalent to [`EvictionCmd::None`] —
/// no eviction; the KV cache grows up to `--max-seq-len`.
#[derive(Subcommand, Debug, Clone)]
pub enum EvictionCmd {
    /// No eviction (default). KV cache grows up to --max-seq-len.
    None,

    /// Plugin-supplied eviction stage, selected by registry name (the only stage selector — the
    /// stage-axis analogue of `--kv-format <name>`). Any technique crate registered statically
    /// (linkme `KV_CACHE_STAGES`) or dynamically (`--load-plugin`) is selectable with no engine
    /// edit. CLI form: `eviction plugin --name <stage> [--set k=v]...`. Built-ins (sliding/
    /// streaming/h2o/h2o_plus/d2o) and feature-gated stages (caote/rkv) are selected the same way;
    /// a name the registry doesn't know (e.g. a feature-disabled stage) fails at construction.
    Plugin(PluginArgs),
}

impl EvictionCmd {
    /// Canonical policy name — the stage registry key (a built-in like "h2o", a feature-gated name
    /// like "caote"/"rkv", or any `eviction plugin --name <name>`). Also used by manager IPC, the
    /// lua policy DSL, and JSON dumps, so downstream code can keep matching on the policy name.
    pub fn policy_name(&self) -> &str {
        match self {
            EvictionCmd::None => "none",
            // The runtime stage name, borrowed for &self's lifetime — this is why policy_name /
            // eviction_policy return &str rather than &'static str.
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
    fn parses_builtin_via_plugin_set_blob() {
        // Built-ins are now selected generically: `plugin --name h2o --set k=v`. The blob carries
        // every former typed knob; policy_name() returns the runtime stage name.
        let w = parse(&[
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
        ]);
        match w.ev {
            Some(EvictionCmd::Plugin(ref a)) => {
                assert_eq!(a.name, "h2o");
                assert!(a.sets.contains(&("keep_ratio".into(), "0.3".into())));
                assert!(a.sets.contains(&("tracked_layers".into(), "8".into())));
                assert!(a.sets.contains(&("decay".into(), "0.1".into())));
                assert!(a.sets.contains(&("raw_scores".into(), "true".into())));
            }
            _ => panic!("expected Plugin"),
        }
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "h2o");
    }

    #[test]
    fn parses_d2o_via_plugin_set_blob() {
        // d2o's technique-private knobs ride the same `--set` blob (no typed mirror in the engine).
        let w = parse(&[
            "plugin",
            "--name",
            "d2o",
            "--set",
            "keep_ratio=0.8",
            "--set",
            "merge_axis=value_only",
            "--set",
            "protected_layers=0,1,2",
        ]);
        match w.ev {
            Some(EvictionCmd::Plugin(ref a)) => {
                assert_eq!(a.name, "d2o");
                assert!(a.sets.contains(&("merge_axis".into(), "value_only".into())));
                assert!(
                    a.sets
                        .contains(&("protected_layers".into(), "0,1,2".into()))
                );
            }
            _ => panic!("expected Plugin"),
        }
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "d2o");
    }

    #[test]
    fn unknown_set_key_is_accepted_not_rejected() {
        // UX-loss note (B1-3): the generic `--set` blob has no schema — a typo'd key is carried
        // opaquely (the plugin ignores it), not rejected at parse time (the former typed clap
        // rejection of unknown flags is gone).
        let w = parse(&["plugin", "--name", "h2o", "--set", "windwo=256"]);
        match w.ev {
            Some(EvictionCmd::Plugin(ref a)) => {
                assert!(a.sets.contains(&("windwo".into(), "256".into())));
            }
            _ => panic!("expected Plugin"),
        }
    }

    // caote/rkv are now selected like any other stage: `eviction plugin --name caote|rkv`. There is
    // no typed subcommand, so the build-time clap-reject isolation moved to a RUNTIME registry miss
    // (B1-3): `plugin --name <name>` always parses; feature-OFF means the stage crate isn't linked,
    // so `find_stage(name)` resolves to None and construction fails.

    /// feature ON: caote is registered, so `plugin --name caote` resolves.
    #[cfg(feature = "caote")]
    #[test]
    fn caote_registered_when_feature_present() {
        let w = parse(&["plugin", "--name", "caote"]);
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "caote");
        assert!(argus_extension_api::find_stage("caote").is_some());
    }

    /// feature OFF: `plugin --name caote` parses, but the registry resolves to None (runtime miss
    /// replaces the former build-time clap reject).
    #[cfg(not(feature = "caote"))]
    #[test]
    fn caote_unregistered_when_feature_absent() {
        let w = parse(&["plugin", "--name", "caote"]);
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "caote");
        assert!(argus_extension_api::find_stage("caote").is_none());
    }

    /// feature ON: rkv measurement stage is registered.
    #[cfg(feature = "rkv")]
    #[test]
    fn rkv_registered_when_feature_present() {
        let w = parse(&["plugin", "--name", "rkv"]);
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "rkv");
        assert!(argus_extension_api::find_stage("rkv").is_some());
    }

    /// feature OFF: rkv unregistered → runtime registry miss (was a build-time clap reject — the
    /// production-surface-invariance guarantee now holds at construction, not parse).
    #[cfg(not(feature = "rkv"))]
    #[test]
    fn rkv_unregistered_when_feature_absent() {
        let w = parse(&["plugin", "--name", "rkv"]);
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "rkv");
        assert!(argus_extension_api::find_stage("rkv").is_none());
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
