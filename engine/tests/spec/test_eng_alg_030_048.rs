//! ENG-ALG-030: SWIFT SkipConfig — uniform_init and validate.
//!
//! The ENG-ALG-032/045/046/047/048 cases here measured the QCF metric family
//! (layer importance, quant NMSE/OPR flush proxies, the skip acceptance tracker)
//! and went with it.

use argus_engine::inference::skip_config::SkipConfig;

// ══════════════════════════════════════════════════════════════
// ENG-ALG-030: SkipConfig uniform_init
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_030_uniform_init_16_layers() {
    let config = SkipConfig::uniform_init(16, 0.5);
    // (16-2)*2 = 28 candidates, 50% = 14 skips
    assert_eq!(config.total_skips(), 14);
    // Layer 0 and 15 never skipped
    assert!(!config.skip_attn(0));
    assert!(!config.skip_mlp(0));
    assert!(!config.skip_attn(15));
    assert!(!config.skip_mlp(15));
    assert!(config.validate(16));
}

#[test]
fn test_eng_alg_030_uniform_init_small() {
    let config = SkipConfig::uniform_init(2, 0.5);
    assert_eq!(config.total_skips(), 0);
}

#[test]
fn test_eng_alg_030_uniform_init_zero_ratio() {
    let config = SkipConfig::uniform_init(16, 0.0);
    assert_eq!(config.total_skips(), 0);
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-030/C03: SkipConfig validate — first/last layer 보호
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_030_c03_validate_first_layer_skip() {
    let mut config = SkipConfig::new();
    config.attn_skip.insert(0);
    assert!(!config.validate(16));
}

#[test]
fn test_eng_alg_030_c03_validate_last_layer_skip() {
    let mut config = SkipConfig::new();
    config.mlp_skip.insert(15);
    assert!(!config.validate(16));
}
