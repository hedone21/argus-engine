//! ENG-ALG-030 ~ ENG-ALG-048: SWIFT SkipConfig + QCF (Quality Cost Function)
//!
//! SkipConfig uniform_init, validate, layer_importance QCF,
//! eviction_qcf proxy, quant_qcf NMSE/OPR, skip_qcf acceptance tracker.

use argus_engine::inference::skip_config::SkipConfig;
use argus_engine::qcf::layer_importance::{ImportanceEntry, SubLayer};
use argus_engine::qcf::quant_qcf::compute_nmse_block;
use argus_engine::qcf::{ImportanceTable, SkipQcfTracker};

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

// ══════════════════════════════════════════════════════════════
// ENG-ALG-032: ImportanceTable QCF — 레이어 중요도 기반 skip cost
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_032_importance_table_full_skip() {
    let entries = vec![
        ImportanceEntry {
            layer_id: 0,
            sublayer: SubLayer::Full,
            importance: 0.5,
            opr: 0.1,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 1,
            sublayer: SubLayer::Full,
            importance: 0.3,
            opr: 0.2,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 2,
            sublayer: SubLayer::Full,
            importance: 0.2,
            opr: 0.3,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
    ];
    let table = ImportanceTable::from_entries(entries);

    // 모든 레이어 skip → QCF = 1.0
    let skip_set = vec![
        (0, SubLayer::Full),
        (1, SubLayer::Full),
        (2, SubLayer::Full),
    ];
    let qcf = table.compute_qcf_weight(&skip_set);
    assert!((qcf - 1.0).abs() < 1e-6);
}

#[test]
fn test_eng_alg_032_importance_table_partial_skip() {
    let entries = vec![
        ImportanceEntry {
            layer_id: 0,
            sublayer: SubLayer::Full,
            importance: 0.5,
            opr: 0.1,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 1,
            sublayer: SubLayer::Full,
            importance: 0.3,
            opr: 0.2,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 2,
            sublayer: SubLayer::Full,
            importance: 0.2,
            opr: 0.3,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
    ];
    let table = ImportanceTable::from_entries(entries);

    // layer 2만 skip → QCF = 0.2 / 1.0 = 0.2
    let skip_set = vec![(2, SubLayer::Full)];
    let qcf = table.compute_qcf_weight(&skip_set);
    assert!((qcf - 0.2).abs() < 1e-6);
}

#[test]
fn test_eng_alg_032_importance_table_empty_skip() {
    let entries = vec![ImportanceEntry {
        layer_id: 0,
        sublayer: SubLayer::Full,
        importance: 0.5,
        opr: 0.1,
        importance_mean_pool: None,
        importance_shortgpt_bi: None,
    }];
    let table = ImportanceTable::from_entries(entries);
    let qcf = table.compute_qcf_weight(&[]);
    assert_eq!(qcf, 0.0);
}

#[test]
fn test_eng_alg_032_estimate_qcf_protects_first_last() {
    let entries = vec![
        ImportanceEntry {
            layer_id: 0,
            sublayer: SubLayer::Full,
            importance: 0.1,
            opr: 0.05,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 1,
            sublayer: SubLayer::Full,
            importance: 0.3,
            opr: 0.1,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 2,
            sublayer: SubLayer::Full,
            importance: 0.5,
            opr: 0.2,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 3,
            sublayer: SubLayer::Full,
            importance: 0.1,
            opr: 0.05,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
    ];
    let table = ImportanceTable::from_entries(entries);

    // skip_count=2, num_layers=4: layer 0과 3은 보호
    let (qcf, skip_set) = table.estimate_qcf_for_count(2, 4);
    assert!(qcf > 0.0);

    // 보호된 레이어가 skip_set에 포함되지 않아야 함
    for &(layer_id, _) in &skip_set {
        assert_ne!(layer_id, 0, "첫 번째 레이어는 보호되어야 함");
        assert_ne!(layer_id, 3, "마지막 레이어는 보호되어야 함");
    }
}

#[test]
fn test_eng_alg_032_estimate_qcf_selects_least_important() {
    let entries = vec![
        ImportanceEntry {
            layer_id: 0,
            sublayer: SubLayer::Full,
            importance: 0.9,
            opr: 0.5,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 1,
            sublayer: SubLayer::Full,
            importance: 0.1,
            opr: 0.02,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 2,
            sublayer: SubLayer::Full,
            importance: 0.8,
            opr: 0.4,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 3,
            sublayer: SubLayer::Full,
            importance: 0.2,
            opr: 0.05,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
        ImportanceEntry {
            layer_id: 4,
            sublayer: SubLayer::Full,
            importance: 0.7,
            opr: 0.3,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        },
    ];
    let table = ImportanceTable::from_entries(entries);

    // skip_count=2, num_layers=5: 가장 덜 중요한 layer 1(0.1), 3(0.2) 선택
    let (_qcf, skip_set) = table.estimate_qcf_for_count(2, 5);
    let layer_ids: Vec<usize> = skip_set.iter().map(|&(id, _)| id).collect();

    // layer 1과 3이 가장 덜 중요 (0, 4는 보호)
    assert!(layer_ids.contains(&1));
    assert!(layer_ids.contains(&3));
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-045/046: quant_qcf — NMSE block 검증
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_045_nmse_block_q8() {
    use argus_engine::quant::QKKV;

    let mut original = [0.0f32; QKKV];
    for (i, slot) in original.iter_mut().enumerate().take(QKKV) {
        *slot = (i as f32) / QKKV as f32;
    }

    let nmse = compute_nmse_block(&original, 8, 1e-8);
    // Q8은 정밀도가 높으므로 NMSE가 매우 작아야 함
    assert!(nmse < 0.1, "Q8 NMSE should be small, got {nmse}");
}

#[test]
fn test_eng_alg_045_nmse_block_q2() {
    use argus_engine::quant::QKKV;

    let mut original = [0.0f32; QKKV];
    for (i, slot) in original.iter_mut().enumerate().take(QKKV) {
        *slot = (i as f32) / QKKV as f32;
    }

    let nmse_q2 = compute_nmse_block(&original, 2, 1e-8);
    let nmse_q8 = compute_nmse_block(&original, 8, 1e-8);
    // Q2는 Q8보다 NMSE가 높아야 함
    assert!(
        nmse_q2 > nmse_q8,
        "Q2 NMSE ({nmse_q2}) should be higher than Q8 ({nmse_q8})"
    );
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-046/047: flush_qcf / flush_opr (QuantFlushParams 기반)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_046_flush_proxy_basic() {
    use argus_engine::qcf::QcfConfig;
    use argus_engine::qcf::quant_qcf::compute_flush_nmse;
    use argus_engine::qcf_types::QuantFlushParams;
    use argus_engine::quant::QKKV;

    let kv_heads = 1;
    let head_dim = QKKV; // = 32
    let flush_tokens = QKKV; // = 32 (1 group)
    let res_cap = flush_tokens;

    // 선형 증가 데이터
    let n = kv_heads * res_cap * head_dim;
    let res_k: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let res_v: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    let params = QuantFlushParams {
        res_k: &res_k,
        res_v: &res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits: 8,
    };
    let config = QcfConfig::default();
    let metric = compute_flush_nmse(&params, &config);

    assert_eq!(metric.action, "kv.quant_flush_nmse");
    assert!(metric.raw_value >= 0.0);
    assert_eq!(metric.tokens_affected, flush_tokens);
}

#[test]
fn test_eng_alg_046_flush_proxy_q2_higher_than_q8() {
    use argus_engine::qcf::QcfConfig;
    use argus_engine::qcf::quant_qcf::compute_flush_nmse;
    use argus_engine::qcf_types::QuantFlushParams;
    use argus_engine::quant::QKKV;

    let kv_heads = 1;
    let head_dim = QKKV;
    let flush_tokens = QKKV;
    let res_cap = flush_tokens;

    let n = kv_heads * res_cap * head_dim;
    let res_k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let res_v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();

    let config = QcfConfig::default();

    let params_q8 = QuantFlushParams {
        res_k: &res_k,
        res_v: &res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits: 8,
    };
    let metric_q8 = compute_flush_nmse(&params_q8, &config);

    let params_q2 = QuantFlushParams {
        res_k: &res_k,
        res_v: &res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits: 2,
    };
    let metric_q2 = compute_flush_nmse(&params_q2, &config);

    assert!(
        metric_q2.raw_value >= metric_q8.raw_value,
        "Q2 proxy ({}) should >= Q8 proxy ({})",
        metric_q2.raw_value,
        metric_q8.raw_value
    );
}

#[test]
fn test_eng_alg_047_flush_opr_basic() {
    use argus_engine::qcf::QcfConfig;
    use argus_engine::qcf::quant_qcf::compute_flush_opr;
    use argus_engine::qcf_types::QuantFlushParams;
    use argus_engine::quant::QKKV;

    let kv_heads = 1;
    let head_dim = QKKV;
    let flush_tokens = QKKV;
    let res_cap = flush_tokens;

    let n = kv_heads * res_cap * head_dim;
    let res_k: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let res_v: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    let params = QuantFlushParams {
        res_k: &res_k,
        res_v: &res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits: 8,
    };
    let config = QcfConfig::default();
    let metric = compute_flush_opr(&params, &config);
    assert_eq!(metric.action, "kv.quant_flush_opr");
    assert!(metric.raw_value >= 0.0);
}

#[test]
fn test_eng_alg_047_flush_opr_q2_higher() {
    use argus_engine::qcf::QcfConfig;
    use argus_engine::qcf::quant_qcf::compute_flush_opr;
    use argus_engine::qcf_types::QuantFlushParams;
    use argus_engine::quant::QKKV;

    let kv_heads = 1;
    let head_dim = QKKV;
    let flush_tokens = QKKV;
    let res_cap = flush_tokens;

    let n = kv_heads * res_cap * head_dim;
    let res_k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let res_v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();

    let config = QcfConfig::default();

    let params_q8 = QuantFlushParams {
        res_k: &res_k,
        res_v: &res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits: 8,
    };
    let opr_q8 = compute_flush_opr(&params_q8, &config);

    let params_q2 = QuantFlushParams {
        res_k: &res_k,
        res_v: &res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits: 2,
    };
    let opr_q2 = compute_flush_opr(&params_q2, &config);

    assert!(
        opr_q2.raw_value >= opr_q8.raw_value,
        "Q2 OPR ({}) should >= Q8 OPR ({})",
        opr_q2.raw_value,
        opr_q8.raw_value
    );
}

// ══════════════════════════════════════════════════════════════
// ENG-ALG-048: SkipQcfTracker — acceptance rate 추적
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_alg_048_perfect_acceptance() {
    let mut tracker = SkipQcfTracker::new(10);
    tracker.record(5, 5);
    tracker.record(3, 3);

    let metric = tracker.current_proxy();
    assert!(
        (metric.raw_value - 0.0).abs() < 1e-6,
        "100% acceptance → 0% rejection"
    );
    assert!((tracker.lifetime_acceptance_rate() - 1.0).abs() < 1e-6);
}

#[test]
fn test_eng_alg_048_total_rejection() {
    let mut tracker = SkipQcfTracker::new(10);
    tracker.record(0, 5);

    let metric = tracker.current_proxy();
    assert!(
        (metric.raw_value - 1.0).abs() < 1e-6,
        "0% acceptance → 100% rejection"
    );
}

#[test]
fn test_eng_alg_048_partial_acceptance() {
    let mut tracker = SkipQcfTracker::new(10);
    tracker.record(3, 5); // 60% acceptance → 40% rejection

    let metric = tracker.current_proxy();
    assert!((metric.raw_value - 0.4).abs() < 1e-6);
}

#[test]
fn test_eng_alg_048_window_rolling() {
    let mut tracker = SkipQcfTracker::new(3);
    tracker.record(5, 5); // 0.0
    tracker.record(0, 5); // 1.0
    tracker.record(5, 5); // 0.0
    // Window: [0.0, 1.0, 0.0] → avg = 0.333

    let metric = tracker.current_proxy();
    assert!(
        (metric.raw_value - 1.0 / 3.0).abs() < 0.01,
        "expected ~0.333, got {}",
        metric.raw_value
    );

    // 하나 더 추가 → oldest (0.0) 탈락
    tracker.record(0, 5); // 1.0
    // Window: [1.0, 0.0, 1.0] → avg = 0.666
    let metric = tracker.current_proxy();
    assert!(
        (metric.raw_value - 2.0 / 3.0).abs() < 0.01,
        "expected ~0.666, got {}",
        metric.raw_value
    );
    assert_eq!(tracker.window_len(), 3);
}
