//! Layer-swap QCF dump tests (Phase 1 in zazzy-herding-bonbon plan).
//!
//! Validates `dump_qcf_swap_json` + `QcfSwapDumpContext` serialization behaviour.
//! Covers schema round-trip, ratio boundary conditions, noise NaN exclusion,
//! and null serialization for generation-mode (ppl/avg_nll absent).
//!
//! Decider correctness (importance × ε bottom-k sort, protected layers) is
//! already covered by unit tests in decider.rs; here we focus on the dump
//! mapping and JSON schema contract consumed by the external harness.
//!
//! Spec: Phase 1 (zazzy-herding-bonbon), ENG-ALG-215, ENG-ALG-217.

use argus_engine::observability::eval::qcf_helpers::{QcfSwapDumpContext, dump_qcf_swap_json};
use argus_engine::qcf::layer_importance::{ImportanceEntry, ImportanceTable, SubLayer};
use argus_engine::weight::QuantNoiseTable;

// ── Fixture helpers ───────────────────────────────────────────────────────────

fn make_importance(entries: Vec<(usize, f32, f32)>) -> ImportanceTable {
    let entries = entries
        .into_iter()
        .map(|(id, imp, opr)| ImportanceEntry {
            layer_id: id,
            sublayer: SubLayer::Full,
            importance: imp,
            opr,
            importance_mean_pool: None,
            importance_shortgpt_bi: None,
        })
        .collect();
    ImportanceTable::from_entries(entries)
}

fn make_noise(vals: Vec<f32>) -> QuantNoiseTable {
    QuantNoiseTable::from_values(vals)
}

/// Serialize ctx to a temp file and parse back as serde_json::Value.
fn dump_and_parse(ctx: &QcfSwapDumpContext<'_>) -> serde_json::Value {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("qcf.json");
    dump_qcf_swap_json(&path, ctx).expect("dump_qcf_swap_json should succeed");
    let content = std::fs::read_to_string(&path).expect("read JSON");
    serde_json::from_str(&content).expect("valid JSON")
}

// ── Test 1: schema round-trip ─────────────────────────────────────────────────

/// All top-level keys from schema_version 1 must be present, with correct types.
#[test]
fn test_dump_schema_round_trip() {
    let imp = make_importance(vec![(0, 0.42, 0.31), (1, 0.28, 0.19)]);
    let noise = make_noise(vec![0.018, 0.022]);
    let swap_set = vec![1usize];

    let ctx = QcfSwapDumpContext {
        model_arch: "llama",
        model_path: "models/llama-3.2-1b-f16.gguf",
        secondary_path: Some("models/llama-3.2-1b-mixed.auf"),
        primary_dtype: "F16",
        secondary_dtype: "Q4_0",
        num_layers: 2,
        force_swap_ratio: Some(0.33),
        swap_set: &swap_set,
        qcf_swap_predicted: 0.214,
        fallback_used: false,
        importance_table: Some(&imp),
        noise_table: Some(&noise),
        ppl: Some(12.34),
        avg_nll: Some(2.51),
        n_eval_tokens: 4096,
        wall_time_s: 18.7,
        warmup_tokens: 256,
        backend: "cpu",
        kv_type: "f16",
        ppl_corpus: Some("experiments/prompts/prefill_4096.txt"),
        eval_ll_output: None,
        swap_algorithm: None,
        trajectory: None,
        dpllm_epsilon: None,
        dpllm_epsilon_multi: None,
        dpllm_epsilon_abs: None,
        dpllm_epsilon_qcf: None,
        direct_attn_f4: None,
        direct_attn_f5: None,
        direct_attn_f5_decode_only: None,
        direct_attn_f5_prefill_decode: None,
    };

    let v = dump_and_parse(&ctx);

    // schema_version
    assert_eq!(v["schema_version"].as_u64().unwrap(), 1);
    // string fields
    assert_eq!(v["model_arch"].as_str().unwrap(), "llama");
    assert_eq!(
        v["model_path"].as_str().unwrap(),
        "models/llama-3.2-1b-f16.gguf"
    );
    assert_eq!(
        v["secondary_path"].as_str().unwrap(),
        "models/llama-3.2-1b-mixed.auf"
    );
    assert_eq!(v["primary_dtype"].as_str().unwrap(), "F16");
    assert_eq!(v["secondary_dtype"].as_str().unwrap(), "Q4_0");
    // numeric fields
    assert_eq!(v["num_layers"].as_u64().unwrap(), 2);
    assert!((v["force_swap_ratio"].as_f64().unwrap() - 0.33).abs() < 1e-5);
    assert_eq!(v["swap_count"].as_u64().unwrap(), 1);
    assert!((v["qcf_swap_predicted"].as_f64().unwrap() - 0.214).abs() < 1e-4);
    assert!(!v["fallback_used"].as_bool().unwrap());
    // ppl / avg_nll
    assert!((v["ppl"].as_f64().unwrap() - 12.34).abs() < 1e-5);
    assert!((v["avg_nll"].as_f64().unwrap() - 2.51).abs() < 1e-5);
    assert_eq!(v["n_eval_tokens"].as_u64().unwrap(), 4096);
    assert!((v["wall_time_s"].as_f64().unwrap() - 18.7).abs() < 0.1);
    assert_eq!(v["warmup_tokens"].as_u64().unwrap(), 256);
    assert_eq!(v["backend"].as_str().unwrap(), "cpu");
    assert_eq!(v["kv_type"].as_str().unwrap(), "f16");
    assert_eq!(
        v["ppl_corpus"].as_str().unwrap(),
        "experiments/prompts/prefill_4096.txt"
    );

    // arrays
    assert!(v["importance_table"].is_array());
    assert_eq!(v["importance_table"].as_array().unwrap().len(), 2);
    assert!(v["noise_table"].is_array());
    assert_eq!(v["noise_table"].as_array().unwrap().len(), 2);

    // swap_set array
    let ss = v["swap_set"].as_array().unwrap();
    assert_eq!(ss.len(), 1);
    assert_eq!(ss[0].as_u64().unwrap(), 1);
}

// ── Test 2: ratio=0.0 → swap_count=0, qcf_swap_predicted=0 ──────────────────

/// When force_swap_ratio=0 (baseline), swap_set is empty and qcf is 0.
#[test]
fn test_ratio_zero_swap_count_zero() {
    let swap_set: Vec<usize> = vec![];

    let ctx = QcfSwapDumpContext {
        model_arch: "llama",
        model_path: "models/llama.gguf",
        secondary_path: None,
        primary_dtype: "F16",
        secondary_dtype: "Q4_0",
        num_layers: 16,
        force_swap_ratio: Some(0.0),
        swap_set: &swap_set,
        qcf_swap_predicted: 0.0,
        fallback_used: false,
        importance_table: None,
        noise_table: None,
        ppl: Some(10.0),
        avg_nll: Some(2.30),
        n_eval_tokens: 512,
        wall_time_s: 5.0,
        warmup_tokens: 256,
        backend: "cpu",
        kv_type: "f32",
        ppl_corpus: None,
        eval_ll_output: None,
        swap_algorithm: None,
        trajectory: None,
        dpllm_epsilon: None,
        dpllm_epsilon_multi: None,
        dpllm_epsilon_abs: None,
        dpllm_epsilon_qcf: None,
        direct_attn_f4: None,
        direct_attn_f5: None,
        direct_attn_f5_decode_only: None,
        direct_attn_f5_prefill_decode: None,
    };

    let v = dump_and_parse(&ctx);

    assert_eq!(v["swap_count"].as_u64().unwrap(), 0);
    assert_eq!(v["swap_set"].as_array().unwrap().len(), 0);
    assert!((v["qcf_swap_predicted"].as_f64().unwrap()).abs() < 1e-8);
    // secondary_path and ppl_corpus null
    assert!(v["secondary_path"].is_null());
    assert!(v["ppl_corpus"].is_null());
}

// ── Test 3: ratio=0.33 on 16L → swap_count=5, layer 0/last excluded ──────────

/// With 16 layers and ratio=0.33, floor(0.33*16)=5 layers.
/// WeightSwapDecider protects layer 0 and 15; selected must not contain them.
#[test]
fn test_ratio_033_skip_count_5_for_16l() {
    use argus_engine::weight::decider::{SwapAlgorithm, WeightSwapDecider};

    // MW-C: decider 는 flat per-layer importance/noise 슬라이스를 받는다
    // (index = layer_id). 16 레이어 uniform.
    let imp_flat: Vec<f32> = vec![0.1f32; 16];
    let noise_flat: Vec<f32> = vec![0.05f32; 16];

    let decider = WeightSwapDecider {
        importance: Some(&imp_flat),
        noise: Some(&noise_flat),
        n_decoder_layers: 16,
        currently_swapped: &[],
        allow_boundary_layers: false,
        algorithm: SwapAlgorithm::ImportanceAware,
    };
    // budget = floor(0.33 * 16) - 0 = 5
    let decision = decider.decide(5);

    // swap_count = floor(0.33 * 16) = 5
    assert_eq!(
        decision.selected_layers.len(),
        5,
        "expected 5 layers for ratio=0.33 on 16L"
    );
    // layer 0 and 15 must not be in the set
    assert!(
        !decision.selected_layers.contains(&0),
        "layer 0 is protected"
    );
    assert!(
        !decision.selected_layers.contains(&15),
        "layer 15 (last) is protected"
    );

    // Now verify the dump reflects this correctly.
    let ctx = QcfSwapDumpContext {
        model_arch: "llama",
        model_path: "models/llama.gguf",
        secondary_path: None,
        primary_dtype: "F16",
        secondary_dtype: "Q4_0",
        num_layers: 16,
        force_swap_ratio: Some(0.33),
        swap_set: &decision.selected_layers,
        qcf_swap_predicted: decision.qcf_swap_estimate,
        fallback_used: decision.fallback_used,
        importance_table: None,
        noise_table: None,
        ppl: Some(12.0),
        avg_nll: Some(2.48),
        n_eval_tokens: 1024,
        wall_time_s: 10.0,
        warmup_tokens: 256,
        backend: "cpu",
        kv_type: "f16",
        ppl_corpus: None,
        eval_ll_output: None,
        swap_algorithm: None,
        trajectory: None,
        dpllm_epsilon: None,
        dpllm_epsilon_multi: None,
        dpllm_epsilon_abs: None,
        dpllm_epsilon_qcf: None,
        direct_attn_f4: None,
        direct_attn_f5: None,
        direct_attn_f5_decode_only: None,
        direct_attn_f5_prefill_decode: None,
    };

    let v = dump_and_parse(&ctx);
    assert_eq!(v["swap_count"].as_u64().unwrap(), 5);

    let ss = v["swap_set"].as_array().unwrap();
    let ss_vals: Vec<u64> = ss.iter().map(|x| x.as_u64().unwrap()).collect();
    assert!(
        !ss_vals.contains(&0),
        "layer 0 must not be in dump swap_set"
    );
    assert!(
        !ss_vals.contains(&15),
        "layer 15 must not be in dump swap_set"
    );
}

// ── Test 4: ratio=1.0 → capped at n_layers-2 (layer 0/last protected) ────────

/// WeightSwapDecider with ratio=1.0 on a 16-layer model selects at most 14 layers
/// (layer 0 and 15 are always protected), and fallback_used=false when tables are valid.
#[test]
fn test_ratio_one_caps_at_n_minus_2() {
    use argus_engine::weight::decider::{SwapAlgorithm, WeightSwapDecider};

    // MW-C: decider 는 flat per-layer importance/noise 슬라이스를 받는다.
    let noise_flat: Vec<f32> = (0..16).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let imp_flat: Vec<f32> = (0..16).map(|i| 0.05 * (i as f32 + 1.0)).collect();

    let decider = WeightSwapDecider {
        importance: Some(&imp_flat),
        noise: Some(&noise_flat),
        n_decoder_layers: 16,
        currently_swapped: &[],
        allow_boundary_layers: false,
        algorithm: SwapAlgorithm::ImportanceAware,
    };
    // budget = floor(1.0 * 16) - 0 = 16
    let decision = decider.decide(16);

    // floor(1.0 * 16) = 16 target, but 14 candidates (0 and 15 excluded)
    assert_eq!(
        decision.selected_layers.len(),
        14,
        "ratio=1.0 on 16L should yield 14 (0 and last protected)"
    );
    assert!(
        !decision.fallback_used,
        "should not use fallback with valid tables"
    );
    assert!(!decision.selected_layers.contains(&0));
    assert!(!decision.selected_layers.contains(&15));

    let ctx = QcfSwapDumpContext {
        model_arch: "llama",
        model_path: "models/llama.gguf",
        secondary_path: None,
        primary_dtype: "F16",
        secondary_dtype: "Q4_0",
        num_layers: 16,
        force_swap_ratio: Some(1.0),
        swap_set: &decision.selected_layers,
        qcf_swap_predicted: decision.qcf_swap_estimate,
        fallback_used: decision.fallback_used,
        importance_table: None,
        noise_table: None,
        ppl: Some(15.0),
        avg_nll: Some(2.71),
        n_eval_tokens: 512,
        wall_time_s: 8.0,
        warmup_tokens: 256,
        backend: "cpu",
        kv_type: "f16",
        ppl_corpus: None,
        eval_ll_output: None,
        swap_algorithm: None,
        trajectory: None,
        dpllm_epsilon: None,
        dpllm_epsilon_multi: None,
        dpllm_epsilon_abs: None,
        dpllm_epsilon_qcf: None,
        direct_attn_f4: None,
        direct_attn_f5: None,
        direct_attn_f5_decode_only: None,
        direct_attn_f5_prefill_decode: None,
    };

    let v = dump_and_parse(&ctx);
    assert_eq!(v["swap_count"].as_u64().unwrap(), 14);
    assert!(!v["fallback_used"].as_bool().unwrap());
}

// ── Test 5: NaN ε → excluded from noise_table dump ───────────────────────────

/// Layers with NaN epsilon must NOT appear in the noise_table array in the dump.
#[test]
fn test_nan_epsilon_excluded() {
    // Layer 1 has NaN ε (INV-127)
    let noise = make_noise(vec![0.02, f32::NAN, 0.04, 0.03]);
    let swap_set: Vec<usize> = vec![];

    let ctx = QcfSwapDumpContext {
        model_arch: "llama",
        model_path: "models/llama.gguf",
        secondary_path: None,
        primary_dtype: "F16",
        secondary_dtype: "Q4_0",
        num_layers: 4,
        force_swap_ratio: None,
        swap_set: &swap_set,
        qcf_swap_predicted: 0.0,
        fallback_used: false,
        importance_table: None,
        noise_table: Some(&noise),
        ppl: None,
        avg_nll: None,
        n_eval_tokens: 0,
        wall_time_s: 0.0,
        warmup_tokens: 256,
        backend: "cpu",
        kv_type: "f16",
        ppl_corpus: None,
        eval_ll_output: None,
        swap_algorithm: None,
        trajectory: None,
        dpllm_epsilon: None,
        dpllm_epsilon_multi: None,
        dpllm_epsilon_abs: None,
        dpllm_epsilon_qcf: None,
        direct_attn_f4: None,
        direct_attn_f5: None,
        direct_attn_f5_decode_only: None,
        direct_attn_f5_prefill_decode: None,
    };

    let v = dump_and_parse(&ctx);
    let nt = v["noise_table"].as_array().unwrap();

    // Must have 3 entries (layer 1 excluded)
    assert_eq!(
        nt.len(),
        3,
        "NaN epsilon layer must be excluded from noise_table"
    );

    // Verify no entry has layer=1
    let layer_ids: Vec<u64> = nt.iter().map(|e| e["layer"].as_u64().unwrap()).collect();
    assert!(
        !layer_ids.contains(&1),
        "layer 1 (NaN ε) must not appear in noise_table"
    );
    assert!(layer_ids.contains(&0));
    assert!(layer_ids.contains(&2));
    assert!(layer_ids.contains(&3));
}

// ── Test 6: importance_table and noise_table fields include all valid layers ──

/// When both tables are provided, importance_table and noise_table contain
/// all valid entries with correct field names (layer, sublayer, importance, opr / epsilon).
#[test]
fn test_importance_noise_in_dump() {
    let imp = make_importance(vec![(0, 0.42, 0.31), (1, 0.28, 0.19), (2, 0.15, 0.10)]);
    let noise = make_noise(vec![0.018, 0.022, 0.031]);
    let swap_set = vec![2usize];

    let ctx = QcfSwapDumpContext {
        model_arch: "llama",
        model_path: "models/llama.gguf",
        secondary_path: None,
        primary_dtype: "F16",
        secondary_dtype: "Q4_0",
        num_layers: 3,
        force_swap_ratio: Some(0.5),
        swap_set: &swap_set,
        qcf_swap_predicted: 0.3,
        fallback_used: false,
        importance_table: Some(&imp),
        noise_table: Some(&noise),
        ppl: Some(11.0),
        avg_nll: Some(2.40),
        n_eval_tokens: 512,
        wall_time_s: 4.0,
        warmup_tokens: 128,
        backend: "cpu",
        kv_type: "f16",
        ppl_corpus: None,
        eval_ll_output: None,
        swap_algorithm: None,
        trajectory: None,
        dpllm_epsilon: None,
        dpllm_epsilon_multi: None,
        dpllm_epsilon_abs: None,
        dpllm_epsilon_qcf: None,
        direct_attn_f4: None,
        direct_attn_f5: None,
        direct_attn_f5_decode_only: None,
        direct_attn_f5_prefill_decode: None,
    };

    let v = dump_and_parse(&ctx);

    // importance_table: 3 entries with required fields
    let it = v["importance_table"].as_array().unwrap();
    assert_eq!(
        it.len(),
        3,
        "importance_table must have 3 entries for 3-layer model"
    );
    for entry in it {
        assert!(
            entry["layer"].is_number(),
            "importance entry must have layer field"
        );
        assert!(
            entry["sublayer"].is_string(),
            "importance entry must have sublayer field"
        );
        assert!(
            entry["importance"].is_number(),
            "importance entry must have importance field"
        );
        assert!(
            entry["opr"].is_number(),
            "importance entry must have opr field"
        );
    }

    // noise_table: 3 entries (all finite)
    let nt = v["noise_table"].as_array().unwrap();
    assert_eq!(
        nt.len(),
        3,
        "noise_table must have 3 entries when all ε are finite"
    );
    for entry in nt {
        assert!(
            entry["layer"].is_number(),
            "noise entry must have layer field"
        );
        assert!(
            entry["epsilon"].is_number(),
            "noise entry must have epsilon field"
        );
        let eps = entry["epsilon"].as_f64().unwrap();
        assert!(
            eps.is_finite(),
            "epsilon must be finite in dump (NaN filtered)"
        );
    }

    // Verify first entry values
    let it0 = &it[0];
    assert_eq!(it0["layer"].as_u64().unwrap(), 0);
    assert!((it0["importance"].as_f64().unwrap() - 0.42).abs() < 1e-4);
    assert!((it0["opr"].as_f64().unwrap() - 0.31).abs() < 1e-4);

    let nt0 = &nt[0];
    assert_eq!(nt0["layer"].as_u64().unwrap(), 0);
    assert!((nt0["epsilon"].as_f64().unwrap() - 0.018).abs() < 1e-5);
}

// ── Test 7: ppl=None serializes as null (generation mode) ────────────────────

/// In generation mode, ppl and avg_nll must serialize as JSON null, not be omitted.
#[test]
fn test_ppl_none_serializes_as_null() {
    let swap_set: Vec<usize> = vec![3usize, 5usize];

    let ctx = QcfSwapDumpContext {
        model_arch: "llama",
        model_path: "models/llama.gguf",
        secondary_path: None,
        primary_dtype: "F16",
        secondary_dtype: "Q4_0",
        num_layers: 16,
        force_swap_ratio: Some(0.25),
        swap_set: &swap_set,
        qcf_swap_predicted: 0.18,
        fallback_used: false,
        importance_table: None,
        noise_table: None,
        ppl: None,     // generation mode: no PPL
        avg_nll: None, // generation mode: no NLL
        n_eval_tokens: 0,
        wall_time_s: 3.0,
        warmup_tokens: 256,
        backend: "cuda",
        kv_type: "f16",
        ppl_corpus: None,
        eval_ll_output: None,
        swap_algorithm: None,
        trajectory: None,
        dpllm_epsilon: None,
        dpllm_epsilon_multi: None,
        dpllm_epsilon_abs: None,
        dpllm_epsilon_qcf: None,
        direct_attn_f4: None,
        direct_attn_f5: None,
        direct_attn_f5_decode_only: None,
        direct_attn_f5_prefill_decode: None,
    };

    let v = dump_and_parse(&ctx);

    // ppl and avg_nll must be present as null (not missing)
    assert!(
        v.get("ppl").is_some(),
        "ppl key must exist even when None (null, not omitted)"
    );
    assert!(
        v["ppl"].is_null(),
        "ppl must be JSON null in generation mode"
    );
    assert!(
        v.get("avg_nll").is_some(),
        "avg_nll key must exist even when None"
    );
    assert!(
        v["avg_nll"].is_null(),
        "avg_nll must be JSON null in generation mode"
    );

    // n_eval_tokens still serialized as 0
    assert_eq!(v["n_eval_tokens"].as_u64().unwrap(), 0);

    // importance_table and noise_table null when absent
    assert!(v["importance_table"].is_null());
    assert!(v["noise_table"].is_null());

    // ppl_corpus null
    assert!(v["ppl_corpus"].is_null());

    // backend and kv_type
    assert_eq!(v["backend"].as_str().unwrap(), "cuda");
    assert_eq!(v["kv_type"].as_str().unwrap(), "f16");
}

// ── Test 8: eval_ll_output Some → included in JSON ───────────────────────────

/// When eval_ll_output is Some, the JSON must contain an `eval_ll_output`
/// field with at least the `results` array from the EvalOutput.
#[test]
fn test_eval_ll_output_some_included_in_json() {
    use argus_engine::observability::eval::output::EvalOutput;

    let eval_output = EvalOutput {
        results: vec![
            serde_json::json!({
                "id": "race-h-001",
                "choice_nlls": [12.34_f32, 11.89_f32, 13.10_f32, 14.55_f32],
                "predicted_norm": 1_i64,
                "predicted_raw": 1_i64,
                "qcf_layer_skip": serde_json::Value::Null,
            }),
            serde_json::json!({
                "id": "race-h-002",
                "choice_nlls": [9.1_f32, 10.2_f32, 8.7_f32, 11.3_f32],
                "predicted_norm": 2_i64,
                "predicted_raw": 2_i64,
                "qcf_layer_skip": serde_json::Value::Null,
            }),
        ],
        config: serde_json::json!({"model": "test", "eviction_policy": "none"}),
        wall_time_s: 18.7,
        layer_importance: None,
        layer_skip_qcf: None,
        layer_skip_qcf_normalized: None,
        qcf_layer_skip: None,
        qcf_layer_skip_layers: None,
    };

    let swap_set = vec![1usize, 3usize, 5usize];

    let ctx = QcfSwapDumpContext {
        model_arch: "qwen2",
        model_path: "models/qwen2.5-1.5b-f16.gguf",
        secondary_path: Some("models/qwen2.5-1.5b-mixed.auf"),
        primary_dtype: "F16",
        secondary_dtype: "Q4_0",
        num_layers: 28,
        force_swap_ratio: Some(0.33),
        swap_set: &swap_set,
        qcf_swap_predicted: 0.214,
        fallback_used: false,
        importance_table: None,
        noise_table: None,
        ppl: None,
        avg_nll: None,
        n_eval_tokens: 0,
        wall_time_s: 18.7,
        warmup_tokens: 256,
        backend: "cuda",
        kv_type: "f16",
        ppl_corpus: None,
        eval_ll_output: Some(&eval_output),
        swap_algorithm: None,
        trajectory: None,
        dpllm_epsilon: None,
        dpllm_epsilon_multi: None,
        dpllm_epsilon_abs: None,
        dpllm_epsilon_qcf: None,
        direct_attn_f4: None,
        direct_attn_f5: None,
        direct_attn_f5_decode_only: None,
        direct_attn_f5_prefill_decode: None,
    };

    let v = dump_and_parse(&ctx);

    // eval_ll_output must be present and be an object
    let elo = &v["eval_ll_output"];
    assert!(
        elo.is_object(),
        "eval_ll_output must be a JSON object when Some, got: {:?}",
        elo
    );

    // results array must contain 2 items
    let results = &elo["results"];
    assert!(
        results.is_array(),
        "eval_ll_output.results must be an array"
    );
    assert_eq!(
        results.as_array().unwrap().len(),
        2,
        "eval_ll_output.results must have 2 items"
    );

    // First result id matches
    assert_eq!(
        results[0]["id"].as_str().unwrap(),
        "race-h-001",
        "first result id must match"
    );

    // wall_time_s present
    assert!(
        elo["wall_time_s"].is_number(),
        "eval_ll_output.wall_time_s must be a number"
    );
    assert!(
        (elo["wall_time_s"].as_f64().unwrap() - 18.7).abs() < 0.1,
        "wall_time_s value mismatch"
    );

    // ppl and avg_nll still null at the top level
    assert!(
        v["ppl"].is_null(),
        "top-level ppl must be null in eval-ll mode"
    );
    assert!(
        v["avg_nll"].is_null(),
        "top-level avg_nll must be null in eval-ll mode"
    );
}

// ── Test 9: eval_ll_output None → JSON field is null (not absent) ────────────

/// When eval_ll_output is None (PPL / generation mode), `eval_ll_output` in the
/// JSON must be present and serialize as JSON null (consistent with the policy
/// that all top-level keys are always present in schema_version 1).
#[test]
fn test_eval_ll_output_none_serializes_as_null() {
    let swap_set: Vec<usize> = vec![];

    let ctx = QcfSwapDumpContext {
        model_arch: "llama",
        model_path: "models/llama.gguf",
        secondary_path: None,
        primary_dtype: "F16",
        secondary_dtype: "Q4_0",
        num_layers: 16,
        force_swap_ratio: None,
        swap_set: &swap_set,
        qcf_swap_predicted: 0.0,
        fallback_used: false,
        importance_table: None,
        noise_table: None,
        ppl: Some(10.5),
        avg_nll: Some(2.35),
        n_eval_tokens: 2048,
        wall_time_s: 6.0,
        warmup_tokens: 256,
        backend: "cpu",
        kv_type: "f32",
        ppl_corpus: Some("experiments/prompts/prefill_4096.txt"),
        eval_ll_output: None,
        swap_algorithm: None,
        trajectory: None,
        dpllm_epsilon: None,
        dpllm_epsilon_multi: None,
        dpllm_epsilon_abs: None,
        dpllm_epsilon_qcf: None,
        direct_attn_f4: None,
        direct_attn_f5: None,
        direct_attn_f5_decode_only: None,
        direct_attn_f5_prefill_decode: None,
    };

    let v = dump_and_parse(&ctx);

    // eval_ll_output key must exist and be null
    assert!(
        v.get("eval_ll_output").is_some(),
        "eval_ll_output key must be present even when None"
    );
    assert!(
        v["eval_ll_output"].is_null(),
        "eval_ll_output must be JSON null in PPL/generation mode"
    );

    // Confirm schema_version and other fields unaffected
    assert_eq!(v["schema_version"].as_u64().unwrap(), 1);
    assert!((v["ppl"].as_f64().unwrap() - 10.5).abs() < 1e-4);
}
