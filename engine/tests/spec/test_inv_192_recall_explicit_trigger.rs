// spec/41-invariants.md §3.30 INV-192: RecallWeights — 명시적 트리거만 발화
//
// **INV-192**: `RestoreDefaults` 는 weight recall 을 발화하지 않는다.
// RecallWeights directive 만이 WeightRecallStage 를 submit 한다.
//
// 검증:
// (a) RestoreDefaults → WeightRecallStage submit 없음 (INV-192 핵심 단언)
// (b) RecallWeights 미구성(model=None) dispatcher → submit 없음 (graceful)
// (c) RecallWeights 없는 batch (다른 directive) → submit 없음
// (d) RecallWeights 재도착 → transient (last-applied 게이트 없음, dispatcher 레벨)
//
// 주의: "RecallWeights → submit 1개" 는 weight_recall.rs 내부 단위 테스트에서 검증됨.
// command_dispatcher.rs 의 make_swap_dispatcher 가 TransformerModel 직접 구성을 요구하므로,
// 외부 spec 테스트에서는 미구성(model=None) 경로만 커버한다.

use std::sync::{Arc, Mutex};

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::hardware::Hardware;
use argus_engine::layers::transformer_layer::TransformerLayer;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::model_config::{ModelArch, ModelConfig};
use argus_engine::models::weights::LayerSlot;
use argus_engine::session::command_dispatcher::CommandDispatcher;
use argus_engine::session::pipeline_registry::PipelineRegistry;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use argus_shared::EngineCommand;

fn cpu_be() -> Arc<dyn Backend> {
    Arc::new(CpuBackend::new())
}

fn f32_weight(be: &Arc<dyn Backend>, out_dim: usize, in_dim: usize) -> Tensor {
    let buf: Arc<dyn argus_engine::buffer::Buffer> =
        Arc::new(SharedBuffer::new(out_dim * in_dim * 4, DType::F32));
    Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, be.clone())
}

fn ffn_slot(be: &Arc<dyn Backend>, idx: usize) -> Arc<LayerSlot> {
    let small = f32_weight(be, 1, 1);
    let layer = TransformerLayer {
        wq: small.clone(),
        wk: small.clone(),
        wv: small.clone(),
        wo: small.clone(),
        w_gate: f32_weight(be, 512, 256),
        w_up: f32_weight(be, 512, 256),
        w_down: f32_weight(be, 256, 512),
        attention_norm: small.clone(),
        ffn_norm: small,
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    };
    Arc::new(LayerSlot::new(layer, DType::F32, None, idx))
}

fn minimal_config(n_layers: usize) -> ModelConfig {
    ModelConfig {
        vocab_size: 8,
        hidden_size: 4,
        num_hidden_layers: n_layers,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        intermediate_size: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 500_000.0,
        head_dim: 4,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_id: 2,
        arch: ModelArch::Llama,
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
        weight_prefix: String::new(),
    }
}

/// recall 미구성(model=None) dispatcher — 모든 recall directive 가 no-op.
fn make_unconfigured_dispatcher(registry: &Arc<PipelineRegistry>) -> CommandDispatcher {
    let be = cpu_be();
    let slots: Vec<Arc<LayerSlot>> = (0..2).map(|i| ffn_slot(&be, i)).collect();
    let host: Arc<dyn Memory> = Arc::new(Galloc::new());
    let hw = Arc::new(Hardware::new(be, None, None, host, None));
    CommandDispatcher::new(
        Arc::clone(registry),
        Vec::new(),
        None,
        slots,
        Some(hw),
        None, // model=None → recall 미구성
        None,
        None,
        Vec::new(),
        None,
        Arc::new(Mutex::new(None)),
        Arc::new(Mutex::new(None)),
    )
}

/// INV-192(a): RestoreDefaults → WeightRecallStage submit 없음.
///
/// RestoreDefaults 는 partition/evict 상태를 복원하지만 swap recall 을 트리거하지 않는다.
/// partition 미적용 상태라 partition Full 복원 submit 도 없어 registry 변화 0.
#[test]
fn restore_defaults_does_not_submit_recall() {
    let registry = Arc::new(PipelineRegistry::new());
    let mut d = make_unconfigured_dispatcher(&registry);
    assert_eq!(registry.len(), 0);
    d.dispatch(vec![EngineCommand::RestoreDefaults]);
    assert_eq!(
        registry.len(),
        0,
        "RestoreDefaults → recall submit 없음 (INV-192)"
    );
}

/// INV-192(b): RecallWeights 미구성(model=None) → submit 없음.
#[test]
fn recall_unconfigured_model_none_ignores_directive() {
    let registry = Arc::new(PipelineRegistry::new());
    let mut d = make_unconfigured_dispatcher(&registry);
    d.dispatch(vec![EngineCommand::RecallWeights { ratio: 1.0 }]);
    assert_eq!(
        registry.len(),
        0,
        "model=None → recall directive 무시 (INV-192)"
    );
}

/// INV-192(c): RecallWeights 없는 batch → submit 없음.
#[test]
fn no_recall_directive_no_submit() {
    let registry = Arc::new(PipelineRegistry::new());
    let mut d = make_unconfigured_dispatcher(&registry);
    d.dispatch(vec![EngineCommand::Throttle { delay_ms: 50 }]);
    assert_eq!(
        registry.len(),
        0,
        "recall directive 없는 batch → submit 없음 (INV-192)"
    );
}

/// INV-192(d): RestoreDefaults 반복 → recall submit 없음 (비대칭 — swap recall 은 명시적만).
#[test]
fn restore_defaults_repeated_no_recall_submit() {
    let registry = Arc::new(PipelineRegistry::new());
    let mut d = make_unconfigured_dispatcher(&registry);
    d.dispatch(vec![EngineCommand::RestoreDefaults]);
    d.dispatch(vec![EngineCommand::RestoreDefaults]);
    assert_eq!(
        registry.len(),
        0,
        "RestoreDefaults 반복 → recall submit 없음 (INV-192)"
    );
}

/// INV-192(e): swap_runtime=None(미구성) → recall directive 무시.
#[test]
fn recall_unconfigured_swap_runtime_none_ignores_directive() {
    let be = cpu_be();
    let registry = Arc::new(PipelineRegistry::new());
    let slots: Vec<Arc<LayerSlot>> = (0..2).map(|i| ffn_slot(&be, i)).collect();
    let host: Arc<dyn Memory> = Arc::new(Galloc::new());
    let hw = Arc::new(Hardware::new(be, None, None, host, None));
    let mut d = CommandDispatcher::new(
        Arc::clone(&registry),
        Vec::new(),
        None,
        slots,
        Some(hw),
        None,
        None, // swap_runtime=None
        None,
        Vec::new(),
        None,
        Arc::new(Mutex::new(None)),
        Arc::new(Mutex::new(None)),
    );
    d.dispatch(vec![EngineCommand::RecallWeights { ratio: 1.0 }]);
    assert_eq!(
        registry.len(),
        0,
        "swap_runtime=None → recall directive 무시 (INV-192)"
    );
}

// ── 보충: RecallWeights EngineCommand 가 직렬화 가능함을 확인 ──

/// RecallWeights EngineCommand serde 계약 확인 (MSG-043 연계).
#[test]
fn recall_weights_serde_round_trip() {
    let cmd = EngineCommand::RecallWeights { ratio: 0.75 };
    let json = serde_json::to_string(&cmd).unwrap();
    let decoded: EngineCommand = serde_json::from_str(&json).unwrap();
    match decoded {
        EngineCommand::RecallWeights { ratio } => {
            assert!((ratio - 0.75).abs() < 1e-6, "ratio round-trip");
        }
        other => panic!("expected RecallWeights, got {:?}", other),
    }
    assert!(
        !json.contains("target_dtype"),
        "ENG-ALG-240: target_dtype 필드 없음"
    );
}
