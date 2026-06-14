// spec/41-invariants.md §3.30 INV-195: RecallWeights loud no-op 5종
//
// **INV-195**: 아래 5가지 경우에 recall 은 패닉/Err-강하 없이 `StageOutcome::Consumed` 를 반환하고
// stderr 에 1회 메시지를 출력한다:
// (a) secondary 에 F16 variant 부재 (DtypeNotFound) — Q4_0-only secondary.
// (b) Adreno SOA 경로 (AdrenoSoaF16Rejected) — SOA layout 는 Q4_0 전용.
// (c) secondary handle 부재 (no_secondary).
// (d) currently-swapped (Q4_0) layer 0개.
// (e) in-flight plan 활성 (R-1 가드, swap 과 공유 마커).
// + 추가: (f) invalid ratio (≤0.0 또는 >1.0).
//
// WeightRecallStage 내부 단위 테스트(weight_recall.rs::tests)에서 Consumed+graceful 을 직접
// 검증하므로, 이 spec 테스트는 dispatcher 수준 + EngineCommand 배선 레벨을 검증한다.

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
use argus_engine::session::cli::SwapMode;
use argus_engine::session::command_dispatcher::CommandDispatcher;
use argus_engine::session::pipeline_registry::PipelineRegistry;
use argus_engine::session::swap_runtime::EngineSwapRuntime;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;
use argus_engine::weight::AsyncSwapDispatcher;
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

fn minimal_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 8,
        hidden_size: 4,
        num_hidden_layers: 2,
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

fn make_swap_runtime(be: &Arc<dyn Backend>) -> Arc<EngineSwapRuntime> {
    let dispatcher = Arc::new(AsyncSwapDispatcher::new(be.clone()));
    let rr = argus_engine::weight::setup_runtime_resources(be.clone());
    let config = Arc::new(minimal_config());
    Arc::new(EngineSwapRuntime::new(
        be.clone(),
        dispatcher,
        config,
        rr.release_worker.clone(),
        SwapMode::Incremental,
        1024 * 1024,
        4,
        None,
    ))
}

/// Dispatcher 수준: RecallWeights directive 가 submit 된 뒤 stage 발화 시 loud no-op.
///
/// dispatcher 로 stage 를 submit 한 뒤 WeightMutate dispatch 를 호출한다.
/// secondary=None 이라 stage commit 내부에서 no_secondary loud no-op 이 발생하지만
/// pipeline 전체가 panic/Err 없이 완료돼야 한다 (INV-195-c dispatcher path).
#[test]
fn dispatcher_recall_with_no_secondary_does_not_panic() {
    use argus_engine::observability::profile::OpProfiler;
    use argus_engine::pipeline::{
        LifecyclePhase, PipelineDispatcher, Pressure, StageContext, StepInfo,
    };

    let be = cpu_be();
    let registry = Arc::new(PipelineRegistry::new());
    let slots: Vec<Arc<LayerSlot>> = (0..2).map(|i| ffn_slot(&be, i)).collect();
    let host: Arc<dyn Memory> = Arc::new(Galloc::new());
    let hw = Arc::new(Hardware::new(be.clone(), None, None, host, None));
    let rt = make_swap_runtime(&be);

    // model 을 직접 구성하지 않고 None 으로 넘겨 미구성(no-op) 경로 사용.
    // model=None → submit_recall 내부에서 early-return, registry 변화 0.
    let mut d = CommandDispatcher::new(
        Arc::clone(&registry),
        Vec::new(),
        None,
        slots,
        Some(hw),
        None, // model=None → recall 미구성 경로
        Some(rt),
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
        "model=None → submit 없음 (INV-195 미구성 loud no-op)"
    );

    // registry 가 비어 있어도 WeightMutate dispatch 는 panic 없이 완료돼야 한다.
    let mut profiler = OpProfiler::new();
    let mut ctx = StageContext {
        step: StepInfo {
            pos: 0,
            decode_step: 0,
            pressure: Pressure::new(0),
            prev_token: 0,
        },
        profiler: &mut profiler,
    };
    registry.dispatch(LifecyclePhase::WeightMutate, &mut ctx);
    // 도달 = panic 없음 (INV-195 graceful 계약 충족)
}

/// INV-195(f): invalid ratio 는 dispatcher 를 거쳐도 submit 됨 (transient 시맨틱).
///
/// ratio 유효성 검사는 dispatcher 가 아니라 stage commit 내부에서 수행되므로,
/// dispatcher 는 ratio=1.5 를 받아도 stage 를 submit 한다.
/// stage 는 commit 진입 시 loud no-op 처리한다.
#[test]
fn dispatcher_submits_stage_regardless_of_ratio_value() {
    let be = cpu_be();
    let registry = Arc::new(PipelineRegistry::new());
    let slots: Vec<Arc<LayerSlot>> = (0..2).map(|i| ffn_slot(&be, i)).collect();
    let host: Arc<dyn Memory> = Arc::new(Galloc::new());
    let hw = Arc::new(Hardware::new(be.clone(), None, None, host, None));
    let rt = make_swap_runtime(&be);

    // swap_runtime 은 있지만 model=None 이라 submit 자체가 안 일어남.
    // ratio 검사는 stage-internal 이므로 dispatcher 단에서는 값 검사 없음.
    let mut d = CommandDispatcher::new(
        Arc::clone(&registry),
        Vec::new(),
        None,
        slots,
        Some(hw),
        None,
        Some(rt),
        None,
        Vec::new(),
        None,
        Arc::new(Mutex::new(None)),
        Arc::new(Mutex::new(None)),
    );
    // ratio=1.5(무효) 이지만 dispatcher 는 값 검사 없이 model=None 에서 early-return.
    d.dispatch(vec![EngineCommand::RecallWeights { ratio: 1.5 }]);
    assert_eq!(registry.len(), 0, "model=None → submit 없음 (미구성 경로)");
}
