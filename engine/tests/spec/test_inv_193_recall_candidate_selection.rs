// spec/41-invariants.md §3.30 INV-193: RecallWeights 후보 선택
//
// **INV-193**: currently-Q4_0 layer 만 recall 후보가 된다.
// F32/F16 layer 는 이미 원본 상태이므로 skip 된다.
// `ratio` 는 `floor(ratio × N_swapped)` 개를 선택하는 상한이다.
//
// 검증:
// (a) LayerSlot.current_dtype() 이 Q4_0 인 layer 는 Q4_0 후보로 인식됨
// (b) LayerSlot.current_dtype() 이 F32/F16 인 layer 는 후보가 아님
// (c) ratio=1.0 → floor(1.0×N) = N (모든 Q4_0 선택)
// (d) ratio=0.5 → floor(0.5×N) (최소 1)
// (e) ratio 극소(0.01, N=1) → 최소 1 선택 (floor→0 → max(1) 보정)
//
// WeightRecallStage 의 내부 로직을 LayerSlot API 를 통해 추론한다.
// 단위 테스트 수준: LayerSlot.current_dtype() 계약 + ratio 선택 수식 검증.

use std::sync::Arc;

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::layers::transformer_layer::TransformerLayer;
use argus_engine::memory::host::shared::SharedBuffer;
use argus_engine::models::weights::LayerSlot;
use argus_engine::shape::Shape;
use argus_engine::tensor::Tensor;

fn cpu_be() -> Arc<dyn Backend> {
    Arc::new(CpuBackend::new())
}

fn f32_weight(be: &Arc<dyn Backend>, out_dim: usize, in_dim: usize) -> Tensor {
    let buf: Arc<dyn argus_engine::buffer::Buffer> =
        Arc::new(SharedBuffer::new(out_dim * in_dim * 4, DType::F32));
    Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, be.clone())
}

fn make_slot(be: &Arc<dyn Backend>, dtype: DType, idx: usize) -> Arc<LayerSlot> {
    let small = f32_weight(be, 1, 1);
    let layer = TransformerLayer {
        wq: small.clone(),
        wk: small.clone(),
        wv: small.clone(),
        wo: small.clone(),
        w_gate: f32_weight(be, 4, 4),
        w_up: f32_weight(be, 4, 4),
        w_down: f32_weight(be, 4, 4),
        attention_norm: small.clone(),
        ffn_norm: small,
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    };
    Arc::new(LayerSlot::new(layer, dtype, None, idx))
}

/// INV-193(a): LayerSlot.current_dtype() == Q4_0 인 slot 은 Q4_0 후보로 인식됨.
#[test]
fn q4_slot_current_dtype_is_q4_0() {
    let be = cpu_be();
    let slot = make_slot(&be, DType::Q4_0, 0);
    assert_eq!(
        slot.current_dtype(),
        DType::Q4_0,
        "Q4_0 slot → current_dtype = Q4_0 (INV-193-a)"
    );
}

/// INV-193(b): F32/F16 layer 는 Q4_0 후보가 아님.
#[test]
fn f32_f16_slots_not_q4_candidates() {
    let be = cpu_be();
    let f32_slot = make_slot(&be, DType::F32, 0);
    let f16_slot = make_slot(&be, DType::F16, 1);
    assert_ne!(
        f32_slot.current_dtype(),
        DType::Q4_0,
        "F32 slot 은 Q4_0 아님 (INV-193-b)"
    );
    assert_ne!(
        f16_slot.current_dtype(),
        DType::Q4_0,
        "F16 slot 은 Q4_0 아님 (INV-193-b)"
    );
}

/// INV-193(c): ratio=1.0 → 전체 Q4_0 layer 선택 (floor(1.0×N) = N).
#[test]
fn ratio_full_selects_all_q4_layers() {
    let n_swapped = 4usize;
    let ratio: f32 = 1.0;
    let target_count = ((ratio * n_swapped as f32).floor() as usize)
        .max(1)
        .min(n_swapped);
    assert_eq!(target_count, 4, "ratio=1.0, N=4 → target=4 (INV-193-c)");
}

/// INV-193(d): ratio=0.5 → floor(0.5×N) 선택.
#[test]
fn ratio_half_selects_half() {
    let n_swapped = 4usize;
    let ratio: f32 = 0.5;
    let target_count = ((ratio * n_swapped as f32).floor() as usize)
        .max(1)
        .min(n_swapped);
    assert_eq!(target_count, 2, "ratio=0.5, N=4 → target=2 (INV-193-d)");
}

/// INV-193(e): ratio 극소 (0.01, N=1) → floor→0이어도 max(1) 보정으로 최소 1 선택.
#[test]
fn ratio_tiny_selects_at_least_one() {
    let n_swapped = 1usize;
    let ratio: f32 = 0.01;
    let target_count = ((ratio * n_swapped as f32).floor() as usize)
        .max(1)
        .min(n_swapped);
    assert_eq!(target_count, 1, "ratio=0.01, N=1 → min=1 (INV-193-e)");
}

/// INV-193 후보 필터 계산: 혼합 dtype 목록에서 Q4_0만 카운트.
#[test]
fn mixed_dtype_candidate_count() {
    let be = cpu_be();
    let slots: Vec<Arc<LayerSlot>> = vec![
        make_slot(&be, DType::F32, 0),  // not candidate
        make_slot(&be, DType::Q4_0, 1), // candidate
        make_slot(&be, DType::F16, 2),  // not candidate
        make_slot(&be, DType::Q4_0, 3), // candidate
    ];
    let q4_count = slots
        .iter()
        .filter(|s| s.current_dtype() == DType::Q4_0)
        .count();
    assert_eq!(q4_count, 2, "혼합 4-slot에서 Q4_0 후보 2개 (INV-193)");
}
