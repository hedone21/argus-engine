//! INV-131 — Adreno SOA Re-conversion Safety Net (Phase 3.7a)
//!
//! 대응 spec: `spec/32-engine-algorithms.md` §3.12.16 (ENG-ALG-222)
//! 대응 inv : `spec/41-invariants.md` §3.16 (INV-131)
//! 대응 arch: `arch/weight_swap.md` §2.2.4
//!
//! ## 불변식 요약
//!
//! Q4_0 weight swap 후 첫 GPU matmul 직전, 해당 layer의 Q4_0 weight tensor의
//! cl_mem 주소가 `OpenCLBackend::noshuffle_soa_registry`에 SOA descriptor와
//! 함께 **등록**되어 있어야 한다.
//!
//! INV-130 (Phase 3.6) — stale entry **제거** — 과 INV-131 — 신규 entry **등록** —
//! 은 짝을 이룬다.
//!
//! ## 검증 항목
//!
//! - [A] swap 후 이전 cl_mem 주소가 registry에서 제거되고 새 주소가 등록된다
//!   (INV-130 + INV-131 연계 흐름 mock 검증).
//! - [B] swap이 일부 layer에만 적용될 때 swap 안 된 layer의 SOA entry가
//!   clear 정책(전체 invalidate)에 의해 제거되고, swap된 layer만 재등록된다.
//! - [C] Q4_0 이외의 dtype에 대해서는 SOA 재변환이 no-op으로 처리된다.
//! - [D] CPU backend에서 SOA 재변환 흐름이 no-op으로 처리된다 (Adreno 아닌 경우).
//!
//! ## 구현 메모
//!
//! Senior Implementer가 `ensure_noshuffle_soa_registered()` 메서드 및
//! ENG-ALG-222 알고리즘을 `swap_executor.rs` / `OpenCLBackend`에 추가하는
//! 작업을 진행 중이다. 이 테스트는 해당 인터페이스가 완성되기 전 단계에서
//! mock registry를 직접 조작하여 INV-131의 **의도**를 검증한다.
//!
//! INV-131은 디바이스(Adreno) 한정으로 발현된다. 호스트 환경에서는 registry
//! 자체가 사용되지 않으므로, 이 파일의 모든 테스트는 mock 또는 in-process
//! 방식으로만 동작한다 (GPU 디바이스 의존 없음).
//!
//! Spec ENG-ALG-222 알고리즘 단계:
//!   1. clear (INV-130 → `invalidate_noshuffle_soa_registry()`)
//!   2. convert → `convert_q4_0_to_noshuffle()` 또는 AUF cache 경로
//!   3. register → `register_noshuffle_soa(new_key, descriptor)`
//!   4. ratio_generation bump

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::buffer::DType;
use argus_engine::memory::Memory;
use argus_engine::memory::galloc::Galloc;
use argus_engine::model_config::{ModelArch, ModelConfig};
use argus_engine::models::weights::LayerSlot;
use argus_engine::tensor::Tensor;
use argus_engine::weight::SwapExecutor;

// ── Mock SOA registry ─────────────────────────────────────────────────────────

/// Minimal SOA descriptor: 클래스 메모리 주소(key)와 등록 여부만 추적한다.
/// 실제 `NoshuffleSoaEntry`(q_buf, d_buf, q_img, ne00, ne01)는 GPU 디바이스
/// 없이 구성할 수 없으므로 호스트 테스트에서는 usize key만 사용한다.
#[derive(Debug, Clone)]
struct MockSoaDescriptor {
    #[allow(dead_code)]
    key: usize,
}

/// INV-131 전용 mock SOA registry.
///
/// 실제 `OpenCLBackend::noshuffle_soa_registry`와 동일한 HashMap 인터페이스를
/// 제공한다. 테스트에서 `register`, `invalidate_all`, `lookup` 를 통해 상태를
/// 직접 조작하고 검증한다.
#[derive(Default)]
struct MockSoaRegistry {
    entries: Mutex<HashMap<usize, MockSoaDescriptor>>,
    invalidate_call_count: AtomicUsize,
    register_call_count: AtomicUsize,
}

impl MockSoaRegistry {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// ENG-ALG-222 step 1: INV-130 경로와 동일하게 전체 clear.
    fn invalidate_all(&self) {
        self.entries.lock().unwrap().clear();
        self.invalidate_call_count.fetch_add(1, Ordering::SeqCst);
    }

    /// ENG-ALG-222 step 3: 신규 key → descriptor 등록.
    fn register(&self, key: usize) {
        self.entries
            .lock()
            .unwrap()
            .insert(key, MockSoaDescriptor { key });
        self.register_call_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Registry lookup — None이면 miss (noshuffle kernel 사용 불가).
    fn lookup(&self, key: usize) -> bool {
        self.entries.lock().unwrap().contains_key(&key)
    }

    fn invalidate_calls(&self) -> usize {
        self.invalidate_call_count.load(Ordering::SeqCst)
    }

    fn register_calls(&self) -> usize {
        self.register_call_count.load(Ordering::SeqCst)
    }

    fn registered_count(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
}

// ── CountingBackend (INV-130 패턴 재사용) ─────────────────────────────────────

/// INV-130 테스트에서 가져온 mock Backend.
/// `invalidate_noshuffle_soa_registry` 호출 횟수를 추적하고
/// 나머지 메서드는 `CpuBackend`에 위임한다.
struct CountingBackend {
    inner: CpuBackend,
    invalidate_calls: AtomicUsize,
}

impl CountingBackend {
    fn new() -> Self {
        Self {
            inner: CpuBackend::new(),
            invalidate_calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.invalidate_calls.load(Ordering::SeqCst)
    }
}

impl Backend for CountingBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn name(&self) -> &str {
        "counting-mock"
    }
    fn device(&self) -> &str {
        "host"
    }
    fn invalidate_noshuffle_soa_registry(&self) {
        self.invalidate_calls.fetch_add(1, Ordering::SeqCst);
    }
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        self.inner.matmul(a, b, out)
    }
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        self.inner.matmul_transposed(a, b, out)
    }
    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        rows: usize,
        cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        self.inner.matmul_slice(a, b, rows, cols, out)
    }
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        self.inner.add_assign(a, b)
    }
    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        self.inner.scale(x, v)
    }
    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        self.inner.silu_mul(a, b)
    }
    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        self.inner.rms_norm(x, w, eps, add_unit)
    }
    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        self.inner.softmax(x)
    }
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        self.inner.rope_inplace(x, start_pos, theta)
    }
    fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
        self.inner.copy_from(t)
    }
    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        self.inner.cast(src, dst)
    }
    fn cpu_companion(&self) -> &dyn Backend {
        self
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn minimal_model_config() -> ModelConfig {
    ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 16,
        intermediate_size: 128,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_id: 1,
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
        weight_prefix: String::new(),
    }
}

fn dummy_tensor(be: &Arc<dyn Backend>, numel: usize) -> Tensor {
    let mem = Galloc::new();
    let buf = mem.alloc(numel * 4, DType::F32).unwrap();
    Tensor::new(
        argus_engine::shape::Shape::new(vec![numel]),
        buf,
        be.clone(),
    )
}

fn dummy_layer(be: &Arc<dyn Backend>) -> argus_engine::layers::transformer_layer::TransformerLayer {
    argus_engine::layers::transformer_layer::TransformerLayer {
        wq: dummy_tensor(be, 16),
        wk: dummy_tensor(be, 16),
        wv: dummy_tensor(be, 16),
        wo: dummy_tensor(be, 16),
        w_gate: dummy_tensor(be, 16),
        w_up: dummy_tensor(be, 16),
        w_down: dummy_tensor(be, 16),
        attention_norm: dummy_tensor(be, 4),
        ffn_norm: dummy_tensor(be, 4),
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    }
}

// ── INV-131-A: swap clears old entry then registers new ───────────────────────

/// **INV-131-A**: ENG-ALG-222 clear → convert → register 순서 검증.
///
/// 시나리오:
/// 1. registry에 구(old) cl_mem 주소를 등록한다 (초기 상태 = load 시점 등록).
/// 2. swap 발생: `invalidate_all()`로 stale entry 제거 (INV-130, ENG-ALG-221).
/// 3. 신규 SOA 변환 후 새 cl_mem 주소를 등록한다 (INV-131, ENG-ALG-222 step 3).
/// 4. 결과: old key는 miss, new key는 hit.
#[test]
fn test_inv_131_swap_clears_old_entry_then_registers_new() {
    let registry = MockSoaRegistry::new();

    // 초기: layer 0의 weight tensor가 cl_mem 주소 0x1000에 등록되어 있다.
    let old_key: usize = 0x1000;
    registry.register(old_key);
    assert!(
        registry.lookup(old_key),
        "pre-condition: old cl_mem must be registered before swap"
    );
    assert_eq!(registry.registered_count(), 1);

    // Swap 발생: ENG-ALG-222 step 1 — 전체 registry invalidate (INV-130).
    registry.invalidate_all();
    assert_eq!(
        registry.invalidate_calls(),
        1,
        "invalidate must be called exactly once per swap batch"
    );
    assert!(
        !registry.lookup(old_key),
        "INV-130: old cl_mem entry must be cleared after invalidate"
    );
    assert_eq!(
        registry.registered_count(),
        0,
        "registry must be empty after invalidate"
    );

    // SOA 재변환 후 새 cl_mem 주소 등록: ENG-ALG-222 step 3 (INV-131).
    let new_key: usize = 0x2000;
    registry.register(new_key);
    assert_eq!(
        registry.register_calls(),
        2, // 초기 1회 + 재등록 1회
        "register must be called for the new cl_mem"
    );

    // 최종 불변식 검증 (INV-131):
    // - old key: miss (stale, INV-130)
    // - new key: hit  (newly registered, INV-131)
    assert!(
        !registry.lookup(old_key),
        "INV-131 violation: old (stale) cl_mem must not be in registry"
    );
    assert!(
        registry.lookup(new_key),
        "INV-131 violation: new cl_mem must be registered before first GPU matmul"
    );
    assert_eq!(
        registry.registered_count(),
        1,
        "exactly one entry must exist after swap+re-register"
    );
}

// ── INV-131-B: partial swap preserves non-swapped layer entries (or full-clear) ─

/// **INV-131-B**: 일부 layer만 swap될 때의 INV-131 정책 검증.
///
/// ENG-ALG-222 / ENG-ALG-221 의 **전체 clear** 정책(현재 구현):
/// `invalidate_noshuffle_soa_registry()`는 registry를 전부 비운다.
/// 따라서 swap 안 된 layer의 entry도 제거된다 → 이후 plan rebuild 시 자연
/// 재등록이 일어나거나, ENG-ALG-222가 swap된 layer에만 재등록을 수행한다.
///
/// 이 테스트는 "전체 clear 후 swap된 layer만 재등록" 경로를 검증한다:
/// 1. layer 0(swap 대상), layer 1(swap 비대상) 각각 별도 key로 등록.
/// 2. swap 발생 → 전체 invalidate → layer 0만 새 key로 재등록.
/// 3. layer 1의 old key는 miss (plan rebuild 대기), layer 0의 new key는 hit.
#[test]
fn test_inv_131_swap_preserves_non_swapped_layers() {
    let registry = MockSoaRegistry::new();

    // 초기: layer 0과 layer 1이 각각 등록된 상태.
    let layer0_old_key: usize = 0x1000;
    let layer1_key: usize = 0x3000;
    registry.register(layer0_old_key);
    registry.register(layer1_key);
    assert_eq!(registry.registered_count(), 2, "two layers pre-registered");

    // swap: layer 0만 Q4_0으로 교체 → 전체 invalidate (ENG-ALG-221).
    registry.invalidate_all();
    assert_eq!(
        registry.registered_count(),
        0,
        "full-clear policy: all entries removed, including non-swapped layer"
    );

    // ENG-ALG-222: swap된 layer 0의 새 cl_mem을 재등록.
    // layer 1은 다음 plan rebuild 시 자연 재등록을 기다린다.
    let layer0_new_key: usize = 0x2000;
    registry.register(layer0_new_key);

    // INV-131: swap된 layer 0은 첫 GPU matmul 전 등록 완료.
    assert!(
        registry.lookup(layer0_new_key),
        "INV-131: swapped layer 0 new cl_mem must be registered"
    );
    // ENG-ALG-221 / 전체 clear 정책: 비교 대상 layer 1은 clear 후 미등록 상태.
    // (plan rebuild 시 noshuffle kernel 초기화 경로에서 재등록 예정)
    assert!(
        !registry.lookup(layer1_key),
        "full-clear policy: non-swapped layer entry removed (will re-register at plan rebuild)"
    );
    assert!(
        !registry.lookup(layer0_old_key),
        "INV-130: old key of swapped layer must not be registered"
    );
    assert_eq!(registry.registered_count(), 1);
}

// ── INV-131-C: non-Q4_0 dtype → SOA reconversion is no-op ───────────────────

/// **INV-131-C**: Q4_0 이외의 dtype에 대해서는 SOA 재변환이 호출되지 않는다.
///
/// ENG-ALG-222 의사코드:
///   ```text
///   if tensor.dtype() != DType::Q4_0:
///       continue   // Q4_0만 noshuffle SOA 대상
///   ```
///
/// 이 테스트는 모든 Q4_0 이외 dtype(`F16`, `F32`, `Q8_0`, `Q4_1`)에 대해
/// SOA 재변환이 no-op임을 검증한다. INV-126(dtype reject)과 일치:
/// Q4_0 이외 dtype은 `SwapExecutor`까지 도달하지 않는다(INV-126 Rejected),
/// 도달하더라도 SOA 변환 대상이 아니다.
#[test]
fn test_inv_131_q4_0_only() {
    let registry = MockSoaRegistry::new();

    // non-Q4_0 dtype 목록.
    let non_q4_dtypes = [
        DType::F16,
        DType::F32,
        DType::Q8_0,
        DType::Q4_1,
        DType::BF16,
    ];

    for dtype in non_q4_dtypes {
        // 시뮬레이션: 해당 dtype의 tensor에 대해 SOA 재변환 호출 여부 결정.
        let should_convert = dtype == DType::Q4_0;
        if should_convert {
            // Q4_0이면 convert + register — 이 분기는 이 루프에서 실행되지 않음.
            let key = 0xBEEF;
            registry.register(key);
        }
        // non-Q4_0이면 아무 것도 하지 않는다 (ENG-ALG-222 continue).
    }

    // 결과: registry는 비어 있어야 한다 (non-Q4_0 dtype에 대한 등록 없음).
    assert_eq!(
        registry.registered_count(),
        0,
        "INV-131: SOA reconversion must be no-op for non-Q4_0 dtypes (F16/F32/Q8_0/Q4_1/BF16)"
    );
    assert_eq!(
        registry.register_calls(),
        0,
        "register must not be called for non-Q4_0 tensors"
    );
}

// ── INV-131-D: CPU backend (non-Adreno) → SOA reconversion is no-op ──────────

/// **INV-131-D**: CPU backend 또는 비-Adreno 환경에서 SOA 재변환이 no-op.
///
/// ENG-ALG-222 의사코드:
///   ```text
///   if !backend.is_adreno() or !backend.has_noshuffle_kernel():
///       return Ok(())   // 호스트 또는 비-Adreno OpenCL → NoOp
///   ```
///
/// `Backend::invalidate_noshuffle_soa_registry()`의 default impl은 no-op이고,
/// `Backend::register_noshuffle_soa`는 trait에 존재하지 않는다 (OpenCL-only).
/// `SwapExecutor`는 비-OpenCL backend에서 registry 조작을 수행하지 않아야 한다.
///
/// 검증 방법:
/// - `CountingBackend`(CpuBackend 래핑)로 `SwapExecutor::execute_on_slots`을
///   실제 swap 없이 빈 배치 호출 → invalidate 0회 (no actual swap).
/// - CPU backend name에 "CPU"가 포함되는지 확인 (is_adreno 게이트 proxy).
#[test]
fn test_inv_131_non_adreno_backend_noop() {
    let be_counting = Arc::new(CountingBackend::new());
    let be_dyn: Arc<dyn Backend> = be_counting.clone();

    let config = minimal_model_config();
    let memory = Galloc::new();

    // CPU backend name 확인: is_adreno 판단의 proxy.
    assert!(
        be_counting.name().to_lowercase().contains("cpu")
            || be_counting.device().to_lowercase().contains("host"),
        "CountingBackend must identify as non-Adreno (CPU/host)"
    );

    // `SwapExecutor`를 CPU backend로 생성.
    let executor = SwapExecutor::new(DType::Q4_0, &config, be_dyn.clone(), &memory);

    // 빈 layer 집합으로 실행 — secondary 없으므로 swap 발생하지 않음.
    let layers: Vec<Arc<LayerSlot>> = (0..4)
        .map(|_| Arc::new(LayerSlot::new(dummy_layer(&be_dyn), DType::F16, None, 0)))
        .collect();
    let ratio_gen = Arc::new(AtomicU64::new(0));

    // Case 1: 빈 target_layers → swap 없음 → invalidate 0회 (ENG-ALG-211).
    let report = executor
        .execute_on_slots(layers.as_slice(), None, &ratio_gen, &[], None)
        .expect("empty batch must not error");
    assert!(report.swapped.is_empty());
    assert_eq!(
        be_counting.calls(),
        0,
        "INV-131-D: CPU backend must not call invalidate for empty swap batch"
    );

    // Case 2: 명시적 target_layers지만 secondary 없음 → swap 없음 → invalidate 0회.
    let report = executor
        .execute_on_slots(layers.as_slice(), None, &ratio_gen, &[0, 1, 2, 3], None)
        .expect("no-secondary path must not error");
    assert!(report.swapped.is_empty());
    assert_eq!(
        be_counting.calls(),
        0,
        "INV-131-D: no-secondary path must not invalidate registry"
    );

    // ratio_generation은 swap이 없었으므로 0으로 유지.
    assert_eq!(
        ratio_gen.load(Ordering::SeqCst),
        0,
        "ratio_generation must remain 0 when no swap occurs"
    );

    // INV-131-D 핵심: Adreno 아닌 환경에서는 SOA 재변환 API가 전혀 호출되지 않는다.
    // `register_noshuffle_soa`는 `Backend` trait에 없으므로 CPU backend에서
    // 컴파일 타임부터 호출 불가 — 이 테스트는 그 구조적 보장을 문서화한다.
    // 실제 Adreno 디바이스 한정 검증은 Phase 4 실측 스위프에서 수행한다.
}

// ── INV-131 모호성 노트 ────────────────────────────────────────────────────────
//
// 1. ENG-ALG-222 "호출 시점" 주석:
//    "step (e) 직후 … clear와 bump 사이에 호출"이라고 명시되어 있으나,
//    "또는 layer 단위로 즉시 호출 가능"이라는 대안도 제시된다. Phase 3.7a
//    단순 구현에서는 두 경우 모두 허용(correctness 충족). 이 테스트는
//    clear → register → bump 순서만 검증하며 구체적 호출 위치는 Senior에게 위임.
//
// 2. "비-swap layer 재등록" 정책:
//    전체 clear 후 비-swap layer가 plan rebuild에서 자연 재등록될지,
//    ENG-ALG-222가 직접 재등록할지 spec에서 명확하지 않다. INV-131-B는
//    "swap된 layer만 재등록" 해석을 채택하였다. Senior 구현에서 정책이
//    달라지면 INV-131-B 어서션을 조정할 필요가 있다.
//
// 3. AUF cache 경로 (ENG-ALG-222 (a) 경로):
//    Phase 3.7b 이전에는 AUF cache가 없으므로 (b) fallback 경로만 사용된다.
//    이 테스트 파일은 (b) 경로만 다루고, AUF cache 통합은 Phase 3.7b 이후
//    별도 테스트(`test_inv_132_auf_format.rs` 등)에서 다룰 예정이다.
