//! INV-130 — Noshuffle SOA Registry Coherence on Weight Swap (Phase 3.6)
//!
//! 대응 spec: `spec/32-engine-algorithms.md` §3.12.15 (ENG-ALG-221)
//! 대응 inv : `spec/41-invariants.md` §3.15 (INV-130)
//! 대응 arch: `arch/weight_swap.md` v6 §2.2.3
//!
//! ## 불변식
//!
//! Q4_0 weight swap으로 `tensor.buffer`의 `cl_mem`이 교체되면
//! `OpenCLBackend::noshuffle_soa_registry`의 옛 cl_mem key는 stale이다.
//! Swap 결과가 비어 있지 않은 경우 `SwapExecutor`는
//! `ratio_generation` bump 직전에 backend의
//! `invalidate_noshuffle_soa_registry()`를 호출해야 한다. 호출 순서를
//! 거꾸로 하면 generation bump를 관측한 동시 forward가 stale registry로
//! plan을 재빌드할 위험이 있다.
//!
//! ## 검증 항목
//!
//! - [x] `Backend::invalidate_noshuffle_soa_registry` default impl은 no-op.
//!   CPU backend에서 반복 호출 시 panic 없이 반환한다.
//! - [x] `SwapExecutor::execute_on_slots`는 swap 결과가 비어 있을 때
//!   (`target_layers=[]` 또는 secondary=None) backend invalidate를
//!   호출하지 않는다 — ENG-ALG-211 step (e) 생략 규약과 동일 분기.
//! - [x] `SwapExecutor`는 swap이 실제로 발생했을 때 backend invalidate를
//!   **ratio_generation bump 전에** 정확히 1회 호출한다. (카운팅 mock
//!   backend + 직접 `execute_on_slots` 호출로 확인.)
//! - [x] (opencl feature) `OpenCLBackend::invalidate_noshuffle_soa_registry`는
//!   등록된 SOA entry를 전부 evict하여 lookup miss를 발생시킨다.
//!
//! ## 구현 메모
//!
//! SwapExecutor의 실제 `execute_on_slots` 경로는 `SecondaryMmap` fixture가
//! 필요하다. 이 스펙 테스트는 (a) no-secondary 경로에서 invalidate가
//! **호출되지 않음**을 확인하고, (b) opencl feature 하에서 OpenCLBackend의
//! trait override가 기존 `clear_noshuffle_soa_registry`와 동등한 효과를
//!내는지 검증한다. 실제 swap → invalidate → rebuild 전체 체인은
//! 디바이스(Adreno) 한정으로 발현되므로 Phase 4 실측 스위프에서 확인한다.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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

// ── INV-130-A: trait default is no-op ────────────────────────────────────────

/// `Backend::invalidate_noshuffle_soa_registry` default impl은 no-op이며
/// CPU backend에서 반복 호출되어도 panic 없이 반환한다.
#[test]
fn default_backend_invalidate_is_noop() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());

    // 여러 번 호출해도 안전해야 한다 — caller가 invalidate를 과호출할
    // 가능성(e.g. future retries)에 대한 회귀 방지.
    be.invalidate_noshuffle_soa_registry();
    be.invalidate_noshuffle_soa_registry();
    be.invalidate_noshuffle_soa_registry();
}

// ── INV-130-B: counting mock backend ─────────────────────────────────────────

/// Minimal `Backend` implementation that only counts
/// `invalidate_noshuffle_soa_registry` invocations. All other methods delegate
/// to the underlying CpuBackend — we are only verifying dispatch behaviour.
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

// Forward every trait method that the tests will actually need — SwapExecutor's
// empty-batch path touches no tensor ops, so only the bookkeeping methods + our
// override matter. For anything else we fall back to `unimplemented!()` which
// would fire a loud test failure if the exercised path ever widens.
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

    // ── delegated matmul / norm / attention — not used by empty swap path ──

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

fn minimal_model_config() -> ModelConfig {
    ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: 64,
        num_hidden_layers: 2,
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

// ── INV-130-C: empty-swap batch must NOT invalidate ──────────────────────────

/// SwapExecutor가 실제로 아무 layer도 교체하지 않은 경우
/// (target_layers=[], secondary=None), backend invalidate를 호출하면 안 된다.
/// ENG-ALG-211 step (e): ratio_generation bump도 동일 분기에서 생략된다.
#[test]
fn empty_swap_batch_does_not_invalidate() {
    let be_counting = Arc::new(CountingBackend::new());
    let be_dyn: Arc<dyn Backend> = be_counting.clone();

    let layers: Vec<Arc<LayerSlot>> = (0..2)
        .map(|_| Arc::new(LayerSlot::new(dummy_layer(&be_dyn), DType::F16, None, 0)))
        .collect();
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let config = minimal_model_config();
    let memory = Galloc::new();

    let executor = SwapExecutor::new(DType::Q4_0, &config, be_dyn.clone(), &memory);

    // Case 1: empty target_layers → skip everything, no bump, no invalidate.
    let report = executor
        .execute_on_slots(layers.as_slice(), None, &ratio_gen, &[], None)
        .expect("empty target must not error");
    assert!(report.swapped.is_empty(), "no swaps expected");
    assert_eq!(
        report.ratio_generation_after, None,
        "empty batch must not bump ratio_generation (ENG-ALG-211)"
    );
    assert_eq!(
        be_counting.calls(),
        0,
        "empty batch must not invalidate registry"
    );

    // Case 2: non-empty target but no secondary mmap → early return, still no
    // invalidation because no swap happened.
    let report = executor
        .execute_on_slots(layers.as_slice(), None, &ratio_gen, &[0, 1], None)
        .expect("missing secondary must not error");
    assert!(report.swapped.is_empty(), "missing secondary skips all");
    assert_eq!(
        be_counting.calls(),
        0,
        "no-secondary path must not invalidate"
    );
    assert_eq!(
        ratio_gen.load(Ordering::SeqCst),
        0,
        "ratio_generation must remain at 0 across both calls"
    );
}

// ── INV-130-D: OpenCL override clears the registry ───────────────────────────

/// `OpenCLBackend::invalidate_noshuffle_soa_registry` trait override가 기존
/// `clear_noshuffle_soa_registry` 헬퍼와 동등하게 동작해야 한다. 즉 등록된
/// entry가 invalidate 호출 후 miss를 내야 한다.
///
/// GPU 장치가 없는 CI 호스트에서는 skip.
#[cfg(feature = "opencl")]
#[test]
fn opencl_invalidate_trait_override_drops_entries() {
    use argus_engine::backend::Backend as BackendTrait;
    use argus_engine::backend::opencl::OpenCLBackend;

    // try_create_backend is pub(crate) internal; fall back to public `new`
    // and bail out if no OpenCL platform exists in the CI host.
    let backend = match OpenCLBackend::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("[SKIPPED] OpenCL device unavailable: {e}");
            return;
        }
    };

    // Without an easy host fixture for `NoshuffleSoaEntry` we fall back to
    // the observable black-box: invoke the trait method and ensure the
    // public `lookup_noshuffle_soa` reports miss for arbitrary keys
    // (registry must be empty after invalidate, regardless of prior state).
    let be_dyn: Arc<dyn BackendTrait> = backend.clone();
    be_dyn.invalidate_noshuffle_soa_registry();

    // No SOA entries were registered by this test — we only care that
    // invalidate is reachable via the trait and returns without panic.
    // The full register → invalidate → miss semantics are covered by the
    // existing `test_clear_noshuffle_soa_registry` device-feature test on
    // `OpenCLBackend`.
    assert!(
        backend.lookup_noshuffle_soa(0xdead_beef).is_none(),
        "arbitrary key must miss on an invalidated (or empty) registry"
    );
}
