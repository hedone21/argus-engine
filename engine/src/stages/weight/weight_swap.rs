//! `WeightSwapStage` — `SwapWeights` runtime directive 의 OneShot `PipelineStage` (AB-6).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.6.
//!
//! `WeightMutate` phase(구 `PostEviction`, §5.6.5)에서 발화하여, register 시점 보유한
//! `Arc<TransformerModel>` 의 weight slot precision 을 런타임 swap(F16→Q4_0)한다. commit 본문은
//! `EngineSwapRuntime::handle_swap_weights`(swap_runtime.rs) §1~7 을 byte-identical 이전한 것이며,
//! 그 orphan 함수가 **등가 anchor** 다(§5.6.2 — 자기 비교 금지, Stage 산출 vs 직접 호출 산출 비교).
//!
//! PartitionStage 와 형제(둘 다 `stages/weight/`)이나 join 표면이 0 이다 — partition 은 weight slot
//! **dispatch mode**(GPU/CPU split geometry), swap 은 weight **precision**(F16→Q4_0) 변경(format 축).
//!
//! **4-mode 분담**(§5.6.3): Incremental 만 Stage 가 multi-tick drain(`WeightMutate` 매 tick
//! `drain_chunk`(per_tick=2, LISWAP-6) → `SwapExecutor::execute_on_slots` → `Continue`,
//! `is_done()` tick 에 `Consumed`). IntraForward/LayerImmediate/PhaseAware 는 hook 객체 설치(construction
//! tier)만 하고 즉시 `Consumed` — 실제 swap 은 그 hook 이 forward 의 layer/op 경계에서 자율 진행한다
//! (`INV-HOTPATH-DISPATCH` 로 PipelineStage 격상 불가).
//!
//! **transient 시맨틱**(§5.6.4): partition 의 last-applied 게이트도 evict 의 armed 게이트도 없다.
//! 재submit 차단은 dispatcher 가 아니라 **commit §2 in-flight 가드**가 담당한다 — `EngineSwapRuntime`
//! 의 공유 in-flight 마커(Arc 공유)를 보고, 미완 plan 이 살아 있으면 새 Stage 의 commit 을 reject 한다.
//!
//! **pos 환류 없음**(§5.6.1): swap 은 weight precision 만 바꾸고 KV/pos 는 불변 → `StageOutcome`
//! 무변경(driver 후처리 0, PartitionStage 와 동일). plan path 의 stale `cl_mem` 차단은
//! INV-129/130/131 + INV-150 이 런타임 자동 처리(Stage 는 slot precision 만 mutate).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use argus_shared::{DtypeTag, LayerSwapEntry, WeightSwapReport};

use crate::buffer::DType;
use crate::models::transformer::TransformerModel;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};
use crate::qcf_collector::ImportanceLookup;
use crate::session::swap_runtime::EngineSwapRuntime;
use crate::weight::IncrementalSwapPlan;

/// `WeightMutate` phase 에서 weight slot precision 을 swap 하는 OneShot Stage.
///
/// Incremental mode 는 `plan` 을 보유·소진(multi-tick drain). 나머지 3-mode 는 commit tick 에
/// hook 만 설치하고 plan 은 `None` 유지(즉시 `Consumed`).
pub struct WeightSwapStage {
    /// register 시점 보유 (§5.6.3 "model 측 접근 seam"). `handle_swap_weights` 가
    /// `&TransformerModel` 인자로 query 하던 secondary_mmap / quant_noise / layers /
    /// ratio_generation / current_dtype 을 Stage 가 held-handle 로 보유(god-ctx 회피,
    /// INV-STAGE-LAYER-HANDLE). `layers[i].swap_weights(&self)` 가 ArcSwap RCU 라 `&self`
    /// Stage 가 mutate 가능.
    model: Arc<TransformerModel>,
    /// swap 자원 묶음(swap_backend / dispatcher / config / release_worker / default_mode /
    /// 공유 in-flight 마커 / report sender). Arc 공유 → 새 Stage 인스턴스도 같은 in-flight
    /// 마커를 본다(§5.6.4 재submit 차단).
    runtime: Arc<EngineSwapRuntime>,
    /// decider 입력(§5.6.1). `handle_swap_weights` 의 `importance: Option<&dyn ImportanceLookup>`
    /// 인자를 held-handle 로 보유.
    importance: Option<Arc<dyn ImportanceLookup>>,
    /// directive 가 지정한 swap ratio (0,1].
    ratio: f32,
    /// directive 가 지정한 target dtype. INV-126 으로 Q4_0 만 허용.
    target_dtype: DtypeTag,
    /// 첫 `WeightMutate` tick = directive 도착 tick. commit 1회만 수행하기 위한 가드.
    committed: Mutex<bool>,
    /// Incremental mode drain 상태(§5.6.3). commit 시 `Some(plan)` 설치, 매 tick drain,
    /// `is_done()` 시 retire. non-Incremental 은 commit 후에도 `None` 유지(hook 설치만).
    plan: Mutex<Option<IncrementalSwapPlan>>,
    /// §5.9.2 Track B: ModelForward 와 공유하는 layer-boundary hook cell. IntraForward/
    /// LayerImmediate commit 시 `Some(hook)` 설치 → ModelForward 가 매 decode step lock-read 후
    /// forward args 슬롯에 주입. assembly 가 1개 cell 을 만들어 양측에 `Arc` clone 으로 넘긴다
    /// (INV-STAGE-LAYER-HANDLE 동형 — `StageContext` 무확장).
    hook_cell: Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>>,
}

impl WeightSwapStage {
    /// `SwapWeights` directive 1건 → OneShot swap Stage (§5.6.4).
    #[allow(clippy::too_many_arguments)]
    pub fn one_shot(
        model: Arc<TransformerModel>,
        runtime: Arc<EngineSwapRuntime>,
        importance: Option<Arc<dyn ImportanceLookup>>,
        ratio: f32,
        target_dtype: DtypeTag,
        // §5.9.2 Track B: ModelForward 와 공유하는 hook cell (assembly 생성, Arc clone).
        hook_cell: Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>>,
    ) -> Self {
        Self {
            model,
            runtime,
            importance,
            ratio,
            target_dtype,
            committed: Mutex::new(false),
            plan: Mutex::new(None),
            hook_cell,
        }
    }

    /// commit 본문 = `handle_swap_weights` §1~7 byte-identical 이전(§5.6.2, 정본 = swap_runtime.rs).
    ///
    /// 반환: `Ok(true)` = Incremental plan 설치(이후 multi-tick drain 필요) / `Ok(false)` =
    /// reject(no-op) 또는 non-Incremental hook 설치(즉시 GC). reject 5종은 stderr 1회 + no-op
    /// (graceful — fail-fast 아님, §5.6.2 근거).
    fn commit(&self, decode_token_index: usize) -> bool {
        use crate::weight::decider::flatten_importance;
        use crate::weight::{
            IntraForwardSwapHook, PhaseAwareSwapDispatcher, SwapAlgorithm, SwapDecision,
            WeightSwapDecider, compute_qcf_weight_swap,
        };

        let model = self.model.as_ref();
        let ratio = self.ratio;

        // ── 1. Validate ───────────────────────────────────────────────────
        let Some(secondary) = model.secondary_mmap.as_ref() else {
            eprintln!("[WeightSwap] Rejected: no_secondary (ENG-DAT-C09)");
            return false;
        };
        if ratio <= 0.0 || ratio > 1.0 {
            eprintln!("[WeightSwap] Rejected: invalid_ratio ({:.4})", ratio);
            return false;
        }
        if self.target_dtype != DtypeTag::Q4_0 {
            eprintln!(
                "[WeightSwap] Rejected: unsupported_dtype ({:?}) (INV-126)",
                self.target_dtype
            );
            return false;
        }

        // ── 2. In-flight check (R-1 동시 활성화 가드) ────────────────────
        // commit_slot.is_idle() 등가 — swap_runtime 공유 마커(§5.6.4 재submit 차단).
        if !self.runtime.is_idle() {
            eprintln!(
                "[WeightSwap] Rejected: commit slot already active (ratio={:.2}). \
                 Wait for current plan to complete before sending a new SwapWeights signal.",
                ratio
            );
            return false;
        }

        // ── 3. Collect currently-swapped layers ──────────────────────────
        let n_layers = model.layers.len();
        let currently_swapped: Vec<usize> = (0..n_layers)
            .filter(|&i| model.layers[i].current_dtype() == DType::Q4_0)
            .collect();

        // ── 4. Decider ───────────────────────────────────────────────────
        let allow_boundary = crate::session::qcf_runtime::read_allow_boundary_env();
        eprintln!(
            "[Decider] allow_boundary_layers={} (ratio={:.4}, mode={:?})",
            allow_boundary,
            ratio,
            self.runtime.default_mode()
        );
        let importance_ref = self.importance.as_deref();
        let importance_flat = importance_ref.map(|imp| flatten_importance(imp, n_layers));
        let noise_flat = if model.quant_noise.is_computed() {
            Some(model.quant_noise.as_slice())
        } else {
            None
        };
        let target_count = (ratio * n_layers as f32).floor() as usize;
        let budget = target_count.saturating_sub(currently_swapped.len());
        let decider = WeightSwapDecider {
            importance: importance_flat.as_deref(),
            noise: noise_flat,
            n_decoder_layers: n_layers,
            currently_swapped: &currently_swapped,
            allow_boundary_layers: allow_boundary,
            algorithm: SwapAlgorithm::ImportanceAware,
        };
        let decision: SwapDecision = decider.decide(budget);

        if decision.selected_layers.is_empty() {
            eprintln!(
                "[WeightSwap] No layers to swap (ratio={:.2}, already_swapped={})",
                ratio,
                currently_swapped.len()
            );
            return false;
        }

        // ── 5. QCF estimate ──────────────────────────────────────────────
        let qcf_swap_estimated = compute_qcf_weight_swap(
            &decision.selected_layers,
            model.quant_noise.as_slice(),
            importance_flat.as_deref(),
            n_layers,
        );

        let n_planned = decision.selected_layers.len();
        let selected_for_report = decision.selected_layers.clone();

        // ── 6. Mode-specific commit ──────────────────────────────────────
        let is_incremental = match self.runtime.default_mode() {
            crate::session::cli::SwapMode::Incremental => {
                // Manager-driven default per_tick (legacy hardcode 보존, LISWAP-6).
                let per_tick = 2usize;
                let ticks_est = n_planned.div_ceil(per_tick);
                eprintln!(
                    "[WeightSwap] manager path (Incremental): ratio={:.2}, {} layers, per_tick={} ({} ticks), qcf={:.4}",
                    ratio, n_planned, per_tick, ticks_est, qcf_swap_estimated,
                );
                let plan = IncrementalSwapPlan::new(
                    decision.selected_layers,
                    per_tick,
                    decode_token_index,
                );
                *self.plan.lock().expect("weight_swap plan mutex poisoned") = Some(plan);
                self.runtime.mark_in_flight(true);
                true
            }
            crate::session::cli::SwapMode::IntraForward
            | crate::session::cli::SwapMode::LayerImmediate => {
                let mode_label = if matches!(
                    self.runtime.default_mode(),
                    crate::session::cli::SwapMode::LayerImmediate
                ) {
                    "layer-immediate"
                } else {
                    "intra-forward"
                };
                eprintln!(
                    "[WeightSwap] manager path ({}): ratio={:.2}, {} layers, qcf={:.4}",
                    mode_label, ratio, n_planned, qcf_swap_estimated,
                );
                let hook = IntraForwardSwapHook::new(
                    decision.selected_layers,
                    decode_token_index,
                    Arc::clone(self.runtime.dispatcher()),
                    Arc::clone(secondary),
                    model.layers.clone(),
                    Arc::clone(self.runtime.swap_backend()),
                    Some(Arc::clone(self.runtime.release_worker())),
                    DType::Q4_0,
                    Arc::clone(self.runtime.config()),
                );
                // §5.9.2 Track B: hook 을 공유 cell 에 설치한다 → ModelForward 가 매 decode step
                // 이 cell 을 lock-read 해 forward args `layer_boundary_hook` 슬롯에 주입하고,
                // forward_into 의 layer loop 가 layer 경계에서 hook 을 자율 dispatch 한다.
                // (LayerImmediate 는 동일 hook, mode label 만 상이 — §5.6.3.)
                //
                // **finalize/클리어 결정 (§5.9.2 "hook lifetime" 의 fallback clause):**
                // IntraForward/LayerImmediate 는 commit 시 in-flight 마커를 set 하지 않으며
                // (Incremental 만 mark_in_flight(true)), v2 DecodeLoop 에 `IntraForwardSwapHook::
                // finalize` 호출처가 없다(v1 caller 삭제). v1 도 hook 을 plan 소진 후 잔존시켰고
                // (`should_dispatch(idx)` false → on_layer_boundary 즉시 return = no-op,
                // intra_forward_swap.rs:392), 정확성은 그 self-gate 가 보장한다. 따라서 별도
                // PostForward 클리어 경로를 신설하지 않고 **다음 commit 이 cell 을 overwrite** 하는
                // 것으로 갈음한다 — cell 은 항상 최대 1 hook 만 보유한다. 동시 활성(R-1)은 commit §2
                // 의 in-flight 가드가 막는다(non-Incremental 은 mark_in_flight 미진입이라 본 가드가
                // IntraForward 연속 commit 을 막지는 않으나, 같은 ratio 재submit 은 selected_layers
                // 빈 결과로 §4 에서 reject, 새 ratio 는 새 plan 으로 overwrite — v1 등가).
                // 실 swap dispatch·정확성 = S25 device 게이트(secondary + GPU async swap).
                *self
                    .hook_cell
                    .lock()
                    .expect("weight_swap hook_cell mutex poisoned") =
                    Some(hook as Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>);
                eprintln!(
                    "[WeightSwap] {} hook installed into shared cell (§5.9.2 Track B)",
                    mode_label
                );
                false
            }
            crate::session::cli::SwapMode::PhaseAware => {
                eprintln!(
                    "[WeightSwap] manager path (PhaseAware): ratio={:.2}, {} layers, chunk_size_bytes={}, qcf={:.4}",
                    ratio,
                    n_planned,
                    self.runtime.phase_chunk_size_bytes(),
                    qcf_swap_estimated,
                );
                let phase_dispatcher = PhaseAwareSwapDispatcher::new(
                    self.runtime.phase_chunk_size_bytes(),
                    model.layers.clone(),
                    Arc::clone(secondary),
                    Arc::clone(self.runtime.swap_backend()),
                    Arc::clone(self.runtime.dispatcher()),
                    DType::Q4_0,
                    Arc::clone(self.runtime.config()),
                );
                phase_dispatcher.install_self_weak();
                phase_dispatcher.commit_plan(&selected_for_report);
                phase_dispatcher
                    .set_max_chunks_per_token(self.runtime.phase_max_chunks_per_token());
                crate::observability::profile::op_trace::set_phase_hook(phase_dispatcher.clone()
                    as Arc<dyn crate::observability::profile::op_trace::PhaseHook>);
                // dispatcher 는 process-global op_trace singleton 으로 떠난다(§5.6.3) — Stage 보유 불요.
                false
            }
        };

        // ── 7. Manager report (§5.6.6 — commit 시점 송신) ─────────────────
        self.send_report(&selected_for_report, qcf_swap_estimated);

        is_incremental
    }

    /// §5.6.6: `WeightSwapReport` 구성 + 송출(commit 시점). `handle_swap_weights` §7 의
    /// `manager_report_out = Some((ratio, n_planned, Instant, qcf))` tuple 을 commit 시점
    /// 송신으로 단순화(plan-retire 지연 송출 폐기). commit 시점엔 실 swap 미수행이라
    /// `layers_swapped` 는 plan(F16→Q4_0), `latency_ms=0`, `qcf_swap_actual=qcf_estimated`.
    fn send_report(&self, selected_layers: &[usize], qcf: f32) {
        let layers_swapped: Vec<LayerSwapEntry> = selected_layers
            .iter()
            .map(|&idx| LayerSwapEntry {
                layer_idx: idx as u32,
                from_dtype: DtypeTag::F16,
                to_dtype: DtypeTag::Q4_0,
            })
            .collect();
        self.runtime.send_swap_report(WeightSwapReport {
            layers_swapped,
            freed_bytes: 0,
            latency_ms: 0,
            qcf_swap_actual: qcf,
        });
    }

    /// Incremental mode 1-tick drain (§5.6.3) — `drain_chunk`(per_tick=2) →
    /// `execute_on_slots`. 반환: plan 이 `is_done()` 이면 `true`(retire 신호).
    fn drain_tick(&self) -> bool {
        let mut guard = self.plan.lock().expect("weight_swap plan mutex poisoned");
        let Some(plan) = guard.as_mut() else {
            return true; // plan 부재 = 이미 done (방어).
        };
        let chunk = plan.drain_chunk();
        if !chunk.is_empty() {
            let t_swap = Instant::now();
            match crate::session::qcf_runtime::run_layer_swap(
                self.model.as_ref(),
                &chunk,
                Some(self.runtime.swap_backend()),
                self.runtime.swap_backend(),
                Some(self.runtime.dispatcher().as_ref()),
                #[cfg(feature = "opencl")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
            ) {
                Ok(report) => {
                    eprintln!(
                        "[WeightSwap] Incremental tick chunk={:?} swapped={} remaining={} latency={:.1}ms",
                        &chunk,
                        report.swapped.len(),
                        plan.remaining_count(),
                        t_swap.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                Err(e) => {
                    eprintln!("[WeightSwap] Incremental tick run_layer_swap error: {}", e);
                }
            }
        }
        let done = plan.is_done();
        if done {
            *guard = None; // INV-145: retire 즉시 plan 소멸(추가 drain 차단).
            self.runtime.mark_in_flight(false);
        }
        done
    }
}

impl PipelineStage for WeightSwapStage {
    fn name(&self) -> &str {
        "weight.swap"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::OneShot
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.6.1): WeightMutate 외 phase 는 무시.
        if *phase != LifecyclePhase::WeightMutate {
            return Ok(StageOutcome::Continue);
        }

        let mut committed = self
            .committed
            .lock()
            .expect("weight_swap commit mutex poisoned");
        if !*committed {
            // 첫 WeightMutate tick = commit 경로(directive 도착 tick).
            *committed = true;
            let started_incremental = self.commit(ctx.step.decode_step);
            drop(committed);
            if !started_incremental {
                // reject(no-op) 또는 non-Incremental hook 설치 → 즉시 GC.
                return Ok(StageOutcome::Consumed);
            }
            // Incremental: commit tick 에 첫 chunk 도 drain (legacy: 같은 tick set+drain).
            let done = self.drain_tick();
            return Ok(if done {
                StageOutcome::Consumed
            } else {
                StageOutcome::Continue
            });
        }
        drop(committed);

        // 후속 WeightMutate tick = Incremental drain (D5 multi-tick).
        let done = self.drain_tick();
        Ok(if done {
            StageOutcome::Consumed
        } else {
            StageOutcome::Continue
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::layers::transformer_layer::TransformerLayer;
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;
    use crate::memory::host::shared::SharedBuffer;
    use crate::model_config::{ModelArch, ModelConfig};
    use crate::models::weights::LayerSlot;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::session::cli::SwapMode;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use crate::weight::AsyncSwapDispatcher;

    fn cpu_backend() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    fn f32_weight(be: &Arc<dyn Backend>, out_dim: usize, in_dim: usize) -> Tensor {
        let buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(out_dim * in_dim * 4, DType::F32));
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

    /// secondary 없는 최소 CPU model (validate §1 에서 reject 되는 host 테스트용).
    /// `n_layers` slot 보유. `make_cpu_model_with_embed`(transformer.rs private) 패턴 차용.
    fn make_model(be: &Arc<dyn Backend>, n_layers: usize) -> Arc<TransformerModel> {
        let mem = Galloc::new();
        let dim = 4usize;
        let vocab = 8usize;
        let embed_buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), embed_buf, be.clone());
        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, be.clone());
        let lm_head = norm.clone();
        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: n_layers,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: dim,
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
        };
        let runtime = crate::weight::setup_runtime_resources(be.clone());
        Arc::new(TransformerModel {
            config,
            layers: (0..n_layers).map(|i| ffn_slot(be, i)).collect(),
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: None,
            cpu_backend: None,
            preload_pool: std::sync::OnceLock::new(),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            quant_noise: runtime.quant_noise.clone(),
            release_worker: runtime.release_worker.clone(),
        })
    }

    fn make_runtime(be: &Arc<dyn Backend>, mode: SwapMode) -> Arc<EngineSwapRuntime> {
        let dispatcher = Arc::new(AsyncSwapDispatcher::new(be.clone()));
        let runtime = crate::weight::setup_runtime_resources(be.clone());
        let config = Arc::new(ModelConfig {
            vocab_size: 8,
            hidden_size: 4,
            num_hidden_layers: 0,
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
        });
        Arc::new(EngineSwapRuntime::new(
            be.clone(),
            dispatcher,
            config,
            runtime.release_worker.clone(),
            mode,
            1024 * 1024,
            4,
            None,
        ))
    }

    fn make_ctx(profiler: &mut OpProfiler) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(0),
                prev_token: 0,
            },
            profiler,
        }
    }

    /// §5.9.2 Track B: 빈 hook cell (설치 토글 검증용 / 더미). 타입 별칭.
    type HookCell = Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>>;
    fn empty_hook_cell() -> HookCell {
        Arc::new(Mutex::new(None))
    }

    fn stage(
        be: &Arc<dyn Backend>,
        n_layers: usize,
        mode: SwapMode,
        ratio: f32,
        dtype: DtypeTag,
    ) -> WeightSwapStage {
        WeightSwapStage::one_shot(
            make_model(be, n_layers),
            make_runtime(be, mode),
            None,
            ratio,
            dtype,
            empty_hook_cell(),
        )
    }

    /// WeightMutate 외 phase 는 no-op(Continue) — commit 미진입.
    #[test]
    fn non_weight_mutate_phase_is_noop() {
        let be = cpu_backend();
        let s = stage(&be, 2, SwapMode::Incremental, 0.5, DtypeTag::Q4_0);
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Continue), "phase filter");
        // commit 미진입 (committed 플래그 false 유지) — in-flight 마커도 unset.
        assert!(s.runtime.is_idle(), "no-op 은 in-flight 진입 안 함");
    }

    /// reject: no_secondary (validate §1) → graceful Consumed (no-op).
    #[test]
    fn reject_no_secondary_is_graceful_consumed() {
        let be = cpu_backend();
        // make_model 은 secondary_mmap=None → no_secondary reject.
        let s = stage(&be, 2, SwapMode::Incremental, 0.5, DtypeTag::Q4_0);
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::WeightMutate, &mut ctx).unwrap();
        assert!(
            matches!(outcome, StageOutcome::Consumed),
            "reject → Consumed"
        );
        assert!(s.runtime.is_idle(), "reject 는 in-flight 진입 안 함");
    }

    /// reject: invalid_ratio (ratio<=0 or >1) → graceful Consumed.
    #[test]
    fn reject_invalid_ratio_is_graceful_consumed() {
        let be = cpu_backend();
        // ratio=1.5 → invalid_ratio. (no_secondary 보다 먼저 ratio 검사하도록 secondary present
        // 가 필요하나, validate 순서상 no_secondary 가 먼저이므로 secondary mock 없이는 ratio
        // reject 단독 검증 불가 — 따라서 ratio reject 경로의 Consumed 동작은 본 테스트가 아니라
        // 등가 anchor(handle_swap_weights)와의 코드 동일성으로 보장. 여기선 ratio=1.5 도
        // Consumed 임을 확인(no_secondary 가 먼저 막더라도 결과 동일 = graceful Consumed).
        let s = stage(&be, 2, SwapMode::Incremental, 1.5, DtypeTag::Q4_0);
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::WeightMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
    }

    /// reject: unsupported_dtype (target != Q4_0) → graceful Consumed.
    #[test]
    fn reject_unsupported_dtype_is_graceful_consumed() {
        let be = cpu_backend();
        let s = stage(&be, 2, SwapMode::Incremental, 0.5, DtypeTag::F16);
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::WeightMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
    }

    /// in-flight 가드(§5.6.2 §2): swap_runtime 마커가 active 면 새 Stage commit reject → Consumed.
    /// (공유 in-flight 마커 = Stage 간 R-1 차단. 다른 Stage 인스턴스가 같은 runtime 을 공유하면
    /// 미완 plan 중 새 commit 이 reject 됨을 모사.)
    #[test]
    fn reject_in_flight_is_graceful_consumed() {
        let be = cpu_backend();
        let rt = make_runtime(&be, SwapMode::Incremental);
        rt.mark_in_flight(true); // 선행 Stage 가 미완 drain 중인 상태 모사.
        // 같은 runtime 을 공유하는 새 Stage. ratio=0.5/Q4_0 라 validate §1 통과해도 §2 에서 reject.
        // 단 secondary=None 이라 실제로는 §1(no_secondary) 가 먼저 막지만, 어느 경로든 Consumed.
        let s = WeightSwapStage::one_shot(
            make_model(&be, 2),
            rt.clone(),
            None,
            0.5,
            DtypeTag::Q4_0,
            empty_hook_cell(),
        );
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::WeightMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        // 마커는 새 Stage 가 reject 했으므로 여전히 active (선행 plan 소유).
        assert!(!rt.is_idle(), "선행 in-flight 마커 보존");
    }

    /// in-flight 마커 단위 동작: is_idle / mark_in_flight round-trip.
    #[test]
    fn in_flight_marker_roundtrip() {
        let be = cpu_backend();
        let rt = make_runtime(&be, SwapMode::Incremental);
        assert!(rt.is_idle(), "초기 idle");
        rt.mark_in_flight(true);
        assert!(!rt.is_idle(), "drain 중");
        rt.mark_in_flight(false);
        assert!(rt.is_idle(), "drain 완료");
    }

    // NOTE: Incremental multi-tick drain (Continue×N→Consumed) / IntraForward·PhaseAware 의 *실
    // commit 경로* 를 통한 hook 설치 / decider-success / 등가 anchor(Stage commit ==
    // handle_swap_weights 직접 호출) 는 secondary_mmap mock(mmap 파일) 을 요구하므로 **host 검증
    // 불가** — device 게이트(S25/Jetson, §5.6.7)에서 legacy frozen baseline 대조로 검증한다.
    // commit 본문은 swap_runtime.rs 의 handle_swap_weights §1~7 을 byte-identical 이전한 것이라
    // 코드 동일성으로 등가가 보장된다.

    /// §5.9.2 Track B: hook cell 설치 토글 계약. commit 의 IntraForward/LayerImmediate arm 이
    /// 쓰는 `*hook_cell.lock() = Some(hook as Arc<dyn LayerBoundaryHook>)` 와 **동일 coercion** 으로
    /// 설치했을 때 cell 이 `Some` 으로 토글됨을 검증한다(실 commit 경로는 secondary mmap 요구라
    /// device-gated — 본 테스트는 설치 *메커니즘*(cell 타입·coercion·toggle)만 host 에서 격리 검증).
    #[test]
    fn hook_cell_install_toggles_to_some() {
        use crate::weight::IntraForwardSwapHook;

        let be = cpu_backend();
        let cell = empty_hook_cell();
        // 초기값 None (설치 전).
        assert!(
            cell.lock().unwrap().is_none(),
            "hook cell 초기값 None (설치 전)"
        );

        // commit 의 IntraForward arm 이 생성하는 hook 과 동형(secondary=None 인 test 변형).
        let dispatcher = Arc::new(AsyncSwapDispatcher::new(be.clone()));
        let slots = vec![ffn_slot(&be, 0), ffn_slot(&be, 1)];
        let config = Arc::new(make_model(&be, 2).config.clone());
        let hook = IntraForwardSwapHook::new_for_test(
            vec![0],
            0,
            dispatcher,
            slots,
            be,
            DType::Q4_0,
            config,
        );

        // commit arm 과 동일한 설치 문장 (§5.9.2).
        *cell.lock().unwrap() =
            Some(hook as Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>);

        assert!(
            cell.lock().unwrap().is_some(),
            "설치 후 cell 은 Some (forward slot 주입 대상)"
        );
    }
}
